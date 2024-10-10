import Mathlib

namespace rogers_coin_piles_l3816_381648

theorem rogers_coin_piles (num_quarter_piles num_dime_piles coins_per_pile total_coins : ℕ) :
  num_quarter_piles = num_dime_piles →
  coins_per_pile = 7 →
  total_coins = 42 →
  num_quarter_piles * coins_per_pile + num_dime_piles * coins_per_pile = total_coins →
  num_quarter_piles = 3 := by
  sorry

end rogers_coin_piles_l3816_381648


namespace sixty_second_pair_l3816_381632

/-- Definition of our sequence of pairs -/
def pair_sequence : ℕ → ℕ × ℕ
| 0 => (1, 1)
| n + 1 =>
  let (a, b) := pair_sequence n
  if a = 1 then (b + 1, 1)
  else (a - 1, b + 1)

/-- The 62nd pair in the sequence is (7,5) -/
theorem sixty_second_pair :
  pair_sequence 61 = (7, 5) :=
sorry

end sixty_second_pair_l3816_381632


namespace lydia_apple_eating_age_l3816_381650

/-- The age at which Lydia will eat an apple from her tree for the first time -/
def apple_eating_age (planting_age : ℕ) (years_to_bear_fruit : ℕ) : ℕ :=
  planting_age + years_to_bear_fruit

/-- Theorem stating Lydia's age when she first eats an apple from her tree -/
theorem lydia_apple_eating_age :
  apple_eating_age 4 7 = 11 := by
  sorry

end lydia_apple_eating_age_l3816_381650


namespace rectangle_horizontal_length_l3816_381662

/-- Given a square with side length 80 cm and a rectangle with vertical length 100 cm,
    if their perimeters are equal, then the horizontal length of the rectangle is 60 cm. -/
theorem rectangle_horizontal_length (square_side : ℝ) (rect_vertical : ℝ) (rect_horizontal : ℝ) :
  square_side = 80 ∧ rect_vertical = 100 ∧ 4 * square_side = 2 * (rect_vertical + rect_horizontal) →
  rect_horizontal = 60 := by
  sorry

end rectangle_horizontal_length_l3816_381662


namespace agent_007_encryption_possible_l3816_381691

theorem agent_007_encryption_possible : ∃ (m n : ℕ), (0.07 : ℝ) = 1 / m + 1 / n := by
  sorry

end agent_007_encryption_possible_l3816_381691


namespace greatest_power_of_three_l3816_381637

def v : ℕ := (List.range 30).foldl (· * ·) 1

theorem greatest_power_of_three (a : ℕ) : 
  (∀ k : ℕ, k ≤ 30 → k > 0 → v % 3^k = 0) → 
  (∀ m : ℕ, m > a → ¬(v % 3^m = 0)) → 
  a = 14 := by sorry

end greatest_power_of_three_l3816_381637


namespace channel_system_properties_l3816_381629

/-- Represents a water channel system with nodes A to H -/
structure ChannelSystem where
  /-- Flow rate in channel BC -/
  q₀ : ℝ
  /-- Flow rate in channel AB -/
  q_AB : ℝ
  /-- Flow rate in channel AH -/
  q_AH : ℝ

/-- The flow rates in the channel system satisfy the given conditions -/
def is_valid_system (sys : ChannelSystem) : Prop :=
  sys.q_AB = (1/2) * sys.q₀ ∧
  sys.q_AH = (3/4) * sys.q₀

/-- The total flow rate entering at node A -/
def total_flow_A (sys : ChannelSystem) : ℝ :=
  sys.q_AB + sys.q_AH

/-- Theorem stating the properties of the channel system -/
theorem channel_system_properties (sys : ChannelSystem) 
  (h : is_valid_system sys) : 
  sys.q_AB = (1/2) * sys.q₀ ∧ 
  sys.q_AH = (3/4) * sys.q₀ ∧ 
  total_flow_A sys = (7/4) * sys.q₀ := by
  sorry

end channel_system_properties_l3816_381629


namespace midpoint_of_intersection_l3816_381614

-- Define the ellipse C
def ellipse_C (x y : ℝ) : Prop := x^2 / 9 + y^2 = 1

-- Define the line that intersects the ellipse
def intersecting_line (x y : ℝ) : Prop := y = x + 2

-- Define the intersection points A and B
def intersection_points (A B : ℝ × ℝ) : Prop :=
  ellipse_C A.1 A.2 ∧ ellipse_C B.1 B.2 ∧
  intersecting_line A.1 A.2 ∧ intersecting_line B.1 B.2 ∧
  A ≠ B

-- Theorem statement
theorem midpoint_of_intersection :
  ∀ A B : ℝ × ℝ,
  intersection_points A B →
  (A.1 + B.1) / 2 = -9/5 ∧ (A.2 + B.2) / 2 = 1/5 :=
sorry

end midpoint_of_intersection_l3816_381614


namespace problem_statement_l3816_381697

theorem problem_statement (x y : ℝ) (h1 : x + y > 0) (h2 : x * y ≠ 0) :
  (x^3 + y^3 ≥ x^2*y + y^2*x) ∧
  (Set.Icc (-6 : ℝ) 2 = {m : ℝ | x / y^2 + y / x^2 ≥ m / 2 * (1 / x + 1 / y)}) := by
  sorry

end problem_statement_l3816_381697


namespace opposite_pairs_l3816_381654

theorem opposite_pairs (a b : ℝ) : 
  (∀ x, (a + b) + (-a - b) = x ↔ x = 0) ∧ 
  (∀ x, (-a + b) + (a - b) = x ↔ x = 0) ∧ 
  ¬(∀ x, (a - b) + (-a - b) = x ↔ x = 0) ∧ 
  ¬(∀ x, (a + 1) + (1 - a) = x ↔ x = 0) :=
by sorry

end opposite_pairs_l3816_381654


namespace equation_solutions_l3816_381635

theorem equation_solutions :
  let f : ℝ → ℝ := fun x ↦ x * (x - 3)^2 * (5 - x)
  {x : ℝ | f x = 0} = {0, 3, 5} := by
sorry

end equation_solutions_l3816_381635


namespace closest_integer_to_cube_root_250_l3816_381695

theorem closest_integer_to_cube_root_250 : 
  ∀ n : ℤ, |n - (250 : ℝ)^(1/3)| ≥ |6 - (250 : ℝ)^(1/3)| :=
by sorry

end closest_integer_to_cube_root_250_l3816_381695


namespace extremum_at_one_lower_bound_ln_two_l3816_381653

noncomputable section

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := Real.log (a * x + 1) + (1 - x) / (1 + x)

-- Theorem for the first part of the problem
theorem extremum_at_one (a : ℝ) (h : a > 0) :
  (∃ ε > 0, ∀ x ∈ Set.Icc (1 - ε) (1 + ε), f a x ≥ f a 1) ↔ a = 1 :=
sorry

-- Theorem for the second part of the problem
theorem lower_bound_ln_two (a : ℝ) (h : a > 0) :
  (∀ x ≥ 0, f a x ≥ Real.log 2) ↔ a ≥ 1 :=
sorry

end extremum_at_one_lower_bound_ln_two_l3816_381653


namespace dandelion_seed_production_l3816_381699

-- Define the number of seeds produced by a single dandelion plant
def seeds_per_plant : ℕ := 50

-- Define the germination rate (half of the seeds)
def germination_rate : ℚ := 1 / 2

-- Theorem statement
theorem dandelion_seed_production :
  let initial_seeds := seeds_per_plant
  let germinated_plants := (initial_seeds : ℚ) * germination_rate
  let total_seeds := (germinated_plants * seeds_per_plant : ℚ)
  total_seeds = 1250 := by
  sorry

end dandelion_seed_production_l3816_381699


namespace tan_alpha_value_l3816_381688

theorem tan_alpha_value (α β : ℝ) 
  (h1 : Real.tan (α + β) = 3)
  (h2 : Real.tan β = 2) : 
  Real.tan α = 1/7 := by
  sorry

end tan_alpha_value_l3816_381688


namespace largest_binomial_coefficient_sum_binomial_coefficient_sum_equals_six_largest_n_is_six_l3816_381672

theorem largest_binomial_coefficient_sum (n : ℕ) : 
  (Nat.choose 10 4 + Nat.choose 10 5 = Nat.choose 11 n) → n ≤ 6 :=
by sorry

theorem binomial_coefficient_sum_equals_six : 
  Nat.choose 10 4 + Nat.choose 10 5 = Nat.choose 11 6 :=
by sorry

theorem largest_n_is_six : 
  ∃ (n : ℕ), n = 6 ∧ 
    Nat.choose 10 4 + Nat.choose 10 5 = Nat.choose 11 n ∧
    ∀ (m : ℕ), Nat.choose 10 4 + Nat.choose 10 5 = Nat.choose 11 m → m ≤ n :=
by sorry

end largest_binomial_coefficient_sum_binomial_coefficient_sum_equals_six_largest_n_is_six_l3816_381672


namespace cos_sin_sum_equals_half_l3816_381683

theorem cos_sin_sum_equals_half : 
  Real.cos (25 * π / 180) * Real.cos (85 * π / 180) + 
  Real.sin (25 * π / 180) * Real.sin (85 * π / 180) = 1/2 :=
by sorry

end cos_sin_sum_equals_half_l3816_381683


namespace not_perfect_square_l3816_381600

theorem not_perfect_square (n : ℕ) : ¬∃ (m : ℕ), 2 * 13^n + 5 * 7^n + 26 = m^2 := by
  sorry

end not_perfect_square_l3816_381600


namespace inequality_proof_l3816_381676

theorem inequality_proof (a b c : ℝ) (ha : 0 ≤ a ∧ a ≤ 1) (hb : 0 ≤ b ∧ b ≤ 1) (hc : 0 ≤ c ∧ c ≤ 1) :
  (a / (b + c + 1)) + (b / (c + a + 1)) + (c / (a + b + 1)) + (1 - a) * (1 - b) * (1 - c) ≤ 1 := by
sorry

end inequality_proof_l3816_381676


namespace stratified_sampling_sample_size_l3816_381611

theorem stratified_sampling_sample_size 
  (total_employees : ℕ) 
  (young_employees : ℕ) 
  (young_in_sample : ℕ) 
  (h1 : total_employees = 750) 
  (h2 : young_employees = 350) 
  (h3 : young_in_sample = 7) : 
  (young_in_sample * total_employees) / young_employees = 15 := by
  sorry

end stratified_sampling_sample_size_l3816_381611


namespace find_y_l3816_381633

theorem find_y (a b : ℝ) (y : ℝ) (h1 : a > 0) (h2 : b > 0) :
  let s := (3 * a) ^ (2 * b)
  s = 5 * a^b * y^b →
  y = 9 * a / 5 := by
sorry

end find_y_l3816_381633


namespace triangle_property_l3816_381696

open Real

theorem triangle_property (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧
  A + B + C = π ∧
  0 < a ∧ 0 < b ∧ 0 < c ∧
  a * cos B - b * cos A = c / 2 ∧
  B = π / 4 ∧
  b = sqrt 5 →
  tan A = 3 * tan B ∧
  (1 / 2) * a * b * sin C = 3 :=
by sorry

end triangle_property_l3816_381696


namespace expression_value_l3816_381641

theorem expression_value (a b : ℝ) (h : a + 3*b = 4) : 2*a + 6*b - 1 = 7 := by
  sorry

end expression_value_l3816_381641


namespace product_of_decimals_l3816_381625

theorem product_of_decimals : (0.7 : ℝ) * 0.3 = 0.21 := by
  sorry

end product_of_decimals_l3816_381625


namespace students_playing_neither_sport_l3816_381603

theorem students_playing_neither_sport (total : ℕ) (football : ℕ) (tennis : ℕ) (both : ℕ)
  (h_total : total = 40)
  (h_football : football = 26)
  (h_tennis : tennis = 20)
  (h_both : both = 17) :
  total - (football + tennis - both) = 11 := by
  sorry

end students_playing_neither_sport_l3816_381603


namespace intersection_complement_equality_l3816_381669

-- Define the universal set U
def U : Set Nat := {1, 2, 3, 4}

-- Define set A
def A : Set Nat := {1, 3}

-- Define set B
def B : Set Nat := {2, 3}

-- Theorem statement
theorem intersection_complement_equality : B ∩ (U \ A) = {2} := by
  sorry

end intersection_complement_equality_l3816_381669


namespace some_number_value_l3816_381626

theorem some_number_value (a n : ℕ) (h1 : a = 105) (h2 : a^3 = n * 49 * 45 * 25) : n = 21 := by
  sorry

end some_number_value_l3816_381626


namespace inequality_solution_sets_l3816_381659

theorem inequality_solution_sets (a b : ℝ) :
  (∀ x, 2 * a * x^2 - 2 * x + 3 < 0 ↔ 2 < x ∧ x < b) →
  (∀ x, 3 * x^2 + 2 * x + 2 * a < 0 ↔ -1/2 < x ∧ x < -1/6) :=
by sorry

end inequality_solution_sets_l3816_381659


namespace inequality_proof_l3816_381606

theorem inequality_proof (a b : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : a^2 + b^2 = 4) :
  (a * b) / (a + b + 2) ≤ Real.sqrt 2 - 1 ∧
  ((a * b) / (a + b + 2) = Real.sqrt 2 - 1 ↔ a = Real.sqrt 2 ∧ b = Real.sqrt 2) :=
by sorry

#check inequality_proof

end inequality_proof_l3816_381606


namespace area_of_graph_l3816_381618

/-- The equation of the graph -/
def graph_equation (x y : ℝ) : Prop := abs (2 * x) + abs (3 * y) = 6

/-- The set of points satisfying the equation -/
def graph_set : Set (ℝ × ℝ) := {p | graph_equation p.1 p.2}

/-- The area enclosed by the graph -/
noncomputable def enclosed_area : ℝ := sorry

theorem area_of_graph : enclosed_area = 12 := by sorry

end area_of_graph_l3816_381618


namespace cube_sum_reciprocal_l3816_381647

theorem cube_sum_reciprocal (x : ℝ) (h : x ≠ 0) :
  x + 1 / x = 3 → x^3 + 1 / x^3 = 18 := by
  sorry

end cube_sum_reciprocal_l3816_381647


namespace simplify_trig_expression_l3816_381624

theorem simplify_trig_expression (α : Real) (h : π / 2 < α ∧ α < π) :
  Real.sqrt ((1 + Real.sin α) / (1 - Real.sin α)) - Real.sqrt ((1 - Real.sin α) / (1 + Real.sin α)) = -2 * Real.tan α := by
  sorry

end simplify_trig_expression_l3816_381624


namespace unique_solutions_l3816_381622

-- Define the coprime relation
def coprime (a b : ℕ) : Prop := Nat.gcd a b = 1

-- Define the equation
def satisfies_equation (x y : ℕ) : Prop := x^2 - x + 1 = y^3

-- Main theorem
theorem unique_solutions :
  ∀ x y : ℕ, x > 0 ∧ y > 0 →
  coprime x (y-1) →
  satisfies_equation x y →
  (x = 1 ∧ y = 1) ∨ (x = 19 ∧ y = 7) :=
sorry

end unique_solutions_l3816_381622


namespace largest_circle_radius_is_b_l3816_381616

/-- An ellipsoid with semi-axes a > b > c -/
structure Ellipsoid where
  a : ℝ
  b : ℝ
  c : ℝ
  h1 : a > b
  h2 : b > c

/-- The radius of the largest circle on an ellipsoid -/
def largest_circle_radius (e : Ellipsoid) : ℝ := e.b

/-- Theorem: The radius of the largest circle on an ellipsoid with semi-axes a > b > c is b -/
theorem largest_circle_radius_is_b (e : Ellipsoid) :
  largest_circle_radius e = e.b :=
by sorry

end largest_circle_radius_is_b_l3816_381616


namespace price_restoration_l3816_381677

theorem price_restoration (original_price : ℝ) (original_price_positive : original_price > 0) : 
  let reduced_price := original_price * (1 - 0.15)
  let restoration_factor := original_price / reduced_price
  let percentage_increase := (restoration_factor - 1) * 100
  ∃ ε > 0, abs (percentage_increase - 17.65) < ε ∧ ε < 0.01 :=
by sorry

end price_restoration_l3816_381677


namespace perpendicular_tangents_point_l3816_381615

/-- The point on the line y = x from which two perpendicular tangents 
    can be drawn to the parabola y = x^2 -/
theorem perpendicular_tangents_point :
  ∃! P : ℝ × ℝ, 
    (P.1 = P.2) ∧ 
    (∃ m₁ m₂ : ℝ, 
      (m₁ * m₂ = -1) ∧
      (∀ x y : ℝ, y = m₁ * (x - P.1) + P.2 → y = x^2 → x = P.1) ∧
      (∀ x y : ℝ, y = m₂ * (x - P.1) + P.2 → y = x^2 → x = P.1)) ∧
    P = (-1/4, -1/4) :=
by sorry

end perpendicular_tangents_point_l3816_381615


namespace min_points_for_proximity_l3816_381642

-- Define the circle and points
def Circle := {p : ℝ × ℝ | p.1^2 + p.2^2 = 1}

-- Define the distance function between two points on the circle
def circleDistance (p q : Circle) : ℝ := sorry

-- Define the sequence of points
def circlePoints : ℕ → Circle := sorry

-- Theorem statement
theorem min_points_for_proximity :
  ∀ n : ℕ, n < 20 →
  ∃ i j : ℕ, i < j ∧ j < n ∧ circleDistance (circlePoints i) (circlePoints j) ≥ 1/5 :=
sorry

end min_points_for_proximity_l3816_381642


namespace polynomial_sum_l3816_381620

/-- Two distinct polynomials with real coefficients -/
def f (a b x : ℝ) : ℝ := x^2 + a*x + b
def g (c d x : ℝ) : ℝ := x^2 + c*x + d

/-- The theorem statement -/
theorem polynomial_sum (a b c d : ℝ) : 
  (∃ x, f a b x = 0 ∧ x = -c/2) →  -- x-coordinate of vertex of g is root of f
  (∃ x, g c d x = 0 ∧ x = -a/2) →  -- x-coordinate of vertex of f is root of g
  (∀ x, f a b x ≥ -144) →          -- minimum value of f is -144
  (∀ x, g c d x ≥ -144) →          -- minimum value of g is -144
  (∃ x, f a b x = -144) →          -- f achieves its minimum
  (∃ x, g c d x = -144) →          -- g achieves its minimum
  f a b 150 = -200 →               -- f(150) = -200
  g c d 150 = -200 →               -- g(150) = -200
  a + c = -300 :=
by sorry

end polynomial_sum_l3816_381620


namespace total_gum_pieces_l3816_381664

theorem total_gum_pieces (packages : ℕ) (pieces_per_package : ℕ) (h1 : packages = 9) (h2 : pieces_per_package = 15) :
  packages * pieces_per_package = 135 := by
  sorry

end total_gum_pieces_l3816_381664


namespace modulus_of_complex_number_l3816_381665

theorem modulus_of_complex_number (z : ℂ) : z = 3 - 2*I → Complex.abs z = Real.sqrt 13 := by
  sorry

end modulus_of_complex_number_l3816_381665


namespace parity_equality_of_extrema_l3816_381644

/-- A set of elements related to positioning in a function or polynomial -/
def A_P : Set ℤ := sorry

/-- The maximum element of A_P -/
def max_element (A : Set ℤ) : ℤ := sorry

/-- The minimum element of A_P -/
def min_element (A : Set ℤ) : ℤ := sorry

/-- Parity of an integer -/
def parity (n : ℤ) : Bool := n % 2 = 0

/-- Theorem: The parity of the smallest and largest elements of A_P is the same -/
theorem parity_equality_of_extrema :
  parity (min_element A_P) = parity (max_element A_P) := by
  sorry

end parity_equality_of_extrema_l3816_381644


namespace claire_earnings_l3816_381666

def total_flowers : ℕ := 400
def tulips : ℕ := 120
def white_roses : ℕ := 80
def red_rose_price : ℚ := 3/4

def roses : ℕ := total_flowers - tulips
def red_roses : ℕ := roses - white_roses
def red_roses_to_sell : ℕ := red_roses / 2

theorem claire_earnings : 
  (red_roses_to_sell : ℚ) * red_rose_price = 75 := by sorry

end claire_earnings_l3816_381666


namespace log_exponent_sum_l3816_381628

theorem log_exponent_sum (a : ℝ) (h : a = Real.log 5 / Real.log 4) :
  2^a + 2^(-a) = 6 * Real.sqrt 5 / 5 := by
  sorry

end log_exponent_sum_l3816_381628


namespace unique_number_six_times_sum_of_digits_l3816_381685

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Predicate for numbers that are 6 times the sum of their digits -/
def is_six_times_sum_of_digits (n : ℕ) : Prop :=
  n = 6 * sum_of_digits n

theorem unique_number_six_times_sum_of_digits :
  ∃! n : ℕ, n < 1000 ∧ is_six_times_sum_of_digits n :=
sorry

end unique_number_six_times_sum_of_digits_l3816_381685


namespace min_period_cosine_l3816_381657

/-- The minimum positive period of the cosine function Y = 3cos(2/5x - π/6) is 5π. -/
theorem min_period_cosine (x : ℝ) : 
  let Y : ℝ → ℝ := λ x => 3 * Real.cos ((2/5) * x - π/6)
  ∃ (T : ℝ), T > 0 ∧ (∀ t, Y (t + T) = Y t) ∧ (∀ S, S > 0 ∧ (∀ t, Y (t + S) = Y t) → T ≤ S) ∧ T = 5 * π :=
by sorry

end min_period_cosine_l3816_381657


namespace sugar_consumption_reduction_l3816_381698

theorem sugar_consumption_reduction (initial_price new_price : ℝ) 
  (initial_price_positive : initial_price > 0)
  (new_price_positive : new_price > 0)
  (price_increase : new_price > initial_price) :
  let reduction_percentage := (1 - initial_price / new_price) * 100
  reduction_percentage = 60 :=
by sorry

end sugar_consumption_reduction_l3816_381698


namespace probability_at_least_one_die_shows_one_or_ten_l3816_381670

/-- The number of sides on each die -/
def num_sides : ℕ := 10

/-- The number of outcomes where a die doesn't show 1 or 10 -/
def favorable_outcomes_per_die : ℕ := num_sides - 2

/-- The total number of outcomes when rolling two dice -/
def total_outcomes : ℕ := num_sides * num_sides

/-- The number of outcomes where neither die shows 1 or 10 -/
def unfavorable_outcomes : ℕ := favorable_outcomes_per_die * favorable_outcomes_per_die

/-- The number of favorable outcomes (at least one die shows 1 or 10) -/
def favorable_outcomes : ℕ := total_outcomes - unfavorable_outcomes

/-- The probability of at least one die showing 1 or 10 -/
theorem probability_at_least_one_die_shows_one_or_ten :
  (favorable_outcomes : ℚ) / total_outcomes = 9 / 25 := by
  sorry

end probability_at_least_one_die_shows_one_or_ten_l3816_381670


namespace derivative_roots_in_triangle_l3816_381634

/-- A polynomial of degree three with complex roots -/
def cubic_polynomial (a b c : ℂ) (x : ℂ) : ℂ :=
  (x - a) * (x - b) * (x - c)

/-- The derivative of the cubic polynomial -/
def cubic_derivative (a b c : ℂ) (x : ℂ) : ℂ :=
  (x - a) * (x - b) + (x - a) * (x - c) + (x - b) * (x - c)

/-- The triangle formed by the roots of the cubic polynomial -/
def root_triangle (a b c : ℂ) : Set ℂ :=
  {z : ℂ | ∃ (t₁ t₂ t₃ : ℝ), t₁ + t₂ + t₃ = 1 ∧ t₁ ≥ 0 ∧ t₂ ≥ 0 ∧ t₃ ≥ 0 ∧ z = t₁ * a + t₂ * b + t₃ * c}

/-- Theorem stating that the roots of the derivative lie inside the triangle formed by the roots of the original polynomial -/
theorem derivative_roots_in_triangle (a b c : ℂ) :
  ∀ z : ℂ, cubic_derivative a b c z = 0 → z ∈ root_triangle a b c :=
sorry

end derivative_roots_in_triangle_l3816_381634


namespace job_selection_ways_l3816_381671

theorem job_selection_ways (method1_people : ℕ) (method2_people : ℕ) 
  (h1 : method1_people = 3) (h2 : method2_people = 5) : 
  method1_people + method2_people = 8 := by
  sorry

end job_selection_ways_l3816_381671


namespace scientific_notation_of_120_l3816_381631

/-- Scientific notation representation of a real number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  h1 : 1 ≤ coefficient
  h2 : coefficient < 10

/-- Convert a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem scientific_notation_of_120 :
  toScientificNotation 120 = ScientificNotation.mk 1.2 2 (by norm_num) (by norm_num) :=
sorry

end scientific_notation_of_120_l3816_381631


namespace brand_z_percentage_approx_l3816_381658

/-- Represents the capacity of the fuel tank -/
def tank_capacity : ℚ := 12

/-- Represents the amount of brand Z gasoline after the final filling -/
def final_brand_z : ℚ := 10

/-- Represents the total amount of gasoline after the final filling -/
def final_total : ℚ := 12

/-- Calculates the percentage of a part relative to the whole -/
def percentage (part whole : ℚ) : ℚ := (part / whole) * 100

/-- Theorem stating that the percentage of brand Z gasoline is approximately 83.33% -/
theorem brand_z_percentage_approx : 
  abs (percentage final_brand_z final_total - 83.33) < 0.01 := by
  sorry

#eval percentage final_brand_z final_total

end brand_z_percentage_approx_l3816_381658


namespace solution_in_interval_l3816_381613

open Real

theorem solution_in_interval (x₀ : ℝ) (k : ℤ) : 
  (8 - x₀ = log x₀) → 
  (x₀ ∈ Set.Ioo (k : ℝ) (k + 1)) → 
  k = 7 := by
sorry

end solution_in_interval_l3816_381613


namespace smallest_five_digit_congruent_to_3_mod_17_l3816_381674

theorem smallest_five_digit_congruent_to_3_mod_17 :
  ∀ n : ℕ, 10000 ≤ n ∧ n < 100000 ∧ n % 17 = 3 → n ≥ 10018 :=
by sorry

end smallest_five_digit_congruent_to_3_mod_17_l3816_381674


namespace first_protest_duration_l3816_381681

/-- 
Given a person who attends two protests where the second protest duration is 25% longer 
than the first, and the total time spent protesting is 9 days, prove that the duration 
of the first protest is 4 days.
-/
theorem first_protest_duration (first_duration : ℝ) 
  (h1 : first_duration > 0)
  (h2 : first_duration + (1.25 * first_duration) = 9) : 
  first_duration = 4 := by
sorry

end first_protest_duration_l3816_381681


namespace y_coordinate_abs_value_l3816_381619

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The distance from a point to the y-axis -/
def distToYAxis (p : Point) : ℝ := |p.x|

/-- The distance from a point to the x-axis -/
def distToXAxis (p : Point) : ℝ := |p.y|

theorem y_coordinate_abs_value (p : Point) 
  (h1 : distToXAxis p = (1/2) * distToYAxis p) 
  (h2 : distToYAxis p = 12) : 
  |p.y| = 6 := by sorry

end y_coordinate_abs_value_l3816_381619


namespace limit_of_sequence_a_l3816_381607

def a (n : ℕ) : ℚ := (1 - 2 * n^2) / (2 + 4 * n^2)

theorem limit_of_sequence_a :
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |a n - (-1/2)| < ε :=
sorry

end limit_of_sequence_a_l3816_381607


namespace three_pump_fill_time_l3816_381682

/-- Represents the time taken (in hours) for three pumps to fill a tank when working together. -/
def combined_fill_time (rate1 rate2 rate3 : ℚ) : ℚ :=
  1 / (rate1 + rate2 + rate3)

/-- Theorem stating that three pumps with given rates will fill a tank in 6/29 hours. -/
theorem three_pump_fill_time :
  combined_fill_time (1/3) 4 (1/2) = 6/29 := by
  sorry

#eval combined_fill_time (1/3) 4 (1/2)

end three_pump_fill_time_l3816_381682


namespace octagon_area_in_circle_l3816_381608

theorem octagon_area_in_circle (R : ℝ) : 
  R > 0 → 
  (4 * (1/2 * R^2 * Real.sin (π/4)) + 4 * (1/2 * R^2 * Real.sin (π/2))) = R^2 * (Real.sqrt 2 + 2) := by
  sorry

end octagon_area_in_circle_l3816_381608


namespace smaller_cuboid_height_l3816_381617

/-- Represents the dimensions of a cuboid -/
structure Cuboid where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculate the volume of a cuboid -/
def volume (c : Cuboid) : ℝ := c.length * c.width * c.height

/-- The original large cuboid -/
def original : Cuboid := { length := 18, width := 15, height := 2 }

/-- The smaller cuboid with unknown height -/
def smaller (h : ℝ) : Cuboid := { length := 5, width := 6, height := h }

/-- The number of smaller cuboids that can be formed -/
def num_smaller : ℕ := 6

/-- Theorem: The height of each smaller cuboid is 3 meters -/
theorem smaller_cuboid_height :
  ∃ h : ℝ, volume original = num_smaller * volume (smaller h) ∧ h = 3 := by
  sorry


end smaller_cuboid_height_l3816_381617


namespace steven_peach_apple_difference_l3816_381652

/-- The number of peaches Steven has -/
def steven_peaches : ℕ := 17

/-- The number of apples Steven has -/
def steven_apples : ℕ := 16

/-- The number of peaches Jake has -/
def jake_peaches : ℕ := steven_peaches - 6

/-- The number of apples Jake has -/
def jake_apples : ℕ := steven_apples + 8

/-- Theorem stating that Steven has 1 more peach than apples -/
theorem steven_peach_apple_difference :
  steven_peaches - steven_apples = 1 := by sorry

end steven_peach_apple_difference_l3816_381652


namespace inequality_proof_l3816_381694

-- Define the functions f and g
def f (x : ℝ) : ℝ := |2*x + 1| - |2*x - 4|
def g (x : ℝ) : ℝ := 9 + 2*x - x^2

-- State the theorem
theorem inequality_proof (x : ℝ) : |8*x - 16| ≥ g x - 2 * f x := by
  sorry

end inequality_proof_l3816_381694


namespace functional_equation_solution_l3816_381601

theorem functional_equation_solution (a : ℝ) (ha : a ≠ 0) :
  ∀ f : ℝ → ℝ, (∀ x : ℝ, f (a + x) = f x - x) →
  ∃ C : ℝ, ∀ x : ℝ, f x = C + x^2 / (2 * a) - x / 2 :=
by
  sorry

end functional_equation_solution_l3816_381601


namespace angle_C_measure_l3816_381610

-- Define a triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  angleA : ℝ
  angleB : ℝ
  angleC : ℝ

-- Define the conditions
def satisfiesConditions (t : Triangle) : Prop :=
  t.b^2 + t.c^2 - Real.sqrt 3 * t.b * t.c = t.a^2 ∧
  t.b * t.c = Real.sqrt 3 * t.a^2

-- Theorem statement
theorem angle_C_measure (t : Triangle) 
  (h : satisfiesConditions t) : t.angleC = 2 * π / 3 := by
  sorry

end angle_C_measure_l3816_381610


namespace cab_cost_for_event_l3816_381687

/-- Calculates the total cost of cab rides for a one-week event -/
def total_cab_cost (event_duration : ℕ) (distance : ℝ) (fare_per_mile : ℝ) (rides_per_day : ℕ) : ℝ :=
  event_duration * distance * fare_per_mile * rides_per_day

/-- Proves that the total cost of cab rides for the given conditions is $7000 -/
theorem cab_cost_for_event : 
  total_cab_cost 7 200 2.5 2 = 7000 := by sorry

end cab_cost_for_event_l3816_381687


namespace division_with_remainder_4032_98_l3816_381675

theorem division_with_remainder_4032_98 : ∃ (q r : ℤ), 4032 = 98 * q + r ∧ 0 ≤ r ∧ r < 98 ∧ r = 14 := by
  sorry

end division_with_remainder_4032_98_l3816_381675


namespace no_real_roots_l3816_381621

theorem no_real_roots : 
  ¬∃ x : ℝ, (3 * x^2) / (x - 2) - (3 * x + 10) / 4 + (5 - 9 * x) / (x - 2) + 2 = 0 :=
by sorry

end no_real_roots_l3816_381621


namespace student_score_l3816_381678

def max_marks : ℕ := 400
def pass_percentage : ℚ := 30 / 100
def fail_margin : ℕ := 40

theorem student_score : 
  ∀ (student_marks : ℕ),
    (student_marks = max_marks * pass_percentage - fail_margin) →
    student_marks = 80 := by
  sorry

end student_score_l3816_381678


namespace ellipse_k_range_l3816_381623

/-- 
Given an equation (x^2)/(15-k) + (y^2)/(k-9) = 1 that represents an ellipse with foci on the y-axis,
prove that k is in the open interval (12, 15).
-/
theorem ellipse_k_range (k : ℝ) :
  (∀ x y : ℝ, x^2 / (15 - k) + y^2 / (k - 9) = 1) →  -- equation represents an ellipse
  (∃ c : ℝ, ∀ x y : ℝ, x^2 / (15 - k) + y^2 / (k - 9) = 1 ↔ 
    y^2 / (k - 9) + x^2 / (15 - k) = 1 ∧ 
    y^2 / c^2 - x^2 / (k - 9 - c^2) = 1) →  -- foci are on y-axis
  k > 12 ∧ k < 15 :=
by sorry

end ellipse_k_range_l3816_381623


namespace value_of_x_l3816_381609

theorem value_of_x (n : ℝ) (x : ℝ) 
  (h1 : x = 3 * n) 
  (h2 : 2 * n + 3 = 0.20 * 25) : 
  x = 3 := by
sorry

end value_of_x_l3816_381609


namespace cistern_filling_fraction_l3816_381630

theorem cistern_filling_fraction (fill_time : ℝ) (fraction : ℝ) : 
  (fill_time = 25) → 
  (fraction * fill_time = 25) → 
  (fraction = 1 / 25) :=
by sorry

end cistern_filling_fraction_l3816_381630


namespace jenna_costume_cost_l3816_381663

/-- Represents the cost of material for Jenna's costume --/
def costume_cost (skirt_length : ℝ) (skirt_width : ℝ) (num_skirts : ℕ) 
                 (bodice_area : ℝ) (sleeve_area : ℝ) (num_sleeves : ℕ) 
                 (cost_per_sqft : ℝ) : ℝ :=
  let skirt_area := skirt_length * skirt_width
  let total_skirt_area := skirt_area * num_skirts
  let total_sleeve_area := sleeve_area * num_sleeves
  let total_area := total_skirt_area + total_sleeve_area + bodice_area
  total_area * cost_per_sqft

/-- Theorem: The total cost of material for Jenna's costume is $468 --/
theorem jenna_costume_cost : 
  costume_cost 12 4 3 2 5 2 3 = 468 := by
  sorry

end jenna_costume_cost_l3816_381663


namespace smallest_sum_of_squares_with_difference_l3816_381604

theorem smallest_sum_of_squares_with_difference (x y : ℕ) : 
  x^2 - y^2 = 221 → 
  x^2 + y^2 ≥ 229 :=
by sorry

end smallest_sum_of_squares_with_difference_l3816_381604


namespace solution_set_when_a_is_2_range_of_a_l3816_381656

-- Define the functions f and g
def f (x : ℝ) : ℝ := |x - 1|
def g (a x : ℝ) : ℝ := 2 * |x - a|

-- Part 1
theorem solution_set_when_a_is_2 :
  {x : ℝ | f x - g 2 x ≤ x - 3} = {x : ℝ | x ≤ 1 ∨ x ≥ 3} := by sorry

-- Part 2
theorem range_of_a :
  (∀ m > 1, ∃ x₀ : ℝ, f x₀ + g a x₀ ≤ (m^2 + m + 4) / (m - 1)) →
  a ∈ Set.Icc (-2 * Real.sqrt 6 - 2) (2 * Real.sqrt 6 + 4) := by sorry

end solution_set_when_a_is_2_range_of_a_l3816_381656


namespace dice_surface_area_l3816_381612

/-- The surface area of a cube with edge length 11 cm is 726 cm^2. -/
theorem dice_surface_area (edge_length : ℝ) (h : edge_length = 11) :
  6 * edge_length^2 = 726 := by
  sorry

end dice_surface_area_l3816_381612


namespace root_equation_l3816_381645

theorem root_equation (k : ℝ) : 
  ((-2 : ℝ)^2 + k*(-2) - 2 = 0) → k = 1 := by
  sorry

end root_equation_l3816_381645


namespace sum_at_one_and_neg_one_l3816_381649

/-- A cubic polynomial Q satisfying specific conditions -/
structure CubicPolynomial (l : ℝ) where
  Q : ℝ → ℝ
  is_cubic : ∃ a b c : ℝ, ∀ x, Q x = a * x^3 + b * x^2 + c * x + l
  cond_0 : Q 0 = l
  cond_2 : Q 2 = 3 * l
  cond_neg_2 : Q (-2) = 5 * l

/-- Theorem stating the sum of Q(1) and Q(-1) -/
theorem sum_at_one_and_neg_one (l : ℝ) (poly : CubicPolynomial l) : 
  poly.Q 1 + poly.Q (-1) = (7/2) * l := by
  sorry

end sum_at_one_and_neg_one_l3816_381649


namespace right_triangle_hypotenuse_l3816_381638

theorem right_triangle_hypotenuse : 
  ∀ (a : ℝ), a > 0 → a^2 = 8^2 + 15^2 → a = 17 :=
sorry

end right_triangle_hypotenuse_l3816_381638


namespace no_infinite_set_with_perfect_square_property_l3816_381692

theorem no_infinite_set_with_perfect_square_property : 
  ¬ ∃ (S : Set ℕ), Set.Infinite S ∧ 
    (∀ a b c : ℕ, a ∈ S → b ∈ S → c ∈ S → ∃ k : ℕ, a * b * c + 1 = k * k) := by
  sorry

end no_infinite_set_with_perfect_square_property_l3816_381692


namespace sine_function_translation_l3816_381668

theorem sine_function_translation (ω : ℝ) (h_ω_pos : ω > 0) :
  let f : ℝ → ℝ := λ x ↦ Real.sin (ω * x)
  let g : ℝ → ℝ := λ x ↦ f (x + π / (4 * ω))
  (∀ x : ℝ, g (2 * ω - x) = g x) →
  (∀ x y : ℝ, -ω < x ∧ x < y ∧ y < ω → g x < g y) →
  ω = Real.sqrt (π / 2) := by
sorry

end sine_function_translation_l3816_381668


namespace calculate_required_hours_johns_work_schedule_l3816_381684

/-- Calculates the required weekly work hours for a target income given previous work data --/
theorem calculate_required_hours (winter_hours_per_week : ℕ) (winter_weeks : ℕ) (winter_earnings : ℕ) 
  (target_weeks : ℕ) (target_earnings : ℕ) : ℕ :=
  let hourly_rate := winter_earnings / (winter_hours_per_week * winter_weeks)
  let total_hours := target_earnings / hourly_rate
  total_hours / target_weeks

/-- John's work schedule problem --/
theorem johns_work_schedule : 
  calculate_required_hours 40 8 3200 24 4800 = 20 := by
  sorry

end calculate_required_hours_johns_work_schedule_l3816_381684


namespace geometric_progression_existence_l3816_381602

/-- A geometric progression containing 27, 8, and 12 exists, and their positions satisfy m = 3p - 2n -/
theorem geometric_progression_existence :
  ∃ (a q : ℝ) (m n p : ℕ), 
    (a * q^(m-1) = 27) ∧ 
    (a * q^(n-1) = 8) ∧ 
    (a * q^(p-1) = 12) ∧ 
    (m = 3*p - 2*n) :=
sorry

end geometric_progression_existence_l3816_381602


namespace second_term_of_specific_sequence_l3816_381651

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1) * d

theorem second_term_of_specific_sequence :
  ∀ (d : ℝ),
  arithmetic_sequence 2020 d 1 = 2020 ∧
  arithmetic_sequence 2020 d 5 = 4040 →
  arithmetic_sequence 2020 d 2 = 2525 :=
by
  sorry

end second_term_of_specific_sequence_l3816_381651


namespace systematic_sampling_largest_number_l3816_381660

/-- Systematic sampling theorem for class selection -/
theorem systematic_sampling_largest_number
  (total_classes : ℕ)
  (selected_classes : ℕ)
  (smallest_number : ℕ)
  (h1 : total_classes = 24)
  (h2 : selected_classes = 4)
  (h3 : smallest_number = 3)
  (h4 : smallest_number > 0)
  (h5 : smallest_number ≤ total_classes)
  (h6 : selected_classes > 0)
  (h7 : selected_classes ≤ total_classes) :
  ∃ (largest_number : ℕ),
    largest_number = 21 ∧
    largest_number ≤ total_classes ∧
    (largest_number - smallest_number) = (selected_classes - 1) * (total_classes / selected_classes) :=
by sorry

end systematic_sampling_largest_number_l3816_381660


namespace sin_cos_identity_l3816_381605

theorem sin_cos_identity : 
  Real.sin (15 * π / 180) * Real.sin (105 * π / 180) - 
  Real.cos (15 * π / 180) * Real.cos (105 * π / 180) = 1 / 2 := by
  sorry

end sin_cos_identity_l3816_381605


namespace quadratic_solution_l3816_381640

theorem quadratic_solution :
  ∀ x : ℝ, (x - 2) * (x + 3) = 0 ↔ x = 2 ∨ x = -3 := by
  sorry

end quadratic_solution_l3816_381640


namespace extra_day_percentage_increase_l3816_381639

/-- Calculates the percentage increase in daily rate for an extra workday --/
theorem extra_day_percentage_increase
  (regular_daily_rate : ℚ)
  (regular_work_days_per_week : ℕ)
  (weeks_per_month : ℕ)
  (total_monthly_earnings_with_extra_day : ℚ)
  (h1 : regular_daily_rate = 8)
  (h2 : regular_work_days_per_week = 5)
  (h3 : weeks_per_month = 4)
  (h4 : total_monthly_earnings_with_extra_day = 208) :
  let regular_monthly_earnings := regular_daily_rate * regular_work_days_per_week * weeks_per_month
  let extra_day_earnings := total_monthly_earnings_with_extra_day - regular_monthly_earnings
  let extra_day_rate := extra_day_earnings / weeks_per_month
  let percentage_increase := (extra_day_rate - regular_daily_rate) / regular_daily_rate * 100
  percentage_increase = 50 := by
sorry

end extra_day_percentage_increase_l3816_381639


namespace B_power_101_l3816_381686

def B : Matrix (Fin 3) (Fin 3) ℤ :=
  !![0, 1, 0;
     0, 0, 1;
     1, 0, 0]

theorem B_power_101 :
  B ^ 101 = !![0, 0, 1;
                1, 0, 0;
                0, 1, 0] := by sorry

end B_power_101_l3816_381686


namespace race_course_length_is_correct_l3816_381693

/-- The length of a race course where two runners finish at the same time -/
def race_course_length (v_B : ℝ) : ℝ :=
  let v_A : ℝ := 4 * v_B
  let head_start : ℝ := 75
  100

theorem race_course_length_is_correct (v_B : ℝ) (h : v_B > 0) :
  let v_A : ℝ := 4 * v_B
  let head_start : ℝ := 75
  let L : ℝ := race_course_length v_B
  L / v_A = (L - head_start) / v_B :=
by
  sorry

#check race_course_length_is_correct

end race_course_length_is_correct_l3816_381693


namespace f_negative_two_lt_f_one_l3816_381643

/-- A function f : ℝ → ℝ is even if f(x) = f(-x) for all x -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

/-- The derivative of f for positive x -/
def DerivativePositive (f : ℝ → ℝ) : Prop :=
  ∀ x > 0, deriv f x = (x - 1) * (x - 2)

theorem f_negative_two_lt_f_one
  (f : ℝ → ℝ)
  (heven : IsEven f)
  (hderiv : DerivativePositive f) :
  f (-2) < f 1 :=
sorry

end f_negative_two_lt_f_one_l3816_381643


namespace transportation_time_savings_l3816_381667

def walking_time : ℕ := 98
def bicycle_saved_time : ℕ := 64
def car_saved_time : ℕ := 85
def bus_saved_time : ℕ := 55

theorem transportation_time_savings :
  (walking_time - (walking_time - bicycle_saved_time) = bicycle_saved_time) ∧
  (walking_time - (walking_time - car_saved_time) = car_saved_time) ∧
  (walking_time - (walking_time - bus_saved_time) = bus_saved_time) := by
  sorry

end transportation_time_savings_l3816_381667


namespace polynomial_equality_l3816_381661

theorem polynomial_equality (x : ℝ) :
  (∃ t c : ℝ, (6*x^2 - 8*x + 9)*(3*x^2 + t*x + 8) = 18*x^4 - 54*x^3 + c*x^2 - 56*x + 72) ↔
  (∃ t c : ℝ, t = -5 ∧ c = 115) :=
by sorry

end polynomial_equality_l3816_381661


namespace data_mode_and_mean_l3816_381679

def data : List ℕ := [5, 6, 8, 6, 8, 8, 8]

def mode (l : List ℕ) : ℕ :=
  l.foldl (fun acc x => if l.count x > l.count acc then x else acc) 0

def mean (l : List ℕ) : ℚ :=
  (l.sum : ℚ) / l.length

theorem data_mode_and_mean :
  mode data = 8 ∧ mean data = 7 := by
  sorry

end data_mode_and_mean_l3816_381679


namespace exp_greater_than_linear_l3816_381689

theorem exp_greater_than_linear (x : ℝ) (h : x > 0) : Real.exp x ≥ Real.exp 1 * x := by
  sorry

end exp_greater_than_linear_l3816_381689


namespace burning_time_3x5_rectangle_l3816_381680

/-- Represents a rectangular arrangement of toothpicks -/
structure ToothpickRectangle where
  rows : ℕ
  cols : ℕ
  total_toothpicks : ℕ

/-- Represents the burning properties of toothpicks -/
structure BurningProperties where
  single_toothpick_time : ℕ

/-- Calculates the maximum burning time for a toothpick rectangle -/
def max_burning_time (rect : ToothpickRectangle) (props : BurningProperties) : ℕ :=
  sorry

/-- Theorem: The maximum burning time for a 3x5 toothpick rectangle is 65 seconds -/
theorem burning_time_3x5_rectangle :
  let rect := ToothpickRectangle.mk 3 5 38
  let props := BurningProperties.mk 10
  max_burning_time rect props = 65 := by
  sorry

end burning_time_3x5_rectangle_l3816_381680


namespace intersection_point_is_unique_l3816_381673

/-- The intersection point of two lines in 2D space -/
def intersection_point : ℚ × ℚ := (-9/7, 20/7)

/-- First line equation: 3y = -2x + 6 -/
def line1 (x y : ℚ) : Prop := 3 * y = -2 * x + 6

/-- Second line equation: -2y = 6x + 2 -/
def line2 (x y : ℚ) : Prop := -2 * y = 6 * x + 2

theorem intersection_point_is_unique :
  ∃! p : ℚ × ℚ, line1 p.1 p.2 ∧ line2 p.1 p.2 ∧ p = intersection_point :=
sorry

end intersection_point_is_unique_l3816_381673


namespace tan_thirteen_pi_fourths_l3816_381690

theorem tan_thirteen_pi_fourths : Real.tan (13 * π / 4) = 1 := by
  sorry

end tan_thirteen_pi_fourths_l3816_381690


namespace cubic_polynomial_with_rational_roots_l3816_381655

def P (x : ℚ) : ℚ := x^3 + x^2 - x - 1

theorem cubic_polynomial_with_rational_roots :
  ∃ (r₁ r₂ r₃ : ℚ), 
    (∀ x, P x = 0 ↔ x = r₁ ∨ x = r₂ ∨ x = r₃) ∧ 
    (r₁ ≠ r₂ ∨ r₁ ≠ r₃ ∨ r₂ ≠ r₃) :=
by
  sorry

end cubic_polynomial_with_rational_roots_l3816_381655


namespace powers_of_two_sum_theorem_l3816_381627

/-- A sequence of powers of 2 -/
def PowersOfTwoSequence := List ℕ

/-- The sum of a sequence of powers of 2 -/
def sumPowersOfTwo (seq : PowersOfTwoSequence) : ℕ :=
  seq.foldl (λ sum power => sum + 2^power) 0

/-- The target sum we're aiming for -/
def targetSum : ℚ := (2^97 + 1) / (2^5 + 1)

/-- A proposition stating that a sequence of powers of 2 sums to the target sum -/
def sumsToTarget (seq : PowersOfTwoSequence) : Prop :=
  (sumPowersOfTwo seq : ℚ) = targetSum

/-- The main theorem: there exists a unique sequence of 10 powers of 2 that sums to the target -/
theorem powers_of_two_sum_theorem :
  ∃! (seq : PowersOfTwoSequence), seq.length = 10 ∧ sumsToTarget seq :=
sorry

end powers_of_two_sum_theorem_l3816_381627


namespace alex_calculation_l3816_381636

theorem alex_calculation (x : ℝ) : 
  (x / 9 - 21 = 24) → (x * 9 + 21 = 3666) := by
  sorry

end alex_calculation_l3816_381636


namespace weaving_problem_l3816_381646

def arithmetic_sequence (a₁ : ℚ) (d : ℚ) (n : ℕ) : ℚ := a₁ + (n - 1 : ℚ) * d

def sum_arithmetic_sequence (a₁ : ℚ) (d : ℚ) (n : ℕ) : ℚ :=
  (n : ℚ) / 2 * (2 * a₁ + (n - 1 : ℚ) * d)

theorem weaving_problem (a₁ d : ℚ) (n : ℕ) 
  (h₁ : a₁ = 5)
  (h₂ : n = 30)
  (h₃ : sum_arithmetic_sequence a₁ d n = 390) :
  d = 16/29 := by
  sorry

end weaving_problem_l3816_381646
