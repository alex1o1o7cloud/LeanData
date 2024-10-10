import Mathlib

namespace solve_linear_equation_l2243_224304

theorem solve_linear_equation :
  ∃ x : ℤ, 9773 + x = 13200 ∧ x = 3427 :=
by
  sorry

end solve_linear_equation_l2243_224304


namespace total_books_sold_l2243_224345

/-- Represents the sales data for a salesperson over 5 days -/
structure SalesData where
  monday : Float
  tuesday_multiplier : Float
  wednesday_multiplier : Float
  friday_multiplier : Float

/-- Calculates the total books sold by a salesperson over 5 days -/
def total_sales (data : SalesData) : Float :=
  let tuesday := data.monday * data.tuesday_multiplier
  let wednesday := tuesday * data.wednesday_multiplier
  data.monday + tuesday + wednesday + data.monday + (data.monday * data.friday_multiplier)

/-- Theorem stating the total books sold by all three salespeople -/
theorem total_books_sold (matias_data olivia_data luke_data : SalesData) 
  (h_matias : matias_data = { monday := 7, tuesday_multiplier := 2.5, wednesday_multiplier := 3.5, friday_multiplier := 4.2 })
  (h_olivia : olivia_data = { monday := 5, tuesday_multiplier := 1.5, wednesday_multiplier := 2.2, friday_multiplier := 3 })
  (h_luke : luke_data = { monday := 12, tuesday_multiplier := 0.75, wednesday_multiplier := 1.5, friday_multiplier := 0.8 }) :
  total_sales matias_data + total_sales olivia_data + total_sales luke_data = 227.75 := by
  sorry


end total_books_sold_l2243_224345


namespace max_side_length_l2243_224355

/-- A triangle with three different integer side lengths and a perimeter of 20 units -/
structure TriangleWithConstraints where
  a : ℕ
  b : ℕ
  c : ℕ
  different : a ≠ b ∧ b ≠ c ∧ a ≠ c
  perimeter : a + b + c = 20

/-- The maximum length of any side in a TriangleWithConstraints is 9 -/
theorem max_side_length (t : TriangleWithConstraints) :
  max t.a (max t.b t.c) = 9 :=
by sorry

end max_side_length_l2243_224355


namespace specific_tetrahedron_volume_l2243_224376

/-- Tetrahedron PQRS with given edge lengths -/
structure Tetrahedron where
  PQ : ℝ
  PR : ℝ
  PS : ℝ
  QR : ℝ
  QS : ℝ
  RS : ℝ

/-- The volume of a tetrahedron given its edge lengths -/
noncomputable def volume (t : Tetrahedron) : ℝ := sorry

/-- Theorem: The volume of the specific tetrahedron is 10.25 -/
theorem specific_tetrahedron_volume :
  let t : Tetrahedron := {
    PQ := 4,
    PR := 5,
    PS := 6,
    QR := 3,
    QS := Real.sqrt 37,
    RS := 7
  }
  volume t = 10.25 := by sorry

end specific_tetrahedron_volume_l2243_224376


namespace cubic_equation_root_squared_l2243_224374

theorem cubic_equation_root_squared (r : ℝ) : 
  r^3 - r + 3 = 0 → (r^2)^3 - 2*(r^2)^2 + r^2 - 9 = 0 := by
  sorry

end cubic_equation_root_squared_l2243_224374


namespace divisibility_relation_l2243_224366

theorem divisibility_relation (x y z : ℤ) (h : (11 : ℤ) ∣ (7 * x + 2 * y - 5 * z)) :
  (11 : ℤ) ∣ (3 * x - 7 * y + 12 * z) := by
  sorry

end divisibility_relation_l2243_224366


namespace combinatorial_sum_equality_l2243_224389

theorem combinatorial_sum_equality (n m k : ℕ) (h1 : 1 ≤ k) (h2 : k < m) (h3 : m ≤ n) :
  (Finset.range (k + 1)).sum (λ j => Nat.choose k j * Nat.choose n (m - j)) = Nat.choose (n + k) m := by
  sorry

end combinatorial_sum_equality_l2243_224389


namespace otimes_inequality_range_l2243_224352

-- Define the ⊗ operation
def otimes (x y : ℝ) := x * (2 - y)

-- Theorem statement
theorem otimes_inequality_range (m : ℝ) :
  (∀ x : ℝ, otimes (x + m) x < 1) ↔ -4 < m ∧ m < 0 := by sorry

end otimes_inequality_range_l2243_224352


namespace camera_pictures_l2243_224372

def picture_problem (total_albums : ℕ) (pics_per_album : ℕ) (pics_from_phone : ℕ) : Prop :=
  let total_pics := total_albums * pics_per_album
  total_pics - pics_from_phone = 13

theorem camera_pictures :
  picture_problem 5 4 7 := by
  sorry

end camera_pictures_l2243_224372


namespace fixed_point_on_line_l2243_224354

theorem fixed_point_on_line (a : ℝ) : 
  let line := fun (x y : ℝ) => a * x - y + 1 = 0
  line 0 1 := by
  sorry

end fixed_point_on_line_l2243_224354


namespace interior_exterior_angle_ratio_octagon_l2243_224314

/-- The ratio of an interior angle to an exterior angle in a regular octagon is 3:1 -/
theorem interior_exterior_angle_ratio_octagon : 
  ∀ (interior_angle exterior_angle : ℝ),
  interior_angle > 0 → 
  exterior_angle > 0 →
  (∀ (n : ℕ), n = 8 → interior_angle = (n - 2) * 180 / n) →
  (∀ (n : ℕ), n = 8 → exterior_angle = 360 / n) →
  interior_angle / exterior_angle = 3 := by
sorry

end interior_exterior_angle_ratio_octagon_l2243_224314


namespace divisibility_equivalence_l2243_224326

theorem divisibility_equivalence (a b c d : ℤ) (h : a ≠ c) :
  (∃ k : ℤ, a * b + c * d = k * (a - c)) ↔ (∃ m : ℤ, a * d + b * c = m * (a - c)) :=
by sorry

end divisibility_equivalence_l2243_224326


namespace master_title_possibilities_l2243_224371

/-- Represents a chess tournament with the given rules --/
structure ChessTournament where
  num_players : Nat
  points_for_win : Rat
  points_for_draw : Rat
  points_for_loss : Rat
  master_threshold : Rat

/-- Determines if it's possible for a given number of players to earn the Master of Sports title --/
def can_earn_master_title (t : ChessTournament) (num_masters : Nat) : Prop :=
  num_masters ≤ t.num_players ∧
  ∃ (point_distribution : Fin t.num_players → Rat),
    (∀ i, point_distribution i ≥ (t.num_players - 1 : Rat) * t.points_for_win * t.master_threshold) ∧
    (∀ i j, i ≠ j → point_distribution i + point_distribution j ≤ t.points_for_win)

/-- The specific tournament described in the problem --/
def tournament : ChessTournament :=
  { num_players := 12
  , points_for_win := 1
  , points_for_draw := 1/2
  , points_for_loss := 0
  , master_threshold := 7/10 }

theorem master_title_possibilities :
  (can_earn_master_title tournament 7) ∧
  ¬(can_earn_master_title tournament 8) := by sorry

end master_title_possibilities_l2243_224371


namespace quadratic_inequality_condition_l2243_224310

theorem quadratic_inequality_condition (a : ℝ) : 
  (∀ x : ℝ, x^2 - 2*a*x + a > 0) → (0 < a ∧ a ≤ 1) ∧ 
  ¬(0 < a ∧ a ≤ 1 → ∀ x : ℝ, x^2 - 2*a*x + a > 0) :=
by sorry

end quadratic_inequality_condition_l2243_224310


namespace margies_change_is_6_25_l2243_224388

/-- The amount of change Margie received after buying apples -/
def margies_change (num_apples : ℕ) (cost_per_apple : ℚ) (amount_paid : ℚ) : ℚ :=
  amount_paid - (num_apples : ℚ) * cost_per_apple

/-- Theorem stating that Margie's change is $6.25 given the problem conditions -/
theorem margies_change_is_6_25 :
  margies_change 5 (75 / 100) 10 = 25 / 4 :=
by sorry

end margies_change_is_6_25_l2243_224388


namespace smallest_result_l2243_224347

def S : Finset ℕ := {2, 5, 8, 11, 14}

def process (a b c : ℕ) : ℕ := (a + b) * c

theorem smallest_result :
  ∃ (a b c : ℕ), a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  process a b c = 26 ∧
  ∀ (x y z : ℕ), x ∈ S → y ∈ S → z ∈ S → x ≠ y → y ≠ z → x ≠ z →
  process x y z ≥ 26 :=
by sorry

end smallest_result_l2243_224347


namespace positive_interval_for_quadratic_l2243_224332

theorem positive_interval_for_quadratic (x : ℝ) :
  (x + 1) * (x - 3) > 0 ↔ x < -1 ∨ x > 3 := by
  sorry

end positive_interval_for_quadratic_l2243_224332


namespace car_wash_earnings_l2243_224342

theorem car_wash_earnings (friday_earnings : ℕ) (x : ℚ) : 
  friday_earnings = 147 →
  friday_earnings + (friday_earnings * x + 7) + (friday_earnings + 78) = 673 →
  x = 2 := by
sorry

end car_wash_earnings_l2243_224342


namespace binomial_expected_value_l2243_224303

/-- A random variable following a binomial distribution -/
structure BinomialRV where
  n : ℕ    -- number of trials
  p : ℝ    -- probability of success
  h1 : 0 ≤ p ∧ p ≤ 1  -- probability is between 0 and 1

/-- Expected value of a binomial random variable -/
def expected_value (ξ : BinomialRV) : ℝ := ξ.n * ξ.p

/-- Theorem: The expected value of ξ ~ B(6, 1/3) is 2 -/
theorem binomial_expected_value :
  ∀ ξ : BinomialRV, ξ.n = 6 ∧ ξ.p = 1/3 → expected_value ξ = 2 :=
by sorry

end binomial_expected_value_l2243_224303


namespace u_2023_equals_3_l2243_224315

-- Define the function g
def g : ℕ → ℕ
| 1 => 5
| 2 => 3
| 3 => 1
| 4 => 2
| 5 => 4
| _ => 0  -- For completeness, though not used in the problem

-- Define the sequence u
def u : ℕ → ℕ
| 0 => 5
| (n + 1) => g (u n)

-- Theorem statement
theorem u_2023_equals_3 : u 2023 = 3 := by
  sorry

end u_2023_equals_3_l2243_224315


namespace f_of_f_3_l2243_224341

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 1 then x^2 + 1 else 2/x

-- Theorem statement
theorem f_of_f_3 : f (f 3) = 13/9 := by
  sorry

end f_of_f_3_l2243_224341


namespace sports_club_overlapping_members_l2243_224309

theorem sports_club_overlapping_members 
  (total_members : ℕ) 
  (badminton_players : ℕ) 
  (tennis_players : ℕ) 
  (neither_players : ℕ) 
  (h1 : total_members = 30)
  (h2 : badminton_players = 17)
  (h3 : tennis_players = 21)
  (h4 : neither_players = 2) :
  badminton_players + tennis_players - total_members + neither_players = 10 := by
  sorry

end sports_club_overlapping_members_l2243_224309


namespace rectangle_circle_square_area_l2243_224313

theorem rectangle_circle_square_area : 
  ∀ (r l b : ℝ), 
    l = (2/5) * r → 
    b = 10 → 
    l * b = 220 → 
    r^2 = 3025 :=
by
  sorry

end rectangle_circle_square_area_l2243_224313


namespace cindy_marbles_l2243_224361

/-- Given Cindy's initial marbles and distribution to friends, calculate five times her remaining marbles -/
theorem cindy_marbles (initial_marbles : ℕ) (friends : ℕ) (marbles_per_friend : ℕ)
  (h1 : initial_marbles = 800)
  (h2 : friends = 6)
  (h3 : marbles_per_friend = 120) :
  5 * (initial_marbles - friends * marbles_per_friend) = 400 := by
  sorry

end cindy_marbles_l2243_224361


namespace sculpture_cost_in_rupees_l2243_224369

/-- Exchange rate from US dollars to Namibian dollars -/
def usd_to_namibian : ℝ := 5

/-- Exchange rate from US dollars to Indian rupees -/
def usd_to_rupees : ℝ := 8

/-- Cost of the sculpture in Namibian dollars -/
def sculpture_cost_namibian : ℝ := 200

/-- Theorem stating the cost of the sculpture in Indian rupees -/
theorem sculpture_cost_in_rupees :
  (sculpture_cost_namibian / usd_to_namibian) * usd_to_rupees = 320 := by
  sorry

end sculpture_cost_in_rupees_l2243_224369


namespace line_m_equation_l2243_224339

-- Define the plane
def Plane := ℝ × ℝ

-- Define a line in the plane
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define a point in the plane
def Point := Plane

-- Define the given lines
def ℓ : Line := { a := 2, b := -5, c := 0 }
def m : Line := { a := 5, b := 2, c := 0 }

-- Define the given points
def Q : Point := (3, -2)
def Q'' : Point := (-2, 3)

-- Define the reflection operation
def reflect (p : Point) (L : Line) : Point := sorry

-- State the theorem
theorem line_m_equation :
  ∃ (Q' : Point),
    reflect Q m = Q' ∧
    reflect Q' ℓ = Q'' ∧
    m.a = 5 ∧ m.b = 2 ∧ m.c = 0 := by sorry

end line_m_equation_l2243_224339


namespace rectangular_box_volume_l2243_224399

theorem rectangular_box_volume (x : ℕ) (h : x > 0) :
  (∃ (a b c : ℕ), a > 0 ∧ b > 0 ∧ c > 0 ∧ 
   a = x ∧ b = 3*x ∧ c = 4*x ∧
   a * b * c = 96) ↔ x = 2 :=
sorry

end rectangular_box_volume_l2243_224399


namespace short_track_speed_skating_selection_l2243_224365

theorem short_track_speed_skating_selection
  (p : Prop) -- A gets first place
  (q : Prop) -- B gets second place
  (r : Prop) -- C gets third place
  (h1 : p ∨ q) -- p ∨ q is true
  (h2 : ¬(p ∧ q)) -- p ∧ q is false
  (h3 : (¬q) ∧ r) -- (¬q) ∧ r is true
  : p ∧ ¬q ∧ r := by sorry

end short_track_speed_skating_selection_l2243_224365


namespace triangle_identity_l2243_224395

/-- Definition of the △ operation for ordered pairs of real numbers -/
def triangle (a b c d : ℝ) : ℝ × ℝ := (a * c + b * d, a * d + b * c)

/-- Theorem stating that if (a, b) △ (x, y) = (a, b) for all real a and b, then (x, y) = (1, 0) -/
theorem triangle_identity (x y : ℝ) : 
  (∀ a b : ℝ, triangle a b x y = (a, b)) → (x, y) = (1, 0) := by
  sorry

end triangle_identity_l2243_224395


namespace algebraic_expression_value_l2243_224327

theorem algebraic_expression_value (a : ℝ) (h : a^2 + a - 1 = 0) :
  2*a^2 + 2*a + 2021 = 2023 := by
  sorry

end algebraic_expression_value_l2243_224327


namespace initial_concentration_proof_l2243_224397

theorem initial_concentration_proof (volume_replaced : ℝ) 
  (replacement_concentration : ℝ) (final_concentration : ℝ) :
  volume_replaced = 0.7142857142857143 →
  replacement_concentration = 0.25 →
  final_concentration = 0.35 →
  ∃ initial_concentration : ℝ,
    initial_concentration = 0.6 ∧
    (1 - volume_replaced) * initial_concentration + 
      volume_replaced * replacement_concentration = final_concentration :=
by
  sorry

end initial_concentration_proof_l2243_224397


namespace min_value_parallel_vectors_l2243_224324

theorem min_value_parallel_vectors (m n : ℝ) (hm : m > 0) (hn : n > 0) :
  let a : Fin 2 → ℝ := ![2, 1]
  let b : Fin 2 → ℝ := ![4 - n, m]
  (∃ (k : ℝ), ∀ i, a i = k * b i) →
  (∀ x y, x > 0 → y > 0 → x / y + y / x ≥ 2) →
  (n / m + 8 / n ≥ 6) ∧ (∃ m₀ n₀, m₀ > 0 ∧ n₀ > 0 ∧ n₀ / m₀ + 8 / n₀ = 6) :=
by sorry

end min_value_parallel_vectors_l2243_224324


namespace curve_in_fourth_quadrant_implies_a_range_l2243_224383

-- Define the curve
def curve (x y a : ℝ) : Prop :=
  x^2 + y^2 + 2*a*x - 4*a*y + 5*a^2 - 4 = 0

-- Define the fourth quadrant
def fourth_quadrant (x y : ℝ) : Prop :=
  x > 0 ∧ y < 0

-- Theorem statement
theorem curve_in_fourth_quadrant_implies_a_range :
  (∀ x y : ℝ, curve x y a → fourth_quadrant x y) →
  a < -2 ∧ a ∈ Set.Iio (-2 : ℝ) :=
sorry

end curve_in_fourth_quadrant_implies_a_range_l2243_224383


namespace parabola_coefficients_l2243_224370

/-- A parabola with vertex (h, k) passing through point (x₀, y₀) has equation y = a(x - h)² + k -/
def is_parabola (a h k x₀ y₀ : ℝ) : Prop :=
  y₀ = a * (x₀ - h)^2 + k

/-- The general form of a parabola y = ax² + bx + c can be derived from the vertex form -/
def general_form (a h k : ℝ) : ℝ × ℝ × ℝ :=
  (a, -2*a*h, a*h^2 + k)

theorem parabola_coefficients :
  ∀ (a : ℝ), is_parabola a 4 (-1) 2 3 →
  general_form a 4 (-1) = (1, -8, 15) := by
  sorry

end parabola_coefficients_l2243_224370


namespace unique_n_with_conditions_l2243_224322

theorem unique_n_with_conditions :
  ∃! n : ℕ,
    50 ≤ n ∧ n ≤ 150 ∧
    7 ∣ n ∧
    n % 9 = 3 ∧
    n % 6 = 3 ∧
    n % 11 = 5 ∧
    n = 109 := by
  sorry

end unique_n_with_conditions_l2243_224322


namespace log_equation_solution_l2243_224350

theorem log_equation_solution :
  ∀ x : ℝ, x > 0 → (Real.log x / Real.log 2 = -1/2 ↔ x = Real.sqrt 2 / 2) := by
  sorry

end log_equation_solution_l2243_224350


namespace smallest_number_divisibility_l2243_224363

theorem smallest_number_divisibility (x : ℕ) : x = 257 ↔ 
  (x > 0) ∧ 
  (∀ z : ℕ, z > 0 → z < x → ¬((z + 7) % 8 = 0 ∧ (z + 7) % 11 = 0 ∧ (z + 7) % 24 = 0)) ∧ 
  ((x + 7) % 8 = 0) ∧ 
  ((x + 7) % 11 = 0) ∧ 
  ((x + 7) % 24 = 0) := by
sorry

end smallest_number_divisibility_l2243_224363


namespace eunji_has_most_marbles_l2243_224305

def minyoung_marbles : ℕ := 4
def yujeong_marbles : ℕ := 2
def eunji_marbles : ℕ := minyoung_marbles + 1

theorem eunji_has_most_marbles :
  eunji_marbles > minyoung_marbles ∧ eunji_marbles > yujeong_marbles :=
by
  sorry

end eunji_has_most_marbles_l2243_224305


namespace missing_number_in_mean_l2243_224373

theorem missing_number_in_mean (known_numbers : List ℤ) (mean : ℚ) : 
  known_numbers = [22, 23, 24, 25, 26, 27, 2] ∧ 
  mean = 20 ∧ 
  (List.sum known_numbers + (missing_number : ℤ)) / 7 = mean →
  missing_number = -9 :=
by
  sorry

end missing_number_in_mean_l2243_224373


namespace triangle_side_length_l2243_224353

open Real

theorem triangle_side_length 
  (g : ℝ → ℝ)
  (A B C : ℝ)
  (a b c : ℝ)
  (h1 : ∀ x, g x = cos (2 * x + π / 6))
  (h2 : (1/2) * b * c * sin A = 2)
  (h3 : b = 2)
  (h4 : g A = -1/2)
  (h5 : a < c) :
  a = 2 := by
sorry

end triangle_side_length_l2243_224353


namespace square_plus_one_geq_two_abs_l2243_224311

theorem square_plus_one_geq_two_abs (x : ℝ) : x^2 + 1 ≥ 2 * |x| := by
  sorry

end square_plus_one_geq_two_abs_l2243_224311


namespace girls_in_class_l2243_224381

theorem girls_in_class (total : ℕ) (girls : ℕ) (boys : ℕ) : 
  total = 70 → 
  4 * boys = 3 * girls → 
  total = girls + boys → 
  girls = 40 := by
sorry

end girls_in_class_l2243_224381


namespace min_value_quadratic_expression_l2243_224330

theorem min_value_quadratic_expression :
  ∀ x y : ℝ, 3 * x^2 + 4 * x * y + 2 * y^2 - 6 * x + 4 * y + 5 ≥ -1 :=
by sorry

end min_value_quadratic_expression_l2243_224330


namespace unique_prime_product_power_l2243_224364

/-- Given a natural number k, returns the product of the first k prime numbers -/
def primeProd (k : ℕ) : ℕ := sorry

/-- The only natural number k for which the product of the first k prime numbers 
    minus 1 is an exact power (greater than 1) of a natural number is 1 -/
theorem unique_prime_product_power : 
  ∀ k : ℕ, k > 0 → 
  (∃ (a n : ℕ), n > 1 ∧ primeProd k - 1 = a^n) → 
  k = 1 := by sorry

end unique_prime_product_power_l2243_224364


namespace zongzi_profit_maximization_l2243_224357

/-- Problem statement for zongzi profit maximization --/
theorem zongzi_profit_maximization 
  (cost_A cost_B : ℚ)  -- Cost prices of type A and B zongzi
  (sell_A sell_B : ℚ)  -- Selling prices of type A and B zongzi
  (total : ℕ)          -- Total number of zongzi to purchase
  :
  (cost_B = cost_A + 2) →  -- Condition 1
  (1000 / cost_A = 1200 / cost_B) →  -- Condition 2
  (sell_A = 12) →  -- Condition 5
  (sell_B = 15) →  -- Condition 6
  (total = 200) →  -- Condition 3
  ∃ (m : ℕ),  -- Number of type A zongzi purchased
    (m ≥ 2 * (total - m)) ∧  -- Condition 4
    (m < total) ∧
    (∀ (n : ℕ), n ≥ 2 * (total - n) → n < total →
      (sell_A - cost_A) * m + (sell_B - cost_B) * (total - m) ≥
      (sell_A - cost_A) * n + (sell_B - cost_B) * (total - n)) ∧
    ((sell_A - cost_A) * m + (sell_B - cost_B) * (total - m) = 466) ∧
    (m = 134) :=
by sorry

end zongzi_profit_maximization_l2243_224357


namespace lemonade_water_calculation_l2243_224396

/-- Represents the ratio of water to lemon juice in the lemonade recipe -/
def water_to_juice_ratio : ℚ := 5 / 3

/-- Represents the number of gallons of lemonade to be made -/
def gallons_to_make : ℚ := 2

/-- Represents the number of quarts in a gallon -/
def quarts_per_gallon : ℚ := 4

/-- Calculates the number of quarts of water needed for the lemonade recipe -/
def quarts_of_water_needed : ℚ :=
  (water_to_juice_ratio * gallons_to_make * quarts_per_gallon) / (water_to_juice_ratio + 1)

/-- Theorem stating that 5 quarts of water are needed for the lemonade recipe -/
theorem lemonade_water_calculation :
  quarts_of_water_needed = 5 := by sorry

end lemonade_water_calculation_l2243_224396


namespace inequality_solution_set_l2243_224392

-- Define the function f
def f (x : ℝ) : ℝ := |x| + |x - 4|

-- Define the solution set
def solution_set : Set ℝ := {x | x < -2 ∨ x > Real.sqrt 2}

-- Theorem statement
theorem inequality_solution_set :
  ∀ x : ℝ, f (x^2 + 2) > f x ↔ x ∈ solution_set :=
sorry

end inequality_solution_set_l2243_224392


namespace triangle_area_l2243_224329

/-- In triangle ABC, prove that given specific side lengths and an angle relation, 
    the area of the triangle is √3/2. -/
theorem triangle_area (A B C : ℝ) (a b c : ℝ) :
  b = 1 →
  c = 2 →
  (2 * c - b) * Real.cos A = a * Real.cos B →
  (1/2) * b * c * Real.sin A = Real.sqrt 3 / 2 :=
by sorry

end triangle_area_l2243_224329


namespace sin_theta_value_l2243_224379

theorem sin_theta_value (θ : Real) (h : Real.cos (π / 4 - θ / 2) = 2 / 3) : 
  Real.sin θ = -1 / 9 := by
  sorry

end sin_theta_value_l2243_224379


namespace unique_intersection_l2243_224386

-- Define the two functions
def f (x : ℝ) : ℝ := |3 * x + 6|
def g (x : ℝ) : ℝ := -|2 * x - 1|

-- State the theorem
theorem unique_intersection :
  ∃! p : ℝ × ℝ, 
    f p.1 = g p.1 ∧ 
    p.1 = -1 ∧ 
    p.2 = -3 := by sorry

end unique_intersection_l2243_224386


namespace min_value_of_function_l2243_224359

theorem min_value_of_function (x : ℝ) (h : x > 0) :
  (x^2 + 3*x + 1) / x ≥ 5 ∧
  ((x^2 + 3*x + 1) / x = 5 ↔ x = 1) :=
sorry

end min_value_of_function_l2243_224359


namespace horse_lap_time_l2243_224367

-- Define the given parameters
def field_area : Real := 625
def horse_speed : Real := 25

-- Define the theorem
theorem horse_lap_time : 
  ∀ (side_length perimeter time : Real),
  side_length^2 = field_area →
  perimeter = 4 * side_length →
  time = perimeter / horse_speed →
  time = 4 := by
  sorry

end horse_lap_time_l2243_224367


namespace coats_collected_at_elementary_schools_l2243_224393

theorem coats_collected_at_elementary_schools 
  (total_coats : ℕ) 
  (high_school_coats : ℕ) 
  (h1 : total_coats = 9437) 
  (h2 : high_school_coats = 6922) : 
  total_coats - high_school_coats = 2515 := by
  sorry

end coats_collected_at_elementary_schools_l2243_224393


namespace isosceles_triangle_leg_length_l2243_224336

/-- Given an isosceles triangle with perimeter 24 and base 10, prove the leg length is 7 -/
theorem isosceles_triangle_leg_length 
  (perimeter : ℝ) 
  (base : ℝ) 
  (leg : ℝ) 
  (h1 : perimeter = 24) 
  (h2 : base = 10) 
  (h3 : perimeter = base + 2 * leg) : 
  leg = 7 :=
sorry

end isosceles_triangle_leg_length_l2243_224336


namespace matrix_multiplication_example_l2243_224302

theorem matrix_multiplication_example :
  let A : Matrix (Fin 2) (Fin 2) ℤ := !![3, 1; 4, -2]
  let B : Matrix (Fin 2) (Fin 2) ℤ := !![5, -3; 2, 2]
  A * B = !![17, -7; 16, -16] := by
  sorry

end matrix_multiplication_example_l2243_224302


namespace solve_average_age_problem_l2243_224338

def average_age_problem (T : ℝ) (original_size : ℕ) (replaced_age : ℝ) (age_decrease : ℝ) : Prop :=
  let new_size : ℕ := original_size
  let new_average : ℝ := (T - replaced_age + (T / original_size - age_decrease)) / new_size
  (T / original_size) - age_decrease = new_average

theorem solve_average_age_problem :
  ∀ (T : ℝ) (original_size : ℕ) (replaced_age : ℝ) (age_decrease : ℝ),
  original_size = 20 →
  replaced_age = 60 →
  age_decrease = 4 →
  average_age_problem T original_size replaced_age age_decrease →
  (T / original_size - age_decrease) = 40 :=
sorry

end solve_average_age_problem_l2243_224338


namespace integral_equals_ten_implies_k_equals_one_l2243_224321

theorem integral_equals_ten_implies_k_equals_one :
  (∫ x in (0:ℝ)..2, (3 * x^2 + k)) = 10 → k = 1 := by
  sorry

end integral_equals_ten_implies_k_equals_one_l2243_224321


namespace swimmer_speed_in_still_water_l2243_224325

/-- Represents the speed of a swimmer in still water and the speed of the stream. -/
structure SwimmingScenario where
  v_m : ℝ  -- Speed of the man in still water (km/h)
  v_s : ℝ  -- Speed of the stream (km/h)

/-- Theorem stating that given the downstream and upstream swimming distances and times,
    the speed of the swimmer in still water is 12 km/h. -/
theorem swimmer_speed_in_still_water 
  (scenario : SwimmingScenario)
  (h_downstream : (scenario.v_m + scenario.v_s) * 3 = 54)
  (h_upstream : (scenario.v_m - scenario.v_s) * 3 = 18) :
  scenario.v_m = 12 := by
  sorry

end swimmer_speed_in_still_water_l2243_224325


namespace greatest_solution_of_equation_l2243_224394

theorem greatest_solution_of_equation (x : ℝ) : 
  (((5*x - 20)/(4*x - 5))^2 + ((5*x - 20)/(4*x - 5)) = 20) → x ≤ 9/5 :=
by sorry

end greatest_solution_of_equation_l2243_224394


namespace integer_distance_implies_horizontal_segment_l2243_224384

/-- A polynomial function with integer coefficients -/
def IntPolynomial := ℤ → ℤ

/-- The squared Euclidean distance between two points -/
def squaredDistance (x₁ y₁ x₂ y₂ : ℤ) : ℤ :=
  (x₂ - x₁)^2 + (y₂ - y₁)^2

theorem integer_distance_implies_horizontal_segment
  (f : IntPolynomial) (a b : ℤ) :
  (∃ d : ℤ, d^2 = squaredDistance a (f a) b (f b)) →
  f a = f b :=
sorry

end integer_distance_implies_horizontal_segment_l2243_224384


namespace marathon_average_time_l2243_224391

/-- Calculates the average time per mile for a marathon --/
def average_time_per_mile (distance : ℕ) (hours : ℕ) (minutes : ℕ) : ℚ :=
  (hours * 60 + minutes : ℚ) / distance

/-- Theorem: The average time per mile for a 24-mile marathon completed in 3 hours and 36 minutes is 9 minutes --/
theorem marathon_average_time :
  average_time_per_mile 24 3 36 = 9 := by
  sorry

end marathon_average_time_l2243_224391


namespace rain_on_monday_l2243_224346

theorem rain_on_monday (tuesday_rain : Real) (no_rain : Real) (both_rain : Real) 
  (h1 : tuesday_rain = 0.55)
  (h2 : no_rain = 0.35)
  (h3 : both_rain = 0.60) : 
  ∃ monday_rain : Real, monday_rain = 0.70 := by
  sorry

end rain_on_monday_l2243_224346


namespace sector_area_for_unit_radian_l2243_224323

/-- Given a circle where the arc length corresponding to a central angle of 1 radian is 2,
    prove that the area of the sector corresponding to this central angle is 2. -/
theorem sector_area_for_unit_radian (r : ℝ) (l : ℝ) (α : ℝ) : 
  α = 1 → l = 2 → α = l / r → (1 / 2) * r * l = 2 := by
  sorry

end sector_area_for_unit_radian_l2243_224323


namespace teacher_age_l2243_224337

theorem teacher_age (num_students : ℕ) (student_avg : ℝ) (new_avg : ℝ) : 
  num_students = 50 → 
  student_avg = 14 → 
  new_avg = 15 → 
  (num_students : ℝ) * student_avg + (new_avg * (num_students + 1) - num_students * student_avg) = 65 := by
  sorry

end teacher_age_l2243_224337


namespace arithmetic_expression_proof_l2243_224358

/-- Proves that the given arithmetic expression evaluates to 1320 -/
theorem arithmetic_expression_proof : 1583 + 240 / 60 * 5 - 283 = 1320 := by
  sorry

end arithmetic_expression_proof_l2243_224358


namespace sqrt_twelve_minus_sqrt_three_equals_sqrt_three_l2243_224300

theorem sqrt_twelve_minus_sqrt_three_equals_sqrt_three :
  Real.sqrt 12 - Real.sqrt 3 = Real.sqrt 3 := by
  sorry

end sqrt_twelve_minus_sqrt_three_equals_sqrt_three_l2243_224300


namespace divisors_of_36_l2243_224377

/-- The number of integer divisors of 36 -/
def num_divisors_36 : ℕ := 18

/-- A function that counts the number of integer divisors of a natural number -/
def count_divisors (n : ℕ) : ℕ :=
  (Finset.filter (λ i => n % i = 0) (Finset.range (n + 1))).card * 2

theorem divisors_of_36 :
  count_divisors 36 = num_divisors_36 := by
  sorry

#eval count_divisors 36

end divisors_of_36_l2243_224377


namespace divisibility_property_l2243_224380

def is_divisible (a b : ℕ) : Prop := ∃ k : ℕ, b * k = a

def sequence_property (A : ℕ → ℕ) : Prop :=
  ∀ n k : ℕ, is_divisible (A (n + k) - A k) (A n)

def B (A : ℕ → ℕ) : ℕ → ℕ
  | 0 => 1
  | n + 1 => B A n * A (n + 1)

theorem divisibility_property (A : ℕ → ℕ) (h : sequence_property A) :
  ∀ n k : ℕ, is_divisible (B A (n + k)) ((B A n) * (B A k)) :=
sorry

end divisibility_property_l2243_224380


namespace larger_number_proof_l2243_224312

theorem larger_number_proof (a b : ℕ) (h1 : a > 0) (h2 : b > 0) : 
  Nat.gcd a b = 84 → Nat.lcm a b = 21 → 4 * a = b → b = 84 := by
  sorry

end larger_number_proof_l2243_224312


namespace equal_money_distribution_l2243_224349

/-- Represents the money distribution problem with Carmela and her cousins -/
def money_distribution (carmela_initial : ℕ) (cousin_initial : ℕ) (num_cousins : ℕ) (amount_given : ℕ) : Prop :=
  let total_money := carmela_initial + num_cousins * cousin_initial
  let people_count := num_cousins + 1
  let carmela_final := carmela_initial - num_cousins * amount_given
  let cousin_final := cousin_initial + amount_given
  (carmela_final = cousin_final) ∧ (total_money = people_count * carmela_final)

/-- Theorem stating that giving $1 to each cousin results in equal distribution -/
theorem equal_money_distribution :
  money_distribution 7 2 4 1 := by
  sorry

end equal_money_distribution_l2243_224349


namespace john_initial_money_l2243_224344

/-- Represents John's financial transactions and final balance --/
def john_money (initial spent allowance final : ℕ) : Prop :=
  initial - spent + allowance = final

/-- Proves that John's initial money was $5 --/
theorem john_initial_money : 
  ∃ (initial : ℕ), john_money initial 2 26 29 ∧ initial = 5 := by
  sorry

end john_initial_money_l2243_224344


namespace tiling_theorem_l2243_224348

/-- Represents a tile on the board -/
inductive Tile
  | SmallTile : Tile  -- 1 x 3 tile
  | LargeTile : Tile  -- 2 x 2 tile

/-- Represents the position of the 2 x 2 tile -/
inductive LargeTilePosition
  | Central : LargeTilePosition
  | Corner : LargeTilePosition

/-- Represents a board configuration -/
structure Board :=
  (size : Nat)
  (largeTilePos : LargeTilePosition)

/-- Checks if a board can be tiled -/
def canBeTiled (b : Board) : Prop :=
  match b.largeTilePos with
  | LargeTilePosition.Central => true
  | LargeTilePosition.Corner => false

/-- The main theorem to be proved -/
theorem tiling_theorem (b : Board) (h : b.size = 10000) :
  canBeTiled b ↔ b.largeTilePos = LargeTilePosition.Central :=
sorry

end tiling_theorem_l2243_224348


namespace inequality_system_solution_set_l2243_224362

theorem inequality_system_solution_set (x : ℝ) : 
  (Set.Icc (-2 : ℝ) 0).inter (Set.Ioo (-3 : ℝ) 1) = 
  {x | |x^2 + 5*x| < 6 ∧ |x + 1| ≤ 1} :=
sorry

end inequality_system_solution_set_l2243_224362


namespace sam_book_purchase_l2243_224333

theorem sam_book_purchase (initial_amount : ℕ) (book_cost : ℕ) (remaining_amount : ℕ) 
  (h1 : initial_amount = 79)
  (h2 : book_cost = 7)
  (h3 : remaining_amount = 16) :
  (initial_amount - remaining_amount) / book_cost = 9 := by
  sorry

end sam_book_purchase_l2243_224333


namespace discount_card_saves_money_l2243_224319

-- Define the cost of the discount card
def discount_card_cost : ℝ := 100

-- Define the discount percentage
def discount_percentage : ℝ := 0.03

-- Define the cost of cakes
def cake_cost : ℝ := 500

-- Define the number of cakes
def num_cakes : ℕ := 4

-- Define the cost of fruits
def fruit_cost : ℝ := 1600

-- Calculate the total cost without discount
def total_cost_without_discount : ℝ := cake_cost * num_cakes + fruit_cost

-- Calculate the discounted amount
def discounted_amount : ℝ := total_cost_without_discount * discount_percentage

-- Calculate the total cost with discount
def total_cost_with_discount : ℝ := 
  total_cost_without_discount - discounted_amount + discount_card_cost

-- Theorem to prove that buying the discount card saves money
theorem discount_card_saves_money : 
  total_cost_with_discount < total_cost_without_discount :=
by sorry

end discount_card_saves_money_l2243_224319


namespace garage_spokes_count_l2243_224382

/-- Represents a bicycle or tricycle -/
structure Vehicle where
  front_spokes : ℕ
  back_spokes : ℕ
  middle_spokes : Option ℕ

/-- The collection of vehicles in the garage -/
def garage : List Vehicle :=
  [
    { front_spokes := 12, back_spokes := 10, middle_spokes := none },
    { front_spokes := 14, back_spokes := 12, middle_spokes := none },
    { front_spokes := 10, back_spokes := 14, middle_spokes := none },
    { front_spokes := 14, back_spokes := 16, middle_spokes := some 12 }
  ]

/-- Calculates the total number of spokes for a single vehicle -/
def spokes_per_vehicle (v : Vehicle) : ℕ :=
  v.front_spokes + v.back_spokes + (v.middle_spokes.getD 0)

/-- Calculates the total number of spokes in the garage -/
def total_spokes : ℕ :=
  garage.map spokes_per_vehicle |>.sum

/-- Theorem stating that the total number of spokes in the garage is 114 -/
theorem garage_spokes_count : total_spokes = 114 := by
  sorry

end garage_spokes_count_l2243_224382


namespace smallest_M_bound_l2243_224398

theorem smallest_M_bound : ∃ (M : ℕ),
  (∀ (a b c : ℝ), (∀ (x : ℝ), |x| ≤ 1 → |a*x^2 + b*x + c| ≤ 1) →
    (∀ (x : ℝ), |x| ≤ 1 → |2*a*x + b| ≤ M)) ∧
  (∀ (N : ℕ), N < M →
    ∃ (a b c : ℝ), (∀ (x : ℝ), |x| ≤ 1 → |a*x^2 + b*x + c| ≤ 1) ∧
      (∃ (x : ℝ), |x| ≤ 1 ∧ |2*a*x + b| > N)) ∧
  M = 4 :=
sorry

end smallest_M_bound_l2243_224398


namespace commuting_matrices_ratio_l2243_224328

/-- Given two 2x2 matrices A and B that commute, prove that (2a - 3d) / (4b - 3c) = -3 --/
theorem commuting_matrices_ratio (a b c d : ℝ) : 
  let A : Matrix (Fin 2) (Fin 2) ℝ := !![2, 3; 4, 5]
  let B : Matrix (Fin 2) (Fin 2) ℝ := !![a, b; c, d]
  (A * B = B * A) → (4 * b ≠ 3 * c) → ((2 * a - 3 * d) / (4 * b - 3 * c) = -3) := by
  sorry


end commuting_matrices_ratio_l2243_224328


namespace b_paisa_per_a_rupee_l2243_224306

-- Define the total sum of money in rupees
def total_sum : ℚ := 164

-- Define C's share in rupees
def c_share : ℚ := 32

-- Define the ratio of C's paisa to A's rupees
def c_to_a_ratio : ℚ := 40 / 100

-- Define A's share in rupees
def a_share : ℚ := c_share / c_to_a_ratio

-- Define B's share in paisa
def b_share : ℚ := (total_sum - a_share - c_share) * 100

-- Theorem to prove
theorem b_paisa_per_a_rupee : b_share / a_share = 65 := by
  sorry

end b_paisa_per_a_rupee_l2243_224306


namespace tan_sum_pi_third_l2243_224307

theorem tan_sum_pi_third (x : ℝ) (h : Real.tan x = 3) :
  Real.tan (x + π / 3) = (3 + Real.sqrt 3) / (1 - 3 * Real.sqrt 3) := by
  sorry

end tan_sum_pi_third_l2243_224307


namespace inequality_proof_l2243_224356

theorem inequality_proof (a b c : ℝ) 
  (ha : a = 17/18)
  (hb : b = Real.cos (1/3))
  (hc : c = 3 * Real.sin (1/3)) :
  c > b ∧ b > a :=
sorry

end inequality_proof_l2243_224356


namespace probability_x_plus_y_le_five_l2243_224351

/-- The probability of randomly selecting a point (x,y) from the rectangle [0,4] × [0,7] such that x + y ≤ 5 is equal to 5/14. -/
theorem probability_x_plus_y_le_five : 
  let total_area : ℝ := 4 * 7
  let favorable_area : ℝ := (1 / 2) * 5 * 4
  favorable_area / total_area = 5 / 14 := by sorry

end probability_x_plus_y_le_five_l2243_224351


namespace diophantine_equation_solutions_l2243_224360

theorem diophantine_equation_solutions :
  ∀ x y : ℤ, x^3 - y^3 = 2*x*y + 8 ↔ (x = 0 ∧ y = -2) ∨ (x = 2 ∧ y = 0) :=
sorry

end diophantine_equation_solutions_l2243_224360


namespace derivative_at_pi_over_two_l2243_224318

open Real

theorem derivative_at_pi_over_two (f : ℝ → ℝ) (hf : ∀ x, f x = sin x + 2 * x * (deriv f 0)) :
  deriv f (π / 2) = -2 := by
  sorry

end derivative_at_pi_over_two_l2243_224318


namespace collinear_points_xy_value_l2243_224343

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Check if three points are collinear -/
def collinear (p q r : Point3D) : Prop :=
  ∃ t : ℝ, (r.x - p.x) = t * (q.x - p.x) ∧
            (r.y - p.y) = t * (q.y - p.y) ∧
            (r.z - p.z) = t * (q.z - p.z)

/-- The main theorem -/
theorem collinear_points_xy_value :
  ∀ (x y : ℝ),
  let A : Point3D := ⟨1, -2, 11⟩
  let B : Point3D := ⟨4, 2, 3⟩
  let C : Point3D := ⟨x, y, 15⟩
  collinear A B C → x * y = 2 := by
  sorry

end collinear_points_xy_value_l2243_224343


namespace base4_addition_l2243_224387

/-- Addition in base 4 --/
def base4_add (a b c d : ℕ) : ℕ := sorry

/-- Convert a natural number to its base 4 representation --/
def to_base4 (n : ℕ) : List ℕ := sorry

theorem base4_addition :
  to_base4 (base4_add 1 13 313 1313) = [2, 0, 2, 0, 0] := by sorry

end base4_addition_l2243_224387


namespace tangent_line_equation_l2243_224385

-- Define the function f(x) = x³ + 1
def f (x : ℝ) : ℝ := x^3 + 1

-- Define the derivative of f(x)
def f_derivative (x : ℝ) : ℝ := 3 * x^2

-- Theorem statement
theorem tangent_line_equation :
  let x₀ : ℝ := 1
  let y₀ : ℝ := f x₀
  let m : ℝ := f_derivative x₀
  ∀ x y : ℝ, y - y₀ = m * (x - x₀) ↔ 3 * x - y - 1 = 0 :=
by sorry

end tangent_line_equation_l2243_224385


namespace geometric_sequence_sum_l2243_224368

/-- Given a geometric sequence {aₙ}, prove that a₃ + a₁₁ = 17 when a₇ = 4 and a₅ + a₉ = 10 -/
theorem geometric_sequence_sum (a : ℕ → ℝ) (r : ℝ) :
  (∀ n : ℕ, a (n + 1) = a n * r) →  -- Geometric sequence definition
  a 7 = 4 →                         -- Given condition
  a 5 + a 9 = 10 →                  -- Given condition
  a 3 + a 11 = 17 := by
sorry

end geometric_sequence_sum_l2243_224368


namespace find_x_l2243_224378

theorem find_x : ∃ x : ℝ, x = 120 ∧ 5.76 = 0.12 * (0.40 * x) := by sorry

end find_x_l2243_224378


namespace cubic_root_sum_squares_l2243_224335

theorem cubic_root_sum_squares (p q r : ℝ) : 
  (p^3 - 18*p^2 + 40*p - 15 = 0) →
  (q^3 - 18*q^2 + 40*q - 15 = 0) →
  (r^3 - 18*r^2 + 40*r - 15 = 0) →
  (p + q + r = 18) →
  (p*q + q*r + r*p = 40) →
  (p+q)^2 + (q+r)^2 + (r+p)^2 = 568 := by
sorry

end cubic_root_sum_squares_l2243_224335


namespace constant_term_of_given_equation_l2243_224308

/-- The quadratic equation 2x^2 - 3x - 1 = 0 -/
def quadratic_equation (x : ℝ) : Prop := 2 * x^2 - 3 * x - 1 = 0

/-- The constant term of a quadratic equation ax^2 + bx + c = 0 is c -/
def constant_term (a b c : ℝ) : ℝ := c

theorem constant_term_of_given_equation :
  constant_term 2 (-3) (-1) = -1 := by sorry

end constant_term_of_given_equation_l2243_224308


namespace hot_dog_stand_mayo_bottles_l2243_224316

/-- Given a ratio of ketchup : mustard : mayo bottles and the number of ketchup bottles,
    calculate the number of mayo bottles -/
def mayo_bottles (ketchup_ratio mustard_ratio mayo_ratio ketchup_bottles : ℕ) : ℕ :=
  (mayo_ratio * ketchup_bottles) / ketchup_ratio

/-- Theorem: Given the ratio 3:3:2 for ketchup:mustard:mayo and 6 ketchup bottles,
    there are 4 mayo bottles -/
theorem hot_dog_stand_mayo_bottles :
  mayo_bottles 3 3 2 6 = 4 := by
  sorry

end hot_dog_stand_mayo_bottles_l2243_224316


namespace dagger_example_l2243_224375

/-- The dagger operation on rational numbers -/
def dagger (a b : ℚ) : ℚ :=
  (a.num ^ 2 : ℚ) * b * (b.den : ℚ) / (a.den : ℚ)

/-- Theorem stating that 5/11 † 9/4 = 225/11 -/
theorem dagger_example : dagger (5 / 11) (9 / 4) = 225 / 11 := by
  sorry

end dagger_example_l2243_224375


namespace revenue_increase_80_percent_l2243_224390

/-- Represents the change in revenue given a price decrease and sales increase --/
def revenue_change (price_decrease : ℝ) (sales_increase_ratio : ℝ) : ℝ :=
  let new_price_factor := 1 - price_decrease
  let sales_increase := price_decrease * sales_increase_ratio
  let new_quantity_factor := 1 + sales_increase
  new_price_factor * new_quantity_factor - 1

/-- 
Theorem: Given a 10% price decrease and a sales increase ratio of 10,
the total revenue will increase by 80%
-/
theorem revenue_increase_80_percent :
  revenue_change 0.1 10 = 0.8 := by
  sorry

end revenue_increase_80_percent_l2243_224390


namespace increasing_linear_function_not_in_fourth_quadrant_l2243_224301

/-- A linear function that passes through (-2, 0) and increases with x -/
structure IncreasingLinearFunction where
  k : ℝ
  b : ℝ
  k_neq_zero : k ≠ 0
  passes_through_neg_two_zero : 0 = -2 * k + b
  increasing : k > 0

/-- The fourth quadrant of the Cartesian plane -/
def fourth_quadrant : Set (ℝ × ℝ) :=
  {p | p.1 > 0 ∧ p.2 < 0}

/-- The graph of a linear function -/
def graph (f : IncreasingLinearFunction) : Set (ℝ × ℝ) :=
  {p | p.2 = f.k * p.1 + f.b}

theorem increasing_linear_function_not_in_fourth_quadrant (f : IncreasingLinearFunction) :
  graph f ∩ fourth_quadrant = ∅ :=
sorry

end increasing_linear_function_not_in_fourth_quadrant_l2243_224301


namespace only_c_is_perfect_square_l2243_224340

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m ^ 2

def number_a : ℕ := 4^4 * 5^5 * 6^6
def number_b : ℕ := 4^4 * 5^6 * 6^5
def number_c : ℕ := 4^5 * 5^4 * 6^6
def number_d : ℕ := 4^6 * 5^4 * 6^5
def number_e : ℕ := 4^6 * 5^5 * 6^4

theorem only_c_is_perfect_square :
  ¬(is_perfect_square number_a) ∧
  ¬(is_perfect_square number_b) ∧
  is_perfect_square number_c ∧
  ¬(is_perfect_square number_d) ∧
  ¬(is_perfect_square number_e) :=
sorry

end only_c_is_perfect_square_l2243_224340


namespace green_ball_probability_l2243_224317

/-- Represents a container with balls -/
structure Container where
  green : ℕ
  red : ℕ

/-- The probability of selecting a green ball from a container -/
def greenProbability (c : Container) : ℚ :=
  c.green / (c.green + c.red)

/-- The containers given in the problem -/
def containers : List Container := [
  ⟨8, 2⟩,  -- Container A
  ⟨6, 4⟩,  -- Container B
  ⟨5, 5⟩,  -- Container C
  ⟨8, 2⟩   -- Container D
]

/-- The number of containers -/
def numContainers : ℕ := containers.length

/-- The theorem stating the probability of selecting a green ball -/
theorem green_ball_probability : 
  (1 / numContainers) * (containers.map greenProbability).sum = 43 / 160 := by
  sorry


end green_ball_probability_l2243_224317


namespace total_blue_balloons_l2243_224331

theorem total_blue_balloons (joan_balloons melanie_balloons john_balloons : ℕ) 
  (h1 : joan_balloons = 40)
  (h2 : melanie_balloons = 41)
  (h3 : john_balloons = 55) :
  joan_balloons + melanie_balloons + john_balloons = 136 := by
sorry

end total_blue_balloons_l2243_224331


namespace cone_sphere_ratio_l2243_224320

/-- Proves that for a right circular cone and a sphere with the same radius,
    if the volume of the cone is one-third that of the sphere,
    then the ratio of the cone's altitude to its base radius is 4/3. -/
theorem cone_sphere_ratio (r h : ℝ) (hr : r > 0) (hh : h > 0) :
  (1 / 3 * π * r^2 * h) = (1 / 3 * (4 / 3 * π * r^3)) →
  h / r = 4 / 3 := by
  sorry

end cone_sphere_ratio_l2243_224320


namespace smallest_k_for_multiple_of_180_k_1080_is_multiple_of_180_k_1080_is_smallest_l2243_224334

def sum_of_squares (k : ℕ) : ℕ := k * (k + 1) * (2 * k + 1) / 6

theorem smallest_k_for_multiple_of_180 :
  ∀ k : ℕ, k > 0 → sum_of_squares k % 180 = 0 → k ≥ 1080 :=
by sorry

theorem k_1080_is_multiple_of_180 :
  sum_of_squares 1080 % 180 = 0 :=
by sorry

theorem k_1080_is_smallest :
  ∀ k : ℕ, k > 0 → sum_of_squares k % 180 = 0 → k = 1080 :=
by sorry

end smallest_k_for_multiple_of_180_k_1080_is_multiple_of_180_k_1080_is_smallest_l2243_224334
