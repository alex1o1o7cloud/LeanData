import Mathlib

namespace NUMINAMATH_CALUDE_minimum_travel_time_for_problem_scenario_l1374_137458

/-- Represents the travel scenario between two cities -/
structure TravelScenario where
  distance : ℝ
  num_people : ℕ
  num_bicycles : ℕ
  cyclist_speed : ℝ
  pedestrian_speed : ℝ

/-- The minimum time for all people to reach the destination -/
def minimum_travel_time (scenario : TravelScenario) : ℝ :=
  sorry

/-- The specific travel scenario from the problem -/
def problem_scenario : TravelScenario :=
  { distance := 45
    num_people := 3
    num_bicycles := 2
    cyclist_speed := 15
    pedestrian_speed := 5 }

theorem minimum_travel_time_for_problem_scenario :
  minimum_travel_time problem_scenario = 3 :=
by sorry

end NUMINAMATH_CALUDE_minimum_travel_time_for_problem_scenario_l1374_137458


namespace NUMINAMATH_CALUDE_megacorp_oil_refining_earnings_l1374_137492

/-- MegaCorp's financial data and fine calculation --/
theorem megacorp_oil_refining_earnings 
  (daily_mining_earnings : ℝ)
  (monthly_expenses : ℝ)
  (fine_amount : ℝ)
  (fine_rate : ℝ)
  (days_per_month : ℕ)
  (months_per_year : ℕ)
  (h1 : daily_mining_earnings = 3000000)
  (h2 : monthly_expenses = 30000000)
  (h3 : fine_amount = 25600000)
  (h4 : fine_rate = 0.01)
  (h5 : days_per_month = 30)
  (h6 : months_per_year = 12) :
  ∃ daily_oil_earnings : ℝ,
    daily_oil_earnings = 5111111.11 ∧
    fine_amount = fine_rate * months_per_year * 
      (days_per_month * (daily_mining_earnings + daily_oil_earnings) - monthly_expenses) :=
by sorry

end NUMINAMATH_CALUDE_megacorp_oil_refining_earnings_l1374_137492


namespace NUMINAMATH_CALUDE_T_properties_l1374_137457

-- Define the operation T
def T (m n x y : ℚ) : ℚ := (m*x + n*y) * (x + 2*y)

-- State the theorem
theorem T_properties (m n : ℚ) (hm : m ≠ 0) (hn : n ≠ 0) :
  T m n 1 (-1) = 0 ∧ T m n 0 2 = 8 →
  (m = 1 ∧ n = 1) ∧
  (∀ x y : ℚ, x^2 ≠ y^2 → T m n x y = T m n y x → m = 2*n) :=
by sorry

end NUMINAMATH_CALUDE_T_properties_l1374_137457


namespace NUMINAMATH_CALUDE_binary_11011_equals_27_l1374_137419

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

theorem binary_11011_equals_27 :
  binary_to_decimal [true, true, false, true, true] = 27 := by
  sorry

end NUMINAMATH_CALUDE_binary_11011_equals_27_l1374_137419


namespace NUMINAMATH_CALUDE_cube_volume_from_diagonal_l1374_137474

theorem cube_volume_from_diagonal (d : ℝ) (h : d = 8 * Real.sqrt 3) :
  let s := d / Real.sqrt 3
  s ^ 3 = 512 := by sorry

end NUMINAMATH_CALUDE_cube_volume_from_diagonal_l1374_137474


namespace NUMINAMATH_CALUDE_cubic_function_constraint_l1374_137437

def f (a b c x : ℝ) : ℝ := 4 * x^3 + a * x^2 + b * x + c

theorem cubic_function_constraint (a b c : ℝ) :
  (∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → -1 ≤ f a b c x ∧ f a b c x ≤ 1) →
  a = 0 ∧ b = -3 ∧ c = 0 := by
  sorry

end NUMINAMATH_CALUDE_cubic_function_constraint_l1374_137437


namespace NUMINAMATH_CALUDE_line_equation_with_parallel_intersections_l1374_137464

/-- The equation of a line passing through point P(1,2) and intersecting two parallel lines,
    forming a line segment of length √2. --/
theorem line_equation_with_parallel_intersections
  (l : Set (ℝ × ℝ))  -- The line we're looking for
  (P : ℝ × ℝ)        -- Point P
  (l₁ l₂ : Set (ℝ × ℝ))  -- The two parallel lines
  (A B : ℝ × ℝ)      -- Points of intersection
  (h_P : P = (1, 2))
  (h_l₁ : l₁ = {(x, y) : ℝ × ℝ | 4*x + 3*y + 1 = 0})
  (h_l₂ : l₂ = {(x, y) : ℝ × ℝ | 4*x + 3*y + 6 = 0})
  (h_A : A ∈ l ∩ l₁)
  (h_B : B ∈ l ∩ l₂)
  (h_dist : Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = Real.sqrt 2)
  (h_P_on_l : P ∈ l) :
  l = {(x, y) : ℝ × ℝ | 7*x - y - 5 = 0} ∨
  l = {(x, y) : ℝ × ℝ | x + 7*y - 15 = 0} :=
sorry

end NUMINAMATH_CALUDE_line_equation_with_parallel_intersections_l1374_137464


namespace NUMINAMATH_CALUDE_equation_solution_l1374_137471

theorem equation_solution (x : ℝ) : 
  (8 / (Real.sqrt (x - 5) - 10) + 2 / (Real.sqrt (x - 5) - 5) + 
   9 / (Real.sqrt (x - 5) + 5) + 16 / (Real.sqrt (x - 5) + 10) = 0) ↔ 
  (x = 145 / 9 ∨ x = 1200 / 121) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l1374_137471


namespace NUMINAMATH_CALUDE_brownie_pieces_l1374_137455

/-- Proves that a 24-inch by 30-inch pan can be divided into exactly 60 pieces of 3-inch by 4-inch brownies. -/
theorem brownie_pieces (pan_length : ℕ) (pan_width : ℕ) (piece_length : ℕ) (piece_width : ℕ) :
  pan_length = 24 →
  pan_width = 30 →
  piece_length = 3 →
  piece_width = 4 →
  (pan_length * pan_width) / (piece_length * piece_width) = 60 :=
by sorry

end NUMINAMATH_CALUDE_brownie_pieces_l1374_137455


namespace NUMINAMATH_CALUDE_log_division_simplification_l1374_137489

theorem log_division_simplification :
  Real.log 27 / Real.log (1 / 27) = -1 := by
  sorry

end NUMINAMATH_CALUDE_log_division_simplification_l1374_137489


namespace NUMINAMATH_CALUDE_possible_values_of_a_l1374_137448

theorem possible_values_of_a (a b c d : ℕ+) 
  (h1 : a > b ∧ b > c ∧ c > d)
  (h2 : a + b + c + d = 2014)
  (h3 : a^2 - b^2 + c^2 - d^2 = 2014) :
  ∃! s : Finset ℕ+, s.card = 502 ∧ ∀ x : ℕ+, x ∈ s ↔ 
    ∃ b' c' d' : ℕ+, 
      x > b' ∧ b' > c' ∧ c' > d' ∧
      x + b' + c' + d' = 2014 ∧
      x^2 - b'^2 + c'^2 - d'^2 = 2014 :=
by
  sorry

end NUMINAMATH_CALUDE_possible_values_of_a_l1374_137448


namespace NUMINAMATH_CALUDE_down_payment_calculation_l1374_137402

theorem down_payment_calculation (purchase_price : ℝ) 
  (monthly_payment : ℝ) (num_payments : ℕ) (interest_rate : ℝ) :
  purchase_price = 118 →
  monthly_payment = 10 →
  num_payments = 12 →
  interest_rate = 0.15254237288135593 →
  ∃ (down_payment : ℝ),
    down_payment + (monthly_payment * num_payments) = 
      purchase_price * (1 + interest_rate) ∧
    down_payment = 16 :=
by sorry

end NUMINAMATH_CALUDE_down_payment_calculation_l1374_137402


namespace NUMINAMATH_CALUDE_min_value_theorem_l1374_137465

theorem min_value_theorem (a b k m n : ℝ) : 
  a > 0 → 
  a ≠ 1 → 
  (∀ x, a^(x-1) + 1 = b → x = k) → 
  m > 0 → 
  n > 0 → 
  m + n = b - k → 
  ∀ m' n', m' > 0 → n' > 0 → m' + n' = b - k → 
    9/m + 1/n ≤ 9/m' + 1/n' :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1374_137465


namespace NUMINAMATH_CALUDE_coefficient_of_x_in_second_equation_l1374_137462

theorem coefficient_of_x_in_second_equation 
  (x y z : ℝ) 
  (eq1 : 6*x - 5*y + 3*z = 22)
  (eq2 : x + 8*y - 11*z = 7/4)
  (eq3 : 5*x - 6*y + 2*z = 12)
  (sum_xyz : x + y + z = 10) :
  1 = 1 := by sorry

end NUMINAMATH_CALUDE_coefficient_of_x_in_second_equation_l1374_137462


namespace NUMINAMATH_CALUDE_divisibility_condition_l1374_137425

theorem divisibility_condition (x y : ℕ+) :
  (x * y^2 + 2 * y) ∣ (2 * x^2 * y + x * y^2 + 8 * x) ↔
  (∃ a : ℕ+, x = a ∧ y = 2 * a) ∨ (x = 3 ∧ y = 1) ∨ (x = 8 ∧ y = 1) :=
by sorry

end NUMINAMATH_CALUDE_divisibility_condition_l1374_137425


namespace NUMINAMATH_CALUDE_min_value_expression_l1374_137436

theorem min_value_expression (c d : ℝ) (hc : c ≠ 0) (hd : d ≠ 0) :
  c^2 + d^2 + 4/c^2 + 2*d/c ≥ 2 * Real.sqrt 3 ∧
  ∃ (c₀ d₀ : ℝ), c₀ ≠ 0 ∧ d₀ ≠ 0 ∧ c₀^2 + d₀^2 + 4/c₀^2 + 2*d₀/c₀ = 2 * Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_l1374_137436


namespace NUMINAMATH_CALUDE_expression_evaluation_l1374_137442

theorem expression_evaluation : 
  2 * 3 + 4 - 5 / 6 = 37 / 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1374_137442


namespace NUMINAMATH_CALUDE_river_lengths_theorem_l1374_137410

/-- The lengths of the Danube, Dnieper, and Don rivers satisfy the given conditions -/
theorem river_lengths_theorem (danube dnieper don : ℝ) : 
  (dnieper / danube = 5 / (19 / 3)) →
  (don / danube = 6.5 / 9.5) →
  (dnieper - don = 300) →
  (danube = 2850 ∧ dnieper = 2250 ∧ don = 1950) :=
by sorry

end NUMINAMATH_CALUDE_river_lengths_theorem_l1374_137410


namespace NUMINAMATH_CALUDE_cube_sum_problem_l1374_137418

theorem cube_sum_problem (x y : ℝ) (h1 : x + y = 10) (h2 : x * y = 12) :
  x^3 + y^3 = 640 := by sorry

end NUMINAMATH_CALUDE_cube_sum_problem_l1374_137418


namespace NUMINAMATH_CALUDE_mari_buttons_l1374_137416

theorem mari_buttons (sue_buttons : ℕ) (kendra_buttons : ℕ) (mari_buttons : ℕ) 
  (h1 : sue_buttons = 6)
  (h2 : sue_buttons = kendra_buttons / 2)
  (h3 : mari_buttons = 5 * kendra_buttons + 4) :
  mari_buttons = 64 := by
  sorry

end NUMINAMATH_CALUDE_mari_buttons_l1374_137416


namespace NUMINAMATH_CALUDE_sphere_roll_coplanar_l1374_137406

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a sphere -/
structure Sphere where
  center : Point3D
  radius : ℝ

/-- Represents a rectangular box -/
structure RectangularBox where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Represents the transformation of a point on a sphere's surface after rolling -/
def sphereRoll (s : Sphere) (b : RectangularBox) (p : Point3D) : Point3D :=
  sorry

/-- States that four points lie in the same plane -/
def coplanar (p1 p2 p3 p4 : Point3D) : Prop :=
  sorry

/-- The main theorem to prove -/
theorem sphere_roll_coplanar (s : Sphere) (b : RectangularBox) (X : Point3D) :
  let X₁ := sphereRoll s b X
  let X₂ := sphereRoll s b X₁
  let X₃ := sphereRoll s b X₂
  coplanar X X₁ X₂ X₃ :=
sorry

end NUMINAMATH_CALUDE_sphere_roll_coplanar_l1374_137406


namespace NUMINAMATH_CALUDE_bob_has_81_robots_l1374_137412

/-- The number of car robots Tom and Michael have together -/
def tom_and_michael_robots : ℕ := 9

/-- The factor by which Bob has more car robots than Tom and Michael -/
def bob_factor : ℕ := 9

/-- The total number of car robots Bob has -/
def bob_robots : ℕ := tom_and_michael_robots * bob_factor

/-- Theorem stating that Bob has 81 car robots -/
theorem bob_has_81_robots : bob_robots = 81 := by
  sorry

end NUMINAMATH_CALUDE_bob_has_81_robots_l1374_137412


namespace NUMINAMATH_CALUDE_misread_weight_calculation_l1374_137481

/-- Proves that the misread weight in a class of 20 boys is 56 kg given the initial and correct average weights --/
theorem misread_weight_calculation (n : ℕ) (initial_avg : ℝ) (correct_avg : ℝ) (correct_weight : ℝ) :
  n = 20 →
  initial_avg = 58.4 →
  correct_avg = 58.65 →
  correct_weight = 61 →
  ∃ (misread_weight : ℝ),
    misread_weight = 56 ∧
    n * initial_avg + (correct_weight - misread_weight) = n * correct_avg :=
by
  sorry

end NUMINAMATH_CALUDE_misread_weight_calculation_l1374_137481


namespace NUMINAMATH_CALUDE_smallest_n_divisible_by_2010_l1374_137456

theorem smallest_n_divisible_by_2010 (a : ℕ → ℤ) 
  (h1 : ∃ k, a 1 = 2 * k + 1)
  (h2 : ∀ n : ℕ, n > 0 → n * (a (n + 1) - a n + 3) = a (n + 1) + a n + 3)
  (h3 : ∃ k, a 2009 = 2010 * k) :
  ∃ n : ℕ, n ≥ 2 ∧ (∃ k, a n = 2010 * k) ∧ (∀ m, 2 ≤ m ∧ m < n → ¬∃ k, a m = 2010 * k) ∧ n = 671 :=
sorry

end NUMINAMATH_CALUDE_smallest_n_divisible_by_2010_l1374_137456


namespace NUMINAMATH_CALUDE_prob_first_ace_is_one_eighth_l1374_137414

/-- Represents a deck of cards -/
structure Deck :=
  (size : ℕ)
  (num_aces : ℕ)

/-- Represents the card game setup -/
structure CardGame :=
  (deck : Deck)
  (num_players : ℕ)

/-- The probability of a player getting the first Ace -/
def prob_first_ace (game : CardGame) (player : ℕ) : ℚ :=
  1 / game.num_players

/-- Theorem stating that the probability of each player getting the first Ace is 1/8 -/
theorem prob_first_ace_is_one_eighth (game : CardGame) :
  game.deck.size = 32 ∧ game.deck.num_aces = 4 ∧ game.num_players = 4 →
  ∀ player, player > 0 ∧ player ≤ game.num_players →
    prob_first_ace game player = 1 / 8 :=
sorry

end NUMINAMATH_CALUDE_prob_first_ace_is_one_eighth_l1374_137414


namespace NUMINAMATH_CALUDE_triangle_properties_l1374_137452

/-- Properties of a triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Theorem about triangle properties -/
theorem triangle_properties (t : Triangle) :
  (t.c = 2 ∧ t.C = π / 3 ∧ (1 / 2) * t.a * t.b * Real.sin t.C = Real.sqrt 3) →
  (Real.cos (t.A + t.B) = -1 / 2 ∧ t.a = 2 ∧ t.b = 2) ∧
  (t.B > π / 2 ∧ Real.cos t.A = 3 / 5 ∧ Real.sin t.B = 12 / 13) →
  Real.sin t.C = 16 / 65 := by
  sorry


end NUMINAMATH_CALUDE_triangle_properties_l1374_137452


namespace NUMINAMATH_CALUDE_set_equality_implies_difference_l1374_137477

theorem set_equality_implies_difference (a b : ℝ) :
  ({a, 1} : Set ℝ) = {0, a + b} → b - a = 1 := by
  sorry

end NUMINAMATH_CALUDE_set_equality_implies_difference_l1374_137477


namespace NUMINAMATH_CALUDE_picture_book_shelves_l1374_137497

theorem picture_book_shelves (books_per_shelf : ℕ) (mystery_shelves : ℕ) (total_books : ℕ)
  (h1 : books_per_shelf = 8)
  (h2 : mystery_shelves = 5)
  (h3 : total_books = 72) :
  (total_books - mystery_shelves * books_per_shelf) / books_per_shelf = 4 := by
  sorry

end NUMINAMATH_CALUDE_picture_book_shelves_l1374_137497


namespace NUMINAMATH_CALUDE_digit_sum_theorem_l1374_137422

theorem digit_sum_theorem (f o g : ℕ) : 
  f < 10 → o < 10 → g < 10 →
  4 * (100 * f + 10 * o + g) = 1464 →
  f + o + g = 15 := by
sorry

end NUMINAMATH_CALUDE_digit_sum_theorem_l1374_137422


namespace NUMINAMATH_CALUDE_problem_statement_l1374_137493

-- Define the function f
noncomputable def f (a m x : ℝ) : ℝ := m * x^a + (Real.log (1 + x))^a - a * Real.log (1 - x) - 2

-- State the theorem
theorem problem_statement (a : ℝ) (h1 : a^(1/2) ≤ 3) (h2 : Real.log 3 / Real.log a ≤ 1/2) :
  ((0 < a ∧ a < 1) ∨ a = 9) ∧
  (a > 1 → ∃ m : ℝ, f a m (1/2) = a → f a m (-1/2) = -13) :=
sorry

end NUMINAMATH_CALUDE_problem_statement_l1374_137493


namespace NUMINAMATH_CALUDE_repetend_of_5_17_l1374_137440

/-- The repetend of a fraction is the repeating sequence of digits in its decimal representation. -/
def is_repetend (n : ℕ) (d : ℕ) (r : ℕ) : Prop :=
  ∃ (k : ℕ), 10^6 * (10 * n - d * r) = d * (10^k - 1)

/-- The 6-digit repetend in the decimal representation of 5/17 is 294117. -/
theorem repetend_of_5_17 : is_repetend 5 17 294117 := by
  sorry

end NUMINAMATH_CALUDE_repetend_of_5_17_l1374_137440


namespace NUMINAMATH_CALUDE_transmission_time_l1374_137431

/-- Proves that given the specified conditions, the transmission time is 5 minutes -/
theorem transmission_time (blocks : ℕ) (chunks_per_block : ℕ) (transmission_rate : ℕ) :
  blocks = 80 →
  chunks_per_block = 640 →
  transmission_rate = 160 →
  (blocks * chunks_per_block : ℝ) / transmission_rate / 60 = 5 := by
  sorry

end NUMINAMATH_CALUDE_transmission_time_l1374_137431


namespace NUMINAMATH_CALUDE_cycle_reappearance_l1374_137405

theorem cycle_reappearance (letter_cycle_length digit_cycle_length : ℕ) 
  (h1 : letter_cycle_length = 7)
  (h2 : digit_cycle_length = 4) :
  Nat.lcm letter_cycle_length digit_cycle_length = 28 := by
  sorry

end NUMINAMATH_CALUDE_cycle_reappearance_l1374_137405


namespace NUMINAMATH_CALUDE_fraction_equality_l1374_137429

theorem fraction_equality (a b : ℝ) 
  (ha : a ≠ 0) 
  (hb : b ≠ 0) 
  (hab : a - b * (1 / a) ≠ 0) : 
  (a^2 - 1/b^2) / (b^2 - 1/a^2) = a^2 / b^2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1374_137429


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l1374_137409

theorem fraction_to_decimal : (47 : ℚ) / (2 * 5^3) = 0.188 := by sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l1374_137409


namespace NUMINAMATH_CALUDE_sum_three_fourths_power_inequality_l1374_137482

theorem sum_three_fourths_power_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a^(3/4) + b^(3/4) + c^(3/4) > (a + b + c)^(3/4) := by
  sorry

end NUMINAMATH_CALUDE_sum_three_fourths_power_inequality_l1374_137482


namespace NUMINAMATH_CALUDE_imaginary_unit_cube_l1374_137461

theorem imaginary_unit_cube (i : ℂ) (h : i^2 = -1) : 1 + i^3 = 1 - i := by
  sorry

end NUMINAMATH_CALUDE_imaginary_unit_cube_l1374_137461


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l1374_137407

theorem geometric_sequence_problem (a : ℝ) (h1 : a > 0) 
  (h2 : ∃ r : ℝ, r > 0 ∧ 30 * r = a ∧ a * r = 9/4) : 
  a = 15 * Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l1374_137407


namespace NUMINAMATH_CALUDE_prob_less_than_8_l1374_137434

/-- The probability of an archer scoring less than 8 in a single shot -/
theorem prob_less_than_8 (p_10 p_9 p_8 : ℝ) 
  (h1 : p_10 = 0.24)
  (h2 : p_9 = 0.28)
  (h3 : p_8 = 0.19) : 
  1 - (p_10 + p_9 + p_8) = 0.29 := by
  sorry

end NUMINAMATH_CALUDE_prob_less_than_8_l1374_137434


namespace NUMINAMATH_CALUDE_sum_of_roots_l1374_137430

theorem sum_of_roots (k c x₁ x₂ : ℝ) (h_distinct : x₁ ≠ x₂) 
  (h₁ : 2 * x₁^2 - k * x₁ = 2 * c) (h₂ : 2 * x₂^2 - k * x₂ = 2 * c) : 
  x₁ + x₂ = k / 2 := by
sorry

end NUMINAMATH_CALUDE_sum_of_roots_l1374_137430


namespace NUMINAMATH_CALUDE_smallest_square_side_length_l1374_137488

theorem smallest_square_side_length : ∃ (n : ℕ), 
  (∀ (a b c d : ℕ), a * b * c * d = n * n) ∧ 
  (∃ (x y z w : ℕ), x * 7 = n ∧ y * 8 = n ∧ z * 9 = n ∧ w * 10 = n) ∧
  (∀ (m : ℕ), 
    (∃ (a b c d : ℕ), a * b * c * d = m * m) ∧ 
    (∃ (x y z w : ℕ), x * 7 = m ∧ y * 8 = m ∧ z * 9 = m ∧ w * 10 = m) →
    m ≥ n) ∧
  n = 1008 := by
  sorry

end NUMINAMATH_CALUDE_smallest_square_side_length_l1374_137488


namespace NUMINAMATH_CALUDE_ellipse_vertex_distance_l1374_137447

/-- The distance between vertices of an ellipse with equation x²/121 + y²/49 = 1 is 22 -/
theorem ellipse_vertex_distance :
  let a : ℝ := Real.sqrt 121
  let b : ℝ := Real.sqrt 49
  let ellipse_equation := fun (x y : ℝ) => x^2 / 121 + y^2 / 49 = 1
  2 * a = 22 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_vertex_distance_l1374_137447


namespace NUMINAMATH_CALUDE_sum_of_divisors_882_prime_factors_l1374_137432

def sum_of_divisors (n : ℕ) : ℕ := sorry

def count_distinct_prime_factors (n : ℕ) : ℕ := sorry

theorem sum_of_divisors_882_prime_factors :
  count_distinct_prime_factors (sum_of_divisors 882) = 3 := by sorry

end NUMINAMATH_CALUDE_sum_of_divisors_882_prime_factors_l1374_137432


namespace NUMINAMATH_CALUDE_trig_equation_solution_l1374_137478

theorem trig_equation_solution (z : ℝ) :
  (1 - Real.sin z ^ 6 - Real.cos z ^ 6) / (1 - Real.sin z ^ 4 - Real.cos z ^ 4) = 2 * (Real.cos (3 * z)) ^ 2 →
  ∃ k : ℤ, z = π / 18 * (6 * ↑k + 1) ∨ z = π / 18 * (6 * ↑k - 1) :=
by sorry

end NUMINAMATH_CALUDE_trig_equation_solution_l1374_137478


namespace NUMINAMATH_CALUDE_factor_x_10_minus_1024_l1374_137453

theorem factor_x_10_minus_1024 (x : ℝ) :
  x^10 - 1024 = (x^5 + 32) * (x^(5/2) + Real.sqrt 32) * (x^(5/2) - Real.sqrt 32) := by
  sorry

end NUMINAMATH_CALUDE_factor_x_10_minus_1024_l1374_137453


namespace NUMINAMATH_CALUDE_unique_integers_sum_l1374_137473

theorem unique_integers_sum (x : ℝ) : x = Real.sqrt ((Real.sqrt 77) / 2 + 5 / 2) →
  ∃! (a b c : ℕ+), 
    x^100 = 4*x^98 + 18*x^96 + 19*x^94 - x^50 + (a : ℝ)*x^46 + (b : ℝ)*x^44 + (c : ℝ)*x^40 ∧
    (a : ℕ) + (b : ℕ) + (c : ℕ) = 534 := by
  sorry

end NUMINAMATH_CALUDE_unique_integers_sum_l1374_137473


namespace NUMINAMATH_CALUDE_chicken_wings_distribution_l1374_137496

theorem chicken_wings_distribution (num_friends : ℕ) (initial_wings : ℕ) (additional_wings : ℕ) :
  num_friends = 4 →
  initial_wings = 9 →
  additional_wings = 7 →
  (initial_wings + additional_wings) / num_friends = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_chicken_wings_distribution_l1374_137496


namespace NUMINAMATH_CALUDE_shape_sum_theorem_l1374_137420

-- Define the shapes as real numbers
variable (triangle : ℝ) (circle : ℝ) (square : ℝ)

-- Define the conditions from the problem
def condition1 : Prop := 2 * triangle + 2 * circle + square = 27
def condition2 : Prop := 2 * circle + triangle + square = 26
def condition3 : Prop := 2 * square + triangle + circle = 23

-- Define the theorem
theorem shape_sum_theorem 
  (h1 : condition1 triangle circle square)
  (h2 : condition2 triangle circle square)
  (h3 : condition3 triangle circle square) :
  2 * triangle + 3 * circle + square = 45.5 := by
  sorry

end NUMINAMATH_CALUDE_shape_sum_theorem_l1374_137420


namespace NUMINAMATH_CALUDE_total_strings_is_72_l1374_137428

/-- Calculates the total number of strings John needs to restring his instruments. -/
def total_strings : ℕ :=
  let num_basses : ℕ := 3
  let strings_per_bass : ℕ := 4
  let num_guitars : ℕ := 2 * num_basses
  let strings_per_guitar : ℕ := 6
  let num_eight_string_guitars : ℕ := num_guitars - 3
  let strings_per_eight_string_guitar : ℕ := 8
  
  (num_basses * strings_per_bass) +
  (num_guitars * strings_per_guitar) +
  (num_eight_string_guitars * strings_per_eight_string_guitar)

theorem total_strings_is_72 : total_strings = 72 := by
  sorry

end NUMINAMATH_CALUDE_total_strings_is_72_l1374_137428


namespace NUMINAMATH_CALUDE_product_calculation_l1374_137423

theorem product_calculation : 2.4 * 8.2 * (5.3 - 4.7) = 11.52 := by
  sorry

end NUMINAMATH_CALUDE_product_calculation_l1374_137423


namespace NUMINAMATH_CALUDE_no_integer_solution_for_ten_l1374_137415

theorem no_integer_solution_for_ten :
  ¬ ∃ (x y z : ℤ), 3 * x^2 + 4 * y^2 - 5 * z^2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solution_for_ten_l1374_137415


namespace NUMINAMATH_CALUDE_dayan_20th_term_dayan_even_term_formula_l1374_137485

def dayan_sequence : ℕ → ℕ
| 0 => 0
| 1 => 2
| 2 => 4
| 3 => 8
| 4 => 12
| 5 => 18
| 6 => 24
| 7 => 32
| 8 => 40
| 9 => 50
| n + 10 => dayan_sequence n  -- placeholder for terms beyond 10th

theorem dayan_20th_term : dayan_sequence 19 = 200 := by
  sorry

theorem dayan_even_term_formula (n : ℕ) : dayan_sequence (2 * n - 1) = 2 * n^2 := by
  sorry

end NUMINAMATH_CALUDE_dayan_20th_term_dayan_even_term_formula_l1374_137485


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l1374_137417

/-- Given a hyperbola and a line passing through its right focus, 
    prove the equations of the asymptotes -/
theorem hyperbola_asymptotes 
  (a b : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (C : ℝ × ℝ → Prop) 
  (l : ℝ × ℝ → Prop) 
  (F : ℝ × ℝ) :
  (C = λ (x, y) ↦ x^2 / a^2 - y^2 / b^2 = 1) →
  (l = λ (x, y) ↦ x + 3*y - 2*b = 0) →
  (∃ c, F = (c, 0) ∧ l F) →
  (∃ f : ℝ → ℝ, f x = Real.sqrt 3 / 3 * x ∧ 
   ∀ (x y : ℝ), (C (x, y) → (y = f x ∨ y = -f x))) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l1374_137417


namespace NUMINAMATH_CALUDE_binomial_10_choose_5_l1374_137433

theorem binomial_10_choose_5 : Nat.choose 10 5 = 252 := by
  sorry

end NUMINAMATH_CALUDE_binomial_10_choose_5_l1374_137433


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_l1374_137426

theorem perfect_square_trinomial : 120^2 - 40 * 120 + 20^2 = 10000 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_trinomial_l1374_137426


namespace NUMINAMATH_CALUDE_max_difference_bounded_l1374_137469

theorem max_difference_bounded (a : Fin 2017 → ℝ) 
  (h1 : a 1 = a 2017)
  (h2 : ∀ i : Fin 2015, |a i + a (i + 2) - 2 * a (i + 1)| ≤ 1) :
  ∃ M : ℝ, M = 508032 ∧ 
  (∀ i j : Fin 2017, i < j → |a i - a j| ≤ M) ∧
  (∃ i j : Fin 2017, i < j ∧ |a i - a j| = M) := by
sorry

end NUMINAMATH_CALUDE_max_difference_bounded_l1374_137469


namespace NUMINAMATH_CALUDE_expected_rolls_in_year_l1374_137466

/-- Represents the outcome of rolling an 8-sided die -/
inductive DieOutcome
  | Composite
  | Prime
  | RollAgain

/-- The probability distribution of the die outcomes -/
def dieProb : DieOutcome → ℚ
  | DieOutcome.Composite => 3/8
  | DieOutcome.Prime => 1/2
  | DieOutcome.RollAgain => 1/8

/-- The expected number of rolls on a single day -/
def expectedRollsPerDay : ℚ := 1

/-- The number of days in a non-leap year -/
def daysInYear : ℕ := 365

/-- The expected number of rolls in a non-leap year -/
def expectedRollsInYear : ℚ := expectedRollsPerDay * daysInYear

theorem expected_rolls_in_year :
  expectedRollsInYear = 365 := by sorry

end NUMINAMATH_CALUDE_expected_rolls_in_year_l1374_137466


namespace NUMINAMATH_CALUDE_parallelogram_perimeter_l1374_137495

/-- Represents a parallelogram EFGH with given side lengths and diagonal -/
structure Parallelogram where
  EF : ℝ
  FG : ℝ
  EH : ℝ

/-- The perimeter of a parallelogram -/
def perimeter (p : Parallelogram) : ℝ := 2 * (p.EF + p.FG)

/-- Theorem: The perimeter of parallelogram EFGH is 140 units -/
theorem parallelogram_perimeter (p : Parallelogram) 
  (h1 : p.EF = 40)
  (h2 : p.FG = 30)
  (h3 : p.EH = 50) : 
  perimeter p = 140 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_perimeter_l1374_137495


namespace NUMINAMATH_CALUDE_sphere_cylinder_volume_difference_l1374_137459

/-- The volume of the space inside a sphere and outside an inscribed right cylinder -/
theorem sphere_cylinder_volume_difference (r_sphere : ℝ) (r_cylinder : ℝ) :
  r_sphere = 6 →
  r_cylinder = 4 →
  let h_cylinder := 2 * Real.sqrt (r_sphere ^ 2 - r_cylinder ^ 2)
  let v_sphere := (4 / 3) * Real.pi * r_sphere ^ 3
  let v_cylinder := Real.pi * r_cylinder ^ 2 * h_cylinder
  v_sphere - v_cylinder = (288 - 64 * Real.sqrt 5) * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_sphere_cylinder_volume_difference_l1374_137459


namespace NUMINAMATH_CALUDE_chessboard_not_fully_covered_l1374_137498

/-- Represents the dimensions of a square chessboard -/
def BoardSize : ℕ := 10

/-- Represents the number of squares covered by one L-shaped tromino piece -/
def SquaresPerPiece : ℕ := 3

/-- Represents the number of L-shaped tromino pieces available -/
def NumberOfPieces : ℕ := 25

/-- Theorem stating that the chessboard cannot be fully covered by the given pieces -/
theorem chessboard_not_fully_covered :
  NumberOfPieces * SquaresPerPiece < BoardSize * BoardSize := by
  sorry

end NUMINAMATH_CALUDE_chessboard_not_fully_covered_l1374_137498


namespace NUMINAMATH_CALUDE_intersection_A_B_when_m_3_range_of_m_when_A_subset_B_l1374_137403

-- Define set A
def A : Set ℝ := {x | ∃ y, y = Real.sqrt (3 - 2*x - x^2)}

-- Define set B
def B (m : ℝ) : Set ℝ := {x | x^2 - 2*x + 1 - m^2 ≤ 0}

-- Part 1
theorem intersection_A_B_when_m_3 : 
  A ∩ B 3 = {x | -2 ≤ x ∧ x ≤ 1} := by sorry

-- Part 2
theorem range_of_m_when_A_subset_B : 
  ∀ m > 0, A ⊆ B m → m ≥ 4 := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_when_m_3_range_of_m_when_A_subset_B_l1374_137403


namespace NUMINAMATH_CALUDE_intersection_point_is_solution_l1374_137413

-- Define the two lines
def line1 (x y : ℚ) : Prop := 2 * x - 3 * y = 3
def line2 (x y : ℚ) : Prop := 4 * x + 2 * y = 2

-- Define the intersection point
def intersection_point : ℚ × ℚ := (3/4, -1/2)

-- Theorem statement
theorem intersection_point_is_solution :
  let (x, y) := intersection_point
  (line1 x y ∧ line2 x y) ∧
  ∀ x' y', line1 x' y' ∧ line2 x' y' → (x', y') = (x, y) := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_is_solution_l1374_137413


namespace NUMINAMATH_CALUDE_program_result_l1374_137483

def program_output (a₀ b₀ : ℕ) : ℕ × ℕ :=
  let a₁ := a₀ + b₀
  let b₁ := b₀ * a₁
  (a₁, b₁)

theorem program_result :
  program_output 1 3 = (4, 12) := by sorry

end NUMINAMATH_CALUDE_program_result_l1374_137483


namespace NUMINAMATH_CALUDE_segments_5_6_10_form_triangle_l1374_137427

/-- Triangle inequality theorem: the sum of the lengths of any two sides of a triangle
    must be greater than the length of the remaining side. -/
def triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- A function that determines if three line segments can form a triangle. -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ triangle_inequality a b c

/-- Theorem stating that the line segments 5, 6, and 10 can form a triangle. -/
theorem segments_5_6_10_form_triangle :
  can_form_triangle 5 6 10 := by sorry

end NUMINAMATH_CALUDE_segments_5_6_10_form_triangle_l1374_137427


namespace NUMINAMATH_CALUDE_ada_original_seat_l1374_137443

-- Define the seats
inductive Seat : Type
| one : Seat
| two : Seat
| three : Seat
| four : Seat
| five : Seat
| six : Seat

-- Define the friends
inductive Friend : Type
| ada : Friend
| bea : Friend
| ceci : Friend
| dee : Friend
| edie : Friend
| fred : Friend

-- Define the seating arrangement as a function from Friend to Seat
def Seating := Friend → Seat

-- Define the movement function
def move (s : Seating) : Seating :=
  fun f => match f with
    | Friend.bea => match s Friend.bea with
      | Seat.one => Seat.two
      | Seat.two => Seat.three
      | Seat.three => Seat.four
      | Seat.four => Seat.five
      | Seat.five => Seat.six
      | Seat.six => Seat.six
    | Friend.ceci => match s Friend.ceci with
      | Seat.one => Seat.one
      | Seat.two => Seat.one
      | Seat.three => Seat.one
      | Seat.four => Seat.two
      | Seat.five => Seat.three
      | Seat.six => Seat.four
    | Friend.dee => s Friend.edie
    | Friend.edie => s Friend.dee
    | Friend.fred => s Friend.fred
    | Friend.ada => s Friend.ada

-- Theorem stating Ada's original seat
theorem ada_original_seat (initial : Seating) :
  (move initial) Friend.ada = Seat.one →
  initial Friend.ada = Seat.two :=
by
  sorry


end NUMINAMATH_CALUDE_ada_original_seat_l1374_137443


namespace NUMINAMATH_CALUDE_paper_boat_time_l1374_137486

/-- The time it takes for a paper boat to travel along an embankment -/
theorem paper_boat_time (embankment_length : ℝ) (boat_length : ℝ) 
  (downstream_time : ℝ) (upstream_time : ℝ) 
  (h1 : embankment_length = 50) 
  (h2 : boat_length = 10)
  (h3 : downstream_time = 5)
  (h4 : upstream_time = 4) : 
  ∃ (paper_boat_time : ℝ), paper_boat_time = 40 := by
  sorry

end NUMINAMATH_CALUDE_paper_boat_time_l1374_137486


namespace NUMINAMATH_CALUDE_eliana_steps_l1374_137499

def steps_day1 (x : ℕ) := 200 + x
def steps_day2 (x : ℕ) := 2 * steps_day1 x
def steps_day3 (x : ℕ) := steps_day2 x + 100

theorem eliana_steps (x : ℕ) :
  steps_day1 x + steps_day2 x + steps_day3 x = 1600 → x = 100 := by
  sorry

end NUMINAMATH_CALUDE_eliana_steps_l1374_137499


namespace NUMINAMATH_CALUDE_correct_addition_l1374_137470

theorem correct_addition (x : ℤ) (h : x + 42 = 50) : x + 24 = 32 := by
  sorry

end NUMINAMATH_CALUDE_correct_addition_l1374_137470


namespace NUMINAMATH_CALUDE_least_x_for_even_prime_quotient_l1374_137404

theorem least_x_for_even_prime_quotient :
  ∃ (x p q : ℕ),
    x > 0 ∧
    Prime p ∧
    Prime q ∧
    p ≠ q ∧
    q - p = 3 ∧
    x / (11 * p * q) = 2 ∧
    (∀ y, y > 0 → y / (11 * p * q) = 2 → y ≥ x) ∧
    x = 770 :=
by sorry

end NUMINAMATH_CALUDE_least_x_for_even_prime_quotient_l1374_137404


namespace NUMINAMATH_CALUDE_largest_four_digit_divisible_by_five_l1374_137467

theorem largest_four_digit_divisible_by_five : ∃ n : ℕ,
  n = 9995 ∧
  n ≥ 1000 ∧ n < 10000 ∧
  n % 5 = 0 ∧
  ∀ m : ℕ, m ≥ 1000 ∧ m < 10000 ∧ m % 5 = 0 → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_largest_four_digit_divisible_by_five_l1374_137467


namespace NUMINAMATH_CALUDE_sum_of_squares_of_roots_l1374_137490

theorem sum_of_squares_of_roots (a b c : ℝ) : 
  (3 * a^3 + 2 * a^2 - 3 * a - 8 = 0) →
  (3 * b^3 + 2 * b^2 - 3 * b - 8 = 0) →
  (3 * c^3 + 2 * c^2 - 3 * c - 8 = 0) →
  a^2 + b^2 + c^2 = 22 / 9 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_roots_l1374_137490


namespace NUMINAMATH_CALUDE_triangle_area_is_12_l1374_137445

/-- The area of the triangular region bounded by the coordinate axes and the line 3x + 2y = 12 -/
def triangleArea : ℝ := 12

/-- The equation of the bounding line -/
def boundingLine (x y : ℝ) : Prop := 3 * x + 2 * y = 12

theorem triangle_area_is_12 : 
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    x₁ ≥ 0 ∧ y₁ ≥ 0 ∧ 
    x₂ ≥ 0 ∧ y₂ ≥ 0 ∧ 
    boundingLine x₁ y₁ ∧ 
    boundingLine x₂ y₂ ∧ 
    x₁ ≠ x₂ ∧ 
    triangleArea = (1/2) * x₁ * y₂ := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_is_12_l1374_137445


namespace NUMINAMATH_CALUDE_grade_difference_l1374_137444

theorem grade_difference (x y : ℕ) (h : 3 * y = 4 * x) :
  y - x = 3 ∧ y - x = 4 :=
sorry

end NUMINAMATH_CALUDE_grade_difference_l1374_137444


namespace NUMINAMATH_CALUDE_smallest_proportional_part_l1374_137424

theorem smallest_proportional_part (total : ℕ) (parts : List ℕ) : 
  total = 360 → 
  parts = [5, 7, 4, 8] → 
  List.length parts = 4 → 
  (List.sum parts) ∣ total → 
  (List.minimum parts).isSome → 
  (total / (List.sum parts)) * (List.minimum parts).get! = 60 :=
sorry

end NUMINAMATH_CALUDE_smallest_proportional_part_l1374_137424


namespace NUMINAMATH_CALUDE_no_parallel_solution_perpendicular_solutions_l1374_137454

-- Define the lines
def line1 (m : ℝ) (x y : ℝ) : Prop := (2*m^2 + m - 3)*x + (m^2 - m)*y = 2*m
def line2 (x y : ℝ) : Prop := x - y = 1

def line3 (a : ℝ) (x y : ℝ) : Prop := a*x + (1 - a)*y = 3
def line4 (a : ℝ) (x y : ℝ) : Prop := (a - 1)*x + (2*a + 3)*y = 2

-- Define parallel and perpendicular conditions
def parallel (m : ℝ) : Prop := ∃ k : ℝ, k ≠ 0 ∧ 2*m^2 + m - 3 = k ∧ m^2 - m = -k

def perpendicular (a : ℝ) : Prop := a*(a - 1) + (1 - a)*(2*a + 3) = 0

-- State the theorems
theorem no_parallel_solution : ¬∃ m : ℝ, parallel m := sorry

theorem perpendicular_solutions : ∀ a : ℝ, perpendicular a ↔ (a = 1 ∨ a = -3) := sorry

end NUMINAMATH_CALUDE_no_parallel_solution_perpendicular_solutions_l1374_137454


namespace NUMINAMATH_CALUDE_max_sum_of_squares_l1374_137438

theorem max_sum_of_squares (x y z : ℝ) 
  (h1 : x + y = z - 1) 
  (h2 : x * y = z^2 - 7*z + 14) : 
  (∃ (max : ℝ), ∀ (x' y' z' : ℝ), 
    x' + y' = z' - 1 → 
    x' * y' = z'^2 - 7*z' + 14 → 
    x'^2 + y'^2 ≤ max ∧ 
    (x'^2 + y'^2 = max ↔ z' = 3) ∧ 
    max = 2) :=
sorry

end NUMINAMATH_CALUDE_max_sum_of_squares_l1374_137438


namespace NUMINAMATH_CALUDE_A_equals_set_l1374_137439

def A : Set ℝ :=
  {x | ∃ a b c : ℝ, a * b * c ≠ 0 ∧ 
       x = a / |a| + |b| / b + |c| / c + (a * b * c) / |a * b * c|}

theorem A_equals_set : A = {-4, 0, 4} := by
  sorry

end NUMINAMATH_CALUDE_A_equals_set_l1374_137439


namespace NUMINAMATH_CALUDE_rationalize_denominator_l1374_137463

theorem rationalize_denominator : (14 : ℝ) / Real.sqrt 14 = Real.sqrt 14 := by
  sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l1374_137463


namespace NUMINAMATH_CALUDE_largest_smallest_divisible_by_165_l1374_137401

def is_valid_number (n : ℕ) : Prop :=
  (n ≥ 1000000 ∧ n ≤ 9999999) ∧  -- 7-digit number
  (n % 165 = 0) ∧  -- divisible by 165
  ∀ d : ℕ, d ∈ [0, 1, 2, 3, 4, 5, 6] →
    (∃! i : ℕ, i < 7 ∧ (n / 10^i) % 10 = d)  -- each digit appears exactly once

theorem largest_smallest_divisible_by_165 :
  (∀ n : ℕ, is_valid_number n → n ≤ 6431205) ∧
  (∀ n : ℕ, is_valid_number n → n ≥ 1042635) ∧
  is_valid_number 6431205 ∧
  is_valid_number 1042635 :=
sorry

end NUMINAMATH_CALUDE_largest_smallest_divisible_by_165_l1374_137401


namespace NUMINAMATH_CALUDE_sarah_hair_product_usage_l1374_137408

/-- Calculates the total volume of hair care products used over a given number of days -/
def total_hair_product_usage (shampoo_daily : ℝ) (conditioner_ratio : ℝ) (days : ℕ) : ℝ :=
  let conditioner_daily := shampoo_daily * conditioner_ratio
  let total_daily := shampoo_daily + conditioner_daily
  total_daily * days

/-- Proves that Sarah's total hair product usage over two weeks is 21 ounces -/
theorem sarah_hair_product_usage : 
  total_hair_product_usage 1 0.5 14 = 21 := by
sorry

#eval total_hair_product_usage 1 0.5 14

end NUMINAMATH_CALUDE_sarah_hair_product_usage_l1374_137408


namespace NUMINAMATH_CALUDE_circle_max_area_center_l1374_137472

/-- Given a circle represented by the equation x^2 + y^2 + kx + 2y + k^2 = 0 in the Cartesian 
coordinate system, this theorem states that when the circle has maximum area, its center 
coordinates are (-k/2, -1). -/
theorem circle_max_area_center (k : ℝ) :
  let circle_equation := fun (x y : ℝ) => x^2 + y^2 + k*x + 2*y + k^2 = 0
  let center := (-k/2, -1)
  let is_max_area := ∀ k' : ℝ, 
    (∃ x y, circle_equation x y) → 
    (∃ x' y', x'^2 + y'^2 + k'*x' + 2*y' + k'^2 = 0 ∧ 
              (x' - (-k'/2))^2 + (y' - (-1))^2 ≤ (x - (-k/2))^2 + (y - (-1))^2)
  is_max_area → 
  ∃ x y, circle_equation x y ∧ 
         (x - center.1)^2 + (y - center.2)^2 = 
         (x - (-k/2))^2 + (y - (-1))^2 := by
  sorry

end NUMINAMATH_CALUDE_circle_max_area_center_l1374_137472


namespace NUMINAMATH_CALUDE_mod_equivalence_unique_solution_l1374_137446

theorem mod_equivalence_unique_solution :
  ∃! n : ℤ, 0 ≤ n ∧ n ≤ 8 ∧ n ≡ -4982 [ZMOD 9] ∧ n = 4 := by
  sorry

end NUMINAMATH_CALUDE_mod_equivalence_unique_solution_l1374_137446


namespace NUMINAMATH_CALUDE_some_number_value_l1374_137421

theorem some_number_value : ∃ (x : ℚ), 
  (1 / 2 : ℚ) + ((2 / 3 : ℚ) * (3 / 8 : ℚ) + x) - (8 / 16 : ℚ) = (17 / 4 : ℚ) ∧ x = 4 := by
  sorry

end NUMINAMATH_CALUDE_some_number_value_l1374_137421


namespace NUMINAMATH_CALUDE_special_op_is_addition_l1374_137487

/-- An operation on real numbers satisfying (a * b) * c = a + b + c for all a, b, c -/
def special_op (a b : ℝ) : ℝ := sorry

/-- The property that (a * b) * c = a + b + c for all a, b, c -/
axiom special_op_property (a b c : ℝ) : special_op (special_op a b) c = a + b + c

/-- Theorem: The special operation is equivalent to addition -/
theorem special_op_is_addition (a b : ℝ) : special_op a b = a + b := by
  sorry

end NUMINAMATH_CALUDE_special_op_is_addition_l1374_137487


namespace NUMINAMATH_CALUDE_polynomial_factorization_l1374_137484

theorem polynomial_factorization (x : ℝ) :
  x^8 + x^4 + 1 = (x^2 - Real.sqrt 3 * x + 1) * (x^2 + Real.sqrt 3 * x + 1) * (x^2 - x + 1) * (x^2 + x + 1) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l1374_137484


namespace NUMINAMATH_CALUDE_max_value_of_f_l1374_137449

noncomputable def f (x : ℝ) : ℝ := min (3 * x + 1) (min (-1/3 * x + 2) (x + 4))

theorem max_value_of_f :
  ∃ (M : ℝ), M = 5/2 ∧ ∀ (x : ℝ), f x ≤ M ∧ ∃ (x₀ : ℝ), f x₀ = M :=
sorry

end NUMINAMATH_CALUDE_max_value_of_f_l1374_137449


namespace NUMINAMATH_CALUDE_senior_mean_score_l1374_137451

-- Define the total number of students
def total_students : ℕ := 120

-- Define the overall mean score
def overall_mean : ℝ := 110

-- Define the relationship between number of seniors and juniors
def junior_senior_ratio : ℝ := 0.75

-- Define the relationship between senior and junior mean scores
def senior_junior_mean_ratio : ℝ := 1.4

-- Theorem statement
theorem senior_mean_score :
  ∃ (seniors juniors : ℕ) (senior_mean junior_mean : ℝ),
    seniors + juniors = total_students ∧
    juniors = Int.floor (junior_senior_ratio * seniors) ∧
    senior_mean = senior_junior_mean_ratio * junior_mean ∧
    (seniors * senior_mean + juniors * junior_mean) / total_students = overall_mean ∧
    Int.floor senior_mean = 124 := by
  sorry

end NUMINAMATH_CALUDE_senior_mean_score_l1374_137451


namespace NUMINAMATH_CALUDE_sixth_term_is_64_l1374_137460

/-- A geometric sequence with the given properties -/
structure GeometricSequence where
  a : ℕ → ℝ
  is_geometric : ∀ n, a (n + 1) / a n = a 2 / a 1
  sum_2_4 : a 2 + a 4 = 20
  sum_3_5 : a 3 + a 5 = 40

/-- The sixth term of the geometric sequence is 64 -/
theorem sixth_term_is_64 (seq : GeometricSequence) : seq.a 6 = 64 := by
  sorry

end NUMINAMATH_CALUDE_sixth_term_is_64_l1374_137460


namespace NUMINAMATH_CALUDE_vector_addition_l1374_137435

theorem vector_addition : 
  let v1 : Fin 3 → ℝ := ![4, -9, 2]
  let v2 : Fin 3 → ℝ := ![-3, 8, -5]
  v1 + v2 = ![1, -1, -3] := by
sorry

end NUMINAMATH_CALUDE_vector_addition_l1374_137435


namespace NUMINAMATH_CALUDE_sum_of_cubes_and_fourth_powers_l1374_137411

theorem sum_of_cubes_and_fourth_powers (a b : ℝ) 
  (sum_eq : a + b = 2) 
  (sum_squares_eq : a^2 + b^2 = 2) : 
  a^3 + b^3 = 2 ∧ a^4 + b^4 = 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_cubes_and_fourth_powers_l1374_137411


namespace NUMINAMATH_CALUDE_g_equals_4_at_2_l1374_137450

/-- The function g(x) = 5x - 6 -/
def g (x : ℝ) : ℝ := 5 * x - 6

/-- Theorem: For the function g(x) = 5x - 6, the value of a that satisfies g(a) = 4 is a = 2 -/
theorem g_equals_4_at_2 : g 2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_g_equals_4_at_2_l1374_137450


namespace NUMINAMATH_CALUDE_cylinder_volume_equals_cube_surface_l1374_137441

theorem cylinder_volume_equals_cube_surface (side : ℝ) (h r V : ℝ) : 
  side = 3 → 
  6 * side^2 = 2 * π * r^2 + 2 * π * r * h → 
  h = r → 
  V = π * r^2 * h → 
  V = (81 * Real.sqrt 3 / 2) * Real.sqrt 5 / Real.sqrt π :=
by sorry

end NUMINAMATH_CALUDE_cylinder_volume_equals_cube_surface_l1374_137441


namespace NUMINAMATH_CALUDE_unique_triple_l1374_137476

theorem unique_triple : 
  ∀ a b c : ℝ,
  (a > 0 ∧ b > 0 ∧ c > 0) →
  (a + b + c = 3) →
  (a^2 - a ≥ 1 - b*c) →
  (b^2 - b ≥ 1 - a*c) →
  (c^2 - c ≥ 1 - a*b) →
  (a = 1 ∧ b = 1 ∧ c = 1) :=
by sorry

end NUMINAMATH_CALUDE_unique_triple_l1374_137476


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l1374_137400

theorem complex_fraction_equality (a b : ℝ) : 
  (1 + I : ℂ) / (1 - I) = a + b * I → b = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l1374_137400


namespace NUMINAMATH_CALUDE_consecutive_integers_squares_minus_product_l1374_137468

theorem consecutive_integers_squares_minus_product (n : ℕ) :
  n = 9 → (n^2 + (n+1)^2) - (n * (n+1)) = 91 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_squares_minus_product_l1374_137468


namespace NUMINAMATH_CALUDE_problem_statement_l1374_137480

theorem problem_statement (ℓ : ℝ) (h : (1 + ℓ)^2 / (1 + ℓ^2) = 13/37) :
  (1 + ℓ)^3 / (1 + ℓ^3) = 156/1369 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1374_137480


namespace NUMINAMATH_CALUDE_perpendicular_line_plane_implies_perpendicular_lines_l1374_137479

structure Plane where
  -- Define plane structure

structure Line where
  -- Define line structure

-- Define perpendicularity between a line and a plane
def perpendicular_line_plane (l : Line) (p : Plane) : Prop :=
  sorry

-- Define a line being contained in a plane
def line_in_plane (l : Line) (p : Plane) : Prop :=
  sorry

-- Define perpendicularity between two lines
def perpendicular_lines (l1 l2 : Line) : Prop :=
  sorry

-- Theorem statement
theorem perpendicular_line_plane_implies_perpendicular_lines
  (m n : Line) (α : Plane) 
  (h1 : m ≠ n) 
  (h2 : perpendicular_line_plane m α) 
  (h3 : line_in_plane n α) : 
  perpendicular_lines m n :=
sorry

end NUMINAMATH_CALUDE_perpendicular_line_plane_implies_perpendicular_lines_l1374_137479


namespace NUMINAMATH_CALUDE_onion_to_carrot_ratio_l1374_137475

/-- Represents the number of vegetables Maria wants to cut -/
structure Vegetables where
  potatoes : ℕ
  carrots : ℕ
  onions : ℕ
  green_beans : ℕ

/-- The conditions of Maria's vegetable cutting plan -/
def cutting_plan (v : Vegetables) : Prop :=
  v.carrots = 6 * v.potatoes ∧
  v.onions = v.carrots ∧
  v.green_beans = v.onions / 3 ∧
  v.potatoes = 2 ∧
  v.green_beans = 8

theorem onion_to_carrot_ratio (v : Vegetables) 
  (h : cutting_plan v) : v.onions = v.carrots := by
  sorry

#check onion_to_carrot_ratio

end NUMINAMATH_CALUDE_onion_to_carrot_ratio_l1374_137475


namespace NUMINAMATH_CALUDE_pyramid_lego_count_l1374_137494

/-- Calculates the number of legos for a square level -/
def square_level (side : ℕ) : ℕ := side * side

/-- Calculates the number of legos for a rectangular level -/
def rectangular_level (length width : ℕ) : ℕ := length * width

/-- Calculates the number of legos for a triangular level -/
def triangular_level (side : ℕ) : ℕ := side * (side + 1) / 2 - 3

/-- Calculates the total number of legos for the pyramid -/
def total_legos : ℕ :=
  square_level 10 + rectangular_level 8 6 + triangular_level 4 + 1

theorem pyramid_lego_count : total_legos = 156 := by
  sorry

end NUMINAMATH_CALUDE_pyramid_lego_count_l1374_137494


namespace NUMINAMATH_CALUDE_hyperbola_equation_l1374_137491

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos_a : a > 0
  h_pos_b : b > 0

/-- Represents a line with slope m and y-intercept c -/
structure Line where
  m : ℝ
  c : ℝ

/-- States that a line passes through a focus of the hyperbola -/
def passes_through_focus (l : Line) (h : Hyperbola) : Prop :=
  ∃ x y, y = l.m * x + l.c ∧ x^2 + y^2 = h.a^2 + h.b^2

/-- States that a line is parallel to an asymptote of the hyperbola -/
def parallel_to_asymptote (l : Line) (h : Hyperbola) : Prop :=
  l.m = h.b / h.a ∨ l.m = -h.b / h.a

theorem hyperbola_equation (h : Hyperbola) (l : Line) 
  (h_focus : passes_through_focus l h)
  (h_parallel : parallel_to_asymptote l h)
  (h_line : l.m = 2 ∧ l.c = 10) :
  h.a^2 = 5 ∧ h.b^2 = 20 := by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l1374_137491
