import Mathlib

namespace NUMINAMATH_CALUDE_polynomial_expansion_l3966_396635

theorem polynomial_expansion (t : ℝ) : 
  (2*t^2 - 3*t + 2) * (-3*t^2 + t - 5) = -6*t^4 + 11*t^3 - 19*t^2 + 17*t - 10 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_expansion_l3966_396635


namespace NUMINAMATH_CALUDE_fraction_of_As_l3966_396689

theorem fraction_of_As (total_students : ℕ) (fraction_Bs fraction_Cs : ℚ) (num_Ds : ℕ) :
  total_students = 100 →
  fraction_Bs = 1/4 →
  fraction_Cs = 1/2 →
  num_Ds = 5 →
  (total_students - (fraction_Bs * total_students + fraction_Cs * total_students + num_Ds)) / total_students = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_of_As_l3966_396689


namespace NUMINAMATH_CALUDE_B_coords_when_A_on_y_axis_a_value_when_AB_parallel_x_axis_l3966_396698

-- Define points A and B in the Cartesian coordinate system
def A (a : ℝ) : ℝ × ℝ := (a + 1, -3)
def B (a : ℝ) : ℝ × ℝ := (3, 2 * a + 1)

-- Theorem 1: When A lies on the y-axis, B has coordinates (3, -1)
theorem B_coords_when_A_on_y_axis (a : ℝ) :
  A a = (0, -3) → B a = (3, -1) := by sorry

-- Theorem 2: When AB is parallel to x-axis, a = -2
theorem a_value_when_AB_parallel_x_axis (a : ℝ) :
  (A a).2 = (B a).2 → a = -2 := by sorry

end NUMINAMATH_CALUDE_B_coords_when_A_on_y_axis_a_value_when_AB_parallel_x_axis_l3966_396698


namespace NUMINAMATH_CALUDE_no_descending_nat_function_exists_descending_int_function_l3966_396671

-- Define φ as a function from ℕ to ℕ
variable (φ : ℕ → ℕ)

-- Theorem 1: No such function exists when the range is ℕ
theorem no_descending_nat_function :
  ¬ ∃ f : ℕ → ℕ, ∀ x : ℕ, f x > f (φ x) :=
sorry

-- Theorem 2: Such a function exists when the range is ℤ
theorem exists_descending_int_function :
  ∃ f : ℕ → ℤ, ∀ x : ℕ, f x > f (φ x) :=
sorry

end NUMINAMATH_CALUDE_no_descending_nat_function_exists_descending_int_function_l3966_396671


namespace NUMINAMATH_CALUDE_discount_effect_l3966_396651

/-- Represents the sales discount as a percentage -/
def discount : ℝ := 10

/-- Represents the increase in the number of items sold as a percentage -/
def items_increase : ℝ := 30

/-- Represents the increase in gross income as a percentage -/
def income_increase : ℝ := 17

theorem discount_effect (P N : ℝ) (h₁ : P > 0) (h₂ : N > 0) : 
  P * (1 - discount / 100) * N * (1 + items_increase / 100) = 
  P * N * (1 + income_increase / 100) := by
  sorry

#check discount_effect

end NUMINAMATH_CALUDE_discount_effect_l3966_396651


namespace NUMINAMATH_CALUDE_anne_drawings_per_marker_l3966_396690

/-- Given:
  * Anne has 12 markers
  * She has already made 8 drawings
  * She can make 10 more drawings before running out of markers
  Prove that Anne can make 1.5 drawings with one marker -/
theorem anne_drawings_per_marker (markers : ℕ) (made_drawings : ℕ) (remaining_drawings : ℕ) 
  (h1 : markers = 12)
  (h2 : made_drawings = 8)
  (h3 : remaining_drawings = 10) :
  (made_drawings + remaining_drawings : ℚ) / markers = 1.5 := by
  sorry

end NUMINAMATH_CALUDE_anne_drawings_per_marker_l3966_396690


namespace NUMINAMATH_CALUDE_parallelogram_height_l3966_396644

-- Define the parallelogram
def parallelogram_area : ℝ := 72
def parallelogram_base : ℝ := 12

-- Theorem to prove
theorem parallelogram_height :
  parallelogram_area / parallelogram_base = 6 :=
by
  sorry


end NUMINAMATH_CALUDE_parallelogram_height_l3966_396644


namespace NUMINAMATH_CALUDE_smallest_product_l3966_396623

def S : Set Int := {-10, -5, 0, 2, 4}

theorem smallest_product (a b : Int) (ha : a ∈ S) (hb : b ∈ S) :
  ∃ (x y : Int), x ∈ S ∧ y ∈ S ∧ x * y ≤ a * b ∧ x * y = -40 :=
sorry

end NUMINAMATH_CALUDE_smallest_product_l3966_396623


namespace NUMINAMATH_CALUDE_same_heads_probability_l3966_396634

/-- The number of pennies Keiko tosses -/
def keiko_pennies : ℕ := 2

/-- The number of pennies Ephraim tosses -/
def ephraim_pennies : ℕ := 3

/-- The total number of possible outcomes when tossing n pennies -/
def total_outcomes (n : ℕ) : ℕ := 2^n

/-- The number of ways to get k heads when tossing n pennies -/
def ways_to_get_heads (n k : ℕ) : ℕ := Nat.choose n k

/-- The number of favorable outcomes where Keiko and Ephraim get the same number of heads -/
def favorable_outcomes : ℕ :=
  (ways_to_get_heads keiko_pennies 0 * ways_to_get_heads ephraim_pennies 0) +
  (ways_to_get_heads keiko_pennies 1 * ways_to_get_heads ephraim_pennies 1) +
  (ways_to_get_heads keiko_pennies 2 * ways_to_get_heads ephraim_pennies 2)

/-- The probability of Ephraim getting the same number of heads as Keiko -/
theorem same_heads_probability :
  (favorable_outcomes : ℚ) / (total_outcomes keiko_pennies * total_outcomes ephraim_pennies) = 5/16 := by
  sorry

end NUMINAMATH_CALUDE_same_heads_probability_l3966_396634


namespace NUMINAMATH_CALUDE_dad_steps_l3966_396629

/-- Represents the number of steps taken by each person -/
structure Steps where
  dad : ℕ
  masha : ℕ
  yasha : ℕ

/-- The ratio of steps between Dad and Masha -/
def dad_masha_ratio (s : Steps) : Prop :=
  3 * s.masha = 5 * s.dad

/-- The ratio of steps between Masha and Yasha -/
def masha_yasha_ratio (s : Steps) : Prop :=
  3 * s.yasha = 5 * s.masha

/-- The total number of steps taken by Masha and Yasha -/
def masha_yasha_total (s : Steps) : Prop :=
  s.masha + s.yasha = 400

theorem dad_steps (s : Steps) 
  (h1 : dad_masha_ratio s)
  (h2 : masha_yasha_ratio s)
  (h3 : masha_yasha_total s) :
  s.dad = 90 := by
  sorry

end NUMINAMATH_CALUDE_dad_steps_l3966_396629


namespace NUMINAMATH_CALUDE_quadratic_equation_solutions_l3966_396645

theorem quadratic_equation_solutions (a b m : ℝ) (h : a ≠ 0) :
  (∃ x₁ x₂ : ℝ, x₁ = 2 ∧ x₂ = -1 ∧ ∀ x : ℝ, a * (x + m)^2 + b = 0 ↔ x = x₁ ∨ x = x₂) →
  (∃ y₁ y₂ : ℝ, y₁ = -3 ∧ y₂ = 0 ∧ ∀ x : ℝ, a * (x - m + 2)^2 + b = 0 ↔ x = y₁ ∨ x = y₂) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solutions_l3966_396645


namespace NUMINAMATH_CALUDE_consecutive_page_numbers_l3966_396606

theorem consecutive_page_numbers (n : ℕ) : 
  n > 0 ∧ n * (n + 1) = 20460 → n + (n + 1) = 285 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_page_numbers_l3966_396606


namespace NUMINAMATH_CALUDE_total_buttons_is_117_l3966_396613

/-- The number of buttons Jack uses for all shirts -/
def total_buttons : ℕ :=
  let jack_kids : ℕ := 3
  let jack_shirts_per_kid : ℕ := 3
  let jack_buttons_per_shirt : ℕ := 7
  let neighbor_kids : ℕ := 2
  let neighbor_shirts_per_kid : ℕ := 3
  let neighbor_buttons_per_shirt : ℕ := 9
  
  let jack_total_shirts := jack_kids * jack_shirts_per_kid
  let jack_total_buttons := jack_total_shirts * jack_buttons_per_shirt
  
  let neighbor_total_shirts := neighbor_kids * neighbor_shirts_per_kid
  let neighbor_total_buttons := neighbor_total_shirts * neighbor_buttons_per_shirt
  
  jack_total_buttons + neighbor_total_buttons

/-- Theorem stating that the total number of buttons Jack uses is 117 -/
theorem total_buttons_is_117 : total_buttons = 117 := by
  sorry

end NUMINAMATH_CALUDE_total_buttons_is_117_l3966_396613


namespace NUMINAMATH_CALUDE_line_parallel_to_plane_parallel_to_line_set_l3966_396675

-- Define the necessary structures
structure Line3D where
  -- Add necessary fields for a 3D line

structure Plane3D where
  -- Add necessary fields for a 3D plane

-- Define parallelism between a line and a plane
def line_parallel_to_plane (l : Line3D) (p : Plane3D) : Prop :=
  sorry

-- Define a set of parallel lines within a plane
def parallel_lines_in_plane (p : Plane3D) : Set Line3D :=
  sorry

-- Define parallelism between two lines
def lines_parallel (l1 l2 : Line3D) : Prop :=
  sorry

-- The theorem to be proved
theorem line_parallel_to_plane_parallel_to_line_set 
  (a : Line3D) (α : Plane3D) 
  (h : line_parallel_to_plane a α) :
  ∃ (S : Set Line3D), S ⊆ parallel_lines_in_plane α ∧ 
    ∀ l ∈ S, lines_parallel a l :=
  sorry

end NUMINAMATH_CALUDE_line_parallel_to_plane_parallel_to_line_set_l3966_396675


namespace NUMINAMATH_CALUDE_sequence_growth_l3966_396616

theorem sequence_growth (a : ℕ → ℤ) 
  (h1 : a 1 > a 0) 
  (h2 : a 1 > 0) 
  (h3 : ∀ r : ℕ, r ≤ 98 → a (r + 2) = 3 * a (r + 1) - 2 * a r) : 
  a 100 > 2^99 := by
  sorry

end NUMINAMATH_CALUDE_sequence_growth_l3966_396616


namespace NUMINAMATH_CALUDE_chess_tournament_players_l3966_396639

theorem chess_tournament_players (total_games : ℕ) (h : total_games = 56) :
  ∃ (n : ℕ), n > 0 ∧ total_games = n * (n - 1) ∧ n = 14 := by
  sorry

end NUMINAMATH_CALUDE_chess_tournament_players_l3966_396639


namespace NUMINAMATH_CALUDE_expression_range_l3966_396660

theorem expression_range (a b c : ℝ) 
  (h1 : a - b + c = 0)
  (h2 : c > 0)
  (h3 : 3 * a - 2 * b + c > 0) :
  4/3 < (a + 3*b + 7*c) / (2*a + b) ∧ (a + 3*b + 7*c) / (2*a + b) < 7/2 :=
sorry

end NUMINAMATH_CALUDE_expression_range_l3966_396660


namespace NUMINAMATH_CALUDE_parabola_chord_length_l3966_396668

/-- The length of a chord passing through the focus of a parabola -/
theorem parabola_chord_length (x₁ x₂ y₁ y₂ : ℝ) : 
  y₁^2 = 4*x₁ →  -- Point A satisfies the parabola equation
  y₂^2 = 4*x₂ →  -- Point B satisfies the parabola equation
  x₁ + x₂ = 6 →  -- Given condition
  -- The line passes through the focus (1, 0) of y^2 = 4x
  ∃ (m : ℝ), y₁ = m*(x₁ - 1) ∧ y₂ = m*(x₂ - 1) →
  -- Then the length of chord AB is 8
  ((x₁ - x₂)^2 + (y₁ - y₂)^2)^(1/2 : ℝ) = 8 :=
by sorry

end NUMINAMATH_CALUDE_parabola_chord_length_l3966_396668


namespace NUMINAMATH_CALUDE_tan_seven_pi_sixths_l3966_396643

theorem tan_seven_pi_sixths : Real.tan (7 * Real.pi / 6) = 1 / Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_seven_pi_sixths_l3966_396643


namespace NUMINAMATH_CALUDE_triangle_theorem_l3966_396695

/-- Triangle ABC with sides a, b, c corresponding to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : Real
  B : Real
  C : Real

/-- The given condition a^2 + c^2 = b^2 - ac -/
def triangleCondition (t : Triangle) : Prop :=
  t.a^2 + t.c^2 = t.b^2 - t.a * t.c

/-- The angle bisector condition -/
def angleBisectorCondition (t : Triangle) (AD BD : ℝ) : Prop :=
  AD = 2 * Real.sqrt 3 ∧ BD = 1

theorem triangle_theorem (t : Triangle) 
  (h1 : triangleCondition t)
  (h2 : angleBisectorCondition t (2 * Real.sqrt 3) 1) :
  t.B = 2 * Real.pi / 3 ∧ Real.sin t.A = Real.sqrt 15 / 8 := by
  sorry

end NUMINAMATH_CALUDE_triangle_theorem_l3966_396695


namespace NUMINAMATH_CALUDE_principal_amount_calculation_l3966_396632

/-- Calculate the principal amount given the difference between compound and simple interest -/
theorem principal_amount_calculation (interest_rate : ℝ) (compounding_frequency : ℕ) (time : ℝ) (interest_difference : ℝ) :
  interest_rate = 0.10 →
  compounding_frequency = 2 →
  time = 1 →
  interest_difference = 3.9999999999999147 →
  ∃ (principal : ℝ), 
    (principal * ((1 + interest_rate / compounding_frequency) ^ (compounding_frequency * time) - 1) - 
     principal * interest_rate * time = interest_difference) ∧
    (abs (principal - 1600) < 1) :=
by sorry

end NUMINAMATH_CALUDE_principal_amount_calculation_l3966_396632


namespace NUMINAMATH_CALUDE_total_mangoes_l3966_396685

/-- The number of mangoes each person has -/
structure MangoDistribution where
  alexis : ℝ
  dilan : ℝ
  ashley : ℝ
  ben : ℝ

/-- The conditions of the mango distribution problem -/
def mango_conditions (m : MangoDistribution) : Prop :=
  m.alexis = 4 * (m.dilan + m.ashley) ∧
  m.ashley = 2 * m.dilan ∧
  m.alexis = 60 ∧
  m.ben = (m.ashley + m.dilan) / 2

/-- The theorem stating the total number of mangoes -/
theorem total_mangoes (m : MangoDistribution) 
  (h : mango_conditions m) : 
  m.alexis + m.dilan + m.ashley + m.ben = 82.5 :=
by sorry

end NUMINAMATH_CALUDE_total_mangoes_l3966_396685


namespace NUMINAMATH_CALUDE_f_g_derivatives_neg_l3966_396628

-- Define the functions f and g
variable (f g : ℝ → ℝ)

-- Define the conditions
axiom f_odd : ∀ x : ℝ, f (-x) = -f x
axiom g_even : ∀ x : ℝ, g (-x) = g x
axiom f_deriv_pos : ∀ x : ℝ, x > 0 → deriv f x > 0
axiom g_deriv_neg_pos : ∀ x : ℝ, x > 0 → deriv g (-x) > 0

-- State the theorem
theorem f_g_derivatives_neg (x : ℝ) (h : x < 0) :
  deriv f x > 0 ∧ deriv g (-x) < 0 := by sorry

end NUMINAMATH_CALUDE_f_g_derivatives_neg_l3966_396628


namespace NUMINAMATH_CALUDE_boat_travel_difference_l3966_396670

/-- Represents the difference in distance traveled downstream vs upstream for a boat -/
def boat_distance_difference (a b : ℝ) : ℝ :=
  let downstream_speed := a + b
  let upstream_speed := a - b
  let downstream_distance := 3 * downstream_speed
  let upstream_distance := 2 * upstream_speed
  downstream_distance - upstream_distance

/-- Theorem stating the difference in distance traveled by the boat -/
theorem boat_travel_difference (a b : ℝ) (h : a > b) :
  boat_distance_difference a b = a + 5*b := by
  sorry

#check boat_travel_difference

end NUMINAMATH_CALUDE_boat_travel_difference_l3966_396670


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_find_m_value_l3966_396610

-- Part 1
theorem simplify_and_evaluate (a b : ℚ) (h1 : a = 1/2) (h2 : b = -2) :
  2 * (3 * a^2 * b - a * b^2) - 3 * (2 * a^2 * b - a * b^2 + a * b) = 5 := by sorry

-- Part 2
theorem find_m_value (a b m : ℚ) :
  (a^2 + 2*a*b - b^2) - (a^2 + m*a*b + 2*b^2) = -3*b^2 → m = 2 := by sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_find_m_value_l3966_396610


namespace NUMINAMATH_CALUDE_parallelepiped_dimensions_l3966_396615

theorem parallelepiped_dimensions (n : ℕ) : 
  n > 6 →
  (n - 2) * (n - 4) * (n - 6) = (2 / 3) * n * (n - 2) * (n - 4) →
  n = 18 :=
by sorry

end NUMINAMATH_CALUDE_parallelepiped_dimensions_l3966_396615


namespace NUMINAMATH_CALUDE_log_8_40_l3966_396620

theorem log_8_40 (a c : ℝ) (h1 : Real.log 2 / Real.log 10 = a) (h2 : Real.log 5 / Real.log 10 = c) :
  Real.log 40 / Real.log 8 = 1 + c / (3 * a) := by
  sorry

end NUMINAMATH_CALUDE_log_8_40_l3966_396620


namespace NUMINAMATH_CALUDE_raise_doubles_earnings_l3966_396602

/-- Calculates the new weekly earnings after a percentage raise -/
def new_earnings (initial_earnings : ℕ) (percentage_raise : ℕ) : ℕ :=
  initial_earnings + initial_earnings * percentage_raise / 100

/-- Proves that a 100% raise on $40 results in $80 weekly earnings -/
theorem raise_doubles_earnings : new_earnings 40 100 = 80 := by
  sorry

end NUMINAMATH_CALUDE_raise_doubles_earnings_l3966_396602


namespace NUMINAMATH_CALUDE_function_value_problem_l3966_396693

theorem function_value_problem (f : ℝ → ℝ) 
  (h : ∀ x, 2 * f x + f (-x) = 3 * x + 2) : 
  f 2 = 20 / 3 := by
sorry

end NUMINAMATH_CALUDE_function_value_problem_l3966_396693


namespace NUMINAMATH_CALUDE_race_winner_race_result_l3966_396604

theorem race_winner (race_length : ℝ) (speed_ratio_A : ℝ) (speed_ratio_B : ℝ) (head_start : ℝ) : ℝ :=
  let time_B_finish := race_length / speed_ratio_B
  let distance_A := speed_ratio_A * time_B_finish + head_start
  distance_A - race_length

theorem race_result :
  race_winner 500 3 4 140 = 15 := by sorry

end NUMINAMATH_CALUDE_race_winner_race_result_l3966_396604


namespace NUMINAMATH_CALUDE_club_members_after_four_years_l3966_396664

def club_members (n : ℕ) : ℕ :=
  match n with
  | 0 => 20
  | k + 1 => 3 * club_members k - 10

theorem club_members_after_four_years :
  club_members 4 = 1220 := by
  sorry

end NUMINAMATH_CALUDE_club_members_after_four_years_l3966_396664


namespace NUMINAMATH_CALUDE_independence_test_confidence_l3966_396617

/-- The critical value for the independence test -/
def critical_value : ℝ := 5.024

/-- The confidence level for "X and Y are related" given k > critical_value -/
def confidence_level (k : ℝ) : ℝ := 97.5

/-- Theorem stating that when k > critical_value, the confidence level is 97.5% -/
theorem independence_test_confidence (k : ℝ) (h : k > critical_value) :
  confidence_level k = 97.5 := by sorry

end NUMINAMATH_CALUDE_independence_test_confidence_l3966_396617


namespace NUMINAMATH_CALUDE_addition_puzzle_solution_l3966_396678

/-- A digit is a natural number from 0 to 9 -/
def Digit : Type := { n : ℕ // n ≤ 9 }

/-- Function to convert a four-digit number to its decimal representation -/
def toDecimal (a b c d : Digit) : ℕ := 1000 * a.val + 100 * b.val + 10 * c.val + d.val

/-- Predicate to check if three digits are distinct -/
def areDistinct (a b c : Digit) : Prop := a ≠ b ∧ b ≠ c ∧ a ≠ c

theorem addition_puzzle_solution :
  ∃ (possibleD : Finset Digit),
    (∀ a b c d : Digit,
      areDistinct a b c →
      toDecimal a b a c + toDecimal c a b a = toDecimal d c d d →
      d ∈ possibleD) ∧
    possibleD.card = 7 := by sorry

end NUMINAMATH_CALUDE_addition_puzzle_solution_l3966_396678


namespace NUMINAMATH_CALUDE_same_solution_k_value_l3966_396605

theorem same_solution_k_value (x : ℝ) (k : ℝ) : 
  (2 * x = 4) ∧ (3 * x + k = -2) → k = -8 := by
  sorry

end NUMINAMATH_CALUDE_same_solution_k_value_l3966_396605


namespace NUMINAMATH_CALUDE_laura_debt_after_one_year_l3966_396680

/-- Calculates the total amount owed after applying simple interest -/
def totalAmountOwed (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * (1 + rate * time)

/-- Theorem: Laura's debt after one year -/
theorem laura_debt_after_one_year :
  totalAmountOwed 35 0.05 1 = 36.75 := by
  sorry

end NUMINAMATH_CALUDE_laura_debt_after_one_year_l3966_396680


namespace NUMINAMATH_CALUDE_hat_price_theorem_l3966_396649

theorem hat_price_theorem (final_price : ℚ) 
  (h1 : final_price = 8)
  (h2 : ∃ original_price : ℚ, 
    final_price = original_price * (1/5) * (1 + 1/5)) : 
  ∃ original_price : ℚ, original_price = 100/3 ∧ 
    final_price = original_price * (1/5) * (1 + 1/5) := by
  sorry

end NUMINAMATH_CALUDE_hat_price_theorem_l3966_396649


namespace NUMINAMATH_CALUDE_expression_simplification_l3966_396687

theorem expression_simplification 
  (a b c d : ℝ) 
  (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) :
  ∃ k : ℝ, ∀ x : ℝ, 
    (x + a)^4 / ((a - b) * (a - c) * (a - d)) +
    (x + b)^4 / ((b - a) * (b - c) * (b - d)) +
    (x + c)^4 / ((c - a) * (c - b) * (c - d)) +
    (x + d)^4 / ((d - a) * (d - b) * (d - c)) =
    k * (x + a) * (x + b) * (x + c) * (x + d) := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3966_396687


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_l3966_396637

theorem necessary_but_not_sufficient :
  (∀ x : ℝ, x * (x - 3) < 0 → |x - 1| < 2) ∧
  (∃ x : ℝ, |x - 1| < 2 ∧ x * (x - 3) ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_l3966_396637


namespace NUMINAMATH_CALUDE_sales_growth_rate_is_ten_percent_l3966_396658

/-- The average monthly growth rate of total sales from February to April -/
def average_monthly_growth_rate : ℝ := 0.1

/-- Total sales in February (in yuan) -/
def february_sales : ℝ := 240000

/-- Total sales in April (in yuan) -/
def april_sales : ℝ := 290400

/-- Number of months between February and April -/
def months_between : ℕ := 2

theorem sales_growth_rate_is_ten_percent :
  april_sales = february_sales * (1 + average_monthly_growth_rate) ^ months_between := by
  sorry

end NUMINAMATH_CALUDE_sales_growth_rate_is_ten_percent_l3966_396658


namespace NUMINAMATH_CALUDE_jackie_working_hours_l3966_396681

/-- Represents the number of hours in a day -/
def hours_in_day : ℕ := 24

/-- Represents the number of hours Jackie spends exercising -/
def exercise_hours : ℕ := 3

/-- Represents the number of hours Jackie spends sleeping -/
def sleep_hours : ℕ := 8

/-- Represents the number of hours Jackie has as free time -/
def free_time_hours : ℕ := 5

/-- Calculates the number of hours Jackie spends working -/
def working_hours : ℕ := hours_in_day - (sleep_hours + exercise_hours + free_time_hours)

theorem jackie_working_hours :
  working_hours = 8 := by sorry

end NUMINAMATH_CALUDE_jackie_working_hours_l3966_396681


namespace NUMINAMATH_CALUDE_impossibleGrid_l3966_396641

/-- Represents a 6x6 grid filled with numbers from 1 to 6 -/
def Grid := Fin 6 → Fin 6 → Fin 6

/-- The sum of numbers in a 2x2 subgrid starting at (i, j) -/
def subgridSum (g : Grid) (i j : Fin 5) : ℕ :=
  g i j + g i (j + 1) + g (i + 1) j + g (i + 1) (j + 1)

/-- A predicate that checks if all 2x2 subgrids have different sums -/
def allSubgridSumsDifferent (g : Grid) : Prop :=
  ∀ i j k l : Fin 5, (i, j) ≠ (k, l) → subgridSum g i j ≠ subgridSum g k l

theorem impossibleGrid : ¬ ∃ g : Grid, allSubgridSumsDifferent g := by
  sorry

end NUMINAMATH_CALUDE_impossibleGrid_l3966_396641


namespace NUMINAMATH_CALUDE_two_car_problem_l3966_396694

/-- Proves that given the conditions of the two-car problem, the speeds of cars A and B are 30 km/h and 25 km/h respectively. -/
theorem two_car_problem (distance_A distance_B : ℝ) (speed_difference : ℝ) 
  (h1 : distance_A = 300)
  (h2 : distance_B = 250)
  (h3 : speed_difference = 5)
  (h4 : ∃ (t : ℝ), t > 0 ∧ distance_A / (speed_B + speed_difference) = t ∧ distance_B / speed_B = t) :
  ∃ (speed_A speed_B : ℝ), 
    speed_A = 30 ∧ 
    speed_B = 25 ∧ 
    speed_A = speed_B + speed_difference ∧
    distance_A / speed_A = distance_B / speed_B :=
by
  sorry


end NUMINAMATH_CALUDE_two_car_problem_l3966_396694


namespace NUMINAMATH_CALUDE_sum_proper_divisors_540_l3966_396673

theorem sum_proper_divisors_540 : 
  (Finset.filter (λ x => x < 540 ∧ 540 % x = 0) (Finset.range 540)).sum id = 1140 := by
  sorry

end NUMINAMATH_CALUDE_sum_proper_divisors_540_l3966_396673


namespace NUMINAMATH_CALUDE_greatest_integer_b_for_quadratic_range_l3966_396691

theorem greatest_integer_b_for_quadratic_range : 
  ∃ (b : ℤ), b = 9 ∧ 
  (∀ x : ℝ, x^2 + b*x + 20 ≠ -4) ∧
  (∀ c : ℤ, c > b → ∃ x : ℝ, x^2 + c*x + 20 = -4) := by
  sorry

end NUMINAMATH_CALUDE_greatest_integer_b_for_quadratic_range_l3966_396691


namespace NUMINAMATH_CALUDE_playlist_repetitions_l3966_396653

def song1_duration : ℕ := 3
def song2_duration : ℕ := 2
def song3_duration : ℕ := 3
def ride_duration : ℕ := 40

def playlist_duration : ℕ := song1_duration + song2_duration + song3_duration

theorem playlist_repetitions :
  ride_duration / playlist_duration = 5 := by sorry

end NUMINAMATH_CALUDE_playlist_repetitions_l3966_396653


namespace NUMINAMATH_CALUDE_existence_of_function_with_properties_l3966_396655

theorem existence_of_function_with_properties : ∃ f : ℝ → ℝ, 
  (∀ x : ℝ, f (1 + x) = f (1 - x)) ∧ 
  (∀ x y : ℝ, x ≤ y ∧ y ≤ 0 → f y ≤ f x) ∧ 
  (∃ z : ℝ, f z < f 0) ∧
  (let g : ℝ → ℝ := fun x ↦ (x - 1)^2;
   (∀ x : ℝ, g (1 + x) = g (1 - x)) ∧ 
   (∀ x y : ℝ, x ≤ y ∧ y ≤ 0 → g y ≤ g x) ∧ 
   (∃ z : ℝ, g z < g 0)) :=
by sorry

end NUMINAMATH_CALUDE_existence_of_function_with_properties_l3966_396655


namespace NUMINAMATH_CALUDE_cuboid_third_edge_length_l3966_396674

/-- Given a cuboid with two known edge lengths and its volume, 
    calculate the length of the third edge. -/
theorem cuboid_third_edge_length 
  (edge1 : ℝ) (edge2 : ℝ) (volume : ℝ) (third_edge : ℝ) :
  edge1 = 2 →
  edge2 = 5 →
  volume = 30 →
  volume = edge1 * edge2 * third_edge →
  third_edge = 3 := by
  sorry

end NUMINAMATH_CALUDE_cuboid_third_edge_length_l3966_396674


namespace NUMINAMATH_CALUDE_product_mod_seventeen_l3966_396679

theorem product_mod_seventeen : (1520 * 1521 * 1522) % 17 = 11 := by
  sorry

end NUMINAMATH_CALUDE_product_mod_seventeen_l3966_396679


namespace NUMINAMATH_CALUDE_smallest_difference_36k_5m_l3966_396609

theorem smallest_difference_36k_5m :
  (∀ k m : ℕ+, 36^k.val - 5^m.val ≥ 11) ∧
  (∃ k m : ℕ+, 36^k.val - 5^m.val = 11) :=
by sorry

end NUMINAMATH_CALUDE_smallest_difference_36k_5m_l3966_396609


namespace NUMINAMATH_CALUDE_min_of_quadratic_l3966_396608

/-- The quadratic function f(x) = x^2 + px + 2q -/
def f (p q : ℝ) (x : ℝ) : ℝ := x^2 + p*x + 2*q

/-- Theorem stating that the minimum of f occurs at x = -p/2 -/
theorem min_of_quadratic (p q : ℝ) (hp : p > 0) (hq : q > 0) :
  ∀ x : ℝ, f p q (-p/2) ≤ f p q x :=
sorry

end NUMINAMATH_CALUDE_min_of_quadratic_l3966_396608


namespace NUMINAMATH_CALUDE_vovochka_candy_theorem_l3966_396646

/-- Given a total number of candies and classmates, calculates the maximum number
    of candies Vovochka can keep while satisfying the distribution condition. -/
def max_candies_for_vovochka (total_candies : ℕ) (num_classmates : ℕ) : ℕ :=
  total_candies - (num_classmates - 1) * 7 + 4

/-- Checks if the candy distribution satisfies the condition that
    any 16 classmates have at least 100 candies. -/
def satisfies_condition (candies_kept : ℕ) (total_candies : ℕ) (num_classmates : ℕ) : Prop :=
  ∀ (group : Finset (Fin num_classmates)),
    group.card = 16 →
    (total_candies - candies_kept) * 16 / num_classmates ≥ 100

theorem vovochka_candy_theorem (total_candies num_classmates : ℕ)
    (h1 : total_candies = 200)
    (h2 : num_classmates = 25) :
    let max_candies := max_candies_for_vovochka total_candies num_classmates
    satisfies_condition max_candies total_candies num_classmates ∧
    ∀ k, k > max_candies →
      ¬satisfies_condition k total_candies num_classmates :=
  sorry

end NUMINAMATH_CALUDE_vovochka_candy_theorem_l3966_396646


namespace NUMINAMATH_CALUDE_solution_exists_l3966_396642

theorem solution_exists : ∃ x : ℝ, 0.05 < x ∧ x < 0.051 :=
by
  use 0.0505
  sorry

#check solution_exists

end NUMINAMATH_CALUDE_solution_exists_l3966_396642


namespace NUMINAMATH_CALUDE_divisible_by_eight_l3966_396618

theorem divisible_by_eight (n : ℕ) : ∃ m : ℤ, 3^(4*n+1) + 5^(2*n+1) = 8*m := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_eight_l3966_396618


namespace NUMINAMATH_CALUDE_john_needs_thirteen_more_l3966_396640

def saturday_earnings : ℕ := 18
def sunday_earnings : ℕ := saturday_earnings / 2
def previous_weekend_earnings : ℕ := 20
def pogo_stick_cost : ℕ := 60

theorem john_needs_thirteen_more : 
  pogo_stick_cost - (saturday_earnings + sunday_earnings + previous_weekend_earnings) = 13 := by
  sorry

end NUMINAMATH_CALUDE_john_needs_thirteen_more_l3966_396640


namespace NUMINAMATH_CALUDE_problem_solution_l3966_396630

theorem problem_solution (a b : ℚ) 
  (h1 : 2 * a + 3 = 5 - b) 
  (h2 : 5 + 2 * b = 10 + a) : 
  2 - a = 11 / 5 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l3966_396630


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l3966_396654

theorem absolute_value_inequality (x : ℝ) : 
  |x - 1| + |x + 2| ≥ 5 ↔ x ≤ -3 ∨ x ≥ 2 := by
sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l3966_396654


namespace NUMINAMATH_CALUDE_complex_modulus_product_l3966_396662

theorem complex_modulus_product : Complex.abs (5 - 3 * Complex.I) * Complex.abs (5 + 3 * Complex.I) = 34 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_product_l3966_396662


namespace NUMINAMATH_CALUDE_factorization_proof_l3966_396625

theorem factorization_proof (x : ℝ) : (x^2 + 4)^2 - 16*x^2 = (x + 2)^2 * (x - 2)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_proof_l3966_396625


namespace NUMINAMATH_CALUDE_algebraic_expression_equality_l3966_396665

theorem algebraic_expression_equality (x y : ℝ) (h : x - 2*y = -2) : 
  9 - 2*x + 4*y = 13 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_equality_l3966_396665


namespace NUMINAMATH_CALUDE_red_boxcars_count_l3966_396656

/-- The number of blue boxcars -/
def num_blue_boxcars : ℕ := 4

/-- The number of black boxcars -/
def num_black_boxcars : ℕ := 7

/-- The capacity of a black boxcar in pounds -/
def black_boxcar_capacity : ℕ := 4000

/-- The capacity of a blue boxcar in pounds -/
def blue_boxcar_capacity : ℕ := 2 * black_boxcar_capacity

/-- The capacity of a red boxcar in pounds -/
def red_boxcar_capacity : ℕ := 3 * blue_boxcar_capacity

/-- The total capacity of all boxcars in pounds -/
def total_capacity : ℕ := 132000

/-- The number of red boxcars -/
def num_red_boxcars : ℕ := 
  (total_capacity - num_black_boxcars * black_boxcar_capacity - num_blue_boxcars * blue_boxcar_capacity) / red_boxcar_capacity

theorem red_boxcars_count : num_red_boxcars = 3 := by
  sorry

end NUMINAMATH_CALUDE_red_boxcars_count_l3966_396656


namespace NUMINAMATH_CALUDE_florist_fertilizer_usage_l3966_396676

/-- A florist's fertilizer usage problem -/
theorem florist_fertilizer_usage 
  (daily_usage : ℝ) 
  (num_days : ℕ) 
  (total_usage : ℝ) 
  (h1 : daily_usage = 2) 
  (h2 : num_days = 9) 
  (h3 : total_usage = 22) : 
  total_usage - (daily_usage * num_days) = 4 := by
sorry

end NUMINAMATH_CALUDE_florist_fertilizer_usage_l3966_396676


namespace NUMINAMATH_CALUDE_square_root_product_squared_l3966_396636

theorem square_root_product_squared : (Real.sqrt 900 * Real.sqrt 784)^2 = 705600 := by
  sorry

end NUMINAMATH_CALUDE_square_root_product_squared_l3966_396636


namespace NUMINAMATH_CALUDE_right_triangle_integer_sides_l3966_396696

theorem right_triangle_integer_sides (a b c : ℕ) : 
  a^2 + b^2 = c^2 → -- Pythagorean theorem (right-angled triangle)
  Nat.gcd a (Nat.gcd b c) = 1 → -- GCD of sides is 1
  ∃ m n : ℕ, 
    (a = 2*m*n ∧ b = m^2 - n^2 ∧ c = m^2 + n^2) ∨ 
    (b = 2*m*n ∧ a = m^2 - n^2 ∧ c = m^2 + n^2) :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_integer_sides_l3966_396696


namespace NUMINAMATH_CALUDE_ice_cream_cost_theorem_l3966_396669

def calculate_ice_cream_cost (chapati_count : ℕ) (chapati_price : ℚ)
                             (rice_count : ℕ) (rice_price : ℚ)
                             (veg_count : ℕ) (veg_price : ℚ)
                             (soup_count : ℕ) (soup_price : ℚ)
                             (dessert_count : ℕ) (dessert_price : ℚ)
                             (drink_count : ℕ) (drink_price : ℚ)
                             (discount_rate : ℚ) (tax_rate : ℚ)
                             (ice_cream_count : ℕ) (total_paid : ℚ) : ℚ :=
  let chapati_total := chapati_count * chapati_price
  let rice_total := rice_count * rice_price
  let veg_total := veg_count * veg_price
  let soup_total := soup_count * soup_price
  let dessert_total := dessert_count * dessert_price
  let drink_total := drink_count * drink_price * (1 - discount_rate)
  let subtotal := chapati_total + rice_total + veg_total + soup_total + dessert_total + drink_total
  let total_with_tax := subtotal * (1 + tax_rate)
  let ice_cream_total := total_paid - total_with_tax
  ice_cream_total / ice_cream_count

theorem ice_cream_cost_theorem :
  let chapati_count : ℕ := 16
  let chapati_price : ℚ := 6
  let rice_count : ℕ := 5
  let rice_price : ℚ := 45
  let veg_count : ℕ := 7
  let veg_price : ℚ := 70
  let soup_count : ℕ := 4
  let soup_price : ℚ := 30
  let dessert_count : ℕ := 3
  let dessert_price : ℚ := 85
  let drink_count : ℕ := 2
  let drink_price : ℚ := 50
  let discount_rate : ℚ := 0.1
  let tax_rate : ℚ := 0.18
  let ice_cream_count : ℕ := 6
  let total_paid : ℚ := 2159
  abs (calculate_ice_cream_cost chapati_count chapati_price rice_count rice_price
                                veg_count veg_price soup_count soup_price
                                dessert_count dessert_price drink_count drink_price
                                discount_rate tax_rate ice_cream_count total_paid - 108.89) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ice_cream_cost_theorem_l3966_396669


namespace NUMINAMATH_CALUDE_sum_of_bases_l3966_396682

-- Define the fractions F₁ and F₂
def F₁ (R : ℕ) : ℚ := (4 * R + 5) / (R^2 - 1)
def F₂ (R : ℕ) : ℚ := (5 * R + 4) / (R^2 - 1)

-- Define the conditions
def conditions (R₁ R₂ : ℕ) : Prop :=
  F₁ R₁ = F₁ R₂ ∧ F₂ R₁ = F₂ R₂ ∧
  R₁ ≥ 2 ∧ R₂ ≥ 2 ∧ -- Ensure bases are valid
  (∃ k : ℕ, F₁ R₁ = k / 11) ∧ -- Represents the repeating decimal 0.454545...
  (∃ k : ℕ, F₂ R₁ = k / 11) ∧ -- Represents the repeating decimal 0.545454...
  (∃ k : ℕ, F₁ R₂ = k / 11) ∧ -- Represents the repeating decimal 0.363636...
  (∃ k : ℕ, F₂ R₂ = k / 11)   -- Represents the repeating decimal 0.636363...

-- State the theorem
theorem sum_of_bases (R₁ R₂ : ℕ) : 
  conditions R₁ R₂ → R₁ + R₂ = 19 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_bases_l3966_396682


namespace NUMINAMATH_CALUDE_bicycle_owners_without_scooters_l3966_396607

theorem bicycle_owners_without_scooters (total : ℕ) (bicycle_owners : ℕ) (scooter_owners : ℕ) 
  (h_total : total = 500)
  (h_bicycle : bicycle_owners = 485)
  (h_scooter : scooter_owners = 150)
  (h_subset : bicycle_owners + scooter_owners ≥ total) :
  bicycle_owners - (bicycle_owners + scooter_owners - total) = 350 := by
  sorry

#check bicycle_owners_without_scooters

end NUMINAMATH_CALUDE_bicycle_owners_without_scooters_l3966_396607


namespace NUMINAMATH_CALUDE_first_thousand_diff_l3966_396686

/-- The sum of the first n even numbers starting from 0 -/
def sumEven (n : ℕ) : ℕ := n * (n - 1)

/-- The sum of the first n odd numbers starting from 1 -/
def sumOdd (n : ℕ) : ℕ := n^2

/-- The difference between the sum of the first n even numbers (including 0) 
    and the sum of the first n odd numbers -/
def diffEvenOdd (n : ℕ) : ℤ := (sumEven n : ℤ) - (sumOdd n : ℤ)

theorem first_thousand_diff : diffEvenOdd 1000 = -1000 := by
  sorry

end NUMINAMATH_CALUDE_first_thousand_diff_l3966_396686


namespace NUMINAMATH_CALUDE_triangle_inequality_theorem_l3966_396677

-- Define a triangle by its side lengths
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  pos_a : 0 < a
  pos_b : 0 < b
  pos_c : 0 < c
  triangle_ineq : a < b + c ∧ b < a + c ∧ c < a + b

-- State the theorem
theorem triangle_inequality_theorem (t : Triangle) :
  (t.a / (2 * (t.b + t.c))) + (t.b / (2 * (t.c + t.a))) + (t.c / (2 * (t.a + t.b))) ≥ 3/4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_theorem_l3966_396677


namespace NUMINAMATH_CALUDE_arson_sentence_calculation_l3966_396638

/-- Calculates the sentence for each arson count given the total sentence and other crime details. -/
theorem arson_sentence_calculation (total_sentence : ℕ) (arson_counts : ℕ) (burglary_charges : ℕ) 
  (burglary_sentence : ℕ) (petty_larceny_ratio : ℕ) (petty_larceny_sentence_fraction : ℚ) :
  total_sentence = 216 →
  arson_counts = 3 →
  burglary_charges = 2 →
  burglary_sentence = 18 →
  petty_larceny_ratio = 6 →
  petty_larceny_sentence_fraction = 1/3 →
  ∃ (arson_sentence : ℕ),
    arson_sentence = 36 ∧
    total_sentence = arson_counts * arson_sentence + 
                     burglary_charges * burglary_sentence +
                     petty_larceny_ratio * burglary_charges * (petty_larceny_sentence_fraction * burglary_sentence) :=
by
  sorry


end NUMINAMATH_CALUDE_arson_sentence_calculation_l3966_396638


namespace NUMINAMATH_CALUDE_weight_swap_l3966_396601

structure Weight :=
  (value : ℝ)

def WeighingScale (W X Y Z : Weight) : Prop :=
  (Z.value > Y.value) ∧
  (X.value > W.value) ∧
  (Y.value + Z.value > W.value + X.value) ∧
  (Z.value > W.value)

theorem weight_swap (W X Y Z : Weight) 
  (h : WeighingScale W X Y Z) : 
  (W.value + X.value > Y.value + Z.value) → 
  (Z.value + X.value > Y.value + W.value) :=
sorry

end NUMINAMATH_CALUDE_weight_swap_l3966_396601


namespace NUMINAMATH_CALUDE_assistant_productivity_increase_l3966_396666

theorem assistant_productivity_increase 
  (base_output : ℝ) 
  (base_hours : ℝ) 
  (output_increase_factor : ℝ) 
  (hours_decrease_factor : ℝ) 
  (h1 : output_increase_factor = 1.8) 
  (h2 : hours_decrease_factor = 0.9) :
  (output_increase_factor * base_output) / (hours_decrease_factor * base_hours) / 
  (base_output / base_hours) - 1 = 1 := by
sorry

end NUMINAMATH_CALUDE_assistant_productivity_increase_l3966_396666


namespace NUMINAMATH_CALUDE_num_correct_propositions_is_one_l3966_396688

/-- Represents a geometric proposition -/
inductive GeometricProposition
  | ThreePointsCircle
  | EqualArcEqualAngle
  | RightTrianglesSimilar
  | RhombusesSimilar

/-- Determines if a geometric proposition is correct -/
def is_correct (prop : GeometricProposition) : Bool :=
  match prop with
  | GeometricProposition.ThreePointsCircle => false
  | GeometricProposition.EqualArcEqualAngle => true
  | GeometricProposition.RightTrianglesSimilar => false
  | GeometricProposition.RhombusesSimilar => false

/-- The list of all propositions to be evaluated -/
def all_propositions : List GeometricProposition :=
  [GeometricProposition.ThreePointsCircle,
   GeometricProposition.EqualArcEqualAngle,
   GeometricProposition.RightTrianglesSimilar,
   GeometricProposition.RhombusesSimilar]

/-- Theorem stating that the number of correct propositions is 1 -/
theorem num_correct_propositions_is_one :
  (all_propositions.filter is_correct).length = 1 := by
  sorry

end NUMINAMATH_CALUDE_num_correct_propositions_is_one_l3966_396688


namespace NUMINAMATH_CALUDE_sum_of_dimensions_l3966_396603

-- Define the dimensions of the rectangular box
variable (X Y Z : ℝ)

-- Define the surface areas of the faces
def surfaceArea1 : ℝ := 18
def surfaceArea2 : ℝ := 18
def surfaceArea3 : ℝ := 36
def surfaceArea4 : ℝ := 36
def surfaceArea5 : ℝ := 54
def surfaceArea6 : ℝ := 54

-- State the theorem
theorem sum_of_dimensions (h1 : X * Y = surfaceArea1)
                          (h2 : X * Z = surfaceArea5)
                          (h3 : Y * Z = surfaceArea3)
                          (h4 : X > 0) (h5 : Y > 0) (h6 : Z > 0) :
  X + Y + Z = 11 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_dimensions_l3966_396603


namespace NUMINAMATH_CALUDE_remainder_theorem_l3966_396611

theorem remainder_theorem (x : ℝ) : 
  let R := fun x => 3^125 * x - 2^125 * x + 2^125 - 2 * 3^125
  let divisor := fun x => x^2 - 5*x + 6
  ∃ Q : ℝ → ℝ, x^125 = Q x * divisor x + R x ∧ 
  (∀ a b : ℝ, R x = a * x + b → (a = 3^125 - 2^125 ∧ b = 2^125 - 2 * 3^125)) :=
by sorry

end NUMINAMATH_CALUDE_remainder_theorem_l3966_396611


namespace NUMINAMATH_CALUDE_total_beverage_amount_l3966_396612

/-- Given 5 bottles, each containing 242.7 ml of beverage, 
    the total amount of beverage is 1213.5 ml. -/
theorem total_beverage_amount :
  let num_bottles : ℕ := 5
  let amount_per_bottle : ℝ := 242.7
  num_bottles * amount_per_bottle = 1213.5 := by
sorry

end NUMINAMATH_CALUDE_total_beverage_amount_l3966_396612


namespace NUMINAMATH_CALUDE_solve_for_b_l3966_396633

theorem solve_for_b (a b c d m : ℝ) (h1 : a ≠ b) (h2 : m = (c * a * d * b) / (a - b)) :
  b = (m * a) / (c * a * d + m) := by
sorry

end NUMINAMATH_CALUDE_solve_for_b_l3966_396633


namespace NUMINAMATH_CALUDE_triangle_sine_inequality_l3966_396657

theorem triangle_sine_inequality (A B C : ℝ) (h : A + B + C = 180) :
  (3 : ℝ) * Real.sqrt 3 / 2 ≥ Real.sin (3 * A) + Real.sin (3 * B) + Real.sin (3 * C) ∧
  Real.sin (3 * A) + Real.sin (3 * B) + Real.sin (3 * C) ≥ -2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_sine_inequality_l3966_396657


namespace NUMINAMATH_CALUDE_gravel_bags_per_truckload_l3966_396652

/-- Represents the roadwork company's asphalt paving problem -/
def roadwork_problem (road_length : ℕ) (gravel_pitch_ratio : ℕ) (truckloads_per_mile : ℕ)
  (day1_miles : ℕ) (day2_miles : ℕ) (remaining_pitch : ℕ) : Prop :=
  let total_paved := day1_miles + day2_miles
  let remaining_miles := road_length - total_paved
  let remaining_truckloads := remaining_miles * truckloads_per_mile
  let pitch_per_truckload : ℚ := remaining_pitch / remaining_truckloads
  let gravel_per_truckload := gravel_pitch_ratio * pitch_per_truckload
  gravel_per_truckload = 2

/-- The main theorem stating that the number of bags of gravel per truckload is 2 -/
theorem gravel_bags_per_truckload :
  roadwork_problem 16 5 3 4 7 6 :=
sorry

end NUMINAMATH_CALUDE_gravel_bags_per_truckload_l3966_396652


namespace NUMINAMATH_CALUDE_count_true_propositions_l3966_396647

theorem count_true_propositions : 
  (∀ (x : ℝ), x^2 + 1 > 0) ∧ 
  (∃ (x : ℤ), x^3 < 1) ∧ 
  (∀ (x : ℚ), x^2 ≠ 2) ∧ 
  ¬(∀ (x : ℕ), x^4 ≥ 1) := by
  sorry

#check count_true_propositions

end NUMINAMATH_CALUDE_count_true_propositions_l3966_396647


namespace NUMINAMATH_CALUDE_quotient_of_A_and_B_l3966_396661

/-- Given A and B as defined, prove that A / B = 31 -/
theorem quotient_of_A_and_B : 
  let A := 8 * 10 + 13 * 1
  let B := 30 - 9 - 9 - 9
  A / B = 31 := by
sorry

end NUMINAMATH_CALUDE_quotient_of_A_and_B_l3966_396661


namespace NUMINAMATH_CALUDE_matching_socks_probability_l3966_396697

/-- The number of pairs of socks -/
def num_pairs : ℕ := 5

/-- The total number of socks -/
def total_socks : ℕ := 2 * num_pairs

/-- The number of socks selected each day -/
def socks_per_day : ℕ := 2

/-- The probability of selecting matching socks for the first time on Wednesday -/
def prob_match_wednesday : ℚ :=
  26 / 315

theorem matching_socks_probability :
  let monday_selections := Nat.choose total_socks socks_per_day
  let tuesday_selections := Nat.choose (total_socks - socks_per_day) socks_per_day
  let wednesday_selections := Nat.choose (total_socks - 2 * socks_per_day) socks_per_day
  prob_match_wednesday =
    (monday_selections - num_pairs) / monday_selections *
    ((1 / tuesday_selections * 1 / 5) +
     (12 / tuesday_selections * 2 / 15) +
     (12 / tuesday_selections * 1 / 15)) :=
by sorry

#eval prob_match_wednesday

end NUMINAMATH_CALUDE_matching_socks_probability_l3966_396697


namespace NUMINAMATH_CALUDE_jimmy_pens_purchase_l3966_396659

theorem jimmy_pens_purchase (pen_cost : ℕ) (notebook_cost : ℕ) (folder_cost : ℕ)
  (notebooks_bought : ℕ) (folders_bought : ℕ) (paid : ℕ) (change : ℕ) :
  pen_cost = 1 →
  notebook_cost = 3 →
  folder_cost = 5 →
  notebooks_bought = 4 →
  folders_bought = 2 →
  paid = 50 →
  change = 25 →
  (paid - change - (notebooks_bought * notebook_cost + folders_bought * folder_cost)) / pen_cost = 3 :=
by sorry

end NUMINAMATH_CALUDE_jimmy_pens_purchase_l3966_396659


namespace NUMINAMATH_CALUDE_min_shift_for_trig_transformation_l3966_396672

open Real

/-- The minimum positive shift required to transform sin(2x) + √3cos(2x) into 2sin(2x) -/
theorem min_shift_for_trig_transformation : ∃ (m : ℝ), m > 0 ∧
  (∀ (x : ℝ), sin (2*x) + Real.sqrt 3 * cos (2*x) = 2 * sin (2*(x + m))) ∧
  (∀ (m' : ℝ), m' > 0 → 
    (∀ (x : ℝ), sin (2*x) + Real.sqrt 3 * cos (2*x) = 2 * sin (2*(x + m'))) → 
    m ≤ m') ∧
  m = π / 6 := by
sorry

end NUMINAMATH_CALUDE_min_shift_for_trig_transformation_l3966_396672


namespace NUMINAMATH_CALUDE_hyperbola_equation_l3966_396600

/-- The standard equation of a hyperbola with given foci and asymptotes -/
theorem hyperbola_equation (c : ℝ) (m : ℝ) :
  c = Real.sqrt 10 →
  m = 1 / 2 →
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧
    (∀ (x y : ℝ), (x^2 / a^2 - y^2 / b^2 = 1 ↔
      ((x + c)^2 + y^2)^(1/2) - ((x - c)^2 + y^2)^(1/2) = 2*a)) ∧
    (∀ (x : ℝ), y = m*x ∨ y = -m*x ↔ x^2 / a^2 - y^2 / b^2 = 0) ∧
    a^2 = 8 ∧ b^2 = 2 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l3966_396600


namespace NUMINAMATH_CALUDE_max_expression_value_l3966_396619

def expression (a b c d : ℕ) : ℕ := c * a^b + d

theorem max_expression_value :
  ∃ (a b c d : ℕ),
    a ∈ ({0, 1, 3, 4} : Set ℕ) ∧
    b ∈ ({0, 1, 3, 4} : Set ℕ) ∧
    c ∈ ({0, 1, 3, 4} : Set ℕ) ∧
    d ∈ ({0, 1, 3, 4} : Set ℕ) ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    expression a b c d = 196 ∧
    ∀ (x y z w : ℕ),
      x ∈ ({0, 1, 3, 4} : Set ℕ) →
      y ∈ ({0, 1, 3, 4} : Set ℕ) →
      z ∈ ({0, 1, 3, 4} : Set ℕ) →
      w ∈ ({0, 1, 3, 4} : Set ℕ) →
      x ≠ y ∧ x ≠ z ∧ x ≠ w ∧ y ≠ z ∧ y ≠ w ∧ z ≠ w →
      expression x y z w ≤ 196 :=
by
  sorry

#check max_expression_value

end NUMINAMATH_CALUDE_max_expression_value_l3966_396619


namespace NUMINAMATH_CALUDE_log_function_passes_through_point_l3966_396699

-- Define the logarithm function for any base a > 0 and a ≠ 1
noncomputable def log (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

-- Define the function f(x) = log_a(x-1) + 2
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := log a (x - 1) + 2

-- Theorem statement
theorem log_function_passes_through_point (a : ℝ) (ha : a > 0) (ha' : a ≠ 1) :
  f a 2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_log_function_passes_through_point_l3966_396699


namespace NUMINAMATH_CALUDE_newspaper_weeks_l3966_396622

/-- The cost of a weekday newspaper --/
def weekday_cost : ℚ := 1/2

/-- The cost of a Sunday newspaper --/
def sunday_cost : ℚ := 2

/-- The number of weekday newspapers bought per week --/
def weekday_papers : ℕ := 3

/-- The total amount spent on newspapers --/
def total_spent : ℚ := 28

/-- The number of weeks Hillary buys newspapers --/
def weeks_buying : ℕ := 8

theorem newspaper_weeks : 
  (weekday_papers * weekday_cost + sunday_cost) * weeks_buying = total_spent := by
  sorry

end NUMINAMATH_CALUDE_newspaper_weeks_l3966_396622


namespace NUMINAMATH_CALUDE_max_true_statements_l3966_396650

theorem max_true_statements (y : ℝ) : 
  let statement1 := 0 < y^3 ∧ y^3 < 1
  let statement2 := y^3 > 1
  let statement3 := -1 < y ∧ y < 0
  let statement4 := 0 < y ∧ y < 1
  let statement5 := 0 < y^2 - y^3 ∧ y^2 - y^3 < 1
  (∃ y : ℝ, (statement1 ∧ statement4 ∧ statement5)) ∧
  (∀ y : ℝ, ¬(statement1 ∧ statement2 ∧ statement3 ∧ statement4) ∧
            ¬(statement1 ∧ statement2 ∧ statement3 ∧ statement5) ∧
            ¬(statement1 ∧ statement2 ∧ statement4 ∧ statement5) ∧
            ¬(statement1 ∧ statement3 ∧ statement4 ∧ statement5) ∧
            ¬(statement2 ∧ statement3 ∧ statement4 ∧ statement5)) :=
by
  sorry

end NUMINAMATH_CALUDE_max_true_statements_l3966_396650


namespace NUMINAMATH_CALUDE_point_in_second_quadrant_l3966_396667

def second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0

theorem point_in_second_quadrant :
  let x : ℝ := -3
  let y : ℝ := 2 * Real.sqrt 2
  second_quadrant x y :=
by
  sorry

end NUMINAMATH_CALUDE_point_in_second_quadrant_l3966_396667


namespace NUMINAMATH_CALUDE_tg_2alpha_l3966_396648

theorem tg_2alpha (α : Real) 
  (h1 : Real.cos (α - Real.pi/2) = 0.2) 
  (h2 : Real.pi/2 < α ∧ α < Real.pi) : 
  Real.tan (2*α) = -4 * Real.sqrt 6 / 23 := by
sorry

end NUMINAMATH_CALUDE_tg_2alpha_l3966_396648


namespace NUMINAMATH_CALUDE_batsman_innings_properties_l3966_396626

/-- Represents a cricket batsman's innings statistics -/
structure BatsmanInnings where
  total_runs : ℕ
  total_balls : ℕ
  singles : ℕ
  doubles : ℕ

/-- Calculates the percentage of runs scored by running between wickets -/
def runs_by_running_percentage (innings : BatsmanInnings) : ℚ :=
  ((innings.singles + 2 * innings.doubles : ℚ) / innings.total_runs) * 100

/-- Calculates the strike rate of the batsman -/
def strike_rate (innings : BatsmanInnings) : ℚ :=
  (innings.total_runs : ℚ) / innings.total_balls * 100

/-- Theorem stating the properties of the given batsman's innings -/
theorem batsman_innings_properties :
  let innings : BatsmanInnings := {
    total_runs := 180,
    total_balls := 120,
    singles := 35,
    doubles := 15
  }
  runs_by_running_percentage innings = 36.11 ∧
  strike_rate innings = 150 := by
  sorry

end NUMINAMATH_CALUDE_batsman_innings_properties_l3966_396626


namespace NUMINAMATH_CALUDE_minimize_segment_expression_l3966_396692

/-- Given a line segment AB of length a, the point C that minimizes AC^2 + 3CB^2 is at 3a/4 from A -/
theorem minimize_segment_expression (a : ℝ) (h : a > 0) :
  ∃ c : ℝ, c = 3*a/4 ∧ 
    ∀ x : ℝ, 0 ≤ x ∧ x ≤ a → 
      x^2 + 3*(a-x)^2 ≥ c^2 + 3*(a-c)^2 :=
by sorry


end NUMINAMATH_CALUDE_minimize_segment_expression_l3966_396692


namespace NUMINAMATH_CALUDE_power_of_two_expression_l3966_396614

theorem power_of_two_expression : 2^4 * 2^2 + 2^4 / 2^2 = 68 := by
  sorry

end NUMINAMATH_CALUDE_power_of_two_expression_l3966_396614


namespace NUMINAMATH_CALUDE_number_of_divisors_of_36_l3966_396627

theorem number_of_divisors_of_36 : Nat.card {d : ℕ | d > 0 ∧ 36 % d = 0} = 9 := by
  sorry

end NUMINAMATH_CALUDE_number_of_divisors_of_36_l3966_396627


namespace NUMINAMATH_CALUDE_root_product_l3966_396621

theorem root_product (b c : ℝ) : 
  (∀ x : ℝ, x^2 - 2*x - 1 = 0 → x^7 - b*x - c = 0) → 
  b * c = 11830 := by
sorry

end NUMINAMATH_CALUDE_root_product_l3966_396621


namespace NUMINAMATH_CALUDE_ln_intersection_and_exponential_inequality_l3966_396631

open Real

theorem ln_intersection_and_exponential_inequality (m n : ℝ) (h : m < n) :
  (∃! x : ℝ, log x = x - 1) ∧
  ((exp n - exp m) / (n - m) > exp ((m + n) / 2)) := by
  sorry

end NUMINAMATH_CALUDE_ln_intersection_and_exponential_inequality_l3966_396631


namespace NUMINAMATH_CALUDE_percentage_error_calculation_l3966_396683

theorem percentage_error_calculation : 
  let correct_multiplier : ℚ := 5/3
  let incorrect_multiplier : ℚ := 3/5
  let percentage_error := ((correct_multiplier - incorrect_multiplier) / correct_multiplier) * 100
  percentage_error = 64
  := by sorry

end NUMINAMATH_CALUDE_percentage_error_calculation_l3966_396683


namespace NUMINAMATH_CALUDE_gas_station_distance_l3966_396663

theorem gas_station_distance (x : ℝ) : 
  (¬ (x ≥ 10)) →   -- Adam's statement is false
  (¬ (x ≤ 7)) →    -- Betty's statement is false
  (¬ (x < 5)) →    -- Carol's statement is false
  (¬ (x ≤ 9)) →    -- Dave's statement is false
  x > 9 :=         -- Conclusion: x is in the interval (9, ∞)
by
  sorry            -- Proof is omitted

#check gas_station_distance

end NUMINAMATH_CALUDE_gas_station_distance_l3966_396663


namespace NUMINAMATH_CALUDE_exists_function_with_properties_l3966_396684

-- Define the function type
def RealFunction := ℝ → ℝ

-- Define the properties of the function
def HasFunctionalEquation (f : RealFunction) : Prop :=
  ∀ x₁ x₂ : ℝ, x₁ > 0 → x₂ > 0 → f (x₁ * x₂) = f x₁ + f x₂

def HasNegativeDerivative (f : RealFunction) : Prop :=
  ∀ x : ℝ, x > 0 → (deriv f x) < 0

-- State the theorem
theorem exists_function_with_properties :
  ∃ f : RealFunction,
    HasFunctionalEquation f ∧ HasNegativeDerivative f := by
  sorry

end NUMINAMATH_CALUDE_exists_function_with_properties_l3966_396684


namespace NUMINAMATH_CALUDE_smallest_divisible_number_l3966_396624

theorem smallest_divisible_number (n : ℕ) : 
  n = 1008 → 
  (1020 - 12 = n) → 
  (∃ k : ℕ, 36 * k = n) → 
  (∃ k : ℕ, 48 * k = n) → 
  (∃ k : ℕ, 56 * k = n) → 
  ∀ m : ℕ, m ∣ 1008 ∧ m ∣ 36 ∧ m ∣ 48 ∧ m ∣ 56 → m ≤ n :=
by sorry

#check smallest_divisible_number

end NUMINAMATH_CALUDE_smallest_divisible_number_l3966_396624
