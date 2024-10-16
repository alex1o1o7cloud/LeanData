import Mathlib

namespace NUMINAMATH_CALUDE_prob_five_is_one_thirteenth_l3489_348901

/-- Represents a standard deck of cards -/
structure Deck :=
  (cards : Finset (Nat × Nat))
  (card_count : cards.card = 52)
  (rank_count : (cards.image (·.1)).card = 13)
  (suit_count : (cards.image (·.2)).card = 4)
  (unique_cards : ∀ r s, (r, s) ∈ cards ↔ r ∈ Finset.range 13 ∧ s ∈ Finset.range 4)

/-- The probability of drawing a specific rank from a standard deck -/
def prob_rank (d : Deck) (rank : Nat) : ℚ :=
  (d.cards.filter (·.1 = rank)).card / d.cards.card

/-- Theorem: The probability of drawing a 5 from a standard deck is 1/13 -/
theorem prob_five_is_one_thirteenth (d : Deck) : prob_rank d 5 = 1 / 13 := by
  sorry

end NUMINAMATH_CALUDE_prob_five_is_one_thirteenth_l3489_348901


namespace NUMINAMATH_CALUDE_investment_sum_l3489_348978

/-- Given a sum invested at different interest rates, prove the sum equals 8400 --/
theorem investment_sum (P : ℝ) : 
  (P * 0.15 * 2) - (P * 0.10 * 2) = 840 → P = 8400 := by
  sorry

end NUMINAMATH_CALUDE_investment_sum_l3489_348978


namespace NUMINAMATH_CALUDE_greatest_integer_problem_l3489_348947

theorem greatest_integer_problem : 
  ⌊100 * (Real.cos (18.5 * π / 180) / Real.sin (17.5 * π / 180))⌋ = 273 := by
  sorry

end NUMINAMATH_CALUDE_greatest_integer_problem_l3489_348947


namespace NUMINAMATH_CALUDE_parabola_equidistant_point_l3489_348974

/-- 
For a parabola y^2 = 2px where p > 0, with point P(2, 2p) on the parabola, 
origin O(0, 0), and focus F, the point M satisfying |MP| = |MO| = |MF| 
has coordinates (1/4, 7/4).
-/
theorem parabola_equidistant_point (p : ℝ) (h : p > 0) : 
  let parabola := {(x, y) : ℝ × ℝ | y^2 = 2*p*x}
  let P := (2, 2*p)
  let O := (0, 0)
  let F := (p/2, 0)
  ∃ M : ℝ × ℝ, M ∈ parabola ∧ 
    dist M P = dist M O ∧ 
    dist M O = dist M F ∧ 
    M = (1/4, 7/4) :=
by sorry

end NUMINAMATH_CALUDE_parabola_equidistant_point_l3489_348974


namespace NUMINAMATH_CALUDE_min_sum_of_reciprocal_sum_eq_one_l3489_348939

theorem min_sum_of_reciprocal_sum_eq_one (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : 1 / a + 1 / b = 1) : 
  a + b ≥ 4 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ 1 / a₀ + 1 / b₀ = 1 ∧ a₀ + b₀ = 4 :=
sorry

end NUMINAMATH_CALUDE_min_sum_of_reciprocal_sum_eq_one_l3489_348939


namespace NUMINAMATH_CALUDE_grid_sum_property_l3489_348914

def Grid := Matrix (Fin 2) (Fin 3) ℕ

def is_valid_grid (g : Grid) : Prop :=
  ∀ i j, 1 ≤ g i j ∧ g i j ≤ 9 ∧
  (∀ i₁ i₂ j, i₁ ≠ i₂ → g i₁ j ≠ g i₂ j) ∧
  (∀ i j₁ j₂, j₁ ≠ j₂ → g i j₁ ≠ g i j₂)

theorem grid_sum_property (g : Grid) (h : is_valid_grid g) :
  (g 0 0 + g 0 1 + g 0 2 = 23) →
  (g 0 0 + g 1 0 = 14) →
  (g 0 1 + g 1 1 = 16) →
  (g 0 2 + g 1 2 = 17) →
  g 1 0 + 2 * g 1 1 + 3 * g 1 2 = 49 := by
sorry

end NUMINAMATH_CALUDE_grid_sum_property_l3489_348914


namespace NUMINAMATH_CALUDE_sqrt_one_minus_sqrt_two_squared_l3489_348946

theorem sqrt_one_minus_sqrt_two_squared (h : 1 < Real.sqrt 2) :
  Real.sqrt ((1 - Real.sqrt 2) ^ 2) = Real.sqrt 2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_one_minus_sqrt_two_squared_l3489_348946


namespace NUMINAMATH_CALUDE_no_special_triangle_exists_l3489_348958

/-- A triangle with sides and angles in arithmetic progression, given area, and circumradius -/
structure SpecialTriangle where
  /-- The common difference of the arithmetic progression of sides -/
  d : ℝ
  /-- The middle term of the arithmetic progression of sides -/
  b : ℝ
  /-- The area of the triangle -/
  area : ℝ
  /-- The radius of the circumscribed circle -/
  circumradius : ℝ
  /-- The sides form an arithmetic progression -/
  sides_progression : d ≥ 0 ∧ b > d
  /-- The angles form an arithmetic progression -/
  angles_progression : ∃ (α β γ : ℝ), α + β + γ = 180 ∧ β = 60 ∧ α < β ∧ β < γ
  /-- The area is 50 cm² -/
  area_constraint : area = 50
  /-- The circumradius is 10 cm -/
  circumradius_constraint : circumradius = 10

/-- Theorem stating that no triangle satisfies all the given conditions -/
theorem no_special_triangle_exists : ¬∃ (t : SpecialTriangle), True := by
  sorry

end NUMINAMATH_CALUDE_no_special_triangle_exists_l3489_348958


namespace NUMINAMATH_CALUDE_original_students_per_section_l3489_348970

theorem original_students_per_section 
  (S : ℕ) -- Initial number of sections
  (x : ℕ) -- Initial number of students per section
  (h1 : S + 3 = 16) -- After admission, there are S + 3 sections, totaling 16
  (h2 : S * x + 24 = 16 * 21) -- Total students after admission equals 16 sections of 21 students each
  : x = 24 := by
  sorry

end NUMINAMATH_CALUDE_original_students_per_section_l3489_348970


namespace NUMINAMATH_CALUDE_chord_squared_sum_l3489_348903

-- Define the circle and points
def Circle : Type := Unit
def Point : Type := Unit
def O : Point := Unit.unit -- Center of the circle
def A : Point := Unit.unit
def B : Point := Unit.unit
def C : Point := Unit.unit
def D : Point := Unit.unit
def E : Point := Unit.unit

-- Define the radius of the circle
def radius : ℝ := 6

-- Define the distance function
def distance : Point → Point → ℝ := sorry

-- Define the angle function
def angle : Point → Point → Point → ℝ := sorry

-- State the theorem
theorem chord_squared_sum (c : Circle) :
  distance O A = radius ∧
  distance O B = radius ∧
  distance A B = 2 * radius ∧
  distance B E = 3 ∧
  angle A E C = π / 3 →
  (distance C E)^2 + (distance D E)^2 = 108 := by
  sorry

end NUMINAMATH_CALUDE_chord_squared_sum_l3489_348903


namespace NUMINAMATH_CALUDE_f_composition_equals_pi_plus_one_l3489_348999

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then x + 1
  else if x = 0 then Real.pi
  else 0

theorem f_composition_equals_pi_plus_one :
  f (f (f (-1))) = Real.pi + 1 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_equals_pi_plus_one_l3489_348999


namespace NUMINAMATH_CALUDE_besfamilies_children_count_l3489_348927

/-- Represents the Besfamilies family structure and age calculations -/
structure Besfamilies where
  initialAge : ℕ  -- Family age when youngest child was born
  finalAge : ℕ    -- Family age after several years
  yearsPassed : ℕ -- Number of years passed

/-- Calculates the number of children in the Besfamilies -/
def numberOfChildren (family : Besfamilies) : ℕ :=
  ((family.finalAge - family.initialAge) / family.yearsPassed) - 2

/-- Theorem stating the number of children in the Besfamilies -/
theorem besfamilies_children_count 
  (family : Besfamilies) 
  (h1 : family.initialAge = 101)
  (h2 : family.finalAge = 150)
  (h3 : family.yearsPassed > 1)
  (h4 : (family.finalAge - family.initialAge) % family.yearsPassed = 0) :
  numberOfChildren family = 5 := by
  sorry

#eval numberOfChildren { initialAge := 101, finalAge := 150, yearsPassed := 7 }

end NUMINAMATH_CALUDE_besfamilies_children_count_l3489_348927


namespace NUMINAMATH_CALUDE_coffee_container_weight_l3489_348954

def suki_bags : ℝ := 6.5
def suki_weight_per_bag : ℝ := 22
def jimmy_bags : ℝ := 4.5
def jimmy_weight_per_bag : ℝ := 18
def num_containers : ℕ := 28

theorem coffee_container_weight :
  (suki_bags * suki_weight_per_bag + jimmy_bags * jimmy_weight_per_bag) / num_containers = 8 := by
  sorry

end NUMINAMATH_CALUDE_coffee_container_weight_l3489_348954


namespace NUMINAMATH_CALUDE_binomial_coefficient_problem_l3489_348916

theorem binomial_coefficient_problem (h1 : Nat.choose 18 11 = 31824)
                                     (h2 : Nat.choose 18 12 = 18564)
                                     (h3 : Nat.choose 20 13 = 77520) :
  Nat.choose 19 13 = 27132 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_problem_l3489_348916


namespace NUMINAMATH_CALUDE_max_difference_on_board_l3489_348933

/-- A type representing a 10x10 board with numbers from 1 to 100 -/
def Board := Fin 10 → Fin 10 → Fin 100

/-- A predicate that checks if a board is valid (each number appears exactly once) -/
def is_valid_board (b : Board) : Prop :=
  ∀ n : Fin 100, ∃! (i j : Fin 10), b i j = n

/-- The main theorem statement -/
theorem max_difference_on_board :
  ∀ b : Board, is_valid_board b →
    ∃ (i j k : Fin 10), 
      (i = k ∨ j = k) ∧ 
      ((b i j : ℕ) ≥ (b k j : ℕ) + 54 ∨ (b k j : ℕ) ≥ (b i j : ℕ) + 54) :=
by sorry

end NUMINAMATH_CALUDE_max_difference_on_board_l3489_348933


namespace NUMINAMATH_CALUDE_work_completion_time_l3489_348910

/-- The number of days it takes for A to complete the work alone -/
def days_for_A : ℕ := 6

/-- The number of days it takes for B to complete the work alone -/
def days_for_B : ℕ := 8

/-- The number of days it takes for A, B, and C to complete the work together -/
def days_for_ABC : ℕ := 3

/-- The total payment for the work in rupees -/
def total_payment : ℕ := 1200

/-- C's share of the payment in rupees -/
def C_share : ℕ := 150

theorem work_completion_time :
  (1 : ℚ) / days_for_A + (1 : ℚ) / days_for_B + 
  ((C_share : ℚ) / total_payment) * ((1 : ℚ) / days_for_ABC) = 
  (1 : ℚ) / days_for_ABC := by sorry

end NUMINAMATH_CALUDE_work_completion_time_l3489_348910


namespace NUMINAMATH_CALUDE_family_heights_l3489_348906

/-- Represents the heights of family members and proves statements about their relationships -/
theorem family_heights (binbin_height mother_height : Real) 
  (father_taller_by : Real) (h1 : binbin_height = 1.46) 
  (h2 : father_taller_by = 0.32) (h3 : mother_height = 1.5) : 
  (binbin_height + father_taller_by = 1.78) ∧ 
  ((binbin_height + father_taller_by) - mother_height = 0.28) := by
  sorry

#check family_heights

end NUMINAMATH_CALUDE_family_heights_l3489_348906


namespace NUMINAMATH_CALUDE_specific_frustum_volume_l3489_348981

/-- A frustum with given base areas and lateral surface area -/
structure Frustum where
  upper_base_area : ℝ
  lower_base_area : ℝ
  lateral_surface_area : ℝ

/-- The volume of a frustum -/
def volume (f : Frustum) : ℝ := sorry

/-- Theorem stating the volume of the specific frustum -/
theorem specific_frustum_volume :
  ∃ (f : Frustum),
    f.upper_base_area = π ∧
    f.lower_base_area = 4 * π ∧
    f.lateral_surface_area = 6 * π ∧
    volume f = (7 * Real.sqrt 3 / 3) * π := by sorry

end NUMINAMATH_CALUDE_specific_frustum_volume_l3489_348981


namespace NUMINAMATH_CALUDE_smallest_n_for_terminating_decimal_l3489_348908

/-- A fraction a/b is a terminating decimal if and only if b can be written as 2^m * 5^n for some non-negative integers m and n. -/
def is_terminating_decimal (a b : ℕ) : Prop :=
  ∃ (m n : ℕ), b = 2^m * 5^n

/-- 50 is the smallest positive integer n such that n/(n+150) is a terminating decimal. -/
theorem smallest_n_for_terminating_decimal :
  (∀ k : ℕ, 0 < k → k < 50 → ¬ is_terminating_decimal k (k + 150)) ∧
  is_terminating_decimal 50 200 := by
sorry

end NUMINAMATH_CALUDE_smallest_n_for_terminating_decimal_l3489_348908


namespace NUMINAMATH_CALUDE_absolute_value_squared_l3489_348997

theorem absolute_value_squared (a b : ℝ) : |a| < b → a^2 < b^2 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_squared_l3489_348997


namespace NUMINAMATH_CALUDE_odd_power_sum_coprime_l3489_348990

theorem odd_power_sum_coprime (a n m : ℕ) (hodd : Odd a) (hpos_n : n > 0) (hpos_m : m > 0) (hne : n ≠ m) :
  Nat.gcd (a^(2^n) + 2^(2^n)) (a^(2^m) + 2^(2^m)) = 1 :=
by sorry

end NUMINAMATH_CALUDE_odd_power_sum_coprime_l3489_348990


namespace NUMINAMATH_CALUDE_tangent_line_at_one_unique_zero_implies_a_one_max_value_of_g_l3489_348988

noncomputable section

-- Define the functions f and g
def f (a : ℝ) (x : ℝ) : ℝ := (x^2 - 2*x) * Real.log x + a * x^2 + 2
def g (a : ℝ) (x : ℝ) : ℝ := f a x - x - 2

-- Part I
theorem tangent_line_at_one (a : ℝ) (h : a = -1) :
  ∃ (m b : ℝ), m = 3 ∧ b = -4 ∧ ∀ x y, y = f a x → y = m * (x - 1) + f a 1 :=
sorry

-- Part II
theorem unique_zero_implies_a_one (a : ℝ) (h : a > 0) :
  (∃! x, g a x = 0) → a = 1 :=
sorry

-- Part III
theorem max_value_of_g (x : ℝ) (h1 : Real.exp (-2) < x) (h2 : x < Real.exp 1) :
  g 1 x ≤ 2 * Real.exp 2 - 3 * Real.exp 1 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_at_one_unique_zero_implies_a_one_max_value_of_g_l3489_348988


namespace NUMINAMATH_CALUDE_jan_math_problem_l3489_348905

-- Define the operation of rounding to the nearest ten
def roundToNearestTen (x : ℤ) : ℤ :=
  10 * ((x + 5) / 10)

-- Theorem statement
theorem jan_math_problem :
  roundToNearestTen (83 - 29 + 58) = 110 := by
  sorry

end NUMINAMATH_CALUDE_jan_math_problem_l3489_348905


namespace NUMINAMATH_CALUDE_sphere_volume_and_circumference_l3489_348966

/-- Given a sphere with surface area 256π cm², prove its volume and circumference. -/
theorem sphere_volume_and_circumference :
  ∀ (r : ℝ),
  (4 * π * r^2 = 256 * π) →
  (4/3 * π * r^3 = 2048/3 * π) ∧ (2 * π * r = 16 * π) := by
  sorry


end NUMINAMATH_CALUDE_sphere_volume_and_circumference_l3489_348966


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l3489_348995

def U : Set Nat := {1, 2, 3, 4, 5}
def M : Set Nat := {3, 4, 5}
def N : Set Nat := {2, 3}

theorem complement_intersection_theorem :
  (U \ N) ∩ M = {4, 5} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l3489_348995


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l3489_348919

/-- A geometric sequence with the given properties has a common ratio of 1/2 -/
theorem geometric_sequence_common_ratio 
  (a : ℕ → ℝ) 
  (h_geometric : ∀ n : ℕ, a (n + 1) = a n * (a 2 / a 1)) 
  (h_sum_1_3 : a 1 + a 3 = 10) 
  (h_sum_4_6 : a 4 + a 6 = 5/4) : 
  a 2 / a 1 = 1/2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l3489_348919


namespace NUMINAMATH_CALUDE_aristocrat_spending_l3489_348922

/-- Proves that the total number of people is 3552 given the conditions of the aristocrat's spending --/
theorem aristocrat_spending (M W : ℕ) : 
  (M / 9 : ℚ) * 45 + (W / 12 : ℚ) * 60 = 17760 → M + W = 3552 := by
  sorry

end NUMINAMATH_CALUDE_aristocrat_spending_l3489_348922


namespace NUMINAMATH_CALUDE_root_configurations_l3489_348900

-- Define the polynomial
def polynomial (a b c x : ℂ) : ℂ := x^4 - a*x^3 - b*x + c

-- Define the theorem
theorem root_configurations (a b c : ℂ) :
  (∃ d : ℂ, d ≠ a ∧ d ≠ b ∧ d ≠ c ∧
    (∀ x : ℂ, polynomial a b c x = 0 ↔ x = a ∨ x = b ∨ x = c ∨ x = d)) →
  ((a = 0 ∧ b = 0 ∧ c ≠ 0) ∨
   (a ≠ 0 ∧ b = c ∧ c ≠ 0 ∧ c^2 + c + 1 = 0)) :=
by sorry


end NUMINAMATH_CALUDE_root_configurations_l3489_348900


namespace NUMINAMATH_CALUDE_trick_deck_purchase_total_l3489_348924

theorem trick_deck_purchase_total (deck_price : ℕ) (tom_decks : ℕ) (friend_decks : ℕ) : 
  deck_price = 8 → tom_decks = 3 → friend_decks = 5 → 
  deck_price * (tom_decks + friend_decks) = 64 := by
sorry

end NUMINAMATH_CALUDE_trick_deck_purchase_total_l3489_348924


namespace NUMINAMATH_CALUDE_bike_jog_swim_rates_sum_of_squares_l3489_348950

theorem bike_jog_swim_rates_sum_of_squares : 
  ∃ (b j s : ℕ),
    3 * b + 2 * j + 4 * s = 80 ∧
    4 * b + 3 * j + 2 * s = 98 ∧
    b * b + j * j + s * s = 536 := by
  sorry

end NUMINAMATH_CALUDE_bike_jog_swim_rates_sum_of_squares_l3489_348950


namespace NUMINAMATH_CALUDE_sum_of_solutions_eq_twelve_l3489_348945

theorem sum_of_solutions_eq_twelve : 
  ∃ (x₁ x₂ : ℝ), (x₁ - 6)^2 = 50 ∧ (x₂ - 6)^2 = 50 ∧ x₁ + x₂ = 12 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_solutions_eq_twelve_l3489_348945


namespace NUMINAMATH_CALUDE_neighborhood_cable_cost_l3489_348998

/-- Calculate the total cost of cable for a neighborhood given the following conditions:
- 18 east-west streets, each 2 miles long
- 10 north-south streets, each 4 miles long
- 5 miles of cable needed to electrify 1 mile of street
- Cable costs $2000 per mile
-/
theorem neighborhood_cable_cost :
  let east_west_streets := 18
  let east_west_length := 2
  let north_south_streets := 10
  let north_south_length := 4
  let cable_per_mile := 5
  let cost_per_mile := 2000
  let total_street_length := east_west_streets * east_west_length + north_south_streets * north_south_length
  let total_cable_length := total_street_length * cable_per_mile
  let total_cost := total_cable_length * cost_per_mile
  total_cost = 760000 := by
  sorry

end NUMINAMATH_CALUDE_neighborhood_cable_cost_l3489_348998


namespace NUMINAMATH_CALUDE_max_e_value_l3489_348969

def is_valid_number (d e : ℕ) : Prop :=
  d ≤ 9 ∧ e ≤ 9 ∧ 
  (600000 + d * 10000 + 28000 + e) % 18 = 0

theorem max_e_value :
  ∃ (d : ℕ), is_valid_number d 8 ∧
  ∀ (d' e' : ℕ), is_valid_number d' e' → e' ≤ 8 :=
sorry

end NUMINAMATH_CALUDE_max_e_value_l3489_348969


namespace NUMINAMATH_CALUDE_alice_wins_l3489_348938

/-- Represents a position on the chocolate tablet. -/
structure Position :=
  (row : Nat)
  (col : Nat)

/-- Represents the game state. -/
structure GameState :=
  (m : Nat)
  (n : Nat)
  (currentPosition : Position)

/-- Defines a valid move in the game. -/
def ValidMove (state : GameState) (pos : Position) : Prop :=
  pos.row ≤ state.m ∧ pos.col ≤ state.n ∧
  (pos.row > state.currentPosition.row ∨ pos.col > state.currentPosition.col)

/-- Defines a winning position for the current player. -/
def WinningPosition (state : GameState) : Prop :=
  state.currentPosition.row = 1 ∧ state.currentPosition.col = 1

/-- Defines the concept of a winning strategy for Alice. -/
def AliceHasWinningStrategy (m n : Nat) : Prop :=
  m ≥ 2 ∧ n ≥ 2 →
  ∃ (strategy : GameState → Position),
    ∀ (state : GameState),
      ValidMove state (strategy state) ∧
      (WinningPosition state ∨
       ∀ (bobMove : Position),
         ValidMove (GameState.mk state.m state.n bobMove) (strategy (GameState.mk state.m state.n bobMove)))

/-- Theorem stating that Alice has a winning strategy. -/
theorem alice_wins (m n : Nat) : AliceHasWinningStrategy m n := by
  sorry


end NUMINAMATH_CALUDE_alice_wins_l3489_348938


namespace NUMINAMATH_CALUDE_completing_square_transformation_l3489_348912

/-- Given a quadratic equation x^2 - 2x - 4 = 0, prove that when transformed
    into the form (x-1)^2 = a using the completing the square method,
    the value of a is 5. -/
theorem completing_square_transformation (x : ℝ) :
  (x^2 - 2*x - 4 = 0) → ∃ a : ℝ, ((x - 1)^2 = a) ∧ (a = 5) := by
  sorry

end NUMINAMATH_CALUDE_completing_square_transformation_l3489_348912


namespace NUMINAMATH_CALUDE_max_true_statements_l3489_348982

theorem max_true_statements (x : ℝ) : 
  let statements := [
    (0 < x^3 ∧ x^3 < 1),
    (x^3 > 1),
    (-1 < x ∧ x < 0),
    (0 < x ∧ x < 1),
    (0 < x - x^3 ∧ x - x^3 < 1),
    (x^3 - x > 1)
  ]
  ∃ (true_statements : List Bool), 
    (∀ i, true_statements.get! i = true → statements.get! i) ∧
    true_statements.count true ≤ 3 ∧
    ∀ (other_true_statements : List Bool),
      (∀ i, other_true_statements.get! i = true → statements.get! i) →
      other_true_statements.count true ≤ true_statements.count true :=
by sorry


end NUMINAMATH_CALUDE_max_true_statements_l3489_348982


namespace NUMINAMATH_CALUDE_triangle_side_length_l3489_348944

theorem triangle_side_length (a b c : ℝ) (A B C : ℝ) :
  a = 2 →
  B = π / 6 →
  c = 2 * Real.sqrt 3 →
  b^2 = a^2 + c^2 - 2*a*c*(Real.cos B) →
  b = 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l3489_348944


namespace NUMINAMATH_CALUDE_max_expected_score_l3489_348986

/-- Xiao Zhang's box configuration -/
structure BoxConfig where
  red : ℕ
  yellow : ℕ
  white : ℕ
  sum_six : red + yellow + white = 6

/-- Expected score for a given box configuration -/
def expectedScore (config : BoxConfig) : ℚ :=
  (3 * config.red + 4 * config.yellow + 3 * config.white) / 36

/-- Theorem stating the maximum expected score and optimal configuration -/
theorem max_expected_score :
  ∃ (config : BoxConfig),
    expectedScore config = 2/3 ∧
    ∀ (other : BoxConfig), expectedScore other ≤ expectedScore config ∧
    config.red = 0 ∧ config.yellow = 6 ∧ config.white = 0 := by
  sorry


end NUMINAMATH_CALUDE_max_expected_score_l3489_348986


namespace NUMINAMATH_CALUDE_lcm_12_18_l3489_348973

theorem lcm_12_18 : Nat.lcm 12 18 = 36 := by
  sorry

end NUMINAMATH_CALUDE_lcm_12_18_l3489_348973


namespace NUMINAMATH_CALUDE_cake_eaten_percentage_l3489_348956

theorem cake_eaten_percentage (total_pieces : ℕ) (sisters : ℕ) (pieces_per_sister : ℕ) 
  (h1 : total_pieces = 240)
  (h2 : sisters = 3)
  (h3 : pieces_per_sister = 32) :
  (total_pieces - sisters * pieces_per_sister) / total_pieces * 100 = 60 := by
  sorry

#check cake_eaten_percentage

end NUMINAMATH_CALUDE_cake_eaten_percentage_l3489_348956


namespace NUMINAMATH_CALUDE_min_value_of_exponential_sum_l3489_348967

theorem min_value_of_exponential_sum (x y : ℝ) (h : x + y = 3) :
  2^x + 2^y ≥ 4 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_exponential_sum_l3489_348967


namespace NUMINAMATH_CALUDE_specific_hexagon_area_l3489_348940

/-- A hexagon formed by attaching six isosceles triangles to a central rectangle -/
structure Hexagon where
  /-- The base length of each isosceles triangle -/
  triangle_base : ℝ
  /-- The height of each isosceles triangle -/
  triangle_height : ℝ
  /-- The length of the central rectangle -/
  rectangle_length : ℝ
  /-- The width of the central rectangle -/
  rectangle_width : ℝ

/-- Calculate the area of the hexagon -/
def hexagon_area (h : Hexagon) : ℝ :=
  6 * (0.5 * h.triangle_base * h.triangle_height) + h.rectangle_length * h.rectangle_width

/-- Theorem stating that the area of the specific hexagon is 20 square units -/
theorem specific_hexagon_area :
  let h : Hexagon := {
    triangle_base := 2,
    triangle_height := 2,
    rectangle_length := 4,
    rectangle_width := 2
  }
  hexagon_area h = 20 := by sorry

end NUMINAMATH_CALUDE_specific_hexagon_area_l3489_348940


namespace NUMINAMATH_CALUDE_club_count_l3489_348952

theorem club_count (total : ℕ) (black : ℕ) (red : ℕ) (spades : ℕ) (diamonds : ℕ) (hearts : ℕ) (clubs : ℕ) :
  total = 13 →
  black = 7 →
  red = 6 →
  diamonds = 2 * spades →
  hearts = 2 * diamonds →
  total = spades + diamonds + hearts + clubs →
  black = spades + clubs →
  red = diamonds + hearts →
  clubs = 6 := by
sorry

end NUMINAMATH_CALUDE_club_count_l3489_348952


namespace NUMINAMATH_CALUDE_gauss_1998_cycle_l3489_348983

def word_length : Nat := 5
def number_length : Nat := 4

theorem gauss_1998_cycle : Nat.lcm word_length number_length = 20 := by
  sorry

end NUMINAMATH_CALUDE_gauss_1998_cycle_l3489_348983


namespace NUMINAMATH_CALUDE_function_properties_l3489_348913

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x + a * x

theorem function_properties (a : ℝ) (x₁ x₂ : ℝ) 
  (ha : a < -Real.exp 1) 
  (hx : x₁ < x₂) 
  (hf₁ : f a x₁ = 0) 
  (hf₂ : f a x₂ = 0) :
  let t := Real.sqrt (x₂ / x₁)
  (deriv (f a)) ((3 * x₁ + x₂) / 4) < 0 ∧ 
  (t - 1) * (a + Real.sqrt 3) = -2 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_function_properties_l3489_348913


namespace NUMINAMATH_CALUDE_min_value_abc_l3489_348989

theorem min_value_abc (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h : a * b * c = 4 * (a + b)) : 
  a + b + c ≥ 8 ∧ (a + b + c = 8 ↔ a = 2 ∧ b = 2) :=
by sorry

end NUMINAMATH_CALUDE_min_value_abc_l3489_348989


namespace NUMINAMATH_CALUDE_mark_baking_time_l3489_348972

/-- The time Mark spends baking bread -/
def baking_time (total_time rising_time kneading_time : ℕ) : ℕ :=
  total_time - (2 * rising_time + kneading_time)

/-- Theorem stating that Mark spends 30 minutes baking bread -/
theorem mark_baking_time :
  baking_time 280 120 10 = 30 := by
  sorry

end NUMINAMATH_CALUDE_mark_baking_time_l3489_348972


namespace NUMINAMATH_CALUDE_distinct_collections_count_l3489_348987

/-- Represents the count of each letter in ALGEBRAICS --/
structure LetterCount where
  a : Nat
  b : Nat
  c : Nat
  e : Nat
  g : Nat
  i : Nat
  l : Nat
  r : Nat
  s : Nat

/-- The initial count of letters in ALGEBRAICS --/
def initialCount : LetterCount :=
  { a := 2, b := 1, c := 1, e := 1, g := 1, i := 1, l := 1, r := 1, s := 1 }

/-- Counts the number of distinct collections of two vowels and two consonants --/
def countDistinctCollections (count : LetterCount) : Nat :=
  sorry

theorem distinct_collections_count :
  countDistinctCollections initialCount = 68 :=
sorry

end NUMINAMATH_CALUDE_distinct_collections_count_l3489_348987


namespace NUMINAMATH_CALUDE_matrix_power_difference_l3489_348930

theorem matrix_power_difference (B : Matrix (Fin 2) (Fin 2) ℝ) 
  (h : B = ![![2, 4], ![0, 1]]) : 
  B^20 - 3 • B^19 = ![![-1, 4], ![0, -2]] := by
  sorry

end NUMINAMATH_CALUDE_matrix_power_difference_l3489_348930


namespace NUMINAMATH_CALUDE_elephant_weighing_l3489_348963

/-- The weight of a stone block in catties -/
def stone_weight : ℕ := 240

/-- The number of stone blocks initially on the boat -/
def initial_stones : ℕ := 20

/-- The number of workers initially on the boat -/
def initial_workers : ℕ := 3

/-- The number of stone blocks after adjustment -/
def adjusted_stones : ℕ := 21

/-- The number of workers after adjustment -/
def adjusted_workers : ℕ := 1

/-- The weight of the elephant in catties -/
def elephant_weight : ℕ := 5160

theorem elephant_weighing :
  ∃ (worker_weight : ℕ),
    (initial_stones * stone_weight + initial_workers * worker_weight =
     adjusted_stones * stone_weight + adjusted_workers * worker_weight) ∧
    (elephant_weight = initial_stones * stone_weight + initial_workers * worker_weight) :=
by sorry

end NUMINAMATH_CALUDE_elephant_weighing_l3489_348963


namespace NUMINAMATH_CALUDE_log_625_squared_base_5_l3489_348964

-- Define the logarithm function
noncomputable def log (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

-- State the theorem
theorem log_625_squared_base_5 : log 5 (625^2) = 8 := by
  sorry

end NUMINAMATH_CALUDE_log_625_squared_base_5_l3489_348964


namespace NUMINAMATH_CALUDE_same_terminal_side_330_neg_30_l3489_348953

/-- Two angles have the same terminal side if they differ by a multiple of 360° -/
def same_terminal_side (α β : ℝ) : Prop :=
  ∃ k : ℤ, α = β + k * 360

/-- The angle -30° -/
def angle_neg_30 : ℝ := -30

/-- The angle 330° -/
def angle_330 : ℝ := 330

/-- Theorem: 330° has the same terminal side as -30° -/
theorem same_terminal_side_330_neg_30 :
  same_terminal_side angle_330 angle_neg_30 := by
  sorry

end NUMINAMATH_CALUDE_same_terminal_side_330_neg_30_l3489_348953


namespace NUMINAMATH_CALUDE_gold_coin_percentage_is_55_25_percent_l3489_348917

/-- Represents the composition of objects in an urn -/
structure UrnComposition where
  bead_percent : Real
  gold_coin_percent_of_coins : Real

/-- Calculates the percentage of gold coins in the urn -/
def gold_coin_percent (urn : UrnComposition) : Real :=
  (1 - urn.bead_percent) * urn.gold_coin_percent_of_coins

/-- Theorem stating that the percentage of gold coins in the urn is 55.25% -/
theorem gold_coin_percentage_is_55_25_percent 
  (urn : UrnComposition) 
  (h1 : urn.bead_percent = 0.15) 
  (h2 : urn.gold_coin_percent_of_coins = 0.65) : 
  gold_coin_percent urn = 0.5525 := by
  sorry

#eval gold_coin_percent { bead_percent := 0.15, gold_coin_percent_of_coins := 0.65 }

end NUMINAMATH_CALUDE_gold_coin_percentage_is_55_25_percent_l3489_348917


namespace NUMINAMATH_CALUDE_three_digit_number_problem_l3489_348921

-- Define the structure of a three-digit number
structure ThreeDigitNumber where
  hundreds : Nat
  tens : Nat
  units : Nat
  h_hundreds : hundreds < 10
  h_tens : tens < 10
  h_units : units < 10

def ThreeDigitNumber.toNat (n : ThreeDigitNumber) : Nat :=
  100 * n.hundreds + 10 * n.tens + n.units

def ThreeDigitNumber.reverse (n : ThreeDigitNumber) : Nat :=
  100 * n.units + 10 * n.tens + n.hundreds

def ThreeDigitNumber.sumOfDigits (n : ThreeDigitNumber) : Nat :=
  n.hundreds + n.tens + n.units

theorem three_digit_number_problem (n : ThreeDigitNumber) : 
  n.toNat = 253 → 
  n.sumOfDigits = 10 ∧ 
  n.tens = n.hundreds + n.units ∧ 
  n.reverse = n.toNat + 99 := by
  sorry

#eval ThreeDigitNumber.toNat ⟨2, 5, 3, by norm_num, by norm_num, by norm_num⟩

end NUMINAMATH_CALUDE_three_digit_number_problem_l3489_348921


namespace NUMINAMATH_CALUDE_total_students_total_students_is_144_l3489_348991

/-- The total number of students in all halls combined, given the specified conditions. -/
theorem total_students (general : ℕ) (biology : ℕ) (math : ℕ) : ℕ :=
  general + biology + math
  where
  general := 30
  biology := 2 * general
  math := (3 * (general + biology)) / 5

/-- Proof that the total number of students in all halls is 144. -/
theorem total_students_is_144 : total_students 30 (2 * 30) ((3 * (30 + 2 * 30)) / 5) = 144 := by
  sorry

end NUMINAMATH_CALUDE_total_students_total_students_is_144_l3489_348991


namespace NUMINAMATH_CALUDE_impossibility_theorem_l3489_348994

/-- Represents the number of boxes -/
def n : ℕ := 100

/-- Represents the initial number of stones in each box -/
def initial_stones (i : ℕ) : ℕ := i

/-- Represents the condition for moving stones between boxes -/
def can_move (a b : ℕ) : Prop := a + b = 101

/-- Represents the desired final configuration -/
def desired_config (stones : ℕ → ℕ) : Prop :=
  stones 70 = 69 ∧ stones 50 = 51 ∧ 
  ∀ i, i ≠ 70 ∧ i ≠ 50 → stones i = initial_stones i

/-- Main theorem: It's impossible to achieve the desired configuration -/
theorem impossibility_theorem :
  ¬ ∃ (stones : ℕ → ℕ), 
    (∀ i j, i ≠ j → can_move (stones i) (stones j) → 
      ∃ k l, k ≠ l ∧ stones k + stones l = 101) ∧
    desired_config stones :=
sorry

end NUMINAMATH_CALUDE_impossibility_theorem_l3489_348994


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocals_l3489_348975

-- Define the vectors
def OA : ℝ × ℝ := (-2, 4)
def OB (a : ℝ) : ℝ × ℝ := (-a, 2)
def OC (b : ℝ) : ℝ × ℝ := (b, 0)

-- Define the collinearity condition
def collinear (A B C : ℝ × ℝ) : Prop :=
  ∃ (t : ℝ), B.1 - A.1 = t * (C.1 - A.1) ∧ B.2 - A.2 = t * (C.2 - A.2)

-- State the theorem
theorem min_value_sum_reciprocals (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_collinear : collinear OA (OB a) (OC b)) :
  (1 / a + 1 / b) ≥ (3 + 2 * Real.sqrt 2) / 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_sum_reciprocals_l3489_348975


namespace NUMINAMATH_CALUDE_athletes_meeting_time_l3489_348923

/-- 
Given two athletes running on a closed track:
- Their speeds are constant
- The first athlete completes the track 'a' seconds faster than the second
- They meet every 'b' seconds when running in the same direction

This theorem proves that the time it takes for them to meet when running 
in opposite directions is (a * b) / sqrt(a^2 + 4*a*b) seconds.
-/
theorem athletes_meeting_time (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let t := (a * b) / Real.sqrt (a^2 + 4*a*b)
  ∃ (x y z : ℝ), 
    x > 0 ∧ y > 0 ∧ z > 0 ∧  -- positive speeds and track length
    z / y - z / x = a ∧     -- first athlete is 'a' seconds faster
    z / (x - y) = b ∧       -- they meet every 'b' seconds in same direction
    z / (x + y) = t         -- time to meet in opposite directions
    := by sorry

end NUMINAMATH_CALUDE_athletes_meeting_time_l3489_348923


namespace NUMINAMATH_CALUDE_quadratic_roots_distinct_l3489_348909

theorem quadratic_roots_distinct (m : ℝ) : 
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ x₁^2 + m*x₁ - 8 = 0 ∧ x₂^2 + m*x₂ - 8 = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_distinct_l3489_348909


namespace NUMINAMATH_CALUDE_rugby_team_lineup_count_l3489_348957

/-- The number of ways to form a team lineup -/
def team_lineup_ways (total_members : ℕ) (specialized_kickers : ℕ) (lineup_size : ℕ) : ℕ :=
  specialized_kickers * (Nat.choose (total_members - 1) (lineup_size - 1))

/-- Theorem: The number of ways to form the team lineup is 151164 -/
theorem rugby_team_lineup_count :
  team_lineup_ways 20 2 9 = 151164 := by
  sorry

end NUMINAMATH_CALUDE_rugby_team_lineup_count_l3489_348957


namespace NUMINAMATH_CALUDE_magnitude_of_b_l3489_348959

def vector_a : ℝ × ℝ := (1, -2)
def vector_sum : ℝ × ℝ := (0, 2)

def vector_b : ℝ × ℝ := (vector_sum.1 - vector_a.1, vector_sum.2 - vector_a.2)

theorem magnitude_of_b : Real.sqrt ((vector_b.1)^2 + (vector_b.2)^2) = Real.sqrt 17 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_b_l3489_348959


namespace NUMINAMATH_CALUDE_expression_simplification_l3489_348949

variables (x y a b : ℝ)

theorem expression_simplification :
  (-3*x + 2*y - 5*x - 7*y = -8*x - 5*y) ∧
  (5*(3*a^2*b - a*b^2) - 4*(-a*b^2 + 3*a^2*b) = 3*a^2*b - a*b^2) := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3489_348949


namespace NUMINAMATH_CALUDE_exists_eight_numbers_sum_divisible_l3489_348928

theorem exists_eight_numbers_sum_divisible : 
  ∃ (S : Finset ℕ), 
    S.card = 8 ∧ 
    (∀ n ∈ S, n ≤ 100) ∧
    (∀ n ∈ S, (S.sum id) % n = 0) :=
sorry

end NUMINAMATH_CALUDE_exists_eight_numbers_sum_divisible_l3489_348928


namespace NUMINAMATH_CALUDE_rectangular_hall_dimension_difference_l3489_348985

theorem rectangular_hall_dimension_difference 
  (length width : ℝ) 
  (width_half_length : width = length / 2)
  (area_constraint : length * width = 578) :
  length - width = 17 := by
sorry

end NUMINAMATH_CALUDE_rectangular_hall_dimension_difference_l3489_348985


namespace NUMINAMATH_CALUDE_original_price_proof_l3489_348960

/-- The original price of a part before discount -/
def original_price : ℝ := 62.71

/-- The number of parts Clark bought -/
def num_parts : ℕ := 7

/-- The total amount Clark paid after discount -/
def total_paid : ℝ := 439

/-- Theorem stating that the original price multiplied by the number of parts equals the total amount paid -/
theorem original_price_proof : original_price * num_parts = total_paid := by sorry

end NUMINAMATH_CALUDE_original_price_proof_l3489_348960


namespace NUMINAMATH_CALUDE_isosceles_trapezoid_area_l3489_348920

/-- The area of an isosceles trapezoid with given dimensions -/
theorem isosceles_trapezoid_area (side : ℝ) (base1 base2 : ℝ) :
  side > 0 ∧ base1 > 0 ∧ base2 > 0 ∧ base1 < base2 ∧ side^2 > ((base2 - base1)/2)^2 →
  let height := Real.sqrt (side^2 - ((base2 - base1)/2)^2)
  (1/2 : ℝ) * (base1 + base2) * height = 48 ∧ side = 5 ∧ base1 = 9 ∧ base2 = 15 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_trapezoid_area_l3489_348920


namespace NUMINAMATH_CALUDE_factory_production_l3489_348977

/-- Represents the number of toys produced in a week at a factory -/
def toysPerWeek (daysWorked : ℕ) (toysPerDay : ℕ) : ℕ :=
  daysWorked * toysPerDay

/-- Theorem stating that the factory produces 4560 toys per week -/
theorem factory_production :
  toysPerWeek 4 1140 = 4560 := by
  sorry

end NUMINAMATH_CALUDE_factory_production_l3489_348977


namespace NUMINAMATH_CALUDE_unique_solution_l3489_348935

/-- Represents a digit (0-9) -/
def Digit := Fin 10

/-- Represents the equation TETA + BETA = GAMMA -/
def EquationSatisfied (T E B G M A : Digit) : Prop :=
  1000 * T.val + 100 * E.val + 10 * T.val + A.val +
  1000 * B.val + 100 * E.val + 10 * T.val + A.val =
  10000 * G.val + 1000 * A.val + 100 * M.val + 10 * M.val + A.val

/-- All digits are different except for repeated letters -/
def DigitsDifferent (T E B G M A : Digit) : Prop :=
  T ≠ E ∧ T ≠ B ∧ T ≠ G ∧ T ≠ M ∧ T ≠ A ∧
  E ≠ B ∧ E ≠ G ∧ E ≠ M ∧ E ≠ A ∧
  B ≠ G ∧ B ≠ M ∧ B ≠ A ∧
  G ≠ M ∧ G ≠ A ∧
  M ≠ A

theorem unique_solution :
  ∃! (T E B G M A : Digit),
    EquationSatisfied T E B G M A ∧
    DigitsDifferent T E B G M A ∧
    T.val = 4 ∧ E.val = 9 ∧ B.val = 5 ∧ G.val = 1 ∧ M.val = 8 ∧ A.val = 0 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_l3489_348935


namespace NUMINAMATH_CALUDE_room_length_calculation_l3489_348948

/-- Given a room with specified width, total paving cost, and paving rate per square meter,
    calculate the length of the room. -/
theorem room_length_calculation (width : ℝ) (total_cost : ℝ) (rate_per_sqm : ℝ) :
  width = 3.75 →
  total_cost = 6187.5 →
  rate_per_sqm = 300 →
  (total_cost / rate_per_sqm) / width = 5.5 :=
by
  sorry

end NUMINAMATH_CALUDE_room_length_calculation_l3489_348948


namespace NUMINAMATH_CALUDE_banana_arrangements_eq_60_l3489_348937

def banana_arrangements : ℕ :=
  let total_letters : ℕ := 6
  let b_count : ℕ := 1
  let n_count : ℕ := 2
  let a_count : ℕ := 3
  Nat.factorial total_letters / (Nat.factorial b_count * Nat.factorial n_count * Nat.factorial a_count)

theorem banana_arrangements_eq_60 : banana_arrangements = 60 := by
  sorry

end NUMINAMATH_CALUDE_banana_arrangements_eq_60_l3489_348937


namespace NUMINAMATH_CALUDE_green_shirt_pairs_l3489_348902

theorem green_shirt_pairs (red_students green_students total_students total_pairs red_red_pairs : ℕ)
  (h1 : red_students = 63)
  (h2 : green_students = 69)
  (h3 : total_students = red_students + green_students)
  (h4 : total_pairs = 66)
  (h5 : red_red_pairs = 25)
  (h6 : total_students = 2 * total_pairs) :
  2 * red_red_pairs + (red_students - 2 * red_red_pairs) + 2 * ((green_students - (red_students - 2 * red_red_pairs)) / 2) = total_students :=
by sorry

end NUMINAMATH_CALUDE_green_shirt_pairs_l3489_348902


namespace NUMINAMATH_CALUDE_min_reciprocal_sum_l3489_348936

theorem min_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  (∀ a b : ℝ, a > 0 → b > 0 → a + b = 1 → 1/x + 1/y ≤ 1/a + 1/b) ∧ (1/x + 1/y = 4 → x = 1/2 ∧ y = 1/2) :=
by sorry

end NUMINAMATH_CALUDE_min_reciprocal_sum_l3489_348936


namespace NUMINAMATH_CALUDE_remainder_proof_l3489_348932

theorem remainder_proof (R1 : ℕ) : 
  (129 = Nat.gcd (1428 - R1) (2206 - 13)) → 
  (2206 % 129 = 13) → 
  (1428 % 129 = 19) :=
by
  sorry

end NUMINAMATH_CALUDE_remainder_proof_l3489_348932


namespace NUMINAMATH_CALUDE_light_glow_theorem_l3489_348961

/-- The number of times a light glows in a given time interval -/
def glowCount (interval : ℕ) (period : ℕ) : ℕ :=
  interval / period

/-- The number of times all lights glow simultaneously in a given time interval -/
def simultaneousGlowCount (interval : ℕ) (periodA periodB periodC : ℕ) : ℕ :=
  interval / (lcm (lcm periodA periodB) periodC)

theorem light_glow_theorem (totalInterval : ℕ) (periodA periodB periodC : ℕ)
    (h1 : totalInterval = 4969)
    (h2 : periodA = 18)
    (h3 : periodB = 24)
    (h4 : periodC = 30) :
    glowCount totalInterval periodA = 276 ∧
    glowCount totalInterval periodB = 207 ∧
    glowCount totalInterval periodC = 165 ∧
    simultaneousGlowCount totalInterval periodA periodB periodC = 13 := by
  sorry

end NUMINAMATH_CALUDE_light_glow_theorem_l3489_348961


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l3489_348907

def I : Set ℕ := Set.univ
def A : Set ℕ := {1,2,3,4,5,6}
def B : Set ℕ := {2,3,5}

theorem complement_intersection_theorem :
  (I \ B) ∩ A = {1,4,6} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l3489_348907


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l3489_348965

theorem complex_modulus_problem (a : ℝ) (i : ℂ) : 
  i ^ 2 = -1 →
  (Complex.I : ℂ) ^ 2 = -1 →
  ((a - Real.sqrt 2 + i) / i).im = 0 →
  Complex.abs (2 * a + Complex.I * Real.sqrt 2) = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l3489_348965


namespace NUMINAMATH_CALUDE_complex_number_condition_l3489_348976

/-- A complex number z satisfying the given conditions -/
def Z : ℂ := sorry

/-- The real part of Z -/
def m : ℝ := Z.re

/-- The imaginary part of Z -/
def n : ℝ := Z.im

/-- The condition that z+2i is a real number -/
axiom h1 : (Z + 2*Complex.I).im = 0

/-- The condition that z/(2-i) is a real number -/
axiom h2 : ((Z / (2 - Complex.I))).im = 0

/-- Definition of the function representing (z+ai)^2 -/
def f (a : ℝ) : ℂ := (Z + a*Complex.I)^2

/-- The theorem to be proved -/
theorem complex_number_condition (a : ℝ) :
  (f a).re < 0 ∧ (f a).im > 0 → a > 6 := by sorry

end NUMINAMATH_CALUDE_complex_number_condition_l3489_348976


namespace NUMINAMATH_CALUDE_minimum_concerts_required_l3489_348955

/-- Represents a concert configuration --/
structure Concert where
  performers : Finset Nat
  listeners : Finset Nat

/-- Represents the festival configuration --/
structure Festival where
  musicians : Finset Nat
  concerts : List Concert

/-- Checks if a festival configuration is valid --/
def isValidFestival (f : Festival) : Prop :=
  f.musicians.card = 6 ∧
  ∀ c ∈ f.concerts, c.performers ⊆ f.musicians ∧
                    c.listeners ⊆ f.musicians ∧
                    c.performers ∩ c.listeners = ∅ ∧
                    c.performers ∪ c.listeners = f.musicians

/-- Checks if each musician has listened to all others --/
def allMusiciansListened (f : Festival) : Prop :=
  ∀ m ∈ f.musicians, ∀ n ∈ f.musicians, m ≠ n →
    ∃ c ∈ f.concerts, m ∈ c.listeners ∧ n ∈ c.performers

/-- The main theorem --/
theorem minimum_concerts_required :
  ∀ f : Festival,
    isValidFestival f →
    allMusiciansListened f →
    f.concerts.length ≥ 4 :=
by sorry

end NUMINAMATH_CALUDE_minimum_concerts_required_l3489_348955


namespace NUMINAMATH_CALUDE_farmer_loss_l3489_348996

/-- Represents the total weight of onions in pounds -/
def total_weight : ℝ := 100

/-- Represents the market price per pound of onions in dollars -/
def market_price : ℝ := 3

/-- Represents the dealer's price per pound for both leaves and whites in dollars -/
def dealer_price : ℝ := 1.5

/-- Theorem stating the farmer's loss -/
theorem farmer_loss : 
  total_weight * market_price - total_weight * dealer_price = 150 := by
  sorry

end NUMINAMATH_CALUDE_farmer_loss_l3489_348996


namespace NUMINAMATH_CALUDE_circle_radius_problem_l3489_348926

theorem circle_radius_problem (circle_A circle_B : ℝ) : 
  circle_A = 4 * circle_B →  -- Radius of A is 4 times radius of B
  2 * circle_A = 80 →        -- Diameter of A is 80 cm
  circle_B = 10 := by        -- Radius of B is 10 cm
sorry

end NUMINAMATH_CALUDE_circle_radius_problem_l3489_348926


namespace NUMINAMATH_CALUDE_vector_equality_implies_m_equals_two_l3489_348968

def a (m : ℝ) : ℝ × ℝ := (m, -2)
def b : ℝ × ℝ := (1, 1)

theorem vector_equality_implies_m_equals_two (m : ℝ) :
  ‖a m - b‖ = ‖a m + b‖ → m = 2 := by sorry

end NUMINAMATH_CALUDE_vector_equality_implies_m_equals_two_l3489_348968


namespace NUMINAMATH_CALUDE_chocolate_cost_proof_l3489_348918

def candy_bar_cost : ℝ := 2
def chocolate_cost_difference : ℝ := 1

theorem chocolate_cost_proof :
  let chocolate_cost := candy_bar_cost + chocolate_cost_difference
  chocolate_cost = 3 := by sorry

end NUMINAMATH_CALUDE_chocolate_cost_proof_l3489_348918


namespace NUMINAMATH_CALUDE_volume_ratio_l3489_348992

variable (V_A V_B V_C : ℝ)

theorem volume_ratio 
  (h1 : V_A = (V_B + V_C) / 2)
  (h2 : V_B = (V_A + V_C) / 5)
  (h3 : V_C ≠ 0) :
  V_C / (V_A + V_B) = 1 := by
sorry

end NUMINAMATH_CALUDE_volume_ratio_l3489_348992


namespace NUMINAMATH_CALUDE_circle_intersection_problem_l3489_348929

theorem circle_intersection_problem (k : ℝ) :
  let center : ℝ × ℝ := ((27 - 3) / 2 + -3, 0)
  let radius : ℝ := (27 - (-3)) / 2
  let circle_equation (x y : ℝ) : Prop := (x - center.1)^2 + (y - center.2)^2 = radius^2
  (∃ y₁ y₂ : ℝ, y₁ ≠ y₂ ∧ circle_equation k y₁ ∧ circle_equation k y₂) →
  (∃ y : ℝ, circle_equation k y ∧ y = 12) →
  k = 3 ∨ k = 21 :=
by sorry


end NUMINAMATH_CALUDE_circle_intersection_problem_l3489_348929


namespace NUMINAMATH_CALUDE_smallest_all_blue_count_l3489_348931

/-- Represents the colors of chameleons -/
inductive Color
| Red
| C2
| C3
| C4
| Blue

/-- Represents the result of a bite interaction between two chameleons -/
def bite_result (biter bitten : Color) : Color :=
  match biter, bitten with
  | Color.Red, Color.Red => Color.C2
  | Color.Red, Color.C2 => Color.C3
  | Color.Red, Color.C3 => Color.C4
  | Color.Red, Color.C4 => Color.Blue
  | Color.C2, Color.Red => Color.C2
  | Color.C3, Color.Red => Color.C3
  | Color.C4, Color.Red => Color.C4
  | Color.Blue, Color.Red => Color.Blue
  | _, Color.Blue => Color.Blue
  | _, _ => bitten  -- For all other cases, no color change

/-- A sequence of bites that transforms all chameleons to blue -/
def all_blue_sequence (n : ℕ) : List (Fin n × Fin n) → Prop := sorry

/-- The theorem stating that 5 is the smallest number of red chameleons that can guarantee becoming all blue -/
theorem smallest_all_blue_count :
  (∃ (seq : List (Fin 5 × Fin 5)), all_blue_sequence 5 seq) ∧
  (∀ k < 5, ¬∃ (seq : List (Fin k × Fin k)), all_blue_sequence k seq) :=
sorry

end NUMINAMATH_CALUDE_smallest_all_blue_count_l3489_348931


namespace NUMINAMATH_CALUDE_brads_red_balloons_l3489_348915

/-- Given that Brad has a total of 17 balloons and 9 of them are green,
    prove that he has 8 red balloons. -/
theorem brads_red_balloons (total : ℕ) (green : ℕ) (h1 : total = 17) (h2 : green = 9) :
  total - green = 8 := by
  sorry

end NUMINAMATH_CALUDE_brads_red_balloons_l3489_348915


namespace NUMINAMATH_CALUDE_f_leq_two_iff_x_leq_eight_l3489_348934

noncomputable def f (x : ℝ) : ℝ :=
  if x < 1 then Real.exp (x - 1) else x ^ (1/3 : ℝ)

theorem f_leq_two_iff_x_leq_eight :
  ∀ x : ℝ, f x ≤ 2 ↔ x ≤ 8 :=
sorry

end NUMINAMATH_CALUDE_f_leq_two_iff_x_leq_eight_l3489_348934


namespace NUMINAMATH_CALUDE_cosine_graph_shift_l3489_348993

theorem cosine_graph_shift (x : ℝ) :
  4 * Real.cos (2 * (x - π/8) + π/4) = 4 * Real.cos (2 * x) := by
  sorry

end NUMINAMATH_CALUDE_cosine_graph_shift_l3489_348993


namespace NUMINAMATH_CALUDE_hexagonal_grid_consecutive_circles_l3489_348941

/-- Represents a hexagonal grid of circles -/
structure HexagonalGrid :=
  (num_circles : ℕ)

/-- Counts the number of ways to choose 3 consecutive circles in a row -/
def count_horizontal_ways (grid : HexagonalGrid) : ℕ :=
  (1 + 2 + 3 + 4 + 5 + 6)

/-- Counts the number of ways to choose 3 consecutive circles in one diagonal direction -/
def count_diagonal_ways (grid : HexagonalGrid) : ℕ :=
  (4 + 4 + 4 + 3 + 2 + 1)

/-- Counts the total number of ways to choose 3 consecutive circles in all directions -/
def count_total_ways (grid : HexagonalGrid) : ℕ :=
  count_horizontal_ways grid + 2 * count_diagonal_ways grid

/-- Theorem: The total number of ways to choose 3 consecutive circles in a hexagonal grid of 33 circles is 57 -/
theorem hexagonal_grid_consecutive_circles (grid : HexagonalGrid) 
  (h : grid.num_circles = 33) : count_total_ways grid = 57 := by
  sorry

end NUMINAMATH_CALUDE_hexagonal_grid_consecutive_circles_l3489_348941


namespace NUMINAMATH_CALUDE_subtract_negative_five_l3489_348984

theorem subtract_negative_five : 2 - (-5) = 7 := by
  sorry

end NUMINAMATH_CALUDE_subtract_negative_five_l3489_348984


namespace NUMINAMATH_CALUDE_two_color_theorem_l3489_348943

-- Define a type for regions in the plane
def Region : Type := ℕ

-- Define a type for colors
inductive Color
| Red : Color
| Blue : Color

-- Define a function type for coloring the map
def Coloring := Region → Color

-- Define a relation for adjacent regions
def Adjacent : Region → Region → Prop := sorry

-- Define a property for a valid coloring
def ValidColoring (coloring : Coloring) : Prop :=
  ∀ r1 r2 : Region, Adjacent r1 r2 → coloring r1 ≠ coloring r2

-- Define a type for the map configuration
structure MapConfiguration :=
  (num_lines : ℕ)
  (num_circles : ℕ)

-- State the theorem
theorem two_color_theorem :
  ∀ (config : MapConfiguration), ∃ (coloring : Coloring), ValidColoring coloring :=
sorry

end NUMINAMATH_CALUDE_two_color_theorem_l3489_348943


namespace NUMINAMATH_CALUDE_broccoli_sales_amount_l3489_348979

def farmers_market_sales (broccoli_sales : ℝ) : Prop :=
  let carrot_sales := 2 * broccoli_sales
  let spinach_sales := carrot_sales / 2 + 16
  let cauliflower_sales := 136
  broccoli_sales + carrot_sales + spinach_sales + cauliflower_sales = 380

theorem broccoli_sales_amount : ∃ (x : ℝ), farmers_market_sales x ∧ x = 57 :=
  sorry

end NUMINAMATH_CALUDE_broccoli_sales_amount_l3489_348979


namespace NUMINAMATH_CALUDE_knights_and_knaves_solution_l3489_348911

-- Define the types for residents and their statements
inductive Resident : Type
| A
| B
| C

inductive Status : Type
| Knight
| Knave

-- Define the statement made by A
def statement_A (status : Resident → Status) : Prop :=
  status Resident.C = Status.Knight → status Resident.B = Status.Knave

-- Define the statement made by C
def statement_C (status : Resident → Status) : Prop :=
  status Resident.A ≠ status Resident.C ∧
  ((status Resident.A = Status.Knight ∧ status Resident.C = Status.Knave) ∨
   (status Resident.A = Status.Knave ∧ status Resident.C = Status.Knight))

-- Define the truthfulness of statements based on the speaker's status
def is_truthful (status : Resident → Status) (r : Resident) (stmt : Prop) : Prop :=
  (status r = Status.Knight ∧ stmt) ∨ (status r = Status.Knave ∧ ¬stmt)

-- Theorem stating the solution
theorem knights_and_knaves_solution :
  ∃ (status : Resident → Status),
    is_truthful status Resident.A (statement_A status) ∧
    is_truthful status Resident.C (statement_C status) ∧
    status Resident.A = Status.Knave ∧
    status Resident.B = Status.Knight ∧
    status Resident.C = Status.Knight :=
sorry

end NUMINAMATH_CALUDE_knights_and_knaves_solution_l3489_348911


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l3489_348904

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Theorem: In an arithmetic sequence, if a_3 + a_5 = 12, then a_4 = 6 -/
theorem arithmetic_sequence_property (a : ℕ → ℝ) 
    (h_arith : ArithmeticSequence a) (h_sum : a 3 + a 5 = 12) : 
    a 4 = 6 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l3489_348904


namespace NUMINAMATH_CALUDE_coloring_problem_l3489_348951

/-- Represents the number of objects colored by each person -/
def objects_per_person (total_colors : ℕ) (num_people : ℕ) : ℕ :=
  total_colors / num_people

/-- Represents the total number of objects colored -/
def total_objects (objects_per_person : ℕ) (num_people : ℕ) : ℕ :=
  objects_per_person * num_people

theorem coloring_problem (total_colors : ℕ) (num_people : ℕ) 
  (h1 : total_colors = 24) 
  (h2 : num_people = 3) :
  total_objects (objects_per_person total_colors num_people) num_people = 24 := by
sorry

end NUMINAMATH_CALUDE_coloring_problem_l3489_348951


namespace NUMINAMATH_CALUDE_problem_solution_l3489_348971

theorem problem_solution (a : ℝ) (h1 : a > 0) : 
  let f := fun x : ℝ => x^2 + 12
  let g := fun x : ℝ => x^2 - x - 4
  f (g a) = 12 → a = (1 + Real.sqrt 17) / 2 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l3489_348971


namespace NUMINAMATH_CALUDE_inequality_system_solution_l3489_348980

theorem inequality_system_solution (b : ℝ) : 
  (∀ (x y : ℝ), 2*b * Real.cos (2*(x-y)) + 8*b^2 * Real.cos (x-y) + 8*b^2*(b+1) + 5*b < 0 ∧
                 x^2 + y^2 + 1 > 2*b*x + 2*y + b - b^2) ↔ 
  (b < -1 - Real.sqrt 2 / 4 ∨ (-1/2 < b ∧ b < 0)) := by sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l3489_348980


namespace NUMINAMATH_CALUDE_inequality_theorem_l3489_348942

theorem inequality_theorem (x : ℝ) (n : ℕ) (h1 : x > 0) (h2 : n > 0) :
  x + (n^n : ℝ)/(x^n) ≥ n + 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_theorem_l3489_348942


namespace NUMINAMATH_CALUDE_fourth_term_is_54_l3489_348962

/-- A positive geometric sequence with specific properties -/
structure SpecialGeometricSequence where
  a : ℕ → ℝ
  is_positive : ∀ n, a n > 0
  is_geometric : ∃ q : ℝ, q > 0 ∧ ∀ n, a (n + 1) = a n * q
  first_term : a 1 = 2
  arithmetic_mean : a 2 + 4 = (a 1 + a 3) / 2

/-- The fourth term of the special geometric sequence is 54 -/
theorem fourth_term_is_54 (seq : SpecialGeometricSequence) : seq.a 4 = 54 := by
  sorry

end NUMINAMATH_CALUDE_fourth_term_is_54_l3489_348962


namespace NUMINAMATH_CALUDE_star_operations_l3489_348925

/-- The ☆ operation for rational numbers -/
def star (a b : ℚ) : ℚ := a * b - a + b

/-- Theorem stating the results of the given operations -/
theorem star_operations :
  (star 2 (-3) = -11) ∧ (star (-2) (star 1 3) = -3) := by
  sorry

end NUMINAMATH_CALUDE_star_operations_l3489_348925
