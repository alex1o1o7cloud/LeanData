import Mathlib

namespace NUMINAMATH_CALUDE_train_crossing_time_l1210_121083

/-- Calculates the time taken for a train to cross a platform -/
theorem train_crossing_time (train_length : ℝ) (signal_crossing_time : ℝ) (platform_length : ℝ) :
  train_length = 300 →
  signal_crossing_time = 18 →
  platform_length = 400 →
  ∃ (platform_crossing_time : ℝ), abs (platform_crossing_time - 42) < 0.1 := by
  sorry


end NUMINAMATH_CALUDE_train_crossing_time_l1210_121083


namespace NUMINAMATH_CALUDE_arithmetic_sequence_fifth_term_l1210_121008

/-- Given an arithmetic sequence with first term -1 and third term 5, prove that the fifth term is 11. -/
theorem arithmetic_sequence_fifth_term :
  ∀ (a : ℕ → ℤ), 
    (∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)) →  -- arithmetic sequence condition
    a 1 = -1 →                                        -- first term
    a 3 = 5 →                                         -- third term
    a 5 = 11 :=                                       -- fifth term (to prove)
by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_fifth_term_l1210_121008


namespace NUMINAMATH_CALUDE_bird_families_difference_l1210_121033

/-- Given the total number of bird families and the number that flew away,
    prove that the difference between those that stayed and those that flew away is 73. -/
theorem bird_families_difference (total : ℕ) (flew_away : ℕ) 
    (h1 : total = 87) (h2 : flew_away = 7) : total - flew_away - flew_away = 73 := by
  sorry

end NUMINAMATH_CALUDE_bird_families_difference_l1210_121033


namespace NUMINAMATH_CALUDE_reaction_weight_equality_l1210_121064

/-- Atomic weight of Calcium in g/mol -/
def Ca_weight : ℝ := 40.08

/-- Atomic weight of Bromine in g/mol -/
def Br_weight : ℝ := 79.904

/-- Atomic weight of Hydrogen in g/mol -/
def H_weight : ℝ := 1.008

/-- Atomic weight of Oxygen in g/mol -/
def O_weight : ℝ := 15.999

/-- Molecular weight of CaBr2 in g/mol -/
def CaBr2_weight : ℝ := Ca_weight + 2 * Br_weight

/-- Molecular weight of H2O in g/mol -/
def H2O_weight : ℝ := 2 * H_weight + O_weight

/-- Molecular weight of Ca(OH)2 in g/mol -/
def CaOH2_weight : ℝ := Ca_weight + 2 * (O_weight + H_weight)

/-- Molecular weight of HBr in g/mol -/
def HBr_weight : ℝ := H_weight + Br_weight

/-- Theorem stating that the molecular weight of reactants equals the molecular weight of products
    and is equal to 235.918 g/mol -/
theorem reaction_weight_equality :
  CaBr2_weight + 2 * H2O_weight = CaOH2_weight + 2 * HBr_weight ∧
  CaBr2_weight + 2 * H2O_weight = 235.918 :=
by sorry

end NUMINAMATH_CALUDE_reaction_weight_equality_l1210_121064


namespace NUMINAMATH_CALUDE_problem_solution_l1210_121019

theorem problem_solution : 2 * Real.sin (60 * π / 180) + |Real.sqrt 3 - 3| + (π - 1)^0 = 4 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1210_121019


namespace NUMINAMATH_CALUDE_orange_crates_pigeonhole_l1210_121090

theorem orange_crates_pigeonhole :
  ∀ (crate_contents : Fin 150 → ℕ),
  (∀ i, 130 ≤ crate_contents i ∧ crate_contents i ≤ 150) →
  ∃ n : ℕ, 130 ≤ n ∧ n ≤ 150 ∧ (Finset.filter (λ i => crate_contents i = n) Finset.univ).card ≥ 8 :=
by sorry

end NUMINAMATH_CALUDE_orange_crates_pigeonhole_l1210_121090


namespace NUMINAMATH_CALUDE_jason_initial_cards_l1210_121067

/-- The number of Pokemon cards Jason gave away -/
def cards_given_away : ℕ := 9

/-- The number of Pokemon cards Jason has left -/
def cards_left : ℕ := 4

/-- The initial number of Pokemon cards Jason had -/
def initial_cards : ℕ := cards_given_away + cards_left

theorem jason_initial_cards : initial_cards = 13 := by
  sorry

end NUMINAMATH_CALUDE_jason_initial_cards_l1210_121067


namespace NUMINAMATH_CALUDE_circle_through_three_points_l1210_121036

theorem circle_through_three_points :
  let A : ℝ × ℝ := (1, 12)
  let B : ℝ × ℝ := (7, 10)
  let C : ℝ × ℝ := (-9, 2)
  let circle_equation (x y : ℝ) := x^2 + y^2 - 2*x - 4*y - 95 = 0
  (circle_equation A.1 A.2) ∧ 
  (circle_equation B.1 B.2) ∧ 
  (circle_equation C.1 C.2) := by
sorry

end NUMINAMATH_CALUDE_circle_through_three_points_l1210_121036


namespace NUMINAMATH_CALUDE_watermelon_ratio_l1210_121016

theorem watermelon_ratio (michael_weight john_weight : ℚ) : 
  michael_weight = 8 →
  john_weight = 12 →
  john_weight / (3 * michael_weight) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_watermelon_ratio_l1210_121016


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l1210_121018

theorem min_value_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + 3 * b = 1) :
  (1 / a + 1 / b) ≥ 8 + 4 * Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l1210_121018


namespace NUMINAMATH_CALUDE_expression_evaluation_l1210_121099

theorem expression_evaluation : 
  1 / (2 - Real.sqrt 3) - Real.pi ^ 0 - 2 * Real.cos (30 * π / 180) = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1210_121099


namespace NUMINAMATH_CALUDE_simplify_polynomial_l1210_121034

/-- Proves that the simplified form of (9x^9+7x^8+4x^7) + (x^11+x^9+2x^7+3x^3+5x+8)
    is x^11+10x^9+7x^8+6x^7+3x^3+5x+8 -/
theorem simplify_polynomial (x : ℝ) :
  (9 * x^9 + 7 * x^8 + 4 * x^7) + (x^11 + x^9 + 2 * x^7 + 3 * x^3 + 5 * x + 8) =
  x^11 + 10 * x^9 + 7 * x^8 + 6 * x^7 + 3 * x^3 + 5 * x + 8 := by
  sorry

#check simplify_polynomial

end NUMINAMATH_CALUDE_simplify_polynomial_l1210_121034


namespace NUMINAMATH_CALUDE_train_length_calculation_l1210_121012

/-- Calculates the length of a train given the length of another train, their speeds, and the time they take to cross each other when traveling in opposite directions. -/
theorem train_length_calculation (length1 : ℝ) (speed1 speed2 : ℝ) (cross_time : ℝ) :
  length1 = 270 →
  speed1 = 120 →
  speed2 = 80 →
  cross_time = 9 →
  ∃ (length2 : ℝ), abs (length2 - 230.04) < 0.01 :=
by
  sorry

end NUMINAMATH_CALUDE_train_length_calculation_l1210_121012


namespace NUMINAMATH_CALUDE_factorization_equality_l1210_121081

theorem factorization_equality (a b : ℝ) : 12 * b^3 - 3 * a^2 * b = 3 * b * (2*b + a) * (2*b - a) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l1210_121081


namespace NUMINAMATH_CALUDE_lcm_of_180_504_169_l1210_121070

def a : ℕ := 180
def b : ℕ := 504
def c : ℕ := 169

theorem lcm_of_180_504_169 : 
  Nat.lcm (Nat.lcm a b) c = 2^3 * 3^2 * 5 * 7 * 13^2 := by sorry

end NUMINAMATH_CALUDE_lcm_of_180_504_169_l1210_121070


namespace NUMINAMATH_CALUDE_rhombus_perimeter_l1210_121057

theorem rhombus_perimeter (x₁ x₂ : ℝ) : 
  x₁^2 - 14*x₁ + 48 = 0 →
  x₂^2 - 14*x₂ + 48 = 0 →
  x₁ ≠ x₂ →
  let s := Real.sqrt ((x₁^2 + x₂^2) / 4)
  4 * s = 20 := by sorry

end NUMINAMATH_CALUDE_rhombus_perimeter_l1210_121057


namespace NUMINAMATH_CALUDE_ratio_evaluation_l1210_121021

theorem ratio_evaluation : (2^2003 * 3^2002) / 6^2002 = 2 := by
  sorry

end NUMINAMATH_CALUDE_ratio_evaluation_l1210_121021


namespace NUMINAMATH_CALUDE_ellipse_properties_l1210_121002

/-- Properties of the ellipse 9x^2 + y^2 = 81 -/
theorem ellipse_properties :
  let ellipse := {(x, y) : ℝ × ℝ | 9 * x^2 + y^2 = 81}
  ∃ (major_axis minor_axis eccentricity : ℝ) 
    (foci_y vertex_y vertex_x : ℝ),
    -- Length of major axis
    major_axis = 18 ∧
    -- Length of minor axis
    minor_axis = 6 ∧
    -- Eccentricity
    eccentricity = 2 * Real.sqrt 2 / 3 ∧
    -- Foci coordinates
    foci_y = 6 * Real.sqrt 2 ∧
    (0, foci_y) ∈ ellipse ∧ (0, -foci_y) ∈ ellipse ∧
    -- Vertex coordinates
    vertex_y = 9 ∧ vertex_x = 3 ∧
    (0, vertex_y) ∈ ellipse ∧ (0, -vertex_y) ∈ ellipse ∧
    (vertex_x, 0) ∈ ellipse ∧ (-vertex_x, 0) ∈ ellipse :=
by
  sorry

end NUMINAMATH_CALUDE_ellipse_properties_l1210_121002


namespace NUMINAMATH_CALUDE_water_consumption_calculation_l1210_121024

/-- Water billing system and consumption calculation -/
theorem water_consumption_calculation 
  (base_rate : ℝ) 
  (excess_rate : ℝ) 
  (sewage_rate : ℝ) 
  (base_volume : ℝ) 
  (total_bill : ℝ) 
  (h1 : base_rate = 1.8) 
  (h2 : excess_rate = 2.3) 
  (h3 : sewage_rate = 1) 
  (h4 : base_volume = 15) 
  (h5 : total_bill = 58.5) : 
  ∃ (consumption : ℝ), 
    consumption = 20 ∧ 
    total_bill = 
      base_rate * min consumption base_volume + 
      excess_rate * max (consumption - base_volume) 0 + 
      sewage_rate * consumption :=
sorry

end NUMINAMATH_CALUDE_water_consumption_calculation_l1210_121024


namespace NUMINAMATH_CALUDE_impossible_to_transform_to_fives_l1210_121025

/-- Represents the three magician tricks -/
inductive MagicTrick
  | subtract_one
  | divide_by_two
  | multiply_by_three

/-- Represents the state of the transformation process -/
structure TransformState where
  numbers : List ℕ
  trick_counts : List ℕ
  deriving Repr

/-- Checks if a number is within the allowed range -/
def is_valid_number (n : ℕ) : Bool :=
  n ≤ 10

/-- Applies a magic trick to a number -/
def apply_trick (trick : MagicTrick) (n : ℕ) : Option ℕ :=
  match trick with
  | MagicTrick.subtract_one => if n > 0 then some (n - 1) else none
  | MagicTrick.divide_by_two => if n % 2 = 0 then some (n / 2) else none
  | MagicTrick.multiply_by_three => if n * 3 ≤ 10 then some (n * 3) else none

/-- Checks if the transformation is complete (all numbers are 5) -/
def is_transformation_complete (state : TransformState) : Bool :=
  state.numbers.all (· = 5)

/-- Checks if the transformation process is still valid -/
def is_valid_state (state : TransformState) : Bool :=
  state.numbers.all is_valid_number ∧
  state.trick_counts.all (· ≤ 5)

/-- The main theorem statement -/
theorem impossible_to_transform_to_fives :
  ¬ ∃ (final_state : TransformState),
    is_transformation_complete final_state ∧
    is_valid_state final_state ∧
    (∃ (initial_state : TransformState),
      initial_state.numbers = [3, 8, 9, 2, 4] ∧
      initial_state.trick_counts = [0, 0, 0] ∧
      -- There exists a sequence of valid transformations from initial_state to final_state
      True) :=
sorry

end NUMINAMATH_CALUDE_impossible_to_transform_to_fives_l1210_121025


namespace NUMINAMATH_CALUDE_max_green_socks_l1210_121055

/-- Represents the count of socks in a basket -/
structure SockBasket where
  green : ℕ
  yellow : ℕ
  total_bound : green + yellow ≤ 2025

/-- The probability of selecting two green socks without replacement -/
def prob_two_green (b : SockBasket) : ℚ :=
  (b.green * (b.green - 1)) / ((b.green + b.yellow) * (b.green + b.yellow - 1))

/-- Theorem stating the maximum number of green socks possible -/
theorem max_green_socks (b : SockBasket) 
  (h : prob_two_green b = 1/3) : 
  b.green ≤ 990 ∧ ∃ b' : SockBasket, b'.green = 990 ∧ prob_two_green b' = 1/3 :=
sorry

end NUMINAMATH_CALUDE_max_green_socks_l1210_121055


namespace NUMINAMATH_CALUDE_inverse_proportion_constant_l1210_121086

/-- Given two points A(3,m) and B(3m-1,2) on the graph of y = k/x, prove that k = 2 -/
theorem inverse_proportion_constant (k m : ℝ) : 
  (3 * m = k) ∧ (2 * (3 * m - 1) = k) → k = 2 := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_constant_l1210_121086


namespace NUMINAMATH_CALUDE_hanna_has_zero_erasers_l1210_121061

/-- The number of erasers Tanya has -/
def tanya_total : ℕ := 30

/-- The number of red erasers Tanya has -/
def tanya_red : ℕ := tanya_total / 2

/-- The number of blue erasers Tanya has -/
def tanya_blue : ℕ := tanya_total / 3

/-- The number of yellow erasers Tanya has -/
def tanya_yellow : ℕ := tanya_total - tanya_red - tanya_blue

/-- Rachel's erasers in terms of Tanya's red erasers -/
def rachel_erasers : ℤ := tanya_red / 3 - 5

/-- Hanna's erasers in terms of Rachel's -/
def hanna_erasers : ℤ := 3 * rachel_erasers

theorem hanna_has_zero_erasers :
  tanya_yellow = 2 * tanya_blue → hanna_erasers = 0 := by sorry

end NUMINAMATH_CALUDE_hanna_has_zero_erasers_l1210_121061


namespace NUMINAMATH_CALUDE_direct_proportion_through_point_one_three_l1210_121084

/-- A direct proportion function passing through (1, 3) has k = 3 -/
theorem direct_proportion_through_point_one_three (k : ℝ) : 
  (∀ x y : ℝ, y = k * x) → -- Direct proportion function
  (3 : ℝ) = k * (1 : ℝ) →  -- Passes through (1, 3)
  k = 3 := by sorry

end NUMINAMATH_CALUDE_direct_proportion_through_point_one_three_l1210_121084


namespace NUMINAMATH_CALUDE_correct_book_arrangements_l1210_121097

/-- The number of ways to arrange 11 books (3 Arabic, 4 German, 4 Spanish) on a shelf, keeping the Arabic books together -/
def book_arrangements : ℕ :=
  let total_books : ℕ := 11
  let arabic_books : ℕ := 3
  let german_books : ℕ := 4
  let spanish_books : ℕ := 4
  let arabic_unit : ℕ := 1
  let total_units : ℕ := arabic_unit + german_books + spanish_books
  (Nat.factorial total_units) * (Nat.factorial arabic_books)

theorem correct_book_arrangements :
  book_arrangements = 2177280 :=
by sorry

end NUMINAMATH_CALUDE_correct_book_arrangements_l1210_121097


namespace NUMINAMATH_CALUDE_simplify_expression_l1210_121069

theorem simplify_expression (a : ℝ) (h : a > 1) :
  (1 - a) * Real.sqrt (1 / (a - 1)) = -Real.sqrt (a - 1) := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1210_121069


namespace NUMINAMATH_CALUDE_linear_function_through_0_3_l1210_121094

/-- A linear function passing through (0,3) -/
def LinearFunctionThrough0_3 (f : ℝ → ℝ) : Prop :=
  (∀ x y : ℝ, f (x + y) = f x + f y - f 0) ∧ f 0 = 3

theorem linear_function_through_0_3 (f : ℝ → ℝ) (hf : LinearFunctionThrough0_3 f) :
  ∃ m : ℝ, ∀ x : ℝ, f x = m * x + 3 :=
sorry

end NUMINAMATH_CALUDE_linear_function_through_0_3_l1210_121094


namespace NUMINAMATH_CALUDE_set_intersection_subset_l1210_121028

theorem set_intersection_subset (a : ℝ) : 
  let A := {x : ℝ | 2*a + 1 ≤ x ∧ x ≤ 3*a - 5}
  let B := {x : ℝ | 3 ≤ x ∧ x ≤ 22}
  (A.Nonempty ∧ B.Nonempty) → (A ⊆ A ∩ B) ↔ (6 ≤ a ∧ a ≤ 9) :=
by sorry

end NUMINAMATH_CALUDE_set_intersection_subset_l1210_121028


namespace NUMINAMATH_CALUDE_largest_root_of_cubic_equation_l1210_121068

theorem largest_root_of_cubic_equation :
  let f (x : ℝ) := 4 * x^3 - 17 * x^2 + x + 10
  ∃ (max_root : ℝ), max_root = (25 + Real.sqrt 545) / 8 ∧
    f max_root = 0 ∧
    ∀ (y : ℝ), f y = 0 → y ≤ max_root :=
by sorry

end NUMINAMATH_CALUDE_largest_root_of_cubic_equation_l1210_121068


namespace NUMINAMATH_CALUDE_expression_factorization_l1210_121032

theorem expression_factorization (y : ℝ) :
  (16 * y^6 + 36 * y^4 - 9) - (4 * y^6 - 6 * y^4 + 9) = 6 * (2 * y^6 + 7 * y^4 - 3) := by
sorry

end NUMINAMATH_CALUDE_expression_factorization_l1210_121032


namespace NUMINAMATH_CALUDE_total_dress_designs_is_40_l1210_121042

/-- The number of color choices for a dress design. -/
def num_colors : ℕ := 4

/-- The number of pattern choices for a dress design. -/
def num_patterns : ℕ := 5

/-- The number of fabric type choices for a dress design. -/
def num_fabric_types : ℕ := 2

/-- The total number of possible dress designs. -/
def total_designs : ℕ := num_colors * num_patterns * num_fabric_types

/-- Theorem stating that the total number of possible dress designs is 40. -/
theorem total_dress_designs_is_40 : total_designs = 40 := by
  sorry

end NUMINAMATH_CALUDE_total_dress_designs_is_40_l1210_121042


namespace NUMINAMATH_CALUDE_sticker_count_l1210_121054

theorem sticker_count (stickers_per_page : ℕ) (total_pages : ℕ) : 
  stickers_per_page = 10 → total_pages = 22 → stickers_per_page * total_pages = 220 :=
by sorry

end NUMINAMATH_CALUDE_sticker_count_l1210_121054


namespace NUMINAMATH_CALUDE_shaded_area_percentage_l1210_121063

theorem shaded_area_percentage (total_squares : Nat) (shaded_squares : Nat) :
  total_squares = 16 →
  shaded_squares = 8 →
  (shaded_squares : ℚ) / (total_squares : ℚ) * 100 = 50 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_percentage_l1210_121063


namespace NUMINAMATH_CALUDE_quadratic_inequality_l1210_121089

theorem quadratic_inequality (y : ℝ) : y^2 - 8*y + 12 < 0 ↔ 2 < y ∧ y < 6 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l1210_121089


namespace NUMINAMATH_CALUDE_painters_time_equation_l1210_121052

/-- The time it takes for two painters to paint a room together, given their individual rates and a lunch break -/
theorem painters_time_equation (doug_rate : ℚ) (dave_rate : ℚ) (t : ℚ) 
  (h_doug : doug_rate = 1 / 5)
  (h_dave : dave_rate = 1 / 7)
  (h_positive : t > 0) :
  (doug_rate + dave_rate) * (t - 1) = 1 ↔ t = 47 / 12 :=
by sorry

end NUMINAMATH_CALUDE_painters_time_equation_l1210_121052


namespace NUMINAMATH_CALUDE_equality_from_fraction_equation_l1210_121041

theorem equality_from_fraction_equation (a b : ℝ) (h : a ≠ 0 ∧ b ≠ 0) :
  1 / (3 * a) + 2 / (3 * b) = 3 / (a + 2 * b) → a = b := by
  sorry

end NUMINAMATH_CALUDE_equality_from_fraction_equation_l1210_121041


namespace NUMINAMATH_CALUDE_f_inequality_solution_l1210_121065

-- Define the function f
def f (x : ℝ) : ℝ := x^3 + x

-- Define the domain of f
def domain (x : ℝ) : Prop := -2 < x ∧ x < 2

-- Define the solution set
def solution_set (a : ℝ) : Prop := (-2 < a ∧ a < 0) ∨ (0 < a ∧ a < 1)

-- State the theorem
theorem f_inequality_solution :
  ∀ a : ℝ, domain a → domain (a^2 - 2) →
  (f a + f (a^2 - 2) < 0 ↔ solution_set a) :=
sorry

end NUMINAMATH_CALUDE_f_inequality_solution_l1210_121065


namespace NUMINAMATH_CALUDE_train_length_l1210_121014

/-- Proves that a train moving at 40 kmph and passing a telegraph post in 7.199424046076314 seconds has a length of 80 meters. -/
theorem train_length (speed_kmph : ℝ) (time_seconds : ℝ) (length_meters : ℝ) : 
  speed_kmph = 40 →
  time_seconds = 7.199424046076314 →
  length_meters = speed_kmph * 1000 / 3600 * time_seconds →
  length_meters = 80 := by
  sorry

#check train_length

end NUMINAMATH_CALUDE_train_length_l1210_121014


namespace NUMINAMATH_CALUDE_overlap_area_is_0_15_l1210_121062

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A square defined by its vertices -/
structure Square where
  v1 : Point
  v2 : Point
  v3 : Point
  v4 : Point

/-- A triangle defined by its vertices -/
structure Triangle where
  v1 : Point
  v2 : Point
  v3 : Point

/-- The area of the overlapping region between a square and a triangle -/
def overlapArea (s : Square) (t : Triangle) : ℝ := sorry

/-- The theorem stating the area of overlap between the specific square and triangle -/
theorem overlap_area_is_0_15 :
  let s := Square.mk
    (Point.mk 0 0)
    (Point.mk 2 0)
    (Point.mk 2 2)
    (Point.mk 0 2)
  let t := Triangle.mk
    (Point.mk 3 0)
    (Point.mk 1 2)
    (Point.mk 2 1)
  overlapArea s t = 0.15 := by sorry

end NUMINAMATH_CALUDE_overlap_area_is_0_15_l1210_121062


namespace NUMINAMATH_CALUDE_win_sector_area_l1210_121078

/-- Given a circular spinner with radius 8 cm and a probability of winning of 1/4,
    the area of the WIN sector is 16π square centimeters. -/
theorem win_sector_area (radius : ℝ) (win_probability : ℝ) (win_sector_area : ℝ) : 
  radius = 8 →
  win_probability = 1 / 4 →
  win_sector_area = win_probability * π * radius^2 →
  win_sector_area = 16 * π := by
  sorry

end NUMINAMATH_CALUDE_win_sector_area_l1210_121078


namespace NUMINAMATH_CALUDE_no_natural_square_difference_2018_l1210_121088

theorem no_natural_square_difference_2018 : ¬ ∃ (m n : ℕ), m^2 = n^2 + 2018 := by
  sorry

end NUMINAMATH_CALUDE_no_natural_square_difference_2018_l1210_121088


namespace NUMINAMATH_CALUDE_scientific_notation_of_1040000000_l1210_121001

theorem scientific_notation_of_1040000000 :
  (1040000000 : ℝ) = 1.04 * (10 : ℝ)^9 := by sorry

end NUMINAMATH_CALUDE_scientific_notation_of_1040000000_l1210_121001


namespace NUMINAMATH_CALUDE_tenPeopleCircularArrangements_l1210_121051

/-- The number of unique circular arrangements of n people around a table,
    where rotations are considered the same. -/
def circularArrangements (n : ℕ) : ℕ := Nat.factorial (n - 1)

/-- Theorem stating that the number of unique circular arrangements
    of 10 people is equal to 9! -/
theorem tenPeopleCircularArrangements :
  circularArrangements 10 = Nat.factorial 9 := by
  sorry

end NUMINAMATH_CALUDE_tenPeopleCircularArrangements_l1210_121051


namespace NUMINAMATH_CALUDE_lara_flowers_theorem_l1210_121048

/-- The number of flowers Lara bought -/
def total_flowers : ℕ := sorry

/-- The number of flowers Lara gave to her mom -/
def flowers_to_mom : ℕ := 15

/-- The number of flowers Lara gave to her grandma -/
def flowers_to_grandma : ℕ := flowers_to_mom + 6

/-- The number of flowers Lara put in the vase -/
def flowers_in_vase : ℕ := 16

/-- Theorem stating the total number of flowers Lara bought -/
theorem lara_flowers_theorem : 
  total_flowers = flowers_to_mom + flowers_to_grandma + flowers_in_vase ∧ 
  total_flowers = 52 := by sorry

end NUMINAMATH_CALUDE_lara_flowers_theorem_l1210_121048


namespace NUMINAMATH_CALUDE_work_problem_solution_l1210_121031

/-- Proves that given the conditions of the work problem, the daily wage of worker c is 115 --/
theorem work_problem_solution (a b c : ℕ) : 
  (a : ℚ) / 3 = (b : ℚ) / 4 ∧ (a : ℚ) / 3 = (c : ℚ) / 5 →  -- daily wages ratio
  6 * a + 9 * b + 4 * c = 1702 →                          -- total earnings
  c = 115 := by
  sorry

end NUMINAMATH_CALUDE_work_problem_solution_l1210_121031


namespace NUMINAMATH_CALUDE_simplify_expression_l1210_121040

theorem simplify_expression (z : ℝ) : z - 2 + 4*z + 3 - 6*z + 5 - 8*z + 7 = -9*z + 13 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1210_121040


namespace NUMINAMATH_CALUDE_y_value_proof_l1210_121095

theorem y_value_proof : ∀ y : ℝ, (1/3 - 1/4 = 4/y) → y = 48 := by
  sorry

end NUMINAMATH_CALUDE_y_value_proof_l1210_121095


namespace NUMINAMATH_CALUDE_sine_cosine_sum_l1210_121047

/-- Given an angle α whose terminal side passes through the point (-3a, 4a) where a > 0,
    prove that sin α + 2cos α = -2/5 -/
theorem sine_cosine_sum (a : ℝ) (α : ℝ) (h1 : a > 0) 
    (h2 : Real.cos α = -3 * a / (5 * a)) (h3 : Real.sin α = 4 * a / (5 * a)) : 
    Real.sin α + 2 * Real.cos α = -2/5 := by
  sorry

end NUMINAMATH_CALUDE_sine_cosine_sum_l1210_121047


namespace NUMINAMATH_CALUDE_delta_y_over_delta_x_l1210_121059

-- Define the function f
def f (x : ℝ) : ℝ := 2 * x^2 - 4

-- Define the theorem
theorem delta_y_over_delta_x (Δx : ℝ) (Δy : ℝ) (h1 : f 1 = -2) (h2 : f (1 + Δx) = -2 + Δy) :
  Δy / Δx = 4 + 2 * Δx := by
  sorry

end NUMINAMATH_CALUDE_delta_y_over_delta_x_l1210_121059


namespace NUMINAMATH_CALUDE_diagonal_cubes_120_270_300_l1210_121039

/-- The number of cubes an internal diagonal passes through in a rectangular solid -/
def internal_diagonal_cubes (x y z : ℕ) : ℕ :=
  x + y + z - (Nat.gcd x y + Nat.gcd y z + Nat.gcd x z) + Nat.gcd x (Nat.gcd y z)

/-- The number of cubes a face diagonal passes through in a rectangular solid -/
def face_diagonal_cubes (x y : ℕ) : ℕ :=
  x + y - Nat.gcd x y

/-- Theorem about the number of cubes diagonals pass through in a 120 × 270 × 300 rectangular solid -/
theorem diagonal_cubes_120_270_300 :
  internal_diagonal_cubes 120 270 300 = 600 ∧
  face_diagonal_cubes 120 270 = 360 := by
  sorry


end NUMINAMATH_CALUDE_diagonal_cubes_120_270_300_l1210_121039


namespace NUMINAMATH_CALUDE_number_of_rattlesnakes_l1210_121003

theorem number_of_rattlesnakes (P B R V : ℕ) : 
  P + B + R + V = 420 →
  P = (3 * B) / 2 →
  V = 8 →
  P + R = 315 →
  R = 162 := by
sorry

end NUMINAMATH_CALUDE_number_of_rattlesnakes_l1210_121003


namespace NUMINAMATH_CALUDE_danny_bottle_caps_l1210_121013

theorem danny_bottle_caps (thrown_away : ℕ) (found : ℕ) (final : ℕ) :
  thrown_away = 60 →
  found = 58 →
  final = 67 →
  final = (thrown_away - found + final) →
  thrown_away - found + final = 69 :=
by sorry

end NUMINAMATH_CALUDE_danny_bottle_caps_l1210_121013


namespace NUMINAMATH_CALUDE_divisibility_condition_l1210_121096

theorem divisibility_condition (n : ℕ) (h : n ≥ 2) :
  (20^n + 19^n) % (20^(n-2) + 19^(n-2)) = 0 ↔ n = 3 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_condition_l1210_121096


namespace NUMINAMATH_CALUDE_negative_fraction_comparison_l1210_121071

theorem negative_fraction_comparison : -5/6 > -6/7 := by
  sorry

end NUMINAMATH_CALUDE_negative_fraction_comparison_l1210_121071


namespace NUMINAMATH_CALUDE_existence_of_twin_prime_divisors_l1210_121073

theorem existence_of_twin_prime_divisors :
  ∃ (n : ℕ) (p₁ p₂ : ℕ), 
    Odd n ∧ 
    0 < n ∧
    Prime p₁ ∧ 
    Prime p₂ ∧ 
    (2^n - 1) % p₁ = 0 ∧ 
    (2^n - 1) % p₂ = 0 ∧ 
    p₁ - p₂ = 2 := by
  sorry

end NUMINAMATH_CALUDE_existence_of_twin_prime_divisors_l1210_121073


namespace NUMINAMATH_CALUDE_problem_solution_l1210_121077

-- Define the set B
def B : Set ℝ := {m | ∀ x ∈ Set.Icc (-1 : ℝ) 2, x^2 - 2*x - m ≤ 0}

-- Define the set A(a)
def A (a : ℝ) : Set ℝ := {x | (x - 2*a) * (x - (a + 1)) ≤ 0}

theorem problem_solution :
  (B = Set.Ici 3) ∧
  ({a : ℝ | A a ⊆ B ∧ A a ≠ B} = Set.Ici 2) := by sorry

end NUMINAMATH_CALUDE_problem_solution_l1210_121077


namespace NUMINAMATH_CALUDE_min_a3_and_a2b2_l1210_121050

theorem min_a3_and_a2b2 (a₁ a₂ a₃ b₁ b₂ b₃ : ℝ) 
  (ha : a₁ > 0 ∧ a₂ > 0 ∧ a₃ > 0 ∧ b₁ > 0 ∧ b₂ > 0 ∧ b₃ > 0)
  (hseq1 : a₂ = a₁ + b₁ ∧ a₃ = a₁ + 2*b₁)
  (hseq2 : b₂ = b₁ * a₁ ∧ b₃ = b₁ * a₁^2)
  (heq : a₃ = b₃) :
  (∀ a₁' a₂' a₃' b₁' b₂' b₃' : ℝ, 
    (a₁' > 0 ∧ a₂' > 0 ∧ a₃' > 0 ∧ b₁' > 0 ∧ b₂' > 0 ∧ b₃' > 0) →
    (a₂' = a₁' + b₁' ∧ a₃' = a₁' + 2*b₁') →
    (b₂' = b₁' * a₁' ∧ b₃' = b₁' * a₁'^2) →
    (a₃' = b₃') →
    a₃' ≥ 3 * Real.sqrt 6 / 2) ∧
  (a₃ = 3 * Real.sqrt 6 / 2 → a₂ * b₂ = 15 * Real.sqrt 6 / 8) :=
sorry

end NUMINAMATH_CALUDE_min_a3_and_a2b2_l1210_121050


namespace NUMINAMATH_CALUDE_angle_in_second_quadrant_l1210_121087

theorem angle_in_second_quadrant (α : Real) (x : Real) :
  -- α is in the second quadrant
  π / 2 < α ∧ α < π →
  -- p(x, √5) is on the terminal side of α
  ∃ (r : Real), r > 0 ∧ x = r * Real.cos α ∧ Real.sqrt 5 = r * Real.sin α →
  -- cos α = (√2/4)x
  Real.cos α = (Real.sqrt 2 / 4) * x →
  -- x = -√3
  x = -Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_angle_in_second_quadrant_l1210_121087


namespace NUMINAMATH_CALUDE_percentage_difference_l1210_121017

theorem percentage_difference (A B C x : ℝ) : 
  A > 0 → B > 0 → C > 0 → 
  A > C → C > B → 
  A = B * (1 + x / 100) → 
  C = 0.75 * A → 
  x > 100 / 3 := by
sorry

end NUMINAMATH_CALUDE_percentage_difference_l1210_121017


namespace NUMINAMATH_CALUDE_smallest_n_for_inequality_l1210_121029

theorem smallest_n_for_inequality : ∃ n : ℕ, (∀ x y z w : ℝ, (x^2 + y^2 + z^2 + w^2)^2 ≤ n * (x^4 + y^4 + z^4 + w^4)) ∧
  (∀ m : ℕ, m < n → ∃ x y z w : ℝ, (x^2 + y^2 + z^2 + w^2)^2 > m * (x^4 + y^4 + z^4 + w^4)) ∧
  n = 4 :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_for_inequality_l1210_121029


namespace NUMINAMATH_CALUDE_max_tiles_theorem_l1210_121076

/-- Represents a rhombic tile with side length 1 and angles 60° and 120° -/
structure RhombicTile :=
  (side_length : ℝ := 1)
  (angle1 : ℝ := 60)
  (angle2 : ℝ := 120)

/-- Represents an equilateral triangle with side length n -/
structure EquilateralTriangle :=
  (side_length : ℕ)

/-- Calculates the maximum number of rhombic tiles that can fit in an equilateral triangle -/
def max_tiles_in_triangle (triangle : EquilateralTriangle) : ℕ :=
  (triangle.side_length^2 - triangle.side_length) / 2

/-- Theorem: The maximum number of rhombic tiles in an equilateral triangle is (n^2 - n) / 2 -/
theorem max_tiles_theorem (n : ℕ) (triangle : EquilateralTriangle) (tile : RhombicTile) :
  triangle.side_length = n →
  tile.side_length = 1 →
  tile.angle1 = 60 →
  tile.angle2 = 120 →
  max_tiles_in_triangle triangle = (n^2 - n) / 2 := by
  sorry

end NUMINAMATH_CALUDE_max_tiles_theorem_l1210_121076


namespace NUMINAMATH_CALUDE_x_square_plus_reciprocal_l1210_121066

theorem x_square_plus_reciprocal (x : ℝ) (h : 31 = x^6 + 1/x^6) : 
  x^2 + 1/x^2 = (34 : ℝ)^(1/3) := by
  sorry

end NUMINAMATH_CALUDE_x_square_plus_reciprocal_l1210_121066


namespace NUMINAMATH_CALUDE_value_of_p_l1210_121020

theorem value_of_p (n : ℝ) (p : ℝ) : 
  n = 9/4 → p = 4*n*(1/2^2009)^(Real.log 1) → p = 9 := by sorry

end NUMINAMATH_CALUDE_value_of_p_l1210_121020


namespace NUMINAMATH_CALUDE_quadratic_term_zero_l1210_121004

theorem quadratic_term_zero (a : ℝ) : 
  (∀ x : ℝ, (a * x + 3) * (6 * x^2 - 2 * x + 1) = 6 * a * x^3 + (18 - 2 * a) * x^2 + (a - 6) * x + 3) →
  (∀ x : ℝ, (a * x + 3) * (6 * x^2 - 2 * x + 1) = 6 * a * x^3 + (a - 6) * x + 3) →
  a = 9 := by
sorry

end NUMINAMATH_CALUDE_quadratic_term_zero_l1210_121004


namespace NUMINAMATH_CALUDE_economics_and_law_tournament_l1210_121007

theorem economics_and_law_tournament (n : ℕ) (m : ℕ) : 
  220 < n → n < 254 →
  m < n →
  (n - 2*m)^2 = n →
  ∀ k : ℕ, (220 < k ∧ k < 254 ∧ k < n ∧ (k - 2*(n-k))^2 = k) → n - m ≤ k - (n - k) →
  n - m = 105 :=
sorry

end NUMINAMATH_CALUDE_economics_and_law_tournament_l1210_121007


namespace NUMINAMATH_CALUDE_f_range_l1210_121074

noncomputable def f (x : ℝ) : ℝ := Real.arctan x + Real.arctan ((1 - x) / (1 + x))

theorem f_range : ∀ x : ℝ, f x = -3 * π / 4 ∨ f x = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_f_range_l1210_121074


namespace NUMINAMATH_CALUDE_triangle_angle_45_l1210_121010

/-- Given a triangle with sides a, b, c, perimeter 2s, and area T,
    if T + (ab/2) = s(s-c), then the angle opposite side c is 45°. -/
theorem triangle_angle_45 (a b c s T : ℝ) (h_positive : a > 0 ∧ b > 0 ∧ c > 0) 
    (h_perimeter : a + b + c = 2 * s) (h_area : T > 0)
    (h_equation : T + (a * b / 2) = s * (s - c)) :
    let γ := Real.arccos ((a^2 + b^2 - c^2) / (2 * a * b))
    γ = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_45_l1210_121010


namespace NUMINAMATH_CALUDE_lizard_feature_difference_l1210_121038

/-- Represents a three-eyed lizard with wrinkles and spots -/
structure Lizard :=
  (eyes : ℕ)
  (wrinkle_factor : ℕ)
  (spot_factor : ℕ)

/-- Calculate the total number of features (eyes, wrinkles, and spots) for a lizard -/
def total_features (l : Lizard) : ℕ :=
  l.eyes + (l.eyes * l.wrinkle_factor) + (l.eyes * l.wrinkle_factor * l.spot_factor)

/-- The main theorem about the difference between total features and eyes for two lizards -/
theorem lizard_feature_difference (jan_lizard cousin_lizard : Lizard)
  (h1 : jan_lizard.eyes = 3)
  (h2 : jan_lizard.wrinkle_factor = 3)
  (h3 : jan_lizard.spot_factor = 7)
  (h4 : cousin_lizard.eyes = 3)
  (h5 : cousin_lizard.wrinkle_factor = 2)
  (h6 : cousin_lizard.spot_factor = 5) :
  (total_features jan_lizard + total_features cousin_lizard) - (jan_lizard.eyes + cousin_lizard.eyes) = 102 :=
sorry

end NUMINAMATH_CALUDE_lizard_feature_difference_l1210_121038


namespace NUMINAMATH_CALUDE_parabola_vertex_and_point_l1210_121082

/-- Represents a parabola of the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℤ
  b : ℤ
  c : ℤ

/-- The y-coordinate of a point on the parabola given its x-coordinate -/
def Parabola.y (p : Parabola) (x : ℚ) : ℚ :=
  p.a * x^2 + p.b * x + p.c

theorem parabola_vertex_and_point (p : Parabola) :
  p.y 2 = 1 → p.y 0 = 5 → p.a + p.b - p.c = -8 := by
  sorry

end NUMINAMATH_CALUDE_parabola_vertex_and_point_l1210_121082


namespace NUMINAMATH_CALUDE_power_sum_equality_l1210_121080

theorem power_sum_equality (x : ℝ) : x^3 * x + x^2 * x^2 = 2 * x^4 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_equality_l1210_121080


namespace NUMINAMATH_CALUDE_brick_width_is_10_cm_l1210_121026

/-- Prove that the width of a brick is 10 cm given the specified conditions -/
theorem brick_width_is_10_cm
  (brick_length : ℝ)
  (brick_height : ℝ)
  (wall_length : ℝ)
  (wall_width : ℝ)
  (wall_height : ℝ)
  (num_bricks : ℕ)
  (h1 : brick_length = 20)
  (h2 : brick_height = 7.5)
  (h3 : wall_length = 2600)
  (h4 : wall_width = 200)
  (h5 : wall_height = 75)
  (h6 : num_bricks = 26000)
  (h7 : wall_length * wall_width * wall_height = num_bricks * (brick_length * brick_height * brick_width)) :
  brick_width = 10 := by
  sorry


end NUMINAMATH_CALUDE_brick_width_is_10_cm_l1210_121026


namespace NUMINAMATH_CALUDE_vector_equation_l1210_121027

/-- Given non-collinear points A, B, C, and a point O satisfying
    16*OA - 12*OB - 3*OC = 0, prove that OA = 12*AB + 3*AC -/
theorem vector_equation (A B C O : EuclideanSpace ℝ (Fin 3)) 
  (h_not_collinear : ¬Collinear ℝ {A, B, C})
  (h_equation : 16 • (O - A) - 12 • (O - B) - 3 • (O - C) = 0) :
  O - A = 12 • (B - A) + 3 • (C - A) := by
  sorry

end NUMINAMATH_CALUDE_vector_equation_l1210_121027


namespace NUMINAMATH_CALUDE_triangle_area_l1210_121046

/-- The area of a triangle with vertices at (2, 1), (2, 7), and (8, 4) is 18 square units -/
theorem triangle_area : ℝ := by
  -- Define the vertices of the triangle
  let A : ℝ × ℝ := (2, 1)
  let B : ℝ × ℝ := (2, 7)
  let C : ℝ × ℝ := (8, 4)

  -- Calculate the area using the formula: Area = (1/2) * |x1(y2 - y3) + x2(y3 - y1) + x3(y1 - y2)|
  let area := (1/2) * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

  -- Prove that the calculated area equals 18
  sorry

end NUMINAMATH_CALUDE_triangle_area_l1210_121046


namespace NUMINAMATH_CALUDE_binomial_30_3_l1210_121056

theorem binomial_30_3 : Nat.choose 30 3 = 4060 := by
  sorry

end NUMINAMATH_CALUDE_binomial_30_3_l1210_121056


namespace NUMINAMATH_CALUDE_percent_of_number_l1210_121053

/-- 0.1 percent of 12,356 is equal to 12.356 -/
theorem percent_of_number : (0.1 / 100) * 12356 = 12.356 := by sorry

end NUMINAMATH_CALUDE_percent_of_number_l1210_121053


namespace NUMINAMATH_CALUDE_power_of_two_geq_n_l1210_121011

theorem power_of_two_geq_n (n : ℕ) (h : n ≥ 1) : 2^n ≥ n := by
  sorry

end NUMINAMATH_CALUDE_power_of_two_geq_n_l1210_121011


namespace NUMINAMATH_CALUDE_range_of_fraction_l1210_121037

theorem range_of_fraction (x y : ℝ) (h1 : 2*x + y = 8) (h2 : 2 ≤ x) (h3 : x ≤ 3) :
  3/2 ≤ (y+1)/(x-1) ∧ (y+1)/(x-1) ≤ 5 := by
sorry

end NUMINAMATH_CALUDE_range_of_fraction_l1210_121037


namespace NUMINAMATH_CALUDE_james_club_night_cost_l1210_121060

def club_night_cost (entry_fee : ℚ) (friend_rounds : ℕ) (friends : ℕ) 
  (self_drinks : ℕ) (cocktail_price : ℚ) (non_alcoholic_price : ℚ) 
  (cocktail_discount : ℚ) (cocktails_bought : ℕ) (burger_price : ℚ) 
  (fries_price : ℚ) (food_tip_rate : ℚ) (drink_tip_rate : ℚ) : ℚ :=
  let total_drinks := friend_rounds * friends + self_drinks
  let non_alcoholic_drinks := total_drinks - cocktails_bought
  let cocktail_cost := cocktails_bought * cocktail_price
  let discounted_cocktail_cost := 
    if cocktails_bought ≥ 3 then cocktail_cost * (1 - cocktail_discount) else cocktail_cost
  let non_alcoholic_cost := non_alcoholic_drinks * non_alcoholic_price
  let food_cost := burger_price + fries_price
  let food_tip := food_cost * food_tip_rate
  let drink_tip := (cocktail_cost + non_alcoholic_cost) * drink_tip_rate
  entry_fee + discounted_cocktail_cost + non_alcoholic_cost + food_cost + food_tip + drink_tip

theorem james_club_night_cost :
  club_night_cost 30 3 10 8 10 5 0.2 7 20 8 0.2 0.15 = 308.35 := by
  sorry

end NUMINAMATH_CALUDE_james_club_night_cost_l1210_121060


namespace NUMINAMATH_CALUDE_polynomial_remainder_theorem_l1210_121023

theorem polynomial_remainder_theorem (x : ℝ) : 
  let f : ℝ → ℝ := λ x => x^5 - 8*x^4 + 15*x^3 + 30*x^2 - 47*x + 20
  (f 2) = 104 := by sorry

end NUMINAMATH_CALUDE_polynomial_remainder_theorem_l1210_121023


namespace NUMINAMATH_CALUDE_doubled_speed_cleaning_time_l1210_121000

def house_cleaning (bruce_rate anne_rate : ℝ) : Prop :=
  bruce_rate > 0 ∧ anne_rate > 0 ∧
  bruce_rate + anne_rate = 1 / 4 ∧
  anne_rate = 1 / 12

theorem doubled_speed_cleaning_time (bruce_rate anne_rate : ℝ) 
  (h : house_cleaning bruce_rate anne_rate) : 
  1 / (bruce_rate + 2 * anne_rate) = 3 := by
  sorry

end NUMINAMATH_CALUDE_doubled_speed_cleaning_time_l1210_121000


namespace NUMINAMATH_CALUDE_inscribed_rectangle_sides_l1210_121093

/-- A rectangle inscribed in a triangle -/
structure InscribedRectangle where
  -- Triangle dimensions
  triangleBase : ℝ
  triangleHeight : ℝ
  -- Rectangle side ratio
  rectRatio : ℝ
  -- Rectangle sides
  rectShortSide : ℝ
  rectLongSide : ℝ
  -- Conditions
  triangleBase_pos : 0 < triangleBase
  triangleHeight_pos : 0 < triangleHeight
  rectRatio_pos : 0 < rectRatio
  rectShortSide_pos : 0 < rectShortSide
  rectLongSide_pos : 0 < rectLongSide
  ratio_cond : rectLongSide / rectShortSide = 9 / 5
  inscribed_cond : rectLongSide ≤ triangleBase
  proportion_cond : (triangleHeight - rectShortSide) / triangleHeight = rectLongSide / triangleBase

/-- The sides of the inscribed rectangle are 10 and 18 -/
theorem inscribed_rectangle_sides (r : InscribedRectangle) 
    (h1 : r.triangleBase = 48) 
    (h2 : r.triangleHeight = 16) 
    (h3 : r.rectRatio = 9/5) : 
    r.rectShortSide = 10 ∧ r.rectLongSide = 18 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_rectangle_sides_l1210_121093


namespace NUMINAMATH_CALUDE_place_value_ratios_l1210_121005

theorem place_value_ratios : 
  ∀ (d : ℕ), d > 0 → d < 10 →
  (d * 10000) / (d * 1000) = 10 ∧ 
  (d * 100000) / (d * 100) = 1000 := by
  sorry

end NUMINAMATH_CALUDE_place_value_ratios_l1210_121005


namespace NUMINAMATH_CALUDE_orange_distribution_l1210_121009

/-- Given a number of oranges, pieces per orange, and number of friends,
    calculate the number of pieces each friend receives. -/
def pieces_per_friend (oranges : ℕ) (pieces_per_orange : ℕ) (friends : ℕ) : ℚ :=
  (oranges * pieces_per_orange : ℚ) / friends

/-- Theorem stating that given 80 oranges, each divided into 10 pieces,
    and 200 friends, each friend will receive 4 pieces. -/
theorem orange_distribution :
  pieces_per_friend 80 10 200 = 4 := by
  sorry

end NUMINAMATH_CALUDE_orange_distribution_l1210_121009


namespace NUMINAMATH_CALUDE_rachel_apple_picking_l1210_121043

theorem rachel_apple_picking (num_trees : ℕ) (total_picked : ℕ) (h1 : num_trees = 4) (h2 : total_picked = 28) :
  total_picked / num_trees = 7 := by
  sorry

end NUMINAMATH_CALUDE_rachel_apple_picking_l1210_121043


namespace NUMINAMATH_CALUDE_probability_in_pascal_triangle_l1210_121022

/-- The number of rows in Pascal's Triangle we're considering --/
def n : ℕ := 20

/-- The total number of elements in the first n rows of Pascal's Triangle --/
def total_elements (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The number of ones in the first n rows of Pascal's Triangle --/
def ones_count (n : ℕ) : ℕ := 2 * (n - 1) + 1

/-- The probability of selecting a 1 from the first n rows of Pascal's Triangle --/
def probability_of_one (n : ℕ) : ℚ :=
  (ones_count n : ℚ) / (total_elements n : ℚ)

theorem probability_in_pascal_triangle :
  probability_of_one n = 13 / 70 := by
  sorry

end NUMINAMATH_CALUDE_probability_in_pascal_triangle_l1210_121022


namespace NUMINAMATH_CALUDE_quadratic_factorization_l1210_121035

theorem quadratic_factorization (b : ℤ) : 
  (∃ (m n p q : ℤ), 15 * x^2 + b * x + 30 = (m * x + n) * (p * x + q)) ↔ b = 43 :=
sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l1210_121035


namespace NUMINAMATH_CALUDE_perpendicular_lines_slope_product_l1210_121075

/-- Two lines in a 2D plane are perpendicular iff the product of their slopes is -1 -/
theorem perpendicular_lines_slope_product (k₁ k₂ l₁ l₂ : ℝ) (hk₁ : k₁ ≠ 0) (hk₂ : k₂ ≠ 0) :
  (∀ x y₁ y₂ : ℝ, y₁ = k₁ * x + l₁ ∧ y₂ = k₂ * x + l₂) →
  (∀ x₁ y₁ x₂ y₂ : ℝ, y₁ = k₁ * x₁ + l₁ ∧ y₂ = k₂ * x₂ + l₂ → 
    ((x₂ - x₁) * (k₁ * x₁ + l₁ - (k₂ * x₂ + l₂)) + (x₂ - x₁) * (y₂ - y₁) = 0)) ↔
  k₁ * k₂ = -1 :=
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_slope_product_l1210_121075


namespace NUMINAMATH_CALUDE_opposite_absolute_value_and_square_l1210_121079

theorem opposite_absolute_value_and_square (x y : ℝ) :
  |x + y - 2| + (2*x - 3*y + 5)^2 = 0 → x = 1/5 ∧ y = 9/5 := by
  sorry

end NUMINAMATH_CALUDE_opposite_absolute_value_and_square_l1210_121079


namespace NUMINAMATH_CALUDE_eddies_sister_pies_per_day_l1210_121049

theorem eddies_sister_pies_per_day :
  let eddie_pies_per_day : ℕ := 3
  let mother_pies_per_day : ℕ := 8
  let total_days : ℕ := 7
  let total_pies : ℕ := 119
  ∃ (sister_pies_per_day : ℕ),
    sister_pies_per_day * total_days + eddie_pies_per_day * total_days + mother_pies_per_day * total_days = total_pies ∧
    sister_pies_per_day = 6 :=
by sorry

end NUMINAMATH_CALUDE_eddies_sister_pies_per_day_l1210_121049


namespace NUMINAMATH_CALUDE_twentieth_number_in_twentieth_row_l1210_121006

/-- Calculates the first number in a given row of the triangular sequence -/
def first_number_in_row (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Calculates the kth number in the nth row of the triangular sequence -/
def number_in_sequence (n k : ℕ) : ℕ := first_number_in_row n + (k - 1)

/-- The 20th number in the 20th row of the triangular sequence is 381 -/
theorem twentieth_number_in_twentieth_row :
  number_in_sequence 20 20 = 381 := by sorry

end NUMINAMATH_CALUDE_twentieth_number_in_twentieth_row_l1210_121006


namespace NUMINAMATH_CALUDE_rectangular_plot_breadth_l1210_121098

/-- Proves that the breadth of a rectangular plot is 14 meters, given that its length is thrice its breadth and its area is 588 square meters. -/
theorem rectangular_plot_breadth (breadth : ℝ) (length : ℝ) (area : ℝ) : 
  length = 3 * breadth → 
  area = length * breadth → 
  area = 588 → 
  breadth = 14 := by
sorry

end NUMINAMATH_CALUDE_rectangular_plot_breadth_l1210_121098


namespace NUMINAMATH_CALUDE_range_of_m_l1210_121058

theorem range_of_m (x y m : ℝ) (hx : x > 0) (hy : y > 0) (h_eq : 2/x + 1/y = 1) :
  (∀ x y, x > 0 → y > 0 → 2/x + 1/y = 1 → x + 2*y > m^2 + 2*m) ↔ -4 < m ∧ m < 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l1210_121058


namespace NUMINAMATH_CALUDE_solve_for_b_l1210_121072

/-- Given a system of equations and its solution, prove the value of b. -/
theorem solve_for_b (a : ℝ) : 
  (∃ x y : ℝ, a * x - 2 * y = 1 ∧ 2 * x + b * y = 5) →
  (∃ x y : ℝ, x = 1 ∧ y = a ∧ a * x - 2 * y = 1 ∧ 2 * x + b * y = 5) →
  b = -3 :=
by
  sorry

end NUMINAMATH_CALUDE_solve_for_b_l1210_121072


namespace NUMINAMATH_CALUDE_yuna_division_l1210_121085

theorem yuna_division (x : ℚ) : 8 * x = 56 → 42 / x = 6 := by
  sorry

end NUMINAMATH_CALUDE_yuna_division_l1210_121085


namespace NUMINAMATH_CALUDE_product_closed_in_P_l1210_121030

/-- The set of perfect squares -/
def P : Set ℕ := {n : ℕ | ∃ m : ℕ, m > 0 ∧ n = m^2}

/-- The theorem stating that the product of two elements in P is also in P -/
theorem product_closed_in_P (a b : ℕ) (ha : a ∈ P) (hb : b ∈ P) : a * b ∈ P := by
  sorry

#check product_closed_in_P

end NUMINAMATH_CALUDE_product_closed_in_P_l1210_121030


namespace NUMINAMATH_CALUDE_smaller_solution_quadratic_equation_l1210_121015

theorem smaller_solution_quadratic_equation :
  let f : ℝ → ℝ := λ x ↦ x^2 - 15*x - 56
  let sol₁ : ℝ := (15 - Real.sqrt 449) / 2
  let sol₂ : ℝ := (15 + Real.sqrt 449) / 2
  f sol₁ = 0 ∧ f sol₂ = 0 ∧ sol₁ < sol₂ ∧ 
  ∀ x : ℝ, f x = 0 → x = sol₁ ∨ x = sol₂ :=
by sorry

end NUMINAMATH_CALUDE_smaller_solution_quadratic_equation_l1210_121015


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l1210_121091

theorem simplify_and_evaluate :
  ∀ x : ℤ, -2 < x ∧ x ≤ 2 ∧ x ≠ -1 ∧ x ≠ 0 ∧ x ≠ 1 →
  (x^2 + x) / (x^2 - 2*x + 1) / ((2 / (x - 1)) - (1 / x)) = x^2 / (x - 1) ∧
  (x = 2 → x^2 / (x - 1) = 4) :=
by sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l1210_121091


namespace NUMINAMATH_CALUDE_parallelepiped_diagonals_edges_squares_sum_equal_l1210_121044

/-- A parallelepiped with side lengths a, b, and c. -/
structure Parallelepiped where
  a : ℝ
  b : ℝ
  c : ℝ
  a_pos : 0 < a
  b_pos : 0 < b
  c_pos : 0 < c

/-- The sum of squares of the lengths of the four diagonals of a parallelepiped. -/
def sum_squares_diagonals (p : Parallelepiped) : ℝ :=
  4 * (p.a^2 + p.b^2 + p.c^2)

/-- The sum of squares of the lengths of the twelve edges of a parallelepiped. -/
def sum_squares_edges (p : Parallelepiped) : ℝ :=
  4 * p.a^2 + 4 * p.b^2 + 4 * p.c^2

/-- 
Theorem: The sum of the squares of the lengths of the four diagonals 
of a parallelepiped is equal to the sum of the squares of the lengths of its twelve edges.
-/
theorem parallelepiped_diagonals_edges_squares_sum_equal (p : Parallelepiped) :
  sum_squares_diagonals p = sum_squares_edges p := by
  sorry

end NUMINAMATH_CALUDE_parallelepiped_diagonals_edges_squares_sum_equal_l1210_121044


namespace NUMINAMATH_CALUDE_hyperbolas_M_value_l1210_121092

/-- Two hyperbolas with the same asymptotes -/
def hyperbolas_same_asymptotes (M : ℝ) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧
  (∀ (x y : ℝ), x^2/9 - y^2/16 = 1 → y = k*x ∨ y = -k*x) ∧
  (∀ (x y : ℝ), y^2/25 - x^2/M = 1 → y = k*x ∨ y = -k*x)

/-- The value of M for which the hyperbolas have the same asymptotes -/
theorem hyperbolas_M_value :
  hyperbolas_same_asymptotes (225/16) :=
sorry

end NUMINAMATH_CALUDE_hyperbolas_M_value_l1210_121092


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l1210_121045

/-- A geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

/-- Theorem: In a geometric sequence, if a_3 * a_4 = 6, then a_2 * a_5 = 6 -/
theorem geometric_sequence_property (a : ℕ → ℝ) (h : GeometricSequence a) 
    (h_prod : a 3 * a 4 = 6) : a 2 * a 5 = 6 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l1210_121045
