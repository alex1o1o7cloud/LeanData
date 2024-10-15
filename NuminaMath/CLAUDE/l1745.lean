import Mathlib

namespace NUMINAMATH_CALUDE_max_rope_piece_length_l1745_174522

theorem max_rope_piece_length : Nat.gcd 60 (Nat.gcd 75 90) = 15 := by
  sorry

end NUMINAMATH_CALUDE_max_rope_piece_length_l1745_174522


namespace NUMINAMATH_CALUDE_choose_3_from_12_l1745_174531

theorem choose_3_from_12 : Nat.choose 12 3 = 220 := by
  sorry

end NUMINAMATH_CALUDE_choose_3_from_12_l1745_174531


namespace NUMINAMATH_CALUDE_sequence_formula_main_theorem_l1745_174527

def a (n : ℕ+) : ℚ := 1 / ((2 * n.val - 1) * (2 * n.val + 1))

def S (n : ℕ+) : ℚ := sorry

theorem sequence_formula (n : ℕ+) :
  S n / (n.val * (2 * n.val - 1)) = a n ∧ 
  S 1 / (1 * (2 * 1 - 1)) = 1 / 3 :=
by sorry

theorem main_theorem (n : ℕ+) : 
  S n / (n.val * (2 * n.val - 1)) = 1 / ((2 * n.val - 1) * (2 * n.val + 1)) :=
by sorry

end NUMINAMATH_CALUDE_sequence_formula_main_theorem_l1745_174527


namespace NUMINAMATH_CALUDE_f_monotone_range_l1745_174512

/-- Definition of the function f --/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (2 - a) * x + 1 else a^x

/-- The main theorem --/
theorem f_monotone_range (a : ℝ) :
  (a > 0 ∧ a ≠ 1) ∧
  (∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → (f a x₁ - f a x₂) / (x₁ - x₂) > 0) →
  a ∈ Set.Icc (3/2) 2 ∧ a < 2 :=
sorry

end NUMINAMATH_CALUDE_f_monotone_range_l1745_174512


namespace NUMINAMATH_CALUDE_line_l_and_AB_are_skew_l1745_174543

-- Define the types for points, lines, and planes
variable (Point Line Plane : Type)

-- Define the relationships
variable (on_plane : Point → Plane → Prop)
variable (on_line : Point → Line → Prop)
variable (line_on_plane : Line → Plane → Prop)
variable (plane_intersection : Plane → Plane → Line)
variable (line_through_points : Point → Point → Line)
variable (skew_lines : Line → Line → Prop)

-- Define the given conditions
variable (α β : Plane)
variable (A B : Point)
variable (l : Line)

-- Theorem statement
theorem line_l_and_AB_are_skew :
  on_plane A α →
  on_plane B β →
  plane_intersection α β = l →
  ¬ on_line A l →
  ¬ on_line B l →
  skew_lines l (line_through_points A B) :=
by sorry

end NUMINAMATH_CALUDE_line_l_and_AB_are_skew_l1745_174543


namespace NUMINAMATH_CALUDE_siena_bookmarks_theorem_l1745_174578

/-- The number of pages Siena bookmarks every day -/
def pages_per_day : ℕ := 30

/-- The number of pages Siena has at the start of March -/
def initial_pages : ℕ := 400

/-- The number of pages Siena will have at the end of March -/
def final_pages : ℕ := 1330

/-- The number of days in March -/
def days_in_march : ℕ := 31

theorem siena_bookmarks_theorem :
  initial_pages + pages_per_day * days_in_march = final_pages :=
by sorry

end NUMINAMATH_CALUDE_siena_bookmarks_theorem_l1745_174578


namespace NUMINAMATH_CALUDE_middle_number_is_nine_l1745_174555

theorem middle_number_is_nine (x y z : ℕ) (h1 : x < y) (h2 : y < z)
  (h3 : x + y = 15) (h4 : x + z = 23) (h5 : y + z = 26) : y = 9 := by
  sorry

end NUMINAMATH_CALUDE_middle_number_is_nine_l1745_174555


namespace NUMINAMATH_CALUDE_max_value_implies_a_equals_one_l1745_174537

def f (a x : ℝ) : ℝ := -x^2 + 2*a*x + a - 1

theorem max_value_implies_a_equals_one :
  ∀ a : ℝ,
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → f a x ≤ 1) ∧
  (∃ x : ℝ, 0 ≤ x ∧ x ≤ 1 ∧ f a x = 1) →
  a = 1 :=
by sorry

end NUMINAMATH_CALUDE_max_value_implies_a_equals_one_l1745_174537


namespace NUMINAMATH_CALUDE_triangle_properties_l1745_174598

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem states properties of a specific triangle -/
theorem triangle_properties (t : Triangle) 
  (h1 : t.b * Real.cos t.C = (2 * t.a + t.c) * Real.cos (π - t.B))
  (h2 : t.b = Real.sqrt 13)
  (h3 : (1 / 2) * t.a * t.c * Real.sin t.B = (3 * Real.sqrt 3) / 4) :
  t.B = 2 * π / 3 ∧ t.a + t.c = 4 := by
  sorry


end NUMINAMATH_CALUDE_triangle_properties_l1745_174598


namespace NUMINAMATH_CALUDE_movie_trip_cost_l1745_174588

/-- The total cost of a movie trip for a group of adults and children -/
def total_cost (num_adults num_children : ℕ) (adult_ticket_price child_ticket_price concession_cost : ℚ) : ℚ :=
  num_adults * adult_ticket_price + num_children * child_ticket_price + concession_cost

/-- Theorem stating that the total cost for the given group is $76 -/
theorem movie_trip_cost : 
  total_cost 5 2 10 7 12 = 76 := by
  sorry

end NUMINAMATH_CALUDE_movie_trip_cost_l1745_174588


namespace NUMINAMATH_CALUDE_cake_serving_solution_l1745_174549

/-- Represents the number of cakes served for each type --/
structure CakeCount where
  chocolate : ℕ
  vanilla : ℕ
  strawberry : ℕ

/-- Represents the conditions of the cake serving problem --/
def cake_serving_conditions (c : CakeCount) : Prop :=
  c.chocolate = 2 * c.vanilla ∧
  c.strawberry = c.chocolate / 2 ∧
  c.vanilla = 12 + 18

/-- Theorem stating the correct number of cakes served for each type --/
theorem cake_serving_solution :
  ∃ c : CakeCount, cake_serving_conditions c ∧ 
    c.chocolate = 60 ∧ c.vanilla = 30 ∧ c.strawberry = 30 := by
  sorry

end NUMINAMATH_CALUDE_cake_serving_solution_l1745_174549


namespace NUMINAMATH_CALUDE_gcd_360_128_l1745_174500

theorem gcd_360_128 : Nat.gcd 360 128 = 8 := by sorry

end NUMINAMATH_CALUDE_gcd_360_128_l1745_174500


namespace NUMINAMATH_CALUDE_license_plate_count_l1745_174570

/-- The number of consonants in the alphabet. -/
def num_consonants : ℕ := 20

/-- The number of vowels in the alphabet (including Y). -/
def num_vowels : ℕ := 6

/-- The number of digits (0-9). -/
def num_digits : ℕ := 10

/-- The total number of letters in the alphabet. -/
def num_letters : ℕ := 26

/-- The number of unique five-character license plates with the sequence:
    consonant, vowel, consonant, digit, any letter. -/
def num_license_plates : ℕ := num_consonants * num_vowels * num_consonants * num_digits * num_letters

theorem license_plate_count :
  num_license_plates = 624000 :=
sorry

end NUMINAMATH_CALUDE_license_plate_count_l1745_174570


namespace NUMINAMATH_CALUDE_arcade_spending_fraction_l1745_174530

def allowance : ℚ := 4.5

theorem arcade_spending_fraction (x : ℚ) 
  (h1 : (2/3) * (1 - x) * allowance = 1.2) : x = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_arcade_spending_fraction_l1745_174530


namespace NUMINAMATH_CALUDE_digit_interchange_effect_l1745_174595

theorem digit_interchange_effect (n : ℕ) (p q : ℕ) : 
  n = 9 → 
  p > q → 
  p - q = 1 → 
  (10 * p + q) - (10 * q + p) = (n : ℤ) := by
  sorry

end NUMINAMATH_CALUDE_digit_interchange_effect_l1745_174595


namespace NUMINAMATH_CALUDE_complex_number_quadrant_l1745_174576

theorem complex_number_quadrant : 
  let z : ℂ := Complex.mk (Real.sin 1) (Real.cos 2)
  z.re > 0 ∧ z.im < 0 :=
by sorry

end NUMINAMATH_CALUDE_complex_number_quadrant_l1745_174576


namespace NUMINAMATH_CALUDE_pizza_pieces_l1745_174535

theorem pizza_pieces (total_people : Nat) (half_eaters : Nat) (three_quarter_eaters : Nat) (pieces_left : Nat) :
  total_people = 4 →
  half_eaters = 2 →
  three_quarter_eaters = 2 →
  pieces_left = 6 →
  ∃ (pieces_per_pizza : Nat),
    pieces_per_pizza * (half_eaters * (1/2) + three_quarter_eaters * (1/4)) = pieces_left ∧
    pieces_per_pizza = 4 := by
  sorry

end NUMINAMATH_CALUDE_pizza_pieces_l1745_174535


namespace NUMINAMATH_CALUDE_number_of_roses_roses_count_l1745_174599

theorem number_of_roses (vase_capacity : ℕ) (carnations : ℕ) (vases : ℕ) : ℕ :=
  let total_flowers := vase_capacity * vases
  total_flowers - carnations

theorem roses_count : number_of_roses 6 7 9 = 47 := by
  sorry

end NUMINAMATH_CALUDE_number_of_roses_roses_count_l1745_174599


namespace NUMINAMATH_CALUDE_two_pairs_probability_l1745_174548

-- Define the total number of socks
def total_socks : ℕ := 10

-- Define the number of colors
def num_colors : ℕ := 5

-- Define the number of socks per color
def socks_per_color : ℕ := 2

-- Define the number of socks drawn
def socks_drawn : ℕ := 4

-- Define the probability of drawing two pairs of different colors
def prob_two_pairs : ℚ := 1 / 21

-- Theorem statement
theorem two_pairs_probability :
  (Nat.choose num_colors 2) / (Nat.choose total_socks socks_drawn) = prob_two_pairs :=
sorry

end NUMINAMATH_CALUDE_two_pairs_probability_l1745_174548


namespace NUMINAMATH_CALUDE_paint_coverage_l1745_174582

/-- Proves that a quart of paint covers 60 square feet given the specified conditions -/
theorem paint_coverage (cube_edge : Real) (paint_cost_per_quart : Real) (total_paint_cost : Real)
  (h1 : cube_edge = 10)
  (h2 : paint_cost_per_quart = 3.2)
  (h3 : total_paint_cost = 32) :
  (6 * cube_edge^2) / (total_paint_cost / paint_cost_per_quart) = 60 := by
  sorry

end NUMINAMATH_CALUDE_paint_coverage_l1745_174582


namespace NUMINAMATH_CALUDE_compare_powers_l1745_174585

theorem compare_powers (x m n : ℝ) (hx_pos : x > 0) (hx_neq_one : x ≠ 1) (hm_gt_n : m > n) (hn_pos : n > 0) :
  x^m + 1/x^m > x^n + 1/x^n := by
  sorry

end NUMINAMATH_CALUDE_compare_powers_l1745_174585


namespace NUMINAMATH_CALUDE_wendy_shoes_theorem_l1745_174509

/-- The number of pairs of shoes Wendy gave away -/
def shoes_given_away : ℕ := 14

/-- The number of pairs of shoes Wendy kept -/
def shoes_kept : ℕ := 19

/-- The total number of pairs of shoes Wendy had -/
def total_shoes : ℕ := shoes_given_away + shoes_kept

theorem wendy_shoes_theorem : total_shoes = 33 := by
  sorry

end NUMINAMATH_CALUDE_wendy_shoes_theorem_l1745_174509


namespace NUMINAMATH_CALUDE_pyramid_height_equals_cube_volume_l1745_174501

theorem pyramid_height_equals_cube_volume (cube_edge : ℝ) (pyramid_base : ℝ) (pyramid_height : ℝ) : 
  cube_edge = 5 →
  pyramid_base = 10 →
  cube_edge ^ 3 = (1 / 3) * pyramid_base ^ 2 * pyramid_height →
  pyramid_height = 3.75 := by
sorry

end NUMINAMATH_CALUDE_pyramid_height_equals_cube_volume_l1745_174501


namespace NUMINAMATH_CALUDE_quadratic_set_single_element_l1745_174528

theorem quadratic_set_single_element (k : ℝ) :
  (∃! x : ℝ, k * x^2 + 4 * x + 4 = 0) → k = 0 ∨ k = 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_set_single_element_l1745_174528


namespace NUMINAMATH_CALUDE_derivative_ln_over_x_l1745_174539

open Real

theorem derivative_ln_over_x (x : ℝ) (h : x > 0) :
  deriv (λ x => (log x) / x) x = (1 - log x) / x^2 := by
  sorry

end NUMINAMATH_CALUDE_derivative_ln_over_x_l1745_174539


namespace NUMINAMATH_CALUDE_alice_has_winning_strategy_l1745_174516

/-- Represents a position on the circular table -/
structure Position :=
  (x : ℝ)
  (y : ℝ)

/-- Represents the game state -/
structure GameState :=
  (placedCoins : Set Position)
  (currentPlayer : Bool)  -- true for Alice, false for Bob

/-- Checks if a move is valid given the current game state -/
def isValidMove (state : GameState) (pos : Position) : Prop :=
  pos ∉ state.placedCoins ∧ pos.x^2 + pos.y^2 ≤ 1

/-- Defines a winning strategy for a player -/
def hasWinningStrategy (player : Bool) : Prop :=
  ∀ (state : GameState), 
    state.currentPlayer = player → 
    ∃ (move : Position), isValidMove state move ∧
      ¬∃ (opponentMove : Position), 
        isValidMove (GameState.mk (state.placedCoins ∪ {move}) (¬player)) opponentMove

/-- The main theorem stating that Alice (the starting player) has a winning strategy -/
theorem alice_has_winning_strategy : 
  hasWinningStrategy true :=
sorry

end NUMINAMATH_CALUDE_alice_has_winning_strategy_l1745_174516


namespace NUMINAMATH_CALUDE_equation_solution_l1745_174569

theorem equation_solution : ∃ x : ℚ, (4 / 7) * (1 / 5) * x + 2 = 8 ∧ x = 105 / 2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1745_174569


namespace NUMINAMATH_CALUDE_square_of_binomial_constant_l1745_174505

theorem square_of_binomial_constant (a : ℝ) : 
  (∃ b : ℝ, ∀ x : ℝ, 9*x^2 + 30*x + a = (3*x + b)^2) → a = 25 := by
  sorry

end NUMINAMATH_CALUDE_square_of_binomial_constant_l1745_174505


namespace NUMINAMATH_CALUDE_opposite_of_two_l1745_174545

-- Define the concept of opposite
def opposite (a : ℝ) : ℝ := -a

-- State the theorem
theorem opposite_of_two : opposite 2 = -2 := by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_opposite_of_two_l1745_174545


namespace NUMINAMATH_CALUDE_find_m_l1745_174515

/-- Given two functions f and g, prove that if 3f(5) = g(5), then m = 10 -/
theorem find_m (m : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ x^2 - 3*x + m
  let g : ℝ → ℝ := λ x ↦ x^2 - 3*x + 5*m
  3 * f 5 = g 5 → m = 10 := by
sorry

end NUMINAMATH_CALUDE_find_m_l1745_174515


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l1745_174546

theorem quadratic_inequality_solution_set 
  (a b : ℝ) 
  (h : Set.Ioo 2 3 = {x : ℝ | x^2 - a*x - b < 0}) : 
  {x : ℝ | b*x^2 - a*x - 1 > 0} = Set.Ioo (-1/2) (-1/3) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l1745_174546


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_inequality_l1745_174575

theorem negation_of_existence (P : ℝ → Prop) :
  (¬∃x < 1, P x) ↔ (∀x < 1, ¬P x) := by sorry

theorem negation_of_quadratic_inequality :
  (¬∃x < 1, x^2 + 2*x + 1 ≤ 0) ↔ (∀x < 1, x^2 + 2*x + 1 > 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_inequality_l1745_174575


namespace NUMINAMATH_CALUDE_weight_gain_proof_l1745_174553

/-- Calculates the final weight after muscle and fat gain -/
def final_weight (initial_weight : ℝ) (muscle_gain_percent : ℝ) (fat_gain_ratio : ℝ) : ℝ :=
  let muscle_gain := initial_weight * muscle_gain_percent
  let fat_gain := muscle_gain * fat_gain_ratio
  initial_weight + muscle_gain + fat_gain

/-- Proves that given the specified conditions, the final weight is 150 kg -/
theorem weight_gain_proof :
  final_weight 120 0.2 0.25 = 150 := by
  sorry

end NUMINAMATH_CALUDE_weight_gain_proof_l1745_174553


namespace NUMINAMATH_CALUDE_unique_modular_congruence_l1745_174513

theorem unique_modular_congruence :
  ∃! n : ℕ, 0 ≤ n ∧ n ≤ 5 ∧ n ≡ -7458 [ZMOD 6] := by
  sorry

end NUMINAMATH_CALUDE_unique_modular_congruence_l1745_174513


namespace NUMINAMATH_CALUDE_race_probability_l1745_174547

theorem race_probability (p_x p_y p_z : ℚ) : 
  p_x = 1/8 →
  p_y = 1/12 →
  p_x + p_y + p_z = 375/1000 →
  p_z = 1/6 := by
  sorry

end NUMINAMATH_CALUDE_race_probability_l1745_174547


namespace NUMINAMATH_CALUDE_exists_sequence_satisfying_conditions_l1745_174536

/-- The number of distinct prime factors shared by two positive integers -/
def d (m n : ℕ+) : ℕ := sorry

/-- The existence of a sequence satisfying the given conditions -/
theorem exists_sequence_satisfying_conditions :
  ∃ (a : ℕ+ → ℕ+),
    (a 1 ≥ 2018^2018) ∧
    (∀ m n, m ≤ n → a m ≤ a n) ∧
    (∀ m n, m ≠ n → d m n = d (a m) (a n)) :=
  sorry

end NUMINAMATH_CALUDE_exists_sequence_satisfying_conditions_l1745_174536


namespace NUMINAMATH_CALUDE_sin_plus_one_quasi_odd_quasi_odd_to_odd_cubic_quasi_odd_l1745_174550

/-- Definition of a quasi-odd function -/
def QuasiOdd (f : ℝ → ℝ) : Prop :=
  ∃ (a b : ℝ), (a ≠ 0 ∨ b ≠ 0) ∧ ∀ x, f (a + x) + f (a - x) = 2 * b

/-- Central point of a quasi-odd function -/
def CentralPoint (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  (a ≠ 0 ∨ b ≠ 0) ∧ ∀ x, f (a + x) + f (a - x) = 2 * b

/-- Theorem stating that sin(x) + 1 is a quasi-odd function with central point (0, 1) -/
theorem sin_plus_one_quasi_odd :
  QuasiOdd (fun x ↦ Real.sin x + 1) ∧ CentralPoint (fun x ↦ Real.sin x + 1) 0 1 := by
  sorry

/-- Theorem stating that if f is quasi-odd with central point (a, f(a)),
    then F(x) = f(x+a) - f(a) is odd -/
theorem quasi_odd_to_odd (f : ℝ → ℝ) (a : ℝ) :
  QuasiOdd f ∧ CentralPoint f a (f a) →
  ∀ x, f ((x + a) + a) - f a = -(f ((-x + a) + a) - f a) := by
  sorry

/-- Theorem stating that x^3 - 3x^2 + 6x - 2 is a quasi-odd function with central point (1, 2) -/
theorem cubic_quasi_odd :
  QuasiOdd (fun x ↦ x^3 - 3*x^2 + 6*x - 2) ∧
  CentralPoint (fun x ↦ x^3 - 3*x^2 + 6*x - 2) 1 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_plus_one_quasi_odd_quasi_odd_to_odd_cubic_quasi_odd_l1745_174550


namespace NUMINAMATH_CALUDE_right_pyramid_height_l1745_174571

/-- The height of a right pyramid with a square base -/
theorem right_pyramid_height (perimeter base_side diagonal_half : ℝ) 
  (apex_to_vertex : ℝ) (h_perimeter : perimeter = 40) 
  (h_base_side : base_side = perimeter / 4)
  (h_diagonal_half : diagonal_half = base_side * Real.sqrt 2 / 2)
  (h_apex_to_vertex : apex_to_vertex = 15) : 
  Real.sqrt (apex_to_vertex ^ 2 - diagonal_half ^ 2) = 5 * Real.sqrt 7 := by
  sorry

#check right_pyramid_height

end NUMINAMATH_CALUDE_right_pyramid_height_l1745_174571


namespace NUMINAMATH_CALUDE_cost_of_second_set_l1745_174510

/-- The cost of a set of pencils and pens -/
def cost_set (pencil_count : ℕ) (pen_count : ℕ) (pencil_cost : ℚ) (pen_cost : ℚ) : ℚ :=
  pencil_count * pencil_cost + pen_count * pen_cost

/-- The theorem stating that the cost of 4 pencils and 5 pens is 2.00 dollars -/
theorem cost_of_second_set :
  ∃ (pen_cost : ℚ),
    cost_set 4 5 0.1 pen_cost = 2 ∧
    cost_set 4 5 0.1 pen_cost = cost_set 4 5 0.1 ((2 - 4 * 0.1) / 5) :=
by
  sorry

end NUMINAMATH_CALUDE_cost_of_second_set_l1745_174510


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1745_174592

theorem inequality_solution_set (x : ℝ) : 
  (x - 2) * (x + 2) < 5 ↔ -3 < x ∧ x < 3 := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1745_174592


namespace NUMINAMATH_CALUDE_parabola_point_relationship_l1745_174519

/-- Parabola function -/
def f (x m : ℝ) : ℝ := x^2 - 4*x - m

theorem parabola_point_relationship (m : ℝ) :
  let y₁ := f 2 m
  let y₂ := f (-3) m
  let y₃ := f (-1) m
  y₁ < y₃ ∧ y₃ < y₂ := by sorry

end NUMINAMATH_CALUDE_parabola_point_relationship_l1745_174519


namespace NUMINAMATH_CALUDE_lead_in_mixture_l1745_174590

theorem lead_in_mixture (total : ℝ) (copper_weight : ℝ) (lead_percent : ℝ) (copper_percent : ℝ)
  (h1 : copper_weight = 12)
  (h2 : copper_percent = 0.60)
  (h3 : lead_percent = 0.25)
  (h4 : copper_weight = copper_percent * total) :
  lead_percent * total = 5 := by
sorry

end NUMINAMATH_CALUDE_lead_in_mixture_l1745_174590


namespace NUMINAMATH_CALUDE_smallest_n_for_inequality_l1745_174577

theorem smallest_n_for_inequality : ∃ (n : ℕ), n = 2 ∧ 
  (∀ (x y : ℝ), (x^2 + y^2)^2 ≤ n * (x^4 + y^4)) ∧ 
  (∀ (m : ℕ), m < n → ∃ (x y : ℝ), (x^2 + y^2)^2 > m * (x^4 + y^4)) := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_for_inequality_l1745_174577


namespace NUMINAMATH_CALUDE_evaluate_expression_l1745_174542

theorem evaluate_expression (b y : ℤ) (h : y = b + 9) : y - b + 5 = 14 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l1745_174542


namespace NUMINAMATH_CALUDE_triangle_special_case_l1745_174540

/-- Given a triangle with sides a, b, and c satisfying (a + b + c)(a + b - c) = 4ab,
    the angle opposite side c is 0 or 2π. -/
theorem triangle_special_case (a b c : ℝ) (h : (a + b + c) * (a + b - c) = 4 * a * b) :
  let C := Real.arccos ((a^2 + b^2 - c^2) / (2 * a * b))
  C = 0 ∨ C = 2 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_triangle_special_case_l1745_174540


namespace NUMINAMATH_CALUDE_expected_value_of_special_die_l1745_174552

def die_faces : List ℕ := [1, 4, 9, 16, 25, 36, 49, 64, 81, 100, 121, 144]

theorem expected_value_of_special_die :
  (List.sum die_faces) / (List.length die_faces : ℚ) = 650 / 12 := by
  sorry

end NUMINAMATH_CALUDE_expected_value_of_special_die_l1745_174552


namespace NUMINAMATH_CALUDE_grill_run_time_theorem_l1745_174526

/-- Represents the time a charcoal grill runs given the rate of coal burning and the amount of coal available -/
def grillRunTime (burnRate : ℕ) (burnTime : ℕ) (bags : ℕ) (coalsPerBag : ℕ) : ℚ :=
  let totalCoals := bags * coalsPerBag
  let minutesPerCycle := burnTime * (totalCoals / burnRate)
  minutesPerCycle / 60

/-- Theorem stating that a grill burning 15 coals every 20 minutes, with 3 bags of 60 coals each, runs for 4 hours -/
theorem grill_run_time_theorem :
  grillRunTime 15 20 3 60 = 4 := by
  sorry

#eval grillRunTime 15 20 3 60

end NUMINAMATH_CALUDE_grill_run_time_theorem_l1745_174526


namespace NUMINAMATH_CALUDE_max_sin_cos_sum_l1745_174507

theorem max_sin_cos_sum (A : Real) : 2 * Real.sin (A / 2) + Real.cos (A / 2) ≤ Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_max_sin_cos_sum_l1745_174507


namespace NUMINAMATH_CALUDE_product_mod_25_l1745_174525

theorem product_mod_25 : 68 * 97 * 113 ≡ 23 [ZMOD 25] := by sorry

end NUMINAMATH_CALUDE_product_mod_25_l1745_174525


namespace NUMINAMATH_CALUDE_fraction_equality_l1745_174574

theorem fraction_equality : (1992^2 - 1985^2) / (2001^2 - 1976^2) = 7 / 25 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1745_174574


namespace NUMINAMATH_CALUDE_compare_x_y_l1745_174517

theorem compare_x_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (hx4 : x^4 = 2) (hy3 : y^3 = 3) : x < y := by
  sorry

end NUMINAMATH_CALUDE_compare_x_y_l1745_174517


namespace NUMINAMATH_CALUDE_farmer_pomelo_shipment_l1745_174597

/-- Calculates the total number of dozens of pomelos shipped given the number of boxes and pomelos from last week and the number of boxes shipped this week. -/
def totalDozensShipped (lastWeekBoxes : ℕ) (lastWeekPomelos : ℕ) (thisWeekBoxes : ℕ) : ℕ :=
  let pomelosPerBox := lastWeekPomelos / lastWeekBoxes
  let totalPomelos := lastWeekPomelos + thisWeekBoxes * pomelosPerBox
  totalPomelos / 12

/-- Proves that given 10 boxes containing 240 pomelos in total last week, and 20 boxes shipped this week, the total number of dozens of pomelos shipped is 40. -/
theorem farmer_pomelo_shipment :
  totalDozensShipped 10 240 20 = 40 := by
  sorry

end NUMINAMATH_CALUDE_farmer_pomelo_shipment_l1745_174597


namespace NUMINAMATH_CALUDE_intersection_point_m_value_l1745_174521

theorem intersection_point_m_value (m : ℝ) :
  (∃ x y : ℝ, y = m * x + 3 ∧ y = x + 1 ∧ y = -x) → m = 5 := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_m_value_l1745_174521


namespace NUMINAMATH_CALUDE_sqrt_combinable_with_sqrt_6_l1745_174573

theorem sqrt_combinable_with_sqrt_6 :
  ∀ x : ℝ, x > 0 →
  (x = 12 ∨ x = 15 ∨ x = 18 ∨ x = 24) →
  (∃ q : ℚ, Real.sqrt x = q * Real.sqrt 6) ↔ x = 24 := by
sorry

end NUMINAMATH_CALUDE_sqrt_combinable_with_sqrt_6_l1745_174573


namespace NUMINAMATH_CALUDE_norris_september_savings_l1745_174554

/-- The amount of money Norris saved in September -/
def september_savings : ℕ := sorry

/-- The amount of money Norris saved in October -/
def october_savings : ℕ := 25

/-- The amount of money Norris saved in November -/
def november_savings : ℕ := 31

/-- The amount of money Norris spent on an online game -/
def game_cost : ℕ := 75

/-- The amount of money Norris has left -/
def money_left : ℕ := 10

/-- Theorem stating that Norris saved $29 in September -/
theorem norris_september_savings :
  september_savings = 29 :=
by
  sorry

end NUMINAMATH_CALUDE_norris_september_savings_l1745_174554


namespace NUMINAMATH_CALUDE_bus_problem_l1745_174566

/-- The number of children on a bus after a stop, given the initial number,
    the number who got off, and the difference between those who got off and on. -/
def children_after_stop (initial : ℕ) (got_off : ℕ) (diff : ℕ) : ℤ :=
  initial - got_off + (got_off - diff)

/-- Theorem stating that given the initial conditions, 
    the number of children on the bus after the stop is 12. -/
theorem bus_problem : children_after_stop 36 68 24 = 12 := by
  sorry

end NUMINAMATH_CALUDE_bus_problem_l1745_174566


namespace NUMINAMATH_CALUDE_marks_initial_friends_l1745_174587

/-- Calculates the initial number of friends Mark had -/
def initial_friends (kept_percentage : ℚ) (contacted_percentage : ℚ) (response_rate : ℚ) (final_friends : ℕ) : ℚ :=
  final_friends / (kept_percentage + contacted_percentage * response_rate)

/-- Proves that Mark initially had 100 friends -/
theorem marks_initial_friends :
  let kept_percentage : ℚ := 2/5
  let contacted_percentage : ℚ := 3/5
  let response_rate : ℚ := 1/2
  let final_friends : ℕ := 70
  initial_friends kept_percentage contacted_percentage response_rate final_friends = 100 := by
  sorry

#eval initial_friends (2/5) (3/5) (1/2) 70

end NUMINAMATH_CALUDE_marks_initial_friends_l1745_174587


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1745_174551

theorem inequality_solution_set (a : ℝ) (h : a > 0) :
  let S := {x : ℝ | x + 1/x > a}
  (a > 1 → S = {x | 0 < x ∧ x < 1/a} ∪ {x | x > a}) ∧
  (a = 1 → S = {x | x > 0 ∧ x ≠ 1}) ∧
  (0 < a ∧ a < 1 → S = {x | 0 < x ∧ x < a} ∪ {x | x > 1/a}) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1745_174551


namespace NUMINAMATH_CALUDE_arithmetic_sequence_general_term_l1745_174514

/-- An arithmetic sequence with specific conditions -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  (∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d) ∧ 
  a 3 = 7 ∧ 
  a 5 = a 2 + 6

/-- The general term formula for the arithmetic sequence -/
def GeneralTerm (n : ℕ) : ℝ := 2 * n + 1

/-- Theorem stating that the general term formula is correct for the given arithmetic sequence -/
theorem arithmetic_sequence_general_term (a : ℕ → ℝ) (h : ArithmeticSequence a) : 
  ∀ n : ℕ, a n = GeneralTerm n :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_general_term_l1745_174514


namespace NUMINAMATH_CALUDE_trig_identity_proof_l1745_174511

theorem trig_identity_proof (θ : Real) (h : Real.tan θ = 2) : 
  Real.sin θ ^ 2 + Real.sin θ * Real.cos θ - 2 * Real.cos θ ^ 2 = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_proof_l1745_174511


namespace NUMINAMATH_CALUDE_bus_speed_on_national_road_l1745_174559

/-- The speed of a bus on the original national road, given specific conditions about a new highway --/
theorem bus_speed_on_national_road :
  ∀ (x : ℝ),
    (200 : ℝ) / (x + 45) = (220 : ℝ) / x / 2 →
    x = 55 := by
  sorry

end NUMINAMATH_CALUDE_bus_speed_on_national_road_l1745_174559


namespace NUMINAMATH_CALUDE_first_player_wins_6x8_l1745_174508

/-- Represents a chocolate bar game --/
structure ChocolateGame where
  rows : Nat
  cols : Nat

/-- Calculates the number of moves required to completely break the chocolate bar --/
def totalMoves (game : ChocolateGame) : Nat :=
  game.rows * game.cols - 1

/-- Determines the winner of the game --/
def firstPlayerWins (game : ChocolateGame) : Prop :=
  Odd (totalMoves game)

/-- Theorem stating that the first player wins in a 6x8 chocolate bar game --/
theorem first_player_wins_6x8 :
  firstPlayerWins { rows := 6, cols := 8 } := by
  sorry

end NUMINAMATH_CALUDE_first_player_wins_6x8_l1745_174508


namespace NUMINAMATH_CALUDE_least_number_with_remainder_l1745_174579

theorem least_number_with_remainder (n : ℕ) : n = 256 ↔ 
  (∀ m, m < n → ¬(m % 7 = 4 ∧ m % 9 = 4 ∧ m % 12 = 4 ∧ m % 18 = 4)) ∧
  n % 7 = 4 ∧ n % 9 = 4 ∧ n % 12 = 4 ∧ n % 18 = 4 :=
by sorry

end NUMINAMATH_CALUDE_least_number_with_remainder_l1745_174579


namespace NUMINAMATH_CALUDE_max_sum_of_digits_24hour_clock_l1745_174568

/-- Represents a time in 24-hour format -/
structure Time24 where
  hours : Nat
  minutes : Nat
  hour_valid : hours < 24
  minute_valid : minutes < 60

/-- Calculates the sum of digits for a natural number -/
def sumOfDigits (n : Nat) : Nat :=
  let digits := n.repr.toList.map (fun c => c.toString.toNat!)
  digits.sum

/-- Calculates the sum of digits for a Time24 -/
def timeSumOfDigits (t : Time24) : Nat :=
  sumOfDigits t.hours + sumOfDigits t.minutes

/-- The largest possible sum of digits in a 24-hour digital clock display is 24 -/
theorem max_sum_of_digits_24hour_clock :
  (∀ t : Time24, timeSumOfDigits t ≤ 24) ∧
  (∃ t : Time24, timeSumOfDigits t = 24) :=
sorry

end NUMINAMATH_CALUDE_max_sum_of_digits_24hour_clock_l1745_174568


namespace NUMINAMATH_CALUDE_train_speed_l1745_174538

/-- The speed of a train given its length, time to cross a walking man, and the man's speed --/
theorem train_speed (train_length : ℝ) (crossing_time : ℝ) (man_speed_kmh : ℝ) : 
  train_length = 700 →
  crossing_time = 41.9966402687785 →
  man_speed_kmh = 3 →
  ∃ (train_speed_kmh : ℝ), abs (train_speed_kmh - 63.0036) < 0.0001 := by
  sorry


end NUMINAMATH_CALUDE_train_speed_l1745_174538


namespace NUMINAMATH_CALUDE_expand_product_l1745_174583

theorem expand_product (x : ℝ) : 3 * (x - 2) * (x^2 + x + 1) = 3*x^3 - 3*x^2 - 3*x - 6 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l1745_174583


namespace NUMINAMATH_CALUDE_smallest_possible_area_l1745_174572

theorem smallest_possible_area (S : ℕ) (A : ℕ) : 
  S * S = 2019 + A  -- Total area equation
  → A ≠ 1  -- Area of 2020th square is not 1
  → A ≥ 9  -- Smallest possible area is at least 9
  → ∃ (S' : ℕ), S' * S' = 2019 + 9  -- There exists a solution with area 9
  → A = 9  -- The smallest area is indeed 9
  := by sorry

end NUMINAMATH_CALUDE_smallest_possible_area_l1745_174572


namespace NUMINAMATH_CALUDE_ben_bonus_allocation_l1745_174589

theorem ben_bonus_allocation (bonus : ℚ) (holiday_fraction : ℚ) (gift_fraction : ℚ) (remaining : ℚ) 
  (h1 : bonus = 1496)
  (h2 : holiday_fraction = 1/4)
  (h3 : gift_fraction = 1/8)
  (h4 : remaining = 867) :
  let kitchen_fraction := (bonus - remaining - holiday_fraction * bonus - gift_fraction * bonus) / bonus
  kitchen_fraction = 221/748 := by sorry

end NUMINAMATH_CALUDE_ben_bonus_allocation_l1745_174589


namespace NUMINAMATH_CALUDE_inequality_preservation_l1745_174581

theorem inequality_preservation (a b : ℝ) (h : a > b) : a - 3 > b - 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_preservation_l1745_174581


namespace NUMINAMATH_CALUDE_unique_prime_product_sum_of_cubes_l1745_174503

/-- The nth prime number -/
def nthPrime (n : ℕ) : ℕ := sorry

/-- Product of the first n prime numbers -/
def primeProduct (n : ℕ) : ℕ := sorry

/-- Predicate to check if a number is expressible as the sum of two positive cubes -/
def isSumOfTwoCubes (n : ℕ) : Prop := ∃ a b : ℕ+, n = a^3 + b^3

theorem unique_prime_product_sum_of_cubes :
  ∀ k : ℕ+, (k = 1 ↔ isSumOfTwoCubes (primeProduct k)) := by sorry

end NUMINAMATH_CALUDE_unique_prime_product_sum_of_cubes_l1745_174503


namespace NUMINAMATH_CALUDE_rectangle_area_l1745_174563

/-- Represents a square in the rectangle --/
structure Square where
  sideLength : ℝ
  area : ℝ
  area_def : area = sideLength ^ 2

/-- The rectangle XYZW with its squares --/
structure Rectangle where
  smallSquares : Fin 3 → Square
  largeSquare : Square
  smallSquaresEqual : ∀ i j : Fin 3, (smallSquares i).sideLength = (smallSquares j).sideLength
  smallSquareArea : ∀ i : Fin 3, (smallSquares i).area = 4
  largeSquareSideLength : largeSquare.sideLength = 2 * (smallSquares 0).sideLength
  noOverlap : True  -- This condition is simplified as it's hard to represent geometrically

/-- The theorem to prove --/
theorem rectangle_area (rect : Rectangle) : 
  (3 * (rect.smallSquares 0).area + rect.largeSquare.area : ℝ) = 28 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l1745_174563


namespace NUMINAMATH_CALUDE_largest_digit_divisible_by_six_l1745_174541

/-- The largest single-digit number N such that 5678N is divisible by 6 is 4 -/
theorem largest_digit_divisible_by_six : 
  (∀ N : ℕ, N ≤ 9 → 56780 + N ≤ 56789 → (56780 + N) % 6 = 0 → N ≤ 4) ∧ 
  (56784 % 6 = 0) := by
  sorry

end NUMINAMATH_CALUDE_largest_digit_divisible_by_six_l1745_174541


namespace NUMINAMATH_CALUDE_brittany_rebecca_age_difference_l1745_174520

/-- The age difference between Brittany and Rebecca -/
def ageDifference (rebecca_age : ℕ) (brittany_age_after_vacation : ℕ) (vacation_duration : ℕ) : ℕ :=
  brittany_age_after_vacation - vacation_duration - rebecca_age

/-- Proof that Brittany is 3 years older than Rebecca -/
theorem brittany_rebecca_age_difference :
  ageDifference 25 32 4 = 3 := by
  sorry

end NUMINAMATH_CALUDE_brittany_rebecca_age_difference_l1745_174520


namespace NUMINAMATH_CALUDE_tournament_battles_one_team_remains_l1745_174580

/-- The number of battles needed to determine a champion in a tournament --/
def battles_to_champion (initial_teams : ℕ) : ℕ :=
  if initial_teams ≤ 1 then 0
  else if initial_teams = 2 then 1
  else (initial_teams - 1) / 2

/-- Theorem: In a tournament with 2017 teams, 1008 battles are needed to determine a champion --/
theorem tournament_battles :
  battles_to_champion 2017 = 1008 := by
  sorry

/-- Lemma: The number of teams remaining after n battles --/
lemma teams_remaining (initial_teams n : ℕ) : ℕ :=
  if n ≥ (initial_teams - 1) / 2 then 1
  else initial_teams - 2 * n

/-- Theorem: After 1008 battles, only one team remains in a tournament of 2017 teams --/
theorem one_team_remains :
  teams_remaining 2017 1008 = 1 := by
  sorry

end NUMINAMATH_CALUDE_tournament_battles_one_team_remains_l1745_174580


namespace NUMINAMATH_CALUDE_pure_imaginary_complex_number_l1745_174502

/-- Given that z = m²(1+i) - m(3+6i) is a pure imaginary number, 
    prove that m = 3 is the only real solution. -/
theorem pure_imaginary_complex_number (m : ℝ) : 
  let z : ℂ := m^2 * (1 + Complex.I) - m * (3 + 6 * Complex.I)
  (z.re = 0 ∧ z.im ≠ 0) → m = 3 :=
by sorry

end NUMINAMATH_CALUDE_pure_imaginary_complex_number_l1745_174502


namespace NUMINAMATH_CALUDE_snails_removed_l1745_174533

def original_snails : ℕ := 11760
def remaining_snails : ℕ := 8278

theorem snails_removed : original_snails - remaining_snails = 3482 := by
  sorry

end NUMINAMATH_CALUDE_snails_removed_l1745_174533


namespace NUMINAMATH_CALUDE_gcd_of_polynomial_and_multiple_l1745_174560

theorem gcd_of_polynomial_and_multiple (x : ℤ) (h : ∃ k : ℤ, x = 46200 * k) :
  let f := fun (x : ℤ) => (3*x + 5) * (5*x + 3) * (11*x + 6) * (x + 11)
  Int.gcd (f x) x = 990 := by
sorry

end NUMINAMATH_CALUDE_gcd_of_polynomial_and_multiple_l1745_174560


namespace NUMINAMATH_CALUDE_min_value_sum_l1745_174565

theorem min_value_sum (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : 2 * a + b = 1) :
  ∀ x y : ℝ, 0 < x ∧ 0 < y ∧ 2 * x + y = 1 → a + b ≤ x + y ∧ a + b = 4 :=
sorry

end NUMINAMATH_CALUDE_min_value_sum_l1745_174565


namespace NUMINAMATH_CALUDE_perpendicular_line_plane_counterexample_l1745_174518

/-- A line in 3D space -/
structure Line3D where
  point : ℝ × ℝ × ℝ
  direction : ℝ × ℝ × ℝ

/-- A plane in 3D space -/
structure Plane where
  normal : ℝ × ℝ × ℝ
  point : ℝ × ℝ × ℝ

/-- Two lines intersect -/
def intersect (l1 l2 : Line3D) : Prop := sorry

/-- A line is perpendicular to another line -/
def perp_line (l1 l2 : Line3D) : Prop := sorry

/-- A line is perpendicular to a plane -/
def perp_plane (l : Line3D) (p : Plane) : Prop := sorry

/-- A line is within a plane -/
def line_in_plane (l : Line3D) (p : Plane) : Prop := sorry

theorem perpendicular_line_plane_counterexample :
  ∃ (l : Line3D) (p : Plane) (l1 l2 : Line3D),
    line_in_plane l1 p ∧
    line_in_plane l2 p ∧
    intersect l1 l2 ∧
    perp_line l l1 ∧
    perp_line l l2 ∧
    ¬(perp_plane l p) := by sorry

end NUMINAMATH_CALUDE_perpendicular_line_plane_counterexample_l1745_174518


namespace NUMINAMATH_CALUDE_min_value_sum_l1745_174524

theorem min_value_sum (a b c d e f : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) 
  (pos_d : 0 < d) (pos_e : 0 < e) (pos_f : 0 < f)
  (sum_eq_9 : a + b + c + d + e + f = 9) :
  (1/a) + (9/b) + (16/c) + (25/d) + (36/e) + (49/f) ≥ 676/9 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_l1745_174524


namespace NUMINAMATH_CALUDE_ratio_from_mean_ratio_l1745_174561

theorem ratio_from_mean_ratio {a b : ℝ} (ha : a > 0) (hb : b > 0) :
  (a + b) / (2 * Real.sqrt (a * b)) = 25 / 24 →
  a / b = 16 / 9 ∨ a / b = 9 / 16 := by
sorry

end NUMINAMATH_CALUDE_ratio_from_mean_ratio_l1745_174561


namespace NUMINAMATH_CALUDE_tangent_line_to_circle_l1745_174529

theorem tangent_line_to_circle (m : ℝ) : 
  (∃ (x y : ℝ), x - Real.sqrt 3 * y + m = 0 ∧ x^2 + y^2 - 2*y - 2 = 0) →
  (m = -Real.sqrt 3 ∨ m = 3 * Real.sqrt 3) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_to_circle_l1745_174529


namespace NUMINAMATH_CALUDE_net_population_increase_l1745_174544

/-- The net population increase in one day given specific birth and death rates -/
theorem net_population_increase (birth_rate : ℚ) (death_rate : ℚ) (seconds_per_day : ℕ) : 
  birth_rate = 5 / 2 → death_rate = 3 / 2 → seconds_per_day = 24 * 60 * 60 →
  (birth_rate - death_rate) * seconds_per_day = 86400 := by
  sorry

end NUMINAMATH_CALUDE_net_population_increase_l1745_174544


namespace NUMINAMATH_CALUDE_problem_statement_l1745_174596

theorem problem_statement (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : b^3 + b ≤ a - a^3) :
  (b < a ∧ a < 1) ∧ a^2 + b^2 < 1 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1745_174596


namespace NUMINAMATH_CALUDE_taxi_fare_calculation_l1745_174594

/-- Proves that the charge for each additional 1/5 mile is $0.40 --/
theorem taxi_fare_calculation (initial_charge : ℚ) (total_distance : ℚ) (total_charge : ℚ) :
  initial_charge = 280/100 →
  total_distance = 8 →
  total_charge = 1840/100 →
  let additional_distance : ℚ := total_distance - 1/5
  let additional_increments : ℚ := additional_distance / (1/5)
  let charge_per_increment : ℚ := (total_charge - initial_charge) / additional_increments
  charge_per_increment = 40/100 := by
  sorry

end NUMINAMATH_CALUDE_taxi_fare_calculation_l1745_174594


namespace NUMINAMATH_CALUDE_unique_five_digit_number_l1745_174593

def is_valid_digit (d : ℕ) : Prop := d ≥ 1 ∧ d ≤ 5

def digits_to_number (a b c : ℕ) : ℕ := 100 * a + 10 * b + c

theorem unique_five_digit_number 
  (P Q R S T : ℕ) 
  (h_valid : ∀ d ∈ [P, Q, R, S, T], is_valid_digit d)
  (h_distinct : P ≠ Q ∧ P ≠ R ∧ P ≠ S ∧ P ≠ T ∧ Q ≠ R ∧ Q ≠ S ∧ Q ≠ T ∧ R ≠ S ∧ R ≠ T ∧ S ≠ T)
  (h_div_4 : digits_to_number P Q R % 4 = 0)
  (h_div_5 : digits_to_number Q R S % 5 = 0)
  (h_div_3 : digits_to_number R S T % 3 = 0) :
  P = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_five_digit_number_l1745_174593


namespace NUMINAMATH_CALUDE_deepak_present_age_l1745_174564

/-- The ratio of ages between Rahul, Deepak, and Sameer -/
def age_ratio : Fin 3 → ℕ
  | 0 => 4  -- Rahul
  | 1 => 3  -- Deepak
  | 2 => 5  -- Sameer

/-- The number of years in the future we're considering -/
def years_future : ℕ := 6

/-- Rahul's age after the specified number of years -/
def rahul_future_age : ℕ := 26

/-- Proves that given the age ratio and Rahul's future age, Deepak's present age is 15 years -/
theorem deepak_present_age :
  ∃ (k : ℕ),
    (age_ratio 0 * k + years_future = rahul_future_age) ∧
    (age_ratio 1 * k = 15) := by
  sorry

end NUMINAMATH_CALUDE_deepak_present_age_l1745_174564


namespace NUMINAMATH_CALUDE_quadratic_equation_conversion_l1745_174532

theorem quadratic_equation_conversion :
  ∃ (a b c : ℝ), a = 1 ∧ b = -5 ∧ c = 3 ∧
  ∀ x, (x - 1)^2 = 3*x - 2 ↔ a*x^2 + b*x + c = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_conversion_l1745_174532


namespace NUMINAMATH_CALUDE_unique_solution_l1745_174506

/-- The solution set of the inequality |ax - 2| < 3 with respect to x -/
def SolutionSet (a : ℝ) : Set ℝ :=
  {x : ℝ | |a * x - 2| < 3}

/-- The given solution set -/
def GivenSet : Set ℝ :=
  {x : ℝ | -5/3 < x ∧ x < 1/3}

/-- The theorem stating that a = -3 is the unique value satisfying the conditions -/
theorem unique_solution : ∃! a : ℝ, SolutionSet a = GivenSet :=
sorry

end NUMINAMATH_CALUDE_unique_solution_l1745_174506


namespace NUMINAMATH_CALUDE_complex_expression_evaluation_l1745_174562

theorem complex_expression_evaluation (z : ℂ) (h : z = 1 + I) :
  2 / z + z^2 = 1 + I := by sorry

end NUMINAMATH_CALUDE_complex_expression_evaluation_l1745_174562


namespace NUMINAMATH_CALUDE_machine_count_l1745_174534

theorem machine_count (hours_R hours_S total_hours : ℕ) (h1 : hours_R = 36) (h2 : hours_S = 9) (h3 : total_hours = 12) :
  ∃ (n : ℕ), n > 0 ∧ n * (1 / hours_R + 1 / hours_S) = 1 / total_hours ∧ n = 15 :=
by sorry

end NUMINAMATH_CALUDE_machine_count_l1745_174534


namespace NUMINAMATH_CALUDE_go_pieces_theorem_l1745_174584

/-- Represents the set of Go pieces -/
structure GoPieces where
  white : ℕ
  black : ℕ

/-- Calculates the probability of drawing two pieces of the same color in the first two draws -/
def prob_same_color (pieces : GoPieces) : ℚ :=
  sorry

/-- Calculates the expected value of the number of white Go pieces drawn in the first four draws -/
def expected_white_pieces (pieces : GoPieces) : ℚ :=
  sorry

theorem go_pieces_theorem (pieces : GoPieces) 
  (h1 : pieces.white = 4) 
  (h2 : pieces.black = 3) : 
  prob_same_color pieces = 3/7 ∧ expected_white_pieces pieces = 16/7 := by
  sorry

end NUMINAMATH_CALUDE_go_pieces_theorem_l1745_174584


namespace NUMINAMATH_CALUDE_max_value_of_f_l1745_174504

open Real

noncomputable def f (x : ℝ) : ℝ := (log x) / x

theorem max_value_of_f :
  ∃ (c : ℝ), c > 0 ∧ ∀ (x : ℝ), x > 0 → f x ≤ 1 / Real.exp 1 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_f_l1745_174504


namespace NUMINAMATH_CALUDE_lcm_from_product_and_hcf_l1745_174556

theorem lcm_from_product_and_hcf (A B : ℕ+) :
  A * B = 62216 →
  Nat.gcd A B = 22 →
  Nat.lcm A B = 2828 := by
sorry

end NUMINAMATH_CALUDE_lcm_from_product_and_hcf_l1745_174556


namespace NUMINAMATH_CALUDE_triangular_prism_ratio_l1745_174591

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a triangular prism -/
structure TriangularPrism where
  A : Point3D
  B : Point3D
  C : Point3D
  A₁ : Point3D
  B₁ : Point3D
  C₁ : Point3D

/-- Checks if two planes are perpendicular -/
def arePlanesPerp (p1 p2 p3 q1 q2 q3 : Point3D) : Prop := sorry

/-- Checks if two vectors are perpendicular -/
def areVectorsPerp (v1 v2 : Point3D) : Prop := sorry

/-- Calculates the distance between two points -/
def distance (p1 p2 : Point3D) : ℝ := sorry

/-- Checks if a point lies on a line segment -/
def isOnLineSegment (p1 p2 p : Point3D) : Prop := sorry

/-- Main theorem -/
theorem triangular_prism_ratio 
  (prism : TriangularPrism)
  (D : Point3D)
  (h1 : distance prism.A prism.A₁ = 4)
  (h2 : distance prism.A prism.C = 4)
  (h3 : distance prism.A₁ prism.C₁ = 4)
  (h4 : distance prism.C prism.C₁ = 4)
  (h5 : arePlanesPerp prism.A prism.B prism.C prism.A prism.A₁ prism.C₁)
  (h6 : distance prism.A prism.B = 3)
  (h7 : distance prism.B prism.C = 5)
  (h8 : isOnLineSegment prism.B prism.C₁ D)
  (h9 : areVectorsPerp (Point3D.mk (D.x - prism.A.x) (D.y - prism.A.y) (D.z - prism.A.z))
                       (Point3D.mk (prism.B.x - prism.A₁.x) (prism.B.y - prism.A₁.y) (prism.B.z - prism.A₁.z))) :
  distance prism.B D / distance prism.B prism.C₁ = 9 / 25 := by
  sorry

end NUMINAMATH_CALUDE_triangular_prism_ratio_l1745_174591


namespace NUMINAMATH_CALUDE_range_of_sum_l1745_174586

theorem range_of_sum (a b : ℝ) (h : a^2 - a*b + b^2 = a + b) :
  ∃ t : ℝ, t = a + b ∧ 0 ≤ t ∧ t ≤ 4 :=
sorry

end NUMINAMATH_CALUDE_range_of_sum_l1745_174586


namespace NUMINAMATH_CALUDE_unique_modular_congruence_l1745_174523

theorem unique_modular_congruence : ∃! n : ℕ, 0 ≤ n ∧ n ≤ 13 ∧ n ≡ 1729 [ZMOD 13] := by
  sorry

end NUMINAMATH_CALUDE_unique_modular_congruence_l1745_174523


namespace NUMINAMATH_CALUDE_cookie_distribution_probability_l1745_174558

/-- Represents the number of cookies of each type -/
def num_cookies_per_type : ℕ := 4

/-- Represents the total number of cookies -/
def total_cookies : ℕ := 3 * num_cookies_per_type

/-- Represents the number of students -/
def num_students : ℕ := 4

/-- Represents the number of cookies each student receives -/
def cookies_per_student : ℕ := 3

/-- Calculates the probability of a specific distribution of cookies -/
def probability_distribution (n : ℕ) : ℚ :=
  (num_cookies_per_type * (num_cookies_per_type - 1) * (num_cookies_per_type - 2)) /
  ((total_cookies - n * cookies_per_student + 2) *
   (total_cookies - n * cookies_per_student + 1) *
   (total_cookies - n * cookies_per_student))

/-- The main theorem stating the probability of each student getting one cookie of each type -/
theorem cookie_distribution_probability :
  (probability_distribution 0 * probability_distribution 1 * probability_distribution 2) = 81 / 3850 := by
  sorry

end NUMINAMATH_CALUDE_cookie_distribution_probability_l1745_174558


namespace NUMINAMATH_CALUDE_weight_replacement_l1745_174567

theorem weight_replacement (n : ℕ) (old_weight new_weight avg_increase : ℝ) :
  n = 8 ∧
  new_weight = 65 ∧
  avg_increase = 2.5 →
  old_weight = new_weight - n * avg_increase :=
by sorry

end NUMINAMATH_CALUDE_weight_replacement_l1745_174567


namespace NUMINAMATH_CALUDE_quadratic_rewrite_l1745_174557

theorem quadratic_rewrite (a b c : ℤ) :
  (∀ x : ℝ, 16 * x^2 - 48 * x + 56 = (a * x + b)^2 + c) →
  a * b = -24 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_rewrite_l1745_174557
