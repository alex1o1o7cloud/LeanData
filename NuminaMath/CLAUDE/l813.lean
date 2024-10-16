import Mathlib

namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l813_81359

/-- Given a hyperbola C₁ and a parabola C₂ in the Cartesian coordinate system (xOy):
    C₁: x²/a² - y²/b² = 1 (a > 0, b > 0)
    C₂: x² = 2py (p > 0)
    
    The asymptotes of C₁ intersect with C₂ at points O, A, B.
    The orthocenter of triangle OAB is the focus of C₂.

    This theorem states that the eccentricity of C₁ is 3/2. -/
theorem hyperbola_eccentricity (a b p : ℝ) (ha : a > 0) (hb : b > 0) (hp : p > 0) : 
  let C₁ := fun (x y : ℝ) => x^2/a^2 - y^2/b^2 = 1
  let C₂ := fun (x y : ℝ) => x^2 = 2*p*y
  let asymptotes := fun (x y : ℝ) => y = (b/a)*x ∨ y = -(b/a)*x
  let O := (0, 0)
  let A := (2*p*b/a, 2*p*b^2/a^2)
  let B := (-2*p*b/a, 2*p*b^2/a^2)
  let focus := (0, p/2)
  let orthocenter := focus
  let eccentricity := Real.sqrt (1 + b^2/a^2)
  (∀ x y, asymptotes x y → C₂ x y → (x = 0 ∨ x = 2*p*b/a ∨ x = -2*p*b/a)) →
  (orthocenter = focus) →
  eccentricity = 3/2 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l813_81359


namespace NUMINAMATH_CALUDE_train_platform_crossing_time_l813_81326

/-- Calculates the time required for a train to cross a platform -/
theorem train_platform_crossing_time
  (train_speed_kmph : ℝ)
  (train_speed_ms : ℝ)
  (time_to_pass_man : ℝ)
  (platform_length : ℝ)
  (h1 : train_speed_kmph = 72)
  (h2 : train_speed_ms = 20)
  (h3 : time_to_pass_man = 16)
  (h4 : platform_length = 280)
  (h5 : train_speed_ms = train_speed_kmph * 1000 / 3600) :
  let train_length := train_speed_ms * time_to_pass_man
  let total_distance := train_length + platform_length
  total_distance / train_speed_ms = 30 := by
  sorry

end NUMINAMATH_CALUDE_train_platform_crossing_time_l813_81326


namespace NUMINAMATH_CALUDE_second_shirt_price_l813_81392

/-- Proves that the price of the second shirt must be $100 given the conditions --/
theorem second_shirt_price (total_shirts : Nat) (first_shirt_price third_shirt_price : ℝ)
  (remaining_shirts_min_avg : ℝ) (overall_avg : ℝ) :
  total_shirts = 10 →
  first_shirt_price = 82 →
  third_shirt_price = 90 →
  remaining_shirts_min_avg = 104 →
  overall_avg = 100 →
  ∃ (second_shirt_price : ℝ),
    second_shirt_price = 100 ∧
    (first_shirt_price + second_shirt_price + third_shirt_price +
      (total_shirts - 3) * remaining_shirts_min_avg) / total_shirts ≥ overall_avg :=
by sorry

end NUMINAMATH_CALUDE_second_shirt_price_l813_81392


namespace NUMINAMATH_CALUDE_inequality_solution_l813_81349

def solution_set : Set ℝ := {x | x < -2 ∨ (x ≥ 0 ∧ x < 2)}

theorem inequality_solution :
  {x : ℝ | x / (x^2 - 4) ≥ 0} = solution_set :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l813_81349


namespace NUMINAMATH_CALUDE_equation_solution_l813_81358

theorem equation_solution :
  let f (x : ℂ) := -x^2 - (4*x + 2)/(x + 2)
  ∃ (s : Finset ℂ), s.card = 3 ∧ 
    (∀ x ∈ s, f x = 0) ∧
    (∃ (a b : ℂ), s = {-1, a, b} ∧ a + b = -2 ∧ a * b = 0) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l813_81358


namespace NUMINAMATH_CALUDE_min_stamps_for_33_cents_l813_81380

def is_valid_combination (c f : ℕ) : Prop :=
  3 * c + 4 * f = 33

def total_stamps (c f : ℕ) : ℕ :=
  c + f

theorem min_stamps_for_33_cents :
  ∃ (c f : ℕ), is_valid_combination c f ∧
    total_stamps c f = 9 ∧
    ∀ (c' f' : ℕ), is_valid_combination c' f' →
      total_stamps c' f' ≥ 9 :=
sorry

end NUMINAMATH_CALUDE_min_stamps_for_33_cents_l813_81380


namespace NUMINAMATH_CALUDE_fraction_denominator_l813_81395

theorem fraction_denominator (x : ℕ) : 
  (4128 : ℚ) / x = 0.9411764705882353 → x = 4387 := by
  sorry

end NUMINAMATH_CALUDE_fraction_denominator_l813_81395


namespace NUMINAMATH_CALUDE_supplementary_angles_ratio_l813_81378

theorem supplementary_angles_ratio (a b : ℝ) : 
  a + b = 180 →  -- angles are supplementary
  a = 4 * b →    -- angles are in ratio 4:1
  b = 36 :=      -- smaller angle is 36°
by sorry

end NUMINAMATH_CALUDE_supplementary_angles_ratio_l813_81378


namespace NUMINAMATH_CALUDE_spruce_tree_height_l813_81374

theorem spruce_tree_height 
  (height_maple : ℝ) 
  (height_pine : ℝ) 
  (height_spruce : ℝ) 
  (h1 : height_maple = height_pine + 1)
  (h2 : height_pine = height_spruce - 4)
  (h3 : height_maple / height_spruce = 25 / 64) :
  height_spruce = 64 / 13 := by
  sorry

end NUMINAMATH_CALUDE_spruce_tree_height_l813_81374


namespace NUMINAMATH_CALUDE_valid_colorings_count_l813_81346

/-- A color used for vertex coloring -/
inductive Color
| Red
| White
| Blue

/-- A vertex in the triangle structure -/
structure Vertex :=
  (id : ℕ)
  (color : Color)

/-- A triangle in the structure -/
structure Triangle :=
  (vertices : Fin 3 → Vertex)

/-- The entire structure of three connected triangles -/
structure TriangleStructure :=
  (triangles : Fin 3 → Triangle)
  (middle_restricted : Vertex)

/-- Predicate to check if a coloring is valid -/
def is_valid_coloring (s : TriangleStructure) : Prop :=
  ∀ i j : Fin 3, ∀ k l : Fin 3,
    (s.triangles i).vertices k ≠ (s.triangles j).vertices l →
    ((s.triangles i).vertices k).color ≠ ((s.triangles j).vertices l).color

/-- Predicate to check if the middle restricted vertex is colored correctly -/
def is_middle_restricted_valid (s : TriangleStructure) : Prop :=
  s.middle_restricted.color = Color.Red ∨ s.middle_restricted.color = Color.White

/-- The number of valid colorings for the triangle structure -/
def num_valid_colorings : ℕ := 36

/-- Theorem stating that the number of valid colorings is 36 -/
theorem valid_colorings_count :
  ∀ s : TriangleStructure,
    is_valid_coloring s →
    is_middle_restricted_valid s →
    num_valid_colorings = 36 :=
sorry

end NUMINAMATH_CALUDE_valid_colorings_count_l813_81346


namespace NUMINAMATH_CALUDE_min_value_of_sum_min_value_is_4_plus_4sqrt3_l813_81333

theorem min_value_of_sum (x y : ℝ) : 
  x > 0 → y > 0 → (1 / (x + 1) + 1 / (y + 1) = 1 / 2) → 
  ∀ a b : ℝ, a > 0 → b > 0 → (1 / (a + 1) + 1 / (b + 1) = 1 / 2) → 
  x + 3 * y ≤ a + 3 * b :=
by sorry

theorem min_value_is_4_plus_4sqrt3 (x y : ℝ) :
  x > 0 → y > 0 → (1 / (x + 1) + 1 / (y + 1) = 1 / 2) →
  x + 3 * y = 4 + 4 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_sum_min_value_is_4_plus_4sqrt3_l813_81333


namespace NUMINAMATH_CALUDE_asterisk_value_for_solution_l813_81351

theorem asterisk_value_for_solution (x : ℝ) (asterisk : ℝ) :
  (2 * x - 7)^2 + (5 * x - asterisk)^2 = 0 → asterisk = 35/2 := by
  sorry

end NUMINAMATH_CALUDE_asterisk_value_for_solution_l813_81351


namespace NUMINAMATH_CALUDE_prob_king_or_queen_in_special_deck_l813_81337

structure Deck :=
  (total_cards : ℕ)
  (num_ranks : ℕ)
  (num_suits : ℕ)
  (cards_per_suit : ℕ)

def probability_king_or_queen (d : Deck) : ℚ :=
  let kings_and_queens := d.num_suits * 2
  kings_and_queens / d.total_cards

theorem prob_king_or_queen_in_special_deck :
  let d : Deck := {
    total_cards := 60,
    num_ranks := 15,
    num_suits := 4,
    cards_per_suit := 15
  }
  probability_king_or_queen d = 2 / 15 := by sorry

end NUMINAMATH_CALUDE_prob_king_or_queen_in_special_deck_l813_81337


namespace NUMINAMATH_CALUDE_probability_three_blue_marbles_l813_81393

/-- The number of red marbles in the jar -/
def red_marbles : ℕ := 4

/-- The number of blue marbles in the jar -/
def blue_marbles : ℕ := 5

/-- The number of white marbles in the jar -/
def white_marbles : ℕ := 8

/-- The number of green marbles in the jar -/
def green_marbles : ℕ := 3

/-- The total number of marbles in the jar -/
def total_marbles : ℕ := red_marbles + blue_marbles + white_marbles + green_marbles

/-- The number of marbles drawn -/
def marbles_drawn : ℕ := 3

theorem probability_three_blue_marbles :
  (blue_marbles : ℚ) / total_marbles *
  ((blue_marbles - 1) : ℚ) / (total_marbles - 1) *
  ((blue_marbles - 2) : ℚ) / (total_marbles - 2) = 1 / 114 :=
by sorry

end NUMINAMATH_CALUDE_probability_three_blue_marbles_l813_81393


namespace NUMINAMATH_CALUDE_modulo_thirteen_seven_l813_81305

theorem modulo_thirteen_seven (n : ℕ) : 
  13^7 ≡ n [ZMOD 7] → 0 ≤ n → n < 7 → n = 6 := by
  sorry

end NUMINAMATH_CALUDE_modulo_thirteen_seven_l813_81305


namespace NUMINAMATH_CALUDE_asha_remaining_money_l813_81353

/-- Calculates the remaining money for Asha after spending 3/4 of her total money --/
def remaining_money (brother_loan sister_loan father_loan mother_loan granny_gift savings : ℚ) : ℚ :=
  let total := brother_loan + sister_loan + father_loan + mother_loan + granny_gift + savings
  total - (3/4 * total)

/-- Theorem stating that Asha remains with $65 after spending --/
theorem asha_remaining_money :
  remaining_money 20 0 40 30 70 100 = 65 := by
  sorry

end NUMINAMATH_CALUDE_asha_remaining_money_l813_81353


namespace NUMINAMATH_CALUDE_fraction_equivalence_l813_81342

theorem fraction_equivalence (a b : ℚ) : 
  (a ≠ 0) → (b ≠ 0) → ((1 / (a / b)) * (5 / 6) = 1 / (5 / 2)) → (a / b = 25 / 12) := by
  sorry

end NUMINAMATH_CALUDE_fraction_equivalence_l813_81342


namespace NUMINAMATH_CALUDE_division_sum_dividend_l813_81300

theorem division_sum_dividend (quotient divisor remainder : ℕ) : 
  quotient = 40 → divisor = 72 → remainder = 64 → 
  (divisor * quotient) + remainder = 2944 := by
sorry

end NUMINAMATH_CALUDE_division_sum_dividend_l813_81300


namespace NUMINAMATH_CALUDE_system_solution_l813_81396

theorem system_solution (x y z : ℝ) : 
  x ≠ y ∧ y ≠ z ∧ x ≠ z →
  (x^2 + y^2 = -x + 3*y + z ∧
   y^2 + z^2 = x + 3*y - z ∧
   x^2 + z^2 = 2*x + 2*y - z) →
  ((x = 0 ∧ y = 1 ∧ z = -2) ∨ 
   (x = -3/2 ∧ y = 5/2 ∧ z = -1/2)) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l813_81396


namespace NUMINAMATH_CALUDE_negation_equivalence_l813_81331

theorem negation_equivalence (a : ℝ) : 
  (¬∃ x ∈ Set.Icc 1 2, x^2 - a < 0) ↔ (∀ x ∈ Set.Icc 1 2, x^2 ≥ a) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l813_81331


namespace NUMINAMATH_CALUDE_cover_room_with_tiles_l813_81311

/-- The length of the room -/
def room_length : ℝ := 8

/-- The width of the room -/
def room_width : ℝ := 12

/-- The length of a tile -/
def tile_length : ℝ := 1.5

/-- The width of a tile -/
def tile_width : ℝ := 2

/-- The number of tiles needed to cover the room -/
def tiles_needed : ℕ := 32

theorem cover_room_with_tiles : 
  (room_length * room_width) / (tile_length * tile_width) = tiles_needed := by
  sorry

end NUMINAMATH_CALUDE_cover_room_with_tiles_l813_81311


namespace NUMINAMATH_CALUDE_set_size_comparison_l813_81362

/-- The size of set A for a given n -/
def size_A (n : ℕ) : ℕ := n^3 + n^5 + n^7 + n^9

/-- The size of set B for a given m -/
def size_B (m : ℕ) : ℕ := m^2 + m^4 + m^6 + m^8

/-- Theorem stating the condition for |B| ≥ |A| when n = 6 -/
theorem set_size_comparison (m : ℕ) :
  size_B m ≥ size_A 6 ↔ m ≥ 8 := by
  sorry

end NUMINAMATH_CALUDE_set_size_comparison_l813_81362


namespace NUMINAMATH_CALUDE_cube_sum_divisible_by_nine_l813_81321

theorem cube_sum_divisible_by_nine (n : ℕ+) :
  ∃ k : ℤ, n^3 + (n+1)^3 + (n+2)^3 = 9 * k := by
sorry

end NUMINAMATH_CALUDE_cube_sum_divisible_by_nine_l813_81321


namespace NUMINAMATH_CALUDE_smallest_valid_number_last_four_digits_l813_81339

def is_valid_number (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ n.digits 10 → d = 2 ∨ d = 7

def has_both_digits (n : ℕ) : Prop :=
  2 ∈ n.digits 10 ∧ 7 ∈ n.digits 10

def last_four_digits (n : ℕ) : ℕ :=
  n % 10000

theorem smallest_valid_number_last_four_digits :
  ∃ m : ℕ,
    m > 0 ∧
    m % 5 = 0 ∧
    m % 7 = 0 ∧
    is_valid_number m ∧
    has_both_digits m ∧
    (∀ k : ℕ, k > 0 ∧ k % 5 = 0 ∧ k % 7 = 0 ∧ is_valid_number k ∧ has_both_digits k → m ≤ k) ∧
    last_four_digits m = 2772 :=
sorry

end NUMINAMATH_CALUDE_smallest_valid_number_last_four_digits_l813_81339


namespace NUMINAMATH_CALUDE_gcd_lcm_45_150_l813_81344

theorem gcd_lcm_45_150 : 
  (Nat.gcd 45 150 = 15) ∧ (Nat.lcm 45 150 = 450) := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_45_150_l813_81344


namespace NUMINAMATH_CALUDE_inequality_solution_implies_m_l813_81308

theorem inequality_solution_implies_m (m : ℝ) : 
  (∀ x, mx + 2 > 0 ↔ x < 2) → m = -1 := by
sorry

end NUMINAMATH_CALUDE_inequality_solution_implies_m_l813_81308


namespace NUMINAMATH_CALUDE_group_size_problem_l813_81368

theorem group_size_problem (n : ℕ) (h : ℝ) : 
  (n : ℝ) * ((n : ℝ) * h) = 362525 → n = 5 := by
  sorry

end NUMINAMATH_CALUDE_group_size_problem_l813_81368


namespace NUMINAMATH_CALUDE_shaded_area_calculation_l813_81391

theorem shaded_area_calculation (R : ℝ) (h : R = 9) :
  let r : ℝ := R / 2
  let larger_circle_area : ℝ := π * R^2
  let smaller_circle_area : ℝ := π * r^2
  let shaded_area : ℝ := larger_circle_area - 3 * smaller_circle_area
  shaded_area = 20.25 * π := by sorry

end NUMINAMATH_CALUDE_shaded_area_calculation_l813_81391


namespace NUMINAMATH_CALUDE_fraction_bounds_l813_81340

theorem fraction_bounds (n : ℕ+) : 1/2 ≤ (n : ℚ) / (n + 1) ∧ (n : ℚ) / (n + 1) < 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_bounds_l813_81340


namespace NUMINAMATH_CALUDE_radical_product_simplification_l813_81386

theorem radical_product_simplification (q : ℝ) (hq : q ≥ 0) :
  Real.sqrt (50 * q) * Real.sqrt (10 * q) * Real.sqrt (15 * q) = 50 * q * Real.sqrt q :=
by sorry

end NUMINAMATH_CALUDE_radical_product_simplification_l813_81386


namespace NUMINAMATH_CALUDE_manipulation_function_l813_81328

theorem manipulation_function (f : ℤ → ℤ) (h : 3 * (f 19 + 5) = 129) :
  ∀ x : ℤ, f x = 2 * x := by
  sorry

end NUMINAMATH_CALUDE_manipulation_function_l813_81328


namespace NUMINAMATH_CALUDE_part_one_part_two_l813_81373

-- Define the sets A, B, and U
def A (a : ℝ) : Set ℝ := {x | x - 3 ≤ x ∧ x ≤ 2*a + 1}
def B : Set ℝ := {x | x^2 + 2*x - 15 ≤ 0}
def U : Set ℝ := Set.univ

-- Part I: Prove the intersection of complement of A and B when a = 1
theorem part_one : (Set.compl (A 1) ∩ B) = {x : ℝ | -5 ≤ x ∧ x < -2} := by sorry

-- Part II: Prove the condition for A to be a subset of B
theorem part_two : ∀ a : ℝ, A a ⊆ B ↔ (a < -4 ∨ (-2 ≤ a ∧ a ≤ 1)) := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l813_81373


namespace NUMINAMATH_CALUDE_ariels_fish_count_l813_81309

theorem ariels_fish_count (total : ℕ) (male_ratio : ℚ) (female_count : ℕ) 
  (h1 : male_ratio = 2/3)
  (h2 : female_count = 15)
  (h3 : ↑female_count = (1 - male_ratio) * ↑total) : 
  total = 45 := by
  sorry

end NUMINAMATH_CALUDE_ariels_fish_count_l813_81309


namespace NUMINAMATH_CALUDE_units_digit_of_expression_l813_81335

theorem units_digit_of_expression : 
  ((5 * 21 * 1933) + 5^4 - (6 * 2 * 1944)) % 10 = 2 := by sorry

end NUMINAMATH_CALUDE_units_digit_of_expression_l813_81335


namespace NUMINAMATH_CALUDE_same_solution_equations_l813_81385

theorem same_solution_equations (b : ℚ) : 
  (∃ x : ℚ, 3 * x + 9 = 0 ∧ 2 * b * x - 15 = -5) → b = -5/3 := by
  sorry

end NUMINAMATH_CALUDE_same_solution_equations_l813_81385


namespace NUMINAMATH_CALUDE_sum_of_cubes_l813_81382

theorem sum_of_cubes (a b : ℝ) (h1 : a + b = 10) (h2 : a * b = 17) : a^3 + b^3 = 490 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_cubes_l813_81382


namespace NUMINAMATH_CALUDE_planar_edge_pairs_4_2_3_l813_81324

/-- A rectangular prism with edge dimensions a, b, and c. -/
structure RectangularPrism where
  a : ℕ
  b : ℕ
  c : ℕ

/-- The number of unordered pairs of edges that determine a plane in a rectangular prism. -/
def planarEdgePairs (prism : RectangularPrism) : ℕ :=
  sorry

/-- Theorem: The number of unordered pairs of edges that determine a plane
    in a rectangular prism with edge dimensions 4, 2, and 3 is equal to 42. -/
theorem planar_edge_pairs_4_2_3 :
  planarEdgePairs { a := 4, b := 2, c := 3 } = 42 := by
  sorry

end NUMINAMATH_CALUDE_planar_edge_pairs_4_2_3_l813_81324


namespace NUMINAMATH_CALUDE_tax_revenue_decrease_l813_81394

theorem tax_revenue_decrease (T C : ℝ) (T_positive : T > 0) (C_positive : C > 0) :
  let new_tax := 0.8 * T
  let new_consumption := 1.05 * C
  let original_revenue := T * C
  let new_revenue := new_tax * new_consumption
  (original_revenue - new_revenue) / original_revenue = 0.16 := by
  sorry

end NUMINAMATH_CALUDE_tax_revenue_decrease_l813_81394


namespace NUMINAMATH_CALUDE_clock_strikes_count_l813_81306

/-- Calculates the number of clock strikes in a 24-hour period -/
def clock_strikes : ℕ :=
  -- Strikes at whole hours: sum of 1 to 12, twice (for AM and PM)
  2 * (List.range 12).sum
  -- Strikes at half hours: 24 (once every half hour)
  + 24

/-- Theorem stating that the clock strikes 180 times in a 24-hour period -/
theorem clock_strikes_count : clock_strikes = 180 := by
  sorry

end NUMINAMATH_CALUDE_clock_strikes_count_l813_81306


namespace NUMINAMATH_CALUDE_dodecahedron_volume_greater_than_icosahedron_l813_81320

/-- A regular dodecahedron -/
structure Dodecahedron where
  radius : ℝ
  volume : ℝ

/-- A regular icosahedron -/
structure Icosahedron where
  radius : ℝ
  volume : ℝ

/-- The volume of a dodecahedron inscribed in a sphere is greater than 
    the volume of an icosahedron inscribed in the same sphere -/
theorem dodecahedron_volume_greater_than_icosahedron 
  (D : Dodecahedron) (I : Icosahedron) (h : D.radius = I.radius) :
  D.volume > I.volume := by
  sorry

end NUMINAMATH_CALUDE_dodecahedron_volume_greater_than_icosahedron_l813_81320


namespace NUMINAMATH_CALUDE_complex_equation_solution_l813_81314

theorem complex_equation_solution (a b : ℝ) : 
  (Complex.I : ℂ) * (1 + 2 * Complex.I) = (a + b * Complex.I) * (1 + Complex.I) → 
  a = 3/2 ∧ b = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l813_81314


namespace NUMINAMATH_CALUDE_correct_sum_calculation_l813_81350

theorem correct_sum_calculation (tens_digit : Nat) : 
  let original_number := tens_digit * 10 + 9
  let mistaken_number := tens_digit * 10 + 6
  mistaken_number + 57 = 123 →
  original_number + 57 = 126 := by
  sorry

end NUMINAMATH_CALUDE_correct_sum_calculation_l813_81350


namespace NUMINAMATH_CALUDE_machine_work_time_l813_81332

theorem machine_work_time (time_A time_B time_ABC : ℚ) (time_C : ℚ) : 
  time_A = 4 → time_B = 2 → time_ABC = 12/11 → 
  1/time_A + 1/time_B + 1/time_C = 1/time_ABC → 
  time_C = 6 := by sorry

end NUMINAMATH_CALUDE_machine_work_time_l813_81332


namespace NUMINAMATH_CALUDE_first_number_in_sequence_l813_81377

def sequence_product (a : ℕ → ℚ) : Prop :=
  ∀ n : ℕ, n ≥ 3 → n ≤ 10 → a n = a (n-1) * a (n-2)

theorem first_number_in_sequence 
  (a : ℕ → ℚ) 
  (h_seq : sequence_product a) 
  (h_8 : a 8 = 36) 
  (h_9 : a 9 = 324) 
  (h_10 : a 10 = 11664) : 
  a 1 = 59049 / 65536 := by
sorry

end NUMINAMATH_CALUDE_first_number_in_sequence_l813_81377


namespace NUMINAMATH_CALUDE_classroom_position_representation_l813_81348

/-- Represents a position in a classroom -/
structure ClassroomPosition where
  column : ℕ
  row : ℕ

/-- Given that (1, 2) represents the 1st column and 2nd row -/
def given_position : ClassroomPosition := ⟨1, 2⟩

/-- The position we want to prove represents the 2nd column and 3rd row -/
def target_position : ClassroomPosition := ⟨2, 3⟩

/-- Theorem stating that if (1, 2) represents the 1st column and 2nd row,
    then (2, 3) represents the 2nd column and 3rd row -/
theorem classroom_position_representation :
  (given_position.column = 1 ∧ given_position.row = 2) →
  (target_position.column = 2 ∧ target_position.row = 3) :=
by sorry

end NUMINAMATH_CALUDE_classroom_position_representation_l813_81348


namespace NUMINAMATH_CALUDE_ages_solution_l813_81301

/-- Represents the ages of four persons --/
structure Ages where
  a : ℕ  -- oldest
  b : ℕ  -- second oldest
  c : ℕ  -- third oldest
  d : ℕ  -- youngest

/-- The conditions given in the problem --/
def satisfies_conditions (ages : Ages) : Prop :=
  ages.a = ages.d + 16 ∧
  ages.b = ages.d + 8 ∧
  ages.c = ages.d + 4 ∧
  ages.a - 6 = 3 * (ages.d - 6) ∧
  ages.a - 6 = 2 * (ages.b - 6) ∧
  ages.a - 6 = (ages.c - 6) + 4

/-- The theorem to be proved --/
theorem ages_solution :
  ∃ (ages : Ages), satisfies_conditions ages ∧ 
    ages.a = 30 ∧ ages.b = 22 ∧ ages.c = 18 ∧ ages.d = 14 :=
  sorry

end NUMINAMATH_CALUDE_ages_solution_l813_81301


namespace NUMINAMATH_CALUDE_negative_three_times_two_l813_81317

theorem negative_three_times_two : (-3 : ℤ) * 2 = -6 := by
  sorry

end NUMINAMATH_CALUDE_negative_three_times_two_l813_81317


namespace NUMINAMATH_CALUDE_dennis_teaching_years_l813_81315

theorem dennis_teaching_years 
  (V A D E N : ℕ) -- Years taught by Virginia, Adrienne, Dennis, Elijah, and Nadine
  (h1 : V + A + D + E + N = 225) -- Total years taught
  (h2 : (V + A + D + E + N) * 5 = (V + A + D + E + N + 150) * 3) -- Total years is 3/5 of age sum
  (h3 : V = A + 9) -- Virginia vs Adrienne
  (h4 : V = D - 15) -- Virginia vs Dennis
  (h5 : E = A - 3) -- Elijah vs Adrienne
  (h6 : E = 2 * N) -- Elijah vs Nadine
  : D = 101 := by
  sorry

end NUMINAMATH_CALUDE_dennis_teaching_years_l813_81315


namespace NUMINAMATH_CALUDE_candy_sampling_percentage_l813_81355

theorem candy_sampling_percentage (caught_percent : ℝ) (not_caught_ratio : ℝ) 
  (h1 : caught_percent = 22)
  (h2 : not_caught_ratio = 0.2) : 
  (caught_percent / (1 - not_caught_ratio)) = 27.5 := by
  sorry

end NUMINAMATH_CALUDE_candy_sampling_percentage_l813_81355


namespace NUMINAMATH_CALUDE_spherical_coords_reflection_l813_81316

/-- Given a point with rectangular coordinates (x, y, z) and spherical coordinates (ρ, θ, φ),
    prove that the point (x, y, -z) has spherical coordinates (ρ, θ, π - φ) -/
theorem spherical_coords_reflection (x y z ρ θ φ : Real) 
  (h1 : ρ > 0) 
  (h2 : 0 ≤ θ ∧ θ < 2 * Real.pi) 
  (h3 : 0 ≤ φ ∧ φ ≤ Real.pi)
  (h4 : x = ρ * Real.sin φ * Real.cos θ)
  (h5 : y = ρ * Real.sin φ * Real.sin θ)
  (h6 : z = ρ * Real.cos φ)
  (h7 : ρ = 4)
  (h8 : θ = Real.pi / 4)
  (h9 : φ = Real.pi / 6) :
  ∃ (ρ' θ' φ' : Real),
    ρ' = ρ ∧
    θ' = θ ∧
    φ' = Real.pi - φ ∧
    x = ρ' * Real.sin φ' * Real.cos θ' ∧
    y = ρ' * Real.sin φ' * Real.sin θ' ∧
    -z = ρ' * Real.cos φ' ∧
    ρ' > 0 ∧
    0 ≤ θ' ∧ θ' < 2 * Real.pi ∧
    0 ≤ φ' ∧ φ' ≤ Real.pi :=
by sorry

end NUMINAMATH_CALUDE_spherical_coords_reflection_l813_81316


namespace NUMINAMATH_CALUDE_permutation_combination_equality_l813_81313

theorem permutation_combination_equality (n : ℕ) : (n.factorial / (n - 3).factorial = 6 * n.choose 4) → n = 7 := by
  sorry

end NUMINAMATH_CALUDE_permutation_combination_equality_l813_81313


namespace NUMINAMATH_CALUDE_ferry_journey_difference_l813_81327

/-- Represents the properties of a ferry journey -/
structure FerryJourney where
  speed : ℝ
  time : ℝ
  distance : ℝ

/-- The ferry problem setup -/
def ferryProblem : Prop :=
  ∃ (P Q : FerryJourney),
    -- Ferry P properties
    P.speed = 6 ∧
    P.time = 3 ∧
    P.distance = P.speed * P.time ∧
    -- Ferry Q properties
    Q.distance = 2 * P.distance ∧
    Q.speed = P.speed + 3 ∧
    Q.time = Q.distance / Q.speed ∧
    -- The time difference is 1 hour
    Q.time - P.time = 1

/-- Theorem stating the solution to the ferry problem -/
theorem ferry_journey_difference : ferryProblem := by
  sorry

end NUMINAMATH_CALUDE_ferry_journey_difference_l813_81327


namespace NUMINAMATH_CALUDE_ellipse_semi_minor_axis_l813_81369

/-- Given an ellipse with specified center, focus, and semi-major axis endpoint,
    prove that its semi-minor axis has length √8. -/
theorem ellipse_semi_minor_axis 
  (center : ℝ × ℝ)
  (focus : ℝ × ℝ)
  (semi_major_endpoint : ℝ × ℝ)
  (h_center : center = (1, -2))
  (h_focus : focus = (1, -3))
  (h_semi_major : semi_major_endpoint = (1, 1)) :
  let c := Real.sqrt ((center.1 - focus.1)^2 + (center.2 - focus.2)^2)
  let a := Real.sqrt ((center.1 - semi_major_endpoint.1)^2 + (center.2 - semi_major_endpoint.2)^2)
  let b := Real.sqrt (a^2 - c^2)
  b = Real.sqrt 8 := by sorry

end NUMINAMATH_CALUDE_ellipse_semi_minor_axis_l813_81369


namespace NUMINAMATH_CALUDE_min_max_values_l813_81371

theorem min_max_values : 
  (∀ a b : ℝ, a > 0 → b > 0 → a * b = 2 → a + 2 * b ≥ 4) ∧
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a * b = 2 ∧ a + 2 * b = 4) ∧
  (∀ a b : ℝ, a > 0 → b > 0 → a^2 + b^2 = 1 → a + b ≤ Real.sqrt 2) ∧
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a^2 + b^2 = 1 ∧ a + b = Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_min_max_values_l813_81371


namespace NUMINAMATH_CALUDE_convention_center_tables_l813_81357

/-- The number of tables in the convention center. -/
def num_tables : ℕ := 26

/-- The number of chairs around each table. -/
def chairs_per_table : ℕ := 8

/-- The number of legs each chair has. -/
def legs_per_chair : ℕ := 4

/-- The number of legs each table has. -/
def legs_per_table : ℕ := 5

/-- The number of extra chairs not linked with any table. -/
def extra_chairs : ℕ := 10

/-- The total number of legs from tables and chairs. -/
def total_legs : ℕ := 1010

theorem convention_center_tables :
  num_tables * chairs_per_table * legs_per_chair +
  num_tables * legs_per_table +
  extra_chairs * legs_per_chair = total_legs :=
by sorry

end NUMINAMATH_CALUDE_convention_center_tables_l813_81357


namespace NUMINAMATH_CALUDE_smallest_a_value_l813_81345

theorem smallest_a_value (a b : ℝ) (h1 : a ≥ 0) (h2 : b ≥ 0)
  (h3 : ∀ (x : ℤ), Real.sin (a * (x : ℝ) + b) = Real.sin (17 * (x : ℝ))) :
  a ≥ 17 ∧ ∃ (a₀ : ℝ), a₀ ≥ 0 ∧ a₀ < 17 ∧ 
    (∀ (x : ℤ), Real.sin (a₀ * (x : ℝ) + b) = Real.sin (17 * (x : ℝ))) → False :=
by sorry

end NUMINAMATH_CALUDE_smallest_a_value_l813_81345


namespace NUMINAMATH_CALUDE_max_product_sum_2024_l813_81399

theorem max_product_sum_2024 : 
  ∃ (x : ℤ), x * (2024 - x) = 1024144 ∧ 
  ∀ (y : ℤ), y * (2024 - y) ≤ 1024144 := by
  sorry

end NUMINAMATH_CALUDE_max_product_sum_2024_l813_81399


namespace NUMINAMATH_CALUDE_pauls_paint_cans_l813_81322

theorem pauls_paint_cans 
  (initial_rooms : ℕ) 
  (lost_cans : ℕ) 
  (remaining_rooms : ℕ) 
  (h1 : initial_rooms = 50)
  (h2 : lost_cans = 5)
  (h3 : remaining_rooms = 38) :
  (initial_rooms : ℚ) * lost_cans / (initial_rooms - remaining_rooms) = 21 :=
by sorry

end NUMINAMATH_CALUDE_pauls_paint_cans_l813_81322


namespace NUMINAMATH_CALUDE_solve_equation_solve_system_l813_81336

-- Problem 1
theorem solve_equation (x : ℝ) : (x + 1) / 3 - 1 = (x - 1) / 2 → x = -1 := by sorry

-- Problem 2
theorem solve_system (x y : ℝ) : x - y = 1 ∧ 3 * x + y = 7 → x = 2 ∧ y = 1 := by sorry

end NUMINAMATH_CALUDE_solve_equation_solve_system_l813_81336


namespace NUMINAMATH_CALUDE_expected_sixes_is_one_third_l813_81325

-- Define a die as having 6 sides
def die_sides : ℕ := 6

-- Define the probability of rolling a 6 on a single die
def prob_six : ℚ := 1 / die_sides

-- Define the probability of not rolling a 6 on a single die
def prob_not_six : ℚ := 1 - prob_six

-- Define the expected number of 6's when rolling two dice
def expected_sixes : ℚ := 
  2 * (prob_six * prob_six) + 
  1 * (2 * prob_six * prob_not_six) + 
  0 * (prob_not_six * prob_not_six)

-- Theorem statement
theorem expected_sixes_is_one_third : expected_sixes = 1 / 3 :=
sorry

end NUMINAMATH_CALUDE_expected_sixes_is_one_third_l813_81325


namespace NUMINAMATH_CALUDE_equation_solution_l813_81312

theorem equation_solution :
  ∃ x : ℚ, (0.05 * x + 0.12 * (30 + x) = 15.6) ∧ (x = 1200 / 17) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l813_81312


namespace NUMINAMATH_CALUDE_cube_side_ratio_l813_81319

theorem cube_side_ratio (a b : ℝ) (h : a > 0 ∧ b > 0) :
  (6 * a^2) / (6 * b^2) = 4 → a / b = 2 := by
  sorry

end NUMINAMATH_CALUDE_cube_side_ratio_l813_81319


namespace NUMINAMATH_CALUDE_math_contest_schools_count_l813_81329

/-- Represents a participant in the math contest -/
structure Participant where
  score : ℕ
  rank : ℕ

/-- Represents a school team in the math contest -/
structure School where
  team : Fin 4 → Participant

/-- The math contest -/
structure MathContest where
  schools : List School
  andrea : Participant
  beth : Participant
  carla : Participant

/-- The conditions of the math contest -/
def ContestConditions (contest : MathContest) : Prop :=
  ∀ s₁ s₂ : School, ∀ p₁ p₂ : Fin 4, 
    (s₁ ≠ s₂ ∨ p₁ ≠ p₂) → (s₁.team p₁).score ≠ (s₂.team p₂).score
  ∧ contest.andrea.rank < contest.beth.rank
  ∧ contest.beth.rank = 46
  ∧ contest.carla.rank = 79
  ∧ contest.andrea.rank = (contest.schools.length * 4 + 1) / 2
  ∧ ∀ s : School, ∀ p : Fin 4, contest.andrea.score ≥ (s.team p).score

theorem math_contest_schools_count 
  (contest : MathContest) 
  (h : ContestConditions contest) : 
  contest.schools.length = 19 := by
  sorry


end NUMINAMATH_CALUDE_math_contest_schools_count_l813_81329


namespace NUMINAMATH_CALUDE_correct_sampling_methods_l813_81303

/-- Represents a population with possible strata --/
structure Population where
  total : Nat
  strata : List Nat

/-- Represents a sampling method --/
inductive SamplingMethod
  | SimpleRandom
  | Stratified

/-- Represents a sampling problem --/
structure SamplingProblem where
  population : Population
  sampleSize : Nat

/-- Determines the appropriate sampling method for a given problem --/
def appropriateSamplingMethod (problem : SamplingProblem) : SamplingMethod :=
  sorry

theorem correct_sampling_methods
  (collegeProblem : SamplingProblem)
  (workshopProblem : SamplingProblem)
  (h1 : collegeProblem.population = { total := 300, strata := [150, 150] })
  (h2 : collegeProblem.sampleSize = 100)
  (h3 : workshopProblem.population = { total := 100, strata := [] })
  (h4 : workshopProblem.sampleSize = 10) :
  appropriateSamplingMethod collegeProblem = SamplingMethod.Stratified ∧
  appropriateSamplingMethod workshopProblem = SamplingMethod.SimpleRandom :=
sorry

end NUMINAMATH_CALUDE_correct_sampling_methods_l813_81303


namespace NUMINAMATH_CALUDE_min_value_expression_l813_81318

theorem min_value_expression (a b c d e f : ℝ) 
  (h_positive : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ 0 < e ∧ 0 < f)
  (h_upper_bound : a ≤ 3 ∧ b ≤ 3 ∧ c ≤ 3 ∧ d ≤ 3 ∧ e ≤ 3 ∧ f ≤ 3)
  (h_sum1 : a + b + c + d = 6)
  (h_sum2 : e + f = 2) :
  (Real.sqrt (a^2 + 4) + Real.sqrt (b^2 + e^2) + Real.sqrt (c^2 + f^2) + Real.sqrt (d^2 + 4))^2 ≥ 72 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l813_81318


namespace NUMINAMATH_CALUDE_or_necessary_not_sufficient_for_and_l813_81398

theorem or_necessary_not_sufficient_for_and (p q : Prop) :
  (p ∧ q → p ∨ q) ∧ ∃ (p q : Prop), (p ∨ q) ∧ ¬(p ∧ q) :=
by sorry

end NUMINAMATH_CALUDE_or_necessary_not_sufficient_for_and_l813_81398


namespace NUMINAMATH_CALUDE_largest_sum_is_253_33_l813_81334

/-- Represents a trapezium ABCD with specific angle properties -/
structure Trapezium where
  -- Internal angles in arithmetic progression
  b : ℝ
  e : ℝ
  -- Smallest angle is 35°
  smallest_angle : b = 35
  -- Sum of internal angles is 360°
  angle_sum : 4 * b + 6 * e = 360

/-- The largest possible sum of the two largest angles in the trapezium -/
def largest_sum_of_two_largest_angles (t : Trapezium) : ℝ :=
  2 * t.b + 5 * t.e

/-- Theorem stating the largest possible sum of the two largest angles -/
theorem largest_sum_is_253_33 (t : Trapezium) :
  largest_sum_of_two_largest_angles t = 253.33 := by
  sorry


end NUMINAMATH_CALUDE_largest_sum_is_253_33_l813_81334


namespace NUMINAMATH_CALUDE_complete_factorization_w4_minus_81_l813_81360

theorem complete_factorization_w4_minus_81 (w : ℝ) : 
  w^4 - 81 = (w - 3) * (w + 3) * (w^2 + 9) := by
  sorry

end NUMINAMATH_CALUDE_complete_factorization_w4_minus_81_l813_81360


namespace NUMINAMATH_CALUDE_tetrahedron_vertices_prove_tetrahedron_vertices_l813_81384

/-- A tetrahedron is a three-dimensional polyhedron with four triangular faces. -/
structure Tetrahedron where
  -- We don't need to define the internal structure for this problem

/-- The number of vertices in a tetrahedron is 4. -/
theorem tetrahedron_vertices (t : Tetrahedron) : Nat :=
  4

#check tetrahedron_vertices

/-- Prove that a tetrahedron has 4 vertices. -/
theorem prove_tetrahedron_vertices (t : Tetrahedron) : tetrahedron_vertices t = 4 := by
  sorry

end NUMINAMATH_CALUDE_tetrahedron_vertices_prove_tetrahedron_vertices_l813_81384


namespace NUMINAMATH_CALUDE_mans_speed_with_current_l813_81363

/-- 
Given a man's speed against a current and the speed of the current,
this theorem proves the man's speed with the current.
-/
theorem mans_speed_with_current 
  (speed_against_current : ℝ) 
  (current_speed : ℝ) 
  (h1 : speed_against_current = 11.2)
  (h2 : current_speed = 3.4) : 
  speed_against_current + 2 * current_speed = 18 := by
  sorry

#check mans_speed_with_current

end NUMINAMATH_CALUDE_mans_speed_with_current_l813_81363


namespace NUMINAMATH_CALUDE_max_prob_second_highest_l813_81367

variable (p₁ p₂ p₃ : ℝ)

-- Define the conditions
axiom prob_order : 0 < p₁ ∧ p₁ < p₂ ∧ p₂ < p₃ ∧ p₃ ≤ 1

-- Define the probability of winning two consecutive games for each scenario
def P_A := 2 * (p₁ * (p₂ + p₃) - 2 * p₁ * p₂ * p₃)
def P_B := 2 * (p₂ * (p₁ + p₃) - 2 * p₁ * p₂ * p₃)
def P_C := 2 * (p₁ * p₃ + p₂ * p₃ - 2 * p₁ * p₂ * p₃)

-- Theorem statement
theorem max_prob_second_highest :
  P_C p₁ p₂ p₃ > P_A p₁ p₂ p₃ ∧ P_C p₁ p₂ p₃ > P_B p₁ p₂ p₃ :=
sorry

end NUMINAMATH_CALUDE_max_prob_second_highest_l813_81367


namespace NUMINAMATH_CALUDE_third_vertex_x_coord_l813_81330

/-- An equilateral triangle with two vertices at (5, 0) and (5, 8) -/
structure EquilateralTriangle where
  v1 : ℝ × ℝ := (5, 0)
  v2 : ℝ × ℝ := (5, 8)
  v3 : ℝ × ℝ
  equilateral : sorry
  v3_in_first_quadrant : v3.1 > 0 ∧ v3.2 > 0

/-- The x-coordinate of the third vertex is 5 + 4√3 -/
theorem third_vertex_x_coord (t : EquilateralTriangle) : t.v3.1 = 5 + 4 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_third_vertex_x_coord_l813_81330


namespace NUMINAMATH_CALUDE_dog_training_weeks_l813_81372

/-- The number of weeks of training for a seeing-eye dog -/
def training_weeks : ℕ := 12

/-- The adoption fee for an untrained dog in dollars -/
def adoption_fee : ℕ := 150

/-- The cost of training per week in dollars -/
def training_cost_per_week : ℕ := 250

/-- The total certification cost in dollars -/
def certification_cost : ℕ := 3000

/-- The percentage of certification cost covered by insurance -/
def insurance_coverage : ℕ := 90

/-- The total out-of-pocket cost in dollars -/
def total_out_of_pocket : ℕ := 3450

theorem dog_training_weeks :
  adoption_fee +
  training_cost_per_week * training_weeks +
  certification_cost * (100 - insurance_coverage) / 100 =
  total_out_of_pocket :=
by sorry

end NUMINAMATH_CALUDE_dog_training_weeks_l813_81372


namespace NUMINAMATH_CALUDE_sum_of_fractions_geq_one_l813_81356

theorem sum_of_fractions_geq_one (x y z : ℝ) :
  x^2 / (x^2 + 2*y*z) + y^2 / (y^2 + 2*z*x) + z^2 / (z^2 + 2*x*y) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_geq_one_l813_81356


namespace NUMINAMATH_CALUDE_cos_negative_ninety_degrees_l813_81352

theorem cos_negative_ninety_degrees : Real.cos (-(π / 2)) = 0 := by sorry

end NUMINAMATH_CALUDE_cos_negative_ninety_degrees_l813_81352


namespace NUMINAMATH_CALUDE_eleventh_number_with_digit_sum_13_l813_81361

/-- A function that returns the sum of digits of a positive integer -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- A function that returns the nth positive integer whose digits sum to 13 -/
def nthNumberWithDigitSum13 (n : ℕ) : ℕ := sorry

/-- Theorem stating that the 11th number with digit sum 13 is 175 -/
theorem eleventh_number_with_digit_sum_13 : nthNumberWithDigitSum13 11 = 175 := by sorry

end NUMINAMATH_CALUDE_eleventh_number_with_digit_sum_13_l813_81361


namespace NUMINAMATH_CALUDE_starting_to_running_current_ratio_l813_81364

/-- Proves the ratio of starting current to running current for machinery units -/
theorem starting_to_running_current_ratio
  (num_units : ℕ)
  (running_current : ℝ)
  (min_transformer_load : ℝ)
  (h1 : num_units = 3)
  (h2 : running_current = 40)
  (h3 : min_transformer_load = 240)
  : min_transformer_load / (num_units * running_current) = 2 := by
  sorry

#check starting_to_running_current_ratio

end NUMINAMATH_CALUDE_starting_to_running_current_ratio_l813_81364


namespace NUMINAMATH_CALUDE_hot_dog_problem_l813_81366

theorem hot_dog_problem :
  let hot_dogs := 12
  let hot_dog_buns := 9
  let mustard := 18
  let ketchup := 24
  Nat.lcm (Nat.lcm (Nat.lcm hot_dogs hot_dog_buns) mustard) ketchup = 72 := by
  sorry

end NUMINAMATH_CALUDE_hot_dog_problem_l813_81366


namespace NUMINAMATH_CALUDE_missing_donuts_percentage_l813_81390

def initial_donuts : ℕ := 30
def remaining_donuts : ℕ := 9

theorem missing_donuts_percentage :
  (initial_donuts - remaining_donuts) / initial_donuts * 100 = 70 := by
  sorry

end NUMINAMATH_CALUDE_missing_donuts_percentage_l813_81390


namespace NUMINAMATH_CALUDE_joan_remaining_kittens_l813_81381

def initial_kittens : ℕ := 8
def kittens_given_away : ℕ := 2

theorem joan_remaining_kittens :
  initial_kittens - kittens_given_away = 6 := by
  sorry

end NUMINAMATH_CALUDE_joan_remaining_kittens_l813_81381


namespace NUMINAMATH_CALUDE_sphere_division_l813_81323

theorem sphere_division (R : ℝ) : 
  (∃ (n : ℕ), n = 216 ∧ (4 / 3 * Real.pi * R^3 = n * (4 / 3 * Real.pi * 1^3))) ↔ R = 6 :=
sorry

end NUMINAMATH_CALUDE_sphere_division_l813_81323


namespace NUMINAMATH_CALUDE_largest_angle_measure_l813_81307

/-- A triangle PQR is obtuse and isosceles with angle P measuring 30 degrees. -/
structure ObtusePQR where
  /-- Triangle PQR is obtuse -/
  obtuse : Bool
  /-- Triangle PQR is isosceles -/
  isosceles : Bool
  /-- Angle P measures 30 degrees -/
  angle_p : ℝ
  /-- Angle P is 30 degrees -/
  h_angle_p : angle_p = 30

/-- The measure of the largest interior angle in triangle PQR is 120 degrees -/
theorem largest_angle_measure (t : ObtusePQR) : ℝ := by
  sorry

#check largest_angle_measure

end NUMINAMATH_CALUDE_largest_angle_measure_l813_81307


namespace NUMINAMATH_CALUDE_point_A_transformation_l813_81370

-- Define the initial point A
def A : ℝ × ℝ := (3, -2)

-- Define the translation vector
def translation : ℝ × ℝ := (4, 3)

-- Define the rotation center
def rotation_center : ℝ × ℝ := (4, 0)

-- Define the translation function
def translate (p : ℝ × ℝ) (t : ℝ × ℝ) : ℝ × ℝ :=
  (p.1 + t.1, p.2 + t.2)

-- Define the 180-degree rotation function around a given point
def rotate_180 (p : ℝ × ℝ) (center : ℝ × ℝ) : ℝ × ℝ :=
  (2 * center.1 - p.1, 2 * center.2 - p.2)

-- Theorem statement
theorem point_A_transformation :
  rotate_180 (translate A translation) rotation_center = (1, -1) := by
  sorry

end NUMINAMATH_CALUDE_point_A_transformation_l813_81370


namespace NUMINAMATH_CALUDE_external_angle_c_l813_81338

theorem external_angle_c (A B C : ℝ) : 
  A = 40 → B = 2 * A → A + B + C = 180 → 180 - C = 120 := by sorry

end NUMINAMATH_CALUDE_external_angle_c_l813_81338


namespace NUMINAMATH_CALUDE_jongkooks_milk_consumption_l813_81376

/-- Converts liters to milliliters -/
def liters_to_ml (l : ℚ) : ℚ := 1000 * l

/-- Represents the amount of milk drunk in milliliters for each day -/
structure MilkConsumption where
  day1 : ℚ
  day2 : ℚ
  day3 : ℚ

/-- Calculates the total milk consumption in milliliters -/
def total_consumption (mc : MilkConsumption) : ℚ :=
  mc.day1 + mc.day2 + mc.day3

theorem jongkooks_milk_consumption :
  ∃ (mc : MilkConsumption),
    mc.day1 = liters_to_ml 3 + 7 ∧
    mc.day3 = 840 ∧
    total_consumption mc = liters_to_ml 6 + 30 ∧
    mc.day2 = 2183 := by
  sorry

end NUMINAMATH_CALUDE_jongkooks_milk_consumption_l813_81376


namespace NUMINAMATH_CALUDE_whisky_replacement_fraction_l813_81354

/-- Proves the fraction of whisky replaced given initial and final alcohol percentages -/
theorem whisky_replacement_fraction (initial_percent : ℝ) (replacement_percent : ℝ) (final_percent : ℝ) :
  initial_percent = 0.40 →
  replacement_percent = 0.19 →
  final_percent = 0.24 →
  ∃ (fraction : ℝ), fraction = 0.16 / 0.21 ∧
    initial_percent * (1 - fraction) + replacement_percent * fraction = final_percent :=
by sorry

end NUMINAMATH_CALUDE_whisky_replacement_fraction_l813_81354


namespace NUMINAMATH_CALUDE_range_of_f_l813_81397

def f (x : ℝ) : ℝ := |x - 3| - |x + 4|

theorem range_of_f :
  ∀ y ∈ Set.range f, -7 ≤ y ∧ y ≤ 7 ∧
  ∀ z, -7 ≤ z ∧ z ≤ 7 → ∃ x, f x = z :=
sorry

end NUMINAMATH_CALUDE_range_of_f_l813_81397


namespace NUMINAMATH_CALUDE_bracelet_sales_average_l813_81365

theorem bracelet_sales_average (bike_cost : ℕ) (bracelet_price : ℕ) (selling_days : ℕ) 
  (h1 : bike_cost = 112)
  (h2 : bracelet_price = 1)
  (h3 : selling_days = 14) :
  (bike_cost / bracelet_price) / selling_days = 8 := by
  sorry

end NUMINAMATH_CALUDE_bracelet_sales_average_l813_81365


namespace NUMINAMATH_CALUDE_common_solution_y_value_l813_81304

theorem common_solution_y_value (x y : ℝ) : 
  x^2 + y^2 - 4 = 0 ∧ x^2 - y + 2 = 0 → y = 2 :=
by sorry

end NUMINAMATH_CALUDE_common_solution_y_value_l813_81304


namespace NUMINAMATH_CALUDE_science_fiction_total_pages_l813_81302

/-- The number of books in the science fiction section -/
def num_books : ℕ := 8

/-- The number of pages in each book -/
def pages_per_book : ℕ := 478

/-- The total number of pages in the science fiction section -/
def total_pages : ℕ := num_books * pages_per_book

theorem science_fiction_total_pages :
  total_pages = 3824 := by
  sorry

end NUMINAMATH_CALUDE_science_fiction_total_pages_l813_81302


namespace NUMINAMATH_CALUDE_distance_between_points_l813_81389

theorem distance_between_points :
  let p1 : ℝ × ℝ := (5, 5)
  let p2 : ℝ × ℝ := (0, 0)
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2) = 5 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_points_l813_81389


namespace NUMINAMATH_CALUDE_no_three_naturals_sum_power_of_three_l813_81343

theorem no_three_naturals_sum_power_of_three :
  ¬ ∃ (a b c : ℕ), 
    (∃ k : ℕ, a + b = 3^k) ∧
    (∃ m : ℕ, b + c = 3^m) ∧
    (∃ n : ℕ, c + a = 3^n) :=
sorry

end NUMINAMATH_CALUDE_no_three_naturals_sum_power_of_three_l813_81343


namespace NUMINAMATH_CALUDE_tunnel_length_l813_81387

/-- The length of a tunnel given a train passing through it -/
theorem tunnel_length (train_length : ℝ) (exit_time : ℝ) (train_speed : ℝ) : 
  train_length = 2 →
  exit_time = 4 →
  train_speed = 90 →
  (train_speed / 60 * exit_time) - train_length = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_tunnel_length_l813_81387


namespace NUMINAMATH_CALUDE_rotten_tomatoes_solution_l813_81379

/-- Represents the problem of calculating rotten tomatoes --/
def RottenTomatoesProblem (crate_capacity : ℕ) (num_crates : ℕ) (total_cost : ℕ) (selling_price : ℕ) (profit : ℕ) : Prop :=
  let total_capacity := crate_capacity * num_crates
  let revenue := total_cost + profit
  let sold_kg := revenue / selling_price
  total_capacity - sold_kg = 3

/-- Theorem stating the solution to the rotten tomatoes problem --/
theorem rotten_tomatoes_solution :
  RottenTomatoesProblem 20 3 330 6 12 := by
  sorry

#check rotten_tomatoes_solution

end NUMINAMATH_CALUDE_rotten_tomatoes_solution_l813_81379


namespace NUMINAMATH_CALUDE_geometric_series_second_term_l813_81347

/-- For an infinite geometric series with common ratio 1/4 and sum 40, the second term is 7.5 -/
theorem geometric_series_second_term : 
  ∀ (a : ℝ), 
  (∑' n, a * (1/4)^n) = 40 → 
  a * (1/4) = 7.5 := by
sorry

end NUMINAMATH_CALUDE_geometric_series_second_term_l813_81347


namespace NUMINAMATH_CALUDE_quadratic_equation_condition_l813_81388

theorem quadratic_equation_condition (m : ℝ) : (|m| = 2 ∧ m + 2 ≠ 0) ↔ m = 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_condition_l813_81388


namespace NUMINAMATH_CALUDE_weight_of_b_l813_81375

theorem weight_of_b (a b c : ℝ) 
  (h1 : (a + b + c) / 3 = 42)
  (h2 : (a + b) / 2 = 40)
  (h3 : (b + c) / 2 = 43) :
  b = 40 := by
sorry

end NUMINAMATH_CALUDE_weight_of_b_l813_81375


namespace NUMINAMATH_CALUDE_mock_exam_is_systematic_sampling_l813_81341

/-- Represents a sampling method --/
inductive SamplingMethod
  | SimpleRandom
  | Systematic
  | Stratified
  | Cluster

/-- Represents an examination room --/
structure ExamRoom where
  seats : Fin 30 → Nat
  selected_seat : Nat

/-- Represents the mock exam setup --/
structure MockExam where
  rooms : Fin 80 → ExamRoom
  selection_method : SamplingMethod

/-- The mock exam setup as described in the problem --/
def mock_exam : MockExam :=
  { rooms := λ _ => { seats := λ _ => Nat.succ (Nat.zero), selected_seat := 15 },
    selection_method := SamplingMethod.Systematic }

/-- Theorem stating that the sampling method used in the mock exam is systematic sampling --/
theorem mock_exam_is_systematic_sampling :
  mock_exam.selection_method = SamplingMethod.Systematic :=
by sorry

end NUMINAMATH_CALUDE_mock_exam_is_systematic_sampling_l813_81341


namespace NUMINAMATH_CALUDE_remaining_sales_to_goal_l813_81310

def goal : ℕ := 100

def grandmother_sales : ℕ := 5
def uncle_initial_sales : ℕ := 12
def neighbor_initial_sales : ℕ := 8
def mother_friend_sales : ℕ := 25
def cousin_initial_sales : ℕ := 3
def uncle_additional_sales : ℕ := 10
def neighbor_returns : ℕ := 4
def cousin_additional_sales : ℕ := 5

def total_sales : ℕ := 
  grandmother_sales + 
  (uncle_initial_sales + uncle_additional_sales) + 
  (neighbor_initial_sales - neighbor_returns) + 
  mother_friend_sales + 
  (cousin_initial_sales + cousin_additional_sales)

theorem remaining_sales_to_goal : goal - total_sales = 36 := by
  sorry

end NUMINAMATH_CALUDE_remaining_sales_to_goal_l813_81310


namespace NUMINAMATH_CALUDE_convex_lattice_nonagon_centroid_l813_81383

/-- A lattice point in 2D space -/
structure LatticePoint where
  x : Int
  y : Int

/-- A convex nonagon represented by 9 lattice points -/
structure ConvexLatticeNonagon where
  vertices : Fin 9 → LatticePoint
  is_convex : Bool  -- We assume this property without defining it explicitly

/-- The centroid of three points -/
def centroid (p1 p2 p3 : LatticePoint) : (Rat × Rat) :=
  ((p1.x + p2.x + p3.x) / 3, (p1.y + p2.y + p3.y) / 3)

/-- Check if a point with rational coordinates is a lattice point -/
def isLatticePoint (p : Rat × Rat) : Prop :=
  ∃ (x y : Int), p.1 = x ∧ p.2 = y

/-- Main theorem: Any convex lattice nonagon has three vertices whose centroid is a lattice point -/
theorem convex_lattice_nonagon_centroid (n : ConvexLatticeNonagon) :
  ∃ (i j k : Fin 9), i ≠ j ∧ j ≠ k ∧ i ≠ k ∧
    isLatticePoint (centroid (n.vertices i) (n.vertices j) (n.vertices k)) :=
  sorry

end NUMINAMATH_CALUDE_convex_lattice_nonagon_centroid_l813_81383
