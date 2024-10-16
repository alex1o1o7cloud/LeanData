import Mathlib

namespace NUMINAMATH_CALUDE_sufficient_to_necessary_contrapositive_l3529_352999

theorem sufficient_to_necessary_contrapositive (a b : Prop) :
  (a → b) → (¬b → ¬a) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_to_necessary_contrapositive_l3529_352999


namespace NUMINAMATH_CALUDE_luke_money_calculation_l3529_352905

theorem luke_money_calculation (initial_amount spent_amount received_amount : ℕ) : 
  initial_amount = 48 → spent_amount = 11 → received_amount = 21 →
  initial_amount - spent_amount + received_amount = 58 := by
sorry

end NUMINAMATH_CALUDE_luke_money_calculation_l3529_352905


namespace NUMINAMATH_CALUDE_biggest_number_in_ratio_l3529_352922

theorem biggest_number_in_ratio (a b c d : ℕ) : 
  a + b + c + d = 1344 →
  2 * b = 3 * a →
  4 * a = 2 * c →
  5 * a = 2 * d →
  d ≤ 480 ∧ (∃ (x : ℕ), d = 480) :=
by sorry

end NUMINAMATH_CALUDE_biggest_number_in_ratio_l3529_352922


namespace NUMINAMATH_CALUDE_missing_fraction_sum_l3529_352918

theorem missing_fraction_sum (a b c d e f g : ℚ) : 
  a = 1/3 → b = 1/2 → c = -5/6 → d = 1/5 → e = -9/20 → f = -5/6 → g = 23/12 →
  a + b + c + d + e + f + g = 0.8333333333333334 := by
  sorry

end NUMINAMATH_CALUDE_missing_fraction_sum_l3529_352918


namespace NUMINAMATH_CALUDE_dan_youngest_l3529_352993

def ages (a b c d e : ℕ) : Prop :=
  a + b = 39 ∧
  b + c = 40 ∧
  c + d = 38 ∧
  d + e = 44 ∧
  a + b + c + d + e = 105

theorem dan_youngest (a b c d e : ℕ) (h : ages a b c d e) : 
  d < a ∧ d < b ∧ d < c ∧ d < e := by
  sorry

end NUMINAMATH_CALUDE_dan_youngest_l3529_352993


namespace NUMINAMATH_CALUDE_triangle_area_l3529_352926

/-- The area of a triangle with vertices at (0,0), (8,8), and (-8,8) is 64 -/
theorem triangle_area : 
  let O : ℝ × ℝ := (0, 0)
  let A : ℝ × ℝ := (8, 8)
  let B : ℝ × ℝ := (-8, 8)
  let base := |A.1 - B.1|
  let height := A.2
  (1 / 2) * base * height = 64 := by sorry

end NUMINAMATH_CALUDE_triangle_area_l3529_352926


namespace NUMINAMATH_CALUDE_possible_distances_l3529_352989

theorem possible_distances (p q r s t : ℝ) 
  (h1 : |p - q| = 3)
  (h2 : |q - r| = 4)
  (h3 : |r - s| = 5)
  (h4 : |s - t| = 6) :
  ∃ (S : Set ℝ), S = {0, 2, 4, 6, 8, 10, 12, 18} ∧ |p - t| ∈ S :=
by sorry

end NUMINAMATH_CALUDE_possible_distances_l3529_352989


namespace NUMINAMATH_CALUDE_x_axis_ellipse_iff_condition_l3529_352915

/-- An ellipse with foci on the x-axis -/
structure XAxisEllipse where
  k : ℝ
  eq : ∀ (x y : ℝ), x^2 / 2 + y^2 / k = 1

/-- The condition for an ellipse with foci on the x-axis -/
def is_x_axis_ellipse_condition (k : ℝ) : Prop :=
  0 < k ∧ k < 2

/-- The theorem stating that 0 < k < 2 is a necessary and sufficient condition 
    for the equation x^2/2 + y^2/k = 1 to represent an ellipse with foci on the x-axis -/
theorem x_axis_ellipse_iff_condition (e : XAxisEllipse) :
  is_x_axis_ellipse_condition e.k ↔ True :=
sorry

end NUMINAMATH_CALUDE_x_axis_ellipse_iff_condition_l3529_352915


namespace NUMINAMATH_CALUDE_quadratic_solution_set_l3529_352987

def quadratic_function (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + x + b

theorem quadratic_solution_set (a b : ℝ) :
  (∀ x : ℝ, x ∈ Set.Ioo 1 2 ↔ quadratic_function a b x > 0) →
  a + b = -1 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_solution_set_l3529_352987


namespace NUMINAMATH_CALUDE_largest_two_digit_divisible_by_six_ending_in_four_l3529_352990

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def ends_in_four (n : ℕ) : Prop := n % 10 = 4

theorem largest_two_digit_divisible_by_six_ending_in_four :
  ∀ n : ℕ, is_two_digit n → n % 6 = 0 → ends_in_four n → n ≤ 84 :=
sorry

end NUMINAMATH_CALUDE_largest_two_digit_divisible_by_six_ending_in_four_l3529_352990


namespace NUMINAMATH_CALUDE_rectangle_square_ratio_l3529_352946

theorem rectangle_square_ratio (s a b : ℝ) (h1 : a * b = 2 * s ^ 2) (h2 : a = 2 * b) :
  a / s = 2 := by sorry

end NUMINAMATH_CALUDE_rectangle_square_ratio_l3529_352946


namespace NUMINAMATH_CALUDE_profit_percentage_is_30_percent_l3529_352947

/-- Calculate the percentage of profit given the cost price and selling price --/
def percentage_profit (cost_price selling_price : ℚ) : ℚ :=
  ((selling_price - cost_price) / cost_price) * 100

/-- Theorem stating that the percentage profit is 30% for the given prices --/
theorem profit_percentage_is_30_percent :
  percentage_profit 350 455 = 30 := by
  sorry

end NUMINAMATH_CALUDE_profit_percentage_is_30_percent_l3529_352947


namespace NUMINAMATH_CALUDE_smallest_c_for_all_real_domain_l3529_352945

theorem smallest_c_for_all_real_domain : ∃ c : ℤ, 
  (∀ x : ℝ, (x^2 + c*x + 15 ≠ 0)) ∧ 
  (∀ k : ℤ, k < c → ∃ x : ℝ, x^2 + k*x + 15 = 0) ∧
  c = -7 :=
sorry

end NUMINAMATH_CALUDE_smallest_c_for_all_real_domain_l3529_352945


namespace NUMINAMATH_CALUDE_series_sum_l3529_352950

/-- The positive real solution to x³ + (1/4)x - 1 = 0 -/
noncomputable def s : ℝ := sorry

/-- The infinite series s³ + 2s⁷ + 3s¹¹ + 4s¹⁵ + ... -/
noncomputable def T : ℝ := sorry

/-- s is a solution to the equation x³ + (1/4)x - 1 = 0 -/
axiom s_def : s^3 + (1/4) * s - 1 = 0

/-- s is positive -/
axiom s_pos : s > 0

/-- T is equal to the infinite series s³ + 2s⁷ + 3s¹¹ + 4s¹⁵ + ... -/
axiom T_def : T = s^3 + 2*s^7 + 3*s^11 + 4*s^15 + sorry

theorem series_sum : T = 16 * s := by sorry

end NUMINAMATH_CALUDE_series_sum_l3529_352950


namespace NUMINAMATH_CALUDE_S_3_5_equals_42_l3529_352954

-- Define the operation S
def S (a b : ℕ) : ℕ := 4 * a + 6 * b

-- Theorem to prove
theorem S_3_5_equals_42 : S 3 5 = 42 := by
  sorry

end NUMINAMATH_CALUDE_S_3_5_equals_42_l3529_352954


namespace NUMINAMATH_CALUDE_projection_incircle_inequality_l3529_352966

/-- Represents a right triangle with legs a and b, hypotenuse c, projections p and q, and incircle radii ρ_a and ρ_b -/
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  p : ℝ
  q : ℝ
  ρ_a : ℝ
  ρ_b : ℝ
  h_right : a^2 + b^2 = c^2
  h_a_lt_b : a < b
  h_p_proj : p * c = a^2
  h_q_proj : q * c = b^2
  h_ρ_a_def : ρ_a * (a + c - b) = a * b
  h_ρ_b_def : ρ_b * (b + c - a) = a * b

/-- Theorem stating the inequalities for projections and incircle radii in a right triangle -/
theorem projection_incircle_inequality (t : RightTriangle) : t.p < t.ρ_a ∧ t.q > t.ρ_b := by
  sorry

end NUMINAMATH_CALUDE_projection_incircle_inequality_l3529_352966


namespace NUMINAMATH_CALUDE_factor_implies_p_value_l3529_352977

theorem factor_implies_p_value (m p : ℤ) : 
  (m - 8) ∣ (m^2 - p*m - 24) → p = 5 := by
sorry

end NUMINAMATH_CALUDE_factor_implies_p_value_l3529_352977


namespace NUMINAMATH_CALUDE_inequality_proof_l3529_352967

theorem inequality_proof (a b : ℝ) (h1 : a > b) (h2 : b > 0) : a^4 + b^4 > 2*a*b^3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3529_352967


namespace NUMINAMATH_CALUDE_arccos_negative_one_equals_pi_l3529_352909

theorem arccos_negative_one_equals_pi : Real.arccos (-1) = π := by
  sorry

end NUMINAMATH_CALUDE_arccos_negative_one_equals_pi_l3529_352909


namespace NUMINAMATH_CALUDE_balls_in_boxes_l3529_352937

theorem balls_in_boxes (n : ℕ) (k : ℕ) : n = 5 ∧ k = 4 → k^n = 1024 := by
  sorry

end NUMINAMATH_CALUDE_balls_in_boxes_l3529_352937


namespace NUMINAMATH_CALUDE_octagon_pebble_arrangements_l3529_352928

/-- The number of symmetries (rotations and reflections) of a regular octagon -/
def octagon_symmetries : ℕ := 16

/-- The number of vertices in a regular octagon -/
def octagon_vertices : ℕ := 8

/-- The number of distinct arrangements of pebbles on a regular octagon -/
def distinct_arrangements : ℕ := Nat.factorial octagon_vertices / octagon_symmetries

theorem octagon_pebble_arrangements :
  distinct_arrangements = 2520 := by sorry

end NUMINAMATH_CALUDE_octagon_pebble_arrangements_l3529_352928


namespace NUMINAMATH_CALUDE_equation_solution_l3529_352924

theorem equation_solution (k : ℝ) : 
  ((-2 : ℝ)^2 + 4*k*(-2) + 2*k^2 = 4) → (k = 0 ∨ k = 4) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3529_352924


namespace NUMINAMATH_CALUDE_sqrt_4_equals_plus_minus_2_cube_root_negative_8_over_27_equals_negative_2_over_3_sqrt_diff_equals_point_1_abs_sqrt_2_minus_1_equals_sqrt_2_minus_1_l3529_352912

-- 1. Prove that ±√4 = ±2
theorem sqrt_4_equals_plus_minus_2 : ∀ x : ℝ, x^2 = 4 ↔ x = 2 ∨ x = -2 := by sorry

-- 2. Prove that ∛(-8/27) = -2/3
theorem cube_root_negative_8_over_27_equals_negative_2_over_3 : 
  ((-8/27 : ℝ) ^ (1/3 : ℝ)) = -2/3 := by sorry

-- 3. Prove that √0.09 - √0.04 = 0.1
theorem sqrt_diff_equals_point_1 : 
  Real.sqrt 0.09 - Real.sqrt 0.04 = 0.1 := by sorry

-- 4. Prove that |√2 - 1| = √2 - 1
theorem abs_sqrt_2_minus_1_equals_sqrt_2_minus_1 : 
  |Real.sqrt 2 - 1| = Real.sqrt 2 - 1 := by sorry

end NUMINAMATH_CALUDE_sqrt_4_equals_plus_minus_2_cube_root_negative_8_over_27_equals_negative_2_over_3_sqrt_diff_equals_point_1_abs_sqrt_2_minus_1_equals_sqrt_2_minus_1_l3529_352912


namespace NUMINAMATH_CALUDE_triangle_area_l3529_352959

theorem triangle_area (a b c : ℝ) (h1 : a = 4) (h2 : b = 4) (h3 : c = 6) :
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c)) = 3 * Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l3529_352959


namespace NUMINAMATH_CALUDE_tank_water_level_l3529_352979

theorem tank_water_level (tank_capacity : ℚ) (initial_fraction : ℚ) (added_water : ℚ) :
  tank_capacity = 72 →
  initial_fraction = 3 / 4 →
  added_water = 9 →
  (initial_fraction * tank_capacity + added_water) / tank_capacity = 7 / 8 := by
  sorry

end NUMINAMATH_CALUDE_tank_water_level_l3529_352979


namespace NUMINAMATH_CALUDE_x_plus_y_equals_9_l3529_352980

theorem x_plus_y_equals_9 (x y m : ℝ) (h1 : x + m = 4) (h2 : y - 5 = m) : x + y = 9 := by
  sorry

end NUMINAMATH_CALUDE_x_plus_y_equals_9_l3529_352980


namespace NUMINAMATH_CALUDE_complex_pure_imaginary_l3529_352910

theorem complex_pure_imaginary (m : ℝ) : 
  (m + (10 : ℂ) / (3 + Complex.I)).im ≠ 0 ∧ (m + (10 : ℂ) / (3 + Complex.I)).re = 0 → m = -3 := by
  sorry

end NUMINAMATH_CALUDE_complex_pure_imaginary_l3529_352910


namespace NUMINAMATH_CALUDE_shaded_area_square_with_circles_l3529_352944

/-- The area of the shaded region in a square with circles at its vertices -/
theorem shaded_area_square_with_circles (s : ℝ) (r : ℝ) (h1 : s = 12) (h2 : r = 2) :
  s^2 - (4 * (π / 2 * r^2) + 4 * (r^2 / 2)) = 136 - 2 * π := by sorry

end NUMINAMATH_CALUDE_shaded_area_square_with_circles_l3529_352944


namespace NUMINAMATH_CALUDE_olly_minimum_cost_l3529_352969

/-- Represents the number of each type of pet Olly has -/
structure Pets where
  dogs : Nat
  cats : Nat
  ferrets : Nat

/-- Represents the pricing and discount structure for Pack A -/
structure PackA where
  small_shoe_price : ℝ
  medium_shoe_price : ℝ
  small_shoe_discount : ℝ
  medium_shoe_discount : ℝ

/-- Represents the pricing and discount structure for Pack B -/
structure PackB where
  small_shoe_price : ℝ
  medium_shoe_price : ℝ
  small_shoe_free_ratio : Nat
  medium_shoe_free_ratio : Nat

/-- Calculates the minimum cost for Olly to purchase shoes for all his pets -/
def minimum_cost (pets : Pets) (pack_a : PackA) (pack_b : PackB) : ℝ := by
  sorry

/-- Theorem stating that the minimum cost for Olly to purchase shoes for all his pets is $64 -/
theorem olly_minimum_cost :
  let pets := Pets.mk 3 2 1
  let pack_a := PackA.mk 12 16 0.2 0.15
  let pack_b := PackB.mk 7 9 3 4
  minimum_cost pets pack_a pack_b = 64 := by
  sorry

end NUMINAMATH_CALUDE_olly_minimum_cost_l3529_352969


namespace NUMINAMATH_CALUDE_product_of_three_integers_summing_to_seven_l3529_352931

theorem product_of_three_integers_summing_to_seven (a b c : ℕ) :
  a > 0 → b > 0 → c > 0 →
  a ≠ b → b ≠ c → a ≠ c →
  a + b + c = 7 →
  a * b * c = 8 := by
sorry

end NUMINAMATH_CALUDE_product_of_three_integers_summing_to_seven_l3529_352931


namespace NUMINAMATH_CALUDE_max_value_theorem_l3529_352963

theorem max_value_theorem (x y z : ℝ) (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : 0 ≤ z) 
  (h4 : x^2 + y^2 + z^2 = 1) : 
  3 * x * y * Real.sqrt 7 + 9 * y * z ≤ (1/2) * Real.sqrt 88 := by
sorry

end NUMINAMATH_CALUDE_max_value_theorem_l3529_352963


namespace NUMINAMATH_CALUDE_two_mutually_exclusive_pairs_l3529_352994

/-- Represents the color of a ball -/
inductive Color
| Red
| White

/-- Represents the outcome of drawing two balls -/
structure Outcome :=
  (first second : Color)

/-- The sample space of all possible outcomes when drawing two balls -/
def sampleSpace : Set Outcome :=
  {o | o.first ∈ [Color.Red, Color.White] ∧ o.second ∈ [Color.Red, Color.White]}

/-- Event: At least one white ball -/
def atLeastOneWhite (o : Outcome) : Prop :=
  o.first = Color.White ∨ o.second = Color.White

/-- Event: Both are white balls -/
def bothWhite (o : Outcome) : Prop :=
  o.first = Color.White ∧ o.second = Color.White

/-- Event: At least one red ball -/
def atLeastOneRed (o : Outcome) : Prop :=
  o.first = Color.Red ∨ o.second = Color.Red

/-- Event: Exactly one white ball -/
def exactlyOneWhite (o : Outcome) : Prop :=
  (o.first = Color.White ∧ o.second = Color.Red) ∨
  (o.first = Color.Red ∧ o.second = Color.White)

/-- Event: Exactly two white balls -/
def exactlyTwoWhite (o : Outcome) : Prop :=
  o.first = Color.White ∧ o.second = Color.White

/-- Event: Both are red balls -/
def bothRed (o : Outcome) : Prop :=
  o.first = Color.Red ∧ o.second = Color.Red

/-- Two events are mutually exclusive if they cannot occur simultaneously -/
def mutuallyExclusive (e1 e2 : Outcome → Prop) : Prop :=
  ∀ o, ¬(e1 o ∧ e2 o)

/-- The main theorem: exactly 2 pairs of events are mutually exclusive -/
theorem two_mutually_exclusive_pairs :
  (mutuallyExclusive exactlyOneWhite exactlyTwoWhite) ∧
  (mutuallyExclusive atLeastOneWhite bothRed) ∧
  (¬mutuallyExclusive atLeastOneWhite bothWhite) ∧
  (¬mutuallyExclusive atLeastOneWhite atLeastOneRed) :=
sorry

end NUMINAMATH_CALUDE_two_mutually_exclusive_pairs_l3529_352994


namespace NUMINAMATH_CALUDE_total_cards_packed_l3529_352940

/-- The number of cards in a standard playing card deck -/
def playing_cards_per_deck : ℕ := 52

/-- The number of cards in a Pinochle deck -/
def pinochle_cards_per_deck : ℕ := 48

/-- The number of cards in a Tarot deck -/
def tarot_cards_per_deck : ℕ := 78

/-- The number of cards in an Uno deck -/
def uno_cards_per_deck : ℕ := 108

/-- The number of playing card decks Elijah packed -/
def playing_card_decks : ℕ := 6

/-- The number of Pinochle decks Elijah packed -/
def pinochle_decks : ℕ := 4

/-- The number of Tarot decks Elijah packed -/
def tarot_decks : ℕ := 2

/-- The number of Uno decks Elijah packed -/
def uno_decks : ℕ := 3

/-- Theorem stating the total number of cards Elijah packed -/
theorem total_cards_packed : 
  playing_card_decks * playing_cards_per_deck + 
  pinochle_decks * pinochle_cards_per_deck + 
  tarot_decks * tarot_cards_per_deck + 
  uno_decks * uno_cards_per_deck = 984 := by
  sorry

end NUMINAMATH_CALUDE_total_cards_packed_l3529_352940


namespace NUMINAMATH_CALUDE_ceiling_times_self_156_l3529_352938

theorem ceiling_times_self_156 :
  ∃! x : ℝ, ⌈x⌉ * x = 156 ∧ x = 12 := by sorry

end NUMINAMATH_CALUDE_ceiling_times_self_156_l3529_352938


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_l3529_352985

universe u

def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 2}
def B : Set ℕ := {2, 3}

theorem complement_A_intersect_B :
  (Set.compl A ∩ B) = {3} := by sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_l3529_352985


namespace NUMINAMATH_CALUDE_tan_alpha_values_l3529_352923

theorem tan_alpha_values (α : Real) (h : 2 * Real.sin (2 * α) = 1 - Real.cos (2 * α)) :
  Real.tan α = 2 ∨ Real.tan α = 0 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_values_l3529_352923


namespace NUMINAMATH_CALUDE_initial_water_percentage_l3529_352998

/-- Proves that the initial water percentage in a mixture is 60% given the specified conditions --/
theorem initial_water_percentage (initial_volume : ℝ) (added_water : ℝ) (final_water_percentage : ℝ) :
  initial_volume = 300 →
  added_water = 100 →
  final_water_percentage = 70 →
  (initial_volume * x / 100 + added_water) / (initial_volume + added_water) * 100 = final_water_percentage →
  x = 60 := by
  sorry

#check initial_water_percentage

end NUMINAMATH_CALUDE_initial_water_percentage_l3529_352998


namespace NUMINAMATH_CALUDE_sock_pairs_count_l3529_352902

def white_socks : ℕ := 5
def brown_socks : ℕ := 4
def blue_socks : ℕ := 3

def different_color_pairs_with_blue : ℕ := (blue_socks * white_socks) + (blue_socks * brown_socks)

theorem sock_pairs_count : different_color_pairs_with_blue = 27 := by
  sorry

end NUMINAMATH_CALUDE_sock_pairs_count_l3529_352902


namespace NUMINAMATH_CALUDE_bakery_earnings_for_five_days_l3529_352929

/-- Represents the daily production and prices of baked goods in Uki's bakery --/
structure BakeryData where
  cupcake_price : ℝ
  cookie_price : ℝ
  biscuit_price : ℝ
  cupcakes_per_day : ℕ
  cookie_packets_per_day : ℕ
  biscuit_packets_per_day : ℕ

/-- Calculates the total earnings for a given number of days --/
def total_earnings (data : BakeryData) (days : ℕ) : ℝ :=
  let daily_earnings := 
    data.cupcake_price * data.cupcakes_per_day +
    data.cookie_price * data.cookie_packets_per_day +
    data.biscuit_price * data.biscuit_packets_per_day
  daily_earnings * days

/-- Theorem stating that the total earnings for 5 days is $350 --/
theorem bakery_earnings_for_five_days :
  let data := BakeryData.mk 1.5 2 1 20 10 20
  total_earnings data 5 = 350 := by
  sorry

end NUMINAMATH_CALUDE_bakery_earnings_for_five_days_l3529_352929


namespace NUMINAMATH_CALUDE_sequence_property_l3529_352920

theorem sequence_property (m : ℤ) (a : ℕ → ℤ) (r s : ℕ) : 
  (|m| ≥ 2) →
  (∃ k, a k ≠ 0) →
  (∀ n : ℕ, a (n + 2) = a (n + 1) - m * a n) →
  (r > s) →
  (s ≥ 2) →
  (a r = a s) →
  (a r = a 1) →
  (r - s ≥ |m|) := by
sorry

end NUMINAMATH_CALUDE_sequence_property_l3529_352920


namespace NUMINAMATH_CALUDE_subset_implies_m_equals_three_l3529_352903

theorem subset_implies_m_equals_three (A B : Set ℕ) (m : ℕ) :
  A = {1, 3} →
  B = {1, 2, m} →
  A ⊆ B →
  m = 3 := by
sorry

end NUMINAMATH_CALUDE_subset_implies_m_equals_three_l3529_352903


namespace NUMINAMATH_CALUDE_problem_solution_l3529_352906

theorem problem_solution (X : ℝ) : 
  (213 * 16 = 3408) → 
  ((213 * 16) + (1.6 * 2.13) = X) → 
  (X - (5/2) * 1.25 = 3408.283) := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l3529_352906


namespace NUMINAMATH_CALUDE_olivia_earnings_l3529_352901

/-- Calculates the earnings for a tutor based on their hours worked and payment conditions. -/
def calculate_earnings (tuesday_hours : ℚ) (wednesday_minutes : ℕ) (thursday_start_hour : ℕ) (thursday_start_minute : ℕ) (thursday_end_hour : ℕ) (thursday_end_minute : ℕ) (saturday_minutes : ℕ) (hourly_rate : ℚ) (bonus_threshold : ℚ) (bonus_rate : ℚ) : ℚ :=
  sorry

/-- Proves that Olivia's earnings for the week are $28.17 given her tutoring schedule and payment conditions. -/
theorem olivia_earnings : 
  let tuesday_hours : ℚ := 3/2
  let wednesday_minutes : ℕ := 40
  let thursday_start_hour : ℕ := 9
  let thursday_start_minute : ℕ := 15
  let thursday_end_hour : ℕ := 11
  let thursday_end_minute : ℕ := 30
  let saturday_minutes : ℕ := 45
  let hourly_rate : ℚ := 5
  let bonus_threshold : ℚ := 4
  let bonus_rate : ℚ := 2
  calculate_earnings tuesday_hours wednesday_minutes thursday_start_hour thursday_start_minute thursday_end_hour thursday_end_minute saturday_minutes hourly_rate bonus_threshold bonus_rate = 28.17 := by
  sorry

end NUMINAMATH_CALUDE_olivia_earnings_l3529_352901


namespace NUMINAMATH_CALUDE_maggie_yellow_packs_l3529_352960

/-- The number of packs of yellow bouncy balls Maggie bought -/
def yellow_packs (red_packs green_packs : ℕ) (balls_per_pack : ℕ) (total_balls : ℕ) : ℕ :=
  (total_balls - (red_packs + green_packs) * balls_per_pack) / balls_per_pack

/-- Theorem stating that Maggie bought 8 packs of yellow bouncy balls -/
theorem maggie_yellow_packs : yellow_packs 4 4 10 160 = 8 := by
  sorry

end NUMINAMATH_CALUDE_maggie_yellow_packs_l3529_352960


namespace NUMINAMATH_CALUDE_ice_cream_combinations_l3529_352949

/-- The number of ways to distribute n indistinguishable items among k distinguishable categories -/
def distribute (n k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- The number of ways to distribute 5 indistinguishable items among 3 distinguishable categories is 21 -/
theorem ice_cream_combinations : distribute 5 3 = 21 := by sorry

end NUMINAMATH_CALUDE_ice_cream_combinations_l3529_352949


namespace NUMINAMATH_CALUDE_graph_symmetry_l3529_352972

-- Define a general real-valued function
variable (f : ℝ → ℝ)

-- Define the symmetry property about the y-axis
def symmetric_about_y_axis (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (-x)

-- Theorem statement
theorem graph_symmetry (f : ℝ → ℝ) : 
  symmetric_about_y_axis f ↔ 
  ∀ x y : ℝ, (x, y) ∈ (Set.range (λ x => (x, f x))) ↔ 
              (-x, y) ∈ (Set.range (λ x => (x, f x))) :=
sorry

end NUMINAMATH_CALUDE_graph_symmetry_l3529_352972


namespace NUMINAMATH_CALUDE_friends_fireworks_count_l3529_352936

/-- The number of fireworks Henry bought -/
def henrys_fireworks : ℕ := 2

/-- The number of fireworks saved from last year -/
def saved_fireworks : ℕ := 6

/-- The total number of fireworks they have now -/
def total_fireworks : ℕ := 11

/-- The number of fireworks Henry's friend bought -/
def friends_fireworks : ℕ := total_fireworks - (henrys_fireworks + saved_fireworks)

theorem friends_fireworks_count : friends_fireworks = 3 := by
  sorry

end NUMINAMATH_CALUDE_friends_fireworks_count_l3529_352936


namespace NUMINAMATH_CALUDE_mango_rate_per_kg_l3529_352919

/-- The rate per kg of mangoes given the purchase details -/
theorem mango_rate_per_kg
  (grape_kg : ℕ)
  (grape_rate : ℕ)
  (mango_kg : ℕ)
  (total_paid : ℕ)
  (h1 : grape_kg = 8)
  (h2 : grape_rate = 70)
  (h3 : mango_kg = 9)
  (h4 : total_paid = 1055)
  : (total_paid - grape_kg * grape_rate) / mango_kg = 55 := by
  sorry

end NUMINAMATH_CALUDE_mango_rate_per_kg_l3529_352919


namespace NUMINAMATH_CALUDE_distinguishable_triangles_l3529_352956

def num_colors : ℕ := 8

def corner_configurations : ℕ := 
  num_colors + num_colors * (num_colors - 1) + (num_colors.choose 3)

def center_configurations : ℕ := num_colors * (num_colors - 1)

theorem distinguishable_triangles : 
  corner_configurations * center_configurations = 6720 := by sorry

end NUMINAMATH_CALUDE_distinguishable_triangles_l3529_352956


namespace NUMINAMATH_CALUDE_square_in_M_l3529_352996

/-- The set of functions f: ℝ → ℝ with the property that there exist real numbers a and k (k ≠ 0)
    such that f(a+x) = kf(a-x) for all x ∈ ℝ -/
def M : Set (ℝ → ℝ) :=
  {f | ∃ (a k : ℝ), k ≠ 0 ∧ ∀ x, f (a + x) = k * f (a - x)}

/-- The square function -/
def square : ℝ → ℝ := fun x ↦ x^2

/-- Theorem: The square function belongs to set M -/
theorem square_in_M : square ∈ M := by sorry

end NUMINAMATH_CALUDE_square_in_M_l3529_352996


namespace NUMINAMATH_CALUDE_gcd_10010_15015_l3529_352914

theorem gcd_10010_15015 : Nat.gcd 10010 15015 = 5005 := by
  sorry

end NUMINAMATH_CALUDE_gcd_10010_15015_l3529_352914


namespace NUMINAMATH_CALUDE_point_on_y_axis_l3529_352935

/-- A point M with coordinates (a+2, 2a-5) lies on the y-axis. -/
theorem point_on_y_axis (a : ℝ) : (a + 2 = 0) → a = -2 := by
  sorry

end NUMINAMATH_CALUDE_point_on_y_axis_l3529_352935


namespace NUMINAMATH_CALUDE_square_properties_l3529_352911

/-- A square in a 2D plane -/
structure Square where
  /-- The line representing the center's x-coordinate -/
  center_line1 : ℝ → ℝ → Prop
  /-- The line representing the center's y-coordinate -/
  center_line2 : ℝ → ℝ → Prop
  /-- The equation of one side of the square -/
  side1 : ℝ → ℝ → Prop
  /-- The equation of the second side of the square -/
  side2 : ℝ → ℝ → Prop
  /-- The equation of the third side of the square -/
  side3 : ℝ → ℝ → Prop
  /-- The equation of the fourth side of the square -/
  side4 : ℝ → ℝ → Prop

/-- Theorem stating the properties of the square -/
theorem square_properties (s : Square) :
  s.center_line1 = fun x y => x - y + 1 = 0 ∧
  s.center_line2 = fun x y => 2*x + y + 2 = 0 ∧
  s.side1 = fun x y => x + 3*y - 2 = 0 →
  s.side2 = fun x y => x + 3*y + 4 = 0 ∧
  s.side3 = fun x y => 3*x - y = 0 ∧
  s.side4 = fun x y => 3*x - y + 6 = 0 :=
by sorry


end NUMINAMATH_CALUDE_square_properties_l3529_352911


namespace NUMINAMATH_CALUDE_wednesday_bags_raked_l3529_352927

theorem wednesday_bags_raked (charge_per_bag : ℕ) (monday_bags : ℕ) (tuesday_bags : ℕ) (total_money : ℕ) :
  charge_per_bag = 4 →
  monday_bags = 5 →
  tuesday_bags = 3 →
  total_money = 68 →
  ∃ wednesday_bags : ℕ, wednesday_bags = 9 ∧ 
    total_money = charge_per_bag * (monday_bags + tuesday_bags + wednesday_bags) :=
by sorry

end NUMINAMATH_CALUDE_wednesday_bags_raked_l3529_352927


namespace NUMINAMATH_CALUDE_partial_fraction_sum_zero_l3529_352992

theorem partial_fraction_sum_zero (x : ℝ) (A B C D E : ℝ) : 
  (1 : ℝ) / (x * (x + 1) * (x + 2) * (x + 3) * (x + 5)) = 
    A / x + B / (x + 1) + C / (x + 2) + D / (x + 3) + E / (x + 5) →
  A + B + C + D + E = 0 := by
sorry

end NUMINAMATH_CALUDE_partial_fraction_sum_zero_l3529_352992


namespace NUMINAMATH_CALUDE_standard_deviation_of_dataset_l3529_352930

def dataset : List ℝ := [3, 4, 5, 5, 6, 7]

theorem standard_deviation_of_dataset :
  let n : ℕ := dataset.length
  let mean : ℝ := (dataset.sum) / n
  let variance : ℝ := (dataset.map (fun x => (x - mean)^2)).sum / n
  Real.sqrt variance = Real.sqrt (5/3) := by sorry

end NUMINAMATH_CALUDE_standard_deviation_of_dataset_l3529_352930


namespace NUMINAMATH_CALUDE_third_median_length_l3529_352983

/-- A triangle with two known medians and area -/
structure TriangleWithMedians where
  -- The length of the first median
  median1 : ℝ
  -- The length of the second median
  median2 : ℝ
  -- The area of the triangle
  area : ℝ

/-- The theorem stating the length of the third median -/
theorem third_median_length (t : TriangleWithMedians) 
  (h1 : t.median1 = 5)
  (h2 : t.median2 = 4)
  (h3 : t.area = 6 * Real.sqrt 5) :
  ∃ (median3 : ℝ), median3 = 3 * Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_third_median_length_l3529_352983


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l3529_352965

/-- Given a hyperbola and related geometric conditions, prove its asymptotes. -/
theorem hyperbola_asymptotes (a b c : ℝ) (E F₁ F₂ D : ℝ × ℝ) :
  a > 0 ∧ b > 0 ∧
  (λ (x y : ℝ) => y^2 / a^2 - x^2 / b^2 = 1) E.1 E.2 ∧  -- E is on the hyperbola
  (λ (x y : ℝ) => 4*x^2 + 4*y^2 = b^2) D.1 D.2 ∧       -- D is on the circle
  (E.1 - F₁.1) * D.1 + (E.2 - F₁.2) * D.2 = 0 ∧       -- EF₁ is tangent to circle at D
  2 * D.1 = E.1 + F₁.1 ∧ 2 * D.2 = E.2 + F₁.2 →       -- D is midpoint of EF₁
  (λ (x y : ℝ) => x + 2*y = 0 ∨ x - 2*y = 0) E.1 E.2  -- Asymptotes equations
  := by sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l3529_352965


namespace NUMINAMATH_CALUDE_quadratic_root_transformation_l3529_352958

theorem quadratic_root_transformation (a b c r s : ℝ) :
  (∀ x, a * x^2 + b * x + c = 0 ↔ x = r ∨ x = s) →
  (∀ x, x^2 - b * x + a * c = 0 ↔ x = a * r + b ∨ x = a * s + b) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_transformation_l3529_352958


namespace NUMINAMATH_CALUDE_ribbon_shortage_l3529_352991

theorem ribbon_shortage (total_ribbon : ℝ) (num_gifts : ℕ) (ribbon_per_gift : ℝ) (ribbon_per_bow : ℝ) :
  total_ribbon = 18 →
  num_gifts = 6 →
  ribbon_per_gift = 2 →
  ribbon_per_bow = 1.5 →
  total_ribbon - (num_gifts * ribbon_per_gift + num_gifts * ribbon_per_bow) = -3 := by
  sorry

end NUMINAMATH_CALUDE_ribbon_shortage_l3529_352991


namespace NUMINAMATH_CALUDE_journey_rate_problem_l3529_352925

/-- Proves that given a 640-mile journey split into two equal halves, 
    where the second half takes 200% longer than the first half, 
    and the average rate for the entire trip is 40 miles per hour, 
    the average rate for the first half of the trip is 80 miles per hour. -/
theorem journey_rate_problem (total_distance : ℝ) (first_half_rate : ℝ) :
  total_distance = 640 →
  (total_distance / 2) / first_half_rate + 3 * ((total_distance / 2) / first_half_rate) = total_distance / 40 →
  first_half_rate = 80 := by
  sorry

end NUMINAMATH_CALUDE_journey_rate_problem_l3529_352925


namespace NUMINAMATH_CALUDE_quadratic_coefficients_unique_l3529_352988

/-- A quadratic function with specific properties -/
def QuadraticFunction (a b c : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 + b * x + c

theorem quadratic_coefficients_unique :
  ∀ a b c : ℝ,
    (∀ x, QuadraticFunction a b c x ≤ QuadraticFunction a b c (-0.75)) ∧
    QuadraticFunction a b c (-0.75) = 3.25 ∧
    QuadraticFunction a b c 0 = 1 →
    a = -4 ∧ b = -6 ∧ c = 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_coefficients_unique_l3529_352988


namespace NUMINAMATH_CALUDE_count_multiples_of_three_l3529_352962

/-- An arithmetic sequence with first term 9 and 8th term 12 -/
structure ArithmeticSequence where
  a₁ : ℕ
  a₈ : ℕ
  h₁ : a₁ = 9
  h₈ : a₈ = 12

/-- The number of terms among the first 2015 that are multiples of 3 -/
def multiples_of_three (seq : ArithmeticSequence) : ℕ :=
  sorry

/-- The main theorem -/
theorem count_multiples_of_three (seq : ArithmeticSequence) :
  multiples_of_three seq = 288 := by
  sorry

end NUMINAMATH_CALUDE_count_multiples_of_three_l3529_352962


namespace NUMINAMATH_CALUDE_selling_price_calculation_l3529_352900

theorem selling_price_calculation (cost_price : ℝ) (gain_percent : ℝ) (selling_price : ℝ) : 
  cost_price = 100 → gain_percent = 10 → selling_price = cost_price * (1 + gain_percent / 100) → selling_price = 110 := by
  sorry

end NUMINAMATH_CALUDE_selling_price_calculation_l3529_352900


namespace NUMINAMATH_CALUDE_divisor_problem_l3529_352948

theorem divisor_problem (initial_number : ℕ) (added_number : ℕ) (divisor : ℕ) : 
  initial_number = 8679921 →
  added_number = 72 →
  divisor = 69 →
  (initial_number + added_number) % divisor = 0 :=
by sorry

end NUMINAMATH_CALUDE_divisor_problem_l3529_352948


namespace NUMINAMATH_CALUDE_four_solutions_to_equation_l3529_352942

theorem four_solutions_to_equation :
  ∃! (s : Finset (ℤ × ℤ)), s.card = 4 ∧ ∀ (x y : ℤ), (x, y) ∈ s ↔ x^2020 + y^2 = 2*y :=
sorry

end NUMINAMATH_CALUDE_four_solutions_to_equation_l3529_352942


namespace NUMINAMATH_CALUDE_remainder_problem_l3529_352968

theorem remainder_problem (a b : ℕ) (h1 : 3 * a > b) (h2 : a % 5 = 1) (h3 : b % 5 = 4) :
  (3 * a - b) % 5 = 4 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l3529_352968


namespace NUMINAMATH_CALUDE_inequality_solution_l3529_352975

theorem inequality_solution (x : ℝ) : (x^2 - 9) / (x^2 - 1) > 0 ↔ x > 3 ∨ x < -3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l3529_352975


namespace NUMINAMATH_CALUDE_v_domain_characterization_l3529_352978

/-- The function v(x) = 1 / sqrt(x^2 - 4) -/
noncomputable def v (x : ℝ) : ℝ := 1 / Real.sqrt (x^2 - 4)

/-- The domain of v(x) -/
def domain_v : Set ℝ := {x | x < -2 ∨ x > 2}

theorem v_domain_characterization :
  ∀ x : ℝ, v x ∈ Set.univ ↔ x ∈ domain_v :=
by sorry

end NUMINAMATH_CALUDE_v_domain_characterization_l3529_352978


namespace NUMINAMATH_CALUDE_brick_surface_area_is_54_l3529_352943

/-- Represents the surface areas of a brick -/
structure BrickAreas where
  front : ℝ
  side : ℝ
  top : ℝ

/-- The surface areas of the three arrangements -/
def arrangement1 (b : BrickAreas) : ℝ := 4 * b.front + 4 * b.side + 2 * b.top
def arrangement2 (b : BrickAreas) : ℝ := 4 * b.front + 2 * b.side + 4 * b.top
def arrangement3 (b : BrickAreas) : ℝ := 2 * b.front + 4 * b.side + 4 * b.top

/-- The surface area of a single brick -/
def brickSurfaceArea (b : BrickAreas) : ℝ := 2 * (b.front + b.side + b.top)

theorem brick_surface_area_is_54 (b : BrickAreas) 
  (h1 : arrangement1 b = 72)
  (h2 : arrangement2 b = 96)
  (h3 : arrangement3 b = 102) : 
  brickSurfaceArea b = 54 := by
  sorry

end NUMINAMATH_CALUDE_brick_surface_area_is_54_l3529_352943


namespace NUMINAMATH_CALUDE_g_neg_501_l3529_352971

-- Define the function g
variable (g : ℝ → ℝ)

-- State the conditions
axiom func_eq : ∀ x y : ℝ, g (x * y) + 2 * x = x * g y + g x
axiom g_neg_one : g (-1) = 7

-- State the theorem to be proved
theorem g_neg_501 : g (-501) = 507 := by sorry

end NUMINAMATH_CALUDE_g_neg_501_l3529_352971


namespace NUMINAMATH_CALUDE_problem_proof_l3529_352964

theorem problem_proof : 289 + 2 * 17 * 8 + 64 = 625 := by
  sorry

end NUMINAMATH_CALUDE_problem_proof_l3529_352964


namespace NUMINAMATH_CALUDE_cycle_sale_result_l3529_352984

/-- Calculates the final selling price and overall profit percentage for a cycle sale --/
def cycle_sale_analysis (initial_cost upgrade_cost : ℚ) (profit_margin sales_tax : ℚ) :
  ℚ × ℚ :=
  let total_cost := initial_cost + upgrade_cost
  let selling_price_before_tax := total_cost * (1 + profit_margin)
  let final_selling_price := selling_price_before_tax * (1 + sales_tax)
  let overall_profit := final_selling_price - total_cost
  let overall_profit_percentage := (overall_profit / total_cost) * 100
  (final_selling_price, overall_profit_percentage)

/-- Theorem stating the correct final selling price and overall profit percentage --/
theorem cycle_sale_result :
  let (final_price, profit_percentage) := cycle_sale_analysis 1400 600 (10/100) (5/100)
  final_price = 2310 ∧ profit_percentage = 15.5 := by
  sorry

end NUMINAMATH_CALUDE_cycle_sale_result_l3529_352984


namespace NUMINAMATH_CALUDE_survey_result_l3529_352939

theorem survey_result (total : ℕ) (thought_diseases : ℕ) (said_rabies : ℕ) :
  (thought_diseases : ℚ) / total = 3 / 4 →
  (said_rabies : ℚ) / thought_diseases = 1 / 2 →
  said_rabies = 18 →
  total = 48 :=
by sorry

end NUMINAMATH_CALUDE_survey_result_l3529_352939


namespace NUMINAMATH_CALUDE_f_range_and_max_value_l3529_352934

def f (a : ℝ) (x : ℝ) : ℝ := x^2 + (2*a - 1)*x - 3

theorem f_range_and_max_value :
  (∀ x ∈ Set.Icc (-2 : ℝ) 3, f 2 x ∈ Set.Icc (-21/4 : ℝ) 15) ∧
  (∀ a : ℝ, (∀ x ∈ Set.Icc (-1 : ℝ) 3, f a x ≤ 1) ∧ 
            (∃ x ∈ Set.Icc (-1 : ℝ) 3, f a x = 1) →
            a = -1/3 ∨ a = -1) :=
by sorry

end NUMINAMATH_CALUDE_f_range_and_max_value_l3529_352934


namespace NUMINAMATH_CALUDE_no_extremum_implies_a_nonnegative_l3529_352916

/-- A function that has no extremum on ℝ -/
def NoExtremum (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, ∃ ε > 0, ∀ y : ℝ, |y - x| < ε → f y ≠ f x ∨ (f y < f x ∧ f y > f x)

/-- The main theorem -/
theorem no_extremum_implies_a_nonnegative (a : ℝ) :
  NoExtremum (fun x => Real.exp x + a * x) → a ≥ 0 := by
  sorry


end NUMINAMATH_CALUDE_no_extremum_implies_a_nonnegative_l3529_352916


namespace NUMINAMATH_CALUDE_saree_price_calculation_l3529_352932

theorem saree_price_calculation (P : ℝ) : 
  (P * (1 - 0.15) * (1 - 0.05) = 323) → P = 400 := by
  sorry

end NUMINAMATH_CALUDE_saree_price_calculation_l3529_352932


namespace NUMINAMATH_CALUDE_gcd_lcm_sum_l3529_352933

theorem gcd_lcm_sum : Nat.gcd 24 54 + Nat.lcm 48 18 = 150 := by sorry

end NUMINAMATH_CALUDE_gcd_lcm_sum_l3529_352933


namespace NUMINAMATH_CALUDE_carousel_revolutions_l3529_352997

/-- Given two circular paths with radii r₁ and r₂, where r₁ = 30 feet and r₂ = 10 feet,
    if a point on the first path makes n₁ = 20 revolutions, then a point on the second path
    needs to make n₂ = 60 revolutions to travel the same total distance. -/
theorem carousel_revolutions (r₁ r₂ n₁ : ℝ) (h₁ : r₁ = 30) (h₂ : r₂ = 10) (h₃ : n₁ = 20) :
  ∃ n₂ : ℝ, n₂ = 60 ∧ r₁ * n₁ = r₂ * n₂ := by
  sorry

end NUMINAMATH_CALUDE_carousel_revolutions_l3529_352997


namespace NUMINAMATH_CALUDE_composition_ratio_l3529_352986

def f (x : ℝ) : ℝ := 3 * x + 4

def g (x : ℝ) : ℝ := 4 * x - 3

theorem composition_ratio : f (g (f 2)) / g (f (g 2)) = 115 / 73 := by
  sorry

end NUMINAMATH_CALUDE_composition_ratio_l3529_352986


namespace NUMINAMATH_CALUDE_milk_buckets_l3529_352981

theorem milk_buckets (bucket_capacity : ℝ) (total_milk : ℝ) : 
  bucket_capacity = 15 → total_milk = 147 → ⌈total_milk / bucket_capacity⌉ = 10 := by
  sorry

end NUMINAMATH_CALUDE_milk_buckets_l3529_352981


namespace NUMINAMATH_CALUDE_fraction_bounds_l3529_352952

theorem fraction_bounds (x y : ℝ) (h : x^2*y^2 + x*y + 1 = 3*y^2) :
  let F := (y - x) / (x + 4*y)
  0 ≤ F ∧ F ≤ 4 := by sorry

end NUMINAMATH_CALUDE_fraction_bounds_l3529_352952


namespace NUMINAMATH_CALUDE_sqrt_15_simplest_l3529_352974

-- Define what it means for a square root to be in its simplest form
def is_simplest_sqrt (n : ℝ) : Prop :=
  ∀ (a b : ℝ), a > 0 ∧ b > 0 → n ≠ a * b^2

-- Theorem statement
theorem sqrt_15_simplest : is_simplest_sqrt 15 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_15_simplest_l3529_352974


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l3529_352913

theorem partial_fraction_decomposition :
  ∃ (P Q R : ℚ),
    (P = -8/15 ∧ Q = -7/6 ∧ R = 27/10) ∧
    ∀ (x : ℚ), x ≠ 1 → x ≠ 4 → x ≠ 6 →
      (x^2 - 9) / ((x - 1) * (x - 4) * (x - 6)) =
      P / (x - 1) + Q / (x - 4) + R / (x - 6) :=
by sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l3529_352913


namespace NUMINAMATH_CALUDE_janes_calculation_l3529_352941

theorem janes_calculation (x y z : ℝ) 
  (h1 : x - (y - z) = 15) 
  (h2 : x - y - z = 7) : 
  x - y = 11 := by
sorry

end NUMINAMATH_CALUDE_janes_calculation_l3529_352941


namespace NUMINAMATH_CALUDE_shooting_match_sequences_l3529_352961

/-- Represents the number of targets in each column --/
structure TargetArrangement where
  columnA : Nat
  columnB : Nat
  columnC : Nat

/-- Calculates the number of valid sequences for breaking targets --/
def validSequences (arrangement : TargetArrangement) : Nat :=
  (Nat.factorial 4 / Nat.factorial 1 / Nat.factorial 3) *
  (Nat.factorial 6 / Nat.factorial 3 / Nat.factorial 3)

/-- Theorem statement for the shooting match problem --/
theorem shooting_match_sequences (arrangement : TargetArrangement)
  (h1 : arrangement.columnA = 4)
  (h2 : arrangement.columnB = 3)
  (h3 : arrangement.columnC = 3) :
  validSequences arrangement = 80 := by
  sorry

end NUMINAMATH_CALUDE_shooting_match_sequences_l3529_352961


namespace NUMINAMATH_CALUDE_hyperbola_k_range_l3529_352995

-- Define the hyperbola equation
def hyperbola_equation (x y k : ℝ) : Prop :=
  x^2 / (k - 3) + y^2 / (2 - k) = 1

-- Define the condition for foci on y-axis
def foci_on_y_axis (k : ℝ) : Prop :=
  2 - k > 0 ∧ k - 3 < 0

-- Theorem statement
theorem hyperbola_k_range :
  ∀ k : ℝ, (∃ x y : ℝ, hyperbola_equation x y k) ∧ foci_on_y_axis k → k < 2 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_k_range_l3529_352995


namespace NUMINAMATH_CALUDE_m_xor_n_equals_target_l3529_352907

-- Define the custom set operation ⊗
def setXor (A B : Set ℝ) : Set ℝ := (A ∪ B) \ (A ∩ B)

-- Define sets M and N
def M : Set ℝ := {x | |x| < 2}
def N : Set ℝ := {x | x^2 - 4*x + 3 < 0}

-- State the theorem
theorem m_xor_n_equals_target : 
  setXor M N = {x | -2 < x ∧ x ≤ 1 ∨ 2 ≤ x ∧ x < 3} := by sorry

end NUMINAMATH_CALUDE_m_xor_n_equals_target_l3529_352907


namespace NUMINAMATH_CALUDE_age_difference_proof_l3529_352908

theorem age_difference_proof (total_age : ℕ) (ratio_a ratio_b ratio_c ratio_d : ℕ) :
  total_age = 190 ∧ ratio_a = 4 ∧ ratio_b = 3 ∧ ratio_c = 7 ∧ ratio_d = 5 →
  ∃ (x : ℚ), x * (ratio_a + ratio_b + ratio_c + ratio_d) = total_age ∧
             x * ratio_a - x * ratio_b = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_age_difference_proof_l3529_352908


namespace NUMINAMATH_CALUDE_fixed_point_of_parabolas_l3529_352904

/-- The function f_m that defines the family of parabolas -/
def f_m (m : ℝ) (x : ℝ) : ℝ := (m^2 + m + 1) * x^2 - 2 * (m^2 + 1) * x + m^2 - m + 1

/-- Theorem stating that (1, 0) is the fixed common point of all parabolas -/
theorem fixed_point_of_parabolas :
  ∀ m : ℝ, f_m m 1 = 1 := by sorry

end NUMINAMATH_CALUDE_fixed_point_of_parabolas_l3529_352904


namespace NUMINAMATH_CALUDE_expression_simplification_l3529_352976

theorem expression_simplification (a y : ℝ) : 
  ((1 : ℝ) * (3 * a^2 - 2 * a) + 2 * (a^2 - a + 2) = 5 * a^2 - 4 * a + 4) ∧ 
  ((2 : ℝ) * (2 * y^2 - 1/2 + 3 * y) - 2 * (y - y^2 + 1/2) = 4 * y^2 + y - 3/2) := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3529_352976


namespace NUMINAMATH_CALUDE_f_min_value_l3529_352953

/-- The function f to be minimized -/
def f (x y z : ℝ) : ℝ :=
  x^2 + 2*y^2 + 3*z^2 + 2*x*y + 4*y*z + 2*z*x - 6*x - 10*y - 12*z

/-- Theorem stating that -14 is the minimum value of f -/
theorem f_min_value :
  ∀ x y z : ℝ, f x y z ≥ -14 :=
by sorry

end NUMINAMATH_CALUDE_f_min_value_l3529_352953


namespace NUMINAMATH_CALUDE_sum_of_absolute_values_l3529_352982

theorem sum_of_absolute_values (a b : ℤ) : 
  (abs a = 5 ∧ abs b = 3) → 
  (a + b = 8 ∨ a + b = 2 ∨ a + b = -2 ∨ a + b = -8) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_absolute_values_l3529_352982


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3529_352955

def A : Set ℤ := {-2, 0, 1}
def B : Set ℤ := {x | x^2 > 1}

theorem intersection_of_A_and_B : A ∩ B = {-2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3529_352955


namespace NUMINAMATH_CALUDE_kathryn_remaining_money_l3529_352951

/-- Calculates the remaining money for Kathryn after expenses --/
def remaining_money (rent : ℕ) (salary : ℕ) : ℕ :=
  let food_travel : ℕ := 2 * rent
  let rent_share : ℕ := rent / 2
  let total_expenses : ℕ := rent_share + food_travel
  salary - total_expenses

/-- Proves that Kathryn's remaining money after expenses is $2000 --/
theorem kathryn_remaining_money :
  remaining_money 1200 5000 = 2000 := by
  sorry

#eval remaining_money 1200 5000

end NUMINAMATH_CALUDE_kathryn_remaining_money_l3529_352951


namespace NUMINAMATH_CALUDE_square_area_10m_l3529_352957

/-- The area of a square with side length 10 meters is 100 square meters. -/
theorem square_area_10m : 
  let side_length : ℝ := 10
  let square_area := side_length ^ 2
  square_area = 100 := by sorry

end NUMINAMATH_CALUDE_square_area_10m_l3529_352957


namespace NUMINAMATH_CALUDE_angle_inequality_equivalence_l3529_352973

theorem angle_inequality_equivalence (θ : Real) (h : 0 ≤ θ ∧ θ ≤ 2 * Real.pi) :
  (∀ x : Real, 0 ≤ x ∧ x ≤ 1 → x^2 * Real.cos θ - x * (1 - x) + (1 - x)^2 * Real.sin θ < 0) ↔
  (Real.pi / 2 < θ ∧ θ < 3 * Real.pi / 2) := by sorry

end NUMINAMATH_CALUDE_angle_inequality_equivalence_l3529_352973


namespace NUMINAMATH_CALUDE_probability_correct_l3529_352970

/-- Represents a standard six-sided die --/
def Die := Fin 6

/-- The probability of the event described in the problem --/
def probability : ℚ :=
  (5 * 4^9) / (6^11)

/-- The function that calculates the probability of the event --/
def calculate_probability : ℚ :=
  -- First roll: any number (1)
  -- Rolls 2 to 10: different from previous, not 4 on 11th (5/6 * (4/5)^9)
  -- 11th and 12th rolls both 4 (1/6 * 1/6)
  1 * (5/6) * (4/5)^9 * (1/6)^2

theorem probability_correct :
  calculate_probability = probability := by sorry

end NUMINAMATH_CALUDE_probability_correct_l3529_352970


namespace NUMINAMATH_CALUDE_line_passes_through_fixed_point_l3529_352917

/-- The line equation as a function of x, y, and a -/
def line_equation (x y a : ℝ) : ℝ := (a + 1) * x + y - 2 - a

/-- Theorem stating that the line always passes through the point (1, 1) -/
theorem line_passes_through_fixed_point :
  ∀ a : ℝ, line_equation 1 1 a = 0 := by
  sorry

end NUMINAMATH_CALUDE_line_passes_through_fixed_point_l3529_352917


namespace NUMINAMATH_CALUDE_product_36_sum_0_l3529_352921

theorem product_36_sum_0 (a b c d e f : ℤ) : 
  a * b * c * d * e * f = 36 ∧ 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧
  c ≠ d ∧ c ≠ e ∧ c ≠ f ∧
  d ≠ e ∧ d ≠ f ∧
  e ≠ f →
  a + b + c + d + e + f = 0 :=
sorry

end NUMINAMATH_CALUDE_product_36_sum_0_l3529_352921
