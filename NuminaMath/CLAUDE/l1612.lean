import Mathlib

namespace NUMINAMATH_CALUDE_product_xyz_l1612_161216

theorem product_xyz (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h1 : x * (y + z) = 162)
  (h2 : y * (z + x) = 180)
  (h3 : z * (x + y) = 198)
  (h4 : x + y + z = 26) :
  x * y * z = 2294.67 := by
sorry

end NUMINAMATH_CALUDE_product_xyz_l1612_161216


namespace NUMINAMATH_CALUDE_necessary_not_sufficient_l1612_161298

theorem necessary_not_sufficient (p q : Prop) :
  (¬p → ¬(p ∨ q)) ∧ ¬(¬p → ¬(p ∨ q)) :=
by sorry

end NUMINAMATH_CALUDE_necessary_not_sufficient_l1612_161298


namespace NUMINAMATH_CALUDE_inscribed_hexagon_side_length_l1612_161218

/-- Triangle ABC with given side lengths and angle -/
structure Triangle where
  AB : ℝ
  AC : ℝ
  BAC : ℝ

/-- Regular hexagon UVWXYZ inscribed in triangle ABC -/
structure InscribedHexagon where
  triangle : Triangle
  sideLength : ℝ

/-- Theorem stating the side length of the inscribed hexagon -/
theorem inscribed_hexagon_side_length (t : Triangle) (h : InscribedHexagon) 
  (h1 : t.AB = 5)
  (h2 : t.AC = 8)
  (h3 : t.BAC = π / 3)
  (h4 : h.triangle = t)
  (h5 : ∃ (U V W X Z : ℝ × ℝ), 
    U.1 + V.1 = t.AB ∧ 
    W.2 + X.2 = t.AC ∧ 
    Z.1^2 + Z.2^2 = t.AB^2 + t.AC^2 - 2 * t.AB * t.AC * Real.cos t.BAC) :
  h.sideLength = 35 / 19 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_hexagon_side_length_l1612_161218


namespace NUMINAMATH_CALUDE_power_six_mod_five_remainder_six_power_23_mod_five_l1612_161288

theorem power_six_mod_five (n : ℕ) : 6^n ≡ 1 [ZMOD 5] := by sorry

theorem remainder_six_power_23_mod_five : 6^23 ≡ 1 [ZMOD 5] := by sorry

end NUMINAMATH_CALUDE_power_six_mod_five_remainder_six_power_23_mod_five_l1612_161288


namespace NUMINAMATH_CALUDE_least_reducible_fraction_l1612_161231

def is_reducible (n : ℕ) : Prop :=
  n > 17 ∧ Nat.gcd (n - 17) (7 * n + 4) > 1

theorem least_reducible_fraction :
  (∀ m : ℕ, m < 20 → ¬ is_reducible m) ∧ is_reducible 20 :=
sorry

end NUMINAMATH_CALUDE_least_reducible_fraction_l1612_161231


namespace NUMINAMATH_CALUDE_five_mondays_in_march_after_five_sunday_february_l1612_161201

/-- Represents the days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a year -/
structure Year where
  value : ℕ
  isLeapYear : Bool

/-- Represents a month in a specific year -/
structure Month where
  year : Year
  monthNumber : ℕ
  days : ℕ
  firstDay : DayOfWeek

/-- Function to get the next day of the week -/
def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

/-- Function to count occurrences of a specific day in a month -/
def countDayInMonth (m : Month) (d : DayOfWeek) : ℕ := sorry

theorem five_mondays_in_march_after_five_sunday_february 
  (y : Year) 
  (feb : Month) 
  (mar : Month) :
  y.isLeapYear = true →
  feb.year = y →
  feb.monthNumber = 2 →
  feb.days = 29 →
  mar.year = y →
  mar.monthNumber = 3 →
  mar.days = 31 →
  countDayInMonth feb DayOfWeek.Sunday = 5 →
  mar.firstDay = nextDay feb.firstDay →
  countDayInMonth mar DayOfWeek.Monday = 5 := by
  sorry


end NUMINAMATH_CALUDE_five_mondays_in_march_after_five_sunday_february_l1612_161201


namespace NUMINAMATH_CALUDE_largest_prime_divisor_of_sum_of_squares_l1612_161290

theorem largest_prime_divisor_of_sum_of_squares :
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ (35^2 + 84^2) ∧ 
  ∀ (q : ℕ), Nat.Prime q → q ∣ (35^2 + 84^2) → q ≤ p :=
by sorry

end NUMINAMATH_CALUDE_largest_prime_divisor_of_sum_of_squares_l1612_161290


namespace NUMINAMATH_CALUDE_seven_digit_multiples_of_three_l1612_161242

theorem seven_digit_multiples_of_three (D B C : ℕ) : 
  D < 10 → B < 10 → C < 10 →
  (8 * 1000000 + 5 * 100000 + D * 10000 + 6 * 1000 + 3 * 100 + B * 10 + 2) % 3 = 0 →
  (4 * 1000000 + 1 * 100000 + 7 * 10000 + D * 1000 + B * 100 + 5 * 10 + C) % 3 = 0 →
  C = 2 := by
sorry

end NUMINAMATH_CALUDE_seven_digit_multiples_of_three_l1612_161242


namespace NUMINAMATH_CALUDE_lemon_juice_fraction_l1612_161228

theorem lemon_juice_fraction (total_members : ℕ) (orange_juice_orders : ℕ) : 
  total_members = 30 →
  orange_juice_orders = 6 →
  ∃ (lemon_fraction : ℚ),
    lemon_fraction = 7 / 10 ∧
    lemon_fraction * total_members +
    (1 / 3) * (total_members - lemon_fraction * total_members) +
    orange_juice_orders = total_members :=
by sorry

end NUMINAMATH_CALUDE_lemon_juice_fraction_l1612_161228


namespace NUMINAMATH_CALUDE_gain_percentage_proof_l1612_161295

theorem gain_percentage_proof (C S : ℝ) (h : 80 * C = 25 * S) : 
  (S - C) / C * 100 = 220 := by
sorry

end NUMINAMATH_CALUDE_gain_percentage_proof_l1612_161295


namespace NUMINAMATH_CALUDE_ellipse_fixed_point_l1612_161214

-- Define the ellipse C
def ellipse_C (x y : ℝ) : Prop := x^2 / 2 + y^2 = 1

-- Define the upper vertex M of the ellipse
def upper_vertex (M : ℝ × ℝ) : Prop := 
  M.1 = 0 ∧ M.2 = 1

-- Define points A and B on the ellipse
def on_ellipse (P : ℝ × ℝ) : Prop := 
  ellipse_C P.1 P.2

-- Define the slopes of lines MA and MB
def slopes_sum_2 (k₁ k₂ : ℝ) : Prop := 
  k₁ + k₂ = 2

-- Theorem statement
theorem ellipse_fixed_point 
  (M A B : ℝ × ℝ) 
  (k₁ k₂ : ℝ) 
  (hM : upper_vertex M) 
  (hA : on_ellipse A) 
  (hB : on_ellipse B) 
  (hk : slopes_sum_2 k₁ k₂) :
  ∃ (t : ℝ), A.1 * t + A.2 * (1 - t) = -1 ∧ 
             B.1 * t + B.2 * (1 - t) = -1 :=
sorry

end NUMINAMATH_CALUDE_ellipse_fixed_point_l1612_161214


namespace NUMINAMATH_CALUDE_average_exercise_days_l1612_161227

def exercise_data : List (Nat × Nat) := [
  (1, 1), (2, 3), (3, 2), (4, 6), (5, 8), (6, 3), (7, 2)
]

def total_exercise_days : Nat :=
  (exercise_data.map (fun (days, freq) => days * freq)).sum

def total_students : Nat :=
  (exercise_data.map (fun (_, freq) => freq)).sum

theorem average_exercise_days :
  (total_exercise_days : ℚ) / (total_students : ℚ) = 436 / 100 := by sorry

end NUMINAMATH_CALUDE_average_exercise_days_l1612_161227


namespace NUMINAMATH_CALUDE_more_sad_left_l1612_161257

/-- Represents the state of a player in the game -/
inductive PlayerState
| Sad
| Cheerful

/-- Represents the game with its rules and initial state -/
structure Game where
  initial_players : ℕ
  remaining_player : ℕ
  sad_left : ℕ
  cheerful_left : ℕ

/-- The game rules ensure that when only one player remains, more sad players have left than cheerful players -/
theorem more_sad_left (g : Game) 
  (h1 : g.initial_players = 36)
  (h2 : g.remaining_player = 1)
  (h3 : g.sad_left + g.cheerful_left = g.initial_players - g.remaining_player) :
  g.sad_left > g.cheerful_left := by
  sorry

#check more_sad_left

end NUMINAMATH_CALUDE_more_sad_left_l1612_161257


namespace NUMINAMATH_CALUDE_orthogonal_vectors_l1612_161246

theorem orthogonal_vectors (y : ℝ) : y = 28 / 3 →
  (3 : ℝ) * y + 7 * (-4 : ℝ) = 0 := by
  sorry

end NUMINAMATH_CALUDE_orthogonal_vectors_l1612_161246


namespace NUMINAMATH_CALUDE_gravelling_cost_theorem_l1612_161255

/-- The cost of gravelling a path around a rectangular plot -/
theorem gravelling_cost_theorem 
  (plot_length : ℝ) 
  (plot_width : ℝ) 
  (path_width : ℝ) 
  (cost_per_sqm_paise : ℝ) 
  (h1 : plot_length = 110) 
  (h2 : plot_width = 65) 
  (h3 : path_width = 2.5) 
  (h4 : cost_per_sqm_paise = 60) : 
  (((plot_length * plot_width) - ((plot_length - 2 * path_width) * (plot_width - 2 * path_width))) * (cost_per_sqm_paise / 100)) = 510 := by
  sorry

end NUMINAMATH_CALUDE_gravelling_cost_theorem_l1612_161255


namespace NUMINAMATH_CALUDE_box_volume_correct_l1612_161213

/-- The volume of an open box created from a rectangular sheet -/
def boxVolume (L W S : ℝ) : ℝ := (L - 2*S) * (W - 2*S) * S

/-- Theorem stating that the boxVolume function correctly calculates the volume of the open box -/
theorem box_volume_correct (L W S : ℝ) (hL : L > 0) (hW : W > 0) (hS : 0 < S ∧ S < L/2 ∧ S < W/2) : 
  boxVolume L W S = (L - 2*S) * (W - 2*S) * S :=
sorry

end NUMINAMATH_CALUDE_box_volume_correct_l1612_161213


namespace NUMINAMATH_CALUDE_fraction_sum_equals_decimal_l1612_161200

theorem fraction_sum_equals_decimal : 
  (1 : ℚ) / 10 + 9 / 100 + 9 / 1000 + 7 / 10000 = 0.1997 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equals_decimal_l1612_161200


namespace NUMINAMATH_CALUDE_planted_fraction_is_correct_l1612_161271

/-- Represents a right triangle with an unplanted square in the corner -/
structure FieldTriangle where
  leg1 : ℝ
  leg2 : ℝ
  square_distance : ℝ

/-- The fraction of the field that is planted -/
def planted_fraction (f : FieldTriangle) : ℚ :=
  367 / 375

theorem planted_fraction_is_correct (f : FieldTriangle) 
  (h1 : f.leg1 = 5)
  (h2 : f.leg2 = 12)
  (h3 : f.square_distance = 4) :
  planted_fraction f = 367 / 375 := by
  sorry

end NUMINAMATH_CALUDE_planted_fraction_is_correct_l1612_161271


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1612_161212

def A : Set ℕ := {2, 3, 4}
def B : Set ℕ := {1, 2, 3}

theorem intersection_of_A_and_B : A ∩ B = {2, 3} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1612_161212


namespace NUMINAMATH_CALUDE_domain_of_f_with_restricted_range_l1612_161240

def f (x : ℝ) : ℝ := x^2

def domain : Set ℝ := {-2, -1, 1, 2}
def range : Set ℝ := {1, 4}

theorem domain_of_f_with_restricted_range :
  ∀ y ∈ range, ∃ x ∈ domain, f x = y ∧
  ∀ x : ℝ, f x ∈ range → x ∈ domain :=
by
  sorry

end NUMINAMATH_CALUDE_domain_of_f_with_restricted_range_l1612_161240


namespace NUMINAMATH_CALUDE_light_source_height_l1612_161287

/-- The length of the cube's edge in centimeters -/
def cube_edge : ℝ := 2

/-- The area of the shadow cast by the cube, excluding the area beneath the cube, in square centimeters -/
def shadow_area : ℝ := 98

/-- The height of the light source above a top vertex of the cube in centimeters -/
def y : ℝ := sorry

/-- The theorem stating that the greatest integer not exceeding 1000y is 500 -/
theorem light_source_height : ⌊1000 * y⌋ = 500 := by sorry

end NUMINAMATH_CALUDE_light_source_height_l1612_161287


namespace NUMINAMATH_CALUDE_subtract_negative_l1612_161256

theorem subtract_negative : 2 - (-3) = 5 := by
  sorry

end NUMINAMATH_CALUDE_subtract_negative_l1612_161256


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l1612_161261

/-- 
Given a geometric sequence {a_n} where a₁ = -1 and a₄ = 8,
prove that the common ratio is -2.
-/
theorem geometric_sequence_common_ratio :
  ∀ (a : ℕ → ℝ),
  (∀ n, a (n + 1) = a n * (a 2 / a 1)) →  -- Definition of geometric sequence
  a 1 = -1 →
  a 4 = 8 →
  a 2 / a 1 = -2 :=
by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l1612_161261


namespace NUMINAMATH_CALUDE_vanessa_recycled_20_pounds_l1612_161245

/-- The number of pounds that earn one point -/
def pounds_per_point : ℕ := 9

/-- The number of pounds Vanessa's friends recycled -/
def friends_pounds : ℕ := 16

/-- The total number of points earned -/
def total_points : ℕ := 4

/-- Vanessa's recycled pounds -/
def vanessa_pounds : ℕ := total_points * pounds_per_point - friends_pounds

theorem vanessa_recycled_20_pounds : vanessa_pounds = 20 := by
  sorry

end NUMINAMATH_CALUDE_vanessa_recycled_20_pounds_l1612_161245


namespace NUMINAMATH_CALUDE_composition_theorem_l1612_161232

def f (x : ℝ) : ℝ := 1 - 2 * x
def g (x : ℝ) : ℝ := x^2 + 3

theorem composition_theorem :
  (∀ x : ℝ, f (g x) = -2 * x^2 - 5) ∧
  (∀ x : ℝ, g (f x) = 4 * x^2 - 4 * x + 4) := by
sorry

end NUMINAMATH_CALUDE_composition_theorem_l1612_161232


namespace NUMINAMATH_CALUDE_mixture_weight_l1612_161247

/-- The weight of the mixture of two brands of vegetable ghee -/
theorem mixture_weight (weight_a weight_b : ℝ) (ratio_a ratio_b : ℕ) (total_volume : ℝ) : 
  weight_a = 900 →
  weight_b = 800 →
  ratio_a = 3 →
  ratio_b = 2 →
  total_volume = 4 →
  (((ratio_a : ℝ) / ((ratio_a : ℝ) + (ratio_b : ℝ))) * total_volume * weight_a +
   ((ratio_b : ℝ) / ((ratio_a : ℝ) + (ratio_b : ℝ))) * total_volume * weight_b) / 1000 = 3.44 := by
  sorry

#check mixture_weight

end NUMINAMATH_CALUDE_mixture_weight_l1612_161247


namespace NUMINAMATH_CALUDE_abs_z_squared_value_l1612_161262

theorem abs_z_squared_value (z : ℂ) (h : z^2 + Complex.abs z^2 = 7 + 6*I) : 
  Complex.abs z^2 = 85/14 := by
  sorry

end NUMINAMATH_CALUDE_abs_z_squared_value_l1612_161262


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l1612_161286

theorem simplify_and_evaluate (m : ℚ) (h : m = 2) : 
  ((2 * m + 1) / m - 1) / ((m^2 - 1) / m) = 1 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l1612_161286


namespace NUMINAMATH_CALUDE_remainder_theorem_l1612_161233

-- Define the polynomial Q
variable (Q : ℝ → ℝ)

-- Define the conditions
axiom rem_20 : ∃ P : ℝ → ℝ, ∀ x, Q x = (x - 20) * (P x) + 120
axiom rem_100 : ∃ P : ℝ → ℝ, ∀ x, Q x = (x - 100) * (P x) + 40

-- Theorem statement
theorem remainder_theorem :
  ∃ R : ℝ → ℝ, ∀ x, Q x = (x - 20) * (x - 100) * (R x) + (-x + 140) :=
sorry

end NUMINAMATH_CALUDE_remainder_theorem_l1612_161233


namespace NUMINAMATH_CALUDE_largest_multiple_of_9_under_100_l1612_161230

theorem largest_multiple_of_9_under_100 : 
  ∃ n : ℕ, n * 9 = 99 ∧ ∀ m : ℕ, m * 9 < 100 → m * 9 ≤ 99 :=
by sorry

end NUMINAMATH_CALUDE_largest_multiple_of_9_under_100_l1612_161230


namespace NUMINAMATH_CALUDE_rogers_shelves_l1612_161234

/-- Given the conditions of Roger's book shelving problem, prove that he needs 4 shelves. -/
theorem rogers_shelves (total_books : ℕ) (librarian_books : ℕ) (books_per_shelf : ℕ) 
  (h1 : total_books = 14) 
  (h2 : librarian_books = 2) 
  (h3 : books_per_shelf = 3) : 
  ((total_books - librarian_books) / books_per_shelf : ℕ) = 4 := by
  sorry

end NUMINAMATH_CALUDE_rogers_shelves_l1612_161234


namespace NUMINAMATH_CALUDE_pizza_not_crust_percentage_l1612_161268

def pizza_weight : ℝ := 800
def crust_weight : ℝ := 200

theorem pizza_not_crust_percentage :
  (pizza_weight - crust_weight) / pizza_weight * 100 = 75 := by
  sorry

end NUMINAMATH_CALUDE_pizza_not_crust_percentage_l1612_161268


namespace NUMINAMATH_CALUDE_deepak_age_l1612_161259

theorem deepak_age (arun_age deepak_age : ℕ) : 
  (arun_age : ℚ) / deepak_age = 4 / 3 →
  arun_age + 6 = 26 →
  deepak_age = 15 := by
sorry

end NUMINAMATH_CALUDE_deepak_age_l1612_161259


namespace NUMINAMATH_CALUDE_fifteen_initial_points_theorem_l1612_161281

/-- Represents the number of points after k densifications -/
def points_after_densification (initial_points : ℕ) (densifications : ℕ) : ℕ :=
  initial_points * 2^densifications - (2^densifications - 1)

/-- Theorem stating that 15 initial points results in 113 points after 3 densifications -/
theorem fifteen_initial_points_theorem :
  ∃ (n : ℕ), n > 0 ∧ points_after_densification n 3 = 113 → n = 15 :=
sorry

end NUMINAMATH_CALUDE_fifteen_initial_points_theorem_l1612_161281


namespace NUMINAMATH_CALUDE_minimum_nickels_needed_l1612_161205

def shoe_cost : ℚ := 45.50
def ten_dollar_bills : ℕ := 4
def quarters : ℕ := 5
def quarter_value : ℚ := 0.25
def nickel_value : ℚ := 0.05

theorem minimum_nickels_needed :
  ∃ n : ℕ, 
    (n : ℚ) * nickel_value + (ten_dollar_bills * 10 : ℚ) + (quarters : ℚ) * quarter_value ≥ shoe_cost ∧
    ∀ m : ℕ, m < n → (m : ℚ) * nickel_value + (ten_dollar_bills * 10 : ℚ) + (quarters : ℚ) * quarter_value < shoe_cost :=
by
  sorry

end NUMINAMATH_CALUDE_minimum_nickels_needed_l1612_161205


namespace NUMINAMATH_CALUDE_prob_diamond_ace_king_l1612_161265

/-- The number of cards in the modified deck -/
def deck_size : ℕ := 56

/-- The number of cards that are either diamonds, aces, or kings -/
def target_cards : ℕ := 20

/-- The probability of drawing a card that is not a diamond, ace, or king -/
def prob_not_target : ℚ := (deck_size - target_cards) / deck_size

/-- The probability of drawing at least one diamond, ace, or king in two draws with replacement -/
def prob_at_least_one_target : ℚ := 1 - prob_not_target^2

theorem prob_diamond_ace_king : prob_at_least_one_target = 115 / 196 := by
  sorry

end NUMINAMATH_CALUDE_prob_diamond_ace_king_l1612_161265


namespace NUMINAMATH_CALUDE_horner_method_v2_equals_6_l1612_161237

def f (x : ℝ) : ℝ := 1 + 2*x + x^2 - 3*x^3 + 2*x^4

def horner_v2 (a₀ a₁ a₂ a₃ a₄ x : ℝ) : ℝ :=
  let v₁ := a₄ * x + a₃
  v₁ * x + a₂

theorem horner_method_v2_equals_6 :
  horner_v2 1 2 1 (-3) 2 (-1) = 6 := by
  sorry

end NUMINAMATH_CALUDE_horner_method_v2_equals_6_l1612_161237


namespace NUMINAMATH_CALUDE_range_of_g_l1612_161269

noncomputable def g (x : ℝ) : ℝ := Real.arctan (2 * x) + Real.arctan ((2 - 3 * x) / (2 + 3 * x))

theorem range_of_g :
  Set.range g = {-3 * Real.pi / 4, Real.pi / 4} := by sorry

end NUMINAMATH_CALUDE_range_of_g_l1612_161269


namespace NUMINAMATH_CALUDE_odd_natural_not_divisible_by_square_l1612_161210

theorem odd_natural_not_divisible_by_square (n : ℕ) : 
  Odd n → (¬(Nat.factorial (n - 1) % (n^2) = 0) ↔ Nat.Prime n ∨ n = 9) :=
by sorry

end NUMINAMATH_CALUDE_odd_natural_not_divisible_by_square_l1612_161210


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_fractions_l1612_161244

theorem arithmetic_mean_of_fractions (x b : ℝ) (hx : x ≠ 0) :
  (1 / 2) * ((2*x + b) / x + (2*x - b) / x) = 2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_fractions_l1612_161244


namespace NUMINAMATH_CALUDE_january_text_messages_l1612_161263

-- Define the sequence
def text_message_sequence : ℕ → ℕ
| 0 => 1  -- November (first month)
| n + 1 => 2 * text_message_sequence n  -- Each subsequent month

-- Theorem statement
theorem january_text_messages : text_message_sequence 2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_january_text_messages_l1612_161263


namespace NUMINAMATH_CALUDE_polar_coordinates_of_point_l1612_161254

theorem polar_coordinates_of_point (x y : ℝ) (ρ θ : ℝ) :
  x = 1/2 ∧ y = -Real.sqrt 3 / 2 →
  ρ = Real.sqrt (x^2 + y^2) ∧
  θ = Real.arccos (x / ρ) + (if y < 0 then 2 * Real.pi else 0) →
  ρ = 1 ∧ θ = 5 * Real.pi / 3 := by
  sorry

end NUMINAMATH_CALUDE_polar_coordinates_of_point_l1612_161254


namespace NUMINAMATH_CALUDE_perfect_games_count_l1612_161294

theorem perfect_games_count (perfect_score : ℕ) (total_points : ℕ) : 
  perfect_score = 21 → total_points = 63 → total_points / perfect_score = 3 := by
sorry

end NUMINAMATH_CALUDE_perfect_games_count_l1612_161294


namespace NUMINAMATH_CALUDE_units_digit_of_base_l1612_161280

/-- Given a natural number, return its unit's digit -/
def unitsDigit (n : ℕ) : ℕ := n % 10

/-- The product of the given terms -/
def product (x : ℕ) : ℕ := (x ^ 41) * (41 ^ 14) * (14 ^ 87) * (87 ^ 76)

/-- The theorem stating that if the unit's digit of the product is 4, 
    then the unit's digit of x must be 1 -/
theorem units_digit_of_base (x : ℕ) : 
  unitsDigit (product x) = 4 → unitsDigit x = 1 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_base_l1612_161280


namespace NUMINAMATH_CALUDE_largest_number_proof_l1612_161272

theorem largest_number_proof (a b c d e : ℕ) : 
  a + b + c + d = 240 →
  a + b + c + e = 260 →
  a + b + d + e = 280 →
  a + c + d + e = 300 →
  b + c + d + e = 320 →
  a + b = 40 →
  a < b ∧ b < c ∧ c < d ∧ d < e →
  e = 160 := by
sorry

end NUMINAMATH_CALUDE_largest_number_proof_l1612_161272


namespace NUMINAMATH_CALUDE_bus_speed_excluding_stoppages_l1612_161279

/-- Given a bus that stops for 15 minutes per hour and has a speed of 48 km/hr including stoppages,
    its speed excluding stoppages is 64 km/hr. -/
theorem bus_speed_excluding_stoppages 
  (stop_time : ℝ) 
  (speed_with_stoppages : ℝ) 
  (h1 : stop_time = 15) 
  (h2 : speed_with_stoppages = 48) : 
  speed_with_stoppages * (60 / (60 - stop_time)) = 64 :=
sorry

end NUMINAMATH_CALUDE_bus_speed_excluding_stoppages_l1612_161279


namespace NUMINAMATH_CALUDE_remaining_distance_is_4430_l1612_161239

/-- Represents the state of the race between Alex and Max -/
structure RaceState where
  total_distance : ℕ
  alex_lead : ℤ

/-- Calculates the final race state after all lead changes -/
def final_race_state : RaceState :=
  let initial_state : RaceState := { total_distance := 5000, alex_lead := 0 }
  let after_uphill : RaceState := { initial_state with alex_lead := 300 }
  let after_downhill : RaceState := { after_uphill with alex_lead := after_uphill.alex_lead - 170 }
  { after_downhill with alex_lead := after_downhill.alex_lead + 440 }

/-- Calculates the remaining distance for Max to catch up -/
def remaining_distance (state : RaceState) : ℕ :=
  state.total_distance - state.alex_lead.toNat

/-- Theorem stating the remaining distance for Max to catch up -/
theorem remaining_distance_is_4430 :
  remaining_distance final_race_state = 4430 := by
  sorry

end NUMINAMATH_CALUDE_remaining_distance_is_4430_l1612_161239


namespace NUMINAMATH_CALUDE_notecard_area_theorem_l1612_161243

/-- Given a rectangle with original dimensions 5 × 7 inches, prove that if shortening one side
    by 2 inches results in an area of 21 square inches, then shortening the other side
    by 2 inches instead will result in an area of 25 square inches. -/
theorem notecard_area_theorem :
  ∀ (original_width original_length : ℝ),
    original_width = 5 →
    original_length = 7 →
    (∃ (new_width new_length : ℝ),
      (new_width = original_width - 2 ∧ new_length = original_length ∨
       new_width = original_width ∧ new_length = original_length - 2) ∧
      new_width * new_length = 21) →
    ∃ (other_width other_length : ℝ),
      (other_width = original_width - 2 ∧ other_length = original_length ∨
       other_width = original_width ∧ other_length = original_length - 2) ∧
      other_width ≠ new_width ∧
      other_length ≠ new_length ∧
      other_width * other_length = 25 :=
by sorry

end NUMINAMATH_CALUDE_notecard_area_theorem_l1612_161243


namespace NUMINAMATH_CALUDE_beads_per_necklace_l1612_161248

theorem beads_per_necklace (total_beads : ℕ) (total_necklaces : ℕ) 
  (h1 : total_beads = 20) 
  (h2 : total_necklaces = 4) : 
  total_beads / total_necklaces = 5 :=
by sorry

end NUMINAMATH_CALUDE_beads_per_necklace_l1612_161248


namespace NUMINAMATH_CALUDE_jake_car_soap_cost_l1612_161223

/-- Represents the cost of car soap for Jake's car washing schedule -/
def car_soap_cost (washes_per_bottle : ℕ) (bottle_cost : ℚ) (total_washes : ℕ) : ℚ :=
  (total_washes / washes_per_bottle : ℚ) * bottle_cost

/-- Theorem: Jake spends $20.00 on car soap for washing his car once a week for 20 weeks -/
theorem jake_car_soap_cost :
  car_soap_cost 4 4 20 = 20 := by
  sorry

end NUMINAMATH_CALUDE_jake_car_soap_cost_l1612_161223


namespace NUMINAMATH_CALUDE_alex_grocery_delivery_l1612_161297

theorem alex_grocery_delivery (saved : ℝ) (car_cost : ℝ) (trip_charge : ℝ) (grocery_fee_percent : ℝ) (num_trips : ℕ) 
  (h1 : saved = 14500)
  (h2 : car_cost = 14600)
  (h3 : trip_charge = 1.5)
  (h4 : grocery_fee_percent = 0.05)
  (h5 : num_trips = 40) :
  ∃ (grocery_value : ℝ), 
    grocery_value * grocery_fee_percent = car_cost - saved - (trip_charge * num_trips) ∧ 
    grocery_value = 800 := by
sorry

end NUMINAMATH_CALUDE_alex_grocery_delivery_l1612_161297


namespace NUMINAMATH_CALUDE_sum_greater_than_six_random_event_l1612_161275

def numbers : Set ℕ := {n : ℕ | 1 ≤ n ∧ n ≤ 10}

def sumGreaterThanSix (a b c : ℕ) : Prop := a + b + c > 6

theorem sum_greater_than_six_random_event :
  ∃ (a b c : ℕ), a ∈ numbers ∧ b ∈ numbers ∧ c ∈ numbers ∧ sumGreaterThanSix a b c ∧
  ∃ (x y z : ℕ), x ∈ numbers ∧ y ∈ numbers ∧ z ∈ numbers ∧ ¬sumGreaterThanSix x y z :=
sorry

end NUMINAMATH_CALUDE_sum_greater_than_six_random_event_l1612_161275


namespace NUMINAMATH_CALUDE_solution_range_l1612_161224

theorem solution_range (m : ℝ) : 
  (∃ x : ℝ, x < 0 ∧ (2 * x + m) / (x - 1) = 1) → m > -1 := by
  sorry

end NUMINAMATH_CALUDE_solution_range_l1612_161224


namespace NUMINAMATH_CALUDE_power_division_rule_l1612_161217

theorem power_division_rule (x : ℝ) : x^6 / x^3 = x^3 := by
  sorry

end NUMINAMATH_CALUDE_power_division_rule_l1612_161217


namespace NUMINAMATH_CALUDE_a1_value_l1612_161252

theorem a1_value (x : ℝ) (a : Fin 8 → ℝ) :
  (x - 1)^7 = a 0 + a 1 * (x + 1) + a 2 * (x + 1)^2 + a 3 * (x + 1)^3 + 
              a 4 * (x + 1)^4 + a 5 * (x + 1)^5 + a 6 * (x + 1)^6 + a 7 * (x + 1)^7 →
  a 1 = 448 := by
sorry

end NUMINAMATH_CALUDE_a1_value_l1612_161252


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_eight_l1612_161208

/-- Represents a geometric sequence with common ratio 2 -/
def GeometricSequence (a : ℝ) : ℕ → ℝ
  | 0 => a
  | n + 1 => 2 * GeometricSequence a n

/-- Sum of the first n terms of the geometric sequence -/
def SumGeometric (a : ℝ) (n : ℕ) : ℝ :=
  (List.range n).map (GeometricSequence a) |>.sum

theorem geometric_sequence_sum_eight (a : ℝ) :
  SumGeometric a 4 = 1 → SumGeometric a 8 = 17 := by
  sorry

#check geometric_sequence_sum_eight

end NUMINAMATH_CALUDE_geometric_sequence_sum_eight_l1612_161208


namespace NUMINAMATH_CALUDE_sum_first_seven_primes_l1612_161282

def first_seven_primes : List Nat := [2, 3, 5, 7, 11, 13, 17]

theorem sum_first_seven_primes : 
  (first_seven_primes.sum = 58) ∧ (∀ p ∈ first_seven_primes, Nat.Prime p) ∧ (first_seven_primes.length = 7) := by
  sorry

end NUMINAMATH_CALUDE_sum_first_seven_primes_l1612_161282


namespace NUMINAMATH_CALUDE_time_from_velocity_and_displacement_l1612_161276

/-- Given the equations for velocity and displacement, prove the formula for time. -/
theorem time_from_velocity_and_displacement
  (g V V₀ S S₀ a t : ℝ)
  (hV : V = g * (t - a) + V₀)
  (hS : S = (1/2) * g * (t - a)^2 + V₀ * (t - a) + S₀) :
  t = a + (V - V₀) / g :=
sorry

end NUMINAMATH_CALUDE_time_from_velocity_and_displacement_l1612_161276


namespace NUMINAMATH_CALUDE_sausage_problem_l1612_161285

/-- Represents the sausage problem --/
theorem sausage_problem (total_meat : ℕ) (total_links : ℕ) (remaining_meat : ℕ) 
  (h1 : total_meat = 10) 
  (h2 : total_links = 40)
  (h3 : remaining_meat = 112) : 
  (total_meat * 16 - remaining_meat) / (total_meat * 16 / total_links) = 12 := by
  sorry

#check sausage_problem

end NUMINAMATH_CALUDE_sausage_problem_l1612_161285


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l1612_161249

theorem polynomial_divisibility (m : ℝ) : 
  (∀ x : ℝ, (3 * x^2 - 5 * x + m) % (x - 2) = 0) → m = -2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l1612_161249


namespace NUMINAMATH_CALUDE_lcm_of_9_12_15_l1612_161225

theorem lcm_of_9_12_15 : Nat.lcm 9 (Nat.lcm 12 15) = 180 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_9_12_15_l1612_161225


namespace NUMINAMATH_CALUDE_line_through_points_l1612_161209

/-- Given two intersecting lines and their intersection point, prove the equation of the line passing through specific points. -/
theorem line_through_points (A₁ B₁ A₂ B₂ : ℝ) :
  (2 * A₁ + 3 * B₁ = 1) →  -- l₁ passes through P(2, 3)
  (2 * A₂ + 3 * B₂ = 1) →  -- l₂ passes through P(2, 3)
  (∀ x y : ℝ, A₁ * x + B₁ * y = 1 → 2 * x + 3 * y = 1) →  -- l₁ equation
  (∀ x y : ℝ, A₂ * x + B₂ * y = 1 → 2 * x + 3 * y = 1) →  -- l₂ equation
  ∀ x y : ℝ, (y - B₁) * (A₂ - A₁) = (x - A₁) * (B₂ - B₁) → 2 * x + 3 * y = 1 :=
by sorry


end NUMINAMATH_CALUDE_line_through_points_l1612_161209


namespace NUMINAMATH_CALUDE_meeting_point_theorem_l1612_161215

/-- Represents the circular path and the walkers' characteristics -/
structure CircularPath where
  totalBlocks : ℕ
  janeSpeedMultiplier : ℕ

/-- Represents the distance walked by each person when they meet -/
structure MeetingPoint where
  hectorDistance : ℕ
  janeDistance : ℕ

/-- Calculates the meeting point given a circular path -/
def calculateMeetingPoint (path : CircularPath) : MeetingPoint :=
  sorry

/-- Theorem stating that Hector walks 6 blocks when they meet -/
theorem meeting_point_theorem (path : CircularPath) 
  (h1 : path.totalBlocks = 24)
  (h2 : path.janeSpeedMultiplier = 3) :
  (calculateMeetingPoint path).hectorDistance = 6 :=
  sorry

end NUMINAMATH_CALUDE_meeting_point_theorem_l1612_161215


namespace NUMINAMATH_CALUDE_function_is_2x_l1612_161278

def is_linear (f : ℝ → ℝ) : Prop := ∃ a b : ℝ, ∀ x, f x = a * x + b

theorem function_is_2x (f : ℝ → ℝ) 
  (h₁ : f (-1) = -2)
  (h₂ : f 0 = 0)
  (h₃ : f 1 = 2)
  (h₄ : f 2 = 4) :
  ∀ x, f x = 2 * x := by
sorry

end NUMINAMATH_CALUDE_function_is_2x_l1612_161278


namespace NUMINAMATH_CALUDE_train_speed_l1612_161283

/-- Proves that a train of given length crossing a bridge of given length in a given time travels at a specific speed. -/
theorem train_speed (train_length bridge_length : ℝ) (crossing_time : ℝ) :
  train_length = 120 →
  bridge_length = 255 →
  crossing_time = 30 →
  (train_length + bridge_length) / crossing_time * 3.6 = 45 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l1612_161283


namespace NUMINAMATH_CALUDE_number_in_interval_l1612_161284

theorem number_in_interval (y : ℝ) (h : y = (1/y) * (-y) + 5) : 2 < y ∧ y ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_number_in_interval_l1612_161284


namespace NUMINAMATH_CALUDE_hank_bake_sale_earnings_l1612_161203

/-- Prove that Hank made $80 in the bake sale given the conditions of his fundraising activities. -/
theorem hank_bake_sale_earnings :
  let carwash_earnings : ℚ := 100
  let carwash_donation_rate : ℚ := 90 / 100
  let bake_sale_donation_rate : ℚ := 75 / 100
  let lawn_mowing_earnings : ℚ := 50
  let lawn_mowing_donation_rate : ℚ := 1
  let total_donation : ℚ := 200
  ∃ bake_sale_earnings : ℚ,
    bake_sale_earnings * bake_sale_donation_rate +
    carwash_earnings * carwash_donation_rate +
    lawn_mowing_earnings * lawn_mowing_donation_rate = total_donation ∧
    bake_sale_earnings = 80 :=
by sorry

end NUMINAMATH_CALUDE_hank_bake_sale_earnings_l1612_161203


namespace NUMINAMATH_CALUDE_vector_problem_l1612_161267

def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (-3, 2)

def perpendicular (v w : ℝ × ℝ) : Prop :=
  v.1 * w.1 + v.2 * w.2 = 0

def parallel (v w : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, v = t • w

theorem vector_problem :
  (∃ k : ℝ, perpendicular (k • a + b) (a - 3 • b) ∧ k = 19) ∧
  (∃ k : ℝ, parallel (k • a + b) (a - 3 • b) ∧ k = -1/3) :=
sorry

end NUMINAMATH_CALUDE_vector_problem_l1612_161267


namespace NUMINAMATH_CALUDE_division_of_squares_l1612_161251

theorem division_of_squares (a : ℝ) : 2 * a^2 / a^2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_division_of_squares_l1612_161251


namespace NUMINAMATH_CALUDE_g_symmetric_to_f_max_value_of_a_l1612_161250

-- Define the function f
def f (x : ℝ) : ℝ := x^2 + 4*x + 3

-- Define the function g (to be proved)
def g (x : ℝ) : ℝ := x^2 - 8*x + 15

-- Theorem 1: Prove that g is symmetric to f about x=1
theorem g_symmetric_to_f : ∀ x : ℝ, g x = f (2 - x) := by sorry

-- Theorem 2: Prove the maximum value of a
theorem max_value_of_a : 
  (∀ x : ℝ, g x ≥ g 6 - 4) ∧ 
  (∀ a : ℝ, (∀ x : ℝ, g x ≥ g a - 4) → a ≤ 6) := by sorry

end NUMINAMATH_CALUDE_g_symmetric_to_f_max_value_of_a_l1612_161250


namespace NUMINAMATH_CALUDE_rectangle_area_l1612_161258

theorem rectangle_area (d : ℝ) (w : ℝ) (h : w > 0) : 
  (3 * w)^2 + w^2 = d^2 → 3 * w^2 = 3 * d^2 / 10 :=
sorry

end NUMINAMATH_CALUDE_rectangle_area_l1612_161258


namespace NUMINAMATH_CALUDE_equal_celsius_fahrenheit_temp_l1612_161226

/-- Converts Celsius temperature to Fahrenheit -/
def celsius_to_fahrenheit (c : ℝ) : ℝ := 1.8 * c + 32

/-- Theorem stating that there exists a unique temperature where Celsius and Fahrenheit are equal -/
theorem equal_celsius_fahrenheit_temp :
  ∃! t : ℝ, t = celsius_to_fahrenheit t :=
by
  sorry

end NUMINAMATH_CALUDE_equal_celsius_fahrenheit_temp_l1612_161226


namespace NUMINAMATH_CALUDE_infinite_sum_reciprocal_squared_plus_two_l1612_161206

/-- The infinite sum of 1/(n^2(n+2)) from n=1 to infinity is equal to π^2/12 -/
theorem infinite_sum_reciprocal_squared_plus_two : 
  ∑' (n : ℕ), 1 / (n^2 * (n + 2 : ℝ)) = π^2 / 12 := by sorry

end NUMINAMATH_CALUDE_infinite_sum_reciprocal_squared_plus_two_l1612_161206


namespace NUMINAMATH_CALUDE_percent_relation_l1612_161264

theorem percent_relation (x y z : ℝ) 
  (h1 : 0.45 * z = 0.39 * y) 
  (h2 : y = 0.75 * x) : 
  z = 0.65 * x := by
  sorry

end NUMINAMATH_CALUDE_percent_relation_l1612_161264


namespace NUMINAMATH_CALUDE_total_practice_time_is_135_l1612_161222

/-- The number of minutes Daniel practices basketball each day during the school week -/
def school_day_practice : ℕ := 15

/-- The number of days in a school week -/
def school_week_days : ℕ := 5

/-- The number of days in a weekend -/
def weekend_days : ℕ := 2

/-- The total number of minutes Daniel practices during a whole week -/
def total_practice_time : ℕ :=
  (school_day_practice * school_week_days) +
  (2 * school_day_practice * weekend_days)

theorem total_practice_time_is_135 :
  total_practice_time = 135 := by
  sorry

end NUMINAMATH_CALUDE_total_practice_time_is_135_l1612_161222


namespace NUMINAMATH_CALUDE_z_profit_share_l1612_161236

/-- Calculates the share of profit for a partner in a business --/
def calculate_profit_share (
  x_capital y_capital z_capital : ℕ)  -- Initial capitals
  (x_months y_months z_months : ℕ)    -- Months of investment
  (total_profit : ℕ)                  -- Total annual profit
  : ℕ :=
  let x_share := x_capital * x_months
  let y_share := y_capital * y_months
  let z_share := z_capital * z_months
  let total_share := x_share + y_share + z_share
  (z_share * total_profit) / total_share

/-- Theorem statement for Z's profit share --/
theorem z_profit_share :
  calculate_profit_share 20000 25000 30000 12 12 7 50000 = 14000 := by
  sorry

end NUMINAMATH_CALUDE_z_profit_share_l1612_161236


namespace NUMINAMATH_CALUDE_polynomial_root_relation_l1612_161299

theorem polynomial_root_relation (p q r s : ℝ) (h_p : p ≠ 0) :
  (p * (4 : ℝ)^3 + q * (4 : ℝ)^2 + r * (4 : ℝ) + s = 0) →
  (p * (-3 : ℝ)^3 + q * (-3 : ℝ)^2 + r * (-3 : ℝ) + s = 0) →
  (q + r) / p = -13 := by
sorry

end NUMINAMATH_CALUDE_polynomial_root_relation_l1612_161299


namespace NUMINAMATH_CALUDE_sum_p_q_r_l1612_161220

/-- The largest real solution to the given equation -/
noncomputable def n : ℝ := 
  Real.sqrt (53 + Real.sqrt 249) + 13

/-- The equation that n satisfies -/
axiom n_eq : (4 / (n - 4)) + (6 / (n - 6)) + (18 / (n - 18)) + (20 / (n - 20)) = n^2 - 13*n - 6

/-- The existence of positive integers p, q, and r -/
axiom exists_p_q_r : ∃ (p q r : ℕ+), n = p + Real.sqrt (q + Real.sqrt r)

/-- The theorem to be proved -/
theorem sum_p_q_r : ∃ (p q r : ℕ+), n = p + Real.sqrt (q + Real.sqrt r) ∧ p + q + r = 315 := by
  sorry

end NUMINAMATH_CALUDE_sum_p_q_r_l1612_161220


namespace NUMINAMATH_CALUDE_dice_sum_pigeonhole_l1612_161238

/-- Represents a fair six-sided die -/
def Die := Fin 6

/-- Represents the sum of four dice rolls -/
def DiceSum := Fin 21

/-- The minimum number of throws required to guarantee a repeated sum -/
def minThrows : Nat := 22

theorem dice_sum_pigeonhole :
  ∀ (rolls : Fin minThrows → DiceSum),
  ∃ (i j : Fin minThrows), i ≠ j ∧ rolls i = rolls j :=
sorry

end NUMINAMATH_CALUDE_dice_sum_pigeonhole_l1612_161238


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l1612_161293

/-- Given a hyperbola with equation 9x^2 - 4y^2 = -36, 
    its asymptotes are y = ±(3/2)(-ix) -/
theorem hyperbola_asymptotes :
  ∀ (x y : ℂ), 9 * x^2 - 4 * y^2 = -36 →
  ∃ (k : ℂ), k = (3 / 2) * Complex.I ∧
  (y = k * x ∨ y = -k * x) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l1612_161293


namespace NUMINAMATH_CALUDE_fraction_integer_iff_q_in_set_l1612_161296

theorem fraction_integer_iff_q_in_set (q : ℕ+) :
  (∃ (k : ℕ+), (5 * q + 35 : ℤ) = k * (3 * q - 7)) ↔ 
  q ∈ ({3, 4, 5, 7, 9, 15, 21, 31} : Set ℕ+) := by
  sorry

end NUMINAMATH_CALUDE_fraction_integer_iff_q_in_set_l1612_161296


namespace NUMINAMATH_CALUDE_system_solution_l1612_161221

theorem system_solution (x y a : ℝ) : 
  3 * x + y = a → 
  2 * x + 5 * y = 2 * a → 
  x = 3 → 
  a = 13 := by sorry

end NUMINAMATH_CALUDE_system_solution_l1612_161221


namespace NUMINAMATH_CALUDE_base_seven_digits_of_956_l1612_161292

theorem base_seven_digits_of_956 : ∃ n : ℕ, (7^(n-1) ≤ 956 ∧ 956 < 7^n) ∧ n = 4 := by
  sorry

end NUMINAMATH_CALUDE_base_seven_digits_of_956_l1612_161292


namespace NUMINAMATH_CALUDE_opposite_of_negative_2023_l1612_161291

def opposite (x : ℤ) : ℤ := -x

theorem opposite_of_negative_2023 : opposite (-2023) = 2023 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_negative_2023_l1612_161291


namespace NUMINAMATH_CALUDE_greatest_prime_factor_sum_even_products_l1612_161202

def double_factorial (n : ℕ) : ℕ :=
  if n ≤ 1 then 1 else n * double_factorial (n - 2)

def even_product (n : ℕ) : ℕ :=
  double_factorial (2 * (n / 2))

theorem greatest_prime_factor_sum_even_products :
  ∃ (p : ℕ), p.Prime ∧ p = 23 ∧
  ∀ (q : ℕ), q.Prime → q ∣ (even_product 22 + even_product 20) → q ≤ p :=
by sorry

end NUMINAMATH_CALUDE_greatest_prime_factor_sum_even_products_l1612_161202


namespace NUMINAMATH_CALUDE_expression_not_simplifiable_l1612_161211

theorem expression_not_simplifiable (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) 
  (h : a + 2*b + 2*c = 0) : 
  ∃ (f : ℝ → ℝ → ℝ → ℝ), f a b c = 
    (1 / (b^2 + c^2 - a^2)) + (1 / (a^2 + c^2 - b^2)) + (1 / (a^2 + b^2 - c^2)) ∧
    ∀ (g : ℝ → ℝ), (∀ x y z, f x y z = g (f x y z)) → g = id := by
  sorry

end NUMINAMATH_CALUDE_expression_not_simplifiable_l1612_161211


namespace NUMINAMATH_CALUDE_radius_of_inscribed_circle_in_curvilinear_triangle_l1612_161207

/-- 
Given a rhombus with height h and acute angle α, and two inscribed circles:
1. One circle inscribed in the rhombus
2. Another circle inscribed in the curvilinear triangle formed by the rhombus and the first circle

This theorem states that the radius r of the second circle (inscribed in the curvilinear triangle)
is equal to (h/2) * tan²(45° - α/4)
-/
theorem radius_of_inscribed_circle_in_curvilinear_triangle 
  (h : ℝ) (α : ℝ) (h_pos : h > 0) (α_acute : 0 < α ∧ α < π/2) :
  ∃ r : ℝ, r = (h/2) * (Real.tan (π/4 - α/4))^2 ∧ 
  r > 0 ∧ 
  r < h/2 := by
sorry

end NUMINAMATH_CALUDE_radius_of_inscribed_circle_in_curvilinear_triangle_l1612_161207


namespace NUMINAMATH_CALUDE_max_value_cos_sin_l1612_161204

theorem max_value_cos_sin (θ : Real) (h : -π/2 < θ ∧ θ < π/2) :
  ∃ (M : Real), M = Real.sqrt 2 ∧ 
  ∀ θ', -π/2 < θ' ∧ θ' < π/2 → 
    Real.cos (θ'/2) * (1 + Real.sin θ') ≤ M :=
by sorry

end NUMINAMATH_CALUDE_max_value_cos_sin_l1612_161204


namespace NUMINAMATH_CALUDE_permutation_5_2_combination_6_3_plus_6_4_l1612_161270

-- Define permutation function
def A (n k : ℕ) : ℕ := sorry

-- Define combination function
def C (n k : ℕ) : ℕ := sorry

-- Theorem for A_5^2
theorem permutation_5_2 : A 5 2 = 20 := by sorry

-- Theorem for C_6^3 + C_6^4
theorem combination_6_3_plus_6_4 : C 6 3 + C 6 4 = 35 := by sorry

end NUMINAMATH_CALUDE_permutation_5_2_combination_6_3_plus_6_4_l1612_161270


namespace NUMINAMATH_CALUDE_equation_solution_l1612_161274

theorem equation_solution : 
  ∃ x : ℚ, (1 : ℚ) / 3 + 1 / x = 7 / 12 ∧ x = 4 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1612_161274


namespace NUMINAMATH_CALUDE_game_a_vs_game_b_l1612_161266

def p_heads : ℚ := 2/3
def p_tails : ℚ := 1/3

def p_win_game_a : ℚ := p_heads^3 + p_tails^3

def p_same_pair : ℚ := p_heads^2 + p_tails^2
def p_win_game_b : ℚ := p_same_pair^2

theorem game_a_vs_game_b : p_win_game_a - p_win_game_b = 2/81 := by
  sorry

end NUMINAMATH_CALUDE_game_a_vs_game_b_l1612_161266


namespace NUMINAMATH_CALUDE_discount_difference_l1612_161289

def initial_amount : ℝ := 12000

def apply_discount (amount : ℝ) (discount : ℝ) : ℝ :=
  amount * (1 - discount)

def scheme1 (amount : ℝ) : ℝ :=
  apply_discount (apply_discount (apply_discount amount 0.25) 0.15) 0.10

def scheme2 (amount : ℝ) : ℝ :=
  apply_discount (apply_discount (apply_discount amount 0.30) 0.10) 0.05

theorem discount_difference :
  scheme1 initial_amount - scheme2 initial_amount = 297 := by
  sorry

end NUMINAMATH_CALUDE_discount_difference_l1612_161289


namespace NUMINAMATH_CALUDE_nancys_apples_calculation_l1612_161235

/-- The number of apples Nancy ate -/
def nancys_apples : ℝ := 3.0

/-- The number of apples Mike picked -/
def mikes_apples : ℝ := 7.0

/-- The number of apples Keith picked -/
def keiths_apples : ℝ := 6.0

/-- The number of apples left -/
def apples_left : ℝ := 10.0

/-- Theorem: Nancy's apples equals the total picked by Mike and Keith minus the apples left -/
theorem nancys_apples_calculation : 
  nancys_apples = mikes_apples + keiths_apples - apples_left := by
  sorry

end NUMINAMATH_CALUDE_nancys_apples_calculation_l1612_161235


namespace NUMINAMATH_CALUDE_cubic_roots_sum_l1612_161241

theorem cubic_roots_sum (p q r : ℝ) : 
  (3 * p^3 - 4 * p^2 + 200 * p - 5 = 0) →
  (3 * q^3 - 4 * q^2 + 200 * q - 5 = 0) →
  (3 * r^3 - 4 * r^2 + 200 * r - 5 = 0) →
  (p + q - 2)^3 + (q + r - 2)^3 + (r + p - 2)^3 = 184/9 := by
sorry

end NUMINAMATH_CALUDE_cubic_roots_sum_l1612_161241


namespace NUMINAMATH_CALUDE_valid_solutions_characterization_l1612_161229

/-- A number is a valid solution if it's a four-digit number,
    divisible by 28, and can be expressed as the sum of squares
    of three consecutive even numbers. -/
def is_valid_solution (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999 ∧
  n % 28 = 0 ∧
  ∃ k : ℕ, n = 12 * k^2 + 8

/-- The set of all valid solutions -/
def solution_set : Set ℕ := {1736, 3080, 4340, 6356, 8120}

/-- Theorem stating that the solution_set contains exactly
    the numbers satisfying is_valid_solution -/
theorem valid_solutions_characterization :
  ∀ n : ℕ, is_valid_solution n ↔ n ∈ solution_set :=
by sorry

#check valid_solutions_characterization

end NUMINAMATH_CALUDE_valid_solutions_characterization_l1612_161229


namespace NUMINAMATH_CALUDE_circle_tangent_sum_radii_l1612_161253

theorem circle_tangent_sum_radii : ∀ r : ℝ,
  (r > 0) →
  ((r - 5)^2 + r^2 = (r + 2)^2) →
  (∃ r₁ r₂ : ℝ, (r = r₁ ∨ r = r₂) ∧ r₁ + r₂ = 14) :=
by sorry

end NUMINAMATH_CALUDE_circle_tangent_sum_radii_l1612_161253


namespace NUMINAMATH_CALUDE_log_base_5_inequality_l1612_161260

theorem log_base_5_inequality (x : ℝ) (h1 : 0 < x) (h2 : Real.log x / Real.log 5 < 1) : 1 < x ∧ x < 5 := by
  sorry

end NUMINAMATH_CALUDE_log_base_5_inequality_l1612_161260


namespace NUMINAMATH_CALUDE_smaller_number_problem_l1612_161277

theorem smaller_number_problem (x y : ℝ) 
  (sum_eq : x + y = 18) 
  (diff_eq : x - y = 8) : 
  min x y = 5 := by sorry

end NUMINAMATH_CALUDE_smaller_number_problem_l1612_161277


namespace NUMINAMATH_CALUDE_sample_variance_estimates_stability_l1612_161219

-- Define the type for sample statistics
inductive SampleStatistic
  | Mean
  | Median
  | Variance
  | Maximum

-- Define a function that determines if a statistic estimates population stability
def estimatesStability (stat : SampleStatistic) : Prop :=
  match stat with
  | SampleStatistic.Variance => True
  | _ => False

-- Theorem statement
theorem sample_variance_estimates_stability :
  ∃ (stat : SampleStatistic), estimatesStability stat ∧
  (stat = SampleStatistic.Mean ∨
   stat = SampleStatistic.Median ∨
   stat = SampleStatistic.Variance ∨
   stat = SampleStatistic.Maximum) :=
by
  sorry

end NUMINAMATH_CALUDE_sample_variance_estimates_stability_l1612_161219


namespace NUMINAMATH_CALUDE_sun_division_l1612_161273

theorem sun_division (x y z : ℝ) (total : ℝ) : 
  (y = 0.45 * x) →
  (z = 0.50 * x) →
  (y = 63) →
  (total = x + y + z) →
  total = 273 := by
sorry

end NUMINAMATH_CALUDE_sun_division_l1612_161273
