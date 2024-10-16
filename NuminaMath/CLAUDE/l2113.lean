import Mathlib

namespace NUMINAMATH_CALUDE_projection_theorem_l2113_211327

-- Define a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- State the projection theorem
theorem projection_theorem (t : Triangle) : 
  t.a = t.b * Real.cos t.C + t.c * Real.cos t.B ∧
  t.b = t.c * Real.cos t.A + t.a * Real.cos t.C ∧
  t.c = t.a * Real.cos t.B + t.b * Real.cos t.A :=
by sorry

end NUMINAMATH_CALUDE_projection_theorem_l2113_211327


namespace NUMINAMATH_CALUDE_first_player_can_ensure_non_trivial_solution_l2113_211330

-- Define the system of equations
structure LinearSystem :=
  (eq1 eq2 eq3 : ℝ → ℝ → ℝ → ℝ)

-- Define the game state
structure GameState :=
  (system : LinearSystem)
  (player_turn : Bool)

-- Define a strategy for the first player
def FirstPlayerStrategy : GameState → GameState := sorry

-- Define a strategy for the second player
def SecondPlayerStrategy : GameState → GameState := sorry

-- Theorem statement
theorem first_player_can_ensure_non_trivial_solution :
  ∀ (initial_state : GameState),
  ∃ (x y z : ℝ), 
    (x ≠ 0 ∨ y ≠ 0 ∨ z ≠ 0) ∧
    (initial_state.system.eq1 x y z = 0) ∧
    (initial_state.system.eq2 x y z = 0) ∧
    (initial_state.system.eq3 x y z = 0) :=
sorry

end NUMINAMATH_CALUDE_first_player_can_ensure_non_trivial_solution_l2113_211330


namespace NUMINAMATH_CALUDE_pythagorean_diagonal_l2113_211301

theorem pythagorean_diagonal (m : ℕ) (h_m : m ≥ 3) : 
  let width : ℕ := 2 * m
  let diagonal : ℕ := m^2 + 1
  let height : ℕ := diagonal - 2
  (width : ℤ)^2 + height^2 = diagonal^2 := by sorry

end NUMINAMATH_CALUDE_pythagorean_diagonal_l2113_211301


namespace NUMINAMATH_CALUDE_circle_symmetry_l2113_211357

-- Define the original circle C
def C (x y : ℝ) : Prop := x^2 + y^2 - 6*x + 8*y + 24 = 0

-- Define the line l
def l (x y : ℝ) : Prop := x - 3*y - 5 = 0

-- Define the symmetric circle S
def S (x y : ℝ) : Prop := (x - 1)^2 + (y - 2)^2 = 1

-- Theorem statement
theorem circle_symmetry :
  ∀ (x y : ℝ), S x y ↔ ∃ (x' y' : ℝ), C x' y' ∧
  (∃ (m : ℝ), l m ((y + y')/2) ∧ m = (x + x')/2) ∧
  ((y - y')/(x - x') = -3 ∨ x = x') :=
sorry

end NUMINAMATH_CALUDE_circle_symmetry_l2113_211357


namespace NUMINAMATH_CALUDE_solve_for_k_l2113_211346

-- Define the system of equations
def system (x y k : ℝ) : Prop :=
  (2 * x + y = 4 * k) ∧ (x - y = k)

-- Define the additional equation
def additional_eq (x y : ℝ) : Prop :=
  x + 2 * y = 12

-- Theorem statement
theorem solve_for_k :
  ∀ x y k : ℝ, system x y k → additional_eq x y → k = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_solve_for_k_l2113_211346


namespace NUMINAMATH_CALUDE_triumphal_arch_proportion_l2113_211352

/-- Represents the number of photographs of each type of attraction -/
structure Photos where
  cathedrals : ℕ
  arches : ℕ
  waterfalls : ℕ
  castles : ℕ

/-- Represents the total number of each type of attraction seen -/
structure Attractions where
  cathedrals : ℕ
  arches : ℕ
  waterfalls : ℕ
  castles : ℕ

/-- The main theorem stating the proportion of photographs featuring triumphal arches -/
theorem triumphal_arch_proportion
  (p : Photos) (a : Attractions)
  (half_photographed : p.cathedrals + p.arches + p.waterfalls + p.castles = (a.cathedrals + a.arches + a.waterfalls + a.castles) / 2)
  (cathedral_arch_ratio : a.cathedrals = 3 * a.arches)
  (castle_waterfall_equal : a.castles = a.waterfalls)
  (quarter_castles : 4 * p.castles = p.cathedrals + p.arches + p.waterfalls + p.castles)
  (half_castles_photographed : 2 * p.castles = a.castles)
  (all_arches_photographed : p.arches = a.arches) :
  4 * p.arches = p.cathedrals + p.arches + p.waterfalls + p.castles :=
by sorry

end NUMINAMATH_CALUDE_triumphal_arch_proportion_l2113_211352


namespace NUMINAMATH_CALUDE_rectangular_garden_area_l2113_211342

/-- A rectangular garden with length three times its width and width of 15 meters has an area of 675 square meters. -/
theorem rectangular_garden_area :
  ∀ (length width area : ℝ),
  length = 3 * width →
  width = 15 →
  area = length * width →
  area = 675 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_garden_area_l2113_211342


namespace NUMINAMATH_CALUDE_circumscribed_circle_radius_l2113_211360

/-- The radius of the circumscribed circle of a triangle with side lengths 3, 5, and 7 is 7√3/3 -/
theorem circumscribed_circle_radius (a b c : ℝ) (h_a : a = 3) (h_b : b = 5) (h_c : c = 7) :
  let R := c / (2 * Real.sqrt (1 - ((a^2 + b^2 - c^2) / (2 * a * b))^2))
  R = 7 * Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_circumscribed_circle_radius_l2113_211360


namespace NUMINAMATH_CALUDE_intersection_empty_implies_a_greater_than_neg_four_l2113_211392

def A (a : ℝ) : Set ℝ := {x | x^2 + (a + 2) * x + 1 = 0}
def B : Set ℝ := {x | x > 0}

theorem intersection_empty_implies_a_greater_than_neg_four (a : ℝ) :
  A a ∩ B = ∅ → a > -4 := by sorry

end NUMINAMATH_CALUDE_intersection_empty_implies_a_greater_than_neg_four_l2113_211392


namespace NUMINAMATH_CALUDE_degree_three_iff_c_eq_neg_seven_fifteenths_l2113_211380

/-- The polynomial f(x) -/
def f (x : ℝ) : ℝ := 1 - 12*x + 3*x^2 - 4*x^3 + 7*x^4

/-- The polynomial g(x) -/
def g (x : ℝ) : ℝ := 5 - 2*x - 6*x^3 + 15*x^4

/-- The combined polynomial h(x, c) = f(x) + c*g(x) -/
def h (x c : ℝ) : ℝ := f x + c * g x

/-- Theorem stating that h(x, c) has degree 3 if and only if c = -7/15 -/
theorem degree_three_iff_c_eq_neg_seven_fifteenths :
  (∀ x, h x (-7/15) = 1 - 12*x + 3*x^2 - 6/5*x^3) ∧
  (∀ c, (∀ x, h x c = 1 - 12*x + 3*x^2 - 6/5*x^3) → c = -7/15) :=
by sorry

end NUMINAMATH_CALUDE_degree_three_iff_c_eq_neg_seven_fifteenths_l2113_211380


namespace NUMINAMATH_CALUDE_two_language_speakers_l2113_211391

/-- Represents the number of students who can speak a given language -/
structure LanguageSpeakers where
  gujarati : ℕ
  hindi : ℕ
  marathi : ℕ

/-- Represents the number of students who can speak exactly two languages -/
structure BilingualStudents where
  gujarati_hindi : ℕ
  gujarati_marathi : ℕ
  hindi_marathi : ℕ

/-- The theorem to be proved -/
theorem two_language_speakers
  (total_students : ℕ)
  (speakers : LanguageSpeakers)
  (trilingual : ℕ)
  (h_total : total_students = 22)
  (h_gujarati : speakers.gujarati = 6)
  (h_hindi : speakers.hindi = 15)
  (h_marathi : speakers.marathi = 6)
  (h_trilingual : trilingual = 1)
  : ∃ (bilingual : BilingualStudents),
    bilingual.gujarati_hindi + bilingual.gujarati_marathi + bilingual.hindi_marathi = 6 ∧
    total_students = speakers.gujarati + speakers.hindi + speakers.marathi -
      (bilingual.gujarati_hindi + bilingual.gujarati_marathi + bilingual.hindi_marathi) +
      trilingual :=
by sorry

end NUMINAMATH_CALUDE_two_language_speakers_l2113_211391


namespace NUMINAMATH_CALUDE_tower_height_range_l2113_211307

theorem tower_height_range (h : ℝ) : 
  (¬(h ≥ 200)) ∧ (¬(h ≤ 150)) ∧ (¬(h ≤ 180)) → h ∈ Set.Ioo 180 200 := by
  sorry

end NUMINAMATH_CALUDE_tower_height_range_l2113_211307


namespace NUMINAMATH_CALUDE_collinear_points_l2113_211367

theorem collinear_points (k : ℝ) : 
  let PA : ℝ × ℝ := (k, 12)
  let PB : ℝ × ℝ := (4, 5)
  let PC : ℝ × ℝ := (10, k)
  (k = -2 ∨ k = 11) ↔ 
    ∃ (t : ℝ), (PC.1 - PA.1, PC.2 - PA.2) = t • (PB.1 - PA.1, PB.2 - PA.2) :=
by sorry

end NUMINAMATH_CALUDE_collinear_points_l2113_211367


namespace NUMINAMATH_CALUDE_intersection_of_M_and_S_l2113_211320

-- Define the set M
def M : Set ℕ := {x | 0 < x ∧ x < 4}

-- Define the set S
def S : Set ℕ := {2, 3, 5}

-- Theorem statement
theorem intersection_of_M_and_S : M ∩ S = {2, 3} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_S_l2113_211320


namespace NUMINAMATH_CALUDE_inequality_proof_l2113_211334

theorem inequality_proof (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a^2 + b^2 = 1/2) :
  (1/(1-a)) + (1/(1-b)) ≥ 4 ∧ ((1/(1-a)) + (1/(1-b)) = 4 ↔ a = 1/2 ∧ b = 1/2) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l2113_211334


namespace NUMINAMATH_CALUDE_simplify_expression_l2113_211322

theorem simplify_expression (x y : ℝ) :
  (3 * x^2 + 4 * x + 6 * y - 9) - (x^2 - 2 * x + 3 * y + 15) = 2 * x^2 + 6 * x + 3 * y - 24 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2113_211322


namespace NUMINAMATH_CALUDE_simple_interest_principal_l2113_211364

/-- Proves that given specific simple interest conditions, the principal amount is 2000 --/
theorem simple_interest_principal :
  ∀ (rate : ℚ) (interest : ℚ) (time : ℚ) (principal : ℚ),
    rate = 25/2 →
    interest = 500 →
    time = 2 →
    principal * rate * time / 100 = interest →
    principal = 2000 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_principal_l2113_211364


namespace NUMINAMATH_CALUDE_equation_solution_l2113_211365

theorem equation_solution :
  ∃ (x : ℚ), x ≠ -2 ∧ (x^2 + 2*x + 2) / (x + 2) = x + 3 ∧ x = -4/3 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2113_211365


namespace NUMINAMATH_CALUDE_sum_of_solutions_eq_19_12_l2113_211303

theorem sum_of_solutions_eq_19_12 : ∃ (x₁ x₂ : ℝ), 
  (4 * x₁ + 7) * (3 * x₁ - 10) = 0 ∧
  (4 * x₂ + 7) * (3 * x₂ - 10) = 0 ∧
  x₁ ≠ x₂ ∧
  x₁ + x₂ = 19 / 12 := by
sorry

end NUMINAMATH_CALUDE_sum_of_solutions_eq_19_12_l2113_211303


namespace NUMINAMATH_CALUDE_triangle_angle_x_l2113_211344

theorem triangle_angle_x (x : ℝ) : 
  x > 0 ∧ 3*x > 0 ∧ 40 > 0 ∧ 
  x + 3*x + 40 = 180 → 
  x = 35 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_x_l2113_211344


namespace NUMINAMATH_CALUDE_recurring_decimal_fraction_sum_l2113_211378

theorem recurring_decimal_fraction_sum (a b : ℕ+) :
  (a.val : ℚ) / (b.val : ℚ) = 36 / 99 →
  Nat.gcd a.val b.val = 1 →
  a.val + b.val = 15 := by
  sorry

end NUMINAMATH_CALUDE_recurring_decimal_fraction_sum_l2113_211378


namespace NUMINAMATH_CALUDE_meal_combinations_count_l2113_211304

/-- The number of items on the menu -/
def menu_items : ℕ := 12

/-- The number of dishes each person orders -/
def dishes_per_person : ℕ := 1

/-- The number of special dishes shared -/
def shared_special_dishes : ℕ := 1

/-- The number of remaining dishes after choosing the special dish -/
def remaining_dishes : ℕ := menu_items - shared_special_dishes

/-- The number of different meal combinations for Yann and Camille -/
def meal_combinations : ℕ := remaining_dishes * remaining_dishes

theorem meal_combinations_count : meal_combinations = 121 := by
  sorry

end NUMINAMATH_CALUDE_meal_combinations_count_l2113_211304


namespace NUMINAMATH_CALUDE_horner_v1_at_negative_two_l2113_211386

def horner_polynomial (x : ℝ) : ℝ := x^6 - 5*x^5 + 6*x^4 + x^2 + 0.3*x + 2

def horner_v0 : ℝ := 1

def horner_v1 (x : ℝ) : ℝ := horner_v0 * x - 5

theorem horner_v1_at_negative_two :
  horner_v1 (-2) = -7 :=
sorry

end NUMINAMATH_CALUDE_horner_v1_at_negative_two_l2113_211386


namespace NUMINAMATH_CALUDE_remaining_students_average_l2113_211372

theorem remaining_students_average (total_students : ℕ) (class_average : ℚ)
  (group1_fraction : ℚ) (group1_average : ℚ)
  (group2_fraction : ℚ) (group2_average : ℚ)
  (group3_fraction : ℚ) (group3_average : ℚ)
  (h1 : total_students = 120)
  (h2 : class_average = 84)
  (h3 : group1_fraction = 1/4)
  (h4 : group1_average = 96)
  (h5 : group2_fraction = 1/5)
  (h6 : group2_average = 75)
  (h7 : group3_fraction = 1/8)
  (h8 : group3_average = 90) :
  let remaining_students := total_students - (group1_fraction * total_students + group2_fraction * total_students + group3_fraction * total_students)
  let remaining_average := (total_students * class_average - (group1_fraction * total_students * group1_average + group2_fraction * total_students * group2_average + group3_fraction * total_students * group3_average)) / remaining_students
  remaining_average = 4050 / 51 := by
  sorry

end NUMINAMATH_CALUDE_remaining_students_average_l2113_211372


namespace NUMINAMATH_CALUDE_g_of_3_equals_12_l2113_211338

def g (x : ℝ) : ℝ := x^3 - 2*x^2 + x

theorem g_of_3_equals_12 : g 3 = 12 := by sorry

end NUMINAMATH_CALUDE_g_of_3_equals_12_l2113_211338


namespace NUMINAMATH_CALUDE_square_roots_sum_l2113_211347

theorem square_roots_sum (x y : ℝ) : 
  x^2 = 16 → y^2 = 9 → x^2 + y^2 + x + 2023 = 2052 := by
  sorry

end NUMINAMATH_CALUDE_square_roots_sum_l2113_211347


namespace NUMINAMATH_CALUDE_v_formation_sum_l2113_211350

def isValidDigit (d : ℕ) : Prop := d ≥ 1 ∧ d ≤ 9

def isDistinct (a b c d e : ℕ) : Prop :=
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧
  c ≠ d ∧ c ≠ e ∧
  d ≠ e

theorem v_formation_sum (a b c d e : ℕ)
  (h_valid : isValidDigit a ∧ isValidDigit b ∧ isValidDigit c ∧ isValidDigit d ∧ isValidDigit e)
  (h_distinct : isDistinct a b c d e)
  (h_left_sum : a + b + c = 16)
  (h_right_sum : c + d = 11) :
  a + b + c + d + e = 18 :=
sorry

end NUMINAMATH_CALUDE_v_formation_sum_l2113_211350


namespace NUMINAMATH_CALUDE_maggi_cupcakes_l2113_211351

/-- Proves that Maggi ate 0 cupcakes given the initial number of packages,
    cupcakes per package, and cupcakes left. -/
theorem maggi_cupcakes (initial_packages : ℕ) (cupcakes_per_package : ℕ) (cupcakes_left : ℕ)
    (h1 : initial_packages = 3)
    (h2 : cupcakes_per_package = 4)
    (h3 : cupcakes_left = 12) :
    initial_packages * cupcakes_per_package - cupcakes_left = 0 := by
  sorry

end NUMINAMATH_CALUDE_maggi_cupcakes_l2113_211351


namespace NUMINAMATH_CALUDE_cost_of_grapes_and_pineapple_l2113_211398

/-- Represents the price of fruits and their combinations -/
structure FruitPrices where
  f : ℚ  -- price of one piece of fruit
  g : ℚ  -- price of a bunch of grapes
  p : ℚ  -- price of a pineapple
  φ : ℚ  -- price of a pack of figs

/-- The conditions given in the problem -/
def satisfiesConditions (prices : FruitPrices) : Prop :=
  3 * prices.f + 2 * prices.g + prices.p + prices.φ = 36 ∧
  prices.φ = 3 * prices.f ∧
  prices.p = prices.f + prices.g

/-- The theorem to be proved -/
theorem cost_of_grapes_and_pineapple (prices : FruitPrices) 
  (h : satisfiesConditions prices) : 
  2 * prices.g + prices.p = (15 * prices.g + 36) / 7 := by
  sorry

end NUMINAMATH_CALUDE_cost_of_grapes_and_pineapple_l2113_211398


namespace NUMINAMATH_CALUDE_magic_square_g_value_l2113_211376

/-- Represents a 3x3 multiplicative magic square --/
structure MagicSquare where
  a : ℕ+
  b : ℕ+
  c : ℕ+
  d : ℕ+
  e : ℕ+
  f : ℕ+
  g : ℕ+
  h : ℕ+
  i : ℕ+
  row_product : a * b * c = d * e * f ∧ d * e * f = g * h * i
  col_product : a * d * g = b * e * h ∧ b * e * h = c * f * i
  diag_product : a * e * i = c * e * g

/-- The theorem stating that the only possible value for g is 3 --/
theorem magic_square_g_value (ms : MagicSquare) (h1 : ms.a = 90) (h2 : ms.i = 3) :
  ms.g = 3 :=
sorry

end NUMINAMATH_CALUDE_magic_square_g_value_l2113_211376


namespace NUMINAMATH_CALUDE_trig_expression_equals_negative_four_l2113_211309

theorem trig_expression_equals_negative_four :
  (Real.sqrt 3 / Real.cos (10 * π / 180)) - (1 / Real.sin (10 * π / 180)) = -4 := by
  sorry

end NUMINAMATH_CALUDE_trig_expression_equals_negative_four_l2113_211309


namespace NUMINAMATH_CALUDE_prob_red_or_black_is_three_fourths_prob_red_or_black_or_white_is_eleven_twelfths_l2113_211353

/-- Represents the color of a ball -/
inductive BallColor
  | Red
  | Black
  | White
  | Green

/-- Represents the box of balls -/
structure BallBox where
  total : ℕ
  red : ℕ
  black : ℕ
  white : ℕ
  green : ℕ
  sum_constraint : red + black + white + green = total

/-- Calculates the probability of drawing a ball of a specific color -/
def prob_color (box : BallBox) (color : BallColor) : ℚ :=
  match color with
  | BallColor.Red => box.red / box.total
  | BallColor.Black => box.black / box.total
  | BallColor.White => box.white / box.total
  | BallColor.Green => box.green / box.total

/-- The box described in the problem -/
def problem_box : BallBox :=
  { total := 12
    red := 5
    black := 4
    white := 2
    green := 1
    sum_constraint := by simp }

theorem prob_red_or_black_is_three_fourths :
    prob_color problem_box BallColor.Red + prob_color problem_box BallColor.Black = 3/4 := by
  sorry

theorem prob_red_or_black_or_white_is_eleven_twelfths :
    prob_color problem_box BallColor.Red + prob_color problem_box BallColor.Black +
    prob_color problem_box BallColor.White = 11/12 := by
  sorry

end NUMINAMATH_CALUDE_prob_red_or_black_is_three_fourths_prob_red_or_black_or_white_is_eleven_twelfths_l2113_211353


namespace NUMINAMATH_CALUDE_inverse_of_A_l2113_211300

def A : Matrix (Fin 2) (Fin 2) ℚ := !![2, 3; -1, 7]

theorem inverse_of_A :
  let A_inv : Matrix (Fin 2) (Fin 2) ℚ := !![7/17, -3/17; 1/17, 2/17]
  A * A_inv = 1 ∧ A_inv * A = 1 :=
sorry

end NUMINAMATH_CALUDE_inverse_of_A_l2113_211300


namespace NUMINAMATH_CALUDE_janet_tickets_l2113_211370

/-- The number of tickets needed for Janet's amusement park rides -/
def total_tickets (roller_coaster_tickets_per_ride : ℕ) 
                  (giant_slide_tickets_per_ride : ℕ) 
                  (roller_coaster_rides : ℕ) 
                  (giant_slide_rides : ℕ) : ℕ :=
  roller_coaster_tickets_per_ride * roller_coaster_rides + 
  giant_slide_tickets_per_ride * giant_slide_rides

/-- Theorem: Janet needs 47 tickets for her amusement park rides -/
theorem janet_tickets : 
  total_tickets 5 3 7 4 = 47 := by
  sorry

end NUMINAMATH_CALUDE_janet_tickets_l2113_211370


namespace NUMINAMATH_CALUDE_equation_solution_l2113_211368

theorem equation_solution : ∃ x : ℝ, (x + 6) / (x - 3) = 4 ∧ x = 6 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2113_211368


namespace NUMINAMATH_CALUDE_vector_subtraction_l2113_211345

/-- Given two vectors AB and AC in R², prove that CB = AB - AC -/
theorem vector_subtraction (AB AC : Fin 2 → ℝ) (h1 : AB = ![2, 3]) (h2 : AC = ![-1, 2]) :
  (fun i => AB i - AC i) = ![3, 1] := by
  sorry

end NUMINAMATH_CALUDE_vector_subtraction_l2113_211345


namespace NUMINAMATH_CALUDE_ratio_problem_l2113_211311

theorem ratio_problem (x : ℝ) (h : 0.75 / x = 5 / 8) : x = 1.2 := by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l2113_211311


namespace NUMINAMATH_CALUDE_evaluate_P_l2113_211354

-- Define the polynomial P(a)
def P (a : ℝ) : ℝ := (6 * a^2 - 14 * a + 5) * (3 * a - 4)

-- Theorem stating the values of P(4/3) and P(2)
theorem evaluate_P : P (4/3) = 0 ∧ P 2 = 2 := by sorry

end NUMINAMATH_CALUDE_evaluate_P_l2113_211354


namespace NUMINAMATH_CALUDE_sum_xyz_equals_2014_l2113_211383

theorem sum_xyz_equals_2014 (x y z : ℝ) : 
  Real.sqrt (x - 3) + Real.sqrt (3 - x) + abs (x - y + 2010) + z^2 + 4*z + 4 = 0 → 
  x + y + z = 2014 := by
  sorry

end NUMINAMATH_CALUDE_sum_xyz_equals_2014_l2113_211383


namespace NUMINAMATH_CALUDE_root_zero_implies_k_eq_one_l2113_211363

/-- The quadratic equation in x with coefficient k -/
def quadratic_equation (k x : ℝ) : ℝ :=
  (k + 2) * x^2 + 6 * x + k^2 + k - 2

/-- Theorem stating that if the quadratic equation has 0 as a root and k + 2 ≠ 0, then k = 1 -/
theorem root_zero_implies_k_eq_one (k : ℝ) :
  quadratic_equation k 0 = 0 → k + 2 ≠ 0 → k = 1 := by
  sorry

end NUMINAMATH_CALUDE_root_zero_implies_k_eq_one_l2113_211363


namespace NUMINAMATH_CALUDE_three_digit_geometric_progression_l2113_211316

theorem three_digit_geometric_progression :
  ∀ a b c : ℕ,
  100 ≤ 100 * a + 10 * b + c ∧ 100 * a + 10 * b + c < 1000 →
  (∃ r : ℚ,
    (100 * b + 10 * c + a : ℚ) = r * (100 * a + 10 * b + c : ℚ) ∧
    (100 * c + 10 * a + b : ℚ) = r * (100 * b + 10 * c + a : ℚ)) →
  ((a = b ∧ b = c ∧ 1 ≤ a ∧ a ≤ 9) ∨
   (a = 2 ∧ b = 4 ∧ c = 3) ∨
   (a = 4 ∧ b = 8 ∧ c = 6)) :=
by sorry

end NUMINAMATH_CALUDE_three_digit_geometric_progression_l2113_211316


namespace NUMINAMATH_CALUDE_total_driving_time_bound_l2113_211329

/-- Represents the driving scenario with given distances and speeds -/
structure DrivingScenario where
  distance_leg1 : ℝ
  time_leg1 : ℝ
  distance_leg2 : ℝ
  distance_leg3 : ℝ
  distance_leg4 : ℝ
  speed_leg2 : ℝ
  speed_leg3 : ℝ
  speed_leg4 : ℝ

/-- The total driving time does not exceed 10 hours -/
theorem total_driving_time_bound (scenario : DrivingScenario) 
  (h1 : scenario.distance_leg1 = 120)
  (h2 : scenario.time_leg1 = 3)
  (h3 : scenario.distance_leg2 = 60)
  (h4 : scenario.distance_leg3 = 90)
  (h5 : scenario.distance_leg4 = 200)
  (h6 : scenario.speed_leg2 > 0)
  (h7 : scenario.speed_leg3 > 0)
  (h8 : scenario.speed_leg4 > 0) :
  scenario.time_leg1 + 
  (scenario.distance_leg2 / scenario.speed_leg2) + 
  (scenario.distance_leg3 / scenario.speed_leg3) + 
  (scenario.distance_leg4 / scenario.speed_leg4) ≤ 10 := by
sorry

end NUMINAMATH_CALUDE_total_driving_time_bound_l2113_211329


namespace NUMINAMATH_CALUDE_bobs_age_problem_l2113_211366

theorem bobs_age_problem :
  ∃ (n : ℕ), 
    (∃ (k : ℕ), n - 3 = k^2) ∧ 
    (∃ (j : ℕ), n + 4 = j^3) ∧ 
    n = 725 := by
  sorry

end NUMINAMATH_CALUDE_bobs_age_problem_l2113_211366


namespace NUMINAMATH_CALUDE_circle_center_polar_coordinates_l2113_211394

/-- The polar coordinates of the center of the circle ρ = √2(cos θ + sin θ) are (1, π/4) -/
theorem circle_center_polar_coordinates :
  let ρ : ℝ → ℝ := λ θ => Real.sqrt 2 * (Real.cos θ + Real.sin θ)
  ∃ r θ_c, r = 1 ∧ θ_c = π/4 ∧
    ∀ θ, ρ θ = 2 * Real.cos (θ - θ_c) :=
by sorry

end NUMINAMATH_CALUDE_circle_center_polar_coordinates_l2113_211394


namespace NUMINAMATH_CALUDE_special_ellipse_equation_l2113_211321

/-- An ellipse with specific properties -/
structure SpecialEllipse where
  /-- The length of the major axis is twice the length of the minor axis -/
  major_twice_minor : ℝ → ℝ → Prop
  /-- The ellipse passes through the point (2, -6) -/
  passes_through_2_neg6 : ℝ → ℝ → Prop
  /-- The ellipse passes through the point (3, 0) -/
  passes_through_3_0 : ℝ → ℝ → Prop
  /-- The eccentricity of the ellipse is √6/3 -/
  eccentricity_sqrt6_div_3 : ℝ → ℝ → Prop

/-- The standard equation of an ellipse -/
def standard_equation (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

/-- Theorem stating that a SpecialEllipse satisfies one of two standard equations -/
theorem special_ellipse_equation (e : SpecialEllipse) :
  (∀ x y, standard_equation 3 (Real.sqrt 3) x y) ∨
  (∀ x y, standard_equation (Real.sqrt 27) 3 x y) :=
sorry

end NUMINAMATH_CALUDE_special_ellipse_equation_l2113_211321


namespace NUMINAMATH_CALUDE_unique_abc_solution_l2113_211356

theorem unique_abc_solution (a b c : ℕ+) 
  (h1 : a < b) (h2 : b < c) 
  (h3 : a * b + b * c + c * a = a * b * c) : 
  a = 2 ∧ b = 3 ∧ c = 6 := by
sorry

end NUMINAMATH_CALUDE_unique_abc_solution_l2113_211356


namespace NUMINAMATH_CALUDE_box_dimensions_theorem_l2113_211359

/-- Represents the dimensions of a rectangular box -/
structure BoxDimensions where
  a : ℕ
  b : ℕ
  c : ℕ

/-- Checks if the given dimensions satisfy the consecutive integer condition -/
def isConsecutive (d : BoxDimensions) : Prop :=
  d.b = d.a + 1 ∧ d.c = d.a + 2

/-- Calculates the volume of the box -/
def volume (d : BoxDimensions) : ℕ :=
  d.a * d.b * d.c

/-- Calculates the surface area of the box -/
def surfaceArea (d : BoxDimensions) : ℕ :=
  2 * (d.a * d.b + d.b * d.c + d.c * d.a)

/-- The main theorem stating the conditions and the result to be proved -/
theorem box_dimensions_theorem (d : BoxDimensions) :
    d.a < d.b ∧ d.b < d.c ∧
    isConsecutive d ∧
    2 * surfaceArea d = volume d →
    d.a = 8 := by
  sorry

end NUMINAMATH_CALUDE_box_dimensions_theorem_l2113_211359


namespace NUMINAMATH_CALUDE_x_value_l2113_211308

theorem x_value (x y : ℝ) (h1 : x - y = 10) (h2 : x + y = 8) : x = 9 := by
  sorry

end NUMINAMATH_CALUDE_x_value_l2113_211308


namespace NUMINAMATH_CALUDE_quadratic_always_positive_l2113_211331

-- Define the quadratic expression
def quadratic (k x : ℝ) : ℝ := x^2 - (k - 4)*x - k + 7

-- State the theorem
theorem quadratic_always_positive (k : ℝ) :
  (∀ x, quadratic k x > 0) ↔ k > -2 ∧ k < 6 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_always_positive_l2113_211331


namespace NUMINAMATH_CALUDE_intersection_nonempty_iff_a_in_range_union_equals_interval_iff_a_equals_two_l2113_211389

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | 2 < x ∧ x < 4}
def B (a : ℝ) : Set ℝ := {x : ℝ | a < x ∧ x < 3*a}

-- Theorem for part (1)
theorem intersection_nonempty_iff_a_in_range (a : ℝ) :
  (A ∩ B a).Nonempty ↔ (4/3 ≤ a ∧ a < 4) :=
sorry

-- Theorem for part (2)
theorem union_equals_interval_iff_a_equals_two (a : ℝ) :
  A ∪ B a = {x : ℝ | 2 < x ∧ x < 6} ↔ a = 2 :=
sorry

end NUMINAMATH_CALUDE_intersection_nonempty_iff_a_in_range_union_equals_interval_iff_a_equals_two_l2113_211389


namespace NUMINAMATH_CALUDE_parabola_properties_l2113_211395

/-- Parabola structure -/
structure Parabola where
  focus : ℝ × ℝ
  equation : ℝ → ℝ → Prop

/-- Line structure -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Two points on a parabola -/
structure ParabolaPoints where
  A : ℝ × ℝ
  B : ℝ × ℝ

/-- Theorem about properties of points on a parabola -/
theorem parabola_properties
  (E : Parabola)
  (pts : ParabolaPoints)
  (symmetry_line : Line)
  (h1 : E.equation = fun x y ↦ y^2 = 4*x)
  (h2 : pts.A.1 ≠ pts.B.1 ∨ pts.A.2 ≠ pts.B.2)
  (h3 : E.equation pts.A.1 pts.A.2 ∧ E.equation pts.B.1 pts.B.2)
  (h4 : symmetry_line.slope = k)
  (h5 : symmetry_line.intercept = 4)
  (h6 : ∃ x₀, pts.A.2 - pts.B.2 = -k * (pts.A.1 - pts.B.1) ∧ 
                pts.A.2 / (pts.A.1 - x₀) = pts.B.2 / (pts.B.1 - x₀)) :
  E.focus = (1, 0) ∧ 
  pts.A.1 + pts.B.1 = 4 ∧ 
  ∃ x₀ : ℝ, -2 < x₀ ∧ x₀ < 2 := by
  sorry

end NUMINAMATH_CALUDE_parabola_properties_l2113_211395


namespace NUMINAMATH_CALUDE_function_periodicity_l2113_211336

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def satisfies_condition (f : ℝ → ℝ) : Prop := ∀ x, f x + f (10 - x) = 4

def periodic (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

theorem function_periodicity (f : ℝ → ℝ) 
  (h_even : is_even f) 
  (h_cond : satisfies_condition f) : 
  periodic f 20 := by sorry

end NUMINAMATH_CALUDE_function_periodicity_l2113_211336


namespace NUMINAMATH_CALUDE_yahs_to_bahs_conversion_l2113_211306

/-- Given the conversion rates between bahs, rahs, and yahs, 
    prove that 1500 yahs are equivalent to 500 bahs. -/
theorem yahs_to_bahs_conversion 
  (bah_to_rah : (20 : ℚ) / 36 = 1 / (36 / 20)) 
  (rah_to_yah : (12 : ℚ) / 20 = 1 / (20 / 12)) : 
  (1500 : ℚ) * (12 / 20) * (20 / 36) = 500 :=
sorry

end NUMINAMATH_CALUDE_yahs_to_bahs_conversion_l2113_211306


namespace NUMINAMATH_CALUDE_more_numbers_with_one_l2113_211348

def range_upper_bound : ℕ := 10^10

def numbers_without_one (n : ℕ) : ℕ := 9^n - 1

theorem more_numbers_with_one :
  range_upper_bound - numbers_without_one 10 > numbers_without_one 10 := by
  sorry

end NUMINAMATH_CALUDE_more_numbers_with_one_l2113_211348


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2113_211355

def A : Set ℕ := {1, 3, 5, 7}
def B : Set ℕ := {2, 3, 5}

theorem intersection_of_A_and_B : A ∩ B = {3, 5} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2113_211355


namespace NUMINAMATH_CALUDE_quadratic_root_square_l2113_211358

theorem quadratic_root_square (a : ℚ) : 
  (∃ x y : ℚ, x^2 - (15/4)*x + a^3 = 0 ∧ y^2 - (15/4)*y + a^3 = 0 ∧ x = y^2) ↔ 
  (a = 3/2 ∨ a = -5/2) :=
sorry

end NUMINAMATH_CALUDE_quadratic_root_square_l2113_211358


namespace NUMINAMATH_CALUDE_bridge_length_l2113_211371

/-- The length of a bridge given train parameters and crossing time -/
theorem bridge_length (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time_s : ℝ) :
  train_length = 148 →
  train_speed_kmh = 45 →
  crossing_time_s = 30 →
  let train_speed_ms := train_speed_kmh * 1000 / 3600
  let total_distance := train_speed_ms * crossing_time_s
  let bridge_length := total_distance - train_length
  bridge_length = 227 := by
  sorry

end NUMINAMATH_CALUDE_bridge_length_l2113_211371


namespace NUMINAMATH_CALUDE_wendy_time_l2113_211326

-- Define the race participants
structure Racer where
  name : String
  time : Real

-- Define the race
def waterslideRace (bonnie wendy : Racer) : Prop :=
  wendy.time + 0.25 = bonnie.time ∧ bonnie.time = 7.80

-- Theorem to prove
theorem wendy_time (bonnie wendy : Racer) :
  waterslideRace bonnie wendy → wendy.time = 7.55 := by
  sorry

end NUMINAMATH_CALUDE_wendy_time_l2113_211326


namespace NUMINAMATH_CALUDE_satellite_survey_is_census_l2113_211325

/-- Represents a survey type -/
inductive SurveyType
| Sample
| Census

/-- Represents a survey option -/
structure SurveyOption where
  description : String
  type : SurveyType

/-- Determines if a survey option is suitable for a census -/
def isSuitableForCensus (survey : SurveyOption) : Prop :=
  survey.type = SurveyType.Census

/-- The satellite component quality survey -/
def satelliteComponentSurvey : SurveyOption :=
  { description := "Investigating the quality of components of the satellite \"Zhangheng-1\""
    type := SurveyType.Census }

/-- Theorem stating that the satellite component survey is suitable for a census -/
theorem satellite_survey_is_census : 
  isSuitableForCensus satelliteComponentSurvey := by
  sorry


end NUMINAMATH_CALUDE_satellite_survey_is_census_l2113_211325


namespace NUMINAMATH_CALUDE_toy_average_price_l2113_211399

theorem toy_average_price (n : ℕ) (dhoni_avg : ℚ) (david_price : ℚ) : 
  n = 5 → dhoni_avg = 10 → david_price = 16 → 
  (n * dhoni_avg + david_price) / (n + 1) = 11 := by sorry

end NUMINAMATH_CALUDE_toy_average_price_l2113_211399


namespace NUMINAMATH_CALUDE_probability_not_greater_than_two_l2113_211341

def card_set : Finset ℕ := {1, 2, 3, 4}

theorem probability_not_greater_than_two :
  (card_set.filter (λ x => x ≤ 2)).card / card_set.card = (1 : ℚ) / 2 := by
  sorry

end NUMINAMATH_CALUDE_probability_not_greater_than_two_l2113_211341


namespace NUMINAMATH_CALUDE_twelve_chairs_subsets_l2113_211362

/-- The number of chairs arranged in a circle -/
def n : ℕ := 12

/-- A function that calculates the number of subsets with at least four adjacent chairs -/
def subsets_with_adjacent_chairs (n : ℕ) : ℕ := sorry

/-- Theorem stating that for 12 chairs in a circle, there are 1704 subsets with at least four adjacent chairs -/
theorem twelve_chairs_subsets : subsets_with_adjacent_chairs n = 1704 := by sorry

end NUMINAMATH_CALUDE_twelve_chairs_subsets_l2113_211362


namespace NUMINAMATH_CALUDE_decimal_to_base_k_l2113_211343

/-- Given that the decimal number 26 is equal to the base-k number 32, prove that k = 8 -/
theorem decimal_to_base_k (k : ℕ) (h : 3 * k + 2 = 26) : k = 8 := by
  sorry

end NUMINAMATH_CALUDE_decimal_to_base_k_l2113_211343


namespace NUMINAMATH_CALUDE_smallest_three_digit_palindrome_times_111_not_five_digit_palindrome_l2113_211315

/-- Checks if a number is a three-digit palindrome -/
def isThreeDigitPalindrome (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999 ∧ (n / 100 = n % 10)

/-- Checks if a number is a five-digit palindrome -/
def isFiveDigitPalindrome (n : ℕ) : Prop :=
  10000 ≤ n ∧ n ≤ 99999 ∧ (n / 10000 = n % 10) ∧ ((n / 1000) % 10 = (n / 10) % 10)

/-- The main theorem -/
theorem smallest_three_digit_palindrome_times_111_not_five_digit_palindrome :
  ∀ n : ℕ, isThreeDigitPalindrome n →
    (n < 111 ∨ isFiveDigitPalindrome (n * 111)) →
    ¬isThreeDigitPalindrome 111 ∨ isFiveDigitPalindrome (111 * 111) :=
by
  sorry

#check smallest_three_digit_palindrome_times_111_not_five_digit_palindrome

end NUMINAMATH_CALUDE_smallest_three_digit_palindrome_times_111_not_five_digit_palindrome_l2113_211315


namespace NUMINAMATH_CALUDE_polar_line_theorem_l2113_211340

/-- Represents a point in polar coordinates -/
structure PolarPoint where
  rho : ℝ
  theta : ℝ

/-- Represents a line in polar coordinates -/
structure PolarLine where
  equation : ℝ → ℝ → Prop

/-- Checks if a point lies on a line in polar coordinates -/
def lies_on (p : PolarPoint) (l : PolarLine) : Prop :=
  l.equation p.rho p.theta

/-- Checks if a line is parallel to the polar axis -/
def parallel_to_polar_axis (l : PolarLine) : Prop :=
  ∀ (rho theta : ℝ), l.equation rho theta ↔ l.equation rho 0

theorem polar_line_theorem (p : PolarPoint) (l : PolarLine) 
  (h1 : p.rho = 2 ∧ p.theta = π/3)
  (h2 : lies_on p l)
  (h3 : parallel_to_polar_axis l) :
  ∀ (rho theta : ℝ), l.equation rho theta ↔ rho * Real.sin theta = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_polar_line_theorem_l2113_211340


namespace NUMINAMATH_CALUDE_quadratic_equation_properties_l2113_211374

/-- The quadratic equation x^2 + 2(m-1)x + m^2 - 1 = 0 has two distinct real roots -/
def has_two_distinct_roots (m : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 + 2*(m-1)*x₁ + m^2 - 1 = 0 ∧ x₂^2 + 2*(m-1)*x₂ + m^2 - 1 = 0

/-- The product of the roots of the equation x^2 + 2(m-1)x + m^2 - 1 = 0 is zero -/
def roots_product_zero (m : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 + 2*(m-1)*x₁ + m^2 - 1 = 0 ∧ x₂^2 + 2*(m-1)*x₂ + m^2 - 1 = 0 ∧ x₁ * x₂ = 0

theorem quadratic_equation_properties :
  (∀ m : ℝ, has_two_distinct_roots m → m < 1) ∧
  (∃ m : ℝ, roots_product_zero m ∧ m = -1) :=
sorry

end NUMINAMATH_CALUDE_quadratic_equation_properties_l2113_211374


namespace NUMINAMATH_CALUDE_modulus_constraint_implies_range_l2113_211313

theorem modulus_constraint_implies_range (a : ℝ) :
  (∀ θ : ℝ, Complex.abs ((a + Real.cos θ) + (2 * a - Real.sin θ) * Complex.I) ≤ 2) →
  a ∈ Set.Icc (-Real.sqrt 5 / 5) (Real.sqrt 5 / 5) :=
by sorry

end NUMINAMATH_CALUDE_modulus_constraint_implies_range_l2113_211313


namespace NUMINAMATH_CALUDE_valid_reasoning_methods_l2113_211382

-- Define the set of reasoning methods
inductive ReasoningMethod
| Method1
| Method2
| Method3
| Method4

-- Define a predicate for valid analogical reasoning
def is_valid_analogical_reasoning (m : ReasoningMethod) : Prop :=
  m = ReasoningMethod.Method1

-- Define a predicate for valid inductive reasoning
def is_valid_inductive_reasoning (m : ReasoningMethod) : Prop :=
  m = ReasoningMethod.Method2 ∨ m = ReasoningMethod.Method4

-- Define a predicate for valid reasoning
def is_valid_reasoning (m : ReasoningMethod) : Prop :=
  is_valid_analogical_reasoning m ∨ is_valid_inductive_reasoning m

-- Theorem statement
theorem valid_reasoning_methods :
  {m : ReasoningMethod | is_valid_reasoning m} =
  {ReasoningMethod.Method1, ReasoningMethod.Method2, ReasoningMethod.Method4} :=
by sorry

end NUMINAMATH_CALUDE_valid_reasoning_methods_l2113_211382


namespace NUMINAMATH_CALUDE_bug_total_distance_l2113_211328

def bug_path : List ℤ := [3, -4, 7, -1]

def distance (a b : ℤ) : ℕ := (a - b).natAbs

def total_distance (path : List ℤ) : ℕ :=
  (path.zip (path.tail!)).foldl (λ acc (a, b) => acc + distance a b) 0

theorem bug_total_distance :
  total_distance bug_path = 26 := by sorry

end NUMINAMATH_CALUDE_bug_total_distance_l2113_211328


namespace NUMINAMATH_CALUDE_proposition_truths_l2113_211323

-- Define the propositions
def proposition1 (a b : ℝ) : Prop := 
  (∃ (q : ℚ), a + b = q) → (∃ (r s : ℚ), a = r ∧ b = s)

def proposition2 (a b : ℝ) : Prop :=
  a + b ≥ 2 → (a ≥ 1 ∨ b ≥ 1)

def proposition3 (a b : ℝ) : Prop :=
  ∀ x, a * x + b > 0 ↔ x > -b / a

def proposition4 (a b c : ℝ) : Prop :=
  (∃ x, a * x^2 + b * x + c = 0 ∧ x = 1) ↔ a + b + c = 0

-- Theorem stating which propositions are true
theorem proposition_truths :
  (∃ a b : ℝ, ¬ proposition1 a b) ∧
  (∀ a b : ℝ, proposition2 a b) ∧
  (∃ a b : ℝ, ¬ proposition3 a b) ∧
  (∀ a b c : ℝ, proposition4 a b c) :=
sorry

end NUMINAMATH_CALUDE_proposition_truths_l2113_211323


namespace NUMINAMATH_CALUDE_poly_sequence_properties_l2113_211337

/-- Represents a polynomial sequence generated by the given operation -/
def PolySequence (a : ℝ) (n : ℕ) : List ℝ :=
  sorry

/-- The product of all polynomials in the sequence after n operations -/
def PolyProduct (a : ℝ) (n : ℕ) : ℝ :=
  sorry

/-- The sum of all polynomials in the sequence after n operations -/
def PolySum (a : ℝ) (n : ℕ) : ℝ :=
  sorry

theorem poly_sequence_properties (a : ℝ) :
  (∀ a, |a| ≥ 2 → PolyProduct a 2 ≤ 0) ∧
  (∀ n, PolySum a n = 2*a + 2*(n+1)) :=
by sorry

end NUMINAMATH_CALUDE_poly_sequence_properties_l2113_211337


namespace NUMINAMATH_CALUDE_fraction_simplification_l2113_211377

theorem fraction_simplification :
  (156 + 72 : ℚ) / 9000 = 19 / 750 := by sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2113_211377


namespace NUMINAMATH_CALUDE_triangle_inequality_l2113_211333

theorem triangle_inequality (a b c : ℝ) 
  (triangle_sides : a > 0 ∧ b > 0 ∧ c > 0)
  (triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b)
  (perimeter : a + b + c = 2) :
  a^2 + b^2 + c^2 < 2*(1 - a*b*c) := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l2113_211333


namespace NUMINAMATH_CALUDE_box_volume_is_four_cubic_feet_l2113_211393

/-- Calculates the internal volume of a box in cubic feet given its external dimensions in inches and wall thickness -/
def internal_volume (length width height wall_thickness : ℚ) : ℚ :=
  let internal_length := length - 2 * wall_thickness
  let internal_width := width - 2 * wall_thickness
  let internal_height := height - 2 * wall_thickness
  (internal_length * internal_width * internal_height) / 1728

/-- Proves that the internal volume of the specified box is 4 cubic feet -/
theorem box_volume_is_four_cubic_feet :
  internal_volume 26 26 14 1 = 4 := by
  sorry

end NUMINAMATH_CALUDE_box_volume_is_four_cubic_feet_l2113_211393


namespace NUMINAMATH_CALUDE_rectangular_solid_length_l2113_211349

/-- The surface area of a rectangular solid given its length, width, and depth. -/
def surface_area (l w h : ℝ) : ℝ := 2 * (l * w + l * h + w * h)

/-- Theorem: For a rectangular solid with width 8 meters, depth 5 meters, and 
    total surface area 314 square meters, the length is 9 meters. -/
theorem rectangular_solid_length :
  ∃ l : ℝ, surface_area l 8 5 = 314 ∧ l = 9 :=
by sorry

end NUMINAMATH_CALUDE_rectangular_solid_length_l2113_211349


namespace NUMINAMATH_CALUDE_ellipse_equation_l2113_211396

/-- The equation of an ellipse given specific conditions -/
theorem ellipse_equation (a b : ℝ) (h1 : a > b) (h2 : b > 0) : 
  (∃ (c : ℝ), c = 4 ∧ a^2 - b^2 = c^2) →  -- Right focus coincides with parabola focus
  (a / c = 3 / Real.sqrt 6) →             -- Eccentricity condition
  (∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1 ↔ x^2 / 24 + y^2 / 8 = 1) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_equation_l2113_211396


namespace NUMINAMATH_CALUDE_alternating_squares_sum_l2113_211387

theorem alternating_squares_sum : 
  23^2 - 21^2 + 19^2 - 17^2 + 15^2 - 13^2 + 11^2 - 9^2 + 7^2 - 5^2 + 3^2 - 1^2 = 288 := by
  sorry

end NUMINAMATH_CALUDE_alternating_squares_sum_l2113_211387


namespace NUMINAMATH_CALUDE_max_tan_A_l2113_211388

theorem max_tan_A (A B : Real) (h1 : 0 < A) (h2 : A < π/2) (h3 : 0 < B) (h4 : B < π/2)
  (h5 : 3 * Real.sin A = Real.cos (A + B) * Real.sin B) :
  ∃ (max_tan_A : Real), ∀ (A' B' : Real),
    0 < A' → A' < π/2 → 0 < B' → B' < π/2 →
    3 * Real.sin A' = Real.cos (A' + B') * Real.sin B' →
    Real.tan A' ≤ max_tan_A ∧
    max_tan_A = Real.sqrt 3 / 12 := by
  sorry

end NUMINAMATH_CALUDE_max_tan_A_l2113_211388


namespace NUMINAMATH_CALUDE_expression_bounds_l2113_211317

theorem expression_bounds (a b c d : ℝ) 
  (ha : 0 ≤ a ∧ a ≤ 1) (hb : 0 ≤ b ∧ b ≤ 1) 
  (hc : 0 ≤ c ∧ c ≤ 1/2) (hd : 0 ≤ d ∧ d ≤ 1/2) :
  let expr := Real.sqrt (a^2 + (1-b)^2) + Real.sqrt (b^2 + (1-c)^2) + 
               Real.sqrt (c^2 + (1-d)^2) + Real.sqrt (d^2 + (1-a)^2)
  2 * Real.sqrt 2 ≤ expr ∧ expr ≤ 4 ∧ 
  ∀ x, 2 * Real.sqrt 2 ≤ x ∧ x ≤ 4 → ∃ a b c d, 
    0 ≤ a ∧ a ≤ 1 ∧ 0 ≤ b ∧ b ≤ 1 ∧ 0 ≤ c ∧ c ≤ 1/2 ∧ 0 ≤ d ∧ d ≤ 1/2 ∧
    x = Real.sqrt (a^2 + (1-b)^2) + Real.sqrt (b^2 + (1-c)^2) + 
        Real.sqrt (c^2 + (1-d)^2) + Real.sqrt (d^2 + (1-a)^2) := by
  sorry

end NUMINAMATH_CALUDE_expression_bounds_l2113_211317


namespace NUMINAMATH_CALUDE_range_of_m_l2113_211390

-- Define the propositions p and q
def p (m : ℝ) : Prop := ∀ x y : ℝ, x - 2*y + 3 ≠ 0 ∨ y^2 ≠ m*x

def q (m : ℝ) : Prop := ∀ x y : ℝ, (x^2)/(5-2*m) + (y^2)/m = 1 → 
  (5-2*m < 0 ∧ m > 0) ∨ (5-2*m > 0 ∧ m < 0)

-- Define the theorem
theorem range_of_m : 
  ∀ m : ℝ, m ≠ 0 → (p m ∨ q m) → ¬(p m ∧ q m) → 
    m ≥ 3 ∨ m < 0 ∨ (0 < m ∧ m ≤ 5/2) := by sorry

end NUMINAMATH_CALUDE_range_of_m_l2113_211390


namespace NUMINAMATH_CALUDE_min_value_and_inequality_l2113_211305

def f (x : ℝ) : ℝ := |2*x + 1| + |2*x - 3|

theorem min_value_and_inequality :
  (∃ (M : ℝ), (∀ (m : ℝ), (∃ (x₀ : ℝ), f x₀ ≤ m) → M ≤ m) ∧ (∃ (x₀ : ℝ), f x₀ ≤ M) ∧ M = 4) ∧
  (∀ (a b : ℝ), a > 0 → b > 0 → 3*a + b = 4 → 3/b + 1/a ≥ 3) :=
by sorry

end NUMINAMATH_CALUDE_min_value_and_inequality_l2113_211305


namespace NUMINAMATH_CALUDE_divisibility_criterion_l2113_211397

theorem divisibility_criterion (x : ℤ) : 
  (∃ k : ℤ, 3 * x + 7 = 14 * k) ↔ (∃ t : ℤ, x = 14 * t + 7) := by sorry

end NUMINAMATH_CALUDE_divisibility_criterion_l2113_211397


namespace NUMINAMATH_CALUDE_bing_dwen_dwen_sales_equation_l2113_211375

/-- The sales equation for Bing Dwen Dwen mascot -/
theorem bing_dwen_dwen_sales_equation (x : ℝ) : 
  (5000 : ℝ) * (1 + x) + (5000 : ℝ) * (1 + x)^2 = 22500 ↔ 
  (∃ (sales_feb4 sales_feb5 sales_feb6 : ℝ),
    sales_feb4 = 5000 ∧
    sales_feb5 = sales_feb4 * (1 + x) ∧
    sales_feb6 = sales_feb5 * (1 + x) ∧
    sales_feb5 + sales_feb6 = 22500) :=
by
  sorry

end NUMINAMATH_CALUDE_bing_dwen_dwen_sales_equation_l2113_211375


namespace NUMINAMATH_CALUDE_system_equation_ratio_l2113_211332

theorem system_equation_ratio (x y a b : ℝ) : 
  x ≠ 0 → 
  y ≠ 0 → 
  b ≠ 0 → 
  8 * x - 6 * y = a → 
  12 * y - 18 * x = b → 
  a / b = -1 / 2 := by
sorry

end NUMINAMATH_CALUDE_system_equation_ratio_l2113_211332


namespace NUMINAMATH_CALUDE_triangle_side_length_l2113_211310

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ)  -- angles
  (a b c : ℝ)  -- sides

-- State the theorem
theorem triangle_side_length 
  (t : Triangle) 
  (ha : t.a = Real.sqrt 5) 
  (hc : t.c = 2) 
  (hcosA : Real.cos t.A = 2/3) : 
  t.b = 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l2113_211310


namespace NUMINAMATH_CALUDE_circle_area_with_diameter_10_l2113_211339

theorem circle_area_with_diameter_10 (π : ℝ) :
  let diameter : ℝ := 10
  let radius : ℝ := diameter / 2
  let area : ℝ := π * radius ^ 2
  area = 25 * π :=
by sorry

end NUMINAMATH_CALUDE_circle_area_with_diameter_10_l2113_211339


namespace NUMINAMATH_CALUDE_correct_product_l2113_211373

def reverse_digits (n : Nat) : Nat :=
  (n % 10) * 10 + (n / 10)

theorem correct_product (a b : Nat) : 
  (10 ≤ a ∧ a < 100) →  -- a is a two-digit number
  (0 < b) →             -- b is positive
  ((reverse_digits a) * b = 143) →  -- erroneous product
  (a * b = 341) :=
by sorry

end NUMINAMATH_CALUDE_correct_product_l2113_211373


namespace NUMINAMATH_CALUDE_unique_prime_triplet_l2113_211319

theorem unique_prime_triplet : ∃! p : ℕ, Prime p ∧ Prime (p + 2) ∧ Prime (p + 4) ∧ p = 3 := by
  sorry

end NUMINAMATH_CALUDE_unique_prime_triplet_l2113_211319


namespace NUMINAMATH_CALUDE_sally_final_count_l2113_211335

def sally_pokemon_cards (initial : ℕ) (from_dan : ℕ) (bought : ℕ) : ℕ :=
  initial + from_dan + bought

theorem sally_final_count :
  sally_pokemon_cards 27 41 20 = 88 := by
  sorry

end NUMINAMATH_CALUDE_sally_final_count_l2113_211335


namespace NUMINAMATH_CALUDE_billy_age_is_47_25_l2113_211318

-- Define Billy's and Joe's ages
def billy_age : ℝ := sorry
def joe_age : ℝ := sorry

-- State the theorem
theorem billy_age_is_47_25 :
  (billy_age = 3 * joe_age) →  -- Billy's age is three times Joe's age
  (billy_age + joe_age = 63) → -- The sum of their ages is 63 years
  (billy_age = 47.25) :=       -- Billy's age is 47.25 years
by
  sorry

end NUMINAMATH_CALUDE_billy_age_is_47_25_l2113_211318


namespace NUMINAMATH_CALUDE_janes_garden_area_l2113_211379

/-- Represents a rectangular garden with fence posts -/
structure Garden where
  total_posts : ℕ
  post_spacing : ℕ
  long_side_posts : ℕ
  short_side_posts : ℕ

/-- Calculates the area of the garden -/
def garden_area (g : Garden) : ℕ :=
  (g.short_side_posts - 1) * g.post_spacing * (g.long_side_posts - 1) * g.post_spacing

/-- Theorem stating the area of Jane's garden -/
theorem janes_garden_area :
  ∀ g : Garden,
    g.total_posts = 24 →
    g.post_spacing = 3 →
    g.long_side_posts = 3 * g.short_side_posts →
    g.total_posts = 2 * (g.short_side_posts + g.long_side_posts) - 4 →
    garden_area g = 144 := by
  sorry


end NUMINAMATH_CALUDE_janes_garden_area_l2113_211379


namespace NUMINAMATH_CALUDE_parabola_intersection_circle_radius_squared_l2113_211324

theorem parabola_intersection_circle_radius_squared (x y : ℝ) : 
  y = (x - 2)^2 ∧ x + 6 = (y - 5)^2 → 
  (x - 5/2)^2 + (y - 9/2)^2 = 83/4 :=
by sorry

end NUMINAMATH_CALUDE_parabola_intersection_circle_radius_squared_l2113_211324


namespace NUMINAMATH_CALUDE_stratified_sampling_theorem_l2113_211302

/-- Represents the number of employees in each age group -/
structure EmployeeCount where
  total : ℕ
  young : ℕ
  middleAged : ℕ
  elderly : ℕ

/-- Represents the sample size for each age group -/
structure SampleSize where
  young : ℕ
  middleAged : ℕ
  elderly : ℕ

/-- Checks if the sample sizes are proportional to the population sizes -/
def isProportionalSample (employees : EmployeeCount) (sample : SampleSize) (totalSampleSize : ℕ) : Prop :=
  sample.young * employees.total = employees.young * totalSampleSize ∧
  sample.middleAged * employees.total = employees.middleAged * totalSampleSize ∧
  sample.elderly * employees.total = employees.elderly * totalSampleSize

/-- The main theorem to prove -/
theorem stratified_sampling_theorem (employees : EmployeeCount) (sample : SampleSize) :
  employees.total = 750 →
  employees.young = 350 →
  employees.middleAged = 250 →
  employees.elderly = 150 →
  sample.young = 7 →
  sample.middleAged = 5 →
  sample.elderly = 3 →
  isProportionalSample employees sample 15 :=
by sorry

end NUMINAMATH_CALUDE_stratified_sampling_theorem_l2113_211302


namespace NUMINAMATH_CALUDE_equal_area_division_l2113_211384

/-- A triangle with vertices A, B, and C -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- The area of a triangle -/
def triangleArea (t : Triangle) : ℝ := sorry

/-- The area of the right part of a triangle divided by a vertical line -/
def rightAreaDivided (t : Triangle) (a : ℝ) : ℝ := sorry

/-- Theorem: For a triangle ABC with vertices A = (0,2), B = (0,0), and C = (6,0),
    where line AC is horizontal, the vertical line x = 3 divides the triangle
    into two regions of equal area -/
theorem equal_area_division (t : Triangle) 
    (h1 : t.A = (0, 2)) 
    (h2 : t.B = (0, 0)) 
    (h3 : t.C = (6, 0)) 
    (h4 : t.A.2 = t.C.2) : -- Line AC is horizontal
  2 * rightAreaDivided t 3 = triangleArea t := by sorry

end NUMINAMATH_CALUDE_equal_area_division_l2113_211384


namespace NUMINAMATH_CALUDE_tan_sum_15_30_l2113_211361

-- Define the tangent function
noncomputable def tan (x : ℝ) : ℝ := Real.tan x

-- State the theorem
theorem tan_sum_15_30 :
  tan (15 * π / 180) + tan (30 * π / 180) + tan (15 * π / 180) * tan (30 * π / 180) = 1 :=
by
  -- Assume the trigonometric identity for the tangent of the sum of two angles
  have tan_sum_identity : ∀ A B : ℝ, tan (A + B) = (tan A + tan B) / (1 - tan A * tan B) := sorry
  -- Assume that tan 45° = 1
  have tan_45 : tan (45 * π / 180) = 1 := sorry
  sorry -- The proof goes here

end NUMINAMATH_CALUDE_tan_sum_15_30_l2113_211361


namespace NUMINAMATH_CALUDE_only_n_two_works_l2113_211312

theorem only_n_two_works (n : ℕ) (a : ℝ) : n ≥ 2 →
  (∃ q₁ q₂ : ℚ, (a + Real.sqrt 2 = q₁) ∧ (a^n + Real.sqrt 2 = q₂)) →
  n = 2 :=
by sorry

end NUMINAMATH_CALUDE_only_n_two_works_l2113_211312


namespace NUMINAMATH_CALUDE_shirley_eggs_theorem_l2113_211381

/-- The number of eggs Shirley started with -/
def initial_eggs : ℕ := 98

/-- The number of eggs Shirley bought -/
def bought_eggs : ℕ := 8

/-- The total number of eggs Shirley ended with -/
def final_eggs : ℕ := 106

/-- Theorem stating that the initial number of eggs plus the bought eggs equals the final number of eggs -/
theorem shirley_eggs_theorem : initial_eggs + bought_eggs = final_eggs := by
  sorry

end NUMINAMATH_CALUDE_shirley_eggs_theorem_l2113_211381


namespace NUMINAMATH_CALUDE_sin_theta_plus_2phi_l2113_211385

theorem sin_theta_plus_2phi (θ φ : ℝ) (h1 : Complex.exp (Complex.I * θ) = (1/5) - (2/5) * Complex.I)
  (h2 : Complex.exp (Complex.I * φ) = (3/5) + (4/5) * Complex.I) :
  Real.sin (θ + 2*φ) = 62/125 := by
  sorry

end NUMINAMATH_CALUDE_sin_theta_plus_2phi_l2113_211385


namespace NUMINAMATH_CALUDE_function_increment_proof_l2113_211314

/-- The function f(x) = 2x^2 + 1 -/
def f (x : ℝ) : ℝ := 2 * x^2 + 1

/-- The initial x value -/
def x₀ : ℝ := 1

/-- The final x value -/
def x₁ : ℝ := 1.02

/-- The increment of x -/
def Δx : ℝ := x₁ - x₀

theorem function_increment_proof :
  f x₁ - f x₀ = 0.0808 :=
sorry

end NUMINAMATH_CALUDE_function_increment_proof_l2113_211314


namespace NUMINAMATH_CALUDE_total_cookies_baked_l2113_211369

/-- Calculates the total number of cookies baked by a baker -/
theorem total_cookies_baked 
  (chocolate_chip_batches : ℕ) 
  (cookies_per_batch : ℕ) 
  (oatmeal_cookies : ℕ) : 
  chocolate_chip_batches * cookies_per_batch + oatmeal_cookies = 10 :=
by
  sorry

#check total_cookies_baked 2 3 4

end NUMINAMATH_CALUDE_total_cookies_baked_l2113_211369
