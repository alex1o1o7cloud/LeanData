import Mathlib

namespace NUMINAMATH_CALUDE_distance_travelled_downstream_l2433_243357

/-- The distance travelled downstream by a boat -/
theorem distance_travelled_downstream 
  (boat_speed : ℝ) -- Speed of the boat in still water (km/hr)
  (current_speed : ℝ) -- Speed of the current (km/hr)
  (travel_time : ℝ) -- Travel time (minutes)
  (h1 : boat_speed = 20) -- Given boat speed
  (h2 : current_speed = 4) -- Given current speed
  (h3 : travel_time = 24) -- Given travel time
  : (boat_speed + current_speed) * (travel_time / 60) = 9.6 := by
  sorry

end NUMINAMATH_CALUDE_distance_travelled_downstream_l2433_243357


namespace NUMINAMATH_CALUDE_max_value_sqrt_x2_y2_l2433_243365

theorem max_value_sqrt_x2_y2 (x y : ℝ) (h : 3 * x^2 + 2 * y^2 = 6 * x) :
  ∃ (max : ℝ), max = 2 ∧ ∀ (x' y' : ℝ), 3 * x'^2 + 2 * y'^2 = 6 * x' → Real.sqrt (x'^2 + y'^2) ≤ max :=
by sorry

end NUMINAMATH_CALUDE_max_value_sqrt_x2_y2_l2433_243365


namespace NUMINAMATH_CALUDE_ellipse_equation_l2433_243393

/-- The equation √(x² + (y-3)²) + √(x² + (y+3)²) = 10 represents an ellipse. -/
theorem ellipse_equation (x y : ℝ) :
  (Real.sqrt (x^2 + (y - 3)^2) + Real.sqrt (x^2 + (y + 3)^2) = 10) ↔
  (y^2 / 25 + x^2 / 16 = 1) :=
sorry

end NUMINAMATH_CALUDE_ellipse_equation_l2433_243393


namespace NUMINAMATH_CALUDE_unit_circle_tangent_l2433_243324

theorem unit_circle_tangent (x y θ : ℝ) : 
  x^2 + y^2 = 1 →  -- Point (x, y) is on the unit circle
  x > 0 →          -- Point is in the first quadrant
  y > 0 →          -- Point is in the first quadrant
  x = Real.cos θ → -- θ is the angle from positive x-axis
  y = Real.sin θ → -- to the ray through (x, y)
  Real.arccos ((4*x + 3*y) / 5) = θ → -- Given condition
  Real.tan θ = 1/3 := by sorry

end NUMINAMATH_CALUDE_unit_circle_tangent_l2433_243324


namespace NUMINAMATH_CALUDE_cornbread_pieces_l2433_243349

def pan_length : ℕ := 24
def pan_width : ℕ := 20
def piece_size : ℕ := 3

theorem cornbread_pieces :
  (pan_length * pan_width) / (piece_size * piece_size) = 53 :=
by sorry

end NUMINAMATH_CALUDE_cornbread_pieces_l2433_243349


namespace NUMINAMATH_CALUDE_remainder_67_power_67_plus_67_mod_68_l2433_243331

theorem remainder_67_power_67_plus_67_mod_68 : (67^67 + 67) % 68 = 66 := by
  sorry

end NUMINAMATH_CALUDE_remainder_67_power_67_plus_67_mod_68_l2433_243331


namespace NUMINAMATH_CALUDE_parabola_focus_hyperbola_equation_l2433_243327

-- Part 1: Parabola
theorem parabola_focus (p : ℝ) (h1 : p > 0) :
  (∃ x y : ℝ, y^2 = 2*p*x ∧ 2*x - y - 4 = 0 ∧ x = p/2 ∧ y = 0) →
  p = 4 := by sorry

-- Part 2: Hyperbola
theorem hyperbola_equation (a b : ℝ) (h1 : a > 0) (h2 : b > 0) :
  (b/a = 3/4 ∧ a^2/(a^2 + b^2)^(1/2) = 16/5) →
  (∀ x y : ℝ, x^2/16 - y^2/9 = 1 ↔ x^2/a^2 - y^2/b^2 = 1) := by sorry

end NUMINAMATH_CALUDE_parabola_focus_hyperbola_equation_l2433_243327


namespace NUMINAMATH_CALUDE_problem_solution_l2433_243333

theorem problem_solution : 
  let a : Float := 0.137
  let b : Float := 0.098
  let c : Float := 0.123
  let d : Float := 0.086
  ((a + b)^2 - (a - b)^2) / (c * d) + (d^3 - c^3) / (a * b * (a + b)) = 4.6886 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2433_243333


namespace NUMINAMATH_CALUDE_histogram_height_is_seven_l2433_243355

/-- Represents a symmetric histogram construction made of sticks -/
structure HistogramConstruction where
  totalSticks : ℕ
  blockWidth : ℕ
  isSymmetric : Bool

/-- Calculates the total height of the histogram construction -/
def calculateHeight (h : HistogramConstruction) : ℕ :=
  sorry

/-- Theorem stating that for a symmetric histogram with 130 sticks and block width 2, the height is 7 -/
theorem histogram_height_is_seven :
  ∀ (h : HistogramConstruction),
    h.totalSticks = 130 ∧
    h.blockWidth = 2 ∧
    h.isSymmetric = true →
    calculateHeight h = 7 :=
by sorry

end NUMINAMATH_CALUDE_histogram_height_is_seven_l2433_243355


namespace NUMINAMATH_CALUDE_base_number_proof_l2433_243388

theorem base_number_proof (x : ℝ) : 9^7 = x^14 → x = 3 := by
  sorry

end NUMINAMATH_CALUDE_base_number_proof_l2433_243388


namespace NUMINAMATH_CALUDE_find_c_l2433_243395

theorem find_c (m c : ℕ) : 
  m < 10 → c < 10 → m = 2 * c → 
  (10 * m + c : ℚ) / 99 = (c + 4 : ℚ) / (m + 5) → 
  c = 3 :=
sorry

end NUMINAMATH_CALUDE_find_c_l2433_243395


namespace NUMINAMATH_CALUDE_product_of_roots_l2433_243308

theorem product_of_roots (x₁ y₁ x₂ y₂ x₃ y₃ x₄ y₄ : ℝ) 
  (h₁ : x₁^3 - 3*x₁*y₁^2 = 2017 ∧ y₁^3 - 3*x₁^2*y₁ = 2016)
  (h₂ : x₂^3 - 3*x₂*y₂^2 = 2017 ∧ y₂^3 - 3*x₂^2*y₂ = 2016)
  (h₃ : x₃^3 - 3*x₃*y₃^2 = 2017 ∧ y₃^3 - 3*x₃^2*y₃ = 2016)
  (h₄ : x₄^3 - 3*x₄*y₄^2 = 2017 ∧ y₄^3 - 3*x₄^2*y₄ = 2016)
  (h₅ : y₁ ≠ 0 ∧ y₂ ≠ 0 ∧ y₃ ≠ 0 ∧ y₄ ≠ 0) :
  (1 - x₁/y₁) * (1 - x₂/y₂) * (1 - x₃/y₃) * (1 - x₄/y₄) = -1/1008 := by
sorry

end NUMINAMATH_CALUDE_product_of_roots_l2433_243308


namespace NUMINAMATH_CALUDE_min_value_xy_expression_l2433_243391

theorem min_value_xy_expression (x y : ℝ) : (x*y - 2)^2 + (x^2 + y^2) ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_xy_expression_l2433_243391


namespace NUMINAMATH_CALUDE_seating_arrangement_l2433_243356

theorem seating_arrangement (total_people : ℕ) (max_rows : ℕ) 
  (h1 : total_people = 57)
  (h2 : max_rows = 8) : 
  ∃ (rows_with_9 rows_with_6 : ℕ),
    rows_with_9 + rows_with_6 ≤ max_rows ∧
    9 * rows_with_9 + 6 * rows_with_6 = total_people ∧
    rows_with_9 = 5 := by
  sorry

end NUMINAMATH_CALUDE_seating_arrangement_l2433_243356


namespace NUMINAMATH_CALUDE_alpha_value_l2433_243364

theorem alpha_value (α β : ℂ) 
  (h1 : (α + β).re > 0)
  (h2 : (Complex.I * (α - 2*β)).re > 0)
  (h3 : β = 3 + 2*Complex.I) :
  α = 6 - 2*Complex.I := by
sorry

end NUMINAMATH_CALUDE_alpha_value_l2433_243364


namespace NUMINAMATH_CALUDE_triangle_longest_side_l2433_243319

theorem triangle_longest_side (x : ℝ) : 
  let side1 := x^2 + 1
  let side2 := x + 5
  let side3 := 3*x - 1
  (side1 + side2 + side3 = 40) →
  (max side1 (max side2 side3) = 26) :=
by sorry

end NUMINAMATH_CALUDE_triangle_longest_side_l2433_243319


namespace NUMINAMATH_CALUDE_border_area_is_198_l2433_243378

-- Define the dimensions of the photograph
def photo_height : ℕ := 12
def photo_width : ℕ := 15

-- Define the width of the border
def border_width : ℕ := 3

-- Define the area of the border
def border_area : ℕ := 
  (photo_height + 2 * border_width) * (photo_width + 2 * border_width) - 
  photo_height * photo_width

-- Theorem statement
theorem border_area_is_198 : border_area = 198 := by
  sorry

end NUMINAMATH_CALUDE_border_area_is_198_l2433_243378


namespace NUMINAMATH_CALUDE_total_cards_after_addition_l2433_243396

theorem total_cards_after_addition (initial_playing_cards initial_id_cards additional_playing_cards additional_id_cards : ℕ) :
  initial_playing_cards = 9 →
  initial_id_cards = 4 →
  additional_playing_cards = 6 →
  additional_id_cards = 3 →
  initial_playing_cards + initial_id_cards + additional_playing_cards + additional_id_cards = 22 :=
by
  sorry

end NUMINAMATH_CALUDE_total_cards_after_addition_l2433_243396


namespace NUMINAMATH_CALUDE_cranberry_juice_unit_cost_l2433_243315

/-- The unit cost of cranberry juice in cents per ounce -/
def unit_cost (total_cost : ℚ) (volume : ℚ) : ℚ :=
  total_cost / volume

/-- Theorem stating that the unit cost of cranberry juice is 7 cents per ounce -/
theorem cranberry_juice_unit_cost :
  let total_cost : ℚ := 84
  let volume : ℚ := 12
  unit_cost total_cost volume = 7 := by sorry

end NUMINAMATH_CALUDE_cranberry_juice_unit_cost_l2433_243315


namespace NUMINAMATH_CALUDE_philips_banana_groups_l2433_243343

theorem philips_banana_groups :
  let total_bananas : ℕ := 392
  let bananas_per_group : ℕ := 2
  let num_groups : ℕ := total_bananas / bananas_per_group
  num_groups = 196 := by
  sorry

end NUMINAMATH_CALUDE_philips_banana_groups_l2433_243343


namespace NUMINAMATH_CALUDE_complex_expression_simplification_l2433_243302

theorem complex_expression_simplification (p q : ℝ) (hp : p > 0) (hpq : p > q) :
  let numerator := Real.sqrt ((p^4 + q^4) / (p^4 - p^2 * q^2) + 2 * q^2 / (p^2 - q^2) * (p^3 - p * q^2)) - 2 * q * Real.sqrt p
  let denominator := Real.sqrt (p / (p - q) - q / (p + q) - 2 * p * q / (p^2 - q^2)) * (p - q)
  numerator / denominator = Real.sqrt (p^2 - q^2) / Real.sqrt p :=
by sorry

end NUMINAMATH_CALUDE_complex_expression_simplification_l2433_243302


namespace NUMINAMATH_CALUDE_product_equals_19404_l2433_243332

theorem product_equals_19404 : 3^2 * 4 * 7^2 * 11 = 19404 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_19404_l2433_243332


namespace NUMINAMATH_CALUDE_smallest_winning_number_l2433_243312

theorem smallest_winning_number : ∃ N : ℕ, 
  N ≤ 499 ∧ 
  27 * N + 360 < 500 ∧ 
  (∀ k : ℕ, k < N → 27 * k + 360 ≥ 500) ∧
  N = 5 := by
sorry

end NUMINAMATH_CALUDE_smallest_winning_number_l2433_243312


namespace NUMINAMATH_CALUDE_part1_part2_l2433_243317

-- Part 1
theorem part1 (f : ℝ → ℝ) (b : ℝ) :
  (∀ x : ℝ, x ≥ 0 → f x = Real.exp x + Real.sin x + b) →
  (∀ x : ℝ, x ≥ 0 → f x ≥ 0) →
  b ≥ -1 := by sorry

-- Part 2
theorem part2 (f : ℝ → ℝ) (b m : ℝ) :
  (∀ x : ℝ, f x = Real.exp x + b) →
  (f 0 = 1 ∧ (deriv f) 0 = 1) →
  (∃! x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f x₁ = (m - 2*x₁) / x₁ ∧ f x₂ = (m - 2*x₂) / x₂) →
  -1 / Real.exp 1 < m ∧ m < 0 := by sorry

end NUMINAMATH_CALUDE_part1_part2_l2433_243317


namespace NUMINAMATH_CALUDE_hearts_on_card_l2433_243385

/-- The number of hearts on each card in a hypothetical deck -/
def hearts_per_card : ℕ := sorry

/-- The number of cows in Devonshire -/
def num_cows : ℕ := 2 * hearts_per_card

/-- The cost of each cow in dollars -/
def cost_per_cow : ℕ := 200

/-- The total cost of all cows in dollars -/
def total_cost : ℕ := 83200

theorem hearts_on_card :
  hearts_per_card = 208 :=
sorry

end NUMINAMATH_CALUDE_hearts_on_card_l2433_243385


namespace NUMINAMATH_CALUDE_range_of_a_l2433_243372

-- Define the propositions p and q
def p (x : ℝ) : Prop := |4*x - 3| ≤ 1
def q (x a : ℝ) : Prop := x^2 - (2*a + 1)*x + a*(a + 1) ≤ 0

-- Define the theorem
theorem range_of_a :
  (∀ x, ¬(p x) → ¬(q x a)) ∧ 
  (∃ x, ¬(q x a) ∧ p x) →
  0 ≤ a ∧ a ≤ 1/2 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l2433_243372


namespace NUMINAMATH_CALUDE_A_power_100_l2433_243335

def A : Matrix (Fin 2) (Fin 2) ℤ := !![4, 1; -9, -2]

theorem A_power_100 : A ^ 100 = !![301, 100; -900, -299] := by sorry

end NUMINAMATH_CALUDE_A_power_100_l2433_243335


namespace NUMINAMATH_CALUDE_expand_expression_l2433_243338

theorem expand_expression (x y : ℝ) : 12 * (3 * x - 4 * y + 2) = 36 * x - 48 * y + 24 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l2433_243338


namespace NUMINAMATH_CALUDE_trig_abs_sum_diff_ge_one_l2433_243353

theorem trig_abs_sum_diff_ge_one (x : ℝ) : 
  max (|Real.cos x - Real.sin x|) (|Real.sin x + Real.cos x|) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_trig_abs_sum_diff_ge_one_l2433_243353


namespace NUMINAMATH_CALUDE_explorer_findings_l2433_243390

/-- Converts a number from base 6 to base 10 -/
def base6ToBase10 (n : ℕ) : ℕ := sorry

/-- The total value of the explorer's findings -/
def totalValue : ℕ :=
  base6ToBase10 1524 + base6ToBase10 305 + base6ToBase10 1432

theorem explorer_findings :
  totalValue = 905 := by sorry

end NUMINAMATH_CALUDE_explorer_findings_l2433_243390


namespace NUMINAMATH_CALUDE_arithmetic_sequence_unique_l2433_243394

/-- Represents a sequence of five natural numbers with a constant difference -/
structure ArithmeticSequence :=
  (first : ℕ)
  (diff : ℕ)

/-- Converts a natural number to a string representation -/
def toLetterRepresentation (n : ℕ) : String :=
  match n with
  | 5 => "T"
  | 12 => "EL"
  | 19 => "EK"
  | 26 => "LA"
  | 33 => "SS"
  | _ => ""

/-- The main theorem to be proved -/
theorem arithmetic_sequence_unique :
  ∀ (seq : ArithmeticSequence),
    (seq.first = 5 ∧ seq.diff = 7) ↔
    (toLetterRepresentation seq.first = "T" ∧
     toLetterRepresentation (seq.first + seq.diff) = "EL" ∧
     toLetterRepresentation (seq.first + 2 * seq.diff) = "EK" ∧
     toLetterRepresentation (seq.first + 3 * seq.diff) = "LA" ∧
     toLetterRepresentation (seq.first + 4 * seq.diff) = "SS") :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_unique_l2433_243394


namespace NUMINAMATH_CALUDE_circle_and_line_equations_l2433_243359

-- Define the circle C
def circle_C (center : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = 4}

-- Define the tangent line
def tangent_line : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | 3 * p.1 - 4 * p.2 + 4 = 0}

-- Define the line l
def line_l (k : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = k * p.1 - 3}

-- Define the dot product of two vectors
def dot_product (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

theorem circle_and_line_equations :
  ∃ (center : ℝ × ℝ) (k : ℝ),
    center.1 > 0 ∧ center.2 = 0 ∧
    (∃ (p : ℝ × ℝ), p ∈ circle_C center ∩ tangent_line) ∧
    (∃ (A B : ℝ × ℝ), A ≠ B ∧ A ∈ circle_C center ∩ line_l k ∧ B ∈ circle_C center ∩ line_l k) ∧
    (∀ (A B : ℝ × ℝ), A ∈ circle_C center ∩ line_l k → B ∈ circle_C center ∩ line_l k → dot_product A B = 3) →
    center = (2, 0) ∧ k = 1 := by
  sorry

#check circle_and_line_equations

end NUMINAMATH_CALUDE_circle_and_line_equations_l2433_243359


namespace NUMINAMATH_CALUDE_better_fit_larger_R_squared_l2433_243370

-- Define the correlation index R²
def correlation_index (R : ℝ) : Prop := 0 ≤ R ∧ R ≤ 1

-- Define the concept of model fit
def model_fit (fit : ℝ) : Prop := 0 ≤ fit

-- Theorem stating that a larger R² indicates a better model fit
theorem better_fit_larger_R_squared 
  (R1 R2 fit1 fit2 : ℝ) 
  (h1 : correlation_index R1) 
  (h2 : correlation_index R2) 
  (h3 : model_fit fit1) 
  (h4 : model_fit fit2) 
  (h5 : R1 < R2) : 
  fit1 < fit2 := by
sorry


end NUMINAMATH_CALUDE_better_fit_larger_R_squared_l2433_243370


namespace NUMINAMATH_CALUDE_f_shifted_positive_set_l2433_243351

/-- An odd function f defined on ℝ satisfying f(x) = 2^x - 4 for x > 0 -/
noncomputable def f : ℝ → ℝ :=
  fun x => if x > 0 then 2^x - 4 else -(2^(-x) - 4)

/-- f is an odd function -/
axiom f_odd : ∀ x, f (-x) = -f x

/-- f(x) = 2^x - 4 for x > 0 -/
axiom f_pos : ∀ x, x > 0 → f x = 2^x - 4

theorem f_shifted_positive_set :
  {x : ℝ | f (x - 1) > 0} = {x : ℝ | -1 < x ∧ x < 1 ∨ x > 3} :=
sorry

end NUMINAMATH_CALUDE_f_shifted_positive_set_l2433_243351


namespace NUMINAMATH_CALUDE_sin_cos_sum_fifteen_seventyfive_l2433_243341

theorem sin_cos_sum_fifteen_seventyfive : 
  Real.sin (15 * π / 180) * Real.cos (75 * π / 180) + 
  Real.cos (15 * π / 180) * Real.sin (75 * π / 180) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_sum_fifteen_seventyfive_l2433_243341


namespace NUMINAMATH_CALUDE_donation_relationship_l2433_243399

/-- Represents the relationship between the number of girls and the total donation in a class. -/
def donation_function (x : ℕ) : ℝ :=
  -5 * x + 1125

/-- Theorem stating the relationship between the number of girls and the total donation. -/
theorem donation_relationship (x : ℕ) (y : ℝ) 
  (h1 : x ≤ 45)  -- Ensure the number of girls is not more than the total number of students
  (h2 : y = 20 * x + 25 * (45 - x)) :  -- Total donation calculation
  y = donation_function x :=
by
  sorry

#check donation_relationship

end NUMINAMATH_CALUDE_donation_relationship_l2433_243399


namespace NUMINAMATH_CALUDE_consecutive_integers_square_sum_l2433_243329

theorem consecutive_integers_square_sum : 
  ∃ (n : ℤ), 
    (n + 1)^2 + (n + 2)^2 = (n - 2)^2 + (n - 1)^2 + n^2 ∧
    n = 12 :=
by sorry

end NUMINAMATH_CALUDE_consecutive_integers_square_sum_l2433_243329


namespace NUMINAMATH_CALUDE_max_pieces_is_seven_l2433_243354

/-- Represents a mapping of letters to digits -/
def LetterDigitMap := Char → Nat

/-- Checks if a mapping is valid (each letter maps to a unique digit) -/
def is_valid_mapping (m : LetterDigitMap) : Prop :=
  ∀ c1 c2, c1 ≠ c2 → m c1 ≠ m c2

/-- Converts a string to a number using the given mapping -/
def string_to_number (s : String) (m : LetterDigitMap) : Nat :=
  s.foldl (fun acc c => acc * 10 + m c) 0

/-- Represents the equation PIE = n * PIECE -/
def satisfies_equation (pie : String) (piece : String) (n : Nat) (m : LetterDigitMap) : Prop :=
  string_to_number pie m = n * string_to_number piece m

theorem max_pieces_is_seven :
  ∃ (pie piece : String) (m : LetterDigitMap),
    pie.length = 5 ∧
    piece.length = 5 ∧
    is_valid_mapping m ∧
    satisfies_equation pie piece 7 m ∧
    (∀ (pie' piece' : String) (m' : LetterDigitMap) (n : Nat),
      pie'.length = 5 →
      piece'.length = 5 →
      is_valid_mapping m' →
      satisfies_equation pie' piece' n m' →
      n ≤ 7) :=
sorry

end NUMINAMATH_CALUDE_max_pieces_is_seven_l2433_243354


namespace NUMINAMATH_CALUDE_sum_of_min_max_x_l2433_243321

theorem sum_of_min_max_x (x y z : ℝ) (h1 : x + y + z = 5) (h2 : x^2 + y^2 + z^2 = 11) :
  ∃ (m M : ℝ), (∀ x', ∃ y' z', x' + y' + z' = 5 ∧ x'^2 + y'^2 + z'^2 = 11 → m ≤ x' ∧ x' ≤ M) ∧
                m + M = 8/3 :=
sorry

end NUMINAMATH_CALUDE_sum_of_min_max_x_l2433_243321


namespace NUMINAMATH_CALUDE_baseball_card_value_decrease_l2433_243366

theorem baseball_card_value_decrease :
  ∀ (initial_value : ℝ), initial_value > 0 →
  let value_after_first_year := initial_value * (1 - 0.2)
  let value_after_second_year := value_after_first_year * (1 - 0.2)
  let total_decrease := initial_value - value_after_second_year
  let percent_decrease := (total_decrease / initial_value) * 100
  percent_decrease = 36 := by
sorry

end NUMINAMATH_CALUDE_baseball_card_value_decrease_l2433_243366


namespace NUMINAMATH_CALUDE_trigonometric_identities_l2433_243337

theorem trigonometric_identities (θ : ℝ) (h : Real.sin (θ - π/3) = 1/3) :
  (Real.sin (θ + 2*π/3) = -1/3) ∧ (Real.cos (θ - 5*π/6) = 1/3) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identities_l2433_243337


namespace NUMINAMATH_CALUDE_principal_is_7500_l2433_243339

/-- Calculates the compound interest amount -/
def compound_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate) ^ time - principal

/-- Proves that the principal is 7500 given the conditions -/
theorem principal_is_7500 
  (rate : ℝ) 
  (time : ℕ) 
  (interest : ℝ) 
  (h_rate : rate = 0.04) 
  (h_time : time = 2) 
  (h_interest : interest = 612) : 
  ∃ (principal : ℝ), principal = 7500 ∧ compound_interest principal rate time = interest :=
sorry

end NUMINAMATH_CALUDE_principal_is_7500_l2433_243339


namespace NUMINAMATH_CALUDE_factorial_ratio_sum_l2433_243386

theorem factorial_ratio_sum (p q : ℕ) : 
  p < 10 → q < 10 → p > 0 → q > 0 → (840 : ℕ) = p! / q! → p + q = 10 := by
  sorry

end NUMINAMATH_CALUDE_factorial_ratio_sum_l2433_243386


namespace NUMINAMATH_CALUDE_sum_60_is_neg_300_l2433_243398

/-- An arithmetic progression with specific properties -/
structure ArithmeticProgression where
  /-- The first term of the progression -/
  a : ℚ
  /-- The common difference of the progression -/
  d : ℚ
  /-- The sum of the first 15 terms is 150 -/
  sum_15 : (15 : ℚ) / 2 * (2 * a + 14 * d) = 150
  /-- The sum of the first 45 terms is 0 -/
  sum_45 : (45 : ℚ) / 2 * (2 * a + 44 * d) = 0

/-- The sum of the first 60 terms of the arithmetic progression is -300 -/
theorem sum_60_is_neg_300 (ap : ArithmeticProgression) :
  (60 : ℚ) / 2 * (2 * ap.a + 59 * ap.d) = -300 := by
  sorry


end NUMINAMATH_CALUDE_sum_60_is_neg_300_l2433_243398


namespace NUMINAMATH_CALUDE_problem_solution_l2433_243383

theorem problem_solution (a : ℝ) (f g : ℝ → ℝ) 
  (h1 : a > 0) 
  (h2 : f (g a) = 18)
  (h3 : ∀ x, f x = x^2 - 2)
  (h4 : ∀ x, g x = x^2 + 6) : 
  a = Real.sqrt 14 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l2433_243383


namespace NUMINAMATH_CALUDE_day_in_consecutive_years_l2433_243305

/-- Represents days of the week -/
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
  number : ℕ
  is_leap : Bool

/-- Function to get the day of the week for a given day number in a year -/
def day_of_week (y : Year) (day_number : ℕ) : DayOfWeek :=
  sorry

/-- Function to check if a given day number is a Friday -/
def is_friday (y : Year) (day_number : ℕ) : Prop :=
  day_of_week y day_number = DayOfWeek.Friday

/-- Theorem stating the relationship between the days in consecutive years -/
theorem day_in_consecutive_years 
  (n : ℕ) 
  (year_n : Year)
  (year_n_plus_1 : Year)
  (year_n_minus_1 : Year)
  (h1 : year_n.number = n)
  (h2 : year_n_plus_1.number = n + 1)
  (h3 : year_n_minus_1.number = n - 1)
  (h4 : is_friday year_n 250)
  (h5 : is_friday year_n_plus_1 150) :
  day_of_week year_n_minus_1 50 = DayOfWeek.Monday :=
sorry

end NUMINAMATH_CALUDE_day_in_consecutive_years_l2433_243305


namespace NUMINAMATH_CALUDE_four_digit_numbers_divisible_by_13_l2433_243340

theorem four_digit_numbers_divisible_by_13 : 
  (Finset.filter (fun n => n % 13 = 0) (Finset.range 9000)).card + 1 = 693 := by
  sorry

end NUMINAMATH_CALUDE_four_digit_numbers_divisible_by_13_l2433_243340


namespace NUMINAMATH_CALUDE_positive_solution_quadratic_equation_l2433_243387

theorem positive_solution_quadratic_equation :
  ∃ x : ℝ, x > 0 ∧ 
  (1/3) * (4 * x^2 - 2) = (x^2 - 75*x - 15) * (x^2 + 50*x + 10) ∧
  x = (75 + Real.sqrt 5693) / 2 := by
  sorry

end NUMINAMATH_CALUDE_positive_solution_quadratic_equation_l2433_243387


namespace NUMINAMATH_CALUDE_parallel_non_coincident_lines_l2433_243379

/-- Two lines are parallel if and only if their slopes are equal -/
axiom parallel_lines_equal_slopes {m1 m2 b1 b2 : ℝ} :
  (∀ x y : ℝ, y = m1 * x + b1 ↔ y = m2 * x + b2) ↔ m1 = m2

/-- Two lines are distinct if and only if their y-intercepts are different -/
axiom distinct_lines_different_intercepts {m b1 b2 : ℝ} :
  (∃ x y : ℝ, y = m * x + b1 ∧ y ≠ m * x + b2) ↔ b1 ≠ b2

theorem parallel_non_coincident_lines (a : ℝ) :
  (∀ x y : ℝ, a * x + 2 * y + 6 = 0 ↔ y = -a/2 * x - 3) ∧
  (∀ x y : ℝ, x + (a - 1) * y + (a^2 - 1) = 0 ↔ y = -1/(a-1) * x - (a^2-1)/(a-1)) ∧
  (∀ x y : ℝ, y = -a/2 * x - 3 ↔ y = -1/(a-1) * x - (a^2-1)/(a-1)) ∧
  (∃ x y : ℝ, y = -a/2 * x - 3 ∧ y ≠ -1/(a-1) * x - (a^2-1)/(a-1)) →
  a = -1 := by sorry

end NUMINAMATH_CALUDE_parallel_non_coincident_lines_l2433_243379


namespace NUMINAMATH_CALUDE_inequality_proof_l2433_243320

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (sum_one : a + b + c = 1) :
  (a/b + a/c + b/a + b/c + c/a + c/b + 6) ≥ 2 * Real.sqrt 2 * (Real.sqrt ((1-a)/a) + Real.sqrt ((1-b)/b) + Real.sqrt ((1-c)/c)) ∧
  ((a/b + a/c + b/a + b/c + c/a + c/b + 6) = 2 * Real.sqrt 2 * (Real.sqrt ((1-a)/a) + Real.sqrt ((1-b)/b) + Real.sqrt ((1-c)/c)) ↔ a = 1/3 ∧ b = 1/3 ∧ c = 1/3) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l2433_243320


namespace NUMINAMATH_CALUDE_greatest_two_digit_with_product_12_l2433_243309

def is_valid_digit (d : ℕ) : Prop := 1 ≤ d ∧ d ≤ 9

def digits_product (n : ℕ) : ℕ := (n / 10) * (n % 10)

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

theorem greatest_two_digit_with_product_12 :
  ∃ (n : ℕ), is_two_digit n ∧ 
             digits_product n = 12 ∧
             (∀ (m : ℕ), is_two_digit m → digits_product m = 12 → m ≤ n) ∧
             n = 62 :=
by sorry

end NUMINAMATH_CALUDE_greatest_two_digit_with_product_12_l2433_243309


namespace NUMINAMATH_CALUDE_max_value_sqrt_sum_l2433_243330

theorem max_value_sqrt_sum (x : ℝ) (h : -49 ≤ x ∧ x ≤ 49) :
  Real.sqrt (49 + x) + Real.sqrt (49 - x) ≤ 14 ∧
  ∃ y : ℝ, -49 ≤ y ∧ y ≤ 49 ∧ Real.sqrt (49 + y) + Real.sqrt (49 - y) = 14 :=
by sorry

end NUMINAMATH_CALUDE_max_value_sqrt_sum_l2433_243330


namespace NUMINAMATH_CALUDE_line_inclination_angle_l2433_243375

/-- The angle of inclination of the line x - √3y + 6 = 0 is 30°. -/
theorem line_inclination_angle (x y : ℝ) :
  x - Real.sqrt 3 * y + 6 = 0 →
  Real.arctan (Real.sqrt 3 / 3) = 30 * π / 180 :=
by sorry

end NUMINAMATH_CALUDE_line_inclination_angle_l2433_243375


namespace NUMINAMATH_CALUDE_chocolate_doughnut_cost_l2433_243368

/-- The cost of a chocolate doughnut given the number of students wanting each type,
    the cost of glazed doughnuts, and the total cost. -/
theorem chocolate_doughnut_cost
  (chocolate_students : ℕ)
  (glazed_students : ℕ)
  (glazed_cost : ℚ)
  (total_cost : ℚ)
  (h1 : chocolate_students = 10)
  (h2 : glazed_students = 15)
  (h3 : glazed_cost = 1)
  (h4 : total_cost = 35) :
  ∃ (chocolate_cost : ℚ),
    chocolate_cost * chocolate_students + glazed_cost * glazed_students = total_cost ∧
    chocolate_cost = 2 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_doughnut_cost_l2433_243368


namespace NUMINAMATH_CALUDE_least_number_divisibility_l2433_243377

theorem least_number_divisibility (x : ℕ) : x = 10315 ↔ 
  (∀ y : ℕ, y < x → ¬((1024 + y) % (17 * 23 * 29) = 0)) ∧ 
  ((1024 + x) % (17 * 23 * 29) = 0) := by
  sorry

end NUMINAMATH_CALUDE_least_number_divisibility_l2433_243377


namespace NUMINAMATH_CALUDE_ratio_problem_l2433_243350

theorem ratio_problem (a b c P : ℝ) 
  (h1 : b / (a + c) = 1 / 2)
  (h2 : a / (b + c) = 1 / P) :
  (a + b + c) / a = 4 := by
sorry

end NUMINAMATH_CALUDE_ratio_problem_l2433_243350


namespace NUMINAMATH_CALUDE_chocolate_bar_reduction_l2433_243318

theorem chocolate_bar_reduction 
  (m n : ℕ) 
  (h_lt : m < n) 
  (a b : ℕ) 
  (h_div_a : n^5 ∣ a) 
  (h_div_b : n^5 ∣ b) : 
  ∃ (x y : ℕ), 
    x ≤ a ∧ 
    y ≤ b ∧ 
    x * y = a * b * (m / n)^10 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_bar_reduction_l2433_243318


namespace NUMINAMATH_CALUDE_prob_at_most_sixes_equals_sum_l2433_243380

def numDice : ℕ := 10
def maxSixes : ℕ := 3

def probExactlySixes (n : ℕ) : ℚ :=
  (Nat.choose numDice n) * (1/6)^n * (5/6)^(numDice - n)

def probAtMostSixes : ℚ :=
  (Finset.range (maxSixes + 1)).sum probExactlySixes

theorem prob_at_most_sixes_equals_sum :
  probAtMostSixes = 
    probExactlySixes 0 + probExactlySixes 1 + 
    probExactlySixes 2 + probExactlySixes 3 := by
  sorry

end NUMINAMATH_CALUDE_prob_at_most_sixes_equals_sum_l2433_243380


namespace NUMINAMATH_CALUDE_canoe_kayak_ratio_l2433_243300

/-- Represents the rental business scenario -/
structure RentalBusiness where
  canoe_cost : ℕ
  kayak_cost : ℕ
  total_revenue : ℕ
  canoe_count : ℕ
  kayak_count : ℕ

/-- Theorem stating the ratio of canoes to kayaks is 3:1 given the conditions -/
theorem canoe_kayak_ratio (rb : RentalBusiness) :
  rb.canoe_cost = 14 →
  rb.kayak_cost = 15 →
  rb.total_revenue = 288 →
  rb.canoe_count = rb.kayak_count + 4 →
  rb.canoe_count = 3 * rb.kayak_count →
  rb.canoe_count / rb.kayak_count = 3 := by
  sorry


end NUMINAMATH_CALUDE_canoe_kayak_ratio_l2433_243300


namespace NUMINAMATH_CALUDE_missing_legos_l2433_243304

theorem missing_legos (total : ℕ) (in_box : ℕ) : 
  total = 500 → in_box = 245 → (total / 2 - in_box : ℤ) = 5 := by
  sorry

end NUMINAMATH_CALUDE_missing_legos_l2433_243304


namespace NUMINAMATH_CALUDE_buns_eaten_proof_l2433_243346

/-- Represents the number of buns eaten by Zhenya -/
def zhenya_buns : ℕ := 40

/-- Represents the number of buns eaten by Sasha -/
def sasha_buns : ℕ := 30

/-- The total number of buns eaten -/
def total_buns : ℕ := 70

/-- The total eating time in minutes -/
def total_time : ℕ := 180

/-- Zhenya's eating rate in buns per minute -/
def zhenya_rate : ℚ := 1/2

/-- Sasha's eating rate in buns per minute -/
def sasha_rate : ℚ := 3/10

theorem buns_eaten_proof :
  zhenya_buns + sasha_buns = total_buns ∧
  zhenya_rate * total_time = zhenya_buns ∧
  sasha_rate * total_time = sasha_buns :=
by sorry

#check buns_eaten_proof

end NUMINAMATH_CALUDE_buns_eaten_proof_l2433_243346


namespace NUMINAMATH_CALUDE_expression_simplification_l2433_243306

theorem expression_simplification (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let x := ((a * (b^(1/3))) / (b * (a^3)^(1/2)))^(3/2) + ((a^(1/2)) / (a * (b^3)^(1/8)))^2
  x / (a^(1/4) + b^(1/4)) = 1 / (a * b) := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2433_243306


namespace NUMINAMATH_CALUDE_delta_problem_l2433_243381

-- Define the delta operation
def delta (a b : ℕ) : ℕ := a^2 - b

-- State the theorem
theorem delta_problem : delta (5^(delta 7 2)) (4^(delta 3 8)) = 5^94 - 4 := by
  sorry

end NUMINAMATH_CALUDE_delta_problem_l2433_243381


namespace NUMINAMATH_CALUDE_sine_integral_negative_l2433_243323

theorem sine_integral_negative : ∫ x in -Real.pi..0, Real.sin x < 0 := by
  sorry

end NUMINAMATH_CALUDE_sine_integral_negative_l2433_243323


namespace NUMINAMATH_CALUDE_smallest_a_value_l2433_243314

theorem smallest_a_value (a b : ℤ) : 
  (a + 2 * b = 32) → 
  (abs a > 2) → 
  (∀ x : ℤ, x + 2 * b = 32 → abs x > 2 → x ≥ 4) → 
  (a = 4) → 
  (b = 14) := by
sorry

end NUMINAMATH_CALUDE_smallest_a_value_l2433_243314


namespace NUMINAMATH_CALUDE_closest_integer_to_cube_root_of_250_l2433_243361

theorem closest_integer_to_cube_root_of_250 : 
  ∃ (n : ℤ), n = 6 ∧ ∀ (m : ℤ), |m ^ 3 - 250| ≥ |n ^ 3 - 250| :=
sorry

end NUMINAMATH_CALUDE_closest_integer_to_cube_root_of_250_l2433_243361


namespace NUMINAMATH_CALUDE_price_reduction_rate_l2433_243310

theorem price_reduction_rate (original_price final_price : ℝ) 
  (h1 : original_price = 200)
  (h2 : final_price = 128)
  (h3 : ∃ x : ℝ, final_price = original_price * (1 - x)^2) :
  ∃ x : ℝ, final_price = original_price * (1 - x)^2 ∧ x = 0.2 :=
by sorry

end NUMINAMATH_CALUDE_price_reduction_rate_l2433_243310


namespace NUMINAMATH_CALUDE_polynomial_remainder_l2433_243348

theorem polynomial_remainder (x : ℝ) : 
  (x^11 + 1) % (x + 1) = 0 := by sorry

end NUMINAMATH_CALUDE_polynomial_remainder_l2433_243348


namespace NUMINAMATH_CALUDE_three_plane_division_l2433_243389

/-- The number of regions that n planes can divide 3-dimensional space into -/
def regions (n : ℕ) : ℕ := sorry

/-- The minimum number of regions that 3 planes can divide 3-dimensional space into -/
def min_regions : ℕ := regions 3

/-- The maximum number of regions that 3 planes can divide 3-dimensional space into -/
def max_regions : ℕ := regions 3

theorem three_plane_division :
  min_regions = 4 ∧ max_regions = 8 := by sorry

end NUMINAMATH_CALUDE_three_plane_division_l2433_243389


namespace NUMINAMATH_CALUDE_max_enclosure_area_l2433_243382

/-- The number of fence pieces --/
def num_pieces : ℕ := 15

/-- The length of each fence piece in meters --/
def piece_length : ℝ := 2

/-- The total length of fencing available in meters --/
def total_length : ℝ := num_pieces * piece_length

/-- The area of the rectangular enclosure as a function of its width --/
def area (w : ℝ) : ℝ := (total_length - 2 * w) * w

/-- The maximum area of the enclosure, rounded down to the nearest integer --/
def max_area : ℕ := 112

theorem max_enclosure_area :
  ∃ (w : ℝ), 0 < w ∧ w < total_length / 2 ∧
  (∀ (x : ℝ), 0 < x → x < total_length / 2 → area x ≤ area w) ∧
  ⌊area w⌋ = max_area :=
sorry

end NUMINAMATH_CALUDE_max_enclosure_area_l2433_243382


namespace NUMINAMATH_CALUDE_salary_increase_proof_l2433_243303

def new_salary : ℝ := 90000
def percent_increase : ℝ := 38.46153846153846

theorem salary_increase_proof :
  let old_salary := new_salary / (1 + percent_increase / 100)
  let increase := new_salary - old_salary
  increase = 25000 := by sorry

end NUMINAMATH_CALUDE_salary_increase_proof_l2433_243303


namespace NUMINAMATH_CALUDE_max_mondays_in_45_days_l2433_243362

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- The number of days we're considering -/
def days_considered : ℕ := 45

/-- The maximum number of Mondays in the first 45 days of a year -/
def max_mondays : ℕ := 7

/-- Theorem: The maximum number of Mondays in the first 45 days of a year is 7 -/
theorem max_mondays_in_45_days : 
  ∀ (start_day : ℕ), start_day < days_in_week →
  (∃ (monday_count : ℕ), 
    monday_count ≤ max_mondays ∧
    monday_count = (days_considered / days_in_week) + 
      (if start_day = 0 then 1 else 0)) :=
sorry

end NUMINAMATH_CALUDE_max_mondays_in_45_days_l2433_243362


namespace NUMINAMATH_CALUDE_parallel_vectors_tan_theta_l2433_243311

open Real

theorem parallel_vectors_tan_theta (θ : ℝ) : 
  let a : Fin 2 → ℝ := ![2, sin θ]
  let b : Fin 2 → ℝ := ![1, cos θ]
  (∃ (k : ℝ), k ≠ 0 ∧ a = k • b) → tan θ = 2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_tan_theta_l2433_243311


namespace NUMINAMATH_CALUDE_all_sides_equal_not_imply_rectangle_l2433_243313

-- Define a quadrilateral
structure Quadrilateral :=
  (a b c d : ℝ)  -- Side lengths
  (α β γ δ : ℝ)  -- Internal angles

-- Define a rectangle
def is_rectangle (q : Quadrilateral) : Prop :=
  q.α = q.β ∧ q.β = q.γ ∧ q.γ = q.δ ∧ q.δ = 90

-- Define a quadrilateral with all sides equal
def all_sides_equal (q : Quadrilateral) : Prop :=
  q.a = q.b ∧ q.b = q.c ∧ q.c = q.d

-- Theorem statement
theorem all_sides_equal_not_imply_rectangle :
  ∃ q : Quadrilateral, all_sides_equal q ∧ ¬(is_rectangle q) := by
  sorry


end NUMINAMATH_CALUDE_all_sides_equal_not_imply_rectangle_l2433_243313


namespace NUMINAMATH_CALUDE_shopkeeper_pricing_l2433_243392

/-- Proves that the original selling price is 800 given the conditions of the problem -/
theorem shopkeeper_pricing (cost_price : ℝ) : 
  (1.25 * cost_price = 800) ∧ (0.8 * cost_price = 512) := by
  sorry

#check shopkeeper_pricing

end NUMINAMATH_CALUDE_shopkeeper_pricing_l2433_243392


namespace NUMINAMATH_CALUDE_max_servings_emily_l2433_243334

/-- Represents the recipe requirements for 4 servings --/
structure Recipe where
  bananas : ℕ
  strawberries : ℕ
  yogurt : ℕ

/-- Represents Emily's available ingredients --/
structure Available where
  bananas : ℕ
  strawberries : ℕ
  yogurt : ℕ

/-- Calculates the maximum number of servings that can be made --/
def max_servings (recipe : Recipe) (available : Available) : ℕ :=
  min
    (available.bananas * 4 / recipe.bananas)
    (min
      (available.strawberries * 4 / recipe.strawberries)
      (available.yogurt * 4 / recipe.yogurt))

theorem max_servings_emily :
  let recipe := Recipe.mk 3 1 2
  let available := Available.mk 10 3 12
  max_servings recipe available = 12 := by
  sorry

end NUMINAMATH_CALUDE_max_servings_emily_l2433_243334


namespace NUMINAMATH_CALUDE_seventh_diagram_shaded_fraction_l2433_243373

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

/-- Total number of triangles in the nth diagram -/
def total_triangles (n : ℕ) : ℕ := Nat.factorial n

/-- Fraction of shaded triangles in the nth diagram -/
def shaded_fraction (n : ℕ) : ℚ :=
  (fib n : ℚ) / (total_triangles n : ℚ)

/-- The main theorem -/
theorem seventh_diagram_shaded_fraction :
  shaded_fraction 7 = 13 / 5040 := by
  sorry

end NUMINAMATH_CALUDE_seventh_diagram_shaded_fraction_l2433_243373


namespace NUMINAMATH_CALUDE_cube_root_sum_zero_implies_opposite_l2433_243336

theorem cube_root_sum_zero_implies_opposite (x y : ℝ) : 
  (x^(1/3 : ℝ) + y^(1/3 : ℝ) = 0) → (x = -y) := by
  sorry

end NUMINAMATH_CALUDE_cube_root_sum_zero_implies_opposite_l2433_243336


namespace NUMINAMATH_CALUDE_blue_hat_cost_is_6_l2433_243371

-- Define the total number of hats
def total_hats : ℕ := 85

-- Define the cost of each green hat
def green_hat_cost : ℕ := 7

-- Define the total price
def total_price : ℕ := 540

-- Define the number of green hats
def green_hats : ℕ := 30

-- Define the number of blue hats
def blue_hats : ℕ := total_hats - green_hats

-- Define the cost of green hats
def green_hats_cost : ℕ := green_hats * green_hat_cost

-- Define the cost of blue hats
def blue_hats_cost : ℕ := total_price - green_hats_cost

-- Theorem: The cost of each blue hat is $6
theorem blue_hat_cost_is_6 : blue_hats_cost / blue_hats = 6 := by
  sorry

end NUMINAMATH_CALUDE_blue_hat_cost_is_6_l2433_243371


namespace NUMINAMATH_CALUDE_sam_remaining_money_l2433_243307

/-- Given an initial amount, number of books, and cost per book, 
    calculate the remaining amount after purchase. -/
def remaining_amount (initial : ℕ) (num_books : ℕ) (cost_per_book : ℕ) : ℕ :=
  initial - (num_books * cost_per_book)

/-- Theorem stating that given the specific conditions of Sam's purchase,
    the remaining amount is 16 dollars. -/
theorem sam_remaining_money :
  remaining_amount 79 9 7 = 16 := by
  sorry

end NUMINAMATH_CALUDE_sam_remaining_money_l2433_243307


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l2433_243374

theorem negation_of_universal_proposition :
  (¬ (∀ x : ℝ, x > Real.sin x)) ↔ (∃ x : ℝ, x ≤ Real.sin x) := by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l2433_243374


namespace NUMINAMATH_CALUDE_sin_value_for_special_angle_l2433_243367

theorem sin_value_for_special_angle (θ : Real) 
  (h1 : 0 < θ) (h2 : θ < Real.pi / 2) 
  (h3 : 1 + Real.sin θ = 2 * Real.cos θ) : 
  Real.sin θ = 3/5 := by
sorry

end NUMINAMATH_CALUDE_sin_value_for_special_angle_l2433_243367


namespace NUMINAMATH_CALUDE_midpoint_ratio_range_l2433_243352

-- Define the lines
def line1 (x y : ℝ) : Prop := x + 2*y - 1 = 0
def line2 (x y : ℝ) : Prop := x + 2*y + 3 = 0

-- Define the midpoint condition
def is_midpoint (x₀ y₀ xp yp xq yq : ℝ) : Prop :=
  x₀ = (xp + xq) / 2 ∧ y₀ = (yp + yq) / 2

-- Main theorem
theorem midpoint_ratio_range 
  (P : ℝ × ℝ) (A : ℝ × ℝ) (Q : ℝ × ℝ) (x₀ y₀ : ℝ)
  (h1 : line1 P.1 P.2)
  (h2 : line2 A.1 A.2)
  (h3 : is_midpoint x₀ y₀ P.1 P.2 Q.1 Q.2)
  (h4 : y₀ > x₀ + 2) :
  -1/2 < y₀/x₀ ∧ y₀/x₀ < -1/5 :=
sorry

end NUMINAMATH_CALUDE_midpoint_ratio_range_l2433_243352


namespace NUMINAMATH_CALUDE_all_analogies_correct_correct_analogies_count_l2433_243376

-- Define the structure for a hyperbola
structure Hyperbola where
  focal_length : ℝ
  real_axis_length : ℝ
  eccentricity : ℝ

-- Define the structure for an ellipse
structure Ellipse where
  focal_length : ℝ
  major_axis_length : ℝ
  eccentricity : ℝ

-- Define the structure for an arithmetic sequence
structure ArithmeticSequence where
  first_term : ℝ
  second_term : ℝ
  third_term : ℝ

-- Define the structure for a geometric sequence
structure GeometricSequence where
  first_term : ℝ
  second_term : ℝ
  third_term : ℝ

-- Define the structure for an equilateral triangle
structure EquilateralTriangle where
  side_length : ℝ
  area : ℝ

-- Define the structure for a regular tetrahedron
structure RegularTetrahedron where
  edge_length : ℝ
  volume : ℝ

def analogy1_correct (h : Hyperbola) (e : Ellipse) : Prop :=
  (h.focal_length = 2 * h.real_axis_length → h.eccentricity = 2) →
  (e.focal_length = 1/2 * e.major_axis_length → e.eccentricity = 1/2)

def analogy2_correct (a : ArithmeticSequence) (g : GeometricSequence) : Prop :=
  (a.first_term + a.second_term + a.third_term = 1 → a.second_term = 1/3) →
  (g.first_term * g.second_term * g.third_term = 1 → g.second_term = 1)

def analogy3_correct (t1 t2 : EquilateralTriangle) (tet1 tet2 : RegularTetrahedron) : Prop :=
  (t2.side_length = 2 * t1.side_length → t2.area = 4 * t1.area) →
  (tet2.edge_length = 2 * tet1.edge_length → tet2.volume = 8 * tet1.volume)

theorem all_analogies_correct 
  (h : Hyperbola) (e : Ellipse) 
  (a : ArithmeticSequence) (g : GeometricSequence)
  (t1 t2 : EquilateralTriangle) (tet1 tet2 : RegularTetrahedron) : 
  analogy1_correct h e ∧ analogy2_correct a g ∧ analogy3_correct t1 t2 tet1 tet2 := by
  sorry

theorem correct_analogies_count : ∃ (n : ℕ), n = 3 ∧ 
  ∀ (h : Hyperbola) (e : Ellipse) 
     (a : ArithmeticSequence) (g : GeometricSequence)
     (t1 t2 : EquilateralTriangle) (tet1 tet2 : RegularTetrahedron),
  (analogy1_correct h e → n ≥ 1) ∧
  (analogy2_correct a g → n ≥ 2) ∧
  (analogy3_correct t1 t2 tet1 tet2 → n = 3) := by
  sorry

end NUMINAMATH_CALUDE_all_analogies_correct_correct_analogies_count_l2433_243376


namespace NUMINAMATH_CALUDE_circle_equation_l2433_243342

theorem circle_equation 
  (a b : ℝ) 
  (h1 : a > 0 ∧ b > 0)  -- Center in first quadrant
  (h2 : 2 * a - b + 1 = 0)  -- Center on the line 2x - y + 1 = 0
  (h3 : (a + 4)^2 + (b - 3)^2 = 5^2)  -- Passes through (-4, 3) with radius 5
  : ∀ x y : ℝ, (x - 1)^2 + (y - 3)^2 = 25 ↔ (x - a)^2 + (y - b)^2 = 5^2 := by
sorry

end NUMINAMATH_CALUDE_circle_equation_l2433_243342


namespace NUMINAMATH_CALUDE_friends_recycled_sixteen_pounds_l2433_243316

/-- Represents the recycling scenario -/
structure RecyclingScenario where
  pounds_per_point : ℕ
  vanessa_pounds : ℕ
  total_points : ℕ

/-- Calculates the amount of paper recycled by Vanessa's friends -/
def friends_recycled_pounds (scenario : RecyclingScenario) : ℕ :=
  scenario.total_points * scenario.pounds_per_point - scenario.vanessa_pounds

/-- Theorem stating that Vanessa's friends recycled 16 pounds -/
theorem friends_recycled_sixteen_pounds :
  ∃ (scenario : RecyclingScenario),
    scenario.pounds_per_point = 9 ∧
    scenario.vanessa_pounds = 20 ∧
    scenario.total_points = 4 ∧
    friends_recycled_pounds scenario = 16 := by
  sorry


end NUMINAMATH_CALUDE_friends_recycled_sixteen_pounds_l2433_243316


namespace NUMINAMATH_CALUDE_units_digit_sum_base8_l2433_243358

/-- The units digit of the sum of two numbers in base 8 -/
def unitsDigitBase8 (a b : ℕ) : ℕ :=
  (a + b) % 8

/-- 35 in base 8 -/
def a : ℕ := 3 * 8 + 5

/-- 47 in base 8 -/
def b : ℕ := 4 * 8 + 7

theorem units_digit_sum_base8 :
  unitsDigitBase8 a b = 4 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_sum_base8_l2433_243358


namespace NUMINAMATH_CALUDE_triangle_max_sum_l2433_243328

theorem triangle_max_sum (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π →
  A + B + C = π →
  a = 3 →
  1 + (Real.tan A) / (Real.tan B) = 2 * c / b →
  a^2 = b^2 + c^2 - 2 * b * c * Real.cos A →
  b > 0 ∧ c > 0 →
  (∀ b' c' : ℝ, b' > 0 ∧ c' > 0 →
    a^2 = b'^2 + c'^2 - 2 * b' * c' * Real.cos A →
    b' + c' ≤ b + c) →
  b + c = 3 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_triangle_max_sum_l2433_243328


namespace NUMINAMATH_CALUDE_intersection_and_complement_when_m_is_3_intersection_equals_B_iff_m_in_range_l2433_243360

-- Define sets A and B
def A : Set ℝ := {x | 1 ≤ x ∧ x ≤ 4}
def B (m : ℝ) : Set ℝ := {x | m ≤ x ∧ x ≤ m + 1}

-- Theorem 1
theorem intersection_and_complement_when_m_is_3 :
  (A ∩ B 3 = {x | 3 ≤ x ∧ x ≤ 4}) ∧
  (A ∩ (Set.univ \ B 3) = {x | 1 ≤ x ∧ x < 3}) := by sorry

-- Theorem 2
theorem intersection_equals_B_iff_m_in_range :
  ∀ m : ℝ, A ∩ B m = B m ↔ 1 ≤ m ∧ m ≤ 3 := by sorry

end NUMINAMATH_CALUDE_intersection_and_complement_when_m_is_3_intersection_equals_B_iff_m_in_range_l2433_243360


namespace NUMINAMATH_CALUDE_smallest_constant_for_sum_squares_inequality_l2433_243369

theorem smallest_constant_for_sum_squares_inequality :
  ∃ k : ℝ, k > 0 ∧
  (∀ y₁ y₂ y₃ A : ℝ,
    y₁ + y₂ + y₃ = 0 →
    A = max (abs y₁) (max (abs y₂) (abs y₃)) →
    y₁^2 + y₂^2 + y₃^2 ≥ k * A^2) ∧
  (∀ k' : ℝ, k' < k →
    ∃ y₁ y₂ y₃ A : ℝ,
      y₁ + y₂ + y₃ = 0 ∧
      A = max (abs y₁) (max (abs y₂) (abs y₃)) ∧
      y₁^2 + y₂^2 + y₃^2 < k' * A^2) ∧
  k = 1.5 := by
sorry

end NUMINAMATH_CALUDE_smallest_constant_for_sum_squares_inequality_l2433_243369


namespace NUMINAMATH_CALUDE_inequality_and_equality_condition_l2433_243397

theorem inequality_and_equality_condition (a b c d : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) (pos_d : 0 < d)
  (sum_one : a + b + c + d = 1) :
  (b * c * d / (1 - a)^2) + (a * c * d / (1 - b)^2) + 
  (a * b * d / (1 - c)^2) + (a * b * c / (1 - d)^2) ≤ 1/9 ∧
  ((b * c * d / (1 - a)^2) + (a * c * d / (1 - b)^2) + 
   (a * b * d / (1 - c)^2) + (a * b * c / (1 - d)^2) = 1/9 ↔ 
   a = 1/4 ∧ b = 1/4 ∧ c = 1/4 ∧ d = 1/4) :=
by sorry

end NUMINAMATH_CALUDE_inequality_and_equality_condition_l2433_243397


namespace NUMINAMATH_CALUDE_population_growth_l2433_243326

theorem population_growth (initial_population : ℝ) : 
  (initial_population * (1 + 0.1)^2 = 16940) → initial_population = 14000 := by
  sorry

end NUMINAMATH_CALUDE_population_growth_l2433_243326


namespace NUMINAMATH_CALUDE_intersection_sum_l2433_243325

-- Define the sets A, B, and C
def A : Set ℝ := {x : ℝ | -5 < x ∧ x < 5}
def B (a : ℝ) : Set ℝ := {x : ℝ | -7 < x ∧ x < a}
def C (b : ℝ) : Set ℝ := {x : ℝ | b < x ∧ x < 2}

-- State the theorem
theorem intersection_sum (a b : ℝ) (h : A ∩ B a = C b) : a + b = -3 := by
  sorry

end NUMINAMATH_CALUDE_intersection_sum_l2433_243325


namespace NUMINAMATH_CALUDE_coin_flip_configurations_l2433_243345

theorem coin_flip_configurations (n : ℕ) (h : n = 10) : 
  (Finset.range n).card + (n.choose 2) = 46 := by
  sorry

end NUMINAMATH_CALUDE_coin_flip_configurations_l2433_243345


namespace NUMINAMATH_CALUDE_laptop_price_difference_l2433_243347

/-- The price difference between two stores for Laptop Y -/
theorem laptop_price_difference
  (list_price : ℝ)
  (gadget_gurus_discount_percent : ℝ)
  (tech_trends_discount_amount : ℝ)
  (h1 : list_price = 300)
  (h2 : gadget_gurus_discount_percent = 0.15)
  (h3 : tech_trends_discount_amount = 45) :
  list_price * (1 - gadget_gurus_discount_percent) = list_price - tech_trends_discount_amount :=
by sorry

end NUMINAMATH_CALUDE_laptop_price_difference_l2433_243347


namespace NUMINAMATH_CALUDE_gift_contribution_total_l2433_243301

/-- Proves that the total contribution is $20 given the specified conditions -/
theorem gift_contribution_total (n : ℕ) (min_contribution max_contribution : ℝ) :
  n = 10 →
  min_contribution = 1 →
  max_contribution = 11 →
  (n - 1 : ℝ) * min_contribution + max_contribution = 20 :=
by sorry

end NUMINAMATH_CALUDE_gift_contribution_total_l2433_243301


namespace NUMINAMATH_CALUDE_angle_sum_zero_l2433_243322

theorem angle_sum_zero (α β : ℝ) (h_acute_α : 0 < α ∧ α < π / 2) (h_acute_β : 0 < β ∧ β < π / 2)
  (h_eq1 : 4 * (Real.cos α)^2 + 3 * (Real.cos β)^2 = 2)
  (h_eq2 : 4 * Real.sin (2 * α) + 3 * Real.sin (2 * β) = 0) :
  α + 3 * β = 0 := by sorry

end NUMINAMATH_CALUDE_angle_sum_zero_l2433_243322


namespace NUMINAMATH_CALUDE_triangle_formation_conditions_l2433_243363

theorem triangle_formation_conditions 
  (E F G H : ℝ × ℝ)  -- Points in 2D plane
  (a b c : ℝ)        -- Lengths
  (θ φ : ℝ)          -- Angles
  (h_distinct : E ≠ F ∧ F ≠ G ∧ G ≠ H)  -- Distinct points
  (h_collinear : ∃ (m k : ℝ), F.2 = m * F.1 + k ∧ G.2 = m * G.1 + k ∧ H.2 = m * H.1 + k)  -- Collinearity
  (h_order : E.1 < F.1 ∧ F.1 < G.1 ∧ G.1 < H.1)  -- Order on line
  (h_lengths : dist E F = a ∧ dist E G = b ∧ dist E H = c)  -- Segment lengths
  (h_rotation : ∃ (E' : ℝ × ℝ), 
    dist F E' = a ∧ 
    dist G H = c - b ∧
    E' = H)  -- Rotation result
  (h_triangle : ∃ (F' G' : ℝ × ℝ), 
    dist E' F' = a ∧ 
    dist F' G' > 0 ∧ 
    dist G' E' = c - b ∧
    (F'.1 - E'.1) * (G'.2 - E'.2) ≠ (G'.1 - E'.1) * (F'.2 - E'.2))  -- Non-degenerate triangle formed
  : a < c / 2 ∧ b < a + c * Real.cos φ ∧ b * Real.cos θ < c / 2 :=
by sorry

end NUMINAMATH_CALUDE_triangle_formation_conditions_l2433_243363


namespace NUMINAMATH_CALUDE_odd_function_zero_value_l2433_243344

/-- A function f is odd if f(-x) = -f(x) for all x in ℝ -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- For any odd function f defined on ℝ, f(0) = 0 -/
theorem odd_function_zero_value (f : ℝ → ℝ) (h : IsOdd f) : f 0 = 0 := by
  sorry

end NUMINAMATH_CALUDE_odd_function_zero_value_l2433_243344


namespace NUMINAMATH_CALUDE_questions_left_blank_l2433_243384

/-- Represents the math test structure and Steve's performance --/
structure MathTest where
  totalQuestions : ℕ
  wordProblems : ℕ
  addSubProblems : ℕ
  algebraProblems : ℕ
  geometryProblems : ℕ
  totalTime : ℕ
  timePerWordProblem : ℚ
  timePerAddSubProblem : ℚ
  timePerAlgebraProblem : ℕ
  timePerGeometryProblem : ℕ
  wordProblemsAnswered : ℕ
  addSubProblemsAnswered : ℕ
  algebraProblemsAnswered : ℕ
  geometryProblemsAnswered : ℕ

/-- Theorem stating the number of questions left blank --/
theorem questions_left_blank (test : MathTest)
  (h1 : test.totalQuestions = 60)
  (h2 : test.wordProblems = 20)
  (h3 : test.addSubProblems = 25)
  (h4 : test.algebraProblems = 10)
  (h5 : test.geometryProblems = 5)
  (h6 : test.totalTime = 90)
  (h7 : test.timePerWordProblem = 2)
  (h8 : test.timePerAddSubProblem = 3/2)
  (h9 : test.timePerAlgebraProblem = 3)
  (h10 : test.timePerGeometryProblem = 4)
  (h11 : test.wordProblemsAnswered = 15)
  (h12 : test.addSubProblemsAnswered = 22)
  (h13 : test.algebraProblemsAnswered = 8)
  (h14 : test.geometryProblemsAnswered = 3) :
  test.totalQuestions - (test.wordProblemsAnswered + test.addSubProblemsAnswered + test.algebraProblemsAnswered + test.geometryProblemsAnswered) = 12 := by
  sorry

end NUMINAMATH_CALUDE_questions_left_blank_l2433_243384
