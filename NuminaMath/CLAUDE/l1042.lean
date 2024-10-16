import Mathlib

namespace NUMINAMATH_CALUDE_min_additional_games_correct_l1042_104248

/-- The minimum number of additional games needed for the Cheetahs to win at least 80% of all games -/
def min_additional_games : ℕ := 15

/-- The initial number of games played -/
def initial_games : ℕ := 5

/-- The initial number of games won by the Cheetahs -/
def initial_cheetah_wins : ℕ := 1

/-- Checks if the given number of additional games satisfies the condition -/
def satisfies_condition (n : ℕ) : Prop :=
  (initial_cheetah_wins + n : ℚ) / (initial_games + n : ℚ) ≥ 4/5

theorem min_additional_games_correct :
  satisfies_condition min_additional_games ∧
  ∀ m : ℕ, m < min_additional_games → ¬satisfies_condition m :=
by sorry

end NUMINAMATH_CALUDE_min_additional_games_correct_l1042_104248


namespace NUMINAMATH_CALUDE_sine_cosine_inequality_l1042_104279

-- Define a periodic function with period 2
def periodic_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (x + 2) = f x

-- Define symmetry around x = 2
def symmetric_around_two (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (2 + x) = f (2 - x)

-- Define decreasing on an interval
def decreasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y : ℝ, a ≤ x ∧ x < y ∧ y ≤ b → f y < f x

-- Define acute angle
def acute_angle (θ : ℝ) : Prop :=
  0 < θ ∧ θ < Real.pi / 2

-- Theorem statement
theorem sine_cosine_inequality
  (f : ℝ → ℝ)
  (h_periodic : periodic_function f)
  (h_symmetric : symmetric_around_two f)
  (h_decreasing : decreasing_on f (-3) (-2))
  (A B : ℝ)
  (h_acute_A : acute_angle A)
  (h_acute_B : acute_angle B)
  (h_triangle : A + B ≤ Real.pi / 2) :
  f (Real.sin A) > f (Real.cos B) :=
sorry

end NUMINAMATH_CALUDE_sine_cosine_inequality_l1042_104279


namespace NUMINAMATH_CALUDE_weight_of_almonds_l1042_104284

/-- Given the total weight of nuts and the weight of pecans, 
    calculate the weight of almonds. -/
theorem weight_of_almonds 
  (total_weight : ℝ) 
  (pecan_weight : ℝ) 
  (h1 : total_weight = 0.52) 
  (h2 : pecan_weight = 0.38) : 
  total_weight - pecan_weight = 0.14 := by
sorry

end NUMINAMATH_CALUDE_weight_of_almonds_l1042_104284


namespace NUMINAMATH_CALUDE_problem_solution_l1042_104292

def M : Set ℝ := {x | 0 < x ∧ x ≤ 3}
def N : Set ℝ := {x | 0 < x ∧ x ≤ 2}

theorem problem_solution :
  (N ⊆ M) ∧
  (∀ a b : ℝ, (a ∈ M → b ∉ M) ↔ (b ∈ M → a ∉ M)) ∧
  (∃ p q : Prop, ¬(p ∧ q) ∧ (p ∨ q)) ∧
  (¬(∃ x : ℝ, x^2 - x - 1 > 0) ↔ (∀ x : ℝ, x^2 - x - 1 ≤ 0)) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1042_104292


namespace NUMINAMATH_CALUDE_min_x_value_l1042_104267

theorem min_x_value (x y : ℕ+) (h : (100 : ℚ)/151 < (y : ℚ)/(x : ℚ) ∧ (y : ℚ)/(x : ℚ) < (200 : ℚ)/251) :
  ∀ z : ℕ+, z < x → ¬∃ w : ℕ+, (100 : ℚ)/151 < (w : ℚ)/(z : ℚ) ∧ (w : ℚ)/(z : ℚ) < (200 : ℚ)/251 :=
by sorry

end NUMINAMATH_CALUDE_min_x_value_l1042_104267


namespace NUMINAMATH_CALUDE_tea_pot_volume_l1042_104241

/-- The amount of tea in milliliters per cup -/
def tea_per_cup : ℕ := 65

/-- The number of cups filled with tea -/
def cups_filled : ℕ := 16

/-- The total amount of tea in the pot in milliliters -/
def total_tea : ℕ := tea_per_cup * cups_filled

/-- Theorem stating that the total amount of tea in the pot is 1040 ml -/
theorem tea_pot_volume : total_tea = 1040 := by
  sorry

end NUMINAMATH_CALUDE_tea_pot_volume_l1042_104241


namespace NUMINAMATH_CALUDE_parabola_translation_theorem_l1042_104260

/-- Represents a parabola of the form y = ax² -/
structure Parabola where
  a : ℝ

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a translation in 2D space -/
structure Translation where
  right : ℝ
  up : ℝ

/-- Returns true if the given equation represents the parabola after translation -/
def is_translated_parabola (p : Parabola) (t : Translation) (eq : ℝ → ℝ) : Prop :=
  ∀ x, eq x = p.a * (x - t.right)^2 + t.up

/-- Returns true if the given point satisfies the equation -/
def satisfies_equation (pt : Point) (eq : ℝ → ℝ) : Prop :=
  eq pt.x = pt.y

theorem parabola_translation_theorem (p : Parabola) (t : Translation) (pt : Point) :
  is_translated_parabola p t (fun x => -4 * (x - 2)^2 + 3) ∧
  satisfies_equation pt (fun x => -4 * (x - 2)^2 + 3) ∧
  t.right = 2 ∧ t.up = 3 ∧ pt.x = 3 ∧ pt.y = -1 :=
sorry

end NUMINAMATH_CALUDE_parabola_translation_theorem_l1042_104260


namespace NUMINAMATH_CALUDE_wednesday_distance_l1042_104204

/-- Represents the distance Mona biked on each day of the week -/
structure BikeDistance where
  monday : ℕ
  wednesday : ℕ
  saturday : ℕ

/-- Defines the conditions of Mona's biking schedule -/
def validBikeSchedule (d : BikeDistance) : Prop :=
  d.monday + d.wednesday + d.saturday = 30 ∧
  d.monday = 6 ∧
  d.saturday = 2 * d.monday

theorem wednesday_distance (d : BikeDistance) (h : validBikeSchedule d) : d.wednesday = 12 := by
  sorry

end NUMINAMATH_CALUDE_wednesday_distance_l1042_104204


namespace NUMINAMATH_CALUDE_geometric_sequence_terms_l1042_104231

theorem geometric_sequence_terms (a₁ aₙ q : ℚ) (n : ℕ) (h₁ : a₁ = 9/8) (h₂ : aₙ = 1/3) (h₃ : q = 2/3) :
  aₙ = a₁ * q^(n - 1) → n = 4 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_terms_l1042_104231


namespace NUMINAMATH_CALUDE_min_value_3a_plus_1_l1042_104213

theorem min_value_3a_plus_1 (a : ℝ) (h : 8 * a^2 + 9 * a + 6 = 2) :
  ∃ (x : ℝ), (3 * a + 1 ≥ x) ∧ (∀ y, 3 * a + 1 ≥ y → x ≥ y) ∧ x = -2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_3a_plus_1_l1042_104213


namespace NUMINAMATH_CALUDE_percentage_both_correct_l1042_104270

theorem percentage_both_correct (p_first : ℝ) (p_second : ℝ) (p_neither : ℝ) 
  (h1 : p_first = 0.75)
  (h2 : p_second = 0.70)
  (h3 : p_neither = 0.20) :
  p_first + p_second - (1 - p_neither) = 0.65 := by
  sorry

end NUMINAMATH_CALUDE_percentage_both_correct_l1042_104270


namespace NUMINAMATH_CALUDE_total_spent_is_20_27_l1042_104287

/-- Calculates the total amount spent on items with discount and tax --/
def totalSpent (initialAmount : ℚ) (candyPrice : ℚ) (chocolatePrice : ℚ) (gumPrice : ℚ) 
  (chipsPrice : ℚ) (discountRate : ℚ) (taxRate : ℚ) : ℚ :=
  let discountedCandyPrice := candyPrice * (1 - discountRate)
  let subtotal := discountedCandyPrice + chocolatePrice + gumPrice + chipsPrice
  let tax := subtotal * taxRate
  subtotal + tax

/-- Theorem stating that the total amount spent is $20.27 --/
theorem total_spent_is_20_27 : 
  totalSpent 50 7 6 3 4 (10/100) (5/100) = 2027/100 := by
  sorry

end NUMINAMATH_CALUDE_total_spent_is_20_27_l1042_104287


namespace NUMINAMATH_CALUDE_second_player_wins_l1042_104283

/-- Represents a player in the game -/
inductive Player
| First
| Second

/-- Represents the state of the game board -/
structure GameBoard where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents a move in the game -/
structure Move where
  player : Player
  position : Fin 3
  value : ℝ

/-- Checks if a move is valid -/
def isValidMove (board : GameBoard) (move : Move) : Prop :=
  match move.position with
  | 0 => move.value ≠ 0  -- a ≠ 0
  | _ => True

/-- Applies a move to the game board -/
def applyMove (board : GameBoard) (move : Move) : GameBoard :=
  match move.position with
  | 0 => { board with a := move.value }
  | 1 => { board with b := move.value }
  | 2 => { board with c := move.value }

/-- Checks if the quadratic equation has real roots -/
def hasRealRoots (board : GameBoard) : Prop :=
  board.b * board.b - 4 * board.a * board.c ≥ 0

/-- The main theorem stating that the second player has a winning strategy -/
theorem second_player_wins :
  ∀ (firstMove : Move),
    isValidMove { a := 0, b := 0, c := 0 } firstMove →
    ∃ (secondMove : Move),
      isValidMove (applyMove { a := 0, b := 0, c := 0 } firstMove) secondMove ∧
      hasRealRoots (applyMove (applyMove { a := 0, b := 0, c := 0 } firstMove) secondMove) :=
sorry


end NUMINAMATH_CALUDE_second_player_wins_l1042_104283


namespace NUMINAMATH_CALUDE_consecutive_squares_difference_l1042_104201

theorem consecutive_squares_difference (n : ℕ) : 
  (n > 0) → 
  (n + (n + 1) < 150) → 
  ((n + 1)^2 - n^2 = 129 ∨ (n + 1)^2 - n^2 = 147) :=
by
  sorry

end NUMINAMATH_CALUDE_consecutive_squares_difference_l1042_104201


namespace NUMINAMATH_CALUDE_ps_length_is_sqrt_461_l1042_104291

/-- A quadrilateral with two right angles and specific side lengths -/
structure RightQuadrilateral where
  PQ : ℝ
  QR : ℝ
  RS : ℝ
  angleQ_is_right : Bool
  angleR_is_right : Bool

/-- The length of PS in the right quadrilateral PQRS -/
def length_PS (quad : RightQuadrilateral) : ℝ :=
  sorry

/-- Theorem stating that for a quadrilateral PQRS with given side lengths and right angles, PS = √461 -/
theorem ps_length_is_sqrt_461 (quad : RightQuadrilateral) 
  (h1 : quad.PQ = 6)
  (h2 : quad.QR = 10)
  (h3 : quad.RS = 25)
  (h4 : quad.angleQ_is_right = true)
  (h5 : quad.angleR_is_right = true) :
  length_PS quad = Real.sqrt 461 :=
by sorry

end NUMINAMATH_CALUDE_ps_length_is_sqrt_461_l1042_104291


namespace NUMINAMATH_CALUDE_simplify_square_roots_l1042_104275

theorem simplify_square_roots : Real.sqrt 81 - Real.sqrt 144 = -3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_square_roots_l1042_104275


namespace NUMINAMATH_CALUDE_manufacturing_employee_percentage_l1042_104222

theorem manufacturing_employee_percentage 
  (total_degrees : ℝ) 
  (manufacturing_degrees : ℝ) 
  (h1 : total_degrees = 360) 
  (h2 : manufacturing_degrees = 72) : 
  (manufacturing_degrees / total_degrees) * 100 = 20 := by
sorry

end NUMINAMATH_CALUDE_manufacturing_employee_percentage_l1042_104222


namespace NUMINAMATH_CALUDE_min_value_complex_ratio_l1042_104250

theorem min_value_complex_ratio (z : ℂ) (h : z.re ≠ 0) :
  ∃ (min : ℝ), min = -8 ∧ 
  (∀ (w : ℂ), w.re ≠ 0 → (w.re^4)⁻¹ * (w^4).re ≥ min) ∧
  (∃ (w : ℂ), w.re ≠ 0 ∧ (w.re^4)⁻¹ * (w^4).re = min) :=
sorry

end NUMINAMATH_CALUDE_min_value_complex_ratio_l1042_104250


namespace NUMINAMATH_CALUDE_min_offers_for_conviction_l1042_104285

/-- The minimum number of additional offers needed to be convinced with high probability. -/
def min_additional_offers : ℕ := 58

/-- The probability threshold for conviction. -/
def conviction_threshold : ℝ := 0.99

/-- The number of models already observed. -/
def observed_models : ℕ := 12

theorem min_offers_for_conviction :
  ∀ n : ℕ, n > observed_models →
    (observed_models : ℝ) / n ^ min_additional_offers < 1 - conviction_threshold :=
by sorry

end NUMINAMATH_CALUDE_min_offers_for_conviction_l1042_104285


namespace NUMINAMATH_CALUDE_smallest_alpha_inequality_l1042_104220

theorem smallest_alpha_inequality (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  ∃ (α : ℝ), α > 0 ∧ α = 1/2 ∧
  ∀ (β : ℝ), β > 0 →
    ((x + y) / 2 ≥ β * Real.sqrt (x * y) + (1 - β) * Real.sqrt ((x^2 + y^2) / 2) →
     β ≥ α) :=
by sorry

end NUMINAMATH_CALUDE_smallest_alpha_inequality_l1042_104220


namespace NUMINAMATH_CALUDE_fraction_sum_l1042_104295

theorem fraction_sum (a b : ℝ) (h : a / b = 2 / 3) : (a + b) / b = 5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_l1042_104295


namespace NUMINAMATH_CALUDE_solve_for_P_l1042_104265

theorem solve_for_P : ∃ P : ℝ, (P ^ 3) ^ (1/2) = 9 * (81 ^ (1/6)) → P = 3 ^ (16/9) := by
  sorry

end NUMINAMATH_CALUDE_solve_for_P_l1042_104265


namespace NUMINAMATH_CALUDE_hyperbola_max_eccentricity_l1042_104247

/-- Given a hyperbola with equation x²/a² - y²/b² = 1, where a > 0 and b > 0,
    and a point P on the right branch of the hyperbola satisfying |PF₁| = 4|PF₂|,
    the maximum value of the eccentricity e is 5/3. -/
theorem hyperbola_max_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  ∃ e_max : ℝ, e_max = 5/3 ∧
  ∀ (x y e : ℝ),
    x^2/a^2 - y^2/b^2 = 1 →
    x ≥ a →
    ∃ (F₁ F₂ : ℝ × ℝ),
      let P := (x, y)
      let d₁ := Real.sqrt ((P.1 - F₁.1)^2 + (P.2 - F₁.2)^2)
      let d₂ := Real.sqrt ((P.1 - F₂.1)^2 + (P.2 - F₂.2)^2)
      d₁ = 4 * d₂ →
      e = Real.sqrt (1 + b^2/a^2) →
      e ≤ e_max :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_max_eccentricity_l1042_104247


namespace NUMINAMATH_CALUDE_expression_evaluation_l1042_104272

theorem expression_evaluation (x : ℚ) (h : x = 1/2) : 
  (1 + x) * (1 - x) + x * (x + 2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1042_104272


namespace NUMINAMATH_CALUDE_baseball_card_money_ratio_l1042_104280

/-- Proves the ratio of Lisa's money to Charlotte's money given the conditions of the baseball card purchase problem -/
theorem baseball_card_money_ratio :
  let card_cost : ℕ := 100
  let patricia_money : ℕ := 6
  let lisa_money : ℕ := 5 * patricia_money
  let additional_money_needed : ℕ := 49
  let total_money : ℕ := card_cost - additional_money_needed
  let charlotte_money : ℕ := total_money - lisa_money - patricia_money
  (lisa_money : ℚ) / (charlotte_money : ℚ) = 2 / 1 :=
by sorry

end NUMINAMATH_CALUDE_baseball_card_money_ratio_l1042_104280


namespace NUMINAMATH_CALUDE_maaza_liters_l1042_104269

/-- The number of liters of Pepsi -/
def pepsi : ℕ := 144

/-- The number of liters of Sprite -/
def sprite : ℕ := 368

/-- The total number of cans required -/
def total_cans : ℕ := 261

/-- The capacity of each can in liters -/
def can_capacity : ℕ := Nat.gcd pepsi sprite

theorem maaza_liters : ∃ M : ℕ, 
  M + pepsi + sprite = total_cans * can_capacity ∧ 
  M = 3664 := by
  sorry

end NUMINAMATH_CALUDE_maaza_liters_l1042_104269


namespace NUMINAMATH_CALUDE_fairCoinThreeFlipsOneHead_l1042_104219

def fairCoinProbability (n : ℕ) (k : ℕ) : ℚ :=
  (Nat.choose n k : ℚ) * (1/2)^k * (1/2)^(n-k)

theorem fairCoinThreeFlipsOneHead :
  fairCoinProbability 3 1 = 3/8 := by
  sorry

end NUMINAMATH_CALUDE_fairCoinThreeFlipsOneHead_l1042_104219


namespace NUMINAMATH_CALUDE_sum_product_ratio_l1042_104257

theorem sum_product_ratio (x y z : ℝ) (h_distinct : x ≠ y ∧ y ≠ z ∧ x ≠ z) (h_sum : x + y + z = 3) :
  (x * y + y * z + z * x) / (x^2 + y^2 + z^2) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_sum_product_ratio_l1042_104257


namespace NUMINAMATH_CALUDE_integer_root_quadratic_l1042_104297

theorem integer_root_quadratic (m n : ℕ+) : 
  (∃ x : ℕ+, x^2 - (m.val * n.val) * x + (m.val + n.val) = 0) ↔ 
  ((m = 2 ∧ n = 3) ∨ (m = 3 ∧ n = 2) ∨ (m = 2 ∧ n = 2) ∨ (m = 1 ∧ n = 5) ∨ (m = 5 ∧ n = 1)) :=
by sorry

end NUMINAMATH_CALUDE_integer_root_quadratic_l1042_104297


namespace NUMINAMATH_CALUDE_rectangle_probability_l1042_104216

/-- A rectangle in the 2D plane --/
structure Rectangle where
  x1 : ℝ
  y1 : ℝ
  x2 : ℝ
  y2 : ℝ

/-- The probability that a randomly selected point from a rectangle is closer to one point than another --/
def closerProbability (r : Rectangle) (p1 : ℝ × ℝ) (p2 : ℝ × ℝ) : ℝ :=
  sorry

/-- The theorem to be proved --/
theorem rectangle_probability : 
  let r := Rectangle.mk 0 0 3 2
  closerProbability r (0, 0) (4, 0) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_probability_l1042_104216


namespace NUMINAMATH_CALUDE_equation_solutions_l1042_104296

theorem equation_solutions : 
  (∃ x1 x2 : ℝ, x1 = 2 + Real.sqrt 5 ∧ x2 = 2 - Real.sqrt 5 ∧ 
    x1^2 - 4*x1 - 1 = 0 ∧ x2^2 - 4*x2 - 1 = 0) ∧ 
  (∃ x1 x2 : ℝ, x1 = -3 ∧ x2 = 6 ∧ 
    (x1 + 3)*(x1 - 3) = 3*(x1 + 3) ∧ (x2 + 3)*(x2 - 3) = 3*(x2 + 3)) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l1042_104296


namespace NUMINAMATH_CALUDE_sports_club_overlap_l1042_104271

theorem sports_club_overlap (total : ℕ) (badminton : ℕ) (tennis : ℕ) (neither : ℕ) 
  (h1 : total = 30)
  (h2 : badminton = 17)
  (h3 : tennis = 19)
  (h4 : neither = 3) :
  badminton + tennis - total + neither = 9 := by
  sorry

end NUMINAMATH_CALUDE_sports_club_overlap_l1042_104271


namespace NUMINAMATH_CALUDE_tangent_line_intersection_l1042_104289

open Real

theorem tangent_line_intersection (x₀ : ℝ) : 
  (∃ m : ℝ, 0 < m ∧ m < 1 ∧ 
    (1/2 * x₀^2 - log m = x₀ * (x₀ - m)) ∧ 
    (1/(2*m) = x₀)) →
  (Real.sqrt 3 < x₀ ∧ x₀ < 2) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_intersection_l1042_104289


namespace NUMINAMATH_CALUDE_skateboard_ramp_speeds_l1042_104209

theorem skateboard_ramp_speeds (S₁ S₂ S₃ : ℝ) :
  (S₁ + S₂ + S₃) / 3 + 4 = 40 →
  ∃ (T₁ T₂ T₃ : ℝ), (T₁ + T₂ + T₃) / 3 + 4 = 40 ∧ (T₁ ≠ S₁ ∨ T₂ ≠ S₂ ∨ T₃ ≠ S₃) :=
by sorry

end NUMINAMATH_CALUDE_skateboard_ramp_speeds_l1042_104209


namespace NUMINAMATH_CALUDE_ellipse_standard_equation_l1042_104203

/-- Definition of an ellipse with given major axis length and eccentricity -/
structure Ellipse where
  major_axis : ℝ
  eccentricity : ℝ

/-- The standard equation of an ellipse -/
def standard_equation (e : Ellipse) : Prop :=
  (∀ x y : ℝ, x^2 / 16 + y^2 / 7 = 1) ∨ (∀ x y : ℝ, x^2 / 7 + y^2 / 16 = 1)

/-- Theorem stating that an ellipse with major axis 8 and eccentricity 3/4 satisfies the standard equation -/
theorem ellipse_standard_equation (e : Ellipse) (h1 : e.major_axis = 8) (h2 : e.eccentricity = 3/4) :
  standard_equation e := by
  sorry

end NUMINAMATH_CALUDE_ellipse_standard_equation_l1042_104203


namespace NUMINAMATH_CALUDE_female_students_count_l1042_104286

theorem female_students_count (total_average : ℝ) (male_count : ℕ) (male_average : ℝ) (female_average : ℝ) :
  total_average = 90 →
  male_count = 8 →
  male_average = 84 →
  female_average = 92 →
  ∃ (female_count : ℕ), 
    (male_count * male_average + female_count * female_average) / (male_count + female_count) = total_average ∧
    female_count = 24 :=
by sorry

end NUMINAMATH_CALUDE_female_students_count_l1042_104286


namespace NUMINAMATH_CALUDE_sixth_television_is_three_l1042_104239

def selected_televisions : List Nat := [20, 26, 24, 19, 23, 3]

theorem sixth_television_is_three : 
  selected_televisions.length = 6 ∧ selected_televisions.getLast? = some 3 :=
sorry

end NUMINAMATH_CALUDE_sixth_television_is_three_l1042_104239


namespace NUMINAMATH_CALUDE_pirate_treasure_problem_l1042_104211

theorem pirate_treasure_problem :
  let n : ℕ := 8  -- Total number of islands
  let k : ℕ := 5  -- Number of islands with treasure
  let p_treasure : ℚ := 1/6  -- Probability of an island having treasure and no traps
  let p_neither : ℚ := 2/3  -- Probability of an island having neither treasure nor traps
  
  (Nat.choose n k : ℚ) * p_treasure^k * p_neither^(n - k) = 7/3328 := by
  sorry

end NUMINAMATH_CALUDE_pirate_treasure_problem_l1042_104211


namespace NUMINAMATH_CALUDE_infinite_divisibility_l1042_104244

theorem infinite_divisibility (a : ℕ) (h : a > 3) :
  ∃ (f : ℕ → ℕ), Monotone f ∧ (∀ i, (a + f i) ∣ (a^(f i) + 1)) :=
sorry

end NUMINAMATH_CALUDE_infinite_divisibility_l1042_104244


namespace NUMINAMATH_CALUDE_triangles_from_parallel_lines_l1042_104299

/-- The number of points on line a -/
def points_on_a : ℕ := 5

/-- The number of points on line b -/
def points_on_b : ℕ := 6

/-- The total number of triangles that can be formed -/
def total_triangles : ℕ := 135

/-- Theorem stating that the total number of triangles formed by points on two parallel lines is correct -/
theorem triangles_from_parallel_lines : 
  (points_on_a.choose 1 * points_on_b.choose 2) + (points_on_a.choose 2 * points_on_b.choose 1) = total_triangles :=
by sorry

end NUMINAMATH_CALUDE_triangles_from_parallel_lines_l1042_104299


namespace NUMINAMATH_CALUDE_school_population_l1042_104249

theorem school_population (total_sample : ℕ) (first_year_sample : ℕ) (third_year_sample : ℕ) (second_year_total : ℕ) :
  total_sample = 45 →
  first_year_sample = 20 →
  third_year_sample = 10 →
  second_year_total = 300 →
  ∃ (total_students : ℕ), total_students = 900 :=
by
  sorry

end NUMINAMATH_CALUDE_school_population_l1042_104249


namespace NUMINAMATH_CALUDE_reinforcement_arrival_time_l1042_104246

/-- Calculates the number of days that passed before reinforcement arrived --/
def days_before_reinforcement (initial_garrison : ℕ) (initial_provisions : ℕ) 
  (reinforcement : ℕ) (remaining_provisions : ℕ) : ℕ :=
  (initial_garrison * initial_provisions - (initial_garrison + reinforcement) * remaining_provisions) / initial_garrison

/-- Theorem stating that 21 days passed before reinforcement arrived --/
theorem reinforcement_arrival_time : 
  days_before_reinforcement 2000 54 1300 20 = 21 := by
  sorry


end NUMINAMATH_CALUDE_reinforcement_arrival_time_l1042_104246


namespace NUMINAMATH_CALUDE_lower_bound_k_squared_l1042_104282

theorem lower_bound_k_squared (k : ℤ) (V : ℤ) (h1 : k^2 > V) (h2 : k^2 < 225) 
  (h3 : ∃ (S : Finset ℤ), S.card ≤ 6 ∧ ∀ x, x ∈ S ↔ x^2 > V ∧ x^2 < 225) :
  81 ≤ k^2 := by
  sorry

end NUMINAMATH_CALUDE_lower_bound_k_squared_l1042_104282


namespace NUMINAMATH_CALUDE_prime_characterization_l1042_104242

theorem prime_characterization (p : ℕ) (h1 : p > 3) (h2 : (p^2 + 15) % 12 = 4) :
  Nat.Prime p :=
sorry

end NUMINAMATH_CALUDE_prime_characterization_l1042_104242


namespace NUMINAMATH_CALUDE_franks_money_l1042_104210

/-- Frank's initial amount of money -/
def initial_money : ℝ := 11

/-- Amount Frank spent on a game -/
def game_cost : ℝ := 3

/-- Frank's allowance -/
def allowance : ℝ := 14

/-- Frank's final amount of money -/
def final_money : ℝ := 22

theorem franks_money :
  initial_money - game_cost + allowance = final_money :=
by sorry

end NUMINAMATH_CALUDE_franks_money_l1042_104210


namespace NUMINAMATH_CALUDE_hurricane_damage_conversion_l1042_104230

theorem hurricane_damage_conversion (damage_aud : ℝ) (exchange_rate : ℝ) : 
  damage_aud = 45000000 → 
  exchange_rate = 2 → 
  damage_aud / exchange_rate = 22500000 :=
by sorry

end NUMINAMATH_CALUDE_hurricane_damage_conversion_l1042_104230


namespace NUMINAMATH_CALUDE_function_range_in_unit_interval_l1042_104234

theorem function_range_in_unit_interval (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, x > y → (f x)^2 ≤ f y) :
  ∀ x : ℝ, 0 ≤ f x ∧ f x ≤ 1 := by
sorry

end NUMINAMATH_CALUDE_function_range_in_unit_interval_l1042_104234


namespace NUMINAMATH_CALUDE_field_trip_students_l1042_104253

/-- The number of seats on each school bus -/
def seats_per_bus : ℕ := 2

/-- The number of buses needed for the trip -/
def number_of_buses : ℕ := 7

/-- The total number of students going on the field trip -/
def total_students : ℕ := seats_per_bus * number_of_buses

theorem field_trip_students : total_students = 14 := by
  sorry

end NUMINAMATH_CALUDE_field_trip_students_l1042_104253


namespace NUMINAMATH_CALUDE_prob_A_shot_twice_correct_l1042_104277

def prob_A : ℚ := 3/4
def prob_B : ℚ := 4/5

def prob_A_shot_twice : ℚ := 19/400

theorem prob_A_shot_twice_correct :
  let p_A_miss := 1 - prob_A
  let p_B_miss := 1 - prob_B
  prob_A_shot_twice = p_A_miss * p_B_miss * prob_A + p_A_miss * p_B_miss * p_A_miss * prob_B :=
by sorry

end NUMINAMATH_CALUDE_prob_A_shot_twice_correct_l1042_104277


namespace NUMINAMATH_CALUDE_perimeter_difference_l1042_104288

/-- Represents the dimensions of a rectangle -/
structure Rectangle where
  length : Real
  width : Real

/-- Calculates the perimeter of a rectangle -/
def perimeter (r : Rectangle) : Real :=
  2 * (r.length + r.width)

/-- Represents a cutting configuration for the plywood -/
structure CuttingConfig where
  piece : Rectangle
  num_pieces : Nat

/-- The original plywood dimensions -/
def plywood : Rectangle :=
  { length := 10, width := 6 }

/-- The number of pieces to cut the plywood into -/
def num_pieces : Nat := 6

/-- Checks if a cutting configuration is valid for the given plywood -/
def is_valid_config (config : CuttingConfig) : Prop :=
  config.num_pieces = num_pieces ∧
  config.piece.length * config.piece.width * config.num_pieces = plywood.length * plywood.width

/-- Theorem stating the difference between max and min perimeter -/
theorem perimeter_difference :
  ∃ (max_config min_config : CuttingConfig),
    is_valid_config max_config ∧
    is_valid_config min_config ∧
    (∀ c : CuttingConfig, is_valid_config c →
      perimeter c.piece ≤ perimeter max_config.piece ∧
      perimeter c.piece ≥ perimeter min_config.piece) ∧
    perimeter max_config.piece - perimeter min_config.piece = 11.34 := by
  sorry

end NUMINAMATH_CALUDE_perimeter_difference_l1042_104288


namespace NUMINAMATH_CALUDE_a_gt_one_sufficient_not_necessary_for_a_squared_gt_one_l1042_104215

theorem a_gt_one_sufficient_not_necessary_for_a_squared_gt_one :
  (∃ a : ℝ, a > 1 ∧ a^2 > 1) ∧ 
  (∃ a : ℝ, a^2 > 1 ∧ ¬(a > 1)) := by
  sorry

end NUMINAMATH_CALUDE_a_gt_one_sufficient_not_necessary_for_a_squared_gt_one_l1042_104215


namespace NUMINAMATH_CALUDE_inequality_proof_l1042_104258

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (Real.sqrt (b + c) / a) + (Real.sqrt (c + a) / b) + (Real.sqrt (a + b) / c) ≥
  (4 * (a + b + c)) / Real.sqrt ((a + b) * (b + c) * (c + a)) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1042_104258


namespace NUMINAMATH_CALUDE_chromosomal_variations_l1042_104208

/-- Represents a biological process or condition -/
inductive BiologicalProcess
| AntherCulture
| DNABaseChange
| NonHomologousRecombination
| CrossingOver
| DownSyndrome

/-- Defines what constitutes a chromosomal variation -/
def isChromosomalVariation (p : BiologicalProcess) : Prop :=
  match p with
  | BiologicalProcess.AntherCulture => true
  | BiologicalProcess.DNABaseChange => false
  | BiologicalProcess.NonHomologousRecombination => false
  | BiologicalProcess.CrossingOver => false
  | BiologicalProcess.DownSyndrome => true

/-- The main theorem stating which processes are chromosomal variations -/
theorem chromosomal_variations :
  (isChromosomalVariation BiologicalProcess.AntherCulture) ∧
  (¬ isChromosomalVariation BiologicalProcess.DNABaseChange) ∧
  (¬ isChromosomalVariation BiologicalProcess.NonHomologousRecombination) ∧
  (¬ isChromosomalVariation BiologicalProcess.CrossingOver) ∧
  (isChromosomalVariation BiologicalProcess.DownSyndrome) :=
by sorry

end NUMINAMATH_CALUDE_chromosomal_variations_l1042_104208


namespace NUMINAMATH_CALUDE_garden_size_l1042_104298

theorem garden_size (garden_size fruit_size vegetable_size strawberry_size : ℝ) : 
  fruit_size = vegetable_size →
  garden_size = fruit_size + vegetable_size →
  strawberry_size = fruit_size / 4 →
  strawberry_size = 8 →
  garden_size = 64 := by
sorry

end NUMINAMATH_CALUDE_garden_size_l1042_104298


namespace NUMINAMATH_CALUDE_cubic_root_form_l1042_104294

theorem cubic_root_form : ∃ (x : ℝ), 
  16 * x^3 - 4 * x^2 - 4 * x - 1 = 0 ∧ 
  x = (Real.rpow 256 (1/3 : ℝ) + Real.rpow 16 (1/3 : ℝ) + 1) / 16 := by
  sorry

end NUMINAMATH_CALUDE_cubic_root_form_l1042_104294


namespace NUMINAMATH_CALUDE_lunch_break_duration_l1042_104252

/-- Given the recess breaks and total time outside of class, prove the lunch break duration. -/
theorem lunch_break_duration 
  (recess1 recess2 recess3 total_outside : ℕ)
  (h1 : recess1 = 15)
  (h2 : recess2 = 15)
  (h3 : recess3 = 20)
  (h4 : total_outside = 80) :
  total_outside - (recess1 + recess2 + recess3) = 30 := by
  sorry

end NUMINAMATH_CALUDE_lunch_break_duration_l1042_104252


namespace NUMINAMATH_CALUDE_fruit_drink_volume_l1042_104235

theorem fruit_drink_volume (orange_percent : ℝ) (watermelon_percent : ℝ) (grape_ounces : ℝ) :
  orange_percent = 0.15 →
  watermelon_percent = 0.60 →
  grape_ounces = 30 →
  ∃ total_ounces : ℝ,
    total_ounces = 120 ∧
    orange_percent * total_ounces + watermelon_percent * total_ounces + grape_ounces = total_ounces :=
by
  sorry

end NUMINAMATH_CALUDE_fruit_drink_volume_l1042_104235


namespace NUMINAMATH_CALUDE_triangle_trip_distance_l1042_104223

/-- Given a right-angled triangle DEF with F as the right angle, 
    where DF = 2000 and DE = 4500, prove that DE + EF + DF = 10531 -/
theorem triangle_trip_distance (DE DF EF : ℝ) : 
  DE = 4500 → 
  DF = 2000 → 
  EF ^ 2 = DE ^ 2 - DF ^ 2 → 
  DE + EF + DF = 10531 := by
sorry

end NUMINAMATH_CALUDE_triangle_trip_distance_l1042_104223


namespace NUMINAMATH_CALUDE_total_junk_mail_l1042_104273

/-- Given a block with houses and junk mail distribution, calculate the total junk mail. -/
theorem total_junk_mail (num_houses : ℕ) (mail_per_house : ℕ) : num_houses = 10 → mail_per_house = 35 → num_houses * mail_per_house = 350 := by
  sorry

end NUMINAMATH_CALUDE_total_junk_mail_l1042_104273


namespace NUMINAMATH_CALUDE_transform_f_eq_g_l1042_104236

/-- The original quadratic function -/
def f (x : ℝ) : ℝ := x^2 - 2

/-- The transformation: shift 1 unit left, then 3 units up -/
def transform (g : ℝ → ℝ) : ℝ → ℝ := λ x => g (x + 1) + 3

/-- The expected result function -/
def g (x : ℝ) : ℝ := (x + 1)^2 + 1

/-- Theorem stating that the transformation of f equals g -/
theorem transform_f_eq_g : transform f = g := by sorry

end NUMINAMATH_CALUDE_transform_f_eq_g_l1042_104236


namespace NUMINAMATH_CALUDE_four_points_probability_l1042_104202

-- Define a circle
def Circle : Type := Unit

-- Define a point on a circle
def Point (c : Circle) : Type := Unit

-- Define a function to choose n points uniformly at random on a circle
def chooseRandomPoints (c : Circle) (n : ℕ) : Type := 
  Fin n → Point c

-- Define a predicate for two points and the center forming an obtuse triangle
def isObtuse (c : Circle) (p1 p2 : Point c) : Prop := sorry

-- Define a function to calculate the probability of an event
def probability (event : Prop) : ℝ := sorry

-- The main theorem
theorem four_points_probability (c : Circle) :
  probability (∀ (points : chooseRandomPoints c 4),
    ∀ (i j : Fin 4), i ≠ j → ¬isObtuse c (points i) (points j)) = 1 / 64 := by
  sorry

end NUMINAMATH_CALUDE_four_points_probability_l1042_104202


namespace NUMINAMATH_CALUDE_parallel_vectors_a_value_l1042_104227

/-- Two vectors are parallel if their components are proportional -/
def are_parallel (v w : ℝ × ℝ) : Prop :=
  v.1 * w.2 = v.2 * w.1

theorem parallel_vectors_a_value :
  let m : ℝ × ℝ := (2, 1)
  let n : ℝ × ℝ := (4, a)
  ∀ a : ℝ, are_parallel m n → a = 2 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_a_value_l1042_104227


namespace NUMINAMATH_CALUDE_number_of_pupils_is_40_l1042_104254

/-- The number of pupils in a class, given a specific mark entry error and its effect on the class average. -/
def number_of_pupils : ℕ :=
  let incorrect_mark : ℕ := 83
  let correct_mark : ℕ := 63
  let average_increase : ℚ := 1/2
  40

/-- Theorem stating that the number of pupils is 40 under the given conditions. -/
theorem number_of_pupils_is_40 :
  let n := number_of_pupils
  let incorrect_mark : ℕ := 83
  let correct_mark : ℕ := 63
  let mark_difference : ℕ := incorrect_mark - correct_mark
  let average_increase : ℚ := 1/2
  (mark_difference : ℚ) / n = average_increase → n = 40 := by
  sorry

end NUMINAMATH_CALUDE_number_of_pupils_is_40_l1042_104254


namespace NUMINAMATH_CALUDE_six_balls_four_boxes_l1042_104228

/-- The number of ways to distribute n distinguishable balls into k indistinguishable boxes -/
def distribute_balls (n k : ℕ) : ℕ := sorry

/-- Theorem stating that there are 257 ways to distribute 6 distinguishable balls into 4 indistinguishable boxes -/
theorem six_balls_four_boxes : distribute_balls 6 4 = 257 := by sorry

end NUMINAMATH_CALUDE_six_balls_four_boxes_l1042_104228


namespace NUMINAMATH_CALUDE_normal_distribution_std_dev_l1042_104221

/-- Represents a normal distribution -/
structure NormalDistribution where
  mean : ℝ
  std_dev : ℝ

/-- The value that is exactly 2 standard deviations less than the mean -/
def two_std_dev_below (d : NormalDistribution) : ℝ :=
  d.mean - 2 * d.std_dev

/-- Theorem: If the mean is 14.0 and the value 2 standard deviations below the mean is 11,
    then the standard deviation is 1.5 -/
theorem normal_distribution_std_dev
  (d : NormalDistribution)
  (h_mean : d.mean = 14.0)
  (h_two_below : two_std_dev_below d = 11) :
  d.std_dev = 1.5 := by
  sorry

end NUMINAMATH_CALUDE_normal_distribution_std_dev_l1042_104221


namespace NUMINAMATH_CALUDE_functional_equation_solution_l1042_104290

theorem functional_equation_solution (f g : ℝ → ℝ) 
  (h : ∀ (x y : ℝ), x ≠ 0 → y ≠ 0 → f (x + y) = g (1/x + 1/y) * (x*y)^2008) :
  ∃ (c : ℝ), ∀ (x : ℝ), f x = c * x^2008 ∧ g x = c * x^2008 := by
sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l1042_104290


namespace NUMINAMATH_CALUDE_value_of_expression_l1042_104276

def smallest_positive_integer : ℕ := 1

def largest_negative_integer : ℤ := -1

def smallest_absolute_rational : ℚ := 0

def rational_at_distance_4 : Set ℚ := {d : ℚ | d = 4 ∨ d = -4}

theorem value_of_expression (a b : ℤ) (c d : ℚ) :
  a = smallest_positive_integer ∧
  b = largest_negative_integer ∧
  c = smallest_absolute_rational ∧
  d ∈ rational_at_distance_4 →
  a - b - c + d = -2 ∨ a - b - c + d = 6 := by
  sorry

end NUMINAMATH_CALUDE_value_of_expression_l1042_104276


namespace NUMINAMATH_CALUDE_impossible_inequalities_l1042_104217

theorem impossible_inequalities (a b c : ℝ) : ¬(|a| < |b - c| ∧ |b| < |c - a| ∧ |c| < |a - b|) := by
  sorry

end NUMINAMATH_CALUDE_impossible_inequalities_l1042_104217


namespace NUMINAMATH_CALUDE_dentist_age_fraction_l1042_104262

/-- Given a dentist's current age A and a fraction F, proves that F = 1/10 when A = 32 and (1/6) * (A - 8) = F * (A + 8) -/
theorem dentist_age_fraction (A : ℕ) (F : ℚ) 
  (h1 : A = 32) 
  (h2 : (1/6 : ℚ) * ((A : ℚ) - 8) = F * ((A : ℚ) + 8)) : 
  F = 1/10 := by sorry

end NUMINAMATH_CALUDE_dentist_age_fraction_l1042_104262


namespace NUMINAMATH_CALUDE_sandys_shopping_money_l1042_104259

/-- Sandy's shopping problem -/
theorem sandys_shopping_money (initial_amount : ℝ) (spent_percentage : ℝ) (amount_left : ℝ) :
  initial_amount = 200 →
  spent_percentage = 30 →
  amount_left = initial_amount - (spent_percentage / 100 * initial_amount) →
  amount_left = 140 :=
by sorry

end NUMINAMATH_CALUDE_sandys_shopping_money_l1042_104259


namespace NUMINAMATH_CALUDE_repeating_decimal_fraction_sum_l1042_104274

theorem repeating_decimal_fraction_sum (a b : ℕ+) : 
  (a.val : ℚ) / b.val = 4 / 11 → 
  Nat.gcd a.val b.val = 1 → 
  a.val + b.val = 15 := by
sorry

end NUMINAMATH_CALUDE_repeating_decimal_fraction_sum_l1042_104274


namespace NUMINAMATH_CALUDE_cloth_cost_calculation_l1042_104218

theorem cloth_cost_calculation (length : Real) (price_per_meter : Real) :
  length = 9.25 ∧ price_per_meter = 43 → length * price_per_meter = 397.75 := by
  sorry

end NUMINAMATH_CALUDE_cloth_cost_calculation_l1042_104218


namespace NUMINAMATH_CALUDE_prob_at_least_one_of_B_or_C_given_A_l1042_104256

/-- The probability of selecting at least one of boy B and girl C, given boy A is already selected -/
theorem prob_at_least_one_of_B_or_C_given_A (total_boys : Nat) (total_girls : Nat) 
  (representatives : Nat) (h1 : total_boys = 5) (h2 : total_girls = 2) (h3 : representatives = 3) :
  let remaining_boys := total_boys - 1
  let remaining_total := total_boys + total_girls - 1
  let total_ways := Nat.choose remaining_total (representatives - 1)
  let ways_without_B_or_C := Nat.choose (remaining_boys - 1) (representatives - 1) + 
                             Nat.choose (remaining_boys - 1) (representatives - 2) * total_girls
  (1 : ℚ) - (ways_without_B_or_C : ℚ) / total_ways = 1 / 15 := by
  sorry

end NUMINAMATH_CALUDE_prob_at_least_one_of_B_or_C_given_A_l1042_104256


namespace NUMINAMATH_CALUDE_two_solutions_set_equiv_l1042_104266

/-- The set of values for 'a' that satisfy the conditions for two distinct solutions -/
def TwoSolutionsSet : Set ℝ :=
  {a | 9 * (a - 2) > 0 ∧ 
       a > 0 ∧ 
       a^2 - 9*a + 18 > 0 ∧
       a ≠ 11 ∧
       ∃ (x y : ℝ), x ≠ y ∧ x = a + 3 * Real.sqrt (a - 2) ∧ y = a - 3 * Real.sqrt (a - 2)}

/-- The theorem stating the equivalence of the solution set -/
theorem two_solutions_set_equiv :
  TwoSolutionsSet = {a | (2 < a ∧ a < 3) ∨ (6 < a ∧ a < 11) ∨ (11 < a)} :=
by sorry

end NUMINAMATH_CALUDE_two_solutions_set_equiv_l1042_104266


namespace NUMINAMATH_CALUDE_square_sum_reciprocal_l1042_104245

theorem square_sum_reciprocal (x : ℝ) (h : x + (1 / x) = 5) : x^2 + (1 / x)^2 = 23 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_reciprocal_l1042_104245


namespace NUMINAMATH_CALUDE_tree_count_proof_l1042_104237

theorem tree_count_proof (total : ℕ) (pine_fraction : ℚ) (fir_percent : ℚ) 
  (h1 : total = 520)
  (h2 : pine_fraction = 1 / 3)
  (h3 : fir_percent = 25 / 100) :
  ⌊total * pine_fraction⌋ + ⌊total * fir_percent⌋ = 390 := by
  sorry

end NUMINAMATH_CALUDE_tree_count_proof_l1042_104237


namespace NUMINAMATH_CALUDE_height_comparison_l1042_104255

theorem height_comparison (p q : ℝ) (h : p = 0.6 * q) :
  (q - p) / p = 2/3 := by sorry

end NUMINAMATH_CALUDE_height_comparison_l1042_104255


namespace NUMINAMATH_CALUDE_complex_power_sum_l1042_104226

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem complex_power_sum : 3 * i^23 + 2 * i^47 = -5 * i := by
  sorry

end NUMINAMATH_CALUDE_complex_power_sum_l1042_104226


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l1042_104225

theorem quadratic_inequality_solution (x : ℝ) :
  (x^2 - 3*x - 10 > 0) ↔ (x < -2 ∨ x > 5) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l1042_104225


namespace NUMINAMATH_CALUDE_quadratic_sum_l1042_104264

/-- Given a quadratic function f(x) = -3x^2 + 24x + 144, prove that when written
    in the form a(x+b)^2 + c, the sum of a, b, and c is 185. -/
theorem quadratic_sum (f : ℝ → ℝ) (a b c : ℝ) :
  (∀ x, f x = -3 * x^2 + 24 * x + 144) →
  (∀ x, f x = a * (x + b)^2 + c) →
  a + b + c = 185 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sum_l1042_104264


namespace NUMINAMATH_CALUDE_triangle_property_l1042_104240

/-- Given a triangle ABC with angles A, B, C satisfying the given condition,
    prove that A = π/3 and the maximum area is 3√3/4 when the circumradius is 1 -/
theorem triangle_property (A B C : Real) (h : 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π) 
  (eq : (Real.sin A - Real.sin B + Real.sin C) / Real.sin C = 
        Real.sin B / (Real.sin A + Real.sin B - Real.sin C)) :
  A = π/3 ∧ 
  (∀ S : Real, S ≤ 3 * Real.sqrt 3 / 4 ∧ 
    ∃ a b c : Real, 0 < a ∧ 0 < b ∧ 0 < c ∧
      a^2 + b^2 + c^2 = 2 * (a*b + b*c + c*a) ∧
      S = (Real.sin A * b * c) / 2) := by
  sorry

end NUMINAMATH_CALUDE_triangle_property_l1042_104240


namespace NUMINAMATH_CALUDE_charity_event_result_l1042_104214

/-- Represents the number of books of each type brought by a participant -/
structure BookContribution where
  encyclopedias : ℕ
  fiction : ℕ
  reference : ℕ

/-- Represents the total number of books collected -/
structure TotalBooks where
  encyclopedias : ℕ
  fiction : ℕ
  reference : ℕ

/-- Represents how books are distributed on shelves -/
structure ShelfDistribution where
  first_shelf : ℕ
  second_shelf : ℕ

def charity_event_books (total : TotalBooks) (shelf : ShelfDistribution) : Prop :=
  -- Each participant brings either 1 encyclopedia, 3 fiction books, or 2 reference books
  ∃ (participants : ℕ) (encyc_part fict_part ref_part : ℕ),
    participants = encyc_part + fict_part + ref_part ∧
    total.encyclopedias = encyc_part * 1 ∧
    total.fiction = fict_part * 3 ∧
    total.reference = ref_part * 2 ∧
    -- 150 encyclopedias were collected
    total.encyclopedias = 150 ∧
    -- Two bookshelves were filled with an equal number of books
    shelf.first_shelf = shelf.second_shelf ∧
    -- The first shelf contained 1/5 of all reference books, 1/7 of all fiction books, and all encyclopedias
    shelf.first_shelf = total.encyclopedias + total.reference / 5 + total.fiction / 7 ∧
    -- Total books on both shelves
    shelf.first_shelf + shelf.second_shelf = total.encyclopedias + total.fiction + total.reference

theorem charity_event_result :
  ∀ (total : TotalBooks) (shelf : ShelfDistribution),
    charity_event_books total shelf →
    ∃ (participants : ℕ),
      participants = 416 ∧
      total.encyclopedias + total.fiction + total.reference = 738 :=
sorry

end NUMINAMATH_CALUDE_charity_event_result_l1042_104214


namespace NUMINAMATH_CALUDE_Q_always_perfect_square_l1042_104232

theorem Q_always_perfect_square (x : ℤ) : ∃ (b : ℤ), x^4 + 4*x^3 + 8*x^2 + 6*x + 9 = b^2 := by
  sorry

end NUMINAMATH_CALUDE_Q_always_perfect_square_l1042_104232


namespace NUMINAMATH_CALUDE_fraction_multiplication_cube_l1042_104281

theorem fraction_multiplication_cube : (1 / 2 : ℚ)^3 * (1 / 7 : ℚ) = 1 / 56 := by
  sorry

end NUMINAMATH_CALUDE_fraction_multiplication_cube_l1042_104281


namespace NUMINAMATH_CALUDE_least_prime_for_integer_roots_l1042_104205

theorem least_prime_for_integer_roots : 
  ∃ (P : ℕ), 
    Prime P ∧ 
    (∃ (x : ℤ), x^2 + 2*(P+1)*x + P^2 - P - 14 = 0) ∧
    (∀ (Q : ℕ), Prime Q ∧ Q < P → ¬∃ (y : ℤ), y^2 + 2*(Q+1)*y + Q^2 - Q - 14 = 0) ∧
    P = 7 :=
sorry

end NUMINAMATH_CALUDE_least_prime_for_integer_roots_l1042_104205


namespace NUMINAMATH_CALUDE_binary_digit_difference_l1042_104233

-- Define a function to calculate the number of digits in the binary representation of a number
def binaryDigits (n : ℕ) : ℕ :=
  if n = 0 then 1 else Nat.log2 n + 1

-- State the theorem
theorem binary_digit_difference : binaryDigits 950 - binaryDigits 150 = 2 := by
  sorry

end NUMINAMATH_CALUDE_binary_digit_difference_l1042_104233


namespace NUMINAMATH_CALUDE_cuboid_edge_length_l1042_104229

/-- Given a cuboid with two edges of 6 cm and a volume of 180 cm³, 
    the length of the third edge is 5 cm. -/
theorem cuboid_edge_length (edge1 edge3 volume : ℝ) : 
  edge1 = 6 → edge3 = 6 → volume = 180 → 
  ∃ edge2 : ℝ, edge1 * edge2 * edge3 = volume ∧ edge2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_cuboid_edge_length_l1042_104229


namespace NUMINAMATH_CALUDE_equation_solutions_l1042_104293

def equation (x : ℝ) : Prop :=
  1 / (x^2 + 10*x - 12) + 1 / (x^2 + 3*x - 12) + 1 / (x^2 - 14*x - 12) = 0

theorem equation_solutions :
  ∀ x : ℝ, equation x ↔ x = 1 ∨ x = -21 ∨ x = 5 + Real.sqrt 37 ∨ x = 5 - Real.sqrt 37 :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l1042_104293


namespace NUMINAMATH_CALUDE_least_subtraction_for_divisibility_problem_solution_l1042_104243

theorem least_subtraction_for_divisibility (n : ℕ) (p : ℕ) (h_prime : Nat.Prime p) :
  ∃ (x : ℕ), x < p ∧ (n - x) % p = 0 ∧ ∀ (y : ℕ), y < x → (n - y) % p ≠ 0 :=
by
  sorry

theorem problem_solution :
  let n := 724946
  let p := 37
  (∃ (x : ℕ), x < p ∧ (n - x) % p = 0 ∧ ∀ (y : ℕ), y < x → (n - y) % p ≠ 0) ∧
  (17 < p ∧ (n - 17) % p = 0 ∧ ∀ (y : ℕ), y < 17 → (n - y) % p ≠ 0) :=
by
  sorry

end NUMINAMATH_CALUDE_least_subtraction_for_divisibility_problem_solution_l1042_104243


namespace NUMINAMATH_CALUDE_triangle_area_at_least_three_l1042_104224

/-- A type representing a point in a plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The area of a triangle formed by three points -/
def triangleArea (p1 p2 p3 : Point) : ℝ := sorry

/-- A set of five points in a plane -/
def FivePoints : Type := Fin 5 → Point

theorem triangle_area_at_least_three (points : FivePoints) 
  (h : ∀ (i j k : Fin 5), i ≠ j → j ≠ k → i ≠ k → 
       triangleArea (points i) (points j) (points k) ≥ 2) :
  ∃ (i j k : Fin 5), i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ 
    triangleArea (points i) (points j) (points k) ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_at_least_three_l1042_104224


namespace NUMINAMATH_CALUDE_two_zeros_implies_a_geq_two_l1042_104268

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 1 then 2 * x - a else Real.log (1 - x)

theorem two_zeros_implies_a_geq_two (a : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ f a x = 0 ∧ f a y = 0) ∧
  (∀ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z → ¬(f a x = 0 ∧ f a y = 0 ∧ f a z = 0)) →
  a ≥ 2 :=
sorry

end NUMINAMATH_CALUDE_two_zeros_implies_a_geq_two_l1042_104268


namespace NUMINAMATH_CALUDE_difference_of_squares_division_l1042_104263

theorem difference_of_squares_division : (245^2 - 205^2) / 40 = 450 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_division_l1042_104263


namespace NUMINAMATH_CALUDE_radio_loss_percentage_l1042_104207

/-- Given the cost price and selling price of a radio, prove the loss percentage. -/
theorem radio_loss_percentage
  (cost_price : ℝ)
  (selling_price : ℝ)
  (h1 : cost_price = 2400)
  (h2 : selling_price = 2100) :
  (cost_price - selling_price) / cost_price * 100 = 12.5 := by
  sorry

end NUMINAMATH_CALUDE_radio_loss_percentage_l1042_104207


namespace NUMINAMATH_CALUDE_complement_of_M_in_U_l1042_104238

-- Define the universal set U
def U : Set ℤ := {-1, -2, -3, 0, 1}

-- Define set M
def M (a : ℤ) : Set ℤ := {-1, 0, a^2 + 1}

-- Theorem statement
theorem complement_of_M_in_U (a : ℤ) (h : M a ⊆ U) :
  U \ M a = {-2, -3} := by sorry

end NUMINAMATH_CALUDE_complement_of_M_in_U_l1042_104238


namespace NUMINAMATH_CALUDE_sqrt_meaningful_range_l1042_104212

theorem sqrt_meaningful_range (x : ℝ) : 
  (∃ y : ℝ, y^2 = x + 1) ↔ x ≥ -1 :=
sorry

end NUMINAMATH_CALUDE_sqrt_meaningful_range_l1042_104212


namespace NUMINAMATH_CALUDE_rectangleB_is_top_leftmost_l1042_104261

-- Define a structure for rectangles
structure Rectangle where
  w : ℕ
  x : ℕ
  y : ℕ
  z : ℕ

-- Define the six rectangles
def rectangleA : Rectangle := ⟨2, 7, 4, 7⟩
def rectangleB : Rectangle := ⟨0, 6, 8, 5⟩
def rectangleC : Rectangle := ⟨6, 3, 1, 1⟩
def rectangleD : Rectangle := ⟨8, 4, 0, 2⟩
def rectangleE : Rectangle := ⟨5, 9, 3, 6⟩
def rectangleF : Rectangle := ⟨7, 5, 9, 0⟩

-- Define a function to check if a rectangle is leftmost
def isLeftmost (r : Rectangle) : Prop :=
  ∀ other : Rectangle, r.w ≤ other.w

-- Define a function to check if a rectangle is topmost among leftmost rectangles
def isTopmostLeftmost (r : Rectangle) : Prop :=
  isLeftmost r ∧ ∀ other : Rectangle, isLeftmost other → r.y ≥ other.y

-- Theorem stating that Rectangle B is the top leftmost rectangle
theorem rectangleB_is_top_leftmost :
  isTopmostLeftmost rectangleB :=
sorry


end NUMINAMATH_CALUDE_rectangleB_is_top_leftmost_l1042_104261


namespace NUMINAMATH_CALUDE_baron_weights_partition_l1042_104206

/-- A set of weights satisfying the Baron's conditions -/
def BaronWeights : Type := 
  { s : Finset ℕ // s.card = 50 ∧ ∀ x ∈ s, x ≤ 100 ∧ Even (s.sum id) }

/-- The proposition that the weights can be partitioned into two subsets with equal sums -/
def CanPartition (weights : BaronWeights) : Prop :=
  ∃ (s₁ s₂ : Finset ℕ), s₁ ∪ s₂ = weights.val ∧ s₁ ∩ s₂ = ∅ ∧ s₁.sum id = s₂.sum id

/-- The theorem stating that any set of weights satisfying the Baron's conditions can be partitioned -/
theorem baron_weights_partition (weights : BaronWeights) : CanPartition weights := by
  sorry


end NUMINAMATH_CALUDE_baron_weights_partition_l1042_104206


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l1042_104251

/-- Perimeter of a rectangle with area equal to a right triangle --/
theorem rectangle_perimeter (a b c : ℝ) (h1 : a^2 + b^2 = c^2) (h2 : a * b = 108) : 
  let triangle_area := a * b / 2
  let rectangle_length := c / 2
  let rectangle_width := triangle_area / rectangle_length
  2 * (rectangle_length + rectangle_width) = 29.4 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l1042_104251


namespace NUMINAMATH_CALUDE_weaving_problem_l1042_104278

/-- An arithmetic sequence with the given properties -/
structure ArithmeticSequence where
  a : ℕ → ℕ  -- The sequence
  first_three_sum : a 1 + a 2 + a 3 = 9
  second_fourth_sixth_sum : a 2 + a 4 + a 6 = 15

/-- The sum of the first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℕ :=
  (List.range n).map seq.a |>.sum

theorem weaving_problem (seq : ArithmeticSequence) : sum_n seq 7 = 35 := by
  sorry

end NUMINAMATH_CALUDE_weaving_problem_l1042_104278


namespace NUMINAMATH_CALUDE_girl_travel_distance_l1042_104200

/-- 
Given a constant speed and time, calculates the distance traveled.
-/
def distance_traveled (speed : ℝ) (time : ℝ) : ℝ :=
  speed * time

/-- 
Theorem: A girl traveling at 4 m/s for 32 seconds covers a distance of 128 meters.
-/
theorem girl_travel_distance : 
  distance_traveled 4 32 = 128 := by
  sorry

end NUMINAMATH_CALUDE_girl_travel_distance_l1042_104200
