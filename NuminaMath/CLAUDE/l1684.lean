import Mathlib

namespace NUMINAMATH_CALUDE_positions_after_179_moves_l1684_168458

/-- Represents the positions of the cat -/
inductive CatPosition
| TopLeft
| TopRight
| BottomRight
| BottomLeft

/-- Represents the positions of the mouse -/
inductive MousePosition
| TopMiddle
| TopRight
| RightMiddle
| BottomRight
| BottomMiddle
| BottomLeft
| LeftMiddle
| TopLeft

/-- Calculates the position of the cat after a given number of moves -/
def catPositionAfterMoves (moves : Nat) : CatPosition :=
  match moves % 4 with
  | 0 => CatPosition.BottomLeft
  | 1 => CatPosition.TopLeft
  | 2 => CatPosition.TopRight
  | _ => CatPosition.BottomRight

/-- Calculates the position of the mouse after a given number of moves -/
def mousePositionAfterMoves (moves : Nat) : MousePosition :=
  match moves % 8 with
  | 0 => MousePosition.TopLeft
  | 1 => MousePosition.TopMiddle
  | 2 => MousePosition.TopRight
  | 3 => MousePosition.RightMiddle
  | 4 => MousePosition.BottomRight
  | 5 => MousePosition.BottomMiddle
  | 6 => MousePosition.BottomLeft
  | _ => MousePosition.LeftMiddle

theorem positions_after_179_moves :
  (catPositionAfterMoves 179 = CatPosition.BottomRight) ∧
  (mousePositionAfterMoves 179 = MousePosition.RightMiddle) := by
  sorry

end NUMINAMATH_CALUDE_positions_after_179_moves_l1684_168458


namespace NUMINAMATH_CALUDE_rectangle_measurement_error_l1684_168414

theorem rectangle_measurement_error (x : ℝ) : 
  (1 + x / 100) * (1 - 4 / 100) = 100.8 / 100 → x = 5 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_measurement_error_l1684_168414


namespace NUMINAMATH_CALUDE_x_squared_geq_one_necessary_not_sufficient_for_x_geq_one_l1684_168481

theorem x_squared_geq_one_necessary_not_sufficient_for_x_geq_one :
  (∀ x : ℝ, x ≥ 1 → x^2 ≥ 1) ∧
  (∃ x : ℝ, x^2 ≥ 1 ∧ x < 1) := by
sorry

end NUMINAMATH_CALUDE_x_squared_geq_one_necessary_not_sufficient_for_x_geq_one_l1684_168481


namespace NUMINAMATH_CALUDE_player_one_wins_l1684_168477

/-- A cubic polynomial with integer coefficients -/
def CubicPolynomial (a b c : ℤ) : ℤ → ℤ := fun x ↦ x^3 + a*x^2 + b*x + c

/-- A proposition stating that a cubic polynomial has three integer roots -/
def HasThreeIntegerRoots (p : ℤ → ℤ) : Prop :=
  ∃ x y z : ℤ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ p x = 0 ∧ p y = 0 ∧ p z = 0

theorem player_one_wins :
  ∀ a b : ℤ, ∃ c : ℤ, HasThreeIntegerRoots (CubicPolynomial a b c) :=
by sorry

end NUMINAMATH_CALUDE_player_one_wins_l1684_168477


namespace NUMINAMATH_CALUDE_joint_investment_l1684_168448

def total_investment : ℝ := 5000

theorem joint_investment (x : ℝ) :
  ∃ (a b : ℝ),
    a + b = total_investment ∧
    a * (1 + x / 100) = 2100 ∧
    b * (1 + (x + 1) / 100) = 3180 ∧
    a = 2000 ∧
    b = 3000 :=
by sorry

end NUMINAMATH_CALUDE_joint_investment_l1684_168448


namespace NUMINAMATH_CALUDE_expression_equality_l1684_168465

theorem expression_equality : 
  (-2^3 = (-2)^3) ∧ 
  (2^3 ≠ 2*3) ∧ 
  (-((-2)^2) ≠ (-2)^2) ∧ 
  (-3^2 ≠ 3^2) := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l1684_168465


namespace NUMINAMATH_CALUDE_initial_apples_count_l1684_168423

theorem initial_apples_count (initial_apples : ℕ) : 
  initial_apples - 2 + (8 - 2 * 2) + (15 - (2 / 3 * 15)) = 14 → 
  initial_apples = 7 := by
sorry

end NUMINAMATH_CALUDE_initial_apples_count_l1684_168423


namespace NUMINAMATH_CALUDE_cow_chicken_problem_l1684_168480

/-- Given a group of cows and chickens, if the total number of legs is 18 more than
    twice the total number of heads, then the number of cows is 9. -/
theorem cow_chicken_problem (cows chickens : ℕ) : 
  4 * cows + 2 * chickens = 2 * (cows + chickens) + 18 → cows = 9 := by
  sorry

end NUMINAMATH_CALUDE_cow_chicken_problem_l1684_168480


namespace NUMINAMATH_CALUDE_intersection_of_modified_functions_l1684_168456

/-- Two functions that intersect at specific points -/
def IntersectingFunctions (p q : ℝ → ℝ) : Prop :=
  p 1 = q 1 ∧ p 1 = 1 ∧
  p 3 = q 3 ∧ p 3 = 3 ∧
  p 5 = q 5 ∧ p 5 = 5 ∧
  p 7 = q 7 ∧ p 7 = 7

/-- Theorem stating that given two functions p and q that intersect at specific points,
    the functions p(2x) and 2q(x) must intersect at (3.5, 7) -/
theorem intersection_of_modified_functions (p q : ℝ → ℝ) 
    (h : IntersectingFunctions p q) : 
    p (2 * 3.5) = 2 * q 3.5 ∧ p (2 * 3.5) = 7 := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_modified_functions_l1684_168456


namespace NUMINAMATH_CALUDE_incorrect_observation_value_l1684_168434

/-- Theorem: Given 20 observations with an original mean of 36, if one observation
    is corrected from an unknown value to 25, resulting in a new mean of 34.9,
    then the unknown (incorrect) value must have been 47. -/
theorem incorrect_observation_value
  (n : ℕ) -- number of observations
  (original_mean : ℝ) -- original mean
  (correct_value : ℝ) -- correct value of the observation
  (new_mean : ℝ) -- new mean after correction
  (h_n : n = 20)
  (h_original_mean : original_mean = 36)
  (h_correct_value : correct_value = 25)
  (h_new_mean : new_mean = 34.9)
  : ∃ (incorrect_value : ℝ),
    n * original_mean - incorrect_value + correct_value = n * new_mean ∧
    incorrect_value = 47 :=
sorry

end NUMINAMATH_CALUDE_incorrect_observation_value_l1684_168434


namespace NUMINAMATH_CALUDE_max_value_quadratic_l1684_168471

theorem max_value_quadratic (x : ℝ) (h : 0 < x ∧ x < 4) :
  ∃ (y : ℝ), y = x * (8 - 2 * x) ∧ ∀ (z : ℝ), z = x * (8 - 2 * x) → z ≤ y ∧ y = 8 := by
  sorry

end NUMINAMATH_CALUDE_max_value_quadratic_l1684_168471


namespace NUMINAMATH_CALUDE_equation_solution_l1684_168404

theorem equation_solution : 
  ∃! n : ℚ, (1 : ℚ) / (n + 2) + (3 : ℚ) / (n + 2) + n / (n + 2) = 4 ∧ n = -4/3 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1684_168404


namespace NUMINAMATH_CALUDE_percent_relation_l1684_168412

/-- Given that c is 25% of a and 10% of b, prove that b is 250% of a. -/
theorem percent_relation (a b c : ℝ) 
  (h1 : c = 0.25 * a) 
  (h2 : c = 0.10 * b) : 
  b = 2.5 * a := by
sorry

end NUMINAMATH_CALUDE_percent_relation_l1684_168412


namespace NUMINAMATH_CALUDE_same_sign_product_and_quotient_abs_l1684_168492

theorem same_sign_product_and_quotient_abs (a b : ℚ) (hb : b ≠ 0) :
  (a * b > 0 ↔ |a| / |b| > 0) ∧ (a * b < 0 ↔ |a| / |b| < 0) ∧ (a * b = 0 ↔ |a| / |b| = 0) :=
sorry

end NUMINAMATH_CALUDE_same_sign_product_and_quotient_abs_l1684_168492


namespace NUMINAMATH_CALUDE_reeya_average_score_l1684_168427

def reeya_scores : List ℝ := [65, 67, 76, 80, 95]

theorem reeya_average_score :
  (reeya_scores.sum / reeya_scores.length : ℝ) = 76.6 := by
  sorry

end NUMINAMATH_CALUDE_reeya_average_score_l1684_168427


namespace NUMINAMATH_CALUDE_student_selection_count_l1684_168469

def num_boys : ℕ := 4
def num_girls : ℕ := 3
def total_selected : ℕ := 4

def select_students (b g s : ℕ) : ℕ :=
  (Nat.choose b 3 * Nat.choose g 1) +
  (Nat.choose b 2 * Nat.choose g 2) +
  (Nat.choose b 1 * Nat.choose g 3)

theorem student_selection_count :
  select_students num_boys num_girls total_selected = 34 := by
  sorry

end NUMINAMATH_CALUDE_student_selection_count_l1684_168469


namespace NUMINAMATH_CALUDE_compound_hydrogen_atoms_l1684_168403

/-- Represents the number of atoms of each element in the compound -/
structure Compound where
  al : ℕ
  o : ℕ
  h : ℕ

/-- Represents the atomic weights of elements in g/mol -/
structure AtomicWeights where
  al : ℝ
  o : ℝ
  h : ℝ

/-- Calculates the molecular weight of a compound given its composition and atomic weights -/
def molecularWeight (c : Compound) (w : AtomicWeights) : ℝ :=
  c.al * w.al + c.o * w.o + c.h * w.h

/-- The theorem to be proved -/
theorem compound_hydrogen_atoms :
  let c : Compound := { al := 1, o := 3, h := 3 }
  let w : AtomicWeights := { al := 27, o := 16, h := 1 }
  molecularWeight c w = 78 := by
  sorry

end NUMINAMATH_CALUDE_compound_hydrogen_atoms_l1684_168403


namespace NUMINAMATH_CALUDE_max_polygon_length_8x8_grid_l1684_168489

/-- Represents a square grid -/
structure SquareGrid where
  size : ℕ

/-- Represents a polygon on a grid -/
structure GridPolygon where
  grid : SquareGrid
  length : ℕ
  closed : Bool
  self_avoiding : Bool

/-- The maximum length of a closed self-avoiding polygon on an 8x8 grid is 80 -/
theorem max_polygon_length_8x8_grid :
  ∃ (p : GridPolygon), p.grid.size = 8 ∧ p.closed ∧ p.self_avoiding ∧
    p.length = 80 ∧
    ∀ (q : GridPolygon), q.grid.size = 8 → q.closed → q.self_avoiding →
      q.length ≤ p.length := by
  sorry

end NUMINAMATH_CALUDE_max_polygon_length_8x8_grid_l1684_168489


namespace NUMINAMATH_CALUDE_cost_doubling_l1684_168485

theorem cost_doubling (t b : ℝ) (t_pos : t > 0) (b_pos : b > 0) :
  let original_cost := t * b^4
  let new_cost := t * (2*b)^4
  (new_cost / original_cost) * 100 = 1600 := by
  sorry

end NUMINAMATH_CALUDE_cost_doubling_l1684_168485


namespace NUMINAMATH_CALUDE_combine_like_terms_l1684_168415

theorem combine_like_terms (x y : ℝ) : 3 * x * y - 6 * x * y + (-2 * x * y) = -5 * x * y := by
  sorry

end NUMINAMATH_CALUDE_combine_like_terms_l1684_168415


namespace NUMINAMATH_CALUDE_four_propositions_two_correct_l1684_168420

theorem four_propositions_two_correct :
  (∀ A B : Set α, A ∩ B = A → A ⊆ B) ∧
  (∀ a : ℝ, (∃ x y : ℝ, a * x + y + 1 = 0 ∧ x - y + 1 = 0 ∧ (∀ x' y' : ℝ, a * x' + y' + 1 = 0 → x' - y' + 1 = 0 → (x, y) ≠ (x', y'))) → a = 1) ∧
  ¬(∀ p q : Prop, p ∨ q → p ∧ q) ∧
  ¬(∀ a b m : ℝ, a < b → a * m^2 < b * m^2) :=
by sorry

end NUMINAMATH_CALUDE_four_propositions_two_correct_l1684_168420


namespace NUMINAMATH_CALUDE_absolute_value_equality_l1684_168402

theorem absolute_value_equality (a b c d e f : ℝ) 
  (h1 : a * c * e ≠ 0)
  (h2 : ∀ x : ℝ, |a * x + b| + |c * x + d| = |e * x + f|) : 
  a * d = b * c := by sorry

end NUMINAMATH_CALUDE_absolute_value_equality_l1684_168402


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l1684_168473

theorem complex_fraction_simplification (z : ℂ) (h : z = -1 + I) :
  (z + 2) / (z^2 + z) = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l1684_168473


namespace NUMINAMATH_CALUDE_solve_average_weight_problem_l1684_168475

def average_weight_problem (initial_average : ℝ) (new_man_weight : ℝ) (weight_increase : ℝ) (crew_size : ℕ) : Prop :=
  let replaced_weight := new_man_weight - (crew_size : ℝ) * weight_increase
  replaced_weight = initial_average * (crew_size : ℝ) + weight_increase * (crew_size : ℝ) - new_man_weight

theorem solve_average_weight_problem :
  average_weight_problem 0 71 1.8 10 = true :=
sorry

end NUMINAMATH_CALUDE_solve_average_weight_problem_l1684_168475


namespace NUMINAMATH_CALUDE_geometric_sequence_first_term_value_l1684_168411

def geometric_sequence (a : ℕ → ℝ) := ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_first_term_value
  (a : ℕ → ℝ)
  (S : ℕ → ℝ)
  (t : ℝ)
  (h1 : a 1 = t)
  (h2 : geometric_sequence a)
  (h3 : ∀ n : ℕ+, a (n + 1) = 2 * S n + 1)
  : t = 1 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_first_term_value_l1684_168411


namespace NUMINAMATH_CALUDE_symmetric_complex_numbers_l1684_168407

theorem symmetric_complex_numbers (z₁ z₂ : ℂ) :
  (z₁ = 2 - 3*I) →
  (z₁ = -z₂) →
  (z₂ = -2 + 3*I) := by
sorry

end NUMINAMATH_CALUDE_symmetric_complex_numbers_l1684_168407


namespace NUMINAMATH_CALUDE_probability_at_least_one_mistake_l1684_168439

-- Define the probability of making a mistake on a single question
def p_mistake : ℝ := 0.1

-- Define the number of questions
def n_questions : ℕ := 3

-- Theorem statement
theorem probability_at_least_one_mistake :
  1 - (1 - p_mistake) ^ n_questions = 1 - 0.9 ^ 3 := by
  sorry

end NUMINAMATH_CALUDE_probability_at_least_one_mistake_l1684_168439


namespace NUMINAMATH_CALUDE_defective_shipped_percentage_l1684_168483

/-- Represents the percentage of defective units shipped from each stage -/
structure DefectiveShipped :=
  (stage1 : ℝ)
  (stage2 : ℝ)
  (stage3 : ℝ)

/-- Represents the percentage of defective units in each stage -/
structure DefectivePercentage :=
  (stage1 : ℝ)
  (stage2 : ℝ)
  (stage3 : ℝ)

/-- Represents the percentage of defective units shipped from each stage -/
structure ShippedPercentage :=
  (stage1 : ℝ)
  (stage2 : ℝ)
  (stage3 : ℝ)

/-- Calculates the percentage of total units that are defective and shipped -/
def calculate_defective_shipped (dp : DefectivePercentage) (sp : ShippedPercentage) : ℝ :=
  let ds : DefectiveShipped := {
    stage1 := dp.stage1 * sp.stage1,
    stage2 := (1 - dp.stage1) * dp.stage2 * sp.stage2,
    stage3 := (1 - dp.stage1) * (1 - dp.stage2) * dp.stage3 * sp.stage3
  }
  ds.stage1 + ds.stage2 + ds.stage3

/-- Theorem: Given the production process conditions, 2% of total units are defective and shipped -/
theorem defective_shipped_percentage :
  let dp : DefectivePercentage := { stage1 := 0.06, stage2 := 0.08, stage3 := 0.10 }
  let sp : ShippedPercentage := { stage1 := 0.05, stage2 := 0.07, stage3 := 0.10 }
  calculate_defective_shipped dp sp = 0.02 := by
  sorry


end NUMINAMATH_CALUDE_defective_shipped_percentage_l1684_168483


namespace NUMINAMATH_CALUDE_real_part_of_complex_fraction_l1684_168466

theorem real_part_of_complex_fraction : 
  Complex.re (5 / (1 - Complex.I * 2)) = 1 := by sorry

end NUMINAMATH_CALUDE_real_part_of_complex_fraction_l1684_168466


namespace NUMINAMATH_CALUDE_russian_players_pairing_probability_l1684_168451

/-- The probability of all Russian players being paired exclusively with other Russian players
    in a random pairing of 10 tennis players, where 4 are from Russia. -/
theorem russian_players_pairing_probability :
  let total_players : ℕ := 10
  let russian_players : ℕ := 4
  let probability : ℚ := (russian_players - 1) / (total_players - 1) *
                         (russian_players - 3) / (total_players - 3)
  probability = 1 / 21 := by
  sorry

end NUMINAMATH_CALUDE_russian_players_pairing_probability_l1684_168451


namespace NUMINAMATH_CALUDE_optimal_selling_price_l1684_168486

/-- The optimal selling price problem -/
theorem optimal_selling_price (purchase_price : ℝ) (initial_price : ℝ) (initial_volume : ℝ) 
  (price_volume_relation : ℝ → ℝ) (profit_function : ℝ → ℝ) :
  purchase_price = 40 →
  initial_price = 50 →
  initial_volume = 50 →
  (∀ x, price_volume_relation x = initial_volume - x) →
  (∀ x, profit_function x = (initial_price + x) * (price_volume_relation x) - purchase_price * (price_volume_relation x)) →
  ∃ max_profit : ℝ, ∀ x, profit_function x ≤ max_profit ∧ profit_function 20 = max_profit →
  initial_price + 20 = 70 := by
sorry

end NUMINAMATH_CALUDE_optimal_selling_price_l1684_168486


namespace NUMINAMATH_CALUDE_function_characterization_l1684_168474

theorem function_characterization (f : ℝ → ℝ) 
  (h1 : ∀ x : ℝ, x * (f (x + 1) - f x) = f x)
  (h2 : ∀ x y : ℝ, |f x - f y| ≤ |x - y|) :
  ∃ c : ℝ, (|c| ≤ 1) ∧ (∀ x : ℝ, f x = c * x) :=
sorry

end NUMINAMATH_CALUDE_function_characterization_l1684_168474


namespace NUMINAMATH_CALUDE_max_value_of_m_l1684_168430

theorem max_value_of_m (a b m : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : ∀ a b, a > 0 → b > 0 → m / (3 * a + b) - 3 / a - 1 / b ≤ 0) : 
  m ≤ 16 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_m_l1684_168430


namespace NUMINAMATH_CALUDE_sector_central_angle_l1684_168490

theorem sector_central_angle (radius : ℝ) (area : ℝ) (angle : ℝ) :
  radius = 1 →
  area = 1 →
  area = (1 / 2) * angle * radius^2 →
  angle = 2 := by
sorry

end NUMINAMATH_CALUDE_sector_central_angle_l1684_168490


namespace NUMINAMATH_CALUDE_negative_five_minus_two_i_in_third_quadrant_l1684_168495

/-- A complex number z is in the third quadrant if its real part is negative and its imaginary part is negative. -/
def is_in_third_quadrant (z : ℂ) : Prop :=
  z.re < 0 ∧ z.im < 0

/-- The complex number -5-2i is in the third quadrant. -/
theorem negative_five_minus_two_i_in_third_quadrant :
  is_in_third_quadrant (-5 - 2*I) := by
  sorry

end NUMINAMATH_CALUDE_negative_five_minus_two_i_in_third_quadrant_l1684_168495


namespace NUMINAMATH_CALUDE_property_P_implications_l1684_168442

def has_property_P (f : ℕ → ℕ) : Prop :=
  ∀ x : ℕ, f x + f (x + 2) ≤ 2 * f (x + 1)

def d (f : ℕ → ℕ) (x : ℕ) : ℤ :=
  f (x + 1) - f x

theorem property_P_implications (f : ℕ → ℕ) (h : has_property_P f) :
  (∀ x : ℕ, d f x ≥ 0 ∧ d f (x + 1) ≤ d f x) ∧
  (∃ c : ℕ, c ≤ d f 1 ∧ Set.Infinite {n : ℕ | d f n = c}) :=
sorry

end NUMINAMATH_CALUDE_property_P_implications_l1684_168442


namespace NUMINAMATH_CALUDE_susan_chair_count_l1684_168463

/-- The number of chairs in Susan's house -/
def total_chairs (red : ℕ) (yellow : ℕ) (blue : ℕ) : ℕ := red + yellow + blue

/-- Susan's chair collection -/
structure SusanChairs where
  red : ℕ
  yellow : ℕ
  blue : ℕ
  red_count : red = 5
  yellow_count : yellow = 4 * red
  blue_count : blue = yellow - 2

theorem susan_chair_count (s : SusanChairs) : total_chairs s.red s.yellow s.blue = 43 := by
  sorry

end NUMINAMATH_CALUDE_susan_chair_count_l1684_168463


namespace NUMINAMATH_CALUDE_a_gt_b_gt_c_l1684_168425

theorem a_gt_b_gt_c : 3^44 > 4^33 ∧ 4^33 > 5^22 := by
  sorry

end NUMINAMATH_CALUDE_a_gt_b_gt_c_l1684_168425


namespace NUMINAMATH_CALUDE_big_eighteen_soccer_league_games_l1684_168449

/-- Calculates the number of games in a soccer league with specific rules --/
def soccer_league_games (num_divisions : ℕ) (teams_per_division : ℕ) 
  (intra_division_games : ℕ) (inter_division_games : ℕ) : ℕ :=
  let intra_games := num_divisions * (teams_per_division * (teams_per_division - 1) / 2) * intra_division_games
  let inter_games := num_divisions * teams_per_division * (num_divisions - 1) * teams_per_division * inter_division_games
  (intra_games + inter_games) / 2

/-- The Big Eighteen Soccer League schedule theorem --/
theorem big_eighteen_soccer_league_games : 
  soccer_league_games 3 6 3 2 = 351 := by
  sorry

end NUMINAMATH_CALUDE_big_eighteen_soccer_league_games_l1684_168449


namespace NUMINAMATH_CALUDE_marble_statue_weight_l1684_168498

/-- The weight of a marble statue after three successive reductions -/
def final_weight (original : ℝ) : ℝ :=
  original * (1 - 0.28) * (1 - 0.18) * (1 - 0.20)

/-- Theorem stating the relationship between the original and final weights -/
theorem marble_statue_weight (original : ℝ) :
  final_weight original = 85.0176 → original = 144 := by
  sorry

#eval final_weight 144

end NUMINAMATH_CALUDE_marble_statue_weight_l1684_168498


namespace NUMINAMATH_CALUDE_gwen_recycled_amount_l1684_168441

-- Define the recycling rate
def recycling_rate : ℕ := 3

-- Define the points earned
def points_earned : ℕ := 6

-- Define the amount recycled by friends
def friends_recycled : ℕ := 13

-- Theorem to prove
theorem gwen_recycled_amount : 
  ∃ (gwen_amount : ℕ), 
    (gwen_amount + friends_recycled) / recycling_rate = points_earned ∧
    gwen_amount = 5 := by
  sorry

end NUMINAMATH_CALUDE_gwen_recycled_amount_l1684_168441


namespace NUMINAMATH_CALUDE_dogwood_planting_correct_l1684_168457

/-- The number of dogwood trees planted in a park --/
def dogwood_trees_planted (current : ℕ) (total : ℕ) : ℕ :=
  total - current

/-- Theorem stating that the number of dogwood trees planted is correct --/
theorem dogwood_planting_correct (current : ℕ) (total : ℕ) 
  (h : current ≤ total) : 
  dogwood_trees_planted current total = total - current :=
by
  sorry

#eval dogwood_trees_planted 34 83

end NUMINAMATH_CALUDE_dogwood_planting_correct_l1684_168457


namespace NUMINAMATH_CALUDE_point_on_x_axis_l1684_168421

/-- If a point P(m, m-3) lies on the x-axis, then its coordinates are (3, 0) -/
theorem point_on_x_axis (m : ℝ) :
  (m : ℝ) = m ∧ (m - 3 : ℝ) = 0 → (m : ℝ) = 3 ∧ (m - 3 : ℝ) = 0 :=
by sorry

end NUMINAMATH_CALUDE_point_on_x_axis_l1684_168421


namespace NUMINAMATH_CALUDE_mass_percentage_Ca_in_mixture_l1684_168488

/-- Molar mass of calcium in g/mol -/
def molar_mass_Ca : ℝ := 40.08

/-- Molar mass of oxygen in g/mol -/
def molar_mass_O : ℝ := 16.00

/-- Molar mass of carbon in g/mol -/
def molar_mass_C : ℝ := 12.01

/-- Molar mass of sulfur in g/mol -/
def molar_mass_S : ℝ := 32.07

/-- Molar mass of calcium oxide (CaO) in g/mol -/
def molar_mass_CaO : ℝ := molar_mass_Ca + molar_mass_O

/-- Molar mass of calcium carbonate (CaCO₃) in g/mol -/
def molar_mass_CaCO3 : ℝ := molar_mass_Ca + molar_mass_C + 3 * molar_mass_O

/-- Molar mass of calcium sulfate (CaSO₄) in g/mol -/
def molar_mass_CaSO4 : ℝ := molar_mass_Ca + molar_mass_S + 4 * molar_mass_O

/-- Percentage of CaO in the mixed compound -/
def percent_CaO : ℝ := 40

/-- Percentage of CaCO₃ in the mixed compound -/
def percent_CaCO3 : ℝ := 30

/-- Percentage of CaSO₄ in the mixed compound -/
def percent_CaSO4 : ℝ := 30

/-- Theorem: The mass percentage of Ca in the mixed compound is approximately 49.432% -/
theorem mass_percentage_Ca_in_mixture : 
  ∃ (x : ℝ), abs (x - 49.432) < 0.001 ∧ 
  x = (percent_CaO / 100 * (molar_mass_Ca / molar_mass_CaO * 100)) +
      (percent_CaCO3 / 100 * (molar_mass_Ca / molar_mass_CaCO3 * 100)) +
      (percent_CaSO4 / 100 * (molar_mass_Ca / molar_mass_CaSO4 * 100)) :=
by sorry

end NUMINAMATH_CALUDE_mass_percentage_Ca_in_mixture_l1684_168488


namespace NUMINAMATH_CALUDE_partner_c_investment_l1684_168478

/-- Calculates the investment of partner C in a partnership business --/
theorem partner_c_investment 
  (a_investment : ℕ) 
  (b_investment : ℕ) 
  (total_profit : ℕ) 
  (a_profit_share : ℕ) 
  (h1 : a_investment = 6300)
  (h2 : b_investment = 4200)
  (h3 : total_profit = 12300)
  (h4 : a_profit_share = 3690) :
  ∃ c_investment : ℕ, 
    c_investment = 10500 ∧ 
    (a_investment : ℚ) / (a_investment + b_investment + c_investment : ℚ) = 
    (a_profit_share : ℚ) / (total_profit : ℚ) :=
by sorry

end NUMINAMATH_CALUDE_partner_c_investment_l1684_168478


namespace NUMINAMATH_CALUDE_cyclic_sum_inequality_l1684_168431

theorem cyclic_sum_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  1 / (a^3 + b^3 + a*b*c) + 1 / (b^3 + c^3 + a*b*c) + 1 / (c^3 + a^3 + a*b*c) ≤ 1 / (a*b*c) := by
  sorry

end NUMINAMATH_CALUDE_cyclic_sum_inequality_l1684_168431


namespace NUMINAMATH_CALUDE_symmetric_points_on_number_line_l1684_168429

/-- Given points A, B, and C on a number line corresponding to real numbers a, b, and c respectively,
    with A and C symmetric with respect to B, a = √5, and b = 3, prove that c = 6 - √5. -/
theorem symmetric_points_on_number_line (a b c : ℝ) 
  (h_symmetric : b = (a + c) / 2) 
  (h_a : a = Real.sqrt 5) 
  (h_b : b = 3) : 
  c = 6 - Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_symmetric_points_on_number_line_l1684_168429


namespace NUMINAMATH_CALUDE_coloring_books_sold_l1684_168497

theorem coloring_books_sold (initial_stock : ℕ) (shelves : ℕ) (books_per_shelf : ℕ) : initial_stock = 87 → shelves = 9 → books_per_shelf = 6 → initial_stock - (shelves * books_per_shelf) = 33 := by
  sorry

end NUMINAMATH_CALUDE_coloring_books_sold_l1684_168497


namespace NUMINAMATH_CALUDE_fraction_problem_l1684_168444

theorem fraction_problem (x : ℚ) : x = 3/5 ↔ (2/5 * 300 : ℚ) - (x * 125) = 45 := by
  sorry

end NUMINAMATH_CALUDE_fraction_problem_l1684_168444


namespace NUMINAMATH_CALUDE_product_of_two_fifteens_l1684_168482

theorem product_of_two_fifteens : ∀ (a b : ℕ), a = 15 → b = 15 → a * b = 225 := by
  sorry

end NUMINAMATH_CALUDE_product_of_two_fifteens_l1684_168482


namespace NUMINAMATH_CALUDE_abs_diff_eq_diff_abs_implies_product_nonneg_but_not_conversely_l1684_168476

theorem abs_diff_eq_diff_abs_implies_product_nonneg_but_not_conversely (a b : ℝ) :
  (∀ a b : ℝ, |a - b| = |a| - |b| → a * b ≥ 0) ∧
  (∃ a b : ℝ, a * b ≥ 0 ∧ |a - b| ≠ |a| - |b|) :=
by sorry

end NUMINAMATH_CALUDE_abs_diff_eq_diff_abs_implies_product_nonneg_but_not_conversely_l1684_168476


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l1684_168438

/-- A hyperbola with given parameters a and b -/
structure Hyperbola (a b : ℝ) :=
  (ha : a > 0)
  (hb : b > 0)

/-- The eccentricity of a hyperbola -/
def eccentricity (h : Hyperbola a b) : ℝ := sorry

/-- The left focus of a hyperbola -/
def left_focus (h : Hyperbola a b) : ℝ × ℝ := sorry

/-- The right focus of a hyperbola -/
def right_focus (h : Hyperbola a b) : ℝ × ℝ := sorry

/-- The asymptotes of a hyperbola -/
def asymptotes (h : Hyperbola a b) : (ℝ × ℝ → Prop) × (ℝ × ℝ → Prop) := sorry

/-- A point is in the first quadrant -/
def is_in_first_quadrant (p : ℝ × ℝ) : Prop := sorry

/-- A point lies on a line -/
def lies_on (p : ℝ × ℝ) (l : ℝ × ℝ → Prop) : Prop := sorry

/-- Two lines are perpendicular -/
def perpendicular (l1 l2 : ℝ × ℝ → Prop) : Prop := sorry

/-- Two lines are parallel -/
def parallel (l1 l2 : ℝ × ℝ → Prop) : Prop := sorry

/-- The line through two points -/
def line_through (p1 p2 : ℝ × ℝ) : ℝ × ℝ → Prop := sorry

theorem hyperbola_eccentricity (a b : ℝ) (h : Hyperbola a b) 
  (p : ℝ × ℝ) (hp1 : is_in_first_quadrant p) 
  (hp2 : lies_on p (asymptotes h).1) 
  (hp3 : perpendicular (line_through p (left_focus h)) (asymptotes h).2)
  (hp4 : parallel (line_through p (right_focus h)) (asymptotes h).2) :
  eccentricity h = 2 := by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l1684_168438


namespace NUMINAMATH_CALUDE_original_price_calculation_l1684_168447

-- Define the discounts
def discount1 : ℝ := 0.20
def discount2 : ℝ := 0.05

-- Define the final price
def final_price : ℝ := 266

-- Theorem statement
theorem original_price_calculation :
  ∃ P : ℝ, P * (1 - discount1) * (1 - discount2) = final_price ∧ P = 350 := by
  sorry

end NUMINAMATH_CALUDE_original_price_calculation_l1684_168447


namespace NUMINAMATH_CALUDE_train_speed_calculation_l1684_168454

/-- Given a train of length 600 m crossing an overbridge of length 100 m in 70 seconds,
    prove that the speed of the train is 36 km/h. -/
theorem train_speed_calculation (train_length : Real) (overbridge_length : Real) (crossing_time : Real)
    (h1 : train_length = 600)
    (h2 : overbridge_length = 100)
    (h3 : crossing_time = 70) :
    (train_length + overbridge_length) / crossing_time * 3.6 = 36 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_calculation_l1684_168454


namespace NUMINAMATH_CALUDE_bacon_calories_per_strip_l1684_168400

theorem bacon_calories_per_strip 
  (total_calories : ℕ) 
  (bacon_percentage : ℚ) 
  (num_bacon_strips : ℕ) 
  (h1 : total_calories = 1250)
  (h2 : bacon_percentage = 1/5)
  (h3 : num_bacon_strips = 2) :
  (total_calories : ℚ) * bacon_percentage / num_bacon_strips = 125 := by
sorry

end NUMINAMATH_CALUDE_bacon_calories_per_strip_l1684_168400


namespace NUMINAMATH_CALUDE_systematic_sampling_l1684_168405

/-- Systematic sampling problem -/
theorem systematic_sampling
  (total_students : ℕ)
  (num_groups : ℕ)
  (group_size : ℕ)
  (group_16_number : ℕ)
  (h1 : total_students = 160)
  (h2 : num_groups = 20)
  (h3 : group_size = total_students / num_groups)
  (h4 : group_16_number = 126) :
  ∃ (first_group_number : ℕ),
    first_group_number ∈ Finset.range group_size ∧
    first_group_number + (16 - 1) * group_size = group_16_number :=
by sorry

end NUMINAMATH_CALUDE_systematic_sampling_l1684_168405


namespace NUMINAMATH_CALUDE_six_n_divisors_l1684_168462

def divisor_count (n : ℕ) : ℕ := (Nat.divisors n).card

theorem six_n_divisors (n : ℕ) 
  (h1 : divisor_count n = 10)
  (h2 : divisor_count (2 * n) = 20)
  (h3 : divisor_count (3 * n) = 15) :
  divisor_count (6 * n) = 30 := by
  sorry

end NUMINAMATH_CALUDE_six_n_divisors_l1684_168462


namespace NUMINAMATH_CALUDE_house_height_calculation_l1684_168413

/-- The height of Lily's house in feet -/
def house_height : ℝ := 56.25

/-- The length of the shadow cast by Lily's house in feet -/
def house_shadow : ℝ := 75

/-- The height of the tree in feet -/
def tree_height : ℝ := 15

/-- The length of the shadow cast by the tree in feet -/
def tree_shadow : ℝ := 20

/-- Theorem stating that the calculated house height is correct -/
theorem house_height_calculation :
  house_height = tree_height * (house_shadow / tree_shadow) :=
by sorry

end NUMINAMATH_CALUDE_house_height_calculation_l1684_168413


namespace NUMINAMATH_CALUDE_range_of_m_l1684_168446

/-- Proposition p: The equation x² + mx + 1 = 0 has two different negative real roots -/
def p (m : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ < 0 ∧ x₂ < 0 ∧ x₁ ≠ x₂ ∧ x₁^2 + m*x₁ + 1 = 0 ∧ x₂^2 + m*x₂ + 1 = 0

/-- Proposition q: The equation 4x² + 4(m-2)x + 1 = 0 has no real roots -/
def q (m : ℝ) : Prop :=
  ∀ x : ℝ, 4*x^2 + 4*(m-2)*x + 1 ≠ 0

/-- The main theorem stating the equivalence of the given conditions and the solution -/
theorem range_of_m :
  ∀ m : ℝ, ((p m ∨ q m) ∧ ¬(p m ∧ q m)) ↔ (m ≥ 3 ∨ (1 < m ∧ m ≤ 2)) :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l1684_168446


namespace NUMINAMATH_CALUDE_OPRQ_shape_l1684_168426

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a quadrilateral defined by four points -/
structure Quadrilateral where
  O : Point
  P : Point
  R : Point
  Q : Point

/-- Checks if a quadrilateral is a parallelogram -/
def isParallelogram (quad : Quadrilateral) : Prop :=
  (quad.P.x - quad.O.x, quad.P.y - quad.O.y) = (quad.R.x - quad.Q.x, quad.R.y - quad.Q.y) ∧
  (quad.Q.x - quad.O.x, quad.Q.y - quad.O.y) = (quad.R.x - quad.P.x, quad.R.y - quad.P.y)

/-- Checks if three points are collinear -/
def areCollinear (p1 p2 p3 : Point) : Prop :=
  (p2.x - p1.x) * (p3.y - p1.y) = (p3.x - p1.x) * (p2.y - p1.y)

/-- Checks if a quadrilateral is a straight line -/
def isStraightLine (quad : Quadrilateral) : Prop :=
  areCollinear quad.O quad.P quad.Q ∧ areCollinear quad.O quad.Q quad.R

/-- Checks if a quadrilateral is a trapezoid -/
def isTrapezoid (quad : Quadrilateral) : Prop :=
  ((quad.P.x - quad.O.x) * (quad.R.y - quad.Q.y) = (quad.R.x - quad.Q.x) * (quad.P.y - quad.O.y) ∧
   (quad.Q.x - quad.O.x) * (quad.R.y - quad.P.y) ≠ (quad.R.x - quad.P.x) * (quad.Q.y - quad.O.y)) ∨
  ((quad.Q.x - quad.O.x) * (quad.R.y - quad.P.y) = (quad.R.x - quad.P.x) * (quad.Q.y - quad.O.y) ∧
   (quad.P.x - quad.O.x) * (quad.R.y - quad.Q.y) ≠ (quad.R.x - quad.Q.x) * (quad.P.y - quad.O.y))

theorem OPRQ_shape (x₁ y₁ x₂ y₂ : ℝ) (h1 : x₁ ≠ x₂) (h2 : y₁ ≠ y₂) :
  let quad := Quadrilateral.mk
    (Point.mk 0 0)
    (Point.mk x₁ y₁)
    (Point.mk (x₁ + 2*x₂) (y₁ + 2*y₂))
    (Point.mk x₂ y₂)
  ¬(isParallelogram quad) ∧ ¬(isStraightLine quad) ∧ (isTrapezoid quad ∨ (¬(isParallelogram quad) ∧ ¬(isStraightLine quad) ∧ ¬(isTrapezoid quad))) := by
  sorry

end NUMINAMATH_CALUDE_OPRQ_shape_l1684_168426


namespace NUMINAMATH_CALUDE_max_dot_product_l1684_168493

/-- An ellipse with focus on the x-axis -/
structure Ellipse where
  /-- The b parameter in the ellipse equation x^2/4 + y^2/b^2 = 1 -/
  b : ℝ
  /-- The eccentricity of the ellipse -/
  e : ℝ
  /-- Condition that the eccentricity is 1/2 -/
  h_e : e = 1/2

/-- A point on the ellipse -/
structure PointOnEllipse (ε : Ellipse) where
  x : ℝ
  y : ℝ
  /-- The point satisfies the ellipse equation -/
  h_on_ellipse : x^2/4 + y^2/ε.b^2 = 1

/-- The left focus of the ellipse -/
def leftFocus (ε : Ellipse) : ℝ × ℝ := sorry

/-- The right vertex of the ellipse -/
def rightVertex (ε : Ellipse) : ℝ × ℝ := sorry

/-- The dot product of vectors PF and PA -/
def dotProduct (ε : Ellipse) (p : PointOnEllipse ε) : ℝ := sorry

/-- Theorem: The maximum value of the dot product of PF and PA is 4 -/
theorem max_dot_product (ε : Ellipse) :
  ∃ (max : ℝ), max = 4 ∧ ∀ (p : PointOnEllipse ε), dotProduct ε p ≤ max :=
sorry

end NUMINAMATH_CALUDE_max_dot_product_l1684_168493


namespace NUMINAMATH_CALUDE_construct_numbers_l1684_168424

/-- Given a natural number n, construct it using only the number 8, 
    arithmetic operations, and exponentiation -/
def construct_number (n : ℕ) : ℚ :=
  match n with
  | 1 => (8 / 8) ^ (8 / 8) * (8 / 8)
  | 2 => 8 / 8 + 8 / 8
  | 3 => (8 + 8 + 8) / 8
  | 4 => 8 / 8 + 8 / 8 + 8 / 8 + 8 / 8
  | 8 => 8
  | _ => 0  -- Default case, not all numbers are constructed

/-- Theorem stating that we can construct the numbers 1, 2, 3, 4, and 8
    using only the number 8, arithmetic operations, and exponentiation -/
theorem construct_numbers : 
  (construct_number 1 = 1) ∧ 
  (construct_number 2 = 2) ∧ 
  (construct_number 3 = 3) ∧ 
  (construct_number 4 = 4) ∧ 
  (construct_number 8 = 8) := by
  sorry


end NUMINAMATH_CALUDE_construct_numbers_l1684_168424


namespace NUMINAMATH_CALUDE_integer_fraction_count_l1684_168470

theorem integer_fraction_count : 
  ∃! (S : Finset ℕ), 
    (∀ m ∈ S, m > 0 ∧ ∃ k : ℕ, k > 0 ∧ 1722 = k * (m^2 - 3)) ∧ 
    S.card = 3 := by
  sorry

end NUMINAMATH_CALUDE_integer_fraction_count_l1684_168470


namespace NUMINAMATH_CALUDE_area_between_circles_l1684_168406

-- Define the radius of the inner circle
def inner_radius : ℝ := 2

-- Define the radius of the outer circle
def outer_radius : ℝ := 2 * inner_radius

-- Define the width of the gray region
def width : ℝ := outer_radius - inner_radius

-- Theorem statement
theorem area_between_circles (h : width = 2) : 
  π * outer_radius^2 - π * inner_radius^2 = 12 * π := by
  sorry

end NUMINAMATH_CALUDE_area_between_circles_l1684_168406


namespace NUMINAMATH_CALUDE_table_rotation_l1684_168467

theorem table_rotation (table_length table_width : ℝ) (S : ℕ) : 
  table_length = 9 →
  table_width = 12 →
  S = ⌈(table_length^2 + table_width^2).sqrt⌉ →
  S = 15 :=
by
  sorry

end NUMINAMATH_CALUDE_table_rotation_l1684_168467


namespace NUMINAMATH_CALUDE_saree_discount_problem_l1684_168484

/-- Proves that the first discount percentage is 20% given the conditions of the saree pricing problem --/
theorem saree_discount_problem (original_price : ℝ) (final_price : ℝ) (second_discount : ℝ) :
  original_price = 350 →
  final_price = 266 →
  second_discount = 0.05 →
  ∃ (first_discount : ℝ),
    final_price = original_price * (1 - first_discount) * (1 - second_discount) ∧
    first_discount = 0.20 := by
  sorry

end NUMINAMATH_CALUDE_saree_discount_problem_l1684_168484


namespace NUMINAMATH_CALUDE_sum_of_min_max_T_l1684_168409

theorem sum_of_min_max_T (B M T : ℝ) 
  (h1 : B^2 + M^2 + T^2 = 2022) 
  (h2 : B + M + T = 72) : 
  ∃ (Tmin Tmax : ℝ), 
    (∀ T' : ℝ, (∃ B' M' : ℝ, B'^2 + M'^2 + T'^2 = 2022 ∧ B' + M' + T' = 72) → Tmin ≤ T' ∧ T' ≤ Tmax) ∧
    Tmin + Tmax = 48 :=
sorry

end NUMINAMATH_CALUDE_sum_of_min_max_T_l1684_168409


namespace NUMINAMATH_CALUDE_max_a4_in_geometric_sequence_l1684_168440

theorem max_a4_in_geometric_sequence (a : ℕ → ℝ) :
  (∀ n : ℕ, a n > 0) →  -- positive sequence
  (∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n) →  -- geometric sequence
  a 3 + a 5 = 4 →  -- given condition
  ∀ x : ℝ, a 4 ≤ x → x ≤ 2  -- maximum value of a_4 is 2
:= by sorry

end NUMINAMATH_CALUDE_max_a4_in_geometric_sequence_l1684_168440


namespace NUMINAMATH_CALUDE_first_marvelous_monday_l1684_168417

/-- Represents a date with a year, month, and day. -/
structure Date where
  year : ℕ
  month : ℕ
  day : ℕ

/-- Represents a day of the week. -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Returns true if the given date is a Monday. -/
def isMonday (d : Date) (startDate : Date) (startDay : DayOfWeek) : Prop :=
  sorry

/-- Returns true if the given date is the fifth Monday of its month. -/
def isFifthMonday (d : Date) (startDate : Date) (startDay : DayOfWeek) : Prop :=
  sorry

/-- Returns true if date d1 is strictly after date d2. -/
def isAfter (d1 d2 : Date) : Prop :=
  sorry

theorem first_marvelous_monday 
  (schoolStartDate : Date)
  (h1 : schoolStartDate.year = 2023)
  (h2 : schoolStartDate.month = 9)
  (h3 : schoolStartDate.day = 11)
  (h4 : isMonday schoolStartDate schoolStartDate DayOfWeek.Monday) :
  ∃ (marvelousMonday : Date), 
    marvelousMonday.year = 2023 ∧ 
    marvelousMonday.month = 10 ∧ 
    marvelousMonday.day = 30 ∧
    isFifthMonday marvelousMonday schoolStartDate DayOfWeek.Monday ∧
    isAfter marvelousMonday schoolStartDate ∧
    ∀ (d : Date), 
      isFifthMonday d schoolStartDate DayOfWeek.Monday → 
      isAfter d schoolStartDate → 
      ¬(isAfter d marvelousMonday) :=
by sorry

end NUMINAMATH_CALUDE_first_marvelous_monday_l1684_168417


namespace NUMINAMATH_CALUDE_factorial_difference_quotient_l1684_168496

theorem factorial_difference_quotient : (Nat.factorial 15 - Nat.factorial 14 - Nat.factorial 13) / Nat.factorial 11 = 30420 := by
  sorry

end NUMINAMATH_CALUDE_factorial_difference_quotient_l1684_168496


namespace NUMINAMATH_CALUDE_fifteenth_clap_theorem_l1684_168468

/-- Represents the circular track and the movement of A and B -/
structure CircularTrack where
  circumference : ℝ
  a_lap_time : ℝ
  b_lap_time : ℝ
  a_reverse_laps : ℕ

/-- Calculates the time and distance for A and B to clap hands 15 times -/
def clap_hands_15_times (track : CircularTrack) : ℝ × ℝ :=
  sorry

/-- Theorem stating the correct time and distance for the 15th clap -/
theorem fifteenth_clap_theorem (track : CircularTrack) 
  (h1 : track.circumference = 400)
  (h2 : track.a_lap_time = 4)
  (h3 : track.b_lap_time = 7)
  (h4 : track.a_reverse_laps = 10) :
  let (time, distance) := clap_hands_15_times track
  time = 66 + 2/11 ∧ distance = 3781 + 9/11 := by
  sorry

end NUMINAMATH_CALUDE_fifteenth_clap_theorem_l1684_168468


namespace NUMINAMATH_CALUDE_worker_a_completion_time_l1684_168464

/-- The time it takes for Worker A and Worker B to complete a job together and independently -/
def combined_time : ℝ := 2.857142857142857

/-- The time it takes for Worker B to complete the job alone -/
def worker_b_time : ℝ := 10

/-- The time it takes for Worker A to complete the job alone -/
def worker_a_time : ℝ := 4

/-- Theorem stating that Worker A takes 4 hours to complete the job alone -/
theorem worker_a_completion_time :
  (1 / worker_a_time + 1 / worker_b_time) * combined_time = 1 := by
  sorry

end NUMINAMATH_CALUDE_worker_a_completion_time_l1684_168464


namespace NUMINAMATH_CALUDE_fixed_point_on_line_l1684_168452

theorem fixed_point_on_line (m : ℝ) : (2 : ℝ) + 1 = m * ((2 : ℝ) - 2) := by sorry

end NUMINAMATH_CALUDE_fixed_point_on_line_l1684_168452


namespace NUMINAMATH_CALUDE_initial_milk_water_ratio_l1684_168450

/-- Proves that the initial ratio of milk to water in a mixture is 3:2, given specific conditions -/
theorem initial_milk_water_ratio
  (total_initial : ℝ)
  (water_added : ℝ)
  (milk : ℝ)
  (water : ℝ)
  (h1 : total_initial = 165)
  (h2 : water_added = 66)
  (h3 : milk + water = total_initial)
  (h4 : milk / (water + water_added) = 3 / 4)
  : milk / water = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_initial_milk_water_ratio_l1684_168450


namespace NUMINAMATH_CALUDE_right_triangle_area_l1684_168487

theorem right_triangle_area (a b : ℝ) (h1 : a = 24) (h2 : b = 30) : 
  (1/2 : ℝ) * a * b = 360 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_area_l1684_168487


namespace NUMINAMATH_CALUDE_quadratic_roots_fraction_l1684_168453

theorem quadratic_roots_fraction (x₁ x₂ : ℝ) : 
  x₁^2 - 2*x₁ - 8 = 0 → x₂^2 - 2*x₂ - 8 = 0 → (x₁ + x₂) / (x₁ * x₂) = -1/4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_fraction_l1684_168453


namespace NUMINAMATH_CALUDE_space_probe_distance_l1684_168494

theorem space_probe_distance (total_distance : ℕ) (distance_after_refuel : ℕ) 
  (h1 : total_distance = 5555555555555)
  (h2 : distance_after_refuel = 3333333333333) :
  total_distance - distance_after_refuel = 2222222222222 := by
  sorry

end NUMINAMATH_CALUDE_space_probe_distance_l1684_168494


namespace NUMINAMATH_CALUDE_units_digit_product_l1684_168479

theorem units_digit_product (n : ℕ) : (2^2023 * 5^2024 * 11^2025) % 10 = 0 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_product_l1684_168479


namespace NUMINAMATH_CALUDE_range_proof_l1684_168401

theorem range_proof (a b : ℝ) 
  (h1 : 1 ≤ a + b) (h2 : a + b ≤ 5) 
  (h3 : -1 ≤ a - b) (h4 : a - b ≤ 3) : 
  (0 ≤ a ∧ a ≤ 4) ∧ 
  (-1 ≤ b ∧ b ≤ 3) ∧ 
  (-2 ≤ 3*a - 2*b ∧ 3*a - 2*b ≤ 10) := by
  sorry

end NUMINAMATH_CALUDE_range_proof_l1684_168401


namespace NUMINAMATH_CALUDE_power_of_three_squared_cubed_squared_l1684_168443

theorem power_of_three_squared_cubed_squared :
  ((3^2)^3)^2 = 531441 := by
  sorry

end NUMINAMATH_CALUDE_power_of_three_squared_cubed_squared_l1684_168443


namespace NUMINAMATH_CALUDE_final_short_bushes_count_l1684_168445

/-- The number of short bushes in the park -/
def initial_short_bushes : ℕ := 37

/-- The number of tall trees in the park -/
def tall_trees : ℕ := 30

/-- The number of short bushes to be planted -/
def new_short_bushes : ℕ := 20

/-- The total number of short bushes after planting -/
def total_short_bushes : ℕ := initial_short_bushes + new_short_bushes

theorem final_short_bushes_count : total_short_bushes = 57 := by
  sorry

end NUMINAMATH_CALUDE_final_short_bushes_count_l1684_168445


namespace NUMINAMATH_CALUDE_solution_306_is_valid_l1684_168418

def is_valid_solution (a b c : Nat) : Prop :=
  a ≠ 0 ∧ 
  b = 0 ∧ 
  c ≠ 0 ∧ 
  a ≠ c ∧
  1995 * (a * 100 + c) = 1995 * a * 100 + 1995 * c

theorem solution_306_is_valid : is_valid_solution 3 0 6 := by
  sorry

#check solution_306_is_valid

end NUMINAMATH_CALUDE_solution_306_is_valid_l1684_168418


namespace NUMINAMATH_CALUDE_square_of_one_plus_i_l1684_168435

theorem square_of_one_plus_i (i : ℂ) (h : i^2 = -1) : (1 + i)^2 = 2*i := by
  sorry

end NUMINAMATH_CALUDE_square_of_one_plus_i_l1684_168435


namespace NUMINAMATH_CALUDE_triangle_area_l1684_168472

theorem triangle_area (a b c : ℝ) (B : ℝ) : 
  B = 2 * Real.pi / 3 →
  b = Real.sqrt 13 →
  a + c = 4 →
  (1 / 2) * a * c * Real.sin B = (3 * Real.sqrt 3) / 4 := by
sorry

end NUMINAMATH_CALUDE_triangle_area_l1684_168472


namespace NUMINAMATH_CALUDE_arithmetic_computation_l1684_168459

theorem arithmetic_computation : -12 * 3 - (-4 * -5) + (-8 * -6) + 2 = -6 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_computation_l1684_168459


namespace NUMINAMATH_CALUDE_major_axis_length_eccentricity_l1684_168436

/-- Definition of the ellipse E -/
def ellipse_E (x y : ℝ) : Prop := y^2 / 4 + x^2 / 3 = 1

/-- F₁ and F₂ are the foci of the ellipse E -/
axiom foci_on_ellipse : ∃ F₁ F₂ : ℝ × ℝ, ellipse_E F₁.1 F₁.2 ∧ ellipse_E F₂.1 F₂.2

/-- Point P lies on the ellipse E -/
axiom P_on_ellipse : ∃ P : ℝ × ℝ, ellipse_E P.1 P.2

/-- The length of the major axis of ellipse E is 4 -/
theorem major_axis_length : ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ 
  (∀ x y : ℝ, ellipse_E x y ↔ (x/a)^2 + (y/b)^2 = 1) ∧ 
  max a b = 2 :=
sorry

/-- The eccentricity of ellipse E is 1/2 -/
theorem eccentricity : ∃ e : ℝ, e = 1/2 ∧
  ∃ c a : ℝ, c^2 = 4 - 3 ∧ a^2 = 3 ∧ e = c/a :=
sorry

end NUMINAMATH_CALUDE_major_axis_length_eccentricity_l1684_168436


namespace NUMINAMATH_CALUDE_exists_larger_area_same_perimeter_l1684_168408

/-- A shape in a 2D plane -/
structure Shape where
  area : ℝ
  perimeter : ℝ

/-- Theorem stating the existence of a shape with larger area and same perimeter -/
theorem exists_larger_area_same_perimeter (Φ Φ' : Shape) 
  (h1 : Φ'.area ≥ Φ.area) 
  (h2 : Φ'.perimeter < Φ.perimeter) : 
  ∃ Ψ : Shape, Ψ.perimeter = Φ.perimeter ∧ Ψ.area > Φ.area := by
  sorry

end NUMINAMATH_CALUDE_exists_larger_area_same_perimeter_l1684_168408


namespace NUMINAMATH_CALUDE_sphere_surface_area_l1684_168491

theorem sphere_surface_area (V : ℝ) (r : ℝ) (h : V = 72 * Real.pi) :
  4 * Real.pi * r^2 = 36 * Real.pi * (2^(1/3))^2 :=
by
  sorry

end NUMINAMATH_CALUDE_sphere_surface_area_l1684_168491


namespace NUMINAMATH_CALUDE_point_B_coordinates_l1684_168410

/-- Given that point A(m+2, m) lies on the y-axis, prove that point B(m+5, m-1) has coordinates (3, -3) -/
theorem point_B_coordinates (m : ℝ) 
  (h_A_on_y_axis : m + 2 = 0) : 
  (m + 5, m - 1) = (3, -3) := by
  sorry

end NUMINAMATH_CALUDE_point_B_coordinates_l1684_168410


namespace NUMINAMATH_CALUDE_roberts_expenses_l1684_168437

theorem roberts_expenses (total : ℝ) (machinery : ℝ) (cash_percentage : ℝ) 
  (h1 : total = 250)
  (h2 : machinery = 125)
  (h3 : cash_percentage = 0.1)
  : total - machinery - (cash_percentage * total) = 100 := by
  sorry

end NUMINAMATH_CALUDE_roberts_expenses_l1684_168437


namespace NUMINAMATH_CALUDE_arithmetic_operations_l1684_168428

theorem arithmetic_operations :
  (12 - (-5) + (-4) - 8 = 5) ∧
  (-1 - (1 + 1/2) * (1/3) / (-4)^2 = -33/32) :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_operations_l1684_168428


namespace NUMINAMATH_CALUDE_action_figure_price_l1684_168455

theorem action_figure_price 
  (sneaker_cost : ℕ) 
  (initial_savings : ℕ) 
  (num_figures_sold : ℕ) 
  (money_left : ℕ) : 
  sneaker_cost = 90 →
  initial_savings = 15 →
  num_figures_sold = 10 →
  money_left = 25 →
  (sneaker_cost - initial_savings + money_left) / num_figures_sold = 10 := by
sorry

end NUMINAMATH_CALUDE_action_figure_price_l1684_168455


namespace NUMINAMATH_CALUDE_hair_group_existence_l1684_168461

theorem hair_group_existence (population : ℕ) (max_hairs : ℕ) 
  (h1 : population ≥ 8000000) 
  (h2 : max_hairs = 400000) : 
  ∃ (hair_count : ℕ), hair_count ≤ max_hairs ∧ 
  (∃ (group : Finset (Fin population)), 
    group.card ≥ 20 ∧ 
    ∀ (person : Fin population), person ∈ group → 
      (∃ (f : Fin population → ℕ), f person = hair_count ∧ f person ≤ max_hairs)) :=
sorry

end NUMINAMATH_CALUDE_hair_group_existence_l1684_168461


namespace NUMINAMATH_CALUDE_fib_sum_39_40_l1684_168422

-- Define the Fibonacci sequence
def fib : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fib n + fib (n + 1)

-- State the theorem
theorem fib_sum_39_40 : fib 39 + fib 40 = fib 41 := by
  sorry

end NUMINAMATH_CALUDE_fib_sum_39_40_l1684_168422


namespace NUMINAMATH_CALUDE_apple_water_bottle_difference_l1684_168416

theorem apple_water_bottle_difference (total_bottles : ℕ) (water_bottles : ℕ) (apple_bottles : ℕ) : 
  total_bottles = 54 →
  water_bottles = 2 * 12 →
  apple_bottles = total_bottles - water_bottles →
  apple_bottles - water_bottles = 6 := by
  sorry

end NUMINAMATH_CALUDE_apple_water_bottle_difference_l1684_168416


namespace NUMINAMATH_CALUDE_rational_square_fractional_parts_l1684_168432

def fractional_part (x : ℚ) : ℚ :=
  x - ↑(⌊x⌋)

theorem rational_square_fractional_parts (S : Set ℚ) :
  (∀ x ∈ S, fractional_part x ∈ {y | ∃ z ∈ S, fractional_part (z^2) = y}) →
  (∀ x ∈ S, fractional_part (x^2) ∈ {y | ∃ z ∈ S, fractional_part z = y}) →
  ∀ x ∈ S, ∃ n : ℤ, x = n := by
  sorry

end NUMINAMATH_CALUDE_rational_square_fractional_parts_l1684_168432


namespace NUMINAMATH_CALUDE_total_people_is_36_l1684_168499

/-- A circular arrangement of people shaking hands -/
structure HandshakeCircle where
  people : ℕ
  handshakes : ℕ
  smallest_set : ℕ

/-- The number of people in the circle equals the number of handshakes -/
def handshakes_equal_people (circle : HandshakeCircle) : Prop :=
  circle.people = circle.handshakes

/-- The smallest set size plus the remaining people equals the total people -/
def smallest_set_property (circle : HandshakeCircle) : Prop :=
  circle.smallest_set + (circle.people - circle.smallest_set) = circle.people

/-- The main theorem: given the conditions, prove the total number of people is 36 -/
theorem total_people_is_36 (circle : HandshakeCircle) 
    (h1 : circle.handshakes = 36)
    (h2 : circle.smallest_set = 12)
    (h3 : handshakes_equal_people circle)
    (h4 : smallest_set_property circle) : 
  circle.people = 36 := by
  sorry

#check total_people_is_36

end NUMINAMATH_CALUDE_total_people_is_36_l1684_168499


namespace NUMINAMATH_CALUDE_pythagorean_triple_product_divisible_by_six_l1684_168419

theorem pythagorean_triple_product_divisible_by_six (A B C : ℤ) : 
  A^2 + B^2 = C^2 → (6 ∣ A * B) := by
sorry

end NUMINAMATH_CALUDE_pythagorean_triple_product_divisible_by_six_l1684_168419


namespace NUMINAMATH_CALUDE_square_sum_equality_l1684_168460

theorem square_sum_equality (x : ℝ) : (x + 2)^2 + 2*(x + 2)*(5 - x) + (5 - x)^2 = 49 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_equality_l1684_168460


namespace NUMINAMATH_CALUDE_trig_identity_l1684_168433

theorem trig_identity (a : ℝ) (h : Real.sin (π / 6 - a) - Real.cos a = 1 / 3) :
  Real.cos (2 * a + π / 3) = 7 / 9 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l1684_168433
