import Mathlib

namespace NUMINAMATH_CALUDE_min_value_xyz_l3673_367362

theorem min_value_xyz (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h_prod : x * y * z = 27) :
  ∀ a b c : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧ a * b * c = 27 → x + 3 * y + 6 * z ≤ a + 3 * b + 6 * c ∧
  ∃ x y z : ℝ, x > 0 ∧ y > 0 ∧ z > 0 ∧ x * y * z = 27 ∧ x + 3 * y + 6 * z = 27 :=
sorry

end NUMINAMATH_CALUDE_min_value_xyz_l3673_367362


namespace NUMINAMATH_CALUDE_valerie_laptop_savings_l3673_367347

/-- Proves that Valerie needs 30 weeks to save for a laptop -/
theorem valerie_laptop_savings :
  let laptop_price : ℕ := 800
  let parents_money : ℕ := 100
  let uncle_money : ℕ := 60
  let siblings_money : ℕ := 40
  let weekly_tutoring_income : ℕ := 20
  let total_graduation_money : ℕ := parents_money + uncle_money + siblings_money
  let remaining_amount : ℕ := laptop_price - total_graduation_money
  let weeks_needed : ℕ := remaining_amount / weekly_tutoring_income
  weeks_needed = 30 := by
sorry


end NUMINAMATH_CALUDE_valerie_laptop_savings_l3673_367347


namespace NUMINAMATH_CALUDE_correct_equation_proof_l3673_367314

def quadratic_equation (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

def has_roots (a b c : ℝ) (r₁ r₂ : ℝ) : Prop :=
  quadratic_equation a b c r₁ = 0 ∧ quadratic_equation a b c r₂ = 0

theorem correct_equation_proof :
  ∃ (a₁ b₁ c₁ : ℝ) (a₂ b₂ c₂ : ℝ),
    has_roots a₁ b₁ c₁ 8 2 ∧
    has_roots a₂ b₂ c₂ (-9) (-1) ∧
    (a₁ = 1 ∧ b₁ = -10 ∧ c₁ ≠ 9) ∧
    (a₂ = 1 ∧ b₂ ≠ -10 ∧ c₂ = 9) ∧
    quadratic_equation 1 (-10) 9 = quadratic_equation a₁ b₁ c₁ ∧
    quadratic_equation 1 (-10) 9 = quadratic_equation a₂ b₂ c₂ :=
by sorry

end NUMINAMATH_CALUDE_correct_equation_proof_l3673_367314


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_l3673_367356

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | x ≤ 1}
def B (a : ℝ) : Set ℝ := {x : ℝ | x ≥ a}

-- State the theorem
theorem necessary_but_not_sufficient :
  (∀ a : ℝ, a = 1 → A ∪ B a = Set.univ) ∧
  (∃ a : ℝ, a ≤ 1 ∧ a ≠ 1 ∧ A ∪ B a = Set.univ) :=
by sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_l3673_367356


namespace NUMINAMATH_CALUDE_product_xyz_equals_25_l3673_367338

/-- Given complex numbers x, y, and z satisfying specific equations, prove that their product is 25. -/
theorem product_xyz_equals_25 
  (x y z : ℂ) 
  (eq1 : 2 * x * y + 5 * y = -20)
  (eq2 : 2 * y * z + 5 * z = -20)
  (eq3 : 2 * z * x + 5 * x = -20) :
  x * y * z = 25 := by
  sorry

end NUMINAMATH_CALUDE_product_xyz_equals_25_l3673_367338


namespace NUMINAMATH_CALUDE_evaluate_expression_l3673_367324

theorem evaluate_expression (x y z : ℚ) 
  (hx : x = 1/2) (hy : y = 1/3) (hz : z = -3) : 
  x^2 * y^3 * z^2 = 1/12 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3673_367324


namespace NUMINAMATH_CALUDE_basketball_free_throws_l3673_367333

theorem basketball_free_throws :
  ∀ (two_points three_points free_throws : ℕ),
    3 * three_points = 2 * two_points →
    free_throws = 2 * two_points - 1 →
    2 * two_points + 3 * three_points + free_throws = 71 →
    free_throws = 23 := by
  sorry

end NUMINAMATH_CALUDE_basketball_free_throws_l3673_367333


namespace NUMINAMATH_CALUDE_negation_equivalence_l3673_367381

theorem negation_equivalence :
  (¬ (∀ x y : ℝ, x^2 + y^2 = 0 → x = 0 ∧ y = 0)) ↔
  (∀ x y : ℝ, x^2 + y^2 ≠ 0 → x ≠ 0 ∨ y ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l3673_367381


namespace NUMINAMATH_CALUDE_call_charge_for_550_seconds_l3673_367309

-- Define the local call charge rule
def local_call_charge (duration : ℕ) : ℚ :=
  let base_charge : ℚ := 22/100  -- 0.22 yuan for first 3 minutes
  let per_minute_charge : ℚ := 11/100  -- 0.11 yuan per minute after
  let full_minutes : ℕ := (duration + 59) / 60  -- Round up to nearest minute
  if full_minutes ≤ 3 then
    base_charge
  else
    base_charge + per_minute_charge * (full_minutes - 3 : ℚ)

-- Theorem statement
theorem call_charge_for_550_seconds :
  local_call_charge 550 = 99/100 := by
  sorry

end NUMINAMATH_CALUDE_call_charge_for_550_seconds_l3673_367309


namespace NUMINAMATH_CALUDE_isosceles_triangle_altitude_ratio_l3673_367331

/-- An isosceles triangle with base to side ratio 4:3 has its altitude dividing the side in ratio 2:1 -/
theorem isosceles_triangle_altitude_ratio :
  ∀ (a b h m n : ℝ),
  a > 0 → b > 0 → h > 0 → m > 0 → n > 0 →
  b = (4/3) * a →  -- base to side ratio is 4:3
  h^2 = a^2 - (b/2)^2 →  -- height formula
  a^2 = h^2 + m^2 →  -- right triangle formed by altitude
  a = m + n →  -- side divided by altitude
  m / n = 2 / 1 :=  -- ratio in which altitude divides the side
by
  sorry


end NUMINAMATH_CALUDE_isosceles_triangle_altitude_ratio_l3673_367331


namespace NUMINAMATH_CALUDE_exponent_calculation_l3673_367310

theorem exponent_calculation : 3 * 3^4 - 27^63 / 27^61 = -486 := by
  sorry

end NUMINAMATH_CALUDE_exponent_calculation_l3673_367310


namespace NUMINAMATH_CALUDE_complement_union_M_N_l3673_367369

def U : Set Nat := {1, 2, 3, 4, 5}
def M : Set Nat := {1, 2}
def N : Set Nat := {3, 4}

theorem complement_union_M_N : (U \ (M ∪ N)) = {5} := by
  sorry

end NUMINAMATH_CALUDE_complement_union_M_N_l3673_367369


namespace NUMINAMATH_CALUDE_prob_ace_heart_queen_l3673_367364

-- Define the structure of a standard deck
def StandardDeck : Type := Unit

-- Define card types
inductive Rank
| Ace | Two | Three | Four | Five | Six | Seven | Eight | Nine | Ten | Jack | Queen | King

inductive Suit
| Hearts | Diamonds | Clubs | Spades

structure Card where
  rank : Rank
  suit : Suit

-- Define the probability of drawing specific cards
def prob_first_ace (deck : StandardDeck) : ℚ := 4 / 52

def prob_second_heart (deck : StandardDeck) : ℚ := 13 / 51

def prob_third_queen (deck : StandardDeck) : ℚ := 4 / 50

-- State the theorem
theorem prob_ace_heart_queen (deck : StandardDeck) :
  prob_first_ace deck * prob_second_heart deck * prob_third_queen deck = 1 / 663 := by
  sorry

end NUMINAMATH_CALUDE_prob_ace_heart_queen_l3673_367364


namespace NUMINAMATH_CALUDE_descending_order_always_possible_ascending_order_sometimes_impossible_l3673_367341

-- Define the grid
def Grid := Fin 10 → Fin 10 → Bool

-- Define piece sizes
inductive PieceSize
| One
| Two
| Three
| Four

-- Define a piece
structure Piece where
  size : PieceSize
  position : Fin 10 × Fin 10
  horizontal : Bool

-- Define the set of pieces
def PieceSet := List Piece

-- Check if a placement is valid
def is_valid_placement (grid : Grid) (pieces : PieceSet) : Prop := sorry

-- Define descending order placement
def descending_order_placement (grid : Grid) (pieces : PieceSet) : Prop := sorry

-- Define ascending order placement
def ascending_order_placement (grid : Grid) (pieces : PieceSet) : Prop := sorry

-- Theorem for descending order placement
theorem descending_order_always_possible (grid : Grid) (pieces : PieceSet) :
  descending_order_placement grid pieces → is_valid_placement grid pieces := by sorry

-- Theorem for ascending order placement
theorem ascending_order_sometimes_impossible : 
  ∃ (grid : Grid) (pieces : PieceSet), 
    ascending_order_placement grid pieces ∧ ¬is_valid_placement grid pieces := by sorry

end NUMINAMATH_CALUDE_descending_order_always_possible_ascending_order_sometimes_impossible_l3673_367341


namespace NUMINAMATH_CALUDE_greater_number_on_cards_l3673_367327

theorem greater_number_on_cards (x y : ℤ) 
  (sum_eq : x + y = 1443) 
  (diff_eq : x - y = 141) : 
  x = 792 ∧ x > y :=
by sorry

end NUMINAMATH_CALUDE_greater_number_on_cards_l3673_367327


namespace NUMINAMATH_CALUDE_kelly_apples_l3673_367336

/-- Given Kelly's initial apples and the number of apples she needs to pick,
    calculate the total number of apples she will have. -/
def total_apples (initial : ℕ) (to_pick : ℕ) : ℕ :=
  initial + to_pick

/-- Theorem stating that Kelly will have 105 apples altogether -/
theorem kelly_apples :
  total_apples 56 49 = 105 := by
  sorry

end NUMINAMATH_CALUDE_kelly_apples_l3673_367336


namespace NUMINAMATH_CALUDE_inequality_condition_l3673_367385

theorem inequality_condition (a b c : ℝ) :
  (∀ x : ℝ, a * Real.sin x + b * Real.cos x + c > 0) ↔ Real.sqrt (a^2 + b^2) < c :=
sorry

end NUMINAMATH_CALUDE_inequality_condition_l3673_367385


namespace NUMINAMATH_CALUDE_alissa_earrings_l3673_367370

/-- Represents the number of pairs of earrings Barbie bought -/
def barbie_pairs : ℕ := 12

/-- Represents the number of earrings Barbie gave to Alissa -/
def earrings_given : ℕ := barbie_pairs * 2 / 2

/-- Represents Alissa's total number of earrings after receiving the gift -/
def alissa_total : ℕ := 3 * earrings_given

theorem alissa_earrings : alissa_total = 36 := by
  sorry

end NUMINAMATH_CALUDE_alissa_earrings_l3673_367370


namespace NUMINAMATH_CALUDE_polygon_exterior_angles_l3673_367392

theorem polygon_exterior_angles (n : ℕ) (exterior_angle : ℝ) : 
  (n > 2) → (exterior_angle = 30) → (n * exterior_angle = 360) → n = 12 := by
  sorry

end NUMINAMATH_CALUDE_polygon_exterior_angles_l3673_367392


namespace NUMINAMATH_CALUDE_probability_sum_15_l3673_367320

/-- The number of ways to roll a sum of 15 with five six-sided dice -/
def waysToRoll15 : ℕ := 95

/-- The total number of possible outcomes when rolling five six-sided dice -/
def totalOutcomes : ℕ := 6^5

/-- A fair, standard six-sided die -/
structure Die :=
  (faces : Finset ℕ)
  (fair : faces = {1, 2, 3, 4, 5, 6})

/-- The probability of rolling a sum of 15 with five fair, standard six-sided dice -/
theorem probability_sum_15 (d1 d2 d3 d4 d5 : Die) :
  (waysToRoll15 : ℚ) / totalOutcomes = 95 / 7776 := by
  sorry


end NUMINAMATH_CALUDE_probability_sum_15_l3673_367320


namespace NUMINAMATH_CALUDE_factor_tree_problem_l3673_367311

theorem factor_tree_problem (H I F G X : ℕ) : 
  H = 7 * 2 →
  I = 11 * 2 →
  F = 7 * H →
  G = 11 * I →
  X = F * G →
  X = 23716 :=
by sorry

end NUMINAMATH_CALUDE_factor_tree_problem_l3673_367311


namespace NUMINAMATH_CALUDE_triangle_angle_theorem_l3673_367339

/-- A triple of integers representing the angles of a triangle in degrees. -/
structure TriangleAngles where
  a : ℕ
  b : ℕ
  c : ℕ
  sum_eq_180 : a + b + c = 180
  all_positive : 0 < a ∧ 0 < b ∧ 0 < c
  all_acute : a < 90 ∧ b < 90 ∧ c < 90

/-- The set of valid angle combinations for the triangle. -/
def validCombinations : Set TriangleAngles := {
  ⟨42, 72, 66, by norm_num, by norm_num, by norm_num⟩,
  ⟨49, 54, 77, by norm_num, by norm_num, by norm_num⟩,
  ⟨56, 36, 88, by norm_num, by norm_num, by norm_num⟩,
  ⟨84, 63, 33, by norm_num, by norm_num, by norm_num⟩
}

/-- Theorem stating that the only valid angle combinations for the triangle
    are those in the validCombinations set. -/
theorem triangle_angle_theorem :
  ∀ t : TriangleAngles,
    (∃ k : ℕ, t.a = 7 * k) ∧
    (∃ l : ℕ, t.b = 9 * l) ∧
    (∃ m : ℕ, t.c = 11 * m) →
    t ∈ validCombinations := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_theorem_l3673_367339


namespace NUMINAMATH_CALUDE_cafe_pricing_l3673_367318

theorem cafe_pricing (s c p : ℝ) 
  (eq1 : 5 * s + 9 * c + 2 * p = 6.50)
  (eq2 : 7 * s + 14 * c + 3 * p = 9.45)
  (eq3 : 4 * s + 8 * c + p = 5.20) :
  s + c + p = 1.30 := by
  sorry

end NUMINAMATH_CALUDE_cafe_pricing_l3673_367318


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l3673_367345

structure RightTriangle :=
  (O X Y : ℝ × ℝ)
  (is_right : (X.1 - O.1) * (Y.1 - O.1) + (X.2 - O.2) * (Y.2 - O.2) = 0)
  (M : ℝ × ℝ)
  (N : ℝ × ℝ)
  (M_midpoint : M = ((X.1 + O.1) / 2, (X.2 + O.2) / 2))
  (N_midpoint : N = ((Y.1 + O.1) / 2, (Y.2 + O.2) / 2))
  (XN_length : Real.sqrt ((X.1 - N.1)^2 + (X.2 - N.2)^2) = 19)
  (YM_length : Real.sqrt ((Y.1 - M.1)^2 + (Y.2 - M.2)^2) = 22)

theorem right_triangle_hypotenuse (t : RightTriangle) :
  Real.sqrt ((t.X.1 - t.Y.1)^2 + (t.X.2 - t.Y.2)^2) = 26 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l3673_367345


namespace NUMINAMATH_CALUDE_largest_number_problem_l3673_367372

theorem largest_number_problem (a b c : ℝ) 
  (h_order : a < b ∧ b < c)
  (h_sum : a + b + c = 67)
  (h_diff_large : c - b = 7)
  (h_diff_small : b - a = 5) :
  c = 86 / 3 := by
sorry

end NUMINAMATH_CALUDE_largest_number_problem_l3673_367372


namespace NUMINAMATH_CALUDE_regular_hexagon_diagonals_l3673_367383

/-- Regular hexagon with side length, shortest diagonal, and longest diagonal -/
structure RegularHexagon where
  a : ℝ  -- side length
  b : ℝ  -- shortest diagonal
  d : ℝ  -- longest diagonal

/-- Theorem: In a regular hexagon, the shortest diagonal is √3 times the side length,
    and the longest diagonal is 4/√3 times the side length -/
theorem regular_hexagon_diagonals (h : RegularHexagon) :
  h.b = Real.sqrt 3 * h.a ∧ h.d = (4 * h.a) / Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_regular_hexagon_diagonals_l3673_367383


namespace NUMINAMATH_CALUDE_a_share_is_240_l3673_367353

/-- Calculates the share of profit for partner A given the initial investments,
    changes in investment, and total profit. -/
def calculate_share_a (initial_a initial_b : ℕ) (withdraw_a advance_b : ℕ) 
                      (total_months : ℕ) (change_month : ℕ) (total_profit : ℕ) : ℕ :=
  let investment_months_a := initial_a * change_month + (initial_a - withdraw_a) * (total_months - change_month)
  let investment_months_b := initial_b * change_month + (initial_b + advance_b) * (total_months - change_month)
  let total_investment_months := investment_months_a + investment_months_b
  (investment_months_a * total_profit) / total_investment_months

/-- Theorem stating that given the problem conditions, A's share of the profit is 240. -/
theorem a_share_is_240 : 
  calculate_share_a 3000 4000 1000 1000 12 8 630 = 240 := by
  sorry

end NUMINAMATH_CALUDE_a_share_is_240_l3673_367353


namespace NUMINAMATH_CALUDE_product_of_fractions_l3673_367316

theorem product_of_fractions : 
  (7 : ℚ) / 4 * 14 / 35 * 21 / 12 * 28 / 56 * 49 / 28 * 42 / 84 * 63 / 36 * 56 / 112 = 1201 / 12800 := by
  sorry

end NUMINAMATH_CALUDE_product_of_fractions_l3673_367316


namespace NUMINAMATH_CALUDE_machine_work_time_solution_l3673_367366

theorem machine_work_time_solution : ∃ x : ℝ, 
  (x > 0) ∧ 
  (1 / (x + 4) + 1 / (x + 2) + 1 / (2 * x + 6) = 1 / x) ∧ 
  (x = 2) := by
  sorry

end NUMINAMATH_CALUDE_machine_work_time_solution_l3673_367366


namespace NUMINAMATH_CALUDE_distance_spain_other_proof_l3673_367376

/-- The distance between Spain and the other country -/
def distance_spain_other : ℕ := 5404

/-- The total distance between two countries -/
def total_distance : ℕ := 7019

/-- The distance between Spain and Germany -/
def distance_spain_germany : ℕ := 1615

/-- Theorem stating that the distance between Spain and the other country
    is equal to the total distance minus the distance between Spain and Germany -/
theorem distance_spain_other_proof :
  distance_spain_other = total_distance - distance_spain_germany :=
by sorry

end NUMINAMATH_CALUDE_distance_spain_other_proof_l3673_367376


namespace NUMINAMATH_CALUDE_right_triangle_inradius_l3673_367321

/-- The inradius of a right triangle with side lengths 9, 12, and 15 is 3 -/
theorem right_triangle_inradius : ∀ (a b c r : ℝ),
  a = 9 ∧ b = 12 ∧ c = 15 →
  a^2 + b^2 = c^2 →
  (a + b + c) / 2 * r = (a * b) / 2 →
  r = 3 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_inradius_l3673_367321


namespace NUMINAMATH_CALUDE_cos_squared_pi_fourth_minus_alpha_l3673_367302

theorem cos_squared_pi_fourth_minus_alpha (α : Real) 
  (h : Real.tan (α + π/4) = 3/4) : 
  Real.cos (π/4 - α)^2 = 9/25 := by
  sorry

end NUMINAMATH_CALUDE_cos_squared_pi_fourth_minus_alpha_l3673_367302


namespace NUMINAMATH_CALUDE_john_hiking_probability_l3673_367301

theorem john_hiking_probability (p_rain : ℝ) (p_hike_given_rain : ℝ) (p_hike_given_sunny : ℝ)
  (h_rain : p_rain = 0.3)
  (h_hike_rain : p_hike_given_rain = 0.1)
  (h_hike_sunny : p_hike_given_sunny = 0.9) :
  p_rain * p_hike_given_rain + (1 - p_rain) * p_hike_given_sunny = 0.66 := by
  sorry

end NUMINAMATH_CALUDE_john_hiking_probability_l3673_367301


namespace NUMINAMATH_CALUDE_complex_number_coordinates_l3673_367397

theorem complex_number_coordinates : (Complex.I + 1)^2 * Complex.I = -2 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_number_coordinates_l3673_367397


namespace NUMINAMATH_CALUDE_circle_properties_l3673_367319

/-- The equation of a circle in the xy-plane -/
def CircleEquation (x y : ℝ) : Prop :=
  x^2 + y^2 + 4*x - 6*y - 3 = 0

/-- The center of the circle -/
def CircleCenter : ℝ × ℝ := (-2, 3)

/-- The radius of the circle -/
def CircleRadius : ℝ := 4

/-- Theorem: The given equation represents a circle with center (-2, 3) and radius 4 -/
theorem circle_properties :
  ∀ (x y : ℝ),
    CircleEquation x y ↔ (x - CircleCenter.1)^2 + (y - CircleCenter.2)^2 = CircleRadius^2 :=
by sorry

end NUMINAMATH_CALUDE_circle_properties_l3673_367319


namespace NUMINAMATH_CALUDE_hcf_problem_l3673_367391

theorem hcf_problem (a b : ℕ) (h1 : a = 391) (h2 : ∃ (hcf : ℕ), Nat.lcm a b = hcf * 13 * 17) : Nat.gcd a b = 23 := by
  sorry

end NUMINAMATH_CALUDE_hcf_problem_l3673_367391


namespace NUMINAMATH_CALUDE_factory_temporary_workers_percentage_l3673_367303

theorem factory_temporary_workers_percentage 
  (total_workers : ℕ) 
  (technician_percentage : ℚ) 
  (non_technician_percentage : ℚ) 
  (permanent_technician_percentage : ℚ) 
  (permanent_non_technician_percentage : ℚ) 
  (h1 : technician_percentage = 80 / 100)
  (h2 : non_technician_percentage = 20 / 100)
  (h3 : permanent_technician_percentage = 80 / 100)
  (h4 : permanent_non_technician_percentage = 20 / 100)
  (h5 : technician_percentage + non_technician_percentage = 1) :
  let permanent_workers := (technician_percentage * permanent_technician_percentage + 
                            non_technician_percentage * permanent_non_technician_percentage) * total_workers
  let temporary_workers := total_workers - permanent_workers
  temporary_workers / total_workers = 32 / 100 := by
sorry

end NUMINAMATH_CALUDE_factory_temporary_workers_percentage_l3673_367303


namespace NUMINAMATH_CALUDE_reading_speed_first_half_l3673_367352

/-- Given a book with specific reading conditions, calculate the reading speed for the first half. -/
theorem reading_speed_first_half (total_pages : ℕ) (second_half_speed : ℕ) (total_days : ℕ) : 
  total_pages = 500 → 
  second_half_speed = 5 → 
  total_days = 75 → 
  (total_pages / 2) / (total_days - (total_pages / 2) / second_half_speed) = 10 := by
  sorry

#check reading_speed_first_half

end NUMINAMATH_CALUDE_reading_speed_first_half_l3673_367352


namespace NUMINAMATH_CALUDE_constant_term_expansion_l3673_367332

theorem constant_term_expansion (x : ℝ) (x_neq_zero : x ≠ 0) :
  ∃ (f : ℝ → ℝ), (∀ y, f y = (y + 4/y - 4)^3) ∧
  (∃ c, ∀ y ≠ 0, f y = c + y * (f y - c) / y) ∧
  c = -160 := by
sorry

end NUMINAMATH_CALUDE_constant_term_expansion_l3673_367332


namespace NUMINAMATH_CALUDE_rocks_difference_l3673_367393

/-- Given the number of rocks collected by Joshua, Jose, and Albert, prove that Albert collected 20 more rocks than Jose. -/
theorem rocks_difference (joshua_rocks : ℕ) (jose_rocks : ℕ) (albert_rocks : ℕ)
  (h1 : joshua_rocks = 80)
  (h2 : jose_rocks = joshua_rocks - 14)
  (h3 : albert_rocks = joshua_rocks + 6) :
  albert_rocks - jose_rocks = 20 := by
sorry

end NUMINAMATH_CALUDE_rocks_difference_l3673_367393


namespace NUMINAMATH_CALUDE_sum_of_specific_S_values_l3673_367350

def S (n : ℕ) : ℤ :=
  if n % 2 = 0 then
    -n / 2
  else
    (n + 1) / 2

theorem sum_of_specific_S_values : S 17 + S 33 + S 50 = 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_specific_S_values_l3673_367350


namespace NUMINAMATH_CALUDE_grocery_store_problem_l3673_367322

/-- Represents the price of an item after applying discount and tax -/
structure ItemPrice where
  base : ℝ
  discount : ℝ
  tax : ℝ

/-- Calculates the final price of an item after applying discount and tax -/
def finalPrice (item : ItemPrice) : ℝ :=
  item.base * (1 - item.discount) * (1 + item.tax)

/-- Represents the grocery store problem -/
theorem grocery_store_problem :
  let spam : ItemPrice := { base := 3, discount := 0.1, tax := 0 }
  let peanutButter : ItemPrice := { base := 5, discount := 0, tax := 0.05 }
  let bread : ItemPrice := { base := 2, discount := 0, tax := 0 }
  let milk : ItemPrice := { base := 4, discount := 0.2, tax := 0.08 }
  let eggs : ItemPrice := { base := 3, discount := 0.05, tax := 0 }
  
  let totalAmount :=
    12 * finalPrice spam +
    3 * finalPrice peanutButter +
    4 * finalPrice bread +
    2 * finalPrice milk +
    1 * finalPrice eggs
  
  totalAmount = 65.92 := by sorry

end NUMINAMATH_CALUDE_grocery_store_problem_l3673_367322


namespace NUMINAMATH_CALUDE_find_a_l3673_367308

theorem find_a : ∃ a : ℕ, 
  (∀ k : ℤ, k ≠ 27 → ∃ m : ℤ, a - k^1964 = m * (27 - k)) → 
  a = 27^1964 := by
sorry

end NUMINAMATH_CALUDE_find_a_l3673_367308


namespace NUMINAMATH_CALUDE_geometric_sequence_log_sum_l3673_367349

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_log_sum
  (a : ℕ → ℝ)
  (h_geom : GeometricSequence a)
  (h_pos : ∀ n, a n > 0)
  (h_prod : a 10 * a 11 = Real.exp 5) :
  (Finset.range 20).sum (λ i => Real.log (a (i + 1))) = 50 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_log_sum_l3673_367349


namespace NUMINAMATH_CALUDE_smallest_n_multiple_of_five_l3673_367361

theorem smallest_n_multiple_of_five (x y : ℤ) 
  (hx : 5 ∣ (x - 2)) 
  (hy : 5 ∣ (y + 4)) : 
  ∃ n : ℕ+, 
    5 ∣ (x^2 + 2*x*y + y^2 + n) ∧ 
    ∀ m : ℕ+, (5 ∣ (x^2 + 2*x*y + y^2 + m) → n ≤ m) ∧
    n = 1 := by
sorry

end NUMINAMATH_CALUDE_smallest_n_multiple_of_five_l3673_367361


namespace NUMINAMATH_CALUDE_ratio_of_multiples_l3673_367373

theorem ratio_of_multiples (w x y z : ℝ) 
  (hx : x = 4 * y) 
  (hy : y = 3 * z) 
  (hz : z = 5 * w) : 
  (x * z) / (y * w) = 20 := by
sorry

end NUMINAMATH_CALUDE_ratio_of_multiples_l3673_367373


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l3673_367387

theorem sufficient_not_necessary (a : ℝ) :
  (∀ a, a > 1 → a^2 > 1) ∧
  (∃ a, a ≤ 1 ∧ a^2 > 1) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l3673_367387


namespace NUMINAMATH_CALUDE_fraction_equals_zero_l3673_367390

theorem fraction_equals_zero (x : ℝ) :
  (x^2 - 1) / (1 - x) = 0 ∧ 1 - x ≠ 0 → x = -1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equals_zero_l3673_367390


namespace NUMINAMATH_CALUDE_negation_equivalence_l3673_367374

theorem negation_equivalence :
  (¬ ∃ x : ℝ, (2 : ℝ) ^ x < x ^ 2) ↔ (∀ x : ℝ, (2 : ℝ) ^ x ≥ x ^ 2) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l3673_367374


namespace NUMINAMATH_CALUDE_shared_foci_implies_m_equals_one_l3673_367377

/-- Given an ellipse and a hyperbola that share the same foci, prove that m = 1 -/
theorem shared_foci_implies_m_equals_one (m : ℝ) :
  (∀ x y : ℝ, x^2/4 + y^2/m^2 = 1 ↔ x^2/m - y^2/2 = 1) →
  (∃ c : ℝ, c^2 = 4 - m^2 ∧ c^2 = m + 2) →
  m = 1 := by
  sorry

end NUMINAMATH_CALUDE_shared_foci_implies_m_equals_one_l3673_367377


namespace NUMINAMATH_CALUDE_root_product_equation_l3673_367371

theorem root_product_equation (p q : ℝ) (α β γ δ : ℂ) : 
  (α^2 + p*α + 2 = 0) →
  (β^2 + p*β + 2 = 0) →
  (γ^2 + q*γ + 2 = 0) →
  (δ^2 + q*δ + 2 = 0) →
  (α - γ)*(β - γ)*(α + δ)*(β + δ) = 4 + 2*(p^2 - q^2) :=
by sorry

end NUMINAMATH_CALUDE_root_product_equation_l3673_367371


namespace NUMINAMATH_CALUDE_mixture_ratio_weight_l3673_367300

theorem mixture_ratio_weight (total_weight : ℝ) (ratio_a ratio_b : ℕ) (weight_b : ℝ) : 
  total_weight = 58.00000000000001 →
  ratio_a = 9 →
  ratio_b = 11 →
  weight_b = (ratio_b : ℝ) / ((ratio_a : ℝ) + (ratio_b : ℝ)) * total_weight →
  weight_b = 31.900000000000006 := by
sorry

end NUMINAMATH_CALUDE_mixture_ratio_weight_l3673_367300


namespace NUMINAMATH_CALUDE_inequality_solution_range_l3673_367396

theorem inequality_solution_range (a : ℝ) : 
  (∃ x : ℝ, |x + 1| - |x - 2| < a^2 - 4*a) ↔ (a < 1 ∨ a > 3) :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_range_l3673_367396


namespace NUMINAMATH_CALUDE_line_intercept_sum_l3673_367363

/-- Given a line 3x + 5y + d = 0, if the sum of its x-intercept and y-intercept is 15,
    then d = -225/8 -/
theorem line_intercept_sum (d : ℚ) : 
  (∃ x y : ℚ, 3 * x + 5 * y + d = 0 ∧ x + y = 15) → d = -225/8 := by
  sorry

end NUMINAMATH_CALUDE_line_intercept_sum_l3673_367363


namespace NUMINAMATH_CALUDE_taxi_driver_theorem_l3673_367307

def driving_distances : List Int := [5, -3, 6, -7, 6, -2, -5, -4, 6, -8]

def starting_price : ℕ := 8
def base_distance : ℕ := 3
def additional_rate : ℚ := 3/2

theorem taxi_driver_theorem :
  (List.sum driving_distances = -6) ∧
  (List.sum (List.take 7 driving_distances) = 0) ∧
  (starting_price + (8 - base_distance) * additional_rate = 31/2) ∧
  (∀ x : ℕ, x > base_distance → starting_price + (x - base_distance) * additional_rate = (3 * x + 7) / 2) :=
by sorry

end NUMINAMATH_CALUDE_taxi_driver_theorem_l3673_367307


namespace NUMINAMATH_CALUDE_f_range_and_triangle_area_l3673_367317

noncomputable def f (x : ℝ) : ℝ := 
  Real.sqrt 3 * (Real.cos x)^2 + Real.sin x * Real.cos x

def triangle_ABC (a b c : ℝ) (A B C : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ 
  A > 0 ∧ A < Real.pi ∧
  B > 0 ∧ B < Real.pi ∧
  C > 0 ∧ C < Real.pi ∧
  A + B + C = Real.pi

theorem f_range_and_triangle_area 
  (h1 : ∀ x, x ∈ Set.Icc 0 (Real.pi / 2) → f x ∈ Set.Icc 0 (1 + Real.sqrt 3 / 2))
  (h2 : ∃ A B C a b c, 
    triangle_ABC a b c A B C ∧ 
    f (A / 2) = Real.sqrt 3 ∧
    a = 4 ∧
    b + c = 5) :
  ∃ A B C a b c, 
    triangle_ABC a b c A B C ∧
    f (A / 2) = Real.sqrt 3 ∧
    a = 4 ∧
    b + c = 5 ∧
    (1/2) * b * c * Real.sin A = 3 * Real.sqrt 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_f_range_and_triangle_area_l3673_367317


namespace NUMINAMATH_CALUDE_polynomial_intersection_l3673_367399

-- Define the polynomials f and g
def f (a b x : ℝ) : ℝ := x^2 + a*x + b
def g (c d x : ℝ) : ℝ := x^2 + c*x + d

-- State the theorem
theorem polynomial_intersection (a b c d : ℝ) : 
  -- f and g are distinct polynomials
  (∃ x, f a b x ≠ g c d x) →
  -- The x-coordinate of the vertex of f is a root of g
  g c d (-a/2) = 0 →
  -- The x-coordinate of the vertex of g is a root of f
  f a b (-c/2) = 0 →
  -- Both f and g have the same minimum value
  (f a b (-a/2) = g c d (-c/2)) →
  -- The graphs of f and g intersect at the point (150, -150)
  (f a b 150 = -150 ∧ g c d 150 = -150) →
  -- Conclusion: a + c = -600
  a + c = -600 :=
by sorry

end NUMINAMATH_CALUDE_polynomial_intersection_l3673_367399


namespace NUMINAMATH_CALUDE_houses_built_l3673_367329

theorem houses_built (original : ℕ) (current : ℕ) (built : ℕ) : 
  original = 20817 → current = 118558 → built = current - original → built = 97741 := by
  sorry

end NUMINAMATH_CALUDE_houses_built_l3673_367329


namespace NUMINAMATH_CALUDE_number_division_problem_l3673_367305

theorem number_division_problem (N : ℕ) (D : ℕ) (h1 : N % D = 0) (h2 : N / D = 2) (h3 : N % 4 = 2) : D = 3 := by
  sorry

end NUMINAMATH_CALUDE_number_division_problem_l3673_367305


namespace NUMINAMATH_CALUDE_xiaopang_birthday_is_26th_l3673_367379

/-- Represents a day in May -/
def MayDay := Fin 31

/-- Xiaopang's birthday -/
def xiaopang_birthday : MayDay := sorry

/-- Xiaoya's birthday -/
def xiaoya_birthday : MayDay := sorry

/-- Days of the week, represented as integers mod 7 -/
def DayOfWeek := Fin 7

/-- Function to determine the day of the week for a given day in May -/
def day_of_week (d : MayDay) : DayOfWeek := sorry

/-- Wednesday, represented as a specific day of the week -/
def wednesday : DayOfWeek := sorry

theorem xiaopang_birthday_is_26th :
  -- Both birthdays are in May (implied by their types)
  -- Both birthdays fall on a Wednesday
  day_of_week xiaopang_birthday = wednesday ∧
  day_of_week xiaoya_birthday = wednesday ∧
  -- Xiaopang's birthday is later than Xiaoya's
  xiaopang_birthday.val > xiaoya_birthday.val ∧
  -- The sum of their birth dates is 38
  xiaopang_birthday.val + xiaoya_birthday.val = 38 →
  -- Conclusion: Xiaopang's birthday is on the 26th
  xiaopang_birthday.val = 26 := by
  sorry

end NUMINAMATH_CALUDE_xiaopang_birthday_is_26th_l3673_367379


namespace NUMINAMATH_CALUDE_eggs_per_box_l3673_367342

theorem eggs_per_box (total_eggs : ℕ) (num_boxes : ℕ) (h1 : total_eggs = 15) (h2 : num_boxes = 5) :
  total_eggs / num_boxes = 3 := by
  sorry

end NUMINAMATH_CALUDE_eggs_per_box_l3673_367342


namespace NUMINAMATH_CALUDE_exam_average_problem_l3673_367337

theorem exam_average_problem (n : ℕ) : 
  (15 : ℝ) * 75 + (10 : ℝ) * 90 = (n : ℝ) * 81 → n = 25 :=
by
  sorry

end NUMINAMATH_CALUDE_exam_average_problem_l3673_367337


namespace NUMINAMATH_CALUDE_expression_value_l3673_367304

theorem expression_value : (35 + 12)^2 - (12^2 + 35^2 - 2 * 12 * 35) = 1680 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l3673_367304


namespace NUMINAMATH_CALUDE_ginger_water_usage_l3673_367312

def water_usage (hours_worked : ℕ) (cups_per_bottle : ℕ) (bottles_for_plants : ℕ) : ℕ :=
  (hours_worked * cups_per_bottle) + (bottles_for_plants * cups_per_bottle)

theorem ginger_water_usage :
  water_usage 8 2 5 = 26 := by
  sorry

end NUMINAMATH_CALUDE_ginger_water_usage_l3673_367312


namespace NUMINAMATH_CALUDE_equation_solution_l3673_367330

theorem equation_solution (a b : ℝ) (h : (a^2 * b^2) / (a^4 - 2*b^4) = 1) :
  (a^2 - b^2) / (a^2 + b^2) = 1/3 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l3673_367330


namespace NUMINAMATH_CALUDE_root_difference_quadratic_l3673_367344

theorem root_difference_quadratic (x : ℝ) : 
  let eq := fun x : ℝ => x^2 + 42*x + 360 + 49
  let roots := {r : ℝ | eq r = 0}
  let diff := fun (a b : ℝ) => |a - b|
  ∃ (r₁ r₂ : ℝ), r₁ ∈ roots ∧ r₂ ∈ roots ∧ diff r₁ r₂ = 8 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_root_difference_quadratic_l3673_367344


namespace NUMINAMATH_CALUDE_sector_area_theorem_l3673_367340

/-- A sector is a portion of a circle enclosed by two radii and an arc. -/
structure Sector where
  centralAngle : ℝ
  perimeter : ℝ

/-- The area of a sector. -/
def sectorArea (s : Sector) : ℝ := sorry

theorem sector_area_theorem (s : Sector) :
  s.centralAngle = 2 ∧ s.perimeter = 8 → sectorArea s = 4 := by
  sorry

end NUMINAMATH_CALUDE_sector_area_theorem_l3673_367340


namespace NUMINAMATH_CALUDE_special_triangle_property_l3673_367355

/-- Triangle with given side, inscribed circle radius, and excircle radius -/
structure SpecialTriangle where
  -- Side length
  a : ℝ
  -- Inscribed circle radius
  r : ℝ
  -- Excircle radius
  r_b : ℝ
  -- Assumption that all values are positive
  a_pos : 0 < a
  r_pos : 0 < r
  r_b_pos : 0 < r_b

/-- Theorem stating the relationship between side length, semiperimeter, and tangent length -/
theorem special_triangle_property (t : SpecialTriangle) :
  ∃ (p : ℝ) (tangent_length : ℝ),
    -- Semiperimeter is positive
    0 < p ∧
    -- Tangent length is positive and less than semiperimeter
    0 < tangent_length ∧ tangent_length < p ∧
    -- The given side length equals semiperimeter minus tangent length
    t.a = p - tangent_length :=
  sorry

end NUMINAMATH_CALUDE_special_triangle_property_l3673_367355


namespace NUMINAMATH_CALUDE_smallest_undefined_value_l3673_367323

theorem smallest_undefined_value : 
  let f (x : ℝ) := (x - 3) / (9*x^2 - 90*x + 225)
  ∃ (y : ℝ), (∀ (x : ℝ), x < y → f x ≠ 0⁻¹) ∧ f y = 0⁻¹ ∧ y = 5 :=
by sorry

end NUMINAMATH_CALUDE_smallest_undefined_value_l3673_367323


namespace NUMINAMATH_CALUDE_surprise_shop_revenue_l3673_367365

/-- Represents the daily potential revenue of a shop during Christmas holidays -/
def daily_potential_revenue (closed_days_per_year : ℕ) (total_years : ℕ) (total_revenue_loss : ℕ) : ℚ :=
  total_revenue_loss / (closed_days_per_year * total_years)

/-- Theorem stating that the daily potential revenue for the given conditions is 5000 dollars -/
theorem surprise_shop_revenue : 
  daily_potential_revenue 3 6 90000 = 5000 := by
  sorry

end NUMINAMATH_CALUDE_surprise_shop_revenue_l3673_367365


namespace NUMINAMATH_CALUDE_least_five_digit_congruent_to_3_mod_17_l3673_367335

theorem least_five_digit_congruent_to_3_mod_17 :
  ∀ n : ℕ, n ≥ 10000 → n ≡ 3 [ZMOD 17] → n ≥ 10004 :=
by sorry

end NUMINAMATH_CALUDE_least_five_digit_congruent_to_3_mod_17_l3673_367335


namespace NUMINAMATH_CALUDE_sum_inequality_l3673_367315

theorem sum_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a / Real.sqrt (a^2 + 8*b*c)) + (b / Real.sqrt (b^2 + 8*a*c)) + (c / Real.sqrt (c^2 + 8*a*b)) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_inequality_l3673_367315


namespace NUMINAMATH_CALUDE_fraction_thousandths_digit_l3673_367367

def fraction : ℚ := 57 / 5000

/-- The thousandths digit of a rational number is the third digit after the decimal point in its decimal representation. -/
def thousandths_digit (q : ℚ) : ℕ :=
  sorry

theorem fraction_thousandths_digit :
  thousandths_digit fraction = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_thousandths_digit_l3673_367367


namespace NUMINAMATH_CALUDE_equation_describes_parabola_l3673_367348

/-- Represents a conic section type -/
inductive ConicSection
  | Circle
  | Parabola
  | Ellipse
  | Hyperbola
  | None

/-- Determines the type of conic section based on the equation |y-3| = √((x+4)² + y²) -/
def determineConicSection : ConicSection := by sorry

/-- Theorem stating that the equation |y-3| = √((x+4)² + y²) describes a parabola -/
theorem equation_describes_parabola : determineConicSection = ConicSection.Parabola := by sorry

end NUMINAMATH_CALUDE_equation_describes_parabola_l3673_367348


namespace NUMINAMATH_CALUDE_expected_total_rainfall_l3673_367394

/-- Represents the weather conditions for a single day --/
structure WeatherCondition where
  sun_prob : ℝ
  light_rain_prob : ℝ
  heavy_rain_prob : ℝ
  light_rain_amount : ℝ
  heavy_rain_amount : ℝ

/-- Calculates the expected rain amount for a single day --/
def expected_rain_per_day (w : WeatherCondition) : ℝ :=
  w.light_rain_prob * w.light_rain_amount + w.heavy_rain_prob * w.heavy_rain_amount

/-- The number of days in the forecast --/
def forecast_days : ℕ := 6

/-- The weather condition for each day in the forecast --/
def daily_weather : WeatherCondition :=
  { sun_prob := 0.3,
    light_rain_prob := 0.3,
    heavy_rain_prob := 0.4,
    light_rain_amount := 5,
    heavy_rain_amount := 12 }

/-- Theorem: The expected total rainfall over the forecast period is 37.8 inches --/
theorem expected_total_rainfall :
  (forecast_days : ℝ) * expected_rain_per_day daily_weather = 37.8 := by
  sorry

end NUMINAMATH_CALUDE_expected_total_rainfall_l3673_367394


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_number_l3673_367389

def imaginary_part (z : ℂ) : ℝ := z.im

theorem imaginary_part_of_complex_number :
  imaginary_part (1/5 - 2/5 * I) = -2/5 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_number_l3673_367389


namespace NUMINAMATH_CALUDE_blueberry_pancakes_l3673_367306

theorem blueberry_pancakes (total : ℕ) (banana : ℕ) (plain : ℕ)
  (h1 : total = 67)
  (h2 : banana = 24)
  (h3 : plain = 23) :
  total - banana - plain = 20 := by
  sorry

end NUMINAMATH_CALUDE_blueberry_pancakes_l3673_367306


namespace NUMINAMATH_CALUDE_cut_cube_volume_l3673_367375

/-- A polyhedron formed by cutting off the eight corners of a cube -/
structure CutCube where
  /-- The polyhedron has 6 octagonal faces -/
  octagonal_faces : Nat
  /-- The polyhedron has 8 triangular faces -/
  triangular_faces : Nat
  /-- All edges of the polyhedron have length 2 -/
  edge_length : ℝ

/-- The volume of the CutCube -/
def volume (c : CutCube) : ℝ := sorry

/-- Theorem stating the volume of the CutCube -/
theorem cut_cube_volume (c : CutCube) 
  (h1 : c.octagonal_faces = 6)
  (h2 : c.triangular_faces = 8)
  (h3 : c.edge_length = 2) :
  volume c = 56 + 112 * Real.sqrt 2 / 3 := by sorry

end NUMINAMATH_CALUDE_cut_cube_volume_l3673_367375


namespace NUMINAMATH_CALUDE_fraction_equivalence_l3673_367334

theorem fraction_equivalence : 
  ∀ (n : ℚ), (2 + n) / (7 + n) = 3 / 8 → n = 1 := by sorry

end NUMINAMATH_CALUDE_fraction_equivalence_l3673_367334


namespace NUMINAMATH_CALUDE_sequence_formulas_l3673_367382

-- Sequence of all positive even numbers
def evenSequence (n : ℕ+) : ℕ := 2 * n

-- Sequence of all positive odd numbers
def oddSequence (n : ℕ+) : ℕ := 2 * n - 1

-- Sequence 1, 4, 9, 16, ...
def squareSequence (n : ℕ+) : ℕ := n ^ 2

-- Sequence -4, -1, 2, 5, ..., 23
def arithmeticSequence (n : ℕ+) : ℤ := 3 * n - 7

theorem sequence_formulas :
  (∀ n : ℕ+, evenSequence n = 2 * n) ∧
  (∀ n : ℕ+, oddSequence n = 2 * n - 1) ∧
  (∀ n : ℕ+, squareSequence n = n ^ 2) ∧
  (∀ n : ℕ+, arithmeticSequence n = 3 * n - 7) := by
  sorry

end NUMINAMATH_CALUDE_sequence_formulas_l3673_367382


namespace NUMINAMATH_CALUDE_prism_height_l3673_367351

/-- A triangular prism with given dimensions -/
structure TriangularPrism where
  volume : ℝ
  base_side1 : ℝ
  base_side2 : ℝ
  height : ℝ

/-- The volume of a triangular prism is equal to the area of its base times its height -/
axiom volume_formula (p : TriangularPrism) : 
  p.volume = (1/2) * p.base_side1 * p.base_side2 * p.height

/-- Theorem: Given a triangular prism with volume 120 cm³ and base sides 3 cm and 4 cm, 
    its height is 20 cm -/
theorem prism_height (p : TriangularPrism) 
  (h_volume : p.volume = 120)
  (h_base1 : p.base_side1 = 3)
  (h_base2 : p.base_side2 = 4) :
  p.height = 20 := by
  sorry

end NUMINAMATH_CALUDE_prism_height_l3673_367351


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l3673_367357

theorem complex_modulus_problem (z : ℂ) (h : z * (1 - Complex.I) = 2 * Complex.I) : 
  Complex.abs z = Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l3673_367357


namespace NUMINAMATH_CALUDE_delta_quotient_on_curve_l3673_367313

/-- Given a point (1,3) on the curve y = x^2 + 2, and a nearby point (1 + Δx, 3 + Δy) on the same curve,
    prove that Δy / Δx = 2 + Δx. -/
theorem delta_quotient_on_curve (Δx Δy : ℝ) : 
  (3 + Δy = (1 + Δx)^2 + 2) → (Δy / Δx = 2 + Δx) := by
  sorry

end NUMINAMATH_CALUDE_delta_quotient_on_curve_l3673_367313


namespace NUMINAMATH_CALUDE_david_chemistry_marks_l3673_367380

/-- Represents the marks obtained in each subject --/
structure Marks where
  english : ℕ
  mathematics : ℕ
  physics : ℕ
  chemistry : ℕ
  biology : ℕ

/-- Calculates the average of a list of natural numbers --/
def average (list : List ℕ) : ℚ :=
  (list.sum : ℚ) / list.length

/-- Theorem: Given David's marks and average, his Chemistry mark must be 67 --/
theorem david_chemistry_marks (m : Marks) (h1 : m.english = 51) (h2 : m.mathematics = 65)
    (h3 : m.physics = 82) (h4 : m.biology = 85)
    (h5 : average [m.english, m.mathematics, m.physics, m.chemistry, m.biology] = 70) :
    m.chemistry = 67 := by
  sorry

#check david_chemistry_marks

end NUMINAMATH_CALUDE_david_chemistry_marks_l3673_367380


namespace NUMINAMATH_CALUDE_felix_lifting_capacity_l3673_367326

/-- Felix's lifting capacity problem -/
theorem felix_lifting_capacity 
  (felix_lift_ratio : ℝ) 
  (brother_weight_ratio : ℝ) 
  (brother_lift_ratio : ℝ) 
  (brother_lift_weight : ℝ) 
  (h1 : felix_lift_ratio = 1.5)
  (h2 : brother_weight_ratio = 2)
  (h3 : brother_lift_ratio = 3)
  (h4 : brother_lift_weight = 600) :
  felix_lift_ratio * (brother_lift_weight / (brother_lift_ratio * brother_weight_ratio)) = 150 := by
  sorry


end NUMINAMATH_CALUDE_felix_lifting_capacity_l3673_367326


namespace NUMINAMATH_CALUDE_leila_spending_difference_l3673_367386

theorem leila_spending_difference : 
  ∀ (total_money sweater_cost jewelry_cost remaining : ℕ),
  sweater_cost = 40 →
  4 * sweater_cost = total_money →
  remaining = 20 →
  total_money = sweater_cost + jewelry_cost + remaining →
  jewelry_cost - sweater_cost = 60 := by
sorry

end NUMINAMATH_CALUDE_leila_spending_difference_l3673_367386


namespace NUMINAMATH_CALUDE_max_value_of_f_l3673_367359

/-- The function we're analyzing -/
def f (x : ℝ) : ℝ := -4 * x^2 + 8 * x + 3

/-- The domain of x -/
def X : Set ℝ := Set.Ioo 0 3

theorem max_value_of_f :
  ∃ (M : ℝ), M = 7 ∧ ∀ x ∈ X, f x ≤ M ∧ ∃ x₀ ∈ X, f x₀ = M :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_f_l3673_367359


namespace NUMINAMATH_CALUDE_polynomial_real_roots_l3673_367378

def polynomial (x : ℝ) : ℝ := x^9 - 37*x^8 - 2*x^7 + 74*x^6 + x^4 - 37*x^3 - 2*x^2 + 74*x

theorem polynomial_real_roots :
  ∃ (s : Finset ℝ), s.card = 5 ∧ (∀ x : ℝ, polynomial x = 0 ↔ x ∈ s) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_real_roots_l3673_367378


namespace NUMINAMATH_CALUDE_square_sum_of_solution_l3673_367325

theorem square_sum_of_solution (x y : ℝ) 
  (h1 : x * y = 6)
  (h2 : x^2 - y^2 + x + y = 44) : 
  x^2 + y^2 = 109 := by
sorry

end NUMINAMATH_CALUDE_square_sum_of_solution_l3673_367325


namespace NUMINAMATH_CALUDE_vector_relation_l3673_367360

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

variable (P A B C : V)

/-- Given that PA + 2PB + 3PC = 0, prove that AP = (1/3)AB + (1/2)AC -/
theorem vector_relation (h : (A - P) + 2 • (B - P) + 3 • (C - P) = 0) :
  P - A = (1/3) • (B - A) + (1/2) • (C - A) := by sorry

end NUMINAMATH_CALUDE_vector_relation_l3673_367360


namespace NUMINAMATH_CALUDE_given_point_in_fourth_quadrant_l3673_367328

/-- A point in the Cartesian plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of the fourth quadrant -/
def is_in_fourth_quadrant (p : Point) : Prop :=
  p.x > 0 ∧ p.y < 0

/-- The given point -/
def given_point : Point :=
  { x := 1, y := -2 }

/-- Theorem: The given point is in the fourth quadrant -/
theorem given_point_in_fourth_quadrant :
  is_in_fourth_quadrant given_point := by
  sorry

end NUMINAMATH_CALUDE_given_point_in_fourth_quadrant_l3673_367328


namespace NUMINAMATH_CALUDE_truncated_cone_radius_l3673_367343

/-- Given three cones touching each other with base radii 6, 24, and 24,
    and a truncated cone sharing a common generator with each,
    the radius of the smaller base of the truncated cone is 2. -/
theorem truncated_cone_radius (r₁ r₂ r₃ : ℝ) (h₁ : r₁ = 6) (h₂ : r₂ = 24) (h₃ : r₃ = 24) :
  ∃ (r : ℝ), r = 2 ∧ 
  (r = r₂ - 24) ∧ 
  (r = r₃ - 24) ∧
  ((24 + r)^2 = 24^2 + (12 - r)^2) :=
by sorry

end NUMINAMATH_CALUDE_truncated_cone_radius_l3673_367343


namespace NUMINAMATH_CALUDE_contestant_paths_count_l3673_367354

/-- Represents the diamond-shaped grid for the word "CONTESTANT" -/
def ContestantGrid : Type := Unit  -- placeholder for the actual grid structure

/-- Represents a valid path in the grid -/
def ValidPath (grid : ContestantGrid) : Type := Unit  -- placeholder for the actual path structure

/-- The number of valid paths in the grid -/
def numValidPaths (grid : ContestantGrid) : ℕ := sorry

/-- The theorem stating that the number of valid paths is 256 -/
theorem contestant_paths_count (grid : ContestantGrid) : numValidPaths grid = 256 := by
  sorry

end NUMINAMATH_CALUDE_contestant_paths_count_l3673_367354


namespace NUMINAMATH_CALUDE_prob_all_cocaptains_l3673_367368

def team_sizes : List Nat := [4, 6, 7, 9]
def num_teams : Nat := 4
def num_cocaptains : Nat := 3

def prob_select_cocaptains (n : Nat) : Rat :=
  6 / (n * (n - 1) * (n - 2))

theorem prob_all_cocaptains : 
  (1 : Rat) / num_teams * (team_sizes.map prob_select_cocaptains).sum = 143 / 1680 := by
  sorry

end NUMINAMATH_CALUDE_prob_all_cocaptains_l3673_367368


namespace NUMINAMATH_CALUDE_cranberry_juice_unit_cost_l3673_367384

/-- Given a 12-ounce can of cranberry juice selling for 84 cents, 
    prove that the unit cost is 7 cents per ounce. -/
theorem cranberry_juice_unit_cost 
  (can_size : ℕ) 
  (total_cost : ℕ) 
  (h1 : can_size = 12)
  (h2 : total_cost = 84) :
  total_cost / can_size = 7 :=
sorry

end NUMINAMATH_CALUDE_cranberry_juice_unit_cost_l3673_367384


namespace NUMINAMATH_CALUDE_inverse_proportion_order_l3673_367398

/-- Given points A, B, C on the graph of y = 3/x, prove y₂ < y₁ < y₃ -/
theorem inverse_proportion_order (y₁ y₂ y₃ : ℝ) : 
  y₁ = 3 / (-2) → y₂ = 3 / (-1) → y₃ = 3 / 1 → y₂ < y₁ ∧ y₁ < y₃ := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_order_l3673_367398


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l3673_367395

/-- Given a geometric sequence {aₙ} where all terms are positive, 
    with a₁ = 3 and a₁ + a₂ + a₃ = 21, prove that a₃ + a₄ + a₅ = 84 -/
theorem geometric_sequence_sum (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a n > 0) →  -- all terms are positive
  a 1 = 3 →  -- first term is 3
  a 1 + a 2 + a 3 = 21 →  -- sum of first three terms is 21
  (∀ n, a (n + 1) = a n * q) →  -- geometric sequence property
  a 3 + a 4 + a 5 = 84 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l3673_367395


namespace NUMINAMATH_CALUDE_fair_coin_three_tosses_one_head_l3673_367346

/-- A fair coin is a coin with equal probability of landing on either side. -/
def fair_coin (p : ℝ) : Prop := p = 1 / 2

/-- The probability of getting exactly k successes in n trials
    with probability p for each trial. -/
def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (n.choose k : ℝ) * p^k * (1 - p)^(n - k)

/-- Theorem: For a fair coin tossed 3 times, the probability
    of getting exactly 1 head and 2 tails is 3/8. -/
theorem fair_coin_three_tosses_one_head (p : ℝ) (h : fair_coin p) :
  binomial_probability 3 1 p = 3/8 := by
  sorry

end NUMINAMATH_CALUDE_fair_coin_three_tosses_one_head_l3673_367346


namespace NUMINAMATH_CALUDE_fraction_zero_implies_x_negative_two_l3673_367358

theorem fraction_zero_implies_x_negative_two (x : ℝ) :
  (x^2 - 4) / (2*x - 4) = 0 ∧ 2*x - 4 ≠ 0 → x = -2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_zero_implies_x_negative_two_l3673_367358


namespace NUMINAMATH_CALUDE_fraction_simplification_l3673_367388

theorem fraction_simplification 
  (x y z u : ℝ) 
  (h1 : x + z ≠ 0) 
  (h2 : y + u ≠ 0) : 
  (x * y^2 + 2 * y * z^2 + y * z * u + 2 * x * y * z + 2 * x * z * u + y^2 * z + 2 * z^2 * u + x * y * u) / 
  (x * u^2 + y * z^2 + y * z * u + x * u * z + x * y * u + u * z^2 + z * u^2 + x * y * z) = 
  (y + 2 * z) / (u + z) := by sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3673_367388
