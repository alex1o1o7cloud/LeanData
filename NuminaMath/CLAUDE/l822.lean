import Mathlib

namespace negation_cube_greater_square_l822_82203

theorem negation_cube_greater_square :
  (¬ ∀ x : ℕ, x^3 > x^2) ↔ (∃ x : ℕ, x^3 ≤ x^2) := by
  sorry

end negation_cube_greater_square_l822_82203


namespace simplify_cube_root_l822_82202

theorem simplify_cube_root : ∃ (c d : ℕ+), 
  (2^10 * 5^6 : ℝ)^(1/3) = c * (2 : ℝ)^(1/3) ∧ 
  c.val + d.val = 202 := by
  sorry

end simplify_cube_root_l822_82202


namespace pyramid_multiplication_l822_82251

theorem pyramid_multiplication (z x : ℕ) : z = 2 → x = 24 →
  (12 * x = 84 ∧ x * 7 = 168 ∧ 12 * z = x) := by
  sorry

end pyramid_multiplication_l822_82251


namespace property_sale_outcome_l822_82215

/-- Calculates the net outcome for a property seller in a specific scenario --/
theorem property_sale_outcome (initial_value : ℝ) (profit_rate : ℝ) (loss_rate : ℝ) (fee_rate : ℝ) : 
  initial_value = 20000 ∧ 
  profit_rate = 0.15 ∧ 
  loss_rate = 0.15 ∧ 
  fee_rate = 0.05 → 
  (initial_value * (1 + profit_rate)) - 
  (initial_value * (1 + profit_rate) * (1 - loss_rate) * (1 + fee_rate)) = 2472.5 := by
sorry

end property_sale_outcome_l822_82215


namespace line_intersection_theorem_l822_82223

/-- The line of intersection of two planes --/
def line_of_intersection (a₁ b₁ c₁ d₁ a₂ b₂ c₂ d₂ : ℝ) :=
  {(x, y, z) : ℝ × ℝ × ℝ | a₁ * x + b₁ * y + c₁ * z + d₁ = 0 ∧
                            a₂ * x + b₂ * y + c₂ * z + d₂ = 0}

/-- The system of equations representing a line --/
def line_equation (p q r s t u : ℝ) :=
  {(x, y, z) : ℝ × ℝ × ℝ | x / p + y / q + z / r = 1 ∧
                            x / s + y / t + z / u = 1}

theorem line_intersection_theorem :
  line_of_intersection 2 3 3 (-9) 4 2 1 (-8) =
  line_equation 4.5 3 3 2 4 8 := by sorry

end line_intersection_theorem_l822_82223


namespace combined_sixth_grade_percent_l822_82258

-- Define the schools
structure School where
  name : String
  total_students : ℕ
  sixth_grade_percent : ℚ

-- Define the given data
def pineview : School := ⟨"Pineview", 150, 15/100⟩
def oakridge : School := ⟨"Oakridge", 180, 17/100⟩
def maplewood : School := ⟨"Maplewood", 170, 15/100⟩

def schools : List School := [pineview, oakridge, maplewood]

-- Function to calculate the number of 6th graders in a school
def sixth_graders (s : School) : ℚ :=
  s.total_students * s.sixth_grade_percent

-- Function to calculate the total number of students
def total_students (schools : List School) : ℕ :=
  schools.foldl (fun acc s => acc + s.total_students) 0

-- Function to calculate the total number of 6th graders
def total_sixth_graders (schools : List School) : ℚ :=
  schools.foldl (fun acc s => acc + sixth_graders s) 0

-- Theorem statement
theorem combined_sixth_grade_percent :
  (total_sixth_graders schools) / (total_students schools : ℚ) = 1572 / 10000 := by
  sorry

end combined_sixth_grade_percent_l822_82258


namespace usb_drive_usage_percentage_l822_82298

theorem usb_drive_usage_percentage (total_capacity : ℝ) (available_space : ℝ) 
  (h1 : total_capacity = 16) 
  (h2 : available_space = 8) : 
  (total_capacity - available_space) / total_capacity * 100 = 50 := by
  sorry

end usb_drive_usage_percentage_l822_82298


namespace polynomial_simplification_l822_82206

theorem polynomial_simplification (p : ℝ) :
  (5 * p^4 - 7 * p^2 + 3 * p + 9) + (-3 * p^3 + 2 * p^2 - 4 * p + 6) =
  5 * p^4 - 3 * p^3 - 5 * p^2 - p + 15 := by
  sorry

end polynomial_simplification_l822_82206


namespace coefficient_of_x_fourth_l822_82230

def expression (x : ℝ) : ℝ :=
  4 * (x^4 - 2*x^3 + x^2) + 2 * (3*x^4 + x^3 - 2*x^2 + x) - 6 * (2*x^2 - x^4 + 3*x^3)

theorem coefficient_of_x_fourth (x : ℝ) :
  ∃ (a b c d e : ℝ), expression x = 4*x^4 + a*x^3 + b*x^2 + c*x + d :=
by
  sorry

end coefficient_of_x_fourth_l822_82230


namespace vector_equation_solution_l822_82259

theorem vector_equation_solution (a b : ℝ × ℝ) (m n : ℝ) :
  a = (2, 1) →
  b = (1, -2) →
  m • a + n • b = (5, -5) →
  m - n = -2 := by
  sorry

end vector_equation_solution_l822_82259


namespace impossible_all_defective_l822_82221

theorem impossible_all_defective (total : Nat) (defective : Nat) (selected : Nat)
  (h1 : total = 10)
  (h2 : defective = 2)
  (h3 : selected = 3)
  (h4 : defective ≤ total)
  (h5 : selected ≤ total) :
  Nat.choose defective selected / Nat.choose total selected = 0 :=
sorry

end impossible_all_defective_l822_82221


namespace fraction_simplification_l822_82249

theorem fraction_simplification :
  (1 / 5 + 1 / 3) / (3 / 4 - 1 / 8) = 64 / 75 := by
  sorry

end fraction_simplification_l822_82249


namespace geometric_series_product_l822_82270

theorem geometric_series_product (y : ℝ) : 
  (∑' n : ℕ, (1/3:ℝ)^n) * (∑' n : ℕ, (-1/3:ℝ)^n) = ∑' n : ℕ, (1/y:ℝ)^n → y = 9 := by
  sorry

end geometric_series_product_l822_82270


namespace simplify_and_evaluate_evaluate_specific_case_l822_82213

theorem simplify_and_evaluate (x y : ℝ) :
  (x - y) * (x + y) + y^2 = x^2 :=
sorry

theorem evaluate_specific_case :
  let x : ℝ := 2
  let y : ℝ := 2023
  (x - y) * (x + y) + y^2 = 4 :=
sorry

end simplify_and_evaluate_evaluate_specific_case_l822_82213


namespace spending_on_games_l822_82267

theorem spending_on_games (total : ℚ) (movies burgers ice_cream music games : ℚ) : 
  total = 40 ∧ 
  movies = 1/4 ∧ 
  burgers = 1/8 ∧ 
  ice_cream = 1/5 ∧ 
  music = 1/4 ∧ 
  games = 3/20 ∧ 
  movies + burgers + ice_cream + music + games = 1 →
  total * games = 7 := by
sorry

end spending_on_games_l822_82267


namespace inequality_solution_set_l822_82269

theorem inequality_solution_set (x : ℝ) :
  x ≠ -7 →
  ((x^2 - 49) / (x + 7) < 0) ↔ (x < -7 ∨ (-7 < x ∧ x < 7)) :=
by sorry

end inequality_solution_set_l822_82269


namespace xyz_sum_l822_82294

theorem xyz_sum (x y z : ℕ+) 
  (h1 : x * y + z = y * z + x)
  (h2 : y * z + x = x * z + y)
  (h3 : x * y + z = 47) : 
  x + y + z = 48 := by
sorry

end xyz_sum_l822_82294


namespace quadratic_equation_roots_condition_l822_82225

theorem quadratic_equation_roots_condition (k : ℝ) : 
  (∃ p q : ℝ, 3 * p^2 + 6 * p + k = 0 ∧ 
              3 * q^2 + 6 * q + k = 0 ∧ 
              |p - q| = (1/2) * (p^2 + q^2)) ↔ 
  (k = -16 + 12 * Real.sqrt 2 ∨ k = -16 - 12 * Real.sqrt 2) :=
by sorry

end quadratic_equation_roots_condition_l822_82225


namespace polynomial_factorization_l822_82275

theorem polynomial_factorization (x : ℝ) :
  (x^2 + 5*x + 3) * (x^2 + 9*x + 20) + (x^2 + 7*x - 8) = 
  (x^2 + 7*x + 8) * (x^2 + 7*x + 14) := by
  sorry

end polynomial_factorization_l822_82275


namespace bedroom_set_price_l822_82218

theorem bedroom_set_price (P : ℝ) : 
  (P * 0.85 * 0.9 - 200 = 1330) → P = 2000 := by
  sorry

end bedroom_set_price_l822_82218


namespace donation_ratio_l822_82229

def monthly_income : ℝ := 240
def groceries_expense : ℝ := 20
def remaining_amount : ℝ := 100

def donation : ℝ := monthly_income - groceries_expense - remaining_amount

theorem donation_ratio : donation / monthly_income = 1 / 2 := by
  sorry

end donation_ratio_l822_82229


namespace translation_iff_equal_movements_l822_82252

/-- Represents the movement of a table's legs -/
structure TableMovement where
  leg1 : ℝ
  leg2 : ℝ
  leg3 : ℝ
  leg4 : ℝ

/-- Determines if a table movement represents a translation -/
def isTranslation (m : TableMovement) : Prop :=
  m.leg1 = m.leg2 ∧ m.leg2 = m.leg3 ∧ m.leg3 = m.leg4

/-- Theorem: A table movement is a translation if and only if all leg movements are equal -/
theorem translation_iff_equal_movements (m : TableMovement) :
  isTranslation m ↔ m.leg1 = m.leg2 ∧ m.leg1 = m.leg3 ∧ m.leg1 = m.leg4 := by sorry

end translation_iff_equal_movements_l822_82252


namespace least_possible_value_z_minus_x_l822_82216

theorem least_possible_value_z_minus_x 
  (x y z : ℤ) 
  (h1 : x < y ∧ y < z)
  (h2 : y - x > 11)
  (h3 : Even x)
  (h4 : Odd y ∧ Odd z) :
  ∀ w : ℤ, w = z - x → w ≥ 14 ∧ ∃ (x' y' z' : ℤ), 
    x' < y' ∧ y' < z' ∧ 
    y' - x' > 11 ∧ 
    Even x' ∧ Odd y' ∧ Odd z' ∧ 
    z' - x' = 14 :=
by sorry

end least_possible_value_z_minus_x_l822_82216


namespace inequality_proof_l822_82255

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h : x * y + y * z + z * x = 6) : 
  1 / (2 * Real.sqrt 2 + x^2 * (y + z)) + 
  1 / (2 * Real.sqrt 2 + y^2 * (x + z)) + 
  1 / (2 * Real.sqrt 2 + z^2 * (x + y)) ≤ 
  1 / (x * y * z) := by
  sorry

end inequality_proof_l822_82255


namespace smallest_positive_angle_2014_l822_82232

def same_terminal_side (a b : ℝ) : Prop :=
  ∃ k : ℤ, a = b + k * 360

theorem smallest_positive_angle_2014 :
  ∃! θ : ℝ, 0 ≤ θ ∧ θ < 360 ∧ same_terminal_side θ (-2014) ∧ θ = 146 := by
  sorry

end smallest_positive_angle_2014_l822_82232


namespace min_sum_of_squares_l822_82256

theorem min_sum_of_squares (x y : ℝ) (h : (x + 3) * (y - 3) = 0) :
  ∃ (m : ℝ), m = 18 ∧ ∀ (a b : ℝ), (a + 3) * (b - 3) = 0 → x^2 + y^2 ≥ m :=
by sorry

end min_sum_of_squares_l822_82256


namespace simplify_trig_fraction_l822_82273

theorem simplify_trig_fraction (x : ℝ) :
  (3 + 2 * Real.sin x + 2 * Real.cos x) / (3 + 2 * Real.sin x - 2 * Real.cos x) = 
  3 / 5 + 2 / 5 * Real.cos x := by
  sorry

end simplify_trig_fraction_l822_82273


namespace det_inequality_equiv_l822_82266

/-- Definition of a second-order determinant -/
def secondOrderDet (a b c d : ℝ) : ℝ := a * d - b * c

/-- Theorem stating the equivalence of the determinant inequality and the simplified inequality -/
theorem det_inequality_equiv (x : ℝ) :
  secondOrderDet 2 (3 - x) 1 x > 0 ↔ 3 * x - 3 > 0 := by
  sorry

end det_inequality_equiv_l822_82266


namespace right_triangle_sin_A_l822_82248

theorem right_triangle_sin_A (A B C : Real) :
  -- Right triangle ABC with ∠B = 90°
  0 < A ∧ A < Real.pi / 2 →
  0 < C ∧ C < Real.pi / 2 →
  A + C = Real.pi / 2 →
  -- 3 tan A = 4
  3 * Real.tan A = 4 →
  -- Conclusion: sin A = 4/5
  Real.sin A = 4 / 5 := by
sorry

end right_triangle_sin_A_l822_82248


namespace susie_babysitting_rate_l822_82260

/-- Susie's babysitting scenario -/
theorem susie_babysitting_rate :
  ∀ (rate : ℚ),
  (∀ (day : ℕ), day ≤ 7 → day * (3 * rate) = day * (3 * rate)) →  -- She works 3 hours every day
  (3/10 + 2/5) * (7 * (3 * rate)) + 63 = 7 * (3 * rate) →  -- Spent fractions and remaining money
  rate = 10 := by
sorry

end susie_babysitting_rate_l822_82260


namespace new_person_weight_l822_82240

/-- The weight of the new person in a group, given:
  * The initial number of people in the group
  * The average weight increase when a new person replaces one person
  * The weight of the person being replaced
-/
def weight_of_new_person (initial_count : ℕ) (avg_increase : ℚ) (replaced_weight : ℚ) : ℚ :=
  replaced_weight + initial_count * avg_increase

theorem new_person_weight :
  weight_of_new_person 12 (37/10) 65 = 1094/10 := by
  sorry

end new_person_weight_l822_82240


namespace max_volume_box_l822_82299

/-- The volume function for the open-top box -/
def volume (a x : ℝ) : ℝ := x * (a - 2*x)^2

/-- The theorem stating the maximum volume and optimal cut length -/
theorem max_volume_box (a : ℝ) (h : a > 0) :
  ∃ (x : ℝ), x ∈ Set.Ioo 0 (a/2) ∧
    (∀ y ∈ Set.Ioo 0 (a/2), volume a x ≥ volume a y) ∧
    x = a/6 ∧
    volume a x = 2*a^3/27 :=
sorry

end max_volume_box_l822_82299


namespace matrix_equation_holds_l822_82244

def B : Matrix (Fin 3) (Fin 3) ℤ := !![0, 2, 1; 2, 0, 2; 1, 2, 0]

theorem matrix_equation_holds :
  B^3 + (-8 : ℤ) • B^2 + (-12 : ℤ) • B + (-28 : ℤ) • (1 : Matrix (Fin 3) (Fin 3) ℤ) = 0 := by
  sorry

end matrix_equation_holds_l822_82244


namespace max_vector_sum_value_l822_82276

/-- The maximum value of |OA + OB + OP| given the specified conditions -/
theorem max_vector_sum_value : ∃ (max : ℝ),
  max = 6 ∧
  ∀ (P : ℝ × ℝ),
  (P.1 - 3)^2 + P.2^2 = 1 →
  ‖(1, 0) + (0, 3) + P‖ ≤ max :=
by sorry

end max_vector_sum_value_l822_82276


namespace scientific_notation_equivalence_l822_82253

theorem scientific_notation_equivalence : 
  ∃ (a : ℝ) (n : ℤ), 0.0000002 = a * (10 : ℝ) ^ n ∧ 1 ≤ a ∧ a < 10 ∧ a = 2.0 ∧ n = -7 := by
  sorry

end scientific_notation_equivalence_l822_82253


namespace concatenated_number_divisible_by_1980_l822_82210

def concatenated_number : ℕ := sorry

theorem concatenated_number_divisible_by_1980 : 
  ∃ k : ℕ, concatenated_number = 1980 * k := by sorry

end concatenated_number_divisible_by_1980_l822_82210


namespace quadratic_vertex_l822_82291

/-- The quadratic function f(x) = (x-3)^2 + 1 -/
def f (x : ℝ) : ℝ := (x - 3)^2 + 1

/-- The vertex of the quadratic function f -/
def vertex : ℝ × ℝ := (3, 1)

/-- Theorem: The vertex of the quadratic function f(x) = (x-3)^2 + 1 is at the point (3,1) -/
theorem quadratic_vertex : 
  ∀ x : ℝ, f x ≥ f (vertex.1) ∧ f (vertex.1) = vertex.2 := by
  sorry

end quadratic_vertex_l822_82291


namespace library_crates_l822_82228

theorem library_crates (novels : ℕ) (comics : ℕ) (documentaries : ℕ) (albums : ℕ) 
  (crate_capacity : ℕ) (h1 : novels = 145) (h2 : comics = 271) (h3 : documentaries = 419) 
  (h4 : albums = 209) (h5 : crate_capacity = 9) : 
  ((novels + comics + documentaries + albums) / crate_capacity : ℕ) = 116 := by
  sorry

end library_crates_l822_82228


namespace laurie_has_37_marbles_l822_82209

/-- The number of marbles each person has. -/
structure Marbles where
  dennis : ℕ
  kurt : ℕ
  laurie : ℕ

/-- The conditions of the marble problem. -/
def marble_problem (m : Marbles) : Prop :=
  m.dennis = 70 ∧
  m.kurt = m.dennis - 45 ∧
  m.laurie = m.kurt + 12

/-- Theorem stating that Laurie has 37 marbles under the given conditions. -/
theorem laurie_has_37_marbles (m : Marbles) (h : marble_problem m) : m.laurie = 37 := by
  sorry


end laurie_has_37_marbles_l822_82209


namespace pages_left_to_read_l822_82214

theorem pages_left_to_read (total_pages : ℕ) (percent_read : ℚ) (pages_left : ℕ) : 
  total_pages = 1250 → 
  percent_read = 37/100 → 
  pages_left = total_pages - Int.floor (percent_read * total_pages) → 
  pages_left = 788 := by
sorry

end pages_left_to_read_l822_82214


namespace normal_distribution_probability_l822_82208

/-- A random variable following a normal distribution -/
structure NormalRandomVariable where
  μ : ℝ
  σ : ℝ
  hσ : σ > 0

/-- The probability that a normal random variable is less than or equal to a given value -/
noncomputable def normalCdf (X : NormalRandomVariable) (x : ℝ) : ℝ := sorry

theorem normal_distribution_probability 
  (X : NormalRandomVariable)
  (h1 : X.μ = 3)
  (h2 : normalCdf X 6 = 0.9) :
  normalCdf X 3 - normalCdf X 0 = 0.4 := by
  sorry

end normal_distribution_probability_l822_82208


namespace inequality_proof_l822_82227

theorem inequality_proof (a b : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : b < 1) :
  (((1 - a) ^ (1 / b) ≤ (1 - a) ^ b) ∧
   ((1 + a) ^ a ≤ (1 + b) ^ b) ∧
   ((1 - a) ^ b ≤ (1 - a) ^ (b / 2))) ∧
  ((1 - a) ^ a > (1 - b) ^ b) :=
by sorry

end inequality_proof_l822_82227


namespace log_sum_theorem_l822_82282

theorem log_sum_theorem (a b : ℤ) : 
  a + 1 = b → 
  (a : ℝ) < Real.log 800 / Real.log 2 → 
  (Real.log 800 / Real.log 2 : ℝ) < b → 
  a + b = 19 := by
sorry

end log_sum_theorem_l822_82282


namespace happy_number_iff_multiple_of_eight_l822_82280

/-- A number is "happy" if it is equal to the square difference of two consecutive odd numbers. -/
def is_happy_number (n : ℤ) : Prop :=
  ∃ k : ℤ, n = (2*k + 1)^2 - (2*k - 1)^2

/-- The theorem states that a number is a "happy number" if and only if it is a multiple of 8. -/
theorem happy_number_iff_multiple_of_eight (n : ℤ) :
  is_happy_number n ↔ ∃ m : ℤ, n = 8 * m :=
by sorry

end happy_number_iff_multiple_of_eight_l822_82280


namespace card_ratio_proof_l822_82235

theorem card_ratio_proof :
  let full_deck : ℕ := 52
  let num_partial_decks : ℕ := 3
  let num_full_decks : ℕ := 3
  let discarded_cards : ℕ := 34
  let remaining_cards : ℕ := 200
  let total_cards : ℕ := remaining_cards + discarded_cards
  let partial_deck_cards : ℕ := (total_cards - num_full_decks * full_deck) / num_partial_decks
  ∃ (a b : ℕ), a ≠ 0 ∧ b ≠ 0 ∧ partial_deck_cards * b = full_deck * a ∧ a = 1 ∧ b = 2 :=
by sorry

end card_ratio_proof_l822_82235


namespace cubic_function_not_monotonic_l822_82290

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := -x^3 + a*x^2 - x - 1

def not_monotonic (f : ℝ → ℝ) : Prop :=
  ∃ x y z : ℝ, x < y ∧ y < z ∧ ((f x < f y ∧ f y > f z) ∨ (f x > f y ∧ f y < f z))

theorem cubic_function_not_monotonic (a : ℝ) :
  not_monotonic (f a) → a ∈ Set.Iio (-Real.sqrt 3) ∪ Set.Ioi (Real.sqrt 3) :=
by sorry

end cubic_function_not_monotonic_l822_82290


namespace perpendicular_lines_b_value_l822_82277

-- Define the slopes of the two lines
def slope1 : ℚ := 1/2
def slope2 (b : ℚ) : ℚ := -b/5

-- Define the perpendicularity condition
def perpendicular (m1 m2 : ℚ) : Prop := m1 * m2 = -1

-- Theorem statement
theorem perpendicular_lines_b_value :
  perpendicular slope1 (slope2 b) → b = 10 :=
by
  sorry

end perpendicular_lines_b_value_l822_82277


namespace center_of_symmetry_condition_l822_82288

/-- A point A(a, b) is a center of symmetry for a function f if and only if
    for all x, f(a-x) + f(a+x) = 2b -/
theorem center_of_symmetry_condition (f : ℝ → ℝ) (a b : ℝ) :
  (∀ x y : ℝ, f x = y → f (2*a - x) = 2*b - y) ↔
  (∀ x : ℝ, f (a-x) + f (a+x) = 2*b) :=
sorry

end center_of_symmetry_condition_l822_82288


namespace algebraic_expression_value_l822_82212

theorem algebraic_expression_value (x y : ℝ) 
  (h : Real.sqrt (x - 3) + y^2 - 4*y + 4 = 0) : 
  (x^2 - y^2) / (x*y) * (1 / (x^2 - 2*x*y + y^2)) / (x / (x^2*y - x*y^2)) - 1 = 2/3 := by
  sorry

end algebraic_expression_value_l822_82212


namespace square_side_length_l822_82246

/-- A square with perimeter 32 cm has sides of length 8 cm -/
theorem square_side_length (s : ℝ) (h₁ : s > 0) (h₂ : 4 * s = 32) : s = 8 := by
  sorry

end square_side_length_l822_82246


namespace power_of_two_equality_l822_82263

theorem power_of_two_equality (x : ℤ) : (1 / 8 : ℚ) * (2 : ℚ)^40 = (2 : ℚ)^x → x = 37 := by
  sorry

end power_of_two_equality_l822_82263


namespace parabola_shift_theorem_l822_82207

/-- Represents a parabola in the form y = a(x - h)^2 + k --/
structure Parabola where
  a : ℝ
  h : ℝ
  k : ℝ

/-- Shifts a parabola horizontally and vertically --/
def shift (p : Parabola) (dx dy : ℝ) : Parabola :=
  { a := p.a
    h := p.h - dx
    k := p.k + dy }

theorem parabola_shift_theorem (p : Parabola) :
  p.a = 2 ∧ p.h = 1 ∧ p.k = 3 →
  (shift (shift p 2 0) 0 (-1)) = { a := 2, h := -1, k := 2 } := by
  sorry

end parabola_shift_theorem_l822_82207


namespace a_payment_l822_82204

/-- The amount paid by three people for school supplies -/
structure Payment where
  a : ℕ
  b : ℕ
  c : ℕ

/-- The conditions of the problem -/
def problem_conditions (p : Payment) : Prop :=
  p.a + p.b = 67 ∧ p.b + p.c = 64 ∧ p.a + p.c = 63

/-- The theorem stating that given the conditions, A paid 33 yuan -/
theorem a_payment (p : Payment) (h : problem_conditions p) : p.a = 33 := by
  sorry


end a_payment_l822_82204


namespace yoongi_has_fewer_apples_l822_82296

def jungkook_initial_apples : ℕ := 6
def jungkook_received_apples : ℕ := 3
def yoongi_apples : ℕ := 4

theorem yoongi_has_fewer_apples :
  yoongi_apples < jungkook_initial_apples + jungkook_received_apples :=
by
  sorry

end yoongi_has_fewer_apples_l822_82296


namespace always_positive_l822_82285

theorem always_positive (x : ℝ) : (-x)^2 + 2 > 0 := by
  sorry

end always_positive_l822_82285


namespace geometric_sequence_common_ratio_l822_82237

-- Define a geometric sequence
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_common_ratio 
  (a : ℕ → ℝ) (q : ℝ) 
  (h_geom : geometric_sequence a q)
  (h_pos : q > 0)
  (h_equality : a 3 * a 9 = (a 5)^2) :
  q = 1 := by
sorry

end geometric_sequence_common_ratio_l822_82237


namespace james_total_cost_l822_82238

/-- Calculates the total amount James has to pay for adopting a puppy and a kitten -/
def total_cost (puppy_fee kitten_fee multiple_pet_discount friend1_contribution friend2_contribution sales_tax_rate pet_supplies : ℚ) : ℚ :=
  let total_adoption_fees := puppy_fee + kitten_fee
  let discounted_fees := total_adoption_fees * (1 - multiple_pet_discount)
  let friend_contributions := friend1_contribution * puppy_fee + friend2_contribution * kitten_fee
  let fees_after_contributions := discounted_fees - friend_contributions
  let sales_tax := fees_after_contributions * sales_tax_rate
  fees_after_contributions + sales_tax + pet_supplies

/-- The total cost James has to pay is $354.48 -/
theorem james_total_cost :
  total_cost 200 150 0.1 0.25 0.15 0.07 95 = 354.48 := by
  sorry

end james_total_cost_l822_82238


namespace total_bones_in_pile_l822_82233

def number_of_dogs : ℕ := 5

def bones_first_dog : ℕ := 3

def bones_second_dog (first : ℕ) : ℕ := first - 1

def bones_third_dog (second : ℕ) : ℕ := 2 * second

def bones_fourth_dog : ℕ := 1

def bones_fifth_dog (fourth : ℕ) : ℕ := 2 * fourth

theorem total_bones_in_pile :
  bones_first_dog +
  bones_second_dog bones_first_dog +
  bones_third_dog (bones_second_dog bones_first_dog) +
  bones_fourth_dog +
  bones_fifth_dog bones_fourth_dog = 12 :=
by sorry

end total_bones_in_pile_l822_82233


namespace three_points_collinear_l822_82257

/-- Three points lie on the same line if and only if the slope between any two pairs of points is equal. -/
def collinear (p1 p2 p3 : ℝ × ℝ) : Prop :=
  (p2.2 - p1.2) * (p3.1 - p1.1) = (p3.2 - p1.2) * (p2.1 - p1.1)

theorem three_points_collinear (b : ℝ) :
  collinear (4, -7) (-2*b + 3, 5) (3*b + 4, 3) → b = -5/28 := by
  sorry

#check three_points_collinear

end three_points_collinear_l822_82257


namespace parallel_vectors_imply_ratio_l822_82281

/-- Given vectors a and b are parallel if their components are proportional -/
def are_parallel (a b : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ a.1 = k * b.1 ∧ a.2 = k * b.2

theorem parallel_vectors_imply_ratio (x : ℝ) :
  let a : ℝ × ℝ := (Real.sin x, -1)
  let b : ℝ × ℝ := (Real.cos x, 2)
  are_parallel a b →
  (Real.cos x - Real.sin x) / (Real.cos x + Real.sin x) = 3 :=
by sorry

end parallel_vectors_imply_ratio_l822_82281


namespace problem_solution_l822_82264

theorem problem_solution (x y : ℝ) 
  (h : |9*y + 1 - x| = Real.sqrt (x - 4) * Real.sqrt (4 - x)) : 
  2*x*Real.sqrt (1/x) + Real.sqrt (9*y) - Real.sqrt x / 2 + y*Real.sqrt (1/y) = 3 + 4*Real.sqrt 3 / 3 :=
by sorry

end problem_solution_l822_82264


namespace cos_alpha_value_l822_82265

-- Define the angle α
variable (α : Real)

-- Define the point P
def P : ℝ × ℝ := (4, 3)

-- Define the condition that the terminal side of α passes through P
def terminal_side_passes_through (α : Real) (p : ℝ × ℝ) : Prop :=
  ∃ (t : ℝ), t > 0 ∧ p.1 = t * Real.cos α ∧ p.2 = t * Real.sin α

-- State the theorem
theorem cos_alpha_value (h : terminal_side_passes_through α P) : 
  Real.cos α = 4/5 := by
  sorry

end cos_alpha_value_l822_82265


namespace conic_is_hyperbola_l822_82289

/-- The equation of a conic section -/
def conic_equation (x y : ℝ) : Prop :=
  (x - 7)^2 = 3*(4*y + 2)^2 - 108

/-- A hyperbola is characterized by having coefficients of x^2 and y^2 with opposite signs
    when the equation is in standard form -/
def is_hyperbola (eq : (ℝ → ℝ → Prop)) : Prop :=
  ∃ (a b c d e f : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ a*b < 0 ∧
    ∀ x y, eq x y ↔ a*x^2 + b*y^2 + c*x + d*y + e*x*y + f = 0

/-- The given conic equation represents a hyperbola -/
theorem conic_is_hyperbola : is_hyperbola conic_equation := by
  sorry

end conic_is_hyperbola_l822_82289


namespace amy_balloons_l822_82201

theorem amy_balloons (james_balloons : ℕ) (difference : ℕ) (h1 : james_balloons = 232) (h2 : difference = 131) :
  james_balloons - difference = 101 :=
sorry

end amy_balloons_l822_82201


namespace pentagon_properties_independent_l822_82243

/-- A pentagon is a polygon with 5 sides --/
structure Pentagon where
  sides : Fin 5 → ℝ
  angles : Fin 5 → ℝ

/-- A pentagon is equilateral if all its sides have the same length --/
def Pentagon.isEquilateral (p : Pentagon) : Prop :=
  ∀ i j : Fin 5, p.sides i = p.sides j

/-- A pentagon is equiangular if all its angles are equal --/
def Pentagon.isEquiangular (p : Pentagon) : Prop :=
  ∀ i j : Fin 5, p.angles i = p.angles j

/-- The properties of equal angles and equal sides in a pentagon are independent --/
theorem pentagon_properties_independent :
  (∃ p : Pentagon, p.isEquiangular ∧ ¬p.isEquilateral) ∧
  (∃ q : Pentagon, q.isEquilateral ∧ ¬q.isEquiangular) := by
  sorry

end pentagon_properties_independent_l822_82243


namespace streetlight_combinations_l822_82274

/-- Represents the number of streetlights -/
def total_lights : ℕ := 12

/-- Represents the number of lights that can be turned off -/
def lights_off : ℕ := 3

/-- Represents the number of positions where lights can be turned off -/
def eligible_positions : ℕ := 8

/-- The number of ways to turn off lights under the given conditions -/
def ways_to_turn_off : ℕ := Nat.choose eligible_positions lights_off

theorem streetlight_combinations : ways_to_turn_off = 56 := by
  sorry

end streetlight_combinations_l822_82274


namespace meeting_probability_approx_point_one_l822_82247

/-- Object movement in a 2D plane -/
structure Object where
  x : ℤ
  y : ℤ

/-- Probability of movement in each direction -/
structure MoveProb where
  right : ℝ
  up : ℝ
  left : ℝ
  down : ℝ

/-- Calculate the probability of two objects meeting after n steps -/
def meetingProbability (a : Object) (c : Object) (aProb : MoveProb) (cProb : MoveProb) (n : ℕ) : ℝ :=
  sorry

/-- Theorem: The probability of A and C meeting after 7 steps is approximately 0.10 -/
theorem meeting_probability_approx_point_one :
  let a := Object.mk 0 0
  let c := Object.mk 6 8
  let aProb := MoveProb.mk 0.5 0.5 0 0
  let cProb := MoveProb.mk 0.1 0.1 0.4 0.4
  abs (meetingProbability a c aProb cProb 7 - 0.1) < 0.01 := by
  sorry

end meeting_probability_approx_point_one_l822_82247


namespace granola_profit_l822_82205

/-- Elizabeth's granola business problem -/
theorem granola_profit (ingredient_cost : ℝ) (full_price : ℝ) (discounted_price : ℝ)
  (total_bags : ℕ) (full_price_sales : ℕ) (discounted_sales : ℕ)
  (h1 : ingredient_cost = 3)
  (h2 : full_price = 6)
  (h3 : discounted_price = 4)
  (h4 : total_bags = 20)
  (h5 : full_price_sales = 15)
  (h6 : discounted_sales = 5)
  (h7 : full_price_sales + discounted_sales = total_bags) :
  (full_price_sales : ℝ) * full_price + (discounted_sales : ℝ) * discounted_price -
  (total_bags : ℝ) * ingredient_cost = 50 := by
  sorry

#check granola_profit

end granola_profit_l822_82205


namespace dubblefud_red_balls_l822_82231

/-- The number of red balls in a Dubblefud game selection -/
def num_red_balls (r b g : ℕ) : Prop :=
  (2 ^ r) * (4 ^ b) * (5 ^ g) = 16000 ∧ b = g ∧ r = 0

/-- Theorem stating that the number of red balls is 0 given the conditions -/
theorem dubblefud_red_balls :
  ∃ (r b g : ℕ), num_red_balls r b g :=
sorry

end dubblefud_red_balls_l822_82231


namespace park_visitors_difference_l822_82236

theorem park_visitors_difference (total : ℕ) (bikers : ℕ) (hikers : ℕ) :
  total = 676 →
  bikers = 249 →
  total = bikers + hikers →
  hikers > bikers →
  hikers - bikers = 178 := by
sorry

end park_visitors_difference_l822_82236


namespace symmetric_line_wrt_origin_l822_82262

/-- Given a line with equation y = 2x + 1, its symmetric line with respect to the origin
    has the equation y = 2x - 1. -/
theorem symmetric_line_wrt_origin :
  ∀ (x y : ℝ), y = 2*x + 1 → ∃ (x' y' : ℝ), y' = 2*x' - 1 ∧ x' = -x ∧ y' = -y :=
by sorry

end symmetric_line_wrt_origin_l822_82262


namespace bus_system_daily_passengers_l822_82242

def total_people : ℕ := 109200000
def num_weeks : ℕ := 13
def days_per_week : ℕ := 7

theorem bus_system_daily_passengers : 
  total_people / (num_weeks * days_per_week) = 1200000 := by
  sorry

end bus_system_daily_passengers_l822_82242


namespace expression_evaluation_l822_82293

theorem expression_evaluation :
  let x : ℝ := 2
  let y : ℝ := -3
  let z : ℝ := 1
  x^2 + y^2 - z^2 + 2*x*y + 2*y*z = -6 := by
  sorry

end expression_evaluation_l822_82293


namespace inequality_proof_equality_condition_l822_82220

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hsum : x + y + z = 1) :
  y * z + z * x + x * y ≥ 4 * (y^2 * z^2 + z^2 * x^2 + x^2 * y^2) + 5 * x * y * z :=
by sorry

theorem equality_condition (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hsum : x + y + z = 1) :
  (y * z + z * x + x * y = 4 * (y^2 * z^2 + z^2 * x^2 + x^2 * y^2) + 5 * x * y * z) ↔ 
  (x = 1/3 ∧ y = 1/3 ∧ z = 1/3) :=
by sorry

end inequality_proof_equality_condition_l822_82220


namespace marbles_problem_l822_82200

theorem marbles_problem (total : ℕ) (marc_initial : ℕ) (jon_initial : ℕ) (bag : ℕ) : 
  total = 66 →
  marc_initial = 2 * jon_initial →
  marc_initial + jon_initial = total →
  jon_initial + bag = 3 * marc_initial →
  bag = 110 := by
sorry

end marbles_problem_l822_82200


namespace invalid_external_diagonals_l822_82283

def is_valid_external_diagonals (d1 d2 d3 : ℝ) : Prop :=
  d1 > 0 ∧ d2 > 0 ∧ d3 > 0 ∧
  d1^2 + d2^2 > d3^2 ∧
  d1^2 + d3^2 > d2^2 ∧
  d2^2 + d3^2 > d1^2

theorem invalid_external_diagonals :
  ¬ (is_valid_external_diagonals 5 6 8) :=
by sorry

end invalid_external_diagonals_l822_82283


namespace cube_sum_divisible_by_nine_l822_82211

theorem cube_sum_divisible_by_nine (n : ℕ+) :
  ∃ k : ℤ, n^3 + (n + 1)^3 + (n + 2)^3 = 9 * k := by
  sorry

end cube_sum_divisible_by_nine_l822_82211


namespace max_area_inscribed_triangle_l822_82287

/-- A convex polygon in a 2D plane -/
structure ConvexPolygon where
  vertices : Set (ℝ × ℝ)
  is_convex : Convex ℝ (convexHull ℝ vertices)

/-- A triangle inscribed in a convex polygon -/
structure InscribedTriangle (M : ConvexPolygon) where
  points : Fin 3 → ℝ × ℝ
  inside : ∀ i, points i ∈ convexHull ℝ M.vertices

/-- The area of a triangle given by three points -/
def triangleArea (p₁ p₂ p₃ : ℝ × ℝ) : ℝ := sorry

/-- The theorem statement -/
theorem max_area_inscribed_triangle (M : ConvexPolygon) :
  ∃ (t : InscribedTriangle M), 
    (∀ i, t.points i ∈ M.vertices) ∧
    (∀ (s : InscribedTriangle M), 
      triangleArea (t.points 0) (t.points 1) (t.points 2) ≥ 
      triangleArea (s.points 0) (s.points 1) (s.points 2)) :=
sorry

end max_area_inscribed_triangle_l822_82287


namespace right_triangle_condition_l822_82217

theorem right_triangle_condition (a b : ℝ) (α β : Real) :
  a > 0 → b > 0 →
  a ≠ b →
  (a / b) ^ 2 = (Real.tan α) / (Real.tan β) →
  α + β = Real.pi / 2 := by
  sorry

end right_triangle_condition_l822_82217


namespace arithmetic_geometric_sequence_l822_82272

/-- Given an arithmetic sequence with common difference 2 and where a₁, a₃, and a₄ form a geometric sequence, prove that a₂ = -6 -/
theorem arithmetic_geometric_sequence (a : ℕ → ℚ) :
  (∀ n, a (n + 1) - a n = 2) →  -- arithmetic sequence with common difference 2
  (∃ r, a 3 = r * a 1 ∧ a 4 = r * a 3) →  -- a₁, a₃, a₄ form a geometric sequence
  a 2 = -6 :=
by sorry

end arithmetic_geometric_sequence_l822_82272


namespace five_digit_sum_contains_zero_l822_82279

-- Define a five-digit number type
def FiveDigitNumber := { n : ℕ // n ≥ 10000 ∧ n < 100000 }

-- Define a function to check if a number contains 0
def containsZero (n : FiveDigitNumber) : Prop :=
  ∃ (a b c d : ℕ), n.val = 10000 * a + 1000 * b + 100 * c + 10 * d ∨
                    n.val = 10000 * a + 1000 * b + 100 * c + d ∨
                    n.val = 10000 * a + 1000 * b + 10 * c + d ∨
                    n.val = 10000 * a + 100 * b + 10 * c + d ∨
                    n.val = 1000 * a + 100 * b + 10 * c + d

-- Define a function to check if two numbers differ by switching two digits
def differByTwoDigits (n m : FiveDigitNumber) : Prop :=
  ∃ (a b c d e f : ℕ),
    (n.val = 10000 * a + 1000 * b + 100 * c + 10 * d + e ∧
     m.val = 10000 * a + 1000 * b + 100 * f + 10 * d + e) ∨
    (n.val = 10000 * a + 1000 * b + 100 * c + 10 * d + e ∧
     m.val = 10000 * a + 1000 * f + 100 * c + 10 * d + e) ∨
    (n.val = 10000 * a + 1000 * b + 100 * c + 10 * d + e ∧
     m.val = 10000 * f + 1000 * b + 100 * c + 10 * d + e)

theorem five_digit_sum_contains_zero (n m : FiveDigitNumber)
  (h1 : differByTwoDigits n m)
  (h2 : n.val + m.val = 111111) :
  containsZero n ∨ containsZero m :=
sorry

end five_digit_sum_contains_zero_l822_82279


namespace prob_non_intersecting_chords_l822_82239

/-- The probability of non-intersecting chords when pairing 2n points on a circle -/
theorem prob_non_intersecting_chords (n : ℕ) : 
  ∃ (P : ℚ), P = (2^n : ℚ) / (n + 1).factorial := by
  sorry

end prob_non_intersecting_chords_l822_82239


namespace students_going_to_zoo_l822_82234

theorem students_going_to_zoo (teachers : ℕ) (students_per_group : ℕ) 
  (h1 : teachers = 8) 
  (h2 : students_per_group = 32) : 
  teachers * students_per_group = 256 := by
  sorry

end students_going_to_zoo_l822_82234


namespace a_range_for_increasing_f_l822_82224

-- Define the piecewise function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (2 - a) * x + 1 else a^x

-- State the theorem
theorem a_range_for_increasing_f :
  ∀ a : ℝ, 
  (∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → (f a x₁ - f a x₂) / (x₁ - x₂) > 0) ↔ 
  (a ≥ 3/2 ∧ a < 2) :=
by sorry

end a_range_for_increasing_f_l822_82224


namespace greatest_abcba_divisible_by_11_l822_82271

/-- Represents a five-digit number in the form AB,CBA --/
structure ABCBA where
  a : Nat
  b : Nat
  c : Nat
  value : Nat := a * 10000 + b * 1000 + c * 100 + b * 10 + a

/-- Checks if the digits a, b, and c are valid for our problem --/
def valid_digits (a b c : Nat) : Prop :=
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 
  a > 0 ∧ a < 10 ∧ b < 10 ∧ c < 10

theorem greatest_abcba_divisible_by_11 :
  ∃ (n : ABCBA), 
    valid_digits n.a n.b n.c ∧ 
    n.value % 11 = 0 ∧
    n.value = 96569 ∧
    (∀ (m : ABCBA), valid_digits m.a m.b m.c → m.value % 11 = 0 → m.value ≤ n.value) := by
  sorry

end greatest_abcba_divisible_by_11_l822_82271


namespace original_number_exists_and_unique_l822_82297

theorem original_number_exists_and_unique : ∃! x : ℝ, 3 * (2 * x + 9) = 51 := by
  sorry

end original_number_exists_and_unique_l822_82297


namespace pencils_left_l822_82222

/-- The number of pencils in a dozen -/
def pencils_per_dozen : ℕ := 12

/-- The initial number of dozens of pencils -/
def initial_dozens : ℕ := 3

/-- The number of students -/
def num_students : ℕ := 11

/-- The number of pencils each student takes -/
def pencils_per_student : ℕ := 3

/-- Theorem stating that after students take pencils, 3 pencils are left -/
theorem pencils_left : 
  initial_dozens * pencils_per_dozen - num_students * pencils_per_student = 3 := by
  sorry

end pencils_left_l822_82222


namespace bird_watching_problem_l822_82241

theorem bird_watching_problem (total_watchers : Nat) (average_birds : Nat) 
  (first_watcher_birds : Nat) (second_watcher_birds : Nat) :
  total_watchers = 3 →
  average_birds = 9 →
  first_watcher_birds = 7 →
  second_watcher_birds = 11 →
  (total_watchers * average_birds - first_watcher_birds - second_watcher_birds) = 9 := by
  sorry

end bird_watching_problem_l822_82241


namespace f_monotonicity_and_extremum_l822_82284

noncomputable def f (x : ℝ) : ℝ := x * Real.log x - x

theorem f_monotonicity_and_extremum :
  (∀ x, x > 0 → f x ≤ f (1 : ℝ)) ∧
  (∀ x y, 0 < x ∧ x < 1 ∧ 1 < y → f x > f 1 ∧ f y > f 1) ∧
  f 1 = -1 := by
  sorry

end f_monotonicity_and_extremum_l822_82284


namespace find_other_number_l822_82292

theorem find_other_number (x y : ℤ) : 
  ((x = 19 ∨ y = 19) ∧ 3 * x + 4 * y = 103) → 
  (x = 9 ∨ y = 9) := by
sorry

end find_other_number_l822_82292


namespace divisibility_by_five_l822_82219

theorem divisibility_by_five (a b : ℕ) (h : 5 ∣ (a * b)) : 5 ∣ a ∨ 5 ∣ b := by
  sorry

end divisibility_by_five_l822_82219


namespace village_population_l822_82245

theorem village_population (P : ℝ) : 
  P > 0 →
  (P * 1.05 * 0.95 = 9975) →
  P = 10000 := by
sorry

end village_population_l822_82245


namespace third_grade_trees_l822_82268

theorem third_grade_trees (second_grade_trees : ℕ) (third_grade_trees : ℕ) : 
  second_grade_trees = 15 →
  third_grade_trees < 3 * second_grade_trees →
  third_grade_trees = 42 →
  true :=
by sorry

end third_grade_trees_l822_82268


namespace larry_channels_l822_82250

/-- Calculates the final number of channels Larry has after all changes --/
def final_channels (initial : ℕ) (removed : ℕ) (added : ℕ) (reduced : ℕ) (sports : ℕ) (supreme : ℕ) : ℕ :=
  initial - removed + added - reduced + sports + supreme

/-- Theorem stating that Larry's final number of channels is 147 --/
theorem larry_channels :
  final_channels 150 20 12 10 8 7 = 147 := by
  sorry

end larry_channels_l822_82250


namespace coin_age_possibilities_l822_82278

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def coin_digits : List ℕ := [3, 3, 3, 5, 1, 8]

def valid_first_digit (d : ℕ) : Prop := d ∈ coin_digits ∧ is_prime d

def count_valid_ages : ℕ := 40

theorem coin_age_possibilities :
  (∀ d ∈ coin_digits, d ≥ 0 ∧ d ≤ 9) →
  (∃ d ∈ coin_digits, valid_first_digit d) →
  count_valid_ages = 40 := by
  sorry

end coin_age_possibilities_l822_82278


namespace min_tablets_extracted_l822_82261

theorem min_tablets_extracted (total_A : ℕ) (total_B : ℕ) : 
  total_A = 10 → total_B = 10 → 
  ∃ (min_extracted : ℕ), 
    (∀ (n : ℕ), n < min_extracted → 
      ∃ (a b : ℕ), a + b = n ∧ (a < 2 ∨ b < 2)) ∧
    (∀ (a b : ℕ), a + b = min_extracted → a ≥ 2 ∧ b ≥ 2) ∧
    min_extracted = 12 :=
by sorry

end min_tablets_extracted_l822_82261


namespace intersecting_lines_length_l822_82286

/-- Given a geometric configuration with two intersecting lines AC and BD, prove that AC = 3√19 -/
theorem intersecting_lines_length (O A B C D : ℝ × ℝ) (x : ℝ) : 
  let dist := λ p q : ℝ × ℝ => Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  dist O A = 5 →
  dist O C = 11 →
  dist O D = 5 →
  dist O B = 6 →
  dist B D = 9 →
  x = dist A C →
  x = 3 * Real.sqrt 19 := by
sorry

end intersecting_lines_length_l822_82286


namespace monotone_decreasing_implies_a_range_l822_82254

open Real

/-- The function f(x) = e^x - (a-1)x + 1 is monotonically decreasing on [0,1] -/
def is_monotone_decreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x ∈ (Set.Icc 0 1) → y ∈ (Set.Icc 0 1) → x ≤ y → f x ≥ f y

/-- The main theorem -/
theorem monotone_decreasing_implies_a_range 
  (a : ℝ) 
  (f : ℝ → ℝ) 
  (h : f = fun x ↦ exp x - (a - 1) * x + 1) 
  (h_monotone : is_monotone_decreasing f) : 
  a ∈ Set.Ici (exp 1 + 1) := by
sorry

end monotone_decreasing_implies_a_range_l822_82254


namespace contribution_rate_of_random_error_l822_82295

theorem contribution_rate_of_random_error 
  (sum_squared_residuals : ℝ) 
  (total_sum_squares : ℝ) 
  (h1 : sum_squared_residuals = 325) 
  (h2 : total_sum_squares = 923) : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.001 ∧ 
  abs (sum_squared_residuals / total_sum_squares - 0.352) < ε :=
sorry

end contribution_rate_of_random_error_l822_82295


namespace opposite_reciprocal_sum_l822_82226

theorem opposite_reciprocal_sum (a b c : ℝ) 
  (h1 : a + b = 0)  -- a and b are opposite numbers
  (h2 : c = 1/4)    -- the reciprocal of c is 4
  : 3*a + 3*b - 4*c = -1 := by
  sorry

end opposite_reciprocal_sum_l822_82226
