import Mathlib

namespace ellipse_center_x_coordinate_l355_35591

/-- An ellipse in the first quadrant tangent to both axes with foci at (3,4) and (3,12) -/
structure Ellipse where
  /-- The ellipse is in the first quadrant -/
  first_quadrant : True
  /-- The ellipse is tangent to both x-axis and y-axis -/
  tangent_to_axes : True
  /-- One focus is at (3,4) -/
  focus1 : ℝ × ℝ := (3, 4)
  /-- The other focus is at (3,12) -/
  focus2 : ℝ × ℝ := (3, 12)

/-- The x-coordinate of the center of the ellipse is 3 -/
theorem ellipse_center_x_coordinate (e : Ellipse) : ∃ (y : ℝ), e.focus1.1 = 3 ∧ e.focus2.1 = 3 := by
  sorry

end ellipse_center_x_coordinate_l355_35591


namespace investment_rate_problem_l355_35566

def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time

theorem investment_rate_problem (r : ℝ) : 
  simple_interest 900 0.045 7 = simple_interest 900 (r / 100) 7 + 31.50 →
  r = 4 := by
  sorry

end investment_rate_problem_l355_35566


namespace candy_distribution_l355_35509

theorem candy_distribution (initial_candies : ℕ) (additional_candies : ℕ) (friends : ℕ) 
  (h1 : initial_candies = 20)
  (h2 : additional_candies = 4)
  (h3 : friends = 6)
  (h4 : friends > 0) :
  (initial_candies + additional_candies) / friends = 4 := by
  sorry

end candy_distribution_l355_35509


namespace opposite_hands_at_343_l355_35539

/-- Represents the number of degrees the minute hand moves per minute -/
def minute_hand_speed : ℝ := 6

/-- Represents the number of degrees the hour hand moves per minute -/
def hour_hand_speed : ℝ := 0.5

/-- Represents the number of minutes past 3:00 -/
def minutes_past_three : ℝ := 43

/-- The position of the minute hand after 5 minutes -/
def minute_hand_position (t : ℝ) : ℝ :=
  minute_hand_speed * (t + 5)

/-- The position of the hour hand 4 minutes ago -/
def hour_hand_position (t : ℝ) : ℝ :=
  90 + hour_hand_speed * (t - 4)

/-- Two angles are opposite if their absolute difference is 180 degrees -/
def are_opposite (a b : ℝ) : Prop :=
  abs (a - b) = 180

theorem opposite_hands_at_343 :
  are_opposite 
    (minute_hand_position minutes_past_three) 
    (hour_hand_position minutes_past_three) := by
  sorry

end opposite_hands_at_343_l355_35539


namespace count_special_sequences_l355_35552

def sequence_length : ℕ := 15

-- Define a function that counts sequences with all ones consecutive
def count_all_ones_consecutive (n : ℕ) : ℕ :=
  (n + 1) * (n + 2) / 2 - 1

-- Define a function that counts sequences with all zeros consecutive
def count_all_zeros_consecutive (n : ℕ) : ℕ :=
  count_all_ones_consecutive n

-- Define a function that counts sequences with both all zeros and all ones consecutive
def count_both_consecutive : ℕ := 2

-- Theorem statement
theorem count_special_sequences :
  count_all_ones_consecutive sequence_length +
  count_all_zeros_consecutive sequence_length -
  count_both_consecutive = 268 := by
  sorry

end count_special_sequences_l355_35552


namespace midpoint_distance_to_y_axis_l355_35580

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the line passing through the focus
def line_through_focus (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  ∃ (t : ℝ), x₁ = t ∧ x₂ = t ∧ y₁^2 = 4*x₁ ∧ y₂^2 = 4*x₂

-- Theorem statement
theorem midpoint_distance_to_y_axis 
  (x₁ y₁ x₂ y₂ : ℝ) 
  (h_parabola₁ : parabola x₁ y₁) 
  (h_parabola₂ : parabola x₂ y₂)
  (h_line : line_through_focus x₁ y₁ x₂ y₂)
  (h_sum : x₁ + x₂ = 3) :
  (x₁ + x₂) / 2 = 3 / 2 :=
sorry

end midpoint_distance_to_y_axis_l355_35580


namespace multiplication_addition_equality_l355_35560

theorem multiplication_addition_equality : 25 * 13 * 2 + 15 * 13 * 7 = 2015 := by
  sorry

end multiplication_addition_equality_l355_35560


namespace fraction_equality_l355_35574

theorem fraction_equality (m n r t : ℚ) 
  (h1 : m / n = 5 / 2) 
  (h2 : r / t = 7 / 8) : 
  (3 * m * r - n * t) / (4 * n * t - 7 * m * r) = -89 / 181 := by
  sorry

end fraction_equality_l355_35574


namespace milk_replacement_l355_35519

theorem milk_replacement (x : ℝ) : 
  x > 0 ∧ x < 30 →
  30 - x - (x - x^2/30) = 14.7 →
  x = 9 := by
sorry

end milk_replacement_l355_35519


namespace negative_two_power_sum_l355_35521

theorem negative_two_power_sum : (-2)^2004 + (-2)^2005 = -2^2004 := by sorry

end negative_two_power_sum_l355_35521


namespace special_pair_sum_l355_35568

theorem special_pair_sum (a b : ℕ+) (q r : ℕ) : 
  a^2 + b^2 = q * (a + b) + r →
  0 ≤ r →
  r < a + b →
  q^2 + r = 1977 →
  ((a = 50 ∧ b = 37) ∨ (a = 37 ∧ b = 50)) :=
sorry

end special_pair_sum_l355_35568


namespace first_row_desks_l355_35575

/-- Calculates the number of desks in the first row given the total number of rows,
    the increase in desks per row, and the total number of students that can be seated. -/
def desks_in_first_row (total_rows : ℕ) (increase_per_row : ℕ) (total_students : ℕ) : ℕ :=
  (2 * total_students - total_rows * (total_rows - 1) * increase_per_row) / (2 * total_rows)

/-- Theorem stating that given 8 rows of desks, where each subsequent row has 2 more desks
    than the previous row, and a total of 136 students can be seated, the number of desks
    in the first row is 10. -/
theorem first_row_desks :
  desks_in_first_row 8 2 136 = 10 := by
  sorry

end first_row_desks_l355_35575


namespace total_cookies_count_l355_35565

/-- Represents a pack of cookies -/
structure CookiePack where
  name : String
  cookies : Nat

/-- Represents a person's cookie purchase -/
structure Purchase where
  packs : List (CookiePack × Nat)

def packA : CookiePack := ⟨"A", 15⟩
def packB : CookiePack := ⟨"B", 30⟩
def packC : CookiePack := ⟨"C", 45⟩
def packD : CookiePack := ⟨"D", 60⟩

def paulPurchase : Purchase := ⟨[(packB, 2), (packA, 1)]⟩
def paulaPurchase : Purchase := ⟨[(packA, 1), (packC, 1)]⟩

def countCookies (purchase : Purchase) : Nat :=
  purchase.packs.foldl (fun acc (pack, quantity) => acc + pack.cookies * quantity) 0

theorem total_cookies_count :
  countCookies paulPurchase + countCookies paulaPurchase = 135 := by
  sorry

#eval countCookies paulPurchase + countCookies paulaPurchase

end total_cookies_count_l355_35565


namespace bridge_length_calculation_l355_35589

-- Define the given parameters
def train_length : ℝ := 140
def train_speed_kmh : ℝ := 45
def crossing_time : ℝ := 30

-- Define the theorem
theorem bridge_length_calculation :
  let train_speed_ms : ℝ := train_speed_kmh * 1000 / 3600
  let total_distance : ℝ := train_speed_ms * crossing_time
  let bridge_length : ℝ := total_distance - train_length
  bridge_length = 235 := by sorry

end bridge_length_calculation_l355_35589


namespace janous_inequality_l355_35569

theorem janous_inequality (α x y z : ℝ) (hα : α > 0) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h_sum : x * y + y * z + z * x = α) :
  (1 + α / x^2) * (1 + α / y^2) * (1 + α / z^2) ≥ 16 * (x / z + z / x + 2) ∧
  ((1 + α / x^2) * (1 + α / y^2) * (1 + α / z^2) = 16 * (x / z + z / x + 2) ↔
   x = y ∧ y = z ∧ z = Real.sqrt (α / 3)) :=
by sorry

end janous_inequality_l355_35569


namespace abs_sum_eq_sum_abs_iff_product_pos_l355_35514

theorem abs_sum_eq_sum_abs_iff_product_pos (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  |a + b| = |a| + |b| ↔ a * b > 0 := by sorry

end abs_sum_eq_sum_abs_iff_product_pos_l355_35514


namespace trigonometric_inequalities_l355_35500

theorem trigonometric_inequalities (x y z : ℝ) 
  (h_pos : x > 0 ∧ y > 0 ∧ z > 0) (h_sum : x + y + z = 1) :
  (Real.sqrt (y * z / x) + Real.sqrt (z * x / y) + Real.sqrt (x * y / z) ≥ Real.sqrt 3) ∧
  (Real.sqrt (x * y / (z + x * y)) + Real.sqrt (y * z / (x + y * z)) + Real.sqrt (z * x / (y + z * x)) ≤ 3 / 2) ∧
  (x / (x + y * z) + y / (y + z * x) + z / (z + x * y) ≤ 9 / 4) ∧
  ((x - y * z) / (x + y * z) + (y - z * x) / (y + z * x) + (z - x * y) / (z + x * y) ≤ 3 / 2) ∧
  ((x - y * z) / (x + y * z) * (y - z * x) / (y + z * x) * (z - x * y) / (z + x * y) ≤ 1 / 8) :=
by sorry

end trigonometric_inequalities_l355_35500


namespace product_evaluation_l355_35567

theorem product_evaluation : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 := by
  sorry

end product_evaluation_l355_35567


namespace cyclic_symmetric_count_l355_35503

-- Definition of cyclic symmetric expression
def is_cyclic_symmetric (σ : ℝ → ℝ → ℝ → ℝ) : Prop :=
  ∀ a b c, σ a b c = σ b c a ∧ σ a b c = σ c a b

-- Define the three expressions
def σ₁ (a b c : ℝ) : ℝ := a * b * c
def σ₂ (a b c : ℝ) : ℝ := a^2 - b^2 + c^2
noncomputable def σ₃ (A B C : ℝ) : ℝ := Real.cos C * Real.cos (A - B) - (Real.cos C)^2

-- Theorem statement
theorem cyclic_symmetric_count :
  (is_cyclic_symmetric σ₁) ∧
  ¬(is_cyclic_symmetric σ₂) ∧
  (is_cyclic_symmetric σ₃) :=
by sorry

end cyclic_symmetric_count_l355_35503


namespace expansion_properties_l355_35557

theorem expansion_properties :
  let f := fun x => (1 - 2*x)^6
  ∃ (c : ℚ) (p : Polynomial ℚ), 
    (f = fun x => p.eval x) ∧ 
    (p.coeff 2 = 60) ∧ 
    (p.eval 1 = 1) := by
  sorry

end expansion_properties_l355_35557


namespace fruit_drink_total_amount_l355_35561

/-- Represents the composition of a fruit drink -/
structure FruitDrink where
  orange_percent : Real
  watermelon_percent : Real
  grape_percent : Real
  pineapple_percent : Real
  grape_ounces : Real
  total_ounces : Real

/-- The theorem stating the total amount of the drink given its composition -/
theorem fruit_drink_total_amount (drink : FruitDrink) 
  (h1 : drink.orange_percent = 0.1)
  (h2 : drink.watermelon_percent = 0.55)
  (h3 : drink.grape_percent = 0.2)
  (h4 : drink.pineapple_percent = 1 - (drink.orange_percent + drink.watermelon_percent + drink.grape_percent))
  (h5 : drink.grape_ounces = 40)
  (h6 : drink.total_ounces * drink.grape_percent = drink.grape_ounces) :
  drink.total_ounces = 200 := by
  sorry

end fruit_drink_total_amount_l355_35561


namespace cubic_polynomial_root_l355_35502

theorem cubic_polynomial_root (b c : ℚ) :
  (∃ (x : ℝ), x^3 + b*x + c = 0 ∧ x = 5 - 2*Real.sqrt 2) →
  (∃ (y : ℤ), y^3 + b*y + c = 0 ∧ y = -10) :=
by sorry

end cubic_polynomial_root_l355_35502


namespace last_digit_of_2_pow_2010_l355_35581

-- Define the function that gives the last digit of 2^n
def lastDigitOf2Pow (n : ℕ) : ℕ :=
  match n % 4 with
  | 0 => 6
  | 1 => 2
  | 2 => 4
  | 3 => 8
  | _ => unreachable!

-- Theorem statement
theorem last_digit_of_2_pow_2010 : lastDigitOf2Pow 2010 = 4 := by
  sorry

end last_digit_of_2_pow_2010_l355_35581


namespace infinite_triples_with_coprime_c_l355_35506

theorem infinite_triples_with_coprime_c : ∃ (a b c : ℕ → ℕ+), 
  (∀ n, (a n)^2 + (b n)^2 = (c n)^4) ∧ 
  (∀ n, Nat.gcd (c n) (c (n + 1)) = 1) := by
  sorry

end infinite_triples_with_coprime_c_l355_35506


namespace smallest_sum_B_d_l355_35505

theorem smallest_sum_B_d : 
  ∃ (B d : ℕ), 
    B < 5 ∧ 
    d > 6 ∧ 
    125 * B + 25 * B + B = 4 * d + 4 ∧
    (∀ (B' d' : ℕ), 
      B' < 5 → 
      d' > 6 → 
      125 * B' + 25 * B' + B' = 4 * d' + 4 → 
      B + d ≤ B' + d') ∧
    B + d = 77 := by
  sorry

end smallest_sum_B_d_l355_35505


namespace sum_of_digits_9ab_l355_35563

/-- The sum of digits of a number in base 10 -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- A number consisting of n repetitions of a digit d in base 10 -/
def repeatDigit (d : ℕ) (n : ℕ) : ℕ := sorry

theorem sum_of_digits_9ab : 
  let a := repeatDigit 8 2000
  let b := repeatDigit 5 2000
  sumOfDigits (9 * a * b) = 18005 := by sorry

end sum_of_digits_9ab_l355_35563


namespace unique_number_l355_35524

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def is_odd (n : ℕ) : Prop := ∃ k : ℕ, n = 2*k + 1

def is_multiple_of_13 (n : ℕ) : Prop := ∃ k : ℕ, n = 13*k

def digit_product (n : ℕ) : ℕ := (n / 10) * (n % 10)

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, n = k^2

theorem unique_number : ∃! n : ℕ, 
  is_two_digit n ∧ 
  is_odd n ∧ 
  is_multiple_of_13 n ∧ 
  is_perfect_square (digit_product n) ∧
  n = 91 := by sorry

end unique_number_l355_35524


namespace unique_a_value_l355_35520

/-- A quadratic function of the form y = 3x^2 + 2(a-1)x + b -/
def quadratic_function (a b : ℝ) (x : ℝ) : ℝ := 3 * x^2 + 2 * (a - 1) * x + b

/-- The derivative of the quadratic function -/
def quadratic_derivative (a : ℝ) (x : ℝ) : ℝ := 6 * x + 2 * (a - 1)

theorem unique_a_value (a b : ℝ) :
  (∀ x < 1, quadratic_derivative a x < 0) →
  (∀ x ≥ 1, quadratic_derivative a x ≥ 0) →
  a = -2 :=
sorry

end unique_a_value_l355_35520


namespace inverse_square_relation_l355_35526

/-- Given that x varies inversely as the square of y, prove that y = 6 when x = 0.25,
    given that y = 3 when x = 1. -/
theorem inverse_square_relation (x y : ℝ) (k : ℝ) (h1 : x = k / (y ^ 2)) 
    (h2 : 1 = k / (3 ^ 2)) (h3 : 0.25 = k / (y ^ 2)) : y = 6 := by
  sorry

end inverse_square_relation_l355_35526


namespace folded_paper_perimeter_ratio_l355_35596

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Calculates the perimeter of a rectangle -/
def perimeter (r : Rectangle) : ℝ :=
  2 * (r.width + r.height)

theorem folded_paper_perimeter_ratio :
  let original := Rectangle.mk 6 8
  let folded := Rectangle.mk original.width (original.height / 2)
  let small := Rectangle.mk (folded.width / 2) folded.height
  let large := Rectangle.mk folded.width folded.height
  perimeter small / perimeter large = 7 / 10 := by
sorry

end folded_paper_perimeter_ratio_l355_35596


namespace max_third_side_length_l355_35576

theorem max_third_side_length (a b c : ℕ) (ha : a = 7) (hb : b = 12) : 
  (a + b > c ∧ a + c > b ∧ b + c > a) → c ≤ 18 :=
by sorry

end max_third_side_length_l355_35576


namespace square_sum_given_sum_square_and_product_l355_35546

theorem square_sum_given_sum_square_and_product (x y : ℝ) 
  (h1 : (x + y)^2 = 4) 
  (h2 : x * y = -9) : 
  x^2 + y^2 = 22 := by
sorry

end square_sum_given_sum_square_and_product_l355_35546


namespace students_with_glasses_l355_35513

theorem students_with_glasses (total : ℕ) (difference : ℕ) : total = 36 → difference = 24 → 
  ∃ (with_glasses : ℕ) (without_glasses : ℕ), 
    with_glasses + without_glasses = total ∧ 
    with_glasses + difference = without_glasses ∧ 
    with_glasses = 6 := by
  sorry

end students_with_glasses_l355_35513


namespace simplify_fraction_l355_35577

theorem simplify_fraction : (24 : ℚ) / 32 = 3 / 4 := by
  sorry

end simplify_fraction_l355_35577


namespace pants_cost_l355_35584

theorem pants_cost (total_spent shirt_cost tie_cost : ℕ) 
  (h1 : total_spent = 198)
  (h2 : shirt_cost = 43)
  (h3 : tie_cost = 15) : 
  total_spent - (shirt_cost + tie_cost) = 140 := by
  sorry

end pants_cost_l355_35584


namespace min_jumps_to_blue_l355_35545

/-- Represents a 4x4 grid where each cell can be either red or blue -/
def Grid := Fin 4 → Fin 4 → Bool

/-- A position on the 4x4 grid -/
structure Position where
  row : Fin 4
  col : Fin 4

/-- Checks if two positions are adjacent (share a side) -/
def adjacent (p1 p2 : Position) : Bool :=
  (p1.row = p2.row ∧ (p1.col = p2.col + 1 ∨ p1.col + 1 = p2.col)) ∨
  (p1.col = p2.col ∧ (p1.row = p2.row + 1 ∨ p1.row + 1 = p2.row))

/-- The effect of jumping on a position, changing it and adjacent positions to blue -/
def jump (g : Grid) (p : Position) : Grid :=
  fun r c => if (r = p.row ∧ c = p.col) ∨ adjacent p ⟨r, c⟩ then true else g r c

/-- A sequence of jumps -/
def JumpSequence := List Position

/-- Apply a sequence of jumps to a grid -/
def applyJumps (g : Grid) : JumpSequence → Grid
  | [] => g
  | p::ps => applyJumps (jump g p) ps

/-- Check if all squares in the grid are blue -/
def allBlue (g : Grid) : Prop :=
  ∀ r c, g r c = true

/-- The initial all-red grid -/
def initialGrid : Grid :=
  fun _ _ => false

/-- Theorem: There exists a sequence of 4 jumps that turns the entire grid blue -/
theorem min_jumps_to_blue :
  ∃ (js : JumpSequence), js.length = 4 ∧ allBlue (applyJumps initialGrid js) :=
sorry


end min_jumps_to_blue_l355_35545


namespace train_length_train_length_proof_l355_35585

/-- The length of a train given its speed and the time it takes to cross a platform. -/
theorem train_length (platform_length : ℝ) (crossing_time : ℝ) (train_speed_kmph : ℝ) : ℝ :=
  let train_speed_mps := train_speed_kmph * (5 / 18)
  let total_distance := train_speed_mps * crossing_time
  total_distance - platform_length

/-- Proof that the train length is approximately 110 meters. -/
theorem train_length_proof :
  ∃ ε > 0, abs (train_length 165 7.499400047996161 132 - 110) < ε :=
by
  sorry


end train_length_train_length_proof_l355_35585


namespace not_proposition_example_l355_35590

-- Define what a proposition is in this context
def is_proposition (s : String) : Prop :=
  ∀ (interpretation : Type), (∃ (truth_value : Bool), true)

-- State the theorem
theorem not_proposition_example : ¬ (is_proposition "x^2 + 2x - 3 < 0") :=
sorry

end not_proposition_example_l355_35590


namespace missing_bulbs_l355_35534

theorem missing_bulbs (total_fixtures : ℕ) (capacity_per_fixture : ℕ) 
  (fixtures_with_4 : ℕ) (fixtures_with_3 : ℕ) (fixtures_with_1 : ℕ) (fixtures_with_0 : ℕ) :
  total_fixtures = 24 →
  capacity_per_fixture = 4 →
  fixtures_with_1 = 2 * fixtures_with_4 →
  fixtures_with_0 = fixtures_with_3 / 2 →
  fixtures_with_4 + fixtures_with_3 + (total_fixtures - fixtures_with_4 - fixtures_with_3 - fixtures_with_1) + fixtures_with_1 + fixtures_with_0 = total_fixtures →
  4 * fixtures_with_4 + 3 * fixtures_with_3 + 2 * (total_fixtures - fixtures_with_4 - fixtures_with_3 - fixtures_with_1) + fixtures_with_1 = total_fixtures * capacity_per_fixture / 2 →
  total_fixtures * capacity_per_fixture - (4 * fixtures_with_4 + 3 * fixtures_with_3 + 2 * (total_fixtures - fixtures_with_4 - fixtures_with_3 - fixtures_with_1) + fixtures_with_1) = 48 :=
by sorry

end missing_bulbs_l355_35534


namespace max_value_abc_expression_l355_35531

theorem max_value_abc_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a * b * c * (a + b + c)) / ((a + b)^2 * (b + c)^2) ≤ (1 : ℝ) / 4 := by
  sorry

end max_value_abc_expression_l355_35531


namespace debate_club_girls_l355_35583

theorem debate_club_girls (total_members : ℕ) (present_members : ℕ) 
  (h_total : total_members = 22)
  (h_present : present_members = 14)
  (h_attendance : ∃ (boys girls : ℕ), 
    boys + girls = total_members ∧
    boys + (girls / 3) = present_members) :
  ∃ (girls : ℕ), girls = 12 ∧ 
    ∃ (boys : ℕ), boys + girls = total_members ∧
      boys + (girls / 3) = present_members := by
sorry

end debate_club_girls_l355_35583


namespace increasing_function_inequality_l355_35564

/-- A function f is increasing on ℝ -/
def IsIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

/-- Main theorem: If f is increasing and f(2m) > f(-m+9), then m > 3 -/
theorem increasing_function_inequality (f : ℝ → ℝ) (m : ℝ) 
    (h_incr : IsIncreasing f) (h_ineq : f (2 * m) > f (-m + 9)) : 
    m > 3 := by
  sorry

end increasing_function_inequality_l355_35564


namespace min_value_theorem_l355_35535

theorem min_value_theorem (a m n : ℝ) (ha : a > 0) (ha_neq : a ≠ 1) 
  (hm : m > 0) (hn : n > 0) : 
  let f := fun x => a^(x + 3) - 2
  let A := (-3, -1)
  (A.1 / m + A.2 / n = -1) → 
  (∀ k l, k > 0 → l > 0 → k / m + l / n = -1 → 3*m + n ≤ 3*k + l) →
  3*m + n ≥ 16 :=
by sorry

end min_value_theorem_l355_35535


namespace chemistry_physics_difference_l355_35530

/-- Represents the scores of a student in three subjects -/
structure Scores where
  math : ℕ
  physics : ℕ
  chemistry : ℕ

/-- The conditions of the problem -/
def satisfies_conditions (s : Scores) : Prop :=
  s.math + s.physics = 30 ∧
  s.chemistry > s.physics ∧
  (s.math + s.chemistry) / 2 = 25

/-- The theorem to be proved -/
theorem chemistry_physics_difference (s : Scores) :
  satisfies_conditions s → s.chemistry - s.physics = 20 := by
  sorry


end chemistry_physics_difference_l355_35530


namespace decagon_triangle_probability_l355_35541

/-- The number of vertices in a regular decagon -/
def n : ℕ := 10

/-- The total number of possible triangles formed by selecting 3 vertices from a decagon -/
def total_triangles : ℕ := n.choose 3

/-- The number of triangles with exactly one side coinciding with a side of the decagon -/
def triangles_one_side : ℕ := n * (n - 4)

/-- The number of triangles with two sides coinciding with sides of the decagon -/
def triangles_two_sides : ℕ := n

/-- The total number of favorable outcomes (triangles with at least one side coinciding with a side of the decagon) -/
def favorable_outcomes : ℕ := triangles_one_side + triangles_two_sides

/-- The probability of randomly selecting three vertices to form a triangle with at least one side coinciding with a side of the decagon -/
theorem decagon_triangle_probability : 
  (favorable_outcomes : ℚ) / total_triangles = 7 / 12 := by sorry

end decagon_triangle_probability_l355_35541


namespace infinite_sum_equals_three_l355_35544

open BigOperators

theorem infinite_sum_equals_three :
  ∑' k, (5^k) / ((4^k - 3^k) * (4^(k+1) - 3^(k+1))) = 3 := by
  sorry

end infinite_sum_equals_three_l355_35544


namespace line_tangent_to_parabola_l355_35504

theorem line_tangent_to_parabola (k : ℝ) :
  (∀ x y : ℝ, 4*x + 3*y + k = 0 → y^2 = 16*x) →
  (∃! p : ℝ × ℝ, (4*(p.1) + 3*(p.2) + k = 0) ∧ (p.2)^2 = 16*(p.1)) →
  k = 9 := by
sorry

end line_tangent_to_parabola_l355_35504


namespace inequality_solution_l355_35553

theorem inequality_solution (x : ℝ) : 
  (1 / (x - 1) - 3 / (x - 2) + 3 / (x - 3) - 1 / (x - 4) < 1 / 24) ↔ 
  (x < 1 ∨ (2 < x ∧ x < 3) ∨ 4 < x) := by
  sorry

end inequality_solution_l355_35553


namespace track_meet_boys_count_l355_35517

theorem track_meet_boys_count :
  ∀ (total girls boys : ℕ),
  total = 55 →
  total = girls + boys →
  (3 * girls : ℚ) / 5 + (2 * girls : ℚ) / 5 = girls →
  (2 * girls : ℚ) / 5 = 10 →
  boys = 30 :=
by
  sorry

end track_meet_boys_count_l355_35517


namespace min_value_of_fraction_l355_35592

theorem min_value_of_fraction (a b : ℝ) (ha : a < 0) (hb : b < 0) :
  (∀ x y : ℝ, x < 0 → y < 0 → a / (a + 2*b) + b / (a + b) ≥ x / (x + 2*y) + y / (x + y)) →
  a / (a + 2*b) + b / (a + b) = 2 * (Real.sqrt 2 - 1) :=
sorry

end min_value_of_fraction_l355_35592


namespace linear_coefficient_of_given_quadratic_l355_35511

/-- The coefficient of the linear term in a quadratic equation ax^2 + bx + c = 0 is b. -/
def linearCoefficient (a b c : ℝ) : ℝ := b

/-- The quadratic equation x^2 - 2x - 1 = 0 -/
def quadraticEquation (x : ℝ) : Prop := x^2 - 2*x - 1 = 0

theorem linear_coefficient_of_given_quadratic :
  linearCoefficient 1 (-2) (-1) = -2 := by sorry

end linear_coefficient_of_given_quadratic_l355_35511


namespace triangle_right_angle_l355_35536

-- Define a triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- State the theorem
theorem triangle_right_angle (t : Triangle) 
  (h : t.b - t.a * Real.cos t.B = t.a * Real.cos t.C - t.c) : 
  t.A = π / 2 := by
  sorry

end triangle_right_angle_l355_35536


namespace simplify_expression_l355_35599

theorem simplify_expression : 
  (625 : ℝ) ^ (1/4 : ℝ) * (343 : ℝ) ^ (1/3 : ℝ) = 35 := by
  sorry

#check simplify_expression

end simplify_expression_l355_35599


namespace average_first_15_even_numbers_l355_35527

theorem average_first_15_even_numbers : 
  let first_15_even : List ℕ := List.range 15 |>.map (fun n => 2 * (n + 1))
  (first_15_even.sum : ℚ) / 15 = 16 := by
  sorry

end average_first_15_even_numbers_l355_35527


namespace expression_equality_l355_35582

theorem expression_equality : 
  Real.sqrt 4 + |Real.sqrt 3 - 3| + 2 * Real.sin (π / 6) - (π - 2023)^0 = 5 - Real.sqrt 3 := by
  sorry

end expression_equality_l355_35582


namespace range_of_m_for_P_and_not_Q_l355_35556

/-- The function f(x) parameterized by m -/
def f (m : ℝ) (x : ℝ) : ℝ := x^3 + m*x^2 + (m+6)*x + 1

/-- Predicate P: 2 ≤ m ≤ 8 -/
def P (m : ℝ) : Prop := 2 ≤ m ∧ m ≤ 8

/-- Predicate Q: f has both a maximum and a minimum value -/
def Q (m : ℝ) : Prop := ∃ (a b : ℝ), ∀ (x : ℝ), f m a ≤ f m x ∧ f m x ≤ f m b

/-- The range of m for which P ∩ ¬Q is true is [2, 6] -/
theorem range_of_m_for_P_and_not_Q :
  {m : ℝ | P m ∧ ¬Q m} = {m : ℝ | 2 ≤ m ∧ m ≤ 6} := by sorry

end range_of_m_for_P_and_not_Q_l355_35556


namespace complex_modulus_l355_35572

theorem complex_modulus (z : ℂ) (h : z * (2 + Complex.I) = Complex.I ^ 10) :
  Complex.abs z = Real.sqrt 5 / 5 := by
  sorry

end complex_modulus_l355_35572


namespace system_solution_l355_35501

/-- 
Given a system of equations x - y = a and xy = b, 
this theorem proves that the solutions are 
(x, y) = ((a + √(a² + 4b))/2, (-a + √(a² + 4b))/2) and 
(x, y) = ((a - √(a² + 4b))/2, (-a - √(a² + 4b))/2).
-/
theorem system_solution (a b : ℝ) :
  let x₁ := (a + Real.sqrt (a^2 + 4*b)) / 2
  let y₁ := (-a + Real.sqrt (a^2 + 4*b)) / 2
  let x₂ := (a - Real.sqrt (a^2 + 4*b)) / 2
  let y₂ := (-a - Real.sqrt (a^2 + 4*b)) / 2
  (x₁ - y₁ = a ∧ x₁ * y₁ = b) ∧ 
  (x₂ - y₂ = a ∧ x₂ * y₂ = b) := by
  sorry

#check system_solution

end system_solution_l355_35501


namespace integer_ratio_difference_l355_35523

theorem integer_ratio_difference (a b c : ℕ) 
  (pos_a : a > 0) (pos_b : b > 0) (pos_c : c > 0)
  (sum_90 : a + b + c = 90)
  (ratio : 3 * a = 2 * b ∧ 5 * a = 2 * c) :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.001 ∧ |((c : ℝ) - (a : ℝ)) - 12.846| < ε :=
by sorry

end integer_ratio_difference_l355_35523


namespace equation_solutions_l355_35571

theorem equation_solutions :
  (∀ x : ℝ, (x + 1)^2 = 4 ↔ x = 1 ∨ x = -3) ∧
  (∀ x : ℝ, 3 * x^2 - 1 = 2 * x ↔ x = 1 ∨ x = -1/3) := by
  sorry

end equation_solutions_l355_35571


namespace expected_value_of_sum_is_seven_l355_35559

def marbles : Finset Nat := {1, 2, 3, 4, 5, 6}

def pairs : Finset (Nat × Nat) :=
  (marbles.product marbles).filter (fun (a, b) => a < b)

def sum_pair (p : Nat × Nat) : Nat := p.1 + p.2

theorem expected_value_of_sum_is_seven :
  (pairs.sum sum_pair) / pairs.card = 7 := by sorry

end expected_value_of_sum_is_seven_l355_35559


namespace right_triangle_third_side_l355_35516

theorem right_triangle_third_side : ∀ (a b c : ℝ),
  a > 0 → b > 0 → c > 0 →
  (a = 3 ∧ b = 4) ∨ (a = 3 ∧ c = 4) ∨ (b = 3 ∧ c = 4) →
  a^2 + b^2 = c^2 →
  c = 5 ∨ c = Real.sqrt 7 :=
by sorry

end right_triangle_third_side_l355_35516


namespace not_square_sum_of_square_and_divisor_l355_35537

theorem not_square_sum_of_square_and_divisor (A B : ℕ) (hA : A ≠ 0) (hAsq : ∃ n : ℕ, A = n^2) (hB : B ∣ A) :
  ¬ ∃ m : ℕ, A + B = m^2 := by
sorry

end not_square_sum_of_square_and_divisor_l355_35537


namespace hyperbola_foci_l355_35507

/-- The hyperbola equation -/
def hyperbola_equation (x y : ℝ) : Prop :=
  x^2 / 4 - y^2 / 12 = 1

/-- The foci of the hyperbola -/
def foci : Set (ℝ × ℝ) :=
  {(4, 0), (-4, 0)}

/-- Theorem: The foci of the given hyperbola are (4,0) and (-4,0) -/
theorem hyperbola_foci :
  ∀ (p : ℝ × ℝ), p ∈ foci ↔
    (∃ (c : ℝ), ∀ (x y : ℝ),
      hyperbola_equation x y →
      (x - p.1)^2 + y^2 = (x + p.1)^2 + y^2 + 4*c) :=
sorry

end hyperbola_foci_l355_35507


namespace oscar_fish_count_l355_35562

/-- Represents the initial number of Oscar fish in Danny's fish tank. -/
def initial_oscar_fish : ℕ := 58

/-- Theorem stating that the initial number of Oscar fish was 58. -/
theorem oscar_fish_count :
  let initial_guppies : ℕ := 94
  let initial_angelfish : ℕ := 76
  let initial_tiger_sharks : ℕ := 89
  let sold_guppies : ℕ := 30
  let sold_angelfish : ℕ := 48
  let sold_tiger_sharks : ℕ := 17
  let sold_oscar_fish : ℕ := 24
  let remaining_fish : ℕ := 198
  initial_oscar_fish = 
    remaining_fish - 
    ((initial_guppies - sold_guppies) + 
     (initial_angelfish - sold_angelfish) + 
     (initial_tiger_sharks - sold_tiger_sharks)) + 
    sold_oscar_fish :=
by sorry

end oscar_fish_count_l355_35562


namespace equal_distribution_of_treats_l355_35508

theorem equal_distribution_of_treats (cookies cupcakes brownies students : ℕ) 
  (h1 : cookies = 20)
  (h2 : cupcakes = 25)
  (h3 : brownies = 35)
  (h4 : students = 20) :
  (cookies + cupcakes + brownies) / students = 4 :=
by sorry

end equal_distribution_of_treats_l355_35508


namespace tan_fraction_equals_two_l355_35595

theorem tan_fraction_equals_two (α : Real) (h : Real.tan α = 3) :
  (Real.sin (α - Real.pi) + Real.cos (Real.pi - α)) /
  (Real.sin (Real.pi / 2 - α) + Real.cos (Real.pi / 2 + α)) = 2 := by
  sorry

end tan_fraction_equals_two_l355_35595


namespace largest_s_value_l355_35547

theorem largest_s_value (r s : ℕ) : 
  r ≥ s ∧ s ≥ 3 ∧ 
  (r - 2) * s * 5 = (s - 2) * r * 4 →
  s ≤ 130 ∧ ∃ (r' : ℕ), r' ≥ 130 ∧ (r' - 2) * 130 * 5 = (130 - 2) * r' * 4 :=
by sorry

end largest_s_value_l355_35547


namespace fishes_from_superior_is_44_l355_35588

/-- The number of fishes taken from Lake Superior -/
def fishes_from_superior (total : ℕ) (ontario_erie : ℕ) (huron_michigan : ℕ) : ℕ :=
  total - ontario_erie - huron_michigan

/-- Theorem: Given the conditions from the problem, prove that the number of fishes
    taken from Lake Superior is 44 -/
theorem fishes_from_superior_is_44 :
  fishes_from_superior 97 23 30 = 44 := by
  sorry

end fishes_from_superior_is_44_l355_35588


namespace calories_in_pound_of_fat_l355_35555

/-- Represents the number of calories in a pound of body fat -/
def calories_per_pound : ℝ := 3500

/-- Represents the number of calories burned per day through light exercise -/
def calories_burned_per_day : ℝ := 2500

/-- Represents the number of calories consumed per day -/
def calories_consumed_per_day : ℝ := 2000

/-- Represents the number of days it takes to lose the weight -/
def days_to_lose_weight : ℝ := 35

/-- Represents the number of pounds lost -/
def pounds_lost : ℝ := 5

theorem calories_in_pound_of_fat :
  calories_per_pound = 
    ((calories_burned_per_day - calories_consumed_per_day) * days_to_lose_weight) / pounds_lost :=
by sorry

end calories_in_pound_of_fat_l355_35555


namespace iphone_savings_l355_35529

/-- Represents the cost of an iPhone X in dollars -/
def iphone_cost : ℝ := 600

/-- Represents the discount percentage for buying multiple smartphones -/
def discount_percentage : ℝ := 5

/-- Represents the number of iPhones being purchased -/
def num_iphones : ℕ := 3

/-- Theorem stating that the savings from buying 3 iPhones X together with a 5% discount,
    compared to buying them individually without a discount, is $90 -/
theorem iphone_savings :
  (num_iphones * iphone_cost) * (discount_percentage / 100) = 90 := by
  sorry

end iphone_savings_l355_35529


namespace negation_of_existence_ln_positive_l355_35554

theorem negation_of_existence_ln_positive :
  (¬ ∃ x : ℝ, x > 0 ∧ Real.log x > 0) ↔ (∀ x : ℝ, x > 0 → Real.log x ≤ 0) := by
  sorry

end negation_of_existence_ln_positive_l355_35554


namespace unique_triple_divisibility_l355_35532

theorem unique_triple_divisibility (a b c : ℕ) : 
  (∃ k : ℕ, (a * b + 1) = k * (2 * c)) ∧
  (∃ m : ℕ, (b * c + 1) = m * (2 * a)) ∧
  (∃ n : ℕ, (c * a + 1) = n * (2 * b)) →
  a = 1 ∧ b = 1 ∧ c = 1 := by
  sorry

end unique_triple_divisibility_l355_35532


namespace max_elevation_is_288_l355_35573

-- Define the elevation function
def s (t : ℝ) : ℝ := 144 * t - 18 * t^2

-- Theorem stating that the maximum elevation is 288
theorem max_elevation_is_288 : 
  ∃ t_max : ℝ, ∀ t : ℝ, s t ≤ s t_max ∧ s t_max = 288 :=
sorry

end max_elevation_is_288_l355_35573


namespace line_intersects_circle_l355_35549

/-- Given a line l: y = k(x + 1/2) and a circle C: x^2 + y^2 = 1,
    prove that the line always intersects the circle for any real k. -/
theorem line_intersects_circle (k : ℝ) : 
  ∃ x y : ℝ, y = k * (x + 1/2) ∧ x^2 + y^2 = 1 := by
  sorry

#check line_intersects_circle

end line_intersects_circle_l355_35549


namespace product_of_elements_is_zero_l355_35512

theorem product_of_elements_is_zero
  (n : ℕ)
  (M : Finset ℝ)
  (h_odd : Odd n)
  (h_gt_one : n > 1)
  (h_card : M.card = n)
  (h_sum_invariant : ∀ x ∈ M, M.sum id = (M.erase x).sum id + (M.sum id - x)) :
  M.prod id = 0 := by
sorry

end product_of_elements_is_zero_l355_35512


namespace no_roots_implies_non_integer_difference_l355_35522

theorem no_roots_implies_non_integer_difference (a b : ℝ) : 
  a ≠ b → 
  (∀ x : ℝ, (x^2 + 20*a*x + 10*b) * (x^2 + 20*b*x + 10*a) ≠ 0) → 
  ¬(∃ n : ℤ, 20*(b - a) = n) :=
by sorry

end no_roots_implies_non_integer_difference_l355_35522


namespace divisors_equidistant_from_third_l355_35587

theorem divisors_equidistant_from_third (n : ℕ) : 
  (∃ (a b : ℕ), a ≠ b ∧ a ∣ n ∧ b ∣ n ∧ 
   (n : ℚ) / 3 - (a : ℚ) = (b : ℚ) - (n : ℚ) / 3) → 
  ∃ (k : ℕ), n = 6 * k :=
sorry

end divisors_equidistant_from_third_l355_35587


namespace f_properties_l355_35558

def f (x : ℝ) := 4 - x^2

theorem f_properties :
  (∀ x : ℝ, f (-x) = f x) ∧
  (∀ x₁ x₂ : ℝ, 0 ≤ x₁ → x₁ < x₂ → f x₁ > f x₂) ∧
  (∀ x : ℝ, f x ≥ 3*x ↔ -4 ≤ x ∧ x ≤ 1) :=
by sorry

end f_properties_l355_35558


namespace reciprocal_and_square_properties_l355_35528

theorem reciprocal_and_square_properties : 
  (∀ x : ℝ, x ≠ 0 → (x = 1/x ↔ x = 1 ∨ x = -1)) ∧ 
  (∀ x : ℝ, x = x^2 ↔ x = 0 ∨ x = 1) := by
  sorry

end reciprocal_and_square_properties_l355_35528


namespace mango_production_l355_35597

/-- Prove that the total produce of mangoes is 400 kg -/
theorem mango_production (apple_production mango_production orange_production : ℕ) 
  (h1 : apple_production = 2 * mango_production)
  (h2 : orange_production = mango_production + 200)
  (h3 : 50 * (apple_production + mango_production + orange_production) = 90000) :
  mango_production = 400 := by
  sorry

end mango_production_l355_35597


namespace transformed_roots_l355_35542

-- Define the polynomial and its roots
def P (b : ℝ) (x : ℝ) : ℝ := x^4 - b*x^2 - 6

-- Define the roots of P
def roots (b : ℝ) : Set ℝ := {x | P b x = 0}

-- Define the transformed equation
def Q (b : ℝ) (y : ℝ) : ℝ := 6*y^2 + b*y + 1

-- Theorem statement
theorem transformed_roots (b : ℝ) (a c d : ℝ) (ha : a ∈ roots b) (hc : c ∈ roots b) (hd : d ∈ roots b) :
  Q b ((a + c) / b^3) = 0 ∧ Q b ((a + b) / c^3) = 0 ∧ Q b ((b + c) / a^3) = 0 ∧ Q b ((a + b + c) / d^3) = 0 := by
  sorry

end transformed_roots_l355_35542


namespace no_prime_roots_for_quadratic_l355_35579

theorem no_prime_roots_for_quadratic : ¬∃ (k : ℤ), ∃ (p q : ℕ), 
  Prime p ∧ Prime q ∧ p ≠ q ∧ p + q = 65 ∧ p * q = k := by
  sorry

end no_prime_roots_for_quadratic_l355_35579


namespace batsman_average_l355_35540

theorem batsman_average (previous_total : ℕ) (previous_average : ℚ) : 
  previous_average * 10 = previous_total →
  (previous_total + 90) / 11 = previous_average + 5 →
  (previous_total + 90) / 11 = 40 := by
sorry

end batsman_average_l355_35540


namespace right_triangle_area_l355_35518

theorem right_triangle_area (a b c : ℝ) (h1 : a^2 + b^2 = c^2) (h2 : c = 13) (h3 : a = 5) :
  (1/2) * a * b = 30 := by
sorry

end right_triangle_area_l355_35518


namespace variance_of_transformed_binomial_l355_35593

/-- A random variable following a binomial distribution with n trials and probability p -/
structure BinomialRandomVariable (n : ℕ) (p : ℝ) where
  X : ℝ

/-- The variance of a binomial random variable -/
def binomialVariance (n : ℕ) (p : ℝ) : ℝ := n * p * (1 - p)

/-- The variance of a linear transformation of a random variable -/
def linearTransformVariance (a : ℝ) (X : ℝ) : ℝ := a^2 * X

theorem variance_of_transformed_binomial :
  let n : ℕ := 10
  let p : ℝ := 0.8
  let X : BinomialRandomVariable n p := ⟨0⟩  -- The actual value doesn't matter for this theorem
  let var_X : ℝ := binomialVariance n p
  let var_2X_plus_1 : ℝ := linearTransformVariance 2 var_X
  var_2X_plus_1 = 6.4 := by sorry

end variance_of_transformed_binomial_l355_35593


namespace smallest_k_with_multiple_sequences_l355_35510

/-- A sequence of positive integers satisfying the given conditions -/
def ValidSequence (a : ℕ → ℕ) : Prop :=
  (∀ n, a n > 0) ∧
  (∀ n, a (n + 1) ≥ a n) ∧
  (∀ n > 2, a n = a (n - 1) + a (n - 2))

/-- The existence of at least two distinct valid sequences with a₉ = k -/
def HasMultipleSequences (k : ℕ) : Prop :=
  ∃ a b : ℕ → ℕ, ValidSequence a ∧ ValidSequence b ∧ a ≠ b ∧ a 9 = k ∧ b 9 = k

/-- 748 is the smallest k for which multiple valid sequences exist -/
theorem smallest_k_with_multiple_sequences :
  HasMultipleSequences 748 ∧ ∀ k < 748, ¬HasMultipleSequences k :=
sorry

end smallest_k_with_multiple_sequences_l355_35510


namespace diamond_jewel_percentage_is_35_percent_l355_35550

/-- Represents the composition of objects in an urn -/
structure UrnComposition where
  bead_percent : ℝ
  ruby_jewel_percent : ℝ
  diamond_jewel_percent : ℝ

/-- Calculates the percentage of diamond jewels in the urn -/
def diamond_jewel_percentage (u : UrnComposition) : ℝ :=
  u.diamond_jewel_percent

/-- The theorem stating the percentage of diamond jewels in the urn -/
theorem diamond_jewel_percentage_is_35_percent (u : UrnComposition) 
  (h1 : u.bead_percent = 30)
  (h2 : u.ruby_jewel_percent = 35)
  (h3 : u.bead_percent + u.ruby_jewel_percent + u.diamond_jewel_percent = 100) :
  diamond_jewel_percentage u = 35 := by
  sorry

#check diamond_jewel_percentage_is_35_percent

end diamond_jewel_percentage_is_35_percent_l355_35550


namespace find_a_value_l355_35586

-- Define an odd function
def OddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- Define the specific function for x < 0
def SpecificFunction (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x, x < 0 → f x = x^2 + a*x

theorem find_a_value (f : ℝ → ℝ) (a : ℝ) 
  (h_odd : OddFunction f)
  (h_spec : SpecificFunction f a)
  (h_f3 : f 3 = 6) :
  a = 5 := by
sorry

end find_a_value_l355_35586


namespace num_factors_of_2000_l355_35538

/-- The number of positive factors of a natural number n -/
def num_factors (n : ℕ) : ℕ := sorry

/-- 2000 expressed as a product of prime factors -/
def two_thousand_factorization : ℕ := 2^4 * 5^3

/-- Theorem stating that the number of positive factors of 2000 is 20 -/
theorem num_factors_of_2000 : num_factors two_thousand_factorization = 20 := by sorry

end num_factors_of_2000_l355_35538


namespace projection_implies_y_value_l355_35533

/-- Given two vectors v and w in R², where v = (2, y) and w = (7, 2),
    if the projection of v onto w is (8, 16/7), then y = 163/7. -/
theorem projection_implies_y_value (y : ℝ) :
  let v : ℝ × ℝ := (2, y)
  let w : ℝ × ℝ := (7, 2)
  let proj_w_v : ℝ × ℝ := ((v.1 * w.1 + v.2 * w.2) / (w.1^2 + w.2^2)) • w
  proj_w_v = (8, 16/7) →
  y = 163/7 := by
sorry

end projection_implies_y_value_l355_35533


namespace arithmetic_sequence_a13_l355_35548

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_a13
  (a : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_a5 : a 5 = 3)
  (h_a9 : a 9 = 6) :
  a 13 = 9 := by
sorry

end arithmetic_sequence_a13_l355_35548


namespace chess_tournament_players_l355_35515

/-- Represents a chess tournament with the given conditions -/
structure ChessTournament where
  n : ℕ  -- number of players
  winner_wins : ℕ  -- number of wins by the winner
  winner_draws : ℕ  -- number of draws by the winner

/-- The conditions of the tournament are satisfied -/
def valid_tournament (t : ChessTournament) : Prop :=
  t.n > 1 ∧  -- more than one player
  t.winner_wins = t.winner_draws ∧  -- winner won half and drew half
  t.winner_wins + t.winner_draws = t.n - 1 ∧  -- winner played against every other player once
  (t.winner_wins : ℚ) + (t.winner_draws : ℚ) / 2 = (t.n * (t.n - 1) : ℚ) / 20  -- winner's points are 9 times less than others'

theorem chess_tournament_players (t : ChessTournament) :
  valid_tournament t → t.n = 15 := by
  sorry

#check chess_tournament_players

end chess_tournament_players_l355_35515


namespace bakery_storage_ratio_l355_35598

/-- Proves that the ratio of flour to baking soda is 10:1 given the conditions in the bakery storage room. -/
theorem bakery_storage_ratio : 
  ∀ (sugar flour baking_soda : ℕ),
  -- Ratio of sugar to flour is 3:8
  8 * sugar = 3 * flour →
  -- There are 900 pounds of sugar
  sugar = 900 →
  -- If 60 more pounds of baking soda were added, the ratio of flour to baking soda would be 8:1
  8 * (baking_soda + 60) = flour →
  -- The ratio of flour to baking soda is 10:1
  10 * baking_soda = flour :=
by
  sorry


end bakery_storage_ratio_l355_35598


namespace sons_present_age_l355_35551

/-- Proves that given the conditions about a father and son's ages, the son's present age is 22 years -/
theorem sons_present_age (son_age father_age : ℕ) : 
  father_age = son_age + 24 →
  father_age + 2 = 2 * (son_age + 2) →
  son_age = 22 := by
  sorry

#check sons_present_age

end sons_present_age_l355_35551


namespace laundry_time_ratio_l355_35543

/-- Proves that the ratio of time to wash towels to time to wash clothes is 2:1 --/
theorem laundry_time_ratio :
  ∀ (towel_time sheet_time clothes_time : ℕ),
    clothes_time = 30 →
    sheet_time = towel_time - 15 →
    towel_time + sheet_time + clothes_time = 135 →
    towel_time / clothes_time = 2 :=
by
  sorry


end laundry_time_ratio_l355_35543


namespace taxes_paid_equals_135_l355_35594

/-- Calculate taxes paid given gross pay and net pay -/
def calculate_taxes (gross_pay : ℝ) (net_pay : ℝ) : ℝ :=
  gross_pay - net_pay

/-- Theorem: Taxes paid are 135 dollars given the conditions -/
theorem taxes_paid_equals_135 :
  let gross_pay : ℝ := 450
  let net_pay : ℝ := 315
  calculate_taxes gross_pay net_pay = 135 := by
  sorry

end taxes_paid_equals_135_l355_35594


namespace child_b_share_after_investment_l355_35525

def total_amount : ℝ := 4500
def ratio_sum : ℕ := 2 + 3 + 4
def child_b_ratio : ℕ := 3
def interest_rate : ℝ := 0.04
def time_period : ℝ := 1

theorem child_b_share_after_investment :
  let principal := (child_b_ratio : ℝ) / ratio_sum * total_amount
  let interest := principal * interest_rate * time_period
  principal + interest = 1560 := by sorry

end child_b_share_after_investment_l355_35525


namespace product_comparison_l355_35578

theorem product_comparison (a b c d : ℝ) (h1 : a ≥ b) (h2 : c ≥ d) :
  (∃ (p : ℕ), p ≥ 3 ∧ (a > 0 ∨ b > 0) ∧ (a > 0 ∨ c > 0) ∧ (a > 0 ∨ d > 0) ∧
               (b > 0 ∨ c > 0) ∧ (b > 0 ∨ d > 0) ∧ (c > 0 ∨ d > 0)) →
    a * c ≥ b * d ∧
  (∃ (n : ℕ), n ≥ 3 ∧ (a < 0 ∨ b < 0) ∧ (a < 0 ∨ c < 0) ∧ (a < 0 ∨ d < 0) ∧
               (b < 0 ∨ c < 0) ∧ (b < 0 ∨ d < 0) ∧ (c < 0 ∨ d < 0)) →
    a * c ≤ b * d ∧
  (((a > 0 ∧ b > 0) ∨ (a > 0 ∧ c > 0) ∨ (a > 0 ∧ d > 0) ∨ (b > 0 ∧ c > 0) ∨
    (b > 0 ∧ d > 0) ∨ (c > 0 ∧ d > 0)) ∧
   ((a < 0 ∧ b < 0) ∨ (a < 0 ∧ c < 0) ∨ (a < 0 ∧ d < 0) ∨ (b < 0 ∧ c < 0) ∨
    (b < 0 ∧ d < 0) ∨ (c < 0 ∧ d < 0))) →
    ¬(∀ x y : ℝ, (x = a * c ∧ y = b * d) → x = y) ∧
    ¬(∀ x y : ℝ, (x = a * c ∧ y = b * d) → x < y) ∧
    ¬(∀ x y : ℝ, (x = a * c ∧ y = b * d) → x > y) :=
by sorry

end product_comparison_l355_35578


namespace impossible_arrangement_l355_35570

theorem impossible_arrangement : ¬ ∃ (A B C : ℕ), 
  (A + B = 45) ∧ 
  (3 * A + B = 6 * C) ∧ 
  (A ≥ 0) ∧ (B ≥ 0) ∧ (C > 0) :=
by sorry

end impossible_arrangement_l355_35570
