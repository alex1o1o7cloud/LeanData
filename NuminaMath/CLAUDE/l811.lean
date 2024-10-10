import Mathlib

namespace sin_75_degrees_l811_81180

theorem sin_75_degrees : Real.sin (75 * π / 180) = (Real.sqrt 6 + Real.sqrt 2) / 4 := by
  sorry

end sin_75_degrees_l811_81180


namespace complex_fraction_simplification_l811_81194

theorem complex_fraction_simplification :
  (5 + 6 * Complex.I) / (2 - 3 * Complex.I) = (-8 : ℚ) / 13 + (27 : ℚ) / 13 * Complex.I :=
by sorry

end complex_fraction_simplification_l811_81194


namespace matrix_multiplication_l811_81156

def A : Matrix (Fin 2) (Fin 2) ℤ := !![4, -2; -1, 5]
def B : Matrix (Fin 2) (Fin 2) ℤ := !![0, 3; 2, -2]

theorem matrix_multiplication :
  A * B = !![(-4), 16; 10, (-13)] := by sorry

end matrix_multiplication_l811_81156


namespace largest_angle_in_special_quadrilateral_l811_81149

/-- A quadrilateral with angles in the ratio 3:4:5:6 has its largest angle equal to 120°. -/
theorem largest_angle_in_special_quadrilateral : 
  ∀ (a b c d : ℝ), 
  a > 0 → b > 0 → c > 0 → d > 0 →
  (a + b + c + d = 360) →
  (b = 4/3 * a) → (c = 5/3 * a) → (d = 2 * a) →
  d = 120 := by
sorry

end largest_angle_in_special_quadrilateral_l811_81149


namespace percentage_relation_l811_81120

theorem percentage_relation (x y z : ℝ) 
  (h1 : x = 0.2 * y) 
  (h2 : x = 0.5 * z) : 
  z = 0.4 * y := by
  sorry

end percentage_relation_l811_81120


namespace flooring_boxes_needed_l811_81128

/-- Calculates the number of flooring boxes needed to complete a room -/
theorem flooring_boxes_needed
  (room_length : ℝ)
  (room_width : ℝ)
  (area_covered : ℝ)
  (area_per_box : ℝ)
  (h1 : room_length = 16)
  (h2 : room_width = 20)
  (h3 : area_covered = 250)
  (h4 : area_per_box = 10)
  : ⌈(room_length * room_width - area_covered) / area_per_box⌉ = 7 := by
  sorry

end flooring_boxes_needed_l811_81128


namespace school_girls_count_l811_81104

theorem school_girls_count (total_pupils : ℕ) (girl_boy_difference : ℕ) :
  total_pupils = 1455 →
  girl_boy_difference = 281 →
  ∃ (boys girls : ℕ),
    boys + girls = total_pupils ∧
    girls = boys + girl_boy_difference ∧
    girls = 868 := by
  sorry

end school_girls_count_l811_81104


namespace texas_tech_game_profit_l811_81178

/-- Represents the discount tiers for t-shirt sales -/
inductive DiscountTier
  | NoDiscount
  | MediumDiscount
  | HighDiscount

/-- Calculates the discount tier based on the number of t-shirts sold -/
def getDiscountTier (numSold : ℕ) : DiscountTier :=
  if numSold ≤ 50 then DiscountTier.NoDiscount
  else if numSold ≤ 100 then DiscountTier.MediumDiscount
  else DiscountTier.HighDiscount

/-- Calculates the profit per t-shirt based on the discount tier -/
def getProfitPerShirt (tier : DiscountTier) (fullPrice : ℕ) : ℕ :=
  match tier with
  | DiscountTier.NoDiscount => fullPrice
  | DiscountTier.MediumDiscount => fullPrice - 5
  | DiscountTier.HighDiscount => fullPrice - 10

/-- Theorem: The money made from selling t-shirts during the Texas Tech game is $1092 -/
theorem texas_tech_game_profit (totalSold arkansasSold fullPrice : ℕ) 
    (h1 : totalSold = 186)
    (h2 : arkansasSold = 172)
    (h3 : fullPrice = 78) :
    let texasTechSold := totalSold - arkansasSold
    let tier := getDiscountTier texasTechSold
    let profitPerShirt := getProfitPerShirt tier fullPrice
    texasTechSold * profitPerShirt = 1092 := by
  sorry

end texas_tech_game_profit_l811_81178


namespace sequence_integer_value_l811_81198

def u (M : ℤ) : ℕ → ℚ
  | 0 => M + 1/2
  | n + 1 => u M n * ⌊u M n⌋

theorem sequence_integer_value (M : ℤ) (h : M ≥ 1) :
  (∃ n : ℕ, ∃ k : ℤ, u M n = k) ↔ M > 1 :=
sorry

end sequence_integer_value_l811_81198


namespace range_of_m_l811_81184

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - x - 2 > 0}
def B : Set ℝ := {x | 3 - |x| ≥ 0}

-- Define set C parameterized by m
def C (m : ℝ) : Set ℝ := {x | (x - m + 1) * (x - 2*m - 1) < 0}

-- State the theorem
theorem range_of_m :
  ∀ m : ℝ, (C m ⊆ B) ↔ (m ∈ Set.Icc (-2) 1) :=
by sorry

end range_of_m_l811_81184


namespace parallel_angles_theorem_l811_81132

/-- Two angles in space with parallel sides --/
structure ParallelAngles where
  α : Real
  β : Real
  sides_parallel : Bool

/-- The theorem stating that if two angles have parallel sides and one is 30°, the other is either 30° or 150° --/
theorem parallel_angles_theorem (angles : ParallelAngles) 
  (h1 : angles.sides_parallel = true) 
  (h2 : angles.α = 30) : 
  angles.β = 30 ∨ angles.β = 150 := by
  sorry

end parallel_angles_theorem_l811_81132


namespace four_square_prod_inequality_l811_81187

theorem four_square_prod_inequality (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  (a^2 + b^2) * (b^2 + c^2) * (c^2 + d^2) * (d^2 + a^2) ≥ 64 * a * b * c * d * |((a - b) * (b - c) * (c - d) * (d - a))| := by
  sorry

end four_square_prod_inequality_l811_81187


namespace inequality_comparison_l811_81189

theorem inequality_comparison : 
  (-0.1 < -0.01) ∧ ¬(-1 > 0) ∧ ¬((1:ℚ)/2 < (1:ℚ)/3) ∧ ¬(-5 > 3) := by
  sorry

end inequality_comparison_l811_81189


namespace smallest_odd_divisible_by_three_l811_81199

theorem smallest_odd_divisible_by_three :
  ∀ n : ℕ, n % 2 = 1 → n % 3 = 0 → n ≥ 3 :=
sorry

end smallest_odd_divisible_by_three_l811_81199


namespace max_pieces_is_seven_l811_81172

def is_five_digit (n : ℕ) : Prop := 10000 ≤ n ∧ n ≤ 99999

def has_distinct_digits (n : ℕ) : Prop :=
  let digits := n.digits 10
  List.length digits = List.length (List.dedup digits)

theorem max_pieces_is_seven :
  (∃ (max : ℕ), 
    (∀ (n : ℕ), ∃ (P Q : ℕ), is_five_digit P ∧ is_five_digit Q ∧ has_distinct_digits P ∧ P = Q * n → n ≤ max) ∧
    (∃ (P Q : ℕ), is_five_digit P ∧ is_five_digit Q ∧ has_distinct_digits P ∧ P = Q * max)) ∧
  (∀ (m : ℕ), 
    (∀ (n : ℕ), ∃ (P Q : ℕ), is_five_digit P ∧ is_five_digit Q ∧ has_distinct_digits P ∧ P = Q * n → n ≤ m) ∧
    (∃ (P Q : ℕ), is_five_digit P ∧ is_five_digit Q ∧ has_distinct_digits P ∧ P = Q * m) → 
    m ≤ 7) :=
by sorry

end max_pieces_is_seven_l811_81172


namespace skewReflectionAndShrinkIsCorrectTransformation_l811_81101

/-- A point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- A rigid transformation in 2D space -/
structure RigidTransformation where
  transform : Point2D → Point2D

/-- Skew-reflection across y=x followed by a vertical shrink by a factor of -1 -/
def skewReflectionAndShrink : RigidTransformation :=
  { transform := λ p => Point2D.mk p.y (-p.x) }

theorem skewReflectionAndShrinkIsCorrectTransformation :
  let C := Point2D.mk 3 (-2)
  let D := Point2D.mk 4 (-3)
  let C' := Point2D.mk 1 2
  let D' := Point2D.mk (-2) 3
  (skewReflectionAndShrink.transform C = C') ∧
  (skewReflectionAndShrink.transform D = D') := by
  sorry


end skewReflectionAndShrinkIsCorrectTransformation_l811_81101


namespace sum_of_monomials_is_monomial_l811_81124

/-- 
Given two monomials 2x^3y^n and -6x^(m+5)y, if their sum is still a monomial,
then m + n = -1.
-/
theorem sum_of_monomials_is_monomial (m n : ℤ) : 
  (∃ (x y : ℝ), ∀ (a b : ℝ), 2 * (x^3) * (y^n) + (-6) * (x^(m+5)) * y = a * (x^b) * y) → 
  m + n = -1 := by
  sorry

end sum_of_monomials_is_monomial_l811_81124


namespace angle_order_l811_81192

-- Define the angles of inclination
variable (α₁ α₂ α₃ : Real)

-- Define the slopes of the lines
def m₁ : Real := 1
def m₂ : Real := -1
def m₃ : Real := -2

-- Define the relationship between angles and slopes
axiom tan_α₁ : Real.tan α₁ = m₁
axiom tan_α₂ : Real.tan α₂ = m₂
axiom tan_α₃ : Real.tan α₃ = m₃

-- Theorem to prove
theorem angle_order : α₁ < α₃ ∧ α₃ < α₂ := by
  sorry

end angle_order_l811_81192


namespace quadratic_completion_of_square_l811_81185

theorem quadratic_completion_of_square (x : ℝ) : 
  2 * x^2 - 4 * x - 3 = 0 ↔ (x - 1)^2 = 5/2 :=
by sorry

end quadratic_completion_of_square_l811_81185


namespace unique_solution_l811_81150

theorem unique_solution : ∃! (x : ℝ), 
  x > 0 ∧ 
  (Real.log x / Real.log 4) * (Real.log 9 / Real.log x) = Real.log 9 / Real.log 4 ∧ 
  x^2 = 16 := by
  sorry

end unique_solution_l811_81150


namespace coin_toss_experiment_l811_81116

theorem coin_toss_experiment (total_tosses : ℕ) (heads_frequency : ℚ) (tails_count : ℕ) :
  total_tosses = 100 →
  heads_frequency = 49/100 →
  tails_count = total_tosses - (total_tosses * heads_frequency).num →
  tails_count = 51 := by
  sorry

end coin_toss_experiment_l811_81116


namespace quadratic_inequality_solution_l811_81157

-- Define the quadratic function
def f (x : ℝ) := -2 * x^2 + x + 1

-- Define the solution set
def solution_set := {x : ℝ | -1/2 < x ∧ x < 1}

-- Theorem statement
theorem quadratic_inequality_solution :
  {x : ℝ | f x > 0} = solution_set :=
by sorry

end quadratic_inequality_solution_l811_81157


namespace daisy_count_proof_l811_81147

theorem daisy_count_proof (white : ℕ) (pink : ℕ) (red : ℕ) 
  (h1 : white = 6)
  (h2 : pink = 9 * white)
  (h3 : red = 4 * pink - 3) :
  white + pink + red = 273 := by
  sorry

end daisy_count_proof_l811_81147


namespace seahorse_penguin_ratio_l811_81162

theorem seahorse_penguin_ratio :
  let seahorses : ℕ := 70
  let penguins : ℕ := seahorses + 85
  (seahorses : ℚ) / penguins = 14 / 31 := by
  sorry

end seahorse_penguin_ratio_l811_81162


namespace a_10_value_l811_81143

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem a_10_value (a : ℕ → ℝ) :
  geometric_sequence a →
  (∀ n, a n > 0) →
  (a 1)^2 - 10 * (a 1) + 16 = 0 →
  (a 19)^2 - 10 * (a 19) + 16 = 0 →
  a 10 = 4 := by
  sorry

end a_10_value_l811_81143


namespace least_three_digit_multiple_of_eight_l811_81151

theorem least_three_digit_multiple_of_eight : 
  (∀ n : ℕ, 100 ≤ n ∧ n < 104 → n % 8 ≠ 0) ∧ 104 % 8 = 0 := by
  sorry

end least_three_digit_multiple_of_eight_l811_81151


namespace inequality_proof_l811_81135

theorem inequality_proof (S a b c x y z : ℝ) 
  (hS : S > 0)
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (eq1 : a + x = S) (eq2 : b + y = S) (eq3 : c + z = S) : 
  a * y + b * z + c * x < S^2 := by
sorry

end inequality_proof_l811_81135


namespace probability_seven_heads_ten_coins_prove_probability_seven_heads_ten_coins_l811_81107

/-- The probability of getting exactly 7 heads when flipping 10 fair coins -/
theorem probability_seven_heads_ten_coins : ℚ :=
  15 / 128

/-- Proof that the probability of getting exactly 7 heads when flipping 10 fair coins is 15/128 -/
theorem prove_probability_seven_heads_ten_coins :
  probability_seven_heads_ten_coins = 15 / 128 := by
  sorry

end probability_seven_heads_ten_coins_prove_probability_seven_heads_ten_coins_l811_81107


namespace max_value_of_expression_l811_81182

theorem max_value_of_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  let A := (a^4 + b^4 + c^4) / ((a + b + c)^4 - 80 * (a * b * c)^(4/3))
  A ≤ 3 ∧ (A = 3 ↔ a = b ∧ b = c) :=
by sorry

end max_value_of_expression_l811_81182


namespace profit_without_discount_l811_81148

theorem profit_without_discount (discount_percent : ℝ) (profit_with_discount_percent : ℝ) :
  discount_percent = 5 →
  profit_with_discount_percent = 20.65 →
  let cost_price := 100
  let selling_price_with_discount := cost_price * (1 - discount_percent / 100)
  let profit := cost_price * profit_with_discount_percent / 100
  let selling_price_without_discount := cost_price + profit
  profit / cost_price * 100 = 20.65 :=
by sorry

end profit_without_discount_l811_81148


namespace odd_function_value_l811_81154

/-- Given a function f(x) = sin(x + φ) + √3 cos(x + φ) where 0 ≤ φ ≤ π,
    if f(x) is an odd function, then f(π/6) = -1 -/
theorem odd_function_value (φ : Real) (h1 : 0 ≤ φ) (h2 : φ ≤ π) :
  let f : Real → Real := λ x => Real.sin (x + φ) + Real.sqrt 3 * Real.cos (x + φ)
  (∀ x, f (-x) = -f x) →
  f (π / 6) = -1 := by
sorry

end odd_function_value_l811_81154


namespace range_of_b_l811_81193

def f (b c x : ℝ) : ℝ := x^2 + b*x + c

theorem range_of_b (b c : ℝ) :
  (∃ x₀ : ℝ, f (f b c x₀) b c = 0 ∧ f b c x₀ ≠ 0) →
  b < 0 ∨ b ≥ 4 :=
by sorry

end range_of_b_l811_81193


namespace intersection_point_l811_81113

def line1 (x y : ℚ) : Prop := 8 * x - 5 * y = 4
def line2 (x y : ℚ) : Prop := 6 * x + 2 * y = 18

theorem intersection_point :
  ∃! p : ℚ × ℚ, line1 p.1 p.2 ∧ line2 p.1 p.2 ∧ p = (49/23, 60/23) := by sorry

end intersection_point_l811_81113


namespace intersection_of_A_and_B_l811_81144

-- Define sets A and B
def A : Set ℝ := {x | x < 2}
def B : Set ℝ := {x | 3 - 2*x > 0}

-- Theorem statement
theorem intersection_of_A_and_B : A ∩ B = {x : ℝ | x < 3/2} := by sorry

end intersection_of_A_and_B_l811_81144


namespace abc_remainder_mod_9_l811_81145

theorem abc_remainder_mod_9 (a b c : ℕ) : 
  a < 9 → b < 9 → c < 9 →
  (a + 2*b + 3*c) % 9 = 1 →
  (2*a + 3*b + c) % 9 = 2 →
  (3*a + b + 2*c) % 9 = 3 →
  (a * b * c) % 9 = 0 := by
sorry

end abc_remainder_mod_9_l811_81145


namespace track_circumference_l811_81161

/-- Represents the circumference of the circular track -/
def circumference : ℝ := 720

/-- Represents the distance B has traveled at the first meeting -/
def first_meeting_distance : ℝ := 150

/-- Represents the distance A has left to complete one lap at the second meeting -/
def second_meeting_remaining : ℝ := 90

/-- Represents the number of laps A has completed at the third meeting -/
def third_meeting_laps : ℝ := 1.5

theorem track_circumference :
  (first_meeting_distance + (circumference - first_meeting_distance) = circumference) ∧
  (circumference - second_meeting_remaining + (circumference / 2 + second_meeting_remaining) = circumference) ∧
  (third_meeting_laps * circumference + (circumference + first_meeting_distance) = 2 * circumference) :=
by sorry

end track_circumference_l811_81161


namespace cos_sixty_degrees_l811_81103

theorem cos_sixty_degrees : Real.cos (π / 3) = 1 / 2 := by
  sorry

end cos_sixty_degrees_l811_81103


namespace value_of_expression_l811_81175

theorem value_of_expression (x y : ℝ) (hx : x = 12) (hy : y = 7) :
  (x - y) * (x + y) = 95 := by
sorry

end value_of_expression_l811_81175


namespace max_squares_is_seven_l811_81140

/-- A shape formed by unit-length sticks on a plane -/
structure StickShape where
  sticks : ℕ
  squares : ℕ
  rows : ℕ
  first_row_squares : ℕ

/-- Predicate to check if a shape is valid according to the problem constraints -/
def is_valid_shape (s : StickShape) : Prop :=
  s.sticks = 20 ∧
  s.rows ≥ 1 ∧
  s.first_row_squares ≥ 1 ∧
  s.first_row_squares ≤ s.squares ∧
  (s.squares - s.first_row_squares) % (s.rows - 1) = 0

/-- The maximum number of squares that can be formed -/
def max_squares : ℕ := 7

/-- Theorem stating that the maximum number of squares is 7 -/
theorem max_squares_is_seven :
  ∀ s : StickShape, is_valid_shape s → s.squares ≤ max_squares :=
sorry

end max_squares_is_seven_l811_81140


namespace integral_f_equals_one_plus_pi_over_four_l811_81100

noncomputable section

-- Define the function f
def f (x : ℝ) : ℝ :=
  if 0 < x ∧ x ≤ 1 then Real.sqrt (1 - x^2)
  else if -1 ≤ x ∧ x ≤ 0 then x + 1
  else 0  -- This case is added to make the function total

-- State the theorem
theorem integral_f_equals_one_plus_pi_over_four :
  ∫ x in (-1)..1, f x = (1 + Real.pi) / 4 := by
  sorry

end

end integral_f_equals_one_plus_pi_over_four_l811_81100


namespace smallest_cyclic_divisible_by_1989_l811_81106

def is_cyclic_divisible_by_1989 (n : ℕ) : Prop :=
  ∀ k : ℕ, k < 10^n → ∀ i : ℕ, i < n → (k * 10^i + k / 10^(n - i)) % 1989 = 0

theorem smallest_cyclic_divisible_by_1989 :
  (∀ m < 48, ¬ is_cyclic_divisible_by_1989 m) ∧ is_cyclic_divisible_by_1989 48 :=
sorry

end smallest_cyclic_divisible_by_1989_l811_81106


namespace bookstore_shipment_l811_81105

/-- Proves that the total number of books in a shipment is 240, given that 25% are displayed
    in the front and the remaining 180 books are in the storage room. -/
theorem bookstore_shipment (displayed_percent : ℚ) (storage_count : ℕ) : ℕ :=
  let total_books : ℕ := 240
  have h1 : displayed_percent = 25 / 100 := by sorry
  have h2 : storage_count = 180 := by sorry
  have h3 : (1 - displayed_percent) * total_books = storage_count := by sorry
  total_books

#check bookstore_shipment

end bookstore_shipment_l811_81105


namespace negation_equivalence_angle_sine_equivalence_l811_81126

-- Define the proposition for the first part
def P (x : ℝ) : Prop := x^2 - x > 0

-- Theorem for the first part
theorem negation_equivalence : (¬ ∃ x, P x) ↔ (∀ x, ¬(P x)) := by sorry

-- Define a triangle
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  sum_angles : A + B + C = π
  positive_angles : 0 < A ∧ 0 < B ∧ 0 < C

-- Theorem for the second part
theorem angle_sine_equivalence (t : Triangle) : t.A > t.B ↔ Real.sin t.A > Real.sin t.B := by sorry

end negation_equivalence_angle_sine_equivalence_l811_81126


namespace bark_ratio_is_two_to_one_l811_81125

/-- The number of times the terrier's owner says "hush" -/
def hush_count : ℕ := 6

/-- The number of times the poodle barks -/
def poodle_barks : ℕ := 24

/-- The number of times the terrier barks before being hushed -/
def terrier_barks_per_hush : ℕ := 2

/-- Calculates the total number of times the terrier barks -/
def total_terrier_barks : ℕ := hush_count * terrier_barks_per_hush

/-- The ratio of poodle barks to terrier barks -/
def bark_ratio : ℚ := poodle_barks / total_terrier_barks

theorem bark_ratio_is_two_to_one : bark_ratio = 2 / 1 := by
  sorry

end bark_ratio_is_two_to_one_l811_81125


namespace quadratic_equation_solution_l811_81177

theorem quadratic_equation_solution :
  let f : ℝ → ℝ := λ x => x^2 - 2*x
  ∃ x₁ x₂ : ℝ, x₁ = 0 ∧ x₂ = 2 ∧ (∀ x : ℝ, f x = 0 ↔ x = x₁ ∨ x = x₂) :=
by sorry

end quadratic_equation_solution_l811_81177


namespace square_sum_zero_implies_both_zero_l811_81191

theorem square_sum_zero_implies_both_zero (a b : ℝ) : a^2 + b^2 = 0 → a = 0 ∧ b = 0 := by
  sorry

end square_sum_zero_implies_both_zero_l811_81191


namespace alice_unanswered_questions_l811_81109

/-- Represents the scoring systems and Alice's results in a math competition. -/
structure MathCompetition where
  total_questions : ℕ
  new_correct_points : ℕ
  new_incorrect_points : ℕ
  new_unanswered_points : ℕ
  old_start_points : ℕ
  old_correct_points : ℕ
  old_incorrect_points : Int
  old_unanswered_points : ℕ
  new_score : ℕ
  old_score : ℕ

/-- Calculates the number of unanswered questions in the math competition. -/
def calculate_unanswered_questions (comp : MathCompetition) : ℕ :=
  sorry

/-- Theorem stating that Alice left 2 questions unanswered. -/
theorem alice_unanswered_questions (comp : MathCompetition)
  (h1 : comp.total_questions = 30)
  (h2 : comp.new_correct_points = 4)
  (h3 : comp.new_incorrect_points = 0)
  (h4 : comp.new_unanswered_points = 1)
  (h5 : comp.old_start_points = 20)
  (h6 : comp.old_correct_points = 3)
  (h7 : comp.old_incorrect_points = -1)
  (h8 : comp.old_unanswered_points = 0)
  (h9 : comp.new_score = 87)
  (h10 : comp.old_score = 75) :
  calculate_unanswered_questions comp = 2 := by
  sorry

end alice_unanswered_questions_l811_81109


namespace six_last_digit_to_appear_l811_81170

-- Define the Fibonacci sequence
def fib : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fib n + fib (n + 1)

-- Define a function to get the units digit
def unitsDigit (n : ℕ) : ℕ := n % 10

-- Define a function to check if a digit has appeared in the sequence up to n
def digitAppeared (d : ℕ) (n : ℕ) : Prop :=
  ∃ k : ℕ, k ≤ n ∧ unitsDigit (fib k) = d

-- Theorem statement
theorem six_last_digit_to_appear :
  ∀ d : ℕ, d < 10 → d ≠ 6 →
    ∃ n : ℕ, digitAppeared d n ∧ ¬digitAppeared 6 n :=
by sorry

end six_last_digit_to_appear_l811_81170


namespace train_passing_time_train_passing_man_time_l811_81176

/-- The time it takes for a train to pass a man moving in the opposite direction -/
theorem train_passing_time (train_length : ℝ) (train_speed : ℝ) (man_speed : ℝ) : ℝ :=
  let relative_speed := train_speed + man_speed
  let relative_speed_ms := relative_speed * (1000 / 3600)
  train_length / relative_speed_ms

/-- Proof that the time for a 110m train moving at 40 km/h to pass a man moving at 4 km/h in the opposite direction is approximately 8.99 seconds -/
theorem train_passing_man_time :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
  |train_passing_time 110 40 4 - 8.99| < ε :=
sorry

end train_passing_time_train_passing_man_time_l811_81176


namespace meadowbrook_impossibility_l811_81102

theorem meadowbrook_impossibility : ¬ ∃ (h c : ℕ), 21 * h + 6 * c = 74 := by
  sorry

end meadowbrook_impossibility_l811_81102


namespace consecutive_integers_sum_l811_81108

theorem consecutive_integers_sum (x : ℤ) : x * (x + 1) = 440 → x + (x + 1) = 43 := by
  sorry

end consecutive_integers_sum_l811_81108


namespace part_time_employees_l811_81171

theorem part_time_employees (total_employees full_time_employees : ℕ) 
  (h1 : total_employees = 65134)
  (h2 : full_time_employees = 63093)
  (h3 : total_employees ≥ full_time_employees) :
  total_employees - full_time_employees = 2041 := by
  sorry

end part_time_employees_l811_81171


namespace weather_period_days_l811_81119

/-- Represents the weather conditions over a period of time. -/
structure WeatherPeriod where
  totalRainyDays : ℕ
  clearEvenings : ℕ
  clearMornings : ℕ
  morningRainImpliesClearEvening : Unit
  eveningRainImpliesClearMorning : Unit

/-- Calculates the total number of days in the weather period. -/
def totalDays (w : WeatherPeriod) : ℕ :=
  w.totalRainyDays + (w.clearEvenings + w.clearMornings - w.totalRainyDays) / 2

/-- Theorem stating that given the specific weather conditions, the total period is 11 days. -/
theorem weather_period_days (w : WeatherPeriod)
  (h1 : w.totalRainyDays = 9)
  (h2 : w.clearEvenings = 6)
  (h3 : w.clearMornings = 7) :
  totalDays w = 11 := by
  sorry

end weather_period_days_l811_81119


namespace decagon_vertex_sum_l811_81136

theorem decagon_vertex_sum (π : Fin 10 → Fin 10) 
  (hπ : Function.Bijective π) :
  ∃ k : Fin 10, 
    π k + π ((k + 9) % 10) + π ((k + 1) % 10) ≥ 17 := by
  sorry

end decagon_vertex_sum_l811_81136


namespace last_two_digits_sum_factorials_15_l811_81169

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def last_two_digits (n : ℕ) : ℕ := n % 100

def sum_factorials (n : ℕ) : ℕ := (List.range n).map factorial |>.sum

theorem last_two_digits_sum_factorials_15 :
  last_two_digits (sum_factorials 15) = 13 :=
by sorry

end last_two_digits_sum_factorials_15_l811_81169


namespace product_of_solutions_abs_y_eq_3_abs_y_minus_2_l811_81118

theorem product_of_solutions_abs_y_eq_3_abs_y_minus_2 :
  ∃ (y₁ y₂ : ℝ), (|y₁| = 3 * (|y₁| - 2)) ∧ (|y₂| = 3 * (|y₂| - 2)) ∧ y₁ ≠ y₂ ∧ y₁ * y₂ = -9 :=
sorry

end product_of_solutions_abs_y_eq_3_abs_y_minus_2_l811_81118


namespace probability_sum_10_l811_81138

/-- The number of faces on a standard die -/
def numFaces : ℕ := 6

/-- The number of dice rolled -/
def numDice : ℕ := 3

/-- The target sum we're looking for -/
def targetSum : ℕ := 10

/-- The total number of possible outcomes when rolling three dice -/
def totalOutcomes : ℕ := numFaces ^ numDice

/-- The number of favorable outcomes (sum of 10) -/
def favorableOutcomes : ℕ := 24

/-- The probability of rolling a sum of 10 with three standard six-sided dice -/
theorem probability_sum_10 : 
  (favorableOutcomes : ℚ) / totalOutcomes = 1 / 9 := by
  sorry


end probability_sum_10_l811_81138


namespace math_competition_problem_l811_81166

theorem math_competition_problem (p_a p_either : ℝ) (h1 : p_a = 0.6) (h2 : p_either = 0.92) :
  ∃ p_b : ℝ, p_b = 0.8 ∧ 1 - p_either = (1 - p_a) * (1 - p_b) :=
by sorry

end math_competition_problem_l811_81166


namespace dagger_example_l811_81181

-- Define the † operation
def dagger (m n p q : ℚ) : ℚ := m * p * (n / q)

-- Theorem statement
theorem dagger_example : dagger (5/9) (12/4) = 135 := by
  sorry

end dagger_example_l811_81181


namespace dance_team_quitters_l811_81114

theorem dance_team_quitters (initial_members : ℕ) (new_members : ℕ) (final_members : ℕ) 
  (h1 : initial_members = 25)
  (h2 : new_members = 13)
  (h3 : final_members = 30)
  : initial_members - (initial_members - final_members + new_members) = 8 := by
  sorry

end dance_team_quitters_l811_81114


namespace pizza_combinations_l811_81130

theorem pizza_combinations (n m : ℕ) (h1 : n = 8) (h2 : m = 5) : 
  Nat.choose n m = 56 := by
  sorry

end pizza_combinations_l811_81130


namespace imaginary_part_of_z_l811_81188

theorem imaginary_part_of_z (i : ℂ) (h : i * i = -1) :
  let z : ℂ := (1 + 2*i) / (1 - i)
  Complex.im z = 3/2 := by sorry

end imaginary_part_of_z_l811_81188


namespace matthew_cakes_l811_81112

theorem matthew_cakes (initial_crackers : ℕ) (friends : ℕ) (total_eaten_per_friend : ℕ)
  (h1 : initial_crackers = 14)
  (h2 : friends = 7)
  (h3 : total_eaten_per_friend = 5)
  (h4 : initial_crackers / friends = initial_crackers % friends) :
  ∃ initial_cakes : ℕ, initial_cakes = 21 := by
  sorry

end matthew_cakes_l811_81112


namespace geometric_sequence_problem_l811_81160

theorem geometric_sequence_problem (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a (n + 1) = a n * q) →  -- {aₙ} is a geometric sequence with common ratio q
  a 1 = 1 / 4 →                 -- a₁ = 1/4
  a 3 * a 5 = 4 * (a 4 - 1) →   -- a₃a₅ = 4(a₄ - 1)
  a 2 = 1 / 2 := by             -- a₂ = 1/2
sorry

end geometric_sequence_problem_l811_81160


namespace circles_intersect_example_l811_81167

/-- Two circles are intersecting if the distance between their centers is less than the sum of their radii
    and greater than the absolute difference of their radii. -/
def circles_intersect (r₁ r₂ d : ℝ) : Prop :=
  d < r₁ + r₂ ∧ d > |r₁ - r₂|

/-- Theorem: Two circles with radii 4 and 5, whose centers are 7 units apart, are intersecting. -/
theorem circles_intersect_example : circles_intersect 4 5 7 := by
  sorry


end circles_intersect_example_l811_81167


namespace corrected_mean_l811_81195

theorem corrected_mean (n : ℕ) (original_mean : ℚ) (incorrect_value correct_value : ℚ) :
  n = 50 ∧ original_mean = 36 ∧ incorrect_value = 23 ∧ correct_value = 34 →
  (n : ℚ) * original_mean + (correct_value - incorrect_value) = n * 36.22 :=
by sorry

end corrected_mean_l811_81195


namespace exists_polygon_with_n_axes_of_symmetry_l811_81110

/-- A convex polygon. -/
structure ConvexPolygon where
  -- Add necessary fields here
  -- This is a placeholder definition

/-- The number of axes of symmetry of a convex polygon. -/
def axesOfSymmetry (p : ConvexPolygon) : ℕ :=
  sorry -- Placeholder definition

/-- For any natural number n, there exists a convex polygon with exactly n axes of symmetry. -/
theorem exists_polygon_with_n_axes_of_symmetry :
  ∀ n : ℕ, ∃ p : ConvexPolygon, axesOfSymmetry p = n :=
sorry

end exists_polygon_with_n_axes_of_symmetry_l811_81110


namespace square_plus_double_eq_one_implies_double_square_plus_quad_minus_one_eq_one_l811_81168

/-- Given that a^2 + 2a = 1, prove that 2a^2 + 4a - 1 = 1 -/
theorem square_plus_double_eq_one_implies_double_square_plus_quad_minus_one_eq_one
  (a : ℝ) (h : a^2 + 2*a = 1) : 2*a^2 + 4*a - 1 = 1 := by
  sorry

end square_plus_double_eq_one_implies_double_square_plus_quad_minus_one_eq_one_l811_81168


namespace max_product_sum_constant_l811_81134

theorem max_product_sum_constant (a b M : ℝ) : 
  a > 0 → b > 0 → a + b = M → (∀ x y : ℝ, x > 0 → y > 0 → x + y = M → x * y ≤ 2) → M = 2 * Real.sqrt 2 :=
by
  sorry

end max_product_sum_constant_l811_81134


namespace clothing_sales_theorem_l811_81174

/-- Represents the sales data for a clothing store --/
structure SalesData where
  typeA_sold : ℕ
  typeB_sold : ℕ
  total_sales : ℕ

/-- Represents the pricing and sales increase data --/
structure ClothingData where
  typeA_price : ℕ
  typeB_price : ℕ
  typeA_increase : ℚ
  typeB_increase : ℚ

def store_A : SalesData := ⟨60, 15, 3600⟩
def store_B : SalesData := ⟨40, 60, 4400⟩

theorem clothing_sales_theorem (d : ClothingData) :
  d.typeA_price = 50 ∧ 
  d.typeB_price = 40 ∧ 
  d.typeA_increase = 1/5 ∧
  d.typeB_increase = 1/2 →
  (store_A.typeA_sold * d.typeA_price + store_A.typeB_sold * d.typeB_price = store_A.total_sales) ∧
  (store_B.typeA_sold * d.typeA_price + store_B.typeB_sold * d.typeB_price = store_B.total_sales) ∧
  ((store_A.typeA_sold + store_B.typeA_sold) * d.typeA_price * (1 + d.typeA_increase) : ℚ) / 
  ((store_A.typeB_sold + store_B.typeB_sold) * d.typeB_price * (1 + d.typeB_increase) : ℚ) = 4/3 :=
by sorry

end clothing_sales_theorem_l811_81174


namespace special_polyhedron_properties_l811_81142

/-- A convex polyhedron with triangular and hexagonal faces -/
structure Polyhedron where
  V : ℕ  -- number of vertices
  E : ℕ  -- number of edges
  F : ℕ  -- number of faces
  t : ℕ  -- number of triangular faces
  h : ℕ  -- number of hexagonal faces
  T : ℕ  -- number of triangular faces meeting at each vertex
  H : ℕ  -- number of hexagonal faces meeting at each vertex

/-- The properties of our specific polyhedron -/
def special_polyhedron : Polyhedron where
  V := 50
  E := 78
  F := 30
  t := 8
  h := 22
  T := 2
  H := 2

/-- Theorem stating the properties of the special polyhedron -/
theorem special_polyhedron_properties (p : Polyhedron) 
  (h1 : p.V - p.E + p.F = 2)  -- Euler's formula
  (h2 : p.F = 30)
  (h3 : p.F = p.t + p.h)
  (h4 : p.T = 2)
  (h5 : p.H = 2)
  (h6 : p.t = 8)
  (h7 : p.h = 22)
  (h8 : p.E = (3 * p.t + 6 * p.h) / 2) :
  100 * p.H + 10 * p.T + p.V = 270 := by
  sorry

#check special_polyhedron_properties

end special_polyhedron_properties_l811_81142


namespace positive_integer_triplets_equation_l811_81146

theorem positive_integer_triplets_equation :
  ∀ a b c : ℕ+,
    (6 : ℕ) ^ a.val = 1 + 2 ^ b.val + 3 ^ c.val ↔
    ((a, b, c) = (1, 1, 1) ∨ (a, b, c) = (2, 3, 3) ∨ (a, b, c) = (2, 5, 1)) :=
by sorry

end positive_integer_triplets_equation_l811_81146


namespace lottery_winnings_calculation_l811_81123

/-- Calculates the amount taken home from lottery winnings after tax and processing fee --/
def amountTakenHome (winnings : ℝ) (taxRate : ℝ) (processingFee : ℝ) : ℝ :=
  winnings - (winnings * taxRate) - processingFee

/-- Theorem stating that given specific lottery winnings, tax rate, and processing fee, 
    the amount taken home is $35 --/
theorem lottery_winnings_calculation :
  amountTakenHome 50 0.2 5 = 35 := by
  sorry

end lottery_winnings_calculation_l811_81123


namespace summer_camp_duration_l811_81155

def summer_camp (n : ℕ) (k : ℕ) (d : ℕ) : Prop :=
  -- n is the number of participants
  -- k is the number of participants chosen each day
  -- d is the number of days
  n = 15 ∧ 
  k = 3 ∧
  Nat.choose n 2 = d * Nat.choose k 2

theorem summer_camp_duration : 
  ∃ d : ℕ, summer_camp 15 3 d ∧ d = 35 := by
  sorry

end summer_camp_duration_l811_81155


namespace point_relationship_l811_81141

-- Define the function
def f (x : ℝ) : ℝ := -x^2 + 5

-- Define the points
def point1 : ℝ × ℝ := (-4, f (-4))
def point2 : ℝ × ℝ := (-1, f (-1))
def point3 : ℝ × ℝ := (2, f 2)

-- Theorem statement
theorem point_relationship :
  let y₁ := point1.2
  let y₂ := point2.2
  let y₃ := point3.2
  y₂ > y₃ ∧ y₃ > y₁ := by sorry

end point_relationship_l811_81141


namespace parabola_c_value_l811_81111

/-- Represents a parabola of the form x = ay² + by + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The x-coordinate of a point on the parabola -/
def Parabola.x_coord (p : Parabola) (y : ℝ) : ℝ :=
  p.a * y^2 + p.b * y + p.c

theorem parabola_c_value (p : Parabola) :
  p.x_coord 1 = -3 →  -- vertex at (-3, 1)
  p.x_coord 3 = -1 →  -- passes through (-1, 3)
  p.c = -5/2 := by
  sorry

end parabola_c_value_l811_81111


namespace machine_selling_price_l811_81183

/-- Calculates the selling price of a machine given its costs and desired profit percentage --/
def selling_price (purchase_price repair_cost transportation_charges profit_percentage : ℕ) : ℕ :=
  let total_cost := purchase_price + repair_cost + transportation_charges
  let profit := total_cost * profit_percentage / 100
  total_cost + profit

/-- Theorem stating that the selling price of the machine is 27000 --/
theorem machine_selling_price :
  selling_price 12000 5000 1000 50 = 27000 := by
  sorry

end machine_selling_price_l811_81183


namespace tax_discount_order_invariance_l811_81137

/-- Proves that the order of applying tax and discount doesn't affect the final price --/
theorem tax_discount_order_invariance 
  (original_price tax_rate discount_rate : ℝ) 
  (h_tax : 0 ≤ tax_rate) (h_discount : 0 ≤ discount_rate) (h_price : 0 < original_price) :
  original_price * (1 + tax_rate) * (1 - discount_rate) = 
  original_price * (1 - discount_rate) * (1 + tax_rate) :=
by sorry

end tax_discount_order_invariance_l811_81137


namespace hexagonal_board_cell_count_l811_81115

/-- The number of cells in a hexagonal board with side length m -/
def hexagonal_board_cells (m : ℕ) : ℕ := 3 * m^2 - 3 * m + 1

/-- Theorem: The number of cells in a hexagonal board with side length m is 3m^2 - 3m + 1 -/
theorem hexagonal_board_cell_count (m : ℕ) :
  hexagonal_board_cells m = 3 * m^2 - 3 * m + 1 := by
  sorry

end hexagonal_board_cell_count_l811_81115


namespace lcm_14_25_l811_81121

theorem lcm_14_25 : Nat.lcm 14 25 = 350 := by
  sorry

end lcm_14_25_l811_81121


namespace solve_system_l811_81196

theorem solve_system (a b c d : ℚ)
  (eq1 : a = 2 * b + c)
  (eq2 : b = 2 * c + d)
  (eq3 : 2 * c = d + a - 1)
  (eq4 : d = a - c) :
  b = 2 / 9 := by
  sorry

end solve_system_l811_81196


namespace residue_of_15_power_1234_mod_19_l811_81129

theorem residue_of_15_power_1234_mod_19 :
  (15 : ℤ)^1234 ≡ 6 [ZMOD 19] := by sorry

end residue_of_15_power_1234_mod_19_l811_81129


namespace range_of_z_l811_81153

theorem range_of_z (x y : ℝ) (h1 : x + 2 ≥ y) (h2 : x + 2*y ≥ 4) (h3 : y ≤ 5 - 2*x) :
  let z := (2*x + y - 1) / (x + 1)
  ∃ (z_min z_max : ℝ), z_min = 1 ∧ z_max = 2 ∧ ∀ z', z' = z → z_min ≤ z' ∧ z' ≤ z_max :=
by sorry

end range_of_z_l811_81153


namespace inequality_proof_l811_81158

/-- Given positive real numbers a, b, c, and the function f(x) = |x+a| * |x+b|,
    prove that f(1)f(c) ≥ 16abc -/
theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  let f := fun x => |x + a| * |x + b|
  f 1 * f c ≥ 16 * a * b * c := by
sorry

end inequality_proof_l811_81158


namespace expression_value_l811_81190

theorem expression_value (x y : ℤ) (hx : x = -5) (hy : y = 8) :
  2 * (x - y)^2 - x * y = 378 := by
  sorry

end expression_value_l811_81190


namespace cristobal_beatrix_pages_difference_l811_81127

theorem cristobal_beatrix_pages_difference (beatrix_pages cristobal_extra_pages : ℕ) 
  (h1 : beatrix_pages = 704)
  (h2 : cristobal_extra_pages = 1423) :
  (beatrix_pages + cristobal_extra_pages) - (3 * beatrix_pages) = 15 := by
  sorry

end cristobal_beatrix_pages_difference_l811_81127


namespace general_term_k_n_l811_81164

/-- An arithmetic sequence with the given properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  d : ℝ
  d_nonzero : d ≠ 0
  is_arithmetic : ∀ n, a (n + 1) = a n + d
  a2_geometric_mean : a 2 ^ 2 = a 1 * a 4
  geometric_subseq : ∀ n, a (3^n) / a (3^(n-1)) = a 3 / a 1

/-- The theorem stating the general term of k_n -/
theorem general_term_k_n (seq : ArithmeticSequence) : 
  ∀ n : ℕ, ∃ k_n : ℕ, seq.a k_n = seq.a 1 * (3 : ℝ)^(n-1) ∧ k_n = 3^(n-1) := by
  sorry

end general_term_k_n_l811_81164


namespace solution_count_l811_81173

/-- The greatest integer function (floor function) -/
noncomputable def floor (x : ℝ) : ℤ :=
  Int.floor x

/-- The number of solutions to x^2 - ⌊x⌋^2 = (x - ⌊x⌋)^2 in [1, n] -/
def num_solutions (n : ℕ) : ℕ :=
  n^2 - n + 1

/-- Theorem stating the number of solutions to the equation -/
theorem solution_count (n : ℕ) :
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ n →
    x^2 - (floor x)^2 = (x - floor x)^2) →
  num_solutions n = n^2 - n + 1 :=
by sorry

end solution_count_l811_81173


namespace min_fence_posts_for_grazing_area_l811_81159

/-- Calculates the number of fence posts required for a rectangular grazing area -/
def fence_posts (length width post_spacing : ℕ) : ℕ :=
  let perimeter := 2 * (length + width)
  let long_side_posts := length / post_spacing + 1
  let short_side_posts := 2 * (width / post_spacing)
  long_side_posts + short_side_posts

/-- Theorem stating the minimum number of fence posts required for the given conditions -/
theorem min_fence_posts_for_grazing_area :
  fence_posts 80 40 10 = 17 :=
sorry

end min_fence_posts_for_grazing_area_l811_81159


namespace average_of_ABC_l811_81117

theorem average_of_ABC (A B C : ℝ) 
  (eq1 : 501 * C - 1002 * A = 2002)
  (eq2 : 501 * B + 2002 * A = 2505) :
  (A + B + C) / 3 = -A / 3 + 3 := by
  sorry

end average_of_ABC_l811_81117


namespace problem_solution_l811_81179

theorem problem_solution : ∃ x : ℝ, 550 - (x / 20.8) = 545 ∧ x = 104 := by
  sorry

end problem_solution_l811_81179


namespace average_marks_combined_l811_81122

theorem average_marks_combined (n1 n2 : ℕ) (avg1 avg2 : ℝ) :
  n1 = 20 →
  n2 = 50 →
  avg1 = 40 →
  avg2 = 60 →
  let total_marks := n1 * avg1 + n2 * avg2
  let total_students := n1 + n2
  abs ((total_marks / total_students) - 54.29) < 0.01 := by
  sorry

end average_marks_combined_l811_81122


namespace unique_solution_is_five_l811_81139

/-- The function f(x) = 2x - 3 -/
def f (x : ℝ) : ℝ := 2 * x - 3

/-- The theorem stating that x = 5 is the unique solution to the equation -/
theorem unique_solution_is_five :
  ∃! x : ℝ, 2 * (f x) - 11 = f (x - 2) :=
by
  -- The proof goes here
  sorry

end unique_solution_is_five_l811_81139


namespace height_opposite_y_is_8_l811_81163

/-- Regular triangle XYZ with pillars -/
structure Triangle where
  /-- Side length of the triangle -/
  side : ℝ
  /-- Height of pillar at X -/
  height_x : ℝ
  /-- Height of pillar at Y -/
  height_y : ℝ
  /-- Height of pillar at Z -/
  height_z : ℝ

/-- Calculate the height of the pillar opposite to Y -/
def height_opposite_y (t : Triangle) : ℝ :=
  -- Implementation details omitted
  sorry

/-- Theorem stating that the height of the pillar opposite Y is 8m -/
theorem height_opposite_y_is_8 (t : Triangle) 
  (h_regular : t.side > 0)
  (h_x : t.height_x = 8)
  (h_y : t.height_y = 5)
  (h_z : t.height_z = 7) : 
  height_opposite_y t = 8 :=
sorry

end height_opposite_y_is_8_l811_81163


namespace divisor_calculation_l811_81131

theorem divisor_calculation (quotient dividend : ℚ) (h1 : quotient = -5/16) (h2 : dividend = -5/2) :
  dividend / quotient = 8 := by
  sorry

end divisor_calculation_l811_81131


namespace set_partition_l811_81152

def S : Set ℝ := {-5/6, 0, -3.5, 1.2, 6}

def N : Set ℝ := {x ∈ S | x < 0}

def NN : Set ℝ := {x ∈ S | x ≥ 0}

theorem set_partition :
  N = {-5/6, -3.5} ∧ NN = {0, 1.2, 6} := by sorry

end set_partition_l811_81152


namespace negation_of_p_l811_81133

-- Define the proposition p
def p (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, (f x₂ - f x₁) * (x₂ - x₁) ≥ 0

-- State the theorem
theorem negation_of_p (f : ℝ → ℝ) : ¬(p f) ↔ ∃ x₁ x₂ : ℝ, (f x₂ - f x₁) * (x₂ - x₁) < 0 :=
  sorry

end negation_of_p_l811_81133


namespace alcohol_percentage_first_vessel_l811_81197

theorem alcohol_percentage_first_vessel
  (vessel1_capacity : ℝ)
  (vessel2_capacity : ℝ)
  (vessel2_alcohol_percentage : ℝ)
  (total_liquid : ℝ)
  (final_vessel_capacity : ℝ)
  (final_mixture_concentration : ℝ)
  (h1 : vessel1_capacity = 2)
  (h2 : vessel2_capacity = 6)
  (h3 : vessel2_alcohol_percentage = 50)
  (h4 : total_liquid = 8)
  (h5 : final_vessel_capacity = 10)
  (h6 : final_mixture_concentration = 37)
  : ∃ (vessel1_alcohol_percentage : ℝ),
    vessel1_alcohol_percentage = 35 ∧
    (vessel1_alcohol_percentage / 100 * vessel1_capacity +
     vessel2_alcohol_percentage / 100 * vessel2_capacity =
     final_mixture_concentration / 100 * final_vessel_capacity) :=
by sorry

end alcohol_percentage_first_vessel_l811_81197


namespace quadratic_function_passes_through_points_l811_81186

-- Define the quadratic function
def f (x : ℝ) : ℝ := 4 * x^2 + 5 * x

-- Define the three points
def p1 : ℝ × ℝ := (0, 0)
def p2 : ℝ × ℝ := (-1, -1)
def p3 : ℝ × ℝ := (1, 9)

-- Theorem statement
theorem quadratic_function_passes_through_points :
  f p1.1 = p1.2 ∧ f p2.1 = p2.2 ∧ f p3.1 = p3.2 := by
  sorry

end quadratic_function_passes_through_points_l811_81186


namespace total_books_l811_81165

theorem total_books (joan_books tom_books : ℕ) 
  (h1 : joan_books = 10) 
  (h2 : tom_books = 38) : 
  joan_books + tom_books = 48 := by
  sorry

end total_books_l811_81165
