import Mathlib

namespace NUMINAMATH_CALUDE_probability_between_X_and_Z_l1548_154841

/-- Given a line segment XW where XW = 4XZ = 8YW, the probability of selecting a point between X and Z is 1/4 -/
theorem probability_between_X_and_Z (XW XZ YW : ℝ) 
  (h1 : XW = 4 * XZ) 
  (h2 : XW = 8 * YW) 
  (h3 : XW > 0) : 
  XZ / XW = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_probability_between_X_and_Z_l1548_154841


namespace NUMINAMATH_CALUDE_arithmetic_equality_l1548_154826

theorem arithmetic_equality : 5 - 4 * 3 / 2 + 1 = 0 := by sorry

end NUMINAMATH_CALUDE_arithmetic_equality_l1548_154826


namespace NUMINAMATH_CALUDE_cyclists_time_apart_l1548_154868

/-- Calculates the time taken for two cyclists to be 200 miles apart -/
theorem cyclists_time_apart (v_east : ℝ) (v_west : ℝ) (distance : ℝ) : 
  v_east = 22 →
  v_west = v_east + 4 →
  distance = 200 →
  (distance / (v_east + v_west) : ℝ) = 25 / 6 := by
  sorry

#check cyclists_time_apart

end NUMINAMATH_CALUDE_cyclists_time_apart_l1548_154868


namespace NUMINAMATH_CALUDE_line_mb_product_l1548_154836

theorem line_mb_product (m b : ℚ) : 
  (∀ x y : ℚ, y = m * x + b) →  -- Line equation
  b = -3 →                      -- y-intercept
  5 = m * 3 + b →               -- Line passes through (3, 5)
  m * b = -8 := by sorry

end NUMINAMATH_CALUDE_line_mb_product_l1548_154836


namespace NUMINAMATH_CALUDE_find_A_l1548_154815

def round_down_hundreds (n : ℕ) : ℕ := n / 100 * 100

def is_valid_number (n : ℕ) : Prop := 
  ∃ (A : ℕ), A < 10 ∧ n = 1000 + A * 100 + 77

theorem find_A : 
  ∀ (n : ℕ), is_valid_number n → round_down_hundreds n = 1700 → n = 1777 :=
sorry

end NUMINAMATH_CALUDE_find_A_l1548_154815


namespace NUMINAMATH_CALUDE_unique_pair_divisibility_l1548_154853

theorem unique_pair_divisibility (a b : ℕ) : 
  (7^a - 3^b) ∣ (a^4 + b^2) → a = 2 ∧ b = 4 := by
  sorry

end NUMINAMATH_CALUDE_unique_pair_divisibility_l1548_154853


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l1548_154874

/-- Two lines in the plane -/
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if two lines are parallel -/
def areParallel (l1 l2 : Line2D) : Prop :=
  l1.a * l2.b = l2.a * l1.b

/-- The first line l₁: ax + y + 1 = 0 -/
def l1 (a : ℝ) : Line2D :=
  ⟨a, 1, 1⟩

/-- The second line l₂: 2x + (a + 1)y + 3 = 0 -/
def l2 (a : ℝ) : Line2D :=
  ⟨2, a + 1, 3⟩

/-- a = 1 is sufficient but not necessary for the lines to be parallel -/
theorem sufficient_not_necessary :
  (∀ a : ℝ, a = 1 → areParallel (l1 a) (l2 a)) ∧
  ¬(∀ a : ℝ, areParallel (l1 a) (l2 a) → a = 1) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l1548_154874


namespace NUMINAMATH_CALUDE_photo_arrangements_l1548_154888

/-- The number of ways to arrange n distinct objects. -/
def arrangements (n : ℕ) : ℕ := Nat.factorial n

/-- The number of ways to arrange k objects out of n distinct objects. -/
def permutations (n k : ℕ) : ℕ := 
  if k ≤ n then Nat.factorial n / Nat.factorial (n - k) else 0

theorem photo_arrangements (teachers students : ℕ) 
  (h1 : teachers = 4) (h2 : students = 4) : 
  /- Students stand together -/
  (arrangements students * arrangements (teachers + 1) = 2880) ∧ 
  /- No two students are adjacent -/
  (arrangements teachers * permutations (teachers + 1) students = 2880) ∧
  /- Teachers and students alternate -/
  (2 * arrangements teachers * arrangements students = 1152) := by
  sorry

#check photo_arrangements

end NUMINAMATH_CALUDE_photo_arrangements_l1548_154888


namespace NUMINAMATH_CALUDE_price_after_discounts_l1548_154835

/-- The original price of an article before discounts -/
def original_price : ℝ := 70.59

/-- The final price after discounts -/
def final_price : ℝ := 36

/-- The first discount rate -/
def discount1 : ℝ := 0.15

/-- The second discount rate -/
def discount2 : ℝ := 0.25

/-- The third discount rate -/
def discount3 : ℝ := 0.20

/-- Theorem stating that the original price results in the final price after applying the discounts -/
theorem price_after_discounts : 
  ∃ ε > 0, abs (original_price * (1 - discount1) * (1 - discount2) * (1 - discount3) - final_price) < ε :=
sorry

end NUMINAMATH_CALUDE_price_after_discounts_l1548_154835


namespace NUMINAMATH_CALUDE_quadratic_equation_transformation_l1548_154899

theorem quadratic_equation_transformation (x : ℝ) :
  (x^2 + 2*x - 2 = 0) ↔ ((x + 1)^2 = 3) :=
sorry

end NUMINAMATH_CALUDE_quadratic_equation_transformation_l1548_154899


namespace NUMINAMATH_CALUDE_overlap_squares_area_l1548_154810

/-- Given two identical squares with side length 12 that overlap to form a 12 by 20 rectangle,
    the area of the non-overlapping region of one square is 48. -/
theorem overlap_squares_area (square_side : ℝ) (rect_length : ℝ) (rect_width : ℝ) : 
  square_side = 12 →
  rect_length = 20 →
  rect_width = 12 →
  (2 * square_side^2) - (rect_length * rect_width) = 48 := by
  sorry

end NUMINAMATH_CALUDE_overlap_squares_area_l1548_154810


namespace NUMINAMATH_CALUDE_factorial_ratio_plus_two_l1548_154834

theorem factorial_ratio_plus_two : Nat.factorial 50 / Nat.factorial 48 + 2 = 2452 := by
  sorry

end NUMINAMATH_CALUDE_factorial_ratio_plus_two_l1548_154834


namespace NUMINAMATH_CALUDE_chess_tournament_games_l1548_154812

theorem chess_tournament_games (n : ℕ) (total_games : ℕ) : 
  n = 6 → total_games = 12 → (n * (n - 1)) / 2 = total_games → n - 1 = 5 :=
by
  sorry

#check chess_tournament_games

end NUMINAMATH_CALUDE_chess_tournament_games_l1548_154812


namespace NUMINAMATH_CALUDE_floor_inequality_and_factorial_divisibility_l1548_154850

theorem floor_inequality_and_factorial_divisibility 
  (x y : ℝ) (m n : ℕ+) 
  (hx : x ≥ 0) (hy : y ≥ 0) : 
  (⌊5 * x⌋ + ⌊5 * y⌋ ≥ ⌊3 * x + y⌋ + ⌊3 * y + x⌋) ∧ 
  (∃ k : ℕ, k * (m.val.factorial * n.val.factorial * (3 * m.val + n.val).factorial * (3 * n.val + m.val).factorial) = 
   (5 * m.val).factorial * (5 * n.val).factorial) :=
sorry

end NUMINAMATH_CALUDE_floor_inequality_and_factorial_divisibility_l1548_154850


namespace NUMINAMATH_CALUDE_sarah_bowling_score_l1548_154878

theorem sarah_bowling_score (greg_score sarah_score : ℕ) : 
  sarah_score = greg_score + 30 →
  (sarah_score + greg_score) / 2 = 95 →
  sarah_score = 110 := by
  sorry

end NUMINAMATH_CALUDE_sarah_bowling_score_l1548_154878


namespace NUMINAMATH_CALUDE_smallest_b_for_composite_polynomial_l1548_154861

theorem smallest_b_for_composite_polynomial : 
  ∃ (b : ℕ), b > 0 ∧ 
  (∀ (x : ℤ), ¬ Nat.Prime (x^4 + x^3 + b^2 + 5).natAbs) ∧
  (∀ (b' : ℕ), 0 < b' ∧ b' < b → 
    ∃ (x : ℤ), Nat.Prime (x^4 + x^3 + b'^2 + 5).natAbs) ∧
  b = 7 := by
sorry

end NUMINAMATH_CALUDE_smallest_b_for_composite_polynomial_l1548_154861


namespace NUMINAMATH_CALUDE_pocket_probabilities_l1548_154831

/-- Represents the number of balls in the pocket -/
def total_balls : ℕ := 5

/-- Represents the number of white balls in the pocket -/
def white_balls : ℕ := 3

/-- Represents the number of black balls in the pocket -/
def black_balls : ℕ := 2

/-- Represents the number of balls drawn at once -/
def drawn_balls : ℕ := 2

/-- The total number of ways to draw 2 balls from 5 balls -/
def total_events : ℕ := Nat.choose total_balls drawn_balls

/-- The probability of drawing two white balls -/
def prob_two_white : ℚ := (Nat.choose white_balls drawn_balls : ℚ) / total_events

/-- The probability of drawing one black and one white ball -/
def prob_one_black_one_white : ℚ := (white_balls * black_balls : ℚ) / total_events

theorem pocket_probabilities :
  total_events = 10 ∧
  prob_two_white = 3 / 10 ∧
  prob_one_black_one_white = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_pocket_probabilities_l1548_154831


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l1548_154833

open Real

theorem sufficient_not_necessary (α : ℝ) :
  (∀ α, α = π/4 → sin α = cos α) ∧
  (∃ α, α ≠ π/4 ∧ sin α = cos α) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l1548_154833


namespace NUMINAMATH_CALUDE_right_triangle_area_divisibility_l1548_154882

theorem right_triangle_area_divisibility (a b c : ℕ) : 
  a^2 + b^2 = c^2 → -- Pythagorean theorem
  c % 5 ≠ 0 → -- hypotenuse not divisible by 5
  ∃ k : ℕ, a * b = 20 * k -- area is divisible by 10
  := by sorry

end NUMINAMATH_CALUDE_right_triangle_area_divisibility_l1548_154882


namespace NUMINAMATH_CALUDE_modular_inverse_15_mod_16_l1548_154807

theorem modular_inverse_15_mod_16 : ∃ x : ℤ, (15 * x) % 16 = 1 :=
by
  use 15
  sorry

end NUMINAMATH_CALUDE_modular_inverse_15_mod_16_l1548_154807


namespace NUMINAMATH_CALUDE_polynomial_remainder_l1548_154880

def polynomial (y : ℝ) : ℝ := y^5 - 8*y^4 + 12*y^3 + 25*y^2 - 40*y + 24

theorem polynomial_remainder : 
  ∃ q : ℝ → ℝ, polynomial = (λ y => (y - 4) * q y + 8) := by sorry

end NUMINAMATH_CALUDE_polynomial_remainder_l1548_154880


namespace NUMINAMATH_CALUDE_wife_account_percentage_l1548_154806

def income : ℝ := 1000000

def children_percentage : ℝ := 0.2
def num_children : ℕ := 3
def orphan_house_percentage : ℝ := 0.05
def final_amount : ℝ := 50000

theorem wife_account_percentage : 
  let children_total := children_percentage * num_children * income
  let remaining_after_children := income - children_total
  let orphan_house_donation := orphan_house_percentage * remaining_after_children
  let remaining_after_donation := remaining_after_children - orphan_house_donation
  let wife_account := remaining_after_donation - final_amount
  (wife_account / income) * 100 = 33 := by sorry

end NUMINAMATH_CALUDE_wife_account_percentage_l1548_154806


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1548_154852

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  is_arithmetic_sequence a →
  (a 2 + a 3 + a 10 + a 11 = 48) →
  a 6 + a 7 = 24 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1548_154852


namespace NUMINAMATH_CALUDE_f_properties_l1548_154843

noncomputable section

def f (x : ℝ) := (Real.log x) / x

theorem f_properties :
  ∀ x > 0,
  (∃ y, f x = y ∧ x - y - 1 = 0) ∧ 
  (∀ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < Real.exp 1 → f x₁ < f x₂) ∧
  (∀ x₁ x₂, Real.exp 1 < x₁ ∧ x₁ < x₂ → f x₁ > f x₂) ∧
  (f (Real.exp 1) = (Real.exp 1)⁻¹) ∧
  (∀ x, x > 0 → f x ≤ (Real.exp 1)⁻¹) :=
by sorry

end

end NUMINAMATH_CALUDE_f_properties_l1548_154843


namespace NUMINAMATH_CALUDE_difference_even_plus_five_minus_odd_l1548_154875

/-- Sum of the first n odd counting numbers -/
def sumOddNumbers (n : ℕ) : ℕ := n * (2 * n - 1)

/-- Sum of the first n even counting numbers plus 5 added to each number -/
def sumEvenNumbersPlusFive (n : ℕ) : ℕ := n * (2 * n + 5)

/-- The difference between the sum of the first 3000 even counting numbers plus 5 
    added to each number and the sum of the first 3000 odd counting numbers is 18000 -/
theorem difference_even_plus_five_minus_odd : 
  sumEvenNumbersPlusFive 3000 - sumOddNumbers 3000 = 18000 := by
  sorry

end NUMINAMATH_CALUDE_difference_even_plus_five_minus_odd_l1548_154875


namespace NUMINAMATH_CALUDE_third_player_win_probability_l1548_154881

/-- Represents a fair six-sided die --/
def FairDie : Finset ℕ := Finset.range 6

/-- The probability of rolling a 6 on a fair die --/
def probWin : ℚ := 1 / 6

/-- The probability of not rolling a 6 on a fair die --/
def probLose : ℚ := 1 - probWin

/-- The number of players --/
def numPlayers : ℕ := 3

/-- The probability that the third player wins the game --/
def probThirdPlayerWins : ℚ := 1 / 91

theorem third_player_win_probability :
  probThirdPlayerWins = (probWin^numPlayers) / (1 - probLose^numPlayers) :=
by sorry

end NUMINAMATH_CALUDE_third_player_win_probability_l1548_154881


namespace NUMINAMATH_CALUDE_negation_of_proposition_l1548_154818

theorem negation_of_proposition (a : ℝ) : 
  ¬(a ≠ 0 → a^2 > 0) ↔ (a = 0 → a^2 ≤ 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l1548_154818


namespace NUMINAMATH_CALUDE_pentagon_fifth_angle_l1548_154824

/-- The sum of angles in a pentagon is 540 degrees -/
def pentagon_angle_sum : ℝ := 540

/-- The known angles of the pentagon -/
def known_angles : List ℝ := [130, 80, 105, 110]

/-- The measure of the unknown angle Q -/
def angle_q : ℝ := 115

/-- Theorem: In a pentagon with four known angles measuring 130°, 80°, 105°, and 110°, 
    the measure of the fifth angle is 115°. -/
theorem pentagon_fifth_angle :
  pentagon_angle_sum = (known_angles.sum + angle_q) :=
by sorry

end NUMINAMATH_CALUDE_pentagon_fifth_angle_l1548_154824


namespace NUMINAMATH_CALUDE_division_problem_l1548_154894

theorem division_problem (total : ℚ) (a b c : ℚ) : 
  total = 527 →
  a = (2/3) * b →
  b = (1/4) * c →
  a + b + c = total →
  c = 372 :=
by sorry

end NUMINAMATH_CALUDE_division_problem_l1548_154894


namespace NUMINAMATH_CALUDE_ellipse_higher_focus_coordinates_l1548_154896

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse -/
structure Ellipse where
  majorAxis : Point × Point
  minorAxis : Point × Point

/-- The focus of an ellipse with higher y-coordinate -/
def higherFocus (e : Ellipse) : Point :=
  sorry

theorem ellipse_higher_focus_coordinates :
  let e : Ellipse := {
    majorAxis := (⟨3, 0⟩, ⟨3, 8⟩),
    minorAxis := (⟨1, 4⟩, ⟨5, 4⟩)
  }
  let focus := higherFocus e
  focus.x = 3 ∧ focus.y = 4 + 2 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_higher_focus_coordinates_l1548_154896


namespace NUMINAMATH_CALUDE_ice_cream_sales_theorem_l1548_154864

/-- The number of ice cream cones sold on Tuesday -/
def tuesday_sales : ℕ := 12000

/-- The number of ice cream cones sold on Wednesday -/
def wednesday_sales : ℕ := 2 * tuesday_sales

/-- The total number of ice cream cones sold on Tuesday and Wednesday -/
def total_sales : ℕ := tuesday_sales + wednesday_sales

theorem ice_cream_sales_theorem : total_sales = 36000 := by
  sorry

end NUMINAMATH_CALUDE_ice_cream_sales_theorem_l1548_154864


namespace NUMINAMATH_CALUDE_missing_digit_is_four_l1548_154890

def set_of_numbers : List Nat := [8, 88, 888, 8888, 88888, 888888, 8888888, 88888888, 888888888]

def arithmetic_mean (numbers : List Nat) : Rat :=
  (numbers.sum : Rat) / numbers.length

theorem missing_digit_is_four :
  let mean := arithmetic_mean set_of_numbers
  ∃ (n : Nat), 
    (n = Int.floor mean) ∧ 
    (n ≥ 100000000 ∧ n < 1000000000) ∧ 
    (∀ d₁ d₂, d₁ ∈ n.digits 10 → d₂ ∈ n.digits 10 → d₁ ≠ d₂ → d₁ ≠ d₂) ∧
    (4 ∉ n.digits 10) :=
by sorry

end NUMINAMATH_CALUDE_missing_digit_is_four_l1548_154890


namespace NUMINAMATH_CALUDE_bake_sale_girls_l1548_154858

theorem bake_sale_girls (initial_total : ℕ) : 
  -- Initial conditions
  (3 * initial_total / 5 : ℚ) = initial_total * (60 : ℚ) / 100 →
  -- Changes in group composition
  let new_total := initial_total - 1 + 3
  let new_girls := (3 * initial_total / 5 : ℚ) - 3
  -- Final condition
  new_girls / new_total = (1 : ℚ) / 2 →
  -- Conclusion
  (3 * initial_total / 5 : ℚ) = 24 := by
sorry

end NUMINAMATH_CALUDE_bake_sale_girls_l1548_154858


namespace NUMINAMATH_CALUDE_ones_digit_of_large_power_l1548_154802

theorem ones_digit_of_large_power : ∃ n : ℕ, n > 0 ∧ 17^(17*(5^5)) ≡ 7 [ZMOD 10] := by
  sorry

end NUMINAMATH_CALUDE_ones_digit_of_large_power_l1548_154802


namespace NUMINAMATH_CALUDE_particle_probability_l1548_154883

/-- Represents the probability of a particle hitting (0,0) starting from (x,y) -/
def P (x y : ℕ) : ℚ :=
  if x = 0 ∧ y = 0 then 1
  else if x = 0 ∨ y = 0 then 0
  else (P (x-1) y + P x (y-1) + P (x-1) (y-1) + P (x-2) (y-2)) / 4

/-- The probability of hitting (0,0) starting from (5,5) is 3805/16384 -/
theorem particle_probability : P 5 5 = 3805 / 16384 := by
  sorry

end NUMINAMATH_CALUDE_particle_probability_l1548_154883


namespace NUMINAMATH_CALUDE_tanya_bought_eleven_pears_l1548_154898

/-- Represents the number of pears Tanya bought -/
def num_pears : ℕ := sorry

/-- Represents the number of Granny Smith apples Tanya bought -/
def num_apples : ℕ := 4

/-- Represents the number of pineapples Tanya bought -/
def num_pineapples : ℕ := 2

/-- Represents the basket of plums as a single item -/
def num_plum_baskets : ℕ := 1

/-- Represents the total number of fruit items Tanya bought -/
def total_fruits : ℕ := num_pears + num_apples + num_pineapples + num_plum_baskets

/-- Represents the number of fruits remaining in the bag after half fell out -/
def remaining_fruits : ℕ := 9

theorem tanya_bought_eleven_pears :
  num_pears = 11 ∧
  total_fruits = 2 * remaining_fruits :=
by sorry

end NUMINAMATH_CALUDE_tanya_bought_eleven_pears_l1548_154898


namespace NUMINAMATH_CALUDE_inverse_sum_property_l1548_154867

-- Define a function f with domain ℝ and its inverse
variable (f : ℝ → ℝ)
variable (f_inv : ℝ → ℝ)

-- Define the property that f is invertible
def is_inverse (f f_inv : ℝ → ℝ) : Prop :=
  ∀ x, f (f_inv x) = x ∧ f_inv (f x) = x

-- State the theorem
theorem inverse_sum_property
  (h1 : is_inverse f f_inv)
  (h2 : ∀ x : ℝ, f x + f (-x) = 1) :
  ∀ x : ℝ, f_inv (2010 - x) + f_inv (x - 2009) = 0 :=
sorry

end NUMINAMATH_CALUDE_inverse_sum_property_l1548_154867


namespace NUMINAMATH_CALUDE_range_of_f_l1548_154862

def f (x : ℝ) := |x + 5| - |x - 3|

theorem range_of_f :
  ∀ y ∈ Set.range f, -8 ≤ y ∧ y ≤ 8 ∧
  ∀ z, -8 ≤ z ∧ z ≤ 8 → ∃ x, f x = z :=
by sorry

end NUMINAMATH_CALUDE_range_of_f_l1548_154862


namespace NUMINAMATH_CALUDE_triangle_area_theorem_l1548_154851

theorem triangle_area_theorem (x : ℝ) (h1 : x > 0) : 
  (1/2 : ℝ) * x * (3*x) = 72 → x = 4 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_theorem_l1548_154851


namespace NUMINAMATH_CALUDE_angle_of_inclination_l1548_154872

theorem angle_of_inclination (x y : ℝ) :
  let line_equation := (Real.sqrt 3) * x + y - 3 = 0
  let angle_of_inclination := 2 * Real.pi / 3
  line_equation → angle_of_inclination = Real.arctan (-(Real.sqrt 3)) + Real.pi :=
by
  sorry

end NUMINAMATH_CALUDE_angle_of_inclination_l1548_154872


namespace NUMINAMATH_CALUDE_parabola_directrix_parameter_l1548_154805

/-- Given a parabola with equation x² = ay and directrix y = 1, prove that a = -4 -/
theorem parabola_directrix_parameter (a : ℝ) : 
  (∀ x y : ℝ, x^2 = a*y) →  -- Parabola equation
  (1 = -a/4) →              -- Relation between 'a' and directrix
  a = -4 := by
sorry

end NUMINAMATH_CALUDE_parabola_directrix_parameter_l1548_154805


namespace NUMINAMATH_CALUDE_inequality_proof_l1548_154895

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a + b)^2 / 2 + (a + b) / 4 ≥ a * Real.sqrt b + b * Real.sqrt a := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1548_154895


namespace NUMINAMATH_CALUDE_greatest_common_measure_of_segments_l1548_154887

/-- The greatest common measure of two segments of lengths 19 cm and 190 cm is 19 cm, not 1 cm -/
theorem greatest_common_measure_of_segments (segment1 : ℕ) (segment2 : ℕ) 
  (h1 : segment1 = 19) (h2 : segment2 = 190) :
  Nat.gcd segment1 segment2 = 19 ∧ Nat.gcd segment1 segment2 ≠ 1 := by
  sorry

end NUMINAMATH_CALUDE_greatest_common_measure_of_segments_l1548_154887


namespace NUMINAMATH_CALUDE_bananas_profit_theorem_l1548_154820

/-- The number of pounds of bananas purchased by the grocer -/
def bananas_purchased : ℝ := 84

/-- The purchase price in dollars for 3 pounds of bananas -/
def purchase_price : ℝ := 0.50

/-- The selling price in dollars for 4 pounds of bananas -/
def selling_price : ℝ := 1.00

/-- The total profit in dollars -/
def total_profit : ℝ := 7.00

/-- Theorem stating that the number of pounds of bananas purchased is correct -/
theorem bananas_profit_theorem :
  bananas_purchased * (selling_price / 4 - purchase_price / 3) = total_profit :=
by sorry

end NUMINAMATH_CALUDE_bananas_profit_theorem_l1548_154820


namespace NUMINAMATH_CALUDE_calculate_interest_rate_l1548_154803

/-- Given a sum of money invested at simple interest, this theorem proves
    that if the interest earned is a certain amount more than what would
    be earned at a reference rate, then the actual interest rate can be
    calculated. -/
theorem calculate_interest_rate 
  (principal : ℝ) 
  (time : ℝ) 
  (reference_rate : ℝ) 
  (interest_difference : ℝ) 
  (h1 : principal = 4200)
  (h2 : time = 2)
  (h3 : reference_rate = 0.12)
  (h4 : interest_difference = 504)
  : (principal * time * reference_rate + interest_difference) / (principal * time) = 0.18 := by
  sorry

end NUMINAMATH_CALUDE_calculate_interest_rate_l1548_154803


namespace NUMINAMATH_CALUDE_sufficient_condition_for_x_squared_minus_a_nonnegative_l1548_154800

theorem sufficient_condition_for_x_squared_minus_a_nonnegative 
  (a : ℝ) : 
  (∀ x : ℝ, x ∈ Set.Icc (-1) 2 → x^2 - a ≥ 0) ↔ 
  (a ≤ -1 ∧ ∃ b : ℝ, b > -1 ∧ ∀ x : ℝ, x ∈ Set.Icc (-1) 2 → x^2 - b ≥ 0) :=
sorry

end NUMINAMATH_CALUDE_sufficient_condition_for_x_squared_minus_a_nonnegative_l1548_154800


namespace NUMINAMATH_CALUDE_efficient_coefficient_computation_l1548_154828

/-- Represents a method to compute polynomial coefficients -/
structure ComputationMethod where
  (compute : (ℝ → ℝ) → List ℝ)
  (addition_count : ℕ)
  (multiplication_count : ℕ)

/-- A 6th degree polynomial -/
def Polynomial6 (a₁ a₂ a₃ a₄ a₅ a₆ : ℝ) : ℝ → ℝ :=
  fun x ↦ x^6 + a₁*x^5 + a₂*x^4 + a₃*x^3 + a₄*x^2 + a₅*x + a₆

/-- Theorem: There exists a method to compute coefficients of a 6th degree polynomial
    using its roots with no more than 15 additions and 15 multiplications -/
theorem efficient_coefficient_computation :
  ∃ (method : ComputationMethod),
    (∀ (p : ℝ → ℝ) (r₁ r₂ r₃ r₄ r₅ r₆ : ℝ),
      (∀ x, p x = (x + r₁) * (x + r₂) * (x + r₃) * (x + r₄) * (x + r₅) * (x + r₆)) →
      ∃ (a₁ a₂ a₃ a₄ a₅ a₆ : ℝ),
        p = Polynomial6 a₁ a₂ a₃ a₄ a₅ a₆ ∧
        method.compute p = [a₁, a₂, a₃, a₄, a₅, a₆]) ∧
    method.addition_count ≤ 15 ∧
    method.multiplication_count ≤ 15 :=
by sorry


end NUMINAMATH_CALUDE_efficient_coefficient_computation_l1548_154828


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l1548_154848

-- Define the sets M and N
def M : Set ℝ := {x | -2 ≤ x ∧ x < 2}
def N : Set ℝ := {x | x ≥ -2}

-- The theorem to prove
theorem intersection_of_M_and_N :
  M ∩ N = {x | -2 ≤ x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l1548_154848


namespace NUMINAMATH_CALUDE_ab_range_l1548_154886

-- Define the function f
def f (x : ℝ) : ℝ := |2 - x^2|

-- State the theorem
theorem ab_range (a b : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : f a = f b) :
  0 < a * b ∧ a * b < 2 := by
  sorry

end NUMINAMATH_CALUDE_ab_range_l1548_154886


namespace NUMINAMATH_CALUDE_deductive_reasoning_not_always_correct_l1548_154860

/-- Represents a deductive argument --/
structure DeductiveArgument where
  premises : List Prop
  conclusion : Prop

/-- Represents the form of a deductive argument --/
structure DeductiveForm where
  form : DeductiveArgument → Prop

/-- Defines when a deductive argument conforms to a deductive form --/
def conformsToForm (arg : DeductiveArgument) (form : DeductiveForm) : Prop :=
  form.form arg

/-- Defines when a deductive argument is valid --/
def isValid (arg : DeductiveArgument) : Prop :=
  ∀ (form : DeductiveForm), conformsToForm arg form → arg.conclusion

/-- Theorem: A deductive argument that conforms to a deductive form is not always valid --/
theorem deductive_reasoning_not_always_correct :
  ∃ (arg : DeductiveArgument) (form : DeductiveForm),
    conformsToForm arg form ∧ ¬isValid arg := by
  sorry

end NUMINAMATH_CALUDE_deductive_reasoning_not_always_correct_l1548_154860


namespace NUMINAMATH_CALUDE_division_problem_l1548_154822

theorem division_problem (dividend : ℕ) (quotient : ℕ) (divisor : ℕ) : 
  dividend = 64 → quotient = 8 → dividend = divisor * quotient → divisor = 8 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l1548_154822


namespace NUMINAMATH_CALUDE_power_ratio_equals_nine_l1548_154856

/-- Given real numbers a and b satisfying the specified conditions, 
    prove that 3^a / b^3 = 9 -/
theorem power_ratio_equals_nine 
  (a b : ℝ) 
  (h1 : 3^(a-2) + a = 1/2) 
  (h2 : (1/3)*b^3 + Real.log b / Real.log 3 = -1/2) : 
  3^a / b^3 = 9 := by
  sorry

end NUMINAMATH_CALUDE_power_ratio_equals_nine_l1548_154856


namespace NUMINAMATH_CALUDE_difference_multiple_of_nine_l1548_154845

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def reverse_digits (n : ℕ) : ℕ :=
  (n % 10) * 10 + (n / 10)

theorem difference_multiple_of_nine (q r : ℕ) :
  is_two_digit q ∧ 
  is_two_digit r ∧ 
  r = reverse_digits q ∧
  (∀ x y : ℕ, is_two_digit x ∧ is_two_digit y ∧ y = reverse_digits x → x - y ≤ 27) →
  ∃ k : ℕ, q - r = 9 * k ∨ r - q = 9 * k :=
sorry

end NUMINAMATH_CALUDE_difference_multiple_of_nine_l1548_154845


namespace NUMINAMATH_CALUDE_cubic_roots_sum_l1548_154897

theorem cubic_roots_sum (a b c : ℝ) : 
  a^3 - 6*a^2 + 11*a - 6 = 0 →
  b^3 - 6*b^2 + 11*b - 6 = 0 →
  c^3 - 6*c^2 + 11*c - 6 = 0 →
  a * b / c + b * c / a + c * a / b = 49 / 6 := by
sorry

end NUMINAMATH_CALUDE_cubic_roots_sum_l1548_154897


namespace NUMINAMATH_CALUDE_quadratic_t_range_l1548_154889

/-- Represents a quadratic function of the form ax² + bx - 2 --/
structure QuadraticFunction where
  a : ℝ
  b : ℝ

/-- Theorem statement for the range of t in the given quadratic equation --/
theorem quadratic_t_range (f : QuadraticFunction) 
  (h1 : f.a * (-1)^2 + f.b * (-1) - 2 = 0)  -- -1 is a root
  (h2 : 0 < -f.b / (2 * f.a))  -- vertex x-coordinate is positive (4th quadrant)
  (h3 : 0 < f.a)  -- parabola opens upward (4th quadrant)
  : -2 < 3 * f.a + f.b ∧ 3 * f.a + f.b < 6 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_t_range_l1548_154889


namespace NUMINAMATH_CALUDE_green_blue_difference_l1548_154866

/-- Represents a hexagonal figure with blue and green tiles -/
structure HexFigure where
  blue_tiles : ℕ
  green_tiles : ℕ

/-- Calculates the number of tiles needed for a two-layer border of a hexagon -/
def border_tiles : ℕ := 6 * 6

/-- Represents the new figure after adding a border -/
def new_figure (fig : HexFigure) : HexFigure :=
  { blue_tiles := fig.blue_tiles,
    green_tiles := fig.green_tiles + border_tiles }

/-- The main theorem to prove -/
theorem green_blue_difference (fig : HexFigure) 
  (h1 : fig.blue_tiles = 20) 
  (h2 : fig.green_tiles = 8) : 
  (new_figure fig).green_tiles - (new_figure fig).blue_tiles = 24 := by
  sorry

#check green_blue_difference

end NUMINAMATH_CALUDE_green_blue_difference_l1548_154866


namespace NUMINAMATH_CALUDE_people_studying_cooking_and_weaving_l1548_154870

/-- Represents the number of people in various curriculum combinations -/
structure CurriculumParticipation where
  yoga : ℕ
  cooking : ℕ
  weaving : ℕ
  cookingOnly : ℕ
  cookingAndYoga : ℕ
  allCurriculums : ℕ

/-- Theorem stating the number of people studying both cooking and weaving -/
theorem people_studying_cooking_and_weaving 
  (cp : CurriculumParticipation)
  (h1 : cp.yoga = 35)
  (h2 : cp.cooking = 20)
  (h3 : cp.weaving = 15)
  (h4 : cp.cookingOnly = 7)
  (h5 : cp.cookingAndYoga = 5)
  (h6 : cp.allCurriculums = 3) :
  ∃ n : ℕ, n = cp.cooking - cp.cookingOnly - (cp.cookingAndYoga - cp.allCurriculums) - cp.allCurriculums ∧ n = 8 := by
  sorry

#check people_studying_cooking_and_weaving

end NUMINAMATH_CALUDE_people_studying_cooking_and_weaving_l1548_154870


namespace NUMINAMATH_CALUDE_min_honey_amount_l1548_154840

theorem min_honey_amount (o h : ℝ) : 
  (o ≥ 8 + h / 3 ∧ o ≤ 3 * h) → h ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_min_honey_amount_l1548_154840


namespace NUMINAMATH_CALUDE_james_arthur_muffin_ratio_muffin_baking_problem_l1548_154865

theorem james_arthur_muffin_ratio : ℕ → ℕ → ℕ
  | arthur_muffins, james_muffins =>
    james_muffins / arthur_muffins

theorem muffin_baking_problem (arthur_muffins james_muffins : ℕ) 
  (h1 : arthur_muffins = 115)
  (h2 : james_muffins = 1380) :
  james_arthur_muffin_ratio arthur_muffins james_muffins = 12 := by
  sorry

end NUMINAMATH_CALUDE_james_arthur_muffin_ratio_muffin_baking_problem_l1548_154865


namespace NUMINAMATH_CALUDE_target_hit_probability_l1548_154819

theorem target_hit_probability 
  (prob_A : ℝ) 
  (prob_B : ℝ) 
  (h1 : prob_A = 1/2) 
  (h2 : prob_B = 1/3) : 
  1 - (1 - prob_A) * (1 - prob_B) = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_target_hit_probability_l1548_154819


namespace NUMINAMATH_CALUDE_complex_magnitude_equality_l1548_154849

theorem complex_magnitude_equality (t : ℝ) (h : t > 0) :
  Complex.abs (-3 + t * Complex.I) = 3 * Real.sqrt 5 → t = 6 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_equality_l1548_154849


namespace NUMINAMATH_CALUDE_bus_stop_walking_time_l1548_154891

/-- The time taken to walk to the bus stop at the usual speed, given that walking at 4/5 of the usual speed results in arriving 9 minutes later than normal, is 36 minutes. -/
theorem bus_stop_walking_time : ∀ T : ℝ, T > 0 → (5 / 4 = (T + 9) / T) → T = 36 := by
  sorry

end NUMINAMATH_CALUDE_bus_stop_walking_time_l1548_154891


namespace NUMINAMATH_CALUDE_only_D_satisfies_all_preferences_l1548_154808

-- Define the set of movies
inductive Movie : Type
  | A | B | C | D | E

-- Define the preferences of each person
def xiao_zhao_preference (m : Movie) : Prop := m ≠ Movie.B
def xiao_zhang_preference (m : Movie) : Prop := m = Movie.B ∨ m = Movie.C ∨ m = Movie.D ∨ m = Movie.E
def xiao_li_preference (m : Movie) : Prop := m ≠ Movie.C
def xiao_liu_preference (m : Movie) : Prop := m ≠ Movie.E

-- Define a function that checks if a movie satisfies all preferences
def satisfies_all_preferences (m : Movie) : Prop :=
  xiao_zhao_preference m ∧
  xiao_zhang_preference m ∧
  xiao_li_preference m ∧
  xiao_liu_preference m

-- Theorem: D is the only movie that satisfies all preferences
theorem only_D_satisfies_all_preferences :
  ∀ m : Movie, satisfies_all_preferences m ↔ m = Movie.D :=
by sorry


end NUMINAMATH_CALUDE_only_D_satisfies_all_preferences_l1548_154808


namespace NUMINAMATH_CALUDE_fantasia_license_plates_l1548_154827

/-- The number of letters in the alphabet used for license plates. -/
def alphabet_size : ℕ := 26

/-- The number of digits used for license plates. -/
def digit_size : ℕ := 10

/-- The number of letter positions in a license plate. -/
def letter_positions : ℕ := 3

/-- The number of digit positions in a license plate. -/
def digit_positions : ℕ := 4

/-- The total number of possible valid license plates in Fantasia. -/
def total_license_plates : ℕ := alphabet_size ^ letter_positions * digit_size ^ digit_positions

theorem fantasia_license_plates :
  total_license_plates = 175760000 := by
  sorry

end NUMINAMATH_CALUDE_fantasia_license_plates_l1548_154827


namespace NUMINAMATH_CALUDE_fertilizer_weight_calculation_l1548_154804

/-- Calculates the total weight of fertilizers applied to a given area -/
theorem fertilizer_weight_calculation 
  (field_area : ℝ) 
  (fertilizer_a_rate : ℝ) 
  (fertilizer_a_area : ℝ) 
  (fertilizer_b_rate : ℝ) 
  (fertilizer_b_area : ℝ) 
  (area_to_fertilize : ℝ) : 
  field_area = 10800 ∧ 
  fertilizer_a_rate = 150 ∧ 
  fertilizer_a_area = 3000 ∧ 
  fertilizer_b_rate = 180 ∧ 
  fertilizer_b_area = 4000 ∧ 
  area_to_fertilize = 3600 → 
  (fertilizer_a_rate * area_to_fertilize / fertilizer_a_area) + 
  (fertilizer_b_rate * area_to_fertilize / fertilizer_b_area) = 342 := by
  sorry

#check fertilizer_weight_calculation

end NUMINAMATH_CALUDE_fertilizer_weight_calculation_l1548_154804


namespace NUMINAMATH_CALUDE_jackie_apples_l1548_154885

/-- 
Given that Adam has 10 apples and 8 more apples than Jackie,
prove that Jackie has 2 apples.
-/
theorem jackie_apples (adam_apples : ℕ) (difference : ℕ) (jackie_apples : ℕ)
  (h1 : adam_apples = 10)
  (h2 : adam_apples = jackie_apples + difference)
  (h3 : difference = 8) :
  jackie_apples = 2 := by
  sorry

end NUMINAMATH_CALUDE_jackie_apples_l1548_154885


namespace NUMINAMATH_CALUDE_probability_specific_arrangement_l1548_154877

theorem probability_specific_arrangement (n : ℕ) (k : ℕ) (h1 : n = 7) (h2 : k = 4) :
  (1 : ℚ) / (n.choose k) = (1 : ℚ) / 35 :=
sorry

end NUMINAMATH_CALUDE_probability_specific_arrangement_l1548_154877


namespace NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_9_with_specific_digits_l1548_154838

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def is_divisible_by_9 (n : ℕ) : Prop := n % 9 = 0

def has_odd_units_and_thousands (n : ℕ) : Prop :=
  n % 2 = 1 ∧ (n / 1000) % 2 = 1

def has_even_tens_and_hundreds (n : ℕ) : Prop :=
  ((n / 10) % 10) % 2 = 0 ∧ ((n / 100) % 10) % 2 = 0

theorem smallest_four_digit_divisible_by_9_with_specific_digits : 
  ∀ n : ℕ, is_four_digit n → 
  is_divisible_by_9 n → 
  has_odd_units_and_thousands n → 
  has_even_tens_and_hundreds n → 
  3609 ≤ n :=
sorry

end NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_9_with_specific_digits_l1548_154838


namespace NUMINAMATH_CALUDE_sum_of_three_numbers_l1548_154893

theorem sum_of_three_numbers (a b c : ℝ) (h1 : a^2 + b^2 + c^2 = 156) (h2 : a*b + b*c + a*c = 50) :
  a + b + c = 16 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_three_numbers_l1548_154893


namespace NUMINAMATH_CALUDE_absolute_value_sum_range_l1548_154857

theorem absolute_value_sum_range : 
  ∃ (min_value : ℝ), 
    (∀ x : ℝ, |x - 1| + |x - 2| ≥ min_value) ∧ 
    (∃ x : ℝ, |x - 1| + |x - 2| = min_value) ∧
    (∀ a : ℝ, (∃ x : ℝ, |x - 1| + |x - 2| ≤ a) ↔ a ≥ min_value) ∧
    min_value = 1 :=
by sorry

end NUMINAMATH_CALUDE_absolute_value_sum_range_l1548_154857


namespace NUMINAMATH_CALUDE_days_to_clear_land_l1548_154829

/-- Represents the number of feet in a yard -/
def feet_per_yard : ℝ := 3

/-- Represents the length of the land in feet -/
def land_length_feet : ℝ := 900

/-- Represents the width of the land in feet -/
def land_width_feet : ℝ := 200

/-- Represents the number of rabbits -/
def num_rabbits : ℕ := 100

/-- Represents the area one rabbit can clear per day in square yards -/
def area_per_rabbit_per_day : ℝ := 10

/-- Theorem stating the number of days needed to clear the land -/
theorem days_to_clear_land : 
  ⌈(land_length_feet / feet_per_yard) * (land_width_feet / feet_per_yard) / 
   (num_rabbits : ℝ) / area_per_rabbit_per_day⌉ = 21 := by sorry

end NUMINAMATH_CALUDE_days_to_clear_land_l1548_154829


namespace NUMINAMATH_CALUDE_cos_2alpha_value_l1548_154842

theorem cos_2alpha_value (α : Real) 
  (h1 : α ∈ Set.Ioo (π / 2) π) 
  (h2 : Real.sin α + Real.cos α = 1 / 5) : 
  Real.cos (2 * α) = -7 / 25 := by
  sorry

end NUMINAMATH_CALUDE_cos_2alpha_value_l1548_154842


namespace NUMINAMATH_CALUDE_orange_cost_l1548_154813

theorem orange_cost (cost_three_dozen : ℝ) (h : cost_three_dozen = 28.20) :
  let cost_per_dozen : ℝ := cost_three_dozen / 3
  let cost_five_dozen : ℝ := 5 * cost_per_dozen
  cost_five_dozen = 47.00 := by
sorry

end NUMINAMATH_CALUDE_orange_cost_l1548_154813


namespace NUMINAMATH_CALUDE_train_length_l1548_154846

/-- Given a train that crosses a platform of length 350 meters in 39 seconds
    and crosses a signal pole in 18 seconds, the length of the train is 300 meters. -/
theorem train_length (platform_length : ℝ) (platform_time : ℝ) (pole_time : ℝ)
    (h1 : platform_length = 350)
    (h2 : platform_time = 39)
    (h3 : pole_time = 18) :
    (platform_length * pole_time) / (platform_time - pole_time) = 300 :=
by sorry

end NUMINAMATH_CALUDE_train_length_l1548_154846


namespace NUMINAMATH_CALUDE_meaningful_fraction_l1548_154859

theorem meaningful_fraction (x : ℝ) : 
  (∃ y : ℝ, y = x / (x - 4)) ↔ x ≠ 4 := by sorry

end NUMINAMATH_CALUDE_meaningful_fraction_l1548_154859


namespace NUMINAMATH_CALUDE_rhombus_constructible_l1548_154839

/-- Represents a rhombus in 2D space -/
structure Rhombus where
  /-- Side length of the rhombus -/
  side : ℝ
  /-- Difference between the two diagonals -/
  diag_diff : ℝ
  /-- Assumption that side length is positive -/
  side_pos : side > 0
  /-- Assumption that diagonal difference is non-negative and less than twice the side length -/
  diag_diff_valid : 0 ≤ diag_diff ∧ diag_diff < 2 * side

/-- Theorem stating that a rhombus can be constructed given a side length and diagonal difference -/
theorem rhombus_constructible (a : ℝ) (d : ℝ) (h1 : a > 0) (h2 : 0 ≤ d ∧ d < 2 * a) :
  ∃ (r : Rhombus), r.side = a ∧ r.diag_diff = d :=
sorry

end NUMINAMATH_CALUDE_rhombus_constructible_l1548_154839


namespace NUMINAMATH_CALUDE_fraction_equivalence_l1548_154863

theorem fraction_equivalence : (3 : ℚ) / 7 = 27 / 63 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equivalence_l1548_154863


namespace NUMINAMATH_CALUDE_sqrt_mixed_number_simplification_l1548_154844

theorem sqrt_mixed_number_simplification :
  Real.sqrt (12 + 1/9) = Real.sqrt 109 / 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_mixed_number_simplification_l1548_154844


namespace NUMINAMATH_CALUDE_square_area_error_l1548_154811

theorem square_area_error (s : ℝ) (h : s > 0) :
  let measured_side := s * (1 + 0.02)
  let actual_area := s^2
  let calculated_area := measured_side^2
  (calculated_area - actual_area) / actual_area * 100 = 4.04 := by
sorry

end NUMINAMATH_CALUDE_square_area_error_l1548_154811


namespace NUMINAMATH_CALUDE_max_markers_is_16_l1548_154823

-- Define the prices and quantities for each option
def single_marker_price : ℕ := 2
def pack4_price : ℕ := 6
def pack8_price : ℕ := 10
def pack4_quantity : ℕ := 4
def pack8_quantity : ℕ := 8

-- Define Lisa's budget
def budget : ℕ := 20

-- Define a function to calculate the number of markers for a given combination of purchases
def markers_bought (singles pack4s pack8s : ℕ) : ℕ :=
  singles + pack4s * pack4_quantity + pack8s * pack8_quantity

-- Define a function to calculate the total cost of a combination of purchases
def total_cost (singles pack4s pack8s : ℕ) : ℕ :=
  singles * single_marker_price + pack4s * pack4_price + pack8s * pack8_price

-- Theorem: The maximum number of markers that can be bought with the given budget is 16
theorem max_markers_is_16 :
  ∀ (singles pack4s pack8s : ℕ),
    total_cost singles pack4s pack8s ≤ budget →
    markers_bought singles pack4s pack8s ≤ 16 :=
by sorry

end NUMINAMATH_CALUDE_max_markers_is_16_l1548_154823


namespace NUMINAMATH_CALUDE_gcd_eight_factorial_six_factorial_l1548_154809

-- Define factorial function
def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

-- State the theorem
theorem gcd_eight_factorial_six_factorial :
  Nat.gcd (factorial 8) (factorial 6) = 720 := by
  sorry

end NUMINAMATH_CALUDE_gcd_eight_factorial_six_factorial_l1548_154809


namespace NUMINAMATH_CALUDE_optimal_rate_l1548_154873

/- Define the initial conditions -/
def totalRooms : ℕ := 100
def initialRate : ℕ := 400
def initialOccupancy : ℕ := 50
def rateReduction : ℕ := 20
def occupancyIncrease : ℕ := 5

/- Define the revenue function -/
def revenue (rate : ℕ) : ℕ :=
  let occupancy := initialOccupancy + ((initialRate - rate) / rateReduction) * occupancyIncrease
  rate * occupancy

/- Theorem statement -/
theorem optimal_rate :
  ∀ (rate : ℕ), rate ≤ initialRate → revenue 300 ≥ revenue rate :=
sorry

end NUMINAMATH_CALUDE_optimal_rate_l1548_154873


namespace NUMINAMATH_CALUDE_remainder_proof_l1548_154884

theorem remainder_proof (y : Nat) (h1 : y > 0) (h2 : (7 * y) % 29 = 1) :
  (8 + y) % 29 = 4 := by
  sorry

end NUMINAMATH_CALUDE_remainder_proof_l1548_154884


namespace NUMINAMATH_CALUDE_loom_weaving_rate_l1548_154837

/-- The rate at which an industrial loom weaves cloth, given the time and length of cloth woven. -/
theorem loom_weaving_rate (time : ℝ) (length : ℝ) (h : time = 195.3125 ∧ length = 25) :
  length / time = 0.128 := by
  sorry

end NUMINAMATH_CALUDE_loom_weaving_rate_l1548_154837


namespace NUMINAMATH_CALUDE_g_difference_l1548_154832

/-- Given g(x) = 3x^2 + 4x + 5, prove that g(x + h) - g(x) = h(6x + 3h + 4) for all real x and h. -/
theorem g_difference (x h : ℝ) : 
  let g : ℝ → ℝ := λ t ↦ 3 * t^2 + 4 * t + 5
  g (x + h) - g x = h * (6 * x + 3 * h + 4) := by
  sorry

end NUMINAMATH_CALUDE_g_difference_l1548_154832


namespace NUMINAMATH_CALUDE_first_month_sale_l1548_154816

def average_sale : ℝ := 6000
def num_months : ℕ := 5
def sale_2 : ℝ := 5660
def sale_3 : ℝ := 6200
def sale_4 : ℝ := 6350
def sale_5 : ℝ := 6500
def sale_6 : ℝ := 5870

theorem first_month_sale (sale_1 : ℝ) :
  (sale_1 + sale_2 + sale_3 + sale_4 + sale_5) / num_months = average_sale →
  sale_1 = 5290 := by
sorry

end NUMINAMATH_CALUDE_first_month_sale_l1548_154816


namespace NUMINAMATH_CALUDE_solve_for_q_l1548_154821

theorem solve_for_q (p q : ℝ) (hp : p > 1) (hq : q > 1) 
  (h1 : 1/p + 1/q = 1) (h2 : p * q = 9) : q = (9 + 3 * Real.sqrt 5) / 2 :=
by sorry

end NUMINAMATH_CALUDE_solve_for_q_l1548_154821


namespace NUMINAMATH_CALUDE_cider_pints_is_180_l1548_154879

/-- Represents the number of pints of cider that can be made given the following conditions:
  * 20 golden delicious, 40 pink lady, and 30 granny smith apples make one pint of cider
  * Each farmhand can pick 120 golden delicious, 240 pink lady, and 180 granny smith apples per hour
  * There are 6 farmhands working 5 hours
  * The ratio of golden delicious : pink lady : granny smith apples gathered is 1:2:1.5
-/
def cider_pints : ℕ :=
  let golden_per_pint : ℕ := 20
  let pink_per_pint : ℕ := 40
  let granny_per_pint : ℕ := 30
  let golden_per_hour : ℕ := 120
  let pink_per_hour : ℕ := 240
  let granny_per_hour : ℕ := 180
  let farmhands : ℕ := 6
  let hours : ℕ := 5
  let golden_total : ℕ := golden_per_hour * hours * farmhands
  let pink_total : ℕ := pink_per_hour * hours * farmhands
  let granny_total : ℕ := granny_per_hour * hours * farmhands
  golden_total / golden_per_pint

theorem cider_pints_is_180 : cider_pints = 180 := by
  sorry

end NUMINAMATH_CALUDE_cider_pints_is_180_l1548_154879


namespace NUMINAMATH_CALUDE_initial_bottle_caps_l1548_154876

theorem initial_bottle_caps (initial : ℕ) (added : ℕ) (total : ℕ) : 
  added = 7 → total = 14 → total = initial + added → initial = 7 := by sorry

end NUMINAMATH_CALUDE_initial_bottle_caps_l1548_154876


namespace NUMINAMATH_CALUDE_line_not_in_second_quadrant_l1548_154854

-- Define the line l with equation x - y - a² = 0
def line_equation (x y a : ℝ) : Prop := x - y - a^2 = 0

-- Define the second quadrant
def second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0

-- Theorem statement
theorem line_not_in_second_quadrant (a : ℝ) (h : a ≠ 0) :
  ∀ x y : ℝ, line_equation x y a → ¬ second_quadrant x y :=
sorry

end NUMINAMATH_CALUDE_line_not_in_second_quadrant_l1548_154854


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l1548_154814

theorem min_value_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (1 / a + 1 / (b + 1) ≥ 2) ∧
  (1 / a + 1 / (b + 1) = 2 ↔ a = 1 / 2 ∧ b = 1 / 2) := by
sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l1548_154814


namespace NUMINAMATH_CALUDE_five_lines_max_sections_l1548_154892

/-- The maximum number of sections created by drawing n line segments through a rectangle -/
def max_sections (n : ℕ) : ℕ := n * (n + 1) / 2 + 1

/-- Theorem: The maximum number of sections created by drawing 5 line segments through a rectangle is 16 -/
theorem five_lines_max_sections : max_sections 5 = 16 := by
  sorry

end NUMINAMATH_CALUDE_five_lines_max_sections_l1548_154892


namespace NUMINAMATH_CALUDE_x_squared_plus_inverse_squared_l1548_154869

theorem x_squared_plus_inverse_squared (x : ℝ) : x^2 - x - 1 = 0 → x^2 + 1/x^2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_x_squared_plus_inverse_squared_l1548_154869


namespace NUMINAMATH_CALUDE_jos_number_l1548_154817

theorem jos_number (n k l : ℕ) : 
  0 < n ∧ n < 150 ∧ n = 9 * k - 2 ∧ n = 8 * l - 4 →
  n ≤ 132 ∧ (∃ (k' l' : ℕ), 132 = 9 * k' - 2 ∧ 132 = 8 * l' - 4) :=
by sorry

end NUMINAMATH_CALUDE_jos_number_l1548_154817


namespace NUMINAMATH_CALUDE_pens_distribution_l1548_154871

/-- The number of pens in each pack -/
def pens_per_pack : ℕ := 3

/-- The number of packs Kendra has -/
def kendra_packs : ℕ := 4

/-- The number of packs Tony has -/
def tony_packs : ℕ := 2

/-- The number of pens Kendra and Tony each keep for themselves -/
def pens_kept_each : ℕ := 2

/-- The total number of friends who will receive pens -/
def friends_receiving_pens : ℕ := 
  kendra_packs * pens_per_pack + tony_packs * pens_per_pack - 2 * pens_kept_each

theorem pens_distribution :
  friends_receiving_pens = 14 := by
  sorry

end NUMINAMATH_CALUDE_pens_distribution_l1548_154871


namespace NUMINAMATH_CALUDE_parallel_line_intersection_parallel_planes_intersection_l1548_154825

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the basic relations
variable (parallel : Line → Line → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (parallel_plane : Plane → Plane → Prop)
variable (subset : Line → Plane → Prop)
variable (intersect : Plane → Plane → Line)

-- Theorem 1
theorem parallel_line_intersection 
  (l m : Line) (α β : Plane) :
  parallel_line_plane l α →
  subset l β →
  intersect α β = m →
  parallel l m := by sorry

-- Theorem 2
theorem parallel_planes_intersection 
  (l m : Line) (α β γ : Plane) :
  parallel_plane α β →
  intersect α γ = l →
  intersect β γ = m →
  parallel l m := by sorry

end NUMINAMATH_CALUDE_parallel_line_intersection_parallel_planes_intersection_l1548_154825


namespace NUMINAMATH_CALUDE_least_number_for_divisibility_l1548_154847

theorem least_number_for_divisibility (n m : ℕ) (h : n = 1055 ∧ m = 23) :
  ∃ (x : ℕ), (n + x) % m = 0 ∧ ∀ (y : ℕ), y < x → (n + y) % m ≠ 0 ∧ x = 3 :=
sorry

end NUMINAMATH_CALUDE_least_number_for_divisibility_l1548_154847


namespace NUMINAMATH_CALUDE_equation_three_solutions_l1548_154801

theorem equation_three_solutions :
  ∃ (s : Finset ℝ), (s.card = 3) ∧ 
  (∀ x ∈ s, (x^2 - 6*x + 9) / (x - 1) - (3 - x) / (x^2 - 1) = 0) ∧
  (∀ y : ℝ, (y^2 - 6*y + 9) / (y - 1) - (3 - y) / (y^2 - 1) = 0 → y ∈ s) := by
  sorry

end NUMINAMATH_CALUDE_equation_three_solutions_l1548_154801


namespace NUMINAMATH_CALUDE_term_2500_mod_7_l1548_154830

/-- Defines the sequence where the (2n)th positive integer appears n times
    and the (2n-1)th positive integer appears n+1 times -/
def sequence_term (k : ℕ) : ℕ := sorry

/-- The 2500th term of the sequence -/
def term_2500 : ℕ := sequence_term 2500

theorem term_2500_mod_7 : term_2500 % 7 = 1 := by sorry

end NUMINAMATH_CALUDE_term_2500_mod_7_l1548_154830


namespace NUMINAMATH_CALUDE_simplify_logarithmic_expression_l1548_154855

theorem simplify_logarithmic_expression :
  let x := 1 / (Real.log 3 / Real.log 6 + 1) +
           1 / (Real.log 7 / Real.log 15 + 1) +
           1 / (Real.log 4 / Real.log 12 + 1)
  x = -Real.log 84 / Real.log 10 := by
  sorry

end NUMINAMATH_CALUDE_simplify_logarithmic_expression_l1548_154855
