import Mathlib

namespace circle_center_radius_sum_l2755_275513

theorem circle_center_radius_sum :
  ∀ (a b r : ℝ),
  (∀ (x y : ℝ), x^2 - 16*x + y^2 + 6*y = 20 ↔ (x - a)^2 + (y - b)^2 = r^2) →
  a + b + r = 5 + Real.sqrt 93 :=
by sorry

end circle_center_radius_sum_l2755_275513


namespace arithmetic_sequence_common_difference_l2755_275557

/-- An arithmetic sequence with given terms -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  a2_eq_9 : a 2 = 9
  a5_eq_33 : a 5 = 33

/-- The common difference of an arithmetic sequence is 8 given the conditions -/
theorem arithmetic_sequence_common_difference
  (seq : ArithmeticSequence) :
  ∃ d : ℝ, (∀ n, seq.a (n + 1) - seq.a n = d) ∧ d = 8 := by
  sorry

end arithmetic_sequence_common_difference_l2755_275557


namespace common_chord_equation_l2755_275531

/-- The equation of the line where the common chord of two circles lies -/
theorem common_chord_equation (x y : ℝ) :
  (x^2 + y^2 + 4*x - 6*y + 12 = 0) ∧ (x^2 + y^2 - 2*x - 14*y + 15 = 0) →
  (6*x + 8*y - 3 = 0) :=
by sorry

end common_chord_equation_l2755_275531


namespace acute_triangle_condition_l2755_275555

/-- A triangle is represented by its incircle radius and circumcircle radius -/
structure Triangle where
  r : ℝ  -- radius of the incircle
  R : ℝ  -- radius of the circumcircle

/-- A triangle is acute if all its angles are less than 90 degrees -/
def Triangle.isAcute (t : Triangle) : Prop :=
  sorry  -- definition of an acute triangle

/-- The main theorem: if R < r(√2 + 1), then the triangle is acute -/
theorem acute_triangle_condition (t : Triangle) 
  (h : t.R < t.r * (Real.sqrt 2 + 1)) : t.isAcute :=
sorry

end acute_triangle_condition_l2755_275555


namespace distance_AB_equals_5_l2755_275528

-- Define the line l₁
def line_l₁ (x y : ℝ) : Prop := y = Real.sqrt 3 * x

-- Define the curve C
def curve_C (x y φ : ℝ) : Prop :=
  x = 1 + Real.sqrt 3 * Real.cos φ ∧
  y = Real.sqrt 3 * Real.sin φ ∧
  0 ≤ φ ∧ φ ≤ Real.pi

-- Define the line l₂ in polar coordinates
def line_l₂ (ρ θ : ℝ) : Prop :=
  2 * ρ * Real.sin (θ + Real.pi / 3) + 3 * Real.sqrt 3 = 0

-- Define the intersection point A of l₁ and C
def point_A : ℝ × ℝ := sorry

-- Define the intersection point B of l₁ and l₂
def point_B : ℝ × ℝ := sorry

-- Theorem statement
theorem distance_AB_equals_5 :
  Real.sqrt ((point_A.1 - point_B.1)^2 + (point_A.2 - point_B.2)^2) = 5 := by sorry

end distance_AB_equals_5_l2755_275528


namespace existence_of_special_quadratic_l2755_275592

theorem existence_of_special_quadratic (n : ℕ) (h_odd : Odd n) (h_gt_one : n > 1) :
  ∃ (a b : ℕ), a > 0 ∧ b > 0 ∧
    (Nat.gcd a n = 1) ∧
    (Nat.gcd b n = 1) ∧
    (n ∣ (a^2 + b)) ∧
    (∀ x : ℕ, x ≥ 1 → ∃ p : ℕ, Prime p ∧ p ∣ ((x + a)^2 + b) ∧ ¬(p ∣ n)) := by
  sorry

end existence_of_special_quadratic_l2755_275592


namespace original_number_proof_l2755_275566

theorem original_number_proof (x : ℝ) : ((3 * x^2 + 8) * 2) / 4 = 56 → x = 2 * Real.sqrt 78 / 3 := by
  sorry

end original_number_proof_l2755_275566


namespace square_sum_and_product_l2755_275506

theorem square_sum_and_product (x y : ℝ) 
  (h1 : (x + y)^2 = 1) 
  (h2 : (x - y)^2 = 9) : 
  x^2 + y^2 = 5 ∧ x * y = -2 := by
  sorry

end square_sum_and_product_l2755_275506


namespace existence_of_n_l2755_275595

theorem existence_of_n (p : ℕ) (a k : ℕ+) (h_prime : Nat.Prime p) 
  (h_bound : p^(a : ℕ) < k ∧ k < 2 * p^(a : ℕ)) : 
  ∃ n : ℕ+, n < p^(2 * (a : ℕ)) ∧ 
    (Nat.choose n k : ZMod (p^(a : ℕ))) = n ∧ 
    (n : ZMod (p^(a : ℕ))) = k :=
sorry

end existence_of_n_l2755_275595


namespace dice_sum_multiple_of_5_prob_correct_l2755_275515

/-- The probability that the sum of n rolls of a 6-sided die is a multiple of 5 -/
def dice_sum_multiple_of_5_prob (n : ℕ) : ℚ :=
  if 5 ∣ n then
    (6^n + 4) / (5 * 6^n)
  else
    (6^n - 1) / (5 * 6^n)

/-- Theorem: The probability that the sum of n rolls of a 6-sided die is a multiple of 5
    is (6^n - 1) / (5 * 6^n) if 5 doesn't divide n, and (6^n + 4) / (5 * 6^n) if 5 divides n -/
theorem dice_sum_multiple_of_5_prob_correct (n : ℕ) :
  dice_sum_multiple_of_5_prob n =
    if 5 ∣ n then
      (6^n + 4) / (5 * 6^n)
    else
      (6^n - 1) / (5 * 6^n) := by
  sorry

end dice_sum_multiple_of_5_prob_correct_l2755_275515


namespace five_digit_number_theorem_l2755_275583

def is_valid_digit (d : ℕ) : Prop := d = 1 ∨ d = 2 ∨ d = 3 ∨ d = 6 ∨ d = 8

def are_distinct (p q r s t : ℕ) : Prop :=
  p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ p ≠ t ∧
  q ≠ r ∧ q ≠ s ∧ q ≠ t ∧
  r ≠ s ∧ r ≠ t ∧
  s ≠ t

theorem five_digit_number_theorem (p q r s t : ℕ) :
  is_valid_digit p ∧ is_valid_digit q ∧ is_valid_digit r ∧ is_valid_digit s ∧ is_valid_digit t ∧
  are_distinct p q r s t ∧
  (100 * p + 10 * q + r) % 6 = 0 ∧
  (100 * q + 10 * r + s) % 8 = 0 ∧
  (100 * r + 10 * s + t) % 3 = 0 →
  p = 2 := by
sorry

end five_digit_number_theorem_l2755_275583


namespace arithmetic_sequence_difference_l2755_275599

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_difference
  (a b : ℕ → ℝ)
  (ha : ArithmeticSequence a)
  (hb : ArithmeticSequence b)
  (ha1 : a 1 = 3)
  (hb1 : b 1 = -3)
  (h19 : a 19 - b 19 = 16) :
  a 10 - b 10 = 11 := by
sorry

end arithmetic_sequence_difference_l2755_275599


namespace first_player_win_prob_l2755_275522

/-- The probability of the first player getting heads -/
def p1 : ℚ := 1/3

/-- The probability of the second player getting heads -/
def p2 : ℚ := 2/5

/-- The game where two players flip coins alternately until one gets heads -/
def coin_flip_game (p1 p2 : ℚ) : Prop :=
  p1 > 0 ∧ p1 < 1 ∧ p2 > 0 ∧ p2 < 1

/-- The probability of the first player winning the game -/
noncomputable def win_prob (p1 p2 : ℚ) : ℚ :=
  p1 / (1 - (1 - p1) * (1 - p2))

/-- Theorem stating that the probability of the first player winning is 5/9 -/
theorem first_player_win_prob :
  coin_flip_game p1 p2 → win_prob p1 p2 = 5/9 := by
  sorry

end first_player_win_prob_l2755_275522


namespace complement_of_A_l2755_275590

-- Define the set A
def A : Set ℝ := {x | x^2 - 2*x - 3 > 0}

-- State the theorem
theorem complement_of_A : 
  (Set.univ \ A : Set ℝ) = Set.Icc (-1) 3 := by sorry

end complement_of_A_l2755_275590


namespace inequality_proof_l2755_275549

theorem inequality_proof (a b c : ℝ) 
  (ha : 0 < a ∧ a ≤ 1) 
  (hb : 0 < b ∧ b ≤ 1) 
  (hc : 0 < c ∧ c ≤ 1) 
  (h_sum : a^2 + b^2 + c^2 = 2) : 
  (1 - b^2) / a + (1 - c^2) / b + (1 - a^2) / c ≤ 5/4 := by
sorry

end inequality_proof_l2755_275549


namespace erased_grid_squares_l2755_275570

/-- Represents a square grid with erased line segments -/
structure ErasedSquareGrid :=
  (size : Nat)
  (erasedLines : Nat)

/-- Counts the number of squares of a given size in the grid -/
def countSquares (grid : ErasedSquareGrid) (squareSize : Nat) : Nat :=
  sorry

/-- Calculates the total number of squares of all sizes in the grid -/
def totalSquares (grid : ErasedSquareGrid) : Nat :=
  sorry

/-- The main theorem stating that a 4x4 grid with 2 erased lines has 22 squares -/
theorem erased_grid_squares :
  let grid : ErasedSquareGrid := ⟨4, 2⟩
  totalSquares grid = 22 :=
by sorry

end erased_grid_squares_l2755_275570


namespace pet_store_birds_count_l2755_275504

theorem pet_store_birds_count :
  let num_cages : ℕ := 8
  let parrots_per_cage : ℕ := 2
  let parakeets_per_cage : ℕ := 7
  let total_birds : ℕ := num_cages * (parrots_per_cage + parakeets_per_cage)
  total_birds = 72 :=
by
  sorry

end pet_store_birds_count_l2755_275504


namespace abc_sum_l2755_275572

theorem abc_sum (A B C : ℕ+) (h1 : Nat.gcd A.val (Nat.gcd B.val C.val) = 1)
  (h2 : (A : ℝ) * Real.log 5 / Real.log 500 + (B : ℝ) * Real.log 2 / Real.log 500 = C) :
  A + B + C = 6 := by
sorry

end abc_sum_l2755_275572


namespace tangent_slope_at_point_two_l2755_275526

/-- The curve function -/
def f (x : ℝ) : ℝ := x^2 + 3*x

/-- The derivative of the curve function -/
def f' (x : ℝ) : ℝ := 2*x + 3

theorem tangent_slope_at_point_two :
  f' 2 = 7 :=
sorry

end tangent_slope_at_point_two_l2755_275526


namespace f_sum_negative_l2755_275558

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the properties of f
axiom f_odd : ∀ x, f x + f (-x) = 0
axiom f_increasing_neg : ∀ x y, x < y → y ≤ 0 → f x < f y

-- Define the theorem
theorem f_sum_negative (x₁ x₂ : ℝ) 
  (h₁ : x₁ + x₂ < 0) (h₂ : x₁ * x₂ < 0) : 
  f x₁ + f x₂ < 0 := by sorry

end f_sum_negative_l2755_275558


namespace function_properties_l2755_275524

-- Define the function f
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^3 + x^2 + b * x

-- Define the function g
def g (a b : ℝ) (x : ℝ) : ℝ := f a b x + (3 * a * x^2 + 2 * x + b)

-- State the theorem
theorem function_properties (a b : ℝ) :
  (∀ x, g a b x = -g a b (-x)) →  -- g is an odd function
  (∃ c, ∀ x, f a b x = -1/3 * x^3 + x^2 + c) ∧
  (∀ x ∈ Set.Icc 1 2, g (-1/3) 0 x ≤ 4 * Real.sqrt 2 / 3) ∧
  (∀ x ∈ Set.Icc 1 2, g (-1/3) 0 x ≥ 4 / 3) ∧
  (∃ x ∈ Set.Icc 1 2, g (-1/3) 0 x = 4 * Real.sqrt 2 / 3) ∧
  (∃ x ∈ Set.Icc 1 2, g (-1/3) 0 x = 4 / 3) :=
by sorry

end function_properties_l2755_275524


namespace arithmetic_sequence_sum_l2755_275569

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- Sum function
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_formula : ∀ n, S n = n * (a 1 + a n) / 2

/-- Theorem: Given an arithmetic sequence with S_15 = 30 and a_7 = 1, then S_9 = -9 -/
theorem arithmetic_sequence_sum (seq : ArithmeticSequence)
  (h1 : seq.S 15 = 30)
  (h2 : seq.a 7 = 1) :
  seq.S 9 = -9 := by
  sorry


end arithmetic_sequence_sum_l2755_275569


namespace reciprocal_not_always_plus_minus_one_l2755_275574

theorem reciprocal_not_always_plus_minus_one : 
  ¬ (∀ x : ℝ, x ≠ 0 → (1 / x = 1 ∨ 1 / x = -1)) := by
  sorry

end reciprocal_not_always_plus_minus_one_l2755_275574


namespace equation_holds_for_all_y_l2755_275550

theorem equation_holds_for_all_y (x : ℚ) : 
  (∀ y : ℚ, 8 * x * y - 12 * y + 2 * x - 3 = 0) ↔ x = 3/2 := by
  sorry

end equation_holds_for_all_y_l2755_275550


namespace right_triangle_hypotenuse_l2755_275573

theorem right_triangle_hypotenuse (a b c : ℝ) :
  a = 12 ∧ b = 16 ∧ c^2 = a^2 + b^2 → c = 20 := by
  sorry

end right_triangle_hypotenuse_l2755_275573


namespace leftover_milk_proof_l2755_275525

/-- Represents the amount of milk used for each type of milkshake -/
structure MilkUsage where
  vanilla : ℕ
  chocolate : ℕ

/-- Represents the amount of ice cream used for each type of milkshake -/
structure IceCreamUsage where
  vanilla : ℕ
  chocolate : ℕ

/-- Represents the available ingredients -/
structure Ingredients where
  milk : ℕ
  vanilla_ice_cream : ℕ
  chocolate_ice_cream : ℕ

/-- Represents the number of milkshakes to make -/
structure Milkshakes where
  vanilla : ℕ
  chocolate : ℕ

def milk_usage : MilkUsage := ⟨4, 5⟩
def ice_cream_usage : IceCreamUsage := ⟨12, 10⟩
def available_ingredients : Ingredients := ⟨72, 96, 96⟩
def max_milkshakes : ℕ := 16

def valid_milkshake_count (m : Milkshakes) : Prop :=
  m.vanilla + m.chocolate ≤ max_milkshakes ∧
  2 * m.chocolate = m.vanilla

def enough_ingredients (m : Milkshakes) : Prop :=
  m.vanilla * milk_usage.vanilla + m.chocolate * milk_usage.chocolate ≤ available_ingredients.milk ∧
  m.vanilla * ice_cream_usage.vanilla ≤ available_ingredients.vanilla_ice_cream ∧
  m.chocolate * ice_cream_usage.chocolate ≤ available_ingredients.chocolate_ice_cream

def optimal_milkshakes : Milkshakes := ⟨10, 5⟩

theorem leftover_milk_proof :
  valid_milkshake_count optimal_milkshakes ∧
  enough_ingredients optimal_milkshakes ∧
  ∀ m : Milkshakes, valid_milkshake_count m → enough_ingredients m →
    m.vanilla + m.chocolate ≤ optimal_milkshakes.vanilla + optimal_milkshakes.chocolate →
  available_ingredients.milk - (optimal_milkshakes.vanilla * milk_usage.vanilla + optimal_milkshakes.chocolate * milk_usage.chocolate) = 7 :=
sorry

end leftover_milk_proof_l2755_275525


namespace circle_radius_from_triangle_l2755_275530

/-- Given a right-angled triangle with area 60 cm² and one side 15 cm that touches a circle,
    prove that the radius of the circle is 20 cm. -/
theorem circle_radius_from_triangle (triangle_area : ℝ) (triangle_side : ℝ) (circle_radius : ℝ) :
  triangle_area = 60 →
  triangle_side = 15 →
  -- Additional properties to define the relationship between the triangle and circle
  -- These are simplified representations of the problem conditions
  ∃ (triangle_height : ℝ) (triangle_hypotenuse : ℝ),
    triangle_area = (1/2) * triangle_side * triangle_height ∧
    triangle_hypotenuse^2 = triangle_side^2 + triangle_height^2 ∧
    circle_radius - triangle_height + circle_radius - triangle_side = triangle_hypotenuse →
  circle_radius = 20 :=
by sorry

end circle_radius_from_triangle_l2755_275530


namespace friends_bill_split_l2755_275585

/-- The cost each friend pays when splitting a bill equally -/
def split_bill (num_friends : ℕ) (item1_cost : ℚ) (item2_count : ℕ) (item2_cost : ℚ)
                (item3_count : ℕ) (item3_cost : ℚ) (item4_count : ℕ) (item4_cost : ℚ) : ℚ :=
  (item1_cost + item2_count * item2_cost + item3_count * item3_cost + item4_count * item4_cost) / num_friends

/-- Theorem: When 5 friends split a bill with the given items, each pays $11 -/
theorem friends_bill_split :
  split_bill 5 10 5 5 4 (5/2) 5 2 = 11 := by
  sorry

end friends_bill_split_l2755_275585


namespace chord_length_when_a_is_3_2_symmetrical_circle_equation_l2755_275560

-- Define the circle C
def circle_C (a : ℝ) (x y : ℝ) : Prop :=
  x^2 + y^2 + 4*x - 4*a*y + 4*a^2 + 1 = 0

-- Define the line l
def line_l (a : ℝ) (x y : ℝ) : Prop :=
  a*x + y + 2*a = 0

-- Part 1: Length of chord AB when a = 3/2
theorem chord_length_when_a_is_3_2 :
  ∃ (A B : ℝ × ℝ),
    circle_C (3/2) A.1 A.2 ∧
    circle_C (3/2) B.1 B.2 ∧
    line_l (3/2) A.1 A.2 ∧
    line_l (3/2) B.1 B.2 ∧
    ((A.1 - B.1)^2 + (A.2 - B.2)^2) = (2*Real.sqrt 39 / 13)^2 :=
sorry

-- Part 2: Equation of symmetrical circle C' when line l is tangent to circle C
theorem symmetrical_circle_equation (a : ℝ) :
  a > 0 →
  (∃ (x₀ y₀ : ℝ), circle_C a x₀ y₀ ∧ line_l a x₀ y₀ ∧
    ∀ (x y : ℝ), circle_C a x y → line_l a x y → x = x₀ ∧ y = y₀) →
  ∃ (x₁ y₁ : ℝ),
    x₁ = -5 ∧ y₁ = Real.sqrt 3 ∧
    ∀ (x y : ℝ), (x - x₁)^2 + (y - y₁)^2 = 3 ↔
      circle_C a (2*x₁ - x) (2*y₁ - y) :=
sorry

end chord_length_when_a_is_3_2_symmetrical_circle_equation_l2755_275560


namespace calculation_proof_l2755_275545

theorem calculation_proof :
  ((-7) * 5 - (-36) / 4 = -26) ∧
  (-1^4 - (1-0.4) * (1/3) * (2-3^2) = 0.4) := by
sorry

end calculation_proof_l2755_275545


namespace train_speed_constant_l2755_275523

/-- A train crossing a stationary man on a platform --/
structure Train :=
  (initial_length : ℝ)
  (initial_speed : ℝ)
  (length_increase_rate : ℝ)

/-- The final speed of the train after crossing the man --/
def final_speed (t : Train) : ℝ := t.initial_speed

theorem train_speed_constant (t : Train) 
  (h1 : t.initial_length = 160)
  (h2 : t.initial_speed = 30)
  (h3 : t.length_increase_rate = 2)
  (h4 : final_speed t = t.initial_speed) :
  final_speed t = 30 := by sorry

end train_speed_constant_l2755_275523


namespace shirt_cost_problem_l2755_275519

theorem shirt_cost_problem (x : ℝ) : 
  (3 * x + 2 * 20 = 85) → x = 15 := by
  sorry

end shirt_cost_problem_l2755_275519


namespace contrapositive_example_l2755_275540

theorem contrapositive_example :
  (∀ x : ℝ, x > 1 → x^2 > 1) ↔ (∀ x : ℝ, x^2 ≤ 1 → x ≤ 1) :=
by sorry

end contrapositive_example_l2755_275540


namespace product_of_repeating_decimal_and_eight_l2755_275500

-- Define the repeating decimal 0.6̄
def repeating_decimal : ℚ := 2/3

-- State the theorem
theorem product_of_repeating_decimal_and_eight :
  repeating_decimal * 8 = 16/3 := by
  sorry

end product_of_repeating_decimal_and_eight_l2755_275500


namespace repeating_decimal_fraction_sum_l2755_275584

theorem repeating_decimal_fraction_sum : ∃ (n d : ℕ), 
  (n.gcd d = 1) ∧ 
  (n : ℚ) / d = 45 / 99 ∧ 
  n + d = 16 :=
sorry

end repeating_decimal_fraction_sum_l2755_275584


namespace tv_sales_effect_l2755_275581

theorem tv_sales_effect (P Q : ℝ) (h1 : P > 0) (h2 : Q > 0) : 
  let new_price := 0.82 * P
  let new_quantity := 1.72 * Q
  let original_value := P * Q
  let new_value := new_price * new_quantity
  (new_value / original_value - 1) * 100 = 41.04 := by
sorry

end tv_sales_effect_l2755_275581


namespace garbage_classification_repost_l2755_275533

theorem garbage_classification_repost (n : ℕ) : 
  (1 + n + n^2 = 111) ↔ (n = 10) :=
sorry

end garbage_classification_repost_l2755_275533


namespace tan_product_less_than_one_l2755_275502

theorem tan_product_less_than_one (A B C : Real) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧  -- Angles are positive
  A + B + C = π ∧  -- Sum of angles in a triangle
  π / 2 < C ∧ C < π →  -- Angle C is obtuse
  Real.tan A * Real.tan B < 1 := by
  sorry

end tan_product_less_than_one_l2755_275502


namespace greatest_power_under_600_l2755_275510

theorem greatest_power_under_600 (a b : ℕ) : 
  a > 0 → b > 1 → a^b < 600 → 
  (∀ c d : ℕ, c > 0 → d > 1 → c^d < 600 → c^d ≤ a^b) →
  a + b = 26 := by
sorry

end greatest_power_under_600_l2755_275510


namespace children_on_bus_after_stop_l2755_275593

theorem children_on_bus_after_stop (initial_children : ℕ) (children_off : ℕ) (extra_children_on : ℕ) : 
  initial_children = 5 →
  children_off = 63 →
  extra_children_on = 9 →
  (initial_children - children_off + (children_off + extra_children_on) : ℤ) = 14 :=
by sorry

end children_on_bus_after_stop_l2755_275593


namespace one_eighth_of_2_36_equals_2_y_l2755_275589

theorem one_eighth_of_2_36_equals_2_y (y : ℕ) : (1 / 8 : ℝ) * 2^36 = 2^y → y = 33 := by
  sorry

end one_eighth_of_2_36_equals_2_y_l2755_275589


namespace right_angle_and_trig_relation_l2755_275576

-- Define a triangle ABC
structure Triangle :=
  (A B C : Real)
  (sum_angles : A + B + C = 180)
  (positive_angles : 0 < A ∧ 0 < B ∧ 0 < C)

-- Define the condition for right angle
def is_right_angled (t : Triangle) : Prop :=
  t.C = 90

-- Define the condition for equal sum of sine and cosine
def equal_sin_cos_sum (t : Triangle) : Prop :=
  Real.cos t.A + Real.sin t.A = Real.cos t.B + Real.sin t.B

-- Theorem statement
theorem right_angle_and_trig_relation (t : Triangle) :
  (is_right_angled t → equal_sin_cos_sum t) ∧
  ∃ t', equal_sin_cos_sum t' ∧ ¬is_right_angled t' :=
sorry

end right_angle_and_trig_relation_l2755_275576


namespace arithmetic_sequence_sum_l2755_275556

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  ArithmeticSequence a →
  (a 2 + a 9 = 11) →
  (a 4 + a 10 = 14) →
  (a 6 + a 11 = 17) := by
  sorry

end arithmetic_sequence_sum_l2755_275556


namespace quadratic_integer_solution_l2755_275539

theorem quadratic_integer_solution (a : ℤ) : 
  a < 0 → 
  (∃ x : ℤ, a * x^2 - 2*(a-3)*x + (a-2) = 0) ↔ 
  (a = -10 ∨ a = -4) :=
sorry

end quadratic_integer_solution_l2755_275539


namespace kylie_apple_picking_l2755_275568

/-- Represents the number of apples Kylie picked in the first hour -/
def first_hour_apples : ℕ := 66

/-- Theorem stating that given the conditions of Kylie's apple picking,
    she picked 66 apples in the first hour -/
theorem kylie_apple_picking :
  ∃ (x : ℕ), 
    x + 2*x + x/3 = 220 ∧ 
    x = first_hour_apples :=
by sorry

end kylie_apple_picking_l2755_275568


namespace hyperbola_eccentricity_is_two_l2755_275571

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos : a > 0 ∧ b > 0

/-- Represents a parabola with equation y² = -8x -/
def Parabola := Unit

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Theorem: Given a hyperbola and a parabola satisfying certain conditions, 
    the eccentricity of the hyperbola is 2 -/
theorem hyperbola_eccentricity_is_two 
  (h : Hyperbola) 
  (p : Parabola) 
  (A B O : Point)
  (h_asymptotes : A.x = 2 ∧ B.x = 2)  -- Asymptotes intersect directrix x = 2
  (h_origin : O.x = 0 ∧ O.y = 0)      -- O is the origin
  (h_area : abs ((A.x - O.x) * (B.y - O.y) - (B.x - O.x) * (A.y - O.y)) / 2 = 4 * Real.sqrt 3)
  : h.a / Real.sqrt (h.a^2 - h.b^2) = 2 := by
  sorry

end hyperbola_eccentricity_is_two_l2755_275571


namespace glass_volume_proof_l2755_275534

theorem glass_volume_proof (V : ℝ) 
  (h1 : 0.4 * V = V - 0.6 * V)  -- pessimist's glass is 60% empty (40% full)
  (h2 : 0.6 * V - 0.4 * V = 46) -- difference between optimist's and pessimist's water volumes
  : V = 230 := by
  sorry

end glass_volume_proof_l2755_275534


namespace max_x0_value_l2755_275518

theorem max_x0_value (x : Fin 1996 → ℝ) 
  (h1 : x 0 = x 1995)
  (h2 : ∀ i : Fin 1995, x i.val + 2 / x i.val = 2 * x (i.val + 1) + 1 / x (i.val + 1))
  (h3 : ∀ i : Fin 1996, x i > 0) :
  x 0 ≤ 2^997 ∧ ∃ y : Fin 1996 → ℝ, 
    y 0 = 2^997 ∧ 
    y 1995 = y 0 ∧ 
    (∀ i : Fin 1995, y i + 2 / y i = 2 * y (i.val + 1) + 1 / y (i.val + 1)) ∧
    (∀ i : Fin 1996, y i > 0) :=
by sorry

end max_x0_value_l2755_275518


namespace middle_number_proof_l2755_275538

theorem middle_number_proof (x y z : ℕ) 
  (h1 : x < y) (h2 : y < z)
  (h3 : x + y = 14) (h4 : x + z = 20) (h5 : y + z = 22)
  (h6 : x + y + z = 27) : y = 7 := by
  sorry

end middle_number_proof_l2755_275538


namespace angle_BAD_measure_l2755_275586

-- Define the triangle ABC
structure Triangle (A B C : ℝ × ℝ) : Prop where
  -- No specific conditions needed for a general triangle

-- Define an isosceles triangle
def IsIsosceles (t : Triangle A B C) : Prop :=
  dist A B = dist A C

-- Define the angle BAD
def AngleBAD (A B D : ℝ × ℝ) : ℝ := sorry

-- Define the angle DAC
def AngleDAC (A C D : ℝ × ℝ) : ℝ := sorry

-- Define the distance between two points
def dist (p q : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem angle_BAD_measure 
  (A B C D : ℝ × ℝ) 
  (t1 : Triangle A B C) 
  (t2 : Triangle A B D) :
  IsIsosceles t1 →
  IsIsosceles t2 →
  AngleDAC A C D = 39 →
  AngleBAD A B D = 70.5 := by
  sorry

end angle_BAD_measure_l2755_275586


namespace couponA_greatest_discount_specific_prices_l2755_275505

/-- Represents the discount amount for Coupon A -/
def couponA (p : ℝ) : ℝ := 0.15 * p

/-- Represents the discount amount for Coupon B -/
def couponB : ℝ := 30

/-- Represents the discount amount for Coupon C -/
def couponC (p : ℝ) : ℝ := 0.25 * (p - 150)

/-- Theorem stating when Coupon A offers the greatest discount -/
theorem couponA_greatest_discount (p : ℝ) :
  (couponA p > couponB ∧ couponA p > couponC p) ↔ (200 < p ∧ p < 375) :=
sorry

/-- Function to check if a price satisfies the condition for Coupon A being the best -/
def is_couponA_best (p : ℝ) : Prop := 200 < p ∧ p < 375

/-- Theorem for the specific price points given in the problem -/
theorem specific_prices :
  is_couponA_best 209.95 ∧
  is_couponA_best 229.95 ∧
  is_couponA_best 249.95 ∧
  ¬is_couponA_best 169.95 ∧
  ¬is_couponA_best 189.95 :=
sorry

end couponA_greatest_discount_specific_prices_l2755_275505


namespace carol_weight_l2755_275535

/-- Given that the sum of Alice's and Carol's weights is 220 pounds, and the difference
    between Carol's and Alice's weights is one-third of Carol's weight plus 10 pounds,
    prove that Carol weighs 138 pounds. -/
theorem carol_weight (alice_weight carol_weight : ℝ) 
  (h1 : alice_weight + carol_weight = 220)
  (h2 : carol_weight - alice_weight = (1/3) * carol_weight + 10) : 
  carol_weight = 138 := by
  sorry

end carol_weight_l2755_275535


namespace quadratic_fixed_point_l2755_275577

/-- The quadratic function y = -x² + (m-1)x + m has a fixed point at (-1, 0) for all m -/
theorem quadratic_fixed_point :
  ∀ (m : ℝ), -(-1)^2 + (m - 1)*(-1) + m = 0 := by
sorry

end quadratic_fixed_point_l2755_275577


namespace remove_one_gives_average_seven_point_five_l2755_275507

def original_list : List ℕ := [1,2,3,4,5,6,7,8,9,10,11,12,13]

def remove_number (l : List ℕ) (n : ℕ) : List ℕ :=
  l.filter (· ≠ n)

def average (l : List ℕ) : ℚ :=
  (l.sum : ℚ) / l.length

theorem remove_one_gives_average_seven_point_five :
  average (remove_number original_list 1) = 7.5 := by
  sorry

end remove_one_gives_average_seven_point_five_l2755_275507


namespace scissors_count_l2755_275588

/-- The total number of scissors after adding more -/
def total_scissors (initial : ℕ) (added : ℕ) : ℕ :=
  initial + added

/-- Theorem: The total number of scissors is 76 -/
theorem scissors_count : total_scissors 54 22 = 76 := by
  sorry

end scissors_count_l2755_275588


namespace smallest_n_congruence_l2755_275575

theorem smallest_n_congruence : ∃ n : ℕ+, 
  (∀ m : ℕ+, m < n → ¬(503 * m.val ≡ 1019 * m.val [ZMOD 48])) ∧
  (503 * n.val ≡ 1019 * n.val [ZMOD 48]) :=
sorry

end smallest_n_congruence_l2755_275575


namespace honey_distribution_l2755_275553

/-- Represents the volume of honey in a barrel -/
structure HoneyVolume where
  volume : ℚ
  positive : volume > 0

/-- The volume of honey in a large barrel -/
def large_barrel : HoneyVolume :=
  { volume := 1, positive := by norm_num }

/-- The volume of honey in a small barrel -/
def small_barrel : HoneyVolume :=
  { volume := 5/9, positive := by norm_num }

/-- The total volume of honey in Winnie-the-Pooh's possession -/
def total_honey : ℚ := 25 * large_barrel.volume

theorem honey_distribution (h : 25 * large_barrel.volume = 45 * small_barrel.volume) :
  total_honey = 20 * large_barrel.volume + 9 * small_barrel.volume :=
by sorry

end honey_distribution_l2755_275553


namespace fixed_point_of_exponential_function_l2755_275561

theorem fixed_point_of_exponential_function (a : ℝ) (h : a > 0) :
  let f : ℝ → ℝ := λ x ↦ a^(x - 3) + 3
  f 3 = 4 := by sorry

end fixed_point_of_exponential_function_l2755_275561


namespace stating_bus_students_theorem_l2755_275548

/-- 
Calculates the number of students on a bus after a series of stops where 
students get off and on, given an initial number of students.
-/
def students_after_stops (initial : ℚ) (fraction_off : ℚ) (num_stops : ℕ) (new_students : ℚ) : ℚ :=
  (initial * (1 - fraction_off)^num_stops) + new_students

/-- 
Theorem stating that given 72 initial students, with 1/3 getting off at each of 
the first four stops, and 12 new students boarding at the fifth stop, 
the final number of students is 236/9.
-/
theorem bus_students_theorem : 
  students_after_stops 72 (1/3) 4 12 = 236/9 := by
  sorry

end stating_bus_students_theorem_l2755_275548


namespace complex_modulus_problem_l2755_275594

theorem complex_modulus_problem (w z : ℂ) :
  w * z = 24 - 10 * I ∧ Complex.abs w = Real.sqrt 29 →
  Complex.abs z = (26 * Real.sqrt 29) / 29 :=
by sorry

end complex_modulus_problem_l2755_275594


namespace system_solutions_l2755_275542

/-- The system of equations -/
def system (x₁ x₂ x₃ x₄ x₅ : ℝ) : Prop :=
  (x₃ + x₄ + x₅)^5 = 3*x₁ ∧
  (x₄ + x₅ + x₁)^5 = 3*x₂ ∧
  (x₅ + x₁ + x₂)^5 = 3*x₃ ∧
  (x₁ + x₂ + x₃)^5 = 3*x₄ ∧
  (x₂ + x₃ + x₄)^5 = 3*x₅

/-- The solutions to the system of equations -/
def solutions : Set (ℝ × ℝ × ℝ × ℝ × ℝ) :=
  {(0, 0, 0, 0, 0), (1/3, 1/3, 1/3, 1/3, 1/3), (-1/3, -1/3, -1/3, -1/3, -1/3)}

/-- Theorem stating that the solutions are correct and complete -/
theorem system_solutions :
  ∀ x₁ x₂ x₃ x₄ x₅ : ℝ, system x₁ x₂ x₃ x₄ x₅ ↔ (x₁, x₂, x₃, x₄, x₅) ∈ solutions := by
  sorry

end system_solutions_l2755_275542


namespace sequence_gcd_property_l2755_275551

theorem sequence_gcd_property (a : ℕ → ℕ) :
  (∀ i j : ℕ, i ≠ j → Nat.gcd (a i) (a j) = Nat.gcd i j) →
  ∀ i : ℕ, a i = i :=
by sorry

end sequence_gcd_property_l2755_275551


namespace fraction_to_decimal_l2755_275509

theorem fraction_to_decimal : (7 : ℚ) / 16 = 0.4375 := by sorry

end fraction_to_decimal_l2755_275509


namespace even_quadratic_max_value_l2755_275520

/-- A quadratic function f(x) = ax^2 + bx + 1 defined on [-1-a, 2a] -/
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + 1

/-- The domain of the function -/
def domain (a : ℝ) : Set ℝ := Set.Icc (-1 - a) (2 * a)

/-- Theorem: If f is even on its domain, its maximum value is 5 -/
theorem even_quadratic_max_value (a b : ℝ) :
  (∀ x ∈ domain a, f a b x = f a b (-x)) →
  (∃ x ∈ domain a, ∀ y ∈ domain a, f a b y ≤ f a b x) →
  (∃ x ∈ domain a, f a b x = 5) :=
sorry

end even_quadratic_max_value_l2755_275520


namespace wrapping_paper_usage_l2755_275563

theorem wrapping_paper_usage (total_fraction : ℚ) (num_presents : ℕ) :
  total_fraction = 3 / 10 →
  num_presents = 3 →
  (total_fraction / num_presents : ℚ) = 1 / 10 := by
  sorry

end wrapping_paper_usage_l2755_275563


namespace fourth_quadrant_condition_l2755_275552

theorem fourth_quadrant_condition (m : ℝ) :
  let z := (m + Complex.I) / (1 + Complex.I)
  (z.re > 0 ∧ z.im < 0) ↔ m > 1 := by
  sorry

end fourth_quadrant_condition_l2755_275552


namespace book_cost_theorem_l2755_275579

/-- Calculates the cost of a single book given the total budget, remaining money, number of series bought, books per series, and tax rate. -/
def calculate_book_cost (total_budget : ℚ) (remaining_money : ℚ) (series_bought : ℕ) (books_per_series : ℕ) (tax_rate : ℚ) : ℚ :=
  let total_spent := total_budget - remaining_money
  let books_bought := series_bought * books_per_series
  let pre_tax_total := total_spent / (1 + tax_rate)
  let pre_tax_per_book := pre_tax_total / books_bought
  pre_tax_per_book * (1 + tax_rate)

/-- The cost of each book is approximately $5.96 given the problem conditions. -/
theorem book_cost_theorem :
  let total_budget : ℚ := 200
  let remaining_money : ℚ := 56
  let series_bought : ℕ := 3
  let books_per_series : ℕ := 8
  let tax_rate : ℚ := 1/10
  abs (calculate_book_cost total_budget remaining_money series_bought books_per_series tax_rate - 596/100) < 1/100 := by
  sorry


end book_cost_theorem_l2755_275579


namespace rotated_square_distance_l2755_275541

/-- Represents a square with a given side length -/
structure Square where
  sideLength : ℝ

/-- Represents the configuration of four squares -/
structure SquareConfiguration where
  squares : Fin 4 → Square
  aligned : Bool
  rotatedSquareIndex : Fin 4
  rotatedSquareTouching : Bool

/-- The distance from the top vertex of the rotated square to the original line -/
def distanceToOriginalLine (config : SquareConfiguration) : ℝ :=
  sorry

theorem rotated_square_distance
  (config : SquareConfiguration)
  (h1 : ∀ i, (config.squares i).sideLength = 2)
  (h2 : config.aligned)
  (h3 : config.rotatedSquareIndex = 1)
  (h4 : config.rotatedSquareTouching) :
  distanceToOriginalLine config = 2 :=
sorry

end rotated_square_distance_l2755_275541


namespace ptolemy_special_cases_l2755_275598

/-- Ptolemy's theorem for cyclic quadrilaterals -/
def ptolemyTheorem (a b c d e f : ℝ) : Prop := a * c + b * d = e * f

/-- A cyclic quadrilateral with one side zero -/
def cyclicQuadrilateralOneSideZero (b c d e f : ℝ) : Prop :=
  ptolemyTheorem 0 b c d e f

/-- A rectangle -/
def rectangle (a b : ℝ) : Prop :=
  a > 0 ∧ b > 0

/-- An isosceles trapezoid -/
def isoscelesTrapezoid (a b c e : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ e > 0

theorem ptolemy_special_cases :
  (∀ b c d e f : ℝ, cyclicQuadrilateralOneSideZero b c d e f → b * d = e * f) ∧
  (∀ a b : ℝ, rectangle a b → 2 * a * b = a^2 + b^2) ∧
  (∀ a b c e : ℝ, isoscelesTrapezoid a b c e → e^2 = c^2 + a * b) :=
sorry

end ptolemy_special_cases_l2755_275598


namespace days_missed_difference_l2755_275503

/-- Represents the number of students who missed a certain number of days -/
structure DaysMissed where
  days : ℕ
  count : ℕ

/-- The histogram data -/
def histogram : List DaysMissed := [
  ⟨0, 3⟩, ⟨1, 1⟩, ⟨2, 4⟩, ⟨3, 1⟩, ⟨4, 1⟩, ⟨5, 5⟩
]

/-- The total number of students -/
def totalStudents : ℕ := 15

/-- Calculates the median number of days missed -/
def median (h : List DaysMissed) (total : ℕ) : ℚ := sorry

/-- Calculates the mean number of days missed -/
def mean (h : List DaysMissed) (total : ℕ) : ℚ := sorry

/-- The main theorem -/
theorem days_missed_difference :
  mean histogram totalStudents - median histogram totalStudents = 11 / 15 := by sorry

end days_missed_difference_l2755_275503


namespace condition_sufficient_not_necessary_l2755_275514

theorem condition_sufficient_not_necessary (a b : ℝ) :
  (∀ a b : ℝ, (abs a < 1 ∧ abs b < 1) → abs (1 - a * b) > abs (a - b)) ∧
  (∃ a b : ℝ, abs (1 - a * b) > abs (a - b) ∧ ¬(abs a < 1 ∧ abs b < 1)) :=
by sorry

end condition_sufficient_not_necessary_l2755_275514


namespace sum_of_fractions_l2755_275532

theorem sum_of_fractions : (3 : ℚ) / 462 + 17 / 42 + 1 / 11 = 116 / 231 := by
  sorry

end sum_of_fractions_l2755_275532


namespace pond_length_l2755_275527

theorem pond_length (field_length field_width pond_area : ℝ) : 
  field_length = 28 ∧ 
  field_width = 14 ∧ 
  field_length = 2 * field_width ∧ 
  pond_area = (field_length * field_width) / 8 → 
  Real.sqrt pond_area = 7 := by
  sorry

end pond_length_l2755_275527


namespace bacterium_probability_l2755_275567

/-- The probability of selecting a single bacterium in a smaller volume from a larger volume --/
theorem bacterium_probability (total_volume small_volume : ℝ) (h1 : total_volume > 0) 
  (h2 : small_volume > 0) (h3 : small_volume ≤ total_volume) :
  small_volume / total_volume = 0.05 → 
  (total_volume = 2 ∧ small_volume = 0.1) := by
  sorry


end bacterium_probability_l2755_275567


namespace simplify_expression_l2755_275597

theorem simplify_expression : (45000 - 32000) * 10 + (2500 / 5) - 21005 * 3 = 67485 := by
  sorry

end simplify_expression_l2755_275597


namespace inclination_angle_range_l2755_275512

-- Define the slope range
def slope_range : Set ℝ := {k : ℝ | -Real.sqrt 3 ≤ k ∧ k ≤ Real.sqrt 3 / 3}

-- Define the inclination angle range
def angle_range : Set ℝ := {α : ℝ | (0 ≤ α ∧ α ≤ Real.pi / 6) ∨ (2 * Real.pi / 3 ≤ α ∧ α < Real.pi)}

-- Theorem statement
theorem inclination_angle_range (k : ℝ) (α : ℝ) :
  k ∈ slope_range → α = Real.arctan k → α ∈ angle_range := by sorry

end inclination_angle_range_l2755_275512


namespace scientific_notation_equality_l2755_275516

theorem scientific_notation_equality : 0.0000012 = 1.2 * 10^(-6) := by
  sorry

end scientific_notation_equality_l2755_275516


namespace powerless_common_divisor_l2755_275544

def is_powerless_digit (d : ℕ) : Prop :=
  d ≤ 9 ∧ d ≠ 0 ∧ d ≠ 1 ∧ d ≠ 4 ∧ d ≠ 8 ∧ d ≠ 9

def is_powerless_number (n : ℕ) : Prop :=
  n ≥ 10 ∧ n ≤ 99 ∧ is_powerless_digit (n / 10) ∧ is_powerless_digit (n % 10)

def smallest_powerless : ℕ := 22
def largest_powerless : ℕ := 77

theorem powerless_common_divisor :
  is_powerless_number smallest_powerless ∧
  is_powerless_number largest_powerless ∧
  smallest_powerless % 11 = 0 ∧
  largest_powerless % 11 = 0 := by sorry

end powerless_common_divisor_l2755_275544


namespace solve_cubic_equation_l2755_275521

theorem solve_cubic_equation (m : ℝ) : (m - 3)^3 = (1/27)⁻¹ → m = 6 := by
  sorry

end solve_cubic_equation_l2755_275521


namespace smallest_p_is_12_l2755_275559

/-- The set of complex numbers with real part between 1/2 and √2/2 -/
def T : Set ℂ := {z : ℂ | 1/2 ≤ z.re ∧ z.re ≤ Real.sqrt 2 / 2}

/-- The property that for all n ≥ p, there exists z ∈ T such that z^n = 1 -/
def has_root_in_T (p : ℕ) : Prop :=
  ∀ n : ℕ, n ≥ p → ∃ z ∈ T, z^n = 1

/-- 12 is the smallest positive integer satisfying the property -/
theorem smallest_p_is_12 : 
  has_root_in_T 12 ∧ ∀ p : ℕ, 0 < p → p < 12 → ¬has_root_in_T p :=
sorry

end smallest_p_is_12_l2755_275559


namespace pi_half_irrational_l2755_275582

theorem pi_half_irrational : Irrational (π / 2) :=
by
  sorry

end pi_half_irrational_l2755_275582


namespace adjacent_probability_l2755_275536

/-- The number of students in the arrangement -/
def num_students : ℕ := 9

/-- The number of rows in the seating grid -/
def num_rows : ℕ := 3

/-- The number of columns in the seating grid -/
def num_cols : ℕ := 3

/-- The number of ways two students can be adjacent in a row -/
def row_adjacencies : ℕ := num_rows * (num_cols - 1)

/-- The number of ways two students can be adjacent along a main diagonal -/
def diagonal_adjacencies : ℕ := 2 * (num_rows - 1)

/-- The total number of adjacent positions (row + diagonal) -/
def total_adjacencies : ℕ := row_adjacencies + diagonal_adjacencies

/-- The probability of two specific students being adjacent in a 3x3 grid -/
theorem adjacent_probability :
  (total_adjacencies * 2 : ℚ) / (num_students * (num_students - 1)) = 13 / 36 := by
  sorry

end adjacent_probability_l2755_275536


namespace solution_value_l2755_275511

theorem solution_value (x a : ℝ) (h : 5 * 3 - a = 8) : a = 7 := by
  sorry

end solution_value_l2755_275511


namespace candy_ratio_l2755_275587

theorem candy_ratio (cherry : ℕ) (grape : ℕ) (apple : ℕ) (total_cost : ℚ) :
  grape = 3 * cherry →
  apple = 2 * grape →
  total_cost = 200 →
  (cherry + grape + apple) * (5/2) = total_cost →
  grape / cherry = 3 := by
sorry

end candy_ratio_l2755_275587


namespace right_triangle_side_length_l2755_275537

/-- Given a right triangle ABC with angle C = 90°, BC = 6, and tan B = 0.75, prove that AC = 4.5 -/
theorem right_triangle_side_length (A B C : ℝ × ℝ) : 
  let triangle := (A, B, C)
  (∃ (AC BC : ℝ), 
    -- ABC is a right triangle with angle C = 90°
    (C.2 - A.2) * (B.1 - A.1) = (C.1 - A.1) * (B.2 - A.2) ∧
    -- BC = 6
    Real.sqrt ((C.1 - B.1)^2 + (C.2 - B.2)^2) = 6 ∧
    -- tan B = 0.75
    (C.2 - B.2) / (C.1 - B.1) = 0.75 ∧
    -- AC is the length we're solving for
    AC = Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2)) →
  AC = 4.5 := by
sorry


end right_triangle_side_length_l2755_275537


namespace natural_numbers_less_than_two_l2755_275546

theorem natural_numbers_less_than_two : 
  {n : ℕ | n < 2} = {0, 1} := by sorry

end natural_numbers_less_than_two_l2755_275546


namespace consecutive_odd_numbers_equation_l2755_275591

theorem consecutive_odd_numbers_equation (n : ℕ) : 
  let first := 7
  let second := first + 2
  let third := second + 2
  8 * first = 3 * third + 2 * second + 5 :=
by
  sorry

end consecutive_odd_numbers_equation_l2755_275591


namespace scientific_notation_of_14nm_l2755_275543

theorem scientific_notation_of_14nm (nm14 : ℝ) (h : nm14 = 0.000000014) :
  ∃ (a b : ℝ), a = 1.4 ∧ b = -8 ∧ nm14 = a * (10 : ℝ) ^ b :=
sorry

end scientific_notation_of_14nm_l2755_275543


namespace ones_digit_of_36_power_ones_digit_of_36_to_large_power_l2755_275501

theorem ones_digit_of_36_power (n : ℕ) : (36 ^ n) % 10 = 6 := by sorry

theorem ones_digit_of_36_to_large_power :
  (36 ^ (36 * (5 ^ 5))) % 10 = 6 := by sorry

end ones_digit_of_36_power_ones_digit_of_36_to_large_power_l2755_275501


namespace inequality_problem_l2755_275580

theorem inequality_problem (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 1/a + 4/b = 1) :
  (ab ≥ 16) ∧ 
  (2*a + b ≥ 6 + 4*Real.sqrt 2) ∧ 
  (1/a^2 + 16/b^2 ≥ 1/2) ∧
  ¬(∀ (a b : ℝ), a > 0 → b > 0 → 1/a + 4/b = 1 → a - b < 0) :=
by sorry

end inequality_problem_l2755_275580


namespace arithmetic_mean_difference_l2755_275565

theorem arithmetic_mean_difference (p q r : ℝ) (G : ℝ) : 
  G = (p * q * r) ^ (1/3) →
  (p + q) / 2 = 10 →
  (q + r) / 2 = 25 →
  r - p = 30 := by
  sorry

end arithmetic_mean_difference_l2755_275565


namespace cos_seven_pi_sixths_l2755_275517

theorem cos_seven_pi_sixths : Real.cos (7 * π / 6) = -Real.sqrt 3 / 2 := by
  sorry

end cos_seven_pi_sixths_l2755_275517


namespace root_exists_and_bisection_applicable_l2755_275554

-- Define the function f(x) = x^2 - 2x - 3
def f (x : ℝ) : ℝ := x^2 - 2*x - 3

-- Define the interval (-2, 2)
def interval : Set ℝ := Set.Ioo (-2) 2

-- Theorem statement
theorem root_exists_and_bisection_applicable :
  ∃ (x : ℝ), x ∈ interval ∧ f x = 0 ∧
  (∃ (a b : ℝ), a ∈ interval ∧ b ∈ interval ∧ f a * f b ≤ 0) :=
sorry

end root_exists_and_bisection_applicable_l2755_275554


namespace gmat_test_results_l2755_275529

theorem gmat_test_results (first_correct : ℝ) (second_correct : ℝ) (neither_correct : ℝ)
  (h1 : first_correct = 85)
  (h2 : second_correct = 80)
  (h3 : neither_correct = 5)
  : first_correct + second_correct - (100 - neither_correct) = 70 := by
  sorry

end gmat_test_results_l2755_275529


namespace count_less_than_04_l2755_275596

def numbers : Finset ℚ := {0.8, 1/2, 0.3, 1/3}

theorem count_less_than_04 : Finset.card (numbers.filter (λ x => x < 0.4)) = 2 := by
  sorry

end count_less_than_04_l2755_275596


namespace smallest_a1_l2755_275562

/-- A sequence of positive real numbers satisfying the given recurrence relation -/
def SequenceA (a : ℕ → ℝ) : Prop :=
  (∀ n, a n > 0) ∧ (∀ n > 1, a n = 9 * a (n - 1) - 2 * n)

/-- The theorem stating the smallest possible value of a₁ -/
theorem smallest_a1 (a : ℕ → ℝ) (h : SequenceA a) :
  ∀ a1 : ℝ, a 1 ≥ a1 → a1 ≥ 19/36 :=
by sorry

end smallest_a1_l2755_275562


namespace matrix_sum_equality_l2755_275578

def A : Matrix (Fin 2) (Fin 2) ℚ := !![2/3, -1/2; 4, -5/2]
def B : Matrix (Fin 2) (Fin 2) ℚ := !![-5/6, 1/4; 3/2, -7/4]

theorem matrix_sum_equality : A + B = !![-1/6, -1/4; 11/2, -17/4] := by
  sorry

end matrix_sum_equality_l2755_275578


namespace arithmetic_mean_after_removal_l2755_275547

theorem arithmetic_mean_after_removal (S : Finset ℝ) (x y z : ℝ) :
  S.card = 60 →
  x = 48 ∧ y = 58 ∧ z = 52 →
  (S.sum id) / S.card = 42 →
  ((S.sum id) - (x + y + z)) / (S.card - 3) = 41.4 := by
  sorry

end arithmetic_mean_after_removal_l2755_275547


namespace circle_max_distance_squared_l2755_275508

theorem circle_max_distance_squared (x y : ℝ) (h : x^2 + (y - 2)^2 = 1) :
  x^2 + y^2 ≤ 9 := by
sorry

end circle_max_distance_squared_l2755_275508


namespace yellow_jelly_bean_probability_l2755_275564

theorem yellow_jelly_bean_probability :
  let red : ℕ := 4
  let green : ℕ := 8
  let yellow : ℕ := 9
  let blue : ℕ := 5
  let total : ℕ := red + green + yellow + blue
  (yellow : ℚ) / total = 9 / 26 := by sorry

end yellow_jelly_bean_probability_l2755_275564
