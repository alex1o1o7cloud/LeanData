import Mathlib

namespace functional_equation_solution_l577_57757

theorem functional_equation_solution (f : ℤ → ℤ) 
  (h : ∀ (x y : ℤ), f (x + y) = f x + f y + 2) :
  ∃ (a : ℤ), ∀ (x : ℤ), f x = a * x - 2 := by
sorry

end functional_equation_solution_l577_57757


namespace closest_fraction_l577_57778

def medals_won : ℚ := 20 / 120

def options : List ℚ := [1/5, 1/6, 1/7, 1/8, 1/9]

theorem closest_fraction :
  ∃ (x : ℚ), x ∈ options ∧ 
  ∀ (y : ℚ), y ∈ options → |x - medals_won| ≤ |y - medals_won| :=
by sorry

end closest_fraction_l577_57778


namespace external_tangent_chord_length_l577_57744

theorem external_tangent_chord_length (r₁ r₂ R : ℝ) (h₁ : r₁ = 4) (h₂ : r₂ = 8) (h₃ : R = 12) 
  (h₄ : r₁ + r₂ = R - r₁) (h₅ : r₁ + r₂ = R - r₂) : 
  ∃ (l : ℝ), l^2 = 518.4 ∧ 
  l^2 = 4 * ((R^2) - (((2 * r₂ + r₁) / 3)^2)) :=
by sorry

end external_tangent_chord_length_l577_57744


namespace solution_set_f_geq_zero_range_m_three_zero_points_l577_57725

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := |x + 2| - |x - 2| + m

-- Define the function g
def g (m : ℝ) (x : ℝ) : ℝ := f m x - x

-- Theorem 1: Solution set of f(x) ≥ 0 when m = 1
theorem solution_set_f_geq_zero (x : ℝ) :
  f 1 x ≥ 0 ↔ x ≥ -1/2 :=
sorry

-- Theorem 2: Range of m when g(x) has three zero points
theorem range_m_three_zero_points :
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ g m x = 0 ∧ g m y = 0 ∧ g m z = 0) →
  -2 < m ∧ m < 2 :=
sorry

end solution_set_f_geq_zero_range_m_three_zero_points_l577_57725


namespace fraction_equality_problem_l577_57756

theorem fraction_equality_problem (x y : ℚ) :
  x / y = 12 / 5 → y = 25 → x = 60 := by
  sorry

end fraction_equality_problem_l577_57756


namespace max_quadratic_equations_l577_57790

def is_valid_number (n : ℕ) : Prop :=
  300 ≤ n ∧ n ≤ 999 ∧ (n / 100) % 2 = 1 ∧ (n / 100) > 1

def has_real_roots (a b c : ℕ) : Prop :=
  b * b ≥ 4 * a * c

def valid_equation (a b c : ℕ) : Prop :=
  is_valid_number a ∧ is_valid_number b ∧ is_valid_number c ∧
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  has_real_roots a b c

theorem max_quadratic_equations :
  ∃ (equations : Finset (ℕ × ℕ × ℕ)),
    (∀ (e : ℕ × ℕ × ℕ), e ∈ equations → valid_equation e.1 e.2.1 e.2.2) ∧
    equations.card = 100 ∧
    (∀ (equations' : Finset (ℕ × ℕ × ℕ)),
      (∀ (e : ℕ × ℕ × ℕ), e ∈ equations' → valid_equation e.1 e.2.1 e.2.2) →
      equations'.card ≤ 100) :=
sorry

end max_quadratic_equations_l577_57790


namespace arithmetic_sequence_formula_l577_57735

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_formula 
  (a : ℕ → ℝ) 
  (h_arithmetic : is_arithmetic_sequence a) 
  (h_a1 : a 1 = 6) 
  (h_sum : a 3 + a 5 = 0) :
  ∀ n : ℕ, a n = 8 - 2 * n :=
sorry

end arithmetic_sequence_formula_l577_57735


namespace no_solution_l577_57743

/-- Sequence definition -/
def u : ℕ → ℤ
  | 0 => 0
  | 1 => 1
  | (n + 2) => 6 * u (n + 1) + 7 * u n

/-- Main theorem -/
theorem no_solution :
  ¬ ∃ (a b c n : ℕ), a * b * (a + b) * (a^2 + a*b + b^2) = c^2022 + 42 ∧ c^2022 + 42 = u n :=
by sorry

end no_solution_l577_57743


namespace garden_breadth_l577_57771

/-- The breadth of a rectangular garden with given perimeter and length -/
theorem garden_breadth (perimeter length : ℝ) (h₁ : perimeter = 950) (h₂ : length = 375) :
  perimeter = 2 * (length + 100) := by
  sorry

end garden_breadth_l577_57771


namespace prob_draw_heart_is_one_fourth_l577_57761

/-- A deck of cards with a specific number of cards, ranks, and suits. -/
structure Deck where
  total_cards : ℕ
  num_ranks : ℕ
  num_suits : ℕ
  cards_per_suit : ℕ
  h1 : total_cards = num_suits * cards_per_suit
  h2 : cards_per_suit = num_ranks

/-- The probability of drawing a card from a specific suit in a given deck. -/
def prob_draw_suit (d : Deck) : ℚ :=
  d.cards_per_suit / d.total_cards

/-- The special deck described in the problem. -/
def special_deck : Deck where
  total_cards := 60
  num_ranks := 15
  num_suits := 4
  cards_per_suit := 15
  h1 := by rfl
  h2 := by rfl

theorem prob_draw_heart_is_one_fourth :
  prob_draw_suit special_deck = 1 / 4 := by
  sorry

end prob_draw_heart_is_one_fourth_l577_57761


namespace triangle_area_is_seven_l577_57745

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define a line by two points
structure Line2D where
  p1 : Point2D
  p2 : Point2D

-- Define the three lines
def line1 : Line2D := { p1 := { x := 0, y := 5 }, p2 := { x := 10, y := 2 } }
def line2 : Line2D := { p1 := { x := 2, y := 6 }, p2 := { x := 8, y := 1 } }
def line3 : Line2D := { p1 := { x := 0, y := 3 }, p2 := { x := 5, y := 0 } }

-- Function to calculate the area of a triangle formed by three lines
def triangleArea (l1 l2 l3 : Line2D) : ℝ :=
  sorry

-- Theorem stating that the area of the triangle is 7
theorem triangle_area_is_seven :
  triangleArea line1 line2 line3 = 7 := by
  sorry

end triangle_area_is_seven_l577_57745


namespace function_periodicity_l577_57791

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

theorem function_periodicity (f : ℝ → ℝ) 
  (h1 : ∀ x, f (x + 2) = f (2 - x))
  (h2 : ∀ x, f (x + 7) = f (7 - x)) :
  is_periodic f 10 := by
sorry

end function_periodicity_l577_57791


namespace inclination_angle_of_line_l577_57706

/-- The inclination angle of a line with equation y = x - 3 is 45 degrees. -/
theorem inclination_angle_of_line (x y : ℝ) :
  y = x - 3 → Real.arctan 1 = π / 4 := by
  sorry

end inclination_angle_of_line_l577_57706


namespace contest_order_l577_57700

/-- Represents the scores of contestants in a mathematics competition. -/
structure ContestScores where
  adam : ℝ
  bob : ℝ
  charles : ℝ
  david : ℝ
  nonnegative : adam ≥ 0 ∧ bob ≥ 0 ∧ charles ≥ 0 ∧ david ≥ 0
  sum_equality : adam + bob = charles + david
  interchange_inequality : charles + adam > bob + david
  charles_exceeds_sum : charles > adam + bob

/-- Proves that given the contest conditions, the order of scores from highest to lowest is Charles, Adam, Bob, David. -/
theorem contest_order (scores : ContestScores) : 
  scores.charles > scores.adam ∧ 
  scores.adam > scores.bob ∧ 
  scores.bob > scores.david := by
  sorry


end contest_order_l577_57700


namespace translated_minimum_point_l577_57776

-- Define the original function
def f (x : ℝ) : ℝ := |x + 1| - 2

-- Define the translated function
def g (x : ℝ) : ℝ := f (x - 3) + 4

-- State the theorem
theorem translated_minimum_point :
  ∃ (x_min : ℝ), (∀ (x : ℝ), g x_min ≤ g x) ∧ x_min = 2 ∧ g x_min = 2 :=
sorry

end translated_minimum_point_l577_57776


namespace largest_four_digit_negative_congruent_to_2_mod_17_l577_57716

theorem largest_four_digit_negative_congruent_to_2_mod_17 :
  ∀ x : ℤ, -9999 ≤ x ∧ x < -999 ∧ x ≡ 2 [ZMOD 17] → x ≤ -1001 :=
by sorry

end largest_four_digit_negative_congruent_to_2_mod_17_l577_57716


namespace five_digit_automorphic_number_l577_57714

theorem five_digit_automorphic_number :
  ∃ n : ℕ, 10000 ≤ n ∧ n < 100000 ∧ n^2 % 100000 = n :=
sorry

end five_digit_automorphic_number_l577_57714


namespace max_intersections_cubic_curve_l577_57798

/-- Given a cubic curve y = x^3 - x, the maximum number of intersections
    with any tangent line passing through a point (t, 0) on the x-axis is 3 -/
theorem max_intersections_cubic_curve (t : ℝ) :
  let f (x : ℝ) := x^3 - x
  let tangent_line (x₀ : ℝ) (x : ℝ) := (3 * x₀^2 - 1) * (x - x₀) + f x₀
  ∃ (n : ℕ), n ≤ 3 ∧
    ∀ (m : ℕ), (∃ (S : Finset ℝ), S.card = m ∧
      (∀ x ∈ S, f x = tangent_line x x ∧ tangent_line x t = 0)) →
    m ≤ n :=
by sorry

end max_intersections_cubic_curve_l577_57798


namespace sufficient_not_necessary_condition_l577_57765

def f (a : ℝ) (x : ℝ) : ℝ := |x - a|

theorem sufficient_not_necessary_condition (a : ℝ) :
  (a ≤ -2 → ∀ x y, -1 ≤ x ∧ x ≤ y → f a x ≤ f a y) ∧
  (∃ a', a' > -2 ∧ ∀ x y, -1 ≤ x ∧ x ≤ y → f a' x ≤ f a' y) :=
sorry

end sufficient_not_necessary_condition_l577_57765


namespace sin4_tan2_product_positive_l577_57760

theorem sin4_tan2_product_positive :
  ∀ (sin4 tan2 : ℝ), sin4 < 0 → tan2 < 0 → sin4 * tan2 > 0 := by sorry

end sin4_tan2_product_positive_l577_57760


namespace geometric_sum_eight_thirds_l577_57713

/-- The sum of the first n terms of a geometric sequence -/
def geometric_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

/-- The sum of the first 8 terms of a geometric sequence with first term 1/3 and common ratio 1/3 -/
theorem geometric_sum_eight_thirds : geometric_sum (1/3) (1/3) 8 = 3280/6561 := by
  sorry

end geometric_sum_eight_thirds_l577_57713


namespace simplify_expression_l577_57742

theorem simplify_expression (x : ℝ) : x + 3 - 4*x - 5 + 6*x + 7 - 8*x - 9 = -5*x - 4 := by
  sorry

end simplify_expression_l577_57742


namespace f_10_sqrt_3_l577_57780

/-- A function f is odd if f(-x) = -f(x) for all x -/
def OddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem f_10_sqrt_3 (f : ℝ → ℝ) 
    (hodd : OddFunction f)
    (hperiod : ∀ x, f (x + 2) = -f x)
    (hunit : ∀ x ∈ Set.Icc 0 1, f x = 2 * x) :
    f (10 * Real.sqrt 3) = -1.36 := by
  sorry

end f_10_sqrt_3_l577_57780


namespace min_value_sin_cos_l577_57740

theorem min_value_sin_cos (α β : ℝ) (h1 : α ≥ 0) (h2 : β ≥ 0) (h3 : α + β ≤ 2 * Real.pi) :
  Real.sin α + 2 * Real.cos β ≥ -Real.sqrt 5 := by
  sorry

end min_value_sin_cos_l577_57740


namespace area_swept_is_14_l577_57763

/-- The area swept by a line segment during a transformation -/
def area_swept (length1 width1 length2 width2 : ℝ) : ℝ :=
  length1 * width1 + length2 * width2

/-- Theorem: The area swept by the line segment is 14 -/
theorem area_swept_is_14 :
  area_swept 4 2 3 2 = 14 := by
  sorry

end area_swept_is_14_l577_57763


namespace motorcyclists_travel_time_l577_57728

/-- 
Two motorcyclists start simultaneously from opposite points A and B.
They meet at some point between A and B.
The first motorcyclist (from A to B) arrives at B 2.5 hours after meeting.
The second motorcyclist (from B to A) arrives at A 1.6 hours after meeting.
This theorem proves that their total travel times are 4.5 hours and 3.6 hours respectively.
-/
theorem motorcyclists_travel_time (s : ℝ) (h : s > 0) : 
  ∃ (t : ℝ), t > 0 ∧ 
    (s / (t + 2.5) * 2.5 = s / (t + 1.6) * t) ∧ 
    (t + 2.5 = 4.5) ∧ 
    (t + 1.6 = 3.6) := by
  sorry

#check motorcyclists_travel_time

end motorcyclists_travel_time_l577_57728


namespace quadratic_equations_properties_l577_57737

theorem quadratic_equations_properties (b c : ℤ) 
  (x₁ x₂ x₁' x₂' : ℤ) :
  (x₁^2 + b*x₁ + c = 0) →
  (x₂^2 + b*x₂ + c = 0) →
  (x₁'^2 + c*x₁' + b = 0) →
  (x₂'^2 + c*x₂' + b = 0) →
  (x₁ * x₂ > 0) →
  (x₁' * x₂' > 0) →
  (
    (x₁ < 0 ∧ x₂ < 0 ∧ x₁' < 0 ∧ x₂' < 0) ∧
    (b - 1 ≤ c ∧ c ≤ b + 1) ∧
    ((b = 4 ∧ c = 4) ∨ (b = 5 ∧ c = 6) ∨ (b = 6 ∧ c = 5))
  ) := by sorry

end quadratic_equations_properties_l577_57737


namespace cosine_of_geometric_triangle_l577_57785

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if a, b, c form a geometric sequence and c = 2a, then cos B = 3/4 -/
theorem cosine_of_geometric_triangle (a b c : ℝ) (h_positive : 0 < a ∧ 0 < b ∧ 0 < c)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_geometric : b^2 = a * c)
  (h_relation : c = 2 * a) :
  let cos_B := (a^2 + c^2 - b^2) / (2 * a * c)
  cos_B = 3/4 := by sorry

end cosine_of_geometric_triangle_l577_57785


namespace election_win_margin_l577_57787

theorem election_win_margin :
  ∀ (total_votes : ℕ) (winner_votes loser_votes : ℕ),
    winner_votes = (52 : ℕ) * total_votes / 100 →
    winner_votes = 3744 →
    loser_votes = total_votes - winner_votes →
    winner_votes - loser_votes = 288 :=
by
  sorry

end election_win_margin_l577_57787


namespace bubble_sort_iterations_for_given_list_l577_57788

def bubble_sort_iterations (list : List Int) : Nat :=
  sorry

theorem bubble_sort_iterations_for_given_list :
  bubble_sort_iterations [6, -3, 0, 15] = 3 := by
  sorry

end bubble_sort_iterations_for_given_list_l577_57788


namespace geometric_sequence_ratio_l577_57715

theorem geometric_sequence_ratio (a : ℕ → ℝ) :
  (∀ n : ℕ, ∃ q : ℝ, a (n + 1) = a n * q) →  -- a_n is a geometric sequence
  a 7 * a 11 = 6 →                           -- a_7 * a_11 = 6
  a 4 + a 14 = 5 →                           -- a_4 + a_14 = 5
  (a 20 / a 10 = 3/2 ∨ a 20 / a 10 = 2/3) :=  -- a_20 / a_10 is either 3/2 or 2/3
by
  sorry

end geometric_sequence_ratio_l577_57715


namespace cubic_roots_determinant_l577_57774

theorem cubic_roots_determinant (r s t : ℝ) (a b c : ℝ) : 
  a^3 - r*a^2 + s*a + t = 0 →
  b^3 - r*b^2 + s*b + t = 0 →
  c^3 - r*c^2 + s*c + t = 0 →
  Matrix.det !![1 + a^2, 1, 1; 1, 1 + b^2, 1; 1, 1, 1 + c^2] = r^2 + s^2 - 2*t :=
by sorry

end cubic_roots_determinant_l577_57774


namespace f_properties_l577_57733

-- Define the ⊕ operation
def oplus (a b : ℝ) : ℝ := a * b

-- Define the ⊗ operation
def otimes (a b : ℝ) : ℝ := a + b

-- Define the function f
def f (x : ℝ) : ℝ := otimes x 2 - oplus 2 x

-- Theorem statement
theorem f_properties :
  (¬ ∀ x, f (-x) = f x) ∧  -- not even
  (¬ ∀ x, f (-x) = -f x) ∧ -- not odd
  (∀ x y, x < y → f x > f y) -- decreasing
  := by sorry

end f_properties_l577_57733


namespace cos_sum_of_complex_exponentials_l577_57753

theorem cos_sum_of_complex_exponentials (α β : ℝ) 
  (h1 : Complex.exp (α * Complex.I) = (4 / 5 : ℂ) + (3 / 5 : ℂ) * Complex.I)
  (h2 : Complex.exp (β * Complex.I) = -(5 / 13 : ℂ) + (12 / 13 : ℂ) * Complex.I) :
  Real.cos (α + β) = -(7 / 13) := by
  sorry

end cos_sum_of_complex_exponentials_l577_57753


namespace clothing_percentage_is_half_l577_57703

/-- The percentage of total amount spent on clothing -/
def clothing_percentage : ℝ := sorry

/-- The percentage of total amount spent on food -/
def food_percentage : ℝ := 0.20

/-- The percentage of total amount spent on other items -/
def other_percentage : ℝ := 0.30

/-- The tax rate on clothing -/
def clothing_tax_rate : ℝ := 0.05

/-- The tax rate on food -/
def food_tax_rate : ℝ := 0

/-- The tax rate on other items -/
def other_tax_rate : ℝ := 0.10

/-- The total tax rate as a percentage of the total amount spent excluding taxes -/
def total_tax_rate : ℝ := 0.055

theorem clothing_percentage_is_half :
  clothing_percentage +
  food_percentage +
  other_percentage = 1 ∧
  clothing_percentage * clothing_tax_rate +
  food_percentage * food_tax_rate +
  other_percentage * other_tax_rate = total_tax_rate →
  clothing_percentage = 0.5 := by sorry

end clothing_percentage_is_half_l577_57703


namespace parallelogram_properties_l577_57724

-- Define a parallelogram
structure Parallelogram where
  is_quadrilateral : Bool
  opposite_sides_parallel : Bool

-- Define the properties
def has_equal_sides (p : Parallelogram) : Prop := sorry
def is_square (p : Parallelogram) : Prop := sorry

-- Theorem statement
theorem parallelogram_properties (p : Parallelogram) :
  (p.is_quadrilateral ∧ p.opposite_sides_parallel) →
  (∃ p1 : Parallelogram, has_equal_sides p1 ∧ ¬is_square p1) ∧
  (∀ p2 : Parallelogram, is_square p2 → has_equal_sides p2) ∧
  (∃ p3 : Parallelogram, has_equal_sides p3 ∧ is_square p3) :=
sorry

end parallelogram_properties_l577_57724


namespace improper_fraction_subtraction_l577_57775

theorem improper_fraction_subtraction (a b n : ℕ) 
  (h1 : a > b) 
  (h2 : n < b) : 
  (a - n : ℚ) / (b - n) > (a : ℚ) / b := by
sorry

end improper_fraction_subtraction_l577_57775


namespace unique_natural_pair_l577_57749

theorem unique_natural_pair : 
  ∃! (k n : ℕ), 
    120 < k * n ∧ k * n < 130 ∧ 
    2 < (k : ℚ) / n ∧ (k : ℚ) / n < 3 ∧
    k = 18 ∧ n = 7 := by
  sorry

end unique_natural_pair_l577_57749


namespace complex_modulus_l577_57777

theorem complex_modulus (z : ℂ) : z - Complex.I = 1 + Complex.I → Complex.abs z = Real.sqrt 5 := by
  sorry

end complex_modulus_l577_57777


namespace triangle_side_formulas_l577_57750

/-- Given a triangle ABC with sides a, b, c, altitude m from A, and midline k from A,
    where b + c = 2l, prove the expressions for sides a, b, and c. -/
theorem triangle_side_formulas (a b c l m k : ℝ) : 
  b + c = 2 * l →
  k^2 = (b^2 + c^2) / 4 + (a / 2)^2 →
  m = (b * c) / a →
  b = l + Real.sqrt ((k^2 - m^2) * (k^2 - l^2) / (k^2 - m^2 - l^2)) ∧
  c = l - Real.sqrt ((k^2 - m^2) * (k^2 - l^2) / (k^2 - m^2 - l^2)) ∧
  a = 2 * l * Real.sqrt ((k^2 - l^2) / (k^2 - m^2 - l^2)) := by
  sorry

end triangle_side_formulas_l577_57750


namespace geometric_sequence_sum_l577_57741

/-- A geometric sequence with common ratio q > 1 -/
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  q > 1 ∧ ∀ n, a (n + 1) = a n * q

theorem geometric_sequence_sum (a : ℕ → ℝ) (q : ℝ) :
  geometric_sequence a q →
  (4 * a 2005 ^ 2 - 8 * a 2005 + 3 = 0) →
  (4 * a 2006 ^ 2 - 8 * a 2006 + 3 = 0) →
  a 2007 + a 2008 = 18 :=
by sorry

end geometric_sequence_sum_l577_57741


namespace shooting_game_probability_l577_57726

theorem shooting_game_probability (A B : Type) 
  (hit_score : ℕ) (miss_score : ℕ) 
  (A_hit_rate : ℚ) (B_hit_rate : ℚ) 
  (sum_two_prob : ℚ) :
  hit_score = 2 →
  miss_score = 0 →
  A_hit_rate = 3/5 →
  sum_two_prob = 9/20 →
  (A_hit_rate * (1 - B_hit_rate) + (1 - A_hit_rate) * B_hit_rate = sum_two_prob) →
  B_hit_rate = 3/4 := by
sorry

end shooting_game_probability_l577_57726


namespace pastries_sold_l577_57752

theorem pastries_sold (cupcakes cookies left : ℕ) 
  (h1 : cupcakes = 7) 
  (h2 : cookies = 5) 
  (h3 : left = 8) : 
  cupcakes + cookies - left = 4 := by
  sorry

end pastries_sold_l577_57752


namespace negative_m_squared_n_identity_l577_57793

theorem negative_m_squared_n_identity (m n : ℝ) : -m^2*n - 2*m^2*n = -3*m^2*n := by
  sorry

end negative_m_squared_n_identity_l577_57793


namespace intersection_M_N_l577_57782

def M : Set ℤ := {1, 2, 3, 4, 5, 6}
def N : Set ℤ := {x | -2 < x ∧ x < 5}

theorem intersection_M_N : M ∩ N = {1, 2, 3, 4} := by sorry

end intersection_M_N_l577_57782


namespace line_contains_point_l577_57702

theorem line_contains_point (k : ℝ) : 
  (3 / 4 - 3 * k * (1 / 3) = 7 * (-4)) ↔ k = 28.75 := by sorry

end line_contains_point_l577_57702


namespace rational_trig_sums_l577_57747

theorem rational_trig_sums (x : ℝ) 
  (s_rational : ∃ q : ℚ, (Real.sin (64 * x) + Real.sin (65 * x)) = ↑q)
  (t_rational : ∃ q : ℚ, (Real.cos (64 * x) + Real.cos (65 * x)) = ↑q) :
  (∃ q1 q2 : ℚ, Real.cos (64 * x) = ↑q1 ∧ Real.cos (65 * x) = ↑q2) ∨
  (∃ q1 q2 : ℚ, Real.sin (64 * x) = ↑q1 ∧ Real.sin (65 * x) = ↑q2) :=
by sorry

end rational_trig_sums_l577_57747


namespace consecutive_odd_integers_sum_l577_57797

theorem consecutive_odd_integers_sum (x y : ℤ) : 
  (Odd x ∧ Odd y) →  -- x and y are odd
  y = x + 4 →        -- y is the next consecutive odd integer after x
  y = 5 * x →        -- y is five times x
  x + y = 6 :=       -- their sum is 6
by
  sorry

end consecutive_odd_integers_sum_l577_57797


namespace trigonometric_identity_l577_57708

theorem trigonometric_identity : 
  2 * Real.sin (50 * π / 180) * (1 + Real.sqrt 3 * Real.tan (10 * π / 180)) = 2 := by
  sorry

end trigonometric_identity_l577_57708


namespace carol_fraction_l577_57721

/-- Represents the money each person has -/
structure Money where
  alice : ℚ
  bob : ℚ
  carol : ℚ

/-- The conditions of the problem -/
def problem_conditions (m : Money) : Prop :=
  m.carol = 0 ∧ 
  m.alice > 0 ∧ 
  m.bob > 0 ∧ 
  m.alice / 6 = m.bob / 3 ∧
  m.alice / 6 > 0

/-- The final state after Alice and Bob give money to Carol -/
def final_state (m : Money) : Money :=
  { alice := m.alice * (5/6),
    bob := m.bob * (2/3),
    carol := m.alice / 6 + m.bob / 3 }

/-- The theorem to be proved -/
theorem carol_fraction (m : Money) 
  (h : problem_conditions m) : 
  (final_state m).carol / ((final_state m).alice + (final_state m).bob + (final_state m).carol) = 2/9 := by
  sorry

end carol_fraction_l577_57721


namespace right_triangle_altitude_segment_ratio_l577_57779

theorem right_triangle_altitude_segment_ratio :
  ∀ (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0),
  a^2 + b^2 = c^2 →  -- right triangle condition
  a = 3 * b →        -- leg ratio condition
  ∃ (d e : ℝ), d > 0 ∧ e > 0 ∧ d + e = c ∧ d / e = 9 := by
  sorry

end right_triangle_altitude_segment_ratio_l577_57779


namespace min_white_pairs_8x8_20black_l577_57764

/-- Represents a grid with black and white cells -/
structure Grid :=
  (size : ℕ)
  (black_cells : ℕ)

/-- Calculates the total number of adjacent cell pairs in a square grid -/
def total_pairs (g : Grid) : ℕ :=
  2 * (g.size - 1) * g.size

/-- Calculates the maximum number of central black cells that can be placed without adjacency -/
def max_central_black (g : Grid) : ℕ :=
  (g.size - 2)^2 / 2

/-- Calculates the minimum number of adjacent white cell pairs -/
def min_white_pairs (g : Grid) : ℕ :=
  total_pairs g - (60 + min g.black_cells (max_central_black g))

/-- Theorem stating the minimum number of adjacent white cell pairs for an 8x8 grid with 20 black cells -/
theorem min_white_pairs_8x8_20black :
  let g : Grid := { size := 8, black_cells := 20 }
  min_white_pairs g = 34 := by
  sorry

end min_white_pairs_8x8_20black_l577_57764


namespace score_96_not_possible_l577_57755

/-- Represents the score on a test with 25 questions -/
structure TestScore where
  correct : Nat
  unanswered : Nat
  incorrect : Nat
  h_total : correct + unanswered + incorrect = 25

/-- Calculates the total score for a given TestScore -/
def totalScore (ts : TestScore) : Nat :=
  4 * ts.correct + 2 * ts.unanswered

/-- Theorem stating that a score of 96 is not achievable -/
theorem score_96_not_possible :
  ¬ ∃ (ts : TestScore), totalScore ts = 96 := by
  sorry

end score_96_not_possible_l577_57755


namespace sphere_surface_area_l577_57711

theorem sphere_surface_area (a b c : ℝ) (h1 : a * b * c = Real.sqrt 6) 
  (h2 : a * b = Real.sqrt 2) (h3 : b * c = Real.sqrt 3) : 
  4 * Real.pi * ((a^2 + b^2 + c^2).sqrt / 2)^2 = 6 * Real.pi := by
  sorry

end sphere_surface_area_l577_57711


namespace corrected_mean_l577_57799

theorem corrected_mean (n : ℕ) (incorrect_mean : ℚ) (incorrect_value : ℚ) (correct_value : ℚ) :
  n = 50 ∧ incorrect_mean = 30 ∧ incorrect_value = 23 ∧ correct_value = 48 →
  (n : ℚ) * incorrect_mean - incorrect_value + correct_value = n * (30.5 : ℚ) := by
  sorry

end corrected_mean_l577_57799


namespace product_is_zero_matrix_l577_57704

def skew_symmetric_matrix (d e f : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  !![0, d, -e;
     -d, 0, f;
     e, -f, 0]

def symmetric_matrix (d e f : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  !![d^2, d*e, d*f;
     d*e, e^2, e*f;
     d*f, e*f, f^2]

theorem product_is_zero_matrix (d e f : ℝ) : 
  skew_symmetric_matrix d e f * symmetric_matrix d e f = 0 := by
  sorry

end product_is_zero_matrix_l577_57704


namespace association_membership_l577_57719

theorem association_membership (M : ℕ) : 
  (525 : ℕ) ≤ M ∧ 
  (315 : ℕ) = (525 * 60 : ℕ) / 100 ∧ 
  (315 : ℝ) = (M : ℝ) * 19.6875 / 100 →
  M = 1600 := by
sorry

end association_membership_l577_57719


namespace percentage_material_B_in_solution_Y_l577_57781

/-- Given two solutions X and Y, and their mixture, this theorem proves
    the percentage of material B in solution Y. -/
theorem percentage_material_B_in_solution_Y
  (percent_A_X : ℝ) (percent_B_X : ℝ) (percent_A_Y : ℝ)
  (percent_X_in_mixture : ℝ) (percent_A_in_mixture : ℝ)
  (h1 : percent_A_X = 0.20)
  (h2 : percent_B_X = 0.80)
  (h3 : percent_A_Y = 0.30)
  (h4 : percent_X_in_mixture = 0.80)
  (h5 : percent_A_in_mixture = 0.22)
  (h6 : percent_X_in_mixture * percent_A_X + (1 - percent_X_in_mixture) * percent_A_Y = percent_A_in_mixture) :
  1 - percent_A_Y = 0.70 := by
sorry

end percentage_material_B_in_solution_Y_l577_57781


namespace tank_cart_friction_l577_57730

/-- The frictional force acting on a tank resting on an accelerating cart --/
theorem tank_cart_friction (m₁ m₂ a μ g : ℝ) (h₁ : m₁ = 3) (h₂ : m₂ = 15) (h₃ : a = 4) (h₄ : μ = 0.6) (h₅ : g = 9.8) :
  let F_friction := m₁ * a
  let F_max_static := μ * m₁ * g
  F_friction ≤ F_max_static ∧ F_friction = 12 := by
  sorry

end tank_cart_friction_l577_57730


namespace cubic_function_extrema_difference_l577_57768

/-- A cubic function with specific properties -/
def f (a b c : ℝ) (x : ℝ) : ℝ := x^3 + 3*a*x^2 + 3*b*x + c

/-- The derivative of f -/
def f' (a b : ℝ) (x : ℝ) : ℝ := 3*x^2 + 6*a*x + 3*b

/-- The second derivative of f -/
def f'' (a : ℝ) (x : ℝ) : ℝ := 6*x + 6*a

theorem cubic_function_extrema_difference (a b c : ℝ) :
  (f' a b 2 = 0) →  -- Extremum at x = 2
  (f' a b 1 = -3) →  -- Tangent line at x = 1 has slope -3 (parallel to 6x + 2y + 5 = 0)
  (∃ (x_max x_min : ℝ), 
    (∀ x, f a b c x ≤ f a b c x_max) ∧ 
    (∀ x, f a b c x ≥ f a b c x_min) ∧
    (f a b c x_max - f a b c x_min = 4)) := by
  sorry

end cubic_function_extrema_difference_l577_57768


namespace quadratic_roots_relation_l577_57720

theorem quadratic_roots_relation (q : ℝ) : 
  let eq1 := fun x : ℂ => x^2 + 2*x + q
  let eq2 := fun x : ℂ => (1+q)*(x^2 + 2*x + q) - 2*(q-1)*(x^2 + 1)
  (∃ x y : ℝ, x ≠ y ∧ eq1 x = 0 ∧ eq1 y = 0) ↔ 
  (∀ z : ℂ, eq2 z = 0 → z.im ≠ 0) :=
by sorry

end quadratic_roots_relation_l577_57720


namespace praveen_age_multiplier_l577_57709

def present_age : ℕ := 20

def age_3_years_back : ℕ := present_age - 3

def age_after_10_years : ℕ := present_age + 10

theorem praveen_age_multiplier :
  (age_after_10_years : ℚ) / age_3_years_back = 30 / 17 := by sorry

end praveen_age_multiplier_l577_57709


namespace twenty_paise_coins_count_l577_57770

/-- Given a total of 324 coins consisting of 20 paise and 25 paise denominations,
    and a total sum of Rs. 70, prove that the number of 20 paise coins is 220. -/
theorem twenty_paise_coins_count (x y : ℕ) : 
  x + y = 324 → 
  20 * x + 25 * y = 7000 → 
  x = 220 :=
by sorry

end twenty_paise_coins_count_l577_57770


namespace train_length_l577_57717

theorem train_length (speed : ℝ) (train_length : ℝ) : 
  (train_length + 130) / 15 = speed ∧ 
  (train_length + 250) / 20 = speed → 
  train_length = 230 :=
by
  sorry

end train_length_l577_57717


namespace opposite_of_neg_three_squared_l577_57796

theorem opposite_of_neg_three_squared : -(-(3^2)) = 9 := by
  sorry

end opposite_of_neg_three_squared_l577_57796


namespace theo_cookie_consumption_l577_57769

def cookies_per_sitting : ℕ := 25
def sittings_per_day : ℕ := 5
def days_per_month : ℕ := 27
def months : ℕ := 9

theorem theo_cookie_consumption :
  cookies_per_sitting * sittings_per_day * days_per_month * months = 30375 :=
by
  sorry

end theo_cookie_consumption_l577_57769


namespace green_balls_count_l577_57784

theorem green_balls_count (total : ℕ) (white yellow red purple : ℕ) (prob_not_red_purple : ℚ) :
  total = 100 ∧
  white = 50 ∧
  yellow = 8 ∧
  red = 9 ∧
  purple = 3 ∧
  prob_not_red_purple = 88/100 →
  ∃ green : ℕ, green = 30 ∧ white + yellow + green + red + purple = total ∧
  (white + yellow + green : ℚ) / total = prob_not_red_purple :=
by sorry

end green_balls_count_l577_57784


namespace complex_sum_magnitude_l577_57705

theorem complex_sum_magnitude (a b c : ℂ) 
  (h1 : Complex.abs a = 1)
  (h2 : Complex.abs b = 1)
  (h3 : Complex.abs c = 1)
  (h4 : a^3 / (b^2 * c) + b^3 / (a^2 * c) + c^3 / (a^2 * b) = 1) :
  Complex.abs (a + b + c) = Real.sqrt 15 / 3 := by
sorry

end complex_sum_magnitude_l577_57705


namespace gcd_45_75_l577_57723

theorem gcd_45_75 : Nat.gcd 45 75 = 15 := by
  sorry

end gcd_45_75_l577_57723


namespace sphere_radius_ratio_l577_57736

theorem sphere_radius_ratio (V₁ V₂ : ℝ) (h₁ : V₁ = 512 * Real.pi) (h₂ : V₂ = 32 * Real.pi) :
  (V₂ / V₁) ^ (1/3 : ℝ) = 1/2 := by
  sorry

end sphere_radius_ratio_l577_57736


namespace solution_sets_equality_l577_57748

theorem solution_sets_equality (a b : ℝ) : 
  (∀ x, |x - 2| > 1 ↔ x^2 + a*x + b > 0) → a + b = -1 := by
  sorry

end solution_sets_equality_l577_57748


namespace channel_probabilities_l577_57729

/-- Represents a binary communication channel with error probabilities -/
structure Channel where
  α : Real
  β : Real
  h_α_pos : 0 < α
  h_α_lt_one : α < 1
  h_β_pos : 0 < β
  h_β_lt_one : β < 1

/-- Probability of receiving 1,0,1 when sending 1,0,1 in single transmission -/
def singleTransProb (c : Channel) : Real :=
  (1 - c.α) * (1 - c.β)^2

/-- Probability of receiving 1,0,1 when sending 1 in triple transmission -/
def tripleTransProb (c : Channel) : Real :=
  c.β * (1 - c.β)^2

/-- Probability of decoding as 1 when sending 1 in triple transmission -/
def tripleTransDecodeProb (c : Channel) : Real :=
  c.β * (1 - c.β)^2 + (1 - c.β)^3

/-- Probability of decoding as 0 when sending 0 in single transmission -/
def singleTransDecodeZeroProb (c : Channel) : Real :=
  1 - c.α

/-- Probability of decoding as 0 when sending 0 in triple transmission -/
def tripleTransDecodeZeroProb (c : Channel) : Real :=
  3 * c.α * (1 - c.α)^2 + (1 - c.α)^3

theorem channel_probabilities (c : Channel) :
  (singleTransProb c = (1 - c.α) * (1 - c.β)^2) ∧
  (tripleTransProb c = c.β * (1 - c.β)^2) ∧
  (tripleTransDecodeProb c = c.β * (1 - c.β)^2 + (1 - c.β)^3) ∧
  (∀ h : 0 < c.α ∧ c.α < 0.5,
    tripleTransDecodeZeroProb c > singleTransDecodeZeroProb c) :=
by sorry

end channel_probabilities_l577_57729


namespace sams_morning_run_l577_57751

theorem sams_morning_run (morning_run : ℝ) 
  (store_walk : ℝ)
  (bike_ride : ℝ)
  (total_distance : ℝ)
  (h1 : store_walk = 2 * morning_run)
  (h2 : bike_ride = 12)
  (h3 : total_distance = 18)
  (h4 : morning_run + store_walk + bike_ride = total_distance) :
  morning_run = 2 := by
sorry

end sams_morning_run_l577_57751


namespace two_diamonds_balance_l577_57712

/-- Represents the balance between shapes -/
structure Balance where
  triangle : ℝ
  diamond : ℝ
  circle : ℝ

/-- The given balance conditions -/
def balance_conditions (b : Balance) : Prop :=
  3 * b.triangle + b.diamond = 9 * b.circle ∧
  b.triangle = b.diamond + 2 * b.circle

/-- The theorem to prove -/
theorem two_diamonds_balance (b : Balance) :
  balance_conditions b → 2 * b.diamond = 1.5 * b.circle := by
  sorry

end two_diamonds_balance_l577_57712


namespace trajectory_of_midpoint_M_l577_57783

-- Define the circle C
def circle_C (k : ℝ) (x y : ℝ) : Prop :=
  (x - 2)^2 + (y - 4)^2 = k ∧ k > 0

-- Define the intersection points
def intersection_points (k : ℝ) : Prop :=
  ∃ (yA yB xE xF : ℝ),
    circle_C k 0 yA ∧ circle_C k 0 yB ∧ yA > yB ∧
    circle_C k xE 0 ∧ circle_C k xF 0 ∧ xE > xF

-- Define the midpoint M of AE
def midpoint_M (x y yA xE : ℝ) : Prop :=
  x = (0 + xE) / 2 ∧ y = (yA + 0) / 2

-- Theorem statement
theorem trajectory_of_midpoint_M
  (k : ℝ) (x y : ℝ) :
  circle_C k x y →
  intersection_points k →
  (∃ (yA xE : ℝ), midpoint_M x y yA xE) →
  x > 1 →
  y > 2 + Real.sqrt 3 →
  (y - 2)^2 - (x - 1)^2 = 3 :=
sorry

end trajectory_of_midpoint_M_l577_57783


namespace function_inequality_l577_57762

theorem function_inequality (f : ℝ → ℝ) (h_diff : Differentiable ℝ f) 
  (h_ineq : ∀ x : ℝ, f x > deriv f x) : 
  (Real.exp 2016 * f (-2016) > f 0) ∧ (f 2016 < Real.exp 2016 * f 0) := by
  sorry

end function_inequality_l577_57762


namespace dragon_legs_count_l577_57732

/-- Represents the number of legs per centipede -/
def centipede_legs : ℕ := 40

/-- Represents the number of heads per dragon -/
def dragon_heads : ℕ := 9

/-- Represents the total number of heads in the cage -/
def total_heads : ℕ := 50

/-- Represents the total number of legs in the cage -/
def total_legs : ℕ := 220

/-- Represents the number of centipedes in the cage -/
def num_centipedes : ℕ := 40

/-- Represents the number of dragons in the cage -/
def num_dragons : ℕ := total_heads - num_centipedes

/-- Theorem stating that each dragon has 4 legs -/
theorem dragon_legs_count : 
  ∃ (dragon_legs : ℕ), 
    dragon_legs = 4 ∧ 
    num_centipedes * centipede_legs + num_dragons * dragon_legs = total_legs :=
sorry

end dragon_legs_count_l577_57732


namespace correct_calculation_l577_57746

theorem correct_calculation (x : ℤ) : 
  x + 238 = 637 → x - 382 = 17 := by
  sorry

end correct_calculation_l577_57746


namespace average_speed_calculation_l577_57795

theorem average_speed_calculation (total_distance : ℝ) (first_half_speed : ℝ) (second_half_time_factor : ℝ) :
  total_distance = 640 →
  first_half_speed = 80 →
  second_half_time_factor = 3 →
  let first_half_distance := total_distance / 2
  let first_half_time := first_half_distance / first_half_speed
  let second_half_time := first_half_time * second_half_time_factor
  let total_time := first_half_time + second_half_time
  (total_distance / total_time) = 40 := by
sorry

end average_speed_calculation_l577_57795


namespace pentagon_sum_l577_57707

/-- Definition of a pentagon -/
structure Pentagon where
  sides : ℕ
  vertices : ℕ
  is_pentagon : sides = 5 ∧ vertices = 5

/-- Theorem: The sum of sides and vertices of a pentagon is 10 -/
theorem pentagon_sum (p : Pentagon) : p.sides + p.vertices = 10 := by
  sorry

end pentagon_sum_l577_57707


namespace simplest_common_denominator_l577_57701

-- Define the fractions
def fraction1 (x y : ℚ) : ℚ := 1 / (2 * x^2 * y)
def fraction2 (x y : ℚ) : ℚ := 1 / (6 * x * y^3)

-- Define the common denominator
def common_denominator (x y : ℚ) : ℚ := 6 * x^2 * y^3

-- Theorem statement
theorem simplest_common_denominator (x y : ℚ) (hx : x ≠ 0) (hy : y ≠ 0) :
  ∃ (a b : ℚ), 
    fraction1 x y = a / common_denominator x y ∧
    fraction2 x y = b / common_denominator x y ∧
    (∀ (c : ℚ), c > 0 → 
      (∃ (d e : ℚ), fraction1 x y = d / c ∧ fraction2 x y = e / c) →
      c ≥ common_denominator x y) :=
sorry

end simplest_common_denominator_l577_57701


namespace inequality_solution_set_l577_57722

theorem inequality_solution_set 
  (a b : ℝ) 
  (h : Set.Ioo 1 2 = {x | x^2 - a*x + b < 0}) :
  {x : ℝ | 1/x < b/a} = Set.union (Set.Iio 0) (Set.Ioi (3/2)) :=
by sorry

end inequality_solution_set_l577_57722


namespace three_number_sum_l577_57738

theorem three_number_sum (a b c : ℝ) : 
  a + b = 35 ∧ b + c = 54 ∧ c + a = 58 → a + b + c = 73.5 := by
  sorry

end three_number_sum_l577_57738


namespace polynomial_simplification_l577_57789

theorem polynomial_simplification (r : ℝ) :
  (2 * r^3 + 5 * r^2 - 4 * r + 8) - (r^3 + 9 * r^2 - 2 * r - 3) =
  r^3 - 4 * r^2 - 2 * r + 11 := by
sorry

end polynomial_simplification_l577_57789


namespace rectangular_field_area_l577_57773

/-- The area of a rectangular field with one side of 15 m and a diagonal of 18 m -/
theorem rectangular_field_area : 
  ∀ (a b : ℝ), 
  a = 15 → 
  a^2 + b^2 = 18^2 → 
  a * b = 45 * Real.sqrt 11 := by
  sorry

end rectangular_field_area_l577_57773


namespace marble_fraction_after_tripling_l577_57772

theorem marble_fraction_after_tripling (total : ℚ) (h_pos : total > 0) : 
  let green := (4/7) * total
  let blue := (1/7) * total
  let initial_white := total - green - blue
  let new_white := 3 * initial_white
  let new_total := green + blue + new_white
  new_white / new_total = 6/11 := by sorry

end marble_fraction_after_tripling_l577_57772


namespace white_shirt_cost_is_25_l577_57754

/-- Represents the t-shirt sale scenario -/
structure TShirtSale where
  totalShirts : ℕ
  saleTime : ℕ
  blackShirtCost : ℕ
  revenuePerMinute : ℕ

/-- Calculates the cost of white t-shirts given the sale conditions -/
def whiteShirtCost (sale : TShirtSale) : ℕ :=
  let totalRevenue := sale.revenuePerMinute * sale.saleTime
  let blackShirts := sale.totalShirts / 2
  let whiteShirts := sale.totalShirts / 2
  let blackRevenue := blackShirts * sale.blackShirtCost
  let whiteRevenue := totalRevenue - blackRevenue
  whiteRevenue / whiteShirts

/-- Theorem stating that the white t-shirt cost is $25 under the given conditions -/
theorem white_shirt_cost_is_25 (sale : TShirtSale) 
  (h1 : sale.totalShirts = 200)
  (h2 : sale.saleTime = 25)
  (h3 : sale.blackShirtCost = 30)
  (h4 : sale.revenuePerMinute = 220) :
  whiteShirtCost sale = 25 := by
  sorry

#eval whiteShirtCost { totalShirts := 200, saleTime := 25, blackShirtCost := 30, revenuePerMinute := 220 }

end white_shirt_cost_is_25_l577_57754


namespace quadratic_root_sum_inequality_l577_57731

theorem quadratic_root_sum_inequality (a b c x₁ : ℝ) (h₁ : x₁ > 0) (h₂ : a * x₁^2 + b * x₁ + c = 0) :
  ∃ x₂ : ℝ, x₂ > 0 ∧ c * x₂^2 + b * x₂ + a = 0 ∧ x₁ + x₂ ≥ 2 := by
  sorry

end quadratic_root_sum_inequality_l577_57731


namespace events_mutually_exclusive_but_not_complementary_l577_57767

-- Define the set of people
inductive Person : Type
| A : Person
| B : Person
| C : Person
| D : Person

-- Define the set of cards
inductive Card : Type
| Red : Card
| Yellow : Card
| Green : Card
| Blue : Card

-- Define a distribution as a function from Person to Card
def Distribution := Person → Card

-- Define the event "A receives the red card"
def A_gets_red (d : Distribution) : Prop := d Person.A = Card.Red

-- Define the event "B receives the red card"
def B_gets_red (d : Distribution) : Prop := d Person.B = Card.Red

-- State the theorem
theorem events_mutually_exclusive_but_not_complementary :
  (∀ d : Distribution, ¬(A_gets_red d ∧ B_gets_red d)) ∧
  (∃ d : Distribution, ¬(A_gets_red d ∨ B_gets_red d)) :=
by sorry

end events_mutually_exclusive_but_not_complementary_l577_57767


namespace tree_growth_problem_l577_57786

/-- Tree growth problem -/
theorem tree_growth_problem (initial_height : ℝ) (yearly_growth : ℝ) (height_ratio : ℝ) :
  initial_height = 4 →
  yearly_growth = 1 →
  height_ratio = 5/4 →
  ∃ (years : ℕ), 
    (initial_height + years * yearly_growth) = 
    height_ratio * (initial_height + 4 * yearly_growth) ∧
    years = 6 :=
by sorry

end tree_growth_problem_l577_57786


namespace system_solution_is_one_two_l577_57739

theorem system_solution_is_one_two :
  ∃! (s : Set ℝ), s = {1, 2} ∧
  (∀ x y : ℝ, (x^4 + y^4 = 17 ∧ x + y = 3) ↔ (x ∈ s ∧ y ∈ s ∧ x ≠ y)) :=
sorry

end system_solution_is_one_two_l577_57739


namespace three_digit_numbers_count_l577_57718

-- Define the set of numbers
def S : Finset ℕ := {1, 2, 3, 4}

-- Define the number of elements to choose
def r : ℕ := 3

-- Theorem statement
theorem three_digit_numbers_count : Nat.descFactorial (Finset.card S) r = 24 := by
  sorry

end three_digit_numbers_count_l577_57718


namespace new_supervisor_salary_l577_57794

-- Define the number of workers
def num_workers : ℕ := 8

-- Define the total number of people (workers + supervisor)
def total_people : ℕ := num_workers + 1

-- Define the initial average salary
def initial_average : ℚ := 430

-- Define the old supervisor's salary
def old_supervisor_salary : ℚ := 870

-- Define the new average salary
def new_average : ℚ := 390

-- Theorem to prove
theorem new_supervisor_salary :
  ∃ (workers_total_salary new_supervisor_salary : ℚ),
    (workers_total_salary + old_supervisor_salary) / total_people = initial_average ∧
    workers_total_salary / num_workers ≤ old_supervisor_salary ∧
    (workers_total_salary + new_supervisor_salary) / total_people = new_average ∧
    new_supervisor_salary = 510 :=
sorry

end new_supervisor_salary_l577_57794


namespace largest_prime_factor_of_expression_l577_57734

theorem largest_prime_factor_of_expression : 
  ∃ p : ℕ, Nat.Prime p ∧ 
    p ∣ (20^3 + 15^4 - 10^5) ∧ 
    ∀ q : ℕ, Nat.Prime q → q ∣ (20^3 + 15^4 - 10^5) → q ≤ p ∧ 
    p = 113 := by
  sorry

end largest_prime_factor_of_expression_l577_57734


namespace factor_expression_l577_57758

theorem factor_expression (a b c : ℝ) :
  a^4 * (b^3 - c^3) + b^4 * (c^3 - a^3) + c^4 * (a^3 - b^3) =
  (a - b) * (b - c) * (c - a) * (a*b^3 + a*c^3 + a*b*c^2 + b^2*c^2) := by
  sorry

end factor_expression_l577_57758


namespace tan_x_equals_zero_l577_57727

theorem tan_x_equals_zero (x : Real) 
  (h1 : 0 ≤ x ∧ x ≤ π) 
  (h2 : 3 * Real.sin (x/2) = Real.sqrt (1 + Real.sin x) - Real.sqrt (1 - Real.sin x)) : 
  Real.tan x = 0 := by
sorry

end tan_x_equals_zero_l577_57727


namespace selling_price_loss_percentage_l577_57792

theorem selling_price_loss_percentage (cost_price : ℝ) 
  (h : cost_price > 0) : 
  let selling_price_100 := 40 * cost_price
  let cost_price_100 := 100 * cost_price
  (selling_price_100 / cost_price_100) * 100 = 40 → 
  ((cost_price_100 - selling_price_100) / cost_price_100) * 100 = 60 := by
sorry

end selling_price_loss_percentage_l577_57792


namespace isosceles_triangle_perimeter_l577_57766

/-- An isosceles triangle with sides of 4cm and 8cm has a perimeter of 20cm -/
theorem isosceles_triangle_perimeter (a b c : ℝ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →  -- positive sides
  (a = 4 ∧ b = 8) ∨ (a = 8 ∧ b = 4) →  -- given side lengths
  (a = b ∨ b = c ∨ a = c) →  -- isosceles condition
  a + b + c = 20 :=
by sorry

end isosceles_triangle_perimeter_l577_57766


namespace joan_football_games_l577_57710

theorem joan_football_games (games_this_year games_total : ℕ) 
  (h1 : games_this_year = 4)
  (h2 : games_total = 13) :
  games_total - games_this_year = 9 := by
  sorry

end joan_football_games_l577_57710


namespace root_property_l577_57759

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem root_property (f : ℝ → ℝ) (x₀ : ℝ) 
  (h_odd : is_odd f) 
  (h_root : f x₀ = Real.exp x₀) :
  f (-x₀) * Real.exp (-x₀) + 1 = 0 := by
  sorry

end root_property_l577_57759
