import Mathlib

namespace infinite_geometric_series_first_term_l399_39932

/-- For an infinite geometric series with common ratio 1/3 and sum 18, the first term is 12. -/
theorem infinite_geometric_series_first_term 
  (r : ℝ) 
  (S : ℝ) 
  (h1 : r = 1/3) 
  (h2 : S = 18) 
  (h3 : S = a / (1 - r)) 
  (a : ℝ) : 
  a = 12 := by
sorry

end infinite_geometric_series_first_term_l399_39932


namespace f_sum_opposite_l399_39937

def f (x : ℝ) : ℝ := 5 * x^3

theorem f_sum_opposite : f 2012 + f (-2012) = 0 := by
  sorry

end f_sum_opposite_l399_39937


namespace smallest_side_of_triangle_l399_39901

/-- Given a triangle ABC with ∠A = 60°, ∠C = 45°, and side b = 4,
    prove that the smallest side of the triangle is 4√3 - 4. -/
theorem smallest_side_of_triangle (A B C : ℝ) (a b c : ℝ) :
  0 < a ∧ 0 < b ∧ 0 < c →
  A = 60 * π / 180 →
  C = 45 * π / 180 →
  b = 4 →
  a + b > c ∧ b + c > a ∧ c + a > b →
  A + B + C = π →
  c / Real.sin C = b / Real.sin B →
  c = 4 * Real.sqrt 3 - 4 :=
by sorry

end smallest_side_of_triangle_l399_39901


namespace warehouse_chocolate_count_l399_39912

/-- The number of large boxes in the warehouse -/
def num_large_boxes : ℕ := 150

/-- The number of small boxes in each large box -/
def small_boxes_per_large : ℕ := 45

/-- The number of chocolate bars in each small box -/
def chocolates_per_small : ℕ := 35

/-- The total number of chocolate bars in the warehouse -/
def total_chocolates : ℕ := num_large_boxes * small_boxes_per_large * chocolates_per_small

theorem warehouse_chocolate_count :
  total_chocolates = 236250 :=
by sorry

end warehouse_chocolate_count_l399_39912


namespace min_circles_cover_square_l399_39913

/-- A circle with radius 1 -/
structure UnitCircle where
  center : ℝ × ℝ

/-- A square with side length 2 -/
structure TwoSquare where
  bottomLeft : ℝ × ℝ

/-- A covering of a TwoSquare by UnitCircles -/
structure Covering where
  circles : List UnitCircle
  square : TwoSquare
  covers : ∀ (x y : ℝ), 
    (x - square.bottomLeft.1 ∈ Set.Icc 0 2 ∧ 
     y - square.bottomLeft.2 ∈ Set.Icc 0 2) → 
    ∃ (c : UnitCircle), c ∈ circles ∧ 
      (x - c.center.1)^2 + (y - c.center.2)^2 ≤ 1

/-- The theorem stating that the minimum number of unit circles 
    needed to cover a 2x2 square is 4 -/
theorem min_circles_cover_square :
  ∀ (cov : Covering), cov.circles.length ≥ 4 ∧ 
  ∃ (cov' : Covering), cov'.circles.length = 4 := by
  sorry


end min_circles_cover_square_l399_39913


namespace tangent_difference_l399_39935

noncomputable section

-- Define the curve
def curve (a : ℝ) (x : ℝ) : ℝ := a * x + 2 * Real.log (abs x)

-- Define the tangent line
def tangent_line (k : ℝ) (x : ℝ) : ℝ := k * x

-- Define the property of being a tangent line to the curve
def is_tangent (a k : ℝ) : Prop :=
  ∃ x : ℝ, x ≠ 0 ∧ curve a x = tangent_line k x ∧
    (∀ y : ℝ, y ≠ x → curve a y ≠ tangent_line k y)

-- Theorem statement
theorem tangent_difference (a k₁ k₂ : ℝ) :
  is_tangent a k₁ → is_tangent a k₂ → k₁ > k₂ → k₁ - k₂ = 4 / Real.exp 1 := by
  sorry

end

end tangent_difference_l399_39935


namespace denise_crayons_l399_39995

/-- The number of friends Denise shares her crayons with -/
def num_friends : ℕ := 30

/-- The number of crayons each friend receives -/
def crayons_per_friend : ℕ := 7

/-- The total number of crayons Denise has -/
def total_crayons : ℕ := num_friends * crayons_per_friend

/-- Theorem stating that Denise has 210 crayons -/
theorem denise_crayons : total_crayons = 210 := by
  sorry

end denise_crayons_l399_39995


namespace pink_yards_calculation_l399_39961

/-- The total number of yards dyed for the order -/
def total_yards : ℕ := 111421

/-- The number of yards dyed green -/
def green_yards : ℕ := 61921

/-- The number of yards dyed pink -/
def pink_yards : ℕ := total_yards - green_yards

theorem pink_yards_calculation : pink_yards = 49500 := by
  sorry

end pink_yards_calculation_l399_39961


namespace evaluate_trigonometric_expression_l399_39962

theorem evaluate_trigonometric_expression :
  let angle_27 : Real := 27 * Real.pi / 180
  let angle_18 : Real := 18 * Real.pi / 180
  let angle_63 : Real := 63 * Real.pi / 180
  (Real.cos angle_63 = Real.sin angle_27) →
  (angle_27 = 45 * Real.pi / 180 - angle_18) →
  (Real.cos angle_27 - Real.sqrt 2 * Real.sin angle_18) / Real.cos angle_63 = 1 := by
  sorry

end evaluate_trigonometric_expression_l399_39962


namespace ball_probabilities_l399_39938

/-- Represents the color of a ball -/
inductive BallColor
  | Yellow
  | White

/-- Represents the bag with balls -/
structure Bag :=
  (yellow : ℕ)
  (white : ℕ)

/-- The probability of drawing a white ball from the bag -/
def prob_white (bag : Bag) : ℚ :=
  bag.white / (bag.yellow + bag.white)

/-- The probability of drawing a yellow ball from the bag -/
def prob_yellow (bag : Bag) : ℚ :=
  bag.yellow / (bag.yellow + bag.white)

/-- The probability that two drawn balls have the same color -/
def prob_same_color (bag : Bag) : ℚ :=
  (prob_yellow bag)^2 + (prob_white bag)^2

theorem ball_probabilities (bag : Bag) 
  (h1 : bag.yellow = 1) 
  (h2 : bag.white = 2) : 
  prob_white bag = 2/3 ∧ prob_same_color bag = 5/9 := by
  sorry

#check ball_probabilities

end ball_probabilities_l399_39938


namespace manuscript_revision_l399_39989

/-- Proves that the number of pages revised twice is 15, given the manuscript typing conditions --/
theorem manuscript_revision (total_pages : ℕ) (revised_once : ℕ) (total_cost : ℕ) 
  (first_typing_cost : ℕ) (revision_cost : ℕ) :
  total_pages = 100 →
  revised_once = 35 →
  total_cost = 860 →
  first_typing_cost = 6 →
  revision_cost = 4 →
  ∃ (revised_twice : ℕ),
    revised_twice = 15 ∧
    total_cost = (total_pages - revised_once - revised_twice) * first_typing_cost +
                 revised_once * (first_typing_cost + revision_cost) +
                 revised_twice * (first_typing_cost + 2 * revision_cost) :=
by sorry


end manuscript_revision_l399_39989


namespace product_of_fractions_l399_39980

theorem product_of_fractions : (1 : ℚ) / 3 * 4 / 7 * 9 / 11 = 12 / 77 := by
  sorry

end product_of_fractions_l399_39980


namespace triangle_sequence_solution_l399_39982

theorem triangle_sequence_solution (b d c k : ℤ) 
  (h1 : b % d = 0)
  (h2 : c % k = 0)
  (h3 : b^2 + (b+2*d)^2 = (c+6*k)^2) :
  ∃ (b d c k : ℤ), c = 0 ∧ 
    b % d = 0 ∧ 
    c % k = 0 ∧ 
    b^2 + (b+2*d)^2 = (c+6*k)^2 :=
by sorry

end triangle_sequence_solution_l399_39982


namespace binary_101_equals_5_l399_39983

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (λ acc (i, bit) => acc + if bit then 2^i else 0) 0

theorem binary_101_equals_5 :
  binary_to_decimal [true, false, true] = 5 := by
  sorry

end binary_101_equals_5_l399_39983


namespace town_employment_theorem_l399_39916

/-- Represents the employment statistics of town X -/
structure TownEmployment where
  total_population : ℝ
  employed_percentage : ℝ
  employed_male_percentage : ℝ
  employed_female_percentage : ℝ

/-- The employment theorem for town X -/
theorem town_employment_theorem (stats : TownEmployment) 
  (h1 : stats.employed_male_percentage = 24)
  (h2 : stats.employed_female_percentage = 75) :
  stats.employed_percentage = 96 := by
  sorry

end town_employment_theorem_l399_39916


namespace distance_between_points_l399_39952

theorem distance_between_points (A B : ℝ) : 
  (|A| = 2 ∧ |B| = 7) → (|A - B| = 5 ∨ |A - B| = 9) := by
  sorry

end distance_between_points_l399_39952


namespace square_root_of_1024_l399_39946

theorem square_root_of_1024 (y : ℝ) (h1 : y > 0) (h2 : y^2 = 1024) : y = 32 := by
  sorry

end square_root_of_1024_l399_39946


namespace slope_of_AB_l399_39918

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 2*x

-- Define a point on the parabola
def point_on_parabola (P : ℝ × ℝ) : Prop :=
  P.1 = 1 ∧ P.2 = Real.sqrt 2 ∧ parabola P.1 P.2

-- Define complementary inclination angles
def complementary_angles (k : ℝ) (PA PB : ℝ → ℝ) : Prop :=
  (∀ x, PA x = k*(x - 1) + Real.sqrt 2) ∧
  (∀ x, PB x = -k*(x - 1) + Real.sqrt 2)

-- Define intersection points
def intersection_points (A B : ℝ × ℝ) (PA PB : ℝ → ℝ) : Prop :=
  parabola A.1 A.2 ∧ parabola B.1 B.2 ∧
  A.2 = PA A.1 ∧ B.2 = PB B.1

-- Theorem statement
theorem slope_of_AB (P A B : ℝ × ℝ) (k : ℝ) (PA PB : ℝ → ℝ) :
  point_on_parabola P →
  complementary_angles k PA PB →
  intersection_points A B PA PB →
  (B.2 - A.2) / (B.1 - A.1) = -2 - 2 * Real.sqrt 2 :=
sorry

end slope_of_AB_l399_39918


namespace second_question_percentage_l399_39955

/-- Represents the percentage of boys in a test scenario -/
structure TestPercentages where
  first : ℝ  -- Percentage who answered the first question correctly
  neither : ℝ  -- Percentage who answered neither question correctly
  both : ℝ  -- Percentage who answered both questions correctly

/-- 
Given the percentages of boys who answered the first question correctly, 
neither question correctly, and both questions correctly, 
proves that the percentage who answered the second question correctly is 55%.
-/
theorem second_question_percentage (p : TestPercentages) 
  (h1 : p.first = 75)
  (h2 : p.neither = 20)
  (h3 : p.both = 50) : 
  ∃ second : ℝ, second = 55 := by
  sorry

#check second_question_percentage

end second_question_percentage_l399_39955


namespace ab_greater_than_ac_l399_39973

theorem ab_greater_than_ac (a b c : ℝ) (h1 : c < b) (h2 : b < a) (h3 : a * c < 0) : a * b > a * c := by
  sorry

end ab_greater_than_ac_l399_39973


namespace ryan_coin_value_l399_39910

/-- Represents the types of coins Ryan has --/
inductive Coin
| Penny
| Nickel

/-- The value of a coin in cents --/
def coinValue : Coin → Nat
| Coin.Penny => 1
| Coin.Nickel => 5

/-- Ryan's coin collection --/
structure CoinCollection where
  pennies : Nat
  nickels : Nat
  total_coins : pennies + nickels = 17
  equal_count : pennies = nickels

theorem ryan_coin_value (c : CoinCollection) : 
  c.pennies * coinValue Coin.Penny + c.nickels * coinValue Coin.Nickel = 49 := by
  sorry

#check ryan_coin_value

end ryan_coin_value_l399_39910


namespace circle_properties_1_circle_properties_2_l399_39909

/-- Definition of a circle in the xy-plane -/
structure Circle where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  e : ℝ
  eq : ∀ x y : ℝ, x^2 + y^2 + a*x + b*y + c = 0

/-- The center and radius of a circle -/
structure CircleProperties where
  center : ℝ × ℝ
  radius : ℝ

/-- Theorem for the first circle -/
theorem circle_properties_1 :
  let C : Circle := {
    a := 2
    b := -4
    c := -3
    d := 1
    e := 1
    eq := by sorry
  }
  ∃ (props : CircleProperties), props.center = (-1, 2) ∧ props.radius = 2 * Real.sqrt 2 := by sorry

/-- Theorem for the second circle -/
theorem circle_properties_2 (m : ℝ) :
  let C : Circle := {
    a := 2*m
    b := 0
    c := 0
    d := 1
    e := 1
    eq := by sorry
  }
  ∃ (props : CircleProperties), props.center = (-m, 0) ∧ props.radius = |m| := by sorry

end circle_properties_1_circle_properties_2_l399_39909


namespace prob_non_first_class_l399_39922

theorem prob_non_first_class (A B C : ℝ) 
  (hA : A = 0.65) 
  (hB : B = 0.2) 
  (hC : C = 0.1) : 
  1 - A = 0.35 := by
  sorry

end prob_non_first_class_l399_39922


namespace imaginary_part_of_one_minus_i_squared_l399_39914

theorem imaginary_part_of_one_minus_i_squared :
  Complex.im ((1 - Complex.I) ^ 2) = -2 := by
  sorry

end imaginary_part_of_one_minus_i_squared_l399_39914


namespace marble_probability_l399_39949

/-- Given a box of marbles with the following properties:
  - There are 120 marbles in total
  - Each marble is either red, green, blue, or white
  - The probability of drawing a white marble is 1/4
  - The probability of drawing a green marble is 1/3
  This theorem proves that the probability of drawing either a red or blue marble is 5/12. -/
theorem marble_probability (total_marbles : ℕ) (p_white p_green : ℚ)
  (h_total : total_marbles = 120)
  (h_white : p_white = 1/4)
  (h_green : p_green = 1/3) :
  1 - (p_white + p_green) = 5/12 := by
  sorry

end marble_probability_l399_39949


namespace smallest_positive_b_squared_l399_39978

-- Define the circles w₁ and w₂
def w₁ (x y : ℝ) : Prop := x^2 + y^2 - 8*x - 6*y - 23 = 0
def w₂ (x y : ℝ) : Prop := x^2 + y^2 + 8*x - 6*y + 41 = 0

-- Define the condition for a point (x, y) to be on the line y = bx
def on_line (x y b : ℝ) : Prop := y = b * x

-- Define the condition for a circle to be externally tangent to w₂ and internally tangent to w₁
def tangent_condition (x y r : ℝ) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    w₁ x₁ y₁ ∧ w₂ x₂ y₂ ∧
    (x - x₂)^2 + (y - y₂)^2 = (r + Real.sqrt 10)^2 ∧
    (x - x₁)^2 + (y - y₁)^2 = (Real.sqrt 50 - r)^2

-- Main theorem
theorem smallest_positive_b_squared (b : ℝ) :
  (∀ b' : ℝ, b' > 0 ∧ b' < b →
    ¬∃ (x y r : ℝ), on_line x y b' ∧ tangent_condition x y r) →
  (∃ (x y r : ℝ), on_line x y b ∧ tangent_condition x y r) →
  b^2 = 21/16 :=
sorry

end smallest_positive_b_squared_l399_39978


namespace system_solution_l399_39939

theorem system_solution :
  ∃ (x y : ℚ), 
    (12 * x^2 + 4 * x * y + 3 * y^2 + 16 * x = -6) ∧
    (4 * x^2 - 12 * x * y + y^2 + 12 * x - 10 * y = -7) ∧
    (x = -3/4) ∧ (y = 1/2) := by
  sorry

end system_solution_l399_39939


namespace sum_first_four_eq_40_l399_39936

/-- A geometric sequence with a_2 = 6 and a_3 = -18 -/
def geometric_sequence (n : ℕ) : ℝ :=
  let q := -3  -- common ratio
  let a1 := -2 -- first term
  a1 * q^(n-1)

/-- The sum of the first four terms of the geometric sequence -/
def sum_first_four : ℝ :=
  (geometric_sequence 1) + (geometric_sequence 2) + (geometric_sequence 3) + (geometric_sequence 4)

/-- Theorem stating that the sum of the first four terms equals 40 -/
theorem sum_first_four_eq_40 : sum_first_four = 40 := by
  sorry

end sum_first_four_eq_40_l399_39936


namespace gcd_765432_654321_l399_39997

theorem gcd_765432_654321 : Nat.gcd 765432 654321 = 2 := by
  sorry

end gcd_765432_654321_l399_39997


namespace smallest_prime_cube_sum_fourth_power_l399_39964

theorem smallest_prime_cube_sum_fourth_power :
  ∃ (p : ℕ), Prime p ∧ 
  (∃ (a b : ℕ), a > 0 ∧ b > 0 ∧ a^2 + p^3 = b^4) ∧
  (∀ (q : ℕ), Prime q → q < p → 
    ¬∃ (c d : ℕ), c > 0 ∧ d > 0 ∧ c^2 + q^3 = d^4) ∧
  p = 23 := by
sorry

end smallest_prime_cube_sum_fourth_power_l399_39964


namespace intersection_point_on_fixed_line_l399_39968

-- Define the hyperbola C
structure Hyperbola where
  center : ℝ × ℝ
  left_focus : ℝ × ℝ
  eccentricity : ℝ
  left_vertex : ℝ × ℝ
  right_vertex : ℝ × ℝ

-- Define the intersection points M and N
structure IntersectionPoints where
  M : ℝ × ℝ
  N : ℝ × ℝ

-- Define the point P
def P (h : Hyperbola) (i : IntersectionPoints) : ℝ × ℝ := sorry

-- Theorem statement
theorem intersection_point_on_fixed_line 
  (h : Hyperbola) 
  (i : IntersectionPoints) :
  h.center = (0, 0) →
  h.left_focus = (-2 * Real.sqrt 5, 0) →
  h.eccentricity = Real.sqrt 5 →
  h.left_vertex = (-2, 0) →
  h.right_vertex = (2, 0) →
  (∃ (m : ℝ), i.M.1 = m * i.M.2 - 4 ∧ i.N.1 = m * i.N.2 - 4) →
  i.M.2 > 0 →
  (P h i).1 = -1 := by sorry

end intersection_point_on_fixed_line_l399_39968


namespace favorite_numbers_parity_l399_39956

/-- Represents a person's favorite number -/
structure FavoriteNumber where
  value : ℤ

/-- Represents whether a number is even or odd -/
inductive Parity
  | Even
  | Odd

/-- Returns the parity of an integer -/
def parity (n : ℤ) : Parity :=
  if n % 2 = 0 then Parity.Even else Parity.Odd

/-- The problem setup -/
structure FavoriteNumbers where
  jan : FavoriteNumber
  dan : FavoriteNumber
  anna : FavoriteNumber
  hana : FavoriteNumber
  h1 : parity (dan.value + 3 * jan.value) = Parity.Odd
  h2 : parity ((anna.value - hana.value) * 5) = Parity.Odd
  h3 : parity (dan.value * hana.value + 17) = Parity.Even

/-- The main theorem to prove -/
theorem favorite_numbers_parity (nums : FavoriteNumbers) :
  parity nums.dan.value = Parity.Odd ∧
  parity nums.hana.value = Parity.Odd ∧
  parity nums.anna.value = Parity.Even ∧
  parity nums.jan.value = Parity.Even :=
sorry

end favorite_numbers_parity_l399_39956


namespace sufficient_not_necessary_l399_39963

theorem sufficient_not_necessary (x : ℝ) : 
  (|x - 1| < 2 → x < 3) ∧ ¬(x < 3 → |x - 1| < 2) := by
  sorry

end sufficient_not_necessary_l399_39963


namespace tom_dance_lesson_payment_l399_39925

/-- The amount Tom pays for dance lessons -/
def tom_payment (total_lessons : ℕ) (cost_per_lesson : ℕ) (free_lessons : ℕ) : ℕ :=
  (total_lessons - free_lessons) * cost_per_lesson

/-- Proof that Tom pays $80 for dance lessons -/
theorem tom_dance_lesson_payment :
  tom_payment 10 10 2 = 80 := by
  sorry

end tom_dance_lesson_payment_l399_39925


namespace arithmetic_mean_of_ages_l399_39921

theorem arithmetic_mean_of_ages : 
  let ages : List ℝ := [18, 27, 35, 46]
  (ages.sum / ages.length : ℝ) = 31.5 := by
  sorry

end arithmetic_mean_of_ages_l399_39921


namespace median_invariance_l399_39996

def judges_scores := Fin 7 → ℝ
def reduced_scores := Fin 5 → ℝ

def median (scores : Fin n → ℝ) : ℝ :=
  sorry

def remove_extremes (scores : judges_scores) : reduced_scores :=
  sorry

theorem median_invariance (scores : judges_scores) :
  median scores = median (remove_extremes scores) :=
sorry

end median_invariance_l399_39996


namespace guess_who_i_am_l399_39948

theorem guess_who_i_am : ∃ x y : ℕ,
  120 = 4 * x ∧
  87 = y - 40 ∧
  x = 30 ∧
  y = 127 := by
sorry

end guess_who_i_am_l399_39948


namespace power_six_mod_72_l399_39986

theorem power_six_mod_72 : 6^700 % 72 = 0 := by
  sorry

end power_six_mod_72_l399_39986


namespace sum_fourth_powers_of_roots_l399_39953

/-- Given a cubic polynomial x^3 - x^2 + x - 3 = 0 with roots p, q, and r,
    prove that p^4 + q^4 + r^4 = 11 -/
theorem sum_fourth_powers_of_roots (p q r : ℂ) : 
  p^3 - p^2 + p - 3 = 0 → 
  q^3 - q^2 + q - 3 = 0 → 
  r^3 - r^2 + r - 3 = 0 → 
  p^4 + q^4 + r^4 = 11 := by
  sorry

end sum_fourth_powers_of_roots_l399_39953


namespace point_move_upward_l399_39966

def Point := ℝ × ℝ

def move_upward (p : Point) (units : ℝ) : Point :=
  (p.1, p.2 + units)

theorem point_move_upward (A B : Point) (h : ℝ) :
  A = (1, -2) →
  h = 1 →
  B = move_upward A h →
  B = (1, -1) := by
  sorry

end point_move_upward_l399_39966


namespace largest_power_of_three_dividing_expression_l399_39998

theorem largest_power_of_three_dividing_expression (m : ℕ) : 
  (∃ (k : ℕ), (3^k : ℕ) ∣ (2^(3^m) + 1)) ∧ 
  (∀ (k : ℕ), k > 2 → ¬((3^k : ℕ) ∣ (2^(3^m) + 1))) :=
sorry

end largest_power_of_three_dividing_expression_l399_39998


namespace number_greater_than_fifteen_l399_39979

theorem number_greater_than_fifteen (x : ℝ) : 0.4 * x > 0.8 * 5 + 2 → x > 15 := by
  sorry

end number_greater_than_fifteen_l399_39979


namespace target_has_six_more_tools_l399_39960

/-- The number of tools in the Walmart multitool -/
def walmart_tools : ℕ := 1 + 3 + 2

/-- The number of tools in the Target multitool -/
def target_tools : ℕ := 1 + (2 * 3) + 3 + 1

/-- The difference in the number of tools between Target and Walmart multitools -/
def tool_difference : ℕ := target_tools - walmart_tools

theorem target_has_six_more_tools : tool_difference = 6 := by
  sorry

end target_has_six_more_tools_l399_39960


namespace table_covered_area_l399_39987

/-- Represents a rectangular paper strip -/
structure Strip where
  length : ℕ
  width : ℕ

/-- Calculates the area of a strip -/
def stripArea (s : Strip) : ℕ := s.length * s.width

/-- Represents the overlap between two strips -/
structure Overlap where
  width : ℕ
  length : ℕ

/-- Calculates the area of an overlap -/
def overlapArea (o : Overlap) : ℕ := o.width * o.length

theorem table_covered_area (strip1 strip2 strip3 : Strip)
  (overlap12 overlap13 overlap23 : Overlap)
  (h1 : strip1.length = 12)
  (h2 : strip2.length = 15)
  (h3 : strip3.length = 9)
  (h4 : strip1.width = 2)
  (h5 : strip2.width = 2)
  (h6 : strip3.width = 2)
  (h7 : overlap12.width = 2)
  (h8 : overlap12.length = 2)
  (h9 : overlap13.width = 1)
  (h10 : overlap13.length = 2)
  (h11 : overlap23.width = 1)
  (h12 : overlap23.length = 2) :
  stripArea strip1 + stripArea strip2 + stripArea strip3 -
  (overlapArea overlap12 + overlapArea overlap13 + overlapArea overlap23) = 64 := by
  sorry

end table_covered_area_l399_39987


namespace die_roll_probability_l399_39985

/-- The probability of rolling a 5 on a standard die -/
def prob_five : ℚ := 1/6

/-- The number of rolls -/
def num_rolls : ℕ := 8

/-- The probability of not rolling a 5 in a single roll -/
def prob_not_five : ℚ := 1 - prob_five

theorem die_roll_probability : 
  1 - prob_not_five ^ num_rolls = 1288991/1679616 := by
sorry

end die_roll_probability_l399_39985


namespace living_room_set_cost_l399_39927

/-- The total cost of a living room set -/
def total_cost (sofa_cost armchair_cost coffee_table_cost : ℕ) (num_armchairs : ℕ) : ℕ :=
  sofa_cost + num_armchairs * armchair_cost + coffee_table_cost

/-- Theorem: The total cost of the specified living room set is $2,430 -/
theorem living_room_set_cost : total_cost 1250 425 330 2 = 2430 := by
  sorry

end living_room_set_cost_l399_39927


namespace simple_interest_calculation_l399_39951

/-- Simple interest calculation -/
theorem simple_interest_calculation 
  (principal : ℝ) 
  (rate : ℝ) 
  (time : ℝ) 
  (h1 : principal = 10000)
  (h2 : rate = 0.04)
  (h3 : time = 1) :
  principal * rate * time = 400 := by
  sorry

#check simple_interest_calculation

end simple_interest_calculation_l399_39951


namespace quadrilateral_area_is_31_l399_39950

/-- Represents a quadrilateral with vertices A, B, C, D and intersection point O of diagonals -/
structure Quadrilateral :=
  (A B C D O : ℝ × ℝ)

/-- The area of a quadrilateral given its side lengths and angle between diagonals -/
def area_quadrilateral (q : Quadrilateral) (AB BC CD DA : ℝ) (angle_COB : ℝ) : ℝ :=
  sorry

/-- Theorem stating that the area of the given quadrilateral is 31 -/
theorem quadrilateral_area_is_31 (q : Quadrilateral) 
  (h1 : area_quadrilateral q 10 6 8 2 (π/4) = 31) : 
  area_quadrilateral q 10 6 8 2 (π/4) = 31 := by
  sorry

end quadrilateral_area_is_31_l399_39950


namespace bacteria_growth_proof_l399_39908

/-- The growth factor of the bacteria colony per day -/
def growth_factor : ℕ := 3

/-- The initial number of bacteria -/
def initial_bacteria : ℕ := 5

/-- The threshold number of bacteria -/
def threshold : ℕ := 200

/-- The number of bacteria after n days -/
def bacteria_count (n : ℕ) : ℕ := initial_bacteria * growth_factor ^ n

/-- The smallest number of days for the bacteria count to exceed the threshold -/
def days_to_exceed_threshold : ℕ := 4

theorem bacteria_growth_proof :
  (∀ k : ℕ, k < days_to_exceed_threshold → bacteria_count k ≤ threshold) ∧
  bacteria_count days_to_exceed_threshold > threshold :=
sorry

end bacteria_growth_proof_l399_39908


namespace lcm_gcf_ratio_294_490_l399_39919

theorem lcm_gcf_ratio_294_490 : 
  (Nat.lcm 294 490) / (Nat.gcd 294 490) = 15 := by sorry

end lcm_gcf_ratio_294_490_l399_39919


namespace probability_is_one_twelfth_l399_39945

/-- A rectangle in the 2D plane --/
structure Rectangle where
  x_min : ℝ
  x_max : ℝ
  y_min : ℝ
  y_max : ℝ
  h_x : x_min < x_max
  h_y : y_min < y_max

/-- The probability of a point satisfying certain conditions within a rectangle --/
def probability_in_rectangle (R : Rectangle) (P : ℝ × ℝ → Prop) : ℝ :=
  sorry

/-- The specific rectangle in the problem --/
def problem_rectangle : Rectangle := {
  x_min := 0
  x_max := 6
  y_min := 0
  y_max := 2
  h_x := by norm_num
  h_y := by norm_num
}

/-- The condition that needs to be satisfied --/
def condition (p : ℝ × ℝ) : Prop :=
  p.1 < p.2 ∧ p.1 + p.2 < 2

/-- The main theorem --/
theorem probability_is_one_twelfth :
  probability_in_rectangle problem_rectangle condition = 1/12 := by
  sorry

end probability_is_one_twelfth_l399_39945


namespace min_probability_is_601_1225_l399_39917

/-- The number of cards in the deck -/
def num_cards : ℕ := 52

/-- The probability that Charlie and Jane are on the same team, given that they draw cards a and a+11 -/
def p (a : ℕ) : ℚ :=
  let remaining_combinations := (num_cards - 2).choose 2
  let lower_team_combinations := (a - 1).choose 2
  let higher_team_combinations := (num_cards - (a + 11) - 1).choose 2
  (lower_team_combinations + higher_team_combinations : ℚ) / remaining_combinations

/-- The minimum value of a for which p(a) is at least 1/2 -/
def min_a : ℕ := 36

theorem min_probability_is_601_1225 :
  p min_a = 601 / 1225 ∧ ∀ a : ℕ, 1 ≤ a ∧ a ≤ num_cards - 11 → p a ≥ 1 / 2 → p a ≥ p min_a :=
sorry

end min_probability_is_601_1225_l399_39917


namespace parallel_iff_parallel_sum_l399_39924

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

/-- Two vectors are parallel if one is a scalar multiple of the other -/
def IsParallel (u v : V) : Prop :=
  ∃ (k : ℝ), v = k • u ∨ u = k • v

theorem parallel_iff_parallel_sum {a b : V} (ha : a ≠ 0) (hb : b ≠ 0) :
  IsParallel a b ↔ IsParallel a (a + b) :=
sorry

end parallel_iff_parallel_sum_l399_39924


namespace oil_bill_problem_l399_39902

/-- The oil bill problem -/
theorem oil_bill_problem (january_bill : ℝ) (february_bill : ℝ) (additional_amount : ℝ) :
  january_bill = 180 →
  february_bill / january_bill = 5 / 4 →
  (february_bill + additional_amount) / january_bill = 3 / 2 →
  additional_amount = 45 := by
  sorry

end oil_bill_problem_l399_39902


namespace complement_A_intersect_B_l399_39907

def U : Set Nat := {1, 2, 3, 4, 5}
def A : Set Nat := {1, 3, 4}
def B : Set Nat := {2, 3}

theorem complement_A_intersect_B :
  (Aᶜ ∩ B) = {2} :=
sorry

end complement_A_intersect_B_l399_39907


namespace min_sum_with_product_144_l399_39976

theorem min_sum_with_product_144 :
  (∃ (a b : ℤ), a * b = 144 ∧ a + b = -145) ∧
  (∀ (a b : ℤ), a * b = 144 → a + b ≥ -145) := by
sorry

end min_sum_with_product_144_l399_39976


namespace water_overflow_l399_39930

/-- Given a tap producing water at a constant rate and a water tank with a fixed capacity,
    calculate the amount of water that overflows after a certain time. -/
theorem water_overflow (flow_rate : ℕ) (time : ℕ) (tank_capacity : ℕ) : 
  flow_rate = 200 → time = 24 → tank_capacity = 4000 → 
  flow_rate * time - tank_capacity = 800 := by
  sorry

end water_overflow_l399_39930


namespace equation_solution_l399_39926

theorem equation_solution : 
  ∃! y : ℝ, (y^3 + 3*y^2) / (y^2 + 5*y + 6) + y = -8 ∧ y^2 + 5*y + 6 ≠ 0 := by
  sorry

end equation_solution_l399_39926


namespace student_correct_sums_l399_39994

theorem student_correct_sums (total : ℕ) (correct : ℕ) (wrong : ℕ) : 
  total = 24 → wrong = 2 * correct → total = correct + wrong → correct = 8 := by
  sorry

end student_correct_sums_l399_39994


namespace distance_P_to_y_axis_l399_39906

/-- The distance from a point to the y-axis is the absolute value of its x-coordinate. -/
def distance_to_y_axis (x y : ℝ) : ℝ := |x|

/-- The point P has coordinates (3, -5). -/
def P : ℝ × ℝ := (3, -5)

/-- Theorem: The distance from point P(3, -5) to the y-axis is 3. -/
theorem distance_P_to_y_axis : distance_to_y_axis P.1 P.2 = 3 := by
  sorry

end distance_P_to_y_axis_l399_39906


namespace best_fitting_model_l399_39988

/-- Represents a regression model with its coefficient of determination (R²) -/
structure RegressionModel where
  r_squared : ℝ
  h_r_squared_nonneg : 0 ≤ r_squared
  h_r_squared_le_one : r_squared ≤ 1

/-- Given four regression models, proves that the model with the highest R² has the best fitting effect -/
theorem best_fitting_model
  (model1 model2 model3 model4 : RegressionModel)
  (h1 : model1.r_squared = 0.98)
  (h2 : model2.r_squared = 0.80)
  (h3 : model3.r_squared = 0.50)
  (h4 : model4.r_squared = 0.25) :
  model1.r_squared = max model1.r_squared (max model2.r_squared (max model3.r_squared model4.r_squared)) :=
sorry

end best_fitting_model_l399_39988


namespace eighth_term_value_l399_39941

theorem eighth_term_value (S : ℕ → ℕ) (a : ℕ → ℕ) 
  (h : ∀ n : ℕ, S n = n^2) : a 8 = 15 := by
  sorry

end eighth_term_value_l399_39941


namespace parallel_vectors_magnitude_l399_39977

def a (m : ℝ) : ℝ × ℝ := (1, m + 2)
def b (m : ℝ) : ℝ × ℝ := (m, -1)

theorem parallel_vectors_magnitude (m : ℝ) :
  (∃ k : ℝ, k ≠ 0 ∧ a m = k • b m) →
  Real.sqrt ((b m).1^2 + (b m).2^2) = Real.sqrt 2 :=
by sorry

end parallel_vectors_magnitude_l399_39977


namespace parabola_shift_theorem_l399_39934

/-- Represents a parabola in the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Shifts a parabola horizontally and vertically -/
def shift_parabola (p : Parabola) (h : ℝ) (v : ℝ) : Parabola :=
  { a := p.a,
    b := -2 * p.a * h + p.b,
    c := p.a * h^2 - p.b * h + p.c + v }

theorem parabola_shift_theorem (p : Parabola) :
  p.a = -2 ∧ p.b = 0 ∧ p.c = 0 →
  (shift_parabola (shift_parabola p 1 0) 0 (-3)) = { a := -2, b := -4, c := -5 } := by
  sorry

end parabola_shift_theorem_l399_39934


namespace dried_mushroom_mass_dried_mushroom_mass_44kg_l399_39931

/-- Given fresh mushrooms with 90% water content and dried mushrooms with 12% water content,
    calculate the mass of dried mushrooms obtained from a given mass of fresh mushrooms. -/
theorem dried_mushroom_mass (fresh_mass : ℝ) : 
  fresh_mass > 0 →
  (fresh_mass * (1 - 0.9)) / (1 - 0.12) = 5 →
  fresh_mass = 44 := by
sorry

/-- The mass of dried mushrooms obtained from 44 kg of fresh mushrooms is 5 kg. -/
theorem dried_mushroom_mass_44kg : 
  (44 * (1 - 0.9)) / (1 - 0.12) = 5 := by
sorry

end dried_mushroom_mass_dried_mushroom_mass_44kg_l399_39931


namespace root_product_sum_l399_39967

theorem root_product_sum (a b c : ℂ) : 
  (5 * a^3 - 4 * a^2 + 15 * a - 12 = 0) →
  (5 * b^3 - 4 * b^2 + 15 * b - 12 = 0) →
  (5 * c^3 - 4 * c^2 + 15 * c - 12 = 0) →
  a * b + a * c + b * c = -3 := by
sorry

end root_product_sum_l399_39967


namespace probability_two_girls_l399_39928

def total_students : ℕ := 5
def num_boys : ℕ := 2
def num_girls : ℕ := 3
def students_selected : ℕ := 2

theorem probability_two_girls :
  (Nat.choose num_girls students_selected) / (Nat.choose total_students students_selected) = 3 / 10 := by
  sorry

end probability_two_girls_l399_39928


namespace multiply_72515_9999_l399_39972

theorem multiply_72515_9999 : 72515 * 9999 = 725077485 := by sorry

end multiply_72515_9999_l399_39972


namespace ladder_length_l399_39905

/-- The length of a ladder given specific conditions --/
theorem ladder_length : ∃ (L : ℝ), 
  (∀ (H : ℝ), L^2 = H^2 + 5^2) ∧ 
  (∀ (H : ℝ), L^2 = (H - 4)^2 + 10.658966865741546^2) ∧
  (abs (L - 14.04) < 0.01) := by
  sorry

end ladder_length_l399_39905


namespace exists_valid_coloring_l399_39991

/-- A coloring of integers from 1 to 2014 using four colors -/
def Coloring := Fin 2014 → Fin 4

/-- An arithmetic progression of length 11 within the range 1 to 2014 -/
structure ArithmeticProgression :=
  (start : Fin 2014)
  (step : Nat)
  (h : ∀ i : Fin 11, (start.val : ℕ) + i.val * step ≤ 2014)

/-- A coloring is valid if no arithmetic progression of length 11 is monochromatic -/
def ValidColoring (c : Coloring) : Prop :=
  ∀ ap : ArithmeticProgression, ∃ i j : Fin 11, i ≠ j ∧ 
    c ⟨(ap.start.val + i.val * ap.step : ℕ), by sorry⟩ ≠ 
    c ⟨(ap.start.val + j.val * ap.step : ℕ), by sorry⟩

/-- There exists a valid coloring of integers from 1 to 2014 using four colors -/
theorem exists_valid_coloring : ∃ c : Coloring, ValidColoring c := by sorry

end exists_valid_coloring_l399_39991


namespace kayla_apples_l399_39981

theorem kayla_apples (total : ℕ) (kylie : ℕ) (kayla : ℕ) : 
  total = 340 →
  kayla = 4 * kylie + 10 →
  total = kylie + kayla →
  kayla = 274 := by
sorry

end kayla_apples_l399_39981


namespace no_simultaneous_divisibility_l399_39971

theorem no_simultaneous_divisibility (k n : ℕ) (hk : k > 1) (hn : n > 1)
  (hk_odd : Odd k) (hn_odd : Odd n)
  (h_exists : ∃ a : ℕ, k ∣ 2^a + 1 ∧ n ∣ 2^a - 1) :
  ¬∃ b : ℕ, k ∣ 2^b - 1 ∧ n ∣ 2^b + 1 := by
  sorry

end no_simultaneous_divisibility_l399_39971


namespace negation_of_cube_odd_is_odd_l399_39992

theorem negation_of_cube_odd_is_odd :
  ¬(∀ x : ℤ, Odd x → Odd (x^3)) ↔ ∃ x : ℤ, Odd x ∧ ¬Odd (x^3) :=
sorry

end negation_of_cube_odd_is_odd_l399_39992


namespace rachel_money_left_l399_39957

theorem rachel_money_left (initial_amount : ℚ) : 
  initial_amount = 200 →
  initial_amount - (initial_amount / 4 + initial_amount / 5 + initial_amount / 10 + initial_amount / 8) = 65 := by
  sorry

end rachel_money_left_l399_39957


namespace next_simultaneous_event_l399_39943

/-- Represents the number of minutes between events for a clock -/
structure ClockEvents where
  lightup : ℕ  -- Number of minutes between light-ups
  ring : ℕ     -- Number of minutes between rings

/-- Calculates the time until the next simultaneous light-up and ring -/
def timeToNextSimultaneousEvent (c : ClockEvents) : ℕ :=
  Nat.lcm c.lightup c.ring

/-- The theorem stating that for a clock that lights up every 9 minutes
    and rings every 60 minutes, the next simultaneous event occurs after 180 minutes -/
theorem next_simultaneous_event :
  let c := ClockEvents.mk 9 60
  timeToNextSimultaneousEvent c = 180 := by
  sorry

end next_simultaneous_event_l399_39943


namespace cylinder_surface_area_from_hemisphere_l399_39974

/-- Given a hemisphere with total surface area Q and a cylinder with the same base and volume,
    prove that the total surface area of the cylinder is (10/9)Q. -/
theorem cylinder_surface_area_from_hemisphere (Q : ℝ) (R : ℝ) (h : ℝ) :
  Q > 0 →  -- Ensure Q is positive
  Q = 3 * Real.pi * R^2 →  -- Total surface area of hemisphere
  h = (2/3) * R →  -- Height of cylinder with same volume
  (2 * Real.pi * R^2 + 2 * Real.pi * R * h) = (10/9) * Q := by
  sorry

end cylinder_surface_area_from_hemisphere_l399_39974


namespace banana_cantaloupe_cost_l399_39970

/-- Represents the prices of fruits in dollars -/
structure FruitPrices where
  apples : ℝ
  bananas : ℝ
  cantaloupe : ℝ
  dates : ℝ

/-- The conditions of the fruit purchase problem -/
def fruitPurchaseConditions (p : FruitPrices) : Prop :=
  p.apples + p.bananas + p.cantaloupe + p.dates = 30 ∧
  p.dates = 3 * p.apples ∧
  p.cantaloupe = p.apples - p.bananas

/-- The theorem stating the cost of bananas and cantaloupe -/
theorem banana_cantaloupe_cost (p : FruitPrices) 
  (h : fruitPurchaseConditions p) : 
  p.bananas + p.cantaloupe = 6 := by
  sorry


end banana_cantaloupe_cost_l399_39970


namespace greatest_b_value_l399_39958

theorem greatest_b_value (b : ℝ) : 
  (∀ x : ℝ, -x^2 + 9*x - 18 ≥ 0 → x ≤ 6) ∧ 
  (-6^2 + 9*6 - 18 ≥ 0) := by
  sorry

end greatest_b_value_l399_39958


namespace simplify_sqrt_sum_l399_39954

theorem simplify_sqrt_sum : 
  Real.sqrt (12 + 6 * Real.sqrt 3) + Real.sqrt (12 - 6 * Real.sqrt 3) = 6 := by
  sorry

end simplify_sqrt_sum_l399_39954


namespace smallest_banana_total_l399_39999

/-- Represents the number of bananas taken by each monkey -/
structure BananaTaken where
  first : ℕ
  second : ℕ
  third : ℕ
  fourth : ℕ

/-- Represents the final distribution of bananas among the monkeys -/
structure BananaDistribution where
  first : ℕ
  second : ℕ
  third : ℕ
  fourth : ℕ

/-- Checks if the given banana distribution satisfies the problem conditions -/
def isValidDistribution (taken : BananaTaken) (dist : BananaDistribution) : Prop :=
  dist.first = taken.first / 2 + taken.second / 6 + taken.third / 9 + 7 * taken.fourth / 72 ∧
  dist.second = taken.first / 6 + taken.second / 3 + taken.third / 9 + 7 * taken.fourth / 72 ∧
  dist.third = taken.first / 6 + taken.second / 6 + taken.third / 6 + 7 * taken.fourth / 72 ∧
  dist.fourth = taken.first / 6 + taken.second / 6 + taken.third / 9 + taken.fourth / 8 ∧
  dist.first = 4 * dist.fourth ∧
  dist.second = 3 * dist.fourth ∧
  dist.third = 2 * dist.fourth

/-- The main theorem stating the smallest possible total number of bananas -/
theorem smallest_banana_total :
  ∀ taken : BananaTaken,
  ∀ dist : BananaDistribution,
  isValidDistribution taken dist →
  taken.first + taken.second + taken.third + taken.fourth ≥ 432 :=
by sorry

end smallest_banana_total_l399_39999


namespace consecutive_integers_around_sqrt_40_l399_39990

theorem consecutive_integers_around_sqrt_40 (a b : ℤ) : 
  (a + 1 = b) → (a < Real.sqrt 40) → (Real.sqrt 40 < b) → (a + b = 13) := by
  sorry

end consecutive_integers_around_sqrt_40_l399_39990


namespace perfect_pairing_S8_exists_no_perfect_pairing_S5_l399_39920

def Sn (n : ℕ) : Set ℕ := {x | 1 ≤ x ∧ x ≤ 2*n}

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m^2

def is_perfect_pairing (n : ℕ) (pairing : List (ℕ × ℕ)) : Prop :=
  (∀ (pair : ℕ × ℕ), pair ∈ pairing → pair.1 ∈ Sn n ∧ pair.2 ∈ Sn n) ∧
  (∀ x ∈ Sn n, ∃ pair ∈ pairing, x = pair.1 ∨ x = pair.2) ∧
  (∀ pair ∈ pairing, is_perfect_square (pair.1 + pair.2)) ∧
  pairing.length = n

theorem perfect_pairing_S8_exists : ∃ pairing : List (ℕ × ℕ), is_perfect_pairing 8 pairing :=
sorry

theorem no_perfect_pairing_S5 : ¬∃ pairing : List (ℕ × ℕ), is_perfect_pairing 5 pairing :=
sorry

end perfect_pairing_S8_exists_no_perfect_pairing_S5_l399_39920


namespace monic_cubic_polynomial_sum_l399_39923

/-- A monic cubic polynomial -/
def monicCubicPolynomial (p : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, ∀ x, p x = x^3 + a*x^2 + b*x + c

/-- The main theorem -/
theorem monic_cubic_polynomial_sum (p : ℝ → ℝ) 
  (h_monic : monicCubicPolynomial p)
  (h1 : p 1 = 10)
  (h2 : p 2 = 20)
  (h3 : p 3 = 30) :
  p 0 + p 5 = 68 := by sorry

end monic_cubic_polynomial_sum_l399_39923


namespace simplify_trigonometric_expression_sector_central_angle_l399_39959

-- Problem 1
theorem simplify_trigonometric_expression (x : ℝ) :
  (1 + Real.sin x) / Real.cos x * Real.sin (2 * x) / (2 * (Real.cos (π / 4 - x / 2))^2) = 2 * Real.sin x :=
sorry

-- Problem 2
theorem sector_central_angle (r α : ℝ) (h1 : 2 * r + α * r = 4) (h2 : 1/2 * α * r^2 = 1) :
  α = 2 :=
sorry

end simplify_trigonometric_expression_sector_central_angle_l399_39959


namespace condition_neither_sufficient_nor_necessary_l399_39940

-- Define a geometric sequence
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (r : ℝ), ∀ (n : ℕ), a (n + 1) = r * a n

-- Define an increasing sequence
def increasing_sequence (a : ℕ → ℝ) : Prop :=
  ∀ (n : ℕ), a (n + 1) > a n

-- Define the condition 8a_2 - a_5 = 0
def condition (a : ℕ → ℝ) : Prop :=
  8 * a 2 - a 5 = 0

-- Theorem stating that the condition is neither sufficient nor necessary
theorem condition_neither_sufficient_nor_necessary :
  ¬(∀ (a : ℕ → ℝ), geometric_sequence a → condition a → increasing_sequence a) ∧
  ¬(∀ (a : ℕ → ℝ), geometric_sequence a → increasing_sequence a → condition a) :=
sorry

end condition_neither_sufficient_nor_necessary_l399_39940


namespace abs_sum_equals_two_l399_39903

theorem abs_sum_equals_two (a b c : ℤ) 
  (h : |a - b|^19 + |c - a|^2010 = 1) : 
  |a - b| + |b - c| + |c - a| = 2 := by
sorry

end abs_sum_equals_two_l399_39903


namespace painted_cubes_count_l399_39904

/-- Given a cube with side length 4, composed of unit cubes, where the interior 2x2x2 cube is unpainted,
    the number of unit cubes with at least one face painted is 56. -/
theorem painted_cubes_count (n : ℕ) (h1 : n = 4) : 
  n^3 - (n - 2)^3 = 56 := by
  sorry

end painted_cubes_count_l399_39904


namespace mt_everest_summit_distance_l399_39969

/-- The distance from the base camp to the summit of Mt. Everest --/
def summit_distance : ℝ := 5800

/-- Hillary's climbing rate in feet per hour --/
def hillary_climb_rate : ℝ := 800

/-- Eddy's climbing rate in feet per hour --/
def eddy_climb_rate : ℝ := 500

/-- Hillary's descent rate in feet per hour --/
def hillary_descent_rate : ℝ := 1000

/-- The distance in feet that Hillary stops short of the summit --/
def hillary_stop_distance : ℝ := 1000

/-- The time in hours from start until Hillary and Eddy pass each other --/
def time_until_pass : ℝ := 6

theorem mt_everest_summit_distance :
  summit_distance = 
    hillary_climb_rate * time_until_pass + hillary_stop_distance ∧
  summit_distance = 
    eddy_climb_rate * time_until_pass + 
    hillary_descent_rate * (time_until_pass - hillary_climb_rate * time_until_pass / hillary_descent_rate) :=
by sorry

end mt_everest_summit_distance_l399_39969


namespace special_rhombus_center_distance_l399_39942

/-- A rhombus with a specific acute angle and projection length. -/
structure SpecialRhombus where
  /-- The acute angle of the rhombus in degrees -/
  acute_angle : ℝ
  /-- The length of the projection of side AB onto side AD -/
  projection_length : ℝ
  /-- The acute angle is 45 degrees -/
  angle_is_45 : acute_angle = 45
  /-- The projection length is 12 -/
  projection_is_12 : projection_length = 12

/-- The distance from the center of the rhombus to any side -/
def center_to_side_distance (r : SpecialRhombus) : ℝ := 6

/-- 
Theorem: In a rhombus where the acute angle is 45° and the projection of one side 
onto an adjacent side is 12, the distance from the center to any side is 6.
-/
theorem special_rhombus_center_distance (r : SpecialRhombus) : 
  center_to_side_distance r = 6 := by
  sorry

end special_rhombus_center_distance_l399_39942


namespace find_V_l399_39947

-- Define the relationship between U, V, and W
def relationship (k : ℝ) (U V W : ℝ) : Prop :=
  U = k * (V / W)

-- Define the theorem
theorem find_V (k : ℝ) :
  relationship k 16 2 (1/4) →
  relationship k 25 (5/2) (1/5) :=
by sorry

end find_V_l399_39947


namespace segment_area_approx_l399_39929

/-- Represents a circular segment -/
structure CircularSegment where
  arcLength : ℝ
  chordLength : ℝ

/-- Calculates the area of a circular segment -/
noncomputable def segmentArea (segment : CircularSegment) : ℝ :=
  sorry

/-- Theorem stating that the area of the given circular segment is approximately 14.6 -/
theorem segment_area_approx :
  let segment : CircularSegment := { arcLength := 10, chordLength := 8 }
  abs (segmentArea segment - 14.6) < 0.1 := by
  sorry

end segment_area_approx_l399_39929


namespace geometric_sequence_parabola_vertex_l399_39900

-- Define a geometric sequence
def is_geometric_sequence (a b c d : ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ b = a * r ∧ c = b * r ∧ d = c * r

-- Define the parabola
def parabola (x : ℝ) : ℝ := x^2 - 2*x + 3

-- Define the vertex of a parabola
def is_vertex (x y : ℝ) : Prop :=
  parabola x = y ∧ ∀ t : ℝ, parabola t ≥ y

-- Theorem statement
theorem geometric_sequence_parabola_vertex (a b c d : ℝ) :
  is_geometric_sequence a b c d →
  is_vertex b c →
  a * d = 2 := by sorry

end geometric_sequence_parabola_vertex_l399_39900


namespace imaginary_part_of_one_over_one_minus_i_l399_39911

theorem imaginary_part_of_one_over_one_minus_i :
  Complex.im (1 / (1 - Complex.I)) = 1 / 2 := by
  sorry

end imaginary_part_of_one_over_one_minus_i_l399_39911


namespace prob_different_colors_l399_39993

/-- Represents the color of a chip -/
inductive ChipColor
  | Blue
  | Red
  | Yellow

/-- Represents the state of the bag after the first draw -/
structure BagState where
  blue : Nat
  red : Nat
  yellow : Nat

/-- The initial state of the bag -/
def initialBag : BagState :=
  { blue := 6, red := 5, yellow := 4 }

/-- The state of the bag after drawing a blue chip -/
def bagAfterBlue : BagState :=
  { blue := 7, red := 5, yellow := 4 }

/-- The probability of drawing two chips of different colors -/
def probDifferentColors : ℚ := 593 / 900

/-- The theorem stating the probability of drawing two chips of different colors -/
theorem prob_different_colors :
  let totalChips := initialBag.blue + initialBag.red + initialBag.yellow
  let probFirstBlue := initialBag.blue / totalChips
  let probFirstRed := initialBag.red / totalChips
  let probFirstYellow := initialBag.yellow / totalChips
  let probSecondNotBlueAfterBlue := (bagAfterBlue.red + bagAfterBlue.yellow) / (bagAfterBlue.blue + bagAfterBlue.red + bagAfterBlue.yellow)
  let probSecondNotRedAfterRed := (initialBag.blue + initialBag.yellow) / totalChips
  let probSecondNotYellowAfterYellow := (initialBag.blue + initialBag.red) / totalChips
  probFirstBlue * probSecondNotBlueAfterBlue +
  probFirstRed * probSecondNotRedAfterRed +
  probFirstYellow * probSecondNotYellowAfterYellow = probDifferentColors :=
by
  sorry


end prob_different_colors_l399_39993


namespace rectangle_area_l399_39984

/-- Proves that a rectangle with width 4 inches and perimeter 30 inches has an area of 44 square inches -/
theorem rectangle_area (width : ℝ) (perimeter : ℝ) (height : ℝ) (area : ℝ) : 
  width = 4 →
  perimeter = 30 →
  perimeter = 2 * (width + height) →
  area = width * height →
  area = 44 :=
by
  sorry

end rectangle_area_l399_39984


namespace donation_amount_l399_39915

def barbara_stuffed_animals : ℕ := 9
def trish_stuffed_animals : ℕ := 2 * barbara_stuffed_animals
def sam_stuffed_animals : ℕ := barbara_stuffed_animals + 5

def barbara_price : ℚ := 2
def trish_price : ℚ := (3 : ℚ) / 2
def sam_price : ℚ := (5 : ℚ) / 2

def total_donation : ℚ := 
  barbara_stuffed_animals * barbara_price + 
  trish_stuffed_animals * trish_price + 
  sam_stuffed_animals * sam_price

theorem donation_amount : total_donation = 80 := by
  sorry

end donation_amount_l399_39915


namespace min_value_theorem_l399_39933

theorem min_value_theorem (x y : ℝ) (h1 : x^2 + y^2 = 2) (h2 : |x| ≠ |y|) :
  ∃ (min : ℝ), min = 1 ∧ ∀ (z : ℝ), z = (1 / (x + y)^2) + (1 / (x - y)^2) → z ≥ min :=
by sorry

end min_value_theorem_l399_39933


namespace complex_multiplication_l399_39975

theorem complex_multiplication : (1 + Complex.I) ^ 6 * (1 - Complex.I) = -8 - 8 * Complex.I := by
  sorry

end complex_multiplication_l399_39975


namespace extended_line_point_l399_39965

-- Define points A and B
def A : ℝ × ℝ := (3, 1)
def B : ℝ × ℝ := (17, 7)

-- Define the ratio of BC to AB
def ratio : ℚ := 2 / 5

-- Define point C
def C : ℝ × ℝ := (22.6, 9.4)

-- Theorem statement
theorem extended_line_point : 
  let AB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)
  let BC : ℝ × ℝ := (C.1 - B.1, C.2 - B.2)
  BC.1 = ratio * AB.1 ∧ BC.2 = ratio * AB.2 := by sorry

end extended_line_point_l399_39965


namespace hyperbola_properties_l399_39944

/-- Given a hyperbola with equation x²/2 - y² = 1, prove its asymptotes and eccentricity -/
theorem hyperbola_properties :
  let h : ℝ → ℝ → Prop := λ x y ↦ x^2 / 2 - y^2 = 1
  ∃ (a b c : ℝ),
    a^2 = 2 ∧ 
    b^2 = 1 ∧
    c^2 = a^2 + b^2 ∧
    (∀ x y, h x y ↔ x^2 / a^2 - y^2 / b^2 = 1) ∧
    (∀ x, (h x (x * (b / a)) ∨ h x (-x * (b / a))) ↔ x ≠ 0) ∧
    c / a = Real.sqrt 6 / 2 :=
by sorry

end hyperbola_properties_l399_39944
