import Mathlib

namespace range_of_square_root_set_l867_86737

theorem range_of_square_root_set (A : Set ℝ) (a : ℝ) :
  (A.Nonempty) →
  (A = {x : ℝ | x^2 = a}) →
  (∃ (y : ℝ), ∀ (x : ℝ), x ∈ A ↔ y ≤ x ∧ x^2 = a) :=
by sorry

end range_of_square_root_set_l867_86737


namespace largest_n_is_max_l867_86714

/-- The largest value of n for which 3x^2 + nx + 108 can be factored as the product of two linear factors with integer coefficients -/
def largest_n : ℕ := 325

/-- A polynomial of the form 3x^2 + nx + 108 -/
def polynomial (n : ℕ) (x : ℝ) : ℝ := 3 * x^2 + n * x + 108

/-- Predicate to check if a polynomial can be factored as the product of two linear factors with integer coefficients -/
def can_be_factored (n : ℕ) : Prop :=
  ∃ (a b : ℤ), ∀ (x : ℝ), polynomial n x = (3 * x + a) * (x + b)

/-- Theorem stating that largest_n is the largest value of n for which the polynomial can be factored -/
theorem largest_n_is_max :
  can_be_factored largest_n ∧
  ∀ (m : ℕ), m > largest_n → ¬(can_be_factored m) :=
sorry

end largest_n_is_max_l867_86714


namespace unique_polynomial_function_l867_86778

/-- A polynomial function over ℝ -/
def PolynomialFunction := ℝ → ℝ

/-- Predicate to check if a function is a polynomial of degree ≥ 1 -/
def IsPolynomialDegreeGEOne (f : PolynomialFunction) : Prop := sorry

/-- The conditions that the polynomial function must satisfy -/
def SatisfiesConditions (f : PolynomialFunction) : Prop :=
  IsPolynomialDegreeGEOne f ∧
  (∀ x : ℝ, f (x^2) = (f x)^3) ∧
  (∀ x : ℝ, f (f x) = f x)

/-- Theorem stating that there exists exactly one polynomial function satisfying the conditions -/
theorem unique_polynomial_function :
  ∃! f : PolynomialFunction, SatisfiesConditions f := by sorry

end unique_polynomial_function_l867_86778


namespace ellipse_inequality_l867_86780

theorem ellipse_inequality (a b x y : ℝ) (ha : a > 0) (hb : b > 0)
  (h : x^2 / a^2 + y^2 / b^2 ≤ 1) : a^2 + b^2 ≥ (x + y)^2 := by
  sorry

end ellipse_inequality_l867_86780


namespace right_triangle_leg_length_l867_86734

theorem right_triangle_leg_length 
  (A B C : ℝ × ℝ) 
  (is_right_triangle : (A.1 - B.1) * (A.1 - C.1) + (A.2 - B.2) * (A.2 - C.2) = 0) 
  (leg_length : Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 1) 
  (hypotenuse_length : Real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2) = Real.sqrt 5) :
  Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2) = 2 := by
  sorry

end right_triangle_leg_length_l867_86734


namespace reams_needed_l867_86711

-- Define the constants
def stories_per_week : ℕ := 3
def pages_per_story : ℕ := 50
def novel_pages_per_year : ℕ := 1200
def pages_per_sheet : ℕ := 2
def sheets_per_ream : ℕ := 500
def weeks_in_year : ℕ := 52
def weeks_to_calculate : ℕ := 12

-- Theorem to prove
theorem reams_needed : 
  (stories_per_week * pages_per_story * weeks_in_year + novel_pages_per_year) / pages_per_sheet / sheets_per_ream = 9 := by
  sorry


end reams_needed_l867_86711


namespace event_B_more_likely_l867_86749

/-- Represents the number of sides on a fair die -/
def numSides : ℕ := 6

/-- Represents the number of throws -/
def numThrows : ℕ := 3

/-- Probability of event B: three different numbers appear in three throws -/
def probB : ℚ := 5 / 9

/-- Probability of event A: at least one number appears at least twice in three throws -/
def probA : ℚ := 4 / 9

/-- Theorem stating that event B is more likely than event A -/
theorem event_B_more_likely : probB > probA := by sorry

end event_B_more_likely_l867_86749


namespace solve_equation_l867_86702

theorem solve_equation (x : ℚ) : 3 * x + 15 = (1 / 3) * (4 * x + 28) → x = -17 / 5 := by
  sorry

end solve_equation_l867_86702


namespace weight_measurement_l867_86781

def weights : List ℕ := [1, 3, 9, 27]

/-- The maximum weight that can be measured using the given weights -/
def max_weight : ℕ := 40

/-- The number of distinct weights that can be measured using the given weights -/
def distinct_weights : ℕ := 40

/-- Theorem stating the maximum weight and number of distinct weights that can be measured -/
theorem weight_measurement :
  (List.sum weights = max_weight) ∧
  (∀ w : ℕ, w ≤ max_weight → ∃ subset : List ℕ, subset.Sublist weights ∧ List.sum subset = w) ∧
  (distinct_weights = max_weight) := by
  sorry

end weight_measurement_l867_86781


namespace henrys_brothers_ages_sum_l867_86716

theorem henrys_brothers_ages_sum :
  ∀ (a b c : ℕ),
    a ≠ b ∧ b ≠ c ∧ a ≠ c →
    a < 10 ∧ b < 10 ∧ c < 10 →
    a > 0 ∧ b > 0 ∧ c > 0 →
    a = 2 * b →
    c * c = b →
    a + b + c = 14 :=
by sorry

end henrys_brothers_ages_sum_l867_86716


namespace three_digit_sum_divisible_by_eleven_l867_86709

theorem three_digit_sum_divisible_by_eleven (a b : ℕ) : 
  (100 ≤ 400 + 10*a + 3 ∧ 400 + 10*a + 3 < 1000) →  -- 4a3 is a 3-digit number
  (400 + 10*a + 3) + 984 = 1000 + 300 + 10*b + 7 →  -- 4a3 + 984 = 13b7
  (1000 + 300 + 10*b + 7) % 11 = 0 →                -- 13b7 is divisible by 11
  a + b = 10 := by
sorry

end three_digit_sum_divisible_by_eleven_l867_86709


namespace score_theorem_l867_86752

/-- Represents the bags from which balls are drawn -/
inductive Bag
| A
| B

/-- Represents the color of the balls -/
inductive Color
| Black
| White
| Red

/-- Represents the score obtained from drawing a ball -/
def score (bag : Bag) (color : Color) : ℕ :=
  match bag, color with
  | Bag.A, Color.Black => 2
  | Bag.B, Color.Black => 1
  | _, _ => 0

/-- The probability of drawing a black ball from bag B -/
def probBlackB : ℝ := 0.8

/-- The probability of getting a total score of 1 -/
def probScoreOne : ℝ := 0.24

/-- The expected value of the total score -/
def expectedScore : ℝ := 1.94

/-- Theorem stating the expected value of the total score and comparing probabilities -/
theorem score_theorem :
  ∃ (probBlackA : ℝ),
    0 ≤ probBlackA ∧ probBlackA ≤ 1 ∧
    (let pA := probBlackA * (1 - probBlackB) + (1 - probBlackA) * probBlackB
     let pB := probBlackB * probBlackB
     pB > pA) ∧
    expectedScore = 1.94 := by
  sorry

end score_theorem_l867_86752


namespace cubic_minus_four_xy_squared_factorization_l867_86767

theorem cubic_minus_four_xy_squared_factorization (x y : ℝ) :
  x^3 - 4*x*y^2 = x*(x+2*y)*(x-2*y) := by
  sorry

end cubic_minus_four_xy_squared_factorization_l867_86767


namespace interest_equality_problem_l867_86741

theorem interest_equality_problem (total : ℝ) (first_part : ℝ) (second_part : ℝ)
  (h1 : total = 2717)
  (h2 : total = first_part + second_part)
  (h3 : first_part * (3/100) * 8 = second_part * (5/100) * 3) :
  second_part = 2449 := by
  sorry

end interest_equality_problem_l867_86741


namespace percentage_difference_l867_86782

theorem percentage_difference (w q y z : ℝ) 
  (hw : w = 0.6 * q) 
  (hq : q = 0.6 * y) 
  (hz : z = 0.54 * y) : 
  (z - w) / w = 0.5 := by
sorry

end percentage_difference_l867_86782


namespace water_tower_shortage_l867_86751

theorem water_tower_shortage : 
  let tower_capacity : ℝ := 2700
  let first_neighborhood : ℝ := 300
  let second_neighborhood : ℝ := 2 * first_neighborhood
  let third_neighborhood : ℝ := second_neighborhood + 100
  let fourth_neighborhood : ℝ := 3 * first_neighborhood
  let fifth_neighborhood : ℝ := third_neighborhood / 2
  let leakage_loss : ℝ := 50
  let first_increased : ℝ := first_neighborhood * 1.1
  let third_increased : ℝ := third_neighborhood * 1.1
  let second_decreased : ℝ := second_neighborhood * 0.95
  let fifth_decreased : ℝ := fifth_neighborhood * 0.95
  let total_consumption : ℝ := first_increased + second_decreased + third_increased + fourth_neighborhood + fifth_decreased + leakage_loss
  total_consumption - tower_capacity = 252.5 :=
by sorry

end water_tower_shortage_l867_86751


namespace ellipse_equation_equivalence_l867_86750

theorem ellipse_equation_equivalence (x y : ℝ) :
  (Real.sqrt (x^2 + (y - 3)^2) + Real.sqrt (x^2 + (y + 3)^2) = 10) ↔
  (x^2 / 25 + y^2 / 16 = 1) :=
sorry

end ellipse_equation_equivalence_l867_86750


namespace number_difference_proof_l867_86768

theorem number_difference_proof (L S : ℕ) (h1 : L = 1636) (h2 : L = 6 * S + 10) : 
  L - S = 1365 := by
  sorry

end number_difference_proof_l867_86768


namespace picture_on_wall_l867_86728

theorem picture_on_wall (wall_width picture_width : ℝ) 
  (hw : wall_width = 26) (hp : picture_width = 4) :
  let distance := (wall_width - picture_width) / 2
  distance = 11 := by
  sorry

end picture_on_wall_l867_86728


namespace part_one_part_two_l867_86796

-- Define the propositions p and q
def p (a x : ℝ) : Prop := x^2 - 4*a*x + 3*a^2 < 0

def q (x : ℝ) : Prop := x^2 - x - 6 ≤ 0 ∧ x^2 + 2*x - 8 > 0

-- Theorem for part (I)
theorem part_one (a x : ℝ) (h1 : a > 0) (h2 : a = 1) (h3 : p a x ∧ q x) : 
  2 < x ∧ x < 3 := by sorry

-- Theorem for part (II)
theorem part_two (a : ℝ) (h1 : a > 0) 
  (h2 : ∀ x, ¬(p a x) → ¬(q x)) 
  (h3 : ∃ x, ¬(p a x) ∧ q x) : 
  1 < a ∧ a ≤ 2 := by sorry

end part_one_part_two_l867_86796


namespace candy_distribution_l867_86790

theorem candy_distribution (total_candies : ℕ) (candies_per_student : ℕ) (leftover_candies : ℕ) :
  total_candies = 67 →
  candies_per_student = 4 →
  leftover_candies = 3 →
  (total_candies - leftover_candies) / candies_per_student = 16 :=
by sorry

end candy_distribution_l867_86790


namespace least_positive_integer_congruence_l867_86730

theorem least_positive_integer_congruence :
  ∃! x : ℕ+, x.val + 7391 ≡ 167 [ZMOD 12] ∧
  ∀ y : ℕ+, y.val + 7391 ≡ 167 [ZMOD 12] → x ≤ y :=
by sorry

end least_positive_integer_congruence_l867_86730


namespace tank_length_is_25_l867_86775

/-- Given a tank with specific dimensions and plastering costs, prove its length is 25 meters -/
theorem tank_length_is_25 (width : ℝ) (depth : ℝ) (plaster_cost_per_sqm : ℝ) (total_plaster_cost : ℝ) :
  width = 12 →
  depth = 6 →
  plaster_cost_per_sqm = 0.45 →
  total_plaster_cost = 334.8 →
  (∃ (length : ℝ), 
    total_plaster_cost / plaster_cost_per_sqm = 2 * (length * depth) + 2 * (width * depth) + (length * width) ∧
    length = 25) := by
  sorry

end tank_length_is_25_l867_86775


namespace ann_keeps_36_cookies_l867_86717

/-- The number of cookies in a dozen -/
def dozen : ℕ := 12

/-- The number of oatmeal raisin cookies Ann bakes -/
def oatmeal_baked : ℕ := 3 * dozen

/-- The number of sugar cookies Ann bakes -/
def sugar_baked : ℕ := 2 * dozen

/-- The number of chocolate chip cookies Ann bakes -/
def chocolate_baked : ℕ := 4 * dozen

/-- The number of oatmeal raisin cookies Ann gives away -/
def oatmeal_given : ℕ := 2 * dozen

/-- The number of sugar cookies Ann gives away -/
def sugar_given : ℕ := (3 * dozen) / 2

/-- The number of chocolate chip cookies Ann gives away -/
def chocolate_given : ℕ := (5 * dozen) / 2

/-- The total number of cookies Ann keeps -/
def total_kept : ℕ := (oatmeal_baked - oatmeal_given) + (sugar_baked - sugar_given) + (chocolate_baked - chocolate_given)

theorem ann_keeps_36_cookies : total_kept = 36 := by
  sorry

end ann_keeps_36_cookies_l867_86717


namespace floor_sqrt_50_squared_l867_86795

theorem floor_sqrt_50_squared : ⌊Real.sqrt 50⌋^2 = 49 := by
  sorry

end floor_sqrt_50_squared_l867_86795


namespace complex_modulus_problem_l867_86766

theorem complex_modulus_problem (z : ℂ) (h : z * (2 + Complex.I) = 5 * Complex.I) :
  Complex.abs (z - 1) = 2 := by sorry

end complex_modulus_problem_l867_86766


namespace geometric_sequence_property_l867_86770

/-- A geometric sequence with the given properties -/
structure GeometricSequence where
  a : ℕ → ℝ
  is_geometric : ∀ n, a (n + 1) / a n = a (n + 2) / a (n + 1)
  product_345 : a 3 * a 4 * a 5 = 3
  product_678 : a 6 * a 7 * a 8 = 24

/-- The theorem statement -/
theorem geometric_sequence_property (seq : GeometricSequence) :
  seq.a 9 * seq.a 10 * seq.a 11 = 192 := by
  sorry

end geometric_sequence_property_l867_86770


namespace min_area_MAB_l867_86701

-- Define the curve C
def C (x y : ℝ) : Prop := x^2 = 4*y

-- Define point F
def F : ℝ × ℝ := (0, 1)

-- Define a line passing through F
def line_through_F (k : ℝ) (x y : ℝ) : Prop := y = k*x + 1

-- Define the area of triangle MAB
def area_MAB (k : ℝ) : ℝ := 4*(1 + k^2)^(3/2)

-- State the theorem
theorem min_area_MAB :
  ∃ (min_area : ℝ), min_area = 4 ∧
  ∀ (k : ℝ), area_MAB k ≥ min_area :=
sorry

end min_area_MAB_l867_86701


namespace average_age_of_new_men_l867_86762

theorem average_age_of_new_men (n : ℕ) (old_avg : ℝ) (age1 age2 : ℕ) (increase : ℝ) :
  n = 15 →
  age1 = 21 →
  age2 = 23 →
  increase = 2 →
  (n * (old_avg + increase) - n * old_avg) = ((n * increase + age1 + age2) / 2) →
  ((n * increase + age1 + age2) / 2) = 37 :=
by sorry

end average_age_of_new_men_l867_86762


namespace race_probability_l867_86746

theorem race_probability (total_cars : ℕ) (prob_X prob_Y prob_total : ℝ) : 
  total_cars = 12 →
  prob_X = 1/6 →
  prob_Y = 1/10 →
  prob_total = 0.39166666666666666 →
  prob_total = prob_X + prob_Y + (0.125 : ℝ) := by
sorry

end race_probability_l867_86746


namespace large_cube_single_color_face_l867_86774

/-- Represents a small cube with colored faces -/
structure SmallCube :=
  (white_faces : Fin 2)
  (blue_faces : Fin 2)
  (red_faces : Fin 2)

/-- Represents the large cube assembled from small cubes -/
def LargeCube := Fin 10 → Fin 10 → Fin 10 → SmallCube

/-- Predicate to check if two adjacent small cubes have matching colors -/
def matching_colors (c1 c2 : SmallCube) : Prop := sorry

/-- Predicate to check if a face of the large cube is a single color -/
def single_color_face (cube : LargeCube) : Prop := sorry

/-- Main theorem: The large cube has at least one face that is a single color -/
theorem large_cube_single_color_face 
  (cube : LargeCube)
  (h_matching : ∀ i j k i' j' k', 
    (i = i' ∧ j = j' ∧ (k + 1 = k' ∨ k = k' + 1)) ∨
    (i = i' ∧ (j + 1 = j' ∨ j = j' + 1) ∧ k = k') ∨
    ((i + 1 = i' ∨ i = i' + 1) ∧ j = j' ∧ k = k') →
    matching_colors (cube i j k) (cube i' j' k')) :
  single_color_face cube :=
sorry

end large_cube_single_color_face_l867_86774


namespace f_difference_l867_86754

def f (n : ℕ) : ℚ :=
  (Finset.range (n + 1)).sum (fun i => 1 / ((n + 1 + i) : ℚ))

theorem f_difference (n : ℕ) : f (n + 1) - f n = 1 / (2 * n + 3 : ℚ) := by
  sorry

end f_difference_l867_86754


namespace simplify_expression_l867_86786

theorem simplify_expression (x : ℝ) : (3*x - 6)*(x + 8) - (x + 6)*(3*x + 2) = -2*x - 60 := by
  sorry

end simplify_expression_l867_86786


namespace existence_of_special_point_l867_86706

/-- A point in a plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A triangle defined by three points -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Checks if a triangle is acute-angled -/
def isAcute (t : Triangle) : Prop := sorry

/-- Checks if a point is inside a triangle -/
def isInside (p : Point) (t : Triangle) : Prop := sorry

/-- The feet of perpendiculars from a point to the sides of a triangle -/
def feetOfPerpendiculars (p : Point) (t : Triangle) : Triangle := sorry

/-- Checks if a triangle is equilateral -/
def isEquilateral (t : Triangle) : Prop := sorry

/-- The main theorem -/
theorem existence_of_special_point (t : Triangle) (h : isAcute t) : 
  ∃ Q : Point, isInside Q t ∧ isEquilateral (feetOfPerpendiculars Q t) := by
  sorry

end existence_of_special_point_l867_86706


namespace similar_triangles_leg_ratio_l867_86759

/-- Given two similar right triangles, where one has legs 12 and 9, and the other has legs x and 7,
    prove that x = 84/9 -/
theorem similar_triangles_leg_ratio (x : ℝ) : 
  (12 : ℝ) / x = 9 / 7 → x = 84 / 9 := by sorry

end similar_triangles_leg_ratio_l867_86759


namespace absolute_difference_21st_terms_l867_86722

-- Define arithmetic sequence
def arithmeticSequence (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ := a₁ + d * (n - 1)

-- Define the sequences C and D
def C (n : ℕ) : ℤ := arithmeticSequence 50 12 n
def D (n : ℕ) : ℤ := arithmeticSequence 50 (-14) n

-- State the theorem
theorem absolute_difference_21st_terms :
  |C 21 - D 21| = 520 := by sorry

end absolute_difference_21st_terms_l867_86722


namespace smallest_winning_N_for_berta_l867_86758

/-- A game where two players take turns removing marbles from a table. -/
structure MarbleGame where
  initialMarbles : ℕ
  currentMarbles : ℕ
  playerTurn : Bool  -- True for Anna, False for Berta

/-- The rules for removing marbles in a turn -/
def validMove (game : MarbleGame) (k : ℕ) : Prop :=
  k ≥ 1 ∧
  ((k % 2 = 0 ∧ k ≤ game.currentMarbles / 2) ∨
   (k % 2 = 1 ∧ game.currentMarbles / 2 ≤ k ∧ k ≤ game.currentMarbles))

/-- The condition for a winning position -/
def isWinningPosition (n : ℕ) : Prop :=
  ∃ m : ℕ, m ≥ 2 ∧ n = 2^m - 2

/-- The theorem to prove -/
theorem smallest_winning_N_for_berta :
  ∃ N : ℕ,
    N ≥ 100000 ∧
    isWinningPosition N ∧
    (∀ M : ℕ, M ≥ 100000 ∧ M < N → ¬isWinningPosition M) :=
  sorry

end smallest_winning_N_for_berta_l867_86758


namespace team_average_score_l867_86704

theorem team_average_score (player1_score : ℕ) (player2_score : ℕ) (player3_score : ℕ) :
  player1_score = 20 →
  player2_score = player1_score / 2 →
  player3_score = 6 * player2_score →
  (player1_score + player2_score + player3_score) / 3 = 30 := by
  sorry

end team_average_score_l867_86704


namespace polygon_area_is_300_l867_86765

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The polygon described in the problem -/
def polygon : List Point := [
  ⟨0, 0⟩, ⟨10, 0⟩, ⟨10, 10⟩, ⟨10, 20⟩, ⟨10, 30⟩, ⟨0, 30⟩, ⟨0, 20⟩, ⟨0, 10⟩
]

/-- Calculate the area of a polygon given its vertices -/
def polygonArea (vertices : List Point) : ℝ :=
  sorry

/-- Theorem: The area of the given polygon is 300 square units -/
theorem polygon_area_is_300 : polygonArea polygon = 300 := by
  sorry

end polygon_area_is_300_l867_86765


namespace desired_average_sale_l867_86787

def sales_first_five_months : List ℝ := [6435, 6927, 6855, 7230, 6562]
def sale_sixth_month : ℝ := 7991
def number_of_months : ℕ := 6

theorem desired_average_sale (sales : List ℝ) (sixth_sale : ℝ) (num_months : ℕ) :
  sales = sales_first_five_months →
  sixth_sale = sale_sixth_month →
  num_months = number_of_months →
  (sales.sum + sixth_sale) / num_months = 7000 := by
  sorry

end desired_average_sale_l867_86787


namespace cannot_reach_123456_l867_86708

def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sumOfDigits (n / 10)

def sequenceElement (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => sequenceElement n + sumOfDigits (sequenceElement n)

theorem cannot_reach_123456 : ∀ n : ℕ, sequenceElement n ≠ 123456 := by
  sorry

end cannot_reach_123456_l867_86708


namespace min_value_sum_squares_l867_86798

theorem min_value_sum_squares (a b : ℝ) 
  (ha : a > 0) (hb : b > 0) 
  (h : (a - 1)^3 + (b - 1)^3 ≥ 3*(2 - a - b)) : 
  a^2 + b^2 ≥ 2 := by
sorry

end min_value_sum_squares_l867_86798


namespace golden_raisins_fraction_of_total_cost_l867_86776

/-- Represents the cost of ingredients relative to golden raisins -/
structure IngredientCost where
  goldenRaisins : ℚ
  almonds : ℚ
  cashews : ℚ
  walnuts : ℚ

/-- Represents the weight of ingredients in pounds -/
structure IngredientWeight where
  goldenRaisins : ℚ
  almonds : ℚ
  cashews : ℚ
  walnuts : ℚ

def mixtureCost (cost : IngredientCost) (weight : IngredientWeight) : ℚ :=
  cost.goldenRaisins * weight.goldenRaisins +
  cost.almonds * weight.almonds +
  cost.cashews * weight.cashews +
  cost.walnuts * weight.walnuts

theorem golden_raisins_fraction_of_total_cost 
  (cost : IngredientCost)
  (weight : IngredientWeight)
  (h1 : cost.goldenRaisins = 1)
  (h2 : cost.almonds = 2 * cost.goldenRaisins)
  (h3 : cost.cashews = 3 * cost.goldenRaisins)
  (h4 : cost.walnuts = 4 * cost.goldenRaisins)
  (h5 : weight.goldenRaisins = 4)
  (h6 : weight.almonds = 2)
  (h7 : weight.cashews = 1)
  (h8 : weight.walnuts = 1) :
  (cost.goldenRaisins * weight.goldenRaisins) / mixtureCost cost weight = 4 / 15 := by
  sorry

end golden_raisins_fraction_of_total_cost_l867_86776


namespace smallest_sum_of_digits_of_sum_l867_86726

/-- A function that returns the sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- A function that checks if all digits in a natural number are different -/
def all_digits_different (n : ℕ) : Prop := sorry

/-- A function that checks if two natural numbers have all different digits between them -/
def all_digits_different_between (a b : ℕ) : Prop := sorry

theorem smallest_sum_of_digits_of_sum :
  ∀ a b : ℕ,
    100 ≤ a ∧ a < 1000 ∧
    100 ≤ b ∧ b < 1000 ∧
    a ≠ b ∧
    all_digits_different a ∧
    all_digits_different b ∧
    all_digits_different_between a b ∧
    1000 ≤ a + b ∧ a + b < 10000 →
    ∃ (s : ℕ), s = a + b ∧ sum_of_digits s = 1 ∧
    ∀ (t : ℕ), t = a + b → sum_of_digits s ≤ sum_of_digits t :=
by sorry

end smallest_sum_of_digits_of_sum_l867_86726


namespace total_water_consumed_water_consumed_is_686_l867_86792

/-- Represents a medication schedule --/
structure MedicationSchedule where
  name : String
  timesPerDay : Nat
  waterPerDose : Nat

/-- Represents missed doses for a medication --/
structure MissedDoses where
  medication : String
  count : Nat

/-- Calculates the total water consumed for a medication over two weeks --/
def waterConsumedForMedication (schedule : MedicationSchedule) : Nat :=
  schedule.timesPerDay * schedule.waterPerDose * 7 * 2

/-- Calculates the water missed due to skipped doses --/
def waterMissedForMedication (schedule : MedicationSchedule) (missed : Nat) : Nat :=
  schedule.waterPerDose * missed

/-- The main theorem to prove --/
theorem total_water_consumed 
  (schedules : List MedicationSchedule)
  (missedDoses : List MissedDoses) : Nat :=
  let totalWater := schedules.map waterConsumedForMedication |>.sum
  let missedWater := missedDoses.map (fun m => 
    let schedule := schedules.find? (fun s => s.name == m.medication)
    match schedule with
    | some s => waterMissedForMedication s m.count
    | none => 0
  ) |>.sum
  totalWater - missedWater

/-- The specific medication schedules --/
def medicationSchedules : List MedicationSchedule := [
  { name := "A", timesPerDay := 3, waterPerDose := 4 },
  { name := "B", timesPerDay := 4, waterPerDose := 5 },
  { name := "C", timesPerDay := 2, waterPerDose := 6 },
  { name := "D", timesPerDay := 1, waterPerDose := 8 }
]

/-- The specific missed doses --/
def missedDosesList : List MissedDoses := [
  { medication := "A", count := 3 },
  { medication := "B", count := 2 },
  { medication := "C", count := 2 },
  { medication := "D", count := 1 }
]

/-- The main theorem for this specific problem --/
theorem water_consumed_is_686 : 
  total_water_consumed medicationSchedules missedDosesList = 686 := by
  sorry

end total_water_consumed_water_consumed_is_686_l867_86792


namespace probability_red_then_black_specific_l867_86799

/-- Represents a deck of cards with red and black cards -/
structure Deck :=
  (total : ℕ)
  (red : ℕ)
  (black : ℕ)
  (h1 : red + black = total)

/-- Calculates the probability of drawing a red card first and a black card second -/
def probability_red_then_black (d : Deck) : ℚ :=
  (d.red : ℚ) / d.total * (d.black : ℚ) / (d.total - 1)

/-- Theorem: The probability of drawing a red card first and a black card second
    from a deck with 20 red cards and 32 black cards (total 52 cards) is 160/663 -/
theorem probability_red_then_black_specific :
  let d : Deck := ⟨52, 20, 32, by simp⟩
  probability_red_then_black d = 160 / 663 := by sorry

end probability_red_then_black_specific_l867_86799


namespace equation_has_four_real_solutions_l867_86724

theorem equation_has_four_real_solutions :
  ∃ (s : Finset ℝ), (∀ x ∈ s, x^2 + 1/x^2 = 2006 + 1/2006) ∧ s.card = 4 ∧
  (∀ y : ℝ, y^2 + 1/y^2 = 2006 + 1/2006 → y ∈ s) := by
  sorry

end equation_has_four_real_solutions_l867_86724


namespace rectangle_area_l867_86748

/-- Given a rectangle where the length is 3 times the width and the width is 5 inches,
    prove that its area is 75 square inches. -/
theorem rectangle_area (width : ℝ) (length : ℝ) (area : ℝ) : 
  width = 5 → length = 3 * width → area = length * width → area = 75 := by sorry

end rectangle_area_l867_86748


namespace square_difference_l867_86755

theorem square_difference : (15 + 7)^2 - (15^2 + 7^2) = 210 := by
  sorry

end square_difference_l867_86755


namespace identical_roots_quadratic_l867_86761

/-- If the quadratic equation 3x^2 - 6x + k = 0 has two identical real roots, then k = 3 -/
theorem identical_roots_quadratic (k : ℝ) : 
  (∃ x : ℝ, (3 * x^2 - 6 * x + k = 0) ∧ 
   (∀ y : ℝ, 3 * y^2 - 6 * y + k = 0 → y = x)) → 
  k = 3 := by sorry

end identical_roots_quadratic_l867_86761


namespace field_trip_adults_l867_86719

/-- The number of adults going on a field trip --/
theorem field_trip_adults (van_capacity : ℕ) (num_students : ℕ) (num_vans : ℕ) : 
  van_capacity = 7 → num_students = 33 → num_vans = 6 → 
  (num_vans * van_capacity) - num_students = 9 := by
  sorry

end field_trip_adults_l867_86719


namespace min_employees_for_given_requirements_l867_86793

/-- Represents the number of employees needed for each pollution type and their intersections -/
structure PollutionMonitoring where
  water : ℕ
  air : ℕ
  soil : ℕ
  water_air : ℕ
  air_soil : ℕ
  soil_water : ℕ
  all_three : ℕ

/-- Calculates the minimum number of employees needed given the monitoring requirements -/
def min_employees (p : PollutionMonitoring) : ℕ :=
  p.water + p.air + p.soil - p.water_air - p.air_soil - p.soil_water + p.all_three

/-- Theorem stating that given the specific monitoring requirements, 225 employees are needed -/
theorem min_employees_for_given_requirements :
  let p : PollutionMonitoring := {
    water := 115,
    air := 92,
    soil := 60,
    water_air := 32,
    air_soil := 20,
    soil_water := 10,
    all_three := 5
  }
  min_employees p = 225 := by
  sorry


end min_employees_for_given_requirements_l867_86793


namespace max_ratio_on_circle_l867_86777

-- Define a point with integer coordinates
structure IntPoint where
  x : ℤ
  y : ℤ

-- Define a function to check if a point is on the circle x^2 + y^2 = 16
def onCircle (p : IntPoint) : Prop :=
  p.x^2 + p.y^2 = 16

-- Define a function to calculate the squared distance between two points
def squaredDistance (p1 p2 : IntPoint) : ℤ :=
  (p1.x - p2.x)^2 + (p1.y - p2.y)^2

-- Theorem statement
theorem max_ratio_on_circle (A B C D : IntPoint) :
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D →
  onCircle A ∧ onCircle B ∧ onCircle C ∧ onCircle D →
  ¬ ∃ m n : ℤ, m^2 = squaredDistance A B ∧ n > 0 →
  ¬ ∃ m n : ℤ, m^2 = squaredDistance C D ∧ n > 0 →
  ∀ r : ℚ, r * (squaredDistance C D : ℚ) ≤ (squaredDistance A B : ℚ) → r ≤ 1 :=
by sorry

end max_ratio_on_circle_l867_86777


namespace jeff_trucks_count_l867_86715

theorem jeff_trucks_count :
  ∀ (trucks cars : ℕ),
    cars = 2 * trucks →
    trucks + cars = 60 →
    trucks = 20 :=
by
  sorry

end jeff_trucks_count_l867_86715


namespace davids_age_twice_daughters_l867_86764

/-- 
Given:
- David is currently 40 years old
- David's daughter is currently 12 years old

Prove that 16 years will pass before David's age is twice his daughter's age
-/
theorem davids_age_twice_daughters (david_age : ℕ) (daughter_age : ℕ) :
  david_age = 40 →
  daughter_age = 12 →
  ∃ (years : ℕ), david_age + years = 2 * (daughter_age + years) ∧ years = 16 :=
by sorry

end davids_age_twice_daughters_l867_86764


namespace bugs_meeting_point_l867_86745

/-- A quadrilateral with sides of length 5, 7, 8, and 6 -/
structure Quadrilateral :=
  (A B C D : ℝ × ℝ)
  (ab_length : dist A B = 5)
  (bc_length : dist B C = 7)
  (cd_length : dist C D = 8)
  (da_length : dist D A = 6)

/-- The point where two bugs meet when starting from A and moving in opposite directions -/
def meeting_point (q : Quadrilateral) : ℝ × ℝ := sorry

/-- The distance between point B and the meeting point E -/
def BE (q : Quadrilateral) : ℝ := dist q.B (meeting_point q)

theorem bugs_meeting_point (q : Quadrilateral) : BE q = 6 := by sorry

end bugs_meeting_point_l867_86745


namespace parabola_translation_theorem_l867_86731

/-- Represents a parabola in the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The original parabola y = x^2 -/
def original_parabola : Parabola := { a := 1, b := 0, c := 0 }

/-- Translates a parabola vertically by a given amount -/
def translate_vertical (p : Parabola) (amount : ℝ) : Parabola :=
  { a := p.a, b := p.b, c := p.c + amount }

/-- Translates a parabola horizontally by a given amount -/
def translate_horizontal (p : Parabola) (amount : ℝ) : Parabola :=
  { a := p.a, b := -2 * p.a * amount + p.b, c := p.a * amount^2 - p.b * amount + p.c }

/-- The resulting parabola after translations -/
def resulting_parabola : Parabola :=
  translate_horizontal (translate_vertical original_parabola 3) 5

theorem parabola_translation_theorem :
  resulting_parabola = { a := 1, b := -10, c := 28 } :=
by sorry

end parabola_translation_theorem_l867_86731


namespace trigonometric_identities_l867_86733

theorem trigonometric_identities :
  (∃ (x y : ℝ), 
    x = Real.sin (-1395 * π / 180) * Real.cos (1140 * π / 180) + 
        Real.cos (-1020 * π / 180) * Real.sin (750 * π / 180) ∧
    y = Real.sin (-11 * π / 6) + Real.cos (3 * π / 4) * Real.tan (4 * π) ∧
    x = (Real.sqrt 2 + 1) / 4 ∧
    y = 1 / 2) :=
by
  sorry

end trigonometric_identities_l867_86733


namespace probability_multiple_of_five_l867_86713

theorem probability_multiple_of_five (total_pages : ℕ) (h : total_pages = 300) :
  (Finset.filter (fun n => n % 5 = 0) (Finset.range total_pages)).card / total_pages = 1 / 5 := by
  sorry

end probability_multiple_of_five_l867_86713


namespace roots_of_equation_l867_86788

theorem roots_of_equation (x : ℝ) : 
  (x - 1) * (x - 2) = 0 ↔ x = 1 ∨ x = 2 := by
  sorry

end roots_of_equation_l867_86788


namespace range_of_m_l867_86773

/-- For the equation m/(x-2) = 3 with positive solutions for x, 
    the range of m is {m ∈ ℝ | m > -6 and m ≠ 0} -/
theorem range_of_m (m : ℝ) : 
  (∃ x : ℝ, x > 0 ∧ m / (x - 2) = 3) ↔ m > -6 ∧ m ≠ 0 := by
  sorry

end range_of_m_l867_86773


namespace complex_locus_ellipse_l867_86710

/-- For a complex number z with |z| = 3, the locus of points traced by z + 2/z forms an ellipse -/
theorem complex_locus_ellipse (z : ℂ) (h : Complex.abs z = 3) :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ 
  ∀ (w : ℂ), w = z + 2 / z → (w.re / a) ^ 2 + (w.im / b) ^ 2 = 1 :=
by sorry

end complex_locus_ellipse_l867_86710


namespace problem_statement_l867_86757

theorem problem_statement : (-1 : ℤ) ^ 53 + 2 ^ (4^3 + 3^2 - 7^2) = 16777215 := by
  sorry

end problem_statement_l867_86757


namespace pullup_median_is_5_point_5_l867_86789

def pullup_counts : List ℕ := [4, 4, 5, 5, 5, 6, 6, 7, 7, 8]

def median (l : List ℝ) : ℝ := sorry

theorem pullup_median_is_5_point_5 :
  median (pullup_counts.map (λ x => (x : ℝ))) = 5.5 := by sorry

end pullup_median_is_5_point_5_l867_86789


namespace fiftieth_islander_statement_l867_86700

/-- Represents the type of islander: Knight (always tells the truth) or Liar (always lies) -/
inductive IslanderType
| Knight
| Liar

/-- Represents what an islander says about their neighbor -/
inductive Statement
| Knight
| Liar

/-- A function that determines what an islander at a given position says about their right neighbor -/
def whatTheySay (position : Nat) : Statement :=
  if position % 2 = 1 then Statement.Knight else Statement.Liar

/-- The main theorem to prove -/
theorem fiftieth_islander_statement :
  ∀ (islanders : Fin 50 → IslanderType),
  (∀ (i : Fin 50), 
    (islanders i = IslanderType.Knight → whatTheySay i.val = Statement.Knight → islanders (i + 1) = IslanderType.Knight) ∧
    (islanders i = IslanderType.Liar → whatTheySay i.val = Statement.Knight → islanders (i + 1) = IslanderType.Liar) ∧
    (islanders i = IslanderType.Knight → whatTheySay i.val = Statement.Liar → islanders (i + 1) = IslanderType.Liar) ∧
    (islanders i = IslanderType.Liar → whatTheySay i.val = Statement.Liar → islanders (i + 1) = IslanderType.Knight)) →
  whatTheySay 50 = Statement.Knight :=
sorry

end fiftieth_islander_statement_l867_86700


namespace fraction_puzzle_solvable_l867_86743

def is_valid_fraction (a b : ℕ) : Prop := 
  a > 0 ∧ b > 0 ∧ a ≤ 9 ∧ b ≤ 9 ∧ a ≠ b

def are_distinct (a b c d e f g h i : ℕ) : Prop :=
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧ a ≠ h ∧ a ≠ i ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧ b ≠ h ∧ b ≠ i ∧
  c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧ c ≠ h ∧ c ≠ i ∧
  d ≠ e ∧ d ≠ f ∧ d ≠ g ∧ d ≠ h ∧ d ≠ i ∧
  e ≠ f ∧ e ≠ g ∧ e ≠ h ∧ e ≠ i ∧
  f ≠ g ∧ f ≠ h ∧ f ≠ i ∧
  g ≠ h ∧ g ≠ i ∧
  h ≠ i

theorem fraction_puzzle_solvable : 
  ∃ (a b c d e f g h i : ℕ),
    is_valid_fraction a b ∧ 
    is_valid_fraction c d ∧ 
    is_valid_fraction e f ∧ 
    is_valid_fraction g h ∧
    are_distinct a b c d e f g h i ∧
    (a : ℚ) / b + (c : ℚ) / d + (e : ℚ) / f + (g : ℚ) / h = i := by
  sorry

end fraction_puzzle_solvable_l867_86743


namespace jeans_cost_thirty_l867_86756

/-- The price of socks in dollars -/
def socks_price : ℕ := 5

/-- The price difference between t-shirt and socks in dollars -/
def tshirt_socks_diff : ℕ := 10

/-- The price of a t-shirt in dollars -/
def tshirt_price : ℕ := socks_price + tshirt_socks_diff

/-- The price of jeans in dollars -/
def jeans_price : ℕ := 2 * tshirt_price

theorem jeans_cost_thirty : jeans_price = 30 := by
  sorry

end jeans_cost_thirty_l867_86756


namespace andy_final_position_l867_86794

-- Define the direction as an enumeration
inductive Direction
  | North
  | West
  | South
  | East

-- Define the position as a pair of integers
def Position := ℤ × ℤ

-- Define the function to get the next direction after turning left
def turn_left (d : Direction) : Direction :=
  match d with
  | Direction.North => Direction.West
  | Direction.West => Direction.South
  | Direction.South => Direction.East
  | Direction.East => Direction.North

-- Define the function to move in a given direction
def move (p : Position) (d : Direction) (distance : ℤ) : Position :=
  match d with
  | Direction.North => (p.1, p.2 + distance)
  | Direction.West => (p.1 - distance, p.2)
  | Direction.South => (p.1, p.2 - distance)
  | Direction.East => (p.1 + distance, p.2)

-- Define the function to perform one step of Andy's movement
def step (p : Position) (d : Direction) (n : ℕ) : Position × Direction :=
  let new_p := move p d (n^2)
  let new_d := turn_left d
  (new_p, new_d)

-- Define the function to perform multiple steps
def multi_step (initial_p : Position) (initial_d : Direction) (steps : ℕ) : Position :=
  if steps = 0 then
    initial_p
  else
    let (p, d) := (List.range steps).foldl
      (fun (acc : Position × Direction) n => step acc.1 acc.2 (n + 1))
      (initial_p, initial_d)
    p

-- Theorem statement
theorem andy_final_position :
  multi_step (10, -10) Direction.North 16 = (154, -138) :=
sorry

end andy_final_position_l867_86794


namespace correct_monk_bun_equations_l867_86769

/-- Represents the monk and bun distribution problem -/
def monk_bun_problem (x y : ℕ) : Prop :=
  -- Total number of monks is 100
  x + y = 100 ∧
  -- Total number of buns is 100, distributed as 3 per elder monk and 1/3 per younger monk
  3 * x + y / 3 = 100

/-- The correct system of equations for the monk and bun distribution problem -/
theorem correct_monk_bun_equations :
  ∀ x y : ℕ, monk_bun_problem x y ↔ x + y = 100 ∧ 3 * x + y / 3 = 100 := by
  sorry

end correct_monk_bun_equations_l867_86769


namespace green_pill_cost_proof_l867_86783

/-- The cost of a green pill in dollars -/
def green_pill_cost : ℚ := 43 / 3

/-- The cost of a pink pill in dollars -/
def pink_pill_cost : ℚ := green_pill_cost - 1

/-- The cost of a blue pill in dollars -/
def blue_pill_cost : ℚ := pink_pill_cost - 2

/-- The number of days in the treatment -/
def treatment_days : ℕ := 21

/-- The total cost of the treatment in dollars -/
def total_cost : ℚ := 819

theorem green_pill_cost_proof :
  (green_pill_cost + pink_pill_cost + blue_pill_cost) * treatment_days = total_cost ∧
  green_pill_cost = 43 / 3 := by
  sorry

#eval green_pill_cost -- To check the value

end green_pill_cost_proof_l867_86783


namespace inequality_proof_l867_86732

theorem inequality_proof (a b α β θ : ℝ) (ha : a > 0) (hb : b > 0) (hα : abs α > a) :
  (α * β - Real.sqrt (a^2 * β^2 + b^2 * α^2 - a^2 * b^2)) / (α^2 - a^2) ≤ 
  (β + b * Real.sin θ) / (α + a * Real.cos θ) ∧
  (β + b * Real.sin θ) / (α + a * Real.cos θ) ≤ 
  (α * β + Real.sqrt (a^2 * β^2 + b^2 * α^2 - a^2 * b^2)) / (α^2 - a^2) :=
by sorry

end inequality_proof_l867_86732


namespace power_function_through_point_l867_86738

def is_power_function (f : ℝ → ℝ) : Prop :=
  ∃ a : ℝ, ∀ x : ℝ, f x = x^a

theorem power_function_through_point :
  ∀ f : ℝ → ℝ, is_power_function f →
  f 2 = (1/4 : ℝ) →
  ∃ a : ℝ, (∀ x : ℝ, f x = x^a) ∧ a = -2 :=
by sorry

end power_function_through_point_l867_86738


namespace combination_equality_l867_86742

theorem combination_equality (x : ℕ) : 
  (Nat.choose 10 x = Nat.choose 10 (3 * x - 2)) → (x = 1 ∨ x = 3) := by
  sorry

end combination_equality_l867_86742


namespace kaleb_games_proof_l867_86740

theorem kaleb_games_proof (sold : ℕ) (boxes : ℕ) (games_per_box : ℕ) 
  (h1 : sold = 46)
  (h2 : boxes = 6)
  (h3 : games_per_box = 5) :
  sold + boxes * games_per_box = 76 := by
  sorry

end kaleb_games_proof_l867_86740


namespace at_least_one_not_less_than_two_l867_86707

theorem at_least_one_not_less_than_two (a b : ℕ) (h : a + b ≥ 3) : a ≥ 2 ∨ b ≥ 2 := by
  sorry

end at_least_one_not_less_than_two_l867_86707


namespace square_area_ratio_l867_86739

theorem square_area_ratio (y : ℝ) (hy : y > 0) :
  (y^2) / ((3*y)^2) = 1/9 := by
  sorry

end square_area_ratio_l867_86739


namespace m_less_than_2_necessary_not_sufficient_l867_86705

-- Define the quadratic function
def f (m x : ℝ) := x^2 + m*x + 1

-- Define the condition for the solution set to be ℝ
def solution_is_real (m : ℝ) : Prop :=
  ∀ x, f m x > 0

-- Define the necessary and sufficient condition
def necessary_and_sufficient (m : ℝ) : Prop :=
  m^2 - 4 < 0

-- Theorem: m < 2 is a necessary but not sufficient condition
theorem m_less_than_2_necessary_not_sufficient :
  (∀ m, solution_is_real m → m < 2) ∧
  ¬(∀ m, m < 2 → solution_is_real m) :=
sorry

end m_less_than_2_necessary_not_sufficient_l867_86705


namespace integer_sum_problem_l867_86735

theorem integer_sum_problem (x y : ℕ) (h1 : x > y) (h2 : x - y = 8) (h3 : x * y = 180) : x + y = 28 := by
  sorry

end integer_sum_problem_l867_86735


namespace sylvie_turtle_weight_l867_86723

/-- The weight of turtles Sylvie has, given the feeding conditions -/
theorem sylvie_turtle_weight :
  let food_per_half_pound : ℚ := 1 -- 1 ounce of food per 1/2 pound of body weight
  let ounces_per_jar : ℚ := 15 -- Each jar contains 15 ounces
  let cost_per_jar : ℚ := 2 -- Each jar costs $2
  let total_cost : ℚ := 8 -- It costs $8 to feed the turtles
  
  (total_cost / cost_per_jar) * ounces_per_jar / food_per_half_pound / 2 = 30 := by
  sorry

end sylvie_turtle_weight_l867_86723


namespace equal_prob_when_four_prob_when_six_l867_86763

-- Define the set of paper slips
def slips : Finset ℕ := {1, 2, 3, 4, 5, 6, 7}

-- Define the probability of winning for Xiao Ming and Xiao Ying given Xiao Ming's draw
def win_prob (xiao_ming_draw : ℕ) : ℚ × ℚ :=
  let remaining_slips := slips.erase xiao_ming_draw
  let xiao_ming_wins := (remaining_slips.filter (· < xiao_ming_draw)).card
  let xiao_ying_wins := (remaining_slips.filter (· > xiao_ming_draw)).card
  (xiao_ming_wins / remaining_slips.card, xiao_ying_wins / remaining_slips.card)

-- Theorem 1: When Xiao Ming draws 4, both have equal probability of winning
theorem equal_prob_when_four : win_prob 4 = (1/2, 1/2) := by sorry

-- Theorem 2: When Xiao Ming draws 6, probabilities are 5/6 and 1/6
theorem prob_when_six : win_prob 6 = (5/6, 1/6) := by sorry

end equal_prob_when_four_prob_when_six_l867_86763


namespace max_profit_difference_l867_86772

def total_records : ℕ := 300

def sammy_offer : ℕ → ℚ := λ n => 4 * n

def bryan_offer : ℕ → ℚ := λ n => 6 * (2/3 * n) + 1 * (1/3 * n)

def christine_offer : ℕ → ℚ := λ n => 10 * 30 + 3 * (n - 30)

theorem max_profit_difference (n : ℕ) (h : n = total_records) : 
  max (abs (sammy_offer n - bryan_offer n))
      (max (abs (sammy_offer n - christine_offer n))
           (abs (bryan_offer n - christine_offer n)))
  = 190 :=
sorry

end max_profit_difference_l867_86772


namespace roots_of_g_l867_86744

theorem roots_of_g (a b : ℝ) : 
  (∀ x : ℝ, (x^2 - a*x + b = 0) ↔ (x = 2 ∨ x = 3)) →
  (∀ x : ℝ, (b*x^2 - a*x - 1 = 0) ↔ (x = 1 ∨ x = -1/6)) := by
sorry

end roots_of_g_l867_86744


namespace range_of_a_range_of_m_l867_86720

-- Define the propositions
def p (a : ℝ) : Prop := ∀ x : ℝ, x^2 - 4*x + a^2 > 0

def q (a : ℝ) : Prop := ∃ x y : ℝ, x > 0 ∧ y < 0 ∧ x^2 + (a+1)*x + a - 1 = 0 ∧ y^2 + (a+1)*y + a - 1 = 0

def r (a m : ℝ) : Prop := a^2 - 2*a + 1 - m^2 ≥ 0 ∧ m > 0

-- Theorem 1
theorem range_of_a (a : ℝ) : (p a ∨ q a) ∧ ¬(p a ∧ q a) → (-2 ≤ a ∧ a < 1) ∨ a > 2 :=
sorry

-- Theorem 2
theorem range_of_m (m : ℝ) : (∀ a : ℝ, ¬(r a m) → ¬(p a)) ∧ (∃ a : ℝ, ¬(r a m) ∧ p a) → m > 3 :=
sorry

end range_of_a_range_of_m_l867_86720


namespace braiding_time_for_dance_team_l867_86760

/-- Calculates the time in minutes to braid dancers' hair -/
def braidingTime (num_dancers : ℕ) (braids_per_dancer : ℕ) (seconds_per_braid : ℕ) : ℕ :=
  let total_braids := num_dancers * braids_per_dancer
  let total_seconds := total_braids * seconds_per_braid
  total_seconds / 60

theorem braiding_time_for_dance_team :
  braidingTime 8 5 30 = 20 := by
  sorry

end braiding_time_for_dance_team_l867_86760


namespace tan_alpha_and_expression_l867_86712

theorem tan_alpha_and_expression (α : ℝ) 
  (h : Real.tan (π / 4 + α) = 2) : 
  Real.tan α = 1 / 3 ∧ 
  (2 * Real.sin α ^ 2 + Real.sin (2 * α)) / (1 + Real.tan α) = 3 / 5 := by
  sorry

end tan_alpha_and_expression_l867_86712


namespace total_harvest_kg_l867_86727

def apple_sections : ℕ := 8
def apple_yield_per_section : ℕ := 450

def orange_sections : ℕ := 10
def orange_crates_per_section : ℕ := 60
def orange_kg_per_crate : ℕ := 8

def peach_sections : ℕ := 3
def peach_sacks_per_section : ℕ := 55
def peach_kg_per_sack : ℕ := 12

def cherry_fields : ℕ := 5
def cherry_baskets_per_field : ℕ := 50
def cherry_kg_per_basket : ℚ := 3.5

theorem total_harvest_kg : 
  apple_sections * apple_yield_per_section + 
  orange_sections * orange_crates_per_section * orange_kg_per_crate + 
  peach_sections * peach_sacks_per_section * peach_kg_per_sack + 
  cherry_fields * cherry_baskets_per_field * cherry_kg_per_basket = 11255 := by
  sorry

end total_harvest_kg_l867_86727


namespace solve_for_a_l867_86718

theorem solve_for_a : ∃ a : ℝ, (∀ x : ℝ, x = 2 → a * x - 2 = 4) → a = 3 := by
  sorry

end solve_for_a_l867_86718


namespace third_side_length_l867_86784

/-- A triangle with two known sides and a known perimeter -/
structure Triangle where
  side1 : ℝ
  side2 : ℝ
  perimeter : ℝ

/-- The length of the third side of a triangle -/
def third_side (t : Triangle) : ℝ := t.perimeter - t.side1 - t.side2

/-- Theorem: In a triangle with sides 5 cm and 20 cm, and perimeter 55 cm, the third side is 30 cm -/
theorem third_side_length (t : Triangle) 
  (h1 : t.side1 = 5) 
  (h2 : t.side2 = 20) 
  (h3 : t.perimeter = 55) : 
  third_side t = 30 := by
  sorry

end third_side_length_l867_86784


namespace min_squared_distance_to_line_l867_86791

/-- The minimum squared distance from a point on the line x - y - 1 = 0 to the point (2, 2) -/
theorem min_squared_distance_to_line (x y : ℝ) :
  x - y - 1 = 0 → (∀ x' y' : ℝ, x' - y' - 1 = 0 → (x - 2)^2 + (y - 2)^2 ≤ (x' - 2)^2 + (y' - 2)^2) →
  (x - 2)^2 + (y - 2)^2 = 1/2 :=
by sorry

end min_squared_distance_to_line_l867_86791


namespace constant_k_equality_l867_86747

theorem constant_k_equality (x : ℝ) : 
  -x^2 - (-17 + 11)*x - 8 = -(x - 2)*(x - 4) := by
  sorry

end constant_k_equality_l867_86747


namespace arnel_pencil_sharing_l867_86736

/-- Calculates the number of pencils each friend receives when Arnel shares his pencils --/
def pencils_per_friend (num_boxes : ℕ) (pencils_per_box : ℕ) (kept_pencils : ℕ) (num_friends : ℕ) : ℕ :=
  ((num_boxes * pencils_per_box) - kept_pencils) / num_friends

/-- Proves that each friend receives 8 pencils under the given conditions --/
theorem arnel_pencil_sharing :
  pencils_per_friend 10 5 10 5 = 8 := by
  sorry

end arnel_pencil_sharing_l867_86736


namespace quadratic_solution_system_solution_l867_86797

-- Define the quadratic equation
def quadratic_eq (x : ℝ) : Prop := 2 * x^2 - 50 = 0

-- Define the system of equations
def system_eq (x y : ℝ) : Prop := x = 2*y + 7 ∧ 2*x + 5*y = -4

-- Theorem for the quadratic equation
theorem quadratic_solution : 
  ∃ x₁ x₂ : ℝ, x₁ = -5 ∧ x₂ = 5 ∧ 
  (∀ x : ℝ, quadratic_eq x ↔ (x = x₁ ∨ x = x₂)) :=
sorry

-- Theorem for the system of equations
theorem system_solution : 
  ∃! x y : ℝ, system_eq x y ∧ x = 3 ∧ y = -2 :=
sorry

end quadratic_solution_system_solution_l867_86797


namespace system_solution_l867_86729

theorem system_solution : 
  ∃ (j k : ℚ), (7 * j - 35 * k = -3) ∧ (3 * j - 2 * k = 5) ∧ (j = 547/273) ∧ (k = 44/91) := by
  sorry

end system_solution_l867_86729


namespace polynomial_factor_implies_constant_value_l867_86725

theorem polynomial_factor_implies_constant_value (a : ℝ) : 
  (∃ b : ℝ, ∀ y : ℝ, y^2 + 3*y - a = (y - 3) * (y + b)) → a = 18 := by
  sorry

end polynomial_factor_implies_constant_value_l867_86725


namespace pencil_average_price_l867_86703

/-- Given the purchase of pens and pencils, prove the average price of a pencil -/
theorem pencil_average_price 
  (total_cost : ℝ) 
  (num_pens : ℕ) 
  (num_pencils : ℕ) 
  (pen_avg_price : ℝ) 
  (h1 : total_cost = 570)
  (h2 : num_pens = 30)
  (h3 : num_pencils = 75)
  (h4 : pen_avg_price = 14) :
  (total_cost - num_pens * pen_avg_price) / num_pencils = 2 := by
  sorry

end pencil_average_price_l867_86703


namespace probability_sum_five_l867_86779

/-- The probability of the sum of four standard dice rolls equaling 5 -/
def prob_sum_five : ℚ := 1 / 324

/-- The number of faces on a standard die -/
def standard_die_faces : ℕ := 6

/-- The minimum value on a standard die -/
def min_die_value : ℕ := 1

/-- The maximum value on a standard die -/
def max_die_value : ℕ := 6

/-- A function representing a valid die roll -/
def valid_roll (n : ℕ) : Prop := min_die_value ≤ n ∧ n ≤ max_die_value

/-- The sum we're looking for -/
def target_sum : ℕ := 5

/-- The number of dice rolled -/
def num_dice : ℕ := 4

theorem probability_sum_five :
  ∀ (a b c d : ℕ), valid_roll a → valid_roll b → valid_roll c → valid_roll d →
  (a + b + c + d = target_sum) →
  (prob_sum_five = (↑(Nat.choose num_dice 1) / ↑(standard_die_faces ^ num_dice) : ℚ)) := by
  sorry

end probability_sum_five_l867_86779


namespace max_squares_covered_2inch_card_l867_86753

/-- Represents a square card -/
structure Card where
  side_length : ℝ

/-- Represents a checkerboard -/
structure Checkerboard where
  square_size : ℝ

/-- Calculates the maximum number of squares a card can cover on a checkerboard -/
def max_squares_covered (card : Card) (board : Checkerboard) : ℕ :=
  sorry

/-- Theorem stating the maximum number of squares covered by a 2-inch card on a 1-inch checkerboard -/
theorem max_squares_covered_2inch_card (card : Card) (board : Checkerboard) :
  card.side_length = 2 →
  board.square_size = 1 →
  max_squares_covered card board = 9 :=
sorry

end max_squares_covered_2inch_card_l867_86753


namespace manufacturing_cost_of_shoe_l867_86721

/-- Proves that the manufacturing cost of a shoe is 200, given the transportation cost,
    selling price, and profit margin. -/
theorem manufacturing_cost_of_shoe (transportation_cost : ℚ) (selling_price : ℚ) 
  (profit_margin : ℚ) (h1 : transportation_cost = 500 / 100)
  (h2 : selling_price = 246) (h3 : profit_margin = 20 / 100) :
  ∃ (manufacturing_cost : ℚ), 
    selling_price = (manufacturing_cost + transportation_cost) * (1 + profit_margin) ∧
    manufacturing_cost = 200 :=
by sorry

end manufacturing_cost_of_shoe_l867_86721


namespace speed_of_sound_346_l867_86785

/-- The speed of sound as a function of temperature -/
def speed_of_sound (t : ℝ) : ℝ := 0.6 * t + 331

/-- Theorem: When the speed of sound is 346 m/s, the temperature is 25°C -/
theorem speed_of_sound_346 :
  ∃ (t : ℝ), speed_of_sound t = 346 ∧ t = 25 := by
  sorry

end speed_of_sound_346_l867_86785


namespace six_digit_number_property_l867_86771

/-- Represents a six-digit number in the form 1ABCDE -/
def SixDigitNumber (a b c d e : Nat) : Nat :=
  100000 + 10000 * a + 1000 * b + 100 * c + 10 * d + e

theorem six_digit_number_property 
  (a b c d e : Nat) 
  (h1 : a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧ e < 10) 
  (h2 : SixDigitNumber a b c d e * 3 = SixDigitNumber b c d e a) : 
  a + b + c + d + e = 26 := by
  sorry

end six_digit_number_property_l867_86771
