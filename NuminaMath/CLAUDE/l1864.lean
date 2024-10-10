import Mathlib

namespace units_digit_of_product_units_digit_of_27_times_34_l1864_186431

/-- The units digit of a natural number -/
def unitsDigit (n : ℕ) : ℕ := n % 10

/-- The units digit of a product depends only on the units digits of its factors -/
theorem units_digit_of_product (a b : ℕ) : 
  unitsDigit (a * b) = unitsDigit (unitsDigit a * unitsDigit b) := by sorry

theorem units_digit_of_27_times_34 : unitsDigit (27 * 34) = 8 := by sorry

end units_digit_of_product_units_digit_of_27_times_34_l1864_186431


namespace optimal_weight_combination_l1864_186467

/-- Represents a combination of weights -/
structure WeightCombination where
  weight3 : ℕ
  weight5 : ℕ
  weight7 : ℕ

/-- Calculates the total weight of a combination -/
def totalWeight (c : WeightCombination) : ℕ :=
  3 * c.weight3 + 5 * c.weight5 + 7 * c.weight7

/-- Calculates the total number of weights in a combination -/
def totalWeights (c : WeightCombination) : ℕ :=
  c.weight3 + c.weight5 + c.weight7

/-- Checks if a combination is valid (totals 130 grams) -/
def isValid (c : WeightCombination) : Prop :=
  totalWeight c = 130

/-- The optimal combination of weights -/
def optimalCombination : WeightCombination :=
  { weight3 := 2, weight5 := 1, weight7 := 17 }

theorem optimal_weight_combination :
  isValid optimalCombination ∧
  (∀ c : WeightCombination, isValid c → totalWeights optimalCombination ≤ totalWeights c) :=
by sorry

end optimal_weight_combination_l1864_186467


namespace white_bread_served_l1864_186433

/-- Given that a restaurant served 0.5 loaf of wheat bread and a total of 0.9 loaves,
    prove that 0.4 loaves of white bread were served. -/
theorem white_bread_served (wheat_bread : ℝ) (total_bread : ℝ) (white_bread : ℝ)
    (h1 : wheat_bread = 0.5)
    (h2 : total_bread = 0.9)
    (h3 : white_bread = total_bread - wheat_bread) :
    white_bread = 0.4 := by
  sorry

#check white_bread_served

end white_bread_served_l1864_186433


namespace sunzi_deer_problem_l1864_186455

/-- The number of deer that enter the city -/
def total_deer : ℕ := 100

/-- The number of families in the city -/
def num_families : ℕ := 75

theorem sunzi_deer_problem :
  (num_families : ℚ) + (1 / 3 : ℚ) * num_families = total_deer :=
by sorry

end sunzi_deer_problem_l1864_186455


namespace probability_two_yellow_balls_probability_two_yellow_balls_is_one_third_l1864_186480

/-- The probability of drawing two yellow balls from a bag containing 1 white and 2 yellow balls --/
theorem probability_two_yellow_balls : ℚ :=
  let total_balls : ℕ := 3
  let yellow_balls : ℕ := 2
  let first_draw : ℚ := yellow_balls / total_balls
  let second_draw : ℚ := (yellow_balls - 1) / (total_balls - 1)
  first_draw * second_draw

/-- Proof that the probability of drawing two yellow balls is 1/3 --/
theorem probability_two_yellow_balls_is_one_third :
  probability_two_yellow_balls = 1 / 3 := by
  sorry

end probability_two_yellow_balls_probability_two_yellow_balls_is_one_third_l1864_186480


namespace grey_pairs_coincide_l1864_186471

/-- Represents the number of triangles of each color in each half of the figure -/
structure TriangleCounts where
  green : Nat
  yellow : Nat
  grey : Nat

/-- Represents the number of coinciding pairs of each type when folded -/
structure CoincidingPairs where
  green_green : Nat
  yellow_yellow : Nat
  green_grey : Nat
  grey_grey : Nat

/-- The main theorem statement -/
theorem grey_pairs_coincide (counts : TriangleCounts) (pairs : CoincidingPairs) : 
  counts.green = 4 ∧ 
  counts.yellow = 6 ∧ 
  counts.grey = 10 ∧
  pairs.green_green = 3 ∧
  pairs.yellow_yellow = 4 ∧
  pairs.green_grey = 3 →
  pairs.grey_grey = 5 := by
  sorry

end grey_pairs_coincide_l1864_186471


namespace damaged_manuscript_multiplication_l1864_186401

theorem damaged_manuscript_multiplication : ∃ (x y : ℕ), 
  x > 0 ∧ y > 0 ∧
  10 ≤ x * 8 ∧ x * 8 < 100 ∧
  100 ≤ x * (y / 10) ∧ x * (y / 10) < 1000 ∧
  y % 10 = 8 ∧
  x * y = 1176 := by
sorry

end damaged_manuscript_multiplication_l1864_186401


namespace side_c_length_l1864_186470

-- Define the triangle ABC
def triangle_ABC (A B C a b c : Real) : Prop :=
  -- Angles sum to 180°
  A + B + C = Real.pi ∧
  -- Positive side lengths
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  -- Sine law
  a / Real.sin A = b / Real.sin B ∧
  b / Real.sin B = c / Real.sin C ∧
  a / Real.sin A = c / Real.sin C

-- Theorem statement
theorem side_c_length :
  ∀ (A B C a b c : Real),
    triangle_ABC A B C a b c →
    A = Real.pi / 6 →  -- 30°
    B = 7 * Real.pi / 12 →  -- 105°
    a = 2 →
    c = 2 * Real.sqrt 2 :=
by
  sorry


end side_c_length_l1864_186470


namespace box_makers_solution_l1864_186487

/-- Represents the possible makers of the boxes -/
inductive Maker
| Bellini
| BelliniSon
| Cellini

/-- Represents the two boxes -/
inductive Box
| Gold
| Silver

/-- The inscription on the gold box -/
def gold_inscription (gold_maker silver_maker : Maker) : Prop :=
  (gold_maker = Maker.Bellini ∨ gold_maker = Maker.BelliniSon) → silver_maker = Maker.Cellini

/-- The inscription on the silver box -/
def silver_inscription (gold_maker : Maker) : Prop :=
  gold_maker = Maker.BelliniSon

/-- The theorem stating the solution to the problem -/
theorem box_makers_solution :
  ∃ (gold_maker silver_maker : Maker),
    gold_inscription gold_maker silver_maker ∧
    silver_inscription gold_maker ∧
    gold_maker = Maker.Bellini ∧
    silver_maker = Maker.Cellini :=
sorry

end box_makers_solution_l1864_186487


namespace train_passing_jogger_time_l1864_186468

/-- Time for a train to pass a jogger given their speeds, train length, and initial distance -/
theorem train_passing_jogger_time 
  (jogger_speed : ℝ) 
  (train_speed : ℝ) 
  (train_length : ℝ) 
  (initial_distance : ℝ) 
  (h1 : jogger_speed = 9) 
  (h2 : train_speed = 45) 
  (h3 : train_length = 210) 
  (h4 : initial_distance = 240) : 
  (initial_distance + train_length) / (train_speed - jogger_speed) * (3600 / 1000) = 45 := by
  sorry

end train_passing_jogger_time_l1864_186468


namespace sum_of_digits_eight_to_hundred_l1864_186448

theorem sum_of_digits_eight_to_hundred (n : ℕ) (h : n = 8^100) : 
  (n % 100 / 10 + n % 10) = 13 := by
  sorry

end sum_of_digits_eight_to_hundred_l1864_186448


namespace relationship_between_exponents_l1864_186411

theorem relationship_between_exponents 
  (a b c d : ℝ) (x y q z : ℝ) 
  (h1 : a^(x+1) = c^(q+2)) 
  (h2 : a^(x+1) = b) 
  (h3 : c^(y+3) = a^(z+4)) 
  (h4 : c^(y+3) = d) 
  (h5 : a ≠ 0) 
  (h6 : c ≠ 0) : 
  (q+2)*(z+4) = (y+3)*(x+1) := by
sorry

end relationship_between_exponents_l1864_186411


namespace roots_sum_of_squares_l1864_186494

theorem roots_sum_of_squares (a b : ℝ) : 
  a^2 - a - 2023 = 0 → b^2 - b - 2023 = 0 → a^2 + b^2 = 4047 := by
  sorry

end roots_sum_of_squares_l1864_186494


namespace small_cup_volume_l1864_186425

theorem small_cup_volume (small_cup : ℝ) (large_container : ℝ) : 
  (8 * small_cup + 5400 = large_container) →
  (12 * 530 = large_container) →
  small_cup = 120 := by
sorry

end small_cup_volume_l1864_186425


namespace inequality_proof_l1864_186486

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  Real.sqrt (a * b * (a + b)) + Real.sqrt (b * c * (b + c)) + Real.sqrt (c * a * (c + a)) >
  Real.sqrt ((a + b) * (b + c) * (c + a)) := by
  sorry

end inequality_proof_l1864_186486


namespace sum_of_roots_is_51_l1864_186472

-- Define the function f
def f (x : ℝ) : ℝ := 16 * x + 3

-- Define the inverse function of f
noncomputable def f_inv (x : ℝ) : ℝ := (x - 3) / 16

-- Theorem statement
theorem sum_of_roots_is_51 :
  ∃ (x₁ x₂ : ℝ), 
    (f_inv x₁ = f ((2 * x₁)⁻¹)) ∧
    (f_inv x₂ = f ((2 * x₂)⁻¹)) ∧
    (∀ x : ℝ, f_inv x = f ((2 * x)⁻¹) → x = x₁ ∨ x = x₂) ∧
    x₁ + x₂ = 51 :=
sorry

end sum_of_roots_is_51_l1864_186472


namespace ancient_chinese_math_problem_l1864_186488

theorem ancient_chinese_math_problem (x y : ℚ) : 
  8 * x = y + 3 ∧ 7 * x = y - 4 → (y + 3) / 8 = (y - 4) / 7 := by
  sorry

end ancient_chinese_math_problem_l1864_186488


namespace b_months_is_nine_l1864_186412

/-- Represents the pasture rental scenario -/
structure PastureRental where
  total_cost : ℝ
  a_horses : ℕ
  a_months : ℕ
  b_horses : ℕ
  c_horses : ℕ
  c_months : ℕ
  b_payment : ℝ

/-- Theorem stating that given the conditions, b put in horses for 9 months -/
theorem b_months_is_nine (pr : PastureRental)
  (h1 : pr.total_cost = 435)
  (h2 : pr.a_horses = 12)
  (h3 : pr.a_months = 8)
  (h4 : pr.b_horses = 16)
  (h5 : pr.c_horses = 18)
  (h6 : pr.c_months = 6)
  (h7 : pr.b_payment = 180) :
  ∃ x : ℝ, x = 9 ∧ 
    pr.b_payment = (pr.total_cost / (pr.a_horses * pr.a_months + pr.b_horses * x + pr.c_horses * pr.c_months)) * (pr.b_horses * x) :=
by sorry


end b_months_is_nine_l1864_186412


namespace quadratic_root_bound_l1864_186450

theorem quadratic_root_bound (a b c : ℤ) (h_distinct : ∃ (x y : ℝ), x ≠ y ∧ 0 < x ∧ x < 1 ∧ 0 < y ∧ y < 1 ∧ a * x^2 + b * x + c = 0 ∧ a * y^2 + b * y + c = 0) (h_pos : 0 < a) : 5 ≤ a := by
  sorry

end quadratic_root_bound_l1864_186450


namespace negation_of_all_integers_squared_geq_one_l1864_186420

theorem negation_of_all_integers_squared_geq_one :
  (¬ ∀ (x : ℤ), x^2 ≥ 1) ↔ (∃ (x : ℤ), x^2 < 1) := by
  sorry

end negation_of_all_integers_squared_geq_one_l1864_186420


namespace marble_difference_l1864_186456

theorem marble_difference (jar1_blue jar1_red jar2_blue jar2_red : ℕ) :
  jar1_blue + jar1_red = jar2_blue + jar2_red →
  7 * jar1_red = 3 * jar1_blue →
  3 * jar2_red = 2 * jar2_blue →
  jar1_red + jar2_red = 80 →
  jar1_blue - jar2_blue = 80 / 7 := by
sorry

end marble_difference_l1864_186456


namespace total_amount_l1864_186406

theorem total_amount (z : ℚ) (y : ℚ) (x : ℚ) 
  (hz : z = 200)
  (hy : y = 1.2 * z)
  (hx : x = 1.25 * y) :
  x + y + z = 740 := by
  sorry

end total_amount_l1864_186406


namespace jeans_purchase_savings_l1864_186439

/-- Calculates the total savings on a purchase with multiple discounts and a rebate --/
theorem jeans_purchase_savings 
  (original_price : ℝ)
  (sale_discount_percent : ℝ)
  (coupon_discount : ℝ)
  (credit_card_discount_percent : ℝ)
  (voucher_discount_percent : ℝ)
  (rebate : ℝ)
  (sales_tax_percent : ℝ)
  (h1 : original_price = 200)
  (h2 : sale_discount_percent = 30)
  (h3 : coupon_discount = 15)
  (h4 : credit_card_discount_percent = 15)
  (h5 : voucher_discount_percent = 10)
  (h6 : rebate = 20)
  (h7 : sales_tax_percent = 8.25) :
  ∃ (savings : ℝ), abs (savings - 116.49) < 0.01 := by
  sorry

end jeans_purchase_savings_l1864_186439


namespace james_baked_1380_muffins_l1864_186484

/-- The number of muffins Arthur baked -/
def arthur_muffins : ℕ := 115

/-- The multiplier for James's muffins compared to Arthur's -/
def james_multiplier : ℕ := 12

/-- The number of muffins James baked -/
def james_muffins : ℕ := arthur_muffins * james_multiplier

/-- Proof that James baked 1380 muffins -/
theorem james_baked_1380_muffins : james_muffins = 1380 := by
  sorry

end james_baked_1380_muffins_l1864_186484


namespace binomial_30_3_l1864_186432

theorem binomial_30_3 : Nat.choose 30 3 = 4060 := by
  sorry

end binomial_30_3_l1864_186432


namespace remainder_sum_l1864_186437

theorem remainder_sum (D : ℕ) (h1 : D > 0) (h2 : 242 % D = 4) (h3 : 698 % D = 8) :
  (242 + 698) % D = 12 := by
  sorry

end remainder_sum_l1864_186437


namespace water_drainage_proof_l1864_186424

/-- Represents the fraction of water remaining after n steps of draining -/
def waterRemaining (n : ℕ) : ℚ :=
  2 / (n + 2 : ℚ)

/-- The number of steps after which one-seventh of the water remains -/
def stepsToOneSeventh : ℕ := 12

theorem water_drainage_proof :
  waterRemaining stepsToOneSeventh = 1 / 7 := by
  sorry

end water_drainage_proof_l1864_186424


namespace tan_22_5_deg_over_one_minus_tan_squared_l1864_186496

theorem tan_22_5_deg_over_one_minus_tan_squared (
  angle_22_5 : ℝ)
  (h1 : 45 * Real.pi / 180 = 2 * angle_22_5)
  (h2 : Real.tan (45 * Real.pi / 180) = 1)
  (h3 : ∀ θ : ℝ, Real.tan (2 * θ) = (2 * Real.tan θ) / (1 - Real.tan θ ^ 2)) :
  Real.tan angle_22_5 / (1 - Real.tan angle_22_5 ^ 2) = 1 / 2 := by
sorry

end tan_22_5_deg_over_one_minus_tan_squared_l1864_186496


namespace parabola_translation_l1864_186465

/-- Represents a parabola in the form y = a(x - h)^2 + k -/
structure Parabola where
  a : ℝ
  h : ℝ
  k : ℝ

/-- Translates a parabola horizontally and vertically -/
def translate (p : Parabola) (dx dy : ℝ) : Parabola :=
  { a := p.a
    h := p.h - dx
    k := p.k + dy }

theorem parabola_translation (p : Parabola) :
  p.a = -1/3 ∧ p.h = 5 ∧ p.k = 3 →
  let p' := translate p 5 3
  p'.a = -1/3 ∧ p'.h = 0 ∧ p'.k = 6 := by
  sorry

end parabola_translation_l1864_186465


namespace sum_of_cuboid_vertices_l1864_186421

/-- Given that the sum of edges and faces of all cuboids is 216, 
    prove that the sum of vertices of all cuboids is 96. -/
theorem sum_of_cuboid_vertices (n : ℕ) : 
  n * (12 + 6) = 216 → n * 8 = 96 := by
  sorry

end sum_of_cuboid_vertices_l1864_186421


namespace inequality_holds_iff_l1864_186402

theorem inequality_holds_iff (a : ℝ) : 
  (∀ x : ℝ, (4:ℝ)^(x^2) + 2*(2*a+1) * (2:ℝ)^(x^2) + 4*a^2 - 3 > 0) ↔ 
  (a < -1 ∨ a ≥ Real.sqrt 3 / 2) :=
sorry

end inequality_holds_iff_l1864_186402


namespace math_club_team_selection_l1864_186426

def math_club_selection (boys girls : ℕ) (team_size : ℕ) (team_boys team_girls : ℕ) : ℕ :=
  Nat.choose boys team_boys * Nat.choose girls team_girls

theorem math_club_team_selection :
  math_club_selection 7 9 6 4 2 = 1260 :=
by sorry

end math_club_team_selection_l1864_186426


namespace alternate_color_probability_l1864_186475

/-- The probability of drawing BWBW from a box with 5 white and 6 black balls -/
theorem alternate_color_probability :
  let initial_white : ℕ := 5
  let initial_black : ℕ := 6
  let total_balls : ℕ := initial_white + initial_black
  let prob_first_black : ℚ := initial_black / total_balls
  let prob_second_white : ℚ := initial_white / (total_balls - 1)
  let prob_third_black : ℚ := (initial_black - 1) / (total_balls - 2)
  let prob_fourth_white : ℚ := (initial_white - 1) / (total_balls - 3)
  prob_first_black * prob_second_white * prob_third_black * prob_fourth_white = 2 / 33 :=
by sorry

end alternate_color_probability_l1864_186475


namespace no_real_roots_for_distinct_abc_l1864_186464

theorem no_real_roots_for_distinct_abc (a b c : ℝ) 
  (hab : a ≠ b) (hac : a ≠ c) (hbc : b ≠ c) : 
  let discriminant := 4 * (a + b + c)^2 - 12 * (a^2 + b^2 + c^2)
  discriminant < 0 := by
sorry

end no_real_roots_for_distinct_abc_l1864_186464


namespace power_inequality_l1864_186491

theorem power_inequality (x : ℝ) (h1 : 0 < x) (h2 : x < 1) :
  (0.2 : ℝ) ^ x < (1/2 : ℝ) ^ x ∧ (1/2 : ℝ) ^ x < 2 ^ x := by
  sorry

end power_inequality_l1864_186491


namespace catherine_pens_problem_l1864_186415

theorem catherine_pens_problem (initial_pens initial_pencils : ℕ) :
  initial_pens = initial_pencils →
  initial_pens - 7 * 8 + initial_pencils - 7 * 6 = 22 →
  initial_pens = 60 :=
by
  sorry

end catherine_pens_problem_l1864_186415


namespace dyck_path_correspondence_l1864_186435

/-- A Dyck path is a lattice path of upsteps and downsteps that starts at the origin and never dips below the x-axis. -/
def DyckPath (n : ℕ) : Type := sorry

/-- A return in a Dyck path is a maximal sequence of contiguous downsteps that terminates on the x-axis. -/
def Return (path : DyckPath n) : Type := sorry

/-- Predicate to check if a return has even length -/
def hasEvenLengthReturn (path : DyckPath n) : Prop := sorry

/-- The number of Dyck n-paths -/
def numDyckPaths (n : ℕ) : ℕ := sorry

/-- The number of Dyck n-paths with no return of even length -/
def numDyckPathsNoEvenReturn (n : ℕ) : ℕ := sorry

/-- Theorem: The number of Dyck n-paths with no return of even length is equal to the number of Dyck (n-1) paths -/
theorem dyck_path_correspondence (n : ℕ) (h : n ≥ 1) :
  numDyckPathsNoEvenReturn n = numDyckPaths (n - 1) := by sorry

end dyck_path_correspondence_l1864_186435


namespace line_point_sum_l1864_186453

/-- The line equation y = -5/3x + 15 -/
def line_equation (x y : ℝ) : Prop := y = -5/3 * x + 15

/-- Point P is where the line crosses the x-axis -/
def point_P : ℝ × ℝ := (9, 0)

/-- Point Q is where the line crosses the y-axis -/
def point_Q : ℝ × ℝ := (0, 15)

/-- Point T is on the line segment PQ -/
def point_T_on_PQ (r s : ℝ) : Prop :=
  ∃ (t : ℝ), 0 ≤ t ∧ t ≤ 1 ∧
  r = t * point_P.1 + (1 - t) * point_Q.1 ∧
  s = t * point_P.2 + (1 - t) * point_Q.2

/-- The area of triangle POQ is twice the area of triangle TOP -/
def area_condition (r s : ℝ) : Prop :=
  abs ((point_P.1 - 0) * (point_Q.2 - 0) - (point_Q.1 - 0) * (point_P.2 - 0)) / 2 =
  2 * abs ((point_P.1 - 0) * (s - 0) - (r - 0) * (point_P.2 - 0)) / 2

theorem line_point_sum (r s : ℝ) :
  line_equation r s →
  point_T_on_PQ r s →
  area_condition r s →
  r + s = 12 := by sorry

end line_point_sum_l1864_186453


namespace quadratic_factorization_l1864_186422

theorem quadratic_factorization (y a b : ℤ) :
  2 * y^2 - 5 * y - 12 = (2 * y + a) * (y + b) →
  a - b = 1 := by
sorry

end quadratic_factorization_l1864_186422


namespace area_between_line_and_curve_l1864_186418

/-- The area enclosed by the line y=4x and the curve y=x^3 is 8 -/
theorem area_between_line_and_curve : 
  let f (x : ℝ) := 4 * x
  let g (x : ℝ) := x^3
  ∫ x in (-2)..2, |f x - g x| = 8 := by sorry

end area_between_line_and_curve_l1864_186418


namespace quadratic_function_monotonicity_l1864_186460

-- Define the function
def f (x : ℝ) : ℝ := x^2

-- State the theorem
theorem quadratic_function_monotonicity :
  (∀ x₁ x₂ : ℝ, x₁ < x₂ ∧ x₂ < 0 → f x₁ > f x₂) ∧
  (∀ x₁ x₂ : ℝ, 0 < x₁ ∧ x₁ < x₂ → f x₁ < f x₂) :=
by sorry

end quadratic_function_monotonicity_l1864_186460


namespace linear_regression_passes_through_mean_point_l1864_186403

/-- Linear regression equation passes through the mean point -/
theorem linear_regression_passes_through_mean_point 
  (b a x_bar y_bar : ℝ) : 
  y_bar = b * x_bar + a :=
sorry

end linear_regression_passes_through_mean_point_l1864_186403


namespace distinct_z_values_l1864_186490

def is_two_digit (n : ℤ) : Prop := 10 ≤ n ∧ n ≤ 99

def reverse_digits (n : ℤ) : ℤ :=
  10 * (n % 10) + (n / 10)

def z (x : ℤ) : ℤ := |x - reverse_digits x|

theorem distinct_z_values (x : ℤ) (hx : is_two_digit x) :
  ∃ (S : Finset ℤ), (∀ y, is_two_digit y → z y ∈ S) ∧ Finset.card S = 8 := by
  sorry

#check distinct_z_values

end distinct_z_values_l1864_186490


namespace angle_properties_l1864_186461

theorem angle_properties (α : Real) 
  (h1 : 0 < α) (h2 : α < π) (h3 : Real.sin α + Real.cos α = 1/5) : 
  (Real.tan α = -4/3) ∧ 
  (Real.sin α ^ 2 - 3 * Real.sin α * Real.cos α - 4 * Real.cos α ^ 2 = 16/25) := by
  sorry

end angle_properties_l1864_186461


namespace regression_line_at_25_l1864_186413

/-- The regression line equation is y = 0.5x - 0.81 -/
def regression_line (x : ℝ) : ℝ := 0.5 * x - 0.81

/-- Theorem: Given the regression line equation y = 0.5x - 0.81, when x = 25, y = 11.69 -/
theorem regression_line_at_25 : regression_line 25 = 11.69 := by
  sorry

end regression_line_at_25_l1864_186413


namespace parabola_expression_l1864_186483

theorem parabola_expression (f : ℝ → ℝ) (h1 : f (-3) = 0) (h2 : f 1 = 0) (h3 : f 0 = 2) :
  ∀ x, f x = -2/3 * x^2 - 4/3 * x + 2 :=
by sorry

end parabola_expression_l1864_186483


namespace chess_tournament_players_l1864_186428

/-- Represents a chess tournament with the given conditions -/
structure ChessTournament where
  n : ℕ  -- Number of players not in the lowest 15
  -- Each player plays exactly one match against every other player
  total_games : ℕ := (n + 15).choose 2
  -- Points from games between n players not in the lowest 15
  points_among_n : ℕ := n.choose 2
  -- Points earned by n players against the lowest 15
  points_n_vs_15 : ℕ := n.choose 2
  -- Points earned by the lowest 15 players among themselves
  points_among_15 : ℕ := 105
  -- Total points in the tournament
  total_points : ℕ := 2 * points_among_n + 2 * points_among_15

/-- The theorem stating that the total number of players in the tournament is 50 -/
theorem chess_tournament_players (t : ChessTournament) : t.n + 15 = 50 := by
  sorry

end chess_tournament_players_l1864_186428


namespace parabola_equation_l1864_186434

/-- A parabola with vertex at the origin, axis of symmetry along the x-axis,
    and passing through the point (-2, 2√2) has the equation y^2 = -4x. -/
theorem parabola_equation (p : ℝ × ℝ) 
    (vertex_origin : p.1 = 0 ∧ p.2 = 0)
    (axis_x : ∀ (x y : ℝ), y^2 = -4*x → y^2 = -4*(-x))
    (point_on_parabola : (-2)^2 + (2*Real.sqrt 2)^2 = -4*(-2)) :
  ∀ (x y : ℝ), y^2 = -4*x ↔ (x, y) ∈ {(a, b) | b^2 = -4*a} :=
sorry

end parabola_equation_l1864_186434


namespace correct_equation_l1864_186410

theorem correct_equation (y : ℝ) : -9 * y^2 + 16 * y^2 = 7 * y^2 := by
  sorry

end correct_equation_l1864_186410


namespace exists_cousin_180_problems_l1864_186416

/-- Represents the homework scenario for me and my cousin -/
structure HomeworkScenario where
  p : ℕ+  -- My rate (problems per hour)
  t : ℕ+  -- Time I take to finish homework (hours)
  n : ℕ   -- Number of problems I complete

/-- Calculates the number of problems my cousin does -/
def cousin_problems (s : HomeworkScenario) : ℕ :=
  ((3 * s.p.val - 5) * (s.t.val + 3)) / 2

/-- Theorem stating that there exists a scenario where my cousin does 180 problems -/
theorem exists_cousin_180_problems :
  ∃ (s : HomeworkScenario), 
    s.p ≥ 15 ∧ 
    s.n = s.p.val * s.t.val ∧ 
    cousin_problems s = 180 := by
  sorry

end exists_cousin_180_problems_l1864_186416


namespace union_of_A_and_B_l1864_186499

-- Define sets A and B
def A : Set ℝ := {x | 3 ≤ x ∧ x < 7}
def B : Set ℝ := {x | 2 < x ∧ x < 10}

-- Theorem statement
theorem union_of_A_and_B : A ∪ B = {x | 2 < x ∧ x < 10} := by
  sorry

end union_of_A_and_B_l1864_186499


namespace time_difference_walk_vs_bicycle_l1864_186473

/-- Represents the number of blocks from Henrikh's home to his office -/
def distance : ℕ := 12

/-- Represents the time in minutes to walk one block -/
def walkingTimePerBlock : ℚ := 1

/-- Represents the time in minutes to ride a bicycle for one block -/
def bicycleTimePerBlock : ℚ := 20 / 60

/-- Calculates the total time to travel the distance by walking -/
def walkingTime : ℚ := distance * walkingTimePerBlock

/-- Calculates the total time to travel the distance by bicycle -/
def bicycleTime : ℚ := distance * bicycleTimePerBlock

theorem time_difference_walk_vs_bicycle :
  walkingTime - bicycleTime = 8 := by sorry

end time_difference_walk_vs_bicycle_l1864_186473


namespace min_value_x_plus_four_over_x_l1864_186407

theorem min_value_x_plus_four_over_x :
  ∃ (min : ℝ), min > 0 ∧
  (∀ x : ℝ, x > 0 → x + 4 / x ≥ min) ∧
  (∃ x : ℝ, x > 0 ∧ x + 4 / x = min) :=
by sorry

end min_value_x_plus_four_over_x_l1864_186407


namespace line_segment_endpoint_l1864_186479

/-- Given a line segment with midpoint (3, -1) and one endpoint at (7, 2),
    prove that the other endpoint is at (-1, -4). -/
theorem line_segment_endpoint (midpoint endpoint1 endpoint2 : ℝ × ℝ) :
  midpoint = (3, -1) →
  endpoint1 = (7, 2) →
  midpoint = ((endpoint1.1 + endpoint2.1) / 2, (endpoint1.2 + endpoint2.2) / 2) →
  endpoint2 = (-1, -4) := by
  sorry

end line_segment_endpoint_l1864_186479


namespace line_segment_proportions_l1864_186400

theorem line_segment_proportions (a b x : ℝ) : 
  (a / b = 3 / 2) → 
  (a + 2 * b = 28) → 
  (x^2 = a * b) →
  (a = 12 ∧ b = 8 ∧ x = 4 * Real.sqrt 6) := by
sorry

end line_segment_proportions_l1864_186400


namespace rulers_equation_initial_rulers_count_l1864_186454

/-- The number of rulers initially in the drawer -/
def initial_rulers : ℕ := sorry

/-- The number of rulers added to the drawer -/
def added_rulers : ℕ := 14

/-- The final number of rulers in the drawer -/
def final_rulers : ℕ := 25

/-- Theorem stating that the initial number of rulers plus the added rulers equals the final number of rulers -/
theorem rulers_equation : initial_rulers + added_rulers = final_rulers := by sorry

/-- Theorem proving that the initial number of rulers is 11 -/
theorem initial_rulers_count : initial_rulers = 11 := by sorry

end rulers_equation_initial_rulers_count_l1864_186454


namespace sqrt_product_simplification_l1864_186457

theorem sqrt_product_simplification (x : ℝ) (hx : x ≥ 0) :
  Real.sqrt (50 * x) * Real.sqrt (18 * x) * Real.sqrt (8 * x) = 60 * x * Real.sqrt (2 * x) :=
by sorry

end sqrt_product_simplification_l1864_186457


namespace circle_radius_sqrt_61_l1864_186482

/-- Given a circle with center on the x-axis passing through points (2,5) and (3,2),
    prove that its radius is √61. -/
theorem circle_radius_sqrt_61 (x : ℝ) :
  (∀ (y : ℝ), y = 0 →  -- Center is on x-axis
    (x - 2)^2 + (y - 5)^2 = (x - 3)^2 + (y - 2)^2) →  -- Points (2,5) and (3,2) are equidistant from center
  (x - 2)^2 + 5^2 = 61 :=
by
  sorry

end circle_radius_sqrt_61_l1864_186482


namespace impossible_to_divide_into_l_pieces_l1864_186423

/-- Represents a chessboard cell --/
inductive Cell
| Black
| White

/-- Represents an L-shaped piece --/
structure LPiece :=
(cells : Fin 4 → Cell)

/-- Represents a chessboard --/
def Chessboard := Fin 8 → Fin 8 → Cell

/-- Returns the color of a cell based on its coordinates --/
def cellColor (row col : Fin 8) : Cell :=
  if (row.val + col.val) % 2 = 0 then Cell.Black else Cell.White

/-- Checks if a cell is in the central 2x2 square --/
def isCentralSquare (row col : Fin 8) : Prop :=
  (row = 3 ∨ row = 4) ∧ (col = 3 ∨ col = 4)

/-- Represents the modified chessboard with central 2x2 square removed --/
def ModifiedChessboard : Type :=
  { cell : Fin 8 × Fin 8 // ¬isCentralSquare cell.1 cell.2 }

/-- The main theorem stating that it's impossible to divide the modified chessboard into L-shaped pieces --/
theorem impossible_to_divide_into_l_pieces :
  ¬∃ (pieces : List LPiece), 
    (pieces.length > 0) ∧ 
    (∀ (cell : ModifiedChessboard), ∃! (piece : LPiece) (i : Fin 4), 
      piece ∈ pieces ∧ piece.cells i = cellColor cell.val.1 cell.val.2) :=
sorry

end impossible_to_divide_into_l_pieces_l1864_186423


namespace cos_2017pi_over_3_l1864_186438

theorem cos_2017pi_over_3 : Real.cos (2017 * Real.pi / 3) = 1 / 2 := by
  sorry

end cos_2017pi_over_3_l1864_186438


namespace project_hours_l1864_186449

theorem project_hours (total_hours : ℕ) (kate_hours : ℕ) : 
  total_hours = 135 → 
  2 * kate_hours + kate_hours + 6 * kate_hours = total_hours →
  6 * kate_hours - kate_hours = 75 :=
by
  sorry

end project_hours_l1864_186449


namespace equation_implies_conditions_l1864_186408

theorem equation_implies_conditions (a b c d : ℝ) 
  (h : (a^2 + b^2) / (b^2 + c^2) = (c^2 + d^2) / (d^2 + a^2)) :
  a = c ∨ a = -c ∨ a^2 - c^2 + d^2 = b^2 := by
sorry

end equation_implies_conditions_l1864_186408


namespace prob_adjacent_knights_l1864_186442

/-- The number of knights at the round table -/
def n : ℕ := 30

/-- The number of knights chosen for the quest -/
def k : ℕ := 4

/-- The probability of choosing k knights from n such that at least two are adjacent -/
def prob_adjacent (n k : ℕ) : ℚ :=
  1 - (n * (n - 3) * (n - 4) * (n - 5) : ℚ) / (n.choose k : ℚ)

/-- The theorem stating the probability of choosing 4 knights from 30 such that at least two are adjacent -/
theorem prob_adjacent_knights : prob_adjacent n k = 53 / 183 := by sorry

end prob_adjacent_knights_l1864_186442


namespace coeff_x2y2_is_168_l1864_186452

/-- The coefficient of x^2y^2 in the expansion of ((1+x)^8(1+y)^4) -/
def coeff_x2y2 : ℕ :=
  (Nat.choose 8 2) * (Nat.choose 4 2)

/-- Theorem stating that the coefficient of x^2y^2 in ((1+x)^8(1+y)^4) is 168 -/
theorem coeff_x2y2_is_168 : coeff_x2y2 = 168 := by
  sorry

end coeff_x2y2_is_168_l1864_186452


namespace candle_flower_groupings_l1864_186445

def choose (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

theorem candle_flower_groupings :
  (choose 6 3) * (choose 15 12) = 9100 := by
sorry

end candle_flower_groupings_l1864_186445


namespace license_plate_count_l1864_186405

/-- The number of possible digits (0-9) -/
def num_digits : ℕ := 10

/-- The number of possible letters (A-Z) -/
def num_letters : ℕ := 26

/-- The number of digits in a license plate -/
def digits_count : ℕ := 5

/-- The number of letters in a license plate -/
def letters_count : ℕ := 3

/-- The number of possible positions for the letter block -/
def block_positions : ℕ := digits_count + 1

/-- The total number of distinct license plates -/
def total_license_plates : ℕ := 
  block_positions * (num_digits ^ digits_count) * (num_letters ^ letters_count)

theorem license_plate_count : total_license_plates = 105456000 := by
  sorry

end license_plate_count_l1864_186405


namespace max_stones_upper_bound_max_stones_achievable_max_stones_theorem_l1864_186474

/-- Represents the state of the piles -/
def PileState := List Nat

/-- The initial state of the piles -/
def initial_state : PileState := List.replicate 2009 2

/-- The operation of transferring stones -/
def transfer (state : PileState) : PileState :=
  sorry

/-- Predicate to check if a state is valid -/
def is_valid_state (state : PileState) : Prop :=
  state.sum = 2009 * 2 ∧ state.all (· ≥ 1)

/-- The maximum number of stones in any pile -/
def max_stones (state : PileState) : Nat :=
  state.foldl Nat.max 0

theorem max_stones_upper_bound :
  ∀ (state : PileState), is_valid_state state → max_stones state ≤ 2010 :=
  sorry

theorem max_stones_achievable :
  ∃ (state : PileState), is_valid_state state ∧ max_stones state = 2010 :=
  sorry

theorem max_stones_theorem :
  (∀ (state : PileState), is_valid_state state → max_stones state ≤ 2010) ∧
  (∃ (state : PileState), is_valid_state state ∧ max_stones state = 2010) :=
  sorry

end max_stones_upper_bound_max_stones_achievable_max_stones_theorem_l1864_186474


namespace symmetry_of_shifted_function_l1864_186477

-- Define a function f
variable (f : ℝ → ℝ)

-- Define the property of f being even when shifted by 2
def is_even_shifted (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x + 2) = f (x + 2)

-- Define the symmetry axis of a function
def symmetry_axis (g : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x, g (a + x) = g (a - x)

-- Theorem statement
theorem symmetry_of_shifted_function (f : ℝ → ℝ) 
  (h : is_even_shifted f) : 
  symmetry_axis (fun x ↦ f (x - 1) + 2) 3 := by
  sorry

end symmetry_of_shifted_function_l1864_186477


namespace gold_tetrahedron_volume_l1864_186441

/-- Represents a cube with alternately colored vertices -/
structure ColoredCube where
  sideLength : ℝ
  vertexColors : Fin 8 → Bool  -- True for gold, False for red

/-- Calculates the volume of a tetrahedron formed by selected vertices of a cube -/
def tetrahedronVolume (cube : ColoredCube) (selectVertex : Fin 8 → Bool) : ℝ :=
  sorry

/-- The main theorem stating the volume of the gold-colored tetrahedron -/
theorem gold_tetrahedron_volume (cube : ColoredCube) 
  (h1 : cube.sideLength = 8)
  (h2 : ∀ i : Fin 8, cube.vertexColors i = (i.val % 2 == 0))  -- Alternating colors
  : tetrahedronVolume cube cube.vertexColors = 170.67 := by
  sorry

end gold_tetrahedron_volume_l1864_186441


namespace system_solution_l1864_186436

theorem system_solution : 
  ∃! (x y : ℝ), (2 * x + y = 6) ∧ (x - y = 3) ∧ (x = 3) ∧ (y = 0) := by
  sorry

end system_solution_l1864_186436


namespace lcm_of_20_45_75_l1864_186489

theorem lcm_of_20_45_75 : Nat.lcm (Nat.lcm 20 45) 75 = 900 := by
  sorry

end lcm_of_20_45_75_l1864_186489


namespace workers_payment_schedule_l1864_186458

theorem workers_payment_schedule (total_days : ℕ) (pay_per_day_worked : ℤ) (pay_returned_per_day_not_worked : ℤ) 
  (h1 : total_days = 30)
  (h2 : pay_per_day_worked = 100)
  (h3 : pay_returned_per_day_not_worked = 25)
  (h4 : ∃ (days_worked days_not_worked : ℕ), 
    days_worked + days_not_worked = total_days ∧ 
    pay_per_day_worked * days_worked - pay_returned_per_day_not_worked * days_not_worked = 0) :
  ∃ (days_not_worked : ℕ), days_not_worked = 24 := by
sorry

end workers_payment_schedule_l1864_186458


namespace inverse_of_B_squared_l1864_186443

def B_inv : Matrix (Fin 2) (Fin 2) ℤ := !![1, 4; -2, -7]

theorem inverse_of_B_squared :
  let B_squared_inv : Matrix (Fin 2) (Fin 2) ℤ := !![(-7), (-24); 12, 41]
  (B_inv * B_inv) * (B_inv⁻¹ * B_inv⁻¹) = 1 ∧ (B_inv⁻¹ * B_inv⁻¹) * (B_inv * B_inv) = 1 := by
  sorry

end inverse_of_B_squared_l1864_186443


namespace not_coprime_sum_equal_l1864_186495

/-- For any two natural numbers a and b, if a+n and b+n are not coprime for all natural numbers n, then a = b. -/
theorem not_coprime_sum_equal (a b : ℕ) 
  (h : ∀ n : ℕ, ¬ Nat.Coprime (a + n) (b + n)) : 
  a = b := by
  sorry

end not_coprime_sum_equal_l1864_186495


namespace marble_count_l1864_186492

/-- The number of marbles each person has --/
structure Marbles where
  ed : ℕ
  doug : ℕ
  charlie : ℕ

/-- The initial state of marbles before Ed lost some --/
def initial_marbles : Marbles → Marbles
| ⟨ed, doug, charlie⟩ => ⟨ed + 20, doug, charlie⟩

theorem marble_count (m : Marbles) :
  (initial_marbles m).ed = (initial_marbles m).doug + 12 →
  m.ed = 17 →
  m.charlie = 4 * m.doug →
  m.doug = 25 ∧ m.charlie = 100 := by
  sorry

end marble_count_l1864_186492


namespace smallest_m_plus_n_l1864_186462

/-- Given that m and n are natural numbers satisfying 3n^3 = 5m^2, 
    the smallest possible value of m + n is 60. -/
theorem smallest_m_plus_n : ∃ (m n : ℕ), 
  (3 * n^3 = 5 * m^2) ∧ 
  (m + n = 60) ∧ 
  (∀ (m' n' : ℕ), (3 * n'^3 = 5 * m'^2) → (m' + n' ≥ 60)) := by
  sorry

end smallest_m_plus_n_l1864_186462


namespace event_attendance_l1864_186404

theorem event_attendance (total : ℕ) (movie picnic gaming : ℕ) 
  (movie_picnic movie_gaming picnic_gaming : ℕ) (all_three : ℕ) 
  (h1 : total = 200)
  (h2 : movie = 50)
  (h3 : picnic = 80)
  (h4 : gaming = 60)
  (h5 : movie_picnic = 35)
  (h6 : movie_gaming = 10)
  (h7 : picnic_gaming = 20)
  (h8 : all_three = 8) :
  movie + picnic + gaming - (movie_picnic + movie_gaming + picnic_gaming) + all_three = 133 := by
sorry

end event_attendance_l1864_186404


namespace fifteenth_student_age_l1864_186446

theorem fifteenth_student_age
  (total_students : Nat)
  (average_age : ℕ)
  (group1_students : Nat)
  (group1_average : ℕ)
  (group2_students : Nat)
  (group2_average : ℕ)
  (h1 : total_students = 15)
  (h2 : average_age = 15)
  (h3 : group1_students = 6)
  (h4 : group1_average = 14)
  (h5 : group2_students = 8)
  (h6 : group2_average = 16)
  (h7 : group1_students + group2_students + 1 = total_students) :
  total_students * average_age - (group1_students * group1_average + group2_students * group2_average) = 13 :=
by sorry

end fifteenth_student_age_l1864_186446


namespace value_difference_is_50p_minus_250_l1864_186463

/-- The value of a fifty-cent coin in pennies -/
def fifty_cent_value : ℕ := 50

/-- The number of fifty-cent coins Liam has -/
def liam_coins (p : ℕ) : ℕ := 3 * p + 2

/-- The number of fifty-cent coins Mia has -/
def mia_coins (p : ℕ) : ℕ := 2 * p + 7

/-- The difference in total value (in pennies) between Liam's and Mia's fifty-cent coins -/
def value_difference (p : ℕ) : ℤ := fifty_cent_value * (liam_coins p - mia_coins p)

theorem value_difference_is_50p_minus_250 (p : ℕ) :
  value_difference p = 50 * p - 250 := by sorry

end value_difference_is_50p_minus_250_l1864_186463


namespace sum_of_specific_terms_l1864_186414

def a (n : ℕ+) : ℕ := 2 * n.val - 1

theorem sum_of_specific_terms : 
  a 4 + a 5 + a 6 + a 7 + a 8 = 55 := by sorry

end sum_of_specific_terms_l1864_186414


namespace yellow_ball_packs_l1864_186459

theorem yellow_ball_packs (red_packs green_packs balls_per_pack total_balls : ℕ) 
  (h1 : red_packs = 3)
  (h2 : green_packs = 8)
  (h3 : balls_per_pack = 19)
  (h4 : total_balls = 399) :
  ∃ yellow_packs : ℕ, 
    yellow_packs * balls_per_pack + red_packs * balls_per_pack + green_packs * balls_per_pack = total_balls ∧
    yellow_packs = 10 := by
  sorry

end yellow_ball_packs_l1864_186459


namespace decreasing_linear_function_l1864_186429

def is_decreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f y < f x

theorem decreasing_linear_function (k : ℝ) :
  is_decreasing (λ x : ℝ => (k + 1) * x) → k < -1 :=
by sorry

end decreasing_linear_function_l1864_186429


namespace incorrect_inequality_l1864_186419

theorem incorrect_inequality (x y : ℝ) (h : x > y) : ¬(-3 * x > -3 * y) := by
  sorry

end incorrect_inequality_l1864_186419


namespace smallest_consecutive_even_sum_l1864_186497

theorem smallest_consecutive_even_sum (a : ℤ) : 
  (∃ (b c d e : ℤ), 
    (a + 2 = b) ∧ (b + 2 = c) ∧ (c + 2 = d) ∧ (d + 2 = e) ∧  -- Consecutive even integers
    (a % 2 = 0) ∧                                            -- First number is even
    (a + b + c + d + e = 380)) →                             -- Sum is 380
  a = 72 := by
sorry

end smallest_consecutive_even_sum_l1864_186497


namespace cows_equivalent_to_buffaloes_or_oxen_l1864_186466

-- Define the variables
variable (B : ℕ) -- Daily fodder consumption of a buffalo
variable (C : ℕ) -- Daily fodder consumption of a cow
variable (O : ℕ) -- Daily fodder consumption of an ox
variable (F : ℕ) -- Total available fodder

-- Define the conditions
axiom buffalo_ox_equiv : 3 * B = 2 * O
axiom initial_fodder : F = (15 * B + 8 * O + 24 * C) * 48
axiom additional_cattle : F = (30 * B + 64 * C) * 24

-- The theorem to prove
theorem cows_equivalent_to_buffaloes_or_oxen : ∃ x : ℕ, x = 2 ∧ 3 * B = x * C := by
  sorry

end cows_equivalent_to_buffaloes_or_oxen_l1864_186466


namespace min_value_theorem_l1864_186427

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (x + 1/y) * (x + 1/y - 100) + (y + 1/x) * (y + 1/x - 100) ≥ -2500 ∧
  (∃ x₀ : ℝ, x₀ > 0 ∧ x₀ + 1/x₀ = 50 ∧
    (x₀ + 1/x₀) * (x₀ + 1/x₀ - 100) + (x₀ + 1/x₀) * (x₀ + 1/x₀ - 100) = -2500) :=
by sorry

end min_value_theorem_l1864_186427


namespace diet_soda_bottles_l1864_186430

/-- Given a grocery store inventory, calculate the number of diet soda bottles -/
theorem diet_soda_bottles (total : ℕ) (regular : ℕ) (h1 : total = 17) (h2 : regular = 9) :
  total - regular = 8 := by
  sorry

end diet_soda_bottles_l1864_186430


namespace sum_of_squares_equals_half_sum_of_other_squares_l1864_186476

theorem sum_of_squares_equals_half_sum_of_other_squares (a b : ℝ) :
  a^2 + b^2 = ((a + b)^2 + (a - b)^2) / 2 := by
  sorry

end sum_of_squares_equals_half_sum_of_other_squares_l1864_186476


namespace remainder_of_sum_of_primes_l1864_186469

def first_eight_primes : List Nat := [2, 3, 5, 7, 11, 13, 17, 19]

theorem remainder_of_sum_of_primes :
  (3 * (List.sum (List.take 7 first_eight_primes))) % (List.get! first_eight_primes 7) = 3 := by
  sorry

end remainder_of_sum_of_primes_l1864_186469


namespace max_sum_is_42_l1864_186481

/-- Represents the configuration of numbers in the squares -/
structure SquareConfig where
  a : ℕ
  b : ℕ
  c : ℕ
  d : ℕ
  e : ℕ
  numbers : Finset ℕ
  sum_equality : a + b + e = b + d + e
  valid_numbers : numbers = {2, 5, 8, 11, 14, 17}
  all_different : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e

/-- The maximum sum of either horizontal or vertical line is 42 -/
theorem max_sum_is_42 (config : SquareConfig) : 
  (max (config.a + config.b + config.e) (config.b + config.d + config.e)) ≤ 42 ∧ 
  ∃ (config : SquareConfig), (config.a + config.b + config.e) = 42 := by
  sorry

end max_sum_is_42_l1864_186481


namespace factorization_problems_l1864_186498

theorem factorization_problems (a b x y : ℝ) : 
  (2 * x * (a - b) - (b - a) = (a - b) * (2 * x + 1)) ∧ 
  ((x^2 + y^2)^2 - 4 * x^2 * y^2 = (x - y)^2 * (x + y)^2) := by
  sorry

end factorization_problems_l1864_186498


namespace equation_equivalence_l1864_186417

theorem equation_equivalence (x y : ℝ) 
  (hx : x ≠ 0 ∧ x ≠ 5) (hy : y ≠ 0 ∧ y ≠ 7) : 
  (3 / x + 2 / y = 1 / 3) ↔ (x = 9 * y / (y - 6)) :=
sorry

end equation_equivalence_l1864_186417


namespace birdseed_solution_l1864_186478

/-- The number of boxes of birdseed Leah already had in the pantry -/
def birdseed_problem (new_boxes : ℕ) (parrot_consumption : ℕ) (cockatiel_consumption : ℕ) 
  (box_content : ℕ) (weeks : ℕ) : ℕ :=
  let total_consumption := parrot_consumption + cockatiel_consumption
  let total_needed := total_consumption * weeks
  let total_boxes := (total_needed + box_content - 1) / box_content
  total_boxes - new_boxes

/-- Theorem stating the solution to the birdseed problem -/
theorem birdseed_solution : 
  birdseed_problem 3 100 50 225 12 = 5 := by
  sorry

end birdseed_solution_l1864_186478


namespace circle_center_quadrant_l1864_186440

theorem circle_center_quadrant (α : Real) :
  (∃ x y : Real, x^2 * Real.cos α - y^2 * Real.sin α + 2 = 0) →  -- hyperbola condition
  let center := (- Real.cos α, Real.sin α)
  (center.1 < 0 ∧ center.2 > 0) ∨ (center.1 > 0 ∧ center.2 < 0) :=
by sorry

end circle_center_quadrant_l1864_186440


namespace yankees_to_mets_ratio_l1864_186451

/-- Represents the number of fans for each team -/
structure FanCounts where
  yankees : ℕ
  mets : ℕ
  red_sox : ℕ

/-- The total number of baseball fans in the town -/
def total_fans : ℕ := 330

/-- The given number of NY Mets fans -/
def mets_fans : ℕ := 88

/-- Theorem stating that the ratio of NY Yankees fans to NY Mets fans is 3:2 -/
theorem yankees_to_mets_ratio (fc : FanCounts) : 
  fc.yankees = fc.mets * 3 / 2 ∧ 
  fc.mets = mets_fans ∧ 
  fc.red_sox = fc.mets * 5 / 4 ∧ 
  fc.yankees + fc.mets + fc.red_sox = total_fans :=
by sorry

end yankees_to_mets_ratio_l1864_186451


namespace team_omega_score_l1864_186493

/-- Given a basketball match between Team Alpha and Team Omega where:
  - The total points scored by both teams is 60
  - Team Alpha won by a margin of 12 points
  This theorem proves that Team Omega scored 24 points. -/
theorem team_omega_score (total_points : ℕ) (margin : ℕ) 
  (h1 : total_points = 60) 
  (h2 : margin = 12) : 
  (total_points - margin) / 2 = 24 := by
  sorry

#check team_omega_score

end team_omega_score_l1864_186493


namespace cubic_factorization_l1864_186447

theorem cubic_factorization (x : ℝ) : x^3 + 3*x^2 - 4 = (x-1)*(x+2)^2 := by
  sorry

end cubic_factorization_l1864_186447


namespace triangle_exists_but_not_isosceles_l1864_186444

def stick_lengths : List ℝ := [1, 1.9, 1.9^2, 1.9^3, 1.9^4, 1.9^5, 1.9^6, 1.9^7, 1.9^8, 1.9^9]

def is_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

def is_isosceles_triangle (a b c : ℝ) : Prop :=
  (a = b ∧ a + b > c) ∨ (b = c ∧ b + c > a) ∨ (c = a ∧ c + a > b)

theorem triangle_exists_but_not_isosceles :
  (∃ (a b c : ℝ), a ∈ stick_lengths ∧ b ∈ stick_lengths ∧ c ∈ stick_lengths ∧ is_triangle a b c) ∧
  (¬ ∃ (a b c : ℝ), a ∈ stick_lengths ∧ b ∈ stick_lengths ∧ c ∈ stick_lengths ∧ is_isosceles_triangle a b c) :=
by sorry

end triangle_exists_but_not_isosceles_l1864_186444


namespace no_solution_implies_m_leq_two_l1864_186485

theorem no_solution_implies_m_leq_two (m : ℝ) : 
  (∀ x : ℝ, ¬(x - 1 > 1 ∧ x < m)) → m ≤ 2 := by
  sorry

end no_solution_implies_m_leq_two_l1864_186485


namespace game_packing_l1864_186409

theorem game_packing (initial_games : Nat) (sold_games : Nat) (games_per_box : Nat) :
  initial_games = 35 →
  sold_games = 19 →
  games_per_box = 8 →
  (initial_games - sold_games) / games_per_box = 2 := by
  sorry

end game_packing_l1864_186409
