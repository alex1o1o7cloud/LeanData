import Mathlib

namespace infinitely_many_common_divisors_l934_93470

theorem infinitely_many_common_divisors :
  Set.Infinite {n : ℕ | ∃ d : ℕ, d > 1 ∧ d ∣ (2*n - 3) ∧ d ∣ (3*n - 2)} :=
by
  sorry

end infinitely_many_common_divisors_l934_93470


namespace power_calculation_l934_93443

theorem power_calculation (m n : ℕ) (h1 : 2^m = 3) (h2 : 4^n = 8) :
  2^(3*m - 2*n + 3) = 27 := by
  sorry

end power_calculation_l934_93443


namespace multiplication_addition_equality_l934_93476

theorem multiplication_addition_equality : 24 * 44 + 56 * 24 = 2400 := by
  sorry

end multiplication_addition_equality_l934_93476


namespace sqrt_product_quotient_l934_93415

theorem sqrt_product_quotient : (Real.sqrt 3 * Real.sqrt 15) / Real.sqrt 5 = 3 := by
  sorry

end sqrt_product_quotient_l934_93415


namespace simplified_multiplication_l934_93446

def factor1 : Nat := 20213
def factor2 : Nat := 732575

theorem simplified_multiplication (f1 f2 : Nat) (h1 : f1 = factor1) (h2 : f2 = factor2) :
  ∃ (partial_products : List Nat),
    f1 * f2 = partial_products.sum ∧
    partial_products.length < 5 :=
sorry

end simplified_multiplication_l934_93446


namespace box_depth_proof_l934_93417

/-- Represents the dimensions of a box -/
structure BoxDimensions where
  length : ℕ
  width : ℕ
  depth : ℕ

/-- Represents a cube -/
structure Cube where
  edgeLength : ℕ

/-- Theorem: Given a box with specific dimensions filled with cubes, prove its depth -/
theorem box_depth_proof (box : BoxDimensions) (cube : Cube) (numCubes : ℕ) :
  box.length = 36 →
  box.width = 45 →
  numCubes = 40 →
  (box.length * box.width * box.depth = numCubes * cube.edgeLength ^ 3) →
  (box.length % cube.edgeLength = 0) →
  (box.width % cube.edgeLength = 0) →
  (box.depth % cube.edgeLength = 0) →
  box.depth = 18 := by
  sorry


end box_depth_proof_l934_93417


namespace function_max_at_zero_implies_a_geq_three_l934_93407

/-- Given a function f(x) = x + a / (x + 1) defined on [0, 2] with maximum at x = 0, prove a ≥ 3 -/
theorem function_max_at_zero_implies_a_geq_three (a : ℝ) :
  (∀ x : ℝ, 0 ≤ x → x ≤ 2 → x + a / (x + 1) ≤ a) →
  a ≥ 3 := by
  sorry

end function_max_at_zero_implies_a_geq_three_l934_93407


namespace local_min_implies_a_equals_one_l934_93406

/-- Given a function f(x) = ax^3 - 2x^2 + a^2x, where a is a real number,
    if f has a local minimum at x = 1, then a = 1. -/
theorem local_min_implies_a_equals_one (a : ℝ) :
  let f := λ x : ℝ => a * x^3 - 2 * x^2 + a^2 * x
  (∃ δ > 0, ∀ x, |x - 1| < δ → f x ≥ f 1) →
  a = 1 := by sorry

end local_min_implies_a_equals_one_l934_93406


namespace income_mean_difference_l934_93418

/-- The number of families --/
def num_families : ℕ := 1200

/-- The correct highest income --/
def correct_highest_income : ℕ := 150000

/-- The incorrect highest income --/
def incorrect_highest_income : ℕ := 1500000

/-- The sum of all incomes except the highest --/
def S : ℕ := sorry

/-- The difference between the mean of incorrect data and actual data --/
def mean_difference : ℚ :=
  (S + incorrect_highest_income : ℚ) / num_families -
  (S + correct_highest_income : ℚ) / num_families

theorem income_mean_difference :
  mean_difference = 1125 := by sorry

end income_mean_difference_l934_93418


namespace value_of_m_l934_93423

theorem value_of_m (m : ℝ) (a₀ a₁ a₂ a₃ a₄ a₅ a₆ : ℝ) 
  (expansion : ∀ x, (1 + m * x)^6 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6)
  (alternating_sum : a₁ - a₂ + a₃ - a₄ + a₅ - a₆ = -63) :
  m = 3 ∨ m = -1 := by
sorry

end value_of_m_l934_93423


namespace simplify_expression_evaluate_neg_one_evaluate_zero_undefined_for_one_undefined_for_two_l934_93490

theorem simplify_expression (a : ℝ) (h1 : a ≠ 1) (h2 : a ≠ 2) :
  (1 + 3 / (a - 1)) / ((a^2 - 4) / (a - 1)) = 1 / (a - 2) := by
  sorry

-- Evaluation for a = -1
theorem evaluate_neg_one :
  (1 + 3 / (-1 - 1)) / ((-1^2 - 4) / (-1 - 1)) = -1/3 := by
  sorry

-- Evaluation for a = 0
theorem evaluate_zero :
  (1 + 3 / (0 - 1)) / ((0^2 - 4) / (0 - 1)) = -1/2 := by
  sorry

-- Undefined for a = 1
theorem undefined_for_one (h : (1 : ℝ) ≠ 2) :
  ¬∃x, (1 + 3 / (1 - 1)) / ((1^2 - 4) / (1 - 1)) = x := by
  sorry

-- Undefined for a = 2
theorem undefined_for_two (h : (2 : ℝ) ≠ 1) :
  ¬∃x, (1 + 3 / (2 - 1)) / ((2^2 - 4) / (2 - 1)) = x := by
  sorry

end simplify_expression_evaluate_neg_one_evaluate_zero_undefined_for_one_undefined_for_two_l934_93490


namespace net_effect_on_revenue_l934_93465

theorem net_effect_on_revenue 
  (original_price original_sales : ℝ) 
  (price_reduction : ℝ) 
  (sales_increase : ℝ) 
  (h1 : price_reduction = 0.2) 
  (h2 : sales_increase = 0.8) : 
  let new_price := original_price * (1 - price_reduction)
  let new_sales := original_sales * (1 + sales_increase)
  let original_revenue := original_price * original_sales
  let new_revenue := new_price * new_sales
  (new_revenue - original_revenue) / original_revenue = 0.44 := by
sorry

end net_effect_on_revenue_l934_93465


namespace f_monotone_increasing_implies_a_bound_l934_93482

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x - 2 * Real.sin x * Real.cos x + a * Real.cos x

def monotone_increasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x → x < y → y ≤ b → f x ≤ f y

theorem f_monotone_increasing_implies_a_bound :
  ∀ a : ℝ, monotone_increasing (f a) (π/4) (3*π/4) → a ≤ Real.sqrt 2 := by sorry

end f_monotone_increasing_implies_a_bound_l934_93482


namespace function_identity_l934_93411

theorem function_identity (f : ℕ+ → ℤ) 
  (h1 : f 2 = 2)
  (h2 : ∀ m n : ℕ+, f (m * n) = f m * f n)
  (h3 : ∀ m n : ℕ+, m > n → f m > f n) :
  ∀ n : ℕ+, f n = n := by
sorry

end function_identity_l934_93411


namespace problem_statement_l934_93437

theorem problem_statement :
  ∀ d : ℕ, (5 ^ 5) * (9 ^ 3) = 3 * d ∧ d = 15 ^ 5 → d = 759375 := by
  sorry

end problem_statement_l934_93437


namespace petya_more_likely_to_win_petya_wins_in_game_l934_93499

/-- Represents a game between Petya and Vasya with two boxes of candies. -/
structure CandyGame where
  total_candies : ℕ
  prob_two_caramels : ℝ

/-- Defines the game setup with the given conditions. -/
def game : CandyGame :=
  { total_candies := 25,
    prob_two_caramels := 0.54 }

/-- Calculates the probability of Vasya winning (getting two chocolate candies). -/
def prob_vasya_wins (g : CandyGame) : ℝ :=
  1 - g.prob_two_caramels

/-- Theorem stating that Petya has a higher chance of winning than Vasya. -/
theorem petya_more_likely_to_win (g : CandyGame) :
  prob_vasya_wins g < 1 - prob_vasya_wins g :=
by sorry

/-- Corollary proving that Petya has a higher chance of winning in the specific game setup. -/
theorem petya_wins_in_game : prob_vasya_wins game < 1 - prob_vasya_wins game :=
by sorry

end petya_more_likely_to_win_petya_wins_in_game_l934_93499


namespace not_equal_necessary_not_sufficient_l934_93449

-- Define the relationship between α and β
def not_equal (α β : Real) : Prop := α ≠ β

-- Define the relationship between sin α and sin β
def sin_not_equal (α β : Real) : Prop := Real.sin α ≠ Real.sin β

-- Theorem stating that not_equal is a necessary but not sufficient condition for sin_not_equal
theorem not_equal_necessary_not_sufficient :
  (∀ α β : Real, sin_not_equal α β → not_equal α β) ∧
  ¬(∀ α β : Real, not_equal α β → sin_not_equal α β) :=
sorry

end not_equal_necessary_not_sufficient_l934_93449


namespace percent_equality_l934_93442

theorem percent_equality (x : ℝ) : (75 / 100 * 600 = 50 / 100 * x) → x = 900 := by
  sorry

end percent_equality_l934_93442


namespace regular_polygon_150_degrees_has_12_sides_l934_93402

/-- A regular polygon with interior angles of 150 degrees has 12 sides -/
theorem regular_polygon_150_degrees_has_12_sides :
  ∀ n : ℕ, 
    n > 2 →
    (∀ angle : ℝ, angle = 150 → n * angle = (n - 2) * 180) →
    n = 12 := by
  sorry

end regular_polygon_150_degrees_has_12_sides_l934_93402


namespace trig_equation_solution_l934_93456

theorem trig_equation_solution (a b c α β : ℝ) 
  (h1 : a * Real.cos α + b * Real.sin α = c)
  (h2 : a * Real.cos β + b * Real.sin β = c)
  (h3 : a^2 + b^2 ≠ 0)
  (h4 : ∀ k : ℤ, α ≠ β + 2 * k * Real.pi) :
  (Real.cos ((α - β) / 2))^2 = c^2 / (a^2 + b^2) := by
  sorry

end trig_equation_solution_l934_93456


namespace batsman_average_theorem_l934_93498

/-- Calculates the average runs for a batsman over multiple sets of matches -/
def average_runs (runs_per_set : List ℕ) (matches_per_set : List ℕ) : ℚ :=
  (runs_per_set.zip matches_per_set).map (fun (r, m) => r * m)
    |> List.sum
    |> (fun total_runs => total_runs / matches_per_set.sum)

theorem batsman_average_theorem (first_10_avg : ℕ) (next_10_avg : ℕ) :
  first_10_avg = 40 →
  next_10_avg = 30 →
  average_runs [first_10_avg, next_10_avg] [10, 10] = 35 := by
  sorry

#eval average_runs [40, 30] [10, 10]

end batsman_average_theorem_l934_93498


namespace certain_number_proof_l934_93457

theorem certain_number_proof (x : ℝ) : x / 14.5 = 177 → x = 2566.5 := by
  sorry

end certain_number_proof_l934_93457


namespace power_sum_difference_equals_ten_l934_93424

theorem power_sum_difference_equals_ten : 2^5 + 5^2 / 5^1 - 3^3 = 10 := by
  sorry

end power_sum_difference_equals_ten_l934_93424


namespace non_similar_500_pointed_stars_l934_93460

/-- A regular n-pointed star is the union of n line segments. -/
def RegularStar (n : ℕ) (m : ℕ) : Prop :=
  (n > 0) ∧ (m > 0) ∧ (m < n) ∧ (Nat.gcd m n = 1)

/-- Two stars are similar if they have the same number of points and
    their m values are either equal or complementary modulo n. -/
def SimilarStars (n : ℕ) (m1 m2 : ℕ) : Prop :=
  RegularStar n m1 ∧ RegularStar n m2 ∧ (m1 = m2 ∨ m1 + m2 = n)

/-- The number of non-similar regular n-pointed stars -/
def NonSimilarStarCount (n : ℕ) : ℕ :=
  (Nat.totient n - 2) / 2 + 1

theorem non_similar_500_pointed_stars :
  NonSimilarStarCount 500 = 99 := by
  sorry

#eval NonSimilarStarCount 500  -- This should output 99

end non_similar_500_pointed_stars_l934_93460


namespace percentage_problem_l934_93493

theorem percentage_problem (x : ℝ) : 0.15 * 0.30 * 0.50 * x = 99 → x = 4400 := by
  sorry

end percentage_problem_l934_93493


namespace twenty_paise_coins_count_l934_93468

theorem twenty_paise_coins_count 
  (total_coins : ℕ) 
  (total_value : ℚ) 
  (h_total_coins : total_coins = 334)
  (h_total_value : total_value = 71)
  : ∃ (coins_20p coins_25p : ℕ), 
    coins_20p + coins_25p = total_coins ∧ 
    (1/5 : ℚ) * coins_20p + (1/4 : ℚ) * coins_25p = total_value ∧
    coins_20p = 250 := by
  sorry

end twenty_paise_coins_count_l934_93468


namespace isabellas_final_hair_length_l934_93474

/-- The final hair length given an initial length and growth --/
def finalHairLength (initialLength growth : ℕ) : ℕ :=
  initialLength + growth

/-- Theorem: Isabella's final hair length is 24 inches --/
theorem isabellas_final_hair_length :
  finalHairLength 18 6 = 24 := by
  sorry

end isabellas_final_hair_length_l934_93474


namespace robin_hair_growth_l934_93434

/-- Calculates hair growth given initial length, cut length, and final length -/
def hair_growth (initial_length cut_length final_length : ℕ) : ℕ :=
  final_length - (initial_length - cut_length)

/-- Theorem: Given the problem conditions, hair growth is 12 inches -/
theorem robin_hair_growth :
  hair_growth 16 11 17 = 12 := by sorry

end robin_hair_growth_l934_93434


namespace fifteenth_student_age_l934_93494

theorem fifteenth_student_age
  (total_students : ℕ)
  (total_average_age : ℝ)
  (group1_students : ℕ)
  (group1_average_age : ℝ)
  (group2_students : ℕ)
  (group2_average_age : ℝ)
  (h1 : total_students = 15)
  (h2 : total_average_age = 15)
  (h3 : group1_students = 4)
  (h4 : group1_average_age = 14)
  (h5 : group2_students = 10)
  (h6 : group2_average_age = 16)
  : ℝ := by
  sorry

#check fifteenth_student_age

end fifteenth_student_age_l934_93494


namespace karen_rolls_count_l934_93486

/-- The number of egg rolls Omar rolled -/
def omar_rolls : ℕ := 219

/-- The total number of egg rolls Omar and Karen rolled -/
def total_rolls : ℕ := 448

/-- The number of egg rolls Karen rolled -/
def karen_rolls : ℕ := total_rolls - omar_rolls

theorem karen_rolls_count : karen_rolls = 229 := by
  sorry

end karen_rolls_count_l934_93486


namespace perfect_square_condition_l934_93453

theorem perfect_square_condition (a b k : ℝ) :
  (∃ (n : ℝ), a^2 + k*a*b + 9*b^2 = n^2) → (k = 6 ∨ k = -6) := by
  sorry

end perfect_square_condition_l934_93453


namespace intersection_of_P_and_Q_l934_93466

-- Define set P
def P : Set ℝ := {x | x^2 - x - 2 ≥ 0}

-- Define set Q
def Q : Set ℝ := {y | ∃ x ∈ P, y = (1/2) * x^2 - 1}

-- Theorem statement
theorem intersection_of_P_and_Q :
  P ∩ Q = {m : ℝ | m ≥ 2} := by sorry

end intersection_of_P_and_Q_l934_93466


namespace monthly_fee_calculation_l934_93462

def cost_per_minute : ℚ := 25 / 100

def total_bill : ℚ := 1202 / 100

def minutes_used : ℚ := 2808 / 100

theorem monthly_fee_calculation :
  ∃ (monthly_fee : ℚ),
    monthly_fee + cost_per_minute * minutes_used = total_bill ∧
    monthly_fee = 5 := by
  sorry

end monthly_fee_calculation_l934_93462


namespace trigonometric_expression_equals_three_l934_93489

theorem trigonometric_expression_equals_three (α : ℝ) 
  (h : Real.tan (3 * Real.pi + α) = 3) : 
  (Real.sin (α - 3 * Real.pi) + Real.cos (Real.pi - α) + 
   Real.sin (Real.pi / 2 - α) - 2 * Real.cos (Real.pi / 2 + α)) / 
  (-Real.sin (-α) + Real.cos (Real.pi + α)) = 3 := by
  sorry

end trigonometric_expression_equals_three_l934_93489


namespace inequality_equivalence_l934_93421

theorem inequality_equivalence (x : ℝ) : 
  (x ∈ Set.Icc (-1 : ℝ) 1) ↔ 
  (∀ (n : ℕ) (a : ℕ → ℝ), n ≥ 2 → (∀ i, i ∈ Finset.range n → a i ≥ 1) → 
    ((Finset.range n).prod (λ i => (a i + x) / 2) ≤ 
     ((Finset.range n).prod (λ i => a i) + x) / 2)) :=
by sorry

end inequality_equivalence_l934_93421


namespace arithmetic_calculation_l934_93405

theorem arithmetic_calculation : 8 / 4 - 3 - 9 + 3 * 7 - 2^2 = 7 := by sorry

end arithmetic_calculation_l934_93405


namespace triangle_problem_l934_93471

theorem triangle_problem (a b c : ℝ) (A B C : ℝ) :
  a = 3 →
  b = 2 →
  Real.cos A = 1/2 →
  -- (I)
  Real.sin B = Real.sqrt 3 / 3 ∧
  -- (II)
  c = 1 + Real.sqrt 6 :=
by
  sorry

end triangle_problem_l934_93471


namespace horner_evaluation_approx_l934_93496

/-- Horner's method for polynomial evaluation -/
def horner (coeffs : List Float) (x : Float) : Float :=
  coeffs.foldl (fun acc a => acc * x + a) 0

/-- The polynomial function f(x) = 1 + x + 0.5x^2 + 0.16667x^3 + 0.04167x^4 + 0.00833x^5 -/
def f (x : Float) : Float :=
  horner [1, 1, 0.5, 0.16667, 0.04167, 0.00833] x

theorem horner_evaluation_approx :
  (f (-0.2) - 0.81873).abs < 1e-5 := by
  sorry

end horner_evaluation_approx_l934_93496


namespace arithmetic_sequence_seventh_term_l934_93420

/-- An arithmetic sequence is a sequence where the difference between any two consecutive terms is constant. -/
structure ArithmeticSequence where
  /-- The first term of the sequence -/
  a : ℝ
  /-- The common difference between consecutive terms -/
  d : ℝ

/-- The nth term of an arithmetic sequence -/
def ArithmeticSequence.nthTerm (seq : ArithmeticSequence) (n : ℕ) : ℝ :=
  seq.a + (n - 1 : ℝ) * seq.d

theorem arithmetic_sequence_seventh_term
  (seq : ArithmeticSequence)
  (h3 : seq.nthTerm 3 = 17)
  (h5 : seq.nthTerm 5 = 39) :
  seq.nthTerm 7 = 61 := by
  sorry


end arithmetic_sequence_seventh_term_l934_93420


namespace cookie_count_equivalence_l934_93426

/-- Represents the shape of a cookie -/
inductive CookieShape
  | Circle
  | Rectangle
  | Parallelogram
  | Triangle
  | Square

/-- Represents a friend who bakes cookies -/
structure Friend where
  name : String
  shape : CookieShape

/-- Represents the dimensions of a cookie -/
structure CookieDimensions where
  base : ℝ
  height : ℝ

theorem cookie_count_equivalence 
  (friends : List Friend)
  (carlos_dims : CookieDimensions)
  (lisa_side : ℝ)
  (carlos_count : ℕ)
  (h1 : friends.length = 5)
  (h2 : ∃ f ∈ friends, f.name = "Carlos" ∧ f.shape = CookieShape.Triangle)
  (h3 : ∃ f ∈ friends, f.name = "Lisa" ∧ f.shape = CookieShape.Square)
  (h4 : carlos_dims.base = 4)
  (h5 : carlos_dims.height = 5)
  (h6 : carlos_count = 20)
  (h7 : lisa_side = 5)
  : (200 : ℝ) / (lisa_side ^ 2) = 8 := by
  sorry

end cookie_count_equivalence_l934_93426


namespace sports_classes_theorem_l934_93477

/-- The number of students in different sports classes -/
def sports_classes (x : ℕ) : ℕ × ℕ × ℕ :=
  let basketball := x
  let soccer := 2 * x - 2
  let volleyball := (soccer / 2) + 2
  (basketball, soccer, volleyball)

theorem sports_classes_theorem (x : ℕ) (h : 2 * x - 6 = 34) :
  sports_classes x = (20, 34, 19) := by
  sorry

end sports_classes_theorem_l934_93477


namespace difference_d_minus_b_l934_93480

theorem difference_d_minus_b (a b c d : ℕ+) 
  (h1 : a^5 = b^4) 
  (h2 : c^3 = d^2) 
  (h3 : c - a = 19) : 
  d - b = 757 := by sorry

end difference_d_minus_b_l934_93480


namespace sum_f_negative_l934_93422

/-- A monotonically decreasing odd function -/
def MonoDecreasingOddFunction (f : ℝ → ℝ) : Prop :=
  (∀ x y, x < y → f x > f y) ∧ (∀ x, f (-x) = -f x)

/-- Theorem: Sum of function values is negative under given conditions -/
theorem sum_f_negative
  (f : ℝ → ℝ)
  (hf : MonoDecreasingOddFunction f)
  (x₁ x₂ x₃ : ℝ)
  (h₁₂ : x₁ + x₂ > 0)
  (h₂₃ : x₂ + x₃ > 0)
  (h₃₁ : x₃ + x₁ > 0) :
  f x₁ + f x₂ + f x₃ < 0 :=
sorry

end sum_f_negative_l934_93422


namespace tangent_point_determines_b_l934_93483

-- Define the curve and line
def curve (x a b : ℝ) : ℝ := x^3 + a*x + b
def line (x k : ℝ) : ℝ := k*x + 1

-- Define the tangent condition
def is_tangent (a b k : ℝ) : Prop :=
  ∃ x, curve x a b = line x k ∧ 
       (deriv (fun x => curve x a b)) x = k

theorem tangent_point_determines_b :
  ∀ a b k : ℝ, 
    is_tangent a b k →  -- The line is tangent to the curve
    curve 1 a b = 3 →   -- The point of tangency is (1, 3)
    b = 3 :=
by sorry

end tangent_point_determines_b_l934_93483


namespace fixed_point_theorem_l934_93413

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Checks if a point lies on a line given by the equation (3+k)x + (1-2k)y + 1 + 5k = 0 -/
def lies_on_line (p : Point) (k : ℝ) : Prop :=
  (3 + k) * p.x + (1 - 2*k) * p.y + 1 + 5*k = 0

/-- The theorem stating that (-1, 2) is the unique fixed point for all lines -/
theorem fixed_point_theorem :
  ∃! p : Point, ∀ k : ℝ, lies_on_line p k :=
sorry

end fixed_point_theorem_l934_93413


namespace fruit_stand_problem_l934_93435

def fruit_problem (apple_price orange_price : ℚ) 
                  (total_fruits : ℕ) 
                  (initial_avg_price desired_avg_price : ℚ) : Prop :=
  let oranges_to_remove := 10
  let remaining_fruits := total_fruits - oranges_to_remove
  ∃ (apples oranges : ℕ),
    apples + oranges = total_fruits ∧
    (apple_price * apples + orange_price * oranges) / total_fruits = initial_avg_price ∧
    (apple_price * apples + orange_price * (oranges - oranges_to_remove)) / remaining_fruits = desired_avg_price

theorem fruit_stand_problem :
  fruit_problem (40/100) (60/100) 20 (56/100) (52/100) :=
by
  sorry

end fruit_stand_problem_l934_93435


namespace wendy_furniture_time_l934_93436

/-- The time Wendy spent putting together all the furniture -/
def total_time (num_chairs num_tables time_per_piece : ℕ) : ℕ :=
  (num_chairs + num_tables) * time_per_piece

/-- Proof that Wendy spent 48 minutes putting together all the furniture -/
theorem wendy_furniture_time :
  total_time 4 4 6 = 48 := by
  sorry

end wendy_furniture_time_l934_93436


namespace kims_sweater_difference_l934_93419

/-- The number of sweaters Kim knit on each day of the week --/
structure WeeklySweaters where
  monday : ℕ
  tuesday : ℕ
  wednesday : ℕ
  thursday : ℕ
  friday : ℕ

/-- The conditions of Kim's sweater knitting for the week --/
def kimsSweaterWeek (s : WeeklySweaters) : Prop :=
  s.monday = 8 ∧
  s.tuesday > s.monday ∧
  s.wednesday = s.tuesday - 4 ∧
  s.thursday = s.tuesday - 4 ∧
  s.friday = s.monday / 2 ∧
  s.monday + s.tuesday + s.wednesday + s.thursday + s.friday = 34

theorem kims_sweater_difference (s : WeeklySweaters) 
  (h : kimsSweaterWeek s) : s.tuesday - s.monday = 2 := by
  sorry

end kims_sweater_difference_l934_93419


namespace sat_markings_count_l934_93410

/-- The number of ways to mark a single question on the SAT answer sheet -/
def markings_per_question : ℕ := 32

/-- The number of questions to be marked -/
def num_questions : ℕ := 10

/-- Function to calculate the number of valid sequences of length n with no consecutive 1s -/
def f : ℕ → ℕ
| 0 => 1
| 1 => 2
| (n + 2) => f (n + 1) + f n

/-- The number of letters in the SAT answer sheet -/
def num_letters : ℕ := 5

/-- Theorem stating the total number of ways to mark the SAT answer sheet -/
theorem sat_markings_count :
  (f num_questions) ^ num_letters = 2^20 * 3^10 := by sorry

end sat_markings_count_l934_93410


namespace two_card_selections_65_l934_93425

/-- The number of ways to select two different cards from a deck of 65 cards, where the order of selection matters. -/
def two_card_selections (total_cards : ℕ) : ℕ :=
  total_cards * (total_cards - 1)

/-- Theorem stating that selecting two different cards from a deck of 65 cards, where the order matters, can be done in 4160 ways. -/
theorem two_card_selections_65 :
  two_card_selections 65 = 4160 := by
  sorry

end two_card_selections_65_l934_93425


namespace ultramarathon_training_l934_93467

theorem ultramarathon_training (initial_time initial_speed : ℝ)
  (time_increase_percent speed_increase : ℝ)
  (h1 : initial_time = 8)
  (h2 : initial_speed = 8)
  (h3 : time_increase_percent = 75)
  (h4 : speed_increase = 4) :
  let new_time := initial_time * (1 + time_increase_percent / 100)
  let new_speed := initial_speed + speed_increase
  new_time * new_speed = 168 := by
  sorry

end ultramarathon_training_l934_93467


namespace unopened_box_cards_l934_93491

theorem unopened_box_cards (initial_cards given_away_cards final_total_cards : ℕ) :
  initial_cards = 26 →
  given_away_cards = 18 →
  final_total_cards = 48 →
  final_total_cards = (initial_cards - given_away_cards) + (final_total_cards - (initial_cards - given_away_cards)) :=
by
  sorry

end unopened_box_cards_l934_93491


namespace committee_count_l934_93429

theorem committee_count (n m : ℕ) (hn : n = 8) (hm : m = 4) :
  (n.choose 1) * ((n - 1).choose 1) * ((n - 2).choose (m - 2)) = 840 := by
  sorry

end committee_count_l934_93429


namespace quadratic_minimum_l934_93472

-- Define the quadratic function
def f (x : ℝ) : ℝ := 3 * x^2 + 6 * x + 9

-- State the theorem
theorem quadratic_minimum :
  ∃ (m : ℝ), (∀ x, f x ≥ m) ∧ (∃ x₀, f x₀ = m) ∧ m = 6 := by
  sorry

end quadratic_minimum_l934_93472


namespace min_values_theorem_l934_93438

/-- Given positive real numbers a and b satisfying 4a + b = ab, 
    prove the following statements about their minimum values. -/
theorem min_values_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 4*a + b = a*b) :
  (∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ 4*a₀ + b₀ = a₀*b₀ ∧ ∀ a' b', a' > 0 → b' > 0 → 4*a' + b' = a'*b' → a₀*b₀ ≤ a'*b') ∧
  (∃ (a₁ b₁ : ℝ), a₁ > 0 ∧ b₁ > 0 ∧ 4*a₁ + b₁ = a₁*b₁ ∧ ∀ a' b', a' > 0 → b' > 0 → 4*a' + b' = a'*b' → a₁ + b₁ ≤ a' + b') ∧
  (∃ (a₂ b₂ : ℝ), a₂ > 0 ∧ b₂ > 0 ∧ 4*a₂ + b₂ = a₂*b₂ ∧ ∀ a' b', a' > 0 → b' > 0 → 4*a' + b' = a'*b' → 1/a₂^2 + 4/b₂^2 ≤ 1/a'^2 + 4/b'^2) ∧
  (∀ a' b', a' > 0 → b' > 0 → 4*a' + b' = a'*b' → a'*b' ≥ 16) ∧
  (∀ a' b', a' > 0 → b' > 0 → 4*a' + b' = a'*b' → a' + b' ≥ 9) ∧
  (∀ a' b', a' > 0 → b' > 0 → 4*a' + b' = a'*b' → 1/a'^2 + 4/b'^2 ≥ 1/5) :=
by sorry

end min_values_theorem_l934_93438


namespace functional_equation_solution_l934_93433

/-- A function satisfying the given functional equation. -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f ((x + y) / 2) = (f x + f y) / 2

/-- The main theorem stating that any function satisfying the functional equation
    is of the form f(x) = ax + b for some constants a and b. -/
theorem functional_equation_solution (f : ℝ → ℝ) (h : FunctionalEquation f) :
  ∃ a b : ℝ, ∀ x : ℝ, f x = a * x + b := by
  sorry

end functional_equation_solution_l934_93433


namespace intersection_and_slope_l934_93427

theorem intersection_and_slope (k : ℝ) :
  (∃ y : ℝ, -3 * 3 + y = k ∧ 3 + y = 8) →
  k = -4 ∧ 
  (∀ x y : ℝ, x + y = 8 → y = -x + 8) :=
by sorry

end intersection_and_slope_l934_93427


namespace qqLive_higher_score_l934_93458

structure SoftwareRating where
  name : String
  studentRatings : List Nat
  studentAverage : Float
  teacherAverage : Float

def comprehensiveScore (rating : SoftwareRating) : Float :=
  rating.studentAverage * 0.4 + rating.teacherAverage * 0.6

def dingtalk : SoftwareRating := {
  name := "DingTalk",
  studentRatings := [1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5],
  studentAverage := 3.4,
  teacherAverage := 3.9
}

def qqLive : SoftwareRating := {
  name := "QQ Live",
  studentRatings := [1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5],
  studentAverage := 3.35,
  teacherAverage := 4.0
}

theorem qqLive_higher_score : comprehensiveScore qqLive > comprehensiveScore dingtalk := by
  sorry

end qqLive_higher_score_l934_93458


namespace abc_inequality_l934_93440

theorem abc_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a * b * c = 1) :
  a / (a * b + 1) + b / (b * c + 1) + c / (c * a + 1) ≥ 3 / 2 := by
  sorry

end abc_inequality_l934_93440


namespace parallel_line_through_point_l934_93403

/-- Given a point P and a line L, this theorem proves that a specific equation
    represents a line passing through P and parallel to L. -/
theorem parallel_line_through_point (x y : ℝ) :
  let P : ℝ × ℝ := (2, Real.sqrt 3)
  let L : ℝ → ℝ → ℝ := fun x y => Real.sqrt 3 * x - y + 2
  let parallel_line : ℝ → ℝ → ℝ := fun x y => Real.sqrt 3 * x - y - Real.sqrt 3
  (parallel_line P.1 P.2 = 0) ∧
  (∃ (k : ℝ), k ≠ 0 ∧ ∀ x y, parallel_line x y = k * L x y) :=
by sorry


end parallel_line_through_point_l934_93403


namespace contrapositive_equivalence_l934_93408

theorem contrapositive_equivalence (a b : ℝ) :
  (¬(a = 0 ∧ b = 0) → a^2 + b^2 ≠ 0) ↔ (a^2 + b^2 = 0 → a = 0 ∧ b = 0) :=
by sorry

end contrapositive_equivalence_l934_93408


namespace cone_height_for_right_angle_vertex_l934_93450

/-- Represents a cone with given volume and vertex angle -/
structure Cone where
  volume : ℝ
  vertexAngle : ℝ

/-- The height of a cone given its volume and vertex angle -/
def coneHeight (c : Cone) : ℝ :=
  sorry

theorem cone_height_for_right_angle_vertex (c : Cone) 
  (h_volume : c.volume = 20000 * Real.pi)
  (h_angle : c.vertexAngle = Real.pi / 2) :
  ∃ (r : ℝ), coneHeight c = r * Real.sqrt 2 ∧ 
  r^3 * Real.sqrt 2 = 60000 :=
sorry

end cone_height_for_right_angle_vertex_l934_93450


namespace purely_imaginary_complex_number_l934_93492

theorem purely_imaginary_complex_number (a : ℝ) : 
  (Complex.I * (a - 1) : ℂ).re = a^2 - 1 ∧ (a - 1 ≠ 0) → a = -1 :=
by sorry

end purely_imaginary_complex_number_l934_93492


namespace quadrupled_base_exponent_l934_93497

theorem quadrupled_base_exponent (a b x : ℝ) (ha : a > 0) (hb : b > 0) (hx : x > 0) :
  (4*a)^(4*b) = a^b * x^(2*b) → x = 16 * a^(3/2) := by
  sorry

end quadrupled_base_exponent_l934_93497


namespace circle_center_satisfies_conditions_l934_93488

/-- The center of a circle satisfying given conditions -/
theorem circle_center_satisfies_conditions :
  let center : ℝ × ℝ := (-18, -11)
  let line1 : ℝ → ℝ → ℝ := λ x y => 3 * x - 4 * y - 20
  let line2 : ℝ → ℝ → ℝ := λ x y => 3 * x - 4 * y + 40
  let midline : ℝ → ℝ → ℝ := λ x y => 3 * x - 4 * y + 10
  let line3 : ℝ → ℝ → ℝ := λ x y => x - 3 * y - 15
  (midline center.1 center.2 = 0) ∧ (line3 center.1 center.2 = 0) :=
by sorry

end circle_center_satisfies_conditions_l934_93488


namespace sum_of_multiples_l934_93454

def largest_three_digit_multiple_of_4 : ℕ := 996

def smallest_four_digit_multiple_of_3 : ℕ := 1002

theorem sum_of_multiples : 
  largest_three_digit_multiple_of_4 + smallest_four_digit_multiple_of_3 = 1998 := by
  sorry

end sum_of_multiples_l934_93454


namespace expression_simplification_l934_93459

theorem expression_simplification (a : ℝ) (h : a ≠ 0) :
  (a * (a + 1) + (a - 1)^2 - 1) / (-a) = -2 * a + 1 := by
  sorry

end expression_simplification_l934_93459


namespace parabola_vertex_relationship_l934_93479

/-- Given a parabola y = x^2 - 2mx + 2m^2 - 3m + 1, prove that the functional relationship
    between the vertical coordinate y and the horizontal coordinate x of its vertex
    is y = x^2 - 3x + 1, regardless of the value of m. -/
theorem parabola_vertex_relationship (m x y : ℝ) :
  y = x^2 - 2*m*x + 2*m^2 - 3*m + 1 →
  (x = m ∧ y = m^2 - 3*m + 1) →
  y = x^2 - 3*x + 1 :=
by sorry

end parabola_vertex_relationship_l934_93479


namespace diamond_equation_solution_l934_93447

/-- Definition of the ⋄ operation -/
noncomputable def diamond (a b : ℝ) : ℝ := a + Real.sqrt (b + Real.sqrt (b + Real.sqrt b))

/-- Theorem stating that if 3 ⋄ y = 12, then y = 72 -/
theorem diamond_equation_solution :
  ∃ y : ℝ, diamond 3 y = 12 ∧ y = 72 := by
  sorry

end diamond_equation_solution_l934_93447


namespace probability_of_guessing_two_questions_correctly_l934_93464

theorem probability_of_guessing_two_questions_correctly :
  let num_questions : ℕ := 2
  let options_per_question : ℕ := 4
  let prob_one_correct : ℚ := 1 / options_per_question
  prob_one_correct ^ num_questions = (1 : ℚ) / 16 := by sorry

end probability_of_guessing_two_questions_correctly_l934_93464


namespace at_least_two_positive_roots_l934_93439

def f (x : ℝ) : ℝ := x^11 + 8*x^10 + 15*x^9 - 1729*x^8 + 1379*x^7 - 172*x^6

theorem at_least_two_positive_roots :
  ∃ (a b : ℝ), 0 < a ∧ 0 < b ∧ a ≠ b ∧ f a = 0 ∧ f b = 0 :=
sorry

end at_least_two_positive_roots_l934_93439


namespace cos_150_degrees_l934_93451

theorem cos_150_degrees : Real.cos (150 * π / 180) = -Real.sqrt 3 / 2 := by
  sorry

end cos_150_degrees_l934_93451


namespace max_product_min_sum_l934_93404

theorem max_product_min_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 2) :
  (∀ x y, x > 0 → y > 0 → x + y = 2 → x * y ≤ a * b → x * y ≤ 1) ∧
  (∀ x y, x > 0 → y > 0 → x + y = 2 → 2/x + 8/y ≥ 2/a + 8/b → 2/x + 8/y ≥ 9) := by
sorry

end max_product_min_sum_l934_93404


namespace power_of_power_l934_93487

theorem power_of_power (a : ℝ) : (a^2)^3 = a^6 := by
  sorry

end power_of_power_l934_93487


namespace geometry_class_ratio_l934_93448

theorem geometry_class_ratio (total_students : ℕ) (boys_under_6ft : ℕ) :
  total_students = 38 →
  (2 : ℚ) / 3 * total_students = 25 →
  boys_under_6ft = 19 →
  (boys_under_6ft : ℚ) / 25 = 19 / 25 := by
  sorry

end geometry_class_ratio_l934_93448


namespace equal_price_sheets_is_12_l934_93444

/-- The number of sheets for which two photo companies charge the same amount -/
def equal_price_sheets : ℕ :=
  let john_per_sheet : ℚ := 275 / 100
  let john_sitting_fee : ℚ := 125
  let sam_per_sheet : ℚ := 150 / 100
  let sam_sitting_fee : ℚ := 140
  ⌊(sam_sitting_fee - john_sitting_fee) / (john_per_sheet - sam_per_sheet)⌋₊

theorem equal_price_sheets_is_12 : equal_price_sheets = 12 := by
  sorry

#eval equal_price_sheets

end equal_price_sheets_is_12_l934_93444


namespace min_value_theorem_l934_93445

theorem min_value_theorem (c : ℝ) (hc : c > 0) (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0)
  (heq : a^2 - 2*a*b + 2*b^2 - c = 0) (hmax : ∀ a' b' : ℝ, a'^2 - 2*a'*b' + 2*b'^2 - c = 0 → a' + b' ≤ a + b) :
  ∃ (m : ℝ), m = -1/4 ∧ ∀ a' b' : ℝ, a' ≠ 0 → b' ≠ 0 → a'^2 - 2*a'*b' + 2*b'^2 - c = 0 →
    m ≤ (3/a' - 4/b' + 5/c) := by
  sorry

end min_value_theorem_l934_93445


namespace square_measurement_error_l934_93431

theorem square_measurement_error (S : ℝ) (S' : ℝ) (h : S > 0) :
  S'^2 = S^2 * (1 + 0.0404) → (S' - S) / S * 100 = 2 := by sorry

end square_measurement_error_l934_93431


namespace k_range_l934_93430

theorem k_range (k : ℝ) : (1 - k > -1 ∧ 1 - k ≤ 3) ↔ -2 ≤ k ∧ k < 2 := by
  sorry

end k_range_l934_93430


namespace unknown_percentage_of_250_l934_93412

/-- Given that 28% of 400 plus some percentage of 250 equals 224.5,
    prove that the unknown percentage of 250 is 45%. -/
theorem unknown_percentage_of_250 (p : ℝ) : 
  (0.28 * 400 + p / 100 * 250 = 224.5) → p = 45 :=
by sorry

end unknown_percentage_of_250_l934_93412


namespace probability_same_color_l934_93455

def num_green_balls : ℕ := 7
def num_white_balls : ℕ := 7

def total_balls : ℕ := num_green_balls + num_white_balls

def same_color_combinations : ℕ := (num_green_balls.choose 2) + (num_white_balls.choose 2)
def total_combinations : ℕ := total_balls.choose 2

theorem probability_same_color :
  (same_color_combinations : ℚ) / total_combinations = 42 / 91 := by
  sorry

end probability_same_color_l934_93455


namespace certain_number_problem_l934_93452

theorem certain_number_problem (x : ℝ) : 
  ((x + 10) * 2) / 2 - 2 = 88 / 2 → x = 36 := by
  sorry

end certain_number_problem_l934_93452


namespace min_points_eleventh_game_l934_93416

/-- Represents the scores of a basketball player -/
structure BasketballScores where
  scores_7_to_10 : Fin 4 → ℕ
  total_after_6 : ℕ
  total_after_10 : ℕ
  total_after_11 : ℕ

/-- The minimum number of points required in the 11th game -/
def min_points_11th_game (bs : BasketballScores) : ℕ := bs.total_after_11 - bs.total_after_10

/-- Theorem stating the minimum points required in the 11th game -/
theorem min_points_eleventh_game 
  (bs : BasketballScores)
  (h1 : bs.scores_7_to_10 = ![21, 15, 12, 19])
  (h2 : (bs.total_after_10 : ℚ) / 10 > (bs.total_after_6 : ℚ) / 6)
  (h3 : (bs.total_after_11 : ℚ) / 11 > 20)
  (h4 : bs.total_after_10 = bs.total_after_6 + (bs.scores_7_to_10 0) + (bs.scores_7_to_10 1) + 
                            (bs.scores_7_to_10 2) + (bs.scores_7_to_10 3))
  : min_points_11th_game bs = 58 := by
  sorry


end min_points_eleventh_game_l934_93416


namespace smallest_four_digit_divisible_by_four_l934_93475

theorem smallest_four_digit_divisible_by_four : 
  ∀ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 4 = 0 → n ≥ 1000 := by
  sorry

end smallest_four_digit_divisible_by_four_l934_93475


namespace initial_milk_water_ratio_l934_93463

/-- Proves that in a 60-litre mixture, if adding 60 litres of water changes
    the milk-to-water ratio to 1:2, then the initial milk-to-water ratio was 2:1 -/
theorem initial_milk_water_ratio (m w : ℝ) : 
  m + w = 60 →  -- Total initial volume is 60 litres
  2 * m = w + 60 →  -- After adding 60 litres of water, milk:water = 1:2
  m / w = 2 / 1 :=  -- Initial ratio of milk to water is 2:1
by sorry

end initial_milk_water_ratio_l934_93463


namespace problem_statement_l934_93469

theorem problem_statement (x y : ℝ) (h : x + 2*y = 30) : 
  x/5 + 2*y/3 + 2*y/5 + x/3 = 16 := by
sorry

end problem_statement_l934_93469


namespace remaining_pennies_l934_93432

theorem remaining_pennies (initial : ℝ) (spent : ℝ) (remaining : ℝ) 
  (h1 : initial = 98.5) 
  (h2 : spent = 93.25) 
  (h3 : remaining = initial - spent) : 
  remaining = 5.25 := by
  sorry

end remaining_pennies_l934_93432


namespace exists_player_in_interval_l934_93478

/-- Represents a round-robin tournament with 2n+1 players -/
structure Tournament (n : ℕ) where
  /-- The number of matches where the weaker player wins -/
  k : ℕ
  /-- The strength of each player, which are all different -/
  strength : Fin (2*n+1) → ℕ
  strength_injective : Function.Injective strength
  /-- The result of each match, where true means the first player won -/
  result : Fin (2*n+1) → Fin (2*n+1) → Bool
  /-- Each player plays exactly one match against every other player -/
  played_all : ∀ i j, i ≠ j → (result i j = true ∧ result j i = false) ∨ (result i j = false ∧ result j i = true)
  /-- Exactly k matches are won by the weaker player -/
  weaker_wins : (Finset.univ.filter (λ (p : Fin (2*n+1) × Fin (2*n+1)) => 
    p.1 ≠ p.2 ∧ strength p.1 < strength p.2 ∧ result p.1 p.2 = true)).card = k

/-- The number of victories for a player -/
def victories (t : Tournament n) (i : Fin (2*n+1)) : ℕ :=
  (Finset.univ.filter (λ j => j ≠ i ∧ t.result i j = true)).card

/-- The main theorem -/
theorem exists_player_in_interval (n : ℕ) (t : Tournament n) :
  ∃ i : Fin (2*n+1), n - Real.sqrt (2 * t.k) ≤ victories t i ∧ victories t i ≤ n + Real.sqrt (2 * t.k) := by
  sorry

end exists_player_in_interval_l934_93478


namespace sum_not_odd_l934_93414

theorem sum_not_odd (n m : ℤ) 
  (h1 : Even (n^3 + m^3))
  (h2 : (n^3 + m^3) % 4 = 0) : 
  ¬(Odd (n + m)) := by
sorry

end sum_not_odd_l934_93414


namespace darcy_laundry_theorem_l934_93428

/-- Given the number of shirts and shorts Darcy has, and the number he has folded,
    calculate the number of remaining pieces to fold. -/
def remaining_to_fold (total_shirts : ℕ) (total_shorts : ℕ) 
                      (folded_shirts : ℕ) (folded_shorts : ℕ) : ℕ :=
  (total_shirts - folded_shirts) + (total_shorts - folded_shorts)

/-- Theorem stating that with 20 shirts and 8 shorts, 
    if 12 shirts and 5 shorts are folded, 
    11 pieces remain to be folded. -/
theorem darcy_laundry_theorem : 
  remaining_to_fold 20 8 12 5 = 11 := by
  sorry

end darcy_laundry_theorem_l934_93428


namespace rectangular_field_area_l934_93400

theorem rectangular_field_area (length breadth : ℝ) : 
  breadth = 0.6 * length →
  2 * (length + breadth) = 800 →
  length * breadth = 37500 := by
sorry

end rectangular_field_area_l934_93400


namespace arithmetic_mean_of_range_l934_93481

def integer_range : List Int := List.range 10 |>.map (λ x => x - 3)

theorem arithmetic_mean_of_range : 
  (integer_range.sum : ℚ) / integer_range.length = 3/2 := by
  sorry

end arithmetic_mean_of_range_l934_93481


namespace inequality_proof_l934_93441

theorem inequality_proof (a b : ℝ) (h1 : a < b) (h2 : b < 0) :
  (a + 1/b)^2 > (b + 1/a)^2 := by
  sorry

end inequality_proof_l934_93441


namespace prime_difference_product_l934_93461

theorem prime_difference_product (a b : ℕ) : 
  Nat.Prime a → Nat.Prime b → a - b = 35 → a * b = 74 := by
  sorry

end prime_difference_product_l934_93461


namespace pie_chart_most_suitable_for_air_l934_93473

/-- Represents different types of statistical graphs -/
inductive StatGraph
  | PieChart
  | LineChart
  | BarChart

/-- Represents a substance composed of various components -/
structure Substance where
  components : List String

/-- Determines if a statistical graph is suitable for representing the composition of a substance -/
def is_suitable (graph : StatGraph) (substance : Substance) : Prop :=
  match graph with
  | StatGraph.PieChart => substance.components.length > 1
  | _ => False

/-- Air is a substance composed of various gases -/
def air : Substance :=
  { components := ["nitrogen", "oxygen", "argon", "carbon dioxide", "other gases"] }

/-- Theorem stating that a pie chart is the most suitable graph for representing air composition -/
theorem pie_chart_most_suitable_for_air :
  is_suitable StatGraph.PieChart air ∧
  ∀ (graph : StatGraph), graph ≠ StatGraph.PieChart → ¬(is_suitable graph air) :=
sorry

end pie_chart_most_suitable_for_air_l934_93473


namespace num_ways_to_sum_eq_two_pow_n_minus_one_l934_93409

/-- The number of ways to express a natural number as a sum of one or more natural numbers, considering the order of the terms. -/
def numWaysToSum (n : ℕ) : ℕ := 2^(n-1)

/-- Theorem: For any natural number n, the number of ways to express n as a sum of one or more natural numbers, considering the order of the terms, is equal to 2^(n-1). -/
theorem num_ways_to_sum_eq_two_pow_n_minus_one (n : ℕ) : 
  numWaysToSum n = 2^(n-1) := by
  sorry

end num_ways_to_sum_eq_two_pow_n_minus_one_l934_93409


namespace class_size_l934_93401

theorem class_size (initial_absent : ℚ) (final_absent : ℚ) (total : ℕ) : 
  initial_absent = 1 / 6 →
  final_absent = 1 / 5 →
  (initial_absent / (1 + initial_absent)) * total + 1 = (final_absent / (1 + final_absent)) * total →
  total = 42 := by
  sorry

end class_size_l934_93401


namespace lineup_ways_proof_l934_93484

/-- The number of ways to arrange 5 people in a line with restrictions -/
def lineupWays : ℕ := 72

/-- The number of people in the line -/
def totalPeople : ℕ := 5

/-- The number of positions where the youngest person can be placed -/
def youngestPositions : ℕ := 3

/-- The number of choices for the first position -/
def firstPositionChoices : ℕ := 4

/-- The number of ways to arrange the remaining people after placing the youngest -/
def remainingArrangements : ℕ := 6

theorem lineup_ways_proof :
  lineupWays = firstPositionChoices * youngestPositions * remainingArrangements :=
sorry

end lineup_ways_proof_l934_93484


namespace mice_breeding_experiment_l934_93485

/-- Calculates the final number of mice after two generations -/
def final_mice_count (initial_mice : ℕ) (pups_per_mouse : ℕ) (pups_eaten : ℕ) : ℕ :=
  let first_gen_pups := initial_mice * pups_per_mouse
  let total_after_first_gen := initial_mice + first_gen_pups
  let surviving_pups_per_mouse := pups_per_mouse - pups_eaten
  let second_gen_pups := total_after_first_gen * surviving_pups_per_mouse
  total_after_first_gen + second_gen_pups

/-- Theorem stating that the final number of mice is 280 given the initial conditions -/
theorem mice_breeding_experiment :
  final_mice_count 8 6 2 = 280 := by
  sorry

end mice_breeding_experiment_l934_93485


namespace sum_of_xy_l934_93495

theorem sum_of_xy (x y : ℕ) : 
  0 < x ∧ x < 30 ∧ 0 < y ∧ y < 30 ∧ x + y + x * y = 143 → 
  x + y = 22 ∨ x + y = 23 ∨ x + y = 24 := by
  sorry

end sum_of_xy_l934_93495
