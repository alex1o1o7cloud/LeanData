import Mathlib

namespace NUMINAMATH_CALUDE_intersection_of_P_and_Q_l564_56441

-- Define the set P
def P : Set ℝ := {x | x^2 - 7*x + 10 < 0}

-- Define the set Q
def Q : Set ℝ := {y | ∃ x ∈ P, y = x^2 - 8*x + 19}

-- Theorem statement
theorem intersection_of_P_and_Q : P ∩ Q = Set.Icc 3 5 := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_P_and_Q_l564_56441


namespace NUMINAMATH_CALUDE_equal_division_of_money_l564_56466

/-- Proves that when $3.75 is equally divided among 3 people, each person receives $1.25. -/
theorem equal_division_of_money (total_amount : ℚ) (num_people : ℕ) :
  total_amount = 3.75 ∧ num_people = 3 →
  total_amount / (num_people : ℚ) = 1.25 := by
  sorry

end NUMINAMATH_CALUDE_equal_division_of_money_l564_56466


namespace NUMINAMATH_CALUDE_equality_or_opposite_equality_l564_56460

theorem equality_or_opposite_equality (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  a^2 + b^3/a = b^2 + a^3/b → a = b ∨ a = -b := by
  sorry

end NUMINAMATH_CALUDE_equality_or_opposite_equality_l564_56460


namespace NUMINAMATH_CALUDE_exterior_angle_square_octagon_exterior_angle_square_octagon_proof_l564_56408

/-- The measure of the exterior angle formed by a regular square and a regular octagon that share a common side in a coplanar configuration is 135 degrees. -/
theorem exterior_angle_square_octagon : ℝ → Prop :=
  λ angle : ℝ =>
    let square_interior_angle : ℝ := 90
    let octagon_interior_angle : ℝ := 135
    let total_angle : ℝ := 360
    angle = total_angle - (square_interior_angle + octagon_interior_angle) ∧
    angle = 135

/-- Proof of the theorem -/
theorem exterior_angle_square_octagon_proof :
  ∃ angle : ℝ, exterior_angle_square_octagon angle :=
sorry

end NUMINAMATH_CALUDE_exterior_angle_square_octagon_exterior_angle_square_octagon_proof_l564_56408


namespace NUMINAMATH_CALUDE_iceland_visitors_l564_56417

theorem iceland_visitors (total : ℕ) (norway : ℕ) (both : ℕ) (neither : ℕ) :
  total = 60 →
  norway = 23 →
  both = 31 →
  neither = 33 →
  ∃ iceland : ℕ, iceland = 35 ∧ total = iceland + norway - both + neither :=
by sorry

end NUMINAMATH_CALUDE_iceland_visitors_l564_56417


namespace NUMINAMATH_CALUDE_range_of_a_l564_56455

theorem range_of_a (a b c d : ℝ) 
  (sum_eq : a + b + c + d = 3) 
  (sum_sq : a^2 + 2*b^2 + 3*c^2 + 6*d^2 = 5) : 
  1 ≤ a ∧ a ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l564_56455


namespace NUMINAMATH_CALUDE_mikes_shopping_cost_l564_56482

/-- The total amount Mike spent on shopping --/
def total_spent (food_cost wallet_cost shirt_cost : ℝ) : ℝ :=
  food_cost + wallet_cost + shirt_cost

/-- Theorem stating the total amount Mike spent on shopping --/
theorem mikes_shopping_cost :
  ∀ (food_cost wallet_cost shirt_cost : ℝ),
    food_cost = 30 →
    wallet_cost = food_cost + 60 →
    shirt_cost = wallet_cost / 3 →
    total_spent food_cost wallet_cost shirt_cost = 150 := by
  sorry

end NUMINAMATH_CALUDE_mikes_shopping_cost_l564_56482


namespace NUMINAMATH_CALUDE_last_letter_of_93rd_perm_l564_56402

def word := "BRAVE"

/-- Represents a permutation of the word "BRAVE" -/
def Permutation := Fin 5 → Char

/-- The set of all permutations of "BRAVE" -/
def all_permutations : Finset Permutation :=
  sorry

/-- Dictionary order for permutations -/
def dict_order (p q : Permutation) : Prop :=
  sorry

/-- The 93rd permutation in dictionary order -/
def perm_93 : Permutation :=
  sorry

theorem last_letter_of_93rd_perm :
  (perm_93 4) = 'R' :=
sorry

end NUMINAMATH_CALUDE_last_letter_of_93rd_perm_l564_56402


namespace NUMINAMATH_CALUDE_inequality_implies_m_upper_bound_l564_56475

theorem inequality_implies_m_upper_bound :
  (∀ x₁ : ℝ, ∃ x₂ ∈ Set.Icc 3 4, x₁^2 + x₁*x₂ + x₂^2 ≥ 2*x₁ + m*x₂ + 3) →
  m ≤ 3 :=
by sorry

end NUMINAMATH_CALUDE_inequality_implies_m_upper_bound_l564_56475


namespace NUMINAMATH_CALUDE_age_condition_l564_56428

/-- Given three people A, B, and C, this theorem states that if A is older than B,
    then "C is older than B" is a necessary but not sufficient condition for
    "the sum of B and C's ages is greater than twice A's age". -/
theorem age_condition (a b c : ℕ) (h : a > b) :
  (c > b → b + c > 2 * a) ∧ ¬(b + c > 2 * a → c > b) := by
  sorry

end NUMINAMATH_CALUDE_age_condition_l564_56428


namespace NUMINAMATH_CALUDE_chessboard_uniquely_determined_l564_56440

/-- Represents a cell on the chessboard --/
structure Cell :=
  (row : Fin 8)
  (col : Fin 8)

/-- Represents the chessboard configuration --/
def Chessboard := Cell → Fin 64

/-- Represents a 2-cell rectangle on the chessboard --/
structure Rectangle :=
  (cell1 : Cell)
  (cell2 : Cell)

/-- Function to get the sum of numbers in a 2-cell rectangle --/
def getRectangleSum (board : Chessboard) (rect : Rectangle) : Nat :=
  (board rect.cell1).val + 1 + (board rect.cell2).val + 1

/-- Predicate to check if two cells are on the same diagonal --/
def onSameDiagonal (c1 c2 : Cell) : Prop :=
  (c1.row.val + c1.col.val = c2.row.val + c2.col.val) ∨
  (c1.row.val - c1.col.val = c2.row.val - c2.col.val)

/-- The main theorem --/
theorem chessboard_uniquely_determined
  (board : Chessboard)
  (h1 : ∃ c1 c2 : Cell, (board c1 = 0) ∧ (board c2 = 63) ∧ onSameDiagonal c1 c2)
  (h2 : ∀ rect : Rectangle, ∃ s : Nat, getRectangleSum board rect = s) :
  ∀ c : Cell, ∃! n : Fin 64, board c = n :=
sorry

end NUMINAMATH_CALUDE_chessboard_uniquely_determined_l564_56440


namespace NUMINAMATH_CALUDE_shape_count_l564_56477

theorem shape_count (total_shapes : ℕ) (total_edges : ℕ) 
  (h1 : total_shapes = 13) 
  (h2 : total_edges = 47) : 
  ∃ (triangles squares : ℕ),
    triangles + squares = total_shapes ∧ 
    3 * triangles + 4 * squares = total_edges ∧
    triangles = 5 ∧ 
    squares = 8 := by
  sorry

end NUMINAMATH_CALUDE_shape_count_l564_56477


namespace NUMINAMATH_CALUDE_F_minimum_value_G_two_zeros_range_inequality_for_positive_x_l564_56416

noncomputable section

-- Define the functions
def f (x : ℝ) : ℝ := (1/2) * x^2
def g (a : ℝ) (x : ℝ) : ℝ := a * Real.log x
def F (a : ℝ) (x : ℝ) : ℝ := f x * g a x
def G (a : ℝ) (x : ℝ) : ℝ := f x - g a x + (a - 1) * x

-- Theorem 1: Minimum value of F(x)
theorem F_minimum_value (a : ℝ) (h : a > 0) :
  ∃ x₀ : ℝ, x₀ > 0 ∧ F a x₀ = -a / (4 * Real.exp 1) ∧ ∀ x > 0, F a x ≥ -a / (4 * Real.exp 1) :=
sorry

-- Theorem 2: Range of a for G(x) to have two zeros
theorem G_two_zeros_range :
  ∃ a₁ a₂ : ℝ, a₁ = (2 * Real.exp 1 - 1) / (2 * Real.exp 1^2 + 2 * Real.exp 1) ∧
               a₂ = 1/2 ∧
               ∀ a : ℝ, (∃ x₁ x₂ : ℝ, 1/Real.exp 1 < x₁ ∧ x₁ < x₂ ∧ x₂ < Real.exp 1 ∧
                                      G a x₁ = 0 ∧ G a x₂ = 0) ↔
                        (a₁ < a ∧ a < a₂) :=
sorry

-- Theorem 3: Inequality for x > 0
theorem inequality_for_positive_x (x : ℝ) (h : x > 0) :
  Real.log x + 3 / (4 * x^2) - 1 / Real.exp x > 0 :=
sorry

end NUMINAMATH_CALUDE_F_minimum_value_G_two_zeros_range_inequality_for_positive_x_l564_56416


namespace NUMINAMATH_CALUDE_natural_number_solution_xy_l564_56463

theorem natural_number_solution_xy : 
  ∀ (x y : ℕ), x + y = x * y ↔ x = 2 ∧ y = 2 := by sorry

end NUMINAMATH_CALUDE_natural_number_solution_xy_l564_56463


namespace NUMINAMATH_CALUDE_whatsis_equals_so_equals_four_l564_56411

/-- Given positive real numbers, prove that whatsis equals so and both equal 4 -/
theorem whatsis_equals_so_equals_four
  (whosis whatsis is so : ℝ)
  (h1 : whosis > 0)
  (h2 : whatsis > 0)
  (h3 : is > 0)
  (h4 : so > 0)
  (h5 : whosis = is)
  (h6 : so = so)
  (h7 : whosis = so)
  (h8 : so - is = 2)
  : whatsis = so ∧ so = 4 := by
  sorry

end NUMINAMATH_CALUDE_whatsis_equals_so_equals_four_l564_56411


namespace NUMINAMATH_CALUDE_binomial_coefficient_problem_l564_56478

theorem binomial_coefficient_problem (a₀ a₁ a₂ a₃ a₄ : ℝ) : 
  (∀ x, (2 + x)^4 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4) →
  (a₀ + a₂ + a₄)^2 - (a₁ + a₃)^2 = 625 := by
sorry

end NUMINAMATH_CALUDE_binomial_coefficient_problem_l564_56478


namespace NUMINAMATH_CALUDE_pyramid_volume_is_four_thirds_l564_56401

-- Define the cube IJKLMNO
structure Cube where
  volume : ℝ

-- Define the pyramid IJMO
structure Pyramid where
  base : Cube

-- Define the volume of the pyramid
def pyramid_volume (p : Pyramid) : ℝ := sorry

-- Theorem statement
theorem pyramid_volume_is_four_thirds (c : Cube) (p : Pyramid) 
  (h1 : c.volume = 8) 
  (h2 : p.base = c) : 
  pyramid_volume p = 4/3 := by sorry

end NUMINAMATH_CALUDE_pyramid_volume_is_four_thirds_l564_56401


namespace NUMINAMATH_CALUDE_deductible_increase_ratio_l564_56420

/-- The ratio of the increase in health insurance deductibles to the current annual deductible amount -/
theorem deductible_increase_ratio : 
  let current_deductible : ℚ := 3000
  let deductible_increase : ℚ := 2000
  (deductible_increase / current_deductible) = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_deductible_increase_ratio_l564_56420


namespace NUMINAMATH_CALUDE_line_through_point_with_opposite_intercepts_l564_56414

-- Define a line in 2D space
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define a point in 2D space
structure Point where
  x : ℝ
  y : ℝ

-- Function to check if a line passes through a point
def passesThrough (l : Line) (p : Point) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

-- Function to check if a line has opposite intercepts
def hasOppositeIntercepts (l : Line) : Prop :=
  (l.a ≠ 0 ∧ l.b ≠ 0) ∧ (l.c / l.a) * (l.c / l.b) < 0

-- Theorem statement
theorem line_through_point_with_opposite_intercepts :
  ∀ (l : Line),
    passesThrough l {x := 2, y := 3} →
    hasOppositeIntercepts l →
    (∃ (k : ℝ), l.a = k ∧ l.b = -k ∧ l.c = k) ∨
    (l.a = 3 ∧ l.b = -2 ∧ l.c = 0) :=
sorry

end NUMINAMATH_CALUDE_line_through_point_with_opposite_intercepts_l564_56414


namespace NUMINAMATH_CALUDE_max_product_sum_2024_l564_56450

theorem max_product_sum_2024 :
  (∃ (a b : ℤ), a + b = 2024 ∧ a * b = 1024144) ∧
  (∀ (x y : ℤ), x + y = 2024 → x * y ≤ 1024144) := by
  sorry

end NUMINAMATH_CALUDE_max_product_sum_2024_l564_56450


namespace NUMINAMATH_CALUDE_dealer_articles_purchased_l564_56431

theorem dealer_articles_purchased
  (total_purchase_price : ℝ)
  (num_articles_sold : ℕ)
  (total_selling_price : ℝ)
  (profit_percentage : ℝ)
  (h1 : total_purchase_price = 25)
  (h2 : num_articles_sold = 12)
  (h3 : total_selling_price = 33)
  (h4 : profit_percentage = 0.65)
  : ∃ (num_articles_purchased : ℕ),
    (num_articles_purchased : ℝ) * (total_selling_price / num_articles_sold) =
    total_purchase_price * (1 + profit_percentage) ∧
    num_articles_purchased = 15 :=
by sorry

end NUMINAMATH_CALUDE_dealer_articles_purchased_l564_56431


namespace NUMINAMATH_CALUDE_min_distance_four_points_l564_56430

/-- Given four points P, Q, R, and S on a line with specified distances between them,
    prove that the minimum possible distance between P and S is 0. -/
theorem min_distance_four_points (P Q R S : ℝ) 
  (h1 : |Q - P| = 12) 
  (h2 : |R - Q| = 7) 
  (h3 : |S - R| = 5) : 
  ∃ (P' Q' R' S' : ℝ), 
    |Q' - P'| = 12 ∧ 
    |R' - Q'| = 7 ∧ 
    |S' - R'| = 5 ∧ 
    |S' - P'| = 0 :=
sorry

end NUMINAMATH_CALUDE_min_distance_four_points_l564_56430


namespace NUMINAMATH_CALUDE_k_values_l564_56429

theorem k_values (p q r s k : ℂ) 
  (h_nonzero : p ≠ 0 ∧ q ≠ 0 ∧ r ≠ 0 ∧ s ≠ 0)
  (h_eq1 : p * k^3 + q * k^2 + r * k + s = 0)
  (h_eq2 : q * k^3 + r * k^2 + s * k + p = 0)
  (h_pqrs : p * q = r * s) :
  k = 1 ∨ k = -1 ∨ k = Complex.I ∨ k = -Complex.I :=
sorry

end NUMINAMATH_CALUDE_k_values_l564_56429


namespace NUMINAMATH_CALUDE_ramp_installation_cost_l564_56443

/-- Calculates the total cost of installing a ramp given specific conditions --/
theorem ramp_installation_cost :
  let permit_base_cost : ℝ := 250
  let permit_tax_rate : ℝ := 0.1
  let contractor_labor_rate : ℝ := 150
  let raw_materials_rate : ℝ := 50
  let work_days : ℕ := 3
  let work_hours_per_day : ℝ := 5
  let tool_rental_rate : ℝ := 30
  let lunch_break_hours : ℝ := 0.5
  let raw_materials_markup : ℝ := 0.15
  let inspector_rate_discount : ℝ := 0.8
  let inspector_hours_per_day : ℝ := 2

  let permit_cost : ℝ := permit_base_cost * (1 + permit_tax_rate)
  let raw_materials_cost_with_markup : ℝ := raw_materials_rate * (1 + raw_materials_markup)
  let contractor_hourly_cost : ℝ := contractor_labor_rate + raw_materials_cost_with_markup
  let total_work_hours : ℝ := work_days * work_hours_per_day
  let total_lunch_hours : ℝ := work_days * lunch_break_hours
  let tool_rental_cost : ℝ := tool_rental_rate * work_days
  let contractor_cost : ℝ := contractor_hourly_cost * (total_work_hours - total_lunch_hours) + tool_rental_cost
  let inspector_rate : ℝ := contractor_labor_rate * (1 - inspector_rate_discount)
  let inspector_cost : ℝ := inspector_rate * inspector_hours_per_day * work_days

  let total_cost : ℝ := permit_cost + contractor_cost + inspector_cost

  total_cost = 3432.5 := by sorry

end NUMINAMATH_CALUDE_ramp_installation_cost_l564_56443


namespace NUMINAMATH_CALUDE_grandmas_salad_ratio_l564_56446

/-- Prove that the ratio of bacon bits to pickles is 4:1 given the conditions in Grandma's salad --/
theorem grandmas_salad_ratio : 
  ∀ (mushrooms cherry_tomatoes pickles bacon_bits red_bacon_bits : ℕ),
    mushrooms = 3 →
    cherry_tomatoes = 2 * mushrooms →
    pickles = 4 * cherry_tomatoes →
    red_bacon_bits = 32 →
    3 * red_bacon_bits = bacon_bits →
    (bacon_bits : ℚ) / pickles = 4 / 1 := by
  sorry

end NUMINAMATH_CALUDE_grandmas_salad_ratio_l564_56446


namespace NUMINAMATH_CALUDE_stratified_sample_male_teachers_l564_56437

theorem stratified_sample_male_teachers 
  (male_teachers : ℕ) 
  (female_teachers : ℕ) 
  (sample_size : ℕ) 
  (h1 : male_teachers = 56) 
  (h2 : female_teachers = 42) 
  (h3 : sample_size = 14) : 
  ℕ :=
by
  -- The proof goes here
  sorry

#check stratified_sample_male_teachers

end NUMINAMATH_CALUDE_stratified_sample_male_teachers_l564_56437


namespace NUMINAMATH_CALUDE_fraction_product_l564_56407

theorem fraction_product : (1 / 3) * 9 * (1 / 27) * 81 * (1 / 243) * 729 = 27 := by
  sorry

end NUMINAMATH_CALUDE_fraction_product_l564_56407


namespace NUMINAMATH_CALUDE_cubic_polynomial_satisfies_conditions_l564_56433

def q (x : ℝ) : ℝ := -x^3 + 4*x^2 - 7*x - 4

theorem cubic_polynomial_satisfies_conditions :
  q 1 = -8 ∧ q 2 = -10 ∧ q 3 = -16 ∧ q 4 = -32 := by
  sorry

end NUMINAMATH_CALUDE_cubic_polynomial_satisfies_conditions_l564_56433


namespace NUMINAMATH_CALUDE_unique_solution_quadratic_l564_56434

theorem unique_solution_quadratic (k : ℝ) : 
  (∃! x : ℝ, (x + 5) * (x + 2) = k + 3 * x) ↔ k = 6 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_quadratic_l564_56434


namespace NUMINAMATH_CALUDE_john_mean_score_l564_56473

def john_scores : List ℝ := [86, 90, 88, 82, 91]

theorem john_mean_score : (john_scores.sum / john_scores.length : ℝ) = 87.4 := by
  sorry

end NUMINAMATH_CALUDE_john_mean_score_l564_56473


namespace NUMINAMATH_CALUDE_not_p_or_not_q_must_be_true_l564_56448

theorem not_p_or_not_q_must_be_true (h1 : ¬(p ∧ q)) (h2 : p ∨ q) : ¬p ∨ ¬q :=
by
  sorry

end NUMINAMATH_CALUDE_not_p_or_not_q_must_be_true_l564_56448


namespace NUMINAMATH_CALUDE_sqrt_square_eq_abs_l564_56456

theorem sqrt_square_eq_abs (x : ℝ) : Real.sqrt (x ^ 2) = |x| := by sorry

end NUMINAMATH_CALUDE_sqrt_square_eq_abs_l564_56456


namespace NUMINAMATH_CALUDE_b_age_is_ten_l564_56406

/-- Given three people a, b, and c, with the following conditions:
  1. a is two years older than b
  2. b is twice as old as c
  3. The sum of their ages is 27
  Prove that b is 10 years old. -/
theorem b_age_is_ten (a b c : ℕ) 
  (h1 : a = b + 2)
  (h2 : b = 2 * c)
  (h3 : a + b + c = 27) :
  b = 10 := by
  sorry

end NUMINAMATH_CALUDE_b_age_is_ten_l564_56406


namespace NUMINAMATH_CALUDE_evaluate_expression_l564_56479

theorem evaluate_expression (a : ℚ) (h : a = 4/3) :
  (6 * a^2 - 17 * a + 7) * (3 * a - 4) = 0 := by
sorry

end NUMINAMATH_CALUDE_evaluate_expression_l564_56479


namespace NUMINAMATH_CALUDE_summer_sales_is_seven_l564_56439

/-- The number of million hamburgers sold in each season --/
structure SeasonalSales where
  spring : ℝ
  summer : ℝ
  fall : ℝ
  winter : ℝ

/-- The total annual sales of hamburgers in millions --/
def total_sales (s : SeasonalSales) : ℝ :=
  s.spring + s.summer + s.fall + s.winter

/-- Theorem stating that the number of million hamburgers sold in the summer is 7 --/
theorem summer_sales_is_seven (s : SeasonalSales) 
  (h1 : s.fall = 0.2 * total_sales s)
  (h2 : s.fall = 3)
  (h3 : s.spring = 2)
  (h4 : s.winter = 3) : 
  s.summer = 7 := by
  sorry

end NUMINAMATH_CALUDE_summer_sales_is_seven_l564_56439


namespace NUMINAMATH_CALUDE_central_square_illumination_l564_56438

theorem central_square_illumination (n : ℕ) (h_odd : Odd n) :
  ∃ (min_lamps : ℕ),
    min_lamps = (n + 1)^2 / 2 ∧
    (∀ (lamps : ℕ),
      (∀ (i j : ℕ), i ≤ n ∧ j ≤ n →
        ∃ (k₁ k₂ : ℕ), k₁ ≠ k₂ ∧ k₁ ≤ lamps ∧ k₂ ≤ lamps ∧
          ((i = 0 ∨ i = n ∨ j = 0 ∨ j = n) →
            (k₁ ≤ 4 ∧ k₂ ≤ 4))) →
      lamps ≥ min_lamps) :=
by sorry

end NUMINAMATH_CALUDE_central_square_illumination_l564_56438


namespace NUMINAMATH_CALUDE_total_fertilizer_used_l564_56499

/-- The amount of fertilizer used per day for the first 9 days -/
def normal_amount : ℕ := 2

/-- The number of days the florist uses the normal amount of fertilizer -/
def normal_days : ℕ := 9

/-- The extra amount of fertilizer used on the final day -/
def extra_amount : ℕ := 4

/-- The total number of days the florist uses fertilizer -/
def total_days : ℕ := normal_days + 1

/-- Theorem: The total amount of fertilizer used over 10 days is 24 pounds -/
theorem total_fertilizer_used : 
  normal_amount * normal_days + (normal_amount + extra_amount) = 24 := by
  sorry

end NUMINAMATH_CALUDE_total_fertilizer_used_l564_56499


namespace NUMINAMATH_CALUDE_smallest_solution_of_equation_l564_56484

theorem smallest_solution_of_equation (y : ℝ) :
  (3 * y^2 + 33 * y - 90 = y * (y + 16)) →
  y ≥ -10 :=
by sorry

end NUMINAMATH_CALUDE_smallest_solution_of_equation_l564_56484


namespace NUMINAMATH_CALUDE_cubic_root_sum_cubes_l564_56462

theorem cubic_root_sum_cubes (r s t : ℂ) : 
  (9 * r^3 + 2023 * r + 4047 = 0) →
  (9 * s^3 + 2023 * s + 4047 = 0) →
  (9 * t^3 + 2023 * t + 4047 = 0) →
  (r + s)^3 + (s + t)^3 + (t + r)^3 = 1349 := by
  sorry

end NUMINAMATH_CALUDE_cubic_root_sum_cubes_l564_56462


namespace NUMINAMATH_CALUDE_paiges_team_size_l564_56442

theorem paiges_team_size (total_points : ℕ) (paige_points : ℕ) (other_player_points : ℕ) :
  total_points = 41 →
  paige_points = 11 →
  other_player_points = 6 →
  ∃ (team_size : ℕ), team_size = (total_points - paige_points) / other_player_points + 1 ∧ team_size = 6 :=
by sorry

end NUMINAMATH_CALUDE_paiges_team_size_l564_56442


namespace NUMINAMATH_CALUDE_second_car_distance_l564_56449

/-- Calculates the distance traveled by the second car given the initial separation,
    the distance traveled by the first car, and the final distance between the cars. -/
def distance_traveled_by_second_car (initial_separation : ℝ) (distance_first_car : ℝ) (final_distance : ℝ) : ℝ :=
  initial_separation - (distance_first_car + final_distance)

/-- Theorem stating that given the conditions of the problem, 
    the second car must have traveled 87 km. -/
theorem second_car_distance : 
  let initial_separation : ℝ := 150
  let distance_first_car : ℝ := 25
  let final_distance : ℝ := 38
  distance_traveled_by_second_car initial_separation distance_first_car final_distance = 87 := by
  sorry

#eval distance_traveled_by_second_car 150 25 38

end NUMINAMATH_CALUDE_second_car_distance_l564_56449


namespace NUMINAMATH_CALUDE_furniture_purchase_cost_l564_56469

/-- Calculate the final cost of furniture purchase --/
theorem furniture_purchase_cost :
  let table_cost : ℚ := 140
  let chair_cost : ℚ := table_cost / 7
  let sofa_cost : ℚ := 2 * table_cost
  let num_chairs : ℕ := 4
  let table_discount_rate : ℚ := 1 / 10
  let sales_tax_rate : ℚ := 7 / 100
  let exchange_rate : ℚ := 12 / 10

  let total_chair_cost : ℚ := num_chairs * chair_cost
  let discounted_table_cost : ℚ := table_cost * (1 - table_discount_rate)
  let subtotal : ℚ := discounted_table_cost + total_chair_cost + sofa_cost
  let sales_tax : ℚ := subtotal * sales_tax_rate
  let final_cost : ℚ := subtotal + sales_tax

  final_cost = 52002 / 100 := by sorry

end NUMINAMATH_CALUDE_furniture_purchase_cost_l564_56469


namespace NUMINAMATH_CALUDE_georgie_enter_exit_ways_l564_56459

/-- The number of windows in the haunted mansion -/
def num_windows : ℕ := 8

/-- The number of ways Georgie can enter and exit the mansion -/
def num_ways : ℕ := num_windows * (num_windows - 1)

/-- Theorem stating that the number of ways Georgie can enter and exit is 56 -/
theorem georgie_enter_exit_ways : num_ways = 56 := by
  sorry

end NUMINAMATH_CALUDE_georgie_enter_exit_ways_l564_56459


namespace NUMINAMATH_CALUDE_unique_root_in_interval_l564_56465

/-- Theorem: Given a cubic function f(x) = -2x^3 - x + 1 defined on the interval [m, n],
    where f(m)f(n) < 0, the equation f(x) = 0 has exactly one real root in the interval [m, n]. -/
theorem unique_root_in_interval (m n : ℝ) (h : m ≤ n) :
  let f : ℝ → ℝ := λ x ↦ -2 * x^3 - x + 1
  (f m) * (f n) < 0 →
  ∃! x, m ≤ x ∧ x ≤ n ∧ f x = 0 :=
by sorry

end NUMINAMATH_CALUDE_unique_root_in_interval_l564_56465


namespace NUMINAMATH_CALUDE_pedestrian_speed_problem_l564_56400

/-- The problem of two pedestrians traveling between points A and B -/
theorem pedestrian_speed_problem (x : ℝ) :
  x > 0 →  -- The speed must be positive
  (11 / (x + 1.5) + 8 / x ≥ 12 / (x + 2) + 2) →  -- The inequality from the problem
  x ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_pedestrian_speed_problem_l564_56400


namespace NUMINAMATH_CALUDE_stratified_sampling_l564_56458

theorem stratified_sampling (seniors juniors freshmen sampled_freshmen : ℕ) :
  seniors = 1000 →
  juniors = 1200 →
  freshmen = 1500 →
  sampled_freshmen = 75 →
  (seniors + juniors + freshmen) * sampled_freshmen / freshmen = 185 := by
sorry

end NUMINAMATH_CALUDE_stratified_sampling_l564_56458


namespace NUMINAMATH_CALUDE_students_playing_sports_l564_56405

theorem students_playing_sports (B C : Finset Nat) : 
  (B.card = 7) → 
  (C.card = 8) → 
  ((B ∩ C).card = 5) → 
  ((B ∪ C).card = 10) := by
sorry

end NUMINAMATH_CALUDE_students_playing_sports_l564_56405


namespace NUMINAMATH_CALUDE_minimize_expression_l564_56404

theorem minimize_expression (a b : ℝ) (ha : a > 0) (hb : b > 2) (hab : a + b = 3) :
  ∃ (min_a : ℝ), min_a = 2/3 ∧
  ∀ (x : ℝ), x > 0 → x + b = 3 →
  (4/x + 1/(b-2)) ≥ (4/min_a + 1/(b-2)) :=
by sorry

end NUMINAMATH_CALUDE_minimize_expression_l564_56404


namespace NUMINAMATH_CALUDE_machine_production_in_10_seconds_l564_56474

/-- A machine that produces items at a constant rate -/
structure Machine where
  items_per_minute : ℕ

/-- Calculate the number of items produced in a given number of seconds -/
def items_produced (m : Machine) (seconds : ℕ) : ℚ :=
  (m.items_per_minute : ℚ) * (seconds : ℚ) / 60

theorem machine_production_in_10_seconds (m : Machine) 
  (h : m.items_per_minute = 150) : 
  items_produced m 10 = 25 := by
  sorry

end NUMINAMATH_CALUDE_machine_production_in_10_seconds_l564_56474


namespace NUMINAMATH_CALUDE_aquarium_visitors_l564_56481

/-- Calculates the number of healthy visitors given total visitors and ill percentage --/
def healthyVisitors (total : ℕ) (illPercentage : ℕ) : ℕ :=
  total - (total * illPercentage) / 100

/-- Properties of the aquarium visits over three days --/
theorem aquarium_visitors :
  let mondayTotal := 300
  let mondayIllPercentage := 15
  let tuesdayTotal := 500
  let tuesdayIllPercentage := 30
  let wednesdayTotal := 400
  let wednesdayIllPercentage := 20
  
  (healthyVisitors mondayTotal mondayIllPercentage +
   healthyVisitors tuesdayTotal tuesdayIllPercentage +
   healthyVisitors wednesdayTotal wednesdayIllPercentage) = 925 := by
  sorry

end NUMINAMATH_CALUDE_aquarium_visitors_l564_56481


namespace NUMINAMATH_CALUDE_rationalize_sqrt_sum_l564_56432

def rationalize_and_simplify (x y z : ℝ) : ℝ × ℝ × ℝ × ℝ × ℝ × ℝ :=
  sorry

theorem rationalize_sqrt_sum : 
  let (A, B, C, D, E, F) := rationalize_and_simplify 5 2 7
  A + B + C + D + E + F = 84 := by sorry

end NUMINAMATH_CALUDE_rationalize_sqrt_sum_l564_56432


namespace NUMINAMATH_CALUDE_parabola_axis_equation_l564_56412

-- Define the curve f(x)
def f (x : ℝ) : ℝ := x^3 + x^2 + x + 3

-- Define the tangent line at x = -1
def tangent_line (x : ℝ) : ℝ := 2*x + 4

-- Define the parabola y = 2px²
def parabola (p : ℝ) (x : ℝ) : ℝ := 2*p*x^2

-- Theorem statement
theorem parabola_axis_equation :
  ∃ (p : ℝ), (∀ (x : ℝ), tangent_line x = parabola p x → x = -1 ∨ x ≠ -1) →
  (∀ (x : ℝ), parabola p x = -(1/4)*x^2) →
  (∀ (x : ℝ), x^2 = -4*1 → x = 0) :=
sorry

end NUMINAMATH_CALUDE_parabola_axis_equation_l564_56412


namespace NUMINAMATH_CALUDE_cloth_selling_price_l564_56487

/-- Given a cloth with the following properties:
  * Total length: 60 meters
  * Cost price per meter: 128 Rs
  * Profit per meter: 12 Rs
  Prove that the total selling price is 8400 Rs. -/
theorem cloth_selling_price 
  (total_length : ℕ) 
  (cost_price_per_meter : ℕ) 
  (profit_per_meter : ℕ) 
  (h1 : total_length = 60)
  (h2 : cost_price_per_meter = 128)
  (h3 : profit_per_meter = 12) :
  (cost_price_per_meter + profit_per_meter) * total_length = 8400 := by
  sorry

end NUMINAMATH_CALUDE_cloth_selling_price_l564_56487


namespace NUMINAMATH_CALUDE_pentagonal_base_monochromatic_l564_56496

-- Define the vertices of the prism
inductive Vertex : Type
| A : Fin 5 → Vertex
| B : Fin 5 → Vertex

-- Define the color of an edge
inductive Color : Type
| Red
| Blue

-- Define the edge coloring function
def edge_color : Vertex → Vertex → Color := sorry

-- No triangle has all edges of the same color
axiom no_monochromatic_triangle :
  ∀ (v1 v2 v3 : Vertex),
    v1 ≠ v2 → v2 ≠ v3 → v3 ≠ v1 →
    ¬(edge_color v1 v2 = edge_color v2 v3 ∧ edge_color v2 v3 = edge_color v3 v1)

-- Theorem: All edges of each pentagonal base are the same color
theorem pentagonal_base_monochromatic :
  (∀ (i j : Fin 5), edge_color (Vertex.A i) (Vertex.A j) = edge_color (Vertex.A 0) (Vertex.A 1)) ∧
  (∀ (i j : Fin 5), edge_color (Vertex.B i) (Vertex.B j) = edge_color (Vertex.B 0) (Vertex.B 1)) :=
sorry

end NUMINAMATH_CALUDE_pentagonal_base_monochromatic_l564_56496


namespace NUMINAMATH_CALUDE_u_closed_form_l564_56403

def u : ℕ → ℤ
  | 0 => 1
  | 1 => 4
  | (n + 2) => 5 * u (n + 1) - 6 * u n

theorem u_closed_form (n : ℕ) : u n = 2 * 3^n - 2^n := by
  sorry

end NUMINAMATH_CALUDE_u_closed_form_l564_56403


namespace NUMINAMATH_CALUDE_de_plus_ef_sum_l564_56494

/-- Represents a polygon ABCDEF with specific properties -/
structure Polygon where
  area : ℝ
  ab : ℝ
  bc : ℝ
  fa : ℝ
  de_parallel_ab : Prop
  df_horizontal : ℝ

/-- Theorem stating the sum of DE and EF in the given polygon -/
theorem de_plus_ef_sum (p : Polygon) 
  (h1 : p.area = 75)
  (h2 : p.ab = 7)
  (h3 : p.bc = 10)
  (h4 : p.fa = 6)
  (h5 : p.de_parallel_ab)
  (h6 : p.df_horizontal = 8) :
  ∃ (de ef : ℝ), de + ef = 8.25 := by
  sorry

end NUMINAMATH_CALUDE_de_plus_ef_sum_l564_56494


namespace NUMINAMATH_CALUDE_total_palm_trees_l564_56435

theorem total_palm_trees (forest_trees : ℕ) (desert_reduction : ℚ) (river_trees : ℕ)
  (h1 : forest_trees = 5000)
  (h2 : desert_reduction = 3 / 5)
  (h3 : river_trees = 1200) :
  forest_trees + (forest_trees - desert_reduction * forest_trees) + river_trees = 8200 :=
by sorry

end NUMINAMATH_CALUDE_total_palm_trees_l564_56435


namespace NUMINAMATH_CALUDE_hyperbola_a_value_l564_56425

/-- The value of 'a' for a hyperbola with equation x²/a² - y² = 1, a > 0, and eccentricity √5 -/
theorem hyperbola_a_value (a : ℝ) (h1 : a > 0) 
  (h2 : ∀ x y : ℝ, x^2 / a^2 - y^2 = 1) 
  (h3 : ∃ c : ℝ, c / a = Real.sqrt 5) : a = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_a_value_l564_56425


namespace NUMINAMATH_CALUDE_quadratic_one_zero_l564_56488

def f (x : ℝ) := x^2 - 4*x + 4

theorem quadratic_one_zero :
  ∃! x, f x = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_one_zero_l564_56488


namespace NUMINAMATH_CALUDE_mp3_song_count_l564_56485

theorem mp3_song_count (initial_songs : ℕ) (deleted_songs : ℕ) (added_songs : ℕ) 
  (h1 : initial_songs = 15)
  (h2 : deleted_songs = 8)
  (h3 : added_songs = 50) :
  initial_songs - deleted_songs + added_songs = 57 := by
  sorry

end NUMINAMATH_CALUDE_mp3_song_count_l564_56485


namespace NUMINAMATH_CALUDE_smallest_r_is_two_l564_56451

theorem smallest_r_is_two :
  ∃ (r : ℝ), r > 0 ∧ r = 2 ∧
  (∀ (a : ℝ), a > 0 →
    ∃ (x : ℝ), (2 - a * r ≤ x) ∧ (x ≤ 2) ∧ (a * x^3 + x^2 - 4 = 0)) ∧
  (∀ (r' : ℝ), r' > 0 →
    (∀ (a : ℝ), a > 0 →
      ∃ (x : ℝ), (2 - a * r' ≤ x) ∧ (x ≤ 2) ∧ (a * x^3 + x^2 - 4 = 0)) →
    r' ≥ r) :=
by sorry

end NUMINAMATH_CALUDE_smallest_r_is_two_l564_56451


namespace NUMINAMATH_CALUDE_solution_set_inequality_l564_56423

theorem solution_set_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) (h_eq : 1/a + 2/b = 1) :
  {x : ℝ | (2 : ℝ)^(|x-1|-|x+2|) < 1} = {x : ℝ | x > -1/2} := by
  sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l564_56423


namespace NUMINAMATH_CALUDE_modulus_of_complex_fraction_l564_56497

theorem modulus_of_complex_fraction : 
  let z : ℂ := (1 + Complex.I * Real.sqrt 3) / (1 - Complex.I)
  Complex.abs z = Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_modulus_of_complex_fraction_l564_56497


namespace NUMINAMATH_CALUDE_order_of_abc_l564_56426

theorem order_of_abc : 
  let a : ℝ := (Real.exp 0.6)⁻¹
  let b : ℝ := 0.4
  let c : ℝ := (Real.log 1.4) / 1.4
  c < b ∧ b < a := by sorry

end NUMINAMATH_CALUDE_order_of_abc_l564_56426


namespace NUMINAMATH_CALUDE_number_of_apples_l564_56427

/-- Given a box of fruit with the following properties:
  * The total number of fruit pieces is 56
  * One-fourth of the fruit are oranges
  * The number of peaches is half the number of oranges
  * The number of apples is five times the number of peaches
  This theorem proves that the number of apples in the box is 35. -/
theorem number_of_apples (total : ℕ) (oranges peaches apples : ℕ) : 
  total = 56 →
  oranges = total / 4 →
  peaches = oranges / 2 →
  apples = 5 * peaches →
  apples = 35 := by
  sorry

end NUMINAMATH_CALUDE_number_of_apples_l564_56427


namespace NUMINAMATH_CALUDE_betty_boxes_l564_56436

theorem betty_boxes (total_oranges : ℕ) (oranges_per_box : ℕ) (boxes : ℕ) : 
  total_oranges = 24 → 
  oranges_per_box = 8 → 
  total_oranges = boxes * oranges_per_box → 
  boxes = 3 := by
sorry

end NUMINAMATH_CALUDE_betty_boxes_l564_56436


namespace NUMINAMATH_CALUDE_inequalities_proof_l564_56453

theorem inequalities_proof (a b : ℝ) : 
  (a^2 + b^2 ≥ (a + b)^2 / 2) ∧ (a^2 + b^2 ≥ 2*(a - b - 1)) := by
  sorry

end NUMINAMATH_CALUDE_inequalities_proof_l564_56453


namespace NUMINAMATH_CALUDE_milk_production_theorem_l564_56468

/-- Represents the milk production scenario with varying cow efficiencies -/
structure MilkProduction where
  a : ℕ  -- number of cows in original group
  b : ℝ  -- gallons of milk produced by original group
  c : ℕ  -- number of days for original group
  d : ℕ  -- number of cows in new group
  e : ℕ  -- number of days for new group
  h : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0  -- ensure positive values

/-- The theorem stating the milk production for the new group -/
theorem milk_production_theorem (mp : MilkProduction) :
  let avg_rate := mp.b / (mp.a * mp.c)
  let efficient_rate := 2 * avg_rate
  let inefficient_rate := avg_rate / 2
  let new_production := mp.d * (efficient_rate * mp.a / 2 + inefficient_rate * mp.a / 2) / mp.a * mp.e
  new_production = mp.d * mp.b * mp.e / (mp.a * mp.c) := by
  sorry

#check milk_production_theorem

end NUMINAMATH_CALUDE_milk_production_theorem_l564_56468


namespace NUMINAMATH_CALUDE_probability_green_or_blue_l564_56409

/-- The probability of drawing a green or blue marble from a bag -/
theorem probability_green_or_blue (green blue yellow : ℕ) 
  (hg : green = 4) (hb : blue = 3) (hy : yellow = 8) : 
  (green + blue : ℚ) / (green + blue + yellow) = 7 / 15 := by
  sorry

end NUMINAMATH_CALUDE_probability_green_or_blue_l564_56409


namespace NUMINAMATH_CALUDE_sqrt_three_squared_five_fourth_l564_56471

theorem sqrt_three_squared_five_fourth (x : ℝ) : 
  x = Real.sqrt (3^2 * 5^4) → x = 75 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_three_squared_five_fourth_l564_56471


namespace NUMINAMATH_CALUDE_hawkeye_battery_charges_l564_56419

def battery_problem (cost_per_charge : ℚ) (initial_budget : ℚ) (remaining_money : ℚ) : Prop :=
  let total_spent : ℚ := initial_budget - remaining_money
  let number_of_charges : ℚ := total_spent / cost_per_charge
  number_of_charges = 4

theorem hawkeye_battery_charges : 
  battery_problem (35/10) 20 6 := by
  sorry

end NUMINAMATH_CALUDE_hawkeye_battery_charges_l564_56419


namespace NUMINAMATH_CALUDE_total_percentage_increase_l564_56444

/-- Calculates the total percentage increase in a purchase of three items given their initial and final prices. -/
theorem total_percentage_increase
  (book_initial : ℝ) (book_final : ℝ)
  (album_initial : ℝ) (album_final : ℝ)
  (poster_initial : ℝ) (poster_final : ℝ)
  (h1 : book_initial = 300)
  (h2 : book_final = 480)
  (h3 : album_initial = 15)
  (h4 : album_final = 20)
  (h5 : poster_initial = 5)
  (h6 : poster_final = 10) :
  (((book_final + album_final + poster_final) - (book_initial + album_initial + poster_initial)) / (book_initial + album_initial + poster_initial)) * 100 = 59.375 := by
  sorry

end NUMINAMATH_CALUDE_total_percentage_increase_l564_56444


namespace NUMINAMATH_CALUDE_probability_two_same_color_l564_56447

def total_balls : ℕ := 6
def balls_per_color : ℕ := 2
def num_colors : ℕ := 3
def balls_drawn : ℕ := 3

def total_ways : ℕ := Nat.choose total_balls balls_drawn

def ways_two_same_color : ℕ := num_colors * (Nat.choose balls_per_color 2) * (total_balls - balls_per_color)

theorem probability_two_same_color :
  (ways_two_same_color : ℚ) / total_ways = 3 / 5 := by sorry

end NUMINAMATH_CALUDE_probability_two_same_color_l564_56447


namespace NUMINAMATH_CALUDE_consecutive_integers_median_l564_56410

theorem consecutive_integers_median (n : ℕ) (sum : ℕ) (median : ℕ) : 
  n = 36 →
  sum = 3125 →
  (∃ (start : ℤ), sum = (start + start + n - 1) * n / 2) →
  median = 89 :=
by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_median_l564_56410


namespace NUMINAMATH_CALUDE_angle_with_same_terminal_side_l564_56454

/-- Given an angle α and another angle θ, proves that if α = 1560°, 
    θ has the same terminal side as α, and -360° < θ < 360°, 
    then θ = 120° or θ = -240°. -/
theorem angle_with_same_terminal_side 
  (α θ : ℝ) 
  (h1 : α = 1560)
  (h2 : ∃ (k : ℤ), θ = 360 * k + 120)
  (h3 : -360 < θ ∧ θ < 360) :
  θ = 120 ∨ θ = -240 :=
sorry

end NUMINAMATH_CALUDE_angle_with_same_terminal_side_l564_56454


namespace NUMINAMATH_CALUDE_fence_cost_circular_plot_l564_56472

/-- The cost of building a fence around a circular plot -/
theorem fence_cost_circular_plot (area : ℝ) (price_per_foot : ℝ) : 
  area = 289 → price_per_foot = 58 → 
  (2 * Real.sqrt area * price_per_foot : ℝ) = 1972 := by
  sorry

#check fence_cost_circular_plot

end NUMINAMATH_CALUDE_fence_cost_circular_plot_l564_56472


namespace NUMINAMATH_CALUDE_cube_roots_opposite_implies_a_eq_neg_three_l564_56498

theorem cube_roots_opposite_implies_a_eq_neg_three (a : ℝ) :
  (∃ x : ℝ, x^3 = 2*a + 1 ∧ (-x)^3 = 2 - a) → a = -3 := by
sorry

end NUMINAMATH_CALUDE_cube_roots_opposite_implies_a_eq_neg_three_l564_56498


namespace NUMINAMATH_CALUDE_value_of_c_l564_56490

theorem value_of_c (a c : ℝ) (h1 : 3 * a + 2 = 2) (h2 : c - a = 3) : c = 3 := by
  sorry

end NUMINAMATH_CALUDE_value_of_c_l564_56490


namespace NUMINAMATH_CALUDE_rectangle_area_equals_44_l564_56457

/-- Given a triangle with sides a, b, c and a rectangle with one side length d,
    if the perimeters are equal and d = 8, then the area of the rectangle is 44. -/
theorem rectangle_area_equals_44 (a b c d : ℝ) : 
  a = 7.5 → b = 9 → c = 10.5 → d = 8 → 
  a + b + c = 2 * (d + (a + b + c) / 2 - d) → 
  d * ((a + b + c) / 2 - d) = 44 := by sorry

end NUMINAMATH_CALUDE_rectangle_area_equals_44_l564_56457


namespace NUMINAMATH_CALUDE_max_rented_trucks_l564_56452

theorem max_rented_trucks (total_trucks : ℕ) (return_rate : ℚ) (min_saturday_trucks : ℕ) :
  total_trucks = 20 →
  return_rate = 1/2 →
  min_saturday_trucks = 10 →
  ∃ (max_rented : ℕ), max_rented ≤ total_trucks ∧
    max_rented * return_rate = total_trucks - min_saturday_trucks ∧
    ∀ (rented : ℕ), rented ≤ total_trucks ∧ 
      rented * return_rate = total_trucks - min_saturday_trucks →
      rented ≤ max_rented :=
by sorry

end NUMINAMATH_CALUDE_max_rented_trucks_l564_56452


namespace NUMINAMATH_CALUDE_kendra_evening_minivans_l564_56415

/-- The number of minivans Kendra saw in the afternoon -/
def afternoon_minivans : ℕ := 4

/-- The total number of minivans Kendra saw -/
def total_minivans : ℕ := 5

/-- The number of minivans Kendra saw in the evening -/
def evening_minivans : ℕ := total_minivans - afternoon_minivans

theorem kendra_evening_minivans : evening_minivans = 1 := by
  sorry

end NUMINAMATH_CALUDE_kendra_evening_minivans_l564_56415


namespace NUMINAMATH_CALUDE_smallest_integer_l564_56422

theorem smallest_integer (n : ℕ+) : 
  (Nat.lcm 36 n.val) / (Nat.gcd 36 n.val) = 24 → n.val ≥ 96 := by
  sorry

end NUMINAMATH_CALUDE_smallest_integer_l564_56422


namespace NUMINAMATH_CALUDE_counterexample_exists_l564_56480

theorem counterexample_exists : ∃ (a b : ℝ), a^2 > b^2 ∧ a ≤ b := by
  sorry

end NUMINAMATH_CALUDE_counterexample_exists_l564_56480


namespace NUMINAMATH_CALUDE_parabola_directrix_l564_56467

/-- Given a parabola defined by x = -1/4 * y^2, its directrix is the line x = 1 -/
theorem parabola_directrix (x y : ℝ) : 
  (x = -(1/4) * y^2) → (∃ (k : ℝ), k = 1 ∧ k = x) := by sorry

end NUMINAMATH_CALUDE_parabola_directrix_l564_56467


namespace NUMINAMATH_CALUDE_polynomial_expansion_properties_l564_56489

theorem polynomial_expansion_properties 
  (x a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) 
  (h : ∀ x, (2*x - 1)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) : 
  (a₀ = -1) ∧ (a₀ + a₁ + a₂ + a₃ + a₄ + a₅ = 1) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_expansion_properties_l564_56489


namespace NUMINAMATH_CALUDE_second_integer_value_l564_56486

theorem second_integer_value (n : ℤ) : (n - 2) + (n + 2) = 132 → n = 66 := by
  sorry

end NUMINAMATH_CALUDE_second_integer_value_l564_56486


namespace NUMINAMATH_CALUDE_arcsin_neg_one_l564_56491

theorem arcsin_neg_one : Real.arcsin (-1) = -π / 2 := by sorry

end NUMINAMATH_CALUDE_arcsin_neg_one_l564_56491


namespace NUMINAMATH_CALUDE_least_addition_for_divisibility_l564_56495

theorem least_addition_for_divisibility (n : ℕ) : 
  (1024 + n) % 25 = 0 ∧ ∀ m : ℕ, m < n → (1024 + m) % 25 ≠ 0 ↔ n = 1 :=
by sorry

end NUMINAMATH_CALUDE_least_addition_for_divisibility_l564_56495


namespace NUMINAMATH_CALUDE_total_amount_after_ten_years_l564_56445

/-- Calculates the total amount after compound interest -/
def compound_interest (principal : ℝ) (rate : ℝ) (years : ℕ) : ℝ :=
  principal * (1 + rate) ^ years

/-- Theorem: The total amount after 10 years with 5% annual interest rate -/
theorem total_amount_after_ten_years :
  let initial_deposit : ℝ := 100000
  let interest_rate : ℝ := 0.05
  let years : ℕ := 10
  compound_interest initial_deposit interest_rate years = initial_deposit * (1 + interest_rate) ^ years :=
by sorry

end NUMINAMATH_CALUDE_total_amount_after_ten_years_l564_56445


namespace NUMINAMATH_CALUDE_log_ratio_squared_l564_56413

theorem log_ratio_squared (x y : ℝ) (hx : x > 0) (hy : y > 0) (hx1 : x ≠ 1) (hy1 : y ≠ 1)
  (h1 : Real.log x / Real.log 3 = Real.log 81 / Real.log y)
  (h2 : x * y = 243) :
  (Real.log (x / y) / Real.log 3)^2 = 9 := by
sorry

end NUMINAMATH_CALUDE_log_ratio_squared_l564_56413


namespace NUMINAMATH_CALUDE_complex_equation_solution_l564_56418

theorem complex_equation_solution (z : ℂ) (h : z * (1 - Complex.I) = 2) : z = 1 + Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l564_56418


namespace NUMINAMATH_CALUDE_remy_water_usage_l564_56464

theorem remy_water_usage (roman_usage : ℕ) 
  (h1 : roman_usage + (3 * roman_usage + 1) = 33) : 
  3 * roman_usage + 1 = 25 := by
  sorry

end NUMINAMATH_CALUDE_remy_water_usage_l564_56464


namespace NUMINAMATH_CALUDE_diagonal_cells_in_rectangle_diagonal_cells_199_991_l564_56424

theorem diagonal_cells_in_rectangle : ℕ → ℕ → ℕ
  | m, n => m + n - Nat.gcd m n

theorem diagonal_cells_199_991 :
  diagonal_cells_in_rectangle 199 991 = 1189 := by
  sorry

end NUMINAMATH_CALUDE_diagonal_cells_in_rectangle_diagonal_cells_199_991_l564_56424


namespace NUMINAMATH_CALUDE_power_mod_seven_l564_56493

theorem power_mod_seven : 2^2004 % 7 = 1 := by sorry

end NUMINAMATH_CALUDE_power_mod_seven_l564_56493


namespace NUMINAMATH_CALUDE_ratio_problem_l564_56483

theorem ratio_problem (a b : ℚ) (h1 : b / a = 5) (h2 : b = 18 - 3 * a) : a = 9 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l564_56483


namespace NUMINAMATH_CALUDE_carolyn_silverware_knife_percentage_l564_56476

/-- Represents the composition of a silverware set -/
structure Silverware :=
  (knives : ℕ)
  (forks : ℕ)
  (spoons : ℕ)

/-- Calculates the total number of pieces in a silverware set -/
def Silverware.total (s : Silverware) : ℕ :=
  s.knives + s.forks + s.spoons

/-- Represents a trade of silverware pieces -/
structure Trade :=
  (knives_gained : ℕ)
  (spoons_lost : ℕ)

/-- Applies a trade to a silverware set -/
def Silverware.apply_trade (s : Silverware) (t : Trade) : Silverware :=
  { knives := s.knives + t.knives_gained,
    forks := s.forks,
    spoons := s.spoons - t.spoons_lost }

/-- Calculates the percentage of knives in a silverware set -/
def Silverware.knife_percentage (s : Silverware) : ℚ :=
  (s.knives : ℚ) / (s.total : ℚ) * 100

theorem carolyn_silverware_knife_percentage :
  let initial_set : Silverware := { knives := 6, forks := 12, spoons := 6 * 3 }
  let trade : Trade := { knives_gained := 10, spoons_lost := 6 }
  let final_set := initial_set.apply_trade trade
  final_set.knife_percentage = 40 := by
  sorry

end NUMINAMATH_CALUDE_carolyn_silverware_knife_percentage_l564_56476


namespace NUMINAMATH_CALUDE_equivalence_condition_l564_56470

theorem equivalence_condition (x y : ℝ) (h : x * y ≠ 0) :
  (x + y = 0) ↔ (y / x + x / y = -2) := by
  sorry

end NUMINAMATH_CALUDE_equivalence_condition_l564_56470


namespace NUMINAMATH_CALUDE_clock_marks_and_polygon_l564_56421

-- Define the revolution time of the hour hand in minutes
def revolution_time : ℕ := 720

-- Define the interval between marks in minutes
def mark_interval : ℕ := 80

-- Define the number of distinct marks
def num_marks : ℕ := revolution_time / mark_interval

-- Define the sum of interior angles of a polygon
def sum_interior_angles (n : ℕ) : ℕ := 180 * (n - 2)

theorem clock_marks_and_polygon :
  (num_marks = 9) ∧ 
  (sum_interior_angles num_marks = 1260) := by
  sorry

end NUMINAMATH_CALUDE_clock_marks_and_polygon_l564_56421


namespace NUMINAMATH_CALUDE_maria_average_sales_l564_56461

/-- The average number of kilograms of apples sold per hour by Maria at the market -/
def average_apples_sold (first_hour_sales second_hour_sales : ℕ) (total_hours : ℕ) : ℚ :=
  (first_hour_sales + second_hour_sales : ℚ) / total_hours

/-- Theorem stating that Maria's average apple sales per hour is 6 kg/hour -/
theorem maria_average_sales :
  average_apples_sold 10 2 2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_maria_average_sales_l564_56461


namespace NUMINAMATH_CALUDE_daps_equivalent_to_dips_l564_56492

/-- Represents the conversion rate between daps and dops -/
def daps_to_dops : ℚ := 5 / 4

/-- Represents the conversion rate between dops and dips -/
def dops_to_dips : ℚ := 3 / 9

/-- The number of dips we want to convert -/
def target_dips : ℚ := 54

theorem daps_equivalent_to_dips :
  (daps_to_dops * (1 / dops_to_dips) * target_dips : ℚ) = 22.5 := by
  sorry

end NUMINAMATH_CALUDE_daps_equivalent_to_dips_l564_56492
