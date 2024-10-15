import Mathlib

namespace NUMINAMATH_CALUDE_matchmaking_theorem_l1796_179609

-- Define a bipartite graph
def BipartiteGraph (α : Type) := (α → Bool) → α → α → Prop

-- Define a matching in a bipartite graph
def Matching (α : Type) (G : BipartiteGraph α) (M : α → α → Prop) :=
  ∀ x y z, M x y → M x z → y = z

-- Define a perfect matching for a subset
def PerfectMatchingForSubset (α : Type) (G : BipartiteGraph α) (S : Set α) (M : α → α → Prop) :=
  Matching α G M ∧ ∀ x ∈ S, ∃ y, M x y

-- Main theorem
theorem matchmaking_theorem (α : Type) (G : BipartiteGraph α) 
  (B W : Set α) (B1 : Set α) (W2 : Set α) 
  (hB1 : B1 ⊆ B) (hW2 : W2 ⊆ W)
  (M1 : α → α → Prop) (M2 : α → α → Prop)
  (hM1 : PerfectMatchingForSubset α G B1 M1)
  (hM2 : PerfectMatchingForSubset α G W2 M2) :
  ∃ M : α → α → Prop, 
    Matching α G M ∧ 
    (∀ x y, M1 x y → M x y) ∧ 
    (∀ x y, M2 x y → M x y) :=
sorry

end NUMINAMATH_CALUDE_matchmaking_theorem_l1796_179609


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1796_179681

theorem sufficient_not_necessary_condition (x y : ℝ) :
  (∀ x y, x ≥ 2 ∧ y ≥ 2 → x^2 + y^2 ≥ 4) ∧
  (∃ x y, x^2 + y^2 ≥ 4 ∧ ¬(x ≥ 2 ∧ y ≥ 2)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1796_179681


namespace NUMINAMATH_CALUDE_abs_neg_two_thirds_eq_two_thirds_l1796_179602

theorem abs_neg_two_thirds_eq_two_thirds : |(-2 : ℚ) / 3| = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_abs_neg_two_thirds_eq_two_thirds_l1796_179602


namespace NUMINAMATH_CALUDE_candy_removal_time_l1796_179693

/-- Represents a position in a 3D grid -/
structure Position where
  x : Nat
  y : Nat
  z : Nat

/-- Calculates the layer sum for a given position -/
def layerSum (p : Position) : Nat :=
  p.x + p.y + p.z

/-- Represents the rectangular prism of candies -/
def CandyPrism :=
  {p : Position | p.x ≤ 3 ∧ p.y ≤ 4 ∧ p.z ≤ 5}

/-- Theorem stating that it takes 10 minutes to remove all candies -/
theorem candy_removal_time : 
  ∀ (start : Position), 
    start ∈ CandyPrism → 
    layerSum start = 3 → 
    (∀ (p : Position), p ∈ CandyPrism → layerSum p ≤ 12) →
    (∃ (p : Position), p ∈ CandyPrism ∧ layerSum p = 12) →
    10 = (12 - layerSum start + 1) :=
by sorry

end NUMINAMATH_CALUDE_candy_removal_time_l1796_179693


namespace NUMINAMATH_CALUDE_triangle_area_l1796_179665

/-- Given a triangle ABC with |AB| = 2, |AC| = 3, and AB · AC = -3, 
    prove that the area of triangle ABC is (3√3)/2 -/
theorem triangle_area (A B C : ℝ × ℝ) : 
  let AB := (B.1 - A.1, B.2 - A.2)
  let AC := (C.1 - A.1, C.2 - A.2)
  (AB.1^2 + AB.2^2 = 4) →
  (AC.1^2 + AC.2^2 = 9) →
  (AB.1 * AC.1 + AB.2 * AC.2 = -3) →
  (1/2 * Real.sqrt ((AB.1 * AC.2 - AB.2 * AC.1)^2) = (3 * Real.sqrt 3) / 2) :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_l1796_179665


namespace NUMINAMATH_CALUDE_equal_wins_losses_probability_l1796_179641

/-- Represents the result of a single match -/
inductive MatchResult
  | Win
  | Loss
  | Tie

/-- Probability distribution for match results -/
def matchProbability : MatchResult → Rat
  | MatchResult.Win => 1/4
  | MatchResult.Loss => 1/4
  | MatchResult.Tie => 1/2

/-- Total number of matches played -/
def totalMatches : Nat := 10

/-- Calculates the probability of having equal wins and losses in a season -/
def probabilityEqualWinsLosses : Rat :=
  63/262144

theorem equal_wins_losses_probability :
  probabilityEqualWinsLosses = 63/262144 := by
  sorry

#check equal_wins_losses_probability

end NUMINAMATH_CALUDE_equal_wins_losses_probability_l1796_179641


namespace NUMINAMATH_CALUDE_white_closed_under_add_mul_l1796_179600

/-- A color type representing black and white --/
inductive Color
| Black
| White

/-- A function that assigns a color to each positive integer --/
def coloring : ℕ+ → Color := sorry

/-- The property that the sum of two differently colored numbers is black --/
axiom sum_diff_color_black :
  ∀ (a b : ℕ+), coloring a ≠ coloring b → coloring (a + b) = Color.Black

/-- The property that there are infinitely many white numbers --/
axiom infinitely_many_white :
  ∀ (n : ℕ), ∃ (m : ℕ+), m > n ∧ coloring m = Color.White

/-- The theorem stating that the set of white numbers is closed under addition and multiplication --/
theorem white_closed_under_add_mul :
  ∀ (a b : ℕ+),
    coloring a = Color.White →
    coloring b = Color.White →
    coloring (a + b) = Color.White ∧ coloring (a * b) = Color.White :=
by sorry

end NUMINAMATH_CALUDE_white_closed_under_add_mul_l1796_179600


namespace NUMINAMATH_CALUDE_complex_power_pure_integer_l1796_179662

def i : ℂ := Complex.I

theorem complex_power_pure_integer :
  ∃ (n : ℤ), ∃ (m : ℤ), (3 * n + 2 * i) ^ 6 = m := by
  sorry

end NUMINAMATH_CALUDE_complex_power_pure_integer_l1796_179662


namespace NUMINAMATH_CALUDE_circle_center_sum_l1796_179608

/-- The equation of a circle in the xy-plane -/
def CircleEquation (x y : ℝ) : Prop :=
  x^2 + y^2 = 10*x - 4*y + 14

/-- The center of a circle given by its equation -/
def CircleCenter (x y : ℝ) : Prop :=
  CircleEquation x y ∧ ∀ a b : ℝ, CircleEquation a b → (a - x)^2 + (b - y)^2 ≤ (x - x)^2 + (y - y)^2

theorem circle_center_sum :
  ∀ x y : ℝ, CircleCenter x y → x + y = 3 := by sorry

end NUMINAMATH_CALUDE_circle_center_sum_l1796_179608


namespace NUMINAMATH_CALUDE_problem_solution_l1796_179650

theorem problem_solution : 
  let left_sum := 5 + 6 + 7 + 8 + 9
  let right_sum := 2005 + 2006 + 2007 + 2008 + 2009
  ∀ N : ℝ, (left_sum / 5 : ℝ) = (right_sum / N : ℝ) → N = 1433 :=
by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1796_179650


namespace NUMINAMATH_CALUDE_exists_periodic_product_l1796_179638

/-- A function f: ℝ → ℝ is periodic with period p if it's not constant and
    f(x) = f(x + p) for all x ∈ ℝ -/
def IsPeriodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  p > 0 ∧ (∃ x y, f x ≠ f y) ∧ ∀ x, f x = f (x + p)

/-- The period of a periodic function is the smallest positive p satisfying the periodicity condition -/
def Period (f : ℝ → ℝ) (p : ℝ) : Prop :=
  IsPeriodic f p ∧ ∀ q, 0 < q ∧ q < p → ¬ IsPeriodic f q

/-- Given any two positive real numbers a and b, there exist two periodic functions
    f₁ and f₂ with periods a and b respectively, such that their product f₁(x) · f₂(x)
    is also a periodic function -/
theorem exists_periodic_product (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  ∃ (f₁ f₂ : ℝ → ℝ), Period f₁ a ∧ Period f₂ b ∧
  ∃ p, p > 0 ∧ IsPeriodic (fun x ↦ f₁ x * f₂ x) p := by
  sorry

end NUMINAMATH_CALUDE_exists_periodic_product_l1796_179638


namespace NUMINAMATH_CALUDE_toy_football_sales_performance_toy_football_sales_performance_equality_l1796_179616

/-- Represents the sales performance of two students selling toy footballs --/
theorem toy_football_sales_performance
  (x y z : ℝ)  -- Prices of toy footballs in three sessions
  (hx : x > 0) (hy : y > 0) (hz : z > 0)  -- Prices are positive
  : (x + y + z) / 3 ≥ 3 / (1/x + 1/y + 1/z) := by
  sorry

/-- Equality condition for the sales performance --/
theorem toy_football_sales_performance_equality
  (x y z : ℝ)
  (hx : x > 0) (hy : y > 0) (hz : z > 0)
  : (x + y + z) / 3 = 3 / (1/x + 1/y + 1/z) ↔ x = y ∧ y = z := by
  sorry

end NUMINAMATH_CALUDE_toy_football_sales_performance_toy_football_sales_performance_equality_l1796_179616


namespace NUMINAMATH_CALUDE_division_problem_l1796_179659

theorem division_problem : (250 : ℝ) / (15 + 13 * 3 - 4) = 5 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l1796_179659


namespace NUMINAMATH_CALUDE_two_digit_squares_mod_15_l1796_179691

theorem two_digit_squares_mod_15 : ∃ (S : Finset Nat), (∀ a ∈ S, 10 ≤ a ∧ a < 100 ∧ a^2 % 15 = 1) ∧ S.card = 24 := by
  sorry

end NUMINAMATH_CALUDE_two_digit_squares_mod_15_l1796_179691


namespace NUMINAMATH_CALUDE_rectangle_width_l1796_179649

/-- Given a rectangle ABCD with length 25 yards and an inscribed rhombus AFCE with perimeter 82 yards, 
    the width of the rectangle is equal to √(420.25 / 2) yards. -/
theorem rectangle_width (length : ℝ) (perimeter : ℝ) (width : ℝ) : 
  length = 25 →
  perimeter = 82 →
  width = Real.sqrt (420.25 / 2) :=
by sorry

end NUMINAMATH_CALUDE_rectangle_width_l1796_179649


namespace NUMINAMATH_CALUDE_ellipse_x_intercept_l1796_179664

-- Define the ellipse
def ellipse (P : ℝ × ℝ) : Prop :=
  let F₁ := (0, 3)
  let F₂ := (4, 0)
  Real.sqrt ((P.1 - F₁.1)^2 + (P.2 - F₁.2)^2) +
  Real.sqrt ((P.1 - F₂.1)^2 + (P.2 - F₂.2)^2) = 8

-- Theorem statement
theorem ellipse_x_intercept :
  ellipse (0, 0) →
  ∃ x : ℝ, x ≠ 0 ∧ ellipse (x, 0) ∧ x = 45/8 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_x_intercept_l1796_179664


namespace NUMINAMATH_CALUDE_luther_clothing_line_l1796_179670

/-- The number of silk pieces in Luther's clothing line -/
def silk_pieces : ℕ := 7

/-- The number of cashmere pieces in Luther's clothing line -/
def cashmere_pieces : ℕ := silk_pieces / 2

/-- The number of blended pieces using both cashmere and silk -/
def blended_pieces : ℕ := 2

/-- The total number of pieces in Luther's clothing line -/
def total_pieces : ℕ := 13

theorem luther_clothing_line :
  silk_pieces + cashmere_pieces + blended_pieces = total_pieces ∧
  cashmere_pieces = silk_pieces / 2 ∧
  silk_pieces = 7 := by sorry

end NUMINAMATH_CALUDE_luther_clothing_line_l1796_179670


namespace NUMINAMATH_CALUDE_grocery_total_l1796_179644

/-- The number of cookie packs Lucy bought -/
def cookie_packs : ℕ := 23

/-- The number of cake packs Lucy bought -/
def cake_packs : ℕ := 4

/-- The total number of grocery packs Lucy bought -/
def total_packs : ℕ := cookie_packs + cake_packs

theorem grocery_total : total_packs = 27 := by
  sorry

end NUMINAMATH_CALUDE_grocery_total_l1796_179644


namespace NUMINAMATH_CALUDE_intersection_nonempty_implies_a_greater_than_neg_one_l1796_179660

open Set

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | -1 ≤ x ∧ x < 2}
def B (a : ℝ) : Set ℝ := {x : ℝ | x < a}

-- State the theorem
theorem intersection_nonempty_implies_a_greater_than_neg_one (a : ℝ) :
  (A ∩ B a).Nonempty → a > -1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_nonempty_implies_a_greater_than_neg_one_l1796_179660


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_special_set_l1796_179678

theorem arithmetic_mean_of_special_set (n : ℕ) (h : n > 1) : 
  let set := [1 - 1 / n, 1 + 1 / n] ++ List.replicate (n - 1) 1
  (List.sum set) / (n + 1) = 1 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_special_set_l1796_179678


namespace NUMINAMATH_CALUDE_catch_up_theorem_l1796_179604

/-- The distance traveled by both tourists when the second catches up to the first -/
def catch_up_distance : ℝ := 56

/-- The speed of the first tourist on bicycle in km/h -/
def speed_bicycle : ℝ := 16

/-- The speed of the second tourist on motorcycle in km/h -/
def speed_motorcycle : ℝ := 56

/-- The initial travel time of the first tourist before the break in hours -/
def initial_travel_time : ℝ := 1.5

/-- The break time of the first tourist in hours -/
def break_time : ℝ := 1.5

/-- The time delay between the start of the first and second tourist in hours -/
def start_delay : ℝ := 4

theorem catch_up_theorem :
  ∃ t : ℝ, t > 0 ∧
  speed_bicycle * (initial_travel_time + t) = 
  speed_motorcycle * t ∧
  catch_up_distance = speed_motorcycle * t :=
sorry

end NUMINAMATH_CALUDE_catch_up_theorem_l1796_179604


namespace NUMINAMATH_CALUDE_max_elevation_l1796_179651

/-- The elevation function of a particle projected vertically upward -/
def s (t : ℝ) : ℝ := 160 * t - 16 * t^2

/-- The maximum elevation reached by the particle -/
theorem max_elevation : ∃ (t : ℝ), ∀ (u : ℝ), s u ≤ s t ∧ s t = 400 := by
  sorry

end NUMINAMATH_CALUDE_max_elevation_l1796_179651


namespace NUMINAMATH_CALUDE_susie_piggy_bank_l1796_179697

theorem susie_piggy_bank (X : ℝ) : X + 0.2 * X = 240 → X = 200 := by
  sorry

end NUMINAMATH_CALUDE_susie_piggy_bank_l1796_179697


namespace NUMINAMATH_CALUDE_inverse_square_theorem_l1796_179610

-- Define the relationship between x and y
def inverse_square_relation (x y : ℝ) : Prop := ∃ k : ℝ, x = k / (y * y)

-- Define the theorem
theorem inverse_square_theorem :
  ∀ x y : ℝ,
  inverse_square_relation x y →
  (9 : ℝ) * (9 : ℝ) * (0.1111111111111111 : ℝ) = (3 : ℝ) * (3 : ℝ) * (1 : ℝ) →
  (x = (1 : ℝ) → y = (3 : ℝ)) :=
by sorry

end NUMINAMATH_CALUDE_inverse_square_theorem_l1796_179610


namespace NUMINAMATH_CALUDE_picasso_paintings_probability_l1796_179615

/-- The probability of placing 4 Picasso paintings consecutively among 12 art pieces -/
theorem picasso_paintings_probability (total_pieces : ℕ) (picasso_paintings : ℕ) :
  total_pieces = 12 →
  picasso_paintings = 4 →
  (picasso_paintings.factorial * (total_pieces - picasso_paintings + 1).factorial) / total_pieces.factorial = 1 / 55 :=
by sorry

end NUMINAMATH_CALUDE_picasso_paintings_probability_l1796_179615


namespace NUMINAMATH_CALUDE_student_assignment_l1796_179692

/-- The number of ways to assign n indistinguishable objects to k distinct containers,
    with each container receiving at least one object. -/
def assign_objects (n k : ℕ) : ℕ :=
  sorry

/-- The number of ways to assign 5 students to 3 towns -/
theorem student_assignment : assign_objects 5 3 = 150 := by
  sorry

end NUMINAMATH_CALUDE_student_assignment_l1796_179692


namespace NUMINAMATH_CALUDE_equation_solutions_l1796_179633

theorem equation_solutions :
  ∀ a b c : ℕ+,
  (1 : ℚ) / a + (2 : ℚ) / b - (3 : ℚ) / c = 1 ↔
  (∃ n : ℕ+, a = 1 ∧ b = 2 * n ∧ c = 3 * n) ∨
  (a = 2 ∧ b = 1 ∧ c = 2) ∨
  (a = 2 ∧ b = 3 ∧ c = 18) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l1796_179633


namespace NUMINAMATH_CALUDE_triangle_probability_l1796_179667

theorem triangle_probability (total_figures : ℕ) (triangle_count : ℕ) 
  (h1 : total_figures = 8) (h2 : triangle_count = 3) :
  (triangle_count : ℚ) / total_figures = 3 / 8 := by
sorry

end NUMINAMATH_CALUDE_triangle_probability_l1796_179667


namespace NUMINAMATH_CALUDE_reciprocal_of_five_l1796_179669

theorem reciprocal_of_five (x : ℚ) : x = 5 → (1 : ℚ) / x = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_of_five_l1796_179669


namespace NUMINAMATH_CALUDE_smallest_solution_quadratic_equation_l1796_179695

theorem smallest_solution_quadratic_equation :
  let f : ℝ → ℝ := λ x => 10 * x^2 - 66 * x + 56
  ∃ (x : ℝ), f x = 0 ∧ (∀ y : ℝ, f y = 0 → x ≤ y) ∧ x = 8/5 := by
  sorry

end NUMINAMATH_CALUDE_smallest_solution_quadratic_equation_l1796_179695


namespace NUMINAMATH_CALUDE_max_table_sum_l1796_179645

def numbers : List ℕ := [2, 3, 5, 7, 11, 13, 17]

def is_valid_partition (l1 l2 : List ℕ) : Prop :=
  l1.length = 4 ∧ l2.length = 3 ∧ (l1 ++ l2).toFinset = numbers.toFinset

def table_sum (l1 l2 : List ℕ) : ℕ := (l1.sum * l2.sum)

theorem max_table_sum :
  ∃ (l1 l2 : List ℕ), is_valid_partition l1 l2 ∧
    (∀ (m1 m2 : List ℕ), is_valid_partition m1 m2 →
      table_sum m1 m2 ≤ table_sum l1 l2) ∧
    table_sum l1 l2 = 841 := by sorry

end NUMINAMATH_CALUDE_max_table_sum_l1796_179645


namespace NUMINAMATH_CALUDE_octal_minus_septenary_in_decimal_l1796_179620

/-- Converts a number from base b to base 10 -/
def to_base_10 (digits : List Nat) (b : Nat) : Nat :=
  digits.foldr (fun d acc => d + b * acc) 0

/-- The problem statement -/
theorem octal_minus_septenary_in_decimal : 
  let octal := [2, 1, 3]
  let septenary := [1, 4, 2]
  to_base_10 octal 8 - to_base_10 septenary 7 = 60 := by
  sorry


end NUMINAMATH_CALUDE_octal_minus_septenary_in_decimal_l1796_179620


namespace NUMINAMATH_CALUDE_consecutive_even_integers_square_product_l1796_179668

theorem consecutive_even_integers_square_product : 
  ∀ (a b c : ℤ),
  (b = a + 2 ∧ c = b + 2) →  -- consecutive even integers
  (a * b * c = 12 * (a + b + c)) →  -- product is 12 times their sum
  (a^2 * b^2 * c^2 = 36864) :=  -- product of squares is 36864
by
  sorry

end NUMINAMATH_CALUDE_consecutive_even_integers_square_product_l1796_179668


namespace NUMINAMATH_CALUDE_no_solution_implies_a_less_than_two_l1796_179635

theorem no_solution_implies_a_less_than_two (a : ℝ) :
  (∀ x : ℝ, |x - 3| + |x - 1| > a) → a < 2 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_implies_a_less_than_two_l1796_179635


namespace NUMINAMATH_CALUDE_dice_prob_same_color_l1796_179619

def prob_same_color (d1_sides d2_sides : ℕ)
  (d1_maroon d1_teal d1_cyan d1_sparkly : ℕ)
  (d2_maroon d2_teal d2_cyan d2_sparkly : ℕ) : ℚ :=
  let p_maroon := (d1_maroon : ℚ) / d1_sides * (d2_maroon : ℚ) / d2_sides
  let p_teal := (d1_teal : ℚ) / d1_sides * (d2_teal : ℚ) / d2_sides
  let p_cyan := (d1_cyan : ℚ) / d1_sides * (d2_cyan : ℚ) / d2_sides
  let p_sparkly := (d1_sparkly : ℚ) / d1_sides * (d2_sparkly : ℚ) / d2_sides
  p_maroon + p_teal + p_cyan + p_sparkly

theorem dice_prob_same_color :
  prob_same_color 20 16 5 8 6 1 4 6 5 1 = 99 / 320 := by
  sorry

end NUMINAMATH_CALUDE_dice_prob_same_color_l1796_179619


namespace NUMINAMATH_CALUDE_distributive_property_l1796_179617

theorem distributive_property (a : ℝ) : 2 * (a - 1) = 2 * a - 2 := by
  sorry

end NUMINAMATH_CALUDE_distributive_property_l1796_179617


namespace NUMINAMATH_CALUDE_sum_of_angles_satisfying_equation_l1796_179684

theorem sum_of_angles_satisfying_equation (x : Real) : 
  (0 ≤ x ∧ x ≤ 2 * Real.pi) →
  (Real.sin x ^ 3 + Real.cos x ^ 3 = 1 / Real.cos x + 1 / Real.sin x) →
  ∃ (y : Real), (0 ≤ y ∧ y ≤ 2 * Real.pi) ∧
    (Real.sin y ^ 3 + Real.cos y ^ 3 = 1 / Real.cos y + 1 / Real.sin y) ∧
    (x + y = 3 * Real.pi / 2) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_angles_satisfying_equation_l1796_179684


namespace NUMINAMATH_CALUDE_problem_solution_l1796_179686

theorem problem_solution : 
  (Real.sqrt 48 / Real.sqrt 3 - 2 * Real.sqrt (1/5) * Real.sqrt 30 + Real.sqrt 24 = 4) ∧
  ((2 * Real.sqrt 3 - 1)^2 + (Real.sqrt 3 + 2) * (Real.sqrt 3 - 2) = 12 - 4 * Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1796_179686


namespace NUMINAMATH_CALUDE_four_numbers_problem_l1796_179632

theorem four_numbers_problem (a b c d : ℝ) : 
  a + b + c + d = 45 ∧ 
  a + 2 = b - 2 ∧ 
  a + 2 = 2 * c ∧ 
  a + 2 = d / 2 → 
  a = 8 ∧ b = 12 ∧ c = 5 ∧ d = 20 := by
sorry

end NUMINAMATH_CALUDE_four_numbers_problem_l1796_179632


namespace NUMINAMATH_CALUDE_schedule_arrangements_eq_192_l1796_179625

/-- The number of ways to arrange 6 distinct lessons into 6 time slots -/
def schedule_arrangements (total_lessons : ℕ) (morning_slots : ℕ) (afternoon_slots : ℕ) 
  (morning_constraint : ℕ) (afternoon_constraint : ℕ) : ℕ := 
  (morning_slots.choose morning_constraint) * 
  (afternoon_slots.choose afternoon_constraint) * 
  (Nat.factorial (total_lessons - morning_constraint - afternoon_constraint))

/-- Theorem stating that the number of schedule arrangements is 192 -/
theorem schedule_arrangements_eq_192 : 
  schedule_arrangements 6 4 2 1 1 = 192 := by
  sorry

end NUMINAMATH_CALUDE_schedule_arrangements_eq_192_l1796_179625


namespace NUMINAMATH_CALUDE_cat_weight_sum_l1796_179688

/-- The combined weight of three cats -/
def combined_weight (w1 w2 w3 : ℕ) : ℕ := w1 + w2 + w3

/-- Theorem: The combined weight of three cats weighing 2, 7, and 4 pounds is 13 pounds -/
theorem cat_weight_sum : combined_weight 2 7 4 = 13 := by
  sorry

end NUMINAMATH_CALUDE_cat_weight_sum_l1796_179688


namespace NUMINAMATH_CALUDE_solution_set_theorem_l1796_179696

def inequality_system (x : ℝ) : Prop :=
  (4 * x + 1 > 2) ∧ (1 - 2 * x < 7)

theorem solution_set_theorem :
  {x : ℝ | inequality_system x} = {x : ℝ | x > 1/4} := by sorry

end NUMINAMATH_CALUDE_solution_set_theorem_l1796_179696


namespace NUMINAMATH_CALUDE_tv_show_minor_characters_l1796_179634

/-- The problem of determining the number of minor characters in a TV show. -/
theorem tv_show_minor_characters :
  let main_characters : ℕ := 5
  let minor_character_pay : ℕ := 15000
  let main_character_pay : ℕ := 3 * minor_character_pay
  let total_pay : ℕ := 285000
  let minor_characters : ℕ := (total_pay - main_characters * main_character_pay) / minor_character_pay
  minor_characters = 4 := by sorry

end NUMINAMATH_CALUDE_tv_show_minor_characters_l1796_179634


namespace NUMINAMATH_CALUDE_digit_at_position_l1796_179690

/-- The fraction we're examining -/
def f : ℚ := 17 / 270

/-- The length of the repeating sequence in the decimal representation of f -/
def period : ℕ := 3

/-- The repeating sequence in the decimal representation of f -/
def repeating_sequence : List ℕ := [6, 2, 9]

/-- The position we're interested in -/
def target_position : ℕ := 145

theorem digit_at_position :
  (target_position - 1) % period = 0 →
  List.get! repeating_sequence (period - 1) = 9 := by
  sorry

end NUMINAMATH_CALUDE_digit_at_position_l1796_179690


namespace NUMINAMATH_CALUDE_unique_numbers_proof_l1796_179689

theorem unique_numbers_proof (a b : ℕ) : 
  a ≠ b →                 -- The numbers are distinct
  a > 11 →                -- a is greater than 11
  b > 11 →                -- b is greater than 11
  a + b = 28 →            -- Their sum is 28
  (Even a ∨ Even b) →     -- At least one of them is even
  ((a = 12 ∧ b = 16) ∨ (a = 16 ∧ b = 12)) := by
sorry

end NUMINAMATH_CALUDE_unique_numbers_proof_l1796_179689


namespace NUMINAMATH_CALUDE_university_theater_tickets_l1796_179661

/-- The total number of tickets sold at University Theater -/
def total_tickets (adult_price senior_price : ℕ) (total_receipts : ℕ) (senior_tickets : ℕ) : ℕ :=
  senior_tickets + ((total_receipts - senior_price * senior_tickets) / adult_price)

/-- Theorem stating that the total number of tickets sold is 509 -/
theorem university_theater_tickets :
  total_tickets 21 15 8748 327 = 509 := by
  sorry

end NUMINAMATH_CALUDE_university_theater_tickets_l1796_179661


namespace NUMINAMATH_CALUDE_max_distance_with_tire_swap_l1796_179631

/-- Represents the maximum distance a car can travel with tire swapping -/
def maxDistanceWithTireSwap (frontTireLife : ℕ) (rearTireLife : ℕ) : ℕ :=
  frontTireLife + (rearTireLife - frontTireLife)

/-- Theorem: The maximum distance a car can travel with given tire lifespans -/
theorem max_distance_with_tire_swap :
  maxDistanceWithTireSwap 20000 30000 = 30000 := by
  sorry

#eval maxDistanceWithTireSwap 20000 30000

end NUMINAMATH_CALUDE_max_distance_with_tire_swap_l1796_179631


namespace NUMINAMATH_CALUDE_h_equality_l1796_179680

-- Define the functions f and g
def f (x : ℝ) : ℝ := x^2 - x + 1
def g (x : ℝ) : ℝ := -x^2 + x + 1

-- Define the polynomial h
def h (x : ℝ) : ℝ := (x - 1)^2

-- Theorem statement
theorem h_equality (x : ℝ) : h (f x) = h (g x) := by
  sorry

end NUMINAMATH_CALUDE_h_equality_l1796_179680


namespace NUMINAMATH_CALUDE_percentage_problem_l1796_179672

theorem percentage_problem (P : ℝ) (N : ℝ) 
  (h1 : (P / 100) * N = 200)
  (h2 : 1.2 * N = 1200) : P = 20 := by
sorry

end NUMINAMATH_CALUDE_percentage_problem_l1796_179672


namespace NUMINAMATH_CALUDE_point_sum_on_reciprocal_function_l1796_179654

theorem point_sum_on_reciprocal_function (p q : ℝ → ℝ) (h1 : p 4 = 8) (h2 : ∀ x, q x = 1 / p x) :
  4 + q 4 = 33 / 8 := by
  sorry

end NUMINAMATH_CALUDE_point_sum_on_reciprocal_function_l1796_179654


namespace NUMINAMATH_CALUDE_parakeets_per_cage_l1796_179674

theorem parakeets_per_cage 
  (num_cages : ℕ) 
  (parrots_per_cage : ℕ) 
  (total_birds : ℕ) 
  (h1 : num_cages = 8)
  (h2 : parrots_per_cage = 2)
  (h3 : total_birds = 72) :
  (total_birds - num_cages * parrots_per_cage) / num_cages = 7 := by
  sorry

end NUMINAMATH_CALUDE_parakeets_per_cage_l1796_179674


namespace NUMINAMATH_CALUDE_triangle_angle_proof_l1796_179621

theorem triangle_angle_proof (a b c A B C : ℝ) (S_ABC : ℝ) : 
  b = 2 →
  S_ABC = 2 * Real.sqrt 3 →
  c * Real.cos B + b * Real.cos C - 2 * a * Real.cos A = 0 →
  0 < A → A < π →
  0 < B → B < π →
  0 < C → C < π →
  S_ABC = (1 / 2) * a * b * Real.sin C →
  S_ABC = (1 / 2) * b * c * Real.sin A →
  S_ABC = (1 / 2) * c * a * Real.sin B →
  a^2 = b^2 + c^2 - 2 * b * c * Real.cos A →
  b^2 = a^2 + c^2 - 2 * a * c * Real.cos B →
  c^2 = a^2 + b^2 - 2 * a * b * Real.cos C →
  C = π / 2 := by sorry

end NUMINAMATH_CALUDE_triangle_angle_proof_l1796_179621


namespace NUMINAMATH_CALUDE_y_investment_calculation_l1796_179673

/-- Represents the investment and profit sharing of two business partners -/
structure BusinessPartnership where
  /-- The amount invested by partner X -/
  x_investment : ℕ
  /-- The amount invested by partner Y -/
  y_investment : ℕ
  /-- The profit share ratio of partner X -/
  x_profit_ratio : ℕ
  /-- The profit share ratio of partner Y -/
  y_profit_ratio : ℕ

/-- Theorem stating that if the profit is shared in ratio 2:6 and X invested 5000, then Y invested 15000 -/
theorem y_investment_calculation (bp : BusinessPartnership) 
  (h1 : bp.x_investment = 5000)
  (h2 : bp.x_profit_ratio = 2)
  (h3 : bp.y_profit_ratio = 6) :
  bp.y_investment = 15000 := by
  sorry


end NUMINAMATH_CALUDE_y_investment_calculation_l1796_179673


namespace NUMINAMATH_CALUDE_gcd_lcm_sum_l1796_179685

theorem gcd_lcm_sum : Nat.gcd 45 75 + Nat.lcm 40 10 = 55 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_sum_l1796_179685


namespace NUMINAMATH_CALUDE_profit_starts_third_year_max_average_profit_at_six_option_i_more_cost_effective_l1796_179637

-- Define f(n) in ten thousand yuan
def f (n : ℕ) : ℤ := -2*n^2 + 40*n - 72

-- Question 1: Prove that the factory starts to make a profit from the third year
theorem profit_starts_third_year : 
  ∀ n : ℕ, n > 0 → (f n > 0 ↔ n ≥ 3) :=
sorry

-- Question 2: Prove that the annual average net profit reaches its maximum when n = 6
theorem max_average_profit_at_six :
  ∀ n : ℕ, n > 0 → f n / n ≤ f 6 / 6 :=
sorry

-- Question 3: Prove that option (i) is more cost-effective
theorem option_i_more_cost_effective :
  f 6 + 48 > f 10 + 10 :=
sorry

end NUMINAMATH_CALUDE_profit_starts_third_year_max_average_profit_at_six_option_i_more_cost_effective_l1796_179637


namespace NUMINAMATH_CALUDE_triangle_inequality_l1796_179627

theorem triangle_inequality (a b c : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 → a + b > c ∧ b + c > a ∧ c + a > b →
  ¬(a = 3 ∧ b = 5 ∧ c = 2) := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l1796_179627


namespace NUMINAMATH_CALUDE_related_chord_midpoint_x_max_related_chord_length_l1796_179694

/-- Represents a point on a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola y^2 = 4x -/
def Parabola : Set Point :=
  {p : Point | p.y^2 = 4 * p.x}

/-- Checks if a chord AB is a "related chord" of point P -/
def isRelatedChord (A B P : Point) : Prop :=
  A ∈ Parabola ∧ B ∈ Parabola ∧ A ≠ B ∧
  P.y = 0 ∧
  ∃ (M : Point), M.x = (A.x + B.x) / 2 ∧ M.y = (A.y + B.y) / 2 ∧
  (M.y - P.y) * (A.x - B.x) = (P.x - M.x) * (A.y - B.y)

/-- The x-coordinate of the midpoint of any "related chord" of P(4,0) is 2 -/
theorem related_chord_midpoint_x (A B : Point) :
  isRelatedChord A B (Point.mk 4 0) → (A.x + B.x) / 2 = 2 := by sorry

/-- The maximum length of all "related chords" of P(4,0) is 6 -/
theorem max_related_chord_length (A B : Point) :
  isRelatedChord A B (Point.mk 4 0) →
  ∃ (max_length : ℝ), max_length = 6 ∧
  ∀ (C D : Point), isRelatedChord C D (Point.mk 4 0) →
  ((C.x - D.x)^2 + (C.y - D.y)^2)^(1/2) ≤ max_length := by sorry

end NUMINAMATH_CALUDE_related_chord_midpoint_x_max_related_chord_length_l1796_179694


namespace NUMINAMATH_CALUDE_spider_human_leg_ratio_l1796_179655

/-- The number of legs a spider has -/
def spider_legs : ℕ := 8

/-- The number of legs a human has -/
def human_legs : ℕ := 2

/-- The ratio of spider legs to human legs -/
def leg_ratio : ℚ := spider_legs / human_legs

/-- Theorem: The ratio of spider legs to human legs is 4 -/
theorem spider_human_leg_ratio : leg_ratio = 4 := by
  sorry

end NUMINAMATH_CALUDE_spider_human_leg_ratio_l1796_179655


namespace NUMINAMATH_CALUDE_expand_and_simplify_l1796_179658

theorem expand_and_simplify (x : ℝ) : (2 * x + 6) * (x + 9) = 2 * x^2 + 24 * x + 54 := by
  sorry

end NUMINAMATH_CALUDE_expand_and_simplify_l1796_179658


namespace NUMINAMATH_CALUDE_max_pie_pieces_l1796_179683

/-- Represents a five-digit number with distinct digits -/
def DistinctFiveDigitNumber (x : ℕ) : Prop :=
  10000 ≤ x ∧ x < 100000 ∧ (∀ i j, i ≠ j → (x / 10^i) % 10 ≠ (x / 10^j) % 10)

/-- The maximum number of pieces that can be obtained when dividing a pie -/
theorem max_pie_pieces :
  ∃ (n : ℕ) (pie piece : ℕ),
    n = 7 ∧
    DistinctFiveDigitNumber pie ∧
    10000 ≤ piece ∧ piece < 100000 ∧
    pie = piece * n ∧
    (∀ m > n, ¬∃ (p q : ℕ),
      DistinctFiveDigitNumber p ∧
      10000 ≤ q ∧ q < 100000 ∧
      p = q * m) :=
by sorry

end NUMINAMATH_CALUDE_max_pie_pieces_l1796_179683


namespace NUMINAMATH_CALUDE_p_necessary_not_sufficient_for_q_l1796_179656

theorem p_necessary_not_sufficient_for_q :
  (∀ x : ℝ, x > 3 → x > 2) ∧
  (∃ x : ℝ, x > 2 ∧ x ≤ 3) := by
  sorry

end NUMINAMATH_CALUDE_p_necessary_not_sufficient_for_q_l1796_179656


namespace NUMINAMATH_CALUDE_max_area_MPNQ_l1796_179613

noncomputable section

-- Define the curves C₁ and C₂ in polar coordinates
def C₁ (θ : Real) : Real := 2 * Real.sqrt 2

def C₂ (θ : Real) : Real := 4 * Real.sqrt 2 * (Real.cos θ + Real.sin θ)

-- Define the area of quadrilateral MPNQ as a function of α
def area_MPNQ (α : Real) : Real :=
  4 * Real.sqrt 2 * Real.sin (2 * α + Real.pi / 4) + 4 - 2 * Real.sqrt 2

-- Theorem statement
theorem max_area_MPNQ :
  ∃ α, 0 < α ∧ α < Real.pi / 2 ∧
  ∀ β, 0 < β → β < Real.pi / 2 →
  area_MPNQ β ≤ area_MPNQ α ∧
  area_MPNQ α = 4 + 2 * Real.sqrt 2 :=
sorry

end

end NUMINAMATH_CALUDE_max_area_MPNQ_l1796_179613


namespace NUMINAMATH_CALUDE_line_equation_through_point_with_slope_point_satisfies_equation_l1796_179612

/-- Proves that the equation of a line passing through the point (1, 2) with a slope of 3 is y = 3x - 1 -/
theorem line_equation_through_point_with_slope (x y : ℝ) : 
  (y - 2 = 3 * (x - 1)) ↔ (y = 3 * x - 1) := by
  sorry

/-- Verifies that the point (1, 2) satisfies the equation y = 3x - 1 -/
theorem point_satisfies_equation : 
  2 = 3 * 1 - 1 := by
  sorry

end NUMINAMATH_CALUDE_line_equation_through_point_with_slope_point_satisfies_equation_l1796_179612


namespace NUMINAMATH_CALUDE_soda_price_ratio_l1796_179629

theorem soda_price_ratio (v p : ℝ) (hv : v > 0) (hp : p > 0) : 
  let brand_x_volume := 1.3 * v
  let brand_x_price := 0.85 * p
  (brand_x_price / brand_x_volume) / (p / v) = 17 / 26 := by
sorry

end NUMINAMATH_CALUDE_soda_price_ratio_l1796_179629


namespace NUMINAMATH_CALUDE_path_length_for_73_l1796_179643

/-- The length of a path around squares constructed on segments of a line -/
def path_length (segment_length : ℝ) : ℝ :=
  3 * segment_length

theorem path_length_for_73 :
  path_length 73 = 219 := by sorry

end NUMINAMATH_CALUDE_path_length_for_73_l1796_179643


namespace NUMINAMATH_CALUDE_quadratic_coefficients_l1796_179699

/-- A quadratic function with vertex (h, k) and y-intercept c can be represented as f(x) = a(x - h)² + k,
    where a ≠ 0 and f(0) = c. -/
def quadratic_function (a h k c : ℝ) (ha : a ≠ 0) (f : ℝ → ℝ) :=
  ∀ x, f x = a * (x - h)^2 + k ∧ f 0 = c

theorem quadratic_coefficients (f : ℝ → ℝ) (a h k c : ℝ) (ha : a ≠ 0) :
  quadratic_function a h k c ha f →
  h = 2 ∧ k = -1 ∧ c = 11 →
  a = 3 ∧ 
  ∃ b, ∀ x, f x = 3 * x^2 + b * x + 11 ∧ b = -12 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_coefficients_l1796_179699


namespace NUMINAMATH_CALUDE_remainder_sum_of_powers_l1796_179636

theorem remainder_sum_of_powers (n : ℕ) : (9^6 + 8^8 + 7^9) % 7 = 2 := by
  sorry

end NUMINAMATH_CALUDE_remainder_sum_of_powers_l1796_179636


namespace NUMINAMATH_CALUDE_iris_count_after_addition_l1796_179606

/-- Calculates the number of irises needed to maintain a ratio of 3:7 with roses -/
def calculate_irises (initial_roses : ℕ) (added_roses : ℕ) : ℕ :=
  let total_roses := initial_roses + added_roses
  let irises := (3 * total_roses) / 7
  irises

theorem iris_count_after_addition 
  (initial_roses : ℕ) 
  (added_roses : ℕ) 
  (h1 : initial_roses = 35) 
  (h2 : added_roses = 25) : 
  calculate_irises initial_roses added_roses = 25 := by
sorry

#eval calculate_irises 35 25

end NUMINAMATH_CALUDE_iris_count_after_addition_l1796_179606


namespace NUMINAMATH_CALUDE_quick_calculation_formula_l1796_179652

theorem quick_calculation_formula (a b : ℝ) :
  (100 + a) * (100 + b) = ((100 + a) + (100 + b) - 100) * 100 + a * b ∧
  (100 + a) * (100 - b) = ((100 + a) + (100 - b) - 100) * 100 + a * (-b) ∧
  (100 - a) * (100 + b) = ((100 - a) + (100 + b) - 100) * 100 + (-a) * b ∧
  (100 - a) * (100 - b) = ((100 - a) + (100 - b) - 100) * 100 + (-a) * (-b) :=
by sorry

end NUMINAMATH_CALUDE_quick_calculation_formula_l1796_179652


namespace NUMINAMATH_CALUDE_lcm_of_numbers_with_given_hcf_and_product_l1796_179676

theorem lcm_of_numbers_with_given_hcf_and_product :
  ∀ a b : ℕ+,
  (Nat.gcd a.val b.val = 11) →
  (a * b = 1991) →
  (Nat.lcm a.val b.val = 181) :=
by
  sorry

end NUMINAMATH_CALUDE_lcm_of_numbers_with_given_hcf_and_product_l1796_179676


namespace NUMINAMATH_CALUDE_units_digit_of_2749_pow_987_l1796_179624

theorem units_digit_of_2749_pow_987 :
  (2749^987) % 10 = 9 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_2749_pow_987_l1796_179624


namespace NUMINAMATH_CALUDE_white_square_area_l1796_179671

/-- Given a cube with edge length 12 feet and 432 square feet of green paint used equally on all faces as a border, the area of the white square centered on each face is 72 square feet. -/
theorem white_square_area (cube_edge : ℝ) (green_paint_area : ℝ) (white_square_area : ℝ) : 
  cube_edge = 12 →
  green_paint_area = 432 →
  white_square_area = 72 →
  white_square_area = cube_edge^2 - green_paint_area / 6 :=
by sorry

end NUMINAMATH_CALUDE_white_square_area_l1796_179671


namespace NUMINAMATH_CALUDE_segment_count_after_16_iterations_l1796_179630

/-- The number of segments after n iterations of the division process -/
def num_segments (n : ℕ) : ℕ := 2^n

/-- The length of each segment after n iterations of the division process -/
def segment_length (n : ℕ) : ℚ := (1 : ℚ) / 3^n

theorem segment_count_after_16_iterations :
  num_segments 16 = 2^16 := by sorry

end NUMINAMATH_CALUDE_segment_count_after_16_iterations_l1796_179630


namespace NUMINAMATH_CALUDE_locus_of_Q_is_ellipse_l1796_179628

/-- The ellipse C -/
def C (x y : ℝ) : Prop := x^2 / 24 + y^2 / 16 = 1

/-- The line l -/
def l (x y : ℝ) : Prop := x / 12 + y / 8 = 1

/-- Point R is on ellipse C -/
def R_on_C (xR yR : ℝ) : Prop := C xR yR

/-- Point P is on line l -/
def P_on_l (xP yP : ℝ) : Prop := l xP yP

/-- Q is on OP and satisfies |OQ| * |OP| = |OR|² -/
def Q_condition (xQ yQ xP yP xR yR : ℝ) : Prop :=
  ∃ (t : ℝ), 0 < t ∧ t < 1 ∧ xQ = t * xP ∧ yQ = t * yP ∧
  t * (xP^2 + yP^2) = xR^2 + yR^2

/-- The resulting ellipse for Q -/
def Q_ellipse (x y : ℝ) : Prop := (x - 1)^2 / (5/2) + (y - 1)^2 / (5/3) = 1

/-- Main theorem: The locus of Q is the ellipse (x-1)²/(5/2) + (y-1)²/(5/3) = 1 -/
theorem locus_of_Q_is_ellipse :
  ∀ (xQ yQ xP yP xR yR : ℝ),
    P_on_l xP yP →
    R_on_C xR yR →
    Q_condition xQ yQ xP yP xR yR →
    Q_ellipse xQ yQ :=
sorry

end NUMINAMATH_CALUDE_locus_of_Q_is_ellipse_l1796_179628


namespace NUMINAMATH_CALUDE_exactly_three_valid_pairs_l1796_179611

/-- The interior angle of a regular polygon with n sides -/
def interior_angle (n : ℕ) : ℚ := 180 - (360 / n)

/-- Predicate for valid pairs of regular polygons -/
def valid_pair (k r : ℕ) : Prop :=
  k > 2 ∧ r > 2 ∧ (interior_angle r) / (interior_angle k) = 4 / 3

/-- The number of valid pairs of regular polygons -/
def num_valid_pairs : ℕ := 3

/-- Theorem stating that there are exactly 3 valid pairs -/
theorem exactly_three_valid_pairs :
  ∃! (s : Finset (ℕ × ℕ)), s.card = num_valid_pairs ∧ 
  (∀ (k r : ℕ), (k, r) ∈ s ↔ valid_pair k r) :=
sorry

end NUMINAMATH_CALUDE_exactly_three_valid_pairs_l1796_179611


namespace NUMINAMATH_CALUDE_megan_files_added_l1796_179601

theorem megan_files_added 
  (initial_files : ℝ) 
  (files_per_folder : ℝ) 
  (num_folders : ℝ) 
  (h1 : initial_files = 93.0) 
  (h2 : files_per_folder = 8.0) 
  (h3 : num_folders = 14.25) : 
  num_folders * files_per_folder - initial_files = 21.0 := by
sorry

end NUMINAMATH_CALUDE_megan_files_added_l1796_179601


namespace NUMINAMATH_CALUDE_matrix_equality_l1796_179640

theorem matrix_equality (A B : Matrix (Fin 2) (Fin 2) ℝ) 
  (h1 : A + B = A * B) 
  (h2 : A * B = !![2, 1; 4, 3]) : 
  B * A = !![2, 1; 4, 3] := by sorry

end NUMINAMATH_CALUDE_matrix_equality_l1796_179640


namespace NUMINAMATH_CALUDE_volume_of_intersected_prism_l1796_179622

/-- The volume of a solid formed by the intersection of a plane with a prism -/
theorem volume_of_intersected_prism (a : ℝ) (h : ℝ) :
  let prism_base_area : ℝ := (a^2 * Real.sqrt 3) / 2
  let prism_volume : ℝ := prism_base_area * h
  let intersection_volume : ℝ := (77 * Real.sqrt 3) / 54
  (h = 2) →
  (intersection_volume < prism_volume) →
  (intersection_volume > 0) →
  intersection_volume = (77 * Real.sqrt 3) / 54 :=
by sorry

end NUMINAMATH_CALUDE_volume_of_intersected_prism_l1796_179622


namespace NUMINAMATH_CALUDE_math_score_calculation_l1796_179618

theorem math_score_calculation (initial_average : ℝ) (num_initial_subjects : ℕ) (average_drop : ℝ) :
  initial_average = 95 →
  num_initial_subjects = 3 →
  average_drop = 3 →
  let total_initial_score := initial_average * num_initial_subjects
  let new_average := initial_average - average_drop
  let new_total_score := new_average * (num_initial_subjects + 1)
  new_total_score - total_initial_score = 83 := by
  sorry

end NUMINAMATH_CALUDE_math_score_calculation_l1796_179618


namespace NUMINAMATH_CALUDE_intercepts_congruence_l1796_179653

/-- Proof of x-intercept and y-intercept properties for the congruence 6x ≡ 5y - 1 (mod 28) --/
theorem intercepts_congruence :
  ∃ (x₀ y₀ : ℕ),
    x₀ < 28 ∧ y₀ < 28 ∧
    (6 * x₀) % 28 = 27 ∧
    (5 * y₀) % 28 = 1 ∧
    x₀ + y₀ = 20 := by
  sorry


end NUMINAMATH_CALUDE_intercepts_congruence_l1796_179653


namespace NUMINAMATH_CALUDE_time_to_park_l1796_179605

/-- Represents the jogging scenario with constant pace -/
structure JoggingScenario where
  pace : ℝ  -- Jogging pace in minutes per mile
  cafe_distance : ℝ  -- Distance to café in miles
  cafe_time : ℝ  -- Time to jog to café in minutes
  park_distance : ℝ  -- Distance to park in miles

/-- Given a jogging scenario with constant pace, proves that the time to jog to the park is 36 minutes -/
theorem time_to_park (scenario : JoggingScenario)
  (h1 : scenario.cafe_distance = 3)
  (h2 : scenario.cafe_time = 24)
  (h3 : scenario.park_distance = 4.5)
  (h4 : scenario.pace > 0) :
  scenario.pace * scenario.park_distance = 36 := by
  sorry

#check time_to_park

end NUMINAMATH_CALUDE_time_to_park_l1796_179605


namespace NUMINAMATH_CALUDE_students_walking_to_school_l1796_179607

theorem students_walking_to_school 
  (total_students : ℕ) 
  (walking_minus_public : ℕ) 
  (h1 : total_students = 41)
  (h2 : walking_minus_public = 3) :
  let walking := (total_students + walking_minus_public) / 2
  walking = 22 := by
  sorry

end NUMINAMATH_CALUDE_students_walking_to_school_l1796_179607


namespace NUMINAMATH_CALUDE_hyperbola_equation_l1796_179687

-- Define the general form of a hyperbola
def hyperbola (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1

-- Define the given hyperbola with known asymptotes
def given_hyperbola (x y : ℝ) : Prop :=
  x^2 / 8 - y^2 = 1

-- Define the given ellipse with known foci
def given_ellipse (x y : ℝ) : Prop :=
  x^2 / 20 + y^2 / 2 = 1

-- Theorem stating the equation of the hyperbola
theorem hyperbola_equation :
  ∃ (a b : ℝ),
    (∀ (x y : ℝ), hyperbola a b x y ↔ given_hyperbola x y) ∧
    (∀ (x : ℝ), x^2 = 18 → hyperbola a b x 0) ∧
    a = 4 ∧ b = Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l1796_179687


namespace NUMINAMATH_CALUDE_quadratic_rewrite_product_l1796_179698

theorem quadratic_rewrite_product (p q r : ℤ) : 
  (∀ x, 4 * x^2 - 20 * x - 32 = (p * x + q)^2 + r) → p * q = -10 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_rewrite_product_l1796_179698


namespace NUMINAMATH_CALUDE_congruent_count_l1796_179679

theorem congruent_count : Nat.card {n : ℕ | 0 < n ∧ n < 500 ∧ n % 7 = 3} = 71 := by
  sorry

end NUMINAMATH_CALUDE_congruent_count_l1796_179679


namespace NUMINAMATH_CALUDE_sine_graph_shift_l1796_179675

theorem sine_graph_shift (x : ℝ) :
  2 * Real.sin (3 * (x - 5 * π / 18) + π / 2) = 2 * Real.sin (3 * x - π / 3) := by
  sorry

end NUMINAMATH_CALUDE_sine_graph_shift_l1796_179675


namespace NUMINAMATH_CALUDE_composition_equality_l1796_179614

/-- Given two functions f and g, prove that their composition at x = 3 equals 103 -/
theorem composition_equality (f g : ℝ → ℝ) 
  (hf : ∀ x, f x = 4 * x + 3) 
  (hg : ∀ x, g x = (x + 2) ^ 2) : 
  f (g 3) = 103 := by
  sorry

end NUMINAMATH_CALUDE_composition_equality_l1796_179614


namespace NUMINAMATH_CALUDE_simplify_fraction_l1796_179626

theorem simplify_fraction : (48 : ℚ) / 72 = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l1796_179626


namespace NUMINAMATH_CALUDE_cubic_roots_sum_l1796_179666

theorem cubic_roots_sum (a b c : ℝ) : 
  (a^3 - a - 2 = 0) → 
  (b^3 - b - 2 = 0) → 
  (c^3 - c - 2 = 0) → 
  2*a*(b - c)^2 + 2*b*(c - a)^2 + 2*c*(a - b)^2 = -36 := by
  sorry

end NUMINAMATH_CALUDE_cubic_roots_sum_l1796_179666


namespace NUMINAMATH_CALUDE_point_on_line_l1796_179647

theorem point_on_line : ∃ (x y : ℚ), x = 3 ∧ y = 16/7 ∧ 4*x + 7*y = 28 := by
  sorry

end NUMINAMATH_CALUDE_point_on_line_l1796_179647


namespace NUMINAMATH_CALUDE_new_york_to_cape_town_duration_l1796_179639

/-- Represents a time in hours and minutes -/
structure Time where
  hours : ℕ
  minutes : ℕ
  valid : minutes < 60

/-- Represents a day of the week -/
inductive Day
| Monday
| Tuesday

/-- Calculates the time difference between two times in hours -/
def timeDifference (t1 t2 : Time) (d1 d2 : Day) : ℕ :=
  sorry

/-- The departure time from London -/
def londonDeparture : Time := { hours := 6, minutes := 0, valid := by simp }

/-- The arrival time in Cape Town -/
def capeTownArrival : Time := { hours := 10, minutes := 0, valid := by simp }

/-- Theorem stating the duration of the New York to Cape Town flight -/
theorem new_york_to_cape_town_duration :
  let londonToNewYorkDuration : ℕ := 18
  let newYorkArrival : Time := 
    { hours := 0, minutes := 0, valid := by simp }
  let newYorkToCapeArrivalDay : Day := Day.Tuesday
  timeDifference newYorkArrival capeTownArrival Day.Tuesday newYorkToCapeArrivalDay = 10 :=
sorry

end NUMINAMATH_CALUDE_new_york_to_cape_town_duration_l1796_179639


namespace NUMINAMATH_CALUDE_train_length_l1796_179623

/-- The length of a train given its speed and time to pass a stationary observer -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 144 → time = 4 → speed * time * (1000 / 3600) = 160 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l1796_179623


namespace NUMINAMATH_CALUDE_inequality_proof_l1796_179657

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 1) :
  (1 / (a^2 * (b + c))) + (1 / (b^2 * (c + a))) + (1 / (c^2 * (a + b))) ≥ 3/2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1796_179657


namespace NUMINAMATH_CALUDE_vectors_opposite_x_value_l1796_179682

-- Define the vectors a and b
def a (x : ℝ) : Fin 2 → ℝ := ![2*x, 1]
def b (x : ℝ) : Fin 2 → ℝ := ![4, x]

-- Define the condition that vectors are in opposite directions
def opposite_directions (v w : Fin 2 → ℝ) : Prop :=
  ∃ (k : ℝ), k < 0 ∧ (∀ i, v i = -k * w i)

-- Theorem statement
theorem vectors_opposite_x_value :
  ∀ x : ℝ, opposite_directions (a x) (b x) → x = -Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_vectors_opposite_x_value_l1796_179682


namespace NUMINAMATH_CALUDE_exists_four_axes_symmetry_l1796_179648

/-- A type representing a figure on a grid paper -/
structure GridFigure where
  cells : Set (ℤ × ℤ)

/-- A type representing an axis of symmetry -/
structure AxisOfSymmetry where
  -- Define properties of an axis of symmetry

/-- Function to count the number of axes of symmetry in a figure -/
def countAxesOfSymmetry (f : GridFigure) : ℕ := sorry

/-- Function to shade one more cell in a figure -/
def shadeOneMoreCell (f : GridFigure) : GridFigure := sorry

/-- Theorem stating that it's possible to create a figure with four axes of symmetry 
    by shading one more cell in a figure with no axes of symmetry -/
theorem exists_four_axes_symmetry :
  ∃ (f : GridFigure), 
    countAxesOfSymmetry f = 0 ∧ 
    ∃ (g : GridFigure), g = shadeOneMoreCell f ∧ countAxesOfSymmetry g = 4 := by
  sorry

end NUMINAMATH_CALUDE_exists_four_axes_symmetry_l1796_179648


namespace NUMINAMATH_CALUDE_moon_speed_km_per_second_l1796_179646

-- Define the speed of the moon in kilometers per hour
def moon_speed_km_per_hour : ℝ := 3672

-- Define the number of seconds in an hour
def seconds_per_hour : ℝ := 3600

-- Theorem statement
theorem moon_speed_km_per_second :
  moon_speed_km_per_hour / seconds_per_hour = 1.02 := by
  sorry

end NUMINAMATH_CALUDE_moon_speed_km_per_second_l1796_179646


namespace NUMINAMATH_CALUDE_abc_inequality_l1796_179642

theorem abc_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a * b * c = 1) :
  a + b + c ≤ a^2 + b^2 + c^2 := by
  sorry

end NUMINAMATH_CALUDE_abc_inequality_l1796_179642


namespace NUMINAMATH_CALUDE_product_congruence_l1796_179663

theorem product_congruence : 198 * 963 ≡ 24 [ZMOD 50] := by sorry

end NUMINAMATH_CALUDE_product_congruence_l1796_179663


namespace NUMINAMATH_CALUDE_partner_b_contribution_l1796_179677

/-- Represents the capital contribution of a business partner -/
structure Capital where
  amount : ℝ
  duration : ℕ

/-- Calculates the adjusted capital contribution -/
def adjustedCapital (c : Capital) : ℝ := c.amount * c.duration

/-- Represents the profit-sharing ratio between two partners -/
structure ProfitRatio where
  partner1 : ℕ
  partner2 : ℕ

theorem partner_b_contribution 
  (a : Capital) 
  (b : Capital)
  (ratio : ProfitRatio)
  (h1 : a.amount = 3500)
  (h2 : a.duration = 12)
  (h3 : b.duration = 7)
  (h4 : ratio.partner1 = 2)
  (h5 : ratio.partner2 = 3)
  (h6 : (adjustedCapital a) / (adjustedCapital b) = ratio.partner1 / ratio.partner2) :
  b.amount = 4500 := by
  sorry

end NUMINAMATH_CALUDE_partner_b_contribution_l1796_179677


namespace NUMINAMATH_CALUDE_clerk_forms_per_hour_l1796_179603

theorem clerk_forms_per_hour 
  (total_forms : ℕ) 
  (work_hours : ℕ) 
  (num_clerks : ℕ) 
  (h1 : total_forms = 2400) 
  (h2 : work_hours = 8) 
  (h3 : num_clerks = 12) : 
  (total_forms / work_hours) / num_clerks = 25 := by
sorry

end NUMINAMATH_CALUDE_clerk_forms_per_hour_l1796_179603
