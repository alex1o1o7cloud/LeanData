import Mathlib

namespace NUMINAMATH_CALUDE_vector_eq_quadratic_eq_l1_116

/-- The vector representing k(3, -4, 1) - (6, 9, -2) --/
def v (k : ℝ) : Fin 3 → ℝ := λ i =>
  match i with
  | 0 => 3*k - 6
  | 1 => -4*k - 9
  | 2 => k + 2

/-- The squared norm of the vector --/
def squared_norm (k : ℝ) : ℝ := (v k 0)^2 + (v k 1)^2 + (v k 2)^2

/-- The theorem stating the equivalence between the vector equation and the quadratic equation --/
theorem vector_eq_quadratic_eq (k : ℝ) :
  squared_norm k = (3 * Real.sqrt 26)^2 ↔ 26 * k^2 + 40 * k - 113 = 0 := by sorry

end NUMINAMATH_CALUDE_vector_eq_quadratic_eq_l1_116


namespace NUMINAMATH_CALUDE_original_number_l1_167

theorem original_number (x : ℝ) : x * 1.5 = 150 → x = 100 := by
  sorry

end NUMINAMATH_CALUDE_original_number_l1_167


namespace NUMINAMATH_CALUDE_sum_of_quadratic_and_linear_l1_146

/-- Given a quadratic function q and a linear function p satisfying certain conditions,
    prove that their sum has a specific form. -/
theorem sum_of_quadratic_and_linear 
  (q : ℝ → ℝ) 
  (p : ℝ → ℝ) 
  (hq_quad : ∃ a b c : ℝ, ∀ x, q x = a * x^2 + b * x + c)
  (hq_zeros : q 1 = 0 ∧ q 3 = 0)
  (hq_value : q 4 = 8)
  (hp_linear : ∃ m b : ℝ, ∀ x, p x = m * x + b)
  (hp_value : p 5 = 15) :
  ∀ x, p x + q x = (8/3) * x^2 - (29/3) * x + 8 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_quadratic_and_linear_l1_146


namespace NUMINAMATH_CALUDE_unique_cube_difference_61_l1_196

theorem unique_cube_difference_61 :
  ∃! (n k : ℕ), n^3 - k^3 = 61 :=
by sorry

end NUMINAMATH_CALUDE_unique_cube_difference_61_l1_196


namespace NUMINAMATH_CALUDE_souvenir_shop_theorem_l1_163

/-- Represents the purchase and profit scenario of a souvenir shop. -/
structure SouvenirShop where
  price_A : ℚ  -- Purchase price of souvenir A
  price_B : ℚ  -- Purchase price of souvenir B
  profit_A : ℚ -- Profit per piece of souvenir A
  profit_B : ℚ -- Profit per piece of souvenir B

/-- Theorem stating the correct purchase prices and total profit -/
theorem souvenir_shop_theorem (shop : SouvenirShop) : 
  (7 * shop.price_A + 8 * shop.price_B = 380) →
  (10 * shop.price_A + 6 * shop.price_B = 380) →
  shop.profit_A = 5 →
  shop.profit_B = 7 →
  (∃ (m n : ℚ), m + n = 40 ∧ shop.price_A * m + shop.price_B * n = 900) →
  (shop.price_A = 20 ∧ shop.price_B = 30 ∧ 
   ∃ (m n : ℚ), m + n = 40 ∧ shop.price_A * m + shop.price_B * n = 900 ∧
                m * shop.profit_A + n * shop.profit_B = 220) := by
  sorry


end NUMINAMATH_CALUDE_souvenir_shop_theorem_l1_163


namespace NUMINAMATH_CALUDE_M_eq_302_l1_187

/-- The number of ways to write 3010 as a sum of powers of 10 with restricted coefficients -/
def M : ℕ :=
  (Finset.filter (fun (b₃ : ℕ) =>
    (Finset.filter (fun (b₂ : ℕ) =>
      (Finset.filter (fun (b₁ : ℕ) =>
        (Finset.filter (fun (b₀ : ℕ) =>
          b₃ * 1000 + b₂ * 100 + b₁ * 10 + b₀ = 3010
        ) (Finset.range 100)).card > 0
      ) (Finset.range 100)).card > 0
    ) (Finset.range 100)).card > 0
  ) (Finset.range 100)).card

/-- The theorem stating that M equals 302 -/
theorem M_eq_302 : M = 302 := by
  sorry

end NUMINAMATH_CALUDE_M_eq_302_l1_187


namespace NUMINAMATH_CALUDE_cow_chicken_problem_l1_172

theorem cow_chicken_problem (cows chickens : ℕ) : 
  (4 * cows + 2 * chickens = 2 * (cows + chickens) + 20) → cows = 10 := by
  sorry

end NUMINAMATH_CALUDE_cow_chicken_problem_l1_172


namespace NUMINAMATH_CALUDE_inequalities_always_true_l1_181

theorem inequalities_always_true (a b : ℝ) (h : a * b > 0) :
  (a^2 + b^2 ≥ 2*a*b) ∧ (b/a + a/b ≥ 2) := by sorry

end NUMINAMATH_CALUDE_inequalities_always_true_l1_181


namespace NUMINAMATH_CALUDE_distance_between_locations_l1_126

/-- The distance between two locations A and B given the conditions of two couriers --/
theorem distance_between_locations (x : ℝ) (y : ℝ) : 
  (x > 0) →  -- x is the number of days until the couriers meet
  (y > 0) →  -- y is the total distance between A and B
  (y / (x + 9) + y / (x + 16) = y) →  -- sum of distances traveled equals total distance
  (y / (x + 9) - y / (x + 16) = 12) →  -- difference in distances traveled is 12 miles
  (x^2 = 144) →  -- derived from solving the equations
  y = 84 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_locations_l1_126


namespace NUMINAMATH_CALUDE_comic_collection_overtake_l1_112

/-- The number of months after which LaShawn's collection becomes 1.5 times Kymbrea's --/
def months_to_overtake : ℕ := 70

/-- Kymbrea's initial number of comic books --/
def kymbrea_initial : ℕ := 40

/-- LaShawn's initial number of comic books --/
def lashawn_initial : ℕ := 25

/-- Kymbrea's monthly collection rate --/
def kymbrea_rate : ℕ := 3

/-- LaShawn's monthly collection rate --/
def lashawn_rate : ℕ := 5

/-- Theorem stating that after the specified number of months, 
    LaShawn's collection is 1.5 times Kymbrea's --/
theorem comic_collection_overtake :
  (lashawn_initial + lashawn_rate * months_to_overtake : ℚ) = 
  1.5 * (kymbrea_initial + kymbrea_rate * months_to_overtake) :=
by sorry

end NUMINAMATH_CALUDE_comic_collection_overtake_l1_112


namespace NUMINAMATH_CALUDE_b_85_mod_50_l1_173

/-- The sequence b_n is defined as 7^n + 9^n -/
def b (n : ℕ) : ℕ := 7^n + 9^n

/-- The 85th term of the sequence b_n is congruent to 36 modulo 50 -/
theorem b_85_mod_50 : b 85 ≡ 36 [ZMOD 50] := by sorry

end NUMINAMATH_CALUDE_b_85_mod_50_l1_173


namespace NUMINAMATH_CALUDE_circle_area_ratio_l1_184

/-- Given two circles C and D, if an arc of 60° on C has the same length as an arc of 40° on D,
    then the ratio of the area of C to the area of D is 9/4. -/
theorem circle_area_ratio (C D : Real) (hC : C > 0) (hD : D > 0) 
  (h : C * (60 / 360) = D * (40 / 360)) : 
  (C^2 / D^2 : Real) = 9/4 := by
  sorry

end NUMINAMATH_CALUDE_circle_area_ratio_l1_184


namespace NUMINAMATH_CALUDE_negation_of_existence_exponential_cube_inequality_l1_100

theorem negation_of_existence (p : ℝ → Prop) :
  (¬ ∃ x : ℝ, x > 0 ∧ p x) ↔ (∀ x : ℝ, x > 0 → ¬ p x) := by sorry

theorem exponential_cube_inequality :
  (¬ ∃ x : ℝ, x > 0 ∧ 3^x < x^3) ↔ (∀ x : ℝ, x > 0 → 3^x ≥ x^3) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_exponential_cube_inequality_l1_100


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_proposition_l1_192

theorem negation_of_existence (p : ℝ → Prop) :
  (¬ ∃ x : ℝ, p x) ↔ (∀ x : ℝ, ¬ p x) := by sorry

theorem negation_of_proposition :
  (¬ ∃ x : ℝ, x > Real.sin x) ↔ (∀ x : ℝ, x ≤ Real.sin x) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_proposition_l1_192


namespace NUMINAMATH_CALUDE_cyclic_sum_inequality_l1_104

theorem cyclic_sum_inequality (k : ℕ) (x y z : ℝ) 
  (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0)
  (h_sum : x + y + z = 1) :
  (x^(k+2) / (x^(k+1) + y^k + z^k)) + 
  (y^(k+2) / (y^(k+1) + z^k + x^k)) + 
  (z^(k+2) / (z^(k+1) + x^k + y^k)) ≥ 1/7 ∧
  ((x^(k+2) / (x^(k+1) + y^k + z^k)) + 
   (y^(k+2) / (y^(k+1) + z^k + x^k)) + 
   (z^(k+2) / (z^(k+1) + x^k + y^k)) = 1/7 ↔ 
   x = 1/3 ∧ y = 1/3 ∧ z = 1/3) := by
sorry

end NUMINAMATH_CALUDE_cyclic_sum_inequality_l1_104


namespace NUMINAMATH_CALUDE_fixed_point_on_line_l1_132

theorem fixed_point_on_line (t : ℝ) : 
  (t + 1) * (-4) - (2 * t + 5) * (-2) - 6 = 0 := by
  sorry

#check fixed_point_on_line

end NUMINAMATH_CALUDE_fixed_point_on_line_l1_132


namespace NUMINAMATH_CALUDE_chessboard_divisibility_theorem_l1_115

/-- Represents a chessboard with natural numbers -/
def Chessboard := Matrix (Fin 8) (Fin 8) ℕ

/-- Represents an operation on the chessboard -/
inductive Operation
  | inc_3x3 (i j : Fin 6) : Operation
  | inc_4x4 (i j : Fin 5) : Operation

/-- Applies an operation to a chessboard -/
def apply_operation (board : Chessboard) (op : Operation) : Chessboard :=
  match op with
  | Operation.inc_3x3 i j => sorry
  | Operation.inc_4x4 i j => sorry

/-- Checks if all elements in the chessboard are divisible by 10 -/
def all_divisible_by_10 (board : Chessboard) : Prop :=
  ∀ i j, board i j % 10 = 0

/-- Main theorem: There exists a sequence of operations that makes all numbers divisible by 10 -/
theorem chessboard_divisibility_theorem (initial_board : Chessboard) :
  ∃ (ops : List Operation), all_divisible_by_10 (ops.foldl apply_operation initial_board) :=
sorry

end NUMINAMATH_CALUDE_chessboard_divisibility_theorem_l1_115


namespace NUMINAMATH_CALUDE_teal_survey_result_l1_149

/-- Represents the survey results about the color teal --/
structure TealSurvey where
  total : ℕ
  blue : ℕ
  both : ℕ
  neither : ℕ

/-- Calculates the number of people who believe teal is a shade of green --/
def green_believers (survey : TealSurvey) : ℕ :=
  survey.total - survey.blue + survey.both - survey.neither

/-- Theorem stating the result of the survey --/
theorem teal_survey_result (survey : TealSurvey) 
  (h_total : survey.total = 200)
  (h_blue : survey.blue = 130)
  (h_both : survey.both = 45)
  (h_neither : survey.neither = 35) :
  green_believers survey = 80 := by
  sorry

#eval green_believers { total := 200, blue := 130, both := 45, neither := 35 }

end NUMINAMATH_CALUDE_teal_survey_result_l1_149


namespace NUMINAMATH_CALUDE_gain_percent_for_50_and_28_l1_127

/-- Calculates the gain percent given the number of articles at cost price and selling price that are equal in total value -/
def gainPercent (costArticles : ℕ) (sellArticles : ℕ) : ℚ :=
  let gain := (costArticles : ℚ) / sellArticles - 1
  gain * 100

/-- Proves that when 50 articles at cost price equal 28 articles at selling price, the gain percent is (11/14) * 100 -/
theorem gain_percent_for_50_and_28 :
  gainPercent 50 28 = 11 / 14 * 100 := by
  sorry

#eval gainPercent 50 28

end NUMINAMATH_CALUDE_gain_percent_for_50_and_28_l1_127


namespace NUMINAMATH_CALUDE_total_commission_is_4200_l1_129

def coupe_price : ℝ := 30000
def suv_price : ℝ := 2 * coupe_price
def luxury_sedan_price : ℝ := 80000
def commission_rate_coupe_suv : ℝ := 0.02
def commission_rate_luxury : ℝ := 0.03

def total_commission : ℝ :=
  coupe_price * commission_rate_coupe_suv +
  suv_price * commission_rate_coupe_suv +
  luxury_sedan_price * commission_rate_luxury

theorem total_commission_is_4200 :
  total_commission = 4200 := by sorry

end NUMINAMATH_CALUDE_total_commission_is_4200_l1_129


namespace NUMINAMATH_CALUDE_cafeteria_extra_apples_l1_131

/-- The number of red apples ordered by the cafeteria -/
def red_apples : ℕ := 33

/-- The number of green apples ordered by the cafeteria -/
def green_apples : ℕ := 23

/-- The number of students who wanted fruit -/
def students_wanting_fruit : ℕ := 21

/-- Each student takes one apple -/
axiom one_apple_per_student : ℕ

/-- The number of extra apples the cafeteria ended up with -/
def extra_apples : ℕ := (red_apples + green_apples) - students_wanting_fruit

theorem cafeteria_extra_apples : extra_apples = 35 := by
  sorry

end NUMINAMATH_CALUDE_cafeteria_extra_apples_l1_131


namespace NUMINAMATH_CALUDE_direction_vectors_of_line_l1_122

/-- Given a line with equation 3x - 4y + 1 = 0, prove that (4, 3) and (1, 3/4) are valid direction vectors. -/
theorem direction_vectors_of_line (x y : ℝ) : 
  (3 * x - 4 * y + 1 = 0) →
  (∃ (k : ℝ), k ≠ 0 ∧ (k * 4, k * 3) = (3, -4)) ∧
  (∃ (k : ℝ), k ≠ 0 ∧ (k * 1, k * (3/4)) = (3, -4)) :=
by sorry

end NUMINAMATH_CALUDE_direction_vectors_of_line_l1_122


namespace NUMINAMATH_CALUDE_frederick_tyson_age_ratio_l1_171

/-- Represents the ages and relationships between Kyle, Julian, Frederick, and Tyson -/
structure AgeRelationships where
  kyle_age : ℕ
  tyson_age : ℕ
  kyle_julian_diff : ℕ
  julian_frederick_diff : ℕ
  kyle_age_is_25 : kyle_age = 25
  tyson_age_is_20 : tyson_age = 20
  kyle_older_than_julian : kyle_age = kyle_julian_diff + (kyle_age - kyle_julian_diff)
  julian_younger_than_frederick : kyle_age - kyle_julian_diff = (kyle_age - kyle_julian_diff + julian_frederick_diff) - julian_frederick_diff

/-- The ratio of Frederick's age to Tyson's age is 2:1 -/
theorem frederick_tyson_age_ratio (ar : AgeRelationships) :
  (ar.kyle_age - ar.kyle_julian_diff + ar.julian_frederick_diff) / ar.tyson_age = 2 := by
  sorry

end NUMINAMATH_CALUDE_frederick_tyson_age_ratio_l1_171


namespace NUMINAMATH_CALUDE_pencil_cost_l1_176

/-- Given that 120 pencils cost $36, prove that 3000 pencils cost $900 -/
theorem pencil_cost (pencils_per_box : ℕ) (cost_per_box : ℚ) (total_pencils : ℕ) :
  pencils_per_box = 120 →
  cost_per_box = 36 →
  total_pencils = 3000 →
  (cost_per_box / pencils_per_box) * total_pencils = 900 :=
by sorry

end NUMINAMATH_CALUDE_pencil_cost_l1_176


namespace NUMINAMATH_CALUDE_line_through_point_parallel_to_line_l1_153

/-- A line in the 2D plane represented by its equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line -/
def Point.liesOn (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Check if two lines are parallel -/
def Line.isParallelTo (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

theorem line_through_point_parallel_to_line
  (p : Point)
  (l1 l2 : Line)
  (h1 : p.liesOn l2)
  (h2 : l2.isParallelTo l1)
  (h3 : l1.a = 1)
  (h4 : l1.b = -2)
  (h5 : l1.c = 3)
  (h6 : p.x = 1)
  (h7 : p.y = -3)
  (h8 : l2.a = 1)
  (h9 : l2.b = -2)
  (h10 : l2.c = -7) :
  True := by sorry

end NUMINAMATH_CALUDE_line_through_point_parallel_to_line_l1_153


namespace NUMINAMATH_CALUDE_egyptian_fraction_sum_l1_165

theorem egyptian_fraction_sum : ∃ (b₂ b₃ b₄ b₅ : ℕ),
  (3 : ℚ) / 5 = b₂ / 2 + b₃ / 6 + b₄ / 24 + b₅ / 120 ∧
  b₂ < 2 ∧ b₃ < 3 ∧ b₄ < 4 ∧ b₅ < 5 ∧
  b₂ + b₃ + b₄ + b₅ = 4 := by
  sorry

end NUMINAMATH_CALUDE_egyptian_fraction_sum_l1_165


namespace NUMINAMATH_CALUDE_one_plus_x_geq_two_sqrt_x_l1_186

theorem one_plus_x_geq_two_sqrt_x (x : ℝ) (h : x ≥ 0) : 1 + x ≥ 2 * Real.sqrt x := by
  sorry

end NUMINAMATH_CALUDE_one_plus_x_geq_two_sqrt_x_l1_186


namespace NUMINAMATH_CALUDE_gcd_102_238_l1_136

theorem gcd_102_238 : Nat.gcd 102 238 = 34 := by sorry

end NUMINAMATH_CALUDE_gcd_102_238_l1_136


namespace NUMINAMATH_CALUDE_parallel_line_through_point_l1_169

/-- Given a line L1 with equation x - 2y - 2 = 0, prove that the line L2 with equation x - 2y - 1 = 0
    passes through the point (1,0) and is parallel to L1. -/
theorem parallel_line_through_point (x y : ℝ) : 
  (x - 2*y - 1 = 0) ∧ 
  (1 - 2*0 - 1 = 0) ∧ 
  (∃ k : ℝ, k ≠ 0 ∧ (1 : ℝ) = k * 1 ∧ (-2 : ℝ) = k * (-2)) := by
  sorry

end NUMINAMATH_CALUDE_parallel_line_through_point_l1_169


namespace NUMINAMATH_CALUDE_calculate_expression_l1_113

theorem calculate_expression : 2^2 + |-3| - Real.sqrt 25 = 2 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l1_113


namespace NUMINAMATH_CALUDE_three_distinct_roots_l1_138

/-- The equation has exactly three distinct roots if and only if a is in the set {-1.5, -0.75, 0, 1/4} -/
theorem three_distinct_roots (a : ℝ) :
  (∃ (x y z : ℝ), x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    (∀ w : ℝ, (w^2 + (2*a - 1)*w - 4*a - 2) * (w^2 + w + a) = 0 ↔ w = x ∨ w = y ∨ w = z)) ↔
  a = -1.5 ∨ a = -0.75 ∨ a = 0 ∨ a = 1/4 := by
sorry

end NUMINAMATH_CALUDE_three_distinct_roots_l1_138


namespace NUMINAMATH_CALUDE_permutation_equation_solution_l1_118

def A (k m : ℕ) : ℕ := (k.factorial) / (k - m).factorial

theorem permutation_equation_solution :
  ∃! n : ℕ, n > 0 ∧ A (2*n) 3 = 10 * A n 3 :=
sorry

end NUMINAMATH_CALUDE_permutation_equation_solution_l1_118


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l1_111

theorem quadratic_inequality_range (k : ℝ) : 
  (∀ x : ℝ, k * x^2 - k * x - 1 < 0) ↔ -4 < k ∧ k ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l1_111


namespace NUMINAMATH_CALUDE_coefficient_of_x_squared_l1_121

def original_expression (x : ℝ) : ℝ :=
  2 * (x - 6) + 5 * (10 - 3 * x^2 + 4 * x) - 7 * (3 * x^2 - 2 * x + 1)

theorem coefficient_of_x_squared :
  ∃ (a b c : ℝ), ∀ x, original_expression x = a * x^2 + b * x + c ∧ a = -36 :=
sorry

end NUMINAMATH_CALUDE_coefficient_of_x_squared_l1_121


namespace NUMINAMATH_CALUDE_project_completion_time_l1_170

/-- The number of days A takes to complete the project alone -/
def A_days : ℝ := 20

/-- The number of days it takes A and B together to complete the project -/
def total_days : ℝ := 15

/-- The number of days before completion that A quits -/
def A_quit_days : ℝ := 5

/-- The number of days B takes to complete the project alone -/
def B_days : ℝ := 30

/-- Theorem stating that given the conditions, B can complete the project alone in 30 days -/
theorem project_completion_time :
  A_days = 20 ∧ total_days = 15 ∧ A_quit_days = 5 →
  (total_days - A_quit_days) * (1 / A_days + 1 / B_days) + A_quit_days * (1 / B_days) = 1 :=
by sorry

end NUMINAMATH_CALUDE_project_completion_time_l1_170


namespace NUMINAMATH_CALUDE_four_right_angles_implies_plane_figure_less_than_four_right_angles_can_be_non_planar_l1_128

-- Define a quadrilateral
structure Quadrilateral :=
  (is_plane : Bool)
  (right_angles : Nat)

-- Define the property of being a plane figure
def is_plane_figure (q : Quadrilateral) : Prop :=
  q.is_plane = true

-- Define the property of having four right angles
def has_four_right_angles (q : Quadrilateral) : Prop :=
  q.right_angles = 4

-- Theorem stating that a quadrilateral with four right angles must be a plane figure
theorem four_right_angles_implies_plane_figure (q : Quadrilateral) :
  has_four_right_angles q → is_plane_figure q :=
by sorry

-- Theorem stating that quadrilaterals with less than four right angles can be non-planar
theorem less_than_four_right_angles_can_be_non_planar :
  ∃ (q : Quadrilateral), q.right_angles < 4 ∧ ¬(is_plane_figure q) :=
by sorry

end NUMINAMATH_CALUDE_four_right_angles_implies_plane_figure_less_than_four_right_angles_can_be_non_planar_l1_128


namespace NUMINAMATH_CALUDE_min_value_of_function_l1_105

theorem min_value_of_function (x a b : ℝ) 
  (hx : 0 < x ∧ x < 1) (ha : a > 0) (hb : b > 0) : 
  a^2 / x + b^2 / (1 - x) ≥ (a + b)^2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_function_l1_105


namespace NUMINAMATH_CALUDE_omega_sequence_monotone_l1_110

def is_omega_sequence (d : ℕ → ℕ) : Prop :=
  (∀ n, (d n + d (n + 2)) / 2 ≤ d (n + 1)) ∧
  (∃ M : ℝ, ∀ n, (d n : ℝ) ≤ M)

theorem omega_sequence_monotone (d : ℕ → ℕ) 
  (h_omega : is_omega_sequence d) :
  ∀ n, d n ≤ d (n + 1) := by
sorry

end NUMINAMATH_CALUDE_omega_sequence_monotone_l1_110


namespace NUMINAMATH_CALUDE_function_properties_imply_specific_form_and_result_l1_164

noncomputable def f (ω φ : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x + φ)

theorem function_properties_imply_specific_form_and_result 
  (ω φ : ℝ) (h_ω : ω > 0) (h_φ : 0 < φ ∧ φ < Real.pi) :
  (∀ x : ℝ, f ω φ (x + Real.pi / 2) = f ω φ (Real.pi / 2 - x)) →
  (∀ x : ℝ, ∃ k : ℤ, f ω φ (x + Real.pi / (2 * ω)) = f ω φ (x + k * Real.pi / ω)) →
  (∃ α : ℝ, 0 < α ∧ α < Real.pi / 2 ∧ f ω φ (α / 2 + Real.pi / 12) = 3 / 5) →
  (∀ x : ℝ, f ω φ x = Real.cos (2 * x)) ∧
  (∀ α : ℝ, 0 < α → α < Real.pi / 2 → f ω φ (α / 2 + Real.pi / 12) = 3 / 5 → 
    Real.sin (2 * α) = (24 + 7 * Real.sqrt 3) / 50) := by
  sorry

end NUMINAMATH_CALUDE_function_properties_imply_specific_form_and_result_l1_164


namespace NUMINAMATH_CALUDE_f_1989_value_l1_133

/-- A function satisfying the given functional equation -/
def SatisfiesEquation (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x + 2) * (1 - f x) = 1 + f x

theorem f_1989_value (f : ℝ → ℝ) 
    (h_eq : SatisfiesEquation f) 
    (h_f1 : f 1 = 2 + Real.sqrt 3) : 
    f 1989 = -2 + Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_f_1989_value_l1_133


namespace NUMINAMATH_CALUDE_smallest_x_divisible_l1_119

theorem smallest_x_divisible : ∃ (x : ℤ), x = 36629 ∧ 
  (∀ (y : ℤ), y < x → ¬(33 ∣ (2 * y + 2) ∧ 44 ∣ (2 * y + 2) ∧ 55 ∣ (2 * y + 2) ∧ 666 ∣ (2 * y + 2))) ∧
  (33 ∣ (2 * x + 2) ∧ 44 ∣ (2 * x + 2) ∧ 55 ∣ (2 * x + 2) ∧ 666 ∣ (2 * x + 2)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_x_divisible_l1_119


namespace NUMINAMATH_CALUDE_min_value_theorem_l1_168

/-- Given f(x) = a^x - b, where a > 0, a ≠ 1, and b is real,
    and g(x) = x + 1, if f(x) * g(x) ≤ 0 for all real x,
    then the minimum value of 1/a + 4/b is 4. -/
theorem min_value_theorem (a b : ℝ) (ha : a > 0) (ha' : a ≠ 1) 
  (hf : ∀ x : ℝ, a^x - b ≤ 0 ∨ x + 1 ≤ 0) :
  ∀ ε > 0, ∃ a₀ b₀ : ℝ, 1/a₀ + 4/b₀ < 4 + ε :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1_168


namespace NUMINAMATH_CALUDE_sum_equation_solution_l1_191

/-- Given a real number k > 1 satisfying the infinite sum equation,
    prove that k equals the given expression. -/
theorem sum_equation_solution (k : ℝ) 
  (h1 : k > 1)
  (h2 : ∑' n, (7 * n - 2) / k^n = 3) :
  k = (21 + Real.sqrt 477) / 6 := by
  sorry

end NUMINAMATH_CALUDE_sum_equation_solution_l1_191


namespace NUMINAMATH_CALUDE_project_delay_without_additional_workers_l1_155

/-- Represents the construction project parameters and outcome -/
structure ConstructionProject where
  plannedDays : ℕ
  initialWorkers : ℕ
  additionalWorkers : ℕ
  additionalWorkersStartDay : ℕ
  actualCompletionDays : ℕ

/-- Calculates the total man-days of work for the project -/
def totalManDays (project : ConstructionProject) : ℕ :=
  project.initialWorkers * project.additionalWorkersStartDay +
  (project.initialWorkers + project.additionalWorkers) *
  (project.actualCompletionDays - project.additionalWorkersStartDay)

/-- Theorem: If a project is completed on time with additional workers,
    it would take longer without them -/
theorem project_delay_without_additional_workers
  (project : ConstructionProject)
  (h1 : project.plannedDays = 100)
  (h2 : project.initialWorkers = 100)
  (h3 : project.additionalWorkers = 100)
  (h4 : project.additionalWorkersStartDay = 50)
  (h5 : project.actualCompletionDays = 100) :
  (totalManDays project) / project.initialWorkers = 150 :=
sorry

end NUMINAMATH_CALUDE_project_delay_without_additional_workers_l1_155


namespace NUMINAMATH_CALUDE_compound_molecular_weight_l1_188

/-- Atomic weight of Carbon in g/mol -/
def carbon_weight : ℝ := 12.01

/-- Atomic weight of Hydrogen in g/mol -/
def hydrogen_weight : ℝ := 1.008

/-- Atomic weight of Oxygen in g/mol -/
def oxygen_weight : ℝ := 16.00

/-- Number of Carbon atoms in the compound -/
def carbon_count : ℕ := 4

/-- Number of Hydrogen atoms in the compound -/
def hydrogen_count : ℕ := 8

/-- Number of Oxygen atoms in the compound -/
def oxygen_count : ℕ := 2

/-- Calculation of molecular weight -/
def molecular_weight : ℝ :=
  (carbon_count : ℝ) * carbon_weight +
  (hydrogen_count : ℝ) * hydrogen_weight +
  (oxygen_count : ℝ) * oxygen_weight

/-- Theorem stating that the molecular weight of the compound is 88.104 g/mol -/
theorem compound_molecular_weight :
  molecular_weight = 88.104 := by sorry

end NUMINAMATH_CALUDE_compound_molecular_weight_l1_188


namespace NUMINAMATH_CALUDE_divisibility_by_48_l1_156

theorem divisibility_by_48 :
  (∀ (n : ℕ), n > 0 → ¬(48 ∣ (7^n + 1))) ∧
  (∀ (n : ℕ), n > 0 → (48 ∣ (7^n - 1) ↔ Even n)) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_48_l1_156


namespace NUMINAMATH_CALUDE_cannot_tile_removed_square_board_l1_117

/-- Represents a chessboard with one square removed -/
def RemovedSquareBoard : Nat := 63

/-- Represents the size of a domino -/
def DominoSize : Nat := 2

theorem cannot_tile_removed_square_board :
  ¬ ∃ (n : Nat), n * DominoSize = RemovedSquareBoard :=
sorry

end NUMINAMATH_CALUDE_cannot_tile_removed_square_board_l1_117


namespace NUMINAMATH_CALUDE_lizard_comparison_l1_130

/-- Represents a three-eyed lizard with wrinkles and spots -/
structure Lizard where
  eyes : Nat
  wrinkle_multiplier : Nat
  spot_multiplier : Nat

/-- Calculates the number of wrinkles for a lizard -/
def wrinkles (l : Lizard) : Nat :=
  l.eyes * l.wrinkle_multiplier

/-- Calculates the number of spots for a lizard -/
def spots (l : Lizard) : Nat :=
  l.spot_multiplier * (wrinkles l) ^ 2

/-- Calculates the total number of spots and wrinkles for a lizard -/
def total_spots_and_wrinkles (l : Lizard) : Nat :=
  spots l + wrinkles l

/-- The main theorem to prove -/
theorem lizard_comparison : 
  let jans_lizard : Lizard := { eyes := 3, wrinkle_multiplier := 3, spot_multiplier := 7 }
  let cousin_lizard : Lizard := { eyes := 3, wrinkle_multiplier := 2, spot_multiplier := 5 }
  total_spots_and_wrinkles jans_lizard + total_spots_and_wrinkles cousin_lizard - 
  (jans_lizard.eyes + cousin_lizard.eyes) = 756 := by
  sorry

end NUMINAMATH_CALUDE_lizard_comparison_l1_130


namespace NUMINAMATH_CALUDE_complex_number_in_first_quadrant_l1_102

theorem complex_number_in_first_quadrant : ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ (Complex.I / (1 + Complex.I) = ↑a + ↑b * Complex.I) := by
  sorry

end NUMINAMATH_CALUDE_complex_number_in_first_quadrant_l1_102


namespace NUMINAMATH_CALUDE_factor_divisor_statements_l1_135

theorem factor_divisor_statements :
  (∃ k : ℕ, 45 = 5 * k) ∧
  (∃ m : ℕ, 42 = 14 * m) ∧
  (∀ n : ℕ, 63 ≠ 14 * n) ∧
  (∃ p : ℕ, 180 = 9 * p) := by
  sorry

end NUMINAMATH_CALUDE_factor_divisor_statements_l1_135


namespace NUMINAMATH_CALUDE_unique_aabb_perfect_square_l1_120

/-- A 4-digit number of the form aabb in base 10 -/
def aabb (a b : ℕ) : ℕ := 1000 * a + 100 * a + 10 * b + b

/-- Predicate for a number being a perfect square -/
def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, k * k = n

theorem unique_aabb_perfect_square :
  ∀ a b : ℕ, 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 →
    (is_perfect_square (aabb a b) ↔ a = 7 ∧ b = 4) :=
sorry

end NUMINAMATH_CALUDE_unique_aabb_perfect_square_l1_120


namespace NUMINAMATH_CALUDE_root_transformation_l1_198

theorem root_transformation (r₁ r₂ r₃ : ℂ) : 
  (r₁^3 - 3*r₁^2 + 8 = 0) ∧ 
  (r₂^3 - 3*r₂^2 + 8 = 0) ∧ 
  (r₃^3 - 3*r₃^2 + 8 = 0) → 
  ((2*r₁)^3 - 6*(2*r₁)^2 + 64 = 0) ∧
  ((2*r₂)^3 - 6*(2*r₂)^2 + 64 = 0) ∧
  ((2*r₃)^3 - 6*(2*r₃)^2 + 64 = 0) := by
sorry

end NUMINAMATH_CALUDE_root_transformation_l1_198


namespace NUMINAMATH_CALUDE_ladder_length_l1_193

theorem ladder_length (initial_distance : ℝ) (pull_distance : ℝ) (slide_distance : ℝ) :
  initial_distance = 15 →
  pull_distance = 9 →
  slide_distance = 13 →
  ∃ (ladder_length : ℝ) (initial_height : ℝ),
    ladder_length ^ 2 = initial_distance ^ 2 + initial_height ^ 2 ∧
    ladder_length ^ 2 = (initial_distance + pull_distance) ^ 2 + (initial_height - slide_distance) ^ 2 ∧
    ladder_length = 25 := by
sorry

end NUMINAMATH_CALUDE_ladder_length_l1_193


namespace NUMINAMATH_CALUDE_system_solutions_correct_l1_157

theorem system_solutions_correct :
  -- System (1)
  (∃ x y : ℚ, x - y = 2 ∧ x + 1 = 2 * (y - 1) ∧ x = 7 ∧ y = 5) ∧
  -- System (2)
  (∃ x y : ℚ, 2 * x + 3 * y = 1 ∧ (y - 1) / 4 = (x - 2) / 3 ∧ x = 1 ∧ y = -1/3) :=
by sorry

end NUMINAMATH_CALUDE_system_solutions_correct_l1_157


namespace NUMINAMATH_CALUDE_alvin_marbles_l1_183

theorem alvin_marbles (initial_marbles : ℕ) : 
  initial_marbles - 18 + 25 = 64 → initial_marbles = 57 := by
  sorry

end NUMINAMATH_CALUDE_alvin_marbles_l1_183


namespace NUMINAMATH_CALUDE_tan_product_pi_ninths_l1_180

theorem tan_product_pi_ninths : 
  Real.tan (π / 9) * Real.tan (2 * π / 9) * Real.tan (4 * π / 9) = 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_product_pi_ninths_l1_180


namespace NUMINAMATH_CALUDE_tea_containers_theorem_l1_179

/-- Given a total amount of tea in gallons, the number of containers Geraldo drank,
    and the amount of tea Geraldo consumed in pints, calculate the total number of
    containers filled with tea. -/
def totalContainers (totalTea : ℚ) (containersDrunk : ℚ) (teaDrunk : ℚ) : ℚ :=
  (totalTea * 8) / (teaDrunk / containersDrunk)

/-- Prove that given 20 gallons of tea, where 3.5 containers contain 7 pints,
    the total number of containers filled is 80. -/
theorem tea_containers_theorem :
  totalContainers 20 (7/2) 7 = 80 := by
  sorry

end NUMINAMATH_CALUDE_tea_containers_theorem_l1_179


namespace NUMINAMATH_CALUDE_boys_in_class_l1_139

theorem boys_in_class (total : ℕ) (girl_ratio boy_ratio : ℕ) (h1 : total = 56) (h2 : girl_ratio = 4) (h3 : boy_ratio = 3) :
  (total * boy_ratio) / (girl_ratio + boy_ratio) = 24 :=
by sorry

end NUMINAMATH_CALUDE_boys_in_class_l1_139


namespace NUMINAMATH_CALUDE_multiple_with_binary_digits_l1_189

theorem multiple_with_binary_digits (n : ℤ) : ∃ m : ℤ, 
  (n ∣ m) ∧ 
  (∃ k : ℕ, k ≤ n ∧ m < 10^k) ∧
  (∀ d : ℕ, d < 10 → (m / 10^d % 10 = 0 ∨ m / 10^d % 10 = 1)) :=
by sorry

end NUMINAMATH_CALUDE_multiple_with_binary_digits_l1_189


namespace NUMINAMATH_CALUDE_exp_greater_or_equal_e_l1_177

theorem exp_greater_or_equal_e : ∀ x : ℝ, Real.exp x ≥ Real.exp 1 := by sorry

end NUMINAMATH_CALUDE_exp_greater_or_equal_e_l1_177


namespace NUMINAMATH_CALUDE_problem_solution_l1_123

theorem problem_solution (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : 2*a + b = 1) :
  (∀ a b, a*b ≥ 1/8) ∧
  (∀ a b, 1/a + 2/b ≥ 8) ∧
  (∀ a b, Real.sqrt (2*a) + Real.sqrt b ≤ Real.sqrt 2) ∧
  (∀ a b, (a+1)*(b+1) < 2) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l1_123


namespace NUMINAMATH_CALUDE_sqrt_2450_minus_2_theorem_l1_101

theorem sqrt_2450_minus_2_theorem (a b : ℕ+) :
  (Real.sqrt 2450 - 2 : ℝ) = ((Real.sqrt a.val : ℝ) - b.val)^2 →
  a.val + b.val = 2451 := by
sorry

end NUMINAMATH_CALUDE_sqrt_2450_minus_2_theorem_l1_101


namespace NUMINAMATH_CALUDE_chocolate_division_l1_151

/-- The amount of chocolate Shaina receives when Jordan divides his chocolate -/
theorem chocolate_division (total : ℚ) (keep_fraction : ℚ) (num_piles : ℕ) (piles_to_shaina : ℕ) : 
  total = 60 / 7 →
  keep_fraction = 1 / 3 →
  num_piles = 5 →
  piles_to_shaina = 2 →
  (1 - keep_fraction) * total * (piles_to_shaina / num_piles) = 16 / 7 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_division_l1_151


namespace NUMINAMATH_CALUDE_doug_initial_marbles_l1_160

theorem doug_initial_marbles (ed_marbles : ℕ) (ed_more_than_doug : ℕ) (doug_lost : ℕ)
  (h1 : ed_marbles = 27)
  (h2 : ed_more_than_doug = 5)
  (h3 : doug_lost = 3) :
  ed_marbles - ed_more_than_doug + doug_lost = 25 := by
  sorry

end NUMINAMATH_CALUDE_doug_initial_marbles_l1_160


namespace NUMINAMATH_CALUDE_right_triangle_side_length_l1_145

theorem right_triangle_side_length 
  (Q R S : ℝ × ℝ) -- Points in 2D plane
  (h_right : (R.1 - S.1) * (Q.1 - S.1) + (R.2 - S.2) * (Q.2 - S.2) = 0) -- Right angle at S
  (h_cos : (R.1 - Q.1) / Real.sqrt ((R.1 - Q.1)^2 + (R.2 - Q.2)^2) = 3/5) -- cos R = 3/5
  (h_rs : Real.sqrt ((R.1 - S.1)^2 + (R.2 - S.2)^2) = 10) -- RS = 10
  : Real.sqrt ((Q.1 - S.1)^2 + (Q.2 - S.2)^2) = 8 := by -- QS = 8
  sorry

end NUMINAMATH_CALUDE_right_triangle_side_length_l1_145


namespace NUMINAMATH_CALUDE_square_difference_l1_137

theorem square_difference (a b : ℝ) 
  (h1 : a^2 + a*b = 8) 
  (h2 : a*b + b^2 = 9) : 
  a^2 - b^2 = -1 := by
sorry

end NUMINAMATH_CALUDE_square_difference_l1_137


namespace NUMINAMATH_CALUDE_hyperbola_with_foci_on_y_axis_l1_103

theorem hyperbola_with_foci_on_y_axis 
  (m n : ℝ) 
  (h : m * n < 0) : 
  ∃ (a b : ℝ), 
    a > 0 ∧ b > 0 ∧ 
    (∀ (x y : ℝ), m * x^2 - m * y^2 = n ↔ y^2 / a^2 - x^2 / b^2 = 1) ∧
    (∀ (c : ℝ), c > a → ∃ (f₁ f₂ : ℝ), 
      f₁ = 0 ∧ f₂ = 0 ∧ 
      ∀ (x y : ℝ), m * x^2 - m * y^2 = n → 
        (x - f₁)^2 + (y - f₂)^2 - ((x - f₁)^2 + (y + f₂)^2) = 4 * c^2) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_with_foci_on_y_axis_l1_103


namespace NUMINAMATH_CALUDE_cos_derivative_at_pi_sixth_l1_154

theorem cos_derivative_at_pi_sixth (f : ℝ → ℝ) :
  (∀ x, f x = Real.cos x) → HasDerivAt f (-1/2) (π/6) := by
  sorry

end NUMINAMATH_CALUDE_cos_derivative_at_pi_sixth_l1_154


namespace NUMINAMATH_CALUDE_count_perfect_square_factors_7560_l1_178

def prime_factorization (n : Nat) : List (Nat × Nat) :=
  [(2, 3), (3, 3), (5, 1), (7, 1)]

def is_perfect_square (factor : List (Nat × Nat)) : Bool :=
  factor.all (fun (p, e) => e % 2 = 0)

def count_perfect_square_factors (n : Nat) : Nat :=
  let factors := List.filter is_perfect_square 
    (List.map (fun l => List.map (fun (p, e) => (p, Nat.min e l.2)) (prime_factorization n)) 
      [(2, 0), (2, 2), (3, 0), (3, 2), (5, 0), (7, 0)])
  factors.length

theorem count_perfect_square_factors_7560 :
  count_perfect_square_factors 7560 = 4 := by
  sorry

end NUMINAMATH_CALUDE_count_perfect_square_factors_7560_l1_178


namespace NUMINAMATH_CALUDE_baseball_team_points_l1_148

/- Define the structure of the team -/
structure BaseballTeam where
  totalPlayers : Nat
  totalPoints : Nat
  startingPlayers : Nat
  reservePlayers : Nat
  rookiePlayers : Nat
  totalGames : Nat

/- Define the theorem -/
theorem baseball_team_points 
  (team : BaseballTeam)
  (h1 : team.totalPlayers = 15)
  (h2 : team.totalPoints = 900)
  (h3 : team.startingPlayers = 7)
  (h4 : team.reservePlayers = 3)
  (h5 : team.rookiePlayers = 5)
  (h6 : team.totalGames = 20) :
  ∃ (startingAvg reserveAvg rookieAvg : ℕ),
    (startingAvg * team.startingPlayers * team.totalGames +
     reserveAvg * team.reservePlayers * 15 +
     rookieAvg * team.rookiePlayers * ((20 + 10 + 10 + 5 + 5) / 5) + 
     (10 + 15 + 15)) = team.totalPoints := by
  sorry


end NUMINAMATH_CALUDE_baseball_team_points_l1_148


namespace NUMINAMATH_CALUDE_fraction_problem_l1_150

theorem fraction_problem : ∃ x : ℚ, 
  x * (3/4 : ℚ) = (1/6 : ℚ) ∧ x - (1/12 : ℚ) = (5/36 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_fraction_problem_l1_150


namespace NUMINAMATH_CALUDE_hot_sauce_duration_l1_197

-- Define the size of a quart in ounces
def quart_size : ℝ := 32

-- Define the size of the hot sauce jar
def jar_size : ℝ := quart_size - 2

-- Define the size of each serving
def serving_size : ℝ := 0.5

-- Define the number of servings used daily
def daily_servings : ℕ := 3

-- Define the daily consumption
def daily_consumption : ℝ := serving_size * daily_servings

-- Theorem to prove
theorem hot_sauce_duration : 
  (jar_size / daily_consumption : ℝ) = 20 := by sorry

end NUMINAMATH_CALUDE_hot_sauce_duration_l1_197


namespace NUMINAMATH_CALUDE_biased_coin_probability_l1_140

/-- The probability of getting exactly k successes in n trials of a binomial experiment -/
def binomialProbability (n k : ℕ) (p : ℚ) : ℚ :=
  (n.choose k : ℚ) * p^k * (1 - p)^(n - k)

/-- The probability of getting exactly 9 heads in 12 flips of a biased coin with 1/3 probability of landing heads -/
theorem biased_coin_probability : 
  binomialProbability 12 9 (1/3) = 1760/531441 := by
  sorry

end NUMINAMATH_CALUDE_biased_coin_probability_l1_140


namespace NUMINAMATH_CALUDE_tv_price_reduction_l1_141

theorem tv_price_reduction (x : ℝ) : 
  (1 - x/100)^2 = 1 - 19/100 → x = 10 := by
  sorry

end NUMINAMATH_CALUDE_tv_price_reduction_l1_141


namespace NUMINAMATH_CALUDE_range_of_m_l1_107

/-- The function f(x) = x^2 - 3x + 4 -/
def f (x : ℝ) : ℝ := x^2 - 3*x + 4

theorem range_of_m (m : ℝ) :
  (∀ x ∈ Set.Icc 0 m, f x ∈ Set.Icc (7/4) 4) ∧
  (∃ x ∈ Set.Icc 0 m, f x = 7/4) ∧
  (∃ x ∈ Set.Icc 0 m, f x = 4) →
  m ∈ Set.Icc (3/2) 3 :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l1_107


namespace NUMINAMATH_CALUDE_base_conversion_and_division_l1_143

/-- Given that 746 in base 8 is equal to 4cd in base 10, where c and d are base-10 digits,
    prove that (c * d) / 12 = 4 -/
theorem base_conversion_and_division (c d : ℕ) : 
  c < 10 → d < 10 → 746 = 4 * c * 10 + d → (c * d) / 12 = 4 := by
  sorry

end NUMINAMATH_CALUDE_base_conversion_and_division_l1_143


namespace NUMINAMATH_CALUDE_circle_radius_l1_199

theorem circle_radius (x y : ℝ) (h : x + y = 72 * Real.pi) :
  ∃ r : ℝ, r > 0 ∧ x = Real.pi * r^2 ∧ y = 2 * Real.pi * r ∧ r = 6 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_l1_199


namespace NUMINAMATH_CALUDE_boat_speed_l1_174

/-- Given a boat that travels 11 km/h along a stream and 5 km/h against the same stream,
    the speed of the boat in still water is 8 km/h. -/
theorem boat_speed (b s : ℝ) 
    (h1 : b + s = 11)  -- Speed along the stream
    (h2 : b - s = 5)   -- Speed against the stream
    : b = 8 := by
  sorry

end NUMINAMATH_CALUDE_boat_speed_l1_174


namespace NUMINAMATH_CALUDE_two_congruent_rectangles_l1_182

/-- A point on a circle --/
structure CirclePoint where
  angle : ℝ
  angleInRange : 0 ≤ angle ∧ angle < 2 * Real.pi

/-- A rectangle inscribed in a circle --/
structure InscribedRectangle where
  vertices : Fin 4 → CirclePoint
  isRectangle : ∀ i : Fin 4, (vertices i).angle - (vertices ((i + 1) % 4)).angle = Real.pi / 2 ∨
                             (vertices i).angle - (vertices ((i + 1) % 4)).angle = -3 * Real.pi / 2

/-- The main theorem --/
theorem two_congruent_rectangles 
  (points : Fin 40 → CirclePoint)
  (equallySpaced : ∀ i : Fin 39, (points (i + 1)).angle - (points i).angle = Real.pi / 20)
  (rectangles : Fin 10 → InscribedRectangle)
  (verticesOnPoints : ∀ r : Fin 10, ∀ v : Fin 4, ∃ p : Fin 40, (rectangles r).vertices v = points p) :
  ∃ r1 r2 : Fin 10, r1 ≠ r2 ∧ rectangles r1 = rectangles r2 :=
sorry

end NUMINAMATH_CALUDE_two_congruent_rectangles_l1_182


namespace NUMINAMATH_CALUDE_silver_solution_percentage_second_solution_percentage_l1_147

/-- Given two silver solutions mixed to form a new solution, 
    calculate the silver percentage in the second solution. -/
theorem silver_solution_percentage 
  (volume1 : ℝ) (percent1 : ℝ) 
  (volume2 : ℝ) (final_percent : ℝ) : ℝ :=
  let total_volume := volume1 + volume2
  let silver_volume1 := volume1 * (percent1 / 100)
  let total_silver := total_volume * (final_percent / 100)
  let silver_volume2 := total_silver - silver_volume1
  (silver_volume2 / volume2) * 100

/-- Prove that the percentage of silver in the second solution is 10% -/
theorem second_solution_percentage : 
  silver_solution_percentage 5 4 2.5 6 = 10 := by
  sorry

end NUMINAMATH_CALUDE_silver_solution_percentage_second_solution_percentage_l1_147


namespace NUMINAMATH_CALUDE_chocolate_milk_amount_l1_124

/-- Represents the ingredients for making chocolate milk -/
structure Ingredients where
  milk : ℕ
  chocolate_syrup : ℕ
  whipped_cream : ℕ

/-- Represents the recipe for one glass of chocolate milk -/
structure Recipe where
  milk : ℕ
  chocolate_syrup : ℕ
  whipped_cream : ℕ
  total : ℕ

/-- Calculates the number of full glasses that can be made with given ingredients and recipe -/
def fullGlasses (i : Ingredients) (r : Recipe) : ℕ :=
  min (i.milk / r.milk) (min (i.chocolate_syrup / r.chocolate_syrup) (i.whipped_cream / r.whipped_cream))

/-- Theorem: Charles will drink 96 ounces of chocolate milk -/
theorem chocolate_milk_amount (i : Ingredients) (r : Recipe) :
  i.milk = 130 ∧ i.chocolate_syrup = 60 ∧ i.whipped_cream = 25 ∧
  r.milk = 4 ∧ r.chocolate_syrup = 2 ∧ r.whipped_cream = 2 ∧ r.total = 8 →
  fullGlasses i r * r.total = 96 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_milk_amount_l1_124


namespace NUMINAMATH_CALUDE_equation_solution_l1_144

theorem equation_solution : 
  ∃ x₁ x₂ : ℝ, x₁ = 2 ∧ x₂ = -4 ∧ 
  (∀ x : ℝ, (x - 1) * (x + 3) = 5 ↔ x = x₁ ∨ x = x₂) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l1_144


namespace NUMINAMATH_CALUDE_circle_radius_isosceles_right_triangle_l1_108

/-- The radius of a circle tangent to both axes and the hypotenuse of an isosceles right triangle -/
theorem circle_radius_isosceles_right_triangle (O : ℝ × ℝ) (P Q R S T U : ℝ × ℝ) (r : ℝ) :
  -- PQR is an isosceles right triangle
  (P.1 - Q.1)^2 + (P.2 - Q.2)^2 = 4 →
  (P.1 - R.1)^2 + (P.2 - R.2)^2 = (Q.1 - R.1)^2 + (Q.2 - R.2)^2 →
  (P.1 - R.1) * (Q.1 - R.1) + (P.2 - R.2) * (Q.2 - R.2) = 0 →
  -- S is on PQ
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ S = (t * P.1 + (1 - t) * Q.1, t * P.2 + (1 - t) * Q.2) →
  -- Circle with center O is tangent to coordinate axes
  O.1 = r ∧ O.2 = r →
  -- Circle is tangent to PQ at T
  (T.1 - O.1)^2 + (T.2 - O.2)^2 = r^2 →
  ∃ s : ℝ, 0 ≤ s ∧ s ≤ 1 ∧ T = (s * P.1 + (1 - s) * Q.1, s * P.2 + (1 - s) * Q.2) →
  -- U is on x-axis and circle is tangent at U
  U.2 = 0 ∧ (U.1 - O.1)^2 + (U.2 - O.2)^2 = r^2 →
  -- The radius of the circle is 2 + √2
  r = 2 + Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_circle_radius_isosceles_right_triangle_l1_108


namespace NUMINAMATH_CALUDE_cloth_sale_problem_l1_162

/-- Represents the problem of determining the number of metres of cloth sold --/
theorem cloth_sale_problem (total_selling_price : ℕ) (loss_per_metre : ℕ) (cost_price_per_metre : ℕ) :
  total_selling_price = 18000 →
  loss_per_metre = 5 →
  cost_price_per_metre = 95 →
  (total_selling_price / (cost_price_per_metre - loss_per_metre) : ℕ) = 200 := by
  sorry

#check cloth_sale_problem

end NUMINAMATH_CALUDE_cloth_sale_problem_l1_162


namespace NUMINAMATH_CALUDE_seth_sold_78_candy_bars_l1_159

def max_candy_bars : ℕ := 24

def seth_candy_bars : ℕ := 3 * max_candy_bars + 6

theorem seth_sold_78_candy_bars : seth_candy_bars = 78 := by
  sorry

end NUMINAMATH_CALUDE_seth_sold_78_candy_bars_l1_159


namespace NUMINAMATH_CALUDE_orlans_rope_problem_l1_195

theorem orlans_rope_problem (total_length : ℝ) (allan_portion : ℝ) (jack_portion : ℝ) (remaining : ℝ) :
  total_length = 20 →
  jack_portion = (2/3) * (total_length - allan_portion) →
  remaining = 5 →
  total_length = allan_portion + jack_portion + remaining →
  allan_portion / total_length = 1/4 :=
by sorry

end NUMINAMATH_CALUDE_orlans_rope_problem_l1_195


namespace NUMINAMATH_CALUDE_distributive_analogy_l1_125

theorem distributive_analogy (a b c : ℝ) (h : c ≠ 0) :
  (a + b) * c = a * c + b * c ↔ (a + b) / c = a / c + b / c :=
sorry

end NUMINAMATH_CALUDE_distributive_analogy_l1_125


namespace NUMINAMATH_CALUDE_fraction_value_l1_114

theorem fraction_value (x : ℝ) (h : x + 1/x = 3) : 
  x^2 / (x^4 + x^2 + 1) = 1/8 := by sorry

end NUMINAMATH_CALUDE_fraction_value_l1_114


namespace NUMINAMATH_CALUDE_arithmetic_sequence_a6_l1_134

/-- Given an arithmetic sequence {aₙ} with a₁ = 2 and S₃ = 12, prove that a₆ = 12 -/
theorem arithmetic_sequence_a6 (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (∀ n, a (n + 1) - a n = a 2 - a 1) →  -- arithmetic sequence condition
  a 1 = 2 →                             -- a₁ = 2
  S 3 = 12 →                            -- S₃ = 12
  a 6 = 12 := by
sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_a6_l1_134


namespace NUMINAMATH_CALUDE_ball_path_on_5x2_table_l1_152

/-- A rectangular table with integer dimensions -/
structure RectTable where
  length : ℕ
  width : ℕ

/-- The path of a ball on a rectangular table -/
def BallPath (table : RectTable) :=
  { bounces : ℕ // bounces ≤ table.length + table.width }

/-- Theorem: A ball on a 5x2 table reaches the opposite corner in 5 bounces -/
theorem ball_path_on_5x2_table :
  ∀ (table : RectTable),
    table.length = 5 →
    table.width = 2 →
    ∃ (path : BallPath table),
      path.val = 5 ∧
      (∀ (other_path : BallPath table), other_path.val ≥ 5) :=
sorry

end NUMINAMATH_CALUDE_ball_path_on_5x2_table_l1_152


namespace NUMINAMATH_CALUDE_eight_sum_product_theorem_l1_109

theorem eight_sum_product_theorem : 
  ∃ (a b c d e f g h : ℤ), 
    (a + b + c + d + e + f + g + h = 8) ∧ 
    (a * b * c * d * e * f * g * h = 8) :=
sorry

end NUMINAMATH_CALUDE_eight_sum_product_theorem_l1_109


namespace NUMINAMATH_CALUDE_distance_to_work_l1_166

/-- Proves that the distance from home to work is 10 km given the conditions --/
theorem distance_to_work (outbound_speed return_speed : ℝ) (distance : ℝ) : 
  return_speed = 2 * outbound_speed →
  distance / outbound_speed + distance / return_speed = 6 →
  return_speed = 5 →
  distance = 10 := by
sorry

end NUMINAMATH_CALUDE_distance_to_work_l1_166


namespace NUMINAMATH_CALUDE_rhombus_area_l1_190

/-- The area of a rhombus with side length 20 cm and an angle of 60 degrees between two adjacent sides is 200√3 cm². -/
theorem rhombus_area (side : ℝ) (angle : ℝ) (h1 : side = 20) (h2 : angle = π / 3) :
  side * side * Real.sin angle = 200 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_area_l1_190


namespace NUMINAMATH_CALUDE_quarter_to_fourth_power_decimal_l1_185

theorem quarter_to_fourth_power_decimal : (1 / 4 : ℝ) ^ 4 = 0.00390625 := by sorry

end NUMINAMATH_CALUDE_quarter_to_fourth_power_decimal_l1_185


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l1_175

theorem imaginary_part_of_z (z : ℂ) : (3 - 4*I)*z = Complex.abs (4 + 3*I) → Complex.im z = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l1_175


namespace NUMINAMATH_CALUDE_sum_of_fractions_l1_158

theorem sum_of_fractions : (2 : ℚ) / 5 + (3 : ℚ) / 11 = (37 : ℚ) / 55 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_l1_158


namespace NUMINAMATH_CALUDE_homework_pages_l1_106

theorem homework_pages (math_pages reading_pages total_pages : ℕ) : 
  math_pages = 10 ∧ 
  reading_pages = math_pages + 3 →
  total_pages = math_pages + reading_pages →
  total_pages = 23 := by
sorry

end NUMINAMATH_CALUDE_homework_pages_l1_106


namespace NUMINAMATH_CALUDE_composite_blackboard_theorem_l1_161

def is_composite (n : ℕ) : Prop := ∃ a b, 1 < a ∧ 1 < b ∧ n = a * b

def proper_divisor (d n : ℕ) : Prop := d ∣ n ∧ 1 < d ∧ d < n

def blackboard_numbers (n : ℕ) : Set ℕ :=
  {x | ∃ d, proper_divisor d n ∧ x = d + 1}

theorem composite_blackboard_theorem (n : ℕ) :
  is_composite n →
  (∃ m, blackboard_numbers n = {x | proper_divisor x m}) ↔
  n = 4 ∨ n = 8 := by
  sorry

end NUMINAMATH_CALUDE_composite_blackboard_theorem_l1_161


namespace NUMINAMATH_CALUDE_sum_a7_a9_eq_zero_l1_142

theorem sum_a7_a9_eq_zero (a : ℕ+ → ℤ) 
  (h : ∀ n : ℕ+, a n = 3 * n.val - 24) : 
  a 7 + a 9 = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_a7_a9_eq_zero_l1_142


namespace NUMINAMATH_CALUDE_inequality_and_equality_condition_l1_194

theorem inequality_and_equality_condition (a b c d : ℝ) 
  (h : a^2 + b^2 + c^2 + d^2 = 4) : 
  (a + 2) * (b + 2) ≥ c * d ∧ 
  (∃ (a₀ b₀ c₀ d₀ : ℝ), a₀^2 + b₀^2 + c₀^2 + d₀^2 = 4 ∧ 
    (a₀ + 2) * (b₀ + 2) = c₀ * d₀ ∧ 
    a₀ = -1 ∧ b₀ = -1 ∧ c₀ = -1 ∧ d₀ = -1) := by
  sorry

end NUMINAMATH_CALUDE_inequality_and_equality_condition_l1_194
