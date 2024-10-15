import Mathlib

namespace NUMINAMATH_CALUDE_marble_prism_weight_l153_15302

/-- Calculates the weight of a rectangular prism with a square base -/
def weight_rectangular_prism (height : ℝ) (base_side : ℝ) (density : ℝ) : ℝ :=
  height * base_side * base_side * density

/-- Proves that the weight of the given marble rectangular prism is 86400 kg -/
theorem marble_prism_weight :
  weight_rectangular_prism 8 2 2700 = 86400 := by
  sorry

end NUMINAMATH_CALUDE_marble_prism_weight_l153_15302


namespace NUMINAMATH_CALUDE_total_cost_after_rebate_l153_15303

def polo_shirt_price : ℕ := 26
def polo_shirt_quantity : ℕ := 3
def necklace_price : ℕ := 83
def necklace_quantity : ℕ := 2
def computer_game_price : ℕ := 90
def computer_game_quantity : ℕ := 1
def rebate : ℕ := 12

theorem total_cost_after_rebate :
  (polo_shirt_price * polo_shirt_quantity +
   necklace_price * necklace_quantity +
   computer_game_price * computer_game_quantity) - rebate = 322 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_after_rebate_l153_15303


namespace NUMINAMATH_CALUDE_greatest_integer_fraction_inequality_l153_15328

theorem greatest_integer_fraction_inequality :
  ∀ y : ℤ, (8 : ℚ) / 11 > (y : ℚ) / 17 ↔ y ≤ 12 :=
by sorry

end NUMINAMATH_CALUDE_greatest_integer_fraction_inequality_l153_15328


namespace NUMINAMATH_CALUDE_perfect_square_factors_of_7200_eq_12_l153_15383

/-- The number of factors of 7200 that are perfect squares -/
def perfect_square_factors_of_7200 : ℕ :=
  let n := 7200
  let factorization := [(2, 4), (3, 2), (5, 2)]
  (List.map (fun (p : ℕ × ℕ) => (p.2 / 2 + 1)) factorization).prod

/-- Theorem stating that the number of factors of 7200 that are perfect squares is 12 -/
theorem perfect_square_factors_of_7200_eq_12 :
  perfect_square_factors_of_7200 = 12 := by sorry

end NUMINAMATH_CALUDE_perfect_square_factors_of_7200_eq_12_l153_15383


namespace NUMINAMATH_CALUDE_instantaneous_velocity_at_2_l153_15304

-- Define the motion equation
def s (t : ℝ) : ℝ := 3 + t^2

-- Define the velocity function as the derivative of s
def v (t : ℝ) : ℝ := 2 * t

-- Theorem statement
theorem instantaneous_velocity_at_2 :
  v 2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_instantaneous_velocity_at_2_l153_15304


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l153_15355

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 - 2*x + 3 ≥ 0) ↔ (∃ x : ℝ, x^2 - 2*x + 3 < 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l153_15355


namespace NUMINAMATH_CALUDE_geometric_progression_ratio_equation_l153_15390

theorem geometric_progression_ratio_equation 
  (x y z r : ℝ) 
  (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) 
  (hdistinct : x ≠ y ∧ y ≠ z ∧ x ≠ z) 
  (hgp : ∃ a : ℝ, a ≠ 0 ∧ 
    y * (z + x) = r * (x * (y + z)) ∧ 
    z * (x + y) = r * (y * (z + x))) : 
  r^2 + r + 1 = 0 := by
sorry

end NUMINAMATH_CALUDE_geometric_progression_ratio_equation_l153_15390


namespace NUMINAMATH_CALUDE_tangent_line_at_M_l153_15343

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x + 1)^2 + (y - 1)^2 = 1

-- Define point M on the circle and the line y = -x
def point_M (x y : ℝ) : Prop := circle_C x y ∧ y = -x

-- Define the equation of the tangent line
def tangent_line (x y : ℝ) : Prop := y = x + 2 - Real.sqrt 2

-- Theorem statement
theorem tangent_line_at_M :
  ∀ x y : ℝ, point_M x y → tangent_line x y :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_at_M_l153_15343


namespace NUMINAMATH_CALUDE_mean_reading_days_l153_15332

def reading_data : List (Nat × Nat) := [
  (2, 1), (4, 2), (5, 3), (10, 4), (7, 5), (3, 6), (2, 7)
]

def total_days : Nat := (reading_data.map (λ (students, days) => students * days)).sum

def total_students : Nat := (reading_data.map (λ (students, _) => students)).sum

theorem mean_reading_days : 
  (total_days : ℚ) / (total_students : ℚ) = 4 := by sorry

end NUMINAMATH_CALUDE_mean_reading_days_l153_15332


namespace NUMINAMATH_CALUDE_min_pages_per_day_l153_15317

theorem min_pages_per_day (total_pages : ℕ) (days_in_week : ℕ) : 
  total_pages = 220 → days_in_week = 7 → 
  ∃ (min_pages : ℕ), 
    min_pages * days_in_week ≥ total_pages ∧ 
    ∀ (x : ℕ), x * days_in_week ≥ total_pages → x ≥ min_pages ∧
    min_pages = 32 := by
  sorry

end NUMINAMATH_CALUDE_min_pages_per_day_l153_15317


namespace NUMINAMATH_CALUDE_ring_toss_total_earnings_l153_15361

/-- The ring toss game's earnings over a period of days -/
def ring_toss_earnings (days : ℕ) (daily_income : ℕ) : ℕ :=
  days * daily_income

/-- Theorem: The ring toss game's total earnings over 3 days at $140 per day is $420 -/
theorem ring_toss_total_earnings : ring_toss_earnings 3 140 = 420 := by
  sorry

end NUMINAMATH_CALUDE_ring_toss_total_earnings_l153_15361


namespace NUMINAMATH_CALUDE_man_ownership_proof_l153_15368

/-- The fraction of the business owned by the man -/
def man_ownership : ℚ := 2/3

/-- The value of the entire business in rupees -/
def business_value : ℕ := 60000

/-- The amount received from selling 3/4 of the man's shares in rupees -/
def sale_amount : ℕ := 30000

/-- The fraction of the man's shares that were sold -/
def sold_fraction : ℚ := 3/4

theorem man_ownership_proof :
  man_ownership * sold_fraction * business_value = sale_amount :=
sorry

end NUMINAMATH_CALUDE_man_ownership_proof_l153_15368


namespace NUMINAMATH_CALUDE_shoe_selection_probability_l153_15326

/-- The number of pairs of shoes -/
def num_pairs : ℕ := 5

/-- The total number of shoes -/
def total_shoes : ℕ := 2 * num_pairs

/-- The number of shoes to be selected -/
def selected_shoes : ℕ := 2

/-- The number of ways to select 2 shoes of the same color -/
def same_color_selections : ℕ := num_pairs

/-- The total number of ways to select 2 shoes from 10 shoes -/
def total_selections : ℕ := Nat.choose total_shoes selected_shoes

theorem shoe_selection_probability :
  same_color_selections / total_selections = 1 / 9 := by sorry

end NUMINAMATH_CALUDE_shoe_selection_probability_l153_15326


namespace NUMINAMATH_CALUDE_w_squared_value_l153_15325

theorem w_squared_value (w : ℝ) (h : 2 * (w + 15)^2 = (4 * w + 9) * (3 * w + 6)) :
  w^2 = (9 + Real.sqrt 15921) / 20 := by
  sorry

end NUMINAMATH_CALUDE_w_squared_value_l153_15325


namespace NUMINAMATH_CALUDE_lines_coplanar_iff_k_eq_neg_half_l153_15391

/-- First line parameterization --/
def line1 (s : ℝ) (k : ℝ) : ℝ × ℝ × ℝ := (2 + s, 4 - k*s, -1 + k*s)

/-- Second line parameterization --/
def line2 (t : ℝ) : ℝ × ℝ × ℝ := (2*t, 2 + t, 3 - t)

/-- Direction vector of the first line --/
def dir1 (k : ℝ) : ℝ × ℝ × ℝ := (1, -k, k)

/-- Direction vector of the second line --/
def dir2 : ℝ × ℝ × ℝ := (2, 1, -1)

/-- Two lines are coplanar if and only if k = -1/2 --/
theorem lines_coplanar_iff_k_eq_neg_half :
  (∃ (a b : ℝ), a • dir1 k + b • dir2 = (0, 0, 0)) ↔ k = -1/2 := by sorry

end NUMINAMATH_CALUDE_lines_coplanar_iff_k_eq_neg_half_l153_15391


namespace NUMINAMATH_CALUDE_angle_E_measure_l153_15308

-- Define the quadrilateral EFGH
structure Quadrilateral :=
  (E F G H : ℝ)

-- Define the conditions for the quadrilateral
def is_special_quadrilateral (q : Quadrilateral) : Prop :=
  q.E = 3 * q.F ∧ q.E = 4 * q.G ∧ q.E = 6 * q.H ∧
  q.E + q.F + q.G + q.H = 360

-- Theorem statement
theorem angle_E_measure (q : Quadrilateral) 
  (h : is_special_quadrilateral q) : 
  205 < q.E ∧ q.E < 206 :=
sorry

end NUMINAMATH_CALUDE_angle_E_measure_l153_15308


namespace NUMINAMATH_CALUDE_square_roots_theorem_l153_15388

theorem square_roots_theorem (a : ℝ) :
  (∃ x : ℝ, x^2 = (a + 3)^2 ∧ x^2 = (2*a - 9)^2) →
  (a + 3)^2 = 25 :=
by sorry

end NUMINAMATH_CALUDE_square_roots_theorem_l153_15388


namespace NUMINAMATH_CALUDE_cube_root_equation_solution_l153_15371

theorem cube_root_equation_solution :
  ∃! x : ℝ, (5 - x / 3) ^ (1/3 : ℝ) = -4 :=
by
  sorry

end NUMINAMATH_CALUDE_cube_root_equation_solution_l153_15371


namespace NUMINAMATH_CALUDE_least_positive_slope_line_l153_15309

/-- The curve equation -/
def curve (x y : ℝ) : Prop := 4 * x^2 - y^2 - 8 * x = 12

/-- The line equation -/
def line (m : ℝ) (x y : ℝ) : Prop := y = m * x - m

/-- The line contains the point (1, 0) -/
def contains_point (m : ℝ) : Prop := line m 1 0

/-- The line does not intersect the curve -/
def no_intersection (m : ℝ) : Prop := ∀ x y : ℝ, line m x y → ¬curve x y

/-- The slope is positive -/
def positive_slope (m : ℝ) : Prop := m > 0

theorem least_positive_slope_line :
  ∃ m : ℝ, m = 2 ∧
    contains_point m ∧
    no_intersection m ∧
    positive_slope m ∧
    ∀ m' : ℝ, m' ≠ m → contains_point m' → no_intersection m' → positive_slope m' → m' > m :=
sorry

end NUMINAMATH_CALUDE_least_positive_slope_line_l153_15309


namespace NUMINAMATH_CALUDE_student_take_home_pay_l153_15358

/-- Calculates the take-home pay for a well-performing student at a fast-food chain --/
def takeHomePay (baseSalary : ℝ) (bonus : ℝ) (taxRate : ℝ) : ℝ :=
  let totalEarnings := baseSalary + bonus
  let taxAmount := totalEarnings * taxRate
  totalEarnings - taxAmount

/-- Theorem: The take-home pay for a well-performing student is 26,100 rubles --/
theorem student_take_home_pay :
  takeHomePay 25000 5000 0.13 = 26100 := by
  sorry

#eval takeHomePay 25000 5000 0.13

end NUMINAMATH_CALUDE_student_take_home_pay_l153_15358


namespace NUMINAMATH_CALUDE_greatest_n_for_inequality_l153_15381

theorem greatest_n_for_inequality (n : ℤ) (h : 101 * n^2 ≤ 3600) : n ≤ 5 ∧ ∃ (m : ℤ), m = 5 ∧ 101 * m^2 ≤ 3600 :=
sorry

end NUMINAMATH_CALUDE_greatest_n_for_inequality_l153_15381


namespace NUMINAMATH_CALUDE_notebook_difference_l153_15337

theorem notebook_difference (price : ℚ) (mika_count leo_count : ℕ) : 
  price > (1 / 10 : ℚ) →
  price * mika_count = (12 / 5 : ℚ) →
  price * leo_count = (16 / 5 : ℚ) →
  leo_count - mika_count = 4 := by
  sorry

#check notebook_difference

end NUMINAMATH_CALUDE_notebook_difference_l153_15337


namespace NUMINAMATH_CALUDE_candies_per_friend_l153_15353

/-- Given 36 candies shared equally among 9 friends, prove that each friend receives 4 candies. -/
theorem candies_per_friend (total_candies : ℕ) (num_friends : ℕ) (candies_per_friend : ℕ) :
  total_candies = 36 →
  num_friends = 9 →
  candies_per_friend = total_candies / num_friends →
  candies_per_friend = 4 := by
  sorry

end NUMINAMATH_CALUDE_candies_per_friend_l153_15353


namespace NUMINAMATH_CALUDE_optimal_purchase_plan_l153_15360

/-- Represents the purchase and selling prices of keychains --/
structure KeychainPrices where
  purchase_a : ℕ
  purchase_b : ℕ
  selling_a : ℕ
  selling_b : ℕ

/-- Represents the purchase plan for keychains --/
structure PurchasePlan where
  quantity_a : ℕ
  quantity_b : ℕ

/-- Calculates the total purchase cost for a given plan --/
def total_purchase_cost (prices : KeychainPrices) (plan : PurchasePlan) : ℕ :=
  prices.purchase_a * plan.quantity_a + prices.purchase_b * plan.quantity_b

/-- Calculates the total profit for a given plan --/
def total_profit (prices : KeychainPrices) (plan : PurchasePlan) : ℕ :=
  (prices.selling_a - prices.purchase_a) * plan.quantity_a +
  (prices.selling_b - prices.purchase_b) * plan.quantity_b

/-- Theorem: The optimal purchase plan maximizes profit --/
theorem optimal_purchase_plan (prices : KeychainPrices)
  (h_prices : prices.purchase_a = 30 ∧ prices.purchase_b = 25 ∧
              prices.selling_a = 45 ∧ prices.selling_b = 37) :
  ∃ (plan : PurchasePlan),
    plan.quantity_a + plan.quantity_b = 80 ∧
    total_purchase_cost prices plan ≤ 2200 ∧
    total_profit prices plan = 1080 ∧
    ∀ (other_plan : PurchasePlan),
      other_plan.quantity_a + other_plan.quantity_b = 80 →
      total_purchase_cost prices other_plan ≤ 2200 →
      total_profit prices other_plan ≤ total_profit prices plan :=
sorry

end NUMINAMATH_CALUDE_optimal_purchase_plan_l153_15360


namespace NUMINAMATH_CALUDE_ellipse_problem_l153_15310

-- Define the points F₁ and F₂
def F₁ : ℝ × ℝ := (-1, 0)
def F₂ : ℝ × ℝ := (1, 0)

-- Define the distance condition for point P
def distance_condition (P : ℝ × ℝ) : Prop :=
  Real.sqrt ((P.1 - F₁.1)^2 + (P.2 - F₁.2)^2) +
  Real.sqrt ((P.1 - F₂.1)^2 + (P.2 - F₂.2)^2) = 4

-- Define the trajectory C
def trajectory_C (x y : ℝ) : Prop := x^2/4 + y^2/3 = 1

-- Define the line l
def line_l (x y : ℝ) : Prop := y = 2 * (x + 1)

-- Define the perpendicular condition
def perpendicular_condition (M A : ℝ × ℝ) : Prop :=
  (A.2 - M.2) = -1/2 * (A.1 - M.1)

-- Theorem statement
theorem ellipse_problem :
  ∀ (P : ℝ × ℝ), distance_condition P →
  (∀ x y, trajectory_C x y ↔ (x, y) = P) ∧
  (∃ (M : ℝ × ℝ), trajectory_C M.1 M.2 ∧
    ∀ (A : ℝ × ℝ), line_l A.1 A.2 ∧ perpendicular_condition M A →
    Real.sqrt ((A.1 - F₁.1)^2 + (A.2 - F₁.2)^2) ≤ Real.sqrt 5) ∧
  (∃ (M : ℝ × ℝ), M = (1, 3/2) ∧ trajectory_C M.1 M.2 ∧
    ∃ (A : ℝ × ℝ), line_l A.1 A.2 ∧ perpendicular_condition M A ∧
    Real.sqrt ((A.1 - F₁.1)^2 + (A.2 - F₁.2)^2) = Real.sqrt 5) :=
sorry

end NUMINAMATH_CALUDE_ellipse_problem_l153_15310


namespace NUMINAMATH_CALUDE_selection_methods_count_l153_15377

/-- The number of ways to select k items from n items -/
def binomial (n k : ℕ) : ℕ := sorry

/-- The number of boys in the class -/
def num_boys : ℕ := 4

/-- The number of girls in the class -/
def num_girls : ℕ := 2

/-- The total number of students to be selected -/
def num_selected : ℕ := 4

/-- The number of ways to select 4 members from 4 boys and 2 girls, with at least 1 girl -/
def num_selections : ℕ := 
  binomial num_girls 1 * binomial num_boys 3 + 
  binomial num_girls 2 * binomial num_boys 2

theorem selection_methods_count : num_selections = 14 := by sorry

end NUMINAMATH_CALUDE_selection_methods_count_l153_15377


namespace NUMINAMATH_CALUDE_triangle_properties_l153_15387

theorem triangle_properties (a b c : ℝ) (A B C : ℝ) :
  -- Triangle ABC exists
  0 < a ∧ 0 < b ∧ 0 < c ∧ 
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π →
  -- Given conditions
  b * (1 + Real.cos C) = c * (2 - Real.cos B) →
  C = π / 3 →
  1/2 * a * b * Real.sin C = 4 * Real.sqrt 3 →
  -- Conclusions to prove
  (a + b = 2 * c ∧ c = 4) := by
  sorry


end NUMINAMATH_CALUDE_triangle_properties_l153_15387


namespace NUMINAMATH_CALUDE_certain_to_select_genuine_l153_15339

/-- A set of products with genuine and defective items -/
structure ProductSet where
  total : ℕ
  genuine : ℕ
  defective : ℕ
  h1 : genuine + defective = total

/-- The number of products to be selected -/
def selection_size : ℕ := 3

/-- The specific product set in the problem -/
def problem_set : ProductSet where
  total := 12
  genuine := 10
  defective := 2
  h1 := by rfl

/-- The probability of selecting at least one genuine product -/
def prob_at_least_one_genuine (ps : ProductSet) : ℚ :=
  1 - (Nat.choose ps.defective selection_size : ℚ) / (Nat.choose ps.total selection_size : ℚ)

theorem certain_to_select_genuine :
  prob_at_least_one_genuine problem_set = 1 := by
  sorry

end NUMINAMATH_CALUDE_certain_to_select_genuine_l153_15339


namespace NUMINAMATH_CALUDE_fruit_arrangement_theorem_l153_15370

def number_of_arrangements (total : ℕ) (group1 : ℕ) (group2 : ℕ) (group3 : ℕ) : ℕ :=
  Nat.factorial total / (Nat.factorial group1 * Nat.factorial group2 * Nat.factorial group3)

theorem fruit_arrangement_theorem :
  number_of_arrangements 7 4 2 1 = 105 := by
  sorry

end NUMINAMATH_CALUDE_fruit_arrangement_theorem_l153_15370


namespace NUMINAMATH_CALUDE_gcd_1617_1225_l153_15349

theorem gcd_1617_1225 : Nat.gcd 1617 1225 = 49 := by sorry

end NUMINAMATH_CALUDE_gcd_1617_1225_l153_15349


namespace NUMINAMATH_CALUDE_numerator_greater_than_denominator_l153_15356

theorem numerator_greater_than_denominator (x : ℝ) :
  -1 ≤ x ∧ x ≤ 3 ∧ 4 * x - 3 > 9 - 2 * x → 2 < x ∧ x ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_numerator_greater_than_denominator_l153_15356


namespace NUMINAMATH_CALUDE_marked_up_percentage_l153_15398

theorem marked_up_percentage 
  (cost_price selling_price : ℝ)
  (discount_percentage : ℝ)
  (h1 : cost_price = 540)
  (h2 : selling_price = 456)
  (h3 : discount_percentage = 26.570048309178745) :
  (((selling_price / (1 - discount_percentage / 100) - cost_price) / cost_price) * 100 = 15) :=
by sorry

end NUMINAMATH_CALUDE_marked_up_percentage_l153_15398


namespace NUMINAMATH_CALUDE_rectangle_b_product_l153_15359

theorem rectangle_b_product : ∀ b₁ b₂ : ℝ,
  (∃ (rect : Set (ℝ × ℝ)), 
    rect = {(x, y) | 3 ≤ y ∧ y ≤ 8 ∧ ((x = 2 ∧ b₁ ≤ x) ∨ (x = b₁ ∧ x ≤ 2)) ∧
            ((x = 2 ∧ x ≤ b₂) ∨ (x = b₂ ∧ 2 ≤ x))} ∧
    (∀ (p q : ℝ × ℝ), p ∈ rect ∧ q ∈ rect → 
      (p.1 = q.1 ∨ p.2 = q.2) ∧ 
      (p.1 ≠ q.1 ∨ p.2 ≠ q.2))) →
  b₁ * b₂ = -21 :=
by sorry


end NUMINAMATH_CALUDE_rectangle_b_product_l153_15359


namespace NUMINAMATH_CALUDE_polynomial_expansion_l153_15362

theorem polynomial_expansion (a : ℝ) :
  (a + 1) * (a + 3) * (a + 4) * (a + 5) * (a + 6) =
  a^5 + 19*a^4 + 137*a^3 + 461*a^2 + 702*a + 360 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_expansion_l153_15362


namespace NUMINAMATH_CALUDE_simplify_expression_l153_15341

theorem simplify_expression (a b : ℝ) : (2*a - b) - (2*a + b) = -2*b := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l153_15341


namespace NUMINAMATH_CALUDE_product_of_difference_of_squares_l153_15392

theorem product_of_difference_of_squares (a b x1 y1 x2 y2 : ℤ) 
  (ha : a = x1^2 - 5*y1^2) (hb : b = x2^2 - 5*y2^2) :
  ∃ u v : ℤ, a * b = u^2 - 5*v^2 := by
sorry

end NUMINAMATH_CALUDE_product_of_difference_of_squares_l153_15392


namespace NUMINAMATH_CALUDE_chemistry_class_b_count_l153_15364

/-- Represents the number of students who earn each grade in a chemistry class. -/
structure GradeDistribution where
  a : ℝ  -- Number of students earning A
  b : ℝ  -- Number of students earning B
  c : ℝ  -- Number of students earning C
  d : ℝ  -- Number of students earning D

/-- The grade distribution in a chemistry class of 50 students satisfies given probability ratios. -/
def chemistryClass (g : GradeDistribution) : Prop :=
  g.a = 0.5 * g.b ∧
  g.c = 1.2 * g.b ∧
  g.d = 0.3 * g.b ∧
  g.a + g.b + g.c + g.d = 50

/-- The number of students earning a B in the chemistry class is 50/3. -/
theorem chemistry_class_b_count :
  ∀ g : GradeDistribution, chemistryClass g → g.b = 50 / 3 := by
  sorry

end NUMINAMATH_CALUDE_chemistry_class_b_count_l153_15364


namespace NUMINAMATH_CALUDE_cattle_transport_time_l153_15351

/-- Calculates the total driving time to transport cattle to higher ground -/
def total_driving_time (total_cattle : ℕ) (distance : ℕ) (truck_capacity : ℕ) (speed : ℕ) : ℕ :=
  let num_trips := total_cattle / truck_capacity
  let round_trip_time := 2 * (distance / speed)
  num_trips * round_trip_time

/-- Theorem stating that under given conditions, the total driving time is 40 hours -/
theorem cattle_transport_time : total_driving_time 400 60 20 60 = 40 := by
  sorry

end NUMINAMATH_CALUDE_cattle_transport_time_l153_15351


namespace NUMINAMATH_CALUDE_covid_cases_after_growth_l153_15373

/-- Calculates the total number of COVID-19 cases in New York, California, and Texas after one month of growth --/
theorem covid_cases_after_growth (new_york_initial : ℕ) 
  (h1 : new_york_initial = 2000)
  (h2 : ∃ california_initial : ℕ, california_initial = new_york_initial / 2)
  (h3 : ∃ texas_initial : ℕ, ∃ california_initial : ℕ, 
    california_initial = new_york_initial / 2 ∧ 
    california_initial = texas_initial + 400)
  (h4 : ∃ new_york_growth : ℚ, new_york_growth = 25 / 100)
  (h5 : ∃ california_growth : ℚ, california_growth = 15 / 100)
  (h6 : ∃ texas_growth : ℚ, texas_growth = 30 / 100) :
  ∃ total_cases : ℕ, total_cases = 4430 := by
sorry

end NUMINAMATH_CALUDE_covid_cases_after_growth_l153_15373


namespace NUMINAMATH_CALUDE_product_mod_twenty_l153_15323

theorem product_mod_twenty : (93 * 68 * 105) % 20 = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_mod_twenty_l153_15323


namespace NUMINAMATH_CALUDE_probability_sum_12_l153_15334

/-- The number of faces on each die -/
def numFaces : ℕ := 6

/-- The target sum we're aiming for -/
def targetSum : ℕ := 12

/-- The number of dice rolled -/
def numDice : ℕ := 3

/-- The total number of possible outcomes when rolling three dice -/
def totalOutcomes : ℕ := numFaces ^ numDice

/-- The number of favorable outcomes (sum of 12) -/
def favorableOutcomes : ℕ := 10

/-- The probability of rolling a sum of 12 with three standard six-sided dice -/
theorem probability_sum_12 : 
  (favorableOutcomes : ℚ) / totalOutcomes = 10 / 216 := by sorry

end NUMINAMATH_CALUDE_probability_sum_12_l153_15334


namespace NUMINAMATH_CALUDE_product_equality_l153_15399

noncomputable def P : ℝ := Real.sqrt 1011 + Real.sqrt 1012
noncomputable def Q : ℝ := -Real.sqrt 1011 - Real.sqrt 1012
noncomputable def R : ℝ := Real.sqrt 1011 - Real.sqrt 1012
noncomputable def S : ℝ := Real.sqrt 1012 - Real.sqrt 1011

theorem product_equality : (P * Q)^2 * R * S = 8136957 := by
  sorry

end NUMINAMATH_CALUDE_product_equality_l153_15399


namespace NUMINAMATH_CALUDE_metro_earnings_proof_l153_15379

/-- Calculates the earnings from ticket sales given the ticket price, average tickets sold per minute, and duration in minutes. -/
def calculate_earnings (ticket_price : ℝ) (tickets_per_minute : ℝ) (duration : ℝ) : ℝ :=
  ticket_price * tickets_per_minute * duration

/-- Proves that the earnings from ticket sales for the given conditions equal $90. -/
theorem metro_earnings_proof (ticket_price : ℝ) (tickets_per_minute : ℝ) (duration : ℝ)
  (h1 : ticket_price = 3)
  (h2 : tickets_per_minute = 5)
  (h3 : duration = 6) :
  calculate_earnings ticket_price tickets_per_minute duration = 90 := by
  sorry

end NUMINAMATH_CALUDE_metro_earnings_proof_l153_15379


namespace NUMINAMATH_CALUDE_length_PQ_is_two_l153_15307

/-- Represents a point in polar coordinates -/
structure PolarPoint where
  ρ : ℝ
  θ : ℝ

/-- Represents a line in parametric form -/
structure ParametricLine where
  x : ℝ → ℝ
  y : ℝ → ℝ

/-- Represents a circle in polar form -/
structure PolarCircle where
  equation : ℝ → ℝ → Prop

/-- Represents a ray in polar form -/
structure PolarRay where
  θ : ℝ

theorem length_PQ_is_two 
  (l : ParametricLine)
  (C : PolarCircle)
  (OM : PolarRay)
  (h1 : l.x = fun t ↦ -1/2 * t)
  (h2 : l.y = fun t ↦ 3 * Real.sqrt 3 + (Real.sqrt 3 / 2) * t)
  (h3 : C.equation = fun ρ θ ↦ ρ = 2 * Real.cos θ)
  (h4 : OM.θ = π / 3)
  (P : PolarPoint)
  (Q : PolarPoint)
  (h5 : C.equation P.ρ P.θ)
  (h6 : P.θ = OM.θ)
  (h7 : Q.θ = OM.θ)
  (h8 : Real.sqrt 3 * Q.ρ * Real.cos Q.θ + Q.ρ * Real.sin Q.θ - 3 * Real.sqrt 3 = 0) :
  abs (P.ρ - Q.ρ) = 2 := by
sorry


end NUMINAMATH_CALUDE_length_PQ_is_two_l153_15307


namespace NUMINAMATH_CALUDE_equation_solutions_l153_15365

theorem equation_solutions :
  (∃ x : ℚ, 3 * x - 8 = x + 4 ∧ x = 6) ∧
  (∃ x : ℚ, 1 - 3 * (x + 1) = 2 * (1 - 0.5 * x) ∧ x = -2) ∧
  (∃ x : ℚ, (1 / 6) * (3 * x - 6) = (2 / 5) * x - 3 ∧ x = -20) ∧
  (∃ y : ℚ, (3 * y - 1) / 4 - 1 = (5 * y - 7) / 6 ∧ y = -1) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l153_15365


namespace NUMINAMATH_CALUDE_two_color_no_power_of_two_sum_l153_15319

theorem two_color_no_power_of_two_sum :
  ∃ (f : ℕ → Bool), ∀ (a b : ℕ), a ≠ b → f a = f b → ¬∃ (n : ℕ), a + b = 2^n :=
sorry

end NUMINAMATH_CALUDE_two_color_no_power_of_two_sum_l153_15319


namespace NUMINAMATH_CALUDE_distance_between_points_l153_15352

theorem distance_between_points : 
  let p1 : ℝ × ℝ := (2, 3)
  let p2 : ℝ × ℝ := (5, 9)
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2) = 3 * Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_distance_between_points_l153_15352


namespace NUMINAMATH_CALUDE_square_park_area_l153_15367

/-- The area of a square park with a side length of 30 meters is 900 square meters. -/
theorem square_park_area : 
  ∀ (park_side_length : ℝ), 
  park_side_length = 30 → 
  park_side_length * park_side_length = 900 :=
by
  sorry

end NUMINAMATH_CALUDE_square_park_area_l153_15367


namespace NUMINAMATH_CALUDE_small_panda_bamboo_consumption_l153_15386

/-- The amount of bamboo eaten by small pandas each day -/
def small_panda_bamboo : ℝ := 100

/-- The number of small panda bears -/
def num_small_pandas : ℕ := 4

/-- The number of bigger panda bears -/
def num_big_pandas : ℕ := 5

/-- The amount of bamboo eaten by each bigger panda bear per day -/
def big_panda_bamboo : ℝ := 40

/-- The total amount of bamboo eaten by all pandas in a week -/
def total_weekly_bamboo : ℝ := 2100

theorem small_panda_bamboo_consumption :
  small_panda_bamboo * num_small_pandas +
  big_panda_bamboo * num_big_pandas =
  total_weekly_bamboo / 7 :=
by sorry

end NUMINAMATH_CALUDE_small_panda_bamboo_consumption_l153_15386


namespace NUMINAMATH_CALUDE_belize_homes_without_features_l153_15321

/-- The town of Belize with its home characteristics -/
structure BelizeTown where
  total_homes : ℕ
  white_homes : ℕ
  non_white_homes : ℕ
  non_white_with_fireplace : ℕ
  non_white_with_fireplace_and_basement : ℕ
  non_white_without_fireplace : ℕ
  non_white_without_fireplace_with_garden : ℕ

/-- Properties of the Belize town -/
def belize_properties (t : BelizeTown) : Prop :=
  t.total_homes = 400 ∧
  t.white_homes = t.total_homes / 4 ∧
  t.non_white_homes = t.total_homes - t.white_homes ∧
  t.non_white_with_fireplace = t.non_white_homes / 5 ∧
  t.non_white_with_fireplace_and_basement = t.non_white_with_fireplace / 3 ∧
  t.non_white_without_fireplace = t.non_white_homes - t.non_white_with_fireplace ∧
  t.non_white_without_fireplace_with_garden = t.non_white_without_fireplace / 2

/-- Theorem: The number of non-white homes without fireplace, basement, or garden is 120 -/
theorem belize_homes_without_features (t : BelizeTown) 
  (h : belize_properties t) : 
  t.non_white_without_fireplace - t.non_white_without_fireplace_with_garden = 120 := by
  sorry

end NUMINAMATH_CALUDE_belize_homes_without_features_l153_15321


namespace NUMINAMATH_CALUDE_estimate_passing_papers_l153_15347

/-- Estimates the number of passing papers in a population based on a sample --/
theorem estimate_passing_papers 
  (total_papers : ℕ) 
  (sample_size : ℕ) 
  (passing_in_sample : ℕ) 
  (h1 : total_papers = 10000)
  (h2 : sample_size = 500)
  (h3 : passing_in_sample = 420) :
  ⌊(total_papers : ℝ) * (passing_in_sample : ℝ) / (sample_size : ℝ)⌋ = 8400 :=
sorry

end NUMINAMATH_CALUDE_estimate_passing_papers_l153_15347


namespace NUMINAMATH_CALUDE_max_value_under_constraint_l153_15314

theorem max_value_under_constraint (x y : ℝ) :
  (|x| + |y| ≤ 1) → (x + 2*y ≤ 2) :=
by sorry

end NUMINAMATH_CALUDE_max_value_under_constraint_l153_15314


namespace NUMINAMATH_CALUDE_hyperbola_equation_l153_15324

theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a^2 + b^2 = 4) →
  (b / a = Real.sqrt 3) →
  ∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1 ↔ x^2 - y^2 / 3 = 1 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l153_15324


namespace NUMINAMATH_CALUDE_only_B_is_certain_l153_15380

-- Define the type for events
inductive Event : Type
  | A : Event  -- It will be sunny on New Year's Day in 2020
  | B : Event  -- The sun rises from the east
  | C : Event  -- The TV is turned on and broadcasting the news
  | D : Event  -- Drawing a red ball from a box without any red balls

-- Define what it means for an event to be certain
def is_certain (e : Event) : Prop :=
  match e with
  | Event.B => True
  | _ => False

-- Theorem statement
theorem only_B_is_certain :
  ∀ e : Event, is_certain e ↔ e = Event.B :=
by sorry

end NUMINAMATH_CALUDE_only_B_is_certain_l153_15380


namespace NUMINAMATH_CALUDE_rotational_homothety_similarity_l153_15327

-- Define the rotational homothety transformation
def rotationalHomothety (k : ℝ) (θ : ℝ) (O : ℝ × ℝ) : (ℝ × ℝ) → (ℝ × ℝ) := sorry

-- Define the fourth vertex of a parallelogram
def fourthVertex (O A A₁ : ℝ × ℝ) : ℝ × ℝ := sorry

-- Define similarity of triangles
def trianglesSimilar (A B C A' B' C' : ℝ × ℝ) : Prop := sorry

theorem rotational_homothety_similarity 
  (A B C : ℝ × ℝ) -- Original triangle vertices
  (k : ℝ) (θ : ℝ) (O : ℝ × ℝ) -- Rotational homothety parameters
  (A₁ B₁ C₁ : ℝ × ℝ) -- Transformed triangle vertices
  (A₂ B₂ C₂ : ℝ × ℝ) -- Fourth vertices of parallelograms
  (h₁ : A₁ = rotationalHomothety k θ O A)
  (h₂ : B₁ = rotationalHomothety k θ O B)
  (h₃ : C₁ = rotationalHomothety k θ O C)
  (h₄ : A₂ = fourthVertex O A A₁)
  (h₅ : B₂ = fourthVertex O B B₁)
  (h₆ : C₂ = fourthVertex O C C₁) :
  trianglesSimilar A B C A₂ B₂ C₂ := by sorry

end NUMINAMATH_CALUDE_rotational_homothety_similarity_l153_15327


namespace NUMINAMATH_CALUDE_opposite_sides_difference_equal_l153_15389

/-- An equiangular hexagon with sides a, b, c, d, e, f in order -/
structure EquiangularHexagon where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  e : ℝ
  f : ℝ
  equiangular : True  -- This represents the equiangular property

/-- The differences between opposite sides in an equiangular hexagon are equal -/
theorem opposite_sides_difference_equal (h : EquiangularHexagon) :
  h.a - h.d = h.e - h.b ∧ h.e - h.b = h.c - h.f :=
by sorry

end NUMINAMATH_CALUDE_opposite_sides_difference_equal_l153_15389


namespace NUMINAMATH_CALUDE_athlete_arrangement_count_l153_15397

theorem athlete_arrangement_count : ℕ := by
  -- Define the number of athletes and tracks
  let num_athletes : ℕ := 6
  let num_tracks : ℕ := 6

  -- Define the restrictions for athletes A and B
  let a_possible_tracks : ℕ := 4  -- A can't be on 1st or 2nd track
  let b_possible_tracks : ℕ := 2  -- B must be on 5th or 6th track

  -- Define the number of remaining athletes to be arranged
  let remaining_athletes : ℕ := num_athletes - 2  -- excluding A and B

  -- Calculate the total number of arrangements
  let total_arrangements : ℕ := b_possible_tracks * a_possible_tracks * (Nat.factorial remaining_athletes)

  -- Prove that the total number of arrangements is 144
  sorry

end NUMINAMATH_CALUDE_athlete_arrangement_count_l153_15397


namespace NUMINAMATH_CALUDE_h_satisfies_condition_l153_15305

def f (x : ℝ) : ℝ := 2 * x + 3

def g (x : ℝ) : ℝ := 4 * x - 5

def h (x : ℝ) : ℝ := 2 * x - 4

theorem h_satisfies_condition : ∀ x : ℝ, f (h x) = g x := by
  sorry

end NUMINAMATH_CALUDE_h_satisfies_condition_l153_15305


namespace NUMINAMATH_CALUDE_min_a_2005_l153_15300

theorem min_a_2005 (a : Fin 2005 → ℕ+) 
  (h_increasing : ∀ i j, i < j → a i < a j)
  (h_product : ∀ i j k, i ≠ j → i < 2005 → j < 2005 → k < 2005 → a i * a j ≠ a k) :
  a 2004 ≥ 2048 := by
  sorry

end NUMINAMATH_CALUDE_min_a_2005_l153_15300


namespace NUMINAMATH_CALUDE_bracelet_capacity_is_fifteen_l153_15329

/-- Represents the jewelry store inventory and pricing --/
structure JewelryStore where
  necklace_capacity : ℕ
  current_necklaces : ℕ
  ring_capacity : ℕ
  current_rings : ℕ
  current_bracelets : ℕ
  necklace_price : ℕ
  ring_price : ℕ
  bracelet_price : ℕ
  total_fill_cost : ℕ

/-- Calculates the bracelet display capacity given the store's inventory and pricing --/
def bracelet_display_capacity (store : JewelryStore) : ℕ :=
  store.current_bracelets +
    ((store.total_fill_cost -
      (store.necklace_price * (store.necklace_capacity - store.current_necklaces) +
       store.ring_price * (store.ring_capacity - store.current_rings)))
     / store.bracelet_price)

/-- Theorem stating that for the given store configuration, the bracelet display capacity is 15 --/
theorem bracelet_capacity_is_fifteen :
  let store : JewelryStore := {
    necklace_capacity := 12,
    current_necklaces := 5,
    ring_capacity := 30,
    current_rings := 18,
    current_bracelets := 8,
    necklace_price := 4,
    ring_price := 10,
    bracelet_price := 5,
    total_fill_cost := 183
  }
  bracelet_display_capacity store = 15 := by
  sorry


end NUMINAMATH_CALUDE_bracelet_capacity_is_fifteen_l153_15329


namespace NUMINAMATH_CALUDE_miss_graysons_class_fund_miss_graysons_class_fund_proof_l153_15335

theorem miss_graysons_class_fund (initial_fund : ℕ) (student_contribution : ℕ) (num_students : ℕ) (trip_cost : ℕ) : ℕ :=
  let total_contribution := student_contribution * num_students
  let total_fund := initial_fund + total_contribution
  let total_trip_cost := trip_cost * num_students
  let remaining_fund := total_fund - total_trip_cost
  remaining_fund

theorem miss_graysons_class_fund_proof :
  miss_graysons_class_fund 50 5 20 7 = 10 := by
  sorry

end NUMINAMATH_CALUDE_miss_graysons_class_fund_miss_graysons_class_fund_proof_l153_15335


namespace NUMINAMATH_CALUDE_function_and_tangent_line_l153_15311

/-- Given a function f with the property that 
    f(x) = (1/4) * f'(1) * x^2 + 2 * f(1) * x - 4,
    prove that f(x) = 2x^2 + 4x - 4 and its tangent line
    at (0, f(0)) has the equation 4x - y - 4 = 0 -/
theorem function_and_tangent_line 
  (f : ℝ → ℝ) 
  (h : ∀ x, f x = (1/4) * (deriv f 1) * x^2 + 2 * (f 1) * x - 4) :
  (∀ x, f x = 2*x^2 + 4*x - 4) ∧ 
  (∃ a b c : ℝ, a = 4 ∧ b = -1 ∧ c = -4 ∧ 
    ∀ x y, y = (deriv f 0) * x + f 0 ↔ a*x + b*y + c = 0) :=
by sorry

end NUMINAMATH_CALUDE_function_and_tangent_line_l153_15311


namespace NUMINAMATH_CALUDE_exponent_simplification_l153_15346

theorem exponent_simplification :
  (10 ^ 0.4) * (10 ^ 0.6) * (10 ^ 0.3) * (10 ^ 0.2) * (10 ^ 0.5) = 100 := by
  sorry

end NUMINAMATH_CALUDE_exponent_simplification_l153_15346


namespace NUMINAMATH_CALUDE_centers_form_rectangle_l153_15333

/-- Represents a quadrilateral with side lengths -/
structure Quadrilateral :=
  (a b c d : ℝ)

/-- Represents a point in 2D space -/
structure Point :=
  (x y : ℝ)

/-- Checks if a quadrilateral is inscribed -/
def is_inscribed (q : Quadrilateral) : Prop :=
  sorry

/-- Calculates the center of a rectangle given two adjacent corners -/
def rectangle_center (p1 p2 : Point) (width height : ℝ) : Point :=
  sorry

/-- Checks if four points form a rectangle -/
def is_rectangle (p1 p2 p3 p4 : Point) : Prop :=
  sorry

/-- Main theorem: The centers of rectangles constructed on the sides of an inscribed quadrilateral form a rectangle -/
theorem centers_form_rectangle (q : Quadrilateral) (h : is_inscribed q) :
  let a := q.a
  let b := q.b
  let c := q.c
  let d := q.d
  let A := Point.mk 0 0  -- Arbitrary placement of A
  let B := Point.mk a 0  -- B is a units away from A on x-axis
  let C := sorry         -- C's position depends on the quadrilateral's shape
  let D := sorry         -- D's position depends on the quadrilateral's shape
  let P := rectangle_center A B a c
  let Q := rectangle_center B C b d
  let R := rectangle_center C D c a
  let S := rectangle_center D A d b
  is_rectangle P Q R S :=
sorry

end NUMINAMATH_CALUDE_centers_form_rectangle_l153_15333


namespace NUMINAMATH_CALUDE_repeating_decimal_sum_l153_15382

theorem repeating_decimal_sum : 
  (4 : ℚ) / 33 + 34 / 999 + 567 / 99999 = 134255 / 32929667 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_sum_l153_15382


namespace NUMINAMATH_CALUDE_ratio_w_to_y_l153_15336

theorem ratio_w_to_y (w x y z : ℚ) 
  (hw : w / x = 4 / 3)
  (hy : y / z = 3 / 2)
  (hz : z / x = 1 / 6) :
  w / y = 16 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ratio_w_to_y_l153_15336


namespace NUMINAMATH_CALUDE_max_value_of_function_l153_15395

theorem max_value_of_function (x : ℝ) (h : 0 < x ∧ x < 1/2) :
  ∃ (y : ℝ), y = 1/27 ∧ ∀ (z : ℝ), 0 < z ∧ z < 1/2 → x^2 * (1 - 2*x) ≤ y := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_function_l153_15395


namespace NUMINAMATH_CALUDE_sum_of_2001_and_1015_l153_15350

theorem sum_of_2001_and_1015 : 2001 + 1015 = 3016 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_2001_and_1015_l153_15350


namespace NUMINAMATH_CALUDE_shopping_expense_l153_15357

theorem shopping_expense (total_spent shirt_cost : ℕ) (h1 : total_spent = 300) (h2 : shirt_cost = 97) :
  ∃ (shoe_cost : ℕ), 
    shoe_cost > 2 * shirt_cost ∧ 
    shirt_cost + shoe_cost = total_spent ∧ 
    shoe_cost - 2 * shirt_cost = 9 :=
by sorry

end NUMINAMATH_CALUDE_shopping_expense_l153_15357


namespace NUMINAMATH_CALUDE_hyperbola_equation_l153_15378

/-- A curve in the rectangular coordinate system (xOy) -/
structure Curve where
  -- The equation of the curve is implicitly defined by this function
  equation : ℝ → ℝ → Prop

/-- The eccentricity of a curve -/
def eccentricity (c : Curve) : ℝ := sorry

/-- Whether a point lies on a curve -/
def lies_on (p : ℝ × ℝ) (c : Curve) : Prop :=
  c.equation p.1 p.2

/-- The standard equation of a hyperbola -/
def is_standard_hyperbola_equation (c : Curve) : Prop :=
  ∃ a : ℝ, a ≠ 0 ∧ ∀ x y : ℝ, c.equation x y ↔ y^2 - x^2 = a^2

theorem hyperbola_equation (c : Curve) 
  (h_ecc : eccentricity c = Real.sqrt 2)
  (h_point : lies_on (1, Real.sqrt 2) c) :
  is_standard_hyperbola_equation c ∧ 
  ∃ x y : ℝ, c.equation x y ↔ y^2 - x^2 = 1 := by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l153_15378


namespace NUMINAMATH_CALUDE_square_condition_implies_value_l153_15318

theorem square_condition_implies_value (k : ℕ) :
  (∃ m n : ℕ, (4 * k + 5 = m^2) ∧ (9 * k + 4 = n^2)) →
  (7 * k + 4 = 39) := by
sorry

end NUMINAMATH_CALUDE_square_condition_implies_value_l153_15318


namespace NUMINAMATH_CALUDE_average_sq_feet_per_person_closest_to_500000_l153_15376

/-- The population of the United States in 1980 -/
def us_population : ℕ := 226504825

/-- The area of the United States in square miles -/
def us_area : ℕ := 3615122

/-- The number of square feet in one square mile -/
def sq_feet_per_sq_mile : ℕ := 5280 * 5280

/-- The options for the average number of square feet per person -/
def options : List ℕ := [5000, 10000, 50000, 100000, 500000]

/-- The theorem stating that the average number of square feet per person
    is closest to 500,000 among the given options -/
theorem average_sq_feet_per_person_closest_to_500000 :
  let total_sq_feet : ℕ := us_area * sq_feet_per_sq_mile
  let avg_sq_feet_per_person : ℚ := (total_sq_feet : ℚ) / us_population
  (500000 : ℚ) = options.argmin (fun x => |avg_sq_feet_per_person - x|) := by
  sorry

end NUMINAMATH_CALUDE_average_sq_feet_per_person_closest_to_500000_l153_15376


namespace NUMINAMATH_CALUDE_sqrt_4_minus_abs_sqrt_3_minus_2_plus_neg_1_pow_2023_l153_15331

theorem sqrt_4_minus_abs_sqrt_3_minus_2_plus_neg_1_pow_2023 :
  Real.sqrt 4 - |Real.sqrt 3 - 2| + (-1)^2023 = Real.sqrt 3 - 1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_4_minus_abs_sqrt_3_minus_2_plus_neg_1_pow_2023_l153_15331


namespace NUMINAMATH_CALUDE_exactlyOneOfThreeCount_l153_15344

/-- The number of math majors taking exactly one of Galois Theory, Hyperbolic Geometry, or Topology -/
def exactlyOneOfThree (total : ℕ) (noElective : ℕ) (ant_gt : ℕ) (gt_hg : ℕ) (hg_cry : ℕ) (cry_top : ℕ) (top_ant : ℕ) (ant_or_cry : ℕ) : ℕ :=
  total - noElective - ant_gt - gt_hg - hg_cry - cry_top - top_ant - ant_or_cry

theorem exactlyOneOfThreeCount :
  exactlyOneOfThree 100 22 7 12 3 15 8 16 = 17 :=
sorry

end NUMINAMATH_CALUDE_exactlyOneOfThreeCount_l153_15344


namespace NUMINAMATH_CALUDE_election_votes_calculation_l153_15315

theorem election_votes_calculation 
  (winning_percentage : Real) 
  (majority : Nat) 
  (total_votes : Nat) : 
  winning_percentage = 0.6 → 
  majority = 1504 → 
  (winning_percentage - (1 - winning_percentage)) * total_votes = majority → 
  total_votes = 7520 := by
sorry

end NUMINAMATH_CALUDE_election_votes_calculation_l153_15315


namespace NUMINAMATH_CALUDE_y_intercept_of_parallel_line_l153_15316

/-- A line in two-dimensional space. -/
structure Line where
  slope : ℝ
  yIntercept : ℝ

/-- Returns true if two lines are parallel. -/
def parallel (l1 l2 : Line) : Prop := l1.slope = l2.slope

/-- Returns true if a point lies on a line. -/
def pointOnLine (l : Line) (x y : ℝ) : Prop := y = l.slope * x + l.yIntercept

/-- The given line y = -3x + 6 -/
def givenLine : Line := { slope := -3, yIntercept := 6 }

theorem y_intercept_of_parallel_line :
  ∀ b : Line,
    parallel b givenLine →
    pointOnLine b 3 (-2) →
    b.yIntercept = 7 :=
by sorry

end NUMINAMATH_CALUDE_y_intercept_of_parallel_line_l153_15316


namespace NUMINAMATH_CALUDE_parallelogram_area_l153_15385

theorem parallelogram_area (side1 side2 angle : ℝ) (h_side1 : side1 = 20) (h_side2 : side2 = 30) (h_angle : angle = 40 * π / 180) :
  let height := side1 * Real.sin angle
  let area := side2 * height
  ∃ ε > 0, |area - 385.68| < ε :=
sorry

end NUMINAMATH_CALUDE_parallelogram_area_l153_15385


namespace NUMINAMATH_CALUDE_factor_theorem_quadratic_l153_15342

theorem factor_theorem_quadratic (t : ℚ) : 
  (∀ x, (x - t) ∣ (4*x^2 + 9*x + 2)) ↔ (t = -1/4 ∨ t = -2) :=
by sorry

end NUMINAMATH_CALUDE_factor_theorem_quadratic_l153_15342


namespace NUMINAMATH_CALUDE_operation_2011_result_l153_15375

def operation_result (n : ℕ) : ℕ :=
  match n % 3 with
  | 1 => 133
  | 2 => 55
  | 0 => 250
  | _ => 0  -- This case should never occur, but Lean requires it for completeness

theorem operation_2011_result :
  operation_result 2011 = 133 :=
sorry

end NUMINAMATH_CALUDE_operation_2011_result_l153_15375


namespace NUMINAMATH_CALUDE_ellipse_axis_endpoints_distance_l153_15345

/-- The distance between an endpoint of the major axis and an endpoint of the minor axis
    of the ellipse 4(x-2)^2 + 16(y-3)^2 = 64 is 2√5. -/
theorem ellipse_axis_endpoints_distance : 
  ∃ (C D : ℝ × ℝ),
    (∀ (x y : ℝ), 4 * (x - 2)^2 + 16 * (y - 3)^2 = 64 ↔ 
      (x - 2)^2 / 16 + (y - 3)^2 / 4 = 1) → 
    (C.1 - 2)^2 / 16 + (C.2 - 3)^2 / 4 = 1 →
    (D.1 - 2)^2 / 16 + (D.2 - 3)^2 / 4 = 1 →
    (C.1 - 2)^2 / 16 = 1 ∨ (C.2 - 3)^2 / 4 = 1 →
    (D.1 - 2)^2 / 16 = 1 ∨ (D.2 - 3)^2 / 4 = 1 →
    ((C.1 - 2)^2 / 16 = 1 ∧ (D.2 - 3)^2 / 4 = 1) ∨
    ((C.2 - 3)^2 / 4 = 1 ∧ (D.1 - 2)^2 / 16 = 1) →
    Real.sqrt ((C.1 - D.1)^2 + (C.2 - D.2)^2) = 2 * Real.sqrt 5 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_axis_endpoints_distance_l153_15345


namespace NUMINAMATH_CALUDE_fifteenth_term_of_geometric_sequence_l153_15320

/-- Given a geometric sequence with first term 32 and common ratio 1/4,
    prove that the 15th term is 1/8388608 -/
theorem fifteenth_term_of_geometric_sequence :
  let a₁ : ℚ := 32  -- First term
  let r : ℚ := 1/4  -- Common ratio
  let n : ℕ := 15   -- Term number we're looking for
  let aₙ : ℚ := a₁ * r^(n-1)  -- General term formula
  aₙ = 1/8388608 := by sorry

end NUMINAMATH_CALUDE_fifteenth_term_of_geometric_sequence_l153_15320


namespace NUMINAMATH_CALUDE_display_rows_count_l153_15366

/-- The number of cans in the nth row of the display -/
def cans_in_row (n : ℕ) : ℕ := 2 + 3 * (n - 1)

/-- The total number of cans in the first n rows of the display -/
def total_cans (n : ℕ) : ℕ := n * (cans_in_row 1 + cans_in_row n) / 2

theorem display_rows_count :
  ∃ n : ℕ, total_cans n = 169 ∧ n = 10 :=
sorry

end NUMINAMATH_CALUDE_display_rows_count_l153_15366


namespace NUMINAMATH_CALUDE_nathan_added_half_blankets_l153_15306

/-- The fraction of blankets Nathan added to his bed -/
def blanket_fraction (total_blankets : ℕ) (temp_per_blanket : ℕ) (total_temp_increase : ℕ) : ℚ :=
  (total_temp_increase / temp_per_blanket : ℚ) / total_blankets

/-- Theorem stating that Nathan added 1/2 of his blankets -/
theorem nathan_added_half_blankets :
  blanket_fraction 14 3 21 = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_nathan_added_half_blankets_l153_15306


namespace NUMINAMATH_CALUDE_coins_value_is_78_percent_of_dollar_l153_15330

-- Define the value of each coin in cents
def penny_value : ℕ := 1
def nickel_value : ℕ := 5
def dime_value : ℕ := 10
def quarter_value : ℕ := 25

-- Define the number of each coin
def num_pennies : ℕ := 3
def num_nickels : ℕ := 2
def num_dimes : ℕ := 4
def num_quarters : ℕ := 1

-- Define the total value in cents
def total_cents : ℕ := 
  num_pennies * penny_value + 
  num_nickels * nickel_value + 
  num_dimes * dime_value + 
  num_quarters * quarter_value

-- Define one dollar in cents
def dollar_in_cents : ℕ := 100

-- Theorem to prove
theorem coins_value_is_78_percent_of_dollar : 
  (total_cents : ℚ) / (dollar_in_cents : ℚ) = 78 / 100 := by
  sorry

end NUMINAMATH_CALUDE_coins_value_is_78_percent_of_dollar_l153_15330


namespace NUMINAMATH_CALUDE_basketball_sales_solution_l153_15393

/-- Represents the cost and sales information for basketballs --/
structure BasketballSales where
  cost_a : ℝ  -- Cost of A brand basketball
  cost_b : ℝ  -- Cost of B brand basketball
  price_a : ℝ  -- Original selling price of A brand basketball
  markup_b : ℝ  -- Markup percentage for B brand basketball
  discount_a : ℝ  -- Discount percentage for A brand basketball

/-- Theorem stating the solution to the basketball sales problem --/
theorem basketball_sales_solution (s : BasketballSales) : 
  (40 * s.cost_a + 40 * s.cost_b = 7200) →
  (50 * s.cost_a + 30 * s.cost_b = 7400) →
  (s.price_a = 140) →
  (s.markup_b = 0.3) →
  (40 * (s.price_a - s.cost_a) + 10 * (s.price_a * (1 - s.discount_a / 100) - s.cost_a) + 
   30 * s.cost_b * s.markup_b = 2440) →
  (s.cost_a = 100 ∧ s.cost_b = 80 ∧ s.discount_a = 8) := by
  sorry


end NUMINAMATH_CALUDE_basketball_sales_solution_l153_15393


namespace NUMINAMATH_CALUDE_map_distance_twenty_cm_distance_l153_15384

-- Define the scale of the map
def map_scale (cm : ℝ) (km : ℝ) : Prop := cm * (54 / 9) = km

-- Theorem statement
theorem map_distance (cm : ℝ) :
  map_scale 9 54 → map_scale cm (cm * 6) :=
by
  sorry

-- The specific case for 20 cm
theorem twenty_cm_distance :
  map_scale 9 54 → map_scale 20 120 :=
by
  sorry

end NUMINAMATH_CALUDE_map_distance_twenty_cm_distance_l153_15384


namespace NUMINAMATH_CALUDE_intersection_of_P_and_Q_l153_15394

-- Define the sets P and Q
def P : Set ℝ := {y | ∃ x, y = -x^2 + 2}
def Q : Set ℝ := {y | ∃ x, y = -x + 2}

-- State the theorem
theorem intersection_of_P_and_Q : P ∩ Q = {y | y ≤ 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_P_and_Q_l153_15394


namespace NUMINAMATH_CALUDE_book_pages_count_l153_15363

/-- The number of pages Liam read in a week -/
def total_pages : ℕ :=
  let first_three_days := 3 * 40
  let next_three_days := 3 * 50
  let seventh_day_first_session := 15
  let seventh_day_second_session := 2 * seventh_day_first_session
  first_three_days + next_three_days + seventh_day_first_session + seventh_day_second_session

/-- Theorem stating that the total number of pages in the book is 315 -/
theorem book_pages_count : total_pages = 315 := by
  sorry

end NUMINAMATH_CALUDE_book_pages_count_l153_15363


namespace NUMINAMATH_CALUDE_arithmetic_mean_after_removal_l153_15348

theorem arithmetic_mean_after_removal (S : Finset ℝ) (x y : ℝ) :
  S.card = 100 →
  x = 50 →
  y = 60 →
  x ∈ S →
  y ∈ S →
  (S.sum id) / S.card = 45 →
  ((S.sum id - (x + y)) / (S.card - 2)) = 44.8 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_after_removal_l153_15348


namespace NUMINAMATH_CALUDE_sum_of_decimals_l153_15322

theorem sum_of_decimals : 
  let addend1 : ℚ := 57/100
  let addend2 : ℚ := 23/100
  addend1 + addend2 = 4/5 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_decimals_l153_15322


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l153_15374

/-- An arithmetic sequence {a_n} with specified properties -/
def ArithmeticSequence (a : ℕ → ℚ) : Prop :=
  ∃ (a₁ d : ℚ), ∀ n, a n = a₁ + (n - 1) * d

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℚ)
  (h_arith : ArithmeticSequence a)
  (h_diff : a 7 - 2 * a 4 = -1)
  (h_third : a 3 = 0) :
  ∃ d : ℚ, (∀ n, a n = a 1 + (n - 1) * d) ∧ d = -1/2 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l153_15374


namespace NUMINAMATH_CALUDE_function_properties_l153_15313

noncomputable section

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x - a * Real.log x - a

-- State the theorem
theorem function_properties (a : ℝ) (h_a : a > 0) :
  -- Part 1
  (∃ x₀ : ℝ, x₀ > 0 ∧ ∀ x > 0, f e x ≥ f e x₀) ∧
  (∀ M : ℝ, ∃ x > 0, f e x > M) ∧
  -- Part 2
  (∀ x₁ x₂ : ℝ, 0 < x₁ ∧ x₁ < x₂ ∧ f a x₁ = 0 ∧ f a x₂ = 0 →
    1/a < x₁ ∧ x₁ < 1 ∧ 1 < x₂ ∧ x₂ < a) ∧
  -- Part 3
  (∀ x : ℝ, x > 0 → Real.exp (2*x - 2) - Real.exp (x - 1) * Real.log x - x ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l153_15313


namespace NUMINAMATH_CALUDE_max_pairs_sum_l153_15301

theorem max_pairs_sum (n : ℕ) (h : n = 3009) :
  let S := Finset.range n
  ∃ (k : ℕ) (pairs : Finset (ℕ × ℕ)),
    k = 1504 ∧
    pairs.card = k ∧
    (∀ (p : ℕ × ℕ), p ∈ pairs → p.1 ∈ S ∧ p.2 ∈ S ∧ p.1 < p.2) ∧
    (∀ (p q : ℕ × ℕ), p ∈ pairs → q ∈ pairs → p ≠ q → p.1 ≠ q.1 ∧ p.1 ≠ q.2 ∧ p.2 ≠ q.1 ∧ p.2 ≠ q.2) ∧
    (∀ (p : ℕ × ℕ), p ∈ pairs → p.1 + p.2 ≤ n) ∧
    (∀ (p q : ℕ × ℕ), p ∈ pairs → q ∈ pairs → p ≠ q → p.1 + p.2 ≠ q.1 + q.2) ∧
    (∀ (m : ℕ) (pairs' : Finset (ℕ × ℕ)),
      (∀ (p : ℕ × ℕ), p ∈ pairs' → p.1 ∈ S ∧ p.2 ∈ S ∧ p.1 < p.2) →
      (∀ (p q : ℕ × ℕ), p ∈ pairs' → q ∈ pairs' → p ≠ q → p.1 ≠ q.1 ∧ p.1 ≠ q.2 ∧ p.2 ≠ q.1 ∧ p.2 ≠ q.2) →
      (∀ (p : ℕ × ℕ), p ∈ pairs' → p.1 + p.2 ≤ n) →
      (∀ (p q : ℕ × ℕ), p ∈ pairs' → q ∈ pairs' → p ≠ q → p.1 + p.2 ≠ q.1 + q.2) →
      m = pairs'.card →
      m ≤ k) :=
by sorry

end NUMINAMATH_CALUDE_max_pairs_sum_l153_15301


namespace NUMINAMATH_CALUDE_cos_shift_l153_15338

theorem cos_shift (x : ℝ) : 
  2 * Real.cos (2 * (x - π / 8)) = 2 * Real.cos (2 * x - π / 4) := by
  sorry

#check cos_shift

end NUMINAMATH_CALUDE_cos_shift_l153_15338


namespace NUMINAMATH_CALUDE_both_hit_probability_l153_15372

def prob_both_hit (prob_A prob_B : ℝ) : ℝ := prob_A * prob_B

theorem both_hit_probability :
  let prob_A : ℝ := 0.8
  let prob_B : ℝ := 0.7
  prob_both_hit prob_A prob_B = 0.56 := by
  sorry

end NUMINAMATH_CALUDE_both_hit_probability_l153_15372


namespace NUMINAMATH_CALUDE_cyclist_climbing_speed_l153_15340

/-- Proves that the climbing speed is 20 m/min given the specified conditions -/
theorem cyclist_climbing_speed 
  (hill_length : ℝ) 
  (total_time : ℝ) 
  (climbing_speed : ℝ) :
  hill_length = 400 ∧ 
  total_time = 30 ∧ 
  (∃ t : ℝ, t > 0 ∧ t < 30 ∧ 
    hill_length = climbing_speed * t ∧ 
    hill_length = 2 * climbing_speed * (total_time - t)) →
  climbing_speed = 20 := by
  sorry

#check cyclist_climbing_speed

end NUMINAMATH_CALUDE_cyclist_climbing_speed_l153_15340


namespace NUMINAMATH_CALUDE_path_width_calculation_l153_15354

theorem path_width_calculation (field_length field_width path_area : ℝ) 
  (h1 : field_length = 20)
  (h2 : field_width = 15)
  (h3 : path_area = 246)
  (h4 : field_length > 0)
  (h5 : field_width > 0)
  (h6 : path_area > 0) :
  ∃ (path_width : ℝ),
    path_width > 0 ∧
    (field_length + 2 * path_width) * (field_width + 2 * path_width) - field_length * field_width = path_area ∧
    path_width = 3 := by
  sorry

end NUMINAMATH_CALUDE_path_width_calculation_l153_15354


namespace NUMINAMATH_CALUDE_unique_cube_fraction_l153_15312

theorem unique_cube_fraction :
  ∃! (n : ℤ), n ≠ 30 ∧ ∃ (k : ℤ), n / (30 - n) = k^3 := by
  sorry

end NUMINAMATH_CALUDE_unique_cube_fraction_l153_15312


namespace NUMINAMATH_CALUDE_triangle_perimeter_bound_l153_15369

theorem triangle_perimeter_bound (a b c : ℝ) : 
  a = 7 → b = 23 → (a + b > c ∧ a + c > b ∧ b + c > a) → 
  ∃ (n : ℕ), n = 60 ∧ ∀ (p : ℝ), p = a + b + c → ↑n > p ∧ ∀ (m : ℕ), ↑m > p → m ≥ n :=
sorry

end NUMINAMATH_CALUDE_triangle_perimeter_bound_l153_15369


namespace NUMINAMATH_CALUDE_solution_part1_solution_part2_l153_15396

/-- A system of linear equations in two variables x and y -/
structure LinearSystem where
  a : ℝ
  eq1 : ℝ → ℝ → ℝ := λ x y => x + 2 * y - a
  eq2 : ℝ → ℝ → ℝ := λ x y => 2 * x - y - 1

/-- The system has a solution when both equations equal zero -/
def HasSolution (s : LinearSystem) (x y : ℝ) : Prop :=
  s.eq1 x y = 0 ∧ s.eq2 x y = 0

theorem solution_part1 (s : LinearSystem) :
  HasSolution s 1 1 → s.a = 3 := by sorry

theorem solution_part2 (s : LinearSystem) :
  s.a = -2 → HasSolution s 0 (-1) ∧
  (∀ x y : ℝ, HasSolution s x y → x = 0 ∧ y = -1) := by sorry

end NUMINAMATH_CALUDE_solution_part1_solution_part2_l153_15396
