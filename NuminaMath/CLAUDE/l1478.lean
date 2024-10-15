import Mathlib

namespace NUMINAMATH_CALUDE_count_integers_satisfying_inequality_l1478_147852

theorem count_integers_satisfying_inequality :
  ∃ (S : Finset Int),
    (∀ n ∈ S, -15 ≤ n ∧ n ≤ 7 ∧ (n - 4) * (n + 2) * (n + 6) < -1) ∧
    (∀ n : Int, -15 ≤ n ∧ n ≤ 7 ∧ (n - 4) * (n + 2) * (n + 6) < -1 → n ∈ S) ∧
    Finset.card S = 12 :=
by sorry

end NUMINAMATH_CALUDE_count_integers_satisfying_inequality_l1478_147852


namespace NUMINAMATH_CALUDE_oxford_high_total_people_l1478_147818

-- Define the school structure
structure School where
  teachers : Nat
  principal : Nat
  vice_principals : Nat
  other_staff : Nat
  classes : Nat
  avg_students_per_class : Nat

-- Define Oxford High School
def oxford_high : School :=
  { teachers := 75,
    principal := 1,
    vice_principals := 3,
    other_staff := 20,
    classes := 35,
    avg_students_per_class := 23 }

-- Define the function to calculate total people
def total_people (s : School) : Nat :=
  s.teachers + s.principal + s.vice_principals + s.other_staff +
  (s.classes * s.avg_students_per_class)

-- Theorem statement
theorem oxford_high_total_people :
  total_people oxford_high = 904 := by
  sorry

end NUMINAMATH_CALUDE_oxford_high_total_people_l1478_147818


namespace NUMINAMATH_CALUDE_discount_profit_equivalence_l1478_147863

theorem discount_profit_equivalence (cost_price : ℝ) (discount_rate : ℝ) (profit_rate : ℝ)
  (h1 : discount_rate = 0.04)
  (h2 : profit_rate = 0.38) :
  let selling_price := cost_price * (1 + profit_rate)
  let discounted_price := selling_price * (1 - discount_rate)
  let profit_with_discount := discounted_price - cost_price
  let profit_without_discount := selling_price - cost_price
  profit_without_discount / cost_price = profit_rate :=
by
  sorry

end NUMINAMATH_CALUDE_discount_profit_equivalence_l1478_147863


namespace NUMINAMATH_CALUDE_area_segment_proportions_l1478_147898

/-- Given areas and segments, prove proportional relationships -/
theorem area_segment_proportions 
  (S S'' S' : ℝ) 
  (a a' : ℝ) 
  (h : S / S'' = a / a') 
  (h_pos : S > 0 ∧ S'' > 0 ∧ S' > 0 ∧ a > 0 ∧ a' > 0) :
  (S / a = S' / a') ∧ (S * a' = S' * a) := by
  sorry

end NUMINAMATH_CALUDE_area_segment_proportions_l1478_147898


namespace NUMINAMATH_CALUDE_isosceles_triangle_cosine_l1478_147877

def IsoscelesTriangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ a + c > b ∧ b + c > a ∧ b = c

def LargestAngleThreeTimesSmallest (a b c : ℝ) : Prop :=
  let cosSmallest := (b^2 + c^2 - a^2) / (2 * b * c)
  let cosLargest := (a^2 + b^2 - c^2) / (2 * a * b)
  cosLargest = 4 * cosSmallest^3 - 3 * cosSmallest

theorem isosceles_triangle_cosine (n : ℕ) :
  IsoscelesTriangle n (n + 1) (n + 1) →
  LargestAngleThreeTimesSmallest n (n + 1) (n + 1) →
  let cosSmallest := ((n + 1)^2 + (n + 1)^2 - n^2) / (2 * (n + 1) * (n + 1))
  cosSmallest = 7 / 9 :=
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_cosine_l1478_147877


namespace NUMINAMATH_CALUDE_tangent_sum_simplification_l1478_147827

theorem tangent_sum_simplification :
  (Real.tan (20 * π / 180) + Real.tan (30 * π / 180) + Real.tan (60 * π / 180) + Real.tan (70 * π / 180)) / Real.sin (10 * π / 180) =
  1 / (2 * Real.sin (10 * π / 180) ^ 2 * Real.cos (20 * π / 180)) + 4 / (Real.sqrt 3 * Real.sin (10 * π / 180)) := by
  sorry

end NUMINAMATH_CALUDE_tangent_sum_simplification_l1478_147827


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l1478_147874

theorem sufficient_not_necessary (x : ℝ) (h : x ≠ 0) :
  (∀ x > 1, x + 1/x > 2) ∧
  (∃ x, 0 < x ∧ x < 1 ∧ x + 1/x > 2) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l1478_147874


namespace NUMINAMATH_CALUDE_pies_sold_theorem_l1478_147880

/-- Represents the number of slices in an apple pie -/
def apple_slices : ℕ := 8

/-- Represents the number of slices in a peach pie -/
def peach_slices : ℕ := 6

/-- Represents the number of customers who ordered apple pie slices -/
def apple_customers : ℕ := 56

/-- Represents the number of customers who ordered peach pie slices -/
def peach_customers : ℕ := 48

/-- Calculates the total number of pies sold given the number of slices per pie and the number of customers -/
def total_pies (apple_slices peach_slices apple_customers peach_customers : ℕ) : ℕ :=
  (apple_customers / apple_slices) + (peach_customers / peach_slices)

/-- Theorem stating that the total number of pies sold is 15 -/
theorem pies_sold_theorem : total_pies apple_slices peach_slices apple_customers peach_customers = 15 := by
  sorry

end NUMINAMATH_CALUDE_pies_sold_theorem_l1478_147880


namespace NUMINAMATH_CALUDE_real_part_of_z_l1478_147836

theorem real_part_of_z (z : ℂ) (h : (1 + Complex.I) * z = 2 * Complex.I) : 
  Complex.re z = 1 := by sorry

end NUMINAMATH_CALUDE_real_part_of_z_l1478_147836


namespace NUMINAMATH_CALUDE_shoes_lost_l1478_147839

theorem shoes_lost (initial_pairs : ℕ) (remaining_pairs : ℕ) : 
  initial_pairs = 26 → remaining_pairs = 21 → initial_pairs * 2 - remaining_pairs * 2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_shoes_lost_l1478_147839


namespace NUMINAMATH_CALUDE_revenue_difference_l1478_147801

def viewers_game2 : ℕ := 80
def viewers_game1 : ℕ := viewers_game2 - 20
def viewers_game3 : ℕ := viewers_game2 + 15
def viewers_game4 : ℕ := viewers_game3 + (viewers_game3 / 10) + 1 -- Rounded up

def price_game1 : ℕ := 15
def price_game2 : ℕ := 20
def price_game3 : ℕ := 25
def price_game4 : ℕ := 30

def viewers_last_week : ℕ := 350
def price_last_week : ℕ := 18

def revenue_this_week : ℕ := 
  viewers_game1 * price_game1 + 
  viewers_game2 * price_game2 + 
  viewers_game3 * price_game3 + 
  viewers_game4 * price_game4

def revenue_last_week : ℕ := viewers_last_week * price_last_week

theorem revenue_difference : 
  revenue_this_week - revenue_last_week = 1725 := by
  sorry

end NUMINAMATH_CALUDE_revenue_difference_l1478_147801


namespace NUMINAMATH_CALUDE_sin_ratio_comparison_l1478_147846

theorem sin_ratio_comparison : 
  (Real.sin (2014 * π / 180)) / (Real.sin (2015 * π / 180)) < 
  (Real.sin (2016 * π / 180)) / (Real.sin (2017 * π / 180)) := by
  sorry

end NUMINAMATH_CALUDE_sin_ratio_comparison_l1478_147846


namespace NUMINAMATH_CALUDE_expected_subtree_size_ten_vertices_l1478_147802

-- Define a type for rooted trees
structure RootedTree where
  vertices : Nat
  root : Nat

-- Define a function to represent the expected subtree size
def expectedSubtreeSize (t : RootedTree) : ℚ :=
  sorry

-- Theorem statement
theorem expected_subtree_size_ten_vertices :
  ∀ t : RootedTree,
  t.vertices = 10 →
  expectedSubtreeSize t = 7381 / 2520 :=
by sorry

end NUMINAMATH_CALUDE_expected_subtree_size_ten_vertices_l1478_147802


namespace NUMINAMATH_CALUDE_average_age_proof_l1478_147815

theorem average_age_proof (a b c : ℕ) : 
  (a + b + c) / 3 = 26 → 
  b = 20 → 
  (a + c) / 2 = 29 :=
by sorry

end NUMINAMATH_CALUDE_average_age_proof_l1478_147815


namespace NUMINAMATH_CALUDE_range_of_m_for_single_valued_function_l1478_147812

/-- A function is single-valued on an interval if there exists a unique x in the interval
    that satisfies (b-a) * f'(x) = f(b) - f(a) --/
def SingleValued (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∃! x, a < x ∧ x < b ∧ (b - a) * (deriv f x) = f b - f a

/-- The function f(x) = x^3 - x^2 + m --/
def f (m : ℝ) : ℝ → ℝ := fun x ↦ x^3 - x^2 + m

theorem range_of_m_for_single_valued_function (a : ℝ) (h_a : a ≥ 1) :
  ∀ m : ℝ, SingleValued (f m) 0 a ∧ 
  (∃ x y, 0 ≤ x ∧ x < y ∧ y ≤ a ∧ f m x = 0 ∧ f m y = 0) ∧ 
  (∀ z, 0 ≤ z ∧ z ≤ a ∧ f m z = 0 → z = x ∨ z = y) →
  -1 ≤ m ∧ m < 4/27 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_for_single_valued_function_l1478_147812


namespace NUMINAMATH_CALUDE_probability_of_black_ball_l1478_147887

theorem probability_of_black_ball (p_red p_white p_black : ℝ) :
  p_red = 0.41 →
  p_white = 0.27 →
  p_red + p_white + p_black = 1 →
  p_black = 0.32 := by
sorry

end NUMINAMATH_CALUDE_probability_of_black_ball_l1478_147887


namespace NUMINAMATH_CALUDE_quadratic_function_theorem_l1478_147816

/-- A quadratic function with real coefficients -/
def f (a b : ℝ) (x : ℝ) : ℝ := x^2 + a*x + b

/-- The theorem statement -/
theorem quadratic_function_theorem (a b : ℝ) :
  (∀ x, f a b x ≥ 0) →
  (∀ x, x ∈ Set.Ioo 1 7 ↔ f a b x < 9) →
  (∃ c, ∀ x, x ∈ Set.Ioo 1 7 ↔ f a b x < c) →
  (∃! c, ∀ x, x ∈ Set.Ioo 1 7 ↔ f a b x < c) ∧
  (∀ x, x ∈ Set.Ioo 1 7 ↔ f a b x < 9) := by
sorry

end NUMINAMATH_CALUDE_quadratic_function_theorem_l1478_147816


namespace NUMINAMATH_CALUDE_student_weighted_average_l1478_147850

def weighted_average (courses1 courses2 courses3 : ℕ) (grade1 grade2 grade3 : ℚ) : ℚ :=
  (courses1 * grade1 + courses2 * grade2 + courses3 * grade3) / (courses1 + courses2 + courses3)

theorem student_weighted_average :
  let courses1 := 8
  let courses2 := 6
  let courses3 := 10
  let grade1 := 92
  let grade2 := 88
  let grade3 := 76
  abs (weighted_average courses1 courses2 courses3 grade1 grade2 grade3 - 84.3) < 0.05 := by
  sorry

end NUMINAMATH_CALUDE_student_weighted_average_l1478_147850


namespace NUMINAMATH_CALUDE_parabola_properties_l1478_147847

-- Define the parabola and its properties
def parabola (a b c m : ℝ) : Prop :=
  a ≠ 0 ∧ a < 0 ∧ -2 < m ∧ m < -1 ∧
  a * 1^2 + b * 1 + c = 0 ∧
  a * m^2 + b * m + c = 0

-- State the theorem
theorem parabola_properties (a b c m : ℝ) (h : parabola a b c m) :
  a * b * c > 0 ∧ a - b + c > 0 ∧ a * (m + 1) - b + c > 0 := by
  sorry

end NUMINAMATH_CALUDE_parabola_properties_l1478_147847


namespace NUMINAMATH_CALUDE_banana_muffins_count_l1478_147829

/-- Represents the types of pastries in the shop -/
inductive Pastry
  | PlainDoughnut
  | GlazedDoughnut
  | ChocolateChipCookie
  | OatmealCookie
  | BlueberryMuffin
  | BananaMuffin

/-- The ratio of pastries in the shop -/
def pastryRatio : Pastry → ℕ
  | Pastry.PlainDoughnut => 5
  | Pastry.GlazedDoughnut => 4
  | Pastry.ChocolateChipCookie => 3
  | Pastry.OatmealCookie => 2
  | Pastry.BlueberryMuffin => 1
  | Pastry.BananaMuffin => 2

/-- The number of plain doughnuts in the shop -/
def numPlainDoughnuts : ℕ := 50

/-- Theorem stating that the number of banana muffins is 20 -/
theorem banana_muffins_count :
  (numPlainDoughnuts / pastryRatio Pastry.PlainDoughnut) * pastryRatio Pastry.BananaMuffin = 20 := by
  sorry

end NUMINAMATH_CALUDE_banana_muffins_count_l1478_147829


namespace NUMINAMATH_CALUDE_marble_jar_ratio_l1478_147834

/-- 
Given a collection of marbles distributed in three jars, where:
- The first jar contains 80 marbles
- The second jar contains twice the amount of the first jar
- The total number of marbles is 260

This theorem proves that the ratio of marbles in the third jar to the first jar is 1/4.
-/
theorem marble_jar_ratio : 
  ∀ (jar1 jar2 jar3 : ℕ),
  jar1 = 80 →
  jar2 = 2 * jar1 →
  jar1 + jar2 + jar3 = 260 →
  (jar3 : ℚ) / jar1 = 1 / 4 := by
sorry

end NUMINAMATH_CALUDE_marble_jar_ratio_l1478_147834


namespace NUMINAMATH_CALUDE_soda_price_ratio_l1478_147869

theorem soda_price_ratio (v : ℝ) (p : ℝ) (hv : v > 0) (hp : p > 0) :
  let x_volume := 1.3 * v
  let x_price := 0.85 * p
  let x_unit_price := x_price / x_volume
  let y_unit_price := p / v
  x_unit_price / y_unit_price = 17 / 26 := by
sorry

end NUMINAMATH_CALUDE_soda_price_ratio_l1478_147869


namespace NUMINAMATH_CALUDE_donation_problem_l1478_147883

theorem donation_problem (day1_amount day2_amount : ℕ) 
  (day2_extra_donors : ℕ) (h1 : day1_amount = 4800) 
  (h2 : day2_amount = 6000) (h3 : day2_extra_donors = 50) : 
  ∃ (day1_donors : ℕ), 
    (day1_donors > 0 ∧ day1_donors + day2_extra_donors > 0) ∧
    (day1_amount : ℚ) / day1_donors = (day2_amount : ℚ) / (day1_donors + day2_extra_donors) ∧
    day1_donors + (day1_donors + day2_extra_donors) = 450 ∧
    (day1_amount : ℚ) / day1_donors = 24 :=
by
  sorry

#check donation_problem

end NUMINAMATH_CALUDE_donation_problem_l1478_147883


namespace NUMINAMATH_CALUDE_museum_ticket_fraction_l1478_147810

theorem museum_ticket_fraction (total : ℚ) (sandwich_fraction : ℚ) (book_fraction : ℚ) (leftover : ℚ) :
  total = 150 →
  sandwich_fraction = 1/5 →
  book_fraction = 1/2 →
  leftover = 20 →
  (total - (sandwich_fraction * total + book_fraction * total + leftover)) / total = 1/6 := by
  sorry

end NUMINAMATH_CALUDE_museum_ticket_fraction_l1478_147810


namespace NUMINAMATH_CALUDE_medal_distribution_proof_l1478_147897

def distribute_medals (n : ℕ) : ℕ :=
  Nat.choose (n + 2) 2

theorem medal_distribution_proof (n : ℕ) (h : n = 12) : 
  distribute_medals n = 55 := by
  sorry

end NUMINAMATH_CALUDE_medal_distribution_proof_l1478_147897


namespace NUMINAMATH_CALUDE_four_number_sequence_l1478_147817

theorem four_number_sequence (a b c d : ℝ) : 
  (b - a = c - b) →  -- arithmetic sequence condition
  (c * c = b * d) →  -- geometric sequence condition
  (a + d = 16) → 
  (b + c = 12) → 
  ((a = 15 ∧ b = 9 ∧ c = 3 ∧ d = 1) ∨ (a = 0 ∧ b = 4 ∧ c = 8 ∧ d = 16)) :=
by sorry

end NUMINAMATH_CALUDE_four_number_sequence_l1478_147817


namespace NUMINAMATH_CALUDE_largest_a_when_b_equals_c_l1478_147807

theorem largest_a_when_b_equals_c (A B C : ℕ) 
  (h1 : A = 5 * B + C) 
  (h2 : B = C) : 
  A ≤ 24 ∧ ∃ (A₀ : ℕ), A₀ = 24 ∧ ∃ (B₀ C₀ : ℕ), A₀ = 5 * B₀ + C₀ ∧ B₀ = C₀ :=
by sorry

end NUMINAMATH_CALUDE_largest_a_when_b_equals_c_l1478_147807


namespace NUMINAMATH_CALUDE_certain_number_proof_l1478_147894

theorem certain_number_proof (x q : ℝ) 
  (h1 : 3 / x = 8)
  (h2 : 3 / q = 18)
  (h3 : x - q = 0.20833333333333334) :
  x = 0.375 := by
sorry

end NUMINAMATH_CALUDE_certain_number_proof_l1478_147894


namespace NUMINAMATH_CALUDE_cube_dot_path_length_l1478_147824

theorem cube_dot_path_length (cube_edge : ℝ) (h_edge : cube_edge = 2) :
  let face_diagonal := cube_edge * Real.sqrt 2
  let dot_path_radius := face_diagonal / 2
  let dot_path_length := 2 * Real.pi * dot_path_radius
  dot_path_length = 2 * Real.sqrt 2 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_cube_dot_path_length_l1478_147824


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l1478_147845

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- The statement to be proved -/
theorem geometric_sequence_problem (a : ℕ → ℝ) :
  geometric_sequence a →
  (a 3)^2 + 7*(a 3) + 9 = 0 →
  (a 7)^2 + 7*(a 7) + 9 = 0 →
  a 5 = -3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l1478_147845


namespace NUMINAMATH_CALUDE_shirt_price_l1478_147899

theorem shirt_price (total_cost : ℝ) (price_difference : ℝ) (shirt_price : ℝ) :
  total_cost = 80.34 →
  shirt_price = (total_cost + price_difference) / 2 - price_difference →
  price_difference = 7.43 →
  shirt_price = 36.455 := by
  sorry

end NUMINAMATH_CALUDE_shirt_price_l1478_147899


namespace NUMINAMATH_CALUDE_converse_xy_zero_x_zero_is_true_l1478_147841

theorem converse_xy_zero_x_zero_is_true :
  ∀ (x y : ℝ), x = 0 → x * y = 0 :=
by sorry

end NUMINAMATH_CALUDE_converse_xy_zero_x_zero_is_true_l1478_147841


namespace NUMINAMATH_CALUDE_quadratic_discriminant_nonnegative_l1478_147888

/-- 
Given a quadratic equation ax^2 + 4bx + c = 0 where a, b, and c form an arithmetic progression,
prove that the discriminant Δ is always non-negative.
-/
theorem quadratic_discriminant_nonnegative 
  (a b c : ℝ) 
  (h_progression : ∃ d : ℝ, b = a + d ∧ c = a + 2*d) : 
  (4*b)^2 - 4*a*c ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_discriminant_nonnegative_l1478_147888


namespace NUMINAMATH_CALUDE_lesser_fraction_l1478_147868

theorem lesser_fraction (x y : ℚ) (sum_eq : x + y = 5/6) (prod_eq : x * y = 1/8) :
  min x y = 1/6 := by
  sorry

end NUMINAMATH_CALUDE_lesser_fraction_l1478_147868


namespace NUMINAMATH_CALUDE_candy_count_l1478_147860

theorem candy_count : ∃ n : ℕ, n % 3 = 2 ∧ n % 4 = 3 ∧ 32 ≤ n ∧ n ≤ 35 ∧ n = 35 := by
  sorry

end NUMINAMATH_CALUDE_candy_count_l1478_147860


namespace NUMINAMATH_CALUDE_tina_win_probability_l1478_147808

theorem tina_win_probability (p_lose : ℚ) (h_lose : p_lose = 3/7) (h_no_tie : True) :
  1 - p_lose = 4/7 := by
  sorry

end NUMINAMATH_CALUDE_tina_win_probability_l1478_147808


namespace NUMINAMATH_CALUDE_simplify_expression_l1478_147825

theorem simplify_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a^3 + b^3 = 3*(a + b)) :
  a/b + b/a - 3/(a*b) = 1 := by
sorry

end NUMINAMATH_CALUDE_simplify_expression_l1478_147825


namespace NUMINAMATH_CALUDE_trig_expression_value_l1478_147870

theorem trig_expression_value (α : Real) (h : Real.tan α = 3) :
  (6 * Real.sin α - 2 * Real.cos α) / (5 * Real.cos α + 3 * Real.sin α) = 8 / 7 := by
  sorry

end NUMINAMATH_CALUDE_trig_expression_value_l1478_147870


namespace NUMINAMATH_CALUDE_expression_decrease_decrease_percentage_l1478_147828

theorem expression_decrease (x y : ℝ) (h : x ≠ 0 ∧ y ≠ 0) : 
  (2 * ((1/2 * x)^2) * (1/2 * y)) / (2 * x^2 * y) = 1/4 :=
sorry

theorem decrease_percentage : (1 - 1/4) * 100 = 87.5 :=
sorry

end NUMINAMATH_CALUDE_expression_decrease_decrease_percentage_l1478_147828


namespace NUMINAMATH_CALUDE_speech_contest_allocation_l1478_147830

/-- The number of ways to distribute n indistinguishable objects into k distinct boxes,
    with each box containing at least one object. -/
def distribute (n k : ℕ) : ℕ := sorry

/-- The number of ways to choose 2 items from a set of 6 items. -/
def choose_two_from_six : ℕ := sorry

theorem speech_contest_allocation :
  distribute 8 6 = choose_two_from_six + 6 := by sorry

end NUMINAMATH_CALUDE_speech_contest_allocation_l1478_147830


namespace NUMINAMATH_CALUDE_constant_b_value_l1478_147890

theorem constant_b_value (x y b : ℝ) 
  (h1 : (7 * x + b * y) / (x - 2 * y) = 25)
  (h2 : x / (2 * y) = 3 / 2) : 
  b = 4 := by sorry

end NUMINAMATH_CALUDE_constant_b_value_l1478_147890


namespace NUMINAMATH_CALUDE_correct_seating_arrangements_l1478_147843

/-- The number of ways to arrange n distinct objects. -/
def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·+1) 1

/-- The number of people to be seated. -/
def totalPeople : ℕ := 8

/-- The number of ways to seat the people under the given conditions. -/
def seatingArrangements : ℕ :=
  factorial totalPeople - 2 * (factorial (totalPeople - 1) * factorial 2)

theorem correct_seating_arrangements :
  seatingArrangements = 20160 := by sorry

end NUMINAMATH_CALUDE_correct_seating_arrangements_l1478_147843


namespace NUMINAMATH_CALUDE_no_solution_x5_y2_plus4_l1478_147821

theorem no_solution_x5_y2_plus4 : ¬ ∃ (x y : ℕ), x ≥ 1 ∧ y ≥ 1 ∧ x^5 = y^2 + 4 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_x5_y2_plus4_l1478_147821


namespace NUMINAMATH_CALUDE_f_negative_two_equals_negative_eight_l1478_147822

noncomputable def f (x : ℝ) : ℝ := 
  if x ≥ 0 then 3^x - 1 else -3^(-x) + 1

theorem f_negative_two_equals_negative_eight :
  f (-2) = -8 :=
by sorry

end NUMINAMATH_CALUDE_f_negative_two_equals_negative_eight_l1478_147822


namespace NUMINAMATH_CALUDE_temperature_calculation_l1478_147862

/-- Given the average temperatures for two sets of four consecutive days and the temperature of the first day, calculate the temperature of the last day. -/
theorem temperature_calculation (temp_mon tues wed thurs fri : ℝ) :
  (temp_mon + tues + wed + thurs) / 4 = 48 →
  (tues + wed + thurs + fri) / 4 = 46 →
  temp_mon = 39 →
  fri = 31 := by
  sorry

end NUMINAMATH_CALUDE_temperature_calculation_l1478_147862


namespace NUMINAMATH_CALUDE_second_car_departure_time_l1478_147885

/-- Proves that the second car left 45 minutes after the first car --/
theorem second_car_departure_time (first_car_speed : ℝ) (trip_distance : ℝ) 
  (second_car_speed : ℝ) (time_difference : ℝ) : 
  first_car_speed = 30 →
  trip_distance = 80 →
  second_car_speed = 60 →
  time_difference = 1.5 →
  (time_difference - (first_car_speed * time_difference / second_car_speed)) * 60 = 45 := by
  sorry

#check second_car_departure_time

end NUMINAMATH_CALUDE_second_car_departure_time_l1478_147885


namespace NUMINAMATH_CALUDE_rectangle_area_with_inscribed_circle_l1478_147865

/-- Given a right triangle XYZ with coordinates X(0,0), Y(a,0), Z(a,a),
    where 'a' is a positive real number, and a circle with radius 'a'
    inscribed in the rectangle formed by extending sides XY and YZ,
    prove that the area of the rectangle is 4a² when the hypotenuse XZ = 2a. -/
theorem rectangle_area_with_inscribed_circle (a : ℝ) (ha : a > 0) :
  let X : ℝ × ℝ := (0, 0)
  let Y : ℝ × ℝ := (a, 0)
  let Z : ℝ × ℝ := (a, a)
  let hypotenuse := Real.sqrt ((Z.1 - X.1)^2 + (Z.2 - X.2)^2)
  hypotenuse = 2 * a →
  (2 * a) * (2 * a) = 4 * a^2 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_with_inscribed_circle_l1478_147865


namespace NUMINAMATH_CALUDE_oliver_money_problem_l1478_147873

theorem oliver_money_problem (X : ℤ) :
  X + 5 - 4 - 3 + 8 = 15 → X = 13 := by
  sorry

end NUMINAMATH_CALUDE_oliver_money_problem_l1478_147873


namespace NUMINAMATH_CALUDE_point_position_l1478_147811

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space with equation Ax + By + C = 0 -/
structure Line where
  A : ℝ
  B : ℝ
  C : ℝ

/-- Determines if a point is on the upper right side of a line -/
def isUpperRight (l : Line) (p : Point) : Prop :=
  l.A * p.x + l.B * p.y + l.C < 0 ∧ l.A > 0 ∧ l.B < 0

theorem point_position (l : Line) (p : Point) :
  isUpperRight l p → p.y > (-l.A * p.x - l.C) / l.B :=
by sorry

end NUMINAMATH_CALUDE_point_position_l1478_147811


namespace NUMINAMATH_CALUDE_jakes_third_test_score_l1478_147803

/-- Given Jake's test scores, prove he scored 65 in the third test -/
theorem jakes_third_test_score :
  -- Define the number of tests
  let num_tests : ℕ := 4
  -- Define the average score
  let average_score : ℚ := 75
  -- Define the score of the first test
  let first_test_score : ℕ := 80
  -- Define the score difference between second and first tests
  let second_test_difference : ℕ := 10
  -- Define the condition that third and fourth test scores are equal
  ∀ (third_test_score fourth_test_score : ℕ),
    -- Total score equals average multiplied by number of tests
    (first_test_score + (first_test_score + second_test_difference) + third_test_score + fourth_test_score : ℚ) = num_tests * average_score →
    -- Third and fourth test scores are equal
    third_test_score = fourth_test_score →
    -- Prove that the third test score is 65
    third_test_score = 65 := by
  sorry

end NUMINAMATH_CALUDE_jakes_third_test_score_l1478_147803


namespace NUMINAMATH_CALUDE_diagonal_length_in_specific_kite_l1478_147820

/-- A kite is a quadrilateral with two pairs of adjacent sides of equal length -/
structure Kite :=
  (A B C D : ℝ × ℝ)
  (is_kite : (A.1 - B.1)^2 + (A.2 - B.2)^2 = (A.1 - D.1)^2 + (A.2 - D.2)^2 ∧
             (B.1 - C.1)^2 + (B.2 - C.2)^2 = (C.1 - D.1)^2 + (C.2 - D.2)^2)

/-- The theorem about the diagonal length in a specific kite -/
theorem diagonal_length_in_specific_kite (k : Kite) 
  (ab_length : (k.A.1 - k.B.1)^2 + (k.A.2 - k.B.2)^2 = 100)
  (bc_length : (k.B.1 - k.C.1)^2 + (k.B.2 - k.C.2)^2 = 225)
  (sin_B : Real.sin (Real.arcsin ((k.A.2 - k.B.2) / Real.sqrt ((k.A.1 - k.B.1)^2 + (k.A.2 - k.B.2)^2))) = 4/5)
  (angle_ADB : Real.cos (Real.arccos ((k.A.1 - k.D.1) * (k.B.1 - k.D.1) + 
                                      (k.A.2 - k.D.2) * (k.B.2 - k.D.2)) / 
                        (Real.sqrt ((k.A.1 - k.D.1)^2 + (k.A.2 - k.D.2)^2) * 
                         Real.sqrt ((k.B.1 - k.D.1)^2 + (k.B.2 - k.D.2)^2))) = -1/2) :
  (k.A.1 - k.C.1)^2 + (k.A.2 - k.C.2)^2 = 150 := by sorry

end NUMINAMATH_CALUDE_diagonal_length_in_specific_kite_l1478_147820


namespace NUMINAMATH_CALUDE_janes_babysitting_ratio_l1478_147853

/-- Represents the age ratio between a babysitter and a child -/
structure AgeRatio where
  babysitter : ℕ
  child : ℕ

/-- The problem setup for Jane's babysitting scenario -/
structure BabysittingScenario where
  jane_current_age : ℕ
  years_since_stopped : ℕ
  oldest_child_current_age : ℕ

/-- Calculates the age ratio between Jane and the oldest child she could have babysat -/
def calculate_age_ratio (scenario : BabysittingScenario) : AgeRatio :=
  { babysitter := scenario.jane_current_age - scenario.years_since_stopped,
    child := scenario.oldest_child_current_age - scenario.years_since_stopped }

/-- The main theorem to prove -/
theorem janes_babysitting_ratio :
  let scenario : BabysittingScenario := {
    jane_current_age := 34,
    years_since_stopped := 12,
    oldest_child_current_age := 25
  }
  let ratio := calculate_age_ratio scenario
  ratio.babysitter = 22 ∧ ratio.child = 13 := by sorry

end NUMINAMATH_CALUDE_janes_babysitting_ratio_l1478_147853


namespace NUMINAMATH_CALUDE_definite_integral_tan_cos_sin_l1478_147814

theorem definite_integral_tan_cos_sin : 
  ∫ x in (π / 4)..(Real.arcsin (2 / Real.sqrt 5)), (4 * Real.tan x - 5) / (4 * Real.cos x ^ 2 - Real.sin (2 * x) + 1) = 2 * Real.log (5 / 4) - (1 / 2) * Real.arctan (1 / 2) := by
  sorry

end NUMINAMATH_CALUDE_definite_integral_tan_cos_sin_l1478_147814


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1478_147872

theorem inequality_solution_set (x : ℝ) (hx1 : x ≠ 0) (hx2 : x ≠ 2) :
  (2 * x) / (x - 2) + (x + 3) / (3 * x) ≥ 4 ↔ x ∈ Set.Ioc 0 (1/5) ∪ Set.Ioc 2 6 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1478_147872


namespace NUMINAMATH_CALUDE_age_difference_l1478_147809

theorem age_difference (man_age son_age : ℕ) : 
  man_age > son_age →
  son_age = 27 →
  man_age + 2 = 2 * (son_age + 2) →
  man_age - son_age = 29 := by
sorry

end NUMINAMATH_CALUDE_age_difference_l1478_147809


namespace NUMINAMATH_CALUDE_number_problem_l1478_147844

theorem number_problem (x : ℝ) : (1/4 : ℝ) * x = (1/5 : ℝ) * (x + 1) + 1 → x = 24 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l1478_147844


namespace NUMINAMATH_CALUDE_sum_inequality_l1478_147851

theorem sum_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  c / a + a / (b + c) + b / c ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_inequality_l1478_147851


namespace NUMINAMATH_CALUDE_symmetric_line_passes_through_point_l1478_147889

-- Define a line type
structure Line where
  slope : ℝ
  intercept : ℝ

-- Define a point type
structure Point where
  x : ℝ
  y : ℝ

-- Function to check if a point is on a line
def point_on_line (p : Point) (l : Line) : Prop :=
  p.y = l.slope * p.x + l.intercept

-- Function to check if two lines are symmetric about a point
def symmetric_lines (l₁ l₂ : Line) (p : Point) : Prop :=
  ∀ (x y : ℝ), point_on_line ⟨x, y⟩ l₁ ↔ 
    point_on_line ⟨2*p.x - x, 2*p.y - y⟩ l₂

-- Theorem statement
theorem symmetric_line_passes_through_point :
  ∀ (k : ℝ),
  let l₁ : Line := ⟨k, -4*k⟩
  let l₂ : Line := sorry
  let p : Point := ⟨2, 1⟩
  symmetric_lines l₁ l₂ p →
  point_on_line ⟨0, 2⟩ l₂ := by
  sorry

end NUMINAMATH_CALUDE_symmetric_line_passes_through_point_l1478_147889


namespace NUMINAMATH_CALUDE_xiaoming_mother_money_l1478_147892

/-- The amount of money Xiaoming's mother brought to buy soap. -/
def money : ℕ := 36

/-- The price of one unit of brand A soap in yuan. -/
def price_A : ℕ := 6

/-- The price of one unit of brand B soap in yuan. -/
def price_B : ℕ := 9

/-- The number of units of brand A soap that can be bought with the money. -/
def units_A : ℕ := money / price_A

/-- The number of units of brand B soap that can be bought with the money. -/
def units_B : ℕ := money / price_B

theorem xiaoming_mother_money :
  (units_A = units_B + 2) ∧
  (money = units_A * price_A) ∧
  (money = units_B * price_B) := by
  sorry

end NUMINAMATH_CALUDE_xiaoming_mother_money_l1478_147892


namespace NUMINAMATH_CALUDE_final_bacteria_count_l1478_147884

-- Define the initial number of bacteria
def initial_bacteria : ℕ := 50

-- Define the doubling interval in minutes
def doubling_interval : ℕ := 4

-- Define the total time elapsed in minutes
def total_time : ℕ := 15

-- Define the number of complete doubling intervals
def complete_intervals : ℕ := total_time / doubling_interval

-- Function to calculate the bacteria population after a given number of intervals
def bacteria_population (intervals : ℕ) : ℕ :=
  initial_bacteria * (2 ^ intervals)

-- Theorem stating the final bacteria count
theorem final_bacteria_count :
  bacteria_population complete_intervals = 400 := by
  sorry

end NUMINAMATH_CALUDE_final_bacteria_count_l1478_147884


namespace NUMINAMATH_CALUDE_irrational_density_l1478_147895

theorem irrational_density (α : ℝ) (h_irrational : Irrational α) (a b : ℝ) (h_lt : a < b) :
  ∃ (m n : ℕ), a < m * α - n ∧ m * α - n < b :=
sorry

end NUMINAMATH_CALUDE_irrational_density_l1478_147895


namespace NUMINAMATH_CALUDE_five_solutions_l1478_147893

/-- The system of equations has exactly 5 distinct real solutions -/
theorem five_solutions (x y z w : ℝ) : 
  (x = w + z + z*w*x) ∧
  (y = z + x + z*x*y) ∧
  (z = x + y + x*y*z) ∧
  (w = y + z + y*z*w) →
  ∃! (sol : Finset (ℝ × ℝ × ℝ × ℝ)), sol.card = 5 ∧ ∀ (a b c d : ℝ), (a, b, c, d) ∈ sol ↔
    (a = d + c + c*d*a) ∧
    (b = c + a + c*a*b) ∧
    (c = a + b + a*b*c) ∧
    (d = b + c + b*c*d) := by
  sorry

end NUMINAMATH_CALUDE_five_solutions_l1478_147893


namespace NUMINAMATH_CALUDE_PQ_length_range_l1478_147838

/-- The circle C in the Cartesian coordinate system -/
def C : Set (ℝ × ℝ) := {p | p.1^2 + (p.2 - 3)^2 = 2}

/-- A point on the x-axis -/
def A : ℝ → ℝ × ℝ := λ x => (x, 0)

/-- The tangent points P and Q on the circle C -/
noncomputable def P (x : ℝ) : ℝ × ℝ := sorry
noncomputable def Q (x : ℝ) : ℝ × ℝ := sorry

/-- The length of segment PQ -/
noncomputable def PQ_length (x : ℝ) : ℝ :=
  Real.sqrt ((P x).1 - (Q x).1)^2 + ((P x).2 - (Q x).2)^2

/-- The theorem stating the range of PQ length -/
theorem PQ_length_range :
  ∀ x : ℝ, 2 * Real.sqrt 14 / 3 < PQ_length x ∧ PQ_length x < 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_PQ_length_range_l1478_147838


namespace NUMINAMATH_CALUDE_fraction_equality_l1478_147835

theorem fraction_equality (q r s u : ℚ) 
  (h1 : q / r = 8)
  (h2 : s / r = 4)
  (h3 : s / u = 1 / 3) :
  u / q = 3 / 2 := by
sorry

end NUMINAMATH_CALUDE_fraction_equality_l1478_147835


namespace NUMINAMATH_CALUDE_circle_radius_l1478_147857

theorem circle_radius (x y : ℝ) : 
  (16 * x^2 - 32 * x + 16 * y^2 - 48 * y + 68 = 0) → 
  (∃ h k : ℝ, (x - h)^2 + (y - k)^2 = 1) :=
by sorry

end NUMINAMATH_CALUDE_circle_radius_l1478_147857


namespace NUMINAMATH_CALUDE_girls_in_class_l1478_147886

theorem girls_in_class (total : ℕ) (ratio_girls : ℕ) (ratio_boys : ℕ) (ratio_nonbinary : ℕ) 
  (h1 : ratio_girls = 3)
  (h2 : ratio_boys = 2)
  (h3 : ratio_nonbinary = 1)
  (h4 : total = 72) :
  (total * ratio_girls) / (ratio_girls + ratio_boys + ratio_nonbinary) = 36 := by
sorry

end NUMINAMATH_CALUDE_girls_in_class_l1478_147886


namespace NUMINAMATH_CALUDE_equation_solution_l1478_147806

theorem equation_solution : ∃! x : ℝ, (1 / (x - 1) = 3 / (x - 3)) ∧ x = 0 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1478_147806


namespace NUMINAMATH_CALUDE_polynomial_identity_l1478_147878

theorem polynomial_identity : ∀ x : ℝ, 
  (x^2 + 3*x + 2) * (x + 3) = (x + 1) * (x^2 + 5*x + 6) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_identity_l1478_147878


namespace NUMINAMATH_CALUDE_sums_are_equal_l1478_147879

def sum1 : ℕ :=
  1 + 22 + 333 + 4444 + 55555 + 666666 + 7777777 + 88888888 + 999999999

def sum2 : ℕ :=
  9 + 98 + 987 + 9876 + 98765 + 987654 + 9876543 + 98765432 + 987654321

theorem sums_are_equal : sum1 = sum2 := by
  sorry

end NUMINAMATH_CALUDE_sums_are_equal_l1478_147879


namespace NUMINAMATH_CALUDE_no_integer_solution_l1478_147805

theorem no_integer_solution : ∀ x y : ℤ, x^2 + 5 ≠ y^3 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solution_l1478_147805


namespace NUMINAMATH_CALUDE_elevator_max_weight_elevator_problem_l1478_147866

/-- Calculates the maximum weight of the next person to enter an elevator without overloading it. -/
theorem elevator_max_weight (num_adults : ℕ) (num_children : ℕ) (avg_adult_weight : ℝ) 
  (avg_child_weight : ℝ) (original_capacity : ℝ) (capacity_increase : ℝ) : ℝ :=
  let total_adult_weight := num_adults * avg_adult_weight
  let total_child_weight := num_children * avg_child_weight
  let current_weight := total_adult_weight + total_child_weight
  let new_capacity := original_capacity * (1 + capacity_increase)
  new_capacity - current_weight

/-- Proves that the maximum weight of the next person to enter the elevator is 250 pounds. -/
theorem elevator_problem : 
  elevator_max_weight 7 5 150 70 1500 0.1 = 250 := by
  sorry

end NUMINAMATH_CALUDE_elevator_max_weight_elevator_problem_l1478_147866


namespace NUMINAMATH_CALUDE_arc_measures_l1478_147854

-- Define the circle and angles
def Circle : Type := ℝ × ℝ
def CentralAngle (c : Circle) : ℝ := 60
def InscribedAngle (c : Circle) : ℝ := 30

-- Define the theorem
theorem arc_measures (c : Circle) :
  (2 * CentralAngle c = 120) ∧ (2 * InscribedAngle c = 60) :=
by sorry

end NUMINAMATH_CALUDE_arc_measures_l1478_147854


namespace NUMINAMATH_CALUDE_remaining_balloons_l1478_147855

def initial_balloons : ℕ := 709
def given_away : ℕ := 221

theorem remaining_balloons : initial_balloons - given_away = 488 := by
  sorry

end NUMINAMATH_CALUDE_remaining_balloons_l1478_147855


namespace NUMINAMATH_CALUDE_square_roots_problem_l1478_147875

theorem square_roots_problem (a : ℝ) (n : ℝ) :
  n > 0 ∧ 
  (∃ x y : ℝ, x * x = n ∧ y * y = n ∧ x = a ∧ y = 2 * a - 6) →
  a = 6 ∧ 
  n = 36 ∧
  (∃ b : ℝ, b * b * b = 10 * 2 + 7 ∧ b = 3) :=
by sorry

end NUMINAMATH_CALUDE_square_roots_problem_l1478_147875


namespace NUMINAMATH_CALUDE_trig_identities_l1478_147823

theorem trig_identities (α : Real) (h : Real.tan α = 3) :
  (3 * Real.sin α + 2 * Real.cos α) / (Real.sin α - 4 * Real.cos α) = -11 ∧
  (5 * Real.cos α ^ 2 - 3 * Real.sin α ^ 2) / (1 + Real.sin α ^ 2) = -11/5 := by
  sorry

end NUMINAMATH_CALUDE_trig_identities_l1478_147823


namespace NUMINAMATH_CALUDE_second_chapter_page_difference_l1478_147840

/-- A book with three chapters -/
structure Book where
  chapter1_pages : ℕ
  chapter2_pages : ℕ
  chapter3_pages : ℕ

/-- The specific book described in the problem -/
def my_book : Book := {
  chapter1_pages := 35
  chapter2_pages := 18
  chapter3_pages := 3
}

/-- Theorem stating the difference in pages between the second and third chapters -/
theorem second_chapter_page_difference (b : Book := my_book) :
  b.chapter2_pages - b.chapter3_pages = 15 := by
  sorry

end NUMINAMATH_CALUDE_second_chapter_page_difference_l1478_147840


namespace NUMINAMATH_CALUDE_quadratic_roots_k_value_l1478_147867

theorem quadratic_roots_k_value (k : ℝ) :
  (∀ x : ℝ, 2 * x^2 - 10 * x + k = 0 ↔ x = 5 + Real.sqrt 15 ∨ x = 5 - Real.sqrt 15) →
  k = 85 / 8 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_k_value_l1478_147867


namespace NUMINAMATH_CALUDE_polynomial_remainder_l1478_147819

theorem polynomial_remainder (m : ℚ) : 
  (∃ (f g : ℚ → ℚ) (R : ℚ), 
    (∀ y : ℚ, y^2 + m*y + 2 = (y - 1) * f y + R) ∧
    (∀ y : ℚ, y^2 + m*y + 2 = (y + 1) * g y + R)) →
  m = 0 := by
sorry

end NUMINAMATH_CALUDE_polynomial_remainder_l1478_147819


namespace NUMINAMATH_CALUDE_fourth_rectangle_perimeter_l1478_147876

theorem fourth_rectangle_perimeter 
  (a b c d : ℝ) 
  (h1 : 2 * (c + b) = 6) 
  (h2 : 2 * (a + c) = 10) 
  (h3 : 2 * (a + d) = 12) : 
  2 * (b + d) = 8 := by
sorry

end NUMINAMATH_CALUDE_fourth_rectangle_perimeter_l1478_147876


namespace NUMINAMATH_CALUDE_marker_difference_l1478_147882

theorem marker_difference (price : ℚ) (hector_count alicia_count : ℕ) : 
  price > 1/100 →  -- More than a penny each
  price * hector_count = 276/100 →  -- Hector paid $2.76
  price * alicia_count = 407/100 →  -- Alicia paid $4.07
  alicia_count - hector_count = 13 := by
  sorry

end NUMINAMATH_CALUDE_marker_difference_l1478_147882


namespace NUMINAMATH_CALUDE_guitar_sales_l1478_147896

theorem guitar_sales (total_revenue : ℕ) (electric_price acoustic_price : ℕ) (electric_sold : ℕ) : 
  total_revenue = 3611 →
  electric_price = 479 →
  acoustic_price = 339 →
  electric_sold = 4 →
  ∃ (acoustic_sold : ℕ), electric_sold + acoustic_sold = 9 ∧ 
    electric_sold * electric_price + acoustic_sold * acoustic_price = total_revenue := by
  sorry

end NUMINAMATH_CALUDE_guitar_sales_l1478_147896


namespace NUMINAMATH_CALUDE_bank_queue_theorem_l1478_147848

/-- Represents a bank queue with simple and long operations -/
structure BankQueue where
  total_people : Nat
  simple_ops : Nat
  long_ops : Nat
  simple_time : Nat
  long_time : Nat

/-- Calculates the minimum wasted person-minutes -/
def min_wasted_time (q : BankQueue) : Nat :=
  sorry

/-- Calculates the maximum wasted person-minutes -/
def max_wasted_time (q : BankQueue) : Nat :=
  sorry

/-- Calculates the expected wasted person-minutes assuming random order -/
def expected_wasted_time (q : BankQueue) : Nat :=
  sorry

/-- Main theorem about the bank queue problem -/
theorem bank_queue_theorem (q : BankQueue) 
  (h1 : q.total_people = 8)
  (h2 : q.simple_ops = 5)
  (h3 : q.long_ops = 3)
  (h4 : q.simple_time = 1)
  (h5 : q.long_time = 5) :
  min_wasted_time q = 40 ∧ 
  max_wasted_time q = 100 ∧ 
  expected_wasted_time q = 84 := by
  sorry

end NUMINAMATH_CALUDE_bank_queue_theorem_l1478_147848


namespace NUMINAMATH_CALUDE_range_of_a_l1478_147833

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, |x + a| < 3 ↔ 2 < x ∧ x < 3) → 
  -5 ≤ a ∧ a ≤ 0 := by
sorry

end NUMINAMATH_CALUDE_range_of_a_l1478_147833


namespace NUMINAMATH_CALUDE_correct_distribution_probability_l1478_147864

def num_guests : ℕ := 3
def num_roll_types : ℕ := 4
def total_rolls : ℕ := 12
def rolls_per_guest : ℕ := 4

def probability_correct_distribution : ℚ := 2 / 103950

theorem correct_distribution_probability :
  let total_ways := (total_rolls.choose rolls_per_guest) * 
                    ((total_rolls - rolls_per_guest).choose rolls_per_guest) *
                    ((total_rolls - 2*rolls_per_guest).choose rolls_per_guest)
  let correct_ways := (num_roll_types.factorial) * 
                      (2^num_roll_types) * 
                      (1^num_roll_types)
  (correct_ways : ℚ) / total_ways = probability_correct_distribution :=
sorry

end NUMINAMATH_CALUDE_correct_distribution_probability_l1478_147864


namespace NUMINAMATH_CALUDE_largest_integer_less_than_100_remainder_5_mod_8_l1478_147858

theorem largest_integer_less_than_100_remainder_5_mod_8 : 
  ∀ n : ℕ, n < 100 ∧ n % 8 = 5 → n ≤ 93 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_integer_less_than_100_remainder_5_mod_8_l1478_147858


namespace NUMINAMATH_CALUDE_greatest_value_quadratic_inequality_l1478_147826

theorem greatest_value_quadratic_inequality :
  ∀ b : ℝ, -b^2 + 8*b - 15 ≥ 0 → b ≤ 5 :=
by sorry

end NUMINAMATH_CALUDE_greatest_value_quadratic_inequality_l1478_147826


namespace NUMINAMATH_CALUDE_correct_system_of_equations_l1478_147804

theorem correct_system_of_equations :
  ∀ (x y : ℕ),
  (x + y = 12) →
  (4 * x + 3 * y = 40) →
  (∀ (a b : ℕ), (a + b = 12 ∧ 4 * a + 3 * b = 40) → (a = x ∧ b = y)) :=
by sorry

end NUMINAMATH_CALUDE_correct_system_of_equations_l1478_147804


namespace NUMINAMATH_CALUDE_soccer_players_count_l1478_147842

theorem soccer_players_count (total_socks : ℕ) (socks_per_player : ℕ) (h1 : total_socks = 22) (h2 : socks_per_player = 2) :
  total_socks / socks_per_player = 11 := by
  sorry

end NUMINAMATH_CALUDE_soccer_players_count_l1478_147842


namespace NUMINAMATH_CALUDE_common_internal_tangent_length_l1478_147813

/-- Given two circles with centers 50 inches apart, with radii 7 inches and 10 inches respectively,
    the length of their common internal tangent is equal to the square root of the difference between
    the square of the distance between their centers and the square of the sum of their radii. -/
theorem common_internal_tangent_length 
  (center_distance : ℝ) 
  (radius1 : ℝ) 
  (radius2 : ℝ) 
  (h1 : center_distance = 50) 
  (h2 : radius1 = 7) 
  (h3 : radius2 = 10) : 
  ∃ (tangent_length : ℝ), tangent_length = Real.sqrt (center_distance^2 - (radius1 + radius2)^2) :=
by sorry

end NUMINAMATH_CALUDE_common_internal_tangent_length_l1478_147813


namespace NUMINAMATH_CALUDE_square_area_ratio_l1478_147832

theorem square_area_ratio (a b : ℝ) (h : 4 * a = 16 * b) : a^2 = 16 * b^2 := by
  sorry

end NUMINAMATH_CALUDE_square_area_ratio_l1478_147832


namespace NUMINAMATH_CALUDE_shiela_drawings_l1478_147859

theorem shiela_drawings (neighbors : ℕ) (drawings_per_neighbor : ℕ) 
  (h1 : neighbors = 6) 
  (h2 : drawings_per_neighbor = 9) : 
  neighbors * drawings_per_neighbor = 54 := by
  sorry

end NUMINAMATH_CALUDE_shiela_drawings_l1478_147859


namespace NUMINAMATH_CALUDE_root_in_interval_l1478_147856

-- Define the function f(x) = x^3 - x - 5
def f (x : ℝ) : ℝ := x^3 - x - 5

-- State the theorem
theorem root_in_interval :
  (f 1 < 0) → (f 2 > 0) → (f 1.5 < 0) →
  ∃ x : ℝ, x ∈ Set.Ioo 1.5 2 ∧ f x = 0 :=
by
  sorry


end NUMINAMATH_CALUDE_root_in_interval_l1478_147856


namespace NUMINAMATH_CALUDE_point_M_coordinates_l1478_147861

-- Define the curve
def f (x : ℝ) : ℝ := 2 * x^2 + 1

-- Define the derivative of the curve
def f' (x : ℝ) : ℝ := 4 * x

theorem point_M_coordinates :
  ∃ (x y : ℝ), f' x = -4 ∧ f x = y ∧ x = -1 ∧ y = 3 :=
by
  sorry

#check point_M_coordinates

end NUMINAMATH_CALUDE_point_M_coordinates_l1478_147861


namespace NUMINAMATH_CALUDE_min_force_to_submerge_cube_l1478_147837

/-- Minimum force required to submerge a cube -/
theorem min_force_to_submerge_cube (V : Real) (ρ_cube ρ_water g : Real) :
  V = 1e-5 →
  ρ_cube = 700 →
  ρ_water = 1000 →
  g = 10 →
  (ρ_water * V * g) - (ρ_cube * V * g) = 0.03 := by
  sorry

end NUMINAMATH_CALUDE_min_force_to_submerge_cube_l1478_147837


namespace NUMINAMATH_CALUDE_train_B_speed_l1478_147891

-- Define the problem parameters
def distance_between_cities : ℝ := 330
def speed_train_A : ℝ := 60
def time_train_A : ℝ := 3
def time_train_B : ℝ := 2

-- Theorem statement
theorem train_B_speed : 
  ∃ (speed_train_B : ℝ),
    speed_train_B * time_train_B + speed_train_A * time_train_A = distance_between_cities ∧
    speed_train_B = 75 := by
  sorry

end NUMINAMATH_CALUDE_train_B_speed_l1478_147891


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1478_147871

/-- Given an arithmetic sequence {a_n} with the specified conditions, 
    prove that a₅ + a₈ + a₁₁ = 15 -/
theorem arithmetic_sequence_sum (a : ℕ → ℤ) 
  (h1 : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1))  -- arithmetic sequence
  (h2 : a 1 + a 4 + a 7 = 39)
  (h3 : a 2 + a 5 + a 8 = 33) :
  a 5 + a 8 + a 11 = 15 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1478_147871


namespace NUMINAMATH_CALUDE_unique_positive_integer_cube_less_than_triple_l1478_147800

theorem unique_positive_integer_cube_less_than_triple :
  ∃! (n : ℕ), n > 0 ∧ n^3 < 3*n :=
by
  sorry

end NUMINAMATH_CALUDE_unique_positive_integer_cube_less_than_triple_l1478_147800


namespace NUMINAMATH_CALUDE_max_crosses_4x10_impossible_5x10_l1478_147831

/-- Represents a rectangular table with crosses placed in its cells. -/
structure CrossTable (m n : ℕ) :=
  (crosses : Fin m → Fin n → Bool)

/-- Checks if a row has an odd number of crosses. -/
def hasOddCrossesInRow (t : CrossTable m n) (row : Fin m) : Prop :=
  Odd (Finset.sum (Finset.univ : Finset (Fin n)) (λ col => if t.crosses row col then 1 else 0))

/-- Checks if a column has an odd number of crosses. -/
def hasOddCrossesInColumn (t : CrossTable m n) (col : Fin n) : Prop :=
  Odd (Finset.sum (Finset.univ : Finset (Fin m)) (λ row => if t.crosses row col then 1 else 0))

/-- Checks if all rows and columns have an odd number of crosses. -/
def hasOddCrossesEverywhere (t : CrossTable m n) : Prop :=
  (∀ row, hasOddCrossesInRow t row) ∧ (∀ col, hasOddCrossesInColumn t col)

/-- Counts the total number of crosses in the table. -/
def totalCrosses (t : CrossTable m n) : ℕ :=
  Finset.sum (Finset.univ : Finset (Fin m)) (λ row =>
    Finset.sum (Finset.univ : Finset (Fin n)) (λ col =>
      if t.crosses row col then 1 else 0))

/-- Theorem for the 4x10 table. -/
theorem max_crosses_4x10 :
  ∀ t : CrossTable 4 10, hasOddCrossesEverywhere t → totalCrosses t ≤ 30 :=
sorry

/-- Theorem for the 5x10 table. -/
theorem impossible_5x10 :
  ¬∃ t : CrossTable 5 10, hasOddCrossesEverywhere t :=
sorry

end NUMINAMATH_CALUDE_max_crosses_4x10_impossible_5x10_l1478_147831


namespace NUMINAMATH_CALUDE_parallel_condition_distance_condition_l1478_147849

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Given point P with coordinates (a+2, 2a-8) -/
def P (a : ℝ) : Point := ⟨a + 2, 2 * a - 8⟩

/-- Point Q with fixed coordinates (1, -2) -/
def Q : Point := ⟨1, -2⟩

/-- Condition 1: Line PQ is parallel to x-axis -/
def parallel_to_x_axis (P Q : Point) : Prop := P.y = Q.y

/-- Condition 2: Distance from P to y-axis is 4 -/
def distance_to_y_axis (P : Point) : ℝ := |P.x|

/-- Theorem for Condition 1 -/
theorem parallel_condition (a : ℝ) : 
  parallel_to_x_axis (P a) Q → P a = ⟨5, -2⟩ := by sorry

/-- Theorem for Condition 2 -/
theorem distance_condition (a : ℝ) : 
  distance_to_y_axis (P a) = 4 → (P a = ⟨4, -4⟩ ∨ P a = ⟨-4, -20⟩) := by sorry

end NUMINAMATH_CALUDE_parallel_condition_distance_condition_l1478_147849


namespace NUMINAMATH_CALUDE_other_diagonal_length_l1478_147881

/-- Represents a rhombus with given diagonals and area -/
structure Rhombus where
  d1 : ℝ  -- Length of the first diagonal
  d2 : ℝ  -- Length of the second diagonal
  area : ℝ -- Area of the rhombus

/-- The area of a rhombus is half the product of its diagonals -/
axiom rhombus_area (r : Rhombus) : r.area = (r.d1 * r.d2) / 2

/-- Given a rhombus with one diagonal of 15 cm and an area of 90 cm²,
    the length of the other diagonal is 12 cm -/
theorem other_diagonal_length :
  ∀ r : Rhombus, r.d1 = 15 ∧ r.area = 90 → r.d2 = 12 := by
  sorry

end NUMINAMATH_CALUDE_other_diagonal_length_l1478_147881
