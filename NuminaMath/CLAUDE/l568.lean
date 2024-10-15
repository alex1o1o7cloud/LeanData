import Mathlib

namespace NUMINAMATH_CALUDE_next_sales_amount_l568_56865

theorem next_sales_amount (initial_sales : ℝ) (initial_royalties : ℝ) (next_royalties : ℝ) (decrease_ratio : ℝ) :
  initial_sales = 20000000 →
  initial_royalties = 8000000 →
  next_royalties = 9000000 →
  decrease_ratio = 0.7916666666666667 →
  ∃ (next_sales : ℝ),
    next_sales = 108000000 ∧
    (next_royalties / next_sales) = (initial_royalties / initial_sales) * (1 - decrease_ratio) :=
by sorry

end NUMINAMATH_CALUDE_next_sales_amount_l568_56865


namespace NUMINAMATH_CALUDE_min_value_quadratic_l568_56809

theorem min_value_quadratic (x : ℝ) : 
  (∀ x, x^2 - 4*x + 5 ≥ 1) ∧ (∃ x, x^2 - 4*x + 5 = 1) :=
sorry

end NUMINAMATH_CALUDE_min_value_quadratic_l568_56809


namespace NUMINAMATH_CALUDE_gcd_45885_30515_l568_56889

theorem gcd_45885_30515 : Nat.gcd 45885 30515 = 10 := by
  sorry

end NUMINAMATH_CALUDE_gcd_45885_30515_l568_56889


namespace NUMINAMATH_CALUDE_election_lead_probability_l568_56882

theorem election_lead_probability (total votes_A votes_B : ℕ) 
  (h_total : total = votes_A + votes_B)
  (h_A_more : votes_A > votes_B) :
  let prob := (votes_A - votes_B) / total
  prob = 1 / 43 :=
by sorry

end NUMINAMATH_CALUDE_election_lead_probability_l568_56882


namespace NUMINAMATH_CALUDE_boys_without_calculators_l568_56877

/-- Proves the number of boys without calculators in Mrs. Allen's class -/
theorem boys_without_calculators 
  (total_boys : ℕ) 
  (total_with_calculators : ℕ) 
  (girls_with_calculators : ℕ) 
  (h1 : total_boys = 20)
  (h2 : total_with_calculators = 25)
  (h3 : girls_with_calculators = 15) :
  total_boys - (total_with_calculators - girls_with_calculators) = 10 := by
  sorry

end NUMINAMATH_CALUDE_boys_without_calculators_l568_56877


namespace NUMINAMATH_CALUDE_tangent_line_slope_l568_56851

noncomputable section

-- Define the functions f and g
def f (x : ℝ) : ℝ := x^2
def g (x : ℝ) : ℝ := -Real.log x

-- Define the second derivative of g
def g'' (x : ℝ) : ℝ := 1 / x^2

-- Theorem statement
theorem tangent_line_slope :
  ∃ (x₁ x₂ : ℝ), x₁ > 0 ∧ x₂ > 0 ∧ 
  (∀ x, f x - f x₁ = (x - x₁) * (2 * x₁)) ∧
  (∀ x, g'' x - g'' x₂ = (x - x₂) * (1 / x₂^2)) ∧
  2 * x₁ = 1 / x₂^2 →
  2 * x₁ = 4 :=
sorry

end

end NUMINAMATH_CALUDE_tangent_line_slope_l568_56851


namespace NUMINAMATH_CALUDE_study_group_probability_l568_56847

theorem study_group_probability (total_members : ℕ) (h1 : total_members > 0) : 
  let women_percentage : ℝ := 0.9
  let lawyer_percentage : ℝ := 0.6
  let women_count : ℝ := women_percentage * total_members
  let women_lawyers_count : ℝ := lawyer_percentage * women_count
  women_lawyers_count / total_members = 0.54 := by sorry

end NUMINAMATH_CALUDE_study_group_probability_l568_56847


namespace NUMINAMATH_CALUDE_square_area_from_adjacent_points_l568_56880

/-- Given two adjacent points (1,1) and (1,5) on a square in a Cartesian coordinate plane,
    the area of the square is 16. -/
theorem square_area_from_adjacent_points :
  let p1 : ℝ × ℝ := (1, 1)
  let p2 : ℝ × ℝ := (1, 5)
  let side_length := Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)
  let area := side_length^2
  area = 16 := by
sorry


end NUMINAMATH_CALUDE_square_area_from_adjacent_points_l568_56880


namespace NUMINAMATH_CALUDE_milkshake_fraction_l568_56884

theorem milkshake_fraction (total : ℚ) (milkshake_fraction : ℚ) 
  (lost : ℚ) (remaining : ℚ) : 
  total = 28 →
  lost = 11 →
  remaining = 1 →
  (1 - milkshake_fraction) * total / 2 = lost + remaining →
  milkshake_fraction = 1 / 7 := by
sorry

end NUMINAMATH_CALUDE_milkshake_fraction_l568_56884


namespace NUMINAMATH_CALUDE_gum_distribution_l568_56876

theorem gum_distribution (cousins : ℕ) (total_gum : ℕ) (gum_per_cousin : ℕ) 
    (h1 : cousins = 4)
    (h2 : total_gum = 20)
    (h3 : total_gum = cousins * gum_per_cousin) :
  gum_per_cousin = 5 := by
  sorry

end NUMINAMATH_CALUDE_gum_distribution_l568_56876


namespace NUMINAMATH_CALUDE_problem_solution_l568_56862

def A : Set ℝ := {x | -3 ≤ x ∧ x ≤ 6}

def B (m : ℝ) : Set ℝ := {x | 6 - m < x ∧ x < m + 3}

theorem problem_solution :
  (∀ m : ℝ,
    (m = 6 → (Aᶜ ∪ B m) = {x | x < -3 ∨ x > 0})) ∧
  (∀ m : ℝ,
    (A ∪ B m = A) ↔ m ≤ 3) :=
sorry

end NUMINAMATH_CALUDE_problem_solution_l568_56862


namespace NUMINAMATH_CALUDE_star_property_l568_56859

-- Define the set of elements
inductive Element : Type
  | one : Element
  | two : Element
  | three : Element
  | four : Element
  | five : Element

-- Define the operation *
def star : Element → Element → Element
  | Element.one, Element.one => Element.one
  | Element.one, Element.two => Element.two
  | Element.one, Element.three => Element.three
  | Element.one, Element.four => Element.four
  | Element.one, Element.five => Element.five
  | Element.two, Element.one => Element.two
  | Element.two, Element.two => Element.one
  | Element.two, Element.three => Element.five
  | Element.two, Element.four => Element.three
  | Element.two, Element.five => Element.four
  | Element.three, Element.one => Element.three
  | Element.three, Element.two => Element.four
  | Element.three, Element.three => Element.two
  | Element.three, Element.four => Element.five
  | Element.three, Element.five => Element.one
  | Element.four, Element.one => Element.four
  | Element.four, Element.two => Element.five
  | Element.four, Element.three => Element.one
  | Element.four, Element.four => Element.two
  | Element.four, Element.five => Element.three
  | Element.five, Element.one => Element.five
  | Element.five, Element.two => Element.three
  | Element.five, Element.three => Element.four
  | Element.five, Element.four => Element.one
  | Element.five, Element.five => Element.two

theorem star_property : 
  star (star Element.three Element.five) (star Element.two Element.four) = Element.three := by
  sorry

end NUMINAMATH_CALUDE_star_property_l568_56859


namespace NUMINAMATH_CALUDE_share_ratio_l568_56893

theorem share_ratio (total amount : ℕ) (a_share : ℕ) : 
  total = 366 → a_share = 122 → a_share / (total - a_share) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_share_ratio_l568_56893


namespace NUMINAMATH_CALUDE_infinite_geometric_series_first_term_l568_56812

theorem infinite_geometric_series_first_term
  (r : ℝ) (S : ℝ) (a : ℝ)
  (h_r : r = 1 / 4)
  (h_S : S = 80)
  (h_sum : S = a / (1 - r)) :
  a = 60 := by
sorry

end NUMINAMATH_CALUDE_infinite_geometric_series_first_term_l568_56812


namespace NUMINAMATH_CALUDE_xyz_value_l568_56849

theorem xyz_value (x y z : ℝ) 
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 36)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 10) :
  x * y * z = 26 / 3 := by
sorry

end NUMINAMATH_CALUDE_xyz_value_l568_56849


namespace NUMINAMATH_CALUDE_birch_tree_arrangement_probability_l568_56815

def number_of_maple_trees : ℕ := 3
def number_of_oak_trees : ℕ := 4
def number_of_birch_trees : ℕ := 5
def total_trees : ℕ := number_of_maple_trees + number_of_oak_trees + number_of_birch_trees

def probability_no_adjacent_birch : ℚ := 7 / 99

theorem birch_tree_arrangement_probability :
  probability_no_adjacent_birch = 
    (Nat.choose (number_of_maple_trees + number_of_oak_trees + 1) number_of_birch_trees) / 
    (Nat.choose total_trees number_of_birch_trees) :=
sorry

end NUMINAMATH_CALUDE_birch_tree_arrangement_probability_l568_56815


namespace NUMINAMATH_CALUDE_det_A_squared_minus_3A_l568_56861

def A : Matrix (Fin 2) (Fin 2) ℝ := !![2, 3; 2, 2]

theorem det_A_squared_minus_3A : Matrix.det ((A ^ 2) - 3 • A) = 10 := by
  sorry

end NUMINAMATH_CALUDE_det_A_squared_minus_3A_l568_56861


namespace NUMINAMATH_CALUDE_p_necessary_not_sufficient_for_q_l568_56856

-- Define the conditions
def condition_p (x : ℝ) : Prop := x^2 < x
def condition_q (x : ℝ) : Prop := 1/x > 2

-- Theorem statement
theorem p_necessary_not_sufficient_for_q :
  (∀ x : ℝ, condition_q x → condition_p x) ∧
  (∃ x : ℝ, condition_p x ∧ ¬condition_q x) :=
by sorry

end NUMINAMATH_CALUDE_p_necessary_not_sufficient_for_q_l568_56856


namespace NUMINAMATH_CALUDE_num_subcommittee_pairs_l568_56872

/-- Represents the number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := sorry

/-- The total number of committee members -/
def total_members : ℕ := 12

/-- The number of teachers in the committee -/
def teachers : ℕ := 5

/-- The size of each subcommittee -/
def subcommittee_size : ℕ := 4

/-- The number of subcommittees to form -/
def num_subcommittees : ℕ := 2

/-- Calculates the number of subcommittees with at least one teacher -/
def subcommittees_with_teacher (members teachers : ℕ) : ℕ :=
  choose members subcommittee_size - choose (members - teachers) subcommittee_size

/-- The main theorem stating the number of distinct pairs of subcommittees -/
theorem num_subcommittee_pairs : 
  subcommittees_with_teacher total_members teachers * 
  subcommittees_with_teacher (total_members - subcommittee_size) (teachers - 1) = 29900 := by
  sorry

end NUMINAMATH_CALUDE_num_subcommittee_pairs_l568_56872


namespace NUMINAMATH_CALUDE_always_true_inequalities_l568_56874

theorem always_true_inequalities (a b c : ℝ) (h1 : a < 0) (h2 : a < b) (h3 : b < c) :
  (a + b < b + c) ∧ (c / a < 1) := by
  sorry

end NUMINAMATH_CALUDE_always_true_inequalities_l568_56874


namespace NUMINAMATH_CALUDE_sum_of_triple_g_roots_l568_56803

def g (x : ℝ) : ℝ := -x^2 + 6*x - 8

theorem sum_of_triple_g_roots (h : ∀ x, g x ≤ g 3) :
  ∃ S : Finset ℝ, (∀ x ∈ S, g (g (g x)) = 2) ∧ 
                  (∀ x, g (g (g x)) = 2 → x ∈ S) ∧
                  (S.sum id = 6) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_triple_g_roots_l568_56803


namespace NUMINAMATH_CALUDE_dot_product_equals_ten_l568_56824

/-- Given two vectors a and b in ℝ², prove that their dot product is 10 -/
theorem dot_product_equals_ten (a b : ℝ × ℝ) :
  a = (-2, -6) →
  ‖b‖ = Real.sqrt 10 →
  Real.cos (60 * π / 180) = (a.1 * b.1 + a.2 * b.2) / (‖a‖ * ‖b‖) →
  a.1 * b.1 + a.2 * b.2 = 10 := by
  sorry

#check dot_product_equals_ten

end NUMINAMATH_CALUDE_dot_product_equals_ten_l568_56824


namespace NUMINAMATH_CALUDE_cookie_ratio_proof_l568_56833

def cookie_problem (initial_cookies brother_cookies final_cookies : ℕ) : Prop :=
  ∃ (mother_cookies : ℕ),
    let remaining_after_brother := initial_cookies - brother_cookies
    let total_before_sister := remaining_after_brother + mother_cookies
    let sister_cookies := (2 : ℚ) / 3 * total_before_sister
    (total_before_sister - sister_cookies = final_cookies) ∧
    (mother_cookies : ℚ) / brother_cookies = 1 / 2

theorem cookie_ratio_proof :
  cookie_problem 20 10 5 :=
sorry

end NUMINAMATH_CALUDE_cookie_ratio_proof_l568_56833


namespace NUMINAMATH_CALUDE_min_value_x_plus_2y_l568_56829

theorem min_value_x_plus_2y (x y : ℝ) (hx : x + 2 > 0) (hy : y + 2 > 0) 
  (h : 3 / (x + 2) + 3 / (y + 2) = 1) : 
  x + 2*y ≥ 3 + 6 * Real.sqrt 2 := by
  sorry

#check min_value_x_plus_2y

end NUMINAMATH_CALUDE_min_value_x_plus_2y_l568_56829


namespace NUMINAMATH_CALUDE_jacket_price_restoration_l568_56837

/-- Represents the price reduction and restoration process of a jacket --/
theorem jacket_price_restoration (initial_price : ℝ) (h_pos : initial_price > 0) :
  let price_after_first_reduction := initial_price * (1 - 0.25)
  let price_after_second_reduction := price_after_first_reduction * (1 - 0.10)
  let required_increase_percentage := (initial_price / price_after_second_reduction - 1) * 100
  ∃ ε > 0, abs (required_increase_percentage - 48.15) < ε :=
by sorry

end NUMINAMATH_CALUDE_jacket_price_restoration_l568_56837


namespace NUMINAMATH_CALUDE_minimum_additional_wins_l568_56895

def puppy_cost : ℕ := 1000
def weekly_prize : ℕ := 100
def initial_wins : ℕ := 2

theorem minimum_additional_wins : 
  ∃ (n : ℕ), n = (puppy_cost - initial_wins * weekly_prize) / weekly_prize ∧ 
  n * weekly_prize + initial_wins * weekly_prize ≥ puppy_cost ∧
  ∀ m : ℕ, m < n → m * weekly_prize + initial_wins * weekly_prize < puppy_cost :=
by sorry

end NUMINAMATH_CALUDE_minimum_additional_wins_l568_56895


namespace NUMINAMATH_CALUDE_b_95_mod_49_l568_56811

def b (n : ℕ) : ℕ := 5^n + 7^n + 3

theorem b_95_mod_49 : b 95 % 49 = 5 := by
  sorry

end NUMINAMATH_CALUDE_b_95_mod_49_l568_56811


namespace NUMINAMATH_CALUDE_tangent_line_at_2_l568_56820

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 4*x^2 + 5*x - 4

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 3*x^2 - 8*x + 5

-- Theorem statement
theorem tangent_line_at_2 :
  let x₀ : ℝ := 2
  let y₀ : ℝ := f x₀
  let m : ℝ := f' x₀
  ∀ x y : ℝ, (y - y₀ = m * (x - x₀)) ↔ (x - y - 4 = 0) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_at_2_l568_56820


namespace NUMINAMATH_CALUDE_rose_can_afford_supplies_l568_56804

def budget : ℝ := 30

def paintbrush_cost : ℝ := 2.40
def paints_cost : ℝ := 9.20
def easel_cost : ℝ := 6.50
def canvas_cost : ℝ := 12.25
def drawing_pad_cost : ℝ := 4.75

def discount_rate : ℝ := 0.15

def total_cost_before_discount : ℝ := 
  paintbrush_cost + paints_cost + easel_cost + canvas_cost + drawing_pad_cost

def discount_amount : ℝ := discount_rate * total_cost_before_discount

def total_cost_after_discount : ℝ := total_cost_before_discount - discount_amount

theorem rose_can_afford_supplies : 
  total_cost_after_discount ≤ budget ∧ 
  budget - total_cost_after_discount = 0.165 := by sorry

end NUMINAMATH_CALUDE_rose_can_afford_supplies_l568_56804


namespace NUMINAMATH_CALUDE_derivative_of_f_at_2_l568_56894

-- Define the function f(x) = x
def f (x : ℝ) : ℝ := x

-- State the theorem
theorem derivative_of_f_at_2 : 
  HasDerivAt f 1 2 := by sorry

end NUMINAMATH_CALUDE_derivative_of_f_at_2_l568_56894


namespace NUMINAMATH_CALUDE_middle_term_is_average_l568_56805

/-- An arithmetic sequence with 5 terms -/
structure ArithmeticSequence5 where
  a : ℝ  -- first term
  b : ℝ  -- second term
  c : ℝ  -- third term (middle term)
  d : ℝ  -- fourth term
  e : ℝ  -- fifth term
  is_arithmetic : ∃ (r : ℝ), b - a = r ∧ c - b = r ∧ d - c = r ∧ e - d = r

/-- The theorem stating that the middle term of a 5-term arithmetic sequence
    is the average of the first and last terms -/
theorem middle_term_is_average (seq : ArithmeticSequence5) (h1 : seq.a = 23) (h2 : seq.e = 47) :
  seq.c = 35 := by
  sorry

end NUMINAMATH_CALUDE_middle_term_is_average_l568_56805


namespace NUMINAMATH_CALUDE_triangle_inequality_l568_56826

theorem triangle_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (htriangle : a + b > c ∧ b + c > a ∧ c + a > b) :
  let S := (a + b + c) / 2
  2 * S * (Real.sqrt (S - a) + Real.sqrt (S - b) + Real.sqrt (S - c)) ≤
  3 * (Real.sqrt (b * c * (S - a)) + Real.sqrt (c * a * (S - b)) + Real.sqrt (a * b * (S - c))) := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l568_56826


namespace NUMINAMATH_CALUDE_min_days_to_plant_trees_eight_is_min_days_l568_56817

theorem min_days_to_plant_trees (n : ℕ) : n ≥ 8 ↔ 2 * (2^n - 1) ≥ 100 := by
  sorry

theorem eight_is_min_days : ∃ (n : ℕ), n = 8 ∧ 2 * (2^n - 1) ≥ 100 ∧ ∀ (m : ℕ), m < n → 2 * (2^m - 1) < 100 := by
  sorry

end NUMINAMATH_CALUDE_min_days_to_plant_trees_eight_is_min_days_l568_56817


namespace NUMINAMATH_CALUDE_cos_180_degrees_l568_56808

theorem cos_180_degrees : Real.cos (π) = -1 := by
  sorry

end NUMINAMATH_CALUDE_cos_180_degrees_l568_56808


namespace NUMINAMATH_CALUDE_parallel_lines_imply_a_eq_neg_two_l568_56834

/-- Two lines are parallel if and only if their slopes are equal -/
axiom parallel_lines_equal_slopes {m₁ m₂ b₁ b₂ : ℝ} :
  (∀ x y : ℝ, y = m₁ * x + b₁ ↔ y = m₂ * x + b₂) ↔ m₁ = m₂

/-- The line ax + y - 1 = 0 is parallel to the line 2x - y + 2 = 0 -/
def lines_are_parallel (a : ℝ) : Prop :=
  ∀ x y : ℝ, y = -a * x + 1 ↔ y = 2 * x + 2

theorem parallel_lines_imply_a_eq_neg_two :
  ∀ a : ℝ, lines_are_parallel a → a = -2 :=
by sorry

end NUMINAMATH_CALUDE_parallel_lines_imply_a_eq_neg_two_l568_56834


namespace NUMINAMATH_CALUDE_log_power_base_l568_56881

theorem log_power_base (a k P : ℝ) (ha : a > 0) (ha1 : a ≠ 1) :
  Real.log P / Real.log (a^k) = Real.log P / Real.log a / k := by
  sorry

end NUMINAMATH_CALUDE_log_power_base_l568_56881


namespace NUMINAMATH_CALUDE_twelve_valid_grids_l568_56879

/-- Represents a 3x3 grid filled with numbers 1 to 9 -/
def Grid := Fin 3 → Fin 3 → Fin 9

/-- Checks if the grid is valid according to the given conditions -/
def is_valid_grid (g : Grid) : Prop :=
  g 1 1 = 0 ∧  -- 1 is in top-left corner
  g 3 3 = 8 ∧  -- 9 is in bottom-right corner
  g 2 2 = 3 ∧  -- 4 is in center
  (∀ i j, i < j → g i 1 < g j 1) ∧  -- increasing top to bottom in first column
  (∀ i j, i < j → g i 2 < g j 2) ∧  -- increasing top to bottom in second column
  (∀ i j, i < j → g i 3 < g j 3) ∧  -- increasing top to bottom in third column
  (∀ i j, i < j → g 1 i < g 1 j) ∧  -- increasing left to right in first row
  (∀ i j, i < j → g 2 i < g 2 j) ∧  -- increasing left to right in second row
  (∀ i j, i < j → g 3 i < g 3 j)    -- increasing left to right in third row

/-- The number of valid grid arrangements -/
def num_valid_grids : ℕ := sorry

/-- Theorem stating that there are exactly 12 valid grid arrangements -/
theorem twelve_valid_grids : num_valid_grids = 12 := by sorry

end NUMINAMATH_CALUDE_twelve_valid_grids_l568_56879


namespace NUMINAMATH_CALUDE_perfect_squares_as_sum_of_odd_composites_l568_56870

def is_odd_composite (n : ℕ) : Prop := n % 2 = 1 ∧ ∃ a b, a > 1 ∧ b > 1 ∧ n = a * b

def is_sum_of_three_odd_composites (n : ℕ) : Prop :=
  ∃ a b c, is_odd_composite a ∧ is_odd_composite b ∧ is_odd_composite c ∧ n = a + b + c

def perfect_square_set : Set ℕ := {n | ∃ k : ℕ, k ≥ 3 ∧ n = (2 * k + 1)^2}

theorem perfect_squares_as_sum_of_odd_composites :
  ∀ n : ℕ, n ∈ perfect_square_set ↔ is_sum_of_three_odd_composites n :=
sorry

end NUMINAMATH_CALUDE_perfect_squares_as_sum_of_odd_composites_l568_56870


namespace NUMINAMATH_CALUDE_some_number_value_l568_56883

theorem some_number_value (x : ℝ) : (50 + 20/x) * x = 4520 → x = 90 := by
  sorry

end NUMINAMATH_CALUDE_some_number_value_l568_56883


namespace NUMINAMATH_CALUDE_inheritance_investment_rate_l568_56869

theorem inheritance_investment_rate 
  (inheritance : ℝ) 
  (first_investment : ℝ) 
  (second_rate : ℝ) 
  (total_interest : ℝ) : 
  inheritance = 12000 →
  first_investment = 5000 →
  second_rate = 0.08 →
  total_interest = 860 →
  ∃ (r : ℝ), 
    r * first_investment + second_rate * (inheritance - first_investment) = total_interest ∧ 
    r = 0.06 := by
  sorry

end NUMINAMATH_CALUDE_inheritance_investment_rate_l568_56869


namespace NUMINAMATH_CALUDE_find_B_l568_56846

theorem find_B (A C B : ℤ) (h1 : C - A = 204) (h2 : A = 520) (h3 : B + 179 = C) : B = 545 := by
  sorry

end NUMINAMATH_CALUDE_find_B_l568_56846


namespace NUMINAMATH_CALUDE_fence_painting_combinations_l568_56892

theorem fence_painting_combinations :
  let color_choices : ℕ := 6
  let method_choices : ℕ := 3
  let finish_choices : ℕ := 2
  color_choices * method_choices * finish_choices = 36 :=
by sorry

end NUMINAMATH_CALUDE_fence_painting_combinations_l568_56892


namespace NUMINAMATH_CALUDE_sinusoidal_vertical_shift_l568_56857

/-- For a sinusoidal function y = a sin(bx + c) + d, 
    if it oscillates between 4 and -2, then d = 1 -/
theorem sinusoidal_vertical_shift 
  (a b c d : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0) 
  (h_oscillation : ∀ x, -2 ≤ a * Real.sin (b * x + c) + d ∧ 
                        a * Real.sin (b * x + c) + d ≤ 4) : 
  d = 1 := by
  sorry

end NUMINAMATH_CALUDE_sinusoidal_vertical_shift_l568_56857


namespace NUMINAMATH_CALUDE_same_number_probability_l568_56866

/-- The upper bound for the selected numbers -/
def upperBound : ℕ := 200

/-- Alice's number is a multiple of this value -/
def aliceMultiple : ℕ := 16

/-- Alan's number is a multiple of this value -/
def alanMultiple : ℕ := 28

/-- The probability of Alice and Alan selecting the same number -/
def sameProbability : ℚ := 1 / 84

theorem same_number_probability :
  (∃ (n : ℕ), n < upperBound ∧ n % aliceMultiple = 0 ∧ n % alanMultiple = 0) ∧
  (∀ (m : ℕ), m < upperBound → m % aliceMultiple = 0 → m % alanMultiple = 0 → m = lcm aliceMultiple alanMultiple) →
  sameProbability = (Nat.card {n : ℕ | n < upperBound ∧ n % aliceMultiple = 0 ∧ n % alanMultiple = 0}) /
    ((Nat.card {n : ℕ | n < upperBound ∧ n % aliceMultiple = 0}) * (Nat.card {n : ℕ | n < upperBound ∧ n % alanMultiple = 0})) :=
by sorry

end NUMINAMATH_CALUDE_same_number_probability_l568_56866


namespace NUMINAMATH_CALUDE_compatible_polygons_exist_l568_56841

/-- A simple polygon is a polygon that does not intersect itself. -/
def SimplePolygon : Type := sorry

/-- Two simple polygons are compatible if there exists a positive integer k such that 
    each polygon can be partitioned into k congruent polygons similar to the other one. -/
def compatible (P Q : SimplePolygon) : Prop := sorry

/-- The number of sides of a simple polygon. -/
def num_sides (P : SimplePolygon) : ℕ := sorry

theorem compatible_polygons_exist (m n : ℕ) (hm : Even m) (hn : Even n) 
  (hm_ge_4 : m ≥ 4) (hn_ge_4 : n ≥ 4) : 
  ∃ (P Q : SimplePolygon), num_sides P = m ∧ num_sides Q = n ∧ compatible P Q := by
  sorry

end NUMINAMATH_CALUDE_compatible_polygons_exist_l568_56841


namespace NUMINAMATH_CALUDE_sum_of_C_and_D_l568_56842

/-- The number of four-digit numbers that are both odd and divisible by 5 -/
def C : ℕ := 900

/-- The number of four-digit numbers that are divisible by 25 -/
def D : ℕ := 360

/-- Theorem: The sum of C and D is 1260 -/
theorem sum_of_C_and_D : C + D = 1260 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_C_and_D_l568_56842


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l568_56822

theorem right_triangle_hypotenuse : 
  ∀ (a b c : ℝ), 
    a = 5 → 
    b = 12 → 
    c^2 = a^2 + b^2 → 
    c = 13 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l568_56822


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l568_56850

/-- Given a positive term geometric sequence {a_n} with common ratio q,
    prove that if 2a_5 - 3a_4 = 2a_3, then q = 2 -/
theorem geometric_sequence_common_ratio
  (a : ℕ → ℝ)  -- Sequence of real numbers indexed by natural numbers
  (q : ℝ)      -- Common ratio
  (h_pos : ∀ n, a n > 0)  -- Positive term sequence
  (h_geom : ∀ n, a (n + 1) = q * a n)  -- Geometric sequence property
  (h_eq : 2 * a 5 - 3 * a 4 = 2 * a 3)  -- Given equation
  : q = 2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l568_56850


namespace NUMINAMATH_CALUDE_smallest_multiple_21_g_gt_21_l568_56801

/-- g(n) returns the smallest integer m such that m! is divisible by n -/
def g (n : ℕ) : ℕ := sorry

/-- 483 is the smallest multiple of 21 for which g(n) > 21 -/
theorem smallest_multiple_21_g_gt_21 : ∀ n : ℕ, n % 21 = 0 → g n ≤ 21 → n < 483 :=
sorry

end NUMINAMATH_CALUDE_smallest_multiple_21_g_gt_21_l568_56801


namespace NUMINAMATH_CALUDE_triangle_angle_equality_l568_56825

open Real

theorem triangle_angle_equality (a b c : ℝ) (α β γ : ℝ) (R : ℝ) :
  a > 0 → b > 0 → c > 0 → R > 0 →
  α > 0 → β > 0 → γ > 0 →
  α + β + γ = pi →
  (a * cos α + b * cos β + c * cos γ) / (a * sin β + b * sin γ + c * sin α) = (a + b + c) / (9 * R) →
  α = pi / 3 ∧ β = pi / 3 ∧ γ = pi / 3 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_equality_l568_56825


namespace NUMINAMATH_CALUDE_system_solution_l568_56888

theorem system_solution (k : ℝ) : 
  (∃ x y : ℝ, x + y = 5*k ∧ x - 2*y = -k ∧ 2*x - y = 8) → k = 2 := by
sorry

end NUMINAMATH_CALUDE_system_solution_l568_56888


namespace NUMINAMATH_CALUDE_walts_investment_rate_l568_56821

/-- Given Walt's investment scenario, prove the unknown interest rate --/
theorem walts_investment_rate : 
  let total_amount : ℝ := 9000
  let known_rate : ℝ := 0.08
  let known_investment : ℝ := 4000
  let total_interest : ℝ := 770
  let unknown_investment : ℝ := total_amount - known_investment
  ∃ (unknown_rate : ℝ),
    known_investment * known_rate + unknown_investment * unknown_rate = total_interest ∧
    unknown_rate = 0.09
  := by sorry

end NUMINAMATH_CALUDE_walts_investment_rate_l568_56821


namespace NUMINAMATH_CALUDE_a_in_M_neither_sufficient_nor_necessary_for_a_in_N_l568_56814

-- Define sets M and N
def M : Set ℝ := {x | 0 < x ∧ x ≤ 3}
def N : Set ℝ := {x | -1 < x ∧ x ≤ 2}

-- Theorem stating that "a ∈ M" is neither sufficient nor necessary for "a ∈ N"
theorem a_in_M_neither_sufficient_nor_necessary_for_a_in_N :
  (∃ a : ℝ, a ∈ M ∧ a ∉ N) ∧ (∃ a : ℝ, a ∈ N ∧ a ∉ M) := by
  sorry

end NUMINAMATH_CALUDE_a_in_M_neither_sufficient_nor_necessary_for_a_in_N_l568_56814


namespace NUMINAMATH_CALUDE_registration_combinations_l568_56818

/-- The number of people signing up for sports competitions -/
def num_people : ℕ := 4

/-- The number of available sports competitions -/
def num_sports : ℕ := 3

/-- 
Theorem: Given 'num_people' people and 'num_sports' sports competitions, 
where each person must choose exactly one event, the total number of 
possible registration combinations is 'num_sports' raised to the power of 'num_people'.
-/
theorem registration_combinations : 
  (num_sports : ℕ) ^ (num_people : ℕ) = 81 := by
  sorry

#eval (num_sports : ℕ) ^ (num_people : ℕ)

end NUMINAMATH_CALUDE_registration_combinations_l568_56818


namespace NUMINAMATH_CALUDE_population_ratio_x_to_z_l568_56802

/-- The population ratio between two cities -/
structure PopulationRatio :=
  (city1 : ℕ)
  (city2 : ℕ)

/-- The population of three cities X, Y, and Z -/
structure CityPopulations :=
  (X : ℕ)
  (Y : ℕ)
  (Z : ℕ)

/-- Given the populations of cities X, Y, and Z, where X has 3 times the population of Y,
    and Y has twice the population of Z, prove that the ratio of X to Z is 6:1 -/
theorem population_ratio_x_to_z (pop : CityPopulations)
  (h1 : pop.X = 3 * pop.Y)
  (h2 : pop.Y = 2 * pop.Z) :
  PopulationRatio.mk pop.X pop.Z = PopulationRatio.mk 6 1 := by
  sorry

end NUMINAMATH_CALUDE_population_ratio_x_to_z_l568_56802


namespace NUMINAMATH_CALUDE_cinnamon_swirl_division_l568_56852

theorem cinnamon_swirl_division (total_pieces : ℕ) (num_people : ℕ) (pieces_per_person : ℕ) : 
  total_pieces = 12 → 
  num_people = 3 → 
  total_pieces = num_people * pieces_per_person → 
  pieces_per_person = 4 := by
sorry

end NUMINAMATH_CALUDE_cinnamon_swirl_division_l568_56852


namespace NUMINAMATH_CALUDE_solution_satisfies_system_l568_56819

/-- Given a system of linear equations:
    3x₁ - x₂ + 3x₃ = 5
    2x₁ - x₂ + 4x₃ = 5
    x₁ + 2x₂ - 3x₃ = 0
    Prove that (1, 1, 1) is a solution. -/
theorem solution_satisfies_system :
  let x₁ : ℝ := 1
  let x₂ : ℝ := 1
  let x₃ : ℝ := 1
  (3 * x₁ - x₂ + 3 * x₃ = 5) ∧
  (2 * x₁ - x₂ + 4 * x₃ = 5) ∧
  (x₁ + 2 * x₂ - 3 * x₃ = 0) :=
by sorry

#check solution_satisfies_system

end NUMINAMATH_CALUDE_solution_satisfies_system_l568_56819


namespace NUMINAMATH_CALUDE_division_problem_l568_56855

theorem division_problem : (0.25 / 0.005) / 0.1 = 500 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l568_56855


namespace NUMINAMATH_CALUDE_point_in_second_quadrant_m_range_l568_56845

/-- A point in the Cartesian plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of the second quadrant -/
def SecondQuadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y > 0

/-- The theorem stating the range of m for a point in the second quadrant -/
theorem point_in_second_quadrant_m_range (m : ℝ) :
  let p := Point.mk (m - 3) (m + 1)
  SecondQuadrant p ↔ -1 < m ∧ m < 3 := by
  sorry

end NUMINAMATH_CALUDE_point_in_second_quadrant_m_range_l568_56845


namespace NUMINAMATH_CALUDE_property_tax_increase_l568_56890

/-- Calculates the property tax increase when the assessed value changes, given a fixed tax rate. -/
theorem property_tax_increase 
  (tax_rate : ℝ) 
  (initial_value : ℝ) 
  (new_value : ℝ) 
  (h1 : tax_rate = 0.1)
  (h2 : initial_value = 20000)
  (h3 : new_value = 28000) :
  new_value * tax_rate - initial_value * tax_rate = 800 :=
by sorry

end NUMINAMATH_CALUDE_property_tax_increase_l568_56890


namespace NUMINAMATH_CALUDE_prob_sum_27_l568_56816

/-- Represents a die with 20 faces -/
structure Die :=
  (faces : Finset ℕ)
  (fair : faces.card = 20)

/-- The first die with faces 1 through 19 -/
def die1 : Die :=
  { faces := Finset.range 20 \ {20},
    fair := sorry }

/-- The second die with faces 1 through 7 and 9 through 21 -/
def die2 : Die :=
  { faces := (Finset.range 22 \ {0, 8}),
    fair := sorry }

/-- The set of all possible outcomes when rolling both dice -/
def allOutcomes : Finset (ℕ × ℕ) :=
  die1.faces.product die2.faces

/-- The set of outcomes that sum to 27 -/
def sumTo27 : Finset (ℕ × ℕ) :=
  allOutcomes.filter (fun p => p.1 + p.2 = 27)

/-- The probability of rolling a sum of 27 -/
def probSum27 : ℚ :=
  sumTo27.card / allOutcomes.card

theorem prob_sum_27 : probSum27 = 3 / 100 := by sorry

end NUMINAMATH_CALUDE_prob_sum_27_l568_56816


namespace NUMINAMATH_CALUDE_nested_expression_simplification_l568_56858

theorem nested_expression_simplification (x : ℝ) : 1 - (1 + (1 - (1 + (1 - (1 - x))))) = 1 - x := by
  sorry

end NUMINAMATH_CALUDE_nested_expression_simplification_l568_56858


namespace NUMINAMATH_CALUDE_soccer_ball_min_cost_l568_56807

/-- Represents the purchase of soccer balls -/
structure SoccerBallPurchase where
  brand_a_price : ℕ
  brand_b_price : ℕ
  total_balls : ℕ
  min_brand_a : ℕ
  max_cost : ℕ

/-- Calculates the total cost for a given number of brand A balls -/
def total_cost (p : SoccerBallPurchase) (brand_a_count : ℕ) : ℕ :=
  p.brand_a_price * brand_a_count + p.brand_b_price * (p.total_balls - brand_a_count)

/-- Theorem stating the minimum cost of the soccer ball purchase -/
theorem soccer_ball_min_cost (p : SoccerBallPurchase)
  (h1 : p.brand_a_price = p.brand_b_price + 10)
  (h2 : 2 * p.brand_a_price + 3 * p.brand_b_price = 220)
  (h3 : p.total_balls = 60)
  (h4 : p.min_brand_a = 43)
  (h5 : p.max_cost = 2850) :
  ∃ (m : ℕ), m ≥ p.min_brand_a ∧ m ≤ p.total_balls ∧
    total_cost p m ≤ p.max_cost ∧
    ∀ (n : ℕ), n ≥ p.min_brand_a → n ≤ p.total_balls →
      total_cost p n ≤ p.max_cost → total_cost p m ≤ total_cost p n :=
by sorry

end NUMINAMATH_CALUDE_soccer_ball_min_cost_l568_56807


namespace NUMINAMATH_CALUDE_opposite_numbers_problem_l568_56823

theorem opposite_numbers_problem (x y : ℚ) 
  (h1 : x + y = 0)  -- x and y are opposite numbers
  (h2 : x - y = 3)  -- given condition
  : x^2 + 2*x*y + 1 = -5/4 := by sorry

end NUMINAMATH_CALUDE_opposite_numbers_problem_l568_56823


namespace NUMINAMATH_CALUDE_exists_good_interval_and_fixed_point_l568_56835

/-- Definition of a good interval for a function f on [a, b] --/
def is_good_interval (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  (∀ x ∈ Set.Icc a b, f x ∈ Set.Icc a b) ∨ 
  (∀ x ∈ Set.Icc a b, f x ∉ Set.Icc a b)

/-- Main theorem --/
theorem exists_good_interval_and_fixed_point 
  (f : ℝ → ℝ) (hf : Continuous f) 
  (h : ∀ a b : ℝ, a < b → f a - f b > b - a) : 
  (∃ c d : ℝ, c < d ∧ is_good_interval f c d) ∧ 
  (∃ x₀ : ℝ, f x₀ = x₀) := by
  sorry


end NUMINAMATH_CALUDE_exists_good_interval_and_fixed_point_l568_56835


namespace NUMINAMATH_CALUDE_rice_bags_sold_l568_56844

theorem rice_bags_sold (initial_stock restocked final_stock : ℕ) 
  (h1 : initial_stock = 55)
  (h2 : restocked = 132)
  (h3 : final_stock = 164) :
  initial_stock + restocked - final_stock = 23 := by
  sorry

end NUMINAMATH_CALUDE_rice_bags_sold_l568_56844


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_of_quadratic_roots_sum_of_reciprocals_specific_quadratic_l568_56806

theorem sum_of_reciprocals_of_quadratic_roots (a b c : ℝ) (h : a ≠ 0) :
  let r₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let r₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  (a * r₁^2 + b * r₁ + c = 0) ∧ (a * r₂^2 + b * r₂ + c = 0) →
  1/r₁ + 1/r₂ = -b/c :=
by sorry

theorem sum_of_reciprocals_specific_quadratic :
  let r₁ := (17 + Real.sqrt (17^2 - 4*8)) / 2
  let r₂ := (17 - Real.sqrt (17^2 - 4*8)) / 2
  (r₁^2 - 17*r₁ + 8 = 0) ∧ (r₂^2 - 17*r₂ + 8 = 0) →
  1/r₁ + 1/r₂ = 17/8 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_of_quadratic_roots_sum_of_reciprocals_specific_quadratic_l568_56806


namespace NUMINAMATH_CALUDE_inequality_proof_l568_56899

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (2 * x^2) / (y + z) + (2 * y^2) / (z + x) + (2 * z^2) / (x + y) ≥ x + y + z := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l568_56899


namespace NUMINAMATH_CALUDE_interior_lattice_points_collinear_l568_56896

/-- A lattice point in the plane -/
structure LatticePoint where
  x : ℤ
  y : ℤ

/-- A triangle in the plane -/
structure Triangle where
  v1 : LatticePoint
  v2 : LatticePoint
  v3 : LatticePoint

/-- Check if a point is inside a triangle -/
def isInside (p : LatticePoint) (t : Triangle) : Prop := sorry

/-- Check if a point is on the boundary of a triangle -/
def isOnBoundary (p : LatticePoint) (t : Triangle) : Prop := sorry

/-- Check if points are collinear -/
def areCollinear (p1 p2 p3 p4 : LatticePoint) : Prop := sorry

/-- The main theorem -/
theorem interior_lattice_points_collinear (t : Triangle) 
  (h1 : ∀ p, isOnBoundary p t → (p = t.v1 ∨ p = t.v2 ∨ p = t.v3))
  (h2 : ∃ p1 p2 p3 p4, isInside p1 t ∧ isInside p2 t ∧ isInside p3 t ∧ isInside p4 t ∧
    ∀ p, isInside p t → (p = p1 ∨ p = p2 ∨ p = p3 ∨ p = p4)) :
  ∃ p1 p2 p3 p4, isInside p1 t ∧ isInside p2 t ∧ isInside p3 t ∧ isInside p4 t ∧
    areCollinear p1 p2 p3 p4 := by
  sorry

end NUMINAMATH_CALUDE_interior_lattice_points_collinear_l568_56896


namespace NUMINAMATH_CALUDE_polygon_sides_l568_56848

theorem polygon_sides (n : ℕ) : n ≥ 3 →
  (n - 2) * 180 = 3 * 360 - 180 → n = 7 := by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_l568_56848


namespace NUMINAMATH_CALUDE_sequence_problem_l568_56898

theorem sequence_problem (x y : ℝ) : 
  (∃ r : ℝ, y - 1 - 1 = 1 - 2*x ∧ 1 - 2*x = r) →  -- arithmetic sequence condition
  (∃ q : ℝ, |x+1| / (y+3) = q ∧ |x-1| / |x+1| = q) →  -- geometric sequence condition
  (x+1)*(y+1) = 4 ∨ (x+1)*(y+1) = 2*(Real.sqrt 17 - 3) := by
sorry

end NUMINAMATH_CALUDE_sequence_problem_l568_56898


namespace NUMINAMATH_CALUDE_two_digit_R_equal_l568_56854

/-- R(n) is the sum of remainders when n is divided by 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, and 12 -/
def R (n : ℕ) : ℕ :=
  (n % 2) + (n % 3) + (n % 4) + (n % 5) + (n % 6) + (n % 7) + (n % 8) + (n % 9) + (n % 10) + (n % 11) + (n % 12)

/-- A two-digit positive integer is between 10 and 99, inclusive -/
def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

/-- The theorem states that there are exactly 2 two-digit positive integers n such that R(n) = R(n+2) -/
theorem two_digit_R_equal : ∃! (s : Finset ℕ), 
  (∀ n ∈ s, is_two_digit n ∧ R n = R (n + 2)) ∧ Finset.card s = 2 := by
  sorry

end NUMINAMATH_CALUDE_two_digit_R_equal_l568_56854


namespace NUMINAMATH_CALUDE_player_B_hit_rate_player_A_hit_at_least_once_in_two_l568_56863

-- Define the hit rates and probabilities
def player_A_hit_rate : ℚ := 1/2
def player_B_miss_twice_prob : ℚ := 1/16

-- Theorem for player B's hit rate
theorem player_B_hit_rate : 
  ∃ p : ℚ, (1 - p)^2 = player_B_miss_twice_prob ∧ p = 3/4 :=
sorry

-- Theorem for player A's probability of hitting at least once in two shots
theorem player_A_hit_at_least_once_in_two :
  1 - (1 - player_A_hit_rate)^2 = 3/4 :=
sorry

end NUMINAMATH_CALUDE_player_B_hit_rate_player_A_hit_at_least_once_in_two_l568_56863


namespace NUMINAMATH_CALUDE_local_minimum_at_two_l568_56839

noncomputable def f (x : ℝ) : ℝ := 2 / x + Real.log x

theorem local_minimum_at_two :
  ∃ δ > 0, ∀ x : ℝ, x ≠ 2 → |x - 2| < δ → f x ≥ f 2 :=
sorry

end NUMINAMATH_CALUDE_local_minimum_at_two_l568_56839


namespace NUMINAMATH_CALUDE_solution_set_abs_x_times_x_minus_one_l568_56871

theorem solution_set_abs_x_times_x_minus_one (x : ℝ) :
  (|x| * (x - 1) ≥ 0) ↔ (x ≥ 1 ∨ x = 0) :=
by sorry

end NUMINAMATH_CALUDE_solution_set_abs_x_times_x_minus_one_l568_56871


namespace NUMINAMATH_CALUDE_decision_has_two_exits_l568_56864

-- Define the flowchart symbol types
inductive FlowchartSymbol
  | Terminal
  | InputOutput
  | Process
  | Decision

-- Define a function that returns the number of exit paths for each symbol
def exitPaths (symbol : FlowchartSymbol) : Nat :=
  match symbol with
  | FlowchartSymbol.Terminal => 1
  | FlowchartSymbol.InputOutput => 1
  | FlowchartSymbol.Process => 1
  | FlowchartSymbol.Decision => 2

-- Theorem statement
theorem decision_has_two_exits :
  ∀ (symbol : FlowchartSymbol),
    exitPaths symbol = 2 ↔ symbol = FlowchartSymbol.Decision :=
by sorry

end NUMINAMATH_CALUDE_decision_has_two_exits_l568_56864


namespace NUMINAMATH_CALUDE_trajectory_equation_l568_56891

theorem trajectory_equation (x y : ℝ) :
  ((x + 3)^2 + y^2) + ((x - 3)^2 + y^2) = 38 → x^2 + y^2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_trajectory_equation_l568_56891


namespace NUMINAMATH_CALUDE_range_of_m_l568_56860

def A : Set ℝ := {x | 3 < x ∧ x < 10}
def B : Set ℝ := {x | x^2 - 9*x + 14 < 0}
def C (m : ℝ) : Set ℝ := {x | 5 - m < x ∧ x < 2*m}

theorem range_of_m :
  ∀ m : ℝ, (∀ x : ℝ, x ∈ C m → x ∈ A ∩ B) ∧
           (∃ y : ℝ, y ∈ A ∩ B ∧ y ∉ C m) ↔
  m ≤ 2 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l568_56860


namespace NUMINAMATH_CALUDE_E_parity_2023_2024_2025_l568_56875

def E : ℕ → ℤ
  | 0 => 1
  | 1 => 1
  | 2 => 0
  | n + 3 => E (n + 2) + E (n + 1) + E n

theorem E_parity_2023_2024_2025 :
  (E 2023 % 2, E 2024 % 2, E 2025 % 2) = (1, 1, 1) := by
  sorry

end NUMINAMATH_CALUDE_E_parity_2023_2024_2025_l568_56875


namespace NUMINAMATH_CALUDE_marco_cards_l568_56868

theorem marco_cards (C : ℕ) : 
  (C / 4 : ℚ) * (1 / 5 : ℚ) = 25 → C = 500 := by
  sorry

end NUMINAMATH_CALUDE_marco_cards_l568_56868


namespace NUMINAMATH_CALUDE_total_fuel_consumption_l568_56878

/-- Calculates the total fuel consumption over six weeks given specific conditions for each week. -/
theorem total_fuel_consumption
  (week1_consumption : ℝ)
  (week2_increase : ℝ)
  (week3_fraction : ℝ)
  (week4_increase : ℝ)
  (week5_budget : ℝ)
  (week5_price : ℝ)
  (week6_increase : ℝ)
  (h1 : week1_consumption = 25)
  (h2 : week2_increase = 0.1)
  (h3 : week3_fraction = 0.5)
  (h4 : week4_increase = 0.3)
  (h5 : week5_budget = 50)
  (h6 : week5_price = 2.5)
  (h7 : week6_increase = 0.2) :
  week1_consumption +
  (week1_consumption * (1 + week2_increase)) +
  (week1_consumption * week3_fraction) +
  (week1_consumption * week3_fraction * (1 + week4_increase)) +
  (week5_budget / week5_price) +
  (week5_budget / week5_price * (1 + week6_increase)) = 125.25 :=
by sorry

end NUMINAMATH_CALUDE_total_fuel_consumption_l568_56878


namespace NUMINAMATH_CALUDE_watermelon_melon_weight_comparison_l568_56897

theorem watermelon_melon_weight_comparison (W M : ℝ) 
  (h1 : W > 0) (h2 : M > 0)
  (h3 : (2*W > 3*M) ∨ (3*W > 4*M))
  (h4 : ¬((2*W > 3*M) ∧ (3*W > 4*M))) :
  ¬(12*W > 18*M) := by
sorry

end NUMINAMATH_CALUDE_watermelon_melon_weight_comparison_l568_56897


namespace NUMINAMATH_CALUDE_fruit_purchase_cost_l568_56867

/-- The cost of buying fruits given their prices per dozen and quantities --/
def total_cost (apple_price_per_dozen : ℕ) (pear_price_per_dozen : ℕ) (apple_quantity : ℕ) (pear_quantity : ℕ) : ℕ :=
  apple_price_per_dozen * apple_quantity + pear_price_per_dozen * pear_quantity

/-- Theorem: Given the prices and quantities of apples and pears, the total cost is 1260 dollars --/
theorem fruit_purchase_cost :
  total_cost 40 50 14 14 = 1260 := by
  sorry

#eval total_cost 40 50 14 14

end NUMINAMATH_CALUDE_fruit_purchase_cost_l568_56867


namespace NUMINAMATH_CALUDE_parallel_lines_a_equals_two_l568_56840

/-- Two lines in the form y = mx + b are parallel if and only if they have the same slope m -/
axiom parallel_lines_equal_slopes {m1 m2 b1 b2 : ℝ} : 
  (∀ x y : ℝ, y = m1 * x + b1 ↔ y = m2 * x + b2) → m1 = m2

/-- Given that the line ax + 2y + 1 = 0 is parallel to x + y - 2 = 0, prove that a = 2 -/
theorem parallel_lines_a_equals_two (a : ℝ) :
  (∀ x y : ℝ, a * x + 2 * y + 1 = 0 ↔ x + y - 2 = 0) → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_a_equals_two_l568_56840


namespace NUMINAMATH_CALUDE_root_relationship_l568_56836

theorem root_relationship (a b k x₁ x₂ x₃ : ℝ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hk : k ≠ 0)
  (hx₁ : a * x₁^2 = k * x₁ + b)
  (hx₂ : a * x₂^2 = k * x₂ + b)
  (hx₃ : k * x₃ + b = 0) :
  1 / x₁ + 1 / x₂ = 1 / x₃ := by
sorry

end NUMINAMATH_CALUDE_root_relationship_l568_56836


namespace NUMINAMATH_CALUDE_psychology_class_pairs_l568_56832

def number_of_students : ℕ := 12

-- Function to calculate the number of unique pairs
def unique_pairs (n : ℕ) : ℕ := n * (n - 1) / 2

-- Theorem stating that for 12 students, the number of unique pairs is 66
theorem psychology_class_pairs : 
  unique_pairs number_of_students = 66 := by sorry

end NUMINAMATH_CALUDE_psychology_class_pairs_l568_56832


namespace NUMINAMATH_CALUDE_clock_selling_theorem_l568_56810

/-- Represents the clock selling scenario with given conditions -/
def ClockSelling (original_cost : ℚ) : Prop :=
  let initial_sale := original_cost * 1.2
  let buyback_price := initial_sale * 0.5
  let maintenance_cost := buyback_price * 0.1
  let total_spent := buyback_price + maintenance_cost
  let final_sale := total_spent * 1.8
  (original_cost - buyback_price = 100) ∧ (final_sale = 297)

/-- Theorem stating the existence of an original cost satisfying the ClockSelling conditions -/
theorem clock_selling_theorem : ∃ (original_cost : ℚ), ClockSelling original_cost :=
  sorry

end NUMINAMATH_CALUDE_clock_selling_theorem_l568_56810


namespace NUMINAMATH_CALUDE_smallest_clock_equivalent_square_l568_56800

theorem smallest_clock_equivalent_square : ∃ (n : ℕ), 
  n > 4 ∧ 
  24 ∣ (n^2 - n) ∧ 
  ∀ (m : ℕ), m > 4 ∧ m < n → ¬(24 ∣ (m^2 - m)) :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_smallest_clock_equivalent_square_l568_56800


namespace NUMINAMATH_CALUDE_burger_cost_l568_56886

/-- Proves that the cost of each burger is $3.50 given Selena's expenses --/
theorem burger_cost (tip : ℝ) (steak_price : ℝ) (ice_cream_price : ℝ) (remaining : ℝ) :
  tip = 99 →
  steak_price = 24 →
  ice_cream_price = 2 →
  remaining = 38 →
  ∃ (burger_price : ℝ),
    burger_price = 3.5 ∧
    tip = 2 * steak_price + 2 * burger_price + 3 * ice_cream_price + remaining :=
by
  sorry

end NUMINAMATH_CALUDE_burger_cost_l568_56886


namespace NUMINAMATH_CALUDE_complex_pure_imaginary_l568_56831

theorem complex_pure_imaginary (a : ℝ) : 
  let z : ℂ := a + 4*I
  (∃ (b : ℝ), (2 - I) * z = b*I) → a = -2 := by
  sorry

end NUMINAMATH_CALUDE_complex_pure_imaginary_l568_56831


namespace NUMINAMATH_CALUDE_problem_solution_l568_56838

theorem problem_solution (m : ℕ) (q : ℚ) :
  m = 31 →
  ((1^m) / (5^m)) * ((1^16) / (4^16)) = 1 / (q * (10^31)) →
  q = 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l568_56838


namespace NUMINAMATH_CALUDE_equation_solution_range_l568_56853

theorem equation_solution_range (x m : ℝ) : 
  (x / (x - 3) - 2 = m / (x - 3) ∧ x > 0) → (m < 6 ∧ m ≠ 3) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_range_l568_56853


namespace NUMINAMATH_CALUDE_total_wheels_l568_56828

/-- The number of wheels on a bicycle -/
def bicycle_wheels : ℕ := 2

/-- The number of wheels on a tricycle -/
def tricycle_wheels : ℕ := 3

/-- The number of adults riding bicycles -/
def adults_on_bicycles : ℕ := 6

/-- The number of children riding tricycles -/
def children_on_tricycles : ℕ := 15

/-- The total number of wheels Dimitri saw at the park -/
theorem total_wheels : 
  bicycle_wheels * adults_on_bicycles + tricycle_wheels * children_on_tricycles = 57 := by
  sorry

end NUMINAMATH_CALUDE_total_wheels_l568_56828


namespace NUMINAMATH_CALUDE_room_width_calculation_l568_56813

theorem room_width_calculation (length : ℝ) (total_cost : ℝ) (rate_per_sqm : ℝ) :
  length = 5.5 →
  total_cost = 28875 →
  rate_per_sqm = 1400 →
  (total_cost / rate_per_sqm) / length = 3.75 :=
by sorry

end NUMINAMATH_CALUDE_room_width_calculation_l568_56813


namespace NUMINAMATH_CALUDE_multiply_polynomial_difference_of_cubes_l568_56830

theorem multiply_polynomial_difference_of_cubes (x : ℝ) :
  (x^4 + 12*x^2 + 144) * (x^2 - 12) = x^6 - 1728 := by
  sorry

end NUMINAMATH_CALUDE_multiply_polynomial_difference_of_cubes_l568_56830


namespace NUMINAMATH_CALUDE_max_leap_years_in_200_years_l568_56843

/-- A calendar system where leap years occur every 5 years -/
structure Calendar :=
  (leap_year_frequency : ℕ)
  (total_years : ℕ)
  (h_leap_frequency : leap_year_frequency = 5)

/-- The number of leap years in the calendar system -/
def leap_years (c : Calendar) : ℕ := c.total_years / c.leap_year_frequency

/-- Theorem: The maximum number of leap years in a 200-year period is 40 -/
theorem max_leap_years_in_200_years (c : Calendar) 
  (h_total_years : c.total_years = 200) : 
  leap_years c = 40 := by
  sorry

end NUMINAMATH_CALUDE_max_leap_years_in_200_years_l568_56843


namespace NUMINAMATH_CALUDE_determinant_equality_l568_56873

theorem determinant_equality (a b c d : ℝ) :
  Matrix.det !![a, b; c, d] = -7 →
  Matrix.det !![2*a + c, b - 2*d; c, 2*d] = -28 + 3*b*c + 2*c*d + 2*d*c := by
  sorry

end NUMINAMATH_CALUDE_determinant_equality_l568_56873


namespace NUMINAMATH_CALUDE_sin_cos_equation_integer_solution_l568_56887

theorem sin_cos_equation_integer_solution (x : ℤ) :
  (∃ t : ℤ, x = 4 * t + 1 ∨ x = 4 * t - 1) ↔ 
  Real.sin (π * (2 * ↑x - 1)) = Real.cos (π * ↑x / 2) :=
sorry

end NUMINAMATH_CALUDE_sin_cos_equation_integer_solution_l568_56887


namespace NUMINAMATH_CALUDE_sams_remaining_marbles_l568_56885

/-- Given Sam's initial yellow marble count and the number of yellow marbles Joan took,
    prove that Sam's remaining yellow marble count is the difference between the two. -/
theorem sams_remaining_marbles (initial_count : ℕ) (marbles_taken : ℕ) 
    (h : marbles_taken ≤ initial_count) :
  initial_count - marbles_taken = initial_count - marbles_taken :=
by
  sorry

#check sams_remaining_marbles 86 25

end NUMINAMATH_CALUDE_sams_remaining_marbles_l568_56885


namespace NUMINAMATH_CALUDE_gift_expenses_calculation_l568_56827

/-- Calculates the total amount spent on gift wrapping, taxes, and other expenses given the following conditions:
  * Jeremy bought presents for 5 people
  * Total spent: $930
  * Gift costs: $400 (mother), $280 (father), $100 (sister), $60 (brother), $50 (best friend)
  * Gift wrapping fee: 7% of each gift's price
  * Tax rate: 9%
  * Other miscellaneous expenses: $40
-/
theorem gift_expenses_calculation (total_spent : ℝ) (gift_mother gift_father gift_sister gift_brother gift_friend : ℝ)
  (wrapping_rate tax_rate : ℝ) (misc_expenses : ℝ) :
  total_spent = 930 ∧
  gift_mother = 400 ∧ gift_father = 280 ∧ gift_sister = 100 ∧ gift_brother = 60 ∧ gift_friend = 50 ∧
  wrapping_rate = 0.07 ∧ tax_rate = 0.09 ∧ misc_expenses = 40 →
  (gift_mother + gift_father + gift_sister + gift_brother + gift_friend) * wrapping_rate +
  (gift_mother + gift_father + gift_sister + gift_brother + gift_friend) * tax_rate +
  misc_expenses = 182.40 := by
  sorry

end NUMINAMATH_CALUDE_gift_expenses_calculation_l568_56827
