import Mathlib

namespace NUMINAMATH_CALUDE_twenty_fifth_decimal_of_n_over_11_l1391_139147

theorem twenty_fifth_decimal_of_n_over_11 (n : ℕ) (h : n / 11 = 9) :
  (n : ℚ) / 11 - (n / 11 : ℕ) = 0 :=
sorry

end NUMINAMATH_CALUDE_twenty_fifth_decimal_of_n_over_11_l1391_139147


namespace NUMINAMATH_CALUDE_find_divisor_l1391_139169

theorem find_divisor (dividend quotient remainder : ℕ) 
  (h1 : dividend = 140)
  (h2 : quotient = 9)
  (h3 : remainder = 5)
  : ∃ divisor : ℕ, 
    dividend = divisor * quotient + remainder ∧ 
    divisor = 15 := by
  sorry

end NUMINAMATH_CALUDE_find_divisor_l1391_139169


namespace NUMINAMATH_CALUDE_quadratic_equation_solutions_l1391_139138

theorem quadratic_equation_solutions (b c x₁ x₂ : ℝ) : 
  (x₁^2 + b*x₁ + c = 0) →
  (x₂^2 + b*x₂ + c = 0) →
  (x₁ ≠ x₂) →
  (|x₁ - x₂| = 1) →
  (|b - c| = 1) →
  ((b = -1 ∧ c = 0) ∨ (b = 5 ∧ c = 6) ∨ (b = 1 ∧ c = 0) ∨ (b = 3 ∧ c = 2)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solutions_l1391_139138


namespace NUMINAMATH_CALUDE_kelly_games_theorem_l1391_139102

/-- The number of games Kelly gives away -/
def games_given_away : ℕ := 15

/-- The number of games Kelly has left after giving some away -/
def games_left : ℕ := 35

/-- The initial number of games Kelly has -/
def initial_games : ℕ := games_given_away + games_left

theorem kelly_games_theorem : initial_games = 50 := by
  sorry

end NUMINAMATH_CALUDE_kelly_games_theorem_l1391_139102


namespace NUMINAMATH_CALUDE_lottery_savings_calculation_l1391_139145

theorem lottery_savings_calculation (lottery_winnings : ℚ) 
  (tax_rate : ℚ) (student_loan_rate : ℚ) (investment_rate : ℚ) (fun_money : ℚ) :
  lottery_winnings = 12006 →
  tax_rate = 1/2 →
  student_loan_rate = 1/3 →
  investment_rate = 1/5 →
  fun_money = 2802 →
  ∃ (savings : ℚ),
    savings = 1000 ∧
    lottery_winnings * (1 - tax_rate) * (1 - student_loan_rate) - fun_money = savings * (1 + investment_rate) :=
by sorry

end NUMINAMATH_CALUDE_lottery_savings_calculation_l1391_139145


namespace NUMINAMATH_CALUDE_first_part_multiplier_l1391_139153

theorem first_part_multiplier (x : ℝ) : x + 7 * x = 55 → x = 5 → 1 = 1 := by
  sorry

end NUMINAMATH_CALUDE_first_part_multiplier_l1391_139153


namespace NUMINAMATH_CALUDE_chairs_count_l1391_139159

/-- The number of chairs in the auditorium at Yunju's school -/
def total_chairs : ℕ := by sorry

/-- The auditorium is square-shaped -/
axiom is_square : total_chairs = (Nat.sqrt total_chairs) ^ 2

/-- Yunju's seat is 2nd from the front -/
axiom front_distance : 2 ≤ Nat.sqrt total_chairs

/-- Yunju's seat is 5th from the back -/
axiom back_distance : 5 ≤ Nat.sqrt total_chairs

/-- Yunju's seat is 3rd from the right -/
axiom right_distance : 3 ≤ Nat.sqrt total_chairs

/-- Yunju's seat is 4th from the left -/
axiom left_distance : 4 ≤ Nat.sqrt total_chairs

/-- The theorem to be proved -/
theorem chairs_count : total_chairs = 36 := by sorry

end NUMINAMATH_CALUDE_chairs_count_l1391_139159


namespace NUMINAMATH_CALUDE_lcm_gcd_product_36_60_l1391_139150

theorem lcm_gcd_product_36_60 : Nat.lcm 36 60 * Nat.gcd 36 60 = 36 * 60 := by
  sorry

end NUMINAMATH_CALUDE_lcm_gcd_product_36_60_l1391_139150


namespace NUMINAMATH_CALUDE_hot_dog_sales_first_innings_l1391_139195

/-- Represents the number of hot dogs in various states --/
structure HotDogSales where
  total : ℕ
  sold_later : ℕ
  left : ℕ

/-- Calculates the number of hot dogs sold in the first three innings --/
def sold_first (s : HotDogSales) : ℕ :=
  s.total - s.sold_later - s.left

/-- Theorem stating that for the given values, the number of hot dogs
    sold in the first three innings is 19 --/
theorem hot_dog_sales_first_innings
  (s : HotDogSales)
  (h1 : s.total = 91)
  (h2 : s.sold_later = 27)
  (h3 : s.left = 45) :
  sold_first s = 19 := by
  sorry

end NUMINAMATH_CALUDE_hot_dog_sales_first_innings_l1391_139195


namespace NUMINAMATH_CALUDE_range_of_two_alpha_l1391_139124

theorem range_of_two_alpha (α β : ℝ) 
  (h1 : π < α + β ∧ α + β < 4 / 3 * π)
  (h2 : -π < α - β ∧ α - β < -π / 3) :
  0 < 2 * α ∧ 2 * α < π :=
by sorry

end NUMINAMATH_CALUDE_range_of_two_alpha_l1391_139124


namespace NUMINAMATH_CALUDE_distribution_six_twelve_l1391_139162

/-- The number of ways to distribute distinct items among recipients --/
def distribution_ways (num_items : ℕ) (num_recipients : ℕ) : ℕ :=
  num_recipients ^ num_items

/-- Theorem: The number of ways to distribute 6 distinct items among 12 recipients is 2,985,984 --/
theorem distribution_six_twelve : distribution_ways 6 12 = 2985984 := by
  sorry

end NUMINAMATH_CALUDE_distribution_six_twelve_l1391_139162


namespace NUMINAMATH_CALUDE_solution_characterization_l1391_139110

def is_solution (a b : ℕ+) : Prop :=
  (a.val ^ 2 * b.val ^ 2 + 208 : ℕ) = 4 * (Nat.lcm a.val b.val + Nat.gcd a.val b.val) ^ 2

theorem solution_characterization :
  ∀ a b : ℕ+, is_solution a b ↔ 
    ((a = 4 ∧ b = 6) ∨ (a = 6 ∧ b = 4) ∨ (a = 2 ∧ b = 12) ∨ (a = 12 ∧ b = 2)) :=
by sorry

end NUMINAMATH_CALUDE_solution_characterization_l1391_139110


namespace NUMINAMATH_CALUDE_always_unaffected_square_l1391_139134

/-- Represents a square cut on the cake -/
structure Cut where
  x : ℚ
  y : ℚ
  size : ℚ
  h_x : 0 ≤ x ∧ x + size ≤ 3
  h_y : 0 ≤ y ∧ y + size ≤ 3

/-- Represents a small 1/3 x 1/3 square on the cake -/
structure SmallSquare where
  x : ℚ
  y : ℚ
  h_x : x = 0 ∨ x = 1 ∨ x = 2
  h_y : y = 0 ∨ y = 1 ∨ y = 2

/-- Check if a small square is affected by a cut -/
def isAffected (s : SmallSquare) (c : Cut) : Prop :=
  (c.x < s.x + 1/3 ∧ s.x < c.x + c.size) ∧
  (c.y < s.y + 1/3 ∧ s.y < c.y + c.size)

/-- Main theorem: There always exists an unaffected 1/3 x 1/3 square -/
theorem always_unaffected_square (cuts : Finset Cut) (h : cuts.card = 4) (h_size : ∀ c ∈ cuts, c.size = 1) :
  ∃ s : SmallSquare, ∀ c ∈ cuts, ¬isAffected s c :=
sorry

end NUMINAMATH_CALUDE_always_unaffected_square_l1391_139134


namespace NUMINAMATH_CALUDE_loss_percentage_proof_l1391_139142

def cost_price : ℝ := 1250
def price_increase : ℝ := 500
def gain_percentage : ℝ := 0.15

theorem loss_percentage_proof (selling_price : ℝ) 
  (h1 : selling_price + price_increase = cost_price * (1 + gain_percentage)) :
  (cost_price - selling_price) / cost_price = 0.25 := by
  sorry

end NUMINAMATH_CALUDE_loss_percentage_proof_l1391_139142


namespace NUMINAMATH_CALUDE_combination_equality_l1391_139157

theorem combination_equality (x : ℕ) : (Nat.choose 9 x = Nat.choose 9 (2*x - 3)) → (x = 3 ∨ x = 4) :=
by sorry

end NUMINAMATH_CALUDE_combination_equality_l1391_139157


namespace NUMINAMATH_CALUDE_ratio_equality_l1391_139144

theorem ratio_equality (x y z : ℝ) (h : x / 2 = y / 3 ∧ y / 3 = z / 4) :
  (x - 2*y + 3*z) / (x + y + z) = 8 / 9 := by
  sorry

end NUMINAMATH_CALUDE_ratio_equality_l1391_139144


namespace NUMINAMATH_CALUDE_cos_difference_inverse_cos_tan_l1391_139190

theorem cos_difference_inverse_cos_tan (x y : ℝ) 
  (hx : x^2 ≤ 1) (hy : y > 0) : 
  Real.cos (Real.arccos (4/5) - Real.arctan (1/2)) = 11 * Real.sqrt 5 / 25 := by
  sorry

end NUMINAMATH_CALUDE_cos_difference_inverse_cos_tan_l1391_139190


namespace NUMINAMATH_CALUDE_set_A_properties_l1391_139103

def A : Set ℝ := {x | x^2 - 1 = 0}

theorem set_A_properties : 
  (1 ∈ A) ∧ (∅ ⊆ A) ∧ ({1, -1} ⊆ A) := by sorry

end NUMINAMATH_CALUDE_set_A_properties_l1391_139103


namespace NUMINAMATH_CALUDE_rat_speed_l1391_139112

/-- Proves that under given conditions, the rat's speed is 36 kmph -/
theorem rat_speed (head_start : ℝ) (catch_up_time : ℝ) (cat_speed : ℝ)
  (h1 : head_start = 6)
  (h2 : catch_up_time = 4)
  (h3 : cat_speed = 90) :
  let rat_speed := (cat_speed * catch_up_time) / (head_start + catch_up_time)
  rat_speed = 36 := by
sorry

end NUMINAMATH_CALUDE_rat_speed_l1391_139112


namespace NUMINAMATH_CALUDE_perfect_33rd_power_l1391_139105

theorem perfect_33rd_power (x y : ℕ+) (h : ∃ k : ℕ+, (x * y^10 : ℕ) = k^33) :
  ∃ m : ℕ+, (x^10 * y : ℕ) = m^33 := by
  sorry

end NUMINAMATH_CALUDE_perfect_33rd_power_l1391_139105


namespace NUMINAMATH_CALUDE_inscribed_rectangle_theorem_l1391_139128

-- Define the triangle
def triangle_sides : (ℝ × ℝ × ℝ) := (10, 17, 21)

-- Define the perimeter of the inscribed rectangle
def rectangle_perimeter : ℝ := 24

-- Define the function to calculate the sides of the inscribed rectangle
def inscribed_rectangle_sides (triangle : ℝ × ℝ × ℝ) (perimeter : ℝ) : (ℝ × ℝ) :=
  sorry

-- Theorem statement
theorem inscribed_rectangle_theorem :
  inscribed_rectangle_sides triangle_sides rectangle_perimeter = (5 + 7/13, 6 + 6/13) :=
sorry

end NUMINAMATH_CALUDE_inscribed_rectangle_theorem_l1391_139128


namespace NUMINAMATH_CALUDE_inequality_proof_l1391_139183

theorem inequality_proof (a₁ a₂ a₃ b₁ b₂ b₃ : ℝ) 
  (ha : a₁ ≥ a₂ ∧ a₂ ≥ a₃) (hb : b₁ ≥ b₂ ∧ b₂ ≥ b₃) : 
  3 * (a₁ * b₁ + a₂ * b₂ + a₃ * b₃) ≥ (a₁ + a₂ + a₃) * (b₁ + b₂ + b₃) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1391_139183


namespace NUMINAMATH_CALUDE_new_girl_weight_l1391_139192

/-- Given a group of 8 girls, if replacing one girl weighing 70 kg with a new girl
    increases the average weight by 3 kg, then the weight of the new girl is 94 kg. -/
theorem new_girl_weight (W : ℝ) (new_weight : ℝ) : 
  (W / 8 + 3) * 8 = W - 70 + new_weight →
  new_weight = 94 := by
  sorry

end NUMINAMATH_CALUDE_new_girl_weight_l1391_139192


namespace NUMINAMATH_CALUDE_seating_arrangement_theorem_l1391_139191

/-- The number of ways to arrange n people in a row -/
def totalArrangements (n : ℕ) : ℕ := n.factorial

/-- The number of ways to arrange n people in a row where two specific people sit together -/
def arrangementsWithTwoTogether (n : ℕ) : ℕ := (n - 1).factorial * 2

/-- The number of people -/
def numberOfPeople : ℕ := 4

/-- The number of valid seating arrangements -/
def validArrangements : ℕ := totalArrangements numberOfPeople - arrangementsWithTwoTogether numberOfPeople

theorem seating_arrangement_theorem :
  validArrangements = 12 := by sorry

end NUMINAMATH_CALUDE_seating_arrangement_theorem_l1391_139191


namespace NUMINAMATH_CALUDE_original_price_of_dress_l1391_139166

/-- Given a 30% discount and a final price of $35, prove that the original price of the dress was $50. -/
theorem original_price_of_dress (discount_percentage : ℝ) (final_price : ℝ) : 
  discount_percentage = 30 →
  final_price = 35 →
  (1 - discount_percentage / 100) * 50 = final_price :=
by sorry

end NUMINAMATH_CALUDE_original_price_of_dress_l1391_139166


namespace NUMINAMATH_CALUDE_pigeonhole_principle_sports_choices_l1391_139168

/-- Given a set of 50 people, each making choices from three categories with 4, 3, and 2 options respectively,
    there must be at least 3 people who have made exactly the same choices for all three categories. -/
theorem pigeonhole_principle_sports_choices :
  ∀ (choices : Fin 50 → Fin 4 × Fin 3 × Fin 2),
  ∃ (c : Fin 4 × Fin 3 × Fin 2) (s₁ s₂ s₃ : Fin 50),
  s₁ ≠ s₂ ∧ s₁ ≠ s₃ ∧ s₂ ≠ s₃ ∧
  choices s₁ = c ∧ choices s₂ = c ∧ choices s₃ = c :=
by sorry

end NUMINAMATH_CALUDE_pigeonhole_principle_sports_choices_l1391_139168


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_range_of_a_l1391_139165

-- Define the sets A, B, and M
def A : Set ℝ := {x | x^2 - x - 2 < 0}
def B : Set ℝ := {x | x * (3 - x) > 0}
def M (a : ℝ) : Set ℝ := {x | 2 * x - a < 0}

-- Theorem for part 1
theorem intersection_A_complement_B : A ∩ (Set.univ \ B) = {x : ℝ | -1 < x ∧ x ≤ 0} := by sorry

-- Theorem for part 2
theorem range_of_a (a : ℝ) : (A ∪ B) ⊆ M a → a ≥ 6 := by sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_range_of_a_l1391_139165


namespace NUMINAMATH_CALUDE_solve_letter_problem_l1391_139126

def letter_problem (brother_letters : ℕ) (greta_extra : ℕ) : Prop :=
  let greta_letters := brother_letters + greta_extra
  let total_greta_brother := brother_letters + greta_letters
  let mother_letters := 2 * total_greta_brother
  let total_letters := brother_letters + greta_letters + mother_letters
  (brother_letters = 40) ∧ (greta_extra = 10) → (total_letters = 270)

theorem solve_letter_problem : letter_problem 40 10 := by
  sorry

end NUMINAMATH_CALUDE_solve_letter_problem_l1391_139126


namespace NUMINAMATH_CALUDE_g_three_sixteenths_l1391_139170

-- Define the properties of g
def g_properties (g : ℝ → ℝ) : Prop :=
  (∀ x, 0 ≤ x ∧ x ≤ 1 → 0 ≤ g x ∧ g x ≤ 1) ∧
  (g 0 = 0) ∧
  (∀ x y, 0 ≤ x ∧ x < y ∧ y ≤ 1 → g x ≤ g y) ∧
  (∀ x, 0 ≤ x ∧ x ≤ 1 → g (1 - x) = 1 - g x) ∧
  (∀ x, 0 ≤ x ∧ x ≤ 1 → g (x / 4) = g x / 2)

-- Theorem statement
theorem g_three_sixteenths (g : ℝ → ℝ) (h : g_properties g) : g (3/16) = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_g_three_sixteenths_l1391_139170


namespace NUMINAMATH_CALUDE_quadratic_function_k_l1391_139189

/-- A quadratic function f(x) = ax^2 + bx + c with integer coefficients -/
def f (a b c : ℤ) (x : ℤ) : ℤ := a * x^2 + b * x + c

/-- The theorem statement -/
theorem quadratic_function_k (a b c : ℤ) : 
  (f a b c 1 = 0) →
  (60 < f a b c 6 ∧ f a b c 6 < 70) →
  (120 < f a b c 9 ∧ f a b c 9 < 130) →
  (∃ k : ℤ, 10000 * k < f a b c 200 ∧ f a b c 200 < 10000 * (k + 1)) →
  (∃ k : ℤ, 10000 * k < f a b c 200 ∧ f a b c 200 < 10000 * (k + 1) ∧ k = 4) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_k_l1391_139189


namespace NUMINAMATH_CALUDE_power_equality_l1391_139146

theorem power_equality (k m : ℕ) 
  (h1 : 3 ^ (k - 1) = 9) 
  (h2 : 4 ^ (m + 2) = 64) : 
  2 ^ (3 * k + 2 * m) = 2 ^ 11 := by
  sorry

end NUMINAMATH_CALUDE_power_equality_l1391_139146


namespace NUMINAMATH_CALUDE_negation_of_implication_l1391_139187

theorem negation_of_implication :
  (¬(∀ x : ℝ, x > 1 → x > 0)) ↔ (∀ x : ℝ, x ≤ 1 → x ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_implication_l1391_139187


namespace NUMINAMATH_CALUDE_nested_expression_simplification_l1391_139132

theorem nested_expression_simplification (x : ℝ) : 1 - (1 + (1 - (1 + (1 - (1 - x))))) = 1 - x := by
  sorry

end NUMINAMATH_CALUDE_nested_expression_simplification_l1391_139132


namespace NUMINAMATH_CALUDE_cylinder_radius_problem_l1391_139199

/-- Given a cylinder with original height 3 inches, if increasing the radius by 4 inches
    and the height by 6 inches results in the same new volume, then the original radius
    is 2 + 2√3 inches. -/
theorem cylinder_radius_problem (r : ℝ) : 
  (3 * π * (r + 4)^2 = 9 * π * r^2) → r = 2 + 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_cylinder_radius_problem_l1391_139199


namespace NUMINAMATH_CALUDE_geometry_theorem_l1391_139107

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (contains : Plane → Line → Prop)
variable (intersect : Plane → Plane → Line → Prop)
variable (planePerp : Plane → Plane → Prop)

-- State the theorem
theorem geometry_theorem 
  (α β : Plane) (m n : Line) 
  (h_distinct_planes : α ≠ β) 
  (h_distinct_lines : m ≠ n) :
  (∀ (m n : Line) (α : Plane), perpendicular m α → perpendicular n α → parallel m n) ∧
  (∀ (m n : Line) (α β : Plane), perpendicular m α → parallel m n → contains β n → planePerp α β) :=
sorry

end NUMINAMATH_CALUDE_geometry_theorem_l1391_139107


namespace NUMINAMATH_CALUDE_cube_difference_equality_l1391_139139

theorem cube_difference_equality : 
  - (666 : ℤ)^3 + (555 : ℤ)^3 = ((666 : ℤ)^2 - 666 * 555 + (555 : ℤ)^2) * (-124072470) := by
  sorry

end NUMINAMATH_CALUDE_cube_difference_equality_l1391_139139


namespace NUMINAMATH_CALUDE_mr_a_net_gain_l1391_139186

/-- The total net gain for Mr. A after a series of house transactions -/
theorem mr_a_net_gain (house1_value house2_value : ℝ)
  (house1_profit house1_loss house2_profit house2_loss : ℝ)
  (h1 : house1_value = 15000)
  (h2 : house2_value = 20000)
  (h3 : house1_profit = 0.15)
  (h4 : house1_loss = 0.15)
  (h5 : house2_profit = 0.20)
  (h6 : house2_loss = 0.20) :
  let sale1 := house1_value * (1 + house1_profit)
  let sale2 := house2_value * (1 + house2_profit)
  let buyback1 := sale1 * (1 - house1_loss)
  let buyback2 := sale2 * (1 - house2_loss)
  let net_gain := (sale1 - buyback1) + (sale2 - buyback2)
  net_gain = 7387.50 := by
  sorry

end NUMINAMATH_CALUDE_mr_a_net_gain_l1391_139186


namespace NUMINAMATH_CALUDE_officer_selection_ways_l1391_139154

def group_members : Nat := 5
def officer_positions : Nat := 4

theorem officer_selection_ways : 
  (group_members.choose officer_positions) * (officer_positions.factorial) = 120 := by
  sorry

end NUMINAMATH_CALUDE_officer_selection_ways_l1391_139154


namespace NUMINAMATH_CALUDE_garbage_collection_l1391_139182

theorem garbage_collection (D : ℝ) : 
  (∃ (Dewei Zane : ℝ), 
    Dewei = D - 2 ∧ 
    Zane = 4 * Dewei ∧ 
    Zane = 62) → 
  D = 17.5 := by
sorry

end NUMINAMATH_CALUDE_garbage_collection_l1391_139182


namespace NUMINAMATH_CALUDE_geometric_progression_solution_l1391_139111

/-- Given three terms of a geometric progression in the form (15 + x), (45 + x), and (135 + x),
    prove that x = 0 is the unique solution. -/
theorem geometric_progression_solution (x : ℝ) : 
  (∃ r : ℝ, r ≠ 0 ∧ (45 + x) = (15 + x) * r ∧ (135 + x) = (45 + x) * r) ↔ x = 0 := by
  sorry

end NUMINAMATH_CALUDE_geometric_progression_solution_l1391_139111


namespace NUMINAMATH_CALUDE_point_in_second_quadrant_m_range_l1391_139104

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

end NUMINAMATH_CALUDE_point_in_second_quadrant_m_range_l1391_139104


namespace NUMINAMATH_CALUDE_junior_score_l1391_139148

theorem junior_score (n : ℝ) (junior_score : ℝ) :
  n > 0 →
  0.1 * n * junior_score + 0.9 * n * 83 = n * 84 →
  junior_score = 93 := by
sorry

end NUMINAMATH_CALUDE_junior_score_l1391_139148


namespace NUMINAMATH_CALUDE_square_roots_theorem_l1391_139149

theorem square_roots_theorem (x : ℝ) (m : ℝ) : 
  x > 0 → (2*m - 1)^2 = x → (2 - m)^2 = x → x = 9 := by
  sorry

end NUMINAMATH_CALUDE_square_roots_theorem_l1391_139149


namespace NUMINAMATH_CALUDE_part_one_part_two_l1391_139184

-- Define the function y
def y (m x : ℝ) : ℝ := m * x^2 - m * x - 1

-- Part 1
theorem part_one :
  (∀ x : ℝ, y m x < 0) ↔ m ∈ Set.Ioc (-4) 0 :=
sorry

-- Part 2
theorem part_two :
  (∀ x ∈ Set.Icc 1 3, y m x < -m + 5) ↔ m ∈ Set.Iio (6/7) :=
sorry

end NUMINAMATH_CALUDE_part_one_part_two_l1391_139184


namespace NUMINAMATH_CALUDE_square_ratio_side_length_l1391_139137

theorem square_ratio_side_length (area_ratio : ℚ) :
  area_ratio = 250 / 98 →
  ∃ (a b c : ℕ), 
    (a = 25 ∧ b = 5 ∧ c = 7) ∧
    (Real.sqrt area_ratio * c = a * Real.sqrt b) :=
by sorry

end NUMINAMATH_CALUDE_square_ratio_side_length_l1391_139137


namespace NUMINAMATH_CALUDE_arctan_arcsin_sum_equals_pi_l1391_139198

theorem arctan_arcsin_sum_equals_pi (x : ℝ) (h : x > 1) :
  2 * Real.arctan x + Real.arcsin (2 * x / (1 + x^2)) = π := by
sorry

end NUMINAMATH_CALUDE_arctan_arcsin_sum_equals_pi_l1391_139198


namespace NUMINAMATH_CALUDE_solution_set_f_geq_2_min_value_f_l1391_139160

-- Define the function f
def f (x : ℝ) : ℝ := |2 * x + 1| - |x - 4|

-- Theorem for the solution set of f(x) ≥ 2
theorem solution_set_f_geq_2 :
  {x : ℝ | f x ≥ 2} = {x : ℝ | x ≤ -7} ∪ {x : ℝ | x ≥ 5/3} := by sorry

-- Theorem for the minimum value of f(x)
theorem min_value_f :
  ∃ (x : ℝ), f x = -9/2 ∧ ∀ (y : ℝ), f y ≥ -9/2 := by sorry

end NUMINAMATH_CALUDE_solution_set_f_geq_2_min_value_f_l1391_139160


namespace NUMINAMATH_CALUDE_inequality_proof_l1391_139158

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (2 * a^2 + 3 * b^2 ≥ 6/5) ∧ ((a + 1/a) * (b + 1/b) ≥ 25/4) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1391_139158


namespace NUMINAMATH_CALUDE_unit_vector_parallel_l1391_139173

/-- Given two vectors a and b in ℝ², prove that the unit vector parallel to 2a - 3b
    is either (√5/5, 2√5/5) or (-√5/5, -2√5/5) -/
theorem unit_vector_parallel (a b : ℝ × ℝ) (ha : a = (5, 4)) (hb : b = (3, 2)) :
  let v := (2 • a.1 - 3 • b.1, 2 • a.2 - 3 • b.2)
  (v.1 / Real.sqrt (v.1^2 + v.2^2), v.2 / Real.sqrt (v.1^2 + v.2^2)) = (Real.sqrt 5 / 5, 2 * Real.sqrt 5 / 5) ∨
  (v.1 / Real.sqrt (v.1^2 + v.2^2), v.2 / Real.sqrt (v.1^2 + v.2^2)) = (-Real.sqrt 5 / 5, -2 * Real.sqrt 5 / 5) :=
by sorry

end NUMINAMATH_CALUDE_unit_vector_parallel_l1391_139173


namespace NUMINAMATH_CALUDE_neg_neg_two_eq_two_neg_six_plus_six_eq_zero_neg_three_times_five_eq_neg_fifteen_two_x_minus_three_x_eq_neg_x_l1391_139101

-- Problem 1
theorem neg_neg_two_eq_two : -(-2) = 2 := by sorry

-- Problem 2
theorem neg_six_plus_six_eq_zero : -6 + 6 = 0 := by sorry

-- Problem 3
theorem neg_three_times_five_eq_neg_fifteen : (-3) * 5 = -15 := by sorry

-- Problem 4
theorem two_x_minus_three_x_eq_neg_x (x : ℤ) : 2*x - 3*x = -x := by sorry

end NUMINAMATH_CALUDE_neg_neg_two_eq_two_neg_six_plus_six_eq_zero_neg_three_times_five_eq_neg_fifteen_two_x_minus_three_x_eq_neg_x_l1391_139101


namespace NUMINAMATH_CALUDE_paving_stone_width_l1391_139117

/-- Given a rectangular courtyard and paving stones with specific dimensions,
    prove that the width of each paving stone is 2 meters. -/
theorem paving_stone_width
  (courtyard_length : ℝ)
  (courtyard_width : ℝ)
  (stone_length : ℝ)
  (total_stones : ℕ)
  (h1 : courtyard_length = 60)
  (h2 : courtyard_width = 16.5)
  (h3 : stone_length = 2.5)
  (h4 : total_stones = 198) :
  ∃ (stone_width : ℝ), stone_width = 2 ∧
    courtyard_length * courtyard_width = stone_length * stone_width * total_stones :=
by sorry

end NUMINAMATH_CALUDE_paving_stone_width_l1391_139117


namespace NUMINAMATH_CALUDE_total_amount_paid_l1391_139174

-- Define the purchased amounts, rates, and discounts
def grape_amount : ℝ := 3
def mango_amount : ℝ := 9
def orange_amount : ℝ := 5
def banana_amount : ℝ := 7

def grape_rate : ℝ := 70
def mango_rate : ℝ := 55
def orange_rate : ℝ := 40
def banana_rate : ℝ := 20

def grape_discount : ℝ := 0.05
def mango_discount : ℝ := 0.10
def orange_discount : ℝ := 0.08
def banana_discount : ℝ := 0

def sales_tax : ℝ := 0.05

-- Define the theorem
theorem total_amount_paid : 
  let grape_cost := grape_amount * grape_rate
  let mango_cost := mango_amount * mango_rate
  let orange_cost := orange_amount * orange_rate
  let banana_cost := banana_amount * banana_rate

  let grape_discounted := grape_cost * (1 - grape_discount)
  let mango_discounted := mango_cost * (1 - mango_discount)
  let orange_discounted := orange_cost * (1 - orange_discount)
  let banana_discounted := banana_cost * (1 - banana_discount)

  let total_discounted := grape_discounted + mango_discounted + orange_discounted + banana_discounted
  let total_with_tax := total_discounted * (1 + sales_tax)

  total_with_tax = 1017.45 := by
    sorry

end NUMINAMATH_CALUDE_total_amount_paid_l1391_139174


namespace NUMINAMATH_CALUDE_inequality_proof_l1391_139164

theorem inequality_proof (x₁ x₂ : ℝ) (h1 : 0 < x₁) (h2 : x₁ < x₂) (h3 : x₂ < 1) :
  x₂ * Real.exp x₁ > x₁ * Real.exp x₂ := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1391_139164


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_of_quadratic_roots_sum_of_reciprocals_specific_quadratic_l1391_139122

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

end NUMINAMATH_CALUDE_sum_of_reciprocals_of_quadratic_roots_sum_of_reciprocals_specific_quadratic_l1391_139122


namespace NUMINAMATH_CALUDE_probability_two_defective_tubes_l1391_139185

/-- The probability of selecting two defective tubes without replacement from a consignment of picture tubes -/
theorem probability_two_defective_tubes (total : ℕ) (defective : ℕ) 
  (h1 : total = 20) (h2 : defective = 5) (h3 : defective < total) :
  (defective : ℚ) / total * ((defective - 1) : ℚ) / (total - 1) = 1 / 19 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_defective_tubes_l1391_139185


namespace NUMINAMATH_CALUDE_johns_pens_l1391_139171

/-- The number of pens John has -/
def total_pens (blue black red : ℕ) : ℕ := blue + black + red

theorem johns_pens :
  ∀ (blue black red : ℕ),
  blue = 18 →
  blue = 2 * black →
  black = red + 5 →
  total_pens blue black red = 31 := by
sorry

end NUMINAMATH_CALUDE_johns_pens_l1391_139171


namespace NUMINAMATH_CALUDE_line_y_coordinate_l1391_139152

/-- 
Given a line that:
- passes through a point (3, y)
- has a slope of 2
- has an x-intercept of 1

Prove that the y-coordinate of the point (3, y) is 4.
-/
theorem line_y_coordinate (y : ℝ) : 
  (∃ (m b : ℝ), m = 2 ∧ b = -2 ∧ 
    (∀ x : ℝ, y = m * (3 - x) + (m * x + b)) ∧
    (0 = m * 1 + b)) → 
  y = 4 :=
by sorry

end NUMINAMATH_CALUDE_line_y_coordinate_l1391_139152


namespace NUMINAMATH_CALUDE_closest_point_l1391_139172

def v (t : ℝ) : Fin 3 → ℝ := fun i => 
  match i with
  | 0 => 2 + 7*t
  | 1 => -3 + 5*t
  | 2 => -3 - t

def a : Fin 3 → ℝ := fun i =>
  match i with
  | 0 => 4
  | 1 => 4
  | 2 => 5

def direction : Fin 3 → ℝ := fun i =>
  match i with
  | 0 => 7
  | 1 => 5
  | 2 => -1

theorem closest_point (t : ℝ) : 
  (∀ s : ℝ, ‖v t - a‖ ≤ ‖v s - a‖) ↔ t = 41/75 :=
sorry

end NUMINAMATH_CALUDE_closest_point_l1391_139172


namespace NUMINAMATH_CALUDE_larger_number_proof_l1391_139109

theorem larger_number_proof (S L : ℕ) 
  (h1 : L - S = 1365)
  (h2 : L = 6 * S + 10) : 
  L = 1636 := by
sorry

end NUMINAMATH_CALUDE_larger_number_proof_l1391_139109


namespace NUMINAMATH_CALUDE_sum_a_b_c_value_l1391_139197

theorem sum_a_b_c_value :
  ∀ (a b c : ℤ),
  (∀ x : ℤ, x < 0 → x ≤ a) →  -- a is the largest negative integer
  (abs b = 6) →               -- |b| = 6
  (c = -c) →                  -- c is equal to its opposite
  (a + b + c = -7 ∨ a + b + c = 5) := by
sorry

end NUMINAMATH_CALUDE_sum_a_b_c_value_l1391_139197


namespace NUMINAMATH_CALUDE_oranges_in_box_l1391_139140

/-- Given an initial number of oranges in a box and a number of oranges added,
    the final number of oranges in the box is equal to the sum of the initial number and the added number. -/
theorem oranges_in_box (initial : ℝ) (added : ℝ) :
  initial + added = 90 :=
by sorry

end NUMINAMATH_CALUDE_oranges_in_box_l1391_139140


namespace NUMINAMATH_CALUDE_triangle_angle_determinant_zero_l1391_139193

theorem triangle_angle_determinant_zero (A B C : ℝ) (h : A + B + C = π) :
  let M : Matrix (Fin 3) (Fin 3) ℝ := ![![Real.cos A ^ 2, Real.tan A, 1],
                                        ![Real.cos B ^ 2, Real.tan B, 1],
                                        ![Real.cos C ^ 2, Real.tan C, 1]]
  Matrix.det M = 0 := by sorry

end NUMINAMATH_CALUDE_triangle_angle_determinant_zero_l1391_139193


namespace NUMINAMATH_CALUDE_sinusoidal_vertical_shift_l1391_139131

/-- For a sinusoidal function y = a sin(bx + c) + d, 
    if it oscillates between 4 and -2, then d = 1 -/
theorem sinusoidal_vertical_shift 
  (a b c d : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0) 
  (h_oscillation : ∀ x, -2 ≤ a * Real.sin (b * x + c) + d ∧ 
                        a * Real.sin (b * x + c) + d ≤ 4) : 
  d = 1 := by
  sorry

end NUMINAMATH_CALUDE_sinusoidal_vertical_shift_l1391_139131


namespace NUMINAMATH_CALUDE_quadratic_roots_always_positive_implies_a_zero_l1391_139100

theorem quadratic_roots_always_positive_implies_a_zero 
  (a b c : ℝ) 
  (h : ∀ (p : ℝ), p > 0 → ∀ (x : ℝ), a * x^2 + b * x + c + p = 0 → x > 0) : 
  a = 0 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_always_positive_implies_a_zero_l1391_139100


namespace NUMINAMATH_CALUDE_paper_supply_duration_l1391_139116

/-- The number of weeks in a year -/
def weeks_per_year : ℕ := 52

/-- The number of stories John writes per week -/
def stories_per_week : ℕ := 3

/-- The number of pages in each short story -/
def pages_per_story : ℕ := 50

/-- The number of pages in John's yearly novel -/
def novel_pages_per_year : ℕ := 1200

/-- The number of pages that can fit on one sheet of paper -/
def pages_per_sheet : ℕ := 2

/-- The number of reams of paper John buys -/
def reams_bought : ℕ := 3

/-- The number of sheets in each ream of paper -/
def sheets_per_ream : ℕ := 500

/-- The number of weeks John is buying paper for -/
def weeks_of_paper_supply : ℕ := 18

theorem paper_supply_duration :
  let total_pages_per_week := stories_per_week * pages_per_story + novel_pages_per_year / weeks_per_year
  let sheets_per_week := (total_pages_per_week + pages_per_sheet - 1) / pages_per_sheet
  let total_sheets := reams_bought * sheets_per_ream
  (total_sheets + sheets_per_week - 1) / sheets_per_week = weeks_of_paper_supply :=
by sorry

end NUMINAMATH_CALUDE_paper_supply_duration_l1391_139116


namespace NUMINAMATH_CALUDE_max_coach_handshakes_zero_l1391_139133

/-- The total number of handshakes in the tournament -/
def total_handshakes : ℕ := 465

/-- The number of players in the tournament -/
def num_players : ℕ := 31

/-- The number of handshakes between players -/
def player_handshakes (n : ℕ) : ℕ := n * (n - 1) / 2

/-- The number of handshakes the coach participated in -/
def coach_handshakes : ℕ := total_handshakes - player_handshakes num_players

theorem max_coach_handshakes_zero :
  coach_handshakes = 0 ∧ 
  ∀ n : ℕ, n > num_players → player_handshakes n > total_handshakes := by
  sorry


end NUMINAMATH_CALUDE_max_coach_handshakes_zero_l1391_139133


namespace NUMINAMATH_CALUDE_triangle_trig_identity_l1391_139136

theorem triangle_trig_identity (D E F : Real) (DE DF EF : Real) : 
  DE = 7 → DF = 8 → EF = 5 → 
  (Real.cos ((D - E) / 2) / Real.sin (F / 2)) - 
  (Real.sin ((D - E) / 2) / Real.cos (F / 2)) = 16 / 7 := by
  sorry

end NUMINAMATH_CALUDE_triangle_trig_identity_l1391_139136


namespace NUMINAMATH_CALUDE_square_construction_possible_l1391_139196

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A circle in 2D space -/
structure Circle where
  center : Point
  radius : ℝ

/-- Represent a compass operation -/
inductive CompassOp
  | drawCircle (center : Point) (radius : ℝ)
  | findIntersection (c1 : Circle) (c2 : Circle)

/-- Represent a sequence of compass operations -/
def CompassConstruction := List CompassOp

/-- The center of the square -/
def O : Point := sorry

/-- One vertex of the square -/
def A : Point := sorry

/-- The radius of the circumcircle -/
def r : ℝ := sorry

/-- Check if a point is a vertex of the square -/
def isSquareVertex (p : Point) : Prop := sorry

/-- Check if a construction is valid (uses only compass operations) -/
def isValidConstruction (c : CompassConstruction) : Prop := sorry

/-- The main theorem: it's possible to construct the other vertices using only a compass -/
theorem square_construction_possible :
  ∃ (B C D : Point) (construction : CompassConstruction),
    isValidConstruction construction ∧
    isSquareVertex B ∧
    isSquareVertex C ∧
    isSquareVertex D :=
  sorry

end NUMINAMATH_CALUDE_square_construction_possible_l1391_139196


namespace NUMINAMATH_CALUDE_solve_system_l1391_139179

theorem solve_system (x y z w : ℤ)
  (eq1 : x + y = 4)
  (eq2 : x - y = 36)
  (eq3 : x * z + y * w = 50)
  (eq4 : z - w = 5) :
  x = 20 := by
sorry

end NUMINAMATH_CALUDE_solve_system_l1391_139179


namespace NUMINAMATH_CALUDE_mixed_groups_count_l1391_139161

/-- Represents the chess club structure and game results -/
structure ChessClub where
  total_children : ℕ
  total_groups : ℕ
  children_per_group : ℕ
  boy_vs_boy_games : ℕ
  girl_vs_girl_games : ℕ

/-- Calculates the number of mixed groups in the chess club -/
def mixed_groups (club : ChessClub) : ℕ :=
  let total_games := club.total_groups * (club.children_per_group.choose 2)
  let mixed_games := total_games - club.boy_vs_boy_games - club.girl_vs_girl_games
  mixed_games / 2

/-- Theorem stating that the number of mixed groups is 23 -/
theorem mixed_groups_count (club : ChessClub) 
  (h1 : club.total_children = 90)
  (h2 : club.total_groups = 30)
  (h3 : club.children_per_group = 3)
  (h4 : club.boy_vs_boy_games = 30)
  (h5 : club.girl_vs_girl_games = 14) :
  mixed_groups club = 23 := by
  sorry

#eval mixed_groups ⟨90, 30, 3, 30, 14⟩

end NUMINAMATH_CALUDE_mixed_groups_count_l1391_139161


namespace NUMINAMATH_CALUDE_min_sum_same_last_three_digits_l1391_139127

/-- Given two positive integers m and n where n > m ≥ 1, this theorem states that
    if 1978^n and 1978^m have the same last three digits, then m + n ≥ 106. -/
theorem min_sum_same_last_three_digits (m n : ℕ) (hm : m ≥ 1) (hn : n > m) :
  (1978^n : ℕ) % 1000 = (1978^m : ℕ) % 1000 → m + n ≥ 106 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_same_last_three_digits_l1391_139127


namespace NUMINAMATH_CALUDE_parabola_vertex_l1391_139181

/-- The parabola is defined by the equation y = 2(x-5)^2 + 3 -/
def parabola (x y : ℝ) : Prop := y = 2 * (x - 5)^2 + 3

/-- The vertex of a parabola is the point where it reaches its minimum or maximum -/
def is_vertex (x₀ y₀ : ℝ) : Prop :=
  ∀ x y, parabola x y → y ≥ y₀

/-- Theorem: The vertex of the parabola y = 2(x-5)^2 + 3 has coordinates (5, 3) -/
theorem parabola_vertex :
  is_vertex 5 3 :=
sorry

end NUMINAMATH_CALUDE_parabola_vertex_l1391_139181


namespace NUMINAMATH_CALUDE_zoo_elephant_count_l1391_139141

/-- Represents the number of animals of each type in the zoo -/
structure ZooPopulation where
  giraffes : ℕ
  penguins : ℕ
  elephants : ℕ
  total : ℕ

/-- The conditions of the zoo population -/
def zoo_conditions (pop : ZooPopulation) : Prop :=
  pop.giraffes = 5 ∧
  pop.penguins = 2 * pop.giraffes ∧
  pop.penguins = (20 : ℕ) * pop.total / 100 ∧
  pop.elephants = (4 : ℕ) * pop.total / 100 ∧
  pop.total = pop.giraffes + pop.penguins + pop.elephants

theorem zoo_elephant_count :
  ∀ pop : ZooPopulation, zoo_conditions pop → pop.elephants = 2 :=
by sorry

end NUMINAMATH_CALUDE_zoo_elephant_count_l1391_139141


namespace NUMINAMATH_CALUDE_largest_fraction_l1391_139143

theorem largest_fraction : 
  let fractions := [2/5, 3/7, 4/9, 3/8, 9/20]
  ∀ f ∈ fractions, (9:ℚ)/20 ≥ f := by sorry

end NUMINAMATH_CALUDE_largest_fraction_l1391_139143


namespace NUMINAMATH_CALUDE_circle_and_five_lines_max_regions_circle_divides_plane_two_parts_circle_and_one_line_max_four_parts_circle_and_two_lines_max_eight_parts_l1391_139151

/-- The maximum number of regions into which n lines can divide a plane -/
def max_regions_lines (n : ℕ) : ℕ := n * (n + 1) / 2 + 1

/-- The maximum number of additional regions created when k lines intersect a circle -/
def max_additional_regions (k : ℕ) : ℕ := k * 2

/-- The maximum number of regions into which a plane can be divided by 1 circle and n lines -/
def max_regions_circle_and_lines (n : ℕ) : ℕ :=
  max_regions_lines n + max_additional_regions n

theorem circle_and_five_lines_max_regions :
  max_regions_circle_and_lines 5 = 26 :=
by sorry

theorem circle_divides_plane_two_parts :
  max_regions_circle_and_lines 0 = 2 :=
by sorry

theorem circle_and_one_line_max_four_parts :
  max_regions_circle_and_lines 1 = 4 :=
by sorry

theorem circle_and_two_lines_max_eight_parts :
  max_regions_circle_and_lines 2 = 8 :=
by sorry

end NUMINAMATH_CALUDE_circle_and_five_lines_max_regions_circle_divides_plane_two_parts_circle_and_one_line_max_four_parts_circle_and_two_lines_max_eight_parts_l1391_139151


namespace NUMINAMATH_CALUDE_initial_geese_count_l1391_139130

theorem initial_geese_count (G : ℕ) : 
  (G / 2 + 4 = 12) → G = 16 := by
  sorry

end NUMINAMATH_CALUDE_initial_geese_count_l1391_139130


namespace NUMINAMATH_CALUDE_sum_of_extreme_prime_factors_1242_l1391_139188

theorem sum_of_extreme_prime_factors_1242 : 
  ∃ (small large : ℕ), 
    small.Prime ∧ 
    large.Prime ∧ 
    small ≤ large ∧
    (∀ p : ℕ, p.Prime → p ∣ 1242 → small ≤ p ∧ p ≤ large) ∧
    small + large = 25 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_extreme_prime_factors_1242_l1391_139188


namespace NUMINAMATH_CALUDE_negation_existential_proposition_l1391_139175

theorem negation_existential_proposition :
  (¬ ∃ x : ℝ, x^2 + x + 1 ≥ 0) ↔ (∀ x : ℝ, x^2 + x + 1 < 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_existential_proposition_l1391_139175


namespace NUMINAMATH_CALUDE_population_ratio_x_to_z_l1391_139119

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

end NUMINAMATH_CALUDE_population_ratio_x_to_z_l1391_139119


namespace NUMINAMATH_CALUDE_vector_magnitude_problem_l1391_139114

-- Define the vectors
def a (m : ℝ) : ℝ × ℝ := (m, 2)
def b (n : ℝ) : ℝ × ℝ := (-1, n)

-- Define the theorem
theorem vector_magnitude_problem (m n : ℝ) : 
  n > 0 ∧ 
  (a m) • (b n) = 0 ∧ 
  m^2 + n^2 = 5 → 
  ‖2 • (a m) + (b n)‖ = Real.sqrt 34 := by
  sorry

end NUMINAMATH_CALUDE_vector_magnitude_problem_l1391_139114


namespace NUMINAMATH_CALUDE_new_customers_calculation_l1391_139155

theorem new_customers_calculation (initial_customers final_customers : ℕ) :
  initial_customers = 3 →
  final_customers = 8 →
  final_customers - initial_customers = 5 := by
  sorry

end NUMINAMATH_CALUDE_new_customers_calculation_l1391_139155


namespace NUMINAMATH_CALUDE_soccer_balls_per_basket_l1391_139115

theorem soccer_balls_per_basket
  (num_baskets : ℕ)
  (tennis_balls_per_basket : ℕ)
  (total_balls_removed : ℕ)
  (balls_remaining : ℕ)
  (h1 : num_baskets = 5)
  (h2 : tennis_balls_per_basket = 15)
  (h3 : total_balls_removed = 44)
  (h4 : balls_remaining = 56) :
  (num_baskets * tennis_balls_per_basket + num_baskets * 5 = balls_remaining + total_balls_removed) := by
sorry

end NUMINAMATH_CALUDE_soccer_balls_per_basket_l1391_139115


namespace NUMINAMATH_CALUDE_problem_solution_l1391_139125

open Real

/-- The function f(x) = e^x + sin(x) + b -/
noncomputable def f (b : ℝ) (x : ℝ) : ℝ := exp x + sin x + b

/-- The function g(x) = xe^x -/
noncomputable def g (x : ℝ) : ℝ := x * exp x

theorem problem_solution :
  (∀ b : ℝ, (∀ x : ℝ, x ≥ 0 → f b x ≥ 0) → b ≥ -1) ∧
  (∀ m : ℝ, (∃ b : ℝ, (∀ x : ℝ, exp x + b = x - 1) ∧
                     (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ exp x₁ - 2 = (m - 2*x₁)/x₁ ∧
                                   exp x₂ - 2 = (m - 2*x₂)/x₂) ∧
                     (∀ x : ℝ, exp x - 2 = (m - 2*x)/x → x = x₁ ∨ x = x₂)) →
   -1/exp 1 < m ∧ m < 0) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l1391_139125


namespace NUMINAMATH_CALUDE_complement_of_A_l1391_139163

def A : Set ℝ := {x | x^2 - 2*x - 3 > 0}

theorem complement_of_A : 
  Set.compl A = Set.Icc (-1 : ℝ) 3 := by sorry

end NUMINAMATH_CALUDE_complement_of_A_l1391_139163


namespace NUMINAMATH_CALUDE_solve_exponential_equation_l1391_139121

theorem solve_exponential_equation :
  ∃ n : ℝ, (3 : ℝ) ^ n * (3 : ℝ) ^ n * (3 : ℝ) ^ n * (3 : ℝ) ^ n = (81 : ℝ) ^ 2 → n = 2 :=
by sorry

end NUMINAMATH_CALUDE_solve_exponential_equation_l1391_139121


namespace NUMINAMATH_CALUDE_negative_fraction_comparison_l1391_139178

theorem negative_fraction_comparison : -1/3 < -1/4 := by
  sorry

end NUMINAMATH_CALUDE_negative_fraction_comparison_l1391_139178


namespace NUMINAMATH_CALUDE_april_roses_theorem_l1391_139120

/-- The number of roses April started with -/
def initial_roses : ℕ := 9

/-- The price of each rose -/
def rose_price : ℕ := 7

/-- The number of roses left after the sale -/
def roses_left : ℕ := 4

/-- The total earnings from the sale -/
def total_earnings : ℕ := 35

/-- Theorem stating that the initial number of roses is correct given the conditions -/
theorem april_roses_theorem :
  initial_roses = (total_earnings / rose_price) + roses_left :=
sorry

end NUMINAMATH_CALUDE_april_roses_theorem_l1391_139120


namespace NUMINAMATH_CALUDE_unique_magnitude_complex_roots_l1391_139129

theorem unique_magnitude_complex_roots (z : ℂ) :
  (3 * z^2 - 18 * z + 55 = 0) →
  ∃! m : ℝ, ∃ z₁ z₂ : ℂ, (3 * z₁^2 - 18 * z₁ + 55 = 0) ∧
                         (3 * z₂^2 - 18 * z₂ + 55 = 0) ∧
                         (Complex.abs z₁ = m) ∧
                         (Complex.abs z₂ = m) :=
by sorry

end NUMINAMATH_CALUDE_unique_magnitude_complex_roots_l1391_139129


namespace NUMINAMATH_CALUDE_ellipse_foci_l1391_139113

/-- Represents an ellipse with equation x²/a² + y²/b² = 1 -/
structure Ellipse where
  a : ℝ
  b : ℝ

/-- The foci of an ellipse -/
structure Foci where
  x₁ : ℝ
  y₁ : ℝ
  x₂ : ℝ
  y₂ : ℝ

/-- Theorem: The foci of the ellipse x²/1 + y²/10 = 1 are (0, -3) and (0, 3) -/
theorem ellipse_foci (e : Ellipse) (h₁ : e.a = 1) (h₂ : e.b = 10) :
  ∃ f : Foci, f.x₁ = 0 ∧ f.y₁ = -3 ∧ f.x₂ = 0 ∧ f.y₂ = 3 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_foci_l1391_139113


namespace NUMINAMATH_CALUDE_jason_born_1981_l1391_139156

/-- The year of the first AMC 8 competition -/
def first_amc8_year : ℕ := 1985

/-- The number of the AMC 8 competition Jason participated in -/
def jason_amc8_number : ℕ := 10

/-- Jason's age when he participated in the AMC 8 -/
def jason_age : ℕ := 13

/-- Calculates the year of a given AMC 8 competition -/
def amc8_year (n : ℕ) : ℕ := first_amc8_year + n - 1

/-- Jason's birth year -/
def jason_birth_year : ℕ := amc8_year jason_amc8_number - jason_age

theorem jason_born_1981 : jason_birth_year = 1981 := by
  sorry

end NUMINAMATH_CALUDE_jason_born_1981_l1391_139156


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l1391_139180

theorem arithmetic_calculation : 1273 + 120 / 60 - 173 = 1102 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l1391_139180


namespace NUMINAMATH_CALUDE_function_inequality_implies_parameter_bound_l1391_139108

open Real

theorem function_inequality_implies_parameter_bound 
  (f g h : ℝ → ℝ)
  (hf : ∀ x, f x = 1/2 * x^2 - 2*x)
  (hg : ∀ x, g x = a * log x)
  (hh : ∀ x, h x = f x - g x)
  (h_pos : ∀ x, x > 0)
  (h_ineq : ∀ x₁ x₂, x₁ > 0 → x₂ > 0 → x₁ ≠ x₂ → (h x₁ - h x₂) / (x₁ - x₂) > 2)
  : a ≤ -4 :=
sorry

end NUMINAMATH_CALUDE_function_inequality_implies_parameter_bound_l1391_139108


namespace NUMINAMATH_CALUDE_soccer_team_average_goals_l1391_139167

/-- The average number of goals scored by the soccer team per game -/
def team_average_goals (carter_goals shelby_goals judah_goals : ℝ) : ℝ :=
  carter_goals + shelby_goals + judah_goals

/-- Theorem stating the average total number of goals scored by the team per game -/
theorem soccer_team_average_goals :
  ∃ (carter_goals shelby_goals judah_goals : ℝ),
    carter_goals = 4 ∧
    shelby_goals = carter_goals / 2 ∧
    judah_goals = 2 * shelby_goals - 3 ∧
    team_average_goals carter_goals shelby_goals judah_goals = 7 := by
  sorry


end NUMINAMATH_CALUDE_soccer_team_average_goals_l1391_139167


namespace NUMINAMATH_CALUDE_student_rabbit_difference_is_95_l1391_139177

/-- The number of students in each fourth-grade classroom -/
def students_per_classroom : ℕ := 22

/-- The number of rabbits in each fourth-grade classroom -/
def rabbits_per_classroom : ℕ := 3

/-- The number of fourth-grade classrooms -/
def number_of_classrooms : ℕ := 5

/-- The difference between the total number of students and rabbits in all classrooms -/
def student_rabbit_difference : ℕ :=
  (students_per_classroom * number_of_classrooms) - (rabbits_per_classroom * number_of_classrooms)

theorem student_rabbit_difference_is_95 :
  student_rabbit_difference = 95 := by sorry

end NUMINAMATH_CALUDE_student_rabbit_difference_is_95_l1391_139177


namespace NUMINAMATH_CALUDE_train_length_calculation_l1391_139123

/-- Given two trains of equal length running on parallel lines in the same direction,
    this theorem proves the length of each train given their speeds and passing time. -/
theorem train_length_calculation (v_fast v_slow : ℝ) (t : ℝ) (L : ℝ) :
  v_fast = 50 →
  v_slow = 36 →
  t = 36 / 3600 →
  (v_fast - v_slow) * t = 2 * L →
  L = 0.07 :=
by sorry

end NUMINAMATH_CALUDE_train_length_calculation_l1391_139123


namespace NUMINAMATH_CALUDE_smallest_lcm_with_gcd_5_l1391_139135

theorem smallest_lcm_with_gcd_5 (k l : ℕ) : 
  1000 ≤ k ∧ k < 10000 ∧ 
  1000 ≤ l ∧ l < 10000 ∧ 
  Nat.gcd k l = 5 →
  ∀ m n : ℕ, 1000 ≤ m ∧ m < 10000 ∧ 
             1000 ≤ n ∧ n < 10000 ∧ 
             Nat.gcd m n = 5 →
  Nat.lcm k l ≤ Nat.lcm m n ∧
  Nat.lcm k l = 203010 :=
by sorry

end NUMINAMATH_CALUDE_smallest_lcm_with_gcd_5_l1391_139135


namespace NUMINAMATH_CALUDE_fraction_inequality_l1391_139106

theorem fraction_inequality (a b c d e : ℝ)
  (h1 : a > b)
  (h2 : b > 0)
  (h3 : c < d)
  (h4 : d < 0)
  (h5 : e < 0) :
  e / (a - c) > e / (b - d) := by
  sorry

end NUMINAMATH_CALUDE_fraction_inequality_l1391_139106


namespace NUMINAMATH_CALUDE_predicted_weight_approx_l1391_139176

-- Define the regression equation
def regression_equation (x : ℝ) : ℝ := 0.849 * x - 85.712

-- Define the height of the student
def student_height : ℝ := 172

-- Define the tolerance for "approximately" (e.g., within 0.001)
def tolerance : ℝ := 0.001

-- Theorem statement
theorem predicted_weight_approx :
  ∃ (predicted_weight : ℝ), 
    regression_equation student_height = predicted_weight ∧ 
    abs (predicted_weight - 60.316) < tolerance := by
  sorry

end NUMINAMATH_CALUDE_predicted_weight_approx_l1391_139176


namespace NUMINAMATH_CALUDE_smallest_multiple_21_g_gt_21_l1391_139118

/-- g(n) returns the smallest integer m such that m! is divisible by n -/
def g (n : ℕ) : ℕ := sorry

/-- 483 is the smallest multiple of 21 for which g(n) > 21 -/
theorem smallest_multiple_21_g_gt_21 : ∀ n : ℕ, n % 21 = 0 → g n ≤ 21 → n < 483 :=
sorry

end NUMINAMATH_CALUDE_smallest_multiple_21_g_gt_21_l1391_139118


namespace NUMINAMATH_CALUDE_sum_of_exponents_15_factorial_l1391_139194

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def largestPerfectSquareDivisor (n : ℕ) : ℕ :=
  sorry

def sumOfPrimeFactorExponents (n : ℕ) : ℕ :=
  sorry

theorem sum_of_exponents_15_factorial :
  sumOfPrimeFactorExponents (largestPerfectSquareDivisor (factorial 15).sqrt) = 9 :=
sorry

end NUMINAMATH_CALUDE_sum_of_exponents_15_factorial_l1391_139194
