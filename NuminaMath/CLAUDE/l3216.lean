import Mathlib

namespace gum_pack_size_l3216_321602

theorem gum_pack_size : ∃ x : ℕ+, 
  (30 : ℚ) - 2 * x.val = 30 * 40 / (40 + 4 * x.val) ∧ x = 5 := by
  sorry

end gum_pack_size_l3216_321602


namespace fraction_multiplication_addition_l3216_321672

theorem fraction_multiplication_addition : (1/3 : ℚ) * (2/5 : ℚ) + (1/4 : ℚ) = 23/60 := by
  sorry

end fraction_multiplication_addition_l3216_321672


namespace disjunction_truth_implication_l3216_321600

-- Define propositions p and q
variable (p q : Prop)

-- Define the statement to be proven
theorem disjunction_truth_implication (h : p ∨ q) : ¬(p ∧ q) := by
  sorry

-- This theorem states that if p ∨ q is true, it does not necessarily imply that both p and q are true.
-- It directly corresponds to showing that statement D is incorrect.

end disjunction_truth_implication_l3216_321600


namespace tax_discount_commute_price_difference_is_zero_l3216_321629

/-- Proves that the order of applying tax and discount doesn't affect the final price -/
theorem tax_discount_commute (price : ℝ) (tax_rate : ℝ) (discount_rate : ℝ) 
  (h_tax : 0 ≤ tax_rate) (h_discount : 0 ≤ discount_rate) (h_discount_max : discount_rate ≤ 1) :
  price * (1 + tax_rate) * (1 - discount_rate) = price * (1 - discount_rate) * (1 + tax_rate) :=
by sorry

/-- Calculates the difference between applying tax then discount and applying discount then tax -/
def price_difference (price : ℝ) (tax_rate : ℝ) (discount_rate : ℝ) : ℝ :=
  price * (1 + tax_rate) * (1 - discount_rate) - price * (1 - discount_rate) * (1 + tax_rate)

/-- Proves that the price difference is always zero -/
theorem price_difference_is_zero (price : ℝ) (tax_rate : ℝ) (discount_rate : ℝ) 
  (h_tax : 0 ≤ tax_rate) (h_discount : 0 ≤ discount_rate) (h_discount_max : discount_rate ≤ 1) :
  price_difference price tax_rate discount_rate = 0 :=
by sorry

end tax_discount_commute_price_difference_is_zero_l3216_321629


namespace coconut_trips_l3216_321639

def total_coconuts : ℕ := 144
def barbie_capacity : ℕ := 4
def bruno_capacity : ℕ := 8

theorem coconut_trips : 
  (total_coconuts / (barbie_capacity + bruno_capacity) : ℕ) = 12 := by
  sorry

end coconut_trips_l3216_321639


namespace derivative_f_at_one_l3216_321682

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x

theorem derivative_f_at_one : 
  deriv f 1 = 1 :=
sorry

end derivative_f_at_one_l3216_321682


namespace sandy_molly_age_difference_l3216_321661

theorem sandy_molly_age_difference :
  ∀ (sandy_age molly_age : ℕ),
  sandy_age = 56 →
  sandy_age * 9 = molly_age * 7 →
  molly_age - sandy_age = 16 :=
by
  sorry

end sandy_molly_age_difference_l3216_321661


namespace not_always_parallel_lines_l3216_321649

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relations
variable (parallel_line_plane : Line → Plane → Prop)
variable (parallel_plane_plane : Plane → Plane → Prop)
variable (line_in_plane : Line → Plane → Prop)
variable (parallel_line_line : Line → Line → Prop)

-- State the theorem
theorem not_always_parallel_lines 
  (α β : Plane) (m n : Line) 
  (h1 : parallel_line_plane m α) 
  (h2 : parallel_plane_plane α β) 
  (h3 : line_in_plane n β) : 
  ¬(∀ m n, parallel_line_line m n) := by
sorry


end not_always_parallel_lines_l3216_321649


namespace exponential_inequality_l3216_321606

theorem exponential_inequality (x₁ x₂ : ℝ) (h : x₁ ≠ x₂) :
  Real.exp ((x₁ + x₂) / 2) < (Real.exp x₁ + Real.exp x₂) / (x₁ - x₂) := by
  sorry

end exponential_inequality_l3216_321606


namespace largest_c_value_l3216_321623

theorem largest_c_value (c : ℝ) : 
  (3 * c + 4) * (c - 2) = 7 * c →
  c ≤ (9 + Real.sqrt 177) / 6 :=
by sorry

end largest_c_value_l3216_321623


namespace complex_equation_solution_l3216_321643

theorem complex_equation_solution (z : ℂ) :
  (1 - Complex.I)^2 / z = 1 + Complex.I → z = -1 - Complex.I :=
by
  sorry

end complex_equation_solution_l3216_321643


namespace square_grid_15_toothpicks_l3216_321651

/-- Calculates the total number of toothpicks needed for a square grid -/
def toothpicks_in_square_grid (side_length : ℕ) : ℕ :=
  2 * (side_length + 1) * side_length

/-- Theorem: A square grid with 15 toothpicks on each side requires 480 toothpicks -/
theorem square_grid_15_toothpicks :
  toothpicks_in_square_grid 15 = 480 := by
  sorry

#eval toothpicks_in_square_grid 15

end square_grid_15_toothpicks_l3216_321651


namespace min_value_a_plus_2b_l3216_321612

theorem min_value_a_plus_2b (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 1/a + 3/b = 1) :
  ∃ (min : ℝ), (∀ x y, x > 0 → y > 0 → 1/x + 3/y = 1 → x + 2*y ≥ min) ∧ (a + 2*b = min) :=
by
  -- The minimum value is 7 + 2√6
  let min := 7 + 2 * Real.sqrt 6
  -- Proof goes here
  sorry

end min_value_a_plus_2b_l3216_321612


namespace sqrt_real_range_l3216_321644

theorem sqrt_real_range (x : ℝ) : (∃ y : ℝ, y ^ 2 = x - 3) ↔ x ≥ 3 := by sorry

end sqrt_real_range_l3216_321644


namespace cos_sin_75_deg_l3216_321646

theorem cos_sin_75_deg : Real.cos (75 * π / 180) ^ 4 - Real.sin (75 * π / 180) ^ 4 = -Real.sqrt 3 / 2 := by
  sorry

end cos_sin_75_deg_l3216_321646


namespace sum_abcd_equals_1986_l3216_321699

theorem sum_abcd_equals_1986 
  (h1 : 6 * a + 2 * b = 3848) 
  (h2 : 6 * c + 3 * d = 4410) 
  (h3 : a + 3 * b + 2 * d = 3080) : 
  a + b + c + d = 1986 := by
  sorry

end sum_abcd_equals_1986_l3216_321699


namespace adult_admission_price_l3216_321619

theorem adult_admission_price
  (total_people : ℕ)
  (total_receipts : ℕ)
  (num_children : ℕ)
  (child_price : ℕ)
  (h1 : total_people = 610)
  (h2 : total_receipts = 960)
  (h3 : num_children = 260)
  (h4 : child_price = 1) :
  (total_receipts - num_children * child_price) / (total_people - num_children) = 2 :=
by sorry

end adult_admission_price_l3216_321619


namespace min_sum_a_b_l3216_321688

theorem min_sum_a_b (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 1/a + 9/b = 1) : 
  ∀ x y : ℝ, x > 0 → y > 0 → 1/x + 9/y = 1 → a + b ≤ x + y ∧ ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ 1/x + 9/y = 1 ∧ x + y = 16 :=
sorry

end min_sum_a_b_l3216_321688


namespace dislike_tv_and_books_l3216_321601

/-- Given a survey of people, calculate the number who dislike both TV and books -/
theorem dislike_tv_and_books (total : ℕ) (tv_dislike_percent : ℚ) (book_dislike_percent : ℚ) :
  total = 1500 →
  tv_dislike_percent = 40 / 100 →
  book_dislike_percent = 15 / 100 →
  (total * tv_dislike_percent * book_dislike_percent).floor = 90 := by
  sorry

end dislike_tv_and_books_l3216_321601


namespace supplies_to_budget_ratio_l3216_321633

def total_budget : ℚ := 3000
def food_fraction : ℚ := 1/3
def wages : ℚ := 1250

def supplies : ℚ := total_budget - (food_fraction * total_budget) - wages

theorem supplies_to_budget_ratio : 
  supplies / total_budget = 1/4 := by sorry

end supplies_to_budget_ratio_l3216_321633


namespace rain_probability_three_days_l3216_321610

def prob_rain_friday : ℝ := 0.4
def prob_rain_saturday : ℝ := 0.7
def prob_rain_sunday : ℝ := 0.3

theorem rain_probability_three_days :
  let prob_all_days := prob_rain_friday * prob_rain_saturday * prob_rain_sunday
  prob_all_days = 0.084 := by
  sorry

end rain_probability_three_days_l3216_321610


namespace unique_line_through_points_l3216_321650

-- Define a type for points in Euclidean geometry
variable (Point : Type)

-- Define a type for lines in Euclidean geometry
variable (Line : Type)

-- Define a relation for a point being on a line
variable (on_line : Point → Line → Prop)

-- Axiom: For any two distinct points, there exists a line passing through both points
axiom line_through_points (P Q : Point) (h : P ≠ Q) : ∃ (l : Line), on_line P l ∧ on_line Q l

-- Axiom: Any line passing through two distinct points is unique
axiom line_uniqueness (P Q : Point) (h : P ≠ Q) (l1 l2 : Line) :
  on_line P l1 ∧ on_line Q l1 → on_line P l2 ∧ on_line Q l2 → l1 = l2

-- Theorem: There exists exactly one line passing through any two distinct points
theorem unique_line_through_points (P Q : Point) (h : P ≠ Q) :
  ∃! (l : Line), on_line P l ∧ on_line Q l :=
sorry

end unique_line_through_points_l3216_321650


namespace money_split_proof_l3216_321662

/-- The total amount of money found by Donna and her friends -/
def total_money : ℝ := 97.50

/-- Donna's share of the money as a percentage -/
def donna_share : ℝ := 0.40

/-- The amount Donna received in dollars -/
def donna_amount : ℝ := 39

/-- Theorem stating that if Donna received 40% of the total money and her share was $39, 
    then the total amount of money found was $97.50 -/
theorem money_split_proof : 
  donna_share * total_money = donna_amount → total_money = 97.50 := by
  sorry

end money_split_proof_l3216_321662


namespace triangle_trig_identities_l3216_321691

/-- Given an acute triangle ABC with area 3√3, side lengths AB = 3 and AC = 4, 
    prove the following trigonometric identities involving its angles. -/
theorem triangle_trig_identities 
  (A B C : Real) 
  (h_acute : A > 0 ∧ B > 0 ∧ C > 0 ∧ A + B + C = π)
  (h_area : (1/2) * 3 * 4 * Real.sin A = 3 * Real.sqrt 3)
  (h_AB : 3 = 3)
  (h_AC : 4 = 4) :
  Real.sin (π/2 + A) = 1/2 ∧ 
  Real.cos (A - B) = (7 * Real.sqrt 13) / 26 := by
sorry

end triangle_trig_identities_l3216_321691


namespace amount_to_return_l3216_321647

/-- Represents the exchange rate in rubles per dollar -/
def exchange_rate : ℝ := 58.15

/-- Represents the initial deposit in USD -/
def initial_deposit : ℝ := 10000

/-- Calculates the amount to be returned in rubles -/
def amount_in_rubles : ℝ := initial_deposit * exchange_rate

/-- Theorem stating that the amount to be returned is 581,500 rubles -/
theorem amount_to_return : amount_in_rubles = 581500 := by
  sorry

end amount_to_return_l3216_321647


namespace inclination_angle_tangent_l3216_321665

theorem inclination_angle_tangent (α : ℝ) : 
  (∃ (x y : ℝ), 2 * x + y + 1 = 0 ∧ α = Real.arctan (-2)) → 
  Real.tan (α - π / 4) = 3 := by
sorry

end inclination_angle_tangent_l3216_321665


namespace smallest_number_divisible_when_increased_l3216_321645

def is_divisible_by_all (n : ℕ) (divisors : List ℕ) : Prop :=
  ∀ d ∈ divisors, (n % d = 0)

theorem smallest_number_divisible_when_increased : ∃! n : ℕ, 
  (is_divisible_by_all (n + 9) [8, 11, 24]) ∧ 
  (∀ m : ℕ, m < n → ¬ is_divisible_by_all (m + 9) [8, 11, 24]) ∧
  n = 255 := by
  sorry

end smallest_number_divisible_when_increased_l3216_321645


namespace max_distance_product_l3216_321689

/-- Triangle with side lengths 3, 4, and 5 -/
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  a_eq : a = 3
  b_eq : b = 4
  c_eq : c = 5

/-- Point inside the triangle -/
structure InteriorPoint (t : RightTriangle) where
  x : ℝ
  y : ℝ
  interior : 0 < x ∧ 0 < y ∧ x + y < 1

/-- Distances from a point to the sides of the triangle -/
def distances (t : RightTriangle) (p : InteriorPoint t) : ℝ × ℝ × ℝ :=
  (p.x, p.y, 1 - p.x - p.y)

/-- Product of distances from a point to the sides of the triangle -/
def distanceProduct (t : RightTriangle) (p : InteriorPoint t) : ℝ :=
  let (d₁, d₂, d₃) := distances t p
  d₁ * d₂ * d₃

theorem max_distance_product (t : RightTriangle) :
  ∀ p : InteriorPoint t, distanceProduct t p ≤ 1/125 := by
  sorry

end max_distance_product_l3216_321689


namespace truthful_dwarfs_count_l3216_321603

-- Define the total number of dwarfs
def total_dwarfs : ℕ := 10

-- Define the number of dwarfs who raised hands for each ice cream type
def vanilla_hands : ℕ := total_dwarfs
def chocolate_hands : ℕ := total_dwarfs / 2
def fruit_hands : ℕ := 1

-- Define the total number of hands raised
def total_hands_raised : ℕ := vanilla_hands + chocolate_hands + fruit_hands

-- Theorem to prove
theorem truthful_dwarfs_count : 
  ∃ (truthful : ℕ) (lying : ℕ), 
    truthful + lying = total_dwarfs ∧ 
    lying = total_hands_raised - total_dwarfs ∧
    truthful = 4 := by
  sorry

end truthful_dwarfs_count_l3216_321603


namespace sqrt_two_over_two_gt_sqrt_three_over_three_l3216_321674

theorem sqrt_two_over_two_gt_sqrt_three_over_three :
  (Real.sqrt 2) / 2 > (Real.sqrt 3) / 3 := by
  sorry

end sqrt_two_over_two_gt_sqrt_three_over_three_l3216_321674


namespace sector_area_rate_of_change_l3216_321681

/-- The rate of change of a circular sector's area --/
theorem sector_area_rate_of_change
  (r : ℝ)
  (θ : ℝ → ℝ)
  (h_r : r = 12)
  (h_θ : ∀ t, θ t = 38 + 5 * t) :
  ∀ t, (deriv (λ t => (1/2) * r^2 * (θ t * π / 180))) t = 2 * π :=
sorry

end sector_area_rate_of_change_l3216_321681


namespace cubic_equality_solution_l3216_321617

theorem cubic_equality_solution : ∃ n : ℤ, 3^3 - 5 = 4^2 + n ∧ n = 6 := by
  sorry

end cubic_equality_solution_l3216_321617


namespace a_55_divisible_by_55_l3216_321683

/-- Concatenation of integers from 1 to n -/
def a (n : ℕ) : ℕ :=
  -- Definition of a_n goes here
  sorry

/-- Theorem: a_55 is divisible by 55 -/
theorem a_55_divisible_by_55 : 55 ∣ a 55 := by
  sorry

end a_55_divisible_by_55_l3216_321683


namespace jen_profit_l3216_321666

/-- Calculates the profit in cents for a candy bar business -/
def candy_bar_profit (buy_price sell_price bought_quantity sold_quantity : ℕ) : ℤ :=
  (sell_price * sold_quantity : ℤ) - (buy_price * bought_quantity : ℤ)

/-- Proves that Jen's profit from her candy bar business is 800 cents -/
theorem jen_profit : candy_bar_profit 80 100 50 48 = 800 := by
  sorry

end jen_profit_l3216_321666


namespace cos_alpha_value_l3216_321669

theorem cos_alpha_value (α : Real) (h : Real.sin (α / 2) = Real.sqrt 3 / 3) : 
  Real.cos α = 1 / 3 := by
  sorry

end cos_alpha_value_l3216_321669


namespace rocket_ascent_time_l3216_321615

theorem rocket_ascent_time (n : ℕ) (a₁ d : ℝ) (h₁ : a₁ = 2) (h₂ : d = 2) :
  n * a₁ + (n * (n - 1) * d) / 2 = 240 → n = 15 :=
by sorry

end rocket_ascent_time_l3216_321615


namespace square_sum_implies_product_l3216_321640

theorem square_sum_implies_product (x : ℝ) :
  (Real.sqrt (10 + x) + Real.sqrt (15 - x) = 8) →
  ((10 + x) * (15 - x) = 1521 / 4) :=
by sorry

end square_sum_implies_product_l3216_321640


namespace rhombus_transformations_l3216_321621

/-- Represents a point transformation on the plane -/
def PointTransformation := (ℤ × ℤ) → (ℤ × ℤ)

/-- Transformation of type (i) -/
def transform_i (α : ℤ) : PointTransformation :=
  λ (x, y) => (x, α * x + y)

/-- Transformation of type (ii) -/
def transform_ii (α : ℤ) : PointTransformation :=
  λ (x, y) => (x + α * y, y)

/-- A rhombus with integer-coordinate vertices -/
structure IntegerRhombus :=
  (v1 v2 v3 v4 : ℤ × ℤ)

/-- Checks if a quadrilateral is a square -/
def is_square (q : IntegerRhombus) : Prop := sorry

/-- Checks if a quadrilateral is a non-square rectangle -/
def is_non_square_rectangle (q : IntegerRhombus) : Prop := sorry

/-- Applies a series of transformations to a rhombus -/
def apply_transformations (r : IntegerRhombus) (ts : List PointTransformation) : IntegerRhombus := sorry

/-- Main theorem statement -/
theorem rhombus_transformations :
  (¬ ∃ (r : IntegerRhombus) (ts : List PointTransformation),
     is_square (apply_transformations r ts)) ∧
  (∃ (r : IntegerRhombus) (ts : List PointTransformation),
     is_non_square_rectangle (apply_transformations r ts)) := by
  sorry

end rhombus_transformations_l3216_321621


namespace denominator_divisor_not_zero_l3216_321616

theorem denominator_divisor_not_zero :
  ∀ (a : ℝ), a ≠ 0 → (∃ (b : ℝ), b / a = b / a) ∧ (∃ (c d : ℝ), c / d = c / d) :=
by sorry

end denominator_divisor_not_zero_l3216_321616


namespace restaurant_group_l3216_321687

/-- Proves the number of kids in a group given the total number of people, 
    adult meal cost, and total cost. -/
theorem restaurant_group (total_people : ℕ) (adult_meal_cost : ℕ) (total_cost : ℕ) 
  (h1 : total_people = 9)
  (h2 : adult_meal_cost = 2)
  (h3 : total_cost = 14) :
  ∃ (num_kids : ℕ), 
    num_kids = total_people - (total_cost / adult_meal_cost) ∧ 
    num_kids = 2 := by
  sorry

end restaurant_group_l3216_321687


namespace cross_product_result_l3216_321608

def vector1 : ℝ × ℝ × ℝ := (3, -4, 7)
def vector2 : ℝ × ℝ × ℝ := (2, 5, -1)

def cross_product (v1 v2 : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (a1, a2, a3) := v1
  let (b1, b2, b3) := v2
  (a2 * b3 - a3 * b2, a3 * b1 - a1 * b3, a1 * b2 - a2 * b1)

theorem cross_product_result :
  cross_product vector1 vector2 = (-31, 17, 23) := by
  sorry

end cross_product_result_l3216_321608


namespace total_wattage_after_increase_l3216_321636

theorem total_wattage_after_increase (light_a light_b light_c light_d : ℝ)
  (increase_a increase_b increase_c increase_d : ℝ) :
  light_a = 60 →
  light_b = 40 →
  light_c = 50 →
  light_d = 80 →
  increase_a = 0.12 →
  increase_b = 0.20 →
  increase_c = 0.15 →
  increase_d = 0.10 →
  (light_a * (1 + increase_a) +
   light_b * (1 + increase_b) +
   light_c * (1 + increase_c) +
   light_d * (1 + increase_d)) = 260.7 := by
  sorry

end total_wattage_after_increase_l3216_321636


namespace quadratic_always_two_roots_l3216_321656

theorem quadratic_always_two_roots (k : ℝ) : 
  let a := (1 : ℝ)
  let b := 2 * k
  let c := k - 1
  let discriminant := b^2 - 4*a*c
  0 < discriminant := by sorry

end quadratic_always_two_roots_l3216_321656


namespace point_not_on_line_l3216_321678

theorem point_not_on_line (p q : ℝ) (h : p * q > 0) :
  ¬(∃ (x y : ℝ), x = 2023 ∧ y = 0 ∧ y = p * x + q) :=
by sorry

end point_not_on_line_l3216_321678


namespace opposite_signs_and_larger_absolute_value_l3216_321660

theorem opposite_signs_and_larger_absolute_value (a b : ℚ) 
  (h1 : a * b < 0) (h2 : a + b > 0) : 
  (a < 0 ∧ b > 0) ∨ (a > 0 ∧ b < 0) ∧ 
  ((a < 0 ∧ b > 0 → abs b > abs a) ∧ (a > 0 ∧ b < 0 → abs a > abs b)) :=
sorry

end opposite_signs_and_larger_absolute_value_l3216_321660


namespace range_of_a_range_of_m_l3216_321613

-- Define sets A, B, and C
def A (a : ℝ) := {x : ℝ | a^2 - a*x + x - 1 = 0}
def B (m : ℝ) := {x : ℝ | x^2 + x + m = 0}
def C := {x : ℝ | Real.sqrt (x^2) = x}

-- Theorem for part (1)
theorem range_of_a (a : ℝ) : A a ∪ C = C → a ∈ Set.Icc (-1) 1 ∪ Set.Ioi 1 := by
  sorry

-- Theorem for part (2)
theorem range_of_m (m : ℝ) : C ∩ B m = ∅ → m ∈ Set.Ioi 0 := by
  sorry

end range_of_a_range_of_m_l3216_321613


namespace initial_weight_calculation_l3216_321675

/-- 
Given a person who:
1. Loses 10% of their initial weight
2. Then gains 2 pounds
3. Ends up weighing 200 pounds

Their initial weight was 220 pounds.
-/
theorem initial_weight_calculation (initial_weight : ℝ) : 
  (initial_weight * 0.9 + 2 = 200) → initial_weight = 220 := by
  sorry

end initial_weight_calculation_l3216_321675


namespace equilateral_triangles_in_54gon_l3216_321628

/-- Represents a regular polygon with its center -/
structure RegularPolygonWithCenter (n : ℕ) where
  vertices : Fin n → ℝ × ℝ
  center : ℝ × ℝ

/-- Represents a selection of three points -/
structure TriangleSelection (n : ℕ) where
  p1 : Fin (n + 1)
  p2 : Fin (n + 1)
  p3 : Fin (n + 1)

/-- Checks if three points form an equilateral triangle -/
def isEquilateralTriangle (n : ℕ) (poly : RegularPolygonWithCenter n) (sel : TriangleSelection n) : Prop :=
  sorry

/-- Counts the number of equilateral triangles in a regular polygon with center -/
def countEquilateralTriangles (n : ℕ) (poly : RegularPolygonWithCenter n) : ℕ :=
  sorry

/-- The main theorem: there are 72 ways to select three points forming an equilateral triangle in a regular 54-gon with center -/
theorem equilateral_triangles_in_54gon :
  ∀ (poly : RegularPolygonWithCenter 54),
  countEquilateralTriangles 54 poly = 72 :=
sorry

end equilateral_triangles_in_54gon_l3216_321628


namespace sin_two_alpha_on_line_l3216_321635

/-- Given an angle α where its terminal side intersects the line y = 2x, prove that sin(2α) = 4/5 -/
theorem sin_two_alpha_on_line (α : Real) : 
  (∃ (P : Real × Real), P.2 = 2 * P.1 ∧ P ≠ (0, 0) ∧ 
    P.1 = Real.cos α * Real.sqrt (P.1^2 + P.2^2) ∧ 
    P.2 = Real.sin α * Real.sqrt (P.1^2 + P.2^2)) → 
  Real.sin (2 * α) = 4/5 := by
sorry

end sin_two_alpha_on_line_l3216_321635


namespace acorn_theorem_l3216_321668

def acorn_problem (total_acorns : ℕ) 
                  (first_month_allocation : ℚ) 
                  (second_month_allocation : ℚ) 
                  (third_month_allocation : ℚ) 
                  (first_month_consumption : ℚ) 
                  (second_month_consumption : ℚ) 
                  (third_month_consumption : ℚ) : Prop :=
  let first_month := (first_month_allocation * total_acorns : ℚ)
  let second_month := (second_month_allocation * total_acorns : ℚ)
  let third_month := (third_month_allocation * total_acorns : ℚ)
  let remaining_first := first_month * (1 - first_month_consumption)
  let remaining_second := second_month * (1 - second_month_consumption)
  let remaining_third := third_month * (1 - third_month_consumption)
  let total_remaining := remaining_first + remaining_second + remaining_third
  total_acorns = 500 ∧
  first_month_allocation = 2/5 ∧
  second_month_allocation = 3/10 ∧
  third_month_allocation = 3/10 ∧
  first_month_consumption = 1/5 ∧
  second_month_consumption = 1/4 ∧
  third_month_consumption = 3/20 ∧
  total_remaining = 400

theorem acorn_theorem : 
  ∃ (total_acorns : ℕ) 
    (first_month_allocation second_month_allocation third_month_allocation : ℚ)
    (first_month_consumption second_month_consumption third_month_consumption : ℚ),
  acorn_problem total_acorns 
                first_month_allocation 
                second_month_allocation 
                third_month_allocation 
                first_month_consumption 
                second_month_consumption 
                third_month_consumption :=
by
  sorry

end acorn_theorem_l3216_321668


namespace circles_diameter_sum_l3216_321614

theorem circles_diameter_sum (D d : ℝ) (h1 : D > d) (h2 : D - d = 9) (h3 : D / 2 - 5 > 0) :
  let TO := D / 2 - 5
  let OC := (D - d) / 2
  let CT := d / 2
  TO ^ 2 + OC ^ 2 = CT ^ 2 → d + D = 91 := by
sorry

end circles_diameter_sum_l3216_321614


namespace initial_speed_is_40_l3216_321670

/-- Represents a journey with increasing speed -/
structure Journey where
  totalDistance : ℝ
  totalTime : ℝ
  speedIncrease : ℝ
  intervalTime : ℝ

/-- Calculates the initial speed for a given journey -/
def calculateInitialSpeed (j : Journey) : ℝ :=
  sorry

/-- Theorem stating that for the given journey parameters, the initial speed is 40 km/h -/
theorem initial_speed_is_40 :
  let j : Journey := {
    totalDistance := 56,
    totalTime := 48 / 60, -- converting minutes to hours
    speedIncrease := 20,
    intervalTime := 12 / 60 -- converting minutes to hours
  }
  calculateInitialSpeed j = 40 := by
  sorry

end initial_speed_is_40_l3216_321670


namespace factorization_equality_l3216_321679

theorem factorization_equality (x y : ℝ) :
  -1/2 * x^3 + 1/8 * x * y^2 = -1/8 * x * (2*x + y) * (2*x - y) := by
  sorry

end factorization_equality_l3216_321679


namespace geometric_arithmetic_sequence_comparison_l3216_321693

theorem geometric_arithmetic_sequence_comparison 
  (a b : ℕ → ℝ) 
  (h_geom : ∀ n : ℕ, ∃ q : ℝ, a (n + 1) = a n * q) 
  (h_arith : ∀ n : ℕ, ∃ d : ℝ, b (n + 1) = b n + d)
  (h_pos : a 1 > 0)
  (h_eq1 : a 1 = b 1)
  (h_eq3 : a 3 = b 3)
  (h_neq : a 1 ≠ a 3) :
  a 5 > b 5 := by
  sorry

end geometric_arithmetic_sequence_comparison_l3216_321693


namespace cos_two_alpha_plus_beta_l3216_321676

theorem cos_two_alpha_plus_beta
  (α β : ℝ)
  (h1 : 3 * (Real.sin α)^2 + 2 * (Real.sin β)^2 = 1)
  (h2 : 3 * (Real.sin α + Real.cos α)^2 - 2 * (Real.sin β + Real.cos β)^2 = 1) :
  Real.cos (2 * (α + β)) = -1/3 :=
by sorry

end cos_two_alpha_plus_beta_l3216_321676


namespace binary_101101_equals_45_l3216_321607

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

theorem binary_101101_equals_45 :
  binary_to_decimal [true, false, true, true, false, true] = 45 := by
  sorry

end binary_101101_equals_45_l3216_321607


namespace cube_sum_integer_l3216_321695

theorem cube_sum_integer (a : ℝ) (ha : a ≠ 0) :
  ∃ k : ℤ, (a + 1/a : ℝ) = k → ∃ m : ℤ, (a^3 + 1/a^3 : ℝ) = m := by
  sorry

end cube_sum_integer_l3216_321695


namespace opposite_of_negative_2023_l3216_321653

theorem opposite_of_negative_2023 : -((-2023 : ℤ)) = 2023 := by
  sorry

end opposite_of_negative_2023_l3216_321653


namespace line_plane_intersection_l3216_321637

-- Define a structure for a 3D space
structure Space3D where
  Point : Type
  Line : Type
  Plane : Type
  IsParallel : Line → Plane → Prop
  HasCommonPoint : Line → Plane → Prop

-- State the theorem
theorem line_plane_intersection
  (S : Space3D)
  (a : S.Line)
  (α : S.Plane)
  (h : ¬S.IsParallel a α) :
  S.HasCommonPoint a α :=
sorry

end line_plane_intersection_l3216_321637


namespace complement_union_theorem_l3216_321667

universe u

def U : Set (Fin 4) := {1, 2, 3, 4}
def S : Set (Fin 4) := {1, 3}
def T : Set (Fin 4) := {4}

theorem complement_union_theorem : 
  (U \ S) ∪ T = {2, 4} := by sorry

end complement_union_theorem_l3216_321667


namespace simplify_expression_l3216_321690

theorem simplify_expression : ((- Real.sqrt 3) ^ 2) ^ (-1/2 : ℝ) = Real.sqrt 3 / 3 := by
  sorry

end simplify_expression_l3216_321690


namespace triangle_properties_l3216_321630

/-- Triangle ABC with vertices A(-1,4), B(-2,-1), and C(2,3) -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- The altitude from BC in triangle ABC -/
def altitude (t : Triangle) : ℝ × ℝ → Prop :=
  λ p => p.1 + p.2 - 3 = 0

/-- The area of triangle ABC -/
def area (t : Triangle) : ℝ := 8

theorem triangle_properties :
  let t : Triangle := { A := (-1, 4), B := (-2, -1), C := (2, 3) }
  (∀ p, altitude t p ↔ p.1 + p.2 - 3 = 0) ∧ area t = 8 := by
  sorry

end triangle_properties_l3216_321630


namespace carpet_area_calculation_l3216_321663

theorem carpet_area_calculation (rectangle_length rectangle_width triangle_base triangle_height : ℝ) 
  (h1 : rectangle_length = 12)
  (h2 : rectangle_width = 8)
  (h3 : triangle_base = 10)
  (h4 : triangle_height = 6) : 
  rectangle_length * rectangle_width + (triangle_base * triangle_height) / 2 = 126 := by
  sorry

end carpet_area_calculation_l3216_321663


namespace exists_real_less_than_one_exists_natural_in_real_exists_real_between_two_and_three_forall_int_exists_real_outside_interval_l3216_321655

-- 1. Prove that there exists a real number less than 1
theorem exists_real_less_than_one : ∃ x : ℝ, x < 1 := by sorry

-- 2. Prove that there exists a natural number that is also a real number
theorem exists_natural_in_real : ∃ x : ℕ, ∃ y : ℝ, x = y := by sorry

-- 3. Prove that there exists a real number greater than 2 and less than 3
theorem exists_real_between_two_and_three : ∃ x : ℝ, x > 2 ∧ x < 3 := by sorry

-- 4. Prove that for all integers n, there exists a real number x that is either less than n or greater than or equal to n + 1
theorem forall_int_exists_real_outside_interval : ∀ n : ℤ, ∃ x : ℝ, x < n ∨ x ≥ n + 1 := by sorry

end exists_real_less_than_one_exists_natural_in_real_exists_real_between_two_and_three_forall_int_exists_real_outside_interval_l3216_321655


namespace girls_to_boys_ratio_l3216_321652

/-- Given a class with four more girls than boys and 30 total students, 
    prove the ratio of girls to boys is 17/13 -/
theorem girls_to_boys_ratio (g b : ℕ) : 
  g = b + 4 → g + b = 30 → g / b = 17 / 13 := by
  sorry

end girls_to_boys_ratio_l3216_321652


namespace room_population_problem_l3216_321625

theorem room_population_problem (initial_men initial_women : ℕ) : 
  initial_men * 5 = initial_women * 4 →
  (initial_men + 2) = 14 →
  (2 * (initial_women - 3)) = 24 :=
by
  sorry

end room_population_problem_l3216_321625


namespace special_collection_returned_percentage_l3216_321692

/-- Calculates the percentage of returned books given initial count, final count, and loaned count. -/
def percentage_returned (initial : ℕ) (final : ℕ) (loaned : ℕ) : ℚ :=
  (1 - (initial - final : ℚ) / (loaned : ℚ)) * 100

/-- Theorem stating that the percentage of returned books is 65% given the problem conditions. -/
theorem special_collection_returned_percentage :
  percentage_returned 75 61 40 = 65 := by
  sorry

end special_collection_returned_percentage_l3216_321692


namespace intersection_of_A_and_B_l3216_321658

def A : Set ℝ := {x | -2 < x ∧ x ≤ 2}
def B : Set ℝ := {-2, -1, 0}

theorem intersection_of_A_and_B : A ∩ B = {-1, 0} := by
  sorry

end intersection_of_A_and_B_l3216_321658


namespace fencing_cost_is_1950_l3216_321622

/-- A rectangular plot with specific dimensions and fencing cost. -/
structure RectangularPlot where
  width : ℝ
  length : ℝ
  perimeter : ℝ
  fencing_rate : ℝ
  length_width_relation : length = width + 10
  perimeter_constraint : perimeter = 2 * (length + width)
  perimeter_value : perimeter = 300
  fencing_rate_value : fencing_rate = 6.5

/-- The cost of fencing the rectangular plot. -/
def fencing_cost (plot : RectangularPlot) : ℝ :=
  plot.perimeter * plot.fencing_rate

/-- Theorem stating the fencing cost for the given rectangular plot. -/
theorem fencing_cost_is_1950 (plot : RectangularPlot) : fencing_cost plot = 1950 := by
  sorry


end fencing_cost_is_1950_l3216_321622


namespace circle_intersection_theorem_l3216_321677

open Set
open Real

-- Define a type for points in the coordinate plane
structure Point where
  x : ℝ
  y : ℝ

-- Define a function to check if a point has integer coordinates
def isIntegerPoint (p : Point) : Prop :=
  ∃ (m n : ℤ), p.x = m ∧ p.y = n

-- Define a circle with center and radius
def Circle (center : Point) (radius : ℝ) : Set Point :=
  {p : Point | (p.x - center.x)^2 + (p.y - center.y)^2 ≤ radius^2}

-- Define the intersection of two circles
def circlesIntersect (c1 c2 : Set Point) : Prop :=
  ∃ (p : Point), p ∈ c1 ∧ p ∈ c2

-- State the theorem
theorem circle_intersection_theorem :
  ∀ (O : Point),
    ∃ (I : Point),
      isIntegerPoint I ∧
      circlesIntersect (Circle O 100) (Circle I (1/14)) :=
sorry

end circle_intersection_theorem_l3216_321677


namespace workers_per_team_lead_is_ten_l3216_321611

/-- Represents the hierarchical structure of a company -/
structure CompanyStructure where
  supervisors : ℕ
  workers : ℕ
  team_leads_per_supervisor : ℕ
  workers_per_team_lead : ℕ

/-- Calculates the number of workers per team lead given a company structure -/
def calculate_workers_per_team_lead (c : CompanyStructure) : ℕ :=
  c.workers / (c.supervisors * c.team_leads_per_supervisor)

/-- Theorem stating that for the given company structure, there are 10 workers per team lead -/
theorem workers_per_team_lead_is_ten :
  let c := CompanyStructure.mk 13 390 3 10
  calculate_workers_per_team_lead c = 10 := by
  sorry


end workers_per_team_lead_is_ten_l3216_321611


namespace no_three_digit_special_couples_l3216_321609

/-- Definition of a special couple for three-digit numbers -/
def is_special_couple (abc cba : ℕ) : Prop :=
  ∃ (a b c : ℕ),
    a < 10 ∧ b < 10 ∧ c < 10 ∧  -- Digits are single-digit natural numbers
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧     -- Digits are distinct
    abc = 100 * a + 10 * b + c ∧
    cba = 100 * c + 10 * b + a ∧
    a + b + c = 9               -- Sum of digits is 9

/-- Theorem: There are no special couples with three-digit numbers -/
theorem no_three_digit_special_couples :
  ¬ ∃ (abc cba : ℕ), is_special_couple abc cba :=
sorry

end no_three_digit_special_couples_l3216_321609


namespace curve_arc_length_l3216_321697

noncomputable def arcLength (t₁ t₂ : Real) : Real :=
  ∫ t in t₁..t₂, Real.sqrt ((12 * Real.cos t ^ 2 * Real.sin t) ^ 2 + (12 * Real.sin t ^ 2 * Real.cos t) ^ 2)

theorem curve_arc_length :
  arcLength (π / 6) (π / 4) = 3 / 2 := by
  sorry

end curve_arc_length_l3216_321697


namespace line_intersection_theorem_l3216_321627

/-- The line L in the xy-plane --/
def line_L (m : ℝ) (x y : ℝ) : Prop :=
  5 * y + (2 * m - 4) * x - 10 * m = 0

/-- The rectangle OABC --/
def rectangle : Set (ℝ × ℝ) :=
  {p | 0 ≤ p.1 ∧ p.1 ≤ 10 ∧ 0 ≤ p.2 ∧ p.2 ≤ 6}

/-- Point D on OA --/
def point_D (m : ℝ) : ℝ × ℝ := (0, 2 * m)

/-- Point E on BC --/
def point_E (m : ℝ) : ℝ × ℝ := (10, 8 - 2 * m)

/-- Area of quadrilateral ADEB --/
def area_ADEB (m : ℝ) : ℝ := 20

/-- Area of rectangle OABC --/
def area_OABC : ℝ := 60

/-- Parallel line that divides the rectangle into three equal areas --/
def parallel_line (m : ℝ) (x y : ℝ) : Prop :=
  y = ((4 - 2 * m) / 5) * x + (2 * m - 2)

theorem line_intersection_theorem (m : ℝ) :
  (1 ≤ m ∧ m ≤ 3) ∧
  (area_ADEB m = (1 / 3) * area_OABC) ∧
  (∀ x y, parallel_line m x y → 
    ∃ F G, F ∈ rectangle ∧ G ∈ rectangle ∧
    line_L m F.1 F.2 ∧ line_L m G.1 G.2 ∧
    area_ADEB m = area_OABC / 3) :=
sorry

end line_intersection_theorem_l3216_321627


namespace sin_sum_of_angles_l3216_321673

theorem sin_sum_of_angles (θ φ : ℝ) 
  (h1 : Complex.exp (θ * Complex.I) = (4/5 : ℂ) + (3/5 : ℂ) * Complex.I)
  (h2 : Complex.exp (φ * Complex.I) = -(5/13 : ℂ) + (12/13 : ℂ) * Complex.I) : 
  Real.sin (θ + φ) = 33/65 := by
sorry

end sin_sum_of_angles_l3216_321673


namespace sphere_area_equals_volume_l3216_321626

theorem sphere_area_equals_volume (r : ℝ) (h : r > 0) :
  4 * Real.pi * r^2 = (4/3) * Real.pi * r^3 → r = 3 := by
  sorry

end sphere_area_equals_volume_l3216_321626


namespace remainder_theorem_l3216_321654

def f (x : ℝ) : ℝ := 4*x^5 - 9*x^4 + 3*x^3 + 5*x^2 - x - 15

theorem remainder_theorem :
  f 4 = 2045 :=
by sorry

end remainder_theorem_l3216_321654


namespace nonagon_diagonals_l3216_321642

/-- The number of distinct diagonals in a convex nonagon -/
def num_diagonals_nonagon : ℕ := 27

/-- A convex nonagon has 27 distinct diagonals -/
theorem nonagon_diagonals : 
  num_diagonals_nonagon = 27 := by sorry

end nonagon_diagonals_l3216_321642


namespace range_of_m_l3216_321680

theorem range_of_m (x y m : ℝ) 
  (hx : x > 0) (hy : y > 0) 
  (h_eq : 2/x + 3/y = 1) 
  (h_ineq : ∀ (x y : ℝ), x > 0 → y > 0 → 2/x + 3/y = 1 → 3*x + 2*y > m^2 + 2*m) : 
  -6 < m ∧ m < 4 := by
sorry

end range_of_m_l3216_321680


namespace seven_balls_three_boxes_l3216_321620

/-- The number of ways to distribute n distinguishable balls into k indistinguishable boxes -/
def distributeBalls (n k : ℕ) : ℕ := sorry

/-- Theorem: There are 95 ways to distribute 7 distinguishable balls into 3 indistinguishable boxes -/
theorem seven_balls_three_boxes : distributeBalls 7 3 = 95 := by sorry

end seven_balls_three_boxes_l3216_321620


namespace stating_max_principals_l3216_321671

/-- Represents the duration of the period in years -/
def period_duration : ℕ := 10

/-- Represents the duration of a principal's term in years -/
def term_duration : ℕ := 4

/-- 
Theorem stating that the maximum number of principals 
that can serve during the given period is 3
-/
theorem max_principals :
  ∀ (principal_count : ℕ),
  (∀ (year : ℕ), year ≤ period_duration → 
    ∃ (principal : ℕ), principal ≤ principal_count ∧ 
    ∃ (start_year : ℕ), start_year ≤ period_duration ∧ 
    year ∈ Set.Icc start_year (start_year + term_duration - 1)) →
  principal_count ≤ 3 :=
sorry

end stating_max_principals_l3216_321671


namespace alannah_extra_books_l3216_321698

/-- The number of books each person has -/
structure BookCount where
  alannah : ℕ
  beatrix : ℕ
  queen : ℕ

/-- The conditions of the book distribution problem -/
def BookProblem (bc : BookCount) : Prop :=
  bc.alannah > bc.beatrix ∧
  bc.queen = bc.alannah + bc.alannah / 5 ∧
  bc.beatrix = 30 ∧
  bc.alannah + bc.beatrix + bc.queen = 140

/-- The theorem stating that Alannah has 20 more books than Beatrix -/
theorem alannah_extra_books (bc : BookCount) (h : BookProblem bc) : 
  bc.alannah = bc.beatrix + 20 := by
  sorry


end alannah_extra_books_l3216_321698


namespace quadratic_is_perfect_square_l3216_321604

theorem quadratic_is_perfect_square (a : ℝ) :
  (∃ r s : ℝ, ∀ x : ℝ, a * x^2 - 8 * x + 16 = (r * x + s)^2) →
  a = 1 := by
sorry

end quadratic_is_perfect_square_l3216_321604


namespace percentage_failed_hindi_l3216_321664

theorem percentage_failed_hindi (failed_english : Real) (failed_both : Real) (passed_both : Real) :
  failed_english = 35 →
  failed_both = 40 →
  passed_both = 80 →
  ∃ (failed_hindi : Real), failed_hindi = 25 := by
sorry

end percentage_failed_hindi_l3216_321664


namespace charging_pile_growth_l3216_321618

/-- Represents the growth of smart charging piles over two months -/
theorem charging_pile_growth 
  (initial_count : ℕ) 
  (final_count : ℕ) 
  (growth_rate : ℝ) 
  (h1 : initial_count = 301)
  (h2 : final_count = 500)
  : initial_count * (1 + growth_rate)^2 = final_count := by
  sorry

#check charging_pile_growth

end charging_pile_growth_l3216_321618


namespace domain_of_composed_function_inequality_proof_l3216_321696

-- Definition of the function f
def f : Set ℝ := Set.Icc (1/2) 2

-- Theorem 1: Domain of y = f(2^x)
theorem domain_of_composed_function :
  {x : ℝ | 2^x ∈ f} = Set.Icc (-1) 1 := by sorry

-- Theorem 2: Inequality proof
theorem inequality_proof (x y : ℝ) (h1 : -2 < x) (h2 : x < y) (h3 : y < 1) :
  -3 < x - y ∧ x - y < 0 := by sorry

end domain_of_composed_function_inequality_proof_l3216_321696


namespace playground_girls_count_l3216_321632

theorem playground_girls_count (total_children boys : ℕ) 
  (h1 : total_children = 117) 
  (h2 : boys = 40) : 
  total_children - boys = 77 := by
sorry

end playground_girls_count_l3216_321632


namespace quadratic_root_implies_quintic_root_l3216_321634

theorem quadratic_root_implies_quintic_root (r : ℝ) : 
  r^2 - r - 2 = 0 → r^5 - 11*r - 10 = 0 := by
  sorry

end quadratic_root_implies_quintic_root_l3216_321634


namespace quadratic_function_j_value_l3216_321685

theorem quadratic_function_j_value (a b c : ℤ) (j : ℤ) :
  let f := fun (x : ℤ) => a * x^2 + b * x + c
  (f 1 = 0) →
  (60 < f 7) →
  (f 7 < 70) →
  (80 < f 8) →
  (f 8 < 90) →
  (1000 * j < f 10) →
  (f 10 < 1000 * (j + 1)) →
  j = 0 := by
sorry

end quadratic_function_j_value_l3216_321685


namespace sqrt_squared_eq_self_sqrt_784_squared_l3216_321605

theorem sqrt_squared_eq_self (x : ℝ) (h : x ≥ 0) : (Real.sqrt x) ^ 2 = x := by sorry

theorem sqrt_784_squared : (Real.sqrt 784) ^ 2 = 784 := by sorry

end sqrt_squared_eq_self_sqrt_784_squared_l3216_321605


namespace matrix_identity_proof_l3216_321686

variables {n : Type*} [Fintype n] [DecidableEq n]

theorem matrix_identity_proof 
  (B : Matrix n n ℝ) 
  (h_inv : IsUnit B) 
  (h_eq : (B - 3 • 1) * (B - 5 • 1) = 0) : 
  B + 10 • B⁻¹ = 8 • 1 := by
  sorry

end matrix_identity_proof_l3216_321686


namespace hexagon_side_sum_l3216_321694

/-- Given a hexagon PQRSTU with the following properties:
  * The area of PQRSTU is 68
  * PQ = 10
  * QR = 7
  * TU = 6
  Prove that RS + ST = 3 -/
theorem hexagon_side_sum (PQRSTU : Set ℝ × ℝ) (area : ℝ) (PQ QR TU : ℝ) :
  area = 68 → PQ = 10 → QR = 7 → TU = 6 →
  ∃ (RS ST : ℝ), RS + ST = 3 := by
  sorry

#check hexagon_side_sum

end hexagon_side_sum_l3216_321694


namespace train_length_l3216_321641

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed : ℝ) (time : ℝ) :
  speed = 36 →  -- speed in km/hr
  time = 9 →    -- time in seconds
  speed * (time / 3600) = 90 / 1000 := by
  sorry

#check train_length

end train_length_l3216_321641


namespace treasure_count_conversion_l3216_321659

/-- Converts a number from base 7 to base 10 -/
def base7ToBase10 (n : Nat) : Nat :=
  let hundreds := n / 100
  let tens := (n / 10) % 10
  let ones := n % 10
  hundreds * 7^2 + tens * 7^1 + ones * 7^0

/-- The deep-sea creature's treasure count in base 7 -/
def treasureCountBase7 : Nat := 245

theorem treasure_count_conversion :
  base7ToBase10 treasureCountBase7 = 131 := by
  sorry

end treasure_count_conversion_l3216_321659


namespace circle_equation_and_line_slope_l3216_321684

/-- A circle passing through three points -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A line in the form mx + y - 1 = 0 -/
structure Line where
  m : ℝ

/-- The distance between two points -/
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

/-- Check if a point lies on a circle -/
def onCircle (c : Circle) (p : ℝ × ℝ) : Prop := sorry

/-- Check if a point lies on a line -/
def onLine (l : Line) (p : ℝ × ℝ) : Prop := sorry

/-- The intersection points of a circle and a line -/
def intersectionPoints (c : Circle) (l : Line) : Set (ℝ × ℝ) := sorry

theorem circle_equation_and_line_slope 
  (c : Circle) 
  (l : Line) 
  (h1 : onCircle c (0, -4))
  (h2 : onCircle c (2, 0))
  (h3 : onCircle c (3, -1))
  (h4 : ∃ (A B : ℝ × ℝ), A ∈ intersectionPoints c l ∧ B ∈ intersectionPoints c l ∧ distance A B = 4) :
  c.center = (1, -2) ∧ c.radius^2 = 5 ∧ l.m = 4/3 := by sorry

end circle_equation_and_line_slope_l3216_321684


namespace average_equation_solution_l3216_321631

theorem average_equation_solution (x : ℝ) : 
  ((x + 8) + (7 * x + 3) + (3 * x + 9)) / 3 = 5 * x - 10 → x = 12.5 := by
sorry

end average_equation_solution_l3216_321631


namespace tan_half_positive_in_second_quadrant_l3216_321638

theorem tan_half_positive_in_second_quadrant (θ : Real) : 
  (π/2 < θ ∧ θ < π) → 0 < Real.tan (θ/2) := by
  sorry

end tan_half_positive_in_second_quadrant_l3216_321638


namespace item_frequency_proof_l3216_321624

theorem item_frequency_proof (total : ℕ) (second_grade : ℕ) 
  (h1 : total = 400) (h2 : second_grade = 20) : 
  let first_grade := total - second_grade
  (first_grade : ℚ) / total = 95 / 100 ∧ 
  (second_grade : ℚ) / total = 5 / 100 := by
  sorry

end item_frequency_proof_l3216_321624


namespace victory_guarantee_l3216_321657

/-- Represents the state of the archery tournament -/
structure ArcheryTournament where
  totalShots : ℕ
  halfwayPoint : ℕ
  jessicaLead : ℕ
  bullseyeScore : ℕ
  minJessicaScore : ℕ

/-- Calculates the minimum number of bullseyes Jessica needs to guarantee victory -/
def minBullseyesForVictory (tournament : ArcheryTournament) : ℕ :=
  let remainingShots := tournament.totalShots - tournament.halfwayPoint
  let maxOpponentScore := tournament.bullseyeScore * remainingShots
  let jessicaNeededScore := maxOpponentScore - tournament.jessicaLead + 1
  (jessicaNeededScore + remainingShots * tournament.minJessicaScore - 1) / 
    (tournament.bullseyeScore - tournament.minJessicaScore) + 1

theorem victory_guarantee (tournament : ArcheryTournament) 
  (h1 : tournament.totalShots = 80)
  (h2 : tournament.halfwayPoint = 40)
  (h3 : tournament.jessicaLead = 30)
  (h4 : tournament.bullseyeScore = 10)
  (h5 : tournament.minJessicaScore = 2) :
  minBullseyesForVictory tournament = 37 := by
  sorry

#eval minBullseyesForVictory { 
  totalShots := 80, 
  halfwayPoint := 40, 
  jessicaLead := 30, 
  bullseyeScore := 10, 
  minJessicaScore := 2 
}

end victory_guarantee_l3216_321657


namespace arithmetic_sequence_a8_l3216_321648

/-- An arithmetic sequence is a sequence where the difference between
    any two consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_a8 (a : ℕ → ℝ) :
  is_arithmetic_sequence a → a 2 = 2 → a 14 = 18 → a 8 = 10 := by
  sorry

end arithmetic_sequence_a8_l3216_321648
