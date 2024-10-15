import Mathlib

namespace NUMINAMATH_CALUDE_grocery_shopping_remainder_l3109_310986

/-- Calculates the remaining amount after grocery shopping --/
def remaining_amount (initial_amount bread_cost candy_cost cereal_cost milk_cost : ℚ) : ℚ :=
  let initial_purchases := bread_cost + 2 * candy_cost + cereal_cost
  let after_initial := initial_amount - initial_purchases
  let fruit_cost := 0.2 * after_initial
  let after_fruit := after_initial - fruit_cost
  let after_milk := after_fruit - 2 * milk_cost
  let turkey_cost := 0.25 * after_milk
  after_milk - turkey_cost

/-- Theorem stating the remaining amount after grocery shopping --/
theorem grocery_shopping_remainder :
  remaining_amount 100 4 3 6 4.5 = 43.65 := by
  sorry

end NUMINAMATH_CALUDE_grocery_shopping_remainder_l3109_310986


namespace NUMINAMATH_CALUDE_number_subtraction_l3109_310923

theorem number_subtraction (x : ℤ) : x + 30 = 55 → x - 23 = 2 := by
  sorry

end NUMINAMATH_CALUDE_number_subtraction_l3109_310923


namespace NUMINAMATH_CALUDE_inlet_fill_rate_l3109_310930

/-- The rate at which the inlet pipe fills the tank, given the tank's capacity,
    leak emptying time, and combined emptying time with inlet open. -/
theorem inlet_fill_rate (capacity : ℝ) (leak_empty_time : ℝ) (combined_empty_time : ℝ) :
  capacity = 5760 →
  leak_empty_time = 6 →
  combined_empty_time = 8 →
  (capacity / leak_empty_time) - (capacity / combined_empty_time) = 240 := by
  sorry

end NUMINAMATH_CALUDE_inlet_fill_rate_l3109_310930


namespace NUMINAMATH_CALUDE_average_of_remaining_numbers_l3109_310958

theorem average_of_remaining_numbers
  (total_count : Nat)
  (subset_count : Nat)
  (total_average : ℝ)
  (subset_average : ℝ)
  (h_total_count : total_count = 15)
  (h_subset_count : subset_count = 9)
  (h_total_average : total_average = 30.5)
  (h_subset_average : subset_average = 17.75) :
  let remaining_count := total_count - subset_count
  let remaining_sum := total_count * total_average - subset_count * subset_average
  remaining_sum / remaining_count = 49.625 := by
sorry


end NUMINAMATH_CALUDE_average_of_remaining_numbers_l3109_310958


namespace NUMINAMATH_CALUDE_eighth_term_and_half_l3109_310998

def geometric_sequence (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * r ^ (n - 1)

theorem eighth_term_and_half (a : ℚ) (r : ℚ) :
  a = 12 → r = 1/2 →
  geometric_sequence a r 8 = 3/32 ∧
  (1/2 * geometric_sequence a r 8) = 3/64 := by
  sorry

end NUMINAMATH_CALUDE_eighth_term_and_half_l3109_310998


namespace NUMINAMATH_CALUDE_no_valid_numbers_l3109_310938

def is_valid_number (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000 ∧
  ∃ (a x : ℕ), n = 1000 * a + x ∧ 100 ≤ x ∧ x < 1000 ∧ 8 * x = n

theorem no_valid_numbers : ¬∃ (n : ℕ), is_valid_number n := by
  sorry

end NUMINAMATH_CALUDE_no_valid_numbers_l3109_310938


namespace NUMINAMATH_CALUDE_milk_tea_sales_l3109_310991

-- Define the relationship between cups of milk tea and total sales price
def sales_price (x : ℕ) : ℕ := 10 * x + 2

-- Theorem stating the conditions and the result to be proved
theorem milk_tea_sales :
  (sales_price 1 = 12) →
  (sales_price 2 = 22) →
  (∃ x : ℕ, sales_price x = 822) →
  (∃ x : ℕ, sales_price x = 822 ∧ x = 82) :=
by sorry

end NUMINAMATH_CALUDE_milk_tea_sales_l3109_310991


namespace NUMINAMATH_CALUDE_jenny_ate_65_chocolates_l3109_310904

/-- The number of chocolate squares Mike ate -/
def mike_chocolates : ℕ := 20

/-- The number of chocolate squares Jenny ate -/
def jenny_chocolates : ℕ := 3 * mike_chocolates + 5

theorem jenny_ate_65_chocolates : jenny_chocolates = 65 := by
  sorry

end NUMINAMATH_CALUDE_jenny_ate_65_chocolates_l3109_310904


namespace NUMINAMATH_CALUDE_inscribed_hexagon_area_l3109_310963

/-- The area of a regular hexagon inscribed in a circle with area 16π -/
theorem inscribed_hexagon_area : 
  ∀ (circle_area : ℝ) (hexagon_area : ℝ),
  circle_area = 16 * Real.pi →
  hexagon_area = (6 * Real.sqrt 3 * circle_area) / (2 * Real.pi) →
  hexagon_area = 24 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_inscribed_hexagon_area_l3109_310963


namespace NUMINAMATH_CALUDE_inequality_proof_l3109_310914

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x * y + 2 * y * z + 2 * z * x) / (x^2 + y^2 + z^2) ≤ (Real.sqrt 33 + 1) / 4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3109_310914


namespace NUMINAMATH_CALUDE_price_decrease_percentage_l3109_310970

theorem price_decrease_percentage (original_price new_price : ℚ) 
  (h1 : original_price = 1400)
  (h2 : new_price = 1064) :
  (original_price - new_price) / original_price * 100 = 24 := by
  sorry

end NUMINAMATH_CALUDE_price_decrease_percentage_l3109_310970


namespace NUMINAMATH_CALUDE_cube_difference_factorization_l3109_310947

theorem cube_difference_factorization (a b : ℝ) :
  a^3 - 8*b^3 = (a - 2*b) * (a^2 + 2*a*b + 4*b^2) := by
  sorry

end NUMINAMATH_CALUDE_cube_difference_factorization_l3109_310947


namespace NUMINAMATH_CALUDE_tangent_lines_through_M_line_intersects_circle_l3109_310989

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - 1)^2 + (y - 2)^2 = 4

-- Define point M
def point_M : ℝ × ℝ := (3, 1)

-- Define the line ax - y + 3 = 0
def line (a x y : ℝ) : Prop := a * x - y + 3 = 0

-- Theorem for part (I)
theorem tangent_lines_through_M :
  ∃ (k : ℝ), 
    (∀ x y : ℝ, (x = 3 ∨ 3 * x - 4 * y - 5 = 0) → 
      (circle_C x y ∧ (x = point_M.1 ∧ y = point_M.2 ∨ 
       (y - point_M.2) = k * (x - point_M.1)))) :=
sorry

-- Theorem for part (II)
theorem line_intersects_circle :
  ∀ a : ℝ, ∃ x y : ℝ, line a x y ∧ circle_C x y :=
sorry

end NUMINAMATH_CALUDE_tangent_lines_through_M_line_intersects_circle_l3109_310989


namespace NUMINAMATH_CALUDE_scale_and_rotate_complex_l3109_310929

/-- Represents a complex number rotation by 270° clockwise -/
def rotate270Clockwise (z : ℂ) : ℂ := Complex.I * z

/-- Proves that scaling -8 - 4i by 2 and then rotating 270° clockwise results in 8 - 16i -/
theorem scale_and_rotate_complex : 
  let z : ℂ := -8 - 4 * Complex.I
  let scaled : ℂ := 2 * z
  rotate270Clockwise scaled = 8 - 16 * Complex.I := by sorry

end NUMINAMATH_CALUDE_scale_and_rotate_complex_l3109_310929


namespace NUMINAMATH_CALUDE_dogs_neither_long_furred_nor_brown_l3109_310956

/-- Prove that the number of dogs that are neither long-furred nor brown is 8 -/
theorem dogs_neither_long_furred_nor_brown
  (total_dogs : ℕ)
  (long_furred_dogs : ℕ)
  (brown_dogs : ℕ)
  (long_furred_brown_dogs : ℕ)
  (h1 : total_dogs = 45)
  (h2 : long_furred_dogs = 26)
  (h3 : brown_dogs = 22)
  (h4 : long_furred_brown_dogs = 11) :
  total_dogs - (long_furred_dogs + brown_dogs - long_furred_brown_dogs) = 8 := by
  sorry

#check dogs_neither_long_furred_nor_brown

end NUMINAMATH_CALUDE_dogs_neither_long_furred_nor_brown_l3109_310956


namespace NUMINAMATH_CALUDE_log_equality_implies_golden_ratio_l3109_310941

theorem log_equality_implies_golden_ratio (a b : ℝ) :
  a > 0 ∧ b > 0 →
  Real.log a / Real.log 8 = Real.log b / Real.log 18 ∧
  Real.log a / Real.log 8 = Real.log (a + b) / Real.log 32 →
  b / a = (1 + Real.sqrt 5) / 2 := by
sorry

end NUMINAMATH_CALUDE_log_equality_implies_golden_ratio_l3109_310941


namespace NUMINAMATH_CALUDE_abs_x_minus_one_equals_one_minus_x_implies_x_leq_one_l3109_310909

theorem abs_x_minus_one_equals_one_minus_x_implies_x_leq_one (x : ℝ) : 
  |x - 1| = 1 - x → x ≤ 1 := by
sorry

end NUMINAMATH_CALUDE_abs_x_minus_one_equals_one_minus_x_implies_x_leq_one_l3109_310909


namespace NUMINAMATH_CALUDE_min_four_dollar_frisbees_l3109_310903

/-- Given a total of 64 frisbees sold at either $3 or $4 each, with total receipts of $196,
    the minimum number of $4 frisbees sold is 4. -/
theorem min_four_dollar_frisbees :
  ∀ (x y : ℕ),
    x + y = 64 →
    3 * x + 4 * y = 196 →
    y ≥ 4 ∧ ∃ (z : ℕ), z + 4 = 64 ∧ 3 * z + 4 * 4 = 196 :=
by sorry

end NUMINAMATH_CALUDE_min_four_dollar_frisbees_l3109_310903


namespace NUMINAMATH_CALUDE_fast_food_cost_correct_l3109_310992

/-- The cost of fast food given the number of servings of each type -/
def fast_food_cost (a b : ℕ) : ℕ := 30 * a + 20 * b

/-- Theorem stating that the cost of fast food is calculated correctly -/
theorem fast_food_cost_correct (a b : ℕ) : 
  fast_food_cost a b = 30 * a + 20 * b := by
  sorry

end NUMINAMATH_CALUDE_fast_food_cost_correct_l3109_310992


namespace NUMINAMATH_CALUDE_shaded_region_area_l3109_310953

/-- Given a shaded region consisting of congruent squares, proves that the total area is 40 cm² --/
theorem shaded_region_area (n : ℕ) (d : ℝ) (A : ℝ) :
  n = 20 →  -- Total number of congruent squares
  d = 8 →   -- Diagonal of the square formed by 16 smaller squares
  A = d^2 / 2 →  -- Area of the square formed by 16 smaller squares
  A / 16 * n = 40 :=
by sorry

end NUMINAMATH_CALUDE_shaded_region_area_l3109_310953


namespace NUMINAMATH_CALUDE_fraction_decomposition_l3109_310916

-- Define the polynomials
def p (x : ℝ) : ℝ := 6 * x - 15
def q (x : ℝ) : ℝ := 3 * x^3 - 2 * x^2 - 5 * x + 6
def r (x : ℝ) : ℝ := x - 1
def s (x : ℝ) : ℝ := 3 * x^2 - x - 6

-- Define the equality condition
def equality_condition (A B : ℝ → ℝ) : Prop :=
  ∀ x, x ≠ 1 ∧ s x ≠ 0 → p x / q x = A x / r x + B x / s x

-- Theorem statement
theorem fraction_decomposition :
  ∀ A B, equality_condition A B →
    (∀ x, A x = 0) ∧ (∀ x, B x = 6 * x - 15) :=
sorry

end NUMINAMATH_CALUDE_fraction_decomposition_l3109_310916


namespace NUMINAMATH_CALUDE_parallel_perpendicular_lines_l3109_310985

/-- Given a point P and a line l, prove the equations of parallel and perpendicular lines through P --/
theorem parallel_perpendicular_lines 
  (P : ℝ × ℝ) 
  (l : ℝ → ℝ → Prop) 
  (hl : l = fun x y => 3 * x - y - 7 = 0) 
  (hP : P = (2, 1)) :
  let parallel_line := fun x y => 3 * x - y - 5 = 0
  let perpendicular_line := fun x y => x - 3 * y + 1 = 0
  (∀ x y, parallel_line x y ↔ (3 * x - y = 3 * P.1 - P.2)) ∧ 
  (parallel_line P.1 P.2) ∧
  (∀ x y, perpendicular_line x y ↔ (x - 3 * y = P.1 - 3 * P.2)) ∧ 
  (perpendicular_line P.1 P.2) ∧
  (∀ x₁ y₁ x₂ y₂, l x₁ y₁ → l x₂ y₂ → x₁ ≠ x₂ → (y₁ - y₂) / (x₁ - x₂) = 3) ∧
  (∀ x₁ y₁ x₂ y₂, perpendicular_line x₁ y₁ → perpendicular_line x₂ y₂ → x₁ ≠ x₂ → 
    (y₁ - y₂) / (x₁ - x₂) = -1/3) := by
  sorry

end NUMINAMATH_CALUDE_parallel_perpendicular_lines_l3109_310985


namespace NUMINAMATH_CALUDE_parallelogram_area_25_15_l3109_310974

/-- The area of a parallelogram with given base and height -/
def parallelogram_area (base height : ℝ) : ℝ := base * height

/-- Theorem: The area of a parallelogram with base 25 cm and height 15 cm is 375 cm² -/
theorem parallelogram_area_25_15 :
  parallelogram_area 25 15 = 375 := by sorry

end NUMINAMATH_CALUDE_parallelogram_area_25_15_l3109_310974


namespace NUMINAMATH_CALUDE_greg_lunch_payment_l3109_310944

/-- Calculates the total amount paid for a meal including tax and tip -/
def total_amount_paid (cost : ℝ) (tax_rate : ℝ) (tip_rate : ℝ) : ℝ :=
  cost + (cost * tax_rate) + (cost * tip_rate)

/-- Theorem stating that Greg paid $110 for his lunch -/
theorem greg_lunch_payment :
  let cost : ℝ := 100
  let tax_rate : ℝ := 0.04
  let tip_rate : ℝ := 0.06
  total_amount_paid cost tax_rate tip_rate = 110 := by
  sorry

end NUMINAMATH_CALUDE_greg_lunch_payment_l3109_310944


namespace NUMINAMATH_CALUDE_triangle_properties_l3109_310976

-- Define the triangle ABC
structure Triangle where
  A : ℝ  -- Angle A
  B : ℝ  -- Angle B
  C : ℝ  -- Angle C
  a : ℝ  -- Side opposite to angle A
  b : ℝ  -- Side opposite to angle B
  c : ℝ  -- Side opposite to angle C

-- Define the theorem
theorem triangle_properties (t : Triangle) 
  (h1 : t.a * Real.sin t.B + t.b * Real.cos t.A = 0) 
  (h2 : 0 < t.A ∧ t.A < Real.pi) 
  (h3 : 0 < t.B ∧ t.B < Real.pi) 
  (h4 : 0 < t.C ∧ t.C < Real.pi) 
  (h5 : t.A + t.B + t.C = Real.pi) :
  t.A = 3 * Real.pi / 4 ∧ 
  (t.a = 2 * Real.sqrt 5 → t.b = 2 → 
    1/2 * t.b * t.c * Real.sin t.A = 2) := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l3109_310976


namespace NUMINAMATH_CALUDE_parallel_line_equation_l3109_310980

/-- A line in the plane is represented by its slope and a point it passes through. -/
structure Line where
  slope : ℝ
  point : ℝ × ℝ

/-- Check if a point lies on a line given by an equation ax + by + c = 0 -/
def pointOnLine (a b c : ℝ) (x y : ℝ) : Prop :=
  a * x + b * y + c = 0

/-- Two lines are parallel if they have the same slope -/
def parallel (l₁ l₂ : Line) : Prop :=
  l₁.slope = l₂.slope

theorem parallel_line_equation (p : ℝ × ℝ) :
  let l₁ : Line := { slope := 2, point := (0, 0) }  -- y = 2x
  let l₂ : Line := { slope := 2, point := p }       -- parallel line through p
  parallel l₁ l₂ →
  p = (1, -2) →
  pointOnLine 2 (-1) (-4) p.1 p.2 :=
by
  sorry

#check parallel_line_equation

end NUMINAMATH_CALUDE_parallel_line_equation_l3109_310980


namespace NUMINAMATH_CALUDE_arithmetic_expression_evaluation_l3109_310937

theorem arithmetic_expression_evaluation : 2 + 3 * 4^2 - 5 + 6 = 51 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_expression_evaluation_l3109_310937


namespace NUMINAMATH_CALUDE_inequality_proof_l3109_310925

theorem inequality_proof (x y z : ℝ) : x^4 + y^4 + z^2 + 1 ≥ 2*x*(x*y^2 - x + z + 1) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3109_310925


namespace NUMINAMATH_CALUDE_det_3_4_1_2_l3109_310960

-- Define the determinant function for a 2x2 matrix
def det2x2 (a b c d : ℝ) : ℝ := a * d - b * c

-- Theorem statement
theorem det_3_4_1_2 : det2x2 3 4 1 2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_det_3_4_1_2_l3109_310960


namespace NUMINAMATH_CALUDE_fruit_shop_costs_and_profit_l3109_310996

/-- Represents the fruit shop's purchases and sales --/
structure FruitShop where
  first_purchase_cost : ℝ
  first_purchase_price : ℝ
  second_purchase_cost : ℝ
  second_purchase_quantity_increase : ℝ
  second_sale_price : ℝ
  second_sale_quantity : ℝ
  second_sale_discount : ℝ

/-- Calculates the cost per kg and profit for the fruit shop --/
def calculate_costs_and_profit (shop : FruitShop) : ℝ × ℝ × ℝ := sorry

/-- Theorem stating the correct cost per kg and profit given the shop's conditions --/
theorem fruit_shop_costs_and_profit (shop : FruitShop) 
  (h1 : shop.first_purchase_cost = 1200)
  (h2 : shop.first_purchase_price = 8)
  (h3 : shop.second_purchase_cost = 1452)
  (h4 : shop.second_purchase_quantity_increase = 20)
  (h5 : shop.second_sale_price = 9)
  (h6 : shop.second_sale_quantity = 100)
  (h7 : shop.second_sale_discount = 0.5) :
  let (first_cost, second_cost, profit) := calculate_costs_and_profit shop
  first_cost = 6 ∧ second_cost = 6.6 ∧ profit = 388 := by sorry

end NUMINAMATH_CALUDE_fruit_shop_costs_and_profit_l3109_310996


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_from_parabola_focus_l3109_310926

/-- Given a parabola and a hyperbola with shared focus, prove the equations of the hyperbola's asymptotes -/
theorem hyperbola_asymptotes_from_parabola_focus 
  (parabola : ℝ → ℝ → Prop) 
  (hyperbola : ℝ → ℝ → Prop) 
  (b : ℝ) :
  (∀ x y, parabola x y ↔ y^2 = 16*x) →
  (∀ x y, hyperbola x y ↔ x^2/12 - y^2/b^2 = 1) →
  (∃ x₀, x₀ = 4 ∧ parabola x₀ 0 ∧ ∀ y, hyperbola x₀ y → y = 0) →
  (∀ x y, hyperbola x y → y = (Real.sqrt 3 / 3) * x ∨ y = -(Real.sqrt 3 / 3) * x) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_from_parabola_focus_l3109_310926


namespace NUMINAMATH_CALUDE_complement_of_M_in_U_l3109_310982

def U : Set Int := {1, -2, 3, -4, 5, -6}
def M : Set Int := {1, -2, 3, -4}

theorem complement_of_M_in_U : Mᶜ = {5, -6} := by sorry

end NUMINAMATH_CALUDE_complement_of_M_in_U_l3109_310982


namespace NUMINAMATH_CALUDE_count_representations_l3109_310932

/-- The number of ways to represent 5040 in the given form -/
def M : ℕ :=
  (Finset.range 100).sum (fun b₃ =>
    (Finset.range 100).sum (fun b₂ =>
      (Finset.range 100).sum (fun b₁ =>
        (Finset.range 100).sum (fun b₀ =>
          if b₃ * 10^3 + b₂ * 10^2 + b₁ * 10 + b₀ = 5040 then 1 else 0))))

theorem count_representations : M = 504 := by
  sorry

end NUMINAMATH_CALUDE_count_representations_l3109_310932


namespace NUMINAMATH_CALUDE_sum_of_prime_factors_27000001_l3109_310999

theorem sum_of_prime_factors_27000001 :
  ∃ (p₁ p₂ p₃ p₄ : Nat),
    Nat.Prime p₁ ∧ Nat.Prime p₂ ∧ Nat.Prime p₃ ∧ Nat.Prime p₄ ∧
    p₁ * p₂ * p₃ * p₄ = 27000001 ∧
    p₁ + p₂ + p₃ + p₄ = 652 :=
by
  sorry

#check sum_of_prime_factors_27000001

end NUMINAMATH_CALUDE_sum_of_prime_factors_27000001_l3109_310999


namespace NUMINAMATH_CALUDE_smallest_six_digit_divisible_l3109_310969

def is_divisible_by (n m : ℕ) : Prop := m ∣ n

theorem smallest_six_digit_divisible : 
  ∀ n : ℕ, 
    100000 ≤ n → 
    n < 1000000 → 
    (is_divisible_by n 25 ∧ 
     is_divisible_by n 35 ∧ 
     is_divisible_by n 45 ∧ 
     is_divisible_by n 15) → 
    n ≥ 100800 :=
sorry

end NUMINAMATH_CALUDE_smallest_six_digit_divisible_l3109_310969


namespace NUMINAMATH_CALUDE_phyllis_garden_problem_l3109_310921

/-- The number of plants in Phyllis's first garden -/
def plants_in_first_garden : ℕ := 20

/-- The number of plants in Phyllis's second garden -/
def plants_in_second_garden : ℕ := 15

/-- The fraction of tomato plants in the first garden -/
def tomato_fraction_first : ℚ := 1/10

/-- The fraction of tomato plants in the second garden -/
def tomato_fraction_second : ℚ := 1/3

/-- The fraction of tomato plants in both gardens combined -/
def total_tomato_fraction : ℚ := 1/5

theorem phyllis_garden_problem :
  (plants_in_first_garden : ℚ) * tomato_fraction_first +
  (plants_in_second_garden : ℚ) * tomato_fraction_second =
  ((plants_in_first_garden + plants_in_second_garden) : ℚ) * total_tomato_fraction :=
by sorry

end NUMINAMATH_CALUDE_phyllis_garden_problem_l3109_310921


namespace NUMINAMATH_CALUDE_double_papers_double_time_l3109_310940

/-- Represents the time taken to check exam papers under different conditions -/
def exam_check_time (men : ℕ) (days : ℕ) (hours_per_day : ℕ) (papers : ℕ) : ℕ :=
  men * days * hours_per_day

/-- Theorem stating the relationship between different exam checking scenarios -/
theorem double_papers_double_time (men₁ days₁ hours₁ men₂ days₂ papers₁ : ℕ) :
  exam_check_time men₁ days₁ hours₁ papers₁ = 160 →
  men₁ = 4 →
  days₁ = 8 →
  hours₁ = 5 →
  men₂ = 2 →
  days₂ = 20 →
  exam_check_time men₂ days₂ 8 (2 * papers₁) = 320 := by
  sorry

#check double_papers_double_time

end NUMINAMATH_CALUDE_double_papers_double_time_l3109_310940


namespace NUMINAMATH_CALUDE_method_a_cheaper_for_18_hours_l3109_310957

/-- Calculates the cost of internet usage for Method A (Pay-per-use) -/
def costMethodA (hours : ℝ) : ℝ := 3 * hours + 1.2 * hours

/-- Calculates the cost of internet usage for Method B (Monthly subscription) -/
def costMethodB (hours : ℝ) : ℝ := 60 + 1.2 * hours

/-- Theorem stating that Method A is cheaper than Method B for 18 hours of usage -/
theorem method_a_cheaper_for_18_hours :
  costMethodA 18 < costMethodB 18 :=
sorry

end NUMINAMATH_CALUDE_method_a_cheaper_for_18_hours_l3109_310957


namespace NUMINAMATH_CALUDE_percentage_good_fruits_l3109_310933

/-- Calculates the percentage of fruits in good condition given the number of oranges and bananas and their respective rotten percentages. -/
theorem percentage_good_fruits (oranges bananas : ℕ) (rotten_oranges_percent rotten_bananas_percent : ℚ) :
  oranges = 600 →
  bananas = 400 →
  rotten_oranges_percent = 15 / 100 →
  rotten_bananas_percent = 3 / 100 →
  (((oranges + bananas : ℚ) - (oranges * rotten_oranges_percent + bananas * rotten_bananas_percent)) / (oranges + bananas) * 100 : ℚ) = 89.8 := by
  sorry

end NUMINAMATH_CALUDE_percentage_good_fruits_l3109_310933


namespace NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l3109_310949

theorem quadratic_two_distinct_roots :
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ x₁^2 - 3*x₁ + 0 = 0 ∧ x₂^2 - 3*x₂ + 0 = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l3109_310949


namespace NUMINAMATH_CALUDE_base_eight_satisfies_equation_unique_base_satisfies_equation_l3109_310911

/-- Given a base b, converts a number in base b to decimal --/
def toDecimal (digits : List Nat) (b : Nat) : Nat :=
  digits.foldl (fun acc d => acc * b + d) 0

/-- Checks if the equation 245_b + 132_b = 400_b holds for a given base b --/
def equationHolds (b : Nat) : Prop :=
  toDecimal [2, 4, 5] b + toDecimal [1, 3, 2] b = toDecimal [4, 0, 0] b

theorem base_eight_satisfies_equation :
  equationHolds 8 := by sorry

theorem unique_base_satisfies_equation :
  ∀ b : Nat, b > 1 → equationHolds b → b = 8 := by sorry

end NUMINAMATH_CALUDE_base_eight_satisfies_equation_unique_base_satisfies_equation_l3109_310911


namespace NUMINAMATH_CALUDE_log_equality_l3109_310971

theorem log_equality (x k : ℝ) (h1 : Real.log 3 / Real.log 8 = x) (h2 : Real.log 81 / Real.log 2 = k * x) : k = 12 := by
  sorry

end NUMINAMATH_CALUDE_log_equality_l3109_310971


namespace NUMINAMATH_CALUDE_heating_rate_at_10_seconds_l3109_310908

-- Define the temperature function
def temperature (t : ℝ) : ℝ := 0.2 * t^2

-- Define the rate of heating function (derivative of temperature)
def rateOfHeating (t : ℝ) : ℝ := 0.4 * t

-- Theorem statement
theorem heating_rate_at_10_seconds :
  rateOfHeating 10 = 4 := by sorry

end NUMINAMATH_CALUDE_heating_rate_at_10_seconds_l3109_310908


namespace NUMINAMATH_CALUDE_tenth_term_of_specific_sequence_l3109_310977

/-- An arithmetic sequence is defined by its first term and common difference -/
structure ArithmeticSequence where
  a₁ : ℚ  -- First term
  d : ℚ   -- Common difference

/-- Get the nth term of an arithmetic sequence -/
def ArithmeticSequence.nthTerm (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  seq.a₁ + (n - 1 : ℚ) * seq.d

theorem tenth_term_of_specific_sequence :
  ∃ (seq : ArithmeticSequence),
    seq.nthTerm 1 = 5/6 ∧
    seq.nthTerm 16 = 7/8 ∧
    seq.nthTerm 10 = 103/120 := by
  sorry

end NUMINAMATH_CALUDE_tenth_term_of_specific_sequence_l3109_310977


namespace NUMINAMATH_CALUDE_sum_of_coefficients_is_zero_l3109_310997

/-- Given two linear functions f and g, prove that A + B = 0 under certain conditions -/
theorem sum_of_coefficients_is_zero
  (A B : ℝ)
  (f g : ℝ → ℝ)
  (h₁ : ∀ x, f x = A * x + B)
  (h₂ : ∀ x, g x = B * x + A)
  (h₃ : A ≠ B)
  (h₄ : ∀ x, f (g x) - g (f x) = B - A) :
  A + B = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_is_zero_l3109_310997


namespace NUMINAMATH_CALUDE_perfect_squares_among_options_l3109_310952

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m * m

def option_a : ℕ := 3^3 * 4^4 * 5^5
def option_b : ℕ := 3^4 * 4^5 * 5^6
def option_c : ℕ := 3^6 * 4^4 * 5^6
def option_d : ℕ := 3^5 * 4^6 * 5^5
def option_e : ℕ := 3^6 * 4^6 * 5^4

theorem perfect_squares_among_options :
  (¬ is_perfect_square option_a) ∧
  (is_perfect_square option_b) ∧
  (is_perfect_square option_c) ∧
  (¬ is_perfect_square option_d) ∧
  (is_perfect_square option_e) := by
  sorry

end NUMINAMATH_CALUDE_perfect_squares_among_options_l3109_310952


namespace NUMINAMATH_CALUDE_blue_flower_percentage_l3109_310990

/-- Given a total of 10 flowers, with 4 red and 2 white flowers,
    prove that 40% of the flowers are blue. -/
theorem blue_flower_percentage
  (total : ℕ)
  (red : ℕ)
  (white : ℕ)
  (h_total : total = 10)
  (h_red : red = 4)
  (h_white : white = 2) :
  (total - red - white : ℚ) / total * 100 = 40 := by
  sorry

#check blue_flower_percentage

end NUMINAMATH_CALUDE_blue_flower_percentage_l3109_310990


namespace NUMINAMATH_CALUDE_polynomial_evaluation_l3109_310951

theorem polynomial_evaluation :
  ∃ x : ℝ, x > 0 ∧ x^2 - 2*x - 15 = 0 ∧ x^3 - 2*x^2 - 8*x + 16 = 51 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_evaluation_l3109_310951


namespace NUMINAMATH_CALUDE_total_ways_is_eight_l3109_310915

/-- The number of course options available --/
def num_courses : Nat := 2

/-- The number of students choosing courses --/
def num_students : Nat := 3

/-- Calculates the total number of ways students can choose courses --/
def total_ways : Nat := num_courses ^ num_students

/-- Theorem stating that the total number of ways to choose courses is 8 --/
theorem total_ways_is_eight : total_ways = 8 := by
  sorry

end NUMINAMATH_CALUDE_total_ways_is_eight_l3109_310915


namespace NUMINAMATH_CALUDE_bird_families_difference_l3109_310972

theorem bird_families_difference (total : ℕ) (flew_away : ℕ) 
  (h1 : total = 87) (h2 : flew_away = 7) : 
  total - flew_away - flew_away = 73 := by
  sorry

end NUMINAMATH_CALUDE_bird_families_difference_l3109_310972


namespace NUMINAMATH_CALUDE_differentials_of_z_l3109_310968

noncomputable section

variables (x y : ℝ) (dx dy : ℝ)

def z : ℝ := x^5 * y^3

def dz : ℝ := 5 * x^4 * y^3 * dx + 3 * x^5 * y^2 * dy

def d2z : ℝ := 20 * x^3 * y^3 * dx^2 + 30 * x^4 * y^2 * dx * dy + 6 * x^5 * y * dy^2

def d3z : ℝ := 60 * x^2 * y^3 * dx^3 + 180 * x^3 * y^2 * dx^2 * dy + 90 * x^4 * y * dx * dy^2 + 6 * x^5 * dy^3

theorem differentials_of_z :
  (dz x y dx dy = 5 * x^4 * y^3 * dx + 3 * x^5 * y^2 * dy) ∧
  (d2z x y dx dy = 20 * x^3 * y^3 * dx^2 + 30 * x^4 * y^2 * dx * dy + 6 * x^5 * y * dy^2) ∧
  (d3z x y dx dy = 60 * x^2 * y^3 * dx^3 + 180 * x^3 * y^2 * dx^2 * dy + 90 * x^4 * y * dx * dy^2 + 6 * x^5 * dy^3) :=
by sorry

end NUMINAMATH_CALUDE_differentials_of_z_l3109_310968


namespace NUMINAMATH_CALUDE_opposite_of_negative_three_l3109_310978

theorem opposite_of_negative_three : -(- 3) = 3 := by sorry

end NUMINAMATH_CALUDE_opposite_of_negative_three_l3109_310978


namespace NUMINAMATH_CALUDE_race_distance_l3109_310950

/-- The race problem -/
theorem race_distance (time_A time_B : ℝ) (lead : ℝ) (distance : ℝ) : 
  time_A = 36 →
  time_B = 45 →
  lead = 20 →
  (distance / time_A) * time_B = distance + lead →
  distance = 80 := by
sorry

end NUMINAMATH_CALUDE_race_distance_l3109_310950


namespace NUMINAMATH_CALUDE_roof_area_l3109_310920

theorem roof_area (width length : ℝ) (h1 : length = 5 * width) (h2 : length - width = 48) :
  width * length = 720 := by
  sorry

end NUMINAMATH_CALUDE_roof_area_l3109_310920


namespace NUMINAMATH_CALUDE_choose_four_from_fifteen_l3109_310942

theorem choose_four_from_fifteen (n : ℕ) (k : ℕ) : n = 15 ∧ k = 4 → Nat.choose n k = 1365 := by
  sorry

end NUMINAMATH_CALUDE_choose_four_from_fifteen_l3109_310942


namespace NUMINAMATH_CALUDE_sphere_radius_ratio_l3109_310907

theorem sphere_radius_ratio (v_large v_small : ℝ) (h1 : v_large = 324 * Real.pi) (h2 : v_small = 0.25 * v_large) :
  (v_small / v_large) ^ (1/3 : ℝ) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_sphere_radius_ratio_l3109_310907


namespace NUMINAMATH_CALUDE_sin_double_angle_for_specific_tan_l3109_310924

theorem sin_double_angle_for_specific_tan (α : Real) (h : Real.tan α = -1/3) :
  Real.sin (2 * α) = -3/5 := by sorry

end NUMINAMATH_CALUDE_sin_double_angle_for_specific_tan_l3109_310924


namespace NUMINAMATH_CALUDE_exact_three_wins_probability_l3109_310902

/-- The probability of winning a prize in a single draw -/
def p : ℚ := 2/5

/-- The number of participants (trials) -/
def n : ℕ := 4

/-- The number of desired successes -/
def k : ℕ := 3

/-- The probability of exactly k successes in n independent trials 
    with probability p of success in each trial -/
def binomial_probability (n k : ℕ) (p : ℚ) : ℚ :=
  (n.choose k) * p^k * (1-p)^(n-k)

theorem exact_three_wins_probability :
  binomial_probability n k p = 96/625 := by
  sorry

end NUMINAMATH_CALUDE_exact_three_wins_probability_l3109_310902


namespace NUMINAMATH_CALUDE_reflection_of_P_across_y_axis_l3109_310966

/-- Given a point P in the Cartesian coordinate system, this function returns its reflection across the y-axis. -/
def reflect_across_y_axis (P : ℝ × ℝ) : ℝ × ℝ :=
  (-P.1, P.2)

/-- The original point P in the Cartesian coordinate system. -/
def P : ℝ × ℝ := (2, 1)

/-- Theorem: The coordinates of P(2,1) with respect to the y-axis are (-2,1). -/
theorem reflection_of_P_across_y_axis :
  reflect_across_y_axis P = (-2, 1) := by sorry

end NUMINAMATH_CALUDE_reflection_of_P_across_y_axis_l3109_310966


namespace NUMINAMATH_CALUDE_max_value_of_f_on_interval_l3109_310948

def f (x : ℝ) : ℝ := 2 * x^3 - 3 * x^2 - 12 * x + 5

theorem max_value_of_f_on_interval :
  ∃ (x : ℝ), x ∈ Set.Icc 0 3 ∧ f x = 5 ∧ ∀ y ∈ Set.Icc 0 3, f y ≤ f x :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_f_on_interval_l3109_310948


namespace NUMINAMATH_CALUDE_custom_mult_equation_solutions_l3109_310965

/-- Custom multiplication operation for real numbers -/
def custom_mult (a b : ℝ) : ℝ := a * (a + b) + b

/-- Theorem stating the solutions of the equation -/
theorem custom_mult_equation_solutions :
  ∃ (a : ℝ), custom_mult a 2.5 = 28.5 ∧ (a = 4 ∨ a = -13/2) := by
  sorry

end NUMINAMATH_CALUDE_custom_mult_equation_solutions_l3109_310965


namespace NUMINAMATH_CALUDE_p_iff_m_gt_2_p_xor_q_iff_m_range_l3109_310988

/-- Proposition p: The equation x^2 + mx + 1 = 0 has two distinct negative real roots -/
def p (m : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ < 0 ∧ x₂ < 0 ∧ x₁ ≠ x₂ ∧ 
  x₁^2 + m*x₁ + 1 = 0 ∧ x₂^2 + m*x₂ + 1 = 0

/-- Proposition q: The equation 4x^2 + 4(m-2)x + 1 = 0 has no real roots -/
def q (m : ℝ) : Prop :=
  ∀ x : ℝ, 4*x^2 + 4*(m-2)*x + 1 ≠ 0

theorem p_iff_m_gt_2 (m : ℝ) : p m ↔ m > 2 :=
sorry

theorem p_xor_q_iff_m_range (m : ℝ) : (p m ∨ q m) ∧ ¬(p m ∧ q m) ↔ m ≥ 3 ∨ (1 < m ∧ m ≤ 2) :=
sorry

end NUMINAMATH_CALUDE_p_iff_m_gt_2_p_xor_q_iff_m_range_l3109_310988


namespace NUMINAMATH_CALUDE_temporary_wall_area_l3109_310994

theorem temporary_wall_area : 
  let width : Real := 5.4
  let length : Real := 2.5
  width * length = 13.5 := by
sorry

end NUMINAMATH_CALUDE_temporary_wall_area_l3109_310994


namespace NUMINAMATH_CALUDE_circular_lid_area_l3109_310918

/-- The area of a circular lid with diameter 2.75 inches is approximately 5.9375 square inches. -/
theorem circular_lid_area :
  let diameter : ℝ := 2.75
  let radius : ℝ := diameter / 2
  let area : ℝ := Real.pi * radius^2
  ∃ ε > 0, abs (area - 5.9375) < ε :=
by sorry

end NUMINAMATH_CALUDE_circular_lid_area_l3109_310918


namespace NUMINAMATH_CALUDE_chicken_feed_bag_weight_l3109_310927

-- Define the constants from the problem
def chicken_price : ℚ := 3/2
def feed_bag_cost : ℚ := 2
def feed_per_chicken : ℚ := 2
def num_chickens : ℕ := 50
def total_profit : ℚ := 65

-- Define the theorem
theorem chicken_feed_bag_weight :
  ∃ (bag_weight : ℚ),
    bag_weight * (feed_bag_cost / feed_per_chicken) = num_chickens * chicken_price - total_profit ∧
    bag_weight > 0 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_chicken_feed_bag_weight_l3109_310927


namespace NUMINAMATH_CALUDE_arcade_spending_l3109_310900

theorem arcade_spending (allowance : ℚ) (arcade_fraction : ℚ) (remaining : ℚ) :
  allowance = 2.25 →
  remaining = 0.60 →
  remaining = (1 - arcade_fraction) * allowance - (1/3) * ((1 - arcade_fraction) * allowance) →
  arcade_fraction = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_arcade_spending_l3109_310900


namespace NUMINAMATH_CALUDE_fourteenth_root_of_unity_l3109_310954

theorem fourteenth_root_of_unity : 
  ∃ (n : ℕ), 0 ≤ n ∧ n ≤ 13 ∧ 
  (Complex.tan (Real.pi / 7) + Complex.I) / (Complex.tan (Real.pi / 7) - Complex.I) = 
  Complex.exp (Complex.I * (2 * Real.pi * n / 14)) := by
  sorry

end NUMINAMATH_CALUDE_fourteenth_root_of_unity_l3109_310954


namespace NUMINAMATH_CALUDE_partner_a_investment_l3109_310973

/-- Represents the investment and profit distribution scenario described in the problem -/
structure BusinessScenario where
  a_investment : ℚ  -- Investment of partner a
  b_investment : ℚ  -- Investment of partner b
  total_profit : ℚ  -- Total profit
  a_total_received : ℚ  -- Total amount received by partner a
  management_fee_percent : ℚ  -- Percentage of profit for management

/-- The main theorem representing the problem -/
theorem partner_a_investment (scenario : BusinessScenario) : 
  scenario.b_investment = 2500 ∧ 
  scenario.total_profit = 9600 ∧
  scenario.a_total_received = 6000 ∧
  scenario.management_fee_percent = 1/10 →
  scenario.a_investment = 3500 := by
sorry


end NUMINAMATH_CALUDE_partner_a_investment_l3109_310973


namespace NUMINAMATH_CALUDE_no_two_digit_factors_of_1729_l3109_310961

theorem no_two_digit_factors_of_1729 : 
  ¬ ∃ (a b : ℕ), 10 ≤ a ∧ a ≤ 99 ∧ 10 ≤ b ∧ b ≤ 99 ∧ a * b = 1729 := by
  sorry

end NUMINAMATH_CALUDE_no_two_digit_factors_of_1729_l3109_310961


namespace NUMINAMATH_CALUDE_petya_vasya_divisibility_l3109_310981

theorem petya_vasya_divisibility (n m : ℕ) (h : ∀ k ∈ Finset.range 100, ∃ j ∈ Finset.range 99, (m - j) ∣ (n + k)) :
  m > n^3 / 10000000 := by
  sorry

end NUMINAMATH_CALUDE_petya_vasya_divisibility_l3109_310981


namespace NUMINAMATH_CALUDE_bicycle_price_calculation_l3109_310931

theorem bicycle_price_calculation (initial_price : ℝ) : 
  let first_sale_price := initial_price * 1.20
  let final_price := first_sale_price * 1.25
  final_price = 225 → initial_price = 150 := by
sorry

end NUMINAMATH_CALUDE_bicycle_price_calculation_l3109_310931


namespace NUMINAMATH_CALUDE_expand_product_l3109_310936

theorem expand_product (x : ℝ) : (5 * x + 7) * (3 * x^2 + 2 * x + 4) = 15 * x^3 + 31 * x^2 + 34 * x + 28 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l3109_310936


namespace NUMINAMATH_CALUDE_pen_sales_problem_l3109_310979

theorem pen_sales_problem (d : ℕ) : 
  (96 + 44 * d) / (d + 1) = 48 → d = 12 := by
  sorry

end NUMINAMATH_CALUDE_pen_sales_problem_l3109_310979


namespace NUMINAMATH_CALUDE_unique_solution_for_prime_equation_l3109_310946

theorem unique_solution_for_prime_equation (p : ℕ) (hp : Prime p) :
  ∀ x y : ℕ+, p * (x - y) = x * y → x = p^2 - p ∧ y = p + 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_for_prime_equation_l3109_310946


namespace NUMINAMATH_CALUDE_blood_expiration_date_l3109_310928

/-- Represents the number of seconds in a day -/
def seconds_per_day : ℕ := 86400

/-- Represents the number of days in January -/
def days_in_january : ℕ := 31

/-- Calculates the factorial of a natural number -/
def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

/-- Represents the expiration time of blood in seconds -/
def blood_expiration_time : ℕ := factorial 10

theorem blood_expiration_date :
  blood_expiration_time / seconds_per_day = days_in_january + 11 :=
sorry

end NUMINAMATH_CALUDE_blood_expiration_date_l3109_310928


namespace NUMINAMATH_CALUDE_johns_gym_time_l3109_310934

/-- Represents the number of times John goes to the gym per week -/
def gym_visits_per_week : ℕ := 3

/-- Represents the number of hours John spends weightlifting each gym visit -/
def weightlifting_hours : ℚ := 1

/-- Represents the fraction of weightlifting time spent on warming up and cardio -/
def warmup_cardio_fraction : ℚ := 1 / 3

/-- Calculates the total hours John spends at the gym per week -/
def total_gym_hours : ℚ :=
  gym_visits_per_week * (weightlifting_hours + warmup_cardio_fraction * weightlifting_hours)

theorem johns_gym_time : total_gym_hours = 4 := by
  sorry

end NUMINAMATH_CALUDE_johns_gym_time_l3109_310934


namespace NUMINAMATH_CALUDE_triangle_inequality_from_sum_product_l3109_310935

theorem triangle_inequality_from_sum_product (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  6 * (a * b + b * c + c * a) > 5 * (a^2 + b^2 + c^2) →
  c < a + b ∧ a < b + c ∧ b < c + a :=
by sorry

end NUMINAMATH_CALUDE_triangle_inequality_from_sum_product_l3109_310935


namespace NUMINAMATH_CALUDE_intersection_condition_l3109_310983

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | x^2 - 1 = 0}
def B (a : ℝ) : Set ℝ := {x : ℝ | a * x = 1}

-- State the theorem
theorem intersection_condition (a : ℝ) : A ∩ B a = B a → a = 1 ∨ a = -1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_condition_l3109_310983


namespace NUMINAMATH_CALUDE_stating_special_multiples_count_l3109_310912

/-- 
The count of positive integers less than 500 that are multiples of 3 but not multiples of 9.
-/
def count_special_multiples : ℕ := 
  (Finset.filter (fun n => n % 3 = 0 ∧ n % 9 ≠ 0) (Finset.range 500)).card

/-- 
Theorem stating that the count of positive integers less than 500 
that are multiples of 3 but not multiples of 9 is equal to 111.
-/
theorem special_multiples_count : count_special_multiples = 111 := by
  sorry

end NUMINAMATH_CALUDE_stating_special_multiples_count_l3109_310912


namespace NUMINAMATH_CALUDE_sport_participation_l3109_310913

theorem sport_participation (total : ℕ) (cyclists : ℕ) (swimmers : ℕ) (skiers : ℕ) (unsatisfactory : ℕ)
  (h1 : total = 25)
  (h2 : cyclists = 17)
  (h3 : swimmers = 13)
  (h4 : skiers = 8)
  (h5 : unsatisfactory = 6)
  (h6 : ∀ s : ℕ, s ≤ total → s ≤ cyclists + swimmers + skiers - 2)
  (h7 : cyclists + swimmers + skiers = 2 * (total - unsatisfactory)) :
  ∃ swim_and_ski : ℕ, swim_and_ski = 2 ∧ swim_and_ski ≤ swimmers ∧ swim_and_ski ≤ skiers :=
by sorry

end NUMINAMATH_CALUDE_sport_participation_l3109_310913


namespace NUMINAMATH_CALUDE_sum_of_digits_62_l3109_310955

def is_two_digit_number (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def digit_sum (n : ℕ) : ℕ := (n / 10) + (n % 10)

def reverse_digits (n : ℕ) : ℕ := (n % 10) * 10 + (n / 10)

theorem sum_of_digits_62 :
  ∀ n : ℕ,
  is_two_digit_number n →
  n = 62 →
  reverse_digits n + 36 = n →
  digit_sum n = 8 := by
sorry

end NUMINAMATH_CALUDE_sum_of_digits_62_l3109_310955


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l3109_310984

theorem sqrt_equation_solution :
  ∀ a b : ℕ+,
  a < b →
  Real.sqrt (3 + Real.sqrt (45 + 20 * Real.sqrt 5)) = Real.sqrt a + Real.sqrt b →
  a = 3 ∧ b = 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l3109_310984


namespace NUMINAMATH_CALUDE_poultry_farm_hens_l3109_310975

theorem poultry_farm_hens (total_chickens : ℕ) (hen_rooster_ratio : ℚ) (chicks_per_hen : ℕ) : 
  total_chickens = 76 → 
  hen_rooster_ratio = 3 → 
  chicks_per_hen = 5 → 
  ∃ (num_hens : ℕ), num_hens = 12 ∧ 
    num_hens + (num_hens : ℚ) / hen_rooster_ratio + (num_hens * chicks_per_hen) = total_chickens := by
  sorry

end NUMINAMATH_CALUDE_poultry_farm_hens_l3109_310975


namespace NUMINAMATH_CALUDE_abc_inequality_l3109_310919

theorem abc_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h_sum_prod : a + b + c + 2 = a * b * c) :
  (a + 1) * (b + 1) * (c + 1) ≥ 27 ∧
  ((a + 1) * (b + 1) * (c + 1) = 27 ↔ a = 2 ∧ b = 2 ∧ c = 2) :=
by sorry

end NUMINAMATH_CALUDE_abc_inequality_l3109_310919


namespace NUMINAMATH_CALUDE_sqrt_equality_implies_specific_values_l3109_310959

theorem sqrt_equality_implies_specific_values :
  ∀ a b : ℕ+,
  a < b →
  Real.sqrt (4 + Real.sqrt (76 + 40 * Real.sqrt 3)) = Real.sqrt a + Real.sqrt b →
  a = 1 ∧ b = 10 := by
sorry

end NUMINAMATH_CALUDE_sqrt_equality_implies_specific_values_l3109_310959


namespace NUMINAMATH_CALUDE_fraction_subtraction_l3109_310905

theorem fraction_subtraction : 
  (5 + 7 + 9) / (2 + 4 + 6) - (4 + 6 + 8) / (3 + 5 + 7) = 11 / 20 := by
  sorry

end NUMINAMATH_CALUDE_fraction_subtraction_l3109_310905


namespace NUMINAMATH_CALUDE_min_value_of_expression_l3109_310962

theorem min_value_of_expression (x : ℝ) :
  ∃ (min : ℝ), min = -4356 ∧ ∀ y : ℝ, (14 - y) * (8 - y) * (14 + y) * (8 + y) ≥ min :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l3109_310962


namespace NUMINAMATH_CALUDE_fib_like_seq_a9_l3109_310906

/-- An increasing sequence of positive integers with a Fibonacci-like recurrence relation -/
def FibLikeSeq (a : ℕ → ℕ) : Prop :=
  (∀ n, a n < a (n + 1)) ∧ 
  (∀ n, n ≥ 1 → a (n + 2) = a (n + 1) + a n)

theorem fib_like_seq_a9 (a : ℕ → ℕ) (h : FibLikeSeq a) (h7 : a 7 = 210) : 
  a 9 = 550 := by
  sorry

end NUMINAMATH_CALUDE_fib_like_seq_a9_l3109_310906


namespace NUMINAMATH_CALUDE_willies_stickers_l3109_310995

theorem willies_stickers (initial : ℕ) (given : ℕ) (remaining : ℕ) : 
  given = 7 → remaining = 29 → initial = remaining + given :=
by
  sorry

end NUMINAMATH_CALUDE_willies_stickers_l3109_310995


namespace NUMINAMATH_CALUDE_water_volume_ratio_in_cone_l3109_310943

/-- Theorem: Volume ratio of water in a cone filled to 2/3 height -/
theorem water_volume_ratio_in_cone (h r : ℝ) (h_pos : h > 0) (r_pos : r > 0) :
  let water_height : ℝ := 2 / 3 * h
  let water_radius : ℝ := 2 / 3 * r
  let cone_volume : ℝ := (1 / 3) * π * r^2 * h
  let water_volume : ℝ := (1 / 3) * π * water_radius^2 * water_height
  water_volume / cone_volume = 8 / 27 := by
  sorry

end NUMINAMATH_CALUDE_water_volume_ratio_in_cone_l3109_310943


namespace NUMINAMATH_CALUDE_extended_equilateral_area_ratio_l3109_310901

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a triangle -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Calculates the area of a triangle -/
def triangleArea (t : Triangle) : ℝ := sorry

/-- Checks if a triangle is equilateral -/
def isEquilateral (t : Triangle) : Prop := sorry

/-- Extends a line segment by a factor -/
def extendSegment (A B : Point) (factor : ℝ) : Point := sorry

theorem extended_equilateral_area_ratio 
  (P Q R : Point) 
  (t : Triangle)
  (h_equilateral : isEquilateral t)
  (h_t : t = Triangle.mk P Q R)
  (Q' : Point)
  (h_Q' : Q' = extendSegment P Q 3)
  (R' : Point)
  (h_R' : R' = extendSegment Q R 3)
  (P' : Point)
  (h_P' : P' = extendSegment R P 3)
  (t_extended : Triangle)
  (h_t_extended : t_extended = Triangle.mk P' Q' R') :
  triangleArea t_extended / triangleArea t = 9 := by sorry

end NUMINAMATH_CALUDE_extended_equilateral_area_ratio_l3109_310901


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3109_310945

def A : Set ℝ := {x | x ≤ 1}
def B : Set ℝ := {-2, -1, 0, 1, 2}

theorem intersection_of_A_and_B : A ∩ B = {-2, -1, 0, 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3109_310945


namespace NUMINAMATH_CALUDE_trig_expression_equality_l3109_310910

theorem trig_expression_equality : 
  (Real.sin (40 * π / 180) - Real.sqrt 3 * Real.cos (20 * π / 180)) / Real.cos (10 * π / 180) = -1 := by
  sorry

end NUMINAMATH_CALUDE_trig_expression_equality_l3109_310910


namespace NUMINAMATH_CALUDE_peters_savings_l3109_310987

/-- Peter's vacation savings problem -/
theorem peters_savings (total_needed : ℕ) (monthly_savings : ℕ) (months_to_goal : ℕ) 
  (h1 : total_needed = 5000)
  (h2 : monthly_savings = 700)
  (h3 : months_to_goal = 3)
  (h4 : total_needed = monthly_savings * months_to_goal + current_savings) :
  current_savings = 2900 :=
by
  sorry

end NUMINAMATH_CALUDE_peters_savings_l3109_310987


namespace NUMINAMATH_CALUDE_points_below_line_l3109_310922

def arithmetic_sequence (a b c d : ℝ) : Prop :=
  b - a = c - b ∧ c - b = d - c

def geometric_sequence (a b c d : ℝ) : Prop :=
  b / a = c / b ∧ c / b = d / c

theorem points_below_line (x₁ x₂ y₁ y₂ : ℝ) :
  arithmetic_sequence 1 x₁ x₂ 2 →
  geometric_sequence 1 y₁ y₂ 2 →
  x₁ > y₁ ∧ x₂ > y₂ := by
  sorry

end NUMINAMATH_CALUDE_points_below_line_l3109_310922


namespace NUMINAMATH_CALUDE_guard_arrangement_exists_l3109_310993

/-- Represents a guard with a position and direction of sight -/
structure Guard where
  position : ℝ × ℝ
  direction : ℝ × ℝ

/-- Represents the arrangement of guards around a point object -/
structure GuardArrangement where
  guards : List Guard
  object : ℝ × ℝ
  visibility_range : ℝ

/-- Predicate to check if a point is inside or on the boundary of a convex hull -/
def is_inside_or_on_convex_hull (point : ℝ × ℝ) (hull : List (ℝ × ℝ)) : Prop :=
  sorry

/-- Predicate to check if a list of points forms a convex hull -/
def is_convex_hull (points : List (ℝ × ℝ)) : Prop :=
  sorry

/-- Predicate to check if it's impossible to approach any point unnoticed -/
def is_approach_impossible (arrangement : GuardArrangement) : Prop :=
  sorry

/-- Theorem stating that it's possible to arrange guards to prevent unnoticed approach -/
theorem guard_arrangement_exists : ∃ (arrangement : GuardArrangement),
  arrangement.visibility_range = 100 ∧
  arrangement.guards.length ≥ 6 ∧
  is_convex_hull (arrangement.guards.map Guard.position) ∧
  is_inside_or_on_convex_hull arrangement.object (arrangement.guards.map Guard.position) ∧
  is_approach_impossible arrangement :=
by
  sorry

end NUMINAMATH_CALUDE_guard_arrangement_exists_l3109_310993


namespace NUMINAMATH_CALUDE_fuel_cost_savings_l3109_310964

theorem fuel_cost_savings (old_efficiency : ℝ) (old_fuel_cost : ℝ) 
  (trip_distance : ℝ) (efficiency_improvement : ℝ) (fuel_cost_increase : ℝ) :
  old_efficiency > 0 → old_fuel_cost > 0 → trip_distance > 0 →
  efficiency_improvement = 0.6 → fuel_cost_increase = 0.25 → trip_distance = 300 →
  let new_efficiency := old_efficiency * (1 + efficiency_improvement)
  let new_fuel_cost := old_fuel_cost * (1 + fuel_cost_increase)
  let old_trip_cost := (trip_distance / old_efficiency) * old_fuel_cost
  let new_trip_cost := (trip_distance / new_efficiency) * new_fuel_cost
  let savings_percentage := (old_trip_cost - new_trip_cost) / old_trip_cost * 100
  savings_percentage = 21.875 := by
sorry

end NUMINAMATH_CALUDE_fuel_cost_savings_l3109_310964


namespace NUMINAMATH_CALUDE_december_sales_multiple_l3109_310917

/-- Represents the sales data for a department store --/
structure SalesData where
  /-- Average monthly sales from January to November --/
  avg_sales : ℝ
  /-- Multiple of average sales for December --/
  dec_multiple : ℝ

/-- Theorem stating the conditions and the result to be proved --/
theorem december_sales_multiple (data : SalesData) :
  (data.dec_multiple * data.avg_sales) / (11 * data.avg_sales + data.dec_multiple * data.avg_sales) = 0.35294117647058826 →
  data.dec_multiple = 6 := by
  sorry

end NUMINAMATH_CALUDE_december_sales_multiple_l3109_310917


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l3109_310939

theorem quadratic_inequality_solution_set 
  (a b c α β : ℝ) 
  (h1 : α > 0) 
  (h2 : β > α) 
  (h3 : ∀ x, ax^2 + b*x + c > 0 ↔ α < x ∧ x < β) :
  ∀ x, c*x^2 + b*x + a > 0 ↔ 1/β < x ∧ x < 1/α :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l3109_310939


namespace NUMINAMATH_CALUDE_S_31_composite_bound_l3109_310967

def S (k : ℕ+) (n : ℕ) : ℕ :=
  (n.digits k.val).sum

def is_prime (p : ℕ) : Prop :=
  p > 1 ∧ ∀ d : ℕ, d ∣ p → d = 1 ∨ d = p

def is_composite (n : ℕ) : Prop :=
  n > 1 ∧ ∃ d : ℕ, 1 < d ∧ d < n ∧ d ∣ n

theorem S_31_composite_bound :
  ∃ (A : Finset ℕ), A.card ≤ 2 ∧
    ∀ p : ℕ, is_prime p → p < 20000 →
      is_composite (S 31 p) → S 31 p ∈ A :=
sorry

end NUMINAMATH_CALUDE_S_31_composite_bound_l3109_310967
