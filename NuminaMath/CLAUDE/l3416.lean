import Mathlib

namespace NUMINAMATH_CALUDE_square_rectangle_triangle_relation_l3416_341666

/-- Square with side length 2 -/
structure Square :=
  (side : ℝ)
  (is_two : side = 2)

/-- Rectangle with width and height -/
structure Rectangle :=
  (width : ℝ)
  (height : ℝ)

/-- Right triangle with base and height -/
structure RightTriangle :=
  (base : ℝ)
  (height : ℝ)

/-- The main theorem -/
theorem square_rectangle_triangle_relation 
  (ABCD : Square)
  (JKHG : Rectangle)
  (EBC : RightTriangle)
  (h1 : JKHG.width = ABCD.side)
  (h2 : EBC.base = ABCD.side)
  (h3 : JKHG.height = EBC.height)
  (h4 : JKHG.width * JKHG.height = 2 * (EBC.base * EBC.height / 2)) :
  EBC.height = 1 := by
  sorry


end NUMINAMATH_CALUDE_square_rectangle_triangle_relation_l3416_341666


namespace NUMINAMATH_CALUDE_product_expansion_sum_l3416_341650

theorem product_expansion_sum (a b c d : ℝ) :
  (∀ x, (4 * x^2 - 6 * x + 5) * (9 - 3 * x) = a * x^3 + b * x^2 + c * x + d) →
  9 * a + 3 * b + 2 * c + d = -39 := by
sorry

end NUMINAMATH_CALUDE_product_expansion_sum_l3416_341650


namespace NUMINAMATH_CALUDE_atop_difference_l3416_341627

-- Define the @ operation
def atop (x y : ℤ) : ℤ := x * y + x - y

-- Theorem statement
theorem atop_difference : (atop 7 4) - (atop 4 7) = 6 := by
  sorry

end NUMINAMATH_CALUDE_atop_difference_l3416_341627


namespace NUMINAMATH_CALUDE_mans_age_fraction_l3416_341660

theorem mans_age_fraction (mans_age father_age : ℕ) : 
  father_age = 25 →
  mans_age + 5 = (father_age + 5) / 2 →
  mans_age / father_age = 2 / 5 := by
sorry

end NUMINAMATH_CALUDE_mans_age_fraction_l3416_341660


namespace NUMINAMATH_CALUDE_smallest_b_value_l3416_341637

def is_factor (m n : ℕ) : Prop := n % m = 0

theorem smallest_b_value (a b : ℕ) : 
  a = 363 → 
  is_factor 112 (a * 43 * 62 * b) → 
  is_factor 33 (a * 43 * 62 * b) → 
  b ≥ 56 ∧ is_factor 112 (a * 43 * 62 * 56) ∧ is_factor 33 (a * 43 * 62 * 56) :=
sorry

end NUMINAMATH_CALUDE_smallest_b_value_l3416_341637


namespace NUMINAMATH_CALUDE_binomial_n_n_minus_3_l3416_341671

theorem binomial_n_n_minus_3 (n : ℕ) (h : n ≥ 3) :
  Nat.choose n (n - 3) = n * (n - 1) * (n - 2) / 6 := by
  sorry

end NUMINAMATH_CALUDE_binomial_n_n_minus_3_l3416_341671


namespace NUMINAMATH_CALUDE_probability_divisible_by_20_l3416_341600

def digits : Finset ℕ := {1, 1, 2, 3, 4, 5, 6}

def is_valid_arrangement (n : ℕ) : Prop :=
  n ≥ 1000000 ∧ n < 10000000 ∧ (Finset.card (Finset.filter (λ d => d ∈ digits) (Finset.range 10)) = 7)

def is_divisible_by_20 (n : ℕ) : Prop := n % 20 = 0

def total_arrangements : ℕ := 7 * 6 * 5 * 4 * 3 * 2 * 1

def favorable_arrangements : ℕ := 2 * 5 * 4 * 3 * 2 * 1

theorem probability_divisible_by_20 :
  (favorable_arrangements : ℚ) / total_arrangements = 1 / 21 := by sorry

end NUMINAMATH_CALUDE_probability_divisible_by_20_l3416_341600


namespace NUMINAMATH_CALUDE_complex_real_implies_a_eq_neg_one_l3416_341618

theorem complex_real_implies_a_eq_neg_one (a : ℝ) :
  (Complex.I : ℂ) * (a + 1 : ℝ) = (0 : ℂ) → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_real_implies_a_eq_neg_one_l3416_341618


namespace NUMINAMATH_CALUDE_evaluate_expression_l3416_341628

theorem evaluate_expression : 150 * (150 - 5) - (150 * 150 + 13) = -763 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3416_341628


namespace NUMINAMATH_CALUDE_trigonometric_properties_l3416_341643

theorem trigonometric_properties :
  (∀ x, 2 * Real.sin (2 * x - π / 3) = 2 * Real.sin (2 * (5 * π / 6 - x) - π / 3)) ∧
  (∀ x, Real.tan x = -Real.tan (π - x)) ∧
  (∃ x₁ x₂, x₁ > 0 ∧ x₂ > 0 ∧ x₁ < π / 2 ∧ x₂ < π / 2 ∧ x₁ > x₂ ∧ Real.sin x₁ < Real.sin x₂) ∧
  (∀ x₁ x₂, Real.sin (2 * x₁ - π / 4) = Real.sin (2 * x₂ - π / 4) →
    (∃ k : ℤ, x₁ - x₂ = k * π ∨ x₁ + x₂ = k * π + 3 * π / 4)) :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_properties_l3416_341643


namespace NUMINAMATH_CALUDE_comic_books_left_l3416_341635

theorem comic_books_left (initial_total : ℕ) (sold : ℕ) (left : ℕ) : 
  initial_total = 90 → sold = 65 → left = initial_total - sold → left = 25 := by
  sorry

end NUMINAMATH_CALUDE_comic_books_left_l3416_341635


namespace NUMINAMATH_CALUDE_optimal_purchase_plan_l3416_341657

/-- Represents the daily transportation capacity of machine A in tons -/
def machine_A_capacity : ℝ := 90

/-- Represents the daily transportation capacity of machine B in tons -/
def machine_B_capacity : ℝ := 100

/-- Represents the cost of machine A in yuan -/
def machine_A_cost : ℝ := 15000

/-- Represents the cost of machine B in yuan -/
def machine_B_cost : ℝ := 20000

/-- Represents the total number of machines to be purchased -/
def total_machines : ℕ := 30

/-- Represents the minimum daily transportation requirement in tons -/
def min_daily_transportation : ℝ := 2880

/-- Represents the maximum purchase amount in yuan -/
def max_purchase_amount : ℝ := 550000

/-- Represents the optimal number of A machines to purchase -/
def optimal_A_machines : ℕ := 12

/-- Represents the optimal number of B machines to purchase -/
def optimal_B_machines : ℕ := 18

/-- Represents the total purchase amount for the optimal plan in yuan -/
def optimal_purchase_amount : ℝ := 54000

theorem optimal_purchase_plan :
  (machine_B_capacity = machine_A_capacity + 10) ∧
  (450 / machine_A_capacity = 500 / machine_B_capacity) ∧
  (optimal_A_machines + optimal_B_machines = total_machines) ∧
  (optimal_A_machines * machine_A_capacity + optimal_B_machines * machine_B_capacity ≥ min_daily_transportation) ∧
  (optimal_A_machines * machine_A_cost + optimal_B_machines * machine_B_cost = optimal_purchase_amount) ∧
  (optimal_purchase_amount ≤ max_purchase_amount) ∧
  (∀ a b : ℕ, a + b = total_machines →
    a * machine_A_capacity + b * machine_B_capacity ≥ min_daily_transportation →
    a * machine_A_cost + b * machine_B_cost ≤ max_purchase_amount →
    a * machine_A_cost + b * machine_B_cost ≥ optimal_purchase_amount) := by
  sorry


end NUMINAMATH_CALUDE_optimal_purchase_plan_l3416_341657


namespace NUMINAMATH_CALUDE_f_vertex_f_at_zero_f_expression_f_monotonic_interval_l3416_341619

/-- A quadratic function with vertex at (1, 1) and f(0) = 3 -/
def f (x : ℝ) : ℝ := 2 * x^2 - 4 * x + 3

/-- The vertex of f is at (1, 1) -/
theorem f_vertex : ∀ x : ℝ, f x ≥ f 1 := sorry

/-- f(0) = 3 -/
theorem f_at_zero : f 0 = 3 := sorry

/-- f(x) = 2x^2 - 4x + 3 -/
theorem f_expression : ∀ x : ℝ, f x = 2 * x^2 - 4 * x + 3 := sorry

/-- f(x) is monotonic in [a, a+1] iff a ≤ 0 or a ≥ 1 -/
theorem f_monotonic_interval (a : ℝ) :
  (∀ x y : ℝ, a ≤ x ∧ x ≤ y ∧ y ≤ a + 1 → f x ≤ f y) ↔ (a ≤ 0 ∨ a ≥ 1) := sorry

end NUMINAMATH_CALUDE_f_vertex_f_at_zero_f_expression_f_monotonic_interval_l3416_341619


namespace NUMINAMATH_CALUDE_positive_numbers_l3416_341690

theorem positive_numbers (a b c : ℝ) 
  (sum_positive : a + b + c > 0)
  (pairwise_sum_positive : a * b + b * c + c * a > 0)
  (product_positive : a * b * c > 0) :
  a > 0 ∧ b > 0 ∧ c > 0 := by
  sorry

end NUMINAMATH_CALUDE_positive_numbers_l3416_341690


namespace NUMINAMATH_CALUDE_notebook_cost_example_l3416_341613

/-- The cost of notebooks given the number of notebooks, pages per notebook, and cost per page. -/
def notebook_cost (num_notebooks : ℕ) (pages_per_notebook : ℕ) (cost_per_page : ℚ) : ℚ :=
  (num_notebooks * pages_per_notebook : ℚ) * cost_per_page

/-- Theorem stating that the cost of 2 notebooks with 50 pages each, at 5 cents per page, is $5.00 -/
theorem notebook_cost_example : notebook_cost 2 50 (5 / 100) = 5 := by
  sorry

end NUMINAMATH_CALUDE_notebook_cost_example_l3416_341613


namespace NUMINAMATH_CALUDE_point_order_on_parabola_l3416_341606

-- Define the parabola function
def parabola (x : ℝ) : ℝ := -(x - 1)^2 + 2

-- Define the theorem
theorem point_order_on_parabola (a b c : ℝ) :
  parabola a = -2 →
  parabola b = -2 →
  parabola c = -7 →
  a < b →
  c > 2 →
  a < b ∧ b < c := by
  sorry

end NUMINAMATH_CALUDE_point_order_on_parabola_l3416_341606


namespace NUMINAMATH_CALUDE_gcd_7392_15015_l3416_341645

theorem gcd_7392_15015 : Nat.gcd 7392 15015 = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_7392_15015_l3416_341645


namespace NUMINAMATH_CALUDE_harmonic_sum_inequality_l3416_341661

theorem harmonic_sum_inequality : 1 + 1/2 + 1/3 < 2 := by
  sorry

end NUMINAMATH_CALUDE_harmonic_sum_inequality_l3416_341661


namespace NUMINAMATH_CALUDE_cookie_pie_leftover_slices_l3416_341612

theorem cookie_pie_leftover_slices (num_pies : ℕ) (slices_per_pie : ℕ) (num_classmates : ℕ) (num_teachers : ℕ) (slices_per_person : ℕ) :
  num_pies = 3 →
  slices_per_pie = 10 →
  num_classmates = 24 →
  num_teachers = 1 →
  slices_per_person = 1 →
  num_pies * slices_per_pie - (num_classmates + num_teachers + 1) * slices_per_person = 4 :=
by sorry

end NUMINAMATH_CALUDE_cookie_pie_leftover_slices_l3416_341612


namespace NUMINAMATH_CALUDE_chris_candy_distribution_l3416_341638

/-- The number of friends Chris has -/
def num_friends : ℕ := 35

/-- The number of candy pieces each friend receives -/
def candy_per_friend : ℕ := 12

/-- The total number of candy pieces Chris gave to his friends -/
def total_candy : ℕ := num_friends * candy_per_friend

theorem chris_candy_distribution :
  total_candy = 420 :=
by sorry

end NUMINAMATH_CALUDE_chris_candy_distribution_l3416_341638


namespace NUMINAMATH_CALUDE_trigonometric_inequality_l3416_341696

theorem trigonometric_inequality : ∃ (a b c : ℝ),
  a = Real.tan (3 * Real.pi / 4) ∧
  b = Real.cos (2 * Real.pi / 5) ∧
  c = (1 + Real.sin (6 * Real.pi / 5)) ^ 0 ∧
  c > b ∧ b > a := by sorry

end NUMINAMATH_CALUDE_trigonometric_inequality_l3416_341696


namespace NUMINAMATH_CALUDE_leftHandedWomenPercentage_l3416_341664

/-- Represents the population of Smithtown -/
structure Population where
  rightHanded : ℕ
  leftHanded : ℕ
  men : ℕ
  women : ℕ

/-- Conditions for a valid Smithtown population -/
def isValidPopulation (p : Population) : Prop :=
  p.rightHanded = 3 * p.leftHanded ∧
  p.men = 3 * p.women / 2 ∧
  p.rightHanded + p.leftHanded = p.men + p.women

/-- A population with maximized right-handed men -/
def hasMaximizedRightHandedMen (p : Population) : Prop :=
  p.men = p.rightHanded

/-- Theorem: In a valid Smithtown population with maximized right-handed men,
    left-handed women constitute 25% of the total population -/
theorem leftHandedWomenPercentage (p : Population) 
  (hValid : isValidPopulation p) 
  (hMax : hasMaximizedRightHandedMen p) : 
  (p.leftHanded : ℚ) / (p.rightHanded + p.leftHanded : ℚ) = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_leftHandedWomenPercentage_l3416_341664


namespace NUMINAMATH_CALUDE_wall_bricks_l3416_341697

/-- Represents the number of bricks in the wall -/
def num_bricks : ℕ := 288

/-- Represents the time taken by the first bricklayer to build the wall alone -/
def time_bricklayer1 : ℕ := 8

/-- Represents the time taken by the second bricklayer to build the wall alone -/
def time_bricklayer2 : ℕ := 12

/-- Represents the reduction in combined output when working together -/
def output_reduction : ℕ := 12

/-- Represents the time taken by both bricklayers working together -/
def time_together : ℕ := 6

/-- Theorem stating that the number of bricks in the wall is 288 -/
theorem wall_bricks :
  (time_together : ℚ) * ((num_bricks / time_bricklayer1 : ℚ) + 
  (num_bricks / time_bricklayer2 : ℚ) - output_reduction) = num_bricks := by
  sorry

#eval num_bricks

end NUMINAMATH_CALUDE_wall_bricks_l3416_341697


namespace NUMINAMATH_CALUDE_min_value_f_max_value_y_l3416_341668

/-- The minimum value of f(x) = 4/x + x for x > 0 is 4 -/
theorem min_value_f (x : ℝ) (hx : x > 0) :
  (4 / x + x) ≥ 4 ∧ ∃ x₀ > 0, 4 / x₀ + x₀ = 4 := by sorry

/-- The maximum value of y = x(1 - 3x) for 0 < x < 1/3 is 1/12 -/
theorem max_value_y (x : ℝ) (hx1 : x > 0) (hx2 : x < 1/3) :
  x * (1 - 3 * x) ≤ 1/12 ∧ ∃ x₀ ∈ (Set.Ioo 0 (1/3)), x₀ * (1 - 3 * x₀) = 1/12 := by sorry

end NUMINAMATH_CALUDE_min_value_f_max_value_y_l3416_341668


namespace NUMINAMATH_CALUDE_bubble_sort_correct_l3416_341639

def bubbleSort (xs : List Int) : List Int :=
  let rec pass : List Int → List Int
    | [] => []
    | [x] => [x]
    | x :: y :: rest => if x <= y then x :: pass (y :: rest) else y :: pass (x :: rest)
  let rec sort (xs : List Int) (n : Nat) : List Int :=
    if n = 0 then xs else sort (pass xs) (n - 1)
  sort xs xs.length

theorem bubble_sort_correct (xs : List Int) :
  bubbleSort [8, 6, 3, 18, 21, 67, 54] = [3, 6, 8, 18, 21, 54, 67] := by
  sorry

end NUMINAMATH_CALUDE_bubble_sort_correct_l3416_341639


namespace NUMINAMATH_CALUDE_buffalo_count_is_two_l3416_341670

/-- Represents the number of animals seen on each day of Erica's safari --/
structure SafariCount where
  saturday : ℕ
  sunday_leopards : ℕ
  sunday_buffaloes : ℕ
  monday : ℕ

/-- The total number of animals seen during the safari --/
def total_animals : ℕ := 20

/-- The actual count of animals seen on each day --/
def safari_count : SafariCount where
  saturday := 5  -- 3 lions + 2 elephants
  sunday_leopards := 5
  sunday_buffaloes := 2  -- This is what we want to prove
  monday := 8  -- 5 rhinos + 3 warthogs

theorem buffalo_count_is_two :
  safari_count.sunday_buffaloes = 2 :=
by
  sorry

#check buffalo_count_is_two

end NUMINAMATH_CALUDE_buffalo_count_is_two_l3416_341670


namespace NUMINAMATH_CALUDE_customer_b_bought_five_units_l3416_341623

/-- Represents the phone inventory and sales of a store -/
structure PhoneStore where
  total_units : ℕ
  defective_units : ℕ
  customer_a_units : ℕ
  customer_c_units : ℕ

/-- Calculates the number of units sold to Customer B -/
def units_sold_to_b (store : PhoneStore) : ℕ :=
  store.total_units - store.defective_units - store.customer_a_units - store.customer_c_units

/-- Theorem stating that Customer B bought 5 units -/
theorem customer_b_bought_five_units (store : PhoneStore) 
  (h1 : store.total_units = 20)
  (h2 : store.defective_units = 5)
  (h3 : store.customer_a_units = 3)
  (h4 : store.customer_c_units = 7) :
  units_sold_to_b store = 5 := by
  sorry

end NUMINAMATH_CALUDE_customer_b_bought_five_units_l3416_341623


namespace NUMINAMATH_CALUDE_remainder_x_105_divided_by_x_plus_1_4_l3416_341663

theorem remainder_x_105_divided_by_x_plus_1_4 (x : ℤ) :
  x^105 ≡ 195300*x^3 + 580440*x^2 + 576085*x + 189944 [ZMOD (x + 1)^4] := by
  sorry

end NUMINAMATH_CALUDE_remainder_x_105_divided_by_x_plus_1_4_l3416_341663


namespace NUMINAMATH_CALUDE_fraction_equation_transformation_l3416_341680

/-- Given the fractional equation (x / (x - 1)) - (2 / x) = 1,
    prove that eliminating the denominators results in x^2 - 2(x-1) = x(x-1) -/
theorem fraction_equation_transformation (x : ℝ) (h1 : x ≠ 0) (h2 : x ≠ 1) :
  (x / (x - 1)) - (2 / x) = 1 ↔ x^2 - 2*(x-1) = x*(x-1) :=
by sorry

end NUMINAMATH_CALUDE_fraction_equation_transformation_l3416_341680


namespace NUMINAMATH_CALUDE_angle_of_inclination_for_unit_slope_l3416_341615

/-- Given a line with slope of absolute value 1, its angle of inclination is either 45° or 135°. -/
theorem angle_of_inclination_for_unit_slope (slope : ℝ) (h : |slope| = 1) :
  let angle := Real.arctan slope
  angle = π/4 ∨ angle = 3*π/4 := by
sorry

end NUMINAMATH_CALUDE_angle_of_inclination_for_unit_slope_l3416_341615


namespace NUMINAMATH_CALUDE_sculpture_cost_in_yuan_l3416_341679

-- Define the exchange rates
def usd_to_namibian_dollar : ℚ := 8
def usd_to_chinese_yuan : ℚ := 5

-- Define the cost of the sculpture in Namibian dollars
def sculpture_cost_namibian : ℚ := 160

-- Theorem to prove
theorem sculpture_cost_in_yuan :
  (sculpture_cost_namibian / usd_to_namibian_dollar) * usd_to_chinese_yuan = 100 := by
  sorry

end NUMINAMATH_CALUDE_sculpture_cost_in_yuan_l3416_341679


namespace NUMINAMATH_CALUDE_apple_cost_18_pounds_l3416_341665

/-- The cost of apples given a rate and a quantity -/
def apple_cost (rate_dollars : ℚ) (rate_pounds : ℚ) (quantity : ℚ) : ℚ :=
  (rate_dollars / rate_pounds) * quantity

/-- Theorem: The cost of 18 pounds of apples at a rate of 5 dollars per 6 pounds is 15 dollars -/
theorem apple_cost_18_pounds : apple_cost 5 6 18 = 15 := by
  sorry

end NUMINAMATH_CALUDE_apple_cost_18_pounds_l3416_341665


namespace NUMINAMATH_CALUDE_lucy_money_problem_l3416_341684

theorem lucy_money_problem (initial_amount : ℚ) : 
  (initial_amount * (2/3) * (3/4) = 15) → initial_amount = 30 := by
  sorry

end NUMINAMATH_CALUDE_lucy_money_problem_l3416_341684


namespace NUMINAMATH_CALUDE_urns_can_be_emptied_l3416_341667

/-- Represents the two types of operations that can be performed on the urns -/
inductive UrnOperation
  | Remove : ℕ → UrnOperation
  | DoubleFirst : UrnOperation
  | DoubleSecond : UrnOperation

/-- Applies a single operation to the pair of urns -/
def applyOperation (a b : ℕ) (op : UrnOperation) : ℕ × ℕ :=
  match op with
  | UrnOperation.Remove n => (a - min a n, b - min b n)
  | UrnOperation.DoubleFirst => (2 * a, b)
  | UrnOperation.DoubleSecond => (a, 2 * b)

/-- Theorem: Both urns can be made empty after a finite number of operations -/
theorem urns_can_be_emptied (a b : ℕ) :
  ∃ (ops : List UrnOperation), (ops.foldl (fun (pair : ℕ × ℕ) (op : UrnOperation) => applyOperation pair.1 pair.2 op) (a, b)).1 = 0 ∧
                               (ops.foldl (fun (pair : ℕ × ℕ) (op : UrnOperation) => applyOperation pair.1 pair.2 op) (a, b)).2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_urns_can_be_emptied_l3416_341667


namespace NUMINAMATH_CALUDE_xiao_ming_reading_progress_l3416_341617

/-- Calculates the starting page for the 6th day of reading -/
def starting_page_6th_day (total_pages book_pages_per_day days_read : ℕ) : ℕ :=
  book_pages_per_day * days_read + 1

/-- Proves that the starting page for the 6th day is 301 -/
theorem xiao_ming_reading_progress : starting_page_6th_day 500 60 5 = 301 := by
  sorry

end NUMINAMATH_CALUDE_xiao_ming_reading_progress_l3416_341617


namespace NUMINAMATH_CALUDE_barbara_age_when_mike_24_l3416_341662

/-- Given that Mike is 16 years old and Barbara is half his age, 
    prove that Barbara will be 16 years old when Mike is 24. -/
theorem barbara_age_when_mike_24 (mike_current_age barbara_current_age mike_future_age : ℕ) : 
  mike_current_age = 16 →
  barbara_current_age = mike_current_age / 2 →
  mike_future_age = 24 →
  barbara_current_age + (mike_future_age - mike_current_age) = 16 :=
by sorry

end NUMINAMATH_CALUDE_barbara_age_when_mike_24_l3416_341662


namespace NUMINAMATH_CALUDE_inscribed_cone_volume_l3416_341693

/-- The volume of an inscribed cone in a larger cone -/
theorem inscribed_cone_volume 
  (H : ℝ) -- Height of the outer cone
  (α : ℝ) -- Angle between slant height and altitude of outer cone
  (h_pos : H > 0) -- Assumption that height is positive
  (α_range : 0 < α ∧ α < π/2) -- Assumption that α is between 0 and π/2
  : ∃ (V : ℝ), 
    -- V represents the volume of the inscribed cone
    -- The inscribed cone's vertex coincides with the center of the base of the outer cone
    -- The slant heights of both cones are mutually perpendicular
    V = (1/12) * π * H^3 * (Real.sin α)^2 * (Real.sin (2*α))^2 :=
by
  sorry

end NUMINAMATH_CALUDE_inscribed_cone_volume_l3416_341693


namespace NUMINAMATH_CALUDE_expand_quadratic_l3416_341626

theorem expand_quadratic (a : ℝ) : a * (a - 3) = a^2 - 3*a := by
  sorry

end NUMINAMATH_CALUDE_expand_quadratic_l3416_341626


namespace NUMINAMATH_CALUDE_only_elevator_is_pure_translation_l3416_341654

/-- Represents a physical phenomenon --/
inductive Phenomenon
  | RollingSoccerBall
  | RotatingFanBlades
  | ElevatorGoingUp
  | MovingCarRearWheel

/-- Defines whether a phenomenon exhibits pure translation --/
def isPureTranslation (p : Phenomenon) : Prop :=
  match p with
  | Phenomenon.ElevatorGoingUp => True
  | _ => False

/-- The rolling soccer ball involves both rotation and translation --/
axiom rolling_soccer_ball_not_pure_translation :
  ¬ isPureTranslation Phenomenon.RollingSoccerBall

/-- Rotating fan blades involve rotation around a central axis --/
axiom rotating_fan_blades_not_pure_translation :
  ¬ isPureTranslation Phenomenon.RotatingFanBlades

/-- An elevator going up moves from one level to another without rotating --/
axiom elevator_going_up_is_pure_translation :
  isPureTranslation Phenomenon.ElevatorGoingUp

/-- A moving car rear wheel primarily exhibits rotation --/
axiom moving_car_rear_wheel_not_pure_translation :
  ¬ isPureTranslation Phenomenon.MovingCarRearWheel

/-- Theorem: Only the elevator going up exhibits pure translation --/
theorem only_elevator_is_pure_translation :
  ∀ p : Phenomenon, isPureTranslation p ↔ p = Phenomenon.ElevatorGoingUp :=
by sorry


end NUMINAMATH_CALUDE_only_elevator_is_pure_translation_l3416_341654


namespace NUMINAMATH_CALUDE_cube_sum_from_sum_and_square_sum_l3416_341678

theorem cube_sum_from_sum_and_square_sum (x y : ℝ) 
  (h1 : x + y = 5) 
  (h2 : x^2 + y^2 = 13) : 
  x^3 + y^3 = 35 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_from_sum_and_square_sum_l3416_341678


namespace NUMINAMATH_CALUDE_seven_power_plus_one_prime_factors_l3416_341687

theorem seven_power_plus_one_prime_factors (n : ℕ) :
  ∃ (primes : Finset ℕ), 
    (∀ p ∈ primes, Nat.Prime p) ∧ 
    (primes.card ≥ 2 * n + 3) ∧ 
    ((primes.prod id) = 7^(7^n) + 1) :=
sorry

end NUMINAMATH_CALUDE_seven_power_plus_one_prime_factors_l3416_341687


namespace NUMINAMATH_CALUDE_rain_probability_l3416_341616

theorem rain_probability (p : ℚ) (n : ℕ) (hp : p = 4/5) (hn : n = 5) :
  1 - (1 - p)^n = 3124/3125 := by
  sorry

end NUMINAMATH_CALUDE_rain_probability_l3416_341616


namespace NUMINAMATH_CALUDE_solve_system_with_partial_info_l3416_341699

/-- Given a system of linear equations and information about its solutions,
    this theorem proves the values of the coefficients. -/
theorem solve_system_with_partial_info :
  ∀ (a b c : ℚ),
  (∀ x y : ℚ, a*x + b*y = 2 ∧ c*x - 3*y = -2 → x = 1 ∧ y = -1) →
  (a*2 + b*(-6) = 2) →
  (a = 5/2 ∧ b = 1/2 ∧ c = -5) :=
by sorry

end NUMINAMATH_CALUDE_solve_system_with_partial_info_l3416_341699


namespace NUMINAMATH_CALUDE_number_comparisons_l3416_341672

theorem number_comparisons :
  (31^11 < 17^14) ∧
  (33^75 > 63^60) ∧
  (82^33 > 26^44) ∧
  (29^31 > 80^23) := by
  sorry

end NUMINAMATH_CALUDE_number_comparisons_l3416_341672


namespace NUMINAMATH_CALUDE_x_intercept_after_rotation_l3416_341698

/-- The equation of a line in the form ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Rotate a line 90 degrees counterclockwise about a given point -/
def rotate90 (l : Line) (p : Point) : Line := sorry

/-- Find the x-intercept of a line -/
def xIntercept (l : Line) : ℝ := sorry

theorem x_intercept_after_rotation :
  let l : Line := { a := 2, b := -3, c := 30 }
  let p : Point := { x := 15, y := 10 }
  let k' := rotate90 l p
  xIntercept k' = 65 / 3 := by sorry

end NUMINAMATH_CALUDE_x_intercept_after_rotation_l3416_341698


namespace NUMINAMATH_CALUDE_evaluate_expression_l3416_341622

theorem evaluate_expression : 6 - 9 * (10 - 4^2) * 5 = -264 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3416_341622


namespace NUMINAMATH_CALUDE_replacement_concentration_theorem_l3416_341649

/-- Calculates the concentration of a chemical solution after partial replacement -/
def resulting_concentration (initial_conc : ℝ) (replacement_conc : ℝ) (replaced_fraction : ℝ) : ℝ :=
  (initial_conc * (1 - replaced_fraction) + replacement_conc * replaced_fraction)

theorem replacement_concentration_theorem :
  let initial_conc : ℝ := 0.85
  let replacement_conc : ℝ := 0.30
  let replaced_fraction : ℝ := 0.8181818181818182
  resulting_concentration initial_conc replacement_conc replaced_fraction = 0.40 := by
sorry

end NUMINAMATH_CALUDE_replacement_concentration_theorem_l3416_341649


namespace NUMINAMATH_CALUDE_x_squared_minus_y_squared_l3416_341603

theorem x_squared_minus_y_squared (x y : ℚ) 
  (h1 : x + y = 5 / 11) (h2 : x - y = 1 / 77) : x^2 - y^2 = 5 / 847 := by
  sorry

end NUMINAMATH_CALUDE_x_squared_minus_y_squared_l3416_341603


namespace NUMINAMATH_CALUDE_simplify_fraction_calculate_logarithmic_expression_l3416_341691

-- Part 1
theorem simplify_fraction (a : ℝ) (ha : a > 0) :
  a^2 / (Real.sqrt a * 3 * a^2) = a^(5/6) := by sorry

-- Part 2
theorem calculate_logarithmic_expression :
  (2 * Real.log 2 + Real.log 3) / (1 + 1/2 * Real.log 0.36 + 1/3 * Real.log 8) = 1 := by sorry

end NUMINAMATH_CALUDE_simplify_fraction_calculate_logarithmic_expression_l3416_341691


namespace NUMINAMATH_CALUDE_gcd_factorial_eight_and_factorial_six_squared_l3416_341669

theorem gcd_factorial_eight_and_factorial_six_squared :
  Nat.gcd (Nat.factorial 8) ((Nat.factorial 6)^2) = 5760 := by sorry

end NUMINAMATH_CALUDE_gcd_factorial_eight_and_factorial_six_squared_l3416_341669


namespace NUMINAMATH_CALUDE_workshop_workers_l3416_341676

/-- The total number of workers in a workshop with given salary conditions -/
theorem workshop_workers (avg_salary : ℕ) (tech_count : ℕ) (tech_salary : ℕ) (non_tech_salary : ℕ) :
  avg_salary = 8000 →
  tech_count = 7 →
  tech_salary = 12000 →
  non_tech_salary = 6000 →
  ∃ (total_workers : ℕ), 
    (tech_count * tech_salary + (total_workers - tech_count) * non_tech_salary) / total_workers = avg_salary ∧
    total_workers = 21 := by
  sorry

end NUMINAMATH_CALUDE_workshop_workers_l3416_341676


namespace NUMINAMATH_CALUDE_correct_calculation_l3416_341634

theorem correct_calculation (a b : ℝ) : -7 * a * b^2 + 4 * a * b^2 = -3 * a * b^2 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l3416_341634


namespace NUMINAMATH_CALUDE_infinitely_many_squares_l3416_341683

/-- An arithmetic sequence of positive integers -/
def ArithmeticSequence (a d : ℕ) : ℕ → ℕ := fun n => a + n * d

/-- A number is a perfect square -/
def IsPerfectSquare (n : ℕ) : Prop := ∃ k : ℕ, n = k * k

theorem infinitely_many_squares
  (a d : ℕ) -- First term and common difference
  (h_pos : ∀ n, 0 < ArithmeticSequence a d n) -- Sequence is positive
  (h_square : ∃ n, IsPerfectSquare (ArithmeticSequence a d n)) -- At least one square exists
  : ∀ m : ℕ, ∃ n > m, IsPerfectSquare (ArithmeticSequence a d n) :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_squares_l3416_341683


namespace NUMINAMATH_CALUDE_samuel_remaining_amount_samuel_remaining_amount_proof_l3416_341659

theorem samuel_remaining_amount 
  (total : ℕ) 
  (samuel_fraction : ℚ) 
  (spent_fraction : ℚ) 
  (h1 : total = 240) 
  (h2 : samuel_fraction = 3/4) 
  (h3 : spent_fraction = 1/5) : 
  ℕ :=
  let samuel_received : ℚ := total * samuel_fraction
  let samuel_spent : ℚ := total * spent_fraction
  let samuel_remaining : ℚ := samuel_received - samuel_spent
  132

theorem samuel_remaining_amount_proof 
  (total : ℕ) 
  (samuel_fraction : ℚ) 
  (spent_fraction : ℚ) 
  (h1 : total = 240) 
  (h2 : samuel_fraction = 3/4) 
  (h3 : spent_fraction = 1/5) : 
  samuel_remaining_amount total samuel_fraction spent_fraction h1 h2 h3 = 132 := by
  sorry

end NUMINAMATH_CALUDE_samuel_remaining_amount_samuel_remaining_amount_proof_l3416_341659


namespace NUMINAMATH_CALUDE_circle_equation_l3416_341640

-- Define the circle C
def Circle (a : ℝ) := {(x, y) : ℝ × ℝ | (x - a)^2 + y^2 = 4}

-- Define the tangent line
def TangentLine := {(x, y) : ℝ × ℝ | 3*x + 4*y + 4 = 0}

-- Theorem statement
theorem circle_equation (a : ℝ) (h1 : a > 0) 
  (h2 : ∃ (p : ℝ × ℝ), p ∈ Circle a ∧ p ∈ TangentLine) : 
  Circle a = Circle 2 := by
sorry

end NUMINAMATH_CALUDE_circle_equation_l3416_341640


namespace NUMINAMATH_CALUDE_smallest_dual_base_palindrome_l3416_341632

/-- A function to check if a number is a palindrome in a given base -/
def isPalindromeInBase (n : ℕ) (base : ℕ) : Prop :=
  ∃ (digits : List ℕ), n = digits.foldl (λ acc d => acc * base + d) 0 ∧ digits = digits.reverse

/-- The theorem stating that 105 is the smallest natural number greater than 20 
    that is a palindrome in both base 14 and base 20 -/
theorem smallest_dual_base_palindrome :
  ∀ (N : ℕ), N > 20 → isPalindromeInBase N 14 → isPalindromeInBase N 20 → N ≥ 105 :=
by
  sorry

#check smallest_dual_base_palindrome

end NUMINAMATH_CALUDE_smallest_dual_base_palindrome_l3416_341632


namespace NUMINAMATH_CALUDE_club_truncator_probability_l3416_341630

/-- The number of matches Club Truncator plays -/
def num_matches : ℕ := 8

/-- The probability of winning, losing, or tying a single match -/
def single_match_prob : ℚ := 1/3

/-- The probability of finishing with more wins than losses -/
def more_wins_prob : ℚ := 2741/6561

theorem club_truncator_probability :
  let total_outcomes := 3^num_matches
  let same_wins_losses := 1079
  (total_outcomes - same_wins_losses) / (2 * total_outcomes) = more_wins_prob :=
sorry

end NUMINAMATH_CALUDE_club_truncator_probability_l3416_341630


namespace NUMINAMATH_CALUDE_incompatible_inequalities_l3416_341625

theorem incompatible_inequalities :
  ¬∃ (a b c d : ℝ), 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧
    a + b < c + d ∧
    (a + b) * (c + d) < a * b + c * d ∧
    (a + b) * c * d < a * b * (c + d) := by
  sorry

end NUMINAMATH_CALUDE_incompatible_inequalities_l3416_341625


namespace NUMINAMATH_CALUDE_circle_passes_through_origin_l3416_341674

/-- A circle is defined by its center (a, b) and radius r -/
structure Circle where
  a : ℝ
  b : ℝ
  r : ℝ

/-- A point is defined by its coordinates (x, y) -/
structure Point where
  x : ℝ
  y : ℝ

/-- The origin is the point (0, 0) -/
def origin : Point := ⟨0, 0⟩

/-- A point (x, y) is on a circle if and only if (x-a)^2 + (y-b)^2 = r^2 -/
def isOnCircle (p : Point) (c : Circle) : Prop :=
  (p.x - c.a)^2 + (p.y - c.b)^2 = c.r^2

/-- Theorem: A circle passes through the origin if and only if a^2 + b^2 = r^2 -/
theorem circle_passes_through_origin (c : Circle) :
  isOnCircle origin c ↔ c.a^2 + c.b^2 = c.r^2 := by
  sorry

end NUMINAMATH_CALUDE_circle_passes_through_origin_l3416_341674


namespace NUMINAMATH_CALUDE_angle_bisector_d_value_l3416_341611

-- Define the triangle ABC
def A : ℝ × ℝ := (2, 3)
def B : ℝ × ℝ := (-4, -2)
def C : ℝ × ℝ := (7, -1)

-- Define the angle bisector equation
def angleBisectorEq (x y d : ℝ) : Prop := x - 3*y + d = 0

-- Theorem statement
theorem angle_bisector_d_value :
  ∃ d : ℝ, (∀ x y : ℝ, angleBisectorEq x y d ↔ 
    (∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧
      x = B.1 + t * (C.1 - B.1) ∧
      y = B.2 + t * (C.2 - B.2))) ∧
    angleBisectorEq B.1 B.2 d :=
by sorry

end NUMINAMATH_CALUDE_angle_bisector_d_value_l3416_341611


namespace NUMINAMATH_CALUDE_multiply_by_number_l3416_341608

theorem multiply_by_number (x : ℝ) (n : ℝ) : x = 5 → x * n = (16 - x) + 4 → n = 3 := by
  sorry

end NUMINAMATH_CALUDE_multiply_by_number_l3416_341608


namespace NUMINAMATH_CALUDE_percentage_change_difference_l3416_341609

theorem percentage_change_difference (initial_yes initial_no final_yes final_no : ℚ) :
  initial_yes = 60 / 100 →
  initial_no = 40 / 100 →
  final_yes = 80 / 100 →
  final_no = 20 / 100 →
  ∃ (min_change max_change : ℚ),
    min_change ≥ 0 ∧
    max_change ≥ 0 ∧
    min_change ≤ max_change ∧
    max_change - min_change = 20 / 100 :=
by sorry

end NUMINAMATH_CALUDE_percentage_change_difference_l3416_341609


namespace NUMINAMATH_CALUDE_prom_attendance_l3416_341675

theorem prom_attendance (total_students : ℕ) (couples : ℕ) (solo_students : ℕ) : 
  total_students = 123 → couples = 60 → solo_students = total_students - 2 * couples →
  solo_students = 3 := by
  sorry

end NUMINAMATH_CALUDE_prom_attendance_l3416_341675


namespace NUMINAMATH_CALUDE_drone_velocity_at_3_seconds_l3416_341629

-- Define the displacement function
def h (t : ℝ) : ℝ := 15 * t - t^2

-- Define the velocity function as the derivative of the displacement function
def v (t : ℝ) : ℝ := 15 - 2 * t

-- Theorem statement
theorem drone_velocity_at_3_seconds :
  v 3 = 9 := by sorry

end NUMINAMATH_CALUDE_drone_velocity_at_3_seconds_l3416_341629


namespace NUMINAMATH_CALUDE_race_catch_up_time_l3416_341605

/-- Given a 10-mile race with two runners, where the first runner's pace is 8 minutes per mile
    and the second runner's pace is 7 minutes per mile, prove that if the second runner stops
    after 56 minutes, they can remain stopped for 8 minutes before the first runner catches up. -/
theorem race_catch_up_time (race_length : ℝ) (pace1 pace2 stop_time : ℝ) :
  race_length = 10 →
  pace1 = 8 →
  pace2 = 7 →
  stop_time = 56 →
  let distance1 := stop_time / pace1
  let distance2 := stop_time / pace2
  let distance_diff := distance2 - distance1
  distance_diff * pace1 = 8 := by sorry

end NUMINAMATH_CALUDE_race_catch_up_time_l3416_341605


namespace NUMINAMATH_CALUDE_unpartnered_students_correct_l3416_341692

/-- Calculates the number of students unable to partner in square dancing --/
def unpartnered_students (class1_males class1_females class2_males class2_females class3_males class3_females : ℕ) : ℕ :=
  let total_males := class1_males + class2_males + class3_males
  let total_females := class1_females + class2_females + class3_females
  Int.natAbs (total_males - total_females)

/-- Theorem stating that the number of unpartnered students is correct --/
theorem unpartnered_students_correct 
  (class1_males class1_females class2_males class2_females class3_males class3_females : ℕ) :
  unpartnered_students class1_males class1_females class2_males class2_females class3_males class3_females =
  Int.natAbs ((class1_males + class2_males + class3_males) - (class1_females + class2_females + class3_females)) :=
by sorry

#eval unpartnered_students 17 13 14 18 15 17  -- Should evaluate to 2

end NUMINAMATH_CALUDE_unpartnered_students_correct_l3416_341692


namespace NUMINAMATH_CALUDE_coordinate_and_vector_problem_l3416_341681

-- Define the points and vectors
def A : ℝ × ℝ := (1, 0)
def B : ℝ × ℝ := (-2, 1)  -- Calculated from |OB| = √5 and x = -2
def O : ℝ × ℝ := (0, 0)

-- Define the rotation function
def rotate90Clockwise (p : ℝ × ℝ) : ℝ × ℝ := (p.2, -p.1)

-- Define the vector OP
def OP : ℝ × ℝ := (2, 6)  -- Calculated from |OP| = 2√10 and cos θ = √10/10

-- Define the theorem
theorem coordinate_and_vector_problem :
  let C := rotate90Clockwise (B.1 - O.1, B.2 - O.2)
  let x := ((OP.1 * B.2) - (OP.2 * B.1)) / ((A.1 * B.2) - (A.2 * B.1))
  let y := ((OP.1 * A.2) - (OP.2 * A.1)) / ((B.1 * A.2) - (B.2 * A.1))
  C = (1, 2) ∧ x + y = 8 := by
  sorry

end NUMINAMATH_CALUDE_coordinate_and_vector_problem_l3416_341681


namespace NUMINAMATH_CALUDE_pyramid_volume_l3416_341602

/-- A cube ABCDEFGH with volume 8 -/
structure Cube :=
  (volume : ℝ)
  (is_cube : volume = 8)

/-- Pyramid ACDH within the cube ABCDEFGH -/
def pyramid (c : Cube) : ℝ := sorry

/-- Theorem: The volume of pyramid ACDH is 4/3 -/
theorem pyramid_volume (c : Cube) : pyramid c = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_pyramid_volume_l3416_341602


namespace NUMINAMATH_CALUDE_new_average_weight_l3416_341685

theorem new_average_weight (weight_A weight_D : ℝ) : 
  weight_A = 73 →
  (weight_A + (150 - weight_A)) / 3 = 50 →
  ((150 - weight_A) + weight_D + (weight_D + 3)) / 4 = 51 →
  (weight_A + (150 - weight_A) + weight_D) / 4 = 53 :=
by sorry

end NUMINAMATH_CALUDE_new_average_weight_l3416_341685


namespace NUMINAMATH_CALUDE_election_winner_percentage_l3416_341652

theorem election_winner_percentage (total_votes : ℕ) (winner_votes : ℕ) (margin : ℕ) : 
  (total_votes = winner_votes + (winner_votes - margin)) →
  (winner_votes = 650) →
  (margin = 300) →
  (winner_votes : ℚ) / (total_votes : ℚ) = 13/20 :=
by sorry

end NUMINAMATH_CALUDE_election_winner_percentage_l3416_341652


namespace NUMINAMATH_CALUDE_triangle_theorem_l3416_341644

noncomputable def triangle_problem (a b c : ℝ) (A B C : ℝ) : Prop :=
  let condition1 := 2 * a * Real.cos B = b + 2 * c
  let condition2 := 2 * Real.sin C + Real.tan A * Real.cos B + Real.sin B = 0
  let condition3 := (a - c) / Real.sin B = (b + c) / (Real.sin A + Real.sin C)
  b = 2 ∧ c = 4 ∧ 
  (condition1 ∨ condition2 ∨ condition3) →
  A = 2 * Real.pi / 3 ∧
  ∃ (D : ℝ × ℝ), 
    let BC := Real.sqrt ((b - c * Real.cos A)^2 + (c * Real.sin A)^2)
    let BD := BC / 4
    let AD := Real.sqrt (((3/4) * b)^2 + ((1/4) * c)^2 + 
               (3/4) * b * (1/4) * c * Real.cos A)
    AD = Real.sqrt 31 / 2

theorem triangle_theorem : 
  ∀ (a b c A B C : ℝ), triangle_problem a b c A B C :=
sorry

end NUMINAMATH_CALUDE_triangle_theorem_l3416_341644


namespace NUMINAMATH_CALUDE_min_value_theorem_l3416_341620

theorem min_value_theorem (m n : ℝ) (hm : m > 0) (hn : n > 0) (h_sum : m + 2*n = 1) :
  ((m + 1) * (n + 1)) / (m * n) ≥ 8 + 4 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3416_341620


namespace NUMINAMATH_CALUDE_billys_age_l3416_341601

theorem billys_age (billy joe : ℕ) 
  (h1 : billy = 3 * joe) 
  (h2 : billy + joe = 60) : 
  billy = 45 := by
sorry

end NUMINAMATH_CALUDE_billys_age_l3416_341601


namespace NUMINAMATH_CALUDE_number_divided_by_three_l3416_341677

theorem number_divided_by_three : ∃ n : ℝ, n / 3 = 10 ∧ n = 30 := by
  sorry

end NUMINAMATH_CALUDE_number_divided_by_three_l3416_341677


namespace NUMINAMATH_CALUDE_odd_induction_l3416_341610

theorem odd_induction (P : ℕ → Prop) 
  (base : P 1) 
  (step : ∀ k : ℕ, k ≥ 1 → P k → P (k + 2)) : 
  ∀ n : ℕ, n > 0 ∧ Odd n → P n :=
sorry

end NUMINAMATH_CALUDE_odd_induction_l3416_341610


namespace NUMINAMATH_CALUDE_divisibility_equivalence_l3416_341607

theorem divisibility_equivalence (m n : ℕ+) :
  83 ∣ (25 * m + 3 * n) ↔ 83 ∣ (3 * m + 7 * n) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_equivalence_l3416_341607


namespace NUMINAMATH_CALUDE_water_bottles_profit_l3416_341642

def water_bottles_problem (total_bottles : ℕ) (standard_rate_bottles : ℕ) (standard_rate_price : ℚ)
  (discount_threshold : ℕ) (discount_rate : ℚ) (selling_rate_bottles : ℕ) (selling_rate_price : ℚ) : Prop :=
  let standard_price_per_bottle : ℚ := standard_rate_price / standard_rate_bottles
  let total_cost_without_discount : ℚ := total_bottles * standard_price_per_bottle
  let total_cost_with_discount : ℚ := total_cost_without_discount * (1 - discount_rate)
  let selling_price_per_bottle : ℚ := selling_rate_price / selling_rate_bottles
  let total_revenue : ℚ := total_bottles * selling_price_per_bottle
  let profit : ℚ := total_revenue - total_cost_with_discount
  (total_bottles > discount_threshold) ∧ (profit = 325)

theorem water_bottles_profit :
  water_bottles_problem 1500 6 3 1200 (1/10) 3 2 :=
sorry

end NUMINAMATH_CALUDE_water_bottles_profit_l3416_341642


namespace NUMINAMATH_CALUDE_function_properties_l3416_341621

-- Define the function f(x) = ax^3 + bx^2
def f (x : ℝ) : ℝ := -6 * x^3 + 9 * x^2

-- State the theorem
theorem function_properties :
  (f 1 = 3) ∧ 
  (deriv f 1 = 0) ∧ 
  (∀ x : ℝ, f x ≥ 0) ∧
  (∃ x : ℝ, f x = 0) := by
  sorry

end NUMINAMATH_CALUDE_function_properties_l3416_341621


namespace NUMINAMATH_CALUDE_triangle_side_calculation_l3416_341688

theorem triangle_side_calculation (A B C : Real) (a b c : Real) :
  -- Triangle ABC exists
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π →
  -- a, b, c are sides opposite to angles A, B, C respectively
  a > 0 ∧ b > 0 ∧ c > 0 →
  -- Given conditions
  a = Real.sqrt 3 →
  Real.sin B = 1/2 →
  C = π/6 →
  -- Conclusion
  b = 1 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_calculation_l3416_341688


namespace NUMINAMATH_CALUDE_satisfactory_fraction_is_25_29_l3416_341695

/-- Represents the grade distribution in a school science class -/
structure GradeDistribution :=
  (a : Nat) -- number of A grades
  (b : Nat) -- number of B grades
  (c : Nat) -- number of C grades
  (d : Nat) -- number of D grades
  (f : Nat) -- number of F grades

/-- Calculates the fraction of satisfactory grades -/
def satisfactoryFraction (gd : GradeDistribution) : Rat :=
  let satisfactory := gd.a + gd.b + gd.c + gd.d
  let total := satisfactory + gd.f
  satisfactory / total

/-- The main theorem stating that the fraction of satisfactory grades is 25/29 -/
theorem satisfactory_fraction_is_25_29 :
  let gd : GradeDistribution := ⟨8, 7, 6, 4, 4⟩
  satisfactoryFraction gd = 25 / 29 := by
  sorry

end NUMINAMATH_CALUDE_satisfactory_fraction_is_25_29_l3416_341695


namespace NUMINAMATH_CALUDE_incorrect_proposition_statement_l3416_341653

theorem incorrect_proposition_statement : 
  ¬(∀ (p q : Prop), (p ∧ q = False) → (p = False ∧ q = False)) := by
  sorry

end NUMINAMATH_CALUDE_incorrect_proposition_statement_l3416_341653


namespace NUMINAMATH_CALUDE_problem_solution_l3416_341636

theorem problem_solution (p q : ℤ) 
  (h1 : p > 1) 
  (h2 : q > 1) 
  (h3 : ∃ k : ℤ, (2 * p - 1) = k * q) 
  (h4 : ∃ m : ℤ, (2 * q - 1) = m * p) : 
  p + q = 8 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l3416_341636


namespace NUMINAMATH_CALUDE_half_MN_coord_l3416_341648

def OM : Fin 2 → ℝ := ![(-2), 3]
def ON : Fin 2 → ℝ := ![(-1), (-5)]

theorem half_MN_coord : 
  (1/2 : ℝ) • (ON - OM) = ![(1/2), (-4)] := by sorry

end NUMINAMATH_CALUDE_half_MN_coord_l3416_341648


namespace NUMINAMATH_CALUDE_second_smallest_divisible_by_all_less_than_9_sum_of_digits_l3416_341689

def is_divisible_by_all_less_than_9 (n : ℕ) : Prop :=
  ∀ k : ℕ, 0 < k ∧ k < 9 → n % k = 0

def second_smallest (P : ℕ → Prop) (n : ℕ) : Prop :=
  P n ∧ ∃ m : ℕ, P m ∧ m < n ∧ ∀ k : ℕ, P k → k = m ∨ n ≤ k

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

theorem second_smallest_divisible_by_all_less_than_9_sum_of_digits :
  ∃ N : ℕ, second_smallest is_divisible_by_all_less_than_9 N ∧ sum_of_digits N = 15 := by
  sorry

end NUMINAMATH_CALUDE_second_smallest_divisible_by_all_less_than_9_sum_of_digits_l3416_341689


namespace NUMINAMATH_CALUDE_ellipse_major_axis_length_l3416_341646

/-- Given an ellipse with equation x²/a² + y²/b² = 1 (a > b > 0), 
    where its right focus is at (1,0) and b²/a = 2, 
    prove that the length of its major axis is 2√2 + 2. -/
theorem ellipse_major_axis_length 
  (a b : ℝ) 
  (h1 : a > b) 
  (h2 : b > 0) 
  (h3 : b^2 / a = 2) : 
  2 * a = 2 * Real.sqrt 2 + 2 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_major_axis_length_l3416_341646


namespace NUMINAMATH_CALUDE_original_class_size_l3416_341604

theorem original_class_size (initial_avg : ℝ) (new_students : ℕ) (new_avg : ℝ) (avg_decrease : ℝ) :
  initial_avg = 40 →
  new_students = 12 →
  new_avg = 32 →
  avg_decrease = 4 →
  ∃ (original_size : ℕ),
    (original_size * initial_avg + new_students * new_avg) / (original_size + new_students) = initial_avg - avg_decrease ∧
    original_size = 12 := by
  sorry

end NUMINAMATH_CALUDE_original_class_size_l3416_341604


namespace NUMINAMATH_CALUDE_sum_of_ninth_powers_l3416_341673

/-- Given two real numbers a and b satisfying certain conditions, 
    prove that a^9 + b^9 = 76 -/
theorem sum_of_ninth_powers (a b : ℝ) 
  (h1 : a + b = 1)
  (h2 : a^2 + b^2 = 3)
  (h3 : a^3 + b^3 = 4)
  (h4 : a^4 + b^4 = 7)
  (h5 : a^5 + b^5 = 11)
  (h_rec : ∀ n ≥ 3, a^n + b^n = (a^(n-1) + b^(n-1)) + (a^(n-2) + b^(n-2))) :
  a^9 + b^9 = 76 := by
  sorry

#check sum_of_ninth_powers

end NUMINAMATH_CALUDE_sum_of_ninth_powers_l3416_341673


namespace NUMINAMATH_CALUDE_square_sum_equality_l3416_341631

theorem square_sum_equality : 784 + 2 * 14 * 7 + 49 = 1225 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_equality_l3416_341631


namespace NUMINAMATH_CALUDE_parabola_equation_l3416_341651

/-- Given a parabola y^2 = 2px with a point P(2, y_0) on it, and the distance from P to the directrix is 4,
    prove that p = 4 and the standard equation of the parabola is y^2 = 8x. -/
theorem parabola_equation (p : ℝ) (y_0 : ℝ) (h1 : p > 0) (h2 : y_0^2 = 2*p*2) (h3 : p/2 + 2 = 4) :
  p = 4 ∧ ∀ x y, y^2 = 8*x ↔ y^2 = 2*p*x := by sorry

end NUMINAMATH_CALUDE_parabola_equation_l3416_341651


namespace NUMINAMATH_CALUDE_min_words_for_certification_l3416_341641

theorem min_words_for_certification (total_words : ℕ) (min_score : ℚ) : 
  total_words = 800 → 
  min_score = 9/10 → 
  ∃ (words_to_learn : ℕ), 
    (words_to_learn : ℚ) / total_words ≥ min_score ∧ 
    ∀ (w : ℕ), (w : ℚ) / total_words ≥ min_score → w ≥ words_to_learn ∧
    words_to_learn = 720 := by
  sorry

end NUMINAMATH_CALUDE_min_words_for_certification_l3416_341641


namespace NUMINAMATH_CALUDE_b_zero_iff_f_even_l3416_341682

-- Define the function f
def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define what it means for f to be even
def is_even (a b c : ℝ) : Prop :=
  ∀ x, f a b c x = f a b c (-x)

-- State the theorem
theorem b_zero_iff_f_even (a b c : ℝ) :
  b = 0 ↔ is_even a b c :=
sorry

end NUMINAMATH_CALUDE_b_zero_iff_f_even_l3416_341682


namespace NUMINAMATH_CALUDE_bullet_train_length_l3416_341694

/-- The length of a bullet train passing a man running in the opposite direction -/
theorem bullet_train_length (train_speed : ℝ) (man_speed : ℝ) (passing_time : ℝ) :
  train_speed = 59 →
  man_speed = 7 →
  passing_time = 12 →
  (train_speed + man_speed) * (1000 / 3600) * passing_time = 220 :=
by sorry

end NUMINAMATH_CALUDE_bullet_train_length_l3416_341694


namespace NUMINAMATH_CALUDE_train_tunnel_time_l3416_341647

/-- The time taken for a train to pass through a tunnel -/
theorem train_tunnel_time (train_length : ℝ) (train_speed_kmh : ℝ) (tunnel_length : ℝ) : 
  train_length = 100 →
  train_speed_kmh = 72 →
  tunnel_length = 1.1 →
  (train_length + tunnel_length * 1000) / (train_speed_kmh * 1000 / 3600) / 60 = 1 := by
  sorry

end NUMINAMATH_CALUDE_train_tunnel_time_l3416_341647


namespace NUMINAMATH_CALUDE_machine_production_l3416_341614

/-- Given the production rate of 6 machines, calculate the production of 8 machines in 4 minutes -/
theorem machine_production 
  (rate : ℕ) -- Production rate per minute for 6 machines
  (h1 : rate = 270) -- 6 machines produce 270 bottles per minute
  : (8 * 4 * (rate / 6) : ℕ) = 1440 := by
  sorry

end NUMINAMATH_CALUDE_machine_production_l3416_341614


namespace NUMINAMATH_CALUDE_food_expenditure_increase_l3416_341655

/-- Represents the annual income in thousand yuan -/
def annual_income : ℝ → ℝ := id

/-- Represents the annual food expenditure in thousand yuan -/
def annual_food_expenditure (x : ℝ) : ℝ := 2.5 * x + 3.2

/-- Theorem stating that when annual income increases by 1, 
    annual food expenditure increases by 2.5 -/
theorem food_expenditure_increase (x : ℝ) : 
  annual_food_expenditure (annual_income x + 1) - annual_food_expenditure (annual_income x) = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_food_expenditure_increase_l3416_341655


namespace NUMINAMATH_CALUDE_shortest_distance_point_to_parabola_l3416_341686

/-- The shortest distance between the point (3,6) and the parabola x = y^2/4 is √5. -/
theorem shortest_distance_point_to_parabola :
  let point := (3, 6)
  let parabola := {(x, y) : ℝ × ℝ | x = y^2 / 4}
  (∃ (d : ℝ), d = Real.sqrt 5 ∧
    ∀ (p : ℝ × ℝ), p ∈ parabola →
      d ≤ Real.sqrt ((p.1 - point.1)^2 + (p.2 - point.2)^2)) :=
by sorry

end NUMINAMATH_CALUDE_shortest_distance_point_to_parabola_l3416_341686


namespace NUMINAMATH_CALUDE_monthly_salary_calculation_l3416_341656

/-- Proves that a man's monthly salary is 5750 Rs. given the specified conditions -/
theorem monthly_salary_calculation (savings_rate : ℝ) (expense_increase : ℝ) (new_savings : ℝ) : 
  savings_rate = 0.20 →
  expense_increase = 0.20 →
  new_savings = 230 →
  ∃ (salary : ℝ), salary = 5750 ∧ 
    (1 - savings_rate - expense_increase * (1 - savings_rate)) * salary = new_savings :=
by sorry

end NUMINAMATH_CALUDE_monthly_salary_calculation_l3416_341656


namespace NUMINAMATH_CALUDE_tangent_sum_problem_l3416_341624

theorem tangent_sum_problem (p q : ℝ) 
  (h1 : (Real.sin p / Real.cos q) + (Real.sin q / Real.cos p) = 2)
  (h2 : (Real.cos p / Real.sin q) + (Real.cos q / Real.sin p) = 3) :
  (Real.tan p / Real.tan q) + (Real.tan q / Real.tan p) = 8/5 := by
  sorry

end NUMINAMATH_CALUDE_tangent_sum_problem_l3416_341624


namespace NUMINAMATH_CALUDE_counterexample_exists_l3416_341633

theorem counterexample_exists : ∃ (a b : ℝ), (a + b < 0) ∧ ¬(a < 0 ∧ b < 0) := by
  sorry

end NUMINAMATH_CALUDE_counterexample_exists_l3416_341633


namespace NUMINAMATH_CALUDE_square_grid_perimeter_l3416_341658

/-- The perimeter of a 3x3 grid of congruent squares with a total area of 576 square centimeters is 192 centimeters. -/
theorem square_grid_perimeter (total_area : ℝ) (side_length : ℝ) (perimeter : ℝ) : 
  total_area = 576 →
  side_length * side_length * 9 = total_area →
  perimeter = 4 * 3 * side_length →
  perimeter = 192 := by
sorry

end NUMINAMATH_CALUDE_square_grid_perimeter_l3416_341658
