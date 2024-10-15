import Mathlib

namespace NUMINAMATH_CALUDE_root_comparison_l837_83770

theorem root_comparison (m n : ℕ) : 
  min ((n : ℝ) ^ (1 / m : ℝ)) ((m : ℝ) ^ (1 / n : ℝ)) ≤ (3 : ℝ) ^ (1 / 3 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_root_comparison_l837_83770


namespace NUMINAMATH_CALUDE_rohan_monthly_salary_l837_83709

/-- Rohan's monthly expenses and savings --/
structure RohanFinances where
  food_percent : ℝ
  rent_percent : ℝ
  entertainment_percent : ℝ
  conveyance_percent : ℝ
  taxes_percent : ℝ
  miscellaneous_percent : ℝ
  savings : ℝ

/-- Theorem: Rohan's monthly salary calculation --/
theorem rohan_monthly_salary (r : RohanFinances) 
  (h1 : r.food_percent = 0.40)
  (h2 : r.rent_percent = 0.20)
  (h3 : r.entertainment_percent = 0.10)
  (h4 : r.conveyance_percent = 0.10)
  (h5 : r.taxes_percent = 0.05)
  (h6 : r.miscellaneous_percent = 0.07)
  (h7 : r.savings = 1000) :
  ∃ (salary : ℝ), salary = 12500 ∧ 
    (1 - (r.food_percent + r.rent_percent + r.entertainment_percent + 
          r.conveyance_percent + r.taxes_percent + r.miscellaneous_percent)) * salary = r.savings :=
by sorry


end NUMINAMATH_CALUDE_rohan_monthly_salary_l837_83709


namespace NUMINAMATH_CALUDE_probability_not_black_ball_l837_83795

theorem probability_not_black_ball (white black red : ℕ) 
  (h_white : white = 8) 
  (h_black : black = 9) 
  (h_red : red = 3) : 
  (white + red) / (white + black + red : ℚ) = 11 / 20 := by
  sorry

end NUMINAMATH_CALUDE_probability_not_black_ball_l837_83795


namespace NUMINAMATH_CALUDE_syllogism_correctness_l837_83775

theorem syllogism_correctness : 
  (∀ n : ℕ, (n : ℤ) = n) →  -- All natural numbers are integers
  (4 : ℕ) = 4 →             -- 4 is a natural number
  (4 : ℤ) = 4               -- Therefore, 4 is an integer
  := by sorry

end NUMINAMATH_CALUDE_syllogism_correctness_l837_83775


namespace NUMINAMATH_CALUDE_zero_not_in_range_of_g_l837_83729

noncomputable def g (x : ℝ) : ℤ :=
  if x > -3 then Int.ceil (1 / (x + 3))
  else Int.floor (1 / (x + 3))

theorem zero_not_in_range_of_g :
  ¬ ∃ (x : ℝ), g x = 0 :=
sorry

end NUMINAMATH_CALUDE_zero_not_in_range_of_g_l837_83729


namespace NUMINAMATH_CALUDE_certain_number_proof_l837_83736

theorem certain_number_proof (x : ℝ) : (1.68 * x) / 6 = 354.2 ↔ x = 1265 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l837_83736


namespace NUMINAMATH_CALUDE_function_properties_l837_83737

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x / 4 + a / x - Real.log x - 3 / 2

-- Define the derivative of f
def f_derivative (a : ℝ) (x : ℝ) : ℝ := 1 / 4 - a / (x^2) - 1 / x

-- State the theorem
theorem function_properties :
  ∃ (a : ℝ),
    -- The tangent at (1, f(1)) is perpendicular to y = (1/2)x
    f_derivative a 1 = -2 ∧
    -- a = 5/4
    a = 5 / 4 ∧
    -- f(x) is decreasing on (0, 5) and increasing on (5, +∞)
    (∀ x ∈ Set.Ioo 0 5, f_derivative a x < 0) ∧
    (∀ x ∈ Set.Ioi 5, f_derivative a x > 0) ∧
    -- The minimum value of f(x) is -ln(5) at x = 5
    (∀ x > 0, f a x ≥ f a 5) ∧
    f a 5 = -Real.log 5 :=
by
  sorry

end

end NUMINAMATH_CALUDE_function_properties_l837_83737


namespace NUMINAMATH_CALUDE_infinitely_many_divisors_l837_83783

theorem infinitely_many_divisors (a : ℕ) :
  Set.Infinite {n : ℕ | n ∣ a^(n - a + 1) - 1} :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_divisors_l837_83783


namespace NUMINAMATH_CALUDE_salary_increase_comparison_l837_83742

theorem salary_increase_comparison (initial_salary : ℝ) (h : initial_salary > 0) :
  let first_worker_new_salary := 2 * initial_salary
  let second_worker_new_salary := 1.5 * initial_salary
  (first_worker_new_salary - second_worker_new_salary) / second_worker_new_salary = 1 / 3 := by
sorry

end NUMINAMATH_CALUDE_salary_increase_comparison_l837_83742


namespace NUMINAMATH_CALUDE_envelope_width_l837_83711

/-- Given a rectangular envelope with length 4 inches and area 16 square inches, prove its width is 4 inches. -/
theorem envelope_width (length : ℝ) (area : ℝ) (width : ℝ) 
  (h1 : length = 4)
  (h2 : area = 16)
  (h3 : area = length * width) : 
  width = 4 := by
  sorry

end NUMINAMATH_CALUDE_envelope_width_l837_83711


namespace NUMINAMATH_CALUDE_power_product_positive_l837_83788

theorem power_product_positive (m n : ℕ) (hm : m > 2) :
  ∃ k : ℕ+, (2^m - 1) * (2^n + 1) = k := by
  sorry

end NUMINAMATH_CALUDE_power_product_positive_l837_83788


namespace NUMINAMATH_CALUDE_triangle_area_proof_l837_83718

-- Define the slopes of the two lines
def slope1 : ℝ := 3
def slope2 : ℝ := -1

-- Define the intersection point
def intersection_point : ℝ × ℝ := (5, 3)

-- Define the equation of the third line
def third_line (x y : ℝ) : Prop := x + y = 4

-- Define the area of the triangle
def triangle_area : ℝ := 4

-- Theorem statement
theorem triangle_area_proof :
  ∃ (A B C : ℝ × ℝ),
    -- A is on the line with slope1 and passes through intersection_point
    (A.2 - intersection_point.2 = slope1 * (A.1 - intersection_point.1)) ∧
    -- B is on the line with slope2 and passes through intersection_point
    (B.2 - intersection_point.2 = slope2 * (B.1 - intersection_point.1)) ∧
    -- C is on the third line
    third_line C.1 C.2 ∧
    -- The area of the triangle formed by A, B, and C is equal to triangle_area
    abs ((A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)) / 2) = triangle_area :=
sorry

end NUMINAMATH_CALUDE_triangle_area_proof_l837_83718


namespace NUMINAMATH_CALUDE_number_problem_l837_83780

theorem number_problem (x : ℝ) : 0.3 * x - 70 = 20 → x = 300 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l837_83780


namespace NUMINAMATH_CALUDE_parallel_lines_length_l837_83755

/-- Represents a line segment with a length -/
structure LineSegment where
  length : ℝ

/-- Represents a set of parallel line segments -/
structure ParallelLines where
  ab : LineSegment
  cd : LineSegment
  ef : LineSegment
  gh : LineSegment

/-- 
Given parallel lines AB, CD, EF, and GH, where DC = 120 cm and AB = 180 cm, 
the length of GH is 72 cm.
-/
theorem parallel_lines_length (lines : ParallelLines) 
  (h1 : lines.cd.length = 120)
  (h2 : lines.ab.length = 180) :
  lines.gh.length = 72 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_length_l837_83755


namespace NUMINAMATH_CALUDE_expression_factorization_l837_83772

theorem expression_factorization (a b c : ℝ) :
  a^3 * (b^2 - c^2) + b^3 * (c^2 - b^2) + c^3 * (a^2 - b^2) =
  (a - b) * (b - c) * (c - a) * (a*b + a*c + b*c) :=
by sorry

end NUMINAMATH_CALUDE_expression_factorization_l837_83772


namespace NUMINAMATH_CALUDE_work_completion_rate_l837_83744

/-- Given that A can finish a work in 18 days and B can do the same work in half the time taken by A,
    prove that A and B working together can finish 1/6 of the work in a day. -/
theorem work_completion_rate (days_A : ℕ) (days_B : ℕ) : 
  days_A = 18 →
  days_B = days_A / 2 →
  (1 : ℚ) / days_A + (1 : ℚ) / days_B = (1 : ℚ) / 6 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_rate_l837_83744


namespace NUMINAMATH_CALUDE_sum_90_to_99_l837_83760

/-- The sum of consecutive integers from 90 to 99 is equal to 945. -/
theorem sum_90_to_99 : (Finset.range 10).sum (fun i => i + 90) = 945 := by
  sorry

end NUMINAMATH_CALUDE_sum_90_to_99_l837_83760


namespace NUMINAMATH_CALUDE_sin_alpha_value_l837_83797

theorem sin_alpha_value (α : Real) 
  (h : (Real.sqrt 2 / 2) * (Real.sin (α / 2) - Real.cos (α / 2)) = Real.sqrt 6 / 3) : 
  Real.sin α = -1/3 := by
  sorry

end NUMINAMATH_CALUDE_sin_alpha_value_l837_83797


namespace NUMINAMATH_CALUDE_berry_reading_problem_l837_83790

theorem berry_reading_problem (pages_per_day : ℕ) (days_in_week : ℕ) 
  (pages_sun : ℕ) (pages_mon : ℕ) (pages_tue : ℕ) (pages_wed : ℕ) 
  (pages_fri : ℕ) (pages_sat : ℕ) :
  pages_per_day = 50 →
  days_in_week = 7 →
  pages_sun = 43 →
  pages_mon = 65 →
  pages_tue = 28 →
  pages_wed = 0 →
  pages_fri = 56 →
  pages_sat = 88 →
  ∃ pages_thu : ℕ, 
    pages_thu = pages_per_day * days_in_week - 
      (pages_sun + pages_mon + pages_tue + pages_wed + pages_fri + pages_sat) ∧
    pages_thu = 70 :=
by
  sorry

end NUMINAMATH_CALUDE_berry_reading_problem_l837_83790


namespace NUMINAMATH_CALUDE_ab_value_l837_83740

theorem ab_value (a b : ℝ) (h1 : a - b = 3) (h2 : a^2 + b^2 = 29) : a * b = 10 := by
  sorry

end NUMINAMATH_CALUDE_ab_value_l837_83740


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l837_83796

theorem regular_polygon_sides : ∃ n : ℕ, n > 2 ∧ n - (n * (n - 3) / 4) = 0 → n = 7 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l837_83796


namespace NUMINAMATH_CALUDE_least_addition_for_divisibility_l837_83784

theorem least_addition_for_divisibility (n : ℕ) : 
  ∃ (x : ℕ), x ≤ 3 ∧ (1202 + x) % 4 = 0 ∧ ∀ (y : ℕ), y < x → (1202 + y) % 4 ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_least_addition_for_divisibility_l837_83784


namespace NUMINAMATH_CALUDE_divisibility_by_48_l837_83798

theorem divisibility_by_48 (a b c : ℤ) (h1 : a < c) (h2 : a^2 + c^2 = 2*b^2) :
  ∃ k : ℤ, c^2 - a^2 = 48 * k := by
sorry

end NUMINAMATH_CALUDE_divisibility_by_48_l837_83798


namespace NUMINAMATH_CALUDE_problem_solution_l837_83768

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | -1 ≤ x ∧ x ≤ 1}
def B : Set ℝ := {x : ℝ | -2 ≤ x ∧ x ≤ 2}

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := 2 * x^2 + m * x - 1

-- Define the set C
def C (m : ℝ) : Set ℝ := {x : ℝ | f m x ≤ 0}

-- Define the function g
def g (a : ℝ) (m : ℝ) (x : ℝ) : ℝ := 2 * |x - a| - x^2 - m * x

theorem problem_solution :
  (∀ m : ℝ, C m ⊆ (A ∩ B) ↔ -1 ≤ m ∧ m ≤ 1) ∧
  (∀ x : ℝ, f (-4) (1 - x) = f (-4) (1 + x) →
    Set.range (fun x => f (-4) x) ∩ B = {y : ℝ | -3 ≤ y ∧ y ≤ 15}) ∧
  (∀ a : ℝ, 
    (a ≤ -1 → ∀ x : ℝ, f (-4) x + g a (-4) x ≥ -2*a - 2) ∧
    (-1 < a ∧ a < 1 → ∀ x : ℝ, f (-4) x + g a (-4) x ≥ a^2 - 1) ∧
    (1 ≤ a → ∀ x : ℝ, f (-4) x + g a (-4) x ≥ 2*a - 2)) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l837_83768


namespace NUMINAMATH_CALUDE_profit_growth_equation_l837_83764

/-- Represents the profit growth of a supermarket over a 2-month period -/
theorem profit_growth_equation (initial_profit : ℝ) (final_profit : ℝ) (growth_rate : ℝ) :
  initial_profit = 5000 →
  final_profit = 7200 →
  initial_profit * (1 + growth_rate)^2 = final_profit :=
by sorry

end NUMINAMATH_CALUDE_profit_growth_equation_l837_83764


namespace NUMINAMATH_CALUDE_sum_of_roots_l837_83704

theorem sum_of_roots (a b : ℝ) : 
  (∀ x : ℝ, (x + a) * (x + b) = x^2 + 4*x + 3) → a + b = 4 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_l837_83704


namespace NUMINAMATH_CALUDE_inequality_proof_l837_83712

theorem inequality_proof (a b c : ℝ) :
  (a^2 + 1) * (b^2 + 1) * (c^2 + 1) - (a*b + b*c + c*a - 1)^2 ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l837_83712


namespace NUMINAMATH_CALUDE_hexagon_diagonal_length_l837_83705

/-- The length of a diagonal in a regular hexagon --/
theorem hexagon_diagonal_length (side_length : ℝ) (h : side_length = 12) :
  let diagonal_length := side_length * Real.sqrt 3
  diagonal_length = 12 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_diagonal_length_l837_83705


namespace NUMINAMATH_CALUDE_rotation_cycle_implies_equilateral_l837_83728

/-- A point in the plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A triangle defined by three points -/
structure Triangle where
  A₁ : Point
  A₂ : Point
  A₃ : Point

/-- Rotation of a point around a center by an angle -/
def rotate (center : Point) (angle : ℝ) (p : Point) : Point :=
  sorry

/-- Check if a triangle is equilateral -/
def Triangle.isEquilateral (t : Triangle) : Prop :=
  sorry

/-- The sequence of points A_s -/
def A (s : ℕ) : Point :=
  sorry

/-- The sequence of points P_k -/
def P (k : ℕ) : Point :=
  sorry

/-- The main theorem -/
theorem rotation_cycle_implies_equilateral 
  (t : Triangle) (P₀ : Point) : 
  (P 1986 = P₀) → Triangle.isEquilateral t :=
sorry

end NUMINAMATH_CALUDE_rotation_cycle_implies_equilateral_l837_83728


namespace NUMINAMATH_CALUDE_theater_ticket_sales_l837_83726

theorem theater_ticket_sales (total_tickets : ℕ) (adult_price senior_price : ℕ) (senior_tickets : ℕ) : 
  total_tickets = 529 →
  adult_price = 25 →
  senior_price = 15 →
  senior_tickets = 348 →
  (total_tickets - senior_tickets) * adult_price + senior_tickets * senior_price = 9745 :=
by
  sorry

end NUMINAMATH_CALUDE_theater_ticket_sales_l837_83726


namespace NUMINAMATH_CALUDE_smaller_number_problem_l837_83756

theorem smaller_number_problem (a b : ℝ) (h1 : a + b = 15) (h2 : 3 * (a - b) = 21) : 
  min a b = 4 := by
  sorry

end NUMINAMATH_CALUDE_smaller_number_problem_l837_83756


namespace NUMINAMATH_CALUDE_cross_number_puzzle_l837_83701

theorem cross_number_puzzle :
  ∃! (a1 a3 d1 d2 : ℕ),
    (100 ≤ a1 ∧ a1 < 1000) ∧
    (100 ≤ a3 ∧ a3 < 1000) ∧
    (100 ≤ d1 ∧ d1 < 1000) ∧
    (100 ≤ d2 ∧ d2 < 1000) ∧
    (∃ n : ℕ, a1 = n^2) ∧
    (∃ n : ℕ, a3 = n^4) ∧
    (∃ n : ℕ, d1 = 2 * n^5) ∧
    (∃ n : ℕ, d2 = n^3) ∧
    (a1 / 100 = 4) :=
by
  sorry

end NUMINAMATH_CALUDE_cross_number_puzzle_l837_83701


namespace NUMINAMATH_CALUDE_system_is_linear_l837_83778

-- Define a linear equation in two variables
def is_linear_equation (f : ℝ → ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, ∀ x y : ℝ, f x y = a * x + b * y + c

-- Define the system of equations
def equation1 (x y : ℝ) : ℝ := x + y - 2
def equation2 (x y : ℝ) : ℝ := x - 2 * y

-- Theorem statement
theorem system_is_linear :
  is_linear_equation equation1 ∧ is_linear_equation equation2 :=
sorry

end NUMINAMATH_CALUDE_system_is_linear_l837_83778


namespace NUMINAMATH_CALUDE_flag_puzzle_l837_83774

theorem flag_puzzle (x : ℝ) : 
  (8 * 5 : ℝ) + (10 * 7 : ℝ) + (x * 5 : ℝ) = (15 * 9 : ℝ) → x = 5 := by
sorry

end NUMINAMATH_CALUDE_flag_puzzle_l837_83774


namespace NUMINAMATH_CALUDE_collins_total_petals_l837_83791

/-- The number of petals Collin has after receiving flowers from Ingrid -/
theorem collins_total_petals (collins_initial_flowers ingrid_flowers petals_per_flower : ℕ) : 
  collins_initial_flowers = 25 →
  ingrid_flowers = 33 →
  petals_per_flower = 4 →
  (collins_initial_flowers + ingrid_flowers / 3) * petals_per_flower = 144 := by
  sorry

#check collins_total_petals

end NUMINAMATH_CALUDE_collins_total_petals_l837_83791


namespace NUMINAMATH_CALUDE_largest_modulus_root_real_part_l837_83721

theorem largest_modulus_root_real_part 
  (z : ℂ) 
  (hz : 5 * z^4 + 10 * z^3 + 10 * z^2 + 5 * z + 1 = 0) 
  (hmax : ∀ w : ℂ, 5 * w^4 + 10 * w^3 + 10 * w^2 + 5 * w + 1 = 0 → Complex.abs w ≤ Complex.abs z) :
  z.re = -1/2 :=
sorry

end NUMINAMATH_CALUDE_largest_modulus_root_real_part_l837_83721


namespace NUMINAMATH_CALUDE_paulines_garden_capacity_l837_83749

/-- Represents Pauline's garden -/
structure Garden where
  tomato_kinds : ℕ
  tomatoes_per_kind : ℕ
  cucumber_kinds : ℕ
  cucumbers_per_kind : ℕ
  potatoes : ℕ
  rows : ℕ
  spaces_per_row : ℕ

/-- Calculates the number of additional vegetables that can be planted in the garden -/
def additional_vegetables (g : Garden) : ℕ :=
  g.rows * g.spaces_per_row - 
  (g.tomato_kinds * g.tomatoes_per_kind + 
   g.cucumber_kinds * g.cucumbers_per_kind + 
   g.potatoes)

/-- Theorem stating that in Pauline's specific garden, 85 more vegetables can be planted -/
theorem paulines_garden_capacity :
  ∃ (g : Garden), 
    g.tomato_kinds = 3 ∧ 
    g.tomatoes_per_kind = 5 ∧ 
    g.cucumber_kinds = 5 ∧ 
    g.cucumbers_per_kind = 4 ∧ 
    g.potatoes = 30 ∧ 
    g.rows = 10 ∧ 
    g.spaces_per_row = 15 ∧ 
    additional_vegetables g = 85 := by
  sorry

end NUMINAMATH_CALUDE_paulines_garden_capacity_l837_83749


namespace NUMINAMATH_CALUDE_todd_ate_cupcakes_l837_83787

theorem todd_ate_cupcakes (initial_cupcakes : ℕ) (packages : ℕ) (cupcakes_per_package : ℕ) : 
  initial_cupcakes = 18 →
  packages = 5 →
  cupcakes_per_package = 2 →
  initial_cupcakes - packages * cupcakes_per_package = 8 :=
by sorry

end NUMINAMATH_CALUDE_todd_ate_cupcakes_l837_83787


namespace NUMINAMATH_CALUDE_tangent_slope_at_point_l837_83724

/-- The slope of the tangent line to y = x^3 - 4x at (1, -1) is -1 -/
theorem tangent_slope_at_point : 
  let f (x : ℝ) := x^3 - 4*x
  let x₀ : ℝ := 1
  let y₀ : ℝ := -1
  (deriv f) x₀ = -1 := by sorry

end NUMINAMATH_CALUDE_tangent_slope_at_point_l837_83724


namespace NUMINAMATH_CALUDE_mean_proportional_problem_l837_83716

theorem mean_proportional_problem (x : ℝ) :
  (x * 9409 : ℝ).sqrt = 8665 → x = 7981 := by
  sorry

end NUMINAMATH_CALUDE_mean_proportional_problem_l837_83716


namespace NUMINAMATH_CALUDE_base_7_to_base_10_l837_83723

-- Define the base-7 number 435₇
def base_7_435 : ℕ := 4 * 7^2 + 3 * 7 + 5

-- Define the function to convert a three-digit base-10 number to its digits
def to_digits (n : ℕ) : ℕ × ℕ × ℕ :=
  let hundreds := n / 100
  let tens := (n / 10) % 10
  let ones := n % 10
  (hundreds, tens, ones)

-- Theorem statement
theorem base_7_to_base_10 :
  ∃ (c d : ℕ), c < 10 ∧ d < 10 ∧ base_7_435 = 300 + 10 * c + d →
  (c * d : ℚ) / 18 = 2 / 9 := by sorry

end NUMINAMATH_CALUDE_base_7_to_base_10_l837_83723


namespace NUMINAMATH_CALUDE_max_withdrawal_theorem_l837_83700

/-- Represents the possible transactions -/
inductive Transaction
| withdraw : Transaction
| deposit : Transaction

/-- Represents the bank account -/
structure BankAccount where
  balance : ℕ

/-- Applies a transaction to the bank account -/
def applyTransaction (account : BankAccount) (t : Transaction) : BankAccount :=
  match t with
  | Transaction.withdraw => ⟨if account.balance ≥ 300 then account.balance - 300 else account.balance⟩
  | Transaction.deposit => ⟨account.balance + 198⟩

/-- Checks if a sequence of transactions is valid -/
def isValidSequence (initial : ℕ) (transactions : List Transaction) : Prop :=
  let finalAccount := transactions.foldl applyTransaction ⟨initial⟩
  finalAccount.balance ≥ 0

/-- The maximum amount that can be withdrawn -/
def maxWithdrawal (initial : ℕ) : ℕ :=
  initial - (initial % 6)

/-- Theorem stating the maximum withdrawal amount -/
theorem max_withdrawal_theorem (initial : ℕ) :
  initial = 500 →
  maxWithdrawal initial = 300 ∧
  ∃ (transactions : List Transaction), isValidSequence initial transactions ∧
    (initial - (transactions.foldl applyTransaction ⟨initial⟩).balance = maxWithdrawal initial) :=
by sorry

#check max_withdrawal_theorem

end NUMINAMATH_CALUDE_max_withdrawal_theorem_l837_83700


namespace NUMINAMATH_CALUDE_balloon_arrangements_count_l837_83715

/-- The number of distinct arrangements of letters in "balloon" -/
def balloon_arrangements : ℕ :=
  Nat.factorial 7 / (Nat.factorial 2 * Nat.factorial 2)

/-- The word "balloon" has 7 letters -/
axiom balloon_length : balloon_arrangements = Nat.factorial 7 / (Nat.factorial 2 * Nat.factorial 2)

/-- Theorem: The number of distinct arrangements of letters in "balloon" is 1260 -/
theorem balloon_arrangements_count : balloon_arrangements = 1260 := by
  sorry


end NUMINAMATH_CALUDE_balloon_arrangements_count_l837_83715


namespace NUMINAMATH_CALUDE_complex_sum_equals_one_l837_83719

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem complex_sum_equals_one :
  (i + i^3)^100 + (i + i^2 + i^3 + i^4 + i^5)^120 = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_complex_sum_equals_one_l837_83719


namespace NUMINAMATH_CALUDE_family_reunion_count_l837_83773

/-- The number of people at a family reunion -/
def family_reunion_attendance (male_adults female_adults children : ℕ) : ℕ :=
  male_adults + female_adults + children

/-- Theorem stating the total number of people at the family reunion -/
theorem family_reunion_count :
  ∃ (male_adults female_adults children : ℕ),
    male_adults = 100 ∧
    female_adults = male_adults + 50 ∧
    children = 2 * (male_adults + female_adults) ∧
    family_reunion_attendance male_adults female_adults children = 750 :=
by
  sorry


end NUMINAMATH_CALUDE_family_reunion_count_l837_83773


namespace NUMINAMATH_CALUDE_multiple_without_zero_digit_l837_83781

theorem multiple_without_zero_digit (n : ℕ) (hn : n % 10 ≠ 0) :
  ∃ m : ℕ, m > 0 ∧ n ∣ m ∧ ∀ d : ℕ, d < 10 → (m / 10^d) % 10 ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_multiple_without_zero_digit_l837_83781


namespace NUMINAMATH_CALUDE_projectile_max_height_l837_83727

/-- The height of a projectile as a function of time -/
def h (t : ℝ) : ℝ := -18 * t^2 + 72 * t + 25

/-- Theorem: The maximum height reached by the projectile is 97 feet -/
theorem projectile_max_height : 
  ∃ (t_max : ℝ), ∀ (t : ℝ), h t ≤ h t_max ∧ h t_max = 97 := by
  sorry

end NUMINAMATH_CALUDE_projectile_max_height_l837_83727


namespace NUMINAMATH_CALUDE_initial_number_of_men_l837_83794

theorem initial_number_of_men (M : ℕ) (A : ℝ) : 
  (2 * M = 46 - (20 + 10)) →  -- Condition 1 and 2
  (M = 8) :=                  -- Conclusion
by
  sorry  -- Skip the proof

end NUMINAMATH_CALUDE_initial_number_of_men_l837_83794


namespace NUMINAMATH_CALUDE_box_volume_in_cubic_yards_l837_83722

-- Define the conversion factor from feet to yards
def feet_to_yards : ℝ := 3

-- Define the volume of the box in cubic feet
def box_volume_cubic_feet : ℝ := 216

-- Theorem to prove
theorem box_volume_in_cubic_yards :
  box_volume_cubic_feet / (feet_to_yards ^ 3) = 8 := by
  sorry


end NUMINAMATH_CALUDE_box_volume_in_cubic_yards_l837_83722


namespace NUMINAMATH_CALUDE_total_cars_theorem_l837_83739

/-- Calculates the total number of non-defective cars produced by two factories over a week --/
def total_non_defective_cars (factory_a_monday : ℕ) : ℕ :=
  let factory_a_production := [
    factory_a_monday,
    (factory_a_monday * 2 * 95) / 100,  -- Tuesday with 5% defect rate
    factory_a_monday * 4,
    factory_a_monday * 8,
    factory_a_monday * 16
  ]
  let factory_b_production := [
    factory_a_monday * 2,
    factory_a_monday * 4,
    factory_a_monday * 8,
    (factory_a_monday * 16 * 97) / 100,  -- Thursday with 3% defect rate
    factory_a_monday * 32
  ]
  (factory_a_production.sum + factory_b_production.sum)

/-- Theorem stating the total number of non-defective cars produced --/
theorem total_cars_theorem : total_non_defective_cars 60 = 5545 := by
  sorry


end NUMINAMATH_CALUDE_total_cars_theorem_l837_83739


namespace NUMINAMATH_CALUDE_f_increasing_interval_l837_83758

-- Define the function f(x) = 2x³ - ln(x)
noncomputable def f (x : ℝ) : ℝ := 2 * x^3 - Real.log x

-- Theorem statement
theorem f_increasing_interval :
  ∀ x : ℝ, x > 0 → (∀ y : ℝ, y > x → f y > f x) ↔ x > (1/2 : ℝ) :=
sorry

end NUMINAMATH_CALUDE_f_increasing_interval_l837_83758


namespace NUMINAMATH_CALUDE_odd_even_function_sum_l837_83793

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

theorem odd_even_function_sum (f g : ℝ → ℝ) 
  (h_odd : is_odd f) (h_even : is_even g)
  (h_diff : ∀ x, f x - g x = 2 * x^3 + x^2 + 3) :
  f 2 + g 2 = 9 := by
sorry

end NUMINAMATH_CALUDE_odd_even_function_sum_l837_83793


namespace NUMINAMATH_CALUDE_swimming_distance_l837_83769

theorem swimming_distance (x : ℝ) 
  (h1 : x > 0)
  (h2 : (4 * x) / (5 * x) = 4 / 5)
  (h3 : (4 * x - 200) / (5 * x + 100) = 5 / 8) :
  4 * x = 1200 ∧ 5 * x = 1500 := by
  sorry

end NUMINAMATH_CALUDE_swimming_distance_l837_83769


namespace NUMINAMATH_CALUDE_log_4_64_sqrt_4_l837_83741

-- Define the logarithm function
noncomputable def log (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

-- State the theorem
theorem log_4_64_sqrt_4 : log 4 (64 * Real.sqrt 4) = 7/2 := by
  sorry

end NUMINAMATH_CALUDE_log_4_64_sqrt_4_l837_83741


namespace NUMINAMATH_CALUDE_total_wall_length_divisible_by_four_l837_83738

/-- Represents a partition of a square room into smaller square rooms -/
structure RoomPartition where
  size : ℕ  -- Size of the original square room
  partitions : List (ℕ × ℕ × ℕ)  -- List of (x, y, size) for each smaller room

/-- The sum of all partition wall lengths in a room partition -/
def totalWallLength (rp : RoomPartition) : ℕ :=
  sorry

/-- Theorem: The total wall length of any valid room partition is divisible by 4 -/
theorem total_wall_length_divisible_by_four (rp : RoomPartition) :
  4 ∣ totalWallLength rp :=
sorry

end NUMINAMATH_CALUDE_total_wall_length_divisible_by_four_l837_83738


namespace NUMINAMATH_CALUDE_fraction_equals_seven_l837_83771

theorem fraction_equals_seven (x : ℝ) (h : x = 2) : (x^4 + 6*x^2 + 9) / (x^2 + 3) = 7 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equals_seven_l837_83771


namespace NUMINAMATH_CALUDE_order_of_expressions_l837_83779

theorem order_of_expressions : 
  let a : ℝ := (4 : ℝ) ^ (1/10)
  let b : ℝ := Real.log 0.1 / Real.log 4
  let c : ℝ := (0.4 : ℝ) ^ (1/5)
  a > c ∧ c > b := by sorry

end NUMINAMATH_CALUDE_order_of_expressions_l837_83779


namespace NUMINAMATH_CALUDE_custom_op_result_l837_83757

-- Define the custom operation
def custom_op (a b c : ℕ) : ℕ := 
  (a * b * 10000) + (a * c * 100) + (a * (b + c))

-- State the theorem
theorem custom_op_result : custom_op 7 2 5 = 143549 := by
  sorry

end NUMINAMATH_CALUDE_custom_op_result_l837_83757


namespace NUMINAMATH_CALUDE_product_of_numbers_l837_83766

theorem product_of_numbers (x y : ℝ) : x + y = 30 → x^2 + y^2 = 840 → x * y = 30 := by
  sorry

end NUMINAMATH_CALUDE_product_of_numbers_l837_83766


namespace NUMINAMATH_CALUDE_complex_number_real_imag_equal_l837_83767

theorem complex_number_real_imag_equal (a : ℝ) : 
  let x : ℂ := (1 + a * Complex.I) * (2 + Complex.I)
  (x.re = x.im) → a = 1/3 := by sorry

end NUMINAMATH_CALUDE_complex_number_real_imag_equal_l837_83767


namespace NUMINAMATH_CALUDE_xyz_sum_l837_83702

theorem xyz_sum (x y z : ℕ+) 
  (h1 : x * y + z = 47)
  (h2 : y * z + x = 47)
  (h3 : z * x + y = 47) :
  x + y + z = 48 := by
  sorry

end NUMINAMATH_CALUDE_xyz_sum_l837_83702


namespace NUMINAMATH_CALUDE_garden_tulips_percentage_l837_83732

theorem garden_tulips_percentage :
  ∀ (total_flowers : ℕ) (pink_flowers red_flowers pink_roses red_roses pink_tulips red_tulips lilies : ℕ),
    pink_flowers + red_flowers + lilies = total_flowers →
    pink_roses + pink_tulips = pink_flowers →
    red_roses + red_tulips = red_flowers →
    2 * pink_roses = pink_flowers →
    3 * red_tulips = 2 * red_flowers →
    4 * pink_flowers = 3 * total_flowers →
    10 * lilies = total_flowers →
    100 * (pink_tulips + red_tulips) = 61 * total_flowers :=
by
  sorry

end NUMINAMATH_CALUDE_garden_tulips_percentage_l837_83732


namespace NUMINAMATH_CALUDE_oscar_elmer_difference_l837_83751

-- Define the given constants
def elmer_strides_per_gap : ℕ := 44
def oscar_leaps_per_gap : ℕ := 12
def total_poles : ℕ := 41
def total_distance : ℕ := 5280

-- Define the theorem
theorem oscar_elmer_difference : 
  let gaps := total_poles - 1
  let elmer_total_strides := elmer_strides_per_gap * gaps
  let oscar_total_leaps := oscar_leaps_per_gap * gaps
  let elmer_stride_length := total_distance / elmer_total_strides
  let oscar_leap_length := total_distance / oscar_total_leaps
  oscar_leap_length - elmer_stride_length = 8 := by
  sorry

end NUMINAMATH_CALUDE_oscar_elmer_difference_l837_83751


namespace NUMINAMATH_CALUDE_gardening_project_total_cost_l837_83792

def gardening_project_cost (rose_bushes : ℕ) (rose_bush_cost : ℕ) 
  (gardener_hourly_rate : ℕ) (hours_per_day : ℕ) (days_worked : ℕ)
  (soil_volume : ℕ) (soil_cost_per_unit : ℕ) : ℕ :=
  rose_bushes * rose_bush_cost + 
  gardener_hourly_rate * hours_per_day * days_worked +
  soil_volume * soil_cost_per_unit

theorem gardening_project_total_cost : 
  gardening_project_cost 20 150 30 5 4 100 5 = 4100 := by
  sorry

end NUMINAMATH_CALUDE_gardening_project_total_cost_l837_83792


namespace NUMINAMATH_CALUDE_min_colors_100x100_board_l837_83725

/-- Represents a board with cells divided into triangles -/
structure Board :=
  (size : Nat)
  (cells_divided : Bool)

/-- Represents a coloring of the triangles on the board -/
def Coloring := Board → Nat → Nat → Bool → Nat

/-- Checks if a coloring is valid (no adjacent triangles have the same color) -/
def is_valid_coloring (b : Board) (c : Coloring) : Prop := sorry

/-- The minimum number of colors needed for a valid coloring -/
def min_colors (b : Board) : Nat := sorry

/-- Theorem stating the minimum number of colors for a 100x100 board -/
theorem min_colors_100x100_board :
  ∀ (b : Board),
    b.size = 100 ∧
    b.cells_divided →
    min_colors b = 8 := by sorry

end NUMINAMATH_CALUDE_min_colors_100x100_board_l837_83725


namespace NUMINAMATH_CALUDE_square_sum_equals_twice_square_l837_83731

theorem square_sum_equals_twice_square (a : ℝ) : a^2 + a^2 = 2 * a^2 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_equals_twice_square_l837_83731


namespace NUMINAMATH_CALUDE_expr_is_monomial_of_degree_3_l837_83733

/-- A monomial is an algebraic expression consisting of one term. This term can be a constant, a variable, or a product of constants and variables raised to whole number powers. -/
def is_monomial (e : Expr) : Prop := sorry

/-- The degree of a monomial is the sum of the exponents of all its variables. -/
def monomial_degree (e : Expr) : ℕ := sorry

/-- An algebraic expression. -/
inductive Expr
| const : ℚ → Expr
| var : String → Expr
| mul : Expr → Expr → Expr
| pow : Expr → ℕ → Expr

/-- The expression -x^2y -/
def expr : Expr :=
  Expr.mul (Expr.const (-1))
    (Expr.mul (Expr.pow (Expr.var "x") 2) (Expr.var "y"))

theorem expr_is_monomial_of_degree_3 :
  is_monomial expr ∧ monomial_degree expr = 3 := by sorry

end NUMINAMATH_CALUDE_expr_is_monomial_of_degree_3_l837_83733


namespace NUMINAMATH_CALUDE_temperature_at_3pm_l837_83750

-- Define the temperature function
def T (t : ℝ) : ℝ := t^3 - 3*t + 60

-- State the theorem
theorem temperature_at_3pm : T 3 = 78 := by
  sorry

end NUMINAMATH_CALUDE_temperature_at_3pm_l837_83750


namespace NUMINAMATH_CALUDE_forgotten_and_doubled_sum_l837_83710

/-- The sum of the first n positive integers -/
def sum_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

theorem forgotten_and_doubled_sum (luke_sum carissa_sum : ℕ) : 
  sum_first_n 20 = 210 →
  luke_sum = 207 →
  carissa_sum = 225 →
  (sum_first_n 20 - luke_sum) + (carissa_sum - sum_first_n 20) = 18 := by
  sorry

end NUMINAMATH_CALUDE_forgotten_and_doubled_sum_l837_83710


namespace NUMINAMATH_CALUDE_container_capacity_container_capacity_proof_l837_83708

theorem container_capacity : ℝ → Prop :=
  fun capacity =>
    capacity > 0 ∧
    0.4 * capacity + 28 = 0.75 * capacity →
    capacity = 80

-- The proof is omitted
theorem container_capacity_proof : ∃ (capacity : ℝ), container_capacity capacity :=
  sorry

end NUMINAMATH_CALUDE_container_capacity_container_capacity_proof_l837_83708


namespace NUMINAMATH_CALUDE_f_symmetric_l837_83747

def f (a b x : ℝ) : ℝ := x^5 + a*x^3 + b*x + 1

theorem f_symmetric (a b : ℝ) : f a b (-2) = 10 → f a b 2 = -10 := by
  sorry

end NUMINAMATH_CALUDE_f_symmetric_l837_83747


namespace NUMINAMATH_CALUDE_max_value_of_expression_l837_83789

theorem max_value_of_expression (m : ℝ) : 
  (4 - |2 - m|) ≤ 4 ∧ ∃ m : ℝ, 4 - |2 - m| = 4 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l837_83789


namespace NUMINAMATH_CALUDE_sum_odd_integers_eq_1040_l837_83734

/-- The sum of odd integers from 15 to 65, inclusive -/
def sum_odd_integers : ℕ :=
  let first := 15
  let last := 65
  let n := (last - first) / 2 + 1
  n * (first + last) / 2

theorem sum_odd_integers_eq_1040 : sum_odd_integers = 1040 := by
  sorry

end NUMINAMATH_CALUDE_sum_odd_integers_eq_1040_l837_83734


namespace NUMINAMATH_CALUDE_group_age_calculation_l837_83735

theorem group_age_calculation (total_members : ℕ) (total_average_age : ℚ) (zero_age_members : ℕ) : 
  total_members = 50 →
  total_average_age = 5 →
  zero_age_members = 10 →
  let non_zero_members : ℕ := total_members - zero_age_members
  let total_age : ℚ := total_members * total_average_age
  let non_zero_average_age : ℚ := total_age / non_zero_members
  non_zero_average_age = 25/4 := by
sorry

#eval (25 : ℚ) / 4  -- This should output 6.25

end NUMINAMATH_CALUDE_group_age_calculation_l837_83735


namespace NUMINAMATH_CALUDE_repeating_decimal_to_fraction_l837_83730

theorem repeating_decimal_to_fraction :
  ∃ (n d : ℕ), d ≠ 0 ∧ gcd n d = 1 ∧ (n : ℚ) / d = 0.4 + (36 : ℚ) / 99 :=
by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_to_fraction_l837_83730


namespace NUMINAMATH_CALUDE_existence_of_m_and_n_l837_83786

theorem existence_of_m_and_n :
  ∃ (m n : ℕ) (a b : ℝ), (-2 * a^n * b^n)^m + (3 * a^m * b^m)^n = a^6 * b^6 := by
  sorry

end NUMINAMATH_CALUDE_existence_of_m_and_n_l837_83786


namespace NUMINAMATH_CALUDE_circle_equation_tangent_lines_center_x_range_l837_83714

-- Define the circle C
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define line l
def line_l (x : ℝ) : ℝ := 2 * x - 4

-- Define the point A
def point_A : ℝ × ℝ := (0, 3)

-- Define the circle C
def circle_C : Circle :=
  { center := (3, 2), radius := 1 }

-- Theorem 1
theorem circle_equation (C : Circle) (h1 : C.center.2 = line_l C.center.1) 
  (h2 : C.center.2 = -C.center.1 + 5) (h3 : C.radius = 1) :
  ∀ x y, (x - C.center.1)^2 + (y - C.center.2)^2 = 1 ↔ (x - 3)^2 + (y - 2)^2 = 1 :=
sorry

-- Theorem 2
theorem tangent_lines (C : Circle) (h : ∀ x y, (x - C.center.1)^2 + (y - C.center.2)^2 = 1 ↔ (x - 3)^2 + (y - 2)^2 = 1) :
  (∀ x, x = 3) ∨ (∀ x y, 3*x + 4*y - 12 = 0) :=
sorry

-- Theorem 3
theorem center_x_range (C : Circle) 
  (h : ∃ M : ℝ × ℝ, (M.1 - C.center.1)^2 + (M.2 - C.center.2)^2 = C.radius^2 ∧ 
                    (M.1 - point_A.1)^2 + (M.2 - point_A.2)^2 = M.1^2 + M.2^2) :
  9/4 ≤ C.center.1 ∧ C.center.1 ≤ 13/4 :=
sorry

end NUMINAMATH_CALUDE_circle_equation_tangent_lines_center_x_range_l837_83714


namespace NUMINAMATH_CALUDE_sum_remainder_five_l837_83743

theorem sum_remainder_five (n : ℤ) : ((5 - n) + (n + 4)) % 5 = 4 := by
  sorry

end NUMINAMATH_CALUDE_sum_remainder_five_l837_83743


namespace NUMINAMATH_CALUDE_ways_A_to_C_via_B_l837_83785

/-- The number of ways to get from point A to point B -/
def ways_AB : ℕ := 3

/-- The number of ways to get from point B to point C -/
def ways_BC : ℕ := 4

/-- The total number of ways to get from point A to point C via point B -/
def total_ways : ℕ := ways_AB * ways_BC

theorem ways_A_to_C_via_B : total_ways = 12 := by
  sorry

end NUMINAMATH_CALUDE_ways_A_to_C_via_B_l837_83785


namespace NUMINAMATH_CALUDE_triangle_third_side_bounds_l837_83763

theorem triangle_third_side_bounds (a b : ℝ) (ha : a = 7) (hb : b = 11) :
  let c_min := Int.ceil (max (b - a) (a - b))
  let c_max := Int.floor (a + b - 1)
  (c_min = 5 ∧ c_max = 17) := by sorry

end NUMINAMATH_CALUDE_triangle_third_side_bounds_l837_83763


namespace NUMINAMATH_CALUDE_cakes_sold_l837_83754

theorem cakes_sold (total : ℕ) (left : ℕ) (sold : ℕ) 
  (h1 : total = 54)
  (h2 : left = 13)
  (h3 : sold = total - left) : sold = 41 := by
  sorry

end NUMINAMATH_CALUDE_cakes_sold_l837_83754


namespace NUMINAMATH_CALUDE_interest_rate_proof_l837_83748

def simple_interest (P r t : ℝ) : ℝ := P * (1 + r * t)

theorem interest_rate_proof (P : ℝ) (h1 : P > 0) :
  ∃ r : ℝ, r > 0 ∧ simple_interest P r 2 = 100 ∧ simple_interest P r 6 = 200 →
  r = 0.5 := by
sorry

end NUMINAMATH_CALUDE_interest_rate_proof_l837_83748


namespace NUMINAMATH_CALUDE_inequality_proof_l837_83720

theorem inequality_proof (a b c d : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0) 
  (h_eq : (a + 1)⁻¹ + (b + 1)⁻¹ + (c + 1)⁻¹ + (d + 1)⁻¹ = 3) : 
  (a * b * c)^(1/3) + (b * c * d)^(1/3) + (c * d * a)^(1/3) + (d * a * b)^(1/3) ≤ 4/3 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l837_83720


namespace NUMINAMATH_CALUDE_functional_equation_solutions_l837_83745

/-- The functional equation problem -/
def FunctionalEquation (t : ℝ) (f : ℝ → ℝ) : Prop :=
  t ≠ -1 ∧ ∀ x y : ℝ, (t + 1) * f (1 + x * y) - f (x + y) = f (x + 1) * f (y + 1)

/-- The set of solutions to the functional equation -/
def Solutions (t : ℝ) (f : ℝ → ℝ) : Prop :=
  (∀ x, f x = 0) ∨ (∀ x, f x = t) ∨ (∀ x, f x = (t + 1) * x - (t + 2))

/-- The main theorem: all solutions to the functional equation -/
theorem functional_equation_solutions (t : ℝ) (f : ℝ → ℝ) :
  FunctionalEquation t f ↔ Solutions t f := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_solutions_l837_83745


namespace NUMINAMATH_CALUDE_point_p_coordinates_l837_83761

/-- A point P with coordinates (m+3, m-1) that lies on the y-axis -/
structure PointP where
  m : ℝ
  x : ℝ := m + 3
  y : ℝ := m - 1
  on_y_axis : x = 0

/-- Theorem: If a point P(m+3, m-1) lies on the y-axis, then its coordinates are (0, -4) -/
theorem point_p_coordinates (P : PointP) : (P.x = 0 ∧ P.y = -4) := by
  sorry

end NUMINAMATH_CALUDE_point_p_coordinates_l837_83761


namespace NUMINAMATH_CALUDE_fried_chicken_cost_is_12_l837_83753

/-- Calculates the cost of a fried chicken bucket given budget information and beef purchase details. -/
def fried_chicken_cost (total_budget : ℕ) (amount_left : ℕ) (beef_quantity : ℕ) (beef_price : ℕ) : ℕ :=
  total_budget - amount_left - beef_quantity * beef_price

/-- Proves that the cost of the fried chicken bucket is $12 given the problem conditions. -/
theorem fried_chicken_cost_is_12 :
  fried_chicken_cost 80 53 5 3 = 12 := by
  sorry

end NUMINAMATH_CALUDE_fried_chicken_cost_is_12_l837_83753


namespace NUMINAMATH_CALUDE_sector_area_l837_83765

/-- Given a sector with perimeter 16 cm and central angle 2 radians, its area is 16 cm² -/
theorem sector_area (perimeter : ℝ) (central_angle : ℝ) (area : ℝ) : 
  perimeter = 16 → central_angle = 2 → area = (1/2) * central_angle * ((perimeter / (2 + central_angle))^2) → area = 16 := by
  sorry

end NUMINAMATH_CALUDE_sector_area_l837_83765


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l837_83759

theorem min_value_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (h_sum : a + 2*b = 2) :
  (1/a + 1/b) ≥ 3/2 + Real.sqrt 2 ∧ 
  ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ a₀ + 2*b₀ = 2 ∧ 1/a₀ + 1/b₀ = 3/2 + Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l837_83759


namespace NUMINAMATH_CALUDE_work_completion_theorem_l837_83777

/-- Represents the work completion scenario -/
structure WorkCompletion where
  initial_men : ℕ
  initial_hours_per_day : ℕ
  initial_days : ℕ
  new_men : ℕ
  new_days : ℕ

/-- Calculates the hours per day for the new workforce -/
def hours_per_day (w : WorkCompletion) : ℚ :=
  (w.initial_men * w.initial_hours_per_day * w.initial_days : ℚ) / (w.new_men * w.new_days)

theorem work_completion_theorem (w : WorkCompletion) 
    (h1 : w.initial_men = 10)
    (h2 : w.initial_hours_per_day = 7)
    (h3 : w.initial_days = 18)
    (h4 : w.new_days = 12)
    (h5 : w.new_men > 10) :
    hours_per_day w = 1260 / (12 * w.new_men) := by
  sorry

end NUMINAMATH_CALUDE_work_completion_theorem_l837_83777


namespace NUMINAMATH_CALUDE_room_height_from_curtain_l837_83717

/-- The height of a room from the curtain rod to the floor, given curtain length and pooling material. -/
theorem room_height_from_curtain (curtain_length : ℕ) (pooling_material : ℕ) : 
  curtain_length = 101 ∧ pooling_material = 5 → curtain_length - pooling_material = 96 := by
  sorry

#check room_height_from_curtain

end NUMINAMATH_CALUDE_room_height_from_curtain_l837_83717


namespace NUMINAMATH_CALUDE_max_min_difference_2a_minus_b_l837_83703

theorem max_min_difference_2a_minus_b : 
  ∃ (max min : ℝ), 
    (∀ a b : ℝ, a^2 + b^2 - 2*a - 4 = 0 → 2*a - b ≤ max) ∧
    (∀ a b : ℝ, a^2 + b^2 - 2*a - 4 = 0 → 2*a - b ≥ min) ∧
    (∃ a1 b1 a2 b2 : ℝ, 
      a1^2 + b1^2 - 2*a1 - 4 = 0 ∧
      a2^2 + b2^2 - 2*a2 - 4 = 0 ∧
      2*a1 - b1 = max ∧
      2*a2 - b2 = min) ∧
    max - min = 10 :=
by sorry

end NUMINAMATH_CALUDE_max_min_difference_2a_minus_b_l837_83703


namespace NUMINAMATH_CALUDE_best_fit_model_l837_83776

/-- Represents a regression model with its coefficient of determination (R²) -/
structure RegressionModel where
  r_squared : ℝ
  h_r_squared_nonneg : 0 ≤ r_squared
  h_r_squared_le_one : r_squared ≤ 1

/-- Determines if one model has a better fit than another based on R² -/
def better_fit (m1 m2 : RegressionModel) : Prop :=
  m1.r_squared > m2.r_squared

theorem best_fit_model (m1 m2 m3 m4 : RegressionModel)
  (h1 : m1.r_squared = 0.87)
  (h2 : m2.r_squared = 0.97)
  (h3 : m3.r_squared = 0.50)
  (h4 : m4.r_squared = 0.25) :
  better_fit m2 m1 ∧ better_fit m2 m3 ∧ better_fit m2 m4 := by
  sorry

end NUMINAMATH_CALUDE_best_fit_model_l837_83776


namespace NUMINAMATH_CALUDE_flour_bags_comparison_l837_83752

theorem flour_bags_comparison (W : ℝ) : 
  (W > 0) →
  let remaining_first := W - W / 3
  let remaining_second := W - 1000 / 3
  (W > 1000 → remaining_second > remaining_first) ∧
  (W = 1000 → remaining_second = remaining_first) ∧
  (W < 1000 → remaining_first > remaining_second) :=
by sorry

end NUMINAMATH_CALUDE_flour_bags_comparison_l837_83752


namespace NUMINAMATH_CALUDE_lindas_wallet_l837_83707

theorem lindas_wallet (total_amount : ℕ) (total_bills : ℕ) (five_dollar_bills : ℕ) (ten_dollar_bills : ℕ) :
  total_amount = 100 →
  total_bills = 15 →
  five_dollar_bills + ten_dollar_bills = total_bills →
  5 * five_dollar_bills + 10 * ten_dollar_bills = total_amount →
  five_dollar_bills = 10 :=
by sorry

end NUMINAMATH_CALUDE_lindas_wallet_l837_83707


namespace NUMINAMATH_CALUDE_sum_odd_when_sum_of_squares_odd_l837_83799

theorem sum_odd_when_sum_of_squares_odd (n m : ℤ) (h : Odd (n^2 + m^2)) : Odd (n + m) := by
  sorry

end NUMINAMATH_CALUDE_sum_odd_when_sum_of_squares_odd_l837_83799


namespace NUMINAMATH_CALUDE_count_factors_of_eight_squared_nine_cubed_seven_fifth_l837_83746

theorem count_factors_of_eight_squared_nine_cubed_seven_fifth (n : Nat) :
  n = 8^2 * 9^3 * 7^5 →
  (Finset.filter (λ m : Nat => n % m = 0) (Finset.range (n + 1))).card = 294 :=
by sorry

end NUMINAMATH_CALUDE_count_factors_of_eight_squared_nine_cubed_seven_fifth_l837_83746


namespace NUMINAMATH_CALUDE_count_between_multiples_l837_83782

def multiples_of_4 : List Nat := List.filter (fun n => n % 4 = 0) (List.range 100)

def fifth_from_left : Nat := multiples_of_4[4]

def eighth_from_right : Nat := multiples_of_4[multiples_of_4.length - 8]

theorem count_between_multiples :
  (List.filter (fun n => n > fifth_from_left ∧ n < eighth_from_right) multiples_of_4).length = 11 := by
  sorry

end NUMINAMATH_CALUDE_count_between_multiples_l837_83782


namespace NUMINAMATH_CALUDE_chess_tournament_boys_l837_83713

theorem chess_tournament_boys (n : ℕ) (k : ℚ) : 
  n > 2 →  -- There are more than 2 boys
  (6 : ℚ) + n * k = (n + 2) * (n + 1) / 2 →  -- Total points equation
  (∀ m : ℕ, m > 2 ∧ m ≠ n → (6 : ℚ) + m * ((m + 2) * (m + 1) / 2 - 6) / m ≠ (m + 2) * (m + 1) / 2) →  -- n is the only solution > 2
  n = 5 ∨ n = 10 :=
by sorry

end NUMINAMATH_CALUDE_chess_tournament_boys_l837_83713


namespace NUMINAMATH_CALUDE_complex_root_magnitude_one_iff_divisible_by_six_l837_83706

theorem complex_root_magnitude_one_iff_divisible_by_six (n : ℕ) :
  (∃ z : ℂ, z^(n+1) - z^n - 1 = 0 ∧ Complex.abs z = 1) ↔ (∃ k : ℤ, n + 2 = 6 * k) :=
sorry

end NUMINAMATH_CALUDE_complex_root_magnitude_one_iff_divisible_by_six_l837_83706


namespace NUMINAMATH_CALUDE_cloth_sale_quantity_l837_83762

/-- Proves that the number of metres of cloth sold is 300 given the specified conditions --/
theorem cloth_sale_quantity (total_selling_price : ℕ) (loss_per_metre : ℕ) (cost_price_per_metre : ℕ) :
  total_selling_price = 18000 →
  loss_per_metre = 5 →
  cost_price_per_metre = 65 →
  (total_selling_price / (cost_price_per_metre - loss_per_metre) : ℕ) = 300 := by
  sorry

end NUMINAMATH_CALUDE_cloth_sale_quantity_l837_83762
