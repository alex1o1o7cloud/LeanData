import Mathlib

namespace special_arithmetic_sequence_2007th_term_l70_7000

/-- An arithmetic sequence with special properties -/
structure SpecialArithmeticSequence where
  /-- The first term of the sequence -/
  a : ℝ
  /-- The common difference of the sequence -/
  d : ℝ
  /-- The 3rd, 5th, and 11th terms form a geometric sequence -/
  geometric_property : (a + 2*d) * (a + 10*d) = (a + 4*d)^2
  /-- The 4th term is 6 -/
  fourth_term : a + 3*d = 6

/-- The nth term of an arithmetic sequence -/
def arithmetic_term (seq : SpecialArithmeticSequence) (n : ℕ) : ℝ :=
  seq.a + (n - 1) * seq.d

/-- The main theorem -/
theorem special_arithmetic_sequence_2007th_term 
  (seq : SpecialArithmeticSequence) : 
  arithmetic_term seq 2007 = 6015 := by
  sorry

end special_arithmetic_sequence_2007th_term_l70_7000


namespace quadratic_sum_l70_7092

/-- A quadratic function y = ax^2 + bx + c with a minimum value of 61
    that passes through the points (1,0) and (3,0) -/
def QuadraticFunction (a b c : ℝ) : Prop :=
  (∀ x, a*x^2 + b*x + c ≥ 61) ∧
  (∃ x₀, a*x₀^2 + b*x₀ + c = 61) ∧
  (a*1^2 + b*1 + c = 0) ∧
  (a*3^2 + b*3 + c = 0)

theorem quadratic_sum (a b c : ℝ) :
  QuadraticFunction a b c → a + b + c = 0 := by
  sorry

end quadratic_sum_l70_7092


namespace ballet_slipper_price_fraction_l70_7048

/-- The price of one pair of high heels in dollars -/
def high_heels_price : ℚ := 60

/-- The number of pairs of ballet slippers bought -/
def ballet_slippers_count : ℕ := 5

/-- The total amount paid in dollars -/
def total_paid : ℚ := 260

/-- The fraction of the high heels price paid for each pair of ballet slippers -/
def ballet_slipper_fraction : ℚ := 2/3

theorem ballet_slipper_price_fraction :
  high_heels_price + ballet_slippers_count * (ballet_slipper_fraction * high_heels_price) = total_paid :=
by sorry

end ballet_slipper_price_fraction_l70_7048


namespace tangent_line_equation_l70_7026

/-- The equation of the tangent line to y = x^2 + 1 at (-1, 2) is 2x + y = 0 -/
theorem tangent_line_equation : 
  let f : ℝ → ℝ := λ x => x^2 + 1
  let x₀ : ℝ := -1
  let y₀ : ℝ := f x₀
  let m : ℝ := (deriv f) x₀
  ∀ x y : ℝ, (y - y₀ = m * (x - x₀)) ↔ (2*x + y = 0) := by
  sorry

end tangent_line_equation_l70_7026


namespace classroom_paint_area_l70_7074

/-- Calculates the area to be painted in a classroom given its dimensions and the area of doors, windows, and blackboard. -/
def areaToPaint (length width height doorWindowBlackboardArea : Real) : Real :=
  let ceilingArea := length * width
  let wallArea := 2 * (length * height + width * height)
  let totalArea := ceilingArea + wallArea
  totalArea - doorWindowBlackboardArea

/-- Theorem stating that the area to be painted in the given classroom is 121.5 square meters. -/
theorem classroom_paint_area :
  areaToPaint 8 6 3.5 24.5 = 121.5 := by
  sorry

end classroom_paint_area_l70_7074


namespace triangle_altitude_l70_7047

theorem triangle_altitude (A : ℝ) (b : ℝ) (h : ℝ) :
  A = 750 →
  b = 50 →
  A = (1/2) * b * h →
  h = 30 := by
sorry

end triangle_altitude_l70_7047


namespace smallest_multiple_of_24_and_36_not_20_l70_7061

theorem smallest_multiple_of_24_and_36_not_20 : 
  ∃ n : ℕ, n > 0 ∧ 24 ∣ n ∧ 36 ∣ n ∧ ¬(20 ∣ n) ∧ 
  ∀ m : ℕ, m > 0 → 24 ∣ m → 36 ∣ m → ¬(20 ∣ m) → n ≤ m :=
by
  -- The proof goes here
  sorry

#eval Nat.lcm 24 36  -- This should output 72

end smallest_multiple_of_24_and_36_not_20_l70_7061


namespace point_p_coordinates_l70_7088

/-- A point P with coordinates (m+3, m-1) that lies on the y-axis -/
structure PointP where
  m : ℝ
  x : ℝ := m + 3
  y : ℝ := m - 1
  on_y_axis : x = 0

/-- Theorem: If a point P(m+3, m-1) lies on the y-axis, then its coordinates are (0, -4) -/
theorem point_p_coordinates (P : PointP) : (P.x = 0 ∧ P.y = -4) := by
  sorry

end point_p_coordinates_l70_7088


namespace f_increasing_interval_l70_7080

-- Define the function f(x) = 2x³ - ln(x)
noncomputable def f (x : ℝ) : ℝ := 2 * x^3 - Real.log x

-- Theorem statement
theorem f_increasing_interval :
  ∀ x : ℝ, x > 0 → (∀ y : ℝ, y > x → f y > f x) ↔ x > (1/2 : ℝ) :=
sorry

end f_increasing_interval_l70_7080


namespace hyperbola_equation_l70_7016

/-- A hyperbola with given properties has the equation x²/18 - y²/32 = 1 -/
theorem hyperbola_equation (e : ℝ) (a b : ℝ) (h1 : e = 5/3) 
  (h2 : a > 0) (h3 : b > 0) (h4 : e = Real.sqrt (a^2 + b^2) / a) 
  (h5 : ∀ x y : ℝ, (x^2 / a^2) - (y^2 / b^2) = 1 ↔ 
    (8*x + 2*Real.sqrt 7*y - 16)^2 / (64/a^2 + 28/b^2) = 256) : 
  a^2 = 18 ∧ b^2 = 32 := by
sorry

end hyperbola_equation_l70_7016


namespace down_payment_proof_l70_7010

/-- Calculates the down payment for a car loan given the total price, monthly payment, and loan duration in years. -/
def calculate_down_payment (total_price : ℕ) (monthly_payment : ℕ) (loan_years : ℕ) : ℕ :=
  total_price - monthly_payment * loan_years * 12

/-- Proves that the down payment for a $20,000 car with a 5-year loan and $250 monthly payment is $5,000. -/
theorem down_payment_proof :
  calculate_down_payment 20000 250 5 = 5000 := by
  sorry

end down_payment_proof_l70_7010


namespace geometric_progression_m_existence_l70_7018

theorem geometric_progression_m_existence (m : ℂ) : 
  ∃ r : ℂ, r ≠ 0 ∧ 
    r ≠ r^2 ∧ r ≠ r^3 ∧ r^2 ≠ r^3 ∧
    r / (1 - r^2) = m ∧ 
    r^2 / (1 - r^3) = m ∧ 
    r^3 / (1 - r) = m := by
  sorry

end geometric_progression_m_existence_l70_7018


namespace total_length_S_l70_7085

-- Define the set S
def S : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | let (x, y) := p
    ((|x| - 2)^2 + (|y| - 2)^2)^(1/2) = 2 - |1 - ((|x| - 2)^2 + (|y| - 2)^2)^(1/2)|}

-- Define the length function for S
noncomputable def length_S : ℝ := sorry

-- Theorem statement
theorem total_length_S : length_S = 20 * Real.pi := by sorry

end total_length_S_l70_7085


namespace distance_on_number_line_l70_7003

theorem distance_on_number_line : 
  let point_a : ℤ := -2006
  let point_b : ℤ := 17
  abs (point_b - point_a) = 2023 := by sorry

end distance_on_number_line_l70_7003


namespace regular_polygon_sides_l70_7013

theorem regular_polygon_sides (D : ℕ) (h : D = 20) : 
  ∃ (n : ℕ), n > 2 ∧ D = n * (n - 3) / 2 → n = 8 := by
  sorry

end regular_polygon_sides_l70_7013


namespace factorial_difference_sum_l70_7036

theorem factorial_difference_sum : Nat.factorial 10 - Nat.factorial 8 + Nat.factorial 6 = 3589200 := by
  sorry

end factorial_difference_sum_l70_7036


namespace dogs_in_garden_l70_7072

/-- The number of dogs in a garden with ducks and a specific number of feet. -/
def num_dogs (total_feet : ℕ) (num_ducks : ℕ) (feet_per_dog : ℕ) (feet_per_duck : ℕ) : ℕ :=
  (total_feet - num_ducks * feet_per_duck) / feet_per_dog

/-- Theorem stating that under the given conditions, there are 6 dogs in the garden. -/
theorem dogs_in_garden : num_dogs 28 2 4 2 = 6 := by
  sorry

end dogs_in_garden_l70_7072


namespace blue_folder_stickers_l70_7041

/-- The number of stickers on each sheet in the blue folder -/
def blue_stickers_per_sheet : ℕ :=
  let total_stickers : ℕ := 60
  let sheets_per_folder : ℕ := 10
  let red_stickers_per_sheet : ℕ := 3
  let green_stickers_per_sheet : ℕ := 2
  let red_total := sheets_per_folder * red_stickers_per_sheet
  let green_total := sheets_per_folder * green_stickers_per_sheet
  let blue_total := total_stickers - red_total - green_total
  blue_total / sheets_per_folder

theorem blue_folder_stickers :
  blue_stickers_per_sheet = 1 := by
  sorry

end blue_folder_stickers_l70_7041


namespace odd_function_property_l70_7050

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem odd_function_property (f : ℝ → ℝ) 
  (h_odd : is_odd f)
  (h_prop : ∀ x, f (1 + x) = f (-x))
  (h_value : f (-1/3) = 1/3) :
  f (5/3) = 1/3 := by sorry

end odd_function_property_l70_7050


namespace submarine_hit_guaranteed_l70_7032

-- Define the type for the submarine's position and velocity
def Submarine := ℕ × ℕ+

-- Define the type for the firing sequence
def FiringSequence := ℕ → ℕ

-- The theorem statement
theorem submarine_hit_guaranteed :
  ∀ (sub : Submarine), ∃ (fire : FiringSequence), ∃ (t : ℕ),
    fire t = (sub.2 : ℕ) * t + sub.1 :=
by sorry

end submarine_hit_guaranteed_l70_7032


namespace amp_composition_l70_7034

-- Define the & operation (postfix)
def amp (x : ℤ) : ℤ := 7 - x

-- Define the & operation (prefix)
def amp_prefix (x : ℤ) : ℤ := x - 10

-- Theorem statement
theorem amp_composition : amp_prefix (amp 12) = -15 := by sorry

end amp_composition_l70_7034


namespace smaller_number_problem_l70_7078

theorem smaller_number_problem (a b : ℝ) (h1 : a + b = 15) (h2 : 3 * (a - b) = 21) : 
  min a b = 4 := by
  sorry

end smaller_number_problem_l70_7078


namespace probability_two_red_shoes_l70_7068

def total_shoes : ℕ := 8
def red_shoes : ℕ := 4
def green_shoes : ℕ := 4

theorem probability_two_red_shoes :
  (red_shoes : ℚ) / total_shoes * (red_shoes - 1) / (total_shoes - 1) = 3 / 14 :=
sorry

end probability_two_red_shoes_l70_7068


namespace part_1_part_2_l70_7029

-- Define the circle C
def circle_C (x y m : ℝ) : Prop := x^2 + y^2 - 2*x - 4*y + m = 0

-- Define Line 1
def line_1 (x y : ℝ) : Prop := 3*x + 4*y - 6 = 0

-- Define Line 2
def line_2 (x y : ℝ) : Prop := x - y - 1 = 0

-- Part I
theorem part_1 (x y m : ℝ) (M N : ℝ × ℝ) :
  circle_C x y m →
  line_1 (M.1) (M.2) →
  line_1 (N.1) (N.2) →
  (M.1 - N.1)^2 + (M.2 - N.2)^2 = 12 →
  m = 1 :=
sorry

-- Part II
theorem part_2 :
  ∃ m : ℝ, m = -2 ∧
  ∃ A B : ℝ × ℝ,
    circle_C A.1 A.2 m ∧
    circle_C B.1 B.2 m ∧
    line_2 A.1 A.2 ∧
    line_2 B.1 B.2 ∧
    (A.1 * B.1 + A.2 * B.2 = 0) :=
sorry

end part_1_part_2_l70_7029


namespace smallest_cube_ending_888_l70_7021

theorem smallest_cube_ending_888 : 
  ∃ n : ℕ, (∀ m : ℕ, m < n → m^3 % 1000 ≠ 888) ∧ n^3 % 1000 = 888 :=
by sorry

end smallest_cube_ending_888_l70_7021


namespace unique_number_l70_7062

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def digits_product (n : ℕ) : ℕ := (n / 10) * (n % 10)

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

theorem unique_number : 
  ∃! n : ℕ, is_two_digit n ∧ 
            n % 2 = 1 ∧ 
            n % 13 = 0 ∧ 
            is_perfect_square (digits_product n) ∧
            n = 91 := by sorry

end unique_number_l70_7062


namespace number_order_l70_7066

theorem number_order (a b : ℝ) (ha : a = 7) (hb : b = 0.3) :
  a^b > b^a ∧ b^a > Real.log b := by sorry

end number_order_l70_7066


namespace regular_polygon_sides_l70_7098

theorem regular_polygon_sides : ∃ n : ℕ, n > 2 ∧ n - (n * (n - 3) / 4) = 0 → n = 7 := by
  sorry

end regular_polygon_sides_l70_7098


namespace gumball_multiple_proof_l70_7044

theorem gumball_multiple_proof :
  ∀ (joanna_initial jacques_initial total_final multiple : ℕ),
    joanna_initial = 40 →
    jacques_initial = 60 →
    total_final = 500 →
    (joanna_initial + joanna_initial * multiple) +
    (jacques_initial + jacques_initial * multiple) = total_final →
    multiple = 4 := by
  sorry

end gumball_multiple_proof_l70_7044


namespace max_value_of_f_l70_7073

open Real

-- Define the function
def f (x : ℝ) : ℝ := x * (3 - 2 * x)

-- State the theorem
theorem max_value_of_f :
  ∃ (c : ℝ), c ∈ Set.Ioo 0 (3/2) ∧
  (∀ x, x ∈ Set.Ioo 0 (3/2) → f x ≤ f c) ∧
  f c = 9/8 :=
sorry

end max_value_of_f_l70_7073


namespace problem_solution_l70_7097

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

end problem_solution_l70_7097


namespace bertha_family_women_without_daughters_l70_7024

/-- Represents a woman in Bertha's family tree -/
structure Woman where
  has_daughters : Bool

/-- Bertha's family tree -/
structure Family where
  daughters : Finset Woman
  granddaughters : Finset Woman

/-- The number of women who have no daughters in Bertha's family -/
def num_women_without_daughters (f : Family) : Nat :=
  (f.daughters.filter (fun w => !w.has_daughters)).card +
  (f.granddaughters.filter (fun w => !w.has_daughters)).card

theorem bertha_family_women_without_daughters :
  ∃ f : Family,
    f.daughters.card = 8 ∧
    (∀ d ∈ f.daughters, d.has_daughters) ∧
    (∀ d ∈ f.daughters, (f.granddaughters.filter (fun g => g.has_daughters.not)).card = 4) ∧
    (f.daughters.card + f.granddaughters.card = 40) ∧
    num_women_without_daughters f = 32 := by
  sorry

end bertha_family_women_without_daughters_l70_7024


namespace arthurs_purchases_l70_7022

/-- The cost of Arthur's purchases on two days -/
theorem arthurs_purchases (hamburger_price : ℚ) :
  (3 * hamburger_price + 4 * 1 = 10) →
  (2 * hamburger_price + 3 * 1 = 7) :=
by sorry

end arthurs_purchases_l70_7022


namespace average_increase_food_expenditure_l70_7083

/-- Represents the regression line equation for annual income and food expenditure -/
def regression_line (x : ℝ) : ℝ := 0.245 * x + 0.321

/-- Theorem stating that the average increase in food expenditure for a unit increase in income is 0.245 -/
theorem average_increase_food_expenditure :
  ∀ x : ℝ, regression_line (x + 1) - regression_line x = 0.245 := by
sorry


end average_increase_food_expenditure_l70_7083


namespace calculate_partner_b_contribution_b_contribution_is_16200_l70_7053

/-- Calculates the contribution of partner B given the initial investment of A, 
    the time before B joins, and the profit-sharing ratio. -/
theorem calculate_partner_b_contribution 
  (a_investment : ℕ) 
  (total_months : ℕ) 
  (b_join_month : ℕ) 
  (profit_ratio_a : ℕ) 
  (profit_ratio_b : ℕ) : ℕ :=
  let b_investment := 
    (a_investment * total_months * profit_ratio_b) / 
    (profit_ratio_a * (total_months - b_join_month))
  b_investment

/-- Proves that B's contribution is 16200 given the problem conditions -/
theorem b_contribution_is_16200 : 
  calculate_partner_b_contribution 4500 12 7 2 3 = 16200 := by
  sorry

end calculate_partner_b_contribution_b_contribution_is_16200_l70_7053


namespace complex_number_real_imag_equal_l70_7096

theorem complex_number_real_imag_equal (a : ℝ) : 
  let x : ℂ := (1 + a * Complex.I) * (2 + Complex.I)
  (x.re = x.im) → a = 1/3 := by sorry

end complex_number_real_imag_equal_l70_7096


namespace probability_not_black_ball_l70_7082

theorem probability_not_black_ball (white black red : ℕ) 
  (h_white : white = 8) 
  (h_black : black = 9) 
  (h_red : red = 3) : 
  (white + red) / (white + black + red : ℚ) = 11 / 20 := by
  sorry

end probability_not_black_ball_l70_7082


namespace sum_of_four_digit_numbers_l70_7008

/-- The set of digits used to form the numbers -/
def digits : Finset Nat := {1, 2, 3, 4, 5}

/-- A four-digit number formed from the given digits -/
structure FourDigitNumber where
  d₁ : Nat
  d₂ : Nat
  d₃ : Nat
  d₄ : Nat
  h₁ : d₁ ∈ digits
  h₂ : d₂ ∈ digits
  h₃ : d₃ ∈ digits
  h₄ : d₄ ∈ digits
  distinct : d₁ ≠ d₂ ∧ d₁ ≠ d₃ ∧ d₁ ≠ d₄ ∧ d₂ ≠ d₃ ∧ d₂ ≠ d₄ ∧ d₃ ≠ d₄

/-- The value of a four-digit number -/
def value (n : FourDigitNumber) : Nat :=
  1000 * n.d₁ + 100 * n.d₂ + 10 * n.d₃ + n.d₄

/-- The set of all valid four-digit numbers -/
def allFourDigitNumbers : Finset FourDigitNumber :=
  sorry

/-- The theorem to be proved -/
theorem sum_of_four_digit_numbers :
  (allFourDigitNumbers.sum value) = 399960 := by
  sorry

end sum_of_four_digit_numbers_l70_7008


namespace fraction_equality_l70_7005

theorem fraction_equality (a b c : ℝ) (h1 : c ≠ 0) (h2 : a / c = b / c) : a = b := by
  sorry

end fraction_equality_l70_7005


namespace power_of_product_equals_product_of_powers_l70_7099

theorem power_of_product_equals_product_of_powers (a b : ℝ) : 
  (-a * b^2)^2 = a^2 * b^4 := by
  sorry

end power_of_product_equals_product_of_powers_l70_7099


namespace one_non_negative_solution_condition_l70_7049

/-- The quadratic equation defined by parameter a -/
def quadratic_equation (a : ℝ) (x : ℝ) : ℝ := (a - 1) * x^2 - 2*(a + 1) * x + 2*(a + 1)

/-- Predicate to check if the equation has only one non-negative solution -/
def has_one_non_negative_solution (a : ℝ) : Prop :=
  ∃! x : ℝ, x ≥ 0 ∧ quadratic_equation a x = 0

/-- Theorem stating the condition for the equation to have only one non-negative solution -/
theorem one_non_negative_solution_condition (a : ℝ) :
  has_one_non_negative_solution a ↔ ((-1 ≤ a ∧ a ≤ 1) ∨ a = 3) :=
sorry

end one_non_negative_solution_condition_l70_7049


namespace no_intersection_l70_7054

/-- Parabola C: y^2 = 4x -/
def parabola (x y : ℝ) : Prop := y^2 = 4*x

/-- Point M(x₀, y₀) is inside the parabola if y₀² < 4x₀ -/
def inside_parabola (x₀ y₀ : ℝ) : Prop := y₀^2 < 4*x₀

/-- Line l: y₀y = 2(x + x₀) -/
def line (x₀ y₀ x y : ℝ) : Prop := y₀*y = 2*(x + x₀)

theorem no_intersection (x₀ y₀ : ℝ) (h : inside_parabola x₀ y₀) :
  ¬∃ x y, parabola x y ∧ line x₀ y₀ x y :=
sorry

end no_intersection_l70_7054


namespace sum_of_roots_l70_7065

theorem sum_of_roots (a b : ℝ) 
  (ha : a^3 - 3*a^2 + 5*a - 1 = 0)
  (hb : b^3 - 3*b^2 + 5*b - 5 = 0) : 
  a + b = 2 := by
sorry

end sum_of_roots_l70_7065


namespace abs_inequality_solution_set_l70_7090

theorem abs_inequality_solution_set (x : ℝ) : 
  |2*x - 1| - |x - 2| < 0 ↔ -1 < x ∧ x < 1 := by sorry

end abs_inequality_solution_set_l70_7090


namespace square_area_from_adjacent_points_l70_7006

/-- The area of a square with adjacent points (2,1) and (3,4) on a Cartesian coordinate plane is 10. -/
theorem square_area_from_adjacent_points :
  let p1 : ℝ × ℝ := (2, 1)
  let p2 : ℝ × ℝ := (3, 4)
  let side_length := Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)
  let area := side_length^2
  area = 10 := by sorry

end square_area_from_adjacent_points_l70_7006


namespace custom_op_result_l70_7079

-- Define the custom operation
def custom_op (a b c : ℕ) : ℕ := 
  (a * b * 10000) + (a * c * 100) + (a * (b + c))

-- State the theorem
theorem custom_op_result : custom_op 7 2 5 = 143549 := by
  sorry

end custom_op_result_l70_7079


namespace circle_line_intersection_l70_7045

/-- A circle C with center (a, 0) and radius r -/
structure Circle where
  a : ℝ
  r : ℝ
  r_pos : r > 0

/-- A line with slope k passing through (-1, 0) -/
structure Line where
  k : ℝ

/-- Theorem: Given a circle C and a line l satisfying certain conditions, 
    the dot product of OA and OB is -(26 + 9√2) / 5 -/
theorem circle_line_intersection 
  (C : Circle) 
  (l : Line) 
  (h1 : C.r = |C.a - 2 * Real.sqrt 2| / Real.sqrt 2)  -- C is tangent to x + y - 2√2 = 0
  (h2 : 4 * Real.sqrt 2 = 2 * Real.sqrt (C.r^2 - (|C.a| / Real.sqrt 2)^2))  -- chord length on y = x is 4√2
  (h3 : ∃ (m : ℝ), m / l.k^2 = -3 - Real.sqrt 2)  -- condition on slopes product
  : ∃ (A B : ℝ × ℝ), 
    (A.1 - C.a)^2 + A.2^2 = C.r^2 ∧   -- A is on circle C
    (B.1 - C.a)^2 + B.2^2 = C.r^2 ∧   -- B is on circle C
    A.2 = l.k * (A.1 + 1) ∧           -- A is on line l
    B.2 = l.k * (B.1 + 1) ∧           -- B is on line l
    (A.1 - C.a) * (B.1 - C.a) + A.2 * B.2 = -(26 + 9 * Real.sqrt 2) / 5  -- OA · OB
    := by sorry

end circle_line_intersection_l70_7045


namespace jenna_concert_spending_percentage_l70_7089

/-- Proves that Jenna spends 10% of her monthly salary on a concert outing -/
theorem jenna_concert_spending_percentage :
  let concert_ticket_cost : ℚ := 181
  let drink_ticket_cost : ℚ := 7
  let num_drink_tickets : ℕ := 5
  let hourly_wage : ℚ := 18
  let weekly_hours : ℕ := 30
  let weeks_per_month : ℕ := 4

  let total_outing_cost : ℚ := concert_ticket_cost + drink_ticket_cost * num_drink_tickets
  let weekly_salary : ℚ := hourly_wage * weekly_hours
  let monthly_salary : ℚ := weekly_salary * weeks_per_month
  let spending_percentage : ℚ := total_outing_cost / monthly_salary * 100

  spending_percentage = 10 :=
by
  sorry


end jenna_concert_spending_percentage_l70_7089


namespace todd_ate_cupcakes_l70_7075

theorem todd_ate_cupcakes (initial_cupcakes : ℕ) (packages : ℕ) (cupcakes_per_package : ℕ) : 
  initial_cupcakes = 18 →
  packages = 5 →
  cupcakes_per_package = 2 →
  initial_cupcakes - packages * cupcakes_per_package = 8 :=
by sorry

end todd_ate_cupcakes_l70_7075


namespace unanswered_questions_l70_7069

/-- Calculates the number of unanswered questions on a test -/
theorem unanswered_questions
  (total_questions : ℕ)
  (answering_time_hours : ℕ)
  (time_per_question_minutes : ℕ)
  (h1 : total_questions = 100)
  (h2 : answering_time_hours = 2)
  (h3 : time_per_question_minutes = 2) :
  total_questions - (answering_time_hours * 60) / time_per_question_minutes = 40 :=
by sorry

end unanswered_questions_l70_7069


namespace max_sum_circle_50_l70_7033

/-- The maximum sum of x and y for integer solutions of x^2 + y^2 = 50 -/
theorem max_sum_circle_50 : 
  ∀ x y : ℤ, x^2 + y^2 = 50 → x + y ≤ 10 :=
by sorry

end max_sum_circle_50_l70_7033


namespace swimming_practice_months_l70_7059

def total_required_hours : ℕ := 4000
def completed_hours : ℕ := 460
def practice_hours_per_month : ℕ := 400

theorem swimming_practice_months : 
  ∃ (months : ℕ), 
    months * practice_hours_per_month ≥ total_required_hours - completed_hours ∧ 
    (months - 1) * practice_hours_per_month < total_required_hours - completed_hours ∧
    months = 9 := by
  sorry

end swimming_practice_months_l70_7059


namespace x_range_theorem_l70_7030

theorem x_range_theorem (x : ℝ) :
  (∀ a b c : ℝ, a^2 + b^2 + c^2 = 1 → a + b + Real.sqrt 2 * c ≤ |x^2 - 1|) →
  x ≤ -Real.sqrt 3 ∨ x ≥ Real.sqrt 3 :=
by sorry

end x_range_theorem_l70_7030


namespace triangle_abc_proof_l70_7076

/-- Given a triangle ABC with the specified conditions, prove that A = π/6 and a = 2 -/
theorem triangle_abc_proof (a b c : ℝ) (A B C : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →
  A > 0 ∧ A < π ∧ B > 0 ∧ B < π ∧ C > 0 ∧ C < π →
  A + B + C = π →
  Real.sqrt 3 * a * Real.sin C = c * Real.cos A →
  (1 / 2) * b * c * Real.sin A = Real.sqrt 3 →
  b + c = 2 + 2 * Real.sqrt 3 →
  A = π / 6 ∧ a = 2 := by
  sorry

end triangle_abc_proof_l70_7076


namespace test_questions_count_l70_7012

theorem test_questions_count (sections : Nat) (correct_answers : Nat) 
  (h1 : sections = 4)
  (h2 : correct_answers = 20)
  (h3 : ∀ x : Nat, x > 0 → (60 : Real) / 100 < (correct_answers : Real) / x → (correct_answers : Real) / x < (70 : Real) / 100 → x % sections = 0 → x = 32) :
  ∃ total_questions : Nat, 
    total_questions > 0 ∧ 
    (60 : Real) / 100 < (correct_answers : Real) / total_questions ∧ 
    (correct_answers : Real) / total_questions < (70 : Real) / 100 ∧ 
    total_questions % sections = 0 ∧
    total_questions = 32 :=
by sorry

end test_questions_count_l70_7012


namespace abs_fraction_inequality_l70_7035

theorem abs_fraction_inequality (x : ℝ) :
  x ≠ 0 → (|(x - 2) / x| > (x - 2) / x ↔ 0 < x ∧ x < 2) := by sorry

end abs_fraction_inequality_l70_7035


namespace smallest_k_for_inequality_l70_7056

theorem smallest_k_for_inequality : 
  ∃ (k : ℕ), k = 4 ∧ 
  (∀ (a : ℝ) (n : ℕ), 0 ≤ a ∧ a ≤ 1 → a^k * (1-a)^n < 1 / (n+1)^3) ∧
  (∀ (k' : ℕ), k' < k → 
    ∃ (a : ℝ) (n : ℕ), 0 ≤ a ∧ a ≤ 1 ∧ a^k' * (1-a)^n ≥ 1 / (n+1)^3) :=
by sorry

end smallest_k_for_inequality_l70_7056


namespace triangle_condition_implies_a_ge_5_l70_7060

/-- The function f(x) = x^2 - 2x + a -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*x + a

/-- Theorem: If for any three distinct values in [0, 3], f(x) can form a triangle, then a ≥ 5 -/
theorem triangle_condition_implies_a_ge_5 (a : ℝ) :
  (∀ x y z : ℝ, 0 ≤ x ∧ x ≤ 3 ∧ 0 ≤ y ∧ y ≤ 3 ∧ 0 ≤ z ∧ z ≤ 3 →
    x ≠ y ∧ y ≠ z ∧ x ≠ z →
    f a x + f a y > f a z ∧ f a y + f a z > f a x ∧ f a x + f a z > f a y) →
  a ≥ 5 := by
  sorry

end triangle_condition_implies_a_ge_5_l70_7060


namespace total_envelopes_is_975_l70_7051

/-- The number of envelopes Kiera has of each color and in total -/
structure EnvelopeCount where
  blue : ℕ
  yellow : ℕ
  green : ℕ
  red : ℕ
  purple : ℕ
  total : ℕ

/-- Calculates the total number of envelopes given the conditions -/
def calculateEnvelopes : EnvelopeCount :=
  let blue := 120
  let yellow := blue - 25
  let green := 5 * yellow
  let red := (blue + yellow) / 2
  let purple := red + 71
  let total := blue + yellow + green + red + purple
  { blue := blue
  , yellow := yellow
  , green := green
  , red := red
  , purple := purple
  , total := total }

/-- Theorem stating that the total number of envelopes is 975 -/
theorem total_envelopes_is_975 : calculateEnvelopes.total = 975 := by
  sorry

#eval calculateEnvelopes.total

end total_envelopes_is_975_l70_7051


namespace reggie_layups_l70_7020

/-- Represents the score of a player in the basketball shooting contest -/
structure Score where
  layups : ℕ
  freeThrows : ℕ
  longShots : ℕ

/-- Calculates the total points for a given score -/
def totalPoints (s : Score) : ℕ :=
  s.layups + 2 * s.freeThrows + 3 * s.longShots

theorem reggie_layups : 
  ∀ (reggie_score : Score) (brother_score : Score),
    reggie_score.freeThrows = 2 →
    reggie_score.longShots = 1 →
    brother_score.layups = 0 →
    brother_score.freeThrows = 0 →
    brother_score.longShots = 4 →
    totalPoints reggie_score + 2 = totalPoints brother_score →
    reggie_score.layups = 3 := by
  sorry

#check reggie_layups

end reggie_layups_l70_7020


namespace proportion_change_l70_7055

def is_proportion (a b c d : ℚ) : Prop := a * d = b * c

theorem proportion_change (x y : ℚ) :
  is_proportion 3 5 6 10 →
  is_proportion 12 y 6 10 →
  y = 20 := by sorry

end proportion_change_l70_7055


namespace min_value_reciprocal_sum_l70_7086

theorem min_value_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (h_sum : a + 2*b = 2) :
  (1/a + 1/b) ≥ 3/2 + Real.sqrt 2 ∧ 
  ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ a₀ + 2*b₀ = 2 ∧ 1/a₀ + 1/b₀ = 3/2 + Real.sqrt 2 :=
sorry

end min_value_reciprocal_sum_l70_7086


namespace only_second_expression_always_true_l70_7046

theorem only_second_expression_always_true :
  (∀ n a : ℝ, n * a^n = a) = False ∧
  (∀ a : ℝ, (a^2 - 3*a + 3)^0 = 1) = True ∧
  (3 - 3 = 6*(-3)^2) = False := by
sorry

end only_second_expression_always_true_l70_7046


namespace sin_alpha_value_l70_7091

theorem sin_alpha_value (α : Real) 
  (h : (Real.sqrt 2 / 2) * (Real.sin (α / 2) - Real.cos (α / 2)) = Real.sqrt 6 / 3) : 
  Real.sin α = -1/3 := by
  sorry

end sin_alpha_value_l70_7091


namespace louise_and_tom_ages_l70_7017

/-- Given the age relationship between Louise and Tom, prove their current ages sum to 26 -/
theorem louise_and_tom_ages (L T : ℕ) 
  (h1 : L = T + 8) 
  (h2 : L + 4 = 3 * (T - 2)) : 
  L + T = 26 := by
  sorry

end louise_and_tom_ages_l70_7017


namespace sin_function_smallest_c_l70_7040

/-- 
Given a sinusoidal function f(x) = a * sin(b * x + c) where a, b, and c are positive constants,
if f(x) reaches its maximum at x = 0, then the smallest possible value of c is π/2.
-/
theorem sin_function_smallest_c (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (∀ x, a * Real.sin (b * x + c) ≤ a * Real.sin c) → c = π / 2 := by
  sorry

end sin_function_smallest_c_l70_7040


namespace balloon_radius_ratio_l70_7067

theorem balloon_radius_ratio (V_L V_S r_L r_S : ℝ) : 
  V_L = 450 * Real.pi →
  V_S = 0.08 * V_L →
  V_L = (4/3) * Real.pi * r_L^3 →
  V_S = (4/3) * Real.pi * r_S^3 →
  r_S / r_L = Real.rpow 2 (1/3) / 5 := by
  sorry

end balloon_radius_ratio_l70_7067


namespace min_x_prime_factorization_sum_l70_7011

theorem min_x_prime_factorization_sum (x y a b e : ℕ+) (c d f : ℕ) :
  (∀ x' y' : ℕ+, 7 * x'^5 = 13 * y'^11 → x ≤ x') →
  7 * x^5 = 13 * y^11 →
  x = a^c * b^d * e^f →
  a.val ≠ b.val ∧ b.val ≠ e.val ∧ a.val ≠ e.val →
  Nat.Prime a.val ∧ Nat.Prime b.val ∧ Nat.Prime e.val →
  a.val + b.val + c + d + e.val + f = 37 :=
by sorry

end min_x_prime_factorization_sum_l70_7011


namespace power_product_equality_l70_7071

theorem power_product_equality : (-2/3)^2023 * (3/2)^2024 = -3/2 := by sorry

end power_product_equality_l70_7071


namespace real_part_of_i_times_one_plus_i_l70_7070

theorem real_part_of_i_times_one_plus_i : Complex.re (Complex.I * (1 + Complex.I)) = -1 := by
  sorry

end real_part_of_i_times_one_plus_i_l70_7070


namespace inequality_proof_l70_7043

theorem inequality_proof (x y z : ℝ) (n : ℕ) 
  (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) 
  (h4 : x + y + z = 1) (h5 : n > 0) : 
  x^4 / (y * (1 - y^n)) + y^4 / (z * (1 - z^n)) + z^4 / (x * (1 - x^n)) ≥ 3^n / (3^(n+2) - 9) := by
  sorry

end inequality_proof_l70_7043


namespace smaller_number_problem_l70_7052

theorem smaller_number_problem (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + y = 14) (h4 : y = 3 * x) : x = 3.5 := by
  sorry

end smaller_number_problem_l70_7052


namespace arithmetic_calculations_l70_7004

theorem arithmetic_calculations :
  ((5 : ℤ) - (-10) + (-32) - 7 = -24) ∧
  ((1/4 + 1/6 - 1/2 : ℚ) * 12 + (-2)^3 / (-4) = 1) ∧
  ((3^2 : ℚ) + (-2-5) / 7 - |-(1/4)| * (-2)^4 + (-1)^2023 = 3) :=
by sorry

end arithmetic_calculations_l70_7004


namespace quadratic_equation_solution_l70_7007

theorem quadratic_equation_solution : ∃ x₁ x₂ : ℝ, x₁ = 3 ∧ x₂ = -1 ∧ 
  (x₁^2 - 2*x₁ - 3 = 0) ∧ (x₂^2 - 2*x₂ - 3 = 0) := by
  sorry

end quadratic_equation_solution_l70_7007


namespace expected_value_r₃_l70_7038

/-- The expected value of a single fair six-sided die roll -/
def single_die_ev : ℝ := 3.5

/-- The number of dice rolled in the first round -/
def first_round_dice : ℕ := 8

/-- The expected value of r₁ (the sum of first_round_dice fair dice rolls) -/
def r₁_ev : ℝ := first_round_dice * single_die_ev

/-- The expected value of r₂ (the sum of r₁_ev fair dice rolls) -/
def r₂_ev : ℝ := r₁_ev * single_die_ev

/-- The expected value of r₃ (the sum of r₂_ev fair dice rolls) -/
def r₃_ev : ℝ := r₂_ev * single_die_ev

theorem expected_value_r₃ : r₃_ev = 343 := by
  sorry

end expected_value_r₃_l70_7038


namespace fraction_equality_l70_7025

theorem fraction_equality (m n : ℚ) (h : m / n = 2 / 3) : m / (m + n) = 2 / 5 := by
  sorry

end fraction_equality_l70_7025


namespace arithmetic_progression_cube_sum_l70_7028

/-- 
Given integers x, y, z, u forming an arithmetic progression and satisfying x^3 + y^3 + z^3 = u^3,
prove that there exists an integer d such that x = 3d, y = 4d, z = 5d, and u = 6d.
-/
theorem arithmetic_progression_cube_sum (x y z u : ℤ) 
  (h_arith_prog : ∃ (d : ℤ), y = x + d ∧ z = y + d ∧ u = z + d)
  (h_cube_sum : x^3 + y^3 + z^3 = u^3) :
  ∃ (d : ℤ), x = 3*d ∧ y = 4*d ∧ z = 5*d ∧ u = 6*d :=
by sorry

end arithmetic_progression_cube_sum_l70_7028


namespace house_cleaning_time_l70_7064

/-- The time it takes for three people to clean a house together, given their individual cleaning rates. -/
theorem house_cleaning_time 
  (john_time : ℝ) 
  (nick_time : ℝ) 
  (mary_time : ℝ) 
  (h1 : john_time = 6) 
  (h2 : nick_time / 3 = john_time / 2) 
  (h3 : mary_time = nick_time + 2) : 
  1 / (1 / john_time + 1 / nick_time + 1 / mary_time) = 198 / 73 :=
sorry

end house_cleaning_time_l70_7064


namespace quadratic_transformation_l70_7093

theorem quadratic_transformation (a b c : ℝ) (ha : a ≠ 0) :
  ∃ (h k s : ℝ) (hs : s ≠ 0), ∀ x : ℝ,
    a * x^2 + b * x + c = s^2 * ((x - h)^2 + k) :=
by sorry

end quadratic_transformation_l70_7093


namespace period_length_proof_l70_7042

/-- Calculates the length of each period given the number of students, presentation time per student, and number of periods. -/
def period_length (num_students : ℕ) (presentation_time : ℕ) (num_periods : ℕ) : ℕ :=
  (num_students * presentation_time) / num_periods

/-- Proves that given 32 students, 5 minutes per presentation, and 4 periods, the length of each period is 40 minutes. -/
theorem period_length_proof :
  period_length 32 5 4 = 40 := by
  sorry

#eval period_length 32 5 4

end period_length_proof_l70_7042


namespace parallel_lines_length_l70_7077

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

end parallel_lines_length_l70_7077


namespace cost_of_2500_pencils_l70_7095

/-- The cost of a given number of pencils, given the cost of 100 pencils -/
def cost_of_pencils (cost_per_100 : ℚ) (num_pencils : ℕ) : ℚ :=
  (cost_per_100 * num_pencils) / 100

/-- Theorem stating that 2500 pencils cost $750 when 100 pencils cost $30 -/
theorem cost_of_2500_pencils :
  cost_of_pencils 30 2500 = 750 := by
  sorry

end cost_of_2500_pencils_l70_7095


namespace cookie_sales_revenue_l70_7031

-- Define the sales data for each girl on each day
def robyn_day1_packs : ℕ := 25
def robyn_day1_price : ℚ := 4
def lucy_day1_packs : ℕ := 17
def lucy_day1_price : ℚ := 5

def robyn_day2_packs : ℕ := 15
def robyn_day2_price : ℚ := 7/2
def lucy_day2_packs : ℕ := 9
def lucy_day2_price : ℚ := 9/2

def robyn_day3_packs : ℕ := 23
def robyn_day3_price : ℚ := 9/2
def lucy_day3_packs : ℕ := 20
def lucy_day3_price : ℚ := 7/2

-- Define the total revenue calculation
def total_revenue : ℚ :=
  robyn_day1_packs * robyn_day1_price +
  lucy_day1_packs * lucy_day1_price +
  robyn_day2_packs * robyn_day2_price +
  lucy_day2_packs * lucy_day2_price +
  robyn_day3_packs * robyn_day3_price +
  lucy_day3_packs * lucy_day3_price

-- Theorem statement
theorem cookie_sales_revenue :
  total_revenue = 451.5 := by
  sorry

end cookie_sales_revenue_l70_7031


namespace least_positive_linear_combination_l70_7001

theorem least_positive_linear_combination : 
  ∃ (n : ℕ), n > 0 ∧ 
  (∀ (x y z : ℤ), 72 * x + 54 * y + 36 * z > 0 → 72 * x + 54 * y + 36 * z ≥ n) ∧
  (∃ (x y z : ℤ), 72 * x + 54 * y + 36 * z = n) :=
by
  -- The proof goes here
  sorry

end least_positive_linear_combination_l70_7001


namespace remainder_sum_l70_7057

theorem remainder_sum (x y : ℤ) (hx : x % 80 = 75) (hy : y % 120 = 115) :
  (x + y) % 40 = 30 := by
  sorry

end remainder_sum_l70_7057


namespace range_reduction_after_five_trials_l70_7014

/-- The reduction factor for each trial using the 0.618 method -/
def reduction_factor : ℝ := 0.618

/-- The number of trials -/
def num_trials : ℕ := 5

/-- The range reduction after a given number of trials -/
def range_reduction (n : ℕ) : ℝ := reduction_factor ^ n

theorem range_reduction_after_five_trials :
  range_reduction (num_trials - 1) = reduction_factor ^ 4 := by
  sorry

end range_reduction_after_five_trials_l70_7014


namespace inequality_transformation_l70_7084

theorem inequality_transformation (a b c : ℝ) (hc : c ≠ 0) :
  a * c^2 > b * c^2 → a > b := by sorry

end inequality_transformation_l70_7084


namespace constant_relationship_l70_7094

theorem constant_relationship (a b c d : ℝ) :
  (∀ θ : ℝ, 0 < θ ∧ θ < π / 2 →
    (a * Real.sin θ + b * Real.cos θ - c = 0) ∧
    (a * Real.cos θ - b * Real.sin θ + d = 0)) →
  a^2 + b^2 = c^2 + d^2 := by sorry

end constant_relationship_l70_7094


namespace cubic_function_derivative_l70_7019

theorem cubic_function_derivative (a : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ a * x^3 + 3 * x^2 + 2
  let f' : ℝ → ℝ := λ x ↦ (3 * a * x^2) + (6 * x)
  f' (-1) = 4 → a = 10/3 := by
  sorry

end cubic_function_derivative_l70_7019


namespace sum_90_to_99_l70_7087

/-- The sum of consecutive integers from 90 to 99 is equal to 945. -/
theorem sum_90_to_99 : (Finset.range 10).sum (fun i => i + 90) = 945 := by
  sorry

end sum_90_to_99_l70_7087


namespace work_completion_theorem_l70_7023

/-- Represents the number of days required to complete the work -/
def original_days : ℕ := 11

/-- Represents the number of additional men who joined -/
def additional_men : ℕ := 10

/-- Represents the number of days saved after additional men joined -/
def days_saved : ℕ := 3

/-- Calculates the original number of men required to complete the work -/
def original_men : ℕ := 27

theorem work_completion_theorem :
  ∃ (work_rate : ℚ),
    (original_men * work_rate * original_days : ℚ) =
    ((original_men + additional_men) * work_rate * (original_days - days_saved) : ℚ) :=
by sorry

end work_completion_theorem_l70_7023


namespace apples_collected_l70_7058

/-- The number of apples Lexie picked -/
def lexie_apples : ℕ := 12

/-- The number of apples Tom picked -/
def tom_apples : ℕ := 2 * lexie_apples

/-- The total number of apples collected -/
def total_apples : ℕ := lexie_apples + tom_apples

theorem apples_collected : total_apples = 36 := by
  sorry

end apples_collected_l70_7058


namespace central_octagon_area_l70_7027

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a square tile -/
structure SquareTile where
  sideLength : ℝ
  center : Point

/-- Theorem: Area of the central octagon in a square tile -/
theorem central_octagon_area (tile : SquareTile) (X Y Z : Point) :
  tile.sideLength = 8 →
  (X.x - Y.x)^2 + (X.y - Y.y)^2 = 2^2 →
  (Y.x - Z.x)^2 + (Y.y - Z.y)^2 = 2^2 →
  (Z.y - Y.y) / (Z.x - Y.x) = 0 →
  let U : Point := { x := (X.x + Z.x) / 2, y := (X.y + Z.y) / 2 }
  let V : Point := { x := (Y.x + Z.x) / 2, y := (Y.y + Z.y) / 2 }
  let octagonArea := (U.x - V.x)^2 + (U.y - V.y)^2 + 4 * ((X.x - U.x)^2 + (X.y - U.y)^2)
  octagonArea = 10 := by
  sorry


end central_octagon_area_l70_7027


namespace french_students_count_l70_7063

theorem french_students_count (total : ℕ) (german : ℕ) (both : ℕ) (neither : ℕ) 
  (h1 : total = 79)
  (h2 : german = 22)
  (h3 : both = 9)
  (h4 : neither = 25)
  : ∃ french : ℕ, french = 41 ∧ total = french + german - both + neither :=
by sorry

end french_students_count_l70_7063


namespace initial_number_of_men_l70_7081

theorem initial_number_of_men (M : ℕ) (A : ℝ) : 
  (2 * M = 46 - (20 + 10)) →  -- Condition 1 and 2
  (M = 8) :=                  -- Conclusion
by
  sorry  -- Skip the proof

end initial_number_of_men_l70_7081


namespace root_product_cubic_l70_7015

theorem root_product_cubic (a b c : ℂ) : 
  (3 * a^3 - 4 * a^2 + a - 2 = 0) →
  (3 * b^3 - 4 * b^2 + b - 2 = 0) →
  (3 * c^3 - 4 * c^2 + c - 2 = 0) →
  a * b * c = 2/3 := by
  sorry

end root_product_cubic_l70_7015


namespace square_area_ratio_l70_7009

theorem square_area_ratio (side_c side_d : ℝ) 
  (h1 : side_c = 45)
  (h2 : side_d = 60) : 
  (side_c^2) / (side_d^2) = 9 / 16 := by
  sorry

end square_area_ratio_l70_7009


namespace ellipse_equation_l70_7039

/-- An ellipse with given properties has the equation x²/3 + y²/2 = 1 -/
theorem ellipse_equation (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a > b) :
  let e := Real.sqrt 3 / 3
  let c := a * e
  let perimeter := 4 * Real.sqrt 3
  (c^2 = a^2 - b^2) →
  (perimeter = 2 * a + 2 * a) →
  (∀ x y : ℝ, x^2/a^2 + y^2/b^2 = 1 ↔ x^2/3 + y^2/2 = 1) :=
by sorry

end ellipse_equation_l70_7039


namespace largest_of_eight_consecutive_integers_l70_7037

theorem largest_of_eight_consecutive_integers (a : ℕ) 
  (h1 : a > 0) 
  (h2 : (a + (a + 1) + (a + 2) + (a + 3) + (a + 4) + (a + 5) + (a + 6) + (a + 7)) = 5400) : 
  (a + 7) = 678 := by
  sorry

end largest_of_eight_consecutive_integers_l70_7037


namespace curve_inequality_l70_7002

-- Define the logarithm function (base 10)
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Define the curve equation
def curve_equation (a b c x y : ℝ) : Prop :=
  a * (lg x)^2 + 2 * b * (lg x) * (lg y) + c * (lg y)^2 = 1

-- Main theorem
theorem curve_inequality (a b c : ℝ) 
  (h1 : b^2 - a*c < 0) 
  (h2 : curve_equation a b c 10 (1/10)) :
  ∀ x y : ℝ, curve_equation a b c x y →
  -1 / Real.sqrt (a*c - b^2) ≤ lg (x*y) ∧ lg (x*y) ≤ 1 / Real.sqrt (a*c - b^2) := by
  sorry

end curve_inequality_l70_7002
