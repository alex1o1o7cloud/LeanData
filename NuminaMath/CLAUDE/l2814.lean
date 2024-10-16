import Mathlib

namespace NUMINAMATH_CALUDE_tan_sqrt3_inequality_l2814_281421

open Set Real

-- Define the set of x that satisfies the inequality
def S : Set ℝ := {x | tan x - Real.sqrt 3 ≤ 0}

-- Define the solution set
def T : Set ℝ := {x | ∃ k : ℤ, -π/2 + k*π < x ∧ x ≤ π/3 + k*π}

-- Theorem statement
theorem tan_sqrt3_inequality : S = T := by sorry

end NUMINAMATH_CALUDE_tan_sqrt3_inequality_l2814_281421


namespace NUMINAMATH_CALUDE_final_result_l2814_281492

theorem final_result (chosen_number : ℕ) (h : chosen_number = 1152) : 
  (chosen_number / 6 : ℚ) - 189 = 3 := by
  sorry

end NUMINAMATH_CALUDE_final_result_l2814_281492


namespace NUMINAMATH_CALUDE_chocolate_distribution_l2814_281463

theorem chocolate_distribution (total_chocolates : ℕ) (boys_chocolates : ℕ) (girls_chocolates : ℕ) 
  (num_boys : ℕ) (num_girls : ℕ) :
  total_chocolates = 3000 →
  boys_chocolates = 2 →
  girls_chocolates = 3 →
  num_boys = 60 →
  num_girls = 60 →
  num_boys * boys_chocolates + num_girls * girls_chocolates = total_chocolates →
  num_boys + num_girls = 120 := by
sorry

end NUMINAMATH_CALUDE_chocolate_distribution_l2814_281463


namespace NUMINAMATH_CALUDE_no_unchanged_sum_l2814_281426

theorem no_unchanged_sum : ¬∃ (A B : ℕ), A + B = 2022 ∧ A / 2 + 3 * B = A + B := by
  sorry

end NUMINAMATH_CALUDE_no_unchanged_sum_l2814_281426


namespace NUMINAMATH_CALUDE_polar_to_rectangular_l2814_281406

/-- Conversion from polar to rectangular coordinates -/
theorem polar_to_rectangular (r θ : ℝ) :
  r > 0 →
  θ = 11 * π / 6 →
  (r * Real.cos θ, r * Real.sin θ) = (5 * Real.sqrt 3, -5) := by
  sorry

end NUMINAMATH_CALUDE_polar_to_rectangular_l2814_281406


namespace NUMINAMATH_CALUDE_share_purchase_price_l2814_281449

/-- The price at which an investor bought shares, given dividend rate, face value, and return on investment. -/
theorem share_purchase_price 
  (dividend_rate : ℝ) 
  (face_value : ℝ) 
  (roi : ℝ) 
  (h1 : dividend_rate = 0.185) 
  (h2 : face_value = 50) 
  (h3 : roi = 0.25) : 
  ∃ (price : ℝ), price = 37 := by
sorry

end NUMINAMATH_CALUDE_share_purchase_price_l2814_281449


namespace NUMINAMATH_CALUDE_triangle_problem_l2814_281494

def triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem triangle_problem (a b c : ℝ) 
  (h : Real.sqrt (8 - a) + Real.sqrt (a - 8) = abs (c - 17) + b^2 - 30*b + 225) :
  a = 8 ∧ 
  b = 15 ∧ 
  c = 17 ∧
  triangle_inequality a b c ∧
  a^2 + b^2 = c^2 ∧
  a + b + c = 40 ∧
  a * b / 2 = 60 := by
  sorry

end NUMINAMATH_CALUDE_triangle_problem_l2814_281494


namespace NUMINAMATH_CALUDE_multiply_fractions_result_l2814_281466

theorem multiply_fractions_result : (77 / 4) * (5 / 2) = 48 + 1 / 8 := by
  sorry

end NUMINAMATH_CALUDE_multiply_fractions_result_l2814_281466


namespace NUMINAMATH_CALUDE_eugene_pencils_l2814_281450

theorem eugene_pencils (initial : ℕ) (additional : ℕ) (total : ℕ) : 
  initial = 51 → additional = 6 → total = initial + additional → total = 57 := by
  sorry

end NUMINAMATH_CALUDE_eugene_pencils_l2814_281450


namespace NUMINAMATH_CALUDE_intersection_distance_and_max_value_l2814_281480

/-- Curve C₁ in polar coordinates -/
def C₁ (ρ : ℝ) : Prop := ρ = 1

/-- Curve C₂ in parametric form -/
def C₂ (t x y : ℝ) : Prop := x = 1 + t ∧ y = 2 + t

/-- Point M on C₁ -/
def M (x y : ℝ) : Prop := x^2 + y^2 = 1

theorem intersection_distance_and_max_value :
  ∃ (A B : ℝ × ℝ),
    (∀ ρ, C₁ ρ → (A.1^2 + A.2^2 = ρ^2 ∧ B.1^2 + B.2^2 = ρ^2)) ∧
    (∃ t₁ t₂, C₂ t₁ A.1 A.2 ∧ C₂ t₂ B.1 B.2) ∧
    (A.1 - B.1)^2 + (A.2 - B.2)^2 = 2 ∧
    (∀ x y, M x y → (x + 1) * (y + 1) ≤ 3/2 + Real.sqrt 2) ∧
    (∃ x y, M x y ∧ (x + 1) * (y + 1) = 3/2 + Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_intersection_distance_and_max_value_l2814_281480


namespace NUMINAMATH_CALUDE_right_triangle_area_l2814_281405

/-- The area of a right triangle with one leg of 30 inches and a hypotenuse of 34 inches is 240 square inches. -/
theorem right_triangle_area (a b c : ℝ) (h1 : a = 30) (h2 : c = 34) (h3 : a^2 + b^2 = c^2) :
  (1/2) * a * b = 240 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_area_l2814_281405


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l2814_281437

/-- Given a hyperbola with equation y²/2 - x²/8 = 1, its eccentricity is √5 -/
theorem hyperbola_eccentricity :
  ∀ (x y : ℝ), y^2/2 - x^2/8 = 1 → 
  ∃ (e : ℝ), e = Real.sqrt 5 ∧ e = Real.sqrt ((2 + 8) / 2) := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l2814_281437


namespace NUMINAMATH_CALUDE_inequality_problem_l2814_281417

theorem inequality_problem (a b c d : ℝ) 
  (h1 : a > b) (h2 : b > 0) (h3 : c > d) (h4 : d > 0) : 
  (a - d > b - c) ∧ (a / d > b / c) ∧ (a * c > b * d) ∧ ¬(a + d > b + c) := by
  sorry

end NUMINAMATH_CALUDE_inequality_problem_l2814_281417


namespace NUMINAMATH_CALUDE_two_cars_gas_consumption_l2814_281464

/-- Represents the gas consumption and mileage of a car for a week -/
structure CarData where
  mpg : ℝ
  gallons_consumed : ℝ
  miles_driven : ℝ

/-- Calculates the total gas consumption for two cars in a week -/
def total_gas_consumption (car1 : CarData) (car2 : CarData) : ℝ :=
  car1.gallons_consumed + car2.gallons_consumed

/-- Theorem stating the total gas consumption of two cars given specific conditions -/
theorem two_cars_gas_consumption
  (car1 : CarData)
  (car2 : CarData)
  (h1 : car1.mpg = 25)
  (h2 : car2.mpg = 40)
  (h3 : car1.gallons_consumed = 30)
  (h4 : car1.miles_driven + car2.miles_driven = 1825)
  (h5 : car1.miles_driven = car1.mpg * car1.gallons_consumed)
  (h6 : car2.miles_driven = car2.mpg * car2.gallons_consumed) :
  total_gas_consumption car1 car2 = 56.875 := by
  sorry

#eval Float.round ((25 : Float) * 30 + (1825 - 25 * 30) / 40) * 1000 / 1000

end NUMINAMATH_CALUDE_two_cars_gas_consumption_l2814_281464


namespace NUMINAMATH_CALUDE_sqrt_difference_equals_neg_six_sqrt_three_l2814_281418

theorem sqrt_difference_equals_neg_six_sqrt_three :
  Real.sqrt ((5 - 3 * Real.sqrt 3)^2) - Real.sqrt ((5 + 3 * Real.sqrt 3)^2) = -6 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_difference_equals_neg_six_sqrt_three_l2814_281418


namespace NUMINAMATH_CALUDE_jonny_sarah_marble_difference_l2814_281474

/-- The number of marbles Jonny has -/
def jonny_marbles : ℕ := 18

/-- The number of bags Sarah initially has -/
def sarah_bags : ℕ := 4

/-- The number of marbles in each of Sarah's bags -/
def sarah_marbles_per_bag : ℕ := 6

/-- The total number of marbles Sarah initially has -/
def sarah_total_marbles : ℕ := sarah_bags * sarah_marbles_per_bag

/-- The number of marbles Sarah has after giving half to Jared -/
def sarah_remaining_marbles : ℕ := sarah_total_marbles / 2

theorem jonny_sarah_marble_difference :
  jonny_marbles - sarah_remaining_marbles = 6 := by
  sorry

end NUMINAMATH_CALUDE_jonny_sarah_marble_difference_l2814_281474


namespace NUMINAMATH_CALUDE_chess_tournament_participants_l2814_281401

theorem chess_tournament_participants (n : ℕ) (h : n > 0) : 
  (n * (n - 1)) / 2 = 120 → n = 16 := by sorry

end NUMINAMATH_CALUDE_chess_tournament_participants_l2814_281401


namespace NUMINAMATH_CALUDE_five_sixths_of_thirty_l2814_281469

theorem five_sixths_of_thirty : (5 / 6 : ℚ) * 30 = 25 := by
  sorry

end NUMINAMATH_CALUDE_five_sixths_of_thirty_l2814_281469


namespace NUMINAMATH_CALUDE_consecutive_days_sum_l2814_281443

theorem consecutive_days_sum (x : ℕ) : 
  x + (x + 1) + (x + 2) = 33 → x = 10 ∧ x + 1 = 11 ∧ x + 2 = 12 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_days_sum_l2814_281443


namespace NUMINAMATH_CALUDE_rectangle_ratio_is_two_l2814_281461

/-- Geometric arrangement of squares and rectangles -/
structure SquareFrame where
  inner_side : ℝ
  outer_side : ℝ
  rect_short : ℝ
  rect_long : ℝ
  area_ratio : outer_side^2 = 9 * inner_side^2
  outer_side_composition : outer_side = inner_side + 2 * rect_short
  inner_side_composition : inner_side + rect_long = outer_side

/-- Theorem: The ratio of the longer side to the shorter side of each rectangle is 2 -/
theorem rectangle_ratio_is_two (frame : SquareFrame) :
  frame.rect_long / frame.rect_short = 2 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_ratio_is_two_l2814_281461


namespace NUMINAMATH_CALUDE_unique_charming_number_l2814_281414

def is_charming (n : ℕ) : Prop :=
  ∃ a b : ℕ, 
    n = 10 * a + b ∧ 
    1 ≤ a ∧ a ≤ 9 ∧ 
    0 ≤ b ∧ b ≤ 9 ∧
    n = 2 * a + b^3

theorem unique_charming_number : 
  ∃! n : ℕ, is_charming n := by sorry

end NUMINAMATH_CALUDE_unique_charming_number_l2814_281414


namespace NUMINAMATH_CALUDE_abs_value_of_complex_l2814_281422

theorem abs_value_of_complex (z : ℂ) : z = 1 - 2 * Complex.I → Complex.abs z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_abs_value_of_complex_l2814_281422


namespace NUMINAMATH_CALUDE_angle_greater_iff_sin_greater_l2814_281476

-- Define a triangle with angles A, B, and C
structure Triangle where
  A : Real
  B : Real
  C : Real
  angle_sum : A + B + C = π
  positive_angles : 0 < A ∧ 0 < B ∧ 0 < C

-- State the theorem
theorem angle_greater_iff_sin_greater (t : Triangle) : t.A > t.B ↔ Real.sin t.A > Real.sin t.B := by
  sorry

end NUMINAMATH_CALUDE_angle_greater_iff_sin_greater_l2814_281476


namespace NUMINAMATH_CALUDE_bond_return_rate_l2814_281404

-- Define the total investment
def total_investment : ℝ := 10000

-- Define the bank interest rate
def bank_interest_rate : ℝ := 0.05

-- Define the total annual income
def total_annual_income : ℝ := 660

-- Define the amount invested in each method
def investment_per_method : ℝ := 6000

-- Theorem statement
theorem bond_return_rate :
  let bank_income := investment_per_method * bank_interest_rate
  let bond_income := total_annual_income - bank_income
  bond_income / investment_per_method = 0.06 := by
sorry


end NUMINAMATH_CALUDE_bond_return_rate_l2814_281404


namespace NUMINAMATH_CALUDE_quadratic_linear_intersection_l2814_281442

/-- Given a quadratic function f(x) = ax² + bx + c and a linear function g(x) = -bx,
    where a > b > c and f(1) = 0, prove that f and g intersect at two distinct points. -/
theorem quadratic_linear_intersection
  (a b c : ℝ) 
  (h1 : a > b) 
  (h2 : b > c) 
  (h3 : a * 1^2 + b * 1 + c = 0) :
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
    a * x₁^2 + b * x₁ + c = -b * x₁ ∧
    a * x₂^2 + b * x₂ + c = -b * x₂ :=
by sorry

end NUMINAMATH_CALUDE_quadratic_linear_intersection_l2814_281442


namespace NUMINAMATH_CALUDE_mushroom_collection_l2814_281419

theorem mushroom_collection (N : ℕ) : 
  (100 ≤ N ∧ N < 1000) →  -- N is a three-digit number
  (N / 100 + (N / 10) % 10 + N % 10 = 14) →  -- sum of digits is 14
  (N % 50 = 0) →  -- divisible by 50
  (N % 25 = 0) →  -- 8% of N is an integer (since 8% = 2/25)
  (N % 50 = 0) →  -- 14% of N is an integer (since 14% = 7/50)
  N = 950 := by
sorry

end NUMINAMATH_CALUDE_mushroom_collection_l2814_281419


namespace NUMINAMATH_CALUDE_exists_m_divisible_by_1997_l2814_281454

def f (x : ℤ) : ℤ := 3 * x + 2

def f_iter : ℕ → (ℤ → ℤ)
| 0 => id
| n + 1 => f ∘ f_iter n

theorem exists_m_divisible_by_1997 : 
  ∃ m : ℕ+, (1997 : ℤ) ∣ f_iter 99 m.val :=
sorry

end NUMINAMATH_CALUDE_exists_m_divisible_by_1997_l2814_281454


namespace NUMINAMATH_CALUDE_fraction_simplification_l2814_281452

theorem fraction_simplification : (10 : ℝ) / (10 * 11 - 10^2) = 1 := by sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2814_281452


namespace NUMINAMATH_CALUDE_greg_and_earl_final_amount_l2814_281413

/-- Represents the financial state of three individuals and their debts --/
structure FinancialState where
  earl_initial : ℕ
  fred_initial : ℕ
  greg_initial : ℕ
  earl_owes_fred : ℕ
  fred_owes_greg : ℕ
  greg_owes_earl : ℕ

/-- Calculates the final amount Greg and Earl have together after all debts are paid --/
def final_amount (state : FinancialState) : ℕ :=
  (state.earl_initial - state.earl_owes_fred + state.greg_owes_earl) +
  (state.greg_initial + state.fred_owes_greg - state.greg_owes_earl)

/-- Theorem stating that Greg and Earl will have $130 together after all debts are paid --/
theorem greg_and_earl_final_amount (state : FinancialState)
  (h1 : state.earl_initial = 90)
  (h2 : state.fred_initial = 48)
  (h3 : state.greg_initial = 36)
  (h4 : state.earl_owes_fred = 28)
  (h5 : state.fred_owes_greg = 32)
  (h6 : state.greg_owes_earl = 40) :
  final_amount state = 130 := by
  sorry

end NUMINAMATH_CALUDE_greg_and_earl_final_amount_l2814_281413


namespace NUMINAMATH_CALUDE_number_equation_solution_l2814_281473

theorem number_equation_solution : ∃ x : ℝ, (3/5) * x + 7 = (1/4) * x^2 - (1/2) * (1/3) * x := by
  sorry

end NUMINAMATH_CALUDE_number_equation_solution_l2814_281473


namespace NUMINAMATH_CALUDE_remainder_r_15_minus_1_l2814_281485

theorem remainder_r_15_minus_1 (r : ℤ) : (r^15 - 1) % (r + 1) = -2 := by
  sorry

end NUMINAMATH_CALUDE_remainder_r_15_minus_1_l2814_281485


namespace NUMINAMATH_CALUDE_sum_and_diff_of_odds_are_even_l2814_281467

-- Define an odd integer
def is_odd (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k + 1

-- Theorem statement
theorem sum_and_diff_of_odds_are_even (a b : ℤ) 
  (ha : is_odd a) (hb : is_odd b) : 
  ∃ m n : ℤ, a + b = 2 * m ∧ a - b = 2 * n := by
  sorry

end NUMINAMATH_CALUDE_sum_and_diff_of_odds_are_even_l2814_281467


namespace NUMINAMATH_CALUDE_arithmetic_number_difference_l2814_281489

/-- A 3-digit number with distinct digits forming an arithmetic sequence --/
def ArithmeticNumber (n : ℕ) : Prop :=
  ∃ a b c : ℕ,
    n = 100 * a + 10 * b + c ∧
    0 ≤ a ∧ a ≤ 9 ∧
    0 ≤ b ∧ b ≤ 9 ∧
    0 ≤ c ∧ c ≤ 9 ∧
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    b - a = c - b

theorem arithmetic_number_difference : 
  (∃ max min : ℕ, 
    ArithmeticNumber max ∧ 
    ArithmeticNumber min ∧
    (∀ n : ℕ, ArithmeticNumber n → min ≤ n ∧ n ≤ max) ∧
    max - min = 864) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_number_difference_l2814_281489


namespace NUMINAMATH_CALUDE_interest_calculation_l2814_281462

/-- Simple interest calculation function -/
def simpleInterest (principal : ℝ) (rate : ℝ) (time : ℝ := 1) : ℝ :=
  principal * rate * time

theorem interest_calculation 
  (r : ℝ) -- Interest rate as a decimal
  (h1 : simpleInterest 5000 r = 250) -- Condition for the initial investment
  : simpleInterest 20000 r = 1000 := by
  sorry

#check interest_calculation

end NUMINAMATH_CALUDE_interest_calculation_l2814_281462


namespace NUMINAMATH_CALUDE_original_number_proof_l2814_281482

theorem original_number_proof (x : ℝ) (h : 1.40 * x = 700) : x = 500 := by
  sorry

end NUMINAMATH_CALUDE_original_number_proof_l2814_281482


namespace NUMINAMATH_CALUDE_factorization_cubic_minus_xy_squared_l2814_281433

theorem factorization_cubic_minus_xy_squared (x y : ℝ) : 
  x^3 - x*y^2 = x*(x + y)*(x - y) := by sorry

end NUMINAMATH_CALUDE_factorization_cubic_minus_xy_squared_l2814_281433


namespace NUMINAMATH_CALUDE_additional_racks_needed_additional_racks_needed_is_one_l2814_281465

-- Define the given constants
def flour_per_bag : ℕ := 12
def bags_of_flour : ℕ := 5
def cups_per_pound : ℕ := 3
def pounds_per_rack : ℕ := 5
def owned_racks : ℕ := 3

-- Define the theorem
theorem additional_racks_needed : ℕ :=
  let total_flour : ℕ := flour_per_bag * bags_of_flour
  let total_pounds : ℕ := total_flour / cups_per_pound
  let capacity : ℕ := owned_racks * pounds_per_rack
  let remaining : ℕ := total_pounds - capacity
  (remaining + pounds_per_rack - 1) / pounds_per_rack

-- Proof
theorem additional_racks_needed_is_one : additional_racks_needed = 1 := by
  sorry

end NUMINAMATH_CALUDE_additional_racks_needed_additional_racks_needed_is_one_l2814_281465


namespace NUMINAMATH_CALUDE_square_fraction_count_l2814_281451

theorem square_fraction_count : ∃! (n : ℤ), 0 < n ∧ n < 25 ∧ ∃ (k : ℤ), (n : ℚ) / (25 - n) = k^2 := by
  sorry

end NUMINAMATH_CALUDE_square_fraction_count_l2814_281451


namespace NUMINAMATH_CALUDE_x_2021_minus_one_values_l2814_281403

theorem x_2021_minus_one_values (x : ℝ) :
  (x - 1) * (x^5 + x^4 + x^3 + x^2 + x + 1) = 0 →
  x^2021 - 1 = 0 ∨ x^2021 - 1 = -2 := by
sorry

end NUMINAMATH_CALUDE_x_2021_minus_one_values_l2814_281403


namespace NUMINAMATH_CALUDE_ellipse_coincide_hyperbola_focus_l2814_281438

def ellipse_equation (a b x y : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

def hyperbola_equation (x y : ℝ) : Prop := x^2 - y^2 / 3 = 1

def eccentricity (e : ℝ) : Prop := e = 1 / 2

theorem ellipse_coincide_hyperbola_focus (a b : ℝ) :
  eccentricity (1 / 2) →
  (∃ x y, ellipse_equation a b x y) →
  (∃ x y, hyperbola_equation x y) →
  (∀ x y, ellipse_equation a b x y ↔ ellipse_equation 4 (12 : ℝ).sqrt x y) :=
sorry

end NUMINAMATH_CALUDE_ellipse_coincide_hyperbola_focus_l2814_281438


namespace NUMINAMATH_CALUDE_gcd_lcm_product_24_60_l2814_281491

theorem gcd_lcm_product_24_60 : Nat.gcd 24 60 * Nat.lcm 24 60 = 1440 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_product_24_60_l2814_281491


namespace NUMINAMATH_CALUDE_initial_number_proof_l2814_281458

theorem initial_number_proof (x : ℝ) : x - 70 = 70 + 40 → x = 180 := by
  sorry

end NUMINAMATH_CALUDE_initial_number_proof_l2814_281458


namespace NUMINAMATH_CALUDE_book_cost_calculation_l2814_281475

def total_cost : ℝ := 6
def num_books : ℕ := 2

theorem book_cost_calculation :
  (total_cost / num_books : ℝ) = 3 := by sorry

end NUMINAMATH_CALUDE_book_cost_calculation_l2814_281475


namespace NUMINAMATH_CALUDE_combined_degrees_l2814_281486

theorem combined_degrees (summer_degrees jolly_degrees : ℕ) : 
  summer_degrees = 150 → 
  summer_degrees = jolly_degrees + 5 → 
  summer_degrees + jolly_degrees = 295 := by
sorry

end NUMINAMATH_CALUDE_combined_degrees_l2814_281486


namespace NUMINAMATH_CALUDE_smallest_positive_integer_with_remainders_l2814_281488

theorem smallest_positive_integer_with_remainders : ∃ (b : ℕ),
  (b > 0) ∧
  (b % 4 = 3) ∧
  (b % 6 = 5) ∧
  (∀ (x : ℕ), x > 0 ∧ x % 4 = 3 ∧ x % 6 = 5 → x ≥ b) ∧
  (b = 11) := by
sorry

end NUMINAMATH_CALUDE_smallest_positive_integer_with_remainders_l2814_281488


namespace NUMINAMATH_CALUDE_parabola_equation_l2814_281430

/-- A parabola with vertex at the origin and passing through (-4, 4) has the standard equation y² = -4x or x² = 4y -/
theorem parabola_equation (p : ℝ → ℝ → Prop) 
  (vertex_origin : p 0 0)
  (passes_through : p (-4) 4) :
  (∀ x y, p x y ↔ y^2 = -4*x) ∨ (∀ x y, p x y ↔ x^2 = 4*y) :=
sorry

end NUMINAMATH_CALUDE_parabola_equation_l2814_281430


namespace NUMINAMATH_CALUDE_baseball_card_value_decrease_l2814_281407

theorem baseball_card_value_decrease (initial_value : ℝ) (h : initial_value > 0) :
  let value_after_first_year := initial_value * (1 - 0.2)
  let value_after_second_year := value_after_first_year * (1 - 0.1)
  let total_decrease := initial_value - value_after_second_year
  let percent_decrease := (total_decrease / initial_value) * 100
  percent_decrease = 28 := by sorry

end NUMINAMATH_CALUDE_baseball_card_value_decrease_l2814_281407


namespace NUMINAMATH_CALUDE_g_is_self_inverse_l2814_281411

-- Define a function f that is symmetric about y=x-1
def f : ℝ → ℝ := sorry

-- Define the property of f being symmetric about y=x-1
axiom f_symmetric : ∀ x y : ℝ, f x = y ↔ f (y + 1) = x + 1

-- Define g in terms of f
def g : ℝ → ℝ := λ x => f (x + 1)

-- State the theorem
theorem g_is_self_inverse : ∀ x : ℝ, g (g x) = x := by sorry

end NUMINAMATH_CALUDE_g_is_self_inverse_l2814_281411


namespace NUMINAMATH_CALUDE_trapezoid_area_theorem_l2814_281468

/-- Represents a trapezoid with given diagonals and bases -/
structure Trapezoid where
  diagonal1 : ℝ
  diagonal2 : ℝ
  base1 : ℝ
  base2 : ℝ

/-- Calculates the area of a trapezoid -/
def trapezoidArea (t : Trapezoid) : ℝ :=
  sorry

theorem trapezoid_area_theorem (t : Trapezoid) 
  (h1 : t.diagonal1 = 7)
  (h2 : t.diagonal2 = 8)
  (h3 : t.base1 = 3)
  (h4 : t.base2 = 6) :
  trapezoidArea t = 12 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_area_theorem_l2814_281468


namespace NUMINAMATH_CALUDE_smallest_perfect_square_divisible_by_3_and_5_l2814_281490

theorem smallest_perfect_square_divisible_by_3_and_5 : 
  ∃ n : ℕ, n > 0 ∧ (∃ k : ℕ, n = k^2) ∧ 3 ∣ n ∧ 5 ∣ n ∧ 
  ∀ m : ℕ, m > 0 → (∃ j : ℕ, m = j^2) → 3 ∣ m → 5 ∣ m → m ≥ n :=
by
  -- Proof goes here
  sorry

#eval (15 : ℕ)^2  -- Expected output: 225

end NUMINAMATH_CALUDE_smallest_perfect_square_divisible_by_3_and_5_l2814_281490


namespace NUMINAMATH_CALUDE_sara_quarters_l2814_281493

/-- The total number of quarters after receiving additional quarters -/
def total_quarters (initial : ℕ) (additional : ℕ) : ℕ :=
  initial + additional

/-- Theorem: Sara's total quarters is the sum of her initial quarters and additional quarters -/
theorem sara_quarters : total_quarters 21 49 = 70 := by
  sorry

end NUMINAMATH_CALUDE_sara_quarters_l2814_281493


namespace NUMINAMATH_CALUDE_work_completion_time_l2814_281495

/-- 
Given:
- A can complete the work in 60 days
- A and B together can complete the work in 15 days

Prove that B can complete the work alone in 20 days
-/
theorem work_completion_time (a b : ℝ) (h1 : a = 60) (h2 : 1 / a + 1 / b = 1 / 15) : b = 20 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l2814_281495


namespace NUMINAMATH_CALUDE_consecutive_integers_square_sum_odd_l2814_281424

theorem consecutive_integers_square_sum_odd (a b c M : ℤ) : 
  (a = b + 1 ∨ b = a + 1) →  -- a and b are consecutive integers
  c = a * b →               -- c = ab
  M^2 = a^2 + b^2 + c^2 →   -- M^2 = a^2 + b^2 + c^2
  Odd (M^2) :=              -- M^2 is an odd number
by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_square_sum_odd_l2814_281424


namespace NUMINAMATH_CALUDE_symmetry_sum_l2814_281427

/-- Two points are symmetric with respect to the origin if their coordinates are negatives of each other -/
def symmetric_wrt_origin (p1 p2 : ℝ × ℝ) : Prop :=
  p1.1 = -p2.1 ∧ p1.2 = -p2.2

theorem symmetry_sum (a b : ℝ) :
  symmetric_wrt_origin (a, -3) (4, b) → a + b = -1 := by
  sorry

end NUMINAMATH_CALUDE_symmetry_sum_l2814_281427


namespace NUMINAMATH_CALUDE_square_area_from_perimeter_l2814_281457

/-- The area in square centimeters of a square with perimeter 28 dm is 4900 -/
theorem square_area_from_perimeter : 
  let perimeter : ℝ := 28
  let side_length : ℝ := perimeter / 4
  let area_dm : ℝ := side_length ^ 2
  let area_cm : ℝ := area_dm * 100
  area_cm = 4900 := by
sorry

end NUMINAMATH_CALUDE_square_area_from_perimeter_l2814_281457


namespace NUMINAMATH_CALUDE_expand_quadratic_l2814_281432

theorem expand_quadratic (x : ℝ) : (2*x + 3)*(4*x - 9) = 8*x^2 - 6*x - 27 := by
  sorry

end NUMINAMATH_CALUDE_expand_quadratic_l2814_281432


namespace NUMINAMATH_CALUDE_ed_lost_eleven_marbles_l2814_281487

/-- The number of marbles Ed lost -/
def marbles_lost (initial_difference final_difference : ℕ) : ℕ :=
  initial_difference - final_difference

/-- Theorem stating that Ed lost 11 marbles -/
theorem ed_lost_eleven_marbles :
  marbles_lost 19 8 = 11 := by
  sorry

end NUMINAMATH_CALUDE_ed_lost_eleven_marbles_l2814_281487


namespace NUMINAMATH_CALUDE_downstream_distance_84km_l2814_281431

/-- Represents the swimming scenario --/
structure SwimmingScenario where
  current_speed : ℝ
  upstream_distance : ℝ
  upstream_time : ℝ
  downstream_time : ℝ

/-- Calculates the downstream distance given a swimming scenario --/
def downstream_distance (s : SwimmingScenario) : ℝ :=
  sorry

/-- Theorem stating the downstream distance for the given scenario --/
theorem downstream_distance_84km (s : SwimmingScenario) 
  (h1 : s.current_speed = 2.5)
  (h2 : s.upstream_distance = 24)
  (h3 : s.upstream_time = 8)
  (h4 : s.downstream_time = 8) :
  downstream_distance s = 84 := by
  sorry

end NUMINAMATH_CALUDE_downstream_distance_84km_l2814_281431


namespace NUMINAMATH_CALUDE_min_quadrilateral_area_l2814_281459

-- Define the curve E
def curve_E (x y : ℝ) : Prop := y^2 = 4*x

-- Define the tangent circle
def tangent_circle (x y : ℝ) : Prop := x^2 + y^2 - 2*x = 0

-- Define the tangent line
def tangent_line (x : ℝ) : Prop := x = -2

-- Define the point F
def point_F : ℝ × ℝ := (1, 0)

-- Define the quadrilateral area function
def quadrilateral_area (a b c d : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem min_quadrilateral_area :
  ∀ (a b c d : ℝ × ℝ),
    (∃ (m : ℝ), m ≠ 0 ∧
      curve_E a.1 a.2 ∧ curve_E b.1 b.2 ∧ curve_E c.1 c.2 ∧ curve_E d.1 d.2 ∧
      (a.1 - point_F.1) * (c.1 - point_F.1) + (a.2 - point_F.2) * (c.2 - point_F.2) = 0 ∧
      (b.1 - point_F.1) * (d.1 - point_F.1) + (b.2 - point_F.2) * (d.2 - point_F.2) = 0) →
    quadrilateral_area a b c d ≥ 32 :=
sorry

end NUMINAMATH_CALUDE_min_quadrilateral_area_l2814_281459


namespace NUMINAMATH_CALUDE_fair_coin_probability_l2814_281400

theorem fair_coin_probability : 
  let n : ℕ := 8  -- number of coin tosses
  let p : ℚ := 1/2  -- probability of heads for a fair coin
  let favorable_outcomes : ℕ := (n.choose 2) + (n.choose 3) + (n.choose 4)
  let total_outcomes : ℕ := 2^n
  (favorable_outcomes : ℚ) / total_outcomes = 77/128 := by sorry

end NUMINAMATH_CALUDE_fair_coin_probability_l2814_281400


namespace NUMINAMATH_CALUDE_solution_for_x_and_y_l2814_281496

theorem solution_for_x_and_y (a x y : Real) (k : Int) (h1 : x + y = a) (h2 : Real.sin x ^ 2 + Real.sin y ^ 2 = 1 - Real.cos a) (h3 : Real.cos a ≠ 0) :
  x = a / 2 + k * Real.pi ∧ y = a / 2 - k * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_solution_for_x_and_y_l2814_281496


namespace NUMINAMATH_CALUDE_equation_represents_three_lines_l2814_281471

-- Define the equation
def equation (x y : ℝ) : Prop :=
  x^2 * (x + 2*y - 3) = y^2 * (x + 2*y - 3)

-- Define the three lines
def line1 (x y : ℝ) : Prop := y = (3 - x) / 2
def line2 (x y : ℝ) : Prop := y = x
def line3 (x y : ℝ) : Prop := y = -x

-- Theorem statement
theorem equation_represents_three_lines :
  ∃ (p1 p2 p3 : ℝ × ℝ),
    (∀ x y, equation x y ↔ (line1 x y ∨ line2 x y ∨ line3 x y)) ∧
    (line1 p1.1 p1.2 ∧ line2 p2.1 p2.2 ∧ line3 p3.1 p3.2) ∧
    (p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3) :=
by
  sorry

end NUMINAMATH_CALUDE_equation_represents_three_lines_l2814_281471


namespace NUMINAMATH_CALUDE_stratified_sampling_female_count_l2814_281477

/-- Calculates the number of females in a population given stratified sampling data -/
theorem stratified_sampling_female_count 
  (total_population : ℕ) 
  (sample_size : ℕ) 
  (females_in_sample : ℕ) 
  (total_population_pos : 0 < total_population)
  (sample_size_pos : 0 < sample_size)
  (sample_size_le_total : sample_size ≤ total_population)
  (females_in_sample_le_sample : females_in_sample ≤ sample_size) :
  let females_in_population : ℕ := (females_in_sample * total_population) / sample_size
  females_in_population = 760 ∧ females_in_population ≤ total_population :=
by sorry

end NUMINAMATH_CALUDE_stratified_sampling_female_count_l2814_281477


namespace NUMINAMATH_CALUDE_book_arrangement_count_l2814_281479

theorem book_arrangement_count :
  let math_books : ℕ := 4
  let english_books : ℕ := 5
  let science_books : ℕ := 2
  let subject_groups : ℕ := 3
  let total_arrangements : ℕ :=
    (Nat.factorial subject_groups) *
    (Nat.factorial math_books) *
    (Nat.factorial english_books) *
    (Nat.factorial science_books)
  total_arrangements = 34560 := by
  sorry

end NUMINAMATH_CALUDE_book_arrangement_count_l2814_281479


namespace NUMINAMATH_CALUDE_coefficient_x_cubed_is_twenty_l2814_281456

/-- The coefficient of x^3 in the expansion of (2x + 1/(4x))^5 -/
def coefficient_x_cubed : ℚ :=
  let a := 2  -- coefficient of x
  let b := 1 / 4  -- coefficient of 1/x
  let n := 5  -- exponent
  let k := (n - 3) / 2  -- power of x is n - 2k, so n - 2k = 3
  (n.choose k) * (a ^ (n - k)) * (b ^ k)

/-- Theorem stating that the coefficient of x^3 in (2x + 1/(4x))^5 is 20 -/
theorem coefficient_x_cubed_is_twenty : coefficient_x_cubed = 20 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x_cubed_is_twenty_l2814_281456


namespace NUMINAMATH_CALUDE_correct_calculation_l2814_281436

theorem correct_calculation (x : ℤ) : x - 6 = 51 → 6 * x = 342 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l2814_281436


namespace NUMINAMATH_CALUDE_quadratic_trinomial_existence_l2814_281497

theorem quadratic_trinomial_existence (a b c : ℤ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (hab : a ≠ b) (hbc : b ≠ c) (hac : a ≠ c) :
  ∃ (p q r : ℤ), p > 0 ∧ 
    (∀ x : ℤ, p * x^2 + q * x + r = x^3 - (x - a) * (x - b) * (x - c)) ∧
    (p * a^2 + q * a + r = a^3) ∧
    (p * b^2 + q * b + r = b^3) ∧
    (p * c^2 + q * c + r = c^3) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_trinomial_existence_l2814_281497


namespace NUMINAMATH_CALUDE_inverse_variation_problem_l2814_281498

theorem inverse_variation_problem (y z : ℝ) (h1 : y^4 * z^(1/4) = 162) (h2 : y = 6) : z = 1/4096 := by
  sorry

end NUMINAMATH_CALUDE_inverse_variation_problem_l2814_281498


namespace NUMINAMATH_CALUDE_point_2023_coordinates_l2814_281444

/-- Defines the x-coordinate of the nth point in the sequence -/
def x_coord (n : ℕ) : ℤ := 2 * n - 1

/-- Defines the y-coordinate of the nth point in the sequence -/
def y_coord (n : ℕ) : ℤ := (-1 : ℤ) ^ (n - 1) * 2 ^ n

/-- Theorem stating the coordinates of the 2023rd point -/
theorem point_2023_coordinates :
  (x_coord 2023, y_coord 2023) = (4045, 2 ^ 2023) := by
  sorry

end NUMINAMATH_CALUDE_point_2023_coordinates_l2814_281444


namespace NUMINAMATH_CALUDE_product_of_repeating_decimals_l2814_281447

/-- The product of two repeating decimals 0.151515... and 0.353535... is equal to 175/3267 -/
theorem product_of_repeating_decimals : 
  (15 : ℚ) / 99 * (35 : ℚ) / 99 = (175 : ℚ) / 3267 := by
  sorry

end NUMINAMATH_CALUDE_product_of_repeating_decimals_l2814_281447


namespace NUMINAMATH_CALUDE_jenga_blocks_removed_l2814_281483

def blocks_removed (num_players : ℕ) (num_rounds : ℕ) : ℕ :=
  (num_players * num_rounds * (num_rounds + 1)) / 2

def blocks_removed_sixth_round (num_players : ℕ) (num_rounds : ℕ) : ℕ :=
  blocks_removed num_players num_rounds + (num_rounds + 1)

theorem jenga_blocks_removed : 
  let num_players : ℕ := 5
  let num_rounds : ℕ := 5
  blocks_removed_sixth_round num_players num_rounds = 81 := by
  sorry

end NUMINAMATH_CALUDE_jenga_blocks_removed_l2814_281483


namespace NUMINAMATH_CALUDE_trajectory_of_point_m_l2814_281415

/-- The trajectory of point M given a circle and specific conditions -/
theorem trajectory_of_point_m (x y : ℝ) : 
  (∃ m n : ℝ, 
    m^2 + n^2 = 9 ∧  -- P(m, n) is on the circle
    (x - m)^2 + y^2 = ((m - x)^2 + y^2) / 4) -- PM = 2MP'
  → x^2 / 9 + y^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_trajectory_of_point_m_l2814_281415


namespace NUMINAMATH_CALUDE_stating_wholesale_cost_calculation_l2814_281412

/-- The wholesale cost of a sleeping bag -/
def wholesale_cost : ℝ := 24.56

/-- The retailer's profit percentage -/
def profit_percentage : ℝ := 0.14

/-- The selling price of a sleeping bag -/
def selling_price : ℝ := 28

/-- 
Theorem stating that the wholesale cost is correct given the profit percentage and selling price
-/
theorem wholesale_cost_calculation (ε : ℝ) (h : ε > 0) : 
  ∃ (W : ℝ), W > 0 ∧ abs (W - wholesale_cost) < ε ∧ 
  W * (1 + profit_percentage) = selling_price :=
sorry

end NUMINAMATH_CALUDE_stating_wholesale_cost_calculation_l2814_281412


namespace NUMINAMATH_CALUDE_sequence_general_term_l2814_281440

theorem sequence_general_term (a : ℕ → ℝ) :
  (∀ n : ℕ, a n > 0) →
  (a 1 = 1) →
  (∀ n : ℕ, (n + 1) * (a (n + 1))^2 - n * (a n)^2 + (a n) * (a (n + 1)) = 0) →
  (∀ n : ℕ, a n = 1 / n) :=
by sorry

end NUMINAMATH_CALUDE_sequence_general_term_l2814_281440


namespace NUMINAMATH_CALUDE_quadratic_equation_condition_l2814_281402

/-- The equation (m-2)x^2 - 3x = 0 is quadratic in x if and only if m ≠ 2 -/
theorem quadratic_equation_condition (m : ℝ) :
  (∀ x, ∃ a b c : ℝ, a ≠ 0 ∧ (m - 2) * x^2 - 3 * x = a * x^2 + b * x + c) ↔ m ≠ 2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_condition_l2814_281402


namespace NUMINAMATH_CALUDE_initial_meals_proof_l2814_281429

/-- The number of meals initially available for adults -/
def initial_meals : ℕ := 70

/-- The number of children that can be fed with one adult meal -/
def children_per_adult_meal : ℚ := 90 / initial_meals

theorem initial_meals_proof :
  (initial_meals - 21) * children_per_adult_meal = 63 :=
by sorry

end NUMINAMATH_CALUDE_initial_meals_proof_l2814_281429


namespace NUMINAMATH_CALUDE_product_of_roots_l2814_281425

theorem product_of_roots (x : ℝ) : 
  (x^3 - 15*x^2 + 75*x - 125 = 0) → 
  (∃ a b c : ℝ, x^3 - 15*x^2 + 75*x - 125 = (x - a) * (x - b) * (x - c) ∧ a * b * c = 125) :=
by sorry

end NUMINAMATH_CALUDE_product_of_roots_l2814_281425


namespace NUMINAMATH_CALUDE_average_of_solutions_eq_neg_two_thirds_l2814_281416

theorem average_of_solutions_eq_neg_two_thirds : 
  let f (x : ℝ) := 3 * x^2 + 4 * x + 1
  let solutions := {x : ℝ | f x = 28}
  ∃ (x₁ x₂ : ℝ), solutions = {x₁, x₂} ∧ (x₁ + x₂) / 2 = -2/3 := by
  sorry

end NUMINAMATH_CALUDE_average_of_solutions_eq_neg_two_thirds_l2814_281416


namespace NUMINAMATH_CALUDE_candy_bar_cost_l2814_281484

/-- Given that Dan had $4 at the start and $3 left after buying a candy bar,
    prove that the candy bar cost $1. -/
theorem candy_bar_cost (initial_amount : ℕ) (remaining_amount : ℕ) :
  initial_amount = 4 →
  remaining_amount = 3 →
  initial_amount - remaining_amount = 1 :=
by sorry

end NUMINAMATH_CALUDE_candy_bar_cost_l2814_281484


namespace NUMINAMATH_CALUDE_intersection_P_Q_l2814_281460

-- Define the sets P and Q
def P : Set ℝ := {-1, 0, Real.sqrt 2}
def Q : Set ℝ := {y | ∃ θ : ℝ, y = Real.sin θ}

-- State the theorem
theorem intersection_P_Q : P ∩ Q = {-1, 0} := by sorry

end NUMINAMATH_CALUDE_intersection_P_Q_l2814_281460


namespace NUMINAMATH_CALUDE_dvd_packs_cost_l2814_281439

/-- Proves that given the cost of each DVD pack and the number of packs that can be bought,
    the total amount of money available is correct. -/
theorem dvd_packs_cost (cost_per_pack : ℕ) (num_packs : ℕ) (total_money : ℕ) :
  cost_per_pack = 26 → num_packs = 4 → total_money = cost_per_pack * num_packs →
  total_money = 104 := by
  sorry

end NUMINAMATH_CALUDE_dvd_packs_cost_l2814_281439


namespace NUMINAMATH_CALUDE_inequality_proof_l2814_281446

theorem inequality_proof (x y z : ℝ) 
  (h_nonneg_x : x ≥ 0) (h_nonneg_y : y ≥ 0) (h_nonneg_z : z ≥ 0)
  (h_condition : x * y + y * z + z * x = 1) : 
  (1 / (x + y)) + (1 / (y + z)) + (1 / (z + x)) ≥ 5/2 :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l2814_281446


namespace NUMINAMATH_CALUDE_rice_distribution_l2814_281445

theorem rice_distribution (total_weight : ℚ) (num_containers : ℕ) (ounces_per_pound : ℕ) :
  total_weight = 25 / 4 →
  num_containers = 4 →
  ounces_per_pound = 16 →
  (total_weight * ounces_per_pound) / num_containers = 25 := by
  sorry

end NUMINAMATH_CALUDE_rice_distribution_l2814_281445


namespace NUMINAMATH_CALUDE_fruit_punch_water_quarts_l2814_281408

theorem fruit_punch_water_quarts 
  (water_parts juice_parts : ℕ) 
  (total_gallons : ℚ) 
  (quarts_per_gallon : ℕ) : 
  water_parts = 5 → 
  juice_parts = 2 → 
  total_gallons = 3 → 
  quarts_per_gallon = 4 → 
  (water_parts : ℚ) * total_gallons * quarts_per_gallon / (water_parts + juice_parts) = 60 / 7 := by
  sorry

end NUMINAMATH_CALUDE_fruit_punch_water_quarts_l2814_281408


namespace NUMINAMATH_CALUDE_sector_perimeter_l2814_281499

/-- Given a circular sector with area 2 cm² and central angle 4 radians, its perimeter is 6 cm. -/
theorem sector_perimeter (A : ℝ) (α : ℝ) (P : ℝ) : 
  A = 2 → α = 4 → P = 2 * Real.sqrt (2 / α) + Real.sqrt (2 / α) * α → P = 6 := by
  sorry

end NUMINAMATH_CALUDE_sector_perimeter_l2814_281499


namespace NUMINAMATH_CALUDE_f_root_and_positivity_l2814_281441

noncomputable def f (x : ℝ) : ℝ := 2^x - 2/x

theorem f_root_and_positivity :
  (∃! x : ℝ, f x = 0 ∧ x = 1) ∧
  (∀ x : ℝ, x ≠ 0 → (f x > 0 ↔ x < 0 ∨ x > 1)) :=
by sorry

end NUMINAMATH_CALUDE_f_root_and_positivity_l2814_281441


namespace NUMINAMATH_CALUDE_combination_minus_permutation_l2814_281478

-- Define combination
def combination (n k : ℕ) : ℕ :=
  if k > n then 0
  else (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

-- Define permutation
def permutation (n k : ℕ) : ℕ :=
  if k > n then 0
  else (Nat.factorial n) / (Nat.factorial (n - k))

-- Theorem statement
theorem combination_minus_permutation : combination 7 4 - permutation 5 2 = 15 := by
  sorry

end NUMINAMATH_CALUDE_combination_minus_permutation_l2814_281478


namespace NUMINAMATH_CALUDE_student_count_l2814_281472

/-- The number of students in the class -/
def n : ℕ := sorry

/-- The total number of tokens -/
def total_tokens : ℕ := 960

/-- The number of tokens each student gives to the teacher -/
def tokens_to_teacher : ℕ := 4

theorem student_count :
  (n > 0) ∧
  (total_tokens % n = 0) ∧
  (∃ k : ℕ, k > 0 ∧ total_tokens / n - tokens_to_teacher = k ∧ k * (n + 1) = total_tokens) →
  n = 15 := by sorry

end NUMINAMATH_CALUDE_student_count_l2814_281472


namespace NUMINAMATH_CALUDE_infinitely_many_primes_with_special_property_l2814_281420

theorem infinitely_many_primes_with_special_property :
  ∀ k : ℕ, ∃ (p n : ℕ), 
    p > k ∧ 
    Prime p ∧ 
    n > 0 ∧ 
    ¬(n ∣ (p - 1)) ∧ 
    (p ∣ (Nat.factorial n + 1)) :=
by sorry

end NUMINAMATH_CALUDE_infinitely_many_primes_with_special_property_l2814_281420


namespace NUMINAMATH_CALUDE_profit_and_marginal_profit_max_not_equal_l2814_281481

/-- Revenue function --/
def R (x : ℕ+) : ℚ := 3000 * x - 20 * x^2

/-- Cost function --/
def C (x : ℕ+) : ℚ := 500 * x + 4000

/-- Profit function --/
def P (x : ℕ+) : ℚ := R x - C x

/-- Marginal profit function --/
def MP (x : ℕ+) : ℚ := P (x + 1) - P x

/-- The maximum allowed production --/
def max_production : ℕ+ := 100

theorem profit_and_marginal_profit_max_not_equal :
  (∃ x : ℕ+, x ≤ max_production ∧ ∀ y : ℕ+, y ≤ max_production → P y ≤ P x) ≠
  (∃ x : ℕ+, x ≤ max_production ∧ ∀ y : ℕ+, y ≤ max_production → MP y ≤ MP x) :=
by sorry

end NUMINAMATH_CALUDE_profit_and_marginal_profit_max_not_equal_l2814_281481


namespace NUMINAMATH_CALUDE_distance_is_8_sqrt2_div_3_l2814_281410

/-- Two lines l₁ and l₂ in the plane -/
structure ParallelLines where
  a : ℝ
  l₁ : ℝ × ℝ → Prop
  l₂ : ℝ × ℝ → Prop
  l₁_eq : ∀ x y, l₁ (x, y) ↔ x + a * y + 6 = 0
  l₂_eq : ∀ x y, l₂ (x, y) ↔ (a - 2) * x + 3 * y + 2 * a = 0
  parallel : ∃ k : ℝ, k ≠ 0 ∧ ∀ x y, l₁ (x, y) ↔ l₂ (k * x, k * y)

/-- The distance between two parallel lines -/
def distance (lines : ParallelLines) : ℝ := sorry

/-- Theorem: The distance between the parallel lines is 8√2/3 -/
theorem distance_is_8_sqrt2_div_3 (lines : ParallelLines) :
  distance lines = 8 * Real.sqrt 2 / 3 := by sorry

end NUMINAMATH_CALUDE_distance_is_8_sqrt2_div_3_l2814_281410


namespace NUMINAMATH_CALUDE_cereal_eating_time_l2814_281455

/-- The time taken for two people to eat a certain amount of cereal together -/
def time_to_eat_together (fat_rate thin_rate total_amount : ℚ) : ℚ :=
  total_amount / (fat_rate + thin_rate)

theorem cereal_eating_time :
  let fat_rate : ℚ := 1 / 15  -- Mr. Fat's eating rate in pounds per minute
  let thin_rate : ℚ := 1 / 25 -- Mr. Thin's eating rate in pounds per minute
  let total_amount : ℚ := 4   -- Total amount of cereal in pounds
  time_to_eat_together fat_rate thin_rate total_amount = 75 / 2 := by
  sorry

#eval (75 : ℚ) / 2 -- Should output 37.5

end NUMINAMATH_CALUDE_cereal_eating_time_l2814_281455


namespace NUMINAMATH_CALUDE_distance_between_foci_l2814_281453

-- Define the ellipse equation
def ellipse_equation (x y : ℝ) : Prop :=
  Real.sqrt ((x - 4)^2 + (y - 3)^2) + Real.sqrt ((x + 6)^2 + (y - 10)^2) = 24

-- Define the foci
def focus1 : ℝ × ℝ := (4, 3)
def focus2 : ℝ × ℝ := (-6, 10)

-- Theorem: The distance between foci is √149
theorem distance_between_foci :
  Real.sqrt ((focus1.1 - focus2.1)^2 + (focus1.2 - focus2.2)^2) = Real.sqrt 149 := by
  sorry

#check distance_between_foci

end NUMINAMATH_CALUDE_distance_between_foci_l2814_281453


namespace NUMINAMATH_CALUDE_car_speed_problem_l2814_281428

/-- Proves that if a car traveling at speed v km/h takes 15 seconds longer to travel 1 km
    than it would at 48 km/h, then v = 40 km/h. -/
theorem car_speed_problem (v : ℝ) :
  (v > 0) →  -- Ensure speed is positive
  (3600 / v = 3600 / 48 + 15) →  -- Time difference equation
  v = 40 := by
  sorry

end NUMINAMATH_CALUDE_car_speed_problem_l2814_281428


namespace NUMINAMATH_CALUDE_cube_diagonal_l2814_281434

theorem cube_diagonal (surface_area : ℝ) (h : surface_area = 294) :
  let side_length := Real.sqrt (surface_area / 6)
  let diagonal_length := side_length * Real.sqrt 3
  diagonal_length = 7 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_cube_diagonal_l2814_281434


namespace NUMINAMATH_CALUDE_complex_magnitude_proof_l2814_281435

theorem complex_magnitude_proof : 
  let i : ℂ := Complex.I
  let z : ℂ := (1 - i) / i
  Complex.abs z = Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_complex_magnitude_proof_l2814_281435


namespace NUMINAMATH_CALUDE_correct_sum_after_mistake_l2814_281423

/-- Given two two-digit numbers where a ones digit 7 is mistaken for 1
    and a tens digit 4 is mistaken for 6, resulting in a sum of 146,
    prove that the correct sum is 132. -/
theorem correct_sum_after_mistake (a b c d : ℕ) : 
  a ≤ 9 → b ≤ 9 → c ≤ 9 → d ≤ 9 →  -- Ensure all digits are single-digit
  (10 * a + 7) + (40 + d) = 146 →  -- Mistaken sum equation
  (10 * a + 7) + (40 + d) = 132 := by
sorry

end NUMINAMATH_CALUDE_correct_sum_after_mistake_l2814_281423


namespace NUMINAMATH_CALUDE_distinct_values_of_triple_exponent_l2814_281409

-- Define the base number
def base : ℕ := 3

-- Define the function to calculate the number of distinct values
def distinct_values (n : ℕ) : ℕ :=
  -- The actual implementation is not provided, as we're only writing the statement
  sorry

-- Theorem statement
theorem distinct_values_of_triple_exponent :
  distinct_values base = 2 :=
sorry

end NUMINAMATH_CALUDE_distinct_values_of_triple_exponent_l2814_281409


namespace NUMINAMATH_CALUDE_triangle_area_expression_range_l2814_281470

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def arithmeticSequence (t : Triangle) : Prop :=
  ∃ d : ℝ, t.B = t.A + d ∧ t.C = t.B + d

def triangleConditions (t : Triangle) : Prop :=
  arithmeticSequence t ∧ t.b = 7 ∧ t.a + t.c = 13

-- Theorem for the area of the triangle
theorem triangle_area (t : Triangle) (h : triangleConditions t) :
  (1/2) * t.a * t.c * Real.sin t.B = 10 * Real.sqrt 3 := by sorry

-- Theorem for the range of the expression
theorem expression_range (t : Triangle) (h : triangleConditions t) :
  ∃ x : ℝ, x ∈ Set.Ioo 1 2 ∧ 
  x = Real.sqrt 3 * Real.sin t.A + Real.sin (t.C - π/6) := by sorry

end NUMINAMATH_CALUDE_triangle_area_expression_range_l2814_281470


namespace NUMINAMATH_CALUDE_library_shelves_needed_l2814_281448

theorem library_shelves_needed 
  (total_books : ℕ) 
  (sorted_books : ℕ) 
  (books_per_shelf : ℕ) 
  (h1 : total_books = 1500) 
  (h2 : sorted_books = 375) 
  (h3 : books_per_shelf = 45) : 
  (total_books - sorted_books) / books_per_shelf = 25 := by
  sorry

end NUMINAMATH_CALUDE_library_shelves_needed_l2814_281448
