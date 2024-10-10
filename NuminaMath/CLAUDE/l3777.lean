import Mathlib

namespace rectangle_ratio_around_square_l3777_377759

/-- Given a square surrounded by four identical rectangles, this theorem proves
    that the ratio of the longer side to the shorter side of each rectangle is 2,
    when the area of the larger square formed is 9 times that of the inner square. -/
theorem rectangle_ratio_around_square : 
  ∀ (s x y : ℝ),
  s > 0 →  -- inner square side length is positive
  x > y → y > 0 →  -- rectangle dimensions are positive and x is longer
  (s + 2*y)^2 = 9*s^2 →  -- area relation
  (x + s)^2 = 9*s^2 →  -- outer square side length
  x / y = 2 := by
  sorry

end rectangle_ratio_around_square_l3777_377759


namespace quadratic_equations_same_roots_l3777_377794

/-- Two quadratic equations have the same roots if and only if their coefficients are proportional -/
theorem quadratic_equations_same_roots (a₁ b₁ c₁ a₂ b₂ c₂ : ℝ) (ha₁ : a₁ ≠ 0) (ha₂ : a₂ ≠ 0) :
  (∀ x, a₁ * x^2 + b₁ * x + c₁ = 0 ↔ a₂ * x^2 + b₂ * x + c₂ = 0) ↔
  ∃ k : ℝ, k ≠ 0 ∧ a₁ = k * a₂ ∧ b₁ = k * b₂ ∧ c₁ = k * c₂ :=
by sorry

end quadratic_equations_same_roots_l3777_377794


namespace fifth_root_of_1024_l3777_377720

theorem fifth_root_of_1024 : (1024 : ℝ) ^ (1/5) = 4 := by sorry

end fifth_root_of_1024_l3777_377720


namespace curve_C_properties_l3777_377737

-- Define the curve C
def C : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | let (x, y) := p
               (x - 1)^2 + y^2 = 4 * ((x - 4)^2 + y^2)}

-- Define the line l
def l : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | let (x, y) := p
               x - y + 3 = 0}

-- Theorem statement
theorem curve_C_properties :
  -- 1. The equation of C is x^2 + y^2 = 4
  (∀ p : ℝ × ℝ, p ∈ C ↔ let (x, y) := p; x^2 + y^2 = 4) ∧
  -- 2. The minimum distance from C to l is (3√2)/2 - 2
  (∃ d_min : ℝ, d_min = 3 * Real.sqrt 2 / 2 - 2 ∧
    (∀ p ∈ C, ∀ q ∈ l, dist p q ≥ d_min) ∧
    (∃ p ∈ C, ∃ q ∈ l, dist p q = d_min)) ∧
  -- 3. The maximum distance from C to l is 2 + (3√2)/2
  (∃ d_max : ℝ, d_max = 2 + 3 * Real.sqrt 2 / 2 ∧
    (∀ p ∈ C, ∀ q ∈ l, dist p q ≤ d_max) ∧
    (∃ p ∈ C, ∃ q ∈ l, dist p q = d_max)) :=
by sorry

end curve_C_properties_l3777_377737


namespace product_with_decimals_l3777_377708

theorem product_with_decimals (a b c : ℚ) (h : (125 : ℕ) * 384 = 48000) :
  a = 0.125 ∧ b = 3.84 ∧ c = 0.48 → a * b = c := by sorry

end product_with_decimals_l3777_377708


namespace slab_rate_per_sq_meter_l3777_377778

/-- Prove that the rate per square meter for paving a rectangular room is 900 Rs. -/
theorem slab_rate_per_sq_meter (length width total_cost : ℝ) : 
  length = 6 →
  width = 4.75 →
  total_cost = 25650 →
  total_cost / (length * width) = 900 := by
sorry

end slab_rate_per_sq_meter_l3777_377778


namespace inequality_condition_l3777_377751

theorem inequality_condition :
  (∀ a b c d : ℝ, a > b ∧ c > d → a + c > b + d) ∧
  (∃ a b c d : ℝ, a + c > b + d ∧ ¬(a > b ∧ c > d)) :=
by sorry

end inequality_condition_l3777_377751


namespace jimmy_pizza_cost_per_slice_l3777_377799

/-- Represents the cost of a pizza with toppings -/
def pizza_cost (base_cost : ℚ) (num_slices : ℕ) (first_topping_cost : ℚ) 
  (next_two_toppings_cost : ℚ) (rest_toppings_cost : ℚ) (num_toppings : ℕ) : ℚ :=
  let total_cost := base_cost + first_topping_cost + 
    (if num_toppings > 1 then min (num_toppings - 1) 2 * next_two_toppings_cost else 0) +
    (if num_toppings > 3 then (num_toppings - 3) * rest_toppings_cost else 0)
  total_cost / num_slices

theorem jimmy_pizza_cost_per_slice :
  pizza_cost 10 8 2 1 0.5 7 = 2 := by
  sorry

end jimmy_pizza_cost_per_slice_l3777_377799


namespace towel_rate_problem_l3777_377722

/-- Proves that the rate of two towels is 250 given the conditions of the problem -/
theorem towel_rate_problem (price1 price2 avg_price : ℕ) 
  (h1 : price1 = 100)
  (h2 : price2 = 150)
  (h3 : avg_price = 155)
  : ((10 * avg_price) - (3 * price1 + 5 * price2)) / 2 = 250 := by
  sorry

end towel_rate_problem_l3777_377722


namespace lucy_fish_count_l3777_377719

theorem lucy_fish_count (initial_fish : ℝ) (bought_fish : ℝ) : 
  initial_fish = 212.0 → bought_fish = 280.0 → initial_fish + bought_fish = 492.0 := by
  sorry

end lucy_fish_count_l3777_377719


namespace choir_arrangement_l3777_377746

theorem choir_arrangement (n : ℕ) : 
  (∃ k : ℕ, n = k^2 + 11) ∧ 
  (∃ c : ℕ, n = c * (c + 5)) →
  n ≤ 300 :=
by sorry

end choir_arrangement_l3777_377746


namespace odd_function_sum_l3777_377711

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem odd_function_sum (f : ℝ → ℝ) (h_odd : is_odd f) (h_neg : ∀ x, x < 0 → f x = x + 2) :
  f 0 + f 3 = 1 := by
  sorry

end odd_function_sum_l3777_377711


namespace min_value_sum_equality_condition_l3777_377767

theorem min_value_sum (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a / (3 * b) + b / (2 * c^2) + c^2 / (9 * a) ≥ 3 / Real.rpow 54 (1/3) :=
sorry

theorem equality_condition (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a / (3 * b) + b / (2 * c^2) + c^2 / (9 * a) = 3 / Real.rpow 54 (1/3) ↔
  a = 6 * c^2 ∧ b = 2 * c^2 * Real.rpow 54 (1/3) ∧ c = Real.rpow 54 (1/4) :=
sorry

end min_value_sum_equality_condition_l3777_377767


namespace intersection_y_intercept_sum_l3777_377738

/-- Given two lines that intersect at (3,6), prove that the sum of their y-intercepts is 6 -/
theorem intersection_y_intercept_sum (a b : ℝ) : 
  (∀ x y : ℝ, x = (1/3)*y + a ∧ y = (1/3)*x + b → (x = 3 ∧ y = 6)) → 
  a + b = 6 := by
sorry

end intersection_y_intercept_sum_l3777_377738


namespace z_in_first_quadrant_l3777_377771

theorem z_in_first_quadrant :
  ∀ (z : ℂ), (z - Complex.I) * (2 - Complex.I) = 5 →
  ∃ (a b : ℝ), z = Complex.mk a b ∧ a > 0 ∧ b > 0 := by
  sorry

end z_in_first_quadrant_l3777_377771


namespace no_odd_3digit_div5_without5_l3777_377731

theorem no_odd_3digit_div5_without5 : 
  ¬∃ n : ℕ, 
    100 ≤ n ∧ n < 1000 ∧  -- 3-digit number
    n % 2 = 1 ∧           -- odd
    n % 5 = 0 ∧           -- divisible by 5
    ∀ d : ℕ, d < 3 → (n / 10^d) % 10 ≠ 5  -- does not contain digit 5
    := by sorry

end no_odd_3digit_div5_without5_l3777_377731


namespace arcsin_arccos_bound_l3777_377739

theorem arcsin_arccos_bound (x y : ℝ) (h : x^2 + y^2 = 1) :
  -5*π/2 ≤ 3 * Real.arcsin x - 2 * Real.arccos y ∧
  3 * Real.arcsin x - 2 * Real.arccos y ≤ π/2 := by
  sorry

end arcsin_arccos_bound_l3777_377739


namespace student_arrangement_count_l3777_377749

/-- The number of male students -/
def num_male : ℕ := 3

/-- The number of female students -/
def num_female : ℕ := 2

/-- The total number of students -/
def total_students : ℕ := num_male + num_female

/-- The number of positions available for female students (not at the ends) -/
def female_positions : ℕ := total_students - 2

/-- The number of ways to arrange female students in available positions -/
def female_arrangements : ℕ := Nat.choose female_positions num_female

/-- The number of ways to arrange the remaining male students -/
def male_arrangements : ℕ := Nat.factorial num_male

/-- The total number of arrangements -/
def total_arrangements : ℕ := female_arrangements * male_arrangements

theorem student_arrangement_count :
  total_arrangements = 36 := by sorry

end student_arrangement_count_l3777_377749


namespace triangle_area_theorem_l3777_377730

theorem triangle_area_theorem (x : ℝ) (h1 : x > 0) : 
  (1/2 : ℝ) * x * 3*x = 120 → x = 4 * Real.sqrt 5 := by
  sorry

end triangle_area_theorem_l3777_377730


namespace no_square_divisible_by_six_between_39_and_120_l3777_377744

theorem no_square_divisible_by_six_between_39_and_120 :
  ¬∃ (x : ℕ), ∃ (y : ℕ), x = y^2 ∧ 6 ∣ x ∧ 39 < x ∧ x < 120 :=
by
  sorry

end no_square_divisible_by_six_between_39_and_120_l3777_377744


namespace gift_amount_proof_l3777_377713

/-- The amount of money Josie received as a gift -/
def gift_amount : ℕ := 50

/-- The cost of one cassette tape -/
def cassette_cost : ℕ := 9

/-- The number of cassette tapes Josie plans to buy -/
def num_cassettes : ℕ := 2

/-- The cost of the headphone set -/
def headphone_cost : ℕ := 25

/-- The amount of money Josie will have left after her purchases -/
def money_left : ℕ := 7

/-- Theorem stating that the gift amount is equal to the sum of the purchases and remaining money -/
theorem gift_amount_proof : 
  gift_amount = num_cassettes * cassette_cost + headphone_cost + money_left :=
by sorry

end gift_amount_proof_l3777_377713


namespace prob_10_or_7_prob_below_7_l3777_377715

-- Define the probabilities for each ring
def p10 : ℝ := 0.21
def p9 : ℝ := 0.23
def p8 : ℝ := 0.25
def p7 : ℝ := 0.28

-- Theorem for the probability of hitting either 10 ring or 7 ring
theorem prob_10_or_7 : p10 + p7 = 0.49 := by sorry

-- Theorem for the probability of scoring below 7 ring
theorem prob_below_7 : 1 - (p10 + p9 + p8 + p7) = 0.03 := by sorry

end prob_10_or_7_prob_below_7_l3777_377715


namespace polygon_interior_angles_l3777_377733

theorem polygon_interior_angles (n : ℕ) : 
  180 * (n - 2) = 1440 → n = 10 := by sorry

end polygon_interior_angles_l3777_377733


namespace game_cost_l3777_377788

theorem game_cost (initial_money : ℕ) (num_toys : ℕ) (toy_cost : ℕ) (game_cost : ℕ) : 
  initial_money = 57 →
  num_toys = 5 →
  toy_cost = 6 →
  initial_money = game_cost + (num_toys * toy_cost) →
  game_cost = 27 := by
sorry

end game_cost_l3777_377788


namespace five_balls_four_boxes_l3777_377734

/-- The number of ways to distribute indistinguishable balls into indistinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ :=
  sorry

/-- Theorem: There are 6 ways to distribute 5 indistinguishable balls into 4 indistinguishable boxes -/
theorem five_balls_four_boxes : distribute_balls 5 4 = 6 := by
  sorry

end five_balls_four_boxes_l3777_377734


namespace quadratic_solution_property_l3777_377772

theorem quadratic_solution_property (f g : ℝ) : 
  (3 * f^2 - 4 * f + 2 = 0) →
  (3 * g^2 - 4 * g + 2 = 0) →
  (f + 2) * (g + 2) = 22/3 := by
sorry

end quadratic_solution_property_l3777_377772


namespace factorial_square_root_problem_l3777_377732

theorem factorial_square_root_problem : (Real.sqrt (Nat.factorial 5 * Nat.factorial 4))^2 = 2880 := by
  sorry

end factorial_square_root_problem_l3777_377732


namespace perpendicular_line_parallel_line_l3777_377773

-- Define the types for our points and lines
def Point := ℝ × ℝ
def Line := ℝ → ℝ → ℝ → Prop

-- Define the intersection point of two lines
def intersection (l1 l2 : Line) : Point :=
  let x := -1
  let y := 2
  (x, y)

-- Define perpendicularity of two lines
def perpendicular (l1 l2 : Line) : Prop :=
  ∃ (m1 m2 : ℝ), m1 * m2 = -1 ∧
    (∀ x y c, l1 x y c ↔ m1 * x + y = c) ∧
    (∀ x y c, l2 x y c ↔ m2 * x + y = c)

-- Define parallelism of two lines
def parallel (l1 l2 : Line) : Prop :=
  ∃ (m c1 c2 : ℝ), 
    (∀ x y c, l1 x y c ↔ m * x + y = c1) ∧
    (∀ x y c, l2 x y c ↔ m * x + y = c2)

-- Define the given lines
def line1 : Line := λ x y c => 3 * x + 4 * y - 5 = c
def line2 : Line := λ x y c => 2 * x + y = c
def line3 : Line := λ x y c => 3 * x - 2 * y - 1 = c

-- State the theorems
theorem perpendicular_line :
  let p := intersection line1 line2
  ∃ (l : Line), l p.1 p.2 (-4) ∧ perpendicular l line3 ∧ 
    ∀ x y c, l x y c ↔ 2 * x + 3 * y - 4 = c := by sorry

theorem parallel_line :
  let p := intersection line1 line2
  ∃ (l : Line), l p.1 p.2 7 ∧ parallel l line3 ∧ 
    ∀ x y c, l x y c ↔ 3 * x - 2 * y + 7 = c := by sorry

end perpendicular_line_parallel_line_l3777_377773


namespace equation_solution_l3777_377757

theorem equation_solution :
  ∃ (x : ℚ), x ≠ 1 ∧ (x^2 - 2*x + 3) / (x - 1) = x + 4 ↔ x = 7/5 := by
  sorry

end equation_solution_l3777_377757


namespace power_product_equality_l3777_377790

theorem power_product_equality : (3^5 * 4^5) * 6^2 = 8957952 := by
  sorry

end power_product_equality_l3777_377790


namespace gcd_powers_of_two_l3777_377769

theorem gcd_powers_of_two : Nat.gcd (2^2024 - 1) (2^2016 - 1) = 2^8 - 1 := by
  sorry

end gcd_powers_of_two_l3777_377769


namespace sqrt_abs_sum_zero_implies_power_l3777_377706

theorem sqrt_abs_sum_zero_implies_power (a b : ℝ) :
  Real.sqrt (a + 2) + |b - 1| = 0 → (a + b) ^ 2017 = -1 := by
  sorry

end sqrt_abs_sum_zero_implies_power_l3777_377706


namespace intersection_of_A_and_B_l3777_377745

def A : Set ℕ := {1, 2, 9}
def B : Set ℕ := {1, 7}

theorem intersection_of_A_and_B : A ∩ B = {1} := by
  sorry

end intersection_of_A_and_B_l3777_377745


namespace transform_trig_function_l3777_377786

/-- Given a function f(x) = (√2/2)(sin x + cos x), 
    applying a horizontal stretch by a factor of 2 
    and a left shift by π/2 results in cos(x/2) -/
theorem transform_trig_function : 
  ∃ (f g : ℝ → ℝ), 
    (∀ x, f x = (Real.sqrt 2 / 2) * (Real.sin x + Real.cos x)) ∧
    (∀ x, g x = f (x / 2 + π / 2)) ∧
    (∀ x, g x = Real.cos (x / 2)) := by
  sorry

end transform_trig_function_l3777_377786


namespace triangle_problem_l3777_377756

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_problem (t : Triangle) : 
  (Real.sqrt 3 * t.c = 2 * t.a * Real.sin t.C) →  -- Condition 2
  (t.A < π / 2) →  -- Condition 3: A is acute
  (t.a = 2 * Real.sqrt 3) →  -- Condition 4
  (1/2 * t.b * t.c * Real.sin t.A = 2 * Real.sqrt 3) →  -- Condition 5: Area
  (t.A = π / 3 ∧ 
   ((t.b = 4 ∧ t.c = 2) ∨ (t.b = 2 ∧ t.c = 4))) := by
  sorry

end triangle_problem_l3777_377756


namespace smallest_a_for_nonprime_polynomial_l3777_377779

theorem smallest_a_for_nonprime_polynomial :
  ∃ (a : ℕ+), (∀ (x : ℤ), ∃ (p q : ℤ), p > 1 ∧ q > 1 ∧ x^4 + (a + 4)^2 = p * q) ∧
  (∀ (b : ℕ+), b < a → ∃ (y : ℤ), ∀ (p q : ℤ), (p > 1 ∧ q > 1 → y^4 + (b + 4)^2 ≠ p * q)) :=
sorry

end smallest_a_for_nonprime_polynomial_l3777_377779


namespace parabola_shift_l3777_377774

/-- Represents a parabola in the xy-plane -/
structure Parabola where
  f : ℝ → ℝ

/-- Shifts a parabola horizontally and vertically -/
def shift (p : Parabola) (h v : ℝ) : Parabola where
  f := fun x => p.f (x + h) + v

theorem parabola_shift :
  let p : Parabola := ⟨fun x => x^2⟩
  let shifted := shift p 2 (-5)
  ∀ x, shifted.f x = (x + 2)^2 - 5 := by sorry

end parabola_shift_l3777_377774


namespace regular_polygon_sides_l3777_377710

/-- A regular polygon with perimeter 150 cm and side length 15 cm has 10 sides. -/
theorem regular_polygon_sides (perimeter : ℝ) (side_length : ℝ) (n : ℕ) : 
  perimeter = 150 ∧ side_length = 15 ∧ perimeter = n * side_length → n = 10 := by
  sorry

end regular_polygon_sides_l3777_377710


namespace school_store_problem_l3777_377736

/-- Represents the cost of pencils and notebooks given certain pricing conditions -/
def school_store_cost (pencil_price notebook_price : ℚ) : Prop :=
  -- 10 pencils and 6 notebooks cost $3.50
  10 * pencil_price + 6 * notebook_price = (3.50 : ℚ) ∧
  -- 4 pencils and 9 notebooks cost $2.70
  4 * pencil_price + 9 * notebook_price = (2.70 : ℚ)

/-- Calculates the total cost including the fixed fee -/
def total_cost (pencil_price notebook_price : ℚ) (pencil_count notebook_count : ℕ) : ℚ :=
  let base_cost := pencil_count * pencil_price + notebook_count * notebook_price
  if pencil_count + notebook_count > 15 then base_cost + (0.50 : ℚ) else base_cost

/-- Theorem stating the cost of 24 pencils and 15 notebooks -/
theorem school_store_problem :
  ∃ (pencil_price notebook_price : ℚ),
    school_store_cost pencil_price notebook_price →
    total_cost pencil_price notebook_price 24 15 = (9.02 : ℚ) := by
  sorry


end school_store_problem_l3777_377736


namespace fifth_occurrence_of_three_sevenths_l3777_377770

/-- Represents a fraction with numerator and denominator -/
structure Fraction where
  numerator : ℕ
  denominator : ℕ+

/-- The sequence of fractions as described in the problem -/
def fractionSequence : ℕ → Fraction := sorry

/-- Two fractions are equivalent if their cross products are equal -/
def areEquivalent (f1 f2 : Fraction) : Prop :=
  f1.numerator * f2.denominator = f2.numerator * f1.denominator

/-- The position of the nth occurrence of a fraction equivalent to the given fraction -/
def positionOfNthOccurrence (f : Fraction) (n : ℕ) : ℕ := sorry

/-- The main theorem to prove -/
theorem fifth_occurrence_of_three_sevenths :
  positionOfNthOccurrence ⟨3, 7⟩ 5 = 1211 := by sorry

end fifth_occurrence_of_three_sevenths_l3777_377770


namespace sugar_mixture_percentage_l3777_377775

/-- Given two solutions, where one fourth of the first solution is replaced by the second solution,
    resulting in a mixture that is 17% sugar, and the second solution is 38% sugar,
    prove that the first solution was 10% sugar. -/
theorem sugar_mixture_percentage (first_solution second_solution final_mixture : ℝ) 
    (h1 : 3/4 * first_solution + 1/4 * second_solution = final_mixture)
    (h2 : final_mixture = 17)
    (h3 : second_solution = 38) :
    first_solution = 10 := by
  sorry

end sugar_mixture_percentage_l3777_377775


namespace geometric_sequence_from_arithmetic_l3777_377754

/-- An arithmetic sequence with non-zero common difference -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d ≠ 0, ∀ n, a (n + 1) - a n = d

/-- A geometric sequence -/
def GeometricSequence (b : ℕ → ℝ) : Prop :=
  ∃ q ≠ 0, ∀ n, b (n + 1) / b n = q

theorem geometric_sequence_from_arithmetic (a b : ℕ → ℝ) :
  ArithmeticSequence a →
  GeometricSequence b →
  b 2 = 5 →
  a 5 = b 1 →
  a 8 = b 2 →
  a 13 = b 3 →
  ∀ n, b n = 3 * (5/3)^(n-1) := by
  sorry

end geometric_sequence_from_arithmetic_l3777_377754


namespace parentheses_number_l3777_377727

theorem parentheses_number (x : ℤ) : x - (-6) = 20 → x = 14 := by
  sorry

end parentheses_number_l3777_377727


namespace sarahs_stamp_collection_value_l3777_377762

/-- The value of a stamp collection given the total number of stamps,
    the number of stamps in a subset, and the value of that subset. -/
def stamp_collection_value (total_stamps : ℕ) (subset_stamps : ℕ) (subset_value : ℚ) : ℚ :=
  (total_stamps : ℚ) * subset_value / (subset_stamps : ℚ)

/-- Theorem stating that Sarah's stamp collection is worth 60 dollars -/
theorem sarahs_stamp_collection_value :
  stamp_collection_value 24 8 20 = 60 := by
  sorry

end sarahs_stamp_collection_value_l3777_377762


namespace total_hamburger_combinations_l3777_377782

/-- The number of different hamburger combinations -/
def hamburger_combinations (num_buns num_condiments num_patty_choices : ℕ) : ℕ :=
  num_buns * (2 ^ num_condiments) * num_patty_choices

/-- Theorem stating the total number of different hamburger combinations -/
theorem total_hamburger_combinations :
  hamburger_combinations 3 9 3 = 4608 := by
  sorry

end total_hamburger_combinations_l3777_377782


namespace system_of_inequalities_l3777_377707

theorem system_of_inequalities (x : ℝ) : 
  3 * (x + 1) < 4 * x + 5 → 2 * x > (x + 6) / 2 → x > 2 := by
  sorry

end system_of_inequalities_l3777_377707


namespace circle_tangent_to_line_l3777_377768

/-- A circle with center (a, 2a) and radius √5 is tangent to the line 2x + y + 1 = 0 
    if and only if its equation is (x-1)² + (y-2)² = 5 -/
theorem circle_tangent_to_line (x y : ℝ) : 
  (∃ a : ℝ, (x - a)^2 + (y - 2*a)^2 = 5 ∧ 
   (|2*a + 2*a + 1| / Real.sqrt 5 = Real.sqrt 5)) ↔ 
  (x - 1)^2 + (y - 2)^2 = 5 :=
sorry

end circle_tangent_to_line_l3777_377768


namespace three_pencils_two_pens_cost_l3777_377728

/-- The cost of a pencil -/
def pencil_cost : ℝ := sorry

/-- The cost of a pen -/
def pen_cost : ℝ := sorry

/-- The first condition: eight pencils and three pens cost $5.20 -/
axiom condition1 : 8 * pencil_cost + 3 * pen_cost = 5.20

/-- The second condition: two pencils and five pens cost $4.40 -/
axiom condition2 : 2 * pencil_cost + 5 * pen_cost = 4.40

/-- Theorem: The cost of three pencils and two pens is $2.5881 -/
theorem three_pencils_two_pens_cost : 
  3 * pencil_cost + 2 * pen_cost = 2.5881 := by sorry

end three_pencils_two_pens_cost_l3777_377728


namespace inequality_equivalence_l3777_377777

theorem inequality_equivalence (x : ℝ) : x + 1 > 3 ↔ x > 2 := by sorry

end inequality_equivalence_l3777_377777


namespace parallel_line_through_point_l3777_377792

/-- Given a line L1 with equation 3x + 6y = 9 and a point P (2, -3),
    prove that the line L2 with equation y = -1/2x - 2 is parallel to L1 and passes through P. -/
theorem parallel_line_through_point (x y : ℝ) :
  (3 * x + 6 * y = 9) →  -- Equation of L1
  (y = -1/2 * x - 2) →   -- Equation of L2
  (∃ m b : ℝ, 3 * x + 6 * y = 9 ↔ y = m * x + b) →  -- L1 can be written in slope-intercept form
  ((-1/2) = m) →  -- Slopes are equal
  ((-1/2) * 2 - 2 = -3) →  -- L2 passes through (2, -3)
  (y = -1/2 * x - 2) ∧ (3 * 2 + 6 * (-3) = 9)  -- L2 is parallel to L1 and passes through (2, -3)
:= by sorry

end parallel_line_through_point_l3777_377792


namespace range_of_m_m_value_when_sum_eq_neg_product_l3777_377781

-- Define the quadratic equation
def quadratic (m x : ℝ) : ℝ := x^2 - (2*m - 3)*x + m^2

-- Define the roots of the quadratic equation
def roots (m : ℝ) : Set ℝ := {x | quadratic m x = 0}

-- Theorem for the range of m
theorem range_of_m : ∀ m : ℝ, (∃ x₁ x₂ : ℝ, x₁ ∈ roots m ∧ x₂ ∈ roots m) → m ≤ 3/4 := by sorry

-- Theorem for the value of m when x₁ + x₂ = -x₁x₂
theorem m_value_when_sum_eq_neg_product : 
  ∀ m : ℝ, m ≤ 3/4 → 
  (∃ x₁ x₂ : ℝ, x₁ ∈ roots m ∧ x₂ ∈ roots m ∧ x₁ + x₂ = -(x₁ * x₂)) → 
  m = -3 := by sorry

end range_of_m_m_value_when_sum_eq_neg_product_l3777_377781


namespace power_sum_equality_l3777_377791

theorem power_sum_equality : (2 : ℕ)^(3^2) + (-1 : ℤ)^(2^3) = 513 := by sorry

end power_sum_equality_l3777_377791


namespace wire_cut_ratio_l3777_377747

/-- Given a wire cut into two pieces of lengths x and y, where x forms a square and y forms a regular octagon with equal perimeters, prove that x/y = 1 -/
theorem wire_cut_ratio (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h_square : 4 * (x / 4) = x) 
  (h_octagon : 8 * (y / 8) = y)
  (h_equal_perimeter : 4 * (x / 4) = 8 * (y / 8)) : 
  x / y = 1 := by
  sorry

end wire_cut_ratio_l3777_377747


namespace trigonometric_identity_l3777_377714

theorem trigonometric_identity (α β γ : Real) 
  (h : Real.sin α + Real.sin γ = 2 * Real.sin β) : 
  Real.tan ((α + β) / 2) + Real.tan ((β + γ) / 2) = 2 * Real.tan ((γ + α) / 2) := by
  sorry

end trigonometric_identity_l3777_377714


namespace cave_depth_l3777_377787

/-- The depth of the cave given the current depth and remaining distance -/
theorem cave_depth (current_depth remaining_distance : ℕ) 
  (h1 : current_depth = 588)
  (h2 : remaining_distance = 386) : 
  current_depth + remaining_distance = 974 := by
  sorry

end cave_depth_l3777_377787


namespace bob_distance_when_met_l3777_377789

/-- The distance between points X and Y in miles -/
def total_distance : ℝ := 17

/-- Yolanda's speed for the first half of the journey in miles per hour -/
def yolanda_speed1 : ℝ := 3

/-- Yolanda's speed for the second half of the journey in miles per hour -/
def yolanda_speed2 : ℝ := 4

/-- Bob's speed for the first half of the journey in miles per hour -/
def bob_speed1 : ℝ := 4

/-- Bob's speed for the second half of the journey in miles per hour -/
def bob_speed2 : ℝ := 3

/-- The time in hours that Yolanda starts walking before Bob -/
def head_start : ℝ := 1

/-- The distance Bob walked when they met -/
def bob_distance : ℝ := 8.5004

theorem bob_distance_when_met :
  ∃ (t : ℝ), 
    t > 0 ∧ 
    t < total_distance / 2 / bob_speed1 ∧
    bob_distance = bob_speed1 * t ∧
    total_distance = 
      yolanda_speed1 * (total_distance / 2 / yolanda_speed1) +
      yolanda_speed2 * (total_distance / 2 / yolanda_speed2) +
      bob_speed1 * t +
      bob_speed2 * ((total_distance / 2 / bob_speed1 + total_distance / 2 / bob_speed2 - head_start) - t) :=
by sorry

end bob_distance_when_met_l3777_377789


namespace square_root_of_increased_number_l3777_377709

theorem square_root_of_increased_number (x : ℝ) (h : x ≥ 0) :
  Real.sqrt (x^2 + 2) = Real.sqrt ((Real.sqrt x^2) + 2) :=
by sorry

end square_root_of_increased_number_l3777_377709


namespace carlos_jogging_distance_l3777_377703

/-- Calculates the distance traveled given a constant speed and time -/
def distance (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- Theorem: Given Carlos' jogging speed and time, prove the distance he jogged -/
theorem carlos_jogging_distance :
  let jogging_speed : ℝ := 4
  let jogging_time : ℝ := 2
  distance jogging_speed jogging_time = 8 := by
  sorry

end carlos_jogging_distance_l3777_377703


namespace three_lines_theorem_l3777_377766

/-- Three lines in the plane -/
structure ThreeLines where
  l1 : Real → Real → Prop
  l2 : Real → Real → Prop
  l3 : Real → Real → Real → Prop

/-- The condition that the three lines divide the plane into six parts -/
def divides_into_six_parts (lines : ThreeLines) : Prop := sorry

/-- The main theorem -/
theorem three_lines_theorem (k : Real) :
  let lines : ThreeLines := {
    l1 := λ x y => x - 2*y + 1 = 0,
    l2 := λ x _ => x - 1 = 0,
    l3 := λ x y k => x + k*y = 0
  }
  divides_into_six_parts lines → k ∈ ({0, -1, -2} : Set Real) := by
  sorry

end three_lines_theorem_l3777_377766


namespace josephus_69_l3777_377796

/-- The Josephus function that returns the last remaining number given n. -/
def josephus (n : ℕ) : ℕ :=
  sorry

/-- The theorem stating that the Josephus number for n = 69 is 10. -/
theorem josephus_69 : josephus 69 = 10 := by
  sorry

end josephus_69_l3777_377796


namespace equality_of_arithmetic_progressions_l3777_377726

theorem equality_of_arithmetic_progressions (a b c d : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)
  (h1 : ∃ r : ℝ, b^2 - a^2 = r ∧ c^2 - b^2 = r ∧ d^2 - c^2 = r)
  (h2 : ∃ s : ℝ, 1/(a+b+d) - 1/(a+b+c) = s ∧ 
               1/(a+c+d) - 1/(a+b+d) = s ∧ 
               1/(b+c+d) - 1/(a+c+d) = s) :
  a = b ∧ b = c ∧ c = d := by
sorry

end equality_of_arithmetic_progressions_l3777_377726


namespace sqrt_sum_equals_six_implies_product_l3777_377785

theorem sqrt_sum_equals_six_implies_product (x : ℝ) :
  Real.sqrt (8 + x) + Real.sqrt (15 - x) = 6 →
  (8 + x) * (15 - x) = 169 / 4 := by
  sorry

end sqrt_sum_equals_six_implies_product_l3777_377785


namespace inequality_proof_l3777_377797

theorem inequality_proof (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  a^2 + 4*b^2 + 1/(a*b) ≥ 4 := by
  sorry

end inequality_proof_l3777_377797


namespace decimal_sum_difference_l3777_377783

theorem decimal_sum_difference : 0.5 - 0.03 + 0.007 + 0.0008 = 0.4778 := by
  sorry

end decimal_sum_difference_l3777_377783


namespace david_widget_production_difference_l3777_377780

/-- Given David's widget production rates and hours worked, prove the difference between Monday and Tuesday production. -/
theorem david_widget_production_difference 
  (t : ℕ) -- Number of hours worked on Monday
  (w : ℕ) -- Widgets produced per hour on Monday
  (h1 : w = 2 * t) -- Relation between w and t
  : w * t - (w + 5) * (t - 3) = t + 15 := by
  sorry

end david_widget_production_difference_l3777_377780


namespace existence_of_d_l3777_377764

theorem existence_of_d : ∃ d : ℝ,
  (∃ n : ℤ, n = ⌊d⌋ ∧ 3 * (n : ℝ)^2 + 20 * (n : ℝ) - 67 = 0) ∧
  (4 * (d - ⌊d⌋)^2 - 15 * (d - ⌊d⌋) + 5 = 0) ∧
  (0 ≤ d - ⌊d⌋ ∧ d - ⌊d⌋ < 1) ∧
  d = -8.63 := by
  sorry

end existence_of_d_l3777_377764


namespace square_perimeter_l3777_377718

theorem square_perimeter (rectangle_length rectangle_width : ℝ) 
  (h1 : rectangle_length = 32)
  (h2 : rectangle_width = 64)
  (h3 : square_area = 2 * rectangle_length * rectangle_width) : 
  4 * Real.sqrt square_area = 256 :=
by
  sorry

end square_perimeter_l3777_377718


namespace triangle_conditions_equivalence_l3777_377776

theorem triangle_conditions_equivalence (x : ℝ) :
  (∀ (BC AC AB : ℝ),
    BC = x + 11 ∧ AC = x + 6 ∧ AB = 3*x + 2 →
    AB + AC > BC ∧ AB + BC > AC ∧ AC + BC > AB ∧
    BC > AB ∧ BC > AC) ↔
  (1 < x ∧ x < 4.5) :=
sorry

end triangle_conditions_equivalence_l3777_377776


namespace product_equals_243_l3777_377784

theorem product_equals_243 : 
  (1/3 : ℚ) * 9 * (1/27 : ℚ) * 81 * (1/243 : ℚ) * 729 * (1/2187 : ℚ) * 6561 * (1/19683 : ℚ) * 59049 = 243 := by
  sorry

end product_equals_243_l3777_377784


namespace max_yellow_apples_removal_max_total_apples_removal_l3777_377701

/-- Represents the number of apples of each color in the basket -/
structure AppleBasket where
  green : Nat
  yellow : Nat
  red : Nat

/-- Represents the number of apples removed from the basket -/
structure RemovedApples where
  green : Nat
  yellow : Nat
  red : Nat

/-- Checks if the removal condition is satisfied -/
def validRemoval (removed : RemovedApples) : Prop :=
  removed.green < removed.yellow ∧ removed.yellow < removed.red

/-- The initial state of the apple basket -/
def initialBasket : AppleBasket :=
  ⟨8, 11, 16⟩

theorem max_yellow_apples_removal (basket : AppleBasket) 
  (h : basket = initialBasket) :
  ∃ (removed : RemovedApples), 
    validRemoval removed ∧ 
    removed.yellow = 11 ∧
    ∀ (other : RemovedApples), 
      validRemoval other → other.yellow ≤ removed.yellow :=
sorry

theorem max_total_apples_removal (basket : AppleBasket) 
  (h : basket = initialBasket) :
  ∃ (removed : RemovedApples),
    validRemoval removed ∧
    removed.green + removed.yellow + removed.red = 33 ∧
    ∀ (other : RemovedApples),
      validRemoval other →
      other.green + other.yellow + other.red ≤ removed.green + removed.yellow + removed.red :=
sorry

end max_yellow_apples_removal_max_total_apples_removal_l3777_377701


namespace binomial_and_permutation_7_5_l3777_377724

theorem binomial_and_permutation_7_5 :
  (Nat.choose 7 5 = 21) ∧ (Nat.factorial 7 / Nat.factorial 2 = 2520) := by
  sorry

end binomial_and_permutation_7_5_l3777_377724


namespace vasya_lowest_position_l3777_377793

/-- Represents a cyclist in the race -/
structure Cyclist :=
  (id : Nat)

/-- Represents a stage in the race -/
structure Stage :=
  (number : Nat)

/-- Represents the time a cyclist takes to complete a stage -/
structure StageTime :=
  (cyclist : Cyclist)
  (stage : Stage)
  (time : ℝ)

/-- Represents the total time a cyclist takes to complete all stages -/
structure TotalTime :=
  (cyclist : Cyclist)
  (time : ℝ)

/-- The number of cyclists in the race -/
def numCyclists : Nat := 500

/-- The number of stages in the race -/
def numStages : Nat := 15

/-- Vasya's position in each stage -/
def vasyaStagePosition : Nat := 7

/-- Function to get a cyclist's position in a stage -/
def stagePosition (c : Cyclist) (s : Stage) : Nat := sorry

/-- Function to get a cyclist's overall position -/
def overallPosition (c : Cyclist) : Nat := sorry

/-- Vasya's cyclist object -/
def vasya : Cyclist := ⟨0⟩  -- Assuming Vasya's ID is 0

/-- The main theorem -/
theorem vasya_lowest_position :
  (∀ s : Stage, stagePosition vasya s = vasyaStagePosition) →
  (∀ c1 c2 : Cyclist, ∀ s : Stage, c1 ≠ c2 → stagePosition c1 s ≠ stagePosition c2 s) →
  (∀ c1 c2 : Cyclist, c1 ≠ c2 → overallPosition c1 ≠ overallPosition c2) →
  overallPosition vasya ≤ 91 := sorry

end vasya_lowest_position_l3777_377793


namespace two_bedroom_units_l3777_377755

theorem two_bedroom_units (total_units : ℕ) (cost_one_bedroom : ℕ) (cost_two_bedroom : ℕ) (total_cost : ℕ) :
  total_units = 12 →
  cost_one_bedroom = 360 →
  cost_two_bedroom = 450 →
  total_cost = 4950 →
  ∃ (one_bedroom_units two_bedroom_units : ℕ),
    one_bedroom_units + two_bedroom_units = total_units ∧
    cost_one_bedroom * one_bedroom_units + cost_two_bedroom * two_bedroom_units = total_cost ∧
    two_bedroom_units = 7 :=
by sorry

end two_bedroom_units_l3777_377755


namespace root_equation_a_value_l3777_377740

theorem root_equation_a_value (a b : ℚ) : 
  ((-2 : ℝ) - 5 * Real.sqrt 3)^3 + a * ((-2 : ℝ) - 5 * Real.sqrt 3)^2 + 
  b * ((-2 : ℝ) - 5 * Real.sqrt 3) - 48 = 0 → a = 4 := by
sorry

end root_equation_a_value_l3777_377740


namespace faye_pencil_rows_l3777_377700

def total_pencils : ℕ := 35
def pencils_per_row : ℕ := 5

theorem faye_pencil_rows :
  total_pencils / pencils_per_row = 7 := by
  sorry

end faye_pencil_rows_l3777_377700


namespace competition_participants_l3777_377760

theorem competition_participants : ℕ :=
  let initial_participants : ℕ := sorry
  let first_round_survival_rate : ℚ := 1 / 3
  let second_round_survival_rate : ℚ := 1 / 4
  let final_participants : ℕ := 18

  have h1 : (initial_participants : ℚ) * first_round_survival_rate * second_round_survival_rate = final_participants := by sorry

  initial_participants

end competition_participants_l3777_377760


namespace sum_not_divisible_by_ten_l3777_377712

theorem sum_not_divisible_by_ten (n : ℕ) :
  ¬(10 ∣ (1981^n + 1982^n + 1983^n + 1984^n)) ↔ 4 ∣ n :=
sorry

end sum_not_divisible_by_ten_l3777_377712


namespace nancy_savings_l3777_377758

-- Define the value of a dozen
def dozen : ℕ := 12

-- Define the value of a quarter in cents
def quarter_value : ℕ := 25

-- Define the number of cents in a dollar
def cents_per_dollar : ℕ := 100

-- Theorem statement
theorem nancy_savings (nancy_quarters : ℕ) (h1 : nancy_quarters = dozen) : 
  (nancy_quarters * quarter_value) / cents_per_dollar = 3 := by
  sorry

end nancy_savings_l3777_377758


namespace special_action_figure_value_prove_special_figure_value_l3777_377743

theorem special_action_figure_value 
  (total_figures : Nat) 
  (regular_figure_value : Nat) 
  (regular_figure_count : Nat) 
  (discount : Nat) 
  (total_earnings : Nat) : Nat :=
  let special_figure_count := total_figures - regular_figure_count
  let regular_figures_earnings := regular_figure_count * (regular_figure_value - discount)
  let special_figure_earnings := total_earnings - regular_figures_earnings
  special_figure_earnings + discount

theorem prove_special_figure_value :
  special_action_figure_value 5 15 4 5 55 = 20 := by
  sorry

end special_action_figure_value_prove_special_figure_value_l3777_377743


namespace largest_tile_size_is_correct_courtyard_largest_tile_size_l3777_377735

/-- The largest square tile size that can exactly pave a rectangular courtyard -/
def largest_tile_size (length width : ℕ) : ℕ :=
  Nat.gcd length width

theorem largest_tile_size_is_correct (length width : ℕ) (h1 : length > 0) (h2 : width > 0) :
  let tile_size := largest_tile_size length width
  ∀ n : ℕ, n > tile_size → ¬(n ∣ length ∧ n ∣ width) :=
by sorry

theorem courtyard_largest_tile_size :
  largest_tile_size 378 595 = 7 :=
by sorry

end largest_tile_size_is_correct_courtyard_largest_tile_size_l3777_377735


namespace triangle_inequality_check_l3777_377729

def canFormTriangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

theorem triangle_inequality_check : 
  (¬ canFormTriangle 3 5 10) ∧ 
  (canFormTriangle 5 4 8) ∧ 
  (¬ canFormTriangle 2 4 6) ∧ 
  (¬ canFormTriangle 3 3 7) :=
by sorry

end triangle_inequality_check_l3777_377729


namespace max_section_area_is_two_l3777_377725

/-- Represents a cone with its lateral surface unfolded into a sector -/
structure UnfoldedCone where
  radius : ℝ
  centralAngle : ℝ

/-- Calculates the maximum area of a section determined by two generatrices of the cone -/
def maxSectionArea (cone : UnfoldedCone) : ℝ :=
  sorry

/-- Theorem stating that for a cone with lateral surface unfolded into a sector
    with radius 2 and central angle 5π/3, the maximum section area is 2 -/
theorem max_section_area_is_two :
  let cone : UnfoldedCone := ⟨2, 5 * Real.pi / 3⟩
  maxSectionArea cone = 2 :=
sorry

end max_section_area_is_two_l3777_377725


namespace initial_ratio_is_11_to_9_l3777_377761

/-- Represents the contents of a can with milk and water -/
structure CanContents where
  milk : ℝ
  water : ℝ

/-- Proves that the initial ratio of milk to water is 11:9 given the conditions -/
theorem initial_ratio_is_11_to_9 (can : CanContents) : 
  can.milk + can.water = 20 → -- Initial contents
  can.milk + can.water + 10 = 30 → -- Adding 10L fills the can
  (can.milk + 10) / can.water = 5 / 2 → -- Resulting ratio is 5:2
  can.milk / can.water = 11 / 9 := by
  sorry

/-- Verify the solution satisfies the conditions -/
example : 
  let can : CanContents := { milk := 11, water := 9 }
  can.milk + can.water = 20 ∧
  can.milk + can.water + 10 = 30 ∧
  (can.milk + 10) / can.water = 5 / 2 ∧
  can.milk / can.water = 11 / 9 := by
  sorry

end initial_ratio_is_11_to_9_l3777_377761


namespace infinitely_many_divisible_pairs_l3777_377750

def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

theorem infinitely_many_divisible_pairs :
  ∀ k : ℕ, ∃ m n : ℕ+,
    (m : ℕ) ∣ (n : ℕ)^2 + 1 ∧
    (n : ℕ) ∣ (m : ℕ)^2 + 1 ∧
    (m : ℕ) = fib (2 * k + 1) ∧
    (n : ℕ) = fib (2 * k + 3) :=
by sorry

end infinitely_many_divisible_pairs_l3777_377750


namespace food_to_budget_ratio_l3777_377752

def budget : ℚ := 3000
def supplies_fraction : ℚ := 1/4
def wages : ℚ := 1250

def food_expense : ℚ := budget - supplies_fraction * budget - wages

theorem food_to_budget_ratio :
  food_expense / budget = 1/3 := by sorry

end food_to_budget_ratio_l3777_377752


namespace triangle_property_l3777_377795

theorem triangle_property (a b c : ℝ) (A B C : ℝ) :
  let m : ℝ × ℝ := (a, Real.sqrt 3 * b)
  let n : ℝ × ℝ := (Real.cos (π / 2 - B), Real.cos (π - A))
  m.1 * n.1 + m.2 * n.2 = 0 →  -- m ⊥ n
  c = 3 →
  (1 / 2) * b * c * Real.sin A = (3 * Real.sqrt 3) / 2 →  -- Area formula
  A = π / 3 ∧ a = Real.sqrt 7 :=
by sorry

end triangle_property_l3777_377795


namespace circle_equation_l3777_377721

/-- A circle with center (a, b) and radius r -/
structure Circle where
  a : ℝ
  b : ℝ
  r : ℝ

/-- The standard equation of a circle -/
def Circle.equation (c : Circle) (x y : ℝ) : Prop :=
  (x - c.a)^2 + (y - c.b)^2 = c.r^2

/-- A circle is tangent to the x-axis -/
def Circle.tangentToXAxis (c : Circle) : Prop :=
  c.b = c.r

/-- The center of the circle lies on the line y = 3x -/
def Circle.centerOnLine (c : Circle) : Prop :=
  c.b = 3 * c.a

/-- The chord intercepted by the circle on the line y = x has length 2√7 -/
def Circle.chordLength (c : Circle) : Prop :=
  2 * c.r^2 = (c.a - c.b)^2 + 14

/-- The main theorem -/
theorem circle_equation (c : Circle) 
  (h1 : c.tangentToXAxis)
  (h2 : c.centerOnLine)
  (h3 : c.chordLength) :
  (c.equation 1 3 ∧ c.r^2 = 9) ∨ (c.equation (-1) 3 ∧ c.r^2 = 9) :=
sorry

end circle_equation_l3777_377721


namespace dealership_sales_forecast_l3777_377742

theorem dealership_sales_forecast (sports_cars sedan_cars : ℕ) : 
  (5 : ℚ) / 8 = sports_cars / sedan_cars →
  sports_cars = 35 →
  sedan_cars = 56 := by
sorry

end dealership_sales_forecast_l3777_377742


namespace min_value_theorem_l3777_377798

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 3 * y = 5 * x * y) :
  3 * x + 4 * y ≥ 5 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ x₀ + 3 * y₀ = 5 * x₀ * y₀ ∧ 3 * x₀ + 4 * y₀ = 5 :=
sorry

end min_value_theorem_l3777_377798


namespace product_sum_inequality_l3777_377717

theorem product_sum_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a * b * c = 1) :
  (a + b) * (b + c) * (c + a) ≥ 4 * (a + b + c - 1) := by
  sorry

end product_sum_inequality_l3777_377717


namespace perpendicular_vectors_l3777_377748

-- Define the vectors
def a : Fin 2 → ℝ := ![1, 2]
def b (m : ℝ) : Fin 2 → ℝ := ![2, m]

-- Define the sum of vectors
def vector_sum (v w : Fin 2 → ℝ) : Fin 2 → ℝ := λ i => v i + w i

-- Define dot product
def dot_product (v w : Fin 2 → ℝ) : ℝ := (v 0) * (w 0) + (v 1) * (w 1)

-- Theorem statement
theorem perpendicular_vectors (m : ℝ) : 
  dot_product (vector_sum a (b m)) a = 0 ↔ m = -7/2 := by
  sorry

end perpendicular_vectors_l3777_377748


namespace plot_length_is_60_l3777_377705

/-- Represents a rectangular plot with its dimensions and fencing cost. -/
structure RectangularPlot where
  breadth : ℝ
  length : ℝ
  fencingCostPerMeter : ℝ
  totalFencingCost : ℝ

/-- Calculates the perimeter of a rectangular plot. -/
def perimeter (plot : RectangularPlot) : ℝ :=
  2 * (plot.length + plot.breadth)

/-- Theorem stating the length of the plot given the conditions. -/
theorem plot_length_is_60 (plot : RectangularPlot)
  (h1 : plot.length = plot.breadth + 20)
  (h2 : plot.fencingCostPerMeter = 26.5)
  (h3 : plot.totalFencingCost = 5300)
  (h4 : plot.totalFencingCost = plot.fencingCostPerMeter * perimeter plot) :
  plot.length = 60 := by
  sorry

#check plot_length_is_60

end plot_length_is_60_l3777_377705


namespace hotdog_sales_l3777_377704

theorem hotdog_sales (small_hotdogs : ℕ) (total_hotdogs : ℕ) (large_hotdogs : ℕ)
  (h1 : small_hotdogs = 58)
  (h2 : total_hotdogs = 79)
  (h3 : total_hotdogs = small_hotdogs + large_hotdogs) :
  large_hotdogs = 21 := by
  sorry

end hotdog_sales_l3777_377704


namespace actual_lawn_area_l3777_377716

/-- Actual area of a lawn given its blueprint measurements and scale -/
theorem actual_lawn_area 
  (blueprint_area : ℝ) 
  (blueprint_side : ℝ) 
  (actual_side : ℝ) 
  (h1 : blueprint_area = 300) 
  (h2 : blueprint_side = 5) 
  (h3 : actual_side = 1500) : 
  (actual_side / blueprint_side)^2 * blueprint_area = 2700 * 10000 := by
  sorry

end actual_lawn_area_l3777_377716


namespace present_price_l3777_377753

theorem present_price (original_price : ℝ) (discount_rate : ℝ) (num_people : ℕ) 
  (individual_savings : ℝ) :
  original_price > 0 →
  discount_rate = 0.2 →
  num_people = 3 →
  individual_savings = 4 →
  original_price * (1 - discount_rate) = num_people * individual_savings →
  original_price * (1 - discount_rate) = 48 := by
sorry

end present_price_l3777_377753


namespace school_students_count_l3777_377763

theorem school_students_count (boys girls : ℕ) 
  (h1 : 2 * boys / 3 + 3 * girls / 4 = 550)
  (h2 : 3 * girls / 4 = 150) : 
  boys + girls = 800 := by
  sorry

end school_students_count_l3777_377763


namespace line_symmetry_about_bisector_l3777_377765

/-- Given two lines l₁ and l₂ with angle bisector y = x, prove that if l₁ has equation ax + by + c = 0 (ab > 0), then l₂ has equation bx + ay + c = 0 -/
theorem line_symmetry_about_bisector (a b c : ℝ) (hab : a * b > 0) :
  let l₁ := {p : ℝ × ℝ | a * p.1 + b * p.2 + c = 0}
  let l₂ := {p : ℝ × ℝ | b * p.1 + a * p.2 + c = 0}
  let bisector := {p : ℝ × ℝ | p.1 = p.2}
  (∀ p : ℝ × ℝ, p ∈ bisector → (p ∈ l₁ ↔ p ∈ l₂)) →
  ∀ q : ℝ × ℝ, q ∈ l₂ := by
  sorry

end line_symmetry_about_bisector_l3777_377765


namespace min_distance_sum_parabola_l3777_377741

/-- The minimum distance sum from a point on the parabola y^2 = 8x to two fixed points -/
theorem min_distance_sum_parabola :
  let A : ℝ × ℝ := (1, 0)
  let B : ℝ × ℝ := (6, 5)
  let parabola := {P : ℝ × ℝ | P.2^2 = 8 * P.1}
  ∃ (min_dist : ℝ), min_dist = 8 ∧ 
    ∀ P ∈ parabola, Real.sqrt ((P.1 - A.1)^2 + (P.2 - A.2)^2) + 
                     Real.sqrt ((P.1 - B.1)^2 + (P.2 - B.2)^2) ≥ min_dist :=
by sorry

end min_distance_sum_parabola_l3777_377741


namespace problem_solution_l3777_377702

theorem problem_solution (x y : ℝ) (h1 : (x + y)^2 = 4) (h2 : x * y = -1) : 
  x^2 + y^2 = 6 ∧ (x - y)^2 = 8 := by
  sorry

end problem_solution_l3777_377702


namespace function_composition_equality_l3777_377723

theorem function_composition_equality (C D : ℝ) (h : ℝ → ℝ) (k : ℝ → ℝ)
  (h_def : ∀ x, h x = C * x - 3 * D^2)
  (k_def : ∀ x, k x = D * x + 1)
  (D_neq_neg_one : D ≠ -1)
  (h_k_2_eq_zero : h (k 2) = 0) :
  C = 3 * D^2 / (2 * D + 1) := by
sorry

end function_composition_equality_l3777_377723
