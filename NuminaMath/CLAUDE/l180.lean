import Mathlib

namespace pairings_count_l180_18003

/-- The number of bowls -/
def num_bowls : ℕ := 5

/-- The number of glasses -/
def num_glasses : ℕ := 5

/-- The number of possible pairings when choosing one bowl and one glass -/
def num_pairings : ℕ := num_bowls * num_glasses

/-- Theorem stating that the number of possible pairings is 25 -/
theorem pairings_count : num_pairings = 25 := by
  sorry

end pairings_count_l180_18003


namespace stairs_theorem_l180_18049

def stairs_problem (samir veronica ravi : ℕ) : Prop :=
  samir = 318 ∧
  veronica = (samir / 2) + 18 ∧
  ravi = 2 * veronica ∧
  samir + veronica + ravi = 849

theorem stairs_theorem : ∃ samir veronica ravi : ℕ, stairs_problem samir veronica ravi :=
  sorry

end stairs_theorem_l180_18049


namespace proportion_third_number_l180_18097

theorem proportion_third_number (y : ℝ) : 
  (0.6 : ℝ) / 0.96 = y / 8 → y = 5 := by
  sorry

end proportion_third_number_l180_18097


namespace selection_methods_l180_18094

theorem selection_methods (n_boys n_girls n_select : ℕ) : 
  n_boys = 4 → n_girls = 3 → n_select = 4 →
  (Nat.choose (n_boys + n_girls) n_select) - (Nat.choose n_boys n_select) = 34 := by
  sorry

end selection_methods_l180_18094


namespace student_guinea_pig_difference_l180_18074

/-- The number of fourth-grade classrooms -/
def num_classrooms : ℕ := 5

/-- The number of students in each fourth-grade classroom -/
def students_per_classroom : ℕ := 24

/-- The number of guinea pigs in each fourth-grade classroom -/
def guinea_pigs_per_classroom : ℕ := 2

/-- The difference between the total number of students and guinea pigs in all classrooms -/
theorem student_guinea_pig_difference :
  num_classrooms * students_per_classroom - num_classrooms * guinea_pigs_per_classroom = 110 := by
  sorry

end student_guinea_pig_difference_l180_18074


namespace painter_problem_l180_18021

/-- Given two painters with a work ratio of 2:7 painting a total area of 270 square feet,
    the painter with the larger share paints 210 square feet. -/
theorem painter_problem (total_area : ℕ) (ratio_small : ℕ) (ratio_large : ℕ) :
  total_area = 270 →
  ratio_small = 2 →
  ratio_large = 7 →
  (ratio_large * total_area) / (ratio_small + ratio_large) = 210 :=
by sorry

end painter_problem_l180_18021


namespace max_value_x_minus_y_l180_18026

theorem max_value_x_minus_y (θ : Real) (x y : Real)
  (h1 : x = Real.sin θ)
  (h2 : y = Real.cos θ)
  (h3 : 0 ≤ θ ∧ θ ≤ 2 * Real.pi)
  (h4 : (x^2 + y^2)^2 = x + y) :
  ∃ (θ_max : Real), 
    0 ≤ θ_max ∧ θ_max ≤ 2 * Real.pi ∧
    ∀ (θ' : Real), 0 ≤ θ' ∧ θ' ≤ 2 * Real.pi →
      Real.sin θ' - Real.cos θ' ≤ Real.sin θ_max - Real.cos θ_max ∧
      Real.sin θ_max - Real.cos θ_max = Real.sqrt 2 :=
sorry

end max_value_x_minus_y_l180_18026


namespace exponential_decreasing_implies_cubic_increasing_l180_18093

theorem exponential_decreasing_implies_cubic_increasing
  (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1)
  (h3 : ∀ x y : ℝ, x < y → a^x > a^y) :
  (∀ x y : ℝ, x < y → (2 - a) * x^3 < (2 - a) * y^3) ∧
  (∃ b : ℝ, b > 0 ∧ b ≠ 1 ∧
    (∀ x y : ℝ, x < y → (2 - b) * x^3 < (2 - b) * y^3) ∧
    ¬(∀ x y : ℝ, x < y → b^x > b^y)) :=
by sorry

end exponential_decreasing_implies_cubic_increasing_l180_18093


namespace min_value_theorem_l180_18032

theorem min_value_theorem (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) 
  (h4 : a^2 + a*b + a*c + b*c = 4) : 
  ∀ x y z, x > 0 → y > 0 → z > 0 → x^2 + x*y + x*z + y*z = 4 → 2*a + b + c ≤ 2*x + y + z :=
sorry

end min_value_theorem_l180_18032


namespace complex_multiplication_l180_18062

theorem complex_multiplication (i : ℂ) : i^2 = -1 → i * (2 - i) = 1 + 2*i := by
  sorry

end complex_multiplication_l180_18062


namespace least_number_for_divisibility_l180_18068

theorem least_number_for_divisibility (x : ℕ) : 
  (∀ y : ℕ, y < x → ¬((1056 + y) % 35 = 0 ∧ (1056 + y) % 51 = 0)) ∧
  ((1056 + x) % 35 = 0 ∧ (1056 + x) % 51 = 0) →
  x = 729 := by
sorry

end least_number_for_divisibility_l180_18068


namespace odd_function_property_l180_18040

-- Define odd functions
def OddFunction (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- Define the function F
def F (f g : ℝ → ℝ) (x : ℝ) : ℝ := 3 * f x + 5 * g x + 2

-- Theorem statement
theorem odd_function_property (f g : ℝ → ℝ) (a : ℝ) 
  (hf : OddFunction f) (hg : OddFunction g) (hFa : F f g a = 3) : 
  F f g (-a) = 1 := by
  sorry

end odd_function_property_l180_18040


namespace cistern_fill_time_l180_18077

def fill_time (rate1 rate2 rate3 : ℚ) : ℚ :=
  1 / (rate1 + rate2 + rate3)

theorem cistern_fill_time :
  let rate1 : ℚ := 1 / 10
  let rate2 : ℚ := 1 / 12
  let rate3 : ℚ := -1 / 25
  fill_time rate1 rate2 rate3 = 300 / 43 := by
  sorry

end cistern_fill_time_l180_18077


namespace percentage_difference_theorem_l180_18044

theorem percentage_difference_theorem (x : ℝ) : 
  (0.35 * x = 0.50 * x - 24) → x = 160 := by
  sorry

end percentage_difference_theorem_l180_18044


namespace jerry_action_figures_l180_18056

theorem jerry_action_figures
  (complete_collection : ℕ)
  (cost_per_figure : ℕ)
  (total_cost_to_complete : ℕ)
  (h1 : complete_collection = 16)
  (h2 : cost_per_figure = 8)
  (h3 : total_cost_to_complete = 72) :
  complete_collection - (total_cost_to_complete / cost_per_figure) = 7 :=
by
  sorry

end jerry_action_figures_l180_18056


namespace hiking_distance_proof_l180_18008

def hiking_distance (total distance_car_to_stream distance_meadow_to_campsite : ℝ) : Prop :=
  ∃ distance_stream_to_meadow : ℝ,
    distance_stream_to_meadow = total - (distance_car_to_stream + distance_meadow_to_campsite) ∧
    distance_stream_to_meadow = 0.4

theorem hiking_distance_proof :
  hiking_distance 0.7 0.2 0.1 :=
by
  sorry

end hiking_distance_proof_l180_18008


namespace miles_travel_time_l180_18034

/-- Proves that if the distance between two cities is 57 miles and Miles takes 40 hours
    to complete 4 round trips, then Miles takes 10 hours for one round trip. -/
theorem miles_travel_time 
  (distance : ℝ) 
  (total_time : ℝ) 
  (num_round_trips : ℕ) 
  (h1 : distance = 57) 
  (h2 : total_time = 40) 
  (h3 : num_round_trips = 4) : 
  (total_time / num_round_trips) = 10 := by
  sorry


end miles_travel_time_l180_18034


namespace intersection_of_A_and_B_l180_18058

-- Define the universal set U
def U : Set Char := {'a', 'b', 'c', 'd', 'e'}

-- Define set A
def A : Set Char := {'a', 'b', 'c', 'd'}

-- Define set B
def B : Set Char := {'d', 'e'}

-- Theorem statement
theorem intersection_of_A_and_B : A ∩ B = {'d'} := by
  sorry

end intersection_of_A_and_B_l180_18058


namespace proposition_equivalence_l180_18082

theorem proposition_equivalence (m : ℝ) :
  (∃ x : ℝ, 0 ≤ x ∧ x ≤ 3 ∧ x^2 - 2*x > m) ↔ m < 3 := by
  sorry

end proposition_equivalence_l180_18082


namespace angle_bisector_inequality_l180_18013

/-- Represents a triangle with side lengths and angle bisectors -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  aa_prime : ℝ → ℝ → ℝ
  bb_prime : ℝ → ℝ → ℝ

/-- Theorem: In a triangle ABC with angle bisectors AA' and BB', if a > b, then CA' > CB' and BA' > AB' -/
theorem angle_bisector_inequality (t : Triangle) (h : t.a > t.b) :
  (t.c * t.a) / (t.b + t.c) > (t.a * t.b) / (t.a + t.c) ∧
  (t.a * t.b) / (t.b + t.c) > (t.c * t.b) / (t.a + t.c) := by
  sorry


end angle_bisector_inequality_l180_18013


namespace must_divide_p_l180_18045

theorem must_divide_p (p q r s : ℕ+) 
  (h1 : Nat.gcd p q = 28)
  (h2 : Nat.gcd q r = 45)
  (h3 : Nat.gcd r s = 63)
  (h4 : 80 < Nat.gcd s p ∧ Nat.gcd s p < 120) :
  11 ∣ p := by
  sorry

end must_divide_p_l180_18045


namespace function_derivative_at_two_l180_18057

/-- Given a function f(x) = a*ln(x) + b/x where f(1) = -2 and f'(1) = 0, prove that f'(2) = -1/2 -/
theorem function_derivative_at_two 
  (f : ℝ → ℝ) 
  (a b : ℝ) 
  (h1 : ∀ x > 0, f x = a * Real.log x + b / x)
  (h2 : f 1 = -2)
  (h3 : deriv f 1 = 0) :
  deriv f 2 = -1/2 := by
sorry

end function_derivative_at_two_l180_18057


namespace equation_solution_l180_18050

theorem equation_solution :
  ∃ x : ℝ, (Real.sqrt (7 * x - 2) - Real.sqrt (3 * x - 1) = 2) ∧ (x = 0.515625) :=
by sorry

end equation_solution_l180_18050


namespace cube_minus_reciprocal_cube_l180_18076

theorem cube_minus_reciprocal_cube (x : ℝ) (h : x - 1/x = 5) : x^3 - 1/x^3 = 140 := by
  sorry

end cube_minus_reciprocal_cube_l180_18076


namespace nancy_washed_19_shirts_l180_18022

/-- The number of shirts Nancy had to wash -/
def num_shirts (machine_capacity : ℕ) (num_loads : ℕ) (num_sweaters : ℕ) : ℕ :=
  machine_capacity * num_loads - num_sweaters

/-- Proof that Nancy washed 19 shirts -/
theorem nancy_washed_19_shirts :
  num_shirts 9 3 8 = 19 := by
  sorry

end nancy_washed_19_shirts_l180_18022


namespace binary_to_decimal_110101_l180_18084

theorem binary_to_decimal_110101 :
  (1 * 2^5 + 1 * 2^4 + 0 * 2^3 + 1 * 2^2 + 0 * 2^1 + 1 * 2^0 : ℕ) = 53 := by
  sorry

end binary_to_decimal_110101_l180_18084


namespace min_chord_length_implies_m_l180_18088

/-- Given a circle C: x^2 + y^2 = 4 and a line l: y = kx + m, 
    prove that if the minimum chord length cut by l on C is 2, then m = ±√3 -/
theorem min_chord_length_implies_m (k : ℝ) :
  (∀ x y, x^2 + y^2 = 4 → ∃ m, y = k*x + m) →
  (∃ m, ∀ x y, x^2 + y^2 = 4 ∧ y = k*x + m → 
    ∀ x1 y1 x2 y2, x1^2 + y1^2 = 4 ∧ y1 = k*x1 + m ∧ 
                   x2^2 + y2^2 = 4 ∧ y2 = k*x2 + m →
    (x1 - x2)^2 + (y1 - y2)^2 ≥ 4) →
  ∃ m, m = Real.sqrt 3 ∨ m = -Real.sqrt 3 := by
sorry

end min_chord_length_implies_m_l180_18088


namespace couscous_dishes_proof_l180_18007

/-- Calculates the number of dishes that can be made from couscous shipments -/
def couscous_dishes (shipment1 shipment2 shipment3 pounds_per_dish : ℕ) : ℕ :=
  (shipment1 + shipment2 + shipment3) / pounds_per_dish

/-- Proves that given the specified shipments and dish requirement, 13 dishes can be made -/
theorem couscous_dishes_proof :
  couscous_dishes 7 13 45 5 = 13 := by
  sorry

end couscous_dishes_proof_l180_18007


namespace square_of_number_doubled_exceeds_fifth_l180_18019

theorem square_of_number_doubled_exceeds_fifth : ∃ x : ℝ, 2 * x = (1/5) * x + 9 ∧ x^2 = 25 := by
  sorry

end square_of_number_doubled_exceeds_fifth_l180_18019


namespace S_description_l180_18002

-- Define the set S
def S : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | 
    let x := p.1
    let y := p.2
    ((5 = x + 3 ∧ (y - 6 ≤ 5 ∨ y - 6 = 5 / 2)) ∨
     (5 = y - 6 ∧ (x + 3 ≤ 5 ∨ x + 3 = 5 / 2)) ∨
     (x + 3 = y - 6 ∧ 5 = (x + 3) / 2))}

-- Define what it means to be parts of a right triangle
def isPartsOfRightTriangle (S : Set (ℝ × ℝ)) : Prop :=
  ∃ a b c : ℝ × ℝ,
    a ∈ S ∧ b ∈ S ∧ c ∈ S ∧
    a.1 = b.1 ∧ b.2 = c.2 ∧
    (c.1 - a.1) * (b.2 - a.2) = 0

-- Define what it means to have a separate point
def hasSeparatePoint (S : Set (ℝ × ℝ)) : Prop :=
  ∃ p : ℝ × ℝ, p ∈ S ∧ ∀ q ∈ S, q ≠ p → ‖p - q‖ > 0

-- Theorem statement
theorem S_description :
  isPartsOfRightTriangle S ∧ hasSeparatePoint S :=
sorry

end S_description_l180_18002


namespace sum_three_consecutive_divisible_by_three_l180_18092

theorem sum_three_consecutive_divisible_by_three (n : ℕ) :
  ∃ k : ℕ, n + (n + 1) + (n + 2) = 3 * k := by
  sorry

end sum_three_consecutive_divisible_by_three_l180_18092


namespace vertical_bisecting_line_of_circles_l180_18064

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 2*x + 6*y + 2 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 + 4*x - 2*y - 4 = 0

-- Define the vertical bisecting line
def bisecting_line (x y : ℝ) : Prop := 4*x + 3*y + 5 = 0

-- Theorem statement
theorem vertical_bisecting_line_of_circles :
  ∀ (x y : ℝ), (circle1 x y ∧ circle2 x y) → bisecting_line x y :=
by sorry

end vertical_bisecting_line_of_circles_l180_18064


namespace percentage_increase_l180_18096

theorem percentage_increase (initial : ℝ) (final : ℝ) : 
  initial = 100 → final = 150 → (final - initial) / initial * 100 = 50 := by
  sorry

end percentage_increase_l180_18096


namespace complex_root_sum_square_l180_18060

theorem complex_root_sum_square (p q : ℝ) : 
  (6 * (p + q * I)^3 + 5 * (p + q * I)^2 - (p + q * I) + 14 = 0) →
  (6 * (p - q * I)^3 + 5 * (p - q * I)^2 - (p - q * I) + 14 = 0) →
  p + q^2 = 21/4 := by
sorry

end complex_root_sum_square_l180_18060


namespace integer_triple_sum_equation_l180_18061

theorem integer_triple_sum_equation : ∃ (x y z : ℕ),
  1000 < x ∧ x < y ∧ y < z ∧ z < 2000 ∧
  (1 : ℚ) / 2 + 1 / 3 + 1 / 7 + 1 / x + 1 / y + 1 / z + 1 / 45 = 1 ∧
  x = 1806 ∧ y = 1892 ∧ z = 1980 := by
  sorry

end integer_triple_sum_equation_l180_18061


namespace sandwich_combinations_l180_18006

def num_ingredients : ℕ := 7

theorem sandwich_combinations :
  (Nat.choose num_ingredients 3 = 35) ∧ (Nat.choose num_ingredients 4 = 35) := by
  sorry

end sandwich_combinations_l180_18006


namespace sum_100_to_120_l180_18005

def sum_inclusive_range (a b : ℕ) : ℕ := (b - a + 1) * (a + b) / 2

theorem sum_100_to_120 : sum_inclusive_range 100 120 = 2310 := by sorry

end sum_100_to_120_l180_18005


namespace add_twice_eq_thrice_l180_18079

theorem add_twice_eq_thrice (a : ℝ) : a + 2 * a = 3 * a := by
  sorry

end add_twice_eq_thrice_l180_18079


namespace solution_equation_l180_18078

theorem solution_equation : ∃ x : ℝ, 0.4 * x + (0.3 * 0.2) = 0.26 ∧ x = 0.5 := by
  sorry

end solution_equation_l180_18078


namespace smallest_positive_angle_neg_1050_l180_18065

/-- The smallest positive angle (in degrees) with the same terminal side as a given angle -/
def smallestPositiveEquivalentAngle (angle : Int) : Int :=
  (angle % 360 + 360) % 360

/-- Theorem: The smallest positive angle with the same terminal side as -1050° is 30° -/
theorem smallest_positive_angle_neg_1050 :
  smallestPositiveEquivalentAngle (-1050) = 30 := by
  sorry

end smallest_positive_angle_neg_1050_l180_18065


namespace isosceles_triangle_base_l180_18052

/-- An isosceles triangle with perimeter 13 and one side 3 has a base of 3 -/
theorem isosceles_triangle_base (a b c : ℝ) : 
  a + b + c = 13 →  -- perimeter is 13
  a = b →           -- isosceles condition
  (a = 3 ∨ b = 3 ∨ c = 3) →  -- one side is 3
  c = 3 :=          -- base is 3
by sorry

end isosceles_triangle_base_l180_18052


namespace one_of_each_color_probability_l180_18039

def total_marbles : ℕ := 6
def red_marbles : ℕ := 2
def blue_marbles : ℕ := 2
def green_marbles : ℕ := 2
def selected_marbles : ℕ := 3

theorem one_of_each_color_probability :
  (red_marbles * blue_marbles * green_marbles) / Nat.choose total_marbles selected_marbles = 2 / 5 := by
  sorry

end one_of_each_color_probability_l180_18039


namespace unique_solution_ceiling_equation_l180_18004

theorem unique_solution_ceiling_equation :
  ∃! x : ℝ, x > 0 ∧ x + ⌈x⌉ = 21.3 :=
by sorry

end unique_solution_ceiling_equation_l180_18004


namespace product_seven_reciprocal_squares_sum_l180_18086

theorem product_seven_reciprocal_squares_sum (a b : ℕ) (h : a * b = 7) :
  (1 : ℚ) / (a ^ 2) + (1 : ℚ) / (b ^ 2) = 50 / 49 := by
  sorry

end product_seven_reciprocal_squares_sum_l180_18086


namespace no_equal_functions_l180_18083

def f₁ (x : ℤ) : ℤ := x * (x - 2007)
def f₂ (x : ℤ) : ℤ := (x - 1) * (x - 2006)
def f₁₀₀₄ (x : ℤ) : ℤ := (x - 1003) * (x - 1004)

theorem no_equal_functions :
  ∀ x : ℤ, 0 ≤ x ∧ x ≤ 2007 →
    (f₁ x ≠ f₂ x) ∧ (f₁ x ≠ f₁₀₀₄ x) ∧ (f₂ x ≠ f₁₀₀₄ x) := by
  sorry

end no_equal_functions_l180_18083


namespace age_difference_l180_18087

/-- Given two people p and q, prove that p was half of q's age 6 years ago,
    given their current age ratio and sum. -/
theorem age_difference (p q : ℕ) : 
  (p : ℚ) / q = 3 / 4 →  -- Current age ratio
  p + q = 21 →           -- Sum of current ages
  ∃ (y : ℕ), p - y = (q - y) / 2 ∧ y = 6 := by
sorry

end age_difference_l180_18087


namespace min_value_theorem_l180_18069

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) : 
  (4 * x) / (x + 3 * y) + (3 * y) / x ≥ 3 := by
  sorry

end min_value_theorem_l180_18069


namespace six_pencil_sharpeners_affordable_remaining_money_buys_four_pencil_cases_remaining_money_after_ten_pencil_cases_l180_18047

-- Define the given prices and budget
def total_budget : ℕ := 100
def pencil_sharpener_price : ℕ := 15
def notebooks_6_price : ℕ := 24
def pencil_case_price : ℕ := 5
def colored_pencils_2boxes_price : ℕ := 16

-- Theorem 1: 6 pencil sharpeners cost less than or equal to 100 yuan
theorem six_pencil_sharpeners_affordable :
  6 * pencil_sharpener_price ≤ total_budget :=
sorry

-- Theorem 2: After buying 20 notebooks, the remaining money can buy exactly 4 pencil cases
theorem remaining_money_buys_four_pencil_cases :
  (total_budget - (20 * (notebooks_6_price / 6))) / pencil_case_price = 4 :=
sorry

-- Theorem 3: After buying 10 pencil cases, the remaining money is 50 yuan
theorem remaining_money_after_ten_pencil_cases :
  total_budget - (10 * pencil_case_price) = 50 :=
sorry

end six_pencil_sharpeners_affordable_remaining_money_buys_four_pencil_cases_remaining_money_after_ten_pencil_cases_l180_18047


namespace x_four_coefficient_range_l180_18080

-- Define the binomial coefficient function
def binomial (n k : ℕ) : ℕ := sorry

-- Define the coefficient of x^4 in the expansion of (1+x+mx^2)^10
def coefficient (m : ℝ) : ℝ := binomial 10 4 + binomial 10 2 * m^2 + binomial 10 1 * binomial 9 2 * m

-- State the theorem
theorem x_four_coefficient_range :
  {m : ℝ | coefficient m > -330} = {m : ℝ | m < -6 ∨ m > -2} := by sorry

end x_four_coefficient_range_l180_18080


namespace optimal_box_height_l180_18091

/-- Represents the dimensions of an open-top rectangular box with a square base -/
structure BoxDimensions where
  side : ℝ
  height : ℝ

/-- The volume of the box -/
def volume (b : BoxDimensions) : ℝ := b.side^2 * b.height

/-- The surface area of the box -/
def surfaceArea (b : BoxDimensions) : ℝ := b.side^2 + 4 * b.side * b.height

/-- The constraint that the volume must be 4 -/
def volumeConstraint (b : BoxDimensions) : Prop := volume b = 4

theorem optimal_box_height :
  ∃ (b : BoxDimensions), volumeConstraint b ∧
    (∀ (b' : BoxDimensions), volumeConstraint b' → surfaceArea b ≤ surfaceArea b') ∧
    b.height = 1 := by
  sorry

end optimal_box_height_l180_18091


namespace fourth_power_difference_l180_18010

theorem fourth_power_difference (x : ℝ) (h : x - 1/x = 5) : x^4 - 1/x^4 = 727 := by
  sorry

end fourth_power_difference_l180_18010


namespace cubic_polynomial_root_l180_18099

theorem cubic_polynomial_root (x : ℝ) : x = Real.rpow 4 (1/3) + 2 → x^3 - 6*x^2 + 12*x - 16 = 0 := by
  sorry

end cubic_polynomial_root_l180_18099


namespace quadratic_prime_roots_fraction_sum_l180_18043

/-- Two prime numbers that are roots of a quadratic equation --/
def QuadraticPrimeRoots (a b : ℕ) : Prop :=
  Prime a ∧ Prime b ∧ ∃ t : ℤ, a^2 - 21*a + t = 0 ∧ b^2 - 21*b + t = 0

/-- The main theorem --/
theorem quadratic_prime_roots_fraction_sum
  (a b : ℕ) (h : QuadraticPrimeRoots a b) :
  (b : ℚ) / a + (a : ℚ) / b = 365 / 38 := by
  sorry

end quadratic_prime_roots_fraction_sum_l180_18043


namespace exists_n_for_m_l180_18015

-- Define the function f(n) as the sum of n and its digits
def f (n : ℕ) : ℕ :=
  n + (Nat.digits 10 n).sum

-- Theorem statement
theorem exists_n_for_m (m : ℕ) :
  m > 0 → ∃ n : ℕ, f n = m ∨ f n = m + 1 := by
  sorry

end exists_n_for_m_l180_18015


namespace range_of_m_l180_18018

theorem range_of_m (α : ℝ) (m : ℝ) 
  (h1 : α ∈ Set.Ioo 0 (Real.pi / 2))
  (h2 : Real.sqrt 3 * Real.sin α + Real.cos α = m) :
  m ∈ Set.Ioo 1 2 ∪ {2} :=
by sorry

end range_of_m_l180_18018


namespace probability_non_defective_pencils_l180_18037

/-- The probability of selecting 3 non-defective pencils from a box of 11 pencils with 2 defective ones -/
theorem probability_non_defective_pencils (total_pencils : Nat) (defective_pencils : Nat) (selected_pencils : Nat) :
  total_pencils = 11 →
  defective_pencils = 2 →
  selected_pencils = 3 →
  (Nat.choose (total_pencils - defective_pencils) selected_pencils : ℚ) / 
  (Nat.choose total_pencils selected_pencils : ℚ) = 28 / 55 := by
  sorry

end probability_non_defective_pencils_l180_18037


namespace fan_ratio_proof_l180_18075

/-- Represents the number of fans for each team -/
structure FanCount where
  yankees : ℕ
  mets : ℕ
  red_sox : ℕ

/-- The ratio of Mets fans to Red Sox fans is 4:5 -/
def mets_to_red_sox_ratio (fc : FanCount) : Prop :=
  4 * fc.red_sox = 5 * fc.mets

/-- The total number of fans is 330 -/
def total_fans (fc : FanCount) : Prop :=
  fc.yankees + fc.mets + fc.red_sox = 330

/-- There are 88 Mets fans -/
def mets_fan_count (fc : FanCount) : Prop :=
  fc.mets = 88

/-- The ratio of Yankees fans to Mets fans is 3:2 -/
def yankees_to_mets_ratio (fc : FanCount) : Prop :=
  3 * fc.mets = 2 * fc.yankees

theorem fan_ratio_proof (fc : FanCount)
    (h1 : mets_to_red_sox_ratio fc)
    (h2 : total_fans fc)
    (h3 : mets_fan_count fc) :
  yankees_to_mets_ratio fc := by
  sorry

end fan_ratio_proof_l180_18075


namespace choir_arrangement_theorem_l180_18073

theorem choir_arrangement_theorem :
  ∃ (m : ℕ), 
    (∃ (k : ℕ), m = k^2 + 6) ∧ 
    (∃ (n : ℕ), m = n * (n + 6)) ∧
    (∀ (x : ℕ), 
      ((∃ (y : ℕ), x = y^2 + 6) ∧ 
       (∃ (z : ℕ), x = z * (z + 6))) → 
      x ≤ m) ∧
    m = 294 := by
  sorry

end choir_arrangement_theorem_l180_18073


namespace quadratic_vertex_and_symmetry_l180_18035

/-- Given a quadratic function f(x) = -x^2 - 4x + 2, 
    its vertex is at (-2, 6) and its axis of symmetry is x = -2 -/
theorem quadratic_vertex_and_symmetry :
  let f : ℝ → ℝ := λ x ↦ -x^2 - 4*x + 2
  ∃ (vertex : ℝ × ℝ) (axis : ℝ),
    vertex = (-2, 6) ∧
    axis = -2 ∧
    (∀ x, f x = f (2 * axis - x)) ∧
    (∀ x, f x ≤ f axis) :=
by sorry

end quadratic_vertex_and_symmetry_l180_18035


namespace zlatoust_miass_distance_l180_18027

theorem zlatoust_miass_distance :
  ∀ (g m k : ℝ),  -- speeds of GAZ, MAZ, and KamAZ
  (∀ x : ℝ, 
    (x + 18) / k = (x - 18) / m ∧
    (x + 25) / k = (x - 25) / g ∧
    (x + 8) / m = (x - 8) / g) →
  ∃ x : ℝ, x = 60 ∧ 
    (x + 18) / k = (x - 18) / m ∧
    (x + 25) / k = (x - 25) / g ∧
    (x + 8) / m = (x - 8) / g :=
by sorry

end zlatoust_miass_distance_l180_18027


namespace ballet_arrangement_l180_18025

/-- The number of boys participating in the ballet -/
def num_boys : ℕ := 5

/-- The distance between each girl and her two assigned boys (in meters) -/
def distance : ℕ := 5

/-- The maximum number of girls that can participate in the ballet -/
def max_girls : ℕ := 20

/-- Theorem stating the maximum number of girls that can participate in the ballet -/
theorem ballet_arrangement (n : ℕ) (d : ℕ) (m : ℕ) 
  (h1 : n = num_boys) 
  (h2 : d = distance) 
  (h3 : m = max_girls) :
  m = n * (n - 1) := by
  sorry

end ballet_arrangement_l180_18025


namespace largest_n_divisible_by_three_answer_is_199999_l180_18042

theorem largest_n_divisible_by_three (n : ℕ) : 
  n < 200000 → 
  (3 ∣ (10 * (n - 3)^5 - 2 * n^2 + 20 * n - 36)) → 
  n ≤ 199999 :=
by sorry

theorem answer_is_199999 : 
  199999 < 200000 ∧ 
  (3 ∣ (10 * (199999 - 3)^5 - 2 * 199999^2 + 20 * 199999 - 36)) ∧
  ∀ m : ℕ, m > 199999 → m < 200000 → 
    ¬(3 ∣ (10 * (m - 3)^5 - 2 * m^2 + 20 * m - 36)) :=
by sorry

end largest_n_divisible_by_three_answer_is_199999_l180_18042


namespace rectangle_width_l180_18014

/-- Given a rectangle where the length is 2 cm shorter than the width and the perimeter is 16 cm, 
    the width of the rectangle is 5 cm. -/
theorem rectangle_width (w : ℝ) (h1 : 2 * w + 2 * (w - 2) = 16) : w = 5 := by
  sorry

end rectangle_width_l180_18014


namespace candy_box_problem_l180_18038

/-- Given the number of chocolate boxes, caramel boxes, and total pieces of candy,
    calculate the number of pieces in each box. -/
def pieces_per_box (chocolate_boxes caramel_boxes total_pieces : ℕ) : ℕ :=
  total_pieces / (chocolate_boxes + caramel_boxes)

/-- Theorem stating that given 7 boxes of chocolate candy, 3 boxes of caramel candy,
    and a total of 80 pieces, there are 8 pieces in each box. -/
theorem candy_box_problem :
  pieces_per_box 7 3 80 = 8 := by
  sorry

end candy_box_problem_l180_18038


namespace pyramid_side_length_l180_18041

/-- Represents a pyramid with a rectangular base ABCD and vertex E above A -/
structure Pyramid where
  -- Base side lengths
  AB : ℝ
  BC : ℝ
  -- Angles
  BCE : ℝ
  ADE : ℝ

/-- Theorem: In a pyramid with given conditions, BC = 2√2 -/
theorem pyramid_side_length (p : Pyramid)
  (h_AB : p.AB = 4)
  (h_BCE : p.BCE = Real.pi / 3)  -- 60 degrees in radians
  (h_ADE : p.ADE = Real.pi / 4)  -- 45 degrees in radians
  : p.BC = 2 * Real.sqrt 2 := by
  sorry


end pyramid_side_length_l180_18041


namespace line_for_equal_diagonals_l180_18067

-- Define the circle C
def C (x y : ℝ) : Prop := (x - 1)^2 + (y + 2)^2 = 9

-- Define the line l passing through (-1, 0)
def l (k : ℝ) (x y : ℝ) : Prop := y = k * (x + 1)

-- Define the intersection points A and B
def intersectionPoints (k : ℝ) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ : ℝ), C x₁ y₁ ∧ C x₂ y₂ ∧ l k x₁ y₁ ∧ l k x₂ y₂

-- Define vector OS as the sum of OA and OB
def vectorOS (k : ℝ) (x y : ℝ) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ : ℝ), C x₁ y₁ ∧ C x₂ y₂ ∧ l k x₁ y₁ ∧ l k x₂ y₂ ∧
    x = x₁ + x₂ ∧ y = y₁ + y₂

-- Define the condition for equal diagonals in quadrilateral OASB
def equalDiagonals (k : ℝ) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ : ℝ), C x₁ y₁ ∧ C x₂ y₂ ∧ l k x₁ y₁ ∧ l k x₂ y₂ ∧
    x₁^2 + y₁^2 = x₂^2 + y₂^2

-- Theorem statement
theorem line_for_equal_diagonals :
  ∃! k, intersectionPoints k ∧ equalDiagonals k ∧ k = 1 :=
sorry

end line_for_equal_diagonals_l180_18067


namespace unique_non_divisible_by_3_l180_18016

def is_divisible_by_3 (n : ℕ) : Prop := ∃ k : ℕ, n = 3 * k

def sum_of_digits (n : ℕ) : ℕ :=
  (n / 1000) + ((n / 100) % 10) + ((n / 10) % 10) + (n % 10)

def units_digit (n : ℕ) : ℕ := n % 10

def tens_digit (n : ℕ) : ℕ := (n / 10) % 10

theorem unique_non_divisible_by_3 :
  let numbers : List ℕ := [3543, 3555, 3567, 3573, 3581]
  ∀ n ∈ numbers, ¬(is_divisible_by_3 n) → n = 3581 ∧ units_digit n + tens_digit n = 9 :=
by sorry

end unique_non_divisible_by_3_l180_18016


namespace sidney_monday_jj_l180_18033

/-- The number of jumping jacks Sidney did on Monday -/
def monday_jj : ℕ := sorry

/-- The number of jumping jacks Sidney did on Tuesday -/
def tuesday_jj : ℕ := 36

/-- The number of jumping jacks Sidney did on Wednesday -/
def wednesday_jj : ℕ := 40

/-- The number of jumping jacks Sidney did on Thursday -/
def thursday_jj : ℕ := 50

/-- The total number of jumping jacks Brooke did -/
def brooke_total : ℕ := 438

/-- Theorem stating that Sidney did 20 jumping jacks on Monday -/
theorem sidney_monday_jj : monday_jj = 20 := by
  sorry

end sidney_monday_jj_l180_18033


namespace bowling_record_difference_l180_18071

theorem bowling_record_difference (old_record : ℕ) (players : ℕ) (rounds : ℕ) (current_score : ℕ) : 
  old_record = 287 →
  players = 4 →
  rounds = 10 →
  current_score = 10440 →
  old_record - (((old_record * players * rounds) - current_score) / players) = 27 := by
sorry

end bowling_record_difference_l180_18071


namespace common_difference_is_three_l180_18090

def arithmetic_sequence (a : ℕ → ℝ) := ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)

theorem common_difference_is_three
  (a : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_sum1 : a 2 + a 3 = 9)
  (h_sum2 : a 4 + a 5 = 21) :
  ∃ d, d = 3 ∧ ∀ n, a (n + 1) - a n = d :=
sorry

end common_difference_is_three_l180_18090


namespace stratified_sample_size_l180_18070

/-- Represents the ratio of students in grades 10, 11, and 12 -/
def student_ratio : Fin 3 → ℕ
| 0 => 2  -- Grade 10
| 1 => 3  -- Grade 11
| 2 => 5  -- Grade 12
| _ => 0  -- Unreachable case

/-- The total parts in the ratio -/
def total_ratio : ℕ := (student_ratio 0) + (student_ratio 1) + (student_ratio 2)

/-- The number of grade 12 students in the sample -/
def grade_12_sample : ℕ := 150

/-- The total sample size -/
def total_sample : ℕ := 300

theorem stratified_sample_size :
  (student_ratio 2 : ℚ) / total_ratio = grade_12_sample / total_sample :=
sorry

end stratified_sample_size_l180_18070


namespace probability_of_three_pointing_l180_18029

/-- The number of people in the room -/
def n : ℕ := 5

/-- The probability of one person pointing at two specific others -/
def p : ℚ := 1 / 6

/-- The number of ways to choose 2 people out of n -/
def choose_two (n : ℕ) : ℕ := n * (n - 1) / 2

/-- The probability of a group of three all pointing at each other -/
def prob_three_pointing : ℚ := p^3

/-- The main theorem: probability of having a group of three all pointing at each other -/
theorem probability_of_three_pointing :
  (choose_two n : ℚ) * prob_three_pointing = 5 / 108 := by
  sorry

end probability_of_three_pointing_l180_18029


namespace percentage_runs_by_running_approx_l180_18011

-- Define the given conditions
def total_runs : ℕ := 134
def boundaries : ℕ := 12
def sixes : ℕ := 2

-- Define the runs per boundary and six
def runs_per_boundary : ℕ := 4
def runs_per_six : ℕ := 6

-- Calculate runs from boundaries and sixes
def runs_from_boundaries_and_sixes : ℕ := boundaries * runs_per_boundary + sixes * runs_per_six

-- Calculate runs made by running between wickets
def runs_by_running : ℕ := total_runs - runs_from_boundaries_and_sixes

-- Define the percentage of runs made by running
def percentage_runs_by_running : ℚ := (runs_by_running : ℚ) / (total_runs : ℚ) * 100

-- Theorem to prove
theorem percentage_runs_by_running_approx :
  abs (percentage_runs_by_running - 55.22) < 0.01 := by sorry

end percentage_runs_by_running_approx_l180_18011


namespace jack_afternoon_emails_l180_18017

/-- Represents the number of emails Jack received at different times of the day. -/
structure EmailCount where
  morning : Nat
  total : Nat

/-- Calculates the number of emails Jack received in the afternoon. -/
def afternoon_emails (e : EmailCount) : Nat :=
  e.total - e.morning

/-- Theorem: Jack received 1 email in the afternoon. -/
theorem jack_afternoon_emails :
  let e : EmailCount := { morning := 4, total := 5 }
  afternoon_emails e = 1 := by
  sorry

end jack_afternoon_emails_l180_18017


namespace no_two_common_tangents_l180_18098

/-- Two circles in a plane with radii r and 2r -/
structure TwoCircles (r : ℝ) where
  center1 : ℝ × ℝ
  center2 : ℝ × ℝ

/-- The number of common tangents between two circles -/
def numCommonTangents (c : TwoCircles r) : ℕ := sorry

/-- Theorem: It's impossible for two circles with radii r and 2r to have exactly 2 common tangents -/
theorem no_two_common_tangents (r : ℝ) (hr : r > 0) :
  ∀ c : TwoCircles r, numCommonTangents c ≠ 2 := by sorry

end no_two_common_tangents_l180_18098


namespace loan_principal_calculation_l180_18012

theorem loan_principal_calculation (principal : ℝ) : 
  (principal * 0.08 * 10 = principal - 1540) → principal = 7700 := by
  sorry

end loan_principal_calculation_l180_18012


namespace f_is_even_and_increasing_l180_18055

def f (x : ℝ) : ℝ := |x| + 1

theorem f_is_even_and_increasing :
  (∀ x : ℝ, f x = f (-x)) ∧
  (∀ x y : ℝ, 0 < x → x < y → f x < f y) :=
by sorry

end f_is_even_and_increasing_l180_18055


namespace round_windmill_iff_on_diagonal_l180_18020

/-- A square in a 2D plane. -/
structure Square :=
  (A B C D : ℝ × ℝ)

/-- A point in a 2D plane. -/
def Point := ℝ × ℝ

/-- A line in a 2D plane. -/
structure Line :=
  (p1 p2 : Point)

/-- A windmill configuration. -/
structure Windmill :=
  (center : Point)
  (l1 l2 : Line)

/-- Checks if a point is inside a square. -/
def isInside (s : Square) (p : Point) : Prop := sorry

/-- Checks if two lines are perpendicular. -/
def arePerpendicular (l1 l2 : Line) : Prop := sorry

/-- Checks if a quadrilateral is cyclic. -/
def isCyclic (p1 p2 p3 p4 : Point) : Prop := sorry

/-- Checks if a point lies on the diagonal of a square. -/
def isOnDiagonal (s : Square) (p : Point) : Prop := sorry

/-- Theorem: A point P inside a square ABCD produces a round windmill for all
    possible configurations if and only if P lies on the diagonals of the square. -/
theorem round_windmill_iff_on_diagonal (s : Square) (p : Point) :
  isInside s p →
  (∀ (w : Windmill), w.center = p →
    arePerpendicular w.l1 w.l2 →
    (∃ W X Y Z, isCyclic W X Y Z)) ↔
  isOnDiagonal s p :=
sorry

end round_windmill_iff_on_diagonal_l180_18020


namespace triangle_perimeter_l180_18046

theorem triangle_perimeter (m n : ℝ) : 
  let side1 := 3 * m
  let side2 := side1 - (m - n)
  let side3 := side2 + 2 * n
  side1 + side2 + side3 = 7 * m + 4 * n :=
by
  sorry

end triangle_perimeter_l180_18046


namespace tangent_line_at_negative_one_l180_18051

theorem tangent_line_at_negative_one (x y : ℝ) :
  y = 2*x - x^3 → 
  let tangent_point := (-1, 2*(-1) - (-1)^3)
  let tangent_slope := -3*(-1)^2 + 2
  (x + y + 2 = 0) = 
    ((y - tangent_point.2) = tangent_slope * (x - tangent_point.1)) := by
  sorry

end tangent_line_at_negative_one_l180_18051


namespace forgotten_angles_sum_l180_18059

theorem forgotten_angles_sum (n : ℕ) (measured_sum : ℝ) : 
  n > 2 → 
  measured_sum = 2873 → 
  ∃ (missing_sum : ℝ), 
    missing_sum = ((n - 2) * 180 : ℝ) - measured_sum ∧ 
    missing_sum = 7 := by
  sorry

end forgotten_angles_sum_l180_18059


namespace least_x_value_l180_18085

theorem least_x_value (x p : ℕ) (h1 : x > 0) (h2 : Nat.Prime p) 
  (h3 : ∃ q : ℕ, Nat.Prime q ∧ q ≠ 2 ∧ x = 12 * p * q) : x ≥ 72 := by
  sorry

end least_x_value_l180_18085


namespace quadratic_roots_l180_18030

theorem quadratic_roots (a : ℝ) : 
  (3 : ℝ) ^ 2 - 2 * 3 + a = 0 → 
  (-1 : ℝ) ^ 2 - 2 * (-1) + a = 0 := by
sorry

end quadratic_roots_l180_18030


namespace remainder_seven_divisors_l180_18000

theorem remainder_seven_divisors (n : ℕ) : 
  (∃ (divisors : Finset ℕ), 
    divisors = {d : ℕ | d > 7 ∧ 54 % d = 0} ∧ 
    Finset.card divisors = 4) := by
  sorry

end remainder_seven_divisors_l180_18000


namespace perpendicular_lines_a_value_l180_18024

/-- Two lines in the form ax + by + c = 0 and dx + ey + f = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Perpendicular property for two lines -/
def perpendicular (l1 l2 : Line) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

theorem perpendicular_lines_a_value :
  ∀ (a : ℝ),
  let l1 : Line := ⟨a, 2, 1⟩
  let l2 : Line := ⟨1, 3, -2⟩
  perpendicular l1 l2 → a = -6 := by
sorry

end perpendicular_lines_a_value_l180_18024


namespace sin_120_degrees_l180_18095

theorem sin_120_degrees : Real.sin (2 * Real.pi / 3) = Real.sqrt 3 / 2 := by
  sorry

end sin_120_degrees_l180_18095


namespace octagon_diagonals_eq_twenty_l180_18053

/-- The number of diagonals in an octagon -/
def octagon_diagonals : ℕ :=
  let n := 8  -- number of vertices in an octagon
  let sides := 8  -- number of sides in an octagon
  (n * (n - 1)) / 2 - sides

/-- Theorem: The number of diagonals in an octagon is 20 -/
theorem octagon_diagonals_eq_twenty : octagon_diagonals = 20 := by
  sorry

end octagon_diagonals_eq_twenty_l180_18053


namespace min_value_quadratic_function_l180_18009

theorem min_value_quadratic_function :
  ∃ (min : ℝ), min = -11.25 ∧
  ∀ (x y : ℝ), 2 * x^2 + 4 * x * y + 5 * y^2 - 8 * x - 6 * y ≥ min :=
by sorry

end min_value_quadratic_function_l180_18009


namespace crayons_erasers_difference_l180_18023

/-- Given the initial numbers of crayons and erasers, and the final number of crayons,
    prove that the difference between the number of crayons left and the number of erasers is 66. -/
theorem crayons_erasers_difference 
  (initial_crayons : ℕ) 
  (initial_erasers : ℕ) 
  (final_crayons : ℕ) 
  (h1 : initial_crayons = 617) 
  (h2 : initial_erasers = 457) 
  (h3 : final_crayons = 523) : 
  final_crayons - initial_erasers = 66 := by
  sorry

#check crayons_erasers_difference

end crayons_erasers_difference_l180_18023


namespace quadratic_root_relation_l180_18089

theorem quadratic_root_relation (p : ℚ) : 
  (∃ x y : ℚ, x = 3 * y ∧ 
   x^2 - (3*p - 2)*x + p^2 - 1 = 0 ∧ 
   y^2 - (3*p - 2)*y + p^2 - 1 = 0) ↔ 
  (p = 2 ∨ p = 14/11) := by
sorry

end quadratic_root_relation_l180_18089


namespace james_birthday_stickers_l180_18036

/-- The number of stickers James got for his birthday -/
def birthday_stickers (initial : ℕ) (total : ℕ) : ℕ := total - initial

theorem james_birthday_stickers :
  birthday_stickers 39 61 = 22 := by
  sorry

end james_birthday_stickers_l180_18036


namespace no_prime_pairs_for_square_diff_l180_18066

theorem no_prime_pairs_for_square_diff (a b : ℕ) : 
  a ≤ 100 → b ≤ 100 → Prime a → Prime b → a^2 - b^2 ≠ 25 :=
by sorry

end no_prime_pairs_for_square_diff_l180_18066


namespace truck_license_combinations_l180_18028

/-- The number of possible letters for a truck license -/
def num_letters : ℕ := 3

/-- The number of digits in a truck license -/
def num_digits : ℕ := 6

/-- The number of possible digits (0-9) for each position -/
def digits_per_position : ℕ := 10

/-- The total number of possible truck license combinations -/
def total_combinations : ℕ := num_letters * (digits_per_position ^ num_digits)

theorem truck_license_combinations :
  total_combinations = 3000000 := by
  sorry

end truck_license_combinations_l180_18028


namespace license_plate_increase_l180_18048

/-- The number of possible letters in a license plate. -/
def num_letters : ℕ := 26

/-- The number of possible digits in a license plate. -/
def num_digits : ℕ := 10

/-- The number of letters in an old license plate. -/
def old_letters : ℕ := 3

/-- The number of digits in an old license plate. -/
def old_digits : ℕ := 2

/-- The number of letters in a new license plate. -/
def new_letters : ℕ := 2

/-- The number of digits in a new license plate. -/
def new_digits : ℕ := 4

/-- The theorem stating the increase in the number of possible license plates. -/
theorem license_plate_increase :
  (num_letters ^ new_letters * num_digits ^ new_digits) /
  (num_letters ^ old_letters * num_digits ^ old_digits) = 50 / 13 :=
by sorry

end license_plate_increase_l180_18048


namespace probability_x_plus_y_less_than_two_point_five_l180_18031

/-- A square in the 2D plane --/
structure Square where
  bottomLeft : ℝ × ℝ
  topRight : ℝ × ℝ

/-- A point is inside a square --/
def isInside (p : ℝ × ℝ) (s : Square) : Prop :=
  s.bottomLeft.1 ≤ p.1 ∧ p.1 ≤ s.topRight.1 ∧
  s.bottomLeft.2 ≤ p.2 ∧ p.2 ≤ s.topRight.2

/-- The probability of an event for a uniformly distributed point in a square --/
def probability (s : Square) (event : ℝ × ℝ → Prop) : ℝ :=
  sorry

theorem probability_x_plus_y_less_than_two_point_five :
  let s : Square := { bottomLeft := (0, 0), topRight := (3, 3) }
  probability s (fun p => p.1 + p.2 < 2.5) = 125 / 360 := by
  sorry

end probability_x_plus_y_less_than_two_point_five_l180_18031


namespace arithmetic_sequence_difference_l180_18054

/-- The nth term of an arithmetic sequence -/
def arithmeticSequenceTerm (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ :=
  a₁ + (n - 1) * d

/-- The positive difference between two terms of an arithmetic sequence -/
def positiveDifference (a₁ : ℤ) (d : ℤ) (m n : ℕ) : ℕ :=
  (arithmeticSequenceTerm a₁ d m - arithmeticSequenceTerm a₁ d n).natAbs

theorem arithmetic_sequence_difference :
  positiveDifference (-8) 8 1020 1000 = 160 := by
  sorry

end arithmetic_sequence_difference_l180_18054


namespace right_triangle_sets_l180_18072

theorem right_triangle_sets : 
  (1^2 + Real.sqrt 2^2 = Real.sqrt 3^2) ∧ 
  (3^2 + 4^2 = 5^2) ∧ 
  (9^2 + 12^2 = 15^2) ∧ 
  (4^2 + 5^2 ≠ 6^2) := by
  sorry

end right_triangle_sets_l180_18072


namespace dollar_square_sum_l180_18081

/-- Custom operation ▩ for real numbers -/
def dollar (a b : ℝ) : ℝ := (a - b)^2

/-- Theorem stating that (x + y)² ▩ (y² + x²) = 4x²y² -/
theorem dollar_square_sum (x y : ℝ) : dollar ((x + y)^2) (y^2 + x^2) = 4 * x^2 * y^2 := by
  sorry

end dollar_square_sum_l180_18081


namespace expression_evaluation_l180_18063

theorem expression_evaluation : 2 + (0 * 2^2) = 2 := by
  sorry

end expression_evaluation_l180_18063


namespace max_xy_value_l180_18001

theorem max_xy_value (x y : ℝ) (h_pos_x : x > 0) (h_pos_y : y > 0) (h_sum : 2 * x + y = 1) :
  x * y ≤ 1 / 8 ∧ ∃ x y, x > 0 ∧ y > 0 ∧ 2 * x + y = 1 ∧ x * y = 1 / 8 := by
  sorry

end max_xy_value_l180_18001
