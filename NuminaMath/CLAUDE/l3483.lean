import Mathlib

namespace mans_rate_l3483_348387

/-- The man's rate in still water given his speeds with and against the stream -/
theorem mans_rate (speed_with_stream speed_against_stream : ℝ) 
  (h1 : speed_with_stream = 20)
  (h2 : speed_against_stream = 4) : 
  (speed_with_stream + speed_against_stream) / 2 = 12 := by
  sorry

end mans_rate_l3483_348387


namespace mandy_shirt_count_l3483_348386

/-- The number of black shirt packs Mandy bought -/
def black_packs : ℕ := 6

/-- The number of yellow shirt packs Mandy bought -/
def yellow_packs : ℕ := 8

/-- The number of shirts in each black shirt pack -/
def black_per_pack : ℕ := 7

/-- The number of shirts in each yellow shirt pack -/
def yellow_per_pack : ℕ := 4

/-- The total number of shirts Mandy bought -/
def total_shirts : ℕ := black_packs * black_per_pack + yellow_packs * yellow_per_pack

theorem mandy_shirt_count : total_shirts = 74 := by
  sorry

end mandy_shirt_count_l3483_348386


namespace expression_simplification_and_evaluation_l3483_348360

theorem expression_simplification_and_evaluation :
  let x : ℝ := 1 - 3 * Real.tan (π / 4)
  (1 / (3 - x) - (x^2 + 6*x + 9) / (x^2 + 3*x) / ((x^2 - 9) / x)) = 2 / 5 :=
by sorry

end expression_simplification_and_evaluation_l3483_348360


namespace parabola_chord_length_squared_l3483_348359

/-- Given a parabola y = 3x^2 + 4x + 2, with points C and D on the parabola,
    and the origin as the midpoint of CD, and the slope of the tangent at C is 10,
    prove that the square of the length of CD is 8. -/
theorem parabola_chord_length_squared (C D : ℝ × ℝ) : 
  (∃ (x y : ℝ), C = (x, y) ∧ D = (-x, -y)) →  -- Origin is midpoint of CD
  (C.2 = 3 * C.1^2 + 4 * C.1 + 2) →  -- C is on the parabola
  (D.2 = 3 * D.1^2 + 4 * D.1 + 2) →  -- D is on the parabola
  (6 * C.1 + 4 = 10) →  -- Slope of tangent at C is 10
  (C.1 - D.1)^2 + (C.2 - D.2)^2 = 8 := by sorry

end parabola_chord_length_squared_l3483_348359


namespace f_neither_even_nor_odd_l3483_348324

-- Define the function f(x) = x^2 on the domain -1 < x ≤ 1
def f (x : ℝ) : ℝ := x^2

-- Define the domain of the function
def domain (x : ℝ) : Prop := -1 < x ∧ x ≤ 1

-- Define what it means for a function to be even
def is_even (f : ℝ → ℝ) : Prop :=
  ∀ x, domain x → domain (-x) → f (-x) = f x

-- Define what it means for a function to be odd
def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x, domain x → domain (-x) → f (-x) = -f x

-- Theorem stating that f is neither even nor odd
theorem f_neither_even_nor_odd :
  ¬(is_even f) ∧ ¬(is_odd f) :=
sorry

end f_neither_even_nor_odd_l3483_348324


namespace total_students_l3483_348383

/-- Given a row of students where a person is 4th from the left and 9th from the right,
    and there are 5 such rows with an equal number of students in each row,
    prove that the total number of students is 60. -/
theorem total_students (left_position : Nat) (right_position : Nat) (num_rows : Nat) 
  (h1 : left_position = 4)
  (h2 : right_position = 9)
  (h3 : num_rows = 5) :
  (left_position + right_position - 1) * num_rows = 60 := by
  sorry

#check total_students

end total_students_l3483_348383


namespace spread_combination_exists_l3483_348314

/-- Represents the calories in one piece of bread -/
def bread_calories : ℝ := 100

/-- Represents the calories in one serving of peanut butter -/
def peanut_butter_calories : ℝ := 200

/-- Represents the calories in one serving of strawberry jam -/
def strawberry_jam_calories : ℝ := 120

/-- Represents the calories in one serving of almond butter -/
def almond_butter_calories : ℝ := 180

/-- Represents the total calories needed for breakfast -/
def total_calories : ℝ := 500

/-- Theorem stating that there exist non-negative real numbers p, j, and a
    satisfying the calorie equation and ensuring at least one spread is used -/
theorem spread_combination_exists :
  ∃ (p j a : ℝ), p ≥ 0 ∧ j ≥ 0 ∧ a ≥ 0 ∧
  bread_calories + peanut_butter_calories * p + strawberry_jam_calories * j + almond_butter_calories * a = total_calories ∧
  p + j + a > 0 := by
  sorry

end spread_combination_exists_l3483_348314


namespace marbles_given_to_sam_l3483_348303

/-- Given that Mike initially had 8 marbles and now has 4 left, prove that he gave 4 marbles to Sam. -/
theorem marbles_given_to_sam 
  (initial_marbles : Nat) 
  (remaining_marbles : Nat) 
  (h1 : initial_marbles = 8) 
  (h2 : remaining_marbles = 4) : 
  initial_marbles - remaining_marbles = 4 := by
  sorry

end marbles_given_to_sam_l3483_348303


namespace sin_135_degrees_l3483_348347

theorem sin_135_degrees : Real.sin (135 * π / 180) = Real.sqrt 2 / 2 := by
  sorry

end sin_135_degrees_l3483_348347


namespace smallest_dual_base_representation_l3483_348338

def base_conversion (digits : List Nat) (base : Nat) : Nat :=
  digits.foldr (fun digit acc => digit + base * acc) 0

theorem smallest_dual_base_representation :
  ∃ (a b : Nat), a > 2 ∧ b > 2 ∧
  base_conversion [2, 1] a = 7 ∧
  base_conversion [1, 2] b = 7 ∧
  (∀ (x y : Nat), x > 2 → y > 2 →
    base_conversion [2, 1] x = base_conversion [1, 2] y →
    base_conversion [2, 1] x ≥ 7) :=
by sorry

end smallest_dual_base_representation_l3483_348338


namespace xy_coefficient_zero_l3483_348374

theorem xy_coefficient_zero (k : ℚ) (x y : ℚ) :
  k = 1 / 3 → -3 * k + 1 = 0 :=
by
  sorry

#check xy_coefficient_zero

end xy_coefficient_zero_l3483_348374


namespace window_width_window_width_is_six_l3483_348399

/-- The width of a window in a bedroom, given the room dimensions and areas of doors and windows. -/
theorem window_width : ℝ :=
  let room_width : ℝ := 20
  let room_length : ℝ := 20
  let room_height : ℝ := 8
  let door1_width : ℝ := 3
  let door1_height : ℝ := 7
  let door2_width : ℝ := 5
  let door2_height : ℝ := 7
  let window_height : ℝ := 4
  let total_paint_area : ℝ := 560
  let total_wall_area : ℝ := 4 * room_width * room_height
  let door1_area : ℝ := door1_width * door1_height
  let door2_area : ℝ := door2_width * door2_height
  let window_width : ℝ := (total_wall_area - door1_area - door2_area - total_paint_area) / window_height
  window_width

/-- Proof that the window width is 6 feet. -/
theorem window_width_is_six : window_width = 6 := by
  sorry

end window_width_window_width_is_six_l3483_348399


namespace red_points_on_function_l3483_348302

/-- A point in the plane is a red point if its x-coordinate is a natural number
    and its y-coordinate is a perfect square. -/
def is_red_point (p : ℝ × ℝ) : Prop :=
  ∃ (n m : ℕ), p.1 = n ∧ p.2 = m^2

/-- The function y = (x-36)(x-144) - 1991 -/
def f (x : ℝ) : ℝ := (x - 36) * (x - 144) - 1991

theorem red_points_on_function :
  ∀ p : ℝ × ℝ, is_red_point p ∧ f p.1 = p.2 ↔ p = (2544, 6017209) ∨ p = (444, 120409) := by
  sorry

end red_points_on_function_l3483_348302


namespace f_two_eq_zero_f_x_plus_two_f_x_plus_four_l3483_348381

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the properties of f
axiom odd_function : ∀ x, f (-x) = -f x
axiom function_property : ∀ x, f (x + 2) + 2 * f (-x) = 0

-- Theorem statements
theorem f_two_eq_zero : f 2 = 0 := by sorry

theorem f_x_plus_two : ∀ x, f (x + 2) = 2 * f x := by sorry

theorem f_x_plus_four : ∀ x, f (x + 4) = 4 * f x := by sorry

end f_two_eq_zero_f_x_plus_two_f_x_plus_four_l3483_348381


namespace hyperbola_equation_l3483_348371

/-- Given a hyperbola and an ellipse with the same foci, where the eccentricity of the hyperbola is twice that of the ellipse, prove that the equation of the hyperbola is x²/4 - y²/3 = 1 -/
theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x y : ℝ, x^2/a^2 - y^2/b^2 = 1) →
  (∀ x y : ℝ, x^2/16 + y^2/9 = 1) →
  (∃ c : ℝ, c^2 = a^2 + b^2 ∧ c^2 = 16 - 9) →
  (∃ e_e e_h : ℝ, e_h = 2*e_e ∧ e_e = Real.sqrt 7 / 4 ∧ e_h = Real.sqrt 7 / a) →
  (∀ x y : ℝ, x^2/4 - y^2/3 = 1) :=
by sorry

end hyperbola_equation_l3483_348371


namespace envelope_distribution_theorem_l3483_348342

/-- Represents the number of members in the WeChat group -/
def num_members : ℕ := 5

/-- Represents the number of red envelopes -/
def num_envelopes : ℕ := 4

/-- Represents the number of 2-yuan envelopes -/
def num_2yuan : ℕ := 2

/-- Represents the number of 3-yuan envelopes -/
def num_3yuan : ℕ := 2

/-- Represents the number of specific members (A and B) who must get an envelope -/
def num_specific_members : ℕ := 2

/-- Represents the function that calculates the number of ways to distribute the envelopes -/
noncomputable def num_distribution_ways : ℕ := sorry

theorem envelope_distribution_theorem :
  num_distribution_ways = 18 := by sorry

end envelope_distribution_theorem_l3483_348342


namespace range_of_a_when_no_solutions_l3483_348340

/-- The range of a when x^2 + (2a-1)x + a ≠ 0 for all x ∈ (-2,0) -/
theorem range_of_a_when_no_solutions (a : ℝ) : 
  (∀ x ∈ Set.Ioo (-2) 0, x^2 + (2*a - 1)*x + a ≠ 0) ↔ 
  a ∈ Set.Icc 0 ((2 + Real.sqrt 3) / 2) :=
sorry

end range_of_a_when_no_solutions_l3483_348340


namespace min_value_line_circle_l3483_348336

/-- The minimum value of 1/a + 4/b given the conditions -/
theorem min_value_line_circle (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (∃ x y : ℝ, a * x + b * y + 1 = 0 ∧ x^2 + y^2 + 8*x + 2*y + 1 = 0) →
  (∀ a' b' : ℝ, a' > 0 → b' > 0 → 
    (∃ x' y' : ℝ, a' * x' + b' * y' + 1 = 0 ∧ x'^2 + y'^2 + 8*x' + 2*y' + 1 = 0) →
    1/a + 4/b ≤ 1/a' + 4/b') →
  1/a + 4/b = 16 :=
sorry

end min_value_line_circle_l3483_348336


namespace max_area_inscribed_equilateral_triangle_max_area_inscribed_equilateral_triangle_proof_l3483_348352

/-- The maximum area of an equilateral triangle inscribed in a 12 by 13 rectangle --/
theorem max_area_inscribed_equilateral_triangle : ℝ :=
  let rectangle_width : ℝ := 12
  let rectangle_height : ℝ := 13
  let max_area : ℝ := 312 * Real.sqrt 3 - 936
  max_area

/-- Proof that the maximum area of an equilateral triangle inscribed in a 12 by 13 rectangle is 312√3 - 936 --/
theorem max_area_inscribed_equilateral_triangle_proof :
  max_area_inscribed_equilateral_triangle = 312 * Real.sqrt 3 - 936 := by
  sorry

end max_area_inscribed_equilateral_triangle_max_area_inscribed_equilateral_triangle_proof_l3483_348352


namespace discount_percentage_calculation_l3483_348395

theorem discount_percentage_calculation 
  (num_people : ℕ) 
  (discount_per_person : ℝ) 
  (final_price : ℝ) : 
  num_people = 3 →
  discount_per_person = 4 →
  final_price = 48 →
  (((num_people : ℝ) * discount_per_person) / 
   (final_price + (num_people : ℝ) * discount_per_person)) * 100 = 20 := by
sorry

end discount_percentage_calculation_l3483_348395


namespace man_mass_on_boat_l3483_348339

/-- The mass of a man who causes a boat to sink by a certain depth in water. -/
def mass_of_man (boat_length boat_breadth sinking_depth : ℝ) : ℝ :=
  boat_length * boat_breadth * sinking_depth * 1000

theorem man_mass_on_boat :
  mass_of_man 3 2 0.018 = 108 := by
  sorry

end man_mass_on_boat_l3483_348339


namespace remainder_of_M_mod_500_l3483_348306

/-- The number of consecutive 0's at the right end of the decimal representation of n! -/
def trailingZeros (n : ℕ) : ℕ := sorry

/-- The product of factorials from 1 to n -/
def factorialProduct (n : ℕ) : ℕ := sorry

/-- M is the number of consecutive 0's at the right end of the decimal representation of 1!2!3!4!...49!50! -/
def M : ℕ := trailingZeros (factorialProduct 50)

theorem remainder_of_M_mod_500 : M % 500 = 12 := by sorry

end remainder_of_M_mod_500_l3483_348306


namespace matrix_not_invertible_iff_l3483_348354

def matrix (y : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![2*y, 9],
    ![4 - 2*y, 5]]

theorem matrix_not_invertible_iff (y : ℝ) :
  ¬(IsUnit (matrix y).det) ↔ y = 9/7 := by
  sorry

end matrix_not_invertible_iff_l3483_348354


namespace like_terms_imply_equal_exponents_l3483_348331

-- Define what it means for two terms to be "like terms"
def are_like_terms (term1 term2 : ℝ → ℝ → ℝ) : Prop :=
  ∃ (a b : ℝ), ∀ (x y : ℝ), term1 x y = a * (term2 x y)

-- State the theorem
theorem like_terms_imply_equal_exponents :
  are_like_terms (fun x y => 3 * x^4 * y^m) (fun x y => -2 * x^4 * y^2) → m = 2 := by
  sorry

end like_terms_imply_equal_exponents_l3483_348331


namespace point_on_ln_curve_with_specific_tangent_l3483_348309

open Real

/-- Proves that a point on y = ln(x) with tangent line through (-e, -1) has coordinates (e, 1) -/
theorem point_on_ln_curve_with_specific_tangent (x₀ : ℝ) :
  (∃ (A : ℝ × ℝ), 
    A.1 = x₀ ∧ 
    A.2 = log x₀ ∧ 
    (log x₀ - (-1)) / (x₀ - (-Real.exp 1)) = 1 / x₀) →
  x₀ = Real.exp 1 ∧ log x₀ = 1 := by
  sorry


end point_on_ln_curve_with_specific_tangent_l3483_348309


namespace johns_shower_water_usage_rate_l3483_348307

/-- Calculates the water usage rate of John's shower -/
theorem johns_shower_water_usage_rate :
  let weeks : ℕ := 4
  let days_per_week : ℕ := 7
  let shower_frequency : ℕ := 2  -- every other day
  let shower_duration : ℕ := 10  -- minutes
  let total_water_usage : ℕ := 280  -- gallons
  
  let total_days : ℕ := weeks * days_per_week
  let number_of_showers : ℕ := total_days / shower_frequency
  let total_shower_time : ℕ := number_of_showers * shower_duration
  
  (total_water_usage : ℚ) / total_shower_time = 2 := by
  sorry

end johns_shower_water_usage_rate_l3483_348307


namespace lunch_bill_total_l3483_348316

-- Define the costs and discounts
def hotdog_cost : ℝ := 5.36
def salad_cost : ℝ := 5.10
def soda_original_cost : ℝ := 2.95
def chips_original_cost : ℝ := 1.89
def chips_discount : ℝ := 0.15
def soda_discount : ℝ := 0.10

-- Define the function to calculate the discounted price
def apply_discount (original_price : ℝ) (discount : ℝ) : ℝ :=
  original_price * (1 - discount)

-- Define the total cost function
def total_cost : ℝ :=
  hotdog_cost + salad_cost + 
  apply_discount soda_original_cost soda_discount +
  apply_discount chips_original_cost chips_discount

-- Theorem statement
theorem lunch_bill_total : total_cost = 14.7215 := by
  sorry

end lunch_bill_total_l3483_348316


namespace factorization_problems_l3483_348380

theorem factorization_problems :
  (∀ a : ℝ, a^2 - 25 = (a + 5) * (a - 5)) ∧
  (∀ x y : ℝ, 2*x^2*y - 8*x*y + 8*y = 2*y*(x - 2)^2) := by
sorry

end factorization_problems_l3483_348380


namespace triangle_properties_l3483_348322

theorem triangle_properties (a b c : ℝ) (A B C : ℝ) (BD : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →
  a / 2 = b / 3 ∧ b / 3 = c / 4 →
  BD = Real.sqrt 31 →
  BD * 2 = c →
  Real.tan C = -Real.sqrt 15 ∧
  (1 / 2) * a * b * Real.sin C = 3 * Real.sqrt 15 :=
by sorry

end triangle_properties_l3483_348322


namespace sum_of_a_and_b_l3483_348365

theorem sum_of_a_and_b (a b : ℝ) : 
  (abs a = 3) → (abs b = 7) → (abs (a - b) = b - a) → 
  (a + b = 10 ∨ a + b = 4) :=
by
  sorry

end sum_of_a_and_b_l3483_348365


namespace pennys_initial_money_l3483_348337

/-- Penny's initial amount of money given her purchases and remaining balance -/
theorem pennys_initial_money :
  ∀ (sock_pairs : ℕ) (sock_price hat_price remaining : ℚ),
    sock_pairs = 4 →
    sock_price = 2 →
    hat_price = 7 →
    remaining = 5 →
    (sock_pairs : ℚ) * sock_price + hat_price + remaining = 20 :=
by
  sorry

end pennys_initial_money_l3483_348337


namespace binomial_18_choose_4_l3483_348323

theorem binomial_18_choose_4 : Nat.choose 18 4 = 3060 := by
  sorry

end binomial_18_choose_4_l3483_348323


namespace card_game_cost_l3483_348367

theorem card_game_cost (rare_count : ℕ) (uncommon_count : ℕ) (common_count : ℕ)
                       (rare_cost : ℚ) (uncommon_cost : ℚ) (total_cost : ℚ) :
  rare_count = 19 →
  uncommon_count = 11 →
  common_count = 30 →
  rare_cost = 1 →
  uncommon_cost = (1/2) →
  total_cost = 32 →
  (total_cost - (rare_count * rare_cost + uncommon_count * uncommon_cost)) / common_count = (1/4) := by
sorry

end card_game_cost_l3483_348367


namespace valid_a_values_l3483_348319

theorem valid_a_values :
  ∀ a : ℚ, (∃ m : ℤ, a = m + 1/2 ∨ a = m + 1/3 ∨ a = m - 1/3) ↔
  ((∃ m : ℤ, a = m + 1/2) ∨ (∃ m : ℤ, a = m + 1/3) ∨ (∃ m : ℤ, a = m - 1/3)) :=
by sorry

end valid_a_values_l3483_348319


namespace multiply_704_12_by_3_10_l3483_348310

-- Define a function to convert from base 12 to base 10
def base12ToBase10 (n : ℕ) : ℕ := sorry

-- Define a function to convert from base 10 to base 12
def base10ToBase12 (n : ℕ) : ℕ := sorry

-- Define the given number in base 12
def given_number : ℕ := 704

-- Define the multiplier in base 10
def multiplier : ℕ := 3

-- Theorem statement
theorem multiply_704_12_by_3_10 : 
  base10ToBase12 (base12ToBase10 given_number * multiplier) = 1910 := by
  sorry

end multiply_704_12_by_3_10_l3483_348310


namespace circle_not_through_origin_l3483_348379

-- Define the curve
def curve (x y : ℝ) : Prop := 2 * x^2 - y^2 = 5

-- Define the line passing through (0, 2)
def line (k : ℝ) (x y : ℝ) : Prop := y = k * x + 2

-- Define the intersection points A and B
def intersection_points (A B : ℝ × ℝ) (k : ℝ) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    A = (x₁, y₁) ∧ B = (x₂, y₂) ∧
    curve x₁ y₁ ∧ curve x₂ y₂ ∧
    line k x₁ y₁ ∧ line k x₂ y₂

-- Define the circle with diameter AB
def circle_diameter_AB (A B : ℝ × ℝ) (x y : ℝ) : Prop :=
  let midpoint := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  let radius := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) / 2
  (x - midpoint.1)^2 + (y - midpoint.2)^2 = radius^2

-- Theorem statement
theorem circle_not_through_origin (k : ℝ) (A B : ℝ × ℝ) :
  intersection_points A B k →
  ¬ circle_diameter_AB A B 0 0 :=
sorry

end circle_not_through_origin_l3483_348379


namespace parabola_directrix_l3483_348344

/-- The equation of the directrix of a parabola with equation y = -4x² is y = 1/16 -/
theorem parabola_directrix (x y : ℝ) : 
  (y = -4 * x^2) → (∃ k : ℝ, y = k ∧ k = 1/16) :=
by sorry

end parabola_directrix_l3483_348344


namespace complex_power_magnitude_l3483_348389

theorem complex_power_magnitude : Complex.abs ((3 / 4 : ℂ) + (5 / 4 : ℂ) * Complex.I) ^ 4 = 289 / 64 := by
  sorry

end complex_power_magnitude_l3483_348389


namespace percentage_calculation_l3483_348372

theorem percentage_calculation (p : ℝ) : 
  (p / 100) * 170 = 0.20 * 552.50 → p = 65 := by
  sorry

end percentage_calculation_l3483_348372


namespace three_digit_sum_permutations_l3483_348355

/-- Given a three-digit number m = 100a + 10b + c, where a, b, and c are single digits,
    if the sum of m and its five permutations (acb), (bca), (bac), (cab), and (cba) is 3315,
    then m = 015. -/
theorem three_digit_sum_permutations (a b c : ℕ) : 
  (0 ≤ a ∧ a ≤ 9) → 
  (0 ≤ b ∧ b ≤ 9) → 
  (0 ≤ c ∧ c ≤ 9) → 
  let m := 100 * a + 10 * b + c
  (m + (100 * a + 10 * c + b) + (100 * b + 10 * c + a) + 
   (100 * b + 10 * a + c) + (100 * c + 10 * a + b) + 
   (100 * c + 10 * b + a) = 3315) →
  m = 15 :=
by sorry

end three_digit_sum_permutations_l3483_348355


namespace custom_op_seven_five_l3483_348308

/-- Custom operation @ for positive integers -/
def custom_op (a b : ℕ+) : ℚ :=
  (a * b : ℚ) / ((a : ℤ) - (b : ℤ) + 8 : ℚ)

/-- Theorem stating that 7 @ 5 = 7/2 -/
theorem custom_op_seven_five :
  custom_op 7 5 = 7/2 := by
  sorry

end custom_op_seven_five_l3483_348308


namespace jesse_blocks_theorem_l3483_348361

/-- The number of building blocks Jesse started with --/
def total_blocks : ℕ := sorry

/-- The number of blocks used for the cityscape --/
def cityscape_blocks : ℕ := 80

/-- The number of blocks used for the farmhouse --/
def farmhouse_blocks : ℕ := 123

/-- The number of blocks used for the zoo --/
def zoo_blocks : ℕ := 95

/-- The number of blocks used for the first fenced-in area --/
def fence1_blocks : ℕ := 57

/-- The number of blocks used for the second fenced-in area --/
def fence2_blocks : ℕ := 43

/-- The number of blocks used for the third fenced-in area --/
def fence3_blocks : ℕ := 62

/-- The number of blocks borrowed by Jesse's friend --/
def borrowed_blocks : ℕ := 35

/-- The number of blocks Jesse had left over --/
def leftover_blocks : ℕ := 84

/-- Theorem stating that the total number of blocks Jesse started with is equal to the sum of all blocks used in constructions, blocks left over, and blocks borrowed by his friend --/
theorem jesse_blocks_theorem : 
  total_blocks = cityscape_blocks + farmhouse_blocks + zoo_blocks + 
                 fence1_blocks + fence2_blocks + fence3_blocks + 
                 borrowed_blocks + leftover_blocks := by sorry

end jesse_blocks_theorem_l3483_348361


namespace road_system_car_distribution_l3483_348317

theorem road_system_car_distribution :
  ∀ (total_cars : ℕ) (bc de bd ce cd : ℕ),
    total_cars = 36 →
    bc = de + 10 →
    cd = 2 →
    total_cars = bc + bd →
    bc = cd + ce →
    de = bd - cd →
    (bc = 24 ∧ bd = 12 ∧ de = 14 ∧ ce = 22) :=
by
  sorry

end road_system_car_distribution_l3483_348317


namespace parametric_equations_of_Γ_polar_equation_of_perpendicular_line_l3483_348384

-- Define the unit circle
def unit_circle (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define the transformation
def transform (x y : ℝ) : ℝ × ℝ := (2*x, 3*y)

-- Define the curve Γ
def Γ (x y : ℝ) : Prop :=
  ∃ (x₀ y₀ : ℝ), unit_circle x₀ y₀ ∧ transform x₀ y₀ = (x, y)

-- Define the intersecting line l
def line_l (x y : ℝ) : Prop := 3*x + 2*y - 6 = 0

-- Theorem for parametric equations of Γ
theorem parametric_equations_of_Γ :
  ∀ t : ℝ, Γ (2 * Real.cos t) (3 * Real.sin t) := sorry

-- Theorem for polar equation of perpendicular line
theorem polar_equation_of_perpendicular_line :
  ∃ (P₁ P₂ : ℝ × ℝ),
    Γ P₁.1 P₁.2 ∧ Γ P₂.1 P₂.2 ∧
    line_l P₁.1 P₁.2 ∧ line_l P₂.1 P₂.2 ∧
    (∀ ρ θ : ℝ,
      4 * ρ * Real.cos θ - 6 * ρ * Real.sin θ + 5 = 0 ↔
      (∃ (k : ℝ),
        ρ * Real.cos θ = (P₁.1 + P₂.1) / 2 + k * 2 / 3 ∧
        ρ * Real.sin θ = (P₁.2 + P₂.2) / 2 - k * 3 / 2)) := sorry

end parametric_equations_of_Γ_polar_equation_of_perpendicular_line_l3483_348384


namespace budget_utilities_percentage_l3483_348363

theorem budget_utilities_percentage (transportation : ℝ) (research_development : ℝ) 
  (equipment : ℝ) (supplies : ℝ) (salaries_degrees : ℝ) :
  transportation = 15 →
  research_development = 9 →
  equipment = 4 →
  supplies = 2 →
  salaries_degrees = 234 →
  (salaries_degrees / 360) * 100 + transportation + research_development + equipment + supplies + 
    (100 - ((salaries_degrees / 360) * 100 + transportation + research_development + equipment + supplies)) = 100 →
  100 - ((salaries_degrees / 360) * 100 + transportation + research_development + equipment + supplies) = 5 := by
sorry

end budget_utilities_percentage_l3483_348363


namespace bead_necklace_problem_l3483_348348

theorem bead_necklace_problem (total_beads : Nat) (num_necklaces : Nat) (h1 : total_beads = 31) (h2 : num_necklaces = 4) :
  total_beads % num_necklaces = 3 := by
  sorry

end bead_necklace_problem_l3483_348348


namespace valid_parameterization_l3483_348329

/-- Represents a 2D vector -/
structure Vector2D where
  x : ℝ
  y : ℝ

/-- Represents a line parameterization -/
structure LineParam where
  point : Vector2D
  direction : Vector2D

/-- Checks if a point lies on the line y = -2x + 7 -/
def liesOnLine (v : Vector2D) : Prop :=
  v.y = -2 * v.x + 7

/-- Checks if a vector is a scalar multiple of (1, -2) -/
def isValidDirection (v : Vector2D) : Prop :=
  ∃ (k : ℝ), v.x = k * 1 ∧ v.y = k * (-2)

/-- Main theorem: A parameterization is valid iff it satisfies both conditions -/
theorem valid_parameterization (p : LineParam) :
  (liesOnLine p.point ∧ isValidDirection p.direction) ↔
  (∀ (t : ℝ), liesOnLine ⟨p.point.x + t * p.direction.x, p.point.y + t * p.direction.y⟩) :=
by sorry

end valid_parameterization_l3483_348329


namespace rectangle_area_l3483_348341

theorem rectangle_area (L B : ℝ) (h1 : L - B = 23) (h2 : 2 * (L + B) = 266) : L * B = 4290 := by
  sorry

end rectangle_area_l3483_348341


namespace field_trip_students_l3483_348311

theorem field_trip_students (num_vans : ℕ) (num_minibusses : ℕ) (students_per_van : ℕ) (students_per_minibus : ℕ)
  (h1 : num_vans = 6)
  (h2 : num_minibusses = 4)
  (h3 : students_per_van = 10)
  (h4 : students_per_minibus = 24) :
  num_vans * students_per_van + num_minibusses * students_per_minibus = 156 :=
by sorry

end field_trip_students_l3483_348311


namespace solution_difference_l3483_348321

theorem solution_difference (a b : ℝ) : 
  (∀ x : ℝ, (3 * x - 9) / (x^2 + 3 * x - 18) = x + 1 ↔ x = a ∨ x = b) →
  a ≠ b →
  a > b →
  a - b = 1 := by sorry

end solution_difference_l3483_348321


namespace equilateral_triangle_polyhedron_vertices_l3483_348357

/-- A polyhedron with equilateral triangular faces -/
structure EquilateralTrianglePolyhedron where
  /-- Number of faces -/
  f : ℕ
  /-- Number of edges -/
  e : ℕ
  /-- Number of vertices -/
  v : ℕ
  /-- Each face is an equilateral triangle -/
  faces_are_equilateral_triangles : f = 8
  /-- Euler's formula for polyhedra -/
  euler_formula : v - e + f = 2
  /-- Each edge is shared by exactly two faces -/
  edges_shared : e = (3 * f) / 2

/-- Theorem: A polyhedron with 8 equilateral triangular faces has 6 vertices -/
theorem equilateral_triangle_polyhedron_vertices 
  (p : EquilateralTrianglePolyhedron) : p.v = 6 := by
  sorry

end equilateral_triangle_polyhedron_vertices_l3483_348357


namespace max_value_constraint_l3483_348330

theorem max_value_constraint (x y : ℝ) (h : 16 * x^2 + y^2 + 4 * x * y = 3) :
  ∃ (M : ℝ), M = 2 ∧ ∀ (z : ℝ), 4 * x + y ≤ z → z ≤ M :=
sorry

end max_value_constraint_l3483_348330


namespace symmetric_complex_numbers_l3483_348375

theorem symmetric_complex_numbers (z₁ z₂ : ℂ) :
  (z₁ = 2 - 3*I) →
  (z₁ + z₂ = 0) →
  (z₂ = -2 + 3*I) := by
  sorry

end symmetric_complex_numbers_l3483_348375


namespace right_triangle_inequality_l3483_348373

theorem right_triangle_inequality (a b c m : ℝ) 
  (h1 : c^2 = a^2 + b^2) 
  (h2 : a * b = c * m) 
  (h3 : m > 0) : 
  m + c > a + b := by
  sorry

end right_triangle_inequality_l3483_348373


namespace mary_carrots_proof_l3483_348313

/-- The number of carrots Sandy grew -/
def sandys_carrots : ℕ := 8

/-- The total number of carrots grown by Sandy and Mary -/
def total_carrots : ℕ := 14

/-- The number of carrots Mary grew -/
def marys_carrots : ℕ := total_carrots - sandys_carrots

theorem mary_carrots_proof : 
  marys_carrots = total_carrots - sandys_carrots :=
by sorry

end mary_carrots_proof_l3483_348313


namespace two_std_dev_below_mean_is_8_5_l3483_348391

/-- A normal distribution with given mean and standard deviation -/
structure NormalDistribution where
  mean : ℝ
  std_dev : ℝ
  std_dev_pos : 0 < std_dev

/-- The value that is exactly n standard deviations less than the mean -/
def value_n_std_dev_below_mean (d : NormalDistribution) (n : ℝ) : ℝ :=
  d.mean - n * d.std_dev

/-- Theorem: For a normal distribution with mean 10.5 and standard deviation 1,
    the value 2 standard deviations below the mean is 8.5 -/
theorem two_std_dev_below_mean_is_8_5 :
  let d : NormalDistribution := ⟨10.5, 1, by norm_num⟩
  value_n_std_dev_below_mean d 2 = 8.5 := by
  sorry

end two_std_dev_below_mean_is_8_5_l3483_348391


namespace city_distance_l3483_348350

def is_valid_distance (S : ℕ) : Prop :=
  ∀ x : ℕ, x ≤ S → (Nat.gcd x (S - x) = 1 ∨ Nat.gcd x (S - x) = 3 ∨ Nat.gcd x (S - x) = 13)

theorem city_distance : 
  (∃ S : ℕ, is_valid_distance S ∧ ∀ T : ℕ, T < S → ¬is_valid_distance T) ∧
  (∀ S : ℕ, (is_valid_distance S ∧ ∀ T : ℕ, T < S → ¬is_valid_distance T) → S = 39) :=
sorry

end city_distance_l3483_348350


namespace inequality_equivalence_l3483_348377

theorem inequality_equivalence (x y : ℝ) : 
  y^2 - x*y < 0 ↔ (0 < y ∧ y < x) ∨ (y < x ∧ x < 0) := by
sorry

end inequality_equivalence_l3483_348377


namespace factorization_equality_l3483_348378

theorem factorization_equality (a b : ℝ) : a * b^2 - 4 * a * b + 4 * a = a * (b - 2)^2 := by
  sorry

end factorization_equality_l3483_348378


namespace color_box_problem_l3483_348382

theorem color_box_problem (total_pencils : ℕ) (emily_and_friends : ℕ) (colors : ℕ) : 
  total_pencils = 56 → emily_and_friends = 8 → total_pencils = emily_and_friends * colors → colors = 7 := by
  sorry

end color_box_problem_l3483_348382


namespace wolves_games_count_l3483_348390

theorem wolves_games_count : 
  ∀ (initial_games : ℕ) (initial_wins : ℕ),
    initial_wins = (2 * initial_games / 5) →
    ∃ (total_games : ℕ),
      total_games = initial_games + 10 ∧
      (initial_wins + 9 : ℚ) / total_games = 3/5 ∧
      total_games = 25 :=
by sorry

end wolves_games_count_l3483_348390


namespace smallest_factorization_l3483_348333

def is_valid_factorization (b r s : ℤ) : Prop :=
  r * s = 4032 ∧ r + s = b

def has_integer_factorization (b : ℤ) : Prop :=
  ∃ r s : ℤ, is_valid_factorization b r s

theorem smallest_factorization : 
  (∀ b : ℤ, b > 0 ∧ b < 127 → ¬(has_integer_factorization b)) ∧ 
  has_integer_factorization 127 :=
sorry

end smallest_factorization_l3483_348333


namespace parabola_equation_l3483_348325

-- Define a parabola passing through a point
def parabola_through_point (x y : ℝ) : Prop :=
  (y^2 = x) ∨ (x^2 = -8*y)

-- Theorem statement
theorem parabola_equation : parabola_through_point 4 (-2) := by
  sorry

end parabola_equation_l3483_348325


namespace triangle_area_is_71_l3483_348300

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Line in 2D space represented by slope and a point -/
structure Line where
  slope : ℝ
  point : Point

/-- Vertical line represented by x-coordinate -/
structure VerticalLine where
  x : ℝ

theorem triangle_area_is_71 
  (l1 : Line) 
  (l2 : Line) 
  (l3 : VerticalLine)
  (h1 : l1.slope = 3)
  (h2 : l2.slope = -1/3)
  (h3 : l1.point = l2.point)
  (h4 : l1.point = ⟨1, 1⟩)
  (h5 : l3.x + (3 * l3.x - 2) = 12) : 
  ∃ (A B C : Point), 
    (A.x + A.y = 12 ∧ A.y = 3 * A.x - 2) ∧
    (B.x + B.y = 12 ∧ B.y = -1/3 * B.x + 4/3) ∧
    (C = l1.point) ∧
    abs ((A.x * (B.y - C.y) + B.x * (C.y - A.y) + C.x * (A.y - B.y)) / 2) = 71 := by
  sorry

end triangle_area_is_71_l3483_348300


namespace point_on_linear_function_l3483_348388

/-- Given a linear function y = -2x + 3 that passes through the point (a, -4), prove that a = 7/2 -/
theorem point_on_linear_function (a : ℝ) : 
  (∀ x y, y = -2 * x + 3) →  -- Linear function definition
  -4 = -2 * a + 3 →          -- Point (a, -4) lies on the line
  a = 7 / 2 := by
sorry

end point_on_linear_function_l3483_348388


namespace treats_ratio_wanda_to_jane_l3483_348318

/-- Proves that the ratio of treats Wanda brings compared to Jane is 1:2 -/
theorem treats_ratio_wanda_to_jane :
  ∀ (jane_treats jane_bread wanda_treats wanda_bread : ℕ),
    jane_bread = (75 * jane_treats) / 100 →
    wanda_bread = 3 * wanda_treats →
    wanda_bread = 90 →
    jane_treats + jane_bread + wanda_treats + wanda_bread = 225 →
    wanda_treats * 2 = jane_treats :=
by
  sorry

#check treats_ratio_wanda_to_jane

end treats_ratio_wanda_to_jane_l3483_348318


namespace sunflower_seed_distribution_l3483_348334

theorem sunflower_seed_distribution (total_seeds : ℕ) (num_cans : ℕ) (seeds_per_can : ℕ) 
  (h1 : total_seeds = 54)
  (h2 : num_cans = 9)
  (h3 : total_seeds = num_cans * seeds_per_can) :
  seeds_per_can = 6 := by
  sorry

end sunflower_seed_distribution_l3483_348334


namespace two_less_than_negative_one_l3483_348366

theorem two_less_than_negative_one : (- 1) - 2 = - 3 := by
  sorry

end two_less_than_negative_one_l3483_348366


namespace vector_BC_proof_l3483_348315

-- Define the points and vectors
def A : ℝ × ℝ := (0, 1)
def B : ℝ × ℝ := (2, 1)
def AC : ℝ × ℝ := (-3, -2)

-- State the theorem
theorem vector_BC_proof : 
  let BC : ℝ × ℝ := (AC.1 - (B.1 - A.1), AC.2 - (B.2 - A.2))
  BC = (-5, -2) := by sorry

end vector_BC_proof_l3483_348315


namespace largest_six_digit_with_product_40320_l3483_348369

def is_six_digit (n : ℕ) : Prop := 100000 ≤ n ∧ n ≤ 999999

def digit_product (n : ℕ) : ℕ :=
  (n.digits 10).prod

theorem largest_six_digit_with_product_40320 :
  ∃ (n : ℕ), is_six_digit n ∧ digit_product n = 40320 ∧
  ∀ (m : ℕ), is_six_digit m → digit_product m = 40320 → m ≤ n :=
by
  use 988752
  sorry

#eval digit_product 988752  -- Should output 40320

end largest_six_digit_with_product_40320_l3483_348369


namespace inequality_proof_l3483_348358

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^2 + 2) * (b^2 + 2) * (c^2 + 2) ≥ 9 * (a * b + b * c + c * a) := by
  sorry

end inequality_proof_l3483_348358


namespace smallest_four_digit_divisible_by_57_l3483_348394

theorem smallest_four_digit_divisible_by_57 :
  ∀ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 57 = 0 → n ≥ 1026 :=
by sorry

end smallest_four_digit_divisible_by_57_l3483_348394


namespace sum_of_digits_of_product_l3483_348353

def product : ℕ := 11 * 101 * 111 * 110011

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

theorem sum_of_digits_of_product :
  sum_of_digits product = 48 := by
  sorry

end sum_of_digits_of_product_l3483_348353


namespace rest_area_milepost_l3483_348346

def first_exit : ℝ := 20
def seventh_exit : ℝ := 140

theorem rest_area_milepost : 
  let midpoint := (first_exit + seventh_exit) / 2
  midpoint = 80 := by sorry

end rest_area_milepost_l3483_348346


namespace eight_b_value_l3483_348320

theorem eight_b_value (a b : ℚ) (h1 : 7 * a + 3 * b = 0) (h2 : a = b - 3) : 8 * b = 84/5 := by
  sorry

end eight_b_value_l3483_348320


namespace problem_solution_l3483_348305

theorem problem_solution (x y : ℚ) (hx : x = 3/5) (hy : y = 5/3) :
  (1/3) * x^8 * y^9 = 5/9 := by
  sorry

end problem_solution_l3483_348305


namespace set_intersection_problem_l3483_348393

theorem set_intersection_problem :
  let A : Set ℝ := {x | -1 ≤ x ∧ x < 3}
  let B : Set ℝ := {-1, 0, 2, 3}
  A ∩ B = {-1, 0, 2} := by sorry

end set_intersection_problem_l3483_348393


namespace equation_solution_l3483_348396

theorem equation_solution (x : ℝ) (h : 1/x + 1/(2*x) + 1/(3*x) = 1/12) : x = 22 := by
  sorry

end equation_solution_l3483_348396


namespace jelly_bean_probability_l3483_348356

theorem jelly_bean_probability (p_green p_blue p_red p_yellow : ℝ) :
  p_green = 0.25 →
  p_blue = 0.35 →
  p_green + p_blue + p_red + p_yellow = 1 →
  p_red + p_yellow = 0.40 := by
  sorry

end jelly_bean_probability_l3483_348356


namespace complex_product_simplification_l3483_348349

/-- Given real non-zero numbers a and b, and the imaginary unit i,
    prove that (ax+biy)(ax-biy) = a^2x^2 + b^2y^2 -/
theorem complex_product_simplification
  (a b x y : ℝ) (i : ℂ)
  (ha : a ≠ 0)
  (hb : b ≠ 0)
  (hi : i^2 = -1)
  : (a*x + b*i*y) * (a*x - b*i*y) = a^2 * x^2 + b^2 * y^2 :=
by sorry

end complex_product_simplification_l3483_348349


namespace expression_evaluation_l3483_348343

theorem expression_evaluation (a : ℚ) (h : a = -1/3) : 
  (2 - a) * (2 + a) - 2 * a * (a + 3) + 3 * a^2 = 6 := by
  sorry

end expression_evaluation_l3483_348343


namespace smallest_angle_in_pentadecagon_l3483_348327

/-- Represents the number of sides in the polygon -/
def n : ℕ := 15

/-- The sum of internal angles in an n-sided polygon -/
def angleSum : ℝ := (n - 2) * 180

/-- The largest angle in the sequence -/
def largestAngle : ℝ := 176

/-- Theorem: In a convex 15-sided polygon where the angles form an arithmetic sequence 
    and the largest angle is 176°, the smallest angle is 136°. -/
theorem smallest_angle_in_pentadecagon : 
  ∃ (a d : ℝ), 
    (∀ i : ℕ, i < n → a + i * d ≤ a + (n - 1) * d) ∧  -- Convexity condition
    (a + (n - 1) * d = largestAngle) ∧                -- Largest angle condition
    (n * (2 * a + (n - 1) * d) / 2 = angleSum) ∧      -- Sum of angles condition
    (a = 136)                                         -- Conclusion
    := by sorry

end smallest_angle_in_pentadecagon_l3483_348327


namespace sum_of_solutions_quadratic_l3483_348328

theorem sum_of_solutions_quadratic (x : ℝ) : 
  (∀ x, x^2 = 9*x - 20 → x + (9 - x) = 9) := by
  sorry

end sum_of_solutions_quadratic_l3483_348328


namespace exists_nonpositive_square_l3483_348304

theorem exists_nonpositive_square : ∃ x : ℝ, -x^2 ≥ 0 := by
  sorry

end exists_nonpositive_square_l3483_348304


namespace sum_mod_seven_l3483_348370

theorem sum_mod_seven : (8171 + 8172 + 8173 + 8174 + 8175) % 7 = 3 := by
  sorry

end sum_mod_seven_l3483_348370


namespace deleted_items_count_l3483_348335

/-- Calculates the total number of deleted items given initial and final counts of apps and files, and the number of files transferred. -/
def totalDeletedItems (initialApps initialFiles finalApps finalFiles transferredFiles : ℕ) : ℕ :=
  (initialApps - finalApps) + (initialFiles - (finalFiles + transferredFiles))

/-- Theorem stating that the total number of deleted items is 24 given the problem conditions. -/
theorem deleted_items_count :
  totalDeletedItems 17 21 3 7 4 = 24 := by
  sorry

end deleted_items_count_l3483_348335


namespace card_game_probabilities_l3483_348332

-- Define the cards for A and B
def A_cards : Finset ℕ := {2, 3}
def B_cards : Finset ℕ := {1, 2, 3, 4}

-- Define a function to check if a sum is odd
def is_odd_sum (a b : ℕ) : Bool := (a + b) % 2 = 1

-- Define a function to check if B wins
def B_wins (a b : ℕ) : Bool := b > a

theorem card_game_probabilities :
  -- Probability of B drawing two cards with an odd sum
  (Finset.filter (fun p => is_odd_sum p.1 p.2) (Finset.product B_cards B_cards)).card / (Finset.product B_cards B_cards).card = 2/3 ∧
  -- Probability of B winning when A and B each draw a card
  (Finset.filter (fun p => B_wins p.1 p.2) (Finset.product A_cards B_cards)).card / (Finset.product A_cards B_cards).card = 3/8 :=
by sorry

end card_game_probabilities_l3483_348332


namespace remaining_milk_quantities_l3483_348301

/-- Represents the types of milk available in the store -/
inductive MilkType
  | Whole
  | LowFat
  | Almond

/-- Represents the initial quantities of milk bottles -/
def initial_quantity : MilkType → Nat
  | MilkType.Whole => 15
  | MilkType.LowFat => 12
  | MilkType.Almond => 8

/-- Represents Jason's purchase of whole milk -/
def jason_purchase : Nat := 5

/-- Represents Harry's purchase of low-fat milk -/
def harry_lowfat_purchase : Nat := 5  -- 4 bought + 1 free

/-- Represents Harry's purchase of almond milk -/
def harry_almond_purchase : Nat := 2

/-- Calculates the remaining quantity of a given milk type after purchases -/
def remaining_quantity (milk_type : MilkType) : Nat :=
  match milk_type with
  | MilkType.Whole => initial_quantity MilkType.Whole - jason_purchase
  | MilkType.LowFat => initial_quantity MilkType.LowFat - harry_lowfat_purchase
  | MilkType.Almond => initial_quantity MilkType.Almond - harry_almond_purchase

/-- Theorem stating the remaining quantities of milk bottles after purchases -/
theorem remaining_milk_quantities :
  remaining_quantity MilkType.Whole = 10 ∧
  remaining_quantity MilkType.LowFat = 7 ∧
  remaining_quantity MilkType.Almond = 6 := by
  sorry

end remaining_milk_quantities_l3483_348301


namespace parallel_line_through_point_l3483_348345

/-- A line in the 2D plane represented by the equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if a point (x, y) lies on a given line -/
def Line.contains (l : Line) (x y : ℝ) : Prop :=
  l.a * x + l.b * y + l.c = 0

/-- Checks if two lines are parallel -/
def Line.parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

theorem parallel_line_through_point 
  (given_line : Line) 
  (point : ℝ × ℝ) 
  (parallel_line : Line) :
  given_line.a = 1 ∧ 
  given_line.b = -2 ∧ 
  given_line.c = 0 ∧
  point = (1, 6) ∧
  parallel_line.a = 1 ∧
  parallel_line.b = -2 ∧
  parallel_line.c = 11 →
  parallel_line.contains point.1 point.2 ∧
  Line.parallel given_line parallel_line := by
  sorry

#check parallel_line_through_point

end parallel_line_through_point_l3483_348345


namespace chord_intersection_l3483_348362

/-- Given a circle and a line intersecting it, prove the value of the line's slope -/
theorem chord_intersection (x y : ℝ) (a : ℝ) : 
  (x^2 + y^2 - 2*x - 8*y + 13 = 0) →  -- Circle equation
  (a*x + y - 1 = 0) →                -- Line equation
  (∃ (x1 y1 x2 y2 : ℝ), 
    (x1^2 + y1^2 - 2*x1 - 8*y1 + 13 = 0) ∧ 
    (x2^2 + y2^2 - 2*x2 - 8*y2 + 13 = 0) ∧ 
    (a*x1 + y1 - 1 = 0) ∧ 
    (a*x2 + y2 - 1 = 0) ∧ 
    ((x1 - x2)^2 + (y1 - y2)^2 = 12)) →  -- Chord length condition
  (a = -4/3) :=
by sorry

end chord_intersection_l3483_348362


namespace some_humans_are_pondering_l3483_348385

-- Define the universe
variable (U : Type)

-- Define predicates
variable (Freshman : U → Prop)
variable (GradStudent : U → Prop)
variable (Human : U → Prop)
variable (Pondering : U → Prop)

-- State the theorem
theorem some_humans_are_pondering
  (h1 : ∀ x, Freshman x → Human x)
  (h2 : ∀ x, GradStudent x → Human x)
  (h3 : ∃ x, GradStudent x ∧ Pondering x) :
  ∃ x, Human x ∧ Pondering x :=
sorry

end some_humans_are_pondering_l3483_348385


namespace polynomial_expansion_l3483_348368

theorem polynomial_expansion (t : ℝ) :
  (3 * t^2 - 4 * t + 3) * (-2 * t^2 + 3 * t - 4) =
  -6 * t^4 + 17 * t^3 - 30 * t^2 + 25 * t - 12 := by
  sorry

end polynomial_expansion_l3483_348368


namespace five_balls_three_boxes_l3483_348398

/-- Represents the number of ways to distribute n indistinguishable balls into k indistinguishable boxes -/
def distributeBalls (n : ℕ) (k : ℕ) : ℕ := sorry

/-- The number of ways to distribute 5 indistinguishable balls into 3 indistinguishable boxes is 5 -/
theorem five_balls_three_boxes : distributeBalls 5 3 = 5 := by sorry

end five_balls_three_boxes_l3483_348398


namespace initial_tax_rate_is_42_percent_l3483_348392

-- Define the initial tax rate as a real number between 0 and 1
variable (initial_rate : ℝ) (h_rate : 0 ≤ initial_rate ∧ initial_rate ≤ 1)

-- Define the new tax rate (32%)
def new_rate : ℝ := 0.32

-- Define the annual income
def annual_income : ℝ := 42400

-- Define the differential savings
def differential_savings : ℝ := 4240

-- Theorem statement
theorem initial_tax_rate_is_42_percent :
  (initial_rate * annual_income - new_rate * annual_income = differential_savings) →
  initial_rate = 0.42 := by
  sorry


end initial_tax_rate_is_42_percent_l3483_348392


namespace no_movement_after_n_commands_l3483_348312

/-- Represents the state of the line of children -/
inductive ChildState
  | Boy
  | Girl

/-- Represents a line of children -/
def ChildLine := List ChildState

/-- Swaps adjacent boy-girl pairs in the line -/
def swapAdjacent (line : ChildLine) : ChildLine :=
  sorry

/-- Applies the swap command n times -/
def applyNCommands (n : Nat) (line : ChildLine) : ChildLine :=
  sorry

/-- Checks if any more swaps are possible -/
def canSwap (line : ChildLine) : Bool :=
  sorry

/-- Initial line of alternating boys and girls -/
def initialLine (n : Nat) : ChildLine :=
  sorry

theorem no_movement_after_n_commands (n : Nat) :
  canSwap (applyNCommands n (initialLine n)) = false :=
  sorry

end no_movement_after_n_commands_l3483_348312


namespace brick_height_l3483_348364

/-- The surface area of a rectangular prism -/
def surface_area (l w h : ℝ) : ℝ := 2 * l * w + 2 * l * h + 2 * w * h

/-- Theorem: For a rectangular prism with length 10, width 4, and surface area 164, the height is 3 -/
theorem brick_height (l w sa : ℝ) (hl : l = 10) (hw : w = 4) (hsa : sa = 164) :
  ∃ h : ℝ, h = 3 ∧ surface_area l w h = sa := by
  sorry

end brick_height_l3483_348364


namespace magnitude_of_complex_expression_l3483_348376

theorem magnitude_of_complex_expression (z : ℂ) (h : z = 1 - Complex.I) : 
  Complex.abs (1 + Complex.I * z) = Real.sqrt 5 := by
sorry

end magnitude_of_complex_expression_l3483_348376


namespace dave_animal_books_l3483_348326

/-- The number of books about animals Dave bought -/
def num_animal_books : ℕ := sorry

/-- The number of books about outer space Dave bought -/
def num_space_books : ℕ := 6

/-- The number of books about trains Dave bought -/
def num_train_books : ℕ := 3

/-- The cost of each book in dollars -/
def cost_per_book : ℕ := 6

/-- The total amount Dave spent on books in dollars -/
def total_spent : ℕ := 102

theorem dave_animal_books : 
  num_animal_books = 8 ∧
  num_animal_books * cost_per_book + num_space_books * cost_per_book + num_train_books * cost_per_book = total_spent :=
sorry

end dave_animal_books_l3483_348326


namespace returning_players_l3483_348397

theorem returning_players (new_players : ℕ) (group_size : ℕ) (total_groups : ℕ) : 
  new_players = 48 → group_size = 6 → total_groups = 9 → 
  (total_groups * group_size) - new_players = 6 := by
  sorry

end returning_players_l3483_348397


namespace system_of_equations_solutions_l3483_348351

theorem system_of_equations_solutions :
  -- First system
  (∃ x y : ℝ, y = 3*x ∧ 7*x - 2*y = 2 → x = 2 ∧ y = 6) ∧
  -- Second system
  (∃ x y : ℝ, 2*x + 5*y = -4 ∧ 5*x + 2*y = 11 → x = 3 ∧ y = -2) := by
  sorry

end system_of_equations_solutions_l3483_348351
