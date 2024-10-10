import Mathlib

namespace square_plus_n_equals_n_times_n_plus_one_l63_6331

theorem square_plus_n_equals_n_times_n_plus_one (n : ℕ) : n^2 + n = n * (n + 1) := by
  sorry

end square_plus_n_equals_n_times_n_plus_one_l63_6331


namespace problem_solution_l63_6360

theorem problem_solution (x : ℝ) : (3 * x + 20 = (1 / 3) * (7 * x + 60)) → x = 0 := by
  sorry

end problem_solution_l63_6360


namespace angle_conversion_l63_6346

theorem angle_conversion (θ : Real) : 
  θ * (π / 180) = -10 * π + 7 * π / 4 → 
  ∃ (k : ℤ) (α : Real), 
    θ * (π / 180) = 2 * k * π + α ∧ 
    0 < α ∧ 
    α < 2 * π :=
by sorry

end angle_conversion_l63_6346


namespace pure_imaginary_quadratic_l63_6318

/-- A complex number is pure imaginary if its real part is zero and its imaginary part is non-zero -/
def IsPureImaginary (z : ℂ) : Prop :=
  z.re = 0 ∧ z.im ≠ 0

/-- The theorem statement -/
theorem pure_imaginary_quadratic (m : ℝ) :
  IsPureImaginary (Complex.mk (m^2 + m - 2) (m^2 - 1)) → m = -2 :=
by sorry

end pure_imaginary_quadratic_l63_6318


namespace initial_blocks_count_l63_6350

/-- The initial number of blocks Adolfo had -/
def initial_blocks : ℕ := sorry

/-- The number of blocks Adolfo added -/
def added_blocks : ℕ := 30

/-- The total number of blocks after adding -/
def total_blocks : ℕ := 65

/-- Theorem stating that the initial number of blocks is 35 -/
theorem initial_blocks_count : initial_blocks = 35 := by
  sorry

/-- Axiom representing the relationship between initial, added, and total blocks -/
axiom block_relationship : initial_blocks + added_blocks = total_blocks


end initial_blocks_count_l63_6350


namespace line_equation_proof_l63_6374

theorem line_equation_proof (m b k : ℝ) : 
  (∃ k, (k^2 + 4*k + 3 - (m*k + b) = 3 ∨ k^2 + 4*k + 3 - (m*k + b) = -3) ∧ 
        (∀ k', k' ≠ k → ¬(k'^2 + 4*k' + 3 - (m*k' + b) = 3 ∨ k'^2 + 4*k' + 3 - (m*k' + b) = -3))) →
  (m * 2 + b = 5) →
  (b ≠ 0) →
  (m = 9/2 ∧ b = -4) :=
by sorry

end line_equation_proof_l63_6374


namespace sqrt_37_range_l63_6325

theorem sqrt_37_range : 6 < Real.sqrt 37 ∧ Real.sqrt 37 < 7 := by
  sorry

end sqrt_37_range_l63_6325


namespace rays_remaining_nickels_l63_6344

/-- Calculates the number of nickels Ray has left after giving money to Peter and Randi -/
theorem rays_remaining_nickels 
  (initial_cents : ℕ) 
  (cents_to_peter : ℕ) 
  (nickel_value : ℕ) 
  (h1 : initial_cents = 95)
  (h2 : cents_to_peter = 25)
  (h3 : nickel_value = 5) :
  (initial_cents - cents_to_peter - 2 * cents_to_peter) / nickel_value = 4 :=
by sorry

end rays_remaining_nickels_l63_6344


namespace complex_expression_equals_zero_l63_6396

/-- The imaginary unit -/
noncomputable def i : ℂ := Complex.I

/-- The theorem stating that the complex expression equals 0 -/
theorem complex_expression_equals_zero : (1 + i) / (1 - i) + i ^ 3 = 0 := by sorry

end complex_expression_equals_zero_l63_6396


namespace subset_intersection_one_element_l63_6377

/-- Given n+1 distinct subsets of [n], each with exactly 3 elements,
    there must exist a pair of subsets whose intersection has exactly one element. -/
theorem subset_intersection_one_element
  (n : ℕ)
  (A : Fin (n + 1) → Finset (Fin n))
  (h_distinct : ∀ i j, i ≠ j → A i ≠ A j)
  (h_card : ∀ i, (A i).card = 3) :
  ∃ i j, i ≠ j ∧ (A i ∩ A j).card = 1 := by
  sorry

end subset_intersection_one_element_l63_6377


namespace representatives_selection_count_l63_6393

def num_female : ℕ := 3
def num_male : ℕ := 4
def num_representatives : ℕ := 3

theorem representatives_selection_count :
  (Finset.sum (Finset.range (num_representatives - 1)) (λ k =>
    Nat.choose num_female (k + 1) * Nat.choose num_male (num_representatives - k - 1)))
  = 30 := by sorry

end representatives_selection_count_l63_6393


namespace max_sides_in_subdivision_l63_6332

/-- 
Given a convex polygon with n sides and all its diagonals drawn,
the maximum number of sides a polygon in the subdivision can have is n.
-/
theorem max_sides_in_subdivision (n : ℕ) (h : n ≥ 3) :
  ∃ (max_sides : ℕ), max_sides = n ∧ 
  ∀ (subdivided_polygon_sides : ℕ), 
    subdivided_polygon_sides ≤ max_sides :=
by sorry

end max_sides_in_subdivision_l63_6332


namespace division_in_base5_l63_6398

/-- Converts a number from base 5 to base 10 -/
def base5ToBase10 (n : ℕ) : ℕ := sorry

/-- Converts a number from base 10 to base 5 -/
def base10ToBase5 (n : ℕ) : ℕ := sorry

/-- Performs division in base 5 -/
def divBase5 (a b : ℕ) : ℕ := 
  base10ToBase5 (base5ToBase10 a / base5ToBase10 b)

theorem division_in_base5 : divBase5 1302 23 = 30 := by sorry

end division_in_base5_l63_6398


namespace train_passengers_l63_6345

theorem train_passengers (initial_passengers : ℕ) : 
  (initial_passengers - 263 + 419 = 725) → initial_passengers = 569 := by
  sorry

end train_passengers_l63_6345


namespace complex_magnitude_equation_l63_6387

theorem complex_magnitude_equation (t : ℝ) : 
  t > 0 ∧ Complex.abs (-7 + t * Complex.I) = 15 → t = 4 * Real.sqrt 11 := by
  sorry

end complex_magnitude_equation_l63_6387


namespace simplify_fraction_product_l63_6311

theorem simplify_fraction_product : 8 * (18 / 5) * (-40 / 27) = -128 / 3 := by
  sorry

end simplify_fraction_product_l63_6311


namespace geometric_sequence_problem_l63_6305

theorem geometric_sequence_problem (a : ℕ → ℝ) (q : ℝ) (n : ℕ) :
  (∀ k, a (k + 1) = a k * q) →  -- geometric sequence condition
  q > 0 →  -- positive common ratio
  a 1 * a 2 * a 3 = 4 →  -- first condition
  a 4 * a 5 * a 6 = 8 →  -- second condition
  a n * a (n + 1) * a (n + 2) = 128 →  -- third condition
  n = 6 := by
sorry

end geometric_sequence_problem_l63_6305


namespace equation_solution_l63_6391

theorem equation_solution : ∃ x : ℝ, 24 * 2 - 6 = 3 * x + 6 ∧ x = 12 := by
  sorry

end equation_solution_l63_6391


namespace wolf_chase_deer_l63_6389

theorem wolf_chase_deer (t : ℕ) : t ≤ 28 ↔ ∀ (x y : ℝ), x > 0 → y > 0 → x * y > 0.78 * x * y * (1 + t / 100) := by
  sorry

end wolf_chase_deer_l63_6389


namespace f_is_quadratic_l63_6397

/-- A quadratic function is a function of the form f(x) = ax² + bx + c, where a ≠ 0 -/
def is_quadratic (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The function f(x) = 2x² - 2x + 1 -/
def f (x : ℝ) : ℝ := 2 * x^2 - 2 * x + 1

/-- Theorem: f(x) = 2x² - 2x + 1 is a quadratic function -/
theorem f_is_quadratic : is_quadratic f := by
  sorry

end f_is_quadratic_l63_6397


namespace remaining_card_is_seven_l63_6361

def cards : List Nat := [2, 3, 4, 5, 6, 7, 8, 9, 10]

def is_relatively_prime (a b : Nat) : Prop := Nat.gcd a b = 1

def is_consecutive (a b : Nat) : Prop := a.succ = b ∨ b.succ = a

def is_composite (n : Nat) : Prop := n > 3 ∧ ∃ m, 1 < m ∧ m < n ∧ n % m = 0

def is_multiple (a b : Nat) : Prop := ∃ k, k > 1 ∧ (a = k * b ∨ b = k * a)

theorem remaining_card_is_seven (A B C D : List Nat) : 
  A.length = 2 ∧ B.length = 2 ∧ C.length = 2 ∧ D.length = 2 →
  (∀ x ∈ A, x ∈ cards) ∧ (∀ x ∈ B, x ∈ cards) ∧ (∀ x ∈ C, x ∈ cards) ∧ (∀ x ∈ D, x ∈ cards) →
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D →
  (∀ a ∈ A, ∀ b ∈ A, a ≠ b → is_relatively_prime a b ∧ is_consecutive a b) →
  (∀ a ∈ B, ∀ b ∈ B, a ≠ b → ¬is_relatively_prime a b ∧ ¬is_multiple a b) →
  (∀ a ∈ C, ∀ b ∈ C, a ≠ b → is_composite a ∧ is_composite b ∧ is_relatively_prime a b) →
  (∀ a ∈ D, ∀ b ∈ D, a ≠ b → is_multiple a b ∧ ¬is_relatively_prime a b) →
  ∃! x, x ∈ cards ∧ x ∉ A ∧ x ∉ B ∧ x ∉ C ∧ x ∉ D ∧ x = 7 :=
by sorry

end remaining_card_is_seven_l63_6361


namespace equation_solution_l63_6352

theorem equation_solution (n : ℝ) : 
  let m := 5 * n + 5
  2 / (n + 2) + 3 / (n + 2) + m / (n + 2) = 5 := by
sorry

end equation_solution_l63_6352


namespace unbounded_function_identity_l63_6308

/-- A function f: ℤ → ℤ is unbounded if for any integer N, there exists an x such that |f(x)| > N -/
def Unbounded (f : ℤ → ℤ) : Prop :=
  ∀ N : ℤ, ∃ x : ℤ, |f x| > N

/-- The main theorem: if f is unbounded and satisfies the given condition, then f(x) = x for all x -/
theorem unbounded_function_identity
  (f : ℤ → ℤ)
  (h_unbounded : Unbounded f)
  (h_condition : ∀ x y : ℤ, (f (f x - y)) ∣ (x - f y)) :
  ∀ x : ℤ, f x = x :=
sorry

end unbounded_function_identity_l63_6308


namespace problem_statement_l63_6307

theorem problem_statement (x y : ℝ) 
  (h1 : x * y + x + y = 17)
  (h2 : x^2 * y + x * y^2 = 66) :
  x^4 + x^3 * y + x^2 * y^2 + x * y^3 + y^4 = 12499 := by
  sorry

end problem_statement_l63_6307


namespace man_gained_three_toys_cost_l63_6375

/-- The number of toys whose cost price the man gained -/
def toys_gained (num_sold : ℕ) (selling_price : ℕ) (cost_price : ℕ) : ℕ :=
  (selling_price - num_sold * cost_price) / cost_price

theorem man_gained_three_toys_cost :
  toys_gained 18 27300 1300 = 3 := by
  sorry

end man_gained_three_toys_cost_l63_6375


namespace travel_speed_problem_l63_6368

/-- Proves that given the conditions of the problem, the speeds of person A and person B are 4.5 km/h and 6 km/h respectively. -/
theorem travel_speed_problem (distance_A distance_B : ℝ) (speed_ratio : ℚ) (time_difference : ℝ) :
  distance_A = 6 →
  distance_B = 10 →
  speed_ratio = 3/4 →
  time_difference = 1/3 →
  ∃ (speed_A speed_B : ℝ),
    speed_A = 4.5 ∧
    speed_B = 6 ∧
    speed_A / speed_B = speed_ratio ∧
    distance_B / speed_B - distance_A / speed_A = time_difference :=
by sorry

end travel_speed_problem_l63_6368


namespace cubic_equation_solution_l63_6364

theorem cubic_equation_solution (a : ℝ) (h : 2 * a^3 + a^2 - 275 = 0) : a = 5 := by
  sorry

end cubic_equation_solution_l63_6364


namespace probability_second_class_first_given_first_class_second_is_two_fifths_l63_6309

/-- Represents the class of an item -/
inductive ItemClass
| FirstClass
| SecondClass

/-- Represents the box with items -/
structure Box where
  firstClassCount : ℕ
  secondClassCount : ℕ

/-- Represents the outcome of drawing two items -/
structure DrawOutcome where
  first : ItemClass
  second : ItemClass

def Box.totalCount (b : Box) : ℕ := b.firstClassCount + b.secondClassCount

/-- The probability of drawing a second-class item first, given that the second item is first-class -/
def probabilitySecondClassFirstGivenFirstClassSecond (b : Box) : ℚ :=
  let totalOutcomes := b.firstClassCount * (b.firstClassCount - 1) + b.secondClassCount * b.firstClassCount
  let favorableOutcomes := b.secondClassCount * b.firstClassCount
  favorableOutcomes / totalOutcomes

theorem probability_second_class_first_given_first_class_second_is_two_fifths :
  let b : Box := { firstClassCount := 4, secondClassCount := 2 }
  probabilitySecondClassFirstGivenFirstClassSecond b = 2 / 5 := by
  sorry

end probability_second_class_first_given_first_class_second_is_two_fifths_l63_6309


namespace isosceles_triangle_perimeter_isosceles_triangle_perimeter_proof_l63_6349

/-- An isosceles triangle with sides 4, 9, and 9 has a perimeter of 22. -/
theorem isosceles_triangle_perimeter : ℝ → ℝ → ℝ → ℝ → Prop :=
  fun a b c p => 
    (a = 4 ∧ b = 9 ∧ c = 9) →  -- Two sides are 9, one side is 4
    (a + b > c ∧ b + c > a ∧ c + a > b) →  -- Triangle inequality
    (b = c) →  -- Isosceles condition
    (a + b + c = p) →  -- Definition of perimeter
    p = 22  -- The perimeter is 22

-- The proof is omitted
theorem isosceles_triangle_perimeter_proof : 
  ∃ (a b c p : ℝ), isosceles_triangle_perimeter a b c p :=
sorry

end isosceles_triangle_perimeter_isosceles_triangle_perimeter_proof_l63_6349


namespace triangle_angle_value_l63_6324

theorem triangle_angle_value (A B C : Real) (a b c : Real) :
  0 < A ∧ A < π ∧
  0 < B ∧ B < π ∧
  0 < C ∧ C < π ∧
  A + B + C = π ∧
  Real.sqrt 3 * c * Real.sin A = a * Real.cos C →
  C = π / 6 := by
sorry

end triangle_angle_value_l63_6324


namespace quadratic_sum_of_squares_l63_6322

theorem quadratic_sum_of_squares (a b c : ℝ) : 
  a ≠ b ∧ b ≠ c ∧ a ≠ c →
  (∃! x : ℝ, x^2 + a*x + b = 0 ∧ x^2 + b*x + c = 0) →
  (∃! y : ℝ, y^2 + b*y + c = 0 ∧ y^2 + c*y + a = 0) →
  (∃! z : ℝ, z^2 + c*z + a = 0 ∧ z^2 + a*z + b = 0) →
  a^2 + b^2 + c^2 = 6 := by
sorry

end quadratic_sum_of_squares_l63_6322


namespace function_periodicity_l63_6358

/-- A function f: ℝ → ℝ satisfying f(x-1) + f(x+1) = √2 f(x) for all x ∈ ℝ is periodic with period 8. -/
theorem function_periodicity (f : ℝ → ℝ) 
  (h : ∀ x : ℝ, f (x - 1) + f (x + 1) = Real.sqrt 2 * f x) : 
  ∀ x : ℝ, f (x + 8) = f x := by
  sorry

end function_periodicity_l63_6358


namespace additional_plates_count_l63_6385

/-- The number of choices for the first letter in the original configuration -/
def original_first : Nat := 5

/-- The number of choices for the second letter in the original configuration -/
def original_second : Nat := 3

/-- The number of choices for the third letter in both original and new configurations -/
def third : Nat := 4

/-- The number of choices for the first letter in the new configuration -/
def new_first : Nat := 6

/-- The number of choices for the second letter in the new configuration -/
def new_second : Nat := 4

/-- The number of additional license plates that can be made -/
def additional_plates : Nat := new_first * new_second * third - original_first * original_second * third

theorem additional_plates_count : additional_plates = 36 := by
  sorry

end additional_plates_count_l63_6385


namespace bargain_bin_books_l63_6363

/-- The number of books initially in the bargain bin -/
def initial_books : ℝ := 41.0

/-- The number of books added in the first addition -/
def first_addition : ℝ := 33.0

/-- The number of books added in the second addition -/
def second_addition : ℝ := 2.0

/-- The total number of books after both additions -/
def total_books : ℝ := 76.0

/-- Theorem stating that the initial number of books plus the two additions equals the total -/
theorem bargain_bin_books : 
  initial_books + first_addition + second_addition = total_books := by
  sorry

end bargain_bin_books_l63_6363


namespace smallest_non_factor_product_l63_6356

theorem smallest_non_factor_product (m n : ℕ) : 
  m ≠ n → 
  m > 0 → 
  n > 0 → 
  m ∣ 48 → 
  n ∣ 48 → 
  ¬(m * n ∣ 48) → 
  (∀ a b : ℕ, a ≠ b → a > 0 → b > 0 → a ∣ 48 → b ∣ 48 → ¬(a * b ∣ 48) → m * n ≤ a * b) →
  m * n = 18 :=
by sorry

end smallest_non_factor_product_l63_6356


namespace sqrt_difference_inequality_l63_6399

theorem sqrt_difference_inequality (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  Real.sqrt a - Real.sqrt b < Real.sqrt (a - b) := by
  sorry

end sqrt_difference_inequality_l63_6399


namespace degree_to_radian_conversion_l63_6379

theorem degree_to_radian_conversion (π : Real) (h : π * 1 = 180) :
  -885 * (π / 180) = -59 / 12 * π := by
  sorry

end degree_to_radian_conversion_l63_6379


namespace definite_integral_x_plus_one_squared_ln_squared_l63_6320

theorem definite_integral_x_plus_one_squared_ln_squared :
  ∫ x in (0:ℝ)..2, (x + 1)^2 * (Real.log (x + 1))^2 = 9 * (Real.log 3)^2 - 6 * Real.log 3 + 79 / 27 := by
  sorry

end definite_integral_x_plus_one_squared_ln_squared_l63_6320


namespace calculation_proof_l63_6301

theorem calculation_proof (a b c : ℝ) (h1 : a = 3) (h2 : b = 2) (h3 : c = 4) :
  (c * (a^3 + b^3)) / (a^2 - a*b + b^2) = 20 := by
  sorry

end calculation_proof_l63_6301


namespace dealer_truck_sales_l63_6339

theorem dealer_truck_sales (total : ℕ) (car_truck_diff : ℕ) (trucks : ℕ) : 
  total = 69 → car_truck_diff = 27 → trucks + (trucks + car_truck_diff) = total → trucks = 21 :=
by
  sorry

end dealer_truck_sales_l63_6339


namespace expenditure_increase_percentage_l63_6343

theorem expenditure_increase_percentage
  (initial_expenditure : ℝ)
  (initial_savings : ℝ)
  (initial_income : ℝ)
  (h_ratio : initial_expenditure / initial_savings = 3 / 2)
  (h_income : initial_expenditure + initial_savings = initial_income)
  (h_new_income : ℝ)
  (h_income_increase : h_new_income = initial_income * 1.15)
  (h_new_savings : ℝ)
  (h_savings_increase : h_new_savings = initial_savings * 1.06)
  (h_new_expenditure : ℝ)
  (h_new_balance : h_new_expenditure + h_new_savings = h_new_income) :
  (h_new_expenditure - initial_expenditure) / initial_expenditure = 0.21 :=
sorry

end expenditure_increase_percentage_l63_6343


namespace total_oranges_l63_6313

def orange_groups : ℕ := 16
def oranges_per_group : ℕ := 24

theorem total_oranges : orange_groups * oranges_per_group = 384 := by
  sorry

end total_oranges_l63_6313


namespace fraction_subtraction_l63_6373

theorem fraction_subtraction : (18 : ℚ) / 42 - 3 / 8 = 3 / 56 := by
  sorry

end fraction_subtraction_l63_6373


namespace bus_travel_time_l63_6337

/-- Represents a time of day in hours and minutes -/
structure TimeOfDay where
  hours : Nat
  minutes : Nat
  valid : hours < 24 ∧ minutes < 60

/-- Calculates the difference in hours between two times -/
def timeDifference (t1 t2 : TimeOfDay) : Nat :=
  -- Implementation details omitted
  sorry

/-- Theorem: The time difference between 12:30 PM and 9:30 AM is 3 hours -/
theorem bus_travel_time :
  let departure : TimeOfDay := ⟨9, 30, sorry⟩
  let arrival : TimeOfDay := ⟨12, 30, sorry⟩
  timeDifference departure arrival = 3 := by
  sorry

end bus_travel_time_l63_6337


namespace repeating_decimal_sum_l63_6327

theorem repeating_decimal_sum (a b : ℕ+) (h1 : (35 : ℚ) / 99 = (a : ℚ) / b) 
  (h2 : Nat.gcd a.val b.val = 1) : a + b = 134 := by
  sorry

end repeating_decimal_sum_l63_6327


namespace five_T_three_equals_38_l63_6326

-- Define the new operation ⊤
def T (a b : ℤ) : ℤ := 4*a + 6*b

-- Theorem statement
theorem five_T_three_equals_38 : T 5 3 = 38 := by
  sorry

end five_T_three_equals_38_l63_6326


namespace ratio_yz_l63_6304

theorem ratio_yz (x y z : ℝ) 
  (h1 : (x + 53/18 * y - 143/9 * z) / z = 1)
  (h2 : (3/8 * x - 17/4 * y + z) / y = 1) :
  y / z = 352 / 305 := by
sorry

end ratio_yz_l63_6304


namespace hyperbola_properties_l63_6316

/-- Properties of the hyperbola x^2 - y^2 = 2 -/
theorem hyperbola_properties :
  let h : ℝ → ℝ → Prop := λ x y ↦ x^2 - y^2 = 2
  ∃ (a b c : ℝ),
    (∀ x y, h x y ↔ x^2 / a^2 - y^2 / b^2 = 1) ∧
    (2 * a = 2 * Real.sqrt 2) ∧
    (c^2 = a^2 + b^2) ∧
    (c / a = Real.sqrt 2) ∧
    (∀ x y, (y = x ∨ y = -x) → h x y) :=
by sorry

end hyperbola_properties_l63_6316


namespace seven_balance_removal_l63_6306

/-- A function that counts the number of sevens in even positions of a natural number -/
def countSevenEven (n : ℕ) : ℕ := sorry

/-- A function that counts the number of sevens in odd positions of a natural number -/
def countSevenOdd (n : ℕ) : ℕ := sorry

/-- A function that removes the i-th digit from a natural number -/
def removeDigit (n : ℕ) (i : ℕ) : ℕ := sorry

/-- A function that returns the number of digits in a natural number -/
def digitCount (n : ℕ) : ℕ := sorry

theorem seven_balance_removal (n : ℕ) (h : Odd (digitCount n)) :
  ∃ i : ℕ, i < digitCount n ∧ 
    countSevenEven (removeDigit n i) = countSevenOdd (removeDigit n i) := by
  sorry

end seven_balance_removal_l63_6306


namespace basketball_store_problem_l63_6333

/- Define the basketball types -/
inductive BasketballType
| A
| B

/- Define the purchase and selling prices -/
def purchase_price (t : BasketballType) : ℕ :=
  match t with
  | BasketballType.A => 80
  | BasketballType.B => 60

def selling_price (t : BasketballType) : ℕ :=
  match t with
  | BasketballType.A => 120
  | BasketballType.B => 90

/- Define the conditions -/
def condition1 : Prop :=
  20 * purchase_price BasketballType.A + 30 * purchase_price BasketballType.B = 3400

def condition2 : Prop :=
  30 * purchase_price BasketballType.A + 40 * purchase_price BasketballType.B = 4800

def jump_rope_cost : ℕ := 10

/- Define the theorem -/
theorem basketball_store_problem 
  (m n : ℕ) 
  (h1 : condition1)
  (h2 : condition2)
  (h3 : m * selling_price BasketballType.A + n * selling_price BasketballType.B = 5400) :
  (∃ (a b : ℕ), 
    (a * (selling_price BasketballType.A - purchase_price BasketballType.A - jump_rope_cost) + 
     b * (3 * (selling_price BasketballType.B - purchase_price BasketballType.B) - jump_rope_cost) = 600) ∧
    ((a = 12 ∧ b = 3) ∨ (a = 4 ∧ b = 6))) ∧
  (m * (selling_price BasketballType.A - purchase_price BasketballType.A) + 
   n * (selling_price BasketballType.B - purchase_price BasketballType.B) = 1800) :=
by sorry

end basketball_store_problem_l63_6333


namespace coin_overlap_area_l63_6355

theorem coin_overlap_area (square_side : ℝ) (triangle_leg : ℝ) (diamond_side : ℝ) (coin_diameter : ℝ) :
  square_side = 10 →
  triangle_leg = 3 →
  diamond_side = 3 * Real.sqrt 2 →
  coin_diameter = 2 →
  ∃ (overlap_area : ℝ),
    overlap_area = 52 ∧
    overlap_area = (36 + 16 * Real.sqrt 2 + 2 * Real.pi) / 
      ((square_side - coin_diameter) * (square_side - coin_diameter)) :=
by sorry

end coin_overlap_area_l63_6355


namespace apple_profit_calculation_l63_6335

/-- Profit percentage for the first half of apples -/
def P : ℝ := sorry

/-- Cost price of 1 kg of apples -/
def C : ℝ := sorry

theorem apple_profit_calculation :
  (50 * C + 50 * C * (P / 100) + 50 * C + 50 * C * (30 / 100) = 100 * C + 100 * C * (27.5 / 100)) →
  P = 25 := by
  sorry

end apple_profit_calculation_l63_6335


namespace expression_evaluation_l63_6380

theorem expression_evaluation (m : ℝ) (h : m = Real.tan (60 * π / 180) - 1) :
  (1 - 2 / (m + 1)) / ((m^2 - 2*m + 1) / (m^2 - m)) = (3 - Real.sqrt 3) / 3 := by
  sorry

end expression_evaluation_l63_6380


namespace work_completion_time_l63_6336

theorem work_completion_time (x : ℝ) : 
  x > 0 ∧ 
  5 * (1 / x + 1 / 20) = 1 - 0.41666666666666663 →
  x = 15 := by
sorry

end work_completion_time_l63_6336


namespace stream_speed_l63_6341

/-- 
Proves that given a boat with a speed of 51 kmph in still water, 
if the time taken to row upstream is twice the time taken to row downstream, 
then the speed of the stream is 17 kmph.
-/
theorem stream_speed (D : ℝ) (v : ℝ) : 
  (D / (51 - v) = 2 * (D / (51 + v))) → v = 17 := by
  sorry

end stream_speed_l63_6341


namespace complex_fraction_real_iff_m_eq_neg_one_l63_6310

/-- The complex number (m^2 + i) / (1 - mi) is real if and only if m = -1 -/
theorem complex_fraction_real_iff_m_eq_neg_one (m : ℝ) :
  (((m^2 : ℂ) + Complex.I) / (1 - m * Complex.I)).im = 0 ↔ m = -1 := by
  sorry

end complex_fraction_real_iff_m_eq_neg_one_l63_6310


namespace smallest_n_for_2015_divisibility_l63_6340

theorem smallest_n_for_2015_divisibility : ∃ (n : ℕ), n > 0 ∧ 
  (∀ (k : ℕ), k > 0 → (2^k - 1) % 2015 = 0 → k ≥ n) ∧ 
  (2^n - 1) % 2015 = 0 ∧ n = 60 := by
  sorry

end smallest_n_for_2015_divisibility_l63_6340


namespace care_package_weight_l63_6315

/-- Represents the weight of the care package contents -/
structure CarePackage where
  jellyBeans : ℝ
  brownies : ℝ
  gummyWorms : ℝ
  chocolateBars : ℝ
  popcorn : ℝ
  cookies : ℝ

/-- Calculates the total weight of the care package -/
def totalWeight (cp : CarePackage) : ℝ :=
  cp.jellyBeans + cp.brownies + cp.gummyWorms + cp.chocolateBars + cp.popcorn + cp.cookies

/-- The final weight of the care package after all modifications -/
def finalWeight (initialWeight : ℝ) : ℝ :=
  let weightAfterChocolate := initialWeight * 1.5
  let weightAfterPopcorn := weightAfterChocolate + 0.5
  let weightAfterCookies := weightAfterPopcorn * 2
  weightAfterCookies - 0.75

theorem care_package_weight :
  let initialPackage : CarePackage := {
    jellyBeans := 1.5,
    brownies := 0.5,
    gummyWorms := 2,
    chocolateBars := 0,
    popcorn := 0,
    cookies := 0
  }
  let initialWeight := totalWeight initialPackage
  finalWeight initialWeight = 12.25 := by
  sorry

end care_package_weight_l63_6315


namespace sector_area_l63_6321

/-- The area of a circular sector with given radius and arc length. -/
theorem sector_area (r : ℝ) (arc_length : ℝ) (h : r > 0) :
  let area := (1 / 2) * r * arc_length
  r = 15 ∧ arc_length = π / 3 → area = 5 * π / 2 := by
  sorry

end sector_area_l63_6321


namespace min_value_and_valid_a4_l63_6300

def is_valid_sequence (a : Fin 10 → ℕ) : Prop :=
  ∀ i j : Fin 10, i < j → a i < a j

def lcm_of_sequence (a : Fin 10 → ℕ) : ℕ :=
  Finset.lcm (Finset.range 10) (fun i => a i)

theorem min_value_and_valid_a4 (a : Fin 10 → ℕ) (h : is_valid_sequence a) :
  (∀ b : Fin 10 → ℕ, is_valid_sequence b → lcm_of_sequence a / a 3 ≤ lcm_of_sequence b / b 3) ∧
  (lcm_of_sequence a / a 0 = lcm_of_sequence a / a 3) →
  (lcm_of_sequence a / a 3 = 630) ∧
  (a 3 = 360 ∨ a 3 = 720 ∨ a 3 = 1080) ∧
  (1 ≤ a 3) ∧ (a 3 ≤ 1300) :=
by sorry

end min_value_and_valid_a4_l63_6300


namespace money_left_over_calculation_l63_6372

/-- The amount of money left over after purchasing bread and peanut butter -/
def money_left_over (bread_price : ℚ) (bread_quantity : ℕ) (peanut_butter_price : ℚ) (initial_amount : ℚ) : ℚ :=
  initial_amount - (bread_price * bread_quantity + peanut_butter_price)

/-- Theorem stating the amount of money left over in the given scenario -/
theorem money_left_over_calculation :
  let bread_price : ℚ := 9/4  -- $2.25 as a rational number
  let bread_quantity : ℕ := 3
  let peanut_butter_price : ℚ := 2
  let initial_amount : ℚ := 14
  money_left_over bread_price bread_quantity peanut_butter_price initial_amount = 21/4  -- $5.25 as a rational number
  := by sorry

end money_left_over_calculation_l63_6372


namespace toy_cost_proof_l63_6302

-- Define the number of toys
def num_toys : ℕ := 5

-- Define the discount rate (80% of original price)
def discount_rate : ℚ := 4/5

-- Define the total paid after discount
def total_paid : ℚ := 12

-- Define the cost per toy before discount
def cost_per_toy : ℚ := 3

-- Theorem statement
theorem toy_cost_proof :
  discount_rate * (num_toys : ℚ) * cost_per_toy = total_paid :=
sorry

end toy_cost_proof_l63_6302


namespace paul_collected_24_l63_6367

/-- Represents the number of seashells collected by each person -/
structure Seashells where
  henry : ℕ
  paul : ℕ
  leo : ℕ

/-- The initial state of seashell collection -/
def initial_collection : Seashells → Prop
  | s => s.henry = 11 ∧ s.henry + s.paul + s.leo = 59

/-- The state after Leo gave away a quarter of his seashells -/
def after_leo_gives : Seashells → Prop
  | s => s.henry + s.paul + (s.leo - s.leo / 4) = 53

/-- Theorem stating that Paul collected 24 seashells -/
theorem paul_collected_24 (s : Seashells) :
  initial_collection s → after_leo_gives s → s.paul = 24 := by
  sorry

#check paul_collected_24

end paul_collected_24_l63_6367


namespace no_real_j_for_single_solution_l63_6329

theorem no_real_j_for_single_solution :
  ¬ ∃ j : ℝ, ∃! x : ℝ, (2 * x + 7) * (x - 5) + 3 * x^2 = -20 + (j + 3) * x + 3 * x^2 :=
by sorry

end no_real_j_for_single_solution_l63_6329


namespace shopkeeper_decks_l63_6388

/-- The number of cards in a standard deck of playing cards -/
def standard_deck_size : ℕ := 52

/-- The total number of cards the shopkeeper has -/
def total_cards : ℕ := 319

/-- The number of additional cards the shopkeeper has -/
def additional_cards : ℕ := 7

/-- Theorem: The shopkeeper has 6 complete decks of playing cards -/
theorem shopkeeper_decks :
  (total_cards - additional_cards) / standard_deck_size = 6 := by
  sorry

end shopkeeper_decks_l63_6388


namespace ball_bounce_height_l63_6369

theorem ball_bounce_height (h₀ : ℝ) (r : ℝ) (h_target : ℝ) (k : ℕ) 
  (h_initial : h₀ = 800)
  (h_rebound : r = 1 / 2)
  (h_target_def : h_target = 2) :
  (∀ n : ℕ, n < k → h₀ * r ^ n ≥ h_target) ∧
  (h₀ * r ^ k < h_target) →
  k = 9 := by
sorry

end ball_bounce_height_l63_6369


namespace min_triangles_to_cover_l63_6303

theorem min_triangles_to_cover (large_side : ℝ) (small_side : ℝ) : 
  large_side = 8 → small_side = 2 → 
  (large_side^2 / small_side^2 : ℝ) = 16 := by
  sorry

end min_triangles_to_cover_l63_6303


namespace number_of_divisor_pairs_l63_6370

theorem number_of_divisor_pairs : ∃ (count : ℕ), count = 480 ∧
  count = (Finset.filter (fun p : ℕ × ℕ => 
    (p.1 * p.2 ∣ (2008 * 2009 * 2010)) ∧ 
    p.1 > 0 ∧ p.2 > 0
  ) (Finset.product (Finset.range (2008 * 2009 * 2010 + 1)) (Finset.range (2008 * 2009 * 2010 + 1)))).card :=
by
  sorry

#check number_of_divisor_pairs

end number_of_divisor_pairs_l63_6370


namespace factorization_identities_l63_6366

theorem factorization_identities (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  (a^4 - b^4 = (a - b) * (a + b) * (a^2 + b^2)) ∧
  (a + b - 2 * Real.sqrt (a * b) = (Real.sqrt a - Real.sqrt b)^2) := by
  sorry

end factorization_identities_l63_6366


namespace number_is_perfect_square_l63_6386

def N : ℕ := (10^1998 * ((10^1997 - 1) / 9)) + 2 * ((10^1998 - 1) / 9)

theorem number_is_perfect_square : 
  N = (10^1998 + 1)^2 := by
  sorry

end number_is_perfect_square_l63_6386


namespace line_parameterization_l63_6348

/-- Given a line y = (3/2)x - 25 parameterized by (x,y) = (f(t), 15t - 7),
    prove that f(t) = 10t + 12 is the correct parameterization for x. -/
theorem line_parameterization (f : ℝ → ℝ) :
  (∀ t : ℝ, (3/2) * f t - 25 = 15 * t - 7) →
  f = λ t => 10 * t + 12 := by
sorry

end line_parameterization_l63_6348


namespace solve_equation_l63_6395

-- Define the operation * based on the given condition
def star (a b : ℝ) : ℝ := a * (a * b - 7)

-- Theorem statement
theorem solve_equation : 
  (∃ x : ℝ, (star 3 x) = (star 2 (-8))) ∧ 
  (∀ x : ℝ, (star 3 x) = (star 2 (-8)) → x = -25/9) := by
sorry

end solve_equation_l63_6395


namespace trigonometric_identity_l63_6365

theorem trigonometric_identity (α : ℝ) :
  Real.cos (5 / 2 * Real.pi - 6 * α) * Real.sin (Real.pi - 2 * α)^3 -
  Real.cos (6 * α - Real.pi) * Real.sin (Real.pi / 2 - 2 * α)^3 =
  Real.cos (4 * α)^3 := by sorry

end trigonometric_identity_l63_6365


namespace parallelogram_area_l63_6353

theorem parallelogram_area (base height : ℝ) (h1 : base = 24) (h2 : height = 10) :
  base * height = 240 := by sorry

end parallelogram_area_l63_6353


namespace smallest_odd_with_five_prime_factors_l63_6384

def is_odd (n : ℕ) : Prop := ∃ k, n = 2*k + 1

def has_five_different_prime_factors (n : ℕ) : Prop :=
  ∃ p₁ p₂ p₃ p₄ p₅ : ℕ,
    Nat.Prime p₁ ∧ Nat.Prime p₂ ∧ Nat.Prime p₃ ∧ Nat.Prime p₄ ∧ Nat.Prime p₅ ∧
    p₁ < p₂ ∧ p₂ < p₃ ∧ p₃ < p₄ ∧ p₄ < p₅ ∧
    n = p₁ * p₂ * p₃ * p₄ * p₅

theorem smallest_odd_with_five_prime_factors :
  (is_odd 15015 ∧ has_five_different_prime_factors 15015) ∧
  ∀ m : ℕ, m < 15015 → ¬(is_odd m ∧ has_five_different_prime_factors m) :=
sorry

end smallest_odd_with_five_prime_factors_l63_6384


namespace mangoes_purchased_correct_mango_kg_l63_6378

theorem mangoes_purchased (grape_kg : ℕ) (grape_rate : ℕ) (mango_rate : ℕ) (total_paid : ℕ) : ℕ :=
  let mango_kg := (total_paid - grape_kg * grape_rate) / mango_rate
  mango_kg

theorem correct_mango_kg : mangoes_purchased 14 54 62 1376 = 10 := by
  sorry

end mangoes_purchased_correct_mango_kg_l63_6378


namespace unique_solution_implies_m_equals_3_l63_6392

/-- For a quadratic equation ax^2 + bx + c = 0 to have exactly one solution,
    its discriminant (b^2 - 4ac) must be zero. -/
def has_exactly_one_solution (a b c : ℝ) : Prop :=
  b^2 - 4*a*c = 0

/-- The quadratic equation 3x^2 - 6x + m = 0 has exactly one solution
    if and only if m = 3. -/
theorem unique_solution_implies_m_equals_3 :
  ∀ m : ℝ, has_exactly_one_solution 3 (-6) m ↔ m = 3 := by sorry

end unique_solution_implies_m_equals_3_l63_6392


namespace mary_walking_speed_l63_6381

/-- The walking speeds of Mary and Sharon, and the time and distance between them -/
structure WalkingProblem where
  mary_speed : ℝ
  sharon_speed : ℝ
  time : ℝ
  distance : ℝ

/-- The conditions of the problem -/
def problem_conditions (p : WalkingProblem) : Prop :=
  p.sharon_speed = 6 ∧ p.time = 0.3 ∧ p.distance = 3 ∧
  p.distance = p.mary_speed * p.time + p.sharon_speed * p.time

/-- The theorem stating Mary's walking speed -/
theorem mary_walking_speed (p : WalkingProblem) 
  (h : problem_conditions p) : p.mary_speed = 4 := by
  sorry

end mary_walking_speed_l63_6381


namespace alicia_tax_payment_l63_6312

/-- Calculates the total tax paid in cents per hour given an hourly wage and tax rates -/
def total_tax_cents (hourly_wage : ℝ) (local_tax_rate : ℝ) (state_tax_rate : ℝ) : ℝ :=
  hourly_wage * 100 * (local_tax_rate + state_tax_rate)

/-- Proves that Alicia's total tax paid is 62.5 cents per hour -/
theorem alicia_tax_payment :
  total_tax_cents 25 0.02 0.005 = 62.5 := by
  sorry

#eval total_tax_cents 25 0.02 0.005

end alicia_tax_payment_l63_6312


namespace age_difference_l63_6394

theorem age_difference (x y z : ℕ) (h : z = x - 18) : 
  (x + y) - (y + z) = 18 := by
  sorry

end age_difference_l63_6394


namespace probability_at_least_one_multiple_of_four_prob_at_least_one_multiple_of_four_l63_6347

/-- The probability of selecting at least one multiple of 4 when randomly choosing 3 integers from 1 to 50 (inclusive) -/
theorem probability_at_least_one_multiple_of_four : ℚ :=
  28051 / 50000

/-- The set of integers from 1 to 50 -/
def S : Set ℕ := {n | 1 ≤ n ∧ n ≤ 50}

/-- The number of elements in set S -/
def S_size : ℕ := 50

/-- The set of multiples of 4 in S -/
def M : Set ℕ := {n ∈ S | n % 4 = 0}

/-- The number of elements in set M -/
def M_size : ℕ := 12

/-- The probability of selecting a number that is not a multiple of 4 -/
def p_not_multiple_of_four : ℚ := (S_size - M_size) / S_size

/-- The probability of selecting three numbers, none of which are multiples of 4 -/
def p_none_multiple_of_four : ℚ := p_not_multiple_of_four ^ 3

/-- Theorem: The probability of selecting at least one multiple of 4 when randomly choosing 3 integers from 1 to 50 (inclusive) is 28051/50000 -/
theorem prob_at_least_one_multiple_of_four :
  1 - p_none_multiple_of_four = probability_at_least_one_multiple_of_four :=
by sorry

end probability_at_least_one_multiple_of_four_prob_at_least_one_multiple_of_four_l63_6347


namespace estimate_fish_population_l63_6319

/-- Estimates the total number of fish in a pond using the capture-recapture method. -/
theorem estimate_fish_population (initial_catch : ℕ) (second_catch : ℕ) (marked_recaught : ℕ) :
  initial_catch = 60 →
  second_catch = 80 →
  marked_recaught = 5 →
  (initial_catch * second_catch) / marked_recaught = 960 :=
by
  sorry

end estimate_fish_population_l63_6319


namespace minimum_guests_l63_6317

theorem minimum_guests (total_food : ℕ) (max_per_guest : ℕ) (h1 : total_food = 323) (h2 : max_per_guest = 2) :
  ∃ min_guests : ℕ, min_guests = 162 ∧ min_guests * max_per_guest ≥ total_food ∧
  ∀ n : ℕ, n * max_per_guest ≥ total_food → n ≥ min_guests :=
by sorry

end minimum_guests_l63_6317


namespace other_root_of_quadratic_l63_6371

theorem other_root_of_quadratic (a : ℝ) : 
  (3 : ℝ) ^ 2 - a * 3 - 2 * a = 0 → 
  ((-6 / 5 : ℝ) ^ 2 - a * (-6 / 5) - 2 * a = 0) ∧ 
  (3 + (-6 / 5) : ℝ) = a ∧ 
  (3 * (-6 / 5) : ℝ) = -2 * a := by
sorry

end other_root_of_quadratic_l63_6371


namespace business_partnership_timing_l63_6323

/-- Proves that B joined the business 8 months after A started, given the conditions of the problem -/
theorem business_partnership_timing (a_initial_capital b_capital : ℕ) (x : ℕ) : 
  a_initial_capital = 3500 →
  b_capital = 15750 →
  (a_initial_capital * 12) / (b_capital * (12 - x)) = 2 / 3 →
  x = 8 := by
  sorry

end business_partnership_timing_l63_6323


namespace scientific_notation_equality_l63_6382

theorem scientific_notation_equality (n : ℝ) (h : n = 58000) : n = 5.8 * (10 ^ 4) := by
  sorry

end scientific_notation_equality_l63_6382


namespace peru_tst_imo_2006_q1_l63_6334

theorem peru_tst_imo_2006_q1 : 
  {(x, y, z) : ℕ × ℕ × ℕ | 
    x > 0 ∧ y > 0 ∧ z > 0 ∧ 
    ∃ k : ℕ, (Real.sqrt (2006 / (x + y : ℝ)) + 
               Real.sqrt (2006 / (y + z : ℝ)) + 
               Real.sqrt (2006 / (z + x : ℝ))) = k}
  = {(2006, 2006, 2006), (1003, 1003, 7021), (9027, 9027, 9027)} := by
  sorry


end peru_tst_imo_2006_q1_l63_6334


namespace arithmetic_sequence_property_l63_6338

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_property
  (a : ℕ → ℝ)
  (h_arithmetic : arithmetic_sequence a)
  (h_sum : a 3 + a 5 + a 7 + a 9 + a 11 = 100) :
  3 * a 9 - a 13 = 50 := by
sorry

end arithmetic_sequence_property_l63_6338


namespace unique_solution_logarithmic_equation_l63_6314

theorem unique_solution_logarithmic_equation :
  ∃! x : ℝ, x > 0 ∧ x^(Real.log x / Real.log 10) = x^2 / 10 :=
by sorry

end unique_solution_logarithmic_equation_l63_6314


namespace raft_drift_theorem_l63_6330

/-- The time for a raft to drift between two villages -/
def raft_drift_time (distance : ℝ) (steamboat_time : ℝ) (motorboat_time : ℝ) : ℝ :=
  90

/-- Theorem: The raft drift time is 90 minutes given the conditions -/
theorem raft_drift_theorem (distance : ℝ) (steamboat_time : ℝ) (motorboat_time : ℝ) 
  (h1 : distance = 1)
  (h2 : steamboat_time = 1)
  (h3 : motorboat_time = 45 / 60)
  (h4 : ∃ (steamboat_speed : ℝ), 
    motorboat_time = distance / (2 * steamboat_speed + (distance / steamboat_time - steamboat_speed))) :
  raft_drift_time distance steamboat_time motorboat_time = 90 := by
  sorry

#check raft_drift_theorem

end raft_drift_theorem_l63_6330


namespace train_passing_time_l63_6351

/-- The time it takes for a train to pass a person moving in the opposite direction --/
theorem train_passing_time (train_length : ℝ) (train_speed : ℝ) (person_speed : ℝ) :
  train_length = 110 →
  train_speed = 84 * (5 / 18) →
  person_speed = 6 * (5 / 18) →
  (train_length / (train_speed + person_speed)) = 4.4 := by
  sorry

end train_passing_time_l63_6351


namespace lumberjack_trees_l63_6390

theorem lumberjack_trees (logs_per_tree : ℕ) (firewood_per_log : ℕ) (total_firewood : ℕ) :
  logs_per_tree = 4 →
  firewood_per_log = 5 →
  total_firewood = 500 →
  total_firewood / (logs_per_tree * firewood_per_log) = 25 :=
by sorry

end lumberjack_trees_l63_6390


namespace correct_grade12_sample_l63_6354

/-- Calculates the number of students to be drawn from grade 12 in a stratified sample -/
def students_from_grade12 (total_students : ℕ) (grade10_students : ℕ) (grade11_students : ℕ) (sample_size : ℕ) : ℕ :=
  let grade12_students := total_students - (grade10_students + grade11_students)
  (grade12_students * sample_size) / total_students

/-- Theorem stating the correct number of students to be drawn from grade 12 -/
theorem correct_grade12_sample : 
  students_from_grade12 2400 820 780 120 = 40 := by
  sorry

end correct_grade12_sample_l63_6354


namespace basketball_scores_l63_6328

/-- Represents the scores of a team over four quarters -/
structure TeamScores :=
  (q1 q2 q3 q4 : ℚ)

/-- Checks if a sequence of four numbers is geometric -/
def isGeometric (s : TeamScores) : Prop :=
  ∃ r : ℚ, r ≠ 0 ∧ s.q2 = s.q1 * r ∧ s.q3 = s.q2 * r ∧ s.q4 = s.q3 * r

/-- Checks if a sequence of four numbers is arithmetic -/
def isArithmetic (s : TeamScores) : Prop :=
  ∃ d : ℚ, s.q2 = s.q1 + d ∧ s.q3 = s.q2 + d ∧ s.q4 = s.q3 + d

/-- Calculates the total score for a team -/
def totalScore (s : TeamScores) : ℚ :=
  s.q1 + s.q2 + s.q3 + s.q4

/-- Calculates the first half score for a team -/
def firstHalfScore (s : TeamScores) : ℚ :=
  s.q1 + s.q2

theorem basketball_scores (teamA teamB : TeamScores) :
  teamA.q1 = teamB.q1 →  -- Tied at the end of first quarter
  isGeometric teamA →    -- Team A's scores form a geometric sequence
  isArithmetic teamB →   -- Team B's scores form an arithmetic sequence
  totalScore teamA = totalScore teamB + 2 →  -- Team A won by two points
  totalScore teamA ≤ 80 →  -- Team A's total score is not more than 80
  totalScore teamB ≤ 80 →  -- Team B's total score is not more than 80
  firstHalfScore teamA + firstHalfScore teamB = 41 :=
by sorry

end basketball_scores_l63_6328


namespace vector_sum_proof_l63_6342

def a : ℝ × ℝ := (2, 8)
def b : ℝ × ℝ := (-7, 2)

theorem vector_sum_proof : a + 2 • b = (-12, 12) := by
  sorry

end vector_sum_proof_l63_6342


namespace smallest_n_correct_l63_6383

/-- The function f as defined in the problem -/
def f : ℕ → ℤ
| 0 => 0
| n + 1 => -f (n / 3) - 3 * ((n + 1) % 3)

/-- The smallest non-negative integer n such that f(n) = 2010 -/
def smallest_n : ℕ := 3 * (3^2010 - 1) / 4

/-- Theorem stating that f(smallest_n) = 2010 and smallest_n is indeed the smallest such n -/
theorem smallest_n_correct :
  f smallest_n = 2010 ∧ ∀ m : ℕ, m < smallest_n → f m ≠ 2010 :=
sorry

end smallest_n_correct_l63_6383


namespace correct_observation_value_l63_6362

theorem correct_observation_value 
  (n : ℕ) 
  (initial_mean : ℝ) 
  (wrong_value : ℝ) 
  (corrected_mean : ℝ) 
  (h1 : n = 50) 
  (h2 : initial_mean = 36) 
  (h3 : wrong_value = 23) 
  (h4 : corrected_mean = 36.5) : 
  ∃ (correct_value : ℝ), 
    (n : ℝ) * corrected_mean = (n : ℝ) * initial_mean - wrong_value + correct_value ∧ 
    correct_value = 48 := by
  sorry

end correct_observation_value_l63_6362


namespace function_equality_implies_a_equals_two_l63_6376

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x > 0 then a^x else 1 - x

theorem function_equality_implies_a_equals_two (a : ℝ) :
  f a 1 = f a (-1) → a = 2 := by
  sorry

end function_equality_implies_a_equals_two_l63_6376


namespace gcd_120_4_l63_6357

/-- The greatest common divisor of 120 and 4 is 4, given they share exactly three positive divisors -/
theorem gcd_120_4 : 
  (∃ (S : Finset Nat), S = {d : Nat | d ∣ 120 ∧ d ∣ 4} ∧ Finset.card S = 3) →
  Nat.gcd 120 4 = 4 := by
sorry

end gcd_120_4_l63_6357


namespace zeros_after_decimal_of_fraction_l63_6359

/-- The number of zeros after the decimal point in the decimal representation of 1/(100^15) -/
def zeros_after_decimal : ℕ := 30

/-- The fraction we're considering -/
def fraction : ℚ := 1 / (100 ^ 15)

theorem zeros_after_decimal_of_fraction :
  (∃ (x : ℚ), x * 10^zeros_after_decimal = fraction ∧ 
   x ≥ 1/10 ∧ x < 1) ∧
  (∀ (n : ℕ), n < zeros_after_decimal → 
   ∃ (y : ℚ), y * 10^n = fraction ∧ y < 1/10) :=
sorry

end zeros_after_decimal_of_fraction_l63_6359
