import Mathlib

namespace NUMINAMATH_CALUDE_petya_vasya_meeting_l3667_366702

/-- The number of street lamps along the alley -/
def num_lamps : ℕ := 100

/-- The lamp number where Petya is observed -/
def petya_observed : ℕ := 22

/-- The lamp number where Vasya is observed -/
def vasya_observed : ℕ := 88

/-- Calculates the meeting point of Petya and Vasya -/
def meeting_point : ℕ := 64

/-- Theorem stating that Petya and Vasya meet at the calculated meeting point -/
theorem petya_vasya_meeting :
  ∀ (petya_speed vasya_speed : ℚ),
  petya_speed > 0 ∧ vasya_speed > 0 →
  (petya_speed * (meeting_point - 1) = vasya_speed * (num_lamps - meeting_point)) ∧
  (petya_speed * (petya_observed - 1) = vasya_speed * (num_lamps - vasya_observed)) :=
by sorry

#check petya_vasya_meeting

end NUMINAMATH_CALUDE_petya_vasya_meeting_l3667_366702


namespace NUMINAMATH_CALUDE_flour_bags_comparison_l3667_366757

theorem flour_bags_comparison (W : ℝ) : 
  (W > 0) →
  let remaining_first := W - W / 3
  let remaining_second := W - 1000 / 3
  (W > 1000 → remaining_second > remaining_first) ∧
  (W = 1000 → remaining_second = remaining_first) ∧
  (W < 1000 → remaining_first > remaining_second) :=
by sorry

end NUMINAMATH_CALUDE_flour_bags_comparison_l3667_366757


namespace NUMINAMATH_CALUDE_tenth_term_of_specific_arithmetic_sequence_l3667_366763

-- Define an arithmetic sequence
def arithmetic_sequence (a : ℝ) (d : ℝ) (n : ℕ) : ℝ := a + (n - 1) * d

-- State the theorem
theorem tenth_term_of_specific_arithmetic_sequence : 
  ∃ (a d : ℝ), 
    arithmetic_sequence a d 3 = 10 ∧ 
    arithmetic_sequence a d 6 = 16 ∧ 
    arithmetic_sequence a d 10 = 24 := by
  sorry

end NUMINAMATH_CALUDE_tenth_term_of_specific_arithmetic_sequence_l3667_366763


namespace NUMINAMATH_CALUDE_tea_mixture_theorem_l3667_366714

/-- Given two types of tea with prices per kg and their mixing ratio, 
    calculate the price per kg of the resulting mixture -/
def tea_mixture_price (price1 price2 : ℚ) (ratio : ℚ) : ℚ :=
  (price1 * ratio + price2 * ratio) / (ratio + ratio)

/-- Theorem: The price of a 1:1 mixture of tea costing Rs. 62/kg and Rs. 72/kg is Rs. 67/kg -/
theorem tea_mixture_theorem :
  tea_mixture_price 62 72 1 = 67 := by
  sorry

end NUMINAMATH_CALUDE_tea_mixture_theorem_l3667_366714


namespace NUMINAMATH_CALUDE_mistaken_multiplication_l3667_366718

def repeating_decimal (c d : ℕ) : ℚ :=
  1 + (c * 10 + d : ℚ) / 99

theorem mistaken_multiplication (c d : ℕ) 
  (h1 : c < 10) (h2 : d < 10) :
  90 * repeating_decimal c d - 90 * (1 + (c * 10 + d : ℚ) / 100) = 0.9 → 
  c = 9 ∧ d = 9 := by
  sorry

end NUMINAMATH_CALUDE_mistaken_multiplication_l3667_366718


namespace NUMINAMATH_CALUDE_parabola_reflection_l3667_366729

/-- A parabola is a function of the form y = a(x - h)^2 + k, where (h, k) is the vertex -/
structure Parabola where
  a : ℝ
  h : ℝ
  k : ℝ

/-- Reflection of a parabola along the y-axis -/
def reflect_y_axis (p : Parabola) : Parabola :=
  { a := p.a, h := -p.h, k := p.k }

theorem parabola_reflection :
  let original := Parabola.mk 2 1 (-4)
  let reflected := reflect_y_axis original
  reflected = Parabola.mk 2 (-1) (-4) := by sorry

end NUMINAMATH_CALUDE_parabola_reflection_l3667_366729


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l3667_366712

def M : Set ℝ := {x | 0 < x ∧ x ≤ 2}
def N : Set ℝ := {x | 0 ≤ x ∧ x ≤ 2}

theorem sufficient_not_necessary : 
  (∀ a, a ∈ M → a ∈ N) ∧ (∃ a, a ∈ N ∧ a ∉ M) := by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l3667_366712


namespace NUMINAMATH_CALUDE_complex_square_in_fourth_quadrant_l3667_366730

theorem complex_square_in_fourth_quadrant (z : ℂ) :
  (z.re > 0 ∧ z.im < 0) →  -- z is in the fourth quadrant
  z^2 - 2*z + 2 = 0 →      -- z satisfies the given equation
  z^2 = -2*Complex.I :=    -- conclusion: z^2 = -2i
by
  sorry

end NUMINAMATH_CALUDE_complex_square_in_fourth_quadrant_l3667_366730


namespace NUMINAMATH_CALUDE_max_sum_of_factors_l3667_366765

theorem max_sum_of_factors (diamond delta : ℕ) : 
  diamond * delta = 36 → (∀ x y : ℕ, x * y = 36 → x + y ≤ diamond + delta) → diamond + delta = 37 := by
  sorry

end NUMINAMATH_CALUDE_max_sum_of_factors_l3667_366765


namespace NUMINAMATH_CALUDE_solve_linear_equation_l3667_366755

theorem solve_linear_equation (x : ℝ) (h : 4 * x + 12 = 48) : x = 9 := by
  sorry

end NUMINAMATH_CALUDE_solve_linear_equation_l3667_366755


namespace NUMINAMATH_CALUDE_total_pizza_slices_l3667_366709

theorem total_pizza_slices :
  let num_pizzas : ℕ := 21
  let slices_per_pizza : ℕ := 8
  num_pizzas * slices_per_pizza = 168 := by
  sorry

end NUMINAMATH_CALUDE_total_pizza_slices_l3667_366709


namespace NUMINAMATH_CALUDE_chess_game_probability_l3667_366760

theorem chess_game_probability (p_not_losing p_draw : ℝ) 
  (h1 : p_not_losing = 0.8)
  (h2 : p_draw = 0.5) :
  p_not_losing - p_draw = 0.3 := by
sorry

end NUMINAMATH_CALUDE_chess_game_probability_l3667_366760


namespace NUMINAMATH_CALUDE_dance_school_relation_l3667_366727

/-- Represents the dance school scenario -/
structure DanceSchool where
  b : ℕ  -- number of boys
  g : ℕ  -- number of girls

/-- The number of girls the nth boy dances with -/
def girls_danced (n : ℕ) : ℕ := 2 * n + 4

/-- The dance school satisfies the given conditions -/
def valid_dance_school (ds : DanceSchool) : Prop :=
  ∀ n, n ≥ 1 → n ≤ ds.b → girls_danced n ≤ ds.g ∧
  girls_danced ds.b = ds.g

theorem dance_school_relation (ds : DanceSchool) 
  (h : valid_dance_school ds) : 
  ds.b = (ds.g - 4) / 2 :=
sorry

end NUMINAMATH_CALUDE_dance_school_relation_l3667_366727


namespace NUMINAMATH_CALUDE_ariel_current_age_l3667_366733

/-- Calculates Ariel's current age based on given information -/
theorem ariel_current_age :
  let birth_year : ℕ := 1992
  let fencing_start_year : ℕ := 2006
  let years_fencing : ℕ := 16
  let current_year : ℕ := fencing_start_year + years_fencing
  let current_age : ℕ := current_year - birth_year
  current_age = 30 := by sorry

end NUMINAMATH_CALUDE_ariel_current_age_l3667_366733


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l3667_366786

/-- A quadratic function with a non-zero leading coefficient -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  a_nonzero : a ≠ 0

/-- The function value at a given point -/
def QuadraticFunction.value (f : QuadraticFunction) (x : ℝ) : ℝ :=
  f.a * x^2 + f.b * x + f.c

/-- The derivative of the quadratic function -/
def QuadraticFunction.derivative (f : QuadraticFunction) (x : ℝ) : ℝ :=
  2 * f.a * x + f.b

theorem quadratic_function_properties (f : QuadraticFunction) 
  (h1 : f.derivative 1 = 0)
  (h2 : f.value 1 = 3)
  (h3 : f.value 2 = 8) :
  f.value (-1) ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l3667_366786


namespace NUMINAMATH_CALUDE_min_consecutive_even_numbers_divisible_by_384_l3667_366791

-- Define a function that checks if a number is even
def isEven (n : ℕ) : Prop := ∃ k, n = 2 * k

-- Define a function that generates a list of consecutive even numbers
def consecutiveEvenNumbers (start : ℕ) (count : ℕ) : List ℕ :=
  List.range count |>.map (λ i => start + 2 * i)

-- Define a function that calculates the product of a list of numbers
def productOfList (l : List ℕ) : ℕ :=
  l.foldl (·*·) 1

-- The main theorem
theorem min_consecutive_even_numbers_divisible_by_384 :
  ∀ n : ℕ, n ≥ 7 →
    ∀ start : ℕ, isEven start →
      384 ∣ productOfList (consecutiveEvenNumbers start n) ∧
      ∀ m : ℕ, m < 7 →
        ∃ s : ℕ, isEven s ∧ ¬(384 ∣ productOfList (consecutiveEvenNumbers s m)) :=
by sorry


end NUMINAMATH_CALUDE_min_consecutive_even_numbers_divisible_by_384_l3667_366791


namespace NUMINAMATH_CALUDE_p_money_theorem_l3667_366705

theorem p_money_theorem (p q r : ℚ) : 
  (p = (1/6 * p + 1/6 * p) + 32) → p = 48 := by
  sorry

end NUMINAMATH_CALUDE_p_money_theorem_l3667_366705


namespace NUMINAMATH_CALUDE_greatest_three_digit_multiple_of_17_l3667_366779

theorem greatest_three_digit_multiple_of_17 : ∃ n : ℕ, 
  n = 986 ∧ 
  100 ≤ n ∧ n ≤ 999 ∧ 
  17 ∣ n ∧
  ∀ m : ℕ, 100 ≤ m ∧ m ≤ 999 ∧ 17 ∣ m → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_greatest_three_digit_multiple_of_17_l3667_366779


namespace NUMINAMATH_CALUDE_gcd_37_power_plus_one_l3667_366731

theorem gcd_37_power_plus_one (h : Prime 37) : 
  Nat.gcd (37^11 + 1) (37^11 + 37^3 + 1) = 1 := by
sorry

end NUMINAMATH_CALUDE_gcd_37_power_plus_one_l3667_366731


namespace NUMINAMATH_CALUDE_min_value_of_function_l3667_366770

theorem min_value_of_function (x : ℝ) (h : x > 0) : 
  ∃ (y : ℝ), y = x + 4 / x^2 ∧ ∀ (z : ℝ), z = x + 4 / x^2 → y ≤ z ∧ y = 3 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_function_l3667_366770


namespace NUMINAMATH_CALUDE_solution_of_linear_equation_l3667_366776

theorem solution_of_linear_equation :
  let f : ℝ → ℝ := λ x => x + 2
  f (-2) = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_solution_of_linear_equation_l3667_366776


namespace NUMINAMATH_CALUDE_probability_three_common_books_l3667_366700

def total_books : ℕ := 12
def books_selected : ℕ := 6
def books_in_common : ℕ := 3

def probability_common_books : ℚ :=
  (Nat.choose total_books books_in_common * 
   Nat.choose (total_books - books_in_common) (books_selected - books_in_common) * 
   Nat.choose (total_books - books_selected) (books_selected - books_in_common)) /
  (Nat.choose total_books books_selected * Nat.choose total_books books_selected)

theorem probability_three_common_books :
  probability_common_books = 140 / 323 := by sorry

end NUMINAMATH_CALUDE_probability_three_common_books_l3667_366700


namespace NUMINAMATH_CALUDE_smallest_m_satisfying_conditions_l3667_366740

theorem smallest_m_satisfying_conditions : ∃ m : ℕ,
  (100 ≤ m ∧ m < 1000) ∧  -- m is a three-digit number
  (∃ k : ℤ, m + 7 = 9 * k) ∧  -- m + 7 is divisible by 9
  (∃ l : ℤ, m - 9 = 7 * l) ∧  -- m - 9 is divisible by 7
  (∀ n : ℕ, (100 ≤ n ∧ n < 1000 ∧
    (∃ p : ℤ, n + 7 = 9 * p) ∧
    (∃ q : ℤ, n - 9 = 7 * q)) → m ≤ n) ∧
  m = 128 :=
by sorry

end NUMINAMATH_CALUDE_smallest_m_satisfying_conditions_l3667_366740


namespace NUMINAMATH_CALUDE_b_joined_after_six_months_l3667_366713

/-- Represents the number of months after A started the business that B joined --/
def months_before_b_joined : ℕ → Prop := fun x =>
  let a_investment := 3500 * 12
  let b_investment := 10500 * (12 - x)
  (a_investment : ℚ) / b_investment = 2 / 3

/-- The theorem states that B joined 6 months after A started the business --/
theorem b_joined_after_six_months :
  months_before_b_joined 6 := by sorry

end NUMINAMATH_CALUDE_b_joined_after_six_months_l3667_366713


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l3667_366778

def M : Set ℝ := {x | x^2 - 3*x = 0}
def N : Set ℝ := {x | x > -1}

theorem intersection_of_M_and_N : M ∩ N = {0, 3} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l3667_366778


namespace NUMINAMATH_CALUDE_stock_price_calculation_l3667_366796

theorem stock_price_calculation (initial_price : ℝ) : 
  let first_year_increase : ℝ := 1.5
  let second_year_decrease : ℝ := 0.3
  let third_year_increase : ℝ := 0.2
  let price_after_first_year : ℝ := initial_price * (1 + first_year_increase)
  let price_after_second_year : ℝ := price_after_first_year * (1 - second_year_decrease)
  let final_price : ℝ := price_after_second_year * (1 + third_year_increase)
  initial_price = 120 → final_price = 252 := by
sorry

end NUMINAMATH_CALUDE_stock_price_calculation_l3667_366796


namespace NUMINAMATH_CALUDE_triangle_perimeter_l3667_366762

/-- A triangle with side lengths x, x+1, and x-1 has a perimeter of 21 if and only if x = 7 -/
theorem triangle_perimeter (x : ℝ) : 
  x > 0 ∧ x + 1 > 0 ∧ x - 1 > 0 → 
  (x + (x + 1) + (x - 1) = 21 ↔ x = 7) := by
sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l3667_366762


namespace NUMINAMATH_CALUDE_surface_sum_bounds_l3667_366781

/-- Represents a small cube with numbers on its faces -/
structure SmallCube :=
  (faces : Fin 6 → Nat)
  (opposite_sum_seven : ∀ i : Fin 3, faces i + faces (i + 3) = 7)
  (valid_numbers : ∀ i : Fin 6, 1 ≤ faces i ∧ faces i ≤ 6)

/-- Represents the larger cube assembled from 64 small cubes -/
structure LargeCube :=
  (small_cubes : Fin 64 → SmallCube)

/-- The sum of visible numbers on the surface of the larger cube -/
def surface_sum (lc : LargeCube) : Nat :=
  sorry

theorem surface_sum_bounds (lc : LargeCube) :
  144 ≤ surface_sum lc ∧ surface_sum lc ≤ 528 := by
  sorry

end NUMINAMATH_CALUDE_surface_sum_bounds_l3667_366781


namespace NUMINAMATH_CALUDE_computer_cost_l3667_366748

theorem computer_cost (total_budget : ℕ) (tv_cost : ℕ) (fridge_computer_diff : ℕ) 
  (h1 : total_budget = 1600)
  (h2 : tv_cost = 600)
  (h3 : fridge_computer_diff = 500) : 
  ∃ (computer_cost : ℕ), 
    computer_cost + tv_cost + (computer_cost + fridge_computer_diff) = total_budget ∧ 
    computer_cost = 250 := by
  sorry

end NUMINAMATH_CALUDE_computer_cost_l3667_366748


namespace NUMINAMATH_CALUDE_largest_last_digit_l3667_366725

def is_valid_series (s : List Nat) : Prop :=
  s.length = 2023 ∧
  s.head? = some 1 ∧
  ∀ i, i < s.length - 1 →
    let two_digit := s[i]! * 10 + s[i+1]!
    two_digit % 17 = 0 ∨ two_digit % 29 = 0 ∨ two_digit % 23 = 0

theorem largest_last_digit (s : List Nat) (h : is_valid_series s) :
  s.getLast? = some 2 := by
  sorry

end NUMINAMATH_CALUDE_largest_last_digit_l3667_366725


namespace NUMINAMATH_CALUDE_reciprocal_sum_theorem_l3667_366754

theorem reciprocal_sum_theorem (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : x + y = 5 * x * y) : 1 / x + 1 / y = 5 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_sum_theorem_l3667_366754


namespace NUMINAMATH_CALUDE_card_sum_perfect_square_l3667_366784

theorem card_sum_perfect_square (n : ℕ) (h : n ≥ 100) :
  ∃ a b c : ℕ, n ≤ a ∧ a < b ∧ b < c ∧ c ≤ 2*n ∧
  ∃ x y z : ℕ, a + b = x^2 ∧ b + c = y^2 ∧ c + a = z^2 :=
by sorry

end NUMINAMATH_CALUDE_card_sum_perfect_square_l3667_366784


namespace NUMINAMATH_CALUDE_number_equation_solution_l3667_366721

theorem number_equation_solution : ∃ x : ℝ, 
  (0.6667 * x + 1 = 0.75 * x) ∧ 
  (abs (x - 12) < 0.01) := by
  sorry

end NUMINAMATH_CALUDE_number_equation_solution_l3667_366721


namespace NUMINAMATH_CALUDE_c_k_value_l3667_366732

/-- Arithmetic sequence with first term 1 and common difference d -/
def arithmetic_seq (d : ℕ) (n : ℕ) : ℕ :=
  1 + (n - 1) * d

/-- Geometric sequence with first term 1 and common ratio r -/
def geometric_seq (r : ℕ) (n : ℕ) : ℕ :=
  r ^ (n - 1)

/-- Sum of arithmetic and geometric sequences -/
def c_seq (d r : ℕ) (n : ℕ) : ℕ :=
  arithmetic_seq d n + geometric_seq r n

theorem c_k_value (d r k : ℕ) :
  c_seq d r (k - 1) = 50 ∧ c_seq d r (k + 1) = 500 → c_seq d r k = 78 := by
  sorry

end NUMINAMATH_CALUDE_c_k_value_l3667_366732


namespace NUMINAMATH_CALUDE_area_of_EFGH_l3667_366785

/-- Represents a parallelogram with a base and height -/
structure Parallelogram where
  base : ℝ
  height : ℝ

/-- Calculates the area of a parallelogram -/
def area (p : Parallelogram) : ℝ := p.base * p.height

/-- The specific parallelogram EFGH from the problem -/
def EFGH : Parallelogram := { base := 5, height := 3 }

/-- Theorem stating that the area of parallelogram EFGH is 15 square units -/
theorem area_of_EFGH : area EFGH = 15 := by sorry

end NUMINAMATH_CALUDE_area_of_EFGH_l3667_366785


namespace NUMINAMATH_CALUDE_record_deal_profit_difference_l3667_366794

/-- Calculates the difference in profit between two deals for selling records -/
theorem record_deal_profit_difference 
  (total_records : ℕ) 
  (sammy_price : ℚ) 
  (bryan_price_interested : ℚ) 
  (bryan_price_not_interested : ℚ) : 
  total_records = 200 →
  sammy_price = 4 →
  bryan_price_interested = 6 →
  bryan_price_not_interested = 1 →
  (total_records : ℚ) * sammy_price - 
  ((total_records / 2 : ℚ) * bryan_price_interested + 
   (total_records / 2 : ℚ) * bryan_price_not_interested) = 100 := by
  sorry

#check record_deal_profit_difference

end NUMINAMATH_CALUDE_record_deal_profit_difference_l3667_366794


namespace NUMINAMATH_CALUDE_original_selling_price_l3667_366749

theorem original_selling_price (P : ℝ) (S : ℝ) (S_new : ℝ) : 
  S = 1.1 * P →
  S_new = 1.3 * (0.9 * P) →
  S_new = S + 35 →
  S = 550 := by
sorry

end NUMINAMATH_CALUDE_original_selling_price_l3667_366749


namespace NUMINAMATH_CALUDE_point_C_complex_number_l3667_366739

/-- Given three points A, B, C in the complex plane, prove that C corresponds to 4-2i -/
theorem point_C_complex_number 
  (A B C : ℂ) 
  (hA : A = 2 + Complex.I) 
  (hBA : B - A = 1 + 2 * Complex.I) 
  (hBC : C - B = 3 - Complex.I) : 
  C = 4 - 2 * Complex.I := by
  sorry


end NUMINAMATH_CALUDE_point_C_complex_number_l3667_366739


namespace NUMINAMATH_CALUDE_correct_answer_l3667_366723

theorem correct_answer (x : ℝ) (h : 3 * x = 90) : x - 30 = 0 := by
  sorry

end NUMINAMATH_CALUDE_correct_answer_l3667_366723


namespace NUMINAMATH_CALUDE_heartsuit_ratio_l3667_366769

-- Define the ♡ operation
def heartsuit (n m : ℕ) : ℕ := n^2 * m^3

-- Theorem statement
theorem heartsuit_ratio : (heartsuit 3 5) / (heartsuit 5 3) = 5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_heartsuit_ratio_l3667_366769


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l3667_366746

def geometric_sequence (a₁ q : ℝ) : ℕ → ℝ
  | n => a₁ * q ^ (n - 1)

theorem geometric_sequence_sum (a₁ q : ℝ) :
  ∃ (a₁ q : ℝ),
    (geometric_sequence a₁ q 2 + geometric_sequence a₁ q 4 = 20) ∧
    (geometric_sequence a₁ q 3 + geometric_sequence a₁ q 5 = 40) →
    (geometric_sequence a₁ q 5 + geometric_sequence a₁ q 7 = 160) := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l3667_366746


namespace NUMINAMATH_CALUDE_complex_division_l3667_366728

theorem complex_division (i : ℂ) (h : i^2 = -1) : 
  1 / (1 + i)^2 = -1/2 * i := by sorry

end NUMINAMATH_CALUDE_complex_division_l3667_366728


namespace NUMINAMATH_CALUDE_combined_average_age_l3667_366720

/-- Given two groups of people with their respective sizes and average ages,
    calculate the average age of all people combined. -/
theorem combined_average_age
  (size_a : ℕ) (avg_a : ℚ) (size_b : ℕ) (avg_b : ℚ)
  (h1 : size_a = 8)
  (h2 : avg_a = 45)
  (h3 : size_b = 6)
  (h4 : avg_b = 20) :
  (size_a : ℚ) * avg_a + (size_b : ℚ) * avg_b = 240 ∧
  (size_a : ℚ) + (size_b : ℚ) = 14 →
  (size_a : ℚ) * avg_a + (size_b : ℚ) * avg_b / ((size_a : ℚ) + (size_b : ℚ)) = 240 / 7 :=
by sorry

#check combined_average_age

end NUMINAMATH_CALUDE_combined_average_age_l3667_366720


namespace NUMINAMATH_CALUDE_sqrt_9025_squared_l3667_366707

theorem sqrt_9025_squared : (Real.sqrt 9025)^2 = 9025 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_9025_squared_l3667_366707


namespace NUMINAMATH_CALUDE_pentagon_area_l3667_366799

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a pentagon -/
structure Pentagon :=
  (F G H I J : Point)

/-- Calculates the area of a pentagon -/
def area (p : Pentagon) : ℝ := sorry

/-- Calculates the angle between three points -/
def angle (A B C : Point) : ℝ := sorry

/-- Calculates the distance between two points -/
def distance (A B : Point) : ℝ := sorry

/-- Theorem: The area of the specific pentagon FGHIJ is 71√3/4 -/
theorem pentagon_area (p : Pentagon) :
  angle p.F p.G p.H = 120 * π / 180 →
  angle p.J p.F p.G = 120 * π / 180 →
  distance p.J p.F = 3 →
  distance p.F p.G = 3 →
  distance p.G p.H = 3 →
  distance p.H p.I = 5 →
  distance p.I p.J = 5 →
  area p = 71 * Real.sqrt 3 / 4 := by sorry

end NUMINAMATH_CALUDE_pentagon_area_l3667_366799


namespace NUMINAMATH_CALUDE_quadratic_factorization_l3667_366752

theorem quadratic_factorization :
  ∃ (x : ℝ), x^2 + 6*x - 2 = 0 ↔ ∃ (x : ℝ), (x + 3)^2 = 11 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l3667_366752


namespace NUMINAMATH_CALUDE_merchant_printers_l3667_366777

/-- Calculates the number of printers bought given the total cost, cost per item, and number of keyboards --/
def calculate_printers (total_cost : ℕ) (keyboard_cost : ℕ) (printer_cost : ℕ) (num_keyboards : ℕ) : ℕ :=
  (total_cost - keyboard_cost * num_keyboards) / printer_cost

theorem merchant_printers :
  calculate_printers 2050 20 70 15 = 25 := by
  sorry

end NUMINAMATH_CALUDE_merchant_printers_l3667_366777


namespace NUMINAMATH_CALUDE_misha_money_total_l3667_366745

theorem misha_money_total (initial_money earned_money : ℕ) : 
  initial_money = 34 → earned_money = 13 → initial_money + earned_money = 47 := by
  sorry

end NUMINAMATH_CALUDE_misha_money_total_l3667_366745


namespace NUMINAMATH_CALUDE_two_out_graph_partition_theorem_l3667_366738

/-- A directed graph where each vertex has exactly two outgoing edges -/
structure TwoOutGraph (V : Type*) :=
  (edges : V → V × V)

/-- A partition of vertices into districts -/
def DistrictPartition (V : Type*) := V → Fin 1014

theorem two_out_graph_partition_theorem {V : Type*} (G : TwoOutGraph V) :
  ∃ (partition : DistrictPartition V),
    (∀ v w : V, (partition v = partition w) → 
      (G.edges v).1 ≠ w ∧ (G.edges v).2 ≠ w) ∧
    (∀ d1 d2 : Fin 1014, d1 ≠ d2 → 
      (∀ v w : V, partition v = d1 → partition w = d2 → 
        ((G.edges v).1 = w ∨ (G.edges v).2 = w) → 
        ∀ x y : V, partition x = d1 → partition y = d2 → 
          ((G.edges x).1 = y ∨ (G.edges x).2 = y) → 
          ((G.edges v).1 = w ∨ (G.edges x).1 = y) ∧ 
          ((G.edges v).2 = w ∨ (G.edges x).2 = y))) :=
sorry

end NUMINAMATH_CALUDE_two_out_graph_partition_theorem_l3667_366738


namespace NUMINAMATH_CALUDE_merry_go_round_time_l3667_366790

/-- The time taken for the second horse to travel the same distance as the first horse -/
theorem merry_go_round_time (r₁ r₂ : ℝ) (rev : ℕ) (v₁ v₂ : ℝ) : 
  r₁ = 30 → r₂ = 15 → rev = 40 → v₁ = 3 → v₂ = 6 → 
  (2 * Real.pi * r₂ * (rev * 2 * Real.pi * r₁) / v₂) = (400 * Real.pi) := by
  sorry

#check merry_go_round_time

end NUMINAMATH_CALUDE_merry_go_round_time_l3667_366790


namespace NUMINAMATH_CALUDE_fifth_term_value_l3667_366773

def geometric_sequence (a : ℕ → ℚ) : Prop :=
  a 1 = 1 ∧ ∀ n : ℕ, 2 * a (n + 1) = a n

theorem fifth_term_value (a : ℕ → ℚ) (h : geometric_sequence a) : a 5 = 1 / 16 := by
  sorry

end NUMINAMATH_CALUDE_fifth_term_value_l3667_366773


namespace NUMINAMATH_CALUDE_sine_cosine_inequality_l3667_366742

theorem sine_cosine_inequality (x : Real) (h : Real.sin x + Real.cos x ≤ 0) :
  Real.sin x ^ 1993 + Real.cos x ^ 1993 ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_sine_cosine_inequality_l3667_366742


namespace NUMINAMATH_CALUDE_range_of_m_l3667_366772

/-- Given conditions for the problem -/
structure ProblemConditions (m : ℝ) :=
  (h1 : ∃ x : ℝ, (x^2 + 1) * (x^2 - 8*x - 20) ≤ 0)
  (h2 : ∃ x : ℝ, x^2 - 2*x + 1 - m^2 ≤ 0)
  (h3 : m > 0)
  (h4 : ∀ x : ℝ, (x < -2 ∨ x > 10) → (x < 1 - m ∨ x > 1 + m))
  (h5 : ∃ x : ℝ, (x < -2 ∨ x > 10) ∧ ¬(x < 1 - m ∨ x > 1 + m))

/-- The main theorem stating the range of m -/
theorem range_of_m (m : ℝ) (h : ProblemConditions m) : m ≥ 9 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l3667_366772


namespace NUMINAMATH_CALUDE_probability_two_red_balls_l3667_366734

/-- The probability of selecting two red balls from a bag with given ball counts. -/
theorem probability_two_red_balls (red blue green : ℕ) (h : red = 5 ∧ blue = 6 ∧ green = 2) :
  let total := red + blue + green
  let choose_two (n : ℕ) := n * (n - 1) / 2
  (choose_two red : ℚ) / (choose_two total) = 5 / 39 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_red_balls_l3667_366734


namespace NUMINAMATH_CALUDE_least_distinct_values_l3667_366766

theorem least_distinct_values (n : ℕ) (mode_count : ℕ) (total_count : ℕ) :
  n = 2023 →
  mode_count = 15 →
  total_count = n →
  (∃ (x : ℕ), x = 145 ∧ 
    (∀ (y : ℕ), y < x → 
      ¬(y * (mode_count - 1) + mode_count ≥ total_count))) := by
  sorry

end NUMINAMATH_CALUDE_least_distinct_values_l3667_366766


namespace NUMINAMATH_CALUDE_simplify_fraction_product_l3667_366708

theorem simplify_fraction_product : (1 / (2 + Real.sqrt 3)) * (1 / (2 - Real.sqrt 3)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_product_l3667_366708


namespace NUMINAMATH_CALUDE_hyperbola_circle_no_intersection_l3667_366726

/-- The range of real values of a for which the asymptotes of the hyperbola x^2/4 - y^2 = 1
    have no common points with the circle x^2 + y^2 - 2ax + 1 = 0 -/
theorem hyperbola_circle_no_intersection (a : ℝ) : 
  (∀ x y : ℝ, x^2/4 - y^2 = 1 → x^2 + y^2 - 2*a*x + 1 ≠ 0) ↔ 
  (a ∈ Set.Ioo (-Real.sqrt 5 / 2) (-1) ∪ Set.Ioo 1 (Real.sqrt 5 / 2)) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_circle_no_intersection_l3667_366726


namespace NUMINAMATH_CALUDE_natashas_distance_l3667_366744

/-- The distance to Natasha's destination given her speed and travel time -/
theorem natashas_distance (speed_limit : ℝ) (over_limit : ℝ) (travel_time : ℝ) 
  (h1 : speed_limit = 50)
  (h2 : over_limit = 10)
  (h3 : travel_time = 1) :
  speed_limit + over_limit * travel_time = 60 := by
  sorry

end NUMINAMATH_CALUDE_natashas_distance_l3667_366744


namespace NUMINAMATH_CALUDE_positive_number_equation_solution_l3667_366715

theorem positive_number_equation_solution :
  ∃ n : ℝ, n > 0 ∧ 3 * n^2 + n = 219 ∧ abs (n - 8.38) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_positive_number_equation_solution_l3667_366715


namespace NUMINAMATH_CALUDE_abs_sum_inequality_l3667_366783

theorem abs_sum_inequality (x : ℝ) : 
  |x - 2| + |x + 3| < 8 ↔ -9/2 < x ∧ x < 7/2 :=
sorry

end NUMINAMATH_CALUDE_abs_sum_inequality_l3667_366783


namespace NUMINAMATH_CALUDE_no_intersection_quadratic_sets_l3667_366798

theorem no_intersection_quadratic_sets (A B : ℤ) :
  ∃ C : ℤ, ∀ x y : ℤ, x^2 + A*x + B ≠ 2*y^2 + 2*y + C := by
  sorry

end NUMINAMATH_CALUDE_no_intersection_quadratic_sets_l3667_366798


namespace NUMINAMATH_CALUDE_triangle_area_is_20_16_l3667_366717

/-- Represents a line in 2D space --/
structure Line where
  slope : ℚ
  point : ℚ × ℚ

/-- Calculates the area of a triangle given three points --/
def triangleArea (p1 p2 p3 : ℚ × ℚ) : ℚ :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let (x3, y3) := p3
  (1/2) * abs (x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2))

/-- Finds the intersection point of a line with the line x + y = 12 --/
def intersectionWithSum12 (l : Line) : ℚ × ℚ :=
  let (a, b) := l.point
  let m := l.slope
  let x := (12 - b + m*a) / (m + 1)
  (x, 12 - x)

theorem triangle_area_is_20_16 (l1 l2 : Line) :
  l1.point = (4, 4) →
  l2.point = (4, 4) →
  l1.slope = 2/3 →
  l2.slope = 3/2 →
  let p1 := (4, 4)
  let p2 := intersectionWithSum12 l1
  let p3 := intersectionWithSum12 l2
  triangleArea p1 p2 p3 = 20.16 := by
sorry

end NUMINAMATH_CALUDE_triangle_area_is_20_16_l3667_366717


namespace NUMINAMATH_CALUDE_polygon_interior_angles_l3667_366789

theorem polygon_interior_angles (n : ℕ) : 
  (n - 2) * 180 = 720 → n = 6 := by
sorry

end NUMINAMATH_CALUDE_polygon_interior_angles_l3667_366789


namespace NUMINAMATH_CALUDE_root_in_interval_iff_a_in_range_l3667_366775

/-- The function f(x) = x^2 - ax + 1 has a root in the interval (1/2, 3) if and only if a ∈ [2, 10/3) -/
theorem root_in_interval_iff_a_in_range (a : ℝ) : 
  (∃ x : ℝ, 1/2 < x ∧ x < 3 ∧ x^2 - a*x + 1 = 0) ↔ 2 ≤ a ∧ a < 10/3 := by
sorry

end NUMINAMATH_CALUDE_root_in_interval_iff_a_in_range_l3667_366775


namespace NUMINAMATH_CALUDE_servant_payment_theorem_l3667_366711

/-- Calculates the money received by a servant who worked for 9 months, given a yearly salary and uniform value -/
def servant_payment (yearly_salary : ℚ) (uniform_value : ℚ) : ℚ :=
  (yearly_salary * 9 / 12) - uniform_value

/-- The servant payment theorem -/
theorem servant_payment_theorem (yearly_salary : ℚ) (uniform_value : ℚ) 
  (h1 : yearly_salary = 500)
  (h2 : uniform_value = 300) :
  servant_payment yearly_salary uniform_value = 75.03 := by
  sorry

#eval servant_payment 500 300

end NUMINAMATH_CALUDE_servant_payment_theorem_l3667_366711


namespace NUMINAMATH_CALUDE_rays_number_l3667_366741

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def sum_of_digits (n : ℕ) : ℕ := (n / 10) + (n % 10)

def reverse_digits (n : ℕ) : ℕ := 10 * (n % 10) + (n / 10)

theorem rays_number :
  ∃ n : ℕ,
    is_two_digit n ∧
    n > 4 * (sum_of_digits n) + 3 ∧
    n + 18 = reverse_digits n ∧
    n = 35 := by
  sorry

end NUMINAMATH_CALUDE_rays_number_l3667_366741


namespace NUMINAMATH_CALUDE_wizard_elixir_combinations_l3667_366764

/-- Represents the number of roots available to the wizard. -/
def num_roots : ℕ := 4

/-- Represents the number of minerals available to the wizard. -/
def num_minerals : ℕ := 5

/-- Represents the number of incompatible pairs of roots and minerals. -/
def num_incompatible_pairs : ℕ := 3

/-- Theorem stating the number of possible combinations for the wizard's elixir. -/
theorem wizard_elixir_combinations : 
  num_roots * num_minerals - num_incompatible_pairs = 17 := by
  sorry

end NUMINAMATH_CALUDE_wizard_elixir_combinations_l3667_366764


namespace NUMINAMATH_CALUDE_line_parallel_to_plane_l3667_366787

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations between lines and planes
variable (perpendicular : Plane → Plane → Prop)
variable (perpendicularLine : Line → Plane → Prop)
variable (parallel : Line → Plane → Prop)
variable (contains : Plane → Line → Prop)

-- State the theorem
theorem line_parallel_to_plane
  (α β : Plane) (m : Line)
  (h1 : perpendicular α β)
  (h2 : perpendicularLine m β)
  (h3 : ¬ contains α m) :
  parallel m α :=
sorry

end NUMINAMATH_CALUDE_line_parallel_to_plane_l3667_366787


namespace NUMINAMATH_CALUDE_alpha_beta_not_perfect_square_l3667_366747

/-- A polynomial of degree 4 with roots 0, αβ, βγ, and γα -/
def f (α β γ : ℕ) (x : ℤ) : ℤ := x * (x - α * β) * (x - β * γ) * (x - γ * α)

/-- Theorem: Given positive integers α, β, γ, and an integer s such that
    f(-1) = f(s)², αβ is not a perfect square. -/
theorem alpha_beta_not_perfect_square (α β γ : ℕ) (s : ℤ) 
    (hα : α > 0) (hβ : β > 0) (hγ : γ > 0)
    (h_eq : f α β γ (-1) = (f α β γ s)^2) :
    ¬ ∃ (k : ℕ), α * β = k^2 := by
  sorry

end NUMINAMATH_CALUDE_alpha_beta_not_perfect_square_l3667_366747


namespace NUMINAMATH_CALUDE_inequality_proof_l3667_366704

theorem inequality_proof (x : ℝ) (h1 : (3/2 : ℝ) ≤ x) (h2 : x ≤ 5) :
  2 * Real.sqrt (x + 1) + Real.sqrt (2 * x - 3) + Real.sqrt (15 - 3 * x) < 2 * Real.sqrt 19 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3667_366704


namespace NUMINAMATH_CALUDE_denarii_puzzle_l3667_366797

theorem denarii_puzzle (x y : ℚ) : 
  (x + 7 = 5 * (y - 7)) →
  (y + 5 = 7 * (x - 5)) →
  (x = 11 + 9 / 17 ∧ y = 9 + 14 / 17) :=
by sorry

end NUMINAMATH_CALUDE_denarii_puzzle_l3667_366797


namespace NUMINAMATH_CALUDE_intersection_with_complement_l3667_366788

open Set

-- Define the universal set U
def U : Set Nat := {1, 2, 3, 4}

-- Define set P
def P : Set Nat := {1, 2}

-- Define set Q
def Q : Set Nat := {2, 3}

-- Theorem statement
theorem intersection_with_complement : P ∩ (U \ Q) = {1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_with_complement_l3667_366788


namespace NUMINAMATH_CALUDE_two_digit_number_divisible_by_55_l3667_366768

theorem two_digit_number_divisible_by_55 (a b : ℕ) : 
  a ≤ 9 → b ≤ 9 → 
  (10 * a + b) % 55 = 0 → 
  (∀ (x y : ℕ), x ≤ 9 → y ≤ 9 → (10 * x + y) % 55 = 0 → x * y ≤ b * a) →
  b * a ≤ 15 →
  10 * a + b = 55 := by
sorry

end NUMINAMATH_CALUDE_two_digit_number_divisible_by_55_l3667_366768


namespace NUMINAMATH_CALUDE_fraction_of_fraction_one_ninth_of_three_fourths_l3667_366774

theorem fraction_of_fraction (a b c d : ℚ) (h1 : b ≠ 0) (h2 : d ≠ 0) :
  (a / b) / (c / d) = (a * d) / (b * c) := by sorry

theorem one_ninth_of_three_fourths :
  (1 / 9) / (3 / 4) = 4 / 27 := by sorry

end NUMINAMATH_CALUDE_fraction_of_fraction_one_ninth_of_three_fourths_l3667_366774


namespace NUMINAMATH_CALUDE_log_identity_l3667_366737

theorem log_identity (x : ℝ) : 
  x = (Real.log 3 / Real.log 5) ^ (Real.log 5 / Real.log 3) →
  Real.log x / Real.log 7 = -(Real.log (Real.log 5 / Real.log 3) / Real.log 7) * (Real.log 5 / Real.log 3) :=
by sorry

end NUMINAMATH_CALUDE_log_identity_l3667_366737


namespace NUMINAMATH_CALUDE_chocolate_ratio_simplification_l3667_366761

theorem chocolate_ratio_simplification :
  let white_chocolate : ℕ := 20
  let dark_chocolate : ℕ := 15
  let gcd := Nat.gcd white_chocolate dark_chocolate
  (white_chocolate / gcd : ℚ) / (dark_chocolate / gcd : ℚ) = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_ratio_simplification_l3667_366761


namespace NUMINAMATH_CALUDE_expedition_time_theorem_l3667_366780

/-- Represents the expedition parameters and calculates the minimum time to circle the mountain. -/
def minimum_expedition_time (total_distance : ℝ) (walking_speed : ℝ) (food_capacity : ℝ) : ℝ :=
  23.5

/-- Theorem stating that the minimum time to circle the mountain under given conditions is 23.5 days. -/
theorem expedition_time_theorem (total_distance : ℝ) (walking_speed : ℝ) (food_capacity : ℝ) 
  (h1 : total_distance = 100)
  (h2 : walking_speed = 20)
  (h3 : food_capacity = 2) :
  minimum_expedition_time total_distance walking_speed food_capacity = 23.5 := by
  sorry

end NUMINAMATH_CALUDE_expedition_time_theorem_l3667_366780


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l3667_366753

-- Define a real-valued function f
variable (f : ℝ → ℝ)

-- Define the properties of f
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)
def is_decreasing_on_nonneg (f : ℝ → ℝ) : Prop := 
  ∀ x y, 0 ≤ x → 0 ≤ y → x < y → f y < f x

-- State the theorem
theorem solution_set_of_inequality 
  (h_even : is_even f) 
  (h_decreasing : is_decreasing_on_nonneg f) :
  {x : ℝ | f (2*x + 5) > f (x^2 + 2)} = {x : ℝ | x < -1 ∨ x > 3} := by
  sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l3667_366753


namespace NUMINAMATH_CALUDE_fried_chicken_cost_is_12_l3667_366758

/-- Calculates the cost of a fried chicken bucket given budget information and beef purchase details. -/
def fried_chicken_cost (total_budget : ℕ) (amount_left : ℕ) (beef_quantity : ℕ) (beef_price : ℕ) : ℕ :=
  total_budget - amount_left - beef_quantity * beef_price

/-- Proves that the cost of the fried chicken bucket is $12 given the problem conditions. -/
theorem fried_chicken_cost_is_12 :
  fried_chicken_cost 80 53 5 3 = 12 := by
  sorry

end NUMINAMATH_CALUDE_fried_chicken_cost_is_12_l3667_366758


namespace NUMINAMATH_CALUDE_smallest_m_correct_l3667_366792

/-- The smallest positive integer m for which 10x^2 - mx + 420 = 0 has integral solutions -/
def smallest_m : ℕ := 130

/-- Predicate to check if a quadratic equation has integral solutions -/
def has_integral_solutions (a b c : ℤ) : Prop :=
  ∃ x : ℤ, a * x^2 + b * x + c = 0

theorem smallest_m_correct :
  (∀ m : ℕ, m < smallest_m → ¬ has_integral_solutions 10 (-m) 420) ∧
  has_integral_solutions 10 (-smallest_m) 420 :=
sorry

end NUMINAMATH_CALUDE_smallest_m_correct_l3667_366792


namespace NUMINAMATH_CALUDE_boisjoli_farm_egg_boxes_l3667_366793

/-- The number of boxes filled with eggs in a week -/
def boxes_filled_per_week (num_hens : ℕ) (days_per_week : ℕ) (eggs_per_box : ℕ) : ℕ :=
  (num_hens * days_per_week) / eggs_per_box

/-- Theorem stating that 270 hens laying eggs for 7 days, packed in boxes of 6, results in 315 boxes per week -/
theorem boisjoli_farm_egg_boxes :
  boxes_filled_per_week 270 7 6 = 315 := by
  sorry

end NUMINAMATH_CALUDE_boisjoli_farm_egg_boxes_l3667_366793


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l3667_366759

theorem absolute_value_inequality (x : ℝ) : 
  abs (x - 1) + abs (x + 2) < 5 ↔ -3 < x ∧ x < 2 := by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l3667_366759


namespace NUMINAMATH_CALUDE_sum_of_squares_l3667_366795

theorem sum_of_squares (x y : ℝ) (h1 : x + y = 22) (h2 : x * y = 40) : x^2 + y^2 = 404 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_l3667_366795


namespace NUMINAMATH_CALUDE_solution_set_implies_a_greater_than_negative_one_l3667_366756

theorem solution_set_implies_a_greater_than_negative_one (a : ℝ) :
  (∀ x : ℝ, x * (x - a + 1) > a ↔ (x < -1 ∨ x > a)) →
  a > -1 := by
sorry

end NUMINAMATH_CALUDE_solution_set_implies_a_greater_than_negative_one_l3667_366756


namespace NUMINAMATH_CALUDE_complex_root_theorem_l3667_366710

theorem complex_root_theorem (z : ℂ) (p : ℝ) : 
  (z^2 + 2*z + p = 0) → (Complex.abs z = 2) → (p = 4) := by
  sorry

end NUMINAMATH_CALUDE_complex_root_theorem_l3667_366710


namespace NUMINAMATH_CALUDE_equation_solutions_l3667_366735

theorem equation_solutions :
  let f (x : ℝ) := 3 / (Real.sqrt (x - 5) - 7) + 2 / (Real.sqrt (x - 5) - 3) +
                   9 / (Real.sqrt (x - 5) + 3) + 15 / (Real.sqrt (x - 5) + 7)
  ∀ x : ℝ, f x = 0 ↔ x = 54 ∨ x = 846 / 29 := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l3667_366735


namespace NUMINAMATH_CALUDE_annual_profit_calculation_l3667_366743

theorem annual_profit_calculation (second_half_profit first_half_profit total_profit : ℕ) :
  second_half_profit = 442500 →
  first_half_profit = second_half_profit + 2750000 →
  total_profit = first_half_profit + second_half_profit →
  total_profit = 3635000 := by
  sorry

end NUMINAMATH_CALUDE_annual_profit_calculation_l3667_366743


namespace NUMINAMATH_CALUDE_min_segments_proof_l3667_366767

/-- A polyline that passes through the centers of all smaller squares in a larger square divided into n^2 equal parts. -/
structure Polyline (n : ℕ) where
  segments : ℕ
  passes_through_all_centers : Bool

/-- The minimum number of segments for a valid polyline in an n x n grid -/
def min_segments (n : ℕ) : ℕ := 2 * n - 2

/-- Theorem stating that the minimum number of segments in a valid polyline is 2n - 2 -/
theorem min_segments_proof (n : ℕ) (p : Polyline n) (h : p.passes_through_all_centers = true) :
  p.segments ≥ min_segments n :=
sorry

end NUMINAMATH_CALUDE_min_segments_proof_l3667_366767


namespace NUMINAMATH_CALUDE_cap_production_l3667_366716

theorem cap_production (first_week second_week third_week total_target : ℕ) 
  (h1 : first_week = 320)
  (h2 : second_week = 400)
  (h3 : third_week = 300)
  (h4 : total_target = 1360) :
  total_target - (first_week + second_week + third_week) = 340 :=
by sorry

end NUMINAMATH_CALUDE_cap_production_l3667_366716


namespace NUMINAMATH_CALUDE_simplify_expression_l3667_366703

theorem simplify_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a^4 + b^4 = a^2 + b^2) :
  a/b + b/a - 1/(a*b) = 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3667_366703


namespace NUMINAMATH_CALUDE_evaluate_F_of_f_l3667_366750

-- Define the functions f and F
def f (a : ℝ) : ℝ := a^2 - 1
def F (a b : ℝ) : ℝ := b^3 - a

-- State the theorem
theorem evaluate_F_of_f : F 2 (f 3) = 510 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_F_of_f_l3667_366750


namespace NUMINAMATH_CALUDE_factorial_multiple_implies_inequality_l3667_366724

theorem factorial_multiple_implies_inequality (a b : ℕ+) 
  (h : (a.val.factorial * b.val.factorial) % (a.val.factorial + b.val.factorial) = 0) : 
  3 * a.val ≥ 2 * b.val + 2 := by
  sorry

end NUMINAMATH_CALUDE_factorial_multiple_implies_inequality_l3667_366724


namespace NUMINAMATH_CALUDE_road_length_16_trees_l3667_366782

/-- Calculates the length of a road given the number of trees, space per tree, and space between trees. -/
def roadLength (numTrees : ℕ) (spacePerTree : ℕ) (spaceBetweenTrees : ℕ) : ℕ :=
  numTrees * spacePerTree + (numTrees - 1) * spaceBetweenTrees

/-- Proves that the length of the road with 16 trees, 1 foot per tree, and 9 feet between trees is 151 feet. -/
theorem road_length_16_trees : roadLength 16 1 9 = 151 := by
  sorry

end NUMINAMATH_CALUDE_road_length_16_trees_l3667_366782


namespace NUMINAMATH_CALUDE_product_mod_500_l3667_366719

theorem product_mod_500 : (1502 * 2021) % 500 = 42 := by
  sorry

end NUMINAMATH_CALUDE_product_mod_500_l3667_366719


namespace NUMINAMATH_CALUDE_percentage_calculation_l3667_366701

theorem percentage_calculation (number : ℝ) (h : number = 4400) : 
  0.15 * (0.30 * (0.50 * number)) = 99 := by sorry

end NUMINAMATH_CALUDE_percentage_calculation_l3667_366701


namespace NUMINAMATH_CALUDE_flagpole_shadow_length_l3667_366736

/-- Given a flagpole and a building under similar shadow-casting conditions,
    proves that the length of the shadow cast by the flagpole is 45 meters. -/
theorem flagpole_shadow_length
  (flagpole_height : ℝ)
  (building_height : ℝ)
  (building_shadow : ℝ)
  (h_flagpole : flagpole_height = 18)
  (h_building_height : building_height = 20)
  (h_building_shadow : building_shadow = 50)
  (h_similar_conditions : flagpole_height / building_height = building_shadow / building_shadow) :
  flagpole_height * building_shadow / building_height = 45 :=
sorry

end NUMINAMATH_CALUDE_flagpole_shadow_length_l3667_366736


namespace NUMINAMATH_CALUDE_even_monotone_increasing_range_l3667_366751

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def monotone_increasing_on (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
  ∀ {x y}, x ∈ s → y ∈ s → x ≤ y → f x ≤ f y

theorem even_monotone_increasing_range 
  (f : ℝ → ℝ) 
  (h_even : is_even_function f)
  (h_mono : monotone_increasing_on f (Set.Ici 0)) :
  {x : ℝ | f x < f 1} = Set.Ioo (-1) 1 := by
  sorry

end NUMINAMATH_CALUDE_even_monotone_increasing_range_l3667_366751


namespace NUMINAMATH_CALUDE_divisibility_proof_l3667_366706

theorem divisibility_proof (a : ℤ) (n : ℕ) : 
  ∃ k : ℤ, (a + 1)^(2*n + 1) + a^(n + 2) = k * (a^2 + a + 1) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_proof_l3667_366706


namespace NUMINAMATH_CALUDE_quadratic_real_root_l3667_366771

theorem quadratic_real_root (b : ℝ) : 
  (∃ x : ℝ, x^2 + b*x + 25 = 0) ↔ b ≤ -10 ∨ b ≥ 10 := by sorry

end NUMINAMATH_CALUDE_quadratic_real_root_l3667_366771


namespace NUMINAMATH_CALUDE_product_minus_difference_l3667_366722

theorem product_minus_difference (x y : ℝ) (h1 : x > 0) (h2 : y > 0) 
  (h3 : x + y = 6) (h4 : x / y = 6) : x * y - (x - y) = 6 / 49 := by
  sorry

end NUMINAMATH_CALUDE_product_minus_difference_l3667_366722
