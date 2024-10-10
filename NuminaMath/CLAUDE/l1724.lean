import Mathlib

namespace container_volume_comparison_l1724_172494

theorem container_volume_comparison (a r : ℝ) (ha : a > 0) (hr : r > 0) 
  (h_eq : (2 * a)^3 = (4/3) * Real.pi * r^3) : 
  (2*a + 2)^3 > (4/3) * Real.pi * (r + 1)^3 := by
  sorry

end container_volume_comparison_l1724_172494


namespace negative_five_greater_than_negative_seventeen_l1724_172451

theorem negative_five_greater_than_negative_seventeen : -5 > -17 := by
  sorry

end negative_five_greater_than_negative_seventeen_l1724_172451


namespace sum_of_numbers_l1724_172489

theorem sum_of_numbers : (3 : ℚ) / 25 + (1 : ℚ) / 5 + 55.21 = 55.53 := by
  sorry

end sum_of_numbers_l1724_172489


namespace parabola_properties_l1724_172477

/-- Definition of the parabola C: x² = 4y -/
def parabola (x y : ℝ) : Prop := x^2 = 4*y

/-- Definition of the line y = x + 1 -/
def line (x y : ℝ) : Prop := y = x + 1

/-- The focus of the parabola -/
def focus : ℝ × ℝ := (0, 1)

/-- The length of the chord AB -/
def chord_length : ℝ := 8

/-- Theorem stating the properties of the parabola and its intersection with the line -/
theorem parabola_properties :
  (∀ x y, parabola x y → (x, y) ≠ focus → (x - focus.1)^2 + (y - focus.2)^2 > 0) ∧
  (∃ A B : ℝ × ℝ, 
    parabola A.1 A.2 ∧ parabola B.1 B.2 ∧
    line A.1 A.2 ∧ line B.1 B.2 ∧
    A ≠ B ∧
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = chord_length) :=
sorry

end parabola_properties_l1724_172477


namespace sand_price_per_ton_l1724_172482

theorem sand_price_per_ton 
  (total_cost : ℕ) 
  (cement_bags : ℕ) 
  (cement_price_per_bag : ℕ) 
  (sand_lorries : ℕ) 
  (sand_tons_per_lorry : ℕ) 
  (h1 : total_cost = 13000)
  (h2 : cement_bags = 500)
  (h3 : cement_price_per_bag = 10)
  (h4 : sand_lorries = 20)
  (h5 : sand_tons_per_lorry = 10) : 
  (total_cost - cement_bags * cement_price_per_bag) / (sand_lorries * sand_tons_per_lorry) = 40 := by
sorry

end sand_price_per_ton_l1724_172482


namespace opposite_numbers_expression_l1724_172412

theorem opposite_numbers_expression (a b c d : ℤ) : 
  (a + b = 0) →  -- a and b are opposite numbers
  (c = -1) →     -- c is the largest negative integer
  (d = 1) →      -- d is the smallest positive integer
  (a + b) * d + d - c = 2 := by
  sorry

end opposite_numbers_expression_l1724_172412


namespace pit_stop_duration_is_20_minutes_l1724_172459

-- Define the parameters of the problem
def total_trip_time_without_stops : ℕ := 14 -- hours
def stop_interval : ℕ := 2 -- hours
def additional_food_stops : ℕ := 2
def additional_gas_stops : ℕ := 3
def total_trip_time_with_stops : ℕ := 18 -- hours

-- Calculate the number of stops
def total_stops : ℕ := 
  (total_trip_time_without_stops / stop_interval) + additional_food_stops + additional_gas_stops

-- Define the theorem
theorem pit_stop_duration_is_20_minutes : 
  (total_trip_time_with_stops - total_trip_time_without_stops) * 60 / total_stops = 20 := by
  sorry

end pit_stop_duration_is_20_minutes_l1724_172459


namespace walking_problem_l1724_172485

/-- Proves that the given conditions lead to the correct system of equations --/
theorem walking_problem (x y : ℝ) : 
  (∀ t : ℝ, t * x < t * y) → -- Xiao Wang walks faster than Xiao Zhang
  (3 * x + 210 = 5 * y) →    -- Distance condition after 3 and 5 minutes
  (10 * y - 10 * x = 100) →  -- Time and initial distance condition
  (∃ d : ℝ, d > 0 ∧ 10 * x = d ∧ 10 * y = d + 100) → -- Both reach the museum in 10 minutes
  (3 * x + 210 = 5 * y ∧ 10 * y - 10 * x = 100) :=
by sorry

end walking_problem_l1724_172485


namespace rectangular_prism_diagonal_l1724_172434

theorem rectangular_prism_diagonal (width length height : ℝ) 
  (hw : width = 12) (hl : length = 16) (hh : height = 9) : 
  Real.sqrt (width^2 + length^2 + height^2) = Real.sqrt 481 := by
  sorry

end rectangular_prism_diagonal_l1724_172434


namespace inscribed_prism_surface_area_l1724_172484

/-- The surface area of a right square prism inscribed in a sphere -/
theorem inscribed_prism_surface_area (r h : ℝ) (hr : r = Real.sqrt 6) (hh : h = 4) :
  let a := Real.sqrt ((r^2 - h^2/4) / 2)
  2 * a^2 + 4 * a * h = 40 :=
sorry

end inscribed_prism_surface_area_l1724_172484


namespace barney_towel_usage_l1724_172441

/-- The number of towels Barney owns -/
def total_towels : ℕ := 18

/-- The number of days in a week -/
def days_per_week : ℕ := 7

/-- The number of days Barney will not have clean towels -/
def days_without_clean_towels : ℕ := 5

/-- The number of towels Barney uses at a time -/
def towels_per_use : ℕ := 2

theorem barney_towel_usage :
  ∃ (x : ℕ),
    x = towels_per_use ∧
    total_towels - days_per_week * x = (days_per_week - days_without_clean_towels) * x :=
by sorry

end barney_towel_usage_l1724_172441


namespace compressor_stations_theorem_l1724_172424

/-- Represents the distances between three compressor stations -/
structure CompressorStations where
  x : ℝ  -- distance between first and second stations
  y : ℝ  -- distance between second and third stations
  z : ℝ  -- direct distance between first and third stations
  a : ℝ  -- parameter

/-- Conditions for the compressor stations arrangement -/
def valid_arrangement (s : CompressorStations) : Prop :=
  s.x > 0 ∧ s.y > 0 ∧ s.z > 0 ∧
  s.x + s.y = 4 * s.z ∧
  s.z + s.y = s.x + s.a ∧
  s.x + s.z = 85 ∧
  s.x + s.y > s.z ∧
  s.x + s.z > s.y ∧
  s.y + s.z > s.x

theorem compressor_stations_theorem :
  ∀ s : CompressorStations,
  valid_arrangement s →
  (0 < s.a ∧ s.a < 68) ∧
  (s.a = 5 → s.x = 60 ∧ s.y = 40 ∧ s.z = 25) :=
sorry

end compressor_stations_theorem_l1724_172424


namespace ali_baba_max_camels_l1724_172460

/-- Represents the problem of maximizing the number of camels Ali Baba can buy --/
theorem ali_baba_max_camels :
  let gold_capacity : ℝ := 200
  let diamond_capacity : ℝ := 40
  let max_weight : ℝ := 100
  let gold_camel_rate : ℝ := 20
  let diamond_camel_rate : ℝ := 60
  
  ∃ (gold_weight diamond_weight : ℝ),
    gold_weight ≥ 0 ∧
    diamond_weight ≥ 0 ∧
    gold_weight + diamond_weight ≤ max_weight ∧
    gold_weight / gold_capacity + diamond_weight / diamond_capacity ≤ 1 ∧
    ∀ (g d : ℝ),
      g ≥ 0 →
      d ≥ 0 →
      g + d ≤ max_weight →
      g / gold_capacity + d / diamond_capacity ≤ 1 →
      gold_camel_rate * g + diamond_camel_rate * d ≤ gold_camel_rate * gold_weight + diamond_camel_rate * diamond_weight ∧
    gold_camel_rate * gold_weight + diamond_camel_rate * diamond_weight = 3000 := by
  sorry


end ali_baba_max_camels_l1724_172460


namespace quadratic_roots_sum_l1724_172431

theorem quadratic_roots_sum (a b : ℝ) : 
  a ≠ b → 
  a^2 - 8*a + 5 = 0 → 
  b^2 - 8*b + 5 = 0 → 
  (b - 1) / (a - 1) + (a - 1) / (b - 1) = -20 := by
sorry

end quadratic_roots_sum_l1724_172431


namespace smallest_integer_y_l1724_172476

theorem smallest_integer_y : ∀ y : ℤ, (5 : ℚ) / 8 < (y - 3 : ℚ) / 19 → y ≥ 15 :=
by sorry

end smallest_integer_y_l1724_172476


namespace fraction_sum_l1724_172468

theorem fraction_sum (a b : ℚ) (h : a / b = 3 / 4) : (a + b) / b = 7 / 4 := by
  sorry

end fraction_sum_l1724_172468


namespace sum_of_squares_of_roots_l1724_172401

theorem sum_of_squares_of_roots (x₁ x₂ : ℝ) : 
  x₁^2 - 3*x₁ - 1 = 0 → 
  x₂^2 - 3*x₂ - 1 = 0 → 
  x₁^2 + x₂^2 = 11 := by
  sorry

end sum_of_squares_of_roots_l1724_172401


namespace two_thirds_of_five_times_nine_l1724_172483

theorem two_thirds_of_five_times_nine : (2 / 3 : ℚ) * (5 * 9) = 30 := by
  sorry

end two_thirds_of_five_times_nine_l1724_172483


namespace large_rectangle_perimeter_large_rectangle_perimeter_proof_l1724_172438

theorem large_rectangle_perimeter : ℝ → ℝ → ℝ → Prop :=
  fun square_perimeter small_rect_perimeter large_rect_perimeter =>
    square_perimeter = 24 ∧
    small_rect_perimeter = 16 ∧
    (∃ (square_side small_rect_width : ℝ),
      square_side = square_perimeter / 4 ∧
      small_rect_width = small_rect_perimeter / 2 - square_side ∧
      large_rect_perimeter = 2 * (3 * square_side + (square_side + small_rect_width))) →
    large_rect_perimeter = 52

theorem large_rectangle_perimeter_proof :
  large_rectangle_perimeter 24 16 52 :=
sorry

end large_rectangle_perimeter_large_rectangle_perimeter_proof_l1724_172438


namespace transform_equation_l1724_172446

theorem transform_equation (x m : ℝ) : 
  (x^2 + 4*x = m) ∧ ((x + 2)^2 = 5) → m = 1 := by
sorry

end transform_equation_l1724_172446


namespace right_triangle_sides_l1724_172453

theorem right_triangle_sides (x y z : ℝ) : 
  x > 0 → y > 0 → z > 0 →
  x + y + z = 156 →
  x * y / 2 = 1014 →
  x^2 + y^2 = z^2 →
  (x = 39 ∧ y = 52 ∧ z = 65) ∨ (x = 52 ∧ y = 39 ∧ z = 65) :=
by sorry

end right_triangle_sides_l1724_172453


namespace computer_additions_per_hour_l1724_172488

/-- The number of additions a computer can perform per second -/
def additions_per_second : ℕ := 10000

/-- The number of seconds in one hour -/
def seconds_per_hour : ℕ := 3600

/-- The number of additions a computer can perform in one hour -/
def additions_per_hour : ℕ := additions_per_second * seconds_per_hour

/-- Theorem stating that the computer performs 36 million additions in one hour -/
theorem computer_additions_per_hour : 
  additions_per_hour = 36000000 := by sorry

end computer_additions_per_hour_l1724_172488


namespace marcos_dads_strawberries_strawberry_problem_l1724_172413

theorem marcos_dads_strawberries (initial_total : ℕ) (dads_extra : ℕ) (marcos_final : ℕ) : ℕ :=
  let dads_initial := initial_total - (marcos_final - dads_extra)
  dads_initial + dads_extra

theorem strawberry_problem : 
  marcos_dads_strawberries 22 30 36 = 46 := by
  sorry

end marcos_dads_strawberries_strawberry_problem_l1724_172413


namespace lemon_square_price_is_correct_l1724_172480

/-- Represents the price of a lemon square -/
def lemon_square_price : ℝ := 2

/-- The number of brownies sold -/
def brownies_sold : ℕ := 4

/-- The price of each brownie -/
def brownie_price : ℝ := 3

/-- The number of lemon squares sold -/
def lemon_squares_sold : ℕ := 5

/-- The number of cookies to be sold -/
def cookies_to_sell : ℕ := 7

/-- The price of each cookie -/
def cookie_price : ℝ := 4

/-- The total revenue goal -/
def total_revenue_goal : ℝ := 50

theorem lemon_square_price_is_correct :
  (brownies_sold : ℝ) * brownie_price +
  (lemon_squares_sold : ℝ) * lemon_square_price +
  (cookies_to_sell : ℝ) * cookie_price =
  total_revenue_goal :=
by sorry

end lemon_square_price_is_correct_l1724_172480


namespace difference_of_squares_81_49_l1724_172419

theorem difference_of_squares_81_49 : 81^2 - 49^2 = 4160 := by
  sorry

end difference_of_squares_81_49_l1724_172419


namespace grid_paths_7x6_l1724_172410

/-- The number of paths in a grid from (0,0) to (m,n) where each step is either right or up -/
def gridPaths (m n : ℕ) : ℕ := Nat.choose (m + n) n

/-- The dimensions of our grid -/
def gridWidth : ℕ := 7
def gridHeight : ℕ := 6

/-- The total number of steps in our grid -/
def totalSteps : ℕ := gridWidth + gridHeight

theorem grid_paths_7x6 : gridPaths gridWidth gridHeight = 1716 := by sorry

end grid_paths_7x6_l1724_172410


namespace equality_of_exponents_l1724_172420

theorem equality_of_exponents (x y z : ℝ) 
  (h : x * (y + z - x) / x = y * (z + x - y) / y ∧ 
       y * (z + x - y) / y = z * (x + y - z) / z) : 
  x^y * y^x = z^y * y^z ∧ z^y * y^z = x^z * z^x := by
  sorry

end equality_of_exponents_l1724_172420


namespace last_two_digits_product_l1724_172418

theorem last_two_digits_product (n : ℤ) : 
  (n % 100 ≥ 10) →  -- Ensure n has at least two digits
  (n % 4 = 0) →     -- n is divisible by 4
  ((n % 100) / 10 + n % 10 = 16) →  -- Sum of last two digits is 16
  ((n % 100) / 10) * (n % 10) = 64 := by
sorry

end last_two_digits_product_l1724_172418


namespace matrix_vector_computation_l1724_172403

variable {n : ℕ}
variable (N : Matrix (Fin 2) (Fin n) ℝ)
variable (a b : Fin n → ℝ)

theorem matrix_vector_computation 
  (ha : N.mulVec a = ![3, 4]) 
  (hb : N.mulVec b = ![1, -2]) :
  N.mulVec (2 • a - 4 • b) = ![2, 16] := by
  sorry

end matrix_vector_computation_l1724_172403


namespace radical_product_simplification_l1724_172442

theorem radical_product_simplification (x : ℝ) (hx : x > 0) :
  Real.sqrt (98 * x) * Real.sqrt (18 * x) * Real.sqrt (50 * x) = 210 * x * Real.sqrt (2 * x) := by
  sorry

end radical_product_simplification_l1724_172442


namespace smallest_dual_base_representation_l1724_172449

def is_valid_representation (n : ℕ) (a b : ℕ) : Prop :=
  a > 2 ∧ b > 2 ∧ n = 1 * a + 3 ∧ n = 3 * b + 1

theorem smallest_dual_base_representation :
  ∃ (n : ℕ), is_valid_representation n 7 3 ∧
  ∀ (m : ℕ) (a b : ℕ), is_valid_representation m a b → n ≤ m :=
sorry

end smallest_dual_base_representation_l1724_172449


namespace three_digit_sum_reverse_l1724_172414

theorem three_digit_sum_reverse : ∃ (a b c : ℕ), 
  (a ≥ 1 ∧ a ≤ 9) ∧ 
  (b ≥ 0 ∧ b ≤ 9) ∧ 
  (c ≥ 0 ∧ c ≤ 9) ∧
  (100 * a + 10 * b + c) + (100 * c + 10 * b + a) = 1777 := by
  sorry

end three_digit_sum_reverse_l1724_172414


namespace range_of_negative_values_l1724_172479

/-- A function f: ℝ → ℝ is even if f(x) = f(-x) for all x -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

/-- A function f: ℝ → ℝ is increasing on (-∞, 0) if f(x) ≤ f(y) for all x < y < 0 -/
def IncreasingOnNegative (f : ℝ → ℝ) : Prop := ∀ x y, x < y → y < 0 → f x ≤ f y

theorem range_of_negative_values (f : ℝ → ℝ) 
  (h_even : IsEven f) 
  (h_inc_neg : IncreasingOnNegative f) 
  (h_f2 : f 2 = 0) :
  ∀ x, f x < 0 ↔ x < -2 ∨ x > 2 := by sorry

end range_of_negative_values_l1724_172479


namespace simplify_expression_l1724_172463

theorem simplify_expression : 
  (2 * (10^12)) / ((4 * (10^5)) - (1 * (10^4))) = 5.1282 * (10^6) := by
  sorry

end simplify_expression_l1724_172463


namespace p_recurrence_l1724_172405

/-- The probability of having a group of length k or more in n tosses of a symmetric coin -/
def p (n k : ℕ) : ℚ :=
  sorry

/-- The recurrence relation for p(n,k) -/
theorem p_recurrence (n k : ℕ) (h : k < n) :
  p n k = p (n-1) k - (1 / 2^k) * p (n-k) k + (1 / 2^k) :=
sorry

end p_recurrence_l1724_172405


namespace terminal_side_second_quadrant_l1724_172467

-- Define the quadrants
inductive Quadrant
| I
| II
| III
| IV

-- Define a function to determine the quadrant of an angle
def angle_quadrant (θ : ℝ) : Quadrant := sorry

-- Define a function to determine the quadrant of the terminal side of an angle
def terminal_side_quadrant (θ : ℝ) : Quadrant := sorry

-- Theorem statement
theorem terminal_side_second_quadrant (α : ℝ) :
  angle_quadrant α = Quadrant.III →
  |Real.cos (α/2)| = -Real.cos (α/2) →
  terminal_side_quadrant (α/2) = Quadrant.II :=
by sorry

end terminal_side_second_quadrant_l1724_172467


namespace quadratic_polynomial_satisfies_conditions_l1724_172499

theorem quadratic_polynomial_satisfies_conditions :
  ∃ (p : ℝ → ℝ),
    (∀ x, p x = (14 * x^2 + 4 * x + 12) / 15) ∧
    p (-2) = 4 ∧
    p 1 = 2 ∧
    p 3 = 10 := by
  sorry

end quadratic_polynomial_satisfies_conditions_l1724_172499


namespace vector_collinearity_l1724_172421

-- Define the vectors
def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (1, 5)
def c : ℝ → ℝ × ℝ := λ x => (x, 1)

-- Define collinearity
def collinear (v w : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, v.1 * w.2 = k * v.2 * w.1

-- Theorem statement
theorem vector_collinearity (x : ℝ) :
  collinear (2 * a - b) (c x) → x = -1 := by
  sorry

end vector_collinearity_l1724_172421


namespace sqrt_inequality_solution_set_l1724_172430

theorem sqrt_inequality_solution_set (x : ℝ) :
  (∃ y : ℝ, y ^ 2 = x + 3 ∧ y < 2) ↔ x ∈ Set.Icc (-3) 1 := by
  sorry

end sqrt_inequality_solution_set_l1724_172430


namespace dino_expenses_l1724_172454

/-- Calculates Dino's monthly expenses based on his work hours, hourly rates, and remaining money --/
theorem dino_expenses (hours1 hours2 hours3 : ℕ) (rate1 rate2 rate3 : ℕ) (remaining : ℕ) : 
  hours1 = 20 → hours2 = 30 → hours3 = 5 →
  rate1 = 10 → rate2 = 20 → rate3 = 40 →
  remaining = 500 →
  hours1 * rate1 + hours2 * rate2 + hours3 * rate3 - remaining = 500 := by
  sorry

#check dino_expenses

end dino_expenses_l1724_172454


namespace inverse_function_sum_l1724_172400

-- Define the function g and its inverse
def g (c d : ℝ) (x : ℝ) : ℝ := c * x + d
def g_inv (c d : ℝ) (x : ℝ) : ℝ := d * x + c

-- State the theorem
theorem inverse_function_sum (c d : ℝ) : 
  (∀ x : ℝ, g c d (g_inv c d x) = x) ∧ 
  (∀ x : ℝ, g_inv c d (g c d x) = x) → 
  c + d = -2 := by
sorry

end inverse_function_sum_l1724_172400


namespace bake_sale_donation_l1724_172475

/-- Calculates the total donation to the homeless shelter from a bake sale fundraiser --/
theorem bake_sale_donation (total_earnings : ℕ) (ingredient_cost : ℕ) (personal_donation : ℕ) : 
  total_earnings = 400 →
  ingredient_cost = 100 →
  personal_donation = 10 →
  ((total_earnings - ingredient_cost) / 2 + personal_donation : ℕ) = 160 := by
  sorry

end bake_sale_donation_l1724_172475


namespace system_solution_l1724_172402

theorem system_solution :
  ∀ (x y z : ℝ),
    (y + z = x * y * z ∧
     z + x = x * y * z ∧
     x + y = x * y * z) →
    ((x = 0 ∧ y = 0 ∧ z = 0) ∨
     (x = Real.sqrt 2 ∧ y = Real.sqrt 2 ∧ z = Real.sqrt 2) ∨
     (x = -Real.sqrt 2 ∧ y = -Real.sqrt 2 ∧ z = -Real.sqrt 2)) :=
by
  sorry

end system_solution_l1724_172402


namespace f_properties_l1724_172448

-- Define the function f
def f (a b c : ℝ) (x : ℝ) : ℝ := x^3 - 6*x^2 + 9*x - a*b*c

-- State the theorem
theorem f_properties (a b c : ℝ) 
  (h1 : 0 < a) (h2 : a < b) (h3 : b < c)
  (h4 : f a b c a = 0) (h5 : f a b c b = 0) (h6 : f a b c c = 0) :
  (f a b c 0) * (f a b c 1) < 0 ∧ (f a b c 0) * (f a b c 3) > 0 := by
  sorry


end f_properties_l1724_172448


namespace apples_ratio_proof_l1724_172426

theorem apples_ratio_proof (jim_apples jane_apples jerry_apples : ℕ) 
  (h1 : jim_apples = 20)
  (h2 : jane_apples = 60)
  (h3 : jerry_apples = 40) :
  (jim_apples + jane_apples + jerry_apples) / 3 / jim_apples = 2 := by
  sorry

end apples_ratio_proof_l1724_172426


namespace bank_number_inconsistency_l1724_172452

-- Define the initial number of banks
def initial_banks : ℕ := 10

-- Define the splitting rule
def split_rule (n : ℕ) : ℕ := n + 7

-- Define the claimed number of banks
def claimed_banks : ℕ := 2023

-- Theorem stating the impossibility of reaching the claimed number of banks
theorem bank_number_inconsistency :
  ∀ n : ℕ, n % 7 = initial_banks % 7 → n ≠ claimed_banks :=
by
  sorry

end bank_number_inconsistency_l1724_172452


namespace oplus_roots_l1724_172457

-- Define the ⊕ operation
def oplus (a b : ℝ) : ℝ := a^2 - 5*a + 2*b

-- State the theorem
theorem oplus_roots : 
  ∃ (x₁ x₂ : ℝ), x₁ = 2 ∧ x₂ = 3 ∧ 
  (∀ x : ℝ, oplus x 3 = 0 ↔ x = x₁ ∨ x = x₂) :=
by sorry

end oplus_roots_l1724_172457


namespace existence_of_solutions_l1724_172473

theorem existence_of_solutions (n : ℕ) : ∃ x y z : ℕ, x^2 + y^2 = z^n := by
  sorry

end existence_of_solutions_l1724_172473


namespace S_is_bounded_region_l1724_172436

/-- The set S of points (x,y) in the coordinate plane where one of 5, x+1, and y-5 is greater than or equal to the other two -/
def S : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (5 ≥ p.1 + 1 ∧ 5 ≥ p.2 - 5) ∨
               (p.1 + 1 ≥ 5 ∧ p.1 + 1 ≥ p.2 - 5) ∨
               (p.2 - 5 ≥ 5 ∧ p.2 - 5 ≥ p.1 + 1)}

/-- S is a single bounded region in the quadrant -/
theorem S_is_bounded_region : 
  ∃ (a b c d : ℝ), a < b ∧ c < d ∧
  S = {p : ℝ × ℝ | a ≤ p.1 ∧ p.1 ≤ b ∧ c ≤ p.2 ∧ p.2 ≤ d} :=
by
  sorry

end S_is_bounded_region_l1724_172436


namespace cubic_root_sum_l1724_172404

theorem cubic_root_sum (a b c : ℝ) : 
  (0 < a ∧ a < 1) → (0 < b ∧ b < 1) → (0 < c ∧ c < 1) →
  a ≠ b → b ≠ c → a ≠ c →
  20 * a^3 - 34 * a^2 + 15 * a - 1 = 0 →
  20 * b^3 - 34 * b^2 + 15 * b - 1 = 0 →
  20 * c^3 - 34 * c^2 + 15 * c - 1 = 0 →
  1 / (1 - a) + 1 / (1 - b) + 1 / (1 - c) = 1.3 := by
  sorry

end cubic_root_sum_l1724_172404


namespace max_difference_l1724_172429

theorem max_difference (a b : ℝ) (h1 : a < 0) 
  (h2 : ∀ x ∈ Set.Ioo a b, (3 * x^2 + a) * (2 * x + b) ≥ 0) :
  b - a ≤ 1/3 :=
sorry

end max_difference_l1724_172429


namespace equation_solution_l1724_172440

theorem equation_solution : 
  ∃ (x : ℝ), x ≠ 2 ∧ (4*x^2 + 3*x + 2) / (x - 2) = 4*x + 5 → x = -2 :=
by sorry

end equation_solution_l1724_172440


namespace data_median_and_variance_l1724_172481

def data : List ℝ := [5, 9, 8, 8, 10]

def median (l : List ℝ) : ℝ := sorry

def variance (l : List ℝ) : ℝ := sorry

theorem data_median_and_variance :
  median data = 8 ∧ variance data = 2.8 := by sorry

end data_median_and_variance_l1724_172481


namespace product_of_1010_2_and_102_3_l1724_172478

/-- Converts a binary number represented as a list of digits to its decimal value -/
def binary_to_decimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * 2^i) 0

/-- Converts a ternary number represented as a list of digits to its decimal value -/
def ternary_to_decimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * 3^i) 0

theorem product_of_1010_2_and_102_3 : 
  let binary_num := [0, 1, 0, 1]  -- 1010 in binary, least significant bit first
  let ternary_num := [2, 0, 1]    -- 102 in ternary, least significant digit first
  (binary_to_decimal binary_num) * (ternary_to_decimal ternary_num) = 110 := by
  sorry

end product_of_1010_2_and_102_3_l1724_172478


namespace intersection_implies_a_geq_two_l1724_172496

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {x : ℝ | x < a}
def B : Set ℝ := {x : ℝ | x^2 - 3*x + 2 < 0}

-- State the theorem
theorem intersection_implies_a_geq_two (a : ℝ) (h : A a ∩ B = B) : a ≥ 2 := by
  sorry

end intersection_implies_a_geq_two_l1724_172496


namespace vector_coordinates_proof_l1724_172445

/-- Given points A, B, C in a 2D plane, and points M and N satisfying certain conditions,
    prove that M, N, and vector MN have specific coordinates. -/
theorem vector_coordinates_proof (A B C M N : ℝ × ℝ) : 
  A = (-2, 4) → 
  B = (3, -1) → 
  C = (-3, -4) → 
  M - C = 3 • (A - C) → 
  N - C = 2 • (B - C) → 
  M = (0, 20) ∧ 
  N = (9, 2) ∧ 
  N - M = (9, -18) := by
  sorry

end vector_coordinates_proof_l1724_172445


namespace subtract_fractions_l1724_172423

theorem subtract_fractions : (3 : ℚ) / 4 - (1 : ℚ) / 8 = (5 : ℚ) / 8 := by
  sorry

end subtract_fractions_l1724_172423


namespace odot_calculation_l1724_172422

-- Define the ⊙ operation
def odot (a b : ℤ) : ℤ := a * b - (a + b)

-- State the theorem
theorem odot_calculation : odot 6 (odot 5 4) = 49 := by
  sorry

end odot_calculation_l1724_172422


namespace sum_of_456_terms_l1724_172433

/-- An arithmetic progression with first term 2 and sum of second and third terms 13 -/
def ArithmeticProgression (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, 
    (a 1 = 2) ∧ 
    (a 2 + a 3 = 13) ∧ 
    ∀ n : ℕ, a (n + 1) = a n + d

/-- The sum of the 4th, 5th, and 6th terms of the arithmetic progression is 42 -/
theorem sum_of_456_terms (a : ℕ → ℝ) (h : ArithmeticProgression a) : 
  a 4 + a 5 + a 6 = 42 := by
  sorry

end sum_of_456_terms_l1724_172433


namespace card_collection_average_l1724_172425

-- Define the sum of integers from 1 to n
def sum_to_n (n : ℕ) : ℕ := n * (n + 1) / 2

-- Define the sum of squares from 1 to n
def sum_of_squares (n : ℕ) : ℕ := n * (n + 1) * (2 * n + 1) / 6

theorem card_collection_average (n : ℕ) : 
  (sum_of_squares n : ℚ) / (sum_to_n n : ℚ) = 5050 → n = 7575 := by
  sorry

end card_collection_average_l1724_172425


namespace queen_secondary_teachers_queen_secondary_teachers_count_l1724_172439

/-- Calculates the number of teachers required at Queen Secondary School -/
theorem queen_secondary_teachers (total_students : ℕ) (classes_per_student : ℕ) 
  (students_per_class : ℕ) (classes_per_teacher : ℕ) : ℕ :=
  let total_class_instances := total_students * classes_per_student
  let unique_classes := total_class_instances / students_per_class
  unique_classes / classes_per_teacher

/-- Proves that the number of teachers at Queen Secondary School is 48 -/
theorem queen_secondary_teachers_count : 
  queen_secondary_teachers 1500 4 25 5 = 48 := by
  sorry

end queen_secondary_teachers_queen_secondary_teachers_count_l1724_172439


namespace inequality_proof_l1724_172455

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + 2*b = 1) : 
  a * b ≤ 1/8 ∧ 1/a + 2/b ≥ 9 := by
sorry

end inequality_proof_l1724_172455


namespace remainder_876539_mod_7_l1724_172447

theorem remainder_876539_mod_7 : 876539 % 7 = 6 := by
  sorry

end remainder_876539_mod_7_l1724_172447


namespace danielle_popsicle_sticks_l1724_172428

/-- Calculates the number of remaining popsicle sticks after making popsicles -/
def remaining_popsicle_sticks (total_money : ℕ) (mold_cost : ℕ) (stick_pack_cost : ℕ) 
  (stick_pack_size : ℕ) (juice_cost : ℕ) (popsicles_per_juice : ℕ) : ℕ :=
  let remaining_money := total_money - mold_cost - stick_pack_cost
  let juice_bottles := remaining_money / juice_cost
  let popsicles_made := juice_bottles * popsicles_per_juice
  stick_pack_size - popsicles_made

/-- Proves that Danielle will be left with 40 popsicle sticks -/
theorem danielle_popsicle_sticks : 
  remaining_popsicle_sticks 10 3 1 100 2 20 = 40 := by
  sorry

end danielle_popsicle_sticks_l1724_172428


namespace david_scott_age_difference_l1724_172464

/-- Represents the ages of the three sons -/
structure Ages where
  richard : ℕ
  david : ℕ
  scott : ℕ

/-- The conditions of the problem -/
def problem_conditions (ages : Ages) : Prop :=
  ages.richard = ages.david + 6 ∧
  ages.david > ages.scott ∧
  ages.richard + 8 = 2 * (ages.scott + 8) ∧
  ages.david = 11 + 3

/-- The theorem stating that David is 8 years older than Scott -/
theorem david_scott_age_difference (ages : Ages) : 
  problem_conditions ages → ages.david - ages.scott = 8 := by
  sorry

end david_scott_age_difference_l1724_172464


namespace stickers_per_sheet_l1724_172415

theorem stickers_per_sheet 
  (initial_stickers : ℕ) 
  (shared_stickers : ℕ) 
  (remaining_sheets : ℕ) 
  (h1 : initial_stickers = 150)
  (h2 : shared_stickers = 100)
  (h3 : remaining_sheets = 5)
  (h4 : initial_stickers ≥ shared_stickers)
  (h5 : remaining_sheets > 0) :
  (initial_stickers - shared_stickers) / remaining_sheets = 10 :=
by sorry

end stickers_per_sheet_l1724_172415


namespace non_officers_count_correct_l1724_172462

/-- Represents the number of non-officers in an office -/
def num_non_officers : ℕ := 450

/-- Represents the number of officers in an office -/
def num_officers : ℕ := 15

/-- Average salary of all employees in rupees per month -/
def avg_salary_all : ℚ := 120

/-- Average salary of officers in rupees per month -/
def avg_salary_officers : ℚ := 420

/-- Average salary of non-officers in rupees per month -/
def avg_salary_non_officers : ℚ := 110

/-- Theorem stating that the number of non-officers is correct given the salary information -/
theorem non_officers_count_correct :
  (num_officers * avg_salary_officers + num_non_officers * avg_salary_non_officers) / 
  (num_officers + num_non_officers : ℚ) = avg_salary_all := by
  sorry


end non_officers_count_correct_l1724_172462


namespace not_all_greater_than_quarter_l1724_172486

theorem not_all_greater_than_quarter (a b c : ℝ) 
  (ha : 0 < a ∧ a < 1) (hb : 0 < b ∧ b < 1) (hc : 0 < c ∧ c < 1) : 
  ¬ ((1 - a) * b > 1/4 ∧ (1 - b) * c > 1/4 ∧ (1 - c) * a > 1/4) := by
  sorry

end not_all_greater_than_quarter_l1724_172486


namespace pacos_marble_purchase_l1724_172458

theorem pacos_marble_purchase : 
  0.3333333333333333 + 0.3333333333333333 + 0.08333333333333333 = 0.75 := by
  sorry

end pacos_marble_purchase_l1724_172458


namespace complement_of_equal_sets_l1724_172411

def U : Set Nat := {1, 3}
def A : Set Nat := {1, 3}

theorem complement_of_equal_sets :
  (U \ A : Set Nat) = ∅ :=
sorry

end complement_of_equal_sets_l1724_172411


namespace factorization_equality_l1724_172491

theorem factorization_equality (x y : ℝ) : 
  (x - y)^2 - (3*x^2 - 3*x*y + y^2) = x*(y - 2*x) := by
  sorry

end factorization_equality_l1724_172491


namespace concentric_circles_ratio_l1724_172492

theorem concentric_circles_ratio (r R k : ℝ) (hr : r > 0) (hR : R > r) (hk : k > 0) :
  (π * R^2 - π * r^2) = k * (π * r^2) → R / r = Real.sqrt (k + 1) := by
  sorry

end concentric_circles_ratio_l1724_172492


namespace black_cubes_removed_multiple_of_four_l1724_172444

/-- Represents a cube constructed from unit cubes of two colors -/
structure ColoredCube where
  edge_length : ℕ
  black_cubes : ℕ
  white_cubes : ℕ
  adjacent_different : Bool

/-- Represents the removal of unit cubes from a ColoredCube -/
structure CubeRemoval where
  cube : ColoredCube
  removed_cubes : ℕ
  rods_affected : ℕ
  cubes_per_rod : ℕ

/-- Theorem stating that the number of black cubes removed is a multiple of 4 -/
theorem black_cubes_removed_multiple_of_four (removal : CubeRemoval) : 
  removal.cube.edge_length = 10 ∧ 
  removal.cube.black_cubes = 500 ∧ 
  removal.cube.white_cubes = 500 ∧
  removal.cube.adjacent_different = true ∧
  removal.removed_cubes = 100 ∧
  removal.rods_affected = 300 ∧
  removal.cubes_per_rod = 1 →
  ∃ (k : ℕ), (removal.removed_cubes - k) % 4 = 0 := by
  sorry

end black_cubes_removed_multiple_of_four_l1724_172444


namespace square_property_l1724_172417

theorem square_property (n : ℕ) :
  (∃ (d : Finset ℕ), d.card = 6 ∧ ∀ x ∈ d, x ∣ (n^5 + n^4 + 1)) →
  ∃ k : ℕ, n^3 - n + 1 = k^2 := by
  sorry

end square_property_l1724_172417


namespace ellipse_sum_theorem_l1724_172470

/-- Theorem: For an ellipse with given parameters, the sum h + k + a + b + 2c equals 9 + 2√33 -/
theorem ellipse_sum_theorem (h k a b c : ℝ) : 
  h = 3 → 
  k = -5 → 
  a = 7 → 
  b = 4 → 
  c = Real.sqrt (a^2 - b^2) → 
  h + k + a + b + 2*c = 9 + 2 * Real.sqrt 33 := by
  sorry

end ellipse_sum_theorem_l1724_172470


namespace line_parallel_to_intersection_l1724_172456

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel relation between lines and planes
variable (parallel_line_plane : Line → Plane → Prop)

-- Define the parallel relation between lines
variable (parallel_line_line : Line → Line → Prop)

-- Define the intersection of planes
variable (intersection : Plane → Plane → Line)

-- Theorem statement
theorem line_parallel_to_intersection
  (a b : Line) (α β : Plane)
  (h1 : a ≠ b)
  (h2 : α ≠ β)
  (h3 : parallel_line_plane a α)
  (h4 : parallel_line_plane a β)
  (h5 : intersection α β = b) :
  parallel_line_line a b :=
sorry

end line_parallel_to_intersection_l1724_172456


namespace initial_weight_calculation_calvins_initial_weight_l1724_172497

/-- Calculates the initial weight of a person given their weight loss rate and final weight --/
theorem initial_weight_calculation 
  (weight_loss_per_month : ℕ) 
  (months : ℕ) 
  (final_weight : ℕ) : ℕ :=
  let total_weight_loss := weight_loss_per_month * months
  final_weight + total_weight_loss

/-- Proves that given the conditions, the initial weight was 250 pounds --/
theorem calvins_initial_weight :
  let weight_loss_per_month : ℕ := 8
  let months : ℕ := 12
  let final_weight : ℕ := 154
  initial_weight_calculation weight_loss_per_month months final_weight = 250 := by
  sorry

end initial_weight_calculation_calvins_initial_weight_l1724_172497


namespace point_A_in_second_quadrant_l1724_172487

/-- A point in the second quadrant has a negative x-coordinate and a positive y-coordinate -/
def second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0

/-- The x-coordinate of point A -/
def x_coord (x : ℝ) : ℝ := 6 - 2*x

/-- The y-coordinate of point A -/
def y_coord (x : ℝ) : ℝ := x - 5

/-- Theorem: If point A(6-2x, x-5) lies in the second quadrant, then x > 5 -/
theorem point_A_in_second_quadrant (x : ℝ) :
  second_quadrant (x_coord x) (y_coord x) → x > 5 := by
  sorry

end point_A_in_second_quadrant_l1724_172487


namespace symmetric_point_about_x_axis_l1724_172408

/-- Given a point P(3, -4), prove that its symmetric point P' about the x-axis has coordinates (3, 4) -/
theorem symmetric_point_about_x_axis :
  let P : ℝ × ℝ := (3, -4)
  let P' : ℝ × ℝ := (P.1, -P.2)
  P' = (3, 4) := by sorry

end symmetric_point_about_x_axis_l1724_172408


namespace expression_simplification_l1724_172465

theorem expression_simplification (b : ℝ) :
  ((3 * b - 3) - 5 * b) / 3 = -2/3 * b - 1 := by sorry

end expression_simplification_l1724_172465


namespace prob_A_squared_zero_correct_l1724_172474

/-- Probability that A² = O for an n × n matrix A with exactly two 1's -/
def prob_A_squared_zero (n : ℕ) : ℚ :=
  if n < 2 then 0
  else (n - 1) * (n - 2) / (n * (n + 1))

/-- Theorem stating the probability that A² = O for the given conditions -/
theorem prob_A_squared_zero_correct (n : ℕ) (h : n ≥ 2) :
  prob_A_squared_zero n = (n - 1) * (n - 2) / (n * (n + 1)) :=
sorry

end prob_A_squared_zero_correct_l1724_172474


namespace no_integer_roots_l1724_172437

/-- A polynomial with integer coefficients -/
def IntPolynomial := ℕ → ℤ

/-- Evaluate a polynomial at a given integer -/
def eval (p : IntPolynomial) (x : ℤ) : ℤ :=
  sorry

theorem no_integer_roots (P : IntPolynomial) 
  (h2020 : eval P 2020 = 2021) 
  (h2021 : eval P 2021 = 2021) : 
  ∀ x : ℤ, eval P x ≠ 0 := by
  sorry

end no_integer_roots_l1724_172437


namespace sarah_trucks_l1724_172490

theorem sarah_trucks (trucks_to_jeff trucks_to_amy trucks_left : ℕ) 
  (h1 : trucks_to_jeff = 13)
  (h2 : trucks_to_amy = 21)
  (h3 : trucks_left = 38) :
  trucks_to_jeff + trucks_to_amy + trucks_left = 72 := by
  sorry

end sarah_trucks_l1724_172490


namespace quadratic_root_sum_l1724_172407

theorem quadratic_root_sum (m n : ℝ) : 
  (∃ x : ℝ, x^2 + m*x + 2*n = 0 ∧ x = 2) → m + n = -2 := by
  sorry

end quadratic_root_sum_l1724_172407


namespace circle_equation_theorem_l1724_172450

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents the coefficients of a circle equation -/
structure CircleCoefficients where
  D : ℝ
  E : ℝ
  F : ℝ

/-- Checks if a point lies on a circle with given coefficients -/
def pointOnCircle (p : Point) (c : CircleCoefficients) : Prop :=
  p.x^2 + p.y^2 + c.D * p.x + c.E * p.y + c.F = 0

/-- The theorem stating that the given equation represents the circle passing through the specified points -/
theorem circle_equation_theorem (p1 p2 p3 : Point) : 
  p1 = ⟨0, 0⟩ → 
  p2 = ⟨4, 0⟩ → 
  p3 = ⟨-1, 1⟩ → 
  ∃ (c : CircleCoefficients), 
    c.D = -4 ∧ c.E = -6 ∧ c.F = 0 ∧ 
    pointOnCircle p1 c ∧ 
    pointOnCircle p2 c ∧ 
    pointOnCircle p3 c :=
by sorry

end circle_equation_theorem_l1724_172450


namespace river_name_proof_l1724_172435

theorem river_name_proof :
  ∃! (x y z : ℕ),
    x + y + z = 35 ∧
    x - y = y - (z + 1) ∧
    (x + 3) * z = y^2 ∧
    x = 5 ∧ y = 12 ∧ z = 18 := by
  sorry

end river_name_proof_l1724_172435


namespace max_percentage_both_amenities_l1724_172466

/-- Represents the percentage of companies with type A planes -/
def percentage_type_A : ℝ := 0.4

/-- Represents the percentage of companies with type B planes -/
def percentage_type_B : ℝ := 0.6

/-- Represents the percentage of type A planes with wireless internet -/
def wireless_A : ℝ := 0.8

/-- Represents the percentage of type B planes with wireless internet -/
def wireless_B : ℝ := 0.1

/-- Represents the percentage of type A planes offering free snacks -/
def snacks_A : ℝ := 0.9

/-- Represents the percentage of type B planes offering free snacks -/
def snacks_B : ℝ := 0.5

/-- Theorem stating the maximum percentage of companies offering both amenities -/
theorem max_percentage_both_amenities :
  let max_both_A := min wireless_A snacks_A
  let max_both_B := min wireless_B snacks_B
  let max_percentage := percentage_type_A * max_both_A + percentage_type_B * max_both_B
  max_percentage = 0.38 := by sorry

end max_percentage_both_amenities_l1724_172466


namespace function_symmetry_l1724_172472

/-- The function f(x) = 2sin(4x + π/4) is symmetric about the point (-π/16, 0) -/
theorem function_symmetry (x : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ 2 * Real.sin (4 * x + π / 4)
  ∀ y : ℝ, f ((-π/16) + y) = f ((-π/16) - y) :=
by sorry

end function_symmetry_l1724_172472


namespace sum_equals_five_l1724_172427

/-- Definition of the star operation -/
def star (a b : ℕ) : ℤ := a^b - a*b

/-- Theorem statement -/
theorem sum_equals_five (a b : ℕ) (ha : a ≥ 2) (hb : b ≥ 2) (h : star a b = 3) : a + b = 5 := by
  sorry

end sum_equals_five_l1724_172427


namespace platform_length_l1724_172471

/-- Given a train of length 300 m that crosses a platform in 39 seconds
    and a signal pole in 12 seconds, the length of the platform is 675 m. -/
theorem platform_length (train_length : ℝ) (platform_crossing_time : ℝ) (pole_crossing_time : ℝ)
  (h1 : train_length = 300)
  (h2 : platform_crossing_time = 39)
  (h3 : pole_crossing_time = 12) :
  let train_speed := train_length / pole_crossing_time
  let platform_length := train_speed * platform_crossing_time - train_length
  platform_length = 675 := by
  sorry

end platform_length_l1724_172471


namespace ones_digit_73_pow_351_l1724_172461

/-- The ones digit of a natural number -/
def onesDigit (n : ℕ) : ℕ := n % 10

/-- The ones digit pattern for powers of 3 -/
def onesDigitPattern : List ℕ := [3, 9, 7, 1]

theorem ones_digit_73_pow_351 : onesDigit (73^351) = 7 := by
  sorry

end ones_digit_73_pow_351_l1724_172461


namespace line_circle_intersection_l1724_172495

/-- Given a point (a, b) outside a circle and a line ax + by = r^2, 
    prove that the line intersects the circle but doesn't pass through the center. -/
theorem line_circle_intersection (a b r : ℝ) (h : a^2 + b^2 > r^2) :
  ∃ (x y : ℝ), x^2 + y^2 = r^2 ∧ a*x + b*y = r^2 ∧ (x ≠ 0 ∨ y ≠ 0) := by
  sorry

end line_circle_intersection_l1724_172495


namespace clerks_count_l1724_172443

/-- Represents the grocery store employee structure and salaries --/
structure GroceryStore where
  manager_salary : ℕ
  clerk_salary : ℕ
  num_managers : ℕ
  total_salary : ℕ

/-- Calculates the number of clerks in the grocery store --/
def num_clerks (store : GroceryStore) : ℕ :=
  (store.total_salary - store.manager_salary * store.num_managers) / store.clerk_salary

/-- Theorem stating that the number of clerks is 3 given the conditions --/
theorem clerks_count (store : GroceryStore) 
    (h1 : store.manager_salary = 5)
    (h2 : store.clerk_salary = 2)
    (h3 : store.num_managers = 2)
    (h4 : store.total_salary = 16) : 
  num_clerks store = 3 := by
  sorry

end clerks_count_l1724_172443


namespace shadow_length_l1724_172406

theorem shadow_length (h₁ s₁ h₂ : ℝ) (h_h₁ : h₁ = 20) (h_s₁ : s₁ = 10) (h_h₂ : h₂ = 40) :
  ∃ s₂ : ℝ, s₂ = 20 ∧ h₁ / s₁ = h₂ / s₂ :=
by sorry

end shadow_length_l1724_172406


namespace add_point_three_to_twenty_nine_point_eight_l1724_172469

theorem add_point_three_to_twenty_nine_point_eight : 
  29.8 + 0.3 = 30.1 := by
  sorry

end add_point_three_to_twenty_nine_point_eight_l1724_172469


namespace sarah_father_age_double_l1724_172416

/-- Given Sarah's age and her father's age in 2010, find the year when the father's age will be double Sarah's age -/
theorem sarah_father_age_double (sarah_age_2010 : ℕ) (father_age_2010 : ℕ) 
  (h1 : sarah_age_2010 = 10)
  (h2 : father_age_2010 = 6 * sarah_age_2010) :
  ∃ (year : ℕ), 
    year > 2010 ∧ 
    (father_age_2010 + (year - 2010)) = 2 * (sarah_age_2010 + (year - 2010)) ∧
    year = 2030 :=
by sorry

end sarah_father_age_double_l1724_172416


namespace hyperbola_asymptotes_l1724_172409

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := y^2 / 4 - x^2 / 9 = 1

-- Define the asymptotes
def asymptotes (x y : ℝ) : Prop := 2*x + 3*y = 0 ∨ 2*x - 3*y = 0

-- Theorem statement
theorem hyperbola_asymptotes : 
  ∀ x y : ℝ, hyperbola x y → asymptotes x y :=
sorry

end hyperbola_asymptotes_l1724_172409


namespace friends_assignment_count_l1724_172493

theorem friends_assignment_count : 
  (∀ n : ℕ, n > 0 → n ^ 8 = (n * n ^ 7)) →
  4 ^ 8 = 65536 := by
  sorry

end friends_assignment_count_l1724_172493


namespace number_of_girls_l1724_172498

def number_of_boys : ℕ := 5
def committee_size : ℕ := 4
def boys_in_committee : ℕ := 2
def girls_in_committee : ℕ := 2
def total_committees : ℕ := 150

def choose (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

theorem number_of_girls : 
  ∃ g : ℕ, 
    choose number_of_boys boys_in_committee * choose g girls_in_committee = total_committees ∧ 
    g = 6 :=
sorry

end number_of_girls_l1724_172498


namespace card_house_47_floors_l1724_172432

/-- The number of cards needed for the nth floor of a card house -/
def cards_for_floor (n : ℕ) : ℕ := 2 + (n - 1) * 3

/-- The total number of cards needed for a card house with n floors -/
def total_cards (n : ℕ) : ℕ := 
  n * (cards_for_floor 1 + cards_for_floor n) / 2

/-- Theorem: A card house with 47 floors requires 3337 cards -/
theorem card_house_47_floors : total_cards 47 = 3337 := by
  sorry

end card_house_47_floors_l1724_172432
