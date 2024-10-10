import Mathlib

namespace certain_number_proof_l181_18162

theorem certain_number_proof : ∃ x : ℝ, (x / 3 = 400 * 1.005) ∧ (x = 1206) := by
  sorry

end certain_number_proof_l181_18162


namespace regular_polygon_sides_l181_18196

theorem regular_polygon_sides (D : ℕ) (n : ℕ) : D = 15 → n * (n - 3) / 2 = D → n = 8 := by
  sorry

end regular_polygon_sides_l181_18196


namespace inverse_proportion_m_value_l181_18157

def is_inverse_proportion (f : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, ∀ x : ℝ, x ≠ 0 → f x = k / x

theorem inverse_proportion_m_value (m : ℝ) :
  (is_inverse_proportion (λ x => (m - 1) * x^(|m| - 2))) →
  m = -1 :=
by
  sorry

end inverse_proportion_m_value_l181_18157


namespace all_fractions_repeat_l181_18101

theorem all_fractions_repeat (n : ℕ) (h1 : 1 ≤ n) (h2 : n ≤ 20) : 
  ¬ (∃ k : ℕ, n * (5^k * 2^k) = 42 * m) :=
sorry

end all_fractions_repeat_l181_18101


namespace purely_imaginary_square_root_l181_18134

theorem purely_imaginary_square_root (a : ℝ) : 
  (∃ b : ℝ, (a - Complex.I) ^ 2 = Complex.I * b ∧ b ≠ 0) → (a = 1 ∨ a = -1) := by
  sorry

end purely_imaginary_square_root_l181_18134


namespace complex_purely_imaginary_m_l181_18121

def is_purely_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

theorem complex_purely_imaginary_m (m : ℝ) :
  is_purely_imaginary ((m^2 - m) + m * I) → m = 1 := by sorry

end complex_purely_imaginary_m_l181_18121


namespace rational_function_property_l181_18150

theorem rational_function_property (f : ℚ → ℚ) 
  (h1 : f 1 = 2) 
  (h2 : ∀ x y : ℚ, f (x * y) = f x * f y - f (x + y) + 1) : 
  ∀ x : ℚ, f x = x + 1 := by sorry

end rational_function_property_l181_18150


namespace tan_75_degrees_l181_18135

theorem tan_75_degrees : Real.tan (75 * π / 180) = 2 + Real.sqrt 3 := by
  sorry

end tan_75_degrees_l181_18135


namespace shareholders_profit_decrease_l181_18122

/-- Represents the problem of calculating the percentage decrease in shareholders' profit --/
theorem shareholders_profit_decrease (total_machines : ℝ) (operational_machines : ℝ) 
  (annual_output : ℝ) (profit_percentage : ℝ) :
  total_machines = 14 →
  operational_machines = total_machines - 7.14 →
  annual_output = 70000 →
  profit_percentage = 0.125 →
  let new_output := (operational_machines / total_machines) * annual_output
  let original_profit := profit_percentage * annual_output
  let new_profit := profit_percentage * new_output
  let percentage_decrease := ((original_profit - new_profit) / original_profit) * 100
  percentage_decrease = 51 := by
sorry

end shareholders_profit_decrease_l181_18122


namespace necessary_not_sufficient_p_for_q_necessary_not_sufficient_not_p_for_not_q_l181_18179

-- Define the conditions
def p (x : ℝ) : Prop := -x^2 + 7*x + 8 ≥ 0
def q (x m : ℝ) : Prop := x^2 - 2*x + 1 - 4*m^2 ≤ 0

-- Theorem 1
theorem necessary_not_sufficient_p_for_q (m : ℝ) :
  (m > 0) →
  (∀ x, q x m → p x) ∧ (∃ x, p x ∧ ¬q x m) →
  m ≥ 7/2 :=
sorry

-- Theorem 2
theorem necessary_not_sufficient_not_p_for_not_q (m : ℝ) :
  (m > 0) →
  (∀ x, ¬p x → ¬q x m) ∧ (∃ x, ¬q x m ∧ p x) →
  1 ≤ m ∧ m ≤ 7/2 :=
sorry

end necessary_not_sufficient_p_for_q_necessary_not_sufficient_not_p_for_not_q_l181_18179


namespace triangle_side_and_area_l181_18155

theorem triangle_side_and_area 
  (A B C : Real) -- Angles
  (a b c : Real) -- Sides
  (h1 : b = Real.sqrt 7)
  (h2 : c = 1)
  (h3 : B = 2 * π / 3) -- 120° in radians
  (h4 : 0 < a ∧ 0 < b ∧ 0 < c) -- Triangle inequality
  (h5 : b^2 = a^2 + c^2 - 2*a*c*Real.cos B) -- Cosine rule
  : a = 2 ∧ (1/2) * a * c * Real.sin B = Real.sqrt 3 / 2 := by
  sorry

end triangle_side_and_area_l181_18155


namespace fraction_product_simplification_l181_18183

theorem fraction_product_simplification :
  (6 : ℚ) / 3 * (9 : ℚ) / 6 * (12 : ℚ) / 9 * (15 : ℚ) / 12 = 5 := by
  sorry

end fraction_product_simplification_l181_18183


namespace first_watermelon_weight_l181_18114

theorem first_watermelon_weight (total_weight second_weight : ℝ) 
  (h1 : total_weight = 14.02)
  (h2 : second_weight = 4.11) :
  total_weight - second_weight = 9.91 := by
  sorry

end first_watermelon_weight_l181_18114


namespace product_of_coefficients_l181_18148

theorem product_of_coefficients (x y z w A B : ℝ) 
  (eq1 : 4 * x * z + y * w = 3)
  (eq2 : x * w + y * z = 6)
  (eq3 : (A * x + y) * (B * z + w) = 15) :
  A * B = 4 := by sorry

end product_of_coefficients_l181_18148


namespace quadratic_inequality_solution_l181_18141

theorem quadratic_inequality_solution (a : ℝ) : 
  (∀ x : ℝ, ax^2 - 2*x + 2 > 0 ↔ -1/2 < x ∧ x < 1/3) → a = -12 :=
by sorry

end quadratic_inequality_solution_l181_18141


namespace a_101_value_l181_18165

def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  a 1 = 2 ∧ ∀ n : ℕ, a (n + 1) - a n = 1 / 2

theorem a_101_value (a : ℕ → ℚ) (h : arithmetic_sequence a) : a 101 = 52 := by
  sorry

end a_101_value_l181_18165


namespace glasses_in_larger_box_l181_18176

theorem glasses_in_larger_box 
  (small_box : ℕ) 
  (total_boxes : ℕ) 
  (average_glasses : ℕ) :
  small_box = 12 → 
  total_boxes = 2 → 
  average_glasses = 15 → 
  ∃ large_box : ℕ, 
    (small_box + large_box) / total_boxes = average_glasses ∧ 
    large_box = 18 := by
sorry

end glasses_in_larger_box_l181_18176


namespace rectangular_solid_surface_area_l181_18170

/-- The total surface area of a rectangular solid -/
def totalSurfaceArea (length width depth : ℝ) : ℝ :=
  2 * (length * width + width * depth + length * depth)

/-- Theorem: The total surface area of a rectangular solid with length 10 meters, 
    width 9 meters, and depth 6 meters is 408 square meters -/
theorem rectangular_solid_surface_area :
  totalSurfaceArea 10 9 6 = 408 := by
  sorry

end rectangular_solid_surface_area_l181_18170


namespace line_decreasing_direct_proportion_range_l181_18133

/-- A line passing through two points -/
structure Line where
  k : ℝ
  b : ℝ
  x₁ : ℝ
  y₁ : ℝ
  x₂ : ℝ
  y₂ : ℝ
  eq₁ : y₁ = k * x₁ + b
  eq₂ : y₂ = k * x₂ + b

/-- A direct proportion function passing through two points -/
structure DirectProportion where
  m : ℝ
  x₁ : ℝ
  y₁ : ℝ
  x₂ : ℝ
  y₂ : ℝ
  eq₁ : y₁ = (1 - 2*m) * x₁
  eq₂ : y₂ = (1 - 2*m) * x₂

theorem line_decreasing (l : Line) (h₁ : l.k < 0) (h₂ : l.x₁ < l.x₂) : l.y₁ > l.y₂ := by
  sorry

theorem direct_proportion_range (d : DirectProportion) (h₁ : d.x₁ < d.x₂) (h₂ : d.y₁ > d.y₂) : d.m > 1/2 := by
  sorry

end line_decreasing_direct_proportion_range_l181_18133


namespace g_range_g_range_achieves_bounds_l181_18124

noncomputable def g (x : ℝ) : ℝ :=
  (Real.arccos (x/3))^2 + 2*Real.pi * Real.arcsin (x/3) - 3*(Real.arcsin (x/3))^2 + (Real.pi^2/4)*(x^2 - 3*x + 9)

theorem g_range :
  ∀ y ∈ Set.range g, π^2/4 ≤ y ∧ y ≤ 37*π^2/4 :=
by sorry

theorem g_range_achieves_bounds :
  ∃ x₁ x₂ : ℝ, x₁ ∈ Set.Icc (-3) 3 ∧ x₂ ∈ Set.Icc (-3) 3 ∧ 
    g x₁ = π^2/4 ∧ g x₂ = 37*π^2/4 :=
by sorry

end g_range_g_range_achieves_bounds_l181_18124


namespace sum_of_complex_numbers_l181_18100

theorem sum_of_complex_numbers : 
  (2 : ℂ) + 5*I + (3 : ℂ) - 7*I + (-1 : ℂ) + 2*I = 4 := by
  sorry

end sum_of_complex_numbers_l181_18100


namespace smallest_k_value_l181_18169

theorem smallest_k_value : ∃ (k : ℕ), k > 0 ∧
  (∀ (k' : ℕ), k' > 0 →
    (∃ (n : ℕ), n > 0 ∧ 2000 < n ∧ n < 3000 ∧
      (∀ (i : ℕ), 2 ≤ i ∧ i ≤ k' → n % i = i - 1)) →
    k ≤ k') ∧
  k = 9 :=
sorry

end smallest_k_value_l181_18169


namespace chinese_books_probability_l181_18103

theorem chinese_books_probability (total_books : ℕ) (chinese_books : ℕ) (math_books : ℕ) :
  total_books = chinese_books + math_books →
  chinese_books = 3 →
  math_books = 2 →
  (Nat.choose chinese_books 2 : ℚ) / (Nat.choose total_books 2) = 3 / 10 := by
  sorry

end chinese_books_probability_l181_18103


namespace candy_bar_savings_l181_18177

/-- Calculates the number of items saved given weekly receipt, consumption rate, and time period. -/
def items_saved (weekly_receipt : ℕ) (consumption_rate : ℕ) (weeks : ℕ) : ℕ :=
  weekly_receipt * weeks - (weeks / consumption_rate)

/-- Proves that under the given conditions, 28 items are saved after 16 weeks. -/
theorem candy_bar_savings : items_saved 2 4 16 = 28 := by
  sorry

end candy_bar_savings_l181_18177


namespace survey_result_l181_18118

theorem survey_result : ∀ (total : ℕ) (dangerous : ℕ) (fire : ℕ),
  (dangerous : ℚ) / total = 825 / 1000 →
  (fire : ℚ) / dangerous = 524 / 1000 →
  fire = 27 →
  total = 63 := by
sorry

end survey_result_l181_18118


namespace similar_triangle_perimeter_l181_18145

/-- Represents a triangle with side lengths a, b, and c. -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if a triangle satisfies the triangle inequality. -/
def Triangle.isValid (t : Triangle) : Prop :=
  t.a + t.b > t.c ∧ t.b + t.c > t.a ∧ t.c + t.a > t.b

/-- Checks if a triangle is isosceles. -/
def Triangle.isIsosceles (t : Triangle) : Prop :=
  t.a = t.b ∨ t.b = t.c ∨ t.c = t.a

/-- Calculates the perimeter of a triangle. -/
def Triangle.perimeter (t : Triangle) : ℝ :=
  t.a + t.b + t.c

/-- Checks if two triangles are similar. -/
def areSimilar (t1 t2 : Triangle) : Prop :=
  ∃ k : ℝ, k > 0 ∧ t2.a = k * t1.a ∧ t2.b = k * t1.b ∧ t2.c = k * t1.c

theorem similar_triangle_perimeter 
  (t1 t2 : Triangle) 
  (h1 : t1.isValid)
  (h2 : t2.isValid)
  (h3 : t1.isIsosceles)
  (h4 : t2.isIsosceles)
  (h5 : areSimilar t1 t2)
  (h6 : t1.a = 8 ∧ t1.b = 24 ∧ t1.c = 24)
  (h7 : t2.a = 40) : 
  t2.perimeter = 280 := by
  sorry

end similar_triangle_perimeter_l181_18145


namespace fish_population_estimate_l181_18187

/-- Estimates the fish population on January 1 based on capture-recapture data --/
theorem fish_population_estimate 
  (initial_tagged : ℕ)
  (june_sample : ℕ)
  (june_tagged : ℕ)
  (tagged_left_percent : ℚ)
  (new_juvenile_percent : ℚ) :
  initial_tagged = 100 →
  june_sample = 150 →
  june_tagged = 4 →
  tagged_left_percent = 30 / 100 →
  new_juvenile_percent = 50 / 100 →
  ∃ (estimated_population : ℕ), estimated_population = 1312 := by
  sorry


end fish_population_estimate_l181_18187


namespace seventh_selected_number_l181_18139

def random_sequence : List ℕ := [6572, 0802, 6319, 8702, 4369, 9728, 0198, 3204, 9243, 4935, 8200, 3623, 4869, 6938, 7481]

def is_valid (n : ℕ) : Bool := 1 ≤ n ∧ n ≤ 500

def select_valid_numbers (seq : List ℕ) : List ℕ :=
  seq.filter (λ n => is_valid (n % 1000))

theorem seventh_selected_number :
  (select_valid_numbers random_sequence).nthLe 6 sorry = 320 := by sorry

end seventh_selected_number_l181_18139


namespace vector_magnitude_problem_l181_18191

/-- Given vectors a and b in R^2, with b = (-1, 2) and their sum (1, 3), 
    prove that the magnitude of a - 2b is 5. -/
theorem vector_magnitude_problem (a b : ℝ × ℝ) : 
  b = (-1, 2) → a + b = (1, 3) → ‖a - 2 • b‖ = 5 := by
  sorry

end vector_magnitude_problem_l181_18191


namespace niko_sock_profit_l181_18131

theorem niko_sock_profit : ∀ (total_pairs : ℕ) (cost_per_pair : ℚ) 
  (profit_percent : ℚ) (profit_amount : ℚ) (high_profit_pairs : ℕ) (low_profit_pairs : ℕ),
  total_pairs = 9 →
  cost_per_pair = 2 →
  profit_percent = 25 / 100 →
  profit_amount = 1 / 5 →
  high_profit_pairs = 4 →
  low_profit_pairs = 5 →
  high_profit_pairs + low_profit_pairs = total_pairs →
  (high_profit_pairs : ℚ) * (cost_per_pair * profit_percent) + 
  (low_profit_pairs : ℚ) * profit_amount = 3 := by
sorry

end niko_sock_profit_l181_18131


namespace price_change_percentage_l181_18113

theorem price_change_percentage (P : ℝ) (x : ℝ) : 
  P * (1 + x/100) * (1 - x/100) = 0.64 * P → x = 60 := by
  sorry

end price_change_percentage_l181_18113


namespace square_difference_equality_l181_18136

theorem square_difference_equality : 1004^2 - 996^2 - 1000^2 + 1000^2 = 16000 := by
  sorry

end square_difference_equality_l181_18136


namespace product_of_three_numbers_l181_18143

theorem product_of_three_numbers (x y z n : ℝ) 
  (sum_eq : x + y + z = 200)
  (x_eq : 8 * x = n)
  (y_eq : y = n + 12)
  (z_eq : z = n - 12)
  (x_smallest : x < y ∧ x < z) : 
  x * y * z = 502147200 / 4913 := by
  sorry

end product_of_three_numbers_l181_18143


namespace prob_at_least_two_different_fruits_l181_18116

def num_meals : ℕ := 4
def num_fruits : ℕ := 4

def prob_same_fruit_all_day : ℚ := (1 / num_fruits) ^ num_meals * num_fruits

theorem prob_at_least_two_different_fruits :
  1 - prob_same_fruit_all_day = 63 / 64 := by
  sorry

end prob_at_least_two_different_fruits_l181_18116


namespace calculation_proof_l181_18102

theorem calculation_proof : (2013 : ℚ) / (25 * 52 - 46 * 15) * 10 = 33 := by
  sorry

end calculation_proof_l181_18102


namespace betty_rice_purchase_l181_18149

theorem betty_rice_purchase (o r : ℝ) : 
  (o ≥ 8 + r / 3 ∧ o ≤ 3 * r) → r ≥ 3 := by sorry

end betty_rice_purchase_l181_18149


namespace peters_remaining_money_l181_18144

/-- Represents Peter's shopping trips and calculates his remaining money -/
def petersShopping (initialAmount : ℚ) : ℚ :=
  let firstTripTax := 0.05
  let secondTripDiscount := 0.1

  let firstTripItems := [
    (6, 2),    -- potatoes
    (9, 3),    -- tomatoes
    (5, 4),    -- cucumbers
    (3, 5),    -- bananas
    (2, 3.5),  -- apples
    (7, 4.25), -- oranges
    (4, 6),    -- grapes
    (8, 5.5)   -- strawberries
  ]

  let secondTripItems := [
    (2, 1.5),  -- potatoes
    (5, 2.75)  -- tomatoes
  ]

  let firstTripCost := (firstTripItems.map (λ (k, p) => k * p)).sum * (1 + firstTripTax)
  let secondTripCost := (secondTripItems.map (λ (k, p) => k * p)).sum * (1 - secondTripDiscount)

  initialAmount - firstTripCost - secondTripCost

/-- Theorem stating that Peter's remaining money is $297.24 -/
theorem peters_remaining_money :
  petersShopping 500 = 297.24 := by
  sorry


end peters_remaining_money_l181_18144


namespace least_divisible_by_1920_eight_divisible_by_1920_eight_is_least_divisible_by_1920_l181_18128

theorem least_divisible_by_1920 (a : ℕ) : a^6 % 1920 = 0 → a ≥ 8 :=
sorry

theorem eight_divisible_by_1920 : 8^6 % 1920 = 0 :=
sorry

theorem eight_is_least_divisible_by_1920 : ∃ (a : ℕ), a^6 % 1920 = 0 ∧ ∀ (b : ℕ), b < a → b^6 % 1920 ≠ 0 :=
sorry

end least_divisible_by_1920_eight_divisible_by_1920_eight_is_least_divisible_by_1920_l181_18128


namespace traci_road_trip_l181_18189

/-- Proves that the fraction of the remaining distance traveled between the first and second stops is 1/4 -/
theorem traci_road_trip (total_distance : ℝ) (first_stop_fraction : ℝ) (final_leg : ℝ) : 
  total_distance = 600 →
  first_stop_fraction = 1/3 →
  final_leg = 300 →
  (total_distance - first_stop_fraction * total_distance - final_leg) / (total_distance - first_stop_fraction * total_distance) = 1/4 := by
  sorry

end traci_road_trip_l181_18189


namespace inequality_system_solution_l181_18107

-- Define the inequality system
def inequality_system (x : ℝ) : Prop :=
  (4 * x - 8 ≤ 0) ∧ ((x + 3) / 2 > 3 - x)

-- Define the solution set
def solution_set (x : ℝ) : Prop :=
  1 < x ∧ x ≤ 2

-- Theorem statement
theorem inequality_system_solution :
  ∀ x : ℝ, inequality_system x ↔ solution_set x :=
by sorry

end inequality_system_solution_l181_18107


namespace monotonicity_criterion_other_statements_incorrect_l181_18154

/-- A function f is monotonically decreasing on ℝ if for all x₁ < x₂, f(x₁) > f(x₂) -/
def MonotonicallyDecreasing (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂, x₁ < x₂ → f x₁ > f x₂

theorem monotonicity_criterion (f : ℝ → ℝ) :
  (∃ x₁ x₂, x₁ < x₂ ∧ f x₁ ≤ f x₂) → ¬(MonotonicallyDecreasing f) :=
by sorry

theorem other_statements_incorrect (f : ℝ → ℝ) :
  ¬(∀ x₁ x₂, x₁ < x₂ → f x₁ < f x₂ → (∀ y₁ y₂, y₁ < y₂ → f y₁ < f y₂)) ∧
  ¬(∀ x₂ > 0, (∀ x₁, f x₁ < f (x₁ + x₂)) → (∀ y₁ y₂, y₁ < y₂ → f y₁ < f y₂)) ∧
  ¬(∀ x₁ x₂, x₁ < x₂ → f x₁ ≥ f x₂ → MonotonicallyDecreasing f) :=
by sorry

end monotonicity_criterion_other_statements_incorrect_l181_18154


namespace additional_cars_during_play_l181_18178

/-- Calculates the number of additional cars that parked during a play given the initial conditions. -/
theorem additional_cars_during_play
  (front_initial : ℕ)
  (back_initial : ℕ)
  (total_end : ℕ)
  (h1 : front_initial = 100)
  (h2 : back_initial = 2 * front_initial)
  (h3 : total_end = 700) :
  total_end - (front_initial + back_initial) = 300 :=
by sorry

end additional_cars_during_play_l181_18178


namespace track_circumference_l181_18120

/-- Represents the circular track and the movement of A and B -/
structure TrackSystem where
  /-- Half of the track's circumference in yards -/
  half_circumference : ℝ
  /-- Speed of A in yards per unit time -/
  speed_a : ℝ
  /-- Speed of B in yards per unit time -/
  speed_b : ℝ

/-- The theorem stating the conditions and the result to be proven -/
theorem track_circumference (ts : TrackSystem) 
  (h1 : ts.speed_a > 0 ∧ ts.speed_b > 0)  -- A and B travel at uniform (positive) speeds
  (h2 : ts.speed_a + ts.speed_b = ts.half_circumference / 75)  -- They meet after B travels 150 yards
  (h3 : 2 * ts.half_circumference - 90 = (ts.half_circumference + 90) * (ts.speed_a / ts.speed_b)) 
      -- Second meeting condition
  : ts.half_circumference = 360 :=
sorry

end track_circumference_l181_18120


namespace smallest_additional_divisor_l181_18110

def divisors : Set Nat := {30, 48, 74, 100}

theorem smallest_additional_divisor :
  ∃ (n : Nat), n > 0 ∧ 
  (∀ m ∈ divisors, (44402 + 2) % m = 0) ∧
  (44402 + 2) % n = 0 ∧
  n ∉ divisors ∧
  (∀ k : Nat, 0 < k ∧ k < n → (44402 + 2) % k ≠ 0 ∨ k ∈ divisors) ∧
  n = 37 := by
  sorry

end smallest_additional_divisor_l181_18110


namespace x_one_value_l181_18129

theorem x_one_value (x₁ x₂ x₃ x₄ : ℝ) 
  (h_order : 0 ≤ x₄ ∧ x₄ ≤ x₃ ∧ x₃ ≤ x₂ ∧ x₂ ≤ x₁ ∧ x₁ ≤ 1)
  (h_equation : (1-x₁)^2 + (x₁-x₂)^2 + (x₂-x₃)^2 + (x₃-x₄)^2 + x₄^2 = 1/5) :
  x₁ = 4/5 := by
sorry

end x_one_value_l181_18129


namespace reporters_covering_local_politics_l181_18198

/-- The percentage of reporters who do not cover politics -/
def non_politics_reporters : ℝ := 85.71428571428572

/-- The percentage of reporters covering politics who do not cover local politics in country x -/
def non_local_politics_reporters : ℝ := 30

/-- The percentage of reporters covering local politics in country x -/
def local_politics_reporters : ℝ := 10

theorem reporters_covering_local_politics :
  local_politics_reporters = 
    (100 - non_politics_reporters) * (100 - non_local_politics_reporters) / 100 := by
  sorry

end reporters_covering_local_politics_l181_18198


namespace pool_filling_proof_l181_18125

/-- The rate at which the first hose sprays water in gallons per hour -/
def first_hose_rate : ℝ := 50

/-- The rate at which the second hose sprays water in gallons per hour -/
def second_hose_rate : ℝ := 70

/-- The capacity of the pool in gallons -/
def pool_capacity : ℝ := 390

/-- The time the first hose was used alone in hours -/
def first_hose_time : ℝ := 3

/-- The time both hoses were used together in hours -/
def both_hoses_time : ℝ := 2

theorem pool_filling_proof : 
  first_hose_rate * first_hose_time + 
  (first_hose_rate + second_hose_rate) * both_hoses_time = 
  pool_capacity := by sorry

end pool_filling_proof_l181_18125


namespace tutor_reunion_proof_l181_18168

/-- The number of school days until all tutors work together again -/
def tutor_reunion_days : ℕ := 360

/-- Elisa's work schedule (every 5th day) -/
def elisa_schedule : ℕ := 5

/-- Frank's work schedule (every 6th day) -/
def frank_schedule : ℕ := 6

/-- Giselle's work schedule (every 8th day) -/
def giselle_schedule : ℕ := 8

/-- Hector's work schedule (every 9th day) -/
def hector_schedule : ℕ := 9

theorem tutor_reunion_proof :
  Nat.lcm elisa_schedule (Nat.lcm frank_schedule (Nat.lcm giselle_schedule hector_schedule)) = tutor_reunion_days :=
by sorry

end tutor_reunion_proof_l181_18168


namespace daughter_and_child_weight_l181_18185

/-- The combined weight of a daughter and her child given specific family weight conditions -/
theorem daughter_and_child_weight (total_weight mother_weight daughter_weight child_weight : ℝ) :
  total_weight = mother_weight + daughter_weight + child_weight →
  child_weight = (1 / 5 : ℝ) * mother_weight →
  daughter_weight = 48 →
  total_weight = 120 →
  daughter_weight + child_weight = 60 :=
by
  sorry

#check daughter_and_child_weight

end daughter_and_child_weight_l181_18185


namespace repeating_decimal_subtraction_l181_18197

def repeating_decimal_234 : ℚ := 234 / 999
def repeating_decimal_567 : ℚ := 567 / 999
def repeating_decimal_891 : ℚ := 891 / 999

theorem repeating_decimal_subtraction :
  repeating_decimal_234 - repeating_decimal_567 - repeating_decimal_891 = -1224 / 999 := by
sorry

end repeating_decimal_subtraction_l181_18197


namespace six_digit_divisibility_l181_18159

/-- Represents a two-digit number -/
def two_digit_number := { n : ℕ | 10 ≤ n ∧ n < 100 }

/-- Constructs a six-digit number by repeating a two-digit number three times -/
def repeat_three_times (n : two_digit_number) : ℕ :=
  100000 * n + 1000 * n + n

theorem six_digit_divisibility (n : two_digit_number) :
  (repeat_three_times n) % 10101 = 0 := by
  sorry

end six_digit_divisibility_l181_18159


namespace arithmetic_sequence_inequality_l181_18111

theorem arithmetic_sequence_inequality (a b c : ℝ) (h1 : b - a = c - b) (h2 : b - a ≠ 0) :
  ¬ (∀ a b c : ℝ, a^3*b + b^3*c + c^3*a ≥ a^4 + b^4 + c^4) :=
sorry

end arithmetic_sequence_inequality_l181_18111


namespace min_value_of_f_l181_18153

/-- The quadratic function f(x) = x^2 + 12x + 36 -/
def f (x : ℝ) : ℝ := x^2 + 12*x + 36

/-- The minimum value of f(x) is 0 -/
theorem min_value_of_f : 
  ∃ (m : ℝ), ∀ (x : ℝ), f x ≥ m ∧ ∃ (x₀ : ℝ), f x₀ = m ∧ m = 0 :=
sorry

end min_value_of_f_l181_18153


namespace correct_addition_after_digit_change_l181_18173

theorem correct_addition_after_digit_change :
  let num1 : ℕ := 364765
  let num2 : ℕ := 951872
  let incorrect_sum : ℕ := 1496637
  let d : ℕ := 3
  let e : ℕ := 4
  let new_num1 : ℕ := num1 + 100000 * (e - d)
  let new_num2 : ℕ := num2
  let new_sum : ℕ := incorrect_sum + 100000 * (e - d)
  new_num1 + new_num2 = new_sum ∧ d + e = 7 :=
by sorry

end correct_addition_after_digit_change_l181_18173


namespace fraction_simplification_l181_18146

theorem fraction_simplification : (3 * 4) / 6 = 2 := by
  sorry

end fraction_simplification_l181_18146


namespace rob_doubles_l181_18138

/-- Rob has some baseball cards, and Jess has 5 times as many doubles as Rob. 
    Jess has 40 doubles baseball cards. -/
theorem rob_doubles (rob_cards : ℕ) (rob_doubles : ℕ) (jess_doubles : ℕ) 
    (h1 : rob_cards ≥ rob_doubles)
    (h2 : jess_doubles = 5 * rob_doubles)
    (h3 : jess_doubles = 40) : 
  rob_doubles = 8 := by
  sorry

end rob_doubles_l181_18138


namespace range_of_a_l181_18140

theorem range_of_a (a : ℝ) : 
  (∀ n : ℕ+, ((1 - a) * n - a) * Real.log a < 0) ↔ (0 < a ∧ a < 1/2) ∨ a > 1 :=
by sorry

end range_of_a_l181_18140


namespace x_one_value_l181_18160

theorem x_one_value (x₁ x₂ x₃ x₄ : ℝ) 
  (h_order : 0 ≤ x₄ ∧ x₄ ≤ x₃ ∧ x₃ ≤ x₂ ∧ x₂ ≤ x₁ ∧ x₁ ≤ 1)
  (h_equation : (1 - x₁)^2 + (x₁ - x₂)^2 + (x₂ - x₃)^2 + (x₃ - x₄)^2 + x₄^2 = 1/3) :
  x₁ = 4/5 := by
sorry

end x_one_value_l181_18160


namespace symmetric_points_ratio_l181_18163

/-- Given two points A and B symmetric about a line ax + y - b = 0, prove that a/b = 1/3 -/
theorem symmetric_points_ratio (a b : ℝ) : 
  let A : ℝ × ℝ := (-1, 3)
  let B : ℝ × ℝ := (3, 5)
  (∀ (x y : ℝ), (x = -1 ∧ y = 3) ∨ (x = 3 ∧ y = 5) → a * x + y - b = 0) →
  (∀ (x y : ℝ), a * x + y - b = 0 → a * ((3 - (-1))/2 + (-1)) + ((5 - 3)/2 + 3) - b = 0) →
  a / b = 1 / 3 := by
sorry

end symmetric_points_ratio_l181_18163


namespace smallest_angle_of_quadrilateral_l181_18174

theorem smallest_angle_of_quadrilateral (a b c d : ℝ) : 
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 →  -- All angles are positive
  a + b + c + d = 360 →           -- Sum of angles in a quadrilateral
  b = 4/3 * a →                   -- Ratio condition
  c = 5/3 * a →                   -- Ratio condition
  d = 2 * a →                     -- Ratio condition
  a = 60 ∧ a ≤ b ∧ a ≤ c ∧ a ≤ d  -- a is the smallest angle and equals 60°
  := by sorry

end smallest_angle_of_quadrilateral_l181_18174


namespace hooligan_theorem_l181_18193

-- Define the universe
variable (Person : Type)

-- Define predicates
variable (isHooligan : Person → Prop)
variable (hasBeatlesHaircut : Person → Prop)
variable (hasRudeDemeanor : Person → Prop)

-- State the theorem
theorem hooligan_theorem 
  (exists_beatles_hooligan : ∃ x, isHooligan x ∧ hasBeatlesHaircut x)
  (all_hooligans_rude : ∀ y, isHooligan y → hasRudeDemeanor y) :
  (∃ z, isHooligan z ∧ hasRudeDemeanor z ∧ hasBeatlesHaircut z) ∧
  ¬(∀ w, isHooligan w ∧ hasRudeDemeanor w → hasBeatlesHaircut w) :=
by sorry

end hooligan_theorem_l181_18193


namespace floor_expression_equals_eight_l181_18151

def n : ℕ := 2024

theorem floor_expression_equals_eight :
  ⌊(2025^3 : ℚ) / (2023 * 2024) - (2023^3 : ℚ) / (2024 * 2025)⌋ = 8 := by
  sorry

end floor_expression_equals_eight_l181_18151


namespace ruel_stamps_count_l181_18137

theorem ruel_stamps_count : ∀ (books_of_10 books_of_15 : ℕ),
  books_of_10 = 4 →
  books_of_15 = 6 →
  books_of_10 * 10 + books_of_15 * 15 = 130 :=
by
  sorry

end ruel_stamps_count_l181_18137


namespace tobys_friends_l181_18127

theorem tobys_friends (total : ℕ) (boys girls : ℕ) : 
  (boys : ℚ) / total = 55 / 100 →
  girls = 27 →
  total = boys + girls →
  boys = 33 := by
sorry

end tobys_friends_l181_18127


namespace intersection_A_B_l181_18171

def A : Set ℝ := {x | ∃ (α β : ℤ), α ≥ 0 ∧ β ≥ 0 ∧ x = 2^α * 3^β}
def B : Set ℝ := {x | 1 ≤ x ∧ x ≤ 5}

theorem intersection_A_B : A ∩ B = {1, 2, 3, 4} := by sorry

end intersection_A_B_l181_18171


namespace only_f1_is_even_l181_18108

-- Define the functions
def f1 (x : ℝ) : ℝ := x^2 - 3*abs x + 2
def f2 (x : ℝ) : ℝ := x^2
def f3 (x : ℝ) : ℝ := x^3
def f4 (x : ℝ) : ℝ := x - 1

-- Define the domain for f2
def f2_domain : Set ℝ := Set.Ioc (-2) 2

-- Define what it means for a function to be even
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

-- State the theorem
theorem only_f1_is_even :
  is_even f1 ∧ ¬(is_even f2) ∧ ¬(is_even f3) ∧ ¬(is_even f4) :=
sorry

end only_f1_is_even_l181_18108


namespace no_divisible_with_small_digit_sum_l181_18181

/-- Represents a number consisting of m ones -/
def ones (m : ℕ) : ℕ := 
  (10^m - 1) / 9

/-- Calculates the digit sum of a natural number -/
def digitSum (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + digitSum (n / 10)

/-- Theorem stating that no natural number divisible by ones(m) has a digit sum less than m -/
theorem no_divisible_with_small_digit_sum (m : ℕ) : 
  ¬ ∃ (n : ℕ), (n % ones m = 0) ∧ (digitSum n < m) := by
  sorry

end no_divisible_with_small_digit_sum_l181_18181


namespace pet_store_puppies_l181_18104

theorem pet_store_puppies (sold : ℕ) (puppies_per_cage : ℕ) (num_cages : ℕ) :
  sold = 3 ∧ puppies_per_cage = 5 ∧ num_cages = 3 →
  sold + num_cages * puppies_per_cage = 18 := by
  sorry

end pet_store_puppies_l181_18104


namespace pizza_segment_length_squared_l181_18152

theorem pizza_segment_length_squared (diameter : ℝ) (num_pieces : ℕ) (m : ℝ) : 
  diameter = 18 →
  num_pieces = 4 →
  m = 2 * (diameter / 2) * Real.sin (π / (2 * num_pieces)) →
  m^2 = 162 := by sorry

end pizza_segment_length_squared_l181_18152


namespace ellipse_a_plus_k_eq_eight_l181_18142

/-- An ellipse with given properties -/
structure Ellipse where
  foci1 : ℝ × ℝ
  foci2 : ℝ × ℝ
  point : ℝ × ℝ
  h : ℝ
  k : ℝ
  a : ℝ
  b : ℝ
  a_pos : a > 0
  b_pos : b > 0
  foci_x_eq : foci1.1 = foci2.1
  passes_through : (point.1 - h)^2 / a^2 + (point.2 - k)^2 / b^2 = 1

/-- Theorem stating that a + k = 8 for the given ellipse -/
theorem ellipse_a_plus_k_eq_eight (e : Ellipse) 
  (h_foci1 : e.foci1 = (-4, 1)) 
  (h_foci2 : e.foci2 = (-4, 5)) 
  (h_point : e.point = (1, 3)) : 
  e.a + e.k = 8 := by
  sorry

end ellipse_a_plus_k_eq_eight_l181_18142


namespace complex_modulus_problem_l181_18105

theorem complex_modulus_problem (z : ℂ) : 
  z = (1 + 2*I)^2 / (-I + 2) → Complex.abs z = Real.sqrt 5 := by
  sorry

end complex_modulus_problem_l181_18105


namespace five_digit_divisible_by_72_l181_18156

theorem five_digit_divisible_by_72 (a b : Nat) : 
  (a < 10 ∧ b < 10) →
  (a * 10000 + 6 * 1000 + 7 * 100 + 9 * 10 + b) % 72 = 0 ↔ 
  (a = 3 ∧ b = 2) := by
  sorry

end five_digit_divisible_by_72_l181_18156


namespace picture_distance_l181_18188

theorem picture_distance (wall_width picture_width : ℝ) 
  (hw : wall_width = 25)
  (hp : picture_width = 3) :
  (wall_width - picture_width) / 2 = 11 := by
sorry

end picture_distance_l181_18188


namespace computer_cost_l181_18166

theorem computer_cost (initial_amount printer_cost amount_left : ℕ) 
  (h1 : initial_amount = 450)
  (h2 : printer_cost = 40)
  (h3 : amount_left = 10) :
  initial_amount - printer_cost - amount_left = 400 :=
by sorry

end computer_cost_l181_18166


namespace rectangle_perimeter_l181_18117

/-- Given a square with perimeter 144 units divided into 4 congruent rectangles by vertical lines,
    the perimeter of one rectangle is 90 units. -/
theorem rectangle_perimeter (square_perimeter : ℝ) (num_rectangles : ℕ) : 
  square_perimeter = 144 → 
  num_rectangles = 4 → 
  ∃ (rectangle_perimeter : ℝ), rectangle_perimeter = 90 := by
  sorry

end rectangle_perimeter_l181_18117


namespace number_of_girls_in_class_correct_number_of_girls_l181_18147

theorem number_of_girls_in_class (num_boys : ℕ) (group_size : ℕ) (num_groups : ℕ) : ℕ :=
  let total_members := group_size * num_groups
  let num_girls := total_members - num_boys
  num_girls

theorem correct_number_of_girls :
  number_of_girls_in_class 9 3 7 = 12 := by
  sorry

end number_of_girls_in_class_correct_number_of_girls_l181_18147


namespace sandcastle_height_difference_l181_18186

theorem sandcastle_height_difference (miki_height sister_height : ℝ) 
  (h1 : miki_height = 0.83)
  (h2 : sister_height = 0.5) :
  miki_height - sister_height = 0.33 := by sorry

end sandcastle_height_difference_l181_18186


namespace division_equality_may_not_hold_l181_18184

theorem division_equality_may_not_hold (a b c : ℝ) : 
  a = b → ¬(∀ c, a / c = b / c) :=
by
  sorry

end division_equality_may_not_hold_l181_18184


namespace log_sum_simplification_l181_18167

theorem log_sum_simplification :
  1 / (Real.log 3 / Real.log 12 + 1) +
  1 / (Real.log 2 / Real.log 8 + 1) +
  1 / (Real.log 3 / Real.log 9 + 1) = 4 / 3 := by
  sorry

end log_sum_simplification_l181_18167


namespace trig_identity_l181_18130

theorem trig_identity : 
  (3 / (Real.sin (20 * π / 180))^2) - (1 / (Real.cos (20 * π / 180))^2) + 64 * (Real.sin (20 * π / 180))^2 = 32 := by
  sorry

end trig_identity_l181_18130


namespace deepak_age_l181_18119

/-- Given the ratio of Rahul's age to Deepak's age and Rahul's future age, 
    prove Deepak's present age -/
theorem deepak_age (rahul_age deepak_age : ℕ) : 
  (rahul_age : ℚ) / deepak_age = 4 / 3 →
  rahul_age + 22 = 26 →
  deepak_age = 3 := by
  sorry

end deepak_age_l181_18119


namespace nested_subtraction_simplification_l181_18190

theorem nested_subtraction_simplification (z : ℝ) :
  1 - (2 - (3 - (4 - (5 - z)))) = 3 - z := by sorry

end nested_subtraction_simplification_l181_18190


namespace cos_600_degrees_l181_18126

theorem cos_600_degrees : Real.cos (600 * π / 180) = -1/2 := by
  sorry

end cos_600_degrees_l181_18126


namespace same_color_probability_l181_18164

/-- The probability of drawing two balls of the same color from a bag containing black and white balls -/
theorem same_color_probability (total_balls : ℕ) (black_balls : ℕ) (white_balls : ℕ) 
  (h1 : total_balls = black_balls + white_balls)
  (h2 : black_balls = 7)
  (h3 : white_balls = 8) :
  (black_balls * (black_balls - 1) + white_balls * (white_balls - 1)) / (total_balls * (total_balls - 1)) = 7 / 15 := by
sorry

end same_color_probability_l181_18164


namespace multiply_72519_and_9999_l181_18192

theorem multiply_72519_and_9999 : 72519 * 9999 = 724817481 := by
  sorry

end multiply_72519_and_9999_l181_18192


namespace air_conditioning_price_calculation_air_conditioning_price_proof_l181_18195

/-- Calculates the final price of an air-conditioning unit after a discount and subsequent increase -/
theorem air_conditioning_price_calculation (initial_price : ℚ) 
  (discount_rate : ℚ) (increase_rate : ℚ) : ℚ :=
  let discounted_price := initial_price * (1 - discount_rate)
  let final_price := discounted_price * (1 + increase_rate)
  final_price

/-- Proves that the final price of the air-conditioning unit is approximately $442.18 -/
theorem air_conditioning_price_proof : 
  ∃ (ε : ℚ), ε > 0 ∧ ε < 0.005 ∧ 
  |air_conditioning_price_calculation 470 (16/100) (12/100) - 442.18| < ε :=
sorry

end air_conditioning_price_calculation_air_conditioning_price_proof_l181_18195


namespace statue_weight_calculation_l181_18199

/-- The weight of a statue after a series of cuts -/
def final_statue_weight (original_weight : ℝ) : ℝ :=
  let after_first_cut := original_weight * (1 - 0.3)
  let after_second_cut := after_first_cut * (1 - 0.2)
  let after_third_cut := after_second_cut * (1 - 0.25)
  after_third_cut

/-- Theorem stating the final weight of the statue -/
theorem statue_weight_calculation :
  final_statue_weight 250 = 105 := by
  sorry

end statue_weight_calculation_l181_18199


namespace jim_distance_driven_l181_18106

/-- The distance Jim has driven so far in his journey -/
def distance_driven (total_journey : ℕ) (remaining_distance : ℕ) : ℕ :=
  total_journey - remaining_distance

/-- Theorem stating that Jim has driven 215 miles -/
theorem jim_distance_driven :
  distance_driven 1200 985 = 215 := by
  sorry

end jim_distance_driven_l181_18106


namespace approximation_problem_l181_18175

def is_close (x y : ℝ) (ε : ℝ) : Prop := |x - y| ≤ ε

theorem approximation_problem :
  (∀ n : ℕ, 5 ≤ n ∧ n ≤ 9 → is_close (5 * n * 18) 1200 90) ∧
  (∀ m : ℕ, 0 ≤ m ∧ m ≤ 2 → is_close ((3 * 10 + m) * 9 / 5) 60 5) :=
sorry

end approximation_problem_l181_18175


namespace odd_m_triple_g_65_l181_18172

def g (n : ℤ) : ℤ :=
  if n % 2 = 1 then n + 5 else n / 2

theorem odd_m_triple_g_65 (m : ℤ) (h_odd : m % 2 = 1) :
  g (g (g m)) = 65 → m = 255 := by sorry

end odd_m_triple_g_65_l181_18172


namespace quadratic_root_difference_l181_18180

theorem quadratic_root_difference (p q : ℝ) (hp : p > 0) (hq : q > 0) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    x₁^2 + p*x₁ + q = 0 ∧ 
    x₂^2 + p*x₂ + q = 0 ∧ 
    |x₁ - x₂| = 2) →
  p = Real.sqrt (4*q + 4) :=
by sorry

end quadratic_root_difference_l181_18180


namespace gcd_2146_1813_l181_18194

theorem gcd_2146_1813 : Nat.gcd 2146 1813 = 37 := by sorry

end gcd_2146_1813_l181_18194


namespace yans_distance_ratio_l181_18161

theorem yans_distance_ratio (w : ℝ) (x y : ℝ) (hw : w > 0) (hx : x > 0) (hy : y > 0) :
  (y / w = x / w + (x + y) / (6 * w)) → (x / y = 5 / 7) :=
by sorry

end yans_distance_ratio_l181_18161


namespace min_luxury_owners_l181_18109

structure Village where
  population : ℕ
  refrigerator_owners : Finset Nat
  television_owners : Finset Nat
  computer_owners : Finset Nat
  air_conditioner_owners : Finset Nat
  washing_machine_owners : Finset Nat
  microwave_owners : Finset Nat
  internet_owners : Finset Nat
  top_earners : Finset Nat

def Owlna (v : Village) : Prop :=
  v.refrigerator_owners.card = (67 * v.population) / 100 ∧
  v.television_owners.card = (74 * v.population) / 100 ∧
  v.computer_owners.card = (77 * v.population) / 100 ∧
  v.air_conditioner_owners.card = (83 * v.population) / 100 ∧
  v.washing_machine_owners.card = (55 * v.population) / 100 ∧
  v.microwave_owners.card = (48 * v.population) / 100 ∧
  v.internet_owners.card = (42 * v.population) / 100 ∧
  (v.television_owners ∩ v.computer_owners).card = (35 * v.population) / 100 ∧
  (v.washing_machine_owners ∩ v.microwave_owners).card = (30 * v.population) / 100 ∧
  (v.air_conditioner_owners ∩ v.refrigerator_owners).card = (27 * v.population) / 100 ∧
  v.top_earners.card = (10 * v.population) / 100 ∧
  (v.refrigerator_owners ∩ v.television_owners ∩ v.computer_owners ∩
   v.air_conditioner_owners ∩ v.washing_machine_owners ∩ v.microwave_owners ∩
   v.internet_owners) ⊆ v.top_earners

theorem min_luxury_owners (v : Village) (h : Owlna v) :
  (v.refrigerator_owners ∩ v.television_owners ∩ v.computer_owners ∩
   v.air_conditioner_owners ∩ v.washing_machine_owners ∩ v.microwave_owners ∩
   v.internet_owners ∩ v.top_earners).card = (10 * v.population) / 100 :=
by sorry

end min_luxury_owners_l181_18109


namespace bacteria_growth_l181_18182

theorem bacteria_growth (n : ℕ) : n = 4 ↔ (n > 0 ∧ 5 * 3^n > 200 ∧ ∀ m : ℕ, m > 0 → m < n → 5 * 3^m ≤ 200) :=
by sorry

end bacteria_growth_l181_18182


namespace day_crew_load_fraction_l181_18115

theorem day_crew_load_fraction (D : ℝ) (W_d : ℝ) (W_d_pos : W_d > 0) : 
  let night_boxes_per_worker := (1 / 2) * D
  let night_workers := (4 / 5) * W_d
  let day_total := D * W_d
  let night_total := night_boxes_per_worker * night_workers
  (day_total) / (day_total + night_total) = 5 / 7 := by
sorry

end day_crew_load_fraction_l181_18115


namespace turquoise_more_blue_count_l181_18123

/-- Represents the results of a survey about the color turquoise -/
structure TurquoiseSurvey where
  total : ℕ
  more_green : ℕ
  both : ℕ
  neither : ℕ

/-- Calculates the number of people who believe turquoise is "more blue" -/
def more_blue (survey : TurquoiseSurvey) : ℕ :=
  survey.total - (survey.more_green - survey.both) - survey.neither - survey.both

/-- Theorem stating that in the given survey, 65 people believe turquoise is "more blue" -/
theorem turquoise_more_blue_count :
  ∃ (survey : TurquoiseSurvey),
    survey.total = 150 ∧
    survey.more_green = 95 ∧
    survey.both = 35 ∧
    survey.neither = 25 ∧
    more_blue survey = 65 := by
  sorry

#eval more_blue ⟨150, 95, 35, 25⟩

end turquoise_more_blue_count_l181_18123


namespace carly_running_ratio_l181_18112

/-- Carly's running schedule over four weeks -/
def running_schedule (r : ℚ) : Fin 4 → ℚ
  | 0 => 2                    -- First week
  | 1 => 2 * r + 3            -- Second week
  | 2 => 9/7 * (2 * r + 3)    -- Third week
  | 3 => 4                    -- Fourth week

theorem carly_running_ratio :
  ∃ r : ℚ,
    running_schedule r 2 = 9 ∧
    running_schedule r 3 = running_schedule r 2 - 5 ∧
    running_schedule r 1 / running_schedule r 0 = 7/2 := by
  sorry

end carly_running_ratio_l181_18112


namespace algebraic_expression_equality_l181_18158

theorem algebraic_expression_equality (x : ℝ) (h : x = Real.sqrt 2 + 1) :
  (x + 1) / (x - 1) = Real.sqrt 2 + 1 := by
  sorry

end algebraic_expression_equality_l181_18158


namespace ellipse_condition_necessary_not_sufficient_l181_18132

/-- The equation of a potential ellipse with parameter k -/
def ellipse_equation (x y k : ℝ) : Prop :=
  x^2 / (k - 1) + y^2 / (5 - k) = 1

/-- Condition for k -/
def k_condition (k : ℝ) : Prop :=
  1 < k ∧ k < 5

/-- Definition of an ellipse (simplified for this problem) -/
def is_ellipse (k : ℝ) : Prop :=
  k_condition k ∧ k ≠ 3

theorem ellipse_condition_necessary_not_sufficient :
  (∀ k, is_ellipse k → k_condition k) ∧
  ¬(∀ k, k_condition k → is_ellipse k) :=
by sorry

end ellipse_condition_necessary_not_sufficient_l181_18132
