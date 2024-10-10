import Mathlib

namespace expand_binomials_l165_16518

theorem expand_binomials (x : ℝ) : (x + 3) * (4 * x - 8) = 4 * x^2 + 4 * x - 24 := by
  sorry

end expand_binomials_l165_16518


namespace periodic_even_function_extension_l165_16505

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

def is_even (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

theorem periodic_even_function_extension
  (f : ℝ → ℝ)
  (h_periodic : is_periodic f 2)
  (h_even : is_even f)
  (h_def : ∀ x ∈ Set.Ioo 0 1, f x = Real.log (1 - x) / Real.log (1/2)) :
  ∀ x ∈ Set.Ioo 1 2, f x = Real.log (x - 1) / Real.log (1/2) := by
sorry

end periodic_even_function_extension_l165_16505


namespace factor_calculation_l165_16580

theorem factor_calculation : 
  let initial_number : ℕ := 15
  let resultant : ℕ := 2 * initial_number + 5
  let final_result : ℕ := 105
  ∃ factor : ℚ, factor * resultant = final_result ∧ factor = 3 :=
by sorry

end factor_calculation_l165_16580


namespace weight_equivalence_l165_16502

-- Define the weights of shapes as real numbers
variable (triangle circle square : ℝ)

-- Define the conditions from the problem
axiom weight_relation1 : 5 * triangle = 3 * circle
axiom weight_relation2 : circle = triangle + 2 * square

-- Theorem to prove
theorem weight_equivalence : triangle + circle = 3 * square := by
  sorry

end weight_equivalence_l165_16502


namespace no_solution_iff_n_eq_neg_one_l165_16556

theorem no_solution_iff_n_eq_neg_one (n : ℝ) : 
  (∀ x y z : ℝ, nx + y = 1 ∧ ny + z = 1 ∧ x + nz = 1 → False) ↔ n = -1 := by
  sorry

end no_solution_iff_n_eq_neg_one_l165_16556


namespace log_50_bounds_sum_l165_16524

theorem log_50_bounds_sum : ∃ c d : ℤ, (1 : ℝ) ≤ c ∧ (c : ℝ) < Real.log 50 / Real.log 10 ∧ Real.log 50 / Real.log 10 < d ∧ (d : ℝ) ≤ 2 ∧ c + d = 3 := by
  sorry

end log_50_bounds_sum_l165_16524


namespace lauras_workout_speed_l165_16554

theorem lauras_workout_speed :
  ∃! x : ℝ, x > 0 ∧ (25 / (3 * x + 2)) + (8 / x) = 2.25 := by
  sorry

end lauras_workout_speed_l165_16554


namespace pond_draining_time_l165_16537

theorem pond_draining_time 
  (pump1_half_time : ℝ) 
  (pump2_full_time : ℝ) 
  (combined_half_time : ℝ) 
  (h1 : pump2_full_time = 1.25) 
  (h2 : combined_half_time = 0.5) :
  pump1_half_time = 5/12 := by
sorry

end pond_draining_time_l165_16537


namespace trapezoid_area_l165_16512

/-- The area of a trapezoid with sum of bases 36 cm and height 15 cm is 270 square centimeters. -/
theorem trapezoid_area (base_sum : ℝ) (height : ℝ) (h1 : base_sum = 36) (h2 : height = 15) :
  (base_sum * height) / 2 = 270 := by
  sorry

end trapezoid_area_l165_16512


namespace quadratic_equation_roots_and_c_l165_16579

/-- Given a quadratic equation x^2 - 6x + c = 0 with one root being 2,
    prove that the other root is 4 and the value of c is 8. -/
theorem quadratic_equation_roots_and_c (c : ℝ) :
  (∃ x : ℝ, x^2 - 6*x + c = 0 ∧ x = 2) →
  (∃ y : ℝ, y^2 - 6*y + c = 0 ∧ y = 4 ∧ c = 8) :=
by sorry

end quadratic_equation_roots_and_c_l165_16579


namespace profit_range_l165_16533

/-- Price function for books -/
def C (n : ℕ) : ℕ :=
  if n ≤ 24 then 12 * n
  else if n ≤ 48 then 11 * n
  else 10 * n

/-- Total number of books -/
def total_books : ℕ := 60

/-- Cost per book to the company -/
def cost_per_book : ℕ := 5

/-- Profit function given two people buying books -/
def profit (a b : ℕ) : ℤ :=
  (C a + C b) - (cost_per_book * total_books)

/-- Theorem stating the range of profit -/
theorem profit_range :
  ∀ a b : ℕ,
  a + b = total_books →
  a ≥ 1 →
  b ≥ 1 →
  302 ≤ profit a b ∧ profit a b ≤ 384 :=
sorry

end profit_range_l165_16533


namespace remaining_fence_is_48_feet_l165_16501

/-- The length of fence remaining to be whitewashed after three people have worked on it. -/
def remaining_fence (total_length : ℝ) (first_length : ℝ) (second_fraction : ℝ) (third_fraction : ℝ) : ℝ :=
  let remaining_after_first := total_length - first_length
  let remaining_after_second := remaining_after_first * (1 - second_fraction)
  remaining_after_second * (1 - third_fraction)

/-- Theorem stating that the remaining fence to be whitewashed is 48 feet. -/
theorem remaining_fence_is_48_feet :
  remaining_fence 100 10 (1/5) (1/3) = 48 := by
  sorry

end remaining_fence_is_48_feet_l165_16501


namespace function_is_linear_l165_16531

-- Define the function f and constants a and b
variable (f : ℝ → ℝ) (a b : ℝ)

-- State the theorem
theorem function_is_linear
  (h_continuous : Continuous f)
  (h_a : 0 < a ∧ a < 1/2)
  (h_b : 0 < b ∧ b < 1/2)
  (h_functional : ∀ x, f (f x) = a * f x + b * x) :
  ∃ k : ℝ, ∀ x, f x = k * x :=
by sorry

end function_is_linear_l165_16531


namespace initial_orchids_l165_16597

theorem initial_orchids (initial_roses : ℕ) (final_roses : ℕ) (final_orchids : ℕ) 
  (orchid_rose_difference : ℕ) : 
  initial_roses = 7 → 
  final_roses = 11 → 
  final_orchids = 20 → 
  orchid_rose_difference = 9 → 
  final_orchids = final_roses + orchid_rose_difference →
  initial_roses + orchid_rose_difference = 16 := by
  sorry

end initial_orchids_l165_16597


namespace johns_reading_rate_l165_16590

/-- The number of books John read in 6 weeks -/
def total_books : ℕ := 48

/-- The number of weeks John read -/
def weeks : ℕ := 6

/-- The number of days John reads per week -/
def reading_days_per_week : ℕ := 2

/-- The number of books John can read in a day -/
def books_per_day : ℕ := total_books / (weeks * reading_days_per_week)

theorem johns_reading_rate : books_per_day = 4 := by
  sorry

end johns_reading_rate_l165_16590


namespace subsets_containing_five_and_seven_l165_16561

def S : Finset Nat := {1, 2, 3, 4, 5, 6, 7}

theorem subsets_containing_five_and_seven :
  (Finset.filter (fun s => 5 ∈ s ∧ 7 ∈ s) (Finset.powerset S)).card = 32 := by
  sorry

end subsets_containing_five_and_seven_l165_16561


namespace julia_watch_collection_l165_16584

theorem julia_watch_collection (silver_watches : ℕ) (bronze_watches : ℕ) (gold_watches : ℕ) : 
  silver_watches = 20 →
  bronze_watches = 3 * silver_watches →
  gold_watches = (silver_watches + bronze_watches) / 10 →
  silver_watches + bronze_watches + gold_watches = 88 := by
  sorry

end julia_watch_collection_l165_16584


namespace larger_number_problem_l165_16539

theorem larger_number_problem (L S : ℕ) (h1 : L > S) (h2 : L - S = 1000) (h3 : L = 10 * S + 10) : L = 1110 := by
  sorry

end larger_number_problem_l165_16539


namespace base_4_divisibility_l165_16522

def base_4_to_decimal (a b c d : ℕ) : ℕ :=
  a * 4^3 + b * 4^2 + c * 4 + d

def is_divisible_by_13 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 13 * k

theorem base_4_divisibility :
  ∀ x : ℕ, x < 4 →
    is_divisible_by_13 (base_4_to_decimal 2 3 1 x) ↔ x = 1 := by
  sorry

end base_4_divisibility_l165_16522


namespace mango_purchase_l165_16525

/-- The price of grapes per kg -/
def grape_price : ℕ := 70

/-- The price of mangoes per kg -/
def mango_price : ℕ := 55

/-- The amount of grapes purchased in kg -/
def grape_amount : ℕ := 8

/-- The total amount paid to the shopkeeper -/
def total_paid : ℕ := 1055

/-- The amount of mangoes purchased in kg -/
def mango_amount : ℕ := (total_paid - grape_amount * grape_price) / mango_price

theorem mango_purchase : mango_amount = 9 := by
  sorry

end mango_purchase_l165_16525


namespace circle_revolutions_l165_16545

/-- Represents a circle with a given radius -/
structure Circle where
  radius : ℝ

/-- Represents the motion of circles C₁, C₂, and C₃ -/
structure CircleMotion where
  C₁ : Circle
  C₂ : Circle
  C₃ : Circle
  n : ℕ

/-- The number of revolutions C₃ makes relative to the ground -/
def revolutions (motion : CircleMotion) : ℝ := motion.n - 1

/-- The theorem stating the number of revolutions C₃ makes -/
theorem circle_revolutions (motion : CircleMotion) 
  (h₁ : motion.n > 2)
  (h₂ : motion.C₁.radius = motion.n * motion.C₃.radius)
  (h₃ : motion.C₂.radius = 2 * motion.C₃.radius)
  (h₄ : motion.C₃.radius > 0) :
  revolutions motion = motion.n - 1 := by sorry

end circle_revolutions_l165_16545


namespace lemonade_water_quarts_l165_16553

/-- Proves the number of quarts of water needed for a special lemonade recipe -/
theorem lemonade_water_quarts : 
  let total_parts : ℚ := 5 + 3
  let water_parts : ℚ := 5
  let total_gallons : ℚ := 5
  let quarts_per_gallon : ℚ := 4
  (water_parts / total_parts) * total_gallons * quarts_per_gallon = 25 / 2 := by
  sorry

end lemonade_water_quarts_l165_16553


namespace halfway_between_one_eighth_and_one_third_l165_16599

theorem halfway_between_one_eighth_and_one_third :
  (1 / 8 : ℚ) / 2 + (1 / 3 : ℚ) / 2 = 11 / 48 := by
  sorry

end halfway_between_one_eighth_and_one_third_l165_16599


namespace range_of_x_l165_16564

theorem range_of_x (x : ℝ) : 
  (x^2 - 2*x - 3 ≤ 0) → (1/(x-2) ≤ 0) → (-1 ≤ x ∧ x < 2) := by
  sorry

end range_of_x_l165_16564


namespace sum_of_special_primes_is_prime_l165_16573

theorem sum_of_special_primes_is_prime (A B : ℕ+) : 
  Nat.Prime A ∧ 
  Nat.Prime B ∧ 
  Nat.Prime (A - B) ∧ 
  Nat.Prime (A + B) → 
  Nat.Prime (A + B + (A - B) + A + B) :=
sorry

end sum_of_special_primes_is_prime_l165_16573


namespace trigonometric_expression_equals_half_l165_16504

open Real

theorem trigonometric_expression_equals_half : 
  (cos (85 * π / 180) + sin (25 * π / 180) * cos (30 * π / 180)) / cos (25 * π / 180) = 1/2 := by
  sorry

end trigonometric_expression_equals_half_l165_16504


namespace equation_solutions_l165_16527

theorem equation_solutions : 
  {x : ℝ | Real.sqrt ((3 + 2 * Real.sqrt 2) ^ x) + Real.sqrt ((3 - 2 * Real.sqrt 2) ^ x) = 6} = {2, -2} := by
  sorry

end equation_solutions_l165_16527


namespace triangle_problem_l165_16560

theorem triangle_problem (a b c A B C : ℝ) (h1 : 0 < A) (h2 : A < π) : 
  c = a * Real.sin C - c * Real.cos A →
  (A = π / 2) ∧ 
  (a = 2 → 1/2 * b * c * Real.sin A = 2 → b = 2 ∧ c = 2) :=
by sorry

end triangle_problem_l165_16560


namespace average_reduction_percentage_option1_more_favorable_l165_16551

-- Define the original and final prices
def original_price : ℝ := 5
def final_price : ℝ := 3.2

-- Define the quantity to purchase in kilograms
def quantity : ℝ := 5000

-- Define the discount percentage and cash discount
def discount_percentage : ℝ := 0.1
def cash_discount_per_ton : ℝ := 200

-- Theorem for the average percentage reduction
theorem average_reduction_percentage :
  ∃ x : ℝ, x > 0 ∧ x < 1 ∧ original_price * (1 - x)^2 = final_price ∧ x = 0.2 :=
sorry

-- Theorem for the more favorable option
theorem option1_more_favorable :
  final_price * (1 - discount_percentage) * quantity <
  final_price * quantity - (cash_discount_per_ton * (quantity / 1000)) :=
sorry

end average_reduction_percentage_option1_more_favorable_l165_16551


namespace box_height_proof_l165_16513

theorem box_height_proof (length width cube_volume num_cubes : ℝ) 
  (h1 : length = 12)
  (h2 : width = 16)
  (h3 : cube_volume = 3)
  (h4 : num_cubes = 384) :
  (num_cubes * cube_volume) / (length * width) = 6 := by
  sorry

end box_height_proof_l165_16513


namespace sqrt_equation_implies_difference_l165_16582

theorem sqrt_equation_implies_difference (m n : ℕ) : 
  (Real.sqrt (9 - m / n) = 9 * Real.sqrt (m / n)) → (n - m = 73) := by
  sorry

end sqrt_equation_implies_difference_l165_16582


namespace initial_storks_count_l165_16510

theorem initial_storks_count (initial_birds : ℕ) (additional_birds : ℕ) (final_difference : ℕ) :
  initial_birds = 2 →
  additional_birds = 3 →
  final_difference = 1 →
  initial_birds + additional_birds + final_difference = 6 :=
by sorry

end initial_storks_count_l165_16510


namespace two_middle_zeros_in_quotient_l165_16562

/-- Count the number of zeros in the middle of a positive integer -/
def count_middle_zeros (n : ℕ) : ℕ :=
  sorry

/-- The quotient when 2010 is divided by 2 -/
def quotient : ℕ := 2010 / 2

theorem two_middle_zeros_in_quotient : count_middle_zeros quotient = 2 := by
  sorry

end two_middle_zeros_in_quotient_l165_16562


namespace inequality_proof_l165_16563

theorem inequality_proof (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (habc : a * b * c = 1) : 
  (a - 1) / c + (c - 1) / b + (b - 1) / a ≥ 0 := by
  sorry

end inequality_proof_l165_16563


namespace least_three_digit_with_product_12_l165_16509

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def digit_product (n : ℕ) : ℕ :=
  (n / 100) * ((n / 10) % 10) * (n % 10)

theorem least_three_digit_with_product_12 :
  ∀ n : ℕ, is_three_digit n → digit_product n = 12 → 126 ≤ n :=
by sorry

end least_three_digit_with_product_12_l165_16509


namespace solution_added_mass_l165_16536

/-- Represents the composition and manipulation of a solution --/
structure Solution :=
  (total_mass : ℝ)
  (liquid_x_percentage : ℝ)

/-- Calculates the mass of liquid x in a solution --/
def liquid_x_mass (s : Solution) : ℝ :=
  s.total_mass * s.liquid_x_percentage

/-- Represents the problem scenario --/
def solution_problem (initial_solution : Solution) 
  (evaporated_water : ℝ) (added_solution : Solution) : Prop :=
  let remaining_solution : Solution := {
    total_mass := initial_solution.total_mass - evaporated_water,
    liquid_x_percentage := 
      liquid_x_mass initial_solution / (initial_solution.total_mass - evaporated_water)
  }
  let final_solution : Solution := {
    total_mass := remaining_solution.total_mass + added_solution.total_mass,
    liquid_x_percentage := 0.4
  }
  liquid_x_mass remaining_solution + liquid_x_mass added_solution = 
    liquid_x_mass final_solution

/-- The theorem to be proved --/
theorem solution_added_mass : 
  let initial_solution : Solution := { total_mass := 6, liquid_x_percentage := 0.3 }
  let evaporated_water : ℝ := 2
  let added_solution : Solution := { total_mass := 2, liquid_x_percentage := 0.3 }
  solution_problem initial_solution evaporated_water added_solution := by
  sorry

end solution_added_mass_l165_16536


namespace reflection_of_circle_center_l165_16592

def reflect_over_y_eq_x (p : ℝ × ℝ) : ℝ × ℝ := (p.2, p.1)

theorem reflection_of_circle_center :
  let original_center : ℝ × ℝ := (6, -5)
  let reflected_center : ℝ × ℝ := reflect_over_y_eq_x original_center
  reflected_center = (-5, 6) := by sorry

end reflection_of_circle_center_l165_16592


namespace tom_payment_l165_16532

/-- The total amount Tom paid to the shopkeeper for apples and mangoes -/
def total_amount (apple_quantity : ℕ) (apple_rate : ℕ) (mango_quantity : ℕ) (mango_rate : ℕ) : ℕ :=
  apple_quantity * apple_rate + mango_quantity * mango_rate

/-- Proof that Tom paid 1190 to the shopkeeper -/
theorem tom_payment : total_amount 8 70 9 70 = 1190 := by
  sorry

end tom_payment_l165_16532


namespace square_plus_inverse_square_l165_16515

theorem square_plus_inverse_square (x : ℝ) (h : x - 3/x = 2) : x^2 + 9/x^2 = 10 := by
  sorry

end square_plus_inverse_square_l165_16515


namespace maximum_marks_calculation_l165_16544

/-- 
Given:
- The passing mark is 35% of the maximum marks
- A student got 150 marks and failed by 25 marks
Prove that the maximum marks is 500
-/
theorem maximum_marks_calculation (M : ℝ) : 
  (0.35 * M = 150 + 25) → M = 500 := by
  sorry

end maximum_marks_calculation_l165_16544


namespace non_integer_a_implies_b_is_one_l165_16541

theorem non_integer_a_implies_b_is_one (a b : ℝ) : 
  a + b - a * b = 1 → ¬(∃ n : ℤ, a = n) → b = 1 := by
  sorry

end non_integer_a_implies_b_is_one_l165_16541


namespace sin_shift_equivalence_l165_16548

/-- Proves that shifting sin(2x + π/3) right by π/4 results in sin(2x - π/6) -/
theorem sin_shift_equivalence (x : ℝ) :
  Real.sin (2 * (x - π/4) + π/3) = Real.sin (2*x - π/6) := by
  sorry

end sin_shift_equivalence_l165_16548


namespace max_value_theorem_l165_16520

theorem max_value_theorem (x y : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) 
  (h_eq : x^2 - 3*x*y + 4*y^2 = 9) :
  x^2 + 3*x*y + 4*y^2 ≤ 63 ∧ ∃ (x₀ y₀ : ℝ), x₀^2 - 3*x₀*y₀ + 4*y₀^2 = 9 ∧ x₀^2 + 3*x₀*y₀ + 4*y₀^2 = 63 := by
  sorry

end max_value_theorem_l165_16520


namespace paper_products_distribution_l165_16587

theorem paper_products_distribution (total : ℕ) 
  (h1 : total = 20)
  (h2 : total / 2 + total / 4 + total / 5 + paper_cups = total) : 
  paper_cups = 1 := by
  sorry

end paper_products_distribution_l165_16587


namespace airplane_seats_l165_16585

/-- Calculates the total number of seats on an airplane given the number of coach class seats
    and the relationship between coach and first-class seats. -/
theorem airplane_seats (coach_seats : ℕ) (h1 : coach_seats = 310) 
    (h2 : ∃ first_class : ℕ, coach_seats = 4 * first_class + 2) : 
  coach_seats + (coach_seats - 2) / 4 = 387 := by
  sorry

#check airplane_seats

end airplane_seats_l165_16585


namespace sum_of_first_and_third_l165_16508

theorem sum_of_first_and_third (A B C : ℝ) : 
  A + B + C = 330 → 
  A = 2 * B → 
  C = A / 3 → 
  B = 90 → 
  A + C = 240 := by
sorry

end sum_of_first_and_third_l165_16508


namespace sum_of_specific_primes_l165_16550

theorem sum_of_specific_primes : ∃ (S : Finset Nat),
  (∀ p ∈ S, p.Prime ∧ 1 < p ∧ p ≤ 100 ∧ p % 6 = 1 ∧ p % 7 = 6) ∧
  (∀ p, p.Prime → 1 < p → p ≤ 100 → p % 6 = 1 → p % 7 = 6 → p ∈ S) ∧
  S.sum id = 104 := by
sorry

end sum_of_specific_primes_l165_16550


namespace train_crossing_time_l165_16542

/-- Proves that a train 100 meters long, traveling at 144 km/hr, will take 2.5 seconds to cross an electric pole. -/
theorem train_crossing_time (train_length : Real) (train_speed_kmh : Real) :
  train_length = 100 ∧ train_speed_kmh = 144 →
  (train_length / (train_speed_kmh * 1000 / 3600)) = 2.5 := by
  sorry

end train_crossing_time_l165_16542


namespace unique_three_digit_divisible_by_11_l165_16523

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def units_digit (n : ℕ) : ℕ := n % 10

def hundreds_digit (n : ℕ) : ℕ := (n / 100) % 10

theorem unique_three_digit_divisible_by_11 :
  ∃! n : ℕ, is_three_digit n ∧ 
             units_digit n = 5 ∧ 
             hundreds_digit n = 6 ∧ 
             n % 11 = 0 :=
by sorry

end unique_three_digit_divisible_by_11_l165_16523


namespace james_sales_percentage_l165_16589

/-- Represents the number of houses visited on the first day -/
def houses_day1 : ℕ := 20

/-- Represents the number of houses visited on the second day -/
def houses_day2 : ℕ := 2 * houses_day1

/-- Represents the number of items sold per house each day -/
def items_per_house : ℕ := 2

/-- Represents the total number of items sold over both days -/
def total_items_sold : ℕ := 104

/-- Calculates the percentage of houses sold to on the second day -/
def percentage_sold_day2 : ℚ :=
  (total_items_sold - houses_day1 * items_per_house) / (2 * houses_day2)

theorem james_sales_percentage :
  percentage_sold_day2 = 4/5 := by sorry

end james_sales_percentage_l165_16589


namespace storage_tubs_cost_l165_16511

/-- The total cost of storage tubs -/
def total_cost (large_count : ℕ) (small_count : ℕ) (large_price : ℕ) (small_price : ℕ) : ℕ :=
  large_count * large_price + small_count * small_price

/-- Theorem: The total cost of 3 large tubs at $6 each and 6 small tubs at $5 each is $48 -/
theorem storage_tubs_cost :
  total_cost 3 6 6 5 = 48 := by
  sorry

end storage_tubs_cost_l165_16511


namespace second_quadrant_necessary_not_sufficient_for_obtuse_l165_16598

-- Define the properties
def is_in_second_quadrant (α : Real) : Prop := 90 < α ∧ α ≤ 180
def is_obtuse_angle (α : Real) : Prop := 90 < α ∧ α < 180

-- Theorem statement
theorem second_quadrant_necessary_not_sufficient_for_obtuse :
  (∀ α, is_obtuse_angle α → is_in_second_quadrant α) ∧
  (∃ α, is_in_second_quadrant α ∧ ¬is_obtuse_angle α) :=
sorry

end second_quadrant_necessary_not_sufficient_for_obtuse_l165_16598


namespace perpendicular_vectors_condition_l165_16500

/-- Given two vectors m and n in ℝ², if m is perpendicular to n,
    then the second component of n is -2 times the first component of m. -/
theorem perpendicular_vectors_condition (m n : ℝ × ℝ) :
  m = (1, 2) →
  n.1 = a →
  n.2 = -1 →
  m.1 * n.1 + m.2 * n.2 = 0 →
  a = 2 := by
  sorry

end perpendicular_vectors_condition_l165_16500


namespace parallel_transitive_l165_16557

-- Define the type for lines
def Line : Type := ℝ → ℝ → ℝ → Prop

-- Define the parallel relation between lines
def parallel (a b : Line) : Prop := sorry

-- Theorem statement
theorem parallel_transitive (a b c : Line) :
  parallel a b → parallel b c → parallel a c := by
  sorry

end parallel_transitive_l165_16557


namespace minimum_value_of_expression_l165_16558

theorem minimum_value_of_expression (x : ℝ) (h : x > 0) :
  9 * x + 1 / x^6 ≥ 10 ∧ ∃ y > 0, 9 * y + 1 / y^6 = 10 := by
  sorry

end minimum_value_of_expression_l165_16558


namespace pond_side_length_l165_16517

/-- Represents the dimensions and pond of a rectangular garden -/
structure Garden where
  length : ℝ
  width : ℝ
  pond_side : ℝ

/-- Calculates the area of a rectangular garden -/
def garden_area (g : Garden) : ℝ := g.length * g.width

/-- Calculates the remaining area after building a square pond -/
def remaining_area (g : Garden) : ℝ := garden_area g - g.pond_side ^ 2

/-- Theorem stating the side length of the pond given the conditions -/
theorem pond_side_length (g : Garden) 
  (h1 : g.length = 15)
  (h2 : g.width = 10)
  (h3 : remaining_area g = (garden_area g) / 2) :
  g.pond_side = 5 * Real.sqrt 3 := by
  sorry

end pond_side_length_l165_16517


namespace collinear_vectors_solution_l165_16538

/-- Two vectors in R² -/
def m (x : ℝ) : ℝ × ℝ := (x, x + 2)
def n (x : ℝ) : ℝ × ℝ := (1, 3*x)

/-- Collinearity condition for two vectors -/
def collinear (v w : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), v.1 * w.2 = k * v.2 * w.1

theorem collinear_vectors_solution :
  ∀ x : ℝ, collinear (m x) (n x) → x = -2/3 ∨ x = 1 := by sorry

end collinear_vectors_solution_l165_16538


namespace range_of_a_l165_16528

-- Define the propositions p and q
def p (x : ℝ) : Prop := x > 1 ∨ x < -3
def q (x a : ℝ) : Prop := x > a

-- Define the set of x that satisfies p
def P : Set ℝ := {x | p x}

-- Define the set of x that satisfies q for a given a
def Q (a : ℝ) : Set ℝ := {x | q x a}

-- State the theorem
theorem range_of_a :
  (∀ a : ℝ, (∀ x : ℝ, q x a → p x) ∧ (∃ x : ℝ, p x ∧ ¬q x a)) ↔
  (∀ a : ℝ, a ∈ Set.Ici 1) :=
sorry

end range_of_a_l165_16528


namespace stamp_collection_problem_l165_16503

theorem stamp_collection_problem (C K A : ℕ) : 
  C > 2 * K ∧ 
  K = A / 2 ∧ 
  C + K + A = 930 ∧ 
  A = 370 → 
  C - 2 * K = 5 := by
sorry

end stamp_collection_problem_l165_16503


namespace min_max_abs_quadratic_cosine_l165_16514

open Real

theorem min_max_abs_quadratic_cosine :
  (∃ y₀ : ℝ, (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → |x^2 + x*y₀ + cos y₀| ≤ 2)) ∧
  (∀ y : ℝ, ∃ x : ℝ, 0 ≤ x ∧ x ≤ 1 ∧ |x^2 + x*y + cos y| ≥ 2) := by
  sorry

end min_max_abs_quadratic_cosine_l165_16514


namespace f_derivative_at_one_l165_16569

noncomputable def f (x : ℝ) : ℝ := x * Real.exp x

theorem f_derivative_at_one : 
  (deriv f) 1 = 2 * Real.exp 1 := by sorry

end f_derivative_at_one_l165_16569


namespace factorization_sum_l165_16540

theorem factorization_sum (d e f : ℤ) :
  (∀ x : ℝ, x^2 + 19*x + 88 = (x + d)*(x + e)) →
  (∀ x : ℝ, x^2 - 23*x + 120 = (x - e)*(x - f)) →
  d + e + f = 31 := by
sorry

end factorization_sum_l165_16540


namespace complex_24th_power_of_cube_root_of_unity_l165_16546

theorem complex_24th_power_of_cube_root_of_unity (z : ℂ) : z = (1 + Complex.I * Real.sqrt 3) / 2 → z^24 = 1 := by
  sorry

end complex_24th_power_of_cube_root_of_unity_l165_16546


namespace smartphone_demand_l165_16516

theorem smartphone_demand (d p : ℝ) (k : ℝ) :
  (d * p = k) →  -- Demand is inversely proportional to price
  (30 * 600 = k) →  -- 30 customers purchase at $600
  (20 * 900 = k) →  -- 20 customers purchase at $900
  True :=
by
  sorry

end smartphone_demand_l165_16516


namespace sqrt_eight_and_nine_sixteenths_l165_16577

theorem sqrt_eight_and_nine_sixteenths (x : ℝ) : 
  x = Real.sqrt (8 + 9 / 16) → x = Real.sqrt 137 / 4 := by sorry

end sqrt_eight_and_nine_sixteenths_l165_16577


namespace geometric_sequence_properties_l165_16543

/-- A geometric sequence with specific conditions -/
structure GeometricSequence where
  a : ℕ → ℝ
  first_condition : a 1 + a 3 = 10
  second_condition : a 4 + a 6 = 5/4

/-- The general term of the geometric sequence -/
def generalTerm (seq : GeometricSequence) (n : ℕ) : ℝ :=
  2^(4-n)

/-- The sum of the first four terms of the geometric sequence -/
def sumFirstFour (seq : GeometricSequence) : ℝ :=
  15

/-- Theorem stating the correctness of the general term and sum -/
theorem geometric_sequence_properties (seq : GeometricSequence) :
  (∀ n, seq.a n = generalTerm seq n) ∧ sumFirstFour seq = 15 := by
  sorry


end geometric_sequence_properties_l165_16543


namespace log_calculation_l165_16566

-- Define the common logarithm
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Theorem statement
theorem log_calculation :
  (lg 2)^2 + lg 2 * lg 50 + lg 25 = 2 :=
by
  -- Properties of logarithms
  have h1 : lg 50 = lg 2 + lg 25 := by sorry
  have h2 : lg 25 = 2 * lg 5 := by sorry
  have h3 : lg 10 = 1 := by sorry
  
  -- Proof steps would go here
  sorry

end log_calculation_l165_16566


namespace log_8_problem_l165_16588

theorem log_8_problem (x : ℝ) :
  Real.log x / Real.log 8 = 3.5 → x = 1024 * Real.sqrt 2 := by
  sorry

end log_8_problem_l165_16588


namespace regular_hexagon_side_length_l165_16549

/-- The length of a side in a regular hexagon, given the distance between opposite sides -/
theorem regular_hexagon_side_length (distance_between_opposite_sides : ℝ) : 
  distance_between_opposite_sides > 0 →
  ∃ (side_length : ℝ), 
    side_length = (20 * Real.sqrt 3) / 3 * distance_between_opposite_sides / 10 := by
  sorry

end regular_hexagon_side_length_l165_16549


namespace round_trip_combinations_l165_16506

def num_flights_A_to_B : ℕ := 2
def num_flights_B_to_A : ℕ := 3

theorem round_trip_combinations : num_flights_A_to_B * num_flights_B_to_A = 6 := by
  sorry

end round_trip_combinations_l165_16506


namespace angle_measure_l165_16530

theorem angle_measure : ∃ (x : ℝ), 
  (180 - x = 7 * (90 - x)) ∧ 
  (0 < x) ∧ 
  (x < 180) ∧ 
  (x = 75) := by
  sorry

end angle_measure_l165_16530


namespace don_bottles_from_shop_c_l165_16559

/-- The total number of bottles Don can buy -/
def total_bottles : ℕ := 550

/-- The number of bottles Don buys from Shop A -/
def shop_a_bottles : ℕ := 150

/-- The number of bottles Don buys from Shop B -/
def shop_b_bottles : ℕ := 180

/-- The number of bottles Don buys from Shop C -/
def shop_c_bottles : ℕ := total_bottles - (shop_a_bottles + shop_b_bottles)

theorem don_bottles_from_shop_c :
  shop_c_bottles = 220 :=
by sorry

end don_bottles_from_shop_c_l165_16559


namespace m_intersect_n_equals_open_one_closed_three_l165_16583

-- Define the sets M and N
def M : Set ℝ := {x | Real.log x > 0}
def N : Set ℝ := {x | x^2 ≤ 9}

-- State the theorem
theorem m_intersect_n_equals_open_one_closed_three : M ∩ N = Set.Ioo 1 3 := by sorry

end m_intersect_n_equals_open_one_closed_three_l165_16583


namespace prairie_size_and_untouched_percentage_l165_16552

/-- Represents the prairie and its natural events -/
structure Prairie where
  dust_storm1 : ℕ
  dust_storm2 : ℕ
  flood : ℕ
  wildfire : ℕ
  untouched : ℕ
  affected : ℕ

/-- The prairie with given conditions -/
def our_prairie : Prairie :=
  { dust_storm1 := 75000
  , dust_storm2 := 120000
  , flood := 30000
  , wildfire := 80000
  , untouched := 5000
  , affected := 290000
  }

/-- The theorem stating the total size and untouched percentage of the prairie -/
theorem prairie_size_and_untouched_percentage (p : Prairie) 
  (h : p = our_prairie) : 
  (p.affected + p.untouched = 295000) ∧ 
  (p.untouched : ℚ) / (p.affected + p.untouched : ℚ) * 100 = 5000 / 295000 * 100 := by
  sorry

#eval (our_prairie.untouched : ℚ) / (our_prairie.affected + our_prairie.untouched : ℚ) * 100

end prairie_size_and_untouched_percentage_l165_16552


namespace cos_sum_less_than_sum_of_cos_l165_16574

theorem cos_sum_less_than_sum_of_cos (α β : Real) 
  (h1 : 0 < α ∧ α < π / 2) (h2 : 0 < β ∧ β < π / 2) : 
  Real.cos (α + β) < Real.cos α + Real.cos β := by
  sorry

end cos_sum_less_than_sum_of_cos_l165_16574


namespace prob_other_side_red_given_red_l165_16572

/-- Represents the types of cards in the box -/
inductive Card
  | BlackBlack
  | BlackRed
  | RedRed

/-- The total number of cards in the box -/
def total_cards : Nat := 7

/-- The number of black-black cards -/
def black_black_cards : Nat := 2

/-- The number of black-red cards -/
def black_red_cards : Nat := 3

/-- The number of red-red cards -/
def red_red_cards : Nat := 2

/-- The total number of red faces -/
def total_red_faces : Nat := black_red_cards + 2 * red_red_cards

/-- The number of red faces on completely red cards -/
def red_faces_on_red_cards : Nat := 2 * red_red_cards

/-- The probability of seeing a red face and the other side being red -/
theorem prob_other_side_red_given_red (h1 : total_cards = black_black_cards + black_red_cards + red_red_cards)
  (h2 : total_red_faces = black_red_cards + 2 * red_red_cards)
  (h3 : red_faces_on_red_cards = 2 * red_red_cards) :
  (red_faces_on_red_cards : ℚ) / total_red_faces = 4 / 7 := by
  sorry

end prob_other_side_red_given_red_l165_16572


namespace T_mod_1000_l165_16581

/-- The sum of all four-digit positive integers with four distinct digits -/
def T : ℕ := sorry

/-- Theorem stating that T mod 1000 = 465 -/
theorem T_mod_1000 : T % 1000 = 465 := by sorry

end T_mod_1000_l165_16581


namespace max_product_two_digit_numbers_l165_16591

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def unique_digits (a b c d e : ℕ) : Prop :=
  let digits := a.digits 10 ++ b.digits 10 ++ c.digits 10 ++ d.digits 10 ++ e.digits 10
  digits.length = 10 ∧ digits.toFinset.card = 10

theorem max_product_two_digit_numbers :
  ∃ (a b c d e : ℕ),
    is_two_digit a ∧ is_two_digit b ∧ is_two_digit c ∧ is_two_digit d ∧ is_two_digit e ∧
    unique_digits a b c d e ∧
    a * b * c * d * e = 1785641760 ∧
    ∀ (x y z w v : ℕ),
      is_two_digit x ∧ is_two_digit y ∧ is_two_digit z ∧ is_two_digit w ∧ is_two_digit v ∧
      unique_digits x y z w v →
      x * y * z * w * v ≤ 1785641760 :=
by sorry

end max_product_two_digit_numbers_l165_16591


namespace hyperbola_eccentricity_l165_16571

/-- The eccentricity of a hyperbola with given properties -/
theorem hyperbola_eccentricity : ∀ (a : ℝ),
  (∀ (x y : ℝ), x^2 / a^2 - y^2 / 9 = 1 → 
    ∃ (c : ℝ), c = 4 ∧ c^2 = a^2 + 9) →
  (4 : ℝ) * Real.sqrt 7 / 7 = 
    (4 : ℝ) / Real.sqrt (a^2) := by sorry

end hyperbola_eccentricity_l165_16571


namespace worker_problem_l165_16578

theorem worker_problem (time_B time_together : ℝ) 
  (h1 : time_B = 10)
  (h2 : time_together = 4.444444444444445)
  (h3 : 1 / time_A + 1 / time_B = 1 / time_together) :
  time_A = 8 :=
sorry

end worker_problem_l165_16578


namespace count_integers_between_square_roots_l165_16526

theorem count_integers_between_square_roots : 
  (Finset.range 25 \ Finset.range 10).card = 15 := by sorry

end count_integers_between_square_roots_l165_16526


namespace min_volume_base_area_is_8d_squared_l165_16547

/-- Regular quadrilateral pyramid with a plane bisecting the dihedral angle -/
structure RegularPyramid where
  /-- Distance from the base to the intersection point of the bisecting plane and the height -/
  d : ℝ
  /-- The plane bisects the dihedral angle at a side of the base -/
  bisects_dihedral_angle : True

/-- The area of the base that minimizes the volume of the pyramid -/
def min_volume_base_area (p : RegularPyramid) : ℝ := 8 * p.d^2

/-- Theorem stating that the area of the base minimizing the volume is 8d^2 -/
theorem min_volume_base_area_is_8d_squared (p : RegularPyramid) :
  min_volume_base_area p = 8 * p.d^2 := by
  sorry

end min_volume_base_area_is_8d_squared_l165_16547


namespace function_is_identity_l165_16568

def IsNonDegenerateTriangle (a b c : ℕ+) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

def SatisfiesTriangleCondition (f : ℕ+ → ℕ+) : Prop :=
  ∀ a b : ℕ+, IsNonDegenerateTriangle a (f b) (f (b + f a - 1))

theorem function_is_identity (f : ℕ+ → ℕ+) 
  (h : SatisfiesTriangleCondition f) : 
  ∀ x : ℕ+, f x = x := by
  sorry

end function_is_identity_l165_16568


namespace cost_price_is_1200_l165_16596

/-- Calculates the cost price of a toy given the total selling price, number of toys sold, and the gain condition. -/
def cost_price_of_toy (total_selling_price : ℕ) (num_toys_sold : ℕ) (num_toys_gain : ℕ) : ℕ :=
  let selling_price_per_toy := total_selling_price / num_toys_sold
  let cost_price := selling_price_per_toy * num_toys_sold / (num_toys_sold + num_toys_gain)
  cost_price

/-- Theorem stating that under the given conditions, the cost price of a toy is 1200. -/
theorem cost_price_is_1200 :
  cost_price_of_toy 50400 36 6 = 1200 := by
  sorry

#eval cost_price_of_toy 50400 36 6

end cost_price_is_1200_l165_16596


namespace smallest_sum_of_digits_l165_16567

def sum_of_digits (n : ℕ) : ℕ := sorry

theorem smallest_sum_of_digits :
  (∀ n : ℕ, sum_of_digits (3 * n^2 + n + 1) ≥ 3) ∧
  (∃ n : ℕ, sum_of_digits (3 * n^2 + n + 1) = 3) := by sorry

end smallest_sum_of_digits_l165_16567


namespace angle_b_value_l165_16507

-- Define the angles
variable (a b c : ℝ)

-- Define the conditions
axiom straight_line : a + b + c = 180
axiom ratio_b_a : b = 2 * a
axiom ratio_c_b : c = 3 * b

-- Theorem to prove
theorem angle_b_value : b = 40 := by
  sorry

end angle_b_value_l165_16507


namespace average_charge_is_five_l165_16586

/-- Represents the charges and attendance for a three-day show -/
structure ShowData where
  day1_charge : ℚ
  day2_charge : ℚ
  day3_charge : ℚ
  day1_attendance : ℚ
  day2_attendance : ℚ
  day3_attendance : ℚ

/-- Calculates the average charge per person for the whole show -/
def averageCharge (data : ShowData) : ℚ :=
  let total_revenue := data.day1_charge * data.day1_attendance +
                       data.day2_charge * data.day2_attendance +
                       data.day3_charge * data.day3_attendance
  let total_attendance := data.day1_attendance + data.day2_attendance + data.day3_attendance
  total_revenue / total_attendance

/-- Theorem stating that the average charge for the given show data is 5 -/
theorem average_charge_is_five (data : ShowData)
  (h1 : data.day1_charge = 15)
  (h2 : data.day2_charge = 15/2)
  (h3 : data.day3_charge = 5/2)
  (h4 : data.day1_attendance = 2 * x)
  (h5 : data.day2_attendance = 5 * x)
  (h6 : data.day3_attendance = 13 * x)
  (h7 : x > 0) :
  averageCharge data = 5 := by
  sorry

end average_charge_is_five_l165_16586


namespace fourth_root_squared_cubed_l165_16576

theorem fourth_root_squared_cubed (x : ℝ) : ((x^(1/4))^2)^3 = 1296 → x = 256 := by
  sorry

end fourth_root_squared_cubed_l165_16576


namespace archer_fish_count_l165_16521

/-- The total number of fish Archer caught in a day -/
def total_fish (first_round : ℕ) (second_round_increase : ℕ) (third_round_percentage : ℕ) : ℕ :=
  let second_round := first_round + second_round_increase
  let third_round := second_round + (third_round_percentage * second_round) / 100
  first_round + second_round + third_round

/-- Theorem stating that Archer caught 60 fish in total -/
theorem archer_fish_count : total_fish 8 12 60 = 60 := by
  sorry

end archer_fish_count_l165_16521


namespace maximize_electronic_thermometers_l165_16570

/-- Represents the problem of maximizing electronic thermometers purchase --/
theorem maximize_electronic_thermometers
  (total_budget : ℕ)
  (mercury_cost : ℕ)
  (electronic_cost : ℕ)
  (total_students : ℕ)
  (h1 : total_budget = 300)
  (h2 : mercury_cost = 3)
  (h3 : electronic_cost = 10)
  (h4 : total_students = 53) :
  ∃ (x : ℕ), x ≤ total_students ∧
             x * electronic_cost + (total_students - x) * mercury_cost ≤ total_budget ∧
             ∀ (y : ℕ), y ≤ total_students →
                        y * electronic_cost + (total_students - y) * mercury_cost ≤ total_budget →
                        y ≤ x ∧
             x = 20 :=
by sorry

end maximize_electronic_thermometers_l165_16570


namespace unique_valid_denomination_l165_16575

def is_valid_denomination (n : ℕ) : Prop :=
  ∀ k : ℕ, k > 104 → ∃ a b c : ℕ, k = 7 * a + n * b + (n + 2) * c ∧
  ¬∃ a b c : ℕ, 104 = 7 * a + n * b + (n + 2) * c

theorem unique_valid_denomination :
  ∃! n : ℕ, n > 0 ∧ is_valid_denomination n :=
sorry

end unique_valid_denomination_l165_16575


namespace cost_to_selling_price_ratio_l165_16593

theorem cost_to_selling_price_ratio (cost_price selling_price : ℝ) 
  (h_profit : selling_price = cost_price * (1 + 0.25)) :
  cost_price / selling_price = 4 / 5 := by
sorry

end cost_to_selling_price_ratio_l165_16593


namespace different_solution_D_same_solution_B_same_solution_C_l165_16529

-- Define the reference equation
def reference_equation (x : ℚ) : Prop := x - 3 = 3 * x + 4

-- Define the equations from options B, C, and D
def equation_B (x : ℚ) : Prop := 1 / (x + 3) + 2 = 0
def equation_C (x a : ℚ) : Prop := (a^2 + 1) * (x - 3) = (3 * x + 4) * (a^2 + 1)
def equation_D (x : ℚ) : Prop := (7 * x - 4) * (x - 1) = (5 * x - 11) * (x - 1)

-- Theorem stating that D has a different solution set
theorem different_solution_D :
  ∃ x : ℚ, equation_D x ∧ ¬(reference_equation x) :=
sorry

-- Theorems stating that B and C have the same solution as the reference equation
theorem same_solution_B :
  ∀ x : ℚ, equation_B x ↔ reference_equation x :=
sorry

theorem same_solution_C :
  ∀ x a : ℚ, equation_C x a ↔ reference_equation x :=
sorry

end different_solution_D_same_solution_B_same_solution_C_l165_16529


namespace fly_distance_from_ceiling_l165_16594

theorem fly_distance_from_ceiling (z : ℝ) : 
  (2:ℝ)^2 + 6^2 + z^2 = 11^2 → z = 9 := by
  sorry

end fly_distance_from_ceiling_l165_16594


namespace total_rainfall_calculation_l165_16595

theorem total_rainfall_calculation (storm1_rate : ℝ) (storm2_rate : ℝ) 
  (total_duration : ℝ) (storm1_duration : ℝ) :
  storm1_rate = 30 →
  storm2_rate = 15 →
  total_duration = 45 →
  storm1_duration = 20 →
  storm1_rate * storm1_duration + storm2_rate * (total_duration - storm1_duration) = 975 := by
  sorry

end total_rainfall_calculation_l165_16595


namespace expression_value_l165_16535

theorem expression_value : 
  let x : ℝ := 3
  5 * 7 + 9 * 4 - 35 / 5 + x * 2 = 70 := by
sorry

end expression_value_l165_16535


namespace fixed_point_theorem_l165_16534

/-- The line equation as a function of k, x, and y -/
def line_equation (k x y : ℝ) : ℝ := (2*k + 1)*x + (1 - k)*y + 7 - k

/-- The theorem stating that (-2, -5) is a fixed point of the line for all real k -/
theorem fixed_point_theorem :
  ∀ k : ℝ, line_equation k (-2) (-5) = 0 := by
  sorry

end fixed_point_theorem_l165_16534


namespace simplify_expression_l165_16555

theorem simplify_expression (a b c : ℝ) (h1 : a + b + c = 0) (h2 : a ≠ 0) (h3 : b ≠ 0) (h4 : c ≠ 0) :
  a * (1 / b + 1 / c) + b * (1 / a + 1 / c) + c * (1 / a + 1 / b) = -3 := by
  sorry

end simplify_expression_l165_16555


namespace cyclic_quadrilateral_diagonal_length_l165_16565

-- Define the cyclic quadrilateral ABCD and point K
variable (A B C D K : Point)

-- Define the property of being a cyclic quadrilateral
def is_cyclic_quadrilateral (A B C D : Point) : Prop := sorry

-- Define the property of K being the intersection of diagonals
def is_diagonal_intersection (A B C D K : Point) : Prop := sorry

-- Define the distance between two points
def distance (P Q : Point) : ℝ := sorry

-- State the theorem
theorem cyclic_quadrilateral_diagonal_length
  (h_cyclic : is_cyclic_quadrilateral A B C D)
  (h_diagonal : is_diagonal_intersection A B C D K)
  (h_equal_sides : distance A B = distance B C)
  (h_BK : distance B K = b)
  (h_DK : distance D K = d)
  : distance A B = Real.sqrt (b^2 + b*d) := by sorry

end cyclic_quadrilateral_diagonal_length_l165_16565


namespace train_speed_l165_16519

/-- The speed of a train given the time to pass an electric pole and a platform -/
theorem train_speed (pole_time : ℝ) (platform_length : ℝ) (platform_time : ℝ) :
  pole_time = 15 →
  platform_length = 380 →
  platform_time = 52.99696024318054 →
  ∃ (speed : ℝ), abs (speed - 36.0037908) < 0.0000001 := by
  sorry

end train_speed_l165_16519
