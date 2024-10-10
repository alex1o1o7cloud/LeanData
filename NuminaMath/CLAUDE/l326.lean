import Mathlib

namespace probability_factor_less_than_eight_l326_32641

theorem probability_factor_less_than_eight (n : ℕ) (h : n = 90) :
  let factors := {d : ℕ | d > 0 ∧ n % d = 0}
  let factors_less_than_eight := {d ∈ factors | d < 8}
  Nat.card factors_less_than_eight / Nat.card factors = 5 / 12 := by
  sorry

end probability_factor_less_than_eight_l326_32641


namespace move_point_right_point_B_position_l326_32619

def point_on_number_line (x : ℤ) := x

theorem move_point_right (start : ℤ) (distance : ℕ) :
  point_on_number_line (start + distance) = point_on_number_line start + distance :=
by sorry

theorem point_B_position :
  let point_A := point_on_number_line (-3)
  let move_distance := 4
  let point_B := point_on_number_line (point_A + move_distance)
  point_B = 1 :=
by sorry

end move_point_right_point_B_position_l326_32619


namespace system_solution_exists_no_solution_when_m_eq_one_l326_32656

theorem system_solution_exists (m : ℝ) (h : m ≠ 1) :
  ∃ (x y : ℝ), y = m * x + 4 ∧ y = (3 * m - 2) * x + 5 := by
  sorry

theorem no_solution_when_m_eq_one :
  ¬ ∃ (x y : ℝ), y = 1 * x + 4 ∧ y = (3 * 1 - 2) * x + 5 := by
  sorry

end system_solution_exists_no_solution_when_m_eq_one_l326_32656


namespace smallest_integer_satisfying_inequality_l326_32684

theorem smallest_integer_satisfying_inequality : 
  ∃ x : ℤ, (∀ y : ℤ, (5 : ℚ) / 8 < (y + 3 : ℚ) / 15 → x ≤ y) ∧ (5 : ℚ) / 8 < (x + 3 : ℚ) / 15 :=
by sorry

end smallest_integer_satisfying_inequality_l326_32684


namespace john_average_speed_l326_32658

/-- John's average speed in miles per hour -/
def john_speed : ℝ := 30

/-- Carla's average speed in miles per hour -/
def carla_speed : ℝ := 35

/-- Time Carla needs to catch up to John in hours -/
def catch_up_time : ℝ := 3

/-- Time difference between John's and Carla's departure in hours -/
def departure_time_difference : ℝ := 0.5

theorem john_average_speed :
  john_speed = 30 ∧
  carla_speed * catch_up_time = john_speed * (catch_up_time + departure_time_difference) :=
sorry

end john_average_speed_l326_32658


namespace base7_to_base10_ABC21_l326_32624

/-- Converts a base 7 number to base 10 --/
def base7ToBase10 (a b c : Nat) : Nat :=
  a * 2401 + b * 343 + c * 49 + 15

/-- Theorem: The base 10 equivalent of ABC21₇ is A · 2401 + B · 343 + C · 49 + 15 --/
theorem base7_to_base10_ABC21 (A B C : Nat) 
  (hA : A ≤ 6) (hB : B ≤ 6) (hC : C ≤ 6) :
  base7ToBase10 A B C = A * 2401 + B * 343 + C * 49 + 15 := by
  sorry

#check base7_to_base10_ABC21

end base7_to_base10_ABC21_l326_32624


namespace books_together_l326_32652

/-- The number of books Keith and Jason have together -/
def total_books (keith_books jason_books : ℕ) : ℕ :=
  keith_books + jason_books

/-- Theorem: Keith and Jason have 41 books together -/
theorem books_together : total_books 20 21 = 41 := by
  sorry

end books_together_l326_32652


namespace beth_comic_books_percentage_l326_32668

theorem beth_comic_books_percentage
  (total_books : ℕ)
  (novel_percentage : ℚ)
  (graphic_novels : ℕ)
  (h1 : total_books = 120)
  (h2 : novel_percentage = 65/100)
  (h3 : graphic_novels = 18) :
  (total_books - (novel_percentage * total_books).floor - graphic_novels) / total_books = 1/5 := by
sorry

end beth_comic_books_percentage_l326_32668


namespace dice_sum_impossibility_l326_32627

theorem dice_sum_impossibility (a b c d : ℕ) : 
  1 ≤ a ∧ a ≤ 6 →
  1 ≤ b ∧ b ≤ 6 →
  1 ≤ c ∧ c ≤ 6 →
  1 ≤ d ∧ d ≤ 6 →
  a * b * c * d = 216 →
  a + b + c + d ≠ 20 := by
sorry

end dice_sum_impossibility_l326_32627


namespace car_A_time_is_5_hours_l326_32613

/-- Represents the properties of a car's journey -/
structure CarJourney where
  speed : ℝ
  time : ℝ
  distance : ℝ

/-- The problem setup -/
def problem : Prop :=
  ∃ (carA carB : CarJourney),
    carA.speed = 80 ∧
    carB.speed = 100 ∧
    carB.time = 2 ∧
    carA.distance = 2 * carB.distance ∧
    carA.distance = carA.speed * carA.time ∧
    carB.distance = carB.speed * carB.time

/-- The theorem to prove -/
theorem car_A_time_is_5_hours (h : problem) : 
  ∃ (carA : CarJourney), carA.time = 5 := by
  sorry


end car_A_time_is_5_hours_l326_32613


namespace strawberry_harvest_l326_32692

/-- Represents the dimensions of a rectangular garden -/
structure GardenDimensions where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangular garden -/
def gardenArea (d : GardenDimensions) : ℝ :=
  d.length * d.width

/-- Calculates the total number of plants in the garden -/
def totalPlants (d : GardenDimensions) (density : ℝ) : ℝ :=
  gardenArea d * density

/-- Calculates the total number of strawberries harvested -/
def totalStrawberries (d : GardenDimensions) (density : ℝ) (yield : ℝ) : ℝ :=
  totalPlants d density * yield

/-- Theorem: The total number of strawberries harvested is 5400 -/
theorem strawberry_harvest (d : GardenDimensions) (density : ℝ) (yield : ℝ)
    (h1 : d.length = 10)
    (h2 : d.width = 9)
    (h3 : density = 5)
    (h4 : yield = 12) :
    totalStrawberries d density yield = 5400 := by
  sorry

#eval totalStrawberries ⟨10, 9⟩ 5 12

end strawberry_harvest_l326_32692


namespace largest_integer_with_remainder_ninety_five_satisfies_conditions_largest_integer_is_95_l326_32642

theorem largest_integer_with_remainder (n : ℕ) : n < 100 ∧ n % 7 = 4 → n ≤ 95 :=
by
  sorry

theorem ninety_five_satisfies_conditions : 95 < 100 ∧ 95 % 7 = 4 :=
by
  sorry

theorem largest_integer_is_95 : ∀ (n : ℕ), n < 100 ∧ n % 7 = 4 → n ≤ 95 :=
by
  sorry

end largest_integer_with_remainder_ninety_five_satisfies_conditions_largest_integer_is_95_l326_32642


namespace shares_problem_l326_32686

theorem shares_problem (total : ℕ) (a b c : ℕ) : 
  total = 1760 →
  a + b + c = total →
  3 * b = 4 * a →
  5 * a = 3 * c →
  6 * a = 8 * b →
  8 * b = 20 * c →
  c = 250 := by
sorry

end shares_problem_l326_32686


namespace subtraction_of_decimals_l326_32644

theorem subtraction_of_decimals : 3.75 - 1.46 = 2.29 := by
  sorry

end subtraction_of_decimals_l326_32644


namespace alpha_plus_beta_equals_113_l326_32659

theorem alpha_plus_beta_equals_113 (α β : ℝ) : 
  (∀ x : ℝ, x ≠ 45 → (x - α) / (x + β) = (x^2 - 90*x + 1981) / (x^2 + 63*x - 3420)) →
  α + β = 113 := by
sorry

end alpha_plus_beta_equals_113_l326_32659


namespace mass_percentage_H_in_C9H14N3O5_l326_32662

/-- Molar mass of carbon in g/mol -/
def molar_mass_C : ℝ := 12.01

/-- Molar mass of hydrogen in g/mol -/
def molar_mass_H : ℝ := 1.01

/-- Molar mass of nitrogen in g/mol -/
def molar_mass_N : ℝ := 14.01

/-- Molar mass of oxygen in g/mol -/
def molar_mass_O : ℝ := 16.00

/-- Calculate the mass percentage of hydrogen in C9H14N3O5 -/
theorem mass_percentage_H_in_C9H14N3O5 :
  let total_mass := 9 * molar_mass_C + 14 * molar_mass_H + 3 * molar_mass_N + 5 * molar_mass_O
  let mass_H := 14 * molar_mass_H
  let percentage := (mass_H / total_mass) * 100
  ∃ ε > 0, |percentage - 5.79| < ε :=
sorry

end mass_percentage_H_in_C9H14N3O5_l326_32662


namespace tangent_curve_relation_l326_32607

noncomputable def tangent_line (a : ℝ) (x : ℝ) : ℝ := x + a

noncomputable def curve (b : ℝ) (x : ℝ) : ℝ := Real.exp (x - 1) - b + 1

theorem tangent_curve_relation (a b : ℝ) :
  (∃ x₀ : ℝ, tangent_line a x₀ = curve b x₀ ∧ 
    (deriv (tangent_line a)) x₀ = (deriv (curve b)) x₀) →
  a + b = 1 := by
sorry

end tangent_curve_relation_l326_32607


namespace tank_width_is_four_feet_l326_32600

/-- Proves that the width of a rectangular tank is 4 feet given specific conditions. -/
theorem tank_width_is_four_feet 
  (fill_rate : ℝ) 
  (length depth time_to_fill : ℝ) 
  (h_fill_rate : fill_rate = 4)
  (h_length : length = 6)
  (h_depth : depth = 3)
  (h_time_to_fill : time_to_fill = 18)
  : (fill_rate * time_to_fill) / (length * depth) = 4 := by
  sorry

end tank_width_is_four_feet_l326_32600


namespace parking_probability_probability_equals_actual_l326_32673

/-- The probability of finding 3 consecutive empty spaces in a row of 18 spaces 
    where 14 spaces are randomly occupied -/
theorem parking_probability : ℝ := by
  -- Define the total number of spaces
  let total_spaces : ℕ := 18
  -- Define the number of occupied spaces
  let occupied_spaces : ℕ := 14
  -- Define the number of consecutive empty spaces needed
  let required_empty_spaces : ℕ := 3
  
  -- Calculate the probability
  -- We're not providing the actual calculation here, just the structure
  sorry

-- The actual probability value
def actual_probability : ℚ := 171 / 204

-- Prove that the calculated probability equals the actual probability
theorem probability_equals_actual : parking_probability = actual_probability := by
  sorry

end parking_probability_probability_equals_actual_l326_32673


namespace sum_of_square_roots_l326_32654

theorem sum_of_square_roots (x : ℝ) 
  (h1 : -Real.sqrt 15 ≤ x ∧ x ≤ Real.sqrt 15) 
  (h2 : Real.sqrt (25 - x^2) - Real.sqrt (15 - x^2) = 2) : 
  Real.sqrt (25 - x^2) + Real.sqrt (15 - x^2) = 5 := by
sorry

end sum_of_square_roots_l326_32654


namespace charge_per_mile_calculation_l326_32669

/-- Proves that the charge per mile is $0.25 given the rental fee, total amount paid, and miles driven -/
theorem charge_per_mile_calculation (rental_fee total_paid miles_driven : ℚ) 
  (h1 : rental_fee = 20.99)
  (h2 : total_paid = 95.74)
  (h3 : miles_driven = 299) :
  (total_paid - rental_fee) / miles_driven = 0.25 := by
  sorry

end charge_per_mile_calculation_l326_32669


namespace triangle_base_length_l326_32602

/-- Proves that a triangle with area 54 square meters and height 6 meters has a base of 18 meters -/
theorem triangle_base_length (area : ℝ) (height : ℝ) (base : ℝ) :
  area = 54 →
  height = 6 →
  area = (base * height) / 2 →
  base = 18 := by
sorry

end triangle_base_length_l326_32602


namespace corresponding_angles_equal_l326_32606

-- Define the concept of a triangle
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define a function to get the angles of a triangle
def angles (t : Triangle) : ℝ × ℝ × ℝ :=
  sorry

-- Define the property that corresponding angles are either equal or sum to 180°
def corresponding_angles_property (t1 t2 : Triangle) : Prop :=
  let (α1, β1, γ1) := angles t1
  let (α2, β2, γ2) := angles t2
  (α1 = α2 ∨ α1 + α2 = 180) ∧
  (β1 = β2 ∨ β1 + β2 = 180) ∧
  (γ1 = γ2 ∨ γ1 + γ2 = 180)

-- Theorem statement
theorem corresponding_angles_equal (t1 t2 : Triangle) 
  (h : corresponding_angles_property t1 t2) : 
  angles t1 = angles t2 :=
sorry

end corresponding_angles_equal_l326_32606


namespace apple_balance_theorem_l326_32661

variable {α : Type*} [LinearOrderedField α]

def balanced (s t : Finset (α)) : Prop :=
  s.sum id = t.sum id

theorem apple_balance_theorem
  (apples : Finset α)
  (h_count : apples.card = 6)
  (h_tanya : ∃ (s t : Finset α), s ⊆ apples ∧ t ⊆ apples ∧ s ∩ t = ∅ ∧ s ∪ t = apples ∧ s.card = 3 ∧ t.card = 3 ∧ balanced s t)
  (h_sasha : ∃ (u v : Finset α), u ⊆ apples ∧ v ⊆ apples ∧ u ∩ v = ∅ ∧ u ∪ v = apples ∧ u.card = 2 ∧ v.card = 4 ∧ balanced u v) :
  ∃ (x y : Finset α), x ⊆ apples ∧ y ⊆ apples ∧ x ∩ y = ∅ ∧ x ∪ y = apples ∧ x.card = 1 ∧ y.card = 2 ∧ balanced x y :=
by
  sorry

end apple_balance_theorem_l326_32661


namespace f_properties_l326_32640

noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * (Real.cos x)^2 + 2 * Real.sin x * Real.cos x - Real.sqrt 3 * (Real.sin x)^2

theorem f_properties :
  (∀ x, f (x + π/12) = f (π/12 - x)) ∧
  (∀ x, f (π/3 + x) = -f (π/3 - x)) ∧
  (∃ x₁ x₂, |f x₁ - f x₂| ≥ 4) :=
by sorry

end f_properties_l326_32640


namespace right_triangle_area_l326_32622

/-- The area of a right triangle with one leg of 12cm and a hypotenuse of 13cm is 30 cm². -/
theorem right_triangle_area (a b c : ℝ) (h1 : a = 12) (h2 : c = 13) 
    (h3 : a^2 + b^2 = c^2) : (1/2) * a * b = 30 := by
  sorry

end right_triangle_area_l326_32622


namespace volleyball_team_selection_l326_32605

def choose (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

theorem volleyball_team_selection :
  let total_players : ℕ := 14
  let triplets : ℕ := 3
  let starters : ℕ := 6
  let non_triplet_players : ℕ := total_players - triplets
  let lineups_without_triplets : ℕ := choose non_triplet_players starters
  let lineups_with_one_triplet : ℕ := triplets * (choose non_triplet_players (starters - 1))
  lineups_without_triplets + lineups_with_one_triplet = 1848 :=
by sorry

end volleyball_team_selection_l326_32605


namespace shelter_ratio_change_l326_32620

/-- Proves that given an initial ratio of dogs to cats of 15:7, 60 dogs in the shelter,
    and 16 additional cats taken in, the new ratio of dogs to cats is 15:11. -/
theorem shelter_ratio_change (initial_dogs : ℕ) (initial_cats : ℕ) (additional_cats : ℕ) :
  initial_dogs = 60 →
  initial_dogs / initial_cats = 15 / 7 →
  additional_cats = 16 →
  initial_dogs / (initial_cats + additional_cats) = 15 / 11 := by
  sorry

end shelter_ratio_change_l326_32620


namespace x_value_proof_l326_32603

theorem x_value_proof (a b c x : ℝ) 
  (eq1 : a - b + c = 5)
  (eq2 : a^2 + b^2 + c^2 = 29)
  (eq3 : a*b + b*c + a*c = x^2) :
  x = Real.sqrt 2 := by
  sorry

end x_value_proof_l326_32603


namespace fraction_equality_sum_l326_32618

theorem fraction_equality_sum (M N : ℚ) : 
  (4 : ℚ) / 7 = M / 63 ∧ (4 : ℚ) / 7 = 84 / N → M + N = 183 := by
  sorry

end fraction_equality_sum_l326_32618


namespace max_value_quadratic_max_value_sum_products_l326_32674

-- Part 1
theorem max_value_quadratic (a : ℝ) (h : a > 0) :
  ∃ (max : ℝ), ∀ (x : ℝ), x > 0 → a > 2*x → x*(a - 2*x) ≤ max ∧
  ∃ (x_max : ℝ), x_max > 0 ∧ a > 2*x_max ∧ x_max*(a - 2*x_max) = max :=
sorry

-- Part 2
theorem max_value_sum_products (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h_sum_squares : a^2 + b^2 + c^2 = 4) :
  a*b + b*c + a*c ≤ 4 ∧ ∃ (a' b' c' : ℝ), a' > 0 ∧ b' > 0 ∧ c' > 0 ∧
  a'^2 + b'^2 + c'^2 = 4 ∧ a'*b' + b'*c' + a'*c' = 4 :=
sorry

end max_value_quadratic_max_value_sum_products_l326_32674


namespace inequality_proof_l326_32612

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a / (b + c)) + (b / (c + a)) + (c / (a + b)) ≥ (3 : ℝ) / 2 := by
  sorry

end inequality_proof_l326_32612


namespace shift_direct_proportion_l326_32633

def original_function (x : ℝ) : ℝ := -2 * x

def shift_right (f : ℝ → ℝ) (shift : ℝ) : ℝ → ℝ :=
  fun x => f (x - shift)

def resulting_function (x : ℝ) : ℝ := -2 * x + 6

theorem shift_direct_proportion :
  shift_right original_function 3 = resulting_function := by
  sorry

end shift_direct_proportion_l326_32633


namespace min_value_of_D_l326_32625

noncomputable def D (x a : ℝ) : ℝ := Real.sqrt ((x - a)^2 + (Real.exp x - 2 * Real.sqrt a)) + a + 2

theorem min_value_of_D :
  ∃ (min_D : ℝ), min_D = Real.sqrt 2 + 1 ∧
  ∀ (x a : ℝ), D x a ≥ min_D :=
sorry

end min_value_of_D_l326_32625


namespace tangent_line_and_hyperbola_l326_32617

/-- Given two functions f(x) = x + 4 and g(x) = k/x that are tangent to each other, 
    prove that k = -4 -/
theorem tangent_line_and_hyperbola (k : ℝ) :
  (∃ x : ℝ, x + 4 = k / x ∧ 
   ∀ y : ℝ, y ≠ x → (y + 4 - k / y) * (x - y) ≠ 0) → 
  k = -4 :=
by sorry

end tangent_line_and_hyperbola_l326_32617


namespace lynne_magazines_l326_32698

def num_books : ℕ := 9
def book_cost : ℕ := 7
def magazine_cost : ℕ := 4
def total_spent : ℕ := 75

theorem lynne_magazines :
  ∃ (num_magazines : ℕ),
    num_magazines * magazine_cost + num_books * book_cost = total_spent ∧
    num_magazines = 3 := by
  sorry

end lynne_magazines_l326_32698


namespace log_inequality_l326_32670

theorem log_inequality (x : ℝ) (h : x > 0) :
  9.280 * (Real.log x / Real.log 7) - Real.log 7 * (Real.log x / Real.log 3) > Real.log 0.25 / Real.log 2 ↔ 
  x < 3^(2 / (Real.log 7 / Real.log 3 - Real.log 3 / Real.log 7)) :=
by sorry

end log_inequality_l326_32670


namespace fraction_decomposition_l326_32688

theorem fraction_decomposition (x : ℝ) (h1 : x ≠ -2) (h2 : x ≠ 5/3) :
  (7 * x - 15) / (3 * x^2 - x - 10) = (29/11) / (x + 2) + (-9/11) / (3*x - 5) := by
  sorry

end fraction_decomposition_l326_32688


namespace complement_of_union_A_B_l326_32611

-- Define the universal set U as the set of real numbers
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x | x ≤ 1}

-- Define set B
def B : Set ℝ := {x | x ≥ 2}

-- State the theorem
theorem complement_of_union_A_B : 
  (A ∪ B)ᶜ = {x : ℝ | 1 < x ∧ x < 2} := by sorry

end complement_of_union_A_B_l326_32611


namespace divisor_congruence_l326_32650

theorem divisor_congruence (d : ℕ) (x y : ℤ) : 
  d > 0 ∧ d ∣ (5 + 1998^1998) →
  (d = 2*x^2 + 2*x*y + 3*y^2 ↔ d % 20 = 3 ∨ d % 20 = 7) :=
by sorry

end divisor_congruence_l326_32650


namespace mean_equality_problem_l326_32610

theorem mean_equality_problem : ∃ z : ℚ, (7 + 12 + 21) / 3 = (15 + z) / 2 ∧ z = 35 / 3 := by
  sorry

end mean_equality_problem_l326_32610


namespace prime_power_gcd_condition_l326_32646

theorem prime_power_gcd_condition (n : ℕ) (h : n > 1) :
  (∀ m : ℕ, 1 ≤ m ∧ m < n →
    Nat.gcd n ((n - m) / Nat.gcd n m) = 1) ↔
  ∃ (p : ℕ) (k : ℕ), Nat.Prime p ∧ k > 0 ∧ n = p^k :=
by sorry

end prime_power_gcd_condition_l326_32646


namespace greatest_power_of_two_factor_l326_32689

theorem greatest_power_of_two_factor : ∃ k : ℕ, k = 502 ∧ 
  (∀ m : ℕ, 2^m ∣ (12^1002 - 6^501) → m ≤ k) ∧
  (2^k ∣ (12^1002 - 6^501)) := by
  sorry

end greatest_power_of_two_factor_l326_32689


namespace sin_decreasing_interval_l326_32604

/-- The monotonic decreasing interval of sin(π/3 - 2x) -/
theorem sin_decreasing_interval (k : ℤ) :
  let f : ℝ → ℝ := λ x => Real.sin (π/3 - 2*x)
  ∀ x ∈ Set.Icc (k*π - π/12) (k*π + 5*π/12),
    ∀ y ∈ Set.Icc (k*π - π/12) (k*π + 5*π/12),
      x ≤ y → f y ≤ f x :=
by sorry

end sin_decreasing_interval_l326_32604


namespace point_on_line_l326_32678

/-- Given two points A and B in the Cartesian plane, if a point C satisfies the vector equation
    OC = s*OA + t*OB where s + t = 1, then C lies on the line passing through A and B. -/
theorem point_on_line (A B C : ℝ × ℝ) (s t : ℝ) :
  A = (2, 1) →
  B = (-1, -2) →
  C = s • A + t • B →
  s + t = 1 →
  C.1 - C.2 = 1 := by sorry

end point_on_line_l326_32678


namespace four_equidistant_lines_l326_32623

-- Define the distance between two points
def distance (A B : ℝ × ℝ) : ℝ := sorry

-- Define a line in 2D space
def Line : Type := sorry

-- Define the distance from a point to a line
def point_to_line_distance (p : ℝ × ℝ) (l : Line) : ℝ := sorry

theorem four_equidistant_lines 
  (A B : ℝ × ℝ) 
  (h_distance : distance A B = 8) :
  ∃ (lines : Finset Line), 
    lines.card = 4 ∧ 
    (∀ l ∈ lines, point_to_line_distance A l = 3 ∧ point_to_line_distance B l = 4) ∧
    (∀ l : Line, point_to_line_distance A l = 3 ∧ point_to_line_distance B l = 4 → l ∈ lines) :=
sorry

end four_equidistant_lines_l326_32623


namespace short_story_section_pages_l326_32651

/-- The number of pages in each booklet -/
def pages_per_booklet : ℕ := 9

/-- The number of booklets in the short story section -/
def number_of_booklets : ℕ := 49

/-- The total number of pages in the short story section -/
def total_pages : ℕ := pages_per_booklet * number_of_booklets

theorem short_story_section_pages :
  total_pages = 441 :=
by sorry

end short_story_section_pages_l326_32651


namespace negative_three_less_than_negative_one_l326_32691

theorem negative_three_less_than_negative_one : -3 < -1 := by
  sorry

end negative_three_less_than_negative_one_l326_32691


namespace min_sum_of_squares_l326_32609

theorem min_sum_of_squares (m n : ℕ) (h1 : n = m + 1) (h2 : n^2 - m^2 > 20) :
  ∃ (k : ℕ), k = n^2 + m^2 ∧ k ≥ 221 ∧ ∀ (j : ℕ), (∃ (p q : ℕ), q = p + 1 ∧ q^2 - p^2 > 20 ∧ j = q^2 + p^2) → j ≥ k :=
sorry

end min_sum_of_squares_l326_32609


namespace arithmetic_sequence_property_l326_32634

/-- An arithmetic sequence is a sequence where the difference between
    consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_property (a : ℕ → ℝ) :
  is_arithmetic_sequence a →
  a 1 + a 3 + a 9 = 20 →
  4 * a 5 - a 7 = 20 := by
sorry

end arithmetic_sequence_property_l326_32634


namespace constant_term_expansion_l326_32687

/-- The constant term in the expansion of (3x + 2/x)^8 -/
def constant_term : ℕ := 90720

/-- The binomial coefficient (8 choose 4) -/
def binom_8_4 : ℕ := 70

theorem constant_term_expansion :
  constant_term = binom_8_4 * 3^4 * 2^4 := by sorry

end constant_term_expansion_l326_32687


namespace total_popsicles_l326_32655

/-- The number of grape popsicles in the freezer -/
def grape_popsicles : ℕ := 2

/-- The number of cherry popsicles in the freezer -/
def cherry_popsicles : ℕ := 13

/-- The number of banana popsicles in the freezer -/
def banana_popsicles : ℕ := 2

/-- Theorem stating the total number of popsicles in the freezer -/
theorem total_popsicles : grape_popsicles + cherry_popsicles + banana_popsicles = 17 := by
  sorry

end total_popsicles_l326_32655


namespace all_propositions_false_l326_32649

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel_line_plane : Line → Plane → Prop)
variable (parallel_line_line : Line → Line → Prop)
variable (parallel_plane_plane : Plane → Plane → Prop)
variable (perpendicular_plane : Plane → Plane → Prop)
variable (subset_line_plane : Line → Plane → Prop)

-- Define the theorem
theorem all_propositions_false 
  (a b : Line) (α β γ : Plane) : 
  ¬(∀ a b α, (parallel_line_plane a α ∧ parallel_line_plane b α) → parallel_line_line a b) ∧
  ¬(∀ α β γ, (perpendicular_plane α β ∧ perpendicular_plane β γ) → parallel_plane_plane α γ) ∧
  ¬(∀ a α β, (parallel_line_plane a α ∧ parallel_line_plane a β) → parallel_plane_plane α β) ∧
  ¬(∀ a b α, (parallel_line_line a b ∧ subset_line_plane b α) → parallel_line_plane a α) :=
sorry

end all_propositions_false_l326_32649


namespace geometric_sequence_sum_l326_32653

/-- Given a geometric sequence {a_n} with sum of first n terms S_n = 3^n + t,
    prove that t + a_3 = 17. -/
theorem geometric_sequence_sum (a : ℕ → ℝ) (S : ℕ → ℝ) (t : ℝ) :
  (∀ n, S n = 3^n + t) →
  (∀ n, a (n+1) = S (n+1) - S n) →
  (a 1 * a 3 = (a 2)^2) →
  t + a 3 = 17 := by sorry

end geometric_sequence_sum_l326_32653


namespace min_value_problem_l326_32666

theorem min_value_problem (i j k l m n o p : ℝ) 
  (h1 : i * j * k * l = 16) 
  (h2 : m * n * o * p = 25) : 
  (i * m)^2 + (j * n)^2 + (k * o)^2 + (l * p)^2 ≥ 160 := by
  sorry

end min_value_problem_l326_32666


namespace geometric_sequence_problem_l326_32615

theorem geometric_sequence_problem (a₁ a₂ a₃ : ℝ) (h₁ : a₁ = 280) (h₂ : a₂ > 0) (h₃ : a₃ = 90 / 56) 
  (h₄ : ∃ r : ℝ, a₂ = a₁ * r ∧ a₃ = a₂ * r) : a₂ = 15 * Real.sqrt 2 := by
  sorry

end geometric_sequence_problem_l326_32615


namespace a_zero_necessary_not_sufficient_l326_32690

/-- A complex number z is purely imaginary if its real part is zero and its imaginary part is non-zero. -/
def is_purely_imaginary (z : ℂ) : Prop :=
  z.re = 0 ∧ z.im ≠ 0

/-- The statement that "a = 0" is a necessary but not sufficient condition for "a + bi to be purely imaginary". -/
theorem a_zero_necessary_not_sufficient (a b : ℝ) :
  (∀ z : ℂ, z = a + b * I → is_purely_imaginary z → a = 0) ∧
  (∃ z : ℂ, z = a + b * I ∧ a = 0 ∧ ¬is_purely_imaginary z) :=
sorry

end a_zero_necessary_not_sufficient_l326_32690


namespace xNotEqual1_is_valid_l326_32664

/-- Valid conditional operators -/
inductive ConditionalOperator
  | gt  -- >
  | ge  -- >=
  | lt  -- <
  | ne  -- <>
  | le  -- <=
  | eq  -- =

/-- A conditional expression -/
structure ConditionalExpression where
  operator : ConditionalOperator
  value : ℝ

/-- Check if a conditional expression is valid -/
def isValidConditionalExpression (expr : ConditionalExpression) : Prop :=
  expr.operator ∈ [ConditionalOperator.gt, ConditionalOperator.ge, ConditionalOperator.lt, 
                   ConditionalOperator.ne, ConditionalOperator.le, ConditionalOperator.eq]

/-- The specific conditional expression "x <> 1" -/
def xNotEqual1 : ConditionalExpression :=
  { operator := ConditionalOperator.ne, value := 1 }

/-- Theorem: "x <> 1" is a valid conditional expression -/
theorem xNotEqual1_is_valid : isValidConditionalExpression xNotEqual1 := by
  sorry

end xNotEqual1_is_valid_l326_32664


namespace calculation_proof_l326_32665

theorem calculation_proof : (36 / (9 + 2 - 6)) * 4 = 28.8 := by
  sorry

end calculation_proof_l326_32665


namespace ball_arrangements_count_l326_32695

def num_red_balls : ℕ := 2
def num_yellow_balls : ℕ := 3
def num_white_balls : ℕ := 4
def total_balls : ℕ := num_red_balls + num_yellow_balls + num_white_balls

theorem ball_arrangements_count :
  (Nat.factorial total_balls) / (Nat.factorial num_red_balls * Nat.factorial num_yellow_balls * Nat.factorial num_white_balls) = 1260 := by
  sorry

end ball_arrangements_count_l326_32695


namespace normal_dist_symmetry_l326_32637

-- Define a random variable with normal distribution
def normal_dist (μ σ : ℝ) : Type := ℝ

-- Define the probability function
noncomputable def P (ξ : normal_dist 0 σ) (event : Set ℝ) : ℝ := sorry

-- Theorem statement
theorem normal_dist_symmetry (σ : ℝ) (ξ : normal_dist 0 σ) :
  P ξ {x | x < 2} = 0.8 → P ξ {x | x < -2} = 0.2 := by
  sorry

end normal_dist_symmetry_l326_32637


namespace ratio_fraction_equality_l326_32657

theorem ratio_fraction_equality (A B C : ℚ) (h : A / B = 3 / 2 ∧ B / C = 2 / 5) :
  (4 * A + 3 * B) / (5 * C - 2 * A) = 15 / 19 := by
  sorry

end ratio_fraction_equality_l326_32657


namespace sphere_area_and_volume_l326_32636

theorem sphere_area_and_volume (d : ℝ) (h : d = 6) : 
  let r := d / 2
  (4 * Real.pi * r^2 = 36 * Real.pi) ∧ 
  ((4 / 3) * Real.pi * r^3 = 36 * Real.pi) := by
  sorry

end sphere_area_and_volume_l326_32636


namespace net_amount_theorem_l326_32639

def net_amount_spent (shorts_cost shirt_cost jacket_return : ℚ) : ℚ :=
  shorts_cost + shirt_cost - jacket_return

theorem net_amount_theorem (shorts_cost shirt_cost jacket_return : ℚ) :
  net_amount_spent shorts_cost shirt_cost jacket_return =
  shorts_cost + shirt_cost - jacket_return := by
  sorry

#eval net_amount_spent (13.99 : ℚ) (12.14 : ℚ) (7.43 : ℚ)

end net_amount_theorem_l326_32639


namespace find_n_l326_32699

theorem find_n (x y n : ℝ) (h1 : x = 3) (h2 : y = 27) (h3 : n^(n / (2 + x)) = y) : n = 15 := by
  sorry

end find_n_l326_32699


namespace sin_minus_cos_sqrt_two_l326_32663

theorem sin_minus_cos_sqrt_two (x : Real) :
  0 ≤ x ∧ x < 2 * Real.pi ∧ Real.sin x - Real.cos x = Real.sqrt 2 → x = 3 * Real.pi / 4 := by
  sorry

end sin_minus_cos_sqrt_two_l326_32663


namespace simplify_expression_find_expression_value_evaluate_expression_l326_32629

-- Part 1
theorem simplify_expression (a b : ℝ) :
  10 * (a - b)^4 - 25 * (a - b)^4 + 5 * (a - b)^4 = -10 * (a - b)^4 := by sorry

-- Part 2
theorem find_expression_value (x y : ℝ) (h : 2 * x^2 - 3 * y = 8) :
  4 * x^2 - 6 * y - 32 = -16 := by sorry

-- Part 3
theorem evaluate_expression (a b : ℝ) 
  (h1 : a^2 + 2 * a * b = -5) (h2 : a * b - 2 * b^2 = -3) :
  3 * a^2 + 4 * a * b + 4 * b^2 = -9 := by sorry

end simplify_expression_find_expression_value_evaluate_expression_l326_32629


namespace existence_of_counterexample_l326_32635

theorem existence_of_counterexample :
  ∃ (a b c : ℝ), 
    a > 0 ∧ b > 0 ∧ c > 0 ∧ 
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    b / a ≥ (b + c) / (a + c) :=
by sorry

end existence_of_counterexample_l326_32635


namespace symmetric_points_sum_l326_32645

/-- Given that point A(2, -5) is symmetric with respect to the x-axis to point (m, n), prove that m + n = 7. -/
theorem symmetric_points_sum (m n : ℝ) : 
  (2 = m ∧ -5 = -n) → m + n = 7 := by
  sorry

end symmetric_points_sum_l326_32645


namespace solution_inequality1_no_solution_system_l326_32631

-- Define the inequalities
def inequality1 (x : ℝ) : Prop := (2*x - 2)/3 ≤ 2 - (2*x + 2)/2

def inequality2 (x : ℝ) : Prop := 3*(x - 2) - 1 ≥ -4 - 2*(x - 2)

def inequality3 (x : ℝ) : Prop := (1/3)*(1 - 2*x) > (3*(2*x - 1))/2

-- Theorem for the first inequality
theorem solution_inequality1 : 
  ∀ x : ℝ, inequality1 x ↔ x ≤ 1 := by sorry

-- Theorem for the system of inequalities
theorem no_solution_system : 
  ¬∃ x : ℝ, inequality2 x ∧ inequality3 x := by sorry

end solution_inequality1_no_solution_system_l326_32631


namespace b_nonnegative_l326_32679

-- Define the function f
def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- State the theorem
theorem b_nonnegative 
  (a b c m₁ m₂ : ℝ) 
  (h1 : a > b) 
  (h2 : b > c) 
  (h3 : f a b c 1 = 0) 
  (h4 : a^2 + (f a b c m₁ + f a b c m₂) * a + f a b c m₁ * f a b c m₂ = 0) :
  b ≥ 0 :=
sorry

end b_nonnegative_l326_32679


namespace fox_jeans_purchased_l326_32647

/-- Represents the problem of determining the number of Fox jeans purchased during a sale. -/
theorem fox_jeans_purchased (fox_price pony_price total_savings total_jeans pony_jeans sum_discount_rates pony_discount : ℝ) 
  (h1 : fox_price = 15)
  (h2 : pony_price = 20)
  (h3 : total_savings = 9)
  (h4 : total_jeans = 5)
  (h5 : pony_jeans = 2)
  (h6 : sum_discount_rates = 0.22)
  (h7 : pony_discount = 0.18000000000000014) :
  ∃ fox_jeans : ℝ, fox_jeans = 3 ∧ 
  fox_jeans + pony_jeans = total_jeans ∧
  fox_jeans * (fox_price * (sum_discount_rates - pony_discount)) + 
  pony_jeans * (pony_price * pony_discount) = total_savings :=
sorry

end fox_jeans_purchased_l326_32647


namespace cookie_boxes_problem_l326_32697

theorem cookie_boxes_problem (n : ℕ) : n = 12 ↔ 
  n > 0 ∧ 
  n - 11 ≥ 1 ∧ 
  n - 2 ≥ 1 ∧ 
  (n - 11) + (n - 2) < n ∧
  ∀ m : ℕ, m > n → ¬(m > 0 ∧ m - 11 ≥ 1 ∧ m - 2 ≥ 1 ∧ (m - 11) + (m - 2) < m) :=
by sorry

end cookie_boxes_problem_l326_32697


namespace subset_condition_disjoint_condition_l326_32630

-- Define sets A and B
def A : Set ℝ := {x | 1 < x ∧ x < 2}
def B (a : ℝ) : Set ℝ := {x | 2*a - 1 < x ∧ x < 2*a + 1}

-- Theorem for part (I)
theorem subset_condition (a : ℝ) : A ⊆ B a ↔ 1/2 ≤ a ∧ a ≤ 1 := by sorry

-- Theorem for part (II)
theorem disjoint_condition (a : ℝ) : A ∩ B a = ∅ ↔ a ≥ 3/2 ∨ a ≤ 0 := by sorry

end subset_condition_disjoint_condition_l326_32630


namespace correct_statements_are_1_and_3_l326_32616

-- Define the proof methods
inductive ProofMethod
| Synthetic
| Analytic
| Contradiction

-- Define the properties of proof methods
def isCauseToEffect (m : ProofMethod) : Prop := m = ProofMethod.Synthetic
def isEffectToCause (m : ProofMethod) : Prop := m = ProofMethod.Analytic
def isDirectMethod (m : ProofMethod) : Prop := m = ProofMethod.Synthetic ∨ m = ProofMethod.Analytic
def isIndirectMethod (m : ProofMethod) : Prop := m = ProofMethod.Contradiction

-- Define the statements
def statement1 : Prop := isCauseToEffect ProofMethod.Synthetic
def statement2 : Prop := isIndirectMethod ProofMethod.Analytic
def statement3 : Prop := isEffectToCause ProofMethod.Analytic
def statement4 : Prop := isDirectMethod ProofMethod.Contradiction

-- Theorem to prove
theorem correct_statements_are_1_and_3 :
  (statement1 ∧ statement3) ∧ (¬statement2 ∧ ¬statement4) :=
sorry

end correct_statements_are_1_and_3_l326_32616


namespace odd_function_negative_domain_l326_32676

/-- A function f is odd if f(-x) = -f(x) for all x in its domain -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem odd_function_negative_domain 
  (f : ℝ → ℝ) 
  (h_odd : IsOdd f)
  (h_pos : ∀ x > 0, f x = -x + 1) :
  ∀ x < 0, f x = -x - 1 := by
sorry

end odd_function_negative_domain_l326_32676


namespace robin_final_candy_l326_32638

def initial_candy : ℕ := 23

def eaten_fraction : ℚ := 2/3

def sister_bonus_fraction : ℚ := 1/2

theorem robin_final_candy : 
  ∃ (eaten : ℕ) (leftover : ℕ) (bonus : ℕ),
    eaten = ⌊(eaten_fraction : ℚ) * initial_candy⌋ ∧
    leftover = initial_candy - eaten ∧
    bonus = ⌊(sister_bonus_fraction : ℚ) * initial_candy⌋ ∧
    leftover + bonus = 19 :=
by sorry

end robin_final_candy_l326_32638


namespace solution_to_equation_l326_32660

-- Define the custom operation
def custom_op (a b : ℝ) : ℝ := a * b + a + b + 2

-- State the theorem
theorem solution_to_equation :
  ∃ x : ℝ, custom_op x 3 = 1 ∧ x = -1 := by
  sorry

end solution_to_equation_l326_32660


namespace imaginary_part_of_z_l326_32671

theorem imaginary_part_of_z (z : ℂ) (h : (1 - Complex.I) * z = 1 - 5 * Complex.I) : 
  z.im = -2 := by
  sorry

end imaginary_part_of_z_l326_32671


namespace limit_proof_l326_32682

open Real

theorem limit_proof (ε : ℝ) (hε : ε > 0) :
  ∃ δ : ℝ, δ > 0 ∧
  ∀ x : ℝ, 0 < |x + 7/2| ∧ |x + 7/2| < δ →
    |(2*x^2 + 13*x + 21) / (2*x + 7) + 1/2| < ε :=
by
  sorry

end limit_proof_l326_32682


namespace rectangle_longest_side_l326_32667

/-- A rectangle with perimeter 240 feet and area equal to eight times its perimeter has its longest side equal to 80 feet. -/
theorem rectangle_longest_side (l w : ℝ) : 
  (l > 0) → 
  (w > 0) → 
  (2 * l + 2 * w = 240) → 
  (l * w = 8 * (2 * l + 2 * w)) → 
  (max l w = 80) := by
sorry

end rectangle_longest_side_l326_32667


namespace coefficient_of_x_term_l326_32601

theorem coefficient_of_x_term (x : ℝ) : 
  let expansion := (Real.sqrt x - 1)^4 * (x - 1)^2
  ∃ (a b c d e f : ℝ), expansion = a*x^3 + b*x^(5/2) + c*x^2 + d*x^(3/2) + 4*x + f*x^(1/2) + e
  := by sorry

end coefficient_of_x_term_l326_32601


namespace total_animals_is_100_l326_32632

/-- The number of rabbits -/
def num_rabbits : ℕ := 4

/-- The number of ducks -/
def num_ducks : ℕ := num_rabbits + 12

/-- The number of chickens -/
def num_chickens : ℕ := 5 * num_ducks

/-- The total number of animals -/
def total_animals : ℕ := num_chickens + num_ducks + num_rabbits

theorem total_animals_is_100 : total_animals = 100 := by
  sorry

end total_animals_is_100_l326_32632


namespace circle_condition_l326_32626

theorem circle_condition (a : ℝ) : 
  (∃ (x y : ℝ), x^2 + y^2 + 2*y + 2*a - 1 = 0 ∧ ∀ (x' y' : ℝ), x'^2 + y'^2 + 2*y' + 2*a - 1 = 0 → (x - x')^2 + (y - y')^2 = (x' - x)^2 + (y' - y)^2) 
  → a < 1 := by
  sorry

end circle_condition_l326_32626


namespace division_theorem_l326_32694

theorem division_theorem (dividend divisor remainder quotient : ℕ) : 
  dividend = 141 →
  divisor = 17 →
  remainder = 5 →
  dividend = divisor * quotient + remainder →
  quotient = 8 := by
sorry

end division_theorem_l326_32694


namespace smallest_integer_with_remainders_l326_32628

theorem smallest_integer_with_remainders : ∃ n : ℕ, 
  n > 0 ∧
  n % 2 = 1 ∧
  n % 3 = 2 ∧
  n % 4 = 3 ∧
  n % 10 = 9 ∧
  (∀ m : ℕ, m > 0 ∧ m % 2 = 1 ∧ m % 3 = 2 ∧ m % 4 = 3 ∧ m % 10 = 9 → n ≤ m) ∧
  n = 59 := by
sorry

end smallest_integer_with_remainders_l326_32628


namespace bananas_in_E_l326_32681

/-- The number of baskets -/
def num_baskets : ℕ := 5

/-- The average number of fruits per basket -/
def avg_fruits_per_basket : ℕ := 25

/-- The number of fruits in basket A -/
def fruits_in_A : ℕ := 15

/-- The number of fruits in basket B -/
def fruits_in_B : ℕ := 30

/-- The number of fruits in basket C -/
def fruits_in_C : ℕ := 20

/-- The number of fruits in basket D -/
def fruits_in_D : ℕ := 25

/-- Theorem: The number of bananas in basket E is 35 -/
theorem bananas_in_E : 
  num_baskets * avg_fruits_per_basket - (fruits_in_A + fruits_in_B + fruits_in_C + fruits_in_D) = 35 := by
  sorry

end bananas_in_E_l326_32681


namespace lawrence_marbles_l326_32677

theorem lawrence_marbles (num_friends : ℕ) (marbles_per_friend : ℕ) 
  (h1 : num_friends = 64) (h2 : marbles_per_friend = 86) : 
  num_friends * marbles_per_friend = 5504 := by
  sorry

end lawrence_marbles_l326_32677


namespace range_of_m_l326_32683

-- Define propositions p and q
def p (m : ℝ) : Prop := ∀ x > 0, m^2 + 2*m - 1 ≤ x + 1/x

def q (m : ℝ) : Prop := ∀ x, (5 - m^2)^x < (5 - m^2)^(x + 1)

-- Define the theorem
theorem range_of_m (m : ℝ) :
  (p m ∨ q m) ∧ ¬(p m ∧ q m) →
  (-3 ≤ m ∧ m ≤ -2) ∨ (1 < m ∧ m < 2) :=
by sorry

end range_of_m_l326_32683


namespace divisibility_probability_l326_32608

/-- The number of positive divisors of 10^99 -/
def total_divisors : ℕ := 10000

/-- The number of positive divisors of 10^99 that are multiples of 10^88 -/
def favorable_divisors : ℕ := 144

/-- The probability of a randomly chosen positive divisor of 10^99 being an integer multiple of 10^88 -/
def probability : ℚ := favorable_divisors / total_divisors

theorem divisibility_probability :
  probability = 9 / 625 :=
sorry

end divisibility_probability_l326_32608


namespace complex_square_one_plus_i_l326_32672

theorem complex_square_one_plus_i : (1 + Complex.I) ^ 2 = 2 * Complex.I := by
  sorry

end complex_square_one_plus_i_l326_32672


namespace inequality_equivalence_l326_32685

theorem inequality_equivalence (x y : ℝ) (h : x ≠ 0) :
  (y - x)^2 < x^2 ↔ y > 0 ∧ y < 2*x := by sorry

end inequality_equivalence_l326_32685


namespace new_person_weight_l326_32696

theorem new_person_weight (initial_count : ℕ) (weight_increase : ℝ) (replaced_weight : ℝ) :
  initial_count = 8 →
  weight_increase = 2.5 →
  replaced_weight = 45 →
  ∃ (new_weight : ℝ),
    new_weight = initial_count * weight_increase + replaced_weight ∧
    new_weight = 65 :=
by sorry

end new_person_weight_l326_32696


namespace area_ratio_theorem_l326_32648

def total_area : ℝ := 700
def smaller_area : ℝ := 315

theorem area_ratio_theorem :
  let larger_area := total_area - smaller_area
  let difference := larger_area - smaller_area
  let average := (larger_area + smaller_area) / 2
  difference / average = 1 / 5 := by
  sorry

end area_ratio_theorem_l326_32648


namespace least_number_for_divisibility_l326_32621

theorem least_number_for_divisibility : 
  ∃! x : ℕ, x < 577 ∧ (907223 + x) % 577 = 0 ∧ 
  ∀ y : ℕ, y < x → (907223 + y) % 577 ≠ 0 :=
by sorry

end least_number_for_divisibility_l326_32621


namespace mandy_shirts_total_l326_32643

theorem mandy_shirts_total (black_packs yellow_packs : ℕ) 
  (black_per_pack yellow_per_pack : ℕ) : 
  black_packs = 3 → 
  yellow_packs = 3 → 
  black_per_pack = 5 → 
  yellow_per_pack = 2 → 
  black_packs * black_per_pack + yellow_packs * yellow_per_pack = 21 := by
  sorry

#check mandy_shirts_total

end mandy_shirts_total_l326_32643


namespace math_city_intersections_l326_32675

/-- Represents a city with a given number of streets -/
structure City where
  numStreets : ℕ
  noParallel : Bool
  noTripleIntersections : Bool

/-- Calculates the number of intersections in a city -/
def numIntersections (c : City) : ℕ :=
  (c.numStreets.pred * c.numStreets.pred) / 2

theorem math_city_intersections (c : City) 
  (h1 : c.numStreets = 10)
  (h2 : c.noParallel = true)
  (h3 : c.noTripleIntersections = true) :
  numIntersections c = 45 := by
  sorry

end math_city_intersections_l326_32675


namespace senior_ticket_price_l326_32614

theorem senior_ticket_price 
  (total_tickets : ℕ) 
  (adult_price : ℕ) 
  (total_receipts : ℕ) 
  (senior_tickets : ℕ) 
  (h1 : total_tickets = 510) 
  (h2 : adult_price = 21) 
  (h3 : total_receipts = 8748) 
  (h4 : senior_tickets = 327) :
  ∃ (senior_price : ℕ), 
    senior_price * senior_tickets + adult_price * (total_tickets - senior_tickets) = total_receipts ∧ 
    senior_price = 15 := by
  sorry

end senior_ticket_price_l326_32614


namespace hexagon_exterior_angle_sum_hexagon_exterior_angle_sum_proof_l326_32680

/-- The sum of the exterior angles of a hexagon is 360 degrees. -/
theorem hexagon_exterior_angle_sum : ℝ :=
  360

#check hexagon_exterior_angle_sum

/-- Proof of the theorem -/
theorem hexagon_exterior_angle_sum_proof :
  hexagon_exterior_angle_sum = 360 := by
  sorry

end hexagon_exterior_angle_sum_hexagon_exterior_angle_sum_proof_l326_32680


namespace arithmetic_sequence_sum_l326_32693

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a → a 3 + a 4 + a 5 = 12 → a 1 + a 7 = 8 := by
  sorry

end arithmetic_sequence_sum_l326_32693
