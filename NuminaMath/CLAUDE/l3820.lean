import Mathlib

namespace cos_equality_solution_l3820_382047

theorem cos_equality_solution (m : ℤ) (h1 : 0 ≤ m) (h2 : m ≤ 360) 
  (h3 : Real.cos (m * π / 180) = Real.cos (970 * π / 180)) : 
  m = 110 ∨ m = 250 := by
  sorry

end cos_equality_solution_l3820_382047


namespace max_n_value_l3820_382081

theorem max_n_value (a b c : ℝ) (n : ℕ) (h1 : a > b) (h2 : b > c)
  (h3 : ∀ (a b c : ℝ), a > b → b > c → (a - b)⁻¹ + (b - c)⁻¹ ≥ n^2 * (a - c)⁻¹) :
  n ≤ 2 :=
sorry

end max_n_value_l3820_382081


namespace consecutive_points_segment_length_l3820_382009

/-- Given 5 consecutive points on a straight line, if certain segment lengths are known,
    prove that the length of ac is 11. -/
theorem consecutive_points_segment_length 
  (a b c d e : ℝ) -- Representing points as real numbers on a line
  (h_consecutive : a < b ∧ b < c ∧ c < d ∧ d < e) -- Consecutive points
  (h_bc_cd : c - b = 2 * (d - c)) -- bc = 2cd
  (h_de : e - d = 8) -- de = 8
  (h_ab : b - a = 5) -- ab = 5
  (h_ae : e - a = 22) -- ae = 22
  : c - a = 11 := by
  sorry

end consecutive_points_segment_length_l3820_382009


namespace vector_operation_l3820_382093

/-- Given vectors a and b, prove that -3a - 2b equals the expected result. -/
theorem vector_operation (a b : ℝ × ℝ) (h1 : a = (3, -1)) (h2 : b = (-1, 2)) :
  (-3 : ℝ) • a - (2 : ℝ) • b = (-7, -1) := by
  sorry

end vector_operation_l3820_382093


namespace cubic_equation_root_sum_squares_l3820_382003

theorem cubic_equation_root_sum_squares (a b c : ℝ) : 
  a^3 - 6*a^2 - 7*a + 2 = 0 →
  b^3 - 6*b^2 - 7*b + 2 = 0 →
  c^3 - 6*c^2 - 7*c + 2 = 0 →
  a ≠ b → b ≠ c → a ≠ c →
  1/a^2 + 1/b^2 + 1/c^2 = 73/4 := by
sorry

end cubic_equation_root_sum_squares_l3820_382003


namespace drill_bits_purchase_cost_l3820_382006

/-- The total cost of a purchase with tax -/
def total_cost (num_sets : ℕ) (cost_per_set : ℝ) (tax_rate : ℝ) : ℝ :=
  let pre_tax_cost := num_sets * cost_per_set
  let tax_amount := pre_tax_cost * tax_rate
  pre_tax_cost + tax_amount

/-- Theorem stating the total cost for the specific purchase -/
theorem drill_bits_purchase_cost :
  total_cost 5 6 0.1 = 33 := by
  sorry

end drill_bits_purchase_cost_l3820_382006


namespace kendras_change_l3820_382040

/-- Calculates the change received after a purchase -/
def calculate_change (toy_price hat_price : ℕ) (num_toys num_hats : ℕ) (total_money : ℕ) : ℕ :=
  total_money - (toy_price * num_toys + hat_price * num_hats)

/-- Proves that Kendra's change is $30 -/
theorem kendras_change :
  let toy_price : ℕ := 20
  let hat_price : ℕ := 10
  let num_toys : ℕ := 2
  let num_hats : ℕ := 3
  let total_money : ℕ := 100
  calculate_change toy_price hat_price num_toys num_hats total_money = 30 := by
  sorry

#eval calculate_change 20 10 2 3 100

end kendras_change_l3820_382040


namespace mean_temperature_l3820_382028

def temperatures : List ℚ := [75, 80, 78, 82, 85, 90, 87, 84, 88, 93]

theorem mean_temperature : (temperatures.sum / temperatures.length : ℚ) = 421/5 := by
  sorry

end mean_temperature_l3820_382028


namespace shifted_quadratic_coefficient_sum_l3820_382050

/-- Given a quadratic function f(x) = 3x^2 - x + 7, shifting it 5 units to the right
    results in a new quadratic function g(x) = ax^2 + bx + c.
    This theorem proves that the sum of the coefficients a + b + c equals 59. -/
theorem shifted_quadratic_coefficient_sum :
  let f : ℝ → ℝ := λ x ↦ 3 * x^2 - x + 7
  let g : ℝ → ℝ := λ x ↦ f (x - 5)
  ∃ a b c : ℝ, (∀ x, g x = a * x^2 + b * x + c) ∧ (a + b + c = 59) :=
by sorry

end shifted_quadratic_coefficient_sum_l3820_382050


namespace inscribed_squares_ratio_l3820_382011

theorem inscribed_squares_ratio (a b c x y : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  a^2 + b^2 = c^2 →
  x * (a + b - x) = a * b →
  y * (a + b - y) = (c - y)^2 →
  x / y = 37 / 35 := by
  sorry

end inscribed_squares_ratio_l3820_382011


namespace trevor_ride_cost_l3820_382064

/-- The total cost of Trevor's taxi ride downtown including the tip -/
def total_cost (uber_cost lyft_cost taxi_cost : ℚ) : ℚ :=
  taxi_cost + 0.2 * taxi_cost

theorem trevor_ride_cost :
  ∀ (uber_cost lyft_cost taxi_cost : ℚ),
    uber_cost = lyft_cost + 3 →
    lyft_cost = taxi_cost + 4 →
    uber_cost = 22 →
    total_cost uber_cost lyft_cost taxi_cost = 18 := by
  sorry

end trevor_ride_cost_l3820_382064


namespace yonder_license_plates_l3820_382088

/-- The number of possible letters in a license plate. -/
def num_letters : ℕ := 26

/-- The number of possible symbols in a license plate. -/
def num_symbols : ℕ := 5

/-- The number of possible digits in a license plate. -/
def num_digits : ℕ := 10

/-- The number of letter positions in a license plate. -/
def letter_positions : ℕ := 2

/-- The number of symbol positions in a license plate. -/
def symbol_positions : ℕ := 1

/-- The number of digit positions in a license plate. -/
def digit_positions : ℕ := 4

/-- The total number of valid license plates in Yonder. -/
def total_license_plates : ℕ := 33800000

/-- Theorem stating the total number of valid license plates in Yonder. -/
theorem yonder_license_plates :
  (num_letters ^ letter_positions) * (num_symbols ^ symbol_positions) * (num_digits ^ digit_positions) = total_license_plates :=
by sorry

end yonder_license_plates_l3820_382088


namespace probability_of_selecting_girl_l3820_382053

-- Define the total number of candidates
def total_candidates : ℕ := 3 + 1

-- Define the number of girls
def number_of_girls : ℕ := 1

-- Theorem statement
theorem probability_of_selecting_girl :
  (number_of_girls : ℚ) / total_candidates = 1 / 4 := by
  sorry

end probability_of_selecting_girl_l3820_382053


namespace carnival_tickets_l3820_382008

/-- The number of additional tickets needed for even distribution -/
def additional_tickets (friends : ℕ) (total_tickets : ℕ) : ℕ :=
  (friends - (total_tickets % friends)) % friends

/-- Proof that 9 friends need 8 more tickets to evenly split 865 tickets -/
theorem carnival_tickets : additional_tickets 9 865 = 8 := by
  sorry

end carnival_tickets_l3820_382008


namespace quadratic_no_real_roots_l3820_382067

theorem quadratic_no_real_roots : 
  {x : ℝ | x^2 + x + 1 = 0} = ∅ := by sorry

end quadratic_no_real_roots_l3820_382067


namespace mark_collection_amount_l3820_382084

/-- Calculates the total amount collected by Mark for the homeless -/
def totalAmountCollected (householdsPerDay : ℕ) (days : ℕ) (donationAmount : ℕ) : ℕ :=
  let totalHouseholds := householdsPerDay * days
  let donatingHouseholds := totalHouseholds / 2
  donatingHouseholds * donationAmount

/-- Proves that Mark collected $2000 given the problem conditions -/
theorem mark_collection_amount :
  totalAmountCollected 20 5 40 = 2000 := by
  sorry

#eval totalAmountCollected 20 5 40

end mark_collection_amount_l3820_382084


namespace business_value_calculation_l3820_382038

theorem business_value_calculation (man_share : ℚ) (sold_portion : ℚ) (sale_price : ℕ) :
  man_share = 1/3 →
  sold_portion = 3/5 →
  sale_price = 2000 →
  ∃ (total_value : ℕ), total_value = 10000 ∧ 
    (sold_portion * man_share * total_value : ℚ) = sale_price := by
  sorry

end business_value_calculation_l3820_382038


namespace first_number_in_expression_l3820_382005

theorem first_number_in_expression : ∃ x : ℝ, (x * 12 * 20) / 3 + 125 = 2229 ∧ x = 26.3 := by
  sorry

end first_number_in_expression_l3820_382005


namespace equation_solution_l3820_382058

theorem equation_solution :
  ∀ x y : ℝ, x^2 - y^4 = Real.sqrt (18*x - x^2 - 81) ↔ (x = 9 ∧ (y = Real.sqrt 3 ∨ y = -Real.sqrt 3)) :=
by sorry

end equation_solution_l3820_382058


namespace valid_purchase_has_two_notebooks_l3820_382031

/-- Represents the purchase of notebooks and books -/
structure Purchase where
  notebooks : ℕ
  books : ℕ
  notebook_cost : ℕ
  book_cost : ℕ

/-- Checks if the purchase satisfies the given conditions -/
def is_valid_purchase (p : Purchase) : Prop :=
  p.books = p.notebooks + 4 ∧
  p.notebooks * p.notebook_cost = 72 ∧
  p.books * p.book_cost = 660 ∧
  p.notebooks * p.book_cost + p.books * p.notebook_cost < 444

/-- The theorem stating that the valid purchase has 2 notebooks -/
theorem valid_purchase_has_two_notebooks :
  ∃ (p : Purchase), is_valid_purchase p ∧ p.notebooks = 2 :=
sorry


end valid_purchase_has_two_notebooks_l3820_382031


namespace inequality_solution_minimum_value_minimum_exists_l3820_382066

-- Define the function f
def f (x : ℝ) : ℝ := |2 * x - 1|

-- Theorem 1: Inequality solution
theorem inequality_solution :
  ∀ x : ℝ, f (2 * x) ≤ f (x + 1) ↔ 0 ≤ x ∧ x ≤ 1 :=
sorry

-- Theorem 2: Minimum value
theorem minimum_value :
  ∀ a b : ℝ, a + b = 2 → f (a^2) + f (b^2) ≥ 2 :=
sorry

-- Theorem 3: Existence of minimum
theorem minimum_exists :
  ∃ a b : ℝ, a + b = 2 ∧ f (a^2) + f (b^2) = 2 :=
sorry

end inequality_solution_minimum_value_minimum_exists_l3820_382066


namespace base_side_from_sphere_volume_l3820_382037

/-- Regular triangular prism with inscribed sphere -/
structure RegularTriangularPrism :=
  (base_side : ℝ)
  (height : ℝ)
  (sphere_volume : ℝ)

/-- The theorem stating the relationship between the inscribed sphere volume
    and the base side length of a regular triangular prism -/
theorem base_side_from_sphere_volume
  (prism : RegularTriangularPrism)
  (h_positive : prism.base_side > 0)
  (h_sphere_volume : prism.sphere_volume = 36 * Real.pi)
  (h_height_eq_diameter : prism.height = 2 * (prism.base_side * Real.sqrt 3 / 6)) :
  prism.base_side = 3 * Real.sqrt 3 :=
sorry

end base_side_from_sphere_volume_l3820_382037


namespace shaded_area_calculation_l3820_382012

def small_radius : ℝ := 3
def large_radius : ℝ := 5

def left_rectangle_width : ℝ := small_radius
def left_rectangle_height : ℝ := 2 * small_radius
def right_rectangle_width : ℝ := large_radius
def right_rectangle_height : ℝ := 2 * large_radius

def isosceles_triangle_leg : ℝ := small_radius

theorem shaded_area_calculation :
  let left_rectangle_area := left_rectangle_width * left_rectangle_height
  let right_rectangle_area := right_rectangle_width * right_rectangle_height
  let left_semicircle_area := (1/2) * Real.pi * small_radius^2
  let right_semicircle_area := (1/2) * Real.pi * large_radius^2
  let triangle_area := (1/2) * isosceles_triangle_leg^2
  let total_shaded_area := (left_rectangle_area - left_semicircle_area - triangle_area) + 
                           (right_rectangle_area - right_semicircle_area)
  total_shaded_area = 63.5 - 17 * Real.pi := by sorry

end shaded_area_calculation_l3820_382012


namespace even_decreasing_compare_l3820_382001

-- Define an even function
def EvenFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

-- Define a decreasing function on negative reals
def DecreasingOnNegative (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y ∧ y < 0 → f x > f y

-- Theorem statement
theorem even_decreasing_compare (f : ℝ → ℝ) (x₁ x₂ : ℝ) 
  (h_even : EvenFunction f) 
  (h_decreasing : DecreasingOnNegative f) 
  (h_abs : |x₁| < |x₂|) : 
  f x₁ - f x₂ < 0 := by
  sorry

end even_decreasing_compare_l3820_382001


namespace complementary_probability_l3820_382080

theorem complementary_probability (P_snow : ℚ) (h : P_snow = 2/5) :
  1 - P_snow = 3/5 := by
  sorry

end complementary_probability_l3820_382080


namespace tangent_line_at_P_l3820_382083

/-- The curve function -/
def f (x : ℝ) : ℝ := x^2 + 2

/-- The point of tangency -/
def P : ℝ × ℝ := (1, 3)

/-- The tangent line function -/
def tangent_line (x : ℝ) : ℝ := 2*x - 1

theorem tangent_line_at_P : 
  (∀ x : ℝ, tangent_line x = 2*x - 1) ∧ 
  (tangent_line P.1 = P.2) ∧
  (HasDerivAt f 2 P.1) :=
by sorry

end tangent_line_at_P_l3820_382083


namespace x_y_negative_l3820_382068

theorem x_y_negative (x y : ℝ) (h1 : x - y > x) (h2 : 3 * x + 2 * y < 2 * y) : x < 0 ∧ y < 0 := by
  sorry

end x_y_negative_l3820_382068


namespace jean_stuffies_fraction_l3820_382099

theorem jean_stuffies_fraction (total : ℕ) (kept_fraction : ℚ) (janet_received : ℕ) :
  total = 60 →
  kept_fraction = 1/3 →
  janet_received = 10 →
  (janet_received : ℚ) / (total - total * kept_fraction) = 1/4 := by
  sorry

end jean_stuffies_fraction_l3820_382099


namespace log_equality_implies_value_l3820_382004

theorem log_equality_implies_value (x : ℝ) (h : Real.log x = Real.log 4 + Real.log 3) : x = 12 := by
  sorry

end log_equality_implies_value_l3820_382004


namespace find_set_B_l3820_382027

theorem find_set_B (a b : ℝ) : 
  let P : Set ℝ := {1, a/b, b}
  let B : Set ℝ := {0, a+b, b^2}
  P = B → B = {0, -1, 1} := by
sorry

end find_set_B_l3820_382027


namespace conditional_probability_fair_die_l3820_382049

-- Define the sample space
def S : Finset Nat := {1, 2, 3, 4, 5, 6}

-- Define events A and B
def A : Finset Nat := {2, 3, 5}
def B : Finset Nat := {1, 2, 4, 5, 6}

-- Define the probability measure
def P (X : Finset Nat) : ℚ := (X.card : ℚ) / (S.card : ℚ)

-- Define the intersection of events
def AB : Finset Nat := A ∩ B

-- Theorem statement
theorem conditional_probability_fair_die :
  P AB / P B = 2 / 5 := by
  sorry

end conditional_probability_fair_die_l3820_382049


namespace inverse_97_mod_98_l3820_382034

theorem inverse_97_mod_98 : ∃ x : ℕ, x ≥ 0 ∧ x ≤ 97 ∧ (97 * x) % 98 = 1 := by
  sorry

end inverse_97_mod_98_l3820_382034


namespace faith_change_is_ten_l3820_382070

/-- The change Faith receives from her purchase at the baking shop. -/
def faith_change : ℕ :=
  let flour_cost : ℕ := 5
  let cake_stand_cost : ℕ := 28
  let total_cost : ℕ := flour_cost + cake_stand_cost
  let bill_payment : ℕ := 2 * 20
  let coin_payment : ℕ := 3
  let total_payment : ℕ := bill_payment + coin_payment
  total_payment - total_cost

/-- Theorem stating that Faith receives $10 in change. -/
theorem faith_change_is_ten : faith_change = 10 := by
  sorry

end faith_change_is_ten_l3820_382070


namespace regression_difference_is_residual_sum_of_squares_l3820_382036

/-- In regression analysis, the term representing the difference between a data point
    and its corresponding position on the regression line -/
def regression_difference_term : String := "residual sum of squares"

/-- The residual sum of squares represents the difference between data points
    and their corresponding positions on the regression line -/
axiom residual_sum_of_squares_def :
  regression_difference_term = "residual sum of squares"

theorem regression_difference_is_residual_sum_of_squares :
  regression_difference_term = "residual sum of squares" := by
  sorry

end regression_difference_is_residual_sum_of_squares_l3820_382036


namespace equation_has_six_roots_l3820_382046

/-- The number of roots of the equation √(14-x²)(sin x - cos 2x) = 0 in the interval [-√14, √14] -/
def num_roots : ℕ := 6

/-- The equation √(14-x²)(sin x - cos 2x) = 0 -/
def equation (x : ℝ) : Prop :=
  Real.sqrt (14 - x^2) * (Real.sin x - Real.cos (2 * x)) = 0

/-- The domain of the equation -/
def domain (x : ℝ) : Prop :=
  x ≥ -Real.sqrt 14 ∧ x ≤ Real.sqrt 14

/-- Theorem stating that the equation has exactly 6 roots in the given domain -/
theorem equation_has_six_roots :
  ∃! (s : Finset ℝ), s.card = num_roots ∧ 
  (∀ x ∈ s, domain x ∧ equation x) ∧
  (∀ x, domain x → equation x → x ∈ s) :=
sorry

end equation_has_six_roots_l3820_382046


namespace integer_fraction_property_l3820_382025

theorem integer_fraction_property (x y : ℤ) (h : ∃ k : ℤ, 3 * x + 4 * y = 5 * k) :
  ∃ m : ℤ, 4 * x - 3 * y = 5 * m := by
sorry

end integer_fraction_property_l3820_382025


namespace insulation_cost_example_l3820_382042

/-- Calculates the total cost of insulating a rectangular tank with two layers -/
def insulation_cost (length width height : ℝ) (cost1 cost2 : ℝ) : ℝ :=
  let surface_area := 2 * (length * width + length * height + width * height)
  surface_area * (cost1 + cost2)

/-- Theorem: The cost of insulating a 4x5x2 tank with $20 and $15 per sq ft layers is $2660 -/
theorem insulation_cost_example : insulation_cost 4 5 2 20 15 = 2660 := by
  sorry

end insulation_cost_example_l3820_382042


namespace vincent_laundry_week_l3820_382002

def loads_wednesday : ℕ := 6

def loads_thursday (w : ℕ) : ℕ := 2 * w

def loads_friday (t : ℕ) : ℕ := t / 2

def loads_saturday (w : ℕ) : ℕ := w / 3

def total_loads (w t f s : ℕ) : ℕ := w + t + f + s

theorem vincent_laundry_week :
  total_loads loads_wednesday 
              (loads_thursday loads_wednesday)
              (loads_friday (loads_thursday loads_wednesday))
              (loads_saturday loads_wednesday) = 26 := by
  sorry

end vincent_laundry_week_l3820_382002


namespace power_of_square_l3820_382057

theorem power_of_square (b : ℝ) : (b^2)^3 = b^6 := by
  sorry

end power_of_square_l3820_382057


namespace sum_with_radical_conjugate_sum_fifteen_minus_sqrt500_and_conjugate_l3820_382096

/-- The sum of a number and its radical conjugate is twice the real part of the number. -/
theorem sum_with_radical_conjugate (a : ℝ) (b : ℝ) (h : 0 ≤ b) :
  (a - Real.sqrt b) + (a + Real.sqrt b) = 2 * a := by
  sorry

/-- The sum of 15 - √500 and its radical conjugate is 30. -/
theorem sum_fifteen_minus_sqrt500_and_conjugate :
  (15 - Real.sqrt 500) + (15 + Real.sqrt 500) = 30 := by
  sorry

end sum_with_radical_conjugate_sum_fifteen_minus_sqrt500_and_conjugate_l3820_382096


namespace sqrt_difference_equals_two_l3820_382026

theorem sqrt_difference_equals_two :
  Real.sqrt (3 + 2 * Real.sqrt 2) - Real.sqrt (3 - 2 * Real.sqrt 2) = 2 := by
  sorry

end sqrt_difference_equals_two_l3820_382026


namespace unique_divisible_power_of_two_l3820_382030

theorem unique_divisible_power_of_two (n : ℕ) : 
  (100 ≤ n ∧ n ≤ 1997) ∧ (∃ k : ℕ, 2^n + 2 = n * k) ↔ n = 946 := by
  sorry

end unique_divisible_power_of_two_l3820_382030


namespace greatest_valid_partition_l3820_382069

/-- A partition of positive integers into k subsets satisfying the sum property -/
def ValidPartition (k : ℕ) : Prop :=
  ∃ (A : Fin k → Set ℕ), 
    (∀ i j, i ≠ j → A i ∩ A j = ∅) ∧ 
    (⋃ i, A i) = {n : ℕ | n > 0} ∧
    ∀ (n : ℕ) (i : Fin k), n ≥ 15 → 
      ∃ (x y : ℕ), x ∈ A i ∧ y ∈ A i ∧ x ≠ y ∧ x + y = n

/-- The main theorem: 3 is the greatest positive integer satisfying the property -/
theorem greatest_valid_partition : 
  ValidPartition 3 ∧ ∀ k > 3, ¬ValidPartition k :=
sorry

end greatest_valid_partition_l3820_382069


namespace texas_tech_profit_calculation_l3820_382044

/-- The amount of money made per t-shirt sold -/
def profit_per_shirt : ℕ := 78

/-- The total number of t-shirts sold during both games -/
def total_shirts : ℕ := 186

/-- The number of t-shirts sold during the Arkansas game -/
def arkansas_shirts : ℕ := 172

/-- The money made from selling t-shirts during the Texas Tech game -/
def texas_tech_profit : ℕ := (total_shirts - arkansas_shirts) * profit_per_shirt

theorem texas_tech_profit_calculation : texas_tech_profit = 1092 := by
  sorry

end texas_tech_profit_calculation_l3820_382044


namespace eel_length_problem_l3820_382052

theorem eel_length_problem (jenna_length bill_length : ℝ) : 
  jenna_length = 16 ∧ jenna_length = (1/3) * bill_length → 
  jenna_length + bill_length = 64 := by
  sorry

end eel_length_problem_l3820_382052


namespace anyas_initial_seat_l3820_382072

/-- Represents the seat numbers in the theater --/
inductive Seat
| one
| two
| three
| four
| five

/-- Represents the friends --/
inductive Friend
| Anya
| Varya
| Galya
| Diana
| Ella

/-- Represents the seating arrangement before and after Anya left --/
structure SeatingArrangement where
  seats : Friend → Seat

/-- Moves a seat to the right --/
def moveRight (s : Seat) : Seat :=
  match s with
  | Seat.one => Seat.two
  | Seat.two => Seat.three
  | Seat.three => Seat.four
  | Seat.four => Seat.five
  | Seat.five => Seat.five

/-- Moves a seat to the left --/
def moveLeft (s : Seat) : Seat :=
  match s with
  | Seat.one => Seat.one
  | Seat.two => Seat.one
  | Seat.three => Seat.two
  | Seat.four => Seat.three
  | Seat.five => Seat.four

/-- Theorem stating Anya's initial seat was four --/
theorem anyas_initial_seat (initial final : SeatingArrangement) :
  (final.seats Friend.Varya = moveRight (initial.seats Friend.Varya)) →
  (final.seats Friend.Galya = moveLeft (moveLeft (initial.seats Friend.Galya))) →
  (final.seats Friend.Diana = initial.seats Friend.Ella) →
  (final.seats Friend.Ella = initial.seats Friend.Diana) →
  (final.seats Friend.Anya = Seat.five) →
  (initial.seats Friend.Anya = Seat.four) :=
by
  sorry


end anyas_initial_seat_l3820_382072


namespace houses_with_pool_l3820_382055

theorem houses_with_pool (total : ℕ) (garage : ℕ) (neither : ℕ) (both : ℕ) 
  (h_total : total = 85)
  (h_garage : garage = 50)
  (h_neither : neither = 30)
  (h_both : both = 35) :
  ∃ pool : ℕ, pool = 40 ∧ total = (garage - both) + (pool - both) + both + neither :=
by sorry

end houses_with_pool_l3820_382055


namespace ellipse_eccentricity_m_range_l3820_382021

theorem ellipse_eccentricity_m_range :
  ∀ m : ℝ,
  m > 0 →
  (∃ e : ℝ, 1/2 < e ∧ e < 1 ∧
    (∀ x y : ℝ, x^2 + m*y^2 = 1 →
      e = Real.sqrt (1 - min m (1/m)))) →
  (m ∈ Set.Ioo 0 (3/4) ∪ Set.Ioi (4/3)) :=
by sorry

end ellipse_eccentricity_m_range_l3820_382021


namespace participant_count_2019_l3820_382074

/-- The number of participants in the Science Quiz Bowl for different years --/
structure ParticipantCount where
  y2018 : ℕ
  y2019 : ℕ
  y2020 : ℕ

/-- The conditions given in the problem --/
def satisfiesConditions (p : ParticipantCount) : Prop :=
  p.y2018 = 150 ∧
  p.y2020 = p.y2019 / 2 - 40 ∧
  p.y2019 = p.y2020 + 200

/-- The theorem to be proved --/
theorem participant_count_2019 (p : ParticipantCount) 
  (h : satisfiesConditions p) : 
  p.y2019 = 320 ∧ p.y2019 - p.y2018 = 170 :=
by
  sorry

end participant_count_2019_l3820_382074


namespace tangent_line_at_P_l3820_382014

-- Define the curve
def f (x : ℝ) : ℝ := x^3 - 1

-- Define the point of tangency
def P : ℝ × ℝ := (1, 0)

-- Define the slope of the tangent line at P
def m : ℝ := 3

-- Define the equation of the tangent line
def tangent_line (x : ℝ) : ℝ := m * (x - P.1) + P.2

theorem tangent_line_at_P : 
  ∀ x : ℝ, tangent_line x = 3*x - 3 :=
sorry

end tangent_line_at_P_l3820_382014


namespace number_value_l3820_382051

theorem number_value (tens : ℕ) (ones : ℕ) (tenths : ℕ) (hundredths : ℕ) :
  tens = 21 →
  ones = 8 →
  tenths = 5 →
  hundredths = 34 →
  (tens * 10 : ℚ) + ones + (tenths : ℚ) / 10 + (hundredths : ℚ) / 100 = 218.84 :=
by sorry

end number_value_l3820_382051


namespace coin_array_final_row_sum_of_digits_l3820_382016

/-- The sum of the first n natural numbers -/
def triangular_sum (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

theorem coin_array_final_row_sum_of_digits :
  ∃ (n : ℕ), triangular_sum n = 5050 ∧ sum_of_digits n = 1 :=
sorry

end coin_array_final_row_sum_of_digits_l3820_382016


namespace sum_of_roots_quadratic_l3820_382033

theorem sum_of_roots_quadratic (x₁ x₂ : ℝ) : 
  (x₁^2 + 2*x₁ - 1 = 0) → (x₂^2 + 2*x₂ - 1 = 0) → x₁ + x₂ = -2 := by
  sorry

end sum_of_roots_quadratic_l3820_382033


namespace oyster_ratio_proof_l3820_382097

/-- Proves that the ratio of oysters on the second day to the first day is 1:2 -/
theorem oyster_ratio_proof (oysters_day1 crabs_day1 total_count : ℕ) 
  (h1 : oysters_day1 = 50)
  (h2 : crabs_day1 = 72)
  (h3 : total_count = 195)
  (h4 : ∃ (oysters_day2 crabs_day2 : ℕ), 
    crabs_day2 = 2 * crabs_day1 / 3 ∧ 
    oysters_day1 + crabs_day1 + oysters_day2 + crabs_day2 = total_count) :
  ∃ (oysters_day2 : ℕ), oysters_day2 * 2 = oysters_day1 :=
sorry

end oyster_ratio_proof_l3820_382097


namespace becky_lollipops_l3820_382007

theorem becky_lollipops (total_lollipops : ℕ) (num_friends : ℕ) (lemon : ℕ) (peppermint : ℕ) (watermelon : ℕ) (marshmallow : ℕ) :
  total_lollipops = lemon + peppermint + watermelon + marshmallow →
  total_lollipops = 795 →
  num_friends = 13 →
  total_lollipops % num_friends = 2 :=
by sorry

end becky_lollipops_l3820_382007


namespace system_solution_l3820_382010

theorem system_solution :
  ∀ x y z : ℝ,
  x + y + z = 12 ∧
  x^2 + y^2 + z^2 = 230 ∧
  x * y = -15 →
  ((x = 15 ∧ y = -1 ∧ z = -2) ∨
   (x = -1 ∧ y = 15 ∧ z = -2) ∨
   (x = 3 ∧ y = -5 ∧ z = 14) ∨
   (x = -5 ∧ y = 3 ∧ z = 14)) :=
by sorry

end system_solution_l3820_382010


namespace prob_at_most_one_for_given_probabilities_l3820_382077

/-- The probability that at most one of two independent events occurs, given their individual probabilities -/
def prob_at_most_one (p_a p_b : ℝ) : ℝ :=
  1 - p_a * p_b

theorem prob_at_most_one_for_given_probabilities :
  let p_a := 0.6
  let p_b := 0.7
  prob_at_most_one p_a p_b = 0.58 := by
  sorry

end prob_at_most_one_for_given_probabilities_l3820_382077


namespace sphere_volume_from_tetrahedron_surface_l3820_382054

theorem sphere_volume_from_tetrahedron_surface (s : ℝ) (V : ℝ) : 
  s = 3 →
  (4 * π * (V / ((4/3) * π))^((1:ℝ)/3)^2) = (4 * s^2 * Real.sqrt 3) →
  V = (27 * Real.sqrt 2) / Real.sqrt π :=
by
  sorry

end sphere_volume_from_tetrahedron_surface_l3820_382054


namespace fractional_equation_root_l3820_382075

theorem fractional_equation_root (n : ℤ) : 
  (∃ x : ℝ, x > 0 ∧ (x - 2) / (x - 3) = (n + 1) / (3 - x)) → n = -2 := by
  sorry

end fractional_equation_root_l3820_382075


namespace trapezoid_theorem_l3820_382013

/-- Represents a trapezoid with the given properties -/
structure Trapezoid where
  shorter_base : ℝ
  longer_base : ℝ
  height : ℝ
  midline_ratio : ℝ
  equal_area_segment : ℝ
  longer_base_condition : longer_base = shorter_base + 150
  midline_ratio_condition : midline_ratio = 3 / 4
  equal_area_condition : equal_area_segment > shorter_base ∧ equal_area_segment < longer_base

/-- The main theorem about the trapezoid -/
theorem trapezoid_theorem (t : Trapezoid) : 
  ⌊(t.equal_area_segment^2) / 150⌋ = 416 := by
  sorry

end trapezoid_theorem_l3820_382013


namespace cube_surface_area_from_prism_volume_l3820_382063

/-- Given a rectangular prism with dimensions 16, 4, and 24 inches,
    prove that a cube with the same volume has a surface area of
    approximately 798 square inches. -/
theorem cube_surface_area_from_prism_volume :
  let prism_length : ℝ := 16
  let prism_width : ℝ := 4
  let prism_height : ℝ := 24
  let prism_volume : ℝ := prism_length * prism_width * prism_height
  let cube_edge : ℝ := prism_volume ^ (1/3)
  let cube_surface_area : ℝ := 6 * cube_edge ^ 2
  ∃ ε > 0, |cube_surface_area - 798| < ε :=
by
  sorry


end cube_surface_area_from_prism_volume_l3820_382063


namespace resulting_polygon_sides_l3820_382082

/-- Represents a regular polygon with a given number of sides. -/
structure RegularPolygon where
  sides : ℕ
  is_regular : sides ≥ 3

/-- Represents the sequence of polygons in the construction. -/
def polygon_sequence : List RegularPolygon :=
  [⟨3, by norm_num⟩, ⟨4, by norm_num⟩, ⟨5, by norm_num⟩, 
   ⟨6, by norm_num⟩, ⟨7, by norm_num⟩, ⟨8, by norm_num⟩, ⟨9, by norm_num⟩]

/-- Calculates the number of exposed sides in the resulting polygon. -/
def exposed_sides (seq : List RegularPolygon) : ℕ :=
  (seq.map (·.sides)).sum - 2 * (seq.length - 1)

/-- Theorem stating that the resulting polygon has 30 sides. -/
theorem resulting_polygon_sides : exposed_sides polygon_sequence = 30 := by
  sorry


end resulting_polygon_sides_l3820_382082


namespace solve_sues_library_problem_l3820_382065

/-- Represents the number of books and movies Sue has --/
structure LibraryItems where
  books : ℕ
  movies : ℕ

/-- The problem statement about Sue's library items --/
def sues_library_problem (initial_items : LibraryItems) 
  (books_checked_out : ℕ) (final_total : ℕ) : Prop :=
  let movies_returned := initial_items.movies / 3
  let final_movies := initial_items.movies - movies_returned
  let total_books_before_return := initial_items.books + books_checked_out
  let final_books := final_total - final_movies
  let books_returned := total_books_before_return - final_books
  books_returned = 8

/-- Theorem stating the solution to Sue's library problem --/
theorem solve_sues_library_problem :
  sues_library_problem ⟨15, 6⟩ 9 20 := by
  sorry

end solve_sues_library_problem_l3820_382065


namespace correct_factorization_l3820_382089

theorem correct_factorization (x : ℝ) : x^2 - 4*x + 4 = (x - 2)^2 := by
  sorry

end correct_factorization_l3820_382089


namespace ratio_equality_l3820_382018

theorem ratio_equality : (2^3001 * 5^3003) / 10^3002 = 5/2 := by sorry

end ratio_equality_l3820_382018


namespace price_ratio_theorem_l3820_382022

theorem price_ratio_theorem (cost_price : ℝ) (price1 price2 : ℝ) 
  (h1 : price1 = cost_price * (1 + 0.32))
  (h2 : price2 = cost_price * (1 - 0.12)) :
  price2 / price1 = 2 / 3 := by
  sorry

end price_ratio_theorem_l3820_382022


namespace product_modulo_l3820_382024

theorem product_modulo : (1582 * 2031) % 600 = 42 := by
  sorry

end product_modulo_l3820_382024


namespace sin_shift_l3820_382079

theorem sin_shift (x : ℝ) : 
  Real.sin (2 * x - π / 3) = Real.sin (2 * (x - π / 6)) := by
  sorry

end sin_shift_l3820_382079


namespace computer_price_increase_l3820_382076

theorem computer_price_increase (original_price : ℝ) : 
  original_price * 1.3 = 351 → 2 * original_price = 540 := by
  sorry

end computer_price_increase_l3820_382076


namespace no_double_apply_function_exists_l3820_382035

theorem no_double_apply_function_exists : ¬∃ f : ℕ → ℕ, ∀ n : ℕ, f (f n) = n + 2015 := by
  sorry

end no_double_apply_function_exists_l3820_382035


namespace solution_correctness_l3820_382085

-- Define the equation
def equation (x : ℝ) : Prop := 2 * (x + 3) = 5 * x

-- Define the solution steps
def step1 (x : ℝ) : Prop := 2 * x + 6 = 5 * x
def step2 (x : ℝ) : Prop := 2 * x - 5 * x = -6
def step3 (x : ℝ) : Prop := -3 * x = -6
def step4 : ℝ := 2

-- Theorem stating the correctness of the solution and that step3 is not based on associative property
theorem solution_correctness :
  ∀ x : ℝ,
  equation x →
  step1 x ∧
  step2 x ∧
  step3 x ∧
  step4 = x ∧
  ¬(∃ a b c : ℝ, step3 x ↔ (a + b) + c = a + (b + c)) :=
by sorry

end solution_correctness_l3820_382085


namespace sum_of_exponents_outside_radical_l3820_382020

-- Define the original expression
def original_expression (a b c : ℝ) : ℝ := (48 * a^5 * b^8 * c^14)^(1/4)

-- Define the simplified expression
def simplified_expression (a b c : ℝ) : ℝ := 2 * a * b^2 * c^3 * (3 * a * c^2)^(1/4)

-- Theorem statement
theorem sum_of_exponents_outside_radical (a b c : ℝ) : 
  original_expression a b c = simplified_expression a b c → 
  (1 : ℕ) + 2 + 3 = 6 := by
  sorry

end sum_of_exponents_outside_radical_l3820_382020


namespace fair_ticket_cost_amy_ticket_spending_l3820_382073

/-- The total cost of tickets at a fair with regular and discounted prices -/
theorem fair_ticket_cost (initial_tickets : ℕ) (additional_tickets : ℕ) 
  (regular_price : ℚ) (discount_rate : ℚ) : ℚ :=
  let initial_cost := initial_tickets * regular_price
  let discount := regular_price * discount_rate
  let discounted_price := regular_price - discount
  let additional_cost := additional_tickets * discounted_price
  initial_cost + additional_cost

/-- Amy's total spending on fair tickets -/
theorem amy_ticket_spending : 
  fair_ticket_cost 33 21 (3/2) (1/4) = 73125/1000 := by
  sorry

end fair_ticket_cost_amy_ticket_spending_l3820_382073


namespace window_width_theorem_l3820_382060

/-- Represents the dimensions of a window pane -/
structure Pane where
  width : ℝ
  height : ℝ

/-- Represents the dimensions and properties of a window -/
structure Window where
  rows : ℕ
  columns : ℕ
  pane : Pane
  border_width : ℝ

/-- Calculates the total width of the window -/
def total_width (w : Window) : ℝ :=
  w.columns * w.pane.width + (w.columns + 1) * w.border_width

/-- Theorem stating the total width of the window with given conditions -/
theorem window_width_theorem (x : ℝ) : 
  let w : Window := {
    rows := 3,
    columns := 4,
    pane := { width := 4 * x, height := 3 * x },
    border_width := 3
  }
  total_width w = 16 * x + 15 := by sorry

end window_width_theorem_l3820_382060


namespace equal_distribution_of_boxes_l3820_382045

theorem equal_distribution_of_boxes (total_boxes : ℕ) (num_stops : ℕ) 
  (h1 : total_boxes = 27) (h2 : num_stops = 3) :
  total_boxes / num_stops = 9 := by
  sorry

end equal_distribution_of_boxes_l3820_382045


namespace least_time_four_horses_meet_l3820_382091

def horse_lap_time (k : ℕ) : ℕ := k

def all_horses_lcm : ℕ := 840

theorem least_time_four_horses_meet (T : ℕ) : T = 12 := by
  sorry

end least_time_four_horses_meet_l3820_382091


namespace maisys_current_wage_l3820_382095

/-- Proves that Maisy's current wage is $10 per hour given the job conditions --/
theorem maisys_current_wage (current_hours new_hours : ℕ) 
  (new_wage new_bonus difference : ℚ) :
  current_hours = 8 →
  new_hours = 4 →
  new_wage = 15 →
  new_bonus = 35 →
  difference = 15 →
  (new_hours : ℚ) * new_wage + new_bonus = 
    (current_hours : ℚ) * (10 : ℚ) + difference →
  10 = 10 := by
  sorry

#check maisys_current_wage

end maisys_current_wage_l3820_382095


namespace solution_set_implies_a_value_l3820_382090

theorem solution_set_implies_a_value (a : ℝ) : 
  (∀ x : ℝ, |a * x + 2| < 6 ↔ -1 < x ∧ x < 2) → a = -4 :=
by sorry

end solution_set_implies_a_value_l3820_382090


namespace expression_value_l3820_382098

theorem expression_value (x y z : ℚ) (hx : x = 3) (hy : y = 2) (hz : z = 4) :
  (3 * x - 4 * y) / z = 1 / 4 := by
  sorry

end expression_value_l3820_382098


namespace fruits_left_after_selling_l3820_382078

def initial_oranges : ℕ := 40
def initial_apples : ℕ := 70
def orange_sold_fraction : ℚ := 1/4
def apple_sold_fraction : ℚ := 1/2

theorem fruits_left_after_selling :
  (initial_oranges - orange_sold_fraction * initial_oranges) +
  (initial_apples - apple_sold_fraction * initial_apples) = 65 :=
by sorry

end fruits_left_after_selling_l3820_382078


namespace arithmetic_sequence_ratio_l3820_382061

/-- Given arithmetic sequences a and b with sums S and T, prove the ratio of a_6 to b_8 -/
theorem arithmetic_sequence_ratio (a b : ℕ → ℝ) (S T : ℕ → ℝ) :
  (∀ n, S n = (n + 2) * (T n / (n + 1))) →  -- Condition: S_n / T_n = (n + 2) / (n + 1)
  (∀ n, S (n + 1) - S n = a (n + 1)) →      -- Definition of S as sum of a
  (∀ n, T (n + 1) - T n = b (n + 1)) →      -- Definition of T as sum of b
  (∀ n, a (n + 1) - a n = a 2 - a 1) →      -- a is arithmetic sequence
  (∀ n, b (n + 1) - b n = b 2 - b 1) →      -- b is arithmetic sequence
  a 6 / b 8 = 13 / 16 :=
by sorry

end arithmetic_sequence_ratio_l3820_382061


namespace parallelogram_area_l3820_382086

def v : Fin 2 → ℝ := ![7, -4]
def w : Fin 2 → ℝ := ![13, -3]

theorem parallelogram_area : 
  abs (Matrix.det !![v 0, v 1; w 0, w 1]) = 31 := by sorry

end parallelogram_area_l3820_382086


namespace sin_cos_transformation_cos_transformation_sin_cos_equiv_transformation_l3820_382092

theorem sin_cos_transformation (x : Real) : 
  Real.sqrt 2 * Real.sin x = Real.sqrt 2 * Real.cos (x - Real.pi / 2) :=
by sorry

theorem cos_transformation (x : Real) : 
  Real.sqrt 2 * Real.cos (x - Real.pi / 2) = 
  Real.sqrt 2 * Real.cos ((1/2) * (2*x - Real.pi/4) + Real.pi/4) :=
by sorry

theorem sin_cos_equiv_transformation (x : Real) : 
  Real.sqrt 2 * Real.sin x = 
  Real.sqrt 2 * Real.cos ((1/2) * (2*x - Real.pi/4) + Real.pi/4) :=
by sorry

end sin_cos_transformation_cos_transformation_sin_cos_equiv_transformation_l3820_382092


namespace arithmetic_sum_2_to_20_l3820_382041

def arithmetic_sum (a₁ : ℕ) (aₙ : ℕ) (d : ℕ) : ℕ :=
  let n := (aₙ - a₁) / d + 1
  (a₁ + aₙ) * n / 2

theorem arithmetic_sum_2_to_20 :
  arithmetic_sum 2 20 2 = 110 := by
  sorry

end arithmetic_sum_2_to_20_l3820_382041


namespace john_driving_distance_john_driving_distance_proof_l3820_382043

theorem john_driving_distance : ℝ → Prop :=
  fun total_distance =>
    let speed1 : ℝ := 45
    let time1 : ℝ := 2
    let speed2 : ℝ := 50
    let time2 : ℝ := 3
    let distance1 := speed1 * time1
    let distance2 := speed2 * time2
    total_distance = distance1 + distance2 ∧ total_distance = 240

-- Proof
theorem john_driving_distance_proof : ∃ d : ℝ, john_driving_distance d := by
  sorry

end john_driving_distance_john_driving_distance_proof_l3820_382043


namespace sweetsies_leftover_l3820_382015

theorem sweetsies_leftover (n : ℕ) (h : n % 8 = 5) :
  (3 * n) % 8 = 7 :=
sorry

end sweetsies_leftover_l3820_382015


namespace triangle_max_area_l3820_382032

theorem triangle_max_area (a b c : ℝ) (h1 : a + b = 12) (h2 : c = 8) :
  let p := (a + b + c) / 2
  let area := Real.sqrt (p * (p - a) * (p - b) * (p - c))
  ∀ a' b' c', a' + b' = 12 → c' = 8 →
    let p' := (a' + b' + c') / 2
    let area' := Real.sqrt (p' * (p' - a') * (p' - b') * (p' - c'))
    area ≤ area' →
  area = 8 * Real.sqrt 5 := by
sorry


end triangle_max_area_l3820_382032


namespace multiply_101_by_101_l3820_382039

theorem multiply_101_by_101 : 101 * 101 = 10201 := by sorry

end multiply_101_by_101_l3820_382039


namespace fib_100_mod_5_l3820_382017

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fib n + fib (n + 1)

/-- Periodicity of Fibonacci sequence modulo 5 -/
axiom fib_mod_5_periodic (n : ℕ) : fib (n + 5) % 5 = fib n % 5

/-- Theorem: The 100th Fibonacci number modulo 5 is 0 -/
theorem fib_100_mod_5 : fib 100 % 5 = 0 := by
  sorry

end fib_100_mod_5_l3820_382017


namespace kyle_paper_delivery_l3820_382023

/-- The number of papers Kyle delivers in a week -/
def weekly_papers (weekday_houses : ℕ) (sunday_skip : ℕ) (sunday_extra : ℕ) : ℕ :=
  (weekday_houses * 6) + (weekday_houses - sunday_skip + sunday_extra)

/-- Theorem stating the total number of papers Kyle delivers in a week -/
theorem kyle_paper_delivery :
  weekly_papers 100 10 30 = 720 := by
  sorry

end kyle_paper_delivery_l3820_382023


namespace milk_water_ratio_l3820_382019

theorem milk_water_ratio (initial_milk : ℝ) (initial_water : ℝ) : 
  initial_milk > 0 ∧ initial_water > 0 →
  initial_milk + initial_water + 8 = 72 →
  (initial_milk + 8) / initial_water = 2 →
  initial_milk / initial_water = 5 / 3 :=
by sorry

end milk_water_ratio_l3820_382019


namespace sequence_decreases_eventually_l3820_382000

def a (n : ℕ) : ℚ := (100 : ℚ) ^ n / n.factorial

theorem sequence_decreases_eventually :
  ∃ N : ℕ, ∀ n ≥ N, a (n + 1) ≤ a n := by sorry

end sequence_decreases_eventually_l3820_382000


namespace three_team_soccer_game_total_score_l3820_382062

/-- Represents the score of a team in a half of the game -/
structure HalfScore where
  regular : ℕ
  penalties : ℕ

/-- Represents the score of a team for the whole game -/
structure GameScore where
  first_half : HalfScore
  second_half : HalfScore

/-- Calculate the total score for a team -/
def total_score (score : GameScore) : ℕ :=
  score.first_half.regular + score.first_half.penalties +
  score.second_half.regular + score.second_half.penalties

theorem three_team_soccer_game_total_score :
  let team_a : GameScore := {
    first_half := { regular := 8, penalties := 0 },
    second_half := { regular := 8, penalties := 1 }
  }
  let team_b : GameScore := {
    first_half := { regular := 4, penalties := 0 },
    second_half := { regular := 8, penalties := 2 }
  }
  let team_c : GameScore := {
    first_half := { regular := 8, penalties := 0 },
    second_half := { regular := 11, penalties := 0 }
  }
  team_b.first_half.regular = team_a.first_half.regular / 2 →
  team_c.first_half.regular = 2 * team_b.first_half.regular →
  team_a.second_half.regular = team_c.first_half.regular →
  team_b.second_half.regular = team_a.first_half.regular →
  team_c.second_half.regular = team_b.second_half.regular + 3 →
  total_score team_a + total_score team_b + total_score team_c = 50 :=
by
  sorry

end three_team_soccer_game_total_score_l3820_382062


namespace expression_evaluation_l3820_382059

theorem expression_evaluation :
  let x : ℚ := -2
  let y : ℚ := 1
  let expr := ((2 * x - 1/2 * y)^2 - (-y + 2*x) * (2*x + y) + y * (x^2 * y - 5/4 * y)) / x
  expr = -4 := by
sorry

end expression_evaluation_l3820_382059


namespace red_light_probability_l3820_382029

theorem red_light_probability (n : ℕ) (p : ℝ) (h1 : n = 4) (h2 : p = 1/3) :
  let q := 1 - p
  (q * q * p : ℝ) = 4/27 :=
by sorry

end red_light_probability_l3820_382029


namespace stamps_on_last_page_is_four_l3820_382094

/-- The number of stamps on the last page of Jenny's seventh book after reorganization --/
def stamps_on_last_page (
  initial_books : ℕ)
  (pages_per_book : ℕ)
  (initial_stamps_per_page : ℕ)
  (new_stamps_per_page : ℕ)
  (full_books_after_reorg : ℕ)
  (full_pages_in_last_book : ℕ) : ℕ :=
  let total_stamps := initial_books * pages_per_book * initial_stamps_per_page
  let stamps_in_full_books := full_books_after_reorg * pages_per_book * new_stamps_per_page
  let stamps_in_full_pages_of_last_book := full_pages_in_last_book * new_stamps_per_page
  total_stamps - stamps_in_full_books - stamps_in_full_pages_of_last_book

/-- Theorem stating that under the given conditions, there are 4 stamps on the last page --/
theorem stamps_on_last_page_is_four :
  stamps_on_last_page 10 50 8 12 6 37 = 4 := by
  sorry

end stamps_on_last_page_is_four_l3820_382094


namespace P_properties_l3820_382056

/-- P k l n denotes the number of partitions of n into no more than k summands, 
    each not exceeding l -/
def P (k l n : ℕ) : ℕ :=
  sorry

/-- The four properties of P as stated in the problem -/
theorem P_properties (k l n : ℕ) :
  (P k l n - P k (l-1) n = P (k-1) l (n-l)) ∧
  (P k l n - P (k-1) l n = P k (l-1) (n-k)) ∧
  (P k l n = P l k n) ∧
  (P k l n = P k l (k*l - n)) :=
by sorry

end P_properties_l3820_382056


namespace max_marked_cells_theorem_marked_cells_property_l3820_382048

/-- Represents an equilateral triangle divided into n^2 cells -/
structure DividedTriangle where
  n : ℕ
  cells : ℕ := n^2

/-- Represents the maximum number of cells that can be marked -/
def max_marked_cells (t : DividedTriangle) : ℕ :=
  if t.n = 10 then 7
  else if t.n = 9 then 6
  else 0  -- undefined for other values of n

/-- Theorem stating the maximum number of marked cells for n = 10 and n = 9 -/
theorem max_marked_cells_theorem (t : DividedTriangle) :
  (t.n = 10 → max_marked_cells t = 7) ∧
  (t.n = 9 → max_marked_cells t = 6) := by
  sorry

/-- Represents a strip in the divided triangle -/
structure Strip where
  cells : Finset ℕ

/-- Function to check if two cells are in the same strip -/
def in_same_strip (c1 c2 : ℕ) (s : Strip) : Prop :=
  c1 ∈ s.cells ∧ c2 ∈ s.cells

/-- The main theorem to be proved -/
theorem marked_cells_property (t : DividedTriangle) (marked_cells : Finset ℕ) :
  (∀ (s : Strip), ∀ (c1 c2 : ℕ), c1 ∈ marked_cells → c2 ∈ marked_cells →
    in_same_strip c1 c2 s → c1 = c2) →
  (t.n = 10 → marked_cells.card ≤ 7) ∧
  (t.n = 9 → marked_cells.card ≤ 6) := by
  sorry

end max_marked_cells_theorem_marked_cells_property_l3820_382048


namespace jane_earnings_l3820_382087

/-- Represents the number of bulbs planted for each flower type -/
structure BulbCounts where
  tulip : ℕ
  iris : ℕ
  hyacinth : ℕ
  daffodil : ℕ
  crocus : ℕ
  gladiolus : ℕ

/-- Represents the price per bulb for each flower type -/
structure BulbPrices where
  tulip : ℚ
  iris : ℚ
  hyacinth : ℚ
  daffodil : ℚ
  crocus : ℚ
  gladiolus : ℚ

def calculateEarnings (counts : BulbCounts) (prices : BulbPrices) : ℚ :=
  counts.tulip * prices.tulip +
  counts.iris * prices.iris +
  counts.hyacinth * prices.hyacinth +
  counts.daffodil * prices.daffodil +
  counts.crocus * prices.crocus +
  counts.gladiolus * prices.gladiolus

theorem jane_earnings (counts : BulbCounts) (prices : BulbPrices) :
  counts.tulip = 20 ∧
  counts.iris = counts.tulip / 2 ∧
  counts.hyacinth = counts.iris + counts.iris / 3 ∧
  counts.daffodil = 30 ∧
  counts.crocus = 3 * counts.daffodil ∧
  counts.gladiolus = 2 * (counts.crocus - counts.daffodil) + (15 * counts.daffodil / 100) ∧
  prices.tulip = 1/2 ∧
  prices.iris = 2/5 ∧
  prices.hyacinth = 3/4 ∧
  prices.daffodil = 1/4 ∧
  prices.crocus = 3/5 ∧
  prices.gladiolus = 3/10
  →
  calculateEarnings counts prices = 12245/100 := by
  sorry


end jane_earnings_l3820_382087


namespace isosceles_triangle_area_l3820_382071

/-- An isosceles triangle with given altitude and perimeter has area 75 -/
theorem isosceles_triangle_area (b s h : ℝ) : 
  h = 10 →                -- altitude is 10
  2 * s + 2 * b = 40 →    -- perimeter is 40
  s^2 = b^2 + h^2 →       -- Pythagorean theorem
  (1/2) * (2*b) * h = 75  -- area is 75
  := by sorry

end isosceles_triangle_area_l3820_382071
