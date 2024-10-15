import Mathlib

namespace NUMINAMATH_CALUDE_howard_window_washing_earnings_l3708_370842

theorem howard_window_washing_earnings
  (initial_amount : ℝ)
  (final_amount : ℝ)
  (cleaning_expenses : ℝ)
  (h1 : initial_amount = 26)
  (h2 : final_amount = 52)
  (h3 : final_amount = initial_amount + earnings - cleaning_expenses) :
  earnings = 26 + cleaning_expenses :=
by sorry

end NUMINAMATH_CALUDE_howard_window_washing_earnings_l3708_370842


namespace NUMINAMATH_CALUDE_plate_difference_l3708_370886

/- Define the number of kitchen supplies for Angela and Sharon -/
def angela_pots : ℕ := 20
def angela_plates : ℕ := 3 * angela_pots + 6
def angela_cutlery : ℕ := angela_plates / 2

def sharon_pots : ℕ := angela_pots / 2
def sharon_cutlery : ℕ := angela_cutlery * 2
def sharon_total : ℕ := 254

/- Define Sharon's plates as the remaining items after subtracting pots and cutlery from the total -/
def sharon_plates : ℕ := sharon_total - (sharon_pots + sharon_cutlery)

/- Theorem stating the difference between Sharon's plates and three times Angela's plates -/
theorem plate_difference : 
  3 * angela_plates - sharon_plates = 20 := by sorry

end NUMINAMATH_CALUDE_plate_difference_l3708_370886


namespace NUMINAMATH_CALUDE_vector_equation_l3708_370874

def a : ℝ × ℝ := (1, -1)
def b : ℝ × ℝ := (2, 1)
def c : ℝ × ℝ := (-2, 1)

theorem vector_equation (x y : ℝ) (h : c = x • a + y • b) : x - y = -1 := by
  sorry

end NUMINAMATH_CALUDE_vector_equation_l3708_370874


namespace NUMINAMATH_CALUDE_train_length_calculation_l3708_370828

/-- Calculates the length of a train given specific conditions of overtaking a motorbike -/
theorem train_length_calculation 
  (initial_train_speed : Real) 
  (train_acceleration : Real)
  (motorbike_speed : Real)
  (overtake_time : Real)
  (motorbike_length : Real)
  (h1 : initial_train_speed = 25)  -- 90 kmph converted to m/s
  (h2 : train_acceleration = 0.5)
  (h3 : motorbike_speed = 20)      -- 72 kmph converted to m/s
  (h4 : overtake_time = 50)
  (h5 : motorbike_length = 2) :
  let final_train_speed := initial_train_speed + train_acceleration * overtake_time
  let train_distance := initial_train_speed * overtake_time + 0.5 * train_acceleration * overtake_time^2
  let motorbike_distance := motorbike_speed * overtake_time
  let train_length := train_distance - motorbike_distance + motorbike_length
  train_length = 877 := by
  sorry

end NUMINAMATH_CALUDE_train_length_calculation_l3708_370828


namespace NUMINAMATH_CALUDE_remaining_water_bottles_samiras_remaining_bottles_l3708_370876

/-- Calculates the number of water bottles remaining after a soccer game --/
theorem remaining_water_bottles (initial_bottles : ℕ) (players : ℕ) 
  (bottles_first_break : ℕ) (bottles_end_game : ℕ) : ℕ :=
  let bottles_after_first_break := initial_bottles - players * bottles_first_break
  let final_remaining_bottles := bottles_after_first_break - players * bottles_end_game
  final_remaining_bottles

/-- Proves that given the specific conditions of Samira's soccer game, 
    15 water bottles remain --/
theorem samiras_remaining_bottles : 
  remaining_water_bottles (4 * 12) 11 2 1 = 15 := by
  sorry

end NUMINAMATH_CALUDE_remaining_water_bottles_samiras_remaining_bottles_l3708_370876


namespace NUMINAMATH_CALUDE_inequality_proof_l3708_370893

theorem inequality_proof (a b c : ℝ) (h1 : a > b) (h2 : b > c) : 
  (1 / (a - b)) + (1 / (b - c)) + (1 / (c - a)) > 0 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l3708_370893


namespace NUMINAMATH_CALUDE_rectangular_room_length_l3708_370847

theorem rectangular_room_length (area width : ℝ) (h1 : area = 215.6) (h2 : width = 14) :
  area / width = 15.4 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_room_length_l3708_370847


namespace NUMINAMATH_CALUDE_average_b_c_l3708_370825

theorem average_b_c (a b c : ℝ) 
  (h1 : (a + b) / 2 = 35) 
  (h2 : c - a = 90) : 
  (b + c) / 2 = 80 := by
sorry

end NUMINAMATH_CALUDE_average_b_c_l3708_370825


namespace NUMINAMATH_CALUDE_arithmetic_sequence_8th_term_l3708_370888

/-- Proves that the 8th term of an arithmetic sequence with 26 terms, 
    first term 4, and last term 104, is equal to 32. -/
theorem arithmetic_sequence_8th_term 
  (a : ℕ → ℝ) 
  (h1 : a 1 = 4) 
  (h2 : a 26 = 104) 
  (h3 : ∀ n : ℕ, 1 < n → n ≤ 26 → a n - a (n-1) = a 2 - a 1) :
  a 8 = 32 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_8th_term_l3708_370888


namespace NUMINAMATH_CALUDE_matching_shoes_probability_l3708_370887

theorem matching_shoes_probability (total_shoes : ℕ) (total_pairs : ℕ) (h1 : total_shoes = 12) (h2 : total_pairs = 6) :
  let total_selections := total_shoes.choose 2
  let matching_selections := total_pairs
  (matching_selections : ℚ) / total_selections = 1 / 11 := by sorry

end NUMINAMATH_CALUDE_matching_shoes_probability_l3708_370887


namespace NUMINAMATH_CALUDE_nested_fraction_evaluation_l3708_370804

theorem nested_fraction_evaluation : 
  (1 / (3 - 1 / (3 - 1 / (3 - 1 / 3)))) = 8 / 21 := by
  sorry

end NUMINAMATH_CALUDE_nested_fraction_evaluation_l3708_370804


namespace NUMINAMATH_CALUDE_fraction_equation_solution_l3708_370827

theorem fraction_equation_solution :
  ∃! x : ℚ, (x + 5) / (x - 3) = (x - 2) / (x + 4) :=
by
  use -1
  constructor
  · -- Prove that x = -1 satisfies the equation
    sorry
  · -- Prove uniqueness
    sorry

end NUMINAMATH_CALUDE_fraction_equation_solution_l3708_370827


namespace NUMINAMATH_CALUDE_golden_ratio_unique_progression_l3708_370860

theorem golden_ratio_unique_progression : ∃! x : ℝ, 
  x > 0 ∧ 
  let b := ⌊x⌋
  let c := x - b
  (c < b) ∧ (b < x) ∧ (c * x = b * b) ∧ x = (Real.sqrt 5 + 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_golden_ratio_unique_progression_l3708_370860


namespace NUMINAMATH_CALUDE_similar_triangle_longest_side_l3708_370835

def triangle_sides (a b c : ℝ) := a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b

theorem similar_triangle_longest_side 
  (a b c : ℝ) 
  (h_triangle : triangle_sides a b c) 
  (h_sides : a = 5 ∧ b = 12 ∧ c = 13) 
  (k : ℝ) 
  (h_similar : k > 0)
  (h_perimeter : k * (a + b + c) = 150) :
  k * max a (max b c) = 65 := by
sorry

end NUMINAMATH_CALUDE_similar_triangle_longest_side_l3708_370835


namespace NUMINAMATH_CALUDE_find_k_value_l3708_370866

/-- Given two functions f and g, prove that k = 27/25 when f(5) - g(5) = 45 -/
theorem find_k_value (f g : ℝ → ℝ) (k : ℝ) 
    (hf : ∀ x, f x = 2*x^3 - 5*x^2 + 3*x + 7)
    (hg : ∀ x, g x = 3*x^3 - k*x^2 + 4)
    (h_diff : f 5 - g 5 = 45) : 
  k = 27/25 := by
sorry

end NUMINAMATH_CALUDE_find_k_value_l3708_370866


namespace NUMINAMATH_CALUDE_quadratic_equation_properties_l3708_370872

theorem quadratic_equation_properties (m : ℝ) :
  let f : ℝ → ℝ := λ x => x^2 - m*x + m - 2
  (∃ x, f x = 0 ∧ x = -1) → m = 1/2 ∧
  ∃ x y, x ≠ y ∧ f x = 0 ∧ f y = 0 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_properties_l3708_370872


namespace NUMINAMATH_CALUDE_set_equality_sum_l3708_370882

theorem set_equality_sum (x y : ℝ) (A B : Set ℝ) : 
  A = {2, y} → B = {x, 3} → A = B → x + y = 5 := by
  sorry

end NUMINAMATH_CALUDE_set_equality_sum_l3708_370882


namespace NUMINAMATH_CALUDE_circle_and_tangent_line_equations_l3708_370836

/-- A circle with center C on the line x - y + 1 = 0 passing through points (1, 1) and (2, -2) -/
structure CircleC where
  center : ℝ × ℝ
  center_on_line : center.1 - center.2 + 1 = 0
  passes_through_A : (center.1 - 1)^2 + (center.2 - 1)^2 = (center.1 - 2)^2 + (center.2 + 2)^2

/-- The standard equation of the circle and its tangent line -/
def circle_equation (c : CircleC) : Prop :=
  ∀ (x y : ℝ), (x + 3)^2 + (y + 2)^2 = 25 ↔ (x - c.center.1)^2 + (y - c.center.2)^2 = (c.center.1 - 1)^2 + (c.center.2 - 1)^2

def tangent_line_equation (c : CircleC) : Prop :=
  ∀ (x y : ℝ), 4*x + 3*y - 7 = 0 ↔ 
    ((x - 1) * (c.center.1 - 1) + (y - 1) * (c.center.2 - 1) = (c.center.1 - 1)^2 + (c.center.2 - 1)^2) ∧
    ((x, y) ≠ (1, 1))

/-- The main theorem stating that the circle equation and tangent line equation are correct -/
theorem circle_and_tangent_line_equations (c : CircleC) : 
  circle_equation c ∧ tangent_line_equation c :=
sorry

end NUMINAMATH_CALUDE_circle_and_tangent_line_equations_l3708_370836


namespace NUMINAMATH_CALUDE_derivative_f_at_zero_l3708_370868

def f (x : ℝ) := x + x^2

theorem derivative_f_at_zero : 
  deriv f 0 = 1 := by sorry

end NUMINAMATH_CALUDE_derivative_f_at_zero_l3708_370868


namespace NUMINAMATH_CALUDE_ceiling_minus_x_l3708_370859

theorem ceiling_minus_x (x : ℝ) (h : ⌈x⌉ - ⌊x⌋ = 1) :
  ∃ α : ℝ, 0 < α ∧ α < 1 ∧ x = ⌊x⌋ + α ∧ ⌈x⌉ - x = 1 - α :=
sorry

end NUMINAMATH_CALUDE_ceiling_minus_x_l3708_370859


namespace NUMINAMATH_CALUDE_imaginary_complex_implies_m_condition_l3708_370829

theorem imaginary_complex_implies_m_condition (m : ℝ) : 
  (Complex.I * (m^2 - 5*m - 6) ≠ 0) → (m ≠ -1 ∧ m ≠ 6) := by
  sorry

end NUMINAMATH_CALUDE_imaginary_complex_implies_m_condition_l3708_370829


namespace NUMINAMATH_CALUDE_equation_solution_l3708_370817

theorem equation_solution (x : ℂ) : 
  (x - 2)^6 + (x - 6)^6 = 64 ↔ x = 4 + Complex.I * Real.sqrt 2 ∨ x = 4 - Complex.I * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3708_370817


namespace NUMINAMATH_CALUDE_smallest_a_for_positive_integer_roots_l3708_370861

theorem smallest_a_for_positive_integer_roots : ∃ (a : ℕ),
  (∀ (x₁ x₂ : ℕ), x₁ * x₂ = 2022 ∧ x₁ + x₂ = a → x₁^2 - a*x₁ + 2022 = 0 ∧ x₂^2 - a*x₂ + 2022 = 0) ∧
  (∀ (b : ℕ), b < a →
    ¬∃ (y₁ y₂ : ℕ), y₁ * y₂ = 2022 ∧ y₁ + y₂ = b ∧ y₁^2 - b*y₁ + 2022 = 0 ∧ y₂^2 - b*y₂ + 2022 = 0) ∧
  a = 343 :=
by
  sorry

#check smallest_a_for_positive_integer_roots

end NUMINAMATH_CALUDE_smallest_a_for_positive_integer_roots_l3708_370861


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l3708_370810

theorem geometric_sequence_common_ratio 
  (a : ℕ → ℝ) 
  (h_geometric : ∃ (q : ℝ) (a₁ : ℝ), ∀ n, a n = a₁ * q^(n-1))
  (h_condition : 2 * a 4 = a 6 - a 5) :
  ∃ (q : ℝ), (q = -1 ∨ q = 2) ∧ 
    (∃ (a₁ : ℝ), ∀ n, a n = a₁ * q^(n-1)) := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l3708_370810


namespace NUMINAMATH_CALUDE_tomato_price_equality_l3708_370851

/-- Prove that the original price per pound of tomatoes equals the selling price of remaining tomatoes --/
theorem tomato_price_equality (original_price : ℝ) 
  (ruined_percentage : ℝ) (profit_percentage : ℝ) (selling_price : ℝ) : 
  ruined_percentage = 0.2 →
  profit_percentage = 0.08 →
  selling_price = 1.08 →
  (1 - ruined_percentage) * selling_price = (1 + profit_percentage) * original_price :=
by sorry

end NUMINAMATH_CALUDE_tomato_price_equality_l3708_370851


namespace NUMINAMATH_CALUDE_bridge_length_at_least_train_length_l3708_370803

/-- Proves that the length of a bridge is at least as long as a train, given the train's length,
    speed, and time to cross the bridge. -/
theorem bridge_length_at_least_train_length
  (train_length : ℝ)
  (train_speed_kmh : ℝ)
  (crossing_time : ℝ)
  (h1 : train_length = 200)
  (h2 : train_speed_kmh = 32)
  (h3 : crossing_time = 20)
  : ∃ (bridge_length : ℝ), bridge_length ≥ train_length :=
by
  sorry

#check bridge_length_at_least_train_length

end NUMINAMATH_CALUDE_bridge_length_at_least_train_length_l3708_370803


namespace NUMINAMATH_CALUDE_triangle_angle_problem_l3708_370801

theorem triangle_angle_problem (A B C : ℝ) 
  (h1 : B = A + 10)
  (h2 : C = B + 10)
  (h3 : A + B + C = 180) :
  B = 60 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_problem_l3708_370801


namespace NUMINAMATH_CALUDE_digit_sum_property_l3708_370838

/-- A function that checks if a natural number has no zero digits -/
def has_no_zero_digits (n : ℕ) : Prop := sorry

/-- A function that generates all digit permutations of a natural number -/
def digit_permutations (n : ℕ) : Finset ℕ := sorry

/-- A function that checks if a natural number is composed solely of ones -/
def all_ones (n : ℕ) : Prop := sorry

/-- A function that checks if a natural number has at least one digit 5 or greater -/
def has_digit_ge_5 (n : ℕ) : Prop := sorry

theorem digit_sum_property (n : ℕ) :
  has_no_zero_digits n →
  ∃ (p₁ p₂ p₃ : ℕ), p₁ ∈ digit_permutations n ∧ 
                    p₂ ∈ digit_permutations n ∧ 
                    p₃ ∈ digit_permutations n ∧
                    all_ones (n + p₁ + p₂ + p₃) →
  has_digit_ge_5 n :=
sorry

end NUMINAMATH_CALUDE_digit_sum_property_l3708_370838


namespace NUMINAMATH_CALUDE_probability_point_in_circle_l3708_370871

theorem probability_point_in_circle (s : ℝ) (r : ℝ) (h_s : s = 6) (h_r : r = 1.5) :
  (π * r^2) / (s^2) = π / 16 := by
  sorry

end NUMINAMATH_CALUDE_probability_point_in_circle_l3708_370871


namespace NUMINAMATH_CALUDE_tangent_line_circle_l3708_370853

/-- The line 4x - 3y = 0 is tangent to the circle x^2 + y^2 - 2x + ay + 1 = 0 if and only if a = -1 or a = 4 -/
theorem tangent_line_circle (a : ℝ) : 
  (∀ x y : ℝ, (4 * x - 3 * y = 0 ∧ x^2 + y^2 - 2*x + a*y + 1 = 0) → 
    (∀ x' y' : ℝ, x'^2 + y'^2 - 2*x' + a*y' + 1 = 0 → (x = x' ∧ y = y'))) ↔ 
  (a = -1 ∨ a = 4) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_circle_l3708_370853


namespace NUMINAMATH_CALUDE_sqrt_of_nine_equals_three_l3708_370832

theorem sqrt_of_nine_equals_three : Real.sqrt 9 = 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_of_nine_equals_three_l3708_370832


namespace NUMINAMATH_CALUDE_annie_extracurricular_hours_l3708_370820

def chess_hours : ℕ := 2
def drama_hours : ℕ := 8
def glee_hours : ℕ := 3
def total_weeks : ℕ := 12
def sick_weeks : ℕ := 2

def extracurricular_hours_per_week : ℕ := chess_hours + drama_hours + glee_hours
def active_weeks : ℕ := total_weeks - sick_weeks

theorem annie_extracurricular_hours :
  extracurricular_hours_per_week * active_weeks = 130 := by
  sorry

end NUMINAMATH_CALUDE_annie_extracurricular_hours_l3708_370820


namespace NUMINAMATH_CALUDE_raines_change_l3708_370821

/-- Calculates the change Raine receives after purchasing items with a discount --/
theorem raines_change (bracelet_price necklace_price mug_price : ℚ)
  (bracelet_qty necklace_qty mug_qty : ℕ)
  (discount_rate : ℚ)
  (payment : ℚ) :
  bracelet_price = 15 →
  necklace_price = 10 →
  mug_price = 20 →
  bracelet_qty = 3 →
  necklace_qty = 2 →
  mug_qty = 1 →
  discount_rate = 1/10 →
  payment = 100 →
  let total_cost := bracelet_price * bracelet_qty + necklace_price * necklace_qty + mug_price * mug_qty
  let discounted_cost := total_cost * (1 - discount_rate)
  payment - discounted_cost = 23.5 := by sorry

end NUMINAMATH_CALUDE_raines_change_l3708_370821


namespace NUMINAMATH_CALUDE_mean_of_added_numbers_l3708_370889

theorem mean_of_added_numbers (original_count : ℕ) (original_mean : ℚ) 
  (new_count : ℕ) (new_mean : ℚ) (added_count : ℕ) : 
  original_count = 8 →
  original_mean = 72 →
  new_count = 11 →
  new_mean = 85 →
  added_count = 3 →
  (new_count * new_mean - original_count * original_mean) / added_count = 119 + 2/3 := by
  sorry

end NUMINAMATH_CALUDE_mean_of_added_numbers_l3708_370889


namespace NUMINAMATH_CALUDE_hearty_beads_count_l3708_370837

/-- The number of packages of blue beads Hearty bought -/
def blue_packages : ℕ := 3

/-- The number of packages of red beads Hearty bought -/
def red_packages : ℕ := 5

/-- The number of beads in each package -/
def beads_per_package : ℕ := 40

/-- The total number of beads Hearty has -/
def total_beads : ℕ := blue_packages * beads_per_package + red_packages * beads_per_package

theorem hearty_beads_count : total_beads = 320 := by
  sorry

end NUMINAMATH_CALUDE_hearty_beads_count_l3708_370837


namespace NUMINAMATH_CALUDE_prob_defective_second_draw_specific_l3708_370864

/-- Probability of drawing a defective item on the second draw -/
def prob_defective_second_draw (total : ℕ) (defective : ℕ) (good : ℕ) : ℚ :=
  if total > 0 ∧ good > 0 then
    defective / (total - 1 : ℚ)
  else
    0

theorem prob_defective_second_draw_specific :
  prob_defective_second_draw 10 3 7 = 1/3 :=
by sorry

end NUMINAMATH_CALUDE_prob_defective_second_draw_specific_l3708_370864


namespace NUMINAMATH_CALUDE_angle_from_terminal_point_l3708_370865

theorem angle_from_terminal_point (α : Real) :
  (∃ (x y : Real), x = Real.sin (π / 5) ∧ y = -Real.cos (π / 5) ∧ 
   x = Real.sin α ∧ y = Real.cos α) →
  ∃ (k : ℤ), α = -3 * π / 10 + 2 * π * (k : Real) :=
by sorry

end NUMINAMATH_CALUDE_angle_from_terminal_point_l3708_370865


namespace NUMINAMATH_CALUDE_quadratic_reciprocal_roots_l3708_370869

theorem quadratic_reciprocal_roots (p q : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ x^2 + p*x + q = 0 ∧ y^2 + p*y + q = 0 ∧ x*y = 1) →
  ((p ≥ 2 ∨ p ≤ -2) ∧ q = 1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_reciprocal_roots_l3708_370869


namespace NUMINAMATH_CALUDE_odd_divides_power_factorial_minus_one_l3708_370819

theorem odd_divides_power_factorial_minus_one (n : ℕ) (h : Odd n) : n ∣ 2^(n.factorial) - 1 := by
  sorry

end NUMINAMATH_CALUDE_odd_divides_power_factorial_minus_one_l3708_370819


namespace NUMINAMATH_CALUDE_root_difference_equals_1993_l3708_370870

theorem root_difference_equals_1993 : ∃ m n : ℝ,
  (1992 * m)^2 - 1991 * 1993 * m - 1 = 0 ∧
  n^2 + 1991 * n - 1992 = 0 ∧
  (∀ x : ℝ, (1992 * x)^2 - 1991 * 1993 * x - 1 = 0 → x ≤ m) ∧
  (∀ y : ℝ, y^2 + 1991 * y - 1992 = 0 → y ≤ n) ∧
  m - n = 1993 :=
sorry

end NUMINAMATH_CALUDE_root_difference_equals_1993_l3708_370870


namespace NUMINAMATH_CALUDE_sweater_price_theorem_l3708_370812

/-- The marked price of a sweater in yuan -/
def marked_price : ℝ := 150

/-- The selling price as a percentage of the marked price -/
def selling_percentage : ℝ := 0.8

/-- The profit percentage -/
def profit_percentage : ℝ := 0.2

/-- The purchase price of the sweater in yuan -/
def purchase_price : ℝ := 100

theorem sweater_price_theorem : 
  selling_percentage * marked_price = purchase_price * (1 + profit_percentage) :=
sorry

end NUMINAMATH_CALUDE_sweater_price_theorem_l3708_370812


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_l3708_370885

theorem perfect_square_trinomial (m : ℝ) : 
  (∃ a b : ℝ, ∀ x : ℝ, x^2 + (m-1)*x + 9 = (a*x + b)^2) → 
  (m = 7 ∨ m = -5) :=
by sorry

end NUMINAMATH_CALUDE_perfect_square_trinomial_l3708_370885


namespace NUMINAMATH_CALUDE_tiffany_bags_theorem_l3708_370867

/-- The total number of bags Tiffany collected over three days -/
def total_bags (initial : ℕ) (day2 : ℕ) (day3 : ℕ) : ℕ :=
  initial + day2 + day3

/-- Theorem stating that Tiffany's total bags equals 20 given the initial conditions -/
theorem tiffany_bags_theorem :
  total_bags 10 3 7 = 20 := by
  sorry

end NUMINAMATH_CALUDE_tiffany_bags_theorem_l3708_370867


namespace NUMINAMATH_CALUDE_cube_inequality_equivalence_l3708_370873

theorem cube_inequality_equivalence (a b : ℝ) : a > b ↔ a^3 > b^3 := by sorry

end NUMINAMATH_CALUDE_cube_inequality_equivalence_l3708_370873


namespace NUMINAMATH_CALUDE_boys_in_class_l3708_370802

theorem boys_in_class (total : ℕ) (diff : ℕ) (boys : ℕ) : 
  total = 485 →
  diff = 69 →
  total = boys + (boys + diff) →
  boys = 208 := by
sorry

end NUMINAMATH_CALUDE_boys_in_class_l3708_370802


namespace NUMINAMATH_CALUDE_box_volume_from_face_centers_l3708_370890

def rectangular_box_volume (a b c : ℝ) : ℝ := 8 * a * b * c

theorem box_volume_from_face_centers 
  (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h1 : a^2 + b^2 = 4^2)
  (h2 : b^2 + c^2 = 5^2)
  (h3 : a^2 + c^2 = 6^2) :
  rectangular_box_volume a b c = 90 * Real.sqrt 6 :=
sorry

end NUMINAMATH_CALUDE_box_volume_from_face_centers_l3708_370890


namespace NUMINAMATH_CALUDE_variance_of_dataset_l3708_370816

def dataset : List ℝ := [5, 7, 7, 8, 10, 11]

/-- The variance of the dataset [5, 7, 7, 8, 10, 11] is 4 -/
theorem variance_of_dataset : 
  let n : ℝ := dataset.length
  let mean : ℝ := (dataset.sum) / n
  let variance : ℝ := (dataset.map (λ x => (x - mean)^2)).sum / n
  variance = 4 := by sorry

end NUMINAMATH_CALUDE_variance_of_dataset_l3708_370816


namespace NUMINAMATH_CALUDE_admission_methods_l3708_370881

theorem admission_methods (n : ℕ) (k : ℕ) (s : ℕ) : 
  n = 8 → k = 2 → s = 3 → (n.choose k) * s = 84 :=
by sorry

end NUMINAMATH_CALUDE_admission_methods_l3708_370881


namespace NUMINAMATH_CALUDE_max_gcd_of_sequence_l3708_370863

theorem max_gcd_of_sequence (n : ℕ) : 
  Nat.gcd ((8^n - 1) / 7) ((8^(n+1) - 1) / 7) = 1 :=
by sorry

end NUMINAMATH_CALUDE_max_gcd_of_sequence_l3708_370863


namespace NUMINAMATH_CALUDE_hogwarts_total_students_l3708_370840

-- Define the given conditions
def total_participants : ℕ := 246
def total_boys : ℕ := 255

-- Define the relationship between participating boys and non-participating girls
def boys_participating_girls_not (total_students : ℕ) : Prop :=
  ∃ (boys_participating : ℕ) (girls_not_participating : ℕ),
    boys_participating = girls_not_participating + 11 ∧
    boys_participating ≤ total_boys ∧
    girls_not_participating ≤ total_students - total_boys

-- Theorem statement
theorem hogwarts_total_students : 
  ∃ (total_students : ℕ),
    total_students = 490 ∧
    boys_participating_girls_not total_students :=
by
  sorry


end NUMINAMATH_CALUDE_hogwarts_total_students_l3708_370840


namespace NUMINAMATH_CALUDE_min_tablets_to_extract_l3708_370892

/-- Represents the number of tablets of each medicine type in the box -/
def tablets_per_type : ℕ := 10

/-- Represents the minimum number of tablets of each type we want to guarantee -/
def min_tablets_per_type : ℕ := 2

/-- Theorem: The minimum number of tablets to extract to guarantee at least two of each type -/
theorem min_tablets_to_extract :
  tablets_per_type + min_tablets_per_type = 12 := by sorry

end NUMINAMATH_CALUDE_min_tablets_to_extract_l3708_370892


namespace NUMINAMATH_CALUDE_count_positive_area_triangles_l3708_370899

/-- The total number of points in the grid -/
def total_points : ℕ := 7

/-- The number of sets of three collinear points -/
def collinear_sets : ℕ := 5

/-- The number of triangles with positive area -/
def positive_area_triangles : ℕ := 30

/-- Theorem stating the number of triangles with positive area -/
theorem count_positive_area_triangles :
  (Nat.choose total_points 3) - collinear_sets = positive_area_triangles :=
by sorry

end NUMINAMATH_CALUDE_count_positive_area_triangles_l3708_370899


namespace NUMINAMATH_CALUDE_triangle_covering_polygon_l3708_370823

-- Define the types for points and polygons
variable (Point : Type) [NormedAddCommGroup Point] [InnerProductSpace ℝ Point]
variable (Polygon : Type)
variable (Triangle : Type)

-- Define the properties and relations
variable (covers : Triangle → Polygon → Prop)
variable (congruent : Triangle → Triangle → Prop)
variable (has_parallel_side : Triangle → Polygon → Prop)

-- State the theorem
theorem triangle_covering_polygon
  (ABC : Triangle) (M : Polygon) 
  (h_covers : covers ABC M) :
  ∃ (DEF : Triangle), 
    congruent DEF ABC ∧ 
    covers DEF M ∧ 
    has_parallel_side DEF M :=
sorry

end NUMINAMATH_CALUDE_triangle_covering_polygon_l3708_370823


namespace NUMINAMATH_CALUDE_two_digit_swap_difference_divisible_by_nine_l3708_370852

theorem two_digit_swap_difference_divisible_by_nine 
  (a b : ℕ) 
  (h1 : a ≤ 9) 
  (h2 : b ≤ 9) 
  (h3 : a ≠ b) : 
  ∃ k : ℤ, (|(10 * a + b) - (10 * b + a)| : ℤ) = 9 * k := by
  sorry

end NUMINAMATH_CALUDE_two_digit_swap_difference_divisible_by_nine_l3708_370852


namespace NUMINAMATH_CALUDE_largest_n_satisfying_conditions_l3708_370855

theorem largest_n_satisfying_conditions : ∃ (n : ℕ), n = 50 ∧ 
  (∀ m : ℕ, n^2 = (m+1)^3 - m^3 → m ≤ 50) ∧
  (∃ k : ℕ, 2*n + 99 = k^2) ∧
  (∀ n' : ℕ, n' > n → 
    (¬∃ m : ℕ, n'^2 = (m+1)^3 - m^3) ∨ 
    (¬∃ k : ℕ, 2*n' + 99 = k^2)) :=
by sorry

end NUMINAMATH_CALUDE_largest_n_satisfying_conditions_l3708_370855


namespace NUMINAMATH_CALUDE_root_equation_problem_l3708_370815

theorem root_equation_problem (a b m p r : ℝ) : 
  (a^2 - m*a + 4 = 0) →
  (b^2 - m*b + 4 = 0) →
  ((a - 1/b)^2 - p*(a - 1/b) + r = 0) →
  ((b - 1/a)^2 - p*(b - 1/a) + r = 0) →
  r = 9/4 := by sorry

end NUMINAMATH_CALUDE_root_equation_problem_l3708_370815


namespace NUMINAMATH_CALUDE_definite_integral_2x_l3708_370826

theorem definite_integral_2x : ∫ x in (0)..(π/2), 2*x = π^2/4 := by
  sorry

end NUMINAMATH_CALUDE_definite_integral_2x_l3708_370826


namespace NUMINAMATH_CALUDE_twin_brothers_age_l3708_370879

theorem twin_brothers_age :
  ∀ (x : ℕ), 
    (x * x + 9 = (x + 1) * (x + 1)) → 
    x = 4 := by
  sorry

end NUMINAMATH_CALUDE_twin_brothers_age_l3708_370879


namespace NUMINAMATH_CALUDE_sequence_completeness_l3708_370800

theorem sequence_completeness (a : ℕ → ℤ) :
  (∀ n : ℕ, n > 0 → (Finset.range n).card = (Finset.image (λ i => a i % n) (Finset.range n)).card) →
  ∀ k : ℤ, ∃! i : ℕ, a i = k :=
sorry

end NUMINAMATH_CALUDE_sequence_completeness_l3708_370800


namespace NUMINAMATH_CALUDE_x_can_be_negative_one_l3708_370843

theorem x_can_be_negative_one : ∃ (x : ℝ), x = -1 ∧ x^2 ∈ ({0, 1, x} : Set ℝ) := by
  sorry

end NUMINAMATH_CALUDE_x_can_be_negative_one_l3708_370843


namespace NUMINAMATH_CALUDE_sine_equation_solution_l3708_370846

theorem sine_equation_solution (x y : ℝ) :
  |Real.sin x - Real.sin y| + Real.sin x * Real.sin y = 0 →
  ∃ k n : ℤ, x = k * Real.pi ∧ y = n * Real.pi := by
sorry

end NUMINAMATH_CALUDE_sine_equation_solution_l3708_370846


namespace NUMINAMATH_CALUDE_k_eval_at_one_l3708_370830

-- Define the polynomials h and k
def h (p : ℝ) (x : ℝ) : ℝ := x^3 + p*x^2 + 2*x + 15
def k (q r : ℝ) (x : ℝ) : ℝ := x^4 + x^3 + q*x^2 + 150*x + r

-- State the theorem
theorem k_eval_at_one (p q r : ℝ) : 
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ h p x = 0 ∧ h p y = 0 ∧ h p z = 0) →  -- h has three distinct roots
  (∀ x : ℝ, h p x = 0 → k q r x = 0) →  -- each root of h is a root of k
  k q r 1 = -3322.25 := by
sorry

end NUMINAMATH_CALUDE_k_eval_at_one_l3708_370830


namespace NUMINAMATH_CALUDE_cone_lateral_surface_area_l3708_370896

theorem cone_lateral_surface_area (radius : ℝ) (slant_height : ℝ) :
  radius = 3 → slant_height = 5 → π * radius * slant_height = 15 * π := by
  sorry

end NUMINAMATH_CALUDE_cone_lateral_surface_area_l3708_370896


namespace NUMINAMATH_CALUDE_log_sum_equality_l3708_370809

theorem log_sum_equality : Real.log 8 / Real.log 10 + 3 * (Real.log 5 / Real.log 10) = 3 := by
  sorry

end NUMINAMATH_CALUDE_log_sum_equality_l3708_370809


namespace NUMINAMATH_CALUDE_colored_paper_usage_l3708_370839

theorem colored_paper_usage (initial_sheets : ℕ) (sheets_used : ℕ) : 
  initial_sheets = 82 →
  initial_sheets - sheets_used = sheets_used - 6 →
  sheets_used = 44 := by
  sorry

end NUMINAMATH_CALUDE_colored_paper_usage_l3708_370839


namespace NUMINAMATH_CALUDE_min_value_of_m_l3708_370841

theorem min_value_of_m (a b c m : ℝ) (h1 : a > b) (h2 : b > c) (h3 : m > 0)
  (h4 : ∀ a b c, a > b ∧ b > c → 1 / (a - b) + m / (b - c) ≥ 9 / (a - c)) :
  m ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_m_l3708_370841


namespace NUMINAMATH_CALUDE_sqrt_meaningful_range_l3708_370884

theorem sqrt_meaningful_range (x : ℝ) : 
  (∃ y : ℝ, y^2 = 2*x - 6) ↔ x ≥ 3 := by sorry

end NUMINAMATH_CALUDE_sqrt_meaningful_range_l3708_370884


namespace NUMINAMATH_CALUDE_isosceles_triangle_rectangle_perimeter_difference_unique_d_value_count_impossible_d_values_l3708_370818

theorem isosceles_triangle_rectangle_perimeter_difference 
  (d : ℕ) (w : ℝ) : 
  w > 0 → 
  6 * w > 0 → 
  6 * w + 2 * d = 6 * w + 1236 → 
  d = 618 := by
sorry

theorem unique_d_value : 
  ∃! d : ℕ, ∃ w : ℝ, 
    w > 0 ∧ 
    6 * w > 0 ∧ 
    6 * w + 2 * d = 6 * w + 1236 := by
sorry

theorem count_impossible_d_values : 
  (Nat.card {d : ℕ | ¬∃ w : ℝ, w > 0 ∧ 6 * w > 0 ∧ 6 * w + 2 * d = 6 * w + 1236}) = ℵ₀ := by
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_rectangle_perimeter_difference_unique_d_value_count_impossible_d_values_l3708_370818


namespace NUMINAMATH_CALUDE_thabo_book_difference_l3708_370880

/-- Represents the number of books Thabo owns of each type -/
structure BookCollection where
  paperbackFiction : ℕ
  paperbackNonfiction : ℕ
  hardcoverNonfiction : ℕ

/-- The properties of Thabo's book collection -/
def validCollection (books : BookCollection) : Prop :=
  books.paperbackFiction + books.paperbackNonfiction + books.hardcoverNonfiction = 280 ∧
  books.paperbackNonfiction > books.hardcoverNonfiction ∧
  books.paperbackFiction = 2 * books.paperbackNonfiction ∧
  books.hardcoverNonfiction = 55

theorem thabo_book_difference (books : BookCollection) 
  (h : validCollection books) : 
  books.paperbackNonfiction - books.hardcoverNonfiction = 20 := by
  sorry

end NUMINAMATH_CALUDE_thabo_book_difference_l3708_370880


namespace NUMINAMATH_CALUDE_jessica_quarters_l3708_370845

/-- The number of quarters Jessica has after receiving quarters from her sister and friend. -/
def total_quarters (initial : ℕ) (from_sister : ℕ) (from_friend : ℕ) : ℕ :=
  initial + from_sister + from_friend

/-- Theorem stating that Jessica's total quarters is 16 given the initial amount and gifts. -/
theorem jessica_quarters : total_quarters 8 3 5 = 16 := by
  sorry

end NUMINAMATH_CALUDE_jessica_quarters_l3708_370845


namespace NUMINAMATH_CALUDE_cycle_original_price_l3708_370856

/-- Given a cycle sold at a 15% loss for Rs. 1190, prove that the original price was Rs. 1400 -/
theorem cycle_original_price (selling_price : ℝ) (loss_percentage : ℝ) 
  (h1 : selling_price = 1190)
  (h2 : loss_percentage = 15) : 
  ∃ (original_price : ℝ), 
    original_price * (1 - loss_percentage / 100) = selling_price ∧ 
    original_price = 1400 := by
  sorry

end NUMINAMATH_CALUDE_cycle_original_price_l3708_370856


namespace NUMINAMATH_CALUDE_garys_final_amount_l3708_370857

/-- Given Gary's initial amount and the amount he received from selling his snake, 
    calculate his final amount. -/
theorem garys_final_amount 
  (initial_amount : ℝ) 
  (snake_sale_amount : ℝ) 
  (h1 : initial_amount = 73.0) 
  (h2 : snake_sale_amount = 55.0) : 
  initial_amount + snake_sale_amount = 128.0 := by
  sorry

end NUMINAMATH_CALUDE_garys_final_amount_l3708_370857


namespace NUMINAMATH_CALUDE_divisibility_of_sum_and_powers_l3708_370808

theorem divisibility_of_sum_and_powers (a b c : ℤ) 
  (h : 6 ∣ (a + b + c)) : 6 ∣ (a^5 + b^3 + c) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_of_sum_and_powers_l3708_370808


namespace NUMINAMATH_CALUDE_range_f_characterization_l3708_370858

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x - 1

-- Define the range of f(x) on [0, 2]
def range_f (a : ℝ) : Set ℝ :=
  { y | ∃ x ∈ Set.Icc 0 2, f a x = y }

-- Theorem statement
theorem range_f_characterization (a : ℝ) :
  range_f a =
    if a < 0 then Set.Icc (-1) (3 - 4*a)
    else if a < 1 then Set.Icc (-1 - a^2) (3 - 4*a)
    else if a < 2 then Set.Icc (-1 - a^2) (-1)
    else Set.Icc (3 - 4*a) (-1) := by
  sorry

end NUMINAMATH_CALUDE_range_f_characterization_l3708_370858


namespace NUMINAMATH_CALUDE_det_special_matrix_is_zero_l3708_370877

open Real Matrix

theorem det_special_matrix_is_zero (θ φ : ℝ) : 
  det !![0, cos θ, sin θ; -cos θ, 0, cos φ; -sin θ, -cos φ, 0] = 0 := by
  sorry

end NUMINAMATH_CALUDE_det_special_matrix_is_zero_l3708_370877


namespace NUMINAMATH_CALUDE_inequality_implication_l3708_370807

theorem inequality_implication (a b : ℝ) : a^2 + 2*a*b + b^2 + a + b - 2 ≠ 0 → a + b ≠ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_implication_l3708_370807


namespace NUMINAMATH_CALUDE_min_value_theorem_l3708_370834

/-- Given a function y = a^(1-x) where a > 0 and a ≠ 1, 
    and a point A that lies on both the graph of the function and the line mx + ny - 1 = 0,
    where mn > 0, prove that the minimum value of 1/m + 2/n is 3 + 2√2. -/
theorem min_value_theorem (a : ℝ) (m n : ℝ) 
  (h1 : a > 0) (h2 : a ≠ 1) (h3 : m * n > 0) (h4 : m + n = 1) :
  (∀ m' n', m' * n' > 0 → m' + n' = 1 → 1 / m' + 2 / n' ≥ 3 + 2 * Real.sqrt 2) ∧
  (∃ m' n', m' * n' > 0 ∧ m' + n' = 1 ∧ 1 / m' + 2 / n' = 3 + 2 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3708_370834


namespace NUMINAMATH_CALUDE_last_digit_of_one_over_three_to_ten_l3708_370891

theorem last_digit_of_one_over_three_to_ten (n : ℕ) : 
  (1 : ℚ) / (3^10 : ℚ) * 10^n % 10 = 0 :=
by sorry

end NUMINAMATH_CALUDE_last_digit_of_one_over_three_to_ten_l3708_370891


namespace NUMINAMATH_CALUDE_trigonometric_expression_equality_trigonometric_fraction_simplification_l3708_370883

-- Part 1
theorem trigonometric_expression_equality : 
  2 * Real.cos (π / 2) + Real.tan (π / 4) + 3 * Real.sin 0 + (Real.cos (π / 3))^2 + Real.sin (3 * π / 2) = 1 / 4 := by
  sorry

-- Part 2
theorem trigonometric_fraction_simplification (θ : ℝ) : 
  (Real.sin (2 * π - θ) * Real.cos (π + θ) * Real.cos (π / 2 + θ) * Real.cos (11 * π / 2 - θ)) /
  (Real.cos (π - θ) * Real.sin (3 * π - θ) * Real.sin (-π - θ) * Real.sin (9 * π / 2 + θ)) = -Real.tan θ := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_expression_equality_trigonometric_fraction_simplification_l3708_370883


namespace NUMINAMATH_CALUDE_equation_holds_iff_l3708_370813

theorem equation_holds_iff (k y : ℝ) : 
  (∀ x : ℝ, -x^2 - (k+10)*x - 8 = -(x - 2)*(x - 4) + (y - 3)*(y - 6)) ↔ 
  (k = -16 ∧ False) :=
by sorry

end NUMINAMATH_CALUDE_equation_holds_iff_l3708_370813


namespace NUMINAMATH_CALUDE_correct_dye_jobs_scheduled_l3708_370898

def haircut_price : ℕ := 30
def perm_price : ℕ := 40
def dye_job_price : ℕ := 60
def dye_job_cost : ℕ := 10
def haircuts_scheduled : ℕ := 4
def perms_scheduled : ℕ := 1
def tips : ℕ := 50
def total_revenue : ℕ := 310

def dye_jobs_scheduled : ℕ := 
  (total_revenue - (haircut_price * haircuts_scheduled + perm_price * perms_scheduled + tips)) / 
  (dye_job_price - dye_job_cost)

theorem correct_dye_jobs_scheduled : dye_jobs_scheduled = 2 := by
  sorry

end NUMINAMATH_CALUDE_correct_dye_jobs_scheduled_l3708_370898


namespace NUMINAMATH_CALUDE_sequoia_maple_height_difference_l3708_370822

/-- Represents the height of a tree in feet and quarters of a foot -/
structure TreeHeight where
  feet : ℕ
  quarters : Fin 4

/-- Converts a TreeHeight to a rational number -/
def treeHeightToRational (h : TreeHeight) : ℚ :=
  h.feet + h.quarters.val / 4

/-- The height of the maple tree -/
def mapleHeight : TreeHeight := ⟨13, 3⟩

/-- The height of the sequoia -/
def sequoiaHeight : TreeHeight := ⟨20, 2⟩

theorem sequoia_maple_height_difference :
  treeHeightToRational sequoiaHeight - treeHeightToRational mapleHeight = 27 / 4 := by
  sorry

#eval treeHeightToRational sequoiaHeight - treeHeightToRational mapleHeight

end NUMINAMATH_CALUDE_sequoia_maple_height_difference_l3708_370822


namespace NUMINAMATH_CALUDE_john_experience_theorem_l3708_370854

/-- Represents the years of experience for each person -/
structure Experience where
  james : ℕ
  john : ℕ
  mike : ℕ

/-- The conditions of the problem -/
def problem_conditions (e : Experience) : Prop :=
  e.james = 20 ∧
  e.john - 8 = 2 * (e.james - 8) ∧
  e.james + e.john + e.mike = 68

/-- John's experience when Mike started -/
def john_experience_when_mike_started (e : Experience) : ℕ :=
  e.john - e.mike

/-- The theorem to prove -/
theorem john_experience_theorem (e : Experience) :
  problem_conditions e → john_experience_when_mike_started e = 16 := by
  sorry

end NUMINAMATH_CALUDE_john_experience_theorem_l3708_370854


namespace NUMINAMATH_CALUDE_prime_divides_sum_l3708_370862

theorem prime_divides_sum (a b c : ℕ+) (p : ℕ) 
  (h1 : a ^ 3 + 4 * b + c = a * b * c)
  (h2 : a ≥ c)
  (h3 : p = a ^ 2 + 2 * a + 2)
  (h4 : Nat.Prime p) :
  p ∣ (a + 2 * b + 2) := by
sorry

end NUMINAMATH_CALUDE_prime_divides_sum_l3708_370862


namespace NUMINAMATH_CALUDE_infinite_solutions_l3708_370811

theorem infinite_solutions (b : ℝ) :
  (∀ x, 5 * (4 * x - b) = 3 * (5 * x + 15)) ↔ b = -9 := by
  sorry

end NUMINAMATH_CALUDE_infinite_solutions_l3708_370811


namespace NUMINAMATH_CALUDE_success_permutations_l3708_370850

/-- The number of unique arrangements of letters in "SUCCESS" -/
def success_arrangements : ℕ := 420

/-- The total number of letters in "SUCCESS" -/
def total_letters : ℕ := 7

/-- The number of S's in "SUCCESS" -/
def num_s : ℕ := 3

/-- The number of C's in "SUCCESS" -/
def num_c : ℕ := 2

/-- The number of U's in "SUCCESS" -/
def num_u : ℕ := 1

/-- The number of E's in "SUCCESS" -/
def num_e : ℕ := 1

theorem success_permutations :
  success_arrangements = (Nat.factorial total_letters) / ((Nat.factorial num_s) * (Nat.factorial num_c)) :=
by sorry

end NUMINAMATH_CALUDE_success_permutations_l3708_370850


namespace NUMINAMATH_CALUDE_range_of_m_l3708_370824

-- Define a monotonically decreasing function
def MonoDecreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x > f y

-- State the theorem
theorem range_of_m (f : ℝ → ℝ) (m : ℝ) 
  (h1 : MonoDecreasing f) (h2 : f (2 * m) > f (1 + m)) : 
  m < 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l3708_370824


namespace NUMINAMATH_CALUDE_nurses_survey_result_l3708_370805

def total_nurses : ℕ := 150
def high_blood_pressure : ℕ := 90
def heart_trouble : ℕ := 50
def both_conditions : ℕ := 30

theorem nurses_survey_result : 
  (total_nurses - (high_blood_pressure + heart_trouble - both_conditions)) / total_nurses * 100 = 80 / 3 := by
  sorry

end NUMINAMATH_CALUDE_nurses_survey_result_l3708_370805


namespace NUMINAMATH_CALUDE_find_Y_l3708_370894

theorem find_Y : ∃ Y : ℝ, (100 + Y / 90) * 90 = 9020 ∧ Y = 20 := by
  sorry

end NUMINAMATH_CALUDE_find_Y_l3708_370894


namespace NUMINAMATH_CALUDE_quadratic_intersection_and_sum_of_y_l3708_370831

/-- Quadratic function -/
def f (a x : ℝ) : ℝ := a * x^2 - (2*a - 2) * x - 3*a - 1

theorem quadratic_intersection_and_sum_of_y (a : ℝ) (h1 : a > 0) :
  (∃! x, f a x = -3*a - 2) →
  a^2 + 1/a^2 = 7 ∧
  ∀ m n y1 y2 : ℝ, m ≠ n → m + n = -2 → 
    f a m = y1 → f a n = y2 → y1 + y2 > -6 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_intersection_and_sum_of_y_l3708_370831


namespace NUMINAMATH_CALUDE_dividend_calculation_l3708_370897

theorem dividend_calculation (divisor quotient remainder : ℕ) 
  (h1 : divisor = 16) 
  (h2 : quotient = 9) 
  (h3 : remainder = 5) : 
  divisor * quotient + remainder = 149 := by
sorry

end NUMINAMATH_CALUDE_dividend_calculation_l3708_370897


namespace NUMINAMATH_CALUDE_pentagon_largest_angle_l3708_370833

theorem pentagon_largest_angle (F G H I J : ℝ) : 
  F = 90 → 
  G = 70 → 
  H = I → 
  J = 2 * H + 20 → 
  F + G + H + I + J = 540 → 
  max F (max G (max H (max I J))) = 200 :=
by sorry

end NUMINAMATH_CALUDE_pentagon_largest_angle_l3708_370833


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l3708_370895

open Set

def M : Set ℝ := {x | (x + 2) * (x - 1) < 0}
def N : Set ℝ := {x | x + 1 < 0}

theorem intersection_of_M_and_N : M ∩ N = {x : ℝ | -2 < x ∧ x < -1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l3708_370895


namespace NUMINAMATH_CALUDE_exists_valid_custom_division_l3708_370806

/-- A custom division type that allows introducing additional 7s in intermediate calculations -/
structure CustomDivision where
  dividend : Nat
  divisor : Nat
  quotient : Nat
  intermediate_sevens : List Nat

/-- Checks if a number contains at least one 7 -/
def containsSeven (n : Nat) : Prop :=
  ∃ d, d ∈ n.digits 10 ∧ d = 7

/-- Theorem stating the existence of a valid custom division -/
theorem exists_valid_custom_division :
  ∃ (cd : CustomDivision),
    cd.dividend ≥ 1000000000 ∧ cd.dividend < 10000000000 ∧
    cd.divisor ≥ 100000 ∧ cd.divisor < 1000000 ∧
    cd.quotient ≥ 10000 ∧ cd.quotient < 100000 ∧
    containsSeven cd.dividend ∧
    containsSeven cd.divisor ∧
    cd.dividend = cd.divisor * cd.quotient :=
  sorry

#check exists_valid_custom_division

end NUMINAMATH_CALUDE_exists_valid_custom_division_l3708_370806


namespace NUMINAMATH_CALUDE_base_conversion_3050_l3708_370878

def base_10_to_base_8 (n : ℕ) : ℕ :=
  5000 + 700 + 50 + 2

theorem base_conversion_3050 :
  base_10_to_base_8 3050 = 5752 := by
  sorry

end NUMINAMATH_CALUDE_base_conversion_3050_l3708_370878


namespace NUMINAMATH_CALUDE_triangle_properties_l3708_370844

/-- Given a triangle ABC with circumradius R and satisfying the given equation,
    prove that angle C is π/3 and the maximum area is 3√3/2 -/
theorem triangle_properties (A B C : Real) (a b c : Real) (R : Real) :
  R = Real.sqrt 2 →
  2 * Real.sqrt 2 * (Real.sin A ^ 2 - Real.sin C ^ 2) = (a - b) * Real.sin B →
  (C = Real.pi / 3 ∧ 
   ∃ (S : Real), S = 3 * Real.sqrt 3 / 2 ∧ 
   ∀ (S' : Real), S' = 1/2 * a * b * Real.sin C → S' ≤ S) :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l3708_370844


namespace NUMINAMATH_CALUDE_collinear_probability_4x5_l3708_370814

/-- Represents a grid of dots -/
structure Grid :=
  (rows : ℕ)
  (cols : ℕ)

/-- Calculates the number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := sorry

/-- Counts the number of sets of 4 collinear dots in a grid -/
def collinearSets (g : Grid) : ℕ := sorry

/-- The probability of selecting 4 collinear dots from a grid -/
def collinearProbability (g : Grid) : ℚ :=
  (collinearSets g : ℚ) / choose (g.rows * g.cols) 4

theorem collinear_probability_4x5 :
  let g : Grid := ⟨4, 5⟩
  collinearProbability g = 9 / 4845 := by sorry

end NUMINAMATH_CALUDE_collinear_probability_4x5_l3708_370814


namespace NUMINAMATH_CALUDE_choose_four_different_suits_standard_deck_l3708_370848

/-- Represents a standard deck of cards -/
structure Deck :=
  (cards : Nat)
  (suits : Nat)
  (cards_per_suit : Nat)
  (h1 : cards = suits * cards_per_suit)

/-- The number of ways to choose 4 cards from different suits in a standard deck -/
def choose_four_different_suits (d : Deck) : Nat :=
  d.cards_per_suit ^ d.suits

/-- Theorem stating that the number of ways to choose 4 cards from different suits
    in a standard deck of 52 cards is 28,561 -/
theorem choose_four_different_suits_standard_deck :
  ∃ (d : Deck), d.cards = 52 ∧ d.suits = 4 ∧ d.cards_per_suit = 13 ∧
  choose_four_different_suits d = 28561 :=
sorry

end NUMINAMATH_CALUDE_choose_four_different_suits_standard_deck_l3708_370848


namespace NUMINAMATH_CALUDE_ellen_initial_legos_l3708_370875

/-- The number of Legos Ellen lost -/
def lost_legos : ℕ := 17

/-- The number of Legos Ellen currently has -/
def current_legos : ℕ := 2063

/-- The initial number of Legos Ellen had -/
def initial_legos : ℕ := current_legos + lost_legos

theorem ellen_initial_legos : initial_legos = 2080 := by
  sorry

end NUMINAMATH_CALUDE_ellen_initial_legos_l3708_370875


namespace NUMINAMATH_CALUDE_addition_subtraction_elimination_not_factorization_l3708_370849

-- Define the type for factorization methods
inductive FactorizationMethod
  | TakeOutCommonFactor
  | CrossMultiplication
  | Formula
  | AdditionSubtractionElimination

-- Define a predicate to check if a method is a factorization method
def is_factorization_method : FactorizationMethod → Prop
  | FactorizationMethod.TakeOutCommonFactor => true
  | FactorizationMethod.CrossMultiplication => true
  | FactorizationMethod.Formula => true
  | FactorizationMethod.AdditionSubtractionElimination => false

-- Theorem statement
theorem addition_subtraction_elimination_not_factorization :
  ¬(is_factorization_method FactorizationMethod.AdditionSubtractionElimination) :=
by sorry

end NUMINAMATH_CALUDE_addition_subtraction_elimination_not_factorization_l3708_370849
