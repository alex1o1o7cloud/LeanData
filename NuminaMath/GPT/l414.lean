import Mathlib

namespace NUMINAMATH_GPT_gwen_money_remaining_l414_41458

def gwen_money (initial : ℝ) (spent1 : ℝ) (earned : ℝ) (spent2 : ℝ) : ℝ :=
  initial - spent1 + earned - spent2

theorem gwen_money_remaining :
  gwen_money 5 3.25 1.5 0.7 = 2.55 :=
by
  sorry

end NUMINAMATH_GPT_gwen_money_remaining_l414_41458


namespace NUMINAMATH_GPT_bus_stops_for_minutes_per_hour_l414_41434

theorem bus_stops_for_minutes_per_hour (speed_no_stops speed_with_stops : ℕ)
  (h1 : speed_no_stops = 60) (h2 : speed_with_stops = 45) : 
  (60 * (speed_no_stops - speed_with_stops) / speed_no_stops) = 15 :=
by
  sorry

end NUMINAMATH_GPT_bus_stops_for_minutes_per_hour_l414_41434


namespace NUMINAMATH_GPT_otimes_2_3_eq_23_l414_41423

-- Define the new operation
def otimes (a b : ℝ) : ℝ := 4 * a + 5 * b

-- The proof statement
theorem otimes_2_3_eq_23 : otimes 2 3 = 23 := 
  by 
  sorry

end NUMINAMATH_GPT_otimes_2_3_eq_23_l414_41423


namespace NUMINAMATH_GPT_will_3_point_shots_l414_41482

theorem will_3_point_shots :
  ∃ x y : ℕ, 3 * x + 2 * y = 26 ∧ x + y = 11 ∧ x = 4 :=
by
  sorry

end NUMINAMATH_GPT_will_3_point_shots_l414_41482


namespace NUMINAMATH_GPT_smallest_circle_area_l414_41424

noncomputable def function_y (x : ℝ) : ℝ := 6 / x - 4 * x / 3

theorem smallest_circle_area :
  ∃ r : ℝ, (∀ x : ℝ, r * r = x^2 + (function_y x)^2) → r^2 * π = 4 * π :=
sorry

end NUMINAMATH_GPT_smallest_circle_area_l414_41424


namespace NUMINAMATH_GPT_four_digit_numbers_with_8_or_3_l414_41468

theorem four_digit_numbers_with_8_or_3 :
  let total_four_digit_numbers := 9000
  let without_8_or_3_first := 7
  let without_8_or_3_rest := 8
  let numbers_without_8_or_3 := without_8_or_3_first * without_8_or_3_rest^3
  total_four_digit_numbers - numbers_without_8_or_3 = 5416 :=
by
  let total_four_digit_numbers := 9000
  let without_8_or_3_first := 7
  let without_8_or_3_rest := 8
  let numbers_without_8_or_3 := without_8_or_3_first * without_8_or_3_rest^3
  sorry

end NUMINAMATH_GPT_four_digit_numbers_with_8_or_3_l414_41468


namespace NUMINAMATH_GPT_determine_x_l414_41449

theorem determine_x : ∃ (x : ℕ), 
  (3 * x > 91 ∧ x < 120 ∧ 4 * x > 37 ∧ 2 * x ≥ 21 ∧ ¬(x > 7) ∧ 
   (3 * x > 91 ∨ x < 120 ∨ 4 * x > 37 ∨ 2 * x ≥ 21 ∨ x > 7)) ∨
  (3 * x > 91 ∧ x < 120 ∧ 4 * x > 37 ∧ ¬(2 * x ≥ 21) ∧ x > 7 ∧ 
   (3 * x > 91 ∨ x < 120 ∨ 4 * x > 37 ∨ 2 * x ≥ 21 ∨ x > 7)) ∨
  (3 * x > 91 ∧ x < 120 ∧ ¬(4 * x > 37) ∧ 2 * x ≥ 21 ∧ x > 7 ∧ 
   (3 * x > 91 ∨ x < 120 ∨ 4 * x > 37 ∨ 2 * x ≥ 21 ∨ x > 7)) ∨
  (3 * x > 91 ∧ ¬(x < 120) ∧ 4 * x > 37 ∧ 2 * x ≥ 21 ∧ x > 7 ∧ 
   (3 * x > 91 ∨ x < 120 ∨ 4 * x > 37 ∨ 2 * x ≥ 21 ∨ x > 7)) ∨
  (¬(3 * x > 91) ∧ x < 120 ∧ 4 * x > 37 ∧ 2 * x ≥ 21 ∧ x > 7 ∧ 
   (3 * x > 91 ∨ x < 120 ∨ 4 * x > 37 ∨ 2 * x ≥ 21 ∨ x > 7)) ∧
  x = 10 :=
sorry

end NUMINAMATH_GPT_determine_x_l414_41449


namespace NUMINAMATH_GPT_extremum_range_k_l414_41478

noncomputable def f (x k : Real) : Real :=
  Real.exp x / x + k * (Real.log x - x)

/-- 
For the function f(x) = (exp(x) / x) + k * (log(x) - x), if x = 1 is the only extremum point, 
then k is in the interval (-∞, e].
-/
theorem extremum_range_k (k : Real) : 
  (∀ x : Real, (0 < x) → (f x k ≤ f 1 k)) → 
  k ≤ Real.exp 1 :=
sorry

end NUMINAMATH_GPT_extremum_range_k_l414_41478


namespace NUMINAMATH_GPT_center_of_symmetry_l414_41464

theorem center_of_symmetry (k : ℤ) : ∀ (k : ℤ), ∃ x : ℝ, 
  (x = (k * Real.pi / 6 - Real.pi / 9) ∨ x = - (Real.pi / 18)) → False :=
by
  sorry

end NUMINAMATH_GPT_center_of_symmetry_l414_41464


namespace NUMINAMATH_GPT_snail_returns_l414_41481

noncomputable def snail_path : Type := ℕ → ℝ × ℝ

def snail_condition (snail : snail_path) (speed : ℝ) : Prop :=
  ∀ n : ℕ, n % 4 = 0 → snail (n + 4) = snail n

theorem snail_returns (snail : snail_path) (speed : ℝ) (h1 : ∀ n m : ℕ, n ≠ m → snail n ≠ snail m)
    (h2 : snail_condition snail speed) :
  ∃ t : ℕ, t > 0 ∧ t % 4 = 0 ∧ snail t = snail 0 := 
sorry

end NUMINAMATH_GPT_snail_returns_l414_41481


namespace NUMINAMATH_GPT_minimum_balls_same_color_minimum_balls_two_white_l414_41443

-- Define the number of black and white balls.
def num_black_balls : Nat := 100
def num_white_balls : Nat := 100

-- Problem 1: Ensure at least 2 balls of the same color.
theorem minimum_balls_same_color (n_black n_white : Nat) (h_black : n_black = num_black_balls) (h_white : n_white = num_white_balls) : 
  3 ≥ 2 :=
by
  sorry

-- Problem 2: Ensure at least 2 white balls.
theorem minimum_balls_two_white (n_black n_white : Nat) (h_black: n_black = num_black_balls) (h_white: n_white = num_white_balls) :
  102 ≥ 2 :=
by
  sorry

end NUMINAMATH_GPT_minimum_balls_same_color_minimum_balls_two_white_l414_41443


namespace NUMINAMATH_GPT_negation_correct_l414_41444

theorem negation_correct (x : ℝ) : -(3 * x - 2) = -3 * x + 2 := 
by sorry

end NUMINAMATH_GPT_negation_correct_l414_41444


namespace NUMINAMATH_GPT_train_length_calculation_l414_41440

def speed_km_per_hr : ℝ := 60
def time_sec : ℝ := 9
def length_of_train : ℝ := 150

theorem train_length_calculation :
  (speed_km_per_hr * 1000 / 3600) * time_sec = length_of_train := by
  sorry

end NUMINAMATH_GPT_train_length_calculation_l414_41440


namespace NUMINAMATH_GPT_part1_part2_l414_41412

noncomputable def f (x a : ℝ) : ℝ := |x - 1| - 2 * |x + a|
noncomputable def g (x b : ℝ) : ℝ := 0.5 * x + b

theorem part1 (a : ℝ) (h : a = 1/2) : 
  { x : ℝ | f x a ≤ 0 } = { x : ℝ | x ≤ -2 ∨ x ≥ 0 } :=
sorry

theorem part2 (a b : ℝ) (h1 : a ≥ -1) (h2 : ∀ x, g x b ≥ f x a) : 
  2 * b - 3 * a > 2 :=
sorry

end NUMINAMATH_GPT_part1_part2_l414_41412


namespace NUMINAMATH_GPT_size_of_smaller_package_l414_41430

theorem size_of_smaller_package
  (total_coffee : ℕ)
  (n_ten_ounce_packages : ℕ)
  (extra_five_ounce_packages : ℕ)
  (size_smaller_package : ℕ)
  (h1 : total_coffee = 115)
  (h2 : size_smaller_package = 5)
  (h3 : n_ten_ounce_packages = 7)
  (h4 : extra_five_ounce_packages = 2)
  (h5 : total_coffee = n_ten_ounce_packages * 10 + (n_ten_ounce_packages + extra_five_ounce_packages) * size_smaller_package) :
  size_smaller_package = 5 :=
by 
  sorry

end NUMINAMATH_GPT_size_of_smaller_package_l414_41430


namespace NUMINAMATH_GPT_minimum_value_expression_l414_41472

theorem minimum_value_expression (a b : ℝ) (h1 : a > 0) (h2 : b > 0) :
  ( (3*a*b - 6*b + a*(1-a))^2 + (9*b^2 + 2*a + 3*b*(1-a))^2 ) / (a^2 + 9*b^2) ≥ 4 :=
sorry

end NUMINAMATH_GPT_minimum_value_expression_l414_41472


namespace NUMINAMATH_GPT_number_of_littering_citations_l414_41494

variable (L D P : ℕ)
variable (h1 : L = D)
variable (h2 : P = 2 * (L + D))
variable (h3 : L + D + P = 24)

theorem number_of_littering_citations : L = 4 :=
by
  sorry

end NUMINAMATH_GPT_number_of_littering_citations_l414_41494


namespace NUMINAMATH_GPT_emily_sixth_score_l414_41432

theorem emily_sixth_score:
  ∀ (s₁ s₂ s₃ s₄ s₅ sᵣ : ℕ),
  s₁ = 88 →
  s₂ = 90 →
  s₃ = 85 →
  s₄ = 92 →
  s₅ = 97 →
  (s₁ + s₂ + s₃ + s₄ + s₅ + sᵣ) / 6 = 91 →
  sᵣ = 94 :=
by intros s₁ s₂ s₃ s₄ s₅ sᵣ h₁ h₂ h₃ h₄ h₅ h₆;
   rw [h₁, h₂, h₃, h₄, h₅] at h₆;
   sorry

end NUMINAMATH_GPT_emily_sixth_score_l414_41432


namespace NUMINAMATH_GPT_probability_defective_unit_l414_41495

theorem probability_defective_unit 
  (T : ℝ)
  (machine_a_output : ℝ := 0.4 * T)
  (machine_b_output : ℝ := 0.6 * T)
  (machine_a_defective_rate : ℝ := 9 / 1000)
  (machine_b_defective_rate : ℝ := 1 / 50)
  (total_defective_units : ℝ := (machine_a_output * machine_a_defective_rate) + (machine_b_output * machine_b_defective_rate))
  (probability_defective : ℝ := total_defective_units / T) :
  probability_defective = 0.0156 :=
by
  sorry

end NUMINAMATH_GPT_probability_defective_unit_l414_41495


namespace NUMINAMATH_GPT_triangle_other_side_length_l414_41457

theorem triangle_other_side_length (a b : ℝ) (c : ℝ) (h_a : a = 3) (h_b : b = 4) (h_right_angle : c * c = a * a + b * b ∨ a * a = c * c + b * b):
  c = Real.sqrt 7 ∨ c = 5 :=
by
  sorry

end NUMINAMATH_GPT_triangle_other_side_length_l414_41457


namespace NUMINAMATH_GPT_triangle_is_right_triangle_l414_41419

theorem triangle_is_right_triangle (a b c : ℕ) (h_ratio : a = 3 * (36 / 12)) (h_perimeter : 3 * (36 / 12) + 4 * (36 / 12) + 5 * (36 / 12) = 36) :
  a^2 + b^2 = c^2 :=
by
  -- sorry for skipping the proof.
  sorry

end NUMINAMATH_GPT_triangle_is_right_triangle_l414_41419


namespace NUMINAMATH_GPT_add_solution_y_to_solution_x_l414_41437

theorem add_solution_y_to_solution_x
  (x_volume : ℝ) (x_percent : ℝ) (y_percent : ℝ) (desired_percent : ℝ) (final_volume : ℝ)
  (x_alcohol : ℝ := x_volume * x_percent / 100) (y : ℝ := final_volume - x_volume) :
  (x_percent = 10) → (y_percent = 30) → (desired_percent = 15) → (x_volume = 300) →
  (final_volume = 300 + y) →
  ((x_alcohol + y * y_percent / 100) / final_volume = desired_percent / 100) →
  y = 100 := by
    intros h1 h2 h3 h4 h5 h6
    sorry

end NUMINAMATH_GPT_add_solution_y_to_solution_x_l414_41437


namespace NUMINAMATH_GPT_students_answered_both_correctly_l414_41448

theorem students_answered_both_correctly
  (enrolled : ℕ)
  (did_not_take_test : ℕ)
  (answered_q1_correctly : ℕ)
  (answered_q2_correctly : ℕ)
  (total_students_answered_both : ℕ) :
  enrolled = 29 →
  did_not_take_test = 5 →
  answered_q1_correctly = 19 →
  answered_q2_correctly = 24 →
  total_students_answered_both = 19 :=
by
  intros
  sorry

end NUMINAMATH_GPT_students_answered_both_correctly_l414_41448


namespace NUMINAMATH_GPT_removed_term_sequence_l414_41403

theorem removed_term_sequence (S : ℕ → ℤ) (a : ℕ → ℤ) (k : ℕ) :
  (∀ n, S n = 2 * n^2 - n) →
  (∀ n, n ≥ 2 → a n = S n - S (n-1)) →
  (S 21 - a k = 40 * 20) →
  a k = 4 * k - 3 →
  k = 16 :=
by
  intros hs ha h_avg h_ak
  sorry

end NUMINAMATH_GPT_removed_term_sequence_l414_41403


namespace NUMINAMATH_GPT_car_distance_l414_41484

theorem car_distance (t : ℚ) (s : ℚ) (d : ℚ) 
(h1 : t = 2 + 2 / 5) 
(h2 : s = 260) 
(h3 : d = s * t) : 
d = 624 := by
  sorry

end NUMINAMATH_GPT_car_distance_l414_41484


namespace NUMINAMATH_GPT_markup_is_correct_l414_41475

def purchase_price : ℝ := 48
def overhead_percent : ℝ := 0.25
def net_profit : ℝ := 12

def overhead_cost := overhead_percent * purchase_price
def total_cost := purchase_price + overhead_cost
def selling_price := total_cost + net_profit
def markup := selling_price - purchase_price

theorem markup_is_correct : markup = 24 := by sorry

end NUMINAMATH_GPT_markup_is_correct_l414_41475


namespace NUMINAMATH_GPT_marble_ratio_l414_41451

-- Let Allison, Angela, and Albert have some number of marbles denoted by variables.
variable (Albert Angela Allison : ℕ)

-- Given conditions.
axiom h1 : Angela = Allison + 8
axiom h2 : Allison = 28
axiom h3 : Albert + Allison = 136

-- Prove that the ratio of the number of marbles Albert has to the number of marbles Angela has is 3.
theorem marble_ratio : Albert / Angela = 3 := by
  sorry

end NUMINAMATH_GPT_marble_ratio_l414_41451


namespace NUMINAMATH_GPT_inverse_of_B_squared_l414_41456

noncomputable def B_inv : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![2, -3, 0], ![0, -1, 0], ![0, 0, 5]]

theorem inverse_of_B_squared :
  (B_inv * B_inv) = ![![4, -3, 0], ![0, 1, 0], ![0, 0, 25]] := by
  sorry

end NUMINAMATH_GPT_inverse_of_B_squared_l414_41456


namespace NUMINAMATH_GPT_ratio_of_square_sides_l414_41466

theorem ratio_of_square_sides
  (a b : ℝ) 
  (h1 : ∃ square1 : ℝ, square1 = 2 * a)
  (h2 : ∃ square2 : ℝ, square2 = 2 * b)
  (h3 : a ^ 2 - 4 * a * b - 5 * b ^ 2 = 0) :
  2 * a / 2 * b = 5 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_square_sides_l414_41466


namespace NUMINAMATH_GPT_minimum_value_of_f_l414_41477

noncomputable def f (x : ℝ) : ℝ := 2 * x + (3 * x) / (x^2 + 3) + (2 * x * (x + 5)) / (x^2 + 5) + (3 * (x + 3)) / (x * (x^2 + 5))

theorem minimum_value_of_f : ∃ a : ℝ, a > 0 ∧ (∀ x > 0, f x ≥ 7) ∧ (f a = 7) :=
by
  sorry

end NUMINAMATH_GPT_minimum_value_of_f_l414_41477


namespace NUMINAMATH_GPT_quadratic_discriminant_l414_41490

-- Define the quadratic equation coefficients
def a : ℤ := 5
def b : ℤ := -11
def c : ℤ := 2

-- Define the discriminant of the quadratic equation
def discriminant (a b c : ℤ) : ℤ := b^2 - 4 * a * c

-- assert the discriminant for given coefficients
theorem quadratic_discriminant : discriminant a b c = 81 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_discriminant_l414_41490


namespace NUMINAMATH_GPT_find_product_of_roots_l414_41498

namespace ProductRoots

variables {k m : ℝ} {x1 x2 : ℝ}

theorem find_product_of_roots (h1 : x1 ≠ x2) 
    (hx1 : 5 * x1 ^ 2 - k * x1 = m) 
    (hx2 : 5 * x2 ^ 2 - k * x2 = m) : x1 * x2 = -m / 5 :=
sorry

end ProductRoots

end NUMINAMATH_GPT_find_product_of_roots_l414_41498


namespace NUMINAMATH_GPT_otimes_evaluation_l414_41493

def otimes (a b : ℝ) : ℝ := a * b + a - b

theorem otimes_evaluation (a b : ℝ) : 
  otimes a b + otimes (b - a) b = b^2 - b := 
  by
  sorry

end NUMINAMATH_GPT_otimes_evaluation_l414_41493


namespace NUMINAMATH_GPT_fred_earnings_l414_41465
noncomputable def start := 111
noncomputable def now := 115
noncomputable def earnings := now - start

theorem fred_earnings : earnings = 4 :=
by
  sorry

end NUMINAMATH_GPT_fred_earnings_l414_41465


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_l414_41460

variable (x y : ℝ)

theorem sufficient_but_not_necessary (x_gt_y_gt_zero : x > y ∧ y > 0) : (x / y > 1) :=
by
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_l414_41460


namespace NUMINAMATH_GPT_distance_from_point_to_x_axis_l414_41406

def distance_to_x_axis (p : ℝ × ℝ) : ℝ :=
  |p.2|

theorem distance_from_point_to_x_axis :
  let p := (-2, -Real.sqrt 5)
  distance_to_x_axis p = Real.sqrt 5 := by
  sorry

end NUMINAMATH_GPT_distance_from_point_to_x_axis_l414_41406


namespace NUMINAMATH_GPT_geom_sequence_general_formula_l414_41427

theorem geom_sequence_general_formula :
  ∃ (a : ℕ → ℝ) (a₁ q : ℝ), 
  (∀ n, a n = a₁ * q ^ n ∧ abs (q) < 1 ∧ ∑' i, a i = 3 ∧ ∑' i, (a i)^2 = (9 / 2)) →
  (∀ n, a n = 2 * ((1 / 3) ^ (n - 1))) :=
by sorry

end NUMINAMATH_GPT_geom_sequence_general_formula_l414_41427


namespace NUMINAMATH_GPT_at_least_one_alarm_rings_on_time_l414_41474

-- Definitions for the problem
def prob_A : ℝ := 0.5
def prob_B : ℝ := 0.6

def prob_not_A : ℝ := 1 - prob_A
def prob_not_B : ℝ := 1 - prob_B
def prob_neither_A_nor_B : ℝ := prob_not_A * prob_not_B
def prob_at_least_one : ℝ := 1 - prob_neither_A_nor_B

-- Final statement
theorem at_least_one_alarm_rings_on_time : prob_at_least_one = 0.8 :=
by sorry

end NUMINAMATH_GPT_at_least_one_alarm_rings_on_time_l414_41474


namespace NUMINAMATH_GPT_find_a_c_l414_41441

theorem find_a_c (a c : ℝ) (h1 : a + c = 35) (h2 : a < c)
  (h3 : ∀ x : ℝ, a * x^2 + 30 * x + c = 0 → ∃! x, a * x^2 + 30 * x + c = 0) :
  (a = (35 - 5 * Real.sqrt 13) / 2 ∧ c = (35 + 5 * Real.sqrt 13) / 2) :=
by
  sorry

end NUMINAMATH_GPT_find_a_c_l414_41441


namespace NUMINAMATH_GPT_part1_part2_l414_41408
open Real

def f (x : ℝ) : ℝ := abs (x + 3) + abs (x - 2)

theorem part1 : {x : ℝ | f x > 7} = {x : ℝ | x < -4} ∪ {x : ℝ | x > 3} :=
by
  sorry

theorem part2 (m : ℝ) (hm : m > 1) : ∃ x : ℝ, f x = 4 / (m - 1) + m :=
by
  sorry

end NUMINAMATH_GPT_part1_part2_l414_41408


namespace NUMINAMATH_GPT_topsoil_cost_correct_l414_41414

noncomputable def topsoilCost (price_per_cubic_foot : ℝ) (yard_to_foot : ℝ) (discount_threshold : ℝ) (discount_rate : ℝ) (volume_in_yards : ℝ) : ℝ :=
  let volume_in_feet := volume_in_yards * yard_to_foot
  let cost_without_discount := volume_in_feet * price_per_cubic_foot
  if volume_in_feet > discount_threshold then
    cost_without_discount * (1 - discount_rate)
  else
    cost_without_discount

theorem topsoil_cost_correct:
  topsoilCost 8 27 100 0.10 7 = 1360.8 :=
by
  sorry

end NUMINAMATH_GPT_topsoil_cost_correct_l414_41414


namespace NUMINAMATH_GPT_total_cost_is_72_l414_41445

-- Definitions based on conditions
def adults (total_people : ℕ) (kids : ℕ) : ℕ := total_people - kids
def cost_per_adult_meal (cost_per_meal : ℕ) (adults : ℕ) : ℕ := cost_per_meal * adults
def total_cost (total_people : ℕ) (kids : ℕ) (cost_per_meal : ℕ) : ℕ := 
  cost_per_adult_meal cost_per_meal (adults total_people kids)

-- Given values
def total_people := 11
def kids := 2
def cost_per_meal := 8

-- Theorem statement
theorem total_cost_is_72 : total_cost total_people kids cost_per_meal = 72 := by
  sorry

end NUMINAMATH_GPT_total_cost_is_72_l414_41445


namespace NUMINAMATH_GPT_point_of_tangency_is_correct_l414_41416

theorem point_of_tangency_is_correct : 
  (∃ (x y : ℝ), y = x^2 + 20 * x + 63 ∧ x = y^2 + 56 * y + 875 ∧ x = -19 / 2 ∧ y = -55 / 2) :=
by
  sorry

end NUMINAMATH_GPT_point_of_tangency_is_correct_l414_41416


namespace NUMINAMATH_GPT_midpoint_trajectory_of_intersecting_line_l414_41471

theorem midpoint_trajectory_of_intersecting_line 
    (h₁ : ∀ x y, x^2 + 2 * y^2 = 4) 
    (h₂ : ∀ M: ℝ × ℝ, M = (4, 6)) :
    ∃ x y, (x-2)^2 / 22 + (y-3)^2 / 11 = 1 :=
sorry

end NUMINAMATH_GPT_midpoint_trajectory_of_intersecting_line_l414_41471


namespace NUMINAMATH_GPT_elsa_emma_spending_ratio_l414_41420

theorem elsa_emma_spending_ratio
  (E : ℝ)
  (h_emma : ∃ (x : ℝ), x = 58)
  (h_elizabeth : ∃ (y : ℝ), y = 4 * E)
  (h_total : 58 + E + 4 * E = 638) :
  E / 58 = 2 :=
by
  sorry

end NUMINAMATH_GPT_elsa_emma_spending_ratio_l414_41420


namespace NUMINAMATH_GPT_calculate_average_age_l414_41436

variables (k : ℕ) (female_to_male_ratio : ℚ) (avg_young_female : ℚ) (avg_old_female : ℚ) (avg_young_male : ℚ) (avg_old_male : ℚ)

theorem calculate_average_age 
  (h_ratio : female_to_male_ratio = 7/8)
  (h_avg_yf : avg_young_female = 26)
  (h_avg_of : avg_old_female = 42)
  (h_avg_ym : avg_young_male = 28)
  (h_avg_om : avg_old_male = 46) : 
  (534/15 : ℚ) = 36 :=
by sorry

end NUMINAMATH_GPT_calculate_average_age_l414_41436


namespace NUMINAMATH_GPT_eight_bags_weight_l414_41442

theorem eight_bags_weight
  (bags_weight : ℕ → ℕ)
  (h1 : bags_weight 12 = 24) :
  bags_weight 8 = 16 :=
  sorry

end NUMINAMATH_GPT_eight_bags_weight_l414_41442


namespace NUMINAMATH_GPT_angle_triple_complement_l414_41454

theorem angle_triple_complement (x : ℝ) (h : x = 3 * (90 - x)) : x = 67.5 := by
  sorry

end NUMINAMATH_GPT_angle_triple_complement_l414_41454


namespace NUMINAMATH_GPT_first_operation_result_l414_41447

def pattern (x y : ℕ) : ℕ :=
  if (x, y) = (3, 7) then 27
  else if (x, y) = (4, 5) then 32
  else if (x, y) = (5, 8) then 60
  else if (x, y) = (6, 7) then 72
  else if (x, y) = (7, 8) then 98
  else 26

theorem first_operation_result : pattern 2 3 = 26 := by
  sorry

end NUMINAMATH_GPT_first_operation_result_l414_41447


namespace NUMINAMATH_GPT_sum_abc_eq_neg_ten_thirds_l414_41459

variable (a b c d y : ℝ)

-- Define the conditions
def condition_1 : Prop := a + 2 = y
def condition_2 : Prop := b + 3 = y
def condition_3 : Prop := c + 4 = y
def condition_4 : Prop := d + 5 = y
def condition_5 : Prop := a + b + c + d + 6 = y

-- State the theorem
theorem sum_abc_eq_neg_ten_thirds
    (h1 : condition_1 a y)
    (h2 : condition_2 b y)
    (h3 : condition_3 c y)
    (h4 : condition_4 d y)
    (h5 : condition_5 a b c d y) :
    a + b + c + d = -10 / 3 :=
sorry

end NUMINAMATH_GPT_sum_abc_eq_neg_ten_thirds_l414_41459


namespace NUMINAMATH_GPT_weight_of_new_student_l414_41499

-- Definitions from conditions
def total_weight_19 : ℝ := 19 * 15
def total_weight_20 : ℝ := 20 * 14.9

-- Theorem to prove the weight of the new student
theorem weight_of_new_student : (total_weight_20 - total_weight_19) = 13 := by
  sorry

end NUMINAMATH_GPT_weight_of_new_student_l414_41499


namespace NUMINAMATH_GPT_correct_method_eliminates_y_l414_41469

def eliminate_y_condition1 (x y : ℝ) : Prop :=
  5 * x + 2 * y = 20

def eliminate_y_condition2 (x y : ℝ) : Prop :=
  4 * x - y = 8

theorem correct_method_eliminates_y (x y : ℝ) :
  eliminate_y_condition1 x y ∧ eliminate_y_condition2 x y →
  5 * x + 2 * y + 2 * (4 * x - y) = 36 :=
by
  sorry

end NUMINAMATH_GPT_correct_method_eliminates_y_l414_41469


namespace NUMINAMATH_GPT_trivia_team_total_points_l414_41426

/-- Given the points scored by the 5 members who showed up in a trivia team game,
    prove that the total points scored by the team is 29. -/
theorem trivia_team_total_points 
  (points_first : ℕ := 5) 
  (points_second : ℕ := 9) 
  (points_third : ℕ := 7) 
  (points_fourth : ℕ := 5) 
  (points_fifth : ℕ := 3) 
  (total_points : ℕ := points_first + points_second + points_third + points_fourth + points_fifth) :
  total_points = 29 :=
by
  sorry

end NUMINAMATH_GPT_trivia_team_total_points_l414_41426


namespace NUMINAMATH_GPT_distance_between_Stockholm_and_Malmoe_l414_41496

noncomputable def actualDistanceGivenMapDistanceAndScale (mapDistance : ℕ) (scale : ℕ) : ℕ :=
  mapDistance * scale

theorem distance_between_Stockholm_and_Malmoe (mapDistance : ℕ) (scale : ℕ) :
  mapDistance = 150 → scale = 20 → actualDistanceGivenMapDistanceAndScale mapDistance scale = 3000 :=
by
  intros h1 h2
  rw [h1, h2]
  sorry

end NUMINAMATH_GPT_distance_between_Stockholm_and_Malmoe_l414_41496


namespace NUMINAMATH_GPT_probability_excellent_probability_good_or_better_l414_41429

noncomputable def total_selections : ℕ := 10
noncomputable def total_excellent_selections : ℕ := 1
noncomputable def total_good_or_better_selections : ℕ := 7
noncomputable def P_excellent : ℚ := 1 / 10
noncomputable def P_good_or_better : ℚ := 7 / 10

theorem probability_excellent (total_selections total_excellent_selections : ℕ) :
  (total_excellent_selections : ℚ) / total_selections = 1 / 10 := by
  sorry

theorem probability_good_or_better (total_selections total_good_or_better_selections : ℕ) :
  (total_good_or_better_selections : ℚ) / total_selections = 7 / 10 := by
  sorry

end NUMINAMATH_GPT_probability_excellent_probability_good_or_better_l414_41429


namespace NUMINAMATH_GPT_max_area_of_backyard_l414_41486

theorem max_area_of_backyard (fence_length : ℕ) (h1 : fence_length = 500) 
  (l w : ℕ) (h2 : l = 2 * w) (h3 : l + 2 * w = fence_length) : 
  l * w = 31250 := 
by
  sorry

end NUMINAMATH_GPT_max_area_of_backyard_l414_41486


namespace NUMINAMATH_GPT_pies_sold_in_a_week_l414_41491

theorem pies_sold_in_a_week : 
  let Monday := 8
  let Tuesday := 12
  let Wednesday := 14
  let Thursday := 20
  let Friday := 20
  let Saturday := 20
  let Sunday := 20
  Monday + Tuesday + Wednesday + Thursday + Friday + Saturday + Sunday = 114 :=
by 
  let Monday := 8
  let Tuesday := 12
  let Wednesday := 14
  let Thursday := 20
  let Friday := 20
  let Saturday := 20
  let Sunday := 20
  have h1 : Monday + Tuesday + Wednesday + Thursday + Friday + Saturday + Sunday = 8 + 12 + 14 + 20 + 20 + 20 + 20 := by rfl
  have h2 : 8 + 12 + 14 + 20 + 20 + 20 + 20 = 114 := by norm_num
  exact h1.trans h2

end NUMINAMATH_GPT_pies_sold_in_a_week_l414_41491


namespace NUMINAMATH_GPT_min_dominos_in_2x2_l414_41455

/-- A 100 × 100 square is divided into 2 × 2 squares.
Then it is divided into dominos (rectangles 1 × 2 and 2 × 1).
Prove that the minimum number of dominos within the 2 × 2 squares is 100. -/
theorem min_dominos_in_2x2 (N : ℕ) (hN : N = 100) :
  ∃ d : ℕ, d = 100 :=
sorry

end NUMINAMATH_GPT_min_dominos_in_2x2_l414_41455


namespace NUMINAMATH_GPT_fraction_identity_l414_41401

theorem fraction_identity (a b : ℝ) (hb : b ≠ 0) (h : a / b = 3 / 2) : (a + b) / b = 2.5 :=
by
  sorry

end NUMINAMATH_GPT_fraction_identity_l414_41401


namespace NUMINAMATH_GPT_more_stable_scores_l414_41411

-- Define the variances for Student A and Student B
def variance_A : ℝ := 38
def variance_B : ℝ := 15

-- Formulate the theorem
theorem more_stable_scores : variance_A > variance_B → "B" = "B" :=
by
  intro h
  sorry

end NUMINAMATH_GPT_more_stable_scores_l414_41411


namespace NUMINAMATH_GPT_recurrent_sequence_solution_l414_41476

theorem recurrent_sequence_solution (a : ℕ → ℕ) : 
  (a 1 = 1 ∧ ∀ n, n ≥ 2 → a n = 2 * a (n - 1) + 2^n) →
  (∀ n, n ≥ 1 → a n = (2 * n - 1) * 2^(n - 1)) :=
by
  sorry

end NUMINAMATH_GPT_recurrent_sequence_solution_l414_41476


namespace NUMINAMATH_GPT_range_a_l414_41453

def A : Set ℝ :=
  {x | x^2 + 5 * x + 6 ≤ 0}

def B : Set ℝ :=
  {x | -3 ≤ x ∧ x ≤ 5}

def C (a : ℝ) : Set ℝ :=
  {x | a < x ∧ x < a + 1}

theorem range_a (a : ℝ) : ((A ∪ B) ∩ C a = ∅) → (a ≥ 5 ∨ a ≤ -4) :=
  sorry

end NUMINAMATH_GPT_range_a_l414_41453


namespace NUMINAMATH_GPT_solution_set_abs_inequality_l414_41488

theorem solution_set_abs_inequality : {x : ℝ | |x - 2| < 1} = {x : ℝ | 1 < x ∧ x < 3} :=
sorry

end NUMINAMATH_GPT_solution_set_abs_inequality_l414_41488


namespace NUMINAMATH_GPT_investment_doubling_time_l414_41487

theorem investment_doubling_time :
  ∀ (r : ℝ) (initial_investment future_investment : ℝ),
  r = 8 →
  initial_investment = 5000 →
  future_investment = 20000 →
  (future_investment = initial_investment * 2 ^ (70 / r * 2)) →
  70 / r * 2 = 17.5 :=
by
  intros r initial_investment future_investment h_r h_initial h_future h_double
  sorry

end NUMINAMATH_GPT_investment_doubling_time_l414_41487


namespace NUMINAMATH_GPT_circle_equation_range_l414_41428

theorem circle_equation_range (a : ℝ) : 
  (∀ x y : ℝ, x^2 + y^2 + 2 * x - 4 * y + a + 1 = 0) → a < 4 := 
by 
  sorry

end NUMINAMATH_GPT_circle_equation_range_l414_41428


namespace NUMINAMATH_GPT_problem1_problem2_l414_41421

-- Given conditions
def A : Set ℝ := { x | x^2 - 2 * x - 15 > 0 }
def B : Set ℝ := { x | x < 6 }
def p (m : ℝ) : Prop := m ∈ A
def q (m : ℝ) : Prop := m ∈ B

-- Statements to prove
theorem problem1 (m : ℝ) : p m → m ∈ { x | x < -3 } ∪ { x | x > 5 } :=
sorry

theorem problem2 (m : ℝ) : (p m ∨ q m) ∧ (p m ∧ q m) → m ∈ { x | x < -3 } :=
sorry

end NUMINAMATH_GPT_problem1_problem2_l414_41421


namespace NUMINAMATH_GPT_product_ab_l414_41480

theorem product_ab (a b : ℝ) (h : a / 2 = 3 / b) : a * b = 6 := by
  sorry

end NUMINAMATH_GPT_product_ab_l414_41480


namespace NUMINAMATH_GPT_drying_time_correct_l414_41418

theorem drying_time_correct :
  let short_haired_dog_drying_time := 10
  let full_haired_dog_drying_time := 2 * short_haired_dog_drying_time
  let num_short_haired_dogs := 6
  let num_full_haired_dogs := 9
  let total_short_haired_dogs_time := num_short_haired_dogs * short_haired_dog_drying_time
  let total_full_haired_dogs_time := num_full_haired_dogs * full_haired_dog_drying_time
  let total_drying_time_in_minutes := total_short_haired_dogs_time + total_full_haired_dogs_time
  let total_drying_time_in_hours := total_drying_time_in_minutes / 60
  total_drying_time_in_hours = 4 := 
by
  sorry

end NUMINAMATH_GPT_drying_time_correct_l414_41418


namespace NUMINAMATH_GPT_sqrt_sum_of_roots_l414_41400

theorem sqrt_sum_of_roots :
  (36 + 14 * Real.sqrt 6 + 14 * Real.sqrt 5 + 6 * Real.sqrt 30).sqrt
  = (Real.sqrt 15 + Real.sqrt 10 + Real.sqrt 8 + Real.sqrt 3) :=
by
  sorry

end NUMINAMATH_GPT_sqrt_sum_of_roots_l414_41400


namespace NUMINAMATH_GPT_choose_amber_bronze_cells_l414_41470

theorem choose_amber_bronze_cells (a b : ℕ) (h_a_pos : a > 0) (h_b_pos : b > 0)
  (grid : Fin (a+b+1) × Fin (a+b+1) → Prop) 
  (amber_cells : ℕ) (h_amber_cells : amber_cells ≥ a^2 + a * b - b)
  (bronze_cells : ℕ) (h_bronze_cells : bronze_cells ≥ b^2 + b * a - a):
  ∃ (amber_choice : Fin (a+b+1) → Fin (a+b+1)), 
    ∃ (bronze_choice : Fin (a+b+1) → Fin (a+b+1)), 
    amber_choice ≠ bronze_choice ∧ 
    (∀ i j, i ≠ j → grid (amber_choice i) ≠ grid (bronze_choice j)) :=
sorry

end NUMINAMATH_GPT_choose_amber_bronze_cells_l414_41470


namespace NUMINAMATH_GPT_total_value_of_coins_l414_41409

theorem total_value_of_coins (q d : ℕ) (total_value original_value swapped_value : ℚ)
  (h1 : q + d = 30)
  (h2 : total_value = 4.50)
  (h3 : original_value = 25 * q + 10 * d)
  (h4 : swapped_value = 10 * q + 25 * d)
  (h5 : swapped_value = original_value + 1.50) :
  total_value = original_value / 100 :=
sorry

end NUMINAMATH_GPT_total_value_of_coins_l414_41409


namespace NUMINAMATH_GPT_line_of_intersection_in_standard_form_l414_41407

noncomputable def plane1 (x y z : ℝ) := 3 * x + 4 * y - 2 * z = 5
noncomputable def plane2 (x y z : ℝ) := 2 * x + 3 * y - z = 3

theorem line_of_intersection_in_standard_form :
  (∃ x y z : ℝ, plane1 x y z ∧ plane2 x y z ∧ (∀ t : ℝ, (x, y, z) = 
  (3 + 2 * t, -1 - t, t))) :=
by {
  sorry
}

end NUMINAMATH_GPT_line_of_intersection_in_standard_form_l414_41407


namespace NUMINAMATH_GPT_can_capacity_l414_41485

theorem can_capacity (x : ℝ) (milk water : ℝ) (full_capacity : ℝ) : 
  5 * x = milk ∧ 
  3 * x = water ∧ 
  full_capacity = milk + water + 8 ∧ 
  (milk + 8) / water = 2 → 
  full_capacity = 72 := 
sorry

end NUMINAMATH_GPT_can_capacity_l414_41485


namespace NUMINAMATH_GPT_cassie_nails_l414_41473

def num_dogs : ℕ := 4
def nails_per_dog_leg : ℕ := 4
def legs_per_dog : ℕ := 4
def num_parrots : ℕ := 8
def claws_per_parrot_leg : ℕ := 3
def legs_per_parrot : ℕ := 2
def extra_claws : ℕ := 1

def total_nails_to_cut : ℕ :=
  num_dogs * nails_per_dog_leg * legs_per_dog +
  num_parrots * claws_per_parrot_leg * legs_per_parrot + extra_claws

theorem cassie_nails : total_nails_to_cut = 113 :=
  by sorry

end NUMINAMATH_GPT_cassie_nails_l414_41473


namespace NUMINAMATH_GPT_isosceles_triangle_legs_length_l414_41446

theorem isosceles_triangle_legs_length 
  (P : ℝ) (base : ℝ) (leg_length : ℝ) 
  (hp : P = 26) 
  (hb : base = 11) 
  (hP : P = 2 * leg_length + base) : 
  leg_length = 7.5 := 
by 
  sorry

end NUMINAMATH_GPT_isosceles_triangle_legs_length_l414_41446


namespace NUMINAMATH_GPT_school_pays_570_l414_41422

theorem school_pays_570
  (price_per_model : ℕ := 100)
  (models_kindergarten : ℕ := 2)
  (models_elementary_multiple : ℕ := 2)
  (total_models : ℕ := models_kindergarten + models_elementary_multiple * models_kindergarten)
  (price_reduction : ℕ := if total_models > 5 then (price_per_model * 5 / 100) else 0)
  (reduced_price_per_model : ℕ := price_per_model - price_reduction) :
  2 * models_kindergarten * reduced_price_per_model = 570 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_school_pays_570_l414_41422


namespace NUMINAMATH_GPT_arithmetic_sum_l414_41452

theorem arithmetic_sum (a₁ an n : ℕ) (h₁ : a₁ = 5) (h₂ : an = 32) (h₃ : n = 10) :
  (n * (a₁ + an)) / 2 = 185 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sum_l414_41452


namespace NUMINAMATH_GPT_remainder_seven_times_quotient_l414_41425

theorem remainder_seven_times_quotient (n : ℕ) : 
  (∃ q r : ℕ, n = 23 * q + r ∧ r = 7 * q ∧ 0 ≤ r ∧ r < 23) ↔ (n = 30 ∨ n = 60 ∨ n = 90) :=
by 
  sorry

end NUMINAMATH_GPT_remainder_seven_times_quotient_l414_41425


namespace NUMINAMATH_GPT_original_ratio_l414_41413

theorem original_ratio (x y : ℤ)
  (h1 : y = 48)
  (h2 : (x + 12) * 2 = y) :
  x * 4 = y := sorry

end NUMINAMATH_GPT_original_ratio_l414_41413


namespace NUMINAMATH_GPT_total_beats_together_in_week_l414_41417

theorem total_beats_together_in_week :
  let samantha_beats_per_min := 250
  let samantha_hours_per_day := 3
  let michael_beats_per_min := 180
  let michael_hours_per_day := 2.5
  let days_per_week := 5

  let samantha_beats_per_day := samantha_beats_per_min * 60 * samantha_hours_per_day
  let samantha_beats_per_week := samantha_beats_per_day * days_per_week
  let michael_beats_per_day := michael_beats_per_min * 60 * michael_hours_per_day
  let michael_beats_per_week := michael_beats_per_day * days_per_week
  let total_beats_per_week := samantha_beats_per_week + michael_beats_per_week

  total_beats_per_week = 360000 := 
by
  -- The proof will go here
  sorry

end NUMINAMATH_GPT_total_beats_together_in_week_l414_41417


namespace NUMINAMATH_GPT_negation_of_proposition_l414_41492

theorem negation_of_proposition :
  ¬ (∃ x : ℝ, x ≤ 0 ∧ x^2 ≥ 0) ↔ ∀ x : ℝ, x ≤ 0 → x^2 < 0 :=
by
  sorry

end NUMINAMATH_GPT_negation_of_proposition_l414_41492


namespace NUMINAMATH_GPT_water_tank_capacity_l414_41435

theorem water_tank_capacity :
  ∃ (x : ℝ), 0.9 * x - 0.4 * x = 30 → x = 60 :=
by
  sorry

end NUMINAMATH_GPT_water_tank_capacity_l414_41435


namespace NUMINAMATH_GPT_cricket_run_rate_l414_41415

theorem cricket_run_rate (x : ℝ) (hx : 3.2 * x + 6.25 * 40 = 282) : x = 10 :=
by sorry

end NUMINAMATH_GPT_cricket_run_rate_l414_41415


namespace NUMINAMATH_GPT_total_sugar_in_all_candy_l414_41450

-- definitions based on the conditions
def chocolateBars : ℕ := 14
def sugarPerChocolateBar : ℕ := 10
def lollipopSugar : ℕ := 37

-- proof statement
theorem total_sugar_in_all_candy :
  (chocolateBars * sugarPerChocolateBar + lollipopSugar) = 177 := 
by
  sorry

end NUMINAMATH_GPT_total_sugar_in_all_candy_l414_41450


namespace NUMINAMATH_GPT_problem_statement_l414_41438

def f (x : ℝ) : ℝ := x^2 - 2 * x + 5
def g (x : ℝ) : ℝ := 2 * x + 1

theorem problem_statement : f (g 5) - g (f 5) = 63 :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l414_41438


namespace NUMINAMATH_GPT_gcd_problem_l414_41431

def a := 47^11 + 1
def b := 47^11 + 47^3 + 1

theorem gcd_problem : Nat.gcd a b = 1 := 
by
  sorry

end NUMINAMATH_GPT_gcd_problem_l414_41431


namespace NUMINAMATH_GPT_bob_grade_is_35_l414_41463

variable (J : ℕ) (S : ℕ) (B : ℕ)

-- Define Jenny's grade, Jason's grade based on Jenny's, and Bob's grade based on Jason's
def jennyGrade := 95
def jasonGrade := J - 25
def bobGrade := S / 2

-- Theorem to prove Bob's grade is 35 given the conditions
theorem bob_grade_is_35 (h1 : J = 95) (h2 : S = J - 25) (h3 : B = S / 2) : B = 35 :=
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_bob_grade_is_35_l414_41463


namespace NUMINAMATH_GPT_price_of_adult_ticket_l414_41404

theorem price_of_adult_ticket
  (price_child : ℤ)
  (price_adult : ℤ)
  (num_adults : ℤ)
  (num_children : ℤ)
  (total_amount : ℤ)
  (h1 : price_adult = 2 * price_child)
  (h2 : num_adults = 400)
  (h3 : num_children = 200)
  (h4 : total_amount = 16000) :
  num_adults * price_adult + num_children * price_child = total_amount → price_adult = 32 := by
    sorry

end NUMINAMATH_GPT_price_of_adult_ticket_l414_41404


namespace NUMINAMATH_GPT_digit_1035_is_2_l414_41433

noncomputable def sequence_digits (n : ℕ) : ℕ :=
  -- Convert the sequence of numbers from 1 to n to digits and return a specific position.
  sorry

theorem digit_1035_is_2 : sequence_digits 500 = 2 :=
  sorry

end NUMINAMATH_GPT_digit_1035_is_2_l414_41433


namespace NUMINAMATH_GPT_bella_more_than_max_l414_41462

noncomputable def num_students : ℕ := 10
noncomputable def bananas_eaten_by_bella : ℕ := 7
noncomputable def bananas_eaten_by_max : ℕ := 1

theorem bella_more_than_max : 
  bananas_eaten_by_bella - bananas_eaten_by_max = 6 :=
by
  sorry

end NUMINAMATH_GPT_bella_more_than_max_l414_41462


namespace NUMINAMATH_GPT_calculate_value_l414_41483

theorem calculate_value (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (hxy : x = 1 / y) (hzy : z = 1 / y) : 
  (x + 1 / x) * (z - 1 / z) = 4 := 
by 
  -- Proof omitted, this is just the statement
  sorry

end NUMINAMATH_GPT_calculate_value_l414_41483


namespace NUMINAMATH_GPT_equal_roots_quadratic_l414_41405

def quadratic_discriminant (a b c : ℝ) : ℝ :=
  b ^ 2 - 4 * a * c

/--
If the quadratic equation 2x^2 - ax + 2 = 0 has two equal real roots,
then the value of a is ±4.
-/
theorem equal_roots_quadratic (a : ℝ) (h : quadratic_discriminant 2 (-a) 2 = 0) :
  a = 4 ∨ a = -4 :=
sorry

end NUMINAMATH_GPT_equal_roots_quadratic_l414_41405


namespace NUMINAMATH_GPT_least_xy_value_l414_41410

theorem least_xy_value (x y : ℕ) (hx : 0 < x) (hy : 0 < y) (h : (1/x : ℚ) + 1/(2*y) = 1/8) :
  xy ≥ 128 :=
sorry

end NUMINAMATH_GPT_least_xy_value_l414_41410


namespace NUMINAMATH_GPT_acid_solution_replaced_l414_41467

theorem acid_solution_replaced (P : ℝ) :
  (0.5 * 0.50 + 0.5 * P = 0.35) → P = 0.20 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_acid_solution_replaced_l414_41467


namespace NUMINAMATH_GPT_line_equation_through_point_parallel_to_lines_l414_41402

theorem line_equation_through_point_parallel_to_lines (L L1 L2 : ℝ → ℝ → Prop) :
  (∀ x, L1 x (y: ℝ) ↔ 3 * x + y - 6 = 0) →
  (∀ x, L2 x (y: ℝ) ↔ 3 * x + y + 3 = 0) →
  (L 1 0) →
  (∀ x1 y1 x2 y2, L1 x1 y1 → L1 x2 y2 → (y2 - y1) / (x2 - x1) = -3) →
  ∃ A B C, (A = 1 ∧ B = -3 ∧ C = -3) ∧ (∀ x y, L x y ↔ A * x + B * y + C = 0) :=
by sorry

end NUMINAMATH_GPT_line_equation_through_point_parallel_to_lines_l414_41402


namespace NUMINAMATH_GPT_prove_inequality_l414_41489

theorem prove_inequality
  (a : ℕ → ℕ) -- Define a sequence of natural numbers
  (h_initial : a 1 > a 0) -- Initial condition
  (h_recurrence : ∀ n ≥ 2, a n = 3 * a (n - 1) - 2 * a (n - 2)) -- Recurrence relation
  : a 100 > 2^99 := by
  sorry -- Proof placeholder

end NUMINAMATH_GPT_prove_inequality_l414_41489


namespace NUMINAMATH_GPT_customer_count_l414_41479

theorem customer_count :
  let initial_customers := 13
  let customers_after_first_leave := initial_customers - 5
  let customers_after_new_arrival := customers_after_first_leave + 4
  let customers_after_group_join := customers_after_new_arrival + 8
  let final_customers := customers_after_group_join - 6
  final_customers = 14 :=
by
  sorry

end NUMINAMATH_GPT_customer_count_l414_41479


namespace NUMINAMATH_GPT_rectangle_dimension_area_l414_41461

theorem rectangle_dimension_area (x : ℝ) 
  (h_dim : (3 * x - 5) * (x + 7) = 14 * x - 35) : 
  x = 0 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_dimension_area_l414_41461


namespace NUMINAMATH_GPT_period_fraction_sum_nines_l414_41439

theorem period_fraction_sum_nines (q : ℕ) (p : ℕ) (N N1 N2 : ℕ) (n : ℕ) (t : ℕ) 
  (hq_prime : Nat.Prime q) (hq_gt_5 : q > 5) (hp_lt_q : p < q)
  (ht_eq_2n : t = 2 * n) (h_period : 10^t ≡ 1 [MOD q])
  (hN_eq_concat : (N = N1 * 10^n + N2) ∧ (N % 10^n = N2))
  : N1 + N2 = (10^n - 1) := 
sorry

end NUMINAMATH_GPT_period_fraction_sum_nines_l414_41439


namespace NUMINAMATH_GPT_factorization_correct_l414_41497

-- Define the initial expression
def initial_expr (a x : ℝ) : ℝ := a * x^2 - 2 * a * x + a

-- Define the factorized form
def factorized_expr (a x : ℝ) : ℝ := a * (x - 1)^2

-- Create the theorem statement
theorem factorization_correct (a x : ℝ) : initial_expr a x = factorized_expr a x :=
by simp [initial_expr, factorized_expr, pow_two, mul_add, add_mul, add_assoc]; sorry

end NUMINAMATH_GPT_factorization_correct_l414_41497
