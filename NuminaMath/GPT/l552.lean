import Mathlib

namespace min_sum_of_perpendicular_sides_l552_55227

noncomputable def min_sum_perpendicular_sides (a b : ℝ) (h : a * b = 100) : ℝ :=
a + b

theorem min_sum_of_perpendicular_sides {a b : ℝ} (h : a * b = 100) : min_sum_perpendicular_sides a b h = 20 :=
sorry

end min_sum_of_perpendicular_sides_l552_55227


namespace length_of_room_l552_55239

theorem length_of_room (Area Width Length : ℝ) (h1 : Area = 10) (h2 : Width = 2) (h3 : Area = Length * Width) : Length = 5 :=
by
  sorry

end length_of_room_l552_55239


namespace geom_seq_sum_l552_55255

theorem geom_seq_sum {a : ℕ → ℝ} (q : ℝ) (h1 : a 0 + a 1 + a 2 = 2)
    (h2 : a 3 + a 4 + a 5 = 16)
    (h_geom : ∀ n, a (n + 1) = q * a n) :
  a 6 + a 7 + a 8 = 128 :=
sorry

end geom_seq_sum_l552_55255


namespace three_digit_number_441_or_882_l552_55224

def is_valid_number (n : ℕ) : Prop :=
  let a := n / 100
  let b := (n % 100) / 10
  let c := n % 10
  n = 100 * a + 10 * b + c ∧
  n / (100 * c + 10 * b + a) = 3 ∧
  n % (100 * c + 10 * b + a) = a + b + c

theorem three_digit_number_441_or_882:
  ∀ n : ℕ, is_valid_number n → (n = 441 ∨ n = 882) :=
by
  sorry

end three_digit_number_441_or_882_l552_55224


namespace function_solution_l552_55252

theorem function_solution (f : ℝ → ℝ) (a : ℝ) : 
  (∀ x : ℝ, f x = sorry) → f a = sorry → (a = 1 ∨ a = -1) :=
by
  intros hfa hfb
  sorry

end function_solution_l552_55252


namespace sequence_is_arithmetic_not_geometric_l552_55296

noncomputable def a := Real.log 3 / Real.log 2
noncomputable def b := Real.log 6 / Real.log 2
noncomputable def c := Real.log 12 / Real.log 2

theorem sequence_is_arithmetic_not_geometric : 
  (b - a = c - b) ∧ (b / a ≠ c / b) := 
by
  sorry

end sequence_is_arithmetic_not_geometric_l552_55296


namespace sqrt_20n_integer_exists_l552_55259

theorem sqrt_20n_integer_exists : 
  ∃ n : ℤ, 0 ≤ n ∧ ∃ k : ℤ, k * k = 20 * n :=
sorry

end sqrt_20n_integer_exists_l552_55259


namespace inheritance_problem_l552_55205

def wifeAmounts (K J M : ℝ) : Prop :=
  K + J + M = 396 ∧
  J = K + 10 ∧
  M = J + 10

def husbandAmounts (wifeAmount : ℝ) (husbandMultiplier : ℝ := 1) : ℝ :=
  husbandMultiplier * wifeAmount

theorem inheritance_problem (K J M : ℝ)
  (h1 : wifeAmounts K J M)
  : ∃ wifeOf : String → String,
    wifeOf "John Smith" = "Katherine" ∧
    wifeOf "Henry Snooks" = "Jane" ∧
    wifeOf "Tom Crow" = "Mary" ∧
    husbandAmounts K = K ∧
    husbandAmounts J 1.5 = 1.5 * J ∧
    husbandAmounts M 2 = 2 * M :=
by 
  sorry

end inheritance_problem_l552_55205


namespace martha_initial_juice_pantry_l552_55282

theorem martha_initial_juice_pantry (P : ℕ) : 
  4 + P + 5 - 3 = 10 → P = 4 := 
by
  intro h
  sorry

end martha_initial_juice_pantry_l552_55282


namespace rectangles_on_grid_l552_55262

-- Define the grid dimensions
def m := 3
def n := 2

-- Define a function to count the total number of rectangles formed by the grid.
def count_rectangles (m n : ℕ) : ℕ := 
  (m * (m - 1) / 2 + n * (n - 1) / 2) * (n * (n - 1) / 2 + m * (m - 1) / 2) 

-- State the theorem we need to prove
theorem rectangles_on_grid : count_rectangles m n = 14 :=
  sorry

end rectangles_on_grid_l552_55262


namespace parity_expression_l552_55214

theorem parity_expression
  (a b c : ℕ)
  (h_a_pos : a > 0)
  (h_b_pos : b > 0)
  (h_a_odd : a % 2 = 1)
  (h_b_odd : b % 2 = 1) :
  (5^a + (b + 1)^2 * c) % 2 = 1 :=
by
  sorry

end parity_expression_l552_55214


namespace pictures_at_museum_l552_55216

-- Define the given conditions
def z : ℕ := 24
def k : ℕ := 14
def p : ℕ := 22

-- Define the number of pictures taken at the museum
def M : ℕ := 12

-- The theorem to be proven
theorem pictures_at_museum :
  z + M - k = p ↔ M = 12 :=
by
  sorry

end pictures_at_museum_l552_55216


namespace ray_two_digit_number_l552_55223

theorem ray_two_digit_number (a b n : ℕ) (ha : 1 ≤ a) (hb : b ≤ 9) (hn : n = 10 * a + b) (h1 : n = 4 * (a + b) + 3) (h2 : n + 18 = 10 * b + a) : n = 35 := by
  sorry

end ray_two_digit_number_l552_55223


namespace car_speeds_and_arrival_times_l552_55294

theorem car_speeds_and_arrival_times
  (x y z u : ℝ)
  (h1 : x^2 = (y + z) * u)
  (h2 : (y + z) / 4 = u)
  (h3 : x / u = y / z)
  (h4 : x + y + z + u = 210) :
  x = 60 ∧ y = 80 ∧ z = 40 ∧ u = 30 := 
by
  sorry

end car_speeds_and_arrival_times_l552_55294


namespace average_age_of_persons_l552_55283

theorem average_age_of_persons 
  (total_age : ℕ := 270) 
  (average_age : ℕ := 15) : 
  (total_age / average_age) = 18 := 
by { 
  sorry 
}

end average_age_of_persons_l552_55283


namespace sequence_satisfies_recurrence_l552_55203

theorem sequence_satisfies_recurrence (n : ℕ) (a : ℕ → ℕ) (h : ∀ k, 2 ≤ k → k ≤ n - 1 → a (k + 1) = (a k ^ 2 + 1) / (a (k - 1) + 1) - 1) :
  n = 3 ∨ n = 4 := by
  sorry

end sequence_satisfies_recurrence_l552_55203


namespace find_a_l552_55206

theorem find_a {a : ℝ} (h : ∀ x : ℝ, (x^2 - 4 * x + a) + |x - 3| ≤ 5 → x ≤ 3) : a = 8 :=
sorry

end find_a_l552_55206


namespace factorize_cubed_sub_four_l552_55266

theorem factorize_cubed_sub_four (a : ℝ) : a^3 - 4 * a = a * (a + 2) * (a - 2) :=
by
  sorry

end factorize_cubed_sub_four_l552_55266


namespace even_expressions_l552_55276

theorem even_expressions (x y : ℕ) (hx : Even x) (hy : Even y) :
  Even (x + 5 * y) ∧
  Even (4 * x - 3 * y) ∧
  Even (2 * x^2 + 5 * y^2) ∧
  Even ((2 * x * y + 4)^2) ∧
  Even (4 * x * y) :=
by
  sorry

end even_expressions_l552_55276


namespace driver_days_off_l552_55202

theorem driver_days_off 
  (drivers : ℕ) 
  (cars : ℕ) 
  (maintenance_rate : ℚ) 
  (days_in_month : ℕ)
  (needed_driver_days : ℕ)
  (x : ℚ) :
  drivers = 54 →
  cars = 60 →
  maintenance_rate = 0.25 →
  days_in_month = 30 →
  needed_driver_days = 45 * days_in_month →
  54 * (30 - x) = needed_driver_days →
  x = 5 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end driver_days_off_l552_55202


namespace smallest_positive_integer_l552_55299

theorem smallest_positive_integer (N : ℕ) :
  (N % 5 = 2) ∧ (N % 6 = 3) ∧ (N % 7 = 4) ∧ (N % 11 = 9) → N = 207 :=
by
  sorry

end smallest_positive_integer_l552_55299


namespace al_initial_portion_l552_55246

theorem al_initial_portion (a b c : ℝ) 
  (h1 : a + b + c = 1200) 
  (h2 : a - 200 + 2 * b + 1.5 * c = 1800) : 
  a = 600 :=
sorry

end al_initial_portion_l552_55246


namespace length_of_second_dimension_l552_55292

def volume_of_box (w : ℝ) : ℝ :=
  (w - 16) * (46 - 16) * 8

theorem length_of_second_dimension (w : ℝ) (h_volume : volume_of_box w = 4800) : w = 36 :=
by
  sorry

end length_of_second_dimension_l552_55292


namespace circle_radius_squared_l552_55298

theorem circle_radius_squared (r : ℝ) 
  (AB CD: ℝ) 
  (BP angleAPD : ℝ) 
  (P_outside_circle: True) 
  (AB_eq_12 : AB = 12) 
  (CD_eq_9 : CD = 9) 
  (AngleAPD_eq_45 : angleAPD = 45) 
  (BP_eq_10 : BP = 10) : r^2 = 73 :=
sorry

end circle_radius_squared_l552_55298


namespace solve_equation_l552_55274

theorem solve_equation (x : ℝ) (h : (x + 6) / (x - 3) = 4) : x = 6 :=
sorry

end solve_equation_l552_55274


namespace find_p_r_l552_55249

-- Definitions of the polynomials
def f (x : ℝ) (p q : ℝ) : ℝ := x^2 + p * x + q
def g (x : ℝ) (r s : ℝ) : ℝ := x^2 + r * x + s

-- Lean statement of the proof problem:
theorem find_p_r (p q r s : ℝ) (h1 : p ≠ r) (h2 : g (-p / 2) r s = 0) 
  (h3 : f (-r / 2) p q = 0) (h4 : ∀ x : ℝ, f x p q = g x r s) 
  (h5 : f 50 p q = -50) : p + r = -200 := 
sorry

end find_p_r_l552_55249


namespace cyclist_wait_time_l552_55257

noncomputable def hiker_speed : ℝ := 5 / 60
noncomputable def cyclist_speed : ℝ := 25 / 60
noncomputable def wait_time : ℝ := 5
noncomputable def distance_ahead : ℝ := cyclist_speed * wait_time
noncomputable def catching_time : ℝ := distance_ahead / hiker_speed

theorem cyclist_wait_time : catching_time = 25 := by
  sorry

end cyclist_wait_time_l552_55257


namespace multiply_or_divide_inequality_by_negative_number_l552_55295

theorem multiply_or_divide_inequality_by_negative_number {a b c : ℝ} (h : a < b) (hc : c < 0) :
  c * a > c * b ∧ a / c > b / c :=
sorry

end multiply_or_divide_inequality_by_negative_number_l552_55295


namespace line_equation_M_l552_55277

theorem line_equation_M (x y : ℝ) :
  (∃ (m c : ℝ), y = m * x + c ∧ m = -5/4 ∧ c = -3)
  ∧ (∃ (slope intercept : ℝ), slope = 2 * (-5/4) ∧ intercept = (1/2) * -3 ∧ (y - 2 = slope * (x + 4)))
  → ∃ (a b : ℝ), y = a * x + b ∧ a = -5/2 ∧ b = -8 :=
by
  sorry

end line_equation_M_l552_55277


namespace minimum_reciprocal_sum_of_roots_l552_55228

noncomputable def f (x : ℝ) (b : ℝ) (c : ℝ) := 2 * x^2 + b * x + c

theorem minimum_reciprocal_sum_of_roots {b c : ℝ} {x1 x2 : ℝ} 
  (h1: f (-10) b c = f 12 b c)
  (h2: f x1 b c = 0)
  (h3: f x2 b c = 0)
  (h4: 0 < x1)
  (h5: 0 < x2)
  (h6: x1 + x2 = 2) :
  (1 / x1 + 1 / x2) = 2 :=
sorry

end minimum_reciprocal_sum_of_roots_l552_55228


namespace largest_number_in_set_l552_55240

theorem largest_number_in_set :
  ∀ (a b c d : ℤ), (a ∈ [0, 2, -1, -2]) → (b ∈ [0, 2, -1, -2]) → (c ∈ [0, 2, -1, -2]) → (d ∈ [0, 2, -1, -2])
  → (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  → max (max a b) (max c d) = 2
  := 
by
  sorry

end largest_number_in_set_l552_55240


namespace garden_area_increase_l552_55265

/-- A 60-foot by 20-foot rectangular garden is enclosed by a fence. Changing its shape to a square using
the same amount of fencing makes the new garden 400 square feet larger than the old garden. -/
theorem garden_area_increase :
  let length := 60
  let width := 20
  let original_area := length * width
  let perimeter := 2 * (length + width)
  let new_side := perimeter / 4
  let new_area := new_side * new_side
  new_area - original_area = 400 :=
by
  sorry

end garden_area_increase_l552_55265


namespace amount_paid_per_person_is_correct_l552_55229

noncomputable def amount_each_person_paid (total_bill : ℝ) (tip_rate : ℝ) (tax_rate : ℝ) (num_people : ℕ) : ℝ := 
  let tip_amount := tip_rate * total_bill
  let tax_amount := tax_rate * total_bill
  let total_amount := total_bill + tip_amount + tax_amount
  total_amount / num_people

theorem amount_paid_per_person_is_correct :
  amount_each_person_paid 425 0.18 0.08 15 = 35.7 :=
by
  sorry

end amount_paid_per_person_is_correct_l552_55229


namespace value_of_a_l552_55208

theorem value_of_a
  (a : ℝ)
  (h1 : ∀ (ρ θ : ℝ), ρ * (Real.cos θ + Real.sin θ) = 1)
  (h2 : ∀ (ρ : ℝ), ρ = a)
  (h3 : ∃ (ρ θ : ℝ), ρ * (Real.cos θ + Real.sin θ) = 1 ∧ ρ = a ∧ θ = 0)  :
  a = Real.sqrt 2 / 2 := 
sorry

end value_of_a_l552_55208


namespace union_A_B_eq_real_subset_A_B_l552_55211

def A (a : ℝ) : Set ℝ := {x : ℝ | a < x ∧ x < 3 + a}
def B : Set ℝ := {x : ℝ | x ≤ -1 ∨ x ≥ 1}

theorem union_A_B_eq_real (a : ℝ) : (A a ∪ B) = Set.univ ↔ -2 ≤ a ∧ a ≤ -1 :=
by
  sorry

theorem subset_A_B (a : ℝ) : A a ⊆ B ↔ (a ≤ -4 ∨ a ≥ 1) :=
by
  sorry

end union_A_B_eq_real_subset_A_B_l552_55211


namespace difference_fewer_children_than_adults_l552_55272

theorem difference_fewer_children_than_adults : 
  ∀ (C S : ℕ), 2 * C = S → 58 + C + S = 127 → (58 - C = 35) :=
by
  intros C S h1 h2
  sorry

end difference_fewer_children_than_adults_l552_55272


namespace real_solution_four_unknowns_l552_55275

theorem real_solution_four_unknowns (x y z t : ℝ) :
  x^2 + y^2 + z^2 + t^2 = x * (y + z + t) ↔ (x = 0 ∧ y = 0 ∧ z = 0 ∧ t = 0) :=
by
  sorry

end real_solution_four_unknowns_l552_55275


namespace intersection_complement_l552_55290

open Set

noncomputable def U : Set ℝ := univ

def A : Set ℝ := {x | x^2 - 2 * x < 0}

def B : Set ℝ := {x | x > 1}

theorem intersection_complement (x : ℝ) :
  x ∈ (A ∩ (U \ B)) ↔ 0 < x ∧ x ≤ 1 :=
by
  sorry

end intersection_complement_l552_55290


namespace square_perimeter_is_44_8_l552_55247

noncomputable def perimeter_of_congruent_rectangles_division (s : ℝ) (P : ℝ) : ℝ :=
  let rectangle_perimeter := 2 * (s + s / 4)
  if rectangle_perimeter = P then 4 * s else 0

theorem square_perimeter_is_44_8 :
  ∀ (s : ℝ) (P : ℝ), P = 28 → 4 * s = 44.8 → perimeter_of_congruent_rectangles_division s P = 44.8 :=
by intros s P h1 h2
   sorry

end square_perimeter_is_44_8_l552_55247


namespace train_crossing_pole_time_l552_55291

/-- 
Given the conditions:
1. The train is running at a speed of 60 km/hr.
2. The length of the train is 66.66666666666667 meters.
Prove that it takes 4 seconds for the train to cross the pole.
-/
theorem train_crossing_pole_time :
  let speed_km_hr := 60
  let length_m := 66.66666666666667
  let conversion_factor := 1000 / 3600
  let speed_m_s := speed_km_hr * conversion_factor
  let time := length_m / speed_m_s
  time = 4 :=
by
  sorry

end train_crossing_pole_time_l552_55291


namespace repeating_decimal_sum_l552_55207

theorem repeating_decimal_sum :
  (0.6666666666 : ℝ) + (0.7777777777 : ℝ) = (13 : ℚ) / 9 := by
  sorry

end repeating_decimal_sum_l552_55207


namespace jinho_remaining_money_l552_55219

def jinho_initial_money : ℕ := 2500
def cost_per_eraser : ℕ := 120
def erasers_bought : ℕ := 5
def cost_per_pencil : ℕ := 350
def pencils_bought : ℕ := 3

theorem jinho_remaining_money :
  jinho_initial_money - (erasers_bought * cost_per_eraser + pencils_bought * cost_per_pencil) = 850 :=
by
  sorry

end jinho_remaining_money_l552_55219


namespace solve_for_x_l552_55254

theorem solve_for_x (x : ℝ) (h : 8 * (2 + 1 / x) = 18) : x = 4 := by
  sorry

end solve_for_x_l552_55254


namespace parity_of_expression_l552_55263

theorem parity_of_expression
  (a b c : ℕ) 
  (h_a_odd : a % 2 = 1) 
  (h_b_odd : b % 2 = 1) 
  (h_a_pos : a > 0) 
  (h_b_pos : b > 0)
  (h_c_pos : c > 0) :
  ((3^a + (b + 2)^2 * c) % 2 = 1 ↔ c % 2 = 0) ∧ 
  ((3^a + (b + 2)^2 * c) % 2 = 0 ↔ c % 2 = 1) :=
by sorry

end parity_of_expression_l552_55263


namespace triangle_area_correct_l552_55250

noncomputable def area_of_triangle 
  (a b c : ℝ) (ha : a = Real.sqrt 29) (hb : b = Real.sqrt 13) (hc : c = Real.sqrt 34) : ℝ :=
  let cosC := (b^2 + c^2 - a^2) / (2 * b * c)
  let sinC := Real.sqrt (1 - cosC^2)
  (1 / 2) * b * c * sinC

theorem triangle_area_correct : area_of_triangle (Real.sqrt 29) (Real.sqrt 13) (Real.sqrt 34) 
  (by rfl) (by rfl) (by rfl) = 19 / 2 :=
sorry

end triangle_area_correct_l552_55250


namespace koschei_coins_l552_55222

theorem koschei_coins :
  ∃ a : ℕ, (a % 10 = 7) ∧ (a % 12 = 9) ∧ (300 ≤ a) ∧ (a ≤ 400) ∧ a = 357 :=
by 
  sorry

end koschei_coins_l552_55222


namespace marnie_eats_chips_l552_55253

theorem marnie_eats_chips (total_chips : ℕ) (chips_first_batch : ℕ) (chips_second_batch : ℕ) (daily_chips : ℕ) (remaining_chips : ℕ) (total_days : ℕ) :
  total_chips = 100 →
  chips_first_batch = 5 →
  chips_second_batch = 5 →
  daily_chips = 10 →
  remaining_chips = total_chips - (chips_first_batch + chips_second_batch) →
  total_days = remaining_chips / daily_chips + 1 →
  total_days = 10 :=
by
  sorry

end marnie_eats_chips_l552_55253


namespace sin_180_degree_l552_55248

theorem sin_180_degree : Real.sin (Real.pi) = 0 := by sorry

end sin_180_degree_l552_55248


namespace part_a_l552_55210

def A (f : ℝ → ℝ) : Prop := ∀ x y : ℝ, f (x * y) = x * f y

theorem part_a (f : ℝ → ℝ) (h : A f) : ∀ x y : ℝ, f (x + y) = f x + f y :=
sorry

end part_a_l552_55210


namespace eight_digit_product_1400_l552_55236

def eight_digit_numbers_count : Nat :=
  sorry

theorem eight_digit_product_1400 : eight_digit_numbers_count = 5880 :=
  sorry

end eight_digit_product_1400_l552_55236


namespace subset1_squares_equals_product_subset2_squares_equals_product_l552_55279

theorem subset1_squares_equals_product :
  (1^2 + 3^2 + 4^2 + 9^2 + 107^2 = 1 * 3 * 4 * 9 * 107) :=
sorry

theorem subset2_squares_equals_product :
  (3^2 + 4^2 + 9^2 + 107^2 + 11555^2 = 3 * 4 * 9 * 107 * 11555) :=
sorry

end subset1_squares_equals_product_subset2_squares_equals_product_l552_55279


namespace polar_coordinates_of_point_l552_55268

theorem polar_coordinates_of_point :
  ∀ (x y : ℝ) (r θ : ℝ), x = -1 ∧ y = 1 ∧ r = Real.sqrt (x^2 + y^2) ∧ θ = Real.arctan (y / x) ∧ r > 0 ∧ 0 ≤ θ ∧ θ < 2 * Real.pi
  → r = Real.sqrt 2 ∧ θ = 3 * Real.pi / 4 := 
by
  intros x y r θ h
  sorry

end polar_coordinates_of_point_l552_55268


namespace geometric_sequence_a2_l552_55251

theorem geometric_sequence_a2 
  (a : ℕ → ℝ) 
  (q : ℝ)
  (h1 : a 1 = 1/4) 
  (h3_h5 : a 3 * a 5 = 4 * (a 4 - 1)) 
  (h_seq : ∀ n : ℕ, a n = a 1 * q ^ (n - 1)) :
  a 2 = 1/2 :=
sorry

end geometric_sequence_a2_l552_55251


namespace arcsin_inequality_l552_55244

theorem arcsin_inequality (x y : ℝ) (hx : -1 ≤ x ∧ x ≤ 1) (hy : -1 ≤ y ∧ y ≤ 1) :
  (Real.arcsin x + Real.arcsin y > Real.pi / 2) ↔ (x ≥ 0 ∧ y ≥ 0 ∧ (y^2 + x^2 > 1)) := by
sorry

end arcsin_inequality_l552_55244


namespace servings_of_honey_l552_55218

theorem servings_of_honey :
  let total_ounces := 37 + 1/3
  let serving_size := 1 + 1/2
  total_ounces / serving_size = 24 + 8/9 :=
by
  sorry

end servings_of_honey_l552_55218


namespace max_value_of_E_l552_55213

theorem max_value_of_E (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a ^ 5 + b ^ 5 = a ^ 3 + b ^ 3) : 
  a^2 - a*b + b^2 ≤ 1 :=
sorry

end max_value_of_E_l552_55213


namespace complement_of_M_l552_55200

def M : Set ℝ := {x | x^2 - 2 * x > 0}

def U : Set ℝ := Set.univ

theorem complement_of_M :
  (U \ M) = (Set.Icc 0 2) :=
by
  sorry

end complement_of_M_l552_55200


namespace stratified_sampling_females_l552_55278

theorem stratified_sampling_females :
  let males := 500
  let females := 400
  let total_students := 900
  let total_surveyed := 45
  let males_surveyed := 25
  ((males_surveyed : ℚ) / males) * females = 20 := by
  sorry

end stratified_sampling_females_l552_55278


namespace number_of_partners_equation_l552_55237

variable (x : ℕ)

theorem number_of_partners_equation :
  5 * x + 45 = 7 * x - 3 :=
sorry

end number_of_partners_equation_l552_55237


namespace train_speed_l552_55264

noncomputable def speed_of_train_kmph (L V : ℝ) : ℝ :=
  3.6 * V

theorem train_speed
  (L V : ℝ)
  (h1 : L = 18 * V)
  (h2 : L + 340 = 35 * V) :
  speed_of_train_kmph L V = 72 :=
by
  sorry

end train_speed_l552_55264


namespace probability_of_forming_CHORAL_is_correct_l552_55288

-- Definitions for selecting letters with given probabilities
def probability_select_C_A_L_from_CAMEL : ℚ :=
  1 / 10

def probability_select_H_O_R_from_SHRUB : ℚ :=
  1 / 10

def probability_select_G_from_GLOW : ℚ :=
  1 / 2

-- Calculating the total probability of selecting letters to form "CHORAL"
def probability_form_CHORAL : ℚ :=
  probability_select_C_A_L_from_CAMEL * 
  probability_select_H_O_R_from_SHRUB * 
  probability_select_G_from_GLOW

theorem probability_of_forming_CHORAL_is_correct :
  probability_form_CHORAL = 1 / 200 :=
by
  -- Statement to be proven here
  sorry

end probability_of_forming_CHORAL_is_correct_l552_55288


namespace black_balls_probability_both_black_l552_55280

theorem black_balls_probability_both_black (balls_total balls_black balls_gold : ℕ) (prob : ℚ) 
  (h1 : balls_total = 11)
  (h2 : balls_black = 7)
  (h3 : balls_gold = 4)
  (h4 : balls_total = balls_black + balls_gold)
  (h5 : prob = (21 : ℚ) / 55) :
  balls_total.choose 2 * prob = balls_black.choose 2 :=
sorry

end black_balls_probability_both_black_l552_55280


namespace find_cos_beta_l552_55201

variable {α β : ℝ}
variable (h_acute_α : 0 < α ∧ α < π / 2)
variable (h_acute_β : 0 < β ∧ β < π / 2)
variable (h_sin_α : Real.sin α = 2 / 5 * Real.sqrt 5)
variable (h_sin_α_plus_β : Real.sin (α + β) = 3 / 5)

theorem find_cos_beta 
  (h_acute_α : 0 < α ∧ α < π / 2)
  (h_acute_β : 0 < β ∧ β < π / 2)
  (h_sin_α : Real.sin α = 2 / 5 * Real.sqrt 5)
  (h_sin_α_plus_β : Real.sin (α + β) = 3 / 5) :
  Real.cos β = Real.sqrt 5 / 5 := 
sorry

end find_cos_beta_l552_55201


namespace determine_b_l552_55293

noncomputable def f (x : ℝ) (b : ℝ) : ℝ :=
if x < 1 then 3 * x - b else 2 ^ x

theorem determine_b (b : ℝ) :
  f (f (5 / 6) b) b = 4 ↔ b = 1 / 2 :=
by sorry

end determine_b_l552_55293


namespace convex_polyhedron_inequality_l552_55269

noncomputable def convex_polyhedron (B P T : ℕ) : Prop :=
  ∀ (B P T : ℕ), B > 0 ∧ P > 0 ∧ T >= 0 → B * (Nat.sqrt (P + T)) ≥ 2 * P

theorem convex_polyhedron_inequality (B P T : ℕ) (h : convex_polyhedron B P T) : 
  B * (Nat.sqrt (P + T)) ≥ 2 * P :=
by
  sorry

end convex_polyhedron_inequality_l552_55269


namespace quadratic_inequality_solution_l552_55230

theorem quadratic_inequality_solution (x : ℝ) : 
  3 * x^2 - 8 * x - 3 > 0 ↔ (x < -1/3 ∨ x > 3) :=
by
  sorry

end quadratic_inequality_solution_l552_55230


namespace solution_l552_55204

-- Define the equation
def equation (x : ℝ) := x^2 + 4*x + 3 + (x + 3)*(x + 5) = 0

-- State that x = -3 is a solution to the equation
theorem solution : equation (-3) :=
by
  unfold equation
  simp
  sorry

end solution_l552_55204


namespace fraction_simplification_l552_55281

theorem fraction_simplification (a b c : ℝ) :
  (4 * a^2 + 2 * c^2 - 4 * b^2 - 8 * b * c) / (3 * a^2 + 6 * a * c - 3 * c^2 - 6 * a * b) =
  (4 / 3) * ((a - 2 * b + c) * (a - c)) / ((a - b + c) * (a - b - c)) :=
by
  sorry

end fraction_simplification_l552_55281


namespace remainder_of_sum_of_primes_is_eight_l552_55260

-- Define the first eight primes and their sum
def firstEightPrimes : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]
def sumFirstEightPrimes : ℕ := 77

-- Define the ninth prime
def ninthPrime : ℕ := 23

-- Theorem stating the equivalence
theorem remainder_of_sum_of_primes_is_eight :
  (sumFirstEightPrimes % ninthPrime) = 8 := by
  sorry

end remainder_of_sum_of_primes_is_eight_l552_55260


namespace price_of_second_oil_l552_55225

theorem price_of_second_oil : 
  ∃ x : ℝ, 
    (10 * 50 + 5 * x = 15 * 56) → x = 68 := by
  sorry

end price_of_second_oil_l552_55225


namespace Donovan_Mitchell_goal_l552_55297

theorem Donovan_Mitchell_goal 
  (current_avg : ℕ) 
  (current_games : ℕ) 
  (target_avg : ℕ) 
  (total_games : ℕ) 
  (remaining_games : ℕ) 
  (points_scored_so_far : ℕ)
  (points_needed_total : ℕ)
  (points_needed_remaining : ℕ) :
  (current_avg = 26) ∧
  (current_games = 15) ∧
  (target_avg = 30) ∧
  (total_games = 20) ∧
  (remaining_games = 5) ∧
  (points_scored_so_far = current_avg * current_games) ∧
  (points_needed_total = target_avg * total_games) ∧
  (points_needed_remaining = points_needed_total - points_scored_so_far) →
  (points_needed_remaining / remaining_games = 42) :=
by
  sorry

end Donovan_Mitchell_goal_l552_55297


namespace complement_of_angle_is_acute_l552_55271

theorem complement_of_angle_is_acute (θ : ℝ) (h₀ : 0 < θ) (h₁ : θ < 90) : 0 < 90 - θ ∧ 90 - θ < 90 :=
by sorry

end complement_of_angle_is_acute_l552_55271


namespace rationalize_denominator_l552_55284

theorem rationalize_denominator :
  (35 / Real.sqrt 35) = Real.sqrt 35 :=
sorry

end rationalize_denominator_l552_55284


namespace canoes_more_than_kayaks_l552_55258

noncomputable def canoes_and_kayaks (C K : ℕ) : Prop :=
  (2 * C = 3 * K) ∧ (12 * C + 18 * K = 504) ∧ (C - K = 7)

theorem canoes_more_than_kayaks (C K : ℕ) (h : canoes_and_kayaks C K) : C - K = 7 :=
sorry

end canoes_more_than_kayaks_l552_55258


namespace total_trees_in_gray_areas_l552_55289

theorem total_trees_in_gray_areas (x y : ℕ) (h1 : 82 + x = 100) (h2 : 82 + y = 90) :
  x + y = 26 :=
by
  sorry

end total_trees_in_gray_areas_l552_55289


namespace factor_polynomial_l552_55286

theorem factor_polynomial (n : ℕ) (hn : 2 ≤ n) 
  (a : ℝ) (b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  ∃ k : ℤ, n < 2 * k + 1 ∧ 2 * k + 1 < 3 * n ∧ 
  a = (-(2 * Real.cos ((2 * k + 1 : ℝ) * Real.pi / (2 * n)))) ^ (2 * n / (2 * n - 1)) ∧ 
  b = (2 * Real.cos ((2 * k + 1 : ℝ) * Real.pi / (2 * n))) ^ (2 / (2 * n - 1)) := sorry

end factor_polynomial_l552_55286


namespace maximum_acute_triangles_from_four_points_l552_55215

-- Define a point in a plane
structure Point (α : Type) := (x : α) (y : α)

-- Definition of an acute triangle is intrinsic to the problem
def is_acute_triangle {α : Type} [LinearOrderedField α] (A B C : Point α) : Prop :=
  sorry -- Assume implementation for determining if a triangle is acute angles based

def maximum_number_acute_triangles {α : Type} [LinearOrderedField α] (A B C D : Point α) : ℕ :=
  sorry -- Assume implementation for verifying maximum number of acute triangles from four points

theorem maximum_acute_triangles_from_four_points {α : Type} [LinearOrderedField α] (A B C D : Point α) :
  maximum_number_acute_triangles A B C D = 4 :=
  sorry

end maximum_acute_triangles_from_four_points_l552_55215


namespace lisa_total_spoons_l552_55209

def number_of_baby_spoons (num_children num_spoons_per_child : Nat) : Nat :=
  num_children * num_spoons_per_child

def number_of_decorative_spoons : Nat := 2

def number_of_old_spoons (baby_spoons decorative_spoons : Nat) : Nat :=
  baby_spoons + decorative_spoons
  
def number_of_new_spoons (large_spoons teaspoons : Nat) : Nat :=
  large_spoons + teaspoons

def total_number_of_spoons (old_spoons new_spoons : Nat) : Nat :=
  old_spoons + new_spoons

theorem lisa_total_spoons
  (children : Nat)
  (spoons_per_child : Nat)
  (large_spoons : Nat)
  (teaspoons : Nat)
  (children_eq : children = 4)
  (spoons_per_child_eq : spoons_per_child = 3)
  (large_spoons_eq : large_spoons = 10)
  (teaspoons_eq : teaspoons = 15)
  : total_number_of_spoons (number_of_old_spoons (number_of_baby_spoons children spoons_per_child) number_of_decorative_spoons) (number_of_new_spoons large_spoons teaspoons) = 39 :=
by
  sorry

end lisa_total_spoons_l552_55209


namespace find_m_l552_55217

def f (x : ℝ) (m : ℝ) : ℝ := x^2 - 3*x + m
def g (x : ℝ) (m : ℝ) : ℝ := x^2 - 3*x + 5*m

theorem find_m (m : ℝ) : 3 * f 4 m = g 4 m → m = 4 :=
by 
  sorry

end find_m_l552_55217


namespace sum_of_edges_l552_55242

-- Define the properties of the rectangular solid
variables (a b c : ℝ)
variables (V : ℝ) (S : ℝ)

-- Set the conditions
def geometric_progression := (a * b * c = V) ∧ (2 * (a * b + b * c + c * a) = S) ∧ (∃ k : ℝ, k ≠ 0 ∧ a = b / k ∧ c = b * k)

-- Define the main proof statement
theorem sum_of_edges (hV : V = 1000) (hS : S = 600) (hg : geometric_progression a b c V S) : 
  4 * (a + b + c) = 120 :=
sorry

end sum_of_edges_l552_55242


namespace qualified_light_bulb_prob_l552_55267

def prob_factory_A := 0.7
def prob_factory_B := 0.3
def qual_rate_A := 0.9
def qual_rate_B := 0.8

theorem qualified_light_bulb_prob :
  prob_factory_A * qual_rate_A + prob_factory_B * qual_rate_B = 0.87 :=
by
  sorry

end qualified_light_bulb_prob_l552_55267


namespace minimum_value_inequality_l552_55212

theorem minimum_value_inequality
  (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 2 * x + y - 3 = 0) :
  ∃ t : ℝ, (∀ (x y : ℝ), (2 * x + y = 3) → (0 < x) → (0 < y) → (t = (4 * y - x + 6) / (x * y)) → 9 ≤ t) ∧
          (∃ (x_ y_: ℝ), 2 * x_ + y_ = 3 ∧ 0 < x_ ∧ 0 < y_ ∧ (4 * y_ - x_ + 6) / (x_ * y_) = 9) :=
sorry

end minimum_value_inequality_l552_55212


namespace black_pens_removed_l552_55235

theorem black_pens_removed (initial_blue : ℕ) (initial_black : ℕ) (initial_red : ℕ)
    (blue_removed : ℕ) (pens_left : ℕ)
    (h_initial_pens : initial_blue = 9 ∧ initial_black = 21 ∧ initial_red = 6)
    (h_blue_removed : blue_removed = 4)
    (h_pens_left : pens_left = 25) :
    initial_blue + initial_black + initial_red - blue_removed - (initial_blue + initial_black + initial_red - blue_removed - pens_left) = 7 :=
by
  rcases h_initial_pens with ⟨h_ib, h_ibl, h_ir⟩
  simp [h_ib, h_ibl, h_ir, h_blue_removed, h_pens_left]
  sorry

end black_pens_removed_l552_55235


namespace width_of_larger_cuboid_l552_55243

theorem width_of_larger_cuboid
    (length_larger : ℝ)
    (width_larger : ℝ)
    (height_larger : ℝ)
    (length_smaller : ℝ)
    (width_smaller : ℝ)
    (height_smaller : ℝ)
    (num_smaller : ℕ)
    (volume_larger : ℝ)
    (volume_smaller : ℝ)
    (divided_into : Real) :
    length_larger = 12 → height_larger = 10 →
    length_smaller = 5 → width_smaller = 3 → height_smaller = 2 →
    num_smaller = 56 →
    volume_smaller = length_smaller * width_smaller * height_smaller →
    volume_larger = num_smaller * volume_smaller →
    volume_larger = length_larger * width_larger * height_larger →
    divided_into = volume_larger / (length_larger * height_larger) →
    width_larger = 14 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8 h9
  sorry

end width_of_larger_cuboid_l552_55243


namespace total_students_in_school_l552_55241

theorem total_students_in_school 
  (below_8_percent : ℝ) (above_8_ratio : ℝ) (students_8 : ℕ) : 
  below_8_percent = 0.20 → above_8_ratio = 2/3 → students_8 = 12 → 
  (∃ T : ℕ, T = 25) :=
by
  sorry

end total_students_in_school_l552_55241


namespace find_n_values_l552_55233

theorem find_n_values (n : ℕ) (h1 : 0 < n) : 
  (∃ (a : ℕ), n * 2^n + 1 = a * a) ↔ (n = 2 ∨ n = 3) := 
by
  sorry

end find_n_values_l552_55233


namespace obtain_2020_from_20_and_21_l552_55287

theorem obtain_2020_from_20_and_21 :
  ∃ (a b : ℕ), 20 * a + 21 * b = 2020 :=
by
  -- We only need to construct the proof goal, leaving the proof itself out.
  sorry

end obtain_2020_from_20_and_21_l552_55287


namespace max_value_when_a_zero_range_of_a_for_one_zero_l552_55245

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x - (1 / x) - (a + 1) * Real.log x

theorem max_value_when_a_zero :
  ∃ x : ℝ, x > 0 ∧ f 0 x = -1 :=
by sorry

theorem range_of_a_for_one_zero :
  ∀ a : ℝ, (∃ x : ℝ, x > 0 ∧ f a x = 0) ↔ a > 0 :=
by sorry

end max_value_when_a_zero_range_of_a_for_one_zero_l552_55245


namespace paperboy_problem_l552_55270

noncomputable def delivery_ways (n : ℕ) : ℕ :=
  if n = 0 then 0
  else if n = 1 then 2
  else if n = 2 then 4
  else if n = 3 then 8
  else if n = 4 then 15
  else delivery_ways (n - 1) + delivery_ways (n - 2) + delivery_ways (n - 3) + delivery_ways (n - 4)

theorem paperboy_problem : delivery_ways 12 = 2872 :=
  sorry

end paperboy_problem_l552_55270


namespace odd_function_neg_expression_l552_55220

theorem odd_function_neg_expression (f : ℝ → ℝ) (h₀ : ∀ x > 0, f x = x^3 + x + 1)
    (h₁ : ∀ x, f (-x) = -f x) : ∀ x < 0, f x = x^3 + x - 1 :=
by
  sorry

end odd_function_neg_expression_l552_55220


namespace Oshea_needs_50_small_planters_l552_55221

structure Planter :=
  (large : ℕ)     -- Number of large planters
  (medium : ℕ)    -- Number of medium planters
  (small : ℕ)     -- Number of small planters
  (capacity_large : ℕ := 20) -- Capacity of large planter
  (capacity_medium : ℕ := 10) -- Capacity of medium planter
  (capacity_small : ℕ := 4)  -- Capacity of small planter

structure Seeds :=
  (basil : ℕ)     -- Number of basil seeds
  (cilantro : ℕ)  -- Number of cilantro seeds
  (parsley : ℕ)   -- Number of parsley seeds

noncomputable def small_planters_needed (planters : Planter) (seeds : Seeds) : ℕ :=
  let basil_in_large := min seeds.basil (planters.large * planters.capacity_large)
  let basil_left := seeds.basil - basil_in_large
  let basil_in_medium := min basil_left (planters.medium * planters.capacity_medium)
  let basil_remaining := basil_left - basil_in_medium
  
  let cilantro_in_medium := min seeds.cilantro ((planters.medium * planters.capacity_medium) - basil_in_medium)
  let cilantro_remaining := seeds.cilantro - cilantro_in_medium
  
  let parsley_total := seeds.parsley + basil_remaining + cilantro_remaining
  parsley_total / planters.capacity_small

theorem Oshea_needs_50_small_planters :
  small_planters_needed 
    { large := 4, medium := 8, small := 0 }
    { basil := 200, cilantro := 160, parsley := 120 } = 50 := 
sorry

end Oshea_needs_50_small_planters_l552_55221


namespace subsets_neither_A_nor_B_l552_55226

def U : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8}
def A : Finset ℕ := {1, 2, 3, 4, 5}
def B : Finset ℕ := {4, 5, 6, 7, 8}

theorem subsets_neither_A_nor_B : 
  (U.powerset.card - A.powerset.card - B.powerset.card + (A ∩ B).powerset.card) = 196 := by 
  sorry

end subsets_neither_A_nor_B_l552_55226


namespace percent_with_university_diploma_l552_55238

theorem percent_with_university_diploma (a b c d : ℝ) (h1 : a = 0.12) (h2 : b = 0.25) (h3 : c = 0.40) 
    (h4 : d = c - a) (h5 : ¬c = 1) : 
    d + (b * (1 - c)) = 0.43 := 
by 
    sorry

end percent_with_university_diploma_l552_55238


namespace half_angle_quadrant_l552_55232

theorem half_angle_quadrant
  (α : ℝ)
  (h1 : ∃ k : ℤ, 2 * k * Real.pi + Real.pi < α ∧ α < 2 * k * Real.pi + 3 * Real.pi / 2)
  (h2 : |Real.cos (α / 2)| = -Real.cos (α / 2)) :
  ∃ k : ℤ, k * Real.pi / 2 < α / 2 ∧ α / 2 < k * Real.pi * 3 / 4 ∧ Real.cos (α / 2) ≤ 0 := sorry

end half_angle_quadrant_l552_55232


namespace ratio_of_b_to_c_l552_55273

theorem ratio_of_b_to_c (a b c : ℝ) 
  (h1 : a / b = 11 / 3) 
  (h2 : a / c = 0.7333333333333333) : 
  b / c = 1 / 5 := 
by
  sorry

end ratio_of_b_to_c_l552_55273


namespace relationship_between_M_and_N_l552_55256

variable (x y : ℝ)

theorem relationship_between_M_and_N (h1 : x ≠ 3) (h2 : y ≠ -2)
  (M : ℝ) (hm : M = x^2 + y^2 - 6 * x + 4 * y)
  (N : ℝ) (hn : N = -13) : M > N :=
by
  sorry

end relationship_between_M_and_N_l552_55256


namespace aniyah_more_candles_l552_55234

theorem aniyah_more_candles (x : ℝ) (h1 : 4 + 4 * x = 14) : x = 2.5 :=
sorry

end aniyah_more_candles_l552_55234


namespace zero_descriptions_l552_55231

-- Defining the descriptions of zero satisfying the given conditions.
def description1 : String := "The number corresponding to the origin on the number line."
def description2 : String := "The number that represents nothing."
def description3 : String := "The number that, when multiplied by any other number, equals itself."

-- Lean statement to prove the validity of the descriptions.
theorem zero_descriptions : 
  description1 = "The number corresponding to the origin on the number line." ∧
  description2 = "The number that represents nothing." ∧
  description3 = "The number that, when multiplied by any other number, equals itself." :=
by
  -- Proof omitted
  sorry

end zero_descriptions_l552_55231


namespace plane_distance_l552_55261

theorem plane_distance (D : ℕ) (h₁ : D / 300 + D / 400 = 7) : D = 1200 :=
sorry

end plane_distance_l552_55261


namespace thirty_five_million_in_scientific_notation_l552_55285

def million := 10^6

def sales_revenue (x : ℝ) := x * million

theorem thirty_five_million_in_scientific_notation :
  sales_revenue 35 = 3.5 * 10^7 :=
by
  sorry

end thirty_five_million_in_scientific_notation_l552_55285
