import Mathlib

namespace inscribed_sphere_radius_l923_92350

/-- Represents a conical flask with an inscribed sphere -/
structure ConicalFlask where
  base_radius : ℝ
  height : ℝ
  liquid_height : ℝ
  sphere_radius : ℝ

/-- Checks if the sphere is properly inscribed in the flask -/
def is_properly_inscribed (flask : ConicalFlask) : Prop :=
  flask.sphere_radius > 0 ∧
  flask.sphere_radius ≤ flask.base_radius ∧
  flask.sphere_radius + flask.liquid_height ≤ flask.height

/-- The main theorem about the inscribed sphere's radius -/
theorem inscribed_sphere_radius 
  (flask : ConicalFlask)
  (h_base : flask.base_radius = 15)
  (h_height : flask.height = 30)
  (h_liquid : flask.liquid_height = 10)
  (h_inscribed : is_properly_inscribed flask) :
  flask.sphere_radius = 10 :=
sorry

end inscribed_sphere_radius_l923_92350


namespace initial_water_amount_l923_92355

theorem initial_water_amount (initial_amount : ℝ) : 
  initial_amount + 6.8 = 9.8 → initial_amount = 3 := by
  sorry

end initial_water_amount_l923_92355


namespace odd_periodic_function_property_l923_92390

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

theorem odd_periodic_function_property (f : ℝ → ℝ) 
  (h_odd : is_odd f)
  (h_periodic : has_period f 5)
  (h1 : f 1 = 1)
  (h2 : f 2 = 2) :
  f 3 - f 4 = -1 := by
sorry

end odd_periodic_function_property_l923_92390


namespace set_size_from_average_change_l923_92325

theorem set_size_from_average_change (S : Finset ℝ) (initial_avg final_avg : ℝ) :
  initial_avg = (S.sum id) / S.card →
  final_avg = ((S.sum id) + 6) / S.card →
  initial_avg = 6.2 →
  final_avg = 6.8 →
  S.card = 10 := by
  sorry

end set_size_from_average_change_l923_92325


namespace a_33_mod_77_l923_92372

/-- Defines a_n as the large integer formed by concatenating integers from 1 to n -/
def a (n : ℕ) : ℕ :=
  -- Definition of a_n goes here
  sorry

/-- The remainder when a_33 is divided by 77 is 22 -/
theorem a_33_mod_77 : a 33 % 77 = 22 := by
  sorry

end a_33_mod_77_l923_92372


namespace paving_cost_l923_92377

/-- The cost of paving a rectangular floor -/
theorem paving_cost (length width rate : ℝ) (h1 : length = 5.5) (h2 : width = 4) (h3 : rate = 750) :
  length * width * rate = 16500 := by
  sorry

end paving_cost_l923_92377


namespace tile_arrangements_l923_92321

/-- The number of distinguishable arrangements of tiles -/
def num_arrangements (orange purple blue red : ℕ) : ℕ :=
  Nat.factorial (orange + purple + blue + red) /
  (Nat.factorial orange * Nat.factorial purple * Nat.factorial blue * Nat.factorial red)

/-- Theorem stating that the number of distinguishable arrangements
    of 2 orange, 1 purple, 3 blue, and 2 red tiles is 1680 -/
theorem tile_arrangements :
  num_arrangements 2 1 3 2 = 1680 := by
  sorry

end tile_arrangements_l923_92321


namespace sum_of_digits_94_eights_times_94_sevens_l923_92363

/-- Represents a number with 94 repeated digits --/
def RepeatedDigit (d : ℕ) : ℕ := 
  d * (10^94 - 1) / 9

/-- Sum of digits of a natural number --/
def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sumOfDigits (n / 10)

/-- The main theorem --/
theorem sum_of_digits_94_eights_times_94_sevens : 
  sumOfDigits (RepeatedDigit 8 * RepeatedDigit 7) = 1034 := by
  sorry

end sum_of_digits_94_eights_times_94_sevens_l923_92363


namespace dance_studios_total_l923_92308

/-- The total number of students in three dance studios -/
def total_students (studio1 studio2 studio3 : ℕ) : ℕ :=
  studio1 + studio2 + studio3

/-- Theorem: The total number of students in three specific dance studios is 376 -/
theorem dance_studios_total : total_students 110 135 131 = 376 := by
  sorry

end dance_studios_total_l923_92308


namespace absolute_value_equality_l923_92358

theorem absolute_value_equality (x : ℝ) : |x - 3| = |x - 5| → x = 4 := by
  sorry

end absolute_value_equality_l923_92358


namespace system_solution_l923_92393

theorem system_solution :
  let f (x y z : ℝ) := x^2 = 2 * Real.sqrt (y^2 + 1) ∧
                       y^2 = 2 * Real.sqrt (z^2 - 1) - 2 ∧
                       z^2 = 4 * Real.sqrt (x^2 + 2) - 6
  (∀ x y z : ℝ, f x y z ↔ 
    ((x = Real.sqrt 2 ∧ y = 0 ∧ z = Real.sqrt 2) ∨
     (x = Real.sqrt 2 ∧ y = 0 ∧ z = -Real.sqrt 2) ∨
     (x = -Real.sqrt 2 ∧ y = 0 ∧ z = Real.sqrt 2) ∨
     (x = -Real.sqrt 2 ∧ y = 0 ∧ z = -Real.sqrt 2))) :=
by sorry

end system_solution_l923_92393


namespace product_evaluation_l923_92342

theorem product_evaluation (m : ℤ) (h : m = 3) : 
  (m - 2) * (m - 1) * m * (m + 1) * (m + 2) * (m + 3) = 720 := by
  sorry

end product_evaluation_l923_92342


namespace smallest_b_for_integer_solutions_l923_92318

theorem smallest_b_for_integer_solutions : 
  ∃ (b : ℕ), b > 0 ∧ 
  (∀ (x : ℤ), x^2 + b*x = -21 → ∃ (y : ℤ), x = y) ∧
  (∀ (b' : ℕ), 0 < b' ∧ b' < b → 
    ∃ (x : ℝ), x^2 + b'*x = -21 ∧ ¬∃ (y : ℤ), x = y) ∧
  b = 10 := by
sorry

end smallest_b_for_integer_solutions_l923_92318


namespace unique_valid_number_l923_92362

def is_valid_number (n : Fin 10 → Nat) : Prop :=
  (∀ i : Fin 8, n i * n (i + 1) * n (i + 2) = 24) ∧
  n 4 = 2 ∧
  n 8 = 3

theorem unique_valid_number :
  ∃! n : Fin 10 → Nat, is_valid_number n ∧ 
    (∀ i : Fin 10, n i = ([4, 2, 3, 4, 2, 3, 4, 2, 3, 4] : List Nat)[i]) :=
by sorry

end unique_valid_number_l923_92362


namespace correct_match_probability_l923_92327

theorem correct_match_probability (n : Nat) (h : n = 4) :
  (1 : ℚ) / n.factorial = (1 : ℚ) / 24 := by
  sorry

#check correct_match_probability

end correct_match_probability_l923_92327


namespace profit_percentage_10_12_l923_92353

/-- Calculates the profit percentage when selling n articles at the cost price of m articles -/
def profit_percentage (n m : ℕ) : ℚ :=
  ((m : ℚ) - (n : ℚ)) / (n : ℚ) * 100

/-- Theorem: The profit percentage when selling 10 articles at the cost price of 12 articles is 20% -/
theorem profit_percentage_10_12 : profit_percentage 10 12 = 20 := by
  sorry

end profit_percentage_10_12_l923_92353


namespace prime_divisibility_l923_92395

theorem prime_divisibility (a b p q : ℕ) : 
  a > 0 → b > 0 → Prime p → Prime q → 
  ¬(p ∣ q - 1) → (q ∣ a^p - b^p) → (q ∣ a - b) :=
by sorry

end prime_divisibility_l923_92395


namespace spring_length_theorem_l923_92322

/-- Represents the relationship between spring length and attached mass -/
def spring_length (x : ℝ) : ℝ :=
  0.3 * x + 6

/-- Theorem stating the relationship between spring length and attached mass -/
theorem spring_length_theorem (x : ℝ) :
  let initial_length : ℝ := 6
  let extension_rate : ℝ := 0.3
  spring_length x = initial_length + extension_rate * x :=
by
  sorry

#check spring_length_theorem

end spring_length_theorem_l923_92322


namespace absolute_value_equals_sqrt_of_square_l923_92360

theorem absolute_value_equals_sqrt_of_square (x : ℝ) : |x| = Real.sqrt (x^2) := by
  sorry

end absolute_value_equals_sqrt_of_square_l923_92360


namespace weight_of_water_moles_l923_92320

/-- The atomic weight of hydrogen in g/mol -/
def atomic_weight_H : ℝ := 1.008

/-- The atomic weight of oxygen in g/mol -/
def atomic_weight_O : ℝ := 16.00

/-- The number of hydrogen atoms in a water molecule -/
def H_count : ℕ := 2

/-- The number of oxygen atoms in a water molecule -/
def O_count : ℕ := 1

/-- The number of moles of water -/
def moles_of_water : ℝ := 4

/-- The molecular weight of water (H2O) in g/mol -/
def molecular_weight_H2O : ℝ := H_count * atomic_weight_H + O_count * atomic_weight_O

theorem weight_of_water_moles : 
  moles_of_water * molecular_weight_H2O = 72.064 := by sorry

end weight_of_water_moles_l923_92320


namespace jungkook_boxes_l923_92380

/-- The number of boxes needed to hold a given number of balls -/
def boxes_needed (total_balls : ℕ) (balls_per_box : ℕ) : ℕ :=
  (total_balls + balls_per_box - 1) / balls_per_box

theorem jungkook_boxes (total_balls : ℕ) (balls_per_box : ℕ) 
  (h1 : total_balls = 10) (h2 : balls_per_box = 5) : 
  boxes_needed total_balls balls_per_box = 2 := by
sorry

end jungkook_boxes_l923_92380


namespace largest_six_digit_divisible_by_41_l923_92338

theorem largest_six_digit_divisible_by_41 : 
  ∀ n : ℕ, n ≤ 999999 ∧ n % 41 = 0 → n ≤ 999990 :=
by sorry

end largest_six_digit_divisible_by_41_l923_92338


namespace floor_ceil_sum_l923_92384

theorem floor_ceil_sum : ⌊(-3.67 : ℝ)⌋ + ⌈(34.2 : ℝ)⌉ = 31 := by sorry

end floor_ceil_sum_l923_92384


namespace necessary_but_not_sufficient_l923_92356

theorem necessary_but_not_sufficient (x : ℝ) :
  (∀ x, (1/2)^x > 1 → 1/x < 1) ∧
  ¬(∀ x, 1/x < 1 → (1/2)^x > 1) :=
by sorry

end necessary_but_not_sufficient_l923_92356


namespace triangle_area_ratio_l923_92344

/-- Given a right triangle with a point on its hypotenuse and lines drawn parallel to the legs,
    dividing it into a rectangle and two smaller right triangles, this theorem states the
    relationship between the areas of the smaller triangles and the rectangle. -/
theorem triangle_area_ratio (a b m : ℝ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_m : m > 0) :
  let rectangle_area := a * b
  let small_triangle1_area := m * rectangle_area
  let small_triangle2_area := (b^2) / (4 * m)
  (small_triangle2_area / rectangle_area) = b / (4 * m * a) := by
  sorry

end triangle_area_ratio_l923_92344


namespace absolute_value_equality_l923_92369

theorem absolute_value_equality (a b : ℝ) : 
  (∀ x y : ℝ, |a * x + b * y| + |b * x + a * y| = |x| + |y|) ↔ 
  ((a = 1 ∧ b = 0) ∨ (a = 0 ∧ b = 1) ∨ (a = 0 ∧ b = -1) ∨ (a = -1 ∧ b = 0)) :=
by sorry

end absolute_value_equality_l923_92369


namespace perfect_square_equation_l923_92347

theorem perfect_square_equation (a b c : ℤ) 
  (h : (a - 5)^2 + (b - 12)^2 - (c - 13)^2 = a^2 + b^2 - c^2) : 
  ∃ k : ℤ, a^2 + b^2 - c^2 = k^2 := by
  sorry

end perfect_square_equation_l923_92347


namespace product_of_five_consecutive_integers_divisibility_l923_92373

theorem product_of_five_consecutive_integers_divisibility 
  (m : ℤ) 
  (k : ℤ) 
  (h1 : m = k * (k + 1) * (k + 2) * (k + 3) * (k + 4)) 
  (h2 : 11 ∣ m) : 
  (10 ∣ m) ∧ (22 ∣ m) ∧ (33 ∣ m) ∧ (55 ∣ m) ∧ ¬(∀ m, 66 ∣ m) :=
by sorry

end product_of_five_consecutive_integers_divisibility_l923_92373


namespace cube_root_of_product_l923_92354

theorem cube_root_of_product (a b c : ℕ) :
  (2^9 * 5^3 * 7^6 : ℝ)^(1/3) = 1960 := by
  sorry

end cube_root_of_product_l923_92354


namespace produce_worth_l923_92386

theorem produce_worth (asparagus_bundles : ℕ) (asparagus_price : ℚ)
                      (grape_boxes : ℕ) (grape_price : ℚ)
                      (apple_count : ℕ) (apple_price : ℚ) :
  asparagus_bundles = 60 ∧ asparagus_price = 3 ∧
  grape_boxes = 40 ∧ grape_price = 5/2 ∧
  apple_count = 700 ∧ apple_price = 1/2 →
  asparagus_bundles * asparagus_price +
  grape_boxes * grape_price +
  apple_count * apple_price = 630 :=
by sorry

end produce_worth_l923_92386


namespace cross_section_distance_l923_92307

/-- Represents a right hexagonal pyramid -/
structure RightHexagonalPyramid where
  /-- Height of the pyramid -/
  height : ℝ
  /-- Side length of the base hexagon -/
  base_side : ℝ

/-- Represents a cross section of the pyramid -/
structure CrossSection where
  /-- Distance from the apex of the pyramid -/
  distance : ℝ
  /-- Area of the cross section -/
  area : ℝ

/-- 
Theorem: In a right hexagonal pyramid, if two cross sections parallel to the base 
have areas of 300√3 sq ft and 675√3 sq ft, and these planes are 12 feet apart, 
then the distance from the apex to the larger cross section is 36 feet.
-/
theorem cross_section_distance 
  (pyramid : RightHexagonalPyramid) 
  (cs1 cs2 : CrossSection) 
  (h_area1 : cs1.area = 300 * Real.sqrt 3)
  (h_area2 : cs2.area = 675 * Real.sqrt 3)
  (h_distance : cs2.distance - cs1.distance = 12)
  (h_order : cs1.distance < cs2.distance) :
  cs2.distance = 36 := by
  sorry

end cross_section_distance_l923_92307


namespace range_of_c_l923_92388

open Real

theorem range_of_c (c : ℝ) : 
  (∀ x y : ℝ, x < y → c^x > c^y) →  -- y = c^x is decreasing
  (∃ x : ℝ, x^2 - Real.sqrt 2 * x + c ≤ 0) →  -- negation of q
  (0 < c ∧ c < 1) →  -- derived from decreasing function condition
  0 < c ∧ c ≤ (1/2) := by
sorry

end range_of_c_l923_92388


namespace coin_value_equality_l923_92341

theorem coin_value_equality (m : ℕ) : 
  (25 : ℕ) * 25 + 15 * 10 = m * 25 + 40 * 10 → m = 15 := by
  sorry

end coin_value_equality_l923_92341


namespace family_size_l923_92316

theorem family_size (boys girls : ℕ) 
  (sister_condition : boys = girls - 1)
  (brother_condition : girls = 2 * (boys - 1)) : 
  boys + girls = 7 := by
sorry

end family_size_l923_92316


namespace fraction_equality_l923_92389

theorem fraction_equality (a b : ℝ) : |a^2 - b^2| / |(a - b)^2| = |a + b| / |a - b| := by
  sorry

end fraction_equality_l923_92389


namespace bird_weight_equations_l923_92317

/-- Represents the weight of birds in jin -/
structure BirdWeight where
  sparrow : ℝ
  swallow : ℝ

/-- The total weight of 5 sparrows and 6 swallows is 1 jin -/
def total_weight (w : BirdWeight) : Prop :=
  5 * w.sparrow + 6 * w.swallow = 1

/-- Sparrows are heavier than swallows -/
def sparrow_heavier (w : BirdWeight) : Prop :=
  w.sparrow > w.swallow

/-- Exchanging one sparrow with one swallow doesn't change the total weight -/
def exchange_weight (w : BirdWeight) : Prop :=
  4 * w.sparrow + 7 * w.swallow = 5 * w.swallow + w.sparrow

/-- The system of equations correctly represents the bird weight problem -/
theorem bird_weight_equations (w : BirdWeight) 
  (h1 : total_weight w) 
  (h2 : sparrow_heavier w) 
  (h3 : exchange_weight w) : 
  5 * w.sparrow + 6 * w.swallow = 1 ∧ 3 * w.sparrow = -2 * w.swallow := by
  sorry

end bird_weight_equations_l923_92317


namespace problem_1_problem_2_l923_92396

theorem problem_1 (m n : ℝ) : 
  (∀ x, (x - 3) * (x - 4) = x^2 + m*x + n) → m = -7 ∧ n = 12 := by sorry

theorem problem_2 (a b : ℝ) :
  (∀ x, (x + a) * (x + b) = x^2 - 3*x + 1/3) → 
  ((a - 1) * (b - 1) = 13/3) ∧ (1/a^2 + 1/b^2 = 75) := by sorry

end problem_1_problem_2_l923_92396


namespace coin_problem_l923_92309

/-- Given a total of 12 coins consisting of quarters and nickels with a total value of 220 cents, 
    prove that the number of nickels is 4. -/
theorem coin_problem (q n : ℕ) : 
  q + n = 12 → 
  25 * q + 5 * n = 220 → 
  n = 4 := by
  sorry

end coin_problem_l923_92309


namespace range_of_m_l923_92335

def f (a x : ℝ) : ℝ := x^2 - 2*a*x + 1

def set_A : Set ℝ := {a | -1 < a ∧ a < 1}

def set_B (m : ℝ) : Set ℝ := {a | m < a ∧ a < m + 3}

theorem range_of_m (m : ℝ) :
  (∀ x, x ∈ set_A → x ∈ set_B m) ∧
  (∃ x, x ∈ set_B m ∧ x ∉ set_A) →
  -2 ≤ m ∧ m ≤ -1 :=
sorry

end range_of_m_l923_92335


namespace at_least_one_non_negative_l923_92345

theorem at_least_one_non_negative 
  (a b c d e f g h : ℝ) : 
  (max (ac + bd) (max (ae + bf) (max (ag + bh) (max (ce + df) (max (cg + dh) (eg + fh)))))) ≥ 0 :=
by sorry

end at_least_one_non_negative_l923_92345


namespace certain_number_problem_l923_92301

theorem certain_number_problem (x : ℝ) : x * 11 = 99 → x = 9 := by
  sorry

end certain_number_problem_l923_92301


namespace sum_of_x_and_y_equals_two_l923_92336

theorem sum_of_x_and_y_equals_two (x y : ℝ) (h : x^2 + y^2 = 8*x - 4*y - 20) : x + y = 2 := by
  sorry

end sum_of_x_and_y_equals_two_l923_92336


namespace remainder_x14_minus_1_div_x_plus_1_l923_92374

theorem remainder_x14_minus_1_div_x_plus_1 (x : ℝ) : 
  (x^14 - 1) % (x + 1) = 0 := by sorry

end remainder_x14_minus_1_div_x_plus_1_l923_92374


namespace roses_cut_l923_92340

/-- The number of roses Jessica cut is equal to the difference between the final number of roses in the vase and the initial number of roses in the vase. -/
theorem roses_cut (initial_roses final_roses : ℕ) (h : initial_roses = 2 ∧ final_roses = 23) :
  final_roses - initial_roses = 21 := by
  sorry

end roses_cut_l923_92340


namespace min_value_of_f_l923_92339

/-- The quadratic function f(x) = 2x^2 + 8x + 7 -/
def f (x : ℝ) : ℝ := 2 * x^2 + 8 * x + 7

/-- Theorem: The minimum value of f(x) = 2x^2 + 8x + 7 is -1 -/
theorem min_value_of_f :
  ∃ (min : ℝ), min = -1 ∧ ∀ (x : ℝ), f x ≥ min :=
by sorry

end min_value_of_f_l923_92339


namespace wendi_chicken_count_l923_92382

/-- The number of chickens Wendi has after a series of events -/
def final_chicken_count (initial : ℕ) (doubled : ℕ) (lost : ℕ) (found : ℕ) : ℕ :=
  initial + doubled - lost + found

/-- Theorem stating the final number of chickens Wendi has -/
theorem wendi_chicken_count : 
  final_chicken_count 4 4 1 6 = 13 := by sorry

end wendi_chicken_count_l923_92382


namespace slope_of_line_from_equation_l923_92334

theorem slope_of_line_from_equation (x₁ x₂ y₁ y₂ : ℝ) 
  (h₁ : x₁ ≠ x₂) 
  (h₂ : 4 / x₁ + 5 / y₁ = 0) 
  (h₃ : 4 / x₂ + 5 / y₂ = 0) : 
  (y₂ - y₁) / (x₂ - x₁) = -5/4 := by
  sorry

end slope_of_line_from_equation_l923_92334


namespace shortest_path_in_room_l923_92392

theorem shortest_path_in_room (a b h : ℝ) 
  (ha : a = 7) (hb : b = 8) (hh : h = 4) : 
  let diagonal := Real.sqrt (a^2 + b^2 + h^2)
  let floor_path := Real.sqrt ((a^2 + b^2) + h^2)
  diagonal ≥ floor_path ∧ floor_path = Real.sqrt 265 := by
  sorry

end shortest_path_in_room_l923_92392


namespace specific_arrangement_probability_l923_92348

def total_lamps : ℕ := 8
def red_lamps : ℕ := 4
def blue_lamps : ℕ := 4
def lamps_on : ℕ := 4

def ways_to_arrange_colors : ℕ := Nat.choose total_lamps red_lamps
def ways_to_turn_on : ℕ := Nat.choose total_lamps lamps_on

def remaining_positions : ℕ := 5
def remaining_red : ℕ := 3
def remaining_blue : ℕ := 2
def remaining_on : ℕ := 2

def ways_to_arrange_remaining : ℕ := Nat.choose remaining_positions remaining_red
def ways_to_turn_on_remaining : ℕ := Nat.choose remaining_positions remaining_on

theorem specific_arrangement_probability :
  (ways_to_arrange_remaining * ways_to_turn_on_remaining : ℚ) / 
  (ways_to_arrange_colors * ways_to_turn_on) = 1 / 49 := by
  sorry

end specific_arrangement_probability_l923_92348


namespace factorial_division_l923_92337

-- Define factorial function
def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

-- Theorem statement
theorem factorial_division : 
  factorial 8 / factorial (8 - 2) = 56 := by
  sorry

end factorial_division_l923_92337


namespace equation_solution_l923_92329

theorem equation_solution : 
  ∃! x : ℚ, (x^2 + 3*x + 4) / (x + 5) = x + 6 :=
by
  use -13/4
  sorry

end equation_solution_l923_92329


namespace sum_of_roots_for_f_l923_92357

def f (x : ℝ) : ℝ := (3*x)^2 + 2*(3*x) + 1

theorem sum_of_roots_for_f (z : ℝ) : 
  (∃ z₁ z₂, f z₁ = 13 ∧ f z₂ = 13 ∧ z₁ ≠ z₂ ∧ z₁ + z₂ = -2/9) :=
sorry

end sum_of_roots_for_f_l923_92357


namespace parabola_directrix_l923_92359

/-- Given a parabola y = 3x^2 - 6x + 2, its directrix is y = -13/12 -/
theorem parabola_directrix (x y : ℝ) :
  y = 3 * x^2 - 6 * x + 2 →
  ∃ (k : ℝ), k = -13/12 ∧ k = y - 3 * (x - 1)^2 + 1 := by
  sorry

end parabola_directrix_l923_92359


namespace second_side_bisected_l923_92312

/-- A nonagon circumscribed around a circle -/
structure CircumscribedNonagon where
  /-- The lengths of the sides of the nonagon -/
  sides : Fin 9 → ℕ
  /-- All sides have positive integer lengths -/
  all_positive : ∀ i, sides i > 0
  /-- The first and third sides have length 1 -/
  first_third_one : sides 0 = 1 ∧ sides 2 = 1

/-- The point of tangency divides the second side into two equal segments -/
theorem second_side_bisected (n : CircumscribedNonagon) :
  ∃ (x : ℚ), x = 1/2 ∧ x * n.sides 1 = (1 - x) * n.sides 1 :=
sorry

end second_side_bisected_l923_92312


namespace monkey_liar_puzzle_l923_92379

-- Define the possible characteristics
inductive Character
| Monkey
| NonMonkey

inductive Truthfulness
| TruthTeller
| Liar

-- Define a structure for an individual
structure Individual where
  species : Character
  honesty : Truthfulness

-- Define the statements made by A and B
def statement_A (a b : Individual) : Prop :=
  a.species = Character.Monkey ∧ b.species = Character.Monkey

def statement_B (a b : Individual) : Prop :=
  a.honesty = Truthfulness.Liar ∧ b.honesty = Truthfulness.Liar

-- Theorem stating the solution
theorem monkey_liar_puzzle :
  ∃ (a b : Individual),
    (statement_A a b ↔ a.honesty = Truthfulness.TruthTeller) ∧
    (statement_B a b ↔ b.honesty = Truthfulness.Liar) ∧
    a.species = Character.Monkey ∧
    b.species = Character.Monkey ∧
    a.honesty = Truthfulness.TruthTeller ∧
    b.honesty = Truthfulness.Liar :=
  sorry


end monkey_liar_puzzle_l923_92379


namespace books_from_second_shop_l923_92381

/-- Proves the number of books bought from the second shop given the conditions -/
theorem books_from_second_shop 
  (first_shop_books : ℕ) 
  (first_shop_cost : ℕ) 
  (second_shop_cost : ℕ) 
  (average_price : ℚ) 
  (h1 : first_shop_books = 65)
  (h2 : first_shop_cost = 1150)
  (h3 : second_shop_cost = 920)
  (h4 : average_price = 18) : 
  ℕ := by
  sorry

#check books_from_second_shop

end books_from_second_shop_l923_92381


namespace locus_and_fixed_points_l923_92387

-- Define the points and vectors
variable (P Q R M A S B D E F : ℝ × ℝ)
variable (a b : ℝ)

-- Define the conditions
axiom P_on_x_axis : P.2 = 0
axiom Q_on_y_axis : Q.1 = 0
axiom R_coord : R = (0, -3)
axiom S_coord : S = (0, 2)
axiom PR_dot_PM : (R.1 - P.1) * (M.1 - P.1) + (R.2 - P.2) * (M.2 - P.2) = 0
axiom PQ_half_QM : (Q.1 - P.1, Q.2 - P.2) = (1/2 : ℝ) • (M.1 - Q.1, M.2 - Q.2)
axiom A_coord : A = (a, b)
axiom A_outside_C : a ≠ 0 ∧ b ≠ 2
axiom AB_AD_tangent : True  -- This condition is implied but not directly stated
axiom E_on_line : E.2 = -2
axiom F_on_line : F.2 = -2

-- Define the locus C
def C (x y : ℝ) : Prop := x^2 = 4*y

-- State the theorem
theorem locus_and_fixed_points :
  (∀ x y, C x y ↔ x^2 = 4*y) ∧
  (∃ r : ℝ, r > 0 ∧ 
    ((E.1 + F.1)/2 - 0)^2 + ((E.2 + F.2)/2 - (-2 + 2*Real.sqrt 2))^2 = r^2 ∧
    ((E.1 + F.1)/2 - 0)^2 + ((E.2 + F.2)/2 - (-2 - 2*Real.sqrt 2))^2 = r^2) :=
sorry

end locus_and_fixed_points_l923_92387


namespace system_solution_l923_92324

theorem system_solution (x y z : ℚ) : 
  (x * y = 5 * (x + y) ∧ 
   x * z = 4 * (x + z) ∧ 
   y * z = 2 * (y + z)) → 
  ((x = 0 ∧ y = 0 ∧ z = 0) ∨ 
   (x = -40 ∧ y = 40/9 ∧ z = 40/11)) := by
sorry

end system_solution_l923_92324


namespace company_workforce_l923_92365

/-- Proves the number of employees after hiring, given initial conditions and hiring information -/
theorem company_workforce (initial_female_percentage : ℚ) 
                          (final_female_percentage : ℚ)
                          (additional_male_workers : ℕ) : ℕ :=
  let initial_female_percentage : ℚ := 60 / 100
  let final_female_percentage : ℚ := 55 / 100
  let additional_male_workers : ℕ := 30
  360

#check company_workforce

end company_workforce_l923_92365


namespace sqrt_difference_complex_expression_system_of_equations_l923_92326

-- Problem 1
theorem sqrt_difference : Real.sqrt 8 - Real.sqrt 50 = -3 * Real.sqrt 2 := by sorry

-- Problem 2
theorem complex_expression : 
  Real.sqrt 27 * Real.sqrt (1/3) - (Real.sqrt 3 - Real.sqrt 2)^2 = 2 * Real.sqrt 6 - 2 := by sorry

-- Problem 3
theorem system_of_equations :
  ∃ (x y : ℝ), x + y = 2 ∧ x + 2*y = 6 ∧ x = -2 ∧ y = 4 := by sorry

end sqrt_difference_complex_expression_system_of_equations_l923_92326


namespace concert_attendance_l923_92378

/-- Represents the number of adults attending the concert. -/
def num_adults : ℕ := sorry

/-- Represents the number of children attending the concert. -/
def num_children : ℕ := sorry

/-- The cost of an adult ticket in dollars. -/
def adult_ticket_cost : ℕ := 7

/-- The cost of a child ticket in dollars. -/
def child_ticket_cost : ℕ := 3

/-- The total revenue from ticket sales in dollars. -/
def total_revenue : ℕ := 6000

theorem concert_attendance :
  (num_children = 3 * num_adults) ∧
  (num_adults * adult_ticket_cost + num_children * child_ticket_cost = total_revenue) →
  (num_adults + num_children = 1500) := by
  sorry

end concert_attendance_l923_92378


namespace expression_evaluation_l923_92371

theorem expression_evaluation (a b : ℝ) (h1 : a = 3) (h2 : b = 2) :
  (a^2 + b)^2 - (a^2 - b)^2 + 2*a*b = 78 := by
  sorry

end expression_evaluation_l923_92371


namespace simplify_fraction_l923_92368

theorem simplify_fraction (x y : ℚ) (hx : x = 3) (hy : y = 2) :
  (12 * x^2 * y^3) / (9 * x * y^2) = 8 := by
  sorry

end simplify_fraction_l923_92368


namespace orchestra_members_count_l923_92330

theorem orchestra_members_count :
  ∃! n : ℕ, 150 < n ∧ n < 250 ∧
    n % 4 = 2 ∧
    n % 5 = 3 ∧
    n % 7 = 4 ∧
    n = 158 := by
  sorry

end orchestra_members_count_l923_92330


namespace average_of_three_numbers_l923_92323

theorem average_of_three_numbers (y : ℝ) : (14 + 23 + y) / 3 = 21 → y = 26 := by
  sorry

end average_of_three_numbers_l923_92323


namespace conic_eccentricity_l923_92306

/-- Given that 1, m, and 9 form a geometric sequence, 
    the eccentricity of the conic section x²/m + y² = 1 is either √6/3 or 2 -/
theorem conic_eccentricity (m : ℝ) : 
  (1 * 9 = m^2) →  -- geometric sequence condition
  (∃ e : ℝ, (e = Real.sqrt 6 / 3 ∨ e = 2) ∧
   ∀ x y : ℝ, x^2 / m + y^2 = 1 → 
   e = if m > 0 
       then Real.sqrt (1 - 1 / m) 
       else Real.sqrt (1 - m) / Real.sqrt (-m)) :=
by sorry

end conic_eccentricity_l923_92306


namespace fox_can_equalize_cheese_l923_92398

/-- Represents the state of the cheese pieces -/
structure CheeseState where
  piece1 : ℕ
  piece2 : ℕ
  piece3 : ℕ

/-- Represents a single cut operation by the fox -/
inductive CutOperation
  | cut12 : CutOperation  -- Cut 1g from piece1 and piece2
  | cut13 : CutOperation  -- Cut 1g from piece1 and piece3
  | cut23 : CutOperation  -- Cut 1g from piece2 and piece3

/-- Applies a single cut operation to a cheese state -/
def applyCut (state : CheeseState) (cut : CutOperation) : CheeseState :=
  match cut with
  | CutOperation.cut12 => ⟨state.piece1 - 1, state.piece2 - 1, state.piece3⟩
  | CutOperation.cut13 => ⟨state.piece1 - 1, state.piece2, state.piece3 - 1⟩
  | CutOperation.cut23 => ⟨state.piece1, state.piece2 - 1, state.piece3 - 1⟩

/-- Applies a sequence of cut operations to a cheese state -/
def applyCuts (state : CheeseState) (cuts : List CutOperation) : CheeseState :=
  cuts.foldl applyCut state

/-- The theorem to be proved -/
theorem fox_can_equalize_cheese :
  ∃ (cuts : List CutOperation),
    let finalState := applyCuts ⟨5, 8, 11⟩ cuts
    finalState.piece1 = finalState.piece2 ∧
    finalState.piece2 = finalState.piece3 ∧
    finalState.piece1 > 0 :=
  sorry


end fox_can_equalize_cheese_l923_92398


namespace consecutive_integers_average_l923_92383

theorem consecutive_integers_average (a b : ℤ) : 
  (a > 0) → 
  (b = (7 * a + 21) / 7) → 
  ((7 * b + 21) / 7 = a + 6) := by
sorry

end consecutive_integers_average_l923_92383


namespace carnival_game_days_l923_92376

def carnival_game (first_period_earnings : ℕ) (remaining_earnings : ℕ) (daily_earnings : ℕ) : Prop :=
  let first_period_days : ℕ := 20
  let remaining_days : ℕ := remaining_earnings / daily_earnings
  let total_days : ℕ := first_period_days + remaining_days
  (first_period_earnings = first_period_days * daily_earnings) ∧
  (remaining_earnings = remaining_days * daily_earnings) ∧
  (total_days = 31)

theorem carnival_game_days :
  carnival_game 120 66 6 := by
  sorry

end carnival_game_days_l923_92376


namespace brahmagupta_theorem_l923_92302

/-- An inscribed quadrilateral with side lengths a, b, c, d and diagonals p, q -/
structure InscribedQuadrilateral (a b c d p q : ℝ) : Prop where
  positive : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ 0 < p ∧ 0 < q
  inscribed : ∃ (r : ℝ), 0 < r ∧ a + c = b + d -- Condition for inscribability

/-- Brahmagupta's theorem for inscribed quadrilaterals -/
theorem brahmagupta_theorem {a b c d p q : ℝ} (quad : InscribedQuadrilateral a b c d p q) :
  p^2 + q^2 = a^2 + b^2 + c^2 + d^2 ∧ 2*p*q = a^2 + c^2 - b^2 - d^2 := by
  sorry

#check brahmagupta_theorem

end brahmagupta_theorem_l923_92302


namespace fourth_sample_is_nineteen_l923_92375

/-- Represents a systematic sampling scenario in a class. -/
structure SystematicSample where
  total_students : ℕ
  sample_size : ℕ
  known_samples : List ℕ

/-- Calculates the interval for systematic sampling. -/
def sampling_interval (s : SystematicSample) : ℕ :=
  s.total_students / s.sample_size

/-- Checks if a number is part of the systematic sample. -/
def is_in_sample (s : SystematicSample) (n : ℕ) : Prop :=
  ∃ k, n = k * sampling_interval s + (s.known_samples.head? ).getD 0

/-- The main theorem stating that 19 must be the fourth sample in the given scenario. -/
theorem fourth_sample_is_nineteen (s : SystematicSample)
    (h1 : s.total_students = 56)
    (h2 : s.sample_size = 4)
    (h3 : s.known_samples = [5, 33, 47])
    (h4 : ∀ n, is_in_sample s n → n ∈ [5, 19, 33, 47]) :
    is_in_sample s 19 :=
  sorry

#check fourth_sample_is_nineteen

end fourth_sample_is_nineteen_l923_92375


namespace inverse_of_A_l923_92352

def A : Matrix (Fin 2) (Fin 2) ℚ := !![2, 3; -1, 7]

theorem inverse_of_A :
  let A_inv : Matrix (Fin 2) (Fin 2) ℚ := !![7/17, -3/17; 1/17, 2/17]
  A * A_inv = 1 ∧ A_inv * A = 1 :=
sorry

end inverse_of_A_l923_92352


namespace polynomial_roots_sum_l923_92391

theorem polynomial_roots_sum (n : ℤ) (p q r : ℤ) : 
  (∃ (x : ℤ), x^3 - 2023*x + n = 0 ↔ x = p ∨ x = q ∨ x = r) →
  |p| + |q| + |r| = 102 := by
  sorry

end polynomial_roots_sum_l923_92391


namespace triangle_angle_proof_l923_92303

theorem triangle_angle_proof (A B : ℝ) (a b : ℝ) : 
  0 < A ∧ 0 < B ∧ A + B < π →  -- Ensuring A and B are valid triangle angles
  B = 2 * A →                  -- Given condition
  a / b = 1 / Real.sqrt 3 →    -- Given ratio of sides
  A = π / 6                    -- Conclusion (30° in radians)
  := by sorry

end triangle_angle_proof_l923_92303


namespace unit_digit_of_12_pow_100_l923_92304

-- Define the function to get the unit digit of a natural number
def unitDigit (n : ℕ) : ℕ := n % 10

-- Define the theorem
theorem unit_digit_of_12_pow_100 : unitDigit (12^100) = 6 := by
  sorry

end unit_digit_of_12_pow_100_l923_92304


namespace fibonacci_5k_divisible_by_5_l923_92397

def fibonacci : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fibonacci n + fibonacci (n + 1)

theorem fibonacci_5k_divisible_by_5 (k : ℕ) : ∃ m : ℕ, fibonacci (5 * k) = 5 * m := by
  sorry

end fibonacci_5k_divisible_by_5_l923_92397


namespace distance_to_line_l923_92300

/-- Given a line l with slope k passing through point A(0,2), and a normal vector n to l,
    prove that for any point B satisfying |n⋅AB| = |n|, the distance from B to l is 1. -/
theorem distance_to_line (k : ℝ) (n : ℝ × ℝ) (B : ℝ × ℝ) :
  let A : ℝ × ℝ := (0, 2)
  let l := {(x, y) : ℝ × ℝ | y - 2 = k * x}
  n.1 = -k ∧ n.2 = 1 →  -- n is a normal vector to l
  |n.1 * (B.1 - A.1) + n.2 * (B.2 - A.2)| = Real.sqrt (n.1^2 + n.2^2) →
  Real.sqrt ((B.1 - 0)^2 + (B.2 - (k * B.1 + 2))^2) / Real.sqrt (1 + k^2) = 1 :=
by sorry

end distance_to_line_l923_92300


namespace restaurant_bill_calculation_total_bill_is_140_l923_92364

/-- Calculates the total bill for a restaurant order with given conditions -/
theorem restaurant_bill_calculation 
  (tax_rate : ℝ) 
  (striploin_cost : ℝ) 
  (wine_cost : ℝ) 
  (gratuities : ℝ) : ℝ :=
  let total_before_tax := striploin_cost + wine_cost
  let tax_amount := tax_rate * total_before_tax
  let total_after_tax := total_before_tax + tax_amount
  let total_bill := total_after_tax + gratuities
  total_bill

/-- Proves that the total bill is $140 given the specified conditions -/
theorem total_bill_is_140 : 
  restaurant_bill_calculation 0.1 80 10 41 = 140 := by
  sorry

end restaurant_bill_calculation_total_bill_is_140_l923_92364


namespace bijection_between_sets_l923_92332

def N (n : ℕ) : ℕ := n^9 % 10000

def set_greater (m : ℕ) : Set ℕ :=
  {n : ℕ | n < m ∧ n % 2 = 1 ∧ N n > n}

def set_lesser (m : ℕ) : Set ℕ :=
  {n : ℕ | n < m ∧ n % 2 = 1 ∧ N n < n}

theorem bijection_between_sets :
  ∃ (f : set_greater 10000 → set_lesser 10000),
    Function.Bijective f :=
  sorry

end bijection_between_sets_l923_92332


namespace stuffed_animals_gcd_l923_92328

theorem stuffed_animals_gcd : Nat.gcd 26 (Nat.gcd 14 (Nat.gcd 18 22)) = 2 := by
  sorry

end stuffed_animals_gcd_l923_92328


namespace town_population_l923_92314

theorem town_population (present_population : ℝ) 
  (growth_rate : ℝ) (future_population : ℝ) : 
  growth_rate = 0.1 → 
  future_population = present_population * (1 + growth_rate) → 
  future_population = 220 → 
  present_population = 200 := by
sorry

end town_population_l923_92314


namespace derivative_of_even_function_is_odd_l923_92351

/-- A function f: ℝ → ℝ that is even, i.e., f(-x) = f(x) for all x -/
def EvenFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

/-- The derivative of a function f: ℝ → ℝ -/
def DerivativeOf (g f : ℝ → ℝ) : Prop :=
  ∀ x, HasDerivAt f (g x) x

theorem derivative_of_even_function_is_odd
  (f g : ℝ → ℝ) (hf : EvenFunction f) (hg : DerivativeOf g f) :
  ∀ x, g (-x) = -g x :=
sorry

end derivative_of_even_function_is_odd_l923_92351


namespace power_function_difference_l923_92333

-- Define a power function
def isPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ α : ℝ, ∀ x : ℝ, x > 0 → f x = x ^ α

-- State the theorem
theorem power_function_difference (f : ℝ → ℝ) :
  isPowerFunction f → f 9 = 3 → f 2 - f 1 = Real.sqrt 2 - 1 := by
  sorry

end power_function_difference_l923_92333


namespace prove_present_age_of_B_l923_92399

/-- The present age of person B given the conditions:
    1. In 10 years, A will be twice as old as B was 10 years ago
    2. A is now 9 years older than B -/
def present_age_of_B (a b : ℕ) : Prop :=
  (a + 10 = 2 * (b - 10)) ∧ (a = b + 9) → b = 39

theorem prove_present_age_of_B :
  ∀ (a b : ℕ), present_age_of_B a b :=
by
  sorry

end prove_present_age_of_B_l923_92399


namespace restaurant_group_composition_l923_92315

theorem restaurant_group_composition (total_people : ℕ) (adult_meal_cost : ℕ) (total_cost : ℕ) : 
  total_people = 11 → 
  adult_meal_cost = 8 → 
  total_cost = 72 → 
  ∃ (num_adults num_kids : ℕ), 
    num_adults + num_kids = total_people ∧ 
    num_adults * adult_meal_cost = total_cost ∧ 
    num_kids = 2 := by
  sorry


end restaurant_group_composition_l923_92315


namespace tangent_line_equation_l923_92361

/-- The equation of the line tangent to a circle at two points, which also passes through a given point -/
theorem tangent_line_equation (x y : ℝ → ℝ) :
  -- Given circle equation
  (∀ t, x t ^ 2 + (y t - 2) ^ 2 = 4) →
  -- Circle passes through (-2, 6)
  (∃ t, x t = -2 ∧ y t = 6) →
  -- Line equation
  (∃ a b c : ℝ, ∀ t, a * x t + b * y t + c = 0) →
  -- The line equation is x - 2y + 6 = 0
  (∃ t₁ t₂, t₁ ≠ t₂ ∧ x t₁ - 2 * y t₁ + 6 = 0 ∧ x t₂ - 2 * y t₂ + 6 = 0) := by
sorry

end tangent_line_equation_l923_92361


namespace hall_volume_proof_l923_92305

/-- Represents a rectangular wall with a width and height -/
structure RectWall where
  width : ℝ
  height : ℝ

/-- Represents a rectangular hall with length, width, and height -/
structure RectHall where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculate the area of a rectangular wall -/
def wallArea (w : RectWall) : ℝ := w.width * w.height

/-- Calculate the volume of a rectangular hall -/
def hallVolume (h : RectHall) : ℝ := h.length * h.width * h.height

theorem hall_volume_proof (h : RectHall) 
  (a1 a2 : RectWall) 
  (b1 b2 : RectWall) 
  (c1 c2 : RectWall) :
  h.length = 30 ∧ 
  h.width = 20 ∧ 
  h.height = 10 ∧
  a1.width = a2.width ∧
  b1.height = b2.height ∧
  c1.height = c2.height ∧
  b1.height = h.height ∧
  c1.height = h.height ∧
  wallArea a1 + wallArea a2 = wallArea b1 + wallArea b2 ∧
  wallArea c1 + wallArea c2 = 2 * h.length * h.width ∧
  a1.width + a2.width = h.width ∧
  b1.width + b2.width = h.length ∧
  c1.width + c2.width = h.width →
  hallVolume h = 6000 := by
sorry

end hall_volume_proof_l923_92305


namespace sqrt_expression_equality_l923_92346

theorem sqrt_expression_equality : 3 * Real.sqrt 12 / (3 * Real.sqrt (1/3)) - 2 * Real.sqrt 3 = 6 - 2 * Real.sqrt 3 := by
  sorry

end sqrt_expression_equality_l923_92346


namespace winnie_balloons_l923_92313

/-- The number of balloons Winnie keeps for herself when distributing balloons among friends -/
theorem winnie_balloons (total_balloons : ℕ) (num_friends : ℕ) (h1 : total_balloons = 226) (h2 : num_friends = 11) :
  total_balloons % num_friends = 6 := by
  sorry

end winnie_balloons_l923_92313


namespace sum_of_squares_quadratic_roots_sum_of_squares_specific_quadratic_l923_92385

theorem sum_of_squares_quadratic_roots (a b c : ℝ) (h : a ≠ 0) :
  let r₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let r₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  a * x^2 + b * x + c = 0 → r₁^2 + r₂^2 = (b^2 - 2*a*c) / a^2 :=
by sorry

theorem sum_of_squares_specific_quadratic :
  let r₁ := (16 + Real.sqrt 256) / 2
  let r₂ := (16 - Real.sqrt 256) / 2
  x^2 - 16*x + 4 = 0 → r₁^2 + r₂^2 = 248 :=
by sorry

end sum_of_squares_quadratic_roots_sum_of_squares_specific_quadratic_l923_92385


namespace incorrect_statement_identification_l923_92343

theorem incorrect_statement_identification :
  ((-64 : ℚ)^(1/3) = -4) ∧ 
  ((49 : ℚ)^(1/2) = 7) ∧ 
  ((1/27 : ℚ)^(1/3) = 1/3) →
  ¬((1/16 : ℚ)^(1/2) = 1/4) :=
by
  sorry

end incorrect_statement_identification_l923_92343


namespace point_in_second_quadrant_l923_92331

def second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0

theorem point_in_second_quadrant :
  let x : ℝ := -5
  let y : ℝ := 4
  second_quadrant x y :=
by
  sorry

end point_in_second_quadrant_l923_92331


namespace cylinder_volume_change_l923_92366

/-- Theorem: Cylinder Volume Change
  Given a cylinder with an initial volume of 20 cubic feet,
  if its radius is tripled and its height is doubled,
  then its new volume will be 360 cubic feet. -/
theorem cylinder_volume_change (r h : ℝ) : 
  r > 0 → h > 0 → π * r^2 * h = 20 → π * (3*r)^2 * (2*h) = 360 := by
  sorry

end cylinder_volume_change_l923_92366


namespace clothing_combinations_l923_92349

theorem clothing_combinations (hoodies sweatshirts jeans slacks : ℕ) 
  (h_hoodies : hoodies = 5)
  (h_sweatshirts : sweatshirts = 4)
  (h_jeans : jeans = 3)
  (h_slacks : slacks = 5) :
  (hoodies + sweatshirts) * (jeans + slacks) = 72 := by
  sorry

end clothing_combinations_l923_92349


namespace cross_ratio_equality_l923_92310

theorem cross_ratio_equality (p q r s : ℝ) 
  (h : (p - q) * (r - s) / ((q - r) * (s - p)) = 3 / 7) :
  (p - r) * (q - s) / ((p - q) * (r - s)) = -4 / 3 := by
  sorry

end cross_ratio_equality_l923_92310


namespace article_price_l923_92370

theorem article_price (final_price : ℝ) (discount1 : ℝ) (discount2 : ℝ) : 
  final_price = 126 ∧ discount1 = 0.1 ∧ discount2 = 0.2 →
  ∃ (original_price : ℝ), original_price = 175 ∧ 
    final_price = original_price * (1 - discount1) * (1 - discount2) :=
by
  sorry

end article_price_l923_92370


namespace cube_color_theorem_l923_92311

theorem cube_color_theorem (n : ℕ) (h : n = 82) :
  ∀ (coloring : Fin n → Type),
    (∃ (cubes : Fin 10 → Fin n), (∀ i j, i ≠ j → coloring (cubes i) ≠ coloring (cubes j))) ∨
    (∃ (color : Type) (cubes : Fin 10 → Fin n), (∀ i, coloring (cubes i) = color)) :=
by sorry

end cube_color_theorem_l923_92311


namespace seashells_total_l923_92367

theorem seashells_total (sam_shells mary_shells : ℕ) 
  (h1 : sam_shells = 18) (h2 : mary_shells = 47) : 
  sam_shells + mary_shells = 65 := by
  sorry

end seashells_total_l923_92367


namespace point_Q_in_second_quadrant_l923_92394

/-- Given point P(0,a) on the negative half-axis of the y-axis, 
    point Q(-a^2-1, -a+1) lies in the second quadrant. -/
theorem point_Q_in_second_quadrant (a : ℝ) 
  (h_a_neg : a < 0) : 
  let P : ℝ × ℝ := (0, a)
  let Q : ℝ × ℝ := (-a^2 - 1, -a + 1)
  (-a^2 - 1 < 0) ∧ (-a + 1 > 0) := by
sorry

end point_Q_in_second_quadrant_l923_92394


namespace square_sum_representation_l923_92319

theorem square_sum_representation : ∃ (a b c : ℕ), 
  15129 = a^2 + b^2 + c^2 ∧ 
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  (a ≠ 27 ∨ b ≠ 72 ∨ c ≠ 96) ∧
  ∃ (d e f g h i : ℕ), 
    378225 = d^2 + e^2 + f^2 + g^2 + h^2 + i^2 ∧
    d ≠ e ∧ d ≠ f ∧ d ≠ g ∧ d ≠ h ∧ d ≠ i ∧
    e ≠ f ∧ e ≠ g ∧ e ≠ h ∧ e ≠ i ∧
    f ≠ g ∧ f ≠ h ∧ f ≠ i ∧
    g ≠ h ∧ g ≠ i ∧
    h ≠ i := by
  sorry

end square_sum_representation_l923_92319
