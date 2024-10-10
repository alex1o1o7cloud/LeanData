import Mathlib

namespace circle_C_equation_line_l_equation_l3703_370301

-- Define the circle C
def circle_C : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - 1)^2 + (p.2 - 1)^2 ≤ 2}

-- Define the tangent line
def tangent_line : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 + p.2 = 4}

-- Define the point that line l passes through
def point_on_l : ℝ × ℝ := (2, 3)

-- Define the length of the chord intercepted by circle C on line l
def chord_length : ℝ := 2

-- Theorem stating the standard equation of circle C
theorem circle_C_equation :
  ∀ p : ℝ × ℝ, p ∈ circle_C ↔ (p.1 - 1)^2 + (p.2 - 1)^2 = 2 :=
sorry

-- Theorem stating the equation of line l
theorem line_l_equation :
  ∀ p : ℝ × ℝ, (p ∈ circle_C ∧ (∃ q : ℝ × ℝ, q ∈ circle_C ∧ q ≠ p ∧
    ((p.1 - q.1)^2 + (p.2 - q.2)^2 = chord_length^2) ∧
    (p.1 - point_on_l.1) * (q.2 - point_on_l.2) = (q.1 - point_on_l.1) * (p.2 - point_on_l.2)))
  → (3 * p.1 - 4 * p.2 + 6 = 0 ∨ p.1 = 2) :=
sorry

end circle_C_equation_line_l_equation_l3703_370301


namespace sum_of_cubes_is_twelve_l3703_370363

/-- Given real numbers a, b, and c satisfying certain conditions, 
    prove that the sum of their cubes is 12. -/
theorem sum_of_cubes_is_twelve (a b c : ℝ) 
    (sum_eq_three : a + b + c = 3)
    (sum_of_products_eq_three : a * b + a * c + b * c = 3)
    (product_eq_neg_one : a * b * c = -1) : 
  a^3 + b^3 + c^3 = 12 := by
  sorry

end sum_of_cubes_is_twelve_l3703_370363


namespace bianca_cupcakes_l3703_370338

/-- The number of cupcakes Bianca initially made -/
def initial_cupcakes : ℕ := 14

/-- The number of cupcakes Bianca sold -/
def sold_cupcakes : ℕ := 6

/-- The number of additional cupcakes Bianca made -/
def additional_cupcakes : ℕ := 17

/-- The final number of cupcakes Bianca had -/
def final_cupcakes : ℕ := 25

theorem bianca_cupcakes : 
  initial_cupcakes - sold_cupcakes + additional_cupcakes = final_cupcakes :=
by sorry

end bianca_cupcakes_l3703_370338


namespace ones_digit_of_largest_power_of_two_dividing_32_factorial_l3703_370340

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def largest_power_of_two_dividing (n : ℕ) : ℕ :=
  (List.range n).foldl (λ acc x => acc + (x + 1).log 2) 0

def ones_digit (n : ℕ) : ℕ := n % 10

theorem ones_digit_of_largest_power_of_two_dividing_32_factorial :
  ones_digit (2^(largest_power_of_two_dividing (factorial 32))) = 8 := by
  sorry

end ones_digit_of_largest_power_of_two_dividing_32_factorial_l3703_370340


namespace complex_number_quadrant_l3703_370327

theorem complex_number_quadrant (i : ℂ) (z : ℂ) :
  i * i = -1 →
  z = 1 - i →
  (0 < z.re ∧ z.im < 0) :=
by sorry

end complex_number_quadrant_l3703_370327


namespace smallest_two_digit_prime_with_odd_composite_reverse_l3703_370359

/-- A function that reverses the digits of a two-digit number -/
def reverse_digits (n : ℕ) : ℕ :=
  10 * (n % 10) + (n / 10)

/-- A predicate that checks if a number is a two-digit number -/
def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 99

/-- The main theorem statement -/
theorem smallest_two_digit_prime_with_odd_composite_reverse :
  ∃ (n : ℕ), is_two_digit n ∧ Nat.Prime n ∧
  Odd (reverse_digits n) ∧ ¬(Nat.Prime (reverse_digits n)) ∧
  (∀ m : ℕ, is_two_digit m → Nat.Prime m →
    Odd (reverse_digits m) → ¬(Nat.Prime (reverse_digits m)) → n ≤ m) ∧
  n = 19 :=
sorry

end smallest_two_digit_prime_with_odd_composite_reverse_l3703_370359


namespace complex_number_magnitude_l3703_370309

theorem complex_number_magnitude (z : ℂ) (h : z * Complex.I = 1) : Complex.abs z = 1 := by
  sorry

end complex_number_magnitude_l3703_370309


namespace sequence_with_nondivisible_sums_l3703_370394

theorem sequence_with_nondivisible_sums (k : ℕ) (h : Even k) (h' : k > 0) :
  ∃ π : Fin (k - 1) → Fin (k - 1), Function.Bijective π ∧
    ∀ (i j : Fin (k - 1)), i ≤ j →
      ¬(k ∣ (Finset.sum (Finset.Icc i j) (fun n => (π n).val + 1))) :=
sorry

end sequence_with_nondivisible_sums_l3703_370394


namespace fraction_condition_necessary_not_sufficient_l3703_370378

theorem fraction_condition_necessary_not_sufficient :
  ∀ x : ℝ, (|x - 1| < 1 → (x + 3) / (x - 2) < 0) ∧
  ¬(∀ x : ℝ, (x + 3) / (x - 2) < 0 → |x - 1| < 1) :=
by sorry

end fraction_condition_necessary_not_sufficient_l3703_370378


namespace least_five_digit_congruent_to_8_mod_17_l3703_370341

theorem least_five_digit_congruent_to_8_mod_17 : 
  ∀ n : ℕ, n ≥ 10000 ∧ n ≡ 8 [MOD 17] → n ≥ 10004 :=
by sorry

end least_five_digit_congruent_to_8_mod_17_l3703_370341


namespace captain_selection_criterion_l3703_370371

-- Define the universe of players
variable (Player : Type)

-- Define predicates
variable (attends_all_sessions : Player → Prop)
variable (always_on_time : Player → Prop)
variable (considered_for_captain : Player → Prop)

-- Theorem statement
theorem captain_selection_criterion
  (h : ∀ p : Player, (attends_all_sessions p ∧ always_on_time p) → considered_for_captain p) :
  ∀ p : Player, ¬(considered_for_captain p) → (¬(attends_all_sessions p) ∨ ¬(always_on_time p)) :=
by sorry

end captain_selection_criterion_l3703_370371


namespace sqrt_equality_implies_square_l3703_370357

theorem sqrt_equality_implies_square (x : ℝ) : 
  Real.sqrt (3 * x + 5) = 5 → (3 * x + 5)^2 = 625 := by
  sorry

end sqrt_equality_implies_square_l3703_370357


namespace binomial_12_9_l3703_370393

theorem binomial_12_9 : Nat.choose 12 9 = 220 := by
  sorry

end binomial_12_9_l3703_370393


namespace sum_of_first_five_terms_l3703_370386

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) = 2 * a n

theorem sum_of_first_five_terms
  (a : ℕ → ℝ)
  (h1 : geometric_sequence a)
  (h2 : a 1 = 1) :
  a 1 + a 2 + a 3 + a 4 + a 5 = 31 :=
sorry

end sum_of_first_five_terms_l3703_370386


namespace bridge_bricks_l3703_370307

theorem bridge_bricks (type_a : ℕ) (type_b : ℕ) (other_types : ℕ) : 
  type_a ≥ 40 →
  type_b = type_a / 2 →
  type_a + type_b + other_types = 150 →
  other_types = 90 := by
sorry

end bridge_bricks_l3703_370307


namespace angle_is_90_degrees_l3703_370360

/-- Represents a point on or above the Earth's surface -/
structure EarthPoint where
  latitude : Real
  longitude : Real
  elevation : Real

/-- Calculate the angle between three points on or above the Earth's surface -/
def angleBAC (earthRadius : Real) (a b c : EarthPoint) : Real :=
  sorry

theorem angle_is_90_degrees (earthRadius : Real) :
  let a : EarthPoint := { latitude := 0, longitude := 100, elevation := 0 }
  let b : EarthPoint := { latitude := 30, longitude := -90, elevation := 0 }
  let c : EarthPoint := { latitude := 90, longitude := 0, elevation := 2 }
  angleBAC earthRadius a b c = 90 := by
  sorry

end angle_is_90_degrees_l3703_370360


namespace ellipse_triangle_perimeter_l3703_370315

/-- Definition of an ellipse with semi-major axis 4 and semi-minor axis 3 -/
def Ellipse : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1^2 / 16) + (p.2^2 / 9) = 1}

/-- The right focus of the ellipse -/
def F₂ : ℝ × ℝ := sorry

/-- The left focus of the ellipse -/
def F₁ : ℝ × ℝ := sorry

/-- Theorem: The perimeter of triangle AF₁B is 16 for any A and B on the ellipse -/
theorem ellipse_triangle_perimeter 
  (A B : ℝ × ℝ) 
  (hA : A ∈ Ellipse) 
  (hB : B ∈ Ellipse) : 
  dist A F₁ + dist B F₁ + dist A B = 16 := 
sorry

end ellipse_triangle_perimeter_l3703_370315


namespace meaningful_expression_l3703_370379

theorem meaningful_expression (x : ℝ) : 
  (∃ y : ℝ, y = 1 / Real.sqrt (x - 5)) ↔ x > 5 :=
by sorry

end meaningful_expression_l3703_370379


namespace platform_length_l3703_370330

/-- Calculates the length of a platform given train specifications -/
theorem platform_length
  (train_length : ℝ)
  (time_cross_platform : ℝ)
  (time_cross_pole : ℝ)
  (h1 : train_length = 700)
  (h2 : time_cross_platform = 45)
  (h3 : time_cross_pole = 15) :
  let train_speed := train_length / time_cross_pole
  let platform_length := train_speed * time_cross_platform - train_length
  platform_length = 1400 := by
sorry

end platform_length_l3703_370330


namespace f_value_at_inverse_f_3_l3703_370305

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 1 then 1 - 2 * x^2 else x^2 + 3 * x - 2

theorem f_value_at_inverse_f_3 : f (1 / f 3) = 127 / 128 := by
  sorry

end f_value_at_inverse_f_3_l3703_370305


namespace factorial_four_div_one_l3703_370331

/-- Definition of factorial for natural numbers -/
def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

/-- Theorem stating that 4! / (4 - 3)! = 24 -/
theorem factorial_four_div_one : factorial 4 / factorial (4 - 3) = 24 := by
  sorry

end factorial_four_div_one_l3703_370331


namespace remove_number_for_target_average_l3703_370352

def original_list : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

def removed_number : ℕ := 5

def target_average : ℚ := 61/10

theorem remove_number_for_target_average :
  let remaining_list := original_list.filter (· ≠ removed_number)
  (remaining_list.sum : ℚ) / remaining_list.length = target_average := by
  sorry

end remove_number_for_target_average_l3703_370352


namespace triangle_properties_l3703_370303

/-- Proves the properties of an acute triangle ABC with given conditions -/
theorem triangle_properties (A B C : Real) (a b c : Real) :
  0 < A ∧ A < π / 2 →  -- A is acute
  0 < B ∧ B < π / 2 →  -- B is acute
  0 < C ∧ C < π / 2 →  -- C is acute
  a * Real.sin B = Real.sqrt 3 * b * Real.cos A →
  a = Real.sqrt 21 →
  b = 5 →
  A + B + C = π →  -- Sum of angles in a triangle
  a * Real.sin B = b * Real.sin A →  -- Sine law
  a^2 = b^2 + c^2 - 2 * b * c * Real.cos A →  -- Cosine law
  A = π / 3 ∧ c = 4 := by
  sorry


end triangle_properties_l3703_370303


namespace max_unglazed_windows_l3703_370370

/-- Represents a window or a pane of glass with a specific size. -/
structure Pane :=
  (size : ℕ)

/-- Represents the state of glazing process. -/
structure GlazingState :=
  (windows : List Pane)
  (glasses : List Pane)

/-- Simulates the glazier's process of matching glasses to windows. -/
def glazierProcess (state : GlazingState) : ℕ :=
  sorry

/-- Theorem stating the maximum number of unglazed windows. -/
theorem max_unglazed_windows :
  ∀ (initial_state : GlazingState),
    initial_state.windows.length = 15 ∧
    initial_state.glasses.length = 15 ∧
    (∀ w ∈ initial_state.windows, ∃ g ∈ initial_state.glasses, w.size = g.size) →
    glazierProcess initial_state ≤ 7 :=
  sorry

end max_unglazed_windows_l3703_370370


namespace ellipse_vertex_distance_l3703_370398

/-- The distance between the vertices of the ellipse x^2/49 + y^2/64 = 1 is 16 -/
theorem ellipse_vertex_distance :
  let a := Real.sqrt (max 49 64)
  let ellipse := {(x, y) : ℝ × ℝ | x^2/49 + y^2/64 = 1}
  2 * a = 16 := by
  sorry

end ellipse_vertex_distance_l3703_370398


namespace average_study_time_difference_l3703_370367

def daily_differences : List Int := [10, -10, 20, 30, -20]

theorem average_study_time_difference : 
  (daily_differences.sum : ℚ) / daily_differences.length = 6 := by
  sorry

end average_study_time_difference_l3703_370367


namespace cyclic_sum_inequality_l3703_370372

theorem cyclic_sum_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h_sum : x^2 + y^2 + z^2 = 1) :
  x*y/z + y*z/x + z*x/y ≥ Real.sqrt 3 := by
  sorry

end cyclic_sum_inequality_l3703_370372


namespace trigonometric_identity_l3703_370389

theorem trigonometric_identity : 
  (Real.sin (65 * π / 180) + Real.sin (15 * π / 180) * Real.sin (10 * π / 180)) / 
  (Real.sin (25 * π / 180) - Real.cos (15 * π / 180) * Real.cos (80 * π / 180)) = 2 + Real.sqrt 3 := by
  sorry

end trigonometric_identity_l3703_370389


namespace bounded_expression_l3703_370332

theorem bounded_expression (x y : ℝ) :
  -1/2 ≤ ((x + y) * (1 - x * y)) / ((1 + x^2) * (1 + y^2)) ∧
  ((x + y) * (1 - x * y)) / ((1 + x^2) * (1 + y^2)) ≤ 1/2 := by
  sorry

end bounded_expression_l3703_370332


namespace m_minus_n_equals_six_l3703_370319

theorem m_minus_n_equals_six (m n : ℤ) 
  (h1 : |m| = 2)
  (h2 : |n| = 4)
  (h3 : m > 0)
  (h4 : n < 0) :
  m - n = 6 := by
  sorry

end m_minus_n_equals_six_l3703_370319


namespace playground_count_l3703_370344

theorem playground_count (a b c d e x : ℕ) (h1 : a = 6) (h2 : b = 12) (h3 : c = 1) (h4 : d = 12) (h5 : e = 7)
  (h_mean : (a + b + c + d + e + x) / 6 = 7) : x = 4 := by
  sorry

end playground_count_l3703_370344


namespace cosine_function_properties_l3703_370317

/-- Given function f(x) = cos(ωx + φ) -/
noncomputable def f (ω φ : ℝ) (x : ℝ) : ℝ := Real.cos (ω * x + φ)

theorem cosine_function_properties (ω φ : ℝ) 
  (h_ω_pos : ω > 0)
  (h_φ_range : 0 < φ ∧ φ < π / 2)
  (h_period : ∀ x, f ω φ (x + π) = f ω φ x)
  (h_value : f ω φ (π / 3) = -Real.sqrt 3 / 2) :
  (ω = 2 ∧ φ = π / 6) ∧
  (∀ x, f ω φ x > 1 / 2 ↔ ∃ k : ℤ, k * π - π / 4 < x ∧ x < k * π + π / 12) :=
by sorry

end cosine_function_properties_l3703_370317


namespace second_group_size_l3703_370397

/-- The number of men in the first group -/
def men_group1 : ℕ := 4

/-- The number of hours worked per day by the first group -/
def hours_per_day_group1 : ℕ := 10

/-- The earnings per week of the first group in rupees -/
def earnings_group1 : ℕ := 1000

/-- The number of hours worked per day by the second group -/
def hours_per_day_group2 : ℕ := 6

/-- The earnings per week of the second group in rupees -/
def earnings_group2 : ℕ := 1350

/-- The number of days in a week -/
def days_per_week : ℕ := 7

/-- The number of men in the second group -/
def men_group2 : ℕ := 9

theorem second_group_size :
  men_group2 * hours_per_day_group2 * days_per_week * earnings_group1 =
  men_group1 * hours_per_day_group1 * days_per_week * earnings_group2 :=
by sorry

end second_group_size_l3703_370397


namespace sufficient_not_necessary_condition_l3703_370361

theorem sufficient_not_necessary_condition (a : ℝ) : 
  (∀ x : ℝ, x > a → x^2 - 5*x + 6 ≥ 0) ∧ 
  (∃ x : ℝ, x^2 - 5*x + 6 ≥ 0 ∧ x ≤ a) ↔ 
  a ≥ 3 := by
sorry

end sufficient_not_necessary_condition_l3703_370361


namespace polynomial_evaluation_l3703_370345

theorem polynomial_evaluation : 
  ∃ x : ℝ, x > 0 ∧ x^2 - 3*x - 10 = 0 ∧ x^4 - 3*x^3 - 4*x^2 + 12*x + 9 = 219 := by
  sorry

end polynomial_evaluation_l3703_370345


namespace dress_design_combinations_l3703_370358

theorem dress_design_combinations (num_colors num_patterns : ℕ) 
  (h_colors : num_colors = 5)
  (h_patterns : num_patterns = 6) :
  num_colors * num_patterns = 30 := by
sorry

end dress_design_combinations_l3703_370358


namespace min_cubes_for_majority_interior_min_total_cubes_l3703_370343

/-- A function that calculates the number of interior cubes in a cube of side length n -/
def interior_cubes (n : ℕ) : ℕ := (n - 2)^3

/-- A function that calculates the total number of unit cubes in a cube of side length n -/
def total_cubes (n : ℕ) : ℕ := n^3

/-- The minimum side length of a cube where more than half of the cubes are interior -/
def min_side_length : ℕ := 10

theorem min_cubes_for_majority_interior :
  (∀ k < min_side_length, 2 * interior_cubes k ≤ total_cubes k) ∧
  2 * interior_cubes min_side_length > total_cubes min_side_length :=
by sorry

theorem min_total_cubes : total_cubes min_side_length = 1000 :=
by sorry

end min_cubes_for_majority_interior_min_total_cubes_l3703_370343


namespace johns_age_l3703_370374

theorem johns_age : ∃ (j : ℝ), j = 22.5 ∧ (j - 10 = (1 / 3) * (j + 15)) := by
  sorry

end johns_age_l3703_370374


namespace simplify_expression_l3703_370395

theorem simplify_expression : 5 * (18 / (-9)) * (24 / 36) = -20 / 3 := by
  sorry

end simplify_expression_l3703_370395


namespace mary_shirts_problem_l3703_370364

theorem mary_shirts_problem (blue_shirts : ℕ) (brown_shirts : ℕ) (remaining_shirts : ℕ) :
  blue_shirts = 26 →
  brown_shirts = 36 →
  remaining_shirts = 37 →
  ∃ (f : ℚ), f = 1/2 ∧
    blue_shirts * (1 - f) + brown_shirts * (2/3) = remaining_shirts :=
by sorry

end mary_shirts_problem_l3703_370364


namespace shoes_outside_library_l3703_370392

/-- The total number of shoes outside the library -/
def total_shoes (regular_shoes sandals slippers : ℕ) : ℕ :=
  2 * regular_shoes + 2 * sandals + 2 * slippers

/-- Proof that the total number of shoes is 20 -/
theorem shoes_outside_library :
  let total_people : ℕ := 10
  let regular_shoe_wearers : ℕ := 4
  let sandal_wearers : ℕ := 3
  let slipper_wearers : ℕ := 3
  total_people = regular_shoe_wearers + sandal_wearers + slipper_wearers →
  total_shoes regular_shoe_wearers sandal_wearers slipper_wearers = 20 :=
by
  sorry

end shoes_outside_library_l3703_370392


namespace profit_calculation_l3703_370368

/-- The profit calculation for a product with given purchase price, markup percentage, and discount. -/
theorem profit_calculation (purchase_price : ℝ) (markup_percent : ℝ) (discount_percent : ℝ) :
  purchase_price = 200 →
  markup_percent = 1.25 →
  discount_percent = 0.9 →
  purchase_price * markup_percent * discount_percent - purchase_price = 25 := by
  sorry

#check profit_calculation

end profit_calculation_l3703_370368


namespace nathans_score_l3703_370306

theorem nathans_score (total_students : ℕ) (students_without_nathan : ℕ) 
  (avg_without_nathan : ℚ) (avg_with_nathan : ℚ) :
  total_students = 18 →
  students_without_nathan = 17 →
  avg_without_nathan = 84 →
  avg_with_nathan = 87 →
  (total_students * avg_with_nathan - students_without_nathan * avg_without_nathan : ℚ) = 138 :=
by sorry

end nathans_score_l3703_370306


namespace product_sum_inequality_l3703_370318

theorem product_sum_inequality (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) 
  (h_prod : a * b * c * d = 1) : 
  a^2 + b^2 + c^2 + d^2 + a*b + a*c + a*d + b*c + b*d + c*d ≥ 10 := by
  sorry

end product_sum_inequality_l3703_370318


namespace equal_solution_is_two_l3703_370376

/-- Given a system of equations for nonnegative real numbers, prove that the only solution where all numbers are equal is 2. -/
theorem equal_solution_is_two (n : ℕ) (x : ℕ → ℝ) : 
  n > 2 →
  (∀ k, k ∈ Finset.range n → x k ≥ 0) →
  (∀ k, k ∈ Finset.range n → x k + x ((k + 1) % n) = (x ((k + 2) % n))^2) →
  (∀ i j, i ∈ Finset.range n → j ∈ Finset.range n → x i = x j) →
  (∀ k, k ∈ Finset.range n → x k = 2) := by
sorry

end equal_solution_is_two_l3703_370376


namespace bob_position_2023_l3703_370329

-- Define the movement pattern
def spiral_move (n : ℕ) : ℤ × ℤ := sorry

-- Define Bob's position after n steps
def bob_position (n : ℕ) : ℤ × ℤ := sorry

-- Theorem statement
theorem bob_position_2023 :
  bob_position 2023 = (0, 43) := sorry

end bob_position_2023_l3703_370329


namespace circular_binary_arrangement_l3703_370365

/-- A type representing a binary number using only 1 and 2 -/
def BinaryNumber (n : ℕ) := Fin n → Fin 2

/-- A function to check if two binary numbers differ by exactly one digit -/
def differByOneDigit (n : ℕ) (a b : BinaryNumber n) : Prop :=
  ∃! i : Fin n, a i ≠ b i

/-- A type representing an arrangement of binary numbers in a circle -/
def CircularArrangement (n : ℕ) := Fin (2^n) → BinaryNumber n

/-- The main theorem statement -/
theorem circular_binary_arrangement (n : ℕ) :
  ∃ (arrangement : CircularArrangement n),
    (∀ i j : Fin (2^n), i ≠ j → arrangement i ≠ arrangement j) ∧
    (∀ i : Fin (2^n), differByOneDigit n (arrangement i) (arrangement (i + 1))) :=
sorry

end circular_binary_arrangement_l3703_370365


namespace trajectory_and_range_l3703_370383

-- Define the circle D
def circle_D (x y : ℝ) : Prop := (x - 2)^2 + (y + 3)^2 = 32

-- Define point P
def P : ℝ × ℝ := (-6, 3)

-- Define the trajectory of M
def trajectory_M (x y : ℝ) : Prop := (x + 2)^2 + y^2 = 8

-- Define the range of t
def t_range (t : ℝ) : Prop :=
  t ∈ Set.Icc (-Real.sqrt 5 - 1) (-Real.sqrt 5 + 1) ∪
      Set.Icc (Real.sqrt 5 - 1) (Real.sqrt 5 + 1)

theorem trajectory_and_range :
  (∀ x y : ℝ, ∃ x_H y_H : ℝ,
    circle_D x_H y_H ∧
    x = (x_H + P.1) / 2 ∧
    y = (y_H + P.2) / 2 →
    trajectory_M x y) ∧
  (∀ k t : ℝ,
    (∃ x_B y_B x_C y_C : ℝ,
      trajectory_M x_B y_B ∧
      trajectory_M x_C y_C ∧
      y_B = k * x_B ∧
      y_C = k * x_C ∧
      (x_B - 0) * (x_C - 0) + (y_B - t) * (y_C - t) = 0) →
    t_range t) :=
by sorry

end trajectory_and_range_l3703_370383


namespace eggs_per_basket_l3703_370353

theorem eggs_per_basket (red_eggs blue_eggs min_eggs : ℕ) 
  (h1 : red_eggs = 30)
  (h2 : blue_eggs = 42)
  (h3 : min_eggs = 5) :
  ∃ (n : ℕ), n ≥ min_eggs ∧ 
             n ∣ red_eggs ∧ 
             n ∣ blue_eggs ∧
             ∀ (m : ℕ), m ≥ min_eggs ∧ m ∣ red_eggs ∧ m ∣ blue_eggs → m ≤ n :=
by sorry

end eggs_per_basket_l3703_370353


namespace arithmetic_sequence_sum_mod_l3703_370350

theorem arithmetic_sequence_sum_mod (a d : ℕ) (n : ℕ) (h : n > 0) :
  let last_term := a + (n - 1) * d
  let sum := n * (a + last_term) / 2
  sum % 17 = 12 :=
by
  sorry

#check arithmetic_sequence_sum_mod 3 5 21

end arithmetic_sequence_sum_mod_l3703_370350


namespace two_satisfying_functions_l3703_370339

/-- A function satisfying the given property -/
def SatisfiesProperty (f : ℝ → ℝ) : Prop :=
  ∀ x y z : ℝ, f (x^2 - y * f z) = x * f x - z * f y

/-- The set of functions satisfying the property -/
def SatisfyingFunctions : Set (ℝ → ℝ) :=
  {f | SatisfiesProperty f}

/-- The constant zero function -/
def ZeroFunction : ℝ → ℝ := λ _ ↦ 0

/-- The identity function -/
def IdentityFunction : ℝ → ℝ := λ x ↦ x

theorem two_satisfying_functions :
  SatisfyingFunctions = {ZeroFunction, IdentityFunction} := by
  sorry

#check two_satisfying_functions

end two_satisfying_functions_l3703_370339


namespace logarithm_inequality_l3703_370313

-- Define the logarithm function (base 10)
noncomputable def lg (x : ℝ) := Real.log x / Real.log 10

-- State the theorem
theorem logarithm_inequality (x : ℝ) (h : 1 < x ∧ x < 10) :
  lg (x^2) > (lg x)^2 ∧ (lg x)^2 > lg (lg x) := by
  sorry

end logarithm_inequality_l3703_370313


namespace inequality_proof_l3703_370337

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  Real.sqrt (a^2 - a*b + b^2) + Real.sqrt (b^2 - b*c + c^2) ≥ Real.sqrt (a^2 + a*c + c^2) ∧
  (Real.sqrt (a^2 - a*b + b^2) + Real.sqrt (b^2 - b*c + c^2) = Real.sqrt (a^2 + a*c + c^2) ↔ 1/b = 1/a + 1/c) :=
by sorry

end inequality_proof_l3703_370337


namespace inequalities_in_quadrants_I_and_II_exists_points_in_both_quadrants_l3703_370391

/-- Represents a point in the 2D plane -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Defines the region satisfying the given inequalities -/
def SatisfiesInequalities (p : Point) : Prop :=
  p.y > -2 * p.x + 3 ∧ p.y > 1/2 * p.x + 1

/-- Checks if a point is in Quadrant I -/
def InQuadrantI (p : Point) : Prop :=
  p.x > 0 ∧ p.y > 0

/-- Checks if a point is in Quadrant II -/
def InQuadrantII (p : Point) : Prop :=
  p.x < 0 ∧ p.y > 0

/-- Theorem stating that all points satisfying the inequalities are in Quadrants I and II -/
theorem inequalities_in_quadrants_I_and_II :
  ∀ p : Point, SatisfiesInequalities p → (InQuadrantI p ∨ InQuadrantII p) :=
by
  sorry

/-- Theorem stating that there exist points in both Quadrants I and II that satisfy the inequalities -/
theorem exists_points_in_both_quadrants :
  (∃ p : Point, SatisfiesInequalities p ∧ InQuadrantI p) ∧
  (∃ p : Point, SatisfiesInequalities p ∧ InQuadrantII p) :=
by
  sorry

end inequalities_in_quadrants_I_and_II_exists_points_in_both_quadrants_l3703_370391


namespace f_is_quadratic_l3703_370326

-- Define what a quadratic equation is
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

-- Define the specific function we're checking
def f (x : ℝ) : ℝ := x^2 + 6*x

-- Theorem statement
theorem f_is_quadratic : is_quadratic_equation f := by
  sorry

end f_is_quadratic_l3703_370326


namespace total_nylon_needed_l3703_370333

/-- The amount of nylon needed for a dog collar in inches -/
def dog_collar_nylon : ℕ := 18

/-- The amount of nylon needed for a cat collar in inches -/
def cat_collar_nylon : ℕ := 10

/-- The number of dog collars to be made -/
def num_dog_collars : ℕ := 9

/-- The number of cat collars to be made -/
def num_cat_collars : ℕ := 3

/-- Theorem stating the total amount of nylon needed -/
theorem total_nylon_needed : 
  dog_collar_nylon * num_dog_collars + cat_collar_nylon * num_cat_collars = 192 := by
  sorry

end total_nylon_needed_l3703_370333


namespace distinct_sums_count_l3703_370399

/-- The set of ball numbers -/
def BallNumbers : Finset ℕ := {1, 2, 3, 4, 5}

/-- The sum of two numbers drawn from BallNumbers with replacement -/
def SumOfDraws : Finset ℕ := Finset.image (λ (x : ℕ × ℕ) => x.1 + x.2) (BallNumbers.product BallNumbers)

/-- The number of distinct possible sums -/
theorem distinct_sums_count : Finset.card SumOfDraws = 9 := by
  sorry

end distinct_sums_count_l3703_370399


namespace test_problem_value_l3703_370323

theorem test_problem_value (total_points total_problems four_point_problems : ℕ)
  (h1 : total_points = 100)
  (h2 : total_problems = 30)
  (h3 : four_point_problems = 10)
  (h4 : four_point_problems < total_problems) :
  (total_points - 4 * four_point_problems) / (total_problems - four_point_problems) = 3 :=
by sorry

end test_problem_value_l3703_370323


namespace cafeteria_combos_l3703_370377

/-- Represents the number of options for each part of the lunch combo -/
structure LunchOptions where
  mainDishes : Nat
  sides : Nat
  drinks : Nat
  desserts : Nat

/-- Calculates the total number of distinct lunch combos -/
def totalCombos (options : LunchOptions) : Nat :=
  options.mainDishes * options.sides * options.drinks * options.desserts

/-- The specific lunch options available in the cafeteria -/
def cafeteriaOptions : LunchOptions :=
  { mainDishes := 3
  , sides := 2
  , drinks := 2
  , desserts := 2 }

theorem cafeteria_combos :
  totalCombos cafeteriaOptions = 24 := by
  sorry

#eval totalCombos cafeteriaOptions

end cafeteria_combos_l3703_370377


namespace initial_average_production_is_50_l3703_370387

/-- Calculates the initial average daily production given the number of past days,
    today's production, and the new average including today. -/
def initialAverageProduction (n : ℕ) (todayProduction : ℕ) (newAverage : ℚ) : ℚ :=
  (newAverage * (n + 1) - todayProduction) / n

theorem initial_average_production_is_50 :
  initialAverageProduction 10 105 55 = 50 := by sorry

end initial_average_production_is_50_l3703_370387


namespace inverse_proportion_percentage_change_l3703_370396

theorem inverse_proportion_percentage_change 
  (x y x' y' k q : ℝ) 
  (h_positive : x > 0 ∧ y > 0)
  (h_inverse : x * y = k)
  (h_y_decrease : y' = y * (1 - q / 100))
  (h_constant : x' * y' = k) :
  (x' - x) / x * 100 = 100 * q / (100 - q) := by
sorry

end inverse_proportion_percentage_change_l3703_370396


namespace unique_solution_l3703_370356

-- Define the properties of p, q, and r
def is_valid_solution (p q r : ℕ) : Prop :=
  Nat.Prime p ∧ 
  Nat.Prime q ∧ 
  Nat.Prime r ∧
  q - p = r ∧
  5 < p ∧ p < 15 ∧
  q < 15

-- Theorem statement
theorem unique_solution : 
  ∃! q : ℕ, ∃ (p r : ℕ), is_valid_solution p q r ∧ q = 13 :=
sorry

end unique_solution_l3703_370356


namespace original_earnings_l3703_370349

theorem original_earnings (new_earnings : ℝ) (percentage_increase : ℝ) 
  (h1 : new_earnings = 84)
  (h2 : percentage_increase = 40) :
  let original_earnings := new_earnings / (1 + percentage_increase / 100)
  original_earnings = 60 := by
sorry

end original_earnings_l3703_370349


namespace max_value_on_curve_l3703_370324

-- Define the curve C in polar coordinates
def C (ρ θ : ℝ) : Prop := ρ^2 = 36 / (4 * (Real.cos θ)^2 + 9 * (Real.sin θ)^2)

-- Define a point P on the curve C
def P (x y : ℝ) : Prop := ∃ (ρ θ : ℝ), C ρ θ ∧ x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ

-- State the theorem
theorem max_value_on_curve :
  ∀ (x y : ℝ), P x y → (∀ (x' y' : ℝ), P x' y' → 3 * x + 4 * y ≤ 3 * x' + 4 * y') →
  3 * x + 4 * y = Real.sqrt 145 :=
sorry

end max_value_on_curve_l3703_370324


namespace compute_expression_l3703_370346

theorem compute_expression : 6^3 - 4*5 + 2^4 = 212 := by sorry

end compute_expression_l3703_370346


namespace pasta_preference_ratio_l3703_370385

-- Define the total number of students
def total_students : ℕ := 800

-- Define the number of students preferring spaghetti
def spaghetti_preference : ℕ := 320

-- Define the number of students preferring fettuccine
def fettuccine_preference : ℕ := 160

-- Theorem to prove the ratio
theorem pasta_preference_ratio : 
  (spaghetti_preference : ℚ) / (fettuccine_preference : ℚ) = 2 := by
  sorry


end pasta_preference_ratio_l3703_370385


namespace rectangle_dimension_increase_l3703_370312

theorem rectangle_dimension_increase (L B : ℝ) (L' B' : ℝ) (h1 : B' = 1.3 * B) (h2 : L' * B' = 1.43 * L * B) : L' = 1.1 * L := by
  sorry

end rectangle_dimension_increase_l3703_370312


namespace quadratic_inequality_solution_l3703_370351

-- Define the quadratic function
def f (x : ℝ) : ℝ := -3 * x^2 + 9 * x + 12

-- Define the solution set
def solution_set : Set ℝ := {x : ℝ | -1 < x ∧ x < 4}

-- Theorem statement
theorem quadratic_inequality_solution :
  {x : ℝ | f x > 0} = solution_set :=
sorry

end quadratic_inequality_solution_l3703_370351


namespace correct_regression_sequence_l3703_370304

-- Define the steps of linear regression analysis
inductive RegressionStep
  | InterpretEquation
  | CollectData
  | CalculateEquation
  | DrawScatterPlot

-- Define a sequence of steps
def StepSequence := List RegressionStep

-- Define the correct sequence
def correctSequence : StepSequence :=
  [RegressionStep.CollectData,
   RegressionStep.DrawScatterPlot,
   RegressionStep.CalculateEquation,
   RegressionStep.InterpretEquation]

-- Define a proposition that x and y are linearly related
def linearlyRelated (x y : ℝ → ℝ) : Prop :=
  ∃ a b : ℝ, ∀ t : ℝ, y t = a * x t + b

-- Theorem stating that given linear relationship, the correct sequence is as defined
theorem correct_regression_sequence (x y : ℝ → ℝ) :
  linearlyRelated x y →
  (∀ seq : StepSequence,
    seq = correctSequence ↔
    seq = [RegressionStep.CollectData,
           RegressionStep.DrawScatterPlot,
           RegressionStep.CalculateEquation,
           RegressionStep.InterpretEquation]) :=
by sorry


end correct_regression_sequence_l3703_370304


namespace gcd_count_for_360_l3703_370354

theorem gcd_count_for_360 (a b : ℕ+) : 
  (Nat.gcd a b * Nat.lcm a b = 360) → 
  (∃ (S : Finset ℕ), (∀ d ∈ S, ∃ (x y : ℕ+), Nat.gcd x y = d ∧ Nat.gcd x y * Nat.lcm x y = 360) ∧ 
                      (∀ d : ℕ, (∃ (x y : ℕ+), Nat.gcd x y = d ∧ Nat.gcd x y * Nat.lcm x y = 360) → d ∈ S) ∧
                      S.card = 10) :=
sorry

end gcd_count_for_360_l3703_370354


namespace record_storage_cost_l3703_370336

def box_length : ℝ := 15
def box_width : ℝ := 12
def box_height : ℝ := 10
def total_volume : ℝ := 1080000
def cost_per_box : ℝ := 0.8

theorem record_storage_cost : 
  let box_volume := box_length * box_width * box_height
  let num_boxes := total_volume / box_volume
  num_boxes * cost_per_box = 480 := by sorry

end record_storage_cost_l3703_370336


namespace calculation_proof_l3703_370380

theorem calculation_proof :
  ((-36) * (1/3 - 1/2) + 16 / (-2)^3 = 4) ∧
  ((-5 + 2) * (1/3) + 5^2 / (-5) = -6) := by
  sorry

end calculation_proof_l3703_370380


namespace robins_hair_length_l3703_370382

/-- Robin's hair length problem -/
theorem robins_hair_length (initial_length cut_length : ℕ) (h1 : initial_length = 14) (h2 : cut_length = 13) :
  initial_length - cut_length = 1 := by
  sorry

end robins_hair_length_l3703_370382


namespace kelly_snacks_weight_l3703_370325

/-- The weight of peanuts Kelly bought in pounds -/
def peanuts_weight : ℝ := 0.1

/-- The weight of raisins Kelly bought in pounds -/
def raisins_weight : ℝ := 0.4

/-- The weight of almonds Kelly bought in pounds -/
def almonds_weight : ℝ := 0.3

/-- The total weight of snacks Kelly bought -/
def total_weight : ℝ := peanuts_weight + raisins_weight + almonds_weight

theorem kelly_snacks_weight : total_weight = 0.8 := by
  sorry

end kelly_snacks_weight_l3703_370325


namespace abc_sum_l3703_370348

theorem abc_sum (a b c : ℝ) (ha : |a| = 1) (hb : |b| = 2) (hc : |c| = 4) (horder : a > b ∧ b > c) :
  a - b + c = -1 ∨ a - b + c = -3 := by
sorry

end abc_sum_l3703_370348


namespace gasohol_mixture_proof_l3703_370355

/-- Proves that the initial percentage of gasoline in the gasohol mixture is 95% --/
theorem gasohol_mixture_proof (initial_volume : ℝ) (initial_ethanol_percent : ℝ) 
  (desired_ethanol_percent : ℝ) (added_ethanol : ℝ) :
  initial_volume = 45 →
  initial_ethanol_percent = 5 →
  desired_ethanol_percent = 10 →
  added_ethanol = 2.5 →
  (100 - initial_ethanol_percent) = 95 :=
by
  sorry

#check gasohol_mixture_proof

end gasohol_mixture_proof_l3703_370355


namespace tan_identities_l3703_370388

theorem tan_identities (α : Real) (h : Real.tan (π / 4 + α) = 3) :
  (Real.tan α = 1 / 2) ∧
  (Real.tan (2 * α) = 4 / 3) ∧
  ((2 * Real.sin α * Real.cos α + 3 * Real.cos (2 * α)) / 
   (5 * Real.cos (2 * α) - 3 * Real.sin (2 * α)) = 13 / 3) := by
  sorry

end tan_identities_l3703_370388


namespace quadratic_inequality_solution_l3703_370366

theorem quadratic_inequality_solution (a b : ℝ) : 
  (∀ x : ℝ, ax^2 + bx + 2 > 0 ↔ -1/2 < x ∧ x < 1/3) → a + b = -14 := by
  sorry

end quadratic_inequality_solution_l3703_370366


namespace whale_sixth_hour_consumption_l3703_370342

/-- Represents the whale's feeding pattern over 9 hours -/
def WhaleFeedingPattern (x : ℕ) : List ℕ :=
  List.range 9 |>.map (fun i => x + 3 * i)

/-- The total amount of plankton consumed by the whale -/
def TotalConsumption (x : ℕ) : ℕ :=
  (WhaleFeedingPattern x).sum

theorem whale_sixth_hour_consumption :
  ∃ x : ℕ, 
    TotalConsumption x = 450 ∧ 
    (WhaleFeedingPattern x).get! 5 = 53 := by
  sorry

end whale_sixth_hour_consumption_l3703_370342


namespace speaking_orders_eq_720_l3703_370335

/-- The number of ways to select 4 students from 7 students (including A and B) to speak,
    where at least one of A and B must participate. -/
def speaking_orders : ℕ :=
  let n : ℕ := 7  -- Total number of students
  let k : ℕ := 4  -- Number of students to be selected
  let special : ℕ := 2  -- Number of special students (A and B)
  let others : ℕ := n - special  -- Number of other students

  -- Case 1: Exactly one of A and B participates
  let case1 : ℕ := special * (Nat.choose others (k - 1)) * (Nat.factorial k)

  -- Case 2: Both A and B participate
  let case2 : ℕ := (Nat.choose others (k - special)) * (Nat.factorial k)

  -- Total number of ways
  case1 + case2

/-- Theorem stating that the number of speaking orders is 720 -/
theorem speaking_orders_eq_720 : speaking_orders = 720 := by
  sorry

end speaking_orders_eq_720_l3703_370335


namespace fraction_invariance_l3703_370302

theorem fraction_invariance (x y : ℝ) (h : x ≠ y) : x / (x - y) = (2 * x) / (2 * x - 2 * y) := by
  sorry

end fraction_invariance_l3703_370302


namespace shaded_area_of_concentric_circles_l3703_370369

theorem shaded_area_of_concentric_circles (R r : ℝ) : 
  R > 0 → r > 0 → r = R / 2 → R^2 * π = 81 * π → 
  (π * R^2) / 2 + (π * r^2) / 2 = 50.625 * π := by
  sorry

end shaded_area_of_concentric_circles_l3703_370369


namespace x_gt_neg_two_necessary_not_sufficient_l3703_370390

theorem x_gt_neg_two_necessary_not_sufficient :
  (∃ x : ℝ, x > -2 ∧ (x + 2) * (x - 3) ≥ 0) ∧
  (∀ x : ℝ, (x + 2) * (x - 3) < 0 → x > -2) :=
by sorry

end x_gt_neg_two_necessary_not_sufficient_l3703_370390


namespace quadratic_inequality_solution_l3703_370381

-- Define the quadratic inequality
def quadratic_inequality (a x : ℝ) : Prop := a * x^2 - 2 * x + a ≤ 0

-- Define the solution set
def solution_set (a : ℝ) : Set ℝ := {x : ℝ | quadratic_inequality a x}

theorem quadratic_inequality_solution :
  (∃! x, x ∈ solution_set 1) ∧
  (0 ∈ solution_set a ∧ -1 ∉ solution_set a → a ∈ Set.Ioc (-1) 0) :=
by sorry

end quadratic_inequality_solution_l3703_370381


namespace managers_salary_l3703_370384

/-- Proves that the manager's salary is 3400 given the conditions of the problem -/
theorem managers_salary (num_employees : ℕ) (avg_salary : ℚ) (salary_increase : ℚ) : 
  num_employees = 20 →
  avg_salary = 1300 →
  salary_increase = 100 →
  (num_employees * avg_salary + 3400) / (num_employees + 1) = avg_salary + salary_increase :=
by
  sorry

#check managers_salary

end managers_salary_l3703_370384


namespace count_good_pairs_l3703_370375

def is_good_pair (a p : ℕ) : Prop :=
  a > p ∧ (a^3 + p^3) % (a^2 - p^2) = 0

def is_prime_less_than_20 (p : ℕ) : Prop :=
  Nat.Prime p ∧ p < 20

theorem count_good_pairs :
  ∃ (S : Finset (ℕ × ℕ)), 
    S.card = 24 ∧
    (∀ (a p : ℕ), (a, p) ∈ S ↔ is_good_pair a p ∧ is_prime_less_than_20 p) :=
sorry

end count_good_pairs_l3703_370375


namespace enclosed_area_is_four_l3703_370320

-- Define the functions for the curve and the line
def f (x : ℝ) := 3 * x^2
def g (x : ℝ) := 3

-- Define the intersection points
def x₁ : ℝ := -1
def x₂ : ℝ := 1

-- State the theorem
theorem enclosed_area_is_four :
  (∫ (x : ℝ) in x₁..x₂, g x - f x) = 4 := by
  sorry

end enclosed_area_is_four_l3703_370320


namespace complement_intersection_eq_set_l3703_370373

def U : Finset ℕ := {1,2,3,4,5}
def A : Finset ℕ := {1,2,3}
def B : Finset ℕ := {3,4,5}

theorem complement_intersection_eq_set : 
  (U \ (A ∩ B)) = {1,2,4,5} := by sorry

end complement_intersection_eq_set_l3703_370373


namespace eliminate_uvw_l3703_370311

theorem eliminate_uvw (a b c d u v w : ℝ) 
  (eq1 : a = Real.cos u + Real.cos v + Real.cos w)
  (eq2 : b = Real.sin u + Real.sin v + Real.sin w)
  (eq3 : c = Real.cos (2*u) + Real.cos (2*v) + Real.cos (2*w))
  (eq4 : d = Real.sin (2*u) + Real.sin (2*v) + Real.sin (2*w)) :
  (a^2 - b^2 - c)^2 + (2*a*b - d)^2 = 4*(a^2 + b^2) := by
sorry

end eliminate_uvw_l3703_370311


namespace circle_centers_distance_l3703_370300

theorem circle_centers_distance (r R : ℝ) (h : r > 0 ∧ R > 0) :
  let d := Real.sqrt (R^2 + r^2 + (10/3) * R * r)
  ∃ (ext_tangent int_tangent : ℝ),
    ext_tangent > 0 ∧ int_tangent > 0 ∧
    ext_tangent = 2 * int_tangent ∧
    d^2 = (R + r)^2 - int_tangent^2 ∧
    d^2 = (R - r)^2 + ext_tangent^2 / 4 :=
by sorry

end circle_centers_distance_l3703_370300


namespace first_digit_of_1122001_base_3_in_base_9_l3703_370347

def base_3_to_10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (3 ^ i)) 0

def first_digit_base_9 (n : Nat) : Nat :=
  if n = 0 then 0
  else
    let log9 := Nat.log 9 n
    n / (9 ^ log9)

theorem first_digit_of_1122001_base_3_in_base_9 :
  let x := base_3_to_10 [1, 0, 0, 2, 2, 1, 1]
  first_digit_base_9 x = 1 := by
  sorry

end first_digit_of_1122001_base_3_in_base_9_l3703_370347


namespace father_son_age_problem_l3703_370334

theorem father_son_age_problem (x : ℕ) : x = 4 :=
by
  -- Son's current age
  let son_age : ℕ := 8
  -- Father's current age
  let father_age : ℕ := 4 * son_age
  -- In x years, father's age will be 3 times son's age
  have h : father_age + x = 3 * (son_age + x) := by sorry
  sorry

end father_son_age_problem_l3703_370334


namespace hyperbola_real_axis_x_implies_mn_negative_mn_negative_not_sufficient_for_real_axis_x_mn_negative_necessary_not_sufficient_l3703_370328

/-- Represents a hyperbola equation of the form x²/m + y²/n = 1 -/
structure Hyperbola (m n : ℝ) where
  equation : ∀ (x y : ℝ), x^2 / m + y^2 / n = 1

/-- Predicate to check if a hyperbola has its real axis on the x-axis -/
def has_real_axis_on_x (h : Hyperbola m n) : Prop :=
  m > 0 ∧ n < 0

theorem hyperbola_real_axis_x_implies_mn_negative 
  (m n : ℝ) (h : Hyperbola m n) :
  has_real_axis_on_x h → m * n < 0 := by
  sorry

theorem mn_negative_not_sufficient_for_real_axis_x :
  ∃ (m n : ℝ), m * n < 0 ∧ 
  ∃ (h : Hyperbola m n), ¬(has_real_axis_on_x h) := by
  sorry

/-- The main theorem stating that m * n < 0 is a necessary but not sufficient condition -/
theorem mn_negative_necessary_not_sufficient (m n : ℝ) (h : Hyperbola m n) :
  (has_real_axis_on_x h → m * n < 0) ∧
  ¬(m * n < 0 → has_real_axis_on_x h) := by
  sorry

end hyperbola_real_axis_x_implies_mn_negative_mn_negative_not_sufficient_for_real_axis_x_mn_negative_necessary_not_sufficient_l3703_370328


namespace players_who_quit_correct_players_who_quit_l3703_370308

theorem players_who_quit (initial_players : ℕ) (lives_per_player : ℕ) (total_lives : ℕ) : ℕ :=
  initial_players - (total_lives / lives_per_player)

theorem correct_players_who_quit :
  players_who_quit 8 3 15 = 3 := by
  sorry

end players_who_quit_correct_players_who_quit_l3703_370308


namespace water_current_speed_l3703_370314

/-- Proves that the speed of a water current is 2 km/h given specific swimming conditions -/
theorem water_current_speed 
  (swimmer_speed : ℝ) 
  (distance : ℝ) 
  (time : ℝ) 
  (h1 : swimmer_speed = 4) 
  (h2 : distance = 7) 
  (h3 : time = 3.5) : 
  ∃ (current_speed : ℝ), 
    current_speed = 2 ∧ 
    (swimmer_speed - current_speed) * time = distance :=
by sorry

end water_current_speed_l3703_370314


namespace manuscript_revision_cost_l3703_370321

/-- Given a manuscript typing service with the following conditions:
    - 100 total pages
    - 30 pages revised once
    - 20 pages revised twice
    - $5 per page for initial typing
    - $780 total cost
    Prove that the cost per page for each revision is $4. -/
theorem manuscript_revision_cost :
  let total_pages : ℕ := 100
  let pages_revised_once : ℕ := 30
  let pages_revised_twice : ℕ := 20
  let initial_cost_per_page : ℚ := 5
  let total_cost : ℚ := 780
  let revision_cost_per_page : ℚ := 4
  (total_pages * initial_cost_per_page + 
   (pages_revised_once * revision_cost_per_page + 
    pages_revised_twice * 2 * revision_cost_per_page) = total_cost) :=
by sorry

end manuscript_revision_cost_l3703_370321


namespace stratified_sampling_participation_l3703_370310

/-- Given a school with 1000 students, 300 of which are in the third year,
    prove that when 20 students are selected using stratified sampling,
    14 first and second-year students participate in the activity. -/
theorem stratified_sampling_participation
  (total_students : ℕ) (third_year_students : ℕ) (selected_students : ℕ)
  (h_total : total_students = 1000)
  (h_third_year : third_year_students = 300)
  (h_selected : selected_students = 20) :
  (selected_students : ℚ) * (total_students - third_year_students : ℚ) / total_students = 14 :=
by sorry

end stratified_sampling_participation_l3703_370310


namespace intersection_angle_implies_ratio_l3703_370322

-- Define the ellipse and hyperbola
def is_on_ellipse (x y a₁ b₁ : ℝ) : Prop := x^2 / a₁^2 + y^2 / b₁^2 = 1
def is_on_hyperbola (x y a₂ b₂ : ℝ) : Prop := x^2 / a₂^2 - y^2 / b₂^2 = 1

-- Define the common foci
def are_common_foci (F₁ F₂ : ℝ × ℝ) (a₁ b₁ a₂ b₂ : ℝ) : Prop := 
  ∃ c : ℝ, c^2 = a₁^2 - b₁^2 ∧ c^2 = a₂^2 + b₂^2 ∧
  F₁ = (c, 0) ∧ F₂ = (-c, 0)

-- Define the angle between foci
def angle_F₁PF₂ (P F₁ F₂ : ℝ × ℝ) : ℝ := sorry

-- Main theorem
theorem intersection_angle_implies_ratio 
  (P F₁ F₂ : ℝ × ℝ) (a₁ b₁ a₂ b₂ : ℝ) :
  a₁ > b₁ ∧ b₁ > 0 ∧ a₂ > 0 ∧ b₂ > 0 →
  is_on_ellipse P.1 P.2 a₁ b₁ →
  is_on_hyperbola P.1 P.2 a₂ b₂ →
  are_common_foci F₁ F₂ a₁ b₁ a₂ b₂ →
  angle_F₁PF₂ P F₁ F₂ = π / 3 →
  b₁ / b₂ = Real.sqrt 3 := by
  sorry

end intersection_angle_implies_ratio_l3703_370322


namespace ellipse_focal_length_l3703_370316

/-- An ellipse with equation y^2/2 + x^2 = 1 -/
def Ellipse := {p : ℝ × ℝ | p.2^2 / 2 + p.1^2 = 1}

/-- The focal length of an ellipse -/
def focalLength (E : Set (ℝ × ℝ)) : ℝ := sorry

/-- Theorem: The focal length of the ellipse y^2/2 + x^2 = 1 is 2 -/
theorem ellipse_focal_length : focalLength Ellipse = 2 := by sorry

end ellipse_focal_length_l3703_370316


namespace profit_percent_for_2_3_ratio_l3703_370362

/-- Given a cost price to selling price ratio of 2:3, the profit percent is 50%. -/
theorem profit_percent_for_2_3_ratio :
  ∀ (cp sp : ℝ), cp > 0 → sp > 0 →
  cp / sp = 2 / 3 →
  ((sp - cp) / cp) * 100 = 50 := by
sorry

end profit_percent_for_2_3_ratio_l3703_370362
