import Mathlib

namespace probability_of_arithmetic_progression_l1434_143401

/-- Represents an 8-sided die -/
def Die := Fin 8

/-- The number of dice rolled -/
def numDice : ℕ := 4

/-- Checks if a list of four numbers forms an arithmetic progression with common difference 2 -/
def isArithmeticProgression (nums : List ℕ) : Prop :=
  nums.length = numDice ∧
  ∃ a : ℕ, nums = [a, a + 2, a + 4, a + 6]

/-- The set of all possible outcomes when rolling four 8-sided dice -/
def allOutcomes : Finset (List Die) :=
  sorry

/-- The set of favorable outcomes (those forming the desired arithmetic progression) -/
def favorableOutcomes : Finset (List Die) :=
  sorry

/-- The probability of obtaining a favorable outcome -/
theorem probability_of_arithmetic_progression :
  (Finset.card favorableOutcomes : ℚ) / (Finset.card allOutcomes : ℚ) = 3 / 256 :=
sorry

end probability_of_arithmetic_progression_l1434_143401


namespace smallest_undefined_fraction_value_l1434_143447

theorem smallest_undefined_fraction_value : ∃ x : ℚ, x = 2/9 ∧ 
  (∀ y : ℚ, y < x → 9*y^2 - 74*y + 8 ≠ 0) ∧ 9*x^2 - 74*x + 8 = 0 := by
  sorry

end smallest_undefined_fraction_value_l1434_143447


namespace fraction_product_l1434_143488

theorem fraction_product : (2 : ℚ) / 3 * (4 : ℚ) / 9 = (8 : ℚ) / 27 := by
  sorry

end fraction_product_l1434_143488


namespace lin_peeled_fifteen_l1434_143431

/-- Represents the potato peeling scenario -/
structure PotatoPeeling where
  total_potatoes : ℕ
  homer_rate : ℕ
  christen_rate : ℕ
  lin_rate : ℕ
  christen_join_time : ℕ
  lin_join_time : ℕ

/-- Calculates the number of potatoes Lin peeled -/
def lin_potatoes_peeled (scenario : PotatoPeeling) : ℕ :=
  sorry

/-- Theorem stating that Lin peeled 15 potatoes -/
theorem lin_peeled_fifteen (scenario : PotatoPeeling) 
  (h1 : scenario.total_potatoes = 60)
  (h2 : scenario.homer_rate = 2)
  (h3 : scenario.christen_rate = 3)
  (h4 : scenario.lin_rate = 4)
  (h5 : scenario.christen_join_time = 6)
  (h6 : scenario.lin_join_time = 9) :
  lin_potatoes_peeled scenario = 15 := by
  sorry

end lin_peeled_fifteen_l1434_143431


namespace moms_balloons_l1434_143466

/-- The number of balloons Tommy's mom gave him -/
def balloons_from_mom (initial_balloons final_balloons : ℕ) : ℕ :=
  final_balloons - initial_balloons

/-- Proof that Tommy's mom gave him 34 balloons -/
theorem moms_balloons : balloons_from_mom 26 60 = 34 := by
  sorry

end moms_balloons_l1434_143466


namespace inscribed_rectangle_area_l1434_143463

/-- A rectangle inscribed in a triangle -/
structure InscribedRectangle where
  /-- The length of the rectangle's side along the triangle's base -/
  base_length : ℝ
  /-- The width of the rectangle -/
  width : ℝ
  /-- The length of the triangle's base -/
  triangle_base : ℝ
  /-- The height of the triangle -/
  triangle_height : ℝ
  /-- The width is one-third of the base length -/
  width_constraint : width = base_length / 3
  /-- The triangle's base is 15 inches -/
  triangle_base_length : triangle_base = 15
  /-- The triangle's height is 12 inches -/
  triangle_height_value : triangle_height = 12

/-- The area of the inscribed rectangle -/
def area (r : InscribedRectangle) : ℝ := r.base_length * r.width

/-- Theorem: The area of the inscribed rectangle is 10800/289 square inches -/
theorem inscribed_rectangle_area (r : InscribedRectangle) : area r = 10800 / 289 := by
  sorry

end inscribed_rectangle_area_l1434_143463


namespace function_sum_zero_at_five_sevenths_l1434_143475

-- Define the functions f and g
def f (x : ℝ) : ℝ := 5 * x - 6
def g (x : ℝ) : ℝ := 2 * x + 1

-- Theorem statement
theorem function_sum_zero_at_five_sevenths :
  ∃! a : ℝ, f a + g a = 0 ∧ a = 5 / 7 := by
  sorry

end function_sum_zero_at_five_sevenths_l1434_143475


namespace sum_of_arguments_l1434_143477

def complex_equation (z : ℂ) : Prop := z^6 = 64 * Complex.I

theorem sum_of_arguments (z₁ z₂ z₃ z₄ z₅ z₆ : ℂ) 
  (h₁ : complex_equation z₁)
  (h₂ : complex_equation z₂)
  (h₃ : complex_equation z₃)
  (h₄ : complex_equation z₄)
  (h₅ : complex_equation z₅)
  (h₆ : complex_equation z₆)
  (distinct : z₁ ≠ z₂ ∧ z₁ ≠ z₃ ∧ z₁ ≠ z₄ ∧ z₁ ≠ z₅ ∧ z₁ ≠ z₆ ∧
              z₂ ≠ z₃ ∧ z₂ ≠ z₄ ∧ z₂ ≠ z₅ ∧ z₂ ≠ z₆ ∧
              z₃ ≠ z₄ ∧ z₃ ≠ z₅ ∧ z₃ ≠ z₆ ∧
              z₄ ≠ z₅ ∧ z₄ ≠ z₆ ∧
              z₅ ≠ z₆) :
  (Complex.arg z₁ + Complex.arg z₂ + Complex.arg z₃ + 
   Complex.arg z₄ + Complex.arg z₅ + Complex.arg z₆) * (180 / Real.pi) = 990 :=
by sorry

end sum_of_arguments_l1434_143477


namespace edward_binders_l1434_143445

/-- The number of baseball cards Edward had -/
def total_cards : ℕ := 763

/-- The number of cards in each binder -/
def cards_per_binder : ℕ := 109

/-- The number of binders Edward had -/
def number_of_binders : ℕ := total_cards / cards_per_binder

theorem edward_binders : number_of_binders = 7 := by
  sorry

end edward_binders_l1434_143445


namespace problem_statement_l1434_143430

theorem problem_statement (m n : ℝ) (h : 5 * m + 3 * n = 2) : 
  10 * m + 6 * n - 5 = -1 := by
  sorry

end problem_statement_l1434_143430


namespace rational_expressions_theorem_l1434_143438

theorem rational_expressions_theorem 
  (a b c : ℚ) 
  (ha : a ≠ 0) 
  (hb : b ≠ 0) 
  (hc : c ≠ 0) :
  (a < 0 → a / |a| = -1) ∧ 
  (∃ m : ℚ, m = -2 ∧ 
    ∀ x y z : ℚ, x ≠ 0 → y ≠ 0 → z ≠ 0 → 
      m ≤ (x*y/|x*y| + |y*z|/(y*z) + z*x/|z*x| + |x*y*z|/(x*y*z))) :=
by sorry

end rational_expressions_theorem_l1434_143438


namespace video_recorder_price_l1434_143411

def employee_price (wholesale_cost : ℝ) (markup_percentage : ℝ) (discount_percentage : ℝ) : ℝ :=
  let retail_price := wholesale_cost * (1 + markup_percentage)
  let discount := retail_price * discount_percentage
  retail_price - discount

theorem video_recorder_price :
  employee_price 200 0.2 0.2 = 192 := by
  sorry

end video_recorder_price_l1434_143411


namespace antonov_remaining_packs_l1434_143465

/-- Calculates the number of candy packs remaining after giving away one pack -/
def remaining_packs (total_candies : ℕ) (candies_per_pack : ℕ) : ℕ :=
  (total_candies - candies_per_pack) / candies_per_pack

/-- Proves that Antonov has 2 packs of candy remaining -/
theorem antonov_remaining_packs :
  let total_candies : ℕ := 60
  let candies_per_pack : ℕ := 20
  remaining_packs total_candies candies_per_pack = 2 := by
  sorry

end antonov_remaining_packs_l1434_143465


namespace closest_point_on_line_l1434_143410

/-- The line y = -2x + 3 --/
def line (x : ℝ) : ℝ := -2 * x + 3

/-- The point we're finding the closest point to --/
def point : ℝ × ℝ := (2, -1)

/-- The squared distance between two points --/
def squared_distance (p q : ℝ × ℝ) : ℝ :=
  (p.1 - q.1)^2 + (p.2 - q.2)^2

theorem closest_point_on_line :
  ∀ x : ℝ, squared_distance (x, line x) point ≥ squared_distance (2, line 2) point :=
by sorry

end closest_point_on_line_l1434_143410


namespace sufficient_not_necessary_l1434_143499

theorem sufficient_not_necessary (x : ℝ) : 
  (∀ x, Real.sqrt (x + 2) - Real.sqrt (1 - 2*x) > 0 → (x + 1) / (x - 1) ≤ 0) ∧
  (∃ x, (x + 1) / (x - 1) ≤ 0 ∧ Real.sqrt (x + 2) - Real.sqrt (1 - 2*x) ≤ 0) :=
by sorry

end sufficient_not_necessary_l1434_143499


namespace max_value_reciprocal_sum_l1434_143479

theorem max_value_reciprocal_sum (x y a b : ℝ) 
  (ha : a > 1) 
  (hb : b > 1) 
  (hax : a^x = 3) 
  (hby : b^y = 3) 
  (hab : a + b = 2 * Real.sqrt 3) : 
  (∃ (z : ℝ), ∀ (w : ℝ), 1/x + 1/y ≤ w → w ≤ z) ∧ 
  (∃ (x0 y0 : ℝ), 1/x0 + 1/y0 = 1 ∧ 
    a^x0 = 3 ∧ b^y0 = 3 ∧ a + b = 2 * Real.sqrt 3) :=
sorry

end max_value_reciprocal_sum_l1434_143479


namespace a_10_equals_505_l1434_143464

def sequence_a (n : ℕ) : ℕ :=
  if n = 1 then 1
  else
    let start := (n * (n - 1)) / 2 + 1
    (start + start + n - 1) * n / 2

theorem a_10_equals_505 : sequence_a 10 = 505 := by
  sorry

end a_10_equals_505_l1434_143464


namespace negative_fraction_comparison_l1434_143450

theorem negative_fraction_comparison : -3/4 > -4/5 := by sorry

end negative_fraction_comparison_l1434_143450


namespace power_function_through_point_l1434_143436

/-- Given a power function f(x) = x^n that passes through (2, √2/2), prove f(4) = 1/2 -/
theorem power_function_through_point (f : ℝ → ℝ) (n : ℝ) :
  (∀ x, f x = x^n) →
  f 2 = Real.sqrt 2 / 2 →
  f 4 = 1 / 2 := by
sorry

end power_function_through_point_l1434_143436


namespace street_painting_cost_l1434_143403

/-- Calculates the total cost for painting house numbers on a street --/
def total_painting_cost (south_start : ℕ) (north_start : ℕ) (common_diff : ℕ) (houses_per_side : ℕ) : ℚ :=
  let south_end := south_start + common_diff * (houses_per_side - 1)
  let north_end := north_start + common_diff * (houses_per_side - 1)
  let south_two_digit := min houses_per_side (((99 - south_start) / common_diff) + 1)
  let north_two_digit := min houses_per_side (((99 - north_start) / common_diff) + 1)
  let south_three_digit := houses_per_side - south_two_digit
  let north_three_digit := houses_per_side - north_two_digit
  (2 * south_two_digit + 1.5 * south_three_digit + 2 * north_two_digit + 1.5 * north_three_digit : ℚ)

/-- The theorem stating the total cost for the given street configuration --/
theorem street_painting_cost :
  total_painting_cost 5 2 7 25 = 88.5 := by
  sorry

end street_painting_cost_l1434_143403


namespace sin_theta_value_l1434_143492

theorem sin_theta_value (θ : Real) 
  (h1 : 10 * Real.tan θ = 4 * Real.cos θ) 
  (h2 : 0 < θ) (h3 : θ < π) : 
  Real.sin θ = 1/2 := by
  sorry

end sin_theta_value_l1434_143492


namespace rock_skipping_total_l1434_143467

/-- The number of times Bob can skip a rock -/
def bob_skips : ℕ := 12

/-- The number of times Jim can skip a rock -/
def jim_skips : ℕ := 15

/-- The number of rocks each person skipped -/
def rocks_skipped : ℕ := 10

/-- The total number of skips achieved by Bob and Jim -/
def total_skips : ℕ := bob_skips * rocks_skipped + jim_skips * rocks_skipped

theorem rock_skipping_total : total_skips = 270 := by
  sorry

end rock_skipping_total_l1434_143467


namespace binomial_expansion_coefficient_l1434_143432

theorem binomial_expansion_coefficient (n : ℕ) : 
  (Nat.choose n 2) * 3^2 = 54 → n = 4 := by
sorry

end binomial_expansion_coefficient_l1434_143432


namespace test_questions_missed_l1434_143443

theorem test_questions_missed (friend_missed : ℕ) (your_missed : ℕ) : 
  your_missed = 5 * friend_missed →
  your_missed + friend_missed = 216 →
  your_missed = 180 := by
sorry

end test_questions_missed_l1434_143443


namespace max_value_x_sqrt_1_minus_4x_squared_l1434_143468

theorem max_value_x_sqrt_1_minus_4x_squared :
  (∃ (x : ℝ), x > 0 ∧ x * Real.sqrt (1 - 4 * x^2) = 1/4) ∧
  (∀ (x : ℝ), x > 0 → x * Real.sqrt (1 - 4 * x^2) ≤ 1/4) := by
  sorry

end max_value_x_sqrt_1_minus_4x_squared_l1434_143468


namespace x_value_l1434_143425

theorem x_value : ∃ x : ℝ, (3 * x) / 7 = 21 ∧ x = 49 := by
  sorry

end x_value_l1434_143425


namespace zeros_of_f_l1434_143489

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 16*x

-- State the theorem
theorem zeros_of_f :
  {x : ℝ | f x = 0} = {-4, 0, 4} := by sorry

end zeros_of_f_l1434_143489


namespace min_value_of_3a_plus_2_l1434_143448

theorem min_value_of_3a_plus_2 (a : ℝ) (h : 8 * a^2 + 7 * a + 6 = 5) :
  ∃ (m : ℝ), (3 * a + 2 ≥ m) ∧ (∀ x, 8 * x^2 + 7 * x + 6 = 5 → 3 * x + 2 ≥ m) ∧ m = -1 :=
by sorry

end min_value_of_3a_plus_2_l1434_143448


namespace larry_wins_probability_l1434_143491

theorem larry_wins_probability (p : ℝ) (q : ℝ) (hp : p = 1/3) (hq : q = 1/4) :
  let win_prob := p / (1 - (1 - p) * (1 - q))
  win_prob = 2/3 := by sorry

end larry_wins_probability_l1434_143491


namespace yellow_face_probability_l1434_143497

/-- The probability of rolling a yellow face on a modified 10-sided die -/
theorem yellow_face_probability (total_faces : ℕ) (yellow_faces : ℕ) : 
  total_faces = 10 → yellow_faces = 4 → (yellow_faces : ℚ) / total_faces = 2 / 5 := by
  sorry

end yellow_face_probability_l1434_143497


namespace area_equality_l1434_143441

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parallelogram defined by four vertices -/
structure Parallelogram where
  E : Point
  F : Point
  G : Point
  H : Point

/-- Calculates the area of a parallelogram -/
def area (p : Parallelogram) : ℝ := sorry

/-- Calculates the area of the part of the parallelogram below the x-axis -/
def areaBelow (p : Parallelogram) : ℝ := sorry

/-- Calculates the area of the part of the parallelogram above the x-axis -/
def areaAbove (p : Parallelogram) : ℝ := sorry

/-- Theorem: For the given parallelogram, the area below the x-axis equals the area above -/
theorem area_equality (p : Parallelogram) 
  (h1 : p.E = ⟨-1, 2⟩) 
  (h2 : p.F = ⟨5, 2⟩) 
  (h3 : p.G = ⟨1, -2⟩) 
  (h4 : p.H = ⟨-5, -2⟩) : 
  areaBelow p = areaAbove p := by sorry

end area_equality_l1434_143441


namespace walkers_speed_l1434_143482

/-- Proves that given the conditions of the problem, A's walking speed is 10 kmph -/
theorem walkers_speed (v : ℝ) : 
  (∃ t : ℝ, v * (t + 7) = 20 * t) →  -- B catches up with A
  (∃ t : ℝ, v * (t + 7) = 140) →     -- Distance traveled is 140 km
  v = 10 := by sorry

end walkers_speed_l1434_143482


namespace min_output_no_loss_l1434_143404

/-- The total cost function for a product -/
def total_cost (x : ℕ) : ℚ :=
  3000 + 20 * x - 0.1 * x^2

/-- The condition for no loss -/
def no_loss (x : ℕ) : Prop :=
  25 * x ≥ total_cost x

/-- The theorem stating the minimum output for no loss -/
theorem min_output_no_loss :
  ∃ (x : ℕ), x > 0 ∧ x < 240 ∧ no_loss x ∧
  ∀ (y : ℕ), y > 0 ∧ y < 240 ∧ no_loss y → x ≤ y ∧ x = 150 :=
sorry

end min_output_no_loss_l1434_143404


namespace modular_inverse_28_mod_29_l1434_143419

theorem modular_inverse_28_mod_29 : ∃ x : ℕ, x ≤ 28 ∧ (28 * x) % 29 = 1 := by
  sorry

end modular_inverse_28_mod_29_l1434_143419


namespace classroom_gpa_l1434_143405

theorem classroom_gpa (gpa_two_thirds : ℝ) (gpa_whole : ℝ) (gpa_one_third : ℝ) :
  gpa_two_thirds = 66 →
  gpa_whole = 64 →
  (1/3 : ℝ) * gpa_one_third + (2/3 : ℝ) * gpa_two_thirds = gpa_whole →
  gpa_one_third = 60 := by
sorry

end classroom_gpa_l1434_143405


namespace quadratic_max_value_l1434_143417

-- Define the quadratic function
def f (x : ℝ) : ℝ := -3 * (x - 2)^2 - 3

-- Theorem statement
theorem quadratic_max_value :
  ∃ (max : ℝ), max = -3 ∧ ∀ (x : ℝ), f x ≤ max :=
sorry

end quadratic_max_value_l1434_143417


namespace parabola_sum_l1434_143493

/-- A parabola with equation y = px^2 + qx + r, vertex (3, 10), vertical axis of symmetry, and containing the point (0, 7) -/
structure Parabola where
  p : ℝ
  q : ℝ
  r : ℝ
  vertex_x : ℝ := 3
  vertex_y : ℝ := 10
  point_x : ℝ := 0
  point_y : ℝ := 7
  eq_at_vertex : 10 = p * 3^2 + q * 3 + r
  eq_at_point : 7 = p * 0^2 + q * 0 + r
  vertical_symmetry : ∀ (x : ℝ), p * (vertex_x - x)^2 + vertex_y = p * (vertex_x + x)^2 + vertex_y

theorem parabola_sum (par : Parabola) : par.p + par.q + par.r = 26/3 := by
  sorry

end parabola_sum_l1434_143493


namespace sum_first_six_primes_mod_seventh_prime_l1434_143483

def first_six_primes : List Nat := [2, 3, 5, 7, 11, 13]
def seventh_prime : Nat := 17

theorem sum_first_six_primes_mod_seventh_prime :
  (first_six_primes.sum % seventh_prime) = 7 := by
  sorry

end sum_first_six_primes_mod_seventh_prime_l1434_143483


namespace security_deposit_is_1110_l1434_143408

/-- Calculates the security deposit for a cabin rental -/
def calculate_security_deposit (daily_rate : ℚ) (duration : ℕ) (pet_fee : ℚ) 
  (service_fee_rate : ℚ) (deposit_rate : ℚ) : ℚ :=
  let subtotal := daily_rate * duration + pet_fee
  let service_fee := service_fee_rate * subtotal
  let total := subtotal + service_fee
  deposit_rate * total

/-- Theorem stating that the security deposit for the given conditions is $1110.00 -/
theorem security_deposit_is_1110 :
  calculate_security_deposit 125 14 100 (1/5) (1/2) = 1110 := by
  sorry

#eval calculate_security_deposit 125 14 100 (1/5) (1/2)

end security_deposit_is_1110_l1434_143408


namespace changfei_class_problem_l1434_143433

theorem changfei_class_problem (m n : ℕ+) 
  (h : m.val * (m.val - 1) + m.val * n.val + n.val = 51) : 
  m.val + n.val = 9 := by
sorry

end changfei_class_problem_l1434_143433


namespace sum_of_cubes_l1434_143454

theorem sum_of_cubes (x y : ℝ) (h1 : x + y = 3) (h2 : x * y = 2) (h3 : x = 2 * y) : x^3 + y^3 = 9 := by
  sorry

end sum_of_cubes_l1434_143454


namespace geometric_sequence_proof_l1434_143424

def arithmetic_sequence (n : ℕ) : ℝ := 2 * n - 1

def kth_order_derivative_sequence (k m n : ℕ) : ℝ :=
  2^(k+2) * m - 2^(k+2) + 1

theorem geometric_sequence_proof (m : ℕ) (hm : m ≥ 2) :
  ∀ n : ℕ, n ≥ 1 → 
    (kth_order_derivative_sequence n m (n+1) - 1) / (kth_order_derivative_sequence n m n - 1) = 2 :=
by sorry

end geometric_sequence_proof_l1434_143424


namespace m_range_l1434_143494

def A (m : ℝ) : Set ℝ := {x | m + 1 ≤ x ∧ x ≤ 2*m - 1}

def B : Set ℝ := {x | x^2 - 2*x - 15 ≤ 0}

theorem m_range (m : ℝ) :
  (∃ x, x ∈ A m) ∧ (A m ⊆ B) → 2 ≤ m ∧ m ≤ 3 :=
by sorry

end m_range_l1434_143494


namespace original_cost_price_correct_l1434_143407

/-- Represents the original cost price in euros -/
def original_cost_price : ℝ := 55.50

/-- Represents the selling price in dollars -/
def selling_price : ℝ := 100

/-- Represents the profit percentage -/
def profit_percentage : ℝ := 0.30

/-- Represents the exchange rate (dollars per euro) -/
def exchange_rate : ℝ := 1.2

/-- Represents the maintenance cost percentage -/
def maintenance_cost_percentage : ℝ := 0.05

/-- Represents the tax rate for the first 50 euros -/
def tax_rate_first_50 : ℝ := 0.10

/-- Represents the tax rate for amounts above 50 euros -/
def tax_rate_above_50 : ℝ := 0.15

/-- Represents the threshold for the tiered tax system -/
def tax_threshold : ℝ := 50

theorem original_cost_price_correct :
  let cost_price_dollars := selling_price / (1 + profit_percentage)
  let cost_price_euros := cost_price_dollars / exchange_rate
  let maintenance_cost := original_cost_price * maintenance_cost_percentage
  let tax_first_50 := min original_cost_price tax_threshold * tax_rate_first_50
  let tax_above_50 := max (original_cost_price - tax_threshold) 0 * tax_rate_above_50
  cost_price_euros = original_cost_price + maintenance_cost + tax_first_50 + tax_above_50 :=
by sorry

#check original_cost_price_correct

end original_cost_price_correct_l1434_143407


namespace min_tiles_to_cover_region_l1434_143462

/-- The number of tiles needed to cover a rectangular region -/
def tiles_needed (tile_width : ℕ) (tile_height : ℕ) (region_width : ℕ) (region_height : ℕ) : ℕ :=
  let region_area := region_width * region_height
  let tile_area := tile_width * tile_height
  (region_area + tile_area - 1) / tile_area

/-- Conversion factor from feet to inches -/
def feet_to_inches : ℕ := 12

theorem min_tiles_to_cover_region :
  tiles_needed 5 7 (3 * feet_to_inches) (7 * feet_to_inches) = 87 := by
  sorry

#eval tiles_needed 5 7 (3 * feet_to_inches) (7 * feet_to_inches)

end min_tiles_to_cover_region_l1434_143462


namespace evaluate_f_l1434_143459

def f (x : ℝ) : ℝ := 3 * x^2 + 5 * x - 7

theorem evaluate_f : 3 * f 2 - 2 * f (-2) = 55 := by
  sorry

end evaluate_f_l1434_143459


namespace simplify_and_evaluate_l1434_143449

theorem simplify_and_evaluate (a b : ℝ) :
  -(-a^2 + 2*a*b + b^2) + (-a^2 - a*b + b^2) = -3*a*b ∧
  (a*b = 1 → -(-a^2 + 2*a*b + b^2) + (-a^2 - a*b + b^2) = -3) := by
  sorry

end simplify_and_evaluate_l1434_143449


namespace multiplication_and_addition_l1434_143484

theorem multiplication_and_addition : 2 * (-2) + (-3) = -7 := by
  sorry

end multiplication_and_addition_l1434_143484


namespace purple_to_seafoam_ratio_is_one_fourth_l1434_143485

/-- The number of skirts in Azure Valley -/
def azure_skirts : ℕ := 60

/-- The number of skirts in Purple Valley -/
def purple_skirts : ℕ := 10

/-- The number of skirts in Seafoam Valley -/
def seafoam_skirts : ℕ := (2 * azure_skirts) / 3

/-- The ratio of skirts in Purple Valley to Seafoam Valley -/
def purple_to_seafoam_ratio : ℚ := purple_skirts / seafoam_skirts

theorem purple_to_seafoam_ratio_is_one_fourth :
  purple_to_seafoam_ratio = 1 / 4 := by
  sorry

end purple_to_seafoam_ratio_is_one_fourth_l1434_143485


namespace line_param_correct_l1434_143437

/-- The line y = 2x - 6 parameterized as (x, y) = (r, 2) + t(3, k) -/
def line_param (r k t : ℝ) : ℝ × ℝ :=
  (r + 3 * t, 2 + k * t)

/-- The line equation y = 2x - 6 -/
def line_eq (x y : ℝ) : Prop :=
  y = 2 * x - 6

theorem line_param_correct (r k : ℝ) : 
  (∀ t, line_eq (line_param r k t).1 (line_param r k t).2) ↔ r = 4 ∧ k = 6 := by
  sorry

end line_param_correct_l1434_143437


namespace pm25_scientific_notation_l1434_143400

theorem pm25_scientific_notation : 
  ∃ (a : ℝ) (n : ℤ), 0.0000025 = a * (10 : ℝ) ^ n ∧ 1 ≤ |a| ∧ |a| < 10 ∧ n = -6 ∧ a = 2.5 := by
  sorry

end pm25_scientific_notation_l1434_143400


namespace fraction_of_product_l1434_143496

theorem fraction_of_product (x : ℚ) : x * ((3 / 4 : ℚ) * (2 / 5 : ℚ) * 5040) = 756.0000000000001 → x = 1 / 2 := by
  sorry

end fraction_of_product_l1434_143496


namespace sequence_first_element_l1434_143409

def sequence_property (a b c d e : ℚ) : Prop :=
  c = a * b ∧ d = b * c ∧ e = c * d

theorem sequence_first_element :
  ∀ a b c d e : ℚ,
    sequence_property a b c d e →
    c = 3 →
    e = 18 →
    a = 3/2 := by
  sorry

end sequence_first_element_l1434_143409


namespace reservoir_overflow_time_l1434_143434

/-- Represents the state of a reservoir with four pipes -/
structure ReservoirSystem where
  fill_rate_a : ℚ  -- Rate at which Pipe A fills the reservoir (in reservoir/hour)
  fill_rate_c : ℚ  -- Rate at which Pipe C fills the reservoir (in reservoir/hour)
  drain_rate_b : ℚ  -- Rate at which Pipe B drains the reservoir (in reservoir/hour)
  drain_rate_d : ℚ  -- Rate at which Pipe D drains the reservoir (in reservoir/hour)
  initial_level : ℚ  -- Initial water level in the reservoir (as a fraction of full)

/-- Calculates the time until the reservoir overflows -/
def time_to_overflow (sys : ReservoirSystem) : ℚ :=
  sorry

/-- Theorem stating the time to overflow for the given reservoir system -/
theorem reservoir_overflow_time : 
  let sys : ReservoirSystem := {
    fill_rate_a := 1/3,
    fill_rate_c := 1/5,
    drain_rate_b := -1/4,
    drain_rate_d := -1/6,
    initial_level := 1/6
  }
  time_to_overflow sys = 83/4 := by
  sorry


end reservoir_overflow_time_l1434_143434


namespace abs_m_minus_one_geq_abs_m_minus_one_l1434_143498

theorem abs_m_minus_one_geq_abs_m_minus_one (m : ℝ) : |m - 1| ≥ |m| - 1 := by
  sorry

end abs_m_minus_one_geq_abs_m_minus_one_l1434_143498


namespace max_a_cubic_function_l1434_143406

/-- Given a cubic function f(x) = a x^3 + b x^2 + c x + d where a ≠ 0,
    and |f'(x)| ≤ 1 for 0 ≤ x ≤ 1, the maximum value of a is 8/3. -/
theorem max_a_cubic_function (a b c d : ℝ) (h₁ : a ≠ 0) :
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → |3 * a * x^2 + 2 * b * x + c| ≤ 1) →
  a ≤ 8/3 ∧ ∃ b c : ℝ, a = 8/3 ∧ 
    (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → |3 * (8/3) * x^2 + 2 * b * x + c| ≤ 1) :=
by sorry

end max_a_cubic_function_l1434_143406


namespace distance_between_intersections_l1434_143427

-- Define the two curves
def curve1 (x y : ℝ) : Prop := x = y^4
def curve2 (x y : ℝ) : Prop := x - y^2 = 1

-- Define the intersection points
def intersection_points : Set (ℝ × ℝ) :=
  {p | ∃ (x y : ℝ), p = (x, y) ∧ curve1 x y ∧ curve2 x y}

-- Theorem statement
theorem distance_between_intersections :
  ∃ (p1 p2 : ℝ × ℝ), p1 ∈ intersection_points ∧ p2 ∈ intersection_points ∧
  p1 ≠ p2 ∧ Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2) = Real.sqrt (1 + Real.sqrt 5) :=
sorry

end distance_between_intersections_l1434_143427


namespace logarithm_equation_solution_l1434_143416

theorem logarithm_equation_solution (x : ℝ) (hx_pos : x > 0) (hx_neq_one : x ≠ 1) :
  (Real.log 2 / Real.log x) * (Real.log 2 / Real.log (2 * x)) = Real.log 2 / Real.log (4 * x) →
  x = 2 ^ Real.sqrt 2 ∨ x = 2 ^ (-Real.sqrt 2) := by
  sorry

end logarithm_equation_solution_l1434_143416


namespace problem_solution_l1434_143472

theorem problem_solution : 
  (∃ x : ℝ, 1/x < x + 1) ∧ 
  (¬(∃ n : ℕ, n^2 > 2^n) ↔ (∀ n : ℕ, n^2 ≤ 2^n)) := by
  sorry

end problem_solution_l1434_143472


namespace fraction_product_l1434_143414

theorem fraction_product : (2 : ℚ) / 9 * (5 : ℚ) / 8 = (5 : ℚ) / 36 := by
  sorry

end fraction_product_l1434_143414


namespace line_slope_l1434_143481

/-- Given a line with equation 3y + 2x = 6x - 9, its slope is -4/3 -/
theorem line_slope (x y : ℝ) : 3*y + 2*x = 6*x - 9 → (y - 3 = (-4/3) * (x - 0)) := by
  sorry

end line_slope_l1434_143481


namespace decimal_digit_17_99_l1434_143469

/-- The fraction we're examining -/
def f : ℚ := 17 / 99

/-- The position of the digit we're looking for -/
def n : ℕ := 150

/-- Function to get the nth digit after the decimal point in the decimal representation of a rational number -/
noncomputable def nth_decimal_digit (q : ℚ) (n : ℕ) : ℕ := sorry

/-- Theorem stating that the 150th digit after the decimal point in 17/99 is 7 -/
theorem decimal_digit_17_99 : nth_decimal_digit f n = 7 := by sorry

end decimal_digit_17_99_l1434_143469


namespace f_negative_a_l1434_143453

noncomputable def f (x : ℝ) : ℝ := x^2 * Real.log (-x + Real.sqrt (x^2 + 1)) + 1

theorem f_negative_a (a : ℝ) (h : f a = 11) : f (-a) = -9 := by
  sorry

end f_negative_a_l1434_143453


namespace fractional_equation_solution_l1434_143451

theorem fractional_equation_solution (x : ℝ) (hx : x ≠ 0) (hx2 : 2 * x - 1 ≠ 0) :
  (2 / x = 3 / (2 * x - 1)) ↔ x = 2 := by sorry

end fractional_equation_solution_l1434_143451


namespace fraction_sum_equals_decimal_l1434_143442

theorem fraction_sum_equals_decimal : (2 / 5 : ℚ) + (2 / 50 : ℚ) + (2 / 500 : ℚ) = 0.444 := by
  sorry

end fraction_sum_equals_decimal_l1434_143442


namespace trout_ratio_is_three_to_one_l1434_143429

/-- The ratio of trouts caught by Caleb's dad to those caught by Caleb -/
def trout_ratio (caleb_trouts : ℕ) (dad_extra_trouts : ℕ) : ℚ :=
  (caleb_trouts + dad_extra_trouts) / caleb_trouts

theorem trout_ratio_is_three_to_one :
  trout_ratio 2 4 = 3 / 1 := by
  sorry

end trout_ratio_is_three_to_one_l1434_143429


namespace mp3_song_count_l1434_143457

/-- Given an initial number of songs, number of deleted songs, and number of added songs,
    calculate the final number of songs on the mp3 player. -/
def final_song_count (initial : ℕ) (deleted : ℕ) (added : ℕ) : ℕ :=
  initial - deleted + added

/-- Theorem stating that given the specific numbers in the problem,
    the final song count is 64. -/
theorem mp3_song_count : final_song_count 34 14 44 = 64 := by
  sorry

end mp3_song_count_l1434_143457


namespace lcm_hcf_relation_l1434_143412

theorem lcm_hcf_relation (d c : ℕ) (h1 : d > 0) (h2 : Nat.lcm 76 d = 456) (h3 : Nat.gcd 76 d = c) : d = 24 := by
  sorry

end lcm_hcf_relation_l1434_143412


namespace smallest_n_for_roots_of_unity_l1434_143402

theorem smallest_n_for_roots_of_unity : ∃ (n : ℕ), n > 0 ∧ (∀ (z : ℂ), z^4 + z^3 + 1 = 0 → z^n = 1) ∧ (∀ (m : ℕ), m > 0 → (∀ (z : ℂ), z^4 + z^3 + 1 = 0 → z^m = 1) → m ≥ n) ∧ n = 5 := by
  sorry

end smallest_n_for_roots_of_unity_l1434_143402


namespace january_oil_bill_l1434_143490

theorem january_oil_bill (february_bill january_bill : ℚ) : 
  (february_bill / january_bill = 5 / 4) →
  ((february_bill + 45) / january_bill = 3 / 2) →
  january_bill = 180 := by
sorry

end january_oil_bill_l1434_143490


namespace not_fifteen_percent_less_l1434_143428

theorem not_fifteen_percent_less (A B : ℝ) (h : A = B * (1 + 0.15)) : 
  B ≠ A * (1 - 0.15) := by
sorry

end not_fifteen_percent_less_l1434_143428


namespace f_of_4_equals_22_l1434_143440

/-- Given a function f(x) = 5x + 2, prove that f(4) = 22 -/
theorem f_of_4_equals_22 :
  let f : ℝ → ℝ := λ x ↦ 5 * x + 2
  f 4 = 22 := by sorry

end f_of_4_equals_22_l1434_143440


namespace complex_fraction_power_l1434_143435

theorem complex_fraction_power (i : ℂ) : i^2 = -1 → ((1 + i) / (1 - i))^2013 = i := by
  sorry

end complex_fraction_power_l1434_143435


namespace discount_order_difference_l1434_143495

def original_price : ℝ := 50
def flat_discount : ℝ := 5
def percentage_discount : ℝ := 0.1

def price_flat_then_percent : ℝ := (original_price - flat_discount) * (1 - percentage_discount)
def price_percent_then_flat : ℝ := original_price * (1 - percentage_discount) - flat_discount

theorem discount_order_difference :
  price_flat_then_percent - price_percent_then_flat = 0.5 := by
  sorry

end discount_order_difference_l1434_143495


namespace line_inclination_45_degrees_l1434_143422

/-- Given a line passing through points (-2, 1) and (m, 3) with an inclination angle of 45°, prove that m = 0 -/
theorem line_inclination_45_degrees (m : ℝ) : 
  (3 - 1) / (m + 2) = Real.tan (π / 4) → m = 0 := by
  sorry

end line_inclination_45_degrees_l1434_143422


namespace min_value_m_plus_n_l1434_143473

theorem min_value_m_plus_n (m n : ℝ) : 
  m * 1 + n * 1 - 3 * m * n = 0 → 
  m * n > 0 → 
  m + n ≥ 4/3 ∧ ∃ (m₀ n₀ : ℝ), m₀ * 1 + n₀ * 1 - 3 * m₀ * n₀ = 0 ∧ m₀ * n₀ > 0 ∧ m₀ + n₀ = 4/3 :=
by sorry

end min_value_m_plus_n_l1434_143473


namespace b_over_a_range_l1434_143461

-- Define an acute triangle
structure AcuteTriangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real
  acute : A > 0 ∧ B > 0 ∧ C > 0 ∧ A + B + C = π
  sides_positive : a > 0 ∧ b > 0 ∧ c > 0
  sine_law : a / Real.sin A = b / Real.sin B
  B_eq_2A : B = 2 * A

-- Theorem statement
theorem b_over_a_range (t : AcuteTriangle) : Real.sqrt 2 < t.b / t.a ∧ t.b / t.a < Real.sqrt 3 := by
  sorry

end b_over_a_range_l1434_143461


namespace nicole_clothes_proof_l1434_143476

/-- Calculates the total number of clothing pieces Nicole ends up with --/
def nicole_total_clothes (nicole_initial : ℕ) : ℕ :=
  let sister1 := nicole_initial / 2
  let sister2 := nicole_initial + 2
  let sister3 := (nicole_initial + sister1 + sister2) / 3
  nicole_initial + sister1 + sister2 + sister3

/-- Proves that Nicole ends up with 36 pieces of clothing --/
theorem nicole_clothes_proof :
  nicole_total_clothes 10 = 36 := by
  sorry

#eval nicole_total_clothes 10

end nicole_clothes_proof_l1434_143476


namespace initial_mean_calculation_l1434_143415

theorem initial_mean_calculation (n : ℕ) (wrong_value correct_value : ℝ) (corrected_mean : ℝ) :
  n = 50 ∧ 
  wrong_value = 23 ∧ 
  correct_value = 46 ∧ 
  corrected_mean = 36.5 →
  (n * corrected_mean - (correct_value - wrong_value)) / n = 36.04 := by
  sorry

end initial_mean_calculation_l1434_143415


namespace prob_log_inequality_l1434_143446

open Real MeasureTheory ProbabilityTheory

/-- The probability of selecting a number x from [0,3] such that -1 ≤ log_(1/2)(x + 1/2) ≤ 1 is 1/2 -/
theorem prob_log_inequality (μ : Measure ℝ) [IsProbabilityMeasure μ] : 
  μ {x ∈ Set.Icc 0 3 | -1 ≤ log (x + 1/2) / log (1/2) ∧ log (x + 1/2) / log (1/2) ≤ 1} = 1/2 := by
  sorry


end prob_log_inequality_l1434_143446


namespace florist_chrysanthemums_l1434_143418

theorem florist_chrysanthemums (narcissus : ℕ) (bouquets : ℕ) (flowers_per_bouquet : ℕ) 
  (h1 : narcissus = 75)
  (h2 : bouquets = 33)
  (h3 : flowers_per_bouquet = 5)
  (h4 : narcissus + chrysanthemums = bouquets * flowers_per_bouquet) :
  chrysanthemums = 90 :=
by sorry

end florist_chrysanthemums_l1434_143418


namespace inverse_113_mod_114_l1434_143471

theorem inverse_113_mod_114 : ∃ x : ℕ, x ≡ 113 [ZMOD 114] ∧ 113 * x ≡ 1 [ZMOD 114] :=
by sorry

end inverse_113_mod_114_l1434_143471


namespace prime_product_range_l1434_143458

theorem prime_product_range (p q : ℕ) (hp : Prime p) (hq : Prime q) :
  15 < p * q → p * q ≤ 36 → 8 < q → q < 24 → p * q = 33 → p = 3 := by
  sorry

end prime_product_range_l1434_143458


namespace sandy_marbles_count_l1434_143444

/-- The number of red marbles Jessica has -/
def jessica_marbles : ℕ := 3 * 12

/-- The factor by which Sandy has more marbles than Jessica -/
def sandy_factor : ℕ := 4

/-- The number of red marbles Sandy has -/
def sandy_marbles : ℕ := sandy_factor * jessica_marbles

theorem sandy_marbles_count : sandy_marbles = 144 := by
  sorry

end sandy_marbles_count_l1434_143444


namespace absolute_value_of_negative_l1434_143456

theorem absolute_value_of_negative (x : ℝ) : x < 0 → |x| = -x := by
  sorry

end absolute_value_of_negative_l1434_143456


namespace green_ball_probability_l1434_143413

/-- Represents a container with balls of different colors -/
structure Container where
  red : ℕ
  green : ℕ
  blue : ℕ

/-- Calculates the total number of balls in a container -/
def Container.total (c : Container) : ℕ := c.red + c.green + c.blue

/-- Represents the problem setup with three containers -/
structure BallProblem where
  container1 : Container
  container2 : Container
  container3 : Container

/-- The specific problem instance as described -/
def problem : BallProblem :=
  { container1 := { red := 10, green := 2, blue := 3 }
  , container2 := { red := 5, green := 4, blue := 2 }
  , container3 := { red := 3, green := 5, blue := 3 }
  }

/-- Calculates the probability of selecting a green ball given the problem setup -/
def probabilityGreenBall (p : BallProblem) : ℚ :=
  let p1 := (p.container1.green : ℚ) / p.container1.total
  let p2 := (p.container2.green : ℚ) / p.container2.total
  let p3 := (p.container3.green : ℚ) / p.container3.total
  (p1 + p2 + p3) / 3

theorem green_ball_probability :
  probabilityGreenBall problem = 157 / 495 := by sorry

end green_ball_probability_l1434_143413


namespace integer_fraction_characterization_l1434_143421

def solution_set : Set (Nat × Nat) :=
  {(2, 1), (3, 1), (2, 2), (5, 2), (5, 3), (1, 2), (1, 3)}

theorem integer_fraction_characterization (m n : Nat) :
  m > 0 ∧ n > 0 →
  (∃ k : Int, (n^3 + 1 : Int) = k * (m^2 - 1)) ↔ (m, n) ∈ solution_set := by
  sorry

end integer_fraction_characterization_l1434_143421


namespace ticket123123123_is_red_l1434_143423

-- Define the color type
inductive Color
| Red
| Green
| Blue

-- Define a ticket as a 9-digit number and a color
structure Ticket :=
  (number : Fin 9 → Fin 3)
  (color : Color)

-- Function to check if two tickets have no matching digits
def noMatchingDigits (t1 t2 : Ticket) : Prop :=
  ∀ i : Fin 9, t1.number i ≠ t2.number i

-- Define the given conditions
axiom different_colors (t1 t2 : Ticket) :
  noMatchingDigits t1 t2 → t1.color ≠ t2.color

-- Define the specific tickets mentioned in the problem
def ticket122222222 : Ticket :=
  { number := λ i => if i = 0 then 0 else 1,
    color := Color.Red }

def ticket222222222 : Ticket :=
  { number := λ _ => 1,
    color := Color.Green }

def ticket123123123 : Ticket :=
  { number := λ i => i % 3,
    color := Color.Red }  -- We'll prove this color

-- The theorem to prove
theorem ticket123123123_is_red :
  ticket123123123.color = Color.Red :=
sorry

end ticket123123123_is_red_l1434_143423


namespace johns_number_is_eight_l1434_143452

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def reverse_digits (n : ℕ) : ℕ :=
  (n % 10) * 10 + (n / 10)

theorem johns_number_is_eight :
  ∃! x : ℕ, is_two_digit x ∧
    81 ≤ reverse_digits (5 * x + 18) ∧
    reverse_digits (5 * x + 18) ≤ 85 ∧
    x = 8 := by
  sorry

end johns_number_is_eight_l1434_143452


namespace soda_price_before_increase_l1434_143439

/-- The original price of a can of soda -/
def original_price : ℝ := 6

/-- The percentage increase in the price of a can of soda -/
def price_increase_percentage : ℝ := 50

/-- The new price of a can of soda after the price increase -/
def new_price : ℝ := 9

/-- Theorem stating that the original price of a can of soda was 6 pounds -/
theorem soda_price_before_increase :
  original_price * (1 + price_increase_percentage / 100) = new_price :=
by sorry

end soda_price_before_increase_l1434_143439


namespace night_day_crew_ratio_l1434_143426

theorem night_day_crew_ratio (D N : ℕ) (B : ℝ) : 
  (D * B = (3/4) * (D * B + N * ((3/4) * B))) →
  (N : ℝ) / D = 4/3 := by
sorry

end night_day_crew_ratio_l1434_143426


namespace tommy_balloons_l1434_143455

theorem tommy_balloons (x : ℕ) : x + 34 = 60 → x = 26 := by
  sorry

end tommy_balloons_l1434_143455


namespace xiao_li_score_l1434_143460

/-- Calculates the comprehensive score based on content and culture scores -/
def comprehensive_score (content_score culture_score : ℝ) : ℝ :=
  0.4 * content_score + 0.6 * culture_score

/-- Theorem stating that Xiao Li's comprehensive score is 86 points -/
theorem xiao_li_score : comprehensive_score 80 90 = 86 := by
  sorry

end xiao_li_score_l1434_143460


namespace aaron_sweaters_count_l1434_143474

/-- The number of sweaters Aaron made -/
def aaron_sweaters : ℕ := 5

/-- The number of scarves Aaron made -/
def aaron_scarves : ℕ := 10

/-- The number of sweaters Enid made -/
def enid_sweaters : ℕ := 8

/-- The number of balls of wool used for one scarf -/
def wool_per_scarf : ℕ := 3

/-- The number of balls of wool used for one sweater -/
def wool_per_sweater : ℕ := 4

/-- The total number of balls of wool used -/
def total_wool : ℕ := 82

theorem aaron_sweaters_count : 
  aaron_sweaters * wool_per_sweater + 
  aaron_scarves * wool_per_scarf + 
  enid_sweaters * wool_per_sweater = total_wool :=
sorry

end aaron_sweaters_count_l1434_143474


namespace min_games_for_90_percent_win_l1434_143487

theorem min_games_for_90_percent_win (N : ℕ) : 
  (∀ k : ℕ, k < N → (2 + k : ℚ) / (5 + k) ≤ 9/10) ∧
  (2 + N : ℚ) / (5 + N) > 9/10 →
  N = 26 :=
sorry

end min_games_for_90_percent_win_l1434_143487


namespace tv_contest_probabilities_l1434_143470

-- Define the pass rates for each level
def pass_rate_1 : ℝ := 0.6
def pass_rate_2 : ℝ := 0.5
def pass_rate_3 : ℝ := 0.4

-- Define the prize amounts
def first_prize : ℕ := 300
def second_prize : ℕ := 200

-- Define the function to calculate the probability of not winning any prize
def prob_no_prize : ℝ := 1 - pass_rate_1 + pass_rate_1 * (1 - pass_rate_2)

-- Define the function to calculate the probability of total prize money being 700,
-- given both contestants passed the first level
def prob_total_700_given_pass_1 : ℝ :=
  2 * (pass_rate_2 * (1 - pass_rate_3)) * (pass_rate_2 * pass_rate_3)

-- State the theorem
theorem tv_contest_probabilities :
  prob_no_prize = 0.7 ∧
  prob_total_700_given_pass_1 = 0.12 := by
  sorry

end tv_contest_probabilities_l1434_143470


namespace coefficient_x_squared_expansion_l1434_143486

/-- The coefficient of x^2 in the expansion of (1/√x + x)^8 -/
def coefficient_x_squared : ℕ := 70

/-- The binomial coefficient (8 choose 4) -/
def binomial_8_4 : ℕ := 70

theorem coefficient_x_squared_expansion :
  coefficient_x_squared = binomial_8_4 := by sorry

end coefficient_x_squared_expansion_l1434_143486


namespace specific_mixture_problem_l1434_143478

/-- Represents a mixture of three components -/
structure Mixture where
  a : ℝ
  b : ℝ
  c : ℝ
  sum_to_100 : a + b + c = 100

/-- The problem of finding coefficients for mixing three mixtures to obtain a desired mixture -/
def mixture_problem (m₁ m₂ m₃ : Mixture) (desired : Mixture) :=
  ∃ (k₁ k₂ k₃ : ℝ),
    k₁ ≥ 0 ∧ k₂ ≥ 0 ∧ k₃ ≥ 0 ∧
    k₁ + k₂ + k₃ = 1 ∧
    k₁ * m₁.a + k₂ * m₂.a + k₃ * m₃.a = desired.a ∧
    k₁ * m₁.b + k₂ * m₂.b + k₃ * m₃.b = desired.b ∧
    k₁ * m₁.c + k₂ * m₂.c + k₃ * m₃.c = desired.c

/-- The specific mixture problem instance -/
theorem specific_mixture_problem :
  let m₁ : Mixture := ⟨10, 30, 60, by norm_num⟩
  let m₂ : Mixture := ⟨20, 60, 20, by norm_num⟩
  let m₃ : Mixture := ⟨80, 10, 10, by norm_num⟩
  let desired : Mixture := ⟨50, 30, 20, by norm_num⟩
  mixture_problem m₁ m₂ m₃ desired := by
    sorry

end specific_mixture_problem_l1434_143478


namespace percentage_of_democratic_voters_l1434_143420

theorem percentage_of_democratic_voters :
  ∀ (d r : ℝ),
    d + r = 100 →
    0.8 * d + 0.3 * r = 65 →
    d = 70 := by
  sorry

end percentage_of_democratic_voters_l1434_143420


namespace carter_reading_rate_l1434_143480

/-- The number of pages Oliver can read in 1 hour -/
def oliver_pages : ℕ := 40

/-- The number of pages Lucy can read in 1 hour -/
def lucy_pages : ℕ := oliver_pages + 20

/-- The number of pages Carter can read in 1 hour -/
def carter_pages : ℕ := lucy_pages / 2

/-- Proves that Carter can read 30 pages in 1 hour given the conditions -/
theorem carter_reading_rate : carter_pages = 30 := by
  sorry

end carter_reading_rate_l1434_143480
