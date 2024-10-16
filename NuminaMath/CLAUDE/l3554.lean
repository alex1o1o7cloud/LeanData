import Mathlib

namespace NUMINAMATH_CALUDE_largest_number_with_given_hcf_and_lcm_factors_l3554_355467

theorem largest_number_with_given_hcf_and_lcm_factors 
  (a b c : ℕ+) 
  (hcf_eq : Nat.gcd a b = 42 ∧ Nat.gcd (Nat.gcd a b) c = 42)
  (lcm_factors : ∃ (m : ℕ+), Nat.lcm (Nat.lcm a b) c = 42 * 10 * 20 * 25 * 30 * m) :
  max a (max b c) = 1260 := by
sorry

end NUMINAMATH_CALUDE_largest_number_with_given_hcf_and_lcm_factors_l3554_355467


namespace NUMINAMATH_CALUDE_range_of_function_l3554_355496

theorem range_of_function (x : ℝ) (h : x > 0) : x + 1/x ≥ 2 ∧ (x + 1/x = 2 ↔ x = 1) := by
  sorry

end NUMINAMATH_CALUDE_range_of_function_l3554_355496


namespace NUMINAMATH_CALUDE_burger_share_l3554_355476

theorem burger_share (burger_length : ℕ) (inches_per_foot : ℕ) : 
  burger_length = 1 → 
  inches_per_foot = 12 → 
  (burger_length * inches_per_foot) / 2 = 6 :=
by
  sorry

#check burger_share

end NUMINAMATH_CALUDE_burger_share_l3554_355476


namespace NUMINAMATH_CALUDE_sum_of_reciprocal_relations_l3554_355440

theorem sum_of_reciprocal_relations (x y : ℚ) 
  (h1 : x ≠ 0) (h2 : y ≠ 0)
  (h3 : 1 / x + 1 / y = 4) 
  (h4 : 1 / x - 1 / y = -6) : 
  x + y = -4/5 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_reciprocal_relations_l3554_355440


namespace NUMINAMATH_CALUDE_quadratic_one_solution_sum_l3554_355468

theorem quadratic_one_solution_sum (b₁ b₂ : ℝ) : 
  (∀ x, 3 * x^2 + b₁ * x + 6 * x + 4 = 0 → (b₁ + 6)^2 = 48) ∧ 
  (∀ x, 3 * x^2 + b₂ * x + 6 * x + 4 = 0 → (b₂ + 6)^2 = 48) → 
  b₁ + b₂ = -12 := by
sorry

end NUMINAMATH_CALUDE_quadratic_one_solution_sum_l3554_355468


namespace NUMINAMATH_CALUDE_rectangle_area_l3554_355464

theorem rectangle_area (square_area : ℝ) (rectangle_width : ℝ) (rectangle_length : ℝ) : 
  square_area = 49 → 
  rectangle_width ^ 2 = square_area →
  rectangle_length = 3 * rectangle_width →
  rectangle_width * rectangle_length = 147 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_l3554_355464


namespace NUMINAMATH_CALUDE_number_of_ambiguous_dates_l3554_355482

/-- The number of days that cannot be uniquely determined by date notation -/
def ambiguous_dates : ℕ :=
  let total_possible_ambiguous := 12 * 12  -- Days 1-12 for each of the 12 months
  let non_ambiguous := 12  -- Dates where day and month are the same (e.g., 1.1, 2.2, ..., 12.12)
  total_possible_ambiguous - non_ambiguous

/-- Theorem stating that the number of ambiguous dates is 132 -/
theorem number_of_ambiguous_dates : ambiguous_dates = 132 := by
  sorry


end NUMINAMATH_CALUDE_number_of_ambiguous_dates_l3554_355482


namespace NUMINAMATH_CALUDE_equation_value_l3554_355408

theorem equation_value (a b c d : ℝ) 
  (eq1 : a + b - c - d = 5)
  (eq2 : (b - d)^2 = 16) :
  (a - b - c + d = b + 3*d - 4) ∨ (a - b - c + d = b + 3*d + 4) := by
  sorry

end NUMINAMATH_CALUDE_equation_value_l3554_355408


namespace NUMINAMATH_CALUDE_midpoint_distance_range_l3554_355420

/-- Given two parallel lines and a point constrained to lie between them, 
    prove the range of the squared distance from this point to the origin. -/
theorem midpoint_distance_range (x₀ y₀ : ℝ) : 
  (∃ x y u v : ℝ, 
    x - 2*y - 2 = 0 ∧ 
    u - 2*v - 6 = 0 ∧ 
    x₀ = (x + u) / 2 ∧ 
    y₀ = (y + v) / 2 ∧
    (x₀ - 2)^2 + (y₀ + 1)^2 ≤ 5) →
  16/5 ≤ x₀^2 + y₀^2 ∧ x₀^2 + y₀^2 ≤ 16 :=
by sorry

end NUMINAMATH_CALUDE_midpoint_distance_range_l3554_355420


namespace NUMINAMATH_CALUDE_parking_garage_has_four_stories_l3554_355499

/-- Represents a parking garage with the given specifications -/
structure ParkingGarage where
  spots_per_level : ℕ
  open_spots_level1 : ℕ
  open_spots_level2 : ℕ
  open_spots_level3 : ℕ
  open_spots_level4 : ℕ
  full_spots_total : ℕ

/-- Calculates the number of stories in the parking garage -/
def number_of_stories (garage : ParkingGarage) : ℕ :=
  (garage.full_spots_total + garage.open_spots_level1 + garage.open_spots_level2 +
   garage.open_spots_level3 + garage.open_spots_level4) / garage.spots_per_level

/-- Theorem stating that the parking garage has exactly 4 stories -/
theorem parking_garage_has_four_stories (garage : ParkingGarage) :
  garage.spots_per_level = 100 ∧
  garage.open_spots_level1 = 58 ∧
  garage.open_spots_level2 = garage.open_spots_level1 + 2 ∧
  garage.open_spots_level3 = garage.open_spots_level2 + 5 ∧
  garage.open_spots_level4 = 31 ∧
  garage.full_spots_total = 186 →
  number_of_stories garage = 4 := by
  sorry

end NUMINAMATH_CALUDE_parking_garage_has_four_stories_l3554_355499


namespace NUMINAMATH_CALUDE_machine_job_completion_time_l3554_355407

theorem machine_job_completion_time : ∃ (x : ℝ), 
  x > 0 ∧
  (1 / (x + 4) + 1 / (x + 2) + 1 / ((x + 4 + x + 2) / 2) = 1 / x) ∧
  x = 1 := by
  sorry

end NUMINAMATH_CALUDE_machine_job_completion_time_l3554_355407


namespace NUMINAMATH_CALUDE_coin_and_die_probability_l3554_355445

theorem coin_and_die_probability :
  let coin_outcomes : ℕ := 2  -- Fair coin has 2 possible outcomes
  let die_outcomes : ℕ := 8   -- Eight-sided die has 8 possible outcomes
  let total_outcomes : ℕ := coin_outcomes * die_outcomes
  let successful_outcomes : ℕ := 1  -- Only one successful outcome (Tails and 5)
  
  (successful_outcomes : ℚ) / total_outcomes = 1 / 16 :=
by
  sorry


end NUMINAMATH_CALUDE_coin_and_die_probability_l3554_355445


namespace NUMINAMATH_CALUDE_cyclists_meeting_time_l3554_355426

/-- Two cyclists on a circular track problem -/
theorem cyclists_meeting_time 
  (track_circumference : ℝ) 
  (speed1 speed2 : ℝ) 
  (h1 : track_circumference = 600) 
  (h2 : speed1 = 7) 
  (h3 : speed2 = 8) : 
  track_circumference / (speed1 + speed2) = 40 := by
  sorry

#check cyclists_meeting_time

end NUMINAMATH_CALUDE_cyclists_meeting_time_l3554_355426


namespace NUMINAMATH_CALUDE_bank_teller_bills_l3554_355432

theorem bank_teller_bills (total_bills : ℕ) (total_value : ℕ) : 
  total_bills = 54 → total_value = 780 → 
  ∃ (five_dollar_bills twenty_dollar_bills : ℕ), 
    five_dollar_bills + twenty_dollar_bills = total_bills ∧
    5 * five_dollar_bills + 20 * twenty_dollar_bills = total_value ∧
    five_dollar_bills = 20 := by
  sorry

end NUMINAMATH_CALUDE_bank_teller_bills_l3554_355432


namespace NUMINAMATH_CALUDE_workshop_average_age_l3554_355466

theorem workshop_average_age (total_members : ℕ) (overall_avg : ℝ)
  (num_girls num_boys num_adults num_teens : ℕ)
  (avg_girls avg_boys avg_teens : ℝ) :
  total_members = 50 →
  overall_avg = 20 →
  num_girls = 25 →
  num_boys = 15 →
  num_adults = 5 →
  num_teens = 5 →
  avg_girls = 18 →
  avg_boys = 19 →
  avg_teens = 16 →
  (total_members : ℝ) * overall_avg =
    (num_girls : ℝ) * avg_girls + (num_boys : ℝ) * avg_boys +
    (num_adults : ℝ) * ((total_members : ℝ) * overall_avg - (num_girls : ℝ) * avg_girls -
    (num_boys : ℝ) * avg_boys - (num_teens : ℝ) * avg_teens) / num_adults +
    (num_teens : ℝ) * avg_teens →
  ((total_members : ℝ) * overall_avg - (num_girls : ℝ) * avg_girls -
   (num_boys : ℝ) * avg_boys - (num_teens : ℝ) * avg_teens) / num_adults = 37 :=
by sorry

end NUMINAMATH_CALUDE_workshop_average_age_l3554_355466


namespace NUMINAMATH_CALUDE_remaining_sweet_cookies_l3554_355434

def initial_sweet_cookies : ℕ := 22
def eaten_sweet_cookies : ℕ := 15

theorem remaining_sweet_cookies :
  initial_sweet_cookies - eaten_sweet_cookies = 7 :=
by sorry

end NUMINAMATH_CALUDE_remaining_sweet_cookies_l3554_355434


namespace NUMINAMATH_CALUDE_expression_simplification_l3554_355489

theorem expression_simplification : (((3 + 4 + 5 + 6) / 3) + ((3 * 6 + 9) / 4)) = 12.75 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3554_355489


namespace NUMINAMATH_CALUDE_tan_945_degrees_l3554_355400

theorem tan_945_degrees : Real.tan (945 * π / 180) = 1 := by sorry

end NUMINAMATH_CALUDE_tan_945_degrees_l3554_355400


namespace NUMINAMATH_CALUDE_factorial_difference_is_perfect_square_l3554_355413

def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

theorem factorial_difference_is_perfect_square (q : ℕ) (r : ℕ) 
  (h : factorial (q + 2) - factorial (q + 1) = factorial q * r) :
  r = (q + 1) ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_factorial_difference_is_perfect_square_l3554_355413


namespace NUMINAMATH_CALUDE_stratified_sample_size_l3554_355463

/-- Given a stratified sample of three products with a quantity ratio of 2:3:5,
    prove that if 16 units of the first product are in the sample,
    then the total sample size is 80. -/
theorem stratified_sample_size
  (total_ratio : ℕ)
  (ratio_A : ℕ)
  (ratio_B : ℕ)
  (ratio_C : ℕ)
  (sample_A : ℕ)
  (h1 : total_ratio = ratio_A + ratio_B + ratio_C)
  (h2 : ratio_A = 2)
  (h3 : ratio_B = 3)
  (h4 : ratio_C = 5)
  (h5 : sample_A = 16) :
  (sample_A * total_ratio) / ratio_A = 80 := by
sorry

end NUMINAMATH_CALUDE_stratified_sample_size_l3554_355463


namespace NUMINAMATH_CALUDE_triangle_side_length_l3554_355405

theorem triangle_side_length (a b c : ℝ) (C : ℝ) :
  a = 5 → b = 3 → C = 2 * π / 3 → c = 7 := by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l3554_355405


namespace NUMINAMATH_CALUDE_x_plus_q_in_terms_of_q_l3554_355423

theorem x_plus_q_in_terms_of_q (x q : ℝ) (h1 : |x - 5| = q) (h2 : x > 5) : x + q = 2*q + 5 := by
  sorry

end NUMINAMATH_CALUDE_x_plus_q_in_terms_of_q_l3554_355423


namespace NUMINAMATH_CALUDE_total_triangles_is_16_l3554_355418

/-- Represents a square with diagonals and an inner square formed by midpoints -/
structure SquareWithDiagonalsAndInnerSquare :=
  (s : ℝ) -- Side length of the larger square
  (has_diagonals : Bool) -- The larger square has diagonals
  (has_inner_square : Bool) -- There's an inner square formed by midpoints

/-- Counts the total number of triangles in the figure -/
def count_triangles (square : SquareWithDiagonalsAndInnerSquare) : ℕ :=
  sorry -- The actual counting logic would go here

/-- Theorem stating that the total number of triangles is 16 -/
theorem total_triangles_is_16 (square : SquareWithDiagonalsAndInnerSquare) :
  square.has_diagonals = true → square.has_inner_square = true → count_triangles square = 16 :=
by sorry

end NUMINAMATH_CALUDE_total_triangles_is_16_l3554_355418


namespace NUMINAMATH_CALUDE_candy_necklace_problem_l3554_355431

/-- Candy necklace problem -/
theorem candy_necklace_problem 
  (pieces_per_necklace : ℕ) 
  (pieces_per_block : ℕ) 
  (blocks_used : ℕ) 
  (h1 : pieces_per_necklace = 10)
  (h2 : pieces_per_block = 30)
  (h3 : blocks_used = 3)
  : (blocks_used * pieces_per_block) / pieces_per_necklace - 1 = 8 := by
  sorry

#check candy_necklace_problem

end NUMINAMATH_CALUDE_candy_necklace_problem_l3554_355431


namespace NUMINAMATH_CALUDE_fraction_sum_inequality_l3554_355428

theorem fraction_sum_inequality {x y : ℝ} (hx : x > 0) (hy : y > 0) :
  x / y + y / x ≥ 2 ∧ (x / y + y / x = 2 ↔ x = y) := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_inequality_l3554_355428


namespace NUMINAMATH_CALUDE_symmetry_about_one_l3554_355442

/-- A function f is symmetric about x = 1 if f(x-1) = f(1-x) for all x -/
def SymmetricAboutOne (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (x - 1) = f (1 - x)

/-- The theorem stating that any real-valued function defined on ℝ 
    has graphs of f(x-1) and f(1-x) symmetric about x = 1 -/
theorem symmetry_about_one (f : ℝ → ℝ) : SymmetricAboutOne f := by
  sorry

#check symmetry_about_one

end NUMINAMATH_CALUDE_symmetry_about_one_l3554_355442


namespace NUMINAMATH_CALUDE_metallic_sheet_length_l3554_355495

/-- The length of a rectangular metallic sheet, given its width and the dimensions of an open box formed from it. -/
theorem metallic_sheet_length (w h v : ℝ) (hw : w = 36) (hh : h = 8) (hv : v = 5440) : ∃ l : ℝ,
  l = 50 ∧ v = (l - 2 * h) * (w - 2 * h) * h :=
by sorry

end NUMINAMATH_CALUDE_metallic_sheet_length_l3554_355495


namespace NUMINAMATH_CALUDE_square_side_length_range_l3554_355474

theorem square_side_length_range (a : ℝ) (h : a > 0) :
  a^2 = 37 → 6 < a ∧ a < 7 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_range_l3554_355474


namespace NUMINAMATH_CALUDE_softball_team_size_l3554_355460

theorem softball_team_size :
  ∀ (men women : ℕ),
  women = men + 4 →
  (men : ℚ) / (women : ℚ) = 7/11 →
  men + women = 18 :=
by
  sorry

end NUMINAMATH_CALUDE_softball_team_size_l3554_355460


namespace NUMINAMATH_CALUDE_dhoni_toys_l3554_355429

theorem dhoni_toys (x : ℕ) (avg_cost : ℚ) (new_toy_cost : ℚ) (new_avg_cost : ℚ) : 
  avg_cost = 10 →
  new_toy_cost = 16 →
  new_avg_cost = 11 →
  (x * avg_cost + new_toy_cost) / (x + 1) = new_avg_cost →
  x = 5 := by
sorry

end NUMINAMATH_CALUDE_dhoni_toys_l3554_355429


namespace NUMINAMATH_CALUDE_least_sum_equation_l3554_355492

theorem least_sum_equation (x y z : ℕ+) 
  (h1 : 6 * z.val = 2 * x.val) 
  (h2 : x.val + y.val + z.val = 26) : 
  6 * z.val = 36 := by
sorry

end NUMINAMATH_CALUDE_least_sum_equation_l3554_355492


namespace NUMINAMATH_CALUDE_abs_ln_equal_implies_product_one_l3554_355449

theorem abs_ln_equal_implies_product_one (a b : ℝ) (h1 : a ≠ b) (h2 : |Real.log a| = |Real.log b|) : a * b = 1 := by
  sorry

end NUMINAMATH_CALUDE_abs_ln_equal_implies_product_one_l3554_355449


namespace NUMINAMATH_CALUDE_tangent_line_at_origin_l3554_355497

/-- Given a real number a and a function f(x) = x^3 + ax^2 + (a - 2)x where its derivative f'(x) is an even function,
    the equation of the tangent line to the curve y = f(x) at the origin is y = -2x. -/
theorem tangent_line_at_origin (a : ℝ) (f : ℝ → ℝ) (f' : ℝ → ℝ) :
  (∀ x, f x = x^3 + a*x^2 + (a - 2)*x) →
  (∀ x, (deriv f) x = f' x) →
  (∀ x, f' x = f' (-x)) →
  (∀ x, x * (-2) = f x - f 0) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_at_origin_l3554_355497


namespace NUMINAMATH_CALUDE_fraction_multiplication_equality_l3554_355471

theorem fraction_multiplication_equality : 
  (3/4 : ℚ) * (1/2 : ℚ) * (2/5 : ℚ) * 5040.000000000001 = 756.0000000000001 := by
  sorry

end NUMINAMATH_CALUDE_fraction_multiplication_equality_l3554_355471


namespace NUMINAMATH_CALUDE_sin_2x_value_l3554_355490

theorem sin_2x_value (x : ℝ) (h : Real.tan (x + π/4) = 2) : Real.sin (2*x) = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_sin_2x_value_l3554_355490


namespace NUMINAMATH_CALUDE_product_of_integers_l3554_355422

theorem product_of_integers (x y : ℕ+) 
  (sum_eq : x + y = 22)
  (diff_squares_eq : x^2 - y^2 = 44) :
  x * y = 120 := by
  sorry

end NUMINAMATH_CALUDE_product_of_integers_l3554_355422


namespace NUMINAMATH_CALUDE_average_speed_calculation_l3554_355421

theorem average_speed_calculation (motorcycle_speed motorcycle_time jog_speed jog_time : ℝ) :
  motorcycle_speed = 20 ∧ 
  motorcycle_time = 40 / 60 ∧ 
  jog_speed = 5 ∧ 
  jog_time = 60 / 60 → 
  (motorcycle_speed * motorcycle_time + jog_speed * jog_time) / (motorcycle_time + jog_time) = 11 := by
sorry

end NUMINAMATH_CALUDE_average_speed_calculation_l3554_355421


namespace NUMINAMATH_CALUDE_hannahs_measuring_spoons_price_l3554_355478

/-- The price of each measuring spoon set given the conditions of Hannah's sales and purchases -/
theorem hannahs_measuring_spoons_price 
  (cookie_count : ℕ) 
  (cookie_price : ℚ) 
  (cupcake_count : ℕ) 
  (cupcake_price : ℚ) 
  (spoon_set_count : ℕ) 
  (money_left : ℚ)
  (h1 : cookie_count = 40)
  (h2 : cookie_price = 4/5)
  (h3 : cupcake_count = 30)
  (h4 : cupcake_price = 2)
  (h5 : spoon_set_count = 2)
  (h6 : money_left = 79) :
  let total_earnings := cookie_count * cookie_price + cupcake_count * cupcake_price
  let spoon_sets_cost := total_earnings - money_left
  let price_per_spoon_set := spoon_sets_cost / spoon_set_count
  price_per_spoon_set = 13/2 := by sorry

end NUMINAMATH_CALUDE_hannahs_measuring_spoons_price_l3554_355478


namespace NUMINAMATH_CALUDE_x_minus_q_upper_bound_l3554_355414

theorem x_minus_q_upper_bound (x q : ℝ) (h1 : |x - 3| > q) (h2 : x < 3) :
  x - q < 3 - 2*q := by
sorry

end NUMINAMATH_CALUDE_x_minus_q_upper_bound_l3554_355414


namespace NUMINAMATH_CALUDE_isosceles_triangle_base_l3554_355417

/-- Proves that the base of an isosceles triangle is 10, given specific conditions about its perimeter and relationship to an equilateral triangle. -/
theorem isosceles_triangle_base : 
  ∀ (s b : ℝ),
  -- Equilateral triangle perimeter condition
  3 * s = 45 →
  -- Isosceles triangle perimeter condition
  2 * s + b = 40 →
  -- Base of isosceles triangle is 10
  b = 10 := by
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_base_l3554_355417


namespace NUMINAMATH_CALUDE_angle_bisector_product_not_unique_l3554_355403

/-- A triangle represented by its three side lengths -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  positive_a : 0 < a
  positive_b : 0 < b
  positive_c : 0 < c
  triangle_inequality : a < b + c ∧ b < a + c ∧ c < a + b

/-- The product of the lengths of the three angle bisectors of a triangle -/
def angle_bisector_product (t : Triangle) : ℝ := sorry

/-- Statement: The product of the three angle bisectors does not uniquely determine a triangle -/
theorem angle_bisector_product_not_unique :
  ∃ (t1 t2 : Triangle), t1 ≠ t2 ∧ angle_bisector_product t1 = angle_bisector_product t2 :=
sorry

end NUMINAMATH_CALUDE_angle_bisector_product_not_unique_l3554_355403


namespace NUMINAMATH_CALUDE_M_factors_l3554_355433

/-- The number of positive integer factors of M, where
    M = 57^6 + 6 * 57^5 + 15 * 57^4 + 20 * 57^3 + 15 * 57^2 + 6 * 57 + 1 -/
def M : ℕ := 57^6 + 6 * 57^5 + 15 * 57^4 + 20 * 57^3 + 15 * 57^2 + 6 * 57 + 1

/-- The number of positive integer factors of a natural number n -/
def numFactors (n : ℕ) : ℕ := (Finset.filter (· ∣ n) (Finset.range (n + 1))).card

theorem M_factors : numFactors M = 49 := by
  sorry

end NUMINAMATH_CALUDE_M_factors_l3554_355433


namespace NUMINAMATH_CALUDE_x_value_proof_l3554_355448

theorem x_value_proof (x : ℝ) : x = 2 * (1/x) * (-x) - 5 → x = -7 := by
  sorry

end NUMINAMATH_CALUDE_x_value_proof_l3554_355448


namespace NUMINAMATH_CALUDE_oliver_bill_denomination_l3554_355455

/-- The denomination of Oliver's unknown bills -/
def x : ℕ := sorry

/-- Oliver's total money -/
def oliver_money : ℕ := 10 * x + 3 * 5

/-- William's total money -/
def william_money : ℕ := 15 * 10 + 4 * 5

theorem oliver_bill_denomination :
  (oliver_money = william_money + 45) → x = 20 := by sorry

end NUMINAMATH_CALUDE_oliver_bill_denomination_l3554_355455


namespace NUMINAMATH_CALUDE_complex_fraction_equals_i_l3554_355456

theorem complex_fraction_equals_i : (1 + Complex.I) / (1 - Complex.I) = Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equals_i_l3554_355456


namespace NUMINAMATH_CALUDE_ellipse_hyperbola_same_foci_l3554_355458

/-- The value of m for which an ellipse and hyperbola with given equations have the same foci -/
theorem ellipse_hyperbola_same_foci (m : ℝ) : m > 0 →
  (∀ x y : ℝ, x^2 / 4 + y^2 / m^2 = 1) →
  (∀ x y : ℝ, x^2 / m^2 - y^2 / 2 = 1) →
  (∀ c : ℝ, c^2 = 4 - m^2 ↔ c^2 = m^2 + 2) →
  m = 1 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_hyperbola_same_foci_l3554_355458


namespace NUMINAMATH_CALUDE_two_digit_number_condition_l3554_355439

/-- A two-digit number satisfies the given condition if and only if its ones digit is 9 -/
theorem two_digit_number_condition (a b : ℕ) (h1 : a > 0) (h2 : a < 10) (h3 : b < 10) :
  (10 * a + b) - (a * b) = a + b ↔ b = 9 :=
by sorry

end NUMINAMATH_CALUDE_two_digit_number_condition_l3554_355439


namespace NUMINAMATH_CALUDE_eighth_roll_last_probability_l3554_355404

/-- A standard six-sided die -/
def StandardDie : Type := Fin 6

/-- The probability of rolling a different number than the previous roll -/
def probDifferent : ℚ := 5/6

/-- The probability of rolling the same number as the previous roll -/
def probSame : ℚ := 1/6

/-- The number of rolls we're interested in -/
def numRolls : ℕ := 8

/-- The probability that the 8th roll is the last roll -/
def probEighthRollLast : ℚ := probDifferent^(numRolls - 2) * probSame

theorem eighth_roll_last_probability :
  probEighthRollLast = 15625/279936 := by sorry

end NUMINAMATH_CALUDE_eighth_roll_last_probability_l3554_355404


namespace NUMINAMATH_CALUDE_limit_equals_negative_six_l3554_355451

-- Define the function f
def f (x : ℝ) : ℝ := x^2 + x

-- State the theorem
theorem limit_equals_negative_six :
  ∀ ε > 0, ∃ δ > 0, ∀ Δx ≠ 0,
    |Δx| < δ → |((f (1 - 2*Δx) - f 1) / Δx) + 6| < ε :=
by sorry

end NUMINAMATH_CALUDE_limit_equals_negative_six_l3554_355451


namespace NUMINAMATH_CALUDE_concession_stand_theorem_l3554_355447

def concession_stand_revenue (hot_dog_price soda_price : ℚ) 
                             (total_items hot_dogs_sold : ℕ) : ℚ :=
  let sodas_sold := total_items - hot_dogs_sold
  hot_dog_price * hot_dogs_sold + soda_price * sodas_sold

theorem concession_stand_theorem :
  concession_stand_revenue (3/2) (1/2) 87 35 = 157/2 := by
  sorry

end NUMINAMATH_CALUDE_concession_stand_theorem_l3554_355447


namespace NUMINAMATH_CALUDE_next_multiple_age_sum_digits_l3554_355401

/-- Represents a person with an age -/
structure Person where
  age : ℕ

/-- Represents the family with Joey, Chloe, and Zoe -/
structure Family where
  joey : Person
  chloe : Person
  zoe : Person

/-- Returns true if n is a multiple of m -/
def isMultiple (n m : ℕ) : Prop := ∃ k : ℕ, n = m * k

/-- Returns the sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sumOfDigits (n / 10)

/-- The main theorem -/
theorem next_multiple_age_sum_digits (f : Family) : 
  f.zoe.age = 1 →
  f.joey.age = f.chloe.age + 1 →
  (∃ n : ℕ, n ≥ 1 ∧ n ≤ 9 ∧ isMultiple (f.chloe.age + n - 1) n) →
  (∀ m : ℕ, m < f.chloe.age - 1 → ¬isMultiple (f.chloe.age + m - 1) m) →
  (∃ k : ℕ, isMultiple (f.joey.age + k) (f.zoe.age + k) ∧ 
    (∀ j : ℕ, j < k → ¬isMultiple (f.joey.age + j) (f.zoe.age + j)) ∧
    sumOfDigits (f.joey.age + k) = 12) :=
sorry

end NUMINAMATH_CALUDE_next_multiple_age_sum_digits_l3554_355401


namespace NUMINAMATH_CALUDE_unique_quadratic_solution_l3554_355469

/-- Given a quadratic equation ax^2 + 15x + c = 0 with exactly one solution,
    where a + c = 36 and a < c, prove that a = (36 - √1071) / 2 and c = (36 + √1071) / 2 -/
theorem unique_quadratic_solution (a c : ℝ) : 
  (∃! x, a * x^2 + 15 * x + c = 0) → 
  (a + c = 36) → 
  (a < c) → 
  (a = (36 - Real.sqrt 1071) / 2 ∧ c = (36 + Real.sqrt 1071) / 2) := by
sorry

end NUMINAMATH_CALUDE_unique_quadratic_solution_l3554_355469


namespace NUMINAMATH_CALUDE_complex_number_equality_l3554_355481

theorem complex_number_equality : ∀ (i : ℂ), i^2 = -1 →
  (2 * i) / (2 + i) = 2/5 + 4/5 * i := by
  sorry

end NUMINAMATH_CALUDE_complex_number_equality_l3554_355481


namespace NUMINAMATH_CALUDE_angle_D_is_120_l3554_355480

-- Define a quadrilateral
structure Quadrilateral :=
  (A B C D : ℝ)
  (sum_360 : A + B + C + D = 360)
  (all_positive : A > 0 ∧ B > 0 ∧ C > 0 ∧ D > 0)

-- Define the ratio condition
def ratio_condition (q : Quadrilateral) : Prop :=
  ∃ (k : ℝ), k > 0 ∧ q.A = k ∧ q.B = 2*k ∧ q.C = k ∧ q.D = 2*k

-- Theorem statement
theorem angle_D_is_120 (q : Quadrilateral) (h : ratio_condition q) : q.D = 120 := by
  sorry

end NUMINAMATH_CALUDE_angle_D_is_120_l3554_355480


namespace NUMINAMATH_CALUDE_furniture_production_l3554_355446

theorem furniture_production (x : ℝ) (h : x > 0) :
  (540 / x - 540 / (x + 2) = 3) ↔ 
  (∃ (original_days actual_days : ℝ),
    original_days > 0 ∧
    actual_days > 0 ∧
    original_days - actual_days = 3 ∧
    original_days * x = 540 ∧
    actual_days * (x + 2) = 540) :=
  sorry

end NUMINAMATH_CALUDE_furniture_production_l3554_355446


namespace NUMINAMATH_CALUDE_sales_second_month_l3554_355484

def sales_problem (X : ℕ) : Prop :=
  let sales : List ℕ := [2435, X, 2855, 3230, 2560, 1000]
  (sales.sum / sales.length = 2500) ∧ 
  (sales.length = 6)

theorem sales_second_month : 
  ∃ (X : ℕ), sales_problem X ∧ X = 2920 := by
sorry

end NUMINAMATH_CALUDE_sales_second_month_l3554_355484


namespace NUMINAMATH_CALUDE_hyperbola_focal_length_l3554_355473

theorem hyperbola_focal_length (x y : ℝ) :
  x^2 / 7 - y^2 / 3 = 1 → 2 * Real.sqrt 10 = 2 * Real.sqrt (7 + 3) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_focal_length_l3554_355473


namespace NUMINAMATH_CALUDE_fraction_calculation_l3554_355438

theorem fraction_calculation : 
  (1 / 4 - 1 / 5) / (1 / 3 - 1 / 6) = 3 / 10 := by sorry

end NUMINAMATH_CALUDE_fraction_calculation_l3554_355438


namespace NUMINAMATH_CALUDE_farm_tax_problem_l3554_355462

/-- Represents the farm tax problem -/
theorem farm_tax_problem (total_tax : ℝ) (william_tax : ℝ) (taxable_percentage : ℝ) 
  (h1 : total_tax = 5000)
  (h2 : william_tax = 480)
  (h3 : taxable_percentage = 0.60) :
  william_tax / total_tax * 100 = 5.76 := by
  sorry

end NUMINAMATH_CALUDE_farm_tax_problem_l3554_355462


namespace NUMINAMATH_CALUDE_intersection_of_planes_intersects_skew_lines_l3554_355472

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the intersection operation for planes
variable (intersect : Plane → Plane → Line)

-- Define the "lies in" relation between a line and a plane
variable (liesIn : Line → Plane → Prop)

-- Define the "skew" relation between two lines
variable (skew : Line → Line → Prop)

-- Define the "intersects" relation between two lines
variable (intersects : Line → Line → Prop)

-- Theorem statement
theorem intersection_of_planes_intersects_skew_lines 
  (a b ℓ : Line) (α β : Plane) 
  (h1 : skew a b)
  (h2 : liesIn a α)
  (h3 : liesIn b β)
  (h4 : intersect α β = ℓ) :
  intersects ℓ a ∨ intersects ℓ b :=
sorry

end NUMINAMATH_CALUDE_intersection_of_planes_intersects_skew_lines_l3554_355472


namespace NUMINAMATH_CALUDE_inequality_system_solution_l3554_355435

theorem inequality_system_solution (x : ℝ) (hx : x ≠ 0) :
  (abs (2 * x - 3) ≤ 3 ∧ 1 / x < 1) ↔ (1 < x ∧ x ≤ 3) :=
by sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l3554_355435


namespace NUMINAMATH_CALUDE_tan_alpha_value_l3554_355430

theorem tan_alpha_value (α : Real)
  (h1 : α ∈ Set.Ioo (π / 2) π)
  (h2 : Real.sqrt ((1 + Real.sin α) / (1 - Real.sin α)) = (1 - 2 * Real.cos α) / (2 * Real.sin (α / 2)^2 - 1)) :
  Real.tan α = -2 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_value_l3554_355430


namespace NUMINAMATH_CALUDE_prime_square_mod_30_l3554_355479

theorem prime_square_mod_30 (p : ℕ) (hp : Nat.Prime p) (hp_gt_5 : p > 5) :
  p^2 % 30 = 1 ∨ p^2 % 30 = 19 := by
  sorry

end NUMINAMATH_CALUDE_prime_square_mod_30_l3554_355479


namespace NUMINAMATH_CALUDE_birthday_money_l3554_355441

def money_spent : ℕ := 3
def money_left : ℕ := 2

theorem birthday_money : 
  ∃ (total : ℕ), total = money_spent + money_left :=
sorry

end NUMINAMATH_CALUDE_birthday_money_l3554_355441


namespace NUMINAMATH_CALUDE_parabola_symmetry_range_l3554_355450

theorem parabola_symmetry_range (a : ℝ) : 
  a > 0 → 
  (∀ x y : ℝ, y = a * x^2 - 1 → 
    ∃ x1 y1 x2 y2 : ℝ, 
      y1 = a * x1^2 - 1 ∧ 
      y2 = a * x2^2 - 1 ∧ 
      x1 + y1 = -(x2 + y2) ∧ 
      (x1 ≠ x2 ∨ y1 ≠ y2)) → 
  a > 3/4 :=
sorry

end NUMINAMATH_CALUDE_parabola_symmetry_range_l3554_355450


namespace NUMINAMATH_CALUDE_unique_factorial_difference_divisibility_l3554_355402

theorem unique_factorial_difference_divisibility :
  ∃! (x : ℕ), x > 0 ∧ (Nat.factorial x - Nat.factorial (x - 4)) / 29 = 1 :=
by
  -- The unique value is 8
  use 8
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_factorial_difference_divisibility_l3554_355402


namespace NUMINAMATH_CALUDE_student_congress_sample_size_l3554_355486

/-- Represents a school with classes and students -/
structure School where
  num_classes : Nat
  students_per_class : Nat
  selected_students : Nat

/-- Defines the sample size for a given school -/
def sample_size (s : School) : Nat :=
  s.selected_students

/-- Theorem: The sample size for a school with 40 classes of 50 students each,
    and 150 selected students, is 150 -/
theorem student_congress_sample_size :
  let s : School := { num_classes := 40, students_per_class := 50, selected_students := 150 }
  sample_size s = 150 := by
  sorry


end NUMINAMATH_CALUDE_student_congress_sample_size_l3554_355486


namespace NUMINAMATH_CALUDE_complement_M_l3554_355419

def U : Set ℝ := Set.univ

def M : Set ℝ := {x : ℝ | x^2 - 4 ≤ 0}

theorem complement_M : Set.compl M = {x : ℝ | x > 2 ∨ x < -2} := by sorry

end NUMINAMATH_CALUDE_complement_M_l3554_355419


namespace NUMINAMATH_CALUDE_ab_value_l3554_355409

theorem ab_value (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h1 : a^2 + b^2 = 1) (h2 : a^4 + b^4 = 5/8) : a * b = Real.sqrt 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ab_value_l3554_355409


namespace NUMINAMATH_CALUDE_best_fitting_model_l3554_355452

structure Model where
  id : Nat
  r_squared : Real

def models : List Model := [
  { id := 1, r_squared := 0.98 },
  { id := 2, r_squared := 0.80 },
  { id := 3, r_squared := 0.54 },
  { id := 4, r_squared := 0.35 }
]

theorem best_fitting_model :
  ∃ m ∈ models, ∀ m' ∈ models, m.r_squared ≥ m'.r_squared ∧ m.id = 1 :=
by sorry

end NUMINAMATH_CALUDE_best_fitting_model_l3554_355452


namespace NUMINAMATH_CALUDE_expression_value_l3554_355461

theorem expression_value (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + y = 3 * Real.sqrt (x * y)) :
  |(x - y) / (x + y) + (x^2 - y^2) / (x^2 + y^2) + (x^3 - y^3) / (x^3 + y^3)| = 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l3554_355461


namespace NUMINAMATH_CALUDE_y_value_l3554_355454

-- Define the property for y
def satisfies_condition (y : ℝ) : Prop :=
  y = (1 / y) * (-y) - 3

-- Theorem statement
theorem y_value : ∃ y : ℝ, satisfies_condition y ∧ y = -4 := by
  sorry

end NUMINAMATH_CALUDE_y_value_l3554_355454


namespace NUMINAMATH_CALUDE_rationalize_denominator_l3554_355475

theorem rationalize_denominator :
  ∃ (x : ℝ), x = (Real.sqrt 6 - 1) ∧
  x = (Real.sqrt 8 + Real.sqrt 3) / (Real.sqrt 2 + Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l3554_355475


namespace NUMINAMATH_CALUDE_double_fraction_value_l3554_355444

theorem double_fraction_value (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  (2*x * 2*y) / (2*x + 2*y) = 2 * (x*y / (x + y)) :=
by sorry

end NUMINAMATH_CALUDE_double_fraction_value_l3554_355444


namespace NUMINAMATH_CALUDE_team_a_win_probability_l3554_355465

def homeAwaySchedule : List Bool := [true, true, false, false, true, false, true]

def probWinHome : ℝ := 0.6
def probWinAway : ℝ := 0.5

def probabilityWin41 : ℝ := 
  let p1 := (1 - probWinHome) * probWinHome * probWinAway * probWinAway * probWinHome
  let p2 := probWinHome * (1 - probWinHome) * probWinAway * probWinAway * probWinHome
  let p3 := probWinHome * probWinHome * (1 - probWinAway) * probWinAway * probWinHome
  let p4 := probWinHome * probWinHome * probWinAway * (1 - probWinAway) * probWinHome
  p1 + p2 + p3 + p4

theorem team_a_win_probability : probabilityWin41 = 0.18 := by
  sorry

end NUMINAMATH_CALUDE_team_a_win_probability_l3554_355465


namespace NUMINAMATH_CALUDE_N_div_15_is_square_N_div_10_is_cube_N_div_6_is_fifth_power_N_is_smallest_num_divisors_N_div_30_is_8400_l3554_355485

/-- The smallest positive integer N satisfying the given conditions -/
def N : ℕ := 2^16 * 3^21 * 5^25

/-- N/15 is a perfect square -/
theorem N_div_15_is_square : ∃ k : ℕ, N / 15 = k^2 := by sorry

/-- N/10 is a perfect cube -/
theorem N_div_10_is_cube : ∃ k : ℕ, N / 10 = k^3 := by sorry

/-- N/6 is a perfect fifth power -/
theorem N_div_6_is_fifth_power : ∃ k : ℕ, N / 6 = k^5 := by sorry

/-- N is the smallest positive integer satisfying the conditions -/
theorem N_is_smallest : ∀ m : ℕ, m < N → 
  (¬∃ k : ℕ, m / 15 = k^2) ∨ 
  (¬∃ k : ℕ, m / 10 = k^3) ∨ 
  (¬∃ k : ℕ, m / 6 = k^5) := by sorry

/-- The number of positive divisors of N/30 -/
def num_divisors_N_div_30 : ℕ := (15 + 1) * (20 + 1) * (24 + 1)

/-- Theorem: The number of positive divisors of N/30 is 8400 -/
theorem num_divisors_N_div_30_is_8400 : num_divisors_N_div_30 = 8400 := by sorry

end NUMINAMATH_CALUDE_N_div_15_is_square_N_div_10_is_cube_N_div_6_is_fifth_power_N_is_smallest_num_divisors_N_div_30_is_8400_l3554_355485


namespace NUMINAMATH_CALUDE_product_remainder_l3554_355437

theorem product_remainder (a b : ℕ) :
  a % 3 = 1 → b % 3 = 2 → (a * b) % 3 = 2 := by
  sorry

end NUMINAMATH_CALUDE_product_remainder_l3554_355437


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l3554_355416

theorem quadratic_inequality_range (m : ℝ) :
  (∀ x : ℝ, x^2 + m*x + m > 0) → m ∈ Set.Ioo 0 4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l3554_355416


namespace NUMINAMATH_CALUDE_square_less_than_triple_l3554_355477

theorem square_less_than_triple (x : ℤ) : x^2 < 3*x ↔ x = 1 ∨ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_square_less_than_triple_l3554_355477


namespace NUMINAMATH_CALUDE_triangle_base_difference_l3554_355410

theorem triangle_base_difference (b h : ℝ) (hb : b > 0) (hh : h > 0) : 
  let b_A := (0.99 * b * h) / (0.9 * h)
  let h_A := 0.9 * h
  b_A = 1.1 * b := by sorry

end NUMINAMATH_CALUDE_triangle_base_difference_l3554_355410


namespace NUMINAMATH_CALUDE_train_length_l3554_355457

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 60 → time = 15 → ∃ length : ℝ, 
  (abs (length - 250.05) < 0.01) ∧ (length = speed * 1000 / 3600 * time) :=
sorry

end NUMINAMATH_CALUDE_train_length_l3554_355457


namespace NUMINAMATH_CALUDE_divisibility_property_l3554_355494

theorem divisibility_property (a b : ℕ+) : ∃ n : ℕ+, (a : ℕ) ∣ (b : ℕ)^(n : ℕ) - (n : ℕ) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_property_l3554_355494


namespace NUMINAMATH_CALUDE_remaining_balance_is_correct_l3554_355411

-- Define the problem parameters
def initial_balance : ℚ := 100
def daily_spending : ℚ := 8
def exchange_fee_rate : ℚ := 0.03
def days_in_week : ℕ := 7
def flat_fee : ℚ := 2
def bill_denomination : ℚ := 5

-- Define the function to calculate the remaining balance
def calculate_remaining_balance : ℚ := 
  let total_daily_spend := daily_spending * (1 + exchange_fee_rate)
  let weekly_spend := total_daily_spend * days_in_week
  let balance_after_spending := initial_balance - weekly_spend
  let balance_after_fee := balance_after_spending - flat_fee
  let bills_taken := (balance_after_fee / bill_denomination).floor * bill_denomination
  balance_after_fee - bills_taken

-- Theorem statement
theorem remaining_balance_is_correct : 
  calculate_remaining_balance = 0.32 := by sorry

end NUMINAMATH_CALUDE_remaining_balance_is_correct_l3554_355411


namespace NUMINAMATH_CALUDE_right_angled_triangle_l3554_355415

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ)  -- Angles
  (a b c : ℝ)  -- Sides

-- State the theorem
theorem right_angled_triangle (t : Triangle) 
  (h : (Real.cos (t.A / 2))^2 = (t.b + t.c) / (2 * t.c)) : 
  t.C = π / 2 := by
  sorry

end NUMINAMATH_CALUDE_right_angled_triangle_l3554_355415


namespace NUMINAMATH_CALUDE_special_sequence_theorem_l3554_355483

/-- A sequence of 2017 integers satisfying specific conditions -/
def SpecialSequence : Type := Fin 2017 → Int

/-- The sum of squares of any 7 numbers in the sequence is 7 -/
def SumSquaresSevenIsSeven (seq : SpecialSequence) : Prop :=
  ∀ (s : Finset (Fin 2017)), s.card = 7 → (s.sum (λ i => (seq i)^2) = 7)

/-- The sum of any 11 numbers in the sequence is positive -/
def SumElevenIsPositive (seq : SpecialSequence) : Prop :=
  ∀ (s : Finset (Fin 2017)), s.card = 11 → (s.sum (λ i => seq i) > 0)

/-- The sum of all 2017 numbers is divisible by 9 -/
def SumAllDivisibleByNine (seq : SpecialSequence) : Prop :=
  (Finset.univ.sum seq) % 9 = 0

/-- The sequence consists of five -1's and 2012 1's -/
def IsFiveNegativeOnesRestOnes (seq : SpecialSequence) : Prop :=
  (Finset.filter (λ i => seq i = -1) Finset.univ).card = 5 ∧
  (Finset.filter (λ i => seq i = 1) Finset.univ).card = 2012

theorem special_sequence_theorem (seq : SpecialSequence) :
  SumSquaresSevenIsSeven seq →
  SumElevenIsPositive seq →
  SumAllDivisibleByNine seq →
  IsFiveNegativeOnesRestOnes seq :=
by sorry

end NUMINAMATH_CALUDE_special_sequence_theorem_l3554_355483


namespace NUMINAMATH_CALUDE_symmetry_implies_phi_value_l3554_355459

/-- Given a function f(x) = sin(x) + √3 * cos(x), prove that if y = f(x + φ) is symmetric about x = 0, then φ = π/6 -/
theorem symmetry_implies_phi_value (f : ℝ → ℝ) (φ : ℝ) :
  (∀ x, f x = Real.sin x + Real.sqrt 3 * Real.cos x) →
  (∀ x, f (x + φ) = f (-x + φ)) →
  φ = Real.pi / 6 := by
  sorry

end NUMINAMATH_CALUDE_symmetry_implies_phi_value_l3554_355459


namespace NUMINAMATH_CALUDE_function_inequality_l3554_355427

theorem function_inequality (f : ℝ → ℝ) 
  (h_diff : Differentiable ℝ f)
  (h_sym : ∀ x, f x = f (2 - x))
  (h_ineq : ∀ x, x ≠ 1 → (x - 1) * deriv f x < 0)
  (a : ℝ) (b : ℝ) (c : ℝ)
  (h_a : a = f 0.5)
  (h_b : b = f (4/3))
  (h_c : c = f 3) :
  b > a ∧ a > c := by sorry

end NUMINAMATH_CALUDE_function_inequality_l3554_355427


namespace NUMINAMATH_CALUDE_pentagon_reflection_rotation_l3554_355443

/-- A regular pentagon -/
structure RegularPentagon where
  vertices : Fin 5 → ℝ × ℝ
  is_regular : ∀ i j : Fin 5, dist (vertices i) (vertices ((i + 1) % 5)) = dist (vertices j) (vertices ((j + 1) % 5))
  is_pentagon : ∀ i : Fin 5, vertices i ≠ vertices ((i + 1) % 5)

/-- Reflection of a point over a line through the center of the pentagon -/
def reflect (p : RegularPentagon) (line : ℝ × ℝ → Prop) : RegularPentagon :=
  sorry

/-- Rotation of a pentagon by an angle about its center -/
def rotate (p : RegularPentagon) (angle : ℝ) : RegularPentagon :=
  sorry

/-- The center of a regular pentagon -/
def center (p : RegularPentagon) : ℝ × ℝ :=
  sorry

theorem pentagon_reflection_rotation (p : RegularPentagon) (line : ℝ × ℝ → Prop) :
  rotate (reflect p line) (144 * π / 180) = rotate p (144 * π / 180) :=
sorry

end NUMINAMATH_CALUDE_pentagon_reflection_rotation_l3554_355443


namespace NUMINAMATH_CALUDE_local_max_implies_c_eq_6_l3554_355493

/-- The function f(x) = x(x-c)^2 has a local maximum at x=2 -/
def has_local_max_at_2 (c : ℝ) : Prop :=
  let f := fun x => x * (x - c)^2
  (∃ δ > 0, ∀ x, |x - 2| < δ → f x ≤ f 2) ∧
  (∀ ε > 0, ∃ x, |x - 2| < ε ∧ f x < f 2)

/-- If f(x) = x(x-c)^2 has a local maximum at x=2, then c = 6 -/
theorem local_max_implies_c_eq_6 :
  ∀ c : ℝ, has_local_max_at_2 c → c = 6 := by
  sorry

end NUMINAMATH_CALUDE_local_max_implies_c_eq_6_l3554_355493


namespace NUMINAMATH_CALUDE_air_inhaled_24_hours_l3554_355498

/-- The volume of air inhaled in 24 hours given the breathing rate and volume per breath -/
theorem air_inhaled_24_hours 
  (breaths_per_minute : ℕ) 
  (air_per_breath : ℚ) 
  (h1 : breaths_per_minute = 17) 
  (h2 : air_per_breath = 5/9) : 
  (breaths_per_minute : ℚ) * air_per_breath * (24 * 60) = 13600 := by
  sorry

end NUMINAMATH_CALUDE_air_inhaled_24_hours_l3554_355498


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l3554_355487

theorem absolute_value_inequality (x : ℝ) : |x - 3| < 5 ↔ -2 < x ∧ x < 8 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l3554_355487


namespace NUMINAMATH_CALUDE_infinite_geometric_series_ratio_l3554_355412

/-- 
For an infinite geometric series with first term a and sum S,
prove that if a = 500 and S = 3500, then the common ratio r is 6/7.
-/
theorem infinite_geometric_series_ratio 
  (a S : ℝ) 
  (h_a : a = 500) 
  (h_S : S = 3500) 
  (h_sum : S = a / (1 - r)) 
  (h_conv : |r| < 1) : 
  r = 6/7 := by
sorry

end NUMINAMATH_CALUDE_infinite_geometric_series_ratio_l3554_355412


namespace NUMINAMATH_CALUDE_max_expression_value_l3554_355436

def max_expression (a b c d : ℕ) : ℕ := c * a^(b + d)

theorem max_expression_value :
  ∃ (a b c d : ℕ),
    a ∈ ({0, 1, 2, 3, 4} : Set ℕ) ∧
    b ∈ ({0, 1, 2, 3, 4} : Set ℕ) ∧
    c ∈ ({0, 1, 2, 3, 4} : Set ℕ) ∧
    d ∈ ({0, 1, 2, 3, 4} : Set ℕ) ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    max_expression a b c d = 1024 ∧
    ∀ (w x y z : ℕ),
      w ∈ ({0, 1, 2, 3, 4} : Set ℕ) →
      x ∈ ({0, 1, 2, 3, 4} : Set ℕ) →
      y ∈ ({0, 1, 2, 3, 4} : Set ℕ) →
      z ∈ ({0, 1, 2, 3, 4} : Set ℕ) →
      w ≠ x ∧ w ≠ y ∧ w ≠ z ∧ x ≠ y ∧ x ≠ z ∧ y ≠ z →
      max_expression w x y z ≤ 1024 :=
by
  sorry

end NUMINAMATH_CALUDE_max_expression_value_l3554_355436


namespace NUMINAMATH_CALUDE_perpendicular_line_m_value_l3554_355453

/-- Given a line passing through points (1, 2) and (m, 3) that is perpendicular
    to the line 2x - 3y + 1 = 0, prove that m = 1/3 -/
theorem perpendicular_line_m_value (m : ℝ) :
  let line1 := {(x, y) : ℝ × ℝ | 2*x - 3*y + 1 = 0}
  let line2 := {(x, y) : ℝ × ℝ | ∃ t : ℝ, x = 1 + t*(m - 1) ∧ y = 2 + t*(3 - 2)}
  (∀ (p q : ℝ × ℝ), p ∈ line1 → q ∈ line1 → p ≠ q →
    ∀ (r s : ℝ × ℝ), r ∈ line2 → s ∈ line2 → r ≠ s →
      (p.2 - q.2) * (r.1 - s.1) = -(p.1 - q.1) * (r.2 - s.2)) →
  m = 1/3 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_line_m_value_l3554_355453


namespace NUMINAMATH_CALUDE_expression_evaluation_l3554_355406

theorem expression_evaluation : 
  (2002 : ℤ)^3 - 2001 * 2002^2 - 2001^2 * 2002 + 2001^3 + (2002 - 2001)^3 = 4004 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3554_355406


namespace NUMINAMATH_CALUDE_circle_radius_from_spherical_coordinates_l3554_355488

theorem circle_radius_from_spherical_coordinates : 
  let r : ℝ := Real.sqrt 3 / 2
  ∀ θ : ℝ, 
    let x : ℝ := Real.sin (π/3) * Real.cos θ
    let y : ℝ := Real.sin (π/3) * Real.sin θ
    Real.sqrt (x^2 + y^2) = r := by sorry

end NUMINAMATH_CALUDE_circle_radius_from_spherical_coordinates_l3554_355488


namespace NUMINAMATH_CALUDE_x_gt_one_sufficient_not_necessary_l3554_355491

theorem x_gt_one_sufficient_not_necessary :
  (∃ x : ℝ, x > 1 → (1 / x) < 1) ∧
  (∃ x : ℝ, (1 / x) < 1 ∧ ¬(x > 1)) :=
by sorry

end NUMINAMATH_CALUDE_x_gt_one_sufficient_not_necessary_l3554_355491


namespace NUMINAMATH_CALUDE_orange_slices_theorem_l3554_355424

/-- Represents the number of slices each animal received -/
structure OrangeSlices where
  siskin : ℕ
  hedgehog : ℕ
  beaver : ℕ

/-- Conditions for the orange slices distribution -/
def valid_distribution (slices : OrangeSlices) : Prop :=
  slices.hedgehog = 2 * slices.siskin ∧
  slices.beaver = 5 * slices.siskin ∧
  slices.beaver = slices.siskin + 8

/-- The total number of slices in the orange -/
def total_slices (slices : OrangeSlices) : ℕ :=
  slices.siskin + slices.hedgehog + slices.beaver

/-- Theorem stating that the total number of slices is 16 -/
theorem orange_slices_theorem :
  ∃ (slices : OrangeSlices), valid_distribution slices ∧ total_slices slices = 16 :=
sorry

end NUMINAMATH_CALUDE_orange_slices_theorem_l3554_355424


namespace NUMINAMATH_CALUDE_direct_inverse_variation_l3554_355425

theorem direct_inverse_variation (k : ℝ) (P₀ Q₀ R₀ P₁ R₁ : ℝ) :
  P₀ = k * Q₀ / Real.sqrt R₀ →
  P₀ = 9/4 →
  R₀ = 16/25 →
  Q₀ = 5/8 →
  P₁ = 27 →
  R₁ = 1/36 →
  k * (5/8) / Real.sqrt (16/25) = 9/4 →
  ∃ Q₁ : ℝ, P₁ = k * Q₁ / Real.sqrt R₁ ∧ Q₁ = 1.56 := by
sorry

end NUMINAMATH_CALUDE_direct_inverse_variation_l3554_355425


namespace NUMINAMATH_CALUDE_min_singing_in_shower_l3554_355470

/-- Represents the youth summer village population -/
structure Village where
  total : ℕ
  notWorking : ℕ
  withFamilies : ℕ
  workingNoFamilySinging : ℕ

/-- The minimum number of people who like to sing in the shower -/
def minSingingInShower (v : Village) : ℕ := v.workingNoFamilySinging

theorem min_singing_in_shower (v : Village) 
  (h1 : v.total = 100)
  (h2 : v.notWorking = 50)
  (h3 : v.withFamilies = 25)
  (h4 : v.workingNoFamilySinging = 50)
  (h5 : v.workingNoFamilySinging ≤ v.total - v.notWorking)
  (h6 : v.workingNoFamilySinging ≤ v.total - v.withFamilies) :
  minSingingInShower v = 50 := by
  sorry

#check min_singing_in_shower

end NUMINAMATH_CALUDE_min_singing_in_shower_l3554_355470
