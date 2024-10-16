import Mathlib

namespace NUMINAMATH_CALUDE_apple_cost_price_l1474_147434

/-- Proves that given a selling price of 15 and a loss of 1/6th of the cost price, the cost price of the apple is 18. -/
theorem apple_cost_price (selling_price : ℚ) (loss_fraction : ℚ) : 
  selling_price = 15 ∧ loss_fraction = 1/6 → 
  ∃ (cost_price : ℚ), cost_price = 18 ∧ selling_price = cost_price * (1 - loss_fraction) :=
by sorry

end NUMINAMATH_CALUDE_apple_cost_price_l1474_147434


namespace NUMINAMATH_CALUDE_set_operation_result_l1474_147466

def A : Set ℤ := {-1, 0, 1}
def B : Set ℤ := {1, 3, 5}
def C : Set ℤ := {0, 2, 4}

theorem set_operation_result : (A ∩ B) ∪ C = {0, 1, 2, 4} := by
  sorry

end NUMINAMATH_CALUDE_set_operation_result_l1474_147466


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_range_l1474_147450

theorem quadratic_inequality_solution_range (c : ℝ) : 
  (c > 0 ∧ ∃ x : ℝ, x^2 - 10*x + c < 0) ↔ (0 < c ∧ c < 25) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_range_l1474_147450


namespace NUMINAMATH_CALUDE_smallest_n_for_roots_of_unity_l1474_147405

def polynomial (z : ℂ) : ℂ := z^4 + z^3 + z^2 + z + 1

def is_nth_root_of_unity (z : ℂ) (n : ℕ) : Prop := z^n = 1

theorem smallest_n_for_roots_of_unity : 
  ∃ (n : ℕ), n > 0 ∧ 
  (∀ (z : ℂ), polynomial z = 0 → is_nth_root_of_unity z n) ∧
  (∀ (m : ℕ), m > 0 → m < n → 
    ∃ (w : ℂ), polynomial w = 0 ∧ ¬is_nth_root_of_unity w m) ∧
  n = 5 := by sorry

end NUMINAMATH_CALUDE_smallest_n_for_roots_of_unity_l1474_147405


namespace NUMINAMATH_CALUDE_track_length_l1474_147408

/-- The length of a circular track given two cyclists' speeds and meeting time -/
theorem track_length (speed_a speed_b meeting_time : ℝ) : 
  speed_a = 36 →
  speed_b = 72 →
  meeting_time = 19.99840012798976 →
  (speed_b - speed_a) * meeting_time / 60 = 11.999040076793856 :=
by sorry

end NUMINAMATH_CALUDE_track_length_l1474_147408


namespace NUMINAMATH_CALUDE_time_for_one_toy_l1474_147485

/-- Represents the time (in hours) it takes to make a certain number of toys -/
structure ToyProduction where
  hours : ℝ
  toys : ℝ

/-- Given that 50 toys are made in 100 hours, prove that it takes 2 hours to make one toy -/
theorem time_for_one_toy (prod : ToyProduction) 
  (h1 : prod.hours = 100) 
  (h2 : prod.toys = 50) : 
  prod.hours / prod.toys = 2 := by
  sorry

end NUMINAMATH_CALUDE_time_for_one_toy_l1474_147485


namespace NUMINAMATH_CALUDE_circle_radius_from_diameter_l1474_147418

theorem circle_radius_from_diameter (diameter : ℝ) (radius : ℝ) :
  diameter = 14 → radius = diameter / 2 → radius = 7 := by sorry

end NUMINAMATH_CALUDE_circle_radius_from_diameter_l1474_147418


namespace NUMINAMATH_CALUDE_equation_solution_l1474_147424

theorem equation_solution :
  ∀ t : ℂ, (2 / (t + 3) + 3 * t / (t + 3) - 5 / (t + 3) = t + 2) ↔ 
  (t = -1 + 2 * Complex.I * Real.sqrt 2 ∨ t = -1 - 2 * Complex.I * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l1474_147424


namespace NUMINAMATH_CALUDE_equal_to_one_half_l1474_147456

theorem equal_to_one_half : 
  Real.sqrt ((1 + Real.cos (2 * Real.pi / 3)) / 2) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_equal_to_one_half_l1474_147456


namespace NUMINAMATH_CALUDE_simplify_expression_l1474_147451

theorem simplify_expression (x : ℝ) : (3*x + 25) - (2*x - 5) = x + 30 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1474_147451


namespace NUMINAMATH_CALUDE_polygon_diagonals_l1474_147490

/-- For an n-sided polygon, if one vertex has 5 diagonals, then n = 8. -/
theorem polygon_diagonals (n : ℕ) (h : n - 3 = 5) : n = 8 := by
  sorry

end NUMINAMATH_CALUDE_polygon_diagonals_l1474_147490


namespace NUMINAMATH_CALUDE_continuous_with_property_F_is_nondecreasing_l1474_147431

-- Define property F
def has_property_F (f : ℝ → ℝ) : Prop :=
  ∀ a : ℝ, ∃ b : ℝ, b < a ∧ ∀ x ∈ Set.Ioo b a, f x ≤ f a

-- Define nondecreasing
def nondecreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x ≤ y → f x ≤ f y

-- Theorem statement
theorem continuous_with_property_F_is_nondecreasing (f : ℝ → ℝ) 
  (hf : Continuous f) (hF : has_property_F f) : nondecreasing f := by
  sorry


end NUMINAMATH_CALUDE_continuous_with_property_F_is_nondecreasing_l1474_147431


namespace NUMINAMATH_CALUDE_square_area_error_l1474_147402

theorem square_area_error (a : ℝ) (h : a > 0) :
  let measured_side := a * (1 + 0.08)
  let actual_area := a^2
  let calculated_area := measured_side^2
  let area_error := (calculated_area - actual_area) / actual_area
  area_error = 0.1664 := by
sorry

end NUMINAMATH_CALUDE_square_area_error_l1474_147402


namespace NUMINAMATH_CALUDE_sets_equivalence_l1474_147475

-- Define the sets
def A : Set ℝ := {1}
def B : Set ℝ := {y : ℝ | (y - 1)^2 = 0}
def D : Set ℝ := {x : ℝ | x - 1 = 0}

-- C is not defined as a set because it's not a valid set notation

theorem sets_equivalence :
  (A = B) ∧ (A = D) ∧ (B = D) :=
sorry

-- Note: We can't include C in the theorem because it's not a valid set

end NUMINAMATH_CALUDE_sets_equivalence_l1474_147475


namespace NUMINAMATH_CALUDE_inequality_proof_l1474_147474

theorem inequality_proof (a b c : ℝ) (ha : a = 31/32) (hb : b = Real.cos (1/4)) (hc : c = 4 * Real.sin (1/4)) : c > b ∧ b > a := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1474_147474


namespace NUMINAMATH_CALUDE_no_integer_m_for_single_solution_l1474_147460

theorem no_integer_m_for_single_solution :
  ¬ ∃ (m : ℤ), ∃! (x : ℝ), 36 * x^2 - m * x - 4 = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_m_for_single_solution_l1474_147460


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l1474_147416

theorem complex_fraction_equality : (1 + 3 * Complex.I) / (1 - Complex.I) = -1 + 2 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l1474_147416


namespace NUMINAMATH_CALUDE_coefficient_a9_l1474_147455

theorem coefficient_a9 (a a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ : ℝ) :
  (fun x : ℝ => x^2 + x^10) = 
  (fun x : ℝ => a + a₁*(x+1) + a₂*(x+1)^2 + a₃*(x+1)^3 + a₄*(x+1)^4 + 
                a₅*(x+1)^5 + a₆*(x+1)^6 + a₇*(x+1)^7 + a₈*(x+1)^8 + 
                a₉*(x+1)^9 + a₁₀*(x+1)^10) →
  a₉ = -10 := by
sorry

end NUMINAMATH_CALUDE_coefficient_a9_l1474_147455


namespace NUMINAMATH_CALUDE_limes_given_correct_l1474_147471

/-- The number of limes Dan initially picked -/
def initial_limes : ℕ := 9

/-- The number of limes Dan has now -/
def current_limes : ℕ := 5

/-- The number of limes Dan gave to Sara -/
def limes_given : ℕ := initial_limes - current_limes

theorem limes_given_correct : limes_given = 4 := by sorry

end NUMINAMATH_CALUDE_limes_given_correct_l1474_147471


namespace NUMINAMATH_CALUDE_blue_markers_count_l1474_147428

theorem blue_markers_count (total : ℝ) (red : ℝ) (h1 : total = 64.0) (h2 : red = 41.0) :
  total - red = 23.0 := by
  sorry

end NUMINAMATH_CALUDE_blue_markers_count_l1474_147428


namespace NUMINAMATH_CALUDE_no_solution_l1474_147426

theorem no_solution : ¬∃ (A B : ℤ), 
  A = 5 + 3 ∧ 
  B = A - 2 ∧ 
  0 ≤ A ∧ A ≤ 9 ∧ 
  0 ≤ B ∧ B ≤ 9 ∧ 
  0 ≤ A + B ∧ A + B ≤ 9 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_l1474_147426


namespace NUMINAMATH_CALUDE_football_game_attendance_l1474_147438

theorem football_game_attendance (S : ℕ) 
  (hMonday : ℕ → ℕ := λ x => x - 20)
  (hWednesday : ℕ → ℕ := λ x => x + 50)
  (hFriday : ℕ → ℕ := λ x => x * 2 - 20)
  (hExpected : ℕ := 350)
  (hActual : ℕ := hExpected + 40)
  (hTotal : ℕ → ℕ := λ x => x + hMonday x + hWednesday (hMonday x) + hFriday x) :
  hTotal S = hActual → S = 80 := by
sorry

end NUMINAMATH_CALUDE_football_game_attendance_l1474_147438


namespace NUMINAMATH_CALUDE_opposite_of_sqrt_six_l1474_147467

theorem opposite_of_sqrt_six :
  ∀ x : ℝ, x = Real.sqrt 6 → -x = -(Real.sqrt 6) :=
by sorry

end NUMINAMATH_CALUDE_opposite_of_sqrt_six_l1474_147467


namespace NUMINAMATH_CALUDE_square_diff_value_l1474_147417

theorem square_diff_value (x y : ℝ) (h1 : x + y = 20) (h2 : x - y = 4) : x^2 - y^2 = 80 := by
  sorry

end NUMINAMATH_CALUDE_square_diff_value_l1474_147417


namespace NUMINAMATH_CALUDE_french_english_speakers_l1474_147436

theorem french_english_speakers (total_students : ℕ) 
  (non_french_percentage : ℚ) (french_non_english : ℕ) : 
  total_students = 200 →
  non_french_percentage = 3/4 →
  french_non_english = 40 →
  (total_students : ℚ) * (1 - non_french_percentage) - french_non_english = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_french_english_speakers_l1474_147436


namespace NUMINAMATH_CALUDE_gcd_231_154_l1474_147495

theorem gcd_231_154 : Nat.gcd 231 154 = 77 := by sorry

end NUMINAMATH_CALUDE_gcd_231_154_l1474_147495


namespace NUMINAMATH_CALUDE_inequality_proof_l1474_147464

theorem inequality_proof (a b : ℝ) (h : a > b) : -3 * a < -3 * b := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1474_147464


namespace NUMINAMATH_CALUDE_karen_crayons_count_l1474_147492

/-- The number of crayons Cindy has -/
def cindy_crayons : ℕ := 504

/-- The number of additional crayons Karen has compared to Cindy -/
def karen_additional_crayons : ℕ := 135

/-- The number of crayons Karen has -/
def karen_crayons : ℕ := cindy_crayons + karen_additional_crayons

theorem karen_crayons_count : karen_crayons = 639 := by
  sorry

end NUMINAMATH_CALUDE_karen_crayons_count_l1474_147492


namespace NUMINAMATH_CALUDE_socks_needed_to_triple_wardrobe_l1474_147481

/-- Represents the number of items in Jonas' wardrobe -/
structure Wardrobe where
  socks : ℕ
  shoes : ℕ
  pants : ℕ
  tshirts : ℕ
  hats : ℕ
  jackets : ℕ

/-- Calculates the total number of individual items in the wardrobe -/
def totalItems (w : Wardrobe) : ℕ :=
  w.socks * 2 + w.shoes * 2 + w.pants + w.tshirts + w.hats + w.jackets

/-- Jonas' current wardrobe -/
def jonasWardrobe : Wardrobe :=
  { socks := 20
    shoes := 5
    pants := 10
    tshirts := 10
    hats := 6
    jackets := 4 }

/-- Theorem: Jonas needs to buy 80 pairs of socks to triple his wardrobe -/
theorem socks_needed_to_triple_wardrobe :
  let current := totalItems jonasWardrobe
  let target := current * 3
  let difference := target - current
  difference / 2 = 80 := by sorry

end NUMINAMATH_CALUDE_socks_needed_to_triple_wardrobe_l1474_147481


namespace NUMINAMATH_CALUDE_max_product_sum_2020_l1474_147487

theorem max_product_sum_2020 : 
  (∃ (a b : ℤ), a + b = 2020 ∧ a * b = 1020100) ∧ 
  (∀ (x y : ℤ), x + y = 2020 → x * y ≤ 1020100) := by
sorry

end NUMINAMATH_CALUDE_max_product_sum_2020_l1474_147487


namespace NUMINAMATH_CALUDE_custom_distance_additive_on_line_segment_l1474_147440

/-- Custom distance function between two points in 2D space -/
def custom_distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  |x₂ - x₁| + |y₂ - y₁|

/-- Theorem: For any three points A, B, and C, where C is on the line segment AB,
    the sum of the custom distances AC and CB equals the custom distance AB -/
theorem custom_distance_additive_on_line_segment 
  (x₁ y₁ x₂ y₂ x y : ℝ) 
  (h_between_x : (x₁ - x) * (x₂ - x) ≤ 0)
  (h_between_y : (y₁ - y) * (y₂ - y) ≤ 0) :
  custom_distance x₁ y₁ x y + custom_distance x y x₂ y₂ = custom_distance x₁ y₁ x₂ y₂ :=
by sorry

#check custom_distance_additive_on_line_segment

end NUMINAMATH_CALUDE_custom_distance_additive_on_line_segment_l1474_147440


namespace NUMINAMATH_CALUDE_number_of_students_l1474_147444

-- Define the lottery winnings
def lottery_winnings : ℚ := 155250

-- Define the fraction given to each student
def fraction_per_student : ℚ := 1 / 1000

-- Define the total amount received by students
def total_received : ℚ := 15525

-- Theorem to prove
theorem number_of_students : 
  (total_received / (lottery_winnings * fraction_per_student) : ℚ) = 100 := by
  sorry

end NUMINAMATH_CALUDE_number_of_students_l1474_147444


namespace NUMINAMATH_CALUDE_triangle_angle_measure_l1474_147439

theorem triangle_angle_measure (D E F : ℝ) : 
  D = 80 →
  E = 2 * F + 24 →
  D + E + F = 180 →
  F = 76 / 3 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_measure_l1474_147439


namespace NUMINAMATH_CALUDE_right_angled_complex_roots_l1474_147463

open Complex

theorem right_angled_complex_roots (a b : ℂ) (z₁ z₂ : ℂ) : 
  z₁^2 + a*z₁ + b = 0 → 
  z₂^2 + a*z₂ + b = 0 → 
  z₁ ≠ 0 → 
  z₂ ≠ 0 → 
  z₁ ≠ z₂ → 
  (z₁.re * z₂.re + z₁.im * z₂.im = 0) → 
  a^2 / b = 2 := by
sorry

end NUMINAMATH_CALUDE_right_angled_complex_roots_l1474_147463


namespace NUMINAMATH_CALUDE_sale_price_is_twenty_l1474_147407

/-- The sale price of one bottle of detergent, given the number of loads per bottle and the cost per load when buying two bottles. -/
def sale_price (loads_per_bottle : ℕ) (cost_per_load : ℚ) : ℚ :=
  loads_per_bottle * cost_per_load

/-- Theorem stating that the sale price of one bottle of detergent is $20.00 -/
theorem sale_price_is_twenty :
  sale_price 80 (25 / 100) = 20 := by
  sorry

end NUMINAMATH_CALUDE_sale_price_is_twenty_l1474_147407


namespace NUMINAMATH_CALUDE_greatest_fraction_l1474_147496

theorem greatest_fraction (a : ℝ) (m n p : ℝ) 
  (h_a : a < -3)
  (h_m : m = (a + 2) / (a + 3))
  (h_n : n = (a + 1) / (a + 2))
  (h_p : p = a / (a + 1)) :
  m > n ∧ n > p := by
sorry

end NUMINAMATH_CALUDE_greatest_fraction_l1474_147496


namespace NUMINAMATH_CALUDE_max_m_quadratic_inequality_l1474_147400

theorem max_m_quadratic_inequality (a b c : ℝ) (h_real_roots : b^2 - 4*a*c ≥ 0) :
  ∃ (m : ℝ), m = 9/8 ∧ 
  (∀ (k : ℝ), ((a-b)^2 + (b-c)^2 + (c-a)^2 ≥ k*a^2) → k ≤ m) ∧
  ((a-b)^2 + (b-c)^2 + (c-a)^2 ≥ m*a^2) := by
  sorry

end NUMINAMATH_CALUDE_max_m_quadratic_inequality_l1474_147400


namespace NUMINAMATH_CALUDE_max_area_four_squares_l1474_147433

/-- The maximum area covered by 4 squares with side length 2 when arranged to form a larger square -/
theorem max_area_four_squares (n : ℕ) (side_length : ℝ) (h1 : n = 4) (h2 : side_length = 2) :
  n * side_length^2 - (n - 1) = 13 :=
sorry

end NUMINAMATH_CALUDE_max_area_four_squares_l1474_147433


namespace NUMINAMATH_CALUDE_smallest_prime_factor_of_2343_l1474_147493

theorem smallest_prime_factor_of_2343 : 
  Nat.minFac 2343 = 3 := by
sorry

end NUMINAMATH_CALUDE_smallest_prime_factor_of_2343_l1474_147493


namespace NUMINAMATH_CALUDE_david_presents_l1474_147421

theorem david_presents (christmas_presents : ℕ) (birthday_presents : ℕ) : 
  christmas_presents = 2 * birthday_presents →
  christmas_presents = 60 →
  christmas_presents + birthday_presents = 90 := by
sorry

end NUMINAMATH_CALUDE_david_presents_l1474_147421


namespace NUMINAMATH_CALUDE_number_of_clerks_l1474_147406

/-- Proves that the number of clerks is 170 given the salary information -/
theorem number_of_clerks (total_avg : ℚ) (officer_avg : ℚ) (clerk_avg : ℚ) (num_officers : ℕ) :
  total_avg = 90 →
  officer_avg = 600 →
  clerk_avg = 84 →
  num_officers = 2 →
  ∃ (num_clerks : ℕ), 
    (num_officers * officer_avg + num_clerks * clerk_avg) / (num_officers + num_clerks) = total_avg ∧
    num_clerks = 170 := by
  sorry


end NUMINAMATH_CALUDE_number_of_clerks_l1474_147406


namespace NUMINAMATH_CALUDE_smallest_n_perfect_powers_l1474_147457

theorem smallest_n_perfect_powers : ∃ (n : ℕ),
  (n = 1944) ∧
  (∃ (m : ℕ), 2 * n = m^4) ∧
  (∃ (l : ℕ), 3 * n = l^6) ∧
  (∀ (k : ℕ), k < n →
    (∃ (p : ℕ), 2 * k = p^4) →
    (∃ (q : ℕ), 3 * k = q^6) →
    False) :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_perfect_powers_l1474_147457


namespace NUMINAMATH_CALUDE_calculate_tip_percentage_l1474_147483

/-- Calculates the percentage tip given the prices of four ice cream sundaes and the final bill -/
theorem calculate_tip_percentage (price1 price2 price3 price4 final_bill : ℚ) : 
  price1 = 9 ∧ price2 = 7.5 ∧ price3 = 10 ∧ price4 = 8.5 ∧ final_bill = 42 →
  (final_bill - (price1 + price2 + price3 + price4)) / (price1 + price2 + price3 + price4) * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_calculate_tip_percentage_l1474_147483


namespace NUMINAMATH_CALUDE_log_not_always_decreasing_l1474_147446

-- Define the logarithmic function
noncomputable def log (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

-- State the theorem
theorem log_not_always_decreasing :
  ¬ (∀ (a : ℝ), a > 0 → a ≠ 1 → 
    (∀ (x₁ x₂ : ℝ), x₁ > 0 → x₂ > 0 → x₁ < x₂ → log a x₁ > log a x₂)) :=
by sorry

end NUMINAMATH_CALUDE_log_not_always_decreasing_l1474_147446


namespace NUMINAMATH_CALUDE_max_correct_answers_l1474_147484

theorem max_correct_answers (total_questions : ℕ) (correct_points : ℤ) (wrong_points : ℤ) (total_score : ℤ) :
  total_questions = 60 →
  correct_points = 5 →
  wrong_points = -2 →
  total_score = 150 →
  (∃ (correct unpicked wrong : ℕ),
    correct + unpicked + wrong = total_questions ∧
    correct * correct_points + wrong * wrong_points = total_score) →
  (∀ (x : ℕ), x > 38 →
    ¬∃ (unpicked wrong : ℕ),
      x + unpicked + wrong = total_questions ∧
      x * correct_points + wrong * wrong_points = total_score) :=
by sorry

end NUMINAMATH_CALUDE_max_correct_answers_l1474_147484


namespace NUMINAMATH_CALUDE_fruits_given_to_jane_l1474_147443

def initial_plums : ℕ := 16
def initial_guavas : ℕ := 18
def initial_apples : ℕ := 21
def fruits_left : ℕ := 15

def total_initial_fruits : ℕ := initial_plums + initial_guavas + initial_apples

theorem fruits_given_to_jane :
  total_initial_fruits - fruits_left = 40 := by sorry

end NUMINAMATH_CALUDE_fruits_given_to_jane_l1474_147443


namespace NUMINAMATH_CALUDE_f_properties_l1474_147486

noncomputable def f (k : ℝ) (x : ℝ) : ℝ := Real.log x + k * x^2 + (2 * k + 1) * x

theorem f_properties (k : ℝ) :
  (k ≥ 0 → ∀ x₁ x₂ : ℝ, 0 < x₁ → 0 < x₂ → x₁ < x₂ → f k x₁ < f k x₂) ∧
  (k < 0 → ∀ x : ℝ, 0 < x → f k x ≤ -3 / (4 * k) - 2) := by
  sorry

end NUMINAMATH_CALUDE_f_properties_l1474_147486


namespace NUMINAMATH_CALUDE_log_equation_holds_l1474_147419

theorem log_equation_holds (x : ℝ) (h1 : x > 0) (h2 : x ≠ 1) :
  (Real.log x / Real.log 2) * (Real.log 9 / Real.log x) = Real.log 9 / Real.log 2 := by
  sorry

end NUMINAMATH_CALUDE_log_equation_holds_l1474_147419


namespace NUMINAMATH_CALUDE_geometric_sequence_increasing_condition_l1474_147420

/-- A geometric sequence with common ratio q -/
def GeometricSequence (a : ℕ → ℝ) (q : ℝ) :=
  ∀ n, a (n + 1) = q * a n

/-- An increasing sequence -/
def IncreasingSequence (a : ℕ → ℝ) :=
  ∀ n, a (n + 1) > a n

/-- The condition "q > 1" is neither sufficient nor necessary for a geometric sequence to be increasing -/
theorem geometric_sequence_increasing_condition (a : ℕ → ℝ) (q : ℝ) :
  (GeometricSequence a q) →
  (¬(q > 1 → IncreasingSequence a) ∧ ¬(IncreasingSequence a → q > 1)) :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_increasing_condition_l1474_147420


namespace NUMINAMATH_CALUDE_factorization_equality_l1474_147425

theorem factorization_equality (a b : ℝ) : a * b^2 - a = a * (b + 1) * (b - 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l1474_147425


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l1474_147494

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def f_definition (f : ℝ → ℝ) : Prop :=
  ∀ x ≥ 0, f x = x^2 - 4*x

theorem solution_set_of_inequality (f : ℝ → ℝ) 
  (h_even : is_even_function f) 
  (h_def : f_definition f) :
  {x : ℝ | f (x + 2) < 5} = Set.Ioo (-7) 3 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l1474_147494


namespace NUMINAMATH_CALUDE_nancy_gardens_l1474_147404

theorem nancy_gardens (total_seeds : ℕ) (big_garden_seeds : ℕ) (seeds_per_small_garden : ℕ) 
  (h1 : total_seeds = 52)
  (h2 : big_garden_seeds = 28)
  (h3 : seeds_per_small_garden = 4) :
  (total_seeds - big_garden_seeds) / seeds_per_small_garden = 6 :=
by sorry

end NUMINAMATH_CALUDE_nancy_gardens_l1474_147404


namespace NUMINAMATH_CALUDE_fraction_multiplication_equality_l1474_147437

theorem fraction_multiplication_equality : 
  (11/12 - 7/6 + 3/4 - 13/24) * (-48) = 2 := by sorry

end NUMINAMATH_CALUDE_fraction_multiplication_equality_l1474_147437


namespace NUMINAMATH_CALUDE_minimal_fraction_sum_l1474_147414

theorem minimal_fraction_sum (a b : ℕ+) (h : (45 : ℚ) / 11 < (a : ℚ) / b ∧ (a : ℚ) / b < 5 / 11) :
  (∀ c d : ℕ+, (45 : ℚ) / 11 < (c : ℚ) / d ∧ (c : ℚ) / d < 5 / 11 → c + d ≥ a + b) →
  a = 3 ∧ b = 7 :=
sorry

end NUMINAMATH_CALUDE_minimal_fraction_sum_l1474_147414


namespace NUMINAMATH_CALUDE_equal_money_time_l1474_147472

/-- 
Proves that Carol and Mike will have the same amount of money after 5 weeks,
given their initial amounts and weekly savings rates.
-/
theorem equal_money_time (carol_initial : ℕ) (mike_initial : ℕ) 
  (carol_weekly : ℕ) (mike_weekly : ℕ) :
  carol_initial = 60 →
  mike_initial = 90 →
  carol_weekly = 9 →
  mike_weekly = 3 →
  ∃ w : ℕ, w = 5 ∧ carol_initial + w * carol_weekly = mike_initial + w * mike_weekly :=
by
  sorry

#check equal_money_time

end NUMINAMATH_CALUDE_equal_money_time_l1474_147472


namespace NUMINAMATH_CALUDE_log_roll_volume_l1474_147489

theorem log_roll_volume (log_length : ℝ) (large_radius small_radius : ℝ) :
  log_length = 10 ∧ 
  large_radius = 3 ∧ 
  small_radius = 1 →
  let path_radius := large_radius + small_radius
  let cross_section_area := π * large_radius^2 + π * path_radius^2 / 2 - π * small_radius^2 / 2
  cross_section_area * log_length = 155 * π :=
by sorry

end NUMINAMATH_CALUDE_log_roll_volume_l1474_147489


namespace NUMINAMATH_CALUDE_odd_multiple_of_nine_is_multiple_of_three_l1474_147447

theorem odd_multiple_of_nine_is_multiple_of_three :
  (∀ n : ℕ, 9 ∣ n → 3 ∣ n) →
  ∀ k : ℕ, Odd k → 9 ∣ k → 3 ∣ k :=
by
  sorry

end NUMINAMATH_CALUDE_odd_multiple_of_nine_is_multiple_of_three_l1474_147447


namespace NUMINAMATH_CALUDE_jogger_train_distance_l1474_147409

/-- Proves that a jogger is 240 meters ahead of a train given specific conditions -/
theorem jogger_train_distance (jogger_speed : ℝ) (train_speed : ℝ) (train_length : ℝ) (passing_time : ℝ) :
  jogger_speed = 9 * (1000 / 3600) →
  train_speed = 45 * (1000 / 3600) →
  train_length = 120 →
  passing_time = 36 →
  (train_speed - jogger_speed) * passing_time - train_length = 240 :=
by
  sorry

#check jogger_train_distance

end NUMINAMATH_CALUDE_jogger_train_distance_l1474_147409


namespace NUMINAMATH_CALUDE_movies_watched_correct_l1474_147469

/-- The number of movies watched in the 'crazy silly school' series --/
def moviesWatched (totalMovies : ℕ) (moviesToWatch : ℕ) : ℕ :=
  totalMovies - moviesToWatch

/-- Theorem: The number of movies watched is correct --/
theorem movies_watched_correct (totalMovies moviesToWatch : ℕ) 
  (h1 : totalMovies = 17) 
  (h2 : moviesToWatch = 10) : 
  moviesWatched totalMovies moviesToWatch = 7 := by
  sorry

#eval moviesWatched 17 10

end NUMINAMATH_CALUDE_movies_watched_correct_l1474_147469


namespace NUMINAMATH_CALUDE_segment_length_is_15_l1474_147461

/-- The length of a vertical line segment is the absolute difference of y-coordinates -/
def vertical_segment_length (y1 y2 : ℝ) : ℝ := |y2 - y1|

/-- Proof that the length of the segment with endpoints (3, 5) and (3, 20) is 15 units -/
theorem segment_length_is_15 : 
  vertical_segment_length 5 20 = 15 := by
  sorry

end NUMINAMATH_CALUDE_segment_length_is_15_l1474_147461


namespace NUMINAMATH_CALUDE_e_recursive_relation_l1474_147429

def e (n : ℕ) : ℕ := n^5

theorem e_recursive_relation (n : ℕ) :
  e (n + 6) = 6 * e (n + 5) - 15 * e (n + 4) + 20 * e (n + 3) - 15 * e (n + 2) + 6 * e (n + 1) - e n :=
by sorry

end NUMINAMATH_CALUDE_e_recursive_relation_l1474_147429


namespace NUMINAMATH_CALUDE_probability_theorem_l1474_147476

/-- Represents the number of athletes in each association -/
structure Associations where
  A : ℕ
  B : ℕ
  C : ℕ

/-- Represents the number of athletes selected from each association -/
structure SelectedAthletes where
  A : ℕ
  B : ℕ
  C : ℕ

/-- Calculates the probability of selecting at least one athlete from A5 or A6 -/
def probability_A5_or_A6 (total_selected : ℕ) (doubles_team_size : ℕ) : ℚ :=
  let favorable_outcomes := (total_selected - 2) * 2 + 1
  let total_outcomes := total_selected.choose doubles_team_size
  favorable_outcomes / total_outcomes

/-- Main theorem statement -/
theorem probability_theorem (assoc : Associations) (selected : SelectedAthletes) :
    assoc.A = 27 ∧ assoc.B = 9 ∧ assoc.C = 18 →
    selected.A = 3 ∧ selected.B = 1 ∧ selected.C = 2 →
    probability_A5_or_A6 (selected.A + selected.B + selected.C) 2 = 3/5 := by
  sorry


end NUMINAMATH_CALUDE_probability_theorem_l1474_147476


namespace NUMINAMATH_CALUDE_pilot_course_cost_difference_pilot_course_cost_difference_holds_l1474_147427

/-- The cost difference between flight and ground school portions of a private pilot course -/
theorem pilot_course_cost_difference : ℕ → ℕ → ℕ → ℕ → Prop :=
  fun total_cost flight_cost ground_cost difference =>
    total_cost = 1275 ∧
    flight_cost = 950 ∧
    ground_cost = 325 ∧
    total_cost = flight_cost + ground_cost ∧
    flight_cost > ground_cost ∧
    difference = flight_cost - ground_cost ∧
    difference = 625

/-- The theorem holds for the given costs -/
theorem pilot_course_cost_difference_holds :
  ∃ (total_cost flight_cost ground_cost difference : ℕ),
    pilot_course_cost_difference total_cost flight_cost ground_cost difference :=
by
  sorry

end NUMINAMATH_CALUDE_pilot_course_cost_difference_pilot_course_cost_difference_holds_l1474_147427


namespace NUMINAMATH_CALUDE_alien_alphabet_l1474_147412

theorem alien_alphabet (total : ℕ) (both : ℕ) (triangle_only : ℕ) 
  (h1 : total = 120)
  (h2 : both = 32)
  (h3 : triangle_only = 72)
  (h4 : total = both + triangle_only + (total - (both + triangle_only))) :
  total - (both + triangle_only) = 16 := by
  sorry

end NUMINAMATH_CALUDE_alien_alphabet_l1474_147412


namespace NUMINAMATH_CALUDE_range_of_a_for_unique_solution_l1474_147459

/-- The range of 'a' for which the equation lg(x-1) + lg(3-x) = lg(x-a) has exactly one solution for x, where 1 < x < 3 -/
theorem range_of_a_for_unique_solution : 
  ∀ a : ℝ, (∃! x : ℝ, 1 < x ∧ x < 3 ∧ Real.log (x - 1) + Real.log (3 - x) = Real.log (x - a)) ↔ 
  (a ≥ 3/4 ∧ a < 3) := by
sorry

end NUMINAMATH_CALUDE_range_of_a_for_unique_solution_l1474_147459


namespace NUMINAMATH_CALUDE_distance_sum_on_corresponding_segments_l1474_147468

/-- Given two line segments AB and A'B' with lengths 6 and 16 respectively,
    and a linear correspondence between points on these segments,
    prove that the sum of distances from A to P and A' to P' is 18/5 * a,
    where a is the distance from A to P. -/
theorem distance_sum_on_corresponding_segments
  (AB : Real) (A'B' : Real)
  (a : Real)
  (h1 : AB = 6)
  (h2 : A'B' = 16)
  (h3 : 0 ≤ a ∧ a ≤ AB)
  (correspondence : Real → Real)
  (h4 : correspondence 1 = 3)
  (h5 : ∀ x, 0 ≤ x ∧ x ≤ AB → 0 ≤ correspondence x ∧ correspondence x ≤ A'B')
  (h6 : ∀ x y, (0 ≤ x ∧ x ≤ AB ∧ 0 ≤ y ∧ y ≤ AB) →
              (correspondence x - correspondence y) / (x - y) = (correspondence 1 - 0) / (1 - 0)) :
  a + correspondence a = 18/5 * a := by
  sorry

end NUMINAMATH_CALUDE_distance_sum_on_corresponding_segments_l1474_147468


namespace NUMINAMATH_CALUDE_quadratic_root_conditions_l1474_147479

theorem quadratic_root_conditions (a : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ 
    x^2 + (a^2 - 1)*x + a - 2 = 0 ∧ 
    y^2 + (a^2 - 1)*y + a - 2 = 0 ∧ 
    x > 1 ∧ y < 1) ↔ 
  -2 < a ∧ a < 1 := by sorry

end NUMINAMATH_CALUDE_quadratic_root_conditions_l1474_147479


namespace NUMINAMATH_CALUDE_unique_peg_arrangement_l1474_147458

/-- Represents a color of a peg -/
inductive PegColor
  | Yellow
  | Red
  | Green
  | Blue
  | Orange

/-- Represents a position on the triangular peg board -/
structure Position :=
  (row : Nat)
  (col : Nat)

/-- Represents the triangular peg board -/
def Board := Position → Option PegColor

/-- Checks if a given board arrangement is valid -/
def is_valid_arrangement (board : Board) : Prop :=
  (∀ r c, board ⟨r, c⟩ = some PegColor.Yellow → r < 6 ∧ c < 6) ∧
  (∀ r c, board ⟨r, c⟩ = some PegColor.Red → r < 5 ∧ c < 6) ∧
  (∀ r c, board ⟨r, c⟩ = some PegColor.Green → r < 4 ∧ c < 6) ∧
  (∀ r c, board ⟨r, c⟩ = some PegColor.Blue → r < 3 ∧ c < 6) ∧
  (∀ r c, board ⟨r, c⟩ = some PegColor.Orange → r < 2 ∧ c < 6) ∧
  (∀ r, ∃! c, board ⟨r, c⟩ = some PegColor.Yellow) ∧
  (∀ r, r < 5 → ∃! c, board ⟨r, c⟩ = some PegColor.Red) ∧
  (∀ r, r < 4 → ∃! c, board ⟨r, c⟩ = some PegColor.Green) ∧
  (∀ r, r < 3 → ∃! c, board ⟨r, c⟩ = some PegColor.Blue) ∧
  (∀ r, r < 2 → ∃! c, board ⟨r, c⟩ = some PegColor.Orange) ∧
  (∀ c, ∃! r, board ⟨r, c⟩ = some PegColor.Yellow) ∧
  (∀ c, ∃! r, board ⟨r, c⟩ = some PegColor.Red) ∧
  (∀ c, ∃! r, board ⟨r, c⟩ = some PegColor.Green) ∧
  (∀ c, ∃! r, board ⟨r, c⟩ = some PegColor.Blue) ∧
  (∀ c, ∃! r, board ⟨r, c⟩ = some PegColor.Orange)

theorem unique_peg_arrangement :
  ∃! board : Board, is_valid_arrangement board :=
sorry

end NUMINAMATH_CALUDE_unique_peg_arrangement_l1474_147458


namespace NUMINAMATH_CALUDE_inscribed_circle_total_area_l1474_147415

/-- The total area of a figure consisting of a circle inscribed in a square, 
    where the circle has a diameter of 6 meters. -/
theorem inscribed_circle_total_area :
  let circle_diameter : ℝ := 6
  let square_side : ℝ := circle_diameter
  let circle_radius : ℝ := circle_diameter / 2
  let circle_area : ℝ := π * circle_radius ^ 2
  let square_area : ℝ := square_side ^ 2
  let total_area : ℝ := circle_area + square_area
  total_area = 36 + 9 * π := by
  sorry

end NUMINAMATH_CALUDE_inscribed_circle_total_area_l1474_147415


namespace NUMINAMATH_CALUDE_pollen_grain_diameter_scientific_notation_l1474_147435

theorem pollen_grain_diameter_scientific_notation :
  ∃ (a : ℝ) (n : ℤ), 0.0000065 = a * (10 : ℝ) ^ n ∧ 1 ≤ a ∧ a < 10 ∧ a = 6.5 ∧ n = -6 := by
  sorry

end NUMINAMATH_CALUDE_pollen_grain_diameter_scientific_notation_l1474_147435


namespace NUMINAMATH_CALUDE_binary_addition_subtraction_l1474_147491

theorem binary_addition_subtraction : 
  let a : ℕ := 0b1101
  let b : ℕ := 0b1010
  let c : ℕ := 0b1111
  let d : ℕ := 0b1001
  a + b - c + d = 0b11001 := by sorry

end NUMINAMATH_CALUDE_binary_addition_subtraction_l1474_147491


namespace NUMINAMATH_CALUDE_total_carrots_is_nine_l1474_147442

-- Define the number of carrots grown by Sandy
def sandy_carrots : ℕ := 6

-- Define the number of carrots grown by Sam
def sam_carrots : ℕ := 3

-- Define the total number of carrots
def total_carrots : ℕ := sandy_carrots + sam_carrots

-- Theorem stating that the total number of carrots is 9
theorem total_carrots_is_nine : total_carrots = 9 := by
  sorry

end NUMINAMATH_CALUDE_total_carrots_is_nine_l1474_147442


namespace NUMINAMATH_CALUDE_carls_sandwich_options_l1474_147499

/-- Represents the number of different sandwich combinations Carl can order -/
def carlsSandwichCombinations : Nat :=
  let totalBreads : Nat := 5
  let totalMeats : Nat := 7
  let totalCheeses : Nat := 6
  let totalCombinations : Nat := totalBreads * totalMeats * totalCheeses
  let chickenSwissCombinations : Nat := 1 * 1 * totalBreads
  let ryePepperBaconCombinations : Nat := 1 * 1 * totalCheeses
  let chickenRyeCombinations : Nat := 1 * 1 * (totalCheeses - 1)
  let overlapCombinations : Nat := 1
  totalCombinations - (chickenSwissCombinations + ryePepperBaconCombinations + chickenRyeCombinations - overlapCombinations)

/-- Theorem stating that Carl can order 194 different sandwiches -/
theorem carls_sandwich_options : carlsSandwichCombinations = 194 := by
  sorry

end NUMINAMATH_CALUDE_carls_sandwich_options_l1474_147499


namespace NUMINAMATH_CALUDE_circle_and_line_problem_l1474_147462

-- Define the circle C
def circle_C (x y : ℝ) : Prop :=
  (x - 1)^2 + (y - 1)^2 = 4

-- Define the lines l
def line_l1 (x y : ℝ) : Prop := x = 2
def line_l2 (x y : ℝ) : Prop := 4*x + 3*y = 2

-- Theorem statement
theorem circle_and_line_problem :
  ∃ (center_x center_y : ℝ),
    -- Circle C passes through A(1,3) and B(-1,1)
    circle_C 1 3 ∧ circle_C (-1) 1 ∧
    -- Center of the circle is on the line y = x
    center_y = center_x ∧
    -- Circle equation
    (∀ x y, circle_C x y ↔ (x - center_x)^2 + (y - center_y)^2 = 4) ∧
    -- Line l passes through (2,-2)
    (line_l1 2 (-2) ∨ line_l2 2 (-2)) ∧
    -- Line l intersects circle C with chord length 2√3
    (∃ x1 y1 x2 y2,
      ((line_l1 x1 y1 ∧ line_l1 x2 y2) ∨ (line_l2 x1 y1 ∧ line_l2 x2 y2)) ∧
      circle_C x1 y1 ∧ circle_C x2 y2 ∧
      (x2 - x1)^2 + (y2 - y1)^2 = 12) :=
by
  sorry -- Proof omitted

end NUMINAMATH_CALUDE_circle_and_line_problem_l1474_147462


namespace NUMINAMATH_CALUDE_geometric_series_double_sum_l1474_147448

/-- Given two infinite geometric series with the following properties:
    - First series: first term = 20, second term = 5
    - Second series: first term = 20, second term = 5+n
    - Sum of second series is double the sum of first series
    This theorem proves that n = 7.5 -/
theorem geometric_series_double_sum (n : ℝ) : 
  let a₁ : ℝ := 20
  let r₁ : ℝ := 5 / 20
  let r₂ : ℝ := (5 + n) / 20
  let sum₁ : ℝ := a₁ / (1 - r₁)
  let sum₂ : ℝ := a₁ / (1 - r₂)
  sum₂ = 2 * sum₁ → n = 7.5 := by
sorry

end NUMINAMATH_CALUDE_geometric_series_double_sum_l1474_147448


namespace NUMINAMATH_CALUDE_red_balls_count_l1474_147478

theorem red_balls_count (total_balls : ℕ) (prob_red : ℚ) : 
  total_balls = 20 → prob_red = 1/4 → (prob_red * total_balls : ℚ) = 5 := by
  sorry

end NUMINAMATH_CALUDE_red_balls_count_l1474_147478


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l1474_147477

theorem complex_fraction_equality : 
  (3 / 11) * ((1 + 1 / 3) * (1 + 1 / (2^2 - 1)) * (1 + 1 / (3^2 - 1)) * 
               (1 + 1 / (4^2 - 1)) * (1 + 1 / (5^2 - 1)))^5 = 9600000/2673 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l1474_147477


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l1474_147470

/-- Given a hyperbola C with equation x²/a² - y²/b² = 1 where a > 0, b > 0,
    and one of its asymptotes is y = √2 x, prove that the eccentricity of C is √3. -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_asymptote : b / a = Real.sqrt 2) : 
  Real.sqrt (1 + b^2 / a^2) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l1474_147470


namespace NUMINAMATH_CALUDE_breakfast_cost_is_30_25_l1474_147403

/-- Represents the menu prices and orders for a breakfast at a cafe. -/
structure BreakfastOrder where
  toast_price : ℝ
  egg_price : ℝ
  coffee_price : ℝ
  juice_price : ℝ
  dale_toast : ℕ
  dale_eggs : ℕ
  dale_coffee : ℕ
  andrew_toast : ℕ
  andrew_eggs : ℕ
  andrew_juice : ℕ
  melanie_toast : ℕ
  melanie_eggs : ℕ
  melanie_juice : ℕ
  service_charge_rate : ℝ

/-- Calculates the total cost of a breakfast order including service charge. -/
def totalCost (order : BreakfastOrder) : ℝ :=
  let subtotal := 
    order.toast_price * (order.dale_toast + order.andrew_toast + order.melanie_toast : ℝ) +
    order.egg_price * (order.dale_eggs + order.andrew_eggs + order.melanie_eggs : ℝ) +
    order.coffee_price * (order.dale_coffee : ℝ) +
    order.juice_price * (order.andrew_juice + order.melanie_juice : ℝ)
  subtotal * (1 + order.service_charge_rate)

/-- Theorem stating that the total cost of the given breakfast order is £30.25. -/
theorem breakfast_cost_is_30_25 : 
  let order : BreakfastOrder := {
    toast_price := 1,
    egg_price := 3,
    coffee_price := 2,
    juice_price := 1.5,
    dale_toast := 2,
    dale_eggs := 2,
    dale_coffee := 1,
    andrew_toast := 1,
    andrew_eggs := 2,
    andrew_juice := 1,
    melanie_toast := 3,
    melanie_eggs := 1,
    melanie_juice := 2,
    service_charge_rate := 0.1
  }
  totalCost order = 30.25 := by sorry

end NUMINAMATH_CALUDE_breakfast_cost_is_30_25_l1474_147403


namespace NUMINAMATH_CALUDE_quadratic_always_positive_l1474_147454

theorem quadratic_always_positive (m : ℝ) : 
  (∀ x : ℝ, x^2 + 2*x + m > 0) ↔ m > 1 :=
sorry

end NUMINAMATH_CALUDE_quadratic_always_positive_l1474_147454


namespace NUMINAMATH_CALUDE_absolute_value_fraction_inequality_l1474_147445

theorem absolute_value_fraction_inequality (x : ℝ) :
  x ≠ 0 → (|(x + 2) / x| < 1 ↔ x < -1) :=
by sorry

end NUMINAMATH_CALUDE_absolute_value_fraction_inequality_l1474_147445


namespace NUMINAMATH_CALUDE_max_distance_and_total_travel_l1474_147473

/-- Represents a car in the problem -/
structure Car where
  fuelCapacity : ℕ
  fuelEfficiency : ℕ

/-- Represents the problem setup -/
structure ProblemSetup where
  car : Car
  numCars : ℕ

/-- Defines the problem parameters -/
def problem : ProblemSetup :=
  { car := { fuelCapacity := 24, fuelEfficiency := 60 },
    numCars := 2 }

/-- Theorem stating the maximum distance and total distance traveled -/
theorem max_distance_and_total_travel (p : ProblemSetup)
  (h1 : p.numCars = 2)
  (h2 : p.car.fuelCapacity = 24)
  (h3 : p.car.fuelEfficiency = 60) :
  ∃ (maxDistance totalDistance : ℕ),
    maxDistance = 360 ∧
    totalDistance = 2160 ∧
    maxDistance ≤ (p.car.fuelCapacity * p.car.fuelEfficiency) / 2 ∧
    totalDistance = maxDistance * 2 * 3 := by
  sorry

#check max_distance_and_total_travel

end NUMINAMATH_CALUDE_max_distance_and_total_travel_l1474_147473


namespace NUMINAMATH_CALUDE_ring_toss_earnings_l1474_147480

/-- The ring toss game earnings problem -/
theorem ring_toss_earnings (total_earnings : ℕ) (num_days : ℕ) (daily_earnings : ℕ) : 
  total_earnings = 165 → num_days = 5 → total_earnings = num_days * daily_earnings → daily_earnings = 33 := by
  sorry

end NUMINAMATH_CALUDE_ring_toss_earnings_l1474_147480


namespace NUMINAMATH_CALUDE_tan70_cos10_expression_equals_one_l1474_147410

theorem tan70_cos10_expression_equals_one :
  Real.tan (70 * π / 180) * Real.cos (10 * π / 180) * (1 - Real.sqrt 3 * Real.tan (20 * π / 180)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_tan70_cos10_expression_equals_one_l1474_147410


namespace NUMINAMATH_CALUDE_ferris_wheel_capacity_l1474_147401

/-- The number of seats on the Ferris wheel -/
def num_seats : ℕ := 14

/-- The total number of people the Ferris wheel can hold -/
def total_people : ℕ := 84

/-- The number of people each seat can hold -/
def people_per_seat : ℕ := total_people / num_seats

theorem ferris_wheel_capacity : people_per_seat = 6 := by
  sorry

end NUMINAMATH_CALUDE_ferris_wheel_capacity_l1474_147401


namespace NUMINAMATH_CALUDE_different_parrot_extra_toes_l1474_147441

/-- Represents the nail trimming scenario for Cassie's pets -/
structure PetNails where
  num_dogs : Nat
  num_parrots : Nat
  dog_nails_per_foot : Nat
  dog_feet : Nat
  parrot_claws_per_leg : Nat
  parrot_legs : Nat
  total_nails_to_cut : Nat

/-- Calculates the number of extra toes on the different parrot -/
def extra_toes (p : PetNails) : Nat :=
  let standard_dog_nails := p.num_dogs * p.dog_nails_per_foot * p.dog_feet
  let standard_parrot_claws := (p.num_parrots - 1) * p.parrot_claws_per_leg * p.parrot_legs
  let standard_nails := standard_dog_nails + standard_parrot_claws
  p.total_nails_to_cut - standard_nails - (p.parrot_claws_per_leg * p.parrot_legs)

/-- Theorem stating that the number of extra toes on the different parrot is 7 -/
theorem different_parrot_extra_toes :
  ∃ (p : PetNails), 
    p.num_dogs = 4 ∧ 
    p.num_parrots = 8 ∧ 
    p.dog_nails_per_foot = 4 ∧ 
    p.dog_feet = 4 ∧ 
    p.parrot_claws_per_leg = 3 ∧ 
    p.parrot_legs = 2 ∧ 
    p.total_nails_to_cut = 113 ∧ 
    extra_toes p = 7 := by
  sorry

end NUMINAMATH_CALUDE_different_parrot_extra_toes_l1474_147441


namespace NUMINAMATH_CALUDE_section_point_representation_l1474_147497

-- Define the space
variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

-- Define the points
variable (C D Q : V)

-- Define the condition that Q is on line segment CD with ratio 4:1
def is_section_point (C D Q : V) : Prop :=
  ∃ (t : ℝ), t ∈ Set.Icc (0 : ℝ) (1 : ℝ) ∧ Q = (1 - t) • C + t • D ∧ (1 - t) / t = 4

-- The theorem
theorem section_point_representation (h : is_section_point C D Q) :
  Q = (1/5 : ℝ) • C + (4/5 : ℝ) • D :=
sorry

end NUMINAMATH_CALUDE_section_point_representation_l1474_147497


namespace NUMINAMATH_CALUDE_expression_evaluation_l1474_147432

theorem expression_evaluation : 18 * (150 / 3 + 36 / 6 + 16 / 32 + 2) = 1053 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1474_147432


namespace NUMINAMATH_CALUDE_percent_of_a_is_4b_l1474_147423

theorem percent_of_a_is_4b (a b : ℝ) (h : a = 1.2 * b) : (4 * b) / a = 10/3 := by
  sorry

end NUMINAMATH_CALUDE_percent_of_a_is_4b_l1474_147423


namespace NUMINAMATH_CALUDE_max_gcd_bn_l1474_147488

def b (n : ℕ) : ℚ := (15^n - 1) / 14

theorem max_gcd_bn (n : ℕ) : Nat.gcd (Nat.floor (b n)) (Nat.floor (b (n + 1))) = 1 := by
  sorry

end NUMINAMATH_CALUDE_max_gcd_bn_l1474_147488


namespace NUMINAMATH_CALUDE_trapezoid_perimeter_l1474_147413

/-- Represents a trapezoid EFGH with specific properties -/
structure Trapezoid where
  EF : ℝ
  FG : ℝ
  GH : ℝ
  EH : ℝ
  ef_eq_gh : EF = GH
  fg_eq_10 : FG = 10
  eh_eq_20 : EH = 20
  right_triangle : EF^2 = 5^2 + 5^2

/-- The perimeter of the trapezoid EFGH is 30 + 10√2 -/
theorem trapezoid_perimeter (t : Trapezoid) : 
  t.EF + t.FG + t.GH + t.EH = 30 + 10 * Real.sqrt 2 := by
  sorry

#check trapezoid_perimeter

end NUMINAMATH_CALUDE_trapezoid_perimeter_l1474_147413


namespace NUMINAMATH_CALUDE_complex_division_simplification_l1474_147452

theorem complex_division_simplification :
  (2 - Complex.I) / (3 + 4 * Complex.I) = 2/25 - 11/25 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_division_simplification_l1474_147452


namespace NUMINAMATH_CALUDE_sqrt_sin_identity_l1474_147422

theorem sqrt_sin_identity : Real.sqrt (1 - Real.sin 2) + Real.sqrt (1 + Real.sin 2) = 2 * Real.sin 1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sin_identity_l1474_147422


namespace NUMINAMATH_CALUDE_basic_computer_price_l1474_147498

theorem basic_computer_price
  (total_price : ℝ)
  (price_difference : ℝ)
  (printer_ratio : ℝ)
  (h1 : total_price = 2500)
  (h2 : price_difference = 500)
  (h3 : printer_ratio = 1/4)
  : ∃ (basic_computer_price printer_price : ℝ),
    basic_computer_price + printer_price = total_price ∧
    printer_price = printer_ratio * (basic_computer_price + price_difference + printer_price) ∧
    basic_computer_price = 1750 :=
by sorry

end NUMINAMATH_CALUDE_basic_computer_price_l1474_147498


namespace NUMINAMATH_CALUDE_horse_catches_dog_l1474_147411

/-- Represents the relative speed and step distance of animals -/
structure AnimalData where
  steps_per_time_unit : ℕ
  distance_per_steps : ℕ

/-- Calculates the distance an animal covers in one time unit -/
def speed (a : AnimalData) : ℕ := a.steps_per_time_unit * a.distance_per_steps

theorem horse_catches_dog (dog : AnimalData) (horse : AnimalData) 
  (h1 : dog.steps_per_time_unit = 5)
  (h2 : horse.steps_per_time_unit = 3)
  (h3 : 4 * horse.distance_per_steps = 7 * dog.distance_per_steps)
  (initial_distance : ℕ)
  (h4 : initial_distance = 30) :
  (speed horse - speed dog) * 600 = initial_distance * (speed horse) :=
sorry

end NUMINAMATH_CALUDE_horse_catches_dog_l1474_147411


namespace NUMINAMATH_CALUDE_mobile_phone_purchase_price_l1474_147430

/-- The purchase price of the refrigerator in rupees -/
def refrigerator_price : ℝ := 15000

/-- The loss percentage on the refrigerator sale -/
def refrigerator_loss_percent : ℝ := 3

/-- The profit percentage on the mobile phone sale -/
def mobile_profit_percent : ℝ := 10

/-- The overall profit in rupees -/
def overall_profit : ℝ := 350

/-- The purchase price of the mobile phone in rupees -/
def mobile_price : ℝ := 8000

theorem mobile_phone_purchase_price :
  ∃ (x : ℝ),
    x = mobile_price ∧
    refrigerator_price * (1 - refrigerator_loss_percent / 100) +
    x * (1 + mobile_profit_percent / 100) =
    refrigerator_price + x + overall_profit :=
by sorry

end NUMINAMATH_CALUDE_mobile_phone_purchase_price_l1474_147430


namespace NUMINAMATH_CALUDE_remainder_problem_l1474_147482

theorem remainder_problem (m n : ℕ) (h1 : m % n = 2) (h2 : (3 * m) % n = 1) : n = 5 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l1474_147482


namespace NUMINAMATH_CALUDE_xyz_product_magnitude_l1474_147449

theorem xyz_product_magnitude (x y z : ℝ) 
  (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (hdistinct : x ≠ y ∧ y ≠ z ∧ z ≠ x)
  (heq : x + 1/y = y + 1/z ∧ y + 1/z = z + 1/x) : 
  |x * y * z| = 1 := by
sorry

end NUMINAMATH_CALUDE_xyz_product_magnitude_l1474_147449


namespace NUMINAMATH_CALUDE_smallest_value_of_floor_sum_l1474_147453

theorem smallest_value_of_floor_sum (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  ⌊(a + b) / c⌋ + ⌊(b + c) / a⌋ + ⌊(c + a) / b⌋ ≥ 4 :=
by sorry

end NUMINAMATH_CALUDE_smallest_value_of_floor_sum_l1474_147453


namespace NUMINAMATH_CALUDE_expansion_without_x_squared_l1474_147465

theorem expansion_without_x_squared (n : ℕ+) (h1 : 5 ≤ n) (h2 : n ≤ 8) :
  (∀ (r : ℕ), r ≤ n → n - 4 * r ≠ 0 ∧ n - 4 * r ≠ 1 ∧ n - 4 * r ≠ 2) ↔ n = 7 := by
  sorry

end NUMINAMATH_CALUDE_expansion_without_x_squared_l1474_147465
