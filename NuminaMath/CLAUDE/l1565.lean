import Mathlib

namespace NUMINAMATH_CALUDE_geometric_sequence_implies_c_equals_six_l1565_156578

/-- A function f(x) = x^2 + x + c where f(1), f(2), and f(3) form a geometric sequence. -/
def f (c : ℝ) (x : ℝ) : ℝ := x^2 + x + c

/-- The theorem stating that if f(1), f(2), and f(3) form a geometric sequence, then c = 6. -/
theorem geometric_sequence_implies_c_equals_six (c : ℝ) :
  (∃ r : ℝ, f c 2 = f c 1 * r ∧ f c 3 = f c 2 * r) → c = 6 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_implies_c_equals_six_l1565_156578


namespace NUMINAMATH_CALUDE_best_of_three_win_probability_l1565_156567

/-- The probability of winning a single set -/
def p : ℝ := 0.6

/-- The probability of winning a best-of-three match given the probability of winning each set -/
def win_probability (p : ℝ) : ℝ := p^2 + 3 * p^2 * (1 - p)

/-- Theorem: The probability of winning a best-of-three match when p = 0.6 is 0.648 -/
theorem best_of_three_win_probability :
  win_probability p = 0.648 := by sorry

end NUMINAMATH_CALUDE_best_of_three_win_probability_l1565_156567


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_l1565_156507

theorem sufficient_but_not_necessary (x : ℝ) : 
  (x < -1 → 2*x^2 + x - 1 > 0) ∧ 
  ¬(2*x^2 + x - 1 > 0 → x < -1) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_l1565_156507


namespace NUMINAMATH_CALUDE_inequality_proof_l1565_156510

theorem inequality_proof (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (1 / (a * b) + 1 / (a * c) + 1 / (a * d) + 1 / (b * c) + 1 / (b * d) + 1 / (c * d)) ≤ 
  3 / 8 * (1 / a + 1 / b + 1 / c + 1 / d)^2 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l1565_156510


namespace NUMINAMATH_CALUDE_aston_comic_pages_l1565_156539

/-- The number of pages in each comic -/
def pages_per_comic : ℕ := 25

/-- The number of untorn comics initially in the box -/
def initial_comics : ℕ := 5

/-- The total number of comics in the box after Aston put them back together -/
def final_comics : ℕ := 11

/-- The number of pages Aston found on the floor -/
def pages_found : ℕ := (final_comics - initial_comics) * pages_per_comic

theorem aston_comic_pages : pages_found = 150 := by
  sorry

end NUMINAMATH_CALUDE_aston_comic_pages_l1565_156539


namespace NUMINAMATH_CALUDE_average_price_per_book_l1565_156561

def books_shop1 : ℕ := 65
def price_shop1 : ℕ := 1150
def books_shop2 : ℕ := 50
def price_shop2 : ℕ := 920

theorem average_price_per_book :
  (price_shop1 + price_shop2) / (books_shop1 + books_shop2) = 18 := by
  sorry

end NUMINAMATH_CALUDE_average_price_per_book_l1565_156561


namespace NUMINAMATH_CALUDE_inequality_solution_l1565_156591

theorem inequality_solution (x : ℝ) :
  (x * (x + 1)) / ((x - 5)^2) ≥ 15 ↔ (x > Real.sqrt (151 - Real.sqrt 1801) / 2 ∧ x < 5) ∨
                                    (x > 5 ∧ x < Real.sqrt (151 + Real.sqrt 1801) / 2) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l1565_156591


namespace NUMINAMATH_CALUDE_books_from_second_shop_l1565_156529

theorem books_from_second_shop 
  (first_shop_books : ℕ)
  (first_shop_cost : ℕ)
  (second_shop_cost : ℕ)
  (average_price : ℕ)
  (h1 : first_shop_books = 55)
  (h2 : first_shop_cost = 1500)
  (h3 : second_shop_cost = 340)
  (h4 : average_price = 16) :
  ∃ (second_shop_books : ℕ),
    (first_shop_cost + second_shop_cost) = 
    average_price * (first_shop_books + second_shop_books) ∧
    second_shop_books = 60 := by
  sorry

end NUMINAMATH_CALUDE_books_from_second_shop_l1565_156529


namespace NUMINAMATH_CALUDE_inverse_function_symmetry_l1565_156526

-- Define a function and its inverse
variable (f : ℝ → ℝ) (f_inv : ℝ → ℝ)

-- State that f_inv is the inverse of f
axiom inverse_relation : ∀ x, f (f_inv x) = x ∧ f_inv (f x) = x

-- Define symmetry about the line x - y = 0
def symmetric_about_x_eq_y (f g : ℝ → ℝ) : Prop :=
  ∀ x y, f x = y ↔ g y = x

-- Theorem statement
theorem inverse_function_symmetry :
  symmetric_about_x_eq_y f f_inv :=
sorry

end NUMINAMATH_CALUDE_inverse_function_symmetry_l1565_156526


namespace NUMINAMATH_CALUDE_vector_magnitude_relation_l1565_156542

variables {V : Type*} [NormedAddCommGroup V] [NormedSpace ℝ V]

theorem vector_magnitude_relation (a b : V) (ha : a ≠ 0) (hb : b ≠ 0) (h : a = -3 • b) :
  ‖a‖ = 3 * ‖b‖ :=
by sorry

end NUMINAMATH_CALUDE_vector_magnitude_relation_l1565_156542


namespace NUMINAMATH_CALUDE_profit_is_24000_l1565_156550

def initial_value : ℝ := 150000
def depreciation_rate : ℝ := 0.22
def selling_price : ℝ := 115260
def years : ℕ := 2

def value_after_years (initial : ℝ) (rate : ℝ) (years : ℕ) : ℝ :=
  initial * (1 - rate) ^ years

def profit (initial : ℝ) (rate : ℝ) (years : ℕ) (selling_price : ℝ) : ℝ :=
  selling_price - value_after_years initial rate years

theorem profit_is_24000 :
  profit initial_value depreciation_rate years selling_price = 24000 := by
  sorry

end NUMINAMATH_CALUDE_profit_is_24000_l1565_156550


namespace NUMINAMATH_CALUDE_square_wire_length_l1565_156523

theorem square_wire_length (area : ℝ) (side_length : ℝ) (wire_length : ℝ) : 
  area = 324 → 
  area = side_length ^ 2 → 
  wire_length = 4 * side_length → 
  wire_length = 72 := by
sorry

end NUMINAMATH_CALUDE_square_wire_length_l1565_156523


namespace NUMINAMATH_CALUDE_magazine_budget_cut_percentage_l1565_156575

def original_budget : ℝ := 940
def desired_reduction : ℝ := 658

theorem magazine_budget_cut_percentage :
  (original_budget - desired_reduction) / original_budget * 100 = 30 := by
  sorry

end NUMINAMATH_CALUDE_magazine_budget_cut_percentage_l1565_156575


namespace NUMINAMATH_CALUDE_max_value_theorem_l1565_156576

theorem max_value_theorem (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h4 : a + b + c = 3) :
  ∀ x y z : ℝ, 0 ≤ x ∧ 0 ≤ y ∧ 0 ≤ z ∧ x + y + z = 3 → x + y^2 + z^4 ≤ 3 :=
by sorry

end NUMINAMATH_CALUDE_max_value_theorem_l1565_156576


namespace NUMINAMATH_CALUDE_vector_addition_l1565_156540

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

/-- The sum of two vectors is equal to the vector from the start of the first to the end of the second. -/
theorem vector_addition (a b : V) :
  ∃ c : V, a + b = c ∧ ∃ (x y : V), x + a = y ∧ y + b = x + c :=
sorry

end NUMINAMATH_CALUDE_vector_addition_l1565_156540


namespace NUMINAMATH_CALUDE_power_division_l1565_156544

theorem power_division (a : ℝ) : a^5 / a^2 = a^3 := by
  sorry

end NUMINAMATH_CALUDE_power_division_l1565_156544


namespace NUMINAMATH_CALUDE_line_intercepts_sum_l1565_156587

/-- Given a line with equation y = -1/2 + 3x, prove that the sum of its x-intercept and y-intercept is -1/3. -/
theorem line_intercepts_sum (x y : ℝ) : 
  y = -1/2 + 3*x → -- Line equation
  ∃ (x_int y_int : ℝ),
    (0 = -1/2 + 3*x_int) ∧  -- x-intercept
    (y_int = -1/2 + 3*0) ∧  -- y-intercept
    (x_int + y_int = -1/3) := by
  sorry


end NUMINAMATH_CALUDE_line_intercepts_sum_l1565_156587


namespace NUMINAMATH_CALUDE_triangle_inequalities_l1565_156592

/-- Given four collinear points E, F, G, H in order, with EF = a, EG = b, EH = c,
    if EF and GH are rotated to form a triangle with positive area,
    then a < c/3 and b < a + c/3 must be true, while b < c/3 is not necessarily true. -/
theorem triangle_inequalities (a b c : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (hab : a < b) (hbc : b < c) 
  (h_triangle : a + (c - b) > b - a) : 
  (a < c / 3 ∧ b < a + c / 3) ∧ ¬(b < c / 3 → True) := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequalities_l1565_156592


namespace NUMINAMATH_CALUDE_max_female_to_male_ratio_l1565_156500

/-- Proves that the maximum ratio of female to male students is 4:1 given the problem conditions -/
theorem max_female_to_male_ratio :
  ∀ (female_count male_count bench_count : ℕ),
  male_count = 29 →
  bench_count = 29 →
  ∃ (x : ℕ), female_count = x * male_count →
  female_count + male_count ≤ bench_count * 5 →
  x ≤ 4 :=
by sorry

end NUMINAMATH_CALUDE_max_female_to_male_ratio_l1565_156500


namespace NUMINAMATH_CALUDE_units_digit_of_2_pow_2015_l1565_156504

theorem units_digit_of_2_pow_2015 (h : ∀ n : ℕ, n > 0 → (2^n : ℕ) % 10 = (2^(n % 4) : ℕ) % 10) :
  (2^2015 : ℕ) % 10 = 8 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_2_pow_2015_l1565_156504


namespace NUMINAMATH_CALUDE_steves_book_earnings_l1565_156512

/-- The amount Steve gets for each copy of the book sold -/
def amount_per_copy : ℝ := 2

theorem steves_book_earnings :
  let total_copies : ℕ := 1000000
  let advance_copies : ℕ := 100000
  let agent_percentage : ℝ := 0.1
  let earnings_after_advance : ℝ := 1620000
  
  (total_copies - advance_copies : ℝ) * (1 - agent_percentage) * amount_per_copy = earnings_after_advance :=
by
  sorry

#check steves_book_earnings

end NUMINAMATH_CALUDE_steves_book_earnings_l1565_156512


namespace NUMINAMATH_CALUDE_reciprocal_equals_self_l1565_156536

theorem reciprocal_equals_self (x : ℝ) : x ≠ 0 ∧ x = 1 / x → x = 1 ∨ x = -1 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_equals_self_l1565_156536


namespace NUMINAMATH_CALUDE_min_value_of_expression_l1565_156597

/-- The minimum value of 1/a + 4/b given the conditions -/
theorem min_value_of_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (∀ x y : ℝ, a * x - b * y + 2 = 0 → x^2 + y^2 + 2*x - 2*y = 0) → 
  (∃ A B : ℝ × ℝ, A ≠ B ∧ 
    (a * A.1 - b * A.2 + 2 = 0) ∧ (A.1^2 + A.2^2 + 2*A.1 - 2*A.2 = 0) ∧
    (a * B.1 - b * B.2 + 2 = 0) ∧ (B.1^2 + B.2^2 + 2*B.1 - 2*B.2 = 0) ∧
    (∀ C D : ℝ × ℝ, C ≠ D → 
      (a * C.1 - b * C.2 + 2 = 0) → (C.1^2 + C.2^2 + 2*C.1 - 2*C.2 = 0) →
      (a * D.1 - b * D.2 + 2 = 0) → (D.1^2 + D.2^2 + 2*D.1 - 2*D.2 = 0) →
      (A.1 - B.1)^2 + (A.2 - B.2)^2 ≥ (C.1 - D.1)^2 + (C.2 - D.2)^2)) →
  (1/a + 4/b) ≥ 9/2 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l1565_156597


namespace NUMINAMATH_CALUDE_quiz_answer_key_count_l1565_156562

/-- The number of ways to arrange true and false answers for 6 questions 
    with an equal number of true and false answers -/
def true_false_arrangements : ℕ := Nat.choose 6 3

/-- The number of ways to choose answers for 4 multiple-choice questions 
    with 5 options each -/
def multiple_choice_arrangements : ℕ := 5^4

/-- The total number of ways to create an answer key for the quiz -/
def total_arrangements : ℕ := true_false_arrangements * multiple_choice_arrangements

theorem quiz_answer_key_count : total_arrangements = 12500 := by
  sorry

end NUMINAMATH_CALUDE_quiz_answer_key_count_l1565_156562


namespace NUMINAMATH_CALUDE_locus_of_centers_is_hyperbola_l1565_156515

/-- A circle with center (x, y) and radius R that touches the diameter of circle k -/
structure TouchingCircle where
  x : ℝ
  y : ℝ
  R : ℝ
  touches_diameter : (-r : ℝ) ≤ x ∧ x ≤ r
  non_negative_y : y ≥ 0
  tangent_to_diameter : R = y

/-- The locus of centers of circles touching the diameter of k and with closest point at distance R from k -/
def locus_of_centers (r : ℝ) (c : TouchingCircle) : Prop :=
  (c.y - 2*r/3)^2 / (r/3)^2 - c.x^2 / (r/Real.sqrt 3)^2 = 1

theorem locus_of_centers_is_hyperbola (r : ℝ) (h : r > 0) :
  ∀ c : TouchingCircle, locus_of_centers r c ↔ 
    c.R = 2 * c.y ∧ 
    Real.sqrt (c.x^2 + c.y^2) = r - 2 * c.y ∧
    r ≥ 3 * c.y :=
  sorry

end NUMINAMATH_CALUDE_locus_of_centers_is_hyperbola_l1565_156515


namespace NUMINAMATH_CALUDE_num_boys_is_three_l1565_156560

/-- The number of boys sitting at the table -/
def num_boys : ℕ := sorry

/-- The number of girls sitting at the table -/
def num_girls : ℕ := 5

/-- The total number of buns on the plate -/
def total_buns : ℕ := 30

/-- The number of buns given by girls to boys they know -/
def buns_girls_to_boys : ℕ := num_girls * num_boys

/-- The number of buns given by boys to girls they don't know -/
def buns_boys_to_girls : ℕ := num_boys * num_girls

/-- Theorem stating that the number of boys is 3 -/
theorem num_boys_is_three : num_boys = 3 :=
  by
    have h1 : buns_girls_to_boys + buns_boys_to_girls = total_buns := sorry
    have h2 : num_girls * num_boys + num_boys * num_girls = total_buns := sorry
    have h3 : 2 * (num_girls * num_boys) = total_buns := sorry
    have h4 : 2 * (5 * num_boys) = 30 := sorry
    have h5 : 10 * num_boys = 30 := sorry
    sorry

end NUMINAMATH_CALUDE_num_boys_is_three_l1565_156560


namespace NUMINAMATH_CALUDE_square_equation_solutions_cubic_equation_solution_l1565_156505

-- Part 1
theorem square_equation_solutions (x : ℝ) :
  (x - 2)^2 = 9 ↔ x = 5 ∨ x = -1 := by sorry

-- Part 2
theorem cubic_equation_solution (x : ℝ) :
  27 * (x + 1)^3 + 8 = 0 ↔ x = -5/3 := by sorry

end NUMINAMATH_CALUDE_square_equation_solutions_cubic_equation_solution_l1565_156505


namespace NUMINAMATH_CALUDE_dean_transactions_l1565_156518

theorem dean_transactions (mabel anthony cal jade dean : ℕ) : 
  mabel = 90 →
  anthony = mabel + (mabel / 10) →
  cal = (2 * anthony) / 3 →
  jade = cal + 14 →
  dean = jade + (jade / 4) →
  dean = 100 := by
sorry

end NUMINAMATH_CALUDE_dean_transactions_l1565_156518


namespace NUMINAMATH_CALUDE_certain_number_divisibility_l1565_156503

theorem certain_number_divisibility (z x : ℕ) (h1 : z > 0) (h2 : 4 ∣ z) : 
  (z + x + 4 + z + 3) % 2 = 1 ↔ Even x := by sorry

end NUMINAMATH_CALUDE_certain_number_divisibility_l1565_156503


namespace NUMINAMATH_CALUDE_career_preference_theorem_l1565_156594

/-- Represents the ratio of boys to girls in a class -/
def boy_girl_ratio : ℚ := 2 / 3

/-- Represents the fraction of boys who prefer the career -/
def boy_preference : ℚ := 1 / 3

/-- Represents the fraction of girls who prefer the career -/
def girl_preference : ℚ := 2 / 3

/-- Calculates the degrees in a circle graph for a given career preference -/
def career_preference_degrees (ratio : ℚ) (boy_pref : ℚ) (girl_pref : ℚ) : ℚ :=
  360 * ((ratio * boy_pref + girl_pref) / (ratio + 1))

/-- Theorem stating that the career preference degrees is 192 -/
theorem career_preference_theorem :
  career_preference_degrees boy_girl_ratio boy_preference girl_preference = 192 := by
  sorry

#eval career_preference_degrees boy_girl_ratio boy_preference girl_preference

end NUMINAMATH_CALUDE_career_preference_theorem_l1565_156594


namespace NUMINAMATH_CALUDE_exactly_one_correct_l1565_156586

-- Define the four propositions
def proposition1 : Prop := ∀ (p q : Prop), (p ∧ q → p ∨ q) ∧ ¬(p ∨ q → p ∧ q)

def proposition2 : Prop :=
  let p := ∃ x : ℝ, x^2 + 2*x ≤ 0
  ¬p ↔ ∀ x : ℝ, x^2 + 2*x > 0

def proposition3 : Prop :=
  ¬(∀ x : ℝ, x^2 - 2*x + 3 > 0) ↔ (∃ x : ℝ, x^2 - 2*x + 3 < 0)

def proposition4 : Prop :=
  ∀ (p q : Prop), (¬p → q) ↔ (p → ¬q)

-- Theorem stating that exactly one proposition is correct
theorem exactly_one_correct : 
  (proposition2 ∧ ¬proposition1 ∧ ¬proposition3 ∧ ¬proposition4) :=
sorry

end NUMINAMATH_CALUDE_exactly_one_correct_l1565_156586


namespace NUMINAMATH_CALUDE_math_team_selection_l1565_156528

theorem math_team_selection (boys girls : ℕ) (h1 : boys = 10) (h2 : girls = 12) :
  (Nat.choose boys 5) * (Nat.choose girls 3) = 55440 := by
  sorry

end NUMINAMATH_CALUDE_math_team_selection_l1565_156528


namespace NUMINAMATH_CALUDE_andy_distance_to_market_l1565_156525

/-- The distance between Andy's house and the market -/
def distance_to_market (distance_to_school : ℕ) (total_distance : ℕ) : ℕ :=
  total_distance - 2 * distance_to_school

theorem andy_distance_to_market :
  let distance_to_school : ℕ := 50
  let total_distance : ℕ := 140
  distance_to_market distance_to_school total_distance = 40 := by
  sorry

end NUMINAMATH_CALUDE_andy_distance_to_market_l1565_156525


namespace NUMINAMATH_CALUDE_point_count_on_curve_l1565_156532

theorem point_count_on_curve : 
  ∃! (points : Finset (ℤ × ℤ)), 
    points.card = 6 ∧ 
    ∀ p : ℤ × ℤ, p ∈ points ↔ 
      let m := p.1
      let n := p.2
      n^2 = (m^2 - 4) * (m^2 + 12*m + 32) + 4 := by
  sorry

end NUMINAMATH_CALUDE_point_count_on_curve_l1565_156532


namespace NUMINAMATH_CALUDE_shaded_area_of_square_l1565_156519

/-- Given a square composed of 25 congruent smaller squares with a diagonal of 10 cm,
    prove that its area is 50 square cm. -/
theorem shaded_area_of_square (d : ℝ) (n : ℕ) (h1 : d = 10) (h2 : n = 25) :
  (d^2 / 2 : ℝ) = 50 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_of_square_l1565_156519


namespace NUMINAMATH_CALUDE_pizza_coverage_theorem_l1565_156595

/-- Represents the properties of a pizza with pepperoni -/
structure PepperoniPizza where
  pizza_diameter : ℝ
  pepperoni_across : ℕ
  total_pepperoni : ℕ

/-- Calculates the fraction of pizza covered by pepperoni -/
def pepperoni_coverage (p : PepperoniPizza) : ℚ :=
  sorry

/-- Theorem stating the fraction of pizza covered by pepperoni for the given conditions -/
theorem pizza_coverage_theorem (p : PepperoniPizza) 
  (h1 : p.pizza_diameter = 18)
  (h2 : p.pepperoni_across = 8)
  (h3 : p.total_pepperoni = 36) :
  pepperoni_coverage p = 9/16 := by
  sorry

end NUMINAMATH_CALUDE_pizza_coverage_theorem_l1565_156595


namespace NUMINAMATH_CALUDE_rem_neg_five_sixths_three_fourths_l1565_156513

def rem (x y : ℚ) : ℚ := x - y * ⌊x / y⌋

theorem rem_neg_five_sixths_three_fourths :
  rem (-5/6) (3/4) = 2/3 := by sorry

end NUMINAMATH_CALUDE_rem_neg_five_sixths_three_fourths_l1565_156513


namespace NUMINAMATH_CALUDE_distance_AB_l1565_156548

def A : ℝ × ℝ := (3, 2)
def B : ℝ × ℝ := (-1, 6)

theorem distance_AB : Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 4 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_distance_AB_l1565_156548


namespace NUMINAMATH_CALUDE_noon_temperature_l1565_156564

theorem noon_temperature 
  (morning_temp : ℤ) 
  (temp_drop : ℤ) 
  (h1 : morning_temp = 3) 
  (h2 : temp_drop = 9) : 
  morning_temp - temp_drop = -6 := by
sorry

end NUMINAMATH_CALUDE_noon_temperature_l1565_156564


namespace NUMINAMATH_CALUDE_sqrt_245_simplification_l1565_156502

theorem sqrt_245_simplification : Real.sqrt 245 = 7 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_245_simplification_l1565_156502


namespace NUMINAMATH_CALUDE_min_circle_area_l1565_156571

/-- Given a line ax + by = 1 passing through point A(b, a), where O is the origin (0, 0),
    the minimum area of the circle with center O and radius OA is π. -/
theorem min_circle_area (a b : ℝ) (h : a * b = 1 / 2) :
  (π : ℝ) ≤ π * (a^2 + b^2) ∧ ∃ (a₀ b₀ : ℝ), a₀ * b₀ = 1 / 2 ∧ π * (a₀^2 + b₀^2) = π :=
by sorry

end NUMINAMATH_CALUDE_min_circle_area_l1565_156571


namespace NUMINAMATH_CALUDE_other_number_is_three_l1565_156565

theorem other_number_is_three (x y : ℝ) : 
  x + y = 10 → 
  2 * x = 3 * y + 5 → 
  (x = 7 ∨ y = 7) → 
  (x = 3 ∨ y = 3) :=
by sorry

end NUMINAMATH_CALUDE_other_number_is_three_l1565_156565


namespace NUMINAMATH_CALUDE_sphere_radius_touching_cones_l1565_156557

/-- The radius of a sphere touching three cones and a table -/
theorem sphere_radius_touching_cones (r₁ r₂ r₃ : ℝ) (α β γ : ℝ) : 
  r₁ = 1 → 
  r₂ = 12 → 
  r₃ = 12 → 
  α = -4 * Real.arctan (1/3) → 
  β = 4 * Real.arctan (2/3) → 
  γ = 4 * Real.arctan (2/3) → 
  ∃ R : ℝ, R = 40/21 ∧ 
    (∀ x y z : ℝ, 
      x^2 + y^2 + z^2 = R^2 → 
      (∃ t : ℝ, t ≥ 0 ∧ 
        ((x - r₁)^2 + y^2 = (t * Real.tan (α/2))^2 ∧ z = t) ∨
        ((x - (r₁ + r₂))^2 + y^2 = (t * Real.tan (β/2))^2 ∧ z = t) ∨
        (x^2 + (y - (r₂ + r₃))^2 = (t * Real.tan (γ/2))^2 ∧ z = t) ∨
        z = 0)) :=
by
  sorry

end NUMINAMATH_CALUDE_sphere_radius_touching_cones_l1565_156557


namespace NUMINAMATH_CALUDE_rectangular_to_polar_conversion_l1565_156527

theorem rectangular_to_polar_conversion :
  let x : ℝ := 1
  let y : ℝ := -Real.sqrt 3
  let r : ℝ := Real.sqrt (x^2 + y^2)
  let θ : ℝ := 5 * Real.pi / 3
  r > 0 ∧ 0 ≤ θ ∧ θ < 2 * Real.pi ∧ r = 2 ∧ x = r * Real.cos θ ∧ y = r * Real.sin θ :=
by sorry

end NUMINAMATH_CALUDE_rectangular_to_polar_conversion_l1565_156527


namespace NUMINAMATH_CALUDE_odd_function_zero_value_necessary_not_sufficient_l1565_156577

-- Define what it means for a function to be odd
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem odd_function_zero_value_necessary_not_sufficient :
  (∀ f : ℝ → ℝ, IsOdd f → f 0 = 0) ∧
  (∃ g : ℝ → ℝ, g 0 = 0 ∧ ¬IsOdd g) :=
sorry

end NUMINAMATH_CALUDE_odd_function_zero_value_necessary_not_sufficient_l1565_156577


namespace NUMINAMATH_CALUDE_amount_ratio_problem_l1565_156552

theorem amount_ratio_problem (total amount_p amount_q amount_r : ℚ) : 
  total = 1210 →
  amount_p + amount_q + amount_r = total →
  amount_p / amount_q = 5 / 4 →
  amount_r = 400 →
  amount_q / amount_r = 9 / 10 := by
sorry

end NUMINAMATH_CALUDE_amount_ratio_problem_l1565_156552


namespace NUMINAMATH_CALUDE_smallest_constant_is_two_l1565_156506

/-- A function satisfying the given conditions -/
def SpecialFunction (f : ℝ → ℝ) : Prop :=
  (∀ x ∈ Set.Icc 0 1, f x ≥ 0) ∧ 
  (∀ x₁ x₂, x₁ ≥ 0 → x₂ ≥ 0 → x₁ + x₂ ≤ 1 → f (x₁ + x₂) ≥ f x₁ + f x₂) ∧
  f 0 = 0 ∧ f 1 = 1

/-- The theorem stating that 2 is the smallest positive constant c such that f(x) ≤ cx for all x ∈ [0,1] -/
theorem smallest_constant_is_two (f : ℝ → ℝ) (hf : SpecialFunction f) :
  (∀ x ∈ Set.Icc 0 1, f x ≤ 2 * x) ∧
  ∀ c < 2, ∃ g : ℝ → ℝ, SpecialFunction g ∧ ∃ x ∈ Set.Icc 0 1, g x > c * x :=
by sorry

end NUMINAMATH_CALUDE_smallest_constant_is_two_l1565_156506


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l1565_156568

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 + 2*x + 5 > 0) ↔ (∃ x : ℝ, x^2 + 2*x + 5 ≤ 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l1565_156568


namespace NUMINAMATH_CALUDE_min_value_of_S_l1565_156554

def S (x : ℝ) : ℝ := (x - 10)^2 + (x + 5)^2

theorem min_value_of_S :
  ∃ (min : ℝ), min = 112.5 ∧ ∀ (x : ℝ), S x ≥ min :=
sorry

end NUMINAMATH_CALUDE_min_value_of_S_l1565_156554


namespace NUMINAMATH_CALUDE_circular_seating_arrangements_l1565_156546

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def choose (n k : ℕ) : ℕ := 
  factorial n / (factorial k * factorial (n - k))

theorem circular_seating_arrangements :
  let total_people : ℕ := 8
  let seats : ℕ := 7
  let reserved_seats : ℕ := 1
  let remaining_people : ℕ := total_people - reserved_seats
  let people_to_arrange : ℕ := seats - reserved_seats
  
  (choose remaining_people people_to_arrange * factorial people_to_arrange) / seats = 720 := by
sorry

end NUMINAMATH_CALUDE_circular_seating_arrangements_l1565_156546


namespace NUMINAMATH_CALUDE_decimal_expansion_eight_elevenths_repeating_block_size_l1565_156541

/-- The smallest repeating block in the decimal expansion of 8/11 contains 2 digits. -/
theorem decimal_expansion_eight_elevenths_repeating_block_size :
  ∃ (a b : ℕ) (h : b ≠ 0),
    (8 : ℚ) / 11 = (a : ℚ) / (10^b - 1) ∧
    ∀ (c d : ℕ) (h' : d ≠ 0), (8 : ℚ) / 11 = (c : ℚ) / (10^d - 1) → b ≤ d :=
by sorry

end NUMINAMATH_CALUDE_decimal_expansion_eight_elevenths_repeating_block_size_l1565_156541


namespace NUMINAMATH_CALUDE_folded_rectangle_EF_length_l1565_156553

-- Define the rectangle
structure Rectangle :=
  (AB : ℝ)
  (BC : ℝ)

-- Define the folded pentagon
structure FoldedPentagon :=
  (rect : Rectangle)
  (EF : ℝ)

-- Theorem statement
theorem folded_rectangle_EF_length 
  (rect : Rectangle) 
  (pent : FoldedPentagon) : 
  rect.AB = 4 → 
  rect.BC = 8 → 
  pent.rect = rect → 
  pent.EF = 4 := by
sorry

end NUMINAMATH_CALUDE_folded_rectangle_EF_length_l1565_156553


namespace NUMINAMATH_CALUDE_sin_135_degrees_l1565_156570

theorem sin_135_degrees : Real.sin (135 * Real.pi / 180) = Real.sqrt 2 / 2 := by sorry

end NUMINAMATH_CALUDE_sin_135_degrees_l1565_156570


namespace NUMINAMATH_CALUDE_inequality_proof_l1565_156596

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h : a + b + c ≥ a * b * c) : 
  (2/a + 3/b + 6/c ≥ 6 ∧ 2/b + 3/c + 6/a ≥ 6) ∨
  (2/b + 3/c + 6/a ≥ 6 ∧ 2/c + 3/a + 6/b ≥ 6) ∨
  (2/c + 3/a + 6/b ≥ 6 ∧ 2/a + 3/b + 6/c ≥ 6) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1565_156596


namespace NUMINAMATH_CALUDE_sum_of_specific_geometric_series_l1565_156583

def geometric_series_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

theorem sum_of_specific_geometric_series :
  geometric_series_sum (1/4) (1/2) 7 = 127/256 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_specific_geometric_series_l1565_156583


namespace NUMINAMATH_CALUDE_mikaela_savings_l1565_156584

/-- Calculates the total savings for Mikaela over two months of tutoring --/
def total_savings (
  hourly_rate_month1 : ℚ)
  (hours_month1 : ℚ)
  (hourly_rate_month2 : ℚ)
  (additional_hours_month2 : ℚ)
  (spending_ratio_month1 : ℚ)
  (spending_ratio_month2 : ℚ) : ℚ :=
  let earnings_month1 := hourly_rate_month1 * hours_month1
  let savings_month1 := earnings_month1 * (1 - spending_ratio_month1)
  let hours_month2 := hours_month1 + additional_hours_month2
  let earnings_month2 := hourly_rate_month2 * hours_month2
  let savings_month2 := earnings_month2 * (1 - spending_ratio_month2)
  savings_month1 + savings_month2

/-- Proves that Mikaela's total savings from both months is $190 --/
theorem mikaela_savings :
  total_savings 10 35 12 5 (4/5) (3/4) = 190 := by
  sorry

end NUMINAMATH_CALUDE_mikaela_savings_l1565_156584


namespace NUMINAMATH_CALUDE_parallel_vectors_fraction_value_l1565_156511

theorem parallel_vectors_fraction_value (α : ℝ) :
  let a : ℝ × ℝ := (Real.sin α, Real.cos α - 2 * Real.sin α)
  let b : ℝ × ℝ := (1, 2)
  (∃ (k : ℝ), a = k • b) →
  (1 + 2 * Real.sin α * Real.cos α) / (Real.sin α ^ 2 - Real.cos α ^ 2) = -5/3 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_fraction_value_l1565_156511


namespace NUMINAMATH_CALUDE_hash_composition_l1565_156593

-- Define the # operation
def hash (x : ℝ) : ℝ := 8 - x

-- Define the # operation
def hash_prefix (x : ℝ) : ℝ := x - 8

-- Theorem statement
theorem hash_composition : hash_prefix (hash 14) = -14 := by
  sorry

end NUMINAMATH_CALUDE_hash_composition_l1565_156593


namespace NUMINAMATH_CALUDE_polynomial_simplification_l1565_156581

theorem polynomial_simplification (x : ℝ) : 
  5 - 3*x - 7*x^2 + 3 + 12*x - 9*x^2 - 8 + 15*x + 21*x^2 = 5*x^2 + 24*x := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l1565_156581


namespace NUMINAMATH_CALUDE_bret_dinner_coworkers_l1565_156559

theorem bret_dinner_coworkers :
  let main_meal_cost : ℚ := 12
  let appetizer_cost : ℚ := 6
  let num_appetizers : ℕ := 2
  let tip_percentage : ℚ := 0.2
  let rush_order_fee : ℚ := 5
  let total_spent : ℚ := 77

  let total_people (coworkers : ℕ) : ℕ := coworkers + 1
  let main_meals_cost (coworkers : ℕ) : ℚ := main_meal_cost * (total_people coworkers : ℚ)
  let appetizers_total : ℚ := appetizer_cost * (num_appetizers : ℚ)
  let subtotal (coworkers : ℕ) : ℚ := main_meals_cost coworkers + appetizers_total
  let tip (coworkers : ℕ) : ℚ := tip_percentage * subtotal coworkers
  let total_cost (coworkers : ℕ) : ℚ := subtotal coworkers + tip coworkers + rush_order_fee

  ∃ (coworkers : ℕ), total_cost coworkers = total_spent ∧ coworkers = 3 :=
by sorry

end NUMINAMATH_CALUDE_bret_dinner_coworkers_l1565_156559


namespace NUMINAMATH_CALUDE_fish_food_calculation_l1565_156573

/-- The total amount of food Layla needs to give her fish -/
def total_fish_food (goldfish : ℕ) (goldfish_food : ℚ)
                    (swordtails : ℕ) (swordtails_food : ℚ)
                    (guppies : ℕ) (guppies_food : ℚ)
                    (angelfish : ℕ) (angelfish_food : ℚ)
                    (tetra : ℕ) (tetra_food : ℚ) : ℚ :=
  goldfish * goldfish_food +
  swordtails * swordtails_food +
  guppies * guppies_food +
  angelfish * angelfish_food +
  tetra * tetra_food

theorem fish_food_calculation :
  total_fish_food 4 1 5 2 10 (1/2) 3 (3/2) 6 1 = 59/2 := by
  sorry

end NUMINAMATH_CALUDE_fish_food_calculation_l1565_156573


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l1565_156598

theorem min_value_reciprocal_sum (x y : ℝ) (h1 : x * y > 0) (h2 : x + 4 * y = 3) :
  ∀ z w : ℝ, z * w > 0 → z + 4 * w = 3 → (1 / x + 1 / y) ≤ (1 / z + 1 / w) :=
by sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l1565_156598


namespace NUMINAMATH_CALUDE_arbitrarily_large_power_l1565_156572

theorem arbitrarily_large_power (a : ℝ) (h : a > 1) :
  ∀ y : ℝ, y > 0 → ∃ x : ℝ, a^x > y :=
by sorry

end NUMINAMATH_CALUDE_arbitrarily_large_power_l1565_156572


namespace NUMINAMATH_CALUDE_dog_count_l1565_156530

/-- Represents the number of dogs that can perform a specific combination of tricks -/
structure DogTricks where
  sit : ℕ
  stay : ℕ
  rollOver : ℕ
  sitStay : ℕ
  stayRollOver : ℕ
  sitRollOver : ℕ
  allThree : ℕ
  none : ℕ
  stayRollOverPlayDead : ℕ

/-- The total number of dogs in the training center -/
def totalDogs (d : DogTricks) : ℕ := sorry

/-- Theorem stating the total number of dogs in the training center -/
theorem dog_count (d : DogTricks) 
  (h1 : d.sit = 60)
  (h2 : d.stay = 35)
  (h3 : d.rollOver = 40)
  (h4 : d.sitStay = 22)
  (h5 : d.stayRollOver = 15)
  (h6 : d.sitRollOver = 20)
  (h7 : d.allThree = 10)
  (h8 : d.none = 10)
  (h9 : d.stayRollOverPlayDead = 5)
  (h10 : d.stayRollOverPlayDead ≤ d.stayRollOver) :
  totalDogs d = 98 := by
  sorry

end NUMINAMATH_CALUDE_dog_count_l1565_156530


namespace NUMINAMATH_CALUDE_triangle_third_side_length_l1565_156516

theorem triangle_third_side_length 
  (a b : ℝ) 
  (θ : Real) 
  (ha : a = 5) 
  (hb : b = 12) 
  (hθ : θ = 150 * π / 180) : 
  ∃ c : ℝ, c = Real.sqrt (169 + 60 * Real.sqrt 3) ∧ 
  c^2 = a^2 + b^2 - 2*a*b*(Real.cos θ) :=
sorry

end NUMINAMATH_CALUDE_triangle_third_side_length_l1565_156516


namespace NUMINAMATH_CALUDE_polynomial_division_theorem_l1565_156549

theorem polynomial_division_theorem (x : ℝ) :
  x^5 + 8 = (x + 2) * (x^4 - 2*x^3 + 4*x^2 - 8*x + 16) + (-24) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_theorem_l1565_156549


namespace NUMINAMATH_CALUDE_video_game_sales_earnings_l1565_156524

theorem video_game_sales_earnings 
  (total_games : ℕ) 
  (non_working_games : ℕ) 
  (price_per_game : ℕ) : 
  total_games = 16 → 
  non_working_games = 8 → 
  price_per_game = 7 → 
  (total_games - non_working_games) * price_per_game = 56 := by
sorry

end NUMINAMATH_CALUDE_video_game_sales_earnings_l1565_156524


namespace NUMINAMATH_CALUDE_largest_cards_per_page_l1565_156574

theorem largest_cards_per_page (album1 album2 album3 : ℕ) 
  (h1 : album1 = 1080) 
  (h2 : album2 = 1620) 
  (h3 : album3 = 540) : 
  Nat.gcd album1 (Nat.gcd album2 album3) = 540 := by
  sorry

end NUMINAMATH_CALUDE_largest_cards_per_page_l1565_156574


namespace NUMINAMATH_CALUDE_yellow_stamp_price_is_two_l1565_156543

/-- Calculates the price per yellow stamp needed to reach a total sale amount --/
def price_per_yellow_stamp (red_count : ℕ) (red_price : ℚ) (blue_count : ℕ) (blue_price : ℚ) 
  (yellow_count : ℕ) (total_sale : ℚ) : ℚ :=
  let red_earnings := red_count * red_price
  let blue_earnings := blue_count * blue_price
  let remaining := total_sale - (red_earnings + blue_earnings)
  remaining / yellow_count

/-- Theorem stating that the price per yellow stamp is $2 given the problem conditions --/
theorem yellow_stamp_price_is_two :
  price_per_yellow_stamp 20 1.1 80 0.8 7 100 = 2 := by
  sorry

end NUMINAMATH_CALUDE_yellow_stamp_price_is_two_l1565_156543


namespace NUMINAMATH_CALUDE_march_first_is_friday_l1565_156508

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a calendar month -/
structure Month where
  days : Nat
  firstDay : DayOfWeek

/-- Counts the number of occurrences of a specific day in a month -/
def countDayOccurrences (m : Month) (d : DayOfWeek) : Nat :=
  sorry

theorem march_first_is_friday (m : Month) : 
  m.days = 31 ∧ 
  countDayOccurrences m DayOfWeek.Friday = 5 ∧ 
  countDayOccurrences m DayOfWeek.Sunday = 4 → 
  m.firstDay = DayOfWeek.Friday :=
sorry

end NUMINAMATH_CALUDE_march_first_is_friday_l1565_156508


namespace NUMINAMATH_CALUDE_light_bulb_probability_l1565_156514

/-- The number of screw-in light bulbs in the box -/
def screwIn : ℕ := 3

/-- The number of bayonet light bulbs in the box -/
def bayonet : ℕ := 5

/-- The total number of light bulbs in the box -/
def totalBulbs : ℕ := screwIn + bayonet

/-- The number of draws -/
def draws : ℕ := 5

/-- The probability of drawing all screw-in light bulbs by the 5th draw -/
def probability : ℚ := 3 / 28

theorem light_bulb_probability :
  probability = (Nat.choose screwIn (screwIn - 1) * Nat.choose bayonet (draws - screwIn) * Nat.factorial (draws - 1)) /
                (Nat.choose totalBulbs draws * Nat.factorial draws) :=
sorry

end NUMINAMATH_CALUDE_light_bulb_probability_l1565_156514


namespace NUMINAMATH_CALUDE_choose_two_from_fifteen_l1565_156545

theorem choose_two_from_fifteen (n : ℕ) (k : ℕ) : n = 15 → k = 2 → Nat.choose n k = 105 := by
  sorry

end NUMINAMATH_CALUDE_choose_two_from_fifteen_l1565_156545


namespace NUMINAMATH_CALUDE_simplify_sqrt_450_l1565_156589

theorem simplify_sqrt_450 : Real.sqrt 450 = 15 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_450_l1565_156589


namespace NUMINAMATH_CALUDE_interior_cubes_6_5_4_l1565_156522

/-- Represents a rectangular prism -/
structure RectangularPrism where
  width : ℕ
  length : ℕ
  height : ℕ

/-- Calculates the number of interior cubes in a rectangular prism -/
def interiorCubes (prism : RectangularPrism) : ℕ :=
  (prism.width - 2) * (prism.length - 2) * (prism.height - 2)

/-- Theorem: A 6x5x4 rectangular prism cut into 1x1x1 cubes has 24 interior cubes -/
theorem interior_cubes_6_5_4 :
  interiorCubes { width := 6, length := 5, height := 4 } = 24 := by
  sorry

#eval interiorCubes { width := 6, length := 5, height := 4 }

end NUMINAMATH_CALUDE_interior_cubes_6_5_4_l1565_156522


namespace NUMINAMATH_CALUDE_smallest_power_is_four_l1565_156534

def rotation_matrix : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![0, -1],
    ![1,  0]]

theorem smallest_power_is_four :
  (∀ k : ℕ, k > 0 ∧ k < 4 → rotation_matrix ^ k ≠ 1) ∧
  rotation_matrix ^ 4 = 1 :=
sorry

end NUMINAMATH_CALUDE_smallest_power_is_four_l1565_156534


namespace NUMINAMATH_CALUDE_sets_equality_and_inclusion_l1565_156579

def A : Set ℝ := {x | x^2 - 4*x - 12 ≤ 0}
def B (m : ℝ) : Set ℝ := {x | x^2 - m*x - 6*m^2 ≤ 0}

theorem sets_equality_and_inclusion (m : ℝ) (h : m > 0) :
  A = {x : ℝ | -2 ≤ x ∧ x ≤ 6} ∧
  B m = {x : ℝ | -2*m ≤ x ∧ x ≤ 3*m} ∧
  (A ⊆ B m ↔ m ≥ 2) ∧
  (B m ⊆ A ↔ 0 < m ∧ m ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_sets_equality_and_inclusion_l1565_156579


namespace NUMINAMATH_CALUDE_players_who_quit_video_game_problem_l1565_156582

theorem players_who_quit (initial_players : ℕ) (lives_per_player : ℕ) (total_lives : ℕ) : ℕ :=
  initial_players - (total_lives / lives_per_player)

theorem video_game_problem :
  players_who_quit 13 6 30 = 8 := by
  sorry

end NUMINAMATH_CALUDE_players_who_quit_video_game_problem_l1565_156582


namespace NUMINAMATH_CALUDE_circle_intersection_length_l1565_156521

-- Define the right triangle ABC
structure RightTriangle where
  A : Real
  B : Real
  C : Real
  angle_A : A = 30 * Real.pi / 180
  hypotenuse : C = 2 * A
  right_angle : B = 90 * Real.pi / 180

-- Define the circle and point K
structure CircleAndPoint (t : RightTriangle) where
  K : Real
  on_hypotenuse : K ≤ t.C ∧ K ≥ 0
  diameter : t.A = 2

-- Theorem statement
theorem circle_intersection_length (t : RightTriangle) (c : CircleAndPoint t) :
  let CK := Real.sqrt (t.A * (t.C - c.K))
  CK = 1 := by sorry

end NUMINAMATH_CALUDE_circle_intersection_length_l1565_156521


namespace NUMINAMATH_CALUDE_angle_complementary_to_complement_l1565_156535

theorem angle_complementary_to_complement (α : Real) : 
  (90 - α) + (180 - α) = 180 → α = 45 := by
  sorry

end NUMINAMATH_CALUDE_angle_complementary_to_complement_l1565_156535


namespace NUMINAMATH_CALUDE_matrix_determinant_solution_l1565_156517

theorem matrix_determinant_solution (a : ℝ) (ha : a ≠ 0) :
  let matrix (x : ℝ) := !![x + a, 2*x, 2*x; 2*x, x + a, 2*x; 2*x, 2*x, x + a]
  ∀ x : ℝ, Matrix.det (matrix x) = 0 ↔ x = -a ∨ x = a/3 := by
  sorry

end NUMINAMATH_CALUDE_matrix_determinant_solution_l1565_156517


namespace NUMINAMATH_CALUDE_vacation_duration_l1565_156585

/-- Represents the vacation of a family -/
structure Vacation where
  total_days : ℕ
  rain_days : ℕ
  clear_afternoons : ℕ

/-- Theorem stating that given the conditions, the total number of days is 18 -/
theorem vacation_duration (v : Vacation) 
  (h1 : v.rain_days = 13)
  (h2 : v.clear_afternoons = 12)
  (h3 : v.rain_days + v.clear_afternoons ≤ v.total_days)
  (h4 : v.total_days ≤ v.rain_days + v.clear_afternoons + 1) :
  v.total_days = 18 := by
  sorry

#check vacation_duration

end NUMINAMATH_CALUDE_vacation_duration_l1565_156585


namespace NUMINAMATH_CALUDE_intersection_single_element_l1565_156558

/-- The value of k when the intersection of sets A and B has only one element -/
theorem intersection_single_element (x y : ℝ) :
  let A := {p : ℝ × ℝ | p.1^2 - 3*p.1*p.2 + 4*p.2^2 = 7/2}
  let B := {p : ℝ × ℝ | ∃ (k : ℝ), k > 0 ∧ k*p.1 + p.2 = 2}
  (∃! p, p ∈ A ∩ B) → (∃ (k : ℝ), k = 1/4 ∧ k > 0 ∧ ∀ p, p ∈ A ∩ B → k*p.1 + p.2 = 2) :=
by sorry

end NUMINAMATH_CALUDE_intersection_single_element_l1565_156558


namespace NUMINAMATH_CALUDE_line_slope_and_intercept_l1565_156588

/-- Given a line with equation 4y = 6x - 12, prove that its slope is 3/2 and y-intercept is -3. -/
theorem line_slope_and_intercept :
  ∃ (m b : ℚ), m = 3/2 ∧ b = -3 ∧
  ∀ (x y : ℚ), 4*y = 6*x - 12 ↔ y = m*x + b :=
sorry

end NUMINAMATH_CALUDE_line_slope_and_intercept_l1565_156588


namespace NUMINAMATH_CALUDE_exponent_division_l1565_156551

theorem exponent_division (x : ℝ) (h : x ≠ 0) : x^15 / x^3 = x^12 := by
  sorry

end NUMINAMATH_CALUDE_exponent_division_l1565_156551


namespace NUMINAMATH_CALUDE_gcd_problem_l1565_156509

theorem gcd_problem (b : ℤ) (h : ∃ k : ℤ, k % 2 = 1 ∧ b = k * 1177) :
  Nat.gcd (Int.natAbs (2 * b^2 + 31 * b + 71)) (Int.natAbs (b + 15)) = 1 := by
sorry

end NUMINAMATH_CALUDE_gcd_problem_l1565_156509


namespace NUMINAMATH_CALUDE_quadrant_line_relationships_l1565_156566

/-- A line passing through the first, second, and fourth quadrants -/
structure QuadrantLine where
  a : ℝ
  b : ℝ
  c : ℝ
  passes_through_quadrants : 
    ∃ (x₁ y₁ x₂ y₂ x₄ y₄ : ℝ),
      (x₁ > 0 ∧ y₁ > 0 ∧ a * x₁ + b * y₁ + c = 0) ∧
      (x₂ < 0 ∧ y₂ > 0 ∧ a * x₂ + b * y₂ + c = 0) ∧
      (x₄ > 0 ∧ y₄ < 0 ∧ a * x₄ + b * y₄ + c = 0)

/-- The relationships between a, b, and c for a line passing through the first, second, and fourth quadrants -/
theorem quadrant_line_relationships (l : QuadrantLine) : l.a * l.b > 0 ∧ l.b * l.c < 0 := by
  sorry

end NUMINAMATH_CALUDE_quadrant_line_relationships_l1565_156566


namespace NUMINAMATH_CALUDE_largest_b_value_l1565_156563

theorem largest_b_value (b : ℝ) : 
  (3 * b + 4) * (b - 3) = 9 * b → 
  b ≤ (4 + 4 * Real.sqrt 5) / 6 ∧ 
  ∃ (b : ℝ), (3 * b + 4) * (b - 3) = 9 * b ∧ b = (4 + 4 * Real.sqrt 5) / 6 := by
  sorry

end NUMINAMATH_CALUDE_largest_b_value_l1565_156563


namespace NUMINAMATH_CALUDE_worksheet_problems_l1565_156533

theorem worksheet_problems (total_worksheets graded_worksheets remaining_problems : ℕ) 
  (h1 : total_worksheets = 9)
  (h2 : graded_worksheets = 5)
  (h3 : remaining_problems = 16) :
  (total_worksheets - graded_worksheets) * (remaining_problems / (total_worksheets - graded_worksheets)) = 4 := by
  sorry

end NUMINAMATH_CALUDE_worksheet_problems_l1565_156533


namespace NUMINAMATH_CALUDE_infinitely_many_solutions_l1565_156599

theorem infinitely_many_solutions : 
  ∃ (S : Set (ℤ × ℤ × ℤ)), Set.Infinite S ∧ 
  ∀ (x y z : ℤ), (x, y, z) ∈ S → x^2 + y^2 + z^2 = x^3 + y^3 + z^3 :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_solutions_l1565_156599


namespace NUMINAMATH_CALUDE_x_cos_x_necessary_not_sufficient_l1565_156520

theorem x_cos_x_necessary_not_sufficient (x : ℝ) (h : 0 < x ∧ x < Real.pi / 2) :
  (∀ y : ℝ, 0 < y ∧ y < Real.pi / 2 → (y < 1 → y * Real.cos y < 1)) ∧
  (∃ z : ℝ, 0 < z ∧ z < Real.pi / 2 ∧ z * Real.cos z < 1 ∧ z ≥ 1) := by
  sorry

end NUMINAMATH_CALUDE_x_cos_x_necessary_not_sufficient_l1565_156520


namespace NUMINAMATH_CALUDE_f_sixteen_value_l1565_156537

theorem f_sixteen_value (f : ℝ → ℝ) 
  (h1 : ∀ x, f (2 * x) = -2 * f x) 
  (h2 : f 1 = -3) : 
  f 16 = -48 := by
sorry

end NUMINAMATH_CALUDE_f_sixteen_value_l1565_156537


namespace NUMINAMATH_CALUDE_smallest_integer_solution_unique_smallest_solution_l1565_156531

theorem smallest_integer_solution (x : ℤ) : (10 - 5 * x < -18) ↔ x ≥ 6 :=
  sorry

theorem unique_smallest_solution : ∃! x : ℤ, (10 - 5 * x < -18) ∧ ∀ y : ℤ, (10 - 5 * y < -18) → x ≤ y :=
  sorry

end NUMINAMATH_CALUDE_smallest_integer_solution_unique_smallest_solution_l1565_156531


namespace NUMINAMATH_CALUDE_max_value_expression_l1565_156555

theorem max_value_expression (a b c d : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (hsum : a + b + c + d ≤ 4) : 
  (a^2 * (a + b))^(1/4) + (b^2 * (b + c))^(1/4) + 
  (c^2 * (c + d))^(1/4) + (d^2 * (d + a))^(1/4) ≤ 4 * 2^(1/4) :=
sorry

end NUMINAMATH_CALUDE_max_value_expression_l1565_156555


namespace NUMINAMATH_CALUDE_division_with_remainder_l1565_156547

theorem division_with_remainder (n : ℕ) (h1 : n % 17 ≠ 0) (h2 : n / 17 = 25) :
  n ≤ 441 ∧ n ≥ 426 := by
  sorry

end NUMINAMATH_CALUDE_division_with_remainder_l1565_156547


namespace NUMINAMATH_CALUDE_porridge_eaters_today_l1565_156580

/-- Represents the number of children eating porridge daily -/
def daily_eaters : ℕ := 5

/-- Represents the number of children eating porridge every other day -/
def alternate_eaters : ℕ := 7

/-- Represents the number of children who ate porridge yesterday -/
def yesterday_eaters : ℕ := 9

/-- Calculates the number of children eating porridge today -/
def today_eaters : ℕ := daily_eaters + (alternate_eaters - (yesterday_eaters - daily_eaters))

/-- Theorem stating that the number of children eating porridge today is 8 -/
theorem porridge_eaters_today : today_eaters = 8 := by
  sorry

end NUMINAMATH_CALUDE_porridge_eaters_today_l1565_156580


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_fractional_equation_solution_l1565_156590

-- Equation 1
theorem quadratic_equation_solution (x : ℝ) :
  2 * x^2 + 3 * x = 1 ↔ x = (-3 + Real.sqrt 17) / 4 ∨ x = (-3 - Real.sqrt 17) / 4 :=
sorry

-- Equation 2
theorem fractional_equation_solution (x : ℝ) (h : x ≠ 2) :
  3 / (x - 2) = 5 / (2 - x) - 1 ↔ x = -6 :=
sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_fractional_equation_solution_l1565_156590


namespace NUMINAMATH_CALUDE_marble_distribution_l1565_156501

def is_valid_combination (a b c d e : ℕ) : Prop :=
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧
  c ≠ d ∧ c ≠ e ∧
  d ≠ e ∧
  a > 1 ∧ b > 1 ∧ c > 1 ∧ d > 1 ∧ e > 1 ∧
  a + e ≥ 11 ∧
  c + a < 11 ∧
  b + c ≥ 11 ∧
  c + d ≥ 11 ∧
  a + b + c + d + e = 26

theorem marble_distribution :
  ∀ a b c d e : ℕ,
  is_valid_combination a b c d e ↔
  ((a = 1 ∧ b = 2 ∧ c = 3 ∧ d = 9 ∧ e = 11) ∨
   (a = 1 ∧ b = 2 ∧ c = 4 ∧ d = 9 ∧ e = 10) ∨
   (a = 1 ∧ b = 3 ∧ c = 4 ∧ d = 8 ∧ e = 10) ∨
   (a = 2 ∧ b = 3 ∧ c = 4 ∧ d = 8 ∧ e = 9)) :=
by sorry

end NUMINAMATH_CALUDE_marble_distribution_l1565_156501


namespace NUMINAMATH_CALUDE_total_score_is_54_l1565_156538

/-- The number of players on the basketball team -/
def num_players : ℕ := 8

/-- The points scored by each player -/
def player_scores : Fin num_players → ℕ
  | ⟨0, _⟩ => 7
  | ⟨1, _⟩ => 8
  | ⟨2, _⟩ => 2
  | ⟨3, _⟩ => 11
  | ⟨4, _⟩ => 6
  | ⟨5, _⟩ => 12
  | ⟨6, _⟩ => 1
  | ⟨7, _⟩ => 7

/-- The theorem stating that the sum of all player scores is 54 -/
theorem total_score_is_54 : (Finset.univ.sum player_scores) = 54 := by
  sorry

end NUMINAMATH_CALUDE_total_score_is_54_l1565_156538


namespace NUMINAMATH_CALUDE_equation_solution_and_condition_l1565_156569

theorem equation_solution_and_condition :
  ∃ x : ℝ, (3 * x + 7 = 22) ∧ (2 * x + 1 ≠ 9) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_and_condition_l1565_156569


namespace NUMINAMATH_CALUDE_jims_bulb_purchase_l1565_156556

theorem jims_bulb_purchase : 
  let lamp_cost : ℕ := 7
  let bulb_cost : ℕ := lamp_cost - 4
  let num_lamps : ℕ := 2
  let total_cost : ℕ := 32
  let bulbs_cost : ℕ := total_cost - (num_lamps * lamp_cost)
  ∃ (num_bulbs : ℕ), num_bulbs * bulb_cost = bulbs_cost ∧ num_bulbs = 6
  := by sorry

end NUMINAMATH_CALUDE_jims_bulb_purchase_l1565_156556
