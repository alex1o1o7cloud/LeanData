import Mathlib

namespace NUMINAMATH_CALUDE_vanessa_albums_l855_85547

theorem vanessa_albums (phone_pics : ℕ) (camera_pics : ℕ) (pics_per_album : ℕ) : 
  phone_pics = 23 → 
  camera_pics = 7 → 
  pics_per_album = 6 → 
  (phone_pics + camera_pics) / pics_per_album = 5 := by
sorry

end NUMINAMATH_CALUDE_vanessa_albums_l855_85547


namespace NUMINAMATH_CALUDE_remainder_problem_l855_85543

theorem remainder_problem (d r : ℤ) : 
  d > 1 ∧ 
  1225 % d = r ∧ 
  1681 % d = r ∧ 
  2756 % d = r → 
  d - r = 10 := by
sorry

end NUMINAMATH_CALUDE_remainder_problem_l855_85543


namespace NUMINAMATH_CALUDE_range_when_p_true_range_when_p_or_q_true_and_p_and_q_false_l855_85550

-- Define proposition p
def p (a : ℝ) : Prop := ∀ x ∈ Set.Icc (-2) (-1), x^2 - a ≥ 0

-- Define proposition q
def q (a : ℝ) : Prop := ∃ x : ℝ, x^2 + 2*a*x - (a-2) = 0

-- Theorem 1
theorem range_when_p_true (a : ℝ) : p a → a ≤ 1 := by sorry

-- Theorem 2
theorem range_when_p_or_q_true_and_p_and_q_false (a : ℝ) :
  (p a ∨ q a) ∧ ¬(p a ∧ q a) → a ∈ Set.Ioo (-2) 1 ∪ Set.Ioi 1 := by sorry

end NUMINAMATH_CALUDE_range_when_p_true_range_when_p_or_q_true_and_p_and_q_false_l855_85550


namespace NUMINAMATH_CALUDE_no_simultaneously_safe_numbers_l855_85520

def is_p_safe (n p : ℕ) : Prop :=
  ∀ k : ℤ, |n - k * p| > 3

def simultaneously_safe (n : ℕ) : Prop :=
  is_p_safe n 5 ∧ is_p_safe n 9 ∧ is_p_safe n 11

theorem no_simultaneously_safe_numbers : 
  ¬∃ n : ℕ, n > 0 ∧ n ≤ 5000 ∧ simultaneously_safe n :=
sorry

end NUMINAMATH_CALUDE_no_simultaneously_safe_numbers_l855_85520


namespace NUMINAMATH_CALUDE_smallest_n_congruence_l855_85554

theorem smallest_n_congruence (n : ℕ) : 
  (n > 0 ∧ 23 * n ≡ 789 [ZMOD 11]) → n ≥ 9 :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_congruence_l855_85554


namespace NUMINAMATH_CALUDE_calculation_proof_equation_solution_proof_l855_85526

-- Part 1
theorem calculation_proof :
  0.01⁻¹ + (-1 - 2/7)^0 - Real.sqrt 9 = 98 := by sorry

-- Part 2
theorem equation_solution_proof :
  ∀ x : ℝ, (2 / (x - 3) = 3 / (x - 2)) ↔ (x = 5) := by sorry

end NUMINAMATH_CALUDE_calculation_proof_equation_solution_proof_l855_85526


namespace NUMINAMATH_CALUDE_power_sum_of_i_l855_85583

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem power_sum_of_i : i^23 + i^223 = -2*i := by sorry

end NUMINAMATH_CALUDE_power_sum_of_i_l855_85583


namespace NUMINAMATH_CALUDE_johnny_rate_is_four_l855_85591

/-- The walking problem scenario -/
structure WalkingScenario where
  total_distance : ℝ
  matthew_rate : ℝ
  johnny_distance : ℝ
  matthew_head_start : ℝ

/-- Calculate Johnny's walking rate given a WalkingScenario -/
def calculate_johnny_rate (scenario : WalkingScenario) : ℝ :=
  sorry

/-- Theorem stating that Johnny's walking rate is 4 km/h given the specific scenario -/
theorem johnny_rate_is_four (scenario : WalkingScenario) 
  (h1 : scenario.total_distance = 45)
  (h2 : scenario.matthew_rate = 3)
  (h3 : scenario.johnny_distance = 24)
  (h4 : scenario.matthew_head_start = 1) :
  calculate_johnny_rate scenario = 4 := by
  sorry

end NUMINAMATH_CALUDE_johnny_rate_is_four_l855_85591


namespace NUMINAMATH_CALUDE_jump_height_ratio_l855_85553

/-- The jump heights of four people and their ratios -/
theorem jump_height_ratio :
  let mark_height := 6
  let lisa_height := 2 * mark_height
  let jacob_height := 2 * lisa_height
  let james_height := 16
  (james_height : ℚ) / jacob_height = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_jump_height_ratio_l855_85553


namespace NUMINAMATH_CALUDE_problem_statement_l855_85571

theorem problem_statement (x y : ℝ) (h1 : x - y > -x) (h2 : x + y > y) : x > 0 ∧ y < 2*x := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l855_85571


namespace NUMINAMATH_CALUDE_expand_polynomial_l855_85540

theorem expand_polynomial (x : ℝ) : (x - 3) * (x + 3) * (x^2 + 9) = x^4 - 81 := by
  sorry

end NUMINAMATH_CALUDE_expand_polynomial_l855_85540


namespace NUMINAMATH_CALUDE_right_triangle_area_l855_85541

/-- The area of a right triangle with one leg of 30 inches and a hypotenuse of 34 inches is 240 square inches. -/
theorem right_triangle_area (a b c : ℝ) (h1 : a = 30) (h2 : c = 34) (h3 : a^2 + b^2 = c^2) :
  (1/2) * a * b = 240 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_area_l855_85541


namespace NUMINAMATH_CALUDE_no_linear_term_implies_equal_coefficients_l855_85593

theorem no_linear_term_implies_equal_coefficients (x m n : ℝ) : 
  (x + m) * (x - n) = x^2 + (-m * n) → m = n :=
by sorry

end NUMINAMATH_CALUDE_no_linear_term_implies_equal_coefficients_l855_85593


namespace NUMINAMATH_CALUDE_radical_conjugate_sum_product_l855_85531

theorem radical_conjugate_sum_product (x y : ℝ) : 
  (x + Real.sqrt y) + (x - Real.sqrt y) = 6 ∧ 
  (x + Real.sqrt y) * (x - Real.sqrt y) = 9 → 
  x + y = 3 := by
  sorry

end NUMINAMATH_CALUDE_radical_conjugate_sum_product_l855_85531


namespace NUMINAMATH_CALUDE_sum_of_numbers_l855_85530

theorem sum_of_numbers (a b : ℕ) (h1 : a = 64 ∧ b = 32) (h2 : max a b = 2 * min a b) : a + b = 96 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_numbers_l855_85530


namespace NUMINAMATH_CALUDE_expand_polynomial_l855_85555

theorem expand_polynomial (x : ℝ) : (x + 3) * (4 * x^2 - 5 * x + 6) = 4 * x^3 + 7 * x^2 - 9 * x + 18 := by
  sorry

end NUMINAMATH_CALUDE_expand_polynomial_l855_85555


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_is_8_l855_85507

/-- An isosceles triangle with specific properties -/
structure IsoscelesTriangle where
  -- The height of the triangle
  height : ℝ
  -- The ratio of base to side (4:3)
  baseToSideRatio : ℚ
  -- Assumption that the height is 20
  height_is_20 : height = 20
  -- Assumption that the base to side ratio is 4:3
  ratio_is_4_3 : baseToSideRatio = 4 / 3

/-- The radius of the inscribed circle in the isosceles triangle -/
def inscribedCircleRadius (t : IsoscelesTriangle) : ℝ := 8

/-- Theorem stating that the radius of the inscribed circle is 8 -/
theorem inscribed_circle_radius_is_8 (t : IsoscelesTriangle) :
  inscribedCircleRadius t = 8 := by sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_is_8_l855_85507


namespace NUMINAMATH_CALUDE_right_angle_clackers_l855_85594

/-- The number of clackers in a full circle -/
def clackers_in_full_circle : ℕ := 600

/-- The fraction of a full circle that a right angle represents -/
def right_angle_fraction : ℚ := 1/4

/-- The number of clackers in a right angle -/
def clackers_in_right_angle : ℕ := 150

/-- Theorem: The number of clackers in a right angle is 150 -/
theorem right_angle_clackers :
  clackers_in_right_angle = (clackers_in_full_circle : ℚ) * right_angle_fraction := by
  sorry

end NUMINAMATH_CALUDE_right_angle_clackers_l855_85594


namespace NUMINAMATH_CALUDE_intersection_complement_l855_85598

def A : Set ℝ := {x | 1 < x ∧ x < 3}
def B : Set ℝ := {x | 2 < x}

theorem intersection_complement : A ∩ (Bᶜ) = {x : ℝ | 1 < x ∧ x ≤ 2} := by sorry

end NUMINAMATH_CALUDE_intersection_complement_l855_85598


namespace NUMINAMATH_CALUDE_cat_grooming_time_l855_85599

/-- Calculates the total grooming time for a cat given specific grooming tasks and the cat's characteristics. -/
theorem cat_grooming_time :
  let clip_time_per_claw : ℕ := 10
  let clean_time_per_ear : ℕ := 90
  let shampoo_time_minutes : ℕ := 5
  let claws_per_foot : ℕ := 4
  let feet : ℕ := 4
  let ears : ℕ := 2
  clip_time_per_claw * claws_per_foot * feet + 
  clean_time_per_ear * ears + 
  shampoo_time_minutes * 60 = 640 := by
sorry


end NUMINAMATH_CALUDE_cat_grooming_time_l855_85599


namespace NUMINAMATH_CALUDE_triangle_most_stable_l855_85535

-- Define the shapes
inductive Shape
  | Rectangle
  | Trapezoid
  | Parallelogram
  | Triangle

-- Define stability as a property of shapes
def is_stable (s : Shape) : Prop :=
  match s with
  | Shape.Triangle => true
  | _ => false

-- Define the stability comparison
def more_stable (s1 s2 : Shape) : Prop :=
  is_stable s1 ∧ ¬is_stable s2

-- Theorem statement
theorem triangle_most_stable :
  ∀ s : Shape, s ≠ Shape.Triangle → more_stable Shape.Triangle s :=
sorry

end NUMINAMATH_CALUDE_triangle_most_stable_l855_85535


namespace NUMINAMATH_CALUDE_total_price_joan_twice_more_price_difference_l855_85542

/-- The price of Joan's sofa -/
def joan_price : ℕ := 230

/-- The price of Karl's sofa -/
def karl_price : ℕ := 600 - joan_price

/-- The sum of Joan and Karl's sofa prices -/
theorem total_price : joan_price + karl_price = 600 := by sorry

/-- Twice Joan's price is more than Karl's price -/
theorem joan_twice_more : 2 * joan_price > karl_price := by sorry

/-- The difference between twice Joan's price and Karl's price is 90 -/
theorem price_difference : 2 * joan_price - karl_price = 90 := by sorry

end NUMINAMATH_CALUDE_total_price_joan_twice_more_price_difference_l855_85542


namespace NUMINAMATH_CALUDE_horner_method_properties_l855_85551

def f (x : ℝ) : ℝ := 12 + 35*x + 9*x^3 + 5*x^5 + 3*x^6

def horner_v3 (a : List ℝ) (x : ℝ) : ℝ :=
  match a with
  | [] => 0
  | a₀ :: as => List.foldl (fun acc a_i => acc * x + a_i) a₀ as

theorem horner_method_properties :
  let a := [3, 5, 0, 9, 0, 35, 12]
  let x := -1
  ∃ (multiplications additions : ℕ) (v3 : ℝ),
    multiplications = 6 ∧
    additions = 6 ∧
    v3 = horner_v3 (List.take 4 a) x ∧
    v3 = 11 ∧
    f x = horner_v3 a x :=
by sorry

end NUMINAMATH_CALUDE_horner_method_properties_l855_85551


namespace NUMINAMATH_CALUDE_bus_profit_at_2600_passengers_l855_85508

/-- Represents the monthly profit of a minibus based on the number of passengers -/
def monthly_profit (passengers : ℕ) : ℤ :=
  2 * (passengers : ℤ) - 5000

/-- Theorem stating that the bus makes a profit with 2600 passengers -/
theorem bus_profit_at_2600_passengers :
  monthly_profit 2600 > 0 := by
  sorry

end NUMINAMATH_CALUDE_bus_profit_at_2600_passengers_l855_85508


namespace NUMINAMATH_CALUDE_freshmen_sample_size_is_20_l855_85512

/-- Calculates the number of freshmen to be sampled in a stratified sampling scheme -/
def freshmenSampleSize (totalStudents sampleSize freshmen : ℕ) : ℕ :=
  (freshmen * sampleSize) / totalStudents

/-- Theorem stating that the number of freshmen to be sampled is 20 -/
theorem freshmen_sample_size_is_20 :
  freshmenSampleSize 900 45 400 = 20 := by
  sorry

end NUMINAMATH_CALUDE_freshmen_sample_size_is_20_l855_85512


namespace NUMINAMATH_CALUDE_book_page_digits_l855_85519

/-- Calculate the total number of digits used to number pages in a book -/
def totalDigits (n : ℕ) : ℕ :=
  let singleDigit := min n 9
  let doubleDigit := max 0 (min n 99 - 9)
  let tripleDigit := max 0 (n - 99)
  singleDigit + 2 * doubleDigit + 3 * tripleDigit

/-- The total number of digits used in numbering the pages of a book with 360 pages is 972 -/
theorem book_page_digits :
  totalDigits 360 = 972 := by
  sorry

end NUMINAMATH_CALUDE_book_page_digits_l855_85519


namespace NUMINAMATH_CALUDE_custom_operation_result_l855_85548

/-- Custom operation * for non-zero integers -/
def star (a b : ℤ) : ℚ := 1 / a + 1 / b

/-- Theorem: Given the conditions, prove that a * b = 3/8 -/
theorem custom_operation_result (a b : ℤ) 
  (h1 : a + b = 12) 
  (h2 : a * b = 32) 
  (h3 : b = 8) 
  (ha : a ≠ 0) 
  (hb : b ≠ 0) : 
  star a b = 3 / 8 := by
  sorry

#check custom_operation_result

end NUMINAMATH_CALUDE_custom_operation_result_l855_85548


namespace NUMINAMATH_CALUDE_inequality_solution_l855_85563

theorem inequality_solution (x : ℝ) : 2 ≤ x / (3 * x - 7) ∧ x / (3 * x - 7) < 5 ↔ x = 14 / 5 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l855_85563


namespace NUMINAMATH_CALUDE_e₁_e₂_form_basis_l855_85513

def e₁ : ℝ × ℝ := (-1, 2)
def e₂ : ℝ × ℝ := (5, 7)

def are_collinear (v w : ℝ × ℝ) : Prop :=
  v.1 * w.2 = v.2 * w.1

def form_basis (v w : ℝ × ℝ) : Prop :=
  ¬(are_collinear v w)

theorem e₁_e₂_form_basis : form_basis e₁ e₂ := by
  sorry

end NUMINAMATH_CALUDE_e₁_e₂_form_basis_l855_85513


namespace NUMINAMATH_CALUDE_magnitude_of_sum_l855_85532

/-- Given two vectors a and b in ℝ², prove that |2a + b| = 3 -/
theorem magnitude_of_sum (a b : ℝ × ℝ) :
  (‖a‖ = 1) →
  (b = (1, 2)) →
  (a • b = 0) →
  ‖2 • a + b‖ = 3 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_sum_l855_85532


namespace NUMINAMATH_CALUDE_spade_equation_solution_l855_85523

-- Define the ♠ operation
def spade (A B : ℝ) : ℝ := 4 * A + 3 * B + 6

-- Theorem statement
theorem spade_equation_solution :
  ∃! A : ℝ, spade A 5 = 79 ∧ A = 14.5 := by sorry

end NUMINAMATH_CALUDE_spade_equation_solution_l855_85523


namespace NUMINAMATH_CALUDE_portfolio_distribution_l855_85592

theorem portfolio_distribution (total_students : ℕ) (total_portfolios : ℕ) 
  (h1 : total_students = 120) 
  (h2 : total_portfolios = 8365) : 
  ∃ (regular_portfolios : ℕ) (special_portfolios : ℕ) (remaining_portfolios : ℕ),
    let regular_students : ℕ := (85 * total_students) / 100
    let special_students : ℕ := total_students - regular_students
    special_portfolios = regular_portfolios + 10 ∧
    regular_students * regular_portfolios + special_students * special_portfolios + remaining_portfolios = total_portfolios ∧
    remaining_portfolios = 25 :=
by sorry

end NUMINAMATH_CALUDE_portfolio_distribution_l855_85592


namespace NUMINAMATH_CALUDE_max_tiles_on_floor_l855_85579

/-- Represents the dimensions of a rectangle -/
structure Dimensions where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangle given its dimensions -/
def area (d : Dimensions) : ℝ := d.length * d.width

/-- Represents the floor and tile dimensions -/
def floor : Dimensions := { length := 400, width := 600 }
def tile : Dimensions := { length := 20, width := 30 }

/-- Theorem stating the maximum number of tiles that can fit on the floor -/
theorem max_tiles_on_floor :
  (area floor / area tile : ℝ) = 400 := by sorry

end NUMINAMATH_CALUDE_max_tiles_on_floor_l855_85579


namespace NUMINAMATH_CALUDE_function_is_constant_l855_85510

/-- A function satisfying the given conditions is constant and non-zero -/
theorem function_is_constant (f : ℝ → ℝ) 
  (h1 : ∀ x y, f x + f y ≠ 0)
  (h2 : ∀ x y, (f x - f (x - y)) / (f x + f (x + y)) + (f x - f (x + y)) / (f x + f (x - y)) = 0) :
  ∃ c : ℝ, c ≠ 0 ∧ ∀ x, f x = c :=
sorry

end NUMINAMATH_CALUDE_function_is_constant_l855_85510


namespace NUMINAMATH_CALUDE_bill_roof_capacity_l855_85517

/-- The number of leaves that fall on Bill's roof each day -/
def leaves_per_day : ℕ := 100

/-- The number of leaves that weigh one pound -/
def leaves_per_pound : ℕ := 1000

/-- The number of days it takes for Bill's roof to collapse -/
def days_to_collapse : ℕ := 5000

/-- The weight Bill's roof can bear in pounds -/
def roof_weight_capacity : ℚ := (leaves_per_day : ℚ) / (leaves_per_pound : ℚ) * days_to_collapse

theorem bill_roof_capacity :
  roof_weight_capacity = 500 := by sorry

end NUMINAMATH_CALUDE_bill_roof_capacity_l855_85517


namespace NUMINAMATH_CALUDE_winnie_lollipops_left_l855_85505

/-- The number of lollipops left after equal distribution -/
def lollipops_left (total : ℕ) (friends : ℕ) : ℕ :=
  total % friends

theorem winnie_lollipops_left :
  let cherry := 60
  let wintergreen := 145
  let grape := 10
  let shrimp_cocktail := 295
  let total := cherry + wintergreen + grape + shrimp_cocktail
  let friends := 13
  lollipops_left total friends = 3 := by
  sorry

end NUMINAMATH_CALUDE_winnie_lollipops_left_l855_85505


namespace NUMINAMATH_CALUDE_even_function_graph_l855_85564

/-- An even function is a function that satisfies f(-x) = f(x) for all x in its domain -/
def EvenFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

/-- The statement that (-a, f(a)) lies on the graph of f for any even function f and any real a -/
theorem even_function_graph (f : ℝ → ℝ) (h : EvenFunction f) (a : ℝ) :
  f (-a) = f a := by sorry

end NUMINAMATH_CALUDE_even_function_graph_l855_85564


namespace NUMINAMATH_CALUDE_green_balls_count_l855_85567

theorem green_balls_count (total : ℕ) (prob : ℚ) (green : ℕ) : 
  total = 12 → 
  prob = 1 / 22 → 
  (green : ℚ) / total * (green - 1) / (total - 1) = prob → 
  green = 3 := by
  sorry

end NUMINAMATH_CALUDE_green_balls_count_l855_85567


namespace NUMINAMATH_CALUDE_cookies_eaten_l855_85527

theorem cookies_eaten (initial : Real) (remaining : Real) (eaten : Real) : 
  initial = 28.5 → remaining = 7.25 → eaten = initial - remaining → eaten = 21.25 :=
by sorry

end NUMINAMATH_CALUDE_cookies_eaten_l855_85527


namespace NUMINAMATH_CALUDE_square_difference_squared_l855_85562

theorem square_difference_squared : (7^2 - 3^2)^2 = 1600 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_squared_l855_85562


namespace NUMINAMATH_CALUDE_pythagorean_triple_for_odd_integer_l855_85574

theorem pythagorean_triple_for_odd_integer (x : ℕ) 
  (h1 : x > 1) 
  (h2 : Odd x) : 
  ∃ y z : ℕ, 
    y > 0 ∧ 
    z > 0 ∧ 
    y = (x^2 - 1) / 2 ∧ 
    z = (x^2 + 1) / 2 ∧ 
    x^2 + y^2 = z^2 := by
  sorry

end NUMINAMATH_CALUDE_pythagorean_triple_for_odd_integer_l855_85574


namespace NUMINAMATH_CALUDE_carol_rectangle_width_l855_85545

/-- Given two rectangles with equal area, where one has a length of 5 inches
    and the other has dimensions of 3 inches by 40 inches,
    prove that the width of the first rectangle is 24 inches. -/
theorem carol_rectangle_width
  (length_carol : ℝ)
  (width_carol : ℝ)
  (length_jordan : ℝ)
  (width_jordan : ℝ)
  (h1 : length_carol = 5)
  (h2 : length_jordan = 3)
  (h3 : width_jordan = 40)
  (h4 : length_carol * width_carol = length_jordan * width_jordan) :
  width_carol = 24 :=
by sorry

end NUMINAMATH_CALUDE_carol_rectangle_width_l855_85545


namespace NUMINAMATH_CALUDE_binomial_coefficient_equation_unique_solution_l855_85538

theorem binomial_coefficient_equation_unique_solution : 
  ∃! n : ℕ, Nat.choose 25 n + Nat.choose 25 12 = Nat.choose 26 13 ∧ n = 13 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_equation_unique_solution_l855_85538


namespace NUMINAMATH_CALUDE_group_size_calculation_l855_85582

theorem group_size_calculation (n : ℕ) : 
  (n * 15 + 37 = 17 * (n + 1)) → n = 10 := by
  sorry

end NUMINAMATH_CALUDE_group_size_calculation_l855_85582


namespace NUMINAMATH_CALUDE_modified_tic_tac_toe_tie_probability_l855_85533

theorem modified_tic_tac_toe_tie_probability 
  (amy_win_prob : ℚ)
  (lily_win_prob : ℚ)
  (ben_win_prob : ℚ)
  (h1 : amy_win_prob = 4 / 15)
  (h2 : lily_win_prob = 1 / 5)
  (h3 : ben_win_prob = 1 / 6)
  (h4 : amy_win_prob + lily_win_prob + ben_win_prob < 1) : 
  1 - (amy_win_prob + lily_win_prob + ben_win_prob) = 11 / 30 := by
sorry

end NUMINAMATH_CALUDE_modified_tic_tac_toe_tie_probability_l855_85533


namespace NUMINAMATH_CALUDE_min_value_of_expression_l855_85501

theorem min_value_of_expression :
  (∀ x y : ℝ, (2*x*y - 3)^2 + (x + y)^2 ≥ 1) ∧
  (∃ x y : ℝ, (2*x*y - 3)^2 + (x + y)^2 = 1) := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l855_85501


namespace NUMINAMATH_CALUDE_derivative_x_ln_x_l855_85578

theorem derivative_x_ln_x (x : ℝ) (h : x > 0) :
  deriv (fun x => x * Real.log x) x = Real.log x + 1 := by
  sorry

end NUMINAMATH_CALUDE_derivative_x_ln_x_l855_85578


namespace NUMINAMATH_CALUDE_income_mean_difference_l855_85544

/-- The number of families in the dataset -/
def num_families : ℕ := 500

/-- The correct maximum income -/
def correct_max_income : ℕ := 120000

/-- The incorrect maximum income -/
def incorrect_max_income : ℕ := 1200000

/-- The sum of all incomes excluding the maximum -/
def T : ℕ := sorry

theorem income_mean_difference :
  (T + incorrect_max_income) / num_families - (T + correct_max_income) / num_families = 2160 :=
sorry

end NUMINAMATH_CALUDE_income_mean_difference_l855_85544


namespace NUMINAMATH_CALUDE_max_cube_hemisphere_ratio_l855_85502

/-- The maximum ratio of the volume of a cube inscribed in a hemisphere to the volume of the hemisphere -/
theorem max_cube_hemisphere_ratio : 
  let r := Real.sqrt 6 / (3 * Real.pi)
  ∃ (R a : ℝ), R > 0 ∧ a > 0 ∧
  (a^2 + (Real.sqrt 2 * a / 2)^2 = R^2) ∧
  (∀ (b : ℝ), b > 0 → b^2 + (Real.sqrt 2 * b / 2)^2 ≤ R^2 → 
    b^3 / ((2/3) * Real.pi * R^3) ≤ r) ∧
  a^3 / ((2/3) * Real.pi * R^3) = r :=
sorry

end NUMINAMATH_CALUDE_max_cube_hemisphere_ratio_l855_85502


namespace NUMINAMATH_CALUDE_cuboid_volume_example_l855_85536

/-- The volume of a cuboid with given base area and height -/
def cuboid_volume (base_area : ℝ) (height : ℝ) : ℝ :=
  base_area * height

/-- Theorem: The volume of a cuboid with base area 18 m^2 and height 8 m is 144 m^3 -/
theorem cuboid_volume_example : cuboid_volume 18 8 = 144 := by
  sorry

end NUMINAMATH_CALUDE_cuboid_volume_example_l855_85536


namespace NUMINAMATH_CALUDE_solution_set_f_less_than_one_range_of_a_l855_85503

-- Define the function f(x)
def f (x : ℝ) : ℝ := |2*x + 1| - |x - 1|

-- Theorem for the solution set of f(x) < 1
theorem solution_set_f_less_than_one :
  {x : ℝ | f x < 1} = {x : ℝ | -3 < x ∧ x < 1/3} := by sorry

-- Theorem for the range of a
theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, f x ≤ a - a^2/2 + 5/2) → -2 ≤ a ∧ a ≤ 4 := by sorry

end NUMINAMATH_CALUDE_solution_set_f_less_than_one_range_of_a_l855_85503


namespace NUMINAMATH_CALUDE_hotel_rooms_l855_85588

theorem hotel_rooms (total_floors : Nat) (unavailable_floors : Nat) (available_rooms : Nat) :
  total_floors = 10 →
  unavailable_floors = 1 →
  available_rooms = 90 →
  (total_floors - unavailable_floors) * (available_rooms / (total_floors - unavailable_floors)) = available_rooms :=
by
  sorry

end NUMINAMATH_CALUDE_hotel_rooms_l855_85588


namespace NUMINAMATH_CALUDE_existence_of_three_quadratics_l855_85511

theorem existence_of_three_quadratics : ∃ (f₁ f₂ f₃ : ℝ → ℝ),
  (∃ x₁, f₁ x₁ = 0) ∧
  (∃ x₂, f₂ x₂ = 0) ∧
  (∃ x₃, f₃ x₃ = 0) ∧
  (∀ x, (f₁ x + f₂ x) ≠ 0) ∧
  (∀ x, (f₁ x + f₃ x) ≠ 0) ∧
  (∀ x, (f₂ x + f₃ x) ≠ 0) ∧
  (∀ x, f₁ x = (x - 1)^2) ∧
  (∀ x, f₂ x = x^2) ∧
  (∀ x, f₃ x = (x - 2)^2) :=
by sorry

end NUMINAMATH_CALUDE_existence_of_three_quadratics_l855_85511


namespace NUMINAMATH_CALUDE_quadratic_equation_form_l855_85577

/-- 
Given a quadratic equation ax^2 + bx + c = 0,
if a = 3 and c = 1, then the equation is equivalent to 3x^2 + 1 = 0.
-/
theorem quadratic_equation_form (a b c : ℝ) : 
  a = 3 → c = 1 → (∃ x, a * x^2 + b * x + c = 0) ↔ (∃ x, 3 * x^2 + 1 = 0) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_form_l855_85577


namespace NUMINAMATH_CALUDE_find_x_in_set_l855_85521

theorem find_x_in_set (s : Finset ℝ) (x : ℝ) : 
  s = {8, 14, 20, 7, x, 16} →
  (Finset.sum s id) / (Finset.card s : ℝ) = 12 →
  x = 7 := by
sorry

end NUMINAMATH_CALUDE_find_x_in_set_l855_85521


namespace NUMINAMATH_CALUDE_quadratic_root_transformation_l855_85524

theorem quadratic_root_transformation (p q r u v : ℝ) : 
  (p * u^2 + q * u + r = 0) → 
  (p * v^2 + q * v + r = 0) → 
  ((p^2 * u + q)^2 - q * (p^2 * u + q) + p * r = 0) ∧
  ((p^2 * v + q)^2 - q * (p^2 * v + q) + p * r = 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_transformation_l855_85524


namespace NUMINAMATH_CALUDE_least_product_of_two_primes_above_ten_l855_85522

theorem least_product_of_two_primes_above_ten (p q : ℕ) : 
  Prime p → Prime q → p > 10 → q > 10 → p ≠ q → 
  ∀ r s : ℕ, Prime r → Prime s → r > 10 → s > 10 → r ≠ s → 
  p * q ≤ r * s → p * q = 143 := by sorry

end NUMINAMATH_CALUDE_least_product_of_two_primes_above_ten_l855_85522


namespace NUMINAMATH_CALUDE_parallel_planes_line_relations_l855_85572

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the parallel relation for planes
variable (parallel_planes : Plane → Plane → Prop)

-- Define the contained relation for lines in planes
variable (contained_in : Line → Plane → Prop)

-- Define the intersect relation for lines
variable (intersect : Line → Line → Prop)

-- Define the coplanar relation for lines
variable (coplanar : Line → Line → Prop)

-- Define the skew relation for lines
variable (skew : Line → Line → Prop)

-- Theorem statement
theorem parallel_planes_line_relations 
  (α β : Plane) (a b : Line)
  (h1 : parallel_planes α β)
  (h2 : contained_in a α)
  (h3 : contained_in b β) :
  (¬ intersect a b) ∧ (coplanar a b ∨ skew a b) :=
sorry

end NUMINAMATH_CALUDE_parallel_planes_line_relations_l855_85572


namespace NUMINAMATH_CALUDE_inequality_proof_l855_85575

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  a^4 + b^4 ≥ a^3 * b + a * b^3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l855_85575


namespace NUMINAMATH_CALUDE_car_speed_problem_l855_85515

theorem car_speed_problem (distance : ℝ) (time_difference : ℝ) (speed_difference : ℝ) :
  distance = 150 ∧
  time_difference = 2 ∧
  speed_difference = 10 →
  ∃ (speed_r : ℝ),
    speed_r > 0 ∧
    distance / speed_r - time_difference = distance / (speed_r + speed_difference) ∧
    speed_r = 25 :=
by sorry

end NUMINAMATH_CALUDE_car_speed_problem_l855_85515


namespace NUMINAMATH_CALUDE_digital_music_library_space_l855_85514

/-- Calculates the average megabytes per hour of music in a digital library, rounded to the nearest whole number -/
def averageMegabytesPerHour (days : ℕ) (totalMegabytes : ℕ) : ℕ :=
  let hoursPerDay : ℕ := 24
  let totalHours : ℕ := days * hoursPerDay
  let exactAverage : ℚ := totalMegabytes / totalHours
  (exactAverage + 1/2).floor.toNat

theorem digital_music_library_space (days : ℕ) (totalMegabytes : ℕ) 
  (h1 : days = 15) (h2 : totalMegabytes = 20400) :
  averageMegabytesPerHour days totalMegabytes = 57 := by
  sorry

end NUMINAMATH_CALUDE_digital_music_library_space_l855_85514


namespace NUMINAMATH_CALUDE_usual_baking_time_l855_85570

/-- Represents the time in hours for Matthew's cake-making process -/
structure BakingTime where
  assembly : ℝ
  baking : ℝ
  decorating : ℝ

/-- The total time for Matthew's cake-making process -/
def total_time (t : BakingTime) : ℝ := t.assembly + t.baking + t.decorating

/-- Represents the scenario when the oven fails -/
def oven_failure (normal : BakingTime) : BakingTime :=
  { assembly := normal.assembly,
    baking := 2 * normal.baking,
    decorating := normal.decorating }

theorem usual_baking_time :
  ∃ (normal : BakingTime),
    normal.assembly = 1 ∧
    normal.decorating = 1 ∧
    total_time (oven_failure normal) = 5 ∧
    normal.baking = 1.5 := by
  sorry

end NUMINAMATH_CALUDE_usual_baking_time_l855_85570


namespace NUMINAMATH_CALUDE_chess_tournament_games_l855_85586

/-- The number of games in a chess tournament where each player plays twice against every other player. -/
def tournament_games (n : ℕ) : ℕ := n * (n - 1)

/-- Theorem: In a chess tournament with 19 players, where each player plays twice against every other player, the total number of games played is 684. -/
theorem chess_tournament_games :
  tournament_games 19 = 342 ∧ 2 * tournament_games 19 = 684 := by
  sorry

#eval 2 * tournament_games 19

end NUMINAMATH_CALUDE_chess_tournament_games_l855_85586


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l855_85587

theorem complex_fraction_simplification : 
  (((12^4 + 484) * (24^4 + 484) * (36^4 + 484) * (48^4 + 484) * (60^4 + 484)) : ℚ) /
  ((6^4 + 484) * (18^4 + 484) * (30^4 + 484) * (42^4 + 484) * (54^4 + 484)) = 181 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l855_85587


namespace NUMINAMATH_CALUDE_parallel_lines_m_values_l855_85590

/-- Two lines are parallel if their slopes are equal -/
def are_parallel (a1 b1 c1 a2 b2 c2 : ℝ) : Prop :=
  (b1 ≠ 0 ∧ b2 ≠ 0 ∧ a1 / b1 = a2 / b2) ∨ (b1 = 0 ∧ b2 = 0)

/-- The main theorem -/
theorem parallel_lines_m_values (m : ℝ) :
  are_parallel 1 (1+m) (m-2) m 2 6 → m = -2 ∨ m = 1 := by
  sorry


end NUMINAMATH_CALUDE_parallel_lines_m_values_l855_85590


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_isosceles_triangle_perimeter_proof_l855_85504

/-- An isosceles triangle with side lengths 4 and 9 has a perimeter of 22. -/
theorem isosceles_triangle_perimeter : ℝ → ℝ → ℝ → Prop :=
  fun a b c =>
    (a = 4 ∨ a = 9) ∧ (b = 4 ∨ b = 9) ∧ (c = 4 ∨ c = 9) ∧  -- Side lengths are 4 or 9
    (a = b ∨ b = c ∨ a = c) ∧                              -- Isosceles condition
    (a + b > c ∧ b + c > a ∧ a + c > b) →                  -- Triangle inequality
    a + b + c = 22                                         -- Perimeter is 22

/-- Proof of the theorem -/
theorem isosceles_triangle_perimeter_proof : ∃ a b c, isosceles_triangle_perimeter a b c :=
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_isosceles_triangle_perimeter_proof_l855_85504


namespace NUMINAMATH_CALUDE_even_function_extension_l855_85552

/-- Given a real-valued function f that is even and defined as ln(x^2 - 2x + 2) for non-negative x,
    prove that f(x) = ln(x^2 + 2x + 2) for negative x -/
theorem even_function_extension (f : ℝ → ℝ) 
    (h_even : ∀ x, f x = f (-x))
    (h_non_neg : ∀ x ≥ 0, f x = Real.log (x^2 - 2*x + 2)) :
    ∀ x < 0, f x = Real.log (x^2 + 2*x + 2) := by
  sorry

end NUMINAMATH_CALUDE_even_function_extension_l855_85552


namespace NUMINAMATH_CALUDE_twelve_people_circular_arrangements_l855_85558

/-- The number of distinct circular arrangements of n people, considering rotational symmetry -/
def circularArrangements (n : ℕ) : ℕ := Nat.factorial (n - 1)

/-- Theorem: The number of distinct circular arrangements of 12 people, considering rotational symmetry, is equal to 11! -/
theorem twelve_people_circular_arrangements : 
  circularArrangements 12 = Nat.factorial 11 := by
  sorry

end NUMINAMATH_CALUDE_twelve_people_circular_arrangements_l855_85558


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l855_85549

theorem regular_polygon_sides (n : ℕ) (h : n > 0) :
  (360 : ℝ) / n = 12 → n = 30 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l855_85549


namespace NUMINAMATH_CALUDE_trapezoid_area_is_correct_l855_85581

/-- The area of a trapezoid bounded by y = 2x, y = 6, y = 3, and the y-axis -/
def trapezoidArea : ℝ := 6.75

/-- The line y = 2x -/
def line1 (x : ℝ) : ℝ := 2 * x

/-- The line y = 6 -/
def line2 : ℝ := 6

/-- The line y = 3 -/
def line3 : ℝ := 3

/-- The y-axis (x = 0) -/
def yAxis : ℝ := 0

theorem trapezoid_area_is_correct :
  trapezoidArea = 6.75 := by sorry

end NUMINAMATH_CALUDE_trapezoid_area_is_correct_l855_85581


namespace NUMINAMATH_CALUDE_projection_vector_l855_85509

/-- Given two plane vectors a and b, prove that the projection of b onto a is (-1, 2) -/
theorem projection_vector (a b : ℝ × ℝ) : 
  a = (1, -2) → b = (3, 4) → 
  (((a.1 * b.1 + a.2 * b.2) / (a.1^2 + a.2^2)) • a) = (-1, 2) := by
  sorry

end NUMINAMATH_CALUDE_projection_vector_l855_85509


namespace NUMINAMATH_CALUDE_floor_mod_equivalence_l855_85525

theorem floor_mod_equivalence (k : ℤ) (a b : ℝ) (h : k ≥ 2) :
  (∃ m : ℤ, a - b = m * k) ↔
  (∀ n : ℕ, n > 0 → ⌊a * n⌋ % k = ⌊b * n⌋ % k) :=
by sorry

end NUMINAMATH_CALUDE_floor_mod_equivalence_l855_85525


namespace NUMINAMATH_CALUDE_angle_half_in_third_quadrant_l855_85506

open Real

-- Define the first quadrant
def FirstQuadrant (α : ℝ) : Prop := 0 < α ∧ α < π / 2

-- Define the third quadrant
def ThirdQuadrant (θ : ℝ) : Prop := π < θ ∧ θ < 3 * π / 2

-- State the theorem
theorem angle_half_in_third_quadrant (α : ℝ) 
  (h1 : FirstQuadrant α) 
  (h2 : |cos (α / 2)| = -cos (α / 2)) : 
  ThirdQuadrant (α / 2) := by
  sorry

end NUMINAMATH_CALUDE_angle_half_in_third_quadrant_l855_85506


namespace NUMINAMATH_CALUDE_solution_set_inequality_1_solution_set_inequality_2_l855_85560

-- Part 1
theorem solution_set_inequality_1 : 
  {x : ℝ | (2 * x) / (x - 2) ≤ 1} = Set.Ici (-2) ∩ Set.Iio 2 := by sorry

-- Part 2
theorem solution_set_inequality_2 (a : ℝ) (ha : a > 0) :
  {x : ℝ | a * x^2 + 2 * x + 1 > 0} = 
    if a = 1 then 
      {x : ℝ | x ≠ -1}
    else if a > 1 then 
      Set.univ
    else 
      Set.Iic ((- 1 - Real.sqrt (1 - a)) / a) ∪ Set.Ioi ((- 1 + Real.sqrt (1 - a)) / a) := by sorry

end NUMINAMATH_CALUDE_solution_set_inequality_1_solution_set_inequality_2_l855_85560


namespace NUMINAMATH_CALUDE_evaluate_expression_l855_85568

theorem evaluate_expression : (-2 : ℤ) ^ (3 ^ 2) + 2 ^ (3 ^ 2) = 0 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l855_85568


namespace NUMINAMATH_CALUDE_complex_equation_sum_l855_85580

theorem complex_equation_sum (a b : ℝ) : 
  (3 * b : ℂ) + (2 * a - 2) * I = 1 - I → a + b = 5/6 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_sum_l855_85580


namespace NUMINAMATH_CALUDE_smallest_four_nine_divisible_by_four_and_nine_l855_85534

def is_four_nine_number (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ n.digits 10 → d = 4 ∨ d = 9

theorem smallest_four_nine_divisible_by_four_and_nine :
  ∃! n : ℕ, n > 0 ∧ is_four_nine_number n ∧ 4 ∣ n ∧ 9 ∣ n ∧
  ∀ m : ℕ, m > 0 → is_four_nine_number m → 4 ∣ m → 9 ∣ m → n ≤ m :=
by
  use 4944
  sorry

end NUMINAMATH_CALUDE_smallest_four_nine_divisible_by_four_and_nine_l855_85534


namespace NUMINAMATH_CALUDE_leibniz_theorem_l855_85516

/-- Leibniz's Theorem -/
theorem leibniz_theorem (A B C M : ℝ × ℝ) : 
  let G := ((A.1 + B.1 + C.1) / 3, (A.2 + B.2 + C.2) / 3)
  3 * dist M G ^ 2 = 
    dist M A ^ 2 + dist M B ^ 2 + dist M C ^ 2 - 
    (1/3) * (dist A B ^ 2 + dist B C ^ 2 + dist C A ^ 2) := by
  sorry

#check leibniz_theorem

end NUMINAMATH_CALUDE_leibniz_theorem_l855_85516


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l855_85585

/-- A geometric sequence with positive terms -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * q

/-- The first, third, and second terms form an arithmetic sequence -/
def arithmetic_condition (a : ℕ → ℝ) : Prop :=
  2 * ((1 / 2) * a 3) = a 1 + 2 * a 2

/-- All terms in the sequence are positive -/
def positive_terms (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a n > 0

theorem geometric_sequence_ratio (a : ℕ → ℝ) :
  geometric_sequence a →
  positive_terms a →
  arithmetic_condition a →
  (a 9 + a 10) / (a 7 + a 8) = 3 + 2 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l855_85585


namespace NUMINAMATH_CALUDE_sum_and_reciprocal_sum_l855_85584

theorem sum_and_reciprocal_sum (x : ℝ) (h : x ≠ 0) :
  x^2 + (1/x)^2 = 10.25 → x + (1/x) = 3.5 := by
  sorry

end NUMINAMATH_CALUDE_sum_and_reciprocal_sum_l855_85584


namespace NUMINAMATH_CALUDE_first_sat_score_l855_85528

/-- 
Given a 10% improvement from the first score to the second score, 
and a second score of 1100, prove that the first score must be 1000.
-/
theorem first_sat_score (second_score : ℝ) (improvement : ℝ) 
  (h1 : second_score = 1100)
  (h2 : improvement = 0.1)
  (h3 : second_score = (1 + improvement) * first_score) :
  first_score = 1000 := by
  sorry

end NUMINAMATH_CALUDE_first_sat_score_l855_85528


namespace NUMINAMATH_CALUDE_parabola_properties_l855_85569

/-- Parabola passing through given points with specific properties -/
theorem parabola_properties (a b : ℝ) (m : ℝ) : 
  (∀ x y, y = a * x^2 + b * x + 1) →
  (-2 = a * 1^2 + b * 1 + 1) →
  (13 = a * (-2)^2 + b * (-2) + 1) →
  (∃ y₁ y₂, y₁ = a * 5^2 + b * 5 + 1 ∧ 
            y₂ = a * m^2 + b * m + 1 ∧ 
            y₂ = 12 - y₁) →
  (a = 1 ∧ b = -4 ∧ m = -1) := by
sorry

end NUMINAMATH_CALUDE_parabola_properties_l855_85569


namespace NUMINAMATH_CALUDE_intersection_curve_length_theorem_l855_85597

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a cube in 3D space -/
structure Cube where
  vertex : Point3D
  edge_length : ℝ

/-- Represents a sphere in 3D space -/
structure Sphere where
  center : Point3D
  radius : ℝ

/-- The length of the curve formed by the intersection of a unit cube's surface
    and a sphere centered at one of the cube's vertices with radius 2√3/3 -/
def intersection_curve_length (c : Cube) (s : Sphere) : ℝ := sorry

/-- Main theorem statement -/
theorem intersection_curve_length_theorem (c : Cube) (s : Sphere) :
  c.edge_length = 1 ∧
  s.center = c.vertex ∧
  s.radius = 2 * Real.sqrt 3 / 3 →
  intersection_curve_length c s = 5 * Real.sqrt 3 * Real.pi / 6 := by
  sorry

end NUMINAMATH_CALUDE_intersection_curve_length_theorem_l855_85597


namespace NUMINAMATH_CALUDE_prob_same_first_last_pancake_l855_85561

/-- Represents the types of pancake fillings -/
inductive Filling
  | Meat
  | CottageCheese
  | Strawberry

/-- Represents a plate of pancakes -/
structure PlatePancakes where
  total : Nat
  meat : Nat
  cheese : Nat
  strawberry : Nat

/-- Calculates the probability of selecting the same filling for first and last pancake -/
def probSameFirstLast (plate : PlatePancakes) : Rat :=
  sorry

/-- Theorem stating the probability of selecting the same filling for first and last pancake -/
theorem prob_same_first_last_pancake (plate : PlatePancakes) :
  plate.total = 10 ∧ plate.meat = 2 ∧ plate.cheese = 3 ∧ plate.strawberry = 5 →
  probSameFirstLast plate = 14 / 45 := by
  sorry

end NUMINAMATH_CALUDE_prob_same_first_last_pancake_l855_85561


namespace NUMINAMATH_CALUDE_mickey_minnie_horse_difference_l855_85518

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- The number of horses Minnie mounts per day -/
def minnie_horses_per_day : ℕ := days_in_week + 3

/-- The number of horses Mickey mounts per week -/
def mickey_horses_per_week : ℕ := 98

/-- The number of horses Mickey mounts per day -/
def mickey_horses_per_day : ℕ := mickey_horses_per_week / days_in_week

/-- Mickey mounts some less than twice as many horses per day as Minnie -/
axiom mickey_less_than_twice_minnie : mickey_horses_per_day < 2 * minnie_horses_per_day

theorem mickey_minnie_horse_difference :
  2 * minnie_horses_per_day - mickey_horses_per_day = 6 := by
  sorry

end NUMINAMATH_CALUDE_mickey_minnie_horse_difference_l855_85518


namespace NUMINAMATH_CALUDE_sticker_difference_l855_85596

theorem sticker_difference (initial_stickers : ℕ) : 
  initial_stickers > 0 →
  (initial_stickers - 15 : ℤ) / (initial_stickers + 18 : ℤ) = 2 / 5 →
  (initial_stickers + 18) - (initial_stickers - 15) = 33 := by
  sorry

#check sticker_difference

end NUMINAMATH_CALUDE_sticker_difference_l855_85596


namespace NUMINAMATH_CALUDE_sqrt_two_equals_two_to_one_sixth_l855_85539

theorem sqrt_two_equals_two_to_one_sixth : ∃ (x : ℝ), x > 0 ∧ x^2 = 2 ∧ x = 2^(1/6) := by sorry

end NUMINAMATH_CALUDE_sqrt_two_equals_two_to_one_sixth_l855_85539


namespace NUMINAMATH_CALUDE_power_div_reciprocal_l855_85529

theorem power_div_reciprocal (a : ℝ) (h : a ≠ 0) : a^2 / (1/a) = a^3 := by
  sorry

end NUMINAMATH_CALUDE_power_div_reciprocal_l855_85529


namespace NUMINAMATH_CALUDE_existence_of_special_number_l855_85566

def sumOfDigits (n : ℕ) : ℕ :=
  sorry

def isComposedOf1And0 (n : ℕ) : Prop :=
  sorry

theorem existence_of_special_number :
  ∀ m : ℕ, ∃ n : ℕ,
    isComposedOf1And0 n ∧
    sumOfDigits n = m ∧
    sumOfDigits (n^2) = m^2 := by
  sorry

end NUMINAMATH_CALUDE_existence_of_special_number_l855_85566


namespace NUMINAMATH_CALUDE_line_intercept_at_10_l855_85595

/-- A line passing through two points (x₁, y₁) and (x₂, y₂) -/
structure Line where
  x₁ : ℝ
  y₁ : ℝ
  x₂ : ℝ
  y₂ : ℝ

/-- The x-intercept of a line -/
def xIntercept (l : Line) : ℝ := sorry

theorem line_intercept_at_10 : 
  let l : Line := { x₁ := 7, y₁ := 3, x₂ := 3, y₂ := 7 }
  xIntercept l = 10 := by sorry

end NUMINAMATH_CALUDE_line_intercept_at_10_l855_85595


namespace NUMINAMATH_CALUDE_artwork_area_l855_85500

/-- Given a rectangular artwork with a frame, calculate its area -/
theorem artwork_area (outer_width outer_height frame_width_top frame_width_side : ℕ) 
  (h1 : outer_width = 100)
  (h2 : outer_height = 140)
  (h3 : frame_width_top = 15)
  (h4 : frame_width_side = 20) :
  (outer_width - 2 * frame_width_side) * (outer_height - 2 * frame_width_top) = 6600 := by
  sorry

#check artwork_area

end NUMINAMATH_CALUDE_artwork_area_l855_85500


namespace NUMINAMATH_CALUDE_trig_expression_equals_five_fourths_l855_85573

theorem trig_expression_equals_five_fourths :
  2 * (Real.cos (5 * π / 16))^6 + 2 * (Real.sin (11 * π / 16))^6 + (3 * Real.sqrt 2) / 8 = 5 / 4 := by
  sorry

end NUMINAMATH_CALUDE_trig_expression_equals_five_fourths_l855_85573


namespace NUMINAMATH_CALUDE_crackers_per_box_l855_85576

theorem crackers_per_box (darren_boxes calvin_boxes total_crackers : ℕ) : 
  darren_boxes = 4 →
  calvin_boxes = 2 * darren_boxes - 1 →
  total_crackers = 264 →
  (darren_boxes + calvin_boxes) * (total_crackers / (darren_boxes + calvin_boxes)) = total_crackers →
  total_crackers / (darren_boxes + calvin_boxes) = 24 := by
sorry

end NUMINAMATH_CALUDE_crackers_per_box_l855_85576


namespace NUMINAMATH_CALUDE_cubic_polynomials_common_roots_l855_85557

theorem cubic_polynomials_common_roots (a b : ℝ) :
  (∃ r s : ℝ, r ≠ s ∧
    r^3 + a*r^2 + 10*r + 3 = 0 ∧
    r^3 + b*r^2 + 21*r + 12 = 0 ∧
    s^3 + a*s^2 + 10*s + 3 = 0 ∧
    s^3 + b*s^2 + 21*s + 12 = 0) →
  a = 9 ∧ b = 10 := by
sorry

end NUMINAMATH_CALUDE_cubic_polynomials_common_roots_l855_85557


namespace NUMINAMATH_CALUDE_circle_area_circumference_ratio_l855_85546

theorem circle_area_circumference_ratio (r₁ r₂ : ℝ) (h : r₁ > 0 ∧ r₂ > 0) :
  (π * r₁^2) / (π * r₂^2) = 16 / 25 →
  (2 * π * r₁) / (2 * π * r₂) = 4 / 5 := by
sorry

end NUMINAMATH_CALUDE_circle_area_circumference_ratio_l855_85546


namespace NUMINAMATH_CALUDE_min_value_theorem_l855_85589

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2*x + y = 3) :
  (1/x + 2/y) ≥ 8/3 :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l855_85589


namespace NUMINAMATH_CALUDE_negation_equivalence_l855_85565

-- Define the universe of discourse
variable (U : Type)

-- Define predicates for "man" and "tall"
variable (man : U → Prop)
variable (tall : U → Prop)

-- Define the original statement "all men are tall"
def all_men_are_tall : Prop := ∀ x : U, man x → tall x

-- Define the negation of the original statement
def negation_of_all_men_are_tall : Prop := ¬(∀ x : U, man x → tall x)

-- Define "some men are short"
def some_men_are_short : Prop := ∃ x : U, man x ∧ ¬(tall x)

-- Theorem stating that the negation is equivalent to "some men are short"
theorem negation_equivalence : 
  negation_of_all_men_are_tall U man tall ↔ some_men_are_short U man tall :=
sorry

end NUMINAMATH_CALUDE_negation_equivalence_l855_85565


namespace NUMINAMATH_CALUDE_triangle_properties_l855_85559

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ)  -- Angles
  (a b c : ℝ)  -- Sides

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.b^2 = t.a * t.c ∧ Real.cos (t.A - t.C) = Real.cos t.B + 1/2

-- Theorem statement
theorem triangle_properties (t : Triangle) (h : triangle_conditions t) :
  t.B = π/3 ∧ t.A = π/3 ∧
  ∀ (CD : ℝ), CD = 6 → 
    (∃ (max_perimeter : ℝ), max_perimeter = 4 * Real.sqrt 3 + 6 ∧
      ∀ (perimeter : ℝ), perimeter ≤ max_perimeter) :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l855_85559


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_sum_l855_85537

/-- An arithmetic-geometric sequence -/
structure ArithmeticGeometricSequence where
  a : ℕ → ℝ

/-- Sum of the first n terms of a sequence -/
def SumOfFirstNTerms (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (Finset.range n).sum a

theorem arithmetic_geometric_sequence_sum (a : ArithmeticGeometricSequence) :
  let S := SumOfFirstNTerms a.a
  S 2 = 3 ∧ S 4 = 15 → S 6 = 63 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_sum_l855_85537


namespace NUMINAMATH_CALUDE_jane_baking_time_l855_85556

/-- Represents the time it takes Jane to bake cakes individually -/
def jane_time : ℝ := 4

/-- Represents the time it takes Roy to bake cakes individually -/
def roy_time : ℝ := 5

/-- The time Jane and Roy work together -/
def joint_work_time : ℝ := 2

/-- The time Jane works alone after Roy leaves -/
def jane_solo_time : ℝ := 0.4

/-- The theorem stating that Jane's individual baking time is 4 hours -/
theorem jane_baking_time :
  (joint_work_time * (1 / jane_time + 1 / roy_time)) + 
  (jane_solo_time * (1 / jane_time)) = 1 :=
sorry

end NUMINAMATH_CALUDE_jane_baking_time_l855_85556
