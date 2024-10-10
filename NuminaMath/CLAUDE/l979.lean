import Mathlib

namespace article_cost_l979_97919

/-- The cost of an article given specific selling prices and gains -/
theorem article_cost (sp1 sp2 : ℝ) (gain_increase : ℝ) :
  sp1 = 500 →
  sp2 = 570 →
  gain_increase = 0.15 →
  ∃ (cost gain : ℝ),
    cost + gain = sp1 ∧
    cost + gain * (1 + gain_increase) = sp2 ∧
    cost = 100 / 3 :=
sorry

end article_cost_l979_97919


namespace least_sum_of_exponents_for_1000_l979_97915

def is_sum_of_distinct_powers_of_two (n : ℕ) (exponents : List ℕ) : Prop :=
  n = (exponents.map (λ e => 2^e)).sum ∧ exponents.Nodup

theorem least_sum_of_exponents_for_1000 :
  ∀ exponents : List ℕ,
    is_sum_of_distinct_powers_of_two 1000 exponents →
    exponents.length ≥ 3 →
    exponents.sum ≥ 38 :=
by sorry

end least_sum_of_exponents_for_1000_l979_97915


namespace wells_garden_rows_l979_97973

/-- The number of rows in Mr. Wells' garden -/
def num_rows : ℕ := 50

/-- The number of flowers in each row -/
def flowers_per_row : ℕ := 400

/-- The percentage of flowers cut -/
def cut_percentage : ℚ := 60 / 100

/-- The number of flowers remaining after cutting -/
def remaining_flowers : ℕ := 8000

/-- Theorem stating that the number of rows is correct given the conditions -/
theorem wells_garden_rows :
  num_rows * flowers_per_row * (1 - cut_percentage) = remaining_flowers :=
sorry

end wells_garden_rows_l979_97973


namespace monotonically_decreasing_quadratic_l979_97944

/-- A function f is monotonically decreasing on an interval [a, b] if for all x, y in [a, b] with x ≤ y, we have f(x) ≥ f(y) -/
def MonotonicallyDecreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x ≤ y ∧ y ≤ b → f x ≥ f y

/-- The theorem statement -/
theorem monotonically_decreasing_quadratic (a : ℝ) :
  MonotonicallyDecreasing (fun x => a * x^2 - 2 * x + 1) 1 10 ↔ a ≤ 1/10 :=
sorry

end monotonically_decreasing_quadratic_l979_97944


namespace star_properties_l979_97974

/-- The operation "*" for any two numbers -/
noncomputable def star (m : ℝ) (x y : ℝ) : ℝ := (4 * x * y) / (m * x + 3 * y)

/-- Theorem stating the properties of the "*" operation -/
theorem star_properties :
  ∃ m : ℝ, (star m 1 2 = 1) ∧ (m = 2) ∧ (star m 3 12 = 24/7) := by
  sorry

end star_properties_l979_97974


namespace ratio_of_a_to_c_l979_97958

theorem ratio_of_a_to_c (a b c d : ℚ) 
  (hab : a / b = 5 / 2)
  (hcd : c / d = 7 / 3)
  (hdb : d / b = 5 / 4) :
  a / c = 6 / 7 := by
  sorry

end ratio_of_a_to_c_l979_97958


namespace defective_pens_l979_97902

theorem defective_pens (total_pens : ℕ) (prob_non_defective : ℚ) (defective_pens : ℕ) : 
  total_pens = 9 →
  prob_non_defective = 5 / 12 →
  (total_pens - defective_pens : ℚ) / total_pens * ((total_pens - defective_pens - 1) : ℚ) / (total_pens - 1) = prob_non_defective →
  defective_pens = 3 := by
sorry

end defective_pens_l979_97902


namespace max_product_with_linear_constraint_max_product_achieved_l979_97904

theorem max_product_with_linear_constraint (a b : ℝ) :
  a > 0 → b > 0 → 6 * a + 5 * b = 75 → a * b ≤ 46.875 := by
  sorry

theorem max_product_achieved (a b : ℝ) :
  a > 0 → b > 0 → 6 * a + 5 * b = 75 → a * b = 46.875 → a = 75 / 11 ∧ b = 90 / 11 := by
  sorry

end max_product_with_linear_constraint_max_product_achieved_l979_97904


namespace product_from_lcm_and_gcd_l979_97967

theorem product_from_lcm_and_gcd (a b : ℕ+) 
  (h1 : Nat.lcm a b = 48) 
  (h2 : Nat.gcd a b = 8) : 
  a * b = 384 := by
sorry

end product_from_lcm_and_gcd_l979_97967


namespace visitors_scientific_notation_l979_97936

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem visitors_scientific_notation :
  toScientificNotation 564200 = ScientificNotation.mk 5.642 5 (by sorry) :=
sorry

end visitors_scientific_notation_l979_97936


namespace max_areas_formula_l979_97940

/-- Represents a circular disk configuration -/
structure DiskConfiguration where
  n : ℕ
  radii_count : ℕ
  has_secant : Bool
  has_chord : Bool
  chord_intersects_secant : Bool

/-- The maximum number of non-overlapping areas in a disk configuration -/
def max_areas (config : DiskConfiguration) : ℕ :=
  sorry

/-- Theorem stating the maximum number of non-overlapping areas -/
theorem max_areas_formula (config : DiskConfiguration) 
  (h1 : config.n > 0)
  (h2 : config.radii_count = 2 * config.n)
  (h3 : config.has_secant = true)
  (h4 : config.has_chord = true)
  (h5 : config.chord_intersects_secant = false) :
  max_areas config = 4 * config.n - 1 :=
sorry

end max_areas_formula_l979_97940


namespace gcd_50403_40302_l979_97991

theorem gcd_50403_40302 : Nat.gcd 50403 40302 = 1 := by
  sorry

end gcd_50403_40302_l979_97991


namespace exp_gt_one_plus_x_l979_97910

theorem exp_gt_one_plus_x (x : ℝ) (h : x ≠ 0) : Real.exp x > 1 + x := by
  sorry

end exp_gt_one_plus_x_l979_97910


namespace convex_figure_inequalities_isoperimetric_inequality_l979_97921

/-- A convex figure in a plane -/
class ConvexFigure where
  -- Perimeter of the convex figure
  perimeter : ℝ
  -- Area of the convex figure
  area : ℝ
  -- Radius of the inscribed circle
  inscribed_radius : ℝ
  -- Radius of the circumscribed circle
  circumscribed_radius : ℝ
  -- Assumption that the figure is convex
  convex : True

/-- The main theorem stating the inequalities for convex figures -/
theorem convex_figure_inequalities (F : ConvexFigure) :
  let P := F.perimeter
  let S := F.area
  let r := F.inscribed_radius
  let R := F.circumscribed_radius
  (P^2 - 4 * Real.pi * S ≥ (P - 2 * Real.pi * r)^2) ∧
  (P^2 - 4 * Real.pi * S ≥ (2 * Real.pi * R - P)^2) := by
  sorry

/-- Corollary: isoperimetric inequality for planar convex figures -/
theorem isoperimetric_inequality (F : ConvexFigure) :
  F.area / F.perimeter^2 ≤ 1 / (4 * Real.pi) := by
  sorry

end convex_figure_inequalities_isoperimetric_inequality_l979_97921


namespace batsman_highest_score_l979_97977

-- Define the given conditions
def total_innings : ℕ := 46
def average : ℚ := 62
def score_difference : ℕ := 150
def average_excluding_extremes : ℚ := 58

-- Define the theorem
theorem batsman_highest_score :
  ∃ (highest lowest : ℕ),
    (highest - lowest = score_difference) ∧
    (highest + lowest = total_innings * average - (total_innings - 2) * average_excluding_extremes) ∧
    (highest = 225) := by
  sorry

end batsman_highest_score_l979_97977


namespace pizza_slices_l979_97961

theorem pizza_slices (total_slices : ℕ) (num_pizzas : ℕ) (slices_per_pizza : ℕ) :
  total_slices = 16 →
  num_pizzas = 2 →
  total_slices = num_pizzas * slices_per_pizza →
  slices_per_pizza = 8 := by
sorry

end pizza_slices_l979_97961


namespace real_number_in_set_l979_97932

theorem real_number_in_set (a : ℝ) : a ∈ ({a^2 - a, 0} : Set ℝ) → a = 2 := by
  sorry

end real_number_in_set_l979_97932


namespace zero_in_set_implies_m_equals_two_l979_97964

theorem zero_in_set_implies_m_equals_two (m : ℝ) :
  0 ∈ ({m, m^2 - 2*m} : Set ℝ) → m = 2 := by
  sorry

end zero_in_set_implies_m_equals_two_l979_97964


namespace complex_magnitude_l979_97998

theorem complex_magnitude (z : ℂ) (h : (1 - 2*I)*z = 3 + I) : Complex.abs z = Real.sqrt 2 := by
  sorry

end complex_magnitude_l979_97998


namespace increasing_quadratic_function_l979_97999

/-- The function f(x) = x^2 - 2ax is increasing on [1, +∞) if and only if a ≤ 1 -/
theorem increasing_quadratic_function (a : ℝ) :
  (∀ x ≥ 1, Monotone (fun x => x^2 - 2*a*x)) ↔ a ≤ 1 := by
  sorry

end increasing_quadratic_function_l979_97999


namespace goldbach_extension_l979_97939

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

def has_two_prime_pairs (N : ℕ) : Prop :=
  ∃ (p₁ q₁ p₂ q₂ : ℕ),
    is_prime p₁ ∧ is_prime q₁ ∧ is_prime p₂ ∧ is_prime q₂ ∧
    p₁ + q₁ = N ∧ p₂ + q₂ = N ∧
    (p₁ ≠ p₂ ∨ q₁ ≠ q₂) ∧
    ∀ (p q : ℕ), is_prime p → is_prime q → p + q = N →
      ((p = p₁ ∧ q = q₁) ∨ (p = p₂ ∧ q = q₂) ∨ (p = q₁ ∧ q = p₁) ∨ (p = q₂ ∧ q = p₂))

theorem goldbach_extension :
  ∀ N : ℕ, N ≥ 10 →
    (N = 10 ↔ (N % 2 = 0 ∧ has_two_prime_pairs N ∧
      ∀ M : ℕ, M < N → M % 2 = 0 → M > 2 → ¬has_two_prime_pairs M)) :=
sorry

end goldbach_extension_l979_97939


namespace ted_eats_four_cookies_l979_97900

-- Define the problem parameters
def days : ℕ := 6
def trays_per_day : ℕ := 2
def cookies_per_tray : ℕ := 12
def frank_daily_consumption : ℕ := 1
def cookies_left : ℕ := 134

-- Define the function to calculate Ted's consumption
def ted_consumption : ℕ :=
  days * trays_per_day * cookies_per_tray - 
  days * frank_daily_consumption - 
  cookies_left

-- Theorem statement
theorem ted_eats_four_cookies : ted_consumption = 4 := by
  sorry

end ted_eats_four_cookies_l979_97900


namespace equation_solution_l979_97990

theorem equation_solution : ∃ x : ℝ, (10 - 2*x)^2 = 4*x^2 + 16 ∧ x = 21/10 := by
  sorry

end equation_solution_l979_97990


namespace union_of_sets_l979_97996

theorem union_of_sets (S R : Set ℕ) : 
  S = {1} → R = {1, 2} → S ∪ R = {1, 2} := by sorry

end union_of_sets_l979_97996


namespace count_rectangles_3x6_grid_l979_97978

/-- The number of rectangles in a 3 × 6 grid with vertices at grid points -/
def num_rectangles : ℕ :=
  let horizontal_lines := 4
  let vertical_lines := 7
  let horizontal_vertical_rectangles := (horizontal_lines.choose 2) * (vertical_lines.choose 2)
  let diagonal_sqrt2 := 5 * 2
  let diagonal_2sqrt2 := 4 * 2
  let diagonal_sqrt5 := 4 * 2
  horizontal_vertical_rectangles + diagonal_sqrt2 + diagonal_2sqrt2 + diagonal_sqrt5

theorem count_rectangles_3x6_grid :
  num_rectangles = 152 :=
sorry

end count_rectangles_3x6_grid_l979_97978


namespace sun_valley_combined_population_sun_valley_combined_population_proof_l979_97909

/-- Proves that the combined population of Sun City and Valley City is 41550 given the conditions in the problem. -/
theorem sun_valley_combined_population : ℕ → ℕ → ℕ → ℕ → ℕ → Prop :=
  fun willowdale roseville sun x valley =>
    willowdale = 2000 ∧
    roseville = 3 * willowdale - 500 ∧
    sun = 2 * roseville + 1000 ∧
    x = (6 * sun) / 10 ∧
    valley = 4 * x + 750 →
    sun + valley = 41550

/-- Proof of the theorem -/
theorem sun_valley_combined_population_proof : 
  ∃ (willowdale roseville sun x valley : ℕ), 
    sun_valley_combined_population willowdale roseville sun x valley :=
by
  sorry

#check sun_valley_combined_population
#check sun_valley_combined_population_proof

end sun_valley_combined_population_sun_valley_combined_population_proof_l979_97909


namespace octal_123_equals_decimal_83_l979_97943

/-- Converts an octal digit to its decimal equivalent -/
def octal_to_decimal (digit : ℕ) : ℕ := digit

/-- Represents an octal number as a list of natural numbers -/
def octal_number : List ℕ := [1, 2, 3]

/-- Converts an octal number to its decimal equivalent -/
def octal_to_decimal_conversion (octal : List ℕ) : ℕ :=
  octal.enum.foldr (fun (i, digit) acc => acc + octal_to_decimal digit * 8^i) 0

theorem octal_123_equals_decimal_83 :
  octal_to_decimal_conversion octal_number = 83 := by
  sorry

end octal_123_equals_decimal_83_l979_97943


namespace total_packs_bought_l979_97933

/-- The number of index card packs John buys for each student -/
def packs_per_student : ℕ := 3

/-- The number of students in the first class -/
def class1_students : ℕ := 20

/-- The number of students in the second class -/
def class2_students : ℕ := 25

/-- The number of students in the third class -/
def class3_students : ℕ := 18

/-- The number of students in the fourth class -/
def class4_students : ℕ := 22

/-- The number of students in the fifth class -/
def class5_students : ℕ := 15

/-- The total number of students across all classes -/
def total_students : ℕ := class1_students + class2_students + class3_students + class4_students + class5_students

/-- Theorem: The total number of index card packs bought by John is 300 -/
theorem total_packs_bought : packs_per_student * total_students = 300 := by
  sorry

end total_packs_bought_l979_97933


namespace A_intersect_B_l979_97913

def A : Set ℝ := {-3, -2, 0, 2}
def B : Set ℝ := {x : ℝ | |x - 1| < 2}

theorem A_intersect_B : A ∩ B = {0, 2} := by sorry

end A_intersect_B_l979_97913


namespace ellipse_condition_l979_97994

/-- 
Given the equation x^2 + 9y^2 - 6x + 18y = k, 
this theorem states that it represents a non-degenerate ellipse 
if and only if k > -18.
-/
theorem ellipse_condition (k : ℝ) : 
  (∀ x y : ℝ, x^2 + 9*y^2 - 6*x + 18*y = k) → 
  (∃ a b h1 h2 : ℝ, a > 0 ∧ b > 0 ∧ 
    ∀ x y : ℝ, (x - h1)^2 / a^2 + (y - h2)^2 / b^2 = 1) ↔ 
  k > -18 :=
sorry

end ellipse_condition_l979_97994


namespace business_subscription_problem_l979_97951

/-- Proves that given the conditions of the business subscription problem, 
    the total amount subscribed is 50,000 Rs. -/
theorem business_subscription_problem 
  (a b c : ℕ) 
  (h1 : a = b + 4000)
  (h2 : b = c + 5000)
  (total_profit : ℕ)
  (h3 : total_profit = 36000)
  (a_profit : ℕ)
  (h4 : a_profit = 15120)
  (h5 : a_profit * (a + b + c) = a * total_profit) :
  a + b + c = 50000 := by
  sorry

end business_subscription_problem_l979_97951


namespace algebraic_expression_simplification_l979_97988

theorem algebraic_expression_simplification (x : ℝ) :
  x = 2 * Real.cos (45 * π / 180) + 1 →
  (1 / (x - 1) - (x - 3) / (x^2 - 2*x + 1)) / (2 / (x - 1)) = Real.sqrt 2 / 2 :=
by sorry

end algebraic_expression_simplification_l979_97988


namespace b_share_is_302_l979_97922

/-- Given a division of money among five people A, B, C, D, and E, 
    prove that B's share is 302 rupees. -/
theorem b_share_is_302 
  (total : ℕ) 
  (share_a share_b share_c share_d share_e : ℕ) 
  (h_total : total = 1540)
  (h_a : share_a = share_b + 40)
  (h_c : share_c = share_a + 30)
  (h_d : share_d = share_b - 50)
  (h_e : share_e = share_d + 20)
  (h_sum : share_a + share_b + share_c + share_d + share_e = total) : 
  share_b = 302 := by
  sorry


end b_share_is_302_l979_97922


namespace sales_growth_equation_correct_l979_97950

/-- Represents the sales growth scenario of a product over two years -/
structure SalesGrowth where
  initialSales : ℝ
  salesIncrease : ℝ
  growthRate : ℝ

/-- The equation for the sales growth scenario is correct -/
def isCorrectEquation (sg : SalesGrowth) : Prop :=
  20 * (1 + sg.growthRate)^2 - 20 = 3.12

/-- The given sales data matches the equation -/
theorem sales_growth_equation_correct (sg : SalesGrowth) 
  (h1 : sg.initialSales = 200000)
  (h2 : sg.salesIncrease = 31200) :
  isCorrectEquation sg := by
  sorry

end sales_growth_equation_correct_l979_97950


namespace test_questions_count_l979_97911

theorem test_questions_count : ∀ (total_questions : ℕ),
  (total_questions % 4 = 0) →
  (20 : ℚ) / total_questions > (60 : ℚ) / 100 →
  (20 : ℚ) / total_questions < (70 : ℚ) / 100 →
  total_questions = 32 :=
by
  sorry

end test_questions_count_l979_97911


namespace symmetric_sequence_theorem_l979_97965

/-- A symmetric sequence of 7 terms -/
def SymmetricSequence (b : Fin 7 → ℝ) : Prop :=
  ∀ k, k < 7 → b k = b (6 - k)

/-- The first 4 terms form an arithmetic sequence -/
def ArithmeticSequence (b : Fin 7 → ℝ) : Prop :=
  ∃ d : ℝ, ∀ k, k < 3 → b (k + 1) - b k = d

/-- The theorem statement -/
theorem symmetric_sequence_theorem (b : Fin 7 → ℝ) 
  (h_symmetric : SymmetricSequence b)
  (h_arithmetic : ArithmeticSequence b)
  (h_b1 : b 0 = 2)
  (h_sum : b 1 + b 3 = 16) :
  b = ![2, 5, 8, 11, 8, 5, 2] := by
  sorry

end symmetric_sequence_theorem_l979_97965


namespace emmanuel_jelly_beans_l979_97946

theorem emmanuel_jelly_beans (total : ℕ) (thomas_percent : ℚ) (barry_ratio : ℕ) (emmanuel_ratio : ℕ) : 
  total = 200 →
  thomas_percent = 1/10 →
  barry_ratio = 4 →
  emmanuel_ratio = 5 →
  (emmanuel_ratio * (total - thomas_percent * total)) / (barry_ratio + emmanuel_ratio) = 100 := by
sorry

end emmanuel_jelly_beans_l979_97946


namespace digit_puzzle_solution_l979_97981

theorem digit_puzzle_solution :
  ∀ (E F G H : ℕ),
  (E < 10 ∧ F < 10 ∧ G < 10 ∧ H < 10) →
  (10 * E + F) + (10 * G + E) = 10 * H + E →
  (10 * E + F) - (10 * G + E) = E →
  H = 0 := by
  sorry

end digit_puzzle_solution_l979_97981


namespace concert_attendance_problem_l979_97980

theorem concert_attendance_problem (total_attendance : ℕ) (adult_cost child_cost total_receipts : ℚ) 
  (h1 : total_attendance = 578)
  (h2 : adult_cost = 2)
  (h3 : child_cost = 3/2)
  (h4 : total_receipts = 985) :
  ∃ (adults children : ℕ),
    adults + children = total_attendance ∧
    adult_cost * adults + child_cost * children = total_receipts ∧
    adults = 236 := by
  sorry

end concert_attendance_problem_l979_97980


namespace tv_watch_time_two_weeks_l979_97976

/-- Calculates the total hours of TV watched in two weeks -/
def tvWatchTimeInTwoWeeks (minutesPerDay : ℕ) (daysPerWeek : ℕ) : ℚ :=
  let minutesPerWeek : ℕ := minutesPerDay * daysPerWeek
  let hoursPerWeek : ℚ := minutesPerWeek / 60
  hoursPerWeek * 2

/-- Theorem: Children watching 45 minutes of TV per day, 4 days a week, watch 6 hours in two weeks -/
theorem tv_watch_time_two_weeks :
  tvWatchTimeInTwoWeeks 45 4 = 6 := by
  sorry

end tv_watch_time_two_weeks_l979_97976


namespace milk_container_problem_l979_97930

theorem milk_container_problem (A : ℝ) 
  (hB : ℝ) (hC : ℝ) 
  (hB_initial : hB = 0.375 * A) 
  (hC_initial : hC = A - hB) 
  (h_equal_after_transfer : hB + 150 = hC - 150) : A = 1200 :=
by sorry

end milk_container_problem_l979_97930


namespace cubic_factor_identity_l979_97987

theorem cubic_factor_identity (a b c : ℝ) :
  ((a^3 - b^3)^3 + (b^3 - c^3)^3 + (c^3 - a^3)^3) / ((a - b)^3 + (b - c)^3 + (c - a)^3) = 
  (a^2 + a*b + b^2) * (b^2 + b*c + c^2) * (c^2 + c*a + a^2) :=
by sorry

end cubic_factor_identity_l979_97987


namespace triangle_area_approx_l979_97945

/-- The area of a triangle with sides 30, 26, and 10 is approximately 126.72 -/
theorem triangle_area_approx : ∃ (area : ℝ), 
  let a : ℝ := 30
  let b : ℝ := 26
  let c : ℝ := 10
  let s : ℝ := (a + b + c) / 2
  area = Real.sqrt (s * (s - a) * (s - b) * (s - c)) ∧ 
  126.71 < area ∧ area < 126.73 :=
by sorry

end triangle_area_approx_l979_97945


namespace deer_per_hunting_wolf_l979_97908

theorem deer_per_hunting_wolf (hunting_wolves : ℕ) (additional_wolves : ℕ) 
  (meat_per_wolf_per_day : ℕ) (days_between_hunts : ℕ) (meat_per_deer : ℕ) : 
  hunting_wolves = 4 →
  additional_wolves = 16 →
  meat_per_wolf_per_day = 8 →
  days_between_hunts = 5 →
  meat_per_deer = 200 →
  (hunting_wolves + additional_wolves) * meat_per_wolf_per_day * days_between_hunts / 
  (meat_per_deer * hunting_wolves) = 1 := by
sorry

end deer_per_hunting_wolf_l979_97908


namespace monic_quadratic_with_complex_root_l979_97938

theorem monic_quadratic_with_complex_root :
  ∃ (a b : ℝ), ∀ (x : ℂ),
    (x^2 + a*x + b = 0 ↔ x = -3 - Complex.I * Real.sqrt 7 ∨ x = -3 + Complex.I * Real.sqrt 7) ∧
    (a = 6 ∧ b = 16) := by
  sorry

end monic_quadratic_with_complex_root_l979_97938


namespace lcm_gcd_problem_l979_97918

theorem lcm_gcd_problem (a b : ℕ+) (h1 : Nat.lcm a b = 7700) (h2 : Nat.gcd a b = 11) (h3 : a = 308) : b = 275 := by
  sorry

end lcm_gcd_problem_l979_97918


namespace geometric_arithmetic_sequence_l979_97949

theorem geometric_arithmetic_sequence (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a (n + 1) = a n * q) →  -- geometric sequence condition
  a 1 = 4 →                     -- first term condition
  q ≠ 1 →                       -- common ratio condition
  2 * a 5 = 4 * a 1 - 2 * a 3 → -- arithmetic sequence condition
  q = -1 := by
sorry

end geometric_arithmetic_sequence_l979_97949


namespace problem_one_l979_97916

theorem problem_one : Real.sqrt 12 + (-2024)^0 - 4 * Real.sin (60 * π / 180) = 1 := by
  sorry

end problem_one_l979_97916


namespace tree_count_after_planting_l979_97970

theorem tree_count_after_planting (road_length : ℕ) (original_spacing : ℕ) (additional_trees : ℕ) : 
  road_length = 7200 → 
  original_spacing = 120 → 
  additional_trees = 5 → 
  (road_length / original_spacing * (additional_trees + 1) + 1) * 2 = 722 := by
sorry

end tree_count_after_planting_l979_97970


namespace sum_of_reciprocal_F_powers_of_two_l979_97995

-- Define the function F recursively
def F : ℕ → ℚ
  | 0 => 0
  | 1 => 3/2
  | (n+2) => 5/2 * F (n+1) - F n

-- Define the series
def series_sum : ℕ → ℚ
  | 0 => 1 / F (2^0)
  | (n+1) => series_sum n + 1 / F (2^(n+1))

-- State the theorem
theorem sum_of_reciprocal_F_powers_of_two :
  ∃ (L : ℚ), L = 1 ∧ ∀ ε > 0, ∃ N, ∀ n ≥ N, |series_sum n - L| < ε :=
sorry

end sum_of_reciprocal_F_powers_of_two_l979_97995


namespace major_axis_length_is_eight_l979_97952

/-- An ellipse with given properties -/
structure Ellipse where
  /-- The ellipse is tangent to both x-axis and y-axis -/
  tangent_to_axes : Bool
  /-- The first focus of the ellipse -/
  focus1 : ℝ × ℝ
  /-- The second focus of the ellipse -/
  focus2 : ℝ × ℝ

/-- The length of the major axis of an ellipse with given properties -/
def major_axis_length (e : Ellipse) : ℝ :=
  sorry

/-- Theorem stating that the length of the major axis is 8 for the given ellipse -/
theorem major_axis_length_is_eight :
  let e : Ellipse := {
    tangent_to_axes := true,
    focus1 := (5, -4 + 2 * Real.sqrt 3),
    focus2 := (5, -4 - 2 * Real.sqrt 3)
  }
  major_axis_length e = 8 :=
sorry

end major_axis_length_is_eight_l979_97952


namespace product_of_four_integers_l979_97948

theorem product_of_four_integers (A B C D : ℕ) 
  (h_positive : A > 0 ∧ B > 0 ∧ C > 0 ∧ D > 0)
  (h_sum : A + B + C + D = 64)
  (h_relation : A + 3 = B - 3 ∧ A + 3 = C * 3 ∧ A + 3 = D / 3) :
  A * B * C * D = 19440 := by
sorry

end product_of_four_integers_l979_97948


namespace parabola_vertex_on_x_axis_l979_97968

/-- A parabola with equation y = x^2 + 6x + c has its vertex on the x-axis if and only if c = 9 -/
theorem parabola_vertex_on_x_axis (c : ℝ) : 
  (∃ x : ℝ, x^2 + 6*x + c = 0 ∧ ∀ y : ℝ, y^2 + 6*y + c ≥ x^2 + 6*x + c) ↔ c = 9 :=
sorry

end parabola_vertex_on_x_axis_l979_97968


namespace pedro_extra_squares_l979_97959

theorem pedro_extra_squares (jesus_squares linden_squares pedro_squares : ℕ) 
  (h1 : jesus_squares = 60)
  (h2 : linden_squares = 75)
  (h3 : pedro_squares = 200) :
  pedro_squares - (jesus_squares + linden_squares) = 65 := by
  sorry

end pedro_extra_squares_l979_97959


namespace tree_ratio_is_13_3_l979_97927

/-- The ratio of trees planted to fallen Mahogany trees -/
def tree_ratio (initial_mahogany : ℕ) (initial_narra : ℕ) (total_fallen : ℕ) (final_count : ℕ) : ℚ :=
  let mahogany_fallen := (total_fallen + 1) / 2
  let trees_planted := final_count - (initial_mahogany + initial_narra - total_fallen)
  (trees_planted : ℚ) / mahogany_fallen

/-- The ratio of trees planted to fallen Mahogany trees is 13:3 -/
theorem tree_ratio_is_13_3 :
  tree_ratio 50 30 5 88 = 13 / 3 := by
  sorry

end tree_ratio_is_13_3_l979_97927


namespace simplify_fraction_l979_97926

theorem simplify_fraction : (144 : ℚ) / 216 = 2 / 3 := by
  sorry

end simplify_fraction_l979_97926


namespace area_between_circles_and_x_axis_l979_97989

/-- Represents a circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Calculates the area of the region bounded by two circles and the x-axis -/
def areaRegionBetweenCirclesAndXAxis (c1 c2 : Circle) : ℝ :=
  sorry

theorem area_between_circles_and_x_axis :
  let c1 : Circle := { center := (6, 5), radius := 3 }
  let c2 : Circle := { center := (14, 5), radius := 3 }
  areaRegionBetweenCirclesAndXAxis c1 c2 = 40 - 9 * Real.pi := by sorry

end area_between_circles_and_x_axis_l979_97989


namespace product_of_y_values_l979_97920

theorem product_of_y_values (y : ℝ) : 
  (∃ y₁ y₂ : ℝ, 
    (|2 * y₁ * 3| + 5 = 47) ∧ 
    (|2 * y₂ * 3| + 5 = 47) ∧ 
    (y₁ ≠ y₂) ∧
    (y₁ * y₂ = -49)) := by
  sorry

end product_of_y_values_l979_97920


namespace integral_absolute_value_l979_97925

noncomputable def f (x : ℝ) : ℝ := |x - 1| + |x - 3|

theorem integral_absolute_value : ∫ x in (0)..(4), f x = 10 := by sorry

end integral_absolute_value_l979_97925


namespace sine_cosine_shift_l979_97923

/-- The shift amount between two trigonometric functions -/
def shift_amount (f g : ℝ → ℝ) : ℝ :=
  sorry

theorem sine_cosine_shift :
  let f (x : ℝ) := Real.sin x + Real.sqrt 3 * Real.cos x
  let g (x : ℝ) := Real.sin x - Real.sqrt 3 * Real.cos x
  let φ := shift_amount f g
  0 < φ ∧ φ < 2 * Real.pi → φ = 2 * Real.pi / 3 :=
by sorry

end sine_cosine_shift_l979_97923


namespace inequality_holds_l979_97962

theorem inequality_holds (a b : ℕ+) : a^3 + (a+b)^2 + b ≠ b^3 + a + 2 := by
  sorry

end inequality_holds_l979_97962


namespace mary_stickers_l979_97937

/-- The number of stickers Mary brought to class -/
def stickers_brought : ℕ := 50

/-- The number of Mary's friends -/
def num_friends : ℕ := 5

/-- The number of stickers Mary gave to each friend -/
def stickers_per_friend : ℕ := 4

/-- The number of stickers Mary gave to each other student -/
def stickers_per_other : ℕ := 2

/-- The number of stickers Mary has left over -/
def stickers_leftover : ℕ := 8

/-- The total number of students in the class, including Mary -/
def total_students : ℕ := 17

theorem mary_stickers :
  stickers_brought =
    num_friends * stickers_per_friend +
    (total_students - 1 - num_friends) * stickers_per_other +
    stickers_leftover :=
by sorry

end mary_stickers_l979_97937


namespace tree_planting_group_size_l979_97914

theorem tree_planting_group_size :
  ∀ x : ℕ,
  (7 * x + 9 > 9 * (x - 1)) →
  (7 * x + 9 < 9 * (x - 1) + 3) →
  x = 8 :=
by
  sorry

end tree_planting_group_size_l979_97914


namespace binary_multiplication_theorem_l979_97954

/-- Represents a binary number as a list of booleans, where true represents 1 and false represents 0. The least significant bit is at the head of the list. -/
def BinaryNum := List Bool

/-- Converts a binary number to its decimal representation -/
def binary_to_decimal (b : BinaryNum) : ℕ :=
  b.enum.foldr (λ (i, bit) acc => acc + if bit then 2^i else 0) 0

/-- Converts a decimal number to its binary representation -/
def decimal_to_binary (n : ℕ) : BinaryNum :=
  if n = 0 then [false] else
  let rec to_binary_aux (m : ℕ) : BinaryNum :=
    if m = 0 then [] else (m % 2 = 1) :: to_binary_aux (m / 2)
  to_binary_aux n

/-- Multiplies two binary numbers -/
def binary_multiply (a b : BinaryNum) : BinaryNum :=
  decimal_to_binary (binary_to_decimal a * binary_to_decimal b)

theorem binary_multiplication_theorem :
  let a : BinaryNum := [true, true, false, true, true]  -- 11011₂
  let b : BinaryNum := [true, false, true]              -- 101₂
  let result : BinaryNum := [true, true, true, false, true, true, false, false, true]  -- 100110111₂
  binary_multiply a b = result := by
  sorry

end binary_multiplication_theorem_l979_97954


namespace marble_distribution_l979_97955

theorem marble_distribution (total_marbles : ℕ) (num_friends : ℕ) 
  (h1 : total_marbles = 60) (h2 : num_friends = 4) :
  total_marbles / num_friends = 15 := by
  sorry

end marble_distribution_l979_97955


namespace cuboid_surface_area_example_l979_97934

/-- The surface area of a cuboid given its dimensions -/
def cuboid_surface_area (width length height : ℝ) : ℝ :=
  2 * (width * length + width * height + length * height)

/-- Theorem: The surface area of a cuboid with width 3, length 4, and height 5 is 94 -/
theorem cuboid_surface_area_example : cuboid_surface_area 3 4 5 = 94 := by
  sorry

end cuboid_surface_area_example_l979_97934


namespace graph_is_three_lines_lines_not_concurrent_l979_97975

/-- The equation representing the graph -/
def graph_equation (x y : ℝ) : Prop :=
  x^2 * (x + y + 2) = y^2 * (x + y + 2)

/-- Definition of a line in 2D space -/
def is_line (S : Set (ℝ × ℝ)) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∨ b ≠ 0 ∧ 
  S = {(x, y) | a * x + b * y + c = 0}

/-- The graph consists of three distinct lines -/
theorem graph_is_three_lines :
  ∃ (L₁ L₂ L₃ : Set (ℝ × ℝ)),
    (is_line L₁ ∧ is_line L₂ ∧ is_line L₃) ∧
    (L₁ ≠ L₂ ∧ L₁ ≠ L₃ ∧ L₂ ≠ L₃) ∧
    (∀ x y, graph_equation x y ↔ (x, y) ∈ L₁ ∪ L₂ ∪ L₃) :=
sorry

/-- The three lines do not all pass through a common point -/
theorem lines_not_concurrent :
  ¬∃ (p : ℝ × ℝ), ∀ (L : Set (ℝ × ℝ)),
    (is_line L ∧ (∀ x y, graph_equation x y → (x, y) ∈ L)) → p ∈ L :=
sorry

end graph_is_three_lines_lines_not_concurrent_l979_97975


namespace disjunction_and_negation_imply_right_true_l979_97928

theorem disjunction_and_negation_imply_right_true (p q : Prop) :
  (p ∨ q) → ¬p → q := by sorry

end disjunction_and_negation_imply_right_true_l979_97928


namespace bathroom_extension_l979_97935

/-- Given a rectangular bathroom with area and width, calculate the new area after extension --/
theorem bathroom_extension (area : ℝ) (width : ℝ) (extension : ℝ) :
  area = 96 →
  width = 8 →
  extension = 2 →
  (area / width + extension) * (width + extension) = 140 := by
  sorry

end bathroom_extension_l979_97935


namespace quadratic_equation_conversion_l979_97971

theorem quadratic_equation_conversion :
  ∀ x : ℝ, x * (x + 2) = 5 * (x - 2) ↔ x^2 - 3*x - 10 = 0 :=
by sorry

end quadratic_equation_conversion_l979_97971


namespace tan_half_sum_right_triangle_angles_l979_97993

theorem tan_half_sum_right_triangle_angles (A B : ℝ) : 
  0 < A → A < π/2 → 0 < B → B < π/2 → A + B = π/2 → 
  Real.tan ((A + B) / 2) = 1 := by
sorry

end tan_half_sum_right_triangle_angles_l979_97993


namespace hyperbola_real_axis_length_l979_97957

-- Define the hyperbola equation
def hyperbola_equation (x y : ℝ) : Prop := 2 * x^2 - y^2 = 8

-- Define the length of the real axis
def real_axis_length : ℝ := 4

-- Theorem statement
theorem hyperbola_real_axis_length :
  ∀ x y : ℝ, hyperbola_equation x y → real_axis_length = 4 :=
by
  sorry

end hyperbola_real_axis_length_l979_97957


namespace modulus_z_l979_97903

theorem modulus_z (r k : ℝ) (z : ℂ) 
  (hr : |r| < 2) 
  (hk : |k| < 3) 
  (hz : z + k * z⁻¹ = r) : 
  Complex.abs z = Real.sqrt ((r^2 - 2*k) / 2) := by
  sorry

end modulus_z_l979_97903


namespace arithmetic_sequence_sum_l979_97985

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a b : ℕ → ℝ) :
  ArithmeticSequence a →
  ArithmeticSequence b →
  a 1 + b 1 = 7 →
  a 3 + b 3 = 21 →
  a 5 + b 5 = 35 := by
  sorry

end arithmetic_sequence_sum_l979_97985


namespace smallest_integer_cube_root_l979_97969

theorem smallest_integer_cube_root (m n : ℕ) (r : ℝ) : 
  (∀ k < n, ¬∃ (m' : ℕ) (r' : ℝ), m' < m ∧ 0 < r' ∧ r' < 1/500 ∧ m'^(1/3 : ℝ) = k + r') →
  0 < r →
  r < 1/500 →
  m^(1/3 : ℝ) = n + r →
  n = 13 := by
  sorry

end smallest_integer_cube_root_l979_97969


namespace tangent_line_at_one_unique_zero_implies_a_one_max_value_of_g_l979_97906

noncomputable section

-- Define the functions f and g
def f (a : ℝ) (x : ℝ) : ℝ := (x^2 - 2*x) * Real.log x + a * x^2 + 2
def g (a : ℝ) (x : ℝ) : ℝ := f a x - x - 2

-- Part I
theorem tangent_line_at_one (a : ℝ) (h : a = -1) :
  ∃ (m b : ℝ), m = 3 ∧ b = -4 ∧ ∀ x y, y = f a x → y = m * (x - 1) + f a 1 :=
sorry

-- Part II
theorem unique_zero_implies_a_one (a : ℝ) (h : a > 0) :
  (∃! x, g a x = 0) → a = 1 :=
sorry

-- Part III
theorem max_value_of_g (x : ℝ) (h1 : Real.exp (-2) < x) (h2 : x < Real.exp 1) :
  g 1 x ≤ 2 * Real.exp 2 - 3 * Real.exp 1 :=
sorry

end tangent_line_at_one_unique_zero_implies_a_one_max_value_of_g_l979_97906


namespace root_in_interval_l979_97984

def f (x : ℝ) := 3 * x^2 + 3 * x - 8

theorem root_in_interval :
  (f 1.25 < 0) → (f 1.5 > 0) →
  ∃ x ∈ Set.Ioo 1.25 1.5, f x = 0 := by
  sorry

end root_in_interval_l979_97984


namespace circumcenter_distance_theorem_l979_97960

-- Define a structure for a triangle with its circumcircle properties
structure TriangleWithCircumcircle where
  -- Sides of the triangle
  a : ℝ
  b : ℝ
  c : ℝ
  -- Angles opposite to sides a, b, c respectively
  α : ℝ
  β : ℝ
  γ : ℝ
  -- Radius of the circumscribed circle
  r : ℝ
  -- Distances from circumcenter to sides a, b, c respectively
  pa : ℝ
  pb : ℝ
  pc : ℝ
  -- Conditions for a valid triangle
  side_positive : 0 < a ∧ 0 < b ∧ 0 < c
  angle_sum : α + β + γ = π
  -- Relationship between sides and angles
  sine_law_a : a = 2 * r * Real.sin α
  sine_law_b : b = 2 * r * Real.sin β
  sine_law_c : c = 2 * r * Real.sin γ

-- Theorem statement
theorem circumcenter_distance_theorem (t : TriangleWithCircumcircle) :
  t.pa * Real.sin t.α + t.pb * Real.sin t.β + t.pc * Real.sin t.γ =
  2 * t.r * Real.sin t.α * Real.sin t.β * Real.sin t.γ :=
by sorry

end circumcenter_distance_theorem_l979_97960


namespace square_diagonal_half_l979_97942

/-- Given a square with side length 6 cm and AE = 8 cm, prove that OB = 4.5 cm -/
theorem square_diagonal_half (side_length : ℝ) (AE : ℝ) (OB : ℝ) :
  side_length = 6 →
  AE = 8 →
  OB = side_length * Real.sqrt 2 / 2 →
  OB = 4.5 := by
  sorry

end square_diagonal_half_l979_97942


namespace zero_lt_x_lt_two_sufficient_not_necessary_for_x_lt_two_l979_97983

theorem zero_lt_x_lt_two_sufficient_not_necessary_for_x_lt_two :
  (∃ x : ℝ, 0 < x ∧ x < 2 → x < 2) ∧
  (∃ x : ℝ, x < 2 ∧ ¬(0 < x ∧ x < 2)) :=
by sorry

end zero_lt_x_lt_two_sufficient_not_necessary_for_x_lt_two_l979_97983


namespace fifth_term_of_geometric_sequence_l979_97972

/-- A geometric sequence with positive terms and common ratio 2 -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a n > 0) ∧ (∀ n, a (n + 1) = 2 * a n)

theorem fifth_term_of_geometric_sequence
  (a : ℕ → ℝ)
  (h_geom : geometric_sequence a)
  (h_prod : a 3 * a 11 = 16) :
  a 5 = 1 := by
sorry

end fifth_term_of_geometric_sequence_l979_97972


namespace xiaoming_calculation_correction_l979_97905

theorem xiaoming_calculation_correction 
  (A a b c : ℝ) 
  (h : A + 2 * (a * b + 2 * b * c - 4 * a * c) = 3 * a * b - 2 * a * c + 5 * b * c) : 
  A - 2 * (a * b + 2 * b * c - 4 * a * c) = -a * b + 14 * a * c - 3 * b * c := by
sorry

end xiaoming_calculation_correction_l979_97905


namespace diamond_value_l979_97979

theorem diamond_value :
  ∀ (diamond : ℕ),
  diamond < 10 →
  diamond * 6 + 5 = diamond * 9 + 2 →
  diamond = 1 := by
sorry

end diamond_value_l979_97979


namespace parabola_and_line_properties_l979_97929

/-- Parabola C defined by y^2 = 2px where p > 0 -/
structure Parabola where
  p : ℝ
  h_p_pos : p > 0

/-- Line l defined by x = ty + m where m > 0 -/
structure Line where
  t : ℝ
  m : ℝ
  h_m_pos : m > 0

/-- Point on the parabola -/
structure ParabolaPoint (C : Parabola) where
  x : ℝ
  y : ℝ
  h_on_parabola : y^2 = 2 * C.p * x

/-- Intersection points of line and parabola -/
structure Intersection (C : Parabola) (l : Line) where
  A : ParabolaPoint C
  B : ParabolaPoint C
  h_A_on_line : A.x = l.t * A.y + l.m
  h_B_on_line : B.x = l.t * B.y + l.m

/-- Main theorem -/
theorem parabola_and_line_properties
  (C : Parabola)
  (P : ParabolaPoint C)
  (h_P_x : P.x = 2)
  (h_P_dist : (P.x - C.p/2)^2 + P.y^2 = 4^2)
  (l : Line)
  (i : Intersection C l)
  (h_circle : i.A.x * i.B.x + i.A.y * i.B.y = 0) :
  (C.p = 4 ∧ ∀ (y : ℝ), y^2 = 8 * (l.t * y + l.m)) ∧
  (l.m = 8 ∧ ∀ (y : ℝ), l.t * y + l.m = 8) := by
  sorry

end parabola_and_line_properties_l979_97929


namespace julia_bought_496_balls_l979_97924

/-- The number of balls Julia bought -/
def total_balls : ℕ :=
  let red_packs : ℕ := 3
  let yellow_packs : ℕ := 10
  let green_packs : ℕ := 8
  let blue_packs : ℕ := 5
  let red_balls_per_pack : ℕ := 22
  let yellow_balls_per_pack : ℕ := 19
  let green_balls_per_pack : ℕ := 15
  let blue_balls_per_pack : ℕ := 24
  red_packs * red_balls_per_pack +
  yellow_packs * yellow_balls_per_pack +
  green_packs * green_balls_per_pack +
  blue_packs * blue_balls_per_pack

theorem julia_bought_496_balls : total_balls = 496 := by
  sorry

end julia_bought_496_balls_l979_97924


namespace complement_of_intersection_l979_97947

def U : Set ℕ := {1, 2, 3, 4}
def M : Set ℕ := {1, 2, 3}
def N : Set ℕ := {2, 3, 4}

theorem complement_of_intersection (U M N : Set ℕ) 
  (hU : U = {1, 2, 3, 4}) 
  (hM : M = {1, 2, 3}) 
  (hN : N = {2, 3, 4}) : 
  (M ∩ N)ᶜ = {1, 4} := by
  sorry

end complement_of_intersection_l979_97947


namespace intersecting_line_properties_l979_97956

/-- A line that intersects both positive x-axis and positive y-axis -/
structure IntersectingLine where
  /-- The slope of the line -/
  m : ℝ
  /-- The y-intercept of the line -/
  b : ℝ
  /-- The line intersects the positive x-axis -/
  pos_x_intersect : ∃ x : ℝ, x > 0 ∧ m * x + b = 0
  /-- The line intersects the positive y-axis -/
  pos_y_intersect : b > 0

/-- Theorem: An intersecting line has negative slope and positive y-intercept -/
theorem intersecting_line_properties (l : IntersectingLine) : l.m < 0 ∧ l.b > 0 := by
  sorry

end intersecting_line_properties_l979_97956


namespace tan_alpha_neg_three_l979_97982

theorem tan_alpha_neg_three (α : ℝ) (h : Real.tan α = -3) :
  (Real.cos α + 2 * Real.sin α) / (Real.cos α - 3 * Real.sin α) = -1/2 := by
  sorry

end tan_alpha_neg_three_l979_97982


namespace condition_necessary_not_sufficient_l979_97953

-- Define a sequence of real numbers
def Sequence := ℕ → ℝ

-- Define what it means for a sequence to be geometric
def IsGeometric (a : Sequence) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

-- Define the condition given in the problem
def Condition (a : Sequence) : Prop :=
  ∀ n : ℕ, n ≥ 2 → a (n + 1) * a (n - 1) = a n ^ 2

-- Theorem statement
theorem condition_necessary_not_sufficient :
  (∀ a : Sequence, IsGeometric a → Condition a) ∧
  (∃ a : Sequence, Condition a ∧ ¬IsGeometric a) := by
  sorry

end condition_necessary_not_sufficient_l979_97953


namespace age_ratio_proof_l979_97912

def sachin_age : ℚ := 38.5
def age_difference : ℕ := 7

def rahul_age : ℚ := sachin_age - age_difference

theorem age_ratio_proof :
  (sachin_age * 2 / rahul_age * 2).num = 11 ∧
  (sachin_age * 2 / rahul_age * 2).den = 9 := by
  sorry

end age_ratio_proof_l979_97912


namespace original_number_proof_l979_97931

theorem original_number_proof (y : ℚ) : 1 + 1 / y = 8 / 3 → y = 3 / 5 := by
  sorry

end original_number_proof_l979_97931


namespace max_value_theorem_l979_97963

theorem max_value_theorem (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h4 : a^2 + b^2 + c^2 = 1) :
  2*a*b + 2*b*c*(Real.sqrt 2) ≤ Real.sqrt 3 :=
sorry

end max_value_theorem_l979_97963


namespace intersection_point_of_g_and_inverse_l979_97901

-- Define the function g
def g (x : ℝ) : ℝ := x^3 + 9*x^2 + 18*x + 38

-- State the theorem
theorem intersection_point_of_g_and_inverse :
  ∃! p : ℝ × ℝ, 
    (p.1 = g p.2 ∧ p.2 = g p.1) ∧ 
    p = (-2, -2) := by
  sorry

end intersection_point_of_g_and_inverse_l979_97901


namespace percentage_relation_l979_97907

/-- Given three real numbers A, B, and C, where A is 8% of C and 50% of B,
    prove that B is 16% of C. -/
theorem percentage_relation (A B C : ℝ) 
  (h1 : A = 0.08 * C) 
  (h2 : A = 0.5 * B) : 
  B = 0.16 * C := by
  sorry

end percentage_relation_l979_97907


namespace greifswald_schools_l979_97917

-- Define the schools
inductive School
| A
| B
| C

-- Define the student type
structure Student where
  id : Nat
  school : School

-- Define the knowing relation
def knows (s1 s2 : Student) : Prop := sorry

-- Define the set of all students
def AllStudents : Set Student := sorry

-- State the conditions
axiom non_empty_schools :
  ∃ (a b c : Student), a.school = School.A ∧ b.school = School.B ∧ c.school = School.C

axiom knowing_condition :
  ∀ (a b c : Student),
    a.school = School.A → b.school = School.B → c.school = School.C →
    ((knows a b ∧ knows a c ∧ ¬knows b c) ∨
     (knows a b ∧ ¬knows a c ∧ knows b c) ∨
     (¬knows a b ∧ knows a c ∧ knows b c))

-- State the theorem to be proved
theorem greifswald_schools :
  (∃ (a : Student), a.school = School.A ∧ ∀ (b : Student), b.school = School.B → knows a b) ∨
  (∃ (b : Student), b.school = School.B ∧ ∀ (c : Student), c.school = School.C → knows b c) ∨
  (∃ (c : Student), c.school = School.C ∧ ∀ (a : Student), a.school = School.A → knows c a) :=
by
  sorry

end greifswald_schools_l979_97917


namespace function_comparison_l979_97941

open Set

theorem function_comparison (a b : ℝ) (f g : ℝ → ℝ) 
  (hf : Differentiable ℝ f) (hg : Differentiable ℝ g)
  (h_eq : f a = g a)
  (h_deriv : ∀ x ∈ Set.Ioo a b, deriv f x > deriv g x) :
  ∀ x ∈ Set.Ioo a b, f x > g x := by
  sorry

end function_comparison_l979_97941


namespace tesseract_parallel_edge_pairs_l979_97966

/-- A tesseract is a 4-dimensional hypercube -/
structure Tesseract where
  dim : Nat
  dim_eq : dim = 4

/-- The number of pairs of parallel edges in a tesseract -/
def parallel_edge_pairs (t : Tesseract) : Nat := 36

/-- Theorem: A tesseract has 36 pairs of parallel edges -/
theorem tesseract_parallel_edge_pairs (t : Tesseract) : 
  parallel_edge_pairs t = 36 := by sorry

end tesseract_parallel_edge_pairs_l979_97966


namespace cone_volume_l979_97997

/-- The volume of a cone with base radius 1 and unfolded side surface area 2π -/
theorem cone_volume (r : Real) (side_area : Real) (h : Real) : 
  r = 1 → side_area = 2 * Real.pi → h = Real.sqrt 3 → 
  (1 / 3 : Real) * Real.pi * r^2 * h = (Real.sqrt 3 * Real.pi) / 3 := by
  sorry

#check cone_volume

end cone_volume_l979_97997


namespace price_reduction_l979_97986

theorem price_reduction (original_price reduced_price : ℝ) : 
  reduced_price = original_price * 0.5 ∧ reduced_price = 620 → original_price = 1240 := by
  sorry

end price_reduction_l979_97986


namespace black_pens_count_l979_97992

/-- The number of black pens initially in the jar -/
def initial_black_pens : ℕ := 21

theorem black_pens_count :
  let initial_blue_pens : ℕ := 9
  let initial_red_pens : ℕ := 6
  let removed_blue_pens : ℕ := 4
  let removed_black_pens : ℕ := 7
  let remaining_pens : ℕ := 25
  initial_blue_pens + initial_black_pens + initial_red_pens - 
    (removed_blue_pens + removed_black_pens) = remaining_pens →
  initial_black_pens = 21 := by
sorry

end black_pens_count_l979_97992
