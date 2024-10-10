import Mathlib

namespace total_kids_l1706_170675

theorem total_kids (girls : ℕ) (boys : ℕ) 
  (h1 : girls = 3) (h2 : boys = 6) : 
  girls + boys = 9 := by
  sorry

end total_kids_l1706_170675


namespace sine_product_ratio_l1706_170654

theorem sine_product_ratio (c : ℝ) (h : c = 2 * Real.pi / 13) :
  (Real.sin (4 * c) * Real.sin (8 * c) * Real.sin (12 * c) * Real.sin (16 * c) * Real.sin (20 * c)) /
  (Real.sin c * Real.sin (3 * c) * Real.sin (5 * c) * Real.sin (7 * c) * Real.sin (9 * c)) = 1 := by
  sorry

end sine_product_ratio_l1706_170654


namespace quadratic_inequality_solution_set_l1706_170620

theorem quadratic_inequality_solution_set :
  {x : ℝ | 4*x^2 - 12*x + 5 < 0} = Set.Ioo (1/2 : ℝ) (5/2 : ℝ) := by
  sorry

end quadratic_inequality_solution_set_l1706_170620


namespace age_difference_proof_l1706_170669

-- Define the ages of Betty, Mary, and Albert
def betty_age : ℕ := 11
def albert_age (betty_age : ℕ) : ℕ := 4 * betty_age
def mary_age (albert_age : ℕ) : ℕ := albert_age / 2

-- Theorem statement
theorem age_difference_proof :
  albert_age betty_age - mary_age (albert_age betty_age) = 22 := by
  sorry

end age_difference_proof_l1706_170669


namespace largest_common_divisor_of_sticker_albums_l1706_170631

theorem largest_common_divisor_of_sticker_albums : ∃ (n : ℕ), n > 0 ∧ 
  n ∣ 1050 ∧ n ∣ 1260 ∧ n ∣ 945 ∧ 
  ∀ (m : ℕ), m > 0 → m ∣ 1050 → m ∣ 1260 → m ∣ 945 → m ≤ n :=
by sorry

end largest_common_divisor_of_sticker_albums_l1706_170631


namespace tom_payment_proof_l1706_170689

/-- Represents the purchase of a fruit with its quantity and price per kg -/
structure FruitPurchase where
  quantity : Float
  pricePerKg : Float

/-- Calculates the total cost of a fruit purchase -/
def calculateCost (purchase : FruitPurchase) : Float :=
  purchase.quantity * purchase.pricePerKg

/-- Represents Tom's fruit shopping trip -/
def tomShopping : List FruitPurchase := [
  { quantity := 15.3, pricePerKg := 1.85 },  -- apples
  { quantity := 12.7, pricePerKg := 2.45 },  -- mangoes
  { quantity := 10.5, pricePerKg := 3.20 },  -- grapes
  { quantity := 6.2,  pricePerKg := 4.50 }   -- strawberries
]

/-- The discount rate applied to the total bill -/
def discountRate : Float := 0.10

/-- The sales tax rate applied to the discounted amount -/
def taxRate : Float := 0.06

/-- Calculates the final amount Tom pays after discount and tax -/
def calculateFinalAmount (purchases : List FruitPurchase) (discount : Float) (tax : Float) : Float :=
  let totalCost := purchases.map calculateCost |>.sum
  let discountedCost := totalCost * (1 - discount)
  let finalCost := discountedCost * (1 + tax)
  (finalCost * 100).round / 100  -- Round to nearest cent

theorem tom_payment_proof :
  calculateFinalAmount tomShopping discountRate taxRate = 115.36 := by
  sorry

end tom_payment_proof_l1706_170689


namespace inequality_addition_l1706_170628

theorem inequality_addition (a b c : ℝ) (h1 : a > b) (h2 : b > c) : a + b > b + c := by
  sorry

end inequality_addition_l1706_170628


namespace train_carriages_count_l1706_170660

/-- Calculates the number of carriages in a train given specific conditions -/
theorem train_carriages_count (carriage_length engine_length : ℝ)
                               (train_speed : ℝ)
                               (bridge_crossing_time : ℝ)
                               (bridge_length : ℝ) :
  carriage_length = 60 →
  engine_length = 60 →
  train_speed = 60 * 1000 / 60 →
  bridge_crossing_time = 5 →
  bridge_length = 3.5 * 1000 →
  ∃ n : ℕ, n = 24 ∧ 
    n * carriage_length + engine_length = 
    train_speed * bridge_crossing_time - bridge_length :=
by
  sorry

end train_carriages_count_l1706_170660


namespace g_of_3_l1706_170627

def g (x : ℝ) : ℝ := 7 * x^3 - 5 * x^2 + 3 * x - 6

theorem g_of_3 : g 3 = 147 := by
  sorry

end g_of_3_l1706_170627


namespace rachel_apple_picking_l1706_170650

/-- Rachel's apple picking problem -/
theorem rachel_apple_picking (num_trees : ℕ) (apples_left : ℕ) (initial_apples : ℕ) : 
  num_trees = 3 → apples_left = 9 → initial_apples = 33 → 
  (initial_apples - apples_left) / num_trees = 8 := by
  sorry

end rachel_apple_picking_l1706_170650


namespace max_distance_complex_numbers_l1706_170676

theorem max_distance_complex_numbers (z : ℂ) (h : Complex.abs z = 3) :
  Complex.abs ((1 + 2*Complex.I)*z - z^2) ≤ 12 := by
  sorry

end max_distance_complex_numbers_l1706_170676


namespace book_selection_theorem_l1706_170658

/-- The number of ways to select books from odd and even positions -/
def select_books (total : Nat) : Nat :=
  (total / 2) * (total / 2)

/-- Theorem stating the total number of ways to select the books -/
theorem book_selection_theorem :
  let biology_books := 12
  let chemistry_books := 8
  (select_books biology_books) * (select_books chemistry_books) = 576 := by
  sorry

#eval select_books 12 * select_books 8

end book_selection_theorem_l1706_170658


namespace line_separate_from_circle_l1706_170617

/-- The line x₀x + y₀y - a² = 0 is separate from the circle x² + y² = a² (a > 0),
    given that point M(x₀, y₀) is inside the circle and different from its center. -/
theorem line_separate_from_circle
  (a : ℝ) (x₀ y₀ : ℝ) 
  (h_a_pos : a > 0)
  (h_inside : x₀^2 + y₀^2 < a^2)
  (h_not_center : x₀ ≠ 0 ∨ y₀ ≠ 0) :
  let d := a^2 / Real.sqrt (x₀^2 + y₀^2)
  d > a :=
by sorry

end line_separate_from_circle_l1706_170617


namespace square_perimeter_problem_l1706_170600

theorem square_perimeter_problem (M N : Real) (h1 : M = 100) (h2 : N = 4 * M) :
  4 * Real.sqrt N = 80 := by
  sorry

end square_perimeter_problem_l1706_170600


namespace expression_value_l1706_170659

theorem expression_value (a b : ℝ) (ha : a = 0.137) (hb : b = 0.098) :
  ((a + b)^2 - (a - b)^2) / (a * b) = 4 := by
  sorry

end expression_value_l1706_170659


namespace significant_figures_of_number_l1706_170610

/-- Count the number of significant figures in a rational number represented as a string -/
def count_significant_figures (s : String) : ℕ := sorry

/-- The rational number in question -/
def number : String := "0.0050400"

/-- Theorem stating that the number of significant figures in 0.0050400 is 5 -/
theorem significant_figures_of_number : count_significant_figures number = 5 := by sorry

end significant_figures_of_number_l1706_170610


namespace triangle_max_area_l1706_170642

/-- Given a triangle ABC where c = 2 and b = √2 * a, 
    the maximum area of the triangle is 2√2 -/
theorem triangle_max_area (a b c : ℝ) (h1 : c = 2) (h2 : b = Real.sqrt 2 * a) :
  ∃ (S : ℝ), S = (Real.sqrt 2 : ℝ) * 2 ∧ 
  (∀ (S' : ℝ), S' = (1/2) * a * b * Real.sin (Real.arccos ((a^2 + b^2 - c^2)/(2*a*b))) → S' ≤ S) :=
sorry

end triangle_max_area_l1706_170642


namespace sum_m_n_equals_negative_two_l1706_170602

/-- A polynomial in x and y -/
def polynomial (m n : ℝ) (x y : ℝ) : ℝ := m * x^2 - n * x * y - 2 * x * y + y - 3

/-- The condition that the polynomial has no quadratic terms when simplified -/
def no_quadratic_terms (m n : ℝ) : Prop :=
  ∀ x y, polynomial m n x y = (-n - 2) * x * y + y - 3

theorem sum_m_n_equals_negative_two (m n : ℝ) (h : no_quadratic_terms m n) : m + n = -2 := by
  sorry

end sum_m_n_equals_negative_two_l1706_170602


namespace sandwich_change_l1706_170623

/-- Calculates the change received when buying a number of items at a given price and paying with a certain amount. -/
def calculate_change (num_items : ℕ) (price_per_item : ℕ) (amount_paid : ℕ) : ℕ :=
  amount_paid - (num_items * price_per_item)

/-- Proves that buying 3 items at $5 each, paid with a $20 bill, results in $5 change. -/
theorem sandwich_change : calculate_change 3 5 20 = 5 := by
  sorry

#eval calculate_change 3 5 20

end sandwich_change_l1706_170623


namespace women_who_left_l1706_170636

/-- Proves the number of women who left the room given the initial and final conditions --/
theorem women_who_left (initial_men : ℕ) (initial_women : ℕ) 
  (h1 : initial_men * 5 = initial_women * 4)  -- Initial ratio of men to women is 4:5
  (h2 : initial_men + 2 = 14)  -- 2 men entered, final count is 14 men
  (h3 : 2 * (initial_women - women_left) = 24)  -- Women doubled after some left, final count is 24 women
  : women_left = 3 := by
  sorry

#check women_who_left

end women_who_left_l1706_170636


namespace data_grouping_l1706_170619

theorem data_grouping (max_value min_value class_width : ℕ) 
  (h1 : max_value = 141)
  (h2 : min_value = 40)
  (h3 : class_width = 10) :
  Int.ceil ((max_value - min_value : ℝ) / class_width) = 11 := by
  sorry

#check data_grouping

end data_grouping_l1706_170619


namespace circle_intersection_theorem_l1706_170622

structure Circle := (center : ℝ × ℝ) (radius : ℝ)

structure Point := (coords : ℝ × ℝ)

def on_circle (p : Point) (c : Circle) : Prop :=
  let (x, y) := p.coords
  let (cx, cy) := c.center
  (x - cx)^2 + (y - cy)^2 = c.radius^2

def diametrically_opposite (p1 p2 : Point) (c : Circle) : Prop :=
  let (x1, y1) := p1.coords
  let (x2, y2) := p2.coords
  let (cx, cy) := c.center
  (x1 - cx)^2 + (y1 - cy)^2 = c.radius^2 ∧
  (x2 - cx)^2 + (y2 - cy)^2 = c.radius^2 ∧
  (x1 - x2)^2 + (y1 - y2)^2 = 4 * c.radius^2

def angle (p1 p2 p3 : Point) : ℝ := sorry

theorem circle_intersection_theorem (c : Circle) (A B C D M N : Point) :
  on_circle A c ∧ on_circle B c ∧ on_circle C c ∧ on_circle D c →
  (∃ t : ℝ, A.coords = B.coords + t • (C.coords - D.coords)) →
  (∃ s : ℝ, A.coords = D.coords + s • (B.coords - C.coords)) →
  angle B M C = angle C N D ↔
  diametrically_opposite A C c ∨ diametrically_opposite B D c :=
sorry

end circle_intersection_theorem_l1706_170622


namespace negative_division_example_l1706_170656

theorem negative_division_example : (-150) / (-25) = 6 := by
  sorry

end negative_division_example_l1706_170656


namespace arithmetic_geometric_sequence_common_difference_l1706_170625

/-- An arithmetic sequence with the given properties has a common difference of -1/5 -/
theorem arithmetic_geometric_sequence_common_difference :
  ∀ (a : ℕ → ℚ) (d : ℚ),
  d ≠ 0 →
  (∀ n, a (n + 1) = a n + d) →
  a 1 = 1 →
  (a 2) * (a 5) = (a 4)^2 →
  d = -1/5 := by
sorry

end arithmetic_geometric_sequence_common_difference_l1706_170625


namespace max_min_f_on_interval_l1706_170664

def f (x : ℝ) := x^3 - 3*x + 1

theorem max_min_f_on_interval :
  ∃ (max min : ℝ),
    (∀ x ∈ Set.Icc (-3) 0, f x ≤ max) ∧
    (∃ x ∈ Set.Icc (-3) 0, f x = max) ∧
    (∀ x ∈ Set.Icc (-3) 0, min ≤ f x) ∧
    (∃ x ∈ Set.Icc (-3) 0, f x = min) ∧
    max = 1 ∧ min = -17 := by sorry

end max_min_f_on_interval_l1706_170664


namespace identity_function_only_solution_l1706_170672

theorem identity_function_only_solution 
  (f : ℕ+ → ℕ+) 
  (h : ∀ a b : ℕ+, (a - f b) ∣ (a * f a - b * f b)) :
  ∀ x : ℕ+, f x = x :=
by sorry

end identity_function_only_solution_l1706_170672


namespace thirty_people_three_groups_l1706_170609

/-- The number of ways to divide n people into k groups of m people each -/
def group_divisions (n m k : ℕ) : ℕ :=
  if n = m * k then
    Nat.factorial n / (Nat.factorial m ^ k)
  else
    0

/-- Theorem: The number of ways to divide 30 people into 3 groups of 10 each
    is equal to 30! / (10!)³ -/
theorem thirty_people_three_groups :
  group_divisions 30 10 3 = Nat.factorial 30 / (Nat.factorial 10 ^ 3) := by
  sorry

end thirty_people_three_groups_l1706_170609


namespace sqrt_equation_solution_l1706_170694

theorem sqrt_equation_solution (y : ℝ) : 
  Real.sqrt (4 + Real.sqrt (3 * y - 5)) = Real.sqrt 10 → y = 41 / 3 := by
  sorry

end sqrt_equation_solution_l1706_170694


namespace ab_minus_three_l1706_170663

theorem ab_minus_three (a b : ℤ) (h : a - b = -2) : a - b - 3 = -5 := by
  sorry

end ab_minus_three_l1706_170663


namespace larger_integer_of_product_and_sum_l1706_170644

theorem larger_integer_of_product_and_sum (x y : ℤ) 
  (h_product : x * y = 30) 
  (h_sum : x + y = 13) : 
  max x y = 10 := by
  sorry

end larger_integer_of_product_and_sum_l1706_170644


namespace triangle_sine_theorem_l1706_170679

theorem triangle_sine_theorem (area : ℝ) (side : ℝ) (median : ℝ) (θ : ℝ) :
  area = 30 →
  side = 10 →
  median = 9 →
  area = (1/2) * side * median * Real.sin θ →
  0 < θ →
  θ < π/2 →
  Real.sin θ = 2/3 := by
  sorry

end triangle_sine_theorem_l1706_170679


namespace prime_power_congruence_l1706_170678

theorem prime_power_congruence (p : ℕ) (hp : p.Prime) (hp2 : p > 2) :
  (p^(p+2) + (p+2)^p) % (2*p+2) = 0 := by
  sorry

end prime_power_congruence_l1706_170678


namespace sum_of_fourth_powers_l1706_170612

theorem sum_of_fourth_powers (a b : ℂ) 
  (h1 : (a + 1) * (b + 1) = 2)
  (h2 : (a^2 + 1) * (b^2 + 1) = 32) :
  ∃ x y : ℂ, 
    (x^4 + 1) * (y^4 + 1) + (a^4 + 1) * (b^4 + 1) = 1924 ∧
    ((x + 1) * (y + 1) = 2 ∧ (x^2 + 1) * (y^2 + 1) = 32) :=
by sorry

end sum_of_fourth_powers_l1706_170612


namespace barbed_wire_rate_l1706_170698

/-- Given a square field with area 3136 sq m, barbed wire drawn 3 m around it,
    two 1 m wide gates, and a total cost of 1332 Rs, prove that the rate of
    drawing barbed wire per meter is 6 Rs/m. -/
theorem barbed_wire_rate (area : ℝ) (wire_distance : ℝ) (gate_width : ℝ) (num_gates : ℕ) (total_cost : ℝ)
    (h_area : area = 3136)
    (h_wire_distance : wire_distance = 3)
    (h_gate_width : gate_width = 1)
    (h_num_gates : num_gates = 2)
    (h_total_cost : total_cost = 1332) :
    total_cost / (4 * Real.sqrt area - num_gates * gate_width) = 6 := by
  sorry

end barbed_wire_rate_l1706_170698


namespace function_decreasing_iff_a_in_range_l1706_170649

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + 2 * (a - 3) * x + 1

-- Define the property of being decreasing on an interval
def is_decreasing_on (f : ℝ → ℝ) (lb : ℝ) : Prop :=
  ∀ x y, lb ≤ x → x < y → f y < f x

-- State the theorem
theorem function_decreasing_iff_a_in_range :
  ∀ a : ℝ, (is_decreasing_on (f a) (-2)) ↔ a ∈ Set.Icc (-3) 0 :=
sorry

end function_decreasing_iff_a_in_range_l1706_170649


namespace smallest_multiple_eight_is_solution_eight_is_smallest_l1706_170674

theorem smallest_multiple (x : ℕ+) : (450 * x : ℕ) % 720 = 0 → x ≥ 8 := by
  sorry

theorem eight_is_solution : (450 * 8 : ℕ) % 720 = 0 := by
  sorry

theorem eight_is_smallest : ∀ (x : ℕ+), (450 * x : ℕ) % 720 = 0 → x ≥ 8 := by
  sorry

end smallest_multiple_eight_is_solution_eight_is_smallest_l1706_170674


namespace coin_jar_problem_l1706_170662

theorem coin_jar_problem (x : ℕ) : 
  (x : ℚ) * (1 + 5 + 10 + 25) / 100 = 20 → x = 50 := by
  sorry

end coin_jar_problem_l1706_170662


namespace square_triangle_circle_perimeter_l1706_170666

theorem square_triangle_circle_perimeter (x : ℝ) : 
  (4 * x) + (3 * x) = 2 * π * 4 → x = (8 * π) / 7 := by
  sorry

end square_triangle_circle_perimeter_l1706_170666


namespace sum_of_integers_l1706_170686

theorem sum_of_integers (p q r s : ℤ) 
  (eq1 : p - q + r = 7)
  (eq2 : q - r + s = 8)
  (eq3 : r - s + p = 4)
  (eq4 : s - p + q = 3) :
  p + q + r + s = 11 := by
  sorry

end sum_of_integers_l1706_170686


namespace common_factor_of_polynomial_l1706_170639

theorem common_factor_of_polynomial (m a b : ℤ) : 
  ∃ (k₁ k₂ : ℤ), 3*m*a^2 - 6*m*a*b = m * (k₁*a^2 + k₂*a*b) :=
sorry

end common_factor_of_polynomial_l1706_170639


namespace distance_to_destination_l1706_170688

/-- Proves that the distance to a destination is 144 km given specific rowing conditions --/
theorem distance_to_destination (rowing_speed : ℝ) (current_speed : ℝ) (total_time : ℝ) : 
  rowing_speed = 10 →
  current_speed = 2 →
  total_time = 30 →
  let downstream_speed := rowing_speed + current_speed
  let upstream_speed := rowing_speed - current_speed
  ∃ (distance : ℝ), 
    distance / downstream_speed + distance / upstream_speed = total_time ∧
    distance = 144 := by
  sorry

end distance_to_destination_l1706_170688


namespace geometric_ratio_sum_condition_l1706_170697

theorem geometric_ratio_sum_condition (a b c d a' b' c' d' : ℝ) 
  (h1 : a / b = c / d) (h2 : a' / b' = c' / d') :
  (a + a') / (b + b') = (c + c') / (d + d') ↔ a / a' = b / b' ∧ b / b' = c / c' ∧ c / c' = d / d' :=
by sorry

end geometric_ratio_sum_condition_l1706_170697


namespace polynomial_divisibility_l1706_170604

theorem polynomial_divisibility (k n : ℕ) (P : Polynomial ℤ) : 
  Even k → 
  (∀ i : ℕ, i < k → Odd (P.coeff i)) → 
  P.degree = k → 
  (∃ Q : Polynomial ℤ, (X + 1)^n - 1 = P * Q) → 
  (k + 1) ∣ n :=
sorry

end polynomial_divisibility_l1706_170604


namespace dvd_fraction_proof_l1706_170685

def initial_amount : ℚ := 320
def book_fraction : ℚ := 1/4
def book_additional : ℚ := 10
def dvd_additional : ℚ := 8
def final_amount : ℚ := 130

theorem dvd_fraction_proof :
  ∃ f : ℚ, 
    initial_amount - (book_fraction * initial_amount + book_additional) - 
    (f * (initial_amount - (book_fraction * initial_amount + book_additional)) + dvd_additional) = 
    final_amount ∧ f = 46/115 := by
  sorry

end dvd_fraction_proof_l1706_170685


namespace total_numbers_correction_l1706_170696

/-- Given an initial average of 15, where one number was misread as 26 instead of 36,
    and the correct average is 16, prove that the total number of numbers is 10. -/
theorem total_numbers_correction (initial_avg : ℚ) (misread : ℚ) (correct : ℚ) (correct_avg : ℚ)
  (h1 : initial_avg = 15)
  (h2 : misread = 26)
  (h3 : correct = 36)
  (h4 : correct_avg = 16) :
  ∃ (n : ℕ) (S : ℚ), n > 0 ∧ n = 10 ∧ 
    S / n + misread / n = initial_avg ∧
    S / n + correct / n = correct_avg :=
by sorry

end total_numbers_correction_l1706_170696


namespace negative_of_negative_is_positive_l1706_170632

theorem negative_of_negative_is_positive (x : ℝ) : x < 0 → -x > 0 := by
  sorry

end negative_of_negative_is_positive_l1706_170632


namespace probability_at_least_one_pen_l1706_170638

theorem probability_at_least_one_pen
  (p_ball : ℝ)
  (p_ink : ℝ)
  (h_ball : p_ball = 3 / 5)
  (h_ink : p_ink = 2 / 3)
  (h_nonneg_ball : 0 ≤ p_ball)
  (h_nonneg_ink : 0 ≤ p_ink)
  (h_le_one_ball : p_ball ≤ 1)
  (h_le_one_ink : p_ink ≤ 1)
  (h_independent : True)  -- Assumption of independence
  : p_ball + p_ink - p_ball * p_ink = 13 / 15 :=
sorry

end probability_at_least_one_pen_l1706_170638


namespace initial_average_score_l1706_170621

theorem initial_average_score 
  (total_students : Nat) 
  (remaining_students : Nat)
  (dropped_score : Real)
  (new_average : Real) :
  total_students = 16 →
  remaining_students = 15 →
  dropped_score = 24 →
  new_average = 64 →
  (total_students : Real) * (remaining_students * new_average + dropped_score) / total_students = 61.5 :=
by sorry

end initial_average_score_l1706_170621


namespace P_inter_Q_eq_interval_l1706_170652

/-- The set P defined by the inequality 3x - x^2 ≤ 0 -/
def P : Set ℝ := {x : ℝ | 3 * x - x^2 ≤ 0}

/-- The set Q defined by the inequality |x| ≤ 2 -/
def Q : Set ℝ := {x : ℝ | |x| ≤ 2}

/-- The theorem stating that the intersection of P and Q is equal to the set {x | -2 ≤ x ≤ 0} -/
theorem P_inter_Q_eq_interval : P ∩ Q = {x : ℝ | -2 ≤ x ∧ x ≤ 0} := by sorry

end P_inter_Q_eq_interval_l1706_170652


namespace sphere_volume_sphere_surface_area_sphere_surface_eq_cylinder_lateral_l1706_170606

/-- A structure representing a sphere contained in a cylinder -/
structure SphereInCylinder where
  r : ℝ  -- radius of the sphere and base of the cylinder
  h : ℝ  -- height of the cylinder
  sphere_diameter_eq_cylinder : h = 2 * r  -- diameter of sphere equals height of cylinder

/-- The volume of the sphere is (4/3)πr³ -/
theorem sphere_volume (s : SphereInCylinder) : 
  (4 / 3) * Real.pi * s.r ^ 3 = (2 / 3) * Real.pi * s.r ^ 2 * s.h := by sorry

/-- The surface area of the sphere is 4πr² -/
theorem sphere_surface_area (s : SphereInCylinder) :
  4 * Real.pi * s.r ^ 2 = (2 / 3) * (2 * Real.pi * s.r * s.h + 2 * Real.pi * s.r ^ 2) := by sorry

/-- The surface area of the sphere equals the lateral surface area of the cylinder -/
theorem sphere_surface_eq_cylinder_lateral (s : SphereInCylinder) :
  4 * Real.pi * s.r ^ 2 = 2 * Real.pi * s.r * s.h := by sorry

end sphere_volume_sphere_surface_area_sphere_surface_eq_cylinder_lateral_l1706_170606


namespace base4_arithmetic_l1706_170655

/-- Converts a base 4 number to base 10 --/
def base4ToBase10 (n : ℕ) : ℕ := sorry

/-- Converts a base 10 number to base 4 --/
def base10ToBase4 (n : ℕ) : ℕ := sorry

/-- Multiplication operation for base 4 numbers --/
def mulBase4 (a b : ℕ) : ℕ := base10ToBase4 (base4ToBase10 a * base4ToBase10 b)

/-- Division operation for base 4 numbers --/
def divBase4 (a b : ℕ) : ℕ := base10ToBase4 (base4ToBase10 a / base4ToBase10 b)

theorem base4_arithmetic : 
  divBase4 (mulBase4 231 21) 3 = 213 := by sorry

end base4_arithmetic_l1706_170655


namespace floor_plus_self_equals_seventeen_fourths_l1706_170681

theorem floor_plus_self_equals_seventeen_fourths :
  ∃ x : ℚ, (⌊x⌋ : ℚ) + x = 17/4 ∧ x = 9/4 := by
  sorry

end floor_plus_self_equals_seventeen_fourths_l1706_170681


namespace new_person_weight_l1706_170645

/-- The weight of a new person who replaces one person in a group, given the change in average weight -/
def weight_of_new_person (n : ℕ) (avg_increase : ℝ) (replaced_weight : ℝ) : ℝ :=
  replaced_weight + n * avg_increase

/-- Theorem stating the weight of the new person in the given scenario -/
theorem new_person_weight :
  weight_of_new_person 10 6.3 65 = 128 := by
  sorry

end new_person_weight_l1706_170645


namespace minimize_expression_l1706_170687

theorem minimize_expression (a b : ℝ) (h1 : a + b = -2) (h2 : b < 0) :
  ∃ (min_a : ℝ), min_a = 2 ∧
  ∀ (x : ℝ), x + b = -2 → (1 / (2 * |x|) - |x| / b) ≥ (1 / (2 * |min_a|) - |min_a| / b) :=
sorry

end minimize_expression_l1706_170687


namespace ellipse_sum_range_l1706_170618

-- Define the ellipse
def is_on_ellipse (x y : ℝ) : Prop :=
  x^2 / 144 + y^2 / 25 = 1

-- Theorem statement
theorem ellipse_sum_range :
  ∀ x y : ℝ, is_on_ellipse x y →
  ∃ (a b : ℝ), a = -13 ∧ b = 13 ∧
  a ≤ x + y ∧ x + y ≤ b ∧
  (∃ (x₁ y₁ : ℝ), is_on_ellipse x₁ y₁ ∧ x₁ + y₁ = a) ∧
  (∃ (x₂ y₂ : ℝ), is_on_ellipse x₂ y₂ ∧ x₂ + y₂ = b) :=
by sorry

end ellipse_sum_range_l1706_170618


namespace integer_roots_of_polynomial_l1706_170616

def polynomial (x a₂ a₁ : ℤ) : ℤ := x^3 + a₂*x^2 + a₁*x + 24

def possible_roots : Set ℤ := {-24, -12, -8, -6, -4, -3, -2, -1, 1, 2, 3, 4, 6, 8, 12, 24}

theorem integer_roots_of_polynomial (a₂ a₁ : ℤ) :
  {x : ℤ | polynomial x a₂ a₁ = 0} ⊆ possible_roots :=
by sorry

end integer_roots_of_polynomial_l1706_170616


namespace special_sale_discount_l1706_170692

theorem special_sale_discount (list_price : ℝ) (regular_discount_min : ℝ) (regular_discount_max : ℝ) (lowest_sale_price_ratio : ℝ) :
  list_price = 80 →
  regular_discount_min = 0.3 →
  regular_discount_max = 0.5 →
  lowest_sale_price_ratio = 0.4 →
  ∃ (additional_discount : ℝ),
    additional_discount = 0.2 ∧
    list_price * (1 - regular_discount_max) * (1 - additional_discount) = list_price * lowest_sale_price_ratio :=
by
  sorry

end special_sale_discount_l1706_170692


namespace subtraction_value_l1706_170651

theorem subtraction_value (N : ℝ) (h1 : (N - 24) / 10 = 3) : 
  ∃ x : ℝ, (N - x) / 7 = 7 ∧ x = 5 := by
  sorry

end subtraction_value_l1706_170651


namespace new_ratio_after_boarders_join_l1706_170635

/-- Represents the number of students in a school -/
structure School where
  boarders : ℕ
  dayScholars : ℕ

/-- Represents a ratio between two numbers -/
structure Ratio where
  numerator : ℕ
  denominator : ℕ

def School.ratio (s : School) : Ratio :=
  { numerator := s.boarders, denominator := s.dayScholars }

def School.addBoarders (s : School) (n : ℕ) : School :=
  { boarders := s.boarders + n, dayScholars := s.dayScholars }

theorem new_ratio_after_boarders_join
  (initialSchool : School)
  (initialRatio : Ratio)
  (newBoarders : ℕ) :
  initialSchool.ratio = initialRatio →
  initialSchool.boarders = 560 →
  initialRatio.numerator = 7 →
  initialRatio.denominator = 16 →
  newBoarders = 80 →
  (initialSchool.addBoarders newBoarders).ratio =
    { numerator := 1, denominator := 2 } :=
by
  sorry

end new_ratio_after_boarders_join_l1706_170635


namespace value_of_a_l1706_170690

theorem value_of_a : ∀ (a b c d : ℤ),
  a = b + 7 →
  b = c + 15 →
  c = d + 25 →
  d = 90 →
  a = 137 := by
sorry

end value_of_a_l1706_170690


namespace apple_ratio_l1706_170615

theorem apple_ratio (total_weight : ℝ) (apples_per_pie : ℝ) (num_pies : ℝ) 
  (h1 : total_weight = 120)
  (h2 : apples_per_pie = 4)
  (h3 : num_pies = 15) :
  (total_weight - apples_per_pie * num_pies) / total_weight = 1 / 2 :=
by sorry

end apple_ratio_l1706_170615


namespace book_arrangement_theorem_l1706_170671

def num_math_books : ℕ := 4
def num_history_books : ℕ := 7

def ways_to_arrange_books : ℕ :=
  -- Ways to choose math books for the ends
  (num_math_books * (num_math_books - 1)) *
  -- Ways to choose and arrange 2 history books from 7
  (num_history_books * (num_history_books - 1)) *
  -- Ways to choose the third book (math or history)
  (num_math_books + num_history_books - 3) *
  -- Ways to permute the first three books
  6 *
  -- Ways to arrange the remaining 6 books
  (6 * 5 * 4 * 3 * 2 * 1)

theorem book_arrangement_theorem :
  ways_to_arrange_books = 19571200 :=
sorry

end book_arrangement_theorem_l1706_170671


namespace unique_square_pattern_l1706_170608

def fits_pattern (n : ℕ) : Prop :=
  n ≥ 100 ∧ n < 1000 ∧
  ∀ d₁ d₂ d₃, n = 100 * d₁ + 10 * d₂ + d₃ →
    d₁ * d₁ < 10 ∧
    d₁ * d₂ < 10 ∧
    d₁ * d₃ < 10 ∧
    d₂ * d₂ < 10 ∧
    d₂ * d₃ < 10 ∧
    d₃ * d₃ < 10

theorem unique_square_pattern :
  ∃! n : ℕ, fits_pattern n ∧ n = 233 :=
sorry

end unique_square_pattern_l1706_170608


namespace contradiction_assumption_l1706_170648

theorem contradiction_assumption (a b c d : ℝ) :
  (¬ (a < 0 ∨ b < 0 ∨ c < 0 ∨ d < 0)) ↔ (a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0) := by
  sorry

end contradiction_assumption_l1706_170648


namespace inheritance_calculation_l1706_170646

theorem inheritance_calculation (x : ℝ) : 
  (0.25 * x + 0.15 * (x - 0.25 * x) = 15000) → x = 41379.31 := by
  sorry

end inheritance_calculation_l1706_170646


namespace house_application_proof_l1706_170624

/-- The number of houses available -/
def num_houses : ℕ := 3

/-- The number of persons applying for houses -/
def num_persons : ℕ := 3

/-- The probability that all persons apply for the same house -/
def prob_same_house : ℚ := 1 / 9

/-- The number of houses each person applies for -/
def houses_per_person : ℕ := 1

theorem house_application_proof :
  (prob_same_house = (houses_per_person : ℚ)^2 / num_houses^2) →
  houses_per_person = 1 :=
by sorry

end house_application_proof_l1706_170624


namespace max_segment_for_quadrilateral_l1706_170680

theorem max_segment_for_quadrilateral
  (a b c d : ℝ)
  (total_length : a + b + c + d = 2)
  (ordered_segments : a ≤ b ∧ b ≤ c ∧ c ≤ d) :
  (∃ (x : ℝ), x < 1 ∧
    (∀ (y : ℝ), y < x →
      (a + b > y ∧ a + c > y ∧ a + d > y ∧
       b + c > y ∧ b + d > y ∧ c + d > y))) ∧
  (∀ (z : ℝ), z ≥ 1 →
    ¬(a + b > z ∧ a + c > z ∧ a + d > z ∧
      b + c > z ∧ b + d > z ∧ c + d > z)) :=
by sorry

end max_segment_for_quadrilateral_l1706_170680


namespace arithmetic_sequence_fifth_term_l1706_170683

/-- An arithmetic sequence is a sequence where the difference between
    any two consecutive terms is constant. -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Given an arithmetic sequence where the 10th term is 3 and the 12th term is 9,
    the 5th term is -12. -/
theorem arithmetic_sequence_fifth_term
  (a : ℕ → ℝ)
  (h_arith : ArithmeticSequence a)
  (h_10th : a 10 = 3)
  (h_12th : a 12 = 9) :
  a 5 = -12 := by
  sorry

end arithmetic_sequence_fifth_term_l1706_170683


namespace solution_value_l1706_170667

theorem solution_value (a b : ℝ) (h : 2 * (-3) - a + 2 * b = 0) : 
  2 * a - 4 * b + 1 = -11 := by
sorry

end solution_value_l1706_170667


namespace max_imag_part_is_sin_45_l1706_170630

-- Define the complex polynomial
def f (z : ℂ) : ℂ := z^6 - z^4 + z^2 - 1

-- Define the set of roots
def roots : Set ℂ := {z : ℂ | f z = 0}

-- Theorem statement
theorem max_imag_part_is_sin_45 :
  ∃ (z : ℂ), z ∈ roots ∧ 
    ∀ (w : ℂ), w ∈ roots → Complex.im w ≤ Complex.im z ∧ 
      Complex.im z = Real.sin (π/4) :=
sorry

end max_imag_part_is_sin_45_l1706_170630


namespace repeating_decimal_properties_l1706_170665

/-- Represents a repeating decimal with a 3-digit non-repeating part and a 4-digit repeating part -/
structure RepeatingDecimal where
  N : ℕ  -- Non-repeating part (3 digits)
  M : ℕ  -- Repeating part (4 digits)

variable (R : RepeatingDecimal)

/-- The decimal expansion of R -/
noncomputable def decimal_expansion (R : RepeatingDecimal) : ℝ := sorry

theorem repeating_decimal_properties (R : RepeatingDecimal) :
  -- 1. R = 0.NMM... is a correct representation
  decimal_expansion R = (R.N : ℝ) / 1000 + (R.M : ℝ) / 10000 / (1 - 1 / 10000) ∧
  -- 2. 10^3R = N.MMM... is a correct representation
  1000 * decimal_expansion R = R.N + (R.M : ℝ) / 10000 / (1 - 1 / 10000) ∧
  -- 3. 10^7R ≠ NMN.MMM...
  10000000 * decimal_expansion R ≠ (R.N * 1000000 + R.M * 100 + R.N) + (R.M : ℝ) / 10000 / (1 - 1 / 10000) ∧
  -- 4. 10^3(10^4-1)R ≠ 10^4N - M
  1000 * (10000 - 1) * decimal_expansion R ≠ 10000 * R.N - R.M :=
sorry

end repeating_decimal_properties_l1706_170665


namespace alice_bob_meet_l1706_170637

/-- The number of points on the circle -/
def n : ℕ := 18

/-- Alice's clockwise movement per turn -/
def alice_move : ℕ := 7

/-- Bob's counterclockwise movement per turn -/
def bob_move : ℕ := 13

/-- The number of turns after which Alice and Bob meet -/
def meeting_turns : ℕ := 9

theorem alice_bob_meet :
  (meeting_turns * alice_move) % n = (meeting_turns * (n - bob_move)) % n :=
sorry

end alice_bob_meet_l1706_170637


namespace largest_green_socks_l1706_170641

theorem largest_green_socks (g y : ℕ) :
  let t := g + y
  (t ≤ 2023) →
  ((g * (g - 1) + y * (y - 1)) / (t * (t - 1)) = 1/3) →
  g ≤ 990 ∧ ∃ (g' y' : ℕ), g' = 990 ∧ y' + g' ≤ 2023 ∧
    ((g' * (g' - 1) + y' * (y' - 1)) / ((g' + y') * (g' + y' - 1)) = 1/3) :=
by sorry

end largest_green_socks_l1706_170641


namespace tangent_slope_implies_a_l1706_170601

-- Define the curve
def f (a : ℝ) (x : ℝ) : ℝ := x^4 + a*x^2 + 1

-- Define the derivative of the curve
def f' (a : ℝ) (x : ℝ) : ℝ := 4*x^3 + 2*a*x

theorem tangent_slope_implies_a (a : ℝ) :
  f a (-1) = a + 2 →  -- The curve passes through the point (-1, a+2)
  f' a (-1) = 8 →     -- The slope of the tangent line at x = -1 is 8
  a = -6 :=           -- Then a must equal -6
by
  sorry


end tangent_slope_implies_a_l1706_170601


namespace rhombus_longer_diagonal_l1706_170657

/-- A rhombus with side length 65 and shorter diagonal 60 has a longer diagonal of 110 -/
theorem rhombus_longer_diagonal (side_length : ℝ) (shorter_diagonal : ℝ) (longer_diagonal : ℝ)
  (h1 : side_length = 65)
  (h2 : shorter_diagonal = 60)
  (h3 : longer_diagonal * longer_diagonal / 4 + shorter_diagonal * shorter_diagonal / 4 = side_length * side_length) :
  longer_diagonal = 110 := by
  sorry

end rhombus_longer_diagonal_l1706_170657


namespace smallest_divisible_number_l1706_170684

theorem smallest_divisible_number : ∃ (n : ℕ), 
  (n > 2014) ∧ 
  (∀ k : ℕ, k < 10 → n % k = 0) ∧
  (∀ m : ℕ, m > 2014 ∧ m < n → ∃ j : ℕ, j < 10 ∧ m % j ≠ 0) ∧
  n = 2014506 := by
sorry

end smallest_divisible_number_l1706_170684


namespace bethany_portraits_l1706_170629

/-- The number of portraits Bethany saw at the museum -/
def num_portraits : ℕ := 16

/-- The number of still lifes Bethany saw at the museum -/
def num_still_lifes : ℕ := 4 * num_portraits

/-- The total number of paintings Bethany saw at the museum -/
def total_paintings : ℕ := 80

theorem bethany_portraits :
  num_portraits + num_still_lifes = total_paintings ∧
  num_still_lifes = 4 * num_portraits →
  num_portraits = 16 := by sorry

end bethany_portraits_l1706_170629


namespace intersection_A_complement_B_A_subset_B_iff_l1706_170670

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | x ≤ a + 3}
def B : Set ℝ := {x | x < -1 ∨ x > 5}

-- Theorem 1
theorem intersection_A_complement_B (a : ℝ) (h : a = -2) :
  A a ∩ (Set.univ \ B) = {x : ℝ | -1 ≤ x ∧ x ≤ 1} := by sorry

-- Theorem 2
theorem A_subset_B_iff (a : ℝ) :
  A a ∩ B = A a ↔ a < -4 := by sorry

end intersection_A_complement_B_A_subset_B_iff_l1706_170670


namespace proposition_range_l1706_170626

-- Define the propositions p and q
def p (a : ℝ) : Prop := ∀ x : ℝ, x^2 + x > a
def q (a : ℝ) : Prop := ∃ x : ℝ, x^2 + 2*a*x + 2 - a = 0

-- Theorem statement
theorem proposition_range (a : ℝ) : 
  (¬(p a) ∨ q a) = false → -2 < a ∧ a < -1/4 := by
  sorry

end proposition_range_l1706_170626


namespace equation_solution_l1706_170699

theorem equation_solution : ∃ x : ℝ, 24 - 4 * 2 = 3 + x ∧ x = 13 := by sorry

end equation_solution_l1706_170699


namespace rachels_homework_l1706_170611

/-- Rachel's homework problem -/
theorem rachels_homework (reading_pages : ℕ) (math_pages : ℕ) : 
  reading_pages = 4 → reading_pages = math_pages + 1 → math_pages = 3 :=
by sorry

end rachels_homework_l1706_170611


namespace plane_equation_proof_l1706_170640

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents the coefficients of a plane equation Ax + By + Cz + D = 0 -/
structure PlaneCoefficients where
  A : ℤ
  B : ℤ
  C : ℤ
  D : ℤ

/-- Check if a point lies on a plane -/
def pointLiesOnPlane (p : Point3D) (coeff : PlaneCoefficients) : Prop :=
  coeff.A * p.x + coeff.B * p.y + coeff.C * p.z + coeff.D = 0

/-- The greatest common divisor of the absolute values of the coefficients is 1 -/
def coefficientsAreCoprime (coeff : PlaneCoefficients) : Prop :=
  Nat.gcd (Int.natAbs coeff.A) (Nat.gcd (Int.natAbs coeff.B) (Nat.gcd (Int.natAbs coeff.C) (Int.natAbs coeff.D))) = 1

theorem plane_equation_proof (p1 p2 p3 : Point3D) (coeff : PlaneCoefficients) : 
  p1 = ⟨2, -1, 3⟩ →
  p2 = ⟨0, -1, 5⟩ →
  p3 = ⟨-1, -3, 4⟩ →
  coeff = ⟨1, 2, -1, 3⟩ →
  pointLiesOnPlane p1 coeff ∧
  pointLiesOnPlane p2 coeff ∧
  pointLiesOnPlane p3 coeff ∧
  coeff.A > 0 ∧
  coefficientsAreCoprime coeff :=
by sorry

end plane_equation_proof_l1706_170640


namespace grain_oil_production_growth_l1706_170605

theorem grain_oil_production_growth (x : ℝ) : 
  (450000 * (1 + x)^2 = 500000) ↔ 
  (∃ (y : ℝ), 450000 * (1 + x) = y ∧ y * (1 + x) = 500000) :=
sorry

end grain_oil_production_growth_l1706_170605


namespace count_squares_on_marked_grid_l1706_170603

/-- A point on a 2D grid -/
structure GridPoint where
  x : ℕ
  y : ℕ

/-- A square grid with marked points -/
structure MarkedGrid where
  size : ℕ
  points : List GridPoint

/-- A square formed by four points -/
structure Square where
  p1 : GridPoint
  p2 : GridPoint
  p3 : GridPoint
  p4 : GridPoint

/-- Check if four points form a valid square -/
def isValidSquare (s : Square) : Bool :=
  sorry

/-- Count the number of valid squares that can be formed from a list of points -/
def countValidSquares (points : List GridPoint) : ℕ :=
  sorry

/-- The main theorem -/
theorem count_squares_on_marked_grid :
  ∀ (g : MarkedGrid),
    g.size = 4 ∧ 
    g.points.length = 12 ∧ 
    (∀ p ∈ g.points, p.x < 4 ∧ p.y < 4) ∧
    (∀ x y, x = 0 ∨ x = 3 ∨ y = 0 ∨ y = 3 → ¬∃ p ∈ g.points, p.x = x ∧ p.y = y) →
    countValidSquares g.points = 11 :=
  sorry

end count_squares_on_marked_grid_l1706_170603


namespace three_of_a_kind_probability_l1706_170633

/-- Represents a standard deck of cards -/
def StandardDeck : ℕ := 52

/-- Represents the number of cards in a hand -/
def HandSize : ℕ := 5

/-- Represents the number of ranks in a standard deck -/
def NumRanks : ℕ := 13

/-- Represents the number of cards of each rank in a standard deck -/
def CardsPerRank : ℕ := 4

/-- Calculates the probability of drawing a "three of a kind" hand with two other cards of different ranks -/
def probThreeOfAKind : ℚ :=
  let totalHands := Nat.choose StandardDeck HandSize
  let threeOfAKindHands := NumRanks * Nat.choose CardsPerRank 3 * (NumRanks - 1) * CardsPerRank * (NumRanks - 2) * CardsPerRank
  threeOfAKindHands / totalHands

theorem three_of_a_kind_probability : probThreeOfAKind = 1719 / 40921 := by
  sorry

end three_of_a_kind_probability_l1706_170633


namespace min_sum_with_log_condition_l1706_170691

theorem min_sum_with_log_condition (a b : ℝ) 
  (ha : a > 0) (hb : b > 0) (h_log : Real.log a + Real.log b = Real.log (a + b)) : 
  ∀ x y : ℝ, x > 0 → y > 0 → Real.log x + Real.log y = Real.log (x + y) → a + b ≤ x + y ∧ a + b = 4 := by
  sorry

end min_sum_with_log_condition_l1706_170691


namespace vector_angle_difference_l1706_170647

theorem vector_angle_difference (α β : Real) (a b : Fin 2 → Real) 
  (h1 : 0 < α) (h2 : α < β) (h3 : β < π)
  (h4 : a = λ i => if i = 0 then Real.cos α else Real.sin α)
  (h5 : b = λ i => if i = 0 then Real.cos β else Real.sin β)
  (h6 : ‖(2 : Real) • a + b‖ = ‖a - (2 : Real) • b‖) :
  β - α = π / 2 := by
sorry

end vector_angle_difference_l1706_170647


namespace distance_between_places_l1706_170607

/-- The distance between places A and B in kilometers -/
def distance : ℝ := 150

/-- The speed of bicycling in kilometers per hour -/
def bicycle_speed : ℝ := 15

/-- The speed of walking in kilometers per hour -/
def walking_speed : ℝ := 5

/-- The time difference between return trip and going trip in hours -/
def time_difference : ℝ := 2

theorem distance_between_places : 
  ∃ (return_time : ℝ),
    (distance / 2 / bicycle_speed + distance / 2 / walking_speed = return_time - time_difference) ∧
    (distance = return_time / 3 * bicycle_speed + 2 * return_time / 3 * walking_speed) :=
by sorry

end distance_between_places_l1706_170607


namespace max_x_squared_y_l1706_170634

theorem max_x_squared_y (x y : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hsum : x + 3 * y = 9) :
  x^2 * y ≤ 36 :=
sorry

end max_x_squared_y_l1706_170634


namespace expression_defined_iff_l1706_170614

def is_defined (x : ℝ) : Prop :=
  x > 2 ∧ x < 5

theorem expression_defined_iff (x : ℝ) :
  is_defined x ↔ (∃ y : ℝ, y = (Real.log (5 - x)) / Real.sqrt (x - 2)) :=
by sorry

end expression_defined_iff_l1706_170614


namespace salary_increase_l1706_170673

/-- Given a salary increase of 100% resulting in a new salary of $80,
    prove that the original salary was $40. -/
theorem salary_increase (new_salary : ℝ) (increase_percentage : ℝ) : 
  new_salary = 80 ∧ increase_percentage = 100 → 
  new_salary / 2 = 40 := by
  sorry

end salary_increase_l1706_170673


namespace largest_divisor_of_m_l1706_170613

theorem largest_divisor_of_m (m : ℕ+) (h : (m.val ^ 3) % 847 = 0) : 
  ∃ (k : ℕ+), k.val = 77 ∧ k.val ∣ m.val ∧ ∀ (d : ℕ+), d.val ∣ m.val → d.val ≤ k.val :=
sorry

end largest_divisor_of_m_l1706_170613


namespace special_pentagon_exists_l1706_170677

/-- A pentagon that can be divided into three parts by one straight cut,
    such that two of the parts can be combined to form the third part. -/
structure SpecialPentagon where
  /-- The vertices of the pentagon -/
  vertices : Fin 5 → ℝ × ℝ
  /-- The cut line that divides the pentagon -/
  cut_line : ℝ × ℝ → ℝ × ℝ → Prop
  /-- The three parts resulting from the cut -/
  parts : Fin 3 → Set (ℝ × ℝ)
  /-- Proof that the cut line divides the pentagon into exactly three parts -/
  valid_division : sorry
  /-- Proof that two of the parts can be combined to form the third part -/
  recombination : sorry

/-- Theorem stating the existence of a special pentagon -/
theorem special_pentagon_exists : ∃ (p : SpecialPentagon), True := by
  sorry

end special_pentagon_exists_l1706_170677


namespace parabola_and_triangle_area_l1706_170695

/-- Parabola C: y² = 2px where p > 0 -/
structure Parabola where
  p : ℝ
  hp : p > 0

/-- Point on the parabola -/
structure PointOnParabola (c : Parabola) where
  x : ℝ
  y : ℝ
  hy : y^2 = 2 * c.p * x

/-- Circle E: (x-1)² + y² = 1 -/
def CircleE (x y : ℝ) : Prop :=
  (x - 1)^2 + y^2 = 1

/-- Theorem about the parabola equation and minimum area of triangle -/
theorem parabola_and_triangle_area (c : Parabola) (m : PointOnParabola c)
    (h_dist : (m.x - 2)^2 + m.y^2 = 3) (h_x : m.x > 2) :
    c.p = 1 ∧ ∃ (a b : ℝ), CircleE 0 a ∧ CircleE 0 b ∧
    (∀ a' b' : ℝ, CircleE 0 a' ∧ CircleE 0 b' →
      1/2 * |a - b| * m.x ≤ 1/2 * |a' - b'| * m.x) ∧
    1/2 * |a - b| * m.x = 8 := by
  sorry

end parabola_and_triangle_area_l1706_170695


namespace train_length_problem_l1706_170693

/-- Proves that given two trains moving in opposite directions with specified speeds and time to pass,
    the length of the first train is 150 meters. -/
theorem train_length_problem (v1 v2 l2 t : ℝ) (h1 : v1 = 80) (h2 : v2 = 70) (h3 : l2 = 100) 
    (h4 : t = 5.999520038396928) : ∃ l1 : ℝ, l1 = 150 ∧ (v1 + v2) * t * (5/18) = l1 + l2 := by
  sorry

end train_length_problem_l1706_170693


namespace polynomial_divisibility_l1706_170668

theorem polynomial_divisibility (x₀ a₁ a₂ a₃ a₄ : ℝ) 
  (h1 : x₀^4 + a₁*x₀^3 + a₂*x₀^2 + a₃*x₀ + a₄ = 0)
  (h2 : 4*x₀^3 + 3*a₁*x₀^2 + 2*a₂*x₀ + a₃ = 0) :
  ∃ g : ℝ → ℝ, ∀ x : ℝ, 
    x^4 + a₁*x^3 + a₂*x^2 + a₃*x + a₄ = (x - x₀)^2 * g x :=
by sorry

end polynomial_divisibility_l1706_170668


namespace max_profit_morel_purchase_l1706_170643

/-- Represents the purchase and profit calculation for Morel mushrooms. -/
structure MorelPurchase where
  freshPrice : ℝ  -- Purchase price of fresh Morel mushrooms (RMB/kg)
  driedPrice : ℝ  -- Purchase price of dried Morel mushrooms (RMB/kg)
  freshRetail : ℝ  -- Retail price of fresh Morel mushrooms (RMB/kg)
  driedRetail : ℝ  -- Retail price of dried Morel mushrooms (RMB/kg)
  totalQuantity : ℝ  -- Total quantity to purchase (kg)

/-- Calculates the profit for a given purchase plan. -/
def calculateProfit (p : MorelPurchase) (freshQuant : ℝ) : ℝ :=
  let driedQuant := p.totalQuantity - freshQuant
  (p.freshRetail - p.freshPrice) * freshQuant + (p.driedRetail - p.driedPrice) * driedQuant

/-- Theorem stating that the maximum profit is achieved with the specified quantities. -/
theorem max_profit_morel_purchase (p : MorelPurchase)
    (h1 : p.freshPrice = 80)
    (h2 : p.driedPrice = 240)
    (h3 : p.freshRetail = 100)
    (h4 : p.driedRetail = 280)
    (h5 : p.totalQuantity = 1500) :
    ∃ (maxProfit : ℝ) (optimalFresh : ℝ),
      maxProfit = 37500 ∧
      optimalFresh = 1125 ∧
      ∀ (freshQuant : ℝ), 0 ≤ freshQuant ∧ freshQuant ≤ p.totalQuantity ∧
        3 * (p.totalQuantity - freshQuant) ≤ freshQuant →
        calculateProfit p freshQuant ≤ maxProfit := by
  sorry


end max_profit_morel_purchase_l1706_170643


namespace least_perimeter_triangle_l1706_170661

theorem least_perimeter_triangle (a b c : ℕ) : 
  a = 40 → b = 48 → c > 0 → a + b > c → a + c > b → b + c > a → 
  (∀ x : ℕ, x > 0 → a + b > x → a + x > b → b + x > a → a + b + x ≥ a + b + c) →
  a + b + c = 97 := by sorry

end least_perimeter_triangle_l1706_170661


namespace gcd_50421_35343_l1706_170682

theorem gcd_50421_35343 : Nat.gcd 50421 35343 = 23 := by
  sorry

end gcd_50421_35343_l1706_170682


namespace max_sum_of_pairwise_sums_l1706_170653

/-- Given a set of four numbers with six pairwise sums, find the maximum value of x + y -/
theorem max_sum_of_pairwise_sums (a b c d : ℝ) : 
  let sums := [a + b, a + c, a + d, b + c, b + d, c + d]
  ∃ (perm : List ℝ), perm.Perm sums ∧ 
    perm.take 4 = [210, 336, 294, 252] →
  (∃ (x y : ℝ), x ∈ perm.drop 4 ∧ y ∈ perm.drop 4 ∧ x + y ≤ 798) ∧
  (∃ (a' b' c' d' : ℝ), 
    let sums' := [a' + b', a' + c', a' + d', b' + c', b' + d', c' + d']
    ∃ (perm' : List ℝ), perm'.Perm sums' ∧ 
      perm'.take 4 = [210, 336, 294, 252] ∧
      ∃ (x' y' : ℝ), x' ∈ perm'.drop 4 ∧ y' ∈ perm'.drop 4 ∧ x' + y' = 798) := by
  sorry


end max_sum_of_pairwise_sums_l1706_170653
