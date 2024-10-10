import Mathlib

namespace largest_integer_solution_l1739_173950

theorem largest_integer_solution (m : ℤ) : (2 * m + 7 ≤ 3) → m ≤ -2 ∧ ∀ k : ℤ, (2 * k + 7 ≤ 3) → k ≤ m := by
  sorry

end largest_integer_solution_l1739_173950


namespace sin_alpha_minus_pi_third_l1739_173977

theorem sin_alpha_minus_pi_third (α : ℝ) (h : Real.cos (α + π/6) = -1/3) : 
  Real.sin (α - π/3) = 1/3 := by
  sorry

end sin_alpha_minus_pi_third_l1739_173977


namespace perfect_cube_divisibility_l1739_173926

theorem perfect_cube_divisibility (a b : ℕ) (h1 : 0 < b) (h2 : b < a) 
  (h3 : (a^3 + b^3 + a*b) % (a*b*(a-b)) = 0) : 
  ∃ (k : ℕ), a * b = k^3 :=
sorry

end perfect_cube_divisibility_l1739_173926


namespace a4_plus_b4_equals_228_l1739_173934

theorem a4_plus_b4_equals_228 (a b : ℝ) 
  (h1 : (a^2 - b^2)^2 = 100) 
  (h2 : (a^3 * b^3) = 512) : 
  a^4 + b^4 = 228 := by
sorry

end a4_plus_b4_equals_228_l1739_173934


namespace difference_of_squares_l1739_173935

theorem difference_of_squares (x y : ℝ) : (x + 2*y) * (-2*y + x) = x^2 - 4*y^2 := by
  sorry

end difference_of_squares_l1739_173935


namespace science_class_ends_at_350pm_l1739_173959

-- Define the start time and class durations
def school_start_time : Nat := 12 * 60  -- 12:00 pm in minutes
def maths_duration : Nat := 45
def history_duration : Nat := 75  -- 1 hour and 15 minutes
def geography_duration : Nat := 30
def science_duration : Nat := 50
def break_duration : Nat := 10

-- Define a function to calculate the end time of Science class
def science_class_end_time : Nat :=
  school_start_time +
  maths_duration + break_duration +
  history_duration + break_duration +
  geography_duration + break_duration +
  science_duration

-- Convert minutes to hours and minutes
def minutes_to_time (minutes : Nat) : String :=
  let hours := minutes / 60
  let mins := minutes % 60
  s!"{hours}:{mins}"

-- Theorem to prove
theorem science_class_ends_at_350pm :
  minutes_to_time science_class_end_time = "3:50" :=
by sorry

end science_class_ends_at_350pm_l1739_173959


namespace exactly_one_black_and_exactly_two_red_mutually_exclusive_but_not_complementary_l1739_173955

/-- Represents the color of a ball -/
inductive Color
| Red
| Black

/-- Represents the outcome of drawing two balls -/
structure Outcome :=
  (first second : Color)

/-- The sample space of all possible outcomes when drawing two balls -/
def sampleSpace : Finset Outcome :=
  sorry

/-- The event of having exactly one black ball -/
def exactlyOneBlack (outcome : Outcome) : Prop :=
  (outcome.first = Color.Black ∧ outcome.second = Color.Red) ∨
  (outcome.first = Color.Red ∧ outcome.second = Color.Black)

/-- The event of having exactly two red balls -/
def exactlyTwoRed (outcome : Outcome) : Prop :=
  outcome.first = Color.Red ∧ outcome.second = Color.Red

/-- Two events are mutually exclusive if they cannot occur simultaneously -/
def mutuallyExclusive (e1 e2 : Outcome → Prop) : Prop :=
  ∀ outcome, ¬(e1 outcome ∧ e2 outcome)

/-- Two events are complementary if one of them always occurs -/
def complementary (e1 e2 : Outcome → Prop) : Prop :=
  ∀ outcome, e1 outcome ∨ e2 outcome

theorem exactly_one_black_and_exactly_two_red_mutually_exclusive_but_not_complementary :
  mutuallyExclusive exactlyOneBlack exactlyTwoRed ∧
  ¬complementary exactlyOneBlack exactlyTwoRed :=
sorry

end exactly_one_black_and_exactly_two_red_mutually_exclusive_but_not_complementary_l1739_173955


namespace quadratic_root_value_l1739_173914

theorem quadratic_root_value (m : ℝ) : 
  ((m - 2) * 1^2 + 4 * 1 - m^2 = 0) ∧ (m ≠ 2) → m = -1 :=
by sorry

end quadratic_root_value_l1739_173914


namespace direct_proportion_percentage_change_l1739_173987

theorem direct_proportion_percentage_change 
  (x y : ℝ) (q : ℝ) (c : ℝ) (hx : x > 0) (hy : y > 0) (hq : q > 0) (hc : c > 0) 
  (h_prop : y = c * x) :
  let x' := x * (1 - q / 100)
  let y' := c * x'
  (y' - y) / y * 100 = q := by
sorry

end direct_proportion_percentage_change_l1739_173987


namespace city_pairing_equality_l1739_173902

/-- The number of ways to form r pairs in City A -/
def A (n r : ℕ) : ℕ := sorry

/-- The number of ways to form r pairs in City B -/
def B (n r : ℕ) : ℕ := sorry

/-- Girls in City B know a specific number of boys -/
def girls_know_boys (i : ℕ) : ℕ := 2 * i - 1

theorem city_pairing_equality (n : ℕ) (hn : n ≥ 1) :
  ∀ r : ℕ, 1 ≤ r ∧ r ≤ n → A n r = B n r := by
  sorry

end city_pairing_equality_l1739_173902


namespace nickel_dime_difference_l1739_173945

/-- The value of one dollar in cents -/
def dollar : ℕ := 100

/-- The value of a nickel in cents -/
def nickel : ℕ := 5

/-- The value of a dime in cents -/
def dime : ℕ := 10

/-- The number of coins needed to make one dollar using only coins of a given value -/
def coinsNeeded (coinValue : ℕ) : ℕ := dollar / coinValue

theorem nickel_dime_difference :
  coinsNeeded nickel - coinsNeeded dime = 10 := by sorry

end nickel_dime_difference_l1739_173945


namespace quadratic_roots_property_l1739_173900

theorem quadratic_roots_property :
  ∀ x₁ x₂ : ℝ,
  x₁^2 - 3*x₁ - 4 = 0 →
  x₂^2 - 3*x₂ - 4 = 0 →
  x₁ ≠ x₂ →
  x₁*x₂ - x₁ - x₂ = -7 := by
sorry

end quadratic_roots_property_l1739_173900


namespace sin_cos_15_product_l1739_173971

theorem sin_cos_15_product : 
  (Real.sin (15 * π / 180) + Real.cos (15 * π / 180)) * 
  (Real.sin (15 * π / 180) - Real.cos (15 * π / 180)) = 
  -Real.sqrt 3 / 2 := by
  sorry

end sin_cos_15_product_l1739_173971


namespace geometric_sequence_ratio_l1739_173957

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_ratio (a : ℕ → ℝ) :
  is_geometric_sequence a →
  (∀ n : ℕ, a (n + 1) < a n) →
  a 2 * a 8 = 6 →
  a 4 + a 6 = 5 →
  a 3 / a 7 = 9 / 4 := by
  sorry

end geometric_sequence_ratio_l1739_173957


namespace fraction_equality_l1739_173938

theorem fraction_equality (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a - 4*b ≠ 0) (h4 : 4*a - b ≠ 0)
  (h5 : (4*a + b) / (a - 4*b) = 3) : (a + 4*b) / (4*a - b) = 9/53 := by
  sorry

end fraction_equality_l1739_173938


namespace root_product_value_l1739_173963

theorem root_product_value (m n : ℝ) : 
  m^2 - 3*m - 2 = 0 → 
  n^2 - 3*n - 2 = 0 → 
  (7*m^2 - 21*m - 3)*(3*n^2 - 9*n + 5) = 121 := by
sorry

end root_product_value_l1739_173963


namespace expression_nonnegative_l1739_173912

theorem expression_nonnegative (x : ℝ) : 
  (x - 20*x^2 + 100*x^3) / (16 - 2*x^3) ≥ 0 ↔ x ∈ Set.Ici 0 ∩ Set.Iio 2 := by
  sorry

end expression_nonnegative_l1739_173912


namespace exponent_multiplication_l1739_173907

theorem exponent_multiplication (x : ℝ) : x^3 * x^2 = x^5 := by
  sorry

end exponent_multiplication_l1739_173907


namespace simplify_expression_l1739_173958

theorem simplify_expression (a b c : ℝ) (h : a + b + c = 0) :
  a * (1 / b + 1 / c) + b * (1 / c + 1 / a) + c * (1 / a + 1 / b) + 3 = 0 := by
  sorry

end simplify_expression_l1739_173958


namespace distance_sum_bounds_l1739_173948

/-- Given points A, B, C in a 2D plane and a point P satisfying x^2 + y^2 ≤ 4,
    the sum of squared distances from P to A, B, and C is between 72 and 88. -/
theorem distance_sum_bounds (x y : ℝ) :
  x^2 + y^2 ≤ 4 →
  72 ≤ ((x + 2)^2 + (y - 2)^2) + ((x + 2)^2 + (y - 6)^2) + ((x - 4)^2 + (y + 2)^2) ∧
  ((x + 2)^2 + (y - 2)^2) + ((x + 2)^2 + (y - 6)^2) + ((x - 4)^2 + (y + 2)^2) ≤ 88 :=
by sorry

end distance_sum_bounds_l1739_173948


namespace union_M_N_l1739_173962

def M : Set ℕ := {0, 1, 2}

def N : Set ℕ := {x | ∃ a ∈ M, x = 2 * a}

theorem union_M_N : M ∪ N = {0, 1, 2, 4} := by sorry

end union_M_N_l1739_173962


namespace smallest_four_digit_mod_4_3_l1739_173910

theorem smallest_four_digit_mod_4_3 :
  ∀ n : ℕ, 1000 ≤ n ∧ n ≡ 3 [ZMOD 4] → 1003 ≤ n :=
by sorry

end smallest_four_digit_mod_4_3_l1739_173910


namespace function_equals_square_l1739_173903

-- Define the property that f has the same number of intersections as x^2 with any line
def SameIntersections (f : ℝ → ℝ) : Prop :=
  ∀ (m c : ℝ), (∃ (x : ℝ), f x = m * x + c) ↔ (∃ (x : ℝ), x^2 = m * x + c)

-- State the theorem
theorem function_equals_square (f : ℝ → ℝ) (h : SameIntersections f) : 
  ∀ x : ℝ, f x = x^2 := by
sorry

end function_equals_square_l1739_173903


namespace total_gross_profit_calculation_l1739_173966

/-- Represents the sales prices and costs for an item over three months -/
structure ItemData :=
  (sales_prices : Fin 3 → ℕ)
  (costs : Fin 3 → ℕ)
  (gross_profit_percentage : ℕ)

/-- Calculates the gross profit for an item in a given month -/
def gross_profit (item : ItemData) (month : Fin 3) : ℕ :=
  item.sales_prices month - item.costs month

/-- Calculates the total gross profit for an item over three months -/
def total_gross_profit (item : ItemData) : ℕ :=
  (gross_profit item 0) + (gross_profit item 1) + (gross_profit item 2)

/-- The main theorem to prove -/
theorem total_gross_profit_calculation 
  (item_a item_b item_c item_d : ItemData)
  (ha : item_a.sales_prices = ![44, 47, 50])
  (hac : item_a.costs = ![20, 22, 25])
  (hap : item_a.gross_profit_percentage = 120)
  (hb : item_b.sales_prices = ![60, 63, 65])
  (hbc : item_b.costs = ![30, 33, 35])
  (hbp : item_b.gross_profit_percentage = 150)
  (hc : item_c.sales_prices = ![80, 83, 85])
  (hcc : item_c.costs = ![40, 42, 45])
  (hcp : item_c.gross_profit_percentage = 100)
  (hd : item_d.sales_prices = ![100, 103, 105])
  (hdc : item_d.costs = ![50, 52, 55])
  (hdp : item_d.gross_profit_percentage = 130) :
  total_gross_profit item_a + total_gross_profit item_b + 
  total_gross_profit item_c + total_gross_profit item_d = 436 := by
  sorry

end total_gross_profit_calculation_l1739_173966


namespace min_flash_drives_l1739_173991

theorem min_flash_drives (total_files : ℕ) (drive_capacity : ℚ)
  (files_0_9MB : ℕ) (files_0_8MB : ℕ) (files_0_6MB : ℕ) :
  total_files = files_0_9MB + files_0_8MB + files_0_6MB →
  drive_capacity = 2.88 →
  files_0_9MB = 5 →
  files_0_8MB = 18 →
  files_0_6MB = 17 →
  (∃ min_drives : ℕ, 
    min_drives = 13 ∧
    min_drives * drive_capacity ≥ 
      (files_0_9MB * 0.9 + files_0_8MB * 0.8 + files_0_6MB * 0.6) ∧
    ∀ n : ℕ, n < min_drives → 
      n * drive_capacity < 
        (files_0_9MB * 0.9 + files_0_8MB * 0.8 + files_0_6MB * 0.6)) :=
by
  sorry

end min_flash_drives_l1739_173991


namespace triangle_third_side_length_l1739_173972

theorem triangle_third_side_length (a b : ℝ) (ha : a = 6.31) (hb : b = 0.82) :
  ∃! c : ℕ, (c : ℝ) > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b ∧ c = 6 :=
by sorry

end triangle_third_side_length_l1739_173972


namespace polynomial_coefficient_sum_l1739_173995

theorem polynomial_coefficient_sum (a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, a₁ * (x - 1)^4 + a₂ * (x - 1)^3 + a₃ * (x - 1)^2 + a₄ * (x - 1) + a₅ = x^4) →
  a₂ - a₃ + a₄ = 2 := by
sorry


end polynomial_coefficient_sum_l1739_173995


namespace equation_system_solution_l1739_173920

theorem equation_system_solution (x y z : ℝ) 
  (eq1 : 3 * x - 4 * y - 2 * z = 0)
  (eq2 : x + 2 * y - 7 * z = 0)
  (z_nonzero : z ≠ 0) :
  (x^2 - 2*x*y) / (y^2 + 4*z^2) = -0.252 := by
  sorry

end equation_system_solution_l1739_173920


namespace smallest_n_with_constant_term_l1739_173964

theorem smallest_n_with_constant_term :
  ∃ (n : ℕ), n > 0 ∧ 
  (∀ (k : ℕ), k > 0 ∧ k < n → ¬∃ (r : ℕ), 2*k = 5*r) ∧
  (∃ (r : ℕ), 2*n = 5*r) ∧
  n = 5 := by
  sorry

end smallest_n_with_constant_term_l1739_173964


namespace locus_of_midpoints_l1739_173978

-- Define the circle L
structure Circle :=
  (center : ℝ × ℝ)
  (radius : ℝ)

-- Define a point Q inside the circle
def Q (L : Circle) : ℝ × ℝ :=
  sorry

-- Define the distance between two points
def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  sorry

-- State the theorem
theorem locus_of_midpoints (L : Circle) :
  -- Q is an interior point of L
  (distance (Q L) L.center < L.radius) →
  -- Q is not the center of L
  (Q L ≠ L.center) →
  -- The distance from Q to the center of L is one-third the radius of L
  (distance (Q L) L.center = L.radius / 3) →
  -- The locus of midpoints of all chords passing through Q is a complete circle
  ∃ (C : Circle),
    -- The center of the locus circle is Q
    C.center = Q L ∧
    -- The radius of the locus circle is r/6
    C.radius = L.radius / 6 :=
sorry

end locus_of_midpoints_l1739_173978


namespace profit_percentage_is_20_l1739_173989

-- Define the quantities and prices
def wheat1_quantity : ℝ := 30
def wheat1_price : ℝ := 11.50
def wheat2_quantity : ℝ := 20
def wheat2_price : ℝ := 14.25
def selling_price : ℝ := 15.12

-- Define the theorem
theorem profit_percentage_is_20 : 
  let total_cost := wheat1_quantity * wheat1_price + wheat2_quantity * wheat2_price
  let total_weight := wheat1_quantity + wheat2_quantity
  let cost_price_per_kg := total_cost / total_weight
  let profit_per_kg := selling_price - cost_price_per_kg
  let profit_percentage := (profit_per_kg / cost_price_per_kg) * 100
  profit_percentage = 20 := by sorry

end profit_percentage_is_20_l1739_173989


namespace half_abs_diff_squares_21_19_l1739_173943

theorem half_abs_diff_squares_21_19 : (1 / 2 : ℝ) * |21^2 - 19^2| = 40 := by sorry

end half_abs_diff_squares_21_19_l1739_173943


namespace min_correct_answers_for_score_l1739_173970

/-- Given a math test with the following conditions:
  * There are 16 total questions
  * 6 points are awarded for each correct answer
  * 2 points are deducted for each wrong answer
  * No points are deducted for unanswered questions
  * The student did not answer one question
  * The goal is to score more than 60 points

  This theorem proves that the minimum number of correct answers needed is 12. -/
theorem min_correct_answers_for_score (total_questions : ℕ) (correct_points : ℕ) (wrong_points : ℕ) 
  (unanswered : ℕ) (target_score : ℕ) : 
  total_questions = 16 → 
  correct_points = 6 → 
  wrong_points = 2 → 
  unanswered = 1 → 
  target_score = 60 → 
  ∃ (min_correct : ℕ), 
    (∀ (x : ℕ), x ≥ min_correct → 
      x * correct_points - (total_questions - unanswered - x) * wrong_points > target_score) ∧ 
    (∀ (y : ℕ), y < min_correct → 
      y * correct_points - (total_questions - unanswered - y) * wrong_points ≤ target_score) ∧
    min_correct = 12 := by
  sorry

end min_correct_answers_for_score_l1739_173970


namespace gift_wrap_sales_l1739_173932

theorem gift_wrap_sales (total_goal : ℕ) (grandmother_sales uncle_sales neighbor_sales : ℕ) : 
  total_goal = 45 ∧ 
  grandmother_sales = 1 ∧ 
  uncle_sales = 10 ∧ 
  neighbor_sales = 6 → 
  total_goal - (grandmother_sales + uncle_sales + neighbor_sales) = 28 := by
  sorry

end gift_wrap_sales_l1739_173932


namespace greatest_x_cube_less_than_2000_l1739_173937

theorem greatest_x_cube_less_than_2000 :
  ∃ (x : ℕ), x > 0 ∧ ∃ (k : ℕ), x = 5 * k ∧ x^3 < 2000 ∧
  ∀ (y : ℕ), y > 0 → (∃ (m : ℕ), y = 5 * m) → y^3 < 2000 → y ≤ x :=
by
  -- The proof goes here
  sorry

end greatest_x_cube_less_than_2000_l1739_173937


namespace james_spent_six_l1739_173908

/-- Calculates the total amount spent given the cost of milk, cost of bananas, and sales tax rate. -/
def total_spent (milk_cost banana_cost tax_rate : ℚ) : ℚ :=
  let subtotal := milk_cost + banana_cost
  let tax := subtotal * tax_rate
  subtotal + tax

/-- Proves that James spent $6 given the costs and tax rate. -/
theorem james_spent_six :
  let milk_cost : ℚ := 3
  let banana_cost : ℚ := 2
  let tax_rate : ℚ := 1/5
  total_spent milk_cost banana_cost tax_rate = 6 := by
  sorry

#eval total_spent 3 2 (1/5)

end james_spent_six_l1739_173908


namespace number_equation_l1739_173946

theorem number_equation (x : ℝ) : (9 * x) / 3 = 27 ↔ x = 9 := by
  sorry

end number_equation_l1739_173946


namespace horner_method_operations_count_l1739_173990

def horner_polynomial (x : ℝ) : ℝ := 9*x^6 + 12*x^5 + 7*x^4 + 54*x^3 + 34*x^2 + 9*x + 1

def horner_method_operations (p : ℝ → ℝ) : ℕ × ℕ :=
  match p with
  | f => (6, 6)  -- Placeholder for the actual implementation

theorem horner_method_operations_count :
  ∀ x : ℝ, horner_method_operations horner_polynomial = (6, 6) := by
  sorry

end horner_method_operations_count_l1739_173990


namespace limit_of_sequence_l1739_173947

def a (n : ℕ) : ℚ := (2 * n - 5 : ℚ) / (3 * n + 1)

theorem limit_of_sequence : 
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |a n - 2/3| < ε :=
by sorry

end limit_of_sequence_l1739_173947


namespace equation_solution_l1739_173956

theorem equation_solution (x y : ℚ) 
  (h : x^2 - 2*y - Real.sqrt 2*y = 17 - 4*Real.sqrt 2) : 
  2*x + y = 14 ∨ 2*x + y = -6 := by
  sorry

end equation_solution_l1739_173956


namespace pizza_fraction_eaten_l1739_173994

theorem pizza_fraction_eaten (total_slices : ℕ) (whole_slices_eaten : ℕ) (shared_slices : ℕ) :
  total_slices = 16 →
  whole_slices_eaten = 2 →
  shared_slices = 2 →
  (whole_slices_eaten : ℚ) / total_slices + (shared_slices : ℚ) / (2 * total_slices) = 3 / 16 := by
  sorry

end pizza_fraction_eaten_l1739_173994


namespace amaya_total_marks_l1739_173913

/-- Represents the marks scored in different subjects -/
structure Marks where
  arts : ℕ
  maths : ℕ
  music : ℕ
  social_studies : ℕ

/-- Calculates the total marks across all subjects -/
def total_marks (m : Marks) : ℕ :=
  m.arts + m.maths + m.music + m.social_studies

/-- Theorem stating the total marks Amaya scored given the conditions -/
theorem amaya_total_marks :
  ∀ (m : Marks),
    m.arts - m.maths = 20 →
    m.social_studies > m.music →
    m.music = 70 →
    m.maths = (9 * m.arts) / 10 →
    m.social_studies - m.music = 10 →
    total_marks m = 530 := by
  sorry

#check amaya_total_marks

end amaya_total_marks_l1739_173913


namespace sequence_sum_problem_l1739_173982

theorem sequence_sum_problem (N : ℤ) : 
  (995 : ℤ) + 997 + 999 + 1001 + 1003 = 5005 - N → N = 5 := by
sorry

end sequence_sum_problem_l1739_173982


namespace xoz_symmetry_of_M_l1739_173976

/-- Defines a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Defines the xoz plane symmetry operation -/
def xozPlaneSymmetry (p : Point3D) : Point3D :=
  { x := p.x, y := -p.y, z := p.z }

/-- Theorem: The symmetric point of M(5, 1, -2) with respect to the xoz plane is (5, -1, -2) -/
theorem xoz_symmetry_of_M :
  let M : Point3D := { x := 5, y := 1, z := -2 }
  xozPlaneSymmetry M = { x := 5, y := -1, z := -2 } := by
  sorry

end xoz_symmetry_of_M_l1739_173976


namespace brothers_age_proof_l1739_173979

def hannah_age : ℕ := 48
def num_brothers : ℕ := 3

theorem brothers_age_proof (brothers_age : ℕ) 
  (h1 : hannah_age = 2 * (num_brothers * brothers_age)) : 
  brothers_age = 8 := by
  sorry

end brothers_age_proof_l1739_173979


namespace carter_performs_30_nights_l1739_173909

/-- The number of nights Carter performs, given his drum stick usage pattern --/
def carter_performance_nights (sticks_per_show : ℕ) (sticks_tossed : ℕ) (total_sticks : ℕ) : ℕ :=
  total_sticks / (sticks_per_show + sticks_tossed)

/-- Theorem stating that Carter performs for 30 nights under the given conditions --/
theorem carter_performs_30_nights :
  carter_performance_nights 5 6 330 = 30 := by
  sorry

end carter_performs_30_nights_l1739_173909


namespace rectangle_ratio_l1739_173965

/-- Given two rectangles A and B with sides (a, b) and (c, d) respectively,
    if the ratio of their areas is 0.16 and a/c = b/d,
    then a/c = b/d = 0.4 -/
theorem rectangle_ratio (a b c d : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : d > 0)
  (h5 : (a * b) / (c * d) = 0.16) (h6 : a / c = b / d) :
  a / c = 0.4 ∧ b / d = 0.4 := by
sorry

end rectangle_ratio_l1739_173965


namespace fourth_section_size_l1739_173969

/-- The number of students in the fourth section of a chemistry class -/
def fourth_section_students : ℕ :=
  -- We'll define this later in the theorem
  42

/-- Represents the data for a chemistry class section -/
structure Section where
  students : ℕ
  mean_marks : ℚ

/-- Calculates the total marks for a section -/
def total_marks (s : Section) : ℚ :=
  s.students * s.mean_marks

/-- Represents the data for all sections of the chemistry class -/
structure ChemistryClass where
  section1 : Section
  section2 : Section
  section3 : Section
  section4 : Section
  overall_average : ℚ

theorem fourth_section_size (c : ChemistryClass) :
  c.section1.students = 65 →
  c.section2.students = 35 →
  c.section3.students = 45 →
  c.section1.mean_marks = 50 →
  c.section2.mean_marks = 60 →
  c.section3.mean_marks = 55 →
  c.section4.mean_marks = 45 →
  c.overall_average = 51.95 →
  c.section4.students = fourth_section_students :=
by
  sorry

#eval fourth_section_students

end fourth_section_size_l1739_173969


namespace constant_function_proof_l1739_173939

def IsFunctionalRelation (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (2 * x)

theorem constant_function_proof (f : ℝ → ℝ) 
  (h1 : Continuous f) 
  (h2 : IsFunctionalRelation f) : 
  ∀ x : ℝ, f x = f 0 := by
  sorry

end constant_function_proof_l1739_173939


namespace class_size_calculation_l1739_173953

theorem class_size_calculation (total_average : ℝ) (excluded_average : ℝ) (remaining_average : ℝ) (excluded_count : ℕ) :
  total_average = 72 →
  excluded_average = 40 →
  remaining_average = 92 →
  excluded_count = 5 →
  ∃ (total_count : ℕ),
    (total_count : ℝ) * total_average = 
      (total_count - excluded_count : ℝ) * remaining_average + (excluded_count : ℝ) * excluded_average ∧
    total_count = 13 :=
by sorry

end class_size_calculation_l1739_173953


namespace ellipse_major_axis_length_l1739_173941

-- Define the radius of the cylinder
def cylinder_radius : ℝ := 2

-- Define the ratio between major and minor axes
def axis_ratio : ℝ := 1.25

-- Theorem statement
theorem ellipse_major_axis_length :
  let minor_axis : ℝ := 2 * cylinder_radius
  let major_axis : ℝ := minor_axis * axis_ratio
  major_axis = 5 := by sorry

end ellipse_major_axis_length_l1739_173941


namespace function_passes_through_point_l1739_173933

/-- Given a > 0 and a ≠ 1, the function f(x) = a^(x-1) + 3 passes through the point (1, 4) -/
theorem function_passes_through_point (a : ℝ) (ha : a > 0) (hna : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ a^(x - 1) + 3
  f 1 = 4 := by
  sorry

end function_passes_through_point_l1739_173933


namespace condition_p_necessary_not_sufficient_for_q_l1739_173973

theorem condition_p_necessary_not_sufficient_for_q :
  (∀ x : ℝ, |x + 1| ≤ 1 → (x - 1) * (x + 2) ≤ 0) ∧
  (∃ x : ℝ, (x - 1) * (x + 2) ≤ 0 ∧ |x + 1| > 1) := by
  sorry

end condition_p_necessary_not_sufficient_for_q_l1739_173973


namespace pet_store_combinations_l1739_173924

def num_puppies : ℕ := 12
def num_kittens : ℕ := 10
def num_hamsters : ℕ := 5
def num_birds : ℕ := 3
def num_people : ℕ := 4

def ways_to_choose_pets : ℕ := num_puppies * num_kittens * num_hamsters * num_birds

def permutations_of_choices : ℕ := Nat.factorial num_people

theorem pet_store_combinations : 
  ways_to_choose_pets * permutations_of_choices = 43200 := by
  sorry

end pet_store_combinations_l1739_173924


namespace gcd_840_1764_l1739_173968

theorem gcd_840_1764 : Nat.gcd 840 1764 = 84 := by
  sorry

end gcd_840_1764_l1739_173968


namespace function_F_property_l1739_173984

-- Define the function F
noncomputable def F : ℝ → ℝ := sorry

-- State the theorem
theorem function_F_property (x : ℝ) : 
  (F ((1 - x) / (1 + x)) = x) → 
  (F (-2 - x) = -2 - F x) := by sorry

end function_F_property_l1739_173984


namespace abs_x_minus_log_y_equals_x_plus_2log_y_implies_x_0_y_1_l1739_173901

theorem abs_x_minus_log_y_equals_x_plus_2log_y_implies_x_0_y_1
  (x : ℝ) (y : ℝ) (h : y > 0) :
  |x - Real.log y| = x + 2 * Real.log y → x = 0 ∧ y = 1 := by
  sorry

end abs_x_minus_log_y_equals_x_plus_2log_y_implies_x_0_y_1_l1739_173901


namespace train_distance_time_relation_l1739_173917

/-- The distance-time relationship for a train journey -/
theorem train_distance_time_relation 
  (initial_distance : ℝ) 
  (speed : ℝ) 
  (t : ℝ) 
  (h1 : initial_distance = 3) 
  (h2 : speed = 120) 
  (h3 : t ≥ 0) : 
  ∃ s : ℝ, s = initial_distance + speed * t :=
sorry

end train_distance_time_relation_l1739_173917


namespace rectangle_with_hole_area_formula_l1739_173906

/-- The area of a rectangle with a rectangular hole -/
def rectangle_with_hole_area (x : ℝ) : ℝ :=
  let large_length : ℝ := 2 * x + 8
  let large_width : ℝ := x + 6
  let hole_length : ℝ := 3 * x - 4
  let hole_width : ℝ := x - 3
  (large_length * large_width) - (hole_length * hole_width)

/-- Theorem: The area of the rectangle with a hole is equal to -x^2 + 33x + 36 -/
theorem rectangle_with_hole_area_formula (x : ℝ) :
  rectangle_with_hole_area x = -x^2 + 33*x + 36 := by
  sorry

end rectangle_with_hole_area_formula_l1739_173906


namespace max_sin_sum_l1739_173992

theorem max_sin_sum (α β θ : Real) : 
  α + β = 2 * Real.pi / 3 →
  α > 0 →
  β > 0 →
  (∀ x y, x + y = 2 * Real.pi / 3 → x > 0 → y > 0 → 
    Real.sin α + 2 * Real.sin β ≥ Real.sin x + 2 * Real.sin y) →
  α = θ →
  Real.cos θ = Real.sqrt 21 / 7 := by
sorry

end max_sin_sum_l1739_173992


namespace m_range_l1739_173911

def P (m : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁ < 0 ∧ x₂ < 0 ∧ x₁^2 + m*x₁ + 1 = 0 ∧ x₂^2 + m*x₂ + 1 = 0

def Q (m : ℝ) : Prop :=
  ∀ x : ℝ, 4*x^2 + 4*(m-2)*x + 1 ≠ 0

theorem m_range (m : ℝ) :
  (P m ∨ Q m) ∧ ¬(P m ∧ Q m) → m ∈ Set.Ioo 1 2 ∪ Set.Ici 3 :=
by sorry

end m_range_l1739_173911


namespace probability_of_pair_after_removal_l1739_173981

/-- Represents a deck of cards -/
structure Deck :=
  (total : ℕ)
  (numbers : List ℕ)
  (counts : List ℕ)

/-- The probability of selecting a pair from the remaining deck -/
def probability_of_pair (d : Deck) : ℚ :=
  83 / 1035

theorem probability_of_pair_after_removal (d : Deck) : 
  d.total = 50 ∧ 
  d.numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] ∧ 
  d.counts = [5, 5, 5, 5, 5, 5, 5, 5, 5, 5] →
  let remaining_deck := {
    total := d.total - 4,
    numbers := d.numbers,
    counts := d.counts.map (fun c => if c = 5 then 3 else 5)
  }
  probability_of_pair remaining_deck = 83 / 1035 := by
sorry

#eval 83 + 1035  -- Should output 1118

end probability_of_pair_after_removal_l1739_173981


namespace triangle_area_with_given_conditions_l1739_173942

noncomputable def triangle_area (r : ℝ) (R : ℝ) (A B C : ℝ) : ℝ :=
  4 * r * R * Real.sin A

theorem triangle_area_with_given_conditions (r R A B C : ℝ) 
  (h_inradius : r = 4)
  (h_circumradius : R = 9)
  (h_angle_condition : 2 * Real.cos A = Real.cos B + Real.cos C) :
  triangle_area r R A B C = 8 * Real.sqrt 181 := by
  sorry

#check triangle_area_with_given_conditions

end triangle_area_with_given_conditions_l1739_173942


namespace first_term_of_a_10_l1739_173988

def first_term (n : ℕ) : ℕ :=
  1 + 2 * (List.range n).sum

theorem first_term_of_a_10 : first_term 10 = 91 := by
  sorry

end first_term_of_a_10_l1739_173988


namespace quadratic_always_positive_implies_m_greater_than_one_l1739_173922

/-- Theorem: If for all real x, x^2 - 2x + m > 0 is true, then m > 1 -/
theorem quadratic_always_positive_implies_m_greater_than_one (m : ℝ) :
  (∀ x : ℝ, x^2 - 2*x + m > 0) → m > 1 := by
  sorry

end quadratic_always_positive_implies_m_greater_than_one_l1739_173922


namespace sin_negative_ten_pi_thirds_equals_sqrt_three_halves_l1739_173974

theorem sin_negative_ten_pi_thirds_equals_sqrt_three_halves :
  Real.sin (-10 * Real.pi / 3) = Real.sqrt 3 / 2 := by
  sorry

end sin_negative_ten_pi_thirds_equals_sqrt_three_halves_l1739_173974


namespace problem_solution_l1739_173916

theorem problem_solution (a b : ℤ) (h1 : a > b) (h2 : b > 0) (h3 : a + b + a * b = 119) : a = 59 := by
  sorry

end problem_solution_l1739_173916


namespace victors_weekly_earnings_l1739_173936

/-- Calculates the total earnings for a week given an hourly wage and hours worked each day -/
def weeklyEarnings (hourlyWage : ℕ) (hoursWorked : List ℕ) : ℕ :=
  hourlyWage * (hoursWorked.sum)

/-- Theorem: Victor's weekly earnings -/
theorem victors_weekly_earnings :
  let hourlyWage : ℕ := 12
  let hoursWorked : List ℕ := [5, 6, 7, 4, 8]
  weeklyEarnings hourlyWage hoursWorked = 360 := by
  sorry

end victors_weekly_earnings_l1739_173936


namespace remainder_problem_l1739_173980

theorem remainder_problem (x : Int) : 
  x % 14 = 11 → x % 84 = 81 := by
sorry

end remainder_problem_l1739_173980


namespace sum_of_cubes_difference_l1739_173975

theorem sum_of_cubes_difference (d e f : ℕ+) :
  (d + e + f : ℕ)^3 - d^3 - e^3 - f^3 = 300 → d + e + f = 7 := by
  sorry

end sum_of_cubes_difference_l1739_173975


namespace scientific_notation_of_nine_billion_l1739_173960

theorem scientific_notation_of_nine_billion :
  9000000000 = 9 * (10 : ℝ)^9 :=
by sorry

end scientific_notation_of_nine_billion_l1739_173960


namespace small_ring_rotation_l1739_173951

theorem small_ring_rotation (r₁ r₂ : ℝ) (h₁ : r₁ = 1) (h₂ : r₂ = 4) :
  (2 * r₂ * Real.pi - 2 * r₁ * Real.pi) / (2 * r₁ * Real.pi) = 3 := by
  sorry

end small_ring_rotation_l1739_173951


namespace min_fraction_sum_l1739_173918

def digits : Finset ℕ := {1, 2, 3, 4, 5, 6}

theorem min_fraction_sum (W X Y Z : ℕ) 
  (h_distinct : W ≠ X ∧ W ≠ Y ∧ W ≠ Z ∧ X ≠ Y ∧ X ≠ Z ∧ Y ≠ Z)
  (h_in_set : W ∈ digits ∧ X ∈ digits ∧ Y ∈ digits ∧ Z ∈ digits) :
  (W : ℚ) / X + (Y : ℚ) / Z ≥ 17 / 30 :=
sorry

end min_fraction_sum_l1739_173918


namespace banana_orange_equivalence_l1739_173929

/-- The cost relationships between fruits at Minnie's Orchard -/
structure FruitCosts where
  banana_pear : ℚ  -- ratio of bananas to pears
  pear_apple : ℚ   -- ratio of pears to apples
  apple_orange : ℚ -- ratio of apples to oranges

/-- The number of oranges equivalent in cost to a given number of bananas -/
def bananas_to_oranges (costs : FruitCosts) (num_bananas : ℚ) : ℚ :=
  num_bananas * costs.banana_pear * costs.pear_apple * costs.apple_orange

/-- Theorem stating that 80 bananas are equivalent in cost to 18 oranges -/
theorem banana_orange_equivalence (costs : FruitCosts) 
  (h1 : costs.banana_pear = 4/5)
  (h2 : costs.pear_apple = 3/8)
  (h3 : costs.apple_orange = 9/12) :
  bananas_to_oranges costs 80 = 18 := by
  sorry

#eval bananas_to_oranges ⟨4/5, 3/8, 9/12⟩ 80

end banana_orange_equivalence_l1739_173929


namespace angle_CDB_is_15_l1739_173919

/-- A triangle that shares a side with a rectangle -/
structure TriangleWithRectangle where
  /-- The length of the shared side -/
  side : ℝ
  /-- The triangle is equilateral -/
  equilateral : True
  /-- The adjacent side of the rectangle is perpendicular to the shared side -/
  perpendicular : True
  /-- The adjacent side of the rectangle is twice the length of the shared side -/
  adjacent_side : ℝ := 2 * side

/-- The measure of angle CDB in degrees -/
def angle_CDB (t : TriangleWithRectangle) : ℝ := 15

/-- Theorem: The measure of angle CDB is 15 degrees -/
theorem angle_CDB_is_15 (t : TriangleWithRectangle) : angle_CDB t = 15 := by sorry

end angle_CDB_is_15_l1739_173919


namespace mobileRadiationNotSuitable_l1739_173985

/-- Represents a statistical activity that can be potentially collected through a questionnaire. -/
inductive StatisticalActivity
  | BlueCars
  | TVsInHomes
  | WakeUpTime
  | MobileRadiation

/-- Predicate to determine if a statistical activity is suitable for questionnaire data collection. -/
def suitableForQuestionnaire (activity : StatisticalActivity) : Prop :=
  match activity with
  | StatisticalActivity.BlueCars => True
  | StatisticalActivity.TVsInHomes => True
  | StatisticalActivity.WakeUpTime => True
  | StatisticalActivity.MobileRadiation => False

/-- Theorem stating that mobile radiation is the only activity not suitable for questionnaire data collection. -/
theorem mobileRadiationNotSuitable :
    ∀ (activity : StatisticalActivity),
      ¬(suitableForQuestionnaire activity) ↔ activity = StatisticalActivity.MobileRadiation := by
  sorry

end mobileRadiationNotSuitable_l1739_173985


namespace min_value_expression_l1739_173997

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (((x^2 + y^2) * (2 * x^2 + 4 * y^2)).sqrt) / (x * y) ≥ 4 + 6 * Real.sqrt 2 :=
sorry

end min_value_expression_l1739_173997


namespace w_squared_equals_one_fourth_l1739_173952

theorem w_squared_equals_one_fourth (w : ℝ) (h : 13 = 13 * w / (1 - w)) : w^2 = 1/4 := by
  sorry

end w_squared_equals_one_fourth_l1739_173952


namespace interval_intersection_l1739_173999

theorem interval_intersection (x : ℝ) : 
  (2 < 4 * x ∧ 4 * x < 3) ∧ (2 < 5 * x ∧ 5 * x < 3) ↔ 1/2 < x ∧ x < 3/5 := by
  sorry

end interval_intersection_l1739_173999


namespace find_m_l1739_173998

-- Define the universal set U
def U : Set Nat := {1, 2, 3}

-- Define set A
def A (m : Nat) : Set Nat := {1, m}

-- Define the complement of A in U
def complementA : Set Nat := {2}

-- Theorem to prove
theorem find_m : ∃ m : Nat, m ∈ U ∧ A m ∪ complementA = U := by
  sorry

end find_m_l1739_173998


namespace tangent_line_sum_l1739_173944

/-- Given a function f: ℝ → ℝ with a tangent line at x = 1 defined by 2x - y + 1 = 0,
    prove that f(1) + f'(1) = 5 -/
theorem tangent_line_sum (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
    (h_tangent : ∀ x y, y = f x → (x = 1 → 2*x - y + 1 = 0)) : 
    f 1 + (deriv f) 1 = 5 := by
  sorry

end tangent_line_sum_l1739_173944


namespace odd_prime_fifth_power_difference_l1739_173931

theorem odd_prime_fifth_power_difference (p : ℕ) (hp : Nat.Prime p) (hp_odd : Odd p) 
  (hx : ∃ (x y : ℤ), (x : ℝ)^5 - (y : ℝ)^5 = p) :
  ∃ (v : ℤ), Odd v ∧ Real.sqrt ((4 * p + 1 : ℝ) / 5) = ((v^2 : ℝ) + 1) / 2 := by
  sorry

end odd_prime_fifth_power_difference_l1739_173931


namespace minimum_dimes_for_scarf_l1739_173993

/-- The cost of the scarf in cents -/
def scarf_cost : ℕ := 4285

/-- The amount of money Chloe has without dimes, in cents -/
def initial_money : ℕ := 4000 + 100 + 50

/-- The value of a dime in cents -/
def dime_value : ℕ := 10

/-- The minimum number of dimes needed to buy the scarf -/
def min_dimes_needed : ℕ := 14

theorem minimum_dimes_for_scarf :
  min_dimes_needed = (scarf_cost - initial_money + dime_value - 1) / dime_value :=
by sorry

end minimum_dimes_for_scarf_l1739_173993


namespace square_and_reciprocal_square_l1739_173949

theorem square_and_reciprocal_square (n : ℝ) (h : n + 1/n = 10) : n^2 + 1/n^2 + 6 = 104 := by
  sorry

end square_and_reciprocal_square_l1739_173949


namespace split_bill_proof_l1739_173940

def num_friends : ℕ := 5
def num_hamburgers : ℕ := 5
def price_hamburger : ℚ := 3
def num_fries : ℕ := 4
def price_fries : ℚ := 1.20
def num_soda : ℕ := 5
def price_soda : ℚ := 0.50
def price_spaghetti : ℚ := 2.70

theorem split_bill_proof :
  let total_bill := num_hamburgers * price_hamburger +
                    num_fries * price_fries +
                    num_soda * price_soda +
                    price_spaghetti
  (total_bill / num_friends : ℚ) = 5 := by
  sorry

end split_bill_proof_l1739_173940


namespace proposition_implication_l1739_173904

theorem proposition_implication (p q : Prop) 
  (h1 : ¬p) 
  (h2 : p ∨ q) : 
  q := by sorry

end proposition_implication_l1739_173904


namespace fran_required_speed_l1739_173921

/-- Represents a bike ride with total time, break time, and average speed -/
structure BikeRide where
  totalTime : ℝ
  breakTime : ℝ
  avgSpeed : ℝ

/-- Calculates the distance traveled given a BikeRide -/
def distanceTraveled (ride : BikeRide) : ℝ :=
  ride.avgSpeed * (ride.totalTime - ride.breakTime)

theorem fran_required_speed (joann fran : BikeRide)
    (h1 : joann.totalTime = 4)
    (h2 : joann.breakTime = 1)
    (h3 : joann.avgSpeed = 10)
    (h4 : fran.totalTime = 3)
    (h5 : fran.breakTime = 0.5)
    (h6 : distanceTraveled joann = distanceTraveled fran) :
    fran.avgSpeed = 12 := by
  sorry

#check fran_required_speed

end fran_required_speed_l1739_173921


namespace scooter_only_owners_l1739_173967

theorem scooter_only_owners (total : ℕ) (scooter : ℕ) (bike : ℕ) 
  (h1 : total = 450) 
  (h2 : scooter = 380) 
  (h3 : bike = 120) : 
  scooter - (scooter + bike - total) = 330 := by
  sorry

end scooter_only_owners_l1739_173967


namespace repeating_decimal_to_fraction_l1739_173905

/-- Proves that the repeating decimal 7.832̅ is equal to the fraction 70/9 -/
theorem repeating_decimal_to_fraction :
  ∃ (x : ℚ), x = 7 + 832 / 999 ∧ x = 70 / 9 := by sorry

end repeating_decimal_to_fraction_l1739_173905


namespace grid_rectangles_l1739_173961

theorem grid_rectangles (h : ℕ) (v : ℕ) (h_eq : h = 5) (v_eq : v = 6) :
  (h.choose 2) * (v.choose 2) = 150 := by
  sorry

end grid_rectangles_l1739_173961


namespace rectangular_field_with_pond_l1739_173927

theorem rectangular_field_with_pond (w l : ℝ) : 
  l = 2 * w →                 -- length is double the width
  36 = (1/8) * (l * w) →      -- pond area (6^2) is 1/8 of field area
  l = 24 := by               -- length of the field is 24 meters
sorry

end rectangular_field_with_pond_l1739_173927


namespace james_weight_plates_purchase_l1739_173996

/-- Represents the purchase of a weight vest and weight plates -/
structure WeightPurchase where
  vest_cost : ℝ
  plate_cost_per_pound : ℝ
  discounted_200lb_vest_cost : ℝ
  savings : ℝ

/-- Calculates the number of pounds of weight plates purchased -/
def weight_plates_purchased (purchase : WeightPurchase) : ℕ :=
  sorry

/-- Theorem stating that James purchased 291 pounds of weight plates -/
theorem james_weight_plates_purchase :
  let purchase : WeightPurchase := {
    vest_cost := 250,
    plate_cost_per_pound := 1.2,
    discounted_200lb_vest_cost := 700 - 100,
    savings := 110
  }
  weight_plates_purchased purchase = 291 := by
  sorry

end james_weight_plates_purchase_l1739_173996


namespace rectangle_side_difference_l1739_173930

theorem rectangle_side_difference (p d : ℝ) (h_positive : p > 0 ∧ d > 0) :
  ∃ (x y : ℝ), 
    x > 0 ∧ y > 0 ∧
    x = 2 * y ∧
    2 * (x + y) = p ∧
    x^2 + y^2 = d^2 ∧
    x - y = p / 6 := by
  sorry

end rectangle_side_difference_l1739_173930


namespace some_number_solution_l1739_173923

theorem some_number_solution :
  ∃ x : ℝ, x * 13.26 + x * 9.43 + x * 77.31 = 470 ∧ x = 4.7 := by
  sorry

end some_number_solution_l1739_173923


namespace sum_of_digits_M_l1739_173983

/-- Sum of digits function -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Predicate for numbers using only specified digits -/
def uses_specified_digits (n : ℕ) : Prop := sorry

theorem sum_of_digits_M (M : ℕ) 
  (h_even : Even M)
  (h_digits : uses_specified_digits M)
  (h_double : sum_of_digits (2 * M) = 39)
  (h_half : sum_of_digits (M / 2) = 30) :
  sum_of_digits M = 33 := by sorry

end sum_of_digits_M_l1739_173983


namespace locus_of_intersection_point_l1739_173915

/-- The locus of the intersection point of two rotating lines in a triangle --/
theorem locus_of_intersection_point (d e : ℝ) (h1 : d ≠ 0) (h2 : e ≠ 0) :
  ∃ (f : ℝ → ℝ × ℝ),
    (∀ t, ∃ (m : ℝ),
      (f t).1 = -2 * e / m ∧
      (f t).2 = -m * d) ∧
    (∀ x y, (x, y) ∈ Set.range f ↔ x * y = d * e) :=
sorry

end locus_of_intersection_point_l1739_173915


namespace parabola_directrix_l1739_173928

/-- Given a parabola y^2 = 2px where p > 0 that passes through the point (1, 1/2),
    its directrix has the equation x = -1/16 -/
theorem parabola_directrix (p : ℝ) (h1 : p > 0) :
  (∀ x y : ℝ, y^2 = 2*p*x) →
  ((1 : ℝ)^2 = 2*p*(1/2 : ℝ)^2) →
  (∃ k : ℝ, ∀ x : ℝ, x = k ↔ x = -1/16) := by
  sorry

end parabola_directrix_l1739_173928


namespace factors_of_product_l1739_173954

/-- A function that returns the number of factors of a natural number -/
def num_factors (n : ℕ) : ℕ := sorry

/-- A function that returns the number of factors of n^k for a natural number n and exponent k -/
def num_factors_power (n k : ℕ) : ℕ := sorry

theorem factors_of_product (a b c : ℕ) : 
  a ≠ b → b ≠ c → a ≠ c →
  num_factors a = 3 →
  num_factors b = 3 →
  num_factors c = 4 →
  num_factors_power a 3 * num_factors_power b 4 * num_factors_power c 5 = 1008 := by
  sorry

end factors_of_product_l1739_173954


namespace cube_surface_area_l1739_173986

/-- The surface area of a cube with side length 20 cm is 2400 square centimeters. -/
theorem cube_surface_area : 
  let side_length : ℝ := 20
  6 * side_length ^ 2 = 2400 := by sorry

end cube_surface_area_l1739_173986


namespace value_of_expression_l1739_173925

theorem value_of_expression (x y : ℤ) (hx : x = 12) (hy : y = 18) :
  3 * (x - y) * (x + y) = -540 := by
  sorry

end value_of_expression_l1739_173925
