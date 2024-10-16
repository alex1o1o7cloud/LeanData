import Mathlib

namespace NUMINAMATH_CALUDE_min_beacons_required_l3503_350316

/-- Represents a room in the maze --/
structure Room where
  x : Nat
  y : Nat

/-- Represents the maze structure --/
def Maze := List Room

/-- Calculates the distance between two rooms in the maze --/
def distance (maze : Maze) (r1 r2 : Room) : Nat :=
  sorry

/-- Checks if a set of beacons can uniquely identify all rooms --/
def can_identify_all_rooms (maze : Maze) (beacons : List Room) : Prop :=
  sorry

/-- The main theorem stating that at least 3 beacons are required --/
theorem min_beacons_required (maze : Maze) :
  ∀ (beacons : List Room),
    can_identify_all_rooms maze beacons →
    beacons.length ≥ 3 :=
  sorry

end NUMINAMATH_CALUDE_min_beacons_required_l3503_350316


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3503_350397

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The theorem statement -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
  (h_arithmetic : arithmetic_sequence a)
  (h_sum : a 3 + a 6 + a 10 + a 13 = 32) :
  a 8 = 8 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3503_350397


namespace NUMINAMATH_CALUDE_profit_difference_l3503_350368

def business_problem (capital_A capital_B capital_C capital_D capital_E profit_B : ℕ) : Prop :=
  let total_capital := capital_A + capital_B + capital_C + capital_D + capital_E
  let total_profit := profit_B * total_capital / capital_B
  let profit_C := total_profit * capital_C / total_capital
  let profit_E := total_profit * capital_E / total_capital
  profit_E - profit_C = 900

theorem profit_difference :
  business_problem 8000 10000 12000 15000 18000 1500 := by sorry

end NUMINAMATH_CALUDE_profit_difference_l3503_350368


namespace NUMINAMATH_CALUDE_AB_range_l3503_350379

/-- Represents an acute triangle ABC with specific properties -/
structure AcuteTriangle where
  A : Real
  B : Real
  C : Real
  AB : Real
  BC : Real
  AC : Real
  acute : A < 90 ∧ B < 90 ∧ C < 90
  angle_sum : A + B + C = 180
  angle_A : A = 60
  side_BC : BC = 6
  side_AB : AB > 0

/-- Theorem stating the range of AB in the specific acute triangle -/
theorem AB_range (t : AcuteTriangle) : 2 * Real.sqrt 3 < t.AB ∧ t.AB < 4 * Real.sqrt 3 := by
  sorry

#check AB_range

end NUMINAMATH_CALUDE_AB_range_l3503_350379


namespace NUMINAMATH_CALUDE_unique_prime_divisor_l3503_350369

theorem unique_prime_divisor : 
  ∃! p : ℕ, p ≥ 5 ∧ Prime p ∧ (p ∣ (p + 3)^(p-3) + (p + 5)^(p-5)) ∧ p = 2813 := by
  sorry

end NUMINAMATH_CALUDE_unique_prime_divisor_l3503_350369


namespace NUMINAMATH_CALUDE_twelve_point_polygons_l3503_350366

/-- The number of distinct convex polygons with three or more sides
    that can be formed from 12 points on a circle's circumference. -/
def num_polygons (n : ℕ) : ℕ :=
  2^n - (Nat.choose n 0 + Nat.choose n 1 + Nat.choose n 2)

/-- Theorem stating that the number of distinct convex polygons
    with three or more sides formed from 12 points on a circle
    is equal to 4017. -/
theorem twelve_point_polygons :
  num_polygons 12 = 4017 := by
  sorry

end NUMINAMATH_CALUDE_twelve_point_polygons_l3503_350366


namespace NUMINAMATH_CALUDE_f_is_even_l3503_350308

-- Define the function f(x) = x^2
def f (x : ℝ) : ℝ := x^2

-- Theorem: f is an even function
theorem f_is_even : ∀ x : ℝ, f (-x) = f x := by
  sorry

end NUMINAMATH_CALUDE_f_is_even_l3503_350308


namespace NUMINAMATH_CALUDE_rectangular_solid_surface_area_l3503_350348

/-- A rectangular solid with prime edge lengths and volume 399 has surface area 422. -/
theorem rectangular_solid_surface_area : ∀ a b c : ℕ,
  Prime a → Prime b → Prime c →
  a * b * c = 399 →
  2 * (a * b + b * c + c * a) = 422 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_solid_surface_area_l3503_350348


namespace NUMINAMATH_CALUDE_cube_net_theorem_l3503_350337

theorem cube_net_theorem (a b c : ℝ) 
  (eq1 : 3 * a + 2 = 17)
  (eq2 : 7 * b - 4 = 10)
  (eq3 : a + 3 * b - 2 * c = 11) :
  a - b * c = 5 := by
sorry

end NUMINAMATH_CALUDE_cube_net_theorem_l3503_350337


namespace NUMINAMATH_CALUDE_point_in_intersection_l3503_350306

def U : Set (ℝ × ℝ) := Set.univ

def A (m : ℝ) : Set (ℝ × ℝ) := {p | 2 * p.1 - p.2 + m > 0}

def B (n : ℝ) : Set (ℝ × ℝ) := {p | p.1 + p.2 - n ≤ 0}

theorem point_in_intersection (m n : ℝ) :
  (2, 3) ∈ A m ∩ (U \ B n) ↔ m > -1 ∧ n < 5 := by
  sorry

end NUMINAMATH_CALUDE_point_in_intersection_l3503_350306


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_main_theorem_l3503_350376

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_ratio (a : ℕ → ℝ) (h : GeometricSequence a) :
  ∃ q : ℝ, q > 0 ∧ (a 2 = a 1 * q ∧ a 3 = a 2 * q ∧ a 4 = a 3 * q ∧ a 5 = a 4 * q) :=
sorry

/-- The second, half of the third, and twice the first term form an arithmetic sequence -/
def ArithmeticSubsequence (a : ℕ → ℝ) : Prop :=
  a 2 - (1/2 * a 3) = (1/2 * a 3) - (2 * a 1)

theorem main_theorem (a : ℕ → ℝ) (h1 : GeometricSequence a) (h2 : ArithmeticSubsequence a) :
  (a 3 + a 4) / (a 4 + a 5) = 1/2 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_main_theorem_l3503_350376


namespace NUMINAMATH_CALUDE_g_f_neg_four_equals_nine_l3503_350360

/-- Given a function f and a function g, prove that g(f(-4)) = 9 
    under certain conditions. -/
theorem g_f_neg_four_equals_nine 
  (f : ℝ → ℝ) 
  (g : ℝ → ℝ) 
  (h1 : ∀ x, f x = 3 * x^2 - 7) 
  (h2 : g (f 4) = 9) : 
  g (f (-4)) = 9 := by
sorry

end NUMINAMATH_CALUDE_g_f_neg_four_equals_nine_l3503_350360


namespace NUMINAMATH_CALUDE_proposition_equivalences_l3503_350326

-- Define opposite numbers
def opposite (x y : ℝ) : Prop := x = -y

-- Define having real roots for a quadratic equation
def has_real_roots (a b c : ℝ) : Prop := b^2 - 4*a*c ≥ 0

theorem proposition_equivalences :
  -- Converse of "If x+y=0, then x and y are opposite numbers"
  (∀ x y : ℝ, opposite x y → x + y = 0) ∧
  -- Contrapositive of "If q ≤ 1, then x^2+2x+q=0 has real roots"
  (∀ q : ℝ, ¬(has_real_roots 1 2 q) → q > 1) ∧
  -- Existence of α and β satisfying the trigonometric equation
  (∃ α β : ℝ, Real.sin (α + β) = Real.sin α + Real.sin β) :=
by sorry

end NUMINAMATH_CALUDE_proposition_equivalences_l3503_350326


namespace NUMINAMATH_CALUDE_carmen_initial_cats_l3503_350346

/-- Represents the number of cats Carmen initially had -/
def initial_cats : ℕ := sorry

/-- Represents the number of dogs Carmen has -/
def dogs : ℕ := 18

/-- Represents the number of cats Carmen gave up for adoption -/
def cats_given_up : ℕ := 3

/-- Represents the difference between cats and dogs after giving up cats -/
def cat_dog_difference : ℕ := 7

theorem carmen_initial_cats :
  initial_cats = 28 ∧
  initial_cats - cats_given_up = dogs + cat_dog_difference :=
sorry

end NUMINAMATH_CALUDE_carmen_initial_cats_l3503_350346


namespace NUMINAMATH_CALUDE_snack_eaters_count_l3503_350343

def snack_eaters_final (initial_attendees : ℕ) 
  (first_hour_eat_percent : ℚ) 
  (first_hour_not_eat_percent : ℚ)
  (second_hour_undecided_join_percent : ℚ)
  (second_hour_not_eat_join_percent : ℚ)
  (second_hour_newcomers : ℕ)
  (second_hour_newcomers_eat : ℕ)
  (second_hour_leave_percent : ℚ)
  (third_hour_increase_percent : ℚ)
  (third_hour_leave_percent : ℚ)
  (fourth_hour_latecomers : ℕ)
  (fourth_hour_latecomers_eat_percent : ℚ)
  (fourth_hour_workshop_leave : ℕ) : ℕ :=
  sorry

theorem snack_eaters_count : 
  snack_eaters_final 7500 (55/100) (35/100) (20/100) (15/100) 75 50 (40/100) (10/100) (1/2) 150 (60/100) 300 = 1347 := by
  sorry

end NUMINAMATH_CALUDE_snack_eaters_count_l3503_350343


namespace NUMINAMATH_CALUDE_gcd_g_x_l3503_350321

def g (x : ℤ) : ℤ := (4*x+5)*(5*x+2)*(11*x+8)*(3*x+7)

theorem gcd_g_x (x : ℤ) (h : 2520 ∣ x) : Int.gcd (g x) x = 280 := by
  sorry

end NUMINAMATH_CALUDE_gcd_g_x_l3503_350321


namespace NUMINAMATH_CALUDE_mutually_exclusive_not_complementary_l3503_350322

structure Ball :=
  (color : String)

def Bag : Finset Ball := sorry

def draw_two_balls : Finset (Ball × Ball) := sorry

def both_white (pair : Ball × Ball) : Prop :=
  pair.1.color = "white" ∧ pair.2.color = "white"

def both_not_white (pair : Ball × Ball) : Prop :=
  pair.1.color ≠ "white" ∧ pair.2.color ≠ "white"

def exactly_one_white (pair : Ball × Ball) : Prop :=
  (pair.1.color = "white" ∧ pair.2.color ≠ "white") ∨
  (pair.1.color ≠ "white" ∧ pair.2.color = "white")

theorem mutually_exclusive_not_complementary :
  (∀ pair ∈ draw_two_balls, ¬(both_white pair ∧ both_not_white pair)) ∧
  (∀ pair ∈ draw_two_balls, ¬(both_white pair ∧ exactly_one_white pair)) ∧
  (∃ pair ∈ draw_two_balls, ¬both_white pair ∧ ¬both_not_white pair ∧ ¬exactly_one_white pair) :=
sorry

end NUMINAMATH_CALUDE_mutually_exclusive_not_complementary_l3503_350322


namespace NUMINAMATH_CALUDE_min_value_A_over_C_l3503_350334

theorem min_value_A_over_C (x A C : ℝ) (hx : x > 0) (hA : A > 0) (hC : C > 0)
  (h1 : x^2 + 1/x^2 = A) (h2 : x - 1/x = C) (h3 : C = Real.sqrt 3) :
  A / C ≥ 5 * Real.sqrt 3 / 3 := by
sorry

end NUMINAMATH_CALUDE_min_value_A_over_C_l3503_350334


namespace NUMINAMATH_CALUDE_f_composition_value_l3503_350367

def f (x : ℝ) : ℝ := x^2 - 2*x

theorem f_composition_value : f (f (f (-2))) = 4 := by sorry

end NUMINAMATH_CALUDE_f_composition_value_l3503_350367


namespace NUMINAMATH_CALUDE_number_equation_proof_l3503_350330

theorem number_equation_proof : ∃ x : ℝ, 5020 - (x / 100.4) = 5015 ∧ x = 502 := by
  sorry

end NUMINAMATH_CALUDE_number_equation_proof_l3503_350330


namespace NUMINAMATH_CALUDE_gcd_1908_4187_l3503_350311

theorem gcd_1908_4187 : Nat.gcd 1908 4187 = 53 := by
  sorry

end NUMINAMATH_CALUDE_gcd_1908_4187_l3503_350311


namespace NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l3503_350372

theorem quadratic_two_distinct_roots (k : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ k * x₁^2 - 2 * x₁ - 1 = 0 ∧ k * x₂^2 - 2 * x₂ - 1 = 0) ↔
  (k > -1 ∧ k ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l3503_350372


namespace NUMINAMATH_CALUDE_f_is_even_f_increasing_on_nonneg_l3503_350312

-- Define the function f
def f (x : ℝ) : ℝ := x^2 + 1

-- Theorem for the parity of the function (even function)
theorem f_is_even : ∀ x : ℝ, f (-x) = f x := by sorry

-- Theorem for monotonic increase on [0, +∞)
theorem f_increasing_on_nonneg : ∀ x y : ℝ, 0 ≤ x → x < y → f x < f y := by sorry

end NUMINAMATH_CALUDE_f_is_even_f_increasing_on_nonneg_l3503_350312


namespace NUMINAMATH_CALUDE_min_questions_to_identify_apartment_l3503_350378

theorem min_questions_to_identify_apartment (n : ℕ) (h : n = 80) : 
  (∀ m : ℕ, 2^m < n → m < 7) ∧ (2^7 ≥ n) := by
  sorry

end NUMINAMATH_CALUDE_min_questions_to_identify_apartment_l3503_350378


namespace NUMINAMATH_CALUDE_fixed_points_for_specific_values_two_distinct_fixed_points_condition_l3503_350341

-- Define the function f
def f (a b x : ℝ) : ℝ := a * x^2 + (b + 1) * x + b - 1

-- Define what it means to be a fixed point
def is_fixed_point (f : ℝ → ℝ) (x : ℝ) : Prop := f x = x

theorem fixed_points_for_specific_values :
  ∀ x : ℝ, is_fixed_point (f 1 (-2)) x ↔ (x = 3 ∨ x = -1) := by sorry

theorem two_distinct_fixed_points_condition :
  ∀ a : ℝ, (∀ b : ℝ, ∃ x y : ℝ, x ≠ y ∧ is_fixed_point (f a b) x ∧ is_fixed_point (f a b) y) ↔ (0 < a ∧ a < 1) := by sorry

end NUMINAMATH_CALUDE_fixed_points_for_specific_values_two_distinct_fixed_points_condition_l3503_350341


namespace NUMINAMATH_CALUDE_tv_price_changes_l3503_350395

theorem tv_price_changes (P : ℝ) (P_positive : P > 0) :
  let price_after_changes := P * 1.30 * 1.20 * 0.90 * 1.15
  let single_increase := 1.6146
  price_after_changes = P * single_increase :=
by sorry

end NUMINAMATH_CALUDE_tv_price_changes_l3503_350395


namespace NUMINAMATH_CALUDE_function_inequality_l3503_350394

/-- Given a function f(x) = axe^x where a ≠ 0 and a ≥ 4/e^2, 
    prove that f(x)/(x+1) - (x+1)ln(x) > 0 for x > 0 -/
theorem function_inequality (a : ℝ) (h1 : a ≠ 0) (h2 : a ≥ 4 / Real.exp 2) :
  ∀ x > 0, (a * x * Real.exp x) / (x + 1) - (x + 1) * Real.log x > 0 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l3503_350394


namespace NUMINAMATH_CALUDE_lucas_test_score_l3503_350349

def existing_scores : List ℝ := [85, 90, 78, 88, 96]
def sixth_score : ℝ := 91
def desired_mean : ℝ := 88

theorem lucas_test_score :
  (List.sum existing_scores + sixth_score) / 6 = desired_mean := by
  sorry

end NUMINAMATH_CALUDE_lucas_test_score_l3503_350349


namespace NUMINAMATH_CALUDE_simplify_expression_l3503_350328

theorem simplify_expression (r s : ℝ) : 
  (2 * r^2 + 5 * r - 6 * s + 4) - (r^2 + 9 * r - 4 * s - 2) = r^2 - 4 * r - 2 * s + 6 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3503_350328


namespace NUMINAMATH_CALUDE_no_solution_iff_parallel_equation_no_solution_iff_l3503_350304

/-- Two 2D vectors are parallel if one is a scalar multiple of the other -/
def parallel (v w : ℝ × ℝ) : Prop :=
  ∃ c : ℝ, v.1 = c * w.1 ∧ v.2 = c * w.2

/-- The equation has no solution if and only if the direction vectors are parallel -/
theorem no_solution_iff_parallel (m : ℝ) : Prop :=
  parallel (5, 2) (-2, m)

/-- The main theorem: the equation has no solution if and only if m = -4/5 -/
theorem equation_no_solution_iff (m : ℝ) : 
  no_solution_iff_parallel m ↔ m = -4/5 := by sorry

end NUMINAMATH_CALUDE_no_solution_iff_parallel_equation_no_solution_iff_l3503_350304


namespace NUMINAMATH_CALUDE_total_viewing_time_is_900_hours_l3503_350383

/-- Calculates the total viewing time for two people watching multiple videos at different speeds -/
def totalViewingTime (videoLength : ℕ) (numVideos : ℕ) (lilaSpeed : ℕ) (rogerSpeed : ℕ) : ℕ :=
  (videoLength * numVideos / lilaSpeed) + (videoLength * numVideos / rogerSpeed)

/-- Theorem stating that the total viewing time for Lila and Roger is 900 hours -/
theorem total_viewing_time_is_900_hours :
  totalViewingTime 100 6 2 1 = 900 := by
  sorry

end NUMINAMATH_CALUDE_total_viewing_time_is_900_hours_l3503_350383


namespace NUMINAMATH_CALUDE_fraction_equality_l3503_350319

theorem fraction_equality : (36 + 12) / (6 - 3) = 16 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l3503_350319


namespace NUMINAMATH_CALUDE_production_days_l3503_350373

theorem production_days (n : ℕ) (h1 : (50 * n + 90) / (n + 1) = 54) : n = 9 := by
  sorry

end NUMINAMATH_CALUDE_production_days_l3503_350373


namespace NUMINAMATH_CALUDE_prime_sum_less_than_ten_l3503_350351

theorem prime_sum_less_than_ten (d e f : ℕ) : 
  Prime d → Prime e → Prime f →
  d < 10 → e < 10 → f < 10 →
  d + e = f →
  d < e →
  d = 2 := by
sorry

end NUMINAMATH_CALUDE_prime_sum_less_than_ten_l3503_350351


namespace NUMINAMATH_CALUDE_factory_shift_cost_l3503_350355

/-- The cost to employ all workers for one 8-hour shift -/
def total_cost (total_employees : ℕ) (low_wage_employees : ℕ) (mid_wage_employees : ℕ) 
  (low_wage : ℕ) (mid_wage : ℕ) (high_wage : ℕ) (shift_hours : ℕ) : ℕ :=
  let high_wage_employees := total_employees - low_wage_employees - mid_wage_employees
  low_wage_employees * low_wage * shift_hours + 
  mid_wage_employees * mid_wage * shift_hours + 
  high_wage_employees * high_wage * shift_hours

/-- Theorem stating the total cost for the given scenario -/
theorem factory_shift_cost : 
  total_cost 300 200 40 12 14 17 8 = 31840 := by
  sorry

end NUMINAMATH_CALUDE_factory_shift_cost_l3503_350355


namespace NUMINAMATH_CALUDE_root_zero_implies_k_five_l3503_350374

theorem root_zero_implies_k_five (k : ℝ) : 
  (∃ x : ℝ, x ≠ 0 ∧ 8 * x^2 - (k - 1) * x - k + 5 = 0) ∧ 
  (8 * 0^2 - (k - 1) * 0 - k + 5 = 0) → 
  k = 5 := by sorry

end NUMINAMATH_CALUDE_root_zero_implies_k_five_l3503_350374


namespace NUMINAMATH_CALUDE_share_of_y_l3503_350353

/-- The share of y in a sum divided among x, y, and z, where for each rupee x gets,
    y gets 45 paisa and z gets 50 paisa, and the total amount is Rs. 78. -/
theorem share_of_y (x y z : ℝ) : 
  x + y + z = 78 →  -- Total amount condition
  y = 0.45 * x →    -- Relationship between y and x
  z = 0.5 * x →     -- Relationship between z and x
  y = 18 :=         -- Share of y
by sorry

end NUMINAMATH_CALUDE_share_of_y_l3503_350353


namespace NUMINAMATH_CALUDE_math_marks_proof_l3503_350305

/-- Calculates the marks in Mathematics given marks in other subjects and the average -/
def calculate_math_marks (english physics chemistry biology average : ℕ) : ℕ :=
  5 * average - (english + physics + chemistry + biology)

theorem math_marks_proof (english physics chemistry biology average : ℕ) 
  (h_english : english = 96)
  (h_physics : physics = 82)
  (h_chemistry : chemistry = 97)
  (h_biology : biology = 95)
  (h_average : average = 93) :
  calculate_math_marks english physics chemistry biology average = 95 := by
  sorry

end NUMINAMATH_CALUDE_math_marks_proof_l3503_350305


namespace NUMINAMATH_CALUDE_bigger_part_of_60_l3503_350302

theorem bigger_part_of_60 (x y : ℝ) (h1 : x + y = 60) (h2 : 10 * x + 22 * y = 780) 
  (h3 : x > 0) (h4 : y > 0) : max x y = 45 := by
  sorry

end NUMINAMATH_CALUDE_bigger_part_of_60_l3503_350302


namespace NUMINAMATH_CALUDE_geometric_arithmetic_progression_problem_l3503_350364

theorem geometric_arithmetic_progression_problem (a b c : ℝ) :
  (∃ q : ℝ, q ≠ 0 ∧ b = a * q ∧ 12 = a * q^2) ∧  -- Geometric progression condition
  (∃ d : ℝ, b = a + d ∧ 9 = a + 2 * d) →        -- Arithmetic progression condition
  ((a = -9 ∧ b = -6 ∧ c = 12) ∨ (a = 15 ∧ b = 12 ∧ c = 9)) :=
by sorry

end NUMINAMATH_CALUDE_geometric_arithmetic_progression_problem_l3503_350364


namespace NUMINAMATH_CALUDE_ratio_problem_l3503_350382

theorem ratio_problem (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a > b) (h4 : 2 * (a + b) = 3 * (a - b)) :
  a = 5 * b := by
sorry

end NUMINAMATH_CALUDE_ratio_problem_l3503_350382


namespace NUMINAMATH_CALUDE_banana_bread_recipe_l3503_350340

/-- Given a banana bread recipe and baking requirements, determine the number of loaves the recipe can make. -/
theorem banana_bread_recipe (total_loaves : ℕ) (total_bananas : ℕ) (bananas_per_recipe : ℕ) 
  (h1 : total_loaves = 99)
  (h2 : total_bananas = 33)
  (h3 : bananas_per_recipe = 1)
  (h4 : total_bananas > 0) :
  total_loaves / total_bananas = 3 := by
  sorry

#check banana_bread_recipe

end NUMINAMATH_CALUDE_banana_bread_recipe_l3503_350340


namespace NUMINAMATH_CALUDE_hcf_of_210_and_605_l3503_350338

theorem hcf_of_210_and_605 :
  let a := 210
  let b := 605
  let lcm_ab := 2310
  lcm a b = lcm_ab →
  Nat.gcd a b = 55 := by
sorry

end NUMINAMATH_CALUDE_hcf_of_210_and_605_l3503_350338


namespace NUMINAMATH_CALUDE_parallel_vectors_mn_value_l3503_350314

def vector_a (m n : ℝ) : Fin 3 → ℝ := λ i =>
  match i with
  | 0 => 2
  | 1 => 2*m - 3
  | 2 => n + 2

def vector_b (m n : ℝ) : Fin 3 → ℝ := λ i =>
  match i with
  | 0 => 4
  | 1 => 2*m + 1
  | 2 => 3*n - 2

theorem parallel_vectors_mn_value (m n : ℝ) :
  (∃ (k : ℝ), ∀ (i : Fin 3), vector_a m n i = k * vector_b m n i) →
  m * n = 21 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_mn_value_l3503_350314


namespace NUMINAMATH_CALUDE_parallel_line_slope_l3503_350386

/-- Given a line with equation 3x - 6y = 12, prove that the slope of any parallel line is 1/2 -/
theorem parallel_line_slope (x y : ℝ) : 
  (3 * x - 6 * y = 12) → 
  (∃ (m b : ℝ), ∀ (x' y' : ℝ), y' = m * x' + b ∧ (3 * x' - 6 * y' = 12)) → 
  m = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_line_slope_l3503_350386


namespace NUMINAMATH_CALUDE_jump_rope_solution_l3503_350313

/-- The cost of jump ropes A and B satisfy the given conditions -/
def jump_rope_cost (cost_A cost_B : ℝ) : Prop :=
  10 * cost_A + 5 * cost_B = 175 ∧ 15 * cost_A + 10 * cost_B = 300

/-- The solution to the jump rope cost problem -/
theorem jump_rope_solution :
  ∃ (cost_A cost_B : ℝ), jump_rope_cost cost_A cost_B ∧ cost_A = 10 ∧ cost_B = 15 := by
  sorry

#check jump_rope_solution

end NUMINAMATH_CALUDE_jump_rope_solution_l3503_350313


namespace NUMINAMATH_CALUDE_complement_of_union_M_N_l3503_350396

-- Define the sets M and N
def M : Set ℝ := {x | (x + 3) * (x - 1) < 0}
def N : Set ℝ := {x | x ≤ -3}

-- State the theorem
theorem complement_of_union_M_N :
  (M ∪ N)ᶜ = {x : ℝ | x ≥ 1} := by sorry

end NUMINAMATH_CALUDE_complement_of_union_M_N_l3503_350396


namespace NUMINAMATH_CALUDE_sqrt_meaningful_range_l3503_350377

theorem sqrt_meaningful_range (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = 1 + 3 * x) ↔ x ≥ -1/3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_meaningful_range_l3503_350377


namespace NUMINAMATH_CALUDE_root_implies_coefficients_l3503_350332

theorem root_implies_coefficients (p q : ℝ) : 
  (2 * (Complex.I * 2 - 3)^2 + p * (Complex.I * 2 - 3) + q = 0) → 
  (p = 12 ∧ q = 26) := by
  sorry

end NUMINAMATH_CALUDE_root_implies_coefficients_l3503_350332


namespace NUMINAMATH_CALUDE_division_problem_l3503_350345

/-- Given a division problem with quotient, divisor, and remainder, calculate the dividend -/
theorem division_problem (quotient divisor remainder : ℕ) (h1 : quotient = 256) (h2 : divisor = 3892) (h3 : remainder = 354) :
  divisor * quotient + remainder = 996706 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l3503_350345


namespace NUMINAMATH_CALUDE_coffee_shop_spending_l3503_350356

theorem coffee_shop_spending (ryan_spent : ℝ) (sarah_spent : ℝ) : 
  (sarah_spent = 0.60 * ryan_spent) →
  (ryan_spent = sarah_spent + 12.50) →
  (ryan_spent + sarah_spent = 50.00) :=
by
  sorry

end NUMINAMATH_CALUDE_coffee_shop_spending_l3503_350356


namespace NUMINAMATH_CALUDE_twelve_factorial_mod_thirteen_l3503_350387

def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | m + 1 => (m + 1) * factorial m

theorem twelve_factorial_mod_thirteen : factorial 12 % 13 = 12 := by
  sorry

end NUMINAMATH_CALUDE_twelve_factorial_mod_thirteen_l3503_350387


namespace NUMINAMATH_CALUDE_empty_boxes_count_l3503_350384

/-- The number of boxes containing neither pens, pencils, nor markers -/
def empty_boxes (total boxes_with_pencils boxes_with_pens boxes_with_both_pens_pencils
                 boxes_with_markers boxes_with_pencils_markers : ℕ) : ℕ :=
  total - (boxes_with_pencils + boxes_with_pens - boxes_with_both_pens_pencils + 
           boxes_with_markers - boxes_with_pencils_markers)

theorem empty_boxes_count :
  ∀ (total boxes_with_pencils boxes_with_pens boxes_with_both_pens_pencils
     boxes_with_markers boxes_with_pencils_markers : ℕ),
  total = 15 →
  boxes_with_pencils = 9 →
  boxes_with_pens = 5 →
  boxes_with_both_pens_pencils = 3 →
  boxes_with_markers = 4 →
  boxes_with_pencils_markers = 2 →
  boxes_with_markers ≤ boxes_with_pencils →
  boxes_with_both_pens_pencils ≤ min boxes_with_pencils boxes_with_pens →
  boxes_with_pencils_markers ≤ min boxes_with_pencils boxes_with_markers →
  empty_boxes total boxes_with_pencils boxes_with_pens boxes_with_both_pens_pencils
              boxes_with_markers boxes_with_pencils_markers = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_empty_boxes_count_l3503_350384


namespace NUMINAMATH_CALUDE_tangency_point_l3503_350336

def parabola1 (x y : ℝ) : Prop := y = 2 * x^2 + 10 * x + 14

def parabola2 (x y : ℝ) : Prop := x = 4 * y^2 + 16 * y + 68

def point_of_tangency (x y : ℝ) : Prop :=
  parabola1 x y ∧ parabola2 x y

theorem tangency_point : 
  point_of_tangency (-9/4) (-15/8) := by sorry

end NUMINAMATH_CALUDE_tangency_point_l3503_350336


namespace NUMINAMATH_CALUDE_square_of_product_divided_by_square_l3503_350365

theorem square_of_product_divided_by_square (m n : ℝ) :
  (2 * m * n)^2 / n^2 = 4 * m^2 := by
  sorry

end NUMINAMATH_CALUDE_square_of_product_divided_by_square_l3503_350365


namespace NUMINAMATH_CALUDE_min_sum_with_exponential_constraint_l3503_350381

theorem min_sum_with_exponential_constraint (a b : ℝ) :
  a > 0 → b > 0 → (2 : ℝ)^a * 4^b = (2^a)^b →
  (∀ x y : ℝ, x > 0 → y > 0 → (2 : ℝ)^x * 4^y = (2^x)^y → a + b ≤ x + y) →
  a + b = 3 + 2 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_with_exponential_constraint_l3503_350381


namespace NUMINAMATH_CALUDE_angle_between_vectors_not_necessarily_alpha_minus_beta_l3503_350352

theorem angle_between_vectors_not_necessarily_alpha_minus_beta 
  (α β : ℝ) (a b : ℝ × ℝ) :
  a = (Real.cos α, Real.sin α) →
  b = (Real.cos β, Real.sin β) →
  a ≠ b →
  ∃ θ, Real.cos θ = Real.cos α * Real.cos β + Real.sin α * Real.sin β ∧ θ ≠ α - β :=
by sorry

end NUMINAMATH_CALUDE_angle_between_vectors_not_necessarily_alpha_minus_beta_l3503_350352


namespace NUMINAMATH_CALUDE_remaining_content_is_two_fifteenths_l3503_350327

/-- The fraction of content remaining after four days of evaporation -/
def remaining_content : ℚ :=
  let day1_remaining := 1 - 2/3
  let day2_remaining := day1_remaining * (1 - 1/4)
  let day3_remaining := day2_remaining * (1 - 1/5)
  let day4_remaining := day3_remaining * (1 - 1/3)
  day4_remaining

/-- Theorem stating that the remaining content after four days is 2/15 -/
theorem remaining_content_is_two_fifteenths :
  remaining_content = 2/15 := by
  sorry

end NUMINAMATH_CALUDE_remaining_content_is_two_fifteenths_l3503_350327


namespace NUMINAMATH_CALUDE_birch_tree_probability_l3503_350358

/-- The probability of no two birch trees being adjacent when planting trees in a row -/
theorem birch_tree_probability (maple oak birch : ℕ) (h1 : maple = 4) (h2 : oak = 3) (h3 : birch = 6) :
  let total := maple + oak + birch
  let non_birch := maple + oak
  let favorable := Nat.choose (non_birch + 1) birch
  let total_arrangements := Nat.choose total birch
  (favorable : ℚ) / total_arrangements = 7 / 429 :=
by sorry

end NUMINAMATH_CALUDE_birch_tree_probability_l3503_350358


namespace NUMINAMATH_CALUDE_car_distance_theorem_l3503_350363

/-- Given a car traveling at a constant speed for a certain time, 
    calculate the distance covered. -/
def distance_covered (speed : ℝ) (time : ℝ) : ℝ :=
  speed * time

/-- Theorem stating that a car traveling at 107 km/h for 6.5 hours
    covers a distance of 695.5 km. -/
theorem car_distance_theorem :
  distance_covered 107 6.5 = 695.5 := by
  sorry

end NUMINAMATH_CALUDE_car_distance_theorem_l3503_350363


namespace NUMINAMATH_CALUDE_correct_matching_probability_l3503_350359

/-- The number of celebrities and baby pictures --/
def n : ℕ := 3

/-- The total number of possible arrangements --/
def total_arrangements : ℕ := n.factorial

/-- The number of correct arrangements --/
def correct_arrangements : ℕ := 1

/-- The probability of correctly matching all celebrities to their baby pictures --/
def probability : ℚ := correct_arrangements / total_arrangements

theorem correct_matching_probability :
  probability = 1 / 6 := by sorry

end NUMINAMATH_CALUDE_correct_matching_probability_l3503_350359


namespace NUMINAMATH_CALUDE_sin_shift_l3503_350380

theorem sin_shift (x : ℝ) : Real.sin (3 * x - π / 3) = Real.sin (3 * (x - π / 9)) := by
  sorry

end NUMINAMATH_CALUDE_sin_shift_l3503_350380


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3503_350392

theorem complex_equation_solution (a b : ℂ) (t : ℝ) :
  (Complex.abs a = 3) →
  (Complex.abs b = 5) →
  (a * b = t - 3 + 5 * Complex.I) →
  (t > 0) →
  (t = 3 + 10 * Real.sqrt 2) := by
sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3503_350392


namespace NUMINAMATH_CALUDE_parabola_vertex_c_value_l3503_350375

/-- Given a parabola of the form y = 2x^2 + c with vertex at (0,1), prove that c = 1 -/
theorem parabola_vertex_c_value (c : ℝ) : 
  (∀ x y : ℝ, y = 2 * x^2 + c) →   -- Parabola equation
  (0, 1) = (0, 2 * 0^2 + c) →      -- Vertex at (0,1)
  c = 1 := by sorry

end NUMINAMATH_CALUDE_parabola_vertex_c_value_l3503_350375


namespace NUMINAMATH_CALUDE_mosaic_perimeter_l3503_350350

/-- A mosaic constructed with a regular hexagon, squares, and equilateral triangles. -/
structure Mosaic where
  hexagon_side_length : ℝ
  num_squares : ℕ
  num_triangles : ℕ

/-- The outside perimeter of the mosaic. -/
def outside_perimeter (m : Mosaic) : ℝ :=
  (m.num_squares + m.num_triangles) * m.hexagon_side_length

/-- Theorem stating that the outside perimeter of the specific mosaic is 240 cm. -/
theorem mosaic_perimeter :
  ∀ (m : Mosaic),
    m.hexagon_side_length = 20 →
    m.num_squares = 6 →
    m.num_triangles = 6 →
    outside_perimeter m = 240 := by
  sorry

end NUMINAMATH_CALUDE_mosaic_perimeter_l3503_350350


namespace NUMINAMATH_CALUDE_raja_income_proof_l3503_350324

/-- Raja's monthly income in rupees -/
def monthly_income : ℝ := 37500

/-- Percentage spent on household items -/
def household_percentage : ℝ := 35

/-- Percentage spent on clothes -/
def clothes_percentage : ℝ := 20

/-- Percentage spent on medicines -/
def medicine_percentage : ℝ := 5

/-- Amount saved in rupees -/
def savings : ℝ := 15000

theorem raja_income_proof :
  monthly_income * (1 - (household_percentage + clothes_percentage + medicine_percentage) / 100) = savings := by
  sorry

#check raja_income_proof

end NUMINAMATH_CALUDE_raja_income_proof_l3503_350324


namespace NUMINAMATH_CALUDE_parabolic_arch_bridge_width_l3503_350357

/-- Parabolic arch bridge problem -/
theorem parabolic_arch_bridge_width 
  (a : ℝ) 
  (h1 : a = -8) 
  (h2 : 4^2 = a * (-2)) 
  : let new_y := -3/2
    let new_x := Real.sqrt (a * new_y)
    2 * new_x = 4 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_parabolic_arch_bridge_width_l3503_350357


namespace NUMINAMATH_CALUDE_min_sum_given_product_l3503_350315

theorem min_sum_given_product (x y : ℤ) (h : x * y = 144) : 
  ∀ a b : ℤ, a * b = 144 → x + y ≤ a + b ∧ ∃ c d : ℤ, c * d = 144 ∧ c + d = -145 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_given_product_l3503_350315


namespace NUMINAMATH_CALUDE_equilateral_triangle_area_l3503_350390

/-- The area of an equilateral triangle with vertices at (1, 2), (1, 8), and (7, 2) is 9√3 square units. -/
theorem equilateral_triangle_area : 
  let E : ℝ × ℝ := (1, 2)
  let F : ℝ × ℝ := (1, 8)
  let G : ℝ × ℝ := (7, 2)
  let is_equilateral (A B C : ℝ × ℝ) : Prop := 
    (A.1 - B.1)^2 + (A.2 - B.2)^2 = (B.1 - C.1)^2 + (B.2 - C.2)^2 ∧
    (B.1 - C.1)^2 + (B.2 - C.2)^2 = (C.1 - A.1)^2 + (C.2 - A.2)^2
  let triangle_area (A B C : ℝ × ℝ) : ℝ := 
    Real.sqrt 3 / 4 * ((A.1 - B.1)^2 + (A.2 - B.2)^2)
  is_equilateral E F G → triangle_area E F G = 9 * Real.sqrt 3 :=
by
  sorry


end NUMINAMATH_CALUDE_equilateral_triangle_area_l3503_350390


namespace NUMINAMATH_CALUDE_intersection_of_lines_AB_CD_l3503_350325

/-- The intersection point of two lines in 3D space --/
def intersection_point (A B C D : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let a := A
  let b := B
  let c := C
  let d := D
  (20, -18, 11)

/-- Theorem stating that the intersection point of lines AB and CD is (20, -18, 11) --/
theorem intersection_of_lines_AB_CD :
  let A : ℝ × ℝ × ℝ := (8, -6, 5)
  let B : ℝ × ℝ × ℝ := (18, -16, 10)
  let C : ℝ × ℝ × ℝ := (-4, 6, -12)
  let D : ℝ × ℝ × ℝ := (4, -4, 8)
  intersection_point A B C D = (20, -18, 11) := by
  sorry

#check intersection_of_lines_AB_CD

end NUMINAMATH_CALUDE_intersection_of_lines_AB_CD_l3503_350325


namespace NUMINAMATH_CALUDE_total_cleaning_time_is_136_l3503_350399

/-- The time in minutes Richard takes to clean his room once -/
def richard_time : ℕ := 22

/-- The time in minutes Cory takes to clean her room once -/
def cory_time : ℕ := richard_time + 3

/-- The time in minutes Blake takes to clean his room once -/
def blake_time : ℕ := cory_time - 4

/-- The number of times they clean their rooms per week -/
def cleanings_per_week : ℕ := 2

/-- The total time spent cleaning rooms by all three people in a week -/
def total_cleaning_time : ℕ := (richard_time + cory_time + blake_time) * cleanings_per_week

theorem total_cleaning_time_is_136 : total_cleaning_time = 136 := by
  sorry

end NUMINAMATH_CALUDE_total_cleaning_time_is_136_l3503_350399


namespace NUMINAMATH_CALUDE_min_value_sum_l3503_350398

-- Define the function f(x)
def f (x : ℝ) : ℝ := 3 * x - x^3

-- Define a as the point where f(x) reaches its minimum value
def a : ℝ := sorry

-- Define b as the minimum value of f(x)
def b : ℝ := f a

-- Theorem statement
theorem min_value_sum :
  ∀ x : ℝ, f x ≥ b ∧ a + b = -3 :=
sorry

end NUMINAMATH_CALUDE_min_value_sum_l3503_350398


namespace NUMINAMATH_CALUDE_nonagon_diagonal_intersections_l3503_350333

/-- The number of vertices in a regular nonagon -/
def n : ℕ := 9

/-- The number of distinct intersection points of diagonals in the interior of a regular nonagon -/
def intersection_points (n : ℕ) : ℕ := Nat.choose n 4

/-- Theorem: The number of distinct intersection points of diagonals in the interior of a regular nonagon is 126 -/
theorem nonagon_diagonal_intersections :
  intersection_points n = 126 := by sorry

end NUMINAMATH_CALUDE_nonagon_diagonal_intersections_l3503_350333


namespace NUMINAMATH_CALUDE_max_rounds_four_teams_one_match_l3503_350342

/-- Represents a round-robin tournament with 18 teams -/
structure Tournament :=
  (teams : Finset (Fin 18))
  (rounds : Fin 17 → Finset (Fin 18 × Fin 18))
  (round_valid : ∀ r, (rounds r).card = 9)
  (round_pairs : ∀ r t, (t ∈ teams) → (∃! u, (t, u) ∈ rounds r ∨ (u, t) ∈ rounds r))
  (all_play_all : ∀ t u, t ≠ u → (∃! r, (t, u) ∈ rounds r ∨ (u, t) ∈ rounds r))

/-- The property that there exist 4 teams with exactly 1 match played among them -/
def has_four_teams_one_match (T : Tournament) (n : ℕ) : Prop :=
  ∃ (a b c d : Fin 18), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    (∃! (i j : Fin 18) (r : Fin n), 
      ((i = a ∧ j = b) ∨ (i = a ∧ j = c) ∨ (i = a ∧ j = d) ∨
       (i = b ∧ j = c) ∨ (i = b ∧ j = d) ∨ (i = c ∧ j = d)) ∧
      ((i, j) ∈ T.rounds r ∨ (j, i) ∈ T.rounds r))

/-- The main theorem statement -/
theorem max_rounds_four_teams_one_match (T : Tournament) :
  (∀ n ≤ 7, has_four_teams_one_match T n) ∧
  (∃ n > 7, ¬has_four_teams_one_match T n) :=
sorry

end NUMINAMATH_CALUDE_max_rounds_four_teams_one_match_l3503_350342


namespace NUMINAMATH_CALUDE_polynomial_simplification_l3503_350385

theorem polynomial_simplification (x : ℝ) : 
  (3 * x^3 + 3 * x^2 + 8 * x - 5) - (x^3 + 6 * x^2 + 2 * x - 15) - (2 * x^3 + x^2 + 4 * x - 8) = 
  -4 * x^2 + 2 * x + 18 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l3503_350385


namespace NUMINAMATH_CALUDE_fraction_exponent_product_l3503_350320

theorem fraction_exponent_product : 
  (8 / 9 : ℚ) ^ 3 * (1 / 3 : ℚ) ^ 3 = 512 / 19683 := by sorry

end NUMINAMATH_CALUDE_fraction_exponent_product_l3503_350320


namespace NUMINAMATH_CALUDE_investment_return_is_25_percent_l3503_350388

/-- Calculates the percentage return on investment for a given dividend rate, face value, and purchase price of shares. -/
def percentage_return_on_investment (dividend_rate : ℚ) (face_value : ℚ) (purchase_price : ℚ) : ℚ :=
  (dividend_rate * face_value / purchase_price) * 100

/-- Theorem stating that for the given conditions, the percentage return on investment is 25%. -/
theorem investment_return_is_25_percent :
  let dividend_rate : ℚ := 125 / 1000
  let face_value : ℚ := 60
  let purchase_price : ℚ := 30
  percentage_return_on_investment dividend_rate face_value purchase_price = 25 := by
sorry

#eval percentage_return_on_investment (125/1000) 60 30

end NUMINAMATH_CALUDE_investment_return_is_25_percent_l3503_350388


namespace NUMINAMATH_CALUDE_total_blocks_l3503_350323

theorem total_blocks (initial_blocks additional_blocks : ℕ) :
  initial_blocks = 86 →
  additional_blocks = 9 →
  initial_blocks + additional_blocks = 95 :=
by sorry

end NUMINAMATH_CALUDE_total_blocks_l3503_350323


namespace NUMINAMATH_CALUDE_large_pizza_cost_l3503_350354

/-- Represents the cost and size of a pizza --/
structure Pizza where
  side_length : ℝ
  cost : ℝ

/-- Calculates the area of a square pizza --/
def pizza_area (p : Pizza) : ℝ := p.side_length ^ 2

theorem large_pizza_cost : ∃ (large_pizza : Pizza),
  let small_pizza := Pizza.mk 12 10
  let total_budget := 60
  let separate_purchase_area := 2 * (total_budget / small_pizza.cost * pizza_area small_pizza)
  large_pizza.side_length = 18 ∧
  large_pizza.cost = 21.6 ∧
  (total_budget / large_pizza.cost * pizza_area large_pizza) = separate_purchase_area + 36 := by
  sorry

end NUMINAMATH_CALUDE_large_pizza_cost_l3503_350354


namespace NUMINAMATH_CALUDE_complete_residue_system_l3503_350329

theorem complete_residue_system (m : ℕ) (x : Fin m → ℤ) 
  (h_incongruent : ∀ i j : Fin m, i ≠ j → x i % m ≠ x j % m) :
  ∀ y : ℤ, ∃ i : Fin m, y % m = x i % m :=
by sorry

end NUMINAMATH_CALUDE_complete_residue_system_l3503_350329


namespace NUMINAMATH_CALUDE_divisible_by_nine_l3503_350335

/-- Sum of digits function -/
def sum_of_digits (n : ℕ) : ℕ := sorry

theorem divisible_by_nine (N : ℕ) : 
  sum_of_digits N = sum_of_digits (5 * N) → N % 9 = 0 := by sorry

end NUMINAMATH_CALUDE_divisible_by_nine_l3503_350335


namespace NUMINAMATH_CALUDE_isosceles_triangle_base_length_l3503_350389

/-- An isosceles triangle with congruent sides of 7 cm and perimeter of 21 cm has a base of 7 cm. -/
theorem isosceles_triangle_base_length : 
  ∀ (base congruent_side : ℝ),
  congruent_side = 7 →
  base + 2 * congruent_side = 21 →
  base = 7 := by
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_base_length_l3503_350389


namespace NUMINAMATH_CALUDE_farm_animals_feet_count_l3503_350301

theorem farm_animals_feet_count (total_heads : ℕ) (hen_count : ℕ) : 
  total_heads = 48 → hen_count = 28 → 
  (hen_count * 2 + (total_heads - hen_count) * 4 : ℕ) = 136 := by
  sorry

end NUMINAMATH_CALUDE_farm_animals_feet_count_l3503_350301


namespace NUMINAMATH_CALUDE_problem_solution_l3503_350393

def A (a : ℚ) : Set ℚ := {a^2, a+1, -3}
def B (a : ℚ) : Set ℚ := {a-3, 3*a-1, a^2+1}
def C (m : ℚ) : Set ℚ := {x | m*x = 1}

theorem problem_solution (a m : ℚ) 
  (h1 : A a ∩ B a = {-3}) 
  (h2 : C m ⊆ A a ∩ B a) : 
  a = -2/3 ∧ (m = 0 ∨ m = -1/3) := by
  sorry


end NUMINAMATH_CALUDE_problem_solution_l3503_350393


namespace NUMINAMATH_CALUDE_sum_of_integers_l3503_350318

theorem sum_of_integers (a b : ℕ+) (h1 : a - b = 4) (h2 : a * b = 63) : a + b = 18 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_integers_l3503_350318


namespace NUMINAMATH_CALUDE_complex_fraction_equals_two_l3503_350310

theorem complex_fraction_equals_two (z : ℂ) (h : z = 1 - I) : z^2 / (z - 1) = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equals_two_l3503_350310


namespace NUMINAMATH_CALUDE_equation_solution_l3503_350331

theorem equation_solution : ∃! x : ℝ, x + (x + 1) + (x + 2) + (x + 3) = 18 ∧ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3503_350331


namespace NUMINAMATH_CALUDE_no_solution_condition_l3503_350307

theorem no_solution_condition (a : ℝ) : 
  (∀ x : ℝ, x ≠ 1 → (a * x) / (x - 1) ≠ 1 / (x - 1) + 2) ↔ (a = 1 ∨ a = 2) :=
sorry

end NUMINAMATH_CALUDE_no_solution_condition_l3503_350307


namespace NUMINAMATH_CALUDE_viggo_age_ratio_l3503_350347

theorem viggo_age_ratio :
  ∀ (viggo_current_age brother_current_age M Y : ℕ),
    viggo_current_age + brother_current_age = 32 →
    brother_current_age = 10 →
    viggo_current_age - brother_current_age = M * 2 + Y - 2 →
    (M * 2 + Y) / 2 = 7 := by
  sorry

end NUMINAMATH_CALUDE_viggo_age_ratio_l3503_350347


namespace NUMINAMATH_CALUDE_dot_product_of_specific_vectors_l3503_350370

theorem dot_product_of_specific_vectors :
  let a : ℝ × ℝ := (-2, 4)
  let b : ℝ × ℝ := (1, 2)
  (a.1 * b.1 + a.2 * b.2) = 6 := by
  sorry

end NUMINAMATH_CALUDE_dot_product_of_specific_vectors_l3503_350370


namespace NUMINAMATH_CALUDE_arithmetic_to_geometric_sequence_l3503_350371

/-- Given three numbers in an arithmetic sequence with a ratio of 3:4:5,
    if increasing the smallest number by 1 forms a geometric sequence,
    then the original three numbers are 15, 20, and 25. -/
theorem arithmetic_to_geometric_sequence (a d : ℝ) : 
  (a - d : ℝ) / 3 = a / 4 ∧ a / 4 = (a + d) / 5 →  -- arithmetic sequence with ratio 3:4:5
  ∃ r : ℝ, (a - d + 1) / a = a / (a + d) ∧ a / (a + d) = r →  -- geometric sequence after increasing smallest by 1
  a - d = 15 ∧ a = 20 ∧ a + d = 25 := by  -- original numbers are 15, 20, 25
sorry

end NUMINAMATH_CALUDE_arithmetic_to_geometric_sequence_l3503_350371


namespace NUMINAMATH_CALUDE_factorial_sum_representations_l3503_350303

/-- For any natural number n ≥ 4, there exist at least n! ways to write n! as a sum of elements
    from the set {1!, 2!, ..., (n-1)!}, where each element can be used multiple times. -/
theorem factorial_sum_representations (n : ℕ) (h : n ≥ 4) :
  ∃ (ways : ℕ), ways ≥ n! ∧
    ∀ (representation : List ℕ),
      (∀ k ∈ representation, k ∈ Finset.range n ∧ k > 0) →
      representation.sum = n! →
      (ways : ℕ) ≥ (representation.map Nat.factorial).sum :=
by sorry

end NUMINAMATH_CALUDE_factorial_sum_representations_l3503_350303


namespace NUMINAMATH_CALUDE_exists_quadrilateral_equal_angle_tangents_l3503_350309

/-- A planar quadrilateral is represented by its four interior angles -/
structure PlanarQuadrilateral where
  α : Real
  β : Real
  γ : Real
  δ : Real
  sum_360 : α + β + γ + δ = 360

/-- Theorem: There exists a planar quadrilateral where the tangents of all its interior angles are equal -/
theorem exists_quadrilateral_equal_angle_tangents : 
  ∃ q : PlanarQuadrilateral, Real.tan q.α = Real.tan q.β ∧ Real.tan q.β = Real.tan q.γ ∧ Real.tan q.γ = Real.tan q.δ :=
sorry

end NUMINAMATH_CALUDE_exists_quadrilateral_equal_angle_tangents_l3503_350309


namespace NUMINAMATH_CALUDE_speed_of_train2_l3503_350344

-- Define the problem parameters
def distance_between_stations : ℝ := 200
def speed_of_train1 : ℝ := 20
def start_time_train1 : ℝ := 7
def start_time_train2 : ℝ := 8
def meeting_time : ℝ := 12

-- Define the theorem
theorem speed_of_train2 (speed_train2 : ℝ) : speed_train2 = 25 := by
  -- Assuming the conditions of the problem
  have h1 : distance_between_stations = 200 := by rfl
  have h2 : speed_of_train1 = 20 := by rfl
  have h3 : start_time_train1 = 7 := by rfl
  have h4 : start_time_train2 = 8 := by rfl
  have h5 : meeting_time = 12 := by rfl

  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_speed_of_train2_l3503_350344


namespace NUMINAMATH_CALUDE_triangle_with_angle_ratio_1_2_3_is_right_triangle_l3503_350361

/-- A triangle with interior angles in the ratio 1:2:3 is a right triangle. -/
theorem triangle_with_angle_ratio_1_2_3_is_right_triangle (α β γ : ℝ) :
  α > 0 ∧ β > 0 ∧ γ > 0 →  -- Angles are positive
  α + β + γ = 180 →        -- Sum of angles in a triangle is 180°
  β = 2 * α ∧ γ = 3 * α →  -- Angles are in the ratio 1:2:3
  γ = 90                   -- The largest angle is 90°
  := by sorry

end NUMINAMATH_CALUDE_triangle_with_angle_ratio_1_2_3_is_right_triangle_l3503_350361


namespace NUMINAMATH_CALUDE_mixture_replacement_l3503_350317

/-- Represents the mixture replacement problem -/
theorem mixture_replacement (initial_a initial_b replaced_amount : ℝ) : 
  initial_a = 64 →
  initial_b = initial_a / 4 →
  (initial_a - (4/5) * replaced_amount) / (initial_b - (1/5) * replaced_amount + replaced_amount) = 2/3 →
  replaced_amount = 40 :=
by
  sorry

#check mixture_replacement

end NUMINAMATH_CALUDE_mixture_replacement_l3503_350317


namespace NUMINAMATH_CALUDE_ellipse_and_line_intersection_l3503_350391

-- Define the ellipse C
def ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

-- Define the line L
def line (m : ℝ) (x y : ℝ) : Prop :=
  y = x + m

-- Define the theorem
theorem ellipse_and_line_intersection
  (a b : ℝ)
  (h_positive : a > b ∧ b > 0)
  (h_axis : b = a / 2)
  (h_max_distance : a + (a^2 - b^2).sqrt = 2 + Real.sqrt 3)
  (m : ℝ)
  (h_area : ∃ x₁ y₁ x₂ y₂ : ℝ,
    ellipse a b x₁ y₁ ∧
    ellipse a b x₂ y₂ ∧
    line m x₁ y₁ ∧
    line m x₂ y₂ ∧
    x₁ ≠ x₂ ∧
    abs ((x₂ - x₁) * (y₂ + y₁) / 2) = 1) :
  (a^2 = 4 ∧ b^2 = 1) ∧ m^2 = 5/2 :=
sorry

end NUMINAMATH_CALUDE_ellipse_and_line_intersection_l3503_350391


namespace NUMINAMATH_CALUDE_book_sale_fraction_l3503_350339

theorem book_sale_fraction (price : ℝ) (remaining : ℕ) (total_received : ℝ) :
  price = 3.5 →
  remaining = 36 →
  total_received = 252 →
  ∃ (total : ℕ) (sold : ℕ),
    total > 0 ∧
    sold = total - remaining ∧
    (sold : ℝ) / total = 2 / 3 ∧
    price * sold = total_received :=
by sorry

end NUMINAMATH_CALUDE_book_sale_fraction_l3503_350339


namespace NUMINAMATH_CALUDE_difference_of_squares_64_36_l3503_350362

theorem difference_of_squares_64_36 : 64^2 - 36^2 = 2800 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_64_36_l3503_350362


namespace NUMINAMATH_CALUDE_nikkas_stamp_collection_l3503_350300

theorem nikkas_stamp_collection :
  ∀ (total_stamps : ℕ) 
    (chinese_percentage : ℚ) 
    (us_percentage : ℚ) 
    (japanese_stamps : ℕ),
  chinese_percentage = 35 / 100 →
  us_percentage = 20 / 100 →
  japanese_stamps = 45 →
  (1 - chinese_percentage - us_percentage) * total_stamps = japanese_stamps →
  total_stamps = 100 := by
sorry

end NUMINAMATH_CALUDE_nikkas_stamp_collection_l3503_350300
