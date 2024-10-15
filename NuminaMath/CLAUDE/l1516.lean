import Mathlib

namespace NUMINAMATH_CALUDE_triangle_side_length_l1516_151649

theorem triangle_side_length (a b c : ℝ) (A : ℝ) :
  a = Real.sqrt 5 →
  c = 2 →
  Real.cos A = 2/3 →
  b = 3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l1516_151649


namespace NUMINAMATH_CALUDE_watch_cost_price_l1516_151646

theorem watch_cost_price (C : ℝ) : 
  (C * 0.9 = C * (1 - 0.1)) →  -- Selling at 90% of C is a 10% loss
  (C * 1.03 = C * (1 + 0.03)) →  -- Selling at 103% of C is a 3% gain
  (C * 1.03 - C * 0.9 = 140) →  -- Difference between selling prices is 140
  C = 1076.92 := by
sorry

end NUMINAMATH_CALUDE_watch_cost_price_l1516_151646


namespace NUMINAMATH_CALUDE_angle_d_measure_l1516_151657

/-- Given a triangle ABC with angles A = 85°, B = 34°, and C = 21°,
    if a smaller triangle is formed within ABC with one of its angles being D,
    then the measure of angle D is 140°. -/
theorem angle_d_measure (A B C D : Real) : 
  A = 85 → B = 34 → C = 21 → 
  A + B + C = 180 →
  ∃ (E F : Real), E ≥ 0 ∧ F ≥ 0 ∧ D + E + F = 180 ∧ A + B + C + E + F = 180 →
  D = 140 := by sorry

end NUMINAMATH_CALUDE_angle_d_measure_l1516_151657


namespace NUMINAMATH_CALUDE_brick_width_is_10_l1516_151638

-- Define the dimensions of the brick and wall
def brick_length : ℝ := 20
def brick_height : ℝ := 7.5
def wall_length : ℝ := 2700  -- 27 m in cm
def wall_width : ℝ := 200    -- 2 m in cm
def wall_height : ℝ := 75    -- 0.75 m in cm
def num_bricks : ℕ := 27000

-- Theorem to prove the width of the brick
theorem brick_width_is_10 :
  ∃ (brick_width : ℝ),
    brick_width = 10 ∧
    brick_length * brick_width * brick_height * num_bricks =
    wall_length * wall_width * wall_height :=
by sorry

end NUMINAMATH_CALUDE_brick_width_is_10_l1516_151638


namespace NUMINAMATH_CALUDE_simple_interest_problem_l1516_151611

theorem simple_interest_problem (interest : ℝ) (rate : ℝ) (time : ℝ) (principal : ℝ) : 
  interest = 4016.25 →
  rate = 9 →
  time = 5 →
  principal = interest / (rate * time / 100) →
  principal = 8925 := by
sorry

end NUMINAMATH_CALUDE_simple_interest_problem_l1516_151611


namespace NUMINAMATH_CALUDE_zain_coins_count_and_value_l1516_151637

def quarter_value : ℚ := 25 / 100
def dime_value : ℚ := 10 / 100
def nickel_value : ℚ := 5 / 100
def penny_value : ℚ := 1 / 100
def half_dollar_value : ℚ := 50 / 100

def emerie_quarters : ℕ := 6
def emerie_dimes : ℕ := 7
def emerie_nickels : ℕ := 5
def emerie_pennies : ℕ := 10
def emerie_half_dollars : ℕ := 2

def zain_more_coins : ℕ := 10

def zain_quarters : ℕ := emerie_quarters + zain_more_coins
def zain_dimes : ℕ := emerie_dimes + zain_more_coins
def zain_nickels : ℕ := emerie_nickels + zain_more_coins
def zain_pennies : ℕ := emerie_pennies + zain_more_coins
def zain_half_dollars : ℕ := emerie_half_dollars + zain_more_coins

def zain_total_coins : ℕ := zain_quarters + zain_dimes + zain_nickels + zain_pennies + zain_half_dollars

def zain_total_value : ℚ :=
  zain_quarters * quarter_value +
  zain_dimes * dime_value +
  zain_nickels * nickel_value +
  zain_pennies * penny_value +
  zain_half_dollars * half_dollar_value

theorem zain_coins_count_and_value :
  zain_total_coins = 80 ∧ zain_total_value ≤ 20 := by sorry

end NUMINAMATH_CALUDE_zain_coins_count_and_value_l1516_151637


namespace NUMINAMATH_CALUDE_ten_power_plus_eight_div_nine_is_integer_l1516_151690

theorem ten_power_plus_eight_div_nine_is_integer (n : ℕ) : ∃ k : ℤ, (10^n : ℤ) + 8 = 9 * k := by
  sorry

end NUMINAMATH_CALUDE_ten_power_plus_eight_div_nine_is_integer_l1516_151690


namespace NUMINAMATH_CALUDE_sum_of_tens_equal_hundred_to_ten_l1516_151666

theorem sum_of_tens_equal_hundred_to_ten (n : ℕ) : 
  (n * 10 = 100^10) → (n = 10^19) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_tens_equal_hundred_to_ten_l1516_151666


namespace NUMINAMATH_CALUDE_selenas_remaining_money_is_38_l1516_151608

/-- Calculates the remaining money for Selena after her meal -/
def selenas_remaining_money (tip : ℚ) (steak_price : ℚ) (steak_count : ℕ) 
  (burger_price : ℚ) (burger_count : ℕ) (icecream_price : ℚ) (icecream_count : ℕ) : ℚ :=
  tip - (steak_price * steak_count + burger_price * burger_count + icecream_price * icecream_count)

/-- Theorem stating that Selena will be left with $38 after her meal -/
theorem selenas_remaining_money_is_38 :
  selenas_remaining_money 99 24 2 3.5 2 2 3 = 38 := by
  sorry

end NUMINAMATH_CALUDE_selenas_remaining_money_is_38_l1516_151608


namespace NUMINAMATH_CALUDE_tile_arrangement_l1516_151639

/-- The internal angle of a square in degrees -/
def square_angle : ℝ := 90

/-- The internal angle of an octagon in degrees -/
def octagon_angle : ℝ := 135

/-- The sum of angles around a vertex in degrees -/
def vertex_sum : ℝ := 360

/-- The number of square tiles around a vertex -/
def num_square_tiles : ℕ := 1

/-- The number of octagonal tiles around a vertex -/
def num_octagon_tiles : ℕ := 2

theorem tile_arrangement :
  num_square_tiles * square_angle + num_octagon_tiles * octagon_angle = vertex_sum :=
by sorry

end NUMINAMATH_CALUDE_tile_arrangement_l1516_151639


namespace NUMINAMATH_CALUDE_cubic_root_sum_cubes_over_product_l1516_151677

theorem cubic_root_sum_cubes_over_product (p q a b c : ℝ) : 
  q ≠ 0 → 
  (∀ x : ℝ, x^3 + p*x + q = (x-a)*(x-b)*(x-c)) → 
  (a^3 + b^3 + c^3) / (a*b*c) = 3 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_sum_cubes_over_product_l1516_151677


namespace NUMINAMATH_CALUDE_quadratic_inequality_specific_case_l1516_151600

theorem quadratic_inequality (k : ℝ) :
  (∀ x : ℝ, x^2 - k*x + 4 > 0) ↔ -4 < k ∧ k < 4 :=
by sorry

theorem specific_case :
  ∀ x : ℝ, x^2 - 5*x + 4 > 0 ↔ x < 1 ∨ x > 4 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_specific_case_l1516_151600


namespace NUMINAMATH_CALUDE_shane_minimum_score_l1516_151616

def exam_count : ℕ := 5
def max_score : ℕ := 100
def goal_average : ℕ := 86
def first_three_scores : List ℕ := [81, 72, 93]

theorem shane_minimum_score :
  let total_needed : ℕ := goal_average * exam_count
  let scored_so_far : ℕ := first_three_scores.sum
  let remaining_needed : ℕ := total_needed - scored_so_far
  remaining_needed - max_score = 84 :=
by sorry

end NUMINAMATH_CALUDE_shane_minimum_score_l1516_151616


namespace NUMINAMATH_CALUDE_sum_six_equals_twentyfour_l1516_151648

/-- An arithmetic sequence {a_n} with sum S_n -/
structure ArithmeticSequence where
  S : ℕ → ℝ
  is_arithmetic : ∀ n : ℕ, S (n + 2) - S (n + 1) = S (n + 1) - S n

/-- The sum of the first n terms of the arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℝ := seq.S n

theorem sum_six_equals_twentyfour (seq : ArithmeticSequence) 
  (h2 : sum_n seq 2 = 2) 
  (h4 : sum_n seq 4 = 10) : 
  sum_n seq 6 = 24 := by
  sorry

end NUMINAMATH_CALUDE_sum_six_equals_twentyfour_l1516_151648


namespace NUMINAMATH_CALUDE_john_video_release_l1516_151650

/-- Calculates the total minutes of video released per week by John --/
def total_video_minutes_per_week (short_video_length : ℕ) (long_video_multiplier : ℕ) (short_videos_per_day : ℕ) (days_per_week : ℕ) : ℕ :=
  let long_video_length := short_video_length * long_video_multiplier
  let total_minutes_per_day := short_video_length * short_videos_per_day + long_video_length
  total_minutes_per_day * days_per_week

/-- Theorem stating that John releases 112 minutes of video per week --/
theorem john_video_release : 
  total_video_minutes_per_week 2 6 2 7 = 112 := by
  sorry


end NUMINAMATH_CALUDE_john_video_release_l1516_151650


namespace NUMINAMATH_CALUDE_number_problem_l1516_151663

theorem number_problem (N : ℝ) : 
  (1/4 : ℝ) * (1/3 : ℝ) * (2/5 : ℝ) * N = 15 → (40/100 : ℝ) * N = 180 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l1516_151663


namespace NUMINAMATH_CALUDE_marks_trees_l1516_151669

theorem marks_trees (current_trees : ℕ) 
  (h : current_trees + 12 = 25) : current_trees = 13 := by
  sorry

end NUMINAMATH_CALUDE_marks_trees_l1516_151669


namespace NUMINAMATH_CALUDE_three_same_one_different_probability_l1516_151672

/-- The probability of a child being born a boy or a girl -/
def child_probability : ℚ := 1 / 2

/-- The number of children in the family -/
def num_children : ℕ := 4

/-- The number of possible combinations for having three children of one sex and one of the opposite sex -/
def num_combinations : ℕ := 8

/-- The probability of having three children of one sex and one of the opposite sex in a family of four children -/
theorem three_same_one_different_probability :
  (child_probability ^ num_children) * num_combinations = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_three_same_one_different_probability_l1516_151672


namespace NUMINAMATH_CALUDE_expression_factorization_l1516_151640

theorem expression_factorization (x : ℝ) : 
  (12 * x^4 - 27 * x^3 + 45 * x) - (-3 * x^4 - 6 * x^3 + 9 * x) = 3 * x * (5 * x^3 - 7 * x^2 + 12) := by
  sorry

end NUMINAMATH_CALUDE_expression_factorization_l1516_151640


namespace NUMINAMATH_CALUDE_mike_unbroken_seashells_l1516_151622

/-- The number of unbroken seashells Mike found -/
def unbroken_seashells (total : ℕ) (broken : ℕ) : ℕ :=
  total - broken

/-- Theorem stating that Mike found 2 unbroken seashells -/
theorem mike_unbroken_seashells :
  unbroken_seashells 6 4 = 2 := by
  sorry

end NUMINAMATH_CALUDE_mike_unbroken_seashells_l1516_151622


namespace NUMINAMATH_CALUDE_min_distance_circle_parabola_l1516_151641

/-- The minimum distance between a point on a circle and a point on a parabola -/
theorem min_distance_circle_parabola :
  ∀ (A B : ℝ × ℝ),
  (A.1^2 + A.2^2 = 16) →
  (B.2 = B.1^2 - 4) →
  (∃ (θ : ℝ), A = (4 * Real.cos θ, 4 * Real.sin θ)) →
  (∃ (x : ℝ), B = (x, x^2 - 4)) →
  (∃ (d : ℝ), d = Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)) →
  (∀ (d' : ℝ), d' ≥ d) →
  (∃ (x : ℝ), -2*(4*Real.cos θ - x) + 2*(4*Real.sin θ - (x^2 - 4))*(-2*x) = 0) :=
by sorry

end NUMINAMATH_CALUDE_min_distance_circle_parabola_l1516_151641


namespace NUMINAMATH_CALUDE_series_sum_l1516_151618

/-- The sum of the infinite series Σ(n=1 to ∞) ((3n - 2) / (n(n+1)(n+3))) equals 61/24 -/
theorem series_sum : ∑' n, (3 * n - 2) / (n * (n + 1) * (n + 3)) = 61 / 24 := by
  sorry

end NUMINAMATH_CALUDE_series_sum_l1516_151618


namespace NUMINAMATH_CALUDE_olly_owns_three_dogs_l1516_151662

/-- The number of shoes needed for each animal -/
def shoes_per_animal : ℕ := 4

/-- The total number of shoes needed -/
def total_shoes : ℕ := 24

/-- The number of cats Olly owns -/
def num_cats : ℕ := 2

/-- The number of ferrets Olly owns -/
def num_ferrets : ℕ := 1

/-- Calculates the number of dogs Olly owns -/
def num_dogs : ℕ :=
  (total_shoes - (num_cats + num_ferrets) * shoes_per_animal) / shoes_per_animal

theorem olly_owns_three_dogs : num_dogs = 3 := by
  sorry

end NUMINAMATH_CALUDE_olly_owns_three_dogs_l1516_151662


namespace NUMINAMATH_CALUDE_complete_graph_inequality_l1516_151679

/-- Given n points on a plane with no three collinear and some connected by line segments,
    N_k denotes the number of complete graphs of k points. -/
def N (n k : ℕ) : ℕ := sorry

theorem complete_graph_inequality (n : ℕ) (h_n : n > 1) :
  ∀ k ∈ Finset.range (n - 1) \ {0, 1},
  N n k ≠ 0 →
  (N n (k + 1) : ℝ) / (N n k) ≥ 
    (1 : ℝ) / ((k^2 : ℝ) - 1) * ((k^2 : ℝ) * (N n k) / (N n (k + 1)) - n) := by
  sorry

end NUMINAMATH_CALUDE_complete_graph_inequality_l1516_151679


namespace NUMINAMATH_CALUDE_rhombus_diagonal_roots_l1516_151660

theorem rhombus_diagonal_roots (m : ℝ) : 
  let side_length : ℝ := 5
  let quadratic (x : ℝ) := x^2 + (2*m - 1)*x + m^2 + 3
  ∃ (OA OB : ℝ), 
    OA^2 + OB^2 = side_length^2 ∧ 
    quadratic OA = 0 ∧ 
    quadratic OB = 0 →
    m = -3 :=
by sorry

end NUMINAMATH_CALUDE_rhombus_diagonal_roots_l1516_151660


namespace NUMINAMATH_CALUDE_similar_canister_capacity_l1516_151689

/-- Given that a small canister with volume 24 cm³ can hold 100 nails,
    prove that a similar canister with volume 72 cm³ can hold 300 nails,
    assuming the nails are packed in the same manner. -/
theorem similar_canister_capacity
  (small_volume : ℝ)
  (small_nails : ℕ)
  (large_volume : ℝ)
  (h1 : small_volume = 24)
  (h2 : small_nails = 100)
  (h3 : large_volume = 72)
  (h4 : small_volume > 0)
  (h5 : large_volume > 0) :
  (large_volume / small_volume) * small_nails = 300 := by
  sorry

#check similar_canister_capacity

end NUMINAMATH_CALUDE_similar_canister_capacity_l1516_151689


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l1516_151691

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 + 1 > 0) ↔ (∃ x : ℝ, x^2 + 1 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l1516_151691


namespace NUMINAMATH_CALUDE_buy_three_items_ways_l1516_151670

/-- The number of headphones available for sale. -/
def headphones : ℕ := 9

/-- The number of computer mice available for sale. -/
def mice : ℕ := 13

/-- The number of keyboards available for sale. -/
def keyboards : ℕ := 5

/-- The number of "keyboard and mouse" sets available. -/
def keyboard_mouse_sets : ℕ := 4

/-- The number of "headphones and mouse" sets available. -/
def headphones_mouse_sets : ℕ := 5

/-- The total number of ways to buy three items: headphones, a keyboard, and a mouse. -/
def total_ways : ℕ := 646

/-- Theorem stating that the total number of ways to buy three items
    (headphones, keyboard, and mouse) is 646. -/
theorem buy_three_items_ways :
  headphones * keyboard_mouse_sets +
  keyboards * headphones_mouse_sets +
  headphones * mice * keyboards = total_ways := by
  sorry

end NUMINAMATH_CALUDE_buy_three_items_ways_l1516_151670


namespace NUMINAMATH_CALUDE_quadratic_function_property_l1516_151603

/-- Given a positive real number a and a function f(x) = ax^2 + 2ax + 1,
    if f(m) < 0 for some real m, then f(m+2) > 1 -/
theorem quadratic_function_property (a : ℝ) (m : ℝ) (h_a : a > 0) :
  let f := λ x : ℝ ↦ a * x^2 + 2 * a * x + 1
  f m < 0 → f (m + 2) > 1 := by sorry

end NUMINAMATH_CALUDE_quadratic_function_property_l1516_151603


namespace NUMINAMATH_CALUDE_distance_origin_to_line_l1516_151612

/-- The distance from the origin to the line x = 1 is 1. -/
theorem distance_origin_to_line : ∃ d : ℝ, d = 1 ∧ 
  ∀ (x y : ℝ), x = 1 → d = |x| := by sorry

end NUMINAMATH_CALUDE_distance_origin_to_line_l1516_151612


namespace NUMINAMATH_CALUDE_triangle_area_change_l1516_151626

theorem triangle_area_change (base height : ℝ) (base_new height_new : ℝ) :
  base_new = base * 1.1 →
  height_new = height * 0.95 →
  (base_new * height_new) / (base * height) - 1 = 0.045 := by
sorry

end NUMINAMATH_CALUDE_triangle_area_change_l1516_151626


namespace NUMINAMATH_CALUDE_equation_solution_l1516_151693

theorem equation_solution (x y z : ℚ) 
  (eq1 : 3 * x - 2 * y - 3 * z = 0)
  (eq2 : x + 5 * y - 12 * z = 0)
  (z_neq_0 : z ≠ 0) :
  (x^2 - 2*x*y) / (y^2 + 2*z^2) = -1053/1547 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1516_151693


namespace NUMINAMATH_CALUDE_subset_if_intersection_eq_elem_of_union_if_elem_of_intersection_fraction_inequality_necessary_not_sufficient_exists_non_positive_square_l1516_151695

-- Statement 1
theorem subset_if_intersection_eq (A B : Set α) : A ∩ B = A → A ⊆ B := by sorry

-- Statement 2
theorem elem_of_union_if_elem_of_intersection {A B : Set α} {x : α} :
  x ∈ A ∩ B → x ∈ A ∪ B := by sorry

-- Statement 3
theorem fraction_inequality_necessary_not_sufficient {a b : ℝ} :
  (a < b ∧ b < 0) → b / a < a / b := by sorry

-- Statement 4
theorem exists_non_positive_square : ∃ x : ℤ, x^2 ≤ 0 := by sorry

end NUMINAMATH_CALUDE_subset_if_intersection_eq_elem_of_union_if_elem_of_intersection_fraction_inequality_necessary_not_sufficient_exists_non_positive_square_l1516_151695


namespace NUMINAMATH_CALUDE_trig_identity_l1516_151681

theorem trig_identity (α β : ℝ) : 
  Real.sin α ^ 2 + Real.sin β ^ 2 - Real.sin α ^ 2 * Real.sin β ^ 2 + Real.cos α ^ 2 * Real.cos β ^ 2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l1516_151681


namespace NUMINAMATH_CALUDE_line_x_intercept_l1516_151699

/-- The x-intercept of a straight line passing through points (2, -4) and (6, 8) is 10/3 -/
theorem line_x_intercept :
  let p1 : ℝ × ℝ := (2, -4)
  let p2 : ℝ × ℝ := (6, 8)
  let m : ℝ := (p2.2 - p1.2) / (p2.1 - p1.1)
  let b : ℝ := p1.2 - m * p1.1
  let x_intercept : ℝ := -b / m
  x_intercept = 10 / 3 := by
  sorry

end NUMINAMATH_CALUDE_line_x_intercept_l1516_151699


namespace NUMINAMATH_CALUDE_club_officer_selection_l1516_151655

/-- Represents the number of ways to choose officers in a club --/
def choose_officers (total_members : ℕ) (experienced_members : ℕ) : ℕ :=
  experienced_members * (experienced_members - 1) * (total_members - 2) * (total_members - 3)

/-- Theorem stating the number of ways to choose officers in the given club scenario --/
theorem club_officer_selection :
  let total_members : ℕ := 12
  let experienced_members : ℕ := 4
  choose_officers total_members experienced_members = 1080 := by
  sorry

end NUMINAMATH_CALUDE_club_officer_selection_l1516_151655


namespace NUMINAMATH_CALUDE_max_dot_product_CA_CB_l1516_151684

/-- Given planar vectors OA, OB, and OC satisfying certain conditions,
    the maximum value of CA · CB is 3. -/
theorem max_dot_product_CA_CB (OA OB OC : ℝ × ℝ) : 
  (OA.1 * OB.1 + OA.2 * OB.2 = 0) →  -- OA · OB = 0
  (OA.1^2 + OA.2^2 = 1) →            -- |OA| = 1
  (OC.1^2 + OC.2^2 = 1) →            -- |OC| = 1
  (OB.1^2 + OB.2^2 = 3) →            -- |OB| = √3
  (∃ (CA CB : ℝ × ℝ), 
    CA = (OA.1 - OC.1, OA.2 - OC.2) ∧ 
    CB = (OB.1 - OC.1, OB.2 - OC.2) ∧
    ∀ (CA' CB' : ℝ × ℝ), 
      CA' = (OA.1 - OC.1, OA.2 - OC.2) → 
      CB' = (OB.1 - OC.1, OB.2 - OC.2) →
      CA'.1 * CB'.1 + CA'.2 * CB'.2 ≤ 3) :=
by sorry

end NUMINAMATH_CALUDE_max_dot_product_CA_CB_l1516_151684


namespace NUMINAMATH_CALUDE_arcsin_neg_sqrt3_over_2_l1516_151694

theorem arcsin_neg_sqrt3_over_2 : Real.arcsin (-Real.sqrt 3 / 2) = -π / 3 := by
  sorry

end NUMINAMATH_CALUDE_arcsin_neg_sqrt3_over_2_l1516_151694


namespace NUMINAMATH_CALUDE_segment_area_equilateral_triangle_l1516_151675

/-- The area of a circular segment cut off by one side of an equilateral triangle inscribed in a circle -/
theorem segment_area_equilateral_triangle (a : ℝ) (ha : a > 0) :
  let R := a / Real.sqrt 3
  let sector_area := π * R^2 / 3
  let triangle_area := a * R / 4
  sector_area - triangle_area = (a^2 * (4 * π - 3 * Real.sqrt 3)) / 36 := by
  sorry

end NUMINAMATH_CALUDE_segment_area_equilateral_triangle_l1516_151675


namespace NUMINAMATH_CALUDE_equilateral_triangle_area_perimeter_ratio_l1516_151617

/-- The ratio of the area to the perimeter of an equilateral triangle with side length 10 -/
theorem equilateral_triangle_area_perimeter_ratio :
  let side_length : ℝ := 10
  let height : ℝ := side_length * (Real.sqrt 3 / 2)
  let area : ℝ := (1 / 2) * side_length * height
  let perimeter : ℝ := 3 * side_length
  (area / perimeter) = (5 * Real.sqrt 3) / 6 := by
sorry

end NUMINAMATH_CALUDE_equilateral_triangle_area_perimeter_ratio_l1516_151617


namespace NUMINAMATH_CALUDE_pythagorean_triple_identification_l1516_151697

def is_pythagorean_triple (a b c : ℕ) : Prop :=
  a^2 + b^2 = c^2

theorem pythagorean_triple_identification :
  ¬(is_pythagorean_triple 1 2 3) ∧
  ¬(is_pythagorean_triple 4 5 6) ∧
  ¬(is_pythagorean_triple 6 8 9) ∧
  is_pythagorean_triple 7 24 25 :=
by sorry

end NUMINAMATH_CALUDE_pythagorean_triple_identification_l1516_151697


namespace NUMINAMATH_CALUDE_cylinder_line_segment_distance_l1516_151609

/-- Represents a cylinder with a square axial cross-section -/
structure SquareCylinder where
  -- We don't need to define specific properties here

/-- Represents a line segment connecting points on the top and bottom bases of the cylinder -/
structure LineSegment where
  length : ℝ
  angle : ℝ

/-- 
Theorem: For a cylinder with a square axial cross-section, given a line segment of length l 
connecting points on the top and bottom base circumferences and making an angle α with the base plane, 
the distance d from this line segment to the cylinder axis is (l/2) * sqrt(-cos(2α)), 
and the valid range for α is π/4 < α < 3π/4.
-/
theorem cylinder_line_segment_distance (c : SquareCylinder) (seg : LineSegment) :
  let l := seg.length
  let α := seg.angle
  let d := (l / 2) * Real.sqrt (-Real.cos (2 * α))
  d > 0 ∧ π / 4 < α ∧ α < 3 * π / 4 := by
  sorry


end NUMINAMATH_CALUDE_cylinder_line_segment_distance_l1516_151609


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_S_l1516_151687

/-- The set of numbers from 9 to 999999999, where each number consists of all 9s -/
def S : Finset ℕ := Finset.image (λ i => (10^i - 1) / 9) (Finset.range 9)

/-- The arithmetic mean of the set S -/
def M : ℕ := (Finset.sum S id) / Finset.card S

theorem arithmetic_mean_of_S : M = 123456789 := by sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_S_l1516_151687


namespace NUMINAMATH_CALUDE_fraction_equality_l1516_151653

theorem fraction_equality (a b : ℝ) (h : a / b = 1 / 2) : a / (a + b) = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1516_151653


namespace NUMINAMATH_CALUDE_inequality_proof_l1516_151683

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h : x * y + y * z + z * x = 1) : 
  3 - Real.sqrt 3 + x^2 / y + y^2 / z + z^2 / x ≥ (x + y + z)^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1516_151683


namespace NUMINAMATH_CALUDE_power_of_128_l1516_151624

theorem power_of_128 : (128 : ℝ) ^ (7/3) = 65536 * (2 : ℝ) ^ (1/3) := by sorry

end NUMINAMATH_CALUDE_power_of_128_l1516_151624


namespace NUMINAMATH_CALUDE_product_of_numbers_l1516_151685

theorem product_of_numbers (x y : ℝ) 
  (h1 : x - y = 11) 
  (h2 : x^2 + y^2 = 185) : 
  x * y = 26 := by
sorry

end NUMINAMATH_CALUDE_product_of_numbers_l1516_151685


namespace NUMINAMATH_CALUDE_knicks_equivalence_l1516_151661

/-- Conversion rate between knicks and knacks -/
def knicks_to_knacks : ℚ := 3 / 8

/-- Conversion rate between knacks and knocks -/
def knacks_to_knocks : ℚ := 6 / 5

/-- The number of knicks equivalent to 30 knocks -/
def knicks_for_30_knocks : ℚ := 30 * (1 / knacks_to_knocks) * (1 / knicks_to_knacks)

theorem knicks_equivalence :
  knicks_for_30_knocks = 200 / 3 :=
sorry

end NUMINAMATH_CALUDE_knicks_equivalence_l1516_151661


namespace NUMINAMATH_CALUDE_green_blue_difference_l1516_151665

/-- Represents the colors of disks in the bag -/
inductive DiskColor
  | Blue
  | Yellow
  | Green

/-- Represents the bag of disks -/
structure DiskBag where
  total : Nat
  ratio : Fin 3 → Nat
  colorCount : DiskColor → Nat

/-- The theorem stating the difference between green and blue disks -/
theorem green_blue_difference (bag : DiskBag) 
  (h_total : bag.total = 54)
  (h_ratio : bag.ratio = ![3, 7, 8])
  (h_color_count : ∀ c, bag.colorCount c = (bag.total / (bag.ratio 0 + bag.ratio 1 + bag.ratio 2)) * match c with
    | DiskColor.Blue => bag.ratio 0
    | DiskColor.Yellow => bag.ratio 1
    | DiskColor.Green => bag.ratio 2) :
  bag.colorCount DiskColor.Green - bag.colorCount DiskColor.Blue = 15 := by
  sorry

end NUMINAMATH_CALUDE_green_blue_difference_l1516_151665


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l1516_151656

def U : Set ℕ := {x | x ≤ 6}
def A : Set ℕ := {1, 3, 5}
def B : Set ℕ := {4, 5, 6}

theorem complement_intersection_theorem :
  (U \ A) ∩ B = {4, 6} := by
  sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l1516_151656


namespace NUMINAMATH_CALUDE_eight_spotlights_illuminate_space_l1516_151659

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a spotlight that can illuminate an octant -/
structure Spotlight where
  position : Point3D
  direction : Point3D  -- Normalized vector representing the direction

/-- Represents the space to be illuminated -/
def Space : Type := Unit

/-- Checks if a spotlight illuminates a given point in space -/
def illuminates (s : Spotlight) (p : Point3D) : Prop := sorry

/-- Checks if a set of spotlights illuminates the entire space -/
def illuminatesEntireSpace (spotlights : Finset Spotlight) : Prop := 
  ∀ p : Point3D, ∃ s ∈ spotlights, illuminates s p

/-- The main theorem stating that 8 spotlights can illuminate the entire space -/
theorem eight_spotlights_illuminate_space 
  (points : Finset Point3D) 
  (h : points.card = 8) : 
  ∃ spotlights : Finset Spotlight, 
    spotlights.card = 8 ∧ 
    (∀ s ∈ spotlights, ∃ p ∈ points, s.position = p) ∧
    illuminatesEntireSpace spotlights := by
  sorry

end NUMINAMATH_CALUDE_eight_spotlights_illuminate_space_l1516_151659


namespace NUMINAMATH_CALUDE_line_through_point_with_slope_l1516_151629

/-- Represents a line in 2D space -/
structure Line where
  slope : ℝ
  yIntercept : ℝ

/-- Checks if a point (x, y) lies on the line -/
def Line.containsPoint (l : Line) (x y : ℝ) : Prop :=
  y = l.slope * x + l.yIntercept

/-- The equation of the line in the form ax + by + c = 0 -/
def Line.equation (l : Line) (x y : ℝ) : Prop :=
  l.slope * x - y + l.yIntercept = 0

theorem line_through_point_with_slope (x₀ y₀ m : ℝ) :
  ∃ (l : Line), l.slope = m ∧ l.containsPoint x₀ y₀ ∧
  ∀ (x y : ℝ), l.equation x y ↔ (2 : ℝ) * x - y + 3 = 0 :=
sorry

end NUMINAMATH_CALUDE_line_through_point_with_slope_l1516_151629


namespace NUMINAMATH_CALUDE_isosceles_triangle_base_angles_l1516_151678

-- Define an isosceles triangle with one interior angle of 50°
structure IsoscelesTriangle :=
  (base_angle₁ : ℝ)
  (base_angle₂ : ℝ)
  (vertex_angle : ℝ)
  (is_isosceles : base_angle₁ = base_angle₂)
  (has_50_degree_angle : base_angle₁ = 50 ∨ base_angle₂ = 50 ∨ vertex_angle = 50)
  (angle_sum : base_angle₁ + base_angle₂ + vertex_angle = 180)

-- Theorem stating that the base angles are either 50° or 65°
theorem isosceles_triangle_base_angles (t : IsoscelesTriangle) :
  t.base_angle₁ = 50 ∨ t.base_angle₁ = 65 :=
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_base_angles_l1516_151678


namespace NUMINAMATH_CALUDE_quadratic_equation_root_values_l1516_151688

theorem quadratic_equation_root_values (a : ℝ) : 
  (∃ x : ℂ, x^2 - 2*a*x + a^2 - 4*a = 0 ∧ Complex.abs x = 3) →
  (a = 1 ∨ a = 9 ∨ a = 2 - Real.sqrt 13) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_root_values_l1516_151688


namespace NUMINAMATH_CALUDE_lily_correct_answers_percentage_l1516_151605

theorem lily_correct_answers_percentage
  (t : ℝ)
  (h_t_positive : t > 0)
  (h_max_alone : 0.7 * (t / 2) = 0.35 * t)
  (h_max_total : 0.82 * t = 0.82 * t)
  (h_lily_alone : 0.85 * (t / 2) = 0.425 * t)
  (h_solved_together : 0.82 * t - 0.35 * t = 0.47 * t) :
  (0.425 * t + 0.47 * t) / t = 0.895 := by
  sorry

#check lily_correct_answers_percentage

end NUMINAMATH_CALUDE_lily_correct_answers_percentage_l1516_151605


namespace NUMINAMATH_CALUDE_expression_evaluation_l1516_151602

theorem expression_evaluation : 
  (((15^15 / 15^14)^3 * 8^3) / 4^6 : ℚ) = 1728000 / 4096 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1516_151602


namespace NUMINAMATH_CALUDE_value_of_expression_l1516_151642

theorem value_of_expression (x y : ℝ) (hx : x = 12) (hy : y = 7) :
  (x - y) * (x + y) = 95 := by sorry

end NUMINAMATH_CALUDE_value_of_expression_l1516_151642


namespace NUMINAMATH_CALUDE_cookie_count_consistency_l1516_151614

theorem cookie_count_consistency (total_cookies : ℕ) (eaten_cookies : ℕ) (remaining_cookies : ℕ) 
  (h1 : total_cookies = 32)
  (h2 : eaten_cookies = 9)
  (h3 : remaining_cookies = 23) :
  total_cookies - eaten_cookies = remaining_cookies := by
  sorry

#check cookie_count_consistency

end NUMINAMATH_CALUDE_cookie_count_consistency_l1516_151614


namespace NUMINAMATH_CALUDE_max_area_triangle_l1516_151604

-- Define the basic structures
structure Point where
  x : ℝ
  y : ℝ

structure Triangle where
  A : Point
  B : Point
  C : Point

-- Define the properties
def isAcute (t : Triangle) : Prop := sorry

def isSimilar (t1 t2 : Triangle) : Prop := sorry

def circumscribes (t1 t2 : Triangle) : Prop := sorry

def area (t : Triangle) : ℝ := sorry

-- State the theorem
theorem max_area_triangle 
  (A₀B₀C₀ : Triangle) 
  (A'B'C' : Triangle) 
  (h1 : isAcute A₀B₀C₀) 
  (h2 : isAcute A'B'C') :
  ∃ (A₁B₁C₁ : Triangle),
    isSimilar A₁B₁C₁ A'B'C' ∧ 
    circumscribes A₁B₁C₁ A₀B₀C₀ ∧
    ∀ (ABC : Triangle),
      isSimilar ABC A'B'C' → 
      circumscribes ABC A₀B₀C₀ → 
      area ABC ≤ area A₁B₁C₁ :=
sorry

end NUMINAMATH_CALUDE_max_area_triangle_l1516_151604


namespace NUMINAMATH_CALUDE_equation_solution_l1516_151682

theorem equation_solution :
  ∃ x : ℚ, (2 * x + 3 * x = 500 - (4 * x + 5 * x) + 20) ∧ (x = 520 / 14) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1516_151682


namespace NUMINAMATH_CALUDE_prime_pairs_divisibility_l1516_151635

theorem prime_pairs_divisibility (p q : ℕ) : 
  Nat.Prime p → Nat.Prime q → 
  (∃ k : ℤ, 30 * q - 1 = k * p) → 
  (∃ m : ℤ, 30 * p - 1 = m * q) → 
  ((p = 7 ∧ q = 11) ∨ (p = 11 ∧ q = 7) ∨ (p = 59 ∧ q = 61) ∨ (p = 61 ∧ q = 59)) :=
by sorry

end NUMINAMATH_CALUDE_prime_pairs_divisibility_l1516_151635


namespace NUMINAMATH_CALUDE_cosine_of_angle_between_lines_l1516_151625

def vector1 : ℝ × ℝ := (4, -1)
def vector2 : ℝ × ℝ := (2, 5)

theorem cosine_of_angle_between_lines :
  let v1 := vector1
  let v2 := vector2
  let dot_product := v1.1 * v2.1 + v1.2 * v2.2
  let magnitude1 := Real.sqrt (v1.1^2 + v1.2^2)
  let magnitude2 := Real.sqrt (v2.1^2 + v2.2^2)
  dot_product / (magnitude1 * magnitude2) = 3 / Real.sqrt 493 := by
  sorry

end NUMINAMATH_CALUDE_cosine_of_angle_between_lines_l1516_151625


namespace NUMINAMATH_CALUDE_expression_simplification_l1516_151698

theorem expression_simplification (k : ℚ) :
  (6 * k + 12) / 6 = k + 2 ∧
  ∃ (a b : ℤ), k + 2 = a * k + b ∧ a = 1 ∧ b = 2 ∧ a / b = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1516_151698


namespace NUMINAMATH_CALUDE_hyperbola_focus_m_value_l1516_151632

/-- Given a hyperbola with equation (y^2/m) - (x^2/9) = 1 and a focus at (0,5), prove that m = 16 -/
theorem hyperbola_focus_m_value (m : ℝ) : 
  (∃ (x y : ℝ), y^2/m - x^2/9 = 1) →  -- Hyperbola equation exists
  (0, 5) ∈ {p : ℝ × ℝ | p.1 = 0 ∧ p.2^2 = m + 9} →  -- (0,5) is a focus
  m = 16 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_focus_m_value_l1516_151632


namespace NUMINAMATH_CALUDE_point_transformation_l1516_151620

-- Define the rotation function
def rotate180 (x y : ℝ) : ℝ × ℝ := (2 - x, 10 - y)

-- Define the reflection function
def reflect_y_eq_x (x y : ℝ) : ℝ × ℝ := (y, x)

-- Theorem statement
theorem point_transformation (a b : ℝ) :
  let (x', y') := rotate180 a b
  let (x'', y'') := reflect_y_eq_x x' y'
  (x'' = 3 ∧ y'' = -6) → b - a = -1 := by
  sorry

end NUMINAMATH_CALUDE_point_transformation_l1516_151620


namespace NUMINAMATH_CALUDE_isosceles_triangle_exists_unique_l1516_151658

/-- An isosceles triangle with given heights -/
structure IsoscelesTriangle where
  /-- Height corresponding to the base -/
  m_a : ℝ
  /-- Height corresponding to one of the equal sides -/
  m_b : ℝ
  /-- Condition for existence -/
  h : 2 * m_a > m_b

/-- Theorem stating the existence and uniqueness of an isosceles triangle with given heights -/
theorem isosceles_triangle_exists_unique (m_a m_b : ℝ) :
  Nonempty (Unique (IsoscelesTriangle)) ↔ 2 * m_a > m_b :=
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_exists_unique_l1516_151658


namespace NUMINAMATH_CALUDE_pyramid_sphere_theorem_l1516_151615

/-- Represents a triangular pyramid with a sphere touching its edges -/
structure PyramidWithSphere where
  -- Base triangle side length
  base_side : ℝ
  -- Height of the pyramid
  height : ℝ
  -- Radius of the inscribed sphere
  sphere_radius : ℝ

/-- Properties of the pyramid and sphere system -/
def pyramid_sphere_properties (p : PyramidWithSphere) : Prop :=
  -- Base is an equilateral triangle
  p.base_side = 8 ∧
  -- Height of the pyramid
  p.height = 15 ∧
  -- Sphere touches edges of the pyramid
  ∃ (aa₁ : ℝ) (dist_o_bc : ℝ),
    -- Distance from vertex A to point of contact A₁
    aa₁ = 6 ∧
    -- Distance from sphere center O to edge BC
    dist_o_bc = 18 / 5 ∧
    -- Radius of the sphere
    p.sphere_radius = 4 * Real.sqrt 39 / 5

/-- Theorem stating the properties of the pyramid and sphere system -/
theorem pyramid_sphere_theorem (p : PyramidWithSphere) :
  pyramid_sphere_properties p → 
  ∃ (aa₁ : ℝ) (dist_o_bc : ℝ),
    aa₁ = 6 ∧
    dist_o_bc = 18 / 5 ∧
    p.sphere_radius = 4 * Real.sqrt 39 / 5 :=
by sorry

end NUMINAMATH_CALUDE_pyramid_sphere_theorem_l1516_151615


namespace NUMINAMATH_CALUDE_isabella_currency_exchange_l1516_151692

theorem isabella_currency_exchange :
  ∃ d : ℕ+, 
    (8 * d.val : ℚ) / 5 - 80 = d.val ∧ 
    (d.val / 100 + (d.val % 100) / 10 + d.val % 10 = 9) := by
  sorry

end NUMINAMATH_CALUDE_isabella_currency_exchange_l1516_151692


namespace NUMINAMATH_CALUDE_approximate_12000_accuracy_l1516_151607

/-- Represents an approximate number with its value and significant digits -/
structure ApproximateNumber where
  value : ℕ
  significantDigits : ℕ

/-- Determines the number of significant digits in an approximate number -/
def countSignificantDigits (n : ℕ) : ℕ :=
  sorry

theorem approximate_12000_accuracy :
  let n : ApproximateNumber := ⟨12000, countSignificantDigits 12000⟩
  n.significantDigits = 2 := by sorry

end NUMINAMATH_CALUDE_approximate_12000_accuracy_l1516_151607


namespace NUMINAMATH_CALUDE_quadratic_prime_roots_unique_k_l1516_151631

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

theorem quadratic_prime_roots_unique_k : 
  ∃! k : ℕ, ∃ p q : ℕ, 
    is_prime p ∧ is_prime q ∧ 
    p + q = 58 ∧ 
    p * q = k ∧ 
    ∀ x : ℝ, x^2 - 58*x + k = 0 ↔ (x = p ∨ x = q) :=
sorry

end NUMINAMATH_CALUDE_quadratic_prime_roots_unique_k_l1516_151631


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l1516_151696

theorem triangle_abc_properties (A B C : Real) (a b : Real) (S : Real) :
  A = 30 * Real.pi / 180 →
  B = 45 * Real.pi / 180 →
  a = Real.sqrt 2 →
  b = a * Real.sin B / Real.sin A →
  C = Real.pi - A - B →
  S = 1/2 * a * b * Real.sin C →
  b = 2 ∧ S = (Real.sqrt 3 + 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_abc_properties_l1516_151696


namespace NUMINAMATH_CALUDE_circle_C_properties_l1516_151673

/-- The circle C passing through A(4,1) and tangent to x-y-1=0 at B(2,1) -/
def circle_C : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - 3)^2 + p.2^2 = 4}

/-- Point A -/
def point_A : ℝ × ℝ := (4, 1)

/-- Point B -/
def point_B : ℝ × ℝ := (2, 1)

/-- The line x-y-1=0 -/
def tangent_line (p : ℝ × ℝ) : Prop :=
  p.1 - p.2 - 1 = 0

theorem circle_C_properties :
  point_A ∈ circle_C ∧
  point_B ∈ circle_C ∧
  tangent_line point_B ∧
  ∀ (p : ℝ × ℝ), p ∈ circle_C ∧ tangent_line p → p = point_B :=
by sorry

end NUMINAMATH_CALUDE_circle_C_properties_l1516_151673


namespace NUMINAMATH_CALUDE_steps_to_top_floor_l1516_151647

/-- The number of steps between each floor in the building -/
def steps_between_floors : ℕ := 13

/-- The total number of floors in the building -/
def total_floors : ℕ := 7

/-- The number of intervals between floors when going from ground to top floor -/
def floor_intervals : ℕ := total_floors - 1

/-- The total number of steps from ground floor to the top floor -/
def total_steps : ℕ := steps_between_floors * floor_intervals

theorem steps_to_top_floor :
  total_steps = 78 :=
sorry

end NUMINAMATH_CALUDE_steps_to_top_floor_l1516_151647


namespace NUMINAMATH_CALUDE_rectangle_area_l1516_151623

/-- Given a rectangle with length 15 cm and perimeter-to-width ratio of 5:1, its area is 150 cm² -/
theorem rectangle_area (w : ℝ) (h1 : (2 * 15 + 2 * w) / w = 5) : w * 15 = 150 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l1516_151623


namespace NUMINAMATH_CALUDE_sum_of_fractions_inequality_l1516_151652

theorem sum_of_fractions_inequality (x y z : ℝ) 
  (hx : x > -1) (hy : y > -1) (hz : z > -1) : 
  (1 + x^2) / (1 + y + z^2) + 
  (1 + y^2) / (1 + z + x^2) + 
  (1 + z^2) / (1 + x + y^2) ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_inequality_l1516_151652


namespace NUMINAMATH_CALUDE_geometric_series_relation_l1516_151643

/-- Given two infinite geometric series with the specified conditions, prove that n = 20/3 -/
theorem geometric_series_relation (a₁ b₁ a₂ b₂ n : ℝ) : 
  a₁ = 15 ∧ b₁ = 5 ∧ a₂ = 15 ∧ b₂ = 5 + n ∧ 
  (a₁ / (1 - b₁ / a₁)) * 3 = a₂ / (1 - b₂ / a₂) → 
  n = 20 / 3 := by
sorry

end NUMINAMATH_CALUDE_geometric_series_relation_l1516_151643


namespace NUMINAMATH_CALUDE_greatest_constant_for_triangle_inequality_l1516_151628

theorem greatest_constant_for_triangle_inequality (a b c : ℝ) :
  (a > 0) → (b > 0) → (c > 0) →
  (a + b > c) → (b + c > a) → (c + a > b) →
  (∃ (N : ℝ), ∀ (a b c : ℝ), 
    (a > 0) → (b > 0) → (c > 0) →
    (a + b > c) → (b + c > a) → (c + a > b) →
    (a^2 + b^2 + a*b) / c^2 > N) ∧
  (∀ (M : ℝ), 
    (∀ (a b c : ℝ), 
      (a > 0) → (b > 0) → (c > 0) →
      (a + b > c) → (b + c > a) → (c + a > b) →
      (a^2 + b^2 + a*b) / c^2 > M) →
    M ≤ 3/4) :=
sorry

end NUMINAMATH_CALUDE_greatest_constant_for_triangle_inequality_l1516_151628


namespace NUMINAMATH_CALUDE_dihedral_angle_cosine_l1516_151633

/-- Represents a regular hexagonal pyramid -/
structure RegularHexagonalPyramid where
  -- Add necessary fields here
  mk :: -- Constructor

/-- Represents a line in 3D space -/
structure Line3D where
  -- Add necessary fields here
  mk :: -- Constructor

/-- Represents a circle in 3D space -/
structure Circle3D where
  -- Add necessary fields here
  mk :: -- Constructor

/-- Function to check if a line intersects the base side of the pyramid -/
def intersectsBaseSide (p : RegularHexagonalPyramid) (l : Line3D) : Prop :=
  sorry

/-- Function to get the circle inscribed around a lateral face -/
def lateralFaceInscribedCircle (p : RegularHexagonalPyramid) : Circle3D :=
  sorry

/-- Function to get the circle inscribed around the larger diagonal cross-section -/
def diagonalCrossSectionInscribedCircle (p : RegularHexagonalPyramid) : Circle3D :=
  sorry

/-- Function to check if a line passes through the centers of two circles -/
def passesThroughCenters (l : Line3D) (c1 c2 : Circle3D) : Prop :=
  sorry

/-- Function to calculate the dihedral angle at the base -/
def dihedralAngleAtBase (p : RegularHexagonalPyramid) : ℝ :=
  sorry

theorem dihedral_angle_cosine (p : RegularHexagonalPyramid) (l : Line3D) :
  intersectsBaseSide p l ∧
  passesThroughCenters l (lateralFaceInscribedCircle p) (diagonalCrossSectionInscribedCircle p) →
  Real.cos (dihedralAngleAtBase p) = Real.sqrt (3 / 13) :=
  sorry

end NUMINAMATH_CALUDE_dihedral_angle_cosine_l1516_151633


namespace NUMINAMATH_CALUDE_bowling_team_average_weight_l1516_151668

theorem bowling_team_average_weight 
  (original_players : ℕ) 
  (new_player1_weight : ℕ) 
  (new_player2_weight : ℕ) 
  (new_average_weight : ℕ) 
  (h1 : original_players = 7)
  (h2 : new_player1_weight = 110)
  (h3 : new_player2_weight = 60)
  (h4 : new_average_weight = 78) :
  (original_players * (original_players + 2) * new_average_weight - 
   (original_players * new_player1_weight + original_players * new_player2_weight)) / 
  (original_players * original_players) = 76 :=
by sorry

end NUMINAMATH_CALUDE_bowling_team_average_weight_l1516_151668


namespace NUMINAMATH_CALUDE_president_vice_president_count_l1516_151680

/-- The number of ways to select a president and vice president from 5 people -/
def president_vice_president_selections : ℕ := 20

/-- The number of people to choose from -/
def total_people : ℕ := 5

/-- The number of positions to fill -/
def positions_to_fill : ℕ := 2

theorem president_vice_president_count :
  president_vice_president_selections = total_people * (total_people - 1) :=
sorry

end NUMINAMATH_CALUDE_president_vice_president_count_l1516_151680


namespace NUMINAMATH_CALUDE_max_child_fraction_is_11_20_l1516_151621

/-- Represents the babysitting scenario for Jane -/
structure BabysittingScenario where
  jane_start_age : ℕ
  jane_current_age : ℕ
  years_since_stopped : ℕ
  oldest_babysat_current_age : ℕ

/-- The maximum fraction of Jane's age that a child she babysat could be -/
def max_child_fraction (scenario : BabysittingScenario) : ℚ :=
  let jane_stop_age := scenario.jane_current_age - scenario.years_since_stopped
  let child_age_when_jane_stopped := scenario.oldest_babysat_current_age - scenario.years_since_stopped
  child_age_when_jane_stopped / jane_stop_age

/-- The theorem stating the maximum fraction of Jane's age a child could be -/
theorem max_child_fraction_is_11_20 (scenario : BabysittingScenario)
  (h1 : scenario.jane_start_age = 18)
  (h2 : scenario.jane_current_age = 32)
  (h3 : scenario.years_since_stopped = 12)
  (h4 : scenario.oldest_babysat_current_age = 23) :
  max_child_fraction scenario = 11/20 := by
  sorry

end NUMINAMATH_CALUDE_max_child_fraction_is_11_20_l1516_151621


namespace NUMINAMATH_CALUDE_certain_number_problem_l1516_151654

theorem certain_number_problem (y : ℝ) : 0.5 * 10 = 0.05 * y - 20 → y = 500 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_problem_l1516_151654


namespace NUMINAMATH_CALUDE_melanie_turnips_count_l1516_151664

/-- The number of turnips Benny grew -/
def benny_turnips : ℕ := 113

/-- The total number of turnips Melanie and Benny grew together -/
def total_turnips : ℕ := 252

/-- The number of turnips Melanie grew -/
def melanie_turnips : ℕ := total_turnips - benny_turnips

theorem melanie_turnips_count : melanie_turnips = 139 := by
  sorry

end NUMINAMATH_CALUDE_melanie_turnips_count_l1516_151664


namespace NUMINAMATH_CALUDE_chocolate_problem_l1516_151676

theorem chocolate_problem (total : ℕ) (eaten_with_nuts : ℚ) (left : ℕ) : 
  total = 80 →
  eaten_with_nuts = 4/5 →
  left = 28 →
  (total / 2 - (total / 2 * eaten_with_nuts) - (total - left)) / (total / 2) = 1/2 :=
by sorry

end NUMINAMATH_CALUDE_chocolate_problem_l1516_151676


namespace NUMINAMATH_CALUDE_candy_count_l1516_151674

/-- Given the total number of treats, chewing gums, and chocolate bars,
    prove that the number of candies of different flavors is 40. -/
theorem candy_count (total_treats chewing_gums chocolate_bars : ℕ) 
  (h1 : total_treats = 155)
  (h2 : chewing_gums = 60)
  (h3 : chocolate_bars = 55) :
  total_treats - (chewing_gums + chocolate_bars) = 40 := by
  sorry

end NUMINAMATH_CALUDE_candy_count_l1516_151674


namespace NUMINAMATH_CALUDE_millet_sunflower_exceed_half_on_tuesday_l1516_151613

/-- Represents the proportion of seeds in the feeder -/
structure SeedMix where
  millet : ℝ
  sunflower : ℝ
  other : ℝ

/-- Calculates the next day's seed mix based on consumption and refilling -/
def nextDayMix (mix : SeedMix) : SeedMix :=
  { millet := 0.2 + 0.75 * mix.millet,
    sunflower := 0.3 + 0.5 * mix.sunflower,
    other := 0.5 }

/-- The initial seed mix on Monday -/
def initialMix : SeedMix :=
  { millet := 0.2, sunflower := 0.3, other := 0.5 }

/-- Theorem: On Tuesday, millet and sunflower seeds combined exceed 50% of total seeds -/
theorem millet_sunflower_exceed_half_on_tuesday :
  let tuesdayMix := nextDayMix initialMix
  tuesdayMix.millet + tuesdayMix.sunflower > 0.5 := by
  sorry


end NUMINAMATH_CALUDE_millet_sunflower_exceed_half_on_tuesday_l1516_151613


namespace NUMINAMATH_CALUDE_tank_plastering_cost_l1516_151636

/-- Calculates the cost of plastering a tank's walls and bottom -/
def plasteringCost (length width depth : ℝ) (costPerSqMeter : ℝ) : ℝ :=
  let bottomArea := length * width
  let longWallsArea := 2 * (length * depth)
  let shortWallsArea := 2 * (width * depth)
  let totalArea := bottomArea + longWallsArea + shortWallsArea
  totalArea * costPerSqMeter

/-- Theorem stating the cost of plastering the given tank -/
theorem tank_plastering_cost :
  plasteringCost 25 12 6 0.25 = 186 := by
  sorry

end NUMINAMATH_CALUDE_tank_plastering_cost_l1516_151636


namespace NUMINAMATH_CALUDE_circle_tangent_point_relation_l1516_151610

/-- Given a circle C and a point A satisfying certain conditions, prove that a + (3/2)b = 3 -/
theorem circle_tangent_point_relation (a b : ℝ) : 
  (∃ (x y : ℝ), (x - 2)^2 + (y - 3)^2 = 1) →  -- Circle C equation
  (∃ (m_x m_y : ℝ), (m_x - 2)^2 + (m_y - 3)^2 = 1 ∧ 
    ((m_x - a) * (m_x - 2) + (m_y - b) * (m_y - 3) = 0)) →  -- AM is tangent to C at M
  ((a - 2)^2 + (b - 3)^2 - 1 = a^2 + b^2) →  -- |AM| = |AO|
  a + (3/2) * b = 3 := by
sorry

end NUMINAMATH_CALUDE_circle_tangent_point_relation_l1516_151610


namespace NUMINAMATH_CALUDE_binary_1010_is_10_l1516_151671

/-- Converts a binary digit (0 or 1) to its decimal value -/
def binaryToDecimal (digit : Nat) : Nat :=
  if digit = 0 ∨ digit = 1 then digit else 0

/-- Converts a list of binary digits to its decimal representation -/
def binaryListToDecimal (bits : List Nat) : Nat :=
  bits.enum.foldl (fun acc (i, b) => acc + binaryToDecimal b * 2^i) 0

theorem binary_1010_is_10 :
  binaryListToDecimal [0, 1, 0, 1] = 10 := by
  sorry

end NUMINAMATH_CALUDE_binary_1010_is_10_l1516_151671


namespace NUMINAMATH_CALUDE_valid_table_exists_l1516_151651

/-- Represents a geometric property of a shape -/
inductive Property
| HasAcuteAngle
| HasEqualSides
| Property3
| Property4

/-- Represents a geometric shape -/
inductive Shape
| Triangle1
| Triangle2
| Quadrilateral1
| Quadrilateral2

/-- A function that determines if a shape has a property -/
def hasProperty (s : Shape) (p : Property) : Bool :=
  match s, p with
  | Shape.Triangle1, Property.HasAcuteAngle => true
  | Shape.Triangle1, Property.HasEqualSides => false
  | Shape.Triangle2, Property.HasAcuteAngle => true
  | Shape.Triangle2, Property.HasEqualSides => true
  | Shape.Quadrilateral1, Property.HasAcuteAngle => false
  | Shape.Quadrilateral1, Property.HasEqualSides => false
  | Shape.Quadrilateral2, Property.HasAcuteAngle => true
  | Shape.Quadrilateral2, Property.HasEqualSides => false
  | _, _ => false  -- Default case for other combinations

/-- The main theorem stating the existence of a valid table -/
theorem valid_table_exists : ∃ (p3 p4 : Property),
  p3 ≠ Property.HasAcuteAngle ∧ p3 ≠ Property.HasEqualSides ∧
  p4 ≠ Property.HasAcuteAngle ∧ p4 ≠ Property.HasEqualSides ∧ p3 ≠ p4 ∧
  (∀ s : Shape, (hasProperty s Property.HasAcuteAngle).toNat +
                (hasProperty s Property.HasEqualSides).toNat +
                (hasProperty s p3).toNat +
                (hasProperty s p4).toNat = 3) ∧
  (∀ p : Property, (p = Property.HasAcuteAngle ∨ p = Property.HasEqualSides ∨ p = p3 ∨ p = p4) →
    (hasProperty Shape.Triangle1 p).toNat +
    (hasProperty Shape.Triangle2 p).toNat +
    (hasProperty Shape.Quadrilateral1 p).toNat +
    (hasProperty Shape.Quadrilateral2 p).toNat = 3) :=
by sorry

end NUMINAMATH_CALUDE_valid_table_exists_l1516_151651


namespace NUMINAMATH_CALUDE_walnut_trees_after_planting_l1516_151645

theorem walnut_trees_after_planting (initial_trees : ℕ) (new_trees : ℕ) :
  initial_trees = 4 → new_trees = 6 → initial_trees + new_trees = 10 := by
  sorry

end NUMINAMATH_CALUDE_walnut_trees_after_planting_l1516_151645


namespace NUMINAMATH_CALUDE_soda_consumption_theorem_l1516_151619

/-- The number of bottles of soda left after a given period -/
def bottles_left (bottles_per_pack : ℕ) (packs_bought : ℕ) (bottles_per_day : ℚ) (days : ℕ) : ℚ :=
  (bottles_per_pack * packs_bought : ℚ) - (bottles_per_day * days)

/-- Theorem stating that given the conditions, 4 bottles will be left after 4 weeks -/
theorem soda_consumption_theorem :
  bottles_left 6 3 (1/2) (4 * 7) = 4 := by
  sorry

end NUMINAMATH_CALUDE_soda_consumption_theorem_l1516_151619


namespace NUMINAMATH_CALUDE_perimeter_inequality_l1516_151630

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a triangle -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Calculates the perimeter of a triangle -/
def perimeter (t : Triangle) : ℝ := sorry

/-- Checks if a triangle is acute -/
def isAcute (t : Triangle) : Prop := sorry

/-- Calculates the foot of the perpendicular from a point to a line -/
def perpendicularFoot (p : Point) (l : Point × Point) : Point := sorry

/-- Main theorem -/
theorem perimeter_inequality 
  (ABC : Triangle) 
  (h_acute : isAcute ABC) 
  (D : Point) (E : Point) (F : Point)
  (P : Point) (Q : Point) (R : Point)
  (h_D : D = perpendicularFoot ABC.A (ABC.B, ABC.C))
  (h_E : E = perpendicularFoot ABC.B (ABC.C, ABC.A))
  (h_F : F = perpendicularFoot ABC.C (ABC.A, ABC.B))
  (h_P : P = perpendicularFoot ABC.A (E, F))
  (h_Q : Q = perpendicularFoot ABC.B (F, D))
  (h_R : R = perpendicularFoot ABC.C (D, E))
  : perimeter ABC * perimeter {A := P, B := Q, C := R} ≥ (perimeter {A := D, B := E, C := F})^2 := by
  sorry

end NUMINAMATH_CALUDE_perimeter_inequality_l1516_151630


namespace NUMINAMATH_CALUDE_sanya_washing_days_l1516_151601

/-- Represents the number of days needed to wash all towels -/
def days_needed (towels_per_wash : ℕ) (hours_per_day : ℕ) (total_towels : ℕ) : ℕ :=
  (total_towels + towels_per_wash * hours_per_day - 1) / (towels_per_wash * hours_per_day)

/-- Theorem stating that Sanya needs 7 days to wash all towels -/
theorem sanya_washing_days :
  days_needed 7 2 98 = 7 :=
by sorry

#eval days_needed 7 2 98

end NUMINAMATH_CALUDE_sanya_washing_days_l1516_151601


namespace NUMINAMATH_CALUDE_sum_of_fractions_geq_four_l1516_151627

theorem sum_of_fractions_geq_four (a b c d : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) : 
  (a * d + b * c) / (b * d) + (b * c + a * d) / (a * c) ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_geq_four_l1516_151627


namespace NUMINAMATH_CALUDE_equal_candies_after_sharing_l1516_151606

/-- Proves that Minyoung and Taehyung will have the same number of candies
    if Minyoung gives 3 candies to Taehyung. -/
theorem equal_candies_after_sharing (minyoung_initial : ℕ) (taehyung_initial : ℕ) 
  (candies_shared : ℕ) : 
  minyoung_initial = 9 →
  taehyung_initial = 3 →
  candies_shared = 3 →
  minyoung_initial - candies_shared = taehyung_initial + candies_shared :=
by
  sorry

#check equal_candies_after_sharing

end NUMINAMATH_CALUDE_equal_candies_after_sharing_l1516_151606


namespace NUMINAMATH_CALUDE_rectangular_hall_dimensions_l1516_151644

theorem rectangular_hall_dimensions (length width : ℝ) (area : ℝ) : 
  width = length / 2 →
  area = length * width →
  area = 200 →
  length - width = 10 := by
sorry

end NUMINAMATH_CALUDE_rectangular_hall_dimensions_l1516_151644


namespace NUMINAMATH_CALUDE_not_in_range_iff_b_in_interval_l1516_151686

/-- The function f(x) = x^2 + bx + 3 -/
def f (b : ℝ) (x : ℝ) : ℝ := x^2 + b*x + 3

/-- Theorem stating that -3 is not in the range of f(x) if and only if b is in the open interval (-2√6, 2√6) -/
theorem not_in_range_iff_b_in_interval (b : ℝ) :
  (∀ x, f b x ≠ -3) ↔ b ∈ Set.Ioo (-2 * Real.sqrt 6) (2 * Real.sqrt 6) := by
  sorry

end NUMINAMATH_CALUDE_not_in_range_iff_b_in_interval_l1516_151686


namespace NUMINAMATH_CALUDE_greatest_common_divisor_of_arithmetic_sum_l1516_151667

/-- The sum of the first n terms of an arithmetic sequence with first term a and common difference d -/
def arithmetic_sum (n : ℕ) (a d : ℤ) : ℤ := n * (2 * a + (n - 1) * d) / 2

/-- The sum of the first 12 terms of an arithmetic sequence with first term a and common difference d -/
def sum_12_terms (a d : ℤ) : ℤ := arithmetic_sum 12 a d

theorem greatest_common_divisor_of_arithmetic_sum :
  ∃ (k : ℕ), k > 0 ∧ 
  (∀ (a d : ℕ), (sum_12_terms a d).natAbs % k = 0) ∧
  (∀ (m : ℕ), m > k → ∃ (a d : ℕ), (sum_12_terms a d).natAbs % m ≠ 0) ∧
  k = 6 :=
sorry

end NUMINAMATH_CALUDE_greatest_common_divisor_of_arithmetic_sum_l1516_151667


namespace NUMINAMATH_CALUDE_max_value_of_d_l1516_151634

theorem max_value_of_d (a b c d : ℝ) 
  (sum_eq : a + b + c + d = 10) 
  (sum_products_eq : a*b + a*c + a*d + b*c + b*d + c*d = 20) : 
  d ≤ (5 + 5 * Real.sqrt 21) / 2 := by
sorry

end NUMINAMATH_CALUDE_max_value_of_d_l1516_151634
