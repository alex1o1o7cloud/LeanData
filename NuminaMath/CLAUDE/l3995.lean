import Mathlib

namespace NUMINAMATH_CALUDE_min_range_for_largest_angle_l3995_399548

-- Define the triangle sides as functions of x
def side_a (x : ℝ) := 2 * x
def side_b (x : ℝ) := x + 3
def side_c (x : ℝ) := x + 6

-- Define the triangle inequality conditions
def triangle_inequality (x : ℝ) : Prop :=
  side_a x + side_b x > side_c x ∧
  side_a x + side_c x > side_b x ∧
  side_b x + side_c x > side_a x

-- Define the condition for ∠A to be the largest angle
def angle_a_largest (x : ℝ) : Prop :=
  side_c x > side_a x ∧ side_c x > side_b x

-- Theorem stating the minimum range for x
theorem min_range_for_largest_angle :
  ∃ (m n : ℝ), m < n ∧
  (∀ x, m < x ∧ x < n → triangle_inequality x ∧ angle_a_largest x) ∧
  (∀ m' n', m' < n' →
    (∀ x, m' < x ∧ x < n' → triangle_inequality x ∧ angle_a_largest x) →
    n - m ≤ n' - m') ∧
  n - m = 3 := by
  sorry

end NUMINAMATH_CALUDE_min_range_for_largest_angle_l3995_399548


namespace NUMINAMATH_CALUDE_nine_ants_nine_trips_l3995_399503

/-- Represents the number of grains of rice that can be moved by a given number of ants in a given number of trips -/
def rice_moved (ants : ℕ) (trips : ℕ) : ℚ :=
  (24 : ℚ) * ants * trips / (12 * 6)

/-- Theorem stating that 9 ants can move 27 grains of rice in 9 trips -/
theorem nine_ants_nine_trips :
  rice_moved 9 9 = 27 := by
  sorry

end NUMINAMATH_CALUDE_nine_ants_nine_trips_l3995_399503


namespace NUMINAMATH_CALUDE_infinite_sum_evaluation_l3995_399572

theorem infinite_sum_evaluation : 
  (∑' n : ℕ, (n : ℝ) / ((n : ℝ)^4 + 4)) = 3/8 := by sorry

end NUMINAMATH_CALUDE_infinite_sum_evaluation_l3995_399572


namespace NUMINAMATH_CALUDE_kolakoski_next_eight_terms_l3995_399504

/-- The Kolakoski sequence -/
def kolakoski : ℕ → Fin 2
  | 0 => 0  -- represents 1
  | 1 => 1  -- represents 2
  | 2 => 1  -- represents 2
  | n + 3 => sorry

/-- The run-length encoding of the Kolakoski sequence -/
def kolakoski_rle : ℕ → Fin 2
  | n => kolakoski n

theorem kolakoski_next_eight_terms :
  (List.range 8).map (fun i => kolakoski (i + 12)) = [1, 1, 0, 0, 1, 0, 0, 1] := by
  sorry

#check kolakoski_next_eight_terms

end NUMINAMATH_CALUDE_kolakoski_next_eight_terms_l3995_399504


namespace NUMINAMATH_CALUDE_intersection_A_B_l3995_399587

def A : Set ℝ := {-3, -1, 0, 1, 2, 3, 4}
def B : Set ℝ := {x : ℝ | 0 ≤ x ∧ x ≤ 3}

theorem intersection_A_B : A ∩ B = {0, 1, 2, 3} := by
  sorry

end NUMINAMATH_CALUDE_intersection_A_B_l3995_399587


namespace NUMINAMATH_CALUDE_opposite_of_negative_2023_l3995_399575

theorem opposite_of_negative_2023 : -((-2023 : ℤ)) = 2023 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_negative_2023_l3995_399575


namespace NUMINAMATH_CALUDE_cubic_function_extremum_value_l3995_399534

/-- A cubic function with an extremum at x = 1 and f(1) = 10 -/
def f (a b : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + b*x + a

/-- The derivative of f -/
def f' (a b : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*a*x + b

theorem cubic_function_extremum_value (a b : ℝ) :
  f' a b 1 = 0 ∧ f a b 1 = 10 → f a b 2 = 18 := by
  sorry

end NUMINAMATH_CALUDE_cubic_function_extremum_value_l3995_399534


namespace NUMINAMATH_CALUDE_theo_selling_price_l3995_399544

/-- Represents the selling price and profit for a camera seller --/
structure CameraSeller where
  buyPrice : ℕ
  sellPrice : ℕ
  quantity : ℕ

/-- Calculates the total profit for a camera seller --/
def totalProfit (seller : CameraSeller) : ℕ :=
  (seller.sellPrice - seller.buyPrice) * seller.quantity

theorem theo_selling_price 
  (maddox theo : CameraSeller)
  (h1 : maddox.buyPrice = 20)
  (h2 : theo.buyPrice = 20)
  (h3 : maddox.quantity = 3)
  (h4 : theo.quantity = 3)
  (h5 : maddox.sellPrice = 28)
  (h6 : totalProfit maddox = totalProfit theo + 15) :
  theo.sellPrice = 23 := by
sorry

end NUMINAMATH_CALUDE_theo_selling_price_l3995_399544


namespace NUMINAMATH_CALUDE_china_population_scientific_notation_l3995_399569

/-- Represents the population of China in millions at the end of 2021 -/
def china_population : ℝ := 1412.60

/-- Proves that the population of China expressed in scientific notation is 1.4126 × 10^9 -/
theorem china_population_scientific_notation :
  (china_population * 1000000 : ℝ) = 1.4126 * (10 ^ 9) := by
  sorry

end NUMINAMATH_CALUDE_china_population_scientific_notation_l3995_399569


namespace NUMINAMATH_CALUDE_smallest_angle_measure_l3995_399508

-- Define a right triangle with acute angles in the ratio 3:2
structure RightTriangle where
  angle1 : ℝ
  angle2 : ℝ
  right_angle : ℝ
  is_right_triangle : right_angle = 90
  acute_angle_sum : angle1 + angle2 = 90
  angle_ratio : angle1 / angle2 = 3 / 2

-- Theorem statement
theorem smallest_angle_measure (t : RightTriangle) : 
  min t.angle1 t.angle2 = 36 := by
  sorry

end NUMINAMATH_CALUDE_smallest_angle_measure_l3995_399508


namespace NUMINAMATH_CALUDE_steve_socks_l3995_399547

theorem steve_socks (total_socks : ℕ) (matching_pairs : ℕ) (mismatching_socks : ℕ) : 
  total_socks = 25 → matching_pairs = 4 → mismatching_socks = total_socks - 2 * matching_pairs →
  mismatching_socks = 17 := by
sorry

end NUMINAMATH_CALUDE_steve_socks_l3995_399547


namespace NUMINAMATH_CALUDE_max_sum_xyz_l3995_399545

theorem max_sum_xyz (x y z c : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0) (hc : c > 0)
  (h1 : x + c * y ≤ 36) (h2 : 2 * x + 3 * z ≤ 72) :
  (c ≥ 3 → ∃ (x' y' z' : ℝ), x' ≥ 0 ∧ y' ≥ 0 ∧ z' ≥ 0 ∧
    x' + c * y' ≤ 36 ∧ 2 * x' + 3 * z' ≤ 72 ∧ x' + y' + z' = 36 ∧
    ∀ (a b d : ℝ), a ≥ 0 → b ≥ 0 → d ≥ 0 → a + c * b ≤ 36 → 2 * a + 3 * d ≤ 72 → a + b + d ≤ 36) ∧
  (c < 3 → ∃ (x' y' z' : ℝ), x' ≥ 0 ∧ y' ≥ 0 ∧ z' ≥ 0 ∧
    x' + c * y' ≤ 36 ∧ 2 * x' + 3 * z' ≤ 72 ∧ x' + y' + z' = 24 + 36 / c ∧
    ∀ (a b d : ℝ), a ≥ 0 → b ≥ 0 → d ≥ 0 → a + c * b ≤ 36 → 2 * a + 3 * d ≤ 72 → a + b + d ≤ 24 + 36 / c) :=
by sorry

end NUMINAMATH_CALUDE_max_sum_xyz_l3995_399545


namespace NUMINAMATH_CALUDE_probability_is_two_thirds_l3995_399585

structure Diagram where
  total_triangles : ℕ
  triangles_with_G : ℕ
  equal_probability : Bool

def probability_including_G (d : Diagram) : ℚ :=
  d.triangles_with_G / d.total_triangles

theorem probability_is_two_thirds (d : Diagram) 
  (h1 : d.total_triangles = 6)
  (h2 : d.triangles_with_G = 4)
  (h3 : d.equal_probability = true) :
  probability_including_G d = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_probability_is_two_thirds_l3995_399585


namespace NUMINAMATH_CALUDE_companion_pair_example_companion_pair_value_companion_pair_expression_l3995_399589

/-- Definition of companion rational number pairs -/
def is_companion_pair (a b : ℚ) : Prop := a - b = a * b + 1

/-- Theorem 1: (-1/2, -3) is a companion rational number pair -/
theorem companion_pair_example : is_companion_pair (-1/2) (-3) := by sorry

/-- Theorem 2: When (x+1, 5) is a companion rational number pair, x = -5/2 -/
theorem companion_pair_value (x : ℚ) : 
  is_companion_pair (x + 1) 5 → x = -5/2 := by sorry

/-- Theorem 3: For any companion rational number pair (a,b), 
    3ab-a+1/2(a+b-5ab)+1 = 1/2 -/
theorem companion_pair_expression (a b : ℚ) :
  is_companion_pair a b → 3*a*b - a + 1/2*(a+b-5*a*b) + 1 = 1/2 := by sorry

end NUMINAMATH_CALUDE_companion_pair_example_companion_pair_value_companion_pair_expression_l3995_399589


namespace NUMINAMATH_CALUDE_jays_savings_l3995_399529

def savings_sequence (n : ℕ) : ℕ := 20 + 10 * n

def total_savings (weeks : ℕ) : ℕ :=
  (List.range weeks).map savings_sequence |> List.sum

theorem jays_savings : total_savings 4 = 140 := by
  sorry

end NUMINAMATH_CALUDE_jays_savings_l3995_399529


namespace NUMINAMATH_CALUDE_chocolate_sales_l3995_399530

theorem chocolate_sales (cost_price selling_price : ℝ) (N : ℕ) : 
  (121 * cost_price = N * selling_price) →
  (selling_price = cost_price * (1 + 57.142857142857146 / 100)) →
  N = 77 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_sales_l3995_399530


namespace NUMINAMATH_CALUDE_slope_condition_l3995_399528

/-- The slope of a line with y-intercept (0, 8) that intersects the ellipse 4x^2 + 25y^2 = 100 -/
def slope_intersecting_line_ellipse (m : ℝ) : Prop :=
  ∃ x y : ℝ, 
    y = m * x + 8 ∧ 
    4 * x^2 + 25 * y^2 = 100

/-- Theorem stating the condition for the slope of the intersecting line -/
theorem slope_condition : 
  ∀ m : ℝ, slope_intersecting_line_ellipse m ↔ m^2 ≥ 3/77 :=
by sorry

end NUMINAMATH_CALUDE_slope_condition_l3995_399528


namespace NUMINAMATH_CALUDE_abs_neg_five_eq_five_l3995_399584

theorem abs_neg_five_eq_five : |(-5 : ℤ)| = 5 := by
  sorry

end NUMINAMATH_CALUDE_abs_neg_five_eq_five_l3995_399584


namespace NUMINAMATH_CALUDE_division_problem_l3995_399574

theorem division_problem : (150 : ℚ) / ((6 : ℚ) / 3) = 75 := by sorry

end NUMINAMATH_CALUDE_division_problem_l3995_399574


namespace NUMINAMATH_CALUDE_sum_bound_l3995_399532

theorem sum_bound (a b c : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) (h_sum : a + b + c = 1) :
  let S := 1 / (1 + a) + 1 / (1 + b) + 1 / (1 + c)
  9 / 4 ≤ S ∧ S ≤ 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_bound_l3995_399532


namespace NUMINAMATH_CALUDE_consecutive_integers_square_sum_l3995_399502

theorem consecutive_integers_square_sum (a b : ℤ) (h : b = a + 1) :
  a^2 + b^2 + (a*b)^2 = (a*b + 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_square_sum_l3995_399502


namespace NUMINAMATH_CALUDE_board_division_impossibility_l3995_399516

theorem board_division_impossibility : ¬ ∃ (triangle_area : ℚ),
  (63 : ℚ) = 17 * triangle_area ∧
  ∃ (side_length : ℚ), 
    triangle_area = (side_length * side_length * Real.sqrt 3) / 4 ∧
    0 < side_length ∧
    side_length ≤ 8 := by
  sorry

end NUMINAMATH_CALUDE_board_division_impossibility_l3995_399516


namespace NUMINAMATH_CALUDE_canadian_scientist_ratio_l3995_399598

/-- Proves that the ratio of Canadian scientists to total scientists is 1:5 -/
theorem canadian_scientist_ratio (total : ℕ) (usa : ℕ) : 
  total = 70 → 
  usa = 21 → 
  (total - (total / 2) - usa) / total = 1 / 5 := by
sorry

end NUMINAMATH_CALUDE_canadian_scientist_ratio_l3995_399598


namespace NUMINAMATH_CALUDE_monomial_sum_condition_l3995_399579

theorem monomial_sum_condition (a b : ℝ) (m n : ℕ) :
  (∃ k : ℝ, ∃ p q : ℕ, 2 * a^(m+2) * b^(2*n+2) + a^3 * b^8 = k * a^p * b^q) →
  m = 1 ∧ n = 3 := by
  sorry

end NUMINAMATH_CALUDE_monomial_sum_condition_l3995_399579


namespace NUMINAMATH_CALUDE_arithmetic_sequence_2011_l3995_399500

def arithmetic_sequence (a₁ d n : ℕ) : ℕ := a₁ + (n - 1) * d

theorem arithmetic_sequence_2011 :
  arithmetic_sequence 1 3 671 = 2011 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_2011_l3995_399500


namespace NUMINAMATH_CALUDE_solve_equation_l3995_399523

theorem solve_equation (x : ℝ) (h : 0.009 / x = 0.05) : x = 0.18 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l3995_399523


namespace NUMINAMATH_CALUDE_equation_three_roots_l3995_399567

theorem equation_three_roots (m : ℝ) : 
  (∃ (x₁ x₂ x₃ : ℝ), x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧
    (x₁^2 - 2*|x₁| + 2 = m) ∧ 
    (x₂^2 - 2*|x₂| + 2 = m) ∧ 
    (x₃^2 - 2*|x₃| + 2 = m) ∧
    (∀ (x : ℝ), x^2 - 2*|x| + 2 = m → x = x₁ ∨ x = x₂ ∨ x = x₃)) ↔ 
  m = 2 := by
  sorry

end NUMINAMATH_CALUDE_equation_three_roots_l3995_399567


namespace NUMINAMATH_CALUDE_excess_meat_sold_proof_l3995_399559

/-- Calculates the excess meat sold beyond the original plan. -/
def excess_meat_sold (thursday_sales : ℕ) (saturday_sales : ℕ) (original_plan : ℕ) : ℕ :=
  let friday_sales := 2 * thursday_sales
  let sunday_sales := saturday_sales / 2
  let total_sales := thursday_sales + friday_sales + saturday_sales + sunday_sales
  total_sales - original_plan

/-- Proves that the excess meat sold beyond the original plan is 325 kg. -/
theorem excess_meat_sold_proof :
  excess_meat_sold 210 130 500 = 325 := by
  sorry

end NUMINAMATH_CALUDE_excess_meat_sold_proof_l3995_399559


namespace NUMINAMATH_CALUDE_soap_promotion_theorem_l3995_399525

/-- The original price of soap in yuan -/
def original_price : ℝ := 2

/-- The cost of buying n pieces of soap under Promotion 1 -/
def promotion1_cost (n : ℕ) : ℝ :=
  original_price + 0.7 * original_price * (n - 1 : ℝ)

/-- The cost of buying n pieces of soap under Promotion 2 -/
def promotion2_cost (n : ℕ) : ℝ :=
  0.8 * original_price * n

/-- The minimum number of soap pieces for Promotion 1 to be cheaper than Promotion 2 -/
def min_pieces : ℕ := 4

theorem soap_promotion_theorem :
  ∀ n : ℕ, n ≥ min_pieces →
    promotion1_cost n < promotion2_cost n ∧
    ∀ m : ℕ, m < min_pieces → promotion1_cost m ≥ promotion2_cost m :=
by sorry

end NUMINAMATH_CALUDE_soap_promotion_theorem_l3995_399525


namespace NUMINAMATH_CALUDE_business_class_seats_count_l3995_399586

/-- A small airplane with first, business, and economy class seating. -/
structure Airplane where
  first_class_seats : ℕ
  business_class_seats : ℕ
  economy_class_seats : ℕ

/-- Theorem stating the number of business class seats in the airplane. -/
theorem business_class_seats_count (a : Airplane) 
  (h1 : a.first_class_seats = 10)
  (h2 : a.economy_class_seats = 50)
  (h3 : a.economy_class_seats / 2 = a.first_class_seats + (a.business_class_seats - 8))
  (h4 : a.first_class_seats - 7 = 3) : 
  a.business_class_seats = 30 := by
  sorry


end NUMINAMATH_CALUDE_business_class_seats_count_l3995_399586


namespace NUMINAMATH_CALUDE_vehicle_speeds_and_distance_l3995_399542

theorem vehicle_speeds_and_distance (total_distance : ℝ) 
  (speed_ratio : ℝ) (time_delay : ℝ) :
  total_distance = 90 →
  speed_ratio = 1.5 →
  time_delay = 1/3 →
  ∃ (speed_slow speed_fast distance_traveled : ℝ),
    speed_slow = 90 ∧
    speed_fast = 135 ∧
    distance_traveled = 30 ∧
    speed_fast = speed_ratio * speed_slow ∧
    total_distance / speed_slow - total_distance / speed_fast = time_delay ∧
    distance_traveled = speed_slow * time_delay :=
by sorry

end NUMINAMATH_CALUDE_vehicle_speeds_and_distance_l3995_399542


namespace NUMINAMATH_CALUDE_brother_age_l3995_399566

theorem brother_age (man_age brother_age : ℕ) : 
  man_age = brother_age + 12 →
  man_age + 2 = 2 * (brother_age + 2) →
  brother_age = 10 := by
sorry

end NUMINAMATH_CALUDE_brother_age_l3995_399566


namespace NUMINAMATH_CALUDE_probability_spade_heart_spade_l3995_399563

/-- A standard deck of cards. -/
def StandardDeck : ℕ := 52

/-- The number of cards of each suit in a standard deck. -/
def CardsPerSuit : ℕ := 13

/-- The probability of drawing ♠, ♥, ♠ in sequence from a standard deck. -/
def ProbabilitySpadeHeartSpade : ℚ :=
  (CardsPerSuit : ℚ) / StandardDeck *
  (CardsPerSuit : ℚ) / (StandardDeck - 1) *
  (CardsPerSuit - 1 : ℚ) / (StandardDeck - 2)

theorem probability_spade_heart_spade :
  ProbabilitySpadeHeartSpade = 78 / 5100 := by
  sorry

end NUMINAMATH_CALUDE_probability_spade_heart_spade_l3995_399563


namespace NUMINAMATH_CALUDE_seating_problem_l3995_399570

/-- The number of ways to seat people on a bench with given constraints -/
def seating_arrangements (total_seats : ℕ) (people : ℕ) (min_gap : ℕ) : ℕ :=
  -- Definition to be implemented
  sorry

/-- Theorem stating the correct number of seating arrangements for the given problem -/
theorem seating_problem : seating_arrangements 9 3 2 = 60 := by
  sorry

end NUMINAMATH_CALUDE_seating_problem_l3995_399570


namespace NUMINAMATH_CALUDE_interest_rate_proof_l3995_399581

/-- Proves that the interest rate is 5% given the specified loan conditions -/
theorem interest_rate_proof (principal : ℝ) (time : ℝ) (interest : ℝ) :
  principal = 3000 →
  time = 5 →
  interest = principal - 2250 →
  (interest * 100) / (principal * time) = 5 := by
  sorry

end NUMINAMATH_CALUDE_interest_rate_proof_l3995_399581


namespace NUMINAMATH_CALUDE_two_numbers_problem_l3995_399546

theorem two_numbers_problem (x y z : ℝ) 
  (h1 : x > y) 
  (h2 : x + y = 90) 
  (h3 : x - y = 15) 
  (h4 : z = x^2 - y^2) : 
  z = 1350 := by
  sorry

end NUMINAMATH_CALUDE_two_numbers_problem_l3995_399546


namespace NUMINAMATH_CALUDE_total_leaves_on_farm_l3995_399590

/-- Calculate the total number of leaves on all trees on a farm --/
theorem total_leaves_on_farm (
  num_trees : ℕ)
  (branches_per_tree : ℕ)
  (sub_branches_per_branch : ℕ)
  (leaves_per_sub_branch : ℕ)
  (h1 : num_trees = 4)
  (h2 : branches_per_tree = 10)
  (h3 : sub_branches_per_branch = 40)
  (h4 : leaves_per_sub_branch = 60)
  : num_trees * branches_per_tree * sub_branches_per_branch * leaves_per_sub_branch = 96000 := by
  sorry

#check total_leaves_on_farm

end NUMINAMATH_CALUDE_total_leaves_on_farm_l3995_399590


namespace NUMINAMATH_CALUDE_triangle_angle_calculation_l3995_399511

theorem triangle_angle_calculation (A B C : ℝ) (a b c : ℝ) : 
  C = π / 4 →  -- 45° in radians
  c = Real.sqrt 2 → 
  a = Real.sqrt 3 → 
  (A = π / 3 ∨ A = 2 * π / 3) -- 60° or 120° in radians
  :=
by sorry

end NUMINAMATH_CALUDE_triangle_angle_calculation_l3995_399511


namespace NUMINAMATH_CALUDE_appliance_price_difference_l3995_399517

theorem appliance_price_difference : 
  let in_store_price : ℚ := 109.99
  let tv_payment : ℚ := 24.99
  let tv_shipping : ℚ := 14.98
  let tv_price : ℚ := 4 * tv_payment + tv_shipping
  (tv_price - in_store_price) * 100 = 495 := by sorry

end NUMINAMATH_CALUDE_appliance_price_difference_l3995_399517


namespace NUMINAMATH_CALUDE_parallel_line_through_point_l3995_399526

-- Define the slope of the given line
def m : ℚ := 1/2

-- Define the given line
def given_line (x : ℚ) : ℚ := m * x - 1

-- Define the point that the new line passes through
def point : ℚ × ℚ := (1, 0)

-- Define the equation of the new line
def new_line (x : ℚ) : ℚ := m * x - 1/2

theorem parallel_line_through_point :
  (∀ x, new_line x - new_line point.1 = m * (x - point.1)) ∧
  new_line point.1 = point.2 ∧
  ∀ x, new_line x - given_line x = new_line 0 - given_line 0 :=
by sorry

end NUMINAMATH_CALUDE_parallel_line_through_point_l3995_399526


namespace NUMINAMATH_CALUDE_tangent_perpendicular_implies_a_value_l3995_399571

noncomputable def curve (a : ℝ) (x : ℝ) : ℝ := a^x + 1

theorem tangent_perpendicular_implies_a_value (a : ℝ) 
  (h1 : a > 0) 
  (h2 : a ≠ 1) 
  (h3 : let tangent_slope := (Real.log a)
        let perpendicular_line_slope := -1/2
        tangent_slope * perpendicular_line_slope = -1) :
  a = Real.exp 2 := by sorry

end NUMINAMATH_CALUDE_tangent_perpendicular_implies_a_value_l3995_399571


namespace NUMINAMATH_CALUDE_angle_slope_relationship_l3995_399597

theorem angle_slope_relationship (α k : ℝ) :
  (k = Real.tan α) →
  (α < π / 3 → k < Real.sqrt 3) ∧
  ¬(k < Real.sqrt 3 → α < π / 3) :=
sorry

end NUMINAMATH_CALUDE_angle_slope_relationship_l3995_399597


namespace NUMINAMATH_CALUDE_some_athletes_not_honor_society_l3995_399596

-- Define the universe of discourse
variable (U : Type)

-- Define predicates
variable (Athlete : U → Prop)
variable (Disciplined : U → Prop)
variable (HonorSocietyMember : U → Prop)

-- Define the given conditions
variable (some_athletes_not_disciplined : ∃ x, Athlete x ∧ ¬Disciplined x)
variable (all_honor_society_disciplined : ∀ x, HonorSocietyMember x → Disciplined x)

-- State the theorem
theorem some_athletes_not_honor_society :
  ∃ x, Athlete x ∧ ¬HonorSocietyMember x :=
sorry

end NUMINAMATH_CALUDE_some_athletes_not_honor_society_l3995_399596


namespace NUMINAMATH_CALUDE_banana_arrangement_count_l3995_399599

/-- The number of distinct arrangements of the letters in BANANA -/
def banana_arrangements : ℕ := 60

/-- The total number of letters in BANANA -/
def total_letters : ℕ := 6

/-- The number of occurrences of the letter A in BANANA -/
def num_a : ℕ := 3

/-- The number of occurrences of the letter N in BANANA -/
def num_n : ℕ := 2

/-- The number of occurrences of the letter B in BANANA -/
def num_b : ℕ := 1

theorem banana_arrangement_count :
  banana_arrangements = (Nat.factorial total_letters) / ((Nat.factorial num_a) * (Nat.factorial num_n)) :=
by sorry

end NUMINAMATH_CALUDE_banana_arrangement_count_l3995_399599


namespace NUMINAMATH_CALUDE_prob_one_to_three_l3995_399557

/-- A random variable with normal distribution -/
structure NormalRV where
  μ : ℝ
  σ : ℝ
  pos : σ > 0

/-- Probability function for a normal random variable -/
noncomputable def prob (X : NormalRV) (a b : ℝ) : ℝ := sorry

/-- The given property of normal distributions -/
axiom normal_prob_property (X : NormalRV) : 
  prob X (X.μ - X.σ) (X.μ + X.σ) = 0.6826

/-- The specific normal distribution N(1, 4) -/
def X : NormalRV := { μ := 1, σ := 2, pos := by norm_num }

/-- The theorem to prove -/
theorem prob_one_to_three : prob X 1 3 = 0.3413 := by sorry

end NUMINAMATH_CALUDE_prob_one_to_three_l3995_399557


namespace NUMINAMATH_CALUDE_smallest_value_of_complex_expression_l3995_399593

theorem smallest_value_of_complex_expression (a b c d : ℤ) (ω : ℂ) (ζ : ℂ) :
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  ω^4 = 1 →
  ω ≠ 1 →
  ζ = ω^2 →
  ∃ (x y z w : ℤ), x ≠ y ∧ x ≠ z ∧ x ≠ w ∧ y ≠ z ∧ y ≠ w ∧ z ≠ w ∧
    |Complex.abs (↑x + ↑y * ω + ↑z * ζ + ↑w * ω^3)| = Real.sqrt 2 ∧
    ∀ (p q r s : ℤ), p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s →
      |Complex.abs (↑p + ↑q * ω + ↑r * ζ + ↑s * ω^3)| ≥ Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_smallest_value_of_complex_expression_l3995_399593


namespace NUMINAMATH_CALUDE_bee_hive_problem_l3995_399554

theorem bee_hive_problem (B : ℕ) : 
  (B / 5 : ℚ) + (B / 3 : ℚ) + (3 * ((B / 3 : ℚ) - (B / 5 : ℚ))) + 1 = B → B = 15 := by
  sorry

end NUMINAMATH_CALUDE_bee_hive_problem_l3995_399554


namespace NUMINAMATH_CALUDE_inequality_solution_range_l3995_399518

theorem inequality_solution_range (a : ℝ) : 
  (∃ x : ℝ, |x - 1| - |x - 3| > a) → a < 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_range_l3995_399518


namespace NUMINAMATH_CALUDE_bank_account_balance_l3995_399577

theorem bank_account_balance 
  (transferred_amount : ℕ) 
  (remaining_balance : ℕ) 
  (original_balance : ℕ) : 
  transferred_amount = 69 → 
  remaining_balance = 26935 → 
  original_balance = remaining_balance + transferred_amount → 
  original_balance = 27004 := by
  sorry

end NUMINAMATH_CALUDE_bank_account_balance_l3995_399577


namespace NUMINAMATH_CALUDE_skating_minutes_tenth_day_l3995_399562

def minutes_per_day_first_5 : ℕ := 75
def days_first_period : ℕ := 5
def minutes_per_day_next_3 : ℕ := 120
def days_second_period : ℕ := 3
def total_days : ℕ := 10
def target_average : ℕ := 95

theorem skating_minutes_tenth_day : 
  ∃ (x : ℕ), 
    (minutes_per_day_first_5 * days_first_period + 
     minutes_per_day_next_3 * days_second_period + x) / total_days = target_average ∧
    x = 215 := by
  sorry

end NUMINAMATH_CALUDE_skating_minutes_tenth_day_l3995_399562


namespace NUMINAMATH_CALUDE_min_values_xy_l3995_399573

/-- Given positive real numbers x and y satisfying lg x + lg y = lg(x + y + 3),
    prove that the minimum value of xy is 9 and the minimum value of x + y is 6. -/
theorem min_values_xy (x y : ℝ) (hx : x > 0) (hy : y > 0)
  (h : Real.log x + Real.log y = Real.log (x + y + 3)) :
  (∀ a b, a > 0 → b > 0 → Real.log a + Real.log b = Real.log (a + b + 3) → x * y ≤ a * b) ∧
  (∀ a b, a > 0 → b > 0 → Real.log a + Real.log b = Real.log (a + b + 3) → x + y ≤ a + b) ∧
  x * y = 9 ∧ x + y = 6 :=
sorry

end NUMINAMATH_CALUDE_min_values_xy_l3995_399573


namespace NUMINAMATH_CALUDE_operation_result_l3995_399519

def operation (a b : ℝ) : ℝ := a * (b ^ (1/2))

theorem operation_result :
  ∀ x : ℝ, operation x 9 = 12 → x = 4 := by
sorry

end NUMINAMATH_CALUDE_operation_result_l3995_399519


namespace NUMINAMATH_CALUDE_product_remainder_l3995_399539

theorem product_remainder (a b c : ℕ) (ha : a = 2456) (hb : b = 8743) (hc : c = 92431) :
  (a * b * c) % 10 = 8 := by
  sorry

end NUMINAMATH_CALUDE_product_remainder_l3995_399539


namespace NUMINAMATH_CALUDE_book_ratio_is_three_l3995_399524

/-- The number of books read last week -/
def books_last_week : ℕ := 5

/-- The number of pages in each book -/
def pages_per_book : ℕ := 300

/-- The total number of pages read this week -/
def pages_this_week : ℕ := 4500

/-- The ratio of books read this week to books read last week -/
def book_ratio : ℚ := (pages_this_week / pages_per_book) / books_last_week

theorem book_ratio_is_three : book_ratio = 3 := by
  sorry

end NUMINAMATH_CALUDE_book_ratio_is_three_l3995_399524


namespace NUMINAMATH_CALUDE_larger_number_proof_l3995_399550

theorem larger_number_proof (S L : ℤ) : 
  L = 4 * (S + 30) → L - S = 480 → L = 600 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_proof_l3995_399550


namespace NUMINAMATH_CALUDE_no_infinite_sequence_with_sqrt_difference_l3995_399509

theorem no_infinite_sequence_with_sqrt_difference :
  ¬∃ (x : ℕ → ℝ), 
    (∀ n, x n > 0) ∧ 
    (∀ n, x (n + 2) = Real.sqrt (x (n + 1)) - Real.sqrt (x n)) := by
  sorry

end NUMINAMATH_CALUDE_no_infinite_sequence_with_sqrt_difference_l3995_399509


namespace NUMINAMATH_CALUDE_at_least_one_product_contains_seven_l3995_399556

def containsSeven (m : Nat) : Bool :=
  let digits := m.digits 10
  7 ∈ digits

theorem at_least_one_product_contains_seven (n : Nat) (hn : n > 0) :
  ∃ k : Nat, k ≤ 35 ∧ k > 0 ∧ containsSeven (k * n) := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_product_contains_seven_l3995_399556


namespace NUMINAMATH_CALUDE_sarah_pencils_count_l3995_399565

/-- The number of pencils Sarah buys on Monday -/
def monday_pencils : ℕ := 20

/-- The number of pencils Sarah buys on Tuesday -/
def tuesday_pencils : ℕ := 18

/-- The number of pencils Sarah buys on Wednesday -/
def wednesday_pencils : ℕ := 3 * tuesday_pencils

/-- The total number of pencils Sarah has -/
def total_pencils : ℕ := monday_pencils + tuesday_pencils + wednesday_pencils

theorem sarah_pencils_count : total_pencils = 92 := by
  sorry

end NUMINAMATH_CALUDE_sarah_pencils_count_l3995_399565


namespace NUMINAMATH_CALUDE_expression_evaluation_l3995_399564

theorem expression_evaluation :
  let x : ℕ := 3
  let y : ℕ := 4
  5 * x^(y+1) + 6 * y^(x+1) = 2751 := by
sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3995_399564


namespace NUMINAMATH_CALUDE_fault_line_movement_l3995_399583

/-- The movement of a fault line over two years -/
theorem fault_line_movement 
  (movement_past_year : ℝ) 
  (movement_year_before : ℝ) 
  (h1 : movement_past_year = 1.25)
  (h2 : movement_year_before = 5.25) : 
  movement_past_year + movement_year_before = 6.50 := by
sorry

end NUMINAMATH_CALUDE_fault_line_movement_l3995_399583


namespace NUMINAMATH_CALUDE_count_pairs_satisfying_condition_l3995_399568

theorem count_pairs_satisfying_condition : 
  (Finset.filter (fun p : ℕ × ℕ => p.1^2 + p.2 < 50 ∧ p.1 > 0 ∧ p.2 > 0) (Finset.product (Finset.range 50) (Finset.range 50))).card = 204 := by
  sorry

end NUMINAMATH_CALUDE_count_pairs_satisfying_condition_l3995_399568


namespace NUMINAMATH_CALUDE_some_fire_breathing_mystical_l3995_399535

-- Define the sets
variable (U : Type) -- Universe set
variable (Dragon MysticalCreature FireBreathingCreature : Set U)

-- Define the conditions
variable (h1 : Dragon ⊆ FireBreathingCreature)
variable (h2 : ∃ x, x ∈ MysticalCreature ∧ x ∈ Dragon)

-- Theorem to prove
theorem some_fire_breathing_mystical :
  ∃ x, x ∈ FireBreathingCreature ∧ x ∈ MysticalCreature :=
by
  sorry


end NUMINAMATH_CALUDE_some_fire_breathing_mystical_l3995_399535


namespace NUMINAMATH_CALUDE_alexanders_galleries_l3995_399536

/-- Represents the problem of calculating the number of new galleries Alexander drew for. -/
theorem alexanders_galleries
  (first_gallery_pictures : ℕ)
  (new_gallery_pictures : ℕ)
  (pencils_per_picture : ℕ)
  (signing_pencils : ℕ)
  (total_pencils : ℕ)
  (h1 : first_gallery_pictures = 9)
  (h2 : new_gallery_pictures = 2)
  (h3 : pencils_per_picture = 4)
  (h4 : signing_pencils = 2)
  (h5 : total_pencils = 88) :
  (total_pencils - (first_gallery_pictures * pencils_per_picture + signing_pencils)) / 
  (new_gallery_pictures * pencils_per_picture + signing_pencils) = 5 := by
  sorry


end NUMINAMATH_CALUDE_alexanders_galleries_l3995_399536


namespace NUMINAMATH_CALUDE_johnny_planks_needed_l3995_399514

/-- Calculates the number of planks needed to build tables. -/
def planks_needed (num_tables : ℕ) (planks_per_leg : ℕ) (legs_per_table : ℕ) (planks_for_surface : ℕ) : ℕ :=
  num_tables * (legs_per_table * planks_per_leg + planks_for_surface)

/-- Theorem: Johnny needs 45 planks to build 5 tables. -/
theorem johnny_planks_needed : 
  planks_needed 5 1 4 5 = 45 := by
sorry

end NUMINAMATH_CALUDE_johnny_planks_needed_l3995_399514


namespace NUMINAMATH_CALUDE_complex_on_real_axis_l3995_399561

theorem complex_on_real_axis (a b : ℝ) : 
  (Complex.ofReal a + Complex.I * b).im = 0 → b = 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_on_real_axis_l3995_399561


namespace NUMINAMATH_CALUDE_decagon_diagonals_l3995_399540

/-- The number of diagonals in a convex polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A decagon has 10 sides -/
def decagon_sides : ℕ := 10

theorem decagon_diagonals : num_diagonals decagon_sides = 35 := by
  sorry

end NUMINAMATH_CALUDE_decagon_diagonals_l3995_399540


namespace NUMINAMATH_CALUDE_f_has_at_most_two_zeros_l3995_399505

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - 12*x + a

-- State the theorem
theorem f_has_at_most_two_zeros (a : ℝ) (h : a ≥ 16) :
  ∃ (z₁ z₂ : ℝ), ∀ x : ℝ, f a x = 0 → x = z₁ ∨ x = z₂ :=
sorry

end NUMINAMATH_CALUDE_f_has_at_most_two_zeros_l3995_399505


namespace NUMINAMATH_CALUDE_ball_hitting_ground_time_l3995_399543

theorem ball_hitting_ground_time : 
  let f (t : ℝ) := -20 * t^2 + 30 * t + 60
  ∃ t : ℝ, t > 0 ∧ f t = 0 ∧ t = (3 + Real.sqrt 57) / 4 := by
  sorry

end NUMINAMATH_CALUDE_ball_hitting_ground_time_l3995_399543


namespace NUMINAMATH_CALUDE_exponent_and_square_of_negative_two_l3995_399588

theorem exponent_and_square_of_negative_two :
  (-2^2 = -4) ∧ ((-2)^3 = -8) ∧ ((-2)^2 = 4) := by
  sorry

end NUMINAMATH_CALUDE_exponent_and_square_of_negative_two_l3995_399588


namespace NUMINAMATH_CALUDE_min_value_theorem_l3995_399549

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) : 
  (Real.sqrt ((x^2 + y^2) * (4 * x^2 + y^2))) / (x * y) ≥ (2 + Real.rpow 4 (1/3)) / Real.rpow 2 (1/3) := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3995_399549


namespace NUMINAMATH_CALUDE_scientific_notation_of_280_million_l3995_399552

theorem scientific_notation_of_280_million :
  ∃ (a : ℝ) (n : ℤ), 1 ≤ a ∧ a < 10 ∧ 280000000 = a * (10 : ℝ) ^ n ∧ a = 2.8 ∧ n = 8 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_of_280_million_l3995_399552


namespace NUMINAMATH_CALUDE_xyz_value_l3995_399513

theorem xyz_value (x y z : ℝ) 
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 37)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 11) :
  x * y * z = 26 / 3 := by
sorry

end NUMINAMATH_CALUDE_xyz_value_l3995_399513


namespace NUMINAMATH_CALUDE_age_difference_l3995_399522

theorem age_difference (a b c : ℕ) (h : a + b = b + c + 16) : a - c = 16 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l3995_399522


namespace NUMINAMATH_CALUDE_ping_pong_ball_probability_l3995_399501

def is_multiple (n m : ℕ) : Prop := ∃ k : ℕ, n = m * k

def count_multiples (upper_bound divisor : ℕ) : ℕ :=
  (upper_bound / divisor)

theorem ping_pong_ball_probability :
  let total_balls : ℕ := 75
  let multiples_of_6 := count_multiples total_balls 6
  let multiples_of_8 := count_multiples total_balls 8
  let multiples_of_24 := count_multiples total_balls 24
  let favorable_outcomes := multiples_of_6 + multiples_of_8 - multiples_of_24
  (favorable_outcomes : ℚ) / total_balls = 6 / 25 := by
  sorry

end NUMINAMATH_CALUDE_ping_pong_ball_probability_l3995_399501


namespace NUMINAMATH_CALUDE_bowl_capacity_l3995_399576

/-- Given a bowl filled with oil and vinegar, prove its capacity. -/
theorem bowl_capacity (oil_density vinegar_density : ℝ)
                      (oil_fraction vinegar_fraction : ℝ)
                      (total_weight : ℝ) :
  oil_density = 5 →
  vinegar_density = 4 →
  oil_fraction = 2/3 →
  vinegar_fraction = 1/3 →
  total_weight = 700 →
  oil_fraction * oil_density + vinegar_fraction * vinegar_density = total_weight / 150 :=
by sorry

end NUMINAMATH_CALUDE_bowl_capacity_l3995_399576


namespace NUMINAMATH_CALUDE_closest_perfect_square_to_1042_l3995_399591

def is_perfect_square (n : ℤ) : Prop := ∃ m : ℤ, n = m * m

theorem closest_perfect_square_to_1042 :
  ∃ (n : ℤ), is_perfect_square n ∧
    ∀ (m : ℤ), is_perfect_square m → |n - 1042| ≤ |m - 1042| ∧
    n = 1024 :=
by sorry

end NUMINAMATH_CALUDE_closest_perfect_square_to_1042_l3995_399591


namespace NUMINAMATH_CALUDE_pairing_theorem_l3995_399531

def is_valid_pairing (n : ℕ) (pairing : List (ℕ × ℕ)) : Prop :=
  pairing.length = n ∧
  pairing.all (λ p => p.1 ≤ 2*n ∧ p.2 ≤ 2*n) ∧
  (List.range (2*n)).all (λ i => pairing.any (λ p => p.1 = i+1 ∨ p.2 = i+1))

def pairing_product (pairing : List (ℕ × ℕ)) : ℕ :=
  pairing.foldl (λ acc p => acc * (p.1 + p.2)) 1

theorem pairing_theorem (n : ℕ) (h : n > 1) :
  ∃ pairing : List (ℕ × ℕ), is_valid_pairing n pairing ∧
  ∃ m : ℕ, pairing_product pairing = m * m :=
sorry

end NUMINAMATH_CALUDE_pairing_theorem_l3995_399531


namespace NUMINAMATH_CALUDE_solution_bounded_l3995_399555

open Real

/-- A function satisfying the differential equation y'' + e^x y = 0 is bounded -/
theorem solution_bounded (f : ℝ → ℝ) (hf : ∀ x, (deriv^[2] f) x + exp x * f x = 0) :
  ∃ M, ∀ x, |f x| ≤ M :=
sorry

end NUMINAMATH_CALUDE_solution_bounded_l3995_399555


namespace NUMINAMATH_CALUDE_room_width_calculation_l3995_399551

theorem room_width_calculation (length : ℝ) (partial_area : ℝ) (additional_area : ℝ) : 
  length = 11 → 
  partial_area = 16 → 
  additional_area = 149 → 
  (partial_area + additional_area) / length = 15 := by
sorry

end NUMINAMATH_CALUDE_room_width_calculation_l3995_399551


namespace NUMINAMATH_CALUDE_class_size_with_sports_participation_l3995_399538

/-- The number of students in a class with given sports participation. -/
theorem class_size_with_sports_participation
  (football : ℕ)
  (long_tennis : ℕ)
  (both : ℕ)
  (neither : ℕ)
  (h1 : football = 26)
  (h2 : long_tennis = 20)
  (h3 : both = 17)
  (h4 : neither = 6) :
  football + long_tennis - both + neither = 35 := by
  sorry

end NUMINAMATH_CALUDE_class_size_with_sports_participation_l3995_399538


namespace NUMINAMATH_CALUDE_initial_solution_volume_l3995_399521

/-- Proves that the initial amount of solution is 6 litres, given that it is 25% alcohol
    and becomes 50% alcohol when 3 litres of pure alcohol are added. -/
theorem initial_solution_volume (x : ℝ) :
  (0.25 * x) / x = 0.25 →
  ((0.25 * x + 3) / (x + 3) = 0.5) →
  x = 6 := by
sorry

end NUMINAMATH_CALUDE_initial_solution_volume_l3995_399521


namespace NUMINAMATH_CALUDE_cubic_root_sum_l3995_399558

theorem cubic_root_sum (p q r : ℝ) : 
  p^3 - 4*p^2 + 6*p - 3 = 0 ∧ 
  q^3 - 4*q^2 + 6*q - 3 = 0 ∧ 
  r^3 - 4*r^2 + 6*r - 3 = 0 → 
  p / (q*r + 2) + q / (p*r + 2) + r / (p*q + 2) = 4/5 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_sum_l3995_399558


namespace NUMINAMATH_CALUDE_right_triangle_partition_l3995_399510

-- Define the set of points on the sides of an equilateral triangle
def TrianglePoints : Type := Set (ℝ × ℝ)

-- Define a property that a set of points contains a right triangle
def ContainsRightTriangle (S : Set (ℝ × ℝ)) : Prop :=
  ∃ (a b c : ℝ × ℝ), a ∈ S ∧ b ∈ S ∧ c ∈ S ∧
    (a.1 - b.1) * (a.1 - c.1) + (a.2 - b.2) * (a.2 - c.2) = 0

-- State the theorem
theorem right_triangle_partition (T : TrianglePoints) :
  ∀ (S₁ S₂ : Set (ℝ × ℝ)), S₁ ∪ S₂ = T ∧ S₁ ∩ S₂ = ∅ →
    ContainsRightTriangle S₁ ∨ ContainsRightTriangle S₂ :=
sorry

end NUMINAMATH_CALUDE_right_triangle_partition_l3995_399510


namespace NUMINAMATH_CALUDE_a_eq_one_sufficient_not_necessary_l3995_399553

/-- The quadratic function f(x) with parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - 2*x + 1

/-- Predicate that checks if f has only one zero for a given a -/
def has_only_one_zero (a : ℝ) : Prop :=
  ∃! x, f a x = 0

/-- Theorem stating that a=1 is sufficient but not necessary for f to have only one zero -/
theorem a_eq_one_sufficient_not_necessary :
  (∀ a : ℝ, a = 1 → has_only_one_zero a) ∧
  ¬(∀ a : ℝ, has_only_one_zero a → a = 1) :=
sorry

end NUMINAMATH_CALUDE_a_eq_one_sufficient_not_necessary_l3995_399553


namespace NUMINAMATH_CALUDE_factorization_of_quadratic_l3995_399578

theorem factorization_of_quadratic (m : ℝ) : m^2 - 4*m = m*(m - 4) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_quadratic_l3995_399578


namespace NUMINAMATH_CALUDE_two_red_balls_probability_l3995_399537

def total_balls : ℕ := 10
def red_balls : ℕ := 4
def white_balls : ℕ := 6
def selected_balls : ℕ := 5

def probability_two_red : ℚ := 10 / 21

theorem two_red_balls_probability :
  (Nat.choose red_balls 2 * Nat.choose white_balls 3) / Nat.choose total_balls selected_balls = probability_two_red :=
sorry

end NUMINAMATH_CALUDE_two_red_balls_probability_l3995_399537


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_isosceles_triangle_perimeter_is_15_l3995_399527

/-- An isosceles triangle with side lengths 3, 6, and 6 has a perimeter of 15 -/
theorem isosceles_triangle_perimeter : ℝ → ℝ → ℝ → ℝ → Prop :=
  fun a b c p =>
    (a = 3 ∧ b = 6 ∧ c = 6) →  -- Two sides are 6, one side is 3
    (a + b > c ∧ b + c > a ∧ c + a > b) →  -- Triangle inequality
    (b = c) →  -- Isosceles condition
    (p = a + b + c) →  -- Definition of perimeter
    p = 15

theorem isosceles_triangle_perimeter_is_15 : 
  ∃ (a b c p : ℝ), isosceles_triangle_perimeter a b c p :=
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_isosceles_triangle_perimeter_is_15_l3995_399527


namespace NUMINAMATH_CALUDE_log_division_simplification_l3995_399580

theorem log_division_simplification : 
  Real.log 8 / Real.log (1/8) = -1 := by sorry

end NUMINAMATH_CALUDE_log_division_simplification_l3995_399580


namespace NUMINAMATH_CALUDE_postman_pete_miles_l3995_399560

/-- Represents a pedometer with a maximum reading before reset --/
structure Pedometer where
  max_reading : Nat
  resets : Nat
  final_reading : Nat

/-- Calculates the total steps recorded by a pedometer --/
def total_steps (p : Pedometer) : Nat :=
  p.resets * (p.max_reading + 1) + p.final_reading

/-- Converts steps to miles given a steps-per-mile rate --/
def steps_to_miles (steps : Nat) (steps_per_mile : Nat) : Nat :=
  steps / steps_per_mile

/-- Theorem stating the total miles walked by Postman Pete --/
theorem postman_pete_miles :
  let p : Pedometer := { max_reading := 99999, resets := 50, final_reading := 25000 }
  let total_miles := steps_to_miles (total_steps p) 1500
  total_miles = 3350 := by
  sorry

end NUMINAMATH_CALUDE_postman_pete_miles_l3995_399560


namespace NUMINAMATH_CALUDE_beta_value_l3995_399592

theorem beta_value (β : ℂ) 
  (h1 : β ≠ 1)
  (h2 : Complex.abs (β^2 - 1) = 3 * Complex.abs (β - 1))
  (h3 : Complex.abs (β^4 - 1) = 5 * Complex.abs (β - 1)) :
  β = 2 := by
  sorry

end NUMINAMATH_CALUDE_beta_value_l3995_399592


namespace NUMINAMATH_CALUDE_eight_lines_divide_plane_into_37_regions_l3995_399506

/-- The number of regions created by n lines in a plane, where no two are parallel and no three are concurrent -/
def num_regions (n : ℕ) : ℕ := 1 + n + n.choose 2

/-- Theorem stating that 8 lines divide the plane into 37 regions -/
theorem eight_lines_divide_plane_into_37_regions :
  num_regions 8 = 37 := by sorry

end NUMINAMATH_CALUDE_eight_lines_divide_plane_into_37_regions_l3995_399506


namespace NUMINAMATH_CALUDE_wholesale_cost_calculation_l3995_399512

/-- The wholesale cost of a sleeping bag -/
def wholesale_cost : ℝ := 24.56

/-- The selling price of a sleeping bag -/
def selling_price : ℝ := 28

/-- The gross profit percentage -/
def profit_percentage : ℝ := 0.14

theorem wholesale_cost_calculation :
  selling_price = wholesale_cost * (1 + profit_percentage) := by
  sorry

end NUMINAMATH_CALUDE_wholesale_cost_calculation_l3995_399512


namespace NUMINAMATH_CALUDE_five_segments_create_fifteen_sections_l3995_399520

/-- The maximum number of sections created by n line segments in a rectangle,
    where each new line intersects all previously drawn lines inside the rectangle. -/
def max_sections (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | 1 => 2
  | k + 2 => max_sections (k + 1) + k + 1

/-- The theorem stating that 5 line segments create a maximum of 15 sections. -/
theorem five_segments_create_fifteen_sections :
  max_sections 5 = 15 := by
  sorry

end NUMINAMATH_CALUDE_five_segments_create_fifteen_sections_l3995_399520


namespace NUMINAMATH_CALUDE_third_set_size_l3995_399515

/-- The number of students in the third set that satisfies the given conditions -/
def third_set_students : ℕ := 60

/-- The pass percentage of the whole set -/
def total_pass_percentage : ℚ := 266 / 300

theorem third_set_size :
  let first_set := 40
  let second_set := 50
  let first_pass_rate := 1
  let second_pass_rate := 9 / 10
  let third_pass_rate := 4 / 5
  (first_set * first_pass_rate + second_set * second_pass_rate + third_set_students * third_pass_rate) /
    (first_set + second_set + third_set_students) = total_pass_percentage := by
  sorry

#check third_set_size

end NUMINAMATH_CALUDE_third_set_size_l3995_399515


namespace NUMINAMATH_CALUDE_cos_450_degrees_l3995_399594

theorem cos_450_degrees (h1 : ∀ x, Real.cos (x + 2 * Real.pi) = Real.cos x)
                         (h2 : Real.cos (Real.pi / 2) = 0) : 
  Real.cos (5 * Real.pi / 2) = 0 := by
sorry

end NUMINAMATH_CALUDE_cos_450_degrees_l3995_399594


namespace NUMINAMATH_CALUDE_chinese_english_time_difference_l3995_399582

/-- The number of hours Ryan spends daily learning English -/
def english_hours : ℕ := 6

/-- The number of hours Ryan spends daily learning Chinese -/
def chinese_hours : ℕ := 7

/-- Theorem: The difference between the time spent on learning Chinese and English is 1 hour -/
theorem chinese_english_time_difference :
  chinese_hours - english_hours = 1 := by
  sorry

end NUMINAMATH_CALUDE_chinese_english_time_difference_l3995_399582


namespace NUMINAMATH_CALUDE_total_students_count_l3995_399533

/-- The total number of students in a primary school height survey. -/
def total_students : ℕ := 621

/-- The number of students with heights not exceeding 130 cm. -/
def students_under_130 : ℕ := 99

/-- The average height of students not exceeding 130 cm, in cm. -/
def avg_height_under_130 : ℝ := 122

/-- The number of students with heights not less than 160 cm. -/
def students_over_160 : ℕ := 72

/-- The average height of students not less than 160 cm, in cm. -/
def avg_height_over_160 : ℝ := 163

/-- The average height of students exceeding 130 cm, in cm. -/
def avg_height_130_to_160 : ℝ := 155

/-- The average height of students below 160 cm, in cm. -/
def avg_height_under_160 : ℝ := 148

/-- Theorem stating that given the conditions, the total number of students is 621. -/
theorem total_students_count : total_students = students_under_130 + students_over_160 + 
  (total_students - students_under_130 - students_over_160) :=
by sorry

end NUMINAMATH_CALUDE_total_students_count_l3995_399533


namespace NUMINAMATH_CALUDE_janets_employees_work_hours_l3995_399595

/-- Represents the problem of calculating work hours for Janet's employees --/
theorem janets_employees_work_hours :
  let warehouse_workers : ℕ := 4
  let managers : ℕ := 2
  let warehouse_wage : ℚ := 15
  let manager_wage : ℚ := 20
  let fica_tax_rate : ℚ := (1 / 10 : ℚ)
  let days_per_month : ℕ := 25
  let total_monthly_cost : ℚ := 22000

  ∃ (hours_per_day : ℚ),
    (warehouse_workers * warehouse_wage * hours_per_day * days_per_month +
     managers * manager_wage * hours_per_day * days_per_month) * (1 + fica_tax_rate) = total_monthly_cost ∧
    hours_per_day = 8 := by
  sorry

end NUMINAMATH_CALUDE_janets_employees_work_hours_l3995_399595


namespace NUMINAMATH_CALUDE_function_satisfying_divisibility_l3995_399507

theorem function_satisfying_divisibility (f : ℕ+ → ℕ+) :
  (∀ a b : ℕ+, (f a + b) ∣ (a^2 + f a * f b)) →
  ∀ n : ℕ+, f n = n :=
by sorry

end NUMINAMATH_CALUDE_function_satisfying_divisibility_l3995_399507


namespace NUMINAMATH_CALUDE_simplify_trig_expression_l3995_399541

theorem simplify_trig_expression (θ : Real) (h : θ ∈ Set.Icc (5 * Real.pi / 4) (3 * Real.pi / 2)) :
  Real.sqrt (1 - Real.sin (2 * θ)) - Real.sqrt (1 + Real.sin (2 * θ)) = 2 * Real.cos θ := by
  sorry

end NUMINAMATH_CALUDE_simplify_trig_expression_l3995_399541
