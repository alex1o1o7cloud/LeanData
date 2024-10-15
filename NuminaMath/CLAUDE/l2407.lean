import Mathlib

namespace NUMINAMATH_CALUDE_debby_water_bottles_l2407_240743

/-- The number of water bottles Debby drank per day -/
def bottles_per_day : ℕ := 109

/-- The number of days the bottles lasted -/
def days_lasted : ℕ := 74

/-- The total number of bottles Debby bought -/
def total_bottles : ℕ := bottles_per_day * days_lasted

theorem debby_water_bottles : total_bottles = 8066 := by
  sorry

end NUMINAMATH_CALUDE_debby_water_bottles_l2407_240743


namespace NUMINAMATH_CALUDE_maxwell_brad_meeting_time_l2407_240715

/-- Proves that Maxwell walks for 8 hours before meeting Brad given the specified conditions -/
theorem maxwell_brad_meeting_time :
  let distance_between_homes : ℝ := 74
  let maxwell_speed : ℝ := 4
  let brad_speed : ℝ := 6
  let brad_delay : ℝ := 1

  let meeting_time : ℝ := 
    (distance_between_homes - maxwell_speed * brad_delay) / (maxwell_speed + brad_speed)

  let maxwell_total_time : ℝ := meeting_time + brad_delay

  maxwell_total_time = 8 := by
  sorry

end NUMINAMATH_CALUDE_maxwell_brad_meeting_time_l2407_240715


namespace NUMINAMATH_CALUDE_unique_solution_m_l2407_240766

theorem unique_solution_m (m : ℚ) : 
  (∃! x, (x - 3) / (m * x + 4) = 2 * x) ↔ m = 49 / 24 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_m_l2407_240766


namespace NUMINAMATH_CALUDE_second_largest_divisor_sum_l2407_240725

theorem second_largest_divisor_sum (n : ℕ) : 
  n > 1 → 
  (∃ p : ℕ, Prime p ∧ p ∣ n ∧ n + n / p = 2013) → 
  n = 1342 := by
sorry

end NUMINAMATH_CALUDE_second_largest_divisor_sum_l2407_240725


namespace NUMINAMATH_CALUDE_cube_root_expansion_implication_l2407_240775

-- Define the cube root function
noncomputable def cubeRoot (x : ℝ) : ℝ := Real.rpow x (1/3)

theorem cube_root_expansion_implication (n : ℕ) (hn : n > 0) :
  ∃! (a_n b_n c_n : ℤ), (1 + 4 * cubeRoot 2 - 4 * cubeRoot 4)^n = 
    a_n + b_n * cubeRoot 2 + c_n * cubeRoot 4 →
  (c_n = 0 → n = 0) := by
sorry

end NUMINAMATH_CALUDE_cube_root_expansion_implication_l2407_240775


namespace NUMINAMATH_CALUDE_parallelogram_not_always_axisymmetric_and_centrally_symmetric_l2407_240704

-- Define a parallelogram
structure Parallelogram where
  vertices : Fin 4 → ℝ × ℝ
  is_parallelogram : ∀ (i : Fin 4), 
    vertices i - vertices ((i + 1) % 4) = vertices ((i + 2) % 4) - vertices ((i + 3) % 4)

-- Define an axisymmetric figure
def IsAxisymmetric (vertices : Fin 4 → ℝ × ℝ) : Prop :=
  ∃ (axis : ℝ × ℝ → ℝ × ℝ), ∀ (i : Fin 4), 
    axis (vertices i) = vertices ((4 - i) % 4)

-- Define a centrally symmetric figure
def IsCentrallySymmetric (vertices : Fin 4 → ℝ × ℝ) : Prop :=
  ∃ (center : ℝ × ℝ), ∀ (i : Fin 4), 
    vertices i - center = center - vertices ((i + 2) % 4)

-- Theorem statement
theorem parallelogram_not_always_axisymmetric_and_centrally_symmetric :
  ¬(∀ (p : Parallelogram), IsAxisymmetric p.vertices ∧ IsCentrallySymmetric p.vertices) :=
sorry

end NUMINAMATH_CALUDE_parallelogram_not_always_axisymmetric_and_centrally_symmetric_l2407_240704


namespace NUMINAMATH_CALUDE_max_min_f_on_interval_l2407_240723

def f (x : ℝ) := x^3 - 3*x + 1

theorem max_min_f_on_interval :
  let a := -3
  let b := 0
  ∃ (x_max x_min : ℝ), x_max ∈ Set.Icc a b ∧ x_min ∈ Set.Icc a b ∧
    (∀ x ∈ Set.Icc a b, f x ≤ f x_max) ∧
    (∀ x ∈ Set.Icc a b, f x_min ≤ f x) ∧
    f x_max = 1 ∧ f x_min = -17 :=
sorry

end NUMINAMATH_CALUDE_max_min_f_on_interval_l2407_240723


namespace NUMINAMATH_CALUDE_minimum_score_for_average_increase_l2407_240771

def larry_scores : List ℕ := [75, 65, 85, 95, 60]
def target_increase : ℕ := 5

theorem minimum_score_for_average_increase 
  (scores : List ℕ) 
  (target_increase : ℕ) 
  (h1 : scores = larry_scores) 
  (h2 : target_increase = 5) : 
  ∃ (next_score : ℕ),
    (next_score = 106) ∧ 
    ((scores.sum + next_score) / (scores.length + 1) : ℚ) = 
    (scores.sum / scores.length : ℚ) + target_increase ∧
    ∀ (x : ℕ), x < next_score → 
      ((scores.sum + x) / (scores.length + 1) : ℚ) < 
      (scores.sum / scores.length : ℚ) + target_increase := by
  sorry

end NUMINAMATH_CALUDE_minimum_score_for_average_increase_l2407_240771


namespace NUMINAMATH_CALUDE_reeya_average_score_l2407_240726

theorem reeya_average_score : 
  let scores : List ℝ := [50, 60, 70, 80, 80]
  (scores.sum / scores.length : ℝ) = 68 := by
sorry

end NUMINAMATH_CALUDE_reeya_average_score_l2407_240726


namespace NUMINAMATH_CALUDE_unique_sum_of_equation_l2407_240778

theorem unique_sum_of_equation (x y : ℤ) :
  (1 / x + 1 / y) * (1 / x^2 + 1 / y^2) = -2/3 * (1 / x^4 - 1 / y^4) →
  ∃! s : ℤ, s = x + y :=
by sorry

end NUMINAMATH_CALUDE_unique_sum_of_equation_l2407_240778


namespace NUMINAMATH_CALUDE_house_tower_difference_l2407_240705

/-- Represents the number of blocks Randy used for different purposes -/
structure BlockCounts where
  total : ℕ
  house : ℕ
  tower : ℕ

/-- Theorem stating the difference in blocks used for house and tower -/
theorem house_tower_difference (randy : BlockCounts)
  (h1 : randy.total = 90)
  (h2 : randy.house = 89)
  (h3 : randy.tower = 63) :
  randy.house - randy.tower = 26 := by
  sorry

end NUMINAMATH_CALUDE_house_tower_difference_l2407_240705


namespace NUMINAMATH_CALUDE_spending_difference_is_131_75_l2407_240718

/-- Calculates the difference in spending between Coach A and Coach B -/
def spending_difference : ℝ :=
  let coach_a_basketball_cost : ℝ := 10 * 29
  let coach_a_soccer_ball_cost : ℝ := 5 * 15
  let coach_a_total_before_discount : ℝ := coach_a_basketball_cost + coach_a_soccer_ball_cost
  let coach_a_discount : ℝ := 0.05 * coach_a_total_before_discount
  let coach_a_total : ℝ := coach_a_total_before_discount - coach_a_discount

  let coach_b_baseball_cost : ℝ := 14 * 2.5
  let coach_b_baseball_bat_cost : ℝ := 18
  let coach_b_hockey_stick_cost : ℝ := 4 * 25
  let coach_b_hockey_mask_cost : ℝ := 72
  let coach_b_total_before_discount : ℝ := coach_b_baseball_cost + coach_b_baseball_bat_cost + 
                                           coach_b_hockey_stick_cost + coach_b_hockey_mask_cost
  let coach_b_discount : ℝ := 10
  let coach_b_total : ℝ := coach_b_total_before_discount - coach_b_discount

  coach_a_total - coach_b_total

/-- The theorem states that the difference in spending between Coach A and Coach B is $131.75 -/
theorem spending_difference_is_131_75 : spending_difference = 131.75 := by
  sorry

end NUMINAMATH_CALUDE_spending_difference_is_131_75_l2407_240718


namespace NUMINAMATH_CALUDE_chemistry_class_size_l2407_240702

theorem chemistry_class_size 
  (total_students : ℕ) 
  (both_subjects : ℕ) 
  (h1 : total_students = 52) 
  (h2 : both_subjects = 8) 
  (h3 : ∃ (biology_only chemistry_only : ℕ), 
    total_students = biology_only + chemistry_only + both_subjects ∧
    chemistry_only + both_subjects = 2 * (biology_only + both_subjects)) :
  ∃ (chemistry_class : ℕ), chemistry_class = 40 ∧ 
    chemistry_class = (total_students - both_subjects) / 3 * 2 + both_subjects :=
by
  sorry

end NUMINAMATH_CALUDE_chemistry_class_size_l2407_240702


namespace NUMINAMATH_CALUDE_hyperbola_circle_tangency_l2407_240727

/-- Given a hyperbola and a circle satisfying certain conditions, prove the values of a² and b² -/
theorem hyperbola_circle_tangency (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1 →  -- Hyperbola equation
    ∃ t : ℝ, (a * t)^2 + (b * t)^2 = (x - 3)^2 + y^2) →  -- Asymptotes touch the circle
  (a^2 + b^2 = 9) →  -- Right focus coincides with circle center
  (a^2 = 5 ∧ b^2 = 4) := by sorry

end NUMINAMATH_CALUDE_hyperbola_circle_tangency_l2407_240727


namespace NUMINAMATH_CALUDE_log_ratio_identity_l2407_240797

theorem log_ratio_identity 
  (x y a b : ℝ) 
  (hx : x > 0) (hy : y > 0) (ha : a > 0) (hb : b > 0) 
  (ha_neq : a ≠ 1) (hb_neq : b ≠ 1) : 
  (Real.log x / Real.log a) / (Real.log y / Real.log a) = 1 / (Real.log y / Real.log b) := by
  sorry

end NUMINAMATH_CALUDE_log_ratio_identity_l2407_240797


namespace NUMINAMATH_CALUDE_retailer_pens_count_l2407_240793

theorem retailer_pens_count : ℕ :=
  let market_price : ℝ := 1  -- Arbitrary unit price
  let discount_rate : ℝ := 0.01
  let profit_rate : ℝ := 0.09999999999999996
  let cost_36_pens : ℝ := 36 * market_price
  let selling_price : ℝ := market_price * (1 - discount_rate)
  let n : ℕ := 40  -- Number of pens to be proven

  have h1 : n * selling_price - cost_36_pens = profit_rate * cost_36_pens := by sorry
  
  n


end NUMINAMATH_CALUDE_retailer_pens_count_l2407_240793


namespace NUMINAMATH_CALUDE_problem_solution_l2407_240736

theorem problem_solution : -1^2015 + |(-3)| - (1/2)^2 * 8 + (-2)^3 / 4 = -2 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2407_240736


namespace NUMINAMATH_CALUDE_power_calculation_l2407_240799

theorem power_calculation : (8^5 / 8^3) * 3^6 = 46656 := by
  sorry

end NUMINAMATH_CALUDE_power_calculation_l2407_240799


namespace NUMINAMATH_CALUDE_world_book_day_solution_l2407_240772

/-- Represents the number of books bought by each student -/
structure BookCount where
  a : ℕ
  b : ℕ

/-- The conditions of the World Book Day problem -/
def worldBookDayProblem (bc : BookCount) : Prop :=
  bc.a + bc.b = 22 ∧ bc.a = 2 * bc.b + 1

/-- The theorem stating the solution to the World Book Day problem -/
theorem world_book_day_solution :
  ∃ (bc : BookCount), worldBookDayProblem bc ∧ bc.a = 15 ∧ bc.b = 7 := by
  sorry

end NUMINAMATH_CALUDE_world_book_day_solution_l2407_240772


namespace NUMINAMATH_CALUDE_inequality_solution_range_l2407_240710

theorem inequality_solution_range (a : ℝ) : 
  (∃ x ∈ Set.Icc 1 4, x^2 + a*x - 2 < 0) ↔ a < 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_range_l2407_240710


namespace NUMINAMATH_CALUDE_angle_C_is_right_max_sum_CP_CB_l2407_240722

noncomputable section

-- Define the triangle ABC
variable (A B C : ℝ) -- Angles
variable (a b c : ℝ) -- Sides

-- Define the relationship between sides and angles
axiom sine_law : a / (Real.sin A) = b / (Real.sin B)

-- Given condition
axiom condition : 2 * c * Real.cos C + c = a * Real.cos B + b * Real.cos A

-- Point P on AB
variable (P : ℝ)

-- BP = 2
axiom BP_length : P = 2

-- sin∠PCA = 1/3
axiom sin_PCA : Real.sin (A - P) = 1/3

-- Theorem 1: Prove C = π/2
theorem angle_C_is_right : C = Real.pi / 2 := by sorry

-- Theorem 2: Prove CP + CB ≤ 2√3 for any valid P
theorem max_sum_CP_CB : ∀ x y : ℝ, x + y ≤ 2 * Real.sqrt 3 := by sorry

end

end NUMINAMATH_CALUDE_angle_C_is_right_max_sum_CP_CB_l2407_240722


namespace NUMINAMATH_CALUDE_range_of_m_is_zero_to_one_l2407_240749

open Real

-- Define the logarithm function with base 1/2
noncomputable def log_half (x : ℝ) : ℝ := log x / log (1/2)

-- State the theorem
theorem range_of_m_is_zero_to_one :
  ∀ x m : ℝ, 
  0 < x → x < 1 → 
  log_half x = m / (1 - m) → 
  0 < m ∧ m < 1 :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_is_zero_to_one_l2407_240749


namespace NUMINAMATH_CALUDE_number_of_divisors_3465_l2407_240745

theorem number_of_divisors_3465 : Nat.card (Nat.divisors 3465) = 24 := by
  sorry

end NUMINAMATH_CALUDE_number_of_divisors_3465_l2407_240745


namespace NUMINAMATH_CALUDE_sin_minus_cos_tan_one_third_l2407_240760

theorem sin_minus_cos_tan_one_third (θ : Real) 
  (h1 : θ ∈ Set.Ioo 0 (π/2)) 
  (h2 : Real.tan θ = 1/3) : 
  Real.sin θ - Real.cos θ = -Real.sqrt 10 / 5 := by
  sorry

end NUMINAMATH_CALUDE_sin_minus_cos_tan_one_third_l2407_240760


namespace NUMINAMATH_CALUDE_picture_books_count_l2407_240781

theorem picture_books_count (total : ℕ) (fiction : ℕ) : 
  total = 35 →
  fiction = 5 →
  let nonfiction := fiction + 4
  let autobiographies := 2 * fiction
  let other_books := fiction + nonfiction + autobiographies
  total - other_books = 11 :=
by
  sorry

end NUMINAMATH_CALUDE_picture_books_count_l2407_240781


namespace NUMINAMATH_CALUDE_quadratic_roots_imply_m_value_l2407_240707

/-- If the roots of the quadratic 10x^2 - 6x + m are (3 ± i√191)/10, then m = 227/40 -/
theorem quadratic_roots_imply_m_value (m : ℝ) : 
  (∃ x : ℂ, x^2 * 10 - x * 6 + m = 0 ∧ x = (3 + Complex.I * Real.sqrt 191) / 10) →
  m = 227 / 40 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_imply_m_value_l2407_240707


namespace NUMINAMATH_CALUDE_ball_distribution_and_fairness_l2407_240786

-- Define the total number of balls
def total_balls : ℕ := 4

-- Define the probabilities
def prob_red_or_yellow : ℚ := 3/4
def prob_yellow_or_blue : ℚ := 1/2

-- Define the number of balls of each color
def red_balls : ℕ := 2
def yellow_balls : ℕ := 1
def blue_balls : ℕ := 1

-- Define the probabilities of drawing same color and different colors
def prob_same_color : ℚ := 3/8
def prob_diff_color : ℚ := 5/8

theorem ball_distribution_and_fairness :
  (red_balls + yellow_balls + blue_balls = total_balls) ∧
  (red_balls : ℚ) / total_balls + (yellow_balls : ℚ) / total_balls = prob_red_or_yellow ∧
  (yellow_balls : ℚ) / total_balls + (blue_balls : ℚ) / total_balls = prob_yellow_or_blue ∧
  prob_diff_color > prob_same_color :=
sorry

end NUMINAMATH_CALUDE_ball_distribution_and_fairness_l2407_240786


namespace NUMINAMATH_CALUDE_books_from_first_shop_l2407_240787

theorem books_from_first_shop 
  (total_cost_first : ℝ) 
  (books_second : ℕ) 
  (cost_second : ℝ) 
  (avg_price : ℝ) 
  (h1 : total_cost_first = 1160)
  (h2 : books_second = 50)
  (h3 : cost_second = 920)
  (h4 : avg_price = 18.08695652173913)
  : ∃ (books_first : ℕ), books_first = 65 ∧ 
    (total_cost_first + cost_second) / (books_first + books_second : ℝ) = avg_price :=
by sorry

end NUMINAMATH_CALUDE_books_from_first_shop_l2407_240787


namespace NUMINAMATH_CALUDE_range_of_a_l2407_240738

theorem range_of_a (a : ℝ) : 
  (∀ x, |x - 1| < 1 → x ≥ a) ∧ 
  (∃ x, x ≥ a ∧ |x - 1| ≥ 1) → 
  a ≤ 0 := by
sorry


end NUMINAMATH_CALUDE_range_of_a_l2407_240738


namespace NUMINAMATH_CALUDE_billy_game_rounds_l2407_240730

def old_score : ℕ := 725
def min_points_per_round : ℕ := 3
def max_points_per_round : ℕ := 5
def target_score : ℕ := old_score + 1

theorem billy_game_rounds :
  let min_rounds := (target_score + max_points_per_round - 1) / max_points_per_round
  let max_rounds := target_score / min_points_per_round
  (min_rounds = 146 ∧ max_rounds = 242) := by
  sorry

end NUMINAMATH_CALUDE_billy_game_rounds_l2407_240730


namespace NUMINAMATH_CALUDE_number_count_proof_l2407_240779

theorem number_count_proof (total_avg : ℝ) (pair1_avg pair2_avg pair3_avg : ℝ) :
  total_avg = 3.95 →
  pair1_avg = 3.4 →
  pair2_avg = 3.85 →
  pair3_avg = 4.600000000000001 →
  (2 * pair1_avg + 2 * pair2_avg + 2 * pair3_avg) / total_avg = 6 := by
  sorry

#check number_count_proof

end NUMINAMATH_CALUDE_number_count_proof_l2407_240779


namespace NUMINAMATH_CALUDE_quadratic_range_iff_a_values_l2407_240733

/-- The quadratic function f(x) with parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x + 2*a + 4

/-- The theorem stating the relationship between the range of f and the values of a -/
theorem quadratic_range_iff_a_values (a : ℝ) :
  (∀ y : ℝ, y ≥ 1 → ∃ x : ℝ, f a x = y) ∧ (∀ x : ℝ, f a x ≥ 1) ↔ a = -1 ∨ a = 3 :=
sorry

end NUMINAMATH_CALUDE_quadratic_range_iff_a_values_l2407_240733


namespace NUMINAMATH_CALUDE_two_digit_number_property_l2407_240774

theorem two_digit_number_property (N : ℕ) : 
  (10 ≤ N) ∧ (N < 100) →
  (4 * (N / 10) + 2 * (N % 10) = N / 2) →
  (N = 32 ∨ N = 64 ∨ N = 96) :=
by sorry

end NUMINAMATH_CALUDE_two_digit_number_property_l2407_240774


namespace NUMINAMATH_CALUDE_square_fencing_cost_theorem_l2407_240794

/-- Represents the cost of fencing a square. -/
structure SquareFencingCost where
  totalCost : ℝ
  sideCost : ℝ

/-- The cost of fencing a square with equal side costs. -/
def fencingCost (s : SquareFencingCost) : Prop :=
  s.totalCost = 4 * s.sideCost

theorem square_fencing_cost_theorem (s : SquareFencingCost) :
  s.totalCost = 316 → fencingCost s → s.sideCost = 79 := by
  sorry

end NUMINAMATH_CALUDE_square_fencing_cost_theorem_l2407_240794


namespace NUMINAMATH_CALUDE_cylinder_volume_with_square_perimeter_l2407_240756

theorem cylinder_volume_with_square_perimeter (h : ℝ) (h_pos : h > 0) :
  let square_area : ℝ := 121
  let square_side : ℝ := Real.sqrt square_area
  let square_perimeter : ℝ := 4 * square_side
  let cylinder_radius : ℝ := square_perimeter / (2 * Real.pi)
  let cylinder_volume : ℝ := Real.pi * cylinder_radius^2 * h
  cylinder_volume = (484 / Real.pi) * h := by
  sorry

end NUMINAMATH_CALUDE_cylinder_volume_with_square_perimeter_l2407_240756


namespace NUMINAMATH_CALUDE_min_value_of_exponential_sum_l2407_240724

theorem min_value_of_exponential_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (h_sum : x + 2*y = 1) :
  ∀ z, 3^x + 9^y ≥ z → z ≤ 2 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_exponential_sum_l2407_240724


namespace NUMINAMATH_CALUDE_two_dice_sum_ten_max_digits_l2407_240791

theorem two_dice_sum_ten_max_digits : ∀ x y : ℕ,
  1 ≤ x ∧ x ≤ 6 ∧ 1 ≤ y ∧ y ≤ 6 → x + y = 10 → x < 10 ∧ y < 10 :=
by sorry

end NUMINAMATH_CALUDE_two_dice_sum_ten_max_digits_l2407_240791


namespace NUMINAMATH_CALUDE_convex_polygon_diagonal_inequality_l2407_240735

theorem convex_polygon_diagonal_inequality (n : ℕ) (d p : ℝ) (h1 : n ≥ 3) (h2 : d > 0) (h3 : p > 0) : 
  (n : ℝ) - 3 < 2 * d / p ∧ 2 * d / p < ↑(n / 2) * ↑((n + 1) / 2) - 2 := by
  sorry

end NUMINAMATH_CALUDE_convex_polygon_diagonal_inequality_l2407_240735


namespace NUMINAMATH_CALUDE_midpoint_polygon_perimeter_bound_l2407_240712

/-- A convex polygon with n sides -/
structure ConvexPolygon where
  n : ℕ
  vertices : Fin n → ℝ × ℝ
  is_convex : sorry

/-- The perimeter of a polygon -/
def perimeter (P : ConvexPolygon) : ℝ :=
  sorry

/-- The polygon formed by connecting the midpoints of sides of another polygon -/
def midpoint_polygon (P : ConvexPolygon) : ConvexPolygon :=
  sorry

/-- Theorem: The perimeter of the midpoint polygon is at least half the perimeter of the original polygon -/
theorem midpoint_polygon_perimeter_bound (P : ConvexPolygon) :
  perimeter (midpoint_polygon P) ≥ (perimeter P) / 2 := by
  sorry

end NUMINAMATH_CALUDE_midpoint_polygon_perimeter_bound_l2407_240712


namespace NUMINAMATH_CALUDE_sin_135_degrees_l2407_240703

theorem sin_135_degrees : Real.sin (135 * π / 180) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_135_degrees_l2407_240703


namespace NUMINAMATH_CALUDE_chemical_solution_concentration_l2407_240773

/-- Prove that given the conditions of the chemical solution problem, 
    the original solution concentration is 85%. -/
theorem chemical_solution_concentration 
  (x : ℝ) 
  (P : ℝ) 
  (h1 : x = 0.6923076923076923)
  (h2 : (1 - x) * P + x * 20 = 40) : 
  P = 85 := by
  sorry

end NUMINAMATH_CALUDE_chemical_solution_concentration_l2407_240773


namespace NUMINAMATH_CALUDE_simplify_expression_l2407_240796

theorem simplify_expression : 
  (3 * Real.sqrt 12 - 2 * Real.sqrt (1/3) + Real.sqrt 48) / (2 * Real.sqrt 3) = 14/3 := by
sorry

end NUMINAMATH_CALUDE_simplify_expression_l2407_240796


namespace NUMINAMATH_CALUDE_fifth_pythagorean_triple_l2407_240734

def is_pythagorean_triple (a b c : ℕ) : Prop :=
  a * a + b * b = c * c

def is_consecutive (m n : ℕ) : Prop :=
  m + 1 = n

theorem fifth_pythagorean_triple (a b c : ℕ) :
  is_pythagorean_triple 3 4 5 ∧
  is_pythagorean_triple 5 12 13 ∧
  is_pythagorean_triple 7 24 25 ∧
  is_pythagorean_triple 9 40 41 ∧
  (∀ x y z : ℕ, is_pythagorean_triple x y z → Odd x) ∧
  (∀ x y z : ℕ, is_pythagorean_triple x y z → is_consecutive y z) ∧
  (∀ x y z : ℕ, is_pythagorean_triple x y z → x * x = y + z) →
  is_pythagorean_triple 11 60 61 :=
by sorry

end NUMINAMATH_CALUDE_fifth_pythagorean_triple_l2407_240734


namespace NUMINAMATH_CALUDE_sqrt_expression_equals_three_l2407_240770

theorem sqrt_expression_equals_three :
  (Real.sqrt 48 - 3 * Real.sqrt (1/3)) / Real.sqrt 3 = 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_expression_equals_three_l2407_240770


namespace NUMINAMATH_CALUDE_dihedral_angle_segment_length_l2407_240792

/-- Given a dihedral angle of 120°, this theorem calculates the length of the segment
    connecting the ends of two perpendiculars drawn from the ends of a segment on the edge
    of the dihedral angle. -/
theorem dihedral_angle_segment_length 
  (a b c : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (hc : c > 0) : 
  ∃ (length : ℝ), length = Real.sqrt (a^2 + b^2 + a*b + c^2) := by
sorry

end NUMINAMATH_CALUDE_dihedral_angle_segment_length_l2407_240792


namespace NUMINAMATH_CALUDE_room_rent_problem_l2407_240747

theorem room_rent_problem (total_rent_A total_rent_B : ℝ) 
  (rent_difference : ℝ) (h1 : total_rent_A = 4800) (h2 : total_rent_B = 4200) 
  (h3 : rent_difference = 30) :
  let rent_A := 240
  let rent_B := 210
  (total_rent_A / rent_A = total_rent_B / rent_B) ∧ 
  (rent_A = rent_B + rent_difference) := by
  sorry

end NUMINAMATH_CALUDE_room_rent_problem_l2407_240747


namespace NUMINAMATH_CALUDE_uphill_speed_calculation_l2407_240731

theorem uphill_speed_calculation (uphill_distance : ℝ) (downhill_distance : ℝ) 
  (downhill_speed : ℝ) (average_speed : ℝ) :
  uphill_distance = 100 →
  downhill_distance = 50 →
  downhill_speed = 40 →
  average_speed = 32.73 →
  ∃ uphill_speed : ℝ,
    uphill_speed = 30 ∧
    average_speed = (uphill_distance + downhill_distance) / 
      (uphill_distance / uphill_speed + downhill_distance / downhill_speed) :=
by
  sorry

end NUMINAMATH_CALUDE_uphill_speed_calculation_l2407_240731


namespace NUMINAMATH_CALUDE_unique_real_number_from_complex_cube_l2407_240769

theorem unique_real_number_from_complex_cube : 
  ∃! x : ℝ, ∃ a b : ℕ+, x = (a : ℝ)^3 - 3*a*b^2 ∧ 3*a^2*b - b^3 = 107 :=
sorry

end NUMINAMATH_CALUDE_unique_real_number_from_complex_cube_l2407_240769


namespace NUMINAMATH_CALUDE_white_area_is_122_l2407_240765

/-- Represents the dimensions of a rectangular sign -/
structure SignDimensions where
  height : ℕ
  width : ℕ

/-- Represents the areas of black portions for each letter -/
structure LetterAreas where
  m_area : ℕ
  a_area : ℕ
  t_area : ℕ
  h_area : ℕ

/-- Calculates the total area of the sign -/
def total_area (d : SignDimensions) : ℕ :=
  d.height * d.width

/-- Calculates the total black area -/
def black_area (l : LetterAreas) : ℕ :=
  l.m_area + l.a_area + l.t_area + l.h_area

/-- Calculates the white area of the sign -/
def white_area (d : SignDimensions) (l : LetterAreas) : ℕ :=
  total_area d - black_area l

/-- Theorem stating that the white area of the sign is 122 square units -/
theorem white_area_is_122 (sign : SignDimensions) (letters : LetterAreas) :
  sign.height = 8 ∧ sign.width = 24 ∧
  letters.m_area = 24 ∧ letters.a_area = 14 ∧ letters.t_area = 13 ∧ letters.h_area = 19 →
  white_area sign letters = 122 :=
by
  sorry

end NUMINAMATH_CALUDE_white_area_is_122_l2407_240765


namespace NUMINAMATH_CALUDE_power_and_division_equality_l2407_240758

theorem power_and_division_equality : (12 : ℕ)^2 * 6^4 / 432 = 432 := by
  sorry

end NUMINAMATH_CALUDE_power_and_division_equality_l2407_240758


namespace NUMINAMATH_CALUDE_yoongis_class_size_l2407_240790

theorem yoongis_class_size :
  ∀ (students_a students_b students_both : ℕ),
    students_a = 18 →
    students_b = 24 →
    students_both = 7 →
    students_a + students_b - students_both = 35 :=
by
  sorry

end NUMINAMATH_CALUDE_yoongis_class_size_l2407_240790


namespace NUMINAMATH_CALUDE_expression_equality_l2407_240741

theorem expression_equality (y : ℝ) (c : ℝ) (h : y > 0) :
  (4 * y) / 20 + (c * y) / 10 = y / 2 → c = 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l2407_240741


namespace NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l2407_240795

/-- 
Given a quadratic equation ax^2 - 4x - 1 = 0, this theorem states the conditions
on 'a' for the equation to have two distinct real roots.
-/
theorem quadratic_two_distinct_roots (a : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ 
    a * x^2 - 4 * x - 1 = 0 ∧ 
    a * y^2 - 4 * y - 1 = 0) ↔ 
  (a > -4 ∧ a ≠ 0) :=
sorry

end NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l2407_240795


namespace NUMINAMATH_CALUDE_probability_two_eight_sided_dice_less_than_three_l2407_240719

def roll_two_dice (n : ℕ) : ℕ := n * n

def outcomes_both_greater_equal (n : ℕ) (k : ℕ) : ℕ := (n - k + 1) * (n - k + 1)

def probability_at_least_one_less_than (n : ℕ) (k : ℕ) : ℚ :=
  (roll_two_dice n - outcomes_both_greater_equal n k) / roll_two_dice n

theorem probability_two_eight_sided_dice_less_than_three :
  probability_at_least_one_less_than 8 3 = 7/16 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_eight_sided_dice_less_than_three_l2407_240719


namespace NUMINAMATH_CALUDE_base2_to_base4_example_l2407_240785

/-- Converts a natural number from base 2 to base 4 -/
def base2ToBase4 (n : ℕ) : ℕ := sorry

theorem base2_to_base4_example : base2ToBase4 0b10111010000 = 0x11310 := by sorry

end NUMINAMATH_CALUDE_base2_to_base4_example_l2407_240785


namespace NUMINAMATH_CALUDE_triangle_inequality_l2407_240753

theorem triangle_inequality (a b c x y z : ℝ) 
  (triangle_cond : 0 < a ∧ 0 < b ∧ 0 < c ∧ a < b + c ∧ b < a + c ∧ c < a + b)
  (sum_zero : x + y + z = 0) :
  a^2 * y * z + b^2 * z * x + c^2 * x * y ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l2407_240753


namespace NUMINAMATH_CALUDE_gcd_lcm_product_150_225_l2407_240739

theorem gcd_lcm_product_150_225 : Nat.gcd 150 225 * Nat.lcm 150 225 = 33750 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_product_150_225_l2407_240739


namespace NUMINAMATH_CALUDE_decreasing_function_implies_a_leq_2_l2407_240750

/-- The function f(x) = -2x^2 + ax + 1 is decreasing on (1/2, +∞) -/
def is_decreasing_on_interval (a : ℝ) : Prop :=
  ∀ x y, 1/2 < x ∧ x < y → (-2*x^2 + a*x + 1) > (-2*y^2 + a*y + 1)

/-- If f(x) = -2x^2 + ax + 1 is decreasing on (1/2, +∞), then a ≤ 2 -/
theorem decreasing_function_implies_a_leq_2 :
  ∀ a : ℝ, is_decreasing_on_interval a → a ≤ 2 :=
by sorry

end NUMINAMATH_CALUDE_decreasing_function_implies_a_leq_2_l2407_240750


namespace NUMINAMATH_CALUDE_cake_division_l2407_240784

theorem cake_division (K : ℕ) (h_K : K = 1997) : 
  ∃ N : ℕ, 
    (N > 0) ∧ 
    (K ∣ N) ∧ 
    (K ∣ N^3) ∧ 
    (K ∣ 6*N^2) ∧ 
    (∀ M : ℕ, M < N → ¬(K ∣ M) ∨ ¬(K ∣ M^3) ∨ ¬(K ∣ 6*M^2)) :=
by sorry

end NUMINAMATH_CALUDE_cake_division_l2407_240784


namespace NUMINAMATH_CALUDE_hall_width_is_25_l2407_240761

/-- Represents the dimensions and cost parameters of a rectangular hall --/
structure HallParameters where
  length : ℝ
  height : ℝ
  cost_per_sqm : ℝ
  total_cost : ℝ

/-- Calculates the total area to be covered in the hall --/
def total_area (params : HallParameters) (width : ℝ) : ℝ :=
  params.length * width + 2 * (params.length * params.height) + 2 * (width * params.height)

/-- Theorem stating that the width of the hall is 25 meters given the specified parameters --/
theorem hall_width_is_25 (params : HallParameters) 
    (h1 : params.length = 20)
    (h2 : params.height = 5)
    (h3 : params.cost_per_sqm = 40)
    (h4 : params.total_cost = 38000) :
    ∃ w : ℝ, w = 25 ∧ total_area params w * params.cost_per_sqm = params.total_cost :=
  sorry

end NUMINAMATH_CALUDE_hall_width_is_25_l2407_240761


namespace NUMINAMATH_CALUDE_min_value_x_plus_2y_l2407_240782

theorem min_value_x_plus_2y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2*x + 4*y + x*y = 1) : 
  ∃ (m : ℝ), m = 2*Real.sqrt 6 - 4 ∧ x + 2*y ≥ m ∧ ∀ z, (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ 2*a + 4*b + a*b = 1 ∧ z = a + 2*b) → z ≥ m :=
sorry

end NUMINAMATH_CALUDE_min_value_x_plus_2y_l2407_240782


namespace NUMINAMATH_CALUDE_acute_triangle_properties_l2407_240742

theorem acute_triangle_properties (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ A < π / 2 →
  0 < B ∧ B < π / 2 →
  0 < C ∧ C < π / 2 →
  A + B + C = π →
  a = 2 * Real.sin A →
  b = 2 * Real.sin B →
  c = 2 * Real.sin C →
  a - b = 2 * b * Real.cos C →
  (C = 2 * B) ∧
  (π / 6 < B ∧ B < π / 4) ∧
  (Real.sqrt 2 < c / b ∧ c / b < Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_acute_triangle_properties_l2407_240742


namespace NUMINAMATH_CALUDE_max_cubes_is_117_l2407_240754

/-- The maximum number of 64 cubic centimetre cubes that can fit in a 15 cm x 20 cm x 25 cm rectangular box -/
def max_cubes : ℕ :=
  let box_volume : ℕ := 15 * 20 * 25
  let cube_volume : ℕ := 64
  (box_volume / cube_volume : ℕ)

/-- Theorem stating that the maximum number of 64 cubic centimetre cubes
    that can fit in a 15 cm x 20 cm x 25 cm rectangular box is 117 -/
theorem max_cubes_is_117 : max_cubes = 117 := by
  sorry

end NUMINAMATH_CALUDE_max_cubes_is_117_l2407_240754


namespace NUMINAMATH_CALUDE_gcf_lcm_product_4_12_l2407_240713

theorem gcf_lcm_product_4_12 : 
  (Nat.gcd 4 12) * (Nat.lcm 4 12) = 48 := by
  sorry

end NUMINAMATH_CALUDE_gcf_lcm_product_4_12_l2407_240713


namespace NUMINAMATH_CALUDE_sum_of_powers_of_i_equals_zero_l2407_240709

theorem sum_of_powers_of_i_equals_zero (i : ℂ) (hi : i^2 = -1) :
  i^1234 + i^1235 + i^1236 + i^1237 = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_powers_of_i_equals_zero_l2407_240709


namespace NUMINAMATH_CALUDE_train_length_l2407_240721

/-- The length of a train given its speed and time to pass a pole -/
theorem train_length (speed : ℝ) (time : ℝ) (h1 : speed = 72) (h2 : time = 8) :
  speed * (1000 / 3600) * time = 160 :=
by sorry

end NUMINAMATH_CALUDE_train_length_l2407_240721


namespace NUMINAMATH_CALUDE_total_pay_is_1980_l2407_240788

/-- Calculates the total monthly pay for Josh and Carl given their work hours and rates -/
def total_monthly_pay (josh_hours_per_day : ℕ) (work_days_per_week : ℕ) (weeks_per_month : ℕ)
  (carl_hours_less : ℕ) (josh_hourly_rate : ℚ) : ℚ :=
  let josh_monthly_hours := josh_hours_per_day * work_days_per_week * weeks_per_month
  let carl_monthly_hours := (josh_hours_per_day - carl_hours_less) * work_days_per_week * weeks_per_month
  let carl_hourly_rate := josh_hourly_rate / 2
  josh_monthly_hours * josh_hourly_rate + carl_monthly_hours * carl_hourly_rate

theorem total_pay_is_1980 :
  total_monthly_pay 8 5 4 2 9 = 1980 := by
  sorry

end NUMINAMATH_CALUDE_total_pay_is_1980_l2407_240788


namespace NUMINAMATH_CALUDE_polynomial_factorization_l2407_240780

theorem polynomial_factorization (a b c : ℚ) : 
  b^2 - c^2 + a*(a + 2*b) = (a + b + c)*(a + b - c) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l2407_240780


namespace NUMINAMATH_CALUDE_infinitely_many_primes_l2407_240714

theorem infinitely_many_primes : ∀ S : Finset Nat, (∀ p ∈ S, Nat.Prime p) → ∃ q, Nat.Prime q ∧ q ∉ S := by
  sorry

end NUMINAMATH_CALUDE_infinitely_many_primes_l2407_240714


namespace NUMINAMATH_CALUDE_area_ratio_of_nested_squares_l2407_240752

-- Define the squares
structure Square where
  sideLength : ℝ

-- Define the relationship between the squares
structure SquareRelationship where
  outerSquare : Square
  innerSquare : Square
  vertexRatio : ℝ

-- Theorem statement
theorem area_ratio_of_nested_squares (sr : SquareRelationship) 
  (h1 : sr.outerSquare.sideLength = 16)
  (h2 : sr.vertexRatio = 3/4) : 
  (sr.innerSquare.sideLength^2) / (sr.outerSquare.sideLength^2) = 1/8 := by
  sorry

end NUMINAMATH_CALUDE_area_ratio_of_nested_squares_l2407_240752


namespace NUMINAMATH_CALUDE_victors_percentage_l2407_240729

def marks_obtained : ℝ := 368
def maximum_marks : ℝ := 400

theorem victors_percentage : (marks_obtained / maximum_marks) * 100 = 92 := by
  sorry

end NUMINAMATH_CALUDE_victors_percentage_l2407_240729


namespace NUMINAMATH_CALUDE_find_divisor_find_divisor_proof_l2407_240701

theorem find_divisor (original : Nat) (subtracted : Nat) (divisor : Nat) : Prop :=
  let remaining := original - subtracted
  (original = 1387) →
  (subtracted = 7) →
  (remaining % divisor = 0) →
  (∀ d : Nat, d > divisor → remaining % d ≠ 0 ∨ (original - d) % d ≠ 0) →
  divisor = 23

-- The proof would go here
theorem find_divisor_proof : find_divisor 1387 7 23 := by
  sorry

end NUMINAMATH_CALUDE_find_divisor_find_divisor_proof_l2407_240701


namespace NUMINAMATH_CALUDE_education_funds_calculation_l2407_240748

/-- The GDP of China in 2012 in trillion yuan -/
def gdp_2012 : ℝ := 43.5

/-- The proportion of national financial education funds in GDP -/
def education_funds_proportion : ℝ := 0.04

/-- The national financial education funds expenditure for 2012 in billion yuan -/
def education_funds_2012 : ℝ := gdp_2012 * 1000 * education_funds_proportion

/-- Proof that the national financial education funds expenditure for 2012 
    is equal to 1.74 × 10^4 billion yuan -/
theorem education_funds_calculation : 
  education_funds_2012 = 1.74 * (10 : ℝ)^4 := by sorry

end NUMINAMATH_CALUDE_education_funds_calculation_l2407_240748


namespace NUMINAMATH_CALUDE_prime_square_problem_l2407_240777

theorem prime_square_problem (c : ℕ) (h1 : Nat.Prime c) 
  (h2 : ∃ m : ℕ, m > 0 ∧ 11 * c + 1 = m ^ 2) : c = 13 := by
  sorry

end NUMINAMATH_CALUDE_prime_square_problem_l2407_240777


namespace NUMINAMATH_CALUDE_anniversary_day_probability_probability_distribution_l2407_240783

def is_leap_year (year : ℕ) : Bool :=
  year % 4 = 0

def days_in_year (year : ℕ) : ℕ :=
  if is_leap_year year then 366 else 365

def days_between (start_year end_year : ℕ) : ℕ :=
  (List.range (end_year - start_year + 1)).foldl (λ acc y ↦ acc + days_in_year (start_year + y)) 0

theorem anniversary_day_probability (meeting_year : ℕ) 
  (h1 : meeting_year ≥ 1668 ∧ meeting_year ≤ 1671) :
  let total_days := days_between meeting_year (meeting_year + 11)
  let day_shift := total_days % 7
  (day_shift = 0 ∧ meeting_year ∈ [1668, 1670, 1671]) ∨
  (day_shift = 6 ∧ meeting_year = 1669) :=
sorry

theorem probability_distribution :
  let meeting_years := [1668, 1669, 1670, 1671]
  let friday_probability := (meeting_years.filter (λ y ↦ (days_between y (y + 11)) % 7 = 0)).length / meeting_years.length
  let thursday_probability := (meeting_years.filter (λ y ↦ (days_between y (y + 11)) % 7 = 6)).length / meeting_years.length
  friday_probability = 3/4 ∧ thursday_probability = 1/4 :=
sorry

end NUMINAMATH_CALUDE_anniversary_day_probability_probability_distribution_l2407_240783


namespace NUMINAMATH_CALUDE_max_value_e_l2407_240717

theorem max_value_e (a b c d e : ℝ) 
  (sum_condition : a + b + c + d + e = 8)
  (sum_squares_condition : a^2 + b^2 + c^2 + d^2 + e^2 = 16) :
  e ≤ 16/5 ∧ ∃ (a' b' c' d' e' : ℝ), 
    a' + b' + c' + d' + e' = 8 ∧
    a'^2 + b'^2 + c'^2 + d'^2 + e'^2 = 16 ∧
    e' = 16/5 :=
by sorry

end NUMINAMATH_CALUDE_max_value_e_l2407_240717


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l2407_240737

theorem polynomial_division_remainder (x : ℝ) :
  ∃ q : ℝ → ℝ, x^4 + 2 = (x - 2)^2 * q x + (32*x - 46) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l2407_240737


namespace NUMINAMATH_CALUDE_bankers_discount_problem_l2407_240768

theorem bankers_discount_problem (bankers_discount sum_due : ℚ) : 
  bankers_discount = 80 → sum_due = 560 → 
  (bankers_discount / (1 + bankers_discount / sum_due)) = 70 := by
sorry

end NUMINAMATH_CALUDE_bankers_discount_problem_l2407_240768


namespace NUMINAMATH_CALUDE_quadratic_root_difference_l2407_240744

theorem quadratic_root_difference (x : ℝ) : 
  let a : ℝ := 1
  let b : ℝ := -9
  let c : ℝ := 4
  let r₁ : ℝ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let r₂ : ℝ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  x^2 - 9*x + 4 = 0 → abs (r₁ - r₂) = Real.sqrt 65 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_difference_l2407_240744


namespace NUMINAMATH_CALUDE_option_c_not_algorithm_l2407_240732

-- Define what constitutes an algorithm
def is_algorithm (process : String) : Prop :=
  ∃ (steps : List String), steps.length > 0 ∧ steps.all (λ step => step.length > 0)

-- Define the options
def option_a : String := "The process of solving the equation 2x-6=0 involves moving terms and making the coefficient 1"
def option_b : String := "To get from Jinan to Vancouver, one must first take a train to Beijing, then transfer to a plane"
def option_c : String := "Solving the equation 2x^2+x-1=0"
def option_d : String := "Using the formula S=πr^2 to calculate the area of a circle with radius 3 involves computing π×3^2"

-- Theorem stating that option C is not an algorithm while others are
theorem option_c_not_algorithm :
  is_algorithm option_a ∧
  is_algorithm option_b ∧
  ¬is_algorithm option_c ∧
  is_algorithm option_d :=
sorry

end NUMINAMATH_CALUDE_option_c_not_algorithm_l2407_240732


namespace NUMINAMATH_CALUDE_abs_diff_roots_sum_of_cubes_l2407_240759

-- Define the quadratic equation
def quadratic (x : ℝ) : ℝ := 2 * x^2 + 7 * x - 4

-- Define the roots
def x₁ : ℝ := sorry
def x₂ : ℝ := sorry

-- Axioms for the roots
axiom root₁ : quadratic x₁ = 0
axiom root₂ : quadratic x₂ = 0

-- Theorems to prove
theorem abs_diff_roots : |x₁ - x₂| = 9/2 := sorry

theorem sum_of_cubes : x₁^3 + x₂^3 = -511/8 := sorry

end NUMINAMATH_CALUDE_abs_diff_roots_sum_of_cubes_l2407_240759


namespace NUMINAMATH_CALUDE_similar_triangles_shortest_side_l2407_240708

/-- Given two similar right triangles, where one has sides 7, 24, and 25 inches,
    and the other has a hypotenuse of 100 inches, the shortest side of the larger triangle is 28 inches. -/
theorem similar_triangles_shortest_side (a b c d e : ℝ) : 
  a > 0 → b > 0 → c > 0 → d > 0 → e > 0 →
  a^2 + b^2 = c^2 →
  a = 7 →
  b = 24 →
  c = 25 →
  e = 100 →
  d / a = e / c →
  d = 28 := by
sorry

end NUMINAMATH_CALUDE_similar_triangles_shortest_side_l2407_240708


namespace NUMINAMATH_CALUDE_parabola_shift_theorem_l2407_240711

/-- Represents a parabola in the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Shifts a parabola horizontally and vertically -/
def shift_parabola (p : Parabola) (h : ℝ) (v : ℝ) : Parabola :=
  { a := p.a
  , b := -2 * p.a * h + p.b
  , c := p.a * h^2 - p.b * h + p.c - v }

/-- The original parabola y = 4x^2 + 1 -/
def original_parabola : Parabola :=
  { a := 4, b := 0, c := 1 }

theorem parabola_shift_theorem :
  let shifted := shift_parabola original_parabola 3 2
  shifted = { a := 4, b := -24, c := 35 } := by sorry

end NUMINAMATH_CALUDE_parabola_shift_theorem_l2407_240711


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l2407_240751

/-- A geometric sequence is a sequence where the ratio between any two consecutive terms is constant. -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- Given a geometric sequence a where a₃ = 20 and a₆ = 5, prove that a₉ = 5/4 -/
theorem geometric_sequence_problem (a : ℕ → ℝ) 
    (h_geom : GeometricSequence a) 
    (h_a3 : a 3 = 20) 
    (h_a6 : a 6 = 5) : 
  a 9 = 5/4 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l2407_240751


namespace NUMINAMATH_CALUDE_remaining_uncracked_seashells_l2407_240757

def tom_seashells : ℕ := 15
def fred_seashells : ℕ := 43
def cracked_seashells : ℕ := 29
def giveaway_percentage : ℚ := 40 / 100

theorem remaining_uncracked_seashells :
  let total_seashells := tom_seashells + fred_seashells
  let uncracked_seashells := total_seashells - cracked_seashells
  let seashells_to_giveaway := ⌊(giveaway_percentage * uncracked_seashells : ℚ)⌋
  uncracked_seashells - seashells_to_giveaway = 18 := by sorry

end NUMINAMATH_CALUDE_remaining_uncracked_seashells_l2407_240757


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l2407_240789

/-- Given a polynomial P with integer coefficients, 
    if P(2) and P(3) are both multiples of 6, 
    then P(5) is also a multiple of 6. -/
theorem polynomial_divisibility (P : ℤ → ℤ) 
  (h_poly : ∀ x y : ℤ, ∃ k : ℤ, P (x + y) = P x + P y + k * x * y)
  (h_p2 : ∃ m : ℤ, P 2 = 6 * m)
  (h_p3 : ∃ n : ℤ, P 3 = 6 * n) :
  ∃ l : ℤ, P 5 = 6 * l := by
  sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l2407_240789


namespace NUMINAMATH_CALUDE_probability_divisible_by_4_l2407_240798

/-- Represents the possible outcomes of a single spin -/
inductive SpinOutcome
| one
| two
| three

/-- Represents a three-digit number formed by three spins -/
structure ThreeDigitNumber where
  hundreds : SpinOutcome
  tens : SpinOutcome
  units : SpinOutcome

/-- Checks if a ThreeDigitNumber is divisible by 4 -/
def isDivisibleBy4 (n : ThreeDigitNumber) : Prop := sorry

/-- The total number of possible three-digit numbers -/
def totalOutcomes : ℕ := sorry

/-- The number of three-digit numbers divisible by 4 -/
def divisibleBy4Outcomes : ℕ := sorry

/-- The main theorem stating the probability of getting a number divisible by 4 -/
theorem probability_divisible_by_4 :
  (divisibleBy4Outcomes : ℚ) / totalOutcomes = 2 / 9 := sorry

end NUMINAMATH_CALUDE_probability_divisible_by_4_l2407_240798


namespace NUMINAMATH_CALUDE_triangle_angle_C_l2407_240763

theorem triangle_angle_C (A B C : Real) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧  -- Angles are positive
  A + B + C = π ∧  -- Sum of angles in a triangle
  3 * Real.sin A + 4 * Real.cos B = 6 ∧  -- Given condition
  4 * Real.sin B + 3 * Real.cos A = 1  -- Given condition
  → C = π / 6 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_C_l2407_240763


namespace NUMINAMATH_CALUDE_parabola_increasing_condition_l2407_240767

/-- A parabola defined by y = (a - 1)x^2 + 1 -/
def parabola (a : ℝ) (x : ℝ) : ℝ := (a - 1) * x^2 + 1

/-- The parabola increases as x increases when x ≥ 0 -/
def increases_for_nonneg_x (a : ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, 0 ≤ x₁ → x₁ < x₂ → parabola a x₁ < parabola a x₂

theorem parabola_increasing_condition (a : ℝ) :
  increases_for_nonneg_x a → a > 1 := by sorry

end NUMINAMATH_CALUDE_parabola_increasing_condition_l2407_240767


namespace NUMINAMATH_CALUDE_smallest_debt_is_fifty_l2407_240764

/-- The smallest positive debt that can be settled using cows and sheep -/
def smallest_settleable_debt (cow_value sheep_value : ℕ) : ℕ :=
  Nat.gcd cow_value sheep_value

theorem smallest_debt_is_fifty :
  smallest_settleable_debt 400 250 = 50 := by
  sorry

#eval smallest_settleable_debt 400 250

end NUMINAMATH_CALUDE_smallest_debt_is_fifty_l2407_240764


namespace NUMINAMATH_CALUDE_parabola_symmetric_axis_given_parabola_symmetric_axis_l2407_240740

/-- The symmetric axis of a parabola y = ax² + bx + c is x = -b/(2a) -/
theorem parabola_symmetric_axis 
  (a b c : ℝ) (ha : a ≠ 0) :
  let f : ℝ → ℝ := λ x ↦ a * x^2 + b * x + c
  ∀ x₀, (∀ x, f (x₀ + x) = f (x₀ - x)) ↔ x₀ = -b / (2 * a) :=
sorry

/-- The symmetric axis of the parabola y = -1/4 x² + x - 4 is x = 2 -/
theorem given_parabola_symmetric_axis :
  let f : ℝ → ℝ := λ x ↦ -1/4 * x^2 + x - 4
  ∀ x₀, (∀ x, f (x₀ + x) = f (x₀ - x)) ↔ x₀ = 2 :=
sorry

end NUMINAMATH_CALUDE_parabola_symmetric_axis_given_parabola_symmetric_axis_l2407_240740


namespace NUMINAMATH_CALUDE_matrix_product_is_zero_l2407_240776

def A (d e f : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  !![0, d, -e;
    -d, 0, f;
    e, -f, 0]

def B (d e f : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  !![d^2, d*e, d*f;
    d*e, e^2, e*f;
    d*f, e*f, f^2]

theorem matrix_product_is_zero (d e f : ℝ) :
  A d e f * B d e f = 0 := by sorry

end NUMINAMATH_CALUDE_matrix_product_is_zero_l2407_240776


namespace NUMINAMATH_CALUDE_polynomial_factorization_l2407_240755

def polynomial (n x y : ℤ) : ℤ := x^2 + 2*x*y + n*x^2 + y^2 + 2*y - n^2

def is_linear_factor (f : ℤ → ℤ → ℤ) : Prop :=
  ∃ (a b c : ℤ), ∀ (x y : ℤ), f x y = a*x + b*y + c

theorem polynomial_factorization (n : ℤ) :
  (∃ (f g : ℤ → ℤ → ℤ), is_linear_factor f ∧ is_linear_factor g ∧
    (∀ (x y : ℤ), polynomial n x y = f x y * g x y)) ↔ n = 0 ∨ n = 2 ∨ n = -2 :=
sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l2407_240755


namespace NUMINAMATH_CALUDE_extreme_value_implies_a_l2407_240706

def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + 3*x - 9

theorem extreme_value_implies_a (a : ℝ) :
  (∃ (ε : ℝ), ∀ (x : ℝ), |x + 3| < ε → f a x ≤ f a (-3)) →
  a = 5 := by
  sorry

end NUMINAMATH_CALUDE_extreme_value_implies_a_l2407_240706


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ratio_property_l2407_240728

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℚ) : Prop :=
  ∃ (d : ℚ), ∀ n, a (n + 1) = a n + d

/-- Sum of the first n terms of an arithmetic sequence -/
def SumArithmeticSequence (a : ℕ → ℚ) (n : ℕ) : ℚ :=
  (n : ℚ) * (2 * a 1 + (n - 1) * (a 2 - a 1)) / 2

theorem arithmetic_sequence_ratio_property (a : ℕ → ℚ) 
    (h_arith : ArithmeticSequence a) (h_ratio : a 7 / a 4 = 7 / 13) :
    SumArithmeticSequence a 13 / SumArithmeticSequence a 7 = 1 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ratio_property_l2407_240728


namespace NUMINAMATH_CALUDE_triangle_problem_l2407_240762

theorem triangle_problem (A B C : ℝ) (a b c S : ℝ) :
  0 < A ∧ A < π ∧
  0 < B ∧ B < π ∧
  0 < C ∧ C < π ∧
  A + B + C = π ∧
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  S = (1/2) * a * b * Real.sin C ∧
  (Real.cos B) / (Real.cos C) = -b / (2*a + c) →
  (B = 2*π/3 ∧
   (a = 4 ∧ S = 5 * Real.sqrt 3 → b = Real.sqrt 61)) := by
sorry

end NUMINAMATH_CALUDE_triangle_problem_l2407_240762


namespace NUMINAMATH_CALUDE_amount_ratio_l2407_240746

def total_amount : ℕ := 7000
def r_amount : ℕ := 2800

theorem amount_ratio : 
  let pq_amount := total_amount - r_amount
  (r_amount : ℚ) / (pq_amount : ℚ) = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_amount_ratio_l2407_240746


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l2407_240720

-- Define the sets M and N
def M : Set ℝ := {y | ∃ x : ℝ, y = x^2}
def N : Set ℝ := {y | ∃ x : ℝ, x^2 + y^2 = 1}

-- State the theorem
theorem intersection_of_M_and_N : M ∩ N = Set.Icc 0 1 := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l2407_240720


namespace NUMINAMATH_CALUDE_mn_sum_for_5000_l2407_240700

theorem mn_sum_for_5000 (m n : ℕ+) : 
  m * n = 5000 →
  ¬(10 ∣ m) →
  ¬(10 ∣ n) →
  m + n = 633 := by
sorry

end NUMINAMATH_CALUDE_mn_sum_for_5000_l2407_240700


namespace NUMINAMATH_CALUDE_number_product_l2407_240716

theorem number_product (x : ℝ) : x - 7 = 9 → 5 * x = 80 := by
  sorry

end NUMINAMATH_CALUDE_number_product_l2407_240716
