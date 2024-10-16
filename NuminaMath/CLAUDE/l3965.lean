import Mathlib

namespace NUMINAMATH_CALUDE_solve_for_B_l3965_396579

theorem solve_for_B : ∃ B : ℚ, (3 * B - 5 = 23) ∧ (B = 28 / 3) := by sorry

end NUMINAMATH_CALUDE_solve_for_B_l3965_396579


namespace NUMINAMATH_CALUDE_floor_sqrt_23_squared_l3965_396502

theorem floor_sqrt_23_squared : ⌊Real.sqrt 23⌋^2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_floor_sqrt_23_squared_l3965_396502


namespace NUMINAMATH_CALUDE_smallest_n_divisible_by_2013_l3965_396526

def product_of_consecutive_evens (n : ℕ) : ℕ :=
  Finset.prod (Finset.range (n/2)) (fun i => 2 * (i + 1))

theorem smallest_n_divisible_by_2013 :
  ∀ n : ℕ, n % 2 = 0 →
    (product_of_consecutive_evens n % 2013 = 0 →
      n ≥ 122) ∧
    (n ≥ 122 →
      product_of_consecutive_evens n % 2013 = 0) :=
sorry

end NUMINAMATH_CALUDE_smallest_n_divisible_by_2013_l3965_396526


namespace NUMINAMATH_CALUDE_floor_sqrt_20_squared_l3965_396523

theorem floor_sqrt_20_squared : ⌊Real.sqrt 20⌋^2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_floor_sqrt_20_squared_l3965_396523


namespace NUMINAMATH_CALUDE_sum_remainder_by_six_l3965_396515

theorem sum_remainder_by_six : (284917 + 517084) % 6 = 5 := by
  sorry

end NUMINAMATH_CALUDE_sum_remainder_by_six_l3965_396515


namespace NUMINAMATH_CALUDE_vector_magnitude_l3965_396570

def m : ℝ × ℝ := (2, 4)

theorem vector_magnitude (m : ℝ × ℝ) (n : ℝ × ℝ) : 
  let angle := π / 3
  norm m = 2 * Real.sqrt 5 →
  norm n = Real.sqrt 5 →
  m.1 * n.1 + m.2 * n.2 = norm m * norm n * Real.cos angle →
  norm (2 • m - 3 • n) = Real.sqrt 65 := by
  sorry

end NUMINAMATH_CALUDE_vector_magnitude_l3965_396570


namespace NUMINAMATH_CALUDE_cheesecakes_in_fridge_l3965_396500

/-- Given a bakery scenario with cheesecakes, prove the number in the fridge. -/
theorem cheesecakes_in_fridge 
  (initial_display : ℕ) 
  (sold_from_display : ℕ) 
  (total_left : ℕ) 
  (h1 : initial_display = 10)
  (h2 : sold_from_display = 7)
  (h3 : total_left = 18) :
  total_left - (initial_display - sold_from_display) = 15 := by
  sorry

end NUMINAMATH_CALUDE_cheesecakes_in_fridge_l3965_396500


namespace NUMINAMATH_CALUDE_units_digit_17_pow_28_l3965_396534

theorem units_digit_17_pow_28 : (17^28) % 10 = 1 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_17_pow_28_l3965_396534


namespace NUMINAMATH_CALUDE_f_derivative_at_one_l3965_396544

noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.log x

theorem f_derivative_at_one : 
  deriv f 1 = Real.cos 1 + 1 := by sorry

end NUMINAMATH_CALUDE_f_derivative_at_one_l3965_396544


namespace NUMINAMATH_CALUDE_power_of_power_equals_base_l3965_396518

theorem power_of_power_equals_base (x : ℝ) (h : x > 0) : (x^(4/5))^(5/4) = x := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_equals_base_l3965_396518


namespace NUMINAMATH_CALUDE_three_digit_two_digit_operations_l3965_396501

theorem three_digit_two_digit_operations 
  (a b : ℕ) 
  (ha : 100 ≤ a ∧ a ≤ 999) 
  (hb : 10 ≤ b ∧ b ≤ 99) :
  (∀ x y, 100 ≤ x ∧ x ≤ 999 ∧ 10 ≤ y ∧ y ≤ 99 → x + y ≥ a + b) → a + b = 110 ∧
  (∀ x y, 100 ≤ x ∧ x ≤ 999 ∧ 10 ≤ y ∧ y ≤ 99 → x + y ≤ a + b) → a + b = 1098 ∧
  (∀ x y, 100 ≤ x ∧ x ≤ 999 ∧ 10 ≤ y ∧ y ≤ 99 → x - y ≥ a - b) → a - b = 1 ∧
  (∀ x y, 100 ≤ x ∧ x ≤ 999 ∧ 10 ≤ y ∧ y ≤ 99 → x - y ≤ a - b) → a - b = 989 :=
by sorry

end NUMINAMATH_CALUDE_three_digit_two_digit_operations_l3965_396501


namespace NUMINAMATH_CALUDE_bird_watching_percentage_difference_l3965_396582

def gabrielle_robins : ℕ := 5
def gabrielle_cardinals : ℕ := 4
def gabrielle_blue_jays : ℕ := 3

def chase_robins : ℕ := 2
def chase_blue_jays : ℕ := 3
def chase_cardinals : ℕ := 5

def gabrielle_total : ℕ := gabrielle_robins + gabrielle_cardinals + gabrielle_blue_jays
def chase_total : ℕ := chase_robins + chase_blue_jays + chase_cardinals

theorem bird_watching_percentage_difference :
  (gabrielle_total - chase_total : ℚ) / chase_total * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_bird_watching_percentage_difference_l3965_396582


namespace NUMINAMATH_CALUDE_right_triangle_circle_and_trajectory_l3965_396535

/-- Right triangle ABC with hypotenuse AB, where A(-1,0) and B(3,0) -/
structure RightTriangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  hA : A = (-1, 0)
  hB : B = (3, 0)
  isRightTriangle : sorry -- Assume this triangle is right-angled

/-- The general equation of a circle -/
def CircleEquation (h k r : ℝ) (x y : ℝ) : Prop :=
  (x - h)^2 + (y - k)^2 = r^2

/-- The equation of a trajectory -/
def TrajectoryEquation (a b r : ℝ) (x y : ℝ) : Prop :=
  (x - a)^2 + (y - b)^2 = r^2

theorem right_triangle_circle_and_trajectory 
  (triangle : RightTriangle) (x y : ℝ) (hy : y ≠ 0) :
  (CircleEquation 1 0 2 x y ↔ x^2 + y^2 - 2*x - 3 = 0) ∧
  (TrajectoryEquation 2 0 1 x y ↔ (x-2)^2 + y^2 = 1) := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_circle_and_trajectory_l3965_396535


namespace NUMINAMATH_CALUDE_positive_square_iff_greater_l3965_396511

theorem positive_square_iff_greater (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  a > b ↔ a^2 > b^2 := by sorry

end NUMINAMATH_CALUDE_positive_square_iff_greater_l3965_396511


namespace NUMINAMATH_CALUDE_harmonic_mean_4_5_10_l3965_396569

def harmonic_mean (a b c : ℚ) : ℚ := 3 / (1/a + 1/b + 1/c)

theorem harmonic_mean_4_5_10 :
  harmonic_mean 4 5 10 = 60 / 11 := by
  sorry

end NUMINAMATH_CALUDE_harmonic_mean_4_5_10_l3965_396569


namespace NUMINAMATH_CALUDE_exists_x0_implies_a_value_l3965_396558

noncomputable section

open Real

-- Define the functions f and g
def f (a x : ℝ) : ℝ := x + exp (x - a)
def g (a x : ℝ) : ℝ := log (x + 2) - 4 * exp (a - x)

-- State the theorem
theorem exists_x0_implies_a_value (a : ℝ) :
  (∃ x₀ : ℝ, f a x₀ - g a x₀ = 3) →
  a = -log 2 - 1 := by
sorry

end

end NUMINAMATH_CALUDE_exists_x0_implies_a_value_l3965_396558


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l3965_396563

/-- Two vectors are parallel if their components are proportional -/
def are_parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_x_value :
  let a : ℝ × ℝ := (-1, 3)
  let b : ℝ × ℝ := (x, 1)
  are_parallel a b → x = -1/3 :=
by sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l3965_396563


namespace NUMINAMATH_CALUDE_three_digit_with_repeat_l3965_396552

/-- The number of digits available (0 to 9) -/
def num_digits : ℕ := 10

/-- The total number of three-digit numbers -/
def total_three_digit : ℕ := 900

/-- The number of three-digit numbers without repeated digits -/
def no_repeat_three_digit : ℕ := 9 * 9 * 8

/-- Theorem: The number of three-digit numbers with repeated digits using digits 0 to 9 is 252 -/
theorem three_digit_with_repeat : 
  total_three_digit - no_repeat_three_digit = 252 := by
  sorry

end NUMINAMATH_CALUDE_three_digit_with_repeat_l3965_396552


namespace NUMINAMATH_CALUDE_line_intersects_x_axis_l3965_396585

/-- A point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Represents a line passing through two points -/
structure Line where
  p1 : Point2D
  p2 : Point2D

/-- Checks if a point lies on the x-axis -/
def isOnXAxis (p : Point2D) : Prop :=
  p.y = 0

/-- Checks if a point lies on a given line -/
def isOnLine (l : Line) (p : Point2D) : Prop :=
  (p.y - l.p1.y) * (l.p2.x - l.p1.x) = (p.x - l.p1.x) * (l.p2.y - l.p1.y)

theorem line_intersects_x_axis (l : Line) : 
  l.p1 = ⟨3, -1⟩ → l.p2 = ⟨7, 3⟩ → 
  ∃ p : Point2D, isOnLine l p ∧ isOnXAxis p ∧ p = ⟨4, 0⟩ := by
  sorry

end NUMINAMATH_CALUDE_line_intersects_x_axis_l3965_396585


namespace NUMINAMATH_CALUDE_log_relationship_l3965_396543

theorem log_relationship (a b x : ℝ) (h : 0 < a ∧ a ≠ 1 ∧ 0 < b ∧ b ≠ 1 ∧ 0 < x) :
  5 * (Real.log x / Real.log a)^2 + 2 * (Real.log x / Real.log b)^2 = 15 * (Real.log x)^2 / (Real.log a * Real.log b) →
  b = a^((3 + Real.sqrt 37) / 2) ∨ b = a^((3 - Real.sqrt 37) / 2) := by
sorry

end NUMINAMATH_CALUDE_log_relationship_l3965_396543


namespace NUMINAMATH_CALUDE_ratio_proof_l3965_396514

/-- Given two positive integers with specific properties, prove their ratio -/
theorem ratio_proof (A B : ℕ+) (h1 : A = 48) (h2 : Nat.lcm A B = 432) :
  (A : ℚ) / B = 1 / (4.5 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_ratio_proof_l3965_396514


namespace NUMINAMATH_CALUDE_square_triangle_ratio_l3965_396538

theorem square_triangle_ratio (a : ℝ) (h : a > 0) :
  let square_side := a
  let triangle_leg := a * Real.sqrt 2
  let triangle_hypotenuse := triangle_leg * Real.sqrt 2
  triangle_hypotenuse / square_side = 2 := by
sorry

end NUMINAMATH_CALUDE_square_triangle_ratio_l3965_396538


namespace NUMINAMATH_CALUDE_gcd_lcm_sum_for_special_case_l3965_396561

theorem gcd_lcm_sum_for_special_case (a b : ℕ) (h : a = 1999 * b) :
  Nat.gcd a b + Nat.lcm a b = 2000 * b := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_sum_for_special_case_l3965_396561


namespace NUMINAMATH_CALUDE_union_of_sets_l3965_396588

-- Define the sets A and B
def A (a : ℤ) : Set ℤ := {|a + 1|, 3, 5}
def B (a : ℤ) : Set ℤ := {2 * a + 1, a^2 + 2 * a, a^2 + 2 * a - 1}

-- Define the theorem
theorem union_of_sets :
  ∃ a : ℤ, (A a ∩ B a = {2, 3}) → (A a ∪ B a = {-5, 2, 3, 5}) :=
by
  sorry

end NUMINAMATH_CALUDE_union_of_sets_l3965_396588


namespace NUMINAMATH_CALUDE_range_of_a_l3965_396565

-- Define a monotonically decreasing function
def MonotonicallyDecreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x > f y

-- State the theorem
theorem range_of_a (f : ℝ → ℝ) (a : ℝ) 
  (h1 : MonotonicallyDecreasing f) 
  (h2 : f (2 - a^2) > f a) : 
  a > 1 ∨ a < -2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l3965_396565


namespace NUMINAMATH_CALUDE_intersection_of_logarithmic_graphs_l3965_396568

theorem intersection_of_logarithmic_graphs :
  ∃! x : ℝ, x > 0 ∧ 3 * Real.log x = Real.log (3 * x) := by sorry

end NUMINAMATH_CALUDE_intersection_of_logarithmic_graphs_l3965_396568


namespace NUMINAMATH_CALUDE_expression_simplification_l3965_396590

theorem expression_simplification (x y z : ℝ) (h : y ≠ 0) :
  (6 * x^3 * y^4 * z - 4 * x^2 * y^3 * z + 2 * x * y^3) / (2 * x * y^3) = 3 * x^2 * y * z - 2 * x * z + 1 :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_l3965_396590


namespace NUMINAMATH_CALUDE_cos_three_pi_halves_l3965_396519

theorem cos_three_pi_halves : Real.cos (3 * π / 2) = 0 := by
  sorry

end NUMINAMATH_CALUDE_cos_three_pi_halves_l3965_396519


namespace NUMINAMATH_CALUDE_ads_on_first_page_l3965_396537

theorem ads_on_first_page (page1 page2 page3 page4 : ℕ) : 
  page2 = 2 * page1 →
  page3 = page2 + 24 →
  page4 = 3 * page2 / 4 →
  68 = 2 * (page1 + page2 + page3 + page4) / 3 →
  page1 = 12 := by
sorry

end NUMINAMATH_CALUDE_ads_on_first_page_l3965_396537


namespace NUMINAMATH_CALUDE_cos_sum_equals_one_l3965_396593

theorem cos_sum_equals_one (α β : Real) 
  (h : (Real.cos α * Real.cos (β/2)) / Real.cos (α - β/2) + 
       (Real.cos β * Real.cos (α/2)) / Real.cos (β - α/2) = 1) : 
  Real.cos α + Real.cos β = 1 := by
  sorry

end NUMINAMATH_CALUDE_cos_sum_equals_one_l3965_396593


namespace NUMINAMATH_CALUDE_football_players_count_l3965_396528

theorem football_players_count (total : ℕ) (tennis : ℕ) (both : ℕ) (neither : ℕ) 
  (h1 : total = 39)
  (h2 : tennis = 20)
  (h3 : both = 17)
  (h4 : neither = 10) :
  ∃ football : ℕ, football = 26 ∧ (football - both) + (tennis - both) + both + neither = total :=
by sorry

end NUMINAMATH_CALUDE_football_players_count_l3965_396528


namespace NUMINAMATH_CALUDE_mode_most_relevant_for_sales_volume_l3965_396541

/-- Represents a shoe size -/
def ShoeSize := ℕ

/-- Represents a list of shoe sizes sold -/
def SalesList := List ShoeSize

/-- Calculates the mode of a list of shoe sizes -/
def mode (sales : SalesList) : ShoeSize :=
  sorry

/-- Represents the relevance of a statistical measure for determining the shoe size with highest sales volume -/
inductive Relevance
| Low : Relevance
| Medium : Relevance
| High : Relevance

/-- Determines the relevance of a statistical measure for sales volume prediction -/
def relevanceForSalesVolume (measure : String) : Relevance :=
  sorry

theorem mode_most_relevant_for_sales_volume :
  relevanceForSalesVolume "mode" = Relevance.High ∧
  (∀ m : String, m ≠ "mode" → relevanceForSalesVolume m ≠ Relevance.High) :=
sorry

end NUMINAMATH_CALUDE_mode_most_relevant_for_sales_volume_l3965_396541


namespace NUMINAMATH_CALUDE_first_prize_tickets_characterization_l3965_396591

def is_valid_ticket (n : Nat) : Prop := n ≥ 0 ∧ n ≤ 9999

def is_first_prize (n : Nat) : Prop := n % 1000 = 418

def first_prize_tickets : Set Nat :=
  {n : Nat | is_valid_ticket n ∧ is_first_prize n}

theorem first_prize_tickets_characterization :
  first_prize_tickets = {0418, 1418, 2418, 3418, 4418, 5418, 6418, 7418, 8418, 9418} :=
by sorry

end NUMINAMATH_CALUDE_first_prize_tickets_characterization_l3965_396591


namespace NUMINAMATH_CALUDE_final_weight_calculation_l3965_396555

def initial_weight : ℝ := 220
def weight_loss_percentage : ℝ := 0.1
def weight_gain : ℝ := 2

theorem final_weight_calculation :
  initial_weight - (initial_weight * weight_loss_percentage) + weight_gain = 200 :=
by sorry

end NUMINAMATH_CALUDE_final_weight_calculation_l3965_396555


namespace NUMINAMATH_CALUDE_problem_solution_l3965_396542

open Real

theorem problem_solution (α β : ℝ) (h1 : tan α = -1/3) (h2 : cos β = sqrt 5 / 5)
  (h3 : 0 < α) (h4 : α < π) (h5 : 0 < β) (h6 : β < π) :
  (tan (α + β) = 1) ∧
  (∃ (x : ℝ), sqrt 2 * sin (x - α) + cos (x + β) = sqrt 5) ∧
  (∃ (x : ℝ), sqrt 2 * sin (x - α) + cos (x + β) = -sqrt 5) ∧
  (∀ (x : ℝ), sqrt 2 * sin (x - α) + cos (x + β) ≤ sqrt 5) ∧
  (∀ (x : ℝ), -sqrt 5 ≤ sqrt 2 * sin (x - α) + cos (x + β)) := by
  sorry


end NUMINAMATH_CALUDE_problem_solution_l3965_396542


namespace NUMINAMATH_CALUDE_quadratic_one_root_l3965_396521

/-- A quadratic function f(x) = x^2 - 2x + m has exactly one root if and only if m = 1 -/
theorem quadratic_one_root (m : ℝ) : 
  (∃! x, x^2 - 2*x + m = 0) ↔ m = 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_one_root_l3965_396521


namespace NUMINAMATH_CALUDE_mairead_exercise_distance_l3965_396564

theorem mairead_exercise_distance :
  let run_distance : ℝ := 40
  let walk_distance : ℝ := (3/5) * run_distance
  let jog_distance : ℝ := (1/5) * walk_distance
  let total_distance : ℝ := run_distance + walk_distance + jog_distance
  total_distance = 64.8 := by
  sorry

end NUMINAMATH_CALUDE_mairead_exercise_distance_l3965_396564


namespace NUMINAMATH_CALUDE_find_t_l3965_396549

theorem find_t : ∃ t : ℤ, (∃ s : ℤ, 12 * s + 7 * t = 173 ∧ s = t - 3) → t = 11 := by
  sorry

end NUMINAMATH_CALUDE_find_t_l3965_396549


namespace NUMINAMATH_CALUDE_new_average_calculation_l3965_396559

theorem new_average_calculation (num_students : ℕ) (original_avg : ℝ) 
  (increase_percent : ℝ) (bonus : ℝ) (new_avg : ℝ) : 
  num_students = 37 → 
  original_avg = 73 → 
  increase_percent = 65 → 
  bonus = 15 → 
  new_avg = original_avg * (1 + increase_percent / 100) + bonus →
  new_avg = 135.45 := by
sorry

end NUMINAMATH_CALUDE_new_average_calculation_l3965_396559


namespace NUMINAMATH_CALUDE_cubic_sum_of_roots_l3965_396567

theorem cubic_sum_of_roots (p q r s : ℝ) : 
  (r^2 - p*r - q = 0) → (s^2 - p*s - q = 0) → (r^3 + s^3 = p^3 + 3*p*q) :=
by sorry

end NUMINAMATH_CALUDE_cubic_sum_of_roots_l3965_396567


namespace NUMINAMATH_CALUDE_apple_selling_price_l3965_396583

/-- The selling price of an apple, given its cost price and loss ratio. -/
def selling_price (cost_price : ℚ) (loss_ratio : ℚ) : ℚ :=
  cost_price * (1 - loss_ratio)

/-- Theorem stating that the selling price of an apple is 15,
    given a cost price of 18 and a loss ratio of 1/6. -/
theorem apple_selling_price :
  selling_price 18 (1/6) = 15 := by
  sorry

end NUMINAMATH_CALUDE_apple_selling_price_l3965_396583


namespace NUMINAMATH_CALUDE_judes_chair_expenditure_l3965_396527

/-- Proves that the amount spent on chairs is $36 given the conditions of Jude's purchase --/
theorem judes_chair_expenditure
  (table_cost : ℕ)
  (plate_set_cost : ℕ)
  (num_plate_sets : ℕ)
  (money_given : ℕ)
  (change_received : ℕ)
  (h1 : table_cost = 50)
  (h2 : plate_set_cost = 20)
  (h3 : num_plate_sets = 2)
  (h4 : money_given = 130)
  (h5 : change_received = 4) :
  money_given - change_received - (table_cost + num_plate_sets * plate_set_cost) = 36 := by
  sorry

#check judes_chair_expenditure

end NUMINAMATH_CALUDE_judes_chair_expenditure_l3965_396527


namespace NUMINAMATH_CALUDE_balloon_difference_l3965_396572

theorem balloon_difference (yellow_balloons : ℕ) (total_balloons : ℕ) (school_balloons : ℕ) :
  yellow_balloons = 3414 →
  total_balloons % 10 = 0 →
  total_balloons / 10 = school_balloons →
  school_balloons = 859 →
  total_balloons > 2 * yellow_balloons →
  total_balloons - 2 * yellow_balloons = 1762 := by
  sorry

end NUMINAMATH_CALUDE_balloon_difference_l3965_396572


namespace NUMINAMATH_CALUDE_set_equality_implies_a_values_l3965_396546

theorem set_equality_implies_a_values (a : ℝ) : 
  ({0, -1, 2*a} : Set ℝ) = {a-1, -|a|, a+1} ↔ a = 1 ∨ a = -1 := by
  sorry

end NUMINAMATH_CALUDE_set_equality_implies_a_values_l3965_396546


namespace NUMINAMATH_CALUDE_fourth_term_is_27_l3965_396573

def S (n : ℕ) : ℤ := 4 * n^2 - n - 8

def a (n : ℕ) : ℤ := S n - S (n - 1)

theorem fourth_term_is_27 : a 4 = 27 := by
  sorry

end NUMINAMATH_CALUDE_fourth_term_is_27_l3965_396573


namespace NUMINAMATH_CALUDE_range_of_m_l3965_396589

theorem range_of_m (m : ℝ) : 
  m ≠ 0 → 
  (∀ x : ℝ, x ≥ 4 → (m^2 * x - 1) / (m * x + 1) < 0) → 
  m < -1/2 :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l3965_396589


namespace NUMINAMATH_CALUDE_pencil_distribution_problem_l3965_396517

theorem pencil_distribution_problem :
  ∃! n : ℕ, 10 ≤ n ∧ n < 100 ∧ 6 ∣ n ∧ 9 ∣ n ∧ n % 7 = 1 ∧ n = 36 :=
by sorry

end NUMINAMATH_CALUDE_pencil_distribution_problem_l3965_396517


namespace NUMINAMATH_CALUDE_derivative_f_at_one_l3965_396554

noncomputable def f (x : ℝ) : ℝ := Real.exp x * Real.log x

theorem derivative_f_at_one :
  deriv f 1 = Real.exp 1 := by sorry

end NUMINAMATH_CALUDE_derivative_f_at_one_l3965_396554


namespace NUMINAMATH_CALUDE_income_comparison_l3965_396592

theorem income_comparison (juan tim mary : ℝ) 
  (h1 : tim = juan * (1 - 0.4))
  (h2 : mary = tim * (1 + 0.7)) :
  mary = juan * 1.02 := by
sorry

end NUMINAMATH_CALUDE_income_comparison_l3965_396592


namespace NUMINAMATH_CALUDE_original_number_proof_l3965_396594

theorem original_number_proof (x : ℝ) : 
  (x * 1.125 - x * 0.75 = 30) → x = 80 := by
  sorry

end NUMINAMATH_CALUDE_original_number_proof_l3965_396594


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_range_l3965_396562

/-- The eccentricity of an ellipse satisfies the given range -/
theorem ellipse_eccentricity_range (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  let C := {p : ℝ × ℝ | (p.1^2 / a^2) + (p.2^2 / b^2) = 1}
  let B := (0, b)
  let e := Real.sqrt (1 - (b^2 / a^2))
  (∀ p ∈ C, Real.sqrt ((p.1 - B.1)^2 + (p.2 - B.2)^2) ≤ 2*b) →
  0 < e ∧ e ≤ Real.sqrt 2 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ellipse_eccentricity_range_l3965_396562


namespace NUMINAMATH_CALUDE_intersection_distance_l3965_396599

/-- Given a line y = kx - 2 intersecting a parabola y^2 = 8x at two points,
    if the x-coordinate of the midpoint of these points is 2,
    then the distance between these points is 2√15. -/
theorem intersection_distance (k : ℝ) : 
  (∃ x₁ x₂ y₁ y₂ : ℝ, 
    x₁ ≠ x₂ ∧
    y₁^2 = 8*x₁ ∧ y₂^2 = 8*x₂ ∧
    y₁ = k*x₁ - 2 ∧ y₂ = k*x₂ - 2 ∧
    (x₁ + x₂) / 2 = 2) →
  ∃ A B : ℝ × ℝ, 
    A.1 ≠ B.1 ∧
    A.2^2 = 8*A.1 ∧ B.2^2 = 8*B.1 ∧
    A.2 = k*A.1 - 2 ∧ B.2 = k*B.1 - 2 ∧
    (A.1 + B.1) / 2 = 2 ∧
    ((A.1 - B.1)^2 + (A.2 - B.2)^2)^(1/2 : ℝ) = 2 * (15^(1/2 : ℝ)) :=
by sorry

end NUMINAMATH_CALUDE_intersection_distance_l3965_396599


namespace NUMINAMATH_CALUDE_pyramid_base_side_length_l3965_396580

/-- A square pyramid with a hemisphere resting on its base -/
structure PyramidWithHemisphere where
  /-- Height of the pyramid -/
  pyramidHeight : ℝ
  /-- Radius of the hemisphere -/
  hemisphereRadius : ℝ
  /-- The hemisphere is tangent to all four faces and the base of the pyramid -/
  isTangent : Bool

/-- Calculate the side length of the square base of the pyramid -/
def calculateBaseSideLength (p : PyramidWithHemisphere) : ℝ :=
  sorry

/-- Theorem stating that for a pyramid of height 9 and hemisphere of radius 3,
    the side length of the base is 9 -/
theorem pyramid_base_side_length 
  (p : PyramidWithHemisphere) 
  (h1 : p.pyramidHeight = 9) 
  (h2 : p.hemisphereRadius = 3) 
  (h3 : p.isTangent = true) : 
  calculateBaseSideLength p = 9 := by
  sorry

end NUMINAMATH_CALUDE_pyramid_base_side_length_l3965_396580


namespace NUMINAMATH_CALUDE_pistachio_shell_percentage_l3965_396530

theorem pistachio_shell_percentage (total : ℕ) (shell_percent : ℚ) (opened_shells : ℕ) : 
  total = 80 →
  shell_percent = 95 / 100 →
  opened_shells = 57 →
  (opened_shells : ℚ) / (shell_percent * total) * 100 = 75 := by
sorry

end NUMINAMATH_CALUDE_pistachio_shell_percentage_l3965_396530


namespace NUMINAMATH_CALUDE_boris_bowls_l3965_396520

def candy_distribution (initial_candy : ℕ) (daughter_eats : ℕ) (boris_takes : ℕ) (remaining_in_bowl : ℕ) : ℕ :=
  let remaining_candy := initial_candy - daughter_eats
  let pieces_per_bowl := remaining_in_bowl + boris_takes
  remaining_candy / pieces_per_bowl

theorem boris_bowls :
  candy_distribution 100 8 3 20 = 4 := by
  sorry

end NUMINAMATH_CALUDE_boris_bowls_l3965_396520


namespace NUMINAMATH_CALUDE_direct_proportion_equation_l3965_396507

/-- A direct proportion function passing through (-1, 2) -/
def direct_proportion_through_neg1_2 (k : ℝ) (x : ℝ) : Prop :=
  k ≠ 0 ∧ 2 = k * (-1)

/-- The equation of the direct proportion function -/
def equation_of_direct_proportion (x : ℝ) : ℝ := -2 * x

theorem direct_proportion_equation :
  ∀ k : ℝ, direct_proportion_through_neg1_2 k x →
  ∀ x : ℝ, k * x = equation_of_direct_proportion x :=
sorry

end NUMINAMATH_CALUDE_direct_proportion_equation_l3965_396507


namespace NUMINAMATH_CALUDE_sweep_probability_is_one_third_l3965_396547

/-- Represents the positions of flies on a clock -/
inductive ClockPosition
  | twelve
  | three
  | six
  | nine

/-- Represents a time interval in minutes -/
def TimeInterval : ℕ := 20

/-- Calculates the number of favorable intervals where exactly two flies are swept -/
def favorableIntervals : ℕ := 4 * 5

/-- Total minutes in an hour -/
def totalMinutes : ℕ := 60

/-- The probability of sweeping exactly two flies in the given time interval -/
def sweepProbability : ℚ := favorableIntervals / totalMinutes

theorem sweep_probability_is_one_third :
  sweepProbability = 1 / 3 := by sorry

end NUMINAMATH_CALUDE_sweep_probability_is_one_third_l3965_396547


namespace NUMINAMATH_CALUDE_gloria_cypress_price_l3965_396553

/-- The amount Gloria gets for each cypress tree -/
def cypress_price : ℕ := sorry

theorem gloria_cypress_price :
  let cabin_price : ℕ := 129000
  let initial_cash : ℕ := 150
  let num_cypress : ℕ := 20
  let num_pine : ℕ := 600
  let num_maple : ℕ := 24
  let maple_price : ℕ := 300
  let pine_price : ℕ := 200
  let remaining_cash : ℕ := 350

  cypress_price * num_cypress + 
  pine_price * num_pine + 
  maple_price * num_maple + 
  initial_cash = 
  cabin_price + remaining_cash →
  
  cypress_price = 100 :=
by sorry

end NUMINAMATH_CALUDE_gloria_cypress_price_l3965_396553


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l3965_396513

theorem arithmetic_calculation : 12 - 10 + 8 / 2 * 5 + 4 - 6 * 3 + 1 = 9 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l3965_396513


namespace NUMINAMATH_CALUDE_positive_A_value_l3965_396584

-- Define the relation #
def hash (A B : ℝ) : ℝ := A^2 + B^2

-- Theorem statement
theorem positive_A_value (A : ℝ) (h : hash A 7 = 218) : A = 13 := by
  sorry

end NUMINAMATH_CALUDE_positive_A_value_l3965_396584


namespace NUMINAMATH_CALUDE_polygon_interior_exterior_angle_relation_l3965_396531

theorem polygon_interior_exterior_angle_relation (n : ℕ) : 
  (n - 2) * 180 = 2 * 360 → n = 6 := by
  sorry

end NUMINAMATH_CALUDE_polygon_interior_exterior_angle_relation_l3965_396531


namespace NUMINAMATH_CALUDE_polynomial_roots_theorem_l3965_396560

-- Define the polynomial
def P (a b c : ℂ) (x : ℂ) : ℂ := x^4 - a*x^3 - b*x + c

-- Define the set of solutions
def SolutionSet : Set (ℂ × ℂ × ℂ) :=
  {(a, 0, 0) | a : ℂ} ∪
  {((-1 + Complex.I * Real.sqrt 3) / 2, 1, (-1 + Complex.I * Real.sqrt 3) / 2),
   ((-1 - Complex.I * Real.sqrt 3) / 2, 1, (-1 - Complex.I * Real.sqrt 3) / 2),
   ((1 - Complex.I * Real.sqrt 3) / 2, -1, (1 + Complex.I * Real.sqrt 3) / 2),
   ((1 + Complex.I * Real.sqrt 3) / 2, -1, (1 - Complex.I * Real.sqrt 3) / 2)}

-- The main theorem
theorem polynomial_roots_theorem (a b c : ℂ) :
  (∃ d : ℂ, {a, b, c, d} ⊆ {x : ℂ | P a b c x = 0} ∧ (a, b, c) ∈ SolutionSet) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_roots_theorem_l3965_396560


namespace NUMINAMATH_CALUDE_square_sum_from_sum_and_product_l3965_396548

theorem square_sum_from_sum_and_product (a b : ℝ) 
  (h1 : a + b = 5) (h2 : a * b = 6) : a^2 + b^2 = 13 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_from_sum_and_product_l3965_396548


namespace NUMINAMATH_CALUDE_man_birth_year_l3965_396556

theorem man_birth_year (x : ℕ) (h1 : x > 0) (h2 : x^2 - 10 - x > 1850) 
  (h3 : x^2 - 10 - x < 1900) : x^2 - 10 - x = 1882 := by
  sorry

end NUMINAMATH_CALUDE_man_birth_year_l3965_396556


namespace NUMINAMATH_CALUDE_right_triangle_consecutive_sides_l3965_396596

theorem right_triangle_consecutive_sides : 
  ∀ (a b c : ℕ), 
  (a * a + b * b = c * c) →  -- Pythagorean theorem for right-angled triangle
  (b = a + 1 ∧ c = b + 1) →  -- Sides are consecutive natural numbers
  (a = 3 ∧ b = 4) →          -- Two sides are 3 and 4
  c = 5 := by               -- The third side is 5
sorry

end NUMINAMATH_CALUDE_right_triangle_consecutive_sides_l3965_396596


namespace NUMINAMATH_CALUDE_triangle_area_l3965_396510

theorem triangle_area (a b c : ℝ) (h1 : a = 3) (h2 : b = 2) (h3 : c = Real.sqrt 19) :
  let S := (1/2) * a * b * Real.sqrt (1 - ((a^2 + b^2 - c^2) / (2*a*b))^2)
  S = (3 * Real.sqrt 3) / 2 := by
sorry

end NUMINAMATH_CALUDE_triangle_area_l3965_396510


namespace NUMINAMATH_CALUDE_hyperbola_equation_l3965_396529

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 12 + y^2 / 3 = 1

-- Define the asymptote of the hyperbola
def asymptote (x y : ℝ) : Prop := y = (Real.sqrt 5 / 2) * x

-- Define the hyperbola C
def hyperbola_C (x y : ℝ) : Prop := x^2 / 4 - y^2 / 5 = 1

-- Theorem statement
theorem hyperbola_equation :
  ∀ x y : ℝ,
  (∃ f₁ f₂ : ℝ × ℝ, (∀ x y : ℝ, ellipse x y ↔ (x - f₁.1)^2 + y^2 = (x - f₂.1)^2 + y^2) ∧
                    (∀ x y : ℝ, hyperbola_C x y ↔ |(x - f₁.1)^2 + y^2| - |(x - f₂.1)^2 + y^2| = 2 * Real.sqrt 4)) →
  (∃ x y : ℝ, asymptote x y) →
  hyperbola_C x y :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l3965_396529


namespace NUMINAMATH_CALUDE_sum_of_angles_in_divided_hexagon_l3965_396516

-- Define a hexagon divided into two quadrilaterals
structure DividedHexagon where
  quad1 : Finset ℕ
  quad2 : Finset ℕ
  h1 : quad1.card = 4
  h2 : quad2.card = 4
  h3 : quad1 ∩ quad2 = ∅
  h4 : quad1 ∪ quad2 = Finset.range 8

-- Define the sum of angles in a quadrilateral (in degrees)
def quadrilateralAngleSum : ℕ := 360

-- Theorem statement
theorem sum_of_angles_in_divided_hexagon (h : DividedHexagon) :
  (h.quad1.sum (λ i => quadrilateralAngleSum / 4)) +
  (h.quad2.sum (λ i => quadrilateralAngleSum / 4)) = 720 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_angles_in_divided_hexagon_l3965_396516


namespace NUMINAMATH_CALUDE_quadratic_equation_coefficients_l3965_396574

theorem quadratic_equation_coefficients :
  ∃ (a b c : ℝ), 
    (∀ x, (2*x - 1)^2 = (x + 1)*(3*x + 4)) →
    (∀ x, a*x^2 + b*x + c = 0) ∧
    a = 1 ∧ b = -11 ∧ c = -3 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_coefficients_l3965_396574


namespace NUMINAMATH_CALUDE_novels_pages_per_book_l3965_396551

theorem novels_pages_per_book (novels_per_month : ℕ) (pages_per_year : ℕ) : 
  novels_per_month = 4 → pages_per_year = 9600 → 
  (pages_per_year / (novels_per_month * 12) : ℚ) = 200 := by
  sorry

end NUMINAMATH_CALUDE_novels_pages_per_book_l3965_396551


namespace NUMINAMATH_CALUDE_stock_income_calculation_l3965_396525

/-- Calculates the income derived from a stock investment --/
theorem stock_income_calculation
  (interest_rate : ℝ)
  (investment_amount : ℝ)
  (brokerage_rate : ℝ)
  (market_value_per_100 : ℝ)
  (h1 : interest_rate = 0.105)
  (h2 : investment_amount = 6000)
  (h3 : brokerage_rate = 0.0025)
  (h4 : market_value_per_100 = 83.08333333333334) :
  let brokerage_fee := investment_amount * brokerage_rate
  let actual_investment := investment_amount - brokerage_fee
  let num_units := actual_investment / market_value_per_100
  let face_value := num_units * 100
  let income := face_value * interest_rate
  income = 756 := by sorry

end NUMINAMATH_CALUDE_stock_income_calculation_l3965_396525


namespace NUMINAMATH_CALUDE_man_age_year_l3965_396597

theorem man_age_year (x : ℕ) (birth_year : ℕ) : 
  (1850 ≤ birth_year) ∧ (birth_year ≤ 1900) →
  (x^2 = birth_year + x) →
  (birth_year + x = 1892) := by
  sorry

end NUMINAMATH_CALUDE_man_age_year_l3965_396597


namespace NUMINAMATH_CALUDE_parallel_line_necessary_not_sufficient_l3965_396522

-- Define the type for planes and lines
variable (Plane Line : Type)

-- Define the parallel relation for planes and lines
variable (parallelPlanes : Plane → Plane → Prop)
variable (parallelLineToPlane : Line → Plane → Prop)

-- Define the subset relation for a line being in a plane
variable (lineInPlane : Line → Plane → Prop)

-- State the theorem
theorem parallel_line_necessary_not_sufficient
  (α β : Plane) (m : Line)
  (distinct : α ≠ β)
  (m_in_α : lineInPlane m α) :
  (parallelPlanes α β → parallelLineToPlane m β) ∧
  ¬(parallelLineToPlane m β → parallelPlanes α β) :=
sorry

end NUMINAMATH_CALUDE_parallel_line_necessary_not_sufficient_l3965_396522


namespace NUMINAMATH_CALUDE_vasya_has_winning_strategy_l3965_396557

/-- Represents a game state -/
structure GameState where
  board : List Nat
  currentPlayer : Bool  -- true for Petya, false for Vasya

/-- Checks if a list of numbers contains an arithmetic progression -/
def hasArithmeticProgression (numbers : List Nat) : Bool :=
  sorry

/-- Checks if a move is valid -/
def isValidMove (state : GameState) (move : Nat) : Bool :=
  move ≤ 2018 ∧ move ∉ state.board

/-- Applies a move to the game state -/
def applyMove (state : GameState) (move : Nat) : GameState :=
  { board := move :: state.board
  , currentPlayer := ¬state.currentPlayer }

/-- Represents a strategy for a player -/
def Strategy := GameState → Nat

/-- Checks if a strategy is winning for a player -/
def isWinningStrategy (strategy : Strategy) (player : Bool) : Prop :=
  ∀ (initialState : GameState),
    initialState.currentPlayer = player →
    ∃ (finalState : GameState),
      (finalState.board.length ≥ 3 ∧
       hasArithmeticProgression finalState.board) ∧
      finalState.currentPlayer = player

/-- The main theorem stating that Vasya (the second player) has a winning strategy -/
theorem vasya_has_winning_strategy :
  ∃ (strategy : Strategy), isWinningStrategy strategy false :=
sorry

end NUMINAMATH_CALUDE_vasya_has_winning_strategy_l3965_396557


namespace NUMINAMATH_CALUDE_f_derivative_at_zero_l3965_396586

noncomputable def f (x : ℝ) := (2 * x + 1) * Real.exp x

theorem f_derivative_at_zero : 
  deriv f 0 = 3 := by sorry

end NUMINAMATH_CALUDE_f_derivative_at_zero_l3965_396586


namespace NUMINAMATH_CALUDE_twenty_percent_women_without_plan_l3965_396595

/-- Represents a company with workers and their retirement plan status -/
structure Company where
  total_workers : ℕ
  workers_without_plan : ℕ
  men_with_plan : ℕ
  total_men : ℕ
  total_women : ℕ

/-- Conditions for the company -/
def company_conditions (c : Company) : Prop :=
  c.workers_without_plan = c.total_workers / 3 ∧
  c.men_with_plan = (c.total_workers - c.workers_without_plan) * 2 / 5 ∧
  c.total_men = 112 ∧
  c.total_women = 98 ∧
  c.total_workers = c.total_men + c.total_women

/-- The percentage of women without a retirement plan -/
def women_without_plan_percentage (c : Company) : ℚ :=
  let women_without_plan := c.workers_without_plan - (c.total_men - c.men_with_plan)
  (women_without_plan : ℚ) / c.workers_without_plan * 100

/-- Theorem stating that 20% of workers without a retirement plan are women -/
theorem twenty_percent_women_without_plan (c : Company) 
  (h : company_conditions c) : women_without_plan_percentage c = 20 := by
  sorry


end NUMINAMATH_CALUDE_twenty_percent_women_without_plan_l3965_396595


namespace NUMINAMATH_CALUDE_candy_problem_l3965_396577

/-- Given an initial amount of candy and the amounts eaten in two stages,
    calculate the remaining amount of candy. -/
def remaining_candy (initial : ℕ) (first_eaten : ℕ) (second_eaten : ℕ) : ℕ :=
  initial - (first_eaten + second_eaten)

/-- Theorem stating that given 36 initial pieces of candy, 
    after eating 17 and then 15 pieces, 4 pieces remain. -/
theorem candy_problem : remaining_candy 36 17 15 = 4 := by
  sorry

end NUMINAMATH_CALUDE_candy_problem_l3965_396577


namespace NUMINAMATH_CALUDE_min_area_rectangle_l3965_396524

theorem min_area_rectangle (l w : ℕ) : 
  (2 * l + 2 * w = 60) → 
  (l * w ≥ 29) :=
sorry

end NUMINAMATH_CALUDE_min_area_rectangle_l3965_396524


namespace NUMINAMATH_CALUDE_cube_vertex_distances_l3965_396575

/-- Given a cube with edge length a, after transformation by x₁₄ and x₄₅, 
    the sum of the squares of the distances between vertices 1 and 2, 1 and 4, and 1 and 5 
    is equal to 2a². -/
theorem cube_vertex_distances (a : ℝ) (x₁₄ x₄₅ : ℝ → ℝ → ℝ → ℝ × ℝ × ℝ) 
  (h : a > 0) : 
  ∃ (v₁ v₂ v₄ v₅ : ℝ × ℝ × ℝ), 
    let d₁₂ := ‖v₁ - v₂‖
    let d₁₄ := ‖v₁ - v₄‖
    let d₁₅ := ‖v₁ - v₅‖
    d₁₂^2 + d₁₄^2 + d₁₅^2 = 2 * a^2 :=
by
  sorry

end NUMINAMATH_CALUDE_cube_vertex_distances_l3965_396575


namespace NUMINAMATH_CALUDE_sequence_on_line_is_arithmetic_l3965_396578

/-- Given a sequence {a_n} where for any n ∈ ℕ*, the point P_n(n, a_n) lies on the line y = 2x + 1,
    prove that {a_n} is an arithmetic sequence with a common difference of 2. -/
theorem sequence_on_line_is_arithmetic (a : ℕ → ℝ) :
  (∀ n : ℕ, n > 0 → a n = 2 * n + 1) →
  ∃ (a₀ : ℝ), ∀ n : ℕ, a n = a₀ + 2 * n :=
by sorry

end NUMINAMATH_CALUDE_sequence_on_line_is_arithmetic_l3965_396578


namespace NUMINAMATH_CALUDE_smallest_consecutive_integer_l3965_396509

theorem smallest_consecutive_integer (a b c d e : ℤ) : 
  (a + b + c + d + e = 2015) → 
  (b = a + 1) → 
  (c = b + 1) → 
  (d = c + 1) → 
  (e = d + 1) → 
  (a = 401) := by
  sorry

end NUMINAMATH_CALUDE_smallest_consecutive_integer_l3965_396509


namespace NUMINAMATH_CALUDE_complex_sum_cube_ratio_l3965_396571

theorem complex_sum_cube_ratio (x y z : ℂ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (h_sum : x + y + z = 18)
  (h_diff_sq : (x - y)^2 + (x - z)^2 + (y - z)^2 = (x*y*z)/3) :
  (x^3 + y^3 + z^3) / (x*y*z) = 6 := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_cube_ratio_l3965_396571


namespace NUMINAMATH_CALUDE_absolute_value_simplification_l3965_396508

theorem absolute_value_simplification (x : ℝ) (h : x < 0) : |3*x + Real.sqrt (x^2)| = -2*x := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_simplification_l3965_396508


namespace NUMINAMATH_CALUDE_least_number_with_special_division_property_l3965_396566

theorem least_number_with_special_division_property : ∃ k : ℕ, 
  k > 0 ∧ 
  k / 5 = k % 34 + 8 ∧ 
  (∀ m : ℕ, m > 0 → m / 5 = m % 34 + 8 → k ≤ m) ∧
  k = 68 := by
sorry

end NUMINAMATH_CALUDE_least_number_with_special_division_property_l3965_396566


namespace NUMINAMATH_CALUDE_abs_neg_2023_l3965_396536

theorem abs_neg_2023 : |(-2023 : ℤ)| = 2023 := by
  sorry

end NUMINAMATH_CALUDE_abs_neg_2023_l3965_396536


namespace NUMINAMATH_CALUDE_point_b_not_on_curve_l3965_396587

/-- The equation of curve C -/
def curve_equation (x y a : ℝ) : Prop :=
  x^2 + y^2 + 6*a*x - 8*a*y = 0

/-- Point B does not lie on curve C -/
theorem point_b_not_on_curve (a : ℝ) : ¬ curve_equation (2*a) (4*a) a := by
  sorry

end NUMINAMATH_CALUDE_point_b_not_on_curve_l3965_396587


namespace NUMINAMATH_CALUDE_element_in_set_l3965_396581

def U : Set Nat := {1, 2, 3, 4, 5}

theorem element_in_set (M : Set Nat) (h : Set.compl M = {1, 3}) : 2 ∈ M := by
  sorry

end NUMINAMATH_CALUDE_element_in_set_l3965_396581


namespace NUMINAMATH_CALUDE_centroid_division_area_difference_l3965_396506

/-- Represents a triangle in 2D space -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- Calculates the area of a triangle -/
def triangleArea (t : Triangle) : ℝ := sorry

/-- Represents the centroid of a triangle -/
def centroid (t : Triangle) : ℝ × ℝ := sorry

/-- Represents a line passing through a point -/
structure Line where
  point : ℝ × ℝ
  direction : ℝ × ℝ

/-- Calculates the areas of two parts of a triangle divided by a line -/
def dividedAreas (t : Triangle) (l : Line) : ℝ × ℝ := sorry

/-- Theorem: The difference in areas of two parts of a triangle divided by a line
    through its centroid is not greater than 1/9 of the triangle's total area -/
theorem centroid_division_area_difference (t : Triangle) (l : Line) :
  let (A1, A2) := dividedAreas t l
  let G := centroid t
  l.point = G →
  |A1 - A2| ≤ (1/9) * triangleArea t :=
sorry

end NUMINAMATH_CALUDE_centroid_division_area_difference_l3965_396506


namespace NUMINAMATH_CALUDE_only_translation_preserves_pattern_l3965_396576

/-- Represents a shape in the pattern -/
inductive Shape
| Triangle
| Circle

/-- Represents the infinite alternating pattern -/
def Pattern := ℕ → Shape

/-- The alternating pattern of triangles and circles -/
def alternatingPattern : Pattern :=
  fun n => if n % 2 = 0 then Shape.Triangle else Shape.Circle

/-- Represents a transformation on the pattern -/
structure Transformation :=
  (apply : Pattern → Pattern)

/-- Rotation around a point on line ℓ under a triangle apex -/
def rotationTransformation : Transformation :=
  { apply := fun _ => alternatingPattern }

/-- Translation parallel to line ℓ -/
def translationTransformation : Transformation :=
  { apply := fun p n => p (n + 2) }

/-- Reflection across a line perpendicular to line ℓ -/
def reflectionTransformation : Transformation :=
  { apply := fun p n => p (n + 1) }

/-- Checks if a transformation preserves the pattern -/
def preservesPattern (t : Transformation) : Prop :=
  ∀ n, t.apply alternatingPattern n = alternatingPattern n

theorem only_translation_preserves_pattern :
  preservesPattern translationTransformation ∧
  ¬preservesPattern rotationTransformation ∧
  ¬preservesPattern reflectionTransformation :=
sorry

end NUMINAMATH_CALUDE_only_translation_preserves_pattern_l3965_396576


namespace NUMINAMATH_CALUDE_intersection_M_N_l3965_396533

def M : Set ℝ := {x | 0 < x ∧ x < 4}
def N : Set ℝ := {x | 1/3 ≤ x ∧ x ≤ 5}

theorem intersection_M_N : M ∩ N = {x | 1/3 ≤ x ∧ x < 4} := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l3965_396533


namespace NUMINAMATH_CALUDE_arthur_arrival_speed_l3965_396539

theorem arthur_arrival_speed :
  ∀ (distance : ℝ) (n : ℝ),
    (distance / 60 = distance / n + 1/12) →
    (distance / 90 = distance / n - 1/12) →
    n = 72 := by
  sorry

end NUMINAMATH_CALUDE_arthur_arrival_speed_l3965_396539


namespace NUMINAMATH_CALUDE_longest_side_l3965_396540

-- Define the triangle
def triangle (x : ℝ) := {a : ℝ // a = 7 ∨ a = x^2 + 4 ∨ a = 3*x + 1}

-- Define the perimeter condition
def perimeter_condition (x : ℝ) : Prop := 7 + (x^2 + 4) + (3*x + 1) = 45

-- State the theorem
theorem longest_side (x : ℝ) (h : perimeter_condition x) : 
  ∀ (side : triangle x), side.val ≤ x^2 + 4 :=
sorry

end NUMINAMATH_CALUDE_longest_side_l3965_396540


namespace NUMINAMATH_CALUDE_savings_account_theorem_l3965_396550

def initial_deposit : ℚ := 5 / 100
def daily_multiplier : ℚ := 3
def target_amount : ℚ := 500

def total_amount (n : ℕ) : ℚ :=
  initial_deposit * (1 - daily_multiplier^n) / (1 - daily_multiplier)

def exceeds_target (n : ℕ) : Prop :=
  total_amount n > target_amount

theorem savings_account_theorem :
  ∃ (n : ℕ), exceeds_target n ∧ ∀ (m : ℕ), m < n → ¬(exceeds_target m) :=
by sorry

end NUMINAMATH_CALUDE_savings_account_theorem_l3965_396550


namespace NUMINAMATH_CALUDE_first_meeting_cd_l3965_396503

-- Define the cars and their properties
structure Car where
  speed : ℝ
  direction : Bool  -- True for clockwise, False for counterclockwise

-- Define the race scenario
def race_scenario (a b c d : Car) : Prop :=
  a.direction ∧ b.direction ∧ ¬c.direction ∧ ¬d.direction ∧
  a.speed ≠ b.speed ∧ a.speed ≠ c.speed ∧ a.speed ≠ d.speed ∧
  b.speed ≠ c.speed ∧ b.speed ≠ d.speed ∧ c.speed ≠ d.speed ∧
  a.speed + c.speed = b.speed + d.speed ∧
  a.speed - b.speed = d.speed - c.speed

-- Define the meeting times
def first_meeting_ac_bd : ℝ := 7
def first_meeting_ab : ℝ := 53

-- Theorem statement
theorem first_meeting_cd 
  (a b c d : Car) 
  (h : race_scenario a b c d) :
  ∃ t : ℝ, t = first_meeting_ab :=
sorry

end NUMINAMATH_CALUDE_first_meeting_cd_l3965_396503


namespace NUMINAMATH_CALUDE_min_value_3x_plus_4y_l3965_396598

theorem min_value_3x_plus_4y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 3 * y = 5 * x * y) :
  3 * x + 4 * y ≥ 5 := by
sorry

end NUMINAMATH_CALUDE_min_value_3x_plus_4y_l3965_396598


namespace NUMINAMATH_CALUDE_yellow_square_area_percentage_l3965_396545

/-- Represents a square flag with a symmetric cross -/
structure SquareFlag where
  /-- Side length of the square flag -/
  side : ℝ
  /-- Width of each arm of the cross (equal to side length of yellow square) -/
  crossWidth : ℝ
  /-- Assumption that the cross width is positive and less than the flag side -/
  crossWidthValid : 0 < crossWidth ∧ crossWidth < side

/-- The area of the entire flag -/
def SquareFlag.area (flag : SquareFlag) : ℝ := flag.side ^ 2

/-- The area of the cross (including yellow center) -/
def SquareFlag.crossArea (flag : SquareFlag) : ℝ :=
  4 * flag.side * flag.crossWidth - 3 * flag.crossWidth ^ 2

/-- The area of the yellow square at the center -/
def SquareFlag.yellowArea (flag : SquareFlag) : ℝ := flag.crossWidth ^ 2

/-- Theorem stating that if the cross occupies 49% of the flag's area, 
    then the yellow square occupies 12.25% of the flag's area -/
theorem yellow_square_area_percentage (flag : SquareFlag) 
  (h : flag.crossArea = 0.49 * flag.area) : 
  flag.yellowArea / flag.area = 0.1225 := by
  sorry

end NUMINAMATH_CALUDE_yellow_square_area_percentage_l3965_396545


namespace NUMINAMATH_CALUDE_xy_sum_problem_l3965_396504

theorem xy_sum_problem (x y : ℕ) (hx : x > 0) (hy : y > 0) 
  (hx_bound : x < 15) (hy_bound : y < 15) (h_eq : x + y + x * y = 119) : 
  x + y = 21 ∨ x + y = 20 := by
sorry

end NUMINAMATH_CALUDE_xy_sum_problem_l3965_396504


namespace NUMINAMATH_CALUDE_product_of_sum_and_sum_of_squares_l3965_396512

theorem product_of_sum_and_sum_of_squares (m n : ℝ) 
  (h1 : m + n = 3) 
  (h2 : m^2 + n^2 = 3) : 
  m * n = 3 := by sorry

end NUMINAMATH_CALUDE_product_of_sum_and_sum_of_squares_l3965_396512


namespace NUMINAMATH_CALUDE_baker_initial_cakes_l3965_396505

/-- 
Given that Baker made some initial cakes, then made 149 more, 
sold 144, and still has 67 cakes, prove that he initially made 62 cakes.
-/
theorem baker_initial_cakes : 
  ∀ (initial : ℕ), 
  (initial + 149 : ℕ) - 144 = 67 → 
  initial = 62 := by
sorry

end NUMINAMATH_CALUDE_baker_initial_cakes_l3965_396505


namespace NUMINAMATH_CALUDE_property_P_lower_bound_l3965_396532

/-- Property P for a function f: ℝ → ℝ -/
def has_property_P (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, Real.sqrt (2 * f x) - Real.sqrt (2 * f x - f (2 * x)) ≥ 2

/-- The theorem stating that if f has property P, then f(x) ≥ 12 + 8√2 for all real x -/
theorem property_P_lower_bound (f : ℝ → ℝ) (h : has_property_P f) :
  ∀ x : ℝ, f x ≥ 12 + 8 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_property_P_lower_bound_l3965_396532
