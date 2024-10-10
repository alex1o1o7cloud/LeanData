import Mathlib

namespace probability_sum_16_three_dice_rolls_l98_9882

theorem probability_sum_16_three_dice_rolls :
  let die_faces : ℕ := 6
  let total_outcomes : ℕ := die_faces ^ 3
  let favorable_outcomes : ℕ := 6
  (favorable_outcomes : ℚ) / total_outcomes = 1 / 36 :=
by sorry

end probability_sum_16_three_dice_rolls_l98_9882


namespace quadratic_inequality_solution_l98_9878

/-- A quadratic function with real coefficients -/
def f (a b : ℝ) (x : ℝ) : ℝ := x^2 + a*x + b

/-- The theorem stating the value of c given the conditions -/
theorem quadratic_inequality_solution (a b m : ℝ) :
  (∀ x, f a b x ≥ 0) →
  (∃ c, ∀ x, f a b x < c ↔ m < x ∧ x < m + 6) →
  ∃ c, (∀ x, f a b x < c ↔ m < x ∧ x < m + 6) ∧ c = 9 :=
sorry

end quadratic_inequality_solution_l98_9878


namespace complete_square_and_calculate_l98_9895

theorem complete_square_and_calculate :
  ∀ m n p : ℝ,
  (∀ x : ℝ, 2 * x^2 - 8 * x + 19 = m * (x - n)^2 + p) →
  2017 + m * p - 5 * n = 2029 := by
sorry

end complete_square_and_calculate_l98_9895


namespace parcel_weight_l98_9822

theorem parcel_weight (x y z : ℝ) 
  (h1 : x + y = 110) 
  (h2 : y + z = 140) 
  (h3 : z + x = 130) : 
  x + y + z = 190 := by
sorry

end parcel_weight_l98_9822


namespace machine_output_for_68_l98_9846

def number_machine (x : ℕ) : ℕ := x + 15 - 6

theorem machine_output_for_68 : number_machine 68 = 77 := by
  sorry

end machine_output_for_68_l98_9846


namespace amount_ratio_problem_l98_9859

theorem amount_ratio_problem (total amount_p amount_q amount_r : ℚ) : 
  total = 1210 →
  amount_p + amount_q + amount_r = total →
  amount_p / amount_q = 5 / 4 →
  amount_r = 400 →
  amount_q / amount_r = 9 / 10 := by
sorry

end amount_ratio_problem_l98_9859


namespace cururu_jump_theorem_l98_9838

/-- Represents the number of jumps of each type -/
structure JumpCount where
  typeI : ℕ
  typeII : ℕ

/-- Checks if a given jump count reaches the target position -/
def reachesTarget (jumps : JumpCount) (targetEast targetNorth : ℤ) : Prop :=
  10 * jumps.typeI - 20 * jumps.typeII = targetEast ∧
  30 * jumps.typeI - 40 * jumps.typeII = targetNorth

theorem cururu_jump_theorem :
  (∃ jumps : JumpCount, reachesTarget jumps 190 950) ∧
  (¬ ∃ jumps : JumpCount, reachesTarget jumps 180 950) := by
  sorry

#check cururu_jump_theorem

end cururu_jump_theorem_l98_9838


namespace f_2019_equals_2_l98_9897

def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

theorem f_2019_equals_2 (f : ℝ → ℝ) 
  (h_even : is_even_function f)
  (h_period : ∀ x, f x = f (4 - x))
  (h_f_neg3 : f (-3) = 2) :
  f 2019 = 2 := by
  sorry

end f_2019_equals_2_l98_9897


namespace car_sale_profit_l98_9869

theorem car_sale_profit (P : ℝ) (h : P > 0) :
  let discount_rate := 0.2
  let profit_rate := 0.28000000000000004
  let purchase_price := P * (1 - discount_rate)
  let selling_price := P * (1 + profit_rate)
  let increase_rate := (selling_price - purchase_price) / purchase_price
  increase_rate = 0.6 := by sorry

end car_sale_profit_l98_9869


namespace prob_at_least_two_same_dice_l98_9804

-- Define the number of dice and sides
def num_dice : ℕ := 5
def num_sides : ℕ := 6

-- Define the total number of outcomes
def total_outcomes : ℕ := num_sides ^ num_dice

-- Define the number of outcomes with all different numbers
def all_different_outcomes : ℕ := num_sides * (num_sides - 1) * (num_sides - 2) * (num_sides - 3) * (num_sides - 4)

-- Define the probability of at least two dice showing the same number
def prob_at_least_two_same : ℚ := 1 - (all_different_outcomes : ℚ) / total_outcomes

-- Theorem statement
theorem prob_at_least_two_same_dice :
  prob_at_least_two_same = 7056 / 7776 :=
sorry

end prob_at_least_two_same_dice_l98_9804


namespace james_total_cost_l98_9890

/-- Calculates the total cost of James' vehicle purchases, registrations, and maintenance packages --/
def total_cost : ℕ :=
  let dirt_bike_cost : ℕ := 3 * 150
  let off_road_cost : ℕ := 4 * 300
  let atv_cost : ℕ := 2 * 450
  let moped_cost : ℕ := 5 * 200
  let scooter_cost : ℕ := 3 * 100

  let dirt_bike_reg : ℕ := 3 * 25
  let off_road_reg : ℕ := 4 * 25
  let atv_reg : ℕ := 2 * 30
  let moped_reg : ℕ := 5 * 15
  let scooter_reg : ℕ := 3 * 20

  let dirt_bike_maint : ℕ := 3 * 50
  let off_road_maint : ℕ := 4 * 75
  let atv_maint : ℕ := 2 * 100
  let moped_maint : ℕ := 5 * 60

  dirt_bike_cost + off_road_cost + atv_cost + moped_cost + scooter_cost +
  dirt_bike_reg + off_road_reg + atv_reg + moped_reg + scooter_reg +
  dirt_bike_maint + off_road_maint + atv_maint + moped_maint

theorem james_total_cost : total_cost = 5170 := by
  sorry

end james_total_cost_l98_9890


namespace sticker_distribution_theorem_l98_9872

/-- The number of ways to distribute n indistinguishable items into k distinguishable boxes --/
def distribute (n k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- The number of ways to distribute stickers among sheets --/
def distribute_stickers (total_stickers sheets : ℕ) : ℕ :=
  distribute (total_stickers - sheets) sheets

theorem sticker_distribution_theorem :
  distribute_stickers 12 5 = 330 := by sorry

end sticker_distribution_theorem_l98_9872


namespace remainder_of_product_l98_9819

def arithmetic_sequence (a₁ : ℕ) (d : ℕ) (n : ℕ) : List ℕ :=
  List.range n |>.map (fun i => a₁ + i * d)

def product_of_list (l : List ℕ) : ℕ :=
  l.foldl (· * ·) 1

theorem remainder_of_product (a₁ : ℕ) (d : ℕ) (n : ℕ) :
  a₁ = 3 ∧ d = 10 ∧ n = 20 →
  (product_of_list (arithmetic_sequence a₁ d n)) % 6 = 3 := by
  sorry

end remainder_of_product_l98_9819


namespace smallest_absolute_value_l98_9806

theorem smallest_absolute_value (x : ℝ) : |x| ≥ 0 ∧ (|x| = 0 ↔ x = 0) := by
  sorry

end smallest_absolute_value_l98_9806


namespace skylar_starting_donation_age_l98_9800

/-- The age at which Skylar started donating -/
def starting_age (annual_donation : ℕ) (total_donation : ℕ) (current_age : ℕ) : ℕ :=
  current_age - (total_donation / annual_donation)

/-- Theorem stating the age at which Skylar started donating -/
theorem skylar_starting_donation_age :
  starting_age 5000 105000 33 = 12 := by
  sorry

end skylar_starting_donation_age_l98_9800


namespace rectangle_area_with_inscribed_circle_l98_9894

/-- Given a rectangle with an inscribed circle of radius 6 and a length-to-width ratio of 3:1,
    prove that the area of the rectangle is 432. -/
theorem rectangle_area_with_inscribed_circle (r : ℝ) (ratio : ℝ) :
  r = 6 →
  ratio = 3 →
  let width := 2 * r
  let length := ratio * width
  width * length = 432 := by
  sorry

end rectangle_area_with_inscribed_circle_l98_9894


namespace pasta_preference_ratio_l98_9812

theorem pasta_preference_ratio (total_students : ℕ) 
  (fettuccine_preference : ℕ) (tortellini_preference : ℕ) 
  (penne_preference : ℕ) (fusilli_preference : ℕ) : 
  total_students = 800 →
  total_students = fettuccine_preference + tortellini_preference + penne_preference + fusilli_preference →
  fettuccine_preference = 2 * tortellini_preference →
  (fettuccine_preference : ℚ) / tortellini_preference = 2 := by
sorry

end pasta_preference_ratio_l98_9812


namespace conjunction_false_l98_9829

theorem conjunction_false (p q : Prop) (hp : p) (hq : ¬q) : ¬(p ∧ q) := by
  sorry

end conjunction_false_l98_9829


namespace multiple_of_p_capital_l98_9840

theorem multiple_of_p_capital (P Q R : ℚ) (total_profit : ℚ) 
  (h1 : ∃ x : ℚ, x * P = 6 * Q)
  (h2 : ∃ x : ℚ, x * P = 10 * R)
  (h3 : total_profit = 4650)
  (h4 : R * total_profit / (P + Q + R) = 900) :
  ∃ x : ℚ, x * P = 10 * R ∧ x = 10 := by sorry

end multiple_of_p_capital_l98_9840


namespace wax_calculation_l98_9861

/-- The amount of wax required for the feathers -/
def required_wax : ℕ := 166

/-- The additional amount of wax needed -/
def additional_wax : ℕ := 146

/-- The current amount of wax -/
def current_wax : ℕ := required_wax - additional_wax

theorem wax_calculation : current_wax = 20 := by
  sorry

end wax_calculation_l98_9861


namespace bicentric_quadrilateral_theorem_l98_9835

/-- A bicentric quadrilateral is a quadrilateral that has both an inscribed circle and a circumscribed circle. -/
structure BicentricQuadrilateral where
  /-- The radius of the inscribed circle -/
  r : ℝ
  /-- The radius of the circumscribed circle -/
  ρ : ℝ
  /-- The distance between the centers of the inscribed and circumscribed circles -/
  h : ℝ
  /-- Ensure r, ρ, and h are positive -/
  r_pos : r > 0
  ρ_pos : ρ > 0
  h_pos : h > 0
  /-- Ensure h is less than ρ (as the incenter must be inside the circumcircle) -/
  h_lt_ρ : h < ρ

/-- The main theorem about bicentric quadrilaterals -/
theorem bicentric_quadrilateral_theorem (q : BicentricQuadrilateral) :
  1 / (q.ρ + q.h)^2 + 1 / (q.ρ - q.h)^2 = 1 / q.r^2 := by
  sorry

end bicentric_quadrilateral_theorem_l98_9835


namespace line_equation_through_points_l98_9830

/-- Given two points P(3,2) and Q(4,7), prove that the equation 5x - y - 13 = 0
    represents the line passing through these points. -/
theorem line_equation_through_points (x y : ℝ) :
  let P : ℝ × ℝ := (3, 2)
  let Q : ℝ × ℝ := (4, 7)
  (5 * x - y - 13 = 0) ↔ 
    (∃ t : ℝ, (x, y) = ((1 - t) • P.1 + t • Q.1, (1 - t) • P.2 + t • Q.2)) :=
by sorry

end line_equation_through_points_l98_9830


namespace sequence_sum_implies_general_term_l98_9833

/-- Given a sequence (aₙ) with sum Sₙ = (2/3)aₙ + 1/3, prove aₙ = (-2)^(n-1) -/
theorem sequence_sum_implies_general_term (a : ℕ → ℝ) (S : ℕ → ℝ) 
  (h : ∀ n : ℕ, n ≥ 1 → S n = (2/3) * a n + 1/3) :
  ∀ n : ℕ, n ≥ 1 → a n = (-2)^(n-1) := by
  sorry

end sequence_sum_implies_general_term_l98_9833


namespace hyperbola_parallel_line_intersection_l98_9893

-- Define a hyperbola
structure Hyperbola where
  A : ℝ
  B : ℝ
  hAB : A ≠ 0 ∧ B ≠ 0

-- Define a line parallel to the asymptote
structure ParallelLine where
  m : ℝ
  hm : m ≠ 0

-- Theorem statement
theorem hyperbola_parallel_line_intersection (h : Hyperbola) (l : ParallelLine) :
  ∃! p : ℝ × ℝ, 
    (h.A * p.1)^2 - (h.B * p.2)^2 = 1 ∧ 
    h.A * p.1 - h.B * p.2 = l.m :=
sorry

end hyperbola_parallel_line_intersection_l98_9893


namespace g_zero_eq_one_l98_9854

/-- A function satisfying the given functional equation -/
def FunctionalEquation (g : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, g (x + y) = g x + g y - 1

/-- Theorem stating that g(0) = 1 for any function satisfying the functional equation -/
theorem g_zero_eq_one (g : ℝ → ℝ) (h : FunctionalEquation g) : g 0 = 1 := by
  sorry

end g_zero_eq_one_l98_9854


namespace square_of_negative_sqrt_two_equals_two_l98_9881

theorem square_of_negative_sqrt_two_equals_two :
  ((-Real.sqrt 2) ^ 2) = 2 := by
  sorry

end square_of_negative_sqrt_two_equals_two_l98_9881


namespace line_passes_through_fixed_point_l98_9828

/-- A line defined by the equation (m-1)x-y+2m+1=0 for any real number m -/
def line (m : ℝ) (x y : ℝ) : Prop :=
  (m - 1) * x - y + 2 * m + 1 = 0

/-- The fixed point (-2, 3) -/
def fixed_point : ℝ × ℝ := (-2, 3)

/-- Theorem stating that the line passes through the fixed point for any real m -/
theorem line_passes_through_fixed_point :
  ∀ m : ℝ, line m (fixed_point.1) (fixed_point.2) :=
by
  sorry

end line_passes_through_fixed_point_l98_9828


namespace unique_prime_sum_30_l98_9868

def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(m ∣ n)

theorem unique_prime_sum_30 :
  ∃! (A B C : ℕ), 
    isPrime A ∧ isPrime B ∧ isPrime C ∧
    A < 20 ∧ B < 20 ∧ C < 20 ∧
    A + B + C = 30 ∧
    A = 2 ∧ B = 11 ∧ C = 17 :=
by sorry

end unique_prime_sum_30_l98_9868


namespace quadratic_root_range_l98_9870

theorem quadratic_root_range (t : ℝ) :
  (∃ α β : ℝ, (3*t*α^2 + (3-7*t)*α + 2 = 0) ∧
              (3*t*β^2 + (3-7*t)*β + 2 = 0) ∧
              (0 < α) ∧ (α < 1) ∧ (1 < β) ∧ (β < 2)) →
  (5/4 < t) ∧ (t < 4) :=
by sorry

end quadratic_root_range_l98_9870


namespace office_paper_duration_l98_9845

/-- The number of days printer paper will last given the number of packs, sheets per pack, and daily usage. -/
def printer_paper_duration (packs : ℕ) (sheets_per_pack : ℕ) (daily_usage : ℕ) : ℕ :=
  (packs * sheets_per_pack) / daily_usage

/-- Theorem stating that two packs of 240-sheet paper will last 6 days when using 80 sheets per day. -/
theorem office_paper_duration :
  printer_paper_duration 2 240 80 = 6 := by
  sorry

end office_paper_duration_l98_9845


namespace traffic_accident_emergency_number_correct_l98_9849

def emergency_numbers : List ℕ := [122, 110, 120, 114]

def traffic_accident_emergency_number : ℕ := 122

theorem traffic_accident_emergency_number_correct :
  traffic_accident_emergency_number ∈ emergency_numbers ∧
  traffic_accident_emergency_number = 122 := by
  sorry

end traffic_accident_emergency_number_correct_l98_9849


namespace lunks_needed_for_two_dozen_oranges_l98_9817

/-- Exchange rate between lunks and kunks -/
def lunks_to_kunks_rate : ℚ := 4 / 2

/-- Exchange rate between kunks and oranges -/
def kunks_to_oranges_rate : ℚ := 3 / 6

/-- Number of oranges in two dozen -/
def two_dozen : ℕ := 24

/-- The number of lunks required to purchase two dozen oranges -/
def lunks_for_two_dozen : ℕ := 24

theorem lunks_needed_for_two_dozen_oranges :
  (two_dozen : ℚ) / kunks_to_oranges_rate * lunks_to_kunks_rate = lunks_for_two_dozen := by
  sorry

end lunks_needed_for_two_dozen_oranges_l98_9817


namespace box_cost_is_111_kopecks_l98_9855

/-- The cost of a box of matches in kopecks -/
def box_cost : ℕ := sorry

/-- Nine boxes cost more than 9 rubles but less than 10 rubles -/
axiom nine_boxes_cost : 900 < 9 * box_cost ∧ 9 * box_cost < 1000

/-- Ten boxes cost more than 11 rubles but less than 12 rubles -/
axiom ten_boxes_cost : 1100 < 10 * box_cost ∧ 10 * box_cost < 1200

/-- The cost of one box of matches is 1 ruble 11 kopecks -/
theorem box_cost_is_111_kopecks : box_cost = 111 := by sorry

end box_cost_is_111_kopecks_l98_9855


namespace projection_vector_l98_9834

/-- Given a vector b and the dot product of vectors a and b, 
    prove that the projection of a onto b is as calculated. -/
theorem projection_vector (a b : ℝ × ℝ) (h : a • b = 10) 
    (hb : b = (3, 4)) : 
  (a • b / (b • b)) • b = (6/5, 8/5) := by
  sorry

end projection_vector_l98_9834


namespace parabola_translation_theorem_l98_9877

/-- Represents a parabola in the form y = ax² + bx + c --/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Translates a parabola vertically --/
def translate_vertical (p : Parabola) (d : ℝ) : Parabola :=
  { a := p.a, b := p.b, c := p.c - d }

/-- Translates a parabola horizontally --/
def translate_horizontal (p : Parabola) (d : ℝ) : Parabola :=
  { a := p.a, b := -2 * p.a * d + p.b, c := p.a * d^2 - p.b * d + p.c }

theorem parabola_translation_theorem :
  let original := Parabola.mk 3 0 0
  let down_3 := translate_vertical original 3
  let right_2 := translate_horizontal down_3 2
  right_2 = Parabola.mk 3 (-12) 9 := by sorry

end parabola_translation_theorem_l98_9877


namespace b_has_property_P_l98_9899

-- Define property P for a sequence
def has_property_P (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q ≠ 0 ∧ ∀ n : ℕ, (a (n + 1) + a (n + 2)) = q * (a n + a (n + 1))

-- Define the sequence b_n
def b (n : ℕ) : ℝ := 2^n + (-1)^n

-- Theorem statement
theorem b_has_property_P : has_property_P b := by
  sorry

end b_has_property_P_l98_9899


namespace quadratic_function_m_condition_l98_9808

/-- A function f: ℝ → ℝ is quadratic if it can be written as f(x) = ax² + bx + c where a ≠ 0 -/
def is_quadratic (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The function y = (m+1)x² + 2x + 1 -/
def f (m : ℝ) : ℝ → ℝ := λ x ↦ (m + 1) * x^2 + 2 * x + 1

theorem quadratic_function_m_condition :
  ∀ m : ℝ, is_quadratic (f m) ↔ m ≠ -1 := by sorry

end quadratic_function_m_condition_l98_9808


namespace floor_sqrt_eight_count_l98_9891

theorem floor_sqrt_eight_count : 
  (Finset.filter (fun x : ℕ => ⌊Real.sqrt x⌋ = 8) (Finset.range 81)).card = 17 := by
  sorry

end floor_sqrt_eight_count_l98_9891


namespace molly_has_three_brothers_l98_9889

/-- Represents the problem of determining Molly's number of brothers --/
def MollysBrothers (cost_per_package : ℕ) (num_parents : ℕ) (total_cost : ℕ) : Prop :=
  ∃ (num_brothers : ℕ),
    cost_per_package * (num_parents + num_brothers + num_brothers + 2 * num_brothers) = total_cost

/-- Theorem stating that Molly has 3 brothers given the problem conditions --/
theorem molly_has_three_brothers :
  MollysBrothers 5 2 70 → ∃ (num_brothers : ℕ), num_brothers = 3 := by
  sorry

end molly_has_three_brothers_l98_9889


namespace subset_condition_1_subset_condition_2_l98_9871

-- Define set A
def A : Set ℝ := {x | x^2 - 2*x - 8 = 0}

-- Define set B
def B (a : ℝ) : Set ℝ := {x | x^2 + a*x + a^2 - 12 = 0}

-- Theorem for part 1
theorem subset_condition_1 : A ⊆ B a → a = -2 := by sorry

-- Theorem for part 2
theorem subset_condition_2 : B a ⊆ A → a ≥ 4 ∨ a < -4 ∨ a = -2 := by sorry

end subset_condition_1_subset_condition_2_l98_9871


namespace average_difference_l98_9807

def num_students : ℕ := 120
def num_teachers : ℕ := 6
def class_enrollments : List ℕ := [60, 30, 20, 5, 3, 2]

def t : ℚ := (class_enrollments.sum : ℚ) / num_teachers

def s : ℚ := (class_enrollments.map (λ n => n * n)).sum / num_students

theorem average_difference : t - s = -21151/1000 := by sorry

end average_difference_l98_9807


namespace worksheet_problems_l98_9851

theorem worksheet_problems (total_worksheets graded_worksheets remaining_problems : ℕ) 
  (h1 : total_worksheets = 9)
  (h2 : graded_worksheets = 5)
  (h3 : remaining_problems = 16) :
  (total_worksheets - graded_worksheets) * (remaining_problems / (total_worksheets - graded_worksheets)) = 4 := by
  sorry

end worksheet_problems_l98_9851


namespace max_value_x_plus_2y_l98_9876

theorem max_value_x_plus_2y (x y : ℝ) (h : 3 * (x^2 + y^2) = x + y) :
  x + 2*y ≤ Real.sqrt (5/18) + 1/2 := by
  sorry

end max_value_x_plus_2y_l98_9876


namespace distribution_ratio_l98_9874

/-- Represents the distribution of money among four people --/
structure Distribution where
  p : ℚ  -- Amount received by P
  q : ℚ  -- Amount received by Q
  r : ℚ  -- Amount received by R
  s : ℚ  -- Amount received by S

/-- Theorem stating the ratio of P's amount to Q's amount --/
theorem distribution_ratio (d : Distribution) : 
  d.p + d.q + d.r + d.s = 1000 →  -- Total amount condition
  d.s = 4 * d.r →                 -- S gets 4 times R's amount
  d.q = d.r →                     -- Q and R receive equal amounts
  d.s - d.p = 250 →               -- Difference between S and P
  d.p / d.q = 2 / 1 := by          -- Ratio of P's amount to Q's amount
sorry


end distribution_ratio_l98_9874


namespace min_value_trig_expression_l98_9887

theorem min_value_trig_expression (x : ℝ) : 
  Real.sin x ^ 4 + 2 * Real.cos x ^ 4 + Real.sin x ^ 2 * Real.cos x ^ 2 ≥ 3 / 16 := by
  sorry

end min_value_trig_expression_l98_9887


namespace b_minus_a_value_l98_9803

theorem b_minus_a_value (a b : ℝ) (h1 : |a| = 1) (h2 : |b| = 3) (h3 : a < b) :
  b - a = 2 ∨ b - a = 4 := by
  sorry

end b_minus_a_value_l98_9803


namespace system_solution_l98_9862

theorem system_solution : ∃ (x y : ℝ), (2 * x - y = 5) ∧ (7 * x - 3 * y = 20) ∧ (x = 5) ∧ (y = 5) := by
  sorry

end system_solution_l98_9862


namespace math_team_selection_l98_9831

theorem math_team_selection (boys girls : ℕ) (h1 : boys = 10) (h2 : girls = 12) :
  (Nat.choose boys 5) * (Nat.choose girls 3) = 55440 := by
  sorry

end math_team_selection_l98_9831


namespace locus_is_ray_l98_9805

/-- The locus of points P satisfying |PM| - |PN| = 4, where M(-2,0) and N(2,0) are fixed points -/
def locus_of_P (P : ℝ × ℝ) : Prop :=
  let M : ℝ × ℝ := (-2, 0)
  let N : ℝ × ℝ := (2, 0)
  Real.sqrt ((P.1 + 2)^2 + P.2^2) - Real.sqrt ((P.1 - 2)^2 + P.2^2) = 4

/-- The ray starting from the midpoint of MN and extending to the right -/
def ray_from_midpoint (P : ℝ × ℝ) : Prop :=
  P.1 ≥ 0 ∧ P.2 = 0

theorem locus_is_ray :
  ∀ P, locus_of_P P ↔ ray_from_midpoint P :=
sorry

end locus_is_ray_l98_9805


namespace tax_savings_calculation_l98_9841

/-- Calculates the differential savings when tax rate is lowered -/
def differential_savings (income : ℝ) (old_rate new_rate : ℝ) : ℝ :=
  income * (old_rate - new_rate)

/-- Theorem: The differential savings for a taxpayer with an annual income
    of $42,400, when the tax rate is reduced from 42% to 32%, is $4,240 -/
theorem tax_savings_calculation :
  differential_savings 42400 0.42 0.32 = 4240 := by
  sorry

end tax_savings_calculation_l98_9841


namespace quadratic_root_implies_k_value_l98_9810

theorem quadratic_root_implies_k_value (k : ℝ) : 
  (∃ x : ℝ, x^2 - 2*x + k - 1 = 0 ∧ x = -1) → k = -2 := by
  sorry

end quadratic_root_implies_k_value_l98_9810


namespace shuai_fen_solution_l98_9888

/-- Represents the "Shuai Fen" distribution system -/
structure ShuaiFen where
  a : ℝ
  x : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  h_a_pos : a > 0
  h_c : c = 36
  h_bd : b + d = 75
  h_shuai_fen_b : (b - c) / b = x
  h_shuai_fen_c : (c - d) / c = x
  h_shuai_fen_a : (a - b) / a = x
  h_total : a = b + c + d

/-- The "Shuai Fen" problem solution -/
theorem shuai_fen_solution (sf : ShuaiFen) : sf.x = 0.25 ∧ sf.a = 175 := by
  sorry

end shuai_fen_solution_l98_9888


namespace inequality_proof_l98_9814

theorem inequality_proof (a b c : ℝ) (h1 : a > b) (h2 : b > c) (h3 : a + b + c = 0) : a * b > a * c := by
  sorry

end inequality_proof_l98_9814


namespace problem_1_problem_2_l98_9826

-- Problem 1
theorem problem_1 (x : ℝ) : 
  x / (2 * x - 3) + 5 / (3 - 2 * x) = 4 ↔ x = 1 :=
sorry

-- Problem 2
theorem problem_2 : 
  ¬∃ (x : ℝ), (x + 1) / (x - 1) - 4 / (x^2 - 1) = 1 :=
sorry

end problem_1_problem_2_l98_9826


namespace intersection_A_B_union_A_B_range_of_a_l98_9832

-- Define the sets A, B, and C
def A : Set ℝ := {x | -1 ≤ x ∧ x ≤ 3}
def B : Set ℝ := {x | 0 < x ∧ x < 4}
def C (a : ℝ) : Set ℝ := {x | x < a}

-- Theorem for the intersection of A and B
theorem intersection_A_B : A ∩ B = {x | 0 < x ∧ x ≤ 3} := by sorry

-- Theorem for the union of A and B
theorem union_A_B : A ∪ B = {x | -1 ≤ x ∧ x < 4} := by sorry

-- Theorem for the range of a when B is a subset of C
theorem range_of_a (h : B ⊆ C a) : a ≥ 4 := by sorry

end intersection_A_B_union_A_B_range_of_a_l98_9832


namespace negation_of_universal_proposition_l98_9884

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 > 0) ↔ (∃ x : ℝ, x^2 ≤ 0) := by
  sorry

end negation_of_universal_proposition_l98_9884


namespace binomial_expansion_m_value_l98_9815

/-- Given a binomial expansion (mx+1)^n where the 5th term has the largest
    coefficient and the coefficient of x^3 is 448, prove that m = 2 -/
theorem binomial_expansion_m_value (m : ℝ) (n : ℕ) : 
  (∃ k : ℕ, k = 5 ∧ 
    ∀ j : ℕ, j ≤ n + 1 → Nat.choose n (j - 1) * m^(j - 1) ≤ Nat.choose n (k - 1) * m^(k - 1)) ∧
  Nat.choose n 3 * m^3 = 448 →
  m = 2 := by
  sorry

end binomial_expansion_m_value_l98_9815


namespace cellphone_cost_correct_l98_9864

/-- The cost of a single cellphone before discount -/
def cellphone_cost : ℝ := 800

/-- The number of cellphones purchased -/
def num_cellphones : ℕ := 2

/-- The discount rate applied to the total cost -/
def discount_rate : ℝ := 0.05

/-- The final price paid after the discount -/
def final_price : ℝ := 1520

/-- Theorem stating that the given cellphone cost satisfies the conditions -/
theorem cellphone_cost_correct : 
  (num_cellphones : ℝ) * cellphone_cost * (1 - discount_rate) = final_price := by
  sorry

end cellphone_cost_correct_l98_9864


namespace saturday_zoo_visitors_l98_9847

theorem saturday_zoo_visitors (friday_visitors : ℕ) (saturday_multiplier : ℕ) : 
  friday_visitors = 1250 →
  saturday_multiplier = 3 →
  friday_visitors * saturday_multiplier = 3750 :=
by
  sorry

end saturday_zoo_visitors_l98_9847


namespace race_distance_l98_9898

/-- Represents the race scenario -/
structure Race where
  distance : ℝ
  time_A : ℝ
  time_B : ℝ
  speed_A : ℝ
  speed_B : ℝ

/-- The race conditions -/
def race_conditions (r : Race) : Prop :=
  r.time_A = 33 ∧
  r.speed_A = r.distance / r.time_A ∧
  r.speed_B = (r.distance - 35) / r.time_A ∧
  r.speed_B = 35 / 7 ∧
  r.time_B = r.time_A + 7

/-- The theorem stating that the race distance is 200 meters -/
theorem race_distance (r : Race) (h : race_conditions r) : r.distance = 200 :=
sorry

end race_distance_l98_9898


namespace solve_equation_l98_9836

/-- Proves that the solution to the equation 4.7 × 13.26 + 4.7 × 9.43 + 4.7 × x = 470 is x = 77.31 -/
theorem solve_equation : 
  ∃ x : ℝ, 4.7 * 13.26 + 4.7 * 9.43 + 4.7 * x = 470 ∧ x = 77.31 := by
  sorry

end solve_equation_l98_9836


namespace adult_ticket_price_l98_9866

/-- Represents the price of tickets and sales data for a theater --/
structure TheaterSales where
  adult_price : ℚ
  child_price : ℚ
  total_revenue : ℚ
  total_tickets : ℕ
  adult_tickets : ℕ

/-- Theorem stating that the adult ticket price is $10.50 given the conditions --/
theorem adult_ticket_price (sale : TheaterSales)
  (h1 : sale.child_price = 5)
  (h2 : sale.total_revenue = 236)
  (h3 : sale.total_tickets = 34)
  (h4 : sale.adult_tickets = 12)
  : sale.adult_price = 21/2 := by
  sorry

#eval (21 : ℚ) / 2  -- To verify that 21/2 is indeed 10.50

end adult_ticket_price_l98_9866


namespace sqrt2_minus_2_properties_l98_9823

theorem sqrt2_minus_2_properties :
  let x : ℝ := Real.sqrt 2 - 2
  (- x = 2 - Real.sqrt 2) ∧ (|x| = 2 - Real.sqrt 2) := by sorry

end sqrt2_minus_2_properties_l98_9823


namespace abs_ratio_equals_sqrt_eleven_sevenths_l98_9820

theorem abs_ratio_equals_sqrt_eleven_sevenths (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) 
  (h : a^2 + b^2 = 9*a*b) : 
  |((a+b)/(a-b))| = Real.sqrt (11/7) := by
sorry

end abs_ratio_equals_sqrt_eleven_sevenths_l98_9820


namespace longest_line_segment_in_pie_slice_l98_9853

theorem longest_line_segment_in_pie_slice (d : ℝ) (n : ℕ) (h_d : d = 16) (h_n : n = 4) : 
  let r := d / 2
  let θ := 2 * Real.pi / n
  let m := 2 * r * Real.sin (θ / 2)
  m ^ 2 = 128 := by sorry

end longest_line_segment_in_pie_slice_l98_9853


namespace rectangle_perimeter_bound_l98_9875

/-- The curve W defined by y = x^2 + 1/4 -/
def W : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = p.1^2 + 1/4}

/-- A rectangle with vertices as points in ℝ × ℝ -/
structure Rectangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ

/-- Perimeter of a rectangle -/
def perimeter (r : Rectangle) : ℝ :=
  2 * (dist r.A r.B + dist r.B r.C)

/-- Three vertices of the rectangle are on W -/
def three_vertices_on_W (r : Rectangle) : Prop :=
  (r.A ∈ W ∧ r.B ∈ W ∧ r.C ∈ W) ∨
  (r.A ∈ W ∧ r.B ∈ W ∧ r.D ∈ W) ∨
  (r.A ∈ W ∧ r.C ∈ W ∧ r.D ∈ W) ∨
  (r.B ∈ W ∧ r.C ∈ W ∧ r.D ∈ W)

theorem rectangle_perimeter_bound (r : Rectangle) 
  (h : three_vertices_on_W r) : 
  perimeter r > 3 * Real.sqrt 3 := by
  sorry

end rectangle_perimeter_bound_l98_9875


namespace ice_cream_permutations_l98_9858

/-- The number of permutations of n distinct elements -/
def permutations (n : ℕ) : ℕ := Nat.factorial n

/-- There are 4 distinct ice cream flavors -/
def num_flavors : ℕ := 4

/-- Theorem: The number of permutations of 4 distinct elements is 24 -/
theorem ice_cream_permutations : permutations num_flavors = 24 := by
  sorry

end ice_cream_permutations_l98_9858


namespace smallest_number_l98_9860

theorem smallest_number (a b c d : ℤ) (ha : a = -1) (hb : b = -2) (hc : c = 1) (hd : d = 2) :
  b ≤ a ∧ b ≤ c ∧ b ≤ d :=
by sorry

end smallest_number_l98_9860


namespace reverse_digits_square_diff_l98_9824

/-- Given two-digit integers x and y where y is the reverse of x, and x^2 - y^2 = m^2 for some positive integer m, prove that x + y + m = 154 -/
theorem reverse_digits_square_diff (x y m : ℕ) : 
  (10 ≤ x ∧ x < 100) →  -- x is a two-digit integer
  (10 ≤ y ∧ y < 100) →  -- y is a two-digit integer
  (∃ (a b : ℕ), x = 10 * a + b ∧ y = 10 * b + a ∧ 0 < a ∧ a < 10 ∧ 0 < b ∧ b < 10) →  -- y is obtained by reversing the digits of x
  (x^2 - y^2 = m^2) →  -- x^2 - y^2 = m^2
  (0 < m) →  -- m is positive
  (x + y + m = 154) := by
sorry

end reverse_digits_square_diff_l98_9824


namespace square_less_than_triple_l98_9801

theorem square_less_than_triple (x : ℤ) : x^2 < 3*x ↔ x = 1 ∨ x = 2 := by
  sorry

end square_less_than_triple_l98_9801


namespace fifth_root_unity_product_l98_9880

theorem fifth_root_unity_product (r : ℂ) (h1 : r^5 = 1) (h2 : r ≠ 1) :
  (r - 1) * (r^2 - 1) * (r^3 - 1) * (r^4 - 1) = 5 := by
  sorry

end fifth_root_unity_product_l98_9880


namespace candy_mixture_cost_l98_9886

/-- Given a mixture of two types of candy, prove the cost of the second type. -/
theorem candy_mixture_cost
  (first_candy_weight : ℝ)
  (first_candy_cost : ℝ)
  (second_candy_weight : ℝ)
  (mixture_cost : ℝ)
  (h1 : first_candy_weight = 20)
  (h2 : first_candy_cost = 10)
  (h3 : second_candy_weight = 80)
  (h4 : mixture_cost = 6)
  : (((first_candy_weight + second_candy_weight) * mixture_cost
     - first_candy_weight * first_candy_cost) / second_candy_weight) = 5 := by
  sorry

#check candy_mixture_cost

end candy_mixture_cost_l98_9886


namespace valid_arrangements_five_people_l98_9848

/-- The number of people in the arrangement -/
def n : ℕ := 5

/-- The number of ways to arrange n people such that at least one of two specific people (A and B) is at one of the ends -/
def validArrangements (n : ℕ) : ℕ :=
  n.factorial - (n - 2).factorial * (n - 2).factorial

theorem valid_arrangements_five_people :
  validArrangements n = 84 := by
  sorry

end valid_arrangements_five_people_l98_9848


namespace kim_laura_difference_l98_9892

/-- Proves that Kim paints 3 fewer tiles per minute than Laura -/
theorem kim_laura_difference (don ken laura kim : ℕ) : 
  don = 3 →  -- Don paints 3 tiles per minute
  ken = don + 2 →  -- Ken paints 2 more tiles than Don per minute
  laura = 2 * ken →  -- Laura paints twice as many tiles as Ken per minute
  don + ken + laura + kim = 25 →  -- They paint 375 tiles in 15 minutes (375 / 15 = 25)
  laura - kim = 3 := by  -- Kim paints 3 fewer tiles than Laura per minute
sorry

end kim_laura_difference_l98_9892


namespace catherine_caps_proof_l98_9842

/-- The number of bottle caps Nicholas starts with -/
def initial_caps : ℕ := 8

/-- The number of bottle caps Nicholas ends up with -/
def final_caps : ℕ := 93

/-- The number of bottle caps Catherine gave to Nicholas -/
def catherine_caps : ℕ := final_caps - initial_caps

theorem catherine_caps_proof : catherine_caps = 85 := by
  sorry

end catherine_caps_proof_l98_9842


namespace d_eq_l_l98_9883

/-- The number of partitions of n into distinct summands -/
def d (n : ℕ) : ℕ := sorry

/-- The number of partitions of n into odd summands -/
def l (n : ℕ) : ℕ := sorry

/-- The generating function for d(n) -/
noncomputable def d_gen_fun (x : ℝ) : ℝ := ∑' n, d n * x^n

/-- The generating function for l(n) -/
noncomputable def l_gen_fun (x : ℝ) : ℝ := ∑' n, l n * x^n

/-- The product representation of d_gen_fun -/
noncomputable def d_prod (x : ℝ) : ℝ := ∏' k, (1 + x^k)

/-- The product representation of l_gen_fun -/
noncomputable def l_prod (x : ℝ) : ℝ := ∏' k, (1 - x^(2*k+1))⁻¹

/-- The main theorem: d(n) = l(n) for all n -/
theorem d_eq_l : ∀ n : ℕ, d n = l n := by sorry

/-- d(0) = l(0) = 1 -/
axiom d_zero : d 0 = 1
axiom l_zero : l 0 = 1

/-- The generating functions are equal to their product representations -/
axiom d_gen_fun_eq_prod : d_gen_fun = d_prod
axiom l_gen_fun_eq_prod : l_gen_fun = l_prod

end d_eq_l_l98_9883


namespace k_range_for_empty_intersection_l98_9802

-- Define the sets A and B
def A (k : ℝ) : Set ℝ := {x | k * x^2 - (k + 3) * x - 1 ≥ 0}
def B : Set ℝ := {y | ∃ x, y = 2 * x + 1}

-- State the theorem
theorem k_range_for_empty_intersection :
  (∀ k : ℝ, (A k ∩ B = ∅)) ↔ (∀ k : ℝ, -9 < k ∧ k < -1) :=
sorry

end k_range_for_empty_intersection_l98_9802


namespace swimming_pool_length_l98_9816

theorem swimming_pool_length 
  (width : ℝ) 
  (water_removed : ℝ) 
  (water_level_lowered : ℝ) 
  (cubic_foot_to_gallon : ℝ) :
  width = 20 →
  water_removed = 4500 →
  water_level_lowered = 0.5 →
  cubic_foot_to_gallon = 7.5 →
  ∃ (length : ℝ), length = 60 ∧ 
    water_removed / cubic_foot_to_gallon = length * width * water_level_lowered :=
by
  sorry

end swimming_pool_length_l98_9816


namespace f_2008_eq_zero_l98_9837

def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem f_2008_eq_zero
  (f : ℝ → ℝ)
  (h_odd : is_odd_function f)
  (h_f2 : f 2 = 0)
  (h_periodic : ∀ x, f (x + 4) = f x + f 4) :
  f 2008 = 0 := by
  sorry

end f_2008_eq_zero_l98_9837


namespace speed_conversion_l98_9857

/-- Conversion factor from m/s to km/h -/
def mps_to_kmph : ℝ := 3.6

/-- The given speed in m/s -/
def given_speed_mps : ℝ := 15.556799999999999

/-- The speed in km/h we want to prove -/
def speed_kmph : ℝ := 56.00448

theorem speed_conversion : given_speed_mps * mps_to_kmph = speed_kmph := by
  sorry

end speed_conversion_l98_9857


namespace cyclic_fraction_inequality_l98_9811

theorem cyclic_fraction_inequality (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  x / (y * z) + y / (z * x) + z / (x * y) ≥ 1 / x + 1 / y + 1 / z := by
  sorry

end cyclic_fraction_inequality_l98_9811


namespace delegate_seating_probability_l98_9818

/-- Represents the number of delegates -/
def total_delegates : ℕ := 12

/-- Represents the number of countries -/
def num_countries : ℕ := 3

/-- Represents the number of delegates per country -/
def delegates_per_country : ℕ := 4

/-- Calculates the probability of each delegate sitting next to at least one delegate from another country -/
def seating_probability : ℚ :=
  221 / 231

/-- Theorem stating that the probability of each delegate sitting next to at least one delegate 
    from another country is 221/231 -/
theorem delegate_seating_probability :
  seating_probability = 221 / 231 := by sorry

end delegate_seating_probability_l98_9818


namespace point_count_on_curve_l98_9850

theorem point_count_on_curve : 
  ∃! (points : Finset (ℤ × ℤ)), 
    points.card = 6 ∧ 
    ∀ p : ℤ × ℤ, p ∈ points ↔ 
      let m := p.1
      let n := p.2
      n^2 = (m^2 - 4) * (m^2 + 12*m + 32) + 4 := by
  sorry

end point_count_on_curve_l98_9850


namespace anne_solo_cleaning_time_l98_9844

/-- Represents the time it takes Anne to clean the house alone -/
def anne_solo_time : ℝ := 12

/-- Represents Bruce's cleaning rate (houses per hour) -/
noncomputable def bruce_rate : ℝ := sorry

/-- Represents Anne's cleaning rate (houses per hour) -/
noncomputable def anne_rate : ℝ := sorry

theorem anne_solo_cleaning_time :
  (∀ (bruce_rate anne_rate : ℝ),
    bruce_rate > 0 ∧ anne_rate > 0 →
    (bruce_rate + anne_rate) * 4 = 1 →
    (bruce_rate + 2 * anne_rate) * 3 = 1 →
    1 / anne_rate = anne_solo_time) :=
by sorry

end anne_solo_cleaning_time_l98_9844


namespace no_a_in_either_subject_l98_9885

theorem no_a_in_either_subject (total_students : ℕ) (a_in_chemistry : ℕ) (a_in_physics : ℕ) (a_in_both : ℕ) :
  total_students = 40 →
  a_in_chemistry = 10 →
  a_in_physics = 18 →
  a_in_both = 6 →
  total_students - (a_in_chemistry + a_in_physics - a_in_both) = 18 :=
by sorry

end no_a_in_either_subject_l98_9885


namespace green_bows_count_l98_9896

theorem green_bows_count (total : ℕ) (white : ℕ) : 
  (1 / 5 : ℚ) * total + (1 / 2 : ℚ) * total + (1 / 10 : ℚ) * total + white = total →
  white = 30 →
  (1 / 10 : ℚ) * total = 15 := by
sorry

end green_bows_count_l98_9896


namespace square_reciprocal_sum_equality_l98_9821

theorem square_reciprocal_sum_equality (n m k : ℕ+) : 
  (1 : ℚ) / n.val^2 + (1 : ℚ) / m.val^2 = (k : ℚ) / (n.val^2 + m.val^2) → k = 4 := by
  sorry

end square_reciprocal_sum_equality_l98_9821


namespace hannah_dog_food_l98_9843

/-- The amount of dog food Hannah needs to prepare daily for her five dogs -/
def total_dog_food (dog1_meal : ℝ) (dog1_freq : ℕ) (dog2_ratio : ℝ) (dog2_freq : ℕ)
  (dog3_extra : ℝ) (dog3_freq : ℕ) (dog4_ratio : ℝ) (dog4_freq : ℕ)
  (dog5_ratio : ℝ) (dog5_freq : ℕ) : ℝ :=
  (dog1_meal * dog1_freq) +
  (dog1_meal * dog2_ratio * dog2_freq) +
  ((dog1_meal * dog2_ratio + dog3_extra) * dog3_freq) +
  (dog4_ratio * (dog1_meal * dog2_ratio + dog3_extra) * dog4_freq) +
  (dog5_ratio * dog1_meal * dog5_freq)

/-- Theorem stating that Hannah needs to prepare 40.5 cups of dog food daily -/
theorem hannah_dog_food : total_dog_food 1.5 2 2 1 2.5 3 1.2 2 0.8 4 = 40.5 := by
  sorry

end hannah_dog_food_l98_9843


namespace factorial_fraction_l98_9865

theorem factorial_fraction (N : ℕ) :
  (Nat.factorial (N + 2)) / (Nat.factorial (N + 3) - Nat.factorial (N + 2)) = 1 / (N + 2) := by
  sorry

end factorial_fraction_l98_9865


namespace probability_of_snow_l98_9867

/-- The probability of snow on at least one day out of four, given specific conditions --/
theorem probability_of_snow (p : ℝ) (q : ℝ) : 
  p = 3/4 →  -- probability of snow on each of the first three days
  q = 4/5 →  -- probability of snow on the last day if it snowed before
  (1 - (1 - p)^3 * (1 - p) - (1 - (1 - p)^3) * (1 - q)) = 1023/1280 :=
by sorry

end probability_of_snow_l98_9867


namespace platform_length_l98_9856

/-- The length of a platform passed by an accelerating train -/
theorem platform_length (l a t : ℝ) (h1 : l > 0) (h2 : a > 0) (h3 : t > 0) : ∃ P : ℝ,
  (l = (1/2) * a * t^2) →
  (l + P = (1/2) * a * (6*t)^2) →
  P = 17 * l := by
  sorry

end platform_length_l98_9856


namespace hyperbola_real_axis_length_eq_4_div_sqrt5_l98_9852

noncomputable def hyperbola_real_axis_length 
  (a b : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (P : ℝ × ℝ) 
  (hP : P.1^2 / a^2 - P.2^2 / b^2 = 1) 
  (hP_right : P.1 > 0) 
  (A B : ℝ × ℝ) 
  (hA : A.1 > 0 ∧ A.2 > 0) 
  (hB : B.1 > 0 ∧ B.2 < 0) 
  (hAP_PB : (A.1 - P.1, A.2 - P.2) = (-1/3) • (B.1 - P.1, B.2 - P.2)) 
  (hAOB_area : (1/2) * abs (A.1 * B.2 - A.2 * B.1) = 2 * b) : 
  ℝ :=
2 * a

theorem hyperbola_real_axis_length_eq_4_div_sqrt5 
  (a b : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (P : ℝ × ℝ) 
  (hP : P.1^2 / a^2 - P.2^2 / b^2 = 1) 
  (hP_right : P.1 > 0) 
  (A B : ℝ × ℝ) 
  (hA : A.1 > 0 ∧ A.2 > 0) 
  (hB : B.1 > 0 ∧ B.2 < 0) 
  (hAP_PB : (A.1 - P.1, A.2 - P.2) = (-1/3) • (B.1 - P.1, B.2 - P.2)) 
  (hAOB_area : (1/2) * abs (A.1 * B.2 - A.2 * B.1) = 2 * b) : 
  hyperbola_real_axis_length a b ha hb P hP hP_right A B hA hB hAP_PB hAOB_area = 4 / Real.sqrt 5 :=
by sorry

end hyperbola_real_axis_length_eq_4_div_sqrt5_l98_9852


namespace inequality_proof_l98_9873

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h : x^2 + y^2 + z^2 = x + y + z) :
  x + y + z + 3 ≥ 6 * ((xy + yz + zx) / 3)^(1/3) := by
  sorry

end inequality_proof_l98_9873


namespace hyperbola_k_range_l98_9825

-- Define the curve equation
def curve (x y k : ℝ) : Prop := x^2 / (k + 4) + y^2 / (k - 1) = 1

-- Define what it means for the curve to be a hyperbola
def is_hyperbola (k : ℝ) : Prop := ∃ x y, curve x y k ∧ (k + 4) * (k - 1) < 0

-- Theorem statement
theorem hyperbola_k_range :
  ∀ k : ℝ, is_hyperbola k → k ∈ Set.Ioo (-4 : ℝ) 1 :=
by sorry

end hyperbola_k_range_l98_9825


namespace removed_carrots_average_weight_l98_9863

/-- Proves that the average weight of 4 removed carrots is 190 grams -/
theorem removed_carrots_average_weight
  (total_weight : ℝ)
  (remaining_carrots : ℕ)
  (removed_carrots : ℕ)
  (remaining_average : ℝ)
  (h1 : total_weight = 3.64)
  (h2 : remaining_carrots = 16)
  (h3 : removed_carrots = 4)
  (h4 : remaining_average = 180)
  (h5 : remaining_carrots + removed_carrots = 20) :
  (total_weight * 1000 - remaining_carrots * remaining_average) / removed_carrots = 190 :=
by sorry

end removed_carrots_average_weight_l98_9863


namespace f_properties_l98_9839

noncomputable def f (x : ℝ) : ℝ := Real.log (abs x)

theorem f_properties :
  (∀ x y : ℝ, x > 0 → y > 0 → f (x * y) = f x + f y) ∧
  (∀ x : ℝ, x ≠ 0 → f x = f (-x)) ∧
  (∀ x : ℝ, x > 0 → deriv f x > 0) := by sorry

end f_properties_l98_9839


namespace max_hiking_time_l98_9827

/-- Calculates the maximum hiking time for Violet and her dog given their water consumption rates and the total water carried. -/
theorem max_hiking_time (violet_rate : ℝ) (dog_rate : ℝ) (total_water : ℝ) :
  violet_rate = 800 →
  dog_rate = 400 →
  total_water = 4800 →
  (total_water / (violet_rate + dog_rate) : ℝ) = 4 := by
  sorry

end max_hiking_time_l98_9827


namespace smallest_a_satisfying_equation_l98_9813

theorem smallest_a_satisfying_equation :
  ∃ a : ℝ, (a = -Real.sqrt (62/5)) ∧
    (∀ b : ℝ, (8*Real.sqrt ((3*b)^2 + 2^2) - 5*b^2 - 2) / (Real.sqrt (2 + 5*b^2) + 4) = 3 → a ≤ b) ∧
    (8*Real.sqrt ((3*a)^2 + 2^2) - 5*a^2 - 2) / (Real.sqrt (2 + 5*a^2) + 4) = 3 :=
by sorry

end smallest_a_satisfying_equation_l98_9813


namespace probability_proof_l98_9879

def total_balls : ℕ := 6
def white_balls : ℕ := 3
def black_balls : ℕ := 3
def drawn_balls : ℕ := 2

def probability_at_most_one_black : ℚ := 4/5

theorem probability_proof :
  (Nat.choose total_balls drawn_balls - Nat.choose black_balls drawn_balls) / Nat.choose total_balls drawn_balls = probability_at_most_one_black :=
sorry

end probability_proof_l98_9879


namespace multiple_of_larger_integer_l98_9809

theorem multiple_of_larger_integer (s l : ℤ) (m : ℚ) : 
  s + l = 30 →
  s = 10 →
  m * l = 5 * s - 10 →
  m = 2 := by
sorry

end multiple_of_larger_integer_l98_9809
