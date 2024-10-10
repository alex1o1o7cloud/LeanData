import Mathlib

namespace marble_probability_l2676_267616

theorem marble_probability (total : ℕ) (blue red : ℕ) (h1 : total = 50) (h2 : blue = 5) (h3 : red = 9) :
  (red + (total - blue - red)) / total = 9 / 10 := by
  sorry

end marble_probability_l2676_267616


namespace cosine_amplitude_l2676_267602

theorem cosine_amplitude (a b c d : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) :
  (∀ x, ∃ y, y = a * Real.cos (b * x + c) + d) →
  (∃ x1, a * Real.cos (b * x1 + c) + d = 5) →
  (∃ x2, a * Real.cos (b * x2 + c) + d = -3) →
  a = 4 := by
  sorry

end cosine_amplitude_l2676_267602


namespace daily_wage_of_c_l2676_267696

theorem daily_wage_of_c (a b c : ℕ) (total_earning : ℕ) : 
  a * 6 + b * 9 + c * 4 = total_earning →
  4 * a = 3 * b →
  5 * a = 3 * c →
  total_earning = 1554 →
  c = 105 := by
  sorry

end daily_wage_of_c_l2676_267696


namespace intersection_of_M_and_N_l2676_267617

def M : Set ℤ := {m : ℤ | -3 < m ∧ m < 2}
def N : Set ℤ := {n : ℤ | -1 ≤ n ∧ n ≤ 3}

theorem intersection_of_M_and_N : M ∩ N = {-1, 0, 1} := by sorry

end intersection_of_M_and_N_l2676_267617


namespace quadratic_roots_opposite_signs_l2676_267647

theorem quadratic_roots_opposite_signs (a : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ x * y < 0 ∧ 
   a * x^2 - (a + 3) * x + 2 = 0 ∧
   a * y^2 - (a + 3) * y + 2 = 0) ↔ 
  a < 0 := by
sorry

end quadratic_roots_opposite_signs_l2676_267647


namespace gunther_dusting_time_l2676_267693

/-- Represents the time in minutes for Gunther's cleaning tasks -/
structure CleaningTime where
  vacuuming : ℕ
  mopping : ℕ
  brushing_per_cat : ℕ
  num_cats : ℕ
  total_free_time : ℕ
  remaining_free_time : ℕ

/-- Calculates the time spent dusting furniture -/
def dusting_time (ct : CleaningTime) : ℕ :=
  ct.total_free_time - ct.remaining_free_time - 
  (ct.vacuuming + ct.mopping + ct.brushing_per_cat * ct.num_cats)

/-- Theorem stating that Gunther spends 60 minutes dusting furniture -/
theorem gunther_dusting_time :
  let ct : CleaningTime := {
    vacuuming := 45,
    mopping := 30,
    brushing_per_cat := 5,
    num_cats := 3,
    total_free_time := 3 * 60,
    remaining_free_time := 30
  }
  dusting_time ct = 60 := by
  sorry

end gunther_dusting_time_l2676_267693


namespace total_materials_ordered_l2676_267632

-- Define the amounts of materials ordered
def concrete : Real := 0.17
def bricks : Real := 0.237
def sand : Real := 0.646
def stone : Real := 0.5
def steel : Real := 1.73
def wood : Real := 0.894

-- Theorem statement
theorem total_materials_ordered :
  concrete + bricks + sand + stone + steel + wood = 4.177 := by
  sorry

end total_materials_ordered_l2676_267632


namespace inequality_system_solution_l2676_267697

theorem inequality_system_solution (a b : ℝ) :
  (∀ x : ℝ, (0 < x ∧ x < 4) ↔ (x - a < 1 ∧ x + b > 2)) →
  b - a = -1 := by
  sorry

end inequality_system_solution_l2676_267697


namespace triple_solution_l2676_267634

theorem triple_solution (k : ℕ) (hk : k > 0) :
  ∀ a b c : ℕ, 
    a > 0 → b > 0 → c > 0 →
    a + b + c = 3 * k + 1 →
    a * b + b * c + c * a = 3 * k^2 + 2 * k →
    (a = k + 1 ∧ b = k ∧ c = k) :=
by sorry

end triple_solution_l2676_267634


namespace symmetric_line_l2676_267623

/-- Given a line with equation 2x - y + 3 = 0 and a fixed point M(-1, 2),
    the equation of the line symmetric to the given line with respect to M is 2x - y + 5 = 0 -/
theorem symmetric_line (x y : ℝ) : 
  (∀ x y, 2*x - y + 3 = 0 → 2*x - y + 5 = 0) := by
  sorry

end symmetric_line_l2676_267623


namespace min_value_x_plus_2y_min_value_equality_l2676_267699

theorem min_value_x_plus_2y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 1/x + 9/y = 1) :
  ∀ z w : ℝ, z > 0 → w > 0 → 1/z + 9/w = 1 → x + 2*y ≤ z + 2*w :=
by sorry

theorem min_value_equality (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 1/x + 9/y = 1) :
  x + 2*y = 19 + 6*Real.sqrt 2 :=
by sorry

end min_value_x_plus_2y_min_value_equality_l2676_267699


namespace opposite_of_one_seventh_l2676_267630

theorem opposite_of_one_seventh :
  ∀ x : ℚ, x + (1 / 7) = 0 ↔ x = -(1 / 7) := by
  sorry

end opposite_of_one_seventh_l2676_267630


namespace fraction_equality_l2676_267626

theorem fraction_equality : ∃ x : ℚ, x * (7/8 * 1/3) = 0.12499999999999997 := by
  sorry

end fraction_equality_l2676_267626


namespace encyclopedia_pages_l2676_267601

/-- The Encyclopedia of Life and Everything Else --/
structure Encyclopedia where
  chapters : Nat
  pages_per_chapter : Nat

/-- Calculate the total number of pages in the encyclopedia --/
def total_pages (e : Encyclopedia) : Nat :=
  e.chapters * e.pages_per_chapter

/-- Theorem: The encyclopedia has 9384 pages in total --/
theorem encyclopedia_pages :
  ∃ (e : Encyclopedia), e.chapters = 12 ∧ e.pages_per_chapter = 782 ∧ total_pages e = 9384 := by
  sorry

end encyclopedia_pages_l2676_267601


namespace find_N_l2676_267639

theorem find_N : ∃ N : ℝ, (0.2 * N = 0.6 * 2500) ∧ (N = 7500) := by
  sorry

end find_N_l2676_267639


namespace instantaneous_velocity_at_5_l2676_267635

/-- The motion equation of a ball rolling down an inclined plane -/
def motion_equation (t : ℝ) : ℝ := t^2

/-- The velocity function derived from the motion equation -/
def velocity (t : ℝ) : ℝ := 2 * t

theorem instantaneous_velocity_at_5 : velocity 5 = 10 := by
  sorry

end instantaneous_velocity_at_5_l2676_267635


namespace a_17_value_l2676_267615

-- Define a geometric sequence
def geometric_sequence (a : ℕ → ℝ) := ∀ n, a (n + 1) / a n = a 2 / a 1

-- State the theorem
theorem a_17_value (a : ℕ → ℝ) (h_geo : geometric_sequence a) 
  (h_a1 : a 1 = 1) (h_a2a8 : a 2 * a 8 = 16) : a 17 = 256 := by
  sorry

end a_17_value_l2676_267615


namespace scorpion_millipedes_l2676_267690

/-- Calculates the number of millipedes needed to reach a daily segment goal -/
def millipedes_needed (daily_requirement : ℕ) (eaten_segments : ℕ) (remaining_millipede_segments : ℕ) : ℕ :=
  (daily_requirement - eaten_segments) / remaining_millipede_segments

theorem scorpion_millipedes :
  let daily_requirement : ℕ := 800
  let first_millipede_segments : ℕ := 60
  let long_millipede_segments : ℕ := 2 * first_millipede_segments
  let eaten_segments : ℕ := first_millipede_segments + 2 * long_millipede_segments
  let remaining_millipede_segments : ℕ := 50
  millipedes_needed daily_requirement eaten_segments remaining_millipede_segments = 10 := by
  sorry

end scorpion_millipedes_l2676_267690


namespace ap_square_cube_implies_sixth_power_l2676_267605

/-- An arithmetic progression is represented by its first term and common difference -/
structure ArithmeticProgression where
  first_term : ℕ
  common_diff : ℕ

/-- Check if a number is in the arithmetic progression -/
def ArithmeticProgression.contains (ap : ArithmeticProgression) (n : ℕ) : Prop :=
  ∃ k : ℕ, n = ap.first_term + k * ap.common_diff

/-- An arithmetic progression contains a perfect square -/
def contains_square (ap : ArithmeticProgression) : Prop :=
  ∃ x : ℕ, ap.contains (x^2)

/-- An arithmetic progression contains a perfect cube -/
def contains_cube (ap : ArithmeticProgression) : Prop :=
  ∃ y : ℕ, ap.contains (y^3)

/-- An arithmetic progression contains a sixth power -/
def contains_sixth_power (ap : ArithmeticProgression) : Prop :=
  ∃ z : ℕ, ap.contains (z^6)

/-- Main theorem: If an AP contains a square and a cube, it contains a sixth power -/
theorem ap_square_cube_implies_sixth_power (ap : ArithmeticProgression) 
  (h1 : ap.first_term > 0) 
  (h2 : contains_square ap) 
  (h3 : contains_cube ap) : 
  contains_sixth_power ap := by
  sorry

end ap_square_cube_implies_sixth_power_l2676_267605


namespace space_division_by_five_spheres_l2676_267643

/-- Maximum number of regions into which a sphere can be divided by n circles -/
def a : ℕ → ℕ
  | 0 => 1
  | 1 => 2
  | 2 => 4
  | n + 3 => a (n + 2) + 2 * (n + 2)

/-- Maximum number of regions into which space can be divided by n spheres -/
def b : ℕ → ℕ
  | 0 => 1
  | 1 => 2
  | 2 => 4
  | n + 3 => b (n + 2) + a (n + 2)

theorem space_division_by_five_spheres :
  b 5 = 30 := by sorry

end space_division_by_five_spheres_l2676_267643


namespace cost_of_items_l2676_267648

/-- Given the costs of combinations of pencils and pens, prove the cost of one of each item -/
theorem cost_of_items (pencil pen : ℝ) 
  (h1 : 3 * pencil + 2 * pen = 4.10)
  (h2 : 2 * pencil + 3 * pen = 3.70)
  (eraser : ℝ := 0.85) : 
  pencil + pen + eraser = 2.41 := by
  sorry

end cost_of_items_l2676_267648


namespace fraction_inequality_l2676_267694

theorem fraction_inequality (a b : ℝ) (ha : a > 0) (hb : b < 0) : 
  a / b + b / a ≤ -2 := by
  sorry

end fraction_inequality_l2676_267694


namespace delegates_without_badges_l2676_267695

theorem delegates_without_badges (total : ℕ) (preprinted : ℕ) : 
  total = 36 → preprinted = 16 → (total - preprinted - (total - preprinted) / 2) = 10 := by
sorry

end delegates_without_badges_l2676_267695


namespace sum_of_fractions_l2676_267656

theorem sum_of_fractions : (3 / 100 : ℚ) + (5 / 1000 : ℚ) + (7 / 10000 : ℚ) = (357 / 10000 : ℚ) := by
  sorry

end sum_of_fractions_l2676_267656


namespace race_coin_problem_l2676_267640

theorem race_coin_problem (x y : ℕ) (h1 : x > y) (h2 : y > 0) : 
  (∃ n : ℕ, n > 2 ∧ 
   (n - 2) * x + 2 * y = 42 ∧ 
   2 * x + (n - 2) * y = 35) → 
  x = 4 := by
sorry

end race_coin_problem_l2676_267640


namespace pool_water_calculation_l2676_267669

/-- Calculates the amount of water in a pool after five hours of filling and a leak -/
def water_in_pool (rate1 : ℕ) (rate2 : ℕ) (rate3 : ℕ) (leak : ℕ) : ℕ :=
  rate1 + 2 * rate2 + rate3 - leak

theorem pool_water_calculation :
  water_in_pool 8 10 14 8 = 34 := by
  sorry

end pool_water_calculation_l2676_267669


namespace number_added_after_doubling_l2676_267687

theorem number_added_after_doubling (x y : ℝ) : x = 4 → 3 * (2 * x + y) = 51 → y = 9 := by
  sorry

end number_added_after_doubling_l2676_267687


namespace orange_weight_problem_l2676_267685

theorem orange_weight_problem (initial_water_concentration : Real)
                               (water_decrease : Real)
                               (new_weight : Real) :
  initial_water_concentration = 0.95 →
  water_decrease = 0.05 →
  new_weight = 25 →
  ∃ (initial_weight : Real),
    initial_weight = 50 ∧
    (1 - initial_water_concentration) * initial_weight =
    (1 - (initial_water_concentration - water_decrease)) * new_weight :=
by sorry

end orange_weight_problem_l2676_267685


namespace mr_johnson_class_size_l2676_267624

def mrs_finley_class : ℕ := 24

def mr_johnson_class : ℕ := (mrs_finley_class / 2) + 10

theorem mr_johnson_class_size : mr_johnson_class = 22 := by
  sorry

end mr_johnson_class_size_l2676_267624


namespace apples_per_box_l2676_267686

theorem apples_per_box (total_apples : ℕ) (rotten_apples : ℕ) (num_boxes : ℕ) 
  (h1 : total_apples = 40)
  (h2 : rotten_apples = 4)
  (h3 : num_boxes = 4)
  (h4 : rotten_apples < total_apples) :
  (total_apples - rotten_apples) / num_boxes = 9 := by
sorry

end apples_per_box_l2676_267686


namespace tablet_value_proof_compensation_for_m_days_l2676_267679

-- Define the total days of internship
def total_days : ℕ := 30

-- Define the cash compensation for full internship
def full_cash_compensation : ℕ := 1500

-- Define the number of days Xiaomin worked
def worked_days : ℕ := 20

-- Define the cash compensation Xiaomin received
def received_cash_compensation : ℕ := 300

-- Define the value of the M type tablet
def tablet_value : ℕ := 2100

-- Define the daily compensation rate
def daily_rate : ℚ := 120

-- Theorem for the value of the M type tablet
theorem tablet_value_proof :
  (worked_days : ℚ) / total_days * (tablet_value + full_cash_compensation) =
  tablet_value + received_cash_compensation :=
sorry

-- Theorem for the compensation for m days of work
theorem compensation_for_m_days (m : ℕ) :
  (m : ℚ) * daily_rate = (m : ℚ) * ((tablet_value + full_cash_compensation) / total_days) :=
sorry

end tablet_value_proof_compensation_for_m_days_l2676_267679


namespace x_intercept_ratio_l2676_267646

-- Define the slopes and y-intercept
def m₁ : ℝ := 8
def m₂ : ℝ := 4
def c : ℝ := 0  -- y-intercept, defined as non-zero in the theorem

-- Define the x-intercepts
def u : ℝ := 0  -- actual value doesn't matter, will be constrained in the theorem
def v : ℝ := 0  -- actual value doesn't matter, will be constrained in the theorem

-- Theorem statement
theorem x_intercept_ratio (h₁ : c ≠ 0) 
                          (h₂ : m₁ * u + c = 0) 
                          (h₃ : m₂ * v + c = 0) : 
  u / v = 1 / 2 := by
  sorry

end x_intercept_ratio_l2676_267646


namespace spice_jar_cost_is_six_l2676_267671

/-- Represents the cost and point structure for Martha's grocery shopping -/
structure GroceryShopping where
  pointsPerTenDollars : ℕ
  bonusThreshold : ℕ
  bonusPoints : ℕ
  beefPounds : ℕ
  beefPricePerPound : ℕ
  fruitVegPounds : ℕ
  fruitVegPricePerPound : ℕ
  spiceJars : ℕ
  otherGroceriesCost : ℕ
  totalPoints : ℕ

/-- Calculates the cost of each jar of spices based on the given shopping information -/
def calculateSpiceJarCost (shopping : GroceryShopping) : ℕ :=
  sorry

/-- Theorem stating that the cost of each jar of spices is $6 -/
theorem spice_jar_cost_is_six (shopping : GroceryShopping) 
  (h1 : shopping.pointsPerTenDollars = 50)
  (h2 : shopping.bonusThreshold = 100)
  (h3 : shopping.bonusPoints = 250)
  (h4 : shopping.beefPounds = 3)
  (h5 : shopping.beefPricePerPound = 11)
  (h6 : shopping.fruitVegPounds = 8)
  (h7 : shopping.fruitVegPricePerPound = 4)
  (h8 : shopping.spiceJars = 3)
  (h9 : shopping.otherGroceriesCost = 37)
  (h10 : shopping.totalPoints = 850) :
  calculateSpiceJarCost shopping = 6 :=
  sorry


end spice_jar_cost_is_six_l2676_267671


namespace smallest_n_congruence_l2676_267636

theorem smallest_n_congruence : ∃! n : ℕ+, (3 * n : ℤ) ≡ 568 [ZMOD 34] ∧ 
  ∀ m : ℕ+, (3 * m : ℤ) ≡ 568 [ZMOD 34] → n ≤ m :=
by sorry

end smallest_n_congruence_l2676_267636


namespace vicente_spent_25_dollars_l2676_267653

-- Define the quantities and prices
def rice_kg : ℕ := 5
def meat_lb : ℕ := 3
def rice_price_per_kg : ℕ := 2
def meat_price_per_lb : ℕ := 5

-- Define the total cost function
def total_cost (rice_kg meat_lb rice_price_per_kg meat_price_per_lb : ℕ) : ℕ :=
  rice_kg * rice_price_per_kg + meat_lb * meat_price_per_lb

-- Theorem statement
theorem vicente_spent_25_dollars :
  total_cost rice_kg meat_lb rice_price_per_kg meat_price_per_lb = 25 := by
  sorry

end vicente_spent_25_dollars_l2676_267653


namespace angle_covered_in_three_layers_l2676_267663

theorem angle_covered_in_three_layers 
  (total_angle : ℝ) 
  (sum_of_angles : ℝ) 
  (angle_three_layers : ℝ) 
  (h1 : total_angle = 90) 
  (h2 : sum_of_angles = 290) 
  (h3 : angle_three_layers * 3 + (total_angle - angle_three_layers) * 2 = sum_of_angles) :
  angle_three_layers = 20 := by
sorry

end angle_covered_in_three_layers_l2676_267663


namespace backyard_area_l2676_267611

/-- Represents a rectangular backyard with specific walking properties. -/
structure Backyard where
  length : ℝ
  width : ℝ
  total_distance : ℝ
  length_walks : ℕ
  perimeter_walks : ℕ
  length_covers_total : length * length_walks = total_distance
  perimeter_covers_total : (2 * length + 2 * width) * perimeter_walks = total_distance

/-- The theorem stating the area of the backyard with given properties. -/
theorem backyard_area (b : Backyard) (h1 : b.total_distance = 2000)
    (h2 : b.length_walks = 50) (h3 : b.perimeter_walks = 20) :
    b.length * b.width = 400 := by
  sorry


end backyard_area_l2676_267611


namespace least_perimeter_of_special_triangle_l2676_267642

/-- A triangle with integer side lengths -/
structure IntTriangle where
  a : ℕ
  b : ℕ
  c : ℕ
  triangle_ineq : a + b > c ∧ b + c > a ∧ c + a > b

/-- The condition for a triangle to be non-equilateral -/
def is_non_equilateral (t : IntTriangle) : Prop :=
  t.a ≠ t.b ∨ t.b ≠ t.c ∨ t.c ≠ t.a

/-- The condition for points D, C, E, G to be concyclic -/
def is_concyclic (t : IntTriangle) : Prop :=
  -- This is a placeholder for the actual concyclic condition
  -- In reality, this would involve more complex geometric relations
  true

/-- The perimeter of a triangle -/
def perimeter (t : IntTriangle) : ℕ := t.a + t.b + t.c

theorem least_perimeter_of_special_triangle :
  ∃ (t : IntTriangle),
    is_non_equilateral t ∧
    is_concyclic t ∧
    (∀ (s : IntTriangle), is_non_equilateral s → is_concyclic s → perimeter t ≤ perimeter s) ∧
    perimeter t = 37 := by
  sorry

end least_perimeter_of_special_triangle_l2676_267642


namespace milk_problem_l2676_267625

theorem milk_problem (initial_milk : ℚ) (rachel_fraction : ℚ) (joey_fraction : ℚ) :
  initial_milk = 3/4 →
  rachel_fraction = 1/3 →
  joey_fraction = 1/2 →
  joey_fraction * (initial_milk - rachel_fraction * initial_milk) = 1/4 := by
  sorry

end milk_problem_l2676_267625


namespace line_conditions_vector_at_zero_l2676_267689

-- Define the line parameterization
def line_param (t : ℝ) : ℝ × ℝ := sorry

-- Define the conditions
theorem line_conditions :
  line_param 1 = (2, 5) ∧ line_param 4 = (11, -7) := sorry

-- Theorem to prove
theorem vector_at_zero :
  line_param 0 = (-1, 9) := by sorry

end line_conditions_vector_at_zero_l2676_267689


namespace pie_remainder_l2676_267645

theorem pie_remainder (whole_pie : ℝ) (carlos_share : ℝ) (maria_share : ℝ) : 
  carlos_share = 0.8 * whole_pie →
  maria_share = 0.25 * (whole_pie - carlos_share) →
  whole_pie - carlos_share - maria_share = 0.15 * whole_pie :=
by sorry

end pie_remainder_l2676_267645


namespace age_ratio_proof_l2676_267673

theorem age_ratio_proof (b_age : ℕ) (a_age : ℕ) : 
  b_age = 39 →
  a_age = b_age + 9 →
  (a_age + 10) / (b_age - 10) = 2 :=
by
  sorry

end age_ratio_proof_l2676_267673


namespace f_derivative_at_zero_over_f_at_zero_l2676_267657

noncomputable def f (x : ℝ) : ℝ := (x + Real.sqrt (1 + x^2))^10

theorem f_derivative_at_zero_over_f_at_zero : 
  (deriv f 0) / (f 0) = 10 := by sorry

end f_derivative_at_zero_over_f_at_zero_l2676_267657


namespace min_distance_complex_l2676_267661

theorem min_distance_complex (Z : ℂ) (h : Complex.abs (Z + 2 - 2*I) = 1) :
  ∃ (min_val : ℝ), min_val = 3 ∧ ∀ (W : ℂ), Complex.abs (W + 2 - 2*I) = 1 → Complex.abs (W - 2 - 2*I) ≥ min_val :=
sorry

end min_distance_complex_l2676_267661


namespace log_gt_x_squared_over_one_plus_x_l2676_267668

theorem log_gt_x_squared_over_one_plus_x :
  ∃ a : ℝ, a > 0 ∧ ∀ x : ℝ, 0 < x → x < a → Real.log (1 + x) > x^2 / (1 + x) := by
  sorry

end log_gt_x_squared_over_one_plus_x_l2676_267668


namespace Q_roots_l2676_267620

def Q (x : ℝ) : ℝ := x^6 - 5*x^5 - 12*x^3 - x + 16

theorem Q_roots :
  (∀ x < 0, Q x > 0) ∧ 
  (∃ x > 0, Q x = 0) := by
sorry

end Q_roots_l2676_267620


namespace cosine_amplitude_l2676_267613

/-- Given a cosine function y = a * cos(b * x + c) + d with positive constants a, b, c, and d,
    if the maximum value of y is 5 and the minimum value is -3, then a = 4. -/
theorem cosine_amplitude (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (∀ x, a * Real.cos (b * x + c) + d ≤ 5) ∧
  (∀ x, a * Real.cos (b * x + c) + d ≥ -3) ∧
  (∃ x, a * Real.cos (b * x + c) + d = 5) ∧
  (∃ x, a * Real.cos (b * x + c) + d = -3) →
  a = 4 := by
  sorry

end cosine_amplitude_l2676_267613


namespace candidate_votes_l2676_267659

theorem candidate_votes (total_votes : ℕ) (invalid_percent : ℚ) (candidate_percent : ℚ) :
  total_votes = 560000 →
  invalid_percent = 15/100 →
  candidate_percent = 65/100 →
  (1 - invalid_percent) * candidate_percent * total_votes = 309400 := by
  sorry

end candidate_votes_l2676_267659


namespace wednesday_occurs_five_times_l2676_267600

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a specific date in a month -/
structure Date :=
  (day : Nat)
  (dayOfWeek : DayOfWeek)

/-- Properties of December in year M -/
structure DecemberProperties :=
  (sundays : List Date)
  (hasFiveSundays : sundays.length = 5)
  (has31Days : Nat)

/-- Properties of January in year M+1 -/
structure JanuaryProperties :=
  (firstDay : DayOfWeek)
  (has31Days : Nat)

/-- Function to determine the number of occurrences of a day in January -/
def countOccurrencesInJanuary (day : DayOfWeek) (january : JanuaryProperties) : Nat :=
  sorry

/-- Main theorem -/
theorem wednesday_occurs_five_times
  (december : DecemberProperties)
  (january : JanuaryProperties)
  : countOccurrencesInJanuary DayOfWeek.Wednesday january = 5 :=
sorry

end wednesday_occurs_five_times_l2676_267600


namespace ln_b_over_a_range_l2676_267629

theorem ln_b_over_a_range (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h1 : (1 : ℝ) / Real.exp 1 ≤ c / a) (h2 : c / a ≤ 2)
  (h3 : c * Real.log b = a + c * Real.log c) :
  ∃ (x : ℝ), x ∈ Set.Icc 1 (Real.exp 1 - 1) ∧ Real.log (b / a) = x :=
sorry

end ln_b_over_a_range_l2676_267629


namespace intersection_A_B_l2676_267676

-- Define set A
def A : Set ℝ := {x | -1 ≤ x ∧ x ≤ 1}

-- Define set B
def B : Set ℝ := {x | x^2 - 2*x ≤ 0}

-- Theorem statement
theorem intersection_A_B : A ∩ B = {x | 0 ≤ x ∧ x ≤ 1} := by sorry

end intersection_A_B_l2676_267676


namespace triangle_properties_l2676_267622

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def satisfies_conditions (t : Triangle) : Prop :=
  (t.b - t.c)^2 = t.a^2 - t.b * t.c ∧
  t.a = 3 ∧
  Real.sin t.C = 2 * Real.sin t.B

-- Theorem statement
theorem triangle_properties (t : Triangle) (h : satisfies_conditions t) :
  t.A = π/3 ∧ t.a * t.b * Real.sin t.C / 2 = 3 * Real.sqrt 3 / 2 := by
  sorry

end triangle_properties_l2676_267622


namespace age_condition_amount_per_year_is_five_l2676_267651

/-- Mikail's age on his birthday -/
def age : ℕ := 9

/-- The total amount Mikail receives on his birthday -/
def total_amount : ℕ := 45

/-- The condition that Mikail's age is 3 times as old as he was when he was three -/
theorem age_condition : age = 3 * 3 := by sorry

/-- The amount Mikail receives per year of his age -/
def amount_per_year : ℚ := total_amount / age

/-- Proof that the amount Mikail receives per year is $5 -/
theorem amount_per_year_is_five : amount_per_year = 5 := by sorry

end age_condition_amount_per_year_is_five_l2676_267651


namespace sales_difference_l2676_267684

/-- Represents a company selling bottled milk -/
structure Company where
  big_bottle_price : ℝ
  small_bottle_price : ℝ
  big_bottle_discount : ℝ
  small_bottle_discount : ℝ
  big_bottles_sold : ℕ
  small_bottles_sold : ℕ

def tax_rate : ℝ := 0.07

def company_A : Company := {
  big_bottle_price := 4
  small_bottle_price := 2
  big_bottle_discount := 0.1
  small_bottle_discount := 0
  big_bottles_sold := 300
  small_bottles_sold := 400
}

def company_B : Company := {
  big_bottle_price := 3.5
  small_bottle_price := 1.75
  big_bottle_discount := 0
  small_bottle_discount := 0.05
  big_bottles_sold := 350
  small_bottles_sold := 600
}

def calculate_total_sales (c : Company) : ℝ :=
  let big_bottle_revenue := c.big_bottle_price * c.big_bottles_sold
  let small_bottle_revenue := c.small_bottle_price * c.small_bottles_sold
  let total_before_discount := big_bottle_revenue + small_bottle_revenue
  let big_bottle_discount := if c.big_bottles_sold ≥ 10 then c.big_bottle_discount * big_bottle_revenue else 0
  let small_bottle_discount := if c.small_bottles_sold > 20 then c.small_bottle_discount * small_bottle_revenue else 0
  let total_after_discount := total_before_discount - big_bottle_discount - small_bottle_discount
  let total_after_tax := total_after_discount * (1 + tax_rate)
  total_after_tax

theorem sales_difference : 
  calculate_total_sales company_B - calculate_total_sales company_A = 366.475 := by
  sorry

end sales_difference_l2676_267684


namespace octagon_diagonals_l2676_267683

/-- The number of diagonals in a polygon with n vertices -/
def num_diagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

/-- The number of vertices in an octagon -/
def octagon_vertices : ℕ := 8

/-- Theorem: The number of diagonals in an octagon is 20 -/
theorem octagon_diagonals : num_diagonals octagon_vertices = 20 := by
  sorry

end octagon_diagonals_l2676_267683


namespace equality_of_fractions_l2676_267691

theorem equality_of_fractions (x y z k : ℝ) : 
  (5 / (x + y) = k / (x + z)) ∧ (k / (x + z) = 9 / (z - y)) → k = 14 := by
  sorry

end equality_of_fractions_l2676_267691


namespace remainder_problem_l2676_267682

theorem remainder_problem : 123456789012 % 112 = 76 := by
  sorry

end remainder_problem_l2676_267682


namespace quadratic_inequality_solution_l2676_267637

/-- Given a quadratic inequality x^2 + bx - a < 0 with solution set {x | -2 < x < 3}, prove that a + b = 5 -/
theorem quadratic_inequality_solution (a b : ℝ) 
  (h : ∀ x, x^2 + b*x - a < 0 ↔ -2 < x ∧ x < 3) : 
  a + b = 5 := by sorry

end quadratic_inequality_solution_l2676_267637


namespace gravel_pile_volume_l2676_267618

/-- The volume of a hemispherical pile of gravel -/
theorem gravel_pile_volume (d : ℝ) (h : ℝ) (v : ℝ) : 
  d = 10 → -- diameter is 10 feet
  h = d / 2 → -- height is half the diameter
  v = (250 * Real.pi) / 3 → -- volume is (250π)/3 cubic feet
  v = (2 / 3) * Real.pi * (d / 2)^3 := by
  sorry

end gravel_pile_volume_l2676_267618


namespace negation_of_forall_positive_l2676_267681

theorem negation_of_forall_positive (P : ℝ → Prop) :
  (¬ ∀ x : ℝ, x^2 + 1 > 0) ↔ (∃ x : ℝ, x^2 + 1 ≤ 0) :=
by sorry

end negation_of_forall_positive_l2676_267681


namespace last_digit_is_square_of_second_l2676_267677

/-- Represents a 4-digit number -/
structure FourDigitNumber where
  d1 : Nat
  d2 : Nat
  d3 : Nat
  d4 : Nat
  is_four_digit : d1 ≠ 0 ∧ d1 < 10 ∧ d2 < 10 ∧ d3 < 10 ∧ d4 < 10

/-- The specific 4-digit number 1349 -/
def number : FourDigitNumber where
  d1 := 1
  d2 := 3
  d3 := 4
  d4 := 9
  is_four_digit := by sorry

theorem last_digit_is_square_of_second :
  (number.d1 = number.d2 / 3) →
  (number.d3 = number.d1 + number.d2) →
  (number.d4 = number.d2 * number.d2) := by sorry

end last_digit_is_square_of_second_l2676_267677


namespace min_values_a_b_l2676_267614

theorem min_values_a_b (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a * b = 2 * a + b + 2) :
  (∀ x y : ℝ, 0 < x ∧ 0 < y ∧ x * y = 2 * x + y + 2 → a * b ≤ x * y) ∧
  (∀ x y : ℝ, 0 < x ∧ 0 < y ∧ x * y = 2 * x + y + 2 → a + 2 * b ≤ x + 2 * y) ∧
  a * b = 6 + 4 * Real.sqrt 2 ∧
  a + 2 * b = 4 * Real.sqrt 2 + 5 :=
by sorry

end min_values_a_b_l2676_267614


namespace taxi_fare_theorem_l2676_267670

/-- Taxi fare function for distances greater than 5 kilometers -/
def taxi_fare (x : ℝ) : ℝ :=
  10 + 2 * 1.3 + 2.4 * (x - 5)

/-- Theorem stating the taxi fare function and its value for 6 kilometers -/
theorem taxi_fare_theorem (x : ℝ) (h : x > 5) :
  taxi_fare x = 2.4 * x + 0.6 ∧ taxi_fare 6 = 15 := by
  sorry

#check taxi_fare_theorem

end taxi_fare_theorem_l2676_267670


namespace expression_evaluation_l2676_267680

theorem expression_evaluation : -20 + 12 * (8 / 4) * 3 = 52 := by
  sorry

end expression_evaluation_l2676_267680


namespace squirrel_count_l2676_267621

theorem squirrel_count (total_acorns : ℕ) (needed_acorns : ℕ) (shortage : ℕ) : 
  total_acorns = 575 →
  needed_acorns = 130 →
  shortage = 15 →
  (total_acorns / (needed_acorns - shortage) : ℕ) = 5 := by
sorry

end squirrel_count_l2676_267621


namespace greatest_two_digit_product_12_l2676_267688

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def digit_product (n : ℕ) : ℕ :=
  (n / 10) * (n % 10)

theorem greatest_two_digit_product_12 :
  ∀ n : ℕ, is_two_digit n → digit_product n = 12 → n ≤ 62 :=
sorry

end greatest_two_digit_product_12_l2676_267688


namespace final_sum_after_operations_l2676_267633

theorem final_sum_after_operations (S x k : ℝ) (a b : ℝ) (h : a + b = S) :
  k * (a + x) + k * (b + x) = k * S + 2 * k * x := by
  sorry

end final_sum_after_operations_l2676_267633


namespace fish_cost_is_80_l2676_267667

/-- The cost of fish in pesos per kilogram -/
def fish_cost : ℕ := 80

/-- The cost of pork in pesos per kilogram -/
def pork_cost : ℕ := 105

/-- Theorem stating that the cost of fish is 80 pesos per kilogram -/
theorem fish_cost_is_80 :
  (530 = 4 * fish_cost + 2 * pork_cost) →
  (875 = 7 * fish_cost + 3 * pork_cost) →
  fish_cost = 80 := by
  sorry

end fish_cost_is_80_l2676_267667


namespace ab_range_l2676_267655

theorem ab_range (a b : ℝ) : 
  a > 0 → b > 0 → a * b = a + b + 3 → a * b ≥ 9 := by
  sorry

end ab_range_l2676_267655


namespace new_light_wattage_l2676_267604

theorem new_light_wattage (original_wattage : ℝ) (percentage_increase : ℝ) :
  original_wattage = 80 →
  percentage_increase = 25 →
  original_wattage * (1 + percentage_increase / 100) = 100 := by
  sorry

end new_light_wattage_l2676_267604


namespace smallest_fraction_l2676_267698

theorem smallest_fraction (x : ℝ) (h : x = 7) : 
  6 / (x + 1) < 6 / x ∧ 
  6 / (x + 1) < 6 / (x - 1) ∧ 
  6 / (x + 1) < x / 6 ∧ 
  6 / (x + 1) < (x + 1) / 6 := by
sorry

end smallest_fraction_l2676_267698


namespace right_triangle_solution_l2676_267665

theorem right_triangle_solution :
  ∃ (x : ℝ), x > 0 ∧
  (4 * x + 2) > 0 ∧
  ((x - 3)^2) > 0 ∧
  (5 * x + 1) > 0 ∧
  (4 * x + 2)^2 + (x - 3)^4 = (5 * x + 1)^2 ∧
  x = Real.sqrt (3/2) :=
by sorry

end right_triangle_solution_l2676_267665


namespace expression_value_l2676_267628

theorem expression_value (x y : ℝ) (hx : x = 3) (hy : y = 2) :
  3 * x - 4 * y = 1 := by
  sorry

end expression_value_l2676_267628


namespace division_problem_l2676_267692

theorem division_problem (x y : ℕ+) (h1 : x = 10 * y + 3) (h2 : 2 * x = 21 * y + 1) : 
  11 * y - x = 2 := by
  sorry

end division_problem_l2676_267692


namespace goods_payment_calculation_l2676_267658

/-- Calculates the final amount to be paid for goods after rebate and sales tax. -/
def final_amount (total_cost rebate_percent sales_tax_percent : ℚ) : ℚ :=
  let rebate_amount := (rebate_percent / 100) * total_cost
  let amount_after_rebate := total_cost - rebate_amount
  let sales_tax := (sales_tax_percent / 100) * amount_after_rebate
  amount_after_rebate + sales_tax

/-- Proves that given a total cost of 6650, a rebate of 6%, and a sales tax of 10%,
    the final amount to be paid is 6876.10. -/
theorem goods_payment_calculation :
  final_amount 6650 6 10 = 6876.1 := by
  sorry

end goods_payment_calculation_l2676_267658


namespace return_trip_time_l2676_267619

/-- The time taken for a return trip given the conditions of the original journey -/
theorem return_trip_time 
  (total_distance : ℝ) 
  (uphill_speed downhill_speed : ℝ)
  (forward_time : ℝ)
  (h1 : total_distance = 21)
  (h2 : uphill_speed = 4)
  (h3 : downhill_speed = 6)
  (h4 : forward_time = 4.25)
  (h5 : ∃ (uphill_distance downhill_distance : ℝ), 
    uphill_distance + downhill_distance = total_distance ∧
    uphill_distance / uphill_speed + downhill_distance / downhill_speed = forward_time) :
  ∃ (return_time : ℝ), return_time = 4.5 := by
sorry

end return_trip_time_l2676_267619


namespace g_monotone_and_range_l2676_267664

noncomputable def g (b : ℝ) (x : ℝ) : ℝ := (2^x + b) / (2^x - b)

theorem g_monotone_and_range (b : ℝ) :
  (b < 0 → ∀ x y : ℝ, x < y → g b x < g b y) ∧
  (b = -1 → ∀ a : ℝ, (∀ x : ℝ, g (-1) (x^2 + 1) + g (-1) (3 - a*x) > 0) ↔ -4 < a ∧ a < 4) :=
sorry

end g_monotone_and_range_l2676_267664


namespace stream_speed_l2676_267607

/-- The speed of the stream given rowing conditions -/
theorem stream_speed (downstream_distance : ℝ) (upstream_distance : ℝ)
                     (downstream_time : ℝ) (upstream_time : ℝ)
                     (h1 : downstream_distance = 120)
                     (h2 : upstream_distance = 90)
                     (h3 : downstream_time = 4)
                     (h4 : upstream_time = 6) :
  ∃ (boat_speed stream_speed : ℝ),
    downstream_distance = (boat_speed + stream_speed) * downstream_time ∧
    upstream_distance = (boat_speed - stream_speed) * upstream_time ∧
    stream_speed = 7.5 := by
  sorry

end stream_speed_l2676_267607


namespace power_of_product_l2676_267649

theorem power_of_product (x y : ℝ) : (-2 * x^2 * y)^2 = 4 * x^4 * y^2 := by
  sorry

end power_of_product_l2676_267649


namespace car_resale_gain_percentage_car_resale_specific_case_l2676_267608

/-- Calculates the gain percentage when reselling a car --/
theorem car_resale_gain_percentage 
  (original_price : ℝ) 
  (loss_percentage : ℝ) 
  (resale_price : ℝ) : ℝ :=
  let first_sale_price := original_price * (1 - loss_percentage / 100)
  let gain := resale_price - first_sale_price
  let gain_percentage := (gain / first_sale_price) * 100
  gain_percentage

/-- Proves that the gain percentage is approximately 3.55% for the given scenario --/
theorem car_resale_specific_case : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
  |car_resale_gain_percentage 52941.17647058824 15 54000 - 3.55| < ε :=
sorry

end car_resale_gain_percentage_car_resale_specific_case_l2676_267608


namespace green_hat_cost_l2676_267644

/-- Proves that the cost of each green hat is $1 given the conditions of the problem -/
theorem green_hat_cost (total_hats : ℕ) (blue_hat_cost : ℕ) (total_price : ℕ) (green_hats : ℕ) :
  total_hats = 85 →
  blue_hat_cost = 6 →
  total_price = 600 →
  green_hats = 90 →
  ∃ (green_hat_cost : ℕ), green_hat_cost = 1 ∧
    total_price = blue_hat_cost * (total_hats - green_hats) + green_hat_cost * green_hats :=
by
  sorry


end green_hat_cost_l2676_267644


namespace no_multiple_of_five_2c4_l2676_267652

theorem no_multiple_of_five_2c4 :
  ∀ C : ℕ, C < 10 → ¬(∃ k : ℕ, 200 + 10 * C + 4 = 5 * k) :=
by
  sorry

end no_multiple_of_five_2c4_l2676_267652


namespace equation_solution_l2676_267650

theorem equation_solution : ∃! x : ℚ, (7 * x - 2) / (x + 4) - 4 / (x + 4) = 2 / (x + 4) ∧ x = 8 / 7 := by
  sorry

end equation_solution_l2676_267650


namespace comic_book_arrangements_l2676_267672

def batman_comics : ℕ := 5
def superman_comics : ℕ := 3
def xmen_comics : ℕ := 6
def ironman_comics : ℕ := 4

def total_arrangements : ℕ := 2987520000

theorem comic_book_arrangements :
  (batman_comics.factorial * superman_comics.factorial * xmen_comics.factorial * ironman_comics.factorial) *
  (batman_comics + superman_comics + xmen_comics + ironman_comics).factorial =
  total_arrangements := by sorry

end comic_book_arrangements_l2676_267672


namespace zachs_babysitting_pay_rate_l2676_267660

/-- The problem of calculating Zach's babysitting pay rate -/
theorem zachs_babysitting_pay_rate 
  (bike_cost : ℚ)
  (weekly_allowance : ℚ)
  (lawn_mowing_pay : ℚ)
  (current_savings : ℚ)
  (additional_needed : ℚ)
  (babysitting_hours : ℚ)
  (h1 : bike_cost = 100)
  (h2 : weekly_allowance = 5)
  (h3 : lawn_mowing_pay = 10)
  (h4 : current_savings = 65)
  (h5 : additional_needed = 6)
  (h6 : babysitting_hours = 2)
  : ∃ (babysitting_rate : ℚ), 
    babysitting_rate = (current_savings + weekly_allowance + lawn_mowing_pay + additional_needed - bike_cost) / babysitting_hours ∧ 
    babysitting_rate = 3/2 := by
  sorry

end zachs_babysitting_pay_rate_l2676_267660


namespace olaf_game_score_l2676_267606

theorem olaf_game_score (dad_score : ℕ) : 
  (3 * dad_score + dad_score = 28) → dad_score = 7 := by
  sorry

end olaf_game_score_l2676_267606


namespace bear_mass_before_hibernation_l2676_267627

/-- The mass of a bear after hibernation, given as a fraction of its original mass -/
def mass_after_hibernation_fraction : ℚ := 80 / 100

/-- The mass of the bear after hibernation in kilograms -/
def mass_after_hibernation : ℚ := 220

/-- Theorem: If a bear loses 20% of its original mass during hibernation and 
    its mass after hibernation is 220 kg, then its mass before hibernation was 275 kg -/
theorem bear_mass_before_hibernation :
  mass_after_hibernation = mass_after_hibernation_fraction * (275 : ℚ) := by
  sorry

end bear_mass_before_hibernation_l2676_267627


namespace project_completion_time_l2676_267654

/-- The number of days it takes for person A to complete the project alone -/
def days_A : ℝ := 20

/-- The number of days it takes for person B to complete the project alone -/
def days_B : ℝ := 40

/-- The total duration of the project when A and B work together, and A quits 10 days before completion -/
def total_days : ℝ := 20

/-- The number of days A works before quitting -/
def days_A_works : ℝ := total_days - 10

theorem project_completion_time :
  (days_A_works * (1 / days_A + 1 / days_B)) + (10 * (1 / days_B)) = 1 :=
sorry

end project_completion_time_l2676_267654


namespace max_product_xy_l2676_267674

theorem max_product_xy (x y : ℝ) (hx : x > 0) (hy : y > 0)
  (eq1 : x + 1/y = 3) (eq2 : y + 2/x = 3) :
  ∃ (C : ℝ), C = x*y ∧ C ≤ 3 + Real.sqrt 7 ∧
  ∃ (x' y' : ℝ), x' > 0 ∧ y' > 0 ∧ x' + 1/y' = 3 ∧ y' + 2/x' = 3 ∧ x'*y' = 3 + Real.sqrt 7 :=
sorry

end max_product_xy_l2676_267674


namespace parabola_distance_l2676_267631

/-- The parabola y^2 = 8x with focus F and a point M satisfying |MO|^2 = 3|MF| -/
structure Parabola :=
  (F : ℝ × ℝ)
  (M : ℝ × ℝ)
  (h1 : F = (2, 0))
  (h2 : (M.2)^2 = 8 * M.1)
  (h3 : M.1^2 + M.2^2 = 3 * (dist M F))

/-- The distance between M and F is 3 -/
theorem parabola_distance (p : Parabola) : dist p.M p.F = 3 := by
  sorry

end parabola_distance_l2676_267631


namespace inverse_equals_k_times_self_l2676_267641

def A (d : ℝ) : Matrix (Fin 2) (Fin 2) ℝ := !![3, 4; 6, d]

theorem inverse_equals_k_times_self (d k : ℝ) :
  A d * (A d)⁻¹ = 1 ∧ (A d)⁻¹ = k • (A d) → d = -3 ∧ k = 1/33 := by
  sorry

end inverse_equals_k_times_self_l2676_267641


namespace cost_of_groceries_l2676_267638

/-- The cost of groceries problem -/
theorem cost_of_groceries
  (mango_cost : ℝ → ℝ)  -- Cost function for mangos (kg → $)
  (rice_cost : ℝ → ℝ)   -- Cost function for rice (kg → $)
  (flour_cost : ℝ → ℝ)  -- Cost function for flour (kg → $)
  (h1 : mango_cost 10 = rice_cost 10)  -- 10 kg mangos cost same as 10 kg rice
  (h2 : flour_cost 6 = rice_cost 2)    -- 6 kg flour costs same as 2 kg rice
  (h3 : ∀ x, flour_cost x = 21 * x)    -- Flour costs $21 per kg
  : mango_cost 4 + rice_cost 3 + flour_cost 5 = 546 := by
  sorry

#check cost_of_groceries

end cost_of_groceries_l2676_267638


namespace minimum_value_of_f_plus_f_l2676_267662

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := -x^3 + a*x^2 - 4

-- Define the derivative of f
def f' (a : ℝ) (x : ℝ) : ℝ := -3*x^2 + 2*a*x

theorem minimum_value_of_f_plus_f'_derivative (a : ℝ) :
  (∃ x, f' a x = 0 ∧ x = 2) →  -- f has an extremum at x = 2
  (∃ m n : ℝ, m ∈ Set.Icc (-1) 1 ∧ n ∈ Set.Icc (-1) 1 ∧
    ∀ p q : ℝ, p ∈ Set.Icc (-1) 1 → q ∈ Set.Icc (-1) 1 →
      f a m + f' a n ≤ f a p + f' a q) →
  (∃ m n : ℝ, m ∈ Set.Icc (-1) 1 ∧ n ∈ Set.Icc (-1) 1 ∧
    f a m + f' a n = -13) :=
by sorry

end minimum_value_of_f_plus_f_l2676_267662


namespace sum_of_integers_l2676_267666

theorem sum_of_integers (m n : ℕ+) (h1 : m * n = 2 * (m + n)) (h2 : m * n = 6 * (m - n)) :
  m + n = 9 := by sorry

end sum_of_integers_l2676_267666


namespace hyperbola_foci_l2676_267610

/-- The foci of the hyperbola y²/16 - x²/9 = 1 are located at (0, ±5) -/
theorem hyperbola_foci : 
  ∀ (x y : ℝ), (y^2 / 16 - x^2 / 9 = 1) → 
  ∃ (c : ℝ), c = 5 ∧ ((x = 0 ∧ y = c) ∨ (x = 0 ∧ y = -c)) :=
by sorry

end hyperbola_foci_l2676_267610


namespace red_paint_calculation_l2676_267678

/-- Given a mixture with a ratio of red paint to white paint and a total number of cans,
    calculate the number of cans of red paint required. -/
def red_paint_cans (red_ratio white_ratio total_cans : ℕ) : ℕ :=
  (red_ratio * total_cans) / (red_ratio + white_ratio)

/-- Theorem stating that for a 3:2 ratio of red to white paint and 30 total cans,
    18 cans of red paint are required. -/
theorem red_paint_calculation :
  red_paint_cans 3 2 30 = 18 := by
  sorry

end red_paint_calculation_l2676_267678


namespace unique_modular_inverse_l2676_267612

theorem unique_modular_inverse (p : Nat) (a : Nat) (h_p : p.Prime) (h_p_odd : p % 2 = 1)
  (h_a_range : 2 ≤ a ∧ a ≤ p - 2) :
  ∃! i : Nat, 2 ≤ i ∧ i ≤ p - 2 ∧ i ≠ a ∧ (i * a) % p = 1 := by
  sorry

end unique_modular_inverse_l2676_267612


namespace spheres_in_cone_radius_l2676_267609

/-- A right circular cone -/
structure Cone :=
  (base_radius : ℝ)
  (height : ℝ)

/-- A sphere -/
structure Sphere :=
  (radius : ℝ)

/-- Configuration of four spheres in a cone -/
structure SpheresInCone :=
  (cone : Cone)
  (sphere : Sphere)
  (tangent_to_base : Prop)
  (tangent_to_each_other : Prop)
  (tangent_to_side : Prop)

/-- Theorem stating the radius of spheres in the given configuration -/
theorem spheres_in_cone_radius 
  (config : SpheresInCone)
  (h_base_radius : config.cone.base_radius = 6)
  (h_height : config.cone.height = 15)
  (h_tangent_base : config.tangent_to_base)
  (h_tangent_each_other : config.tangent_to_each_other)
  (h_tangent_side : config.tangent_to_side) :
  config.sphere.radius = 15 / 11 :=
sorry

end spheres_in_cone_radius_l2676_267609


namespace max_cross_pattern_sum_l2676_267675

/-- Represents the cross-shaped pattern -/
structure CrossPattern where
  a : ℕ
  b : ℕ
  c : ℕ
  d : ℕ
  e : ℕ

/-- The set of available numbers -/
def availableNumbers : Finset ℕ := {2, 6, 9, 11, 14}

/-- Checks if the pattern satisfies the sum conditions -/
def isValidPattern (p : CrossPattern) : Prop :=
  p.a + p.b + p.e = p.a + p.c + p.e ∧
  p.a + p.c + p.e = p.b + p.d + p.e ∧
  p.a + p.d = p.b + p.c

/-- Checks if the pattern uses all available numbers exactly once -/
def usesAllNumbers (p : CrossPattern) : Prop :=
  {p.a, p.b, p.c, p.d, p.e} = availableNumbers

/-- The sum of any row, column, or diagonal in a valid pattern -/
def patternSum (p : CrossPattern) : ℕ := p.a + p.b + p.e

/-- Theorem: The maximum sum in a valid cross pattern is 31 -/
theorem max_cross_pattern_sum :
  ∀ p : CrossPattern,
    isValidPattern p →
    usesAllNumbers p →
    patternSum p ≤ 31 :=
sorry

end max_cross_pattern_sum_l2676_267675


namespace tin_in_mixed_alloy_tin_amount_in_new_alloy_l2676_267603

/-- Amount of tin in a mixture of two alloys -/
theorem tin_in_mixed_alloy (mass_A mass_B : ℝ) 
  (lead_tin_ratio_A : ℝ) (tin_copper_ratio_B : ℝ) : ℝ :=
  let tin_fraction_A := lead_tin_ratio_A / (1 + lead_tin_ratio_A)
  let tin_fraction_B := tin_copper_ratio_B / (1 + tin_copper_ratio_B)
  tin_fraction_A * mass_A + tin_fraction_B * mass_B

/-- The amount of tin in the new alloy is 221.25 kg -/
theorem tin_amount_in_new_alloy : 
  tin_in_mixed_alloy 170 250 (1/3) (3/5) = 221.25 := by
  sorry

end tin_in_mixed_alloy_tin_amount_in_new_alloy_l2676_267603
