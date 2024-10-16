import Mathlib

namespace NUMINAMATH_CALUDE_line_intersection_yz_plane_specific_line_intersection_l1575_157515

/-- The line passing through two given points intersects the yz-plane at a specific point. -/
theorem line_intersection_yz_plane (p₁ p₂ : ℝ × ℝ × ℝ) (h : p₁ ≠ p₂) :
  let line := λ t : ℝ => p₁ + t • (p₂ - p₁)
  ∃ t, line t = (0, 8, -5/2) :=
by
  sorry

/-- The specific instance of the line intersection problem. -/
theorem specific_line_intersection :
  let p₁ : ℝ × ℝ × ℝ := (3, 5, 1)
  let p₂ : ℝ × ℝ × ℝ := (5, 3, 6)
  let line := λ t : ℝ => p₁ + t • (p₂ - p₁)
  ∃ t, line t = (0, 8, -5/2) :=
by
  sorry

end NUMINAMATH_CALUDE_line_intersection_yz_plane_specific_line_intersection_l1575_157515


namespace NUMINAMATH_CALUDE_fraction_of_blue_cubes_received_l1575_157528

/-- Proves that Gage received 1/3 of Grady's blue cubes --/
theorem fraction_of_blue_cubes_received (grady_red_cubes grady_blue_cubes : ℕ)
  (gage_initial_red_cubes gage_initial_blue_cubes : ℕ)
  (gage_total_cubes_after : ℕ) :
  grady_red_cubes = 20 →
  grady_blue_cubes = 15 →
  gage_initial_red_cubes = 10 →
  gage_initial_blue_cubes = 12 →
  gage_total_cubes_after = 35 →
  (gage_total_cubes_after - (gage_initial_red_cubes + 2 / 5 * grady_red_cubes) - gage_initial_blue_cubes) / grady_blue_cubes = 1 / 3 :=
by sorry

end NUMINAMATH_CALUDE_fraction_of_blue_cubes_received_l1575_157528


namespace NUMINAMATH_CALUDE_share_calculation_l1575_157591

theorem share_calculation (total : ℝ) (a b c : ℝ) : 
  total = 500 →
  a = (2/3) * (b + c) →
  b = (2/3) * (a + c) →
  a + b + c = total →
  a = 200 := by
sorry

end NUMINAMATH_CALUDE_share_calculation_l1575_157591


namespace NUMINAMATH_CALUDE_parallelogram_area_l1575_157532

def v : Fin 2 → ℝ := ![6, -4]
def w : Fin 2 → ℝ := ![8, -1]

theorem parallelogram_area : 
  abs (Matrix.det !![v 0, v 1; 2 * w 0, 2 * w 1]) = 52 := by sorry

end NUMINAMATH_CALUDE_parallelogram_area_l1575_157532


namespace NUMINAMATH_CALUDE_brown_mms_in_first_bag_l1575_157550

/-- The number of bags of M&M's. -/
def num_bags : ℕ := 5

/-- The number of brown M&M's in the second bag. -/
def second_bag : ℕ := 12

/-- The number of brown M&M's in the third bag. -/
def third_bag : ℕ := 8

/-- The number of brown M&M's in the fourth bag. -/
def fourth_bag : ℕ := 8

/-- The number of brown M&M's in the fifth bag. -/
def fifth_bag : ℕ := 3

/-- The average number of brown M&M's per bag. -/
def average : ℕ := 8

/-- Theorem stating the number of brown M&M's in the first bag. -/
theorem brown_mms_in_first_bag :
  ∃ (first_bag : ℕ),
    (first_bag + second_bag + third_bag + fourth_bag + fifth_bag) / num_bags = average ∧
    first_bag = 9 := by
  sorry

end NUMINAMATH_CALUDE_brown_mms_in_first_bag_l1575_157550


namespace NUMINAMATH_CALUDE_sixth_segment_length_l1575_157579

def segment_lengths (a : Fin 7 → ℕ) : Prop :=
  a 0 = 1 ∧ a 6 = 21 ∧ 
  (∀ i j, i < j → a i < a j) ∧
  (∀ i j k, i < j ∧ j < k → a i + a j ≤ a k)

theorem sixth_segment_length (a : Fin 7 → ℕ) (h : segment_lengths a) : a 5 = 13 := by
  sorry

end NUMINAMATH_CALUDE_sixth_segment_length_l1575_157579


namespace NUMINAMATH_CALUDE_complex_multiplication_simplification_l1575_157593

-- Define the imaginary unit i
noncomputable def i : ℂ := Complex.I

-- Define the theorem
theorem complex_multiplication_simplification (x y : ℝ) :
  (x + 3 * i * y) * (x - 3 * i * y) = x^2 - 9 * y^2 := by
  sorry

end NUMINAMATH_CALUDE_complex_multiplication_simplification_l1575_157593


namespace NUMINAMATH_CALUDE_complex_multiplication_l1575_157563

def A : ℂ := 6 - 2 * Complex.I
def M : ℂ := -3 + 4 * Complex.I
def S : ℂ := 2 * Complex.I
def P : ℂ := 3
def C : ℂ := 1 + Complex.I

theorem complex_multiplication :
  (A - M + S - P) * C = 10 + 2 * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_multiplication_l1575_157563


namespace NUMINAMATH_CALUDE_garden_area_is_135_l1575_157575

/-- Represents a rectangular garden with fence posts -/
structure Garden where
  total_posts : ℕ
  post_spacing : ℕ
  short_side_posts : ℕ
  long_side_posts : ℕ

/-- Checks if the garden configuration is valid -/
def is_valid_garden (g : Garden) : Prop :=
  g.total_posts = 24 ∧
  g.post_spacing = 3 ∧
  g.short_side_posts * 2 + g.long_side_posts * 2 - 4 = g.total_posts ∧
  g.long_side_posts = (g.short_side_posts * 3 + 1) / 2

/-- Calculates the area of the garden -/
def garden_area (g : Garden) : ℕ :=
  (g.short_side_posts - 1) * g.post_spacing * ((g.long_side_posts - 1) * g.post_spacing)

theorem garden_area_is_135 (g : Garden) (h : is_valid_garden g) : garden_area g = 135 := by
  sorry

end NUMINAMATH_CALUDE_garden_area_is_135_l1575_157575


namespace NUMINAMATH_CALUDE_loan_split_l1575_157572

theorem loan_split (total : ℝ) (years1 rate1 years2 rate2 : ℝ) 
  (h1 : total = 2704)
  (h2 : years1 = 8)
  (h3 : rate1 = 0.03)
  (h4 : years2 = 3)
  (h5 : rate2 = 0.05)
  (h6 : ∃ x : ℝ, x * years1 * rate1 = (total - x) * years2 * rate2) :
  ∃ y : ℝ, y = total - 1664 ∧ y * years1 * rate1 = (total - y) * years2 * rate2 := by
  sorry

end NUMINAMATH_CALUDE_loan_split_l1575_157572


namespace NUMINAMATH_CALUDE_inequality_equivalence_l1575_157570

theorem inequality_equivalence (x : ℝ) :
  |2*x - 1| < |x| + 1 ↔ 0 < x ∧ x < 2 := by
sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l1575_157570


namespace NUMINAMATH_CALUDE_linear_function_passes_through_point_l1575_157521

/-- The linear function f(x) = -2x - 6 passes through the point (-4, 2) -/
theorem linear_function_passes_through_point :
  let f : ℝ → ℝ := λ x => -2 * x - 6
  f (-4) = 2 := by sorry

end NUMINAMATH_CALUDE_linear_function_passes_through_point_l1575_157521


namespace NUMINAMATH_CALUDE_figure_area_l1575_157573

/-- The area of a rectangle given its width and height -/
def rectangle_area (width : ℕ) (height : ℕ) : ℕ := width * height

/-- The total area of three rectangles -/
def total_area (rect1_width rect1_height rect2_width rect2_height rect3_width rect3_height : ℕ) : ℕ :=
  rectangle_area rect1_width rect1_height +
  rectangle_area rect2_width rect2_height +
  rectangle_area rect3_width rect3_height

/-- Theorem: The total area of the figure is 71 square units -/
theorem figure_area :
  total_area 7 7 3 2 4 4 = 71 := by
  sorry

end NUMINAMATH_CALUDE_figure_area_l1575_157573


namespace NUMINAMATH_CALUDE_units_digit_of_seven_to_six_to_five_l1575_157587

theorem units_digit_of_seven_to_six_to_five (n : ℕ) :
  7^(6^5) ≡ 1 [MOD 10] :=
by sorry

end NUMINAMATH_CALUDE_units_digit_of_seven_to_six_to_five_l1575_157587


namespace NUMINAMATH_CALUDE_remainder_theorem_l1575_157583

-- Define the polynomial
def p (x : ℝ) : ℝ := x^5 - x^3 + 3*x^2 + 2

-- State the theorem
theorem remainder_theorem :
  ∃ q : ℝ → ℝ, p = (λ x => (x + 2) * q x + (-10)) :=
sorry

end NUMINAMATH_CALUDE_remainder_theorem_l1575_157583


namespace NUMINAMATH_CALUDE_solution_satisfies_system_l1575_157537

-- Define the system of equations
def equation1 (x y : ℝ) : Prop := x + y = 7
def equation2 (x y : ℝ) : Prop := 2 * x - y = 2

-- State the theorem
theorem solution_satisfies_system :
  ∃ (x y : ℝ), equation1 x y ∧ equation2 x y ∧ x = 3 ∧ y = 4 := by
  sorry

end NUMINAMATH_CALUDE_solution_satisfies_system_l1575_157537


namespace NUMINAMATH_CALUDE_bag_probability_l1575_157516

theorem bag_probability (n : ℕ) : 
  (6 : ℚ) / (6 + n) = 2 / 5 → n = 9 := by
sorry

end NUMINAMATH_CALUDE_bag_probability_l1575_157516


namespace NUMINAMATH_CALUDE_inequality_range_l1575_157533

theorem inequality_range (a x : ℝ) : 
  (∀ a, |a| ≤ 1 → x^2 + (a - 6) * x + (9 - 3 * a) > 0) ↔ 
  (x < 2 ∨ x > 4) := by sorry

end NUMINAMATH_CALUDE_inequality_range_l1575_157533


namespace NUMINAMATH_CALUDE_board_highest_point_l1575_157578

/-- Represents a rectangular board with length and height -/
structure Board where
  length : ℝ
  height : ℝ

/-- Calculates the distance from the ground to the highest point of an inclined board -/
def highestPoint (board : Board) (angle : ℝ) : ℝ :=
  sorry

theorem board_highest_point :
  let board := Board.mk 64 4
  let angle := 30 * π / 180
  ∃ (a b c : ℕ), 
    (highestPoint board angle = a + b * Real.sqrt c) ∧
    (a = 32) ∧ (b = 2) ∧ (c = 3) ∧
    (∀ (p : ℕ), Nat.Prime p → ¬(p^2 ∣ c)) :=
  sorry

end NUMINAMATH_CALUDE_board_highest_point_l1575_157578


namespace NUMINAMATH_CALUDE_josh_spending_l1575_157526

def initial_amount : ℚ := 9
def drink_cost : ℚ := 1.75
def final_amount : ℚ := 6

theorem josh_spending (amount_spent_after_drink : ℚ) : 
  initial_amount - drink_cost - amount_spent_after_drink = final_amount → 
  amount_spent_after_drink = 1.25 := by
sorry

end NUMINAMATH_CALUDE_josh_spending_l1575_157526


namespace NUMINAMATH_CALUDE_mouse_lives_count_l1575_157538

def cat_lives : ℕ := 9
def dog_lives : ℕ := cat_lives - 3
def mouse_lives : ℕ := dog_lives + 7

theorem mouse_lives_count : mouse_lives = 13 := by
  sorry

end NUMINAMATH_CALUDE_mouse_lives_count_l1575_157538


namespace NUMINAMATH_CALUDE_cylinder_surface_area_l1575_157566

/-- The surface area of a cylinder given its unfolded lateral surface dimensions -/
theorem cylinder_surface_area (h w : ℝ) (h_pos : h > 0) (w_pos : w > 0)
  (h_is_6pi : h = 6 * Real.pi) (w_is_4pi : w = 4 * Real.pi) :
  let r := min (h / (2 * Real.pi)) (w / (2 * Real.pi))
  let surface_area := h * w + 2 * Real.pi * r^2
  surface_area = 24 * Real.pi^2 + 18 * Real.pi ∨
  surface_area = 24 * Real.pi^2 + 8 * Real.pi :=
sorry

end NUMINAMATH_CALUDE_cylinder_surface_area_l1575_157566


namespace NUMINAMATH_CALUDE_coefficient_of_x_l1575_157576

theorem coefficient_of_x (x : ℝ) : 
  let expansion := (1 + x) * (x - 2/x)^3
  ∃ (a b c d e : ℝ), expansion = a*x^3 + b*x^2 + c*x + d + e/x + e/(x^2) + e/(x^3) ∧ c = -6 :=
sorry

end NUMINAMATH_CALUDE_coefficient_of_x_l1575_157576


namespace NUMINAMATH_CALUDE_fuel_usage_proof_l1575_157545

theorem fuel_usage_proof (x : ℝ) : 
  x > 0 ∧ x + 0.8 * x = 27 → x = 15 := by
  sorry

end NUMINAMATH_CALUDE_fuel_usage_proof_l1575_157545


namespace NUMINAMATH_CALUDE_movie_theater_revenue_is_6600_l1575_157529

/-- Calculates the total revenue of a movie theater given ticket prices and quantities sold --/
def movie_theater_revenue (matinee_price evening_price three_d_price : ℕ) 
                          (matinee_sold evening_sold three_d_sold : ℕ) : ℕ :=
  matinee_price * matinee_sold + evening_price * evening_sold + three_d_price * three_d_sold

/-- Theorem stating that the movie theater's revenue is $6600 given the specified prices and quantities --/
theorem movie_theater_revenue_is_6600 :
  movie_theater_revenue 5 12 20 200 300 100 = 6600 := by
  sorry

end NUMINAMATH_CALUDE_movie_theater_revenue_is_6600_l1575_157529


namespace NUMINAMATH_CALUDE_fifteenth_odd_multiple_of_5_fifteenth_odd_multiple_of_5_is_odd_fifteenth_odd_multiple_of_5_is_multiple_of_5_l1575_157506

/-- The nth positive odd multiple of 5 -/
def oddMultipleOf5 (n : ℕ) : ℕ := 10 * n - 5

theorem fifteenth_odd_multiple_of_5 : oddMultipleOf5 15 = 145 := by
  sorry

/-- The 15th positive odd multiple of 5 is odd -/
theorem fifteenth_odd_multiple_of_5_is_odd : Odd (oddMultipleOf5 15) := by
  sorry

/-- The 15th positive odd multiple of 5 is a multiple of 5 -/
theorem fifteenth_odd_multiple_of_5_is_multiple_of_5 : ∃ k : ℕ, oddMultipleOf5 15 = 5 * k := by
  sorry

end NUMINAMATH_CALUDE_fifteenth_odd_multiple_of_5_fifteenth_odd_multiple_of_5_is_odd_fifteenth_odd_multiple_of_5_is_multiple_of_5_l1575_157506


namespace NUMINAMATH_CALUDE_problem_statement_l1575_157513

theorem problem_statement (x y z : ℝ) 
  (sum_eq : x + y + z = 12) 
  (sum_sq_eq : x^2 + y^2 + z^2 = 54) : 
  (9 ≤ x*y ∧ x*y ≤ 25) ∧ 
  (9 ≤ y*z ∧ y*z ≤ 25) ∧ 
  (9 ≤ z*x ∧ z*x ≤ 25) ∧
  ((x ≤ 3 ∧ (y ≥ 5 ∨ z ≥ 5)) ∨ 
   (y ≤ 3 ∧ (x ≥ 5 ∨ z ≥ 5)) ∨ 
   (z ≤ 3 ∧ (x ≥ 5 ∨ y ≥ 5))) := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l1575_157513


namespace NUMINAMATH_CALUDE_three_digit_square_end_same_l1575_157598

theorem three_digit_square_end_same (A : ℕ) : 
  (100 ≤ A ∧ A < 1000) ∧ (A^2 % 1000 = A) ↔ (A = 376 ∨ A = 625) :=
by sorry

end NUMINAMATH_CALUDE_three_digit_square_end_same_l1575_157598


namespace NUMINAMATH_CALUDE_equation_one_l1575_157551

theorem equation_one (x : ℝ) : x * (5 * x + 4) = 5 * x + 4 ↔ x = -4/5 ∨ x = 1 := by sorry

end NUMINAMATH_CALUDE_equation_one_l1575_157551


namespace NUMINAMATH_CALUDE_asterisk_replacement_l1575_157560

theorem asterisk_replacement : ∃! (n : ℝ), n > 0 ∧ (n / 18) * (n / 72) = 1 := by
  sorry

end NUMINAMATH_CALUDE_asterisk_replacement_l1575_157560


namespace NUMINAMATH_CALUDE_race_probability_l1575_157556

theorem race_probability (total_cars : ℕ) (prob_Y prob_Z prob_XYZ : ℚ) : 
  total_cars = 8 →
  prob_Y = 1/4 →
  prob_Z = 1/3 →
  prob_XYZ = 13/12 →
  ∃ prob_X : ℚ, prob_X + prob_Y + prob_Z = prob_XYZ ∧ prob_X = 1/2 :=
by sorry

end NUMINAMATH_CALUDE_race_probability_l1575_157556


namespace NUMINAMATH_CALUDE_truck_driver_gas_cost_l1575_157518

/-- A truck driver's gas cost problem -/
theorem truck_driver_gas_cost 
  (miles_per_gallon : ℝ) 
  (miles_per_hour : ℝ) 
  (pay_per_mile : ℝ) 
  (total_pay : ℝ) 
  (drive_time : ℝ) 
  (h1 : miles_per_gallon = 10)
  (h2 : miles_per_hour = 30)
  (h3 : pay_per_mile = 0.5)
  (h4 : total_pay = 90)
  (h5 : drive_time = 10) :
  (total_pay / (miles_per_hour * drive_time / miles_per_gallon)) = 3 := by
sorry


end NUMINAMATH_CALUDE_truck_driver_gas_cost_l1575_157518


namespace NUMINAMATH_CALUDE_probability_defective_smartphones_l1575_157571

/-- Represents the probability of selecting two defective smartphones --/
def probability_two_defective (total : ℕ) (defective : ℕ) : ℚ :=
  (defective : ℚ) / (total : ℚ) * ((defective - 1) : ℚ) / ((total - 1) : ℚ)

/-- Theorem stating the probability of selecting two defective smartphones --/
theorem probability_defective_smartphones :
  let total := 250
  let type_a_total := 100
  let type_a_defective := 30
  let type_b_total := 80
  let type_b_defective := 25
  let type_c_total := 70
  let type_c_defective := 21
  let total_defective := type_a_defective + type_b_defective + type_c_defective
  abs (probability_two_defective total total_defective - 0.0916) < 0.0001 :=
sorry

end NUMINAMATH_CALUDE_probability_defective_smartphones_l1575_157571


namespace NUMINAMATH_CALUDE_factory_underpayment_l1575_157535

/-- The hourly wage in yuan -/
def hourly_wage : ℚ := 6

/-- The nominal work day duration in hours -/
def nominal_work_day : ℚ := 8

/-- The time for clock hands to coincide in the inaccurate clock (in minutes) -/
def inaccurate_coincidence_time : ℚ := 69

/-- The time for clock hands to coincide in an accurate clock (in minutes) -/
def accurate_coincidence_time : ℚ := 720 / 11

/-- Calculate the actual work time based on the inaccurate clock -/
def actual_work_time : ℚ :=
  (inaccurate_coincidence_time * nominal_work_day) / accurate_coincidence_time

/-- Calculate the underpayment amount -/
def underpayment : ℚ := hourly_wage * (actual_work_time - nominal_work_day)

theorem factory_underpayment :
  underpayment = 13/5 :=
by sorry

end NUMINAMATH_CALUDE_factory_underpayment_l1575_157535


namespace NUMINAMATH_CALUDE_boat_journey_time_l1575_157514

/-- Calculates the total journey time for a boat traveling upstream and downstream in a river -/
theorem boat_journey_time 
  (river_speed : ℝ) 
  (boat_speed : ℝ) 
  (distance : ℝ) 
  (h1 : river_speed = 2) 
  (h2 : boat_speed = 6) 
  (h3 : distance = 48) : 
  (distance / (boat_speed - river_speed)) + (distance / (boat_speed + river_speed)) = 18 :=
by sorry

end NUMINAMATH_CALUDE_boat_journey_time_l1575_157514


namespace NUMINAMATH_CALUDE_quadratic_function_values_l1575_157569

def f (a b x : ℝ) : ℝ := a * x^2 - 2 * a * x + 2 + b

theorem quadratic_function_values (a b : ℝ) (h : a ≠ 0) :
  (∀ x ∈ Set.Icc 2 3, f a b x ≤ 5) ∧
  (∃ x ∈ Set.Icc 2 3, f a b x = 5) ∧
  (∀ x ∈ Set.Icc 2 3, f a b x ≥ 2) ∧
  (∃ x ∈ Set.Icc 2 3, f a b x = 2) →
  ((a = 1 ∧ b = 0) ∨ (a = -1 ∧ b = 3)) :=
sorry

end NUMINAMATH_CALUDE_quadratic_function_values_l1575_157569


namespace NUMINAMATH_CALUDE_abc_maximum_l1575_157530

theorem abc_maximum (a b c : ℝ) (h1 : 2 * a + b = 4) (h2 : a * b + c = 5) :
  ∃ (max : ℝ), ∀ (x y z : ℝ), 2 * x + y = 4 → x * y + z = 5 → x * y * z ≤ max ∧ a * b * c = max :=
by
  sorry

end NUMINAMATH_CALUDE_abc_maximum_l1575_157530


namespace NUMINAMATH_CALUDE_quadratic_factorization_l1575_157525

/-- A quadratic expression can be factored completely if and only if its discriminant is a perfect square. -/
def is_factorable (a b c : ℝ) : Prop :=
  ∃ k : ℤ, (b^2 - 4*a*c : ℝ) = (k : ℝ)^2

theorem quadratic_factorization (m : ℝ) :
  (is_factorable 1 (3 - m) 25) → (m = -7 ∨ m = 13) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l1575_157525


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l1575_157504

/-- The sum of the first n terms of a geometric sequence -/
def geometric_sum (a₁ : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a₁ * (1 - r^n) / (1 - r)

/-- The problem statement -/
theorem geometric_sequence_sum (n : ℕ) :
  geometric_sum 1 (1/3) n = 121/81 → n = 5 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l1575_157504


namespace NUMINAMATH_CALUDE_system_solution_l1575_157508

theorem system_solution (x y : ℝ) : 
  (2 * x + 3 * y = -11 ∧ 6 * x - 5 * y = 9) ↔ (x = -1 ∧ y = -3) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l1575_157508


namespace NUMINAMATH_CALUDE_line_equation_l1575_157585

-- Define the circle F
def circle_F (x y : ℝ) : Prop := x^2 + y^2 - 2*x = 0

-- Define the parabola C
def parabola_C (x y : ℝ) : Prop := y^2 = 4*x

-- Define a line passing through a point
def line_through_point (k m : ℝ) (x y : ℝ) : Prop := y = k*(x - m)

-- Define the arithmetic sequence condition
def arithmetic_sequence (a b c : ℝ) : Prop := b - a = c - b

-- Main theorem
theorem line_equation (P M N Q : ℝ × ℝ) :
  let (xp, yp) := P
  let (xm, ym) := M
  let (xn, yn) := N
  let (xq, yq) := Q
  (∃ k : ℝ, 
    (∀ x y, line_through_point k 1 x y → (circle_F x y ∨ parabola_C x y)) ∧
    line_through_point k 1 xp yp ∧
    line_through_point k 1 xm ym ∧
    line_through_point k 1 xn yn ∧
    line_through_point k 1 xq yq ∧
    arithmetic_sequence (Real.sqrt ((xp-1)^2 + yp^2)) (Real.sqrt ((xm-1)^2 + ym^2)) (Real.sqrt ((xn-1)^2 + yn^2)) ∧
    arithmetic_sequence (Real.sqrt ((xm-1)^2 + ym^2)) (Real.sqrt ((xn-1)^2 + yn^2)) (Real.sqrt ((xq-1)^2 + yq^2))) →
  (k = Real.sqrt 2 ∨ k = -Real.sqrt 2) :=
sorry

end NUMINAMATH_CALUDE_line_equation_l1575_157585


namespace NUMINAMATH_CALUDE_difference_of_squares_l1575_157559

theorem difference_of_squares (a b : ℝ) : (a + b) * (a - b) = a^2 - b^2 := by sorry

end NUMINAMATH_CALUDE_difference_of_squares_l1575_157559


namespace NUMINAMATH_CALUDE_no_solution_exists_l1575_157527

theorem no_solution_exists :
  ¬ ∃ (a b c x y z : ℕ+),
    (a ≥ b) ∧ (b ≥ c) ∧
    (x ≥ y) ∧ (y ≥ z) ∧
    (2 * a + b + 4 * c = 4 * x * y * z) ∧
    (2 * x + y + 4 * z = 4 * a * b * c) :=
by sorry

end NUMINAMATH_CALUDE_no_solution_exists_l1575_157527


namespace NUMINAMATH_CALUDE_cubic_function_properties_l1575_157536

/-- The cubic function f(x) = x^3 - 12x + 12 -/
def f (x : ℝ) : ℝ := x^3 - 12*x + 12

theorem cubic_function_properties :
  (∃ x : ℝ, f x = 28) ∧  -- Maximum value is 28
  (f 2 = -4) ∧           -- Extreme value at x = 2 is -4
  (∀ x ∈ Set.Icc (-3) 3, f x ≥ -4) ∧  -- Minimum value on [-3, 3] is -4
  (∃ x ∈ Set.Icc (-3) 3, f x = -4) -- The minimum is attained on [-3, 3]
  := by sorry

end NUMINAMATH_CALUDE_cubic_function_properties_l1575_157536


namespace NUMINAMATH_CALUDE_apple_pear_ratio_l1575_157509

/-- Proves that the ratio of initial apples to initial pears is 2:1 given the conditions --/
theorem apple_pear_ratio (initial_pears initial_oranges : ℕ) 
  (fruits_given_away fruits_left : ℕ) : 
  initial_pears = 10 →
  initial_oranges = 20 →
  fruits_given_away = 2 →
  fruits_left = 44 →
  ∃ (initial_apples : ℕ), 
    initial_apples - fruits_given_away + 
    (initial_pears - fruits_given_away) + 
    (initial_oranges - fruits_given_away) = fruits_left ∧
    initial_apples / initial_pears = 2 := by
  sorry

end NUMINAMATH_CALUDE_apple_pear_ratio_l1575_157509


namespace NUMINAMATH_CALUDE_no_fixed_point_for_h_h_condition_l1575_157574

-- Define the function h
def h (x : ℝ) : ℝ := x - 6

-- Theorem statement
theorem no_fixed_point_for_h : ¬ ∃ x : ℝ, h x = x := by
  sorry

-- Condition from the original problem
theorem h_condition (x : ℝ) : h (3 * x + 2) = 3 * x - 4 := by
  sorry

end NUMINAMATH_CALUDE_no_fixed_point_for_h_h_condition_l1575_157574


namespace NUMINAMATH_CALUDE_continuous_piecewise_function_sum_l1575_157594

/-- A piecewise function g(x) defined by three parts --/
noncomputable def g (a c : ℝ) (x : ℝ) : ℝ :=
  if x > 3 then 2 * a * x + 4
  else if 0 ≤ x ∧ x ≤ 3 then 2 * x - 6
  else 3 * x - c

/-- Theorem stating that if g is continuous, then a + c = 16/3 --/
theorem continuous_piecewise_function_sum (a c : ℝ) :
  Continuous (g a c) → a + c = 16/3 := by
  sorry

end NUMINAMATH_CALUDE_continuous_piecewise_function_sum_l1575_157594


namespace NUMINAMATH_CALUDE_dinitrogen_pentoxide_molecular_weight_l1575_157531

/-- The molecular weight of Dinitrogen pentoxide in grams per mole. -/
def molecular_weight : ℝ := 108

/-- The number of moles given in the problem. -/
def given_moles : ℝ := 9

/-- The total weight of the given moles in grams. -/
def total_weight : ℝ := 972

/-- Theorem stating that the molecular weight of Dinitrogen pentoxide is 108 grams/mole. -/
theorem dinitrogen_pentoxide_molecular_weight :
  molecular_weight = total_weight / given_moles :=
sorry

end NUMINAMATH_CALUDE_dinitrogen_pentoxide_molecular_weight_l1575_157531


namespace NUMINAMATH_CALUDE_grocery_expense_l1575_157540

/-- Calculates the amount spent on groceries given credit card transactions -/
theorem grocery_expense (initial_balance new_balance returns : ℚ) : 
  initial_balance = 126 ∧ 
  new_balance = 171 ∧ 
  returns = 45 → 
  ∃ (grocery_expense : ℚ), 
    grocery_expense = 60 ∧ 
    initial_balance + grocery_expense + (grocery_expense / 2) - returns = new_balance := by
  sorry

end NUMINAMATH_CALUDE_grocery_expense_l1575_157540


namespace NUMINAMATH_CALUDE_mn_squared_equals_half_sum_l1575_157554

/-- Represents a quadrilateral ABCD with a segment MN parallel to CD -/
structure QuadrilateralWithSegment where
  /-- Length of segment from A parallel to CD intersecting BC -/
  a : ℝ
  /-- Length of segment from B parallel to CD intersecting AD -/
  b : ℝ
  /-- Length of CD -/
  c : ℝ
  /-- Length of MN -/
  mn : ℝ
  /-- MN is parallel to CD -/
  mn_parallel_cd : True
  /-- M lies on BC and N lies on AD -/
  m_on_bc_n_on_ad : True
  /-- MN divides the quadrilateral ABCD into two equal areas -/
  mn_divides_equally : True

/-- Theorem stating the relationship between MN, a, b, and c -/
theorem mn_squared_equals_half_sum (q : QuadrilateralWithSegment) :
  q.mn ^ 2 = (q.a * q.b + q.c ^ 2) / 2 := by sorry

end NUMINAMATH_CALUDE_mn_squared_equals_half_sum_l1575_157554


namespace NUMINAMATH_CALUDE_axis_of_symmetry_shifted_sine_l1575_157519

theorem axis_of_symmetry_shifted_sine (k : ℤ) :
  let f : ℝ → ℝ := λ x => 2 * Real.sin (2 * (x + π / 12))
  let axis : ℝ := k * π / 2 + π / 6
  ∀ x : ℝ, f (axis - x) = f (axis + x) :=
by
  sorry

end NUMINAMATH_CALUDE_axis_of_symmetry_shifted_sine_l1575_157519


namespace NUMINAMATH_CALUDE_area_of_inscribed_rectangle_l1575_157544

/-- Rectangle ABCD inscribed in triangle EFG with the following properties:
    - Side AD of the rectangle is on side EG of the triangle
    - Triangle's altitude from F to side EG is 7 inches
    - EG = 10 inches
    - Length of segment AB is equal to half the length of segment AD -/
structure InscribedRectangle where
  EG : ℝ
  altitude : ℝ
  AB : ℝ
  AD : ℝ
  h_EG : EG = 10
  h_altitude : altitude = 7
  h_AB_AD : AB = AD / 2

/-- The area of the inscribed rectangle ABCD is 1225/72 square inches -/
theorem area_of_inscribed_rectangle (rect : InscribedRectangle) :
  rect.AB * rect.AD = 1225 / 72 := by
  sorry

end NUMINAMATH_CALUDE_area_of_inscribed_rectangle_l1575_157544


namespace NUMINAMATH_CALUDE_nested_fraction_equality_l1575_157505

theorem nested_fraction_equality : (1 : ℚ) / (2 + 1 / (3 + 1 / 4)) = 13 / 30 := by
  sorry

end NUMINAMATH_CALUDE_nested_fraction_equality_l1575_157505


namespace NUMINAMATH_CALUDE_fraction_simplification_l1575_157588

theorem fraction_simplification (x y : ℝ) 
  (hx : x ≠ 0) 
  (hy : y ≠ 0) 
  (hden : y^2 - 1/x^2 ≠ 0) : 
  (x^2 - 1/y^2) / (y^2 - 1/x^2) = x^2 / y^2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1575_157588


namespace NUMINAMATH_CALUDE_exist_decreasing_lcm_sequence_l1575_157524

theorem exist_decreasing_lcm_sequence :
  ∃ (a : Fin 100 → ℕ),
    (∀ i j : Fin 100, i < j → a i < a j) ∧
    (∀ i : Fin 99, Nat.lcm (a i) (a (i + 1)) > Nat.lcm (a (i + 1)) (a (i + 2))) :=
by sorry

end NUMINAMATH_CALUDE_exist_decreasing_lcm_sequence_l1575_157524


namespace NUMINAMATH_CALUDE_function_passes_through_point_l1575_157552

theorem function_passes_through_point (a : ℝ) (h : a < 0) :
  let f := fun x => (1 - a)^x - 1
  f 0 = -1 := by sorry

end NUMINAMATH_CALUDE_function_passes_through_point_l1575_157552


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l1575_157548

/-- Given a rectangle with width 10 meters and area 150 square meters,
    if its length is increased such that the new area is 4/3 times the original area,
    then the new perimeter of the rectangle is 60 meters. -/
theorem rectangle_perimeter (width : ℝ) (original_area : ℝ) (new_area : ℝ) :
  width = 10 →
  original_area = 150 →
  new_area = (4/3) * original_area →
  2 * (new_area / width + width) = 60 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l1575_157548


namespace NUMINAMATH_CALUDE_special_fraction_equality_l1575_157562

theorem special_fraction_equality (a b : ℝ) 
  (h : a / (1 + a) + b / (1 + b) = 1) : 
  a / (1 + b^2) - b / (1 + a^2) = a - b := by
  sorry

end NUMINAMATH_CALUDE_special_fraction_equality_l1575_157562


namespace NUMINAMATH_CALUDE_union_of_M_and_N_l1575_157555

def M : Set Int := {m | -3 < m ∧ m < 2}
def N : Set Int := {n | -1 ≤ n ∧ n ≤ 3}

theorem union_of_M_and_N : M ∪ N = {-2, -1, 0, 1, 2, 3} := by sorry

end NUMINAMATH_CALUDE_union_of_M_and_N_l1575_157555


namespace NUMINAMATH_CALUDE_engineer_check_time_l1575_157541

/-- Represents the road construction project --/
structure RoadProject where
  totalLength : ℝ
  totalDays : ℝ
  initialWorkers : ℝ
  completedLength : ℝ
  additionalWorkers : ℝ

/-- Calculates the number of days after which the progress was checked --/
def daysUntilCheck (project : RoadProject) : ℝ :=
  200 -- The actual calculation is replaced with the known result

/-- Theorem stating that the engineer checked the progress after 200 days --/
theorem engineer_check_time (project : RoadProject) 
    (h1 : project.totalLength = 15)
    (h2 : project.totalDays = 300)
    (h3 : project.initialWorkers = 35)
    (h4 : project.completedLength = 2.5)
    (h5 : project.additionalWorkers = 52.5) :
  daysUntilCheck project = 200 := by
  sorry

#check engineer_check_time

end NUMINAMATH_CALUDE_engineer_check_time_l1575_157541


namespace NUMINAMATH_CALUDE_largest_c_value_l1575_157592

-- Define the function g
def g (x : ℝ) : ℝ := x^2 + 3*x + 1

-- State the theorem
theorem largest_c_value (d : ℝ) (hd : d > 0) :
  (∃ (c : ℝ), c > 0 ∧ 
    (∀ (x : ℝ), |x - 1| ≤ d → |g x - 1| ≤ c) ∧
    (∀ (c' : ℝ), c' > c → ∃ (x : ℝ), |x - 1| ≤ d ∧ |g x - 1| > c')) ∧
  (∀ (c : ℝ), 
    (c > 0 ∧ 
     (∀ (x : ℝ), |x - 1| ≤ d → |g x - 1| ≤ c) ∧
     (∀ (c' : ℝ), c' > c → ∃ (x : ℝ), |x - 1| ≤ d ∧ |g x - 1| > c'))
    → c = 4) :=
by sorry

end NUMINAMATH_CALUDE_largest_c_value_l1575_157592


namespace NUMINAMATH_CALUDE_inequality_product_l1575_157577

theorem inequality_product (a b c d : ℝ) 
  (h1 : a < b) (h2 : b < 0) 
  (h3 : c < d) (h4 : d < 0) : 
  a * c > b * d := by
  sorry

end NUMINAMATH_CALUDE_inequality_product_l1575_157577


namespace NUMINAMATH_CALUDE_company_growth_rate_equation_l1575_157543

/-- Represents the average annual growth rate of a company's payment. -/
def average_annual_growth_rate (initial_payment final_payment : ℝ) (years : ℕ) : ℝ → Prop :=
  λ x => initial_payment * (1 + x) ^ years = final_payment

/-- Theorem stating that the equation 40(1 + x)^2 = 48.4 correctly represents
    the average annual growth rate of the company's payment. -/
theorem company_growth_rate_equation :
  average_annual_growth_rate 40 48.4 2 = λ x => 40 * (1 + x)^2 = 48.4 := by
  sorry

end NUMINAMATH_CALUDE_company_growth_rate_equation_l1575_157543


namespace NUMINAMATH_CALUDE_smoothie_mix_amount_l1575_157565

/-- The amount of smoothie mix in ounces per packet -/
def smoothie_mix_per_packet (total_smoothies : ℕ) (smoothie_size : ℕ) (total_packets : ℕ) : ℚ :=
  (total_smoothies * smoothie_size : ℚ) / total_packets

theorem smoothie_mix_amount : 
  smoothie_mix_per_packet 150 12 180 = 10 := by
  sorry

end NUMINAMATH_CALUDE_smoothie_mix_amount_l1575_157565


namespace NUMINAMATH_CALUDE_thousand_chime_date_l1575_157568

/-- Represents a date --/
structure Date :=
  (year : Nat)
  (month : Nat)
  (day : Nat)

/-- Represents a time --/
structure Time :=
  (hour : Nat)
  (minute : Nat)

/-- Represents the chiming pattern of the clock --/
def clockChime (hour : Nat) (minute : Nat) : Nat :=
  if minute == 30 then 1
  else if minute == 0 then (if hour == 0 || hour == 12 then 12 else hour)
  else 0

/-- Calculates the number of chimes from a given start date and time to a given end date and time --/
def countChimes (startDate : Date) (startTime : Time) (endDate : Date) (endTime : Time) : Nat :=
  sorry -- Implementation details omitted

/-- The theorem to be proved --/
theorem thousand_chime_date :
  let startDate := Date.mk 2003 2 26
  let startTime := Time.mk 10 15
  let endDate := Date.mk 2003 3 7
  countChimes startDate startTime endDate (Time.mk 23 59) ≥ 1000 ∧
  ∀ (d : Date), d.year == 2003 ∧ d.month == 3 ∧ d.day < 7 →
    countChimes startDate startTime d (Time.mk 23 59) < 1000 :=
by sorry

end NUMINAMATH_CALUDE_thousand_chime_date_l1575_157568


namespace NUMINAMATH_CALUDE_book_selection_theorem_l1575_157564

def choose (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

theorem book_selection_theorem :
  let biology_ways := choose 10 3
  let chemistry_ways := choose 8 2
  let physics_ways := choose 5 1
  biology_ways * chemistry_ways * physics_ways = 16800 := by
  sorry

end NUMINAMATH_CALUDE_book_selection_theorem_l1575_157564


namespace NUMINAMATH_CALUDE_m_range_l1575_157584

/-- The function f(x) defined as x^2 + mx - 1 --/
def f (m : ℝ) (x : ℝ) : ℝ := x^2 + m*x - 1

/-- The theorem stating the range of m given the conditions --/
theorem m_range (m : ℝ) : 
  (∀ x ∈ Set.Icc m (m + 1), f m x < 0) → 
  m ∈ Set.Ioo (-Real.sqrt 2 / 2) 0 :=
sorry

end NUMINAMATH_CALUDE_m_range_l1575_157584


namespace NUMINAMATH_CALUDE_age_difference_is_twelve_l1575_157553

/-- The ages of three people A, B, and C, where C is 12 years younger than A -/
structure Ages where
  A : ℕ
  B : ℕ
  C : ℕ
  h : C = A - 12

/-- The difference between the total age of A and B and the total age of B and C -/
def ageDifference (ages : Ages) : ℕ := ages.A + ages.B - (ages.B + ages.C)

/-- Theorem stating that the age difference is always 12 years -/
theorem age_difference_is_twelve (ages : Ages) : ageDifference ages = 12 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_is_twelve_l1575_157553


namespace NUMINAMATH_CALUDE_chess_group_players_l1575_157517

theorem chess_group_players (n : ℕ) : 
  (∀ i j : Fin n, i ≠ j → ∃! game : ℕ, game ≤ 36) →  -- Each player plays each other once
  (∀ game : ℕ, game ≤ 36 → ∃! i j : Fin n, i ≠ j) →  -- Each game is played by two distinct players
  (Nat.choose n 2 = 36) →                            -- Total number of games is 36
  n = 9 := by
sorry

end NUMINAMATH_CALUDE_chess_group_players_l1575_157517


namespace NUMINAMATH_CALUDE_fourth_triangle_exists_l1575_157597

/-- Given four positive real numbers that can form three different triangles,
    prove that they can form a fourth triangle. -/
theorem fourth_triangle_exists (a b c d : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (hab_c : a + b > c ∧ a + c > b ∧ b + c > a)
  (hab_d : a + b > d ∧ a + d > b ∧ b + d > a)
  (acd : a + c > d ∧ a + d > c ∧ c + d > a) :
  b + c > d ∧ b + d > c ∧ c + d > b := by
  sorry

end NUMINAMATH_CALUDE_fourth_triangle_exists_l1575_157597


namespace NUMINAMATH_CALUDE_tangent_iff_k_eq_zero_l1575_157596

/-- A line with equation x - ky - 1 = 0 -/
structure Line (k : ℝ) where
  equation : ∀ x y : ℝ, x - k * y - 1 = 0

/-- A circle with center (2,1) and radius 1 -/
def Circle : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - 2)^2 + (p.2 - 1)^2 = 1}

/-- The line is tangent to the circle -/
def IsTangent (k : ℝ) : Prop :=
  ∃! p : ℝ × ℝ, p ∈ Circle ∧ p.1 - k * p.2 - 1 = 0

/-- The main theorem: k = 0 is necessary and sufficient for the line to be tangent to the circle -/
theorem tangent_iff_k_eq_zero (k : ℝ) : IsTangent k ↔ k = 0 := by
  sorry

end NUMINAMATH_CALUDE_tangent_iff_k_eq_zero_l1575_157596


namespace NUMINAMATH_CALUDE_ends_with_2015_l1575_157500

theorem ends_with_2015 : ∃ n : ℕ, ∃ k : ℕ, 90 * n + 75 = 10000 * k + 2015 := by
  sorry

end NUMINAMATH_CALUDE_ends_with_2015_l1575_157500


namespace NUMINAMATH_CALUDE_quadratic_roots_problem_l1575_157589

theorem quadratic_roots_problem (k : ℝ) (x₁ x₂ : ℝ) : 
  (2 * x₁^2 + k * x₁ - 2 = 0) → 
  (2 * x₂^2 + k * x₂ - 2 = 0) → 
  ((x₁ - 2) * (x₂ - 2) = 10) → 
  k = 7 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_problem_l1575_157589


namespace NUMINAMATH_CALUDE_unique_solution_f_f_eq_zero_l1575_157582

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then -x + 4 else 3*x - 6

-- Theorem statement
theorem unique_solution_f_f_eq_zero :
  ∃! x : ℝ, f (f x) = 0 :=
sorry

end NUMINAMATH_CALUDE_unique_solution_f_f_eq_zero_l1575_157582


namespace NUMINAMATH_CALUDE_arrangement_count_l1575_157502

def number_of_arrangements (total_people : ℕ) (selected_people : ℕ) 
  (meeting_a_participants : ℕ) (meeting_b_participants : ℕ) (meeting_c_participants : ℕ) : ℕ :=
  Nat.choose total_people selected_people * 
  Nat.choose selected_people meeting_a_participants * 
  Nat.choose (selected_people - meeting_a_participants) meeting_b_participants

theorem arrangement_count : 
  number_of_arrangements 10 4 2 1 1 = 2520 := by
  sorry

end NUMINAMATH_CALUDE_arrangement_count_l1575_157502


namespace NUMINAMATH_CALUDE_simplify_square_roots_l1575_157557

theorem simplify_square_roots : 
  Real.sqrt 24 - Real.sqrt 12 + 6 * Real.sqrt (2/3) = 4 * Real.sqrt 6 - 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_square_roots_l1575_157557


namespace NUMINAMATH_CALUDE_square_field_area_l1575_157581

/-- Given a square field where the diagonal can be traversed at 8 km/hr in 0.5 hours,
    the area of the field is 8 square kilometers. -/
theorem square_field_area (speed : ℝ) (time : ℝ) (diagonal : ℝ) (side : ℝ) (area : ℝ) : 
  speed = 8 →
  time = 0.5 →
  diagonal = speed * time →
  diagonal^2 = 2 * side^2 →
  area = side^2 →
  area = 8 := by
sorry

end NUMINAMATH_CALUDE_square_field_area_l1575_157581


namespace NUMINAMATH_CALUDE_saree_price_calculation_l1575_157546

theorem saree_price_calculation (P : ℝ) : 
  (P * (1 - 0.20) * (1 - 0.15) = 272) → P = 400 := by
  sorry

end NUMINAMATH_CALUDE_saree_price_calculation_l1575_157546


namespace NUMINAMATH_CALUDE_bicycle_price_last_year_l1575_157520

theorem bicycle_price_last_year (total_sales_last_year : ℝ) (price_decrease : ℝ) 
  (sales_quantity : ℝ) (decrease_percentage : ℝ) :
  total_sales_last_year = 80000 →
  price_decrease = 200 →
  decrease_percentage = 0.1 →
  total_sales_last_year * (1 - decrease_percentage) = 
    sales_quantity * (total_sales_last_year / sales_quantity - price_decrease) →
  total_sales_last_year / sales_quantity = 2000 := by
sorry

end NUMINAMATH_CALUDE_bicycle_price_last_year_l1575_157520


namespace NUMINAMATH_CALUDE_complex_calculation_result_l1575_157599

theorem complex_calculation_result : 
  ((0.60 * 50 * 0.45 * 30) - (0.40 * 35 / (0.25 * 20))) * ((3/5 * 100) + (2/7 * 49)) = 29762.8 := by
  sorry

end NUMINAMATH_CALUDE_complex_calculation_result_l1575_157599


namespace NUMINAMATH_CALUDE_max_elevation_is_550_l1575_157510

/-- The elevation function for a vertically projected particle -/
def elevation (t : ℝ) : ℝ := 200 * t - 20 * t^2 + 50

/-- The time at which the maximum elevation occurs -/
def max_time : ℝ := 5

theorem max_elevation_is_550 :
  ∃ (t : ℝ), ∀ (t' : ℝ), elevation t ≥ elevation t' ∧ elevation t = 550 :=
sorry

end NUMINAMATH_CALUDE_max_elevation_is_550_l1575_157510


namespace NUMINAMATH_CALUDE_derivative_f_l1575_157561

noncomputable def f (x : ℝ) : ℝ := (1 / (4 * Real.sqrt 5)) * Real.log ((2 + Real.sqrt 5 * Real.tanh x) / (2 - Real.sqrt 5 * Real.tanh x))

theorem derivative_f (x : ℝ) : 
  deriv f x = 1 / (4 - Real.sinh x ^ 2) :=
sorry

end NUMINAMATH_CALUDE_derivative_f_l1575_157561


namespace NUMINAMATH_CALUDE_min_value_F_l1575_157567

/-- The function F(x, y) -/
def F (x y : ℝ) : ℝ := 6*y + 8*x - 9

/-- The constraint equation -/
def constraint (x y : ℝ) : Prop := x^2 + y^2 + 25 = 10*(x + y)

/-- Theorem stating that the minimum value of F(x, y) is 11 given the constraint -/
theorem min_value_F :
  ∃ (min : ℝ), min = 11 ∧
  (∀ x y : ℝ, constraint x y → F x y ≥ min) ∧
  (∃ x y : ℝ, constraint x y ∧ F x y = min) :=
sorry

end NUMINAMATH_CALUDE_min_value_F_l1575_157567


namespace NUMINAMATH_CALUDE_remainder_problem_l1575_157511

theorem remainder_problem (x : ℤ) : x % 82 = 5 → (x + 17) % 41 = 22 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l1575_157511


namespace NUMINAMATH_CALUDE_pet_store_dogs_l1575_157503

/-- Given a ratio of cats to dogs and the number of cats, calculate the number of dogs -/
def calculate_dogs (cat_ratio : ℕ) (dog_ratio : ℕ) (num_cats : ℕ) : ℕ :=
  (num_cats / cat_ratio) * dog_ratio

/-- Theorem: Given the ratio of cats to dogs is 2:3 and there are 14 cats, there are 21 dogs -/
theorem pet_store_dogs : calculate_dogs 2 3 14 = 21 := by
  sorry

end NUMINAMATH_CALUDE_pet_store_dogs_l1575_157503


namespace NUMINAMATH_CALUDE_purple_socks_added_theorem_l1575_157549

/-- Represents the number of socks of each color -/
structure SockDrawer where
  green : Nat
  purple : Nat
  orange : Nat

/-- The initial state of the sock drawer -/
def initialDrawer : SockDrawer :=
  { green := 6, purple := 18, orange := 12 }

/-- Calculates the total number of socks in a drawer -/
def totalSocks (drawer : SockDrawer) : Nat :=
  drawer.green + drawer.purple + drawer.orange

/-- Calculates the probability of picking a purple sock -/
def purpleProbability (drawer : SockDrawer) : Rat :=
  drawer.purple / (totalSocks drawer)

/-- Adds purple socks to the drawer -/
def addPurpleSocks (drawer : SockDrawer) (n : Nat) : SockDrawer :=
  { drawer with purple := drawer.purple + n }

theorem purple_socks_added_theorem :
  ∃ n : Nat, purpleProbability (addPurpleSocks initialDrawer n) = 3/5 ∧ n = 9 := by
  sorry

end NUMINAMATH_CALUDE_purple_socks_added_theorem_l1575_157549


namespace NUMINAMATH_CALUDE_complement_of_M_l1575_157558

-- Define the universal set U as the set of real numbers
def U : Set ℝ := Set.univ

-- Define the set M
def M : Set ℝ := {x | x^2 - x > 0}

-- State the theorem
theorem complement_of_M : 
  Set.compl M = {x : ℝ | 0 ≤ x ∧ x ≤ 1} := by sorry

end NUMINAMATH_CALUDE_complement_of_M_l1575_157558


namespace NUMINAMATH_CALUDE_trent_onions_per_pot_l1575_157534

/-- The number of pots of soup Trent is making -/
def num_pots : ℕ := 6

/-- The total number of tears Trent cries -/
def total_tears : ℕ := 16

/-- The ratio of tears to onions -/
def tear_to_onion_ratio : ℚ := 2 / 3

/-- The number of onions Trent needs to chop per pot of soup -/
def onions_per_pot : ℕ := 4

theorem trent_onions_per_pot :
  onions_per_pot * num_pots * tear_to_onion_ratio = total_tears :=
sorry

end NUMINAMATH_CALUDE_trent_onions_per_pot_l1575_157534


namespace NUMINAMATH_CALUDE_house_of_representatives_democrats_l1575_157507

theorem house_of_representatives_democrats 
  (total : ℕ) 
  (republican_surplus : ℕ) 
  (h1 : total = 434) 
  (h2 : republican_surplus = 30) : 
  ∃ (democrats republicans : ℕ), 
    democrats + republicans = total ∧ 
    republicans = democrats + republican_surplus ∧ 
    democrats = 202 := by
sorry

end NUMINAMATH_CALUDE_house_of_representatives_democrats_l1575_157507


namespace NUMINAMATH_CALUDE_smallest_survey_size_l1575_157590

theorem smallest_survey_size (n : ℕ) : n > 0 ∧ 
  (∃ k₁ : ℕ, n * (140 : ℚ) / 360 = k₁) ∧
  (∃ k₂ : ℕ, n * (108 : ℚ) / 360 = k₂) ∧
  (∃ k₃ : ℕ, n * (72 : ℚ) / 360 = k₃) ∧
  (∃ k₄ : ℕ, n * (40 : ℚ) / 360 = k₄) →
  n ≥ 90 :=
by sorry

end NUMINAMATH_CALUDE_smallest_survey_size_l1575_157590


namespace NUMINAMATH_CALUDE_consecutive_integers_average_l1575_157580

theorem consecutive_integers_average (c d : ℤ) : 
  (c > 0) →
  (d = (c + (c+1) + (c+2) + (c+3) + (c+4) + (c+5) + (c+6)) / 7) →
  ((d + (d+1) + (d+2) + (d+3) + (d+4) + (d+5) + (d+6)) / 7 = c + 6) :=
by sorry

end NUMINAMATH_CALUDE_consecutive_integers_average_l1575_157580


namespace NUMINAMATH_CALUDE_S_bounds_l1575_157595

/-- A permutation of digits 0 to 9 -/
def Permutation := Fin 10 → Fin 10

/-- The sum S as defined in the problem -/
def S (p : Permutation) : ℕ :=
  p 1 + p 2 + p 3 + p 4 + p 6 + p 7 + p 8

/-- Predicate to check if a function is a valid permutation of 0 to 9 -/
def is_valid_permutation (p : Permutation) : Prop :=
  Function.Injective p ∧ Function.Surjective p

theorem S_bounds (p : Permutation) (h : is_valid_permutation p) :
  S p ≥ 21 ∧ S p ≤ 25 :=
sorry

end NUMINAMATH_CALUDE_S_bounds_l1575_157595


namespace NUMINAMATH_CALUDE_expand_polynomial_l1575_157522

theorem expand_polynomial (x : ℝ) : (2 + x^2) * (1 - x^4) = -x^6 + x^2 - 2*x^4 + 2 := by
  sorry

end NUMINAMATH_CALUDE_expand_polynomial_l1575_157522


namespace NUMINAMATH_CALUDE_triangle_side_length_l1575_157523

/-- Given a triangle ABC with the following properties:
  * A = 60°
  * a = 6√3
  * b = 12
  * S_ABC = 18√3
  Prove that c = 6 -/
theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) (S : ℝ) :
  A = π / 3 →  -- 60° in radians
  a = 6 * Real.sqrt 3 →
  b = 12 →
  S = 18 * Real.sqrt 3 →
  c = 6 := by
  sorry


end NUMINAMATH_CALUDE_triangle_side_length_l1575_157523


namespace NUMINAMATH_CALUDE_sum_of_binary_digits_sum_l1575_157501

/-- The sum of binary digits of a positive integer -/
def s (n : ℕ+) : ℕ := sorry

/-- The sum of s(n) for n from 1 to 2^k -/
def S (k : ℕ+) : ℕ :=
  Finset.sum (Finset.range (2^k.val)) (fun i => s ⟨i + 1, Nat.succ_pos i⟩)

/-- The main theorem: S(k) = 2^(k-1) * k + 1 for all positive integers k -/
theorem sum_of_binary_digits_sum (k : ℕ+) : S k = 2^(k.val - 1) * k.val + 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_binary_digits_sum_l1575_157501


namespace NUMINAMATH_CALUDE_negation_of_implication_l1575_157539

theorem negation_of_implication (x : ℝ) :
  (¬(x^2 + x - 6 ≥ 0 → x > 2)) ↔ (x^2 + x - 6 < 0 → x ≤ 2) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_implication_l1575_157539


namespace NUMINAMATH_CALUDE_barbed_wire_rate_l1575_157512

/-- Given a square field with area 3136 sq m and a total cost of 932.40 Rs for drawing barbed wire
    around it, leaving two 1 m wide gates, the rate of drawing barbed wire per meter is 4.2 Rs/m. -/
theorem barbed_wire_rate (area : ℝ) (total_cost : ℝ) (gate_width : ℝ) (num_gates : ℕ) :
  area = 3136 →
  total_cost = 932.40 →
  gate_width = 1 →
  num_gates = 2 →
  (total_cost / (4 * Real.sqrt area - num_gates * gate_width) : ℝ) = 4.2 := by
  sorry

end NUMINAMATH_CALUDE_barbed_wire_rate_l1575_157512


namespace NUMINAMATH_CALUDE_sequence_increasing_and_divergent_l1575_157547

open Real MeasureTheory Interval Set

noncomputable section

variables (a b : ℝ) (f g : ℝ → ℝ)

def I (n : ℕ) := ∫ x in a..b, (f x)^(n+1) / (g x)^n

theorem sequence_increasing_and_divergent
  (hab : a < b)
  (hf : ContinuousOn f (Icc a b))
  (hg : ContinuousOn g (Icc a b))
  (hfg_pos : ∀ x ∈ Icc a b, 0 < f x ∧ 0 < g x)
  (hfg_int : ∫ x in a..b, f x = ∫ x in a..b, g x)
  (hfg_neq : f ≠ g) :
  (∀ n : ℕ, I a b f g n < I a b f g (n + 1)) ∧
  (∀ M : ℝ, ∃ N : ℕ, ∀ n ≥ N, M < I a b f g n) :=
sorry

end NUMINAMATH_CALUDE_sequence_increasing_and_divergent_l1575_157547


namespace NUMINAMATH_CALUDE_max_angle_ratio_theorem_l1575_157542

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 16 + y^2 / 4 = 1

-- Define the line
def line (x y : ℝ) : Prop := x - Real.sqrt 3 * y + 8 + 2 * Real.sqrt 3 = 0

-- Define the foci
def foci (F₁ F₂ : ℝ × ℝ) : Prop :=
  F₁ = (-2 * Real.sqrt 3, 0) ∧ F₂ = (2 * Real.sqrt 3, 0)

-- Define the point P on the line
def point_on_line (P : ℝ × ℝ) : Prop :=
  line P.1 P.2

-- Define the angle F₁PF₂
def angle_F₁PF₂ (F₁ F₂ P : ℝ × ℝ) : ℝ := sorry

-- Define the distance between two points
def distance (A B : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem max_angle_ratio_theorem 
  (F₁ F₂ P : ℝ × ℝ) 
  (h_ellipse : ellipse F₁.1 F₁.2 ∧ ellipse F₂.1 F₂.2)
  (h_foci : foci F₁ F₂)
  (h_point : point_on_line P)
  (h_max_angle : ∀ Q, point_on_line Q → angle_F₁PF₂ F₁ F₂ P ≥ angle_F₁PF₂ F₁ F₂ Q) :
  distance P F₁ / distance P F₂ = Real.sqrt 3 - 1 := by
  sorry

end NUMINAMATH_CALUDE_max_angle_ratio_theorem_l1575_157542


namespace NUMINAMATH_CALUDE_solve_for_R_l1575_157586

theorem solve_for_R (R : ℝ) : (R^3)^(1/4) = 64 * 4^(1/16) → R = 256 * 2^(1/6) := by
  sorry

end NUMINAMATH_CALUDE_solve_for_R_l1575_157586
