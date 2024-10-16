import Mathlib

namespace NUMINAMATH_CALUDE_min_value_expression_min_value_achievable_l5_562

theorem min_value_expression (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + 2*y = 1) :
  x^2 + 4*y^2 + 2*x*y ≥ 3/4 :=
sorry

theorem min_value_achievable :
  ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x + 2*y = 1 ∧ x^2 + 4*y^2 + 2*x*y = 3/4 :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_min_value_achievable_l5_562


namespace NUMINAMATH_CALUDE_display_rows_l5_521

/-- The number of cans in the nth row of the display -/
def cans_in_row (n : ℕ) : ℕ := 2 + 3 * (n - 1)

/-- The total number of cans in the first n rows of the display -/
def total_cans (n : ℕ) : ℕ := n * (cans_in_row 1 + cans_in_row n) / 2

/-- The number of rows in the display -/
def num_rows : ℕ := 12

theorem display_rows :
  total_cans num_rows = 225 ∧
  cans_in_row 1 = 2 ∧
  ∀ n : ℕ, n > 1 → cans_in_row n = cans_in_row (n - 1) + 3 :=
sorry

end NUMINAMATH_CALUDE_display_rows_l5_521


namespace NUMINAMATH_CALUDE_alcohol_mixture_percentage_l5_518

theorem alcohol_mixture_percentage 
  (initial_volume : ℝ) 
  (initial_percentage : ℝ) 
  (added_alcohol : ℝ) 
  (h1 : initial_volume = 6) 
  (h2 : initial_percentage = 25) 
  (h3 : added_alcohol = 3) : 
  let initial_alcohol := initial_volume * (initial_percentage / 100)
  let final_alcohol := initial_alcohol + added_alcohol
  let final_volume := initial_volume + added_alcohol
  final_alcohol / final_volume * 100 = 50 := by
sorry


end NUMINAMATH_CALUDE_alcohol_mixture_percentage_l5_518


namespace NUMINAMATH_CALUDE_nth_S_645_l5_595

/-- The set of positive integers with remainder 5 when divided by 8 -/
def S : Set ℕ := {n : ℕ | n > 0 ∧ n % 8 = 5}

/-- The nth element of S -/
def nth_S (n : ℕ) : ℕ := 8 * (n - 1) + 5

theorem nth_S_645 : nth_S 81 = 645 := by
  sorry

end NUMINAMATH_CALUDE_nth_S_645_l5_595


namespace NUMINAMATH_CALUDE_percentage_fraction_proof_l5_585

theorem percentage_fraction_proof (P : ℚ) : 
  P < 35 → (P / 100) * 180 = 42 → P / 100 = 7 / 30 := by
  sorry

end NUMINAMATH_CALUDE_percentage_fraction_proof_l5_585


namespace NUMINAMATH_CALUDE_general_ticket_price_is_six_l5_551

/-- Represents the ticket sales and pricing scenario -/
structure TicketSale where
  student_price : ℕ
  total_tickets : ℕ
  total_revenue : ℕ
  general_tickets : ℕ

/-- Calculates the price of a general admission ticket -/
def general_price (sale : TicketSale) : ℚ :=
  (sale.total_revenue - sale.student_price * (sale.total_tickets - sale.general_tickets)) / sale.general_tickets

/-- Theorem stating that the general admission ticket price is 6 dollars -/
theorem general_ticket_price_is_six (sale : TicketSale) 
  (h1 : sale.student_price = 4)
  (h2 : sale.total_tickets = 525)
  (h3 : sale.total_revenue = 2876)
  (h4 : sale.general_tickets = 388) :
  general_price sale = 6 := by
  sorry

#eval general_price {
  student_price := 4,
  total_tickets := 525,
  total_revenue := 2876,
  general_tickets := 388
}

end NUMINAMATH_CALUDE_general_ticket_price_is_six_l5_551


namespace NUMINAMATH_CALUDE_subtraction_problem_l5_589

theorem subtraction_problem (x : ℤ) : x - 29 = 63 → x - 47 = 45 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_problem_l5_589


namespace NUMINAMATH_CALUDE_divisible_by_seven_l5_503

/-- The number of repeated digits -/
def n : ℕ := 50

/-- The number formed by n eights followed by x followed by n nines -/
def f (x : ℕ) : ℕ :=
  8 * (10^(2*n + 1) - 1) / 9 + x * 10^n + 9 * (10^n - 1) / 9

/-- The main theorem -/
theorem divisible_by_seven (x : ℕ) : 7 ∣ f x ↔ x = 0 := by sorry

end NUMINAMATH_CALUDE_divisible_by_seven_l5_503


namespace NUMINAMATH_CALUDE_quadratic_root_difference_l5_500

theorem quadratic_root_difference (a b c d e : ℝ) :
  (2 * a^2 - 5 * a + 2 = 3 * a + 24) →
  ∃ x y : ℝ, (x ≠ y) ∧ 
             (2 * x^2 - 5 * x + 2 = 3 * x + 24) ∧ 
             (2 * y^2 - 5 * y + 2 = 3 * y + 24) ∧ 
             (abs (x - y) = 2 * Real.sqrt 15) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_difference_l5_500


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l5_564

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) (d : ℝ) :
  arithmetic_sequence a d → a 3 = 4 → d = -2 → a 2 + a 6 = 4 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l5_564


namespace NUMINAMATH_CALUDE_problem_solution_l5_538

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := a * x - a - Real.log x

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x * g a x

theorem problem_solution :
  (∃ a : ℝ, ∀ x > 0, g a x ≥ 0) ∧
  (∃ a : ℝ, ∃ x₀ : ℝ, 0 < x₀ ∧ x₀ < 1 ∧
    (∀ x > 0, (deriv (f a)) x₀ = 0 ∧ f a x ≤ f a x₀)) :=
by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l5_538


namespace NUMINAMATH_CALUDE_framed_photo_ratio_l5_513

/-- Represents the dimensions of a framed photograph -/
structure FramedPhoto where
  original_width : ℝ
  original_height : ℝ
  frame_width : ℝ

/-- Calculates the area of the original photograph -/
def original_area (photo : FramedPhoto) : ℝ :=
  photo.original_width * photo.original_height

/-- Calculates the area of the framed photograph -/
def framed_area (photo : FramedPhoto) : ℝ :=
  (photo.original_width + 2 * photo.frame_width) * (photo.original_height + 6 * photo.frame_width)

/-- Theorem: The ratio of the shorter to the longer dimension of the framed photograph is 1:2 -/
theorem framed_photo_ratio (photo : FramedPhoto) 
  (h1 : photo.original_width = 20)
  (h2 : photo.original_height = 30)
  (h3 : framed_area photo = 2 * original_area photo) :
  (photo.original_width + 2 * photo.frame_width) / (photo.original_height + 6 * photo.frame_width) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_framed_photo_ratio_l5_513


namespace NUMINAMATH_CALUDE_stamp_cost_problem_l5_530

theorem stamp_cost_problem (total_stamps : ℕ) (high_denom : ℕ) (total_cost : ℚ) (high_denom_count : ℕ) :
  total_stamps = 20 →
  high_denom = 37 →
  total_cost = 706/100 →
  high_denom_count = 18 →
  ∃ (low_denom : ℕ),
    low_denom * (total_stamps - high_denom_count) = (total_cost * 100 - high_denom * high_denom_count : ℚ) ∧
    low_denom = 20 :=
by sorry

end NUMINAMATH_CALUDE_stamp_cost_problem_l5_530


namespace NUMINAMATH_CALUDE_fans_per_bleacher_set_l5_592

theorem fans_per_bleacher_set (total_fans : ℕ) (num_bleacher_sets : ℕ) 
  (h1 : total_fans = 2436) (h2 : num_bleacher_sets = 3) :
  total_fans / num_bleacher_sets = 812 := by
  sorry

end NUMINAMATH_CALUDE_fans_per_bleacher_set_l5_592


namespace NUMINAMATH_CALUDE_smallest_a_value_l5_596

theorem smallest_a_value (a b : ℝ) (h1 : a ≥ 0) (h2 : b ≥ 0)
  (h3 : ∀ x : ℤ, Real.sin (a * ↑x + b) = Real.sin (15 * ↑x)) :
  ∀ a' ≥ 0, (∀ x : ℤ, Real.sin (a' * ↑x + b) = Real.sin (15 * ↑x)) → a' ≥ 15 :=
by sorry

end NUMINAMATH_CALUDE_smallest_a_value_l5_596


namespace NUMINAMATH_CALUDE_fifty_square_tileable_by_one_by_four_l5_576

/-- A square can be perfectly tiled by rectangles if its area is divisible by the area of the rectangle -/
def is_perfectly_tileable (square_side : ℕ) (rect_width rect_height : ℕ) : Prop :=
  (square_side * square_side) % (rect_width * rect_height) = 0

/-- The theorem states that a 50 × 50 square can be perfectly tiled with 1 × 4 rectangles -/
theorem fifty_square_tileable_by_one_by_four : is_perfectly_tileable 50 1 4 := by
  sorry

end NUMINAMATH_CALUDE_fifty_square_tileable_by_one_by_four_l5_576


namespace NUMINAMATH_CALUDE_sum_of_xyz_l5_536

theorem sum_of_xyz (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h1 : x * y = 30) (h2 : x * z = 60) (h3 : y * z = 90) :
  x + y + z = 11 * Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_sum_of_xyz_l5_536


namespace NUMINAMATH_CALUDE_shop_profit_days_l5_509

theorem shop_profit_days (mean_profit : ℝ) (first_15_mean : ℝ) (last_15_mean : ℝ)
  (h1 : mean_profit = 350)
  (h2 : first_15_mean = 255)
  (h3 : last_15_mean = 445) :
  ∃ (total_days : ℕ), 
    total_days = 30 ∧ 
    (first_15_mean * 15 + last_15_mean * 15 : ℝ) = mean_profit * total_days :=
by
  sorry

end NUMINAMATH_CALUDE_shop_profit_days_l5_509


namespace NUMINAMATH_CALUDE_pirate_treasure_probability_l5_540

theorem pirate_treasure_probability :
  let n : ℕ := 5  -- number of islands
  let p_treasure : ℚ := 1/3  -- probability of treasure on an island
  let p_trap : ℚ := 1/6  -- probability of trap on an island
  let p_neither : ℚ := 1/2  -- probability of neither treasure nor trap on an island
  
  -- Probability of exactly 4 islands with treasure and 1 with neither
  (Nat.choose n 4 : ℚ) * p_treasure^4 * p_neither = 5/162 :=
by sorry

end NUMINAMATH_CALUDE_pirate_treasure_probability_l5_540


namespace NUMINAMATH_CALUDE_unit_vector_of_difference_l5_575

/-- Given vectors a and b in ℝ², prove that the unit vector of a - b is (-4/5, 3/5) -/
theorem unit_vector_of_difference (a b : ℝ × ℝ) (ha : a = (3, 1)) (hb : b = (7, -2)) :
  let diff := a - b
  let norm := Real.sqrt ((diff.1)^2 + (diff.2)^2)
  (diff.1 / norm, diff.2 / norm) = (-4/5, 3/5) := by
sorry

end NUMINAMATH_CALUDE_unit_vector_of_difference_l5_575


namespace NUMINAMATH_CALUDE_expression_simplification_l5_563

theorem expression_simplification :
  500 * 997 * 0.4995 * 100 = 997^2 * 25 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l5_563


namespace NUMINAMATH_CALUDE_prob_different_plants_l5_535

/-- The number of distinct plant options available -/
def num_options : ℕ := 4

/-- The probability of two employees choosing different plants -/
def prob_different_choices : ℚ := 3/4

/-- Theorem stating that the probability of two employees choosing different plants
    from four options is 3/4 -/
theorem prob_different_plants :
  (num_options : ℚ)^2 - num_options = (prob_different_choices * num_options^2 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_prob_different_plants_l5_535


namespace NUMINAMATH_CALUDE_common_root_of_quadratics_l5_556

theorem common_root_of_quadratics (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  ∃ x : ℝ, a * x^2 + b * x + c = 0 ∧ b * x^2 + c * x + a = 0 → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_common_root_of_quadratics_l5_556


namespace NUMINAMATH_CALUDE_milk_production_l5_555

/-- Milk production calculation -/
theorem milk_production
  (m n p x q r : ℝ)
  (h1 : m > 0)
  (h2 : p > 0)
  (h3 : 0 ≤ x)
  (h4 : x ≤ m)
  : (q * r * (m + 0.2 * x) * n) / (m * p) =
    q * r * ((m - x) * (n / (m * p)) + x * (1.2 * n / (m * p))) :=
by sorry

end NUMINAMATH_CALUDE_milk_production_l5_555


namespace NUMINAMATH_CALUDE_subscription_total_l5_557

-- Define the subscription amounts for a, b, and c
def subscription (a b c : ℕ) : Prop :=
  a = b + 4000 ∧ b = c + 5000

-- Define the total profit and b's share
def profit_share (total_profit b_profit : ℕ) : Prop :=
  total_profit = 30000 ∧ b_profit = 10200

-- Define the total subscription
def total_subscription (a b c : ℕ) : ℕ :=
  a + b + c

-- Theorem statement
theorem subscription_total 
  (a b c : ℕ) 
  (h1 : subscription a b c) 
  (h2 : profit_share 30000 10200) :
  total_subscription a b c = 14036 :=
sorry

end NUMINAMATH_CALUDE_subscription_total_l5_557


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l5_590

theorem min_value_reciprocal_sum (a b c : ℝ) 
  (ha : 0 < a ∧ a < 1) (hb : 0 < b ∧ b < 1) (hc : 0 < c ∧ c < 1)
  (h_sum : a * b + b * c + a * c = 1) :
  (1 / (1 - a) + 1 / (1 - b) + 1 / (1 - c)) ≥ (9 + 3 * Real.sqrt 3) / 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l5_590


namespace NUMINAMATH_CALUDE_quarter_angle_tangent_line_through_point_line_with_y_intercept_l5_560

-- Define the original line
def original_line (x y : ℝ) : Prop := y = -Real.sqrt 3 * x + 1

-- Define the angle that is one fourth of the slope angle
def quarter_angle : ℝ := 30

-- Theorem 1: The tangent of the quarter angle is √3/3
theorem quarter_angle_tangent :
  Real.tan (quarter_angle * π / 180) = Real.sqrt 3 / 3 := by sorry

-- Theorem 2: Equation of the line passing through (√3, -1)
theorem line_through_point (x y : ℝ) :
  (Real.sqrt 3 * x - 3 * y - 6 = 0) ↔
  (y + 1 = (Real.sqrt 3 / 3) * (x - Real.sqrt 3)) := by sorry

-- Theorem 3: Equation of the line with y-intercept -5
theorem line_with_y_intercept (x y : ℝ) :
  (Real.sqrt 3 * x - 3 * y - 15 = 0) ↔
  (y = (Real.sqrt 3 / 3) * x - 5) := by sorry

end NUMINAMATH_CALUDE_quarter_angle_tangent_line_through_point_line_with_y_intercept_l5_560


namespace NUMINAMATH_CALUDE_smores_per_person_l5_577

theorem smores_per_person (people : ℕ) (cost_per_set : ℚ) (smores_per_set : ℕ) (total_cost : ℚ) :
  people = 8 →
  cost_per_set = 3 →
  smores_per_set = 4 →
  total_cost = 18 →
  (total_cost / cost_per_set * smores_per_set) / people = 3 := by
  sorry

#check smores_per_person

end NUMINAMATH_CALUDE_smores_per_person_l5_577


namespace NUMINAMATH_CALUDE_calculation_proof_l5_541

theorem calculation_proof : (0.0048 * 3.5) / (0.05 * 0.1 * 0.004) = 840 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l5_541


namespace NUMINAMATH_CALUDE_computer_table_cost_price_l5_554

/-- The cost price of a computer table, given the selling price and markup percentage. -/
def cost_price (selling_price : ℚ) (markup_percentage : ℚ) : ℚ :=
  selling_price / (1 + markup_percentage)

/-- Theorem: The cost price of the computer table is 6500, given the conditions. -/
theorem computer_table_cost_price :
  let selling_price : ℚ := 8450
  let markup_percentage : ℚ := 0.30
  cost_price selling_price markup_percentage = 6500 := by
sorry

end NUMINAMATH_CALUDE_computer_table_cost_price_l5_554


namespace NUMINAMATH_CALUDE_distribute_five_into_three_l5_514

/-- The number of ways to distribute n distinct objects into k indistinguishable containers --/
def distribute (n k : ℕ) : ℕ :=
  sorry

/-- Theorem stating that distributing 5 distinct objects into 3 indistinguishable containers
    results in 51 different arrangements --/
theorem distribute_five_into_three :
  distribute 5 3 = 51 := by
  sorry

end NUMINAMATH_CALUDE_distribute_five_into_three_l5_514


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l5_579

-- Define set A
def A : Set ℝ := {x | |x| < 1}

-- Define set B
def B : Set ℝ := {y | ∃ x : ℝ, y = x^2}

-- State the theorem
theorem intersection_A_complement_B : A ∩ (Set.univ \ B) = Set.Ioo (-1 : ℝ) 0 := by sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l5_579


namespace NUMINAMATH_CALUDE_car_efficiency_before_modification_l5_539

/-- Represents the fuel efficiency of a car before and after modification -/
structure CarEfficiency where
  pre_mod : ℝ  -- Fuel efficiency before modification (miles per gallon)
  post_mod : ℝ  -- Fuel efficiency after modification (miles per gallon)
  fuel_capacity : ℝ  -- Fuel tank capacity in gallons
  extra_distance : ℝ  -- Additional distance traveled after modification (miles)

/-- Theorem stating the car's fuel efficiency before modification -/
theorem car_efficiency_before_modification (car : CarEfficiency)
  (h1 : car.post_mod = car.pre_mod / 0.8)
  (h2 : car.fuel_capacity = 15)
  (h3 : car.fuel_capacity * car.post_mod = car.fuel_capacity * car.pre_mod + car.extra_distance)
  (h4 : car.extra_distance = 105) :
  car.pre_mod = 28 := by
  sorry

end NUMINAMATH_CALUDE_car_efficiency_before_modification_l5_539


namespace NUMINAMATH_CALUDE_average_class_size_is_35_l5_507

/-- Represents the number of children in each age group --/
structure AgeGroups where
  three_year_olds : ℕ
  four_year_olds : ℕ
  five_year_olds : ℕ
  six_year_olds : ℕ

/-- Represents the Sunday school setup --/
def SundaySchool (ages : AgeGroups) : Prop :=
  ages.three_year_olds = 13 ∧
  ages.four_year_olds = 20 ∧
  ages.five_year_olds = 15 ∧
  ages.six_year_olds = 22

/-- Calculates the average class size --/
def averageClassSize (ages : AgeGroups) : ℚ :=
  let class1 := ages.three_year_olds + ages.four_year_olds
  let class2 := ages.five_year_olds + ages.six_year_olds
  (class1 + class2) / 2

/-- Theorem stating that the average class size is 35 --/
theorem average_class_size_is_35 (ages : AgeGroups) 
  (h : SundaySchool ages) : averageClassSize ages = 35 := by
  sorry

end NUMINAMATH_CALUDE_average_class_size_is_35_l5_507


namespace NUMINAMATH_CALUDE_binomial_square_constant_l5_537

/-- If x^2 + 80x + c is equal to the square of a binomial, then c = 1600 -/
theorem binomial_square_constant (c : ℝ) : 
  (∃ a : ℝ, ∀ x : ℝ, x^2 + 80*x + c = (x + a)^2) → c = 1600 := by
  sorry

end NUMINAMATH_CALUDE_binomial_square_constant_l5_537


namespace NUMINAMATH_CALUDE_masha_number_l5_568

theorem masha_number (x y : ℕ) : 
  (x + y = 2002 ∨ x * y = 2002) →
  (∀ a : ℕ, (a + y = 2002 ∨ a * y = 2002) → ∃ b ≠ y, (a + b = 2002 ∨ a * b = 2002)) →
  (∀ a : ℕ, (x + a = 2002 ∨ x * a = 2002) → ∃ b ≠ x, (b + a = 2002 ∨ b * a = 2002)) →
  max x y = 1001 :=
by sorry

end NUMINAMATH_CALUDE_masha_number_l5_568


namespace NUMINAMATH_CALUDE_factorization_of_difference_of_squares_l5_532

theorem factorization_of_difference_of_squares (a b : ℝ) :
  3 * a^2 - 3 * b^2 = 3 * (a + b) * (a - b) := by sorry

end NUMINAMATH_CALUDE_factorization_of_difference_of_squares_l5_532


namespace NUMINAMATH_CALUDE_binomial_10_5_l5_510

theorem binomial_10_5 : Nat.choose 10 5 = 252 := by
  sorry

end NUMINAMATH_CALUDE_binomial_10_5_l5_510


namespace NUMINAMATH_CALUDE_sum_of_squares_of_roots_l5_552

theorem sum_of_squares_of_roots (p q r : ℝ) : 
  (3 * p^3 - 2 * p^2 + 5 * p + 15 = 0) →
  (3 * q^3 - 2 * q^2 + 5 * q + 15 = 0) →
  (3 * r^3 - 2 * r^2 + 5 * r + 15 = 0) →
  p^2 + q^2 + r^2 = -26/9 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_roots_l5_552


namespace NUMINAMATH_CALUDE_running_competition_sample_l5_516

/-- Given a school with 2000 students, where 3/5 participate in a running competition
    with grade ratios of 2:3:5, and a sample of 200 students is taken, 
    the number of 2nd grade students in the running competition sample is 36. -/
theorem running_competition_sample (total_students : ℕ) (sample_size : ℕ) 
  (running_ratio : ℚ) (grade_ratios : Fin 3 → ℚ) :
  total_students = 2000 →
  sample_size = 200 →
  running_ratio = 3/5 →
  grade_ratios 0 = 2/10 ∧ grade_ratios 1 = 3/10 ∧ grade_ratios 2 = 5/10 →
  ↑sample_size * running_ratio * grade_ratios 1 = 36 := by
  sorry

#check running_competition_sample

end NUMINAMATH_CALUDE_running_competition_sample_l5_516


namespace NUMINAMATH_CALUDE_student_seat_occupancy_l5_599

/-- Proves that the fraction of occupied student seats is 4/5 --/
theorem student_seat_occupancy
  (total_chairs : ℕ)
  (rows : ℕ)
  (chairs_per_row : ℕ)
  (awardee_rows : ℕ)
  (admin_teacher_rows : ℕ)
  (parent_rows : ℕ)
  (vacant_student_seats : ℕ)
  (h1 : total_chairs = rows * chairs_per_row)
  (h2 : rows = 10)
  (h3 : chairs_per_row = 15)
  (h4 : awardee_rows = 1)
  (h5 : admin_teacher_rows = 2)
  (h6 : parent_rows = 2)
  (h7 : vacant_student_seats = 15) :
  let student_rows := rows - (awardee_rows + admin_teacher_rows + parent_rows)
  let student_chairs := student_rows * chairs_per_row
  let occupied_student_chairs := student_chairs - vacant_student_seats
  (occupied_student_chairs : ℚ) / student_chairs = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_student_seat_occupancy_l5_599


namespace NUMINAMATH_CALUDE_intersection_A_B_range_of_a_l5_546

-- Define the sets A, B, and C
def A : Set ℝ := {x | (x + 3) / (x - 1) ≥ 0}
def B : Set ℝ := {x | x^2 - x - 2 ≤ 0}
def C (a : ℝ) : Set ℝ := {x | x ≥ a^2 - 2}

-- Theorem for the intersection of A and B
theorem intersection_A_B : A ∩ B = {x : ℝ | 1 < x ∧ x ≤ 2} := by sorry

-- Theorem for the range of a when B ∪ C = C
theorem range_of_a (a : ℝ) (h : B ∪ C a = C a) : a ∈ Set.Icc (-1) 1 := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_range_of_a_l5_546


namespace NUMINAMATH_CALUDE_faster_train_speed_l5_566

/-- Proves the speed of the faster train given the conditions of the problem -/
theorem faster_train_speed
  (speed_diff : ℝ)
  (faster_train_length : ℝ)
  (crossing_time : ℝ)
  (h1 : speed_diff = 36)
  (h2 : faster_train_length = 120)
  (h3 : crossing_time = 12)
  : ∃ (faster_speed : ℝ), faster_speed = 72 :=
by
  sorry

end NUMINAMATH_CALUDE_faster_train_speed_l5_566


namespace NUMINAMATH_CALUDE_odd_periodic_function_zeros_l5_524

/-- A function f is odd if f(-x) = -f(x) for all x -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- A function f is periodic with period T if f(x + T) = f(x) for all x -/
def IsPeriodic (f : ℝ → ℝ) (T : ℝ) : Prop :=
  ∀ x, f (x + T) = f x

/-- The number of zeros of a function f in an interval [a, b] -/
def NumberOfZeros (f : ℝ → ℝ) (a b : ℝ) : ℕ :=
  sorry

theorem odd_periodic_function_zeros (f : ℝ → ℝ) (T : ℝ) :
  IsOdd f → IsPeriodic f T → T > 0 → NumberOfZeros f (-T) T = 5 :=
sorry

end NUMINAMATH_CALUDE_odd_periodic_function_zeros_l5_524


namespace NUMINAMATH_CALUDE_seven_at_eight_equals_nineteen_thirds_l5_517

-- Define the @ operation
def at_op (a b : ℚ) : ℚ := (5 * a - 2 * b) / 3

-- Theorem statement
theorem seven_at_eight_equals_nineteen_thirds :
  at_op 7 8 = 19 / 3 := by sorry

end NUMINAMATH_CALUDE_seven_at_eight_equals_nineteen_thirds_l5_517


namespace NUMINAMATH_CALUDE_marked_price_calculation_l5_598

theorem marked_price_calculation (total_price : ℝ) (discount_percentage : ℝ) : 
  total_price = 50 →
  discount_percentage = 60 →
  ∃ (marked_price : ℝ), 
    marked_price = 62.50 ∧ 
    2 * marked_price * (1 - discount_percentage / 100) = total_price :=
by sorry

end NUMINAMATH_CALUDE_marked_price_calculation_l5_598


namespace NUMINAMATH_CALUDE_total_red_cards_l5_572

/-- The number of decks the shopkeeper has -/
def num_decks : ℕ := 7

/-- The number of red cards in one deck -/
def red_cards_per_deck : ℕ := 26

/-- Theorem: The total number of red cards the shopkeeper has is 182 -/
theorem total_red_cards : num_decks * red_cards_per_deck = 182 := by
  sorry

end NUMINAMATH_CALUDE_total_red_cards_l5_572


namespace NUMINAMATH_CALUDE_gcd_of_lcm_and_ratio_l5_567

theorem gcd_of_lcm_and_ratio (A B : ℕ+) : 
  Nat.lcm A B = 180 → 
  A.val * 6 = B.val * 5 → 
  Nat.gcd A B = 6 := by
sorry

end NUMINAMATH_CALUDE_gcd_of_lcm_and_ratio_l5_567


namespace NUMINAMATH_CALUDE_original_count_pingpong_shuttlecock_l5_549

theorem original_count_pingpong_shuttlecock : ∀ (n : ℕ),
  (∃ (x : ℕ), n = 5 * x ∧ n = 3 * x + 16) →
  n = 40 := by
  sorry

end NUMINAMATH_CALUDE_original_count_pingpong_shuttlecock_l5_549


namespace NUMINAMATH_CALUDE_solve_transactions_problem_l5_542

def transactions_problem (mabel_monday : ℕ) : Prop :=
  let mabel_tuesday : ℕ := mabel_monday + mabel_monday / 10
  let anthony_tuesday : ℕ := 2 * mabel_tuesday
  let cal_tuesday : ℕ := (2 * anthony_tuesday + 2) / 3  -- Rounded up
  let jade_tuesday : ℕ := cal_tuesday + 17
  let isla_wednesday : ℕ := mabel_tuesday + cal_tuesday - 12
  let tim_thursday : ℕ := jade_tuesday + isla_wednesday + (jade_tuesday + isla_wednesday) / 2 + 1  -- Rounded up
  (mabel_monday = 100) → (tim_thursday = 614)

theorem solve_transactions_problem :
  transactions_problem 100 := by sorry

end NUMINAMATH_CALUDE_solve_transactions_problem_l5_542


namespace NUMINAMATH_CALUDE_range_of_a_l5_544

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, |x - 3| + |x - a| > 5) ↔ (a > 8 ∨ a < -2) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l5_544


namespace NUMINAMATH_CALUDE_inequality_system_solutions_l5_588

def inequality_system (x t : ℝ) : Prop :=
  6 - (2 * x + 5) > -15 ∧ (x + 3) / 2 - t < x

theorem inequality_system_solutions :
  (∀ x : ℤ, inequality_system x 2 → x ≥ 0) ∧
  (∃ x : ℤ, inequality_system x 2 ∧ x = 0) ∧
  (∀ x : ℝ, inequality_system x 4 ↔ -5 < x ∧ x < 8) ∧
  (∃! t : ℝ, ∀ x : ℝ, inequality_system x t ↔ -5 < x ∧ x < 8) ∧
  (∀ t : ℝ, (∃! (a b c : ℤ), 
    inequality_system (a : ℝ) t ∧ 
    inequality_system (b : ℝ) t ∧ 
    inequality_system (c : ℝ) t ∧ 
    a < b ∧ b < c ∧
    (∀ x : ℤ, inequality_system (x : ℝ) t → x = a ∨ x = b ∨ x = c)) 
    ↔ -1 < t ∧ t ≤ -1/2) :=
by sorry

end NUMINAMATH_CALUDE_inequality_system_solutions_l5_588


namespace NUMINAMATH_CALUDE_total_passengers_per_hour_l5_543

/-- Calculates the total number of different passengers stepping on and off trains at a station within an hour -/
theorem total_passengers_per_hour 
  (train_interval : ℕ) 
  (passengers_leaving : ℕ) 
  (passengers_boarding : ℕ) 
  (hour_in_minutes : ℕ) :
  train_interval = 5 →
  passengers_leaving = 200 →
  passengers_boarding = 320 →
  hour_in_minutes = 60 →
  (hour_in_minutes / train_interval) * (passengers_leaving + passengers_boarding) = 6240 := by
  sorry

end NUMINAMATH_CALUDE_total_passengers_per_hour_l5_543


namespace NUMINAMATH_CALUDE_spider_web_paths_l5_508

/-- The number of paths from (0,0) to (m,n) on a grid, moving only up and right -/
def gridPaths (m n : ℕ) : ℕ := Nat.choose (m + n) m

/-- The coordinates of the target point -/
def target : (ℕ × ℕ) := (4, 3)

theorem spider_web_paths : 
  gridPaths target.1 target.2 = 35 := by sorry

end NUMINAMATH_CALUDE_spider_web_paths_l5_508


namespace NUMINAMATH_CALUDE_egg_cost_calculation_l5_597

def dozen : ℕ := 12

theorem egg_cost_calculation (total_cost : ℚ) (num_dozens : ℕ) 
  (h1 : total_cost = 18) 
  (h2 : num_dozens = 3) : 
  total_cost / (num_dozens * dozen) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_egg_cost_calculation_l5_597


namespace NUMINAMATH_CALUDE_imaginary_part_of_1_plus_2i_l5_558

theorem imaginary_part_of_1_plus_2i : Complex.im (1 + 2*Complex.I) = 2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_1_plus_2i_l5_558


namespace NUMINAMATH_CALUDE_locus_of_midpoints_is_single_point_l5_534

/-- A circle with center O and radius r -/
structure Circle where
  O : ℝ × ℝ
  r : ℝ
  h : r > 0

/-- A point P inside the circle on its diameter -/
structure InteriorPointOnDiameter (K : Circle) where
  P : ℝ × ℝ
  h₁ : dist P K.O < K.r
  h₂ : ∃ (t : ℝ), P = (K.O.1 + t * K.r, K.O.2) ∨ P = (K.O.1, K.O.2 + t * K.r)

/-- The midpoint of a chord passing through P -/
def midpoint_of_chord (K : Circle) (P : InteriorPointOnDiameter K) (θ : ℝ) : ℝ × ℝ :=
  P.P

/-- The theorem stating that the locus of midpoints is a single point -/
theorem locus_of_midpoints_is_single_point (K : Circle) (P : InteriorPointOnDiameter K) :
  ∀ θ : ℝ, midpoint_of_chord K P θ = P.P :=
sorry

end NUMINAMATH_CALUDE_locus_of_midpoints_is_single_point_l5_534


namespace NUMINAMATH_CALUDE_total_cups_calculation_l5_522

/-- Represents the quantities of ingredients in their original units -/
structure Ingredients :=
  (yellow_raisins : Float)  -- in cups
  (black_raisins : Float)   -- in cups
  (almonds : Float)         -- in ounces
  (pumpkin_seeds : Float)   -- in grams

/-- Conversion factors -/
def ounce_to_cup : Float := 0.125
def gram_to_cup : Float := 0.00423

/-- Calculates the total cups of ingredients -/
def total_cups (i : Ingredients) : Float :=
  i.yellow_raisins + i.black_raisins + (i.almonds * ounce_to_cup) + (i.pumpkin_seeds * gram_to_cup)

/-- The main theorem stating the total cups of ingredients -/
theorem total_cups_calculation (i : Ingredients) 
  (h1 : i.yellow_raisins = 0.3)
  (h2 : i.black_raisins = 0.4)
  (h3 : i.almonds = 5.5)
  (h4 : i.pumpkin_seeds = 150) :
  total_cups i = 2.022 := by
  sorry

#eval total_cups { yellow_raisins := 0.3, black_raisins := 0.4, almonds := 5.5, pumpkin_seeds := 150 }

end NUMINAMATH_CALUDE_total_cups_calculation_l5_522


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l5_505

theorem quadratic_equation_roots : ∃! x : ℝ, x^2 - 4*x + 4 = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l5_505


namespace NUMINAMATH_CALUDE_outfits_count_l5_529

/-- The number of unique outfits that can be made from given numbers of shirts, ties, and belts. -/
def number_of_outfits (shirts ties belts : ℕ) : ℕ := shirts * ties * belts

/-- Theorem stating that the number of unique outfits is 192 given 8 shirts, 6 ties, and 4 belts. -/
theorem outfits_count : number_of_outfits 8 6 4 = 192 := by
  sorry

end NUMINAMATH_CALUDE_outfits_count_l5_529


namespace NUMINAMATH_CALUDE_compare_cube_roots_l5_523

theorem compare_cube_roots : 
  (Real.rpow (25/3) (1/3 : ℝ)) < 
  ((Real.rpow 25 (1/3 : ℝ)) / 3 + Real.rpow (6/5) (1/3 : ℝ)) ∧
  ((Real.rpow 25 (1/3 : ℝ)) / 3 + Real.rpow (6/5) (1/3 : ℝ)) < 
  (Real.rpow (1148/135) (1/3 : ℝ)) := by
  sorry

end NUMINAMATH_CALUDE_compare_cube_roots_l5_523


namespace NUMINAMATH_CALUDE_x_intercepts_count_l5_583

-- Define the polynomial
def f (x : ℝ) : ℝ := (x - 5) * (x^2 + 8*x + 12)

-- State the theorem
theorem x_intercepts_count : 
  ∃ (a b c : ℝ), (∀ x : ℝ, f x = 0 ↔ x = a ∨ x = b ∨ x = c) ∧ 
  (a ≠ b ∧ b ≠ c ∧ a ≠ c) := by
  sorry

end NUMINAMATH_CALUDE_x_intercepts_count_l5_583


namespace NUMINAMATH_CALUDE_hyperbola_equation_from_focus_and_midpoint_l5_565

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a hyperbola -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_positive : a > 0 ∧ b > 0

/-- The equation of the hyperbola -/
def hyperbola_equation (h : Hyperbola) (p : Point) : Prop :=
  p.x^2 / h.a^2 - p.y^2 / h.b^2 = 1

theorem hyperbola_equation_from_focus_and_midpoint 
  (h : Hyperbola)
  (focus : Point)
  (midpoint : Point)
  (h_focus : focus.x = -2 ∧ focus.y = 0)
  (h_midpoint : midpoint.x = -3 ∧ midpoint.y = -1)
  (h_intersect : ∃ (A B : Point), 
    hyperbola_equation h A ∧ 
    hyperbola_equation h B ∧
    (A.x + B.x) / 2 = midpoint.x ∧
    (A.y + B.y) / 2 = midpoint.y) :
  h.a^2 = 3 ∧ h.b^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_equation_from_focus_and_midpoint_l5_565


namespace NUMINAMATH_CALUDE_inverse_sum_l5_584

def f (a b x : ℝ) : ℝ := a * x^2 + b * x

def f_inv (a b x : ℝ) : ℝ := b * x^2 + a * x

theorem inverse_sum (a b : ℝ) :
  (∀ x, f a b (f_inv a b x) = x) → a + b = -2 := by
  sorry

end NUMINAMATH_CALUDE_inverse_sum_l5_584


namespace NUMINAMATH_CALUDE_perpendicular_angles_counterexample_l5_547

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents an angle in 3D space -/
structure Angle3D where
  vertex : Point3D
  side1 : Point3D
  side2 : Point3D

/-- Checks if two line segments are perpendicular in 3D space -/
def isPerpendicular (a b c d : Point3D) : Prop := sorry

/-- Calculates the measure of an angle in degrees -/
def angleMeasure (angle : Angle3D) : ℝ := sorry

/-- Theorem: There exist angles with perpendicular sides that are neither equal nor sum to 180° -/
theorem perpendicular_angles_counterexample :
  ∃ (α β : Angle3D),
    isPerpendicular α.vertex α.side1 β.vertex β.side1 ∧
    isPerpendicular α.vertex α.side2 β.vertex β.side2 ∧
    angleMeasure α ≠ angleMeasure β ∧
    angleMeasure α + angleMeasure β ≠ 180 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_angles_counterexample_l5_547


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l5_594

theorem complex_modulus_problem (z : ℂ) (h : z = (2 + Complex.I) / Complex.I + Complex.I) : 
  Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l5_594


namespace NUMINAMATH_CALUDE_infinitely_many_primes_mod_3_eq_2_l5_593

theorem infinitely_many_primes_mod_3_eq_2 : Set.Infinite {p : ℕ | Nat.Prime p ∧ p % 3 = 2} := by
  sorry

end NUMINAMATH_CALUDE_infinitely_many_primes_mod_3_eq_2_l5_593


namespace NUMINAMATH_CALUDE_min_triple_intersection_l5_559

theorem min_triple_intersection (U : Finset Nat) (A B C : Finset Nat) : 
  Finset.card U = 30 →
  Finset.card A = 26 →
  Finset.card B = 23 →
  Finset.card C = 21 →
  A ⊆ U →
  B ⊆ U →
  C ⊆ U →
  10 ≤ Finset.card (A ∩ B ∩ C) :=
by sorry

end NUMINAMATH_CALUDE_min_triple_intersection_l5_559


namespace NUMINAMATH_CALUDE_candy_distribution_l5_550

theorem candy_distribution (total : Nat) (friends : Nat) (to_remove : Nat) : 
  total = 47 → friends = 5 → to_remove = 2 → 
  to_remove = (total % friends) ∧ 
  ∀ k : Nat, k < to_remove → (total - k) % friends ≠ 0 := by
sorry

end NUMINAMATH_CALUDE_candy_distribution_l5_550


namespace NUMINAMATH_CALUDE_root_product_equals_two_l5_501

theorem root_product_equals_two : 
  ∃ (x₁ x₂ x₃ : ℝ), 
    (Real.sqrt 4050 * x₁^3 - 8101 * x₁^2 + 4 = 0) ∧
    (Real.sqrt 4050 * x₂^3 - 8101 * x₂^2 + 4 = 0) ∧
    (Real.sqrt 4050 * x₃^3 - 8101 * x₃^2 + 4 = 0) ∧
    (x₁ < x₂) ∧ (x₂ < x₃) ∧
    (x₂ * (x₁ + x₃) = 2) := by
  sorry

end NUMINAMATH_CALUDE_root_product_equals_two_l5_501


namespace NUMINAMATH_CALUDE_max_closable_companies_l5_553

/-- The number of planets (vertices) in the Intergalactic empire -/
def n : ℕ := 10^2015

/-- The number of travel companies (colors) -/
def m : ℕ := 2015

/-- A function that determines if a graph remains connected after removing k colors -/
def remains_connected (k : ℕ) : Prop :=
  ∀ (removed_colors : Finset (Fin m)),
    removed_colors.card = k →
    ∃ (remaining_graph : SimpleGraph (Fin n)),
      remaining_graph.Connected

/-- The theorem stating the maximum number of companies that can be closed -/
theorem max_closable_companies :
  (∀ k ≤ 1007, remains_connected k) ∧
  ¬(remains_connected 1008) :=
sorry

end NUMINAMATH_CALUDE_max_closable_companies_l5_553


namespace NUMINAMATH_CALUDE_pyramid_on_cylinder_radius_l5_519

/-- A regular square pyramid with all edges equal to 1 -/
structure RegularSquarePyramid where
  edge_length : ℝ
  edge_equal : edge_length = 1

/-- An infinite right circular cylinder -/
structure RightCircularCylinder where
  radius : ℝ

/-- Predicate to check if all vertices of the pyramid lie on the lateral surface of the cylinder -/
def vertices_on_cylinder (p : RegularSquarePyramid) (c : RightCircularCylinder) : Prop :=
  sorry

/-- The main theorem stating the possible values of the cylinder's radius -/
theorem pyramid_on_cylinder_radius (p : RegularSquarePyramid) (c : RightCircularCylinder) :
  vertices_on_cylinder p c → (c.radius = 3 / (4 * Real.sqrt 2) ∨ c.radius = 1 / Real.sqrt 3) :=
sorry

end NUMINAMATH_CALUDE_pyramid_on_cylinder_radius_l5_519


namespace NUMINAMATH_CALUDE_polynomial_characterization_l5_578

/-- A polynomial with real coefficients -/
def RealPolynomial := ℝ → ℝ

/-- The condition that must hold for the polynomial -/
def PolynomialCondition (P : RealPolynomial) : Prop :=
  ∀ (x y z : ℝ), x ≠ 0 → y ≠ 0 → z ≠ 0 → 2*x*y*z = x + y + z →
    P x / (y*z) + P y / (z*x) + P z / (x*y) = P (x-y) + P (y-z) + P (z-x)

/-- The theorem statement -/
theorem polynomial_characterization (P : RealPolynomial) :
  PolynomialCondition P →
  ∃ (c : ℝ), ∀ x, P x = c * (x^2 + 3) :=
sorry

end NUMINAMATH_CALUDE_polynomial_characterization_l5_578


namespace NUMINAMATH_CALUDE_expand_expression_l5_531

theorem expand_expression (x y : ℝ) :
  -2 * (4 * x^3 - 3 * x * y + 5) = -8 * x^3 + 6 * x * y - 10 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l5_531


namespace NUMINAMATH_CALUDE_parabola_properties_l5_587

/-- A parabola with vertex at the origin, focus on the x-axis, and passing through (2, 2) -/
def parabola (x y : ℝ) : Prop := y^2 = 2*x

theorem parabola_properties :
  (parabola 0 0) ∧ 
  (∃ p : ℝ, p > 0 ∧ parabola p 0) ∧ 
  (parabola 2 2) := by
  sorry

end NUMINAMATH_CALUDE_parabola_properties_l5_587


namespace NUMINAMATH_CALUDE_calculate_expression_l5_504

theorem calculate_expression : 15 * 28 + 42 * 15 + 15^2 = 1275 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l5_504


namespace NUMINAMATH_CALUDE_class_fraction_proof_l5_545

/-- 
Given a class of students where:
1) The ratio of boys to girls is 2
2) Half the number of girls is equal to some fraction of the total number of students
This theorem proves that the fraction in condition 2 is 1/6
-/
theorem class_fraction_proof (G : ℚ) (h1 : G > 0) : 
  let B := 2 * G
  let total := G + B
  ∃ (x : ℚ), (1/2) * G = x * total ∧ x = 1/6 := by
sorry

end NUMINAMATH_CALUDE_class_fraction_proof_l5_545


namespace NUMINAMATH_CALUDE_sum_of_three_at_least_fifty_l5_520

theorem sum_of_three_at_least_fifty (S : Finset ℕ) (h1 : S.card = 7) 
  (h2 : ∀ x ∈ S, x > 0) (h3 : S.sum id = 100) :
  ∃ T ⊆ S, T.card = 3 ∧ T.sum id ≥ 50 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_three_at_least_fifty_l5_520


namespace NUMINAMATH_CALUDE_distance_between_homes_l5_528

/-- Proves that the distance between Maxwell's and Brad's homes is 36 km given the problem conditions -/
theorem distance_between_homes : 
  ∀ (maxwell_speed brad_speed maxwell_distance : ℝ),
    maxwell_speed = 2 →
    brad_speed = 4 →
    maxwell_distance = 12 →
    maxwell_distance + maxwell_distance * (brad_speed / maxwell_speed) = 36 :=
by
  sorry


end NUMINAMATH_CALUDE_distance_between_homes_l5_528


namespace NUMINAMATH_CALUDE_traffic_to_driving_ratio_l5_582

theorem traffic_to_driving_ratio (total_time driving_time : ℝ) 
  (h1 : total_time = 15)
  (h2 : driving_time = 5) :
  (total_time - driving_time) / driving_time = 2 := by
  sorry

end NUMINAMATH_CALUDE_traffic_to_driving_ratio_l5_582


namespace NUMINAMATH_CALUDE_chord_equation_of_ellipse_l5_571

/-- Given an ellipse and a point on a bisecting chord, prove the equation of the line containing the chord. -/
theorem chord_equation_of_ellipse (x y : ℝ) :
  let ellipse := fun (x y : ℝ) => x^2/16 + y^2/4 = 1
  let P := (-2, 1)
  let chord_bisector := fun (x y : ℝ) => ∃ (x1 y1 x2 y2 : ℝ),
    ellipse x1 y1 ∧ ellipse x2 y2 ∧ 
    x = (x1 + x2)/2 ∧ y = (y1 + y2)/2
  chord_bisector P.1 P.2 →
  x - 2*y + 4 = 0 := by sorry

end NUMINAMATH_CALUDE_chord_equation_of_ellipse_l5_571


namespace NUMINAMATH_CALUDE_special_number_composite_l5_525

/-- Represents the number formed by n+1 ones, followed by a 2, followed by n+1 ones -/
def special_number (n : ℕ) : ℕ :=
  (10^(n+1) - 1) / 9 * 10^(n+1) + (10^(n+1) - 1) / 9

/-- A number is composite if it has a factor between 1 and itself -/
def is_composite (m : ℕ) : Prop :=
  ∃ k, 1 < k ∧ k < m ∧ m % k = 0

/-- Theorem stating that the special number is composite for all natural numbers n -/
theorem special_number_composite (n : ℕ) : is_composite (special_number n) := by
  sorry


end NUMINAMATH_CALUDE_special_number_composite_l5_525


namespace NUMINAMATH_CALUDE_intersection_when_a_half_range_of_a_for_empty_intersection_l5_533

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {x | a - 1 < x ∧ x < 2*a + 1}
def B : Set ℝ := {x | 0 < x ∧ x < 1}

-- Theorem for part I
theorem intersection_when_a_half : 
  A (1/2) ∩ B = {x | 0 < x ∧ x < 1} := by sorry

-- Theorem for part II
theorem range_of_a_for_empty_intersection :
  ∀ a : ℝ, (A a).Nonempty → (A a ∩ B = ∅) → 
    ((-2 < a ∧ a ≤ -1/2) ∨ a ≥ 2) := by sorry

end NUMINAMATH_CALUDE_intersection_when_a_half_range_of_a_for_empty_intersection_l5_533


namespace NUMINAMATH_CALUDE_percentage_division_equality_l5_502

theorem percentage_division_equality : 
  (208 / 100 * 1265) / 6 = 438.53333333333336 := by sorry

end NUMINAMATH_CALUDE_percentage_division_equality_l5_502


namespace NUMINAMATH_CALUDE_union_of_A_and_B_intersection_condition_l5_580

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {x : ℝ | |x - 2| < a}
def B : Set ℝ := {x : ℝ | (2*x - 1) / (x + 2) < 1}

-- Part 1
theorem union_of_A_and_B :
  A 2 ∪ B = {x : ℝ | -2 < x ∧ x < 4} := by sorry

-- Part 2
theorem intersection_condition (a : ℝ) :
  A a ∩ B = A a ↔ a ≤ 1 := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_intersection_condition_l5_580


namespace NUMINAMATH_CALUDE_max_M_value_l5_511

theorem max_M_value (x y z u : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hu : u > 0)
  (eq1 : x - 2*y = z - 2*u) (eq2 : 2*y*z = u*x) (h_zy : z ≥ y) :
  ∃ M : ℝ, M > 0 ∧ M ≤ z/y ∧ ∀ N : ℝ, (N > 0 ∧ N ≤ z/y → N ≤ M) ∧ M = 6 + 4*Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_max_M_value_l5_511


namespace NUMINAMATH_CALUDE_right_triangular_prism_volume_l5_526

/-- The volume of a right triangular prism with base side lengths 14 and height 8 is 784 cubic units. -/
theorem right_triangular_prism_volume : 
  ∀ (base_side_length height : ℝ), 
    base_side_length = 14 → 
    height = 8 → 
    (1/2 * base_side_length * base_side_length) * height = 784 := by
  sorry

end NUMINAMATH_CALUDE_right_triangular_prism_volume_l5_526


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l5_574

/-- The sum of an arithmetic sequence with first term 1, common difference 2, and last term 17 -/
def arithmetic_sum : ℕ := 81

/-- The first term of the sequence -/
def a₁ : ℕ := 1

/-- The common difference of the sequence -/
def d : ℕ := 2

/-- The last term of the sequence -/
def aₙ : ℕ := 17

/-- The number of terms in the sequence -/
def n : ℕ := (aₙ - a₁) / d + 1

theorem arithmetic_sequence_sum :
  (n : ℕ) * (a₁ + aₙ) / 2 = arithmetic_sum :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l5_574


namespace NUMINAMATH_CALUDE_jeff_donuts_days_l5_573

/-- The number of donuts Jeff makes per day -/
def donuts_per_day : ℕ := 10

/-- The number of donuts Jeff eats per day -/
def jeff_eats_per_day : ℕ := 1

/-- The total number of donuts Chris eats -/
def chris_eats_total : ℕ := 8

/-- The number of donuts that fit in each box -/
def donuts_per_box : ℕ := 10

/-- The number of boxes Jeff can fill with his donuts -/
def boxes_filled : ℕ := 10

/-- The number of days Jeff makes donuts -/
def days_making_donuts : ℕ := 12

theorem jeff_donuts_days :
  days_making_donuts * (donuts_per_day - jeff_eats_per_day) - chris_eats_total =
  boxes_filled * donuts_per_box :=
by
  sorry

end NUMINAMATH_CALUDE_jeff_donuts_days_l5_573


namespace NUMINAMATH_CALUDE_max_A_at_375_l5_561

def A (k : ℕ) : ℝ := (Nat.choose 1500 k) * (0.3 ^ k)

theorem max_A_at_375 : 
  ∀ k : ℕ, k ≤ 1500 → A k ≤ A 375 :=
by sorry

end NUMINAMATH_CALUDE_max_A_at_375_l5_561


namespace NUMINAMATH_CALUDE_ones_digit_of_largest_power_of_three_dividing_18_factorial_l5_581

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

def largest_power_of_three_dividing (n : ℕ) : ℕ :=
  (List.range n).foldl (fun acc i => acc + (i + 1).log 3) 0

def ones_digit (n : ℕ) : ℕ := n % 10

theorem ones_digit_of_largest_power_of_three_dividing_18_factorial :
  ones_digit (3^(largest_power_of_three_dividing (factorial 18))) = 1 := by sorry

end NUMINAMATH_CALUDE_ones_digit_of_largest_power_of_three_dividing_18_factorial_l5_581


namespace NUMINAMATH_CALUDE_expression_evaluation_l5_591

theorem expression_evaluation (x y : ℚ) 
  (hx : x = -1) 
  (hy : y = -1/2) : 
  4*x*y + (2*x^2 + 5*x*y - y^2) - 2*(x^2 + 3*x*y) = 5/4 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l5_591


namespace NUMINAMATH_CALUDE_decimal_point_removal_l5_548

theorem decimal_point_removal (x y z : ℝ) (hx : x = 1.6) (hy : y = 16) (hz : z = 14.4) :
  y - x = z := by sorry

end NUMINAMATH_CALUDE_decimal_point_removal_l5_548


namespace NUMINAMATH_CALUDE_best_of_three_match_probability_l5_506

/-- The probability of player A winning a single set -/
def p : ℝ := 0.6

/-- The probability of player A winning the match in a best-of-three format -/
def prob_A_wins_match : ℝ := p^2 + 2 * p^2 * (1 - p)

theorem best_of_three_match_probability :
  prob_A_wins_match = 0.648 := by
  sorry

end NUMINAMATH_CALUDE_best_of_three_match_probability_l5_506


namespace NUMINAMATH_CALUDE_solution_set_characterization_l5_527

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

theorem solution_set_characterization (f : ℝ → ℝ) 
  (h_even : is_even f)
  (h_nonneg : ∀ x ≥ 0, f x = x^3 - 8) :
  {x : ℝ | f (x - 2) > 0} = {x : ℝ | x < 0 ∨ x > 4} := by
  sorry

end NUMINAMATH_CALUDE_solution_set_characterization_l5_527


namespace NUMINAMATH_CALUDE_sqrt_sum_fractions_l5_512

theorem sqrt_sum_fractions : 
  Real.sqrt ((25 : ℝ) / 36 + 16 / 9) = Real.sqrt 89 / 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_fractions_l5_512


namespace NUMINAMATH_CALUDE_problem_1_l5_586

theorem problem_1 : (1 + 1/4 - 5/6 + 1/2) * (-12) = -11 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_l5_586


namespace NUMINAMATH_CALUDE_fraction_problem_l5_515

theorem fraction_problem (x : ℚ) : 
  x / (4 * x - 6) = 3 / 4 → x = 9 / 4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_problem_l5_515


namespace NUMINAMATH_CALUDE_probability_specific_arrangement_l5_569

def num_letters : ℕ := 8

theorem probability_specific_arrangement (n : ℕ) (h : n = num_letters) :
  (1 : ℚ) / (n.factorial : ℚ) = 1 / 40320 :=
sorry

end NUMINAMATH_CALUDE_probability_specific_arrangement_l5_569


namespace NUMINAMATH_CALUDE_stating_spheres_fit_funnel_iff_l5_570

/-- Represents a conical funnel with two spheres inside it -/
structure ConicalFunnelWithSpheres where
  α : ℝ  -- Half of the axial section angle
  R : ℝ  -- Radius of the larger sphere
  r : ℝ  -- Radius of the smaller sphere
  h_angle_positive : 0 < α
  h_angle_less_than_pi_half : α < π / 2
  h_R_positive : 0 < R
  h_r_positive : 0 < r
  h_R_greater_r : r < R

/-- 
The necessary and sufficient condition for two spheres to be placed in a conical funnel 
such that they both touch its lateral surface
-/
def spheres_fit_condition (funnel : ConicalFunnelWithSpheres) : Prop :=
  Real.sin funnel.α ≤ (funnel.R - funnel.r) / funnel.R

/-- 
Theorem stating the necessary and sufficient condition for two spheres 
to fit in a conical funnel touching its lateral surface
-/
theorem spheres_fit_funnel_iff (funnel : ConicalFunnelWithSpheres) :
  (∃ (pos_R pos_r : ℝ), 
    pos_R > 0 ∧ pos_r > 0 ∧ pos_R = funnel.R ∧ pos_r = funnel.r ∧
    (∃ (config : ℝ × ℝ), 
      (config.1 > 0 ∧ config.2 > 0) ∧
      (config.1 + pos_R) * Real.sin funnel.α = pos_R ∧
      (config.2 + pos_r) * Real.sin funnel.α = pos_r ∧
      config.1 + pos_R + pos_r = config.2)) ↔
  spheres_fit_condition funnel :=
sorry

end NUMINAMATH_CALUDE_stating_spheres_fit_funnel_iff_l5_570
