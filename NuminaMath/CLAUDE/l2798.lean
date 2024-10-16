import Mathlib

namespace NUMINAMATH_CALUDE_janet_action_figures_l2798_279840

/-- Calculates the final number of action figures Janet has after selling, buying, and receiving a gift. -/
theorem janet_action_figures (initial : ℕ) (sold : ℕ) (bought : ℕ) (gift_multiplier : ℕ) : 
  initial = 10 → sold = 6 → bought = 4 → gift_multiplier = 2 →
  (initial - sold + bought) * (gift_multiplier + 1) = 24 :=
by
  sorry

end NUMINAMATH_CALUDE_janet_action_figures_l2798_279840


namespace NUMINAMATH_CALUDE_salt_teaspoons_in_recipe_l2798_279826

/-- Represents the recipe and sodium reduction problem -/
theorem salt_teaspoons_in_recipe : 
  ∀ (S : ℝ) 
    (parmesan_oz : ℝ) 
    (salt_sodium_per_tsp : ℝ) 
    (parmesan_sodium_per_oz : ℝ) 
    (parmesan_reduction : ℝ),
  parmesan_oz = 8 →
  salt_sodium_per_tsp = 50 →
  parmesan_sodium_per_oz = 25 →
  parmesan_reduction = 4 →
  (2 / 3) * (salt_sodium_per_tsp * S + parmesan_sodium_per_oz * parmesan_oz) = 
    salt_sodium_per_tsp * S + parmesan_sodium_per_oz * (parmesan_oz - parmesan_reduction) →
  S = 2 := by
  sorry

end NUMINAMATH_CALUDE_salt_teaspoons_in_recipe_l2798_279826


namespace NUMINAMATH_CALUDE_earnings_difference_proof_l2798_279857

/-- Calculates the difference in annual earnings between two jobs --/
def annual_earnings_difference (
  new_wage : ℕ
  ) (new_hours : ℕ
  ) (old_wage : ℕ
  ) (old_hours : ℕ
  ) (weeks_per_year : ℕ
  ) : ℕ :=
  (new_wage * new_hours * weeks_per_year) - (old_wage * old_hours * weeks_per_year)

/-- Proves that the difference in annual earnings is $20,800 --/
theorem earnings_difference_proof :
  annual_earnings_difference 20 40 16 25 52 = 20800 := by
  sorry

end NUMINAMATH_CALUDE_earnings_difference_proof_l2798_279857


namespace NUMINAMATH_CALUDE_prob_both_three_eq_one_forty_second_l2798_279849

/-- A fair die with n sides -/
def FairDie (n : ℕ) := Fin n

/-- The probability of rolling a specific number on a fair die with n sides -/
def prob_specific_roll (n : ℕ) : ℚ := 1 / n

/-- The probability of rolling a 3 on a 6-sided die and a 7-sided die simultaneously -/
def prob_both_three : ℚ := (prob_specific_roll 6) * (prob_specific_roll 7)

theorem prob_both_three_eq_one_forty_second :
  prob_both_three = 1 / 42 := by sorry

end NUMINAMATH_CALUDE_prob_both_three_eq_one_forty_second_l2798_279849


namespace NUMINAMATH_CALUDE_relationship_abc_l2798_279822

theorem relationship_abc (a b c : ℝ) :
  (∃ u v : ℝ, u - v = a ∧ u^2 - v^2 = b ∧ u^3 - v^3 = c) →
  3 * b^2 + a^4 = 4 * a * c := by
  sorry

end NUMINAMATH_CALUDE_relationship_abc_l2798_279822


namespace NUMINAMATH_CALUDE_transformation_converts_curve_l2798_279811

-- Define the original curve
def original_curve (x y : ℝ) : Prop := y = 2 * Real.sin (3 * x)

-- Define the transformed curve
def transformed_curve (x' y' : ℝ) : Prop := y' = Real.sin x'

-- Define the transformation
def transformation (x y x' y' : ℝ) : Prop := x' = 3 * x ∧ y' = (1/2) * y

-- Theorem statement
theorem transformation_converts_curve :
  ∀ x y x' y' : ℝ,
  original_curve x y →
  transformation x y x' y' →
  transformed_curve x' y' :=
sorry

end NUMINAMATH_CALUDE_transformation_converts_curve_l2798_279811


namespace NUMINAMATH_CALUDE_stock_recovery_l2798_279869

theorem stock_recovery (initial_price : ℝ) (initial_price_pos : initial_price > 0) : 
  let price_after_drops := initial_price * (1 - 0.1)^4
  ∃ n : ℕ, n ≥ 5 ∧ price_after_drops * (1 + 0.1)^n ≥ initial_price :=
by sorry

end NUMINAMATH_CALUDE_stock_recovery_l2798_279869


namespace NUMINAMATH_CALUDE_f_value_at_7_5_l2798_279876

def f_conditions (f : ℝ → ℝ) : Prop :=
  (∀ x, f (-x) = -f x) ∧  -- f is odd
  (∀ x, f (x + 2) = -f x) ∧  -- f(x+2) = -f(x)
  (∀ x, 0 ≤ x ∧ x ≤ 1 → f x = x)  -- f(x) = x for 0 ≤ x ≤ 1

theorem f_value_at_7_5 (f : ℝ → ℝ) (h : f_conditions f) : f 7.5 = -0.5 := by
  sorry

end NUMINAMATH_CALUDE_f_value_at_7_5_l2798_279876


namespace NUMINAMATH_CALUDE_constant_term_zero_implies_m_zero_l2798_279820

theorem constant_term_zero_implies_m_zero :
  ∀ m : ℝ, (m^2 - m = 0) → (m = 0) :=
by sorry

end NUMINAMATH_CALUDE_constant_term_zero_implies_m_zero_l2798_279820


namespace NUMINAMATH_CALUDE_consecutive_number_pair_l2798_279875

theorem consecutive_number_pair (a b : ℤ) : 
  (a = 18 ∨ b = 18) → -- One of the numbers is 18
  abs (a - b) = 1 → -- The numbers are consecutive
  a + b = 35 → -- Their sum is 35
  (a + b) % 5 = 0 → -- The sum is divisible by 5
  (a = 17 ∨ b = 17) := by sorry

end NUMINAMATH_CALUDE_consecutive_number_pair_l2798_279875


namespace NUMINAMATH_CALUDE_fraction_invariance_l2798_279854

theorem fraction_invariance (x y : ℝ) (h : x ≠ y) : 
  (3 * x) / (3 * x - 3 * y) = x / (x - y) := by
  sorry

end NUMINAMATH_CALUDE_fraction_invariance_l2798_279854


namespace NUMINAMATH_CALUDE_opposite_of_three_l2798_279890

theorem opposite_of_three : -(3 : ℝ) = -3 := by sorry

end NUMINAMATH_CALUDE_opposite_of_three_l2798_279890


namespace NUMINAMATH_CALUDE_quadratic_function_sum_l2798_279867

theorem quadratic_function_sum (a b c : ℝ) :
  (∀ x : ℝ, x^2 - 2*x + 2 ≤ a*x^2 + b*x + c) ∧
  (∀ x : ℝ, a*x^2 + b*x + c ≤ 2*x^2 - 4*x + 3) →
  a + b + c = 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_sum_l2798_279867


namespace NUMINAMATH_CALUDE_loan_amount_proof_l2798_279833

/-- Represents a loan with simple interest -/
structure Loan where
  principal : ℕ
  rate : ℕ
  time : ℕ
  interest : ℕ

/-- Calculates the simple interest for a loan -/
def simpleInterest (l : Loan) : ℕ :=
  l.principal * l.rate * l.time / 100

theorem loan_amount_proof (l : Loan) :
  l.rate = 8 ∧ l.time = l.rate ∧ l.interest = 704 →
  simpleInterest l = l.interest →
  l.principal = 1100 := by
sorry

end NUMINAMATH_CALUDE_loan_amount_proof_l2798_279833


namespace NUMINAMATH_CALUDE_calculation_proof_l2798_279879

theorem calculation_proof : 5^2 * 7 + 9 * 4 - 35 / 5 = 204 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l2798_279879


namespace NUMINAMATH_CALUDE_water_volume_in_cone_l2798_279815

/-- 
Theorem: For a cone filled with water up to 2/3 of its height, 
the volume of water is 8/27 of the total volume of the cone.
-/
theorem water_volume_in_cone (h r : ℝ) (h_pos : h > 0) (r_pos : r > 0) : 
  (1/3 * π * (2/3 * r)^2 * (2/3 * h)) / (1/3 * π * r^2 * h) = 8/27 := by
  sorry


end NUMINAMATH_CALUDE_water_volume_in_cone_l2798_279815


namespace NUMINAMATH_CALUDE_xiao_ming_school_time_l2798_279885

/-- Represents time in hours and minutes -/
structure Time where
  hours : ℕ
  minutes : ℕ
  valid : minutes < 60

/-- Calculates the difference between two Time values in minutes -/
def timeDifference (t1 t2 : Time) : ℕ :=
  (t2.hours * 60 + t2.minutes) - (t1.hours * 60 + t1.minutes)

/-- Converts minutes to Time -/
def minutesToTime (m : ℕ) : Time :=
  { hours := m / 60,
    minutes := m % 60,
    valid := by sorry }

theorem xiao_ming_school_time :
  let morning_arrival : Time := { hours := 7, minutes := 50, valid := by sorry }
  let morning_departure : Time := { hours := 11, minutes := 50, valid := by sorry }
  let afternoon_arrival : Time := { hours := 14, minutes := 10, valid := by sorry }
  let afternoon_departure : Time := { hours := 17, minutes := 0, valid := by sorry }
  let morning_time := timeDifference morning_arrival morning_departure
  let afternoon_time := timeDifference afternoon_arrival afternoon_departure
  let total_time := morning_time + afternoon_time
  minutesToTime total_time = { hours := 6, minutes := 50, valid := by sorry } :=
by sorry

end NUMINAMATH_CALUDE_xiao_ming_school_time_l2798_279885


namespace NUMINAMATH_CALUDE_prism_lateral_faces_are_parallelograms_l2798_279873

/-- A prism is a polyhedron with two congruent and parallel faces (called bases) 
    and all other faces (called lateral faces) are parallelograms. -/
structure Prism where
  -- We don't need to define the internal structure for this problem
  mk :: 

/-- A face of a polyhedron -/
structure Face where
  -- We don't need to define the internal structure for this problem
  mk ::

/-- Predicate to check if a face is a lateral face of a prism -/
def is_lateral_face (p : Prism) (f : Face) : Prop :=
  -- Definition omitted for brevity
  sorry

/-- Predicate to check if a face is a parallelogram -/
def is_parallelogram (f : Face) : Prop :=
  -- Definition omitted for brevity
  sorry

theorem prism_lateral_faces_are_parallelograms (p : Prism) :
  ∀ (f : Face), is_lateral_face p f → is_parallelogram f := by
  sorry

end NUMINAMATH_CALUDE_prism_lateral_faces_are_parallelograms_l2798_279873


namespace NUMINAMATH_CALUDE_vertical_complementary_implies_perpendicular_l2798_279861

/-- Two angles are vertical if they are opposite each other when two lines intersect. -/
def are_vertical_angles (α β : Real) : Prop := sorry

/-- Two angles are complementary if their sum is 90 degrees. -/
def are_complementary (α β : Real) : Prop := α + β = 90

/-- Two lines are perpendicular if they form a right angle (90 degrees) at their intersection. -/
def are_perpendicular_lines (l1 l2 : Line) : Prop := sorry

theorem vertical_complementary_implies_perpendicular (α β : Real) (l1 l2 : Line) :
  are_vertical_angles α β → are_complementary α β → are_perpendicular_lines l1 l2 := by
  sorry

end NUMINAMATH_CALUDE_vertical_complementary_implies_perpendicular_l2798_279861


namespace NUMINAMATH_CALUDE_julian_lego_count_l2798_279813

/-- The number of legos required for one airplane model -/
def legos_per_model : ℕ := 240

/-- The number of additional legos Julian needs -/
def additional_legos_needed : ℕ := 80

/-- The number of airplane models Julian wants to make -/
def number_of_models : ℕ := 2

/-- Theorem stating how many legos Julian has -/
theorem julian_lego_count :
  (number_of_models * legos_per_model) - additional_legos_needed = 400 := by
  sorry

end NUMINAMATH_CALUDE_julian_lego_count_l2798_279813


namespace NUMINAMATH_CALUDE_sum_divisible_by_101_iff_digits_congruent_l2798_279878

/-- Represents a four-digit positive integer with different non-zero digits -/
structure FourDigitNumber where
  a : Nat
  b : Nat
  c : Nat
  d : Nat
  a_pos : a > 0
  b_pos : b > 0
  c_pos : c > 0
  d_pos : d > 0
  all_different : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d
  a_lt_10 : a < 10
  b_lt_10 : b < 10
  c_lt_10 : c < 10
  d_lt_10 : d < 10

/-- The value of a four-digit number -/
def value (n : FourDigitNumber) : Nat :=
  1000 * n.a + 100 * n.b + 10 * n.c + n.d

/-- The value of the reverse of a four-digit number -/
def reverse_value (n : FourDigitNumber) : Nat :=
  1000 * n.d + 100 * n.c + 10 * n.b + n.a

/-- The theorem stating the condition for the sum of a number and its reverse to be divisible by 101 -/
theorem sum_divisible_by_101_iff_digits_congruent (n : FourDigitNumber) :
  (value n + reverse_value n) % 101 = 0 ↔ (n.a + n.d) % 101 = (n.b + n.c) % 101 := by
  sorry

end NUMINAMATH_CALUDE_sum_divisible_by_101_iff_digits_congruent_l2798_279878


namespace NUMINAMATH_CALUDE_f_range_l2798_279859

def f (x : ℕ) : ℤ := Int.floor ((x + 1) / 2 : ℚ) - Int.floor (x / 2 : ℚ)

theorem f_range : ∀ x : ℕ, f x = 0 ∨ f x = 1 ∧ ∃ a b : ℕ, f a = 0 ∧ f b = 1 := by
  sorry

end NUMINAMATH_CALUDE_f_range_l2798_279859


namespace NUMINAMATH_CALUDE_roots_in_unit_interval_l2798_279888

noncomputable def f (q : ℕ → ℝ) : ℕ → ℝ → ℝ
| 0, x => 1
| 1, x => x
| (n + 2), x => (1 + q n) * x * f q (n + 1) x - q n * f q n x

theorem roots_in_unit_interval (q : ℕ → ℝ) (h : ∀ n, q n > 0) :
  ∀ n : ℕ, ∀ x : ℝ, |x| > 1 → |f q (n + 1) x| > |f q n x| :=
sorry

end NUMINAMATH_CALUDE_roots_in_unit_interval_l2798_279888


namespace NUMINAMATH_CALUDE_asha_win_probability_l2798_279874

theorem asha_win_probability (lose_prob tie_prob : ℚ) 
  (lose_prob_val : lose_prob = 5/12)
  (tie_prob_val : tie_prob = 1/6)
  (total_prob : lose_prob + tie_prob + (1 - lose_prob - tie_prob) = 1) :
  1 - lose_prob - tie_prob = 5/12 := by
  sorry

end NUMINAMATH_CALUDE_asha_win_probability_l2798_279874


namespace NUMINAMATH_CALUDE_perfect_square_example_l2798_279868

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

theorem perfect_square_example : is_perfect_square (4^10 * 5^5 * 6^10) := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_example_l2798_279868


namespace NUMINAMATH_CALUDE_tony_fish_count_l2798_279807

/-- The number of fish Tony has after a given number of years -/
def fish_count (initial_fish : ℕ) (years : ℕ) : ℕ :=
  initial_fish + years * (3 - 2)

/-- Theorem: Tony will have 15 fish after 10 years -/
theorem tony_fish_count : fish_count 5 10 = 15 := by
  sorry

end NUMINAMATH_CALUDE_tony_fish_count_l2798_279807


namespace NUMINAMATH_CALUDE_cost_price_calculation_cost_price_proof_l2798_279883

theorem cost_price_calculation (selling_price : ℝ) (discount_rate : ℝ) (profit_rate : ℝ) : ℝ :=
  let discounted_price := selling_price * (1 - discount_rate)
  let cost_price := discounted_price / (1 + profit_rate)
  cost_price

theorem cost_price_proof :
  cost_price_calculation 12000 0.1 0.08 = 10000 := by
  sorry

end NUMINAMATH_CALUDE_cost_price_calculation_cost_price_proof_l2798_279883


namespace NUMINAMATH_CALUDE_tank_A_height_approx_5_l2798_279852

/-- The circumference of Tank A in meters -/
def circumference_A : ℝ := 4

/-- The circumference of Tank B in meters -/
def circumference_B : ℝ := 10

/-- The height of Tank B in meters -/
def height_B : ℝ := 8

/-- The ratio of Tank A's capacity to Tank B's capacity -/
def capacity_ratio : ℝ := 0.10000000000000002

/-- The height of Tank A in meters -/
noncomputable def height_A : ℝ := 
  capacity_ratio * (circumference_B / circumference_A)^2 * height_B

theorem tank_A_height_approx_5 : 
  ∃ ε > 0, abs (height_A - 5) < ε := by sorry

end NUMINAMATH_CALUDE_tank_A_height_approx_5_l2798_279852


namespace NUMINAMATH_CALUDE_log_equation_proof_l2798_279839

-- Define the natural logarithm function
noncomputable def ln (x : ℝ) : ℝ := Real.log x

-- State the theorem
theorem log_equation_proof :
  (ln 5) ^ 2 + ln 2 * ln 50 = 1 := by sorry

end NUMINAMATH_CALUDE_log_equation_proof_l2798_279839


namespace NUMINAMATH_CALUDE_perpendicular_lines_n_value_l2798_279897

/-- Two perpendicular lines with a given foot of perpendicular -/
structure PerpendicularLines where
  m : ℝ
  n : ℝ
  p : ℝ
  line1_eq : ∀ x y, m * x + 4 * y - 2 = 0
  line2_eq : ∀ x y, 2 * x - 5 * y + n = 0
  perpendicular : m * 2 + 4 * 5 = 0
  foot_on_line1 : m * 1 + 4 * p - 2 = 0
  foot_on_line2 : 2 * 1 - 5 * p + n = 0

/-- The value of n in the given perpendicular lines setup is -12 -/
theorem perpendicular_lines_n_value (pl : PerpendicularLines) : pl.n = -12 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_lines_n_value_l2798_279897


namespace NUMINAMATH_CALUDE_wind_pressure_theorem_l2798_279886

/-- The pressure-area-velocity relationship for wind on a sail -/
theorem wind_pressure_theorem (k : ℝ) :
  (∃ P A V : ℝ, P = k * A * V^2 ∧ P = 1.25 ∧ A = 1 ∧ V = 20) →
  (∃ P A V : ℝ, P = k * A * V^2 ∧ P = 20 ∧ A = 4 ∧ V = 40) :=
by sorry

end NUMINAMATH_CALUDE_wind_pressure_theorem_l2798_279886


namespace NUMINAMATH_CALUDE_gas_station_candy_boxes_l2798_279821

/-- Given a gas station that sold 2 boxes of chocolate candy, 5 boxes of sugar candy,
    and some boxes of gum, with a total of 9 boxes sold, prove that 2 boxes of gum were sold. -/
theorem gas_station_candy_boxes : 
  let chocolate_boxes : ℕ := 2
  let sugar_boxes : ℕ := 5
  let total_boxes : ℕ := 9
  let gum_boxes : ℕ := total_boxes - chocolate_boxes - sugar_boxes
  gum_boxes = 2 := by sorry

end NUMINAMATH_CALUDE_gas_station_candy_boxes_l2798_279821


namespace NUMINAMATH_CALUDE_rubber_band_difference_l2798_279802

theorem rubber_band_difference (justine bailey ylona : ℕ) : 
  ylona = 24 →
  justine = ylona - 2 →
  bailey + 4 = 8 →
  justine > bailey →
  justine - bailey = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_rubber_band_difference_l2798_279802


namespace NUMINAMATH_CALUDE_egg_sale_remainder_l2798_279856

theorem egg_sale_remainder : (53 + 65 + 26) % 15 = 9 := by sorry

end NUMINAMATH_CALUDE_egg_sale_remainder_l2798_279856


namespace NUMINAMATH_CALUDE_count_triangles_including_center_l2798_279817

/-- Given a regular polygon with 2n + 1 sides, this function calculates the number of triangles
    formed by its vertices that include the center of the polygon. -/
def trianglesIncludingCenter (n : ℕ) : ℕ :=
  n * (n + 1) * (2 * n + 1) / 6

/-- Theorem stating that the number of triangles including the center of a regular polygon
    with 2n + 1 sides is equal to n(n+1)(2n+1)/6 -/
theorem count_triangles_including_center (n : ℕ) :
  trianglesIncludingCenter n = n * (n + 1) * (2 * n + 1) / 6 := by
  sorry

end NUMINAMATH_CALUDE_count_triangles_including_center_l2798_279817


namespace NUMINAMATH_CALUDE_parabola_shift_up_two_l2798_279824

/-- Represents a vertical shift transformation of a parabola -/
def verticalShift (f : ℝ → ℝ) (k : ℝ) : ℝ → ℝ := λ x => f x + k

/-- The original parabola function -/
def originalParabola : ℝ → ℝ := λ x => x^2

/-- Theorem: Shifting the parabola y = x^2 up by 2 units results in y = x^2 + 2 -/
theorem parabola_shift_up_two :
  verticalShift originalParabola 2 = λ x => x^2 + 2 := by
  sorry

end NUMINAMATH_CALUDE_parabola_shift_up_two_l2798_279824


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l2798_279872

-- Define the quadratic function
def f (x : ℝ) : ℝ := -x^2 - x + 2

-- Define the solution set
def solution_set : Set ℝ := {x | -2 ≤ x ∧ x ≤ 1}

-- Theorem statement
theorem quadratic_inequality_solution :
  {x : ℝ | f x ≥ 0} = solution_set := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l2798_279872


namespace NUMINAMATH_CALUDE_tangent_line_parallel_to_given_line_l2798_279850

-- Define the curve
def f (x : ℝ) : ℝ := x^3 + x - 10

-- Define the derivative of the curve
def f' (x : ℝ) : ℝ := 3 * x^2 + 1

theorem tangent_line_parallel_to_given_line :
  ∀ x₀ y₀ : ℝ,
  f x₀ = y₀ →
  f' x₀ = 4 →
  ((x₀ = 1 ∧ y₀ = -8) ∨ (x₀ = -1 ∧ y₀ = -12)) ∧
  ((y₀ = 4 * x₀ - 12) ∨ (y₀ = 4 * x₀ - 8)) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_parallel_to_given_line_l2798_279850


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2798_279853

theorem inequality_solution_set : 
  {x : ℝ | 3 ≤ |5 - 2*x| ∧ |5 - 2*x| < 9} = 
  Set.union (Set.Ioc (-2) 1) (Set.Icc 4 7) := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2798_279853


namespace NUMINAMATH_CALUDE_sum_of_powers_of_i_equals_zero_l2798_279893

-- Define the imaginary unit i
def i : ℂ := Complex.I

-- State the theorem
theorem sum_of_powers_of_i_equals_zero :
  i^14560 + i^14561 + i^14562 + i^14563 = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_powers_of_i_equals_zero_l2798_279893


namespace NUMINAMATH_CALUDE_road_length_l2798_279889

theorem road_length (repaired : ℚ) (remaining_extra : ℚ) : 
  repaired = 7/15 → remaining_extra = 2/5 → repaired + (repaired + remaining_extra) = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_road_length_l2798_279889


namespace NUMINAMATH_CALUDE_ellipse_intersection_midpoint_l2798_279864

-- Define the ellipse C
def ellipse_C (x y : ℝ) : Prop :=
  x^2 / 9 + y^2 = 1

-- Define the line
def line (x y : ℝ) : Prop :=
  y = x + 2

-- Theorem statement
theorem ellipse_intersection_midpoint :
  -- Given conditions
  let f1 : ℝ × ℝ := (-2 * Real.sqrt 2, 0)
  let f2 : ℝ × ℝ := (2 * Real.sqrt 2, 0)
  let major_axis_length : ℝ := 6

  -- Prove that
  -- 1. The standard equation of ellipse C is x²/9 + y² = 1
  (∀ x y : ℝ, ellipse_C x y ↔ x^2 / 9 + y^2 = 1) ∧
  -- 2. The midpoint of intersection points has coordinates (-9/5, 1/5)
  (∃ x1 y1 x2 y2 : ℝ,
    ellipse_C x1 y1 ∧ ellipse_C x2 y2 ∧
    line x1 y1 ∧ line x2 y2 ∧
    x1 ≠ x2 ∧
    (x1 + x2) / 2 = -9/5 ∧
    (y1 + y2) / 2 = 1/5) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_intersection_midpoint_l2798_279864


namespace NUMINAMATH_CALUDE_height_statistics_l2798_279837

/-- Heights of students in Class A -/
def class_a_heights : Finset ℕ := sorry

/-- Heights of students in Class B -/
def class_b_heights : Finset ℕ := sorry

/-- The mode of a finite set of natural numbers -/
def mode (s : Finset ℕ) : ℕ := sorry

/-- The median of a finite set of natural numbers -/
def median (s : Finset ℕ) : ℕ := sorry

/-- Theorem stating the mode of Class A heights and median of Class B heights -/
theorem height_statistics :
  mode class_a_heights = 171 ∧ median class_b_heights = 170 := by sorry

end NUMINAMATH_CALUDE_height_statistics_l2798_279837


namespace NUMINAMATH_CALUDE_concert_longest_song_duration_l2798_279838

/-- Represents the duration of the longest song in a concert --/
def longest_song_duration (total_time intermission_time num_songs regular_song_duration : ℕ) : ℕ :=
  total_time - intermission_time - (num_songs - 1) * regular_song_duration

/-- Theorem stating the duration of the longest song in the given concert scenario --/
theorem concert_longest_song_duration :
  longest_song_duration 80 10 13 5 = 10 := by sorry

end NUMINAMATH_CALUDE_concert_longest_song_duration_l2798_279838


namespace NUMINAMATH_CALUDE_division_problem_l2798_279825

theorem division_problem (x : ℚ) : 
  (2976 / x - 240 = 8) → x = 12 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l2798_279825


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_hyperbola_eccentricity_is_sqrt_5_l2798_279894

/-- The eccentricity of a hyperbola with specific properties -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) : ℝ :=
  let P : ℝ × ℝ := sorry
  let F₁ : ℝ × ℝ := sorry
  let F₂ : ℝ × ℝ := sorry
  let hyperbola := fun (x y : ℝ) ↦ x^2 / a^2 - y^2 / b^2 = 1
  let circle := fun (x y : ℝ) ↦ x^2 + y^2 = a^2 + b^2
  let distance := fun (p q : ℝ × ℝ) ↦ Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  have h1 : hyperbola P.1 P.2 := sorry
  have h2 : circle P.1 P.2 := sorry
  have h3 : P.1 ≥ 0 ∧ P.2 ≥ 0 := sorry  -- P is in the first quadrant
  have h4 : distance P F₁ = 2 * distance P F₂ := sorry
  Real.sqrt 5

theorem hyperbola_eccentricity_is_sqrt_5 (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  hyperbola_eccentricity a b ha hb = Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_hyperbola_eccentricity_is_sqrt_5_l2798_279894


namespace NUMINAMATH_CALUDE_brady_record_theorem_l2798_279866

/-- The minimum average yards per game needed to beat the record -/
def min_avg_yards_per_game (current_record : ℕ) (current_yards : ℕ) (games_left : ℕ) : ℚ :=
  (current_record + 1 - current_yards) / games_left

/-- Theorem stating the minimum average yards per game needed to beat the record -/
theorem brady_record_theorem (current_record : ℕ) (current_yards : ℕ) (games_left : ℕ)
  (h1 : current_record = 5999)
  (h2 : current_yards = 4200)
  (h3 : games_left = 6) :
  min_avg_yards_per_game current_record current_yards games_left = 300 := by
  sorry

end NUMINAMATH_CALUDE_brady_record_theorem_l2798_279866


namespace NUMINAMATH_CALUDE_factorization_of_360_l2798_279842

theorem factorization_of_360 : ∃ (p₁ p₂ p₃ : Nat) (e₁ e₂ e₃ : Nat),
  Prime p₁ ∧ Prime p₂ ∧ Prime p₃ ∧
  p₁ ≠ p₂ ∧ p₁ ≠ p₃ ∧ p₂ ≠ p₃ ∧
  360 = p₁^e₁ * p₂^e₂ * p₃^e₃ ∧
  (∀ q : Nat, Prime q → q ∣ 360 → (q = p₁ ∨ q = p₂ ∨ q = p₃)) ∧
  (e₁ ≤ 3 ∧ e₂ ≤ 3 ∧ e₃ ≤ 3) ∧
  (e₁ = 3 ∨ e₂ = 3 ∨ e₃ = 3) :=
by sorry

end NUMINAMATH_CALUDE_factorization_of_360_l2798_279842


namespace NUMINAMATH_CALUDE_smallest_value_x_plus_inv_x_l2798_279829

theorem smallest_value_x_plus_inv_x (x : ℝ) (h : 11 = x^2 + 1/x^2) :
  ∃ y : ℝ, y = x + 1/x ∧ y ≥ -Real.sqrt 13 ∧ (∀ z : ℝ, z = x + 1/x → z ≥ y) :=
by sorry

end NUMINAMATH_CALUDE_smallest_value_x_plus_inv_x_l2798_279829


namespace NUMINAMATH_CALUDE_gina_money_theorem_l2798_279835

theorem gina_money_theorem (initial_amount : ℚ) : 
  initial_amount = 400 → 
  initial_amount - (initial_amount * (1/4 + 1/8 + 1/5)) = 170 := by
  sorry

end NUMINAMATH_CALUDE_gina_money_theorem_l2798_279835


namespace NUMINAMATH_CALUDE_number_sum_problem_l2798_279865

theorem number_sum_problem (x : ℝ) (h : 20 + x = 30) : x = 10 := by
  sorry

end NUMINAMATH_CALUDE_number_sum_problem_l2798_279865


namespace NUMINAMATH_CALUDE_total_flowers_in_two_weeks_l2798_279809

/-- Represents the flowers Miriam takes care of in a day -/
structure DailyFlowers where
  roses : ℕ
  tulips : ℕ
  daisies : ℕ
  lilies : ℕ
  sunflowers : ℕ

/-- Calculates the total number of flowers for a day -/
def totalFlowers (df : DailyFlowers) : ℕ :=
  df.roses + df.tulips + df.daisies + df.lilies + df.sunflowers

/-- Represents Miriam's work schedule for a week -/
structure WeekSchedule where
  monday : ℕ
  tuesday : ℕ
  wednesday : ℕ
  thursday : ℕ
  saturday : ℕ

/-- Miriam's work hours in the first week -/
def firstWeekSchedule : WeekSchedule :=
  { monday := 4, tuesday := 5, wednesday := 3, thursday := 6, saturday := 5 }

/-- Flowers taken care of in the first week -/
def firstWeekFlowers : DailyFlowers :=
  { roses := 40, tulips := 50, daisies := 36, lilies := 48, sunflowers := 55 }

/-- Calculates the improved number of flowers with 20% increase -/
def improvePerformance (n : ℕ) : ℕ :=
  n + (n / 5)

/-- Theorem stating that the total number of flowers Miriam takes care of in two weeks is 504 -/
theorem total_flowers_in_two_weeks :
  let secondWeekFlowers : DailyFlowers :=
    { roses := improvePerformance firstWeekFlowers.roses,
      tulips := improvePerformance firstWeekFlowers.tulips,
      daisies := improvePerformance firstWeekFlowers.daisies,
      lilies := improvePerformance firstWeekFlowers.lilies,
      sunflowers := improvePerformance firstWeekFlowers.sunflowers }
  totalFlowers firstWeekFlowers + totalFlowers secondWeekFlowers = 504 := by
  sorry

end NUMINAMATH_CALUDE_total_flowers_in_two_weeks_l2798_279809


namespace NUMINAMATH_CALUDE_shipping_cost_calculation_l2798_279846

/-- The shipping cost per unit for an electronic component manufacturer -/
def shipping_cost : ℝ := 1.67

/-- The production cost per component -/
def production_cost : ℝ := 80

/-- The fixed monthly costs -/
def fixed_costs : ℝ := 16500

/-- The number of components produced and sold per month -/
def monthly_sales : ℕ := 150

/-- The lowest selling price per component -/
def selling_price : ℝ := 191.67

theorem shipping_cost_calculation :
  shipping_cost = (selling_price * monthly_sales - production_cost * monthly_sales - fixed_costs) / monthly_sales :=
by sorry

end NUMINAMATH_CALUDE_shipping_cost_calculation_l2798_279846


namespace NUMINAMATH_CALUDE_eel_species_count_l2798_279814

/-- Given the number of species identified in an aquarium, prove the number of eel species. -/
theorem eel_species_count (total : ℕ) (sharks : ℕ) (whales : ℕ) (h1 : total = 55) (h2 : sharks = 35) (h3 : whales = 5) :
  total - sharks - whales = 15 := by
  sorry

end NUMINAMATH_CALUDE_eel_species_count_l2798_279814


namespace NUMINAMATH_CALUDE_reflected_ray_equation_l2798_279804

/-- The equation of a reflected ray given an incident ray and a reflecting line -/
theorem reflected_ray_equation 
  (incident_ray : ℝ → ℝ → Prop) 
  (reflecting_line : ℝ → ℝ → Prop) 
  (reflected_ray : ℝ → ℝ → Prop) : 
  (∀ x y, incident_ray x y ↔ x - 2*y + 3 = 0) →
  (∀ x y, reflecting_line x y ↔ y = x) →
  (∀ x y, reflected_ray x y ↔ 2*x - y - 3 = 0) := by
sorry

end NUMINAMATH_CALUDE_reflected_ray_equation_l2798_279804


namespace NUMINAMATH_CALUDE_unique_factorial_solution_l2798_279832

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem unique_factorial_solution :
  ∃! (n : ℕ), n > 0 ∧ factorial (n + 1) + factorial (n + 4) = factorial n * 3480 :=
sorry

end NUMINAMATH_CALUDE_unique_factorial_solution_l2798_279832


namespace NUMINAMATH_CALUDE_certain_number_proof_l2798_279834

theorem certain_number_proof :
  ∃ x : ℝ, x * (-4.5) = 2 * (-4.5) - 36 ∧ x = 10 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l2798_279834


namespace NUMINAMATH_CALUDE_third_day_sales_formula_l2798_279803

/-- Represents the sales of sportswear over three days -/
structure SportswearSales where
  /-- Sales on the first day -/
  first_day : ℕ
  /-- Parameter m used in calculations -/
  m : ℕ

/-- Calculates the sales on the second day -/
def second_day_sales (s : SportswearSales) : ℤ :=
  3 * s.first_day - 3 * s.m

/-- Calculates the sales on the third day -/
def third_day_sales (s : SportswearSales) : ℤ :=
  second_day_sales s + s.m

/-- Theorem stating that the third day sales equal 3a - 2m -/
theorem third_day_sales_formula (s : SportswearSales) :
  third_day_sales s = 3 * s.first_day - 2 * s.m :=
by
  sorry

end NUMINAMATH_CALUDE_third_day_sales_formula_l2798_279803


namespace NUMINAMATH_CALUDE_odd_square_plus_two_divisor_congruence_l2798_279836

theorem odd_square_plus_two_divisor_congruence (a d : ℤ) : 
  Odd a → a > 0 → d ∣ (a^2 + 2) → d % 8 = 1 ∨ d % 8 = 3 := by
  sorry

end NUMINAMATH_CALUDE_odd_square_plus_two_divisor_congruence_l2798_279836


namespace NUMINAMATH_CALUDE_rice_cooking_is_algorithm_l2798_279891

/-- Characteristics of an algorithm -/
structure AlgorithmCharacteristics where
  finite : Bool
  definite : Bool
  sequential : Bool
  correct : Bool
  nonUnique : Bool
  universal : Bool

/-- Representation of an algorithm -/
inductive AlgorithmRepresentation
  | NaturalLanguage
  | GraphicalLanguage
  | ProgrammingLanguage

/-- Steps for cooking rice -/
inductive RiceCookingStep
  | WashPot
  | RinseRice
  | AddWater
  | Heat

/-- Definition of an algorithm -/
def isAlgorithm (steps : List RiceCookingStep) (representation : AlgorithmRepresentation) 
  (characteristics : AlgorithmCharacteristics) : Prop :=
  characteristics.finite ∧
  characteristics.definite ∧
  characteristics.sequential ∧
  characteristics.correct ∧
  characteristics.nonUnique ∧
  characteristics.universal

/-- Theorem: The steps for cooking rice form an algorithm -/
theorem rice_cooking_is_algorithm : 
  ∃ (representation : AlgorithmRepresentation) (characteristics : AlgorithmCharacteristics),
    isAlgorithm [RiceCookingStep.WashPot, RiceCookingStep.RinseRice, 
                 RiceCookingStep.AddWater, RiceCookingStep.Heat] 
                representation characteristics :=
  sorry

end NUMINAMATH_CALUDE_rice_cooking_is_algorithm_l2798_279891


namespace NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l2798_279844

-- Define the propositions
def p (a : ℝ) : Prop := ∀ x : ℝ, x^2 + 2*a*x - a > 0

def q (a : ℝ) : Prop := a < 0

-- State the theorem
theorem p_sufficient_not_necessary_for_q :
  (∀ a : ℝ, p a → q a) ∧ (∃ a : ℝ, q a ∧ ¬(p a)) :=
sorry

end NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l2798_279844


namespace NUMINAMATH_CALUDE_shortest_altitude_of_triangle_l2798_279830

/-- The shortest altitude of a triangle with sides 9, 12, and 15 is 7.2 -/
theorem shortest_altitude_of_triangle (a b c h : ℝ) : 
  a = 9 → b = 12 → c = 15 → 
  a^2 + b^2 = c^2 →
  h = (2 * (a * b) / 2) / c →
  h = 7.2 := by sorry

end NUMINAMATH_CALUDE_shortest_altitude_of_triangle_l2798_279830


namespace NUMINAMATH_CALUDE_linear_function_through_points_l2798_279895

/-- A linear function passing through two points -/
def linear_function (k b : ℝ) (x : ℝ) : ℝ := k * x + b

theorem linear_function_through_points :
  ∃ k b : ℝ, 
    (linear_function k b 3 = 5) ∧ 
    (linear_function k b (-4) = -9) ∧
    (∀ x : ℝ, linear_function k b x = 2 * x - 1) := by
  sorry

end NUMINAMATH_CALUDE_linear_function_through_points_l2798_279895


namespace NUMINAMATH_CALUDE_final_state_is_green_l2798_279882

/-- Represents the colors of chameleons -/
inductive Color
  | Yellow
  | Red
  | Green

/-- Represents the state of chameleons on the island -/
structure ChameleonState where
  yellow : Nat
  red : Nat
  green : Nat

/-- The initial state of chameleons -/
def initialState : ChameleonState :=
  { yellow := 7, red := 10, green := 17 }

/-- The total number of chameleons -/
def totalChameleons : Nat := 34

/-- Function to model the color change when two chameleons of different colors meet -/
def colorChange (state : ChameleonState) : ChameleonState :=
  sorry

/-- Predicate to check if all chameleons have the same color -/
def allSameColor (state : ChameleonState) : Prop :=
  (state.yellow = totalChameleons) ∨ (state.red = totalChameleons) ∨ (state.green = totalChameleons)

/-- The main theorem to prove -/
theorem final_state_is_green :
  ∃ (finalState : ChameleonState),
    (allSameColor finalState) ∧ (finalState.green = totalChameleons) :=
  sorry

end NUMINAMATH_CALUDE_final_state_is_green_l2798_279882


namespace NUMINAMATH_CALUDE_base_conversion_l2798_279884

/-- Given that in base x, the decimal number 67 is written as 47, prove that x = 15 -/
theorem base_conversion (x : ℕ) (h : 4 * x + 7 = 67) : x = 15 := by
  sorry

end NUMINAMATH_CALUDE_base_conversion_l2798_279884


namespace NUMINAMATH_CALUDE_cara_don_meeting_l2798_279818

/-- The distance Cara walks before meeting Don -/
def distance_cara_walks : ℝ := 18

/-- The total distance between Cara's and Don's homes -/
def total_distance : ℝ := 45

/-- Cara's walking speed in km/h -/
def cara_speed : ℝ := 6

/-- Don's walking speed in km/h -/
def don_speed : ℝ := 5

/-- The time Don starts walking after Cara (in hours) -/
def don_start_delay : ℝ := 2

theorem cara_don_meeting :
  distance_cara_walks = 18 ∧
  distance_cara_walks + don_speed * (distance_cara_walks / cara_speed) =
    total_distance - cara_speed * don_start_delay :=
sorry

end NUMINAMATH_CALUDE_cara_don_meeting_l2798_279818


namespace NUMINAMATH_CALUDE_sue_candy_count_l2798_279810

/-- Represents the number of candies each person has -/
structure CandyCount where
  bob : Nat
  mary : Nat
  john : Nat
  sam : Nat
  sue : Nat

/-- The total number of candies for all friends -/
def totalCandies (cc : CandyCount) : Nat :=
  cc.bob + cc.mary + cc.john + cc.sam + cc.sue

theorem sue_candy_count (cc : CandyCount) 
  (h1 : cc.bob = 10)
  (h2 : cc.mary = 5)
  (h3 : cc.john = 5)
  (h4 : cc.sam = 10)
  (h5 : totalCandies cc = 50) :
  cc.sue = 20 := by
  sorry


end NUMINAMATH_CALUDE_sue_candy_count_l2798_279810


namespace NUMINAMATH_CALUDE_circumcenter_from_equal_distances_l2798_279831

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A triangle in 3D space -/
structure Triangle3D where
  A : Point3D
  B : Point3D
  C : Point3D

/-- Check if a point is outside a plane -/
def isOutsidePlane (P : Point3D) (t : Triangle3D) : Prop := sorry

/-- Check if a line is perpendicular to a plane -/
def isPerpendicularToPlane (P O : Point3D) (t : Triangle3D) : Prop := sorry

/-- Check if a point is the foot of a perpendicular -/
def isFootOfPerpendicular (O : Point3D) (P : Point3D) (t : Triangle3D) : Prop := sorry

/-- Calculate the distance between two points -/
def distance (P Q : Point3D) : ℝ := sorry

/-- Check if a point is the circumcenter of a triangle -/
def isCircumcenter (O : Point3D) (t : Triangle3D) : Prop := sorry

/-- Main theorem -/
theorem circumcenter_from_equal_distances (P O : Point3D) (t : Triangle3D) :
  isOutsidePlane P t →
  isPerpendicularToPlane P O t →
  isFootOfPerpendicular O P t →
  distance P t.A = distance P t.B ∧ distance P t.B = distance P t.C →
  isCircumcenter O t := by sorry

end NUMINAMATH_CALUDE_circumcenter_from_equal_distances_l2798_279831


namespace NUMINAMATH_CALUDE_nephews_ages_sum_l2798_279871

theorem nephews_ages_sum :
  ∀ (a b c d : ℕ),
    a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 →  -- single-digit
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →  -- distinct
    a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 →  -- positive
    ((a * b = 36 ∧ c * d = 40) ∨ (a * c = 36 ∧ b * d = 40) ∨ 
     (a * d = 36 ∧ b * c = 40) ∨ (b * c = 36 ∧ a * d = 40) ∨ 
     (b * d = 36 ∧ a * c = 40) ∨ (c * d = 36 ∧ a * b = 40)) →
    a + b + c + d = 26 :=
by
  sorry

end NUMINAMATH_CALUDE_nephews_ages_sum_l2798_279871


namespace NUMINAMATH_CALUDE_pool_length_is_ten_l2798_279899

/-- Proves that the length of a rectangular pool is 10 feet given its width, depth, and volume. -/
theorem pool_length_is_ten (width : ℝ) (depth : ℝ) (volume : ℝ) :
  width = 8 →
  depth = 6 →
  volume = 480 →
  volume = width * depth * (10 : ℝ) :=
by
  sorry

#check pool_length_is_ten

end NUMINAMATH_CALUDE_pool_length_is_ten_l2798_279899


namespace NUMINAMATH_CALUDE_painted_cubes_count_l2798_279870

/-- Represents a 3D shape composed of unit cubes -/
structure CubeShape where
  top_layer : Nat
  middle_layer : Nat
  bottom_layer : Nat
  unpainted_cubes : Nat

/-- Calculates the total number of cubes in the shape -/
def total_cubes (shape : CubeShape) : Nat :=
  shape.top_layer + shape.middle_layer + shape.bottom_layer

/-- Calculates the number of cubes with at least one face painted -/
def painted_cubes (shape : CubeShape) : Nat :=
  total_cubes shape - shape.unpainted_cubes

/-- Theorem stating the number of cubes with at least one face painted -/
theorem painted_cubes_count (shape : CubeShape) 
  (h1 : shape.top_layer = 9)
  (h2 : shape.middle_layer = 16)
  (h3 : shape.bottom_layer = 9)
  (h4 : shape.unpainted_cubes = 26) :
  painted_cubes shape = 8 := by
  sorry

end NUMINAMATH_CALUDE_painted_cubes_count_l2798_279870


namespace NUMINAMATH_CALUDE_emily_earnings_l2798_279847

-- Define the working hours for each day
def monday_hours : ℝ := 1
def wednesday_start : ℝ := 14.1667  -- 2:10 PM in decimal hours
def wednesday_end : ℝ := 16.8333    -- 4:50 PM in decimal hours
def thursday_hours : ℝ := 0.5
def saturday_hours : ℝ := 0.5

-- Define the hourly rate
def hourly_rate : ℝ := 4

-- Calculate total working hours
def total_hours : ℝ := monday_hours + (wednesday_end - wednesday_start) + thursday_hours + saturday_hours

-- Calculate total earnings
def total_earnings : ℝ := total_hours * hourly_rate

-- Theorem to prove
theorem emily_earnings : total_earnings = 18.68 := by
  sorry

end NUMINAMATH_CALUDE_emily_earnings_l2798_279847


namespace NUMINAMATH_CALUDE_equation_solution_l2798_279860

theorem equation_solution :
  ∃! y : ℚ, 7 * (2 * y - 3) + 4 = 3 * (5 - 9 * y) ∧ y = 32 / 41 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l2798_279860


namespace NUMINAMATH_CALUDE_male_average_score_l2798_279841

theorem male_average_score 
  (female_count : ℕ) 
  (male_count : ℕ) 
  (total_count : ℕ) 
  (female_avg : ℚ) 
  (total_avg : ℚ) 
  (h1 : female_count = 20)
  (h2 : male_count = 30)
  (h3 : total_count = female_count + male_count)
  (h4 : female_avg = 75)
  (h5 : total_avg = 72) :
  (total_count * total_avg - female_count * female_avg) / male_count = 70 := by
sorry

end NUMINAMATH_CALUDE_male_average_score_l2798_279841


namespace NUMINAMATH_CALUDE_one_root_quadratic_sum_l2798_279843

theorem one_root_quadratic_sum (a b : ℝ) : 
  (∃! x : ℝ, x^2 + a*x + b = 0) → 
  (a = 2*b - 3) → 
  (∃ b₁ b₂ : ℝ, (b = b₁ ∨ b = b₂) ∧ b₁ + b₂ = 4) := by
sorry

end NUMINAMATH_CALUDE_one_root_quadratic_sum_l2798_279843


namespace NUMINAMATH_CALUDE_cannot_find_fourth_vertex_l2798_279848

/-- Represents a point in 2D space -/
structure Point where
  x : ℤ
  y : ℤ

/-- Symmetric point operation -/
def symmetricPoint (a b : Point) : Point :=
  { x := 2 * b.x - a.x, y := 2 * b.y - a.y }

/-- Represents a square -/
structure Square where
  v1 : Point
  v2 : Point
  v3 : Point

/-- Checks if a point is a valid fourth vertex of a square -/
def isValidFourthVertex (s : Square) (p : Point) : Prop := sorry

theorem cannot_find_fourth_vertex (s : Square) :
  ¬ ∃ (p : Point), (∃ (a b : Point), p = symmetricPoint a b) ∧ isValidFourthVertex s p := by
  sorry

end NUMINAMATH_CALUDE_cannot_find_fourth_vertex_l2798_279848


namespace NUMINAMATH_CALUDE_star_calculation_l2798_279881

-- Define the ★ operation
def star (a b : ℚ) : ℚ := (a + b) / (a - b)

-- State the theorem
theorem star_calculation : star (star 3 5) 8 = -1/3 := by sorry

end NUMINAMATH_CALUDE_star_calculation_l2798_279881


namespace NUMINAMATH_CALUDE_unique_integer_divisible_by_24_with_specific_cube_root_l2798_279845

theorem unique_integer_divisible_by_24_with_specific_cube_root : 
  ∃! n : ℕ+, (∃ k : ℕ, n = 24 * k) ∧ 9 < (n : ℝ) ^ (1/3) ∧ (n : ℝ) ^ (1/3) < 9.1 :=
by sorry

end NUMINAMATH_CALUDE_unique_integer_divisible_by_24_with_specific_cube_root_l2798_279845


namespace NUMINAMATH_CALUDE_barbed_wire_cost_l2798_279887

theorem barbed_wire_cost (area : ℝ) (gate_width : ℝ) (num_gates : ℕ) (cost_per_meter : ℝ) : area = 3136 ∧ gate_width = 1 ∧ num_gates = 2 ∧ cost_per_meter = 1 → (4 * Real.sqrt area - num_gates * gate_width) * cost_per_meter = 222 := by
  sorry

end NUMINAMATH_CALUDE_barbed_wire_cost_l2798_279887


namespace NUMINAMATH_CALUDE_equal_perimeters_shapes_l2798_279862

theorem equal_perimeters_shapes (x y : ℝ) : 
  (4 * (x + 2) = 6 * x) ∧ (6 * x = 2 * Real.pi * y) → x = 4 ∧ y = 12 / Real.pi := by
  sorry

end NUMINAMATH_CALUDE_equal_perimeters_shapes_l2798_279862


namespace NUMINAMATH_CALUDE_dads_final_strawberry_weight_l2798_279896

/-- Given the initial total weight of strawberries collected by Marco and his dad,
    the additional weight Marco's dad found, and Marco's final weight of strawberries,
    prove that Marco's dad's final weight of strawberries is 46 pounds. -/
theorem dads_final_strawberry_weight
  (initial_total : ℕ)
  (dads_additional : ℕ)
  (marcos_final : ℕ)
  (h1 : initial_total = 22)
  (h2 : dads_additional = 30)
  (h3 : marcos_final = 36) :
  initial_total - (marcos_final - (initial_total - marcos_final)) + dads_additional = 46 :=
by sorry

end NUMINAMATH_CALUDE_dads_final_strawberry_weight_l2798_279896


namespace NUMINAMATH_CALUDE_cubic_equation_result_l2798_279823

theorem cubic_equation_result (x : ℝ) (h : x^3 + 2*x = 4) : x^7 + 32*x^2 = 64 := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_result_l2798_279823


namespace NUMINAMATH_CALUDE_quadratic_distinct_roots_range_l2798_279819

/-- The range of k for which the quadratic equation (k-1)x^2 + 2x - 2 = 0 has two distinct real roots -/
theorem quadratic_distinct_roots_range (k : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    (k - 1) * x₁^2 + 2 * x₁ - 2 = 0 ∧ 
    (k - 1) * x₂^2 + 2 * x₂ - 2 = 0) ↔ 
  (k > 1/2 ∧ k ≠ 1) :=
sorry

end NUMINAMATH_CALUDE_quadratic_distinct_roots_range_l2798_279819


namespace NUMINAMATH_CALUDE_cos_555_degrees_l2798_279812

theorem cos_555_degrees : Real.cos (555 * π / 180) = -(((Real.sqrt 6 + Real.sqrt 2) / 4)) := by
  sorry

end NUMINAMATH_CALUDE_cos_555_degrees_l2798_279812


namespace NUMINAMATH_CALUDE_minimal_solution_l2798_279800

def is_solution (A B C : ℕ) : Prop :=
  A ≠ B ∧ B ≠ C ∧ A ≠ C ∧
  (1 : ℚ) / A + (1 : ℚ) / B + (1 : ℚ) / C = (1 : ℚ) / 6 ∧
  6 ∣ A ∧ 6 ∣ B ∧ 6 ∣ C

theorem minimal_solution :
  ∀ A B C : ℕ, is_solution A B C →
  A + B + C ≥ 12 + 18 + 36 ∧
  is_solution 12 18 36 :=
sorry

end NUMINAMATH_CALUDE_minimal_solution_l2798_279800


namespace NUMINAMATH_CALUDE_intersection_locus_l2798_279851

/-- Represents a point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space -/
structure Line2D where
  slope : ℝ
  intercept : ℝ

/-- Represents a parabola in the form y² = x -/
def parabola (p : Point2D) : Prop :=
  p.y^2 = p.x

/-- Checks if a point lies on a line -/
def pointOnLine (p : Point2D) (l : Line2D) : Prop :=
  p.x = l.slope * p.y + l.intercept

/-- Checks if four points are concyclic (lie on the same circle) -/
def areConcyclic (p1 p2 p3 p4 : Point2D) : Prop :=
  ∃ (center : Point2D) (radius : ℝ),
    (center.x - p1.x)^2 + (center.y - p1.y)^2 = radius^2 ∧
    (center.x - p2.x)^2 + (center.y - p2.y)^2 = radius^2 ∧
    (center.x - p3.x)^2 + (center.y - p3.y)^2 = radius^2 ∧
    (center.x - p4.x)^2 + (center.y - p4.y)^2 = radius^2

theorem intersection_locus
  (a b : ℝ)
  (ha : 0 < a)
  (hab : a < b)
  (l m : Line2D)
  (hl : l.intercept = a)
  (hm : m.intercept = b)
  (p1 p2 p3 p4 : Point2D)
  (h_distinct : p1 ≠ p2 ∧ p1 ≠ p3 ∧ p1 ≠ p4 ∧ p2 ≠ p3 ∧ p2 ≠ p4 ∧ p3 ≠ p4)
  (h_on_parabola : parabola p1 ∧ parabola p2 ∧ parabola p3 ∧ parabola p4)
  (h_on_lines : (pointOnLine p1 l ∨ pointOnLine p1 m) ∧
                (pointOnLine p2 l ∨ pointOnLine p2 m) ∧
                (pointOnLine p3 l ∨ pointOnLine p3 m) ∧
                (pointOnLine p4 l ∨ pointOnLine p4 m))
  (h_concyclic : areConcyclic p1 p2 p3 p4)
  (P : Point2D)
  (h_intersection : pointOnLine P l ∧ pointOnLine P m) :
  P.x = (a + b) / 2 :=
by sorry

end NUMINAMATH_CALUDE_intersection_locus_l2798_279851


namespace NUMINAMATH_CALUDE_longest_segment_through_interior_point_l2798_279806

/-- A convex polygon in 2D space -/
structure ConvexPolygon where
  -- Define the properties of a convex polygon
  -- (This is a simplified representation)
  vertices : Set (ℝ × ℝ)
  is_convex : Bool

/-- A point in 2D space -/
def Point := ℝ × ℝ

/-- A direction in 2D space -/
def Direction := ℝ × ℝ

/-- Checks if a point is inside a convex polygon -/
def is_inside (K : ConvexPolygon) (P : Point) : Prop := sorry

/-- The length of the intersection of a line with a polygon -/
def intersection_length (K : ConvexPolygon) (P : Point) (d : Direction) : ℝ := sorry

/-- The theorem statement -/
theorem longest_segment_through_interior_point 
  (K : ConvexPolygon) (P : Point) (h : is_inside K P) :
  ∃ (d : Direction), 
    ∀ (Q : Point), is_inside K Q → 
      intersection_length K P d ≥ intersection_length K Q d := by sorry

end NUMINAMATH_CALUDE_longest_segment_through_interior_point_l2798_279806


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l2798_279801

-- Define the polynomial representation
def polynomial (a : ℝ) (a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ : ℝ) (x : ℝ) : ℝ :=
  a + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6 + a₇*x^7 + a₈*x^8 + a₉*x^9 + a₁₀*x^10

-- Define the given equation
def equation (a : ℝ) (a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ : ℝ) (x : ℝ) : Prop :=
  (1 - 2*x)^10 = polynomial a a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ x

-- Theorem to prove
theorem sum_of_coefficients 
  (a : ℝ) (a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ : ℝ) :
  (∀ x, equation a a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ x) →
  10*a₁ + 9*a₂ + 8*a₃ + 7*a₄ + 6*a₅ + 5*a₆ + 4*a₇ + 3*a₈ + 2*a₉ + a₁₀ = -20 :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l2798_279801


namespace NUMINAMATH_CALUDE_negative_cube_squared_l2798_279892

theorem negative_cube_squared (a : ℝ) : (-a^3)^2 = a^6 := by
  sorry

end NUMINAMATH_CALUDE_negative_cube_squared_l2798_279892


namespace NUMINAMATH_CALUDE_close_interval_for_f_and_g_l2798_279858

-- Define the functions f and g
def f (x : ℝ) : ℝ := x^2 - 3*x + 4
def g (x : ℝ) : ℝ := 2*x - 3

-- Define the property of being "close functions" on an interval
def are_close_functions (f g : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x, a ≤ x ∧ x ≤ b → |f x - g x| ≤ 1

-- State the theorem
theorem close_interval_for_f_and_g :
  are_close_functions f g 2 3 ∧
  ∀ a b, a < 2 ∨ b > 3 → ¬(are_close_functions f g a b) :=
sorry

end NUMINAMATH_CALUDE_close_interval_for_f_and_g_l2798_279858


namespace NUMINAMATH_CALUDE_hannah_total_spent_l2798_279877

def sweatshirt_count : ℕ := 3
def tshirt_count : ℕ := 2
def sweatshirt_price : ℕ := 15
def tshirt_price : ℕ := 10

theorem hannah_total_spent :
  sweatshirt_count * sweatshirt_price + tshirt_count * tshirt_price = 65 :=
by sorry

end NUMINAMATH_CALUDE_hannah_total_spent_l2798_279877


namespace NUMINAMATH_CALUDE_zero_of_f_l2798_279880

def f (x : ℝ) := 2 * x - 3

theorem zero_of_f :
  ∃ x : ℝ, f x = 0 ∧ x = 3/2 :=
sorry

end NUMINAMATH_CALUDE_zero_of_f_l2798_279880


namespace NUMINAMATH_CALUDE_y_gets_0_45_per_x_rupee_l2798_279805

/-- Represents the distribution of money among three parties -/
structure MoneyDistribution where
  x : ℝ  -- amount x gets
  y : ℝ  -- amount y gets
  z : ℝ  -- amount z gets
  a : ℝ  -- amount y gets for each rupee x gets

/-- Conditions of the money distribution problem -/
def valid_distribution (d : MoneyDistribution) : Prop :=
  d.z = 0.5 * d.x ∧  -- z gets 50 paisa for each rupee x gets
  d.y = 27 ∧  -- y's share is 27 rupees
  d.x + d.y + d.z = 117 ∧  -- total amount is 117 rupees
  d.y = d.a * d.x  -- relationship between y's share and x's share

/-- Theorem stating that under the given conditions, y gets 0.45 rupees for each rupee x gets -/
theorem y_gets_0_45_per_x_rupee (d : MoneyDistribution) :
  valid_distribution d → d.a = 0.45 := by
  sorry


end NUMINAMATH_CALUDE_y_gets_0_45_per_x_rupee_l2798_279805


namespace NUMINAMATH_CALUDE_baguettes_left_l2798_279855

/-- The number of batches of baguettes made per day -/
def batches_per_day : ℕ := 3

/-- The number of baguettes in each batch -/
def baguettes_per_batch : ℕ := 48

/-- The number of baguettes sold after the first batch -/
def sold_after_first : ℕ := 37

/-- The number of baguettes sold after the second batch -/
def sold_after_second : ℕ := 52

/-- The number of baguettes sold after the third batch -/
def sold_after_third : ℕ := 49

/-- Theorem stating that the number of baguettes left is 6 -/
theorem baguettes_left : 
  batches_per_day * baguettes_per_batch - (sold_after_first + sold_after_second + sold_after_third) = 6 := by
  sorry

end NUMINAMATH_CALUDE_baguettes_left_l2798_279855


namespace NUMINAMATH_CALUDE_complement_of_beta_l2798_279816

/-- Given two angles α and β that are complementary and α > β, 
    the complement of β is (α - β)/2 -/
theorem complement_of_beta (α β : ℝ) 
  (h1 : α + β = 90) -- α and β are complementary
  (h2 : α > β) : 
  90 - β = (α - β) / 2 := by
  sorry

end NUMINAMATH_CALUDE_complement_of_beta_l2798_279816


namespace NUMINAMATH_CALUDE_race_time_proof_l2798_279827

-- Define the race parameters
def race_distance : ℝ := 120
def distance_difference : ℝ := 72
def time_difference : ℝ := 10

-- Define the theorem
theorem race_time_proof :
  ∀ (v_a v_b t_a : ℝ),
  v_a > 0 → v_b > 0 → t_a > 0 →
  v_a = race_distance / t_a →
  v_b = (race_distance - distance_difference) / t_a →
  v_b = distance_difference / (t_a + time_difference) →
  t_a = 20 := by
sorry


end NUMINAMATH_CALUDE_race_time_proof_l2798_279827


namespace NUMINAMATH_CALUDE_arrangements_count_l2798_279808

/-- Represents the number of liberal arts classes -/
def liberal_arts_classes : ℕ := 2

/-- Represents the number of science classes -/
def science_classes : ℕ := 3

/-- Represents the total number of classes -/
def total_classes : ℕ := liberal_arts_classes + science_classes

/-- Function to calculate the number of arrangements -/
def arrangements : ℕ :=
  (science_classes.choose liberal_arts_classes) *
  (liberal_arts_classes.factorial) *
  (science_classes - liberal_arts_classes) *
  (liberal_arts_classes.factorial)

/-- Theorem stating that the number of arrangements is 24 -/
theorem arrangements_count : arrangements = 24 := by sorry

end NUMINAMATH_CALUDE_arrangements_count_l2798_279808


namespace NUMINAMATH_CALUDE_quadratic_root_shift_l2798_279828

theorem quadratic_root_shift (a b c t : ℤ) (ha : a ≠ 0) :
  (a * t^2 + b * t + c = 0) →
  ∃ (p q r : ℤ), p ≠ 0 ∧ p * (t + 2)^2 + q * (t + 2) + r = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_shift_l2798_279828


namespace NUMINAMATH_CALUDE_ellipse_k_range_l2798_279898

-- Define the equation of the ellipse
def ellipse_equation (x y k : ℝ) : Prop :=
  x^2 / (k - 4) + y^2 / (10 - k) = 1

-- Define the property of having foci on the x-axis
def foci_on_x_axis (k : ℝ) : Prop :=
  k - 4 > 0 ∧ 10 - k > 0 ∧ k - 4 > 10 - k

-- Theorem statement
theorem ellipse_k_range :
  ∀ k : ℝ, (∃ x y : ℝ, ellipse_equation x y k) ∧ foci_on_x_axis k ↔ 7 < k ∧ k < 10 :=
sorry

end NUMINAMATH_CALUDE_ellipse_k_range_l2798_279898


namespace NUMINAMATH_CALUDE_certain_number_multiplication_l2798_279863

theorem certain_number_multiplication : ∃ x : ℤ, (x - 7 = 9) ∧ (x * 3 = 48) := by
  sorry

end NUMINAMATH_CALUDE_certain_number_multiplication_l2798_279863
