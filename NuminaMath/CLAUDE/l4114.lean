import Mathlib

namespace NUMINAMATH_CALUDE_inequality_solution_set_l4114_411479

theorem inequality_solution_set (x : ℝ) : 3 * x - 2 > x ↔ x > 1 := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l4114_411479


namespace NUMINAMATH_CALUDE_fixed_point_parabola_l4114_411493

theorem fixed_point_parabola :
  ∀ (t : ℝ), 5 = 5 * (-1)^2 + 2 * t * (-1) - 5 * t := by sorry

end NUMINAMATH_CALUDE_fixed_point_parabola_l4114_411493


namespace NUMINAMATH_CALUDE_f_of_f_has_four_roots_l4114_411416

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 3*x + 2

-- State the theorem
theorem f_of_f_has_four_roots :
  ∃ (a b c d : ℝ), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
  (∀ x : ℝ, f (f x) = 0 ↔ x = a ∨ x = b ∨ x = c ∨ x = d) :=
sorry

end NUMINAMATH_CALUDE_f_of_f_has_four_roots_l4114_411416


namespace NUMINAMATH_CALUDE_certain_number_problem_l4114_411458

theorem certain_number_problem : ∃ x : ℕ, 
  220025 = (x + 445) * (2 * (x - 445)) + 25 ∧ 
  x = 555 := by
sorry

end NUMINAMATH_CALUDE_certain_number_problem_l4114_411458


namespace NUMINAMATH_CALUDE_work_time_B_l4114_411496

theorem work_time_B (time_A time_BC time_AC : ℝ) (h1 : time_A = 4) (h2 : time_BC = 3) (h3 : time_AC = 2) : 
  (1 / time_A + 1 / time_BC - 1 / time_AC)⁻¹ = 12 := by
sorry

end NUMINAMATH_CALUDE_work_time_B_l4114_411496


namespace NUMINAMATH_CALUDE_arithmetic_sequence_n_value_l4114_411462

/-- An arithmetic sequence {a_n} with given conditions -/
def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  a 1 = 5 ∧ a 5 = -3 ∧ ∃ n : ℕ, a n = -27

/-- The common difference of the arithmetic sequence -/
def common_difference (a : ℕ → ℤ) : ℤ := (a 5 - a 1) / 4

/-- The theorem stating that n = 17 for the given arithmetic sequence -/
theorem arithmetic_sequence_n_value (a : ℕ → ℤ) (h : arithmetic_sequence a) :
  ∃ n : ℕ, n = 17 ∧ a n = -27 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_n_value_l4114_411462


namespace NUMINAMATH_CALUDE_ann_has_36_blocks_l4114_411431

/-- The number of blocks Ann has at the end, given her initial blocks, 
    blocks found, and blocks lost. -/
def anns_final_blocks (initial : ℕ) (found : ℕ) (lost : ℕ) : ℕ :=
  initial + found - lost

/-- Theorem stating that Ann ends up with 36 blocks -/
theorem ann_has_36_blocks : anns_final_blocks 9 44 17 = 36 := by
  sorry

end NUMINAMATH_CALUDE_ann_has_36_blocks_l4114_411431


namespace NUMINAMATH_CALUDE_elenas_bread_recipe_l4114_411412

/-- Given Elena's bread recipe, prove the amount of butter needed for the original recipe -/
theorem elenas_bread_recipe (original_flour : ℝ) (scale_factor : ℝ) (new_butter : ℝ) (new_flour : ℝ) :
  original_flour = 14 →
  scale_factor = 4 →
  new_butter = 12 →
  new_flour = 56 →
  (new_butter / new_flour) * original_flour = 3 := by
  sorry

end NUMINAMATH_CALUDE_elenas_bread_recipe_l4114_411412


namespace NUMINAMATH_CALUDE_oasis_water_consumption_l4114_411454

theorem oasis_water_consumption (traveler_ounces camel_multiplier ounces_per_gallon : ℕ) 
  (h1 : traveler_ounces = 32)
  (h2 : camel_multiplier = 7)
  (h3 : ounces_per_gallon = 128) :
  (traveler_ounces + camel_multiplier * traveler_ounces) / ounces_per_gallon = 2 := by
  sorry

#check oasis_water_consumption

end NUMINAMATH_CALUDE_oasis_water_consumption_l4114_411454


namespace NUMINAMATH_CALUDE_trains_clearing_time_l4114_411488

/-- Calculates the time for two trains to clear each other -/
theorem trains_clearing_time (length1 length2 speed1 speed2 : ℝ) : 
  length1 = 160 ∧ 
  length2 = 320 ∧ 
  speed1 = 42 ∧ 
  speed2 = 30 → 
  (length1 + length2) / ((speed1 + speed2) * (1000 / 3600)) = 24 := by
  sorry

#check trains_clearing_time

end NUMINAMATH_CALUDE_trains_clearing_time_l4114_411488


namespace NUMINAMATH_CALUDE_intersection_union_equality_l4114_411411

def M : Set Nat := {0, 1, 2, 4, 5, 7}
def N : Set Nat := {1, 4, 6, 8, 9}
def P : Set Nat := {4, 7, 9}

theorem intersection_union_equality : (M ∩ N) ∪ (M ∩ P) = {1, 4, 7} := by
  sorry

end NUMINAMATH_CALUDE_intersection_union_equality_l4114_411411


namespace NUMINAMATH_CALUDE_quadratic_minimum_l4114_411487

/-- The function f(x) = x^2 + 6x + 13 has a minimum value of 4 -/
theorem quadratic_minimum (x : ℝ) : ∀ y : ℝ, x^2 + 6*x + 13 ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_minimum_l4114_411487


namespace NUMINAMATH_CALUDE_d_investment_is_250_l4114_411419

/-- Represents the business investment scenario -/
structure BusinessInvestment where
  c_investment : ℕ
  total_profit : ℕ
  d_profit_share : ℕ

/-- Calculates D's investment based on the given conditions -/
def calculate_d_investment (b : BusinessInvestment) : ℕ :=
  b.c_investment * b.d_profit_share / (b.total_profit - b.d_profit_share)

/-- Theorem stating that D's investment is 250 given the conditions -/
theorem d_investment_is_250 (b : BusinessInvestment) 
  (h1 : b.c_investment = 1000)
  (h2 : b.total_profit = 500)
  (h3 : b.d_profit_share = 100) : 
  calculate_d_investment b = 250 := by
  sorry

#eval calculate_d_investment { c_investment := 1000, total_profit := 500, d_profit_share := 100 }

end NUMINAMATH_CALUDE_d_investment_is_250_l4114_411419


namespace NUMINAMATH_CALUDE_distance_difference_l4114_411484

theorem distance_difference (john_distance nina_distance : ℝ) 
  (h1 : john_distance = 0.7)
  (h2 : nina_distance = 0.4) :
  john_distance - nina_distance = 0.3 := by
sorry

end NUMINAMATH_CALUDE_distance_difference_l4114_411484


namespace NUMINAMATH_CALUDE_brick_width_calculation_l4114_411429

/-- The width of a brick in centimeters -/
def brick_width : ℝ := 11.25

/-- The length of a brick in centimeters -/
def brick_length : ℝ := 25

/-- The height of a brick in centimeters -/
def brick_height : ℝ := 6

/-- The length of the wall in centimeters -/
def wall_length : ℝ := 900

/-- The width of the wall in centimeters -/
def wall_width : ℝ := 600

/-- The height of the wall in centimeters -/
def wall_height : ℝ := 22.5

/-- The number of bricks needed -/
def num_bricks : ℕ := 7200

/-- The volume of the wall in cubic centimeters -/
def wall_volume : ℝ := wall_length * wall_width * wall_height

/-- The volume of a single brick in cubic centimeters -/
def brick_volume : ℝ := brick_length * brick_width * brick_height

theorem brick_width_calculation :
  brick_width = (wall_volume / (num_bricks : ℝ)) / (brick_length * brick_height) :=
by sorry

end NUMINAMATH_CALUDE_brick_width_calculation_l4114_411429


namespace NUMINAMATH_CALUDE_equation_solution_l4114_411405

theorem equation_solution (x : ℝ) : 
  (1 / (Real.sqrt x + Real.sqrt (x - 2)) + 1 / (Real.sqrt (x + 2) + Real.sqrt x) = 1 / 4) → 
  x = 257 / 16 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l4114_411405


namespace NUMINAMATH_CALUDE_sqrt_x_div_sqrt_y_l4114_411427

theorem sqrt_x_div_sqrt_y (x y : ℝ) (h : (1/3)^2 + (1/4)^2 = ((1/5)^2 + (1/6)^2) * (25*x)/(73*y)) :
  Real.sqrt x / Real.sqrt y = 5/2 := by
sorry

end NUMINAMATH_CALUDE_sqrt_x_div_sqrt_y_l4114_411427


namespace NUMINAMATH_CALUDE_single_point_condition_l4114_411400

/-- The equation represents a single point if and only if d equals 125/4 -/
theorem single_point_condition (d : ℝ) : 
  (∃! p : ℝ × ℝ, 3 * p.1^2 + 2 * p.2^2 + 9 * p.1 - 14 * p.2 + d = 0) ↔ 
  d = 125 / 4 := by
  sorry

end NUMINAMATH_CALUDE_single_point_condition_l4114_411400


namespace NUMINAMATH_CALUDE_expand_quadratic_l4114_411432

theorem expand_quadratic (a : ℝ) : a * (a - 3) = a^2 - 3*a := by sorry

end NUMINAMATH_CALUDE_expand_quadratic_l4114_411432


namespace NUMINAMATH_CALUDE_jump_data_mode_l4114_411422

def jump_data : List Nat := [160, 163, 160, 157, 160]

def mode (l : List Nat) : Nat :=
  l.foldl (fun acc x => if l.count x > l.count acc then x else acc) 0

theorem jump_data_mode :
  mode jump_data = 160 := by
  sorry

end NUMINAMATH_CALUDE_jump_data_mode_l4114_411422


namespace NUMINAMATH_CALUDE_simplified_expression_terms_l4114_411451

-- Define the exponent
def n : ℕ := 2008

-- Define the function to count terms
def countTerms (n : ℕ) : ℕ :=
  (n / 2 + 1) * (n + 1)

-- Theorem statement
theorem simplified_expression_terms :
  countTerms n = 2018045 :=
sorry

end NUMINAMATH_CALUDE_simplified_expression_terms_l4114_411451


namespace NUMINAMATH_CALUDE_log_expression_equals_three_l4114_411465

theorem log_expression_equals_three :
  (Real.log 243 / Real.log 3) / (Real.log 27 / Real.log 3) -
  (Real.log 729 / Real.log 3) / (Real.log 9 / Real.log 3) = 3 := by
  sorry

end NUMINAMATH_CALUDE_log_expression_equals_three_l4114_411465


namespace NUMINAMATH_CALUDE_inverse_mod_million_l4114_411418

theorem inverse_mod_million (C D M : Nat) : 
  C = 123456 → 
  D = 142857 → 
  M = 814815 → 
  (C * D * M) % 1000000 = 1 :=
by sorry

end NUMINAMATH_CALUDE_inverse_mod_million_l4114_411418


namespace NUMINAMATH_CALUDE_m_minus_n_equals_eighteen_l4114_411480

theorem m_minus_n_equals_eighteen :
  ∀ m n : ℤ,
  (∀ k : ℤ, k < 0 → k ≤ -m) →  -- m's opposite is the largest negative integer
  (-n = 17) →                  -- n's opposite is 17
  m - n = 18 :=
by
  sorry

end NUMINAMATH_CALUDE_m_minus_n_equals_eighteen_l4114_411480


namespace NUMINAMATH_CALUDE_right_triangle_pythagorean_l4114_411459

theorem right_triangle_pythagorean (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 → -- Ensuring positive lengths
  (a^2 + b^2 = c^2) → -- Pythagorean theorem
  ((a = 12 ∧ b = 5) → c = 13) ∧ -- Part 1
  ((c = 10 ∧ b = 9) → a = Real.sqrt 19) -- Part 2
  := by sorry

end NUMINAMATH_CALUDE_right_triangle_pythagorean_l4114_411459


namespace NUMINAMATH_CALUDE_sin_less_than_x_in_interval_exp_x_plus_one_greater_than_neg_e_squared_l4114_411477

-- Option A
theorem sin_less_than_x_in_interval (x : ℝ) (h : x ∈ Set.Ioo 0 Real.pi) : x > Real.sin x := by
  sorry

-- Option C
theorem exp_x_plus_one_greater_than_neg_e_squared (x : ℝ) : (x + 1) * Real.exp x > -(1 / Real.exp 2) := by
  sorry

end NUMINAMATH_CALUDE_sin_less_than_x_in_interval_exp_x_plus_one_greater_than_neg_e_squared_l4114_411477


namespace NUMINAMATH_CALUDE_mary_income_proof_l4114_411437

/-- Calculates Mary's total income for a week --/
def maryIncome (maxHours regularhourlyRate overtimeRate1 overtimeRate2 bonus dues : ℝ) : ℝ :=
  let regularPay := regularhourlyRate * 20
  let overtimePay1 := overtimeRate1 * 20
  let overtimePay2 := overtimeRate2 * 20
  regularPay + overtimePay1 + overtimePay2 + bonus - dues

/-- Proves that Mary's total income is $650 given the specified conditions --/
theorem mary_income_proof :
  let maxHours : ℝ := 60
  let regularRate : ℝ := 8
  let overtimeRate1 : ℝ := regularRate * 1.25
  let overtimeRate2 : ℝ := regularRate * 1.5
  let bonus : ℝ := 100
  let dues : ℝ := 50
  maryIncome maxHours regularRate overtimeRate1 overtimeRate2 bonus dues = 650 := by
  sorry

#eval maryIncome 60 8 10 12 100 50

end NUMINAMATH_CALUDE_mary_income_proof_l4114_411437


namespace NUMINAMATH_CALUDE_sum_difference_of_squares_l4114_411428

theorem sum_difference_of_squares (n : ℤ) : ∃ a b c d : ℤ, n = a^2 + b^2 - c^2 - d^2 := by
  sorry

end NUMINAMATH_CALUDE_sum_difference_of_squares_l4114_411428


namespace NUMINAMATH_CALUDE_triangle_inequality_incenter_l4114_411404

/-- Given a triangle ABC with sides a, b, c and a point P inside the triangle with distances
    r₁, r₂, r₃ to the sides respectively, prove that (a/r₁ + b/r₂ + c/r₃) ≥ (a + b + c)²/(2S),
    where S is the area of triangle ABC, and equality holds iff P is the incenter. -/
theorem triangle_inequality_incenter (a b c r₁ r₂ r₃ S : ℝ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0 ∧ r₁ > 0 ∧ r₂ > 0 ∧ r₃ > 0 ∧ S > 0)
  (h_area : a * r₁ + b * r₂ + c * r₃ = 2 * S) :
  a / r₁ + b / r₂ + c / r₃ ≥ (a + b + c)^2 / (2 * S) ∧
  (a / r₁ + b / r₂ + c / r₃ = (a + b + c)^2 / (2 * S) ↔ r₁ = r₂ ∧ r₂ = r₃) :=
by sorry

end NUMINAMATH_CALUDE_triangle_inequality_incenter_l4114_411404


namespace NUMINAMATH_CALUDE_expression_simplification_l4114_411434

theorem expression_simplification :
  (2^1002 + 5^1003)^2 - (2^1002 - 5^1003)^2 = 20 * 10^1002 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l4114_411434


namespace NUMINAMATH_CALUDE_dan_picked_nine_apples_l4114_411435

/-- The number of apples Benny picked -/
def benny_apples : ℕ := 2

/-- The total number of apples picked by Benny and Dan -/
def total_apples : ℕ := 11

/-- The number of apples Dan picked -/
def dan_apples : ℕ := total_apples - benny_apples

theorem dan_picked_nine_apples : dan_apples = 9 := by
  sorry

end NUMINAMATH_CALUDE_dan_picked_nine_apples_l4114_411435


namespace NUMINAMATH_CALUDE_room_length_l4114_411421

/-- The length of a rectangular room with given width and area -/
theorem room_length (width : ℝ) (area : ℝ) (h1 : width = 20) (h2 : area = 80) :
  area / width = 4 := by sorry

end NUMINAMATH_CALUDE_room_length_l4114_411421


namespace NUMINAMATH_CALUDE_action_figures_added_l4114_411481

theorem action_figures_added (initial : ℕ) (removed : ℕ) (final : ℕ) : 
  initial = 15 → removed = 7 → final = 10 → initial - removed + (final - (initial - removed)) = 2 := by
sorry

end NUMINAMATH_CALUDE_action_figures_added_l4114_411481


namespace NUMINAMATH_CALUDE_sector_area_l4114_411420

theorem sector_area (arc_length : Real) (central_angle : Real) (area : Real) :
  arc_length = 4 * Real.pi →
  central_angle = Real.pi / 3 →
  area = 24 * Real.pi :=
by
  sorry

end NUMINAMATH_CALUDE_sector_area_l4114_411420


namespace NUMINAMATH_CALUDE_problem_solution_l4114_411453

theorem problem_solution (x : ℝ) (h : x^2 - 3*x - 1 = 0) : -3*x^2 + 9*x + 4 = 1 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l4114_411453


namespace NUMINAMATH_CALUDE_second_trial_point_theorem_l4114_411475

/-- Represents the fractional method for optimization experiments -/
structure FractionalMethod where
  range_start : ℝ
  range_end : ℝ
  rounds : ℕ

/-- Calculates the possible second trial points for the fractional method -/
def second_trial_points (fm : FractionalMethod) : Set ℝ :=
  let interval_length := fm.range_end - fm.range_start
  let num_divisions := 2^fm.rounds
  let step := interval_length / num_divisions
  {fm.range_start + 3 * step, fm.range_end - 3 * step}

/-- Theorem stating that for the given experimental setup, 
    the second trial point is either 40 or 60 -/
theorem second_trial_point_theorem (fm : FractionalMethod) 
  (h1 : fm.range_start = 10) 
  (h2 : fm.range_end = 90) 
  (h3 : fm.rounds = 4) : 
  second_trial_points fm = {40, 60} := by
  sorry

end NUMINAMATH_CALUDE_second_trial_point_theorem_l4114_411475


namespace NUMINAMATH_CALUDE_total_cookies_calculation_l4114_411444

/-- The number of cookies Kristy baked -/
def total_cookies : ℕ := sorry

/-- The number of cookies Kristy ate -/
def kristy_ate : ℕ := 3

/-- The number of cookies Kristy's brother took -/
def brother_took : ℕ := 2

/-- The number of cookies the first friend took -/
def first_friend_took : ℕ := 4

/-- The number of cookies the second friend took (net) -/
def second_friend_took : ℕ := 4

/-- The number of cookies the third friend took -/
def third_friend_took : ℕ := 8

/-- The number of cookies the fourth friend took -/
def fourth_friend_took : ℕ := 3

/-- The number of cookies the fifth friend took -/
def fifth_friend_took : ℕ := 7

/-- The number of cookies left -/
def cookies_left : ℕ := 5

/-- Theorem stating that the total number of cookies is equal to the sum of all distributed cookies and the remaining cookies -/
theorem total_cookies_calculation :
  total_cookies = kristy_ate + brother_took + first_friend_took + second_friend_took +
                  third_friend_took + fourth_friend_took + fifth_friend_took + cookies_left :=
by sorry

end NUMINAMATH_CALUDE_total_cookies_calculation_l4114_411444


namespace NUMINAMATH_CALUDE_impossible_arrangement_l4114_411423

/-- A table is a function from pairs of indices to natural numbers -/
def Table := Fin 10 → Fin 10 → ℕ

/-- Predicate to check if two cells are adjacent in the table -/
def adjacent (i j k l : Fin 10) : Prop :=
  (i = k ∧ j.val + 1 = l.val) ∨ (i = k ∧ l.val + 1 = j.val) ∨
  (j = l ∧ i.val + 1 = k.val) ∧ (j = l ∧ k.val + 1 = i.val)

/-- Predicate to check if a quadratic equation has two integer roots -/
def has_two_int_roots (a b : ℕ) : Prop :=
  ∃ x y : ℤ, x ≠ y ∧ x^2 - a*x + b = 0 ∧ y^2 - a*y + b = 0

theorem impossible_arrangement : ¬∃ (t : Table),
  (∀ i j : Fin 10, 51 ≤ t i j ∧ t i j ≤ 150) ∧
  (∀ i j k l : Fin 10, adjacent i j k l →
    has_two_int_roots (t i j) (t k l) ∨ has_two_int_roots (t k l) (t i j)) :=
sorry

end NUMINAMATH_CALUDE_impossible_arrangement_l4114_411423


namespace NUMINAMATH_CALUDE_arccos_one_over_sqrt_two_l4114_411443

theorem arccos_one_over_sqrt_two (π : Real) : Real.arccos (1 / Real.sqrt 2) = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_arccos_one_over_sqrt_two_l4114_411443


namespace NUMINAMATH_CALUDE_inscribed_triangle_inequality_l4114_411442

/-- A triangle PQR inscribed in a semicircle with diameter PQ and R on the semicircle -/
structure InscribedTriangle where
  /-- The radius of the semicircle -/
  r : ℝ
  /-- Point P -/
  P : ℝ × ℝ
  /-- Point Q -/
  Q : ℝ × ℝ
  /-- Point R -/
  R : ℝ × ℝ
  /-- PQ is the diameter of the semicircle -/
  diameter : dist P Q = 2 * r
  /-- R is on the semicircle -/
  on_semicircle : dist P R = r ∨ dist Q R = r

/-- The sum of distances PR and QR -/
def t (triangle : InscribedTriangle) : ℝ :=
  dist triangle.P triangle.R + dist triangle.Q triangle.R

/-- Theorem: For all inscribed triangles, t^2 ≤ 8r^2 -/
theorem inscribed_triangle_inequality (triangle : InscribedTriangle) :
  (t triangle)^2 ≤ 8 * triangle.r^2 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_triangle_inequality_l4114_411442


namespace NUMINAMATH_CALUDE_overtime_hours_is_eight_l4114_411492

/-- Represents the payment structure and work hours for a worker --/
structure WorkerPayment where
  ordinary_rate : ℚ  -- Rate for ordinary hours in cents
  overtime_rate : ℚ  -- Rate for overtime hours in cents
  total_hours : ℕ    -- Total hours worked
  total_pay : ℚ      -- Total pay in cents

/-- Calculates the number of overtime hours --/
def calculate_overtime_hours (w : WorkerPayment) : ℚ :=
  (w.total_pay - w.ordinary_rate * w.total_hours) / (w.overtime_rate - w.ordinary_rate)

/-- Theorem stating that under given conditions, the overtime hours are 8 --/
theorem overtime_hours_is_eight :
  let w := WorkerPayment.mk 60 90 50 3240
  calculate_overtime_hours w = 8 := by sorry

end NUMINAMATH_CALUDE_overtime_hours_is_eight_l4114_411492


namespace NUMINAMATH_CALUDE_min_sum_squared_distances_l4114_411445

/-- Given points A, B, C, D, and E on a line in that order, with specified distances between them,
    this theorem states that the minimum sum of squared distances from these points to any point P
    on the same line is 66. -/
theorem min_sum_squared_distances (A B C D E P : ℝ) 
  (h_order : A < B ∧ B < C ∧ C < D ∧ D < E)
  (h_AB : B - A = 1)
  (h_BC : C - B = 2)
  (h_CD : D - C = 3)
  (h_DE : E - D = 4)
  (h_P : A ≤ P ∧ P ≤ E) :
  66 ≤ (P - A)^2 + (P - B)^2 + (P - C)^2 + (P - D)^2 + (P - E)^2 :=
sorry

end NUMINAMATH_CALUDE_min_sum_squared_distances_l4114_411445


namespace NUMINAMATH_CALUDE_line_point_order_l4114_411472

theorem line_point_order (b : ℝ) (y₁ y₂ y₃ : ℝ) : 
  (y₁ = 3 * (-2.3) + b) → 
  (y₂ = 3 * (-1.3) + b) → 
  (y₃ = 3 * 2.7 + b) → 
  y₁ < y₂ ∧ y₂ < y₃ :=
by sorry

end NUMINAMATH_CALUDE_line_point_order_l4114_411472


namespace NUMINAMATH_CALUDE_problem_1_l4114_411466

theorem problem_1 : (-1)^3 + (1/7) * (2 - (-3)^2) = -2 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_l4114_411466


namespace NUMINAMATH_CALUDE_total_toys_is_56_l4114_411482

/-- Given the number of toys Mike has, calculate the total number of toys for Annie, Mike, and Tom. -/
def totalToys (mikeToys : ℕ) : ℕ :=
  let annieToys := 3 * mikeToys
  let tomToys := annieToys + 2
  mikeToys + annieToys + tomToys

/-- Theorem stating that given Mike has 6 toys, the total number of toys for Annie, Mike, and Tom is 56. -/
theorem total_toys_is_56 : totalToys 6 = 56 := by
  sorry

#eval totalToys 6  -- This will evaluate to 56

end NUMINAMATH_CALUDE_total_toys_is_56_l4114_411482


namespace NUMINAMATH_CALUDE_unknown_number_value_l4114_411436

theorem unknown_number_value (x n : ℝ) : 
  x = 12 → 5 + n / x = 6 - 5 / x → n = 7 := by
  sorry

end NUMINAMATH_CALUDE_unknown_number_value_l4114_411436


namespace NUMINAMATH_CALUDE_constant_expression_l4114_411460

-- Define the logarithm with base √2
noncomputable def log_sqrt2 (x : ℝ) : ℝ := Real.log x / Real.log (Real.sqrt 2)

-- State the theorem
theorem constant_expression (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x > y) (h4 : x^2 + y^2 = 18*x*y) :
  log_sqrt2 (x - y) - (log_sqrt2 x + log_sqrt2 y) / 2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_constant_expression_l4114_411460


namespace NUMINAMATH_CALUDE_shelly_money_l4114_411408

/-- Calculates the total amount of money Shelly has given the number of $10 and $5 bills -/
def total_money (ten_dollar_bills : ℕ) (five_dollar_bills : ℕ) : ℕ :=
  10 * ten_dollar_bills + 5 * five_dollar_bills

/-- Proves that Shelly has $130 given the conditions -/
theorem shelly_money : 
  let ten_dollar_bills : ℕ := 10
  let five_dollar_bills : ℕ := ten_dollar_bills - 4
  total_money ten_dollar_bills five_dollar_bills = 130 := by
sorry

end NUMINAMATH_CALUDE_shelly_money_l4114_411408


namespace NUMINAMATH_CALUDE_train_crossing_time_l4114_411440

/-- Time taken for a train to cross a man walking in the same direction -/
theorem train_crossing_time (train_length : ℝ) (train_speed : ℝ) (man_speed : ℝ) : 
  train_length = 600 →
  train_speed = 56 * 1000 / 3600 →
  man_speed = 2 * 1000 / 3600 →
  (train_length / (train_speed - man_speed)) = 40 := by
  sorry

#eval Float.round ((600 : Float) / ((56 * 1000 / 3600) - (2 * 1000 / 3600)))

end NUMINAMATH_CALUDE_train_crossing_time_l4114_411440


namespace NUMINAMATH_CALUDE_no_finite_maximum_for_expression_l4114_411450

open Real

theorem no_finite_maximum_for_expression (x y z : ℝ) 
  (pos_x : 0 < x) (pos_y : 0 < y) (pos_z : 0 < z) 
  (sum_constraint : x + y + z = 9) : 
  ¬ ∃ (M : ℝ), ∀ (x y z : ℝ), 
    0 < x → 0 < y → 0 < z → x + y + z = 9 →
    (x^2 + 2*y^2)/(x + y) + (2*x^2 + z^2)/(x + z) + (y^2 + 2*z^2)/(y + z) ≤ M :=
sorry

end NUMINAMATH_CALUDE_no_finite_maximum_for_expression_l4114_411450


namespace NUMINAMATH_CALUDE_prob_more_heads_than_tails_is_correct_l4114_411485

/-- The probability of getting more heads than tails when flipping 10 coins -/
def prob_more_heads_than_tails : ℚ := 193 / 512

/-- The number of coins being flipped -/
def num_coins : ℕ := 10

/-- The total number of possible outcomes when flipping 10 coins -/
def total_outcomes : ℕ := 2^num_coins

/-- The probability of getting exactly 5 heads (and 5 tails) when flipping 10 coins -/
def prob_equal_heads_tails : ℚ := 63 / 256

theorem prob_more_heads_than_tails_is_correct :
  prob_more_heads_than_tails = (1 - prob_equal_heads_tails) / 2 :=
sorry

end NUMINAMATH_CALUDE_prob_more_heads_than_tails_is_correct_l4114_411485


namespace NUMINAMATH_CALUDE_max_value_of_five_numbers_l4114_411497

theorem max_value_of_five_numbers (a b c d e : ℕ) : 
  a < b ∧ b < c ∧ c < d ∧ d < e →  -- distinct and ordered
  (a + b + c + d + e) / 5 = 12 →  -- average is 12
  c = 17 →  -- median is 17
  e ≤ 24 :=  -- maximum possible value is 24
by sorry

end NUMINAMATH_CALUDE_max_value_of_five_numbers_l4114_411497


namespace NUMINAMATH_CALUDE_three_solutions_iff_a_gt_two_l4114_411401

/-- The equation x · |x-a| = 1 has exactly three distinct solutions if and only if a > 2 -/
theorem three_solutions_iff_a_gt_two (a : ℝ) :
  (∃ (x₁ x₂ x₃ : ℝ), x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧
    (∀ (x : ℝ), x * |x - a| = 1 ↔ x = x₁ ∨ x = x₂ ∨ x = x₃)) ↔
  a > 2 := by
  sorry

end NUMINAMATH_CALUDE_three_solutions_iff_a_gt_two_l4114_411401


namespace NUMINAMATH_CALUDE_line_intersects_plane_not_perpendicular_implies_not_parallel_l4114_411491

-- Define the necessary structures
structure Line3D where
  -- Add necessary fields for a 3D line

structure Plane3D where
  -- Add necessary fields for a 3D plane

-- Define the relationships
def intersects (l : Line3D) (α : Plane3D) : Prop :=
  sorry

def perpendicular (l : Line3D) (α : Plane3D) : Prop :=
  sorry

def plane_through_line (l : Line3D) : Plane3D :=
  sorry

def parallel_planes (p1 p2 : Plane3D) : Prop :=
  sorry

-- State the theorem
theorem line_intersects_plane_not_perpendicular_implies_not_parallel 
  (l : Line3D) (α : Plane3D) :
  intersects l α ∧ ¬perpendicular l α →
  ∀ p : Plane3D, p = plane_through_line l → ¬parallel_planes p α :=
sorry

end NUMINAMATH_CALUDE_line_intersects_plane_not_perpendicular_implies_not_parallel_l4114_411491


namespace NUMINAMATH_CALUDE_pinterest_group_pins_l4114_411433

theorem pinterest_group_pins 
  (num_members : ℕ) 
  (initial_pins : ℕ) 
  (daily_contribution : ℕ) 
  (weekly_deletion : ℕ) 
  (days_in_month : ℕ) 
  (h1 : num_members = 20)
  (h2 : initial_pins = 1000)
  (h3 : daily_contribution = 10)
  (h4 : weekly_deletion = 5)
  (h5 : days_in_month = 30) :
  let total_new_pins := num_members * daily_contribution * days_in_month
  let total_deleted_pins := num_members * weekly_deletion * (days_in_month / 7)
  initial_pins + total_new_pins - total_deleted_pins = 6600 := by
  sorry

end NUMINAMATH_CALUDE_pinterest_group_pins_l4114_411433


namespace NUMINAMATH_CALUDE_speaking_orders_count_l4114_411439

/-- The number of contestants -/
def n : ℕ := 6

/-- The number of positions where contestant A can speak -/
def a_positions : ℕ := n - 2

/-- The number of permutations for the remaining contestants -/
def remaining_permutations : ℕ := Nat.factorial (n - 1)

/-- The total number of different speaking orders -/
def total_orders : ℕ := a_positions * remaining_permutations

theorem speaking_orders_count : total_orders = 480 := by
  sorry

end NUMINAMATH_CALUDE_speaking_orders_count_l4114_411439


namespace NUMINAMATH_CALUDE_patrick_current_age_l4114_411474

/-- Patrick's age is half of Robert's age -/
def patrick_age_relation (patrick_age robert_age : ℕ) : Prop :=
  patrick_age = robert_age / 2

/-- Robert will be 30 years old in 2 years -/
def robert_future_age (robert_age : ℕ) : Prop :=
  robert_age + 2 = 30

/-- The theorem stating Patrick's current age -/
theorem patrick_current_age :
  ∃ (patrick_age robert_age : ℕ),
    patrick_age_relation patrick_age robert_age ∧
    robert_future_age robert_age ∧
    patrick_age = 14 := by
  sorry

end NUMINAMATH_CALUDE_patrick_current_age_l4114_411474


namespace NUMINAMATH_CALUDE_total_distance_swam_l4114_411415

/-- Represents the swimming styles -/
inductive SwimmingStyle
| Freestyle
| Butterfly

/-- Calculates the distance swam for a given style -/
def distance_swam (style : SwimmingStyle) (total_time : ℕ) : ℕ :=
  match style with
  | SwimmingStyle.Freestyle =>
    let cycle_time := 26  -- 20 minutes swimming + 6 minutes rest
    let cycles := total_time / cycle_time
    let distance_per_cycle := 500  -- 100 meters in 4 minutes, so 500 meters in 20 minutes
    cycles * distance_per_cycle
  | SwimmingStyle.Butterfly =>
    let cycle_time := 35  -- 30 minutes swimming + 5 minutes rest
    let cycles := total_time / cycle_time
    let distance_per_cycle := 429  -- 100 meters in 7 minutes, so approximately 429 meters in 30 minutes
    cycles * distance_per_cycle

theorem total_distance_swam :
  let freestyle_time := 90  -- 1 hour and 30 minutes in minutes
  let butterfly_time := 90  -- 1 hour and 30 minutes in minutes
  let freestyle_distance := distance_swam SwimmingStyle.Freestyle freestyle_time
  let butterfly_distance := distance_swam SwimmingStyle.Butterfly butterfly_time
  freestyle_distance + butterfly_distance = 2358 := by
  sorry


end NUMINAMATH_CALUDE_total_distance_swam_l4114_411415


namespace NUMINAMATH_CALUDE_candy_redistribution_l4114_411452

/-- Represents the distribution of candies in boxes -/
def CandyDistribution := List Nat

/-- An operation on the candy distribution -/
def redistribute (dist : CandyDistribution) (i j : Nat) : CandyDistribution :=
  sorry

/-- Checks if a distribution is valid (total candies = n^2) -/
def isValidDistribution (n : Nat) (dist : CandyDistribution) : Prop :=
  sorry

/-- Checks if a distribution is the goal distribution (n candies in each box) -/
def isGoalDistribution (n : Nat) (dist : CandyDistribution) : Prop :=
  sorry

/-- Checks if a number is a power of 2 -/
def isPowerOfTwo (n : Nat) : Prop :=
  sorry

theorem candy_redistribution (n : Nat) :
  (n > 2) →
  (∀ (init : CandyDistribution), isValidDistribution n init →
    ∃ (final : CandyDistribution), isGoalDistribution n final ∧
      ∃ (ops : List (Nat × Nat)), final = ops.foldl (fun d (i, j) => redistribute d i j) init) ↔
  isPowerOfTwo n :=
sorry

end NUMINAMATH_CALUDE_candy_redistribution_l4114_411452


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l4114_411483

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_common_difference 
  (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a)
  (h_incr : ∀ n : ℕ, a n < a (n + 1))
  (h_sum_squares : a 1 ^ 2 + a 10 ^ 2 = 101)
  (h_sum_mid : a 5 + a 6 = 11) :
  ∃ d : ℝ, d = 1 ∧ ∀ n : ℕ, a (n + 1) = a n + d :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l4114_411483


namespace NUMINAMATH_CALUDE_simplify_expression_l4114_411469

theorem simplify_expression : 5 * (14 / 3) * (9 / -42) = -5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l4114_411469


namespace NUMINAMATH_CALUDE_rectangle_length_calculation_l4114_411499

/-- Represents a rectangular piece of land -/
structure Rectangle where
  width : ℝ
  length : ℝ

/-- The area of a rectangle -/
def area (r : Rectangle) : ℝ := r.width * r.length

/-- Theorem: For a rectangle with area 215.6 m² and width 14 m, the length is 15.4 m -/
theorem rectangle_length_calculation (r : Rectangle) 
  (h_area : area r = 215.6) 
  (h_width : r.width = 14) : 
  r.length = 15.4 := by
  sorry

#check rectangle_length_calculation

end NUMINAMATH_CALUDE_rectangle_length_calculation_l4114_411499


namespace NUMINAMATH_CALUDE_gcd_product_is_square_l4114_411449

theorem gcd_product_is_square (x y z : ℕ+) 
  (h : (1 : ℚ) / x - (1 : ℚ) / y = (1 : ℚ) / z) :
  ∃ (k : ℕ), (Nat.gcd x.val (Nat.gcd y.val z.val)) * x.val * y.val * z.val = k ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_gcd_product_is_square_l4114_411449


namespace NUMINAMATH_CALUDE_player_a_wins_iff_perfect_square_l4114_411498

/-- The divisor erasing game on a positive integer N -/
def DivisorGame (N : ℕ+) :=
  ∀ (d : ℕ+), d ∣ N → (∃ (m : ℕ+), m ∣ N ∧ (d ∣ m ∨ m ∣ d))

/-- Player A's winning condition -/
def PlayerAWins (N : ℕ+) :=
  ∀ (strategy : ℕ+ → ℕ+),
    (∀ (d : ℕ+), d ∣ N → strategy d ∣ N ∧ (d ∣ strategy d ∨ strategy d ∣ d)) →
    ∃ (move : ℕ+ → ℕ+), 
      (∀ (d : ℕ+), d ∣ N → move d ∣ N ∧ (d ∣ move d ∨ move d ∣ d)) ∧
      (∀ (d : ℕ+), d ∣ N → move (strategy (move d)) ≠ d)

/-- The main theorem: Player A wins if and only if N is a perfect square -/
theorem player_a_wins_iff_perfect_square (N : ℕ+) :
  PlayerAWins N ↔ ∃ (n : ℕ+), N = n * n :=
sorry

end NUMINAMATH_CALUDE_player_a_wins_iff_perfect_square_l4114_411498


namespace NUMINAMATH_CALUDE_largest_integer_proof_l4114_411426

theorem largest_integer_proof (x : ℝ) (h : 20 * Real.sin x = 22 * Real.cos x) :
  ⌊(1 / (Real.sin x * Real.cos x) - 1)^7⌋ = 1 := by
  sorry

end NUMINAMATH_CALUDE_largest_integer_proof_l4114_411426


namespace NUMINAMATH_CALUDE_computers_produced_per_month_l4114_411489

/-- Represents the number of computers produced in a month -/
def computers_per_month (computers_per_interval : ℝ) (days_per_month : ℕ) : ℝ :=
  computers_per_interval * (days_per_month * 24 * 2)

/-- Theorem stating that 4200 computers are produced per month -/
theorem computers_produced_per_month :
  computers_per_month 3.125 28 = 4200 := by
  sorry

end NUMINAMATH_CALUDE_computers_produced_per_month_l4114_411489


namespace NUMINAMATH_CALUDE_sphere_surface_area_circumscribing_cube_l4114_411447

theorem sphere_surface_area_circumscribing_cube (edge_length : ℝ) (sphere_radius : ℝ) :
  edge_length = 2 →
  sphere_radius = edge_length * Real.sqrt 3 / 2 →
  4 * Real.pi * sphere_radius^2 = 12 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_sphere_surface_area_circumscribing_cube_l4114_411447


namespace NUMINAMATH_CALUDE_parallel_resistors_combined_resistance_l4114_411425

/-- The combined resistance of two resistors connected in parallel -/
def combined_resistance (r1 r2 : ℚ) : ℚ :=
  1 / (1 / r1 + 1 / r2)

/-- Theorem: The combined resistance of two resistors with 8 ohms and 9 ohms connected in parallel is 72/17 ohms -/
theorem parallel_resistors_combined_resistance :
  combined_resistance 8 9 = 72 / 17 := by
  sorry

end NUMINAMATH_CALUDE_parallel_resistors_combined_resistance_l4114_411425


namespace NUMINAMATH_CALUDE_average_sum_of_abs_diff_l4114_411407

def sum_of_abs_diff (perm : Fin 8 → Fin 8) : ℕ :=
  |perm 0 - perm 1| + |perm 2 - perm 3| + |perm 4 - perm 5| + |perm 6 - perm 7|

def all_permutations : Finset (Fin 8 → Fin 8) :=
  Finset.univ.filter (fun f => Function.Injective f)

theorem average_sum_of_abs_diff :
  (Finset.sum all_permutations sum_of_abs_diff) / all_permutations.card = 20 := by
  sorry

end NUMINAMATH_CALUDE_average_sum_of_abs_diff_l4114_411407


namespace NUMINAMATH_CALUDE_average_of_three_numbers_l4114_411461

theorem average_of_three_numbers (x : ℝ) : 
  (12 + 21 + x) / 3 = 18 → x = 21 := by
  sorry

end NUMINAMATH_CALUDE_average_of_three_numbers_l4114_411461


namespace NUMINAMATH_CALUDE_all_hanging_pieces_equal_l4114_411438

/-- Represents a square table covered by a square tablecloth -/
structure TableWithCloth where
  table_side : ℝ
  cloth_side : ℝ
  hanging_piece : ℝ → ℝ → ℝ
  no_corner_covered : cloth_side > table_side
  no_overlap : cloth_side ≤ table_side + 2 * (hanging_piece 0 0)
  adjacent_equal : ∀ (i j : Fin 4), (i.val + 1) % 4 = j.val → 
    hanging_piece i.val 0 = hanging_piece j.val 0

/-- All four hanging pieces of the tablecloth are equal -/
theorem all_hanging_pieces_equal (t : TableWithCloth) : 
  ∀ (i j : Fin 4), t.hanging_piece i.val 0 = t.hanging_piece j.val 0 := by
  sorry

end NUMINAMATH_CALUDE_all_hanging_pieces_equal_l4114_411438


namespace NUMINAMATH_CALUDE_sum_of_thousands_and_units_digits_l4114_411402

/-- Represents a 100-digit number with a repeating pattern --/
def RepeatNumber (a b : ℕ) := ℕ

/-- The first 100-digit number: 606060606...060606 --/
def num1 : RepeatNumber 60 6 := sorry

/-- The second 100-digit number: 808080808...080808 --/
def num2 : RepeatNumber 80 8 := sorry

/-- Returns the units digit of a number --/
def unitsDigit (n : ℕ) : ℕ := sorry

/-- Returns the thousands digit of a number --/
def thousandsDigit (n : ℕ) : ℕ := sorry

/-- The product of num1 and num2 --/
def product : ℕ := sorry

theorem sum_of_thousands_and_units_digits :
  thousandsDigit product + unitsDigit product = 14 := by sorry

end NUMINAMATH_CALUDE_sum_of_thousands_and_units_digits_l4114_411402


namespace NUMINAMATH_CALUDE_crayons_lost_or_given_away_l4114_411486

/-- Proves that the number of crayons lost or given away is equal to the sum of crayons given away and crayons lost -/
theorem crayons_lost_or_given_away 
  (initial : ℕ) 
  (given_away : ℕ) 
  (lost : ℕ) 
  (left : ℕ) 
  (h1 : initial = given_away + lost + left) : 
  given_away + lost = initial - left :=
by sorry

end NUMINAMATH_CALUDE_crayons_lost_or_given_away_l4114_411486


namespace NUMINAMATH_CALUDE_water_height_in_aquarium_l4114_411441

/-- Proves that 10 litres of water in an aquarium with dimensions 50 cm length
and 20 cm breadth will rise to a height of 10 cm. -/
theorem water_height_in_aquarium (length : ℝ) (breadth : ℝ) (volume_litres : ℝ) :
  length = 50 →
  breadth = 20 →
  volume_litres = 10 →
  (volume_litres * 1000) / (length * breadth) = 10 := by
  sorry

end NUMINAMATH_CALUDE_water_height_in_aquarium_l4114_411441


namespace NUMINAMATH_CALUDE_cube_difference_l4114_411448

theorem cube_difference (c d : ℝ) (h1 : c - d = 7) (h2 : c^2 + d^2 = 85) : c^3 - d^3 = 721 := by
  sorry

end NUMINAMATH_CALUDE_cube_difference_l4114_411448


namespace NUMINAMATH_CALUDE_art_collection_remaining_l4114_411430

/-- Calculates the remaining number of art pieces after a donation --/
def remaining_art_pieces (initial : ℕ) (donated : ℕ) : ℕ :=
  initial - donated

/-- Theorem: Given 70 initial pieces and 46 donated pieces, 24 pieces remain --/
theorem art_collection_remaining :
  remaining_art_pieces 70 46 = 24 := by
  sorry

end NUMINAMATH_CALUDE_art_collection_remaining_l4114_411430


namespace NUMINAMATH_CALUDE_smallest_valid_student_count_l4114_411478

def is_valid_student_count (n : ℕ) : Prop :=
  20 ∣ n ∧ 
  (Finset.filter (· ∣ n) (Finset.range (n + 1))).card ≥ 15 ∧
  ¬(10 ∣ n) ∧ ¬(25 ∣ n) ∧ ¬(50 ∣ n)

theorem smallest_valid_student_count :
  is_valid_student_count 120 ∧ 
  ∀ m < 120, ¬is_valid_student_count m :=
by sorry

end NUMINAMATH_CALUDE_smallest_valid_student_count_l4114_411478


namespace NUMINAMATH_CALUDE_sum_of_squares_of_roots_l4114_411414

theorem sum_of_squares_of_roots (x₁ x₂ : ℝ) : 
  (6 * x₁^2 - 13 * x₁ + 5 = 0) → 
  (6 * x₂^2 - 13 * x₂ + 5 = 0) → 
  (x₁ ≠ x₂) →
  (x₁^2 + x₂^2 = 109/36) := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_roots_l4114_411414


namespace NUMINAMATH_CALUDE_odd_even_intersection_empty_l4114_411467

def odd_integers : Set ℤ := {x | ∃ k : ℤ, x = 2 * k - 1}
def even_integers : Set ℤ := {x | ∃ k : ℤ, x = 2 * k}

theorem odd_even_intersection_empty : odd_integers ∩ even_integers = ∅ := by
  sorry

end NUMINAMATH_CALUDE_odd_even_intersection_empty_l4114_411467


namespace NUMINAMATH_CALUDE_units_digit_47_power_47_l4114_411464

-- Define a function to get the units digit of a natural number
def unitsDigit (n : ℕ) : ℕ := n % 10

-- Define the theorem
theorem units_digit_47_power_47 : unitsDigit (47^47) = 3 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_47_power_47_l4114_411464


namespace NUMINAMATH_CALUDE_sum_of_squares_divisible_by_three_l4114_411413

theorem sum_of_squares_divisible_by_three (a b c : ℤ) 
  (ha : ¬ 3 ∣ a) (hb : ¬ 3 ∣ b) (hc : ¬ 3 ∣ c) : 
  3 ∣ (a^2 + b^2 + c^2) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_divisible_by_three_l4114_411413


namespace NUMINAMATH_CALUDE_part_one_part_two_l4114_411424

-- Define the functions f and g
def f (a : ℝ) (x : ℝ) : ℝ := |x - 1| + |a * x + 1|
def g (x : ℝ) : ℝ := |x + 1| + 2

-- Part I
theorem part_one :
  {x : ℝ | f (1/2) x < 2} = {x : ℝ | 0 < x ∧ x < 4/3} := by sorry

-- Part II
theorem part_two :
  (∀ x ∈ Set.Ioo 0 1, f a x ≤ g x) → -5 ≤ a ∧ a ≤ 3 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l4114_411424


namespace NUMINAMATH_CALUDE_equation_solution_l4114_411409

theorem equation_solution : ∃ x : ℚ, (1/8 : ℚ) + 8/x = 15/x + (1/15 : ℚ) ∧ x = 120 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l4114_411409


namespace NUMINAMATH_CALUDE_inverse_variation_problem_l4114_411456

/-- Represents the inverse variation relationship between x^3 and ∛w -/
def inverse_variation (x w : ℝ) : Prop := ∃ k : ℝ, x^3 * w^(1/3) = k

/-- Given conditions and theorem statement -/
theorem inverse_variation_problem (x₀ w₀ x₁ w₁ : ℝ) 
  (h₀ : inverse_variation x₀ w₀)
  (h₁ : x₀ = 3)
  (h₂ : w₀ = 8)
  (h₃ : x₁ = 6)
  (h₄ : inverse_variation x₁ w₁) :
  w₁ = 1 / 64 := by
  sorry

end NUMINAMATH_CALUDE_inverse_variation_problem_l4114_411456


namespace NUMINAMATH_CALUDE_bleach_time_is_correct_l4114_411403

/-- Represents the hair dyeing process with given time constraints -/
def HairDyeingProcess (total_time bleach_time : ℝ) : Prop :=
  bleach_time > 0 ∧
  total_time = bleach_time + (4 * bleach_time) + (1/3 * bleach_time)

/-- Theorem stating that given the constraints, the bleaching time is 1.875 hours -/
theorem bleach_time_is_correct (total_time : ℝ) 
  (h : total_time = 10) : 
  ∃ (bleach_time : ℝ), HairDyeingProcess total_time bleach_time ∧ bleach_time = 1.875 := by
  sorry

end NUMINAMATH_CALUDE_bleach_time_is_correct_l4114_411403


namespace NUMINAMATH_CALUDE_value_of_a_l4114_411468

theorem value_of_a (a : ℚ) : a + (2 * a / 5) = 9 / 5 → a = 9 / 7 := by
  sorry

end NUMINAMATH_CALUDE_value_of_a_l4114_411468


namespace NUMINAMATH_CALUDE_tangent_product_upper_bound_l4114_411470

theorem tangent_product_upper_bound (α β : Real) 
  (sum_eq : α + β = Real.pi / 3)
  (α_pos : α > 0)
  (β_pos : β > 0) :
  Real.tan α * Real.tan β ≤ 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_tangent_product_upper_bound_l4114_411470


namespace NUMINAMATH_CALUDE_system_solution_l4114_411406

theorem system_solution (x y a b : ℝ) : 
  x = 1 ∧ 
  y = -2 ∧ 
  3 * x + 2 * y = a ∧ 
  b * x - y = 5 → 
  b - a = 4 := by
sorry

end NUMINAMATH_CALUDE_system_solution_l4114_411406


namespace NUMINAMATH_CALUDE_binomial_expansion_coefficients_l4114_411457

theorem binomial_expansion_coefficients :
  let n : ℕ := 50
  let a : ℕ := 2
  -- Coefficient of x^3
  (n.choose 3) * a^(n - 3) = 19600 * 2^47 ∧
  -- Constant term
  (n.choose 0) * a^n = 2^50 :=
by sorry

end NUMINAMATH_CALUDE_binomial_expansion_coefficients_l4114_411457


namespace NUMINAMATH_CALUDE_green_faction_liars_exceed_truthful_l4114_411476

/-- Represents the three factions in the parliament --/
inductive Faction
  | Blue
  | Red
  | Green

/-- Represents whether a deputy tells the truth or lies --/
inductive Honesty
  | Truthful
  | Liar

/-- Represents the parliament with its properties --/
structure Parliament where
  total_deputies : ℕ
  blue_affirmative : ℕ
  red_affirmative : ℕ
  green_affirmative : ℕ
  deputies : Faction → Honesty → ℕ

/-- The theorem to be proved --/
theorem green_faction_liars_exceed_truthful (p : Parliament)
  (h1 : p.total_deputies = 2016)
  (h2 : p.blue_affirmative = 1208)
  (h3 : p.red_affirmative = 908)
  (h4 : p.green_affirmative = 608)
  (h5 : p.total_deputies = p.deputies Faction.Blue Honesty.Truthful + p.deputies Faction.Blue Honesty.Liar +
                           p.deputies Faction.Red Honesty.Truthful + p.deputies Faction.Red Honesty.Liar +
                           p.deputies Faction.Green Honesty.Truthful + p.deputies Faction.Green Honesty.Liar)
  (h6 : p.blue_affirmative = p.deputies Faction.Blue Honesty.Truthful + p.deputies Faction.Red Honesty.Liar + p.deputies Faction.Green Honesty.Liar)
  (h7 : p.red_affirmative = p.deputies Faction.Red Honesty.Truthful + p.deputies Faction.Blue Honesty.Liar + p.deputies Faction.Green Honesty.Liar)
  (h8 : p.green_affirmative = p.deputies Faction.Green Honesty.Truthful + p.deputies Faction.Blue Honesty.Liar + p.deputies Faction.Red Honesty.Liar) :
  p.deputies Faction.Green Honesty.Liar = p.deputies Faction.Green Honesty.Truthful + 100 := by
  sorry


end NUMINAMATH_CALUDE_green_faction_liars_exceed_truthful_l4114_411476


namespace NUMINAMATH_CALUDE_paint_for_smaller_statues_l4114_411417

/-- The amount of paint (in pints) required for a statue of given height (in feet) -/
def paint_required (height : ℝ) : ℝ := sorry

/-- The number of statues to be painted -/
def num_statues : ℕ := 320

/-- The height (in feet) of the original statue -/
def original_height : ℝ := 8

/-- The height (in feet) of the new statues -/
def new_height : ℝ := 2

/-- The amount of paint (in pints) required for the original statue -/
def original_paint : ℝ := 2

theorem paint_for_smaller_statues :
  paint_required new_height * num_statues = 10 :=
by sorry

end NUMINAMATH_CALUDE_paint_for_smaller_statues_l4114_411417


namespace NUMINAMATH_CALUDE_derivative_of_y_at_2_l4114_411471

-- Define the function y = 3x
def y (x : ℝ) : ℝ := 3 * x

-- State the theorem
theorem derivative_of_y_at_2 :
  deriv y 2 = 3 := by sorry

end NUMINAMATH_CALUDE_derivative_of_y_at_2_l4114_411471


namespace NUMINAMATH_CALUDE_apple_sale_discrepancy_l4114_411495

/-- Represents the number of apples sold for one cent by the first vendor -/
def apples_per_cent_vendor1 : ℕ := 3

/-- Represents the number of apples sold for one cent by the second vendor -/
def apples_per_cent_vendor2 : ℕ := 2

/-- Represents the number of unsold apples each vendor had -/
def unsold_apples_per_vendor : ℕ := 30

/-- Represents the total number of apples to be sold -/
def total_apples : ℕ := 2 * unsold_apples_per_vendor

/-- Represents the number of apples sold for two cents by the friend -/
def apples_per_two_cents_friend : ℕ := 5

/-- Calculates the revenue when apples are sold individually by vendors -/
def revenue_individual : ℕ := 
  (unsold_apples_per_vendor / apples_per_cent_vendor1) + 
  (unsold_apples_per_vendor / apples_per_cent_vendor2)

/-- Calculates the revenue when apples are sold by the friend -/
def revenue_friend : ℕ := 
  2 * (total_apples / apples_per_two_cents_friend)

theorem apple_sale_discrepancy : 
  revenue_individual = revenue_friend + 1 := by
  sorry

end NUMINAMATH_CALUDE_apple_sale_discrepancy_l4114_411495


namespace NUMINAMATH_CALUDE_correct_quadratic_equation_l4114_411455

def quadratic_equation (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

def is_root (f : ℝ → ℝ) (r : ℝ) : Prop := f r = 0

theorem correct_quadratic_equation :
  ∃ (a b c : ℝ),
    (∃ (b₁ c₁ : ℝ), is_root (quadratic_equation a b₁ c₁) 5 ∧ is_root (quadratic_equation a b₁ c₁) 3) ∧
    (∃ (b₂ : ℝ), is_root (quadratic_equation a b₂ c) (-6) ∧ is_root (quadratic_equation a b₂ c) (-4)) ∧
    quadratic_equation a b c = quadratic_equation 1 (-8) 24 :=
by sorry

end NUMINAMATH_CALUDE_correct_quadratic_equation_l4114_411455


namespace NUMINAMATH_CALUDE_final_deleted_files_l4114_411473

def deleted_pictures : ℕ := 5
def deleted_songs : ℕ := 12
def deleted_text_files : ℕ := 10
def deleted_video_files : ℕ := 6
def restored_pictures : ℕ := 3
def restored_video_files : ℕ := 4

theorem final_deleted_files :
  deleted_pictures + deleted_songs + deleted_text_files + deleted_video_files
  - (restored_pictures + restored_video_files) = 26 := by
  sorry

end NUMINAMATH_CALUDE_final_deleted_files_l4114_411473


namespace NUMINAMATH_CALUDE_hyperbola_foci_l4114_411494

/-- The hyperbola equation -/
def hyperbola_equation (x y : ℝ) : Prop := x^2 / 3 - y^2 / 2 = 1

/-- The foci coordinates -/
def foci_coordinates : Set (ℝ × ℝ) := {(-Real.sqrt 5, 0), (Real.sqrt 5, 0)}

/-- Theorem: The foci of the hyperbola are at (±√5, 0) -/
theorem hyperbola_foci :
  ∀ (x y : ℝ), hyperbola_equation x y → (x, y) ∈ foci_coordinates :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_foci_l4114_411494


namespace NUMINAMATH_CALUDE_equation_solutions_l4114_411410

theorem equation_solutions :
  (∀ x : ℝ, (x + 2)^2 = 3*(x + 2) ↔ x = -2 ∨ x = 1) ∧
  (∀ x : ℝ, x^2 - 8*x + 3 = 0 ↔ x = 4 + Real.sqrt 13 ∨ x = 4 - Real.sqrt 13) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l4114_411410


namespace NUMINAMATH_CALUDE_bees_count_l4114_411490

theorem bees_count (first_day_count : ℕ) (second_day_count : ℕ) : 
  (second_day_count = 3 * first_day_count) → 
  (second_day_count = 432) → 
  (first_day_count = 144) := by
sorry

end NUMINAMATH_CALUDE_bees_count_l4114_411490


namespace NUMINAMATH_CALUDE_fraction_difference_equals_difference_over_product_l4114_411463

theorem fraction_difference_equals_difference_over_product 
  (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x + y = x * y) : 
  1 / x - 1 / y = (y - x) / (x * y) := by
  sorry

end NUMINAMATH_CALUDE_fraction_difference_equals_difference_over_product_l4114_411463


namespace NUMINAMATH_CALUDE_reinforcement_arrival_day_l4114_411446

/-- Calculates the number of days passed before reinforcement arrived -/
def days_before_reinforcement (initial_garrison : ℕ) (initial_duration : ℕ) 
  (reinforcement : ℕ) (remaining_duration : ℕ) : ℕ :=
  ((initial_garrison * initial_duration) - 
   ((initial_garrison + reinforcement) * remaining_duration)) / initial_garrison

/-- Theorem stating the number of days passed before reinforcement arrived -/
theorem reinforcement_arrival_day : 
  days_before_reinforcement 2000 54 1600 20 = 18 := by
  sorry

end NUMINAMATH_CALUDE_reinforcement_arrival_day_l4114_411446
