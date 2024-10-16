import Mathlib

namespace NUMINAMATH_CALUDE_woody_savings_l2268_226892

/-- The amount of money Woody already has -/
def money_saved (console_cost weekly_allowance weeks_to_save : ℕ) : ℕ :=
  console_cost - weekly_allowance * weeks_to_save

theorem woody_savings : money_saved 282 24 10 = 42 := by
  sorry

end NUMINAMATH_CALUDE_woody_savings_l2268_226892


namespace NUMINAMATH_CALUDE_six_balls_three_boxes_l2268_226836

/-- The number of ways to place distinguishable balls into distinguishable boxes -/
def place_balls (num_balls : ℕ) (num_boxes : ℕ) : ℕ :=
  num_boxes ^ num_balls

/-- Theorem: There are 729 ways to place 6 distinguishable balls into 3 distinguishable boxes -/
theorem six_balls_three_boxes : place_balls 6 3 = 729 := by
  sorry

end NUMINAMATH_CALUDE_six_balls_three_boxes_l2268_226836


namespace NUMINAMATH_CALUDE_largest_divisor_of_sequence_l2268_226808

theorem largest_divisor_of_sequence (n : ℕ) (h : Even n) (h' : n > 0) :
  ∃ (k : ℕ), k = 105 ∧ 
  (∀ m : ℕ, m > k → ¬(m ∣ (n+1)*(n+3)*(n+5)*(n+7)*(n+9)*(n+11)*(n+13))) ∧
  (k ∣ (n+1)*(n+3)*(n+5)*(n+7)*(n+9)*(n+11)*(n+13)) := by
  sorry

end NUMINAMATH_CALUDE_largest_divisor_of_sequence_l2268_226808


namespace NUMINAMATH_CALUDE_P_divisible_by_Q_l2268_226860

variable (X : ℝ)
variable (n : ℕ)

def P (n : ℕ) (X : ℝ) : ℝ := n * X^(n+2) - (n+2) * X^(n+1) + (n+2) * X - n

def Q (X : ℝ) : ℝ := (X - 1)^3

theorem P_divisible_by_Q (n : ℕ) (h : n > 0) :
  ∃ k : ℝ, P n X = k * Q X := by
  sorry

end NUMINAMATH_CALUDE_P_divisible_by_Q_l2268_226860


namespace NUMINAMATH_CALUDE_quadratic_sum_of_b_and_c_l2268_226852

/-- Given a quadratic expression x^2 - 24x + 50, when written in the form (x+b)^2 + c,
    the sum of b and c is equal to -106. -/
theorem quadratic_sum_of_b_and_c : ∃ (b c : ℝ), 
  (∀ x, x^2 - 24*x + 50 = (x + b)^2 + c) ∧ (b + c = -106) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sum_of_b_and_c_l2268_226852


namespace NUMINAMATH_CALUDE_red_bus_length_l2268_226853

theorem red_bus_length 
  (red_bus orange_car yellow_bus : ℝ) 
  (h1 : red_bus = 4 * orange_car) 
  (h2 : orange_car = yellow_bus / 3.5) 
  (h3 : red_bus = yellow_bus + 6) : 
  red_bus = 48 := by
sorry

end NUMINAMATH_CALUDE_red_bus_length_l2268_226853


namespace NUMINAMATH_CALUDE_ln_20_between_consecutive_integers_l2268_226899

-- Define the natural logarithm function
noncomputable def ln : ℝ → ℝ := Real.log

-- State the theorem
theorem ln_20_between_consecutive_integers :
  ∃ (a b : ℤ), a + 1 = b ∧ (a : ℝ) < ln 20 ∧ ln 20 < (b : ℝ) ∧ a + b = 4 :=
by
  -- Assuming ln e = 1
  have h1 : ln (Real.exp 1) = 1 := by sorry
  -- Assuming ln (e^3) = 3
  have h2 : ln (Real.exp 3) = 3 := by sorry
  -- Assuming ln is an increasing function
  have h3 : Monotone ln := by sorry
  sorry

end NUMINAMATH_CALUDE_ln_20_between_consecutive_integers_l2268_226899


namespace NUMINAMATH_CALUDE_rowing_speed_l2268_226825

/-- The speed of a man rowing in still water, given his downstream speed and current speed. -/
theorem rowing_speed (downstream_speed current_speed : ℝ) 
  (h_downstream : downstream_speed = 18)
  (h_current : current_speed = 3) :
  downstream_speed - current_speed = 15 := by
  sorry

#check rowing_speed

end NUMINAMATH_CALUDE_rowing_speed_l2268_226825


namespace NUMINAMATH_CALUDE_profit_reached_l2268_226897

/-- The number of pencils bought for 6 dollars -/
def pencils_bought : ℕ := 5

/-- The cost in dollars for buying pencils_bought pencils -/
def cost : ℚ := 6

/-- The number of pencils sold for 7 dollars -/
def pencils_sold : ℕ := 4

/-- The revenue in dollars for selling pencils_sold pencils -/
def revenue : ℚ := 7

/-- The target profit in dollars -/
def target_profit : ℚ := 80

/-- The minimum number of pencils that must be sold to reach the target profit -/
def min_pencils_to_sell : ℕ := 146

theorem profit_reached : 
  ∃ (n : ℕ), n ≥ min_pencils_to_sell ∧ 
  n * (revenue / pencils_sold - cost / pencils_bought) ≥ target_profit ∧
  ∀ (m : ℕ), m < min_pencils_to_sell → 
  m * (revenue / pencils_sold - cost / pencils_bought) < target_profit :=
by sorry

end NUMINAMATH_CALUDE_profit_reached_l2268_226897


namespace NUMINAMATH_CALUDE_movie_collection_average_usage_l2268_226820

/-- Given a movie collection that occupies 27,000 megabytes of disk space and lasts for 15 days
    of continuous viewing, the average megabyte usage per hour is 75 megabytes. -/
theorem movie_collection_average_usage
  (total_megabytes : ℕ)
  (total_days : ℕ)
  (h_megabytes : total_megabytes = 27000)
  (h_days : total_days = 15) :
  (total_megabytes : ℚ) / (total_days * 24 : ℚ) = 75 := by
  sorry

end NUMINAMATH_CALUDE_movie_collection_average_usage_l2268_226820


namespace NUMINAMATH_CALUDE_twentieth_term_of_arithmetic_sequence_l2268_226837

/-- An arithmetic sequence is a sequence where the difference between
    consecutive terms is constant. -/
def isArithmeticSequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The 20th term of the specified arithmetic sequence is -49. -/
theorem twentieth_term_of_arithmetic_sequence :
  ∀ a : ℕ → ℤ,
  isArithmeticSequence a →
  a 1 = 8 →
  a 2 = 5 →
  a 3 = 2 →
  a 20 = -49 := by
sorry

end NUMINAMATH_CALUDE_twentieth_term_of_arithmetic_sequence_l2268_226837


namespace NUMINAMATH_CALUDE_solution_and_rationality_l2268_226811

theorem solution_and_rationality 
  (x y : ℝ) 
  (h : Real.sqrt (8 * x - y^2) + |y^2 - 16| = 0) : 
  (x = 2 ∧ (y = 4 ∨ y = -4)) ∧ 
  ((y = 4 → ∃ (q : ℚ), Real.sqrt (y + 12) = ↑q) ∧ 
   (y = -4 → ∀ (q : ℚ), Real.sqrt (y + 12) ≠ ↑q)) := by
  sorry

end NUMINAMATH_CALUDE_solution_and_rationality_l2268_226811


namespace NUMINAMATH_CALUDE_polynomial_coefficient_problem_l2268_226839

theorem polynomial_coefficient_problem (x a : ℝ) (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ : ℝ) 
  (h1 : (x + a)^9 = a₀ + a₁*(x+1) + a₂*(x+1)^2 + a₃*(x+1)^3 + a₄*(x+1)^4 + 
                    a₅*(x+1)^5 + a₆*(x+1)^6 + a₇*(x+1)^7 + a₈*(x+1)^8 + a₉*(x+1)^9)
  (h2 : a₅ = 126) :
  a = 0 ∨ a = 2 := by sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_problem_l2268_226839


namespace NUMINAMATH_CALUDE_investment_comparison_l2268_226885

/-- Represents the final value of an investment after two years --/
def final_value (initial : ℝ) (change1 : ℝ) (change2 : ℝ) (dividend_rate : ℝ) : ℝ :=
  let value1 := initial * (1 + change1)
  let value2 := value1 * (1 + change2)
  value2 + value1 * dividend_rate

/-- Theorem stating the relationship between final investment values --/
theorem investment_comparison : 
  let a := final_value 200 0.15 (-0.10) 0.05
  let b := final_value 150 (-0.20) 0.30 0
  let c := final_value 100 0 0 0
  c < b ∧ b < a := by sorry

end NUMINAMATH_CALUDE_investment_comparison_l2268_226885


namespace NUMINAMATH_CALUDE_inequality_solution_l2268_226842

theorem inequality_solution (x : ℝ) : 
  (2 / (x + 2) - 4 / (x + 8) > 1 / 2) ↔ (x > -4 ∧ x ≠ -2) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l2268_226842


namespace NUMINAMATH_CALUDE_book_reading_theorem_l2268_226845

def days_to_read (n : ℕ) : ℕ := n * (n + 1) / 2

def day_of_week (start_day : ℕ) (days_passed : ℕ) : ℕ :=
  (start_day + days_passed) % 7

theorem book_reading_theorem :
  let num_books := 18
  let start_day := 0  -- 0 represents Sunday
  let total_days := days_to_read num_books
  day_of_week start_day total_days = 3  -- 3 represents Wednesday
:= by sorry

end NUMINAMATH_CALUDE_book_reading_theorem_l2268_226845


namespace NUMINAMATH_CALUDE_wheel_configuration_theorem_l2268_226833

/-- Represents a configuration of wheels with spokes -/
structure WheelConfiguration where
  num_wheels : ℕ
  max_spokes_per_wheel : ℕ
  total_visible_spokes : ℕ

/-- Checks if a given wheel configuration is possible -/
def is_possible_configuration (config : WheelConfiguration) : Prop :=
  config.num_wheels * config.max_spokes_per_wheel ≥ config.total_visible_spokes

/-- Theorem stating the possibility of 3 wheels and impossibility of 2 wheels -/
theorem wheel_configuration_theorem :
  let config_3 : WheelConfiguration := ⟨3, 3, 7⟩
  let config_2 : WheelConfiguration := ⟨2, 3, 7⟩
  is_possible_configuration config_3 ∧ ¬is_possible_configuration config_2 := by
  sorry

#check wheel_configuration_theorem

end NUMINAMATH_CALUDE_wheel_configuration_theorem_l2268_226833


namespace NUMINAMATH_CALUDE_number_divisibility_l2268_226819

theorem number_divisibility (a b : ℕ) : 
  (∃ k : ℤ, (1001 * a + 110 * b : ℤ) = 11 * k) ∧ 
  (∃ m : ℤ, (111000 * a + 111 * b : ℤ) = 37 * m) ∧
  (∃ n : ℤ, (101010 * a + 10101 * b : ℤ) = 7 * n) ∧
  (∃ p q : ℤ, (909 * (a - b) : ℤ) = 9 * p ∧ (909 * (a - b) : ℤ) = 101 * q) :=
by sorry

end NUMINAMATH_CALUDE_number_divisibility_l2268_226819


namespace NUMINAMATH_CALUDE_correct_average_after_error_correction_l2268_226813

theorem correct_average_after_error_correction 
  (n : ℕ) 
  (initial_average : ℚ) 
  (wrong_mark correct_mark : ℚ) : 
  n = 10 → 
  initial_average = 100 → 
  wrong_mark = 90 → 
  correct_mark = 10 → 
  (n * initial_average - (wrong_mark - correct_mark)) / n = 92 := by
sorry

end NUMINAMATH_CALUDE_correct_average_after_error_correction_l2268_226813


namespace NUMINAMATH_CALUDE_largest_share_is_12000_l2268_226848

/-- Represents the profit split ratio for four partners -/
structure ProfitSplit :=
  (a b c d : ℕ)

/-- Calculates the largest share given a total profit and a profit split ratio -/
def largest_share (total_profit : ℕ) (split : ProfitSplit) : ℕ :=
  let total_parts := split.a + split.b + split.c + split.d
  let largest_part := max split.a (max split.b (max split.c split.d))
  (total_profit / total_parts) * largest_part

/-- The theorem stating that the largest share is $12,000 -/
theorem largest_share_is_12000 :
  largest_share 30000 ⟨1, 4, 4, 6⟩ = 12000 := by
  sorry

#eval largest_share 30000 ⟨1, 4, 4, 6⟩

end NUMINAMATH_CALUDE_largest_share_is_12000_l2268_226848


namespace NUMINAMATH_CALUDE_max_sphere_volume_in_prism_l2268_226821

/-- The maximum volume of a sphere inscribed in a right triangular prism -/
theorem max_sphere_volume_in_prism (a b h : ℝ) (ha : 0 < a) (hb : 0 < b) (hh : 0 < h) :
  let r := min (h / 2) (a * b / (a + b + (a^2 + b^2).sqrt))
  (4 / 3) * π * r^3 = (9 * π) / 2 :=
by sorry

end NUMINAMATH_CALUDE_max_sphere_volume_in_prism_l2268_226821


namespace NUMINAMATH_CALUDE_ice_cream_scoop_arrangements_l2268_226822

theorem ice_cream_scoop_arrangements : (Finset.range 5).card.factorial = 120 := by
  sorry

end NUMINAMATH_CALUDE_ice_cream_scoop_arrangements_l2268_226822


namespace NUMINAMATH_CALUDE_even_function_sum_l2268_226844

def f (a b x : ℝ) : ℝ := a * x^2 + (b - 3) * x + 3

theorem even_function_sum (a b : ℝ) :
  (∀ x ∈ Set.Icc (a - 2) a, f a b x = f a b ((a - 2) + a - x)) →
  a + b = 4 := by
sorry

end NUMINAMATH_CALUDE_even_function_sum_l2268_226844


namespace NUMINAMATH_CALUDE_unique_solution_f_f_x_eq_27_l2268_226834

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 3*x^2 - 9*x + 27

-- State the theorem
theorem unique_solution_f_f_x_eq_27 :
  ∃! x : ℝ, x ∈ Set.Icc (-3 : ℝ) 5 ∧ f (f x) = 27 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_f_f_x_eq_27_l2268_226834


namespace NUMINAMATH_CALUDE_jessica_total_cost_l2268_226841

def cat_toy_cost : ℚ := 10.22
def cage_cost : ℚ := 11.73
def cat_food_cost : ℚ := 7.50
def leash_cost : ℚ := 5.15
def cat_treats_cost : ℚ := 3.98

theorem jessica_total_cost :
  cat_toy_cost + cage_cost + cat_food_cost + leash_cost + cat_treats_cost = 38.58 := by
  sorry

end NUMINAMATH_CALUDE_jessica_total_cost_l2268_226841


namespace NUMINAMATH_CALUDE_calculation_proof_equation_solution_l2268_226818

-- Problem 1
theorem calculation_proof :
  |Real.sqrt 3 - 2| + Real.sqrt 12 - 6 * Real.sin (30 * π / 180) + (-1/2)⁻¹ = Real.sqrt 3 - 3 := by
  sorry

-- Problem 2
theorem equation_solution :
  let solutions := {x : ℝ | x * (x + 6) = -5}
  solutions = {-5, -1} := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_equation_solution_l2268_226818


namespace NUMINAMATH_CALUDE_journey_time_proof_l2268_226828

/-- Proves that a journey of 224 km, divided into two equal halves with different speeds, takes 10 hours -/
theorem journey_time_proof (total_distance : ℝ) (speed1 : ℝ) (speed2 : ℝ) 
  (h1 : total_distance = 224)
  (h2 : speed1 = 21)
  (h3 : speed2 = 24) :
  (total_distance / 2 / speed1) + (total_distance / 2 / speed2) = 10 := by
  sorry

end NUMINAMATH_CALUDE_journey_time_proof_l2268_226828


namespace NUMINAMATH_CALUDE_plot_length_is_56_l2268_226807

/-- Proves that the length of a rectangular plot is 56 meters given the specified conditions -/
theorem plot_length_is_56 (breadth : ℝ) (length : ℝ) (perimeter : ℝ) (cost_per_meter : ℝ) (total_cost : ℝ) :
  length = breadth + 12 →
  perimeter = 2 * (length + breadth) →
  cost_per_meter = 26.5 →
  total_cost = 5300 →
  total_cost = perimeter * cost_per_meter →
  length = 56 := by
sorry

end NUMINAMATH_CALUDE_plot_length_is_56_l2268_226807


namespace NUMINAMATH_CALUDE_min_area_archimedean_triangle_l2268_226864

/-- Represents a parabola with equation y^2 = 2px where p > 0 -/
structure Parabola where
  p : ℝ
  h_pos : p > 0

/-- Represents a chord of a parabola -/
structure Chord (para : Parabola) where
  A : ℝ × ℝ
  B : ℝ × ℝ

/-- Represents the Archimedean triangle of a parabola and chord -/
structure ArchimedeanTriangle (para : Parabola) (chord : Chord para) where
  Q : ℝ × ℝ

/-- Predicate to check if a chord passes through the focus of a parabola -/
def passes_through_focus (para : Parabola) (chord : Chord para) : Prop :=
  ∃ t : ℝ, chord.A = (para.p / 2, t) ∨ chord.B = (para.p / 2, t)

/-- Calculate the area of a triangle given its vertices -/
def triangle_area (A B Q : ℝ × ℝ) : ℝ := sorry

/-- The main theorem: The minimum area of the Archimedean triangle is p^2 -/
theorem min_area_archimedean_triangle (para : Parabola) 
  (chord : Chord para) (arch_tri : ArchimedeanTriangle para chord)
  (h_focus : passes_through_focus para chord) :
  ∃ (min_area : ℝ), 
    (∀ (other_chord : Chord para) (other_tri : ArchimedeanTriangle para other_chord),
      passes_through_focus para other_chord → 
      triangle_area arch_tri.Q chord.A chord.B ≤ triangle_area other_tri.Q other_chord.A other_chord.B) ∧
    min_area = para.p^2 := by
  sorry

end NUMINAMATH_CALUDE_min_area_archimedean_triangle_l2268_226864


namespace NUMINAMATH_CALUDE_line_through_point_parallel_to_polar_axis_l2268_226829

/-- A point in polar coordinates -/
structure PolarPoint where
  ρ : ℝ
  θ : ℝ

/-- Represents a line in polar coordinates -/
structure PolarLine where
  -- The equation of the line in the form ρ sin θ = k
  k : ℝ

/-- Checks if a point lies on a given polar line -/
def isOnLine (p : PolarPoint) (l : PolarLine) : Prop :=
  p.ρ * Real.sin p.θ = l.k

theorem line_through_point_parallel_to_polar_axis 
  (p : PolarPoint) (l : PolarLine) 
  (h1 : p.ρ = 1) 
  (h2 : p.θ = Real.pi / 2) 
  (h3 : l.k = 1) : 
  ∀ q : PolarPoint, isOnLine q l ↔ q.ρ * Real.sin q.θ = 1 := by
  sorry

#check line_through_point_parallel_to_polar_axis

end NUMINAMATH_CALUDE_line_through_point_parallel_to_polar_axis_l2268_226829


namespace NUMINAMATH_CALUDE_nearest_city_distance_l2268_226882

theorem nearest_city_distance (d : ℝ) : 
  (¬ (d ≥ 13)) ∧ (¬ (d ≤ 10)) ∧ (¬ (d ≤ 8)) → d ∈ Set.Ioo 10 13 :=
by sorry

end NUMINAMATH_CALUDE_nearest_city_distance_l2268_226882


namespace NUMINAMATH_CALUDE_trigonometric_identity_l2268_226868

theorem trigonometric_identity (α : ℝ) : 
  3 + 4 * Real.sin (4 * α + 3 / 2 * Real.pi) + Real.sin (8 * α + 5 / 2 * Real.pi) = 8 * (Real.sin (2 * α))^4 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l2268_226868


namespace NUMINAMATH_CALUDE_bread_and_ham_percentage_l2268_226891

def bread_cost : ℚ := 50
def ham_cost : ℚ := 150
def cake_cost : ℚ := 200

theorem bread_and_ham_percentage : 
  (bread_cost + ham_cost) / (bread_cost + ham_cost + cake_cost) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_bread_and_ham_percentage_l2268_226891


namespace NUMINAMATH_CALUDE_solve_for_b_l2268_226863

theorem solve_for_b (a b : ℕ) (h1 : a = 105) (h2 : a^3 = 21 * 25 * 45 * b) : b = 49 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_b_l2268_226863


namespace NUMINAMATH_CALUDE_share_multiple_problem_l2268_226812

theorem share_multiple_problem (total a b c x : ℚ) : 
  total = 880 →
  c = 160 →
  a + b + c = total →
  4 * a = 5 * b →
  4 * a = x * c →
  x = 10 := by
  sorry

end NUMINAMATH_CALUDE_share_multiple_problem_l2268_226812


namespace NUMINAMATH_CALUDE_four_digit_integers_with_one_or_seven_l2268_226880

/-- The number of four-digit positive integers -/
def total_four_digit_integers : ℕ := 9000

/-- The number of four-digit positive integers without 1 or 7 -/
def four_digit_integers_without_one_or_seven : ℕ := 3584

/-- Theorem: The number of four-digit positive integers with at least one digit as 1 or 7 -/
theorem four_digit_integers_with_one_or_seven :
  total_four_digit_integers - four_digit_integers_without_one_or_seven = 5416 := by
  sorry

end NUMINAMATH_CALUDE_four_digit_integers_with_one_or_seven_l2268_226880


namespace NUMINAMATH_CALUDE_candy_box_max_money_l2268_226874

/-- Calculates the maximum amount of money that can be made by selling boxed candies. -/
def max_money (total_candies : ℕ) (candies_per_box : ℕ) (price_per_box : ℕ) : ℕ :=
  (total_candies / candies_per_box) * price_per_box

/-- Theorem stating the maximum amount of money for the given candy problem. -/
theorem candy_box_max_money :
  max_money 235 10 3000 = 69000 := by
  sorry

end NUMINAMATH_CALUDE_candy_box_max_money_l2268_226874


namespace NUMINAMATH_CALUDE_set_b_forms_triangle_l2268_226815

/-- Triangle inequality theorem: the sum of the lengths of any two sides of a triangle
    must be greater than the length of the remaining side. -/
def triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- A function to check if three line segments can form a triangle. -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ triangle_inequality a b c

/-- Theorem: The set of line segments (6, 7, 8) can form a triangle. -/
theorem set_b_forms_triangle : can_form_triangle 6 7 8 := by
  sorry

end NUMINAMATH_CALUDE_set_b_forms_triangle_l2268_226815


namespace NUMINAMATH_CALUDE_seventh_term_of_sequence_l2268_226867

def geometric_sequence (a : ℕ) (r : ℕ) (n : ℕ) : ℕ := a * r^(n - 1)

theorem seventh_term_of_sequence (a₁ a₅ : ℕ) (h₁ : a₁ = 3) (h₅ : a₅ = 243) :
  ∃ r : ℕ, 
    (geometric_sequence a₁ r 5 = a₅) ∧ 
    (geometric_sequence a₁ r 7 = 2187) := by
  sorry

end NUMINAMATH_CALUDE_seventh_term_of_sequence_l2268_226867


namespace NUMINAMATH_CALUDE_unique_integer_fraction_l2268_226843

theorem unique_integer_fraction (m n : ℕ) (h1 : m ≥ 3) (h2 : n ≥ 3) :
  (∃ S : Set ℕ, (Set.Infinite S ∧
    ∀ a ∈ S, ∃ k : ℤ, (a^m + a - 1 : ℤ) = k * (a^n + a^2 - 1)))
  ↔ m = 5 ∧ n = 3 := by
sorry

end NUMINAMATH_CALUDE_unique_integer_fraction_l2268_226843


namespace NUMINAMATH_CALUDE_parallel_vectors_magnitude_l2268_226869

/-- Given two parallel vectors a and b, prove that the magnitude of 3a + 2b is √5 -/
theorem parallel_vectors_magnitude (a b : ℝ × ℝ) : 
  a = (1, -2) → 
  b.1 = -2 → 
  ∃ y, b.2 = y → 
  (∃ k : ℝ, k ≠ 0 ∧ a = k • b) → 
  ‖3 • a + 2 • b‖ = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_magnitude_l2268_226869


namespace NUMINAMATH_CALUDE_f_composition_half_l2268_226896

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then Real.exp x else Real.log x

theorem f_composition_half : f (f (1/2)) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_half_l2268_226896


namespace NUMINAMATH_CALUDE_sum_in_quadrant_IV_l2268_226803

/-- The sum of z₁ = 1 - 3i and z₂ = 3 + 2i lies in Quadrant IV of the complex plane. -/
theorem sum_in_quadrant_IV : 
  let z₁ : ℂ := 1 - 3 * Complex.I
  let z₂ : ℂ := 3 + 2 * Complex.I
  let sum := z₁ + z₂
  (sum.re > 0 ∧ sum.im < 0) := by sorry

end NUMINAMATH_CALUDE_sum_in_quadrant_IV_l2268_226803


namespace NUMINAMATH_CALUDE_rectangle_side_difference_l2268_226873

theorem rectangle_side_difference (p d : ℝ) (hp : p > 0) (hd : d > 0) :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a > b ∧ 2 * (a + b) = p ∧ a^2 + b^2 = d^2 ∧ a - b = (Real.sqrt (8 * d^2 - p^2)) / 2 :=
sorry

end NUMINAMATH_CALUDE_rectangle_side_difference_l2268_226873


namespace NUMINAMATH_CALUDE_problem_statement_l2268_226817

/-- Given that a² + ab = -2 and b² - 3ab = -3, prove that a² + 4ab - b² = 1 -/
theorem problem_statement (a b : ℝ) (h1 : a^2 + a*b = -2) (h2 : b^2 - 3*a*b = -3) :
  a^2 + 4*a*b - b^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2268_226817


namespace NUMINAMATH_CALUDE_max_stamps_purchasable_l2268_226879

/-- Given a stamp price of 25 cents and a budget of 5000 cents,
    the maximum number of stamps that can be purchased is 200. -/
theorem max_stamps_purchasable (stamp_price : ℕ) (budget : ℕ) (h1 : stamp_price = 25) (h2 : budget = 5000) :
  ∃ (n : ℕ), n = 200 ∧ n * stamp_price ≤ budget ∧ ∀ m : ℕ, m * stamp_price ≤ budget → m ≤ n :=
sorry

end NUMINAMATH_CALUDE_max_stamps_purchasable_l2268_226879


namespace NUMINAMATH_CALUDE_joe_chocolate_spending_l2268_226855

theorem joe_chocolate_spending (total : ℚ) (fruit_fraction : ℚ) (left : ℚ) 
  (h1 : total = 450)
  (h2 : fruit_fraction = 2/5)
  (h3 : left = 220) :
  (total - left - fruit_fraction * total) / total = 1/9 := by
  sorry

end NUMINAMATH_CALUDE_joe_chocolate_spending_l2268_226855


namespace NUMINAMATH_CALUDE_complex_power_eight_l2268_226809

theorem complex_power_eight : (3 * Complex.cos (π / 4) - 3 * Complex.I * Complex.sin (π / 4)) ^ 8 = 6552 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_eight_l2268_226809


namespace NUMINAMATH_CALUDE_additional_concession_percentage_l2268_226878

def original_price : ℝ := 2000
def standard_concession : ℝ := 30
def final_price : ℝ := 1120

theorem additional_concession_percentage :
  ∃ (additional_concession : ℝ),
    (original_price * (1 - standard_concession / 100) * (1 - additional_concession / 100) = final_price) ∧
    additional_concession = 20 := by
  sorry

end NUMINAMATH_CALUDE_additional_concession_percentage_l2268_226878


namespace NUMINAMATH_CALUDE_circle_satisfies_conditions_l2268_226806

-- Define the circle C1
def C1 (x y : ℝ) : Prop := x^2 + y^2 - 4*y = 0

-- Define our circle
def our_circle (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 2

-- Define what it means for two circles to be tangent
def tangent (f g : ℝ → ℝ → Prop) : Prop :=
  ∃ x y : ℝ, f x y ∧ g x y ∧ 
  ∀ ε > 0, ∃ δ > 0, ∀ x' y' : ℝ, 
    ((x' - x)^2 + (y' - y)^2 < δ^2) → 
    (f x' y' ↔ g x' y') → (x' = x ∧ y' = y)

theorem circle_satisfies_conditions : 
  our_circle 3 1 ∧ 
  our_circle 1 1 ∧ 
  tangent our_circle C1 := by sorry

end NUMINAMATH_CALUDE_circle_satisfies_conditions_l2268_226806


namespace NUMINAMATH_CALUDE_exam_time_allocation_l2268_226838

theorem exam_time_allocation :
  ∀ (total_time total_questions type_a_questions : ℕ) 
    (type_a_time_ratio : ℚ),
  total_time = 180 →
  total_questions = 200 →
  type_a_questions = 20 →
  type_a_time_ratio = 2 →
  ∃ (type_a_time : ℕ),
    type_a_time = 36 ∧
    type_a_time * (total_questions - type_a_questions) = 
      (total_time - type_a_time) * type_a_questions * type_a_time_ratio :=
by sorry

end NUMINAMATH_CALUDE_exam_time_allocation_l2268_226838


namespace NUMINAMATH_CALUDE_infinite_geometric_series_common_ratio_l2268_226876

theorem infinite_geometric_series_common_ratio 
  (a : ℝ) 
  (S : ℝ) 
  (h1 : a = 512) 
  (h2 : S = 8000) : 
  ∃ (r : ℝ), r = 0.936 ∧ S = a / (1 - r) := by
sorry

end NUMINAMATH_CALUDE_infinite_geometric_series_common_ratio_l2268_226876


namespace NUMINAMATH_CALUDE_negation_of_proposition_l2268_226862

theorem negation_of_proposition :
  (¬ ∀ x : ℝ, x > 0 → x^2 - 2*x + 1 ≥ 0) ↔ (∃ x : ℝ, x > 0 ∧ x^2 - 2*x + 1 < 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l2268_226862


namespace NUMINAMATH_CALUDE_power_sum_prime_l2268_226840

theorem power_sum_prime (p a n : ℕ) : 
  Prime p → a > 0 → n > 0 → (2 ^ p + 3 ^ p = a ^ n) → n = 1 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_prime_l2268_226840


namespace NUMINAMATH_CALUDE_cube_root_of_64_l2268_226888

theorem cube_root_of_64 : (64 : ℝ) ^ (1/3 : ℝ) = 4 := by sorry

end NUMINAMATH_CALUDE_cube_root_of_64_l2268_226888


namespace NUMINAMATH_CALUDE_greatest_common_multiple_9_15_under_150_l2268_226831

theorem greatest_common_multiple_9_15_under_150 :
  ∃ n : ℕ, n = 135 ∧ 
  (∀ m : ℕ, m < 150 → m % 9 = 0 → m % 15 = 0 → m ≤ n) ∧
  135 % 9 = 0 ∧ 135 % 15 = 0 ∧ 135 < 150 :=
by sorry

end NUMINAMATH_CALUDE_greatest_common_multiple_9_15_under_150_l2268_226831


namespace NUMINAMATH_CALUDE_total_age_is_22_l2268_226816

/-- Given three people a, b, and c, where:
  - a is two years older than b
  - b is twice as old as c
  - b is 8 years old
  Prove that the total of their ages is 22 years. -/
theorem total_age_is_22 (a b c : ℕ) : 
  b = 8 → 
  a = b + 2 → 
  b = 2 * c → 
  a + b + c = 22 := by sorry

end NUMINAMATH_CALUDE_total_age_is_22_l2268_226816


namespace NUMINAMATH_CALUDE_increasing_function_inequality_l2268_226830

-- Define a real-valued function f
variable (f : ℝ → ℝ)

-- Define the property of f being increasing on ℝ
def IsIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

-- Theorem statement
theorem increasing_function_inequality (h_incr : IsIncreasing f) (m : ℝ) :
  f (2 * m) > f (-m + 9) → m > 3 := by
  sorry


end NUMINAMATH_CALUDE_increasing_function_inequality_l2268_226830


namespace NUMINAMATH_CALUDE_inequality_of_logarithms_l2268_226851

theorem inequality_of_logarithms (a b c : ℝ) 
  (ha : a = Real.log 2) 
  (hb : b = Real.log 3) 
  (hc : c = Real.log 5) : 
  c / 5 < a / 2 ∧ a / 2 < b / 3 := by sorry

end NUMINAMATH_CALUDE_inequality_of_logarithms_l2268_226851


namespace NUMINAMATH_CALUDE_hyperbola_conjugate_axis_length_l2268_226810

/-- Represents a hyperbola with equation x^2/5 - y^2/b^2 = 1 -/
structure Hyperbola where
  b : ℝ
  eq : ∀ x y : ℝ, x^2/5 - y^2/b^2 = 1

/-- The distance from the focus to the asymptote of the hyperbola -/
def focus_to_asymptote_distance (h : Hyperbola) : ℝ := 2

/-- The length of the conjugate axis of the hyperbola -/
def conjugate_axis_length (h : Hyperbola) : ℝ := 2 * h.b

theorem hyperbola_conjugate_axis_length (h : Hyperbola) :
  focus_to_asymptote_distance h = 2 →
  conjugate_axis_length h = 4 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_conjugate_axis_length_l2268_226810


namespace NUMINAMATH_CALUDE_right_triangle_area_l2268_226887

theorem right_triangle_area (a b c : ℝ) (h1 : a = 15) (h2 : c = 17) (h3 : a^2 + b^2 = c^2) :
  (1/2) * a * b = 60 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_area_l2268_226887


namespace NUMINAMATH_CALUDE_min_sum_m_n_in_arithmetic_sequence_l2268_226827

theorem min_sum_m_n_in_arithmetic_sequence (a : ℕ → ℕ) (d m n : ℕ) :
  (∀ k, a k > 0) →
  (∀ k, a (k + 1) = a k + d) →
  a 1 = 1919 →
  a m = 1949 →
  a n = 2019 →
  m > 0 →
  n > 0 →
  ∃ (m' n' : ℕ), m' > 0 ∧ n' > 0 ∧ a m' = 1949 ∧ a n' = 2019 ∧ m' + n' = 15 ∧
    ∀ (p q : ℕ), p > 0 → q > 0 → a p = 1949 → a q = 2019 → m' + n' ≤ p + q :=
by sorry

end NUMINAMATH_CALUDE_min_sum_m_n_in_arithmetic_sequence_l2268_226827


namespace NUMINAMATH_CALUDE_candy_original_pencils_l2268_226877

/-- The number of pencils each person has -/
structure PencilCounts where
  calen : ℕ
  caleb : ℕ
  candy : ℕ
  darlene : ℕ

/-- The conditions of the problem -/
def pencil_problem (p : PencilCounts) : Prop :=
  p.calen = p.caleb + 5 ∧
  p.caleb = 2 * p.candy - 3 ∧
  p.darlene = p.calen + p.caleb + p.candy + 4 ∧
  p.calen - 10 = 10

theorem candy_original_pencils (p : PencilCounts) : 
  pencil_problem p → p.candy = 9 := by
  sorry

end NUMINAMATH_CALUDE_candy_original_pencils_l2268_226877


namespace NUMINAMATH_CALUDE_y_intercept_of_specific_line_l2268_226850

/-- A line in the xy-plane is defined by its slope and a point it passes through. -/
structure Line where
  slope : ℝ
  point : ℝ × ℝ

/-- The y-intercept of a line is the y-coordinate where the line intersects the y-axis. -/
def y_intercept (l : Line) : ℝ :=
  l.point.2 - l.slope * l.point.1

/-- 
Given a line with slope 4 passing through the point (199, 800),
prove that its y-intercept is 4.
-/
theorem y_intercept_of_specific_line :
  let l : Line := { slope := 4, point := (199, 800) }
  y_intercept l = 4 := by
  sorry

end NUMINAMATH_CALUDE_y_intercept_of_specific_line_l2268_226850


namespace NUMINAMATH_CALUDE_greatest_divisor_with_remainder_l2268_226805

theorem greatest_divisor_with_remainder (a b c : ℕ) (h : a = 263 ∧ b = 935 ∧ c = 1383) :
  (∃ (d : ℕ), d > 0 ∧ 
    (a % d = 7 ∧ b % d = 7 ∧ c % d = 7) ∧
    (∀ (k : ℕ), k > d → (a % k ≠ 7 ∨ b % k ≠ 7 ∨ c % k ≠ 7))) →
  (∃ (d : ℕ), d = 16 ∧
    (a % d = 7 ∧ b % d = 7 ∧ c % d = 7) ∧
    (∀ (k : ℕ), k > d → (a % k ≠ 7 ∨ b % k ≠ 7 ∨ c % k ≠ 7))) :=
by sorry

end NUMINAMATH_CALUDE_greatest_divisor_with_remainder_l2268_226805


namespace NUMINAMATH_CALUDE_roots_sum_inverse_squares_l2268_226824

theorem roots_sum_inverse_squares (a b c : ℝ) (r s : ℂ) (h₁ : a ≠ 0) (h₂ : c ≠ 0) 
  (h₃ : a * r^2 + b * r - c = 0) (h₄ : a * s^2 + b * s - c = 0) : 
  1 / r^2 + 1 / s^2 = (b^2 + 2*a*c) / c^2 := by
sorry

end NUMINAMATH_CALUDE_roots_sum_inverse_squares_l2268_226824


namespace NUMINAMATH_CALUDE_even_perfect_square_divisible_by_eight_l2268_226895

theorem even_perfect_square_divisible_by_eight (b n : ℕ) : 
  b > 0 → 
  Even b → 
  n > 1 → 
  ∃ k : ℕ, (b^n - 1) / (b - 1) = k^2 → 
  8 ∣ b :=
sorry

end NUMINAMATH_CALUDE_even_perfect_square_divisible_by_eight_l2268_226895


namespace NUMINAMATH_CALUDE_second_chapter_pages_l2268_226800

/-- A book with three chapters -/
structure Book where
  chapter1 : ℕ
  chapter2 : ℕ
  chapter3 : ℕ

/-- The book satisfies the given conditions -/
def satisfiesConditions (b : Book) : Prop :=
  b.chapter1 = 35 ∧ b.chapter3 = 3 ∧ b.chapter2 = b.chapter3 + 15

theorem second_chapter_pages (b : Book) (h : satisfiesConditions b) : b.chapter2 = 18 := by
  sorry

end NUMINAMATH_CALUDE_second_chapter_pages_l2268_226800


namespace NUMINAMATH_CALUDE_sum_of_max_min_F_l2268_226886

-- Define the function f as an odd function on [-a, a]
def f (a : ℝ) (x : ℝ) : ℝ := sorry

-- Define F(x) = f(x) + 1
def F (a : ℝ) (x : ℝ) : ℝ := f a x + 1

-- Theorem statement
theorem sum_of_max_min_F (a : ℝ) (h : a > 0) :
  ∃ (x_max x_min : ℝ), x_max ∈ Set.Icc (-a) a ∧ x_min ∈ Set.Icc (-a) a ∧
  (∀ x ∈ Set.Icc (-a) a, F a x ≤ F a x_max) ∧
  (∀ x ∈ Set.Icc (-a) a, F a x_min ≤ F a x) ∧
  F a x_max + F a x_min = 2 :=
sorry

end NUMINAMATH_CALUDE_sum_of_max_min_F_l2268_226886


namespace NUMINAMATH_CALUDE_units_digit_of_8_power_47_l2268_226875

theorem units_digit_of_8_power_47 : Nat.mod (8^47) 10 = 2 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_8_power_47_l2268_226875


namespace NUMINAMATH_CALUDE_binomial_congruence_characterization_l2268_226881

theorem binomial_congruence_characterization (n : ℕ) (hn : n ≥ 2) :
  (∀ i j : ℕ, 0 ≤ i → i ≤ j → j ≤ n →
    (i + j) % 2 = (Nat.choose n i + Nat.choose n j) % 2) ↔
  ∃ p : ℕ, p > 0 ∧ n = 2^p - 2 :=
by sorry

end NUMINAMATH_CALUDE_binomial_congruence_characterization_l2268_226881


namespace NUMINAMATH_CALUDE_positive_solution_x_l2268_226894

theorem positive_solution_x (x y z : ℝ) 
  (eq1 : x * y = 12 - 3 * x - 4 * y)
  (eq2 : y * z = 8 - 2 * y - 3 * z)
  (eq3 : x * z = 42 - 5 * x - 6 * z)
  (h_positive : x > 0) : x = 6 := by
  sorry

end NUMINAMATH_CALUDE_positive_solution_x_l2268_226894


namespace NUMINAMATH_CALUDE_sticker_distribution_l2268_226883

/-- The number of ways to distribute indistinguishable objects into distinguishable groups -/
def distribute (n : ℕ) (k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- Theorem: There are 1365 ways to distribute 11 indistinguishable stickers into 5 distinguishable sheets of paper -/
theorem sticker_distribution : distribute 11 5 = 1365 := by
  sorry

end NUMINAMATH_CALUDE_sticker_distribution_l2268_226883


namespace NUMINAMATH_CALUDE_angle_1303_equivalent_to_negative_137_l2268_226857

-- Define a function to reduce an angle to its equivalent angle between 0° and 360°
def reduce_angle (angle : Int) : Int :=
  angle % 360

-- Theorem statement
theorem angle_1303_equivalent_to_negative_137 :
  reduce_angle 1303 = reduce_angle (-137) :=
sorry

end NUMINAMATH_CALUDE_angle_1303_equivalent_to_negative_137_l2268_226857


namespace NUMINAMATH_CALUDE_x_eq_one_iff_z_purely_imaginary_l2268_226823

/-- A complex number is purely imaginary if its real part is zero and its imaginary part is non-zero. -/
def IsPurelyImaginary (z : ℂ) : Prop :=
  z.re = 0 ∧ z.im ≠ 0

/-- The complex number z as a function of x. -/
def z (x : ℝ) : ℂ :=
  ⟨x^2 - 1, x + 1⟩

/-- Theorem stating that x = 1 is necessary and sufficient for z(x) to be purely imaginary. -/
theorem x_eq_one_iff_z_purely_imaginary :
  ∀ x : ℝ, x = 1 ↔ IsPurelyImaginary (z x) :=
sorry

end NUMINAMATH_CALUDE_x_eq_one_iff_z_purely_imaginary_l2268_226823


namespace NUMINAMATH_CALUDE_third_row_is_4213_l2268_226884

/-- Represents a 4x4 grid of numbers -/
def Grid := Fin 4 → Fin 4 → Fin 4

/-- Checks if a number is the first odd or even in a list -/
def isFirstOddOrEven (n : Fin 4) (list : List (Fin 4)) : Prop :=
  n.val % 2 ≠ list.head!.val % 2 ∧ 
  ∀ m ∈ list, m.val < n.val → m.val % 2 = list.head!.val % 2

/-- The constraints of the grid puzzle -/
structure GridConstraints (grid : Grid) : Prop where
  unique_in_row : ∀ i j k, j ≠ k → grid i j ≠ grid i k
  unique_in_col : ∀ i j k, i ≠ k → grid i j ≠ grid k j
  top_indicators : ∀ j, isFirstOddOrEven (grid 0 j) [grid 1 j, grid 2 j, grid 3 j]
  left_indicators : ∀ i, isFirstOddOrEven (grid i 0) [grid i 1, grid i 2, grid i 3]
  right_indicators : ∀ i, isFirstOddOrEven (grid i 3) [grid i 2, grid i 1, grid i 0]
  bottom_indicators : ∀ j, isFirstOddOrEven (grid 3 j) [grid 2 j, grid 1 j, grid 0 j]

/-- The main theorem stating that the third row must be [4, 2, 1, 3] -/
theorem third_row_is_4213 (grid : Grid) (h : GridConstraints grid) :
  (grid 2 0 = 4) ∧ (grid 2 1 = 2) ∧ (grid 2 2 = 1) ∧ (grid 2 3 = 3) := by
  sorry

end NUMINAMATH_CALUDE_third_row_is_4213_l2268_226884


namespace NUMINAMATH_CALUDE_f_strictly_decreasing_on_interval_l2268_226865

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 3*x^2 - 9*x + 1

-- Theorem statement
theorem f_strictly_decreasing_on_interval :
  ∀ x ∈ Set.Ioo (-1 : ℝ) 3, StrictMonoOn f (Set.Ioo (-1 : ℝ) 3) := by
  sorry


end NUMINAMATH_CALUDE_f_strictly_decreasing_on_interval_l2268_226865


namespace NUMINAMATH_CALUDE_triangle_similarity_problem_l2268_226898

theorem triangle_similarity_problem (DC CB AD AB ED : ℝ) (h1 : DC = 12) (h2 : CB = 9) 
  (h3 : AB = (1/3) * AD) (h4 : ED = (3/4) * AD) : 
  ∃ FC : ℝ, FC = 14.625 := by
  sorry

end NUMINAMATH_CALUDE_triangle_similarity_problem_l2268_226898


namespace NUMINAMATH_CALUDE_cos_half_angle_l2268_226890

theorem cos_half_angle (θ : ℝ) (h1 : |Real.cos θ| = (1 : ℝ) / 5) 
  (h2 : (7 : ℝ) * Real.pi / 2 < θ) (h3 : θ < 4 * Real.pi) : 
  Real.cos (θ / 2) = Real.sqrt 15 / 5 := by
  sorry

end NUMINAMATH_CALUDE_cos_half_angle_l2268_226890


namespace NUMINAMATH_CALUDE_angle_relationship_l2268_226893

theorem angle_relationship (α β : Real) 
  (h1 : 0 < α ∧ α < π/2) 
  (h2 : 0 < β ∧ β < π/2) 
  (h3 : 2 * Real.sin α = Real.sin α * Real.cos β + Real.cos α * Real.sin β) : 
  α < β := by
sorry

end NUMINAMATH_CALUDE_angle_relationship_l2268_226893


namespace NUMINAMATH_CALUDE_distance_between_A_and_B_l2268_226858

def A : ℝ × ℝ := (-3, 1)
def B : ℝ × ℝ := (6, -4)

theorem distance_between_A_and_B : Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = Real.sqrt 106 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_A_and_B_l2268_226858


namespace NUMINAMATH_CALUDE_modular_inverse_31_mod_35_l2268_226889

theorem modular_inverse_31_mod_35 : ∃ x : ℕ, x ≤ 34 ∧ (31 * x) % 35 = 1 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_modular_inverse_31_mod_35_l2268_226889


namespace NUMINAMATH_CALUDE_train_speed_l2268_226861

/-- Given a train of length 360 m passing a platform of length 130 m in 39.2 seconds,
    prove that the speed of the train is 45 km/hr. -/
theorem train_speed (train_length platform_length time_to_pass : ℝ) 
  (h1 : train_length = 360)
  (h2 : platform_length = 130)
  (h3 : time_to_pass = 39.2) : 
  (train_length + platform_length) / time_to_pass * 3.6 = 45 := by
  sorry

#check train_speed

end NUMINAMATH_CALUDE_train_speed_l2268_226861


namespace NUMINAMATH_CALUDE_sum_abc_equals_negative_three_l2268_226846

theorem sum_abc_equals_negative_three
  (a b c : ℝ)
  (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c)
  (h_common_root1 : ∃ x : ℝ, x^2 + a*x + 1 = 0 ∧ x^2 + b*x + c = 0)
  (h_common_root2 : ∃ x : ℝ, x^2 + x + a = 0 ∧ x^2 + c*x + b = 0) :
  a + b + c = -3 :=
by sorry

end NUMINAMATH_CALUDE_sum_abc_equals_negative_three_l2268_226846


namespace NUMINAMATH_CALUDE_number_solution_l2268_226802

theorem number_solution (x : ℝ) : 0.6 * x = 0.3 * 10 + 27 → x = 50 := by
  sorry

end NUMINAMATH_CALUDE_number_solution_l2268_226802


namespace NUMINAMATH_CALUDE_zander_sand_lorries_l2268_226856

/-- Represents the construction materials purchase scenario --/
structure ConstructionPurchase where
  total_payment : ℕ
  cement_bags : ℕ
  cement_price_per_bag : ℕ
  sand_tons_per_lorry : ℕ
  sand_price_per_ton : ℕ

/-- Calculates the number of lorries of sand purchased --/
def sand_lorries (purchase : ConstructionPurchase) : ℕ :=
  let cement_cost := purchase.cement_bags * purchase.cement_price_per_bag
  let sand_cost := purchase.total_payment - cement_cost
  let sand_price_per_lorry := purchase.sand_tons_per_lorry * purchase.sand_price_per_ton
  sand_cost / sand_price_per_lorry

/-- Theorem stating that for the given purchase scenario, the number of sand lorries is 20 --/
theorem zander_sand_lorries :
  let purchase := ConstructionPurchase.mk 13000 500 10 10 40
  sand_lorries purchase = 20 := by
  sorry

end NUMINAMATH_CALUDE_zander_sand_lorries_l2268_226856


namespace NUMINAMATH_CALUDE_largest_three_digit_special_number_l2268_226870

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def distinct_nonzero_digits (n : ℕ) : Prop :=
  let hundreds := n / 100
  let tens := (n % 100) / 10
  let ones := n % 10
  hundreds ≠ 0 ∧ hundreds ≠ tens ∧ hundreds ≠ ones ∧ tens ≠ ones ∧ tens ≠ 0 ∧ ones ≠ 0

def sum_of_digits (n : ℕ) : ℕ :=
  (n / 100) + ((n % 100) / 10) + (n % 10)

def divisible_by_digits (n : ℕ) : Prop :=
  let hundreds := n / 100
  let tens := (n % 100) / 10
  let ones := n % 10
  n % hundreds = 0 ∧ (tens ≠ 0 → n % tens = 0) ∧ (ones ≠ 0 → n % ones = 0)

theorem largest_three_digit_special_number :
  ∀ n : ℕ, 100 ≤ n → n < 1000 →
    (distinct_nonzero_digits n ∧
     is_prime (sum_of_digits n) ∧
     divisible_by_digits n) →
    n ≤ 963 :=
sorry

end NUMINAMATH_CALUDE_largest_three_digit_special_number_l2268_226870


namespace NUMINAMATH_CALUDE_largest_k_for_real_roots_l2268_226847

theorem largest_k_for_real_roots (k : ℤ) : 
  (∃ x : ℝ, x * (k * x + 1) - x^2 + 3 = 0) → 
  k ≠ 1 → 
  k ≤ 0 :=
by sorry

end NUMINAMATH_CALUDE_largest_k_for_real_roots_l2268_226847


namespace NUMINAMATH_CALUDE_problem_statement_l2268_226814

theorem problem_statement : (2002^3 + 4 * 2002^2 + 6006) / (2002^2 + 2002) = 2005 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2268_226814


namespace NUMINAMATH_CALUDE_natural_number_decomposition_l2268_226866

theorem natural_number_decomposition (x y z : ℕ) (h : x * y = z^2 + 1) :
  ∃ (a b c d : ℤ), (x : ℤ) = a^2 + b^2 ∧ (y : ℤ) = c^2 + d^2 ∧ (z : ℤ) = a * c + b * d := by
  sorry

end NUMINAMATH_CALUDE_natural_number_decomposition_l2268_226866


namespace NUMINAMATH_CALUDE_events_mutually_exclusive_not_opposite_l2268_226859

-- Define the set of people
inductive Person : Type
| A : Person
| B : Person
| C : Person
| D : Person

-- Define the set of cards
inductive Card : Type
| Red : Card
| Black : Card
| Blue : Card
| White : Card

-- Define a distribution as a function from Person to Card
def Distribution := Person → Card

-- Define the event "Person A gets the red card"
def event_A_red (d : Distribution) : Prop := d Person.A = Card.Red

-- Define the event "Person B gets the red card"
def event_B_red (d : Distribution) : Prop := d Person.B = Card.Red

-- State the theorem
theorem events_mutually_exclusive_not_opposite :
  ∃ (d : Distribution),
    (∀ (p : Person), ∃! (c : Card), d p = c) ∧  -- Each person gets exactly one card
    (∀ (c : Card), ∃! (p : Person), d p = c) ∧  -- Each card is given to exactly one person
    (¬(event_A_red d ∧ event_B_red d)) ∧        -- Events are mutually exclusive
    ¬(event_A_red d ↔ ¬event_B_red d)           -- Events are not opposite
  := by sorry

end NUMINAMATH_CALUDE_events_mutually_exclusive_not_opposite_l2268_226859


namespace NUMINAMATH_CALUDE_inequality_solution_range_l2268_226854

theorem inequality_solution_range (a : ℝ) (h_a : a > 0) :
  (∃ x : ℝ, x ∈ Set.Icc (-1) 2 ∧ 
   Real.exp 2 * Real.log a - Real.exp 2 * x + x - Real.log a ≥ 2 * a / Real.exp x - 2) ↔
  a ∈ Set.Icc (1 / Real.exp 1) (Real.exp 4) :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_range_l2268_226854


namespace NUMINAMATH_CALUDE_min_n_constant_term_is_correct_l2268_226872

/-- The minimum natural number n for which (x^2 + 1/(2x^3))^n contains a constant term -/
def min_n_constant_term : ℕ := 5

/-- Predicate to check if the expansion contains a constant term -/
def has_constant_term (n : ℕ) : Prop :=
  ∃ r : ℕ, 2 * n = 5 * r

theorem min_n_constant_term_is_correct :
  (∀ k < min_n_constant_term, ¬ has_constant_term k) ∧
  has_constant_term min_n_constant_term := by sorry

end NUMINAMATH_CALUDE_min_n_constant_term_is_correct_l2268_226872


namespace NUMINAMATH_CALUDE_shoe_probabilities_l2268_226849

-- Define the type for shoes
inductive Shoe
| left : Shoe
| right : Shoe

-- Define a pair of shoes
structure ShoePair :=
  (left : Shoe)
  (right : Shoe)

-- Define the cabinet with 3 pairs of shoes
def cabinet : Finset ShoePair := sorry

-- Define the sample space of choosing 2 shoes
def sampleSpace : Finset (Shoe × Shoe) := sorry

-- Event A: The taken out shoes do not form a pair
def eventA : Finset (Shoe × Shoe) := sorry

-- Event B: Both taken out shoes are for the same foot
def eventB : Finset (Shoe × Shoe) := sorry

-- Event C: One shoe is for the left foot and the other is for the right foot, but they do not form a pair
def eventC : Finset (Shoe × Shoe) := sorry

theorem shoe_probabilities :
  (Finset.card eventA : ℚ) / Finset.card sampleSpace = 4 / 5 ∧
  (Finset.card eventB : ℚ) / Finset.card sampleSpace = 2 / 5 ∧
  (Finset.card eventC : ℚ) / Finset.card sampleSpace = 2 / 5 :=
sorry

end NUMINAMATH_CALUDE_shoe_probabilities_l2268_226849


namespace NUMINAMATH_CALUDE_straight_flush_probability_l2268_226832

/-- Represents a standard deck of cards -/
def StandardDeck : ℕ := 52

/-- Represents the number of cards in a poker hand -/
def PokerHand : ℕ := 5

/-- Represents the number of possible starting ranks for a straight flush -/
def StartingRanks : ℕ := 10

/-- Represents the number of suits in a standard deck -/
def Suits : ℕ := 4

/-- Calculates the number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- Represents the total number of possible 5-card hands -/
def TotalHands : ℕ := choose StandardDeck PokerHand

/-- Represents the total number of straight flushes -/
def StraightFlushes : ℕ := StartingRanks * Suits

/-- Theorem: The probability of drawing a straight flush is 1/64,974 -/
theorem straight_flush_probability :
  StraightFlushes / TotalHands = 1 / 64974 := by sorry

end NUMINAMATH_CALUDE_straight_flush_probability_l2268_226832


namespace NUMINAMATH_CALUDE_circle_pair_relation_infinite_quadrilaterals_l2268_226871

/-- A structure representing a pair of circles with a quadrilateral inscribed in one and circumscribed around the other. -/
structure CirclePair where
  R : ℝ  -- Radius of the larger circle
  r : ℝ  -- Radius of the smaller circle
  d : ℝ  -- Distance between the centers of the circles
  h_positive_R : R > 0
  h_positive_r : r > 0
  h_positive_d : d > 0
  h_d_less_R : d < R

/-- The main theorem stating the relationship between the radii and distance of the circles. -/
theorem circle_pair_relation (cp : CirclePair) :
  1 / (cp.R + cp.d)^2 + 1 / (cp.R - cp.d)^2 = 1 / cp.r^2 :=
sorry

/-- There exist infinitely many quadrilaterals satisfying the conditions. -/
theorem infinite_quadrilaterals (R r d : ℝ) (h_R : R > 0) (h_r : r > 0) (h_d : d > 0) (h_d_R : d < R) :
  ∃ (cp : CirclePair), cp.R = R ∧ cp.r = r ∧ cp.d = d :=
sorry

end NUMINAMATH_CALUDE_circle_pair_relation_infinite_quadrilaterals_l2268_226871


namespace NUMINAMATH_CALUDE_complex_root_problem_l2268_226801

theorem complex_root_problem (a b c : ℂ) (h_real : b.im = 0) 
  (h_sum : a + b + c = 4)
  (h_prod_sum : a * b + b * c + c * a = 5)
  (h_prod : a * b * c = 6) :
  b = 1 := by sorry

end NUMINAMATH_CALUDE_complex_root_problem_l2268_226801


namespace NUMINAMATH_CALUDE_cookie_banana_price_ratio_l2268_226826

theorem cookie_banana_price_ratio :
  ∀ (cookie_price banana_price : ℝ),
  cookie_price > 0 →
  banana_price > 0 →
  6 * cookie_price + 5 * banana_price > 0 →
  3 * (6 * cookie_price + 5 * banana_price) = 3 * cookie_price + 27 * banana_price →
  cookie_price / banana_price = 4 / 5 := by
sorry

end NUMINAMATH_CALUDE_cookie_banana_price_ratio_l2268_226826


namespace NUMINAMATH_CALUDE_perfect_square_binomial_l2268_226804

theorem perfect_square_binomial (k : ℝ) : 
  (∃ a b : ℝ, ∀ x : ℝ, x^2 - 20*x + k = (x + a)^2) ↔ k = 100 := by sorry

end NUMINAMATH_CALUDE_perfect_square_binomial_l2268_226804


namespace NUMINAMATH_CALUDE_zeros_of_f_shifted_l2268_226835

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 1

-- State the theorem
theorem zeros_of_f_shifted (x : ℝ) :
  f (x - 1) = 0 ↔ x = 0 ∨ x = 2 :=
by sorry

end NUMINAMATH_CALUDE_zeros_of_f_shifted_l2268_226835
