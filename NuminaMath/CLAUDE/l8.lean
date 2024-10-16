import Mathlib

namespace NUMINAMATH_CALUDE_exercise_time_is_9_25_hours_l8_802

/-- Represents the exercise schedule for a week -/
structure ExerciseSchedule where
  initial_jogging : ℕ
  jogging_increment : ℕ
  swimming_increment : ℕ
  wednesday_reduction : ℕ
  friday_kickboxing : ℕ
  kickboxing_multiplier : ℕ

/-- Calculates the total exercise time for the week -/
def total_exercise_time (schedule : ExerciseSchedule) : ℚ :=
  sorry

/-- Theorem stating that the total exercise time is 9.25 hours -/
theorem exercise_time_is_9_25_hours (schedule : ExerciseSchedule) 
  (h1 : schedule.initial_jogging = 30)
  (h2 : schedule.jogging_increment = 5)
  (h3 : schedule.swimming_increment = 10)
  (h4 : schedule.wednesday_reduction = 10)
  (h5 : schedule.friday_kickboxing = 20)
  (h6 : schedule.kickboxing_multiplier = 2) :
  total_exercise_time schedule = 9.25 := by
  sorry

end NUMINAMATH_CALUDE_exercise_time_is_9_25_hours_l8_802


namespace NUMINAMATH_CALUDE_intersection_distance_approx_l8_898

-- Define the centers of the circles
def A : ℝ × ℝ := (0, 0)
def B : ℝ × ℝ := (3, 0)
def C : ℝ × ℝ := (4, 0)
def D : ℝ × ℝ := (5, 0)

-- Define the radii of the circles
def radius_A : ℝ := 2
def radius_B : ℝ := 2
def radius_C : ℝ := 3
def radius_D : ℝ := 3

-- Define the equations of the circles
def circle_A (x y : ℝ) : Prop := x^2 + y^2 = radius_A^2
def circle_C (x y : ℝ) : Prop := (x - C.1)^2 + y^2 = radius_C^2
def circle_D (x y : ℝ) : Prop := (x - D.1)^2 + y^2 = radius_D^2

-- Define the intersection points
def B' : ℝ × ℝ := sorry
def D' : ℝ × ℝ := sorry

-- State the theorem
theorem intersection_distance_approx :
  ∃ ε > 0, abs (Real.sqrt ((B'.1 - D'.1)^2 + (B'.2 - D'.2)^2) - 0.8) < ε :=
sorry

end NUMINAMATH_CALUDE_intersection_distance_approx_l8_898


namespace NUMINAMATH_CALUDE_camel_cost_l8_834

theorem camel_cost (camel horse ox elephant : ℕ → ℚ) 
  (h1 : 10 * camel 1 = 24 * horse 1)
  (h2 : 16 * horse 1 = 4 * ox 1)
  (h3 : 6 * ox 1 = 4 * elephant 1)
  (h4 : 10 * elephant 1 = 120000) :
  camel 1 = 4800 := by
  sorry

end NUMINAMATH_CALUDE_camel_cost_l8_834


namespace NUMINAMATH_CALUDE_total_campers_l8_866

def basketball_campers : ℕ := 24
def football_campers : ℕ := 32
def soccer_campers : ℕ := 32

theorem total_campers : basketball_campers + football_campers + soccer_campers = 88 := by
  sorry

end NUMINAMATH_CALUDE_total_campers_l8_866


namespace NUMINAMATH_CALUDE_enclosure_probability_l8_804

def is_valid_configuration (c₁ c₂ c₃ d₁ d₂ d₃ : ℕ) : Prop :=
  d₁ ≥ 2 * c₁ ∧ d₁ > d₂ ∧ d₂ > d₃ ∧ c₁ > c₂ ∧ c₂ > c₃ ∧
  d₁ > c₁ ∧ d₂ > c₂ ∧ d₃ > c₃

def probability_of_valid_configuration : ℚ :=
  1 / 2

theorem enclosure_probability :
  ∀ (S : Finset ℕ) (c₁ c₂ c₃ d₁ d₂ d₃ : ℕ),
    S = Finset.range 100 →
    c₁ ∈ S ∧ c₂ ∈ S ∧ c₃ ∈ S →
    c₁ ≠ c₂ ∧ c₂ ≠ c₃ ∧ c₁ ≠ c₃ →
    d₁ ∈ S.erase c₁ \ {c₂, c₃} ∧ d₂ ∈ S.erase c₁ \ {c₂, c₃, d₁} ∧ d₃ ∈ S.erase c₁ \ {c₂, c₃, d₁, d₂} →
    probability_of_valid_configuration = 1 / 2 :=
sorry

end NUMINAMATH_CALUDE_enclosure_probability_l8_804


namespace NUMINAMATH_CALUDE_extremum_implies_a_equals_3_l8_874

/-- The function f(x) = x³ + 5x² + ax attains an extremum at x = -3 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + 5*x^2 + a*x

/-- The derivative of f with respect to x -/
def f' (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + 10*x + a

theorem extremum_implies_a_equals_3 (a : ℝ) :
  (f' a (-3) = 0) → a = 3 := by
  sorry

#check extremum_implies_a_equals_3

end NUMINAMATH_CALUDE_extremum_implies_a_equals_3_l8_874


namespace NUMINAMATH_CALUDE_jellybean_problem_l8_888

theorem jellybean_problem (initial_quantity : ℝ) : 
  (initial_quantity * (1 - 0.3)^4 = 48) → initial_quantity = 200 := by
  sorry

end NUMINAMATH_CALUDE_jellybean_problem_l8_888


namespace NUMINAMATH_CALUDE_parameterized_line_matches_equation_l8_862

/-- A line parameterized by a point and a direction vector -/
structure ParametricLine (n : Type*) [NormedAddCommGroup n] where
  point : n
  direction : n

/-- The equation of a line in slope-intercept form -/
structure SlopeInterceptLine (α : Type*) [Field α] where
  slope : α
  intercept : α

def line_equation (l : SlopeInterceptLine ℝ) (x : ℝ) : ℝ :=
  l.slope * x + l.intercept

theorem parameterized_line_matches_equation 
  (r k : ℝ) 
  (param_line : ParametricLine (Fin 2 → ℝ))
  (slope_intercept_line : SlopeInterceptLine ℝ) :
  param_line.point = ![r, 2] ∧ 
  param_line.direction = ![3, k] ∧
  slope_intercept_line.slope = 2 ∧
  slope_intercept_line.intercept = -5 →
  r = 7/2 ∧ k = 6 := by
  sorry

end NUMINAMATH_CALUDE_parameterized_line_matches_equation_l8_862


namespace NUMINAMATH_CALUDE_balls_color_probability_l8_846

def num_balls : ℕ := 6
def probability_black : ℚ := 1/2
def probability_white : ℚ := 1/2

theorem balls_color_probability :
  let favorable_outcomes := (num_balls.choose (num_balls / 2))
  let total_outcomes := 2^num_balls
  (favorable_outcomes : ℚ) / total_outcomes = 5/16 := by
sorry

end NUMINAMATH_CALUDE_balls_color_probability_l8_846


namespace NUMINAMATH_CALUDE_reversed_digits_multiple_l8_837

/-- Given a two-digit number that is k times the sum of its digits, 
    prove that the number formed by reversing its digits is (11 - k) times the sum of its digits. -/
theorem reversed_digits_multiple (k : ℕ) (u v : ℕ) : 
  (u ≤ 9 ∧ v ≤ 9 ∧ u ≠ 0) → 
  (10 * u + v = k * (u + v)) → 
  (10 * v + u = (11 - k) * (u + v)) :=
by sorry

end NUMINAMATH_CALUDE_reversed_digits_multiple_l8_837


namespace NUMINAMATH_CALUDE_negative_two_to_fourth_power_l8_819

theorem negative_two_to_fourth_power : -2 * 2 * 2 * 2 = -2^4 := by
  sorry

end NUMINAMATH_CALUDE_negative_two_to_fourth_power_l8_819


namespace NUMINAMATH_CALUDE_value_of_x_l8_808

theorem value_of_x (x y z : ℝ) : 
  x = (1/2) * y → 
  y = (1/4) * z → 
  z = 80 → 
  x = 10 := by
sorry

end NUMINAMATH_CALUDE_value_of_x_l8_808


namespace NUMINAMATH_CALUDE_bank_account_deposit_fraction_l8_815

theorem bank_account_deposit_fraction (B : ℝ) (f : ℝ) : 
  B > 0 →
  (3/5) * B = B - 400 →
  600 + f * 600 = 750 →
  f = 1/4 := by
sorry

end NUMINAMATH_CALUDE_bank_account_deposit_fraction_l8_815


namespace NUMINAMATH_CALUDE_leigh_has_16_seashells_l8_876

/-- The number of seashells Leigh has, given the conditions of the problem -/
def leighs_seashells : ℕ :=
  let mimis_shells := 2 * 12  -- 2 dozen
  let kyles_shells := 2 * mimis_shells  -- twice as many as Mimi
  kyles_shells / 3  -- one-third of Kyle's shells

/-- Theorem stating that Leigh has 16 seashells -/
theorem leigh_has_16_seashells : leighs_seashells = 16 := by
  sorry

end NUMINAMATH_CALUDE_leigh_has_16_seashells_l8_876


namespace NUMINAMATH_CALUDE_limit_of_f_is_one_fourth_l8_812

def C (n : ℕ) (k : ℕ) : ℕ := Nat.choose n k

def f (n : ℕ) : ℚ := (C n 2 : ℚ) / (2 * n^2 + n : ℚ)

theorem limit_of_f_is_one_fourth :
  ∀ ε > 0, ∃ N, ∀ n ≥ N, |f n - 1/4| < ε :=
sorry

end NUMINAMATH_CALUDE_limit_of_f_is_one_fourth_l8_812


namespace NUMINAMATH_CALUDE_smoking_lung_disease_relation_l8_864

/-- Represents the Chi-square statistic -/
def K_squared : ℝ := 5.231

/-- The probability that K^2 is greater than or equal to 3.841 -/
def P_3_841 : ℝ := 0.05

/-- The probability that K^2 is greater than or equal to 6.635 -/
def P_6_635 : ℝ := 0.01

/-- The confidence level for the relationship between smoking and lung disease -/
def confidence_level : ℝ := 1 - P_3_841

/-- Theorem stating that there is more than 95% confidence that smoking is related to lung disease -/
theorem smoking_lung_disease_relation :
  K_squared > 3.841 ∧ confidence_level > 0.95 := by sorry

end NUMINAMATH_CALUDE_smoking_lung_disease_relation_l8_864


namespace NUMINAMATH_CALUDE_radio_selling_price_l8_838

def purchase_price : ℚ := 232
def overhead_expenses : ℚ := 15
def profit_percent : ℚ := 21.457489878542503

def total_cost_price : ℚ := purchase_price + overhead_expenses

def profit_amount : ℚ := (profit_percent / 100) * total_cost_price

def selling_price : ℚ := total_cost_price + profit_amount

theorem radio_selling_price : 
  ∃ (sp : ℚ), sp = selling_price ∧ round sp = 300 := by sorry

end NUMINAMATH_CALUDE_radio_selling_price_l8_838


namespace NUMINAMATH_CALUDE_complex_magnitude_problem_l8_875

theorem complex_magnitude_problem (z : ℂ) (h : z * (1 - Complex.I) = 1 + Complex.I) :
  Complex.abs z = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_problem_l8_875


namespace NUMINAMATH_CALUDE_jack_marbles_remaining_l8_800

/-- Given Jack starts with 62 marbles and shares 33 marbles, prove that he ends up with 29 marbles. -/
theorem jack_marbles_remaining (initial_marbles : ℕ) (shared_marbles : ℕ) 
  (h1 : initial_marbles = 62)
  (h2 : shared_marbles = 33) :
  initial_marbles - shared_marbles = 29 := by
  sorry

end NUMINAMATH_CALUDE_jack_marbles_remaining_l8_800


namespace NUMINAMATH_CALUDE_cube_properties_l8_877

-- Define the surface area of the cube
def surface_area : ℝ := 150

-- Define the relationship between surface area and edge length
def edge_length (s : ℝ) : Prop := 6 * s^2 = surface_area

-- Define the volume of a cube given its edge length
def volume (s : ℝ) : ℝ := s^3

-- Theorem statement
theorem cube_properties :
  ∃ (s : ℝ), edge_length s ∧ s = 5 ∧ volume s = 125 :=
sorry

end NUMINAMATH_CALUDE_cube_properties_l8_877


namespace NUMINAMATH_CALUDE_subset_iff_a_in_range_l8_894

def A : Set ℝ := {x | x^2 - 4*x + 3 < 0}
def B (a : ℝ) : Set ℝ := {x | 2^(1-x) + a ≤ 0 ∧ x^2 - 2*(a+7)*x + 5 ≤ 0}

theorem subset_iff_a_in_range :
  ∀ a : ℝ, A ⊆ B a ↔ a ∈ Set.Icc (-4 : ℝ) (-1 : ℝ) :=
sorry

end NUMINAMATH_CALUDE_subset_iff_a_in_range_l8_894


namespace NUMINAMATH_CALUDE_joggers_meeting_l8_856

def lap_time_cathy : ℕ := 5
def lap_time_david : ℕ := 9
def lap_time_elena : ℕ := 8

def meeting_time : ℕ := 360
def cathy_laps : ℕ := 72

theorem joggers_meeting :
  (meeting_time % lap_time_cathy = 0) ∧
  (meeting_time % lap_time_david = 0) ∧
  (meeting_time % lap_time_elena = 0) ∧
  (∀ t : ℕ, t < meeting_time →
    ¬(t % lap_time_cathy = 0 ∧ t % lap_time_david = 0 ∧ t % lap_time_elena = 0)) ∧
  (cathy_laps = meeting_time / lap_time_cathy) :=
by sorry

end NUMINAMATH_CALUDE_joggers_meeting_l8_856


namespace NUMINAMATH_CALUDE_function_range_l8_895

noncomputable def f (x : ℝ) : ℝ := 1 / Real.exp x - Real.exp x + 2 * x - (1 / 3) * x^3

theorem function_range (a : ℝ) : f (3 * a^2) + f (2 * a - 1) ≥ 0 → a ∈ Set.Icc (-1) (1/3) :=
by sorry

end NUMINAMATH_CALUDE_function_range_l8_895


namespace NUMINAMATH_CALUDE_shaded_area_of_overlapping_sectors_l8_861

/-- The area of the shaded region formed by two overlapping sectors of a circle -/
theorem shaded_area_of_overlapping_sectors (r : ℝ) (θ : ℝ) (h_r : r = 15) (h_θ : θ = 45 * π / 180) :
  let sector_area := θ / (2 * π) * π * r^2
  let triangle_area := r^2 * Real.sin θ / 2
  2 * (sector_area - triangle_area) = 56.25 * π - 112.5 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_of_overlapping_sectors_l8_861


namespace NUMINAMATH_CALUDE_macaroon_problem_l8_844

/-- Proves that the initial number of macaroons is 12 given the problem conditions -/
theorem macaroon_problem (weight_per_macaroon : ℕ) (num_bags : ℕ) (remaining_weight : ℕ) : 
  weight_per_macaroon = 5 →
  num_bags = 4 →
  remaining_weight = 45 →
  ∃ (initial_macaroons : ℕ),
    initial_macaroons = 12 ∧
    initial_macaroons % num_bags = 0 ∧
    (initial_macaroons / num_bags) * weight_per_macaroon * (num_bags - 1) = remaining_weight :=
by sorry

end NUMINAMATH_CALUDE_macaroon_problem_l8_844


namespace NUMINAMATH_CALUDE_quadratic_solution_sum_l8_878

theorem quadratic_solution_sum (a b : ℝ) : 
  (∀ x : ℂ, (6 * x^2 + 7 = 5 * x - 11) ↔ (x = a + b * I ∨ x = a - b * I)) →
  a + b^2 = 467/144 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_sum_l8_878


namespace NUMINAMATH_CALUDE_string_average_length_l8_867

theorem string_average_length : 
  let string1 : ℝ := 1.5
  let string2 : ℝ := 4.5
  let average := (string1 + string2) / 2
  average = 3 := by
sorry

end NUMINAMATH_CALUDE_string_average_length_l8_867


namespace NUMINAMATH_CALUDE_fraction_calculation_l8_820

theorem fraction_calculation : (1 - 1/4) / (1 - 1/5) = 15/16 := by
  sorry

end NUMINAMATH_CALUDE_fraction_calculation_l8_820


namespace NUMINAMATH_CALUDE_sequence_problem_l8_872

theorem sequence_problem (x : ℕ → ℤ) 
  (h1 : x 1 = 8)
  (h2 : x 4 = 2)
  (h3 : ∀ n : ℕ, n > 0 → x (n + 2) + x n = 2 * x (n + 1)) :
  x 10 = -10 := by sorry

end NUMINAMATH_CALUDE_sequence_problem_l8_872


namespace NUMINAMATH_CALUDE_square_of_binomial_l8_891

theorem square_of_binomial (k : ℝ) : 
  (∃ a b : ℝ, ∀ x : ℝ, x^2 - 20*x + k = (x + a)^2 + b^2) → k = 100 := by
  sorry

end NUMINAMATH_CALUDE_square_of_binomial_l8_891


namespace NUMINAMATH_CALUDE_mary_balloons_l8_892

/-- Given that Nancy has 7 black balloons and Mary has 4 times more black balloons than Nancy,
    prove that Mary has 28 black balloons. -/
theorem mary_balloons (nancy_balloons : ℕ) (mary_multiplier : ℕ) 
    (h1 : nancy_balloons = 7)
    (h2 : mary_multiplier = 4) : 
  nancy_balloons * mary_multiplier = 28 := by
  sorry

end NUMINAMATH_CALUDE_mary_balloons_l8_892


namespace NUMINAMATH_CALUDE_integer_division_property_l8_857

theorem integer_division_property (n : ℕ+) : 
  (∃ k : ℤ, (2^(n : ℕ) + 1 : ℤ) = k * (n : ℤ)^2) ↔ n = 1 ∨ n = 3 := by
sorry

end NUMINAMATH_CALUDE_integer_division_property_l8_857


namespace NUMINAMATH_CALUDE_acute_angle_condition_perpendicular_condition_l8_854

/-- Define a 2D vector -/
def Vector2D := Fin 2 → ℝ

/-- The dot product of two 2D vectors -/
def dot_product (v w : Vector2D) : ℝ := (v 0) * (w 0) + (v 1) * (w 1)

/-- Vector a -/
def a : Vector2D := ![1, 2]

/-- Vector b parameterized by x -/
def b (x : ℝ) : Vector2D := ![x, 1]

/-- Theorem: Vectors a and b form an acute angle iff x ∈ (-2, ∞) \ {1/2} -/
theorem acute_angle_condition (x : ℝ) :
  (dot_product a (b x) > 0 ∧ x ≠ 1/2) ↔ x > -2 ∧ x ≠ 1/2 :=
sorry

/-- Theorem: (a + 2b) is perpendicular to (2a - b) iff x = 7/2 -/
theorem perpendicular_condition (x : ℝ) :
  dot_product (λ i => a i + 2 * (b x i)) (λ i => 2 * a i - b x i) = 0 ↔ x = 7/2 :=
sorry

end NUMINAMATH_CALUDE_acute_angle_condition_perpendicular_condition_l8_854


namespace NUMINAMATH_CALUDE_coin_flip_sequences_l8_859

theorem coin_flip_sequences (n : ℕ) : n = 10 → (2 : ℕ) ^ n = 1024 := by
  sorry

end NUMINAMATH_CALUDE_coin_flip_sequences_l8_859


namespace NUMINAMATH_CALUDE_total_after_discount_rounded_l8_839

-- Define the purchases
def purchase1 : ℚ := 215 / 100
def purchase2 : ℚ := 749 / 100
def purchase3 : ℚ := 1285 / 100

-- Define the discount rate
def discount_rate : ℚ := 1 / 10

-- Function to apply discount to the most expensive item
def apply_discount (p1 p2 p3 : ℚ) (rate : ℚ) : ℚ :=
  let max_purchase := max p1 (max p2 p3)
  let discounted_max := max_purchase * (1 - rate)
  if p1 == max_purchase then discounted_max + p2 + p3
  else if p2 == max_purchase then p1 + discounted_max + p3
  else p1 + p2 + discounted_max

-- Function to round to nearest integer
def round_to_nearest (x : ℚ) : ℤ :=
  ⌊x + 1/2⌋

-- Theorem statement
theorem total_after_discount_rounded :
  round_to_nearest (apply_discount purchase1 purchase2 purchase3 discount_rate) = 21 := by
  sorry

end NUMINAMATH_CALUDE_total_after_discount_rounded_l8_839


namespace NUMINAMATH_CALUDE_statement_is_proposition_l8_832

-- Define what a proposition is
def is_proposition (s : Prop) : Prop := True

-- Define the statement we're examining
def statement : Prop := ∀ a : ℤ, Prime a → Odd a

-- Theorem stating that our statement is a proposition
theorem statement_is_proposition : is_proposition statement := by sorry

end NUMINAMATH_CALUDE_statement_is_proposition_l8_832


namespace NUMINAMATH_CALUDE_xoons_are_zeefs_and_yamps_l8_855

-- Define the types for our sets
variable (U : Type) -- Universal set
variable (Zeef Yamp Xoon Woon : Set U)

-- State the given conditions
variable (h1 : Zeef ⊆ Yamp)
variable (h2 : Xoon ⊆ Yamp)
variable (h3 : Woon ⊆ Zeef)
variable (h4 : Xoon ⊆ Woon)

-- State the theorem to be proved
theorem xoons_are_zeefs_and_yamps : Xoon ⊆ Zeef ∩ Yamp := by
  sorry

end NUMINAMATH_CALUDE_xoons_are_zeefs_and_yamps_l8_855


namespace NUMINAMATH_CALUDE_inequality_proof_l8_807

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a^2 * (b + c - a) + b^2 * (a + c - b) + c^2 * (a + b - c) ≤ 3 * a * b * c := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l8_807


namespace NUMINAMATH_CALUDE_recipe_sugar_requirement_l8_836

/-- The number of cups of sugar Mary has already added to the cake. -/
def sugar_added : ℕ := 10

/-- The number of cups of sugar Mary still needs to add to the cake. -/
def sugar_to_add : ℕ := 1

/-- The total number of cups of sugar required by the recipe. -/
def total_sugar : ℕ := sugar_added + sugar_to_add

/-- The number of cups of flour required by the recipe. -/
def flour_required : ℕ := 9

/-- The number of cups of flour Mary has already added to the cake. -/
def flour_added : ℕ := 12

theorem recipe_sugar_requirement :
  total_sugar = 11 := by sorry

end NUMINAMATH_CALUDE_recipe_sugar_requirement_l8_836


namespace NUMINAMATH_CALUDE_prob_three_odd_six_dice_value_l8_889

/-- The probability of rolling an odd number on a fair 8-sided die -/
def prob_odd_8sided : ℚ := 1/2

/-- The number of ways to choose 3 dice out of 6 -/
def choose_3_from_6 : ℕ := 20

/-- The probability of rolling exactly three odd numbers when rolling six fair 8-sided dice -/
def prob_three_odd_six_dice : ℚ :=
  (choose_3_from_6 : ℚ) * (prob_odd_8sided ^ 3 * (1 - prob_odd_8sided) ^ 3)

theorem prob_three_odd_six_dice_value : prob_three_odd_six_dice = 5/16 := by
  sorry

end NUMINAMATH_CALUDE_prob_three_odd_six_dice_value_l8_889


namespace NUMINAMATH_CALUDE_bridge_length_l8_825

/-- The length of a bridge given train specifications and crossing time -/
theorem bridge_length (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) : 
  train_length = 180 ∧ 
  train_speed_kmh = 45 ∧ 
  crossing_time = 30 →
  (train_speed_kmh * 1000 / 3600) * crossing_time - train_length = 195 := by
sorry

end NUMINAMATH_CALUDE_bridge_length_l8_825


namespace NUMINAMATH_CALUDE_orange_distribution_theorem_l8_824

/-- Represents the number of oranges each person has at a given stage -/
structure OrangeDistribution :=
  (a : ℕ) (b : ℕ) (c : ℕ)

/-- Defines the redistribution rules for oranges -/
def redistribute (d : OrangeDistribution) : OrangeDistribution :=
  let d1 := OrangeDistribution.mk (d.a / 2) (d.b + d.a / 2) d.c
  let d2 := OrangeDistribution.mk d1.a (d1.b * 4 / 5) (d1.c + d1.b / 5)
  OrangeDistribution.mk (d2.a + d2.c / 7) d2.b (d2.c * 6 / 7)

theorem orange_distribution_theorem (initial : OrangeDistribution) :
  initial.a + initial.b + initial.c = 108 →
  let final := redistribute initial
  final.a = final.b ∧ final.b = final.c →
  initial = OrangeDistribution.mk 72 9 27 := by
  sorry

end NUMINAMATH_CALUDE_orange_distribution_theorem_l8_824


namespace NUMINAMATH_CALUDE_tickets_left_kaleb_tickets_left_l8_814

theorem tickets_left (initial_tickets : ℕ) (ticket_cost : ℕ) (spent_on_ride : ℕ) : ℕ :=
  let tickets_used := spent_on_ride / ticket_cost
  initial_tickets - tickets_used

theorem kaleb_tickets_left :
  tickets_left 6 9 27 = 3 := by
  sorry

end NUMINAMATH_CALUDE_tickets_left_kaleb_tickets_left_l8_814


namespace NUMINAMATH_CALUDE_event_C_subset_event_B_l8_828

-- Define the sample space for tossing 3 coins
def SampleSpace := List Bool

-- Define the events A, B, and C
def event_A (outcome : SampleSpace) : Prop := outcome.contains true
def event_B (outcome : SampleSpace) : Prop := outcome.count true ≤ 2
def event_C (outcome : SampleSpace) : Prop := outcome.count true = 0

-- Theorem statement
theorem event_C_subset_event_B : 
  ∀ (outcome : SampleSpace), event_C outcome → event_B outcome :=
by
  sorry


end NUMINAMATH_CALUDE_event_C_subset_event_B_l8_828


namespace NUMINAMATH_CALUDE_find_a_l8_858

-- Define the function f
def f (x : ℝ) : ℝ := sorry

-- State the theorem
theorem find_a : 
  (∀ x : ℝ, f (1/2 * x - 1) = 2*x - 5) → 
  f (7/4) = 6 := by sorry

end NUMINAMATH_CALUDE_find_a_l8_858


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l8_893

theorem hyperbola_eccentricity (m : ℝ) : 
  (∀ x y : ℝ, x^2 / m + y^2 / 2 = 1) →
  (∃ a b c : ℝ, a^2 = 2 ∧ b^2 = -m ∧ c^2 = a^2 + b^2 ∧ c^2 / a^2 = 4) →
  m = -6 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l8_893


namespace NUMINAMATH_CALUDE_sum_of_angles_less_than_90_degrees_l8_843

/-- A line intersecting two perpendicular planes forms angles α and β with these planes. -/
structure LineIntersectingPerpendicularPlanes where
  α : Real
  β : Real

/-- The theorem states that the sum of angles α and β is always less than 90 degrees. -/
theorem sum_of_angles_less_than_90_degrees (l : LineIntersectingPerpendicularPlanes) :
  l.α + l.β < 90 * Real.pi / 180 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_angles_less_than_90_degrees_l8_843


namespace NUMINAMATH_CALUDE_bus_speed_excluding_stoppages_l8_842

/-- Proves that given a bus with a speed of 42 kmph including stoppages
    and stopping for 9.6 minutes per hour, the speed excluding stoppages is 50 kmph. -/
theorem bus_speed_excluding_stoppages
  (speed_with_stoppages : ℝ)
  (stoppage_time : ℝ)
  (h1 : speed_with_stoppages = 42)
  (h2 : stoppage_time = 9.6)
  : (speed_with_stoppages * 60) / (60 - stoppage_time) = 50 := by
  sorry

end NUMINAMATH_CALUDE_bus_speed_excluding_stoppages_l8_842


namespace NUMINAMATH_CALUDE_radioactive_balls_solvable_l8_885

/-- Represents a test strategy for identifying radioactive balls -/
structure TestStrategy where
  -- The number of tests used in the strategy
  num_tests : ℕ
  -- A function that, given the positions of the radioactive balls,
  -- returns true if the strategy successfully identifies both balls
  identifies_balls : Fin 11 → Fin 11 → Prop

/-- Represents the problem of finding radioactive balls -/
def RadioactiveBallsProblem :=
  ∃ (strategy : TestStrategy),
    strategy.num_tests ≤ 7 ∧
    ∀ (pos1 pos2 : Fin 11), pos1 ≠ pos2 →
      strategy.identifies_balls pos1 pos2

/-- The main theorem stating that the radioactive balls problem can be solved -/
theorem radioactive_balls_solvable : RadioactiveBallsProblem := by
  sorry


end NUMINAMATH_CALUDE_radioactive_balls_solvable_l8_885


namespace NUMINAMATH_CALUDE_lg_meaningful_iff_first_or_second_quadrant_l8_813

open Real

-- Define the meaningful condition for lg(cos θ · tan θ)
def is_meaningful (θ : ℝ) : Prop :=
  sin θ > 0 ∧ sin θ ≠ 1

-- Define the first and second quadrants
def in_first_or_second_quadrant (θ : ℝ) : Prop :=
  0 < θ ∧ θ < π

-- Theorem statement
theorem lg_meaningful_iff_first_or_second_quadrant (θ : ℝ) :
  is_meaningful θ ↔ in_first_or_second_quadrant θ :=
sorry

end NUMINAMATH_CALUDE_lg_meaningful_iff_first_or_second_quadrant_l8_813


namespace NUMINAMATH_CALUDE_students_history_not_statistics_l8_845

/-- Given a group of students, prove the number taking history but not statistics -/
theorem students_history_not_statistics 
  (total : ℕ) 
  (history : ℕ) 
  (statistics : ℕ) 
  (history_or_statistics : ℕ) 
  (h_total : total = 90)
  (h_history : history = 36)
  (h_statistics : statistics = 30)
  (h_history_or_statistics : history_or_statistics = 59) :
  history - (history + statistics - history_or_statistics) = 29 := by
  sorry

end NUMINAMATH_CALUDE_students_history_not_statistics_l8_845


namespace NUMINAMATH_CALUDE_computer_price_equation_l8_873

/-- Represents the relationship between the original price, tax rate, and discount rate
    for a computer with a 30% price increase and final price of $351. -/
theorem computer_price_equation (c t d : ℝ) : 
  1.30 * c * (100 + t) * (100 - d) = 3510000 ↔ 
  (c * (1 + 0.3) * (1 + t / 100) * (1 - d / 100) = 351) :=
sorry

end NUMINAMATH_CALUDE_computer_price_equation_l8_873


namespace NUMINAMATH_CALUDE_triangle_properties_l8_816

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The given condition for the triangle -/
def satisfies_condition (t : Triangle) : Prop :=
  2 * Real.sqrt 2 * (Real.sin t.A ^ 2 - Real.sin t.C ^ 2) = (t.a - t.b) * Real.sin t.B

/-- The circumradius of the triangle is √2 -/
def has_circumradius_sqrt2 (t : Triangle) : Prop :=
  ∃ (R : ℝ), R = Real.sqrt 2

/-- The theorem to be proved -/
theorem triangle_properties (t : Triangle) 
  (h1 : satisfies_condition t) 
  (h2 : has_circumradius_sqrt2 t) : 
  t.C = Real.pi / 3 ∧ 
  ∃ (S : ℝ), S ≤ 3 * Real.sqrt 3 / 2 ∧ 
  (∀ (S' : ℝ), S' = 1/2 * t.a * t.b * Real.sin t.C → S' ≤ S) :=
sorry

end NUMINAMATH_CALUDE_triangle_properties_l8_816


namespace NUMINAMATH_CALUDE_star_property_l8_818

-- Define the set of elements
inductive Element : Type
  | one : Element
  | two : Element
  | three : Element
  | four : Element

-- Define the operation *
def star : Element → Element → Element
  | Element.one, Element.one => Element.one
  | Element.one, Element.two => Element.three
  | Element.one, Element.three => Element.two
  | Element.one, Element.four => Element.four
  | Element.two, Element.one => Element.three
  | Element.two, Element.two => Element.one
  | Element.two, Element.three => Element.four
  | Element.two, Element.four => Element.two
  | Element.three, Element.one => Element.two
  | Element.three, Element.two => Element.four
  | Element.three, Element.three => Element.one
  | Element.three, Element.four => Element.three
  | Element.four, Element.one => Element.four
  | Element.four, Element.two => Element.two
  | Element.four, Element.three => Element.three
  | Element.four, Element.four => Element.one

theorem star_property :
  star (star Element.three Element.two) (star Element.four Element.one) = Element.one := by
  sorry

end NUMINAMATH_CALUDE_star_property_l8_818


namespace NUMINAMATH_CALUDE_overtime_hours_calculation_l8_869

theorem overtime_hours_calculation (regular_rate : ℝ) (regular_hours : ℝ) (total_pay : ℝ) :
  regular_rate = 3 →
  regular_hours = 40 →
  total_pay = 180 →
  (total_pay - regular_rate * regular_hours) / (2 * regular_rate) = 10 := by
  sorry

end NUMINAMATH_CALUDE_overtime_hours_calculation_l8_869


namespace NUMINAMATH_CALUDE_expression_evaluation_l8_801

theorem expression_evaluation (m n : ℚ) (hm : m = -1/3) (hn : n = 1/2) :
  -2 * (m * n - 3 * m^2) + 3 * (2 * m * n - 5 * m^2) = -5/3 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l8_801


namespace NUMINAMATH_CALUDE_quadratic_inequality_condition_l8_821

theorem quadratic_inequality_condition (a : ℝ) :
  (∀ x : ℝ, x^2 - 2^(a+2) * x - 2^(a+3) + 12 > 0) ↔ a < 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_condition_l8_821


namespace NUMINAMATH_CALUDE_max_value_of_exponential_difference_l8_868

theorem max_value_of_exponential_difference :
  ∃ (max : ℝ), max = 2/3 ∧ ∀ (x : ℝ), 2^x - 8^x ≤ max :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_exponential_difference_l8_868


namespace NUMINAMATH_CALUDE_regular_polygon_interior_exterior_angle_difference_l8_899

/-- A regular polygon where each interior angle is 90° larger than each exterior angle has 8 sides. -/
theorem regular_polygon_interior_exterior_angle_difference (n : ℕ) : 
  n > 2 → 
  (n - 2) * 180 / n - 360 / n = 90 → 
  n = 8 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_interior_exterior_angle_difference_l8_899


namespace NUMINAMATH_CALUDE_binomial_expansion_coefficient_l8_883

theorem binomial_expansion_coefficient (x : ℝ) :
  ∃ (a a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ : ℝ),
  (1 + x)^10 = a + a₁*(1-x) + a₂*(1-x)^2 + a₃*(1-x)^3 + a₄*(1-x)^4 + 
               a₅*(1-x)^5 + a₆*(1-x)^6 + a₇*(1-x)^7 + a₈*(1-x)^8 + 
               a₉*(1-x)^9 + a₁₀*(1-x)^10 ∧ 
  a₈ = 180 := by
sorry

end NUMINAMATH_CALUDE_binomial_expansion_coefficient_l8_883


namespace NUMINAMATH_CALUDE_no_winning_strategy_l8_840

/-- Represents a player in the game -/
inductive Player
| kezdo
| masodik

/-- Represents a cell in the grid -/
structure Cell where
  row : Fin 19
  col : Fin 19

/-- Represents a move in the game -/
structure Move where
  player : Player
  cell : Cell
  value : Fin 2

/-- Represents the state of the game after all moves -/
def GameState := List Move

/-- Calculates the sum of a row -/
def rowSum (state : GameState) (row : Fin 19) : Nat :=
  sorry

/-- Calculates the sum of a column -/
def colSum (state : GameState) (col : Fin 19) : Nat :=
  sorry

/-- Calculates the maximum row sum -/
def maxRowSum (state : GameState) : Nat :=
  sorry

/-- Calculates the maximum column sum -/
def maxColSum (state : GameState) : Nat :=
  sorry

/-- Represents a strategy for a player -/
def Strategy := GameState → Move

/-- Theorem: No winning strategy exists for either player -/
theorem no_winning_strategy :
  ∀ (kezdo_strategy : Strategy) (masodik_strategy : Strategy),
    ∃ (final_state : GameState),
      (maxRowSum final_state = maxColSum final_state) ∧
      (List.length final_state = 19 * 19) :=
sorry

end NUMINAMATH_CALUDE_no_winning_strategy_l8_840


namespace NUMINAMATH_CALUDE_inequality_proof_equality_condition_l8_827

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h : x + y + z = 9 * x * y * z) :
  x / Real.sqrt (x^2 + 2*y*z + 2) + y / Real.sqrt (y^2 + 2*z*x + 2) + z / Real.sqrt (z^2 + 2*x*y + 2) ≥ 1 :=
by sorry

theorem equality_condition (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h : x + y + z = 9 * x * y * z) :
  x / Real.sqrt (x^2 + 2*y*z + 2) + y / Real.sqrt (y^2 + 2*z*x + 2) + z / Real.sqrt (z^2 + 2*x*y + 2) = 1 ↔
  x = y ∧ y = z ∧ x = Real.sqrt 3 / 3 :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_equality_condition_l8_827


namespace NUMINAMATH_CALUDE_fraction_sum_equality_l8_860

theorem fraction_sum_equality (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ -1) :
  (2*x - 5) / (x^2 - 1) + 3 / (1 - x) = -(x + 8) / (x^2 - 1) := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equality_l8_860


namespace NUMINAMATH_CALUDE_correct_distribution_l8_817

/-- Represents the distribution of chestnuts among three girls -/
structure ChestnutDistribution where
  alya : ℕ
  valya : ℕ
  galya : ℕ

/-- Checks if the given distribution satisfies the problem conditions -/
def isValidDistribution (d : ChestnutDistribution) : Prop :=
  d.alya + d.valya + d.galya = 70 ∧
  4 * d.valya = 3 * d.alya ∧
  7 * d.alya = 6 * d.galya

/-- Theorem stating that the given distribution is correct -/
theorem correct_distribution :
  let d : ChestnutDistribution := ⟨24, 18, 28⟩
  isValidDistribution d := by
  sorry


end NUMINAMATH_CALUDE_correct_distribution_l8_817


namespace NUMINAMATH_CALUDE_parallelogram_area_l8_811

/-- The area of a parallelogram with given properties -/
theorem parallelogram_area (s2 : ℝ) (a : ℝ) (h_s2_pos : s2 > 0) (h_a_pos : a > 0) (h_a_lt_180 : a < 180) :
  let s1 := 2 * s2
  let θ := a * π / 180
  2 * s2^2 * Real.sin θ = s1 * s2 * Real.sin θ :=
by sorry

end NUMINAMATH_CALUDE_parallelogram_area_l8_811


namespace NUMINAMATH_CALUDE_number_equation_solution_l8_882

theorem number_equation_solution : ∃ x : ℚ, (3 * x + 15 = 6 * x - 10) ∧ (x = 25 / 3) := by
  sorry

end NUMINAMATH_CALUDE_number_equation_solution_l8_882


namespace NUMINAMATH_CALUDE_spring_experiment_l8_822

/-- Spring experiment data points -/
def spring_data : List (ℝ × ℝ) := [(0, 20), (1, 22), (2, 24), (3, 26), (4, 28), (5, 30)]

/-- The relationship between spring length y (in cm) and weight x (in kg) -/
def spring_relation (x y : ℝ) : Prop := y = 2 * x + 20

/-- Theorem stating that the spring_relation holds for all data points in spring_data -/
theorem spring_experiment :
  ∀ (point : ℝ × ℝ), point ∈ spring_data → spring_relation point.1 point.2 := by
  sorry

end NUMINAMATH_CALUDE_spring_experiment_l8_822


namespace NUMINAMATH_CALUDE_find_certain_number_l8_831

theorem find_certain_number (x : ℝ) : 
  (20 + 40 + 60) / 3 = ((x + 70 + 16) / 3) + 8 → x = 10 := by
sorry

end NUMINAMATH_CALUDE_find_certain_number_l8_831


namespace NUMINAMATH_CALUDE_inequality_proof_l8_805

theorem inequality_proof (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hab : a > b) :
  1 / (a * b^2) > 1 / (a^2 * b) := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l8_805


namespace NUMINAMATH_CALUDE_collinear_implies_coplanar_exist_coplanar_non_collinear_l8_886

-- Define a Point in 3D space
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define a predicate for three points being collinear
def collinear (p q r : Point3D) : Prop := sorry

-- Define a predicate for four points being coplanar
def coplanar (p q r s : Point3D) : Prop := sorry

-- Theorem: If three out of four points are collinear, then all four points are coplanar
theorem collinear_implies_coplanar (p q r s : Point3D) :
  (collinear p q r ∨ collinear p q s ∨ collinear p r s ∨ collinear q r s) →
  coplanar p q r s :=
sorry

-- Theorem: There exist four coplanar points where no three are collinear
theorem exist_coplanar_non_collinear :
  ∃ (p q r s : Point3D), coplanar p q r s ∧
    ¬(collinear p q r ∨ collinear p q s ∨ collinear p r s ∨ collinear q r s) :=
sorry

end NUMINAMATH_CALUDE_collinear_implies_coplanar_exist_coplanar_non_collinear_l8_886


namespace NUMINAMATH_CALUDE_colored_integers_theorem_l8_863

def ColoredInteger := ℤ → Bool

theorem colored_integers_theorem (color : ColoredInteger) 
  (h1 : color 1 = true)
  (h2 : ∀ a b : ℤ, color a = true → color b = true → color (a + b) ≠ color (a - b)) :
  color 2011 = true := by sorry

end NUMINAMATH_CALUDE_colored_integers_theorem_l8_863


namespace NUMINAMATH_CALUDE_total_cookies_l8_879

def cookies_per_bag : ℕ := 21
def bags_per_box : ℕ := 4
def number_of_boxes : ℕ := 2

theorem total_cookies : cookies_per_bag * bags_per_box * number_of_boxes = 168 := by
  sorry

end NUMINAMATH_CALUDE_total_cookies_l8_879


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l8_897

/-- A hyperbola with foci on the x-axis -/
structure Hyperbola where
  /-- The ratio of y to x in the asymptotic equations -/
  asymptote_ratio : ℝ
  /-- The asymptotic equations are y = ± asymptote_ratio * x -/
  asymptote_eq : asymptote_ratio > 0

/-- The eccentricity of a hyperbola -/
def eccentricity (h : Hyperbola) : ℝ :=
  sorry

/-- Theorem: The eccentricity of a hyperbola with asymptotic ratio 2/3 is √13/3 -/
theorem hyperbola_eccentricity (h : Hyperbola) 
  (h_asymptote : h.asymptote_ratio = 2/3) : 
  eccentricity h = Real.sqrt 13 / 3 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l8_897


namespace NUMINAMATH_CALUDE_chloe_final_score_l8_853

/-- Chloe's points at the end of a trivia game -/
theorem chloe_final_score (first_round second_round last_round : Int) 
  (h1 : first_round = 40)
  (h2 : second_round = 50)
  (h3 : last_round = -4) :
  first_round + second_round + last_round = 86 := by
  sorry

end NUMINAMATH_CALUDE_chloe_final_score_l8_853


namespace NUMINAMATH_CALUDE_no_eight_consecutive_almost_squares_l8_803

/-- Definition of an almost square -/
def is_almost_square (n : ℕ) : Prop :=
  ∃ k p : ℕ, (n = k^2 ∨ n = k^2 * p) ∧ (p = 1 ∨ Nat.Prime p)

/-- Theorem stating that 8 consecutive almost squares are impossible -/
theorem no_eight_consecutive_almost_squares :
  ¬ ∃ n : ℕ, ∀ i : Fin 8, is_almost_square (n + i) :=
sorry

end NUMINAMATH_CALUDE_no_eight_consecutive_almost_squares_l8_803


namespace NUMINAMATH_CALUDE_distribute_4_balls_3_boxes_l8_810

/-- The number of ways to distribute indistinguishable balls into distinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ :=
  sorry

/-- Theorem: There are 15 ways to distribute 4 indistinguishable balls into 3 distinguishable boxes -/
theorem distribute_4_balls_3_boxes : distribute_balls 4 3 = 15 := by
  sorry

end NUMINAMATH_CALUDE_distribute_4_balls_3_boxes_l8_810


namespace NUMINAMATH_CALUDE_constant_term_expansion_l8_884

theorem constant_term_expansion (a : ℝ) : 
  (∃ k : ℝ, k = 24 ∧ k = (3/2) * a^2) → (a = 4 ∨ a = -4) := by
  sorry

end NUMINAMATH_CALUDE_constant_term_expansion_l8_884


namespace NUMINAMATH_CALUDE_log_stack_sum_l8_880

/-- 
Given a stack of logs where:
- The bottom row has 15 logs
- Each successive row has one less log
- The top row has 4 logs
This theorem proves that the total number of logs in the stack is 114.
-/
theorem log_stack_sum : ∀ (n : ℕ) (a l : ℤ),
  n = 15 - 4 + 1 →
  a = 15 →
  l = 4 →
  n * (a + l) / 2 = 114 := by
  sorry

end NUMINAMATH_CALUDE_log_stack_sum_l8_880


namespace NUMINAMATH_CALUDE_two_solutions_cubic_equation_l8_841

theorem two_solutions_cubic_equation : 
  ∃! (s : Finset (ℤ × ℤ)), 
    (∀ (x y : ℤ), (x, y) ∈ s ↔ x^3 + y^2 = 2*y + 1) ∧ 
    s.card = 2 := by
  sorry

end NUMINAMATH_CALUDE_two_solutions_cubic_equation_l8_841


namespace NUMINAMATH_CALUDE_hyperbola_b_value_l8_890

-- Define the hyperbola
def hyperbola (x y b : ℝ) : Prop := x^2 / 4 - y^2 / b^2 = 1

-- Define the asymptotes
def asymptotes (x y : ℝ) : Prop := y = x / 2 ∨ y = -x / 2

-- Theorem statement
theorem hyperbola_b_value (b : ℝ) :
  (b > 0) →
  (∀ x y : ℝ, hyperbola x y b ↔ asymptotes x y) →
  b = 1 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_b_value_l8_890


namespace NUMINAMATH_CALUDE_intersection_condition_l8_830

theorem intersection_condition (a : ℝ) : 
  let M := {x : ℝ | x - a = 0}
  let N := {x : ℝ | a * x - 1 = 0}
  (M ∩ N = N) → (a = 0 ∨ a = 1 ∨ a = -1) :=
by sorry

end NUMINAMATH_CALUDE_intersection_condition_l8_830


namespace NUMINAMATH_CALUDE_carter_red_velvet_cakes_l8_833

/-- The number of red velvet cakes Carter usually bakes per week -/
def usual_red_velvet : ℕ := sorry

/-- The number of cheesecakes Carter usually bakes per week -/
def usual_cheesecakes : ℕ := 6

/-- The number of muffins Carter usually bakes per week -/
def usual_muffins : ℕ := 5

/-- The total number of additional cakes Carter baked this week -/
def additional_cakes : ℕ := 38

/-- The factor by which Carter increased his baking this week -/
def increase_factor : ℕ := 3

theorem carter_red_velvet_cakes :
  (usual_cheesecakes + usual_muffins + usual_red_velvet) + additional_cakes =
  increase_factor * (usual_cheesecakes + usual_muffins + usual_red_velvet) →
  usual_red_velvet = 8 := by
sorry

end NUMINAMATH_CALUDE_carter_red_velvet_cakes_l8_833


namespace NUMINAMATH_CALUDE_yogurt_combinations_l8_847

theorem yogurt_combinations (flavors : ℕ) (toppings : ℕ) (sizes : ℕ) :
  flavors = 5 → toppings = 8 → sizes = 3 →
  flavors * (toppings.choose 2) * sizes = 420 :=
by sorry

end NUMINAMATH_CALUDE_yogurt_combinations_l8_847


namespace NUMINAMATH_CALUDE_beacon_population_proof_l8_849

/-- The population of Richmond -/
def richmond_population : ℕ := 3000

/-- The difference in population between Richmond and Victoria -/
def richmond_victoria_difference : ℕ := 1000

/-- The ratio of Victoria's population to Beacon's population -/
def victoria_beacon_ratio : ℕ := 4

/-- The population of Beacon -/
def beacon_population : ℕ := 500

theorem beacon_population_proof :
  richmond_population - richmond_victoria_difference = victoria_beacon_ratio * beacon_population :=
by sorry

end NUMINAMATH_CALUDE_beacon_population_proof_l8_849


namespace NUMINAMATH_CALUDE_complement_of_A_wrt_U_l8_835

def U : Finset Int := {-2, -1, 1, 3, 5}
def A : Finset Int := {-1, 3}

theorem complement_of_A_wrt_U :
  U \ A = {-2, 1, 5} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_wrt_U_l8_835


namespace NUMINAMATH_CALUDE_range_of_m_l8_829

theorem range_of_m (m : ℝ) : (∀ x : ℝ, |x + 3| ≥ m + 4) → m ≤ -4 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l8_829


namespace NUMINAMATH_CALUDE_find_k_l8_871

theorem find_k : ∃ k : ℝ, ∀ x : ℝ, -x^2 - (k + 12)*x - 8 = -(x - 2)*(x - 4) → k = -18 := by
  sorry

end NUMINAMATH_CALUDE_find_k_l8_871


namespace NUMINAMATH_CALUDE_geometric_sum_example_l8_826

/-- The sum of the first n terms of a geometric sequence -/
def geometric_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

/-- The sum of the first 8 terms of the geometric sequence with first term 1/3 and common ratio 1/3 -/
theorem geometric_sum_example : geometric_sum (1/3) (1/3) 8 = 3280/6561 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sum_example_l8_826


namespace NUMINAMATH_CALUDE_sqrt_n_squared_plus_n_bounds_l8_848

theorem sqrt_n_squared_plus_n_bounds (n : ℕ) :
  (n : ℝ) + 0.4 < Real.sqrt ((n : ℝ)^2 + n) ∧ Real.sqrt ((n : ℝ)^2 + n) < (n : ℝ) + 0.5 :=
by sorry

end NUMINAMATH_CALUDE_sqrt_n_squared_plus_n_bounds_l8_848


namespace NUMINAMATH_CALUDE_tan_equality_implies_45_l8_850

theorem tan_equality_implies_45 (n : ℤ) : 
  -90 < n ∧ n < 90 ∧ Real.tan (n * π / 180) = Real.tan (225 * π / 180) → n = 45 :=
by sorry

end NUMINAMATH_CALUDE_tan_equality_implies_45_l8_850


namespace NUMINAMATH_CALUDE_count_not_divisible_by_8_or_7_l8_896

def count_not_divisible (n : ℕ) (d₁ d₂ : ℕ) : ℕ :=
  n - (n / d₁ + n / d₂ - n / (lcm d₁ d₂))

theorem count_not_divisible_by_8_or_7 :
  count_not_divisible 1199 8 7 = 900 := by
  sorry

end NUMINAMATH_CALUDE_count_not_divisible_by_8_or_7_l8_896


namespace NUMINAMATH_CALUDE_set_condition_l8_809

theorem set_condition (x : ℝ) : x ≠ 3 ∧ x ≠ -1 ↔ x^2 - 2*x ≠ 3 := by
  sorry

end NUMINAMATH_CALUDE_set_condition_l8_809


namespace NUMINAMATH_CALUDE_sports_club_tennis_players_l8_887

/-- Given a sports club with the following properties:
  * There are 30 members in total
  * 17 members play badminton
  * 2 members do not play either badminton or tennis
  * 10 members play both badminton and tennis
  Prove that 21 members play tennis -/
theorem sports_club_tennis_players :
  ∀ (total_members badminton_players neither_players both_players : ℕ),
    total_members = 30 →
    badminton_players = 17 →
    neither_players = 2 →
    both_players = 10 →
    ∃ (tennis_players : ℕ),
      tennis_players = 21 ∧
      tennis_players = total_members - neither_players - (badminton_players - both_players) :=
by sorry

end NUMINAMATH_CALUDE_sports_club_tennis_players_l8_887


namespace NUMINAMATH_CALUDE_coin_value_difference_l8_865

/-- Represents the number of coins of each type -/
structure CoinCount where
  pennies : Nat
  nickels : Nat
  dimes : Nat

/-- Calculates the total value in cents for a given coin count -/
def totalValue (coins : CoinCount) : Nat :=
  coins.pennies + 5 * coins.nickels + 10 * coins.dimes

/-- Represents the constraint that the total number of coins is 3030 -/
def totalCoins (coins : CoinCount) : Prop :=
  coins.pennies + coins.nickels + coins.dimes = 3030

/-- Represents the constraint that there are at least 10 of each coin type -/
def atLeastTenEach (coins : CoinCount) : Prop :=
  coins.pennies ≥ 10 ∧ coins.nickels ≥ 10 ∧ coins.dimes ≥ 10

/-- The main theorem stating the difference between max and min possible values -/
theorem coin_value_difference :
  ∃ (max min : CoinCount),
    totalCoins max ∧ totalCoins min ∧
    atLeastTenEach max ∧ atLeastTenEach min ∧
    (∀ c, totalCoins c → atLeastTenEach c → totalValue c ≤ totalValue max) ∧
    (∀ c, totalCoins c → atLeastTenEach c → totalValue c ≥ totalValue min) ∧
    totalValue max - totalValue min = 27000 := by
  sorry

end NUMINAMATH_CALUDE_coin_value_difference_l8_865


namespace NUMINAMATH_CALUDE_ordering_theorem_l8_851

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the properties of f
def monotonically_decreasing_neg (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y ∧ y ≤ 0 → f y < f x

def even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

-- State the theorem
theorem ordering_theorem (h1 : monotonically_decreasing_neg f) (h2 : even_function f) :
  f (-1) < f 9 ∧ f 9 < f 13 := by
  sorry

end NUMINAMATH_CALUDE_ordering_theorem_l8_851


namespace NUMINAMATH_CALUDE_tangent_product_identity_l8_852

theorem tangent_product_identity : 
  (1 + Real.tan (17 * π / 180)) * 
  (1 + Real.tan (18 * π / 180)) * 
  (1 + Real.tan (27 * π / 180)) * 
  (1 + Real.tan (28 * π / 180)) = 4 := by
  sorry

end NUMINAMATH_CALUDE_tangent_product_identity_l8_852


namespace NUMINAMATH_CALUDE_initial_average_age_proof_l8_881

/-- Proves that the initial average age of a group is 16 years, given the conditions of the problem -/
theorem initial_average_age_proof (initial_count : ℕ) (new_count : ℕ) (new_avg_age : ℚ) (final_avg_age : ℚ) :
  initial_count = 12 →
  new_count = 12 →
  new_avg_age = 15 →
  final_avg_age = 15.5 →
  (initial_count * (initial_count * final_avg_age - new_count * new_avg_age) / (initial_count * initial_count)) = 16 := by
  sorry

end NUMINAMATH_CALUDE_initial_average_age_proof_l8_881


namespace NUMINAMATH_CALUDE_rabbit_escape_theorem_l8_870

/-- The number of additional jumps a rabbit can make before a dog catches it. -/
def rabbit_jumps_before_catch (head_start : ℕ) (dog_jumps : ℕ) (rabbit_jumps : ℕ)
  (dog_distance : ℕ) (rabbit_distance : ℕ) : ℕ :=
  14 * head_start

/-- Theorem stating the number of jumps a rabbit can make before being caught by a dog
    under specific conditions. -/
theorem rabbit_escape_theorem :
  rabbit_jumps_before_catch 50 5 6 7 9 = 700 := by
  sorry

#eval rabbit_jumps_before_catch 50 5 6 7 9

end NUMINAMATH_CALUDE_rabbit_escape_theorem_l8_870


namespace NUMINAMATH_CALUDE_symmetric_point_and_line_in_quadrant_l8_806

-- Define the symmetric point function
def symmetric_point (x y : ℝ) (a b c : ℝ) : ℝ × ℝ := sorry

-- Define the line equation
def line_equation (m : ℝ) (x y : ℝ) : Prop :=
  m * x + y + m - 1 = 0

theorem symmetric_point_and_line_in_quadrant :
  -- Statement C
  symmetric_point 1 0 1 (-1) 1 = (-1, 2) ∧
  -- Statement D
  ∀ m : ℝ, line_equation m (-1) 1 := by sorry

end NUMINAMATH_CALUDE_symmetric_point_and_line_in_quadrant_l8_806


namespace NUMINAMATH_CALUDE_inequalities_proof_l8_823

theorem inequalities_proof (a b : ℝ) (h1 : a > b) (h2 : b ≥ 2) : 
  (b^2 > 3*b - a) ∧ (a*b > a + b) := by
  sorry

end NUMINAMATH_CALUDE_inequalities_proof_l8_823
