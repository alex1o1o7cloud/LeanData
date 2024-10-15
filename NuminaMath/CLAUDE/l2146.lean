import Mathlib

namespace NUMINAMATH_CALUDE_expense_representation_l2146_214604

-- Define income and expense as real numbers
def income : ℝ := 5
def expense : ℝ := 5

-- Define the representation of income
def income_representation : ℝ := 5

-- Theorem to prove
theorem expense_representation : 
  income_representation = income → -expense = -5 :=
by
  sorry

end NUMINAMATH_CALUDE_expense_representation_l2146_214604


namespace NUMINAMATH_CALUDE_seed_germination_percentage_l2146_214664

theorem seed_germination_percentage (seeds_plot1 seeds_plot2 : ℕ) 
  (germination_rate1 germination_rate2 : ℚ) :
  seeds_plot1 = 300 →
  seeds_plot2 = 200 →
  germination_rate1 = 20 / 100 →
  germination_rate2 = 35 / 100 →
  let total_seeds := seeds_plot1 + seeds_plot2
  let germinated_seeds1 := (seeds_plot1 : ℚ) * germination_rate1
  let germinated_seeds2 := (seeds_plot2 : ℚ) * germination_rate2
  let total_germinated := germinated_seeds1 + germinated_seeds2
  total_germinated / total_seeds = 26 / 100 := by
sorry

end NUMINAMATH_CALUDE_seed_germination_percentage_l2146_214664


namespace NUMINAMATH_CALUDE_diamond_equation_solution_l2146_214671

/-- Diamond operation -/
def diamond (A B : ℝ) : ℝ := 4 * A + 3 * B + 7

/-- Theorem stating the unique solution to A ◊ 7 = 76 -/
theorem diamond_equation_solution :
  ∃! A : ℝ, diamond A 7 = 76 ∧ A = 12 := by
sorry

end NUMINAMATH_CALUDE_diamond_equation_solution_l2146_214671


namespace NUMINAMATH_CALUDE_inequality_proof_l2146_214627

/-- An odd function f with the given property -/
def f (x : ℝ) : ℝ := sorry

/-- f is an odd function -/
axiom f_odd (x : ℝ) : f (-x) = -f x

/-- The derivative of f -/
noncomputable def f' : ℝ → ℝ := sorry

/-- The property x * f'(x) - f(x) < 0 for x ≠ 0 -/
axiom f_property (x : ℝ) (h : x ≠ 0) : x * f' x - f x < 0

/-- The main theorem -/
theorem inequality_proof :
  f (-3) / (-3) < f (Real.exp 1) / (Real.exp 1) ∧
  f (Real.exp 1) / (Real.exp 1) < f (Real.log 2) / (Real.log 2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2146_214627


namespace NUMINAMATH_CALUDE_wage_decrease_hours_increase_l2146_214661

theorem wage_decrease_hours_increase 
  (original_wage : ℝ) 
  (original_hours : ℝ) 
  (wage_decrease_percent : ℝ) 
  (new_hours : ℝ) 
  (h1 : wage_decrease_percent = 20) 
  (h2 : original_wage > 0) 
  (h3 : original_hours > 0) 
  (h4 : new_hours > 0) 
  (h5 : original_wage * original_hours = (original_wage * (1 - wage_decrease_percent / 100)) * new_hours) :
  (new_hours - original_hours) / original_hours * 100 = 25 := by
sorry

end NUMINAMATH_CALUDE_wage_decrease_hours_increase_l2146_214661


namespace NUMINAMATH_CALUDE_money_value_difference_l2146_214631

/-- Proves that the percentage difference between Etienne's and Diana's money is -12.5% --/
theorem money_value_difference (exchange_rate : ℝ) (diana_dollars : ℝ) (etienne_euros : ℝ) :
  exchange_rate = 1.5 →
  diana_dollars = 600 →
  etienne_euros = 350 →
  ((diana_dollars - etienne_euros * exchange_rate) / diana_dollars) * 100 = 12.5 := by
  sorry

end NUMINAMATH_CALUDE_money_value_difference_l2146_214631


namespace NUMINAMATH_CALUDE_textbook_ratio_l2146_214616

theorem textbook_ratio (initial : ℚ) (remaining : ℚ) 
  (h1 : initial = 960)
  (h2 : remaining = 360)
  (h3 : ∃ textbook_cost : ℚ, initial - textbook_cost - (1/4) * (initial - textbook_cost) = remaining) :
  ∃ textbook_cost : ℚ, textbook_cost / initial = 1/2 := by
sorry

end NUMINAMATH_CALUDE_textbook_ratio_l2146_214616


namespace NUMINAMATH_CALUDE_inequality_proof_l2146_214639

theorem inequality_proof (a b x y : ℝ) (ha : a > 0) (hb : b > 0) (hx : x > 0) (hy : y > 0) :
  a^2 / x + b^2 / y ≥ (a + b)^2 / (x + y) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2146_214639


namespace NUMINAMATH_CALUDE_replaced_person_weight_l2146_214611

/-- Given a group of 8 people, if replacing one person with a new person weighing 89 kg
    increases the average weight by 3 kg, then the weight of the replaced person is 65 kg. -/
theorem replaced_person_weight
  (n : ℕ)
  (new_weight : ℝ)
  (avg_increase : ℝ)
  (h1 : n = 8)
  (h2 : new_weight = 89)
  (h3 : avg_increase = 3)
  : ∃ (old_weight : ℝ), old_weight = new_weight - n * avg_increase :=
by
  sorry

end NUMINAMATH_CALUDE_replaced_person_weight_l2146_214611


namespace NUMINAMATH_CALUDE_value_of_x_when_sqrt_fraction_is_zero_l2146_214641

theorem value_of_x_when_sqrt_fraction_is_zero :
  ∀ x : ℝ, x ≠ 0 → (Real.sqrt (2 - x)) / x = 0 → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_value_of_x_when_sqrt_fraction_is_zero_l2146_214641


namespace NUMINAMATH_CALUDE_base7_to_base9_conversion_l2146_214653

/-- Converts a number from base 7 to base 10 --/
def base7To10 (n : Nat) : Nat :=
  (n % 10) + 7 * ((n / 10) % 10) + 49 * (n / 100)

/-- Converts a number from base 10 to base 9 --/
def base10To9 (n : Nat) : Nat :=
  if n < 9 then n
  else (n % 9) + 10 * (base10To9 (n / 9))

theorem base7_to_base9_conversion :
  base10To9 (base7To10 536) = 332 :=
sorry

end NUMINAMATH_CALUDE_base7_to_base9_conversion_l2146_214653


namespace NUMINAMATH_CALUDE_divisor_of_number_minus_one_l2146_214665

theorem divisor_of_number_minus_one (n : ℕ) (h : n = 5026) : 5 ∣ (n - 1) := by
  sorry

end NUMINAMATH_CALUDE_divisor_of_number_minus_one_l2146_214665


namespace NUMINAMATH_CALUDE_angle_sum_ninety_degrees_l2146_214692

theorem angle_sum_ninety_degrees (α β : Real) 
  (acute_α : 0 < α ∧ α < Real.pi / 2)
  (acute_β : 0 < β ∧ β < Real.pi / 2)
  (eq1 : 3 * Real.sin α ^ 2 + 2 * Real.sin β ^ 2 = 1)
  (eq2 : 3 * Real.sin (2 * α) - 2 * Real.sin (2 * β) = 0) :
  α + 2 * β = Real.pi / 2 := by
sorry

end NUMINAMATH_CALUDE_angle_sum_ninety_degrees_l2146_214692


namespace NUMINAMATH_CALUDE_bush_distance_theorem_l2146_214632

/-- The distance between equally spaced bushes along a road -/
def bush_distance (n : ℕ) (d : ℝ) : ℝ :=
  d * (n - 1)

/-- Theorem: Given 10 equally spaced bushes where the distance between
    the first and fifth bush is 100 feet, the distance between the first
    and last bush is 225 feet. -/
theorem bush_distance_theorem :
  bush_distance 5 100 = 100 →
  bush_distance 10 (100 / 4) = 225 := by
  sorry

end NUMINAMATH_CALUDE_bush_distance_theorem_l2146_214632


namespace NUMINAMATH_CALUDE_problem_solution_l2146_214678

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := |x - a| + 3 * x

-- Theorem statement
theorem problem_solution (a : ℝ) (h : a > 0) :
  -- Part I
  (∀ x : ℝ, f 1 x ≥ 3 * x + 2 ↔ x ≥ 3 ∨ x ≤ -1) ∧
  -- Part II
  ((∀ x : ℝ, f a x ≤ 0 ↔ x ≤ -1) → a = 2) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l2146_214678


namespace NUMINAMATH_CALUDE_modified_code_system_distinct_symbols_l2146_214673

/-- The number of possible symbols (dot, dash, or blank) -/
def num_symbols : ℕ := 3

/-- The maximum length of a sequence -/
def max_length : ℕ := 3

/-- The number of distinct symbols for a given sequence length -/
def distinct_symbols (length : ℕ) : ℕ := num_symbols ^ length

/-- The total number of distinct symbols for sequences of length 1 to max_length -/
def total_distinct_symbols : ℕ :=
  (Finset.range max_length).sum (λ i => distinct_symbols (i + 1))

theorem modified_code_system_distinct_symbols :
  total_distinct_symbols = 39 := by
  sorry

end NUMINAMATH_CALUDE_modified_code_system_distinct_symbols_l2146_214673


namespace NUMINAMATH_CALUDE_paula_bumper_car_rides_l2146_214625

/-- Calculates the number of bumper car rides Paula can take given the total tickets,
    go-kart ticket cost, and bumper car ticket cost. -/
def bumper_car_rides (total_tickets go_kart_cost bumper_car_cost : ℕ) : ℕ :=
  (total_tickets - go_kart_cost) / bumper_car_cost

/-- Proves that Paula can ride the bumper cars 4 times given the conditions. -/
theorem paula_bumper_car_rides :
  let total_tickets : ℕ := 24
  let go_kart_cost : ℕ := 4
  let bumper_car_cost : ℕ := 5
  bumper_car_rides total_tickets go_kart_cost bumper_car_cost = 4 := by
  sorry


end NUMINAMATH_CALUDE_paula_bumper_car_rides_l2146_214625


namespace NUMINAMATH_CALUDE_square_side_length_l2146_214688

theorem square_side_length (side : ℕ) : side ^ 2 < 20 → side = 4 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l2146_214688


namespace NUMINAMATH_CALUDE_dollar_four_negative_one_l2146_214607

-- Define the $ operation
def dollar (a b : ℤ) : ℤ := a * (b + 1) + a * b

-- Theorem statement
theorem dollar_four_negative_one :
  dollar 4 (-1) = -4 := by
  sorry

end NUMINAMATH_CALUDE_dollar_four_negative_one_l2146_214607


namespace NUMINAMATH_CALUDE_solution_set_of_fraction_inequality_range_of_a_for_empty_solution_set_l2146_214643

-- Problem 1
theorem solution_set_of_fraction_inequality (x : ℝ) :
  (x - 3) / (x + 7) < 0 ↔ -7 < x ∧ x < 3 := by sorry

-- Problem 2
theorem range_of_a_for_empty_solution_set (a : ℝ) :
  (∀ x : ℝ, x^2 - 4*a*x + 4*a^2 + a > 0) → a > 0 := by sorry

end NUMINAMATH_CALUDE_solution_set_of_fraction_inequality_range_of_a_for_empty_solution_set_l2146_214643


namespace NUMINAMATH_CALUDE_round_trip_time_l2146_214651

/-- Calculates the total time for a round trip by boat given the boat's speed in standing water,
    the stream's speed, and the distance to the destination. -/
theorem round_trip_time 
  (boat_speed : ℝ) 
  (stream_speed : ℝ) 
  (distance : ℝ) 
  (h1 : boat_speed = 14) 
  (h2 : stream_speed = 1.2) 
  (h3 : distance = 4864) : 
  (distance / (boat_speed + stream_speed)) + (distance / (boat_speed - stream_speed)) = 700 := by
  sorry

#check round_trip_time

end NUMINAMATH_CALUDE_round_trip_time_l2146_214651


namespace NUMINAMATH_CALUDE_fraction_not_whole_number_l2146_214693

theorem fraction_not_whole_number : 
  (∃ n : ℕ, 60 / 12 = n) ∧ 
  (∀ n : ℕ, 60 / 8 ≠ n) ∧ 
  (∃ n : ℕ, 60 / 5 = n) ∧ 
  (∃ n : ℕ, 60 / 4 = n) ∧ 
  (∃ n : ℕ, 60 / 3 = n) := by
  sorry

end NUMINAMATH_CALUDE_fraction_not_whole_number_l2146_214693


namespace NUMINAMATH_CALUDE_unit_digit_of_product_l2146_214650

-- Define the numbers
def a : ℕ := 7858
def b : ℕ := 1086
def c : ℕ := 4582
def d : ℕ := 9783

-- Define the product
def product : ℕ := a * b * c * d

-- Theorem statement
theorem unit_digit_of_product : product % 10 = 8 := by
  sorry

end NUMINAMATH_CALUDE_unit_digit_of_product_l2146_214650


namespace NUMINAMATH_CALUDE_impossible_2018_after_2019_l2146_214669

/-- Represents a single step in the room occupancy change --/
inductive Step
  | Enter : Step  -- Two people enter (+2)
  | Exit : Step   -- One person exits (-1)

/-- Calculates the change in room occupancy for a given step --/
def stepChange (s : Step) : Int :=
  match s with
  | Step.Enter => 2
  | Step.Exit => -1

/-- Represents a sequence of steps over time --/
def Sequence := List Step

/-- Calculates the final room occupancy given a sequence of steps --/
def finalOccupancy (seq : Sequence) : Int :=
  seq.foldl (fun acc s => acc + stepChange s) 0

/-- Theorem: It's impossible to have 2018 people after 2019 steps --/
theorem impossible_2018_after_2019 :
  ∀ (seq : Sequence), seq.length = 2019 → finalOccupancy seq ≠ 2018 :=
by
  sorry


end NUMINAMATH_CALUDE_impossible_2018_after_2019_l2146_214669


namespace NUMINAMATH_CALUDE_encircling_stripe_probability_theorem_l2146_214659

/-- Represents a cube with 6 faces -/
structure Cube :=
  (faces : Fin 6 → Bool)

/-- The probability of a stripe on a single face -/
def stripe_prob : ℚ := 2/3

/-- The probability of a dot on a single face -/
def dot_prob : ℚ := 1/3

/-- The number of valid stripe configurations that encircle the cube -/
def valid_configurations : ℕ := 12

/-- The probability of a continuous stripe encircling the cube -/
def encircling_stripe_probability : ℚ := 768/59049

/-- Theorem stating the probability of a continuous stripe encircling the cube -/
theorem encircling_stripe_probability_theorem :
  encircling_stripe_probability = 
    (stripe_prob ^ 6) * valid_configurations :=
by sorry

end NUMINAMATH_CALUDE_encircling_stripe_probability_theorem_l2146_214659


namespace NUMINAMATH_CALUDE_sameTotalHeadsProbability_eq_565_2048_l2146_214687

/-- Represents the probability distribution of flipping four coins, 
    where three are fair and one has 5/8 probability of heads -/
def coinFlipDistribution : List ℚ :=
  [3/64, 14/64, 24/64, 18/64, 5/64]

/-- The probability of two people getting the same number of heads 
    when each flips four coins (three fair, one biased) -/
def sameTotalHeadsProbability : ℚ :=
  (coinFlipDistribution.map (λ x => x^2)).sum

theorem sameTotalHeadsProbability_eq_565_2048 :
  sameTotalHeadsProbability = 565/2048 := by
  sorry

end NUMINAMATH_CALUDE_sameTotalHeadsProbability_eq_565_2048_l2146_214687


namespace NUMINAMATH_CALUDE_identity_proof_l2146_214609

theorem identity_proof (a b c x : ℝ) (hab : a ≠ b) (hac : a ≠ c) (hbc : b ≠ c) :
  (a^2 * (x - b) * (x - c)) / ((a - b) * (a - c)) +
  (b^2 * (x - a) * (x - c)) / ((b - a) * (b - c)) +
  (c^2 * (x - a) * (x - b)) / ((c - a) * (c - b)) = x^2 := by
  sorry

end NUMINAMATH_CALUDE_identity_proof_l2146_214609


namespace NUMINAMATH_CALUDE_laurent_series_expansion_l2146_214666

/-- Laurent series expansion of f(z) = 2ia / (z^2 + a^2) in the region 0 < |z - ia| < a -/
theorem laurent_series_expansion
  (a : ℝ) (z : ℂ) (ha : a > 0) (hz : 0 < Complex.abs (z - Complex.I * a) ∧ Complex.abs (z - Complex.I * a) < a) :
  (2 * Complex.I * a) / (z^2 + a^2) =
    1 / (z - Complex.I * a) - ∑' k, (z - Complex.I * a)^k / (Complex.I * a)^(k + 1) :=
by sorry

end NUMINAMATH_CALUDE_laurent_series_expansion_l2146_214666


namespace NUMINAMATH_CALUDE_measure_45_seconds_l2146_214629

/-- Represents a fuse that can be lit from either end -/
structure Fuse :=
  (burn_time : ℝ)
  (is_uniform : Bool)

/-- Represents the state of burning a fuse -/
inductive BurnState
  | Unlit
  | LitOneEnd
  | LitBothEnds

/-- Represents the result of burning fuses -/
structure BurnResult :=
  (time : ℝ)
  (fuse1 : BurnState)
  (fuse2 : BurnState)

/-- Function to simulate burning fuses -/
def burn_fuses (f1 f2 : Fuse) : BurnResult :=
  sorry

theorem measure_45_seconds (f1 f2 : Fuse) 
  (h1 : f1.burn_time = 60)
  (h2 : f2.burn_time = 60) :
  ∃ (result : BurnResult), result.time = 45 :=
sorry

end NUMINAMATH_CALUDE_measure_45_seconds_l2146_214629


namespace NUMINAMATH_CALUDE_num_valid_teams_eq_930_l2146_214662

/-- Represents a debater in the team -/
inductive Debater
| Boy : Fin 4 → Debater
| Girl : Fin 4 → Debater

/-- Represents a debate team -/
def DebateTeam := Fin 4 → Debater

/-- Check if Boy A is in the team -/
def has_boy_A (team : DebateTeam) : Prop :=
  ∃ i, team i = Debater.Boy 0

/-- Check if Girl B is in the team -/
def has_girl_B (team : DebateTeam) : Prop :=
  ∃ i, team i = Debater.Girl 1

/-- Check if Boy A is not the first debater -/
def boy_A_not_first (team : DebateTeam) : Prop :=
  team 0 ≠ Debater.Boy 0

/-- Check if Girl B is not the fourth debater -/
def girl_B_not_fourth (team : DebateTeam) : Prop :=
  team 3 ≠ Debater.Girl 1

/-- Check if the team satisfies all constraints -/
def valid_team (team : DebateTeam) : Prop :=
  boy_A_not_first team ∧
  girl_B_not_fourth team ∧
  (has_boy_A team → has_girl_B team)

/-- The number of valid debate teams -/
def num_valid_teams : ℕ := sorry

theorem num_valid_teams_eq_930 : num_valid_teams = 930 := by sorry

end NUMINAMATH_CALUDE_num_valid_teams_eq_930_l2146_214662


namespace NUMINAMATH_CALUDE_prime_pairs_divisibility_l2146_214696

theorem prime_pairs_divisibility (p q : ℕ) : 
  p.Prime ∧ q.Prime ∧ p < 2005 ∧ q < 2005 ∧ 
  (q ∣ p^2 + 8) ∧ (p ∣ q^2 + 8) → 
  ((p = 2 ∧ q = 2) ∨ (p = 881 ∧ q = 89) ∨ (p = 89 ∧ q = 881)) := by
  sorry

end NUMINAMATH_CALUDE_prime_pairs_divisibility_l2146_214696


namespace NUMINAMATH_CALUDE_nearest_integer_to_three_plus_sqrt_five_fourth_power_l2146_214697

theorem nearest_integer_to_three_plus_sqrt_five_fourth_power :
  ∃ n : ℤ, n = 752 ∧ ∀ m : ℤ, |((3 : ℝ) + Real.sqrt 5)^4 - (n : ℝ)| ≤ |((3 : ℝ) + Real.sqrt 5)^4 - (m : ℝ)| := by
  sorry

end NUMINAMATH_CALUDE_nearest_integer_to_three_plus_sqrt_five_fourth_power_l2146_214697


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l2146_214637

/-- Two vectors in ℝ² are parallel if their cross product is zero -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 - a.2 * b.1 = 0

theorem parallel_vectors_x_value :
  ∀ x : ℝ, parallel (1, 2) (x, -2) → x = -1 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l2146_214637


namespace NUMINAMATH_CALUDE_parabola_line_intersection_l2146_214690

-- Define the parabola C
def parabola_C (x y : ℝ) : Prop := y^2 = 4*x

-- Define the point P
def point_P : ℝ × ℝ := (1, 2)

-- Define a line with slope 1
def line_with_slope_1 (b : ℝ) (x y : ℝ) : Prop := y = x + b

-- Define the intersection points A and B
def intersection_points (b : ℝ) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    parabola_C x₁ y₁ ∧ parabola_C x₂ y₂ ∧
    line_with_slope_1 b x₁ y₁ ∧ line_with_slope_1 b x₂ y₂ ∧
    x₁ ≠ x₂

-- Define the condition for circle AB passing through P
def circle_condition (b : ℝ) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    parabola_C x₁ y₁ ∧ parabola_C x₂ y₂ ∧
    line_with_slope_1 b x₁ y₁ ∧ line_with_slope_1 b x₂ y₂ ∧
    (x₁ - point_P.1) * (x₂ - point_P.1) + (y₁ - point_P.2) * (y₂ - point_P.2) = 0

-- Theorem statement
theorem parabola_line_intersection :
  ∃ (b : ℝ), intersection_points b ∧ circle_condition b ∧ b = -7 :=
sorry

end NUMINAMATH_CALUDE_parabola_line_intersection_l2146_214690


namespace NUMINAMATH_CALUDE_abc_range_l2146_214672

theorem abc_range (a b c : ℝ) (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c)
  (h_sum : a + b + c = 1) (h_sum_squares : a^2 + b^2 + c^2 = 3) :
  -1 < a * b * c ∧ a * b * c < 5/27 := by
  sorry

end NUMINAMATH_CALUDE_abc_range_l2146_214672


namespace NUMINAMATH_CALUDE_fifteen_points_densified_thrice_equals_113_original_points_must_be_fifteen_l2146_214694

/-- Calculates the number of points after one densification -/
def densify (n : ℕ) : ℕ := 2 * n - 1

/-- Represents the process of densification repeated k times -/
def densify_k_times (n : ℕ) (k : ℕ) : ℕ :=
  match k with
  | 0 => n
  | k + 1 => densify (densify_k_times n k)

/-- The theorem stating that 3 densifications of 15 points results in 113 points -/
theorem fifteen_points_densified_thrice_equals_113 :
  densify_k_times 15 3 = 113 :=
by sorry

/-- The main theorem proving that starting with 15 points and applying 3 densifications
    is the only way to end up with 113 points -/
theorem original_points_must_be_fifteen (n : ℕ) :
  densify_k_times n 3 = 113 → n = 15 :=
by sorry

end NUMINAMATH_CALUDE_fifteen_points_densified_thrice_equals_113_original_points_must_be_fifteen_l2146_214694


namespace NUMINAMATH_CALUDE_decimal_point_shift_l2146_214686

theorem decimal_point_shift (x : ℝ) : 10 * x = x + 2.7 → x = 0.3 := by
  sorry

end NUMINAMATH_CALUDE_decimal_point_shift_l2146_214686


namespace NUMINAMATH_CALUDE_option2_saves_money_at_80_l2146_214640

/-- The total charge for Option 1 given x participants -/
def option1_charge (x : ℝ) : ℝ := 1500 + 320 * x

/-- The total charge for Option 2 given x participants -/
def option2_charge (x : ℝ) : ℝ := 360 * x - 1800

/-- The original price per person -/
def original_price : ℝ := 400

theorem option2_saves_money_at_80 :
  ∀ x : ℝ, x > 50 → option2_charge 80 < option1_charge 80 := by
  sorry

end NUMINAMATH_CALUDE_option2_saves_money_at_80_l2146_214640


namespace NUMINAMATH_CALUDE_quadratic_roots_l2146_214634

theorem quadratic_roots (m : ℝ) : 
  (∃ x : ℝ, x^2 - 4*x + m = 0 ∧ x = -1) → 
  (∃ y : ℝ, y^2 - 4*y + m = 0 ∧ y = 5) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_l2146_214634


namespace NUMINAMATH_CALUDE_quadratic_equal_roots_l2146_214698

theorem quadratic_equal_roots (b c : ℝ) : 
  (∃ x : ℝ, x^2 + b*x + c = 0 ∧ (∀ y : ℝ, y^2 + b*y + c = 0 → y = x)) → 
  b^2 - 2*(1+2*c) = -2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equal_roots_l2146_214698


namespace NUMINAMATH_CALUDE_parabola_point_focus_distance_l2146_214642

theorem parabola_point_focus_distance (p : ℝ) (y : ℝ) (h1 : p > 0) :
  y^2 = 2*p*8 ∧ (8 + p/2)^2 + y^2 = 10^2 → p = 4 ∧ (y = 8 ∨ y = -8) := by
  sorry

end NUMINAMATH_CALUDE_parabola_point_focus_distance_l2146_214642


namespace NUMINAMATH_CALUDE_lucilles_earnings_l2146_214623

/-- Represents the earnings in cents for each type of weed -/
structure WeedEarnings where
  small : ℕ
  medium : ℕ
  large : ℕ

/-- Represents the number of weeds in a garden area -/
structure WeedCount where
  small : ℕ
  medium : ℕ
  large : ℕ

/-- Calculates the total earnings from weeding a garden area -/
def calculateEarnings (earnings : WeedEarnings) (count : WeedCount) : ℕ :=
  earnings.small * count.small + earnings.medium * count.medium + earnings.large * count.large

/-- Represents Lucille's weeding earnings problem -/
structure LucillesProblem where
  earnings : WeedEarnings
  flowerBed : WeedCount
  vegetablePatch : WeedCount
  grass : WeedCount
  sodaCost : ℕ
  salesTaxPercent : ℕ

/-- Theorem stating that Lucille has 130 cents left after buying the soda -/
theorem lucilles_earnings (problem : LucillesProblem)
  (h1 : problem.earnings = ⟨4, 8, 12⟩)
  (h2 : problem.flowerBed = ⟨6, 3, 2⟩)
  (h3 : problem.vegetablePatch = ⟨10, 2, 2⟩)
  (h4 : problem.grass = ⟨20, 10, 2⟩)
  (h5 : problem.sodaCost = 99)
  (h6 : problem.salesTaxPercent = 15) :
  let totalEarnings := calculateEarnings problem.earnings problem.flowerBed +
                       calculateEarnings problem.earnings problem.vegetablePatch +
                       calculateEarnings problem.earnings ⟨problem.grass.small / 2, problem.grass.medium / 2, problem.grass.large / 2⟩
  let sodaTotalCost := problem.sodaCost + (problem.sodaCost * problem.salesTaxPercent / 100 + 1)
  totalEarnings - sodaTotalCost = 130 := by sorry


end NUMINAMATH_CALUDE_lucilles_earnings_l2146_214623


namespace NUMINAMATH_CALUDE_probability_of_all_successes_l2146_214621

-- Define the number of trials
def n : ℕ := 7

-- Define the probability of success in each trial
def p : ℚ := 2/7

-- Define the number of successes we're interested in
def k : ℕ := 7

-- State the theorem
theorem probability_of_all_successes :
  (n.choose k) * p^k * (1 - p)^(n - k) = 128/823543 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_all_successes_l2146_214621


namespace NUMINAMATH_CALUDE_simplify_expression_l2146_214613

theorem simplify_expression (x y : ℚ) (hx : x = 5) (hy : y = 2) :
  (10 * x^2 * y^3) / (15 * x * y^2) = 20 / 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2146_214613


namespace NUMINAMATH_CALUDE_divisibility_of_fifth_power_differences_l2146_214605

theorem divisibility_of_fifth_power_differences (x y z : ℤ) 
  (hxy : x ≠ y) (hyz : y ≠ z) (hzx : z ≠ x) : 
  ∃ k : ℤ, (x - y)^5 + (y - z)^5 + (z - x)^5 = k * (5 * (x - y) * (y - z) * (z - x)) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_of_fifth_power_differences_l2146_214605


namespace NUMINAMATH_CALUDE_polar_to_cartesian_circle_l2146_214600

theorem polar_to_cartesian_circle (x y ρ θ : ℝ) :
  (ρ = 4 * Real.sin θ) ∧ (x = ρ * Real.cos θ) ∧ (y = ρ * Real.sin θ) →
  x^2 + (y - 2)^2 = 4 :=
by sorry

end NUMINAMATH_CALUDE_polar_to_cartesian_circle_l2146_214600


namespace NUMINAMATH_CALUDE_modulus_of_z_l2146_214626

-- Define the complex number i
def i : ℂ := Complex.I

-- Define z as (1+i)^2
def z : ℂ := (1 + i)^2

-- Theorem stating that the modulus of z is 2
theorem modulus_of_z : Complex.abs z = 2 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_z_l2146_214626


namespace NUMINAMATH_CALUDE_parallel_line_slope_l2146_214667

/-- Given a line with equation 3x - 6y = 21, prove that any parallel line has slope 1/2 -/
theorem parallel_line_slope (x y : ℝ) :
  (3 * x - 6 * y = 21) → 
  (∃ (m b : ℝ), y = m * x + b ∧ m = (1 : ℝ) / 2) :=
by sorry

end NUMINAMATH_CALUDE_parallel_line_slope_l2146_214667


namespace NUMINAMATH_CALUDE_lisa_expenses_l2146_214622

theorem lisa_expenses (B : ℝ) (book coffee : ℝ) : 
  book = 0.3 * (B - 2 * coffee) →
  coffee = 0.1 * (B - book) →
  book + coffee = (31 : ℝ) / 94 * B := by
sorry

end NUMINAMATH_CALUDE_lisa_expenses_l2146_214622


namespace NUMINAMATH_CALUDE_math_score_proof_l2146_214630

/-- Represents the scores for each subject --/
structure Scores where
  ethics : ℕ
  korean : ℕ
  science : ℕ
  social : ℕ
  math : ℕ

/-- Calculates the average score --/
def average (s : Scores) : ℚ :=
  (s.ethics + s.korean + s.science + s.social + s.math) / 5

theorem math_score_proof (s : Scores) 
  (h1 : s.ethics = 82)
  (h2 : s.korean = 90)
  (h3 : s.science = 88)
  (h4 : s.social = 84)
  (h5 : average s = 88) :
  s.math = 96 := by
  sorry

#eval average { ethics := 82, korean := 90, science := 88, social := 84, math := 96 }

end NUMINAMATH_CALUDE_math_score_proof_l2146_214630


namespace NUMINAMATH_CALUDE_sum_of_first_40_digits_eq_72_l2146_214685

/-- The sum of the first 40 digits after the decimal point in the decimal representation of 1/2222 -/
def sum_of_first_40_digits : ℕ :=
  -- Define the sum here
  72

/-- Theorem stating that the sum of the first 40 digits after the decimal point
    in the decimal representation of 1/2222 is equal to 72 -/
theorem sum_of_first_40_digits_eq_72 :
  sum_of_first_40_digits = 72 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_sum_of_first_40_digits_eq_72_l2146_214685


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_difference_l2146_214699

theorem absolute_value_equation_solution_difference : ∃ (x y : ℝ), 
  (x ≠ y ∧ 
   (|x^2 + 3*x + 3| = 15 ∧ |y^2 + 3*y + 3| = 15) ∧
   ∀ z : ℝ, |z^2 + 3*z + 3| = 15 → (z = x ∨ z = y)) →
  |x - y| = 7 :=
sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_difference_l2146_214699


namespace NUMINAMATH_CALUDE_largest_number_l2146_214636

-- Define a function to convert a number from base n to decimal
def to_decimal (digits : List Nat) (base : Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * base ^ i) 0

-- Define the numbers in their respective bases
def num_A : Nat := to_decimal [5, 8] 9
def num_B : Nat := to_decimal [0, 0, 2] 6
def num_C : Nat := to_decimal [8, 6] 11
def num_D : Nat := 70

-- Theorem statement
theorem largest_number :
  num_A = max num_A (max num_B (max num_C num_D)) :=
by sorry

end NUMINAMATH_CALUDE_largest_number_l2146_214636


namespace NUMINAMATH_CALUDE_function_composition_l2146_214648

theorem function_composition (f : ℝ → ℝ) (h : ∀ x : ℝ, f x = x^2 - 2*x + 5) :
  ∀ x : ℝ, f (x - 1) = x^2 - 4*x + 8 := by
sorry

end NUMINAMATH_CALUDE_function_composition_l2146_214648


namespace NUMINAMATH_CALUDE_solution_set_l2146_214652

-- Define the function f and its derivative f'
variable (f : ℝ → ℝ) (f' : ℝ → ℝ)

-- Define the condition that f' is the derivative of f
def is_derivative (f : ℝ → ℝ) (f' : ℝ → ℝ) : Prop :=
  ∀ x, deriv f x = f' x

-- Define the condition that 2f'(x) > f(x) for all x
def condition (f : ℝ → ℝ) (f' : ℝ → ℝ) : Prop :=
  ∀ x, 2 * f' x > f x

-- Define the inequality we want to solve
def inequality (f : ℝ → ℝ) (x : ℝ) : Prop :=
  Real.exp ((x - 1) / 2) * f x < f (2 * x - 1)

-- State the theorem
theorem solution_set (f : ℝ → ℝ) (f' : ℝ → ℝ) :
  is_derivative f f' → condition f f' →
  (∀ x, inequality f x ↔ x > 1) :=
sorry

end NUMINAMATH_CALUDE_solution_set_l2146_214652


namespace NUMINAMATH_CALUDE_hall_mat_expenditure_l2146_214683

/-- Calculates the total expenditure for covering a rectangular floor with mat. -/
def total_expenditure (length width cost_per_sqm : ℝ) : ℝ :=
  length * width * cost_per_sqm

/-- Proves that the total expenditure for covering a 20m × 15m floor with mat at Rs. 50 per square meter is Rs. 15,000. -/
theorem hall_mat_expenditure :
  total_expenditure 20 15 50 = 15000 := by
  sorry

end NUMINAMATH_CALUDE_hall_mat_expenditure_l2146_214683


namespace NUMINAMATH_CALUDE_max_guests_correct_l2146_214603

/-- The maximum number of guests that can dine at a restaurant with n choices
    for each of starters, main dishes, desserts, and wines, such that:
    1) No two guests have the same order
    2) There is no collection of n guests whose orders coincide in three aspects
       but differ in the fourth -/
def max_guests (n : ℕ+) : ℕ :=
  if n = 1 then 1 else n^4 - n^3

theorem max_guests_correct (n : ℕ+) :
  (max_guests n = 1 ∧ n = 1) ∨
  (max_guests n = n^4 - n^3 ∧ n ≥ 2) :=
sorry

end NUMINAMATH_CALUDE_max_guests_correct_l2146_214603


namespace NUMINAMATH_CALUDE_certain_number_proof_l2146_214682

theorem certain_number_proof (y : ℕ) : (2^14) - (2^12) = 3 * (2^y) → y = 12 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l2146_214682


namespace NUMINAMATH_CALUDE_max_sum_of_digits_is_24_max_sum_of_digits_is_achievable_l2146_214647

/-- Represents a time in 24-hour format -/
structure Time24 where
  hours : Nat
  minutes : Nat
  hour_valid : hours < 24
  minute_valid : minutes < 60

/-- Calculates the sum of digits for a natural number -/
def sumOfDigits (n : Nat) : Nat :=
  if n < 10 then n else (n % 10) + sumOfDigits (n / 10)

/-- Calculates the sum of digits for a Time24 -/
def timeSumOfDigits (t : Time24) : Nat :=
  sumOfDigits t.hours + sumOfDigits t.minutes

/-- The maximum sum of digits possible in a 24-hour format display -/
def maxSumOfDigits : Nat := 24

theorem max_sum_of_digits_is_24 :
  ∀ t : Time24, timeSumOfDigits t ≤ maxSumOfDigits :=
by sorry

theorem max_sum_of_digits_is_achievable :
  ∃ t : Time24, timeSumOfDigits t = maxSumOfDigits :=
by sorry

end NUMINAMATH_CALUDE_max_sum_of_digits_is_24_max_sum_of_digits_is_achievable_l2146_214647


namespace NUMINAMATH_CALUDE_initial_birds_in_tree_l2146_214644

theorem initial_birds_in_tree (additional_birds : ℕ) (total_birds : ℕ) 
  (h1 : additional_birds = 38) 
  (h2 : total_birds = 217) : 
  total_birds - additional_birds = 179 := by
  sorry

end NUMINAMATH_CALUDE_initial_birds_in_tree_l2146_214644


namespace NUMINAMATH_CALUDE_hadley_walk_distance_l2146_214674

/-- The total distance Hadley walked in his boots -/
def total_distance (grocery_store_distance pet_store_distance home_distance : ℕ) : ℕ :=
  grocery_store_distance + pet_store_distance + home_distance

/-- Theorem stating the total distance Hadley walked -/
theorem hadley_walk_distance :
  ∃ (grocery_store_distance pet_store_distance home_distance : ℕ),
    grocery_store_distance = 2 ∧
    pet_store_distance = 2 - 1 ∧
    home_distance = 4 - 1 ∧
    total_distance grocery_store_distance pet_store_distance home_distance = 6 := by
  sorry

end NUMINAMATH_CALUDE_hadley_walk_distance_l2146_214674


namespace NUMINAMATH_CALUDE_omega_range_l2146_214606

theorem omega_range (ω : ℝ) (h_pos : ω > 0) :
  (∃ a b : ℝ, π ≤ a ∧ a < b ∧ b ≤ 2*π ∧ Real.sin (ω * a) + Real.sin (ω * b) = 2) →
  (ω ∈ Set.Icc (9/4) (5/2) ∪ Set.Ici (13/4)) :=
by sorry

end NUMINAMATH_CALUDE_omega_range_l2146_214606


namespace NUMINAMATH_CALUDE_distinguishable_cube_colorings_eq_30240_l2146_214633

/-- The number of colors available to paint the cube. -/
def num_colors : ℕ := 10

/-- The number of faces on the cube. -/
def num_faces : ℕ := 6

/-- The number of rotational symmetries of a cube. -/
def cube_rotations : ℕ := 24

/-- Calculates the number of distinguishable ways to paint a cube. -/
def distinguishable_cube_colorings : ℕ :=
  (num_colors * (num_colors - 1) * (num_colors - 2) * (num_colors - 3) * 
   (num_colors - 4) * (num_colors - 5)) / cube_rotations

/-- Theorem stating that the number of distinguishable ways to paint the cube is 30240. -/
theorem distinguishable_cube_colorings_eq_30240 :
  distinguishable_cube_colorings = 30240 := by
  sorry

end NUMINAMATH_CALUDE_distinguishable_cube_colorings_eq_30240_l2146_214633


namespace NUMINAMATH_CALUDE_strawberry_theft_l2146_214658

/-- Calculates the number of stolen strawberries given the daily harvest rate, 
    number of days, strawberries given away, and final count. -/
def stolen_strawberries (daily_harvest : ℕ) (days : ℕ) (given_away : ℕ) (final_count : ℕ) : ℕ :=
  daily_harvest * days - given_away - final_count

/-- Proves that the number of stolen strawberries is 30 given the specific conditions. -/
theorem strawberry_theft : 
  stolen_strawberries 5 30 20 100 = 30 := by
  sorry

end NUMINAMATH_CALUDE_strawberry_theft_l2146_214658


namespace NUMINAMATH_CALUDE_symmetric_point_wrt_origin_l2146_214620

/-- Given a point P with coordinates (3, -4), this theorem proves that its symmetric point
    with respect to the origin has coordinates (-3, 4). -/
theorem symmetric_point_wrt_origin :
  let P : ℝ × ℝ := (3, -4)
  let symmetric_point := (-P.1, -P.2)
  symmetric_point = (-3, 4) := by
  sorry

end NUMINAMATH_CALUDE_symmetric_point_wrt_origin_l2146_214620


namespace NUMINAMATH_CALUDE_lee_cookies_proportion_l2146_214646

/-- Given that Lee can make 24 cookies with 3 cups of flour, 
    this theorem proves he can make 40 cookies with 5 cups of flour, 
    assuming a proportional relationship between flour and cookies. -/
theorem lee_cookies_proportion (flour_cups : ℚ) (cookies : ℕ) 
  (h1 : flour_cups > 0)
  (h2 : cookies > 0)
  (h3 : flour_cups / 3 = cookies / 24) :
  5 * cookies / flour_cups = 40 := by
  sorry

end NUMINAMATH_CALUDE_lee_cookies_proportion_l2146_214646


namespace NUMINAMATH_CALUDE_geometric_sequence_a10_l2146_214655

def is_geometric_sequence (a : ℕ → ℤ) : Prop :=
  ∃ q : ℤ, ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_a10 (a : ℕ → ℤ) (q : ℤ) :
  is_geometric_sequence a →
  (∃ q : ℤ, ∀ n : ℕ, a (n + 1) = a n * q) →
  a 4 * a 7 = -512 →
  a 3 + a 8 = 124 →
  a 10 = 512 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_a10_l2146_214655


namespace NUMINAMATH_CALUDE_student_pairs_l2146_214601

theorem student_pairs (n : ℕ) (same_letter_pairs : ℕ) (total_pairs : ℕ) :
  n = 12 →
  same_letter_pairs = 3 →
  total_pairs = n.choose 2 →
  total_pairs - same_letter_pairs = 63 := by
  sorry

end NUMINAMATH_CALUDE_student_pairs_l2146_214601


namespace NUMINAMATH_CALUDE_bus_fare_problem_l2146_214618

/-- Represents the denominations of coins available -/
inductive Coin : Type
  | Ten : Coin
  | Fifteen : Coin
  | Twenty : Coin

/-- The value of a coin in kopecks -/
def coinValue : Coin → ℕ
  | Coin.Ten => 10
  | Coin.Fifteen => 15
  | Coin.Twenty => 20

/-- A list of coins -/
def CoinList : Type := List Coin

/-- The total value of a list of coins in kopecks -/
def totalValue (coins : CoinList) : ℕ :=
  coins.foldl (fun acc c => acc + coinValue c) 0

/-- A function that checks if it's possible to distribute coins to passengers -/
def canDistribute (coins : CoinList) (passengers : ℕ) (farePerPassenger : ℕ) : Prop :=
  ∃ (distribution : List CoinList),
    distribution.length = passengers ∧
    (∀ c ∈ distribution, totalValue c = farePerPassenger) ∧
    distribution.join = coins

theorem bus_fare_problem :
  (¬ ∃ (coins : CoinList), coins.length = 24 ∧ canDistribute coins 20 5) ∧
  (∃ (coins : CoinList), coins.length = 25 ∧ canDistribute coins 20 5) := by
  sorry

end NUMINAMATH_CALUDE_bus_fare_problem_l2146_214618


namespace NUMINAMATH_CALUDE_great_grandchildren_count_l2146_214649

theorem great_grandchildren_count (age : ℕ) (grandchildren : ℕ) (n : ℕ) 
  (h1 : age = 91)
  (h2 : grandchildren = 11)
  (h3 : grandchildren * n * age = n * 1000 + n) :
  n = 1 := by
  sorry

end NUMINAMATH_CALUDE_great_grandchildren_count_l2146_214649


namespace NUMINAMATH_CALUDE_identify_fake_coin_in_two_weighings_l2146_214660

/-- Represents the result of a weighing -/
inductive WeighResult
  | Equal : WeighResult
  | LeftHeavier : WeighResult
  | RightHeavier : WeighResult

/-- Represents a coin -/
inductive Coin
  | A : Coin
  | B : Coin
  | C : Coin
  | D : Coin

/-- Represents a weighing operation -/
def weigh (left right : List Coin) : WeighResult :=
  sorry

/-- Represents the process of identifying the fake coin -/
def identifyFakeCoin : Coin :=
  sorry

/-- Theorem stating that it's possible to identify the fake coin in two weighings -/
theorem identify_fake_coin_in_two_weighings :
  ∃ (fakeCoin : Coin),
    (∀ (c : Coin), c ≠ fakeCoin → (weigh [c] [fakeCoin] = WeighResult.Equal ↔ c = fakeCoin)) →
    identifyFakeCoin = fakeCoin :=
  sorry

end NUMINAMATH_CALUDE_identify_fake_coin_in_two_weighings_l2146_214660


namespace NUMINAMATH_CALUDE_complex_cube_root_of_unity_l2146_214657

theorem complex_cube_root_of_unity : (1/2 - Complex.I * (Real.sqrt 3)/2)^3 = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_cube_root_of_unity_l2146_214657


namespace NUMINAMATH_CALUDE_minus_six_otimes_minus_two_l2146_214615

-- Define the new operation ⊗
def otimes (a b : ℚ) : ℚ := a^2 + b

-- Theorem statement
theorem minus_six_otimes_minus_two : otimes (-6) (-2) = 34 := by sorry

end NUMINAMATH_CALUDE_minus_six_otimes_minus_two_l2146_214615


namespace NUMINAMATH_CALUDE_square_sum_product_equality_l2146_214681

theorem square_sum_product_equality : 
  (2^2 + 92 * 3^2) * (4^2 + 92 * 5^2) = 1388^2 + 92 * 2^2 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_product_equality_l2146_214681


namespace NUMINAMATH_CALUDE_bounded_function_satisfying_equation_l2146_214668

def is_bounded (f : ℤ → ℤ) : Prop :=
  ∃ M : ℤ, ∀ n : ℤ, |f n| ≤ M

def satisfies_equation (f : ℤ → ℤ) : Prop :=
  ∀ n k : ℤ, f (n + k) + f (k - n) = 2 * f k * f n

def is_zero_function (f : ℤ → ℤ) : Prop :=
  ∀ n : ℤ, f n = 0

def is_one_function (f : ℤ → ℤ) : Prop :=
  ∀ n : ℤ, f n = 1

def is_alternating_function (f : ℤ → ℤ) : Prop :=
  ∀ n : ℤ, f n = if n % 2 = 0 then 1 else -1

theorem bounded_function_satisfying_equation (f : ℤ → ℤ) 
  (h_bounded : is_bounded f) (h_satisfies : satisfies_equation f) :
  is_zero_function f ∨ is_one_function f ∨ is_alternating_function f :=
sorry

end NUMINAMATH_CALUDE_bounded_function_satisfying_equation_l2146_214668


namespace NUMINAMATH_CALUDE_hyperbola_triangle_perimeter_l2146_214617

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2/16 - y^2/9 = 1

-- Define the points A, B, F₁ (left focus), and F₂ (right focus)
variable (A B F₁ F₂ : ℝ × ℝ)

-- Define the conditions
def chord_passes_through_left_focus : Prop :=
  ∃ t : ℝ, A = (1 - t) • F₁ + t • B ∧ 0 ≤ t ∧ t ≤ 1

def chord_length_is_6 : Prop :=
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 6

-- Define the theorem
theorem hyperbola_triangle_perimeter
  (h1 : hyperbola F₁.1 F₁.2)
  (h2 : hyperbola F₂.1 F₂.2)
  (h3 : chord_passes_through_left_focus A B F₁)
  (h4 : chord_length_is_6 A B) :
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) +
  Real.sqrt ((A.1 - F₂.1)^2 + (A.2 - F₂.2)^2) +
  Real.sqrt ((B.1 - F₂.1)^2 + (B.2 - F₂.2)^2) = 28 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_triangle_perimeter_l2146_214617


namespace NUMINAMATH_CALUDE_y_equals_five_l2146_214612

/-- A line passing through the origin with slope 1/2 -/
def line_k (x y : ℝ) : Prop := y = (1/2) * x

/-- The theorem stating that under the given conditions, y must equal 5 -/
theorem y_equals_five (x y : ℝ) :
  line_k x 6 →
  line_k 10 y →
  x * y = 60 →
  y = 5 := by sorry

end NUMINAMATH_CALUDE_y_equals_five_l2146_214612


namespace NUMINAMATH_CALUDE_min_distance_parabola_point_l2146_214679

/-- The minimum value of |y| + |PQ| for a point P(x, y) on the parabola x² = -4y and Q(-2√2, 0) -/
theorem min_distance_parabola_point : 
  let Q : ℝ × ℝ := (-2 * Real.sqrt 2, 0)
  ∃ (min : ℝ), min = 2 ∧ 
    ∀ (P : ℝ × ℝ), (P.1 ^ 2 = -4 * P.2) → 
      abs P.2 + Real.sqrt ((P.1 - Q.1) ^ 2 + (P.2 - Q.2) ^ 2) ≥ min :=
by sorry

end NUMINAMATH_CALUDE_min_distance_parabola_point_l2146_214679


namespace NUMINAMATH_CALUDE_person_B_processes_8_components_per_hour_l2146_214691

/-- The number of components processed per hour by person B -/
def components_per_hour_B : ℕ := sorry

/-- The number of components processed per hour by person A -/
def components_per_hour_A : ℕ := components_per_hour_B + 2

/-- The time it takes for person A to process 25 components -/
def time_A : ℚ := 25 / components_per_hour_A

/-- The time it takes for person B to process 20 components -/
def time_B : ℚ := 20 / components_per_hour_B

/-- Theorem stating that person B processes 8 components per hour -/
theorem person_B_processes_8_components_per_hour :
  components_per_hour_B = 8 ∧ time_A = time_B := by sorry

end NUMINAMATH_CALUDE_person_B_processes_8_components_per_hour_l2146_214691


namespace NUMINAMATH_CALUDE_ring_weights_sum_to_total_l2146_214670

/-- The weight of the orange ring in ounces -/
def orange_weight : ℚ := 0.08333333333333333

/-- The weight of the purple ring in ounces -/
def purple_weight : ℚ := 0.3333333333333333

/-- The weight of the white ring in ounces -/
def white_weight : ℚ := 0.4166666666666667

/-- The total weight of all rings in ounces -/
def total_weight : ℚ := 0.8333333333333333

/-- Theorem stating that the sum of individual ring weights equals the total weight -/
theorem ring_weights_sum_to_total : 
  orange_weight + purple_weight + white_weight = total_weight := by
  sorry

end NUMINAMATH_CALUDE_ring_weights_sum_to_total_l2146_214670


namespace NUMINAMATH_CALUDE_triangle_angle_bisector_theorem_l2146_214614

/-- In a triangle ABC with ∠C = 120°, given sides a and b, and angle bisector lc,
    the equation 1/a + 1/b = 1/lc holds. -/
theorem triangle_angle_bisector_theorem (a b lc : ℝ) (ha : a > 0) (hb : b > 0) (hlc : lc > 0) :
  let angle_C : ℝ := 120 * Real.pi / 180
  1 / a + 1 / b = 1 / lc :=
by sorry

end NUMINAMATH_CALUDE_triangle_angle_bisector_theorem_l2146_214614


namespace NUMINAMATH_CALUDE_extended_ohara_triple_49_64_l2146_214684

/-- Definition of an Extended O'Hara triple -/
def is_extended_ohara_triple (a b x : ℕ) : Prop :=
  2 * Real.sqrt a + Real.sqrt b = x

/-- Theorem: If (49, 64, x) is an Extended O'Hara triple, then x = 22 -/
theorem extended_ohara_triple_49_64 (x : ℕ) :
  is_extended_ohara_triple 49 64 x → x = 22 := by
  sorry

end NUMINAMATH_CALUDE_extended_ohara_triple_49_64_l2146_214684


namespace NUMINAMATH_CALUDE_derivative_at_pi_third_l2146_214676

theorem derivative_at_pi_third (f : ℝ → ℝ) (h : ∀ x, f x = x + Real.sin x) :
  deriv f (π / 3) = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_derivative_at_pi_third_l2146_214676


namespace NUMINAMATH_CALUDE_f_3_bounds_l2146_214624

/-- Given a quadratic function f(x) = ax^2 - c with specific constraints on f(1) and f(2),
    prove that f(3) is bounded between -1 and 20. -/
theorem f_3_bounds (a c : ℝ) (h1 : -4 ≤ a - c ∧ a - c ≤ -1) (h2 : -1 ≤ 4*a - c ∧ 4*a - c ≤ 5) :
  -1 ≤ 9*a - c ∧ 9*a - c ≤ 20 := by
  sorry

end NUMINAMATH_CALUDE_f_3_bounds_l2146_214624


namespace NUMINAMATH_CALUDE_arithmetic_proof_l2146_214645

theorem arithmetic_proof : 4 * 5 - 3 + 2^3 - 3 * 2 = 19 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_proof_l2146_214645


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l2146_214628

theorem sufficient_not_necessary (a b : ℝ) :
  (((a - b) * a^2 < 0 → a < b) ∧
   ∃ a b : ℝ, a < b ∧ (a - b) * a^2 ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l2146_214628


namespace NUMINAMATH_CALUDE_alice_has_winning_strategy_l2146_214602

/-- Represents a game on a complete graph -/
structure GraphGame where
  n : ℕ  -- number of vertices
  m : ℕ  -- maximum number of edges Bob can direct per turn

/-- Represents a strategy for Alice -/
def Strategy := GraphGame → Bool

/-- Checks if a strategy is winning for Alice -/
def is_winning_strategy (s : Strategy) (g : GraphGame) : Prop :=
  ∀ (bob_moves : ℕ → Fin g.m), ∃ (cycle : List (Fin g.n)), 
    cycle.length > 0 ∧ 
    cycle.Nodup ∧
    (∀ (i : Fin cycle.length), 
      ∃ (edge_directed_by_alice : Bool), 
        edge_directed_by_alice = true)

/-- The main theorem stating that Alice has a winning strategy -/
theorem alice_has_winning_strategy : 
  ∃ (s : Strategy), is_winning_strategy s ⟨2014, 1000⟩ := by
  sorry


end NUMINAMATH_CALUDE_alice_has_winning_strategy_l2146_214602


namespace NUMINAMATH_CALUDE_star_value_l2146_214656

def star (a b : ℤ) : ℚ := 1 / a + 1 / b

theorem star_value (a b : ℤ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a + b = 10) (h4 : a * b = 24) :
  star a b = 5 / 12 := by
  sorry

end NUMINAMATH_CALUDE_star_value_l2146_214656


namespace NUMINAMATH_CALUDE_action_figures_ratio_l2146_214689

theorem action_figures_ratio (initial : ℕ) (sold : ℕ) (remaining : ℕ) : 
  initial = 24 →
  remaining = initial - sold →
  12 = remaining - remaining / 3 →
  (sold : ℚ) / initial = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_action_figures_ratio_l2146_214689


namespace NUMINAMATH_CALUDE_shoebox_height_l2146_214608

/-- The height of a rectangular shoebox given specific conditions -/
theorem shoebox_height (width : ℝ) (block_side : ℝ) (uncovered_area : ℝ)
  (h_width : width = 6)
  (h_block : block_side = 4)
  (h_uncovered : uncovered_area = 8)
  : width * (block_side * block_side + uncovered_area) / width = 4 := by
  sorry

end NUMINAMATH_CALUDE_shoebox_height_l2146_214608


namespace NUMINAMATH_CALUDE_symmetric_arrangement_exists_l2146_214680

/-- Represents a grid figure -/
structure GridFigure where
  -- Add necessary fields to represent a grid figure
  asymmetric : Bool

/-- Represents an arrangement of grid figures -/
structure Arrangement where
  figures : List GridFigure
  symmetric : Bool

/-- Given three identical asymmetric grid figures, 
    there exists a symmetric arrangement -/
theorem symmetric_arrangement_exists : 
  ∀ (f : GridFigure), 
    f.asymmetric → 
    ∃ (a : Arrangement), 
      a.figures.length = 3 ∧ 
      (∀ fig ∈ a.figures, fig = f) ∧ 
      a.symmetric :=
by
  sorry


end NUMINAMATH_CALUDE_symmetric_arrangement_exists_l2146_214680


namespace NUMINAMATH_CALUDE_min_distinct_values_l2146_214675

/-- Given a list of 3000 positive integers with a unique mode occurring exactly 12 times,
    the minimum number of distinct values in the list is 273. -/
theorem min_distinct_values (L : List ℕ+) : 
  L.length = 3000 →
  ∃! m : ℕ+, (L.count m = 12 ∧ ∀ n : ℕ+, L.count n ≤ L.count m) →
  L.toFinset.card ≥ 273 :=
by sorry

end NUMINAMATH_CALUDE_min_distinct_values_l2146_214675


namespace NUMINAMATH_CALUDE_cos_sum_seventh_roots_unity_l2146_214663

theorem cos_sum_seventh_roots_unity : 
  Real.cos (2 * π / 7) + Real.cos (4 * π / 7) + Real.cos (6 * π / 7) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_cos_sum_seventh_roots_unity_l2146_214663


namespace NUMINAMATH_CALUDE_total_books_calculation_l2146_214610

/-- The number of boxes containing children's books. -/
def num_boxes : ℕ := 5

/-- The number of children's books in each box. -/
def books_per_box : ℕ := 20

/-- The total number of children's books in all boxes. -/
def total_books : ℕ := num_boxes * books_per_box

theorem total_books_calculation : total_books = 100 := by
  sorry

end NUMINAMATH_CALUDE_total_books_calculation_l2146_214610


namespace NUMINAMATH_CALUDE_ball_exchange_game_theorem_l2146_214619

/-- Represents a game played by n girls exchanging balls. -/
def BallExchangeGame (n : ℕ) := Unit

/-- A game is nice if at the end nobody has her own ball. -/
def is_nice (game : BallExchangeGame n) : Prop := sorry

/-- A game is tiresome if at the end everybody has her initial ball. -/
def is_tiresome (game : BallExchangeGame n) : Prop := sorry

/-- There exists a nice game for n players. -/
def exists_nice_game (n : ℕ) : Prop :=
  ∃ (game : BallExchangeGame n), is_nice game

/-- There exists a tiresome game for n players. -/
def exists_tiresome_game (n : ℕ) : Prop :=
  ∃ (game : BallExchangeGame n), is_tiresome game

theorem ball_exchange_game_theorem (n : ℕ) (h : n ≥ 2) :
  (exists_nice_game n ↔ n ≠ 3) ∧
  (exists_tiresome_game n ↔ n % 4 = 0 ∨ n % 4 = 1) :=
sorry

end NUMINAMATH_CALUDE_ball_exchange_game_theorem_l2146_214619


namespace NUMINAMATH_CALUDE_sushi_father_lollipops_l2146_214695

/-- The number of lollipops Sushi's father bought -/
def initial_lollipops : ℕ := 12

/-- The number of lollipops eaten -/
def eaten_lollipops : ℕ := 5

/-- The number of lollipops left -/
def remaining_lollipops : ℕ := 7

theorem sushi_father_lollipops : 
  initial_lollipops = eaten_lollipops + remaining_lollipops :=
by sorry

end NUMINAMATH_CALUDE_sushi_father_lollipops_l2146_214695


namespace NUMINAMATH_CALUDE_expected_weekly_rainfall_is_20_point_5_l2146_214654

/-- Represents the possible daily rainfall amounts in inches -/
inductive DailyRainfall
  | NoRain
  | LightRain
  | HeavyRain

/-- The probability of each rainfall outcome -/
def rainProbability : DailyRainfall → ℝ
  | DailyRainfall.NoRain => 0.3
  | DailyRainfall.LightRain => 0.3
  | DailyRainfall.HeavyRain => 0.4

/-- The amount of rainfall for each outcome in inches -/
def rainAmount : DailyRainfall → ℝ
  | DailyRainfall.NoRain => 0
  | DailyRainfall.LightRain => 3
  | DailyRainfall.HeavyRain => 8

/-- The number of days in the week -/
def daysInWeek : ℕ := 5

/-- The expected total rainfall for the week -/
def expectedWeeklyRainfall : ℝ :=
  daysInWeek * (rainProbability DailyRainfall.NoRain * rainAmount DailyRainfall.NoRain +
                rainProbability DailyRainfall.LightRain * rainAmount DailyRainfall.LightRain +
                rainProbability DailyRainfall.HeavyRain * rainAmount DailyRainfall.HeavyRain)

/-- Theorem: The expected total rainfall for the week is 20.5 inches -/
theorem expected_weekly_rainfall_is_20_point_5 :
  expectedWeeklyRainfall = 20.5 := by sorry

end NUMINAMATH_CALUDE_expected_weekly_rainfall_is_20_point_5_l2146_214654


namespace NUMINAMATH_CALUDE_sum_of_digits_of_difference_of_squares_l2146_214635

def a : ℕ := 6666666
def b : ℕ := 3333333

-- Function to calculate the sum of digits
def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

theorem sum_of_digits_of_difference_of_squares :
  sum_of_digits ((a ^ 2) - (b ^ 2)) = 63 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_difference_of_squares_l2146_214635


namespace NUMINAMATH_CALUDE_negative_two_a_cubed_l2146_214677

theorem negative_two_a_cubed (a : ℝ) : (-2 * a)^3 = -8 * a^3 := by
  sorry

end NUMINAMATH_CALUDE_negative_two_a_cubed_l2146_214677


namespace NUMINAMATH_CALUDE_log_max_min_sum_l2146_214638

theorem log_max_min_sum (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (let f := fun x => Real.log x / Real.log a
   max (f a) (f (2 * a)) + min (f a) (f (2 * a)) = 3) →
  a = 2 := by sorry

end NUMINAMATH_CALUDE_log_max_min_sum_l2146_214638
