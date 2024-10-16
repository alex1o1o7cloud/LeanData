import Mathlib

namespace NUMINAMATH_CALUDE_fraction_difference_l75_7542

theorem fraction_difference : (7 : ℚ) / 4 - (2 : ℚ) / 3 = (13 : ℚ) / 12 := by
  sorry

end NUMINAMATH_CALUDE_fraction_difference_l75_7542


namespace NUMINAMATH_CALUDE_intersection_M_complement_N_l75_7577

def U : Set ℝ := Set.univ

def M : Set ℝ := {x | x^2 - 2*x = 0}

def N : Set ℝ := {x | x - 1 > 0}

theorem intersection_M_complement_N : M ∩ (U \ N) = {0} := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_complement_N_l75_7577


namespace NUMINAMATH_CALUDE_xy_plus_one_eq_x_plus_y_l75_7593

theorem xy_plus_one_eq_x_plus_y (x y : ℝ) :
  x * y + 1 = x + y ↔ x = 1 ∨ y = 1 := by
sorry

end NUMINAMATH_CALUDE_xy_plus_one_eq_x_plus_y_l75_7593


namespace NUMINAMATH_CALUDE_original_proposition_contrapositive_proposition_both_true_l75_7556

-- Define the quadratic equation
def has_real_roots (m : ℝ) : Prop :=
  ∃ x : ℝ, x^2 + x - m = 0

-- Original proposition
theorem original_proposition : 
  ∀ m : ℝ, m > 0 → has_real_roots m :=
sorry

-- Contrapositive of the original proposition
theorem contrapositive_proposition :
  ∀ m : ℝ, ¬(has_real_roots m) → ¬(m > 0) :=
sorry

-- Both the original proposition and its contrapositive are true
theorem both_true : 
  (∀ m : ℝ, m > 0 → has_real_roots m) ∧ 
  (∀ m : ℝ, ¬(has_real_roots m) → ¬(m > 0)) :=
sorry

end NUMINAMATH_CALUDE_original_proposition_contrapositive_proposition_both_true_l75_7556


namespace NUMINAMATH_CALUDE_trailer_homes_added_l75_7511

/-- Represents the number of trailer homes added 5 years ago -/
def added_homes : ℕ := sorry

/-- The initial number of trailer homes -/
def initial_homes : ℕ := 30

/-- The initial average age of trailer homes in years -/
def initial_avg_age : ℕ := 15

/-- The age of added homes when they were added, in years -/
def added_homes_age : ℕ := 3

/-- The number of years that have passed since new homes were added -/
def years_passed : ℕ := 5

/-- The current average age of all trailer homes in years -/
def current_avg_age : ℕ := 17

theorem trailer_homes_added :
  (initial_homes * (initial_avg_age + years_passed) + added_homes * (added_homes_age + years_passed)) /
  (initial_homes + added_homes) = current_avg_age →
  added_homes = 10 := by sorry

end NUMINAMATH_CALUDE_trailer_homes_added_l75_7511


namespace NUMINAMATH_CALUDE_larger_number_proof_l75_7560

theorem larger_number_proof (a b : ℕ+) : 
  (Nat.gcd a b = 23) →
  (Nat.lcm a b = 4186) →
  (a = 23 * 13) →
  (b = 23 * 14) →
  max a b = 322 := by
sorry

end NUMINAMATH_CALUDE_larger_number_proof_l75_7560


namespace NUMINAMATH_CALUDE_angle_in_first_or_third_quadrant_l75_7586

/-- An angle is in the first quadrant if it's between 0° and 90° -/
def is_first_quadrant (θ : ℝ) : Prop :=
  0 < θ ∧ θ < 90

/-- An angle is in the third quadrant if it's between 180° and 270° -/
def is_third_quadrant (θ : ℝ) : Prop :=
  180 < θ ∧ θ < 270

/-- The main theorem: for any integer k, the angle k·180° + 45° is in either the first or third quadrant -/
theorem angle_in_first_or_third_quadrant (k : ℤ) :
  let α := k * 180 + 45
  is_first_quadrant (α % 360) ∨ is_third_quadrant (α % 360) :=
sorry

end NUMINAMATH_CALUDE_angle_in_first_or_third_quadrant_l75_7586


namespace NUMINAMATH_CALUDE_parallelogram_area_32_18_l75_7590

/-- The area of a parallelogram with given base and height -/
def parallelogram_area (base height : ℝ) : ℝ := base * height

/-- Theorem: The area of a parallelogram with base 32 cm and height 18 cm is 576 square centimeters -/
theorem parallelogram_area_32_18 : parallelogram_area 32 18 = 576 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_32_18_l75_7590


namespace NUMINAMATH_CALUDE_add_1876_minutes_to_6am_l75_7521

/-- Represents time in 24-hour format -/
structure Time where
  hours : Nat
  minutes : Nat
  h_valid : hours < 24
  m_valid : minutes < 60

/-- Adds minutes to a given time -/
def addMinutes (t : Time) (m : Nat) : Time :=
  sorry

theorem add_1876_minutes_to_6am (start : Time) 
  (h_start : start.hours = 6 ∧ start.minutes = 0) :
  addMinutes start 1876 = Time.mk 13 16 sorry sorry :=
sorry

end NUMINAMATH_CALUDE_add_1876_minutes_to_6am_l75_7521


namespace NUMINAMATH_CALUDE_vector_sum_magnitude_l75_7529

/-- The angle between two vectors in radians -/
def angle_between (a b : ℝ × ℝ) : ℝ := sorry

theorem vector_sum_magnitude (a b : ℝ × ℝ) 
  (h1 : angle_between a b = π / 3)  -- 60 degrees in radians
  (h2 : a = (1, Real.sqrt 3))
  (h3 : Real.sqrt (b.1^2 + b.2^2) = 1) :  -- |b| = 1
  Real.sqrt (((a.1 + 2*b.1)^2) + ((a.2 + 2*b.2)^2)) = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_vector_sum_magnitude_l75_7529


namespace NUMINAMATH_CALUDE_square_area_from_diagonal_l75_7596

theorem square_area_from_diagonal (a b : ℝ) :
  let diagonal := a + b
  ∃ s : ℝ, s > 0 ∧ s * s = (1/2) * diagonal * diagonal :=
by
  sorry

end NUMINAMATH_CALUDE_square_area_from_diagonal_l75_7596


namespace NUMINAMATH_CALUDE_set_equality_implies_sum_l75_7552

def A (x y : ℝ) : Set ℝ := {x, y / x, 1}
def B (x y : ℝ) : Set ℝ := {x^2, x + y, 0}

theorem set_equality_implies_sum (x y : ℝ) (h : A x y = B x y) : x^2023 + y^2024 = -1 := by
  sorry

end NUMINAMATH_CALUDE_set_equality_implies_sum_l75_7552


namespace NUMINAMATH_CALUDE_five_letter_words_count_l75_7585

def word_count : ℕ := 26

theorem five_letter_words_count :
  (Finset.sum (Finset.range 4) (λ k => Nat.choose 5 k)) = word_count := by
  sorry

end NUMINAMATH_CALUDE_five_letter_words_count_l75_7585


namespace NUMINAMATH_CALUDE_original_price_calculation_l75_7597

/-- Given a sale price and a percent decrease, calculate the original price of an item. -/
theorem original_price_calculation (sale_price : ℝ) (percent_decrease : ℝ) 
  (h1 : sale_price = 75)
  (h2 : percent_decrease = 25) : 
  ∃ (original_price : ℝ), 
    original_price * (1 - percent_decrease / 100) = sale_price ∧ 
    original_price = 100 := by
  sorry

end NUMINAMATH_CALUDE_original_price_calculation_l75_7597


namespace NUMINAMATH_CALUDE_subtract_fraction_from_decimal_l75_7518

theorem subtract_fraction_from_decimal : 7.31 - (1 / 5 : ℚ) = 7.11 := by sorry

end NUMINAMATH_CALUDE_subtract_fraction_from_decimal_l75_7518


namespace NUMINAMATH_CALUDE_fifth_employee_speed_is_140_l75_7547

/-- Calculates the typing speed of the fifth employee given the team size, average speed, and speeds of four employees --/
def fifth_employee_speed (team_size : ℕ) (average_speed : ℕ) (speed1 speed2 speed3 speed4 : ℕ) : ℕ :=
  team_size * average_speed - speed1 - speed2 - speed3 - speed4

/-- Proves that the fifth employee's typing speed is 140 words per minute --/
theorem fifth_employee_speed_is_140 :
  fifth_employee_speed 5 80 64 76 91 89 = 140 := by
  sorry

end NUMINAMATH_CALUDE_fifth_employee_speed_is_140_l75_7547


namespace NUMINAMATH_CALUDE_base4_division_l75_7515

/-- Converts a base 4 number to base 10 -/
def base4ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (4 ^ i)) 0

/-- Converts a base 10 number to base 4 -/
def base10ToBase4 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) :=
      if m = 0 then acc
      else aux (m / 4) ((m % 4) :: acc)
    aux n []

/-- Theorem stating that the quotient of 1012₄ ÷ 12₄ is 23₄ -/
theorem base4_division :
  (base4ToBase10 [2, 1, 0, 1]) / (base4ToBase10 [2, 1]) = base4ToBase10 [3, 2] := by
  sorry

#eval base4ToBase10 [2, 1, 0, 1]  -- Should output 70
#eval base4ToBase10 [2, 1]        -- Should output 6
#eval base4ToBase10 [3, 2]        -- Should output 11
#eval base10ToBase4 23            -- Should output [2, 3]

end NUMINAMATH_CALUDE_base4_division_l75_7515


namespace NUMINAMATH_CALUDE_min_n_is_correct_l75_7580

/-- The minimum positive integer n for which (x^5 + 1/x)^n contains a constant term -/
def min_n : ℕ := 6

/-- Predicate to check if (x^5 + 1/x)^n contains a constant term -/
def has_constant_term (n : ℕ) : Prop :=
  ∃ r : ℕ, 5 * n = 6 * r

theorem min_n_is_correct :
  (has_constant_term min_n) ∧
  (∀ m : ℕ, m < min_n → ¬(has_constant_term m)) :=
by sorry

end NUMINAMATH_CALUDE_min_n_is_correct_l75_7580


namespace NUMINAMATH_CALUDE_base_flavors_count_l75_7543

/-- The number of variations for each base flavor of pizza -/
def variations : ℕ := 4

/-- The total number of pizza varieties available -/
def total_varieties : ℕ := 16

/-- The number of base flavors of pizza -/
def base_flavors : ℕ := total_varieties / variations

theorem base_flavors_count : base_flavors = 4 := by
  sorry

end NUMINAMATH_CALUDE_base_flavors_count_l75_7543


namespace NUMINAMATH_CALUDE_strictly_increasing_function_bounds_l75_7575

theorem strictly_increasing_function_bounds (k : ℕ) (f : ℕ → ℕ) 
  (h_incr : ∀ m n, m < n → f m < f n) 
  (h_comp : ∀ n, f (f n) = k * n) :
  ∀ n, (2 * k : ℚ) / (k + 1) * n ≤ f n ∧ (f n : ℚ) ≤ (k + 1 : ℚ) / 2 * n :=
by sorry

end NUMINAMATH_CALUDE_strictly_increasing_function_bounds_l75_7575


namespace NUMINAMATH_CALUDE_club_membership_l75_7557

theorem club_membership (total : ℕ) (lit : ℕ) (hist : ℕ) (both : ℕ) 
  (h1 : total = 80)
  (h2 : lit = 50)
  (h3 : hist = 40)
  (h4 : both = 25) :
  total - (lit + hist - both) = 15 := by
  sorry

end NUMINAMATH_CALUDE_club_membership_l75_7557


namespace NUMINAMATH_CALUDE_smallest_positive_z_l75_7568

open Real

theorem smallest_positive_z (x z : ℝ) (h1 : cos x = 0) (h2 : cos (x + z) = 1/2) :
  ∃ (z_min : ℝ), z_min = π/6 ∧ z_min > 0 ∧ ∀ (z' : ℝ), z' > 0 → cos x = 0 → cos (x + z') = 1/2 → z' ≥ z_min :=
sorry

end NUMINAMATH_CALUDE_smallest_positive_z_l75_7568


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l75_7594

theorem quadratic_inequality_solution_set :
  let f : ℝ → ℝ := λ x => x^2 - 50*x + 601
  let solution_set := {x : ℝ | f x ≤ 9}
  let lower_bound := (50 - Real.sqrt 132) / 2
  let upper_bound := (50 + Real.sqrt 132) / 2
  solution_set = Set.Icc lower_bound upper_bound :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l75_7594


namespace NUMINAMATH_CALUDE_f_second_derivative_positive_l75_7572

open Set

noncomputable def f (x : ℝ) : ℝ := x^2 - 2*x - 4*Real.log x

def f_domain : Set ℝ := {x : ℝ | x > 0}

noncomputable def f'' (x : ℝ) : ℝ := 2 + 4 / x^2

theorem f_second_derivative_positive :
  {x ∈ f_domain | f'' x > 0} = f_domain :=
sorry

end NUMINAMATH_CALUDE_f_second_derivative_positive_l75_7572


namespace NUMINAMATH_CALUDE_farm_animals_l75_7507

/-- The number of chickens on the farm -/
def num_chickens : ℕ := 49

/-- The number of ducks on the farm -/
def num_ducks : ℕ := 37

/-- The number of rabbits on the farm -/
def num_rabbits : ℕ := 21

theorem farm_animals :
  (num_ducks + num_rabbits = num_chickens + 9) →
  num_rabbits = 21 := by
  sorry

end NUMINAMATH_CALUDE_farm_animals_l75_7507


namespace NUMINAMATH_CALUDE_zero_natural_number_ambiguity_l75_7554

-- Define a type for natural number conventions
inductive NatConvention where
  | withZero    : NatConvention
  | withoutZero : NatConvention

-- Define a function that checks if 0 is a natural number based on the convention
def isZeroNatural (conv : NatConvention) : Prop :=
  match conv with
  | NatConvention.withZero    => True
  | NatConvention.withoutZero => False

-- Theorem statement
theorem zero_natural_number_ambiguity :
  ∃ (conv : NatConvention), isZeroNatural conv :=
sorry


end NUMINAMATH_CALUDE_zero_natural_number_ambiguity_l75_7554


namespace NUMINAMATH_CALUDE_prime_pairs_divisibility_l75_7517

theorem prime_pairs_divisibility (p q : Nat) : 
  Nat.Prime p ∧ Nat.Prime q ∧ 
  (p ∣ 5^q + 1) ∧ (q ∣ 5^p + 1) ↔ 
  ((p = 2 ∧ q = 13) ∨ (p = 13 ∧ q = 2) ∨ (p = 3 ∧ q = 7) ∨ (p = 7 ∧ q = 3)) :=
by sorry

end NUMINAMATH_CALUDE_prime_pairs_divisibility_l75_7517


namespace NUMINAMATH_CALUDE_square_flags_count_l75_7504

theorem square_flags_count (total_fabric : ℕ) (square_size wide_size tall_size : ℕ × ℕ)
  (wide_count tall_count : ℕ) (fabric_left : ℕ) :
  total_fabric = 1000 →
  square_size = (4, 4) →
  wide_size = (5, 3) →
  tall_size = (3, 5) →
  wide_count = 20 →
  tall_count = 10 →
  fabric_left = 294 →
  ∃ (square_count : ℕ),
    square_count = 16 ∧
    square_count * (square_size.1 * square_size.2) +
    wide_count * (wide_size.1 * wide_size.2) +
    tall_count * (tall_size.1 * tall_size.2) +
    fabric_left = total_fabric :=
by sorry

end NUMINAMATH_CALUDE_square_flags_count_l75_7504


namespace NUMINAMATH_CALUDE_equivalent_form_l75_7583

theorem equivalent_form (x : ℝ) (h : x < 0) :
  Real.sqrt (x / (1 - (x + 1) / x)) = -x * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_equivalent_form_l75_7583


namespace NUMINAMATH_CALUDE_additional_group_average_weight_l75_7574

theorem additional_group_average_weight 
  (initial_count : ℕ) 
  (additional_count : ℕ) 
  (weight_increase : ℝ) 
  (final_average : ℝ) : 
  initial_count = 30 →
  additional_count = 30 →
  weight_increase = 10 →
  final_average = 40 →
  let total_count := initial_count + additional_count
  let initial_average := final_average - weight_increase
  let initial_total_weight := initial_count * initial_average
  let final_total_weight := total_count * final_average
  let additional_total_weight := final_total_weight - initial_total_weight
  additional_total_weight / additional_count = 50 := by
sorry

end NUMINAMATH_CALUDE_additional_group_average_weight_l75_7574


namespace NUMINAMATH_CALUDE_fruit_arrangements_proof_l75_7533

def numFruitArrangements (apples oranges bananas totalDays : ℕ) : ℕ :=
  let bananasAsBlock := 1
  let nonBananaDays := totalDays - bananas + 1
  let arrangements := (Nat.factorial nonBananaDays) / 
    (Nat.factorial apples * Nat.factorial oranges * Nat.factorial bananasAsBlock)
  arrangements * nonBananaDays

theorem fruit_arrangements_proof :
  numFruitArrangements 4 3 3 10 = 2240 :=
by sorry

end NUMINAMATH_CALUDE_fruit_arrangements_proof_l75_7533


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l75_7522

/-- Given a geometric sequence with common ratio -1/3, 
    prove that the sum of odd-indexed terms up to a₇ 
    divided by the sum of even-indexed terms up to a₈ equals -3 -/
theorem geometric_sequence_ratio (a : ℕ → ℝ) :
  (∀ n : ℕ, a (n + 1) = a n * (-1/3)) →
  (a 1 + a 3 + a 5 + a 7) / (a 2 + a 4 + a 6 + a 8) = -3 := by
sorry


end NUMINAMATH_CALUDE_geometric_sequence_ratio_l75_7522


namespace NUMINAMATH_CALUDE_yellow_light_probability_l75_7532

theorem yellow_light_probability (red_duration green_duration yellow_duration : ℕ) :
  red_duration = 30 →
  yellow_duration = 5 →
  green_duration = 45 →
  (yellow_duration : ℚ) / (red_duration + yellow_duration + green_duration) = 1 / 16 := by
  sorry

end NUMINAMATH_CALUDE_yellow_light_probability_l75_7532


namespace NUMINAMATH_CALUDE_function_inequality_implies_parameter_bound_l75_7588

/-- Given f(x) = 2ln(x) - x^2 and g(x) = xe^x - (a-1)x^2 - x - 2ln(x) for x > 0,
    if f(x) + g(x) ≥ 0 for all x > 0, then a ≤ 1 -/
theorem function_inequality_implies_parameter_bound (a : ℝ) :
  (∀ x > 0, 2 * Real.log x - x^2 + x * Real.exp x - (a - 1) * x^2 - x - 2 * Real.log x ≥ 0) →
  a ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_implies_parameter_bound_l75_7588


namespace NUMINAMATH_CALUDE_age_difference_l75_7589

/-- Given a man and his son, prove that the man is 35 years older than his son. -/
theorem age_difference (man_age son_age : ℕ) : 
  man_age > son_age →
  son_age = 33 →
  man_age + 2 = 2 * (son_age + 2) →
  man_age - son_age = 35 := by
  sorry


end NUMINAMATH_CALUDE_age_difference_l75_7589


namespace NUMINAMATH_CALUDE_increasing_cubic_function_a_range_l75_7526

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + x

-- State the theorem
theorem increasing_cubic_function_a_range :
  (∀ x y : ℝ, x < y → f a x < f a y) → -Real.sqrt 3 ≤ a ∧ a ≤ Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_increasing_cubic_function_a_range_l75_7526


namespace NUMINAMATH_CALUDE_sqrt_eight_minus_sqrt_two_equals_sqrt_two_l75_7509

theorem sqrt_eight_minus_sqrt_two_equals_sqrt_two : 
  Real.sqrt 8 - Real.sqrt 2 = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_eight_minus_sqrt_two_equals_sqrt_two_l75_7509


namespace NUMINAMATH_CALUDE_number_of_children_l75_7549

/-- Given a group of children born at 2-year intervals with the youngest being 6 years old,
    and the sum of their ages being 50 years, prove that there are 5 children. -/
theorem number_of_children (sum_of_ages : ℕ) (age_difference : ℕ) (youngest_age : ℕ) 
  (h1 : sum_of_ages = 50)
  (h2 : age_difference = 2)
  (h3 : youngest_age = 6) :
  ∃ (n : ℕ), n = 5 ∧ 
  sum_of_ages = n * (youngest_age + (n - 1) * age_difference / 2) := by
  sorry

end NUMINAMATH_CALUDE_number_of_children_l75_7549


namespace NUMINAMATH_CALUDE_line_segment_param_sum_l75_7550

/-- Given a line segment connecting (1, -3) and (-4, 5), parameterized by x = pt + q and y = rt + s
    where 0 ≤ t ≤ 2 and t = 0 corresponds to (1, -3), prove that p^2 + q^2 + r^2 + s^2 = 32.25 -/
theorem line_segment_param_sum (p q r s : ℝ) : 
  (∀ t : ℝ, 0 ≤ t ∧ t ≤ 2 → p * t + q = 1 - 5 * t / 2 ∧ r * t + s = -3 + 4 * t) →
  p^2 + q^2 + r^2 + s^2 = 129/4 := by
sorry

end NUMINAMATH_CALUDE_line_segment_param_sum_l75_7550


namespace NUMINAMATH_CALUDE_expression_evaluation_l75_7570

theorem expression_evaluation (a b : ℚ) (h1 : a = 5) (h2 : b = 6) : 
  (3 * b) / (a + b) = 18 / 11 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l75_7570


namespace NUMINAMATH_CALUDE_power_sum_equals_39_l75_7558

theorem power_sum_equals_39 : 
  (-2)^4 + (-2)^3 + (-2)^2 + (-2)^1 + 3 + 2^1 + 2^2 + 2^3 + 2^4 = 39 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_equals_39_l75_7558


namespace NUMINAMATH_CALUDE_mean_goals_is_6_l75_7525

/-- The number of players who scored 5 goals -/
def players_5 : ℕ := 4

/-- The number of players who scored 6 goals -/
def players_6 : ℕ := 3

/-- The number of players who scored 7 goals -/
def players_7 : ℕ := 2

/-- The number of players who scored 8 goals -/
def players_8 : ℕ := 1

/-- The total number of goals scored -/
def total_goals : ℕ := 5 * players_5 + 6 * players_6 + 7 * players_7 + 8 * players_8

/-- The total number of players -/
def total_players : ℕ := players_5 + players_6 + players_7 + players_8

/-- The mean number of goals scored -/
def mean_goals : ℚ := total_goals / total_players

theorem mean_goals_is_6 : mean_goals = 6 := by sorry

end NUMINAMATH_CALUDE_mean_goals_is_6_l75_7525


namespace NUMINAMATH_CALUDE_factorization_4m_squared_minus_16_l75_7578

theorem factorization_4m_squared_minus_16 (m : ℝ) :
  4 * m^2 - 16 = 4 * (m + 2) * (m - 2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_4m_squared_minus_16_l75_7578


namespace NUMINAMATH_CALUDE_bottle_capacity_correct_l75_7536

/-- The capacity of Madeline's water bottle in ounces -/
def bottle_capacity : ℕ := 12

/-- The number of times Madeline refills her water bottle -/
def refills : ℕ := 7

/-- The amount of water Madeline needs to drink after refills in ounces -/
def remaining_water : ℕ := 16

/-- The total amount of water Madeline wants to drink in a day in ounces -/
def total_water : ℕ := 100

/-- Theorem stating that the bottle capacity is correct given the conditions -/
theorem bottle_capacity_correct :
  bottle_capacity * refills + remaining_water = total_water :=
by sorry

end NUMINAMATH_CALUDE_bottle_capacity_correct_l75_7536


namespace NUMINAMATH_CALUDE_bankers_gain_is_nine_l75_7548

/-- Calculates the banker's gain given the true discount, time period, and interest rate. -/
def bankers_gain (true_discount : ℚ) (time : ℚ) (rate : ℚ) : ℚ :=
  let face_value := (true_discount * (100 + (rate * time))) / (rate * time)
  let bankers_discount := (face_value * rate * time) / 100
  bankers_discount - true_discount

/-- Theorem stating that the banker's gain is 9 given the specified conditions. -/
theorem bankers_gain_is_nine :
  bankers_gain 75 1 12 = 9 := by
  sorry

#eval bankers_gain 75 1 12

end NUMINAMATH_CALUDE_bankers_gain_is_nine_l75_7548


namespace NUMINAMATH_CALUDE_toys_per_rabbit_l75_7524

def monday_toys : ℕ := 8
def num_rabbits : ℕ := 34

def total_toys : ℕ :=
  monday_toys +  -- Monday
  (3 * monday_toys) +  -- Tuesday
  (2 * 3 * monday_toys) +  -- Wednesday
  monday_toys +  -- Thursday
  (5 * monday_toys) +  -- Friday
  (2 * 3 * monday_toys) / 2  -- Saturday

theorem toys_per_rabbit : total_toys / num_rabbits = 4 := by
  sorry

end NUMINAMATH_CALUDE_toys_per_rabbit_l75_7524


namespace NUMINAMATH_CALUDE_intersection_point_coordinates_l75_7501

/-- A line in 2D space represented by ax + by = c --/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in 2D space --/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line --/
def Point.onLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y = l.c

/-- Check if two lines are perpendicular --/
def Line.perpendicular (l1 l2 : Line) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

/-- The given line l --/
def l : Line := { a := 2, b := 1, c := 10 }

/-- The point through which l' passes --/
def p : Point := { x := -10, y := 0 }

/-- The theorem to prove --/
theorem intersection_point_coordinates :
  ∃ (l' : Line),
    (p.onLine l') ∧
    (l.perpendicular l') ∧
    (∃ (q : Point), q.onLine l ∧ q.onLine l' ∧ q.x = 2 ∧ q.y = 6) :=
by sorry

end NUMINAMATH_CALUDE_intersection_point_coordinates_l75_7501


namespace NUMINAMATH_CALUDE_evaluate_expression_l75_7592

theorem evaluate_expression : 6 - 8 * (9 - 2^3 + 12/3) * 5 = -194 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l75_7592


namespace NUMINAMATH_CALUDE_joan_book_revenue_l75_7539

def total_revenue (total_books : ℕ) (books_at_4 : ℕ) (books_at_7 : ℕ) (price_4 : ℕ) (price_7 : ℕ) (price_10 : ℕ) : ℕ :=
  let remaining_books := total_books - books_at_4 - books_at_7
  books_at_4 * price_4 + books_at_7 * price_7 + remaining_books * price_10

theorem joan_book_revenue :
  total_revenue 33 15 6 4 7 10 = 222 := by
  sorry

end NUMINAMATH_CALUDE_joan_book_revenue_l75_7539


namespace NUMINAMATH_CALUDE_arcade_candy_cost_l75_7508

theorem arcade_candy_cost (whack_a_mole_tickets : ℕ) (skee_ball_tickets : ℕ) (candies : ℕ) :
  whack_a_mole_tickets = 26 →
  skee_ball_tickets = 19 →
  candies = 5 →
  (whack_a_mole_tickets + skee_ball_tickets) / candies = 9 :=
by sorry

end NUMINAMATH_CALUDE_arcade_candy_cost_l75_7508


namespace NUMINAMATH_CALUDE_spherical_segment_volume_ratio_l75_7563

theorem spherical_segment_volume_ratio (α : ℝ) :
  let R : ℝ := 1  -- Assume unit sphere for simplicity
  let V_sphere : ℝ := (4 / 3) * Real.pi * R^3
  let H : ℝ := 2 * R * Real.sin (α / 4)^2
  let V_seg : ℝ := Real.pi * H^2 * (R - H / 3)
  V_seg / V_sphere = Real.sin (α / 4)^4 * (2 + Real.cos (α / 2)) :=
by sorry

end NUMINAMATH_CALUDE_spherical_segment_volume_ratio_l75_7563


namespace NUMINAMATH_CALUDE_fahrenheit_celsius_conversion_l75_7587

theorem fahrenheit_celsius_conversion (F C : ℝ) : 
  C = (5 / 9) * (F - 32) → C = 40 → F = 104 := by
  sorry

end NUMINAMATH_CALUDE_fahrenheit_celsius_conversion_l75_7587


namespace NUMINAMATH_CALUDE_max_sum_sqrt_inequality_l75_7523

theorem max_sum_sqrt_inequality (x y z : ℝ) 
  (pos_x : 0 < x) (pos_y : 0 < y) (pos_z : 0 < z) 
  (sum_eq_three : x + y + z = 3) : 
  Real.sqrt (2 * x + 1) + Real.sqrt (2 * y + 1) + Real.sqrt (2 * z + 1) ≤ 3 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_max_sum_sqrt_inequality_l75_7523


namespace NUMINAMATH_CALUDE_ten_bulb_signals_l75_7573

/-- The number of different signals that can be transmitted using a given number of light bulbs -/
def signalCount (n : ℕ) : ℕ := 2^n

/-- Theorem: The number of different signals that can be transmitted using 10 light bulbs, 
    each of which can be either on or off, is equal to 2^10 (1024) -/
theorem ten_bulb_signals : signalCount 10 = 1024 := by
  sorry

end NUMINAMATH_CALUDE_ten_bulb_signals_l75_7573


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l75_7564

theorem sqrt_equation_solution (x : ℝ) : Real.sqrt (x + 11) = 10 → x = 89 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l75_7564


namespace NUMINAMATH_CALUDE_circle_center_l75_7512

/-- Represents a point in polar coordinates -/
structure PolarPoint where
  r : ℝ
  θ : ℝ

/-- Represents a circle in polar coordinates -/
structure PolarCircle where
  equation : ℝ → ℝ → Prop

theorem circle_center (C : PolarCircle) :
  C.equation = (fun ρ θ ↦ ρ = 2 * Real.cos (θ + π/4)) →
  ∃ (center : PolarPoint), center.r = 1 ∧ center.θ = -π/4 := by
  sorry

end NUMINAMATH_CALUDE_circle_center_l75_7512


namespace NUMINAMATH_CALUDE_equation_solution_l75_7599

theorem equation_solution : ∃! r : ℚ, (r^2 - 5*r + 4)/(r^2 - 8*r + 7) = (r^2 - 2*r - 15)/(r^2 - r - 20) ∧ r = -5/4 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l75_7599


namespace NUMINAMATH_CALUDE_total_hired_is_35_l75_7598

/-- Represents the daily pay for heavy equipment operators -/
def heavy_equipment_pay : ℕ := 140

/-- Represents the daily pay for general laborers -/
def general_laborer_pay : ℕ := 90

/-- Represents the total payroll -/
def total_payroll : ℕ := 3950

/-- Represents the number of general laborers employed -/
def num_laborers : ℕ := 19

/-- Calculates the total number of people hired given the conditions -/
def total_hired : ℕ := 
  let num_operators := (total_payroll - general_laborer_pay * num_laborers) / heavy_equipment_pay
  num_operators + num_laborers

/-- Proves that the total number of people hired is 35 -/
theorem total_hired_is_35 : total_hired = 35 := by
  sorry

end NUMINAMATH_CALUDE_total_hired_is_35_l75_7598


namespace NUMINAMATH_CALUDE_smallest_x_for_cube_equation_l75_7537

theorem smallest_x_for_cube_equation (N : ℕ+) (h : 1260 * x = N^3) : 
  ∃ (x : ℕ), x = 7350 ∧ ∀ (y : ℕ), 1260 * y = N^3 → x ≤ y := by
  sorry

end NUMINAMATH_CALUDE_smallest_x_for_cube_equation_l75_7537


namespace NUMINAMATH_CALUDE_equation_satisfied_l75_7569

theorem equation_satisfied (x y z : ℤ) (h1 : x = z) (h2 : y - 1 = x) :
  x * (x - y) + y * (y - z) + z * (z - x) = 1 := by
  sorry

end NUMINAMATH_CALUDE_equation_satisfied_l75_7569


namespace NUMINAMATH_CALUDE_exponent_division_l75_7576

theorem exponent_division (a : ℝ) : a^4 / a = a^3 := by
  sorry

end NUMINAMATH_CALUDE_exponent_division_l75_7576


namespace NUMINAMATH_CALUDE_odd_solution_exists_l75_7528

theorem odd_solution_exists (k m n : ℕ+) (h : m * n = k^2 + k + 3) :
  (∃ (x y : ℤ), x^2 + 11 * y^2 = 4 * m ∧ x % 2 ≠ 0 ∧ y % 2 ≠ 0) ∨
  (∃ (x y : ℤ), x^2 + 11 * y^2 = 4 * n ∧ x % 2 ≠ 0 ∧ y % 2 ≠ 0) :=
sorry

end NUMINAMATH_CALUDE_odd_solution_exists_l75_7528


namespace NUMINAMATH_CALUDE_intersection_implies_a_value_l75_7535

def P (a : ℝ) : Set ℝ := {a^2, a+1, -3}
def Q (a : ℝ) : Set ℝ := {a^2+1, 2*a-1, a-3}

theorem intersection_implies_a_value :
  ∀ a : ℝ, P a ∩ Q a = {-3} → a = -1 := by
sorry

end NUMINAMATH_CALUDE_intersection_implies_a_value_l75_7535


namespace NUMINAMATH_CALUDE_bean_jar_count_bean_jar_count_proof_l75_7502

theorem bean_jar_count : ℕ → Prop :=
  fun total_beans =>
    let red_beans := total_beans / 4
    let remaining_after_red := total_beans - red_beans
    let white_beans := remaining_after_red / 3
    let remaining_after_white := remaining_after_red - white_beans
    let green_beans := remaining_after_white / 2
    (green_beans = 143) → (total_beans = 572)

theorem bean_jar_count_proof : bean_jar_count 572 := by
  sorry

end NUMINAMATH_CALUDE_bean_jar_count_bean_jar_count_proof_l75_7502


namespace NUMINAMATH_CALUDE_books_owned_by_three_l75_7500

/-- The number of books owned by Harry, Flora, and Gary -/
def total_books (harry_books : ℕ) : ℕ :=
  let flora_books := 2 * harry_books
  let gary_books := harry_books / 2
  harry_books + flora_books + gary_books

/-- Theorem stating that the total number of books is 175 when Harry has 50 books -/
theorem books_owned_by_three (harry_books : ℕ) 
  (h : harry_books = 50) : total_books harry_books = 175 := by
  sorry

end NUMINAMATH_CALUDE_books_owned_by_three_l75_7500


namespace NUMINAMATH_CALUDE_parabola_properties_l75_7579

/-- Parabola intersecting x-axis -/
def parabola (m : ℝ) (x : ℝ) : ℝ := x^2 - (m^2 + 4) * x - 2 * m^2 - 12

/-- Discriminant of the parabola -/
def discriminant (m : ℝ) : ℝ := (m^2 + 4)^2 + 4 * (2 * m^2 + 12)

/-- Chord length of the parabola intersecting x-axis -/
def chord_length (m : ℝ) : ℝ := m^2 + 8

theorem parabola_properties (m : ℝ) :
  (∀ m, discriminant m > 0) ∧
  (chord_length m = m^2 + 8) ∧
  (∀ m, chord_length m ≥ 8) ∧
  (chord_length 0 = 8) := by
  sorry

end NUMINAMATH_CALUDE_parabola_properties_l75_7579


namespace NUMINAMATH_CALUDE_simplify_fraction_product_l75_7566

theorem simplify_fraction_product : 4 * (15 / 5) * (25 / -75) = -4 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_product_l75_7566


namespace NUMINAMATH_CALUDE_salary_remaining_l75_7562

def salary : ℕ := 180000

def food_fraction : ℚ := 1/5
def rent_fraction : ℚ := 1/10
def clothes_fraction : ℚ := 3/5

def remaining_money : ℕ := 18000

theorem salary_remaining :
  salary - (↑salary * (food_fraction + rent_fraction + clothes_fraction)).floor = remaining_money := by
  sorry

end NUMINAMATH_CALUDE_salary_remaining_l75_7562


namespace NUMINAMATH_CALUDE_quadratic_roots_reciprocal_l75_7516

theorem quadratic_roots_reciprocal (a b : ℝ) (ha : a ≠ 0) :
  let f : ℝ → ℝ := λ x => a * x^2 + b * x + a
  ∀ x₁ x₂ : ℝ, f x₁ = 0 ∧ f x₂ = 0 → x₁ = 1 / x₂ ∧ x₂ = 1 / x₁ := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_reciprocal_l75_7516


namespace NUMINAMATH_CALUDE_tan_negative_3645_degrees_l75_7581

theorem tan_negative_3645_degrees : Real.tan ((-3645 : ℝ) * π / 180) = -1 := by
  sorry

end NUMINAMATH_CALUDE_tan_negative_3645_degrees_l75_7581


namespace NUMINAMATH_CALUDE_arun_weight_average_l75_7510

def arun_weight_range (w : ℝ) : Prop :=
  62 < w ∧ w < 72 ∧ 60 < w ∧ w < 70 ∧ w ≤ 65

theorem arun_weight_average :
  ∃ (min max : ℝ),
    (∀ w, arun_weight_range w → min ≤ w ∧ w ≤ max) ∧
    (∃ w₁ w₂, arun_weight_range w₁ ∧ arun_weight_range w₂ ∧ w₁ = min ∧ w₂ = max) ∧
    (min + max) / 2 = 63.5 :=
sorry

end NUMINAMATH_CALUDE_arun_weight_average_l75_7510


namespace NUMINAMATH_CALUDE_parabola_and_line_properties_l75_7595

-- Define the parabola
structure Parabola where
  vertex : ℝ × ℝ
  focus : ℝ × ℝ
  a : ℝ

-- Define a point on the parabola
def PointOnParabola (p : Parabola) (x y : ℝ) : Prop :=
  y = p.a * x^2

-- Define a circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a line
structure Line where
  slope : ℝ
  y_intercept : ℝ

-- Main theorem
theorem parabola_and_line_properties
  (p : Parabola)
  (A : ℝ × ℝ)
  (c : Circle)
  (m : Line) :
  p.vertex = (0, 0) →
  p.focus.1 = 0 →
  p.focus.2 > 0 →
  PointOnParabola p A.1 A.2 →
  c.center = A →
  c.radius = 2 →
  c.center.2 - c.radius = p.focus.2 →
  m.y_intercept = 6 →
  ∃ (P Q : ℝ × ℝ),
    PointOnParabola p P.1 P.2 ∧
    PointOnParabola p Q.1 Q.2 ∧
    P.2 = m.slope * P.1 + m.y_intercept ∧
    Q.2 = m.slope * Q.1 + m.y_intercept →
  (∀ (x y : ℝ), y = p.a * x^2 ↔ y = (1/4) * x^2) ∧
  (m.slope = 1/2 ∨ m.slope = -1/2) :=
sorry

end NUMINAMATH_CALUDE_parabola_and_line_properties_l75_7595


namespace NUMINAMATH_CALUDE_chess_team_boys_l75_7519

/-- Represents the number of boys on a chess team --/
def num_boys (total : ℕ) (attendees : ℕ) : ℕ :=
  total - 2 * (total - attendees)

/-- Theorem stating the number of boys on the chess team --/
theorem chess_team_boys (total : ℕ) (attendees : ℕ) 
  (h_total : total = 30)
  (h_attendees : attendees = 18)
  (h_attendance : ∃ (girls : ℕ), girls + (total - girls) = total ∧ 
                                  girls / 3 + (total - girls) = attendees) :
  num_boys total attendees = 12 := by
sorry

#eval num_boys 30 18  -- Should output 12

end NUMINAMATH_CALUDE_chess_team_boys_l75_7519


namespace NUMINAMATH_CALUDE_no_solution_equation_one_unique_solution_equation_two_l75_7584

-- Problem 1
theorem no_solution_equation_one (x : ℝ) : 
  (x ≠ 2) → (1 / (x - 2) ≠ (1 - x) / (2 - x) - 3) :=
by sorry

-- Problem 2
theorem unique_solution_equation_two :
  ∃! x : ℝ, (x ≠ 1) ∧ (x^2 ≠ 1) ∧ (x / (x - 1) - (2*x - 1) / (x^2 - 1) = 1) :=
by sorry

end NUMINAMATH_CALUDE_no_solution_equation_one_unique_solution_equation_two_l75_7584


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l75_7561

theorem sum_of_coefficients (P a b c d e f : ℕ) : 
  20112011 = a * P^5 + b * P^4 + c * P^3 + d * P^2 + e * P + f →
  a < P ∧ b < P ∧ c < P ∧ d < P ∧ e < P ∧ f < P →
  a + b + c + d + e + f = 36 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l75_7561


namespace NUMINAMATH_CALUDE_inequality_properties_l75_7565

theorem inequality_properties (m n : ℝ) :
  (∀ a : ℝ, a ≠ 0 → m * a^2 < n * a^2 → m < n) ∧
  (m < n → n < 0 → n / m < 1) := by
  sorry

end NUMINAMATH_CALUDE_inequality_properties_l75_7565


namespace NUMINAMATH_CALUDE_inequality_system_solution_l75_7514

theorem inequality_system_solution (x : ℝ) :
  (4 * x - 6 < 2 * x ∧ (3 * x - 1) / 2 ≥ 2 * x - 1) ↔ x ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l75_7514


namespace NUMINAMATH_CALUDE_smallest_common_factor_l75_7534

theorem smallest_common_factor (n : ℕ) : 
  (∀ m : ℕ, 0 < m → m < 7 → ¬(∃ k : ℕ, k > 1 ∧ k ∣ (8*m - 3) ∧ k ∣ (5*m + 4))) ∧
  (∃ k : ℕ, k > 1 ∧ k ∣ (8*7 - 3) ∧ k ∣ (5*7 + 4)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_common_factor_l75_7534


namespace NUMINAMATH_CALUDE_orchids_unchanged_l75_7591

/-- The number of orchids in a vase remains unchanged when only roses are added --/
theorem orchids_unchanged 
  (initial_roses : ℕ) 
  (initial_orchids : ℕ) 
  (final_roses : ℕ) 
  (roses_added : ℕ) : 
  initial_roses = 15 → 
  initial_orchids = 62 → 
  final_roses = 17 → 
  roses_added = 2 → 
  final_roses = initial_roses + roses_added → 
  initial_orchids = 62 := by
sorry

end NUMINAMATH_CALUDE_orchids_unchanged_l75_7591


namespace NUMINAMATH_CALUDE_trig_identity_l75_7545

theorem trig_identity (α : Real) (h : Real.tan α = 1/2) :
  (Real.cos α + Real.sin α) / (Real.cos α - Real.sin α) = 3 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l75_7545


namespace NUMINAMATH_CALUDE_prime_with_integer_roots_l75_7513

theorem prime_with_integer_roots (p : ℕ) : 
  Prime p → 
  (∃ x y : ℤ, x^2 + p*x - 300*p = 0 ∧ y^2 + p*y - 300*p = 0) → 
  1 < p ∧ p ≤ 11 := by
sorry

end NUMINAMATH_CALUDE_prime_with_integer_roots_l75_7513


namespace NUMINAMATH_CALUDE_probability_divisible_by_9_l75_7520

def number_set : Set ℕ := {n | 8 ≤ n ∧ n ≤ 28}

def is_divisible_by_9 (a b c : ℕ) : Prop :=
  (a + b + c) % 9 = 0

def favorable_outcomes : ℕ := 150

def total_outcomes : ℕ := 1330

theorem probability_divisible_by_9 :
  (favorable_outcomes : ℚ) / total_outcomes = 15 / 133 := by
  sorry

end NUMINAMATH_CALUDE_probability_divisible_by_9_l75_7520


namespace NUMINAMATH_CALUDE_probability_two_qualified_products_l75_7567

theorem probability_two_qualified_products (total : ℕ) (qualified : ℕ) (unqualified : ℕ) 
  (h1 : total = qualified + unqualified)
  (h2 : total = 10)
  (h3 : qualified = 8)
  (h4 : unqualified = 2) :
  let p := (qualified - 1) / (total - 1)
  p = 7 / 11 := by
sorry

end NUMINAMATH_CALUDE_probability_two_qualified_products_l75_7567


namespace NUMINAMATH_CALUDE_vector_difference_magnitude_l75_7531

/-- Given vectors a and b in ℝ², prove that the magnitude of their difference is 5. -/
theorem vector_difference_magnitude (a b : ℝ × ℝ) : 
  a = (2, 1) → b = (-2, 4) → ‖a - b‖ = 5 := by sorry

end NUMINAMATH_CALUDE_vector_difference_magnitude_l75_7531


namespace NUMINAMATH_CALUDE_only_equilateral_forms_triangle_l75_7551

/-- A function that checks if three line segments can form a triangle -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- The given sets of line segments -/
def segment_sets : List (ℝ × ℝ × ℝ) :=
  [(3, 4, 8), (5, 6, 11), (4, 4, 8), (8, 8, 8)]

/-- Theorem stating that only (8, 8, 8) can form a triangle among the given sets -/
theorem only_equilateral_forms_triangle :
  ∃! set : ℝ × ℝ × ℝ, set ∈ segment_sets ∧ can_form_triangle set.1 set.2.1 set.2.2 :=
by sorry

end NUMINAMATH_CALUDE_only_equilateral_forms_triangle_l75_7551


namespace NUMINAMATH_CALUDE_train_bus_cost_difference_proof_l75_7506

def train_bus_cost_difference (train_cost bus_cost : ℝ) : Prop :=
  (train_cost > bus_cost) ∧
  (train_cost + bus_cost = 9.65) ∧
  (bus_cost = 1.40) ∧
  (train_cost - bus_cost = 6.85)

theorem train_bus_cost_difference_proof :
  ∃ (train_cost bus_cost : ℝ), train_bus_cost_difference train_cost bus_cost :=
by sorry

end NUMINAMATH_CALUDE_train_bus_cost_difference_proof_l75_7506


namespace NUMINAMATH_CALUDE_goose_egg_problem_l75_7544

-- Define the total number of goose eggs laid
variable (E : ℕ)

-- Define the conditions
axiom hatch_ratio : (1 : ℚ) / 4 * E = (E / 4 : ℕ)
axiom first_month_survival : (4 : ℚ) / 5 * (E / 4 : ℕ) = (4 * E / 20 : ℕ)
axiom first_year_survival : (4 * E / 20 : ℕ) = 120

-- Define the theorem
theorem goose_egg_problem :
  E = 2400 ∧ ((4 * E / 20 : ℕ) - 120 : ℚ) / (4 * E / 20 : ℕ) = 3 / 4 := by
  sorry


end NUMINAMATH_CALUDE_goose_egg_problem_l75_7544


namespace NUMINAMATH_CALUDE_rectangular_box_width_is_correct_boxes_fit_in_wooden_box_l75_7541

/-- The width of rectangular boxes that fit in a wooden box -/
def rectangular_box_width : ℝ :=
  let wooden_box_length : ℝ := 800  -- 8 m in cm
  let wooden_box_width : ℝ := 700   -- 7 m in cm
  let wooden_box_height : ℝ := 600  -- 6 m in cm
  let box_length : ℝ := 8
  let box_height : ℝ := 6
  let max_boxes : ℕ := 1000000
  7  -- Width of rectangular boxes in cm

theorem rectangular_box_width_is_correct : rectangular_box_width = 7 := by
  sorry

/-- The volume of the wooden box in cubic centimeters -/
def wooden_box_volume : ℝ :=
  800 * 700 * 600

/-- The volume of a single rectangular box in cubic centimeters -/
def single_box_volume (w : ℝ) : ℝ :=
  8 * w * 6

/-- The total volume of all rectangular boxes -/
def total_boxes_volume (w : ℝ) : ℝ :=
  1000000 * single_box_volume w

theorem boxes_fit_in_wooden_box :
  total_boxes_volume rectangular_box_width ≤ wooden_box_volume := by
  sorry

end NUMINAMATH_CALUDE_rectangular_box_width_is_correct_boxes_fit_in_wooden_box_l75_7541


namespace NUMINAMATH_CALUDE_julia_tuesday_playmates_l75_7546

/-- The number of kids Julia played with on Monday -/
def monday_kids : ℕ := 6

/-- The difference in number of kids between Monday and Tuesday -/
def difference : ℕ := 1

/-- The number of kids Julia played with on Tuesday -/
def tuesday_kids : ℕ := monday_kids - difference

theorem julia_tuesday_playmates : tuesday_kids = 5 := by
  sorry

end NUMINAMATH_CALUDE_julia_tuesday_playmates_l75_7546


namespace NUMINAMATH_CALUDE_square_triangle_perimeter_ratio_l75_7571

/-- Given a square with side length s_s and an equilateral triangle with side length s_t,
    if their perimeters are equal, then the ratio of s_t to s_s is 4/3. -/
theorem square_triangle_perimeter_ratio (s_s s_t : ℝ) (h : s_s > 0) (h' : s_t > 0) :
  4 * s_s = 3 * s_t → s_t / s_s = 4 / 3 := by
  sorry


end NUMINAMATH_CALUDE_square_triangle_perimeter_ratio_l75_7571


namespace NUMINAMATH_CALUDE_ratio_problem_l75_7530

/-- Given that a:b = 4:3 and a:c = 4:15, prove that b:c = 1:5 -/
theorem ratio_problem (a b c : ℚ) 
  (hab : a / b = 4 / 3) 
  (hac : a / c = 4 / 15) : 
  b / c = 1 / 5 := by sorry

end NUMINAMATH_CALUDE_ratio_problem_l75_7530


namespace NUMINAMATH_CALUDE_min_nSn_l75_7555

/-- An arithmetic sequence with given properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- The sum function
  h1 : a 5 = 3  -- a_5 = 3
  h2 : S 10 = 40  -- S_10 = 40

/-- The property that the sequence is arithmetic -/
def isArithmetic (seq : ArithmeticSequence) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, seq.a (n + 1) = seq.a n + d

/-- The sum function definition -/
def sumProperty (seq : ArithmeticSequence) : Prop :=
  ∀ n : ℕ, seq.S n = (n : ℝ) * (seq.a 1 + seq.a n) / 2

/-- The main theorem -/
theorem min_nSn (seq : ArithmeticSequence) 
  (hArith : isArithmetic seq) (hSum : sumProperty seq) : 
  ∃ m : ℝ, m = -32 ∧ ∀ n : ℕ, (n : ℝ) * seq.S n ≥ m :=
sorry

end NUMINAMATH_CALUDE_min_nSn_l75_7555


namespace NUMINAMATH_CALUDE_knights_probability_l75_7540

/-- The number of knights seated at the round table -/
def total_knights : ℕ := 30

/-- The number of knights chosen randomly -/
def chosen_knights : ℕ := 4

/-- The probability that at least two of the chosen knights were sitting next to each other -/
def Q : ℚ := 4456 / 4701

theorem knights_probability :
  Q = 1 - (total_knights * (total_knights - 4) * (total_knights - 8) * (total_knights - 12)) /
      (total_knights * (total_knights - 1) * (total_knights - 2) * (total_knights - 3)) :=
sorry

end NUMINAMATH_CALUDE_knights_probability_l75_7540


namespace NUMINAMATH_CALUDE_parabola_vertex_on_x_axis_l75_7503

/-- A parabola with equation y = x^2 - 10x + d + 4 has its vertex on the x-axis if and only if d = 21 -/
theorem parabola_vertex_on_x_axis (d : ℝ) : 
  (∃ x : ℝ, x^2 - 10*x + d + 4 = 0 ∧ 
   ∀ y : ℝ, y^2 - 10*y + d + 4 ≥ x^2 - 10*x + d + 4) ↔ 
  d = 21 := by
sorry

end NUMINAMATH_CALUDE_parabola_vertex_on_x_axis_l75_7503


namespace NUMINAMATH_CALUDE_remainder_sum_l75_7505

theorem remainder_sum (c d : ℤ) 
  (hc : c % 100 = 78) 
  (hd : d % 150 = 123) : 
  (c + d) % 50 = 1 := by
sorry

end NUMINAMATH_CALUDE_remainder_sum_l75_7505


namespace NUMINAMATH_CALUDE_largest_digit_divisible_by_6_l75_7582

def is_divisible_by_6 (n : ℕ) : Prop := n % 6 = 0

def append_digit (n m : ℕ) : ℕ := n * 10 + m

theorem largest_digit_divisible_by_6 :
  ∃ (M : ℕ), M ≤ 9 ∧ 
    is_divisible_by_6 (append_digit 5172 M) ∧ 
    ∀ (K : ℕ), K ≤ 9 → is_divisible_by_6 (append_digit 5172 K) → K ≤ M :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_largest_digit_divisible_by_6_l75_7582


namespace NUMINAMATH_CALUDE_consecutive_integers_base_sum_l75_7553

/-- Given two consecutive positive integers X and Y, 
    if 241 in base X plus 52 in base Y equals 194 in base (X+Y), 
    then X + Y equals 15 -/
theorem consecutive_integers_base_sum (X Y : ℕ) : 
  X > 0 ∧ Y > 0 ∧ Y = X + 1 →
  (2 * X^2 + 4 * X + 1) + (5 * Y + 2) = ((X + Y)^2 + 9 * (X + Y) + 4) →
  X + Y = 15 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_base_sum_l75_7553


namespace NUMINAMATH_CALUDE_sum_place_values_equals_350077055735_l75_7559

def numeral : ℕ := 95378637153370261

def place_value (digit : ℕ) (position : ℕ) : ℕ := digit * (10 ^ position)

def sum_place_values : ℕ :=
  -- Two 3's
  place_value 3 11 + place_value 3 1 +
  -- Three 7's
  place_value 7 10 + place_value 7 6 + place_value 7 2 +
  -- Four 5's
  place_value 5 13 + place_value 5 4 + place_value 5 3 + place_value 5 0

theorem sum_place_values_equals_350077055735 :
  sum_place_values = 350077055735 := by sorry

end NUMINAMATH_CALUDE_sum_place_values_equals_350077055735_l75_7559


namespace NUMINAMATH_CALUDE_total_pupils_after_addition_l75_7538

/-- Given a school with an initial number of girls and boys, and additional girls joining,
    calculate the total number of pupils after the new girls joined. -/
theorem total_pupils_after_addition (initial_girls initial_boys additional_girls : ℕ) :
  initial_girls = 706 →
  initial_boys = 222 →
  additional_girls = 418 →
  initial_girls + initial_boys + additional_girls = 1346 := by
  sorry

end NUMINAMATH_CALUDE_total_pupils_after_addition_l75_7538


namespace NUMINAMATH_CALUDE_min_distance_complex_points_l75_7527

theorem min_distance_complex_points (z : ℂ) (h : Complex.abs (z + 2 - 2*I) = 1) :
  ∃ (min : ℝ), min = 3 ∧ ∀ w : ℂ, Complex.abs (w + 2 - 2*I) = 1 → Complex.abs (w - 2 - 2*I) ≥ min :=
sorry

end NUMINAMATH_CALUDE_min_distance_complex_points_l75_7527
