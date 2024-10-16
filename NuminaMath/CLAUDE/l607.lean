import Mathlib

namespace NUMINAMATH_CALUDE_min_value_product_l607_60742

theorem min_value_product (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h_prod : x * y * z = 1 / 2) :
  (x + y) * (2 * y + 3 * z) * (x * z + 2) ≥ 4 * Real.sqrt 6 ∧
  ∃ (x' y' z' : ℝ), x' > 0 ∧ y' > 0 ∧ z' > 0 ∧ x' * y' * z' = 1 / 2 ∧
    (x' + y') * (2 * y' + 3 * z') * (x' * z' + 2) = 4 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_min_value_product_l607_60742


namespace NUMINAMATH_CALUDE_no_divisibility_by_2013_l607_60748

theorem no_divisibility_by_2013 : ¬∃ (a b c : ℕ), 
  (a^2 + b^2 + c^2) % (2013 * (a*b + b*c + c*a)) = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_divisibility_by_2013_l607_60748


namespace NUMINAMATH_CALUDE_greenSpaceAfterThreeYears_l607_60755

/-- Calculates the green space after a given number of years with a fixed annual increase rate -/
def greenSpaceAfterYears (initialSpace : ℝ) (annualIncrease : ℝ) (years : ℕ) : ℝ :=
  initialSpace * (1 + annualIncrease) ^ years

/-- Theorem stating that the green space after 3 years with initial 1000 acres and 10% annual increase is 1331 acres -/
theorem greenSpaceAfterThreeYears :
  greenSpaceAfterYears 1000 0.1 3 = 1331 := by sorry

end NUMINAMATH_CALUDE_greenSpaceAfterThreeYears_l607_60755


namespace NUMINAMATH_CALUDE_two_players_goals_l607_60756

theorem two_players_goals (total_goals : ℕ) (player1_goals player2_goals : ℕ) : 
  total_goals = 300 →
  player1_goals = player2_goals →
  player1_goals + player2_goals = total_goals / 5 →
  player1_goals = 30 ∧ player2_goals = 30 := by
  sorry

end NUMINAMATH_CALUDE_two_players_goals_l607_60756


namespace NUMINAMATH_CALUDE_repeating_decimal_to_fraction_l607_60749

theorem repeating_decimal_to_fraction : 
  ∃ (x : ℚ), x = 3 + 45 / 99 ∧ x = 38 / 11 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_to_fraction_l607_60749


namespace NUMINAMATH_CALUDE_ethan_reading_pages_l607_60765

theorem ethan_reading_pages : 
  ∀ (total_pages saturday_morning sunday_pages pages_left saturday_night : ℕ),
  total_pages = 360 →
  saturday_morning = 40 →
  sunday_pages = 2 * (saturday_morning + saturday_night) →
  pages_left = 210 →
  total_pages = saturday_morning + saturday_night + sunday_pages + pages_left →
  saturday_night = 10 := by
sorry

end NUMINAMATH_CALUDE_ethan_reading_pages_l607_60765


namespace NUMINAMATH_CALUDE_nonzero_digits_after_decimal_l607_60774

theorem nonzero_digits_after_decimal (n : ℕ) (d : ℕ) (h : n = 60 ∧ d = 2^3 * 5^8) :
  (Nat.digits 10 (n * 10^7 / d)).length - 7 = 3 :=
sorry

end NUMINAMATH_CALUDE_nonzero_digits_after_decimal_l607_60774


namespace NUMINAMATH_CALUDE_buffet_price_theorem_l607_60788

/-- Represents the price of an adult buffet ticket -/
def adult_price : ℝ := 30

/-- Represents the price of a child buffet ticket -/
def child_price : ℝ := 15

/-- Represents the discount rate for senior citizens -/
def senior_discount : ℝ := 0.1

/-- Calculates the total cost for the family's buffet -/
def total_cost (adult_price : ℝ) : ℝ :=
  2 * adult_price +  -- Cost for 2 adults
  2 * (1 - senior_discount) * adult_price +  -- Cost for 2 senior citizens
  3 * child_price  -- Cost for 3 children

theorem buffet_price_theorem :
  total_cost adult_price = 159 :=
by sorry

end NUMINAMATH_CALUDE_buffet_price_theorem_l607_60788


namespace NUMINAMATH_CALUDE_smallest_positive_k_l607_60760

theorem smallest_positive_k (m n : ℕ+) (h : m ≤ 2000) : 
  let k := 3 - (m : ℚ) / n
  ∀ k' > 0, k ≥ k' → k' ≥ 1/667 :=
by sorry

end NUMINAMATH_CALUDE_smallest_positive_k_l607_60760


namespace NUMINAMATH_CALUDE_circle_C_properties_line_l_property_circle_E_fixed_points_l607_60776

-- Define the circle C
def circle_C (x y : ℝ) : Prop :=
  (x - 3)^2 + (y + 4)^2 = 25

-- Define the line l
def line_l (x y : ℝ) : Prop :=
  x - 2*y + 4 = 0

-- Define the circle E
def circle_E (x y y1 y2 : ℝ) : Prop :=
  x^2 + y^2 - 12*x - (y1 + y2)*y - 64 = 0

theorem circle_C_properties :
  (circle_C 0 0) ∧ 
  (circle_C 6 0) ∧ 
  (∃ x : ℝ, circle_C x 1) :=
sorry

theorem line_l_property (a b : ℝ) :
  line_l a b ↔ 
  ∃ t : ℝ, 
    (t - 3)^2 + (b + 4)^2 = 25 ∧
    ((a - t)^2 + (b - 1)^2) = ((a - 2)^2 + (b + 2)^2) :=
sorry

theorem circle_E_fixed_points (y1 y2 : ℝ) :
  (y1 * y2 = -100) →
  (circle_E 16 0 y1 y2) ∧
  (circle_E (-4) 0 y1 y2) :=
sorry

end NUMINAMATH_CALUDE_circle_C_properties_line_l_property_circle_E_fixed_points_l607_60776


namespace NUMINAMATH_CALUDE_distorted_polygon_sides_l607_60794

/-- A regular polygon with a distorted exterior angle -/
structure DistortedPolygon where
  -- The apparent exterior angle in degrees
  apparent_angle : ℝ
  -- The distortion factor
  distortion_factor : ℝ
  -- The number of sides
  sides : ℕ

/-- The theorem stating the number of sides for the given conditions -/
theorem distorted_polygon_sides (p : DistortedPolygon) 
  (h1 : p.apparent_angle = 18)
  (h2 : p.distortion_factor = 1.5)
  (h3 : p.apparent_angle * p.sides = 360 * p.distortion_factor) : 
  p.sides = 30 := by
  sorry

end NUMINAMATH_CALUDE_distorted_polygon_sides_l607_60794


namespace NUMINAMATH_CALUDE_at_least_two_equations_have_solutions_l607_60778

-- Define the functions
def f₁ (a b c x : ℝ) : ℝ := (x - b) * (x - c) - (x - a)
def f₂ (a b c x : ℝ) : ℝ := (x - c) * (x - a) - (x - b)
def f₃ (a b c x : ℝ) : ℝ := (x - a) * (x - b) - (x - c)

-- Define the theorem
theorem at_least_two_equations_have_solutions (a b c : ℝ) (h : a ≠ b ∧ b ≠ c ∧ c ≠ a) :
  (∃ x : ℝ, f₁ a b c x = 0) ∧ (∃ x : ℝ, f₂ a b c x = 0) ∨
  (∃ x : ℝ, f₁ a b c x = 0) ∧ (∃ x : ℝ, f₃ a b c x = 0) ∨
  (∃ x : ℝ, f₂ a b c x = 0) ∧ (∃ x : ℝ, f₃ a b c x = 0) :=
sorry

end NUMINAMATH_CALUDE_at_least_two_equations_have_solutions_l607_60778


namespace NUMINAMATH_CALUDE_function_composition_ratio_l607_60754

def f (x : ℝ) : ℝ := 3 * x + 4

def g (x : ℝ) : ℝ := 2 * x - 1

theorem function_composition_ratio :
  f (g (f 3)) / g (f (g 3)) = 79 / 37 := by
  sorry

end NUMINAMATH_CALUDE_function_composition_ratio_l607_60754


namespace NUMINAMATH_CALUDE_sum_of_coefficients_expansion_l607_60751

theorem sum_of_coefficients_expansion (x y : ℝ) : 
  (fun x y => (x + 2*y - 1)^6) 1 1 = 64 := by sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_expansion_l607_60751


namespace NUMINAMATH_CALUDE_distance_between_homes_l607_60753

/-- The distance between Xiaohong's and Xiaoli's homes given their walking speeds and arrival times -/
theorem distance_between_homes 
  (x_speed : ℝ) 
  (l_speed_to_cinema : ℝ) 
  (l_speed_from_cinema : ℝ) 
  (delay : ℝ) 
  (h_x_speed : x_speed = 52) 
  (h_l_speed_to_cinema : l_speed_to_cinema = 70) 
  (h_l_speed_from_cinema : l_speed_from_cinema = 90) 
  (h_delay : delay = 4) : 
  ∃ (t : ℝ), x_speed * t + l_speed_to_cinema * t = 2196 ∧ 
  x_speed * (t + delay + (x_speed * t / x_speed)) = l_speed_from_cinema * ((x_speed * t / x_speed) - delay) :=
sorry

end NUMINAMATH_CALUDE_distance_between_homes_l607_60753


namespace NUMINAMATH_CALUDE_regular_polygon_interior_angle_sum_l607_60716

theorem regular_polygon_interior_angle_sum 
  (n : ℕ) 
  (h_exterior : (360 : ℝ) / n = 45) : 
  (n - 2) * 180 = 1080 :=
by sorry

end NUMINAMATH_CALUDE_regular_polygon_interior_angle_sum_l607_60716


namespace NUMINAMATH_CALUDE_arithmetic_equality_l607_60713

theorem arithmetic_equality : 5 - 4 * 3 / 2 + 1 = 0 := by sorry

end NUMINAMATH_CALUDE_arithmetic_equality_l607_60713


namespace NUMINAMATH_CALUDE_friday_dressing_time_l607_60722

/-- Represents the dressing times for each day of the week -/
structure DressingTimes where
  monday : ℕ
  tuesday : ℕ
  wednesday : ℕ
  thursday : ℕ
  friday : ℕ

/-- Calculates the average dressing time for the week -/
def weeklyAverage (times : DressingTimes) : ℚ :=
  (times.monday + times.tuesday + times.wednesday + times.thursday + times.friday) / 5

/-- The old average dressing time -/
def oldAverage : ℚ := 3

/-- The given dressing times for Monday through Thursday -/
def givenTimes : DressingTimes := {
  monday := 2
  tuesday := 4
  wednesday := 3
  thursday := 4
  friday := 0  -- We'll solve for this
}

theorem friday_dressing_time :
  ∃ (fridayTime : ℕ),
    let newTimes := { givenTimes with friday := fridayTime }
    weeklyAverage newTimes = oldAverage ∧ fridayTime = 2 := by
  sorry


end NUMINAMATH_CALUDE_friday_dressing_time_l607_60722


namespace NUMINAMATH_CALUDE_function_properties_l607_60775

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x + a * x - a

theorem function_properties (a : ℝ) (h : a ≠ 0) :
  (f a 0 = 2 → a = -1) ∧
  (a = -1 → 
    (∀ x y : ℝ, x < y → x < 0 → f a x > f a y) ∧
    (∀ x y : ℝ, x < y → 0 < x → f a x < f a y) ∧
    (∀ x : ℝ, f a x ≥ 2) ∧
    (f a 0 = 2)) ∧
  ((∀ x : ℝ, f a x ≠ 0) → -Real.exp 2 < a ∧ a < 0) :=
by sorry

#check function_properties

end NUMINAMATH_CALUDE_function_properties_l607_60775


namespace NUMINAMATH_CALUDE_distinct_values_theorem_l607_60736

/-- The number of distinct values expressible as ip + jq -/
def distinct_values (n p q : ℕ) : ℕ :=
  if p = q ∧ p = 1 then
    n + 1
  else if p > q ∧ n < p then
    (n + 1) * (n + 2) / 2
  else if p > q ∧ n ≥ p then
    p * (2 * n - p + 3) / 2
  else
    0  -- This case is not specified in the problem, but needed for completeness

/-- Theorem stating the number of distinct values expressible as ip + jq -/
theorem distinct_values_theorem (n p q : ℕ) (h_coprime : Nat.Coprime p q) :
  distinct_values n p q =
    if p = q ∧ p = 1 then
      n + 1
    else if p > q ∧ n < p then
      (n + 1) * (n + 2) / 2
    else if p > q ∧ n ≥ p then
      p * (2 * n - p + 3) / 2
    else
      0 := by sorry

end NUMINAMATH_CALUDE_distinct_values_theorem_l607_60736


namespace NUMINAMATH_CALUDE_restaurant_gratuity_calculation_l607_60708

def calculate_gratuity (base_price : ℝ) (tax_rate : ℝ) (discount_rate : ℝ) (gratuity_rate : ℝ) : ℝ :=
  let discounted_price := base_price * (1 - discount_rate)
  let price_after_tax := discounted_price * (1 + tax_rate)
  price_after_tax * gratuity_rate

theorem restaurant_gratuity_calculation :
  let striploin_gratuity := calculate_gratuity 80 0.10 0.05 0.15
  let wine_gratuity := calculate_gratuity 10 0.15 0 0.20
  let dessert_gratuity := calculate_gratuity 12 0.05 0.10 0.10
  let water_gratuity := calculate_gratuity 3 0 0 0.05
  striploin_gratuity + wine_gratuity + dessert_gratuity + water_gratuity = 16.12 := by
  sorry

end NUMINAMATH_CALUDE_restaurant_gratuity_calculation_l607_60708


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l607_60740

-- Define the sets M and N
def M : Set ℝ := {x | -2 ≤ x ∧ x < 2}
def N : Set ℝ := {x | x ≥ -2}

-- The theorem to prove
theorem intersection_of_M_and_N :
  M ∩ N = {x | -2 ≤ x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l607_60740


namespace NUMINAMATH_CALUDE_trail_mix_weight_l607_60707

/-- The weight of peanuts in pounds -/
def peanuts : ℝ := 0.17

/-- The weight of chocolate chips in pounds -/
def chocolate_chips : ℝ := 0.17

/-- The weight of raisins in pounds -/
def raisins : ℝ := 0.08

/-- The weight of dried apricots in pounds -/
def dried_apricots : ℝ := 0.12

/-- The weight of sunflower seeds in pounds -/
def sunflower_seeds : ℝ := 0.09

/-- The weight of coconut flakes in pounds -/
def coconut_flakes : ℝ := 0.15

/-- The total weight of trail mix in pounds -/
def total_weight : ℝ := peanuts + chocolate_chips + raisins + dried_apricots + sunflower_seeds + coconut_flakes

theorem trail_mix_weight : total_weight = 0.78 := by
  sorry

end NUMINAMATH_CALUDE_trail_mix_weight_l607_60707


namespace NUMINAMATH_CALUDE_files_deleted_l607_60709

theorem files_deleted (initial_apps : ℕ) (initial_files : ℕ) (final_apps : ℕ) (final_files : ℕ) 
  (h1 : initial_apps = 17)
  (h2 : initial_files = 21)
  (h3 : final_apps = 3)
  (h4 : final_files = 7) :
  initial_files - final_files = 14 := by
  sorry

end NUMINAMATH_CALUDE_files_deleted_l607_60709


namespace NUMINAMATH_CALUDE_dog_grouping_combinations_l607_60750

def number_of_dogs : ℕ := 12
def group_sizes : List ℕ := [4, 5, 3]

def rocky_group : ℕ := 2
def nipper_group : ℕ := 1
def scruffy_group : ℕ := 0

def remaining_dogs : ℕ := number_of_dogs - 3

theorem dog_grouping_combinations : 
  (remaining_dogs.choose (group_sizes[rocky_group] - 1)) * 
  ((remaining_dogs - (group_sizes[rocky_group] - 1)).choose (group_sizes[scruffy_group] - 1)) = 1260 := by
  sorry

end NUMINAMATH_CALUDE_dog_grouping_combinations_l607_60750


namespace NUMINAMATH_CALUDE_i_power_difference_zero_l607_60705

-- Define the imaginary unit i
def i : ℂ := Complex.I

-- State the theorem
theorem i_power_difference_zero : i^45 - i^305 = 0 := by
  sorry

end NUMINAMATH_CALUDE_i_power_difference_zero_l607_60705


namespace NUMINAMATH_CALUDE_max_value_polynomial_l607_60789

theorem max_value_polynomial (a b : ℝ) (h : a + b = 4) :
  (∃ x y : ℝ, x + y = 4 ∧ 
    a^4*b + a^3*b + a^2*b + a*b + a*b^2 + a*b^3 + a*b^4 ≤ 
    x^4*y + x^3*y + x^2*y + x*y + x*y^2 + x*y^3 + x*y^4) ∧
  (∀ x y : ℝ, x + y = 4 → 
    x^4*y + x^3*y + x^2*y + x*y + x*y^2 + x*y^3 + x*y^4 ≤ 7225/56) ∧
  (∃ x y : ℝ, x + y = 4 ∧ 
    x^4*y + x^3*y + x^2*y + x*y + x*y^2 + x*y^3 + x*y^4 = 7225/56) :=
by sorry

end NUMINAMATH_CALUDE_max_value_polynomial_l607_60789


namespace NUMINAMATH_CALUDE_arithmetic_progression_of_primes_l607_60715

theorem arithmetic_progression_of_primes (a : ℕ → ℕ) (d : ℕ) :
  (∀ i ∈ Finset.range 15, Nat.Prime (a i)) →
  (∀ i ∈ Finset.range 14, a (i + 1) = a i + d) →
  d > 0 →
  a 0 > 15 →
  d > 30000 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_progression_of_primes_l607_60715


namespace NUMINAMATH_CALUDE_first_digits_of_powers_of_five_appear_in_powers_of_two_l607_60739

-- Define a function to get the first digit of a natural number
def firstDigit (n : ℕ) : ℕ :=
  if n < 10 then n else firstDigit (n / 10)

-- Define the sequence of first digits of powers of five
def firstDigitsOfPowersOfFive : ℕ → ℕ := λ n => firstDigit (5^n)

-- Define the sequence of first digits of powers of two
def firstDigitsOfPowersOfTwo : ℕ → ℕ := λ n => firstDigit (2^n)

-- Define a function to check if a list is a reverse subsequence of another list
def isReverseSubsequenceOf (subseq : List ℕ) (seq : List ℕ) : Prop :=
  ∃ (start : ℕ), List.takeWhile (λ i => i < start + subseq.length) (List.drop start seq) = subseq.reverse

-- The main theorem
theorem first_digits_of_powers_of_five_appear_in_powers_of_two :
  ∀ (n : ℕ), ∃ (m : ℕ),
    isReverseSubsequenceOf
      (List.map firstDigitsOfPowersOfFive (List.range n))
      (List.map firstDigitsOfPowersOfTwo (List.range m)) :=
sorry

end NUMINAMATH_CALUDE_first_digits_of_powers_of_five_appear_in_powers_of_two_l607_60739


namespace NUMINAMATH_CALUDE_sum_of_ages_l607_60798

/-- Represents the ages of Xavier and Yasmin -/
structure Ages where
  xavier : ℕ
  yasmin : ℕ

/-- The current ages of Xavier and Yasmin satisfy the given conditions -/
def satisfies_conditions (ages : Ages) : Prop :=
  ages.xavier = 2 * ages.yasmin ∧ ages.xavier + 6 = 30

/-- Theorem: The sum of Xavier's and Yasmin's current ages is 36 -/
theorem sum_of_ages (ages : Ages) (h : satisfies_conditions ages) : 
  ages.xavier + ages.yasmin = 36 := by
  sorry


end NUMINAMATH_CALUDE_sum_of_ages_l607_60798


namespace NUMINAMATH_CALUDE_square_sum_reciprocal_l607_60746

theorem square_sum_reciprocal (x : ℝ) (h : x + (1 / x) = 5) : x^2 + (1 / x)^2 = 23 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_reciprocal_l607_60746


namespace NUMINAMATH_CALUDE_field_trip_attendance_is_76_l607_60752

/-- The number of people on a field trip given the number of vans and buses,
    and the number of people in each vehicle type. -/
def field_trip_attendance (num_vans num_buses : ℕ) (people_per_van people_per_bus : ℕ) : ℕ :=
  num_vans * people_per_van + num_buses * people_per_bus

/-- Theorem stating that the total number of people on the field trip is 76. -/
theorem field_trip_attendance_is_76 :
  field_trip_attendance 2 3 8 20 = 76 := by
  sorry

#eval field_trip_attendance 2 3 8 20

end NUMINAMATH_CALUDE_field_trip_attendance_is_76_l607_60752


namespace NUMINAMATH_CALUDE_subset_implies_a_equals_one_l607_60782

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {0, -a}
def B (a : ℝ) : Set ℝ := {1, a-2, 2*a-2}

-- State the theorem
theorem subset_implies_a_equals_one (a : ℝ) :
  A a ⊆ B a → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_subset_implies_a_equals_one_l607_60782


namespace NUMINAMATH_CALUDE_shorter_diagonal_is_25_l607_60747

/-- Represents a trapezoid EFGH -/
structure Trapezoid where
  ef : ℝ  -- length of side EF
  gh : ℝ  -- length of side GH
  eg : ℝ  -- length of side EG
  fh : ℝ  -- length of side FH
  ef_parallel_gh : ef > gh  -- EF is parallel to GH and longer
  e_acute : True  -- angle E is acute
  f_acute : True  -- angle F is acute

/-- The length of the shorter diagonal of the trapezoid -/
def shorter_diagonal (t : Trapezoid) : ℝ := sorry

/-- Theorem stating that for a trapezoid with given side lengths, the shorter diagonal is 25 -/
theorem shorter_diagonal_is_25 (t : Trapezoid) 
  (h1 : t.ef = 39) 
  (h2 : t.gh = 27) 
  (h3 : t.eg = 13) 
  (h4 : t.fh = 15) : 
  shorter_diagonal t = 25 := by sorry

end NUMINAMATH_CALUDE_shorter_diagonal_is_25_l607_60747


namespace NUMINAMATH_CALUDE_water_added_proof_l607_60704

def container_problem (capacity : ℝ) (initial_fill : ℝ) (final_fill : ℝ) : Prop :=
  let initial_volume := capacity * initial_fill
  let final_volume := capacity * final_fill
  final_volume - initial_volume = 20

theorem water_added_proof :
  container_problem 80 0.5 0.75 :=
sorry

end NUMINAMATH_CALUDE_water_added_proof_l607_60704


namespace NUMINAMATH_CALUDE_range_of_m_l607_60729

/-- Represents the condition for proposition p -/
def is_hyperbola_y_axis (m : ℝ) : Prop :=
  (2 - m < 0) ∧ (m - 1 > 0)

/-- Represents the condition for proposition q -/
def has_no_real_roots (m : ℝ) : Prop :=
  16 * (m - 2)^2 - 16 < 0

/-- The main theorem stating the range of m -/
theorem range_of_m (m : ℝ) 
  (h_p_or_q : is_hyperbola_y_axis m ∨ has_no_real_roots m)
  (h_not_q : ¬has_no_real_roots m) :
  m ≥ 3 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l607_60729


namespace NUMINAMATH_CALUDE_broccoli_area_l607_60799

/-- Represents the garden and broccoli production --/
structure BroccoliGarden where
  last_year_side : ℝ
  this_year_side : ℝ
  broccoli_increase : ℕ
  this_year_count : ℕ

/-- The conditions of the broccoli garden problem --/
def broccoli_conditions (g : BroccoliGarden) : Prop :=
  g.broccoli_increase = 79 ∧
  g.this_year_count = 1600 ∧
  g.this_year_side ^ 2 = g.this_year_count ∧
  g.this_year_side ^ 2 - g.last_year_side ^ 2 = g.broccoli_increase

/-- The theorem stating that each broccoli takes 1 square foot --/
theorem broccoli_area (g : BroccoliGarden) 
  (h : broccoli_conditions g) : 
  g.this_year_side ^ 2 / g.this_year_count = 1 := by
  sorry


end NUMINAMATH_CALUDE_broccoli_area_l607_60799


namespace NUMINAMATH_CALUDE_g_of_3_equals_64_l607_60711

/-- The function g satisfies 4g(x) - 3g(1/x) = x^2 for all nonzero x -/
def g_property (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, x ≠ 0 → 4 * g x - 3 * g (1 / x) = x^2

/-- Given g satisfying the property, prove that g(3) = 64 -/
theorem g_of_3_equals_64 (g : ℝ → ℝ) (h : g_property g) : g 3 = 64 := by
  sorry

end NUMINAMATH_CALUDE_g_of_3_equals_64_l607_60711


namespace NUMINAMATH_CALUDE_pascal_triangle_p_row_zeros_l607_60721

theorem pascal_triangle_p_row_zeros (p : ℕ) (k : ℕ) (hp : Nat.Prime p) (hk : 1 ≤ k ∧ k ≤ p - 1) : 
  Nat.choose p k ≡ 0 [MOD p] := by
  sorry

end NUMINAMATH_CALUDE_pascal_triangle_p_row_zeros_l607_60721


namespace NUMINAMATH_CALUDE_fruit_ratio_l607_60700

theorem fruit_ratio (apples bananas total pears : ℕ) : 
  apples = 4 →
  bananas = 5 →
  total = 21 →
  total = apples + pears + bananas →
  ∃ (n : ℕ), pears = n * apples →
  pears / apples = 3 := by
sorry

end NUMINAMATH_CALUDE_fruit_ratio_l607_60700


namespace NUMINAMATH_CALUDE_nonagon_diagonals_l607_60717

/-- The number of distinct diagonals in a convex nonagon -/
def num_diagonals_nonagon : ℕ := 27

/-- A convex polygon with 9 sides -/
structure ConvexNonagon where
  sides : ℕ
  is_convex : Bool
  side_count_eq_9 : sides = 9

/-- Theorem: The number of distinct diagonals in a convex nonagon is 27 -/
theorem nonagon_diagonals (n : ConvexNonagon) : num_diagonals_nonagon = 27 := by
  sorry

end NUMINAMATH_CALUDE_nonagon_diagonals_l607_60717


namespace NUMINAMATH_CALUDE_georges_required_speed_l607_60730

/-- George's usual walking distance to school in miles -/
def usual_distance : ℝ := 1.5

/-- George's usual walking speed in miles per hour -/
def usual_speed : ℝ := 4

/-- Distance George walks at a slower pace today in miles -/
def slow_distance : ℝ := 1

/-- George's slower walking speed today in miles per hour -/
def slow_speed : ℝ := 3

/-- Remaining distance George needs to run in miles -/
def remaining_distance : ℝ := 0.5

/-- Theorem stating the speed George needs to run to arrive on time -/
theorem georges_required_speed : 
  ∃ (required_speed : ℝ),
    (usual_distance / usual_speed = slow_distance / slow_speed + remaining_distance / required_speed) ∧
    required_speed = 12 := by
  sorry

end NUMINAMATH_CALUDE_georges_required_speed_l607_60730


namespace NUMINAMATH_CALUDE_knife_sharpening_problem_l607_60764

/-- Represents the pricing structure for knife sharpening -/
structure KnifeSharpeningPrices where
  first_knife : ℕ → ℕ
  next_three_knives : ℕ → ℕ
  remaining_knives : ℕ → ℕ

/-- Calculates the total cost of sharpening knives -/
def total_cost (prices : KnifeSharpeningPrices) (num_knives : ℕ) : ℕ :=
  prices.first_knife 1 +
  prices.next_three_knives (min 3 (num_knives - 1)) +
  prices.remaining_knives (max 0 (num_knives - 4))

/-- Theorem stating that given the pricing structure and total cost, the number of knives is 9 -/
theorem knife_sharpening_problem (prices : KnifeSharpeningPrices) 
  (h1 : prices.first_knife 1 = 5)
  (h2 : ∀ n : ℕ, n ≤ 3 → prices.next_three_knives n = 4 * n)
  (h3 : ∀ n : ℕ, prices.remaining_knives n = 3 * n)
  (h4 : total_cost prices 9 = 32) :
  ∃ (n : ℕ), total_cost prices n = 32 ∧ n = 9 := by
  sorry


end NUMINAMATH_CALUDE_knife_sharpening_problem_l607_60764


namespace NUMINAMATH_CALUDE_penalty_kicks_count_l607_60781

theorem penalty_kicks_count (total_players : ℕ) (goalies : ℕ) : 
  total_players = 25 → goalies = 4 → total_players * goalies - goalies^2 = 96 :=
by
  sorry

end NUMINAMATH_CALUDE_penalty_kicks_count_l607_60781


namespace NUMINAMATH_CALUDE_cone_max_cross_section_area_l607_60734

/-- Given a cone with lateral surface formed by a sector of radius 1 and central angle 3/2 π,
    the maximum area of a cross-section passing through the vertex is 1/2. -/
theorem cone_max_cross_section_area (r : ℝ) (θ : ℝ) (h : ℝ) : 
  r = 1 → θ = 3/2 * Real.pi → h = Real.sqrt (r^2 - (r * θ / (2 * Real.pi))^2) → 
  (1/2 : ℝ) * (r * θ / (2 * Real.pi)) * h ≤ 1/2 := by
  sorry

#check cone_max_cross_section_area

end NUMINAMATH_CALUDE_cone_max_cross_section_area_l607_60734


namespace NUMINAMATH_CALUDE_stock_price_is_102_l607_60770

/-- Given an income, dividend rate, and investment amount, calculate the price of a stock. -/
def stock_price (income : ℚ) (dividend_rate : ℚ) (investment : ℚ) : ℚ :=
  let face_value := income / dividend_rate
  (investment / face_value) * 100

/-- Theorem stating that given the specific conditions, the stock price is 102. -/
theorem stock_price_is_102 :
  stock_price 900 (20 / 100) 4590 = 102 := by
  sorry

#eval stock_price 900 (20 / 100) 4590

end NUMINAMATH_CALUDE_stock_price_is_102_l607_60770


namespace NUMINAMATH_CALUDE_total_covid_cases_l607_60710

/-- Theorem: Total COVID-19 cases in New York, California, and Texas --/
theorem total_covid_cases (new_york california texas : ℕ) : 
  new_york = 2000 →
  california = new_york / 2 →
  california = texas + 400 →
  new_york + california + texas = 3600 := by
  sorry

end NUMINAMATH_CALUDE_total_covid_cases_l607_60710


namespace NUMINAMATH_CALUDE_equality_proof_l607_60737

theorem equality_proof (x y : ℤ) : 
  (x - 1) * (x + 4) * (x - 3) - (x + 1) * (x - 4) * (x + 3) = 
  (y - 1) * (y + 4) * (y - 3) - (y + 1) * (y - 4) * (y + 3) := by
  sorry

end NUMINAMATH_CALUDE_equality_proof_l607_60737


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_property_l607_60792

/-- An arithmetic sequence with non-zero terms -/
def arithmetic_sequence (a : ℕ+ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ+, a (n + 1) = a n + d

/-- A geometric sequence -/
def geometric_sequence (b : ℕ+ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ+, b (n + 1) = b n * r

theorem arithmetic_geometric_sequence_property
  (a b : ℕ+ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_geo : geometric_sequence b)
  (h_nonzero : ∀ n : ℕ+, a n ≠ 0)
  (h_eq : 2 * (a 3) - (a 7)^2 + 2 * (a 11) = 0)
  (h_b7 : b 7 = a 7) :
  b 6 * b 8 = 16 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_property_l607_60792


namespace NUMINAMATH_CALUDE_real_roots_condition_l607_60769

theorem real_roots_condition (a : ℝ) :
  (∃ x : ℝ, (a - 1) * x^2 - 2 * x + 1 = 0) ↔ a ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_real_roots_condition_l607_60769


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l607_60744

theorem quadratic_inequality_solution_set (m : ℝ) :
  (∀ x : ℝ, m * x^2 - (1 - m) * x + m ≥ 0) ↔ m ≥ 1/3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l607_60744


namespace NUMINAMATH_CALUDE_fireworks_per_letter_l607_60728

/-- The number of fireworks needed to display a digit --/
def fireworks_per_digit : ℕ := 6

/-- The number of digits in the year display --/
def year_digits : ℕ := 4

/-- The number of letters in "HAPPY NEW YEAR" --/
def phrase_letters : ℕ := 12

/-- The number of additional boxes of fireworks --/
def additional_boxes : ℕ := 50

/-- The number of fireworks in each box --/
def fireworks_per_box : ℕ := 8

/-- The total number of fireworks lit during the display --/
def total_fireworks : ℕ := 484

/-- Theorem: The number of fireworks needed to display a letter is 5 --/
theorem fireworks_per_letter :
  ∃ (x : ℕ), 
    x * phrase_letters + 
    fireworks_per_digit * year_digits + 
    additional_boxes * fireworks_per_box = 
    total_fireworks ∧ x = 5 := by
  sorry

end NUMINAMATH_CALUDE_fireworks_per_letter_l607_60728


namespace NUMINAMATH_CALUDE_simple_interest_calculation_l607_60741

/-- Calculate simple interest -/
def simple_interest (principal rate time : ℚ) : ℚ :=
  principal * rate * time / 100

/-- Proof of simple interest calculation -/
theorem simple_interest_calculation :
  let principal : ℚ := 15000
  let rate : ℚ := 6
  let time : ℚ := 3
  simple_interest principal rate time = 2700 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_calculation_l607_60741


namespace NUMINAMATH_CALUDE_opposite_expressions_l607_60763

theorem opposite_expressions (x : ℝ) : (4 * x - 8 = -(3 * x - 6)) ↔ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_opposite_expressions_l607_60763


namespace NUMINAMATH_CALUDE_sqrt_1936_div_11_l607_60718

theorem sqrt_1936_div_11 : Real.sqrt 1936 / 11 = 4 := by sorry

end NUMINAMATH_CALUDE_sqrt_1936_div_11_l607_60718


namespace NUMINAMATH_CALUDE_regression_analysis_conclusions_l607_60768

-- Define the regression model
structure RegressionModel where
  R_squared : ℝ
  sum_of_squares_residuals : ℝ
  residual_plot : Set (ℝ × ℝ)

-- Define the concept of model fit
def better_fit (model1 model2 : RegressionModel) : Prop := sorry

-- Define the concept of evenly scattered residuals
def evenly_scattered_residuals (plot : Set (ℝ × ℝ)) : Prop := sorry

-- Define the concept of horizontal band
def horizontal_band (plot : Set (ℝ × ℝ)) : Prop := sorry

-- Theorem stating the correct conclusions
theorem regression_analysis_conclusions 
  (model1 model2 : RegressionModel) (ε : ℝ) (hε : ε > 0) :
  -- Higher R² indicates better fit
  (model1.R_squared > model2.R_squared + ε → better_fit model1 model2) ∧ 
  -- Smaller sum of squares of residuals indicates better fit
  (model1.sum_of_squares_residuals < model2.sum_of_squares_residuals - ε → 
    better_fit model1 model2) ∧
  -- Evenly scattered residuals around a horizontal band indicate appropriate model
  (evenly_scattered_residuals model1.residual_plot ∧ 
   horizontal_band model1.residual_plot → 
   better_fit model1 model2) := by sorry


end NUMINAMATH_CALUDE_regression_analysis_conclusions_l607_60768


namespace NUMINAMATH_CALUDE_expansion_equality_l607_60759

theorem expansion_equality (x : ℝ) : (x - 2) * (x + 2) * (x^2 + 4) = x^4 - 16 := by
  sorry

end NUMINAMATH_CALUDE_expansion_equality_l607_60759


namespace NUMINAMATH_CALUDE_total_amount_calculation_l607_60766

theorem total_amount_calculation (x y z : ℝ) : 
  x > 0 → 
  y = 0.45 * x → 
  z = 0.30 * x → 
  y = 36 → 
  x + y + z = 140 := by
  sorry

end NUMINAMATH_CALUDE_total_amount_calculation_l607_60766


namespace NUMINAMATH_CALUDE_sallys_peaches_l607_60731

/-- Given that Sally had 13 peaches initially and ended up with 55 peaches,
    prove that she picked 42 peaches. -/
theorem sallys_peaches (initial : ℕ) (final : ℕ) (h1 : initial = 13) (h2 : final = 55) :
  final - initial = 42 := by sorry

end NUMINAMATH_CALUDE_sallys_peaches_l607_60731


namespace NUMINAMATH_CALUDE_fantasia_license_plates_l607_60714

/-- The number of letters in the alphabet used for license plates. -/
def alphabet_size : ℕ := 26

/-- The number of digits used for license plates. -/
def digit_size : ℕ := 10

/-- The number of letter positions in a license plate. -/
def letter_positions : ℕ := 3

/-- The number of digit positions in a license plate. -/
def digit_positions : ℕ := 4

/-- The total number of possible valid license plates in Fantasia. -/
def total_license_plates : ℕ := alphabet_size ^ letter_positions * digit_size ^ digit_positions

theorem fantasia_license_plates :
  total_license_plates = 175760000 := by
  sorry

end NUMINAMATH_CALUDE_fantasia_license_plates_l607_60714


namespace NUMINAMATH_CALUDE_pot_temperature_celsius_l607_60745

/-- Converts temperature from Fahrenheit to Celsius -/
def fahrenheit_to_celsius (f : ℚ) : ℚ :=
  (f - 32) * (5/9)

/-- The temperature of the pot of water in Fahrenheit -/
def pot_temperature_f : ℚ := 122

theorem pot_temperature_celsius :
  fahrenheit_to_celsius pot_temperature_f = 50 := by
  sorry

end NUMINAMATH_CALUDE_pot_temperature_celsius_l607_60745


namespace NUMINAMATH_CALUDE_f_difference_l607_60790

/-- Sum of all positive divisors of n -/
def sigma (n : ℕ+) : ℕ := sorry

/-- Function f as defined in the problem -/
def f (n : ℕ+) : ℚ := (sigma n + n) / n

/-- Theorem stating the result of f(540) - f(180) -/
theorem f_difference : f 540 - f 180 = 7 / 90 := by sorry

end NUMINAMATH_CALUDE_f_difference_l607_60790


namespace NUMINAMATH_CALUDE_shopping_remaining_amount_l607_60725

theorem shopping_remaining_amount (initial_amount : ℝ) (spent_percentage : ℝ) 
  (h1 : initial_amount = 5000)
  (h2 : spent_percentage = 0.30) : 
  initial_amount - (spent_percentage * initial_amount) = 3500 := by
  sorry

end NUMINAMATH_CALUDE_shopping_remaining_amount_l607_60725


namespace NUMINAMATH_CALUDE_money_left_after_purchase_l607_60779

def birthday_money (grandmother aunt uncle cousin brother : ℕ) : ℕ :=
  grandmother + aunt + uncle + cousin + brother

def total_in_wallet : ℕ := 185

def game_costs (game1 game2 game3 game4 game5 : ℕ) : ℕ :=
  game1 + game1 + game2 + game3 + game4 + game5

theorem money_left_after_purchase 
  (grandmother aunt uncle cousin brother : ℕ)
  (game1 game2 game3 game4 game5 : ℕ)
  (h1 : grandmother = 30)
  (h2 : aunt = 35)
  (h3 : uncle = 40)
  (h4 : cousin = 25)
  (h5 : brother = 20)
  (h6 : game1 = 30)
  (h7 : game2 = 40)
  (h8 : game3 = 35)
  (h9 : game4 = 25)
  (h10 : game5 = 0)  -- We use 0 for the fifth game as it's already counted in game1
  : total_in_wallet - (birthday_money grandmother aunt uncle cousin brother + game_costs game1 game2 game3 game4 game5) = 25 := by
  sorry

end NUMINAMATH_CALUDE_money_left_after_purchase_l607_60779


namespace NUMINAMATH_CALUDE_overlap_squares_area_l607_60723

/-- Given two identical squares with side length 12 that overlap to form a 12 by 20 rectangle,
    the area of the non-overlapping region of one square is 48. -/
theorem overlap_squares_area (square_side : ℝ) (rect_length : ℝ) (rect_width : ℝ) : 
  square_side = 12 →
  rect_length = 20 →
  rect_width = 12 →
  (2 * square_side^2) - (rect_length * rect_width) = 48 := by
  sorry

end NUMINAMATH_CALUDE_overlap_squares_area_l607_60723


namespace NUMINAMATH_CALUDE_forum_member_count_l607_60724

/-- The number of members in an online forum. -/
def forum_members : ℕ := 200

/-- The average number of questions posted per hour by each member. -/
def questions_per_hour : ℕ := 3

/-- The ratio of answers to questions posted by each member. -/
def answer_to_question_ratio : ℕ := 3

/-- The total number of posts (questions and answers) in a day. -/
def total_daily_posts : ℕ := 57600

/-- The number of hours in a day. -/
def hours_per_day : ℕ := 24

theorem forum_member_count :
  forum_members * (questions_per_hour * hours_per_day * (1 + answer_to_question_ratio)) = total_daily_posts :=
by sorry

end NUMINAMATH_CALUDE_forum_member_count_l607_60724


namespace NUMINAMATH_CALUDE_sqrt_9801_minus_99_proof_l607_60761

theorem sqrt_9801_minus_99_proof (a b : ℕ) (h1 : 0 < a) (h2 : 0 < b) 
  (h3 : Real.sqrt 9801 - 99 = (Real.sqrt a - b)^3) : a + b = 11 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_9801_minus_99_proof_l607_60761


namespace NUMINAMATH_CALUDE_square_gt_iff_abs_gt_l607_60735

theorem square_gt_iff_abs_gt (a b : ℝ) : a^2 > b^2 ↔ |a| > |b| := by sorry

end NUMINAMATH_CALUDE_square_gt_iff_abs_gt_l607_60735


namespace NUMINAMATH_CALUDE_original_deck_size_l607_60773

/-- Represents a deck of cards with red and black cards -/
structure Deck where
  red : ℕ
  black : ℕ

/-- The probability of selecting a red card from the deck -/
def redProbability (d : Deck) : ℚ :=
  d.red / (d.red + d.black)

theorem original_deck_size :
  ∃ d : Deck,
    redProbability d = 2/5 ∧
    redProbability {red := d.red + 3, black := d.black} = 1/2 ∧
    d.red + d.black = 15 := by
  sorry

end NUMINAMATH_CALUDE_original_deck_size_l607_60773


namespace NUMINAMATH_CALUDE_susan_chairs_l607_60795

def chairs_problem (red_chairs : ℕ) (yellow_multiplier : ℕ) (blue_difference : ℕ) : Prop :=
  let yellow_chairs := red_chairs * yellow_multiplier
  let blue_chairs := yellow_chairs - blue_difference
  red_chairs + yellow_chairs + blue_chairs = 43

theorem susan_chairs : chairs_problem 5 4 2 := by
  sorry

end NUMINAMATH_CALUDE_susan_chairs_l607_60795


namespace NUMINAMATH_CALUDE_necklace_count_l607_60762

/-- Represents the number of beads of each color available -/
structure BeadInventory where
  green : ℕ
  white : ℕ
  orange : ℕ

/-- Represents the number of beads of each color required for one necklace -/
structure NecklacePattern where
  green : ℕ
  white : ℕ
  orange : ℕ

/-- Calculates the maximum number of complete necklaces that can be created -/
def maxNecklaces (inventory : BeadInventory) (pattern : NecklacePattern) : ℕ :=
  min (inventory.green / pattern.green)
      (min (inventory.white / pattern.white)
           (inventory.orange / pattern.orange))

theorem necklace_count 
  (inventory : BeadInventory)
  (pattern : NecklacePattern)
  (h_inventory : inventory = { green := 200, white := 100, orange := 50 })
  (h_pattern : pattern = { green := 3, white := 1, orange := 1 }) :
  maxNecklaces inventory pattern = 50 := by
  sorry

#eval maxNecklaces { green := 200, white := 100, orange := 50 } { green := 3, white := 1, orange := 1 }

end NUMINAMATH_CALUDE_necklace_count_l607_60762


namespace NUMINAMATH_CALUDE_smallest_n_divisible_by_2009_n_42_divisible_by_2009_exists_unique_smallest_n_l607_60784

theorem smallest_n_divisible_by_2009 :
  ∀ n : ℕ, n > 1 → n^2 * (n - 1) % 2009 = 0 → n ≥ 42 :=
by
  sorry

theorem n_42_divisible_by_2009 : 42^2 * (42 - 1) % 2009 = 0 :=
by
  sorry

theorem exists_unique_smallest_n :
  ∃! n : ℕ, n > 1 ∧ n^2 * (n - 1) % 2009 = 0 ∧ ∀ m : ℕ, m > 1 → m^2 * (m - 1) % 2009 = 0 → n ≤ m :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_n_divisible_by_2009_n_42_divisible_by_2009_exists_unique_smallest_n_l607_60784


namespace NUMINAMATH_CALUDE_wait_probability_is_two_thirds_l607_60777

/-- The duration of the red light in seconds -/
def red_light_duration : ℕ := 30

/-- The minimum waiting time in seconds -/
def min_wait_time : ℕ := 10

/-- The probability of waiting at least 'min_wait_time' seconds for the green light -/
def wait_probability : ℚ := (red_light_duration - min_wait_time) / red_light_duration

theorem wait_probability_is_two_thirds : 
  wait_probability = 2 / 3 := by sorry

end NUMINAMATH_CALUDE_wait_probability_is_two_thirds_l607_60777


namespace NUMINAMATH_CALUDE_max_value_of_f_l607_60726

noncomputable def f (x : ℝ) := Real.sin x ^ 2 + Real.sqrt 3 * Real.cos x - 3/4

theorem max_value_of_f :
  ∃ (x : ℝ), x ∈ Set.Icc 0 (Real.pi / 2) ∧
  (∀ (y : ℝ), y ∈ Set.Icc 0 (Real.pi / 2) → f y ≤ f x) ∧
  f x = 1 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_f_l607_60726


namespace NUMINAMATH_CALUDE_iesha_book_count_l607_60793

/-- Represents the number of books Iesha has -/
structure IeshasBooks where
  school : ℕ
  sports : ℕ

/-- The total number of books Iesha has -/
def total_books (b : IeshasBooks) : ℕ := b.school + b.sports

theorem iesha_book_count : 
  ∀ (b : IeshasBooks), b.school = 19 → b.sports = 39 → total_books b = 58 := by
  sorry

end NUMINAMATH_CALUDE_iesha_book_count_l607_60793


namespace NUMINAMATH_CALUDE_stella_restocks_six_bathrooms_l607_60780

/-- The number of bathrooms Stella restocks -/
def num_bathrooms : ℕ :=
  let rolls_per_day : ℕ := 1
  let days_per_week : ℕ := 7
  let num_weeks : ℕ := 4
  let rolls_per_pack : ℕ := 12
  let packs_bought : ℕ := 14
  let rolls_per_bathroom : ℕ := rolls_per_day * days_per_week * num_weeks
  let total_rolls_bought : ℕ := packs_bought * rolls_per_pack
  total_rolls_bought / rolls_per_bathroom

theorem stella_restocks_six_bathrooms : num_bathrooms = 6 := by
  sorry

end NUMINAMATH_CALUDE_stella_restocks_six_bathrooms_l607_60780


namespace NUMINAMATH_CALUDE_pocket_probabilities_l607_60719

/-- Represents the number of balls in the pocket -/
def total_balls : ℕ := 5

/-- Represents the number of white balls in the pocket -/
def white_balls : ℕ := 3

/-- Represents the number of black balls in the pocket -/
def black_balls : ℕ := 2

/-- Represents the number of balls drawn at once -/
def drawn_balls : ℕ := 2

/-- The total number of ways to draw 2 balls from 5 balls -/
def total_events : ℕ := Nat.choose total_balls drawn_balls

/-- The probability of drawing two white balls -/
def prob_two_white : ℚ := (Nat.choose white_balls drawn_balls : ℚ) / total_events

/-- The probability of drawing one black and one white ball -/
def prob_one_black_one_white : ℚ := (white_balls * black_balls : ℚ) / total_events

theorem pocket_probabilities :
  total_events = 10 ∧
  prob_two_white = 3 / 10 ∧
  prob_one_black_one_white = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_pocket_probabilities_l607_60719


namespace NUMINAMATH_CALUDE_treasure_chest_gems_l607_60791

theorem treasure_chest_gems (total_gems rubies : ℕ) 
  (h1 : total_gems = 5155)
  (h2 : rubies = 5110)
  (h3 : total_gems ≥ rubies) :
  total_gems - rubies = 45 := by
  sorry

end NUMINAMATH_CALUDE_treasure_chest_gems_l607_60791


namespace NUMINAMATH_CALUDE_fuel_fraction_proof_l607_60785

def road_trip_fuel_calculation (total_fuel : ℝ) (first_third : ℝ) (second_third_fraction : ℝ) : Prop :=
  let second_third := total_fuel * second_third_fraction
  let final_third := total_fuel - first_third - second_third
  final_third / second_third = 1 / 2

theorem fuel_fraction_proof :
  road_trip_fuel_calculation 60 30 (1/3) :=
by
  sorry

end NUMINAMATH_CALUDE_fuel_fraction_proof_l607_60785


namespace NUMINAMATH_CALUDE_angle_range_l607_60727

theorem angle_range (θ : Real) (k : Int) : 
  (π < θ ∧ θ < 3*π/2) →   -- θ is in the third quadrant
  (Real.sin (θ/4) < Real.cos (θ/4)) →  -- sin(θ/4) < cos(θ/4)
  (∃ k, (2*k*π + 5*π/4 < θ/4 ∧ θ/4 < 2*k*π + 11*π/8) ∨ 
        (2*k*π + 7*π/4 < θ/4 ∧ θ/4 < 2*k*π + 15*π/8)) :=
by sorry

end NUMINAMATH_CALUDE_angle_range_l607_60727


namespace NUMINAMATH_CALUDE_arc_length_for_given_angle_and_radius_l607_60701

/-- Given a circle with radius 3 and a central angle of π/7, 
    the corresponding arc length is 3π/7 -/
theorem arc_length_for_given_angle_and_radius :
  ∀ (r : ℝ) (θ : ℝ),
    r = 3 →
    θ = π / 7 →
    r * θ = 3 * π / 7 := by
  sorry

end NUMINAMATH_CALUDE_arc_length_for_given_angle_and_radius_l607_60701


namespace NUMINAMATH_CALUDE_possible_values_of_a_l607_60757

def A (a : ℝ) : Set ℝ := {2, a^2 - a + 2, 1 - a}

theorem possible_values_of_a (a : ℝ) : 4 ∈ A a → a = -3 ∨ a = 2 := by
  sorry

end NUMINAMATH_CALUDE_possible_values_of_a_l607_60757


namespace NUMINAMATH_CALUDE_smallest_S_value_l607_60712

/-- Represents a standard 6-sided die -/
def Die := Fin 6

/-- The number of dice rolled -/
def n : ℕ := 342

/-- The sum we're comparing against -/
def target_sum : ℕ := 2052

/-- Function to calculate the probability of obtaining a specific sum -/
noncomputable def prob_of_sum (sum : ℕ) : ℝ := sorry

/-- The smallest sum S that has the same probability as the target sum -/
def S : ℕ := 342

theorem smallest_S_value :
  (prob_of_sum target_sum > 0) ∧ 
  (∀ s : ℕ, s < S → prob_of_sum s ≠ prob_of_sum target_sum) ∧
  (prob_of_sum S = prob_of_sum target_sum) := by sorry

end NUMINAMATH_CALUDE_smallest_S_value_l607_60712


namespace NUMINAMATH_CALUDE_x_squared_gt_y_squared_necessary_not_sufficient_l607_60767

theorem x_squared_gt_y_squared_necessary_not_sufficient (x y : ℝ) :
  (∀ x y, x < y ∧ y < 0 → x^2 > y^2) ∧
  (∃ x y, x^2 > y^2 ∧ ¬(x < y ∧ y < 0)) := by
  sorry

end NUMINAMATH_CALUDE_x_squared_gt_y_squared_necessary_not_sufficient_l607_60767


namespace NUMINAMATH_CALUDE_hall_volume_l607_60706

/-- Given a rectangular hall with length 15 m and breadth 12 m, if the sum of the areas of
    the floor and ceiling is equal to the sum of the areas of four walls, then the volume
    of the hall is 8004 m³. -/
theorem hall_volume (height : ℝ) : 
  (15 : ℝ) * 12 * height = 8004 ∧ 
  2 * (15 * 12) = 2 * (15 * height) + 2 * (12 * height) := by
  sorry

#check hall_volume

end NUMINAMATH_CALUDE_hall_volume_l607_60706


namespace NUMINAMATH_CALUDE_line_separation_parameter_range_l607_60772

/-- Given a line 2x - y + a = 0 where the origin (0, 0) and the point (1, 1) 
    are on opposite sides of this line, prove that -1 < a < 0 -/
theorem line_separation_parameter_range :
  ∀ a : ℝ, 
  (∀ x y : ℝ, 2*x - y + a = 0 → 
    ((0 : ℝ) < 2*0 - 0 + a) ≠ ((0 : ℝ) < 2*1 - 1 + a)) →
  -1 < a ∧ a < 0 :=
by sorry

end NUMINAMATH_CALUDE_line_separation_parameter_range_l607_60772


namespace NUMINAMATH_CALUDE_money_distribution_l607_60738

theorem money_distribution (a b c : ℝ) : 
  a = (1/3) * (b + c) →
  b = (2/7) * (a + c) →
  a + b + c = 360 →
  a - b = 10 :=
by sorry

end NUMINAMATH_CALUDE_money_distribution_l607_60738


namespace NUMINAMATH_CALUDE_modulus_of_complex_fraction_l607_60720

theorem modulus_of_complex_fraction : 
  Complex.abs ((3 - 4 * Complex.I) / Complex.I) = 5 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_complex_fraction_l607_60720


namespace NUMINAMATH_CALUDE_johns_age_l607_60743

theorem johns_age (john_age dad_age : ℕ) 
  (h1 : john_age = dad_age - 24)
  (h2 : john_age + dad_age = 68) : 
  john_age = 22 := by
  sorry

end NUMINAMATH_CALUDE_johns_age_l607_60743


namespace NUMINAMATH_CALUDE_gcd_1037_425_l607_60786

theorem gcd_1037_425 : Nat.gcd 1037 425 = 17 := by
  sorry

end NUMINAMATH_CALUDE_gcd_1037_425_l607_60786


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l607_60703

theorem quadratic_inequality_solution_set :
  {x : ℝ | x^2 + 2*x < 3} = {x : ℝ | -3 < x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l607_60703


namespace NUMINAMATH_CALUDE_current_task2_hours_proof_l607_60787

/-- Calculates the current hours spent on task 2 per day given work conditions -/
def current_task2_hours (total_weekly_hours : ℕ) (work_days : ℕ) (task1_daily_hours : ℕ) (task1_reduction : ℕ) : ℕ :=
  let task1_weekly_hours := task1_daily_hours * work_days
  let new_task1_weekly_hours := task1_weekly_hours - task1_reduction
  let task2_weekly_hours := total_weekly_hours - new_task1_weekly_hours
  task2_weekly_hours / work_days

theorem current_task2_hours_proof :
  current_task2_hours 40 5 5 5 = 4 := by
  sorry

end NUMINAMATH_CALUDE_current_task2_hours_proof_l607_60787


namespace NUMINAMATH_CALUDE_set_B_equals_l607_60796

-- Define set A
def A : Set Int := {-1, 0, 1, 2}

-- Define the function f(x) = x^2 - 2x
def f (x : Int) : Int := x^2 - 2*x

-- Define set B
def B : Set Int := {y | ∃ x ∈ A, f x = y}

-- Theorem statement
theorem set_B_equals : B = {-1, 0, 3} := by sorry

end NUMINAMATH_CALUDE_set_B_equals_l607_60796


namespace NUMINAMATH_CALUDE_store_uniforms_problem_l607_60733

theorem store_uniforms_problem :
  ∃ (u : ℕ), 
    u > 0 ∧ 
    (u + 1) % 2 = 0 ∧ 
    u = 3 :=
by sorry

end NUMINAMATH_CALUDE_store_uniforms_problem_l607_60733


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l607_60797

theorem quadratic_inequality_range (m : ℝ) : 
  (∀ x : ℝ, m * x^2 + m * x + 100 > 0) ↔ (0 ≤ m ∧ m < 400) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l607_60797


namespace NUMINAMATH_CALUDE_power_of_product_l607_60758

theorem power_of_product (a b : ℝ) : (3 * a * b)^2 = 9 * a^2 * b^2 := by
  sorry

end NUMINAMATH_CALUDE_power_of_product_l607_60758


namespace NUMINAMATH_CALUDE_absolute_difference_of_roots_l607_60771

-- Define the quadratic equation
def quadratic_equation (k : ℝ) (x : ℝ) : ℝ := x^2 - (k+3)*x + k

-- Define the roots of the quadratic equation
def roots (k : ℝ) : ℝ × ℝ := sorry

-- Theorem statement
theorem absolute_difference_of_roots (k : ℝ) :
  let (r₁, r₂) := roots k
  |r₁ - r₂| = Real.sqrt (k^2 + 2*k + 9) := by sorry

end NUMINAMATH_CALUDE_absolute_difference_of_roots_l607_60771


namespace NUMINAMATH_CALUDE_quadrilateral_circumscribed_l607_60783

-- Define the structure for a quadrilateral
structure Quadrilateral :=
  (A B C D : Point)

-- Define the property of being convex
def is_convex (q : Quadrilateral) : Prop := sorry

-- Define the property of a point being inside a quadrilateral
def is_interior_point (P : Point) (q : Quadrilateral) : Prop := sorry

-- Define angle measure
def angle_measure (P Q R : Point) : ℝ := sorry

-- Define the property of being circumscribed
def is_circumscribed (q : Quadrilateral) : Prop := sorry

theorem quadrilateral_circumscribed 
  (q : Quadrilateral) (P : Point) 
  (h_convex : is_convex q)
  (h_interior : is_interior_point P q)
  (h_angle1 : angle_measure q.A P q.B + angle_measure q.C P q.D = 
              angle_measure q.B P q.C + angle_measure q.D P q.A)
  (h_angle2 : angle_measure P q.A q.D + angle_measure P q.C q.D = 
              angle_measure P q.A q.B + angle_measure P q.C q.B)
  (h_angle3 : angle_measure P q.D q.C + angle_measure P q.B q.C = 
              angle_measure P q.D q.A + angle_measure P q.B q.A) :
  is_circumscribed q :=
sorry

end NUMINAMATH_CALUDE_quadrilateral_circumscribed_l607_60783


namespace NUMINAMATH_CALUDE_julia_wednesday_kids_l607_60732

/-- The number of kids Julia played with on Monday -/
def monday_kids : ℕ := 17

/-- The number of kids Julia played with on Tuesday -/
def tuesday_kids : ℕ := 15

/-- The total number of kids Julia played with over the three days -/
def total_kids : ℕ := 34

/-- The number of kids Julia played with on Wednesday -/
def wednesday_kids : ℕ := total_kids - (monday_kids + tuesday_kids)

theorem julia_wednesday_kids : wednesday_kids = 2 := by
  sorry

end NUMINAMATH_CALUDE_julia_wednesday_kids_l607_60732


namespace NUMINAMATH_CALUDE_a_equals_2_sufficient_not_necessary_l607_60702

def third_term (a : ℝ) : ℝ → ℝ := λ x ↦ 15 * a^2 * x^4

theorem a_equals_2_sufficient_not_necessary :
  (∀ x, third_term 2 x = 60 * x^4) ∧
  (∃ a ≠ 2, ∀ x, third_term a x = 60 * x^4) :=
sorry

end NUMINAMATH_CALUDE_a_equals_2_sufficient_not_necessary_l607_60702
