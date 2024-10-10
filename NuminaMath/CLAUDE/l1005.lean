import Mathlib

namespace circle_center_condition_l1005_100522

-- Define the circle equation
def circle_equation (x y k : ℝ) : Prop :=
  x^2 + y^2 + 2*k*x + 4*y + 3*k + 8 = 0

-- Define the condition for the center to be in the third quadrant
def center_in_third_quadrant (k : ℝ) : Prop :=
  k > 0 ∧ -2 < 0

-- Define the range of k
def k_range (k : ℝ) : Prop :=
  k > 4

-- Theorem statement
theorem circle_center_condition (k : ℝ) :
  (∃ x y : ℝ, circle_equation x y k) ∧ 
  center_in_third_quadrant k →
  k_range k :=
by sorry

end circle_center_condition_l1005_100522


namespace quadratic_rewrite_l1005_100521

theorem quadratic_rewrite (a b c : ℤ) :
  (∀ x : ℝ, 16 * x^2 - 40 * x - 72 = (a * x + b)^2 + c) →
  a * b = -20 := by
sorry

end quadratic_rewrite_l1005_100521


namespace intersection_complement_equality_l1005_100531

open Set

-- Define the sets A and B
def A : Set ℝ := {x | ∃ y, y = Real.log (9 - x^2)}
def B : Set ℝ := {x | ∃ y, y = Real.sqrt (4*x - x^2)}

-- State the theorem
theorem intersection_complement_equality :
  A ∩ (Bᶜ) = Ioo (-3) 0 := by sorry

end intersection_complement_equality_l1005_100531


namespace problem_1_problem_2_l1005_100513

-- Problem 1
theorem problem_1 : Real.sqrt 8 - (1/2)⁻¹ + 4 * Real.sin (30 * π / 180) = 2 * Real.sqrt 2 := by
  sorry

-- Problem 2
theorem problem_2 : (2^2 - 9) / (2^2 + 6*2 + 9) / (1 - 2 / (2 + 3)) = -1/3 := by
  sorry

end problem_1_problem_2_l1005_100513


namespace zero_in_interval_l1005_100581

noncomputable def f (x : ℝ) : ℝ := 6 / x - Real.log x / Real.log 2

theorem zero_in_interval :
  ∃ c ∈ Set.Ioo 2 4, f c = 0 :=
sorry

end zero_in_interval_l1005_100581


namespace sally_total_spent_l1005_100538

/-- The amount Sally paid for peaches after applying a coupon -/
def peaches_price : ℚ := 12.32

/-- The amount of the coupon applied to the peaches purchase -/
def coupon_amount : ℚ := 3

/-- The amount Sally paid for cherries -/
def cherries_price : ℚ := 11.54

/-- The theorem stating that the total amount Sally spent is $23.86 -/
theorem sally_total_spent : peaches_price + cherries_price = 23.86 := by
  sorry

end sally_total_spent_l1005_100538


namespace prob_three_red_marbles_l1005_100552

def total_marbles : ℕ := 5 + 6 + 7

def prob_all_red (red white blue : ℕ) : ℚ :=
  (red : ℚ) / total_marbles *
  ((red - 1) : ℚ) / (total_marbles - 1) *
  ((red - 2) : ℚ) / (total_marbles - 2)

theorem prob_three_red_marbles :
  prob_all_red 5 6 7 = 5 / 408 := by
  sorry

end prob_three_red_marbles_l1005_100552


namespace clara_age_l1005_100515

def anna_age : ℕ := 54
def years_ago : ℕ := 41

theorem clara_age : ℕ :=
  let anna_age_then := anna_age - years_ago
  let clara_age_then := 3 * anna_age_then
  clara_age_then + years_ago

#check clara_age

end clara_age_l1005_100515


namespace cookies_eaten_yesterday_l1005_100596

/-- Given the number of cookies eaten today and the difference between today and yesterday,
    calculate the number of cookies eaten yesterday. -/
def cookies_yesterday (today : ℕ) (difference : ℕ) : ℕ :=
  today - difference

/-- Theorem stating that given 140 cookies eaten today and 30 fewer yesterday,
    the number of cookies eaten yesterday was 110. -/
theorem cookies_eaten_yesterday :
  cookies_yesterday 140 30 = 110 := by
  sorry

end cookies_eaten_yesterday_l1005_100596


namespace no_valid_base_l1005_100540

theorem no_valid_base : ¬ ∃ (base : ℝ), (1/5)^35 * (1/4)^18 = 1/(2*(base^35)) := by
  sorry

end no_valid_base_l1005_100540


namespace sequence_formula_l1005_100595

-- Define the sequence and its sum
def S (n : ℕ) : ℕ := 2 * n^2 + n

-- Define the nth term of the sequence
def a (n : ℕ) : ℕ := 4 * n - 1

-- Theorem statement
theorem sequence_formula (n : ℕ) : S n - S (n-1) = a n :=
sorry

end sequence_formula_l1005_100595


namespace balloon_arrangements_l1005_100510

def word_length : ℕ := 7
def repeating_letters : ℕ := 2
def repetitions_per_letter : ℕ := 2

theorem balloon_arrangements :
  (word_length.factorial) / ((repetitions_per_letter.factorial) ^ repeating_letters) = 1260 := by
  sorry

end balloon_arrangements_l1005_100510


namespace lindsey_bands_count_l1005_100585

/-- The number of exercise bands Lindsey bought -/
def num_bands : ℕ := 2

/-- The resistance added by each band in pounds -/
def resistance_per_band : ℕ := 5

/-- The weight of the dumbbell in pounds -/
def dumbbell_weight : ℕ := 10

/-- The total weight Lindsey squats in pounds -/
def total_squat_weight : ℕ := 30

theorem lindsey_bands_count :
  (2 * num_bands * resistance_per_band + dumbbell_weight = total_squat_weight) :=
by sorry

end lindsey_bands_count_l1005_100585


namespace son_age_is_30_l1005_100576

/-- The age difference between the man and his son -/
def age_difference : ℕ := 32

/-- The present age of the son -/
def son_age : ℕ := 30

/-- The present age of the man -/
def man_age : ℕ := son_age + age_difference

theorem son_age_is_30 :
  (man_age = son_age + age_difference) ∧
  (man_age + 2 = 2 * (son_age + 2)) →
  son_age = 30 := by
  sorry

end son_age_is_30_l1005_100576


namespace train_speed_l1005_100511

/-- Calculate the speed of a train given its length and time to pass a stationary point -/
theorem train_speed (length time : ℝ) (length_positive : length > 0) (time_positive : time > 0) :
  length = 100 ∧ time = 5 → length / time = 20 := by
  sorry

end train_speed_l1005_100511


namespace production_days_l1005_100516

/-- Given that:
    1. The average daily production for the past n days was 50 units.
    2. Today's production is 110 units.
    3. The new average including today's production is 55 units.
    Prove that n = 11. -/
theorem production_days (n : ℕ) : (n * 50 + 110) / (n + 1) = 55 → n = 11 := by
  sorry

end production_days_l1005_100516


namespace min_distance_proof_l1005_100546

/-- The distance between the graphs of y = 2x and y = -x^2 - 2x - 1 at a given x -/
def distance (x : ℝ) : ℝ := 2*x - (-x^2 - 2*x - 1)

/-- The minimum non-negative distance between the graphs -/
def min_distance : ℝ := 1

theorem min_distance_proof : 
  ∀ x : ℝ, distance x ≥ 0 → distance x ≥ min_distance :=
sorry

end min_distance_proof_l1005_100546


namespace pencil_pen_multiple_l1005_100572

theorem pencil_pen_multiple (total : ℕ) (pens : ℕ) (M : ℕ) : 
  total = 108 →
  pens = 16 →
  total = pens + (M * pens + 12) →
  M = 5 := by
sorry

end pencil_pen_multiple_l1005_100572


namespace square_even_implies_even_sqrt_2_irrational_l1005_100502

-- Definition of even number
def is_even (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k

-- Definition of rational number
def is_rational (x : ℝ) : Prop := ∃ a b : ℤ, b ≠ 0 ∧ x = a / b

-- Theorem 1: If p^2 is even, then p is even
theorem square_even_implies_even (p : ℤ) : is_even (p^2) → is_even p := by sorry

-- Theorem 2: √2 is irrational
theorem sqrt_2_irrational : ¬ is_rational (Real.sqrt 2) := by sorry

end square_even_implies_even_sqrt_2_irrational_l1005_100502


namespace arithmetic_sequence_a12_l1005_100517

/-- An arithmetic sequence is a sequence where the difference between
    any two consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_a12 (a : ℕ → ℚ) 
  (h_arith : is_arithmetic_sequence a)
  (h_sum : a 3 + a 4 + a 5 = 3)
  (h_a8 : a 8 = 8) :
  a 12 = 15 := by
sorry

end arithmetic_sequence_a12_l1005_100517


namespace geometric_arithmetic_sequence_l1005_100503

theorem geometric_arithmetic_sequence (x y z : ℝ) 
  (h1 : (4 * y)^2 = (3 * x) * (5 * z))  -- Geometric sequence condition
  (h2 : 2 / y = 1 / x + 1 / z)          -- Arithmetic sequence condition
  : x / z + z / x = 34 / 15 := by
  sorry

end geometric_arithmetic_sequence_l1005_100503


namespace fifth_equation_in_pattern_l1005_100593

theorem fifth_equation_in_pattern : 
  (∀ n : ℕ, n ≥ 1 ∧ n ≤ 4 → 
    (List.range n).sum + (List.range n).sum.succ = n^2) →
  (List.range 5).sum + (List.range 5).sum.succ = 81 :=
sorry

end fifth_equation_in_pattern_l1005_100593


namespace blended_number_property_l1005_100574

def is_blended_number (n : ℕ) : Prop :=
  ∃ (a b : ℕ), 1 ≤ a ∧ a ≤ 9 ∧ 1 ≤ b ∧ b ≤ 9 ∧ n = 1000 * a + 100 * b + 10 * a + b

def F (t : ℕ) : ℚ :=
  let t' := (t % 100) * 100 + (t / 100)
  2 * (t + t') / 1111

theorem blended_number_property (p q : ℕ) (a b c d : ℕ) :
  is_blended_number p →
  is_blended_number q →
  p = 1000 * a + 100 * b + 10 * a + b →
  q = 1000 * c + 100 * d + 10 * c + d →
  1 ≤ a →
  a < b →
  b ≤ 9 →
  1 ≤ c →
  c ≤ 9 →
  1 ≤ d →
  d ≤ 9 →
  c ≠ d →
  ∃ (k : ℤ), F p = 17 * k →
  F p + 2 * F q - (4 * a + 3 * b + 2 * d + c) = 0 →
  F (p - q) = 12 ∨ F (p - q) = 16 := by sorry

end blended_number_property_l1005_100574


namespace projectile_max_height_l1005_100570

/-- The height function of the projectile -/
def h (t : ℝ) : ℝ := -20 * t^2 + 40 * t + 10

/-- Theorem stating that the maximum height of the projectile is 30 -/
theorem projectile_max_height :
  ∃ (t_max : ℝ), ∀ (t : ℝ), h t ≤ h t_max ∧ h t_max = 30 :=
sorry

end projectile_max_height_l1005_100570


namespace calculation_proof_l1005_100597

theorem calculation_proof : 3 * (-4) - ((5 * (-5)) * (-2)) + 6 = -56 := by
  sorry

end calculation_proof_l1005_100597


namespace roots_on_circle_l1005_100557

theorem roots_on_circle (z : ℂ) : 
  (z + 1)^4 = 16 * z^4 → Complex.abs (z - Complex.ofReal (1/3)) = 2/3 := by
  sorry

end roots_on_circle_l1005_100557


namespace english_chinese_difference_l1005_100532

/-- Represents the number of hours Ryan spends studying each subject on weekdays and weekends --/
structure StudyHours where
  english_weekday : ℕ
  chinese_weekday : ℕ
  english_weekend : ℕ
  chinese_weekend : ℕ

/-- Calculates the total hours spent on a subject in a week --/
def total_hours (hours : StudyHours) (weekday : ℕ) (weekend : ℕ) : ℕ :=
  hours.english_weekday * weekday + hours.chinese_weekday * weekday +
  hours.english_weekend * weekend + hours.chinese_weekend * weekend

/-- Theorem stating the difference in hours spent on English vs Chinese --/
theorem english_chinese_difference (hours : StudyHours) 
  (h1 : hours.english_weekday = 6)
  (h2 : hours.chinese_weekday = 3)
  (h3 : hours.english_weekend = 2)
  (h4 : hours.chinese_weekend = 1)
  : total_hours hours 5 2 = 17 := by
  sorry

end english_chinese_difference_l1005_100532


namespace min_sum_on_parabola_l1005_100528

theorem min_sum_on_parabola :
  ∀ n m : ℕ,
  m = 19 * n^2 - 98 * n →
  102 ≤ m + n :=
by sorry

end min_sum_on_parabola_l1005_100528


namespace truck_speed_problem_l1005_100587

/-- The average speed of Truck Y in miles per hour -/
def speed_y : ℝ := 63

/-- The time it takes for Truck Y to overtake Truck X in hours -/
def overtake_time : ℝ := 3

/-- The initial distance Truck X is ahead of Truck Y in miles -/
def initial_gap : ℝ := 14

/-- The distance Truck Y is ahead of Truck X after overtaking in miles -/
def final_gap : ℝ := 4

/-- The average speed of Truck X in miles per hour -/
def speed_x : ℝ := 57

theorem truck_speed_problem :
  speed_y * overtake_time = speed_x * overtake_time + initial_gap + final_gap := by
  sorry

#check truck_speed_problem

end truck_speed_problem_l1005_100587


namespace three_digit_number_appended_to_1220_l1005_100529

theorem three_digit_number_appended_to_1220 :
  ∃! n : ℕ, 100 ≤ n ∧ n < 1000 ∧ (1220000 + n) % 2014 = 0 :=
by
  -- The proof goes here
  sorry

end three_digit_number_appended_to_1220_l1005_100529


namespace greatest_b_value_l1005_100507

theorem greatest_b_value (b : ℝ) : 
  (∀ x : ℝ, -x^2 + 9*x - 14 ≥ 0 → x ≤ 7) ∧ 
  (-7^2 + 9*7 - 14 ≥ 0) :=
sorry

end greatest_b_value_l1005_100507


namespace spending_difference_is_30_l1005_100508

-- Define the quantities and prices
def ice_cream_cartons : ℕ := 10
def yoghurt_cartons : ℕ := 4
def ice_cream_price : ℚ := 4
def yoghurt_price : ℚ := 1

-- Define the discount and tax rates
def ice_cream_discount : ℚ := 15 / 100
def sales_tax : ℚ := 5 / 100

-- Define the function to calculate the difference in spending
def difference_in_spending : ℚ :=
  let ice_cream_cost := ice_cream_cartons * ice_cream_price
  let ice_cream_discounted := ice_cream_cost * (1 - ice_cream_discount)
  let yoghurt_cost := yoghurt_cartons * yoghurt_price
  ice_cream_discounted - yoghurt_cost

-- Theorem statement
theorem spending_difference_is_30 : difference_in_spending = 30 := by
  sorry

end spending_difference_is_30_l1005_100508


namespace gain_percentage_calculation_l1005_100526

theorem gain_percentage_calculation (selling_price gain : ℝ) : 
  selling_price = 225 → gain = 75 → (gain / (selling_price - gain)) * 100 = 50 := by
  sorry

end gain_percentage_calculation_l1005_100526


namespace square_root_of_36_l1005_100551

theorem square_root_of_36 : ∃ x : ℝ, x ^ 2 = 36 ∧ (x = 6 ∨ x = -6) := by
  sorry

end square_root_of_36_l1005_100551


namespace intersection_A_complement_B_l1005_100535

def A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 2}
def B : Set ℝ := {x | x < 1}

theorem intersection_A_complement_B : A ∩ (Set.univ \ B) = Set.Icc 1 2 := by
  sorry

end intersection_A_complement_B_l1005_100535


namespace probability_two_one_is_four_fifths_l1005_100584

def total_balls : ℕ := 15
def black_balls : ℕ := 8
def white_balls : ℕ := 7
def drawn_balls : ℕ := 3

def probability_two_one : ℚ :=
  let total_ways := Nat.choose total_balls drawn_balls
  let two_black_one_white := Nat.choose black_balls 2 * Nat.choose white_balls 1
  let one_black_two_white := Nat.choose black_balls 1 * Nat.choose white_balls 2
  let favorable_ways := two_black_one_white + one_black_two_white
  ↑favorable_ways / ↑total_ways

theorem probability_two_one_is_four_fifths :
  probability_two_one = 4 / 5 := by
  sorry

end probability_two_one_is_four_fifths_l1005_100584


namespace repeating_decimal_ratio_l1005_100555

/-- Represents a repeating decimal with an integer part and a repeating fractional part. -/
structure RepeatingDecimal where
  integerPart : ℤ
  repeatingPart : ℕ

/-- Converts a RepeatingDecimal to a rational number. -/
def toRational (x : RepeatingDecimal) : ℚ :=
  x.integerPart + x.repeatingPart / (99 : ℚ)

/-- The repeating decimal 0.overline{45} -/
def a : RepeatingDecimal := ⟨0, 45⟩

/-- The repeating decimal 2.overline{18} -/
def b : RepeatingDecimal := ⟨2, 18⟩

/-- Theorem stating that the ratio of the given repeating decimals equals 5/24 -/
theorem repeating_decimal_ratio : (toRational a) / (toRational b) = 5 / 24 := by
  sorry

end repeating_decimal_ratio_l1005_100555


namespace divisibility_equivalence_l1005_100534

theorem divisibility_equivalence (a b : ℤ) :
  (13 ∣ (2 * a + 3 * b)) ↔ (13 ∣ (2 * b - 3 * a)) := by
  sorry

end divisibility_equivalence_l1005_100534


namespace polygon_interior_angles_l1005_100565

theorem polygon_interior_angles (n : ℕ) (extra_angle : ℝ) : 
  (n ≥ 3) →
  (180 * (n - 2) + extra_angle = 1800) →
  (n = 11 ∧ extra_angle = 180) :=
by sorry

end polygon_interior_angles_l1005_100565


namespace max_sum_lcm_165_l1005_100547

theorem max_sum_lcm_165 (a b c d : ℕ+) : 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  Nat.lcm (Nat.lcm (Nat.lcm a.val b.val) c.val) d.val = 165 →
  a.val + b.val + c.val + d.val ≤ 268 :=
by sorry

end max_sum_lcm_165_l1005_100547


namespace parabola_vertex_l1005_100550

/-- The vertex of a parabola defined by y = -2x^2 + 3 is (0, 3) -/
theorem parabola_vertex :
  let f : ℝ → ℝ := λ x => -2 * x^2 + 3
  ∃! p : ℝ × ℝ, p.1 = 0 ∧ p.2 = 3 ∧ ∀ x : ℝ, f x ≤ f p.1 :=
by sorry

end parabola_vertex_l1005_100550


namespace line_perp_parallel_planes_l1005_100544

-- Define the types for lines and planes
def Line : Type := sorry
def Plane : Type := sorry

-- Define the relationships between lines and planes
def perpendicular (l : Line) (p : Plane) : Prop := sorry
def parallel (p1 p2 : Plane) : Prop := sorry

-- Theorem statement
theorem line_perp_parallel_planes 
  (α β : Plane) (l : Line) 
  (h1 : α ≠ β) 
  (h2 : perpendicular l β) 
  (h3 : parallel α β) : 
  perpendicular l α :=
sorry

end line_perp_parallel_planes_l1005_100544


namespace thousand_gon_triangles_l1005_100573

/-- Given a polygon with n sides and m internal points, calculates the number of triangles formed when the points are connected to each other and to the vertices of the polygon. -/
def triangles_in_polygon (n : ℕ) (m : ℕ) : ℕ :=
  n + 2 * m - 2

/-- Theorem stating that in a 1000-sided polygon with 500 internal points, 1998 triangles are formed. -/
theorem thousand_gon_triangles :
  triangles_in_polygon 1000 500 = 1998 := by
  sorry

end thousand_gon_triangles_l1005_100573


namespace fahrenheit_celsius_conversion_l1005_100560

theorem fahrenheit_celsius_conversion (F C : ℝ) : C = (5 / 9) * (F - 30) → C = 30 → F = 84 := by
  sorry

end fahrenheit_celsius_conversion_l1005_100560


namespace mo_hot_chocolate_cups_l1005_100543

/-- Represents Mo's drinking habits and last week's statistics -/
structure MoDrinkingHabits where
  rainyDayHotChocolate : ℕ  -- Number of hot chocolate cups on rainy days
  nonRainyDayTea : ℕ        -- Number of tea cups on non-rainy days
  totalCups : ℕ             -- Total cups drunk last week
  teaMoreThanHotChocolate : ℕ  -- Difference between tea and hot chocolate cups
  rainyDays : ℕ             -- Number of rainy days last week

/-- Theorem stating that Mo drinks 11 cups of hot chocolate on rainy mornings -/
theorem mo_hot_chocolate_cups (mo : MoDrinkingHabits)
    (h1 : mo.nonRainyDayTea = 5)
    (h2 : mo.totalCups = 36)
    (h3 : mo.teaMoreThanHotChocolate = 14)
    (h4 : mo.rainyDays = 2) :
    mo.rainyDayHotChocolate = 11 := by
  sorry

end mo_hot_chocolate_cups_l1005_100543


namespace sum_of_distinct_roots_l1005_100548

theorem sum_of_distinct_roots (x y : ℝ) (h1 : x ≠ y) (h2 : x^2 - 2000*x = y^2 - 2000*y) : x + y = 2000 := by
  sorry

end sum_of_distinct_roots_l1005_100548


namespace min_magnitude_u_l1005_100561

/-- The minimum magnitude of vector u -/
theorem min_magnitude_u (a b : ℝ × ℝ) (h1 : a = (Real.cos (25 * π / 180), Real.sin (25 * π / 180)))
  (h2 : b = (Real.sin (20 * π / 180), Real.cos (20 * π / 180))) :
  (∃ (t : ℝ), ∀ (s : ℝ), ‖a + s • b‖ ≥ ‖a + t • b‖) ∧
  (∃ (u : ℝ × ℝ), ∃ (t : ℝ), u = a + t • b ∧ ‖u‖ = Real.sqrt 2 / 2) :=
sorry

end min_magnitude_u_l1005_100561


namespace find_cd_l1005_100563

def repeating_decimal (c d : ℕ) : ℚ :=
  1 + (10 * c + d : ℚ) / 99

theorem find_cd (c d : ℕ) (h_c : c < 10) (h_d : d < 10) : 
  42 * (repeating_decimal c d - (1 + (10 * c + d : ℚ) / 100)) = 4/5 → 
  c = 1 ∧ d = 9 := by
sorry

end find_cd_l1005_100563


namespace uncovered_area_three_circles_l1005_100530

theorem uncovered_area_three_circles (R : ℝ) (h : R = 10) :
  let r := R / 2
  let larger_circle_area := π * R^2
  let smaller_circle_area := π * r^2
  let total_smaller_circles_area := 3 * smaller_circle_area
  larger_circle_area - total_smaller_circles_area = 25 * π :=
by sorry

end uncovered_area_three_circles_l1005_100530


namespace sum_of_roots_quadratic_l1005_100567

theorem sum_of_roots_quadratic (x₁ x₂ : ℝ) : 
  (2 * x₁^2 + 6 * x₁ - 1 = 0) → 
  (2 * x₂^2 + 6 * x₂ - 1 = 0) → 
  (x₁ + x₂ = -3) := by
sorry

end sum_of_roots_quadratic_l1005_100567


namespace fraction_value_l1005_100525

theorem fraction_value : (1 * 2 * 3 * 4) / (1 + 2 + 3 + 6) = 2 := by
  sorry

end fraction_value_l1005_100525


namespace cupcakes_theorem_l1005_100527

/-- The number of cupcakes when shared equally among children -/
def cupcakes_per_child : ℕ := 12

/-- The number of children sharing the cupcakes -/
def number_of_children : ℕ := 8

/-- The total number of cupcakes -/
def total_cupcakes : ℕ := cupcakes_per_child * number_of_children

theorem cupcakes_theorem : total_cupcakes = 96 := by
  sorry

end cupcakes_theorem_l1005_100527


namespace circle_radius_l1005_100569

theorem circle_radius (k : ℝ) : 
  (∀ x y : ℝ, x^2 - 8*x + y^2 + 10*y + k = 0 ↔ (x - 4)^2 + (y + 5)^2 = 5^2) → 
  k = 16 := by
sorry

end circle_radius_l1005_100569


namespace sum_of_squares_l1005_100520

theorem sum_of_squares (a b c : ℝ) : 
  (a * b + b * c + a * c = 131) → (a + b + c = 20) → (a^2 + b^2 + c^2 = 138) := by
  sorry

end sum_of_squares_l1005_100520


namespace helmet_sales_and_pricing_l1005_100519

/-- Helmet sales and pricing problem -/
theorem helmet_sales_and_pricing
  (march_sales : ℕ)
  (may_sales : ℕ)
  (cost_price : ℝ)
  (initial_price : ℝ)
  (initial_monthly_sales : ℕ)
  (price_sensitivity : ℝ)
  (target_profit : ℝ)
  (h_march_sales : march_sales = 256)
  (h_may_sales : may_sales = 400)
  (h_cost_price : cost_price = 30)
  (h_initial_price : initial_price = 40)
  (h_initial_monthly_sales : initial_monthly_sales = 600)
  (h_price_sensitivity : price_sensitivity = 10)
  (h_target_profit : target_profit = 10000)
  :
  ∃ (r : ℝ) (actual_price : ℝ),
    r > 0 ∧
    r = 0.25 ∧
    actual_price = 50 ∧
    march_sales * (1 + r)^2 = may_sales ∧
    (actual_price - cost_price) * (initial_monthly_sales - price_sensitivity * (actual_price - initial_price)) = target_profit ∧
    actual_price ≥ initial_price :=
by sorry

end helmet_sales_and_pricing_l1005_100519


namespace tom_charges_twelve_l1005_100537

/-- Represents Tom's lawn mowing business --/
structure LawnBusiness where
  gas_cost : ℕ
  lawns_mowed : ℕ
  extra_income : ℕ
  total_profit : ℕ

/-- Calculates the price per lawn given Tom's business details --/
def price_per_lawn (b : LawnBusiness) : ℚ :=
  (b.total_profit + b.gas_cost - b.extra_income) / b.lawns_mowed

/-- Theorem stating that Tom charges $12 per lawn --/
theorem tom_charges_twelve (tom : LawnBusiness) 
  (h1 : tom.gas_cost = 17)
  (h2 : tom.lawns_mowed = 3)
  (h3 : tom.extra_income = 10)
  (h4 : tom.total_profit = 29) : 
  price_per_lawn tom = 12 := by
  sorry


end tom_charges_twelve_l1005_100537


namespace min_value_implies_a_inequality_solution_l1005_100577

-- Define the function f
def f (x a : ℝ) : ℝ := |x - a| + |x + 2|

-- Theorem for part (i)
theorem min_value_implies_a (a : ℝ) :
  (∀ x, f x a ≥ 2) ∧ (∃ x, f x a = 2) → a = 0 ∨ a = -4 := by sorry

-- Theorem for part (ii)
theorem inequality_solution (x : ℝ) :
  f x 2 ≤ 6 ↔ x ∈ Set.Icc (-3) 3 := by sorry

-- Note: Set.Icc represents a closed interval [a, b]

end min_value_implies_a_inequality_solution_l1005_100577


namespace sum_of_max_min_xyz_l1005_100542

theorem sum_of_max_min_xyz (x y z : ℝ) 
  (h1 : x ≥ y) (h2 : y ≥ z) (h3 : z ≥ 0) 
  (h4 : 6*x + 5*y + 4*z = 120) : 
  ∃ (max_sum min_sum : ℝ), 
    (∀ (a b c : ℝ), a ≥ b → b ≥ c → c ≥ 0 → 6*a + 5*b + 4*c = 120 → a + b + c ≤ max_sum) ∧
    (∀ (a b c : ℝ), a ≥ b → b ≥ c → c ≥ 0 → 6*a + 5*b + 4*c = 120 → a + b + c ≥ min_sum) ∧
    max_sum + min_sum = 44 :=
by sorry

end sum_of_max_min_xyz_l1005_100542


namespace division_remainder_problem_l1005_100518

theorem division_remainder_problem (dividend : Nat) (divisor : Nat) (quotient : Nat) 
  (h1 : dividend = 122)
  (h2 : divisor = 20)
  (h3 : quotient = 6)
  (h4 : dividend = divisor * quotient + (dividend % divisor)) :
  dividend % divisor = 2 := by
  sorry

end division_remainder_problem_l1005_100518


namespace equation_set_solution_inequality_set_solution_l1005_100568

-- Equation set
theorem equation_set_solution :
  ∃! (x y : ℝ), x - y - 1 = 4 ∧ 4 * (x - y) - y = 5 ∧ x = 20 ∧ y = 15 := by sorry

-- Inequality set
theorem inequality_set_solution :
  ∀ x : ℝ, (4 * x - 1 ≥ x + 1 ∧ (1 - x) / 2 < x) ↔ x ≥ 2/3 := by sorry

end equation_set_solution_inequality_set_solution_l1005_100568


namespace complex_number_real_condition_l1005_100588

theorem complex_number_real_condition (m : ℝ) :
  let z : ℂ := m - 3 + (m^2 - 9) * Complex.I
  z.im = 0 → m = 3 ∨ m = -3 := by
  sorry

end complex_number_real_condition_l1005_100588


namespace vector_operation_l1005_100500

def a : Fin 2 → ℝ := ![1, 1]
def b : Fin 2 → ℝ := ![1, -1]

theorem vector_operation :
  (3 • a - 2 • b) = ![1, 5] := by sorry

end vector_operation_l1005_100500


namespace train_boggies_count_l1005_100578

/-- The length of each boggy in meters -/
def boggy_length : ℝ := 15

/-- The time in seconds for the train to cross a telegraph post before detaching a boggy -/
def initial_crossing_time : ℝ := 18

/-- The time in seconds for the train to cross a telegraph post after detaching a boggy -/
def final_crossing_time : ℝ := 16.5

/-- The number of boggies initially on the train -/
def initial_boggies : ℕ := 12

theorem train_boggies_count :
  ∃ (n : ℕ),
    (n : ℝ) * boggy_length / initial_crossing_time =
    ((n : ℝ) - 1) * boggy_length / final_crossing_time ∧
    n = initial_boggies :=
by sorry

end train_boggies_count_l1005_100578


namespace ledi_age_in_future_l1005_100571

/-- The number of years ago when the sum of Duoduo and Ledi's ages was 12 years -/
def years_ago : ℝ := 12.3

/-- Duoduo's current age -/
def duoduo_current_age : ℝ := 10

/-- The sum of Duoduo and Ledi's ages 12.3 years ago -/
def sum_ages_past : ℝ := 12

/-- The number of years until Ledi will be 10 years old -/
def years_until_ledi_ten : ℝ := 6.3

theorem ledi_age_in_future :
  ∃ (ledi_current_age : ℝ),
    ledi_current_age + duoduo_current_age = sum_ages_past + 2 * years_ago ∧
    ledi_current_age + years_until_ledi_ten = 10 :=
by sorry

end ledi_age_in_future_l1005_100571


namespace complex_ratio_theorem_l1005_100549

/-- A complex cube root of unity -/
noncomputable def ω : ℂ := Complex.exp ((2 * Real.pi * Complex.I) / 3)

/-- The theorem statement -/
theorem complex_ratio_theorem (a b c : ℂ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (hab : a / b = b / c) (hbc : b / c = c / a) :
  (a + b - c) / (a - b + c) = 1 ∨
  (a + b - c) / (a - b + c) = ω ∨
  (a + b - c) / (a - b + c) = ω^2 :=
sorry

end complex_ratio_theorem_l1005_100549


namespace lucy_age_l1005_100586

/-- Given that Lucy's age is three times Helen's age and the sum of their ages is 60,
    prove that Lucy is 45 years old. -/
theorem lucy_age (lucy helen : ℕ) 
  (h1 : lucy = 3 * helen) 
  (h2 : lucy + helen = 60) : 
  lucy = 45 := by
  sorry

end lucy_age_l1005_100586


namespace space_shuttle_speed_conversion_l1005_100580

/-- Converts kilometers per second to kilometers per hour -/
def km_per_second_to_km_per_hour (speed_km_per_second : ℝ) : ℝ :=
  speed_km_per_second * 3600

theorem space_shuttle_speed_conversion :
  km_per_second_to_km_per_hour 12 = 43200 := by
  sorry

end space_shuttle_speed_conversion_l1005_100580


namespace largest_fraction_l1005_100579

theorem largest_fraction :
  let a := 35 / 69
  let b := 7 / 15
  let c := 9 / 19
  let d := 399 / 799
  let e := 150 / 299
  (a > b) ∧ (a > c) ∧ (a > d) ∧ (a > e) := by
  sorry

end largest_fraction_l1005_100579


namespace reciprocal_of_one_third_l1005_100554

theorem reciprocal_of_one_third (x : ℚ) : x * (1/3) = 1 → x = 3 := by
  sorry

end reciprocal_of_one_third_l1005_100554


namespace ratio_problem_l1005_100575

theorem ratio_problem (ratio_percent : ℚ) (first_part : ℚ) (second_part : ℚ) :
  ratio_percent = 200 / 3 →
  first_part = 2 →
  first_part / second_part = ratio_percent / 100 →
  second_part = 3 := by
sorry

end ratio_problem_l1005_100575


namespace slices_per_pizza_large_pizza_has_12_slices_l1005_100504

/-- Calculates the number of slices in a large pizza based on soccer team statistics -/
theorem slices_per_pizza (num_pizzas : ℕ) (num_games : ℕ) (avg_goals_per_game : ℕ) : ℕ :=
  let total_goals := num_games * avg_goals_per_game
  let total_slices := total_goals
  total_slices / num_pizzas

/-- Proves that a large pizza has 12 slices given the problem conditions -/
theorem large_pizza_has_12_slices :
  slices_per_pizza 6 8 9 = 12 := by
  sorry

end slices_per_pizza_large_pizza_has_12_slices_l1005_100504


namespace interval_length_implies_difference_l1005_100509

theorem interval_length_implies_difference (a b : ℝ) : 
  (∃ x, 3 * a ≤ 4 * x + 6 ∧ 4 * x + 6 ≤ 3 * b) → 
  ((3 * b - 6) / 4 - (3 * a - 6) / 4 = 15) → 
  b - a = 20 := by
sorry

end interval_length_implies_difference_l1005_100509


namespace solution_equation1_solution_equation2_l1005_100556

-- Define the first equation
def equation1 (x : ℝ) : Prop := (x + 1) * (x + 3) = 15

-- Define the second equation
def equation2 (y : ℝ) : Prop := (y - 3)^2 + 3*(y - 3) + 2 = 0

-- Theorem for the first equation
theorem solution_equation1 : 
  (∃ x : ℝ, equation1 x) ∧ 
  (∀ x : ℝ, equation1 x ↔ (x = -6 ∨ x = 2)) :=
sorry

-- Theorem for the second equation
theorem solution_equation2 : 
  (∃ y : ℝ, equation2 y) ∧ 
  (∀ y : ℝ, equation2 y ↔ (y = 1 ∨ y = 2)) :=
sorry

end solution_equation1_solution_equation2_l1005_100556


namespace ellipse_m_range_l1005_100589

-- Define the equation
def equation (m x y : ℝ) : Prop :=
  m * (x^2 + y^2 + 2*y + 1) = (x - 2*y + 3)^2

-- Define what it means for the equation to represent an ellipse
def is_ellipse (m : ℝ) : Prop :=
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a ≠ b ∧
  ∀ (x y : ℝ), equation m x y ↔ (x^2 / a^2 + y^2 / b^2 = 1)

-- State the theorem
theorem ellipse_m_range :
  ∀ m : ℝ, is_ellipse m → m > 5 := by sorry

end ellipse_m_range_l1005_100589


namespace complex_number_range_l1005_100553

theorem complex_number_range (a : ℝ) (z : ℂ) : 
  z = a + Complex.I ∧ 
  (z.re < 0 ∧ z.im > 0) ∧ 
  Complex.abs (z * (1 + Complex.I)) > 2 → 
  a < -1 := by
  sorry

end complex_number_range_l1005_100553


namespace misread_weight_calculation_l1005_100505

theorem misread_weight_calculation (n : ℕ) (initial_avg correct_avg correct_weight : ℝ) :
  n = 20 ∧ 
  initial_avg = 58.4 ∧ 
  correct_avg = 58.6 ∧ 
  correct_weight = 60 →
  ∃ misread_weight : ℝ, 
    misread_weight = 56 ∧
    n * correct_avg - n * initial_avg = correct_weight - misread_weight :=
by sorry

end misread_weight_calculation_l1005_100505


namespace symmetric_functions_property_l1005_100599

-- Define the functions f and g
def f : ℝ → ℝ := sorry
def g : ℝ → ℝ := sorry

-- State the theorem
theorem symmetric_functions_property (h1 : ∀ x, f (x - 1) = g⁻¹ x) (h2 : g 2 = 0) : f (-1) = 2 := by
  sorry

end symmetric_functions_property_l1005_100599


namespace sequence_is_arithmetic_l1005_100566

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

theorem sequence_is_arithmetic (a : ℕ → ℝ) 
  (h : ∀ n : ℕ, a (n + 1) - a n = 3) : 
  is_arithmetic_sequence a :=
sorry

end sequence_is_arithmetic_l1005_100566


namespace l₃_equation_min_distance_l₁_l₄_l1005_100590

-- Define the lines
def l₁ (x y : ℝ) : Prop := 2 * x + y = 3
def l₂ (x y : ℝ) : Prop := x - y = 0
def l₄ (x y : ℝ) (m : ℝ) : Prop := 4 * x + 2 * y + m^2 + 1 = 0

-- Define the intersection point A
def A : ℝ × ℝ := (1, 1)

-- Theorem for the equation of l₃
theorem l₃_equation : 
  ∃ (l₃ : ℝ → ℝ → Prop), 
    (l₃ (A.1) (A.2)) ∧ 
    (∀ x y, l₃ x y ↔ x - 2*y + 1 = 0) ∧
    (∀ x y, l₁ x y → (y - A.2 = -1/2 * (x - A.1) ↔ l₃ x y)) :=
sorry

-- Theorem for the minimum distance between l₁ and l₄
theorem min_distance_l₁_l₄ :
  ∃ (d : ℝ), 
    d = 7 * Real.sqrt 5 / 10 ∧
    (∀ x y m, l₁ x y → l₄ x y m → 
      (x - 0)^2 + (y - 0)^2 ≥ d^2) :=
sorry

end l₃_equation_min_distance_l₁_l₄_l1005_100590


namespace distance_polynomial_l1005_100545

theorem distance_polynomial (m n : ℝ) : 
  ∃ (x y : ℝ), x + y = m ∧ x * y = n^2 ∧ 
  (∀ z : ℝ, z^2 - m*z + n^2 = 0 ↔ (z = x ∨ z = y)) := by
sorry

end distance_polynomial_l1005_100545


namespace batsman_new_average_is_38_l1005_100564

/-- Represents a batsman's score statistics -/
structure BatsmanStats where
  totalRuns : ℕ
  innings : ℕ
  average : ℚ

/-- Calculates the new average after an inning -/
def newAverage (stats : BatsmanStats) (newInningScore : ℕ) : ℚ :=
  (stats.totalRuns + newInningScore) / (stats.innings + 1)

/-- Theorem: Given the conditions, the batsman's new average is 38 -/
theorem batsman_new_average_is_38 
  (stats : BatsmanStats)
  (h1 : stats.innings = 16)
  (h2 : newAverage stats 86 = stats.average + 3) :
  newAverage stats 86 = 38 := by
sorry

end batsman_new_average_is_38_l1005_100564


namespace work_completion_time_l1005_100514

/-- Given that A can do a piece of work in 12 days and B is 20% more efficient than A,
    prove that B will complete the same work in 10 days. -/
theorem work_completion_time (work : ℝ) (a_time b_time : ℝ) : 
  work > 0 → 
  a_time = 12 → 
  b_time = work / ((work / a_time) * 1.2) → 
  b_time = 10 :=
by sorry

end work_completion_time_l1005_100514


namespace check_payment_inequality_l1005_100501

theorem check_payment_inequality (x y : ℕ) : 
  10 ≤ x ∧ x ≤ 99 → 
  10 ≤ y ∧ y ≤ 99 → 
  100 * y + x - (100 * x + y) = 2156 →
  100 * y + x < 2 * (100 * x + y) := by
  sorry

end check_payment_inequality_l1005_100501


namespace derivative_of_ln_2_minus_3x_l1005_100523

open Real

theorem derivative_of_ln_2_minus_3x (x : ℝ) : 
  deriv (λ x => Real.log (2 - 3*x)) x = 3 / (3*x - 2) :=
by sorry

end derivative_of_ln_2_minus_3x_l1005_100523


namespace isosceles_triangle_base_length_l1005_100592

/-- An isosceles triangle with congruent sides of length 8 cm and perimeter 27 cm has a base of length 11 cm. -/
theorem isosceles_triangle_base_length :
  ∀ (base : ℝ),
  base > 0 →
  8 > 0 →
  8 + 8 + base = 27 →
  base = 11 :=
by sorry

end isosceles_triangle_base_length_l1005_100592


namespace number_of_ones_l1005_100583

theorem number_of_ones (n : ℕ) (hn : n = 999999999) : 
  ∃ x : ℤ, (n : ℤ) * x = (10^81 - 1) / 9 := by
  sorry

end number_of_ones_l1005_100583


namespace flour_needed_for_loaves_l1005_100562

/-- The number of cups of flour needed for one loaf of bread -/
def flour_per_loaf : ℝ := 2.5

/-- The number of loaves of bread to be baked -/
def number_of_loaves : ℕ := 2

/-- Theorem: The total number of cups of flour needed for baking the desired number of loaves is 5 -/
theorem flour_needed_for_loaves : flour_per_loaf * (number_of_loaves : ℝ) = 5 := by
  sorry

end flour_needed_for_loaves_l1005_100562


namespace inequality_proof_l1005_100591

theorem inequality_proof (x y z : ℝ) : 
  (x^2 + 2*y^2 + 2*z^2)/(x^2 + y*z) + (y^2 + 2*z^2 + 2*x^2)/(y^2 + z*x) + (z^2 + 2*x^2 + 2*y^2)/(z^2 + x*y) > 6 := by
  sorry

end inequality_proof_l1005_100591


namespace cabbage_production_l1005_100558

theorem cabbage_production (last_year_side : ℕ) (this_year_side : ℕ) : 
  (this_year_side : ℤ)^2 - (last_year_side : ℤ)^2 = 127 →
  this_year_side^2 = 4096 := by
  sorry

end cabbage_production_l1005_100558


namespace solve_brownies_problem_l1005_100536

def brownies_problem (initial_brownies : ℕ) (remaining_brownies : ℕ) : Prop :=
  let admin_brownies := initial_brownies / 2
  let after_admin := initial_brownies - admin_brownies
  let carl_brownies := after_admin / 2
  let after_carl := after_admin - carl_brownies
  let final_brownies := 3
  ∃ (simon_brownies : ℕ), 
    simon_brownies = after_carl - final_brownies ∧
    simon_brownies = 2

theorem solve_brownies_problem :
  brownies_problem 20 3 := by
  sorry

end solve_brownies_problem_l1005_100536


namespace books_loaned_out_is_125_l1005_100541

/-- Represents the inter-library loan program between Library A and Library B -/
structure LibraryLoanProgram where
  initial_collection : ℕ -- Initial number of books in Library A's unique collection
  end_year_collection : ℕ -- Number of books from the unique collection in Library A at year end
  return_rate : ℚ -- Rate of return for books loaned out from Library A's unique collection
  same_year_return_rate : ℚ -- Rate of return within the same year for books from Library A's collection
  b_to_a_loan : ℕ -- Number of books loaned from Library B to Library A
  b_to_a_return_rate : ℚ -- Rate of return for books loaned from Library B to Library A

/-- Calculates the number of books loaned out from Library A's unique collection -/
def books_loaned_out (program : LibraryLoanProgram) : ℕ :=
  sorry

/-- Theorem stating that the number of books loaned out from Library A's unique collection is 125 -/
theorem books_loaned_out_is_125 (program : LibraryLoanProgram) 
  (h1 : program.initial_collection = 150)
  (h2 : program.end_year_collection = 100)
  (h3 : program.return_rate = 3/5)
  (h4 : program.same_year_return_rate = 3/10)
  (h5 : program.b_to_a_loan = 20)
  (h6 : program.b_to_a_return_rate = 1/2) :
  books_loaned_out program = 125 := by
  sorry

end books_loaned_out_is_125_l1005_100541


namespace least_faces_triangular_pyramid_l1005_100512

structure Shape where
  name : String
  faces : Nat

def triangular_prism : Shape := { name := "Triangular Prism", faces := 5 }
def quadrangular_prism : Shape := { name := "Quadrangular Prism", faces := 6 }
def triangular_pyramid : Shape := { name := "Triangular Pyramid", faces := 4 }
def quadrangular_pyramid : Shape := { name := "Quadrangular Pyramid", faces := 5 }
def truncated_quadrangular_pyramid : Shape := { name := "Truncated Quadrangular Pyramid", faces := 6 }

def shapes : List Shape := [
  triangular_prism,
  quadrangular_prism,
  triangular_pyramid,
  quadrangular_pyramid,
  truncated_quadrangular_pyramid
]

theorem least_faces_triangular_pyramid :
  ∀ s ∈ shapes, triangular_pyramid.faces ≤ s.faces :=
by sorry

end least_faces_triangular_pyramid_l1005_100512


namespace odd_sum_games_exists_l1005_100582

theorem odd_sum_games_exists (n : ℕ) (h : n = 15) : 
  ∃ (i j : ℕ) (games_played : ℕ → ℕ), 
    i < n ∧ j < n ∧ i ≠ j ∧ 
    (games_played i + games_played j) % 2 = 1 ∧
    ∀ k, k < n → games_played k ≤ n - 2 :=
by sorry

end odd_sum_games_exists_l1005_100582


namespace peters_change_l1005_100594

/-- Calculates the change left after Peter buys glasses -/
theorem peters_change (small_price large_price total_money small_count large_count : ℕ) : 
  small_price = 3 →
  large_price = 5 →
  total_money = 50 →
  small_count = 8 →
  large_count = 5 →
  total_money - (small_price * small_count + large_price * large_count) = 1 := by
sorry

end peters_change_l1005_100594


namespace poster_count_l1005_100524

/-- The total number of posters made by Mario, Samantha, and Jonathan -/
def total_posters (mario_posters samantha_posters jonathan_posters : ℕ) : ℕ :=
  mario_posters + samantha_posters + jonathan_posters

/-- Theorem stating the total number of posters made by Mario, Samantha, and Jonathan -/
theorem poster_count : ∃ (mario_posters samantha_posters jonathan_posters : ℕ),
  mario_posters = 36 ∧
  samantha_posters = mario_posters + 45 ∧
  jonathan_posters = 2 * samantha_posters ∧
  total_posters mario_posters samantha_posters jonathan_posters = 279 :=
by
  sorry

end poster_count_l1005_100524


namespace quarterly_interest_rate_proof_l1005_100506

/-- Proves that the given annual interest payment for a loan with quarterly compounding
    is consistent with the calculated quarterly interest rate. -/
theorem quarterly_interest_rate_proof
  (principal : ℝ)
  (annual_interest : ℝ)
  (quarterly_rate : ℝ)
  (h_principal : principal = 10000)
  (h_annual_interest : annual_interest = 2155.06)
  (h_quarterly_rate : quarterly_rate = 0.05) :
  annual_interest = principal * ((1 + quarterly_rate) ^ 4 - 1) :=
by sorry

end quarterly_interest_rate_proof_l1005_100506


namespace area_AMDN_eq_area_ABC_l1005_100533

-- Define the triangle ABC
variable (A B C : ℝ × ℝ)

-- Define that ABC is an acute triangle
def is_acute_triangle (A B C : ℝ × ℝ) : Prop := sorry

-- Define points E and F on side BC
def E_on_BC (E B C : ℝ × ℝ) : Prop := sorry
def F_on_BC (F B C : ℝ × ℝ) : Prop := sorry

-- Define the angle equality
def angle_BAE_eq_CAF (A B C E F : ℝ × ℝ) : Prop := sorry

-- Define perpendicular lines
def FM_perp_AB (F M A B : ℝ × ℝ) : Prop := sorry
def FN_perp_AC (F N A C : ℝ × ℝ) : Prop := sorry

-- Define D as the intersection of extended AE and the circumcircle
def D_on_circumcircle (A B C D E : ℝ × ℝ) : Prop := sorry

-- Define area function
def area (points : List (ℝ × ℝ)) : ℝ := sorry

-- Theorem statement
theorem area_AMDN_eq_area_ABC 
  (A B C D E F M N : ℝ × ℝ) 
  (h1 : is_acute_triangle A B C)
  (h2 : E_on_BC E B C)
  (h3 : F_on_BC F B C)
  (h4 : angle_BAE_eq_CAF A B C E F)
  (h5 : FM_perp_AB F M A B)
  (h6 : FN_perp_AC F N A C)
  (h7 : D_on_circumcircle A B C D E) :
  area [A, M, D, N] = area [A, B, C] := by sorry

end area_AMDN_eq_area_ABC_l1005_100533


namespace unique_integer_sum_l1005_100539

theorem unique_integer_sum : ∃! (b₃ b₄ b₅ b₆ b₇ b₈ : ℕ),
  (11 : ℚ) / 9 = b₃ / 6 + b₄ / 24 + b₅ / 120 + b₆ / 720 + b₇ / 5040 + b₈ / 40320 ∧
  b₃ < 3 ∧ b₄ < 4 ∧ b₅ < 5 ∧ b₆ < 6 ∧ b₇ < 7 ∧ b₈ < 8 ∧
  b₃ + b₄ + b₅ + b₆ + b₇ + b₈ = 25 :=
by sorry

end unique_integer_sum_l1005_100539


namespace train_length_l1005_100559

/-- Given a train traveling at 45 km/hr that crosses a 220.03-meter bridge in 30 seconds,
    the length of the train is 154.97 meters. -/
theorem train_length (train_speed : ℝ) (bridge_length : ℝ) (crossing_time : ℝ) :
  train_speed = 45 →
  bridge_length = 220.03 →
  crossing_time = 30 →
  (train_speed * 1000 / 3600) * crossing_time - bridge_length = 154.97 := by
  sorry

end train_length_l1005_100559


namespace victoria_shopping_theorem_l1005_100598

def shopping_and_dinner_problem (initial_amount : ℝ) 
  (jacket_price : ℝ) (jacket_quantity : ℕ)
  (trouser_price : ℝ) (trouser_quantity : ℕ)
  (purse_price : ℝ) (purse_quantity : ℕ)
  (discount_rate : ℝ) (dinner_bill : ℝ) : Prop :=
  let jacket_cost := jacket_price * jacket_quantity
  let trouser_cost := trouser_price * trouser_quantity
  let purse_cost := purse_price * purse_quantity
  let discountable_cost := jacket_cost + trouser_cost
  let discount_amount := discountable_cost * discount_rate
  let shopping_cost := discountable_cost - discount_amount + purse_cost
  let dinner_cost := dinner_bill / 1.15
  let total_spent := shopping_cost + dinner_cost
  let remaining_amount := initial_amount - total_spent
  remaining_amount = 3725

theorem victoria_shopping_theorem : 
  shopping_and_dinner_problem 10000 250 8 180 15 450 4 0.15 552.50 :=
by sorry

end victoria_shopping_theorem_l1005_100598
