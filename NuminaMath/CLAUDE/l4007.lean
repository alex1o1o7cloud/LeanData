import Mathlib

namespace NUMINAMATH_CALUDE_gear_speed_ratio_l4007_400730

/-- Represents a gear with a number of teeth and angular speed -/
structure Gear where
  teeth : ℕ
  speed : ℝ

/-- Represents a system of four gears meshed in sequence -/
structure GearSystem where
  A : Gear
  B : Gear
  C : Gear
  D : Gear
  meshed_AB : A.teeth * A.speed = B.teeth * B.speed
  meshed_BC : B.teeth * B.speed = C.teeth * C.speed
  meshed_CD : C.teeth * C.speed = D.teeth * D.speed

/-- The theorem stating the ratio of angular speeds for the given gear system -/
theorem gear_speed_ratio (sys : GearSystem) 
  (hA : sys.A.teeth = 10)
  (hB : sys.B.teeth = 15)
  (hC : sys.C.teeth = 20)
  (hD : sys.D.teeth = 25) :
  ∃ (k : ℝ), k > 0 ∧ 
    sys.A.speed = 24 * k ∧
    sys.B.speed = 25 * k ∧
    sys.C.speed = 12 * k ∧
    sys.D.speed = 20 * k := by
  sorry

end NUMINAMATH_CALUDE_gear_speed_ratio_l4007_400730


namespace NUMINAMATH_CALUDE_problem_solution_l4007_400723

-- Define the sets A, B, and C
def A : Set ℝ := {x | -2 < x ∧ x ≤ 3}
def B : Set ℝ := {x | x > 3 ∨ x < 2}
def C (a : ℝ) : Set ℝ := {x | x < 2 * a + 1}

-- State the theorem
theorem problem_solution :
  (∃ a : ℝ, B ∩ C a = C a) →
  ((A ∩ B = {x : ℝ | -2 < x ∧ x < 2}) ∧
   (∃ a : ℝ, ∀ x : ℝ, x ≤ 1/2 ↔ B ∩ C x = C x)) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l4007_400723


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l4007_400762

/-- Two vectors in ℝ² are parallel if their components are proportional -/
def are_parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_x_value :
  let a : ℝ × ℝ := (-2, 4)
  let b : ℝ × ℝ := (x, -2)
  are_parallel a b → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l4007_400762


namespace NUMINAMATH_CALUDE_product_mod_five_l4007_400738

theorem product_mod_five : 2011 * 2012 * 2013 * 2014 % 5 = 4 := by
  sorry

end NUMINAMATH_CALUDE_product_mod_five_l4007_400738


namespace NUMINAMATH_CALUDE_jester_win_prob_constant_l4007_400798

/-- The probability of the Jester winning in a game with 2n-1 regular townspeople, one Jester, and one goon -/
def jester_win_probability (n : ℕ+) : ℚ :=
  1 / 3

/-- The game ends immediately if the Jester is sent to jail during the morning -/
axiom morning_jail_win (n : ℕ+) : 
  jester_win_probability n = 1 / (2 * n + 1) + 
    ((2 * n - 1) / (2 * n + 1)) * ((2 * n - 2) / (2 * n - 1)) * jester_win_probability (n - 1)

/-- The Jester does not win if sent to jail at night -/
axiom night_jail_no_win (n : ℕ+) :
  jester_win_probability n = 
    ((2 * n - 1) / (2 * n + 1)) * ((2 * n - 2) / (2 * n - 1)) * jester_win_probability (n - 1)

theorem jester_win_prob_constant (n : ℕ+) : 
  jester_win_probability n = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_jester_win_prob_constant_l4007_400798


namespace NUMINAMATH_CALUDE_baker_usual_bread_sales_l4007_400736

/-- Represents the baker's sales and pricing information -/
structure BakerSales where
  usual_pastries : ℕ
  usual_bread : ℕ
  today_pastries : ℕ
  today_bread : ℕ
  pastry_price : ℕ
  bread_price : ℕ

/-- Calculates the difference between usual sales and today's sales -/
def sales_difference (s : BakerSales) : ℤ :=
  (s.usual_pastries * s.pastry_price + s.usual_bread * s.bread_price) -
  (s.today_pastries * s.pastry_price + s.today_bread * s.bread_price)

/-- Theorem stating that given the conditions, the baker usually sells 34 loaves of bread -/
theorem baker_usual_bread_sales :
  ∀ (s : BakerSales),
    s.usual_pastries = 20 ∧
    s.today_pastries = 14 ∧
    s.today_bread = 25 ∧
    s.pastry_price = 2 ∧
    s.bread_price = 4 ∧
    sales_difference s = 48 →
    s.usual_bread = 34 :=
by
  sorry


end NUMINAMATH_CALUDE_baker_usual_bread_sales_l4007_400736


namespace NUMINAMATH_CALUDE_all_propositions_true_l4007_400750

-- Define the original proposition
def original_proposition (x y : ℝ) : Prop :=
  x^2 + y^2 = 0 → x = 0 ∧ y = 0

-- Define the converse proposition
def converse_proposition (x y : ℝ) : Prop :=
  x = 0 ∧ y = 0 → x^2 + y^2 = 0

-- Define the inverse proposition
def inverse_proposition (x y : ℝ) : Prop :=
  x^2 + y^2 ≠ 0 → x ≠ 0 ∨ y ≠ 0

-- Define the contrapositive proposition
def contrapositive_proposition (x y : ℝ) : Prop :=
  x ≠ 0 ∨ y ≠ 0 → x^2 + y^2 ≠ 0

-- Theorem stating that all propositions are true
theorem all_propositions_true :
  ∀ x y : ℝ,
    original_proposition x y ∧
    converse_proposition x y ∧
    inverse_proposition x y ∧
    contrapositive_proposition x y :=
by
  sorry


end NUMINAMATH_CALUDE_all_propositions_true_l4007_400750


namespace NUMINAMATH_CALUDE_abc_max_value_l4007_400705

/-- Given positive reals a, b, c satisfying the constraint b(a^2 + 2) + c(a + 2) = 12,
    the maximum value of abc is 3. -/
theorem abc_max_value (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
    (h_constraint : b * (a^2 + 2) + c * (a + 2) = 12) :
  a * b * c ≤ 3 := by
sorry

end NUMINAMATH_CALUDE_abc_max_value_l4007_400705


namespace NUMINAMATH_CALUDE_sufficient_drivers_and_schedule_l4007_400715

-- Define the duration of trips and rest time
def one_way_trip_duration : ℕ := 160  -- in minutes
def round_trip_duration : ℕ := 320    -- in minutes
def min_rest_duration : ℕ := 60       -- in minutes

-- Define the schedule times (in minutes since midnight)
def driver_a_return : ℕ := 12 * 60 + 40
def driver_d_departure : ℕ := 13 * 60 + 5
def driver_b_return : ℕ := 16 * 60
def driver_a_second_departure : ℕ := 16 * 60 + 10
def driver_b_second_departure : ℕ := 17 * 60 + 30

-- Define the number of drivers
def num_drivers : ℕ := 4

-- Define the end time of the last trip
def last_trip_end : ℕ := 21 * 60 + 30

-- Theorem statement
theorem sufficient_drivers_and_schedule :
  (num_drivers = 4) ∧
  (driver_a_return + min_rest_duration ≤ driver_d_departure) ∧
  (driver_b_return + min_rest_duration ≤ driver_b_second_departure) ∧
  (driver_a_second_departure + round_trip_duration = last_trip_end) ∧
  (last_trip_end ≤ 24 * 60) → 
  (num_drivers ≥ 4) ∧ (last_trip_end = 21 * 60 + 30) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_drivers_and_schedule_l4007_400715


namespace NUMINAMATH_CALUDE_gcd_1729_1309_l4007_400713

theorem gcd_1729_1309 : Nat.gcd 1729 1309 = 7 := by
  sorry

end NUMINAMATH_CALUDE_gcd_1729_1309_l4007_400713


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l4007_400779

theorem sqrt_equation_solution :
  ∀ x : ℝ, x > 0 → (6 * Real.sqrt (4 + x) + 6 * Real.sqrt (4 - x) = 9 * Real.sqrt 2) → x = Real.sqrt 255 / 4 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l4007_400779


namespace NUMINAMATH_CALUDE_square_plot_area_l4007_400784

/-- Given a square plot with fencing cost of 58 Rs per foot and total fencing cost of 1160 Rs,
    the area of the plot is 25 square feet. -/
theorem square_plot_area (side_length : ℝ) (perimeter : ℝ) (area : ℝ) : 
  perimeter = 4 * side_length →
  58 * perimeter = 1160 →
  area = side_length ^ 2 →
  area = 25 := by
sorry

end NUMINAMATH_CALUDE_square_plot_area_l4007_400784


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_equals_two_l4007_400763

theorem sum_of_reciprocals_equals_two (a b c : ℝ) 
  (ha : a^3 - 2020*a + 1010 = 0)
  (hb : b^3 - 2020*b + 1010 = 0)
  (hc : c^3 - 2020*c + 1010 = 0)
  (hab : a ≠ b) (hbc : b ≠ c) (hca : c ≠ a) :
  1/a + 1/b + 1/c = 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_equals_two_l4007_400763


namespace NUMINAMATH_CALUDE_number_of_boys_in_class_l4007_400777

/-- Given the conditions of a class weight calculation, prove the number of boys in the class -/
theorem number_of_boys_in_class 
  (incorrect_avg : ℝ) 
  (misread_weight : ℝ) 
  (correct_weight : ℝ) 
  (correct_avg : ℝ) 
  (h1 : incorrect_avg = 58.4)
  (h2 : misread_weight = 56)
  (h3 : correct_weight = 60)
  (h4 : correct_avg = 58.6) :
  ∃ n : ℕ, n * incorrect_avg + (correct_weight - misread_weight) = n * correct_avg ∧ n = 20 :=
by sorry

end NUMINAMATH_CALUDE_number_of_boys_in_class_l4007_400777


namespace NUMINAMATH_CALUDE_cos_2theta_value_l4007_400703

theorem cos_2theta_value (θ : Real) 
  (h : Real.sin (2 * θ) - 4 * Real.sin (θ + π/3) * Real.sin (θ - π/6) = Real.sqrt 3 / 3) : 
  Real.cos (2 * θ) = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_cos_2theta_value_l4007_400703


namespace NUMINAMATH_CALUDE_square_land_perimeter_l4007_400754

theorem square_land_perimeter (a : ℝ) (h : 5 * a = 10 * (4 * Real.sqrt a) + 45) :
  4 * Real.sqrt a = 36 := by
  sorry

end NUMINAMATH_CALUDE_square_land_perimeter_l4007_400754


namespace NUMINAMATH_CALUDE_probability_multiple_2_3_or_5_l4007_400752

def is_multiple (n m : ℕ) : Prop := ∃ k, n = m * k

def count_multiples (max : ℕ) (divisor : ℕ) : ℕ :=
  (max / divisor : ℕ)

def count_multiples_of_2_3_or_5 (max : ℕ) : ℕ :=
  count_multiples max 2 + count_multiples max 3 + count_multiples max 5 -
  (count_multiples max 6 + count_multiples max 10 + count_multiples max 15) +
  count_multiples max 30

theorem probability_multiple_2_3_or_5 :
  (count_multiples_of_2_3_or_5 120 : ℚ) / 120 = 11 / 15 := by
  sorry

#eval count_multiples_of_2_3_or_5 120

end NUMINAMATH_CALUDE_probability_multiple_2_3_or_5_l4007_400752


namespace NUMINAMATH_CALUDE_union_of_sets_l4007_400756

theorem union_of_sets : 
  let A : Set ℕ := {2, 5, 6}
  let B : Set ℕ := {3, 5}
  A ∪ B = {2, 3, 5, 6} := by sorry

end NUMINAMATH_CALUDE_union_of_sets_l4007_400756


namespace NUMINAMATH_CALUDE_points_per_treasure_l4007_400794

theorem points_per_treasure (treasures_level1 treasures_level2 total_score : ℕ) 
  (h1 : treasures_level1 = 6)
  (h2 : treasures_level2 = 2)
  (h3 : total_score = 32) :
  total_score / (treasures_level1 + treasures_level2) = 4 := by
  sorry

end NUMINAMATH_CALUDE_points_per_treasure_l4007_400794


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l4007_400767

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^4 - x^3 + x^2 + 5 ≤ 0) ↔ (∃ x : ℝ, x^4 - x^3 + x^2 + 5 > 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l4007_400767


namespace NUMINAMATH_CALUDE_swimmer_speed_in_still_water_l4007_400731

/-- Represents the speed of a swimmer in still water and the speed of the stream. -/
structure SwimmerSpeed where
  swimmer : ℝ  -- Speed of the swimmer in still water (km/h)
  stream : ℝ   -- Speed of the stream (km/h)

/-- Calculates the effective speed when swimming downstream. -/
def downstreamSpeed (s : SwimmerSpeed) : ℝ := s.swimmer + s.stream

/-- Calculates the effective speed when swimming upstream. -/
def upstreamSpeed (s : SwimmerSpeed) : ℝ := s.swimmer - s.stream

/-- Theorem stating that given the conditions of the swimming problem, 
    the swimmer's speed in still water is 7 km/h. -/
theorem swimmer_speed_in_still_water : 
  ∀ s : SwimmerSpeed, 
    downstreamSpeed s = 36 / 6 → 
    upstreamSpeed s = 48 / 6 → 
    s.swimmer = 7 := by
  sorry

#check swimmer_speed_in_still_water

end NUMINAMATH_CALUDE_swimmer_speed_in_still_water_l4007_400731


namespace NUMINAMATH_CALUDE_product_of_sum_and_sum_of_cubes_l4007_400725

theorem product_of_sum_and_sum_of_cubes (x y : ℝ) 
  (sum_eq : x + y = 10) 
  (sum_cubes_eq : x^3 + y^3 = 370) : 
  x * y = 21 := by
sorry

end NUMINAMATH_CALUDE_product_of_sum_and_sum_of_cubes_l4007_400725


namespace NUMINAMATH_CALUDE_triangle_side_sum_l4007_400742

/-- Represents a triangle with side lengths a, b, and c, and angles A, B, and C. -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Checks if the angles of a triangle are in arithmetic progression -/
def anglesInArithmeticProgression (t : Triangle) : Prop :=
  ∃ d : ℝ, t.B = t.A + d ∧ t.C = t.A + 2*d

/-- Represents a value that can be expressed as p + √q + √r where p, q, r are integers -/
structure SpecialValue where
  p : ℤ
  q : ℤ
  r : ℤ

/-- The theorem to be proved -/
theorem triangle_side_sum (t : Triangle) (x₁ x₂ : SpecialValue) :
  t.a = 6 ∧ t.b = 8 ∧
  anglesInArithmeticProgression t ∧
  t.A = 30 * π / 180 ∧
  (t.c = Real.sqrt (x₁.q : ℝ) ∨ t.c = (x₂.p : ℝ) + Real.sqrt (x₂.q : ℝ)) →
  (x₁.p : ℝ) + Real.sqrt (x₁.q : ℝ) + Real.sqrt (x₁.r : ℝ) +
  (x₂.p : ℝ) + Real.sqrt (x₂.q : ℝ) + Real.sqrt (x₂.r : ℝ) =
  7 + Real.sqrt 36 + Real.sqrt 83 :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_sum_l4007_400742


namespace NUMINAMATH_CALUDE_least_positive_integer_multiple_of_53_l4007_400790

theorem least_positive_integer_multiple_of_53 :
  ∃ (x : ℕ+), 
    (∀ (y : ℕ+), y < x → ¬(53 ∣ (3 * y.val)^2 + 2 * 41 * (3 * y.val) + 41^2)) ∧
    (53 ∣ (3 * x.val)^2 + 2 * 41 * (3 * x.val) + 41^2) ∧
    x.val = 4 :=
by sorry

end NUMINAMATH_CALUDE_least_positive_integer_multiple_of_53_l4007_400790


namespace NUMINAMATH_CALUDE_highest_divisible_digit_l4007_400702

theorem highest_divisible_digit : 
  ∃ (a : ℕ), a ≤ 9 ∧ 
  (∀ (b : ℕ), b ≤ 9 → 365 * 100 * b + 16 ≡ 0 [MOD 8] → b ≤ a) ∧
  (365 * 100 * a + 16 ≡ 0 [MOD 8]) :=
by sorry

end NUMINAMATH_CALUDE_highest_divisible_digit_l4007_400702


namespace NUMINAMATH_CALUDE_vector_sum_zero_parallel_sufficient_not_necessary_l4007_400710

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

def parallel (a b : V) : Prop := ∃ (k : ℝ), a = k • b

theorem vector_sum_zero_parallel_sufficient_not_necessary :
  ∀ (a b : V), a ≠ 0 → b ≠ 0 →
  (a + b = 0 → parallel a b) ∧
  ¬(parallel a b → a + b = 0) :=
by sorry

end NUMINAMATH_CALUDE_vector_sum_zero_parallel_sufficient_not_necessary_l4007_400710


namespace NUMINAMATH_CALUDE_elevator_weight_problem_l4007_400706

/-- Given 10 people in an elevator with an average weight of 165 lbs, 
    prove that if an 11th person enters and increases the average weight to 170 lbs, 
    then the weight of the 11th person is 220 lbs. -/
theorem elevator_weight_problem (initial_people : ℕ) (initial_avg_weight : ℝ) 
  (new_avg_weight : ℝ) (new_person_weight : ℝ) :
  initial_people = 10 →
  initial_avg_weight = 165 →
  new_avg_weight = 170 →
  (initial_people * initial_avg_weight + new_person_weight) / (initial_people + 1) = new_avg_weight →
  new_person_weight = 220 :=
by sorry

end NUMINAMATH_CALUDE_elevator_weight_problem_l4007_400706


namespace NUMINAMATH_CALUDE_fraction_power_product_l4007_400718

theorem fraction_power_product :
  (3 / 5 : ℚ) ^ 4 * (2 / 9 : ℚ) = 162 / 5625 := by sorry

end NUMINAMATH_CALUDE_fraction_power_product_l4007_400718


namespace NUMINAMATH_CALUDE_painted_portion_is_five_eighths_additional_painting_needed_l4007_400783

-- Define the bridge as having a total length of 1
def bridge_length : ℝ := 1

-- Define the painted portion of the bridge
def painted_portion : ℝ → Prop := λ x => 
  -- The painted and unpainted portions sum to the total length
  x + (bridge_length - x) = bridge_length ∧
  -- If the painted portion increases by 30%, the unpainted portion decreases by 50%
  1.3 * x + 0.5 * (bridge_length - x) = bridge_length

-- Theorem: The painted portion is 5/8 of the bridge length
theorem painted_portion_is_five_eighths : 
  ∃ x : ℝ, painted_portion x ∧ x = 5/8 * bridge_length :=
sorry

-- Theorem: An additional 1/8 of the bridge length needs to be painted to have half the bridge painted
theorem additional_painting_needed : 
  ∃ x : ℝ, painted_portion x ∧ x + 1/8 * bridge_length = 1/2 * bridge_length :=
sorry

end NUMINAMATH_CALUDE_painted_portion_is_five_eighths_additional_painting_needed_l4007_400783


namespace NUMINAMATH_CALUDE_order_of_expressions_l4007_400797

theorem order_of_expressions : 
  let a : ℝ := (1/2)^(1/2)
  let b : ℝ := Real.log 2015 / Real.log 2014
  let c : ℝ := Real.log 2 / Real.log 4
  b > a ∧ a > c := by sorry

end NUMINAMATH_CALUDE_order_of_expressions_l4007_400797


namespace NUMINAMATH_CALUDE_f_triple_composition_equals_self_l4007_400739

def f (n : ℤ) : ℤ :=
  if n % 2 = 1 then n + 3 else n / 2

theorem f_triple_composition_equals_self (k : ℤ) :
  k % 2 = 1 → (f (f (f k)) = k ↔ k = 1) := by
  sorry

end NUMINAMATH_CALUDE_f_triple_composition_equals_self_l4007_400739


namespace NUMINAMATH_CALUDE_day_299_is_tuesday_l4007_400782

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Calculates the day of the week given a day number -/
def dayOfWeek (dayNumber : Nat) : DayOfWeek :=
  match dayNumber % 7 with
  | 0 => DayOfWeek.Sunday
  | 1 => DayOfWeek.Monday
  | 2 => DayOfWeek.Tuesday
  | 3 => DayOfWeek.Wednesday
  | 4 => DayOfWeek.Thursday
  | 5 => DayOfWeek.Friday
  | _ => DayOfWeek.Saturday

theorem day_299_is_tuesday (isLeapYear : Bool) :
  isLeapYear ∧ dayOfWeek 45 = DayOfWeek.Sunday →
  dayOfWeek 299 = DayOfWeek.Tuesday :=
by
  sorry

end NUMINAMATH_CALUDE_day_299_is_tuesday_l4007_400782


namespace NUMINAMATH_CALUDE_freshman_class_size_l4007_400720

theorem freshman_class_size :
  ∃! n : ℕ, n < 700 ∧
    n % 20 = 19 ∧
    n % 25 = 24 ∧
    n % 9 = 3 ∧
    n = 399 := by sorry

end NUMINAMATH_CALUDE_freshman_class_size_l4007_400720


namespace NUMINAMATH_CALUDE_orthocenter_of_triangle_l4007_400704

/-- The orthocenter of a triangle in 3D space. -/
def orthocenter (A B C : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ := sorry

/-- Theorem: The orthocenter of triangle ABC with given coordinates is (2.2375, 2.675, 4.515). -/
theorem orthocenter_of_triangle :
  let A : ℝ × ℝ × ℝ := (1.5, 2, 3.5)
  let B : ℝ × ℝ × ℝ := (4, 3.5, 1)
  let C : ℝ × ℝ × ℝ := (3, 5, 4.5)
  orthocenter A B C = (2.2375, 2.675, 4.515) := by sorry

end NUMINAMATH_CALUDE_orthocenter_of_triangle_l4007_400704


namespace NUMINAMATH_CALUDE_average_equals_median_l4007_400768

theorem average_equals_median (n : ℕ) (k : ℕ) (x : ℝ) : 
  n > 0 → 
  k > 0 → 
  x > 0 → 
  n = 14 → 
  (x * (k + 1) / 2)^2 = (2 * n)^2 → 
  x = n := by
sorry

end NUMINAMATH_CALUDE_average_equals_median_l4007_400768


namespace NUMINAMATH_CALUDE_car_catching_truck_l4007_400744

/-- A problem about a car catching up to a truck on a highway. -/
theorem car_catching_truck (truck_speed : ℝ) (head_start : ℝ) (catch_up_time : ℝ) :
  truck_speed = 45 →
  head_start = 1 →
  catch_up_time = 4 →
  let car_speed := (truck_speed * (catch_up_time + head_start)) / catch_up_time
  car_speed = 56.25 := by
sorry


end NUMINAMATH_CALUDE_car_catching_truck_l4007_400744


namespace NUMINAMATH_CALUDE_systematic_sampling_l4007_400760

/-- Systematic sampling problem -/
theorem systematic_sampling
  (population_size : ℕ)
  (sample_size : ℕ)
  (last_sample : ℕ)
  (h1 : population_size = 2000)
  (h2 : sample_size = 100)
  (h3 : last_sample = 1994)
  : ∃ (first_sample : ℕ), first_sample = 14 ∧
    last_sample = (sample_size - 1) * (population_size / sample_size) + first_sample :=
sorry

end NUMINAMATH_CALUDE_systematic_sampling_l4007_400760


namespace NUMINAMATH_CALUDE_hyperbola_foci_l4007_400743

/-- The hyperbola equation -/
def hyperbola_equation (x y : ℝ) : Prop := 4 * y^2 - 25 * x^2 = 100

/-- The foci coordinates -/
def foci : Set (ℝ × ℝ) := {(0, -Real.sqrt 29), (0, Real.sqrt 29)}

/-- Theorem: The foci of the given hyperbola are located at (0, ±√29) -/
theorem hyperbola_foci :
  ∀ (f : ℝ × ℝ), f ∈ foci ↔ 
    (∃ (x y : ℝ), hyperbola_equation x y ∧ 
      f = (x, y) ∧ 
      (∀ (x' y' : ℝ), hyperbola_equation x' y' → 
        (x - x')^2 + (y - y')^2 = ((Real.sqrt 29) + (Real.sqrt 29))^2 ∨
        (x - x')^2 + (y - y')^2 = ((Real.sqrt 29) - (Real.sqrt 29))^2)) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_foci_l4007_400743


namespace NUMINAMATH_CALUDE_rectangular_prism_volume_l4007_400786

/-- The volume of a rectangular prism with given edge lengths and space diagonal --/
theorem rectangular_prism_volume (AB AD AC1 : ℝ) :
  AB = 2 →
  AD = 2 →
  AC1 = 3 →
  ∃ (AA1 : ℝ), AA1 > 0 ∧ AB * AD * AA1 = 4 ∧ AC1^2 = AB^2 + AD^2 + AA1^2 :=
by sorry

end NUMINAMATH_CALUDE_rectangular_prism_volume_l4007_400786


namespace NUMINAMATH_CALUDE_milk_dilution_l4007_400740

/-- Represents the milk dilution problem -/
theorem milk_dilution (initial_capacity : ℝ) (removal_amount : ℝ) : 
  initial_capacity = 45 →
  removal_amount = 9 →
  let first_milk_remaining := initial_capacity - removal_amount
  let first_mixture_milk_ratio := first_milk_remaining / initial_capacity
  let second_milk_remaining := first_milk_remaining - (first_mixture_milk_ratio * removal_amount)
  second_milk_remaining = 28.8 := by
  sorry

end NUMINAMATH_CALUDE_milk_dilution_l4007_400740


namespace NUMINAMATH_CALUDE_last_digit_alternating_factorial_sum_2014_l4007_400709

def alternatingFactorialSum (n : ℕ) : ℤ :=
  (List.range n).foldl (fun acc i => acc + (if i % 2 = 0 then 1 else -1) * (i + 1).factorial) 0

theorem last_digit_alternating_factorial_sum_2014 :
  (alternatingFactorialSum 2014) % 10 = 1 := by sorry

end NUMINAMATH_CALUDE_last_digit_alternating_factorial_sum_2014_l4007_400709


namespace NUMINAMATH_CALUDE_max_value_sqrt_xy_1_minus_x_2y_l4007_400753

theorem max_value_sqrt_xy_1_minus_x_2y :
  ∀ x y : ℝ, x > 0 → y > 0 →
  Real.sqrt (x * y) * (1 - x - 2 * y) ≤ Real.sqrt 2 / 16 ∧
  (Real.sqrt (x * y) * (1 - x - 2 * y) = Real.sqrt 2 / 16 ↔ x = 1/4) :=
by sorry

end NUMINAMATH_CALUDE_max_value_sqrt_xy_1_minus_x_2y_l4007_400753


namespace NUMINAMATH_CALUDE_mile_to_yard_l4007_400719

-- Define the units
def mile : ℝ := 1
def furlong : ℝ := 1
def yard : ℝ := 1

-- Define the conversion factors
axiom mile_to_furlong : mile = 8 * furlong
axiom furlong_to_yard : furlong = 220 * yard

-- Theorem to prove
theorem mile_to_yard : mile = 1760 * yard := by
  sorry

end NUMINAMATH_CALUDE_mile_to_yard_l4007_400719


namespace NUMINAMATH_CALUDE_multiple_decimals_between_7_5_and_9_5_l4007_400771

theorem multiple_decimals_between_7_5_and_9_5 : 
  ∃ (x y : ℝ), 7.5 < x ∧ x < y ∧ y < 9.5 :=
sorry

end NUMINAMATH_CALUDE_multiple_decimals_between_7_5_and_9_5_l4007_400771


namespace NUMINAMATH_CALUDE_gcd_cube_plus_27_l4007_400780

theorem gcd_cube_plus_27 (n : ℕ) (h : n > 27) :
  Nat.gcd (n^3 + 3^3) (n + 3) = n + 3 := by
  sorry

end NUMINAMATH_CALUDE_gcd_cube_plus_27_l4007_400780


namespace NUMINAMATH_CALUDE_neg_one_is_square_sum_of_three_squares_zero_not_sum_of_three_nonzero_squares_l4007_400775

/-- A field K of characteristic p where p ≡ 1 (mod 4) -/
class CharacteristicP (K : Type) [Field K] where
  char_p : Nat
  char_p_prime : Prime char_p
  char_p_mod_4 : char_p % 4 = 1

variable {K : Type} [Field K] [CharacteristicP K]

/-- -1 is a square in K -/
theorem neg_one_is_square : ∃ x : K, x^2 = -1 := by sorry

/-- Any nonzero element in K can be written as the sum of three nonzero squares -/
theorem sum_of_three_squares (a : K) (ha : a ≠ 0) : 
  ∃ x y z : K, x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 ∧ x^2 + y^2 + z^2 = a := by sorry

/-- 0 cannot be written as the sum of three nonzero squares -/
theorem zero_not_sum_of_three_nonzero_squares :
  ¬∃ x y z : K, x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 ∧ x^2 + y^2 + z^2 = 0 := by sorry

end NUMINAMATH_CALUDE_neg_one_is_square_sum_of_three_squares_zero_not_sum_of_three_nonzero_squares_l4007_400775


namespace NUMINAMATH_CALUDE_person_height_from_shadow_l4007_400799

/-- Given a tree and a person under the same light conditions, calculate the person's height -/
theorem person_height_from_shadow (tree_height tree_shadow person_shadow : ℝ) 
  (h1 : tree_height = 60)
  (h2 : tree_shadow = 18)
  (h3 : person_shadow = 3) :
  (tree_height / tree_shadow) * person_shadow = 10 := by
  sorry

end NUMINAMATH_CALUDE_person_height_from_shadow_l4007_400799


namespace NUMINAMATH_CALUDE_consecutive_integers_sum_l4007_400781

theorem consecutive_integers_sum (n : ℤ) : 
  n * (n + 1) = 20412 → n + (n + 1) = 287 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_sum_l4007_400781


namespace NUMINAMATH_CALUDE_belt_and_road_population_scientific_notation_l4007_400766

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  significand : ℝ
  exponent : ℤ
  property : 1 ≤ significand ∧ significand < 10

/-- The population of the "Belt and Road" region -/
def beltAndRoadPopulation : ℕ := 4400000000

/-- The scientific notation representation of the population -/
def beltAndRoadScientific : ScientificNotation where
  significand := 4.4
  exponent := 9
  property := by sorry

/-- Theorem stating that the population is correctly represented in scientific notation -/
theorem belt_and_road_population_scientific_notation :
  (beltAndRoadPopulation : ℝ) = beltAndRoadScientific.significand * (10 : ℝ) ^ beltAndRoadScientific.exponent := by
  sorry

end NUMINAMATH_CALUDE_belt_and_road_population_scientific_notation_l4007_400766


namespace NUMINAMATH_CALUDE_triangle_side_value_l4007_400751

/-- A triangle with sides a, b, and c satisfies the triangle inequality -/
def is_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- The set of possible values for x -/
def possible_values : Set ℕ := {2, 4, 6, 8}

/-- The theorem statement -/
theorem triangle_side_value (x : ℕ) (hx : x ∈ possible_values) :
  is_triangle 2 x 6 ↔ x = 6 := by
  sorry


end NUMINAMATH_CALUDE_triangle_side_value_l4007_400751


namespace NUMINAMATH_CALUDE_volumes_not_equal_implies_cross_sections_not_equal_cross_sections_equal_not_implies_volumes_not_equal_l4007_400722

/-- Represents a geometric shape with height and volume -/
structure GeometricShape where
  height : ℝ
  volume : ℝ

/-- Represents the cross-sectional area of a shape at a given height -/
def crossSectionalArea (shape : GeometricShape) (h : ℝ) : ℝ :=
  sorry

/-- Cavalieri's Principle -/
axiom cavalieri_principle (A B : GeometricShape) :
  A.height = B.height →
  (∀ h, 0 ≤ h ∧ h ≤ A.height → crossSectionalArea A h = crossSectionalArea B h) →
  A.volume = B.volume

theorem volumes_not_equal_implies_cross_sections_not_equal
  (A B : GeometricShape) (h_height : A.height = B.height) :
  A.volume ≠ B.volume →
  ∃ h, 0 ≤ h ∧ h ≤ A.height ∧ crossSectionalArea A h ≠ crossSectionalArea B h :=
sorry

theorem cross_sections_equal_not_implies_volumes_not_equal
  (A B : GeometricShape) (h_height : A.height = B.height) :
  ¬(∀ h, 0 ≤ h ∧ h ≤ A.height → crossSectionalArea A h = crossSectionalArea B h →
    A.volume ≠ B.volume) :=
sorry

end NUMINAMATH_CALUDE_volumes_not_equal_implies_cross_sections_not_equal_cross_sections_equal_not_implies_volumes_not_equal_l4007_400722


namespace NUMINAMATH_CALUDE_copy_pages_for_ten_dollars_l4007_400717

/-- The number of pages that can be copied for a given amount of money, 
    given the cost of copying 5 pages --/
def pages_copied (cost_5_pages : ℚ) (amount : ℚ) : ℚ :=
  (amount / cost_5_pages) * 5

/-- Theorem stating that given the cost of 10 cents for 5 pages, 
    the number of pages that can be copied for $10 is 500 --/
theorem copy_pages_for_ten_dollars :
  pages_copied (10 / 100) (10 : ℚ) = 500 := by
  sorry

#eval pages_copied (10 / 100) 10

end NUMINAMATH_CALUDE_copy_pages_for_ten_dollars_l4007_400717


namespace NUMINAMATH_CALUDE_vector_difference_magnitude_l4007_400795

/-- Given two 2D vectors a and b, prove that the magnitude of their difference is 5. -/
theorem vector_difference_magnitude (a b : ℝ × ℝ) : 
  a = (2, 1) → b = (-2, 4) → ‖a - b‖ = 5 := by sorry

end NUMINAMATH_CALUDE_vector_difference_magnitude_l4007_400795


namespace NUMINAMATH_CALUDE_xiaopang_had_32_books_l4007_400791

/-- The number of books Xiaopang originally had -/
def xiaopang_books : ℕ := 32

/-- The number of books Xiaoya originally had -/
def xiaoya_books : ℕ := 16

/-- Theorem stating that Xiaopang originally had 32 books -/
theorem xiaopang_had_32_books :
  (xiaopang_books - 8 = xiaoya_books + 8) ∧
  (xiaopang_books + 4 = 3 * (xiaoya_books - 4)) →
  xiaopang_books = 32 := by
  sorry

end NUMINAMATH_CALUDE_xiaopang_had_32_books_l4007_400791


namespace NUMINAMATH_CALUDE_school_wall_stars_l4007_400727

theorem school_wall_stars (num_students : ℕ) (stars_per_student : ℕ) (total_stars : ℕ) :
  num_students = 210 →
  stars_per_student = 6 →
  total_stars = num_students * stars_per_student →
  total_stars = 1260 :=
by sorry

end NUMINAMATH_CALUDE_school_wall_stars_l4007_400727


namespace NUMINAMATH_CALUDE_polynomial_sum_of_coefficients_l4007_400729

theorem polynomial_sum_of_coefficients 
  (a b c d : ℝ) 
  (g : ℂ → ℂ) 
  (h₁ : ∀ x, g x = x^4 + a*x^3 + b*x^2 + c*x + d) 
  (h₂ : g (-3*I) = 0) 
  (h₃ : g (1 + I) = 0) : 
  a + b + c + d = 9 := by
sorry

end NUMINAMATH_CALUDE_polynomial_sum_of_coefficients_l4007_400729


namespace NUMINAMATH_CALUDE_least_seven_digit_binary_l4007_400759

theorem least_seven_digit_binary : ∀ n : ℕ, 
  (n < 64 → (Nat.log2 n).succ < 7) ∧ 
  ((Nat.log2 64).succ = 7) :=
sorry

end NUMINAMATH_CALUDE_least_seven_digit_binary_l4007_400759


namespace NUMINAMATH_CALUDE_experiment_sequences_l4007_400788

/-- The number of procedures in the experiment -/
def num_procedures : ℕ := 6

/-- The number of possible positions for procedure A -/
def a_positions : ℕ := 2

/-- The number of procedures excluding A -/
def remaining_procedures : ℕ := num_procedures - 1

/-- The number of arrangements of B and C -/
def bc_arrangements : ℕ := 2

theorem experiment_sequences :
  (a_positions * remaining_procedures.factorial * bc_arrangements) = 96 := by
  sorry

end NUMINAMATH_CALUDE_experiment_sequences_l4007_400788


namespace NUMINAMATH_CALUDE_triangle_perimeter_in_square_l4007_400774

/-- Given a square with side length 70√2 cm, divided into four congruent 45-45-90 triangles
    by its diagonals, the perimeter of one of these triangles is 140√2 + 140 cm. -/
theorem triangle_perimeter_in_square (side_length : ℝ) (h : side_length = 70 * Real.sqrt 2) :
  let diagonal := side_length * Real.sqrt 2
  let triangle_perimeter := 2 * side_length + diagonal
  triangle_perimeter = 140 * Real.sqrt 2 + 140 := by
  sorry

end NUMINAMATH_CALUDE_triangle_perimeter_in_square_l4007_400774


namespace NUMINAMATH_CALUDE_beef_weight_loss_percentage_l4007_400747

/-- Calculates the percentage of weight lost during processing of a side of beef. -/
theorem beef_weight_loss_percentage 
  (initial_weight : ℝ) 
  (processed_weight : ℝ) 
  (h1 : initial_weight = 892.31)
  (h2 : processed_weight = 580) : 
  ∃ (percentage : ℝ), abs (percentage - 34.99) < 0.01 ∧ 
  percentage = (initial_weight - processed_weight) / initial_weight * 100 :=
sorry

end NUMINAMATH_CALUDE_beef_weight_loss_percentage_l4007_400747


namespace NUMINAMATH_CALUDE_cube_root_equation_solution_l4007_400714

theorem cube_root_equation_solution : 
  ∃! x : ℝ, (7 - x / 3) ^ (1/3 : ℝ) = 5 ∧ x = -354 := by sorry

end NUMINAMATH_CALUDE_cube_root_equation_solution_l4007_400714


namespace NUMINAMATH_CALUDE_sqrt_three_x_minus_two_lt_x_l4007_400773

theorem sqrt_three_x_minus_two_lt_x (x : ℝ) : 
  Real.sqrt 3 * x - 2 < x ↔ x < Real.sqrt 3 + 1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_three_x_minus_two_lt_x_l4007_400773


namespace NUMINAMATH_CALUDE_rectangular_field_length_l4007_400749

theorem rectangular_field_length
  (area : ℝ)
  (length_increase : ℝ)
  (area_increase : ℝ)
  (h1 : area = 144)
  (h2 : length_increase = 6)
  (h3 : area_increase = 54)
  (h4 : ∀ l w, l * w = area → (l + length_increase) * w = area + area_increase) :
  ∃ l w, l * w = area ∧ l = 16 :=
sorry

end NUMINAMATH_CALUDE_rectangular_field_length_l4007_400749


namespace NUMINAMATH_CALUDE_correct_number_of_seasons_l4007_400712

/-- Represents the number of seasons in a TV show. -/
def num_seasons : ℕ := 5

/-- Cost per episode for the first season. -/
def first_season_cost : ℕ := 100000

/-- Cost per episode for other seasons. -/
def other_season_cost : ℕ := 200000

/-- Number of episodes in the first season. -/
def first_season_episodes : ℕ := 12

/-- Number of episodes in middle seasons. -/
def middle_season_episodes : ℕ := 18

/-- Number of episodes in the last season. -/
def last_season_episodes : ℕ := 24

/-- Total cost of producing all episodes. -/
def total_cost : ℕ := 16800000

/-- Theorem stating that the number of seasons is correct given the conditions. -/
theorem correct_number_of_seasons :
  (first_season_cost * first_season_episodes) +
  ((num_seasons - 2) * other_season_cost * middle_season_episodes) +
  (other_season_cost * last_season_episodes) = total_cost :=
sorry

end NUMINAMATH_CALUDE_correct_number_of_seasons_l4007_400712


namespace NUMINAMATH_CALUDE_calculate_interest_rate_loan_interest_rate_proof_l4007_400785

/-- Calculates the rate of interest for a loan with simple interest -/
theorem calculate_interest_rate (principal : ℝ) (interest_paid : ℝ) : ℝ :=
  let rate_squared := (100 * interest_paid) / (principal)
  Real.sqrt rate_squared

/-- Proves that the rate of interest for the given loan conditions is approximately 8.888% -/
theorem loan_interest_rate_proof 
  (principal : ℝ) 
  (interest_paid : ℝ) 
  (h1 : principal = 800) 
  (h2 : interest_paid = 632) : 
  ∃ (ε : ℝ), ε > 0 ∧ |calculate_interest_rate principal interest_paid - 8.888| < ε :=
sorry

end NUMINAMATH_CALUDE_calculate_interest_rate_loan_interest_rate_proof_l4007_400785


namespace NUMINAMATH_CALUDE_polynomial_subtraction_l4007_400772

/-- Given two polynomials in a and b, prove that their difference is -a^2*b -/
theorem polynomial_subtraction (a b : ℝ) :
  (3 * a^2 * b - 6 * a * b^2) - (2 * a^2 * b - 3 * a * b^2) = -a^2 * b := by
  sorry

end NUMINAMATH_CALUDE_polynomial_subtraction_l4007_400772


namespace NUMINAMATH_CALUDE_binary_10010_is_18_l4007_400787

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

theorem binary_10010_is_18 :
  binary_to_decimal [false, true, false, false, true] = 18 := by
  sorry

end NUMINAMATH_CALUDE_binary_10010_is_18_l4007_400787


namespace NUMINAMATH_CALUDE_special_permutation_exists_l4007_400793

/-- A permutation of numbers from 1 to 2^n satisfying the special property -/
def SpecialPermutation (n : ℕ) : List ℕ :=
  sorry

/-- Predicate to check if a list satisfies the special property -/
def SatisfiesProperty (lst : List ℕ) : Prop :=
  ∀ i j, i < j → i < lst.length → j < lst.length →
    ∀ k, i < k ∧ k < j →
      (lst.get ⟨i, sorry⟩ + lst.get ⟨j, sorry⟩) / 2 ≠ lst.get ⟨k, sorry⟩

/-- Theorem stating that for any n, there exists a permutation of numbers
    from 1 to 2^n satisfying the special property -/
theorem special_permutation_exists (n : ℕ) :
  ∃ (perm : List ℕ), perm.length = 2^n ∧
    (∀ i, i ∈ perm ↔ 1 ≤ i ∧ i ≤ 2^n) ∧
    SatisfiesProperty perm :=
  sorry

end NUMINAMATH_CALUDE_special_permutation_exists_l4007_400793


namespace NUMINAMATH_CALUDE_floor_times_self_eq_100_l4007_400726

theorem floor_times_self_eq_100 :
  ∃ (x : ℝ), x > 0 ∧ (⌊x⌋ : ℝ) * x = 100 → x = 10 := by
  sorry

end NUMINAMATH_CALUDE_floor_times_self_eq_100_l4007_400726


namespace NUMINAMATH_CALUDE_tims_soda_cans_l4007_400708

theorem tims_soda_cans (S : ℕ) : 
  (S - 10) + (S - 10) / 2 + 10 = 34 → S = 26 :=
by sorry

end NUMINAMATH_CALUDE_tims_soda_cans_l4007_400708


namespace NUMINAMATH_CALUDE_tethered_unicorn_sum_l4007_400733

/-- Represents the configuration of a unicorn tethered to a cylindrical tower. -/
structure TetheredUnicorn where
  towerRadius : ℝ
  ropeLength : ℝ
  unicornHeight : ℝ
  distanceFromTower : ℝ
  ropeTouchLength : ℝ
  a : ℕ
  b : ℕ
  c : ℕ

/-- The theorem stating the sum of a, b, and c for the given configuration. -/
theorem tethered_unicorn_sum (u : TetheredUnicorn)
  (h1 : u.towerRadius = 10)
  (h2 : u.ropeLength = 30)
  (h3 : u.unicornHeight = 6)
  (h4 : u.distanceFromTower = 6)
  (h5 : u.ropeTouchLength = (u.a - Real.sqrt u.b) / u.c)
  (h6 : Nat.Prime u.c) :
  u.a + u.b + u.c = 940 := by
  sorry

end NUMINAMATH_CALUDE_tethered_unicorn_sum_l4007_400733


namespace NUMINAMATH_CALUDE_cat_finishes_food_on_saturday_l4007_400745

/-- Represents the days of the week -/
inductive Day : Type
| monday | tuesday | wednesday | thursday | friday | saturday | sunday

/-- The amount of food eaten by the cat per day -/
def daily_consumption : ℚ := 1/4 + 1/6

/-- The number of cans of food at the start -/
def initial_cans : ℕ := 6

/-- Calculates the number of cans eaten after a given number of days -/
def cans_eaten (days : ℕ) : ℚ := daily_consumption * days

/-- Determines if all food is eaten on a given day -/
def is_food_finished (d : Day) : Prop :=
  let days : ℕ := 
    match d with
    | Day.monday => 1
    | Day.tuesday => 2
    | Day.wednesday => 3
    | Day.thursday => 4
    | Day.friday => 5
    | Day.saturday => 6
    | Day.sunday => 7
  cans_eaten days > initial_cans ∧ cans_eaten (days - 1) ≤ initial_cans

theorem cat_finishes_food_on_saturday : 
  is_food_finished Day.saturday := by sorry

end NUMINAMATH_CALUDE_cat_finishes_food_on_saturday_l4007_400745


namespace NUMINAMATH_CALUDE_book_pages_calculation_l4007_400748

theorem book_pages_calculation (pages_read : ℕ) (pages_unread : ℕ) (additional_pages : ℕ) :
  pages_read + pages_unread > 0 →
  pages_read = pages_unread / 3 →
  additional_pages = 48 →
  (pages_read + additional_pages : ℚ) / (pages_read + pages_unread + additional_pages) = 2/5 →
  pages_read + pages_unread = 320 := by
sorry

end NUMINAMATH_CALUDE_book_pages_calculation_l4007_400748


namespace NUMINAMATH_CALUDE_contrapositive_even_sum_l4007_400732

theorem contrapositive_even_sum (x y : ℤ) :
  (¬(Even (x + y)) → ¬(Even x ∧ Even y)) ↔
  (∀ x y : ℤ, Even x ∧ Even y → Even (x + y)) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_even_sum_l4007_400732


namespace NUMINAMATH_CALUDE_pages_read_today_l4007_400776

theorem pages_read_today (pages_yesterday pages_total : ℕ) 
  (h1 : pages_yesterday = 21)
  (h2 : pages_total = 38) :
  pages_total - pages_yesterday = 17 := by
sorry

end NUMINAMATH_CALUDE_pages_read_today_l4007_400776


namespace NUMINAMATH_CALUDE_all_points_enclosable_l4007_400700

/-- A type representing a point in a plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A function that checks if three points can be enclosed in a circle of radius 1 -/
def enclosableInUnitCircle (p q r : Point) : Prop :=
  ∃ (center : Point), (center.x - p.x)^2 + (center.y - p.y)^2 ≤ 1 ∧
                      (center.x - q.x)^2 + (center.y - q.y)^2 ≤ 1 ∧
                      (center.x - r.x)^2 + (center.y - r.y)^2 ≤ 1

/-- The main theorem -/
theorem all_points_enclosable (n : ℕ) (points : Fin n → Point)
  (h : ∀ (i j k : Fin n), i ≠ j → j ≠ k → i ≠ k → enclosableInUnitCircle (points i) (points j) (points k)) :
  ∃ (center : Point), ∀ (i : Fin n), (center.x - (points i).x)^2 + (center.y - (points i).y)^2 ≤ 1 :=
sorry

end NUMINAMATH_CALUDE_all_points_enclosable_l4007_400700


namespace NUMINAMATH_CALUDE_train_length_train_length_approximation_l4007_400716

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed_kmh : ℝ) (time_sec : ℝ) : ℝ :=
  let speed_ms := speed_kmh * 1000 / 3600
  speed_ms * time_sec

/-- Prove that a train traveling at 50 km/h and crossing a pole in 18 seconds has a length of approximately 250 meters -/
theorem train_length_approximation :
  ∃ (ε : ℝ), ε > 0 ∧ |train_length 50 18 - 250| < ε :=
sorry

end NUMINAMATH_CALUDE_train_length_train_length_approximation_l4007_400716


namespace NUMINAMATH_CALUDE_curve_C_symmetry_l4007_400758

/-- The curve C in the Cartesian coordinate system -/
def C : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ((p.1 - 1)^2 + p.2^2) * ((p.1 + 1)^2 + p.2^2) = 4}

/-- A point is symmetric about the x-axis -/
def symmetric_x (p : ℝ × ℝ) : Prop := (p.1, -p.2) ∈ C ↔ p ∈ C

/-- A point is symmetric about the y-axis -/
def symmetric_y (p : ℝ × ℝ) : Prop := (-p.1, p.2) ∈ C ↔ p ∈ C

/-- A point is symmetric about the origin -/
def symmetric_origin (p : ℝ × ℝ) : Prop := (-p.1, -p.2) ∈ C ↔ p ∈ C

theorem curve_C_symmetry :
  (∀ p ∈ C, symmetric_x p ∧ symmetric_y p) ∧
  (∀ p ∈ C, symmetric_origin p) := by sorry

end NUMINAMATH_CALUDE_curve_C_symmetry_l4007_400758


namespace NUMINAMATH_CALUDE_one_solution_r_product_l4007_400792

theorem one_solution_r_product (r : ℝ) : 
  (∃! x : ℝ, (1 / (2 * x) = (r - x) / 9)) → 
  (∃ r₁ r₂ : ℝ, r = r₁ ∨ r = r₂) ∧ (r₁ * r₂ = -18) :=
sorry

end NUMINAMATH_CALUDE_one_solution_r_product_l4007_400792


namespace NUMINAMATH_CALUDE_smaller_cube_side_length_l4007_400755

/-- The side length of a smaller cube inscribed between a sphere and one face of a larger cube inscribed in the sphere. -/
theorem smaller_cube_side_length (R : ℝ) : 
  R = Real.sqrt 3 →  -- Radius of the sphere
  ∃ (x : ℝ), 
    x > 0 ∧  -- Side length of smaller cube is positive
    x < 2 ∧  -- Side length of smaller cube is less than that of larger cube
    (1 + x + x * Real.sqrt 2 / 2)^2 = 3 ∧  -- Equation derived from geometric relationships
    x = 2/3 :=
by sorry

end NUMINAMATH_CALUDE_smaller_cube_side_length_l4007_400755


namespace NUMINAMATH_CALUDE_min_value_theorem_l4007_400721

theorem min_value_theorem (x y : ℝ) (h1 : x > y) (h2 : y > 0) (h3 : x + y ≤ 2) :
  ∃ (min_val : ℝ), min_val = (3 + 2 * Real.sqrt 2) / 4 ∧
  ∀ (z : ℝ), z = 2 / (x + 3 * y) + 1 / (x - y) → z ≥ min_val := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l4007_400721


namespace NUMINAMATH_CALUDE_triangle_angle_calculation_l4007_400701

theorem triangle_angle_calculation (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧
  A + B + C = π ∧
  a = Real.sqrt 2 ∧
  b = 2 ∧
  Real.sin B + Real.cos B = Real.sqrt 2 →
  A = π / 6 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_calculation_l4007_400701


namespace NUMINAMATH_CALUDE_simplify_nested_roots_l4007_400789

theorem simplify_nested_roots (a : ℝ) : 
  (((a^9)^(1/6))^(1/3))^4 * (((a^9)^(1/3))^(1/6))^4 = a^4 := by sorry

end NUMINAMATH_CALUDE_simplify_nested_roots_l4007_400789


namespace NUMINAMATH_CALUDE_right_triangle_perimeter_l4007_400724

theorem right_triangle_perimeter (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  a * b / 2 = 180 →
  a = 30 →
  c^2 = a^2 + b^2 →
  a + b + c = 42 + 2 * Real.sqrt 261 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_perimeter_l4007_400724


namespace NUMINAMATH_CALUDE_negative_number_identification_l4007_400761

theorem negative_number_identification :
  (0 ≥ 0) ∧ ((1/2 : ℝ) > 0) ∧ (-(-5) > 0) ∧ (-Real.sqrt 5 < 0) := by
  sorry

end NUMINAMATH_CALUDE_negative_number_identification_l4007_400761


namespace NUMINAMATH_CALUDE_pond_length_l4007_400734

/-- The length of a rectangular pond given its width, depth, and volume. -/
theorem pond_length (width : ℝ) (depth : ℝ) (volume : ℝ) 
  (h_width : width = 12) 
  (h_depth : depth = 5) 
  (h_volume : volume = 1200) : 
  volume / (width * depth) = 20 := by
  sorry

end NUMINAMATH_CALUDE_pond_length_l4007_400734


namespace NUMINAMATH_CALUDE_min_sum_of_product_2310_l4007_400757

theorem min_sum_of_product_2310 (a b c : ℕ+) (h : a * b * c = 2310) :
  (∀ x y z : ℕ+, x * y * z = 2310 → a + b + c ≤ x + y + z) ∧ a + b + c = 52 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_of_product_2310_l4007_400757


namespace NUMINAMATH_CALUDE_division_remainder_l4007_400741

theorem division_remainder : Int.mod 1234567 256 = 503 := by
  sorry

end NUMINAMATH_CALUDE_division_remainder_l4007_400741


namespace NUMINAMATH_CALUDE_x_positive_sufficient_not_necessary_for_x_nonzero_l4007_400711

theorem x_positive_sufficient_not_necessary_for_x_nonzero :
  (∃ x : ℝ, x ≠ 0 ∧ ¬(x > 0)) ∧
  (∀ x : ℝ, x > 0 → x ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_x_positive_sufficient_not_necessary_for_x_nonzero_l4007_400711


namespace NUMINAMATH_CALUDE_sqrt_expression_simplification_l4007_400728

theorem sqrt_expression_simplification :
  let x := Real.sqrt 97
  let y := Real.sqrt 486
  let z := Real.sqrt 125
  let w := Real.sqrt 54
  let v := Real.sqrt 49
  (x + y + z) / (w + v) = (x + 9 * Real.sqrt 6 + 5 * Real.sqrt 5) / (3 * Real.sqrt 6 + 7) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_expression_simplification_l4007_400728


namespace NUMINAMATH_CALUDE_least_number_to_add_l4007_400746

theorem least_number_to_add (x : ℕ) : 
  (∀ y : ℕ, y < x → ¬((28523 + y) % 3 = 0 ∧ (28523 + y) % 5 = 0 ∧ (28523 + y) % 7 = 0 ∧ (28523 + y) % 8 = 0)) ∧
  ((28523 + x) % 3 = 0 ∧ (28523 + x) % 5 = 0 ∧ (28523 + x) % 7 = 0 ∧ (28523 + x) % 8 = 0) →
  x = 137 := by
sorry

end NUMINAMATH_CALUDE_least_number_to_add_l4007_400746


namespace NUMINAMATH_CALUDE_factorization_a_squared_minus_3a_l4007_400769

theorem factorization_a_squared_minus_3a (a : ℝ) : a^2 - 3*a = a*(a - 3) := by
  sorry

end NUMINAMATH_CALUDE_factorization_a_squared_minus_3a_l4007_400769


namespace NUMINAMATH_CALUDE_binomial_identities_l4007_400770

theorem binomial_identities (n k : ℕ) : 
  k * (n.choose k) = n * ((n - 1).choose (k - 1)) ∧ 
  (Finset.range (n + 1)).sum (λ k => k * (n.choose k)) = n * 2^(n - 1) := by
  sorry

end NUMINAMATH_CALUDE_binomial_identities_l4007_400770


namespace NUMINAMATH_CALUDE_quadratic_intersection_theorem_l4007_400707

-- Define the quadratic function
def f (m : ℝ) (x : ℝ) : ℝ := x^2 + 4*x + m

-- Define the condition for three intersection points
def has_three_intersections (m : ℝ) : Prop :=
  m ≠ 0 ∧ ∃ x₁ x₂ x₃, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧
  ((f m x₁ = 0 ∧ x₁ ≠ 0) ∨ (x₁ = 0 ∧ f m 0 = m)) ∧
  ((f m x₂ = 0 ∧ x₂ ≠ 0) ∨ (x₂ = 0 ∧ f m 0 = m)) ∧
  ((f m x₃ = 0 ∧ x₃ ≠ 0) ∨ (x₃ = 0 ∧ f m 0 = m))

-- Define the circle passing through the three intersection points
def circle_through_intersections (m : ℝ) (x y : ℝ) : Prop :=
  x^2 + y^2 + 4*x - (m + 1)*y + m = 0

-- The main theorem
theorem quadratic_intersection_theorem (m : ℝ) :
  has_three_intersections m →
  (m < 4 ∧
   circle_through_intersections m 0 1 ∧
   circle_through_intersections m (-4) 1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_intersection_theorem_l4007_400707


namespace NUMINAMATH_CALUDE_modulus_one_plus_i_to_sixth_l4007_400796

theorem modulus_one_plus_i_to_sixth (i : ℂ) : i * i = -1 → Complex.abs ((1 + i)^6) = 8 := by
  sorry

end NUMINAMATH_CALUDE_modulus_one_plus_i_to_sixth_l4007_400796


namespace NUMINAMATH_CALUDE_rectangular_playground_area_l4007_400765

theorem rectangular_playground_area :
  ∀ (length width : ℝ),
  length > 0 ∧ width > 0 →
  length + width = 36 →
  length = 3 * width →
  length * width = 243 :=
by sorry

end NUMINAMATH_CALUDE_rectangular_playground_area_l4007_400765


namespace NUMINAMATH_CALUDE_income_problem_l4007_400737

theorem income_problem (m n o : ℕ) : 
  (m + n) / 2 = 5050 →
  (n + o) / 2 = 6250 →
  (m + o) / 2 = 5200 →
  m = 4000 := by
sorry

end NUMINAMATH_CALUDE_income_problem_l4007_400737


namespace NUMINAMATH_CALUDE_calligraphy_supplies_problem_l4007_400735

/-- Represents the unit price of a brush in yuan -/
def brush_price : ℝ := 6

/-- Represents the unit price of rice paper in yuan -/
def paper_price : ℝ := 0.4

/-- Represents the maximum number of brushes that can be purchased -/
def max_brushes : ℕ := 50

/-- Theorem stating the solution to the calligraphy supplies problem -/
theorem calligraphy_supplies_problem :
  /- Given conditions -/
  (40 * brush_price + 100 * paper_price = 280) ∧
  (30 * brush_price + 200 * paper_price = 260) ∧
  (∀ m : ℕ, m ≤ 200 → 
    m * brush_price + (200 - m) * paper_price ≤ 360 → 
    m ≤ max_brushes) →
  /- Conclusion -/
  brush_price = 6 ∧ paper_price = 0.4 ∧ max_brushes = 50 :=
by sorry

end NUMINAMATH_CALUDE_calligraphy_supplies_problem_l4007_400735


namespace NUMINAMATH_CALUDE_sum_of_common_ratios_is_three_l4007_400778

/-- Given two nonconstant geometric sequences with different common ratios,
    prove that the sum of their common ratios is 3 under certain conditions. -/
theorem sum_of_common_ratios_is_three
  (k : ℝ) (p r : ℝ) (hp : p ≠ 1) (hr : r ≠ 1) (hpr : p ≠ r) (hk : k ≠ 0)
  (h : k * p^2 - k * r^2 = 3 * (k * p - k * r)) :
  p + r = 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_common_ratios_is_three_l4007_400778


namespace NUMINAMATH_CALUDE_tens_digit_of_6_pow_2047_l4007_400764

/-- The cycle of the last two digits of powers of 6 -/
def last_two_digits_cycle : List ℕ := [16, 96, 76, 56]

/-- The length of the cycle -/
def cycle_length : ℕ := 4

theorem tens_digit_of_6_pow_2047 (h : last_two_digits_cycle = [16, 96, 76, 56]) :
  (6^2047 / 10) % 10 = 7 := by
  sorry

end NUMINAMATH_CALUDE_tens_digit_of_6_pow_2047_l4007_400764
