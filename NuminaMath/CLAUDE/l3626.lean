import Mathlib

namespace NUMINAMATH_CALUDE_elliot_book_pages_l3626_362608

/-- The number of pages in Elliot's book -/
def total_pages : ℕ := 381

/-- The number of pages Elliot has already read -/
def pages_read : ℕ := 149

/-- The number of pages Elliot reads per day -/
def pages_per_day : ℕ := 20

/-- The number of days Elliot reads -/
def days_reading : ℕ := 7

/-- The number of pages left to be read after reading for 7 days -/
def pages_left : ℕ := 92

theorem elliot_book_pages : 
  total_pages = pages_read + (pages_per_day * days_reading) + pages_left :=
by sorry

end NUMINAMATH_CALUDE_elliot_book_pages_l3626_362608


namespace NUMINAMATH_CALUDE_arthur_muffins_l3626_362693

theorem arthur_muffins (total : ℕ) (more : ℕ) (initial : ℕ) 
    (h1 : total = 83)
    (h2 : more = 48)
    (h3 : total = initial + more) :
  initial = 35 := by
  sorry

end NUMINAMATH_CALUDE_arthur_muffins_l3626_362693


namespace NUMINAMATH_CALUDE_reference_city_hospitals_l3626_362631

/-- The number of hospitals in the reference city -/
def reference_hospitals : ℕ := sorry

/-- The number of stores in the reference city -/
def reference_stores : ℕ := 2000

/-- The number of schools in the reference city -/
def reference_schools : ℕ := 200

/-- The number of police stations in the reference city -/
def reference_police : ℕ := 20

/-- The total number of buildings in the new city -/
def new_city_total : ℕ := 2175

theorem reference_city_hospitals :
  reference_stores / 2 + 2 * reference_hospitals + (reference_schools - 50) + (reference_police + 5) = new_city_total →
  reference_hospitals = 500 := by
  sorry

end NUMINAMATH_CALUDE_reference_city_hospitals_l3626_362631


namespace NUMINAMATH_CALUDE_characterization_of_complete_sets_l3626_362699

def is_complete (A : Set ℕ) : Prop :=
  ∀ a b : ℕ, (a + b) ∈ A → (a * b) ∈ A

def complete_sets : Set (Set ℕ) :=
  {{1}, {1, 2}, {1, 2, 3}, {1, 2, 3, 4}, Set.univ}

theorem characterization_of_complete_sets :
  ∀ A : Set ℕ, A.Nonempty → (is_complete A ↔ A ∈ complete_sets) := by
  sorry

end NUMINAMATH_CALUDE_characterization_of_complete_sets_l3626_362699


namespace NUMINAMATH_CALUDE_circle_y_axis_intersection_sum_l3626_362665

theorem circle_y_axis_intersection_sum (h k r : ℝ) : 
  h = 5 → k = -3 → r = 13 →
  let y₁ := k + (r^2 - h^2).sqrt
  let y₂ := k - (r^2 - h^2).sqrt
  y₁ + y₂ = -6 :=
by sorry

end NUMINAMATH_CALUDE_circle_y_axis_intersection_sum_l3626_362665


namespace NUMINAMATH_CALUDE_perfect_square_characterization_l3626_362687

theorem perfect_square_characterization (A : ℕ+) :
  (∃ k : ℕ, A = k^2) ↔
  (∀ n : ℕ+, ∃ k : ℕ+, k ≤ n ∧ n ∣ ((A + k)^2 - A)) := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_characterization_l3626_362687


namespace NUMINAMATH_CALUDE_jonathan_weekly_deficit_l3626_362675

/-- Represents the days of the week -/
inductive Day
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday
| Saturday
| Sunday

/-- Jonathan's daily calorie intake -/
def calorie_intake (d : Day) : ℕ :=
  match d with
  | Day.Monday => 2500
  | Day.Tuesday => 2600
  | Day.Wednesday => 2400
  | Day.Thursday => 2700
  | Day.Friday => 2300
  | Day.Saturday => 3500
  | Day.Sunday => 2400

/-- Jonathan's daily calorie expenditure -/
def calorie_expenditure (d : Day) : ℕ :=
  match d with
  | Day.Monday => 3000
  | Day.Tuesday => 3200
  | Day.Wednesday => 2900
  | Day.Thursday => 3100
  | Day.Friday => 2800
  | Day.Saturday => 3000
  | Day.Sunday => 2700

/-- Calculate the daily caloric deficit -/
def daily_deficit (d : Day) : ℤ :=
  (calorie_expenditure d : ℤ) - (calorie_intake d : ℤ)

/-- The weekly caloric deficit -/
def weekly_deficit : ℤ :=
  (daily_deficit Day.Monday) +
  (daily_deficit Day.Tuesday) +
  (daily_deficit Day.Wednesday) +
  (daily_deficit Day.Thursday) +
  (daily_deficit Day.Friday) +
  (daily_deficit Day.Saturday) +
  (daily_deficit Day.Sunday)

/-- Theorem: Jonathan's weekly caloric deficit is 2800 calories -/
theorem jonathan_weekly_deficit : weekly_deficit = 2800 := by
  sorry

end NUMINAMATH_CALUDE_jonathan_weekly_deficit_l3626_362675


namespace NUMINAMATH_CALUDE_divisibility_condition_l3626_362661

def fibonacci : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fibonacci (n + 1) + fibonacci n

theorem divisibility_condition (a b : ℕ) :
  (a^2 + b^2 + 1) % (a * b) = 0 ↔
  ((a = 1 ∧ b = 1) ∨ ∃ n : ℕ, n ≥ 1 ∧ a = fibonacci (2*n + 1) ∧ b = fibonacci (2*n - 1)) :=
sorry

end NUMINAMATH_CALUDE_divisibility_condition_l3626_362661


namespace NUMINAMATH_CALUDE_age_puzzle_l3626_362691

theorem age_puzzle (A : ℕ) (N : ℚ) (h1 : A = 18) (h2 : N * (A + 3) - N * (A - 3) = A) : N = 3 := by
  sorry

end NUMINAMATH_CALUDE_age_puzzle_l3626_362691


namespace NUMINAMATH_CALUDE_inequality_proof_l3626_362622

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  1 / a + 1 / b + 1 / (a * b) ≥ 8 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3626_362622


namespace NUMINAMATH_CALUDE_band_sections_fraction_l3626_362667

theorem band_sections_fraction (trumpet_fraction trombone_fraction : ℝ) 
  (h1 : trumpet_fraction = 0.5)
  (h2 : trombone_fraction = 0.12) :
  trumpet_fraction + trombone_fraction = 0.62 := by
  sorry

end NUMINAMATH_CALUDE_band_sections_fraction_l3626_362667


namespace NUMINAMATH_CALUDE_perfect_square_condition_l3626_362643

theorem perfect_square_condition (a b : ℤ) : 
  (∀ m n : ℕ, ∃ k : ℕ, a * m^2 + b * n^2 = k^2) → a * b = 0 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_condition_l3626_362643


namespace NUMINAMATH_CALUDE_inequality_proof_l3626_362680

theorem inequality_proof (a b c d : ℝ) (h : a^2 + b^2 + c^2 + d^2 = 4) :
  (2 + a) * (2 + b) ≥ c * d := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3626_362680


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_right_triangle_l3626_362648

/-- Given a right triangle with legs a and b, and hypotenuse c, 
    the radius r of its inscribed circle is (a + b - c) / 2 -/
theorem inscribed_circle_radius_right_triangle 
  (a b c : ℝ) (h_right : a^2 + b^2 = c^2) (h_pos : a > 0 ∧ b > 0 ∧ c > 0) :
  ∃ r : ℝ, r > 0 ∧ r = (a + b - c) / 2 ∧ 
    r * (a + b + c) / 2 = a * b / 2 := by
  sorry


end NUMINAMATH_CALUDE_inscribed_circle_radius_right_triangle_l3626_362648


namespace NUMINAMATH_CALUDE_triangle_transformation_correct_l3626_362635

def initial_triangle : List (ℝ × ℝ) := [(1, -2), (-1, -2), (1, 1)]

def rotate_180 (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, -p.2)

def reflect_x_axis (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

def rotate_270_clockwise (p : ℝ × ℝ) : ℝ × ℝ := (p.2, -p.1)

def transform_triangle (triangle : List (ℝ × ℝ)) : List (ℝ × ℝ) :=
  triangle.map (rotate_270_clockwise ∘ reflect_x_axis ∘ rotate_180)

theorem triangle_transformation_correct :
  transform_triangle initial_triangle = [(2, 1), (2, -1), (-1, -1)] := by
  sorry

end NUMINAMATH_CALUDE_triangle_transformation_correct_l3626_362635


namespace NUMINAMATH_CALUDE_arctan_tan_difference_l3626_362629

theorem arctan_tan_difference (x y : Real) :
  Real.arctan (Real.tan (65 * π / 180) - 2 * Real.tan (40 * π / 180)) = 25 * π / 180 := by
  sorry

end NUMINAMATH_CALUDE_arctan_tan_difference_l3626_362629


namespace NUMINAMATH_CALUDE_discount_calculation_l3626_362641

theorem discount_calculation (list_price : ℝ) (final_price : ℝ) (second_discount : ℝ) :
  list_price = 70 →
  final_price = 56.16 →
  second_discount = 10.857142857142863 →
  ∃ first_discount : ℝ,
    final_price = list_price * (1 - first_discount / 100) * (1 - second_discount / 100) ∧
    first_discount = 10 :=
by sorry

end NUMINAMATH_CALUDE_discount_calculation_l3626_362641


namespace NUMINAMATH_CALUDE_sqrt_pattern_l3626_362696

-- Define the square root function
noncomputable def sqrt (x : ℝ) := Real.sqrt x

-- Define the approximation relation
def approximately_equal (x y : ℝ) := ∃ (ε : ℝ), ε > 0 ∧ |x - y| < ε

-- State the theorem
theorem sqrt_pattern :
  (sqrt 0.0625 = 0.25) →
  (approximately_equal (sqrt 0.625) 0.791) →
  (sqrt 625 = 25) →
  (sqrt 6250 = 79.1) →
  (sqrt 62500 = 250) →
  (sqrt 625000 = 791) →
  (sqrt 6.25 = 2.5) ∧ (approximately_equal (sqrt 62.5) 7.91) :=
by sorry

end NUMINAMATH_CALUDE_sqrt_pattern_l3626_362696


namespace NUMINAMATH_CALUDE_system_no_solution_l3626_362666

def has_no_solution (a b c : ℤ) : Prop :=
  (a * b = 6) ∧ (b * c = 8) ∧ (c / 4 ≠ c / (4 * b))

theorem system_no_solution :
  ∀ a b c : ℤ, has_no_solution a b c ↔ 
    ((a = -6 ∧ b = -1 ∧ c = -8) ∨
     (a = -3 ∧ b = -2 ∧ c = -4) ∨
     (a = 3 ∧ b = 2 ∧ c = 4)) :=
by sorry

end NUMINAMATH_CALUDE_system_no_solution_l3626_362666


namespace NUMINAMATH_CALUDE_john_coin_collection_value_l3626_362620

/-- Represents the value of John's coin collection -/
def coin_collection_value (total_coins : ℕ) (silver_coins : ℕ) (gold_coins : ℕ) 
  (silver_coin_value : ℚ) (regular_coin_value : ℚ) : ℚ :=
  let gold_coin_value := 2 * silver_coin_value
  let regular_coins := total_coins - (silver_coins + gold_coins)
  silver_coins * silver_coin_value + gold_coins * gold_coin_value + regular_coins * regular_coin_value

theorem john_coin_collection_value : 
  coin_collection_value 20 10 5 (30/4) 1 = 155 := by
  sorry


end NUMINAMATH_CALUDE_john_coin_collection_value_l3626_362620


namespace NUMINAMATH_CALUDE_dip_amount_is_twenty_l3626_362602

/-- Represents the amount of dip that can be made given a budget and artichoke-to-dip ratio --/
def dip_amount (budget : ℚ) (artichokes_per_batch : ℕ) (ounces_per_batch : ℚ) (total_ounces : ℚ) : ℚ :=
  let price_per_artichoke : ℚ := budget / (total_ounces / ounces_per_batch * artichokes_per_batch)
  let artichokes_bought : ℚ := budget / price_per_artichoke
  (artichokes_bought / artichokes_per_batch) * ounces_per_batch

/-- Theorem stating that under given conditions, 20 ounces of dip can be made --/
theorem dip_amount_is_twenty :
  dip_amount 15 3 5 20 = 20 := by
  sorry

end NUMINAMATH_CALUDE_dip_amount_is_twenty_l3626_362602


namespace NUMINAMATH_CALUDE_base_for_256_four_digits_l3626_362640

-- Define the property of a number having exactly 4 digits in a given base
def has_four_digits (n : ℕ) (b : ℕ) : Prop :=
  b ^ 3 ≤ n ∧ n < b ^ 4

-- State the theorem
theorem base_for_256_four_digits :
  ∃! b : ℕ, has_four_digits 256 b ∧ b = 6 := by
  sorry

end NUMINAMATH_CALUDE_base_for_256_four_digits_l3626_362640


namespace NUMINAMATH_CALUDE_girls_ran_nine_miles_l3626_362656

/-- The number of laps run by boys -/
def boys_laps : ℕ := 34

/-- The additional laps run by girls compared to boys -/
def additional_girls_laps : ℕ := 20

/-- The fraction of a mile that one lap represents -/
def lap_mile_fraction : ℚ := 1 / 6

/-- The total number of laps run by girls -/
def girls_laps : ℕ := boys_laps + additional_girls_laps

/-- The number of miles run by girls -/
def girls_miles : ℚ := girls_laps * lap_mile_fraction

theorem girls_ran_nine_miles : girls_miles = 9 := by
  sorry

end NUMINAMATH_CALUDE_girls_ran_nine_miles_l3626_362656


namespace NUMINAMATH_CALUDE_three_distinct_real_roots_l3626_362606

/-- A cubic polynomial with specific conditions -/
structure CubicPolynomial where
  a : ℝ
  b : ℝ
  c : ℝ
  b_neg : b < 0
  ab_eq_9c : a * b = 9 * c

/-- The polynomial function -/
def polynomial (p : CubicPolynomial) (x : ℝ) : ℝ :=
  x^3 + p.a * x^2 + p.b * x + p.c

/-- Theorem stating that the polynomial has three different real roots -/
theorem three_distinct_real_roots (p : CubicPolynomial) :
  ∃ (x y z : ℝ), x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    polynomial p x = 0 ∧ polynomial p y = 0 ∧ polynomial p z = 0 := by
  sorry

end NUMINAMATH_CALUDE_three_distinct_real_roots_l3626_362606


namespace NUMINAMATH_CALUDE_solution_set_empty_implies_a_range_main_theorem_l3626_362639

-- Define the quadratic function
def f (a x : ℝ) : ℝ := a * x^2 + a * x + 3

-- State the theorem
theorem solution_set_empty_implies_a_range (a : ℝ) :
  (∀ x, f a x ≥ 0) → 0 ≤ a ∧ a ≤ 12 :=
by sorry

-- Define the range of a
def a_range : Set ℝ := {a | 0 ≤ a ∧ a ≤ 12}

-- State the main theorem
theorem main_theorem : 
  {a : ℝ | ∀ x, f a x ≥ 0} = a_range :=
by sorry

end NUMINAMATH_CALUDE_solution_set_empty_implies_a_range_main_theorem_l3626_362639


namespace NUMINAMATH_CALUDE_teenager_toddler_ratio_l3626_362653

theorem teenager_toddler_ratio (total_children : ℕ) (toddlers : ℕ) (newborns : ℕ) : 
  total_children = 40 → toddlers = 6 → newborns = 4 → 
  (total_children - toddlers - newborns) / toddlers = 5 := by
  sorry

end NUMINAMATH_CALUDE_teenager_toddler_ratio_l3626_362653


namespace NUMINAMATH_CALUDE_min_value_squared_sum_l3626_362657

theorem min_value_squared_sum (x y : ℝ) : (x + y)^2 + (x - 1/y)^2 ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_squared_sum_l3626_362657


namespace NUMINAMATH_CALUDE_quadratic_solution_sum_l3626_362647

theorem quadratic_solution_sum (m n p : ℤ) : 
  (∃ x : ℚ, x * (5 * x - 11) = -6) ∧
  (∃ x y : ℚ, x = (m + n.sqrt : ℚ) / p ∧ y = (m - n.sqrt : ℚ) / p ∧ 
    x * (5 * x - 11) = -6 ∧ y * (5 * y - 11) = -6) ∧
  Nat.gcd (Nat.gcd m.natAbs n.natAbs) p.natAbs = 1 →
  m + n + p = 70 :=
sorry

end NUMINAMATH_CALUDE_quadratic_solution_sum_l3626_362647


namespace NUMINAMATH_CALUDE_probability_not_paying_cash_l3626_362638

theorem probability_not_paying_cash (p_only_cash p_both : ℝ) 
  (h1 : p_only_cash = 0.45)
  (h2 : p_both = 0.15) : 
  1 - (p_only_cash + p_both) = 0.4 := by
sorry

end NUMINAMATH_CALUDE_probability_not_paying_cash_l3626_362638


namespace NUMINAMATH_CALUDE_range_of_a_for_quadratic_function_l3626_362672

theorem range_of_a_for_quadratic_function (a : ℝ) : 
  (∀ x : ℝ, x ≥ -1 → x^2 - 2*a*x + 2 ≥ a) ↔ -3 ≤ a ∧ a ≤ 1 := by
sorry

end NUMINAMATH_CALUDE_range_of_a_for_quadratic_function_l3626_362672


namespace NUMINAMATH_CALUDE_characterize_functions_l3626_362615

-- Define the property of the function f
def satisfies_property (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x * y) = f ((x^2 + y^2) / 2) + (x - y)^2

-- State the theorem
theorem characterize_functions (f : ℝ → ℝ) 
  (hf : Continuous f) 
  (hprop : satisfies_property f) :
  ∃ c : ℝ, ∀ x : ℝ, f x = c - 2 * x :=
sorry

end NUMINAMATH_CALUDE_characterize_functions_l3626_362615


namespace NUMINAMATH_CALUDE_smallest_part_of_proportional_division_l3626_362636

theorem smallest_part_of_proportional_division (total : ℚ) (a b c : ℚ) 
  (h_total : total = 81)
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0)
  (h_proportional : ∃ (k : ℚ), a = 3 * k ∧ b = 5 * k ∧ c = 7 * k)
  (h_sum : a + b + c = total) :
  a = 81 / 5 := by
  sorry

end NUMINAMATH_CALUDE_smallest_part_of_proportional_division_l3626_362636


namespace NUMINAMATH_CALUDE_sara_quarters_l3626_362658

/-- The number of quarters Sara initially had -/
def initial_quarters : ℕ := 783

/-- The number of quarters Sara's dad borrowed -/
def borrowed_quarters : ℕ := 271

/-- The number of quarters Sara has now -/
def remaining_quarters : ℕ := initial_quarters - borrowed_quarters

theorem sara_quarters : remaining_quarters = 512 := by
  sorry

end NUMINAMATH_CALUDE_sara_quarters_l3626_362658


namespace NUMINAMATH_CALUDE_union_perimeter_bound_l3626_362607

/-- A disc in a 2D plane -/
structure Disc where
  center : ℝ × ℝ
  radius : ℝ

/-- A set of discs satisfying the problem conditions -/
structure DiscSet where
  discs : Set Disc
  segment_length : ℝ
  centers_on_segment : ∀ d ∈ discs, ∃ x : ℝ, d.center = (x, 0) ∧ 0 ≤ x ∧ x ≤ segment_length
  radii_bounded : ∀ d ∈ discs, d.radius ≤ 1

/-- The perimeter of the union of discs -/
noncomputable def union_perimeter (ds : DiscSet) : ℝ := sorry

/-- The main theorem -/
theorem union_perimeter_bound (ds : DiscSet) :
  union_perimeter ds ≤ 4 * ds.segment_length + 8 := by
  sorry

end NUMINAMATH_CALUDE_union_perimeter_bound_l3626_362607


namespace NUMINAMATH_CALUDE_smallest_n_for_Q_less_than_threshold_l3626_362628

/-- The probability of stopping at box n, given n ≥ 50 -/
def Q (n : ℕ) : ℚ := 2 / (n + 2)

/-- The smallest n ≥ 50 such that Q(n) < 1/2023 is 1011 -/
theorem smallest_n_for_Q_less_than_threshold : 
  (∀ k : ℕ, k ≥ 50 → k < 1011 → Q k ≥ 1/2023) ∧ 
  (Q 1011 < 1/2023) := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_for_Q_less_than_threshold_l3626_362628


namespace NUMINAMATH_CALUDE_max_value_xy_over_x2_plus_y2_l3626_362684

theorem max_value_xy_over_x2_plus_y2 (x y : ℝ) 
  (hx : 1/4 ≤ x ∧ x ≤ 3/5) (hy : 2/7 ≤ y ∧ y ≤ 1/2) :
  x * y / (x^2 + y^2) ≤ 2/5 := by
  sorry

end NUMINAMATH_CALUDE_max_value_xy_over_x2_plus_y2_l3626_362684


namespace NUMINAMATH_CALUDE_truncated_pyramid_volume_division_l3626_362677

/-- Represents a truncated triangular pyramid -/
structure TruncatedPyramid where
  upperBaseArea : ℝ
  lowerBaseArea : ℝ
  height : ℝ
  baseRatio : upperBaseArea / lowerBaseArea = 1 / 4

/-- Represents the volumes of the two parts created by the plane -/
structure DividedVolumes where
  v1 : ℝ
  v2 : ℝ

/-- 
  Given a truncated triangular pyramid where the corresponding sides of the upper and lower 
  bases are in the ratio 1:2, if a plane is drawn through a side of the upper base parallel 
  to the opposite lateral edge, it divides the volume of the truncated pyramid in the ratio 3:4.
-/
theorem truncated_pyramid_volume_division (p : TruncatedPyramid) : 
  ∃ (v : DividedVolumes), v.v1 / v.v2 = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_truncated_pyramid_volume_division_l3626_362677


namespace NUMINAMATH_CALUDE_fish_catch_total_l3626_362670

/-- The total number of fish caught by Leo, Agrey, and Sierra -/
def total_fish (leo agrey sierra : ℕ) : ℕ := leo + agrey + sierra

/-- Theorem stating the total number of fish caught given the conditions -/
theorem fish_catch_total :
  ∀ (leo agrey sierra : ℕ),
    leo = 40 →
    agrey = leo + 20 →
    sierra = agrey + 15 →
    total_fish leo agrey sierra = 175 := by
  sorry

end NUMINAMATH_CALUDE_fish_catch_total_l3626_362670


namespace NUMINAMATH_CALUDE_value_of_a_l3626_362669

-- Define the conversion rate from paise to rupees
def paiseToRupees (paise : ℚ) : ℚ := paise / 100

-- Define the problem statement
theorem value_of_a (a : ℚ) (h : (0.5 / 100) * a = paiseToRupees 70) : a = 140 := by
  sorry

end NUMINAMATH_CALUDE_value_of_a_l3626_362669


namespace NUMINAMATH_CALUDE_christen_peeled_17_l3626_362617

/-- Represents the potato peeling scenario -/
structure PotatoPeeling where
  initial_potatoes : ℕ
  homer_rate : ℕ
  christen_rate : ℕ
  homer_solo_time : ℕ

/-- Calculates the number of potatoes Christen peeled -/
def christens_potatoes (scenario : PotatoPeeling) : ℕ :=
  let potatoes_after_homer := scenario.initial_potatoes - scenario.homer_rate * scenario.homer_solo_time
  let combined_rate := scenario.homer_rate + scenario.christen_rate
  let remaining_time := potatoes_after_homer / combined_rate
  remaining_time * scenario.christen_rate

/-- Theorem stating that Christen peeled 17 potatoes -/
theorem christen_peeled_17 (scenario : PotatoPeeling) 
  (h1 : scenario.initial_potatoes = 58)
  (h2 : scenario.homer_rate = 4)
  (h3 : scenario.christen_rate = 4)
  (h4 : scenario.homer_solo_time = 6) :
  christens_potatoes scenario = 17 := by
  sorry

#eval christens_potatoes { initial_potatoes := 58, homer_rate := 4, christen_rate := 4, homer_solo_time := 6 }

end NUMINAMATH_CALUDE_christen_peeled_17_l3626_362617


namespace NUMINAMATH_CALUDE_billy_bumper_rides_l3626_362625

/-- The number of times Billy rode the ferris wheel -/
def ferris_rides : ℕ := 7

/-- The cost of each ride in tickets -/
def ticket_cost : ℕ := 5

/-- The total number of tickets Billy used -/
def total_tickets : ℕ := 50

/-- The number of times Billy rode the bumper cars -/
def bumper_rides : ℕ := (total_tickets - ferris_rides * ticket_cost) / ticket_cost

theorem billy_bumper_rides : bumper_rides = 3 := by
  sorry

end NUMINAMATH_CALUDE_billy_bumper_rides_l3626_362625


namespace NUMINAMATH_CALUDE_equation_solution_l3626_362600

theorem equation_solution : 
  let eq (x : ℝ) := x^3 + Real.log 25 + Real.log 32 + Real.log 53 * x - Real.log 23 - Real.log 35 - Real.log 52 * x^2 - 1
  (eq (Real.log 23) = 0) ∧ (eq (Real.log 35) = 0) ∧ (eq (Real.log 52) = 0) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3626_362600


namespace NUMINAMATH_CALUDE_average_book_width_l3626_362604

def book_widths : List ℝ := [7.5, 3, 0.75, 4, 1.25, 12]

theorem average_book_width : 
  (List.sum book_widths) / (List.length book_widths) = 4.75 := by
  sorry

end NUMINAMATH_CALUDE_average_book_width_l3626_362604


namespace NUMINAMATH_CALUDE_negation_of_p_l3626_362612

-- Define the original proposition
def p : Prop := ∃ x₀ : ℝ, x₀^2 + 2*x₀ + 3 > 0

-- State the theorem
theorem negation_of_p : 
  ¬p ↔ ∀ x : ℝ, x^2 + 2*x + 3 ≤ 0 := by sorry

end NUMINAMATH_CALUDE_negation_of_p_l3626_362612


namespace NUMINAMATH_CALUDE_simplify_expression_l3626_362697

theorem simplify_expression : (4 + 2 + 6) / 3 - (2 + 1) / 3 = 3 := by sorry

end NUMINAMATH_CALUDE_simplify_expression_l3626_362697


namespace NUMINAMATH_CALUDE_banana_orange_equivalence_l3626_362668

/-- Given that 2/3 of 15 bananas are worth 12 oranges,
    prove that 1/4 of 20 bananas are worth 6 oranges. -/
theorem banana_orange_equivalence :
  ∀ (banana_value orange_value : ℚ),
  (2 / 3 : ℚ) * 15 * banana_value = 12 * orange_value →
  (1 / 4 : ℚ) * 20 * banana_value = 6 * orange_value :=
by sorry

end NUMINAMATH_CALUDE_banana_orange_equivalence_l3626_362668


namespace NUMINAMATH_CALUDE_coefficient_third_term_binomial_expansion_l3626_362663

theorem coefficient_third_term_binomial_expansion :
  let n : ℕ := 3
  let a : ℝ := 2
  let b : ℝ := 1
  let k : ℕ := 2
  (n.choose k) * a^(n - k) * b^k = 6 := by
sorry

end NUMINAMATH_CALUDE_coefficient_third_term_binomial_expansion_l3626_362663


namespace NUMINAMATH_CALUDE_hyperbola_equation_l3626_362698

/-- The standard equation of a hyperbola given specific conditions -/
theorem hyperbola_equation (F₁ F₂ M : ℝ × ℝ) : 
  F₁ = (0, Real.sqrt 10) →
  F₂ = (0, -Real.sqrt 10) →
  (M.1 - F₁.1) * (M.1 - F₂.1) + (M.2 - F₁.2) * (M.2 - F₂.2) = 0 →
  Real.sqrt ((M.1 - F₁.1)^2 + (M.2 - F₁.2)^2) * 
    Real.sqrt ((M.1 - F₂.1)^2 + (M.2 - F₂.2)^2) = 2 →
  M.2^2 / 9 - M.1^2 = 1 := by
  sorry


end NUMINAMATH_CALUDE_hyperbola_equation_l3626_362698


namespace NUMINAMATH_CALUDE_course_selection_theorem_l3626_362649

theorem course_selection_theorem (total_courses : Nat) (conflicting_courses : Nat) 
  (courses_to_choose : Nat) (h1 : total_courses = 6) (h2 : conflicting_courses = 2) 
  (h3 : courses_to_choose = 2) :
  (Nat.choose (total_courses - conflicting_courses) courses_to_choose + 
   conflicting_courses * Nat.choose (total_courses - conflicting_courses) (courses_to_choose - 1)) = 14 := by
  sorry

end NUMINAMATH_CALUDE_course_selection_theorem_l3626_362649


namespace NUMINAMATH_CALUDE_tangent_half_identities_l3626_362654

theorem tangent_half_identities (α : Real) (h : Real.tan α = 1/2) :
  ((4 * Real.sin α - Real.cos α) / (Real.sin α + Real.cos α) = 2/3) ∧
  (Real.sin α ^ 2 - Real.sin (2 * α) = -3/5) := by
  sorry

end NUMINAMATH_CALUDE_tangent_half_identities_l3626_362654


namespace NUMINAMATH_CALUDE_metal_waste_calculation_l3626_362632

theorem metal_waste_calculation (R : ℝ) (h : R = 10) : 
  let original_circle_area := π * R^2
  let max_square_side := R * Real.sqrt 2
  let max_square_area := max_square_side^2
  let inner_circle_radius := max_square_side / 2
  let inner_circle_area := π * inner_circle_radius^2
  original_circle_area - inner_circle_area = 50 * π - 200 :=
by sorry

end NUMINAMATH_CALUDE_metal_waste_calculation_l3626_362632


namespace NUMINAMATH_CALUDE_tree_height_proof_l3626_362619

/-- The growth rate of the tree in inches per year -/
def growth_rate : ℝ := 0.5

/-- The number of years it takes for the tree to reach its final height -/
def years_to_grow : ℕ := 240

/-- The final height of the tree in inches -/
def final_height : ℝ := 720

/-- The current height of the tree in inches -/
def current_height : ℝ := final_height - (growth_rate * years_to_grow)

theorem tree_height_proof :
  current_height = 600 := by sorry

end NUMINAMATH_CALUDE_tree_height_proof_l3626_362619


namespace NUMINAMATH_CALUDE_seven_patients_three_doctors_l3626_362690

/-- The number of ways to assign n distinct objects to k distinct categories,
    where each object is assigned to exactly one category and
    each category receives at least one object. -/
def assignments (n k : ℕ) : ℕ :=
  k^n - (k * (k-1)^n - k * (k-1) * (k-2)^n)

/-- There are 7 patients and 3 doctors -/
theorem seven_patients_three_doctors :
  assignments 7 3 = 1806 := by
  sorry

end NUMINAMATH_CALUDE_seven_patients_three_doctors_l3626_362690


namespace NUMINAMATH_CALUDE_fraction_invariance_l3626_362694

theorem fraction_invariance (x y : ℝ) :
  (2 * x + y) / (3 * x + y) = (2 * (10 * x) + 10 * y) / (3 * (10 * x) + 10 * y) :=
by sorry

end NUMINAMATH_CALUDE_fraction_invariance_l3626_362694


namespace NUMINAMATH_CALUDE_right_triangle_longest_altitudes_sum_l3626_362660

theorem right_triangle_longest_altitudes_sum (a b c : ℝ) : 
  a = 9 ∧ b = 12 ∧ c = 15 ∧ a^2 + b^2 = c^2 → 
  (max a b + min a b) = 21 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_longest_altitudes_sum_l3626_362660


namespace NUMINAMATH_CALUDE_f_properties_l3626_362688

noncomputable def f (x : ℝ) : ℝ := Real.log (x / (x^2 + 1))

theorem f_properties :
  (∀ x : ℝ, x > 0 → f x ≠ 0) ∧
  (∀ x : ℝ, 0 < x → x < 1 → ∀ y : ℝ, x < y → y < 1 → f x < f y) ∧
  (∀ x : ℝ, x > 1 → ∀ y : ℝ, y > x → f x > f y) :=
sorry

end NUMINAMATH_CALUDE_f_properties_l3626_362688


namespace NUMINAMATH_CALUDE_athletes_leaving_hours_l3626_362644

/-- The number of hours athletes left the camp -/
def hours_athletes_left : ℕ := 4

/-- The initial number of athletes in the camp -/
def initial_athletes : ℕ := 300

/-- The rate at which athletes left the camp (per hour) -/
def leaving_rate : ℕ := 28

/-- The rate at which new athletes entered the camp (per hour) -/
def entering_rate : ℕ := 15

/-- The number of hours new athletes entered the camp -/
def entering_hours : ℕ := 7

/-- The difference in the total number of athletes over the two nights -/
def athlete_difference : ℕ := 7

theorem athletes_leaving_hours :
  initial_athletes - (leaving_rate * hours_athletes_left) + 
  (entering_rate * entering_hours) = initial_athletes + athlete_difference :=
by sorry

end NUMINAMATH_CALUDE_athletes_leaving_hours_l3626_362644


namespace NUMINAMATH_CALUDE_integer_partition_impossibility_l3626_362679

theorem integer_partition_impossibility : 
  ¬ (∃ (A B C : Set ℤ), 
    (∀ (n : ℤ), (n ∈ A ∧ (n - 50) ∈ B ∧ (n + 1987) ∈ C) ∨
                (n ∈ A ∧ (n - 50) ∈ C ∧ (n + 1987) ∈ B) ∨
                (n ∈ B ∧ (n - 50) ∈ A ∧ (n + 1987) ∈ C) ∨
                (n ∈ B ∧ (n - 50) ∈ C ∧ (n + 1987) ∈ A) ∨
                (n ∈ C ∧ (n - 50) ∈ A ∧ (n + 1987) ∈ B) ∨
                (n ∈ C ∧ (n - 50) ∈ B ∧ (n + 1987) ∈ A)) ∧
    (A ∪ B ∪ C = Set.univ) ∧ 
    (A ∩ B = ∅) ∧ (B ∩ C = ∅) ∧ (A ∩ C = ∅)) :=
by sorry

end NUMINAMATH_CALUDE_integer_partition_impossibility_l3626_362679


namespace NUMINAMATH_CALUDE_existence_of_integer_roots_l3626_362671

theorem existence_of_integer_roots : ∃ (a b c d e f : ℤ),
  (∀ x : ℤ, (x + a) * (x^2 + b*x + c) * (x^3 + d*x^2 + e*x + f) = 0 ↔ 
    x = a ∨ x^2 + b*x + c = 0 ∨ x^3 + d*x^2 + e*x + f = 0) ∧
  (∃! (r₁ r₂ r₃ r₄ r₅ r₆ : ℤ), 
    {r₁, r₂, r₃, r₄, r₅, r₆} = {a} ∪ 
      {x : ℤ | x^2 + b*x + c = 0} ∪ 
      {x : ℤ | x^3 + d*x^2 + e*x + f = 0}) :=
sorry

end NUMINAMATH_CALUDE_existence_of_integer_roots_l3626_362671


namespace NUMINAMATH_CALUDE_f_difference_l3626_362603

/-- The function f(x) = 3x^2 - 4x + 2 -/
def f (x : ℝ) : ℝ := 3 * x^2 - 4 * x + 2

/-- Theorem: For all real x and h, f(x + h) - f(x) = h(6x + 3h - 4) -/
theorem f_difference (x h : ℝ) : f (x + h) - f x = h * (6 * x + 3 * h - 4) := by
  sorry

end NUMINAMATH_CALUDE_f_difference_l3626_362603


namespace NUMINAMATH_CALUDE_find_x2_l3626_362637

theorem find_x2 (x₁ x₂ x₃ : ℝ) 
  (h_order : x₁ < x₂ ∧ x₂ < x₃)
  (h_sum1 : x₁ + x₂ = 14)
  (h_sum2 : x₁ + x₃ = 17)
  (h_sum3 : x₂ + x₃ = 33) : 
  x₂ = 15 := by
sorry

end NUMINAMATH_CALUDE_find_x2_l3626_362637


namespace NUMINAMATH_CALUDE_special_sequence_third_term_l3626_362609

/-- A sequence S with special properties -/
def SpecialSequence (S : ℕ → ℝ) : Prop :=
  ∃ a : ℝ, a > 0 ∧
  S 0 = a^4 ∧
  (∀ n : ℕ, S (n + 1) = 4 * Real.sqrt (S n)) ∧
  (S 2 - S 1 = S 1 - S 0)

/-- The third term of the special sequence can only be 16 or 8√5 - 8 -/
theorem special_sequence_third_term (S : ℕ → ℝ) (h : SpecialSequence S) :
  S 2 = 16 ∨ S 2 = 8 * Real.sqrt 5 - 8 := by
  sorry

end NUMINAMATH_CALUDE_special_sequence_third_term_l3626_362609


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l3626_362674

theorem quadratic_equation_solution :
  ∃ x : ℝ, x^2 + 4*x + 3 = -(x + 3)*(x + 5) ∧ x = -3 :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l3626_362674


namespace NUMINAMATH_CALUDE_equation_solution_l3626_362630

/-- Given an equation y = a + b/x where a and b are constants, 
    if y = 3 when x = 2 and y = 2 when x = 4, then a + b = 5 -/
theorem equation_solution (a b : ℝ) : 
  (3 = a + b / 2) → (2 = a + b / 4) → (a + b = 5) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3626_362630


namespace NUMINAMATH_CALUDE_circle_arrangement_divisible_by_three_l3626_362689

/-- A type representing the arrangement of numbers in a circle. -/
def CircularArrangement (n : ℕ) := Fin n → ℕ

/-- Predicate to check if two numbers differ by 1, 2, or factor of two. -/
def ValidDifference (a b : ℕ) : Prop :=
  (a = b + 1) ∨ (b = a + 1) ∨ (a = b + 2) ∨ (b = a + 2) ∨ (a = 2 * b) ∨ (b = 2 * a)

/-- Theorem stating that in any arrangement of 99 natural numbers in a circle
    where any two neighboring numbers differ either by 1, or by 2,
    or by a factor of two, at least one of these numbers is divisible by 3. -/
theorem circle_arrangement_divisible_by_three
  (arr : CircularArrangement 99)
  (h : ∀ i : Fin 99, ValidDifference (arr i) (arr (i + 1))) :
  ∃ i : Fin 99, 3 ∣ arr i :=
sorry

end NUMINAMATH_CALUDE_circle_arrangement_divisible_by_three_l3626_362689


namespace NUMINAMATH_CALUDE_price_change_l3626_362655

theorem price_change (x : ℝ) (h : x > 0) : x * (1 - 0.2) * (1 + 0.2) < x := by
  sorry

end NUMINAMATH_CALUDE_price_change_l3626_362655


namespace NUMINAMATH_CALUDE_sequence_sum_l3626_362682

theorem sequence_sum (a b c d : ℕ) 
  (h1 : 0 < a ∧ a < b ∧ b < c ∧ c < d)
  (h2 : b * a = c * a)
  (h3 : c - b = d - c)
  (h4 : d - a = 36) : 
  a + b + c + d = 1188 := by
  sorry

end NUMINAMATH_CALUDE_sequence_sum_l3626_362682


namespace NUMINAMATH_CALUDE_terminal_side_equivalence_l3626_362673

/-- Two angles have the same terminal side if their difference is an integer multiple of 360° -/
def same_terminal_side (α β : ℝ) : Prop :=
  ∃ k : ℤ, α - β = 360 * k

/-- Prove that -330° has the same terminal side as 30° -/
theorem terminal_side_equivalence : same_terminal_side (-330) 30 := by
  sorry

end NUMINAMATH_CALUDE_terminal_side_equivalence_l3626_362673


namespace NUMINAMATH_CALUDE_prob_select_5_times_expectation_X_l3626_362627

-- Define the review types
inductive ReviewType
  | Good
  | Neutral
  | Bad

-- Define the age groups
inductive AgeGroup
  | Below50
  | Above50

-- Define the sample data
def sampleData : Fin 2 → Fin 3 → Nat
  | ⟨0, _⟩ => fun
    | ⟨0, _⟩ => 10000  -- Good reviews for Below50
    | ⟨1, _⟩ => 2000   -- Neutral reviews for Below50
    | ⟨2, _⟩ => 2000   -- Bad reviews for Below50
  | ⟨1, _⟩ => fun
    | ⟨0, _⟩ => 2000   -- Good reviews for Above50
    | ⟨1, _⟩ => 3000   -- Neutral reviews for Above50
    | ⟨2, _⟩ => 1000   -- Bad reviews for Above50

-- Define the total sample size
def totalSampleSize : Nat := 20000

-- Define the probability of selecting a good review
def probGoodReview : Rat :=
  (sampleData ⟨0, sorry⟩ ⟨0, sorry⟩ + sampleData ⟨1, sorry⟩ ⟨0, sorry⟩) / totalSampleSize

-- Theorem for the probability of selecting 5 times
theorem prob_select_5_times :
  (1 - probGoodReview)^5 + (1 - probGoodReview)^4 * probGoodReview = 16/625 := by sorry

-- Define the number of people giving neutral reviews in each age group
def neutralReviews : Fin 2 → Nat
  | ⟨0, _⟩ => sampleData ⟨0, sorry⟩ ⟨1, sorry⟩
  | ⟨1, _⟩ => sampleData ⟨1, sorry⟩ ⟨1, sorry⟩

-- Define the total number of neutral reviews
def totalNeutralReviews : Nat := neutralReviews ⟨0, sorry⟩ + neutralReviews ⟨1, sorry⟩

-- Define the probability distribution of X
def probX : Fin 4 → Rat
  | ⟨0, _⟩ => 1/6
  | ⟨1, _⟩ => 1/2
  | ⟨2, _⟩ => 3/10
  | ⟨3, _⟩ => 1/30

-- Theorem for the mathematical expectation of X
theorem expectation_X :
  (0 : Rat) * probX ⟨0, sorry⟩ + 1 * probX ⟨1, sorry⟩ + 2 * probX ⟨2, sorry⟩ + 3 * probX ⟨3, sorry⟩ = 6/5 := by sorry

end NUMINAMATH_CALUDE_prob_select_5_times_expectation_X_l3626_362627


namespace NUMINAMATH_CALUDE_square_side_length_l3626_362676

theorem square_side_length : 
  ∃ (x : ℝ), x > 0 ∧ 4 * x = 2 * (x ^ 2) ∧ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l3626_362676


namespace NUMINAMATH_CALUDE_equal_powers_of_negative_one_l3626_362613

theorem equal_powers_of_negative_one : 
  (-7^4 ≠ (-7)^4) ∧ 
  (4^3 ≠ 3^4) ∧ 
  (-(-6) ≠ -|(-6)|) ∧ 
  ((-1)^3 = (-1)^2023) := by
  sorry

end NUMINAMATH_CALUDE_equal_powers_of_negative_one_l3626_362613


namespace NUMINAMATH_CALUDE_pizza_consumption_l3626_362683

theorem pizza_consumption (n : ℕ) (first_trip : ℚ) (subsequent_trips : ℚ) : 
  n = 6 → 
  first_trip = 2/3 → 
  subsequent_trips = 1/2 → 
  (1 - (1 - first_trip) * subsequent_trips^(n-1) : ℚ) = 191/192 := by
  sorry

end NUMINAMATH_CALUDE_pizza_consumption_l3626_362683


namespace NUMINAMATH_CALUDE_sin_2x_value_l3626_362601

theorem sin_2x_value (x : Real) (h : Real.sin (x + π/4) = 1/4) : 
  Real.sin (2*x) = -7/8 := by
sorry

end NUMINAMATH_CALUDE_sin_2x_value_l3626_362601


namespace NUMINAMATH_CALUDE_same_num_digits_l3626_362634

/-- The number of digits in the decimal representation of a positive integer -/
def num_digits (n : ℕ) : ℕ := sorry

/-- Theorem: If 10^b < a^b and 2^b < 10^b, then a^b and a^b + 2^b have the same number of digits -/
theorem same_num_digits (a b : ℕ) (h1 : 10^b < a^b) (h2 : 2^b < 10^b) :
  num_digits (a^b) = num_digits (a^b + 2^b) := by sorry

end NUMINAMATH_CALUDE_same_num_digits_l3626_362634


namespace NUMINAMATH_CALUDE_sum_remainder_l3626_362664

theorem sum_remainder (x y z : ℕ+) 
  (hx : x ≡ 30 [ZMOD 59])
  (hy : y ≡ 27 [ZMOD 59])
  (hz : z ≡ 4 [ZMOD 59]) :
  (x + y + z : ℤ) ≡ 2 [ZMOD 59] := by
  sorry

end NUMINAMATH_CALUDE_sum_remainder_l3626_362664


namespace NUMINAMATH_CALUDE_y_is_function_of_x_f_a_is_constant_not_always_injective_not_always_analytic_l3626_362624

-- Define a function f from real numbers to real numbers
variable (f : ℝ → ℝ)

-- Statement 1: y is a function of x
theorem y_is_function_of_x : ∀ x : ℝ, ∃ y : ℝ, y = f x := by sorry

-- Statement 3: f(a) represents the value of the function f(x) when x = a, which is a constant
theorem f_a_is_constant (a : ℝ) : ∃ k : ℝ, f a = k := by sorry

-- Statement 2 (negation): It is not necessarily true that for different x, the value of y is also different
theorem not_always_injective : ¬ (∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → f x₁ ≠ f x₂) := by sorry

-- Statement 4 (negation): It is not always possible to represent f(x) by a specific formula
theorem not_always_analytic : ¬ (∃ formula : ℝ → ℝ, ∀ x : ℝ, f x = formula x) := by sorry

end NUMINAMATH_CALUDE_y_is_function_of_x_f_a_is_constant_not_always_injective_not_always_analytic_l3626_362624


namespace NUMINAMATH_CALUDE_base8_157_equals_111_l3626_362616

/-- Converts a base-8 number to base-10 --/
def base8To10 (digits : List Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * (8 ^ i)) 0

/-- The base-8 representation of 157 --/
def base8_157 : List Nat := [1, 5, 7]

theorem base8_157_equals_111 :
  base8To10 base8_157 = 111 := by
  sorry

end NUMINAMATH_CALUDE_base8_157_equals_111_l3626_362616


namespace NUMINAMATH_CALUDE_original_deck_size_l3626_362662

/-- Represents a deck of playing cards -/
structure Deck where
  total_cards : ℕ

/-- Represents the game setup -/
structure GameSetup where
  original_deck : Deck
  cards_kept_away : ℕ
  cards_in_play : ℕ

/-- The number of cards in a standard deck -/
def standard_deck_size : ℕ := 52

/-- Theorem: The original deck had 52 cards -/
theorem original_deck_size (setup : GameSetup) 
  (h1 : setup.cards_kept_away = 2) 
  (h2 : setup.cards_in_play + setup.cards_kept_away = setup.original_deck.total_cards) : 
  setup.original_deck.total_cards = standard_deck_size := by
  sorry

end NUMINAMATH_CALUDE_original_deck_size_l3626_362662


namespace NUMINAMATH_CALUDE_coupon_savings_difference_coupon_savings_difference_holds_l3626_362626

theorem coupon_savings_difference : ℝ → Prop :=
  fun difference =>
    ∃ (x y : ℝ),
      x > 120 ∧ y > 120 ∧
      (∀ p : ℝ, p > 120 →
        (0.2 * p ≥ 35 ∧ 0.2 * p ≥ 0.3 * (p - 120)) →
        x ≤ p ∧ p ≤ y) ∧
      (0.2 * x ≥ 35 ∧ 0.2 * x ≥ 0.3 * (x - 120)) ∧
      (0.2 * y ≥ 35 ∧ 0.2 * y ≥ 0.3 * (y - 120)) ∧
      difference = y - x ∧
      difference = 185

theorem coupon_savings_difference_holds : coupon_savings_difference 185 := by
  sorry

end NUMINAMATH_CALUDE_coupon_savings_difference_coupon_savings_difference_holds_l3626_362626


namespace NUMINAMATH_CALUDE_ratio_a_to_c_l3626_362605

theorem ratio_a_to_c (a b c d : ℚ) 
  (hab : a / b = 5 / 2)
  (hcd : c / d = 4 / 1)
  (hdb : d / b = 1 / 4) :
  a / c = 5 / 2 := by
sorry

end NUMINAMATH_CALUDE_ratio_a_to_c_l3626_362605


namespace NUMINAMATH_CALUDE_apple_distribution_l3626_362610

theorem apple_distribution (n : ℕ) (k : ℕ) (h1 : n = 30) (h2 : k = 4) :
  (Nat.choose (n + k - 1) (k - 1)) = 3276 :=
sorry

end NUMINAMATH_CALUDE_apple_distribution_l3626_362610


namespace NUMINAMATH_CALUDE_expression_simplification_l3626_362692

theorem expression_simplification (x : ℝ) : 2*x - 3*(2 - x) + 4*(3 + x) - 5*(1 - 2*x) = 19*x + 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3626_362692


namespace NUMINAMATH_CALUDE_hyperbolas_same_asymptotes_l3626_362686

/-- Given two hyperbolas with equations (x^2/16) - (y^2/25) = 1 and (y^2/49) - (x^2/M) = 1,
    if they have the same asymptotes, then M = 784/25 -/
theorem hyperbolas_same_asymptotes (M : ℝ) :
  (∀ x y : ℝ, x^2/16 - y^2/25 = 1 ↔ y^2/49 - x^2/M = 1) →
  (∀ x y : ℝ, y = (5/4) * x ↔ y = (7/Real.sqrt M) * x) →
  M = 784/25 := by
  sorry

end NUMINAMATH_CALUDE_hyperbolas_same_asymptotes_l3626_362686


namespace NUMINAMATH_CALUDE_power_of_nine_mod_fifty_l3626_362650

theorem power_of_nine_mod_fifty : 9^1002 % 50 = 1 := by
  sorry

end NUMINAMATH_CALUDE_power_of_nine_mod_fifty_l3626_362650


namespace NUMINAMATH_CALUDE_credit_card_balance_l3626_362621

theorem credit_card_balance 
  (G : ℝ) 
  (gold_balance : ℝ) 
  (platinum_balance : ℝ) 
  (h1 : gold_balance = G / 3) 
  (h2 : 0.5833333333333334 = 1 - (platinum_balance + gold_balance) / (2 * G)) : 
  platinum_balance = (1 / 4) * (2 * G) := by
sorry

end NUMINAMATH_CALUDE_credit_card_balance_l3626_362621


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_fractions_l3626_362611

theorem arithmetic_mean_of_fractions :
  let a := (3 : ℚ) / 4
  let b := (5 : ℚ) / 8
  let c := (7 : ℚ) / 9
  (a + b + c) / 3 = 155 / 216 := by sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_fractions_l3626_362611


namespace NUMINAMATH_CALUDE_extreme_value_implies_a_l3626_362652

def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + x^2

theorem extreme_value_implies_a (a : ℝ) :
  (∃ (ε : ℝ), ∀ (h : ℝ), 0 < |h| → |h| < ε → f a (-2 + h) ≤ f a (-2)) →
  a = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_extreme_value_implies_a_l3626_362652


namespace NUMINAMATH_CALUDE_fox_rabbit_bridge_problem_l3626_362623

theorem fox_rabbit_bridge_problem (x : ℝ) : 
  (((2 * ((2 * ((2 * ((2 * x) - 50)) - 50)) - 50)) - 50) = 0) → x = 46.875 := by
  sorry

end NUMINAMATH_CALUDE_fox_rabbit_bridge_problem_l3626_362623


namespace NUMINAMATH_CALUDE_cube_root_function_l3626_362685

theorem cube_root_function (k : ℝ) (y x : ℝ → ℝ) :
  (∀ x, y x = k * (x ^ (1/3))) →
  y 64 = 4 →
  y 8 = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_cube_root_function_l3626_362685


namespace NUMINAMATH_CALUDE_english_score_calculation_l3626_362651

theorem english_score_calculation (average_before : ℝ) (average_after : ℝ) : 
  average_before = 92 →
  average_after = 94 →
  (3 * average_before + english_score) / 4 = average_after →
  english_score = 100 :=
by
  sorry

#check english_score_calculation

end NUMINAMATH_CALUDE_english_score_calculation_l3626_362651


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l3626_362645

theorem simplify_and_evaluate : 
  ∀ x : ℝ, x ≠ -2 → x ≠ 1 → 
  (1 - 3 / (x + 2)) / ((x - 1) / (x^2 + 4*x + 4)) = x + 2 ∧
  (1 - 3 / (-1 + 2)) / ((-1 - 1) / ((-1)^2 + 4*(-1) + 4)) = 1 := by
sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l3626_362645


namespace NUMINAMATH_CALUDE_picture_album_distribution_l3626_362646

theorem picture_album_distribution : ∃ (a b c : ℕ), 
  a + b + c = 40 ∧ 
  a + b = 28 ∧ 
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  a > 0 ∧ b > 0 ∧ c > 0 :=
by sorry

end NUMINAMATH_CALUDE_picture_album_distribution_l3626_362646


namespace NUMINAMATH_CALUDE_second_caterer_more_cost_effective_l3626_362633

/-- Represents the cost function for a caterer -/
structure Caterer where
  base_fee : ℕ
  per_person : ℕ

/-- Calculates the total cost for a given number of people -/
def total_cost (c : Caterer) (people : ℕ) : ℕ :=
  c.base_fee + c.per_person * people

/-- The first caterer's pricing structure -/
def caterer1 : Caterer :=
  { base_fee := 120, per_person := 14 }

/-- The second caterer's pricing structure -/
def caterer2 : Caterer :=
  { base_fee := 210, per_person := 11 }

/-- Theorem stating the minimum number of people for the second caterer to be more cost-effective -/
theorem second_caterer_more_cost_effective :
  (∀ n : ℕ, n ≥ 31 → total_cost caterer2 n < total_cost caterer1 n) ∧
  (∀ n : ℕ, n < 31 → total_cost caterer2 n ≥ total_cost caterer1 n) :=
sorry

end NUMINAMATH_CALUDE_second_caterer_more_cost_effective_l3626_362633


namespace NUMINAMATH_CALUDE_m_four_sufficient_not_necessary_l3626_362678

/-- Represents a line in the form ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if two lines are perpendicular -/
def are_perpendicular (l1 l2 : Line) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

/-- Define the two lines parameterized by m -/
def line1 (m : ℝ) : Line := ⟨2*m - 4, m + 1, 2⟩
def line2 (m : ℝ) : Line := ⟨m + 1, -m, 3⟩

/-- Main theorem -/
theorem m_four_sufficient_not_necessary :
  (∃ m : ℝ, m ≠ 4 ∧ are_perpendicular (line1 m) (line2 m)) ∧
  are_perpendicular (line1 4) (line2 4) := by
  sorry

end NUMINAMATH_CALUDE_m_four_sufficient_not_necessary_l3626_362678


namespace NUMINAMATH_CALUDE_impossibility_of_distinct_differences_l3626_362659

theorem impossibility_of_distinct_differences : ¬ ∃ (a : Fin 2010 → Fin 2010),
  Function.Injective a ∧ 
  (∀ (i j : Fin 2010), i ≠ j → |a i - i| ≠ |a j - j|) :=
by sorry

end NUMINAMATH_CALUDE_impossibility_of_distinct_differences_l3626_362659


namespace NUMINAMATH_CALUDE_expression_defined_iff_l3626_362614

theorem expression_defined_iff (x : ℝ) :
  (∃ y : ℝ, y = (Real.log (3 - x)) / Real.sqrt (x - 1)) ↔ 1 < x ∧ x < 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_defined_iff_l3626_362614


namespace NUMINAMATH_CALUDE_cans_of_frosting_needed_l3626_362681

/-- The number of cans of frosting Bob needs to frost the remaining cakes -/
theorem cans_of_frosting_needed (cakes_per_day : ℕ) (days : ℕ) (cakes_eaten : ℕ) (cans_per_cake : ℕ) : 
  cakes_per_day = 10 → days = 5 → cakes_eaten = 12 → cans_per_cake = 2 →
  (cakes_per_day * days - cakes_eaten) * cans_per_cake = 76 := by sorry

end NUMINAMATH_CALUDE_cans_of_frosting_needed_l3626_362681


namespace NUMINAMATH_CALUDE_parabola_hyperbola_focus_l3626_362642

-- Define the parabola
def parabola (p : ℝ) (x y : ℝ) : Prop := y^2 = 2 * p * x

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - y^2 / 3 = 1

-- Define the directrix of the parabola
def directrix (p : ℝ) (x : ℝ) : Prop := x = -p / 2

-- Define the left focus of the hyperbola
def left_focus_hyperbola (x y : ℝ) : Prop := x = -2 ∧ y = 0

-- Theorem statement
theorem parabola_hyperbola_focus (p : ℝ) :
  (∃ x y : ℝ, directrix p x ∧ left_focus_hyperbola x y) →
  p = 4 := by sorry

end NUMINAMATH_CALUDE_parabola_hyperbola_focus_l3626_362642


namespace NUMINAMATH_CALUDE_complex_fraction_calculation_l3626_362695

theorem complex_fraction_calculation : 
  ((5 / 8 : ℚ) * (3 / 7) - (2 / 3) * (1 / 4)) * ((7 / 9 : ℚ) * (2 / 5) * (1 / 2) * 5040) = 79 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_calculation_l3626_362695


namespace NUMINAMATH_CALUDE_sin_cos_sum_10_50_l3626_362618

theorem sin_cos_sum_10_50 : 
  Real.sin (10 * π / 180) * Real.cos (50 * π / 180) + 
  Real.cos (10 * π / 180) * Real.sin (50 * π / 180) = 
  Real.sqrt 3 / 2 := by sorry

end NUMINAMATH_CALUDE_sin_cos_sum_10_50_l3626_362618
