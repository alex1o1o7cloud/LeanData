import Mathlib

namespace NUMINAMATH_CALUDE_field_ratio_l3869_386919

/-- Proves that a rectangular field with perimeter 360 meters and width 75 meters has a length-to-width ratio of 7:5 -/
theorem field_ratio (perimeter width : ℝ) (h_perimeter : perimeter = 360) (h_width : width = 75) :
  let length := (perimeter - 2 * width) / 2
  (length / width) = 7 / 5 := by
  sorry

end NUMINAMATH_CALUDE_field_ratio_l3869_386919


namespace NUMINAMATH_CALUDE_triangle_side_length_l3869_386932

theorem triangle_side_length (a b c : ℝ) (A B C : ℝ) :
  0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2 →  -- acute triangle
  a = 4 →
  b = 5 →
  (1/2) * a * b * Real.sin C = 5 * Real.sqrt 3 →  -- area condition
  c = Real.sqrt 21 :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l3869_386932


namespace NUMINAMATH_CALUDE_parallel_line_length_l3869_386916

theorem parallel_line_length (base : ℝ) (h1 : base = 24) : 
  ∃ (parallel_line : ℝ), 
    parallel_line^2 / base^2 = 1/2 ∧ 
    parallel_line = 12 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_line_length_l3869_386916


namespace NUMINAMATH_CALUDE_sequence_sum_l3869_386906

theorem sequence_sum : 
  let seq := [3, 15, 27, 53, 65, 17, 29, 41, 71, 83]
  List.sum seq = 404 := by
sorry

end NUMINAMATH_CALUDE_sequence_sum_l3869_386906


namespace NUMINAMATH_CALUDE_inequality_for_positive_integers_l3869_386929

theorem inequality_for_positive_integers (n : ℕ) (h : n > 0) : 2 * n - 1 < (n + 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_for_positive_integers_l3869_386929


namespace NUMINAMATH_CALUDE_min_perimeter_triangle_l3869_386918

theorem min_perimeter_triangle (a b c : ℕ) (h_integer : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_cosA : Real.cos (Real.arccos ((b^2 + c^2 - a^2) / (2 * b * c))) = 11/16)
  (h_cosB : Real.cos (Real.arccos ((a^2 + c^2 - b^2) / (2 * a * c))) = 7/8)
  (h_cosC : Real.cos (Real.arccos ((a^2 + b^2 - c^2) / (2 * a * b))) = -1/4)
  (h_triangle : a + b > c ∧ b + c > a ∧ a + c > b) : 
  a + b + c ≥ 9 := by
  sorry

#check min_perimeter_triangle

end NUMINAMATH_CALUDE_min_perimeter_triangle_l3869_386918


namespace NUMINAMATH_CALUDE_mo_hot_chocolate_cups_l3869_386968

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

end NUMINAMATH_CALUDE_mo_hot_chocolate_cups_l3869_386968


namespace NUMINAMATH_CALUDE_car_speed_is_45_l3869_386942

/-- Represents the scenario of a car and motorcyclist journey --/
structure Journey where
  distance : ℝ  -- Distance from A to B in km
  moto_speed : ℝ  -- Motorcyclist's speed in km/h
  delay : ℝ  -- Delay before motorcyclist starts in hours
  car_speed : ℝ  -- Car's speed in km/h (to be proven)

/-- Theorem stating that under given conditions, the car's speed is 45 km/h --/
theorem car_speed_is_45 (j : Journey) 
  (h1 : j.distance = 82.5)
  (h2 : j.moto_speed = 60)
  (h3 : j.delay = 1/3)
  (h4 : ∃ t : ℝ, 
    t > 0 ∧ 
    j.car_speed * (t + j.delay) = j.moto_speed * t ∧ 
    (j.distance - j.moto_speed * t) / j.car_speed = t / 2) :
  j.car_speed = 45 := by
sorry


end NUMINAMATH_CALUDE_car_speed_is_45_l3869_386942


namespace NUMINAMATH_CALUDE_factors_with_more_than_three_factors_l3869_386937

def number_to_factor := 2550

-- Function to count factors of a number
def count_factors (n : ℕ) : ℕ := sorry

-- Function to count numbers with more than 3 factors
def count_numbers_with_more_than_three_factors (n : ℕ) : ℕ := sorry

theorem factors_with_more_than_three_factors :
  count_numbers_with_more_than_three_factors number_to_factor = 9 := by sorry

end NUMINAMATH_CALUDE_factors_with_more_than_three_factors_l3869_386937


namespace NUMINAMATH_CALUDE_complex_purely_imaginary_l3869_386907

theorem complex_purely_imaginary (a b : ℝ) :
  (∃ (z : ℂ), z = Complex.I * a + b ∧ z.re = 0 ∧ z.im ≠ 0) ↔ (a ≠ 0 ∧ b = 0) := by
  sorry

end NUMINAMATH_CALUDE_complex_purely_imaginary_l3869_386907


namespace NUMINAMATH_CALUDE_train_speed_l3869_386963

/-- Calculate the speed of a train given its length and time to pass a stationary point -/
theorem train_speed (length time : ℝ) (length_positive : length > 0) (time_positive : time > 0) :
  length = 100 ∧ time = 5 → length / time = 20 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l3869_386963


namespace NUMINAMATH_CALUDE_girls_equal_barefoot_children_l3869_386909

/-- Given a lawn with boys and girls, some of whom are barefoot and some wearing shoes,
    prove that the number of girls equals the number of barefoot children
    when the number of barefoot boys equals the number of girls with shoes. -/
theorem girls_equal_barefoot_children
  (num_barefoot_boys : ℕ)
  (num_girls_with_shoes : ℕ)
  (num_barefoot_girls : ℕ)
  (h : num_barefoot_boys = num_girls_with_shoes) :
  num_girls_with_shoes + num_barefoot_girls = num_barefoot_boys + num_barefoot_girls :=
by sorry

end NUMINAMATH_CALUDE_girls_equal_barefoot_children_l3869_386909


namespace NUMINAMATH_CALUDE_flour_needed_for_loaves_l3869_386992

/-- The number of cups of flour needed for one loaf of bread -/
def flour_per_loaf : ℝ := 2.5

/-- The number of loaves of bread to be baked -/
def number_of_loaves : ℕ := 2

/-- Theorem: The total number of cups of flour needed for baking the desired number of loaves is 5 -/
theorem flour_needed_for_loaves : flour_per_loaf * (number_of_loaves : ℝ) = 5 := by
  sorry

end NUMINAMATH_CALUDE_flour_needed_for_loaves_l3869_386992


namespace NUMINAMATH_CALUDE_solution_set_part_i_solution_set_part_ii_l3869_386901

-- Define the function f
def f (x m : ℝ) : ℝ := |x + 1| + |x - 2| - m

-- Part I
theorem solution_set_part_i :
  ∀ x : ℝ, f x 5 > 0 ↔ x > 3 ∨ x < -2 :=
sorry

-- Part II
theorem solution_set_part_ii :
  ∀ m : ℝ, (∀ x : ℝ, f x m ≥ 2) ↔ m ≤ 1 :=
sorry

end NUMINAMATH_CALUDE_solution_set_part_i_solution_set_part_ii_l3869_386901


namespace NUMINAMATH_CALUDE_find_cd_l3869_386983

def repeating_decimal (c d : ℕ) : ℚ :=
  1 + (10 * c + d : ℚ) / 99

theorem find_cd (c d : ℕ) (h_c : c < 10) (h_d : d < 10) : 
  42 * (repeating_decimal c d - (1 + (10 * c + d : ℚ) / 100)) = 4/5 → 
  c = 1 ∧ d = 9 := by
sorry

end NUMINAMATH_CALUDE_find_cd_l3869_386983


namespace NUMINAMATH_CALUDE_train_crossing_time_l3869_386999

/-- The time taken for a faster train to cross a man in a slower train -/
theorem train_crossing_time (faster_speed slower_speed : ℝ) (train_length : ℝ) : 
  faster_speed = 72 → 
  slower_speed = 36 → 
  train_length = 100 → 
  (train_length / ((faster_speed - slower_speed) * (5/18))) = 10 := by
  sorry

end NUMINAMATH_CALUDE_train_crossing_time_l3869_386999


namespace NUMINAMATH_CALUDE_square_root_of_36_l3869_386982

theorem square_root_of_36 : ∃ x : ℝ, x ^ 2 = 36 ∧ (x = 6 ∨ x = -6) := by
  sorry

end NUMINAMATH_CALUDE_square_root_of_36_l3869_386982


namespace NUMINAMATH_CALUDE_quiz_probability_l3869_386920

theorem quiz_probability : 
  let n : ℕ := 6  -- number of questions
  let m : ℕ := 6  -- number of possible answers per question
  let p : ℚ := 1 - (m - 1 : ℚ) / m  -- probability of getting one question right
  1 - (1 - p) ^ n = 31031 / 46656 :=
by sorry

end NUMINAMATH_CALUDE_quiz_probability_l3869_386920


namespace NUMINAMATH_CALUDE_lucy_age_l3869_386960

/-- Given that Lucy's age is three times Helen's age and the sum of their ages is 60,
    prove that Lucy is 45 years old. -/
theorem lucy_age (lucy helen : ℕ) 
  (h1 : lucy = 3 * helen) 
  (h2 : lucy + helen = 60) : 
  lucy = 45 := by
  sorry

end NUMINAMATH_CALUDE_lucy_age_l3869_386960


namespace NUMINAMATH_CALUDE_zero_in_interval_l3869_386948

noncomputable def f (x : ℝ) : ℝ := 6 / x - Real.log x / Real.log 2

theorem zero_in_interval :
  ∃ c ∈ Set.Ioo 2 4, f c = 0 :=
sorry

end NUMINAMATH_CALUDE_zero_in_interval_l3869_386948


namespace NUMINAMATH_CALUDE_interval_length_implies_difference_l3869_386991

theorem interval_length_implies_difference (a b : ℝ) : 
  (∃ x, 3 * a ≤ 4 * x + 6 ∧ 4 * x + 6 ≤ 3 * b) → 
  ((3 * b - 6) / 4 - (3 * a - 6) / 4 = 15) → 
  b - a = 20 := by
sorry

end NUMINAMATH_CALUDE_interval_length_implies_difference_l3869_386991


namespace NUMINAMATH_CALUDE_problem_solution_l3869_386939

theorem problem_solution : (-15) / (1/3 - 3 - 3/2) * 6 = 108/5 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3869_386939


namespace NUMINAMATH_CALUDE_sqrt_two_sqrt_three_minus_three_equals_fourth_root_difference_l3869_386914

theorem sqrt_two_sqrt_three_minus_three_equals_fourth_root_difference : 
  Real.sqrt (2 * Real.sqrt 3 - 3) = (27/4)^(1/4) - (3/4)^(1/4) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_two_sqrt_three_minus_three_equals_fourth_root_difference_l3869_386914


namespace NUMINAMATH_CALUDE_number_of_ones_l3869_386994

theorem number_of_ones (n : ℕ) (hn : n = 999999999) : 
  ∃ x : ℤ, (n : ℤ) * x = (10^81 - 1) / 9 := by
  sorry

end NUMINAMATH_CALUDE_number_of_ones_l3869_386994


namespace NUMINAMATH_CALUDE_additive_inverse_problem_l3869_386938

theorem additive_inverse_problem (m : ℤ) : (m + 1) + (-2) = 0 → m = 1 := by
  sorry

end NUMINAMATH_CALUDE_additive_inverse_problem_l3869_386938


namespace NUMINAMATH_CALUDE_son_age_is_30_l3869_386955

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

end NUMINAMATH_CALUDE_son_age_is_30_l3869_386955


namespace NUMINAMATH_CALUDE_quadratic_rewrite_l3869_386986

theorem quadratic_rewrite (a b c : ℤ) :
  (∀ x : ℝ, 16 * x^2 - 40 * x - 72 = (a * x + b)^2 + c) →
  a * b = -20 := by
sorry

end NUMINAMATH_CALUDE_quadratic_rewrite_l3869_386986


namespace NUMINAMATH_CALUDE_number_equation_solution_l3869_386922

theorem number_equation_solution : 
  ∃ n : ℝ, (n * n - 30158 * 30158) / (n - 30158) = 100000 ∧ n = 69842 := by
  sorry

end NUMINAMATH_CALUDE_number_equation_solution_l3869_386922


namespace NUMINAMATH_CALUDE_school_classrooms_problem_l3869_386902

theorem school_classrooms_problem (original_desks new_desks new_classrooms : ℕ) 
  (h1 : original_desks = 539)
  (h2 : new_desks = 1080)
  (h3 : new_classrooms = 9)
  (h4 : ∃ (original_classrooms : ℕ), original_classrooms > 0 ∧ original_desks % original_classrooms = 0)
  (h5 : ∃ (current_classrooms : ℕ), current_classrooms = original_classrooms + new_classrooms)
  (h6 : ∃ (new_desks_per_classroom : ℕ), new_desks_per_classroom > 0 ∧ new_desks % current_classrooms = 0)
  (h7 : ∀ (original_desks_per_classroom : ℕ), 
    original_desks_per_classroom > 0 → 
    original_desks = original_classrooms * original_desks_per_classroom → 
    new_desks_per_classroom > original_desks_per_classroom) :
  current_classrooms = 20 :=
sorry

end NUMINAMATH_CALUDE_school_classrooms_problem_l3869_386902


namespace NUMINAMATH_CALUDE_ledi_age_in_future_l3869_386980

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

end NUMINAMATH_CALUDE_ledi_age_in_future_l3869_386980


namespace NUMINAMATH_CALUDE_max_t_and_solution_set_l3869_386912

open Real

noncomputable def f (x : ℝ) := 9 / (sin x)^2 + 4 / (cos x)^2

theorem max_t_and_solution_set :
  (∃ (t : ℝ), ∀ (x : ℝ), 0 < x ∧ x < π/2 → f x ≥ t) ∧
  (∀ (x : ℝ), 0 < x ∧ x < π/2 → f x ≥ 25) ∧
  (∀ (t : ℝ), (∀ (x : ℝ), 0 < x ∧ x < π/2 → f x ≥ t) → t ≤ 25) ∧
  ({x : ℝ | |x + 5| + |2*x - 1| ≤ 6} = {x : ℝ | 0 ≤ x ∧ x ≤ 2/3}) := by
  sorry

end NUMINAMATH_CALUDE_max_t_and_solution_set_l3869_386912


namespace NUMINAMATH_CALUDE_fifth_equation_in_pattern_l3869_386947

theorem fifth_equation_in_pattern : 
  (∀ n : ℕ, n ≥ 1 ∧ n ≤ 4 → 
    (List.range n).sum + (List.range n).sum.succ = n^2) →
  (List.range 5).sum + (List.range 5).sum.succ = 81 :=
sorry

end NUMINAMATH_CALUDE_fifth_equation_in_pattern_l3869_386947


namespace NUMINAMATH_CALUDE_largest_inscribed_circle_circumference_l3869_386925

/-- The circumference of the largest circle inscribed in a square -/
theorem largest_inscribed_circle_circumference (s : ℝ) (h : s = 12) :
  2 * s * Real.pi = 24 * Real.pi := by sorry

end NUMINAMATH_CALUDE_largest_inscribed_circle_circumference_l3869_386925


namespace NUMINAMATH_CALUDE_equation_solution_l3869_386931

theorem equation_solution : ∃! y : ℝ, 4 + 2.3 * y = 1.7 * y - 20 :=
by
  use -40
  constructor
  · -- Prove that y = -40 satisfies the equation
    sorry
  · -- Prove uniqueness
    sorry

end NUMINAMATH_CALUDE_equation_solution_l3869_386931


namespace NUMINAMATH_CALUDE_blended_number_property_l3869_386967

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

end NUMINAMATH_CALUDE_blended_number_property_l3869_386967


namespace NUMINAMATH_CALUDE_inequality_proof_l3869_386957

theorem inequality_proof (x y z : ℝ) : 
  (x^2 + 2*y^2 + 2*z^2)/(x^2 + y*z) + (y^2 + 2*z^2 + 2*x^2)/(y^2 + z*x) + (z^2 + 2*x^2 + 2*y^2)/(z^2 + x*y) > 6 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3869_386957


namespace NUMINAMATH_CALUDE_probability_two_black_cards_l3869_386910

theorem probability_two_black_cards (total_cards : ℕ) (black_cards : ℕ) 
  (h1 : total_cards = 52) 
  (h2 : black_cards = 26) :
  (black_cards * (black_cards - 1)) / (total_cards * (total_cards - 1)) = 25 / 102 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_black_cards_l3869_386910


namespace NUMINAMATH_CALUDE_sally_total_spent_l3869_386987

/-- The amount Sally paid for peaches after applying a coupon -/
def peaches_price : ℚ := 12.32

/-- The amount of the coupon applied to the peaches purchase -/
def coupon_amount : ℚ := 3

/-- The amount Sally paid for cherries -/
def cherries_price : ℚ := 11.54

/-- The theorem stating that the total amount Sally spent is $23.86 -/
theorem sally_total_spent : peaches_price + cherries_price = 23.86 := by
  sorry

end NUMINAMATH_CALUDE_sally_total_spent_l3869_386987


namespace NUMINAMATH_CALUDE_triangle_area_from_circumradius_side_angle_l3869_386927

/-- The area of a triangle given its circumradius, one side, and one angle. -/
theorem triangle_area_from_circumradius_side_angle 
  (R a β : ℝ) (h_R : R > 0) (h_a : a > 0) (h_β : 0 < β ∧ β < π) : 
  ∃ (t : ℝ), t = (a^2 * Real.sin (2*β) / 4) + (a * Real.sin β^2 / 2) * Real.sqrt (4*R^2 - a^2) :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_from_circumradius_side_angle_l3869_386927


namespace NUMINAMATH_CALUDE_license_plate_combinations_l3869_386974

def alphabet_size : ℕ := 26
def letter_positions : ℕ := 4
def odd_digits : ℕ := 5

theorem license_plate_combinations :
  (Nat.choose alphabet_size 2) * (Nat.choose letter_positions 2) * (odd_digits * (odd_digits - 1)) = 39000 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_combinations_l3869_386974


namespace NUMINAMATH_CALUDE_sum_of_max_min_xyz_l3869_386977

theorem sum_of_max_min_xyz (x y z : ℝ) 
  (h1 : x ≥ y) (h2 : y ≥ z) (h3 : z ≥ 0) 
  (h4 : 6*x + 5*y + 4*z = 120) : 
  ∃ (max_sum min_sum : ℝ), 
    (∀ (a b c : ℝ), a ≥ b → b ≥ c → c ≥ 0 → 6*a + 5*b + 4*c = 120 → a + b + c ≤ max_sum) ∧
    (∀ (a b c : ℝ), a ≥ b → b ≥ c → c ≥ 0 → 6*a + 5*b + 4*c = 120 → a + b + c ≥ min_sum) ∧
    max_sum + min_sum = 44 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_max_min_xyz_l3869_386977


namespace NUMINAMATH_CALUDE_fermat_500_units_digit_l3869_386908

def fermat (n : ℕ) : ℕ := 2^(2^n) + 1

theorem fermat_500_units_digit :
  fermat 500 % 10 = 7 := by sorry

end NUMINAMATH_CALUDE_fermat_500_units_digit_l3869_386908


namespace NUMINAMATH_CALUDE_reciprocal_of_one_third_l3869_386944

theorem reciprocal_of_one_third (x : ℚ) : x * (1/3) = 1 → x = 3 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_of_one_third_l3869_386944


namespace NUMINAMATH_CALUDE_equation_set_solution_inequality_set_solution_l3869_386979

-- Equation set
theorem equation_set_solution :
  ∃! (x y : ℝ), x - y - 1 = 4 ∧ 4 * (x - y) - y = 5 ∧ x = 20 ∧ y = 15 := by sorry

-- Inequality set
theorem inequality_set_solution :
  ∀ x : ℝ, (4 * x - 1 ≥ x + 1 ∧ (1 - x) / 2 < x) ↔ x ≥ 2/3 := by sorry

end NUMINAMATH_CALUDE_equation_set_solution_inequality_set_solution_l3869_386979


namespace NUMINAMATH_CALUDE_no_real_solutions_cube_root_equation_l3869_386989

theorem no_real_solutions_cube_root_equation :
  ¬∃ x : ℝ, (x ^ (1/3 : ℝ)) = 15 / (6 - x ^ (1/3 : ℝ)) := by
  sorry

end NUMINAMATH_CALUDE_no_real_solutions_cube_root_equation_l3869_386989


namespace NUMINAMATH_CALUDE_balloon_arrangements_l3869_386962

def word_length : ℕ := 7
def repeating_letters : ℕ := 2
def repetitions_per_letter : ℕ := 2

theorem balloon_arrangements :
  (word_length.factorial) / ((repetitions_per_letter.factorial) ^ repeating_letters) = 1260 := by
  sorry

end NUMINAMATH_CALUDE_balloon_arrangements_l3869_386962


namespace NUMINAMATH_CALUDE_no_valid_base_l3869_386975

theorem no_valid_base : ¬ ∃ (base : ℝ), (1/5)^35 * (1/4)^18 = 1/(2*(base^35)) := by
  sorry

end NUMINAMATH_CALUDE_no_valid_base_l3869_386975


namespace NUMINAMATH_CALUDE_train_boggies_count_l3869_386993

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

end NUMINAMATH_CALUDE_train_boggies_count_l3869_386993


namespace NUMINAMATH_CALUDE_snail_return_time_is_whole_hours_l3869_386913

/-- Represents the movement of a snail on a plane -/
structure SnailMovement where
  speed : ℝ
  turnInterval : ℝ
  turnAngle : ℝ

/-- Represents the time taken for the snail to return to its starting point -/
def returnTime (movement : SnailMovement) : ℝ := sorry

/-- Theorem stating that the return time is always an integer multiple of hours -/
theorem snail_return_time_is_whole_hours (movement : SnailMovement) 
  (h1 : movement.speed > 0)
  (h2 : movement.turnInterval = 0.25) -- 15 minutes = 0.25 hours
  (h3 : movement.turnAngle = π / 2) -- right angle
  : ∃ n : ℕ, returnTime movement = n := by sorry

end NUMINAMATH_CALUDE_snail_return_time_is_whole_hours_l3869_386913


namespace NUMINAMATH_CALUDE_f_difference_l3869_386905

/-- The function f(x) = x^4 + 2x^3 + 3x^2 + 7x -/
def f (x : ℝ) : ℝ := x^4 + 2*x^3 + 3*x^2 + 7*x

/-- Theorem: f(3) - f(-3) = 150 -/
theorem f_difference : f 3 - f (-3) = 150 := by sorry

end NUMINAMATH_CALUDE_f_difference_l3869_386905


namespace NUMINAMATH_CALUDE_ratio_problem_l3869_386954

theorem ratio_problem (ratio_percent : ℚ) (first_part : ℚ) (second_part : ℚ) :
  ratio_percent = 200 / 3 →
  first_part = 2 →
  first_part / second_part = ratio_percent / 100 →
  second_part = 3 := by
sorry

end NUMINAMATH_CALUDE_ratio_problem_l3869_386954


namespace NUMINAMATH_CALUDE_min_magnitude_u_l3869_386988

/-- The minimum magnitude of vector u -/
theorem min_magnitude_u (a b : ℝ × ℝ) (h1 : a = (Real.cos (25 * π / 180), Real.sin (25 * π / 180)))
  (h2 : b = (Real.sin (20 * π / 180), Real.cos (20 * π / 180))) :
  (∃ (t : ℝ), ∀ (s : ℝ), ‖a + s • b‖ ≥ ‖a + t • b‖) ∧
  (∃ (u : ℝ × ℝ), ∃ (t : ℝ), u = a + t • b ∧ ‖u‖ = Real.sqrt 2 / 2) :=
sorry

end NUMINAMATH_CALUDE_min_magnitude_u_l3869_386988


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocals_l3869_386923

theorem min_value_sum_reciprocals (x y z : ℝ) 
  (pos_x : 0 < x) (pos_y : 0 < y) (pos_z : 0 < z) 
  (sum_eq_3 : x + y + z = 3) : 
  1 / (x + 3*y) + 1 / (y + 3*z) + 1 / (z + 3*x) ≥ 3/4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_reciprocals_l3869_386923


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_l3869_386978

theorem sum_of_roots_quadratic (x₁ x₂ : ℝ) : 
  (2 * x₁^2 + 6 * x₁ - 1 = 0) → 
  (2 * x₂^2 + 6 * x₂ - 1 = 0) → 
  (x₁ + x₂ = -3) := by
sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_l3869_386978


namespace NUMINAMATH_CALUDE_rationalize_denominator_l3869_386921

theorem rationalize_denominator :
  ∃ (A B C D E F : ℚ),
    (1 : ℝ) / (Real.sqrt 2 + Real.sqrt 5 + Real.sqrt 7) =
    (A * Real.sqrt 2 + B * Real.sqrt 5 + C * Real.sqrt 7 + D * Real.sqrt E) / F ∧
    A = 5 ∧ B = 4 ∧ C = -1 ∧ D = 1 ∧ E = 70 ∧ F = 20 ∧ F > 0 :=
by sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l3869_386921


namespace NUMINAMATH_CALUDE_three_face_painted_count_l3869_386941

/-- Represents a cuboid made of small cubes -/
structure Cuboid where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Represents the state of the cuboid after modifications -/
structure ModifiedCuboid extends Cuboid where
  removed_cubes : ℕ
  surface_painted : Bool

/-- Counts the number of small cubes with three painted faces -/
def count_three_face_painted (c : ModifiedCuboid) : ℕ :=
  sorry

/-- The main theorem to prove -/
theorem three_face_painted_count 
  (c : ModifiedCuboid) 
  (h1 : c.length = 12 ∧ c.width = 3 ∧ c.height = 6)
  (h2 : c.removed_cubes = 3)
  (h3 : c.surface_painted = true) :
  count_three_face_painted c = 8 :=
sorry

end NUMINAMATH_CALUDE_three_face_painted_count_l3869_386941


namespace NUMINAMATH_CALUDE_complex_number_range_l3869_386943

theorem complex_number_range (a : ℝ) (z : ℂ) : 
  z = a + Complex.I ∧ 
  (z.re < 0 ∧ z.im > 0) ∧ 
  Complex.abs (z * (1 + Complex.I)) > 2 → 
  a < -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_range_l3869_386943


namespace NUMINAMATH_CALUDE_thousand_gon_triangles_l3869_386966

/-- Given a polygon with n sides and m internal points, calculates the number of triangles formed when the points are connected to each other and to the vertices of the polygon. -/
def triangles_in_polygon (n : ℕ) (m : ℕ) : ℕ :=
  n + 2 * m - 2

/-- Theorem stating that in a 1000-sided polygon with 500 internal points, 1998 triangles are formed. -/
theorem thousand_gon_triangles :
  triangles_in_polygon 1000 500 = 1998 := by
  sorry

end NUMINAMATH_CALUDE_thousand_gon_triangles_l3869_386966


namespace NUMINAMATH_CALUDE_negation_equivalence_l3869_386900

variable (m : ℤ)

theorem negation_equivalence :
  (¬ ∃ x : ℤ, x^2 + x + m < 0) ↔ (∀ x : ℤ, x^2 + x + m ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l3869_386900


namespace NUMINAMATH_CALUDE_sum_of_squares_l3869_386985

theorem sum_of_squares (a b c : ℝ) : 
  (a * b + b * c + a * c = 131) → (a + b + c = 20) → (a^2 + b^2 + c^2 = 138) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_l3869_386985


namespace NUMINAMATH_CALUDE_water_added_to_tank_l3869_386936

theorem water_added_to_tank (tank_capacity : ℚ) 
  (h1 : tank_capacity = 56)
  (initial_fraction : ℚ) (final_fraction : ℚ)
  (h2 : initial_fraction = 3/4)
  (h3 : final_fraction = 7/8) :
  final_fraction * tank_capacity - initial_fraction * tank_capacity = 7 := by
  sorry

end NUMINAMATH_CALUDE_water_added_to_tank_l3869_386936


namespace NUMINAMATH_CALUDE_complex_number_real_condition_l3869_386964

theorem complex_number_real_condition (m : ℝ) :
  let z : ℂ := m - 3 + (m^2 - 9) * Complex.I
  z.im = 0 → m = 3 ∨ m = -3 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_real_condition_l3869_386964


namespace NUMINAMATH_CALUDE_probability_two_one_is_four_fifths_l3869_386958

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

end NUMINAMATH_CALUDE_probability_two_one_is_four_fifths_l3869_386958


namespace NUMINAMATH_CALUDE_equation_property_l3869_386924

theorem equation_property (a b : ℝ) : 3 * a = 3 * b → a = b := by
  sorry

end NUMINAMATH_CALUDE_equation_property_l3869_386924


namespace NUMINAMATH_CALUDE_distance_polynomial_l3869_386949

theorem distance_polynomial (m n : ℝ) : 
  ∃ (x y : ℝ), x + y = m ∧ x * y = n^2 ∧ 
  (∀ z : ℝ, z^2 - m*z + n^2 = 0 ↔ (z = x ∨ z = y)) := by
sorry

end NUMINAMATH_CALUDE_distance_polynomial_l3869_386949


namespace NUMINAMATH_CALUDE_xiaogang_shooting_probability_l3869_386904

theorem xiaogang_shooting_probability (total_shots : ℕ) (successful_shots : ℕ) 
  (h1 : total_shots = 50) 
  (h2 : successful_shots = 38) : 
  (successful_shots : ℚ) / (total_shots : ℚ) = 0.76 := by
  sorry

end NUMINAMATH_CALUDE_xiaogang_shooting_probability_l3869_386904


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l3869_386950

-- Problem 1
theorem problem_1 : Real.sqrt 8 - (1/2)⁻¹ + 4 * Real.sin (30 * π / 180) = 2 * Real.sqrt 2 := by
  sorry

-- Problem 2
theorem problem_2 : (2^2 - 9) / (2^2 + 6*2 + 9) / (1 - 2 / (2 + 3)) = -1/3 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l3869_386950


namespace NUMINAMATH_CALUDE_arithmetic_sequence_a12_l3869_386971

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

end NUMINAMATH_CALUDE_arithmetic_sequence_a12_l3869_386971


namespace NUMINAMATH_CALUDE_orange_apple_ratio_l3869_386928

/-- Represents the contents of a shopping cart with apples, oranges, and pears. -/
structure ShoppingCart where
  apples : ℕ
  oranges : ℕ
  pears : ℕ

/-- Checks if the shopping cart satisfies the given conditions. -/
def satisfiesConditions (cart : ShoppingCart) : Prop :=
  cart.pears = 4 * cart.oranges ∧
  cart.apples = (1 / 12 : ℚ) * cart.pears

/-- The main theorem stating the relationship between oranges and apples. -/
theorem orange_apple_ratio (cart : ShoppingCart) 
  (h : satisfiesConditions cart) (h_nonzero : cart.apples > 0) : 
  cart.oranges = 3 * cart.apples := by
  sorry


end NUMINAMATH_CALUDE_orange_apple_ratio_l3869_386928


namespace NUMINAMATH_CALUDE_mario_garden_after_two_weeks_l3869_386934

/-- Calculates the number of flowers on a plant after a given number of weeks -/
def flowers_after_weeks (initial : ℕ) (growth_rate : ℕ) (weeks : ℕ) : ℕ :=
  initial + growth_rate * weeks

/-- Calculates the number of flowers on a plant that doubles each week -/
def flowers_doubling (initial : ℕ) (weeks : ℕ) : ℕ :=
  initial * (2^weeks)

/-- Represents Mario's garden and calculates the total number of blossoms -/
def mario_garden (weeks : ℕ) : ℕ :=
  let hibiscus1 := flowers_after_weeks 2 3 weeks
  let hibiscus2 := flowers_after_weeks 4 4 weeks
  let hibiscus3 := flowers_after_weeks 16 5 weeks
  let rose1 := flowers_after_weeks 3 2 weeks
  let rose2 := flowers_after_weeks 5 3 weeks
  let sunflower := flowers_doubling 6 weeks
  hibiscus1 + hibiscus2 + hibiscus3 + rose1 + rose2 + sunflower

theorem mario_garden_after_two_weeks :
  mario_garden 2 = 88 := by
  sorry

end NUMINAMATH_CALUDE_mario_garden_after_two_weeks_l3869_386934


namespace NUMINAMATH_CALUDE_projectile_max_height_l3869_386973

/-- The height function of the projectile -/
def h (t : ℝ) : ℝ := -20 * t^2 + 40 * t + 10

/-- Theorem stating that the maximum height of the projectile is 30 -/
theorem projectile_max_height :
  ∃ (t_max : ℝ), ∀ (t : ℝ), h t ≤ h t_max ∧ h t_max = 30 :=
sorry

end NUMINAMATH_CALUDE_projectile_max_height_l3869_386973


namespace NUMINAMATH_CALUDE_repeating_decimal_ratio_l3869_386970

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

end NUMINAMATH_CALUDE_repeating_decimal_ratio_l3869_386970


namespace NUMINAMATH_CALUDE_square_even_implies_even_sqrt_2_irrational_l3869_386997

-- Definition of even number
def is_even (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k

-- Definition of rational number
def is_rational (x : ℝ) : Prop := ∃ a b : ℤ, b ≠ 0 ∧ x = a / b

-- Theorem 1: If p^2 is even, then p is even
theorem square_even_implies_even (p : ℤ) : is_even (p^2) → is_even p := by sorry

-- Theorem 2: √2 is irrational
theorem sqrt_2_irrational : ¬ is_rational (Real.sqrt 2) := by sorry

end NUMINAMATH_CALUDE_square_even_implies_even_sqrt_2_irrational_l3869_386997


namespace NUMINAMATH_CALUDE_production_days_l3869_386953

/-- Given that:
    1. The average daily production for the past n days was 50 units.
    2. Today's production is 110 units.
    3. The new average including today's production is 55 units.
    Prove that n = 11. -/
theorem production_days (n : ℕ) : (n * 50 + 110) / (n + 1) = 55 → n = 11 := by
  sorry

end NUMINAMATH_CALUDE_production_days_l3869_386953


namespace NUMINAMATH_CALUDE_fertilizer_mixture_problem_l3869_386930

/-- Given two fertilizer solutions, one with unknown percentage P and another with 53%,
    mixed to form 42 liters of 63% solution, where 20 liters of the first solution were used,
    prove that the percentage of fertilizer in the first solution is 74%. -/
theorem fertilizer_mixture_problem (P : ℝ) : 
  (20 * P / 100 + 22 * 53 / 100 = 42 * 63 / 100) → P = 74 := by
  sorry

end NUMINAMATH_CALUDE_fertilizer_mixture_problem_l3869_386930


namespace NUMINAMATH_CALUDE_isosceles_triangle_base_length_l3869_386946

/-- An isosceles triangle with congruent sides of length 8 cm and perimeter 27 cm has a base of length 11 cm. -/
theorem isosceles_triangle_base_length :
  ∀ (base : ℝ),
  base > 0 →
  8 > 0 →
  8 + 8 + base = 27 →
  base = 11 :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_base_length_l3869_386946


namespace NUMINAMATH_CALUDE_spending_difference_is_30_l3869_386990

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

end NUMINAMATH_CALUDE_spending_difference_is_30_l3869_386990


namespace NUMINAMATH_CALUDE_rental_company_fixed_amount_l3869_386926

/-- The fixed amount charged by the first rental company -/
def F : ℝ := 41.95

/-- The per-mile rate charged by the first rental company -/
def rate1 : ℝ := 0.29

/-- The fixed amount charged by City Rentals -/
def fixed2 : ℝ := 38.95

/-- The per-mile rate charged by City Rentals -/
def rate2 : ℝ := 0.31

/-- The number of miles driven -/
def miles : ℝ := 150.0

theorem rental_company_fixed_amount :
  F + rate1 * miles = fixed2 + rate2 * miles :=
sorry

end NUMINAMATH_CALUDE_rental_company_fixed_amount_l3869_386926


namespace NUMINAMATH_CALUDE_unique_A_for_3AA1_multiple_of_9_l3869_386998

def is_multiple_of_9 (n : ℕ) : Prop := ∃ k : ℕ, n = 9 * k

def four_digit_3AA1 (A : ℕ) : ℕ := 3000 + 100 * A + 10 * A + 1

theorem unique_A_for_3AA1_multiple_of_9 :
  ∃! A : ℕ, A < 10 ∧ is_multiple_of_9 (four_digit_3AA1 A) ∧ A = 7 := by
sorry

end NUMINAMATH_CALUDE_unique_A_for_3AA1_multiple_of_9_l3869_386998


namespace NUMINAMATH_CALUDE_clara_age_l3869_386952

def anna_age : ℕ := 54
def years_ago : ℕ := 41

theorem clara_age : ℕ :=
  let anna_age_then := anna_age - years_ago
  let clara_age_then := 3 * anna_age_then
  clara_age_then + years_ago

#check clara_age

end NUMINAMATH_CALUDE_clara_age_l3869_386952


namespace NUMINAMATH_CALUDE_books_loaned_out_is_125_l3869_386976

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

end NUMINAMATH_CALUDE_books_loaned_out_is_125_l3869_386976


namespace NUMINAMATH_CALUDE_largest_negative_integer_l3869_386996

theorem largest_negative_integer :
  ∀ n : ℤ, n < 0 → n ≤ -1 :=
sorry

end NUMINAMATH_CALUDE_largest_negative_integer_l3869_386996


namespace NUMINAMATH_CALUDE_circle_radius_l3869_386972

theorem circle_radius (k : ℝ) : 
  (∀ x y : ℝ, x^2 - 8*x + y^2 + 10*y + k = 0 ↔ (x - 4)^2 + (y + 5)^2 = 5^2) → 
  k = 16 := by
sorry

end NUMINAMATH_CALUDE_circle_radius_l3869_386972


namespace NUMINAMATH_CALUDE_ellipse_m_range_l3869_386965

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

end NUMINAMATH_CALUDE_ellipse_m_range_l3869_386965


namespace NUMINAMATH_CALUDE_sqrt_50_plus_sqrt_32_l3869_386940

theorem sqrt_50_plus_sqrt_32 : Real.sqrt 50 + Real.sqrt 32 = 9 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_50_plus_sqrt_32_l3869_386940


namespace NUMINAMATH_CALUDE_least_faces_triangular_pyramid_l3869_386995

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

end NUMINAMATH_CALUDE_least_faces_triangular_pyramid_l3869_386995


namespace NUMINAMATH_CALUDE_algebraic_expression_evaluation_l3869_386933

theorem algebraic_expression_evaluation :
  ∀ (a b : ℝ), 
  (2 * a * (-1)^3 - 3 * b * (-1) + 8 = 18) → 
  (9 * b - 6 * a + 2 = 32) := by
sorry

end NUMINAMATH_CALUDE_algebraic_expression_evaluation_l3869_386933


namespace NUMINAMATH_CALUDE_helmet_sales_and_pricing_l3869_386984

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

end NUMINAMATH_CALUDE_helmet_sales_and_pricing_l3869_386984


namespace NUMINAMATH_CALUDE_transformer_load_calculation_l3869_386945

/-- Calculates the minimum current load for a transformer given the number of units,
    running current per unit, and the starting current multiplier. -/
def minTransformerLoad (numUnits : ℕ) (runningCurrent : ℕ) (startingMultiplier : ℕ) : ℕ :=
  numUnits * (startingMultiplier * runningCurrent)

theorem transformer_load_calculation :
  let numUnits : ℕ := 3
  let runningCurrent : ℕ := 40
  let startingMultiplier : ℕ := 2
  minTransformerLoad numUnits runningCurrent startingMultiplier = 240 := by
  sorry

#eval minTransformerLoad 3 40 2

end NUMINAMATH_CALUDE_transformer_load_calculation_l3869_386945


namespace NUMINAMATH_CALUDE_pencil_pen_multiple_l3869_386981

theorem pencil_pen_multiple (total : ℕ) (pens : ℕ) (M : ℕ) : 
  total = 108 →
  pens = 16 →
  total = pens + (M * pens + 12) →
  M = 5 := by
sorry

end NUMINAMATH_CALUDE_pencil_pen_multiple_l3869_386981


namespace NUMINAMATH_CALUDE_work_completion_time_l3869_386951

/-- Given that A can do a piece of work in 12 days and B is 20% more efficient than A,
    prove that B will complete the same work in 10 days. -/
theorem work_completion_time (work : ℝ) (a_time b_time : ℝ) : 
  work > 0 → 
  a_time = 12 → 
  b_time = work / ((work / a_time) * 1.2) → 
  b_time = 10 :=
by sorry

end NUMINAMATH_CALUDE_work_completion_time_l3869_386951


namespace NUMINAMATH_CALUDE_calculate_expression_l3869_386935

theorem calculate_expression : 8 * (5 + 2/5) - 3 = 40.2 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l3869_386935


namespace NUMINAMATH_CALUDE_truck_speed_problem_l3869_386961

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

end NUMINAMATH_CALUDE_truck_speed_problem_l3869_386961


namespace NUMINAMATH_CALUDE_sqrt_less_than_5x_iff_l3869_386911

theorem sqrt_less_than_5x_iff (x : ℝ) (h : x > 0) :
  Real.sqrt x < 5 * x ↔ x > 1 / 25 := by sorry

end NUMINAMATH_CALUDE_sqrt_less_than_5x_iff_l3869_386911


namespace NUMINAMATH_CALUDE_lindsey_bands_count_l3869_386959

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

end NUMINAMATH_CALUDE_lindsey_bands_count_l3869_386959


namespace NUMINAMATH_CALUDE_division_remainder_problem_l3869_386969

theorem division_remainder_problem (dividend : Nat) (divisor : Nat) (quotient : Nat) 
  (h1 : dividend = 122)
  (h2 : divisor = 20)
  (h3 : quotient = 6)
  (h4 : dividend = divisor * quotient + (dividend % divisor)) :
  dividend % divisor = 2 := by
  sorry

end NUMINAMATH_CALUDE_division_remainder_problem_l3869_386969


namespace NUMINAMATH_CALUDE_hcf_of_36_and_84_l3869_386915

theorem hcf_of_36_and_84 : Nat.gcd 36 84 = 12 := by
  sorry

end NUMINAMATH_CALUDE_hcf_of_36_and_84_l3869_386915


namespace NUMINAMATH_CALUDE_tangent_line_property_l3869_386917

/-- Given a line tangent to ln x and e^x, prove that 1/x₁ - 2/(x₂-1) = 1 --/
theorem tangent_line_property (x₁ x₂ : ℝ) (h₁ : x₁ > 0) (h₂ : x₂ > 0) : 
  (∃ (m b : ℝ), 
    (∀ x, m * x + b = (1 / x₁) * x + Real.log x₁ - 1) ∧
    (∀ x, m * x + b = Real.exp x₂ * x - Real.exp x₂ * (x₂ - 1))) →
  1 / x₁ - 2 / (x₂ - 1) = 1 := by
  sorry


end NUMINAMATH_CALUDE_tangent_line_property_l3869_386917


namespace NUMINAMATH_CALUDE_l₃_equation_min_distance_l₁_l₄_l3869_386956

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

end NUMINAMATH_CALUDE_l₃_equation_min_distance_l₁_l₄_l3869_386956


namespace NUMINAMATH_CALUDE_fourth_to_first_class_ratio_l3869_386903

def num_classes : ℕ := 6
def students_first_class : ℕ := 20
def students_second_third_class : ℕ := 25
def students_fifth_sixth_class : ℕ := 28
def total_students : ℕ := 136

theorem fourth_to_first_class_ratio :
  ∃ (students_fourth_class : ℕ),
    students_first_class +
    2 * students_second_third_class +
    students_fourth_class +
    2 * students_fifth_sixth_class = total_students ∧
    students_fourth_class * 2 = students_first_class :=
by
  sorry

end NUMINAMATH_CALUDE_fourth_to_first_class_ratio_l3869_386903
