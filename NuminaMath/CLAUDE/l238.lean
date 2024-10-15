import Mathlib

namespace NUMINAMATH_CALUDE_ratio_equations_solution_l238_23827

theorem ratio_equations_solution (x y z a : ℤ) : 
  (∃ k : ℤ, x = k ∧ y = 4*k ∧ z = 5*k) →
  y = 9*a^2 - 2*a - 8 →
  z = 10*a + 2 →
  a = 5 :=
by sorry

end NUMINAMATH_CALUDE_ratio_equations_solution_l238_23827


namespace NUMINAMATH_CALUDE_cube_and_square_root_problem_l238_23810

theorem cube_and_square_root_problem (a b : ℝ) 
  (h1 : (2*b - 2*a)^(1/3 : ℝ) = -2)
  (h2 : (4*a + 3*b)^(1/2 : ℝ) = 3) :
  a = 3 ∧ b = -1 ∧ (5*a - b)^(1/2 : ℝ) = 4 ∨ (5*a - b)^(1/2 : ℝ) = -4 :=
by sorry

end NUMINAMATH_CALUDE_cube_and_square_root_problem_l238_23810


namespace NUMINAMATH_CALUDE_simplify_and_ratio_l238_23853

theorem simplify_and_ratio : 
  ∀ (k : ℝ), 
  (6 * k + 18) / 6 = k + 3 ∧ 
  ∃ (a b : ℤ), k + 3 = a * k + b ∧ (a : ℝ) / (b : ℝ) = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_ratio_l238_23853


namespace NUMINAMATH_CALUDE_arithmetic_square_root_of_sqrt_16_l238_23890

theorem arithmetic_square_root_of_sqrt_16 : Real.sqrt (Real.sqrt 16) = 2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_square_root_of_sqrt_16_l238_23890


namespace NUMINAMATH_CALUDE_snail_well_depth_l238_23884

/-- The minimum depth of a well that allows a snail to reach the top during the day on the fifth day,
    given its daily climbing and nightly sliding distances. -/
def min_well_depth (day_climb : ℕ) (night_slide : ℕ) : ℕ :=
  (day_climb - night_slide) * 3 + day_climb + 1

/-- Theorem stating the minimum well depth for a snail with specific climbing characteristics. -/
theorem snail_well_depth :
  min_well_depth 110 40 = 321 := by
  sorry

#eval min_well_depth 110 40

end NUMINAMATH_CALUDE_snail_well_depth_l238_23884


namespace NUMINAMATH_CALUDE_train_passing_platform_l238_23804

/-- Calculates the time for a train to pass a platform -/
theorem train_passing_platform (train_length : ℝ) (tree_passing_time : ℝ) (platform_length : ℝ) :
  train_length = 1200 →
  tree_passing_time = 120 →
  platform_length = 1100 →
  (train_length + platform_length) / (train_length / tree_passing_time) = 230 := by
sorry

end NUMINAMATH_CALUDE_train_passing_platform_l238_23804


namespace NUMINAMATH_CALUDE_larger_share_theorem_l238_23861

/-- Given two investments and a total profit, calculates the share of profit for the larger investment -/
def calculate_larger_share (investment1 : ℕ) (investment2 : ℕ) (total_profit : ℕ) : ℕ :=
  let larger_investment := max investment1 investment2
  let total_investment := investment1 + investment2
  (larger_investment * total_profit) / total_investment

theorem larger_share_theorem (investment1 investment2 total_profit : ℕ) 
  (h1 : investment1 = 22500) 
  (h2 : investment2 = 35000) 
  (h3 : total_profit = 13800) :
  calculate_larger_share investment1 investment2 total_profit = 8400 := by
  sorry

#eval calculate_larger_share 22500 35000 13800

end NUMINAMATH_CALUDE_larger_share_theorem_l238_23861


namespace NUMINAMATH_CALUDE_unique_solution_condition_inequality_condition_l238_23851

-- Define the functions f and g
def f (x : ℝ) : ℝ := x^2 - 1
def g (a x : ℝ) : ℝ := a * |x - 1|

-- Theorem for part I
theorem unique_solution_condition (a : ℝ) :
  (∃! x, |f x| = g a x) ↔ a < 0 := by sorry

-- Theorem for part II
theorem inequality_condition (a : ℝ) :
  (∀ x, f x ≥ g a x) ↔ a ≤ -2 := by sorry

end NUMINAMATH_CALUDE_unique_solution_condition_inequality_condition_l238_23851


namespace NUMINAMATH_CALUDE_inscribed_cube_volume_l238_23860

theorem inscribed_cube_volume (large_cube_edge : ℝ) (small_cube_edge : ℝ) 
  (h1 : large_cube_edge = 12)
  (h2 : small_cube_edge * Real.sqrt 3 = large_cube_edge) : 
  small_cube_edge ^ 3 = 192 * Real.sqrt 3 := by
  sorry

#check inscribed_cube_volume

end NUMINAMATH_CALUDE_inscribed_cube_volume_l238_23860


namespace NUMINAMATH_CALUDE_truncated_cone_base_area_l238_23882

-- Define the radii of the three cones
def r₁ : ℝ := 10
def r₂ : ℝ := 15
def r₃ : ℝ := 15

-- Define the radius of the smaller base of the truncated cone
def r : ℝ := 2

-- Theorem statement
theorem truncated_cone_base_area 
  (h₁ : r₁ = 10)
  (h₂ : r₂ = 15)
  (h₃ : r₃ = 15)
  (h₄ : (r₁ + r)^2 = r₁^2 + (r₂ + r - r₁)^2)
  (h₅ : (r₂ + r)^2 = r₂^2 + (r₁ + r₂ - r)^2)
  (h₆ : (r₃ + r)^2 = r₃^2 + (r₁ + r₃ - r)^2) :
  π * r^2 = 4 * π := by sorry

end NUMINAMATH_CALUDE_truncated_cone_base_area_l238_23882


namespace NUMINAMATH_CALUDE_certain_number_proof_l238_23811

theorem certain_number_proof : ∃ (x : ℚ), (2994 / x = 179) → x = 167 / 10 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l238_23811


namespace NUMINAMATH_CALUDE_division_of_fractions_l238_23843

theorem division_of_fractions :
  (-4 / 5) / (8 / 25) = -5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_division_of_fractions_l238_23843


namespace NUMINAMATH_CALUDE_existence_of_pair_l238_23858

theorem existence_of_pair (x : Fin 670 → ℝ)
  (h_positive : ∀ i, 0 < x i)
  (h_less_than_one : ∀ i, x i < 1)
  (h_distinct : ∀ i j, i ≠ j → x i ≠ x j) :
  ∃ i j, i ≠ j ∧ 0 < x i * x j * (x j - x i) ∧ x i * x j * (x j - x i) < 1 / 2007 := by
  sorry

end NUMINAMATH_CALUDE_existence_of_pair_l238_23858


namespace NUMINAMATH_CALUDE_janice_office_floor_l238_23888

/-- The floor number of Janice's office -/
def office_floor : ℕ := 3

/-- The number of times Janice goes up the stairs per day -/
def up_times : ℕ := 5

/-- The number of times Janice goes down the stairs per day -/
def down_times : ℕ := 3

/-- The total number of flights of stairs Janice walks in a day -/
def total_flights : ℕ := 24

theorem janice_office_floor :
  office_floor * (up_times + down_times) = total_flights :=
sorry

end NUMINAMATH_CALUDE_janice_office_floor_l238_23888


namespace NUMINAMATH_CALUDE_smallest_reducible_n_is_correct_l238_23808

/-- The smallest positive integer n for which (n-17)/(6n+8) is non-zero and reducible -/
def smallest_reducible_n : ℕ := 127

/-- A fraction is reducible if the GCD of its numerator and denominator is greater than 1 -/
def is_reducible (n : ℕ) : Prop :=
  Nat.gcd (n - 17) (6 * n + 8) > 1

theorem smallest_reducible_n_is_correct :
  (∀ k : ℕ, k > 0 ∧ k < smallest_reducible_n → ¬(is_reducible k)) ∧
  (smallest_reducible_n > 0) ∧
  (is_reducible smallest_reducible_n) :=
sorry

end NUMINAMATH_CALUDE_smallest_reducible_n_is_correct_l238_23808


namespace NUMINAMATH_CALUDE_wendy_albums_l238_23837

/-- Given a total number of pictures, the number of pictures in the first album,
    and the number of pictures per album in the remaining albums,
    calculate the number of albums created for the remaining pictures. -/
def calculate_remaining_albums (total_pictures : ℕ) (first_album_pictures : ℕ) (pictures_per_album : ℕ) : ℕ :=
  (total_pictures - first_album_pictures) / pictures_per_album

theorem wendy_albums :
  calculate_remaining_albums 79 44 7 = 5 := by
  sorry

end NUMINAMATH_CALUDE_wendy_albums_l238_23837


namespace NUMINAMATH_CALUDE_horner_v2_value_l238_23876

def horner_polynomial (x : ℝ) : ℝ := 12 + 35*x - 8*x^2 + 79*x^3 + 6*x^4 + 5*x^5 + 3*x^6

def v0 : ℝ := 3
def v1 (x : ℝ) : ℝ := v0 * x + 5
def v2 (x : ℝ) : ℝ := v1 x * x + 6

theorem horner_v2_value :
  v2 (-4) = 34 := by sorry

end NUMINAMATH_CALUDE_horner_v2_value_l238_23876


namespace NUMINAMATH_CALUDE_womens_average_age_l238_23839

theorem womens_average_age 
  (n : ℕ) 
  (initial_men : ℕ) 
  (replaced_men_ages : ℕ × ℕ) 
  (age_increase : ℚ) :
  initial_men = 8 →
  replaced_men_ages = (20, 10) →
  age_increase = 2 →
  ∃ (total_age : ℚ),
    (total_age / initial_men + age_increase) * initial_men = 
      total_age - (replaced_men_ages.1 + replaced_men_ages.2) + 46 →
    46 / 2 = 23 :=
by sorry

end NUMINAMATH_CALUDE_womens_average_age_l238_23839


namespace NUMINAMATH_CALUDE_jane_brown_sheets_l238_23867

/-- The number of old, brown sheets of drawing paper Jane has -/
def brown_sheets (total : ℕ) (yellow : ℕ) : ℕ := total - yellow

/-- Proof that Jane has 28 old, brown sheets of drawing paper -/
theorem jane_brown_sheets : brown_sheets 55 27 = 28 := by
  sorry

end NUMINAMATH_CALUDE_jane_brown_sheets_l238_23867


namespace NUMINAMATH_CALUDE_complex_quadratic_roots_l238_23850

theorem complex_quadratic_roots (z : ℂ) : 
  z^2 = -63 + 16*I ∧ (7 + 4*I)^2 = -63 + 16*I → 
  z = 7 + 4*I ∨ z = -7 - 4*I :=
by sorry

end NUMINAMATH_CALUDE_complex_quadratic_roots_l238_23850


namespace NUMINAMATH_CALUDE_factorization_proof_l238_23846

theorem factorization_proof (x : ℝ) : 
  (3 * x^2 - 12 = 3 * (x + 2) * (x - 2)) ∧ 
  (x^2 - 2*x - 8 = (x - 4) * (x + 2)) := by
sorry

end NUMINAMATH_CALUDE_factorization_proof_l238_23846


namespace NUMINAMATH_CALUDE_alyosha_age_claim_possible_l238_23807

-- Define a structure for dates
structure Date :=
  (year : ℕ)
  (month : ℕ)
  (day : ℕ)

-- Define a structure for a person's age and birthday
structure Person :=
  (birthday : Date)
  (current_date : Date)

def age (p : Person) : ℕ :=
  p.current_date.year - p.birthday.year

def is_birthday (p : Person) : Prop :=
  p.birthday.month = p.current_date.month ∧ p.birthday.day = p.current_date.day

-- Define Alyosha
def alyosha (birthday : Date) : Person :=
  { birthday := birthday,
    current_date := ⟨2024, 1, 1⟩ }  -- Assuming current year is 2024

-- Theorem statement
theorem alyosha_age_claim_possible :
  ∃ (birthday : Date),
    age (alyosha birthday) = 11 ∧
    age { birthday := birthday, current_date := ⟨2023, 12, 30⟩ } = 9 ∧
    age { birthday := birthday, current_date := ⟨2025, 1, 1⟩ } = 12 ↔
    birthday = ⟨2013, 12, 31⟩ :=
sorry

end NUMINAMATH_CALUDE_alyosha_age_claim_possible_l238_23807


namespace NUMINAMATH_CALUDE_joan_football_games_l238_23836

/-- Given that Joan went to 4 football games this year and 13 games in total,
    prove that she went to 9 games last year. -/
theorem joan_football_games (games_this_year games_total : ℕ)
  (h1 : games_this_year = 4)
  (h2 : games_total = 13) :
  games_total - games_this_year = 9 := by
  sorry

end NUMINAMATH_CALUDE_joan_football_games_l238_23836


namespace NUMINAMATH_CALUDE_intersection_M_N_l238_23814

def M : Set ℕ := {1, 2, 4, 8}

def N : Set ℕ := {x | ∃ k, x = 2 * k}

theorem intersection_M_N : M ∩ N = {2, 4, 8} := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l238_23814


namespace NUMINAMATH_CALUDE_tan_405_degrees_l238_23857

theorem tan_405_degrees : Real.tan (405 * π / 180) = 1 := by
  sorry

end NUMINAMATH_CALUDE_tan_405_degrees_l238_23857


namespace NUMINAMATH_CALUDE_elizabeth_granola_profit_l238_23865

/-- Calculate Elizabeth's net profit from selling granola bags --/
theorem elizabeth_granola_profit :
  let ingredient_cost_per_bag : ℚ := 3
  let total_bags : ℕ := 20
  let full_price : ℚ := 6
  let full_price_sales : ℕ := 15
  let discounted_price : ℚ := 4
  let discounted_sales : ℕ := 5

  let total_cost : ℚ := ingredient_cost_per_bag * total_bags
  let full_price_revenue : ℚ := full_price * full_price_sales
  let discounted_revenue : ℚ := discounted_price * discounted_sales
  let total_revenue : ℚ := full_price_revenue + discounted_revenue
  let net_profit : ℚ := total_revenue - total_cost

  net_profit = 50 := by sorry

end NUMINAMATH_CALUDE_elizabeth_granola_profit_l238_23865


namespace NUMINAMATH_CALUDE_max_c_value_l238_23845

theorem max_c_value (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h1 : 2 * (a + b) = a * b) (h2 : a + b + c = a * b * c) :
  c ≤ 8 / 15 := by
  sorry

end NUMINAMATH_CALUDE_max_c_value_l238_23845


namespace NUMINAMATH_CALUDE_simultaneous_ringing_l238_23897

/-- The least common multiple of the bell ringing periods -/
def bell_lcm : ℕ := sorry

/-- The time difference in minutes between the first and next simultaneous ringing -/
def time_difference : ℕ := sorry

theorem simultaneous_ringing :
  bell_lcm = lcm 18 (lcm 24 (lcm 30 36)) ∧
  time_difference = bell_lcm ∧
  time_difference = 360 := by sorry

end NUMINAMATH_CALUDE_simultaneous_ringing_l238_23897


namespace NUMINAMATH_CALUDE_carrie_first_day_miles_l238_23812

/-- Represents the four-day trip driven by Carrie -/
structure CarrieTrip where
  day1 : ℕ
  day2 : ℕ
  day3 : ℕ
  day4 : ℕ
  chargeDistance : ℕ
  chargeCount : ℕ

/-- The conditions of Carrie's trip -/
def tripConditions (trip : CarrieTrip) : Prop :=
  trip.day2 = trip.day1 + 124 ∧
  trip.day3 = 159 ∧
  trip.day4 = 189 ∧
  trip.chargeDistance = 106 ∧
  trip.chargeCount = 7 ∧
  trip.day1 + trip.day2 + trip.day3 + trip.day4 = trip.chargeDistance * trip.chargeCount

/-- Theorem stating that Carrie drove 135 miles on the first day -/
theorem carrie_first_day_miles :
  ∀ (trip : CarrieTrip), tripConditions trip → trip.day1 = 135 :=
by sorry

end NUMINAMATH_CALUDE_carrie_first_day_miles_l238_23812


namespace NUMINAMATH_CALUDE_store_bottles_l238_23879

/-- The total number of bottles in a grocery store, given the number of regular and diet soda bottles. -/
def total_bottles (regular_soda : ℕ) (diet_soda : ℕ) : ℕ :=
  regular_soda + diet_soda

/-- Theorem stating that the total number of bottles in the store is 38. -/
theorem store_bottles : total_bottles 30 8 = 38 := by
  sorry

end NUMINAMATH_CALUDE_store_bottles_l238_23879


namespace NUMINAMATH_CALUDE_complex_equation_solution_l238_23835

theorem complex_equation_solution (z : ℂ) :
  (1 - Complex.I) * z = 2 * Complex.I → z = -1 + Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l238_23835


namespace NUMINAMATH_CALUDE_independence_test_distribution_X_expected_value_Y_variance_Y_l238_23830

-- Define the contingency table
def male_noodles : ℕ := 30
def male_rice : ℕ := 25
def female_noodles : ℕ := 20
def female_rice : ℕ := 25
def total_students : ℕ := 100

-- Define the chi-square formula
def chi_square (a b c d : ℕ) : ℚ :=
  let n := a + b + c + d
  (n * (a * d - b * c)^2 : ℚ) / ((a + b) * (c + d) * (a + c) * (b + d))

-- Define the critical value at α = 0.05
def critical_value : ℚ := 3841 / 1000

-- Theorem for independence test
theorem independence_test :
  chi_square male_noodles male_rice female_noodles female_rice < critical_value :=
sorry

-- Define the distribution of X
def prob_X (x : ℕ) : ℚ :=
  match x with
  | 0 => 3 / 10
  | 1 => 3 / 5
  | 2 => 1 / 10
  | _ => 0

-- Theorem for the distribution of X
theorem distribution_X :
  (prob_X 0 + prob_X 1 + prob_X 2 = 1) ∧
  (∀ x, x > 2 → prob_X x = 0) :=
sorry

-- Define Y as a binomial distribution
def p_Y : ℚ := 3 / 5
def n_Y : ℕ := 3

-- Theorems for expected value and variance of Y
theorem expected_value_Y :
  (n_Y : ℚ) * p_Y = 9 / 5 :=
sorry

theorem variance_Y :
  (n_Y : ℚ) * p_Y * (1 - p_Y) = 18 / 25 :=
sorry

end NUMINAMATH_CALUDE_independence_test_distribution_X_expected_value_Y_variance_Y_l238_23830


namespace NUMINAMATH_CALUDE_equidistant_from_axes_l238_23869

/-- A point in the 2D plane is equidistant from both coordinate axes if and only if the square of its x-coordinate equals the square of its y-coordinate. -/
theorem equidistant_from_axes (x y : ℝ) : (|x| = |y|) ↔ (x^2 = y^2) := by sorry

end NUMINAMATH_CALUDE_equidistant_from_axes_l238_23869


namespace NUMINAMATH_CALUDE_intersection_point_y_coordinate_l238_23832

/-- Given that point A is an intersection point of y = ax and y = (4-a)/x with x-coordinate 1,
    prove that the y-coordinate of A is 2. -/
theorem intersection_point_y_coordinate (a : ℝ) :
  (∃ A : ℝ × ℝ, A.1 = 1 ∧ A.2 = a * A.1 ∧ A.2 = (4 - a) / A.1) →
  (∃ A : ℝ × ℝ, A.1 = 1 ∧ A.2 = a * A.1 ∧ A.2 = (4 - a) / A.1 ∧ A.2 = 2) :=
by sorry


end NUMINAMATH_CALUDE_intersection_point_y_coordinate_l238_23832


namespace NUMINAMATH_CALUDE_number_of_pencils_l238_23864

/-- Given that the ratio of pens to pencils is 5 to 6 and there are 8 more pencils than pens,
    prove that the number of pencils is 48. -/
theorem number_of_pencils (pens pencils : ℕ) 
    (h1 : pens * 6 = pencils * 5)  -- ratio of pens to pencils is 5 to 6
    (h2 : pencils = pens + 8)      -- 8 more pencils than pens
    : pencils = 48 := by
  sorry

end NUMINAMATH_CALUDE_number_of_pencils_l238_23864


namespace NUMINAMATH_CALUDE_unique_p_for_three_natural_roots_l238_23842

/-- The cubic equation with parameter p -/
def cubic_equation (p : ℝ) (x : ℝ) : ℝ :=
  5 * x^3 - 5 * (p + 1) * x^2 + (71 * p - 1) * x + 1 - 66 * p

/-- Predicate to check if a number is a natural number -/
def is_natural (x : ℝ) : Prop := ∃ n : ℕ, x = n

/-- The theorem to be proved -/
theorem unique_p_for_three_natural_roots :
  ∃! p : ℝ, ∃ x y z : ℝ,
    x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    is_natural x ∧ is_natural y ∧ is_natural z ∧
    cubic_equation p x = 0 ∧ cubic_equation p y = 0 ∧ cubic_equation p z = 0 ∧
    p = 76 :=
sorry

end NUMINAMATH_CALUDE_unique_p_for_three_natural_roots_l238_23842


namespace NUMINAMATH_CALUDE_smallest_n_for_sqrt_20n_integer_l238_23809

theorem smallest_n_for_sqrt_20n_integer (n : ℕ) : 
  (∃ k : ℕ, k ^ 2 = 20 * n) → (∀ m : ℕ, m > 0 ∧ m < n → ¬∃ k : ℕ, k ^ 2 = 20 * m) → n = 5 := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_for_sqrt_20n_integer_l238_23809


namespace NUMINAMATH_CALUDE_final_class_size_l238_23841

def fourth_grade_class_size (initial_students : ℕ) 
  (first_semester_left : ℕ) (first_semester_joined : ℕ)
  (second_semester_joined : ℕ) (second_semester_transferred : ℕ) (second_semester_switched : ℕ) : ℕ :=
  initial_students - first_semester_left + first_semester_joined + 
  second_semester_joined - second_semester_transferred - second_semester_switched

theorem final_class_size : 
  fourth_grade_class_size 11 6 25 15 3 2 = 40 := by
  sorry

end NUMINAMATH_CALUDE_final_class_size_l238_23841


namespace NUMINAMATH_CALUDE_parabola_through_point_2_4_l238_23875

-- Define a parabola type
structure Parabola where
  equation : ℝ → ℝ → Prop

-- Define a function to check if a point is on the parabola
def on_parabola (p : Parabola) (x y : ℝ) : Prop :=
  p.equation x y

-- Theorem statement
theorem parabola_through_point_2_4 :
  ∃ (p : Parabola), 
    (on_parabola p 2 4) ∧ 
    ((∀ x y : ℝ, p.equation x y ↔ y^2 = 8*x) ∨ 
     (∀ x y : ℝ, p.equation x y ↔ x^2 = y)) :=
sorry

end NUMINAMATH_CALUDE_parabola_through_point_2_4_l238_23875


namespace NUMINAMATH_CALUDE_nonagon_ribbon_theorem_l238_23885

def nonagon_ribbon_length (a b c d e f g h i : ℝ) : Prop :=
  a + b + c + d + e + f + g + h + i = 62 →
  1.5 * (a + b + c + d + e + f + g + h + i) = 93

theorem nonagon_ribbon_theorem :
  ∀ a b c d e f g h i : ℝ, nonagon_ribbon_length a b c d e f g h i :=
by
  sorry

end NUMINAMATH_CALUDE_nonagon_ribbon_theorem_l238_23885


namespace NUMINAMATH_CALUDE_second_gym_signup_fee_covers_four_months_l238_23868

-- Define the given constants
def cheap_monthly_fee : ℤ := 10
def cheap_signup_fee : ℤ := 50
def total_paid_first_year : ℤ := 650
def months_in_year : ℕ := 12

-- Define the relationships
def second_monthly_fee : ℤ := 3 * cheap_monthly_fee

-- State the theorem
theorem second_gym_signup_fee_covers_four_months :
  ∃ (second_signup_fee : ℤ),
    (cheap_monthly_fee * months_in_year + cheap_signup_fee +
     second_monthly_fee * months_in_year + second_signup_fee = total_paid_first_year) ∧
    (second_signup_fee / second_monthly_fee = 4) := by
  sorry

end NUMINAMATH_CALUDE_second_gym_signup_fee_covers_four_months_l238_23868


namespace NUMINAMATH_CALUDE_min_value_in_geometric_sequence_l238_23802

-- Define a positive geometric sequence
def is_positive_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a n > 0 ∧ ∃ r > 0, ∀ k, a (k + 1) = r * a k

-- Define the theorem
theorem min_value_in_geometric_sequence (a : ℕ → ℝ) 
  (h1 : is_positive_geometric_sequence a) 
  (h2 : a 4 * a 14 = 8) : 
  (∀ x y, x > 0 ∧ y > 0 ∧ x * y = 8 → 2*x + y ≥ 8) ∧
  (∃ x y, x > 0 ∧ y > 0 ∧ x * y = 8 ∧ 2*x + y = 8) :=
sorry

end NUMINAMATH_CALUDE_min_value_in_geometric_sequence_l238_23802


namespace NUMINAMATH_CALUDE_larger_integer_proof_l238_23887

theorem larger_integer_proof (A B : ℤ) (h1 : A + B = 2010) (h2 : Nat.lcm A.natAbs B.natAbs = 14807) : 
  max A B = 1139 := by
sorry

end NUMINAMATH_CALUDE_larger_integer_proof_l238_23887


namespace NUMINAMATH_CALUDE_inequality_property_l238_23825

theorem inequality_property (a b : ℝ) (h1 : a < b) (h2 : b < 0) : 1 / a > 1 / b := by
  sorry

end NUMINAMATH_CALUDE_inequality_property_l238_23825


namespace NUMINAMATH_CALUDE_last_digit_of_expression_l238_23862

-- Define a function to get the last digit of a natural number
def lastDigit (n : ℕ) : ℕ := n % 10

-- Define the main theorem
theorem last_digit_of_expression : lastDigit (33 * 3 - (1984^1984 - 1)) = 5 := by
  sorry

end NUMINAMATH_CALUDE_last_digit_of_expression_l238_23862


namespace NUMINAMATH_CALUDE_computation_proof_l238_23840

theorem computation_proof : 24 * ((150 / 3) - (36 / 6) + (7.2 / 0.4) + 2) = 1536 := by
  sorry

end NUMINAMATH_CALUDE_computation_proof_l238_23840


namespace NUMINAMATH_CALUDE_smallest_prime_with_reverse_composite_l238_23823

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(m ∣ n)

def reverse_digits (n : ℕ) : ℕ :=
  let tens := n / 10
  let ones := n % 10
  ones * 10 + tens

def is_composite (n : ℕ) : Prop := n > 1 ∧ ∃ m : ℕ, 1 < m ∧ m < n ∧ m ∣ n

theorem smallest_prime_with_reverse_composite : 
  ∀ n : ℕ, 30 ≤ n ∧ n < 41 →
    ¬(is_prime n ∧ 
      is_composite (reverse_digits n) ∧ 
      ∃ m : ℕ, m ≠ 7 ∧ m > 1 ∧ m ∣ reverse_digits n) →
  is_prime 41 ∧ 
  is_composite (reverse_digits 41) ∧ 
  ∃ m : ℕ, m ≠ 7 ∧ m > 1 ∧ m ∣ reverse_digits 41 :=
sorry

end NUMINAMATH_CALUDE_smallest_prime_with_reverse_composite_l238_23823


namespace NUMINAMATH_CALUDE_line_direction_vector_l238_23866

/-- Given a line passing through two points and a direction vector, prove the value of b -/
theorem line_direction_vector (p1 p2 : ℝ × ℝ) (b : ℝ) : 
  p1 = (-3, 6) → p2 = (2, -1) → 
  (∃ k : ℝ, k ≠ 0 ∧ (p2.1 - p1.1, p2.2 - p1.2) = (k * b, k * (-1))) →
  b = 5/7 := by
sorry

end NUMINAMATH_CALUDE_line_direction_vector_l238_23866


namespace NUMINAMATH_CALUDE_floral_shop_sale_total_l238_23883

/-- Represents the total number of bouquets sold during a three-day sale at a floral shop. -/
def total_bouquets_sold (monday_sales : ℕ) : ℕ :=
  let tuesday_sales := 3 * monday_sales
  let wednesday_sales := tuesday_sales / 3
  monday_sales + tuesday_sales + wednesday_sales

/-- Theorem stating that given the conditions of the sale, the total number of bouquets sold is 60. -/
theorem floral_shop_sale_total (h : total_bouquets_sold 12 = 60) : 
  total_bouquets_sold 12 = 60 := by
  sorry

end NUMINAMATH_CALUDE_floral_shop_sale_total_l238_23883


namespace NUMINAMATH_CALUDE_bobs_walking_rate_l238_23891

/-- Proves that Bob's walking rate is 5 miles per hour given the problem conditions -/
theorem bobs_walking_rate
  (total_distance : ℝ)
  (yolanda_rate : ℝ)
  (bob_start_delay : ℝ)
  (bob_distance : ℝ)
  (h1 : total_distance = 60)
  (h2 : yolanda_rate = 5)
  (h3 : bob_start_delay = 1)
  (h4 : bob_distance = 30) :
  bob_distance / (total_distance / yolanda_rate - bob_start_delay) = 5 :=
by sorry

end NUMINAMATH_CALUDE_bobs_walking_rate_l238_23891


namespace NUMINAMATH_CALUDE_pets_remaining_l238_23848

theorem pets_remaining (initial_puppies initial_kittens puppies_sold kittens_sold : ℕ) :
  initial_puppies = 7 →
  initial_kittens = 6 →
  puppies_sold = 2 →
  kittens_sold = 3 →
  initial_puppies + initial_kittens - (puppies_sold + kittens_sold) = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_pets_remaining_l238_23848


namespace NUMINAMATH_CALUDE_root_in_interval_l238_23859

theorem root_in_interval (a : ℤ) : 
  (∃ x : ℝ, x > a ∧ x < a + 1 ∧ Real.log x + x - 4 = 0) → a = 2 := by
sorry

end NUMINAMATH_CALUDE_root_in_interval_l238_23859


namespace NUMINAMATH_CALUDE_geometric_series_first_term_l238_23881

/-- Given an infinite geometric series with first term a and common ratio r,
    if the sum of the series is 30 and the sum of the squares of its terms is 120,
    then the first term a is equal to 120/17. -/
theorem geometric_series_first_term (a r : ℝ) 
  (h1 : a / (1 - r) = 30)
  (h2 : a^2 / (1 - r^2) = 120) :
  a = 120 / 17 := by
  sorry

end NUMINAMATH_CALUDE_geometric_series_first_term_l238_23881


namespace NUMINAMATH_CALUDE_liquid_x_percentage_in_mixed_solution_l238_23895

/-- Given two solutions P and Q, where liquid X makes up 0.5% of P and 1.5% of Q,
    prove that mixing 200g of P with 800g of Q results in a solution containing 1.3% liquid X. -/
theorem liquid_x_percentage_in_mixed_solution :
  let p_weight : ℝ := 200
  let q_weight : ℝ := 800
  let p_percentage : ℝ := 0.5
  let q_percentage : ℝ := 1.5
  let x_in_p : ℝ := p_weight * (p_percentage / 100)
  let x_in_q : ℝ := q_weight * (q_percentage / 100)
  let total_x : ℝ := x_in_p + x_in_q
  let total_weight : ℝ := p_weight + q_weight
  let result_percentage : ℝ := (total_x / total_weight) * 100
  result_percentage = 1.3 := by sorry

end NUMINAMATH_CALUDE_liquid_x_percentage_in_mixed_solution_l238_23895


namespace NUMINAMATH_CALUDE_mark_bread_making_time_l238_23856

/-- The time it takes Mark to finish making bread -/
def bread_making_time (rise_time : ℕ) (rise_count : ℕ) (knead_time : ℕ) (bake_time : ℕ) : ℕ :=
  rise_time * rise_count + knead_time + bake_time

/-- Theorem stating the total time Mark takes to finish making the bread -/
theorem mark_bread_making_time :
  bread_making_time 120 2 10 30 = 280 := by
  sorry

end NUMINAMATH_CALUDE_mark_bread_making_time_l238_23856


namespace NUMINAMATH_CALUDE_f_leq_g_l238_23813

/-- Given functions f and g, prove that f(x) ≤ g(x) for all x > 0 when a ≥ 1 -/
theorem f_leq_g (x a : ℝ) (hx : x > 0) (ha : a ≥ 1) :
  Real.log x + 2 * x ≤ a * (x^2 + x) := by
  sorry

end NUMINAMATH_CALUDE_f_leq_g_l238_23813


namespace NUMINAMATH_CALUDE_sum_greater_than_three_l238_23893

theorem sum_greater_than_three (a b c : ℝ) 
  (h1 : a * b + b * c + c * a > a + b + c) 
  (h2 : a + b + c > 0) : 
  a + b + c > 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_greater_than_three_l238_23893


namespace NUMINAMATH_CALUDE_cube_volume_from_surface_area_l238_23826

theorem cube_volume_from_surface_area :
  ∀ (s : ℝ), s > 0 → 6 * s^2 = 864 → s^3 = 1728 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_from_surface_area_l238_23826


namespace NUMINAMATH_CALUDE_max_min_on_interval_l238_23852

def f (x : ℝ) : ℝ := 2 * x^3 - 3 * x^2 - 12 * x + 5

theorem max_min_on_interval :
  ∃ (a b : ℝ), a ∈ Set.Icc 0 3 ∧ b ∈ Set.Icc 0 3 ∧
  (∀ x, x ∈ Set.Icc 0 3 → f x ≤ f a) ∧
  (∀ x, x ∈ Set.Icc 0 3 → f x ≥ f b) ∧
  f a = 5 ∧ f b = -15 :=
sorry

end NUMINAMATH_CALUDE_max_min_on_interval_l238_23852


namespace NUMINAMATH_CALUDE_swimmer_distance_l238_23805

/-- Calculates the distance swum against a current given the swimmer's speed in still water,
    the speed of the current, and the time taken. -/
def distance_against_current (swimmer_speed : ℝ) (current_speed : ℝ) (time : ℝ) : ℝ :=
  (swimmer_speed - current_speed) * time

/-- Proves that given the specified conditions, the swimmer travels 6 km against the current. -/
theorem swimmer_distance :
  let swimmer_speed := 4
  let current_speed := 2
  let time := 3
  distance_against_current swimmer_speed current_speed time = 6 := by
sorry

end NUMINAMATH_CALUDE_swimmer_distance_l238_23805


namespace NUMINAMATH_CALUDE_proposition_count_l238_23806

theorem proposition_count : 
  let prop1 := ∀ x : ℝ, x^2 + x + 1 ≥ 0
  let prop2 := ∀ x y : ℝ, (x + y ≠ 3) → (x ≠ 2 ∨ y ≠ 1)
  let prop3 := 
    let slope : ℝ := 1.23
    let center : ℝ × ℝ := (4, 5)
    (5 : ℝ) = slope * (4 : ℝ) + 0.08
  let prop4 := 
    ∀ m : ℝ, (m = 3 ↔ 
      ∀ x y : ℝ, ((m + 3) * x + m * y - 2 = 0 → m * x - 6 * y + 5 = 0) ∧
                 (m * x - 6 * y + 5 = 0 → (m + 3) * x + m * y - 2 = 0))
  (prop1 ∧ ¬prop2 ∧ ¬prop3 ∧ ¬prop4) ∨
  (¬prop1 ∧ prop2 ∧ ¬prop3 ∧ ¬prop4) ∨
  (¬prop1 ∧ ¬prop2 ∧ prop3 ∧ ¬prop4) ∨
  (¬prop1 ∧ ¬prop2 ∧ ¬prop3 ∧ prop4) ∨
  (prop1 ∧ prop2 ∧ ¬prop3 ∧ ¬prop4) ∨
  (prop1 ∧ ¬prop2 ∧ prop3 ∧ ¬prop4) ∨
  (prop1 ∧ ¬prop2 ∧ ¬prop3 ∧ prop4) ∨
  (¬prop1 ∧ prop2 ∧ prop3 ∧ ¬prop4) ∨
  (¬prop1 ∧ prop2 ∧ ¬prop3 ∧ prop4) ∨
  (¬prop1 ∧ ¬prop2 ∧ prop3 ∧ prop4) := by
  sorry

end NUMINAMATH_CALUDE_proposition_count_l238_23806


namespace NUMINAMATH_CALUDE_stratified_sampling_female_athletes_l238_23870

theorem stratified_sampling_female_athletes 
  (total_population : ℕ) 
  (sample_size : ℕ) 
  (female_athletes : ℕ) 
  (h1 : total_population = 224) 
  (h2 : sample_size = 32) 
  (h3 : female_athletes = 84) : 
  ↑sample_size * female_athletes / total_population = 12 :=
by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_female_athletes_l238_23870


namespace NUMINAMATH_CALUDE_sum_of_positive_numbers_l238_23892

theorem sum_of_positive_numbers (a b : ℝ) : 
  a > 0 → b > 0 → (a + b) / (a^2 + a*b + b^2) = 4/49 → a + b = 16 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_positive_numbers_l238_23892


namespace NUMINAMATH_CALUDE_lauryn_earnings_l238_23844

theorem lauryn_earnings (x : ℝ) : 
  x + 0.7 * x = 3400 → x = 2000 := by
  sorry

end NUMINAMATH_CALUDE_lauryn_earnings_l238_23844


namespace NUMINAMATH_CALUDE_max_value_of_sum_products_l238_23877

theorem max_value_of_sum_products (a b c : ℝ) (h : a + b + 3 * c = 6) :
  ∃ (max : ℝ), max = 516 / 49 ∧ ∀ (x y z : ℝ), x + y + 3 * z = 6 → x * y + x * z + y * z ≤ max :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_sum_products_l238_23877


namespace NUMINAMATH_CALUDE_ratio_problem_l238_23871

theorem ratio_problem (antecedent consequent : ℚ) : 
  antecedent / consequent = 4 / 6 → antecedent = 20 → consequent = 30 := by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l238_23871


namespace NUMINAMATH_CALUDE_triangle_properties_l238_23880

/-- Given a triangle ABC with dot product conditions, prove the length of AB and a trigonometric ratio -/
theorem triangle_properties (A B C : ℝ × ℝ) 
  (h1 : (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 9)
  (h2 : (B.1 - A.1) * (C.1 - B.1) + (B.2 - A.2) * (C.2 - B.2) = -16) :
  let AB := Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)
  let BC := Real.sqrt ((C.1 - B.1)^2 + (C.2 - B.2)^2)
  let CA := Real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2)
  let cosA := ((B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2)) / (AB * CA)
  let cosB := ((A.1 - B.1) * (C.1 - B.1) + (A.2 - B.2) * (C.2 - B.2)) / (AB * BC)
  let cosC := ((A.1 - C.1) * (B.1 - C.1) + (A.2 - C.2) * (B.2 - C.2)) / (BC * CA)
  let sinA := Real.sqrt (1 - cosA^2)
  let sinB := Real.sqrt (1 - cosB^2)
  let sinC := Real.sqrt (1 - cosC^2)
  AB = 5 ∧ (sinA * cosB - cosA * sinB) / sinC = 7/25 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l238_23880


namespace NUMINAMATH_CALUDE_pet_store_combinations_l238_23818

def num_puppies : ℕ := 20
def num_kittens : ℕ := 4
def num_hamsters : ℕ := 6
def num_rabbits : ℕ := 10
def num_people : ℕ := 4

theorem pet_store_combinations : 
  num_puppies * num_kittens * num_hamsters * num_rabbits * (Nat.factorial num_people) = 115200 := by
  sorry

end NUMINAMATH_CALUDE_pet_store_combinations_l238_23818


namespace NUMINAMATH_CALUDE_students_in_diligence_l238_23873

/-- Represents the number of students in a section before transfers -/
structure SectionCount where
  diligence : ℕ
  industry : ℕ
  progress : ℕ

/-- Represents the transfers between sections -/
structure Transfers where
  industry_to_diligence : ℕ
  progress_to_industry : ℕ

/-- The problem statement -/
theorem students_in_diligence 
  (initial : SectionCount) 
  (transfers : Transfers) 
  (total_students : ℕ) :
  (initial.diligence + initial.industry + initial.progress = total_students) →
  (initial.diligence + transfers.industry_to_diligence = 
   initial.industry - transfers.industry_to_diligence + transfers.progress_to_industry) →
  (initial.diligence + transfers.industry_to_diligence = 
   initial.progress - transfers.progress_to_industry) →
  (transfers.industry_to_diligence = 2) →
  (transfers.progress_to_industry = 3) →
  (total_students = 75) →
  initial.diligence = 23 := by
  sorry

end NUMINAMATH_CALUDE_students_in_diligence_l238_23873


namespace NUMINAMATH_CALUDE_inequality_solution_l238_23874

theorem inequality_solution (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (∀ x : ℝ, a^(x + 5) < a^(4*x - 1) ↔ (0 < a ∧ a < 1 ∧ x < 2) ∨ (a > 1 ∧ x > 2)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l238_23874


namespace NUMINAMATH_CALUDE_finance_charge_rate_example_l238_23849

/-- Given an original balance and a total payment, calculate the finance charge rate. -/
def finance_charge_rate (original_balance total_payment : ℚ) : ℚ :=
  (total_payment - original_balance) / original_balance * 100

/-- Theorem: The finance charge rate is 2% when the original balance is $150 and the total payment is $153. -/
theorem finance_charge_rate_example :
  finance_charge_rate 150 153 = 2 := by
  sorry

end NUMINAMATH_CALUDE_finance_charge_rate_example_l238_23849


namespace NUMINAMATH_CALUDE_sheets_colored_l238_23889

/-- Given 2450 sheets of paper evenly split into 5 binders,
    prove that coloring one-half of the sheets in one binder uses 245 sheets. -/
theorem sheets_colored (total_sheets : ℕ) (num_binders : ℕ) (sheets_per_binder : ℕ) :
  total_sheets = 2450 →
  num_binders = 5 →
  total_sheets = num_binders * sheets_per_binder →
  sheets_per_binder / 2 = 245 := by
  sorry

#check sheets_colored

end NUMINAMATH_CALUDE_sheets_colored_l238_23889


namespace NUMINAMATH_CALUDE_problem_solution_l238_23898

theorem problem_solution (x y : ℝ) : 
  (2*x - 3*y + 5)^2 + |x + y - 2| = 0 → 3*x - 2*y = -3 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l238_23898


namespace NUMINAMATH_CALUDE_divisible_by_nine_l238_23828

theorem divisible_by_nine : ∃ (n : ℕ), 5742 = 9 * n := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_nine_l238_23828


namespace NUMINAMATH_CALUDE_tank_insulation_cost_l238_23821

/-- Calculates the surface area of a rectangular prism -/
def surface_area (length width height : ℝ) : ℝ :=
  2 * (length * width + length * height + width * height)

/-- Calculates the total cost of insulation for a rectangular tank -/
def insulation_cost (length width height cost_per_sqft : ℝ) : ℝ :=
  surface_area length width height * cost_per_sqft

/-- Theorem: The cost of insulating a 4x5x2 feet tank at $20 per square foot is $1520 -/
theorem tank_insulation_cost :
  insulation_cost 4 5 2 20 = 1520 := by
  sorry

#eval insulation_cost 4 5 2 20

end NUMINAMATH_CALUDE_tank_insulation_cost_l238_23821


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l238_23854

/-- An arithmetic sequence with given properties -/
structure ArithmeticSequence where
  a : ℕ → ℤ  -- The sequence
  s : ℕ → ℤ  -- The sum of the first n terms
  first_term : a 1 = 31
  sum_equality : s 10 = s 22

/-- The sum formula for the arithmetic sequence -/
def sum_formula (n : ℕ) : ℤ := 32 * n - n^2

/-- Theorem stating the properties of the arithmetic sequence -/
theorem arithmetic_sequence_properties (seq : ArithmeticSequence) :
  (∀ n, seq.s n = sum_formula n) ∧
  (∃ n, ∀ m, seq.s m ≤ seq.s n) ∧
  (seq.s 16 = 256) := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l238_23854


namespace NUMINAMATH_CALUDE_inequality_solution_l238_23817

theorem inequality_solution (x : ℝ) : 
  (x^3 - 3*x^2 + 2*x) / (x^2 - 2*x + 1) ≥ 0 ↔ x ∈ Set.Ici 0 ∩ Set.Iio 1 ∪ Set.Ici 2 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l238_23817


namespace NUMINAMATH_CALUDE_quadratic_points_relationship_l238_23820

theorem quadratic_points_relationship (m : ℝ) (y₁ y₂ y₃ : ℝ) : 
  (2^2 - 4*2 - m = y₁) →
  (3^2 - 4*3 - m = y₂) →
  ((-1)^2 - 4*(-1) - m = y₃) →
  (y₃ > y₂ ∧ y₂ > y₁) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_points_relationship_l238_23820


namespace NUMINAMATH_CALUDE_min_k_value_l238_23824

theorem min_k_value (x y k : ℝ) : 
  (x - y + 5 ≥ 0) → 
  (x ≤ 3) → 
  (x + y + k ≥ 0) → 
  (∃ z : ℝ, z = 2*x + 4*y ∧ z ≥ -6 ∧ ∀ w : ℝ, w = 2*x + 4*y → w ≥ z) →
  k ≥ 0 :=
by sorry

end NUMINAMATH_CALUDE_min_k_value_l238_23824


namespace NUMINAMATH_CALUDE_probability_theorem_l238_23801

def total_shoes : ℕ := 28
def black_pairs : ℕ := 7
def brown_pairs : ℕ := 4
def gray_pairs : ℕ := 2
def white_pairs : ℕ := 1

def probability_same_color_left_right : ℚ :=
  (black_pairs * 2 / total_shoes) * (black_pairs / (total_shoes - 1)) +
  (brown_pairs * 2 / total_shoes) * (brown_pairs / (total_shoes - 1)) +
  (gray_pairs * 2 / total_shoes) * (gray_pairs / (total_shoes - 1)) +
  (white_pairs * 2 / total_shoes) * (white_pairs / (total_shoes - 1))

theorem probability_theorem :
  probability_same_color_left_right = 35 / 189 := by
  sorry

end NUMINAMATH_CALUDE_probability_theorem_l238_23801


namespace NUMINAMATH_CALUDE_fermat_number_prime_count_l238_23831

/-- Fermat number defined as F_n = 2^(2^n) + 1 -/
def fermat_number (n : ℕ) : ℕ := 2^(2^n) + 1

/-- There are at least n+1 distinct prime numbers less than or equal to F_n -/
theorem fermat_number_prime_count (n : ℕ) :
  ∃ (S : Finset ℕ), S.card = n + 1 ∧ (∀ p ∈ S, Nat.Prime p ∧ p ≤ fermat_number n) :=
sorry

end NUMINAMATH_CALUDE_fermat_number_prime_count_l238_23831


namespace NUMINAMATH_CALUDE_lemon_pie_degrees_l238_23829

/-- The number of degrees in a circle -/
def circle_degrees : ℕ := 360

/-- The total number of students -/
def total_students : ℕ := 45

/-- The number of students preferring chocolate pie -/
def chocolate_preference : ℕ := 15

/-- The number of students preferring apple pie -/
def apple_preference : ℕ := 10

/-- The number of students preferring blueberry pie -/
def blueberry_preference : ℕ := 7

/-- The number of students preferring lemon pie -/
def lemon_preference : ℕ := (total_students - (chocolate_preference + apple_preference + blueberry_preference)) / 2

theorem lemon_pie_degrees :
  (lemon_preference : ℚ) / total_students * circle_degrees = 56 := by
  sorry

end NUMINAMATH_CALUDE_lemon_pie_degrees_l238_23829


namespace NUMINAMATH_CALUDE_wall_thickness_is_5cm_l238_23863

/-- Represents the dimensions of a brick in centimeters -/
structure BrickDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Represents the dimensions of a wall in meters -/
structure WallDimensions where
  length : ℝ
  height : ℝ
  thickness : ℝ

/-- Calculates the volume of a brick in cubic centimeters -/
def brickVolume (b : BrickDimensions) : ℝ :=
  b.length * b.width * b.height

/-- Calculates the total volume of bricks in cubic centimeters -/
def totalBrickVolume (b : BrickDimensions) (n : ℝ) : ℝ :=
  brickVolume b * n

/-- Calculates the area of the wall's face in square centimeters -/
def wallFaceArea (w : WallDimensions) : ℝ :=
  w.length * w.height * 10000 -- Convert m² to cm²

/-- The main theorem stating the wall thickness -/
theorem wall_thickness_is_5cm 
  (brick : BrickDimensions)
  (wall : WallDimensions)
  (num_bricks : ℝ) :
  brick.length = 25 ∧ 
  brick.width = 11 ∧ 
  brick.height = 6 ∧
  wall.length = 8 ∧
  wall.height = 1 ∧
  num_bricks = 242.42424242424244 →
  wall.thickness = 5 := by
sorry

end NUMINAMATH_CALUDE_wall_thickness_is_5cm_l238_23863


namespace NUMINAMATH_CALUDE_inequality_proof_l238_23847

theorem inequality_proof (a b c : ℝ) : a^4 + b^4 + c^4 ≥ a*b*c^2 + b*c*a^2 + c*a*b^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l238_23847


namespace NUMINAMATH_CALUDE_tetrahedron_PQRS_volume_l238_23833

def tetrahedron_volume (PQ PR PS QR QS RS : ℝ) : ℝ := sorry

theorem tetrahedron_PQRS_volume :
  let PQ : ℝ := 3
  let PR : ℝ := Real.sqrt 10
  let PS : ℝ := Real.sqrt 17
  let QR : ℝ := 5
  let QS : ℝ := 3 * Real.sqrt 2
  let RS : ℝ := 6
  let z : ℝ := Real.sqrt (17 - (4/3)^2 - (1/(2*Real.sqrt 10))^2)
  tetrahedron_volume PQ PR PS QR QS RS = (Real.sqrt 10 / 2) * z := by sorry

end NUMINAMATH_CALUDE_tetrahedron_PQRS_volume_l238_23833


namespace NUMINAMATH_CALUDE_tom_investment_is_3000_l238_23878

/-- Represents the initial investment problem with Tom and Jose --/
structure InvestmentProblem where
  jose_investment : ℕ
  jose_months : ℕ
  total_profit : ℕ
  jose_profit : ℕ

/-- Calculates Tom's initial investment given the problem parameters --/
def calculate_tom_investment (p : InvestmentProblem) : ℕ :=
  sorry

/-- Theorem stating that Tom's initial investment is 3000 --/
theorem tom_investment_is_3000 (p : InvestmentProblem)
  (h1 : p.jose_investment = 45000)
  (h2 : p.jose_months = 10)
  (h3 : p.total_profit = 27000)
  (h4 : p.jose_profit = 15000) :
  calculate_tom_investment p = 3000 :=
sorry

end NUMINAMATH_CALUDE_tom_investment_is_3000_l238_23878


namespace NUMINAMATH_CALUDE_supplement_of_complement_of_sixty_degrees_l238_23816

/-- The degree measure of the supplement of the complement of a 60-degree angle is 150°. -/
theorem supplement_of_complement_of_sixty_degrees : 
  let original_angle : ℝ := 60
  let complement := 90 - original_angle
  let supplement := 180 - complement
  supplement = 150 := by sorry

end NUMINAMATH_CALUDE_supplement_of_complement_of_sixty_degrees_l238_23816


namespace NUMINAMATH_CALUDE_angle_A_is_pi_over_four_angle_C_is_5pi_over_12_l238_23855

-- Define a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  S : ℝ
  -- Ensure all sides and angles are positive
  ha : 0 < a
  hb : 0 < b
  hc : 0 < c
  hA : 0 < A ∧ A < π
  hB : 0 < B ∧ B < π
  hC : 0 < C ∧ C < π
  -- Ensure the sum of angles is π
  angle_sum : A + B + C = π
  -- Area formula
  area_formula : S = (1/2) * b * c * Real.sin A

-- Theorem 1
theorem angle_A_is_pi_over_four (t : Triangle) (h : t.a^2 + 4*t.S = t.b^2 + t.c^2) :
  t.A = π/4 := by sorry

-- Theorem 2
theorem angle_C_is_5pi_over_12 (t : Triangle) 
  (h1 : t.a^2 + 4*t.S = t.b^2 + t.c^2) (h2 : t.a = Real.sqrt 2) (h3 : t.b = Real.sqrt 3) :
  t.C = 5*π/12 := by sorry

end NUMINAMATH_CALUDE_angle_A_is_pi_over_four_angle_C_is_5pi_over_12_l238_23855


namespace NUMINAMATH_CALUDE_club_enrollment_l238_23872

theorem club_enrollment (total : ℕ) (math : ℕ) (chem : ℕ) (both : ℕ) :
  total = 150 →
  math = 90 →
  chem = 70 →
  both = 20 →
  total - (math + chem - both) = 10 :=
by sorry

end NUMINAMATH_CALUDE_club_enrollment_l238_23872


namespace NUMINAMATH_CALUDE_count_odd_numbers_between_215_and_500_l238_23899

theorem count_odd_numbers_between_215_and_500 : 
  (Finset.filter (fun n => n % 2 = 1 ∧ n > 215 ∧ n < 500) (Finset.range 500)).card = 142 :=
by sorry

end NUMINAMATH_CALUDE_count_odd_numbers_between_215_and_500_l238_23899


namespace NUMINAMATH_CALUDE_max_value_inequality_max_value_achieved_l238_23819

theorem max_value_inequality (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) 
  (h4 : a^2 + b^2 + 2*c^2 = 1) : 
  a*b*Real.sqrt 3 + 3*b*c ≤ Real.sqrt 7 :=
sorry

theorem max_value_achieved (ε : ℝ) (hε : ε > 0) : 
  ∃ a b c : ℝ, 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ 
  a^2 + b^2 + 2*c^2 = 1 ∧ 
  Real.sqrt 7 - ε < a*b*Real.sqrt 3 + 3*b*c :=
sorry

end NUMINAMATH_CALUDE_max_value_inequality_max_value_achieved_l238_23819


namespace NUMINAMATH_CALUDE_quadratic_abs_equivalence_l238_23838

theorem quadratic_abs_equivalence (a : ℝ) : a^2 + 4*a - 5 > 0 ↔ |a + 2| > 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_abs_equivalence_l238_23838


namespace NUMINAMATH_CALUDE_amicable_iff_ge_seven_l238_23834

/-- An integer n ≥ 2 is amicable if there exist subsets A₁, A₂, ..., Aₙ of {1, 2, ..., n} satisfying:
    (i) i ∉ Aᵢ for any i = 1, 2, ..., n
    (ii) i ∈ Aⱼ for any j ∉ Aᵢ, for any i ≠ j
    (iii) Aᵢ ∩ Aⱼ ≠ ∅ for any i, j ∈ {1, 2, ..., n} -/
def IsAmicable (n : ℕ) : Prop :=
  n ≥ 2 ∧
  ∃ A : Fin n → Set (Fin n),
    (∀ i, i ∉ A i) ∧
    (∀ i j, i ≠ j → (j ∉ A i ↔ i ∈ A j)) ∧
    (∀ i j, (A i ∩ A j).Nonempty)

/-- For any integer n ≥ 2, n is amicable if and only if n ≥ 7 -/
theorem amicable_iff_ge_seven (n : ℕ) : IsAmicable n ↔ n ≥ 7 := by
  sorry

end NUMINAMATH_CALUDE_amicable_iff_ge_seven_l238_23834


namespace NUMINAMATH_CALUDE_arctangent_sum_equals_pi_over_four_l238_23896

theorem arctangent_sum_equals_pi_over_four :
  ∃ (n : ℕ+), (Real.arctan (1/3) + Real.arctan (1/4) + Real.arctan (1/6) + Real.arctan (1/n) = π/4) ∧ n = 113 := by
  sorry

end NUMINAMATH_CALUDE_arctangent_sum_equals_pi_over_four_l238_23896


namespace NUMINAMATH_CALUDE_fathers_sons_age_sum_l238_23886

theorem fathers_sons_age_sum (father_age son_age : ℕ) : 
  father_age = 40 → 
  son_age = 15 → 
  2 * son_age + father_age = 70 → 
  2 * father_age + son_age = 95 :=
by sorry

end NUMINAMATH_CALUDE_fathers_sons_age_sum_l238_23886


namespace NUMINAMATH_CALUDE_sum_even_positive_lt_100_eq_2450_l238_23894

/-- The sum of all even, positive integers less than 100 -/
def sum_even_positive_lt_100 : ℕ :=
  (Finset.range 50).sum (fun i => 2 * i)

/-- Theorem stating that the sum of all even, positive integers less than 100 is 2450 -/
theorem sum_even_positive_lt_100_eq_2450 : sum_even_positive_lt_100 = 2450 := by
  sorry

end NUMINAMATH_CALUDE_sum_even_positive_lt_100_eq_2450_l238_23894


namespace NUMINAMATH_CALUDE_circle_quadratic_intersection_l238_23800

/-- Given a circle and a quadratic equation, prove the center coordinates and condition --/
theorem circle_quadratic_intersection (p q b c : ℝ) : 
  (∀ x y : ℝ, x^2 + y^2 - 2*p*x - 2*q*y + 2*q - 1 = 0 ↔ 
   (y = 0 → x^2 + b*x + c = 0)) →
  (p = -b/2 ∧ q = (1+c)/2 ∧ b^2 - 4*c ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_circle_quadratic_intersection_l238_23800


namespace NUMINAMATH_CALUDE_sum_of_multiples_of_6_and_9_is_multiple_of_3_l238_23803

theorem sum_of_multiples_of_6_and_9_is_multiple_of_3 (a b : ℤ) 
  (ha : ∃ k : ℤ, a = 6 * k) (hb : ∃ m : ℤ, b = 9 * m) : 
  ∃ n : ℤ, a + b = 3 * n := by
sorry

end NUMINAMATH_CALUDE_sum_of_multiples_of_6_and_9_is_multiple_of_3_l238_23803


namespace NUMINAMATH_CALUDE_wheel_of_fortune_probability_l238_23815

theorem wheel_of_fortune_probability (p_D p_E p_F p_G : ℚ) : 
  p_D = 3/8 → p_E = 1/4 → p_G = 1/8 → p_D + p_E + p_F + p_G = 1 → p_F = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_wheel_of_fortune_probability_l238_23815


namespace NUMINAMATH_CALUDE_product_scaling_l238_23822

theorem product_scaling (a b c : ℝ) (h : (a * 100) * (b * 100) = c) : 
  a * b = c / 10000 := by
  sorry

end NUMINAMATH_CALUDE_product_scaling_l238_23822
