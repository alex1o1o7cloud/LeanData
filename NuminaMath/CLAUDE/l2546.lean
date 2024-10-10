import Mathlib

namespace projectile_height_l2546_254630

theorem projectile_height (t : ℝ) : 
  t > 0 ∧ -16 * t^2 + 60 * t = 56 ∧ 
  ∀ s, s > 0 ∧ -16 * s^2 + 60 * s = 56 → t ≤ s → 
  t = 1.75 := by
sorry

end projectile_height_l2546_254630


namespace triangle_properties_l2546_254657

theorem triangle_properties (a b c : ℝ) (A B C : ℝ) :
  (4 * a = Real.sqrt 5 * c) →
  (Real.cos C = 3 / 5) →
  (b = 11) →
  (Real.sin A = Real.sqrt 5 / 5) ∧
  (1 / 2 * a * b * Real.sin C = 22) :=
by sorry

end triangle_properties_l2546_254657


namespace bacteria_growth_time_l2546_254613

theorem bacteria_growth_time (fill_time : ℕ) (initial_count : ℕ) : 
  (fill_time = 64 ∧ initial_count = 1) → 
  (∃ (new_fill_time : ℕ), new_fill_time = 62 ∧ 2^new_fill_time * initial_count * 4 = 2^fill_time) :=
by sorry

end bacteria_growth_time_l2546_254613


namespace trigonometric_expression_evaluation_l2546_254643

theorem trigonometric_expression_evaluation : 
  (2 * Real.sin (100 * π / 180) - Real.cos (70 * π / 180)) / Real.cos (20 * π / 180) = 2 * Real.sqrt 3 - 1 := by
  sorry

end trigonometric_expression_evaluation_l2546_254643


namespace xiao_jun_age_problem_l2546_254649

/-- Represents the current age of Xiao Jun -/
def xiao_jun_age : ℕ := 6

/-- Represents the current age ratio between Xiao Jun's mother and Xiao Jun -/
def current_age_ratio : ℕ := 5

/-- Represents the future age ratio between Xiao Jun's mother and Xiao Jun -/
def future_age_ratio : ℕ := 3

/-- Calculates the number of years that need to pass for Xiao Jun's mother's age 
    to be 3 times Xiao Jun's age -/
def years_passed : ℕ := 6

theorem xiao_jun_age_problem : 
  xiao_jun_age * current_age_ratio + years_passed = 
  (xiao_jun_age + years_passed) * future_age_ratio :=
sorry

end xiao_jun_age_problem_l2546_254649


namespace power_function_through_point_l2546_254627

-- Define a power function
def isPowerFunction (f : ℝ → ℝ) : Prop := ∃ α : ℝ, ∀ x : ℝ, f x = x ^ α

-- Define the theorem
theorem power_function_through_point (f : ℝ → ℝ) :
  isPowerFunction f → f 3 = Real.sqrt 3 → f 4 = 2 := by
  sorry

end power_function_through_point_l2546_254627


namespace rectangle_diagonal_l2546_254636

def Rectangle (O A B : ℝ × ℝ) : Prop :=
  ∃ C : ℝ × ℝ, (O.1 - A.1) * (A.1 - C.1) + (O.2 - A.2) * (A.2 - C.2) = 0 ∧
              (O.1 - B.1) * (B.1 - C.1) + (O.2 - B.2) * (B.2 - C.2) = 0

theorem rectangle_diagonal (O A B : ℝ × ℝ) (h : Rectangle O A B) :
  let OA : ℝ × ℝ := (-3, 1)
  let OB : ℝ × ℝ := (-2, k)
  k = 4 :=
sorry

end rectangle_diagonal_l2546_254636


namespace polynomial_equality_l2546_254647

theorem polynomial_equality (x t s : ℝ) : 
  (3 * x^2 - 4 * x + 9) * (5 * x^2 + t * x + s) = 
  15 * x^4 - 22 * x^3 + (41 + s) * x^2 - 34 * x + 9 * s ↔ 
  t = -2 ∧ s = s := by
sorry

end polynomial_equality_l2546_254647


namespace polygon_150_diagonals_l2546_254653

/-- Number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

theorem polygon_150_diagonals :
  (num_diagonals 150 = 11025) ∧
  (9900 ≠ num_diagonals 150 / 2) :=
by sorry

end polygon_150_diagonals_l2546_254653


namespace complex_roots_unity_l2546_254631

theorem complex_roots_unity (z₁ z₂ z₃ : ℂ) 
  (h1 : Complex.abs z₁ = 1) 
  (h2 : Complex.abs z₂ = 1) 
  (h3 : Complex.abs z₃ = 1)
  (h4 : z₁ + z₂ + z₃ = 1) 
  (h5 : z₁ * z₂ * z₃ = 1) :
  ({z₁, z₂, z₃} : Finset ℂ) = {1, Complex.I, -Complex.I} := by
  sorry

end complex_roots_unity_l2546_254631


namespace remainder_of_binary_div_4_l2546_254641

def binary_number : List Bool := [true, true, false, true, false, true, false, false, true, false, true, true]

def last_two_digits (n : List Bool) : (Bool × Bool) :=
  match n.reverse with
  | b0 :: b1 :: _ => (b1, b0)
  | _ => (false, false)  -- Default case, should not occur for valid input

def remainder_mod_4 (digits : Bool × Bool) : Nat :=
  let (b1, b0) := digits
  2 * (if b1 then 1 else 0) + (if b0 then 1 else 0)

theorem remainder_of_binary_div_4 :
  remainder_mod_4 (last_two_digits binary_number) = 3 := by
  sorry

end remainder_of_binary_div_4_l2546_254641


namespace number_satisfies_equation_l2546_254695

theorem number_satisfies_equation : ∃ x : ℝ, (0.8 * 90 : ℝ) = 0.7 * x + 30 := by
  sorry

end number_satisfies_equation_l2546_254695


namespace race_track_outer_radius_l2546_254688

/-- Given a circular race track with an inner circumference of 880 m and a width of 25 m,
    the radius of the outer circle is 165 m. -/
theorem race_track_outer_radius :
  ∀ (inner_radius outer_radius : ℝ),
    inner_radius * 2 * Real.pi = 880 →
    outer_radius = inner_radius + 25 →
    outer_radius = 165 := by
  sorry

end race_track_outer_radius_l2546_254688


namespace not_perfect_square_product_l2546_254606

/-- A function that checks if a number is a perfect square -/
def is_perfect_square (x : ℕ) : Prop :=
  ∃ m : ℕ, x = m * m

/-- The main theorem stating that 1, 2, and 4 are the only positive integers
    for which n(n+a) is not a perfect square for all positive integers n -/
theorem not_perfect_square_product (a : ℕ) : a > 0 →
  (∀ n : ℕ, n > 0 → ¬is_perfect_square (n * (n + a))) ↔ a = 1 ∨ a = 2 ∨ a = 4 :=
sorry

end not_perfect_square_product_l2546_254606


namespace houses_per_block_l2546_254693

theorem houses_per_block (mail_per_block : ℕ) (mail_per_house : ℕ) 
  (h1 : mail_per_block = 32) (h2 : mail_per_house = 8) :
  mail_per_block / mail_per_house = 4 := by
  sorry

end houses_per_block_l2546_254693


namespace abs_neg_three_eq_three_l2546_254617

theorem abs_neg_three_eq_three : |(-3 : ℤ)| = 3 := by sorry

end abs_neg_three_eq_three_l2546_254617


namespace largest_arithmetic_mean_of_special_pairs_l2546_254665

theorem largest_arithmetic_mean_of_special_pairs : ∃ (a b : ℕ),
  10 ≤ a ∧ a < 100 ∧
  10 ≤ b ∧ b < 100 ∧
  (a + b) / 2 = (25 / 24) * Real.sqrt (a * b) ∧
  ∀ (c d : ℕ),
    10 ≤ c ∧ c < 100 ∧
    10 ≤ d ∧ d < 100 ∧
    (c + d) / 2 = (25 / 24) * Real.sqrt (c * d) →
    (a + b) / 2 ≥ (c + d) / 2 ∧
  (a + b) / 2 = 75 :=
sorry

end largest_arithmetic_mean_of_special_pairs_l2546_254665


namespace parallel_segments_length_l2546_254604

/-- Given three parallel line segments XY, UV, and PQ, where UV = 90 cm and XY = 120 cm,
    prove that the length of PQ is 360/7 cm. -/
theorem parallel_segments_length (XY UV PQ : ℝ) (h1 : XY = 120) (h2 : UV = 90)
    (h3 : ∃ (k : ℝ), XY = k * UV ∧ PQ = k * UV) : PQ = 360 / 7 := by
  sorry

end parallel_segments_length_l2546_254604


namespace cos_225_degrees_l2546_254678

theorem cos_225_degrees : Real.cos (225 * π / 180) = -Real.sqrt 2 / 2 := by
  sorry

end cos_225_degrees_l2546_254678


namespace tiles_for_taylors_room_l2546_254624

/-- Calculates the total number of tiles needed for a rectangular room with a border of smaller tiles --/
def total_tiles (room_length room_width border_tile_size interior_tile_size : ℕ) : ℕ :=
  let border_tiles := 2 * (room_length + room_width) - 4
  let interior_length := room_length - 2 * border_tile_size
  let interior_width := room_width - 2 * border_tile_size
  let interior_area := interior_length * interior_width
  let interior_tiles := interior_area / (interior_tile_size * interior_tile_size)
  border_tiles + interior_tiles

/-- Theorem stating that for a 12x16 room with 1x1 border tiles and 2x2 interior tiles, 87 tiles are needed --/
theorem tiles_for_taylors_room : total_tiles 12 16 1 2 = 87 := by
  sorry

end tiles_for_taylors_room_l2546_254624


namespace fraction_sum_equals_62_l2546_254667

theorem fraction_sum_equals_62 (a b : ℝ) : 
  a = (Real.sqrt 5 + Real.sqrt 3) / (Real.sqrt 5 - Real.sqrt 3) →
  b = (Real.sqrt 5 - Real.sqrt 3) / (Real.sqrt 5 + Real.sqrt 3) →
  b / a + a / b = 62 := by
sorry

end fraction_sum_equals_62_l2546_254667


namespace summer_sun_salutations_l2546_254661

/-- The number of sun salutations Summer performs in a year -/
def sun_salutations_per_year (poses_per_day : ℕ) (weekdays_per_week : ℕ) (weeks_per_year : ℕ) : ℕ :=
  poses_per_day * weekdays_per_week * weeks_per_year

/-- Theorem: Summer performs 1300 sun salutations in a year -/
theorem summer_sun_salutations :
  sun_salutations_per_year 5 5 52 = 1300 := by
  sorry

end summer_sun_salutations_l2546_254661


namespace parabola_triangle_area_l2546_254614

/-- Given a parabola y² = 4x with focus F(1, 0) and directrix x = -1,
    and a line through F with slope √3 intersecting the parabola above
    the x-axis at point A, prove that the area of triangle AFK is 4√3,
    where K is the foot of the perpendicular from A to the directrix. -/
theorem parabola_triangle_area :
  let parabola : ℝ × ℝ → Prop := λ p => p.2^2 = 4 * p.1
  let F : ℝ × ℝ := (1, 0)
  let directrix : ℝ → Prop := λ x => x = -1
  let line : ℝ → ℝ := λ x => Real.sqrt 3 * (x - 1)
  let A : ℝ × ℝ := (3, 2 * Real.sqrt 3)
  let K : ℝ × ℝ := (-1, 2 * Real.sqrt 3)
  parabola A ∧
  (∀ x, line x = A.2 ↔ x = A.1) ∧
  directrix K.1 ∧
  (A.2 - K.2) / (A.1 - K.1) * (F.2 - A.2) / (F.1 - A.1) = -1 →
  (1/2) * abs (A.1 - F.1) * abs (A.2 - K.2) = 4 * Real.sqrt 3 := by
sorry

end parabola_triangle_area_l2546_254614


namespace max_abs_sum_under_condition_l2546_254632

theorem max_abs_sum_under_condition (x y : ℝ) (h : x^2 + y^2 + x*y = 1) :
  |x| + |y| ≤ Real.sqrt (4/3) := by
  sorry

end max_abs_sum_under_condition_l2546_254632


namespace toy_cost_l2546_254644

theorem toy_cost (saved : ℕ) (allowance : ℕ) (num_toys : ℕ) :
  saved = 21 →
  allowance = 15 →
  num_toys = 6 →
  (saved + allowance) / num_toys = 6 :=
by
  sorry

end toy_cost_l2546_254644


namespace sqrt_equality_implies_t_value_l2546_254687

theorem sqrt_equality_implies_t_value :
  ∀ t : ℝ, (Real.sqrt (3 * Real.sqrt (t - 3)) = (9 - t) ^ (1/4)) → t = 3.6 :=
by
  sorry

end sqrt_equality_implies_t_value_l2546_254687


namespace tablets_consumed_l2546_254648

/-- Proves that given a person who takes one tablet every 15 minutes and consumes all tablets in 60 minutes, the total number of tablets taken is 4. -/
theorem tablets_consumed (interval : ℕ) (total_time : ℕ) (h1 : interval = 15) (h2 : total_time = 60) :
  total_time / interval = 4 := by
  sorry

end tablets_consumed_l2546_254648


namespace eldorado_license_plates_l2546_254621

/-- The number of letters in the alphabet -/
def num_letters : ℕ := 26

/-- The number of digits (0-9) -/
def num_digits : ℕ := 10

/-- The number of letter positions in a license plate -/
def letter_positions : ℕ := 3

/-- The number of digit positions in a license plate -/
def digit_positions : ℕ := 4

/-- The total number of possible valid license plates in Eldorado -/
def total_license_plates : ℕ := num_letters ^ letter_positions * num_digits ^ digit_positions

theorem eldorado_license_plates :
  total_license_plates = 175760000 := by
  sorry

end eldorado_license_plates_l2546_254621


namespace largest_n_for_factorization_l2546_254689

theorem largest_n_for_factorization : ∃ (n : ℤ),
  (∀ m : ℤ, (∃ (a b : ℤ), 3 * X^2 + m * X + 54 = (3 * X + a) * (X + b)) → m ≤ n) ∧
  (∃ (a b : ℤ), 3 * X^2 + n * X + 54 = (3 * X + a) * (X + b)) ∧
  n = 163 :=
by sorry


end largest_n_for_factorization_l2546_254689


namespace unique_triplet_sum_l2546_254681

theorem unique_triplet_sum (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (hab : a ≤ b) (hbc : b ≤ c)
  (heq : (25 : ℚ) / 84 = 1 / a + 1 / (a * b) + 1 / (a * b * c)) :
  a + b + c = 17 := by
  sorry

end unique_triplet_sum_l2546_254681


namespace simple_interest_problem_l2546_254659

theorem simple_interest_problem (P R : ℝ) (h1 : P > 0) (h2 : R > 0) :
  (P * (R + 1) * 3 / 100 = P * R * 3 / 100 + 75) → P = 2500 := by
  sorry

end simple_interest_problem_l2546_254659


namespace root_reciprocal_relation_l2546_254600

theorem root_reciprocal_relation (p m q n : ℝ) : 
  (∃ x : ℝ, x^2 + p*x + q = 0 ∧ (1/x)^2 + m*(1/x) + n = 0) → 
  (p*n - m)*(q*m - p) = (q*n - 1)^2 := by
  sorry

end root_reciprocal_relation_l2546_254600


namespace cat_collar_nylon_l2546_254668

/-- The number of inches of nylon needed for one dog collar -/
def dog_collar_nylon : ℝ := 18

/-- The total number of inches of nylon needed for all collars -/
def total_nylon : ℝ := 192

/-- The number of dog collars -/
def num_dog_collars : ℕ := 9

/-- The number of cat collars -/
def num_cat_collars : ℕ := 3

/-- Theorem stating that the number of inches of nylon needed for one cat collar is 10 -/
theorem cat_collar_nylon : 
  (total_nylon - dog_collar_nylon * num_dog_collars) / num_cat_collars = 10 := by
sorry

end cat_collar_nylon_l2546_254668


namespace tank_volume_ratio_l2546_254650

theorem tank_volume_ratio :
  ∀ (tank1_volume tank2_volume : ℚ),
  tank1_volume > 0 →
  tank2_volume > 0 →
  (3 / 4 : ℚ) * tank1_volume = (5 / 8 : ℚ) * tank2_volume →
  tank1_volume / tank2_volume = (5 / 6 : ℚ) :=
by
  sorry

end tank_volume_ratio_l2546_254650


namespace fraction_value_unchanged_keep_fraction_unchanged_l2546_254646

theorem fraction_value_unchanged (a b c : ℤ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) :
  (a : ℚ) / b = (a + c) / (b + ((a + c) * b / a - b)) :=
by sorry

theorem keep_fraction_unchanged :
  let original_numerator := 3
  let original_denominator := 4
  let numerator_increase := 9
  let new_numerator := original_numerator + numerator_increase
  let denominator_increase := new_numerator * original_denominator / original_numerator - original_denominator
  denominator_increase = 12 :=
by sorry

end fraction_value_unchanged_keep_fraction_unchanged_l2546_254646


namespace business_hours_per_week_l2546_254686

-- Define the operating hours for weekdays and weekends
def weekdayHours : ℕ := 6
def weekendHours : ℕ := 4

-- Define the number of weekdays and weekend days in a week
def weekdays : ℕ := 5
def weekendDays : ℕ := 2

-- Define the total hours open in a week
def totalHoursOpen : ℕ := weekdayHours * weekdays + weekendHours * weekendDays

-- Theorem statement
theorem business_hours_per_week :
  totalHoursOpen = 38 := by
sorry

end business_hours_per_week_l2546_254686


namespace unique_root_monotonic_continuous_l2546_254645

theorem unique_root_monotonic_continuous {f : ℝ → ℝ} {a b : ℝ} (h_mono : Monotone f) (h_cont : Continuous f) (h_sign : f a * f b < 0) (h_le : a ≤ b) :
  ∃! x, a ≤ x ∧ x ≤ b ∧ f x = 0 :=
sorry

end unique_root_monotonic_continuous_l2546_254645


namespace simplify_fraction_l2546_254639

theorem simplify_fraction : (2^6 + 2^4) / (2^5 - 2^2) = 20 / 7 := by
  sorry

end simplify_fraction_l2546_254639


namespace circle_properties_l2546_254660

theorem circle_properties :
  let center : ℝ × ℝ := (1, -1)
  let radius : ℝ := Real.sqrt 2
  let origin : ℝ × ℝ := (0, 0)
  let tangent_point : ℝ × ℝ := (2, 0)
  let distance (p q : ℝ × ℝ) := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  let on_circle (p : ℝ × ℝ) := distance p center = radius
  let tangent_line (x y : ℝ) := x + y - 2 = 0
  
  (on_circle origin) ∧ 
  (on_circle tangent_point) ∧
  (tangent_line tangent_point.1 tangent_point.2) ∧
  (∀ (p : ℝ × ℝ), tangent_line p.1 p.2 → distance p center ≥ radius) :=
by sorry

end circle_properties_l2546_254660


namespace min_product_sum_l2546_254677

theorem min_product_sum (a₁ a₂ a₃ b₁ b₂ b₃ : ℕ) : 
  ({a₁, a₂, a₃, b₁, b₂, b₃} : Finset ℕ) = {1, 2, 3, 4, 5, 6} →
  a₁ * a₂ * a₃ + b₁ * b₂ * b₃ ≥ 56 ∧ 
  ∃ (x₁ x₂ x₃ y₁ y₂ y₃ : ℕ), 
    ({x₁, x₂, x₃, y₁, y₂, y₃} : Finset ℕ) = {1, 2, 3, 4, 5, 6} ∧
    x₁ * x₂ * x₃ + y₁ * y₂ * y₃ = 56 :=
by sorry

end min_product_sum_l2546_254677


namespace condition_equivalence_l2546_254683

theorem condition_equivalence (a : ℝ) (h : a > 0) : (a > 1) ↔ (a > Real.sqrt a) := by
  sorry

end condition_equivalence_l2546_254683


namespace man_speed_against_current_l2546_254691

/-- Given a man's speed with the current and the speed of the current,
    calculate the man's speed against the current. -/
def speed_against_current (speed_with_current speed_of_current : ℝ) : ℝ :=
  speed_with_current - 2 * speed_of_current

/-- Theorem stating that given the specific speeds in the problem,
    the man's speed against the current is 10 km/hr. -/
theorem man_speed_against_current :
  speed_against_current 15 2.5 = 10 := by
  sorry

end man_speed_against_current_l2546_254691


namespace evaluate_f_l2546_254629

/-- The function f(x) = x^3 + 3∛x -/
def f (x : ℝ) : ℝ := x^3 + 3 * (x^(1/3))

/-- Theorem stating that 3f(3) + f(27) = 19818 -/
theorem evaluate_f : 3 * f 3 + f 27 = 19818 := by
  sorry

end evaluate_f_l2546_254629


namespace equation_solution_in_interval_l2546_254637

theorem equation_solution_in_interval :
  ∃ x₀ ∈ Set.Ioo 2 3, Real.log x₀ + x₀ - 4 = 0 := by
  sorry

end equation_solution_in_interval_l2546_254637


namespace power_two_greater_than_square_plus_one_l2546_254633

theorem power_two_greater_than_square_plus_one (n : ℕ) (h : n ≥ 5) :
  2^n > n^2 + 1 := by
  sorry

end power_two_greater_than_square_plus_one_l2546_254633


namespace estimate_products_and_quotients_l2546_254674

theorem estimate_products_and_quotients 
  (ε₁ ε₂ ε₃ ε₄ : ℝ) 
  (h₁ : ε₁ > 0) 
  (h₂ : ε₂ > 0) 
  (h₃ : ε₃ > 0) 
  (h₄ : ε₄ > 0) : 
  (|99 * 71 - 7000| ≤ ε₁) ∧ 
  (|25 * 39 - 1000| ≤ ε₂) ∧ 
  (|124 / 3 - 40| ≤ ε₃) ∧ 
  (|398 / 5 - 80| ≤ ε₄) := by
  sorry

end estimate_products_and_quotients_l2546_254674


namespace min_balls_needed_l2546_254619

/-- Represents the number of balls of each color -/
structure BallCounts where
  red : ℕ
  white : ℕ
  blue : ℕ
  green : ℕ

/-- Conditions for drawing balls -/
def satisfiesConditions (counts : BallCounts) : Prop :=
  counts.red ≥ 4 ∧
  counts.white ≥ 1 ∧
  counts.blue ≥ 1 ∧
  counts.green ≥ 1 ∧
  (counts.red.choose 4 : ℚ) = 
    (counts.red.choose 3 * counts.white : ℚ) ∧
  (counts.red.choose 3 * counts.white : ℚ) = 
    (counts.red.choose 2 * counts.white * counts.blue : ℚ) ∧
  (counts.red.choose 2 * counts.white * counts.blue : ℚ) = 
    (counts.red * counts.white * counts.blue * counts.green : ℚ)

/-- The theorem to be proved -/
theorem min_balls_needed : 
  ∃ (counts : BallCounts), 
    satisfiesConditions counts ∧ 
    (∀ (other : BallCounts), satisfiesConditions other → 
      counts.red + counts.white + counts.blue + counts.green ≤ 
      other.red + other.white + other.blue + other.green) ∧
    counts.red + counts.white + counts.blue + counts.green = 21 :=
sorry

end min_balls_needed_l2546_254619


namespace roll_12_with_8_dice_l2546_254675

/-- The number of ways to roll a sum of 12 with 8 fair 6-sided dice -/
def waysToRoll12With8Dice : ℕ := sorry

/-- The number of faces on each die -/
def numFaces : ℕ := 6

/-- The number of dice -/
def numDice : ℕ := 8

/-- The target sum -/
def targetSum : ℕ := 12

theorem roll_12_with_8_dice :
  waysToRoll12With8Dice = 330 := by sorry

end roll_12_with_8_dice_l2546_254675


namespace logarithm_difference_l2546_254672

theorem logarithm_difference (a b c d : ℕ+) 
  (h1 : (Real.log b) / (Real.log a) = 3/2)
  (h2 : (Real.log d) / (Real.log c) = 5/4)
  (h3 : a - c = 9) : 
  b - d = 93 := by
  sorry

end logarithm_difference_l2546_254672


namespace trajectory_is_ellipse_l2546_254625

/-- The trajectory of point P given the conditions in the problem -/
def trajectory (x y : ℝ) : Prop :=
  x^2 / 2 + y^2 = 1

theorem trajectory_is_ellipse (x y : ℝ) :
  let P : ℝ × ℝ := (x, y)
  let M : ℝ × ℝ := (1, 0)
  let d : ℝ := |x - 2|
  (‖P - M‖ : ℝ) / d = Real.sqrt 2 / 2 →
  trajectory x y :=
by sorry

end trajectory_is_ellipse_l2546_254625


namespace complex_multiplication_result_l2546_254605

theorem complex_multiplication_result : (1 + Complex.I) * (-Complex.I) = 1 - Complex.I := by
  sorry

end complex_multiplication_result_l2546_254605


namespace janet_action_figures_l2546_254654

/-- Calculates the final number of action figures Janet has -/
def final_action_figure_count (initial_count : ℕ) (sold_count : ℕ) (bought_count : ℕ) : ℕ :=
  let remaining_count := initial_count - sold_count
  let after_purchase_count := remaining_count + bought_count
  after_purchase_count + 2 * after_purchase_count

theorem janet_action_figures :
  final_action_figure_count 10 6 4 = 24 := by
  sorry

#eval final_action_figure_count 10 6 4

end janet_action_figures_l2546_254654


namespace division_into_proportional_parts_l2546_254664

theorem division_into_proportional_parts (total : ℚ) (a b c : ℚ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  total = 104 →
  a = 1 →
  b = 1/2 →
  c = 1/3 →
  let x := total * b / (a + b + c)
  x = 28 + 4/11 := by
  sorry

end division_into_proportional_parts_l2546_254664


namespace robie_cards_count_l2546_254611

/-- The number of cards in each box -/
def cards_per_box : ℕ := 10

/-- The number of cards not placed in a box -/
def cards_outside_box : ℕ := 5

/-- The number of boxes Robie gave away -/
def boxes_given_away : ℕ := 2

/-- The number of boxes Robie has with him -/
def boxes_remaining : ℕ := 5

/-- The total number of cards Robie had in the beginning -/
def total_cards : ℕ := (boxes_given_away + boxes_remaining) * cards_per_box + cards_outside_box

theorem robie_cards_count : total_cards = 75 := by
  sorry

end robie_cards_count_l2546_254611


namespace furniture_cost_price_l2546_254628

theorem furniture_cost_price (final_price : ℝ) : 
  final_price = 9522.84 →
  ∃ (cost_price : ℝ),
    cost_price = 7695 ∧
    final_price = (1.12 * (0.85 * (1.3 * cost_price))) :=
by sorry

end furniture_cost_price_l2546_254628


namespace f_inequality_l2546_254697

-- Define a function f on the real numbers
variable (f : ℝ → ℝ)

-- Define the properties of f
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)
def is_increasing_on_nonneg (f : ℝ → ℝ) : Prop := ∀ x y, 0 ≤ x → x < y → f x < f y

-- State the theorem
theorem f_inequality (h1 : is_even f) (h2 : is_increasing_on_nonneg f) : f (-2) > f 1 ∧ f 1 > f 0 := by
  sorry

end f_inequality_l2546_254697


namespace exam_students_count_l2546_254656

theorem exam_students_count :
  ∀ (N : ℕ) (T : ℝ),
    N > 5 →
    T = N * 80 →
    (T - 250) / (N - 5 : ℝ) = 90 →
    N = 20 := by
  sorry

end exam_students_count_l2546_254656


namespace block_placement_probability_l2546_254670

/-- Represents a person in the block placement problem -/
inductive Person
  | Louis
  | Maria
  | Neil

/-- Represents the colors of the blocks -/
inductive Color
  | Red
  | Blue
  | Yellow
  | White
  | Green
  | Purple

/-- The number of boxes -/
def num_boxes : ℕ := 6

/-- The number of blocks each person has -/
def num_blocks_per_person : ℕ := 6

/-- A function representing a random block placement for a person -/
def block_placement := Person → Fin num_boxes → Color

/-- The probability of a specific color being chosen for a specific box by all three people -/
def prob_color_match : ℚ := 1 / 216

/-- The probability that at least one box receives exactly 3 blocks of the same color,
    placed in alphabetical order by the people's names -/
def prob_at_least_one_box_match : ℚ := 235 / 1296

theorem block_placement_probability :
  prob_at_least_one_box_match = 1 - (1 - prob_color_match) ^ num_boxes :=
sorry

end block_placement_probability_l2546_254670


namespace shortest_distance_point_to_parabola_l2546_254602

/-- The shortest distance between a point and a parabola -/
theorem shortest_distance_point_to_parabola :
  let point := (6, 12)
  let parabola := {(x, y) : ℝ × ℝ | x = y^2 / 2}
  (shortest_distance : ℝ) →
  shortest_distance = 2 * Real.sqrt 17 ∧
  ∀ (p : ℝ × ℝ), p ∈ parabola → 
    Real.sqrt ((point.1 - p.1)^2 + (point.2 - p.2)^2) ≥ shortest_distance :=
by sorry

end shortest_distance_point_to_parabola_l2546_254602


namespace tom_age_l2546_254658

theorem tom_age (adam_age : ℕ) (future_years : ℕ) (future_combined_age : ℕ) :
  adam_age = 8 →
  future_years = 12 →
  future_combined_age = 44 →
  ∃ tom_age : ℕ, tom_age + adam_age + 2 * future_years = future_combined_age ∧ tom_age = 12 :=
by sorry

end tom_age_l2546_254658


namespace expression_evaluation_l2546_254635

theorem expression_evaluation (a b c d e : ℚ) 
  (h1 : a * b = 1) 
  (h2 : c + d = 0) 
  (h3 : |e| = 2) : 
  (c + d) / 5 - (1 / 2) * a * b + e = 3 / 2 ∨ 
  (c + d) / 5 - (1 / 2) * a * b + e = -(5 / 2) :=
sorry

end expression_evaluation_l2546_254635


namespace difference_largest_third_smallest_l2546_254652

def digits : List Nat := [1, 6, 8]

def largest_number : Nat := 861

def third_smallest_number : Nat := 618

theorem difference_largest_third_smallest :
  largest_number - third_smallest_number = 243 := by
  sorry

end difference_largest_third_smallest_l2546_254652


namespace square_plus_25_divisible_by_2_and_5_l2546_254616

/-- A positive integer with only prime divisors 2 and 5 -/
def HasOnly2And5AsDivisors (n : ℕ) : Prop :=
  ∀ p : ℕ, Nat.Prime p → p ∣ n → p = 2 ∨ p = 5

theorem square_plus_25_divisible_by_2_and_5 :
  ∀ N : ℕ, N > 0 →
  HasOnly2And5AsDivisors N →
  (∃ M : ℕ, N + 25 = M^2) →
  N = 200 ∨ N = 2000 := by
sorry

end square_plus_25_divisible_by_2_and_5_l2546_254616


namespace solve_equations_l2546_254676

theorem solve_equations (t u s : ℝ) : 
  t = 15 * s^2 → 
  u = 5 * s + 3 → 
  t = 3.75 → 
  s = 0.5 ∧ u = 5.5 := by
  sorry

end solve_equations_l2546_254676


namespace digit_sum_theorem_l2546_254682

-- Define the conditions
def is_valid_digits (A B C : ℕ) : Prop :=
  A ≠ 0 ∧ B ≠ 0 ∧ C ≠ 0 ∧ A < 10 ∧ B < 10 ∧ C < 10

def BC (B C : ℕ) : ℕ := 10 * B + C

def ABC (A B C : ℕ) : ℕ := 100 * A + 10 * B + C

-- State the theorem
theorem digit_sum_theorem (A B C : ℕ) :
  is_valid_digits A B C →
  BC B C + ABC A B C + ABC A B C = 876 →
  A + B + C = 14 := by
sorry

end digit_sum_theorem_l2546_254682


namespace trapezoid_shaded_fraction_l2546_254634

/-- Represents a trapezoid divided into strips -/
structure StripedTrapezoid where
  num_strips : ℕ
  shaded_strips : ℕ

/-- The fraction of the trapezoid's area that is shaded -/
def shaded_fraction (t : StripedTrapezoid) : ℚ :=
  t.shaded_strips / t.num_strips

theorem trapezoid_shaded_fraction :
  ∀ t : StripedTrapezoid,
    t.num_strips = 7 →
    shaded_fraction t = 4/7 := by
  sorry

end trapezoid_shaded_fraction_l2546_254634


namespace georges_initial_socks_l2546_254662

theorem georges_initial_socks (bought new_from_dad total_now : ℕ) 
  (h1 : bought = 36)
  (h2 : new_from_dad = 4)
  (h3 : total_now = 68)
  : total_now - bought - new_from_dad = 28 := by
  sorry

end georges_initial_socks_l2546_254662


namespace smallest_n_congruence_l2546_254666

theorem smallest_n_congruence : 
  ∃ (n : ℕ), n > 0 ∧ (5 * n) % 26 = 789 % 26 ∧ 
  ∀ (m : ℕ), m > 0 ∧ (5 * m) % 26 = 789 % 26 → n ≤ m :=
by sorry

end smallest_n_congruence_l2546_254666


namespace scott_running_distance_l2546_254690

/-- Scott's running schedule and total distance for a month --/
theorem scott_running_distance :
  let miles_mon_to_wed : ℕ := 3 * 3
  let miles_thu_fri : ℕ := 2 * (2 * 3)
  let miles_per_week : ℕ := miles_mon_to_wed + miles_thu_fri
  let weeks_in_month : ℕ := 4
  miles_per_week * weeks_in_month = 84 := by
  sorry

end scott_running_distance_l2546_254690


namespace sqrt_equation_solution_l2546_254607

theorem sqrt_equation_solution :
  ∃! x : ℝ, Real.sqrt (x + 15) - 4 / Real.sqrt (x + 15) = 3 :=
by
  -- Proof goes here
  sorry

end sqrt_equation_solution_l2546_254607


namespace quadrilateral_angle_measure_l2546_254615

theorem quadrilateral_angle_measure (A B C D : ℝ) : 
  A + B = 180 →  -- ∠A + ∠B = 180°
  C = D →        -- ∠C = ∠D
  A = 40 →       -- ∠A = 40°
  B + C = 160 → -- ∠B + ∠C = 160°
  D = 20 :=      -- Prove that ∠D = 20°
by sorry

end quadrilateral_angle_measure_l2546_254615


namespace keith_pears_count_l2546_254638

def total_pears : Nat := 5
def jason_pears : Nat := 2

theorem keith_pears_count : total_pears - jason_pears = 3 := by
  sorry

end keith_pears_count_l2546_254638


namespace area_triangle_BFE_l2546_254626

/-- Given a rectangle ABCD with area 48 square units and points E and F dividing sides AD and BC
    in a 2:1 ratio, the area of triangle BFE is 24 square units. -/
theorem area_triangle_BFE (A B C D E F : ℝ × ℝ) : 
  let rectangle_area := 48
  let ratio := (2 : ℝ) / 3
  (∃ u v : ℝ, 
    A = (0, 0) ∧ 
    B = (3*u, 0) ∧ 
    C = (3*u, 3*v) ∧ 
    D = (0, 3*v) ∧
    E = (0, 2*v) ∧ 
    F = (2*u, 0) ∧
    3*u*3*v = rectangle_area ∧
    (D.2 - E.2) / D.2 = ratio ∧
    (C.1 - F.1) / C.1 = ratio) →
  (1/2 * |B.1*(E.2 - F.2) + E.1*(F.2 - B.2) + F.1*(B.2 - E.2)| = 24) :=
by sorry

end area_triangle_BFE_l2546_254626


namespace specific_hexagon_area_l2546_254642

/-- Hexagon formed by two overlapping equilateral triangles -/
structure Hexagon where
  /-- Side length of the equilateral triangles -/
  side_length : ℝ
  /-- Rotation angle in radians -/
  rotation_angle : ℝ
  /-- The hexagon is symmetric about a central point -/
  symmetric : Bool
  /-- Points A and A' coincide -/
  coincident_points : Bool

/-- Calculate the area of the hexagon -/
def hexagon_area (h : Hexagon) : ℝ :=
  sorry

/-- Theorem stating the area of the specific hexagon -/
theorem specific_hexagon_area :
  let h : Hexagon := {
    side_length := 2,
    rotation_angle := Real.pi / 6,  -- 30 degrees in radians
    symmetric := true,
    coincident_points := true
  }
  hexagon_area h = Real.sqrt 3 :=
sorry

end specific_hexagon_area_l2546_254642


namespace diameter_endpoint_l2546_254680

/-- A circle in a 2D plane --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents a diameter of a circle --/
structure Diameter where
  circle : Circle
  endpoint1 : ℝ × ℝ
  endpoint2 : ℝ × ℝ

/-- Given a circle with center (3, 4) and one endpoint of a diameter at (1, -2),
    the other endpoint of the diameter is at (5, 10) --/
theorem diameter_endpoint (P : Circle) (d : Diameter) :
  P.center = (3, 4) →
  d.circle = P →
  d.endpoint1 = (1, -2) →
  d.endpoint2 = (5, 10) := by
sorry

end diameter_endpoint_l2546_254680


namespace circles_tangent_implies_a_value_l2546_254608

/-- Circle C1 with center (a, 0) and radius 2 -/
def C1 (a : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - a)^2 + p.2^2 = 4}

/-- Circle C2 with center (0, √5) and radius |a| -/
def C2 (a : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + (p.2 - Real.sqrt 5)^2 = a^2}

/-- Two circles are externally tangent if the distance between their centers
    equals the sum of their radii -/
def externally_tangent (C1 C2 : Set (ℝ × ℝ)) : Prop :=
  ∃ (c1 c2 : ℝ × ℝ) (r1 r2 : ℝ),
    C1 = {p : ℝ × ℝ | (p.1 - c1.1)^2 + (p.2 - c1.2)^2 = r1^2} ∧
    C2 = {p : ℝ × ℝ | (p.1 - c2.1)^2 + (p.2 - c2.2)^2 = r2^2} ∧
    (c1.1 - c2.1)^2 + (c1.2 - c2.2)^2 = (r1 + r2)^2

theorem circles_tangent_implies_a_value :
  ∀ a : ℝ, externally_tangent (C1 a) (C2 a) → a = 1/4 ∨ a = -1/4 := by
  sorry

end circles_tangent_implies_a_value_l2546_254608


namespace translation_theorem_l2546_254610

/-- The original function -/
def f (x : ℝ) : ℝ := -(x - 1)^2 + 3

/-- The target function -/
def g (x : ℝ) : ℝ := -x^2

/-- The translation function -/
def translate (x : ℝ) : ℝ := x + 1

theorem translation_theorem :
  ∀ x : ℝ, f (translate x) - 3 = g x := by sorry

end translation_theorem_l2546_254610


namespace committee_size_is_24_l2546_254698

/-- The number of sandwiches per person -/
def sandwiches_per_person : ℕ := 2

/-- The number of croissants per pack -/
def croissants_per_pack : ℕ := 12

/-- The cost of one pack of croissants in cents -/
def cost_per_pack : ℕ := 800

/-- The total amount spent on croissants in cents -/
def total_spent : ℕ := 3200

/-- The number of people on the committee -/
def committee_size : ℕ := total_spent / cost_per_pack * croissants_per_pack / sandwiches_per_person

theorem committee_size_is_24 : committee_size = 24 := by
  sorry

end committee_size_is_24_l2546_254698


namespace stating_same_white_wins_exist_l2546_254669

/-- Represents a chess tournament with participants and their scores. -/
structure ChessTournament where
  /-- The number of participants in the tournament. -/
  participants : Nat
  /-- The number of games won with white pieces by each participant. -/
  white_wins : Fin participants → Nat
  /-- Assumption that all participants have the same total score. -/
  same_total_score : ∀ i j : Fin participants, 
    white_wins i + (participants - 1 - white_wins j) = participants - 1

/-- 
Theorem stating that in a chess tournament where all participants have the same total score,
there must be at least two participants who won the same number of games with white pieces.
-/
theorem same_white_wins_exist (t : ChessTournament) : 
  ∃ i j : Fin t.participants, i ≠ j ∧ t.white_wins i = t.white_wins j := by
  sorry


end stating_same_white_wins_exist_l2546_254669


namespace sacks_filled_l2546_254620

theorem sacks_filled (wood_per_sack : ℕ) (total_wood : ℕ) (h1 : wood_per_sack = 20) (h2 : total_wood = 80) :
  total_wood / wood_per_sack = 4 := by
  sorry

end sacks_filled_l2546_254620


namespace f_properties_l2546_254699

noncomputable def f (x : ℝ) : ℝ := (2 - x) * Real.exp x

theorem f_properties :
  (∃ (f' : ℝ → ℝ), DifferentiableAt ℝ f 2 ∧ deriv f 2 = f' 2 ∧ f' 2 < 0) ∧
  (∃ (x_max : ℝ), x_max = 1 ∧ ∀ x, f x ≤ f x_max ∧ f x_max = Real.exp 1) ∧
  (∀ x > 1, ∃ (f' : ℝ → ℝ), DifferentiableAt ℝ f x ∧ deriv f x = f' x ∧ f' x < 0) ∧
  (∀ a : ℝ, (∃! (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ f x₁ = a ∧ f x₂ = a) → 0 < a ∧ a < Real.exp 1) :=
sorry

end f_properties_l2546_254699


namespace derivative_neg_cos_l2546_254692

theorem derivative_neg_cos (x : ℝ) : deriv (fun x => -Real.cos x) x = Real.sin x := by
  sorry

end derivative_neg_cos_l2546_254692


namespace product_over_sum_equals_6608_l2546_254685

theorem product_over_sum_equals_6608 : 
  (1 * 2 * 3 * 4 * 5 * 6 * 7 * 8 * 9) / (1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 + 10) = 6608 := by
  sorry

end product_over_sum_equals_6608_l2546_254685


namespace savings_equality_l2546_254622

def total_salary : ℝ := 4000
def a_salary : ℝ := 3000
def a_spend_rate : ℝ := 0.95
def b_spend_rate : ℝ := 0.85

def b_salary : ℝ := total_salary - a_salary

def a_savings : ℝ := a_salary * (1 - a_spend_rate)
def b_savings : ℝ := b_salary * (1 - b_spend_rate)

theorem savings_equality : a_savings = b_savings := by
  sorry

end savings_equality_l2546_254622


namespace swimming_lane_length_l2546_254671

/-- Represents the length of a swimming lane in meters -/
def lane_length : ℝ := 100

/-- Represents the number of round trips swam -/
def round_trips : ℕ := 4

/-- Represents the total distance swam in meters -/
def total_distance : ℝ := 800

/-- Represents the number of lane lengths in a round trip -/
def lengths_per_round_trip : ℕ := 2

theorem swimming_lane_length :
  lane_length * (round_trips * lengths_per_round_trip) = total_distance :=
sorry

end swimming_lane_length_l2546_254671


namespace function_solution_l2546_254679

-- Define the function type
def FunctionType := ℝ → ℝ

-- Define the functional equation
def SatisfiesEquation (g : FunctionType) : Prop :=
  ∀ x y : ℝ, (g x * g y - g (x * y)) / 4 = x + y + 1

-- State the theorem
theorem function_solution (g : FunctionType) (h : SatisfiesEquation g) :
  (∀ x : ℝ, g x = 2 * x + 3) ∨ (∀ x : ℝ, g x = -2 * x - 1) :=
sorry

end function_solution_l2546_254679


namespace equation_solution_l2546_254603

theorem equation_solution (x : ℝ) (hx : x ≠ 0) :
  4 - 9 / x + 4 / (x^2) = 0 → (3 / x = 12 ∨ 3 / x = 3 / 4) := by
  sorry

end equation_solution_l2546_254603


namespace nancy_jade_amount_l2546_254640

/-- The amount of jade (in grams) needed for a giraffe statue -/
def giraffe_jade : ℝ := 120

/-- The price (in dollars) of a giraffe statue -/
def giraffe_price : ℝ := 150

/-- The amount of jade (in grams) needed for an elephant statue -/
def elephant_jade : ℝ := 2 * giraffe_jade

/-- The price (in dollars) of an elephant statue -/
def elephant_price : ℝ := 350

/-- The additional revenue (in dollars) from making elephant statues instead of giraffe statues -/
def additional_revenue : ℝ := 400

/-- The theorem stating the amount of jade Nancy has -/
theorem nancy_jade_amount :
  ∃ (J : ℝ), J > 0 ∧
    (J / elephant_jade) * elephant_price - (J / giraffe_jade) * giraffe_price = additional_revenue ∧
    J = 1920 := by
  sorry

end nancy_jade_amount_l2546_254640


namespace max_three_roots_l2546_254684

-- Define the function f(x)
def f (a b c : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + b*x + c

-- State the theorem
theorem max_three_roots 
  (a b c : ℝ) 
  (x₁ x₂ : ℝ) 
  (h1 : ∃ x₁' x₂', x₁' ≠ x₂' ∧ (∀ x, x ≠ x₁' ∧ x ≠ x₂' → (deriv (f a b c)) x ≠ 0)) 
  (h2 : f a b c x₁ = x₁) :
  ∃ S : Finset ℝ, (∀ x, 3*(f a b c x)^2 + 2*a*(f a b c x) + b = 0 ↔ x ∈ S) ∧ Finset.card S ≤ 3 :=
sorry

end max_three_roots_l2546_254684


namespace unique_distribution_l2546_254651

/-- Represents the number of ways to distribute n identical balls into boxes with given capacities -/
def distribution_count (n : ℕ) (capacities : List ℕ) : ℕ :=
  sorry

/-- The capacities of the four boxes -/
def box_capacities : List ℕ := [3, 5, 7, 8]

/-- The total number of balls to distribute -/
def total_balls : ℕ := 19

/-- Theorem stating that there's only one way to distribute the balls -/
theorem unique_distribution : distribution_count total_balls box_capacities = 1 := by
  sorry

end unique_distribution_l2546_254651


namespace power_sum_fifth_l2546_254673

/-- Given real numbers a, b, x, y satisfying certain conditions, 
    prove that ax^5 + by^5 = 180.36 -/
theorem power_sum_fifth (a b x y : ℝ) 
  (h1 : a * x + b * y = 5)
  (h2 : a * x^2 + b * y^2 = 11)
  (h3 : a * x^3 + b * y^3 = 24)
  (h4 : a * x^4 + b * y^4 = 56) :
  a * x^5 + b * y^5 = 180.36 := by
  sorry

end power_sum_fifth_l2546_254673


namespace music_tool_cost_proof_l2546_254601

/-- The cost of Joan's purchases at the music store -/
def total_spent : ℚ := 163.28

/-- The cost of the trumpet Joan bought -/
def trumpet_cost : ℚ := 149.16

/-- The cost of the song book Joan bought -/
def song_book_cost : ℚ := 4.14

/-- The cost of the music tool -/
def music_tool_cost : ℚ := total_spent - trumpet_cost - song_book_cost

theorem music_tool_cost_proof : music_tool_cost = 9.98 := by
  sorry

end music_tool_cost_proof_l2546_254601


namespace bisecting_line_sum_l2546_254696

/-- Triangle PQR with vertices P(0, 10), Q(3, 0), and R(10, 0) -/
structure Triangle :=
  (P : ℝ × ℝ)
  (Q : ℝ × ℝ)
  (R : ℝ × ℝ)

/-- A line represented by its slope and y-intercept -/
structure Line :=
  (slope : ℝ)
  (intercept : ℝ)

/-- The line that bisects the area of the triangle -/
def bisecting_line (t : Triangle) : Line :=
  sorry

/-- The theorem to be proved -/
theorem bisecting_line_sum (t : Triangle) :
  let pqr := Triangle.mk (0, 10) (3, 0) (10, 0)
  let l := bisecting_line pqr
  l.slope + l.intercept = -5 :=
sorry

end bisecting_line_sum_l2546_254696


namespace mo_drinks_26_cups_l2546_254612

/-- Represents Mo's drinking habits and the weather conditions for a week -/
structure WeeklyDrinks where
  n : ℕ  -- Number of hot chocolate cups on a rainy day
  rainyDays : ℕ  -- Number of rainy days in the week
  teaPerNonRainyDay : ℕ  -- Number of tea cups on a non-rainy day
  teaExcess : ℕ  -- Excess of tea cups over hot chocolate cups

/-- Calculates the total number of cups (tea and hot chocolate) Mo drinks in a week -/
def totalCups (w : WeeklyDrinks) : ℕ :=
  w.n * w.rainyDays + w.teaPerNonRainyDay * (7 - w.rainyDays)

/-- Theorem stating that under the given conditions, Mo drinks 26 cups in total -/
theorem mo_drinks_26_cups (w : WeeklyDrinks)
  (h1 : w.rainyDays = 1)
  (h2 : w.teaPerNonRainyDay = 3)
  (h3 : w.teaPerNonRainyDay * (7 - w.rainyDays) = w.n * w.rainyDays + w.teaExcess)
  (h4 : w.teaExcess = 10) :
  totalCups w = 26 := by
  sorry

#check mo_drinks_26_cups

end mo_drinks_26_cups_l2546_254612


namespace dans_helmet_craters_l2546_254663

theorem dans_helmet_craters (dans_craters daniel_craters rins_craters : ℕ) : 
  dans_craters = daniel_craters + 10 →
  rins_craters = dans_craters + daniel_craters + 15 →
  rins_craters = 75 →
  dans_craters = 35 := by
  sorry

end dans_helmet_craters_l2546_254663


namespace complement_intersection_problem_l2546_254623

theorem complement_intersection_problem (U M N : Set Nat) : 
  U = {1, 2, 3, 4, 5} →
  M = {1, 2, 3} →
  N = {2, 3, 5} →
  (U \ M) ∩ N = {5} := by
  sorry

end complement_intersection_problem_l2546_254623


namespace house_transaction_loss_l2546_254694

theorem house_transaction_loss (initial_value : ℝ) (loss_percent : ℝ) (gain_percent : ℝ) : 
  initial_value = 12000 →
  loss_percent = 0.15 →
  gain_percent = 0.20 →
  let first_sale := initial_value * (1 - loss_percent)
  let second_sale := first_sale * (1 + gain_percent)
  second_sale - initial_value = 2040 :=
by sorry

end house_transaction_loss_l2546_254694


namespace not_p_sufficient_not_necessary_for_not_q_l2546_254609

-- Define the conditions p and q
def p (x : ℝ) : Prop := x - 3 > 0
def q (x : ℝ) : Prop := (x - 3) * (x - 4) < 0

-- State the theorem
theorem not_p_sufficient_not_necessary_for_not_q :
  (∀ x, ¬(p x) → ¬(q x)) ∧ 
  ∃ x, ¬(q x) ∧ p x :=
sorry

end not_p_sufficient_not_necessary_for_not_q_l2546_254609


namespace martin_ticket_count_l2546_254655

/-- The number of tickets Martin bought at full price -/
def full_price_tickets : ℕ := sorry

/-- The price of a full-price ticket in cents -/
def full_price : ℕ := 200

/-- The number of discounted tickets Martin bought -/
def discounted_tickets : ℕ := 4

/-- The price of a discounted ticket in cents -/
def discounted_price : ℕ := 160

/-- The total amount Martin spent in cents -/
def total_spent : ℕ := 1840

theorem martin_ticket_count :
  full_price_tickets * full_price + discounted_tickets * discounted_price = total_spent ∧
  full_price_tickets + discounted_tickets = 10 :=
sorry

end martin_ticket_count_l2546_254655


namespace parabola_intersection_implies_nonzero_c_l2546_254618

/-- Two points on a parabola -/
structure ParabolaPoints (a b c : ℝ) :=
  (x₁ : ℝ)
  (x₂ : ℝ)
  (y₁ : ℝ)
  (y₂ : ℝ)
  (on_parabola₁ : y₁ = x₁^2)
  (on_parabola₂ : y₂ = x₂^2)
  (on_quadratic₁ : y₁ = a * x₁^2 + b * x₁ + c)
  (on_quadratic₂ : y₂ = a * x₂^2 + b * x₂ + c)
  (opposite_sides : x₁ * x₂ < 0)
  (right_angle : (x₁ - x₂)^2 + (y₁ - y₂)^2 = x₁^2 + y₁^2 + x₂^2 + y₂^2)

theorem parabola_intersection_implies_nonzero_c (a b c : ℝ) :
  (∃ p : ParabolaPoints a b c, True) → c ≠ 0 :=
by sorry

end parabola_intersection_implies_nonzero_c_l2546_254618
