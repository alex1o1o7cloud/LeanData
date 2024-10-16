import Mathlib

namespace NUMINAMATH_CALUDE_f_and_g_odd_and_increasing_l1159_115964

-- Define the functions
def f (x : ℝ) : ℝ := 6 * x
def g (x : ℝ) : ℝ := x * |x|

-- Define what it means for a function to be odd
def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- Define what it means for a function to be increasing
def is_increasing (f : ℝ → ℝ) : Prop := ∀ x y, x < y → f x < f y

-- Theorem statement
theorem f_and_g_odd_and_increasing :
  (is_odd f ∧ is_increasing f) ∧ (is_odd g ∧ is_increasing g) := by sorry

end NUMINAMATH_CALUDE_f_and_g_odd_and_increasing_l1159_115964


namespace NUMINAMATH_CALUDE_equilateral_triangle_perimeter_equilateral_triangle_perimeter_is_60_l1159_115956

/-- The perimeter of an equilateral triangle, given its relationship with an isosceles triangle -/
theorem equilateral_triangle_perimeter : ℝ :=
  let equilateral_side : ℝ := sorry
  let isosceles_base : ℝ := 10
  let isosceles_perimeter : ℝ := 50
  have h1 : isosceles_perimeter = 2 * equilateral_side + isosceles_base := by sorry
  have h2 : equilateral_side = (isosceles_perimeter - isosceles_base) / 2 := by sorry
  3 * equilateral_side

/-- Proof that the perimeter of the equilateral triangle is 60 -/
theorem equilateral_triangle_perimeter_is_60 :
  equilateral_triangle_perimeter = 60 := by sorry

end NUMINAMATH_CALUDE_equilateral_triangle_perimeter_equilateral_triangle_perimeter_is_60_l1159_115956


namespace NUMINAMATH_CALUDE_line_through_points_l1159_115939

/-- A line passing through two points (3,1) and (7,13) has equation y = ax + b. This theorem proves that a - b = 11. -/
theorem line_through_points (a b : ℝ) : 
  (1 = a * 3 + b) → (13 = a * 7 + b) → a - b = 11 := by sorry

end NUMINAMATH_CALUDE_line_through_points_l1159_115939


namespace NUMINAMATH_CALUDE_power_sum_equality_l1159_115923

theorem power_sum_equality (a b c d : ℝ) 
  (h1 : a + b = c + d) 
  (h2 : a^3 + b^3 = c^3 + d^3) : 
  (a^5 + b^5 = c^5 + d^5) ∧ 
  (∃ a b c d : ℝ, (a + b = c + d) ∧ (a^3 + b^3 = c^3 + d^3) ∧ (a^4 + b^4 ≠ c^4 + d^4)) :=
by sorry

end NUMINAMATH_CALUDE_power_sum_equality_l1159_115923


namespace NUMINAMATH_CALUDE_ellipse_intersection_fixed_point_l1159_115933

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2/4 + y^2 = 1

-- Define the line
def line (k : ℝ) (x y : ℝ) : Prop := x = k*y - 1

-- Define the intersection points
def intersection_points (k : ℝ) (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  ellipse x₁ y₁ ∧ ellipse x₂ y₂ ∧ line k x₁ y₁ ∧ line k x₂ y₂

-- Define the reflection point
def reflection_point (x₁ y₁ : ℝ) : ℝ × ℝ := (x₁, -y₁)

-- Theorem statement
theorem ellipse_intersection_fixed_point (k : ℝ) (x₁ y₁ x₂ y₂ : ℝ) :
  intersection_points k x₁ y₁ x₂ y₂ →
  let (x₁', y₁') := reflection_point x₁ y₁
  (x₁' ≠ x₂ ∨ y₁' ≠ y₂) →
  ∃ (t : ℝ), t * (x₂ - x₁') + x₁' = -4 ∧ t * (y₂ - y₁') + y₁' = 0 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_intersection_fixed_point_l1159_115933


namespace NUMINAMATH_CALUDE_beads_per_package_l1159_115934

theorem beads_per_package (total_packages : Nat) (total_beads : Nat) : 
  total_packages = 8 → total_beads = 320 → (total_beads / total_packages : Nat) = 40 := by
  sorry

end NUMINAMATH_CALUDE_beads_per_package_l1159_115934


namespace NUMINAMATH_CALUDE_like_terms_exponents_l1159_115984

theorem like_terms_exponents (a b : ℝ) (x y : ℝ) : 
  (∃ k : ℝ, 2 * a^(2*x) * b^(3*y) = k * (-3 * a^2 * b^(2-x))) → 
  x = 1 ∧ y = 1/3 :=
sorry

end NUMINAMATH_CALUDE_like_terms_exponents_l1159_115984


namespace NUMINAMATH_CALUDE_unique_solution_trigonometric_equation_l1159_115985

theorem unique_solution_trigonometric_equation :
  ∃! (n : ℕ+), (Real.sin (π / (3 * n.val)) + Real.cos (π / (3 * n.val)) = Real.sqrt (3 * n.val) / 3) ∧ n = 6 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_trigonometric_equation_l1159_115985


namespace NUMINAMATH_CALUDE_correct_time_to_write_rearrangements_l1159_115927

/-- The number of unique letters in the name --/
def num_letters : ℕ := 8

/-- The number of rearrangements that can be written per minute --/
def rearrangements_per_minute : ℕ := 10

/-- The number of minutes in an hour --/
def minutes_per_hour : ℕ := 60

/-- Calculates the time in hours to write all rearrangements of a name --/
def time_to_write_all_rearrangements : ℚ :=
  (Nat.factorial num_letters : ℚ) / (rearrangements_per_minute * minutes_per_hour)

theorem correct_time_to_write_rearrangements :
  time_to_write_all_rearrangements = 67.2 := by sorry

end NUMINAMATH_CALUDE_correct_time_to_write_rearrangements_l1159_115927


namespace NUMINAMATH_CALUDE_negation_of_proposition_l1159_115936

theorem negation_of_proposition :
  (¬ ∀ (a : ℝ) (n : ℕ), n > 0 → (a ≠ n → a * n ≠ 2 * n)) ↔
  (∃ (a : ℝ) (n : ℕ), n > 0 ∧ a ≠ n ∧ a * n = 2 * n) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l1159_115936


namespace NUMINAMATH_CALUDE_no_integer_solutions_l1159_115947

theorem no_integer_solutions : ¬ ∃ (x y : ℤ), x^3 + 21*y^2 + 5 = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solutions_l1159_115947


namespace NUMINAMATH_CALUDE_system_solution_l1159_115977

theorem system_solution (x y : ℝ) : 
  x + 2*y = 8 → 2*x + y = -5 → x + y = 1 := by sorry

end NUMINAMATH_CALUDE_system_solution_l1159_115977


namespace NUMINAMATH_CALUDE_five_digit_divisible_by_6_l1159_115929

def is_divisible_by_6 (n : Nat) : Prop :=
  ∃ k : Nat, 7123 * 10 + n = 6 * k

theorem five_digit_divisible_by_6 :
  ∀ n : Nat, n < 10 →
    (is_divisible_by_6 n ↔ (n = 2 ∨ n = 8)) :=
by sorry

end NUMINAMATH_CALUDE_five_digit_divisible_by_6_l1159_115929


namespace NUMINAMATH_CALUDE_expression_simplification_l1159_115945

theorem expression_simplification (a b : ℝ) (h1 : a = 2) (h2 : b = 3) :
  3 * a^2 * b - (a * b^2 - 2 * (2 * a^2 * b - a * b^2)) - a * b^2 = 12 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1159_115945


namespace NUMINAMATH_CALUDE_trigonometric_identities_l1159_115937

theorem trigonometric_identities :
  (Real.cos (2 * Real.pi / 5) - Real.cos (4 * Real.pi / 5) = Real.sqrt 5 / 2) ∧
  (Real.sin (2 * Real.pi / 7) + Real.sin (4 * Real.pi / 7) - Real.sin (6 * Real.pi / 7) = Real.sqrt 7 / 2) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identities_l1159_115937


namespace NUMINAMATH_CALUDE_eleven_twelfths_squared_between_half_and_one_l1159_115965

theorem eleven_twelfths_squared_between_half_and_one :
  (11 / 12 : ℚ)^2 > 1/2 ∧ (11 / 12 : ℚ)^2 < 1 := by
  sorry

end NUMINAMATH_CALUDE_eleven_twelfths_squared_between_half_and_one_l1159_115965


namespace NUMINAMATH_CALUDE_expression_simplification_l1159_115924

theorem expression_simplification (x y : ℝ) 
  (hx : x = Real.tan (60 * π / 180)^2 + 1)
  (hy : y = Real.tan (45 * π / 180) - 2 * Real.cos (30 * π / 180)) :
  (x - (2*x*y - y^2) / x) / ((x^2 - y^2) / (x^2 + x*y)) = 3 + Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1159_115924


namespace NUMINAMATH_CALUDE_parallelogram_area_example_l1159_115931

/-- The area of a parallelogram defined by two vectors -/
def parallelogramArea (v w : ℝ × ℝ) : ℝ :=
  |v.1 * w.2 - v.2 * w.1|

theorem parallelogram_area_example : 
  let v : ℝ × ℝ := (6, -4)
  let w : ℝ × ℝ := (13, -1)
  parallelogramArea v w = 46 := by
sorry

end NUMINAMATH_CALUDE_parallelogram_area_example_l1159_115931


namespace NUMINAMATH_CALUDE_sum_f_1_to_10_l1159_115968

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def is_periodic_3 (f : ℝ → ℝ) : Prop := ∀ x, f (x + 3) = f x

theorem sum_f_1_to_10 (f : ℝ → ℝ) 
  (h_odd : is_odd f) 
  (h_periodic : is_periodic_3 f) 
  (h_f_neg_1 : f (-1) = 1) : 
  f 1 + f 2 + f 3 + f 4 + f 5 + f 6 + f 7 + f 8 + f 9 + f 10 = -1 :=
sorry

end NUMINAMATH_CALUDE_sum_f_1_to_10_l1159_115968


namespace NUMINAMATH_CALUDE_circle_roll_position_l1159_115992

theorem circle_roll_position (d : ℝ) (start : ℝ) (h_d : d = 1) (h_start : start = 3) : 
  let circumference := π * d
  let end_position := start - circumference
  end_position = 3 - π := by
sorry

end NUMINAMATH_CALUDE_circle_roll_position_l1159_115992


namespace NUMINAMATH_CALUDE_juans_number_l1159_115943

theorem juans_number (x : ℝ) : ((3 * (x + 3) - 4) / 2 = 10) → x = 5 := by
  sorry

end NUMINAMATH_CALUDE_juans_number_l1159_115943


namespace NUMINAMATH_CALUDE_race_distance_proof_l1159_115978

/-- The distance of a dogsled race course in Wyoming --/
def race_distance : ℝ := 300

/-- The average speed of Team R in mph --/
def team_r_speed : ℝ := 20

/-- The time difference between Team A and Team R in hours --/
def time_difference : ℝ := 3

/-- The speed difference between Team A and Team R in mph --/
def speed_difference : ℝ := 5

/-- Theorem stating the race distance given the conditions --/
theorem race_distance_proof :
  ∃ (t : ℝ), 
    t > 0 ∧
    race_distance = team_r_speed * t ∧
    race_distance = (team_r_speed + speed_difference) * (t - time_difference) :=
by
  sorry

#check race_distance_proof

end NUMINAMATH_CALUDE_race_distance_proof_l1159_115978


namespace NUMINAMATH_CALUDE_antonia_pills_left_l1159_115976

/-- Calculates the number of pills left after taking supplements for two weeks -/
def pills_left (bottles_120 : Nat) (bottles_30 : Nat) (supplements : Nat) (weeks : Nat) : Nat :=
  let total_pills := bottles_120 * 120 + bottles_30 * 30
  let days := weeks * 7
  let pills_used := days * supplements
  total_pills - pills_used

/-- Theorem stating that given the specific conditions, the number of pills left is 350 -/
theorem antonia_pills_left :
  pills_left 3 2 5 2 = 350 := by
  sorry

end NUMINAMATH_CALUDE_antonia_pills_left_l1159_115976


namespace NUMINAMATH_CALUDE_function_is_linear_l1159_115963

/-- Given a real number k, we define a function f that satisfies two conditions -/
def satisfies_conditions (k : ℝ) (f : ℝ → ℝ) : Prop :=
  (∀ x y : ℝ, f (x^2 + 2*x*y + y^2) = (x + y) * (f x + f y)) ∧ 
  (∀ x : ℝ, |f x - k*x| ≤ |x^2 - x|)

/-- Theorem stating that if f satisfies the conditions, then f(x) = kx for all x ∈ ℝ -/
theorem function_is_linear (k : ℝ) (f : ℝ → ℝ) 
  (h : satisfies_conditions k f) : 
  ∀ x : ℝ, f x = k * x :=
sorry

end NUMINAMATH_CALUDE_function_is_linear_l1159_115963


namespace NUMINAMATH_CALUDE_midnight_temperature_l1159_115912

def morning_temp : Int := 7
def noon_rise : Int := 2
def midnight_drop : Int := 10

theorem midnight_temperature : 
  morning_temp + noon_rise - midnight_drop = -1 := by sorry

end NUMINAMATH_CALUDE_midnight_temperature_l1159_115912


namespace NUMINAMATH_CALUDE_socks_worn_l1159_115962

/-- Given 3 pairs of socks, if the number of pairs that can be formed from worn socks
    (where no worn socks are from the same original pair) is 6,
    then the number of socks worn is 3. -/
theorem socks_worn (total_pairs : ℕ) (formed_pairs : ℕ) (worn_socks : ℕ) :
  total_pairs = 3 →
  formed_pairs = 6 →
  worn_socks ≤ total_pairs * 2 →
  (∀ (i j : ℕ), i < worn_socks → j < worn_socks → i ≠ j →
    ∃ (p q : ℕ), p < total_pairs → q < total_pairs → p ≠ q) →
  (formed_pairs = worn_socks.choose 2) →
  worn_socks = 3 := by
sorry

end NUMINAMATH_CALUDE_socks_worn_l1159_115962


namespace NUMINAMATH_CALUDE_readers_overlap_l1159_115910

theorem readers_overlap (total : ℕ) (science_fiction : ℕ) (literary : ℕ) 
  (h1 : total = 650) 
  (h2 : science_fiction = 250) 
  (h3 : literary = 550) : 
  total = science_fiction + literary - 150 := by
  sorry

#check readers_overlap

end NUMINAMATH_CALUDE_readers_overlap_l1159_115910


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l1159_115922

def vector_a : ℝ × ℝ := (1, 2)
def vector_b (x : ℝ) : ℝ × ℝ := (2*x, -3)

def parallel (v w : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), v.1 = k * w.1 ∧ v.2 = k * w.2

theorem parallel_vectors_x_value :
  parallel vector_a (vector_b x) → x = -3/4 :=
by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l1159_115922


namespace NUMINAMATH_CALUDE_set_distributive_laws_l1159_115917

theorem set_distributive_laws {α : Type*} (A B C : Set α) :
  (A ∪ (B ∩ C) = (A ∪ B) ∩ (A ∪ C)) ∧
  (A ∩ (B ∪ C) = (A ∩ B) ∪ (A ∩ C)) := by
  sorry

end NUMINAMATH_CALUDE_set_distributive_laws_l1159_115917


namespace NUMINAMATH_CALUDE_sum_of_solutions_l1159_115941

theorem sum_of_solutions (a b : ℝ) (ha : a > 1) (hb : b > 0) :
  let f : ℝ → ℝ := λ x => Real.sqrt (a - Real.sqrt (a + b^x))
  ∃ x : ℝ, f x = x ∧
  (∀ y : ℝ, f y = y → y ≤ x) ∧
  x = (Real.sqrt (4 * a - 3 * b) - 1) / 2 := by
sorry

end NUMINAMATH_CALUDE_sum_of_solutions_l1159_115941


namespace NUMINAMATH_CALUDE_last_two_digits_product_l1159_115988

theorem last_two_digits_product (n : ℕ) : 
  (n % 100 ≥ 10) →  -- Ensure n has at least two digits
  (n % 6 = 0) →     -- n is divisible by 6
  ((n % 100) / 10 + n % 10 = 15) →  -- Sum of last two digits is 15
  (((n % 100) / 10) * (n % 10) = 56 ∨ ((n % 100) / 10) * (n % 10) = 54) :=
by sorry

end NUMINAMATH_CALUDE_last_two_digits_product_l1159_115988


namespace NUMINAMATH_CALUDE_hexagonal_pyramid_base_edge_length_l1159_115907

/-- A pyramid with a regular hexagon base -/
structure HexagonalPyramid where
  base_edge_length : ℝ
  side_edge_length : ℝ
  total_edge_length : ℝ

/-- The property that the pyramid satisfies the given conditions -/
def satisfies_conditions (p : HexagonalPyramid) : Prop :=
  p.side_edge_length = 8 ∧ p.total_edge_length = 120

/-- The theorem stating that if a hexagonal pyramid satisfies the conditions, 
    its base edge length is 12 -/
theorem hexagonal_pyramid_base_edge_length 
  (p : HexagonalPyramid) (h : satisfies_conditions p) : 
  p.base_edge_length = 12 := by
  sorry

end NUMINAMATH_CALUDE_hexagonal_pyramid_base_edge_length_l1159_115907


namespace NUMINAMATH_CALUDE_average_of_tenths_and_thousandths_l1159_115957

theorem average_of_tenths_and_thousandths :
  let a : ℚ := 4/10
  let b : ℚ := 5/1000
  (a + b) / 2 = 2025/10000 := by
  sorry

end NUMINAMATH_CALUDE_average_of_tenths_and_thousandths_l1159_115957


namespace NUMINAMATH_CALUDE_sqrt_27_div_sqrt_3_eq_3_l1159_115975

theorem sqrt_27_div_sqrt_3_eq_3 : Real.sqrt 27 / Real.sqrt 3 = 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_27_div_sqrt_3_eq_3_l1159_115975


namespace NUMINAMATH_CALUDE_second_discount_calculation_l1159_115935

/-- Given an initial price increase, two successive discounts, and the overall gain/loss,
    this theorem proves the relationship between these values. -/
theorem second_discount_calculation (initial_increase : ℝ) (first_discount : ℝ) 
    (overall_factor : ℝ) (second_discount : ℝ) : 
    initial_increase = 0.32 → first_discount = 0.10 → overall_factor = 0.98 →
    overall_factor = (1 - second_discount) * (1 + initial_increase) * (1 - first_discount) := by
  sorry

end NUMINAMATH_CALUDE_second_discount_calculation_l1159_115935


namespace NUMINAMATH_CALUDE_motorcycle_price_increase_l1159_115986

/-- Represents the price increase of a motorcycle model --/
def price_increase (original_price : ℝ) (new_price : ℝ) : ℝ :=
  new_price - original_price

/-- Theorem stating the price increase given the problem conditions --/
theorem motorcycle_price_increase :
  ∀ (original_price : ℝ) (original_quantity : ℕ) (new_quantity : ℕ) (revenue_increase : ℝ),
    original_quantity = new_quantity + 8 →
    new_quantity = 63 →
    revenue_increase = 26000 →
    original_price * original_quantity = 594000 - revenue_increase →
    (original_price + price_increase original_price (original_price + price_increase original_price original_price)) * new_quantity = 594000 →
    price_increase original_price (original_price + price_increase original_price original_price) = 1428.57 := by
  sorry


end NUMINAMATH_CALUDE_motorcycle_price_increase_l1159_115986


namespace NUMINAMATH_CALUDE_ellipse_circle_fixed_point_l1159_115993

/-- Given an ellipse with equation (x²/a²) + (y²/b²) = 1, where a > b > 0,
    and a point P(x₀, y₀) on the ellipse different from A₁(-a, 0) and A(a, 0),
    the circle with diameter MM₁ (where M and M₁ are intersections of PA and PA₁
    with the directrix x = a²/c) passes through a fixed point outside the ellipse. -/
theorem ellipse_circle_fixed_point
  (a b c : ℝ) (h_a_pos : a > 0) (h_b_pos : b > 0) (h_c_pos : c > 0) (h_a_gt_b : a > b)
  (x₀ y₀ : ℝ) (h_on_ellipse : x₀^2 / a^2 + y₀^2 / b^2 = 1)
  (h_not_A : x₀ ≠ a ∨ y₀ ≠ 0) (h_not_A₁ : x₀ ≠ -a ∨ y₀ ≠ 0) :
  ∃ (x y : ℝ), x = (a^2 + b^2) / c ∧ y = 0 ∧
  (x - a^2 / c)^2 + (y + b^2 * (x₀ - c) / (c * y₀))^2 = (b^2 * (c * x₀ - a^2) / (a * c * y₀))^2 :=
sorry

end NUMINAMATH_CALUDE_ellipse_circle_fixed_point_l1159_115993


namespace NUMINAMATH_CALUDE_hiking_problem_l1159_115959

/-- A hiking problem with two trails -/
theorem hiking_problem (trail1_length trail1_speed trail2_speed : ℝ)
  (break_time time_difference : ℝ) :
  trail1_length = 20 ∧
  trail1_speed = 5 ∧
  trail2_speed = 3 ∧
  break_time = 1 ∧
  time_difference = 1 ∧
  (trail1_length / trail1_speed = 
    (trail1_length / trail1_speed + time_difference)) →
  ∃ trail2_length : ℝ,
    trail2_length / trail2_speed / 2 + break_time + 
    trail2_length / trail2_speed / 2 = 
    trail1_length / trail1_speed + time_difference ∧
    trail2_length = 12 :=
by sorry


end NUMINAMATH_CALUDE_hiking_problem_l1159_115959


namespace NUMINAMATH_CALUDE_sin_2theta_value_l1159_115979

def line1 (θ : ℝ) : ℝ → ℝ → Prop :=
  fun x y ↦ x * Real.cos θ + 2 * y = 0

def line2 (θ : ℝ) : ℝ → ℝ → Prop :=
  fun x y ↦ 3 * x + y * Real.sin θ + 3 = 0

def perpendicular (f g : ℝ → ℝ → Prop) : Prop :=
  ∃ x₁ y₁ x₂ y₂, f x₁ y₁ ∧ g x₂ y₂ ∧ 
    (y₂ - y₁) * (x₂ - x₁) + (x₂ - x₁) * (y₂ - y₁) = 0

theorem sin_2theta_value (θ : ℝ) :
  perpendicular (line1 θ) (line2 θ) → Real.sin (2 * θ) = -12/13 := by
  sorry

end NUMINAMATH_CALUDE_sin_2theta_value_l1159_115979


namespace NUMINAMATH_CALUDE_susan_carol_car_difference_l1159_115996

/-- Prove that Susan owns 2 fewer cars than Carol given the conditions -/
theorem susan_carol_car_difference :
  let cathy_cars : ℕ := 5
  let carol_cars : ℕ := 2 * cathy_cars
  let lindsey_cars : ℕ := cathy_cars + 4
  let total_cars : ℕ := 32
  let susan_cars : ℕ := total_cars - (cathy_cars + carol_cars + lindsey_cars)
  susan_cars < carol_cars ∧ carol_cars - susan_cars = 2 := by
  sorry

end NUMINAMATH_CALUDE_susan_carol_car_difference_l1159_115996


namespace NUMINAMATH_CALUDE_negation_of_existential_proposition_l1159_115915

theorem negation_of_existential_proposition :
  (¬ ∃ x : ℝ, x^3 > x) ↔ (∀ x : ℝ, x^3 ≤ x) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_existential_proposition_l1159_115915


namespace NUMINAMATH_CALUDE_vectors_form_basis_vectors_not_collinear_basis_iff_not_collinear_l1159_115991

def e₁ : Fin 2 → ℝ := ![(-1 : ℝ), 2]
def e₂ : Fin 2 → ℝ := ![(5 : ℝ), -1]

theorem vectors_form_basis (v : Fin 2 → ℝ) : 
  ∃ (a b : ℝ), v = fun i => a * e₁ i + b * e₂ i :=
sorry

theorem vectors_not_collinear : 
  e₁ 0 * e₂ 1 ≠ e₁ 1 * e₂ 0 :=
sorry

theorem basis_iff_not_collinear :
  (∀ (v : Fin 2 → ℝ), ∃ (a b : ℝ), v = fun i => a * e₁ i + b * e₂ i) ↔
  (e₁ 0 * e₂ 1 ≠ e₁ 1 * e₂ 0) :=
sorry

end NUMINAMATH_CALUDE_vectors_form_basis_vectors_not_collinear_basis_iff_not_collinear_l1159_115991


namespace NUMINAMATH_CALUDE_zeoland_speeding_fine_l1159_115966

/-- The speeding fine structure in Zeoland -/
structure SpeedingFine where
  totalFine : ℕ      -- Total fine amount
  speedLimit : ℕ     -- Posted speed limit
  actualSpeed : ℕ    -- Actual speed of the driver
  finePerMph : ℕ     -- Fine per mile per hour over the limit

/-- Theorem: Given Jed's speeding fine details, prove the fine per mph over the limit -/
theorem zeoland_speeding_fine (fine : SpeedingFine) 
  (h1 : fine.totalFine = 256)
  (h2 : fine.speedLimit = 50)
  (h3 : fine.actualSpeed = 66) :
  fine.finePerMph = 16 := by
  sorry


end NUMINAMATH_CALUDE_zeoland_speeding_fine_l1159_115966


namespace NUMINAMATH_CALUDE_value_of_c_l1159_115944

theorem value_of_c (a b c : ℝ) (h1 : a = 6) (h2 : b = 15) (h3 : 6 * 15 * c = 3) :
  (a * b * c = Real.sqrt ((a + 2) * (b + 3)) / (c + 1)) ↔ c = 3 := by
sorry

end NUMINAMATH_CALUDE_value_of_c_l1159_115944


namespace NUMINAMATH_CALUDE_street_trees_count_l1159_115970

theorem street_trees_count (road_length : ℕ) (interval : ℕ) : 
  road_length = 2575 → interval = 25 → (road_length / interval + 1 : ℕ) = 104 := by
  sorry

end NUMINAMATH_CALUDE_street_trees_count_l1159_115970


namespace NUMINAMATH_CALUDE_triangle_perimeter_is_seven_l1159_115983

-- Define the triangle sides
variable (a b c : ℝ)

-- Define the condition equation
def condition (a b c : ℝ) : Prop :=
  a^2 + b^2 + c^2 - 4*a - 4*b - 6*c + 17 = 0

-- Define what it means for a, b, c to form a triangle
def is_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b

-- Define the perimeter
def perimeter (a b c : ℝ) : ℝ := a + b + c

-- State the theorem
theorem triangle_perimeter_is_seven 
  (h1 : is_triangle a b c) 
  (h2 : condition a b c) : 
  perimeter a b c = 7 :=
sorry

end NUMINAMATH_CALUDE_triangle_perimeter_is_seven_l1159_115983


namespace NUMINAMATH_CALUDE_division_remainder_proof_l1159_115901

theorem division_remainder_proof (L S : ℕ) (h1 : L - S = 1395) (h2 : L = 1656) 
  (h3 : ∃ q r, L = S * q + r ∧ q = 6 ∧ r < S) : 
  ∃ r, L = S * 6 + r ∧ r = 90 := by
sorry

end NUMINAMATH_CALUDE_division_remainder_proof_l1159_115901


namespace NUMINAMATH_CALUDE_wedding_decoration_cost_l1159_115913

/-- Calculates the total cost of decorations for Nathan's wedding reception --/
def total_decoration_cost (num_tables : ℕ) 
  (tablecloth_cost service_charge place_setting_cost place_settings_per_table : ℝ)
  (roses_per_centerpiece rose_cost rose_discount : ℝ)
  (lilies_per_centerpiece lily_cost lily_discount : ℝ)
  (daisies_per_centerpiece daisy_cost : ℝ)
  (sunflowers_per_centerpiece sunflower_cost : ℝ)
  (lighting_cost : ℝ) : ℝ :=
  let tablecloth_total := num_tables * tablecloth_cost * (1 + service_charge)
  let place_settings_total := num_tables * place_settings_per_table * place_setting_cost
  let centerpiece_cost := 
    (roses_per_centerpiece * rose_cost * (1 - rose_discount)) +
    (lilies_per_centerpiece * lily_cost * (1 - lily_discount)) +
    (daisies_per_centerpiece * daisy_cost) +
    (sunflowers_per_centerpiece * sunflower_cost)
  let centerpiece_total := num_tables * centerpiece_cost
  tablecloth_total + place_settings_total + centerpiece_total + lighting_cost

/-- Theorem stating the total cost of decorations for Nathan's wedding reception --/
theorem wedding_decoration_cost : 
  total_decoration_cost 30 25 0.15 12 6 15 6 0.1 20 5 0.05 5 3 3 4 450 = 9562.50 := by
  sorry

end NUMINAMATH_CALUDE_wedding_decoration_cost_l1159_115913


namespace NUMINAMATH_CALUDE_problem_solution_l1159_115995

theorem problem_solution (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h1 : x - 4 = 21 * (1 / x)) (h2 : x + y^2 = 45) : x = 7 ∧ y = Real.sqrt 38 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1159_115995


namespace NUMINAMATH_CALUDE_statements_equivalence_l1159_115961

variable (α : Type)
variable (A B : α → Prop)

theorem statements_equivalence :
  (∀ x, A x → B x) ↔
  (∀ x, A x → B x) ∧
  (∀ x, A x → B x) ∧
  (∀ x, A x → B x) ∧
  (∀ x, ¬ B x → ¬ A x) :=
by sorry

end NUMINAMATH_CALUDE_statements_equivalence_l1159_115961


namespace NUMINAMATH_CALUDE_student_gathering_problem_l1159_115906

theorem student_gathering_problem (male_count : ℕ) (female_count : ℕ) : 
  female_count = male_count + 6 →
  (female_count : ℚ) / (male_count + female_count) = 2 / 3 →
  male_count + female_count = 18 :=
by sorry

end NUMINAMATH_CALUDE_student_gathering_problem_l1159_115906


namespace NUMINAMATH_CALUDE_girls_same_color_marble_l1159_115982

-- Define the total number of marbles
def total_marbles : ℕ := 4

-- Define the number of white marbles
def white_marbles : ℕ := 2

-- Define the number of black marbles
def black_marbles : ℕ := 2

-- Define the number of girls selecting marbles
def girls : ℕ := 2

-- Define the probability of both girls selecting the same colored marble
def prob_same_color : ℚ := 1 / 3

-- Theorem statement
theorem girls_same_color_marble :
  (total_marbles = white_marbles + black_marbles) →
  (white_marbles = black_marbles) →
  (girls = 2) →
  (prob_same_color = 1 / 3) := by
sorry

end NUMINAMATH_CALUDE_girls_same_color_marble_l1159_115982


namespace NUMINAMATH_CALUDE_equation_linearity_l1159_115972

/-- The equation (k^2 - 1)x^2 + (k + 1)x + (k - 7)y = k + 2 -/
def equation (k x y : ℝ) : Prop :=
  (k^2 - 1) * x^2 + (k + 1) * x + (k - 7) * y = k + 2

/-- The equation is linear in one variable -/
def is_linear_one_var (k : ℝ) : Prop :=
  k^2 - 1 = 0 ∧ k + 1 = 0

/-- The equation is linear in two variables -/
def is_linear_two_var (k : ℝ) : Prop :=
  k^2 - 1 = 0 ∧ k + 1 ≠ 0

theorem equation_linearity :
  (is_linear_one_var (-1) ∧ is_linear_two_var 1) :=
by sorry

end NUMINAMATH_CALUDE_equation_linearity_l1159_115972


namespace NUMINAMATH_CALUDE_ceiling_sqrt_225_l1159_115953

theorem ceiling_sqrt_225 : ⌈Real.sqrt 225⌉ = 15 := by sorry

end NUMINAMATH_CALUDE_ceiling_sqrt_225_l1159_115953


namespace NUMINAMATH_CALUDE_cone_sphere_ratio_angle_and_k_permissible_k_values_l1159_115930

-- Define the cone and sphere
structure ConeWithInscribedSphere where
  R : ℝ  -- radius of the cone's base
  α : ℝ  -- angle between slant height and base plane
  k : ℝ  -- ratio of cone volume to sphere volume

-- Define the theorem
theorem cone_sphere_ratio_angle_and_k (c : ConeWithInscribedSphere) :
  c.k ≥ 2 →
  c.α = 2 * Real.arctan (Real.sqrt ((c.k + Real.sqrt (c.k^2 - 2*c.k)) / (2*c.k))) ∨
  c.α = 2 * Real.arctan (Real.sqrt ((c.k - Real.sqrt (c.k^2 - 2*c.k)) / (2*c.k))) :=
by sorry

-- Define the permissible values of k
theorem permissible_k_values (c : ConeWithInscribedSphere) :
  c.k ≥ 2 :=
by sorry

end NUMINAMATH_CALUDE_cone_sphere_ratio_angle_and_k_permissible_k_values_l1159_115930


namespace NUMINAMATH_CALUDE_equation_solution_l1159_115951

theorem equation_solution : ∃! x : ℝ, 3 * (5 - x) = 9 ∧ x = 2 := by sorry

end NUMINAMATH_CALUDE_equation_solution_l1159_115951


namespace NUMINAMATH_CALUDE_geometric_sequence_n_l1159_115989

/-- For a geometric sequence {a_n} with a₁ = 9/8, q = 2/3, and aₙ = 1/3, n = 4 -/
theorem geometric_sequence_n (a : ℕ → ℚ) :
  (∀ k, a (k + 1) = a k * (2/3)) →  -- geometric sequence condition
  a 1 = 9/8 →                      -- first term condition
  (∃ n, a n = 1/3) →               -- nth term condition
  ∃ n, n = 4 ∧ a n = 1/3 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_n_l1159_115989


namespace NUMINAMATH_CALUDE_mens_wages_l1159_115911

theorem mens_wages (total_earnings : ℝ) (num_men : ℕ) (num_boys : ℕ)
  (h_total : total_earnings = 432)
  (h_men : num_men = 15)
  (h_boys : num_boys = 12)
  (h_equal_earnings : ∃ (num_women : ℕ), num_men * (total_earnings / (num_men + num_women + num_boys)) = 
                                         num_women * (total_earnings / (num_men + num_women + num_boys)) ∧
                                         num_women * (total_earnings / (num_men + num_women + num_boys)) = 
                                         num_boys * (total_earnings / (num_men + num_women + num_boys))) :
  num_men * (total_earnings / (num_men + num_men + num_men)) = 144 := by
  sorry

end NUMINAMATH_CALUDE_mens_wages_l1159_115911


namespace NUMINAMATH_CALUDE_square_sum_of_two_integers_l1159_115967

theorem square_sum_of_two_integers (x y : ℕ+) 
  (h1 : x * y + x + y = 17)
  (h2 : x^2 * y + x * y^2 = 72) : 
  x^2 + y^2 = 65 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_of_two_integers_l1159_115967


namespace NUMINAMATH_CALUDE_anniversary_sale_cost_l1159_115946

def original_ice_cream_price : ℚ := 12
def ice_cream_discount : ℚ := 2
def juice_price_per_5_cans : ℚ := 2
def ice_cream_tubs : ℕ := 2
def juice_cans : ℕ := 10

theorem anniversary_sale_cost : 
  (ice_cream_tubs * (original_ice_cream_price - ice_cream_discount)) + 
  (juice_cans / 5 * juice_price_per_5_cans) = 24 := by
  sorry

end NUMINAMATH_CALUDE_anniversary_sale_cost_l1159_115946


namespace NUMINAMATH_CALUDE_polynomial_division_l1159_115949

def dividend (x : ℚ) : ℚ := 10*x^4 + 5*x^3 - 9*x^2 + 7*x + 2
def divisor (x : ℚ) : ℚ := 3*x^2 + 2*x + 1
def quotient (x : ℚ) : ℚ := (10/3)*x^2 - (5/9)*x - 193/243
def remainder (x : ℚ) : ℚ := (592/27)*x + 179/27

theorem polynomial_division :
  ∀ x : ℚ, dividend x = divisor x * quotient x + remainder x :=
by sorry

end NUMINAMATH_CALUDE_polynomial_division_l1159_115949


namespace NUMINAMATH_CALUDE_mark_and_carolyn_money_l1159_115909

theorem mark_and_carolyn_money (mark_money : ℚ) (carolyn_money : ℚ) : 
  mark_money = 5/8 → carolyn_money = 2/5 → mark_money + carolyn_money = 1.025 := by
  sorry

end NUMINAMATH_CALUDE_mark_and_carolyn_money_l1159_115909


namespace NUMINAMATH_CALUDE_ellipse_theorems_l1159_115942

/-- Given an ellipse with equation x²/a² + y²/b² = 1 where a > b > 0,
    and focal length 2√3, prove the following theorems. -/
theorem ellipse_theorems 
  (a b : ℝ) 
  (h_ab : a > b ∧ b > 0) 
  (h_focal : a^2 - b^2 = 3) :
  let C : ℝ × ℝ → Prop := λ p => p.1^2 / 4 + p.2^2 = 1
  ∃ (k : ℝ) (h_k : k ≠ 0),
    let l₁ : ℝ → ℝ := λ x => k * x
    ∃ (A B : ℝ × ℝ) (h_AB : A.2 = l₁ A.1 ∧ B.2 = l₁ B.1),
      let l₂ : ℝ → ℝ := λ x => (B.2 + k/4 * (x - B.1))
      ∃ (D : ℝ × ℝ) (h_D : D.2 = l₂ D.1),
        (A.1 - D.1) * (A.1 - B.1) + (A.2 - D.2) * (A.2 - B.2) = 0 →
        (∀ p : ℝ × ℝ, C p ↔ p.1^2 / 4 + p.2^2 = 1) ∧
        (∃ (M N : ℝ × ℝ), 
          M.2 = 0 ∧ N.1 = 0 ∧ M.2 = l₂ M.1 ∧ N.2 = l₂ N.1 ∧
          ∀ (M' N' : ℝ × ℝ), M'.2 = 0 ∧ N'.1 = 0 ∧ M'.2 = l₂ M'.1 ∧ N'.2 = l₂ N'.1 →
          abs (M.1 * N.2) / 2 ≥ abs (M'.1 * N'.2) / 2 ∧
          abs (M.1 * N.2) / 2 = 9/8) := by
  sorry


end NUMINAMATH_CALUDE_ellipse_theorems_l1159_115942


namespace NUMINAMATH_CALUDE_fraction_proof_l1159_115955

theorem fraction_proof (n : ℝ) (f : ℝ) (h1 : n / 2 = 945.0000000000013) 
  (h2 : (4/15 * 5/7 * n) - (4/9 * f * n) = 24) : f = 0.4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_proof_l1159_115955


namespace NUMINAMATH_CALUDE_kevin_stuffed_animals_l1159_115948

/-- Represents the number of prizes Kevin collected. -/
def total_prizes : ℕ := 50

/-- Represents the number of frisbees Kevin collected. -/
def frisbees : ℕ := 18

/-- Represents the number of yo-yos Kevin collected. -/
def yo_yos : ℕ := 18

/-- Represents the number of stuffed animals Kevin collected. -/
def stuffed_animals : ℕ := total_prizes - frisbees - yo_yos

theorem kevin_stuffed_animals : stuffed_animals = 14 := by
  sorry

end NUMINAMATH_CALUDE_kevin_stuffed_animals_l1159_115948


namespace NUMINAMATH_CALUDE_greatest_gcd_value_l1159_115903

def S (n : ℕ+) : ℕ := n^2

theorem greatest_gcd_value (n : ℕ+) :
  (∃ m : ℕ+, Nat.gcd (2 * S m + 10 * m) (m - 3) = 42) ∧
  (∀ k : ℕ+, Nat.gcd (2 * S k + 10 * k) (k - 3) ≤ 42) :=
sorry

end NUMINAMATH_CALUDE_greatest_gcd_value_l1159_115903


namespace NUMINAMATH_CALUDE_square_sum_divisors_l1159_115938

theorem square_sum_divisors (n : ℕ) : n ≥ 2 →
  (∃ a b : ℕ, a > 1 ∧ a ∣ n ∧ b ∣ n ∧
    (∀ d : ℕ, d > 1 → d ∣ n → d ≥ a) ∧
    n = a^2 + b^2) →
  n = 5 ∨ n = 8 ∨ n = 20 := by
sorry

end NUMINAMATH_CALUDE_square_sum_divisors_l1159_115938


namespace NUMINAMATH_CALUDE_harkamal_payment_l1159_115918

/-- Calculate the total amount paid for fruits given the quantities and rates -/
def totalAmountPaid (grapeQuantity mangoQuantity grapeRate mangoRate : ℕ) : ℕ :=
  grapeQuantity * grapeRate + mangoQuantity * mangoRate

/-- Theorem: Harkamal paid 1055 to the shopkeeper -/
theorem harkamal_payment : totalAmountPaid 8 9 70 55 = 1055 := by
  sorry

end NUMINAMATH_CALUDE_harkamal_payment_l1159_115918


namespace NUMINAMATH_CALUDE_jerry_shelf_difference_l1159_115904

/-- Calculates the difference between books and action figures on Jerry's shelf -/
def shelf_difference (initial_figures : ℕ) (initial_books : ℕ) (added_figures : ℕ) : ℕ :=
  initial_books - (initial_figures + added_figures)

/-- Proves that the difference between books and action figures on Jerry's shelf is 4 -/
theorem jerry_shelf_difference :
  shelf_difference 2 10 4 = 4 := by
  sorry

end NUMINAMATH_CALUDE_jerry_shelf_difference_l1159_115904


namespace NUMINAMATH_CALUDE_conditional_prob_is_two_thirds_l1159_115997

/-- The sample space for two coin flips -/
def S : Finset (Fin 2 × Fin 2) := Finset.univ

/-- Event A: at least one tail shows up -/
def A : Finset (Fin 2 × Fin 2) := {(0, 1), (1, 0), (1, 1)}

/-- Event B: exactly one head shows up -/
def B : Finset (Fin 2 × Fin 2) := {(0, 1), (1, 0)}

/-- The probability measure for the sample space -/
def P (E : Finset (Fin 2 × Fin 2)) : ℚ := (E.card : ℚ) / (S.card : ℚ)

/-- The conditional probability of B given A -/
def conditional_prob : ℚ := P (A ∩ B) / P A

theorem conditional_prob_is_two_thirds : conditional_prob = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_conditional_prob_is_two_thirds_l1159_115997


namespace NUMINAMATH_CALUDE_prob_odd_divisor_15_factorial_l1159_115902

/-- The factorial function -/
def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

/-- The number of positive integer divisors of n -/
def numDivisors (n : ℕ) : ℕ := sorry

/-- The number of odd positive integer divisors of n -/
def numOddDivisors (n : ℕ) : ℕ := sorry

/-- The probability of randomly choosing an odd divisor from the positive integer divisors of n -/
def probOddDivisor (n : ℕ) : ℚ := (numOddDivisors n : ℚ) / (numDivisors n : ℚ)

theorem prob_odd_divisor_15_factorial :
  probOddDivisor (factorial 15) = 1 / 12 := by sorry

end NUMINAMATH_CALUDE_prob_odd_divisor_15_factorial_l1159_115902


namespace NUMINAMATH_CALUDE_sum_of_cubes_and_values_positive_l1159_115981

theorem sum_of_cubes_and_values_positive (a b c : ℝ) 
  (hab : a + b > 0) (hac : a + c > 0) (hbc : b + c > 0) : 
  (a^3 + a) + (b^3 + b) + (c^3 + c) > 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_cubes_and_values_positive_l1159_115981


namespace NUMINAMATH_CALUDE_sum_of_fractions_less_than_target_l1159_115905

theorem sum_of_fractions_less_than_target : 
  (1/2 : ℚ) + (-5/6 : ℚ) + (1/5 : ℚ) + (1/4 : ℚ) + (-9/20 : ℚ) + (-9/20 : ℚ) < (45/100 : ℚ) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_fractions_less_than_target_l1159_115905


namespace NUMINAMATH_CALUDE_complex_fraction_equals_neg_i_l1159_115998

theorem complex_fraction_equals_neg_i : (1 + 2*Complex.I) / (Complex.I - 2) = -Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equals_neg_i_l1159_115998


namespace NUMINAMATH_CALUDE_certain_number_proof_l1159_115952

theorem certain_number_proof : ∃! x : ℚ, x / 4 + 3 = 5 ∧ x = 8 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l1159_115952


namespace NUMINAMATH_CALUDE_point_M_coordinates_l1159_115960

-- Define the coordinates of points M and N
def M (m : ℝ) : ℝ × ℝ := (4*m + 4, 3*m - 6)
def N : ℝ × ℝ := (-8, 12)

-- Define the condition for MN being parallel to x-axis
def parallel_to_x_axis (M N : ℝ × ℝ) : Prop := M.2 = N.2

-- Theorem statement
theorem point_M_coordinates :
  ∃ m : ℝ, parallel_to_x_axis (M m) N ∧ M m = (28, 12) := by
  sorry

end NUMINAMATH_CALUDE_point_M_coordinates_l1159_115960


namespace NUMINAMATH_CALUDE_s_upper_bound_l1159_115999

/-- Represents a triangle with side lengths p, q, r -/
structure Triangle where
  p : ℝ
  q : ℝ
  r : ℝ
  h_positive : 0 < p ∧ 0 < q ∧ 0 < r
  h_inequality : p ≤ r ∧ r ≤ q
  h_triangle : p + r > q ∧ q + r > p ∧ p + q > r
  h_ratio : p / (q + r) = r / (p + q)

/-- Represents a point inside the triangle -/
structure InnerPoint (t : Triangle) where
  x : ℝ
  y : ℝ
  z : ℝ
  h_inside : x > 0 ∧ y > 0 ∧ z > 0 ∧ x + y + z < t.p + t.q + t.r

/-- The sum of distances from inner point to sides -/
def s (t : Triangle) (p : InnerPoint t) : ℝ := p.x + p.y + p.z

/-- The theorem to be proved -/
theorem s_upper_bound (t : Triangle) (p : InnerPoint t) : s t p ≤ 3 * t.p := by sorry

end NUMINAMATH_CALUDE_s_upper_bound_l1159_115999


namespace NUMINAMATH_CALUDE_log_27_3_l1159_115921

theorem log_27_3 : Real.log 3 / Real.log 27 = 1 / 3 := by
  have h : 27 = 3^3 := by sorry
  sorry

end NUMINAMATH_CALUDE_log_27_3_l1159_115921


namespace NUMINAMATH_CALUDE_chord_equation_parabola_l1159_115973

/-- Given a parabola y² = 4x and a chord AB with midpoint P(1,1), 
    the equation of the line containing chord AB is 2x - y - 1 = 0 -/
theorem chord_equation_parabola (A B : ℝ × ℝ) :
  let parabola := fun (p : ℝ × ℝ) ↦ p.2^2 = 4 * p.1
  let midpoint := (1, 1)
  let on_parabola := fun (p : ℝ × ℝ) ↦ parabola p
  let is_midpoint := fun (m p1 p2 : ℝ × ℝ) ↦ 
    m.1 = (p1.1 + p2.1) / 2 ∧ m.2 = (p1.2 + p2.2) / 2
  on_parabola A ∧ on_parabola B ∧ is_midpoint midpoint A B →
  ∃ (a b c : ℝ), a * A.1 + b * A.2 + c = 0 ∧
                  a * B.1 + b * B.2 + c = 0 ∧
                  (a, b, c) = (2, -1, -1) :=
by sorry

end NUMINAMATH_CALUDE_chord_equation_parabola_l1159_115973


namespace NUMINAMATH_CALUDE_parabola_transformation_transformed_vertex_l1159_115971

/-- The original parabola function -/
def original_parabola (x : ℝ) : ℝ := 2 * x^2

/-- The transformed parabola function -/
def transformed_parabola (x : ℝ) : ℝ := 2 * (x - 4)^2 - 1

/-- Theorem stating that the transformed parabola is a shift of the original parabola -/
theorem parabola_transformation :
  ∀ x : ℝ, transformed_parabola x = original_parabola (x - 4) - 1 := by
  sorry

/-- Corollary showing the vertex of the transformed parabola -/
theorem transformed_vertex :
  ∃ x y : ℝ, x = 4 ∧ y = -1 ∧ ∀ t : ℝ, transformed_parabola t ≥ transformed_parabola x := by
  sorry

end NUMINAMATH_CALUDE_parabola_transformation_transformed_vertex_l1159_115971


namespace NUMINAMATH_CALUDE_greatest_difference_l1159_115928

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

theorem greatest_difference (x y : ℕ) 
  (hx1 : 1 < x) (hx2 : x < 20) 
  (hy1 : 20 < y) (hy2 : y < 50) 
  (hxp : is_prime x) 
  (hym : ∃ k : ℕ, y = 7 * k) : 
  (∀ a b : ℕ, 1 < a → a < 20 → 20 < b → b < 50 → is_prime a → (∃ m : ℕ, b = 7 * m) → b - a ≤ y - x) ∧ y - x = 30 := by
  sorry

end NUMINAMATH_CALUDE_greatest_difference_l1159_115928


namespace NUMINAMATH_CALUDE_polygon_E_largest_area_l1159_115919

/-- Represents a polygon composed of unit squares and right triangles --/
structure Polygon where
  squares : ℕ
  triangles : ℕ

/-- Calculates the area of a polygon --/
def area (p : Polygon) : ℚ :=
  p.squares + p.triangles / 2

/-- Theorem stating that polygon E has the largest area --/
theorem polygon_E_largest_area (A B C D E : Polygon)
  (hA : A = ⟨5, 0⟩)
  (hB : B = ⟨5, 0⟩)
  (hC : C = ⟨5, 0⟩)
  (hD : D = ⟨4, 1⟩)
  (hE : E = ⟨5, 1⟩) :
  area E ≥ area A ∧ area E ≥ area B ∧ area E ≥ area C ∧ area E ≥ area D := by
  sorry

#check polygon_E_largest_area

end NUMINAMATH_CALUDE_polygon_E_largest_area_l1159_115919


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l1159_115987

-- Define a geometric sequence
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * q

-- Theorem statement
theorem geometric_sequence_problem (a : ℕ → ℝ) :
  geometric_sequence a →
  (∀ n : ℕ, a n > 0) →
  a 2 = 1 →
  a 8 = a 6 + 2 * a 4 →
  a 6 = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l1159_115987


namespace NUMINAMATH_CALUDE_absolute_value_five_l1159_115969

theorem absolute_value_five (x : ℝ) : |x| = 5 → x = 5 ∨ x = -5 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_five_l1159_115969


namespace NUMINAMATH_CALUDE_zachary_pushups_count_l1159_115932

/-- The number of push-ups Zachary did -/
def zachary_pushups : ℕ := 46

/-- The number of crunches Zachary did -/
def zachary_crunches : ℕ := 58

/-- Zachary did 12 more crunches than push-ups -/
axiom crunches_pushups_difference : zachary_crunches = zachary_pushups + 12

theorem zachary_pushups_count : zachary_pushups = 46 := by
  sorry

end NUMINAMATH_CALUDE_zachary_pushups_count_l1159_115932


namespace NUMINAMATH_CALUDE_smallest_m_pair_l1159_115920

/-- Given the equation 19m + 90 + 8n = 1998, where m and n are positive integers,
    the pair (m, n) with the smallest possible value for m is (4, 229). -/
theorem smallest_m_pair : 
  ∃ (m n : ℕ), 
    (∀ (m' n' : ℕ), 19 * m' + 90 + 8 * n' = 1998 → m ≤ m') ∧ 
    19 * m + 90 + 8 * n = 1998 ∧ 
    m = 4 ∧ 
    n = 229 := by
  sorry

end NUMINAMATH_CALUDE_smallest_m_pair_l1159_115920


namespace NUMINAMATH_CALUDE_at_least_one_even_digit_in_sum_l1159_115940

def is_17_digit (n : ℕ) : Prop := 10^16 ≤ n ∧ n < 10^17

def reverse_number (n : ℕ) : ℕ := 
  let digits := List.reverse (Nat.digits 10 n)
  digits.foldl (λ acc d => acc * 10 + d) 0

theorem at_least_one_even_digit_in_sum (M : ℕ) (hM : is_17_digit M) :
  ∃ d : ℕ, d < 10 ∧ d % 2 = 0 ∧ d ∈ Nat.digits 10 (M + reverse_number M) :=
sorry

end NUMINAMATH_CALUDE_at_least_one_even_digit_in_sum_l1159_115940


namespace NUMINAMATH_CALUDE_phi_minus_phi_squared_l1159_115925

theorem phi_minus_phi_squared (Φ φ : ℝ) : 
  Φ ≠ φ → Φ^2 = Φ + 1 → φ^2 = φ + 1 → (Φ - φ)^2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_phi_minus_phi_squared_l1159_115925


namespace NUMINAMATH_CALUDE_polar_to_cartesian_l1159_115900

/-- Given a curve in polar coordinates r = p * sin(5θ), 
    this theorem states its equivalent form in Cartesian coordinates. -/
theorem polar_to_cartesian (p : ℝ) (x y : ℝ) :
  (∃ (θ : ℝ), x = (p * Real.sin (5 * θ)) * Real.cos θ ∧
               y = (p * Real.sin (5 * θ)) * Real.sin θ) ↔
  x^6 - 5*p*x^4*y + 10*p*x^2*y^3 + y^6 + 3*x^4*y^2 - p*y^5 + 3*x^2*y^4 = 0 :=
by sorry

end NUMINAMATH_CALUDE_polar_to_cartesian_l1159_115900


namespace NUMINAMATH_CALUDE_complex_square_equality_l1159_115954

theorem complex_square_equality (c d : ℕ+) :
  (c + d * Complex.I) ^ 2 = 15 + 8 * Complex.I →
  c + d * Complex.I = 4 + Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_square_equality_l1159_115954


namespace NUMINAMATH_CALUDE_product_sum_fractions_l1159_115926

theorem product_sum_fractions : (3 * 4 * 5 * 6) * (1/3 + 1/4 + 1/5 + 1/6) = 342 := by
  sorry

end NUMINAMATH_CALUDE_product_sum_fractions_l1159_115926


namespace NUMINAMATH_CALUDE_cube_sum_divisibility_l1159_115908

theorem cube_sum_divisibility (a : ℤ) (h1 : a > 1) 
  (h2 : ∃ (k : ℤ), (a - 1)^3 + a^3 + (a + 1)^3 = k^3) : 
  4 ∣ a := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_divisibility_l1159_115908


namespace NUMINAMATH_CALUDE_scientific_notation_of_0_00003_l1159_115974

theorem scientific_notation_of_0_00003 :
  ∃ (a : ℝ) (n : ℤ), 0.00003 = a * (10 : ℝ) ^ n ∧ 1 ≤ |a| ∧ |a| < 10 ∧ a = 3 ∧ n = -5 :=
by sorry

end NUMINAMATH_CALUDE_scientific_notation_of_0_00003_l1159_115974


namespace NUMINAMATH_CALUDE_average_age_proof_l1159_115914

theorem average_age_proof (a b c : ℕ) : 
  (a + b + c) / 3 = 26 → 
  b = 20 → 
  (a + c) / 2 = 29 :=
by sorry

end NUMINAMATH_CALUDE_average_age_proof_l1159_115914


namespace NUMINAMATH_CALUDE_instantaneous_speed_at_3_seconds_l1159_115994

/-- The motion equation of an object -/
def s (t : ℝ) : ℝ := 1 - t + 2 * t^2

/-- The instantaneous speed (derivative of s) -/
def v (t : ℝ) : ℝ := -1 + 4 * t

theorem instantaneous_speed_at_3_seconds :
  v 3 = 11 := by sorry

end NUMINAMATH_CALUDE_instantaneous_speed_at_3_seconds_l1159_115994


namespace NUMINAMATH_CALUDE_bacon_strips_for_fourteen_customers_l1159_115990

/-- Breakfast plate configuration at a cafe -/
structure BreakfastPlate where
  eggs : ℕ
  bacon_multiplier : ℕ

/-- Calculate total bacon strips needed for multiple breakfast plates -/
def total_bacon_strips (plate : BreakfastPlate) (num_customers : ℕ) : ℕ :=
  num_customers * (plate.eggs * plate.bacon_multiplier)

/-- Theorem: The cook needs to fry 56 bacon strips for 14 customers -/
theorem bacon_strips_for_fourteen_customers :
  ∃ (plate : BreakfastPlate),
    plate.eggs = 2 ∧
    plate.bacon_multiplier = 2 ∧
    total_bacon_strips plate 14 = 56 := by
  sorry

end NUMINAMATH_CALUDE_bacon_strips_for_fourteen_customers_l1159_115990


namespace NUMINAMATH_CALUDE_souvenir_spending_l1159_115916

def souvenir_problem (key_chains_and_bracelets : ℝ) (difference : ℝ) : Prop :=
  let t_shirts := key_chains_and_bracelets - difference
  let total := t_shirts + key_chains_and_bracelets
  total = 548

theorem souvenir_spending :
  souvenir_problem 347 146 :=
by
  sorry

end NUMINAMATH_CALUDE_souvenir_spending_l1159_115916


namespace NUMINAMATH_CALUDE_shooting_performance_and_probability_l1159_115958

def shooter_A_scores : List ℕ := [7, 8, 7, 9, 5, 4, 9, 10, 7, 4]
def shooter_B_scores : List ℕ := [9, 5, 7, 8, 7, 6, 8, 6, 7, 7]

def mean (scores : List ℕ) : ℚ :=
  (scores.sum : ℚ) / scores.length

def variance (scores : List ℕ) : ℚ :=
  let m := mean scores
  (scores.map (fun x => ((x : ℚ) - m)^2)).sum / scores.length

def is_excellent (score : ℕ) : Bool :=
  score ≥ 8

def excellent_probability (scores : List ℕ) : ℚ :=
  (scores.filter is_excellent).length / scores.length

theorem shooting_performance_and_probability :
  (variance shooter_B_scores < variance shooter_A_scores) ∧
  (excellent_probability shooter_A_scores + excellent_probability shooter_B_scores = 19/25) := by
  sorry

end NUMINAMATH_CALUDE_shooting_performance_and_probability_l1159_115958


namespace NUMINAMATH_CALUDE_sequence_properties_l1159_115950

/-- Given a sequence {a_n} where n ∈ ℕ* and S_n = n^2 + n, prove:
    1) a_n = 2n for all n ∈ ℕ*
    2) The sum of the first n terms of {1/(n+1)a_n} equals n/(2n+2) -/
theorem sequence_properties (a : ℕ+ → ℚ) (S : ℕ+ → ℚ) 
    (h : ∀ n : ℕ+, S n = (n : ℚ)^2 + n) :
  (∀ n : ℕ+, a n = 2 * n) ∧ 
  (∀ n : ℕ+, (Finset.range n.val).sum (λ i => 1 / ((i + 2 : ℚ) * a (⟨i + 1, Nat.succ_pos i⟩))) = n / (2 * n + 2)) :=
by sorry

end NUMINAMATH_CALUDE_sequence_properties_l1159_115950


namespace NUMINAMATH_CALUDE_pink_ratio_theorem_l1159_115980

/-- Given a class with the following properties:
  * There are 30 students in total
  * There are 18 girls in the class
  * Half of the class likes green
  * 9 students like yellow
  * The remaining students like pink (all of whom are girls)
  Then the ratio of girls who like pink to the total number of girls is 1/3 -/
theorem pink_ratio_theorem (total_students : ℕ) (total_girls : ℕ) (yellow_fans : ℕ) :
  total_students = 30 →
  total_girls = 18 →
  yellow_fans = 9 →
  (total_students / 2 + yellow_fans + (total_girls - (total_students - total_students / 2 - yellow_fans)) = total_students) →
  (total_girls - (total_students - total_students / 2 - yellow_fans)) / total_girls = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_pink_ratio_theorem_l1159_115980
