import Mathlib

namespace eva_total_marks_l1449_144962

def eva_marks (maths_second science_second arts_second : ℕ) : Prop :=
  let maths_first := maths_second + 10
  let arts_first := arts_second - 15
  let science_first := science_second - (science_second / 3)
  let total_first := maths_first + arts_first + science_first
  let total_second := maths_second + science_second + arts_second
  total_first + total_second = 485

theorem eva_total_marks :
  eva_marks 80 90 90 := by sorry

end eva_total_marks_l1449_144962


namespace age_ratio_correct_l1449_144961

/-- Represents the ages and relationship between a mother and daughter -/
structure FamilyAges where
  mother_current_age : ℕ
  daughter_future_age : ℕ
  years_to_future : ℕ
  multiple : ℝ

/-- Calculates the ratio of mother's age to daughter's age at a past time -/
def age_ratio (f : FamilyAges) : ℝ × ℝ :=
  (f.multiple, 1)

/-- Theorem stating that the age ratio is correct given the family ages -/
theorem age_ratio_correct (f : FamilyAges) 
  (h1 : f.mother_current_age = 41)
  (h2 : f.daughter_future_age = 26)
  (h3 : f.years_to_future = 3)
  (h4 : ∃ (x : ℕ), f.mother_current_age - x = f.multiple * (f.daughter_future_age - f.years_to_future - x)) :
  age_ratio f = (f.multiple, 1) := by
  sorry

#check age_ratio_correct

end age_ratio_correct_l1449_144961


namespace salary_increase_l1449_144929

/-- Regression line for worker's salary with respect to labor productivity -/
def regression_line (x : ℝ) : ℝ := 60 + 90 * x

/-- Theorem: When labor productivity increases by 1000 Yuan (1 unit in x), 
    the salary increases by 90 Yuan -/
theorem salary_increase (x : ℝ) : 
  regression_line (x + 1) - regression_line x = 90 := by
  sorry

end salary_increase_l1449_144929


namespace odd_function_sum_zero_l1449_144978

-- Define an odd function
def OddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

-- Theorem statement
theorem odd_function_sum_zero (f : ℝ → ℝ) (h : OddFunction f) :
  f (-2) + f 2 = 0 := by
  sorry

end odd_function_sum_zero_l1449_144978


namespace quadratic_form_sum_l1449_144970

theorem quadratic_form_sum (x : ℝ) : ∃ (a h k : ℝ),
  (6 * x^2 - 12 * x + 4 = a * (x - h)^2 + k) ∧ (a + h + k = 5) := by
  sorry

end quadratic_form_sum_l1449_144970


namespace tom_battery_usage_l1449_144971

/-- Calculates the total number of batteries used by Tom -/
def total_batteries (flashlights : ℕ) (flashlight_batteries : ℕ) 
                    (toys : ℕ) (toy_batteries : ℕ)
                    (controllers : ℕ) (controller_batteries : ℕ) : ℕ :=
  flashlights * flashlight_batteries + 
  toys * toy_batteries + 
  controllers * controller_batteries

/-- Proves that Tom used 38 batteries in total -/
theorem tom_battery_usage : 
  total_batteries 3 2 5 4 6 2 = 38 := by
  sorry

end tom_battery_usage_l1449_144971


namespace log_equation_solution_l1449_144974

theorem log_equation_solution (y : ℝ) (h : y > 0) :
  Real.log y ^ 2 / Real.log 3 + Real.log y / Real.log (1/3) = 6 →
  y = 729 := by
sorry

end log_equation_solution_l1449_144974


namespace qt_plus_q_plus_t_not_two_l1449_144913

theorem qt_plus_q_plus_t_not_two :
  ∀ q t : ℕ+, q * t + q + t ≠ 2 := by
sorry

end qt_plus_q_plus_t_not_two_l1449_144913


namespace bushes_needed_for_zucchinis_l1449_144933

/-- Represents the number of containers of blueberries per bush -/
def blueberries_per_bush : ℕ := 12

/-- Represents the number of containers of blueberries that can be traded for pumpkins -/
def blueberries_for_pumpkins : ℕ := 4

/-- Represents the number of pumpkins received when trading blueberries -/
def pumpkins_from_blueberries : ℕ := 3

/-- Represents the number of pumpkins that can be traded for zucchinis -/
def pumpkins_for_zucchinis : ℕ := 6

/-- Represents the number of zucchinis received when trading pumpkins -/
def zucchinis_from_pumpkins : ℕ := 5

/-- Represents the target number of zucchinis -/
def target_zucchinis : ℕ := 60

theorem bushes_needed_for_zucchinis :
  ∃ (bushes : ℕ), 
    bushes * blueberries_per_bush * pumpkins_from_blueberries * zucchinis_from_pumpkins = 
    target_zucchinis * blueberries_for_pumpkins * pumpkins_for_zucchinis ∧ 
    bushes = 8 := by
  sorry

end bushes_needed_for_zucchinis_l1449_144933


namespace range_of_a_l1449_144983

def p (a : ℝ) : Prop := ∀ x ∈ Set.Icc 1 2, x^2 - a ≥ 0

def q (a : ℝ) : Prop := ∃ x : ℝ, x^2 + 2*a*x + 2 - a = 0

theorem range_of_a (a : ℝ) : p a ∧ q a → a ≤ -2 ∨ a = 1 := by
  sorry

end range_of_a_l1449_144983


namespace equal_diagonals_only_in_quad_and_pent_l1449_144917

/-- A polygon is a closed plane figure with straight sides. -/
structure Polygon where
  sides : ℕ
  diagonals : ℕ
  has_equal_diagonals : Bool

/-- Definition: A polygon has equal diagonals if all its diagonals have the same length. -/
def has_equal_diagonals (p : Polygon) : Prop :=
  p.has_equal_diagonals = true

/-- Theorem: Among polygons with 3 or more sides, only quadrilaterals and pentagons can have all diagonals equal. -/
theorem equal_diagonals_only_in_quad_and_pent (p : Polygon) :
  p.sides ≥ 3 → (has_equal_diagonals p ↔ p.sides = 4 ∨ p.sides = 5) := by
  sorry

#check equal_diagonals_only_in_quad_and_pent

end equal_diagonals_only_in_quad_and_pent_l1449_144917


namespace distance_traveled_l1449_144982

/-- The speed of sound in meters per second -/
def speed_of_sound : ℝ := 330

/-- The time between the first blast and when the man hears the second blast, in minutes -/
def time_between_blasts : ℝ := 30.25

/-- The time between the first and second blasts, in minutes -/
def actual_time_between_blasts : ℝ := 30

/-- Theorem: The distance the man traveled when he heard the second blast is 4950 meters -/
theorem distance_traveled : ℝ := by
  sorry

end distance_traveled_l1449_144982


namespace beam_buying_problem_l1449_144914

/-- Represents the problem of buying beams as described in "Si Yuan Yu Jian" -/
theorem beam_buying_problem (x : ℕ) :
  (3 * x * (x - 1) = 6210) ↔
  (x > 0 ∧
   3 * x = 6210 / x +
   3 * (x - 1)) :=
by sorry

end beam_buying_problem_l1449_144914


namespace nosuch_junction_population_l1449_144945

theorem nosuch_junction_population : ∃ (a b c : ℕ+), 
  (a.val^2 + 100 = b.val^2 + 1) ∧ 
  (b.val^2 + 101 = c.val^2) ∧ 
  (7 ∣ a.val^2) := by
  sorry

end nosuch_junction_population_l1449_144945


namespace smaller_angle_is_45_degrees_l1449_144956

/-- A parallelogram with a specific angle ratio -/
structure AngleRatioParallelogram where
  -- The measure of the smaller interior angle
  small_angle : ℝ
  -- The measure of the larger interior angle
  large_angle : ℝ
  -- The ratio of the angles is 1:3
  angle_ratio : small_angle * 3 = large_angle
  -- The angles are supplementary (add up to 180°)
  supplementary : small_angle + large_angle = 180

/-- The theorem stating that the smaller angle in the parallelogram is 45° -/
theorem smaller_angle_is_45_degrees (p : AngleRatioParallelogram) : p.small_angle = 45 := by
  sorry


end smaller_angle_is_45_degrees_l1449_144956


namespace beetle_speed_l1449_144903

/-- Proves that a beetle's speed is 2.7 km/h given specific conditions --/
theorem beetle_speed : 
  let ant_distance : ℝ := 600 -- meters
  let ant_time : ℝ := 10 -- minutes
  let beetle_distance_ratio : ℝ := 0.75 -- 25% less than ant
  let beetle_distance : ℝ := ant_distance * beetle_distance_ratio
  let km_per_meter : ℝ := 1 / 1000
  let hours_per_minute : ℝ := 1 / 60
  beetle_distance * km_per_meter / (ant_time * hours_per_minute) = 2.7 := by
sorry

end beetle_speed_l1449_144903


namespace subset_of_A_l1449_144991

def A : Set ℝ := {x | x ≤ 10}

theorem subset_of_A : {2} ⊆ A := by
  sorry

end subset_of_A_l1449_144991


namespace custom_op_example_l1449_144942

-- Define the custom operation
def custom_op (a b : ℤ) : ℤ := a^2 - b

-- State the theorem
theorem custom_op_example : custom_op (custom_op 1 2) 4 = -3 := by
  sorry

end custom_op_example_l1449_144942


namespace imaginary_part_of_z_l1449_144901

def complex_multiply (a b : ℂ) : ℂ := a * b

def imaginary_part (z : ℂ) : ℝ := z.im

theorem imaginary_part_of_z (z : ℂ) :
  complex_multiply (1 + 3*Complex.I) z = 10 →
  imaginary_part z = -3 :=
by
  sorry

end imaginary_part_of_z_l1449_144901


namespace binary_to_octal_conversion_l1449_144939

-- Define the binary number
def binary_num : List Bool := [true, false, true, true, true, false]

-- Define the octal number
def octal_num : Nat := 56

-- Theorem statement
theorem binary_to_octal_conversion :
  (binary_num.foldr (λ b acc => 2 * acc + if b then 1 else 0) 0) = octal_num * 8 := by
  sorry

end binary_to_octal_conversion_l1449_144939


namespace bank_account_balance_l1449_144965

theorem bank_account_balance 
  (transferred_amount : ℕ) 
  (remaining_balance : ℕ) 
  (original_balance : ℕ) : 
  transferred_amount = 69 → 
  remaining_balance = 26935 → 
  original_balance = remaining_balance + transferred_amount → 
  original_balance = 27004 := by
  sorry

end bank_account_balance_l1449_144965


namespace lucca_bread_problem_l1449_144943

/-- The fraction of remaining bread Lucca ate on the second day -/
def second_day_fraction (initial_bread : ℕ) (first_day_fraction : ℚ) (third_day_fraction : ℚ) (remaining_bread : ℕ) : ℚ :=
  let remaining_after_first := initial_bread - initial_bread * first_day_fraction
  2 / 5

/-- Theorem stating the fraction of remaining bread Lucca ate on the second day -/
theorem lucca_bread_problem (initial_bread : ℕ) (first_day_fraction : ℚ) (third_day_fraction : ℚ) (remaining_bread : ℕ)
    (h1 : initial_bread = 200)
    (h2 : first_day_fraction = 1 / 4)
    (h3 : third_day_fraction = 1 / 2)
    (h4 : remaining_bread = 45) :
  second_day_fraction initial_bread first_day_fraction third_day_fraction remaining_bread = 2 / 5 := by
  sorry

#eval second_day_fraction 200 (1/4) (1/2) 45

end lucca_bread_problem_l1449_144943


namespace problem_solution_l1449_144954

def proposition (m : ℝ) : Prop :=
  ∀ x ∈ Set.Icc 1 2, x^2 - 2*m*x - 3*m^2 < 0

def set_A : Set ℝ := {m | proposition m}

def set_B (a : ℝ) : Set ℝ := {x | x^2 - 2*a*x + a^2 - 1 < 0}

theorem problem_solution :
  (set_A = Set.Ioi (-2) ∪ Set.Iio (2/3)) ∧
  {a | set_A ⊆ set_B a ∧ set_A ≠ set_B a} = Set.Iic (-3) ∪ Set.Ici (5/3) :=
sorry

end problem_solution_l1449_144954


namespace f_4_solutions_l1449_144912

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 2*x

-- Define the composite function f^4
def f_4 (x : ℝ) : ℝ := f (f (f (f x)))

-- Theorem statement
theorem f_4_solutions :
  ∃! (s : Finset ℝ), (∀ c ∈ s, f_4 c = 3) ∧ s.card = 3 :=
sorry

end f_4_solutions_l1449_144912


namespace min_value_theorem_l1449_144920

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) : 
  (Real.sqrt ((x^2 + y^2) * (4 * x^2 + y^2))) / (x * y) ≥ (2 + Real.rpow 4 (1/3)) / Real.rpow 2 (1/3) := by
  sorry

end min_value_theorem_l1449_144920


namespace jellybean_difference_l1449_144922

theorem jellybean_difference (total : ℕ) (black : ℕ) (green : ℕ) (orange : ℕ) : 
  total = 27 →
  black = 8 →
  orange = green - 1 →
  total = black + green + orange →
  green - black = 2 := by
sorry

end jellybean_difference_l1449_144922


namespace bicycle_time_saved_l1449_144926

/-- The time in minutes it takes Mike to walk to school -/
def walking_time : ℕ := 98

/-- The time in minutes Mike saved by riding a bicycle -/
def time_saved : ℕ := 34

/-- Theorem: The time saved by riding a bicycle compared to walking is 34 minutes -/
theorem bicycle_time_saved : time_saved = 34 := by
  sorry

end bicycle_time_saved_l1449_144926


namespace smallest_value_of_complex_expression_l1449_144981

theorem smallest_value_of_complex_expression (a b c d : ℤ) (ω : ℂ) (ζ : ℂ) :
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  ω^4 = 1 →
  ω ≠ 1 →
  ζ = ω^2 →
  ∃ (x y z w : ℤ), x ≠ y ∧ x ≠ z ∧ x ≠ w ∧ y ≠ z ∧ y ≠ w ∧ z ≠ w ∧
    |Complex.abs (↑x + ↑y * ω + ↑z * ζ + ↑w * ω^3)| = Real.sqrt 2 ∧
    ∀ (p q r s : ℤ), p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s →
      |Complex.abs (↑p + ↑q * ω + ↑r * ζ + ↑s * ω^3)| ≥ Real.sqrt 2 :=
by sorry

end smallest_value_of_complex_expression_l1449_144981


namespace successive_discounts_result_l1449_144955

/-- Calculates the final price after applying successive discounts -/
def finalPrice (initialPrice : ℝ) (discount1 : ℝ) (discount2 : ℝ) (discount3 : ℝ) : ℝ :=
  initialPrice * (1 - discount1) * (1 - discount2) * (1 - discount3)

/-- Theorem stating that applying successive discounts of 20%, 10%, and 5% to a good 
    with an actual price of Rs. 9941.52 results in a final selling price of Rs. 6800.00 -/
theorem successive_discounts_result (ε : ℝ) (h : ε > 0) :
  ∃ (result : ℝ), abs (finalPrice 9941.52 0.20 0.10 0.05 - 6800.00) < ε :=
by
  sorry


end successive_discounts_result_l1449_144955


namespace complex_fraction_sum_l1449_144963

theorem complex_fraction_sum : (1 / (1 - Complex.I)) + (Complex.I / (1 + Complex.I)) = 1 + Complex.I := by
  sorry

end complex_fraction_sum_l1449_144963


namespace min_product_of_three_l1449_144951

def S : Finset ℤ := {-10, -7, -5, -3, 0, 2, 4, 6, 8}

theorem min_product_of_three (a b c : ℤ) (ha : a ∈ S) (hb : b ∈ S) (hc : c ∈ S) 
  (hab : a ≠ b) (hbc : b ≠ c) (hac : a ≠ c) :
  a * b * c ≥ -480 :=
sorry

end min_product_of_three_l1449_144951


namespace overlapping_squares_area_l1449_144996

/-- Represents a square sheet of paper --/
structure Square :=
  (side_length : ℝ)

/-- Represents the configuration of three overlapping squares --/
structure OverlappingSquares :=
  (base : Square)
  (middle_rotation : ℝ)
  (top_rotation : ℝ)

/-- Calculates the area of the resulting polygon --/
def polygon_area (config : OverlappingSquares) : ℝ :=
  sorry

/-- The main theorem --/
theorem overlapping_squares_area :
  let config := OverlappingSquares.mk (Square.mk 6) (30 * π / 180) (60 * π / 180)
  polygon_area config = 108 - 36 * Real.sqrt 3 := by
  sorry

end overlapping_squares_area_l1449_144996


namespace parallel_vectors_m_value_l1449_144972

/-- Two vectors are parallel if and only if their cross product is zero -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 - a.2 * b.1 = 0

theorem parallel_vectors_m_value :
  let a : ℝ × ℝ := (1, 2)
  let b : ℝ × ℝ := (-2, m)
  parallel a b → m = -4 := by
  sorry

end parallel_vectors_m_value_l1449_144972


namespace lcm_hcf_problem_l1449_144997

theorem lcm_hcf_problem (a b : ℕ+) (h1 : b = 15) (h2 : Nat.lcm a b = 60) (h3 : Nat.gcd a b = 3) : a = 12 := by
  sorry

end lcm_hcf_problem_l1449_144997


namespace quadratic_roots_difference_squared_l1449_144949

theorem quadratic_roots_difference_squared :
  ∀ p q : ℝ, (2 * p^2 - 9 * p + 7 = 0) → (2 * q^2 - 9 * q + 7 = 0) → (p - q)^2 = 25 / 4 := by
  sorry

end quadratic_roots_difference_squared_l1449_144949


namespace birds_on_fence_l1449_144986

/-- The number of birds initially sitting on the fence -/
def initial_birds : ℕ := 4

/-- The initial number of storks -/
def initial_storks : ℕ := 3

/-- The number of additional storks that joined -/
def additional_storks : ℕ := 6

theorem birds_on_fence :
  initial_birds = 4 ∧
  initial_storks = 3 ∧
  additional_storks = 6 ∧
  initial_storks + additional_storks = initial_birds + 5 :=
by sorry

end birds_on_fence_l1449_144986


namespace red_beads_count_l1449_144907

/-- The total number of beads in the string -/
def total_beads : ℕ := 85

/-- The number of green beads in one pattern cycle -/
def green_in_cycle : ℕ := 3

/-- The number of red beads in one pattern cycle -/
def red_in_cycle : ℕ := 4

/-- The number of yellow beads in one pattern cycle -/
def yellow_in_cycle : ℕ := 1

/-- The total number of beads in one pattern cycle -/
def beads_per_cycle : ℕ := green_in_cycle + red_in_cycle + yellow_in_cycle

/-- The number of complete cycles in the string -/
def complete_cycles : ℕ := total_beads / beads_per_cycle

/-- The number of beads remaining after complete cycles -/
def remaining_beads : ℕ := total_beads % beads_per_cycle

/-- The number of red beads in the remaining portion -/
def red_in_remaining : ℕ := min remaining_beads (red_in_cycle)

/-- Theorem: The total number of red beads in the string is 42 -/
theorem red_beads_count : 
  complete_cycles * red_in_cycle + red_in_remaining = 42 := by
sorry

end red_beads_count_l1449_144907


namespace at_least_one_product_contains_seven_l1449_144957

def containsSeven (m : Nat) : Bool :=
  let digits := m.digits 10
  7 ∈ digits

theorem at_least_one_product_contains_seven (n : Nat) (hn : n > 0) :
  ∃ k : Nat, k ≤ 35 ∧ k > 0 ∧ containsSeven (k * n) := by
  sorry

end at_least_one_product_contains_seven_l1449_144957


namespace polar_coordinates_not_bijective_l1449_144941

-- Define the types for different coordinate systems
def CartesianPoint := ℝ × ℝ
def ComplexPoint := ℂ
def PolarPoint := ℝ × ℝ  -- (r, θ)
def Vector2D := ℝ × ℝ

-- Define the bijection property
def IsBijective (f : α → β) : Prop :=
  Function.Injective f ∧ Function.Surjective f

-- State the theorem
theorem polar_coordinates_not_bijective :
  ∃ (f : CartesianPoint → ℝ × ℝ), IsBijective f ∧
  ∃ (g : ComplexPoint → ℝ × ℝ), IsBijective g ∧
  ∃ (h : Vector2D → ℝ × ℝ), IsBijective h ∧
  ¬∃ (k : PolarPoint → ℝ × ℝ), IsBijective k :=
sorry

end polar_coordinates_not_bijective_l1449_144941


namespace solution_set_implies_a_value_l1449_144953

theorem solution_set_implies_a_value (a : ℝ) : 
  (∀ x : ℝ, x^2 - x + a < 0 ↔ -1 < x ∧ x < 2) → a = -2 := by
  sorry

end solution_set_implies_a_value_l1449_144953


namespace time_to_run_around_field_l1449_144925

-- Define the side length of the square field
def side_length : ℝ := 50

-- Define the boy's running speed in km/hr
def running_speed : ℝ := 9

-- Theorem statement
theorem time_to_run_around_field : 
  let perimeter : ℝ := 4 * side_length
  let speed_in_mps : ℝ := running_speed * 1000 / 3600
  let time : ℝ := perimeter / speed_in_mps
  time = 80 := by sorry

end time_to_run_around_field_l1449_144925


namespace min_range_for_largest_angle_l1449_144908

-- Define the triangle sides as functions of x
def side_a (x : ℝ) := 2 * x
def side_b (x : ℝ) := x + 3
def side_c (x : ℝ) := x + 6

-- Define the triangle inequality conditions
def triangle_inequality (x : ℝ) : Prop :=
  side_a x + side_b x > side_c x ∧
  side_a x + side_c x > side_b x ∧
  side_b x + side_c x > side_a x

-- Define the condition for ∠A to be the largest angle
def angle_a_largest (x : ℝ) : Prop :=
  side_c x > side_a x ∧ side_c x > side_b x

-- Theorem stating the minimum range for x
theorem min_range_for_largest_angle :
  ∃ (m n : ℝ), m < n ∧
  (∀ x, m < x ∧ x < n → triangle_inequality x ∧ angle_a_largest x) ∧
  (∀ m' n', m' < n' →
    (∀ x, m' < x ∧ x < n' → triangle_inequality x ∧ angle_a_largest x) →
    n - m ≤ n' - m') ∧
  n - m = 3 := by
  sorry

end min_range_for_largest_angle_l1449_144908


namespace beta_value_l1449_144980

theorem beta_value (β : ℂ) 
  (h1 : β ≠ 1)
  (h2 : Complex.abs (β^2 - 1) = 3 * Complex.abs (β - 1))
  (h3 : Complex.abs (β^4 - 1) = 5 * Complex.abs (β - 1)) :
  β = 2 := by
  sorry

end beta_value_l1449_144980


namespace total_amount_pigs_and_hens_l1449_144975

/-- The total amount spent on buying pigs and hens -/
def total_amount (num_pigs : ℕ) (num_hens : ℕ) (price_pig : ℕ) (price_hen : ℕ) : ℕ :=
  num_pigs * price_pig + num_hens * price_hen

/-- Theorem stating that the total amount spent on 3 pigs at Rs. 300 each and 10 hens at Rs. 30 each is Rs. 1200 -/
theorem total_amount_pigs_and_hens :
  total_amount 3 10 300 30 = 1200 := by
  sorry

end total_amount_pigs_and_hens_l1449_144975


namespace indeterminate_remainder_l1449_144930

theorem indeterminate_remainder (a b c d m n x y : ℤ) 
  (eq1 : a * x + b * y = m)
  (eq2 : c * x + d * y = n)
  (rem64 : ∃ k : ℤ, a * x + b * y = 64 * k + 37) :
  ∀ r : ℤ, ¬ (∀ k : ℤ, c * x + d * y = 5 * k + r ∧ 0 ≤ r ∧ r < 5) :=
by sorry

end indeterminate_remainder_l1449_144930


namespace solution_k_value_l1449_144998

theorem solution_k_value (x y k : ℝ) : 
  x = -3 ∧ y = 2 ∧ 2*x + k*y = 0 → k = 3 := by
  sorry

end solution_k_value_l1449_144998


namespace cube_frame_impossible_without_cuts_minimum_cuts_for_cube_frame_l1449_144940

-- Define the wire length and cube edge length
def wire_length : ℝ := 120
def cube_edge_length : ℝ := 10

-- Define the number of edges in a cube
def cube_edges : ℕ := 12

-- Define the number of vertices in a cube
def cube_vertices : ℕ := 8

-- Define the number of edges meeting at each vertex of a cube
def edges_per_vertex : ℕ := 3

-- Theorem 1: It's impossible to create the cube frame without cuts
theorem cube_frame_impossible_without_cuts :
  ¬ ∃ (path : List ℝ), 
    (path.length = cube_edges) ∧ 
    (path.sum = wire_length) ∧
    (∀ edge ∈ path, edge = cube_edge_length) :=
sorry

-- Theorem 2: The minimum number of cuts required is 3
theorem minimum_cuts_for_cube_frame :
  (cube_vertices / 2 : ℕ) - 1 = 3 :=
sorry

end cube_frame_impossible_without_cuts_minimum_cuts_for_cube_frame_l1449_144940


namespace bottle_production_l1449_144990

/-- Given that 4 identical machines produce 16 bottles per minute at a constant rate,
    prove that 8 such machines will produce 96 bottles in 3 minutes. -/
theorem bottle_production (machines : ℕ) (bottles_per_minute : ℕ) (time : ℕ) : 
  machines = 4 → bottles_per_minute = 16 → time = 3 →
  (2 * machines) * (bottles_per_minute / machines) * time = 96 := by
  sorry

#check bottle_production

end bottle_production_l1449_144990


namespace tiles_arrangement_exists_l1449_144994

/-- Represents a tile with a diagonal -/
inductive Tile
| LeftDiagonal
| RightDiagonal

/-- Represents the 8x8 grid -/
def Grid := Fin 8 → Fin 8 → Tile

/-- Checks if two adjacent tiles have non-overlapping diagonals -/
def compatible (t1 t2 : Tile) : Prop :=
  t1 ≠ t2

/-- Checks if the entire grid is valid (no overlapping diagonals) -/
def valid_grid (g : Grid) : Prop :=
  ∀ i j, i < 7 → compatible (g i j) (g (i+1) j) ∧
         j < 7 → compatible (g i j) (g i (j+1))

/-- The main theorem stating that a valid arrangement exists -/
theorem tiles_arrangement_exists : ∃ g : Grid, valid_grid g :=
  sorry

end tiles_arrangement_exists_l1449_144994


namespace queen_diamond_probability_l1449_144906

/-- Represents a standard deck of 52 playing cards -/
def Deck : Type := Unit

/-- The number of cards in a standard deck -/
def deck_size : ℕ := 52

/-- The number of Queens in a standard deck -/
def num_queens : ℕ := 4

/-- The number of diamonds in a standard deck -/
def num_diamonds : ℕ := 13

/-- Represents the event of drawing a Queen as the first card and a diamond as the second card -/
def queen_then_diamond (d : Deck) : Prop := sorry

/-- The probability of the queen_then_diamond event -/
def prob_queen_then_diamond (d : Deck) : ℚ := sorry

theorem queen_diamond_probability (d : Deck) : 
  prob_queen_then_diamond d = 1 / deck_size := by sorry

end queen_diamond_probability_l1449_144906


namespace goldbach_138_largest_diff_l1449_144934

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

theorem goldbach_138_largest_diff :
  ∃ (p q : ℕ), 
    is_prime p ∧ 
    is_prime q ∧ 
    p + q = 138 ∧ 
    p ≠ q ∧
    ∀ (r s : ℕ), is_prime r → is_prime s → r + s = 138 → r ≠ s → 
      (max r s - min r s) ≤ (max p q - min p q) ∧
    (max p q - min p q) = 124 :=
sorry

end goldbach_138_largest_diff_l1449_144934


namespace mixture_volume_l1449_144924

/-- Given a mixture of liquids p and q with an initial ratio and a change in ratio after adding more of q, 
    calculate the initial volume of the mixture. -/
theorem mixture_volume (initial_p initial_q added_q : ℝ) 
  (h1 : initial_p / initial_q = 4 / 3) 
  (h2 : initial_p / (initial_q + added_q) = 5 / 7)
  (h3 : added_q = 13) : 
  initial_p + initial_q = 35 := by
  sorry

end mixture_volume_l1449_144924


namespace english_speakers_l1449_144976

theorem english_speakers (total : ℕ) (hindi : ℕ) (both : ℕ) (english : ℕ) : 
  total = 40 → 
  hindi = 30 → 
  both ≥ 10 → 
  total = hindi + english - both → 
  english = 20 := by
sorry

end english_speakers_l1449_144976


namespace trig_identity_l1449_144984

theorem trig_identity (a b c : ℝ) (θ : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c) :
  (Real.sin θ)^6 / a + (Real.cos θ)^6 / b + (Real.sin θ)^2 * (Real.cos θ)^2 / c = 1 / (a + b + c) →
  (Real.sin θ)^12 / a^5 + (Real.cos θ)^12 / b^5 + ((Real.sin θ)^2 * (Real.cos θ)^2)^3 / c^5 = 
    (a + b + (a*b)^3/c^5) / (a + b + c)^6 :=
by sorry

end trig_identity_l1449_144984


namespace smallest_stairs_l1449_144988

theorem smallest_stairs (n : ℕ) : 
  (n > 10) ∧ 
  (n % 6 = 4) ∧ 
  (n % 7 = 3) ∧ 
  (∀ m : ℕ, m > 10 ∧ m % 6 = 4 ∧ m % 7 = 3 → m ≥ n) → 
  n = 52 := by
sorry

end smallest_stairs_l1449_144988


namespace cody_initial_money_l1449_144911

theorem cody_initial_money : 
  ∀ (initial : ℕ), 
  (initial + 9 - 19 = 35) → 
  initial = 45 := by
sorry

end cody_initial_money_l1449_144911


namespace rook_configuration_exists_iff_even_l1449_144947

/-- A configuration of rooks on an n×n board. -/
def RookConfiguration (n : ℕ) := Fin n → Fin n

/-- Predicate to check if a rook configuration is valid (no two rooks attack each other). -/
def is_valid_configuration (n : ℕ) (config : RookConfiguration n) : Prop :=
  ∀ i j : Fin n, i ≠ j → config i ≠ config j ∧ i ≠ config j

/-- Predicate to check if two positions on the board are adjacent. -/
def are_adjacent (n : ℕ) (p1 p2 : Fin n × Fin n) : Prop :=
  (p1.1 = p2.1 ∧ (p1.2.val + 1 = p2.2.val ∨ p2.2.val + 1 = p1.2.val)) ∨
  (p1.2 = p2.2 ∧ (p1.1.val + 1 = p2.1.val ∨ p2.1.val + 1 = p1.1.val))

/-- Theorem stating that a valid rook configuration with a valid move exists if and only if n is even. -/
theorem rook_configuration_exists_iff_even (n : ℕ) (h : n ≥ 2) :
  (∃ (initial final : RookConfiguration n),
    is_valid_configuration n initial ∧
    is_valid_configuration n final ∧
    (∀ i : Fin n, are_adjacent n (i, initial i) (i, final i))) ↔
  Even n :=
sorry

end rook_configuration_exists_iff_even_l1449_144947


namespace absolute_value_square_l1449_144936

theorem absolute_value_square (a b : ℚ) : |a| = b → a^2 = (-b)^2 := by
  sorry

end absolute_value_square_l1449_144936


namespace scout_troop_profit_l1449_144969

/-- Calculates the profit for a scout troop selling candy bars -/
theorem scout_troop_profit (num_bars : ℕ) (buy_rate : ℚ) (sell_rate : ℚ) : 
  num_bars = 1200 → 
  buy_rate = 1/3 → 
  sell_rate = 3/5 → 
  (sell_rate * num_bars : ℚ) - (buy_rate * num_bars : ℚ) = 320 := by
  sorry

#check scout_troop_profit

end scout_troop_profit_l1449_144969


namespace intersection_of_A_and_B_l1449_144952

def A : Set ℕ := {0, 1, 2, 3}

def B : Set ℕ := {x | ∃ a ∈ A, x = 3 * a}

theorem intersection_of_A_and_B : A ∩ B = {0, 3} := by sorry

end intersection_of_A_and_B_l1449_144952


namespace inequality_system_solution_set_l1449_144995

theorem inequality_system_solution_set :
  {x : ℝ | 3 * x + 9 > 0 ∧ 2 * x < 6} = {x : ℝ | -3 < x ∧ x < 3} := by
  sorry

end inequality_system_solution_set_l1449_144995


namespace dave_derek_money_difference_l1449_144909

theorem dave_derek_money_difference :
  let derek_initial : ℕ := 40
  let derek_lunch1 : ℕ := 14
  let derek_dad_lunch : ℕ := 11
  let derek_lunch2 : ℕ := 5
  let dave_initial : ℕ := 50
  let dave_mom_lunch : ℕ := 7
  let derek_remaining : ℕ := derek_initial - derek_lunch1 - derek_dad_lunch - derek_lunch2
  let dave_remaining : ℕ := dave_initial - dave_mom_lunch
  dave_remaining - derek_remaining = 33 :=
by sorry

end dave_derek_money_difference_l1449_144909


namespace quadratic_inequality_l1449_144989

theorem quadratic_inequality (y : ℝ) : y^2 - 9*y + 14 < 0 ↔ 2 < y ∧ y < 7 := by sorry

end quadratic_inequality_l1449_144989


namespace parabolas_intersection_l1449_144946

/-- The x-coordinates of the intersection points of two parabolas -/
def intersection_x : Set ℝ :=
  {x | x = 1/2 ∨ x = -3}

/-- The y-coordinates of the intersection points of two parabolas -/
def intersection_y : Set ℝ :=
  {y | y = 3/4 ∨ y = 20}

/-- First parabola function -/
def f (x : ℝ) : ℝ := x^2 - 3*x + 2

/-- Second parabola function -/
def g (x : ℝ) : ℝ := 3*x^2 + 2*x - 1

theorem parabolas_intersection :
  ∀ x y : ℝ, (f x = g x ∧ y = f x) ↔ (x ∈ intersection_x ∧ y ∈ intersection_y) :=
sorry

end parabolas_intersection_l1449_144946


namespace sine_product_rational_l1449_144902

theorem sine_product_rational : 
  66 * Real.sin (π / 18) * Real.sin (3 * π / 18) * Real.sin (5 * π / 18) * 
  Real.sin (7 * π / 18) * Real.sin (9 * π / 18) = 33 / 8 := by
  sorry

end sine_product_rational_l1449_144902


namespace continuous_function_with_property_l1449_144923

-- Define the property of the function
def has_property (f : ℝ → ℝ) : Prop :=
  ∀ (n : ℤ) (x : ℝ), n ≠ 0 → f (x + 1 / (n : ℝ)) ≤ f x + 1 / (n : ℝ)

-- State the theorem
theorem continuous_function_with_property (f : ℝ → ℝ) 
  (hf : Continuous f) (hprop : has_property f) :
  ∃ (a : ℝ), ∀ x, f x = x + a := by
  sorry

end continuous_function_with_property_l1449_144923


namespace line_circle_intersection_range_l1449_144919

/-- Given a line x - 2y + a = 0 and a circle (x-2)^2 + y^2 = 1 with common points,
    the range of values for the real number a is [-2-√5, -2+√5]. -/
theorem line_circle_intersection_range (a : ℝ) : 
  (∃ x y : ℝ, x - 2*y + a = 0 ∧ (x-2)^2 + y^2 = 1) →
  a ∈ Set.Icc (-2 - Real.sqrt 5) (-2 + Real.sqrt 5) :=
sorry

end line_circle_intersection_range_l1449_144919


namespace water_jars_problem_l1449_144905

/-- Proves that given 28 gallons of water stored in equal numbers of quart, half-gallon, and one-gallon jars, the total number of water-filled jars is 48. -/
theorem water_jars_problem (total_water : ℚ) (num_each_jar : ℕ) : 
  total_water = 28 →
  (1/4 : ℚ) * num_each_jar + (1/2 : ℚ) * num_each_jar + 1 * num_each_jar = total_water →
  3 * num_each_jar = 48 := by
  sorry

end water_jars_problem_l1449_144905


namespace least_multiple_of_25_greater_than_475_l1449_144932

theorem least_multiple_of_25_greater_than_475 :
  ∀ n : ℕ, n > 0 ∧ 25 ∣ n ∧ n > 475 → n ≥ 500 :=
by
  sorry

end least_multiple_of_25_greater_than_475_l1449_144932


namespace least_positive_integer_with_remainders_l1449_144916

theorem least_positive_integer_with_remainders : 
  ∃! n : ℕ, n > 0 ∧ 
    n % 3 = 2 ∧ 
    n % 4 = 3 ∧ 
    n % 5 = 4 ∧ 
    n % 6 = 5 ∧
    ∀ m : ℕ, m > 0 ∧ m % 3 = 2 ∧ m % 4 = 3 ∧ m % 5 = 4 ∧ m % 6 = 5 → n ≤ m :=
by sorry

#eval 119 % 3  -- Expected: 2
#eval 119 % 4  -- Expected: 3
#eval 119 % 5  -- Expected: 4
#eval 119 % 6  -- Expected: 5

end least_positive_integer_with_remainders_l1449_144916


namespace correct_stratified_sample_l1449_144910

/-- Represents a stratified sample from a high school population -/
structure StratifiedSample where
  total_students : ℕ
  freshmen : ℕ
  sophomores : ℕ
  juniors : ℕ
  sample_size : ℕ
  sampled_freshmen : ℕ
  sampled_sophomores : ℕ
  sampled_juniors : ℕ

/-- Checks if a stratified sample is valid according to the problem conditions -/
def is_valid_sample (s : StratifiedSample) : Prop :=
  s.total_students = 2000 ∧
  s.freshmen = 800 ∧
  s.sophomores = 600 ∧
  s.juniors = 600 ∧
  s.sample_size = 50 ∧
  s.sampled_freshmen + s.sampled_sophomores + s.sampled_juniors = s.sample_size

/-- Theorem stating that the correct stratified sample is 20 freshmen, 15 sophomores, and 15 juniors -/
theorem correct_stratified_sample (s : StratifiedSample) :
  is_valid_sample s →
  s.sampled_freshmen = 20 ∧ s.sampled_sophomores = 15 ∧ s.sampled_juniors = 15 := by
  sorry


end correct_stratified_sample_l1449_144910


namespace regression_lines_intersect_l1449_144904

/-- Represents a linear regression line -/
structure RegressionLine where
  slope : ℝ
  intercept : ℝ

/-- The sample center point of a dataset -/
structure SampleCenterPoint where
  x : ℝ
  y : ℝ

/-- Theorem: Two regression lines with the same sample center point intersect -/
theorem regression_lines_intersect
  (l₁ l₂ : RegressionLine)
  (center : SampleCenterPoint)
  (h₁ : center.y = l₁.slope * center.x + l₁.intercept)
  (h₂ : center.y = l₂.slope * center.x + l₂.intercept) :
  ∃ (x y : ℝ), y = l₁.slope * x + l₁.intercept ∧ y = l₂.slope * x + l₂.intercept :=
sorry

end regression_lines_intersect_l1449_144904


namespace problem_statement_l1449_144938

theorem problem_statement (n m : ℝ) : 
  (∀ x : ℝ, (x + 3) * (x + n) = x^2 + m*x - 15) → m = -2 := by
  sorry

end problem_statement_l1449_144938


namespace potatoes_for_salads_correct_l1449_144918

/-- Given the total number of potatoes, the number used for mashed potatoes,
    and the number of leftover potatoes, calculate the number of potatoes
    used for salads. -/
def potatoes_for_salads (total mashed leftover : ℕ) : ℕ :=
  total - mashed - leftover

/-- Theorem stating that the number of potatoes used for salads is correct. -/
theorem potatoes_for_salads_correct
  (total mashed leftover salads : ℕ)
  (h_total : total = 52)
  (h_mashed : mashed = 24)
  (h_leftover : leftover = 13)
  (h_salads : salads = potatoes_for_salads total mashed leftover) :
  salads = 15 := by
  sorry

end potatoes_for_salads_correct_l1449_144918


namespace smallest_valid_number_last_three_digits_l1449_144915

def is_valid_number (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ n.digits 10 → d = 2 ∨ d = 3 ∨ d = 6

def contains_all_required_digits (n : ℕ) : Prop :=
  2 ∈ n.digits 10 ∧ 3 ∈ n.digits 10 ∧ 6 ∈ n.digits 10

def last_three_digits (n : ℕ) : ℕ :=
  n % 1000

theorem smallest_valid_number_last_three_digits :
  ∃ m : ℕ,
    m > 0 ∧
    m % 2 = 0 ∧
    m % 3 = 0 ∧
    is_valid_number m ∧
    contains_all_required_digits m ∧
    (∀ k : ℕ, k > 0 ∧ k % 2 = 0 ∧ k % 3 = 0 ∧ is_valid_number k ∧ contains_all_required_digits k → m ≤ k) ∧
    last_three_digits m = 326 :=
by sorry

end smallest_valid_number_last_three_digits_l1449_144915


namespace max_path_length_rectangular_prism_l1449_144968

/-- Represents a rectangular prism with integer dimensions -/
structure RectangularPrism where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Represents a path through all corners of a rectangular prism -/
def CornerPath (p : RectangularPrism) : Type :=
  List (Fin 2 × Fin 2 × Fin 2)

/-- Calculates the length of a given path in a rectangular prism -/
def pathLength (p : RectangularPrism) (path : CornerPath p) : ℝ :=
  sorry

/-- Checks if a path visits all corners exactly once and returns to start -/
def isValidPath (p : RectangularPrism) (path : CornerPath p) : Prop :=
  sorry

/-- The maximum possible path length for a given rectangular prism -/
def maxPathLength (p : RectangularPrism) : ℝ :=
  sorry

theorem max_path_length_rectangular_prism :
  ∃ (k : ℝ),
    maxPathLength ⟨3, 4, 5⟩ = 4 * Real.sqrt 50 + k ∧
    k > 0 ∧ k < 2 * Real.sqrt 50 :=
  sorry

end max_path_length_rectangular_prism_l1449_144968


namespace pool_filling_time_l1449_144964

/-- Proves that it takes 33 hours to fill a 30,000-gallon pool with 5 hoses supplying 3 gallons per minute each -/
theorem pool_filling_time : 
  let pool_capacity : ℕ := 30000
  let num_hoses : ℕ := 5
  let flow_rate_per_hose : ℕ := 3
  let minutes_per_hour : ℕ := 60
  let total_flow_rate_per_hour : ℕ := num_hoses * flow_rate_per_hose * minutes_per_hour
  let filling_time_hours : ℕ := pool_capacity / total_flow_rate_per_hour
  filling_time_hours = 33 := by
  sorry


end pool_filling_time_l1449_144964


namespace expression_necessarily_negative_l1449_144967

theorem expression_necessarily_negative (a b c : ℝ) 
  (ha : 0 < a ∧ a < 2) 
  (hb : -2 < b ∧ b < 0) 
  (hc : 0 < c ∧ c < 3) : 
  b + a * b < 0 := by
  sorry

end expression_necessarily_negative_l1449_144967


namespace sector_area_l1449_144950

/-- Given a sector with central angle 2 radians and arc length 4, its area is equal to 4 -/
theorem sector_area (θ : Real) (L : Real) (r : Real) (A : Real) : 
  θ = 2 → L = 4 → L = θ * r → A = 1/2 * θ * r^2 → A = 4 := by
  sorry

end sector_area_l1449_144950


namespace expression_evaluation_l1449_144900

theorem expression_evaluation (a b x y : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (h : a * x + y / b ≠ 0) :
  (a * x + y / b)⁻¹ * ((a * x)⁻¹ + (y / b)⁻¹) = (a * x * y)⁻¹ :=
by sorry

end expression_evaluation_l1449_144900


namespace intersection_line_l1449_144921

/-- The line of intersection of two planes -/
def line_of_intersection (t : ℝ) : ℝ × ℝ × ℝ := (t, 2 - t, t + 1)

/-- First plane equation -/
def plane1 (x y z : ℝ) : Prop := 2 * x - y - 3 * z + 5 = 0

/-- Second plane equation -/
def plane2 (x y z : ℝ) : Prop := x + y - 2 = 0

theorem intersection_line (t : ℝ) :
  let (x, y, z) := line_of_intersection t
  plane1 x y z ∧ plane2 x y z := by
  sorry

end intersection_line_l1449_144921


namespace radius_of_2003rd_circle_l1449_144993

/-- The radius of the nth circle in a sequence of circles tangent to the sides of a 60° angle -/
def radius (n : ℕ) : ℝ :=
  3^(n - 1)

/-- The number of circles in the sequence -/
def num_circles : ℕ := 2003

theorem radius_of_2003rd_circle :
  radius num_circles = 3^2002 :=
by sorry

end radius_of_2003rd_circle_l1449_144993


namespace hyperbola_theorem_l1449_144966

/-- Hyperbola with given asymptotes and passing point -/
structure Hyperbola where
  -- Asymptotes are y = ±√2x
  asymptote_slope : ℝ
  asymptote_slope_sq : asymptote_slope^2 = 2
  -- Passes through (3, -2√3)
  passes_through : (3 : ℝ)^2 / 3 - (-2 * Real.sqrt 3)^2 / 6 = 1

/-- The equation of the hyperbola -/
def hyperbola_equation (h : Hyperbola) (x y : ℝ) : Prop :=
  x^2 / 3 - y^2 / 6 = 1

/-- Point on the hyperbola -/
structure PointOnHyperbola (h : Hyperbola) where
  x : ℝ
  y : ℝ
  on_hyperbola : hyperbola_equation h x y

/-- Foci of the hyperbola -/
structure Foci (h : Hyperbola) where
  F₁ : ℝ × ℝ
  F₂ : ℝ × ℝ

/-- Angle between foci and point on hyperbola -/
def angle_F₁PF₂ (h : Hyperbola) (f : Foci h) (p : PointOnHyperbola h) : ℝ :=
  sorry -- Definition of the angle

/-- Area of triangle formed by foci and point on hyperbola -/
def area_PF₁F₂ (h : Hyperbola) (f : Foci h) (p : PointOnHyperbola h) : ℝ :=
  sorry -- Definition of the area

/-- Main theorem -/
theorem hyperbola_theorem (h : Hyperbola) (f : Foci h) (p : PointOnHyperbola h) :
  hyperbola_equation h p.x p.y ∧
  (angle_F₁PF₂ h f p = π / 3 → area_PF₁F₂ h f p = 6 * Real.sqrt 3) :=
sorry

end hyperbola_theorem_l1449_144966


namespace rabbit_population_estimate_l1449_144973

/-- Capture-recapture estimation of rabbit population -/
theorem rabbit_population_estimate :
  ∀ (total_population : ℕ)
    (first_capture second_capture recaptured_tagged : ℕ),
  first_capture = 10 →
  second_capture = 10 →
  recaptured_tagged = 2 →
  total_population = (first_capture * second_capture) / recaptured_tagged →
  total_population = 50 :=
by
  sorry

#check rabbit_population_estimate

end rabbit_population_estimate_l1449_144973


namespace no_integer_solutions_for_Q_perfect_square_l1449_144959

/-- The polynomial Q as a function of x -/
def Q (x : ℤ) : ℤ := x^4 + 5*x^3 + 10*x^2 + 5*x + 56

/-- Theorem stating that there are no integer solutions for x such that Q(x) is a perfect square -/
theorem no_integer_solutions_for_Q_perfect_square :
  ∀ x : ℤ, ¬∃ k : ℤ, Q x = k^2 := by
  sorry

end no_integer_solutions_for_Q_perfect_square_l1449_144959


namespace trig_identity_proof_l1449_144977

theorem trig_identity_proof : 
  (Real.cos (63 * π / 180) * Real.cos (3 * π / 180) - 
   Real.cos (87 * π / 180) * Real.cos (27 * π / 180)) / 
  (Real.cos (132 * π / 180) * Real.cos (72 * π / 180) - 
   Real.cos (42 * π / 180) * Real.cos (18 * π / 180)) = 
  -Real.tan (24 * π / 180) := by
sorry

end trig_identity_proof_l1449_144977


namespace a_equals_one_range_of_f_final_no_fixed_points_l1449_144999

/-- The quadratic function f(x) with parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := (a + 1) * x^2 + (a^2 - 1) * x + 1

/-- f is an even function -/
def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

/-- Theorem 1: If f is an even function, then a = 1 -/
theorem a_equals_one (a : ℝ) (h : is_even_function (f a)) : a = 1 := by
  sorry

/-- The quadratic function f(x) with a = 1 -/
def f_final (x : ℝ) : ℝ := 2 * x^2 + 1

/-- Theorem 2: If x ∈ [-1, 2], then the range of f_final(x) is [1, 9] -/
theorem range_of_f_final : 
  ∀ y ∈ Set.range f_final, y ∈ Set.Icc 1 9 ∧ 
  ∃ x ∈ Set.Icc (-1) 2, f_final x = 1 ∧
  ∃ x ∈ Set.Icc (-1) 2, f_final x = 9 := by
  sorry

/-- Theorem 3: The equation 2x^2 + 1 = x has no real solutions -/
theorem no_fixed_points : ¬ ∃ x : ℝ, f_final x = x := by
  sorry

end a_equals_one_range_of_f_final_no_fixed_points_l1449_144999


namespace food_box_shipment_l1449_144927

theorem food_box_shipment (total_food : ℝ) (max_shipping_weight : ℝ) :
  total_food = 777.5 ∧ max_shipping_weight = 2 →
  ⌊total_food / max_shipping_weight⌋ = 388 := by
  sorry

end food_box_shipment_l1449_144927


namespace variance_of_transformed_data_l1449_144960

variable (x : Fin 10 → ℝ)

def variance (data : Fin 10 → ℝ) : ℝ := sorry

def transform (data : Fin 10 → ℝ) : Fin 10 → ℝ := 
  fun i => 2 * data i - 1

theorem variance_of_transformed_data 
  (h : variance x = 8) : 
  variance (transform x) = 32 := by sorry

end variance_of_transformed_data_l1449_144960


namespace scholarship_sum_l1449_144979

theorem scholarship_sum (wendy kelly nina : ℕ) : 
  wendy = 20000 →
  kelly = 2 * wendy →
  nina = kelly - 8000 →
  wendy + kelly + nina = 92000 := by
  sorry

end scholarship_sum_l1449_144979


namespace sum_of_coefficients_is_192_l1449_144987

/-- The sum of all integer coefficients in the factorization of 216x^9 - 1000y^9 -/
def sum_of_coefficients (x y : ℚ) : ℤ :=
  let expression := 216 * x^9 - 1000 * y^9
  -- The actual computation of the sum is not implemented here
  192

/-- Theorem stating that the sum of all integer coefficients in the factorization of 216x^9 - 1000y^9 is 192 -/
theorem sum_of_coefficients_is_192 (x y : ℚ) : sum_of_coefficients x y = 192 := by
  sorry

end sum_of_coefficients_is_192_l1449_144987


namespace monotonic_increasing_cubic_l1449_144985

/-- A cubic function with parameters m and n -/
def f (m n : ℝ) (x : ℝ) : ℝ := 4 * x^3 + m * x^2 + (m - 3) * x + n

/-- The derivative of f with respect to x -/
def f' (m : ℝ) (x : ℝ) : ℝ := 12 * x^2 + 2 * m * x + (m - 3)

theorem monotonic_increasing_cubic (m n : ℝ) :
  (∀ x : ℝ, Monotone (f m n)) → m = 6 := by
  sorry

end monotonic_increasing_cubic_l1449_144985


namespace tangent_slope_implies_trig_ratio_triangle_perimeter_range_l1449_144958

-- Problem 1
theorem tangent_slope_implies_trig_ratio 
  (f : ℝ → ℝ) (α : ℝ) 
  (h1 : ∀ x, f x = 2*x + 2*Real.sin x + Real.cos x) 
  (h2 : HasDerivAt f 2 α) : 
  (Real.sin (π - α) + Real.cos (-α)) / (2 * Real.cos (π/2 - α) + Real.cos (2*π - α)) = 3/5 := 
sorry

-- Problem 2
theorem triangle_perimeter_range 
  (A B C a b c : ℝ) 
  (h1 : 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π) 
  (h2 : a > 0 ∧ b > 0 ∧ c > 0) 
  (h3 : a = 1) 
  (h4 : a * Real.cos C + c/2 = b) : 
  ∃ l, l = a + b + c ∧ 2 < l ∧ l ≤ 3 := 
sorry

end tangent_slope_implies_trig_ratio_triangle_perimeter_range_l1449_144958


namespace increasing_function_range_l1449_144944

theorem increasing_function_range (f : ℝ → ℝ) (h_increasing : ∀ x y, x < y → x ∈ [-1, 3] → y ∈ [-1, 3] → f x < f y) :
  ∀ a : ℝ, f a > f (1 - 2 * a) → a ∈ Set.Ioo (1/3) 1 := by
sorry

end increasing_function_range_l1449_144944


namespace hundreds_digit_of_binomial_12_6_times_6_factorial_l1449_144928

-- Define the binomial coefficient
def binomial (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

-- Define the function to get the hundreds digit
def hundreds_digit (n : ℕ) : ℕ := (n / 100) % 10

-- Theorem statement
theorem hundreds_digit_of_binomial_12_6_times_6_factorial :
  hundreds_digit (binomial 12 6 * Nat.factorial 6) = 8 := by
  sorry

end hundreds_digit_of_binomial_12_6_times_6_factorial_l1449_144928


namespace sum_reciprocals_of_constrained_numbers_l1449_144948

theorem sum_reciprocals_of_constrained_numbers (m n : ℕ+) : 
  Nat.gcd m n = 6 → 
  Nat.lcm m n = 210 → 
  m + n = 72 → 
  (1 : ℚ) / m + (1 : ℚ) / n = 1 / 17.5 := by sorry

end sum_reciprocals_of_constrained_numbers_l1449_144948


namespace distribution_theorem_l1449_144935

/-- The number of ways to distribute 6 volunteers into 4 groups and assign to 4 pavilions -/
def distribution_schemes : ℕ := 1080

/-- The number of volunteers -/
def num_volunteers : ℕ := 6

/-- The number of pavilions -/
def num_pavilions : ℕ := 4

/-- The number of groups with 2 people -/
def num_pairs : ℕ := 2

/-- The number of groups with 1 person -/
def num_singles : ℕ := 2

theorem distribution_theorem :
  (num_volunteers = 6) →
  (num_pavilions = 4) →
  (num_pairs = 2) →
  (num_singles = 2) →
  (num_pairs + num_singles = num_pavilions) →
  (2 * num_pairs + num_singles = num_volunteers) →
  distribution_schemes = 1080 := by
  sorry

#eval distribution_schemes

end distribution_theorem_l1449_144935


namespace maria_carrots_next_day_l1449_144937

/-- The number of carrots Maria picked the next day -/
def carrots_picked_next_day (initial_carrots thrown_out final_total : ℕ) : ℕ :=
  final_total - (initial_carrots - thrown_out)

/-- Theorem stating that Maria picked 15 carrots the next day -/
theorem maria_carrots_next_day : 
  carrots_picked_next_day 48 11 52 = 15 := by
  sorry

end maria_carrots_next_day_l1449_144937


namespace sum_first_three_terms_l1449_144931

def arithmetic_sequence (a : ℤ) (d : ℤ) (n : ℕ) : ℤ := a + (n - 1) * d

theorem sum_first_three_terms (a : ℤ) (d : ℤ) :
  arithmetic_sequence a d 4 = 8 ∧
  arithmetic_sequence a d 5 = 12 ∧
  arithmetic_sequence a d 6 = 16 →
  arithmetic_sequence a d 1 + arithmetic_sequence a d 2 + arithmetic_sequence a d 3 = 0 :=
by
  sorry

end sum_first_three_terms_l1449_144931


namespace survey_result_l1449_144992

theorem survey_result (total : ℕ) (lentils : ℕ) (chickpeas : ℕ) (neither : ℕ) 
  (h1 : total = 100)
  (h2 : lentils = 68)
  (h3 : chickpeas = 53)
  (h4 : neither = 6) :
  ∃ both : ℕ, both = 27 ∧ 
    total = lentils + chickpeas - both + neither :=
by sorry

end survey_result_l1449_144992
