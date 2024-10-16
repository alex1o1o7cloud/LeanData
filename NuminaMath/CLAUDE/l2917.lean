import Mathlib

namespace NUMINAMATH_CALUDE_heart_shaped_chocolate_weight_l2917_291771

/-- Represents the weight of a chocolate bar -/
def chocolate_bar_weight (whole_squares : ℕ) (triangles : ℕ) (square_weight : ℕ) : ℕ :=
  whole_squares * square_weight + triangles * (square_weight / 2)

/-- Theorem stating the weight of the heart-shaped chocolate bar -/
theorem heart_shaped_chocolate_weight :
  chocolate_bar_weight 32 16 6 = 240 := by
  sorry

end NUMINAMATH_CALUDE_heart_shaped_chocolate_weight_l2917_291771


namespace NUMINAMATH_CALUDE_tan_seven_pi_sixths_l2917_291747

theorem tan_seven_pi_sixths : Real.tan (7 * Real.pi / 6) = 1 / Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_seven_pi_sixths_l2917_291747


namespace NUMINAMATH_CALUDE_percentage_increase_proof_l2917_291736

def initial_earnings : ℝ := 40
def new_earnings : ℝ := 60

theorem percentage_increase_proof :
  (new_earnings - initial_earnings) / initial_earnings * 100 = 50 := by
  sorry

end NUMINAMATH_CALUDE_percentage_increase_proof_l2917_291736


namespace NUMINAMATH_CALUDE_zeros_in_square_of_near_power_of_ten_l2917_291765

theorem zeros_in_square_of_near_power_of_ten : 
  ∃ n : ℕ, (10^12 - 3)^2 = n * 10^11 ∧ n % 10 ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_zeros_in_square_of_near_power_of_ten_l2917_291765


namespace NUMINAMATH_CALUDE_min_distance_point_to_line_l2917_291775

/-- Circle O₁ with center (a, b) and radius √(b² + 1) -/
def circle_O₁ (a b x y : ℝ) : Prop :=
  (x - a)^2 + (y - b)^2 = b^2 + 1

/-- Circle O₂ with center (c, d) and radius √(d² + 1) -/
def circle_O₂ (c d x y : ℝ) : Prop :=
  (x - c)^2 + (y - d)^2 = d^2 + 1

/-- Line l: 3x - 4y - 25 = 0 -/
def line_l (x y : ℝ) : Prop :=
  3*x - 4*y - 25 = 0

/-- The minimum distance between a point on the intersection of two circles and a line -/
theorem min_distance_point_to_line
  (a b c d : ℝ)
  (h1 : a * c = 8)
  (h2 : a / b = c / d)
  : ∃ (P : ℝ × ℝ),
    (circle_O₁ a b P.1 P.2 ∧ circle_O₂ c d P.1 P.2) →
    (∀ (M : ℝ × ℝ), line_l M.1 M.2 →
      ∃ (dist : ℝ),
        dist = Real.sqrt ((P.1 - M.1)^2 + (P.2 - M.2)^2) ∧
        dist ≥ 2 ∧
        (∃ (M₀ : ℝ × ℝ), line_l M₀.1 M₀.2 ∧
          Real.sqrt ((P.1 - M₀.1)^2 + (P.2 - M₀.2)^2) = 2)) :=
sorry

end NUMINAMATH_CALUDE_min_distance_point_to_line_l2917_291775


namespace NUMINAMATH_CALUDE_opposite_of_negative_five_is_five_l2917_291795

theorem opposite_of_negative_five_is_five : 
  -(- 5) = 5 := by sorry

end NUMINAMATH_CALUDE_opposite_of_negative_five_is_five_l2917_291795


namespace NUMINAMATH_CALUDE_salary_after_four_months_l2917_291756

def salary_calculation (initial_salary : ℝ) (initial_increase_rate : ℝ) (initial_bonus : ℝ) (bonus_increase_rate : ℝ) (months : ℕ) : ℝ :=
  let rec helper (current_salary : ℝ) (current_bonus : ℝ) (current_increase_rate : ℝ) (month : ℕ) : ℝ :=
    if month = 0 then
      current_salary + current_bonus
    else
      let new_salary := current_salary * (1 + current_increase_rate)
      let new_bonus := current_bonus * (1 + bonus_increase_rate)
      let new_increase_rate := current_increase_rate * 2
      helper new_salary new_bonus new_increase_rate (month - 1)
  helper initial_salary initial_bonus initial_increase_rate months

theorem salary_after_four_months :
  salary_calculation 2000 0.05 150 0.1 4 = 4080.45 := by
  sorry

end NUMINAMATH_CALUDE_salary_after_four_months_l2917_291756


namespace NUMINAMATH_CALUDE_residue_calculation_l2917_291722

theorem residue_calculation : 196 * 18 - 21 * 9 + 5 ≡ 14 [ZMOD 18] := by
  sorry

end NUMINAMATH_CALUDE_residue_calculation_l2917_291722


namespace NUMINAMATH_CALUDE_number_difference_l2917_291730

theorem number_difference (L S : ℕ) (h1 : L = 1620) (h2 : L = 6 * S + 15) : L - S = 1353 := by
  sorry

end NUMINAMATH_CALUDE_number_difference_l2917_291730


namespace NUMINAMATH_CALUDE_harolds_class_size_l2917_291741

/-- Represents the number of apples Harold split among classmates -/
def total_apples : ℕ := 15

/-- Represents the number of apples each classmate received -/
def apples_per_classmate : ℕ := 5

/-- Theorem stating the number of people in Harold's class who received apples -/
theorem harolds_class_size : 
  total_apples / apples_per_classmate = 3 := by sorry

end NUMINAMATH_CALUDE_harolds_class_size_l2917_291741


namespace NUMINAMATH_CALUDE_lcm_ratio_sum_l2917_291723

theorem lcm_ratio_sum (a b : ℕ) (h1 : a > 0) (h2 : b > 0) : 
  Nat.lcm a b = 54 → a * 3 = b * 2 → a + b = 45 := by
  sorry

end NUMINAMATH_CALUDE_lcm_ratio_sum_l2917_291723


namespace NUMINAMATH_CALUDE_infinite_segment_sum_l2917_291725

/-- Given a triangle ABC with sides a, b, c where b > c, and an infinite sequence
    of line segments constructed as follows:
    - BB1 is antiparallel to BC, intersecting AC at B1
    - B1C1 is parallel to BC, intersecting AB at C1
    - This process continues infinitely
    Then the sum of the lengths of these segments (BC + BB1 + B1C1 + ...) is ab / (b - c) -/
theorem infinite_segment_sum (a b c : ℝ) (h : b > c) (h_pos : a > 0 ∧ b > 0 ∧ c > 0) :
  ∃ (sequence : ℕ → ℝ),
    (sequence 0 = a) ∧
    (∀ n, sequence (n + 1) = sequence n * (c / b)) ∧
    (∑' n, sequence n) = a * b / (b - c) := by
  sorry

end NUMINAMATH_CALUDE_infinite_segment_sum_l2917_291725


namespace NUMINAMATH_CALUDE_percent_value_in_quarters_l2917_291738

theorem percent_value_in_quarters : 
  let num_dimes : ℕ := 40
  let num_quarters : ℕ := 30
  let dime_value : ℕ := 10
  let quarter_value : ℕ := 25
  let total_dimes_value : ℕ := num_dimes * dime_value
  let total_quarters_value : ℕ := num_quarters * quarter_value
  let total_value : ℕ := total_dimes_value + total_quarters_value
  (total_quarters_value : ℝ) / (total_value : ℝ) * 100 = 65.22 :=
by sorry

end NUMINAMATH_CALUDE_percent_value_in_quarters_l2917_291738


namespace NUMINAMATH_CALUDE_tg_2alpha_l2917_291742

theorem tg_2alpha (α : Real) 
  (h1 : Real.cos (α - Real.pi/2) = 0.2) 
  (h2 : Real.pi/2 < α ∧ α < Real.pi) : 
  Real.tan (2*α) = -4 * Real.sqrt 6 / 23 := by
sorry

end NUMINAMATH_CALUDE_tg_2alpha_l2917_291742


namespace NUMINAMATH_CALUDE_slightly_used_crayons_l2917_291788

theorem slightly_used_crayons (total : ℕ) (new_fraction : ℚ) (broken_percent : ℚ) : 
  total = 120 → 
  new_fraction = 1 / 3 →
  broken_percent = 1 / 5 →
  ∃ (slightly_used : ℕ), slightly_used = 56 ∧ 
    slightly_used = total - (total * new_fraction).floor - (total * broken_percent).floor :=
by sorry

end NUMINAMATH_CALUDE_slightly_used_crayons_l2917_291788


namespace NUMINAMATH_CALUDE_hyperbola_equation_l2917_291721

/-- A hyperbola with given properties has the equation x²/18 - y²/32 = 1 -/
theorem hyperbola_equation (e : ℝ) (a b : ℝ) (h1 : e = 5/3) 
  (h2 : a > 0) (h3 : b > 0) (h4 : e = Real.sqrt (a^2 + b^2) / a) 
  (h5 : ∀ x y : ℝ, (x^2 / a^2) - (y^2 / b^2) = 1 ↔ 
    (8*x + 2*Real.sqrt 7*y - 16)^2 / (64/a^2 + 28/b^2) = 256) : 
  a^2 = 18 ∧ b^2 = 32 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l2917_291721


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2917_291711

theorem complex_equation_solution (z : ℂ) (i : ℂ) (h1 : i * i = -1) (h2 : i * z = 1) : z = -i := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2917_291711


namespace NUMINAMATH_CALUDE_farrah_matchsticks_l2917_291735

/-- Calculates the total number of matchsticks given the number of boxes, matchboxes per box, and sticks per matchbox. -/
def total_matchsticks (x y z : ℕ) : ℕ := x * y * z

/-- Theorem stating that for the given values, the total number of matchsticks is 300,000. -/
theorem farrah_matchsticks :
  let x : ℕ := 10
  let y : ℕ := 50
  let z : ℕ := 600
  total_matchsticks x y z = 300000 := by
  sorry

end NUMINAMATH_CALUDE_farrah_matchsticks_l2917_291735


namespace NUMINAMATH_CALUDE_concurrent_or_parallel_iff_concyclic_l2917_291710

/-- A point in the Euclidean plane -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- A line in the Euclidean plane -/
structure Line :=
  (a : ℝ) (b : ℝ) (c : ℝ)

/-- Definition of a triangle -/
structure Triangle :=
  (A : Point) (B : Point) (C : Point)

/-- Definition of a circumcenter -/
def circumcenter (t : Triangle) : Point :=
  sorry

/-- Definition of concurrency for three lines -/
def are_concurrent (l1 l2 l3 : Line) : Prop :=
  sorry

/-- Definition of parallel lines -/
def are_parallel (l1 l2 : Line) : Prop :=
  sorry

/-- Definition of pairwise parallel lines -/
def are_pairwise_parallel (l1 l2 l3 : Line) : Prop :=
  sorry

/-- Definition of concyclic points -/
def are_concyclic (A B C D : Point) : Prop :=
  sorry

/-- The main theorem -/
theorem concurrent_or_parallel_iff_concyclic 
  (A B C D E F : Point) 
  (G : Point := circumcenter ⟨B, C, E⟩) 
  (H : Point := circumcenter ⟨A, D, F⟩) 
  (AB CD GH : Line) :
  (are_concurrent AB CD GH ∨ are_pairwise_parallel AB CD GH) ↔ 
  are_concyclic A B E F :=
sorry

end NUMINAMATH_CALUDE_concurrent_or_parallel_iff_concyclic_l2917_291710


namespace NUMINAMATH_CALUDE_smallest_prime_20_less_than_square_l2917_291724

theorem smallest_prime_20_less_than_square : ∃ (m : ℕ), 
  (∀ (n : ℕ), n > 0 ∧ Nat.Prime n ∧ (∃ (k : ℕ), n = k^2 - 20) → n ≥ 5) ∧
  5 > 0 ∧ Nat.Prime 5 ∧ 5 = m^2 - 20 :=
by sorry

end NUMINAMATH_CALUDE_smallest_prime_20_less_than_square_l2917_291724


namespace NUMINAMATH_CALUDE_wayne_shrimp_cost_l2917_291718

/-- Calculates the cost of shrimp for an appetizer given the number of shrimp per guest,
    number of guests, cost per pound, and number of shrimp per pound. -/
def shrimpAppetizer (shrimpPerGuest : ℕ) (numGuests : ℕ) (costPerPound : ℚ) (shrimpPerPound : ℕ) : ℚ :=
  (shrimpPerGuest * numGuests : ℚ) / shrimpPerPound * costPerPound

/-- Proves that Wayne will spend $170.00 on the shrimp appetizer given the specified conditions. -/
theorem wayne_shrimp_cost :
  shrimpAppetizer 5 40 17 20 = 170 := by
  sorry

end NUMINAMATH_CALUDE_wayne_shrimp_cost_l2917_291718


namespace NUMINAMATH_CALUDE_sqrt_plus_inverse_geq_two_l2917_291702

theorem sqrt_plus_inverse_geq_two (x : ℝ) (hx : x > 0) : Real.sqrt x + 1 / Real.sqrt x ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_plus_inverse_geq_two_l2917_291702


namespace NUMINAMATH_CALUDE_complex_magnitude_of_i_times_one_minus_i_l2917_291726

theorem complex_magnitude_of_i_times_one_minus_i : 
  let z : ℂ := Complex.I * (1 - Complex.I)
  Complex.abs z = Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_complex_magnitude_of_i_times_one_minus_i_l2917_291726


namespace NUMINAMATH_CALUDE_eliminate_x_y_l2917_291793

-- Define the tangent and cotangent functions
noncomputable def tg (x : ℝ) : ℝ := Real.tan x
noncomputable def ctg (x : ℝ) : ℝ := 1 / Real.tan x

-- State the theorem
theorem eliminate_x_y (x y a b c : ℝ) 
  (h1 : tg x + tg y = a)
  (h2 : ctg x + ctg y = b)
  (h3 : x + y = c) :
  ctg c = 1 / a - 1 / b :=
by sorry

end NUMINAMATH_CALUDE_eliminate_x_y_l2917_291793


namespace NUMINAMATH_CALUDE_circle_equation_is_correct_l2917_291768

/-- A circle with center on the x-axis, radius 2, and passing through (1, 2) -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ
  passes_through : ℝ × ℝ
  center_on_x_axis : center.2 = 0
  radius_is_2 : radius = 2
  passes_through_point : passes_through = (1, 2)

/-- The equation of the circle -/
def circle_equation (c : Circle) (x y : ℝ) : Prop :=
  (x - c.center.1)^2 + y^2 = c.radius^2

theorem circle_equation_is_correct (c : Circle) :
  circle_equation c = λ x y ↦ (x - 1)^2 + y^2 = 4 := by
sorry

end NUMINAMATH_CALUDE_circle_equation_is_correct_l2917_291768


namespace NUMINAMATH_CALUDE_equidistant_after_1_min_equidistant_after_5_min_speed_ratio_l2917_291744

-- Define the speeds of A and B
def v_A : ℝ := sorry
def v_B : ℝ := sorry

-- Define the initial position of B
def initial_B_position : ℝ := -800

-- Define the equidistant condition after 1 minute
theorem equidistant_after_1_min : v_A = |initial_B_position + v_B| := sorry

-- Define the equidistant condition after 5 minutes
theorem equidistant_after_5_min : 5 * v_A = |initial_B_position + 5 * v_B| := sorry

-- Theorem to prove
theorem speed_ratio : v_A / v_B = 1 / 9 := sorry

end NUMINAMATH_CALUDE_equidistant_after_1_min_equidistant_after_5_min_speed_ratio_l2917_291744


namespace NUMINAMATH_CALUDE_complement_of_35_degrees_l2917_291764

theorem complement_of_35_degrees :
  ∀ α : Real,
  α = 35 →
  90 - α = 55 :=
by
  sorry

end NUMINAMATH_CALUDE_complement_of_35_degrees_l2917_291764


namespace NUMINAMATH_CALUDE_track_length_is_480_l2917_291780

/-- Represents the circular track and the runners' properties -/
structure Track :=
  (length : ℝ)
  (brenda_speed : ℝ)
  (sally_speed : ℝ)

/-- The conditions of the problem -/
def problem_conditions (track : Track) : Prop :=
  ∃ (t1 t2 : ℝ),
    -- First meeting
    track.brenda_speed * t1 = 120 ∧
    track.sally_speed * t1 = track.length / 2 - 120 ∧
    -- Second meeting
    track.sally_speed * (t1 + t2) = track.length / 2 + 60 ∧
    track.brenda_speed * (t1 + t2) = track.length / 2 - 60 ∧
    -- Constant speeds
    track.brenda_speed > 0 ∧
    track.sally_speed > 0

/-- The theorem stating that the track length is 480 meters -/
theorem track_length_is_480 :
  ∃ (track : Track), problem_conditions track ∧ track.length = 480 :=
sorry

end NUMINAMATH_CALUDE_track_length_is_480_l2917_291780


namespace NUMINAMATH_CALUDE_banana_arrangements_count_l2917_291701

/-- The number of unique arrangements of the letters in BANANA -/
def banana_arrangements : ℕ :=
  Nat.factorial 6 / (Nat.factorial 3 * Nat.factorial 2)

/-- Theorem stating that the number of unique arrangements of BANANA is 60 -/
theorem banana_arrangements_count : banana_arrangements = 60 := by
  sorry

end NUMINAMATH_CALUDE_banana_arrangements_count_l2917_291701


namespace NUMINAMATH_CALUDE_existence_of_h_l2917_291740

theorem existence_of_h : ∃ h : ℝ, ∀ n : ℕ, 
  ¬(⌊h * 1969^n⌋ ∣ ⌊h * 1969^(n-1)⌋) :=
sorry

end NUMINAMATH_CALUDE_existence_of_h_l2917_291740


namespace NUMINAMATH_CALUDE_yogurt_calories_l2917_291753

def calories_per_ounce_yogurt (strawberries : ℕ) (yogurt_ounces : ℕ) (calories_per_strawberry : ℕ) (total_calories : ℕ) : ℕ :=
  (total_calories - strawberries * calories_per_strawberry) / yogurt_ounces

theorem yogurt_calories (strawberries : ℕ) (yogurt_ounces : ℕ) (calories_per_strawberry : ℕ) (total_calories : ℕ)
  (h1 : strawberries = 12)
  (h2 : yogurt_ounces = 6)
  (h3 : calories_per_strawberry = 4)
  (h4 : total_calories = 150) :
  calories_per_ounce_yogurt strawberries yogurt_ounces calories_per_strawberry total_calories = 17 := by
  sorry

end NUMINAMATH_CALUDE_yogurt_calories_l2917_291753


namespace NUMINAMATH_CALUDE_runner_speed_proof_l2917_291782

def total_distance : ℝ := 1000
def total_time : ℝ := 380
def first_segment_distance : ℝ := 720
def first_segment_speed : ℝ := 3

def second_segment_speed : ℝ := 2

theorem runner_speed_proof :
  let first_segment_time := first_segment_distance / first_segment_speed
  let second_segment_distance := total_distance - first_segment_distance
  let second_segment_time := total_time - first_segment_time
  second_segment_speed = second_segment_distance / second_segment_time :=
by
  sorry

end NUMINAMATH_CALUDE_runner_speed_proof_l2917_291782


namespace NUMINAMATH_CALUDE_pages_read_total_l2917_291787

theorem pages_read_total (jairus_pages arniel_pages total_pages : ℕ) : 
  jairus_pages = 20 →
  arniel_pages = 2 + 2 * jairus_pages →
  total_pages = jairus_pages + arniel_pages →
  total_pages = 62 :=
by
  sorry

end NUMINAMATH_CALUDE_pages_read_total_l2917_291787


namespace NUMINAMATH_CALUDE_boat_travel_difference_l2917_291728

/-- Represents the difference in distance traveled downstream vs upstream for a boat -/
def boat_distance_difference (a b : ℝ) : ℝ :=
  let downstream_speed := a + b
  let upstream_speed := a - b
  let downstream_distance := 3 * downstream_speed
  let upstream_distance := 2 * upstream_speed
  downstream_distance - upstream_distance

/-- Theorem stating the difference in distance traveled by the boat -/
theorem boat_travel_difference (a b : ℝ) (h : a > b) :
  boat_distance_difference a b = a + 5*b := by
  sorry

#check boat_travel_difference

end NUMINAMATH_CALUDE_boat_travel_difference_l2917_291728


namespace NUMINAMATH_CALUDE_not_divisible_by_nine_l2917_291739

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

theorem not_divisible_by_nine : ¬(∃ k : ℕ, 48767621 = 9 * k) :=
  by
  have h1 : ∀ n : ℕ, (∃ k : ℕ, n = 9 * k) ↔ (∃ m : ℕ, sum_of_digits n = 9 * m) := by sorry
  have h2 : sum_of_digits 48767621 = 41 := by sorry
  have h3 : ¬(∃ m : ℕ, 41 = 9 * m) := by sorry
  sorry

end NUMINAMATH_CALUDE_not_divisible_by_nine_l2917_291739


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l2917_291750

theorem sufficient_not_necessary : 
  (∀ x : ℝ, |x| < 2 → x^2 - x - 6 < 0) ∧ 
  (∃ x : ℝ, x^2 - x - 6 < 0 ∧ ¬(|x| < 2)) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l2917_291750


namespace NUMINAMATH_CALUDE_max_fraction_of_three_numbers_l2917_291732

/-- Two-digit natural number -/
def TwoDigitNat : Type := {n : ℕ // 10 ≤ n ∧ n ≤ 99}

theorem max_fraction_of_three_numbers (x y z : TwoDigitNat) 
  (h : (x.val + y.val + z.val) / 3 = 60) :
  (x.val + y.val) / z.val ≤ 17 := by
  sorry

end NUMINAMATH_CALUDE_max_fraction_of_three_numbers_l2917_291732


namespace NUMINAMATH_CALUDE_max_mogs_bill_can_buy_l2917_291749

/-- The cost of one mag -/
def mag_cost : ℕ := 3

/-- The cost of one mig -/
def mig_cost : ℕ := 4

/-- The cost of one mog -/
def mog_cost : ℕ := 8

/-- The total amount Bill will spend -/
def total_spent : ℕ := 100

/-- Theorem stating the maximum number of mogs Bill can buy -/
theorem max_mogs_bill_can_buy :
  ∃ (mags migs mogs : ℕ),
    mags ≥ 1 ∧
    migs ≥ 1 ∧
    mogs ≥ 1 ∧
    mag_cost * mags + mig_cost * migs + mog_cost * mogs = total_spent ∧
    mogs = 10 ∧
    (∀ (m : ℕ), m > 10 →
      ¬∃ (x y : ℕ), x ≥ 1 ∧ y ≥ 1 ∧
        mag_cost * x + mig_cost * y + mog_cost * m = total_spent) :=
sorry

end NUMINAMATH_CALUDE_max_mogs_bill_can_buy_l2917_291749


namespace NUMINAMATH_CALUDE_expression_is_integer_l2917_291777

theorem expression_is_integer (a b c : ℝ) 
  (h1 : a^2 + b^2 = 2*c^2) 
  (h2 : a ≠ b) 
  (h3 : c ≠ -a) 
  (h4 : c ≠ -b) : 
  ∃ n : ℤ, ((a+b+2*c)*(2*a^2-b^2-c^2)) / ((a-b)*(a+c)*(b+c)) = n :=
sorry

end NUMINAMATH_CALUDE_expression_is_integer_l2917_291777


namespace NUMINAMATH_CALUDE_max_expression_value_l2917_291712

def expression (a b c d : ℕ) : ℕ := c * a^b + d

theorem max_expression_value :
  ∃ (a b c d : ℕ),
    a ∈ ({0, 1, 3, 4} : Set ℕ) ∧
    b ∈ ({0, 1, 3, 4} : Set ℕ) ∧
    c ∈ ({0, 1, 3, 4} : Set ℕ) ∧
    d ∈ ({0, 1, 3, 4} : Set ℕ) ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    expression a b c d = 196 ∧
    ∀ (x y z w : ℕ),
      x ∈ ({0, 1, 3, 4} : Set ℕ) →
      y ∈ ({0, 1, 3, 4} : Set ℕ) →
      z ∈ ({0, 1, 3, 4} : Set ℕ) →
      w ∈ ({0, 1, 3, 4} : Set ℕ) →
      x ≠ y ∧ x ≠ z ∧ x ≠ w ∧ y ≠ z ∧ y ≠ w ∧ z ≠ w →
      expression x y z w ≤ 196 :=
by
  sorry

#check max_expression_value

end NUMINAMATH_CALUDE_max_expression_value_l2917_291712


namespace NUMINAMATH_CALUDE_root_property_l2917_291776

/-- Given that x₀ is a real root of the equation 2x²e^(2x)+lnx=0, prove that 2x₀ + ln(x₀) = 0 -/
theorem root_property (x₀ : ℝ) (h : 2 * x₀^2 * Real.exp (2 * x₀) + Real.log x₀ = 0) :
  2 * x₀ + Real.log x₀ = 0 := by
  sorry

end NUMINAMATH_CALUDE_root_property_l2917_291776


namespace NUMINAMATH_CALUDE_joey_age_digit_sum_l2917_291751

def joey_age_sum_of_digits (n : ℕ) : ℕ :=
  (n / 10) + (n % 10)

theorem joey_age_digit_sum :
  ∃ (chloe_age : ℕ) (joey_age : ℕ),
    joey_age = chloe_age + 2 ∧
    chloe_age > 2 ∧
    chloe_age % 5 = 0 ∧
    joey_age % 5 = 0 ∧
    ∀ k : ℕ, k < chloe_age → k % 5 ≠ 0 ∧
    joey_age_sum_of_digits joey_age = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_joey_age_digit_sum_l2917_291751


namespace NUMINAMATH_CALUDE_no_intersection_l2917_291785

/-- Parabola C: y^2 = 4x -/
def parabola (x y : ℝ) : Prop := y^2 = 4*x

/-- Point M(x₀, y₀) is inside the parabola if y₀² < 4x₀ -/
def inside_parabola (x₀ y₀ : ℝ) : Prop := y₀^2 < 4*x₀

/-- Line l: y₀y = 2(x + x₀) -/
def line (x₀ y₀ x y : ℝ) : Prop := y₀*y = 2*(x + x₀)

theorem no_intersection (x₀ y₀ : ℝ) (h : inside_parabola x₀ y₀) :
  ¬∃ x y, parabola x y ∧ line x₀ y₀ x y :=
sorry

end NUMINAMATH_CALUDE_no_intersection_l2917_291785


namespace NUMINAMATH_CALUDE_second_to_last_digit_of_power_of_three_is_even_l2917_291706

theorem second_to_last_digit_of_power_of_three_is_even (n : ℕ) :
  ∃ (k : ℕ), 3^n ≡ 20 * k + 2 * (3^n / 10 % 10) [ZMOD 100] :=
sorry

end NUMINAMATH_CALUDE_second_to_last_digit_of_power_of_three_is_even_l2917_291706


namespace NUMINAMATH_CALUDE_battle_gathering_count_l2917_291766

theorem battle_gathering_count :
  -- Define the number of cannoneers
  ∀ (cannoneers : ℕ),
  -- Define the number of women as double the number of cannoneers
  ∀ (women : ℕ),
  women = 2 * cannoneers →
  -- Define the number of men as twice the number of women
  ∀ (men : ℕ),
  men = 2 * women →
  -- Given condition: there are 63 cannoneers
  cannoneers = 63 →
  -- Prove that the total number of people is 378
  men + women = 378 := by
sorry

end NUMINAMATH_CALUDE_battle_gathering_count_l2917_291766


namespace NUMINAMATH_CALUDE_reduce_to_single_digit_l2917_291705

/-- Represents the operation of splitting digits and summing -/
def digit_split_sum (n : ℕ) : ℕ → ℕ → ℕ := sorry

/-- Predicate for a number being single-digit -/
def is_single_digit (n : ℕ) : Prop := n < 10

/-- Theorem stating that any natural number can be reduced to a single digit in at most 15 steps -/
theorem reduce_to_single_digit (N : ℕ) :
  ∃ (sequence : Fin 16 → ℕ),
    sequence 0 = N ∧
    (∀ i : Fin 15, ∃ a b : ℕ, sequence (i.succ) = digit_split_sum (sequence i) a b) ∧
    is_single_digit (sequence 15) :=
  sorry

end NUMINAMATH_CALUDE_reduce_to_single_digit_l2917_291705


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2917_291794

def A : Set ℤ := {-1, 1}
def B : Set ℤ := {-1, 3}

theorem intersection_of_A_and_B : A ∩ B = {-1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2917_291794


namespace NUMINAMATH_CALUDE_longer_train_length_l2917_291790

/-- Calculates the length of the longer train given the speeds of two trains,
    the time they take to cross each other, and the length of the shorter train. -/
theorem longer_train_length
  (speed1 : ℝ)
  (speed2 : ℝ)
  (crossing_time : ℝ)
  (shorter_train_length : ℝ)
  (h1 : speed1 = 60)
  (h2 : speed2 = 40)
  (h3 : crossing_time = 11.159107271418288)
  (h4 : shorter_train_length = 140)
  : ∃ (longer_train_length : ℝ),
    longer_train_length = 170 ∧
    (speed1 + speed2) * (1000 / 3600) * crossing_time =
      shorter_train_length + longer_train_length :=
by
  sorry

end NUMINAMATH_CALUDE_longer_train_length_l2917_291790


namespace NUMINAMATH_CALUDE_special_multiplication_l2917_291796

theorem special_multiplication (a b : ℤ) :
  (∀ x y, x * y = 5*x + 2*y - 1) → (-4) * 6 = -9 := by
  sorry

end NUMINAMATH_CALUDE_special_multiplication_l2917_291796


namespace NUMINAMATH_CALUDE_intersection_implies_B_equals_one_three_l2917_291716

def A : Set ℝ := {1, 2, 4}

def B (m : ℝ) : Set ℝ := {x | x^2 - 4*x + m = 0}

theorem intersection_implies_B_equals_one_three :
  ∃ m : ℝ, (A ∩ B m = {1}) → (B m = {1, 3}) :=
by sorry

end NUMINAMATH_CALUDE_intersection_implies_B_equals_one_three_l2917_291716


namespace NUMINAMATH_CALUDE_inequality_proof_l2917_291767

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  a / Real.sqrt b + b / Real.sqrt a ≥ Real.sqrt a + Real.sqrt b := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2917_291767


namespace NUMINAMATH_CALUDE_square_instead_of_multiply_by_8_error_percentage_l2917_291791

theorem square_instead_of_multiply_by_8_error_percentage
  (x : ℝ) (h : x > 0) :
  let correct_result := 8 * x
  let mistaken_result := x ^ 2
  let error := |mistaken_result - correct_result|
  let error_percentage := (error / correct_result) * 100
  error_percentage = |(x - 8) / 8| * 100 :=
by sorry

end NUMINAMATH_CALUDE_square_instead_of_multiply_by_8_error_percentage_l2917_291791


namespace NUMINAMATH_CALUDE_square_root_of_negative_two_squared_l2917_291762

theorem square_root_of_negative_two_squared (x : ℝ) : x = 2 → x ^ 2 = (-2) ^ 2 := by sorry

end NUMINAMATH_CALUDE_square_root_of_negative_two_squared_l2917_291762


namespace NUMINAMATH_CALUDE_min_value_theorem_l2917_291755

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (1 / a + 1 / b + 2 * Real.sqrt (a * b)) ≥ 4 ∧
  (1 / a + 1 / b + 2 * Real.sqrt (a * b) = 4 ↔ a = b) :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2917_291755


namespace NUMINAMATH_CALUDE_inverse_false_implies_negation_false_l2917_291752

theorem inverse_false_implies_negation_false (p : Prop) :
  (p → False) → ¬p = False :=
by sorry

end NUMINAMATH_CALUDE_inverse_false_implies_negation_false_l2917_291752


namespace NUMINAMATH_CALUDE_point_in_second_quadrant_l2917_291783

def second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0

theorem point_in_second_quadrant :
  let x : ℝ := -3
  let y : ℝ := 2 * Real.sqrt 2
  second_quadrant x y :=
by
  sorry

end NUMINAMATH_CALUDE_point_in_second_quadrant_l2917_291783


namespace NUMINAMATH_CALUDE_diameter_line_equation_l2917_291734

/-- Given a circle and a point inside it, prove the equation of the line containing the diameter through the point. -/
theorem diameter_line_equation (x y : ℝ) :
  (x - 1)^2 + y^2 = 4 →  -- Circle equation
  (2 : ℝ) - 1 < 2 →      -- Point (2,1) is inside the circle
  ∃ (m b : ℝ), x - y - 1 = 0 ∧ 
    (∀ (x' y' : ℝ), (x' - 1)^2 + y'^2 = 4 → (y' - 1) = m * (x' - 2) + b) :=
by sorry

end NUMINAMATH_CALUDE_diameter_line_equation_l2917_291734


namespace NUMINAMATH_CALUDE_characterization_of_expressible_numbers_l2917_291737

theorem characterization_of_expressible_numbers (n : ℕ) :
  (∃ k : ℕ, n = k + 2 * Int.floor (Real.sqrt k) + 2) ↔
  (∀ y : ℕ, n ≠ y^2 ∧ n ≠ y^2 - 1) :=
sorry

end NUMINAMATH_CALUDE_characterization_of_expressible_numbers_l2917_291737


namespace NUMINAMATH_CALUDE_product_sequence_sum_l2917_291745

theorem product_sequence_sum (a b : ℕ) : 
  (a : ℚ) / 4 = 42 → b = a - 1 → a + b = 335 := by sorry

end NUMINAMATH_CALUDE_product_sequence_sum_l2917_291745


namespace NUMINAMATH_CALUDE_election_vote_difference_l2917_291770

theorem election_vote_difference 
  (total_votes : ℕ) 
  (winner_votes second_votes third_votes fourth_votes : ℕ) 
  (h_total : total_votes = 979)
  (h_candidates : winner_votes + second_votes + third_votes + fourth_votes = total_votes)
  (h_second : winner_votes = second_votes + 53)
  (h_third : winner_votes = third_votes + 79)
  (h_fourth : fourth_votes = 199) :
  winner_votes - fourth_votes = 105 := by
sorry

end NUMINAMATH_CALUDE_election_vote_difference_l2917_291770


namespace NUMINAMATH_CALUDE_inscribed_rectangle_sides_l2917_291757

theorem inscribed_rectangle_sides (a b c : ℝ) (x y : ℝ) : 
  a = 10 ∧ b = 17 ∧ c = 21 →  -- Triangle sides
  c > a ∧ c > b →  -- c is the longest side
  x + y = 12 →  -- Half of rectangle's perimeter
  y < 8 →  -- Rectangle's height is less than triangle's height
  (8 - y) / 8 = (c - x) / c →  -- Similarity of triangles
  x = 72 / 13 ∧ y = 84 / 13 := by
sorry

end NUMINAMATH_CALUDE_inscribed_rectangle_sides_l2917_291757


namespace NUMINAMATH_CALUDE_kind_wizard_strategy_exists_l2917_291763

-- Define a type for gnomes
def Gnome := ℕ

-- Define a friendship relation
def Friendship := Gnome × Gnome

-- Define a strategy for the kind wizard
def KindWizardStrategy := ℕ → List Friendship

-- Define the evil wizard's action
def EvilWizardAction := List Friendship → List Friendship

-- Define a circular arrangement of gnomes
def CircularArrangement := List Gnome

-- Function to check if an arrangement is valid (all neighbors are friends)
def IsValidArrangement (arrangement : CircularArrangement) (friendships : List Friendship) : Prop :=
  sorry

-- Main theorem
theorem kind_wizard_strategy_exists (n : ℕ) (h : n > 1 ∧ Odd n) :
  ∃ (strategy : KindWizardStrategy),
    ∀ (evil_action : EvilWizardAction),
      ∃ (arrangement : CircularArrangement),
        IsValidArrangement arrangement (evil_action (strategy n)) :=
sorry

end NUMINAMATH_CALUDE_kind_wizard_strategy_exists_l2917_291763


namespace NUMINAMATH_CALUDE_profit_increase_l2917_291719

theorem profit_increase (x : ℝ) : 
  (1 + x / 100) * 0.8 * 1.5 = 1.68 → x = 40 := by
  sorry

end NUMINAMATH_CALUDE_profit_increase_l2917_291719


namespace NUMINAMATH_CALUDE_division_remainder_proof_l2917_291786

theorem division_remainder_proof (dividend : ℕ) (divisor : ℕ) (quotient : ℕ) (h1 : dividend = 729) (h2 : divisor = 38) (h3 : quotient = 19) :
  dividend - divisor * quotient = 7 := by
sorry

end NUMINAMATH_CALUDE_division_remainder_proof_l2917_291786


namespace NUMINAMATH_CALUDE_pencil_buyers_difference_l2917_291760

theorem pencil_buyers_difference (pencil_cost : ℕ) 
  (h1 : pencil_cost > 0)
  (h2 : 234 % pencil_cost = 0)
  (h3 : 312 % pencil_cost = 0) :
  312 / pencil_cost - 234 / pencil_cost = 6 :=
by sorry

end NUMINAMATH_CALUDE_pencil_buyers_difference_l2917_291760


namespace NUMINAMATH_CALUDE_servant_pay_problem_l2917_291707

/-- The amount of money a servant receives for partial work -/
def servant_pay (full_year_pay : ℕ) (uniform_cost : ℕ) (months_worked : ℕ) : ℕ :=
  (full_year_pay * months_worked / 12) + uniform_cost

theorem servant_pay_problem :
  let full_year_pay : ℕ := 900
  let uniform_cost : ℕ := 100
  let months_worked : ℕ := 9
  servant_pay full_year_pay uniform_cost months_worked = 775 := by
sorry

#eval servant_pay 900 100 9

end NUMINAMATH_CALUDE_servant_pay_problem_l2917_291707


namespace NUMINAMATH_CALUDE_election_votes_l2917_291799

theorem election_votes (total_votes : ℕ) 
  (h1 : ∃ (winner loser : ℕ), winner + loser = total_votes) 
  (h2 : ∃ (winner : ℕ), winner = (70 * total_votes) / 100) 
  (h3 : ∃ (winner loser : ℕ), winner - loser = 188) : 
  total_votes = 470 := by
sorry

end NUMINAMATH_CALUDE_election_votes_l2917_291799


namespace NUMINAMATH_CALUDE_vertical_shift_graph_l2917_291792

-- Define a function f from real numbers to real numbers
variable (f : ℝ → ℝ)

-- Define a constant k for the vertical shift
variable (k : ℝ)

-- Define a point (x, y) on the original graph
variable (x y : ℝ)

-- Theorem statement
theorem vertical_shift_graph :
  y = f x → (y + k) = (f x + k) :=
by
  sorry

end NUMINAMATH_CALUDE_vertical_shift_graph_l2917_291792


namespace NUMINAMATH_CALUDE_parallelogram_height_l2917_291748

-- Define the parallelogram
def parallelogram_area : ℝ := 72
def parallelogram_base : ℝ := 12

-- Theorem to prove
theorem parallelogram_height :
  parallelogram_area / parallelogram_base = 6 :=
by
  sorry


end NUMINAMATH_CALUDE_parallelogram_height_l2917_291748


namespace NUMINAMATH_CALUDE_planes_parallel_if_perpendicular_to_same_line_l2917_291778

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Plane → Plane → Prop)

-- Define the theorem
theorem planes_parallel_if_perpendicular_to_same_line 
  (m : Line) (α β : Plane) :
  perpendicular m α → perpendicular m β → parallel α β :=
sorry

end NUMINAMATH_CALUDE_planes_parallel_if_perpendicular_to_same_line_l2917_291778


namespace NUMINAMATH_CALUDE_chord_length_concentric_circles_l2917_291789

/-- Given two concentric circles with radii R and r, where the area of the ring between them is 18π,
    the length of a chord of the larger circle that is tangent to the smaller circle is 6√2. -/
theorem chord_length_concentric_circles (R r : ℝ) (h : R > r) :
  (π * R^2 - π * r^2 = 18 * π) →
  ∃ c : ℝ, c = 6 * Real.sqrt 2 ∧ c^2 = 4 * (R^2 - r^2) := by
  sorry

end NUMINAMATH_CALUDE_chord_length_concentric_circles_l2917_291789


namespace NUMINAMATH_CALUDE_sector_area_l2917_291715

theorem sector_area (α : ℝ) (p : ℝ) (h1 : α = 2) (h2 : p = 8) :
  let r := p / (α + 2)
  (1/2) * α * r^2 = 4 := by sorry

end NUMINAMATH_CALUDE_sector_area_l2917_291715


namespace NUMINAMATH_CALUDE_log_8_40_l2917_291713

theorem log_8_40 (a c : ℝ) (h1 : Real.log 2 / Real.log 10 = a) (h2 : Real.log 5 / Real.log 10 = c) :
  Real.log 40 / Real.log 8 = 1 + c / (3 * a) := by
  sorry

end NUMINAMATH_CALUDE_log_8_40_l2917_291713


namespace NUMINAMATH_CALUDE_tangent_line_y_intercept_l2917_291708

theorem tangent_line_y_intercept (a : ℝ) :
  let f : ℝ → ℝ := λ x ↦ a * x - Real.log x
  let f' : ℝ → ℝ := λ x ↦ a - 1 / x
  let tangent_slope : ℝ := f' 1
  let tangent_intercept : ℝ := f 1 - tangent_slope * 1
  tangent_intercept = 1 := by sorry

end NUMINAMATH_CALUDE_tangent_line_y_intercept_l2917_291708


namespace NUMINAMATH_CALUDE_molecular_weight_independent_of_moles_l2917_291797

/-- The molecular weight of an acid in g/mol -/
def molecular_weight : ℝ := 408

/-- The number of moles of the acid -/
def moles : ℝ := 6

/-- Theorem stating that the molecular weight is independent of the number of moles -/
theorem molecular_weight_independent_of_moles :
  molecular_weight = molecular_weight := by sorry

end NUMINAMATH_CALUDE_molecular_weight_independent_of_moles_l2917_291797


namespace NUMINAMATH_CALUDE_arctans_sum_to_pi_l2917_291781

theorem arctans_sum_to_pi : 
  Real.arctan (1/3) + Real.arctan (3/8) + Real.arctan (8/3) = π := by
  sorry

end NUMINAMATH_CALUDE_arctans_sum_to_pi_l2917_291781


namespace NUMINAMATH_CALUDE_no_descending_nat_function_exists_descending_int_function_l2917_291729

-- Define φ as a function from ℕ to ℕ
variable (φ : ℕ → ℕ)

-- Theorem 1: No such function exists when the range is ℕ
theorem no_descending_nat_function :
  ¬ ∃ f : ℕ → ℕ, ∀ x : ℕ, f x > f (φ x) :=
sorry

-- Theorem 2: Such a function exists when the range is ℤ
theorem exists_descending_int_function :
  ∃ f : ℕ → ℤ, ∀ x : ℕ, f x > f (φ x) :=
sorry

end NUMINAMATH_CALUDE_no_descending_nat_function_exists_descending_int_function_l2917_291729


namespace NUMINAMATH_CALUDE_percentage_relation_l2917_291733

theorem percentage_relation (A B C : ℝ) (h1 : A = 1.71 * C) (h2 : A = 0.05 * B) : B = 14.2 * C := by
  sorry

end NUMINAMATH_CALUDE_percentage_relation_l2917_291733


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l2917_291703

/-- Given a quadratic inequality x^2 - mx + t < 0 with solution set {x | 2 < x < 3}, prove that m - t = -1 -/
theorem quadratic_inequality_solution (m t : ℝ) : 
  (∀ x : ℝ, x^2 - m*x + t < 0 ↔ 2 < x ∧ x < 3) → 
  m - t = -1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l2917_291703


namespace NUMINAMATH_CALUDE_triangle_area_theorem_l2917_291717

theorem triangle_area_theorem (a b : ℝ) (h1 : a > 0) (h2 : b > 0) : 
  (∃ (x y : ℝ), x ≥ 0 ∧ y ≥ 0 ∧ 4*a*x + 4*b*y = 48) →
  ((1/2) * (12/a) * (12/b) = 48) →
  a * b = 3/2 := by sorry

end NUMINAMATH_CALUDE_triangle_area_theorem_l2917_291717


namespace NUMINAMATH_CALUDE_right_triangle_area_l2917_291758

theorem right_triangle_area (a b : ℝ) (h1 : a = 40) (h2 : b = 42) :
  (1 / 2 : ℝ) * a * b = 840 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_area_l2917_291758


namespace NUMINAMATH_CALUDE_product_divisible_by_twelve_l2917_291704

theorem product_divisible_by_twelve (a b c d : ℤ) :
  ∃ k : ℤ, (b - a) * (c - a) * (d - a) * (d - c) * (d - b) * (c - b) = 12 * k := by
  sorry

end NUMINAMATH_CALUDE_product_divisible_by_twelve_l2917_291704


namespace NUMINAMATH_CALUDE_ice_cream_cost_theorem_l2917_291727

def calculate_ice_cream_cost (chapati_count : ℕ) (chapati_price : ℚ)
                             (rice_count : ℕ) (rice_price : ℚ)
                             (veg_count : ℕ) (veg_price : ℚ)
                             (soup_count : ℕ) (soup_price : ℚ)
                             (dessert_count : ℕ) (dessert_price : ℚ)
                             (drink_count : ℕ) (drink_price : ℚ)
                             (discount_rate : ℚ) (tax_rate : ℚ)
                             (ice_cream_count : ℕ) (total_paid : ℚ) : ℚ :=
  let chapati_total := chapati_count * chapati_price
  let rice_total := rice_count * rice_price
  let veg_total := veg_count * veg_price
  let soup_total := soup_count * soup_price
  let dessert_total := dessert_count * dessert_price
  let drink_total := drink_count * drink_price * (1 - discount_rate)
  let subtotal := chapati_total + rice_total + veg_total + soup_total + dessert_total + drink_total
  let total_with_tax := subtotal * (1 + tax_rate)
  let ice_cream_total := total_paid - total_with_tax
  ice_cream_total / ice_cream_count

theorem ice_cream_cost_theorem :
  let chapati_count : ℕ := 16
  let chapati_price : ℚ := 6
  let rice_count : ℕ := 5
  let rice_price : ℚ := 45
  let veg_count : ℕ := 7
  let veg_price : ℚ := 70
  let soup_count : ℕ := 4
  let soup_price : ℚ := 30
  let dessert_count : ℕ := 3
  let dessert_price : ℚ := 85
  let drink_count : ℕ := 2
  let drink_price : ℚ := 50
  let discount_rate : ℚ := 0.1
  let tax_rate : ℚ := 0.18
  let ice_cream_count : ℕ := 6
  let total_paid : ℚ := 2159
  abs (calculate_ice_cream_cost chapati_count chapati_price rice_count rice_price
                                veg_count veg_price soup_count soup_price
                                dessert_count dessert_price drink_count drink_price
                                discount_rate tax_rate ice_cream_count total_paid - 108.89) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ice_cream_cost_theorem_l2917_291727


namespace NUMINAMATH_CALUDE_parabola_chord_length_l2917_291784

/-- The length of a chord passing through the focus of a parabola -/
theorem parabola_chord_length (x₁ x₂ y₁ y₂ : ℝ) : 
  y₁^2 = 4*x₁ →  -- Point A satisfies the parabola equation
  y₂^2 = 4*x₂ →  -- Point B satisfies the parabola equation
  x₁ + x₂ = 6 →  -- Given condition
  -- The line passes through the focus (1, 0) of y^2 = 4x
  ∃ (m : ℝ), y₁ = m*(x₁ - 1) ∧ y₂ = m*(x₂ - 1) →
  -- Then the length of chord AB is 8
  ((x₁ - x₂)^2 + (y₁ - y₂)^2)^(1/2 : ℝ) = 8 :=
by sorry

end NUMINAMATH_CALUDE_parabola_chord_length_l2917_291784


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l2917_291772

theorem quadratic_equation_solution :
  {x : ℝ | x^2 = -2*x} = {0, -2} := by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l2917_291772


namespace NUMINAMATH_CALUDE_f_sum_two_three_l2917_291709

/-- An odd function satisfying the given conditions -/
def f (x : ℝ) : ℝ := sorry

/-- f is an odd function -/
axiom f_odd (x : ℝ) : f (-x) = -f x

/-- f satisfies the symmetry condition -/
axiom f_sym (x : ℝ) : f (3/2 + x) = -f (3/2 - x)

/-- f(1) = 2 -/
axiom f_one : f 1 = 2

/-- Theorem: f(2) + f(3) = -2 -/
theorem f_sum_two_three : f 2 + f 3 = -2 := by sorry

end NUMINAMATH_CALUDE_f_sum_two_three_l2917_291709


namespace NUMINAMATH_CALUDE_train_crossing_time_l2917_291761

/-- Given a train and platform with specific properties, calculate the time for the train to cross a tree -/
theorem train_crossing_time (train_length platform_length : ℝ) (platform_crossing_time : ℝ) 
  (h1 : train_length = 1400)
  (h2 : platform_length = 700)
  (h3 : platform_crossing_time = 150) : 
  (train_length / ((train_length + platform_length) / platform_crossing_time)) = 100 := by
  sorry

end NUMINAMATH_CALUDE_train_crossing_time_l2917_291761


namespace NUMINAMATH_CALUDE_sum_of_cubes_l2917_291774

theorem sum_of_cubes (a b : ℝ) (h1 : a + b = 2) (h2 : a * b = -3) : 
  a^3 + b^3 = 26 := by
sorry

end NUMINAMATH_CALUDE_sum_of_cubes_l2917_291774


namespace NUMINAMATH_CALUDE_vector_scalar_addition_l2917_291773

theorem vector_scalar_addition (v₁ v₂ : Fin 3 → ℝ) (c : ℝ) :
  v₁ = ![2, -3, 4] →
  v₂ = ![-4, 7, -1] →
  c = 3 →
  c • v₁ + v₂ = ![2, -2, 11] := by sorry

end NUMINAMATH_CALUDE_vector_scalar_addition_l2917_291773


namespace NUMINAMATH_CALUDE_apple_lovers_count_l2917_291769

/-- The number of people who like apple -/
def apple_lovers : ℕ := 40

/-- The number of people who like orange and mango but dislike apple -/
def orange_mango_lovers : ℕ := 7

/-- The number of people who like mango and apple and dislike orange -/
def mango_apple_lovers : ℕ := 10

/-- The number of people who like all three fruits -/
def all_fruit_lovers : ℕ := 4

/-- Theorem stating that the number of people who like apple is 40 -/
theorem apple_lovers_count : apple_lovers = 40 := by
  sorry

end NUMINAMATH_CALUDE_apple_lovers_count_l2917_291769


namespace NUMINAMATH_CALUDE_P_on_x_axis_P_parallel_y_axis_P_second_quadrant_distance_l2917_291779

-- Define point P
def P (a : ℝ) := (a - 1, 6 + 2*a)

-- Question 1
theorem P_on_x_axis (a : ℝ) : 
  P a = (-4, 0) ↔ (P a).2 = 0 := by sorry

-- Question 2
def Q : ℝ × ℝ := (5, 8)

theorem P_parallel_y_axis (a : ℝ) : 
  P a = (5, 18) ↔ (P a).1 = Q.1 := by sorry

-- Question 3
theorem P_second_quadrant_distance (a : ℝ) : 
  (P a).1 < 0 ∧ (P a).2 > 0 ∧ |(P a).2| = 2 * |(P a).1| → 
  a^2023 + 2024 = 2023 := by sorry

end NUMINAMATH_CALUDE_P_on_x_axis_P_parallel_y_axis_P_second_quadrant_distance_l2917_291779


namespace NUMINAMATH_CALUDE_convergence_of_difference_series_l2917_291759

open Topology
open Real

-- Define a monotonic sequence
def IsMonotonic (a : ℕ → ℝ) : Prop :=
  ∀ n m : ℕ, n ≤ m → a n ≤ a m ∨ ∀ n m : ℕ, n ≤ m → a m ≤ a n

-- Define the theorem
theorem convergence_of_difference_series (a : ℕ → ℝ) 
  (h_monotonic : IsMonotonic a) 
  (h_converge : Summable a) :
  Summable (fun n => n • (a n - a (n + 1))) :=
sorry

end NUMINAMATH_CALUDE_convergence_of_difference_series_l2917_291759


namespace NUMINAMATH_CALUDE_root_product_l2917_291714

theorem root_product (b c : ℝ) : 
  (∀ x : ℝ, x^2 - 2*x - 1 = 0 → x^7 - b*x - c = 0) → 
  b * c = 11830 := by
sorry

end NUMINAMATH_CALUDE_root_product_l2917_291714


namespace NUMINAMATH_CALUDE_range_of_m_l2917_291798

/-- The function f(x) = x^2 - 2ax + 1 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x + 1

/-- Set A: range of a for which f(x) has no real roots -/
def A : Set ℝ := {a | ∀ x, f a x ≠ 0}

/-- Set B: range of a for which f(x) is not monotonic on (m, m+3) -/
def B (m : ℝ) : Set ℝ := {a | ∃ x y, m < x ∧ x < y ∧ y < m + 3 ∧ (f a x - f a y) * (x - y) < 0}

/-- Theorem: If x ∈ A is a sufficient but not necessary condition for x ∈ B, 
    then -2 ≤ m ≤ -1 -/
theorem range_of_m (m : ℝ) : 
  (∀ x, x ∈ A → x ∈ B m) ∧ (∃ x, x ∈ B m ∧ x ∉ A) → -2 ≤ m ∧ m ≤ -1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l2917_291798


namespace NUMINAMATH_CALUDE_solution_exists_l2917_291746

theorem solution_exists : ∃ x : ℝ, 0.05 < x ∧ x < 0.051 :=
by
  use 0.0505
  sorry

#check solution_exists

end NUMINAMATH_CALUDE_solution_exists_l2917_291746


namespace NUMINAMATH_CALUDE_total_revenue_calculation_l2917_291731

def ticket_revenue (student_price adult_price child_price senior_price : ℚ)
                   (group_discount : ℚ)
                   (separate_students separate_adults separate_children separate_seniors : ℕ)
                   (group_students group_adults group_children group_seniors : ℕ) : ℚ :=
  let separate_revenue := student_price * separate_students +
                          adult_price * separate_adults +
                          child_price * separate_children +
                          senior_price * separate_seniors
  let group_subtotal := student_price * group_students +
                        adult_price * group_adults +
                        child_price * group_children +
                        senior_price * group_seniors
  let group_size := group_students + group_adults + group_children + group_seniors
  let group_revenue := if group_size > 10 then group_subtotal * (1 - group_discount) else group_subtotal
  separate_revenue + group_revenue

theorem total_revenue_calculation :
  ticket_revenue 6 8 4 7 (1/10)
                 20 12 15 10
                 5 8 10 9 = 523.3 := by
  sorry

end NUMINAMATH_CALUDE_total_revenue_calculation_l2917_291731


namespace NUMINAMATH_CALUDE_power_multiplication_l2917_291754

theorem power_multiplication (x : ℝ) : x^2 * x^3 = x^5 := by
  sorry

end NUMINAMATH_CALUDE_power_multiplication_l2917_291754


namespace NUMINAMATH_CALUDE_f_g_derivatives_neg_l2917_291720

-- Define the functions f and g
variable (f g : ℝ → ℝ)

-- Define the conditions
axiom f_odd : ∀ x : ℝ, f (-x) = -f x
axiom g_even : ∀ x : ℝ, g (-x) = g x
axiom f_deriv_pos : ∀ x : ℝ, x > 0 → deriv f x > 0
axiom g_deriv_neg_pos : ∀ x : ℝ, x > 0 → deriv g (-x) > 0

-- State the theorem
theorem f_g_derivatives_neg (x : ℝ) (h : x < 0) :
  deriv f x > 0 ∧ deriv g (-x) < 0 := by sorry

end NUMINAMATH_CALUDE_f_g_derivatives_neg_l2917_291720


namespace NUMINAMATH_CALUDE_abs_negative_two_l2917_291700

theorem abs_negative_two : |(-2 : ℤ)| = 2 := by sorry

end NUMINAMATH_CALUDE_abs_negative_two_l2917_291700


namespace NUMINAMATH_CALUDE_initial_boys_count_l2917_291743

/-- Given a school with an initial number of girls and boys, and after some additions,
    prove that the initial number of boys was 214. -/
theorem initial_boys_count (initial_girls : ℕ) (initial_boys : ℕ) 
  (added_girls : ℕ) (added_boys : ℕ) (final_boys : ℕ) : 
  initial_girls = 135 → 
  added_girls = 496 → 
  added_boys = 910 → 
  final_boys = 1124 → 
  initial_boys + added_boys = final_boys → 
  initial_boys = 214 := by
sorry

end NUMINAMATH_CALUDE_initial_boys_count_l2917_291743
