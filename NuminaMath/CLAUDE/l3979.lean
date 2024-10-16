import Mathlib

namespace NUMINAMATH_CALUDE_root_sum_absolute_value_l3979_397923

theorem root_sum_absolute_value (m : ℝ) (α β : ℝ) 
  (h1 : α^2 - 22*α + m = 0)
  (h2 : β^2 - 22*β + m = 0)
  (h3 : m ≤ 121) : 
  |α| + |β| = if 0 ≤ m then 22 else Real.sqrt (484 - 4*m) :=
by sorry

end NUMINAMATH_CALUDE_root_sum_absolute_value_l3979_397923


namespace NUMINAMATH_CALUDE_bill_percentage_increase_l3979_397935

/-- 
Given Maximoff's original monthly bill and new monthly bill, 
prove that the percentage increase is 30%.
-/
theorem bill_percentage_increase 
  (original_bill : ℝ) 
  (new_bill : ℝ) 
  (h1 : original_bill = 60) 
  (h2 : new_bill = 78) : 
  (new_bill - original_bill) / original_bill * 100 = 30 := by
sorry

end NUMINAMATH_CALUDE_bill_percentage_increase_l3979_397935


namespace NUMINAMATH_CALUDE_buses_needed_l3979_397978

def fifth_graders : ℕ := 109
def sixth_graders : ℕ := 115
def seventh_graders : ℕ := 118
def teachers_per_grade : ℕ := 4
def parents_per_grade : ℕ := 2
def seats_per_bus : ℕ := 72

def total_students : ℕ := fifth_graders + sixth_graders + seventh_graders
def total_chaperones : ℕ := (teachers_per_grade + parents_per_grade) * 3
def total_people : ℕ := total_students + total_chaperones

theorem buses_needed : 
  ∃ n : ℕ, n * seats_per_bus ≥ total_people ∧ 
  ∀ m : ℕ, m * seats_per_bus ≥ total_people → n ≤ m := by
  sorry

end NUMINAMATH_CALUDE_buses_needed_l3979_397978


namespace NUMINAMATH_CALUDE_total_pay_calculation_l3979_397920

theorem total_pay_calculation (pay_B : ℕ) (pay_A : ℕ) : 
  pay_B = 224 → 
  pay_A = (150 * pay_B) / 100 → 
  pay_A + pay_B = 560 :=
by
  sorry

end NUMINAMATH_CALUDE_total_pay_calculation_l3979_397920


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l3979_397984

theorem quadratic_equation_roots : ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
  x₁^2 - 2023*x₁ - 1 = 0 ∧ x₂^2 - 2023*x₂ - 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l3979_397984


namespace NUMINAMATH_CALUDE_master_bedroom_size_l3979_397995

theorem master_bedroom_size 
  (master_bath : ℝ) 
  (new_room : ℝ) 
  (h1 : master_bath = 150) 
  (h2 : new_room = 918) 
  (h3 : new_room = 2 * (master_bedroom + master_bath)) : 
  master_bedroom = 309 :=
by
  sorry

end NUMINAMATH_CALUDE_master_bedroom_size_l3979_397995


namespace NUMINAMATH_CALUDE_fraction_denominator_l3979_397963

theorem fraction_denominator (y : ℝ) (x : ℝ) (h1 : y > 0) 
  (h2 : (9 * y) / x + (3 * y) / 10 = 0.75 * y) : x = 20 := by
  sorry

end NUMINAMATH_CALUDE_fraction_denominator_l3979_397963


namespace NUMINAMATH_CALUDE_max_correct_guesses_proof_l3979_397998

/-- Represents the maximum number of guaranteed correct hat color guesses 
    for n wise men with k insane among them. -/
def max_correct_guesses (n k : ℕ) : ℕ := n - k - 1

/-- Theorem stating that the maximum number of guaranteed correct hat color guesses
    is equal to n - k - 1, where n is the total number of wise men and k is the
    number of insane wise men. -/
theorem max_correct_guesses_proof (n k : ℕ) (h1 : k < n) :
  max_correct_guesses n k = n - k - 1 := by
  sorry

end NUMINAMATH_CALUDE_max_correct_guesses_proof_l3979_397998


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l3979_397974

theorem regular_polygon_sides (D : ℕ) : D = 12 → ∃ n : ℕ, n = 6 ∧ D = n * (n - 3) / 2 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l3979_397974


namespace NUMINAMATH_CALUDE_smallest_positive_integers_difference_l3979_397936

def m : ℕ := sorry

def n : ℕ := sorry

theorem smallest_positive_integers_difference : 
  (m ≥ 100) ∧ 
  (m < 1000) ∧ 
  (m % 13 = 6) ∧ 
  (∀ k : ℕ, k ≥ 100 ∧ k < 1000 ∧ k % 13 = 6 → m ≤ k) ∧
  (n ≥ 1000) ∧ 
  (n < 10000) ∧ 
  (n % 17 = 7) ∧ 
  (∀ l : ℕ, l ≥ 1000 ∧ l < 10000 ∧ l % 17 = 7 → n ≤ l) →
  n - m = 900 := by sorry

end NUMINAMATH_CALUDE_smallest_positive_integers_difference_l3979_397936


namespace NUMINAMATH_CALUDE_magnitude_of_complex_number_l3979_397997

theorem magnitude_of_complex_number (i : ℂ) (z : ℂ) :
  i^2 = -1 →
  z = (1 + 2*i) / i →
  Complex.abs z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_complex_number_l3979_397997


namespace NUMINAMATH_CALUDE_shooting_competition_score_l3979_397986

theorem shooting_competition_score 
  (team_size : ℕ) 
  (best_score : ℕ) 
  (hypothetical_best_score : ℕ) 
  (hypothetical_average : ℕ) 
  (h1 : team_size = 8)
  (h2 : best_score = 85)
  (h3 : hypothetical_best_score = 92)
  (h4 : hypothetical_average = 84)
  (h5 : hypothetical_average * team_size = 
        (hypothetical_best_score - best_score) + total_score) :
  total_score = 665 :=
by
  sorry

#check shooting_competition_score

end NUMINAMATH_CALUDE_shooting_competition_score_l3979_397986


namespace NUMINAMATH_CALUDE_min_value_squared_sum_l3979_397988

theorem min_value_squared_sum (a b : ℝ) (h : a^2 + 2*a*b - 3*b^2 = 1) :
  ∃ (m : ℝ), m = (Real.sqrt 5 + 1) / 4 ∧ ∀ (x y : ℝ), x^2 + 2*x*y - 3*y^2 = 1 → x^2 + y^2 ≥ m :=
sorry

end NUMINAMATH_CALUDE_min_value_squared_sum_l3979_397988


namespace NUMINAMATH_CALUDE_pond_to_field_area_ratio_l3979_397921

/-- Theorem: Ratio of pond area to field area
  Given a rectangular field with length double its width and a length of 20 m,
  containing a square-shaped pond with a side length of 5 m,
  the ratio of the area of the pond to the area of the field is 1:8.
-/
theorem pond_to_field_area_ratio :
  ∀ (field_length field_width pond_side : ℝ),
  field_length = 20 →
  field_length = 2 * field_width →
  pond_side = 5 →
  (pond_side^2) / (field_length * field_width) = 1/8 := by
  sorry

end NUMINAMATH_CALUDE_pond_to_field_area_ratio_l3979_397921


namespace NUMINAMATH_CALUDE_factorization_theorem_l3979_397961

theorem factorization_theorem (x y a : ℝ) : 2*x*(a-2) - y*(2-a) = (a-2)*(2*x+y) := by
  sorry

end NUMINAMATH_CALUDE_factorization_theorem_l3979_397961


namespace NUMINAMATH_CALUDE_intersection_A_B_l3979_397914

def set_A : Set ℝ := {x : ℝ | ∃ y : ℝ, y = Real.log (x - 1)}
def set_B : Set ℝ := {0, 1, 2, 3}

theorem intersection_A_B : set_A ∩ set_B = {2, 3} := by
  sorry

end NUMINAMATH_CALUDE_intersection_A_B_l3979_397914


namespace NUMINAMATH_CALUDE_unique_number_l3979_397991

def is_six_digit (n : ℕ) : Prop := 100000 ≤ n ∧ n < 1000000

def begins_and_ends_with_2 (n : ℕ) : Prop :=
  n % 10 = 2 ∧ (n / 100000) % 10 = 2

def product_of_three_consecutive_even_integers (n : ℕ) : Prop :=
  ∃ k : ℕ, n = (2*k - 2) * (2*k) * (2*k + 2)

theorem unique_number : 
  ∃! n : ℕ, is_six_digit n ∧ 
             begins_and_ends_with_2 n ∧ 
             product_of_three_consecutive_even_integers n ∧
             n = 287232 :=
by sorry

end NUMINAMATH_CALUDE_unique_number_l3979_397991


namespace NUMINAMATH_CALUDE_distinct_dice_designs_count_l3979_397994

/-- Represents a dice design -/
structure DiceDesign where
  -- We don't need to explicitly define the structure,
  -- as we're only concerned with the count of distinct designs

/-- The number of ways to choose 2 numbers from 4 -/
def choose_two_from_four : Nat := 6

/-- The number of ways to arrange the chosen numbers on opposite faces -/
def arrangement_ways : Nat := 2

/-- The number of ways to color three pairs of opposite faces -/
def coloring_ways : Nat := 8

/-- The total number of distinct dice designs -/
def distinct_dice_designs : Nat := 
  (choose_two_from_four * arrangement_ways / 2) * coloring_ways

theorem distinct_dice_designs_count :
  distinct_dice_designs = 48 := by
  sorry

end NUMINAMATH_CALUDE_distinct_dice_designs_count_l3979_397994


namespace NUMINAMATH_CALUDE_complex_ratio_l3979_397975

theorem complex_ratio (z₁ z₂ : ℂ) (h1 : Complex.abs z₁ = 1) (h2 : Complex.abs z₂ = 5/2) 
  (h3 : Complex.abs (3 * z₁ - 2 * z₂) = 7) :
  z₁ / z₂ = -1/5 * (1 - Complex.I * Real.sqrt 3) ∨ z₁ / z₂ = -1/5 * (1 + Complex.I * Real.sqrt 3) := by
sorry

end NUMINAMATH_CALUDE_complex_ratio_l3979_397975


namespace NUMINAMATH_CALUDE_sum_of_squares_is_312_l3979_397929

/-- Represents the rates and distances for biking, jogging, and swimming activities. -/
structure ActivityRates where
  bike_rate : ℕ
  jog_rate : ℕ
  swim_rate : ℕ

/-- Calculates the total distance covered given rates and times. -/
def total_distance (rates : ActivityRates) (bike_time jog_time swim_time : ℕ) : ℕ :=
  rates.bike_rate * bike_time + rates.jog_rate * jog_time + rates.swim_rate * swim_time

/-- Theorem stating that given the conditions, the sum of squares of rates is 312. -/
theorem sum_of_squares_is_312 (rates : ActivityRates) : 
  total_distance rates 1 4 3 = 66 ∧ 
  total_distance rates 3 3 2 = 76 → 
  rates.bike_rate ^ 2 + rates.jog_rate ^ 2 + rates.swim_rate ^ 2 = 312 := by
  sorry

#check sum_of_squares_is_312

end NUMINAMATH_CALUDE_sum_of_squares_is_312_l3979_397929


namespace NUMINAMATH_CALUDE_log_inequality_l3979_397967

theorem log_inequality : (Real.log 2 / Real.log 3) < (Real.log 3 / Real.log 2) ∧ (Real.log 3 / Real.log 2) < (Real.log 5 / Real.log 2) := by
  sorry

end NUMINAMATH_CALUDE_log_inequality_l3979_397967


namespace NUMINAMATH_CALUDE_n_value_l3979_397970

theorem n_value : ∃ n : ℕ, (n : ℚ) / 24 > 1 / 6 ∧ (n : ℚ) / 24 < 1 / 4 ∧ n = 5 := by
  sorry

end NUMINAMATH_CALUDE_n_value_l3979_397970


namespace NUMINAMATH_CALUDE_james_socks_l3979_397993

/-- The total number of socks James has -/
def total_socks (red_pairs black_pairs white_socks : ℕ) : ℕ :=
  2 * red_pairs + 2 * black_pairs + white_socks

/-- Theorem stating the total number of socks James has -/
theorem james_socks : 
  ∀ (red_pairs black_pairs white_socks : ℕ),
    red_pairs = 20 →
    black_pairs = red_pairs / 2 →
    white_socks = 2 * (2 * red_pairs + 2 * black_pairs) →
    total_socks red_pairs black_pairs white_socks = 180 := by
  sorry

end NUMINAMATH_CALUDE_james_socks_l3979_397993


namespace NUMINAMATH_CALUDE_set_c_not_right_triangle_set_a_right_triangle_set_b_right_triangle_set_d_right_triangle_l3979_397953

/-- A function to check if three numbers can form a right triangle -/
def can_form_right_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2

/-- Theorem stating that (7, 8, 9) cannot form a right triangle -/
theorem set_c_not_right_triangle : ¬(can_form_right_triangle 7 8 9) := by sorry

/-- Theorem stating that (1, 1, √2) can form a right triangle -/
theorem set_a_right_triangle : can_form_right_triangle 1 1 (Real.sqrt 2) := by sorry

/-- Theorem stating that (5, 12, 13) can form a right triangle -/
theorem set_b_right_triangle : can_form_right_triangle 5 12 13 := by sorry

/-- Theorem stating that (1.5, 2, 2.5) can form a right triangle -/
theorem set_d_right_triangle : can_form_right_triangle 1.5 2 2.5 := by sorry

end NUMINAMATH_CALUDE_set_c_not_right_triangle_set_a_right_triangle_set_b_right_triangle_set_d_right_triangle_l3979_397953


namespace NUMINAMATH_CALUDE_A_minus_B_equality_A_minus_B_at_negative_two_l3979_397958

-- Define A and B as functions of x
def A (x : ℝ) : ℝ := 2 * x^2 - 3 * x + 2
def B (x : ℝ) : ℝ := x^2 - 3 * x - 2

-- Theorem 1: A - B = x² + 4 for all real x
theorem A_minus_B_equality (x : ℝ) : A x - B x = x^2 + 4 := by
  sorry

-- Theorem 2: A - B = 8 when x = -2
theorem A_minus_B_at_negative_two : A (-2) - B (-2) = 8 := by
  sorry

end NUMINAMATH_CALUDE_A_minus_B_equality_A_minus_B_at_negative_two_l3979_397958


namespace NUMINAMATH_CALUDE_vector_problem_l3979_397987

/-- Given vectors a, b, c in ℝ², prove the coordinates of c and the cosine of the angle between a and b -/
theorem vector_problem (a b c : ℝ × ℝ) : 
  a = (-Real.sqrt 2, 1) →
  (c.1 * c.1 + c.2 * c.2 = 4) →
  (∃ (k : ℝ), c = k • a) →
  (b.1 * b.1 + b.2 * b.2 = 2) →
  ((a.1 + 3 * b.1) * (a.1 - b.1) + (a.2 + 3 * b.2) * (a.2 - b.2) = 0) →
  ((c = (-2 * Real.sqrt 6 / 3, 2 * Real.sqrt 3 / 3)) ∨ 
   (c = (2 * Real.sqrt 6 / 3, -2 * Real.sqrt 3 / 3))) ∧
  ((a.1 * b.1 + a.2 * b.2) / 
   (Real.sqrt (a.1 * a.1 + a.2 * a.2) * Real.sqrt (b.1 * b.1 + b.2 * b.2)) = Real.sqrt 6 / 4) :=
by sorry

end NUMINAMATH_CALUDE_vector_problem_l3979_397987


namespace NUMINAMATH_CALUDE_stone_145_is_1_l3979_397927

/-- Represents the counting pattern for stones -/
def stone_count (n : ℕ) : ℕ :=
  if n ≤ 10 then n
  else if n ≤ 19 then 20 - n
  else stone_count ((n - 1) % 18 + 1)

/-- The theorem stating that the 145th count corresponds to the first stone -/
theorem stone_145_is_1 : stone_count 145 = 1 := by
  sorry

end NUMINAMATH_CALUDE_stone_145_is_1_l3979_397927


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l3979_397905

theorem complex_fraction_simplification :
  (Complex.I : ℂ) / (1 - Complex.I) = -1/2 + Complex.I/2 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l3979_397905


namespace NUMINAMATH_CALUDE_latoya_card_credit_l3979_397924

/-- Calculates the remaining credit on a prepaid phone card after a call -/
def remaining_credit (initial_value : ℚ) (cost_per_minute : ℚ) (call_duration : ℕ) : ℚ :=
  initial_value - (cost_per_minute * call_duration)

/-- Theorem stating the remaining credit on Latoya's prepaid phone card -/
theorem latoya_card_credit :
  let initial_value : ℚ := 30
  let cost_per_minute : ℚ := 16 / 100
  let call_duration : ℕ := 22
  remaining_credit initial_value cost_per_minute call_duration = 2648 / 100 := by
sorry

end NUMINAMATH_CALUDE_latoya_card_credit_l3979_397924


namespace NUMINAMATH_CALUDE_additive_iff_extended_l3979_397940

/-- A function f: ℝ → ℝ satisfies the additive property if f(x + y) = f(x) + f(y) for all x, y ∈ ℝ. -/
def AdditiveProp (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + y) = f x + f y

/-- A function f: ℝ → ℝ satisfies the extended property if f(xy + x + y) = f(xy) + f(x) + f(y) for all x, y ∈ ℝ. -/
def ExtendedProp (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x * y + x + y) = f (x * y) + f x + f y

/-- Theorem: For any function f: ℝ → ℝ, the additive property is equivalent to the extended property. -/
theorem additive_iff_extended (f : ℝ → ℝ) : AdditiveProp f ↔ ExtendedProp f := by
  sorry

end NUMINAMATH_CALUDE_additive_iff_extended_l3979_397940


namespace NUMINAMATH_CALUDE_mountain_lake_depth_l3979_397930

/-- Represents a cone-shaped mountain partially submerged in water -/
structure Mountain where
  height : ℝ
  above_water_ratio : ℝ

/-- Calculates the depth of the lake at the base of a partially submerged mountain -/
def lake_depth (m : Mountain) : ℝ :=
  m.height * (1 - (1 - m.above_water_ratio)^(1/3))

/-- Theorem stating that for a mountain of height 12000 feet with 1/6 of its volume above water,
    the depth of the lake at its base is 780 feet -/
theorem mountain_lake_depth :
  let m : Mountain := { height := 12000, above_water_ratio := 1/6 }
  lake_depth m = 780 := by sorry

end NUMINAMATH_CALUDE_mountain_lake_depth_l3979_397930


namespace NUMINAMATH_CALUDE_cloud_same_color_tangents_iff_l3979_397926

/-- A configuration of n points on a line with circumferences painted in k colors -/
structure Cloud (n k : ℕ) where
  n_ge_two : n ≥ 2
  colors : Fin k → Type
  circumferences : Fin n → Fin n → Option (Fin k)
  different_points : ∀ i j : Fin n, i ≠ j → circumferences i j ≠ none

/-- Two circumferences are mutually exterior tangent -/
def mutually_exterior_tangent (c : Cloud n k) (i j m l : Fin n) : Prop :=
  (i ≠ j ∧ m ≠ l) ∧ (i = m ∨ i = l ∨ j = m ∨ j = l)

/-- A cloud has two mutually exterior tangent circumferences of the same color -/
def has_same_color_tangents (c : Cloud n k) : Prop :=
  ∃ i j m l : Fin n, ∃ color : Fin k,
    mutually_exterior_tangent c i j m l ∧
    c.circumferences i j = some color ∧
    c.circumferences m l = some color

/-- Main theorem: characterization of n for which all (n,k)-clouds have same color tangents -/
theorem cloud_same_color_tangents_iff (k : ℕ) :
  (∀ n : ℕ, ∀ c : Cloud n k, has_same_color_tangents c) ↔ n ≥ 2^k + 1 :=
sorry

end NUMINAMATH_CALUDE_cloud_same_color_tangents_iff_l3979_397926


namespace NUMINAMATH_CALUDE_first_night_rate_is_30_l3979_397915

/-- Represents the pricing structure of a guest house -/
structure GuestHousePricing where
  firstNightRate : ℕ  -- Flat rate for the first night
  additionalNightRate : ℕ  -- Fixed rate for each additional night

/-- Calculates the total cost for a stay -/
def totalCost (p : GuestHousePricing) (nights : ℕ) : ℕ :=
  p.firstNightRate + (nights - 1) * p.additionalNightRate

/-- Theorem stating that the flat rate for the first night is 30 -/
theorem first_night_rate_is_30 :
  ∃ (p : GuestHousePricing),
    totalCost p 4 = 210 ∧
    totalCost p 8 = 450 ∧
    p.firstNightRate = 30 := by
  sorry

end NUMINAMATH_CALUDE_first_night_rate_is_30_l3979_397915


namespace NUMINAMATH_CALUDE_johns_spending_l3979_397933

theorem johns_spending (initial_amount : ℚ) (snack_fraction : ℚ) (final_amount : ℚ)
  (h1 : initial_amount = 20)
  (h2 : snack_fraction = 1/5)
  (h3 : final_amount = 4) :
  let remaining_after_snacks := initial_amount - snack_fraction * initial_amount
  (remaining_after_snacks - final_amount) / remaining_after_snacks = 3/4 := by
sorry

end NUMINAMATH_CALUDE_johns_spending_l3979_397933


namespace NUMINAMATH_CALUDE_circle_center_coordinate_sum_l3979_397964

theorem circle_center_coordinate_sum :
  ∀ (x y h k : ℝ),
  (∀ (x' y' : ℝ), x'^2 + y'^2 = 4*x' - 6*y' + 9 ↔ (x' - h)^2 + (y' - k)^2 = (h^2 + k^2 - 9 + 4*h - 6*k)) →
  h + k = -1 :=
by sorry

end NUMINAMATH_CALUDE_circle_center_coordinate_sum_l3979_397964


namespace NUMINAMATH_CALUDE_sqrt_x_plus_4_meaningful_l3979_397980

theorem sqrt_x_plus_4_meaningful (x : ℝ) : 
  (∃ y : ℝ, y^2 = x + 4) ↔ x ≥ -4 := by sorry

end NUMINAMATH_CALUDE_sqrt_x_plus_4_meaningful_l3979_397980


namespace NUMINAMATH_CALUDE_range_of_a_l3979_397939

-- Define the function
def f (x : ℝ) : ℝ := x^2 - 4*x

-- Define the theorem
theorem range_of_a (a : ℝ) :
  (∀ x ∈ Set.Icc (-4 : ℝ) a, f x ∈ Set.Icc (-4 : ℝ) 32) ∧
  (∀ y ∈ Set.Icc (-4 : ℝ) 32, ∃ x ∈ Set.Icc (-4 : ℝ) a, f x = y) →
  a ∈ Set.Icc 2 8 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l3979_397939


namespace NUMINAMATH_CALUDE_davids_english_marks_l3979_397965

theorem davids_english_marks 
  (math_marks : ℕ) 
  (physics_marks : ℕ) 
  (chemistry_marks : ℕ) 
  (biology_marks : ℕ) 
  (average_marks : ℚ) 
  (num_subjects : ℕ) 
  (h1 : math_marks = 35) 
  (h2 : physics_marks = 52) 
  (h3 : chemistry_marks = 47) 
  (h4 : biology_marks = 55) 
  (h5 : average_marks = 46.8) 
  (h6 : num_subjects = 5) : 
  ∃ (english_marks : ℕ), 
    english_marks = 45 ∧ 
    (english_marks + math_marks + physics_marks + chemistry_marks + biology_marks : ℚ) / num_subjects = average_marks := by
  sorry

end NUMINAMATH_CALUDE_davids_english_marks_l3979_397965


namespace NUMINAMATH_CALUDE_solution_set_for_inequality_l3979_397937

theorem solution_set_for_inequality (a b : ℝ) : 
  (∀ x : ℝ, x^2 - a*x + b = 0 ↔ x = 1 ∨ x = 2) →
  {x : ℝ | x ≤ 1} = Set.Iic 1 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_for_inequality_l3979_397937


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l3979_397942

theorem quadratic_inequality_range (x : ℝ) : 
  x^2 - 5*x + 6 ≤ 0 → 
  28 ≤ x^2 + 7*x + 10 ∧ x^2 + 7*x + 10 ≤ 40 := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l3979_397942


namespace NUMINAMATH_CALUDE_sage_code_is_8129_l3979_397977

-- Define the mapping of letters to digits
def letter_to_digit : Char → Nat
| 'M' => 0
| 'A' => 1
| 'G' => 2
| 'I' => 3
| 'C' => 4
| 'H' => 5
| 'O' => 6
| 'R' => 7
| 'S' => 8
| 'E' => 9
| _ => 0  -- Default case for other characters

-- Define a function to convert a string to a number
def code_to_number (code : String) : Nat :=
  code.foldl (fun acc c => 10 * acc + letter_to_digit c) 0

-- Theorem statement
theorem sage_code_is_8129 : code_to_number "SAGE" = 8129 := by
  sorry

end NUMINAMATH_CALUDE_sage_code_is_8129_l3979_397977


namespace NUMINAMATH_CALUDE_x_value_l3979_397968

theorem x_value (h1 : 25 * x^2 - 9 = 7) (h2 : 8 * (x - 2)^3 = 27) : x = 7/2 := by
  sorry

end NUMINAMATH_CALUDE_x_value_l3979_397968


namespace NUMINAMATH_CALUDE_min_value_a_sqrt_inequality_l3979_397902

theorem min_value_a_sqrt_inequality : 
  (∃ (a : ℝ), ∀ (x y : ℝ), x > 0 → y > 0 → Real.sqrt x + Real.sqrt y ≤ a * Real.sqrt (x + y)) ∧ 
  (∀ (a : ℝ), (∀ (x y : ℝ), x > 0 → y > 0 → Real.sqrt x + Real.sqrt y ≤ a * Real.sqrt (x + y)) → a ≥ Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_min_value_a_sqrt_inequality_l3979_397902


namespace NUMINAMATH_CALUDE_new_arithmetic_mean_l3979_397946

def original_set_size : ℕ := 60
def original_mean : ℚ := 42
def removed_numbers : List ℚ := [50, 60, 70]

theorem new_arithmetic_mean :
  let original_sum : ℚ := original_mean * original_set_size
  let removed_sum : ℚ := removed_numbers.sum
  let new_sum : ℚ := original_sum - removed_sum
  let new_set_size : ℕ := original_set_size - removed_numbers.length
  (new_sum / new_set_size : ℚ) = 41 := by sorry

end NUMINAMATH_CALUDE_new_arithmetic_mean_l3979_397946


namespace NUMINAMATH_CALUDE_cannot_determine_unique_ages_l3979_397983

-- Define variables for Julie and Aaron's ages
variable (J A : ℕ)

-- Define the relationship between their current ages
def current_age_relation : Prop := J = 4 * A

-- Define the relationship between their ages in 10 years
def future_age_relation : Prop := J + 10 = 4 * (A + 10)

-- Theorem stating that unique ages cannot be determined
theorem cannot_determine_unique_ages 
  (h1 : current_age_relation J A) 
  (h2 : future_age_relation J A) :
  ∃ (J' A' : ℕ), J' ≠ J ∧ A' ≠ A ∧ current_age_relation J' A' ∧ future_age_relation J' A' :=
sorry

end NUMINAMATH_CALUDE_cannot_determine_unique_ages_l3979_397983


namespace NUMINAMATH_CALUDE_sock_selection_theorem_l3979_397912

/-- Represents the number of socks of a given color -/
def num_socks : Fin 3 → Nat
  | 0 => 5  -- white
  | 1 => 5  -- brown
  | 2 => 3  -- blue

/-- Calculates the number of socks in odd positions for a given color -/
def odd_positions (color : Fin 3) : Nat :=
  (num_socks color + 1) / 2

/-- Calculates the number of socks in even positions for a given color -/
def even_positions (color : Fin 3) : Nat :=
  num_socks color / 2

/-- Calculates the number of ways to select a pair of socks of different colors from either odd or even positions -/
def select_pair_ways : Nat :=
  let white := 0
  let brown := 1
  let blue := 2
  (odd_positions white * odd_positions brown + even_positions white * even_positions brown) +
  (odd_positions brown * odd_positions blue + even_positions brown * even_positions blue) +
  (odd_positions white * odd_positions blue + even_positions white * even_positions blue)

/-- The main theorem stating that the number of ways to select a pair of socks is 29 -/
theorem sock_selection_theorem : select_pair_ways = 29 := by
  sorry

end NUMINAMATH_CALUDE_sock_selection_theorem_l3979_397912


namespace NUMINAMATH_CALUDE_customs_duration_l3979_397985

theorem customs_duration (navigation_time transport_time total_time : ℕ) 
  (h1 : navigation_time = 21)
  (h2 : transport_time = 7)
  (h3 : total_time = 30) :
  total_time - navigation_time - transport_time = 2 := by
  sorry

end NUMINAMATH_CALUDE_customs_duration_l3979_397985


namespace NUMINAMATH_CALUDE_y_intercept_of_line_l3979_397907

/-- Given a line with equation 4x + 6y - 2z = 24 and z = 3, prove that the y-intercept is (0, 5) -/
theorem y_intercept_of_line (x y z : ℝ) :
  4 * x + 6 * y - 2 * z = 24 →
  z = 3 →
  x = 0 →
  y = 5 := by
sorry

end NUMINAMATH_CALUDE_y_intercept_of_line_l3979_397907


namespace NUMINAMATH_CALUDE_smallest_a1_l3979_397996

def sequence_property (a : ℕ → ℝ) : Prop :=
  ∀ n > 1, a n = 7 * a (n - 1) - n

theorem smallest_a1 (a : ℕ → ℝ) :
  (∀ n, a n > 0) →
  sequence_property a →
  ∀ a1 : ℝ, (a 1 = a1 ∧ ∀ n, a n > 0) → a1 ≥ 13/36 :=
sorry

end NUMINAMATH_CALUDE_smallest_a1_l3979_397996


namespace NUMINAMATH_CALUDE_repair_cost_calculation_l3979_397950

-- Define the parameters
def purchase_price : ℝ := 4700
def selling_price : ℝ := 5800
def gain_percent : ℝ := 1.7543859649122806

-- Define the theorem
theorem repair_cost_calculation (repair_cost : ℝ) :
  (selling_price - (purchase_price + repair_cost)) / (purchase_price + repair_cost) * 100 = gain_percent →
  repair_cost = 1000 := by
  sorry

end NUMINAMATH_CALUDE_repair_cost_calculation_l3979_397950


namespace NUMINAMATH_CALUDE_cistern_fill_fraction_l3979_397955

/-- Represents the time in minutes it takes to fill a portion of the cistern -/
def fill_time : ℝ := 35

/-- Represents the fraction of the cistern filled in the given time -/
def fraction_filled : ℝ := 1

/-- Proves that the fraction of the cistern filled is 1 given the conditions -/
theorem cistern_fill_fraction :
  (fill_time = 35) → fraction_filled = 1 := by
  sorry

end NUMINAMATH_CALUDE_cistern_fill_fraction_l3979_397955


namespace NUMINAMATH_CALUDE_intersection_equality_condition_l3979_397973

theorem intersection_equality_condition (M N P : Set α) :
  (∀ (M N P : Set α), M = N → M ∩ P = N ∩ P) ∧
  (∃ (M N P : Set α), M ∩ P = N ∩ P ∧ M ≠ N) :=
sorry

end NUMINAMATH_CALUDE_intersection_equality_condition_l3979_397973


namespace NUMINAMATH_CALUDE_max_y_value_l3979_397918

theorem max_y_value (x y : ℤ) (h : x * y + 3 * x + 2 * y = -4) : y ≤ -1 := by
  sorry

end NUMINAMATH_CALUDE_max_y_value_l3979_397918


namespace NUMINAMATH_CALUDE_largest_value_l3979_397957

theorem largest_value : 
  let a := 3 + 1 + 2 + 8
  let b := 3 * 1 + 2 + 8
  let c := 3 + 1 * 2 + 8
  let d := 3 + 1 + 2 * 8
  let e := 3 * 1 * 2 * 8
  (e ≥ a) ∧ (e ≥ b) ∧ (e ≥ c) ∧ (e ≥ d) :=
by sorry

end NUMINAMATH_CALUDE_largest_value_l3979_397957


namespace NUMINAMATH_CALUDE_triangle_projection_shapes_l3979_397992

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A plane in 3D space -/
structure Plane3D where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- A triangle in 3D space -/
structure Triangle3D where
  a : Point3D
  b : Point3D
  c : Point3D

/-- Possible projection shapes -/
inductive ProjectionShape
  | Angle
  | Strip
  | TwoAngles
  | Triangle
  | CompositeShape

/-- Function to project a triangle onto a plane from a point -/
def project (t : Triangle3D) (p : Plane3D) (o : Point3D) : ProjectionShape :=
  sorry

/-- Theorem stating the possible projection shapes -/
theorem triangle_projection_shapes (t : Triangle3D) (p : Plane3D) (o : Point3D) 
  (h : o ∉ {x : Point3D | t.a.z = t.b.z ∧ t.b.z = t.c.z}) :
  ∃ (shape : ProjectionShape), project t p o = shape :=
sorry

end NUMINAMATH_CALUDE_triangle_projection_shapes_l3979_397992


namespace NUMINAMATH_CALUDE_body_speeds_correct_l3979_397962

/-- The distance between points A and B in meters -/
def distance : ℝ := 270

/-- The time (in seconds) after which the second body starts moving -/
def delay : ℝ := 11

/-- The time (in seconds) of the first meeting after the second body starts moving -/
def first_meeting : ℝ := 10

/-- The time (in seconds) of the second meeting after the second body starts moving -/
def second_meeting : ℝ := 40

/-- The speed of the first body in meters per second -/
def v1 : ℝ := 16

/-- The speed of the second body in meters per second -/
def v2 : ℝ := 9.6

theorem body_speeds_correct : 
  (delay + first_meeting) * v1 + first_meeting * v2 = distance ∧
  (delay + second_meeting) * v1 - second_meeting * v2 = distance ∧
  v1 > v2 ∧ v2 > 0 := by sorry

end NUMINAMATH_CALUDE_body_speeds_correct_l3979_397962


namespace NUMINAMATH_CALUDE_max_c_value_l3979_397952

theorem max_c_value (c : ℝ) : 
  (∀ x y : ℝ, x > y ∧ y > 0 → x^2 - 2*y^2 ≤ c*x*(y-x)) → 
  c ≤ 2*Real.sqrt 2 - 4 :=
by sorry

end NUMINAMATH_CALUDE_max_c_value_l3979_397952


namespace NUMINAMATH_CALUDE_complex_product_of_three_l3979_397928

theorem complex_product_of_three (α₁ α₂ α₃ : ℝ) (z₁ z₂ z₃ : ℂ) :
  z₁ = Complex.exp (Complex.I * α₁) →
  z₂ = Complex.exp (Complex.I * α₂) →
  z₃ = Complex.exp (Complex.I * α₃) →
  z₁ * z₂ = Complex.exp (Complex.I * (α₁ + α₂)) →
  z₂ * z₃ = Complex.exp (Complex.I * (α₂ + α₃)) →
  z₁ * z₂ * z₃ = Complex.exp (Complex.I * (α₁ + α₂ + α₃)) :=
by sorry

end NUMINAMATH_CALUDE_complex_product_of_three_l3979_397928


namespace NUMINAMATH_CALUDE_simplify_fraction_l3979_397944

theorem simplify_fraction : (4^5 + 4^3) / (4^4 - 4^2) = 68 / 15 := by sorry

end NUMINAMATH_CALUDE_simplify_fraction_l3979_397944


namespace NUMINAMATH_CALUDE_f_10_equals_10_l3979_397948

/-- An odd function satisfying certain conditions -/
def f (x : ℝ) : ℝ :=
  sorry

/-- Theorem stating the properties of f and the result to be proved -/
theorem f_10_equals_10 :
  (∀ x : ℝ, f (-x) = -f x) →  -- f is odd
  f 1 = 1 →  -- f(1) = 1
  (∀ x : ℝ, f (x + 2) = f x + f 2) →  -- f(x+2) = f(x) + f(2)
  f 10 = 10 :=
by sorry

end NUMINAMATH_CALUDE_f_10_equals_10_l3979_397948


namespace NUMINAMATH_CALUDE_complete_square_sum_l3979_397919

theorem complete_square_sum (a b c d : ℤ) : 
  (∀ x : ℝ, 64 * x^2 + 96 * x - 36 = 0 ↔ (a * x + b)^2 + d = c) →
  a > 0 →
  a + b + c + d = -94 :=
by sorry

end NUMINAMATH_CALUDE_complete_square_sum_l3979_397919


namespace NUMINAMATH_CALUDE_arithmetic_sequence_reciprocal_S_general_term_formula_l3979_397956

def sequence_a (n : ℕ) : ℚ := sorry

def sum_S (n : ℕ) : ℚ := sorry

axiom a_1 : sequence_a 1 = 3

axiom relation_a_S (n : ℕ) : n ≥ 2 → 2 * sequence_a n = sum_S n * sum_S (n - 1)

theorem arithmetic_sequence_reciprocal_S :
  ∀ n : ℕ, n ≥ 2 → (1 / sum_S n - 1 / sum_S (n - 1) = -1 / 2) :=
sorry

theorem general_term_formula :
  ∀ n : ℕ, n ≥ 2 →
    sequence_a n = 18 / ((8 - 3 * n) * (5 - 3 * n)) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_reciprocal_S_general_term_formula_l3979_397956


namespace NUMINAMATH_CALUDE_train_speed_calculation_l3979_397969

/-- Given a train and bridge scenario, calculate the train's speed in km/h -/
theorem train_speed_calculation (train_length bridge_length : ℝ) (time : ℝ) 
  (h1 : train_length = 360) 
  (h2 : bridge_length = 140) 
  (h3 : time = 40) : 
  (train_length + bridge_length) / time * 3.6 = 45 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_calculation_l3979_397969


namespace NUMINAMATH_CALUDE_rectangle_painting_possibilities_l3979_397913

theorem rectangle_painting_possibilities : 
  ∃! n : ℕ, n = (Finset.filter 
    (fun p : ℕ × ℕ => 
      p.2 > p.1 ∧ 
      (p.1 - 4) * (p.2 - 4) = 2 * p.1 * p.2 / 3 ∧ 
      p.1 > 0 ∧ p.2 > 0)
    (Finset.product (Finset.range 1000) (Finset.range 1000))).card ∧ 
  n = 3 :=
sorry

end NUMINAMATH_CALUDE_rectangle_painting_possibilities_l3979_397913


namespace NUMINAMATH_CALUDE_greatest_divisor_with_remainders_l3979_397917

theorem greatest_divisor_with_remainders : 
  let a := 1657
  let b := 2037
  let r1 := 6
  let r2 := 5
  Int.gcd (a - r1) (b - r2) = 127 := by sorry

end NUMINAMATH_CALUDE_greatest_divisor_with_remainders_l3979_397917


namespace NUMINAMATH_CALUDE_mary_travel_time_l3979_397925

/-- The total time Mary spends from calling the Uber to the plane being ready for takeoff -/
def total_time (uber_to_house : ℕ) (bag_check : ℕ) (wait_for_boarding : ℕ) : ℕ :=
  let uber_to_airport := 5 * uber_to_house
  let security := 3 * bag_check
  let wait_for_takeoff := 2 * wait_for_boarding
  uber_to_house + uber_to_airport + bag_check + security + wait_for_boarding + wait_for_takeoff

/-- The theorem stating that Mary's total travel preparation time is 3 hours -/
theorem mary_travel_time :
  total_time 10 15 20 = 180 :=
sorry

end NUMINAMATH_CALUDE_mary_travel_time_l3979_397925


namespace NUMINAMATH_CALUDE_ruby_pizza_tip_l3979_397982

/-- Represents the pizza order scenario --/
structure PizzaOrder where
  base_price : ℕ        -- Price of a pizza without toppings
  topping_price : ℕ     -- Price of each topping
  num_pizzas : ℕ        -- Number of pizzas ordered
  num_toppings : ℕ      -- Total number of toppings
  total_with_tip : ℕ    -- Total cost including tip

/-- Calculates the tip amount for a given pizza order --/
def calculate_tip (order : PizzaOrder) : ℕ :=
  order.total_with_tip - (order.base_price * order.num_pizzas + order.topping_price * order.num_toppings)

/-- Theorem stating that the tip for Ruby's pizza order is $5 --/
theorem ruby_pizza_tip :
  let order : PizzaOrder := {
    base_price := 10,
    topping_price := 1,
    num_pizzas := 3,
    num_toppings := 4,
    total_with_tip := 39
  }
  calculate_tip order = 5 := by
  sorry


end NUMINAMATH_CALUDE_ruby_pizza_tip_l3979_397982


namespace NUMINAMATH_CALUDE_ned_video_game_earnings_l3979_397971

/-- Given the total number of games, non-working games, and price per working game,
    calculates the total earnings from selling the working games. -/
def calculate_earnings (total_games : ℕ) (non_working_games : ℕ) (price_per_game : ℕ) : ℕ :=
  (total_games - non_working_games) * price_per_game

/-- Proves that Ned's earnings from selling his working video games is $63. -/
theorem ned_video_game_earnings :
  calculate_earnings 15 6 7 = 63 := by
  sorry

#eval calculate_earnings 15 6 7

end NUMINAMATH_CALUDE_ned_video_game_earnings_l3979_397971


namespace NUMINAMATH_CALUDE_target_hit_probability_l3979_397903

theorem target_hit_probability (p_A p_B : ℝ) (h_A : p_A = 0.8) (h_B : p_B = 0.7) :
  1 - (1 - p_A) * (1 - p_B) = 0.94 := by
  sorry

end NUMINAMATH_CALUDE_target_hit_probability_l3979_397903


namespace NUMINAMATH_CALUDE_siwoo_cranes_per_hour_l3979_397976

/-- The number of cranes Siwoo folds in 30 minutes -/
def cranes_per_30_min : ℕ := 180

/-- The number of minutes in an hour -/
def minutes_per_hour : ℕ := 60

/-- Calculates the number of cranes Siwoo folds in 1 hour -/
def cranes_per_hour : ℕ := cranes_per_30_min * (minutes_per_hour / 30)

/-- Theorem stating that Siwoo folds 360 cranes in 1 hour -/
theorem siwoo_cranes_per_hour :
  cranes_per_hour = 360 := by
  sorry

end NUMINAMATH_CALUDE_siwoo_cranes_per_hour_l3979_397976


namespace NUMINAMATH_CALUDE_marbles_lost_l3979_397947

theorem marbles_lost (initial : ℝ) (remaining : ℝ) (lost : ℝ) : 
  initial = 9.5 → remaining = 4.25 → lost = initial - remaining → lost = 5.25 := by
  sorry

end NUMINAMATH_CALUDE_marbles_lost_l3979_397947


namespace NUMINAMATH_CALUDE_log_sum_equality_l3979_397960

theorem log_sum_equality : Real.log 3 / Real.log 2 * (Real.log 4 / Real.log 3) + Real.log 8 / Real.log 4 + (5 : ℝ) ^ (Real.log 2 / Real.log 5) = 11 / 2 := by
  sorry

end NUMINAMATH_CALUDE_log_sum_equality_l3979_397960


namespace NUMINAMATH_CALUDE_total_fish_caught_l3979_397981

/-- The total number of fish caught by Jason, Ryan, and Jeffery is 100 -/
theorem total_fish_caught (jeffery_fish : ℕ) (h1 : jeffery_fish = 60) 
  (h2 : ∃ ryan_fish : ℕ, jeffery_fish = 2 * ryan_fish) 
  (h3 : ∃ jason_fish : ℕ, ∃ ryan_fish : ℕ, ryan_fish = 3 * jason_fish) : 
  ∃ total : ℕ, total = jeffery_fish + ryan_fish + jason_fish ∧ total = 100 :=
by sorry

end NUMINAMATH_CALUDE_total_fish_caught_l3979_397981


namespace NUMINAMATH_CALUDE_number_of_students_in_B_l3979_397949

/-- The number of students in school B -/
def students_B : ℕ := 30

/-- The number of students in school A -/
def students_A : ℕ := 4 * students_B

/-- The number of students in school C -/
def students_C : ℕ := 3 * students_B

/-- Theorem stating that the number of students in school B is 30 -/
theorem number_of_students_in_B : 
  students_A + students_C = 210 → students_B = 30 := by
  sorry


end NUMINAMATH_CALUDE_number_of_students_in_B_l3979_397949


namespace NUMINAMATH_CALUDE_income_ratio_problem_l3979_397990

/-- Given two persons P1 and P2 with incomes and expenditures, prove their income ratio --/
theorem income_ratio_problem (income_P1 income_P2 expenditure_P1 expenditure_P2 : ℚ) : 
  income_P1 = 3000 →
  expenditure_P1 / expenditure_P2 = 3 / 2 →
  income_P1 - expenditure_P1 = 1200 →
  income_P2 - expenditure_P2 = 1200 →
  income_P1 / income_P2 = 5 / 4 := by
  sorry

end NUMINAMATH_CALUDE_income_ratio_problem_l3979_397990


namespace NUMINAMATH_CALUDE_solve_system_l3979_397906

theorem solve_system (a b : ℚ) (h1 : a + a/4 = 3) (h2 : b - 2*a = 1) : a = 12/5 ∧ b = 29/5 := by
  sorry

end NUMINAMATH_CALUDE_solve_system_l3979_397906


namespace NUMINAMATH_CALUDE_problem_solution_l3979_397951

theorem problem_solution (x y : ℝ) 
  (hx : x = Real.sqrt 5 + Real.sqrt 3) 
  (hy : y = Real.sqrt 5 - Real.sqrt 3) : 
  (x^2 + 2*x*y + y^2) / (x^2 - y^2) = Real.sqrt 15 / 3 ∧ 
  Real.sqrt (x^2 + y^2 - 3) = Real.sqrt 13 ∨ 
  Real.sqrt (x^2 + y^2 - 3) = -Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3979_397951


namespace NUMINAMATH_CALUDE_expression_value_l3979_397938

theorem expression_value (a b : ℤ) (h1 : a = 4) (h2 : b = -3) : -a - b^2 + a*b = -25 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l3979_397938


namespace NUMINAMATH_CALUDE_range_of_a_l3979_397931

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, (-2 < x - 1 ∧ x - 1 < 3 ∧ x - a > 0) ↔ (-1 < x ∧ x < 4)) →
  a ≤ -1 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l3979_397931


namespace NUMINAMATH_CALUDE_impossible_to_face_up_all_coins_l3979_397979

/-- Represents the state of all coins -/
def CoinState := List Bool

/-- Represents a flip operation on 6 coins -/
def Flip := List Nat

/-- The initial state of the coins -/
def initialState : CoinState := 
  (List.replicate 1000 true) ++ (List.replicate 997 false)

/-- Applies a flip to a coin state -/
def applyFlip (state : CoinState) (flip : Flip) : CoinState :=
  sorry

/-- Checks if all coins are facing up -/
def allFacingUp (state : CoinState) : Bool :=
  state.all id

/-- Theorem stating that it's impossible to make all coins face up -/
theorem impossible_to_face_up_all_coins :
  ∀ (flips : List Flip), 
    ¬(allFacingUp (flips.foldl applyFlip initialState)) :=
by
  sorry

end NUMINAMATH_CALUDE_impossible_to_face_up_all_coins_l3979_397979


namespace NUMINAMATH_CALUDE_max_value_of_expression_l3979_397966

theorem max_value_of_expression (a b c d e : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) (he : 0 < e) :
  (a * b + b * c + c * d + d * e) / (2 * a^2 + b^2 + 2 * c^2 + d^2 + 2 * e^2) ≤ Real.sqrt (3 / 8) ∧
  ∃ a b c d e : ℝ, 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ 0 < e ∧
    (a * b + b * c + c * d + d * e) / (2 * a^2 + b^2 + 2 * c^2 + d^2 + 2 * e^2) = Real.sqrt (3 / 8) :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l3979_397966


namespace NUMINAMATH_CALUDE_simplify_fraction_l3979_397989

theorem simplify_fraction (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  (6 * a^2 * b * c) / (3 * a * b) = 2 * a * c := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l3979_397989


namespace NUMINAMATH_CALUDE_circles_intersect_l3979_397922

/-- Two circles in a 2D plane --/
structure TwoCircles where
  c1_center : ℝ × ℝ
  c2_center : ℝ × ℝ
  c1_radius : ℝ
  c2_radius : ℝ

/-- Definition of intersecting circles --/
def are_intersecting (tc : TwoCircles) : Prop :=
  let d := Real.sqrt ((tc.c2_center.1 - tc.c1_center.1)^2 + (tc.c2_center.2 - tc.c1_center.2)^2)
  (tc.c1_radius + tc.c2_radius > d) ∧ (d > abs (tc.c1_radius - tc.c2_radius))

/-- The main theorem --/
theorem circles_intersect (tc : TwoCircles) 
  (h1 : tc.c1_center = (-2, 2))
  (h2 : tc.c2_center = (2, 5))
  (h3 : tc.c1_radius = 2)
  (h4 : tc.c2_radius = 4)
  (h5 : Real.sqrt ((tc.c2_center.1 - tc.c1_center.1)^2 + (tc.c2_center.2 - tc.c1_center.2)^2) = 5)
  : are_intersecting tc := by
  sorry

end NUMINAMATH_CALUDE_circles_intersect_l3979_397922


namespace NUMINAMATH_CALUDE_solve_movie_problem_l3979_397910

def movie_problem (rented_movie_cost bought_movie_cost total_spent : ℚ) : Prop :=
  let num_tickets : ℕ := 2
  let other_costs : ℚ := rented_movie_cost + bought_movie_cost
  let ticket_total_cost : ℚ := total_spent - other_costs
  let ticket_cost : ℚ := ticket_total_cost / num_tickets
  ticket_cost = 10.62

theorem solve_movie_problem :
  movie_problem 1.59 13.95 36.78 := by
  sorry

end NUMINAMATH_CALUDE_solve_movie_problem_l3979_397910


namespace NUMINAMATH_CALUDE_exactly_two_correct_statements_l3979_397900

theorem exactly_two_correct_statements : 
  let f : ℝ → ℝ := λ x => x + 1/x
  let statement1 := ∃ (m : ℝ), ∀ (x : ℝ), f x ≥ m ∧ (∃ (x₀ : ℝ), f x₀ = m) ∧ m = 2
  let statement2 := ∀ (a b : ℝ), a^2 + b^2 ≥ 2*a*b
  let statement3 := ∀ (a b c d : ℝ), a > b ∧ b > 0 ∧ c > d ∧ d > 0 → a*c > b*d
  let statement4 := (¬∃ (x : ℝ), x^2 + x + 1 ≥ 0) ↔ (∀ (x : ℝ), x^2 + x + 1 ≥ 0)
  let statement5 := ∀ (x y : ℝ), x > y ↔ 1/x < 1/y
  let statement6 := ∀ (p q : Prop), (¬(p ∨ q)) → (¬(¬p ∨ ¬q))
  (statement2 ∧ statement3 ∧ ¬statement1 ∧ ¬statement4 ∧ ¬statement5 ∧ ¬statement6) := by sorry

end NUMINAMATH_CALUDE_exactly_two_correct_statements_l3979_397900


namespace NUMINAMATH_CALUDE_strawberry_loss_l3979_397959

theorem strawberry_loss (total_weight : ℕ) (marco_weight : ℕ) (dad_weight : ℕ) 
  (h1 : total_weight = 36)
  (h2 : marco_weight = 12)
  (h3 : dad_weight = 16) :
  total_weight - (marco_weight + dad_weight) = 8 := by
  sorry

end NUMINAMATH_CALUDE_strawberry_loss_l3979_397959


namespace NUMINAMATH_CALUDE_train_crossing_time_l3979_397909

/-- Proves that a train with given length and speed takes the calculated time to cross a pole -/
theorem train_crossing_time (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) :
  train_length = 350 →
  train_speed_kmh = 60 →
  crossing_time = 21 →
  crossing_time = train_length / (train_speed_kmh * 1000 / 3600) := by
  sorry

#check train_crossing_time

end NUMINAMATH_CALUDE_train_crossing_time_l3979_397909


namespace NUMINAMATH_CALUDE_joohyeon_snacks_l3979_397901

/-- Represents the number of snacks bought by Joohyeon -/
def num_snacks : ℕ := 3

/-- Represents the number of candies bought by Joohyeon -/
def num_candies : ℕ := 5

/-- Cost of each candy in won -/
def candy_cost : ℕ := 300

/-- Cost of each snack in won -/
def snack_cost : ℕ := 500

/-- Total amount spent in won -/
def total_spent : ℕ := 3000

/-- Total number of items bought -/
def total_items : ℕ := 8

theorem joohyeon_snacks :
  (num_candies * candy_cost + num_snacks * snack_cost = total_spent) ∧
  (num_candies + num_snacks = total_items) :=
by sorry

end NUMINAMATH_CALUDE_joohyeon_snacks_l3979_397901


namespace NUMINAMATH_CALUDE_true_propositions_l3979_397972

-- Define the four propositions
def proposition1 : Prop := sorry
def proposition2 : Prop := sorry
def proposition3 : Prop := sorry
def proposition4 : Prop := sorry

-- Theorem stating which propositions are true
theorem true_propositions : 
  (¬ proposition1) ∧ proposition2 ∧ proposition3 ∧ (¬ proposition4) := by
  sorry

end NUMINAMATH_CALUDE_true_propositions_l3979_397972


namespace NUMINAMATH_CALUDE_simplify_expression_l3979_397945

theorem simplify_expression : (6 * 10^10) / (2 * 10^4) = 3000000 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3979_397945


namespace NUMINAMATH_CALUDE_range_of_a_l3979_397943

-- Define a decreasing function on (-1, 1)
def DecreasingFunction (f : ℝ → ℝ) : Prop :=
  ∀ x y, -1 < x ∧ x < y ∧ y < 1 → f x > f y

-- Theorem statement
theorem range_of_a (f : ℝ → ℝ) (a : ℝ) 
  (h_decreasing : DecreasingFunction f)
  (h_inequality : f (1 - a) > f (2 * a - 1)) :
  2 / 3 < a ∧ a < 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l3979_397943


namespace NUMINAMATH_CALUDE_expo_assignment_count_l3979_397932

/-- Represents the four pavilions at the Shanghai World Expo -/
inductive Pavilion
  | China
  | UK
  | Australia
  | Russia

/-- The total number of volunteers -/
def total_volunteers : Nat := 5

/-- The number of pavilions -/
def num_pavilions : Nat := 4

/-- A function that represents a valid assignment of volunteers to pavilions -/
def is_valid_assignment (assignment : Pavilion → Nat) : Prop :=
  (∀ p : Pavilion, assignment p > 0) ∧
  (assignment Pavilion.China + assignment Pavilion.UK + 
   assignment Pavilion.Australia + assignment Pavilion.Russia = total_volunteers)

/-- The number of ways for A and B to be assigned to pavilions -/
def num_ways_AB : Nat := num_pavilions * num_pavilions

/-- The theorem to be proved -/
theorem expo_assignment_count :
  (∃ (ways : Nat), ways = num_ways_AB ∧
    ∃ (remaining_assignments : Nat),
      ways * remaining_assignments = 72 ∧
      ∀ (assignment : Pavilion → Nat),
        is_valid_assignment assignment →
        remaining_assignments > 0) := by
  sorry

end NUMINAMATH_CALUDE_expo_assignment_count_l3979_397932


namespace NUMINAMATH_CALUDE_plane_from_three_points_l3979_397908

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A plane in 3D space -/
structure Plane3D where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Three points are non-collinear if they do not lie on the same line -/
def NonCollinear (p1 p2 p3 : Point3D) : Prop :=
  ¬∃ (t : ℝ), p3 = Point3D.mk (p1.x + t * (p2.x - p1.x)) (p1.y + t * (p2.y - p1.y)) (p1.z + t * (p2.z - p1.z))

/-- A plane is uniquely determined by three non-collinear points -/
theorem plane_from_three_points (p1 p2 p3 : Point3D) (h : NonCollinear p1 p2 p3) :
  ∃! (plane : Plane3D), (plane.a * p1.x + plane.b * p1.y + plane.c * p1.z + plane.d = 0) ∧
                        (plane.a * p2.x + plane.b * p2.y + plane.c * p2.z + plane.d = 0) ∧
                        (plane.a * p3.x + plane.b * p3.y + plane.c * p3.z + plane.d = 0) :=
by sorry

end NUMINAMATH_CALUDE_plane_from_three_points_l3979_397908


namespace NUMINAMATH_CALUDE_brown_shoes_count_l3979_397911

theorem brown_shoes_count (brown_shoes black_shoes : ℕ) : 
  black_shoes = 2 * brown_shoes →
  brown_shoes + black_shoes = 66 →
  brown_shoes = 22 := by
sorry

end NUMINAMATH_CALUDE_brown_shoes_count_l3979_397911


namespace NUMINAMATH_CALUDE_fraction_of_third_is_eighth_l3979_397904

theorem fraction_of_third_is_eighth : (1 / 8) / (1 / 3) = 3 / 8 := by sorry

end NUMINAMATH_CALUDE_fraction_of_third_is_eighth_l3979_397904


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l3979_397934

theorem quadratic_equation_roots : ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
  x₁^2 - 3*x₁ - 1 = 0 ∧ x₂^2 - 3*x₂ - 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l3979_397934


namespace NUMINAMATH_CALUDE_smaller_prime_l3979_397999

theorem smaller_prime (x y : ℕ) (hx : Nat.Prime x) (hy : Nat.Prime y)
  (h1 : x + y = 36) (h2 : 4 * x + y = 87) : x < y := by
  sorry

end NUMINAMATH_CALUDE_smaller_prime_l3979_397999


namespace NUMINAMATH_CALUDE_calculate_second_solution_percentage_l3979_397941

/-- Given two solutions mixed to form a final solution, calculates the percentage of the second solution. -/
theorem calculate_second_solution_percentage
  (final_volume : ℝ)
  (final_percentage : ℝ)
  (first_volume : ℝ)
  (first_percentage : ℝ)
  (second_volume : ℝ)
  (h_final_volume : final_volume = 40)
  (h_final_percentage : final_percentage = 0.45)
  (h_first_volume : first_volume = 28)
  (h_first_percentage : first_percentage = 0.30)
  (h_second_volume : second_volume = 12)
  (h_volume_sum : first_volume + second_volume = final_volume)
  (h_substance_balance : first_volume * first_percentage + second_volume * (second_percentage / 100) = final_volume * final_percentage) :
  second_percentage = 80 := by
  sorry

#check calculate_second_solution_percentage

end NUMINAMATH_CALUDE_calculate_second_solution_percentage_l3979_397941


namespace NUMINAMATH_CALUDE_train_speed_ratio_l3979_397954

theorem train_speed_ratio (v1 v2 : ℝ) (t1 t2 : ℝ) (h1 : t1 = 4) (h2 : t2 = 36) :
  v1 * t1 / (v2 * t2) = 1 / 9 → v1 = v2 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_ratio_l3979_397954


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l3979_397916

/-- A geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_product (a : ℕ → ℝ) (h : GeometricSequence a) 
    (h1 : a 5 * a 14 = 5) : 
  a 8 * a 9 * a 10 * a 11 = 25 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l3979_397916
