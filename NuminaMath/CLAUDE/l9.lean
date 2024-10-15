import Mathlib

namespace NUMINAMATH_CALUDE_instantaneous_velocity_at_3_l9_937

/-- The motion equation of an object -/
def s (t : ℝ) : ℝ := t^3 + t^2 - 1

/-- The velocity function derived from the motion equation -/
def v (t : ℝ) : ℝ := 3 * t^2 + 2 * t

theorem instantaneous_velocity_at_3 : v 3 = 33 := by
  sorry

end NUMINAMATH_CALUDE_instantaneous_velocity_at_3_l9_937


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l9_972

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a b : ℕ → ℝ) :
  arithmetic_sequence a →
  arithmetic_sequence b →
  a 1 + b 1 = 7 →
  a 3 + b 3 = 21 →
  a 5 + b 5 = 35 :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l9_972


namespace NUMINAMATH_CALUDE_race_outcomes_count_l9_924

/-- The number of participants in the race -/
def n : ℕ := 6

/-- The number of places we're considering -/
def k : ℕ := 4

/-- The number of different possible outcomes for the first four places in the race -/
def race_outcomes : ℕ := n * (n - 1) * (n - 2) * (n - 3)

theorem race_outcomes_count : race_outcomes = 360 := by
  sorry

end NUMINAMATH_CALUDE_race_outcomes_count_l9_924


namespace NUMINAMATH_CALUDE_artist_painted_thirteen_pictures_l9_939

/-- The number of pictures painted by an artist over three months -/
def total_pictures (june july august : ℕ) : ℕ := june + july + august

/-- Theorem stating that the artist painted 13 pictures in total -/
theorem artist_painted_thirteen_pictures : 
  total_pictures 2 2 9 = 13 := by sorry

end NUMINAMATH_CALUDE_artist_painted_thirteen_pictures_l9_939


namespace NUMINAMATH_CALUDE_equation_with_integer_roots_l9_922

theorem equation_with_integer_roots :
  ∃ (a b c : ℕ), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  ∃ (x y : ℤ), x ≠ y ∧
  (1 : ℚ) / (x + a) + (1 : ℚ) / (x + b) = (1 : ℚ) / c ∧
  (1 : ℚ) / (y + a) + (1 : ℚ) / (y + b) = (1 : ℚ) / c :=
by sorry

end NUMINAMATH_CALUDE_equation_with_integer_roots_l9_922


namespace NUMINAMATH_CALUDE_four_digit_number_theorem_l9_980

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def reverse_digits (n : ℕ) : ℕ :=
  let d1 := n / 1000
  let d2 := (n / 100) % 10
  let d3 := (n / 10) % 10
  let d4 := n % 10
  d4 * 1000 + d3 * 100 + d2 * 10 + d1

theorem four_digit_number_theorem (n : ℕ) : 
  is_four_digit n ∧
  (∃ k : ℕ, n + 1 = 15 * k) ∧
  (∃ m : ℕ, n - 3 = 38 * m) ∧
  (∃ l : ℕ, n + reverse_digits n = 10 * l) →
  n = 1409 ∨ n = 1979 :=
by sorry

end NUMINAMATH_CALUDE_four_digit_number_theorem_l9_980


namespace NUMINAMATH_CALUDE_ninth_term_value_l9_977

/-- An arithmetic sequence with specified third and sixth terms -/
structure ArithmeticSequence where
  a : ℝ  -- first term
  d : ℝ  -- common difference
  third_term : a + 2 * d = 25
  sixth_term : a + 5 * d = 31

/-- The ninth term of the arithmetic sequence -/
def ninth_term (seq : ArithmeticSequence) : ℝ := seq.a + 8 * seq.d

theorem ninth_term_value (seq : ArithmeticSequence) : ninth_term seq = 37 := by
  sorry

end NUMINAMATH_CALUDE_ninth_term_value_l9_977


namespace NUMINAMATH_CALUDE_modulus_of_complex_number_l9_908

theorem modulus_of_complex_number (z : ℂ) : z = 2 / (1 - Complex.I) → Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_complex_number_l9_908


namespace NUMINAMATH_CALUDE_product_pass_rate_l9_946

/-- The pass rate of a product going through two independent processing steps -/
def pass_rate (a b : ℝ) : ℝ := (1 - a) * (1 - b)

/-- Theorem stating that the pass rate of a product going through two independent processing steps
    with defect rates a and b is (1-a)·(1-b) -/
theorem product_pass_rate (a b : ℝ) (ha : 0 ≤ a ∧ a ≤ 1) (hb : 0 ≤ b ∧ b ≤ 1) :
  pass_rate a b = (1 - a) * (1 - b) := by
  sorry

end NUMINAMATH_CALUDE_product_pass_rate_l9_946


namespace NUMINAMATH_CALUDE_production_days_l9_925

theorem production_days (n : ℕ) (h1 : (50 * n + 90) / (n + 1) = 54) : n = 9 := by
  sorry

end NUMINAMATH_CALUDE_production_days_l9_925


namespace NUMINAMATH_CALUDE_scientific_notation_of_12400_l9_934

theorem scientific_notation_of_12400 :
  let num_athletes : ℕ := 12400
  1.24 * (10 : ℝ)^4 = num_athletes := by sorry

end NUMINAMATH_CALUDE_scientific_notation_of_12400_l9_934


namespace NUMINAMATH_CALUDE_fibSum_eq_three_l9_953

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

/-- The sum of F_n / 2^n from n = 0 to infinity -/
noncomputable def fibSum : ℝ := ∑' n, (fib n : ℝ) / (2 : ℝ) ^ n

/-- Theorem stating that the sum of F_n / 2^n from n = 0 to infinity equals 3 -/
theorem fibSum_eq_three : fibSum = 3 := by sorry

end NUMINAMATH_CALUDE_fibSum_eq_three_l9_953


namespace NUMINAMATH_CALUDE_partner_c_profit_share_l9_950

/-- Given the investment ratios of partners A, B, and C, and a total profit,
    calculate C's share of the profit. -/
theorem partner_c_profit_share 
  (a b c : ℝ) -- Investments of partners A, B, and C
  (total_profit : ℝ) -- Total profit
  (ha : a = 3 * b) -- A invests 3 times as much as B
  (hc : a = 2 / 3 * c) -- A invests 2/3 of what C invests
  : c / (a + b + c) * total_profit = 9 / 17 * total_profit :=
by sorry

end NUMINAMATH_CALUDE_partner_c_profit_share_l9_950


namespace NUMINAMATH_CALUDE_bacteria_after_10_hours_l9_993

/-- Represents the number of bacteria in the colony after a given number of hours -/
def bacteria_count (hours : ℕ) : ℕ :=
  2^hours

/-- Theorem stating that after 10 hours, the bacteria count is 1024 -/
theorem bacteria_after_10_hours :
  bacteria_count 10 = 1024 := by
  sorry

end NUMINAMATH_CALUDE_bacteria_after_10_hours_l9_993


namespace NUMINAMATH_CALUDE_cos_beta_minus_gamma_bounds_l9_989

theorem cos_beta_minus_gamma_bounds (k : ℝ) (α β γ : ℝ) 
  (h1 : 0 < k) (h2 : k < 2)
  (eq1 : Real.cos α + k * Real.cos β + (2 - k) * Real.cos γ = 0)
  (eq2 : Real.sin α + k * Real.sin β + (2 - k) * Real.sin γ = 0) :
  (∀ x, Real.cos (β - γ) ≤ x → x ≤ -1/2) ∧ 
  (∃ k₁ k₂, 0 < k₁ ∧ k₁ < 2 ∧ 0 < k₂ ∧ k₂ < 2 ∧ 
    Real.cos (β - γ) = -1/2 ∧ Real.cos (β - γ) = -1) :=
by sorry

end NUMINAMATH_CALUDE_cos_beta_minus_gamma_bounds_l9_989


namespace NUMINAMATH_CALUDE_matrix_power_2020_l9_905

def A : Matrix (Fin 2) (Fin 2) ℤ := !![1, 0; 3, 1]

theorem matrix_power_2020 :
  A ^ 2020 = !![1, 0; 6060, 1] := by sorry

end NUMINAMATH_CALUDE_matrix_power_2020_l9_905


namespace NUMINAMATH_CALUDE_max_sum_of_product_60_l9_932

theorem max_sum_of_product_60 (a b c : ℕ) (h : a * b * c = 60) :
  a + b + c ≤ 62 := by
  sorry

end NUMINAMATH_CALUDE_max_sum_of_product_60_l9_932


namespace NUMINAMATH_CALUDE_perpendicular_lines_a_values_l9_919

-- Define the lines
def line1 (a : ℝ) (x y : ℝ) : Prop := a * x - y + 2 * a = 0
def line2 (a : ℝ) (x y : ℝ) : Prop := (2 * a - 1) * x + a * y = 0

-- Define perpendicularity
def perpendicular (a : ℝ) : Prop :=
  ∀ x₁ y₁ x₂ y₂ : ℝ, line1 a x₁ y₁ ∧ line2 a x₂ y₂ →
    (x₁ - x₂) * (y₁ - y₂) = 0

-- Theorem statement
theorem perpendicular_lines_a_values :
  ∀ a : ℝ, perpendicular a → a = 0 ∨ a = 1 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_lines_a_values_l9_919


namespace NUMINAMATH_CALUDE_picnic_cost_is_60_l9_951

/-- Calculates the total cost of a picnic basket given the number of people and item prices. -/
def picnic_cost (num_people : ℕ) (sandwich_price fruit_salad_price soda_price snack_price : ℕ) : ℕ :=
  let sandwich_cost := num_people * sandwich_price
  let fruit_salad_cost := num_people * fruit_salad_price
  let soda_cost := num_people * 2 * soda_price
  let snack_cost := 3 * snack_price
  sandwich_cost + fruit_salad_cost + soda_cost + snack_cost

/-- Theorem stating that the total cost of the picnic basket is $60. -/
theorem picnic_cost_is_60 :
  picnic_cost 4 5 3 2 4 = 60 := by
  sorry

#eval picnic_cost 4 5 3 2 4

end NUMINAMATH_CALUDE_picnic_cost_is_60_l9_951


namespace NUMINAMATH_CALUDE_simplify_expression_l9_968

theorem simplify_expression (w : ℝ) : 3*w + 6*w - 9*w + 12*w - 15*w + 21 = -3*w + 21 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l9_968


namespace NUMINAMATH_CALUDE_multiplication_equality_l9_918

theorem multiplication_equality : 500 * 3986 * 0.3986 * 5 = 0.25 * 3986^2 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_equality_l9_918


namespace NUMINAMATH_CALUDE_unique_root_condition_l9_947

theorem unique_root_condition (k : ℝ) : 
  (∃! x : ℝ, (x / (x + 3) + x / (x + 4) = k * x)) ↔ (k = Real.sqrt 3 ∨ k = -Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_unique_root_condition_l9_947


namespace NUMINAMATH_CALUDE_charlotte_tuesday_poodles_l9_994

/-- Represents the schedule for Charlotte's dog walking --/
structure DogWalkingSchedule where
  poodles_monday : ℕ
  chihuahuas_monday : ℕ
  labradors_wednesday : ℕ
  poodle_time : ℕ
  chihuahua_time : ℕ
  labrador_time : ℕ
  total_time : ℕ

/-- Calculates the number of poodles Charlotte can walk on Tuesday --/
def poodles_tuesday (s : DogWalkingSchedule) : ℕ :=
  let monday_time := s.poodles_monday * s.poodle_time + s.chihuahuas_monday * s.chihuahua_time
  let wednesday_time := s.labradors_wednesday * s.labrador_time
  let tuesday_time := s.total_time - monday_time - wednesday_time - s.chihuahuas_monday * s.chihuahua_time
  tuesday_time / s.poodle_time

/-- Theorem stating that Charlotte can walk 4 poodles on Tuesday --/
theorem charlotte_tuesday_poodles (s : DogWalkingSchedule) 
  (h1 : s.poodles_monday = 4)
  (h2 : s.chihuahuas_monday = 2)
  (h3 : s.labradors_wednesday = 4)
  (h4 : s.poodle_time = 2)
  (h5 : s.chihuahua_time = 1)
  (h6 : s.labrador_time = 3)
  (h7 : s.total_time = 32) :
  poodles_tuesday s = 4 := by
  sorry


end NUMINAMATH_CALUDE_charlotte_tuesday_poodles_l9_994


namespace NUMINAMATH_CALUDE_first_cousin_ate_two_l9_959

/-- The number of sandwiches Ruth prepared -/
def total_sandwiches : ℕ := 10

/-- The number of sandwiches Ruth ate -/
def ruth_ate : ℕ := 1

/-- The number of sandwiches Ruth gave to her brother -/
def brother_ate : ℕ := 2

/-- The number of sandwiches eaten by the two other cousins -/
def other_cousins_ate : ℕ := 2

/-- The number of sandwiches left -/
def sandwiches_left : ℕ := 3

/-- The number of sandwiches eaten by the first cousin -/
def first_cousin_ate : ℕ := total_sandwiches - (ruth_ate + brother_ate + other_cousins_ate + sandwiches_left)

theorem first_cousin_ate_two : first_cousin_ate = 2 := by
  sorry

end NUMINAMATH_CALUDE_first_cousin_ate_two_l9_959


namespace NUMINAMATH_CALUDE_january_savings_l9_912

def savings_challenge (initial_savings : ℚ) : ℕ → ℚ
  | 0 => initial_savings
  | n + 1 => 2 * savings_challenge initial_savings n

theorem january_savings (may_savings : ℚ) :
  may_savings = 160 →
  ∃ (initial_savings : ℚ),
    savings_challenge initial_savings 4 = may_savings ∧
    initial_savings = 10 :=
by sorry

end NUMINAMATH_CALUDE_january_savings_l9_912


namespace NUMINAMATH_CALUDE_polynomial_floor_property_l9_935

/-- A polynomial with real coefficients -/
def RealPolynomial := ℝ → ℝ

/-- The floor function -/
noncomputable def floor : ℝ → ℤ := sorry

/-- The property that P(⌊x⌋) = ⌊P(x)⌋ for all real x -/
def HasFloorProperty (P : RealPolynomial) : Prop :=
  ∀ x : ℝ, P (↑(floor x)) = ↑(floor (P x))

/-- The main theorem -/
theorem polynomial_floor_property (P : RealPolynomial) :
  HasFloorProperty P → ∃ k : ℤ, ∀ x : ℝ, P x = x + k := by sorry

end NUMINAMATH_CALUDE_polynomial_floor_property_l9_935


namespace NUMINAMATH_CALUDE_average_sour_candies_is_correct_l9_902

/-- The number of people in the group -/
def num_people : ℕ := 4

/-- The number of sour candies Wendy's brother has -/
def brother_sour_candies : ℕ := 4

/-- The number of sour candies Wendy has -/
def wendy_sour_candies : ℕ := 5

/-- The number of sour candies their cousin has -/
def cousin_sour_candies : ℕ := 1

/-- The number of sour candies their uncle has -/
def uncle_sour_candies : ℕ := 3

/-- The total number of sour candies -/
def total_sour_candies : ℕ := brother_sour_candies + wendy_sour_candies + cousin_sour_candies + uncle_sour_candies

/-- The average number of sour candies per person -/
def average_sour_candies : ℚ := total_sour_candies / num_people

theorem average_sour_candies_is_correct : average_sour_candies = 13/4 := by
  sorry

end NUMINAMATH_CALUDE_average_sour_candies_is_correct_l9_902


namespace NUMINAMATH_CALUDE_davids_chemistry_marks_l9_901

theorem davids_chemistry_marks
  (english : ℕ)
  (mathematics : ℕ)
  (physics : ℕ)
  (biology : ℕ)
  (average : ℚ)
  (h1 : english = 96)
  (h2 : mathematics = 95)
  (h3 : physics = 82)
  (h4 : biology = 92)
  (h5 : average = 90.4)
  (h6 : (english + mathematics + physics + biology + chemistry : ℚ) / 5 = average) :
  chemistry = 87 :=
by sorry

end NUMINAMATH_CALUDE_davids_chemistry_marks_l9_901


namespace NUMINAMATH_CALUDE_framed_painting_ratio_l9_916

/-- Represents the dimensions of a framed painting -/
structure FramedPainting where
  painting_width : ℝ
  painting_height : ℝ
  side_frame_width : ℝ

/-- Calculates the dimensions of the framed painting -/
def framed_dimensions (fp : FramedPainting) : (ℝ × ℝ) :=
  (fp.painting_width + 2 * fp.side_frame_width,
   fp.painting_height + 6 * fp.side_frame_width)

/-- Calculates the area of the framed painting -/
def framed_area (fp : FramedPainting) : ℝ :=
  let (w, h) := framed_dimensions fp
  w * h

/-- Calculates the area of the painting -/
def painting_area (fp : FramedPainting) : ℝ :=
  fp.painting_width * fp.painting_height

/-- Theorem stating the ratio of smaller to larger dimension of the framed painting -/
theorem framed_painting_ratio (fp : FramedPainting) 
  (h1 : fp.painting_width = 20)
  (h2 : fp.painting_height = 30)
  (h3 : framed_area fp = 3 * painting_area fp) :
  let (w, h) := framed_dimensions fp
  w / h = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_framed_painting_ratio_l9_916


namespace NUMINAMATH_CALUDE_perpendicular_lines_from_parallel_planes_l9_983

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (parallel_plane : Plane → Plane → Prop)
variable (perpendicular_lines : Line → Line → Prop)

-- State the theorem
theorem perpendicular_lines_from_parallel_planes 
  (m n : Line) (α β : Plane) 
  (h1 : perpendicular m α)
  (h2 : parallel_line_plane n β)
  (h3 : parallel_plane α β) :
  perpendicular_lines m n :=
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_from_parallel_planes_l9_983


namespace NUMINAMATH_CALUDE_same_color_probability_problem_die_l9_995

/-- Represents a 30-sided die with colored sides -/
structure ColoredDie :=
  (red : ℕ)
  (green : ℕ)
  (blue : ℕ)
  (golden : ℕ)
  (total : ℕ)
  (h_total : red + green + blue + golden = total)

/-- The probability of rolling the same color on two identical colored dice -/
def same_color_probability (d : ColoredDie) : ℚ :=
  (d.red^2 + d.green^2 + d.blue^2 + d.golden^2 : ℚ) / d.total^2

/-- The specific 30-sided die described in the problem -/
def problem_die : ColoredDie :=
  { red := 6
  , green := 8
  , blue := 10
  , golden := 6
  , total := 30
  , h_total := by simp }

/-- Theorem stating the probability of rolling the same color on two problem_die is 59/225 -/
theorem same_color_probability_problem_die :
  same_color_probability problem_die = 59 / 225 := by
  sorry

end NUMINAMATH_CALUDE_same_color_probability_problem_die_l9_995


namespace NUMINAMATH_CALUDE_aarons_brothers_l9_990

theorem aarons_brothers (bennett_brothers : ℕ) (aaron_brothers : ℕ) 
  (h1 : bennett_brothers = 6) 
  (h2 : bennett_brothers = 2 * aaron_brothers - 2) : 
  aaron_brothers = 4 := by
sorry

end NUMINAMATH_CALUDE_aarons_brothers_l9_990


namespace NUMINAMATH_CALUDE_museum_ticket_fraction_l9_906

def total_amount : ℚ := 180
def sandwich_fraction : ℚ := 1/5
def book_fraction : ℚ := 1/2
def leftover_amount : ℚ := 24

theorem museum_ticket_fraction :
  let spent_amount := total_amount - leftover_amount
  let sandwich_cost := sandwich_fraction * total_amount
  let book_cost := book_fraction * total_amount
  let museum_ticket_cost := spent_amount - sandwich_cost - book_cost
  museum_ticket_cost / total_amount = 1/6 := by sorry

end NUMINAMATH_CALUDE_museum_ticket_fraction_l9_906


namespace NUMINAMATH_CALUDE_isosceles_triangle_base_length_isosceles_triangle_base_length_proof_l9_996

/-- An isosceles triangle with congruent sides of 7 cm and perimeter of 25 cm has a base of 11 cm. -/
theorem isosceles_triangle_base_length : ℝ → Prop :=
  fun base =>
    let congruent_side := 7
    let perimeter := 25
    (2 * congruent_side + base = perimeter) →
    base = 11

-- The proof is omitted
theorem isosceles_triangle_base_length_proof : isosceles_triangle_base_length 11 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_base_length_isosceles_triangle_base_length_proof_l9_996


namespace NUMINAMATH_CALUDE_standardDeviation_best_stability_measure_l9_987

-- Define the type for crop yields
def CropYield := ℝ

-- Define a list of crop yields
def YieldList := List CropYield

-- Define statistical measures
def mean (yields : YieldList) : ℝ := sorry
def standardDeviation (yields : YieldList) : ℝ := sorry
def maximum (yields : YieldList) : ℝ := sorry
def median (yields : YieldList) : ℝ := sorry

-- Define a measure of stability
def stabilityMeasure : (YieldList → ℝ) → Prop := sorry

-- Theorem statement
theorem standardDeviation_best_stability_measure :
  ∀ (yields : YieldList),
    stabilityMeasure standardDeviation ∧
    ¬stabilityMeasure mean ∧
    ¬stabilityMeasure maximum ∧
    ¬stabilityMeasure median :=
  sorry

end NUMINAMATH_CALUDE_standardDeviation_best_stability_measure_l9_987


namespace NUMINAMATH_CALUDE_solve_equation_l9_948

theorem solve_equation : ∃ x : ℝ, 3 * x = (36 - x) + 16 ∧ x = 13 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l9_948


namespace NUMINAMATH_CALUDE_distance_between_polar_points_l9_955

/-- Given two points P and Q in polar coordinates, where the difference of their angles is π/3,
    prove that the distance between them is 8√10. -/
theorem distance_between_polar_points (α β : Real) :
  let P : Real × Real := (4, α)
  let Q : Real × Real := (12, β)
  α - β = π / 3 →
  let distance := Real.sqrt ((12 * Real.cos β - 4 * Real.cos α)^2 + (12 * Real.sin β - 4 * Real.sin α)^2)
  distance = 8 * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_polar_points_l9_955


namespace NUMINAMATH_CALUDE_xyz_value_l9_962

theorem xyz_value (x y z : ℝ) 
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 49)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 19) :
  x * y * z = 10 := by
  sorry

end NUMINAMATH_CALUDE_xyz_value_l9_962


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_3913_l9_938

theorem largest_prime_factor_of_3913 : 
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ 3913 ∧ ∀ (q : ℕ), Nat.Prime q → q ∣ 3913 → q ≤ p :=
by sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_3913_l9_938


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l9_904

theorem quadratic_inequality_range (k : ℝ) : 
  (∀ x : ℝ, k * x^2 + 2 * k * x - 3 < 0) ↔ k ∈ Set.Ioc (-3) 0 :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l9_904


namespace NUMINAMATH_CALUDE_square_of_sum_eleven_five_l9_920

theorem square_of_sum_eleven_five : 11^2 + 2*(11*5) + 5^2 = 256 := by
  sorry

end NUMINAMATH_CALUDE_square_of_sum_eleven_five_l9_920


namespace NUMINAMATH_CALUDE_triangle_ratio_theorem_l9_909

theorem triangle_ratio_theorem (A B C : Real) (hTriangle : A + B + C = PI) 
  (hCondition : 3 * Real.sin B * Real.cos C = Real.sin C * (1 - 3 * Real.cos B)) : 
  Real.sin C / Real.sin A = 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_ratio_theorem_l9_909


namespace NUMINAMATH_CALUDE_sphere_wall_thickness_l9_969

/-- Represents a hollow glass sphere floating in water -/
structure FloatingSphere where
  outer_diameter : ℝ
  specific_gravity : ℝ
  dry_surface_fraction : ℝ

/-- Calculates the wall thickness of a floating sphere -/
noncomputable def wall_thickness (sphere : FloatingSphere) : ℝ :=
  -- Implementation details omitted
  sorry

/-- Theorem stating the wall thickness of the sphere with given properties -/
theorem sphere_wall_thickness :
  let sphere : FloatingSphere := {
    outer_diameter := 16,
    specific_gravity := 2.523,
    dry_surface_fraction := 3/8
  }
  wall_thickness sphere = 0.8 := by sorry

end NUMINAMATH_CALUDE_sphere_wall_thickness_l9_969


namespace NUMINAMATH_CALUDE_total_words_eq_443_l9_978

def count_words (n : ℕ) : ℕ :=
  if n ≤ 20 ∨ n = 30 ∨ n = 40 ∨ n = 50 ∨ n = 60 ∨ n = 70 ∨ n = 80 ∨ n = 90 ∨ n = 100 ∨ n = 200 then 1
  else if n ≤ 99 then 2
  else if n ≤ 199 then 3
  else 0

def total_words : ℕ := (List.range 200).map (λ i => count_words (i + 1)) |>.sum

theorem total_words_eq_443 : total_words = 443 := by
  sorry

end NUMINAMATH_CALUDE_total_words_eq_443_l9_978


namespace NUMINAMATH_CALUDE_intersection_A_B_range_of_p_l9_979

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - x - 2 > 0}
def B : Set ℝ := {x | 3 - |x| ≥ 0}

-- Define the set C
def C (p : ℝ) : Set ℝ := {x | x^2 + 4*x + 4 - p^2 < 0}

-- Statement 1: A ∩ B = {x | -3 ≤ x < -1 or 2 < x ≤ 3}
theorem intersection_A_B : A ∩ B = {x | -3 ≤ x ∧ x < -1 ∨ 2 < x ∧ x ≤ 3} := by sorry

-- Statement 2: The range of p satisfying the given conditions is 0 < p ≤ 1
theorem range_of_p (p : ℝ) (h_p : p > 0) : 
  (C p ⊆ (A ∩ B)) ↔ (p > 0 ∧ p ≤ 1) := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_range_of_p_l9_979


namespace NUMINAMATH_CALUDE_therapy_pricing_theorem_l9_971

/-- Represents the pricing structure and total charges for a psychologist's therapy sessions. -/
structure TherapyPricing where
  first_hour : ℕ  -- Price for the first hour
  additional_hour : ℕ  -- Price for each additional hour
  total_5_hours : ℕ  -- Total charge for 5 hours of therapy

/-- Given the pricing structure, calculates the total charge for 2 hours of therapy. -/
def charge_for_2_hours (pricing : TherapyPricing) : ℕ :=
  pricing.first_hour + pricing.additional_hour

/-- Theorem stating the relationship between the pricing structure and the charge for 2 hours. -/
theorem therapy_pricing_theorem (pricing : TherapyPricing) 
  (h1 : pricing.first_hour = pricing.additional_hour + 40)
  (h2 : pricing.total_5_hours = 375)
  (h3 : pricing.first_hour + 4 * pricing.additional_hour = pricing.total_5_hours) :
  charge_for_2_hours pricing = 174 := by
  sorry

#eval charge_for_2_hours { first_hour := 107, additional_hour := 67, total_5_hours := 375 }

end NUMINAMATH_CALUDE_therapy_pricing_theorem_l9_971


namespace NUMINAMATH_CALUDE_function_bounded_l9_942

/-- The function f(x, y) = √(4 - x² - y²) is bounded between 0 and 2 -/
theorem function_bounded (x y : ℝ) (h : x^2 + y^2 ≤ 4) :
  0 ≤ Real.sqrt (4 - x^2 - y^2) ∧ Real.sqrt (4 - x^2 - y^2) ≤ 2 :=
by sorry

end NUMINAMATH_CALUDE_function_bounded_l9_942


namespace NUMINAMATH_CALUDE_calculate_expression_l9_921

theorem calculate_expression : (2200 - 2090)^2 / (144 + 25) = 64 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l9_921


namespace NUMINAMATH_CALUDE_recycling_project_points_l9_992

/-- Calculates points earned for white paper -/
def whitePoints (pounds : ℕ) : ℕ := (pounds / 6) * 2

/-- Calculates points earned for colored paper -/
def colorPoints (pounds : ℕ) : ℕ := (pounds / 8) * 3

/-- Represents a person's recycling contribution -/
structure Recycler where
  whitePaper : ℕ
  coloredPaper : ℕ

/-- Calculates total points for a recycler -/
def totalPoints (r : Recycler) : ℕ :=
  whitePoints r.whitePaper + colorPoints r.coloredPaper

theorem recycling_project_points : 
  let paige : Recycler := { whitePaper := 12, coloredPaper := 18 }
  let alex : Recycler := { whitePaper := 26, coloredPaper := 10 }
  let jordan : Recycler := { whitePaper := 30, coloredPaper := 0 }
  totalPoints paige + totalPoints alex + totalPoints jordan = 31 := by
  sorry

end NUMINAMATH_CALUDE_recycling_project_points_l9_992


namespace NUMINAMATH_CALUDE_office_persons_count_l9_963

theorem office_persons_count :
  ∀ (N : ℕ) (avg_age : ℚ) (avg_age_5 : ℚ) (avg_age_9 : ℚ) (age_15th : ℕ),
  avg_age = 15 →
  avg_age_5 = 14 →
  avg_age_9 = 16 →
  age_15th = 26 →
  N * avg_age = 5 * avg_age_5 + 9 * avg_age_9 + age_15th →
  N = 16 := by
sorry

end NUMINAMATH_CALUDE_office_persons_count_l9_963


namespace NUMINAMATH_CALUDE_race_head_start_l9_957

/-- Given two runners A and B, where A's speed is 20/15 times B's speed,
    the head start A should give B for a dead heat is 1/4 of the race length. -/
theorem race_head_start (speed_a speed_b race_length head_start : ℝ) :
  speed_a = (20 / 15) * speed_b →
  race_length > 0 →
  speed_a > 0 →
  speed_b > 0 →
  (race_length / speed_a = (race_length - head_start) / speed_b ↔ head_start = (1 / 4) * race_length) :=
by sorry

end NUMINAMATH_CALUDE_race_head_start_l9_957


namespace NUMINAMATH_CALUDE_earth_inhabitable_fraction_l9_984

theorem earth_inhabitable_fraction :
  let water_fraction : ℚ := 3/5
  let inhabitable_land_fraction : ℚ := 2/3
  let total_inhabitable_fraction : ℚ := (1 - water_fraction) * inhabitable_land_fraction
  total_inhabitable_fraction = 4/15 := by sorry

end NUMINAMATH_CALUDE_earth_inhabitable_fraction_l9_984


namespace NUMINAMATH_CALUDE_cube_difference_positive_l9_914

theorem cube_difference_positive {a b : ℝ} (h : a > b) : a^3 - b^3 > 0 := by
  sorry

end NUMINAMATH_CALUDE_cube_difference_positive_l9_914


namespace NUMINAMATH_CALUDE_hexagon_game_theorem_l9_970

/-- Represents a hexagonal grid cell -/
structure HexCell where
  x : ℤ
  y : ℤ

/-- Represents the state of a cell (empty or filled) -/
inductive CellState
  | Empty
  | Filled

/-- Represents the game state -/
structure GameState where
  grid : HexCell → CellState
  turn : ℕ

/-- Represents a player's move -/
inductive Move
  | PlaceCounters (c1 c2 : HexCell)
  | RemoveCounter (c : HexCell)

/-- Checks if two hexagonal cells are adjacent -/
def are_adjacent (c1 c2 : HexCell) : Prop :=
  sorry

/-- Checks if there are k consecutive filled cells in a line -/
def has_k_consecutive_filled (g : GameState) (k : ℕ) : Prop :=
  sorry

/-- Applies a move to the game state -/
def apply_move (g : GameState) (m : Move) : GameState :=
  sorry

/-- Checks if a move is valid according to the game rules -/
def is_valid_move (g : GameState) (m : Move) : Prop :=
  sorry

/-- Represents a winning strategy for player A -/
def winning_strategy (k : ℕ) : Prop :=
  ∃ (strategy : GameState → Move),
    ∀ (g : GameState),
      is_valid_move g (strategy g) ∧
      (∃ (n : ℕ), has_k_consecutive_filled (apply_move g (strategy g)) k)

/-- The main theorem stating that 6 is the minimum value of k for which A cannot win -/
theorem hexagon_game_theorem :
  (∀ (k : ℕ), k < 6 → winning_strategy k) ∧
  ¬(winning_strategy 6) :=
sorry

end NUMINAMATH_CALUDE_hexagon_game_theorem_l9_970


namespace NUMINAMATH_CALUDE_right_triangle_ac_length_l9_964

/-- 
Given a right triangle ABC in the x-y plane where:
- ∠B = 90°
- The slope of line segment AC is 4/3
- The length of AB is 20

Prove that the length of AC is 25.
-/
theorem right_triangle_ac_length 
  (A B C : ℝ × ℝ) 
  (right_angle : (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 0)
  (slope_ac : (C.2 - A.2) / (C.1 - A.1) = 4 / 3)
  (length_ab : Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2) = 20) :
  Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2) = 25 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_ac_length_l9_964


namespace NUMINAMATH_CALUDE_root_zero_implies_k_five_l9_926

theorem root_zero_implies_k_five (k : ℝ) : 
  (∃ x : ℝ, x ≠ 0 ∧ 8 * x^2 - (k - 1) * x - k + 5 = 0) ∧ 
  (8 * 0^2 - (k - 1) * 0 - k + 5 = 0) → 
  k = 5 := by sorry

end NUMINAMATH_CALUDE_root_zero_implies_k_five_l9_926


namespace NUMINAMATH_CALUDE_lebesgue_decomposition_l9_915

variable (E : Type) [MeasurableSpace E]
variable (μ ν : Measure E)

/-- Lebesgue decomposition theorem -/
theorem lebesgue_decomposition :
  ∃ (f : E → ℝ) (D : Set E),
    MeasurableSet D ∧
    (∀ x, 0 ≤ f x) ∧
    Measurable f ∧
    ν D = 0 ∧
    (∀ (B : Set E), MeasurableSet B →
      μ B = ∫ x in B, f x ∂ν + μ (B ∩ D)) ∧
    (∀ (g : E → ℝ) (C : Set E),
      MeasurableSet C →
      (∀ x, 0 ≤ g x) →
      Measurable g →
      ν C = 0 →
      (∀ (B : Set E), MeasurableSet B →
        μ B = ∫ x in B, g x ∂ν + μ (B ∩ C)) →
      (μ (D Δ C) = 0 ∧ ν {x | f x ≠ g x} = 0)) :=
sorry

end NUMINAMATH_CALUDE_lebesgue_decomposition_l9_915


namespace NUMINAMATH_CALUDE_triangle_angle_inequality_l9_928

theorem triangle_angle_inequality (a b c α β γ : ℝ) 
  (h1 : 0 < a ∧ 0 < b ∧ 0 < c) 
  (h2 : α > 0 ∧ β > 0 ∧ γ > 0)
  (h3 : α + β + γ = π)
  (h4 : a + b > c ∧ b + c > a ∧ c + a > b) : 
  π / 3 ≤ (a * α + b * β + c * γ) / (a + b + c) ∧ 
  (a * α + b * β + c * γ) / (a + b + c) < π / 2 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_inequality_l9_928


namespace NUMINAMATH_CALUDE_parabola_perpendicular_range_l9_910

/-- Given a parabola y² = x + 4 and points A(0,2), B(m² - 4, m), and C(x₀² - 4, x₀) where B and C are on the parabola and AB ⊥ BC, 
    the y-coordinate of C (x₀) satisfies: x₀ ≤ 2 - 2√2 or x₀ ≥ 2 + 2√2 -/
theorem parabola_perpendicular_range (m x₀ : ℝ) : 
  (m ^ 2 - 4 ≥ 0) →  -- B is on or above the x-axis
  (x₀ ^ 2 - 4 ≥ 0) →  -- C is on or above the x-axis
  ((m - 2) / (m ^ 2 - 4) * (x₀ - m) / (x₀ ^ 2 - m ^ 2) = -1) →  -- AB ⊥ BC
  (x₀ ≤ 2 - 2 * Real.sqrt 2 ∨ x₀ ≥ 2 + 2 * Real.sqrt 2) :=
by sorry


end NUMINAMATH_CALUDE_parabola_perpendicular_range_l9_910


namespace NUMINAMATH_CALUDE_expression_not_constant_l9_929

theorem expression_not_constant : 
  ¬∀ (x y : ℝ), x ≠ -1 → x ≠ 3 → y ≠ -1 → y ≠ 3 → 
  (3*x^2 + 4*x - 5) / ((x+1)*(x-3)) - (8 + x) / ((x+1)*(x-3)) = 
  (3*y^2 + 4*y - 5) / ((y+1)*(y-3)) - (8 + y) / ((y+1)*(y-3)) :=
by sorry

end NUMINAMATH_CALUDE_expression_not_constant_l9_929


namespace NUMINAMATH_CALUDE_parallelogram_to_triangle_impossibility_l9_900

theorem parallelogram_to_triangle_impossibility (a : ℝ) (h : a > 0) :
  ¬ (a + a > 2*a ∧ a + 2*a > a ∧ 2*a + a > a) :=
by sorry

end NUMINAMATH_CALUDE_parallelogram_to_triangle_impossibility_l9_900


namespace NUMINAMATH_CALUDE_cone_surface_area_l9_982

/-- The surface area of a cone with given slant height and base circumference -/
theorem cone_surface_area (slant_height : ℝ) (base_circumference : ℝ) :
  slant_height = 2 →
  base_circumference = 2 * Real.pi →
  (π * (base_circumference / (2 * π))^2) + (π * (base_circumference / (2 * π)) * slant_height) = 3 * π :=
by sorry

end NUMINAMATH_CALUDE_cone_surface_area_l9_982


namespace NUMINAMATH_CALUDE_cricket_game_solution_l9_940

def cricket_game (initial_run_rate : ℝ) (required_rate : ℝ) (total_target : ℝ) : Prop :=
  ∃ (initial_overs : ℝ),
    initial_overs > 0 ∧
    initial_overs < 50 ∧
    initial_overs + 40 = 50 ∧
    initial_run_rate * initial_overs + required_rate * 40 = total_target

theorem cricket_game_solution :
  cricket_game 3.2 5.5 252 → ∃ (initial_overs : ℝ), initial_overs = 10 := by
  sorry

end NUMINAMATH_CALUDE_cricket_game_solution_l9_940


namespace NUMINAMATH_CALUDE_prob_sum_three_is_one_over_216_l9_981

/-- The probability of rolling a specific number on a fair six-sided die -/
def prob_single_die : ℚ := 1 / 6

/-- The number of dice rolled -/
def num_dice : ℕ := 3

/-- The target sum we're aiming for -/
def target_sum : ℕ := 3

/-- The probability of rolling a sum of 3 with three fair six-sided dice -/
def prob_sum_three : ℚ := prob_single_die ^ num_dice

theorem prob_sum_three_is_one_over_216 : 
  prob_sum_three = 1 / 216 := by sorry

end NUMINAMATH_CALUDE_prob_sum_three_is_one_over_216_l9_981


namespace NUMINAMATH_CALUDE_thirty_factorial_trailing_zeros_l9_966

def trailing_zeros (n : ℕ) : ℕ := 
  (n / 5) + (n / 25)

theorem thirty_factorial_trailing_zeros : 
  trailing_zeros 30 = 7 := by
  sorry

end NUMINAMATH_CALUDE_thirty_factorial_trailing_zeros_l9_966


namespace NUMINAMATH_CALUDE_loan_interest_time_l9_975

/-- Given two loans and their interest rates, calculate the time needed to reach a specific total interest. -/
theorem loan_interest_time (loan1 loan2 rate1 rate2 total_interest : ℚ) : 
  loan1 = 1000 →
  loan2 = 1400 →
  rate1 = 3 / 100 →
  rate2 = 5 / 100 →
  total_interest = 350 →
  ∃ (time : ℚ), time * (loan1 * rate1 + loan2 * rate2) = total_interest ∧ time = 7 / 2 := by
  sorry

#check loan_interest_time

end NUMINAMATH_CALUDE_loan_interest_time_l9_975


namespace NUMINAMATH_CALUDE_food_drive_cans_l9_999

theorem food_drive_cans (rachel jaydon mark : ℕ) : 
  jaydon = 2 * rachel + 5 →
  mark = 4 * jaydon →
  rachel + jaydon + mark = 135 →
  mark = 100 := by
sorry

end NUMINAMATH_CALUDE_food_drive_cans_l9_999


namespace NUMINAMATH_CALUDE_parallel_lines_coefficient_product_l9_973

/-- Two parallel lines with a specific distance between them -/
structure ParallelLines where
  a : ℝ
  b : ℝ
  l₁ : (x y : ℝ) → a * x + 2 * y + b = 0
  l₂ : (x y : ℝ) → (a - 1) * x + y + b = 0
  parallel : ∀ (x y : ℝ), a * x + 2 * y = (a - 1) * x + y
  distance : ∃ (k : ℝ), k * (b - 0) / Real.sqrt ((a - (a - 1))^2 + (2 - 1)^2) = Real.sqrt 2 / 2 ∧ k = 1 ∨ k = -1

/-- The product of coefficients a and b for parallel lines with specific distance -/
theorem parallel_lines_coefficient_product (pl : ParallelLines) : pl.a * pl.b = 4 ∨ pl.a * pl.b = -4 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_coefficient_product_l9_973


namespace NUMINAMATH_CALUDE_initial_packages_l9_907

theorem initial_packages (cupcakes_per_package : ℕ) (eaten_cupcakes : ℕ) (remaining_cupcakes : ℕ) :
  cupcakes_per_package = 4 →
  eaten_cupcakes = 5 →
  remaining_cupcakes = 7 →
  (eaten_cupcakes + remaining_cupcakes) / cupcakes_per_package = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_initial_packages_l9_907


namespace NUMINAMATH_CALUDE_expression_value_l9_988

theorem expression_value : 
  (2020^4 - 3 * 2020^3 * 2021 + 4 * 2020 * 2021^3 - 2021^4 + 1) / (2020 * 2021) = 4096046 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l9_988


namespace NUMINAMATH_CALUDE_set_membership_properties_l9_943

theorem set_membership_properties (M P : Set α) (h_nonempty : M.Nonempty) 
  (h_not_subset : ¬(M ⊆ P)) : 
  (∃ x, x ∈ M ∧ x ∉ P) ∧ (∃ y, y ∈ M ∧ y ∈ P) := by
  sorry

end NUMINAMATH_CALUDE_set_membership_properties_l9_943


namespace NUMINAMATH_CALUDE_triangle_side_length_l9_931

/-- Prove that in a triangle ABC with specific properties, the length of side a is 3√2 -/
theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) : 
  let f := λ x : ℝ => (Real.cos x, -1) • (Real.cos x + Real.sqrt 3 * Real.sin x, -3/2) - 2
  (f A = 1/2) →
  (2 * a = b + c) →
  (b * c / 2 = 9) →
  (a = 3 * Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l9_931


namespace NUMINAMATH_CALUDE_min_value_x_plus_3y_l9_958

theorem min_value_x_plus_3y (x y : ℝ) (hx : x > 0) (hy : y > 0)
  (h : 1 / (x + 1) + 1 / (y + 1) = 1 / 4) :
  x + 3 * y ≥ 5 + 4 * Real.sqrt 3 ∧
  ∃ x₀ y₀ : ℝ, x₀ > 0 ∧ y₀ > 0 ∧
    1 / (x₀ + 1) + 1 / (y₀ + 1) = 1 / 4 ∧
    x₀ + 3 * y₀ = 5 + 4 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_x_plus_3y_l9_958


namespace NUMINAMATH_CALUDE_cricket_team_age_difference_l9_974

theorem cricket_team_age_difference (team_size : ℕ) (captain_age : ℕ) (keeper_age_diff : ℕ) (team_avg_age : ℚ) : 
  team_size = 11 →
  captain_age = 27 →
  keeper_age_diff = 3 →
  team_avg_age = 24 →
  let keeper_age := captain_age + keeper_age_diff
  let total_age := team_avg_age * team_size
  let remaining_players := team_size - 2
  let remaining_age := total_age - (captain_age + keeper_age)
  let remaining_avg_age := remaining_age / remaining_players
  remaining_avg_age = team_avg_age - 1 := by
  sorry

end NUMINAMATH_CALUDE_cricket_team_age_difference_l9_974


namespace NUMINAMATH_CALUDE_prime_power_composite_and_divisor_l9_913

theorem prime_power_composite_and_divisor (p : ℕ) (hp : p.Prime) (hp_gt_3 : p > 3) :
  let q := (4^p - 1) / 3
  (∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ q = a * b) ∧ (q ∣ 2^(q - 1) - 1) := by
  sorry

end NUMINAMATH_CALUDE_prime_power_composite_and_divisor_l9_913


namespace NUMINAMATH_CALUDE_optimal_strategy_l9_961

/-- Represents the cosmetics store problem -/
structure CosmeticsStore where
  m : ℝ  -- Purchase price of cosmetic A
  n : ℝ  -- Purchase price of cosmetic B
  total_items : ℕ  -- Total number of items to purchase

/-- Conditions for the cosmetics store problem -/
def valid_store (store : CosmeticsStore) : Prop :=
  3 * store.m + 4 * store.n = 620 ∧
  5 * store.m + 3 * store.n = 740 ∧
  store.total_items = 200

/-- Calculate the profit for a given purchase strategy -/
def profit (store : CosmeticsStore) (items_a : ℕ) : ℝ :=
  (250 - store.m) * items_a + (200 - store.n) * (store.total_items - items_a)

/-- Check if a purchase strategy is valid -/
def valid_strategy (store : CosmeticsStore) (items_a : ℕ) : Prop :=
  store.m * items_a + store.n * (store.total_items - items_a) ≤ 18100 ∧
  profit store items_a ≥ 27000

/-- Theorem stating the optimal strategy and maximum profit -/
theorem optimal_strategy (store : CosmeticsStore) :
  valid_store store →
  (∃ (items_a : ℕ), valid_strategy store items_a) →
  (∃ (max_items_a : ℕ), 
    valid_strategy store max_items_a ∧
    ∀ (items_a : ℕ), valid_strategy store items_a → 
      profit store max_items_a ≥ profit store items_a) ∧
  (let max_items_a := 105
   profit store max_items_a = 27150 ∧
   valid_strategy store max_items_a ∧
   ∀ (items_a : ℕ), valid_strategy store items_a → 
     profit store max_items_a ≥ profit store items_a) :=
by sorry


end NUMINAMATH_CALUDE_optimal_strategy_l9_961


namespace NUMINAMATH_CALUDE_oranges_per_bag_l9_949

theorem oranges_per_bag (total_oranges : ℕ) (num_bags : ℕ) (h1 : total_oranges = 1035) (h2 : num_bags = 45) (h3 : total_oranges % num_bags = 0) : 
  total_oranges / num_bags = 23 := by
sorry

end NUMINAMATH_CALUDE_oranges_per_bag_l9_949


namespace NUMINAMATH_CALUDE_quadratic_always_positive_l9_985

theorem quadratic_always_positive : ∀ x : ℝ, x^2 - x + 1 > 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_always_positive_l9_985


namespace NUMINAMATH_CALUDE_age_difference_theorem_l9_911

/-- Represents a two-digit number --/
structure TwoDigitNumber where
  tens : Nat
  ones : Nat
  h_tens : tens ≤ 9
  h_ones : ones ≤ 9
  h_not_zero : tens ≠ 0

/-- The value of a two-digit number --/
def TwoDigitNumber.value (n : TwoDigitNumber) : Nat :=
  10 * n.tens + n.ones

theorem age_difference_theorem (jack bill : TwoDigitNumber)
    (h_reversed : jack.tens = bill.ones ∧ jack.ones = bill.tens)
    (h_future : jack.value + 6 = 3 * (bill.value + 6)) :
    jack.value - bill.value = 36 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_theorem_l9_911


namespace NUMINAMATH_CALUDE_fill_time_with_leak_l9_960

/-- Time taken to fill a tank with two pipes and a leak -/
theorem fill_time_with_leak (pipe1_time pipe2_time : ℝ) (leak_fraction : ℝ) : 
  pipe1_time = 20 →
  pipe2_time = 30 →
  leak_fraction = 1/3 →
  (1 / ((1 / pipe1_time + 1 / pipe2_time) * (1 - leak_fraction))) = 18 :=
by sorry

end NUMINAMATH_CALUDE_fill_time_with_leak_l9_960


namespace NUMINAMATH_CALUDE_parabola_vertex_c_value_l9_927

/-- Given a parabola of the form y = 2x^2 + c with vertex at (0,1), prove that c = 1 -/
theorem parabola_vertex_c_value (c : ℝ) : 
  (∀ x y : ℝ, y = 2 * x^2 + c) →   -- Parabola equation
  (0, 1) = (0, 2 * 0^2 + c) →      -- Vertex at (0,1)
  c = 1 := by sorry

end NUMINAMATH_CALUDE_parabola_vertex_c_value_l9_927


namespace NUMINAMATH_CALUDE_geometric_sequence_solution_l9_952

theorem geometric_sequence_solution (a : ℝ) :
  (∃ r : ℝ, r ≠ 0 ∧ (2*a + 2) = a * r ∧ (3*a + 3) = (2*a + 2) * r) → a = -4 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_solution_l9_952


namespace NUMINAMATH_CALUDE_largest_angle_in_special_triangle_l9_933

theorem largest_angle_in_special_triangle :
  ∀ (a b c : ℝ),
    a > 0 ∧ b > 0 ∧ c > 0 →
    b = (3/2) * a →
    c = 2 * a →
    a + b + c = 180 →
    max a (max b c) = 80 :=
by sorry

end NUMINAMATH_CALUDE_largest_angle_in_special_triangle_l9_933


namespace NUMINAMATH_CALUDE_second_half_wants_fifteen_l9_991

/-- Represents the BBQ scenario with given conditions -/
structure BBQScenario where
  cooking_time_per_side : ℕ  -- Time to cook one side of a burger
  grill_capacity : ℕ         -- Number of burgers that can fit on the grill
  total_guests : ℕ           -- Total number of guests
  first_half_burgers : ℕ     -- Number of burgers each guest in the first half wants
  total_cooking_time : ℕ     -- Total time taken to cook all burgers

/-- Calculates the number of burgers wanted by the second half of guests -/
def second_half_burgers (scenario : BBQScenario) : ℕ :=
  let total_burgers := scenario.total_cooking_time / (2 * scenario.cooking_time_per_side) * scenario.grill_capacity
  let first_half_total := scenario.total_guests / 2 * scenario.first_half_burgers
  total_burgers - first_half_total

/-- Theorem stating that the second half of guests want 15 burgers -/
theorem second_half_wants_fifteen (scenario : BBQScenario) 
  (h1 : scenario.cooking_time_per_side = 4)
  (h2 : scenario.grill_capacity = 5)
  (h3 : scenario.total_guests = 30)
  (h4 : scenario.first_half_burgers = 2)
  (h5 : scenario.total_cooking_time = 72) : 
  second_half_burgers scenario = 15 := by
  sorry


end NUMINAMATH_CALUDE_second_half_wants_fifteen_l9_991


namespace NUMINAMATH_CALUDE_k_equals_nine_l9_941

/-- Two circles centered at the origin with specific points and distances -/
structure TwoCircles where
  -- Radius of the larger circle
  R : ℝ
  -- Radius of the smaller circle
  r : ℝ
  -- Point P on the larger circle
  P : ℝ × ℝ
  -- Point S on the smaller circle
  S : ℝ × ℝ
  -- Distance QR
  QR : ℝ
  -- Conditions
  P_on_larger : P.1^2 + P.2^2 = R^2
  S_on_smaller : S.1^2 + S.2^2 = r^2
  P_coords : P = (5, 12)
  S_coords : S = (0, S.2)
  QR_value : QR = 4

/-- The theorem stating that k (the y-coordinate of S) equals 9 -/
theorem k_equals_nine (c : TwoCircles) : c.S.2 = 9 := by
  sorry

end NUMINAMATH_CALUDE_k_equals_nine_l9_941


namespace NUMINAMATH_CALUDE_rectangle_ratio_l9_903

theorem rectangle_ratio (s : ℝ) (x y : ℝ) (h1 : s > 0) (h2 : x > 0) (h3 : y > 0)
  (h4 : s + 2*y = 3*s) -- outer square side length
  (h5 : x + s = 3*s) -- outer square side length
  (h6 : (3*s)^2 = 9*s^2) -- area relation
  : x / y = 2 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_ratio_l9_903


namespace NUMINAMATH_CALUDE_distance_covered_l9_967

/-- Proves that the total distance covered is 16 km given the specified conditions -/
theorem distance_covered (walking_speed running_speed : ℝ) (total_time : ℝ) :
  walking_speed = 4 →
  running_speed = 8 →
  total_time = 3 →
  ∃ (distance : ℝ),
    distance / walking_speed / 2 + distance / running_speed / 2 = total_time ∧
    distance = 16 := by
  sorry

end NUMINAMATH_CALUDE_distance_covered_l9_967


namespace NUMINAMATH_CALUDE_ferris_wheel_theorem_l9_998

/-- The number of people who can ride a Ferris wheel at the same time -/
def ferris_wheel_capacity (seats : ℕ) (people_per_seat : ℕ) : ℕ :=
  seats * people_per_seat

/-- Theorem: The capacity of a Ferris wheel with 2 seats and 2 people per seat is 4 -/
theorem ferris_wheel_theorem : ferris_wheel_capacity 2 2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_ferris_wheel_theorem_l9_998


namespace NUMINAMATH_CALUDE_opposite_of_2023_l9_936

theorem opposite_of_2023 : 
  ∀ x : ℤ, x + 2023 = 0 ↔ x = -2023 := by
sorry

end NUMINAMATH_CALUDE_opposite_of_2023_l9_936


namespace NUMINAMATH_CALUDE_infinite_solutions_l9_976

theorem infinite_solutions (a : ℝ) : 
  (a = 5) → (∀ y : ℝ, 3 * (5 + a * y) = 15 * y + 9) :=
by sorry

end NUMINAMATH_CALUDE_infinite_solutions_l9_976


namespace NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l9_917

theorem quadratic_two_distinct_roots (k : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ x^2 - 2*x + k = 0 ∧ y^2 - 2*y + k = 0) → k < 1 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l9_917


namespace NUMINAMATH_CALUDE_selection_problem_l9_986

theorem selection_problem (total : Nat) (translation_capable : Nat) (software_capable : Nat) 
  (both_capable : Nat) (to_select : Nat) (for_translation : Nat) (for_software : Nat) :
  total = 8 →
  translation_capable = 5 →
  software_capable = 4 →
  both_capable = 1 →
  to_select = 5 →
  for_translation = 3 →
  for_software = 2 →
  (Nat.choose (translation_capable - 1) for_translation * 
   Nat.choose (software_capable - 1) for_software) +
  (Nat.choose (translation_capable - 1) (for_translation - 1) * 
   Nat.choose software_capable for_software) +
  (Nat.choose translation_capable for_translation * 
   Nat.choose (software_capable - 1) (for_software - 1)) = 42 := by
  sorry

#check selection_problem

end NUMINAMATH_CALUDE_selection_problem_l9_986


namespace NUMINAMATH_CALUDE_log_product_simplification_l9_923

theorem log_product_simplification : 
  Real.log 9 / Real.log 8 * (Real.log 32 / Real.log 27) = 10 / 9 := by
  sorry

end NUMINAMATH_CALUDE_log_product_simplification_l9_923


namespace NUMINAMATH_CALUDE_quadratic_integer_root_l9_997

theorem quadratic_integer_root (b : ℤ) : 
  (∃ x : ℤ, x^2 + 4*x + b = 0) ↔ (b = -12 ∨ b = -5 ∨ b = 3 ∨ b = 4) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_integer_root_l9_997


namespace NUMINAMATH_CALUDE_rose_cost_l9_965

/-- Proves that the cost of each rose is $5 given the conditions of Nadia's flower purchase. -/
theorem rose_cost (num_roses : ℕ) (num_lilies : ℚ) (total_cost : ℚ) : 
  num_roses = 20 →
  num_lilies = 3/4 * num_roses →
  total_cost = 250 →
  ∃ (rose_cost : ℚ), 
    rose_cost * num_roses + (2 * rose_cost) * num_lilies = total_cost ∧
    rose_cost = 5 := by
  sorry

end NUMINAMATH_CALUDE_rose_cost_l9_965


namespace NUMINAMATH_CALUDE_solution_set_of_quadratic_inequality_l9_944

-- Define the quadratic function
def f (x : ℝ) : ℝ := x^2 - 3*x + 2

-- State the theorem
theorem solution_set_of_quadratic_inequality :
  {x : ℝ | f x < 0} = {x : ℝ | 1 < x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_solution_set_of_quadratic_inequality_l9_944


namespace NUMINAMATH_CALUDE_units_digit_of_N_l9_956

def N : ℕ := 3^1001 + 7^1002 + 13^1003

theorem units_digit_of_N (n : ℕ) (h : n = N) : n % 10 = 9 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_N_l9_956


namespace NUMINAMATH_CALUDE_macy_running_goal_l9_945

/-- Calculates the remaining miles to run given a weekly goal, daily run distance, and number of days run. -/
def remaining_miles (weekly_goal : ℕ) (daily_run : ℕ) (days_run : ℕ) : ℕ :=
  weekly_goal - daily_run * days_run

/-- Proves that given a weekly goal of 24 miles and a daily run of 3 miles, the remaining distance to run after 6 days is 6 miles. -/
theorem macy_running_goal :
  remaining_miles 24 3 6 = 6 := by
  sorry

#eval remaining_miles 24 3 6

end NUMINAMATH_CALUDE_macy_running_goal_l9_945


namespace NUMINAMATH_CALUDE_ferry_problem_l9_930

/-- The ferry problem -/
theorem ferry_problem (speed_p speed_q : ℝ) (time_p : ℝ) (distance_q : ℝ) :
  speed_p = 8 →
  time_p = 3 →
  speed_q = speed_p + 4 →
  distance_q = 2 * speed_p * time_p →
  distance_q / speed_q - time_p = 1 := by
  sorry

end NUMINAMATH_CALUDE_ferry_problem_l9_930


namespace NUMINAMATH_CALUDE_calculate_fraction_l9_954

/-- A function satisfying the given property for all real numbers -/
def special_function (f : ℝ → ℝ) : Prop :=
  ∀ a b : ℝ, 2 * b^2 * f a = a^2 * f b

/-- The main theorem -/
theorem calculate_fraction (f : ℝ → ℝ) (h1 : special_function f) (h2 : f 6 ≠ 0) :
  (f 7 - f 3) / f 6 = 5 / 9 := by
  sorry

end NUMINAMATH_CALUDE_calculate_fraction_l9_954
