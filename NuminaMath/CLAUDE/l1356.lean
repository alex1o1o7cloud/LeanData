import Mathlib

namespace NUMINAMATH_CALUDE_tan_alpha_two_implies_fraction_l1356_135651

theorem tan_alpha_two_implies_fraction (α : Real) (h : Real.tan α = 2) :
  (3 * Real.sin α - Real.cos α) / (2 * Real.sin α + 3 * Real.cos α) = 5 / 7 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_two_implies_fraction_l1356_135651


namespace NUMINAMATH_CALUDE_louise_boxes_l1356_135636

/-- The number of pencils each box can hold -/
def pencils_per_box : ℕ := 20

/-- The number of red pencils Louise has -/
def red_pencils : ℕ := 20

/-- The number of blue pencils Louise has -/
def blue_pencils : ℕ := 2 * red_pencils

/-- The number of yellow pencils Louise has -/
def yellow_pencils : ℕ := 40

/-- The number of green pencils Louise has -/
def green_pencils : ℕ := red_pencils + blue_pencils

/-- The total number of pencils Louise has -/
def total_pencils : ℕ := red_pencils + blue_pencils + yellow_pencils + green_pencils

/-- The number of boxes Louise needs -/
def boxes_needed : ℕ := total_pencils / pencils_per_box

theorem louise_boxes : boxes_needed = 8 := by
  sorry

end NUMINAMATH_CALUDE_louise_boxes_l1356_135636


namespace NUMINAMATH_CALUDE_max_value_of_d_l1356_135655

def a (n : ℕ) : ℕ := n^3 + 4

def d (n : ℕ) : ℕ := Nat.gcd (a n) (a (n + 1))

theorem max_value_of_d : ∃ (k : ℕ), d k = 433 ∧ ∀ (n : ℕ), d n ≤ 433 := by sorry

end NUMINAMATH_CALUDE_max_value_of_d_l1356_135655


namespace NUMINAMATH_CALUDE_blue_marbles_count_l1356_135665

theorem blue_marbles_count (blue yellow : ℕ) : 
  (blue : ℚ) / yellow = 8 / 5 →
  (blue - 12 : ℚ) / (yellow + 21) = 1 / 3 →
  blue = 24 := by
sorry

end NUMINAMATH_CALUDE_blue_marbles_count_l1356_135665


namespace NUMINAMATH_CALUDE_exists_ratio_preserving_quadrilateral_l1356_135608

/-- A convex quadrilateral -/
structure ConvexQuadrilateral where
  sides : Fin 4 → ℝ
  angles : Fin 4 → ℝ
  sides_positive : ∀ i, sides i > 0
  angles_positive : ∀ i, angles i > 0
  sides_convex : ∀ i, sides i < sides ((i + 1) % 4) + sides ((i + 2) % 4) + sides ((i + 3) % 4)
  angles_convex : ∀ i, angles i < angles ((i + 1) % 4) + angles ((i + 2) % 4) + angles ((i + 3) % 4)
  angle_sum : angles 0 + angles 1 + angles 2 + angles 3 = 2 * Real.pi

/-- The existence of a quadrilateral with side-angle ratio preservation -/
theorem exists_ratio_preserving_quadrilateral (q : ConvexQuadrilateral) :
  ∃ q' : ConvexQuadrilateral,
    ∀ i : Fin 4, (q'.sides i) / (q'.sides ((i + 1) % 4)) = (q.angles i) / (q.angles ((i + 1) % 4)) :=
sorry

end NUMINAMATH_CALUDE_exists_ratio_preserving_quadrilateral_l1356_135608


namespace NUMINAMATH_CALUDE_final_number_is_100_l1356_135606

def board_numbers : List ℚ := List.map (λ i => 1 / i) (List.range 100)

def combine (a b : ℚ) : ℚ := a * b + a + b

theorem final_number_is_100 (numbers : List ℚ) (h : numbers = board_numbers) :
  (numbers.foldl combine 0 : ℚ) = 100 := by
  sorry

end NUMINAMATH_CALUDE_final_number_is_100_l1356_135606


namespace NUMINAMATH_CALUDE_circle_radius_increment_l1356_135691

theorem circle_radius_increment (c₁ c₂ : ℝ) (h₁ : c₁ = 50) (h₂ : c₂ = 60) :
  c₂ / (2 * Real.pi) - c₁ / (2 * Real.pi) = 5 / Real.pi := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_increment_l1356_135691


namespace NUMINAMATH_CALUDE_lcm_150_294_l1356_135674

theorem lcm_150_294 : Nat.lcm 150 294 = 7350 := by
  sorry

end NUMINAMATH_CALUDE_lcm_150_294_l1356_135674


namespace NUMINAMATH_CALUDE_part_one_part_two_l1356_135685

-- Define the sets A and B
def A : Set ℝ := {x | x < -1 ∨ x > 5}
def B (a : ℝ) : Set ℝ := {x | a < x ∧ x < a + 4}

-- Part 1
theorem part_one :
  (A ∩ B 2 = {x | 5 < x ∧ x < 6}) ∧
  (Set.univ \ A = {x | -1 ≤ x ∧ x ≤ 5}) :=
sorry

-- Part 2
theorem part_two (a : ℝ) :
  B a ⊆ (Set.univ \ A) ↔ a ∈ Set.Iic 3 ∪ Set.Ici 5 :=
sorry

end NUMINAMATH_CALUDE_part_one_part_two_l1356_135685


namespace NUMINAMATH_CALUDE_summer_salutations_l1356_135654

/-- The number of sun salutation yoga poses Summer performs on weekdays -/
def poses_per_day : ℕ := 5

/-- The number of weekdays in a week -/
def weekdays_per_week : ℕ := 5

/-- The number of weeks in a year -/
def weeks_per_year : ℕ := 52

/-- The total number of sun salutations Summer performs in a year -/
def total_salutations : ℕ := poses_per_day * weekdays_per_week * weeks_per_year

/-- Theorem stating that Summer performs 1300 sun salutations throughout an entire year -/
theorem summer_salutations : total_salutations = 1300 := by
  sorry

end NUMINAMATH_CALUDE_summer_salutations_l1356_135654


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l1356_135615

/-- The perimeter of a rectangle with area 500 cm² and one side 25 cm is 90 cm. -/
theorem rectangle_perimeter (a b : ℝ) (h_area : a * b = 500) (h_side : a = 25) : 
  2 * (a + b) = 90 := by
sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l1356_135615


namespace NUMINAMATH_CALUDE_fraction_equality_l1356_135642

theorem fraction_equality (x y : ℝ) (h : (1/x + 1/y) / (1/x + 2/y) = 4) :
  (x + y) / (x + 2*y) = 4/11 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1356_135642


namespace NUMINAMATH_CALUDE_expand_expression_l1356_135612

theorem expand_expression (x : ℝ) : 2 * (x + 3) * (x^2 - 2*x + 7) = 2*x^3 + 2*x^2 + 2*x + 42 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l1356_135612


namespace NUMINAMATH_CALUDE_pascals_triangle_20th_row_5th_number_l1356_135692

theorem pascals_triangle_20th_row_5th_number : 
  let n : ℕ := 20  -- The row number (0-indexed)
  let k : ℕ := 4   -- The position in the row (0-indexed)
  Nat.choose n k = 4845 := by
sorry

end NUMINAMATH_CALUDE_pascals_triangle_20th_row_5th_number_l1356_135692


namespace NUMINAMATH_CALUDE_bank_deposit_theorem_l1356_135671

def initial_deposit : ℝ := 20000
def term : ℝ := 2
def annual_interest_rate : ℝ := 0.0325

theorem bank_deposit_theorem :
  initial_deposit * (1 + annual_interest_rate * term) = 21300 := by
  sorry

end NUMINAMATH_CALUDE_bank_deposit_theorem_l1356_135671


namespace NUMINAMATH_CALUDE_ball_count_l1356_135677

theorem ball_count (white green yellow red purple : ℕ)
  (h1 : white = 50)
  (h2 : green = 30)
  (h3 : yellow = 10)
  (h4 : red = 7)
  (h5 : purple = 3)
  (h6 : (white + green + yellow : ℚ) / (white + green + yellow + red + purple) = 0.9) :
  white + green + yellow + red + purple = 100 := by
sorry

end NUMINAMATH_CALUDE_ball_count_l1356_135677


namespace NUMINAMATH_CALUDE_largest_prime_2015_digits_square_minus_one_div_15_l1356_135669

/-- The largest prime with 2015 digits -/
def p : ℕ := sorry

/-- p is prime -/
axiom p_prime : Nat.Prime p

/-- p has 2015 digits -/
axiom p_digits : 10^2014 ≤ p ∧ p < 10^2015

/-- p is the largest such prime -/
axiom p_largest : ∀ q : ℕ, Nat.Prime q → 10^2014 ≤ q ∧ q < 10^2015 → q ≤ p

theorem largest_prime_2015_digits_square_minus_one_div_15 : 15 ∣ (p^2 - 1) := by
  sorry

end NUMINAMATH_CALUDE_largest_prime_2015_digits_square_minus_one_div_15_l1356_135669


namespace NUMINAMATH_CALUDE_lawyer_fee_ratio_l1356_135602

/-- Lawyer fee calculation and payment ratio problem --/
theorem lawyer_fee_ratio :
  let upfront_fee : ℕ := 1000
  let hourly_rate : ℕ := 100
  let court_hours : ℕ := 50
  let prep_hours : ℕ := 2 * court_hours
  let total_fee : ℕ := upfront_fee + hourly_rate * (court_hours + prep_hours)
  let john_payment : ℕ := 8000
  let brother_payment : ℕ := total_fee - john_payment
  brother_payment * 2 = total_fee := by sorry

end NUMINAMATH_CALUDE_lawyer_fee_ratio_l1356_135602


namespace NUMINAMATH_CALUDE_debate_team_groups_l1356_135621

theorem debate_team_groups (boys : ℕ) (girls : ℕ) (group_size : ℕ) : 
  boys = 31 → girls = 32 → group_size = 9 → 
  (boys + girls) / group_size = 7 ∧ (boys + girls) % group_size = 0 := by
  sorry

end NUMINAMATH_CALUDE_debate_team_groups_l1356_135621


namespace NUMINAMATH_CALUDE_smallest_three_digit_multiple_of_17_l1356_135605

theorem smallest_three_digit_multiple_of_17 : 
  ∀ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n → 102 ≤ n :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_three_digit_multiple_of_17_l1356_135605


namespace NUMINAMATH_CALUDE_unique_solution_condition_l1356_135696

/-- A system of equations has exactly one solution if and only if a = 2 and b = -1 -/
theorem unique_solution_condition (a b : ℝ) : 
  (∃! x y, y = x^2 ∧ y = 2*x + b) ↔ (a = 2 ∧ b = -1) :=
sorry

end NUMINAMATH_CALUDE_unique_solution_condition_l1356_135696


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1356_135644

def arithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmeticSequence a →
  (a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 = 420) →
  (a 2 + a 10 = 120) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1356_135644


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1356_135632

def A : Set ℤ := {0, 1, 2, 8}
def B : Set ℤ := {-1, 1, 6, 8}

theorem intersection_of_A_and_B : A ∩ B = {1, 8} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1356_135632


namespace NUMINAMATH_CALUDE_remainder_783245_div_7_l1356_135618

theorem remainder_783245_div_7 : 783245 % 7 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_783245_div_7_l1356_135618


namespace NUMINAMATH_CALUDE_arrangement_count_l1356_135646

/-- The number of ways to arrange 3 male and 3 female students in a row with exactly two female students adjacent -/
def num_arrangements : ℕ := sorry

/-- The total number of students -/
def total_students : ℕ := 6

/-- The number of male students -/
def num_male : ℕ := 3

/-- The number of female students -/
def num_female : ℕ := 3

theorem arrangement_count :
  (total_students = num_male + num_female) →
  (num_arrangements = 432) := by sorry

end NUMINAMATH_CALUDE_arrangement_count_l1356_135646


namespace NUMINAMATH_CALUDE_maggot_feeding_problem_l1356_135603

/-- The number of maggots attempted to be fed in the first feeding -/
def first_feeding : ℕ := 15

/-- The total number of maggots served -/
def total_maggots : ℕ := 20

/-- The number of maggots eaten in the first feeding -/
def eaten_first : ℕ := 1

/-- The number of maggots eaten in the second feeding -/
def eaten_second : ℕ := 3

theorem maggot_feeding_problem :
  first_feeding + eaten_first + eaten_second = total_maggots :=
by sorry

end NUMINAMATH_CALUDE_maggot_feeding_problem_l1356_135603


namespace NUMINAMATH_CALUDE_sum_of_coefficients_cube_expansion_l1356_135614

theorem sum_of_coefficients_cube_expansion : 
  ∃ (a b c d e : ℚ), 
    (∀ x, 1000 * x^3 + 27 = (a*x + b) * (c*x^2 + d*x + e)) ∧
    a + b + c + d + e = 92 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_cube_expansion_l1356_135614


namespace NUMINAMATH_CALUDE_trigonometric_identity_l1356_135662

theorem trigonometric_identity (α : ℝ) : 
  Real.sin (10 * α) * Real.sin (8 * α) + Real.sin (8 * α) * Real.sin (6 * α) - Real.sin (4 * α) * Real.sin (2 * α) = 
  2 * Real.cos (2 * α) * Real.sin (6 * α) * Real.sin (10 * α) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l1356_135662


namespace NUMINAMATH_CALUDE_equation_solutions_l1356_135607

def equation (x : ℝ) : Prop :=
  x ≠ 2/3 ∧ x ≠ -3 ∧ (8*x + 3) / (3*x^2 + 8*x - 6) = 3*x / (3*x - 2)

theorem equation_solutions :
  ∀ x : ℝ, equation x ↔ (x = 1 ∨ x = -1) :=
sorry

end NUMINAMATH_CALUDE_equation_solutions_l1356_135607


namespace NUMINAMATH_CALUDE_fraction_sum_product_l1356_135657

theorem fraction_sum_product : (3 / 5 + 4 / 15) * (2 / 3) = 26 / 45 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_product_l1356_135657


namespace NUMINAMATH_CALUDE_jennys_money_l1356_135694

theorem jennys_money (original : ℚ) : 
  (original - (3/7 * original + 2/5 * original) = 24) → 
  (1/2 * original = 70) := by
sorry

end NUMINAMATH_CALUDE_jennys_money_l1356_135694


namespace NUMINAMATH_CALUDE_probability_of_negative_product_l1356_135652

def set_m : Finset Int := {-7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4}
def set_t : Finset Int := {-4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7}

def negative_product_pairs : Finset (Int × Int) :=
  (set_m.filter (λ x => x < 0) ×ˢ set_t.filter (λ y => y > 0)) ∪
  (set_m.filter (λ x => x > 0) ×ˢ set_t.filter (λ y => y < 0))

theorem probability_of_negative_product :
  (negative_product_pairs.card : ℚ) / ((set_m.card * set_t.card) : ℚ) = 65 / 144 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_negative_product_l1356_135652


namespace NUMINAMATH_CALUDE_quadratic_symmetry_l1356_135676

theorem quadratic_symmetry (a : ℝ) :
  (∃ (a : ℝ), 4 = a * (-2)^2) → (∃ (a : ℝ), 4 = a * 2^2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_symmetry_l1356_135676


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1356_135683

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a → (a 3 + a 4 + a 5 = 12) → (a 1 + a 7 = 8) :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1356_135683


namespace NUMINAMATH_CALUDE_coefficient_a3b3_value_l1356_135689

/-- The coefficient of a^3b^3 in (a+b)^6(c+1/c)^8 -/
def coefficient_a3b3 (a b c : ℝ) : ℕ :=
  (Nat.choose 6 3) * (Nat.choose 8 4)

theorem coefficient_a3b3_value :
  ∀ a b c : ℝ, coefficient_a3b3 a b c = 1400 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_a3b3_value_l1356_135689


namespace NUMINAMATH_CALUDE_min_cars_in_group_l1356_135610

theorem min_cars_in_group (total : ℕ) 
  (no_ac : ℕ) 
  (racing_stripes : ℕ) 
  (ac_no_stripes : ℕ) : 
  no_ac = 47 →
  racing_stripes ≥ 53 →
  ac_no_stripes ≤ 47 →
  total ≥ 100 :=
by
  sorry

end NUMINAMATH_CALUDE_min_cars_in_group_l1356_135610


namespace NUMINAMATH_CALUDE_largest_whole_number_times_eight_less_than_150_l1356_135661

theorem largest_whole_number_times_eight_less_than_150 :
  ∃ y : ℕ, y = 18 ∧ 8 * y < 150 ∧ ∀ z : ℕ, z > y → 8 * z ≥ 150 :=
by sorry

end NUMINAMATH_CALUDE_largest_whole_number_times_eight_less_than_150_l1356_135661


namespace NUMINAMATH_CALUDE_secret_spread_theorem_l1356_135634

/-- The number of students who know the secret on day n -/
def students_knowing_secret (n : ℕ) : ℕ := (3^(n+1) - 1) / 2

/-- The day of the week given a number of days since Monday -/
def day_of_week (n : ℕ) : String :=
  match n % 7 with
  | 0 => "Sunday"
  | 1 => "Monday"
  | 2 => "Tuesday"
  | 3 => "Wednesday"
  | 4 => "Thursday"
  | 5 => "Friday"
  | _ => "Saturday"

theorem secret_spread_theorem : 
  ∃ n : ℕ, students_knowing_secret n = 2186 ∧ day_of_week n = "Sunday" :=
by sorry

end NUMINAMATH_CALUDE_secret_spread_theorem_l1356_135634


namespace NUMINAMATH_CALUDE_y_can_take_any_real_value_l1356_135613

-- Define the equation
def equation (x y : ℝ) : Prop := 2 * x * abs x + y^2 = 1

-- Theorem statement
theorem y_can_take_any_real_value :
  ∀ y : ℝ, ∃ x : ℝ, equation x y :=
by
  sorry

end NUMINAMATH_CALUDE_y_can_take_any_real_value_l1356_135613


namespace NUMINAMATH_CALUDE_vans_needed_l1356_135622

theorem vans_needed (total_people : ℕ) (van_capacity : ℕ) (h1 : total_people = 35) (h2 : van_capacity = 4) :
  ↑⌈(total_people : ℚ) / van_capacity⌉ = 9 := by
  sorry

end NUMINAMATH_CALUDE_vans_needed_l1356_135622


namespace NUMINAMATH_CALUDE_parabola_theorem_l1356_135648

-- Define a parabola type
structure Parabola where
  equation : ℝ → ℝ → Prop
  directrix : ℝ → ℝ → Prop

-- Define the conditions for the parabola
def parabola_conditions (p : Parabola) : Prop :=
  -- Vertex at origin
  p.equation 0 0 ∧
  -- Passes through (-3, 2)
  p.equation (-3) 2 ∧
  -- Axis of symmetry along coordinate axis (implied by the equation forms)
  (∃ (a : ℝ), ∀ (x y : ℝ), p.equation x y ↔ y^2 = a * x) ∨
  (∃ (b : ℝ), ∀ (x y : ℝ), p.equation x y ↔ x^2 = b * y)

-- Define the possible equations and directrices
def parabola1 : Parabola :=
  { equation := λ x y => y^2 = -4/3 * x
    directrix := λ x y => x = 1/3 }

def parabola2 : Parabola :=
  { equation := λ x y => x^2 = 9/2 * y
    directrix := λ x y => y = -9/8 }

-- Theorem statement
theorem parabola_theorem :
  ∀ (p : Parabola), parabola_conditions p →
    (p = parabola1 ∨ p = parabola2) :=
sorry

end NUMINAMATH_CALUDE_parabola_theorem_l1356_135648


namespace NUMINAMATH_CALUDE_unique_denomination_l1356_135639

/-- Given unlimited supply of stamps of denominations 7, n, and n+2 cents,
    120 cents is the greatest postage that cannot be formed -/
def is_valid_denomination (n : ℕ) : Prop :=
  ∀ k : ℕ, k > 120 → ∃ (a b c : ℕ), k = 7 * a + n * b + (n + 2) * c

/-- 120 cents cannot be formed using stamps of denominations 7, n, and n+2 cents -/
def cannot_form_120 (n : ℕ) : Prop :=
  ¬∃ (a b c : ℕ), 120 = 7 * a + n * b + (n + 2) * c

theorem unique_denomination :
  ∃! n : ℕ, n > 0 ∧ is_valid_denomination n ∧ cannot_form_120 n :=
by sorry

end NUMINAMATH_CALUDE_unique_denomination_l1356_135639


namespace NUMINAMATH_CALUDE_area_of_triangle_PQR_l1356_135697

/-- Given two lines intersecting at point P(2,5) with slopes 3 and -1 respectively,
    and forming a triangle PQR with the x-axis, prove that the area of triangle PQR is 25/3 -/
theorem area_of_triangle_PQR (P : ℝ × ℝ) (m₁ m₂ : ℝ) : 
  P = (2, 5) →
  m₁ = 3 →
  m₂ = -1 →
  let Q := (P.1 - P.2 / m₁, 0)
  let R := (P.1 + P.2 / m₂, 0)
  (1/2 : ℝ) * |R.1 - Q.1| * P.2 = 25/3 := by
  sorry

end NUMINAMATH_CALUDE_area_of_triangle_PQR_l1356_135697


namespace NUMINAMATH_CALUDE_remainder_problem_l1356_135688

theorem remainder_problem (N : ℤ) (h : N % 296 = 75) : N % 37 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l1356_135688


namespace NUMINAMATH_CALUDE_divisibility_condition_l1356_135601

theorem divisibility_condition (a b : ℕ+) : 
  (∃ k : ℕ, (b.val ^ 2 + 3 * a.val) = a.val ^ 2 * b.val * k) ↔ 
  ((a, b) = (1, 1) ∨ (a, b) = (1, 3)) := by
sorry

end NUMINAMATH_CALUDE_divisibility_condition_l1356_135601


namespace NUMINAMATH_CALUDE_square_property_implies_zero_l1356_135687

theorem square_property_implies_zero (a b : ℤ) : 
  (∀ n : ℕ, ∃ k : ℤ, 2^n * a + b = k^2) → a = 0 :=
by sorry

end NUMINAMATH_CALUDE_square_property_implies_zero_l1356_135687


namespace NUMINAMATH_CALUDE_inequality_theorem_l1356_135670

open Real

noncomputable def f (x : ℝ) : ℝ := (x^2 + 1) / x

noncomputable def g (x : ℝ) : ℝ := x / (Real.exp x)

theorem inequality_theorem (k : ℝ) :
  (∀ x x₂ : ℝ, x > 0 → x₂ > 0 → g x / k ≤ f x₂ / (k + 1)) →
  k ≥ 1 / (2 * Real.exp 1 - 1) := by
sorry

end NUMINAMATH_CALUDE_inequality_theorem_l1356_135670


namespace NUMINAMATH_CALUDE_p_necessary_not_sufficient_for_q_l1356_135693

-- Define the sets corresponding to p and q
def set_p : Set ℝ := {x | (1 - x^2) / (|x| - 2) < 0}
def set_q : Set ℝ := {x | x^2 + x - 6 > 0}

-- State the theorem
theorem p_necessary_not_sufficient_for_q :
  (set_q ⊆ set_p) ∧ (set_q ≠ set_p) := by
  sorry

end NUMINAMATH_CALUDE_p_necessary_not_sufficient_for_q_l1356_135693


namespace NUMINAMATH_CALUDE_william_wins_l1356_135600

theorem william_wins (total_rounds : ℕ) (william_advantage : ℕ) (william_wins : ℕ) : 
  total_rounds = 15 → 
  william_advantage = 5 → 
  william_wins = total_rounds / 2 + william_advantage → 
  william_wins = 10 := by
sorry

end NUMINAMATH_CALUDE_william_wins_l1356_135600


namespace NUMINAMATH_CALUDE_probability_ten_red_balls_in_twelve_draws_l1356_135638

theorem probability_ten_red_balls_in_twelve_draws 
  (total_balls : Nat) (white_balls : Nat) (red_balls : Nat)
  (h1 : total_balls = white_balls + red_balls)
  (h2 : white_balls = 5)
  (h3 : red_balls = 3) :
  let p_red := red_balls / total_balls
  let p_white := white_balls / total_balls
  let n := 11  -- number of draws before the last one
  let k := 9   -- number of red balls in the first 11 draws
  Nat.choose n k * p_red^k * p_white^(n-k) * p_red = 
    Nat.choose 11 9 * (3/8)^9 * (5/8)^2 * (3/8) :=
by sorry

end NUMINAMATH_CALUDE_probability_ten_red_balls_in_twelve_draws_l1356_135638


namespace NUMINAMATH_CALUDE_port_perry_wellington_ratio_l1356_135699

/-- The ratio of Port Perry's population to Wellington's population -/
def population_ratio (port_perry : ℕ) (wellington : ℕ) (lazy_harbor : ℕ) : ℚ :=
  port_perry / wellington

theorem port_perry_wellington_ratio :
  ∀ (port_perry wellington lazy_harbor : ℕ),
    wellington = 900 →
    port_perry = lazy_harbor + 800 →
    port_perry + lazy_harbor = 11800 →
    population_ratio port_perry wellington lazy_harbor = 7 := by
  sorry

#check port_perry_wellington_ratio

end NUMINAMATH_CALUDE_port_perry_wellington_ratio_l1356_135699


namespace NUMINAMATH_CALUDE_triangle_properties_l1356_135681

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The given conditions for the triangle -/
def TriangleConditions (t : Triangle) : Prop :=
  (t.a + t.c)^2 - t.b^2 = 3 * t.a * t.c ∧
  t.b = 6 ∧
  Real.sin t.C = 2 * Real.sin t.A

theorem triangle_properties (t : Triangle) (h : TriangleConditions t) :
  t.B = π / 3 ∧ (1/2 * t.a * t.b : ℝ) = 6 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l1356_135681


namespace NUMINAMATH_CALUDE_square_of_prime_mod_six_l1356_135626

theorem square_of_prime_mod_six (p : ℕ) (h_prime : Nat.Prime p) (h_gt_five : p > 5) :
  p ^ 2 % 6 = 1 := by
  sorry

end NUMINAMATH_CALUDE_square_of_prime_mod_six_l1356_135626


namespace NUMINAMATH_CALUDE_pencils_left_over_l1356_135684

theorem pencils_left_over (total_pencils : ℕ) (students_class1 : ℕ) (students_class2 : ℕ) 
  (h1 : total_pencils = 210)
  (h2 : students_class1 = 30)
  (h3 : students_class2 = 20) :
  total_pencils - (students_class1 + students_class2) * (total_pencils / (students_class1 + students_class2)) = 10 := by
  sorry

end NUMINAMATH_CALUDE_pencils_left_over_l1356_135684


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l1356_135675

theorem right_triangle_hypotenuse : ∀ x₁ x₂ : ℝ,
  x₁^2 - 36*x₁ + 70 = 0 →
  x₂^2 - 36*x₂ + 70 = 0 →
  x₁ ≠ x₂ →
  Real.sqrt (x₁^2 + x₂^2) = 34 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l1356_135675


namespace NUMINAMATH_CALUDE_diamond_six_three_l1356_135645

/-- Define the diamond operation for real numbers -/
noncomputable def diamond (x y : ℝ) : ℝ :=
  sorry

/-- Properties of the diamond operation -/
axiom diamond_zero (x : ℝ) : diamond x 0 = 2 * x
axiom diamond_comm (x y : ℝ) : diamond x y = diamond y x
axiom diamond_succ (x y : ℝ) : diamond (x + 1) y = diamond x y * (y + 2)

/-- Theorem: The value of 6 ◇ 3 is 93750 -/
theorem diamond_six_three : diamond 6 3 = 93750 := by
  sorry

end NUMINAMATH_CALUDE_diamond_six_three_l1356_135645


namespace NUMINAMATH_CALUDE_min_sum_absolute_values_l1356_135659

theorem min_sum_absolute_values :
  (∀ x : ℝ, |x + 3| + |x + 4| + |x + 6| ≥ 3) ∧
  (∃ x : ℝ, |x + 3| + |x + 4| + |x + 6| = 3) :=
by sorry

end NUMINAMATH_CALUDE_min_sum_absolute_values_l1356_135659


namespace NUMINAMATH_CALUDE_range_of_a_l1356_135617

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x ∈ Set.Icc 1 2 → x^2 - a ≥ 0) → a ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l1356_135617


namespace NUMINAMATH_CALUDE_product_first_three_terms_l1356_135663

/-- An arithmetic sequence with given properties -/
def ArithmeticSequence (a : ℕ → ℕ) : Prop :=
  (a 8 = 20) ∧ (∀ n : ℕ, a (n + 1) = a n + 2)

/-- Theorem stating the product of the first three terms -/
theorem product_first_three_terms (a : ℕ → ℕ) (h : ArithmeticSequence a) :
  a 1 * a 2 * a 3 = 480 := by
  sorry


end NUMINAMATH_CALUDE_product_first_three_terms_l1356_135663


namespace NUMINAMATH_CALUDE_sports_club_members_l1356_135633

/-- Represents a sports club with members playing badminton and tennis -/
structure SportsClub where
  badminton : ℕ  -- Number of members playing badminton
  tennis : ℕ     -- Number of members playing tennis
  neither : ℕ    -- Number of members playing neither badminton nor tennis
  both : ℕ       -- Number of members playing both badminton and tennis

/-- Calculates the total number of members in the sports club -/
def totalMembers (club : SportsClub) : ℕ :=
  club.badminton + club.tennis - club.both + club.neither

/-- Theorem stating that the total number of members in the given sports club is 30 -/
theorem sports_club_members :
  ∃ (club : SportsClub), 
    club.badminton = 16 ∧ 
    club.tennis = 19 ∧ 
    club.neither = 2 ∧ 
    club.both = 7 ∧ 
    totalMembers club = 30 := by
  sorry

end NUMINAMATH_CALUDE_sports_club_members_l1356_135633


namespace NUMINAMATH_CALUDE_jerry_feathers_left_l1356_135682

def feathers_left (hawk_feathers : ℕ) (eagle_ratio : ℕ) (given_away : ℕ) : ℕ :=
  let total_feathers := hawk_feathers + eagle_ratio * hawk_feathers
  let remaining_after_gift := total_feathers - given_away
  remaining_after_gift / 2

theorem jerry_feathers_left : feathers_left 6 17 10 = 49 := by
  sorry

end NUMINAMATH_CALUDE_jerry_feathers_left_l1356_135682


namespace NUMINAMATH_CALUDE_polynomial_factorization_l1356_135637

theorem polynomial_factorization (p q : ℕ) (n : ℕ) (a : ℤ) :
  Prime p ∧ Prime q ∧ p ≠ q ∧ n ≥ 3 →
  (∃ (g h : Polynomial ℤ), (Polynomial.degree g ≥ 1) ∧ (Polynomial.degree h ≥ 1) ∧
    (Polynomial.X : Polynomial ℤ)^n + a * (Polynomial.X : Polynomial ℤ)^(n-1) + (p * q : ℤ) = g * h) ↔
  (a = (-1)^n * (p * q : ℤ) + 1 ∨ a = -(p * q : ℤ) - 1) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l1356_135637


namespace NUMINAMATH_CALUDE_parallel_line_through_point_l1356_135609

/-- Given a point p and a line l, this function returns the equation of the line parallel to l that passes through p. -/
def parallel_line_equation (p : ℝ × ℝ) (l : ℝ → ℝ → ℝ → Prop) : ℝ → ℝ → ℝ → Prop :=
  sorry

theorem parallel_line_through_point :
  let p : ℝ × ℝ := (-1, 3)
  let l : ℝ → ℝ → ℝ → Prop := fun x y z ↦ x - 2*y + z = 0
  let result : ℝ → ℝ → ℝ → Prop := fun x y z ↦ x - 2*y + 7 = 0
  parallel_line_equation p l = result :=
sorry

end NUMINAMATH_CALUDE_parallel_line_through_point_l1356_135609


namespace NUMINAMATH_CALUDE_unique_cyclic_number_l1356_135623

def is_permutation (a b : Nat) : Prop := sorry

def has_distinct_digits (n : Nat) : Prop := sorry

theorem unique_cyclic_number : ∃! n : Nat, 
  100000 ≤ n ∧ n < 1000000 ∧ 
  has_distinct_digits n ∧
  is_permutation n (2*n) ∧
  is_permutation n (3*n) ∧
  is_permutation n (4*n) ∧
  is_permutation n (5*n) ∧
  is_permutation n (6*n) ∧
  n = 142857 := by sorry

end NUMINAMATH_CALUDE_unique_cyclic_number_l1356_135623


namespace NUMINAMATH_CALUDE_society_member_numbers_l1356_135664

theorem society_member_numbers (n : ℕ) (k : ℕ) (members : Fin n → Fin k) :
  n = 1978 →
  k = 6 →
  (∀ i : Fin n, (members i).val + 1 = i.val) →
  ∃ i j l : Fin n,
    (members i = members j ∧ members i = members l ∧ i.val = j.val + l.val) ∨
    (members i = members j ∧ i.val = 2 * j.val) :=
by sorry

end NUMINAMATH_CALUDE_society_member_numbers_l1356_135664


namespace NUMINAMATH_CALUDE_white_blue_line_difference_l1356_135620

/-- The length difference between two lines -/
def length_difference (white_line blue_line : ℝ) : ℝ :=
  white_line - blue_line

/-- Theorem stating the length difference between the white and blue lines -/
theorem white_blue_line_difference :
  let white_line : ℝ := 7.666666666666667
  let blue_line : ℝ := 3.3333333333333335
  length_difference white_line blue_line = 4.333333333333333 := by
  sorry

end NUMINAMATH_CALUDE_white_blue_line_difference_l1356_135620


namespace NUMINAMATH_CALUDE_smooth_transition_iff_tangent_l1356_135680

-- Define a circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a line
structure Line where
  slope : ℝ
  intercept : ℝ

-- Define a point
def Point := ℝ × ℝ

-- Define tangency
def isTangent (c : Circle) (l : Line) (p : Point) : Prop :=
  -- The point lies on both the circle and the line
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2 ∧
  p.2 = l.slope * p.1 + l.intercept ∧
  -- The line is perpendicular to the radius at the point of tangency
  l.slope * (p.1 - c.center.1) = -(p.2 - c.center.2)

-- Define smooth transition
def smoothTransition (c : Circle) (l : Line) (p : Point) : Prop :=
  -- The velocity vector is continuous at the transition point
  isTangent c l p

-- Theorem statement
theorem smooth_transition_iff_tangent (c : Circle) (l : Line) (p : Point) :
  smoothTransition c l p ↔ isTangent c l p :=
sorry

end NUMINAMATH_CALUDE_smooth_transition_iff_tangent_l1356_135680


namespace NUMINAMATH_CALUDE_order_powers_l1356_135668

theorem order_powers : 2^300 < 3^200 ∧ 3^200 < 10^100 := by
  sorry

end NUMINAMATH_CALUDE_order_powers_l1356_135668


namespace NUMINAMATH_CALUDE_power_twenty_equals_R_S_l1356_135698

theorem power_twenty_equals_R_S (a b : ℤ) (R S : ℝ) 
  (hR : R = (4 : ℝ) ^ a) 
  (hS : S = (5 : ℝ) ^ b) : 
  (20 : ℝ) ^ (a * b) = R ^ b * S ^ a := by sorry

end NUMINAMATH_CALUDE_power_twenty_equals_R_S_l1356_135698


namespace NUMINAMATH_CALUDE_min_shaded_triangles_theorem_l1356_135640

/-- Represents an equilateral triangle -/
structure EquilateralTriangle where
  sideLength : ℕ

/-- Represents the division of a large equilateral triangle into smaller ones -/
structure TriangleDivision where
  largeSideLength : ℕ
  smallSideLength : ℕ

/-- Calculates the number of intersection points in a triangle division -/
def intersectionPoints (d : TriangleDivision) : ℕ :=
  let n : ℕ := d.largeSideLength / d.smallSideLength + 1
  n * (n + 1) / 2

/-- Calculates the minimum number of smaller triangles needed to be shaded -/
def minShadedTriangles (d : TriangleDivision) : ℕ :=
  (intersectionPoints d + 2) / 3

/-- The main theorem to prove -/
theorem min_shaded_triangles_theorem (t : EquilateralTriangle) (d : TriangleDivision) :
  t.sideLength = 8 →
  d.largeSideLength = 8 →
  d.smallSideLength = 1 →
  minShadedTriangles d = 15 := by
  sorry

end NUMINAMATH_CALUDE_min_shaded_triangles_theorem_l1356_135640


namespace NUMINAMATH_CALUDE_find_b_value_l1356_135616

theorem find_b_value (b : ℚ) : 
  ((-8 : ℚ)^2 + b * (-8 : ℚ) - 15 = 0) → b = 49/8 := by
  sorry

end NUMINAMATH_CALUDE_find_b_value_l1356_135616


namespace NUMINAMATH_CALUDE_syrup_volume_l1356_135673

/-- The final volume of syrup after reduction and sugar addition -/
theorem syrup_volume (y : ℝ) : 
  let initial_volume : ℝ := 6 * 4  -- 6 quarts to cups
  let reduced_volume : ℝ := initial_volume * (1 / 12)
  let volume_with_sugar : ℝ := reduced_volume + 1
  let final_volume : ℝ := volume_with_sugar * y
  final_volume = 3 * y :=
by sorry

end NUMINAMATH_CALUDE_syrup_volume_l1356_135673


namespace NUMINAMATH_CALUDE_non_decreasing_sequence_count_l1356_135679

theorem non_decreasing_sequence_count :
  let max_value : ℕ := 1003
  let seq_length : ℕ := 7
  let sequence_count := Nat.choose 504 seq_length
  ∀ (b : Fin seq_length → ℕ),
    (∀ i j : Fin seq_length, i ≤ j → b i ≤ b j) →
    (∀ i : Fin seq_length, b i ≤ max_value) →
    (∀ i : Fin seq_length, Odd (b i - i.val.succ)) →
    (∃! c : ℕ, c = sequence_count) :=
by sorry

end NUMINAMATH_CALUDE_non_decreasing_sequence_count_l1356_135679


namespace NUMINAMATH_CALUDE_tangent_line_at_2_minus_6_l1356_135611

-- Define the function f
def f (x : ℝ) : ℝ := x^3 + x - 16

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 3 * x^2 + 1

-- Theorem statement
theorem tangent_line_at_2_minus_6 :
  let x₀ : ℝ := 2
  let y₀ : ℝ := f x₀
  let m : ℝ := f' x₀
  let b : ℝ := y₀ - m * x₀
  (∀ x y, y = m * x + b ↔ y - y₀ = m * (x - x₀)) ∧ 
  y₀ = -6 ∧ 
  m = 13 ∧ 
  b = -32 := by sorry

end NUMINAMATH_CALUDE_tangent_line_at_2_minus_6_l1356_135611


namespace NUMINAMATH_CALUDE_tangent_plane_parallel_to_given_plane_l1356_135666

-- Define the elliptic paraboloid
def elliptic_paraboloid (x y : ℝ) : ℝ := 2 * x^2 + 4 * y^2

-- Define the plane
def plane (x y z : ℝ) : ℝ := 8 * x - 32 * y - 2 * z + 3

-- Define the point of tangency
def point_of_tangency : ℝ × ℝ × ℝ := (1, -2, 18)

-- Define the tangent plane at the point of tangency
def tangent_plane (x y z : ℝ) : ℝ := 4 * x - 16 * y - z - 18

theorem tangent_plane_parallel_to_given_plane :
  let (x₀, y₀, z₀) := point_of_tangency
  ∃ (k : ℝ), k ≠ 0 ∧
    (∀ x y z, tangent_plane x y z = k * plane x y z) ∧
    z₀ = elliptic_paraboloid x₀ y₀ :=
by sorry

end NUMINAMATH_CALUDE_tangent_plane_parallel_to_given_plane_l1356_135666


namespace NUMINAMATH_CALUDE_carnival_participants_l1356_135653

theorem carnival_participants (n : ℕ) (masks costumes both : ℕ) : 
  n ≥ 42 →
  masks = (3 * n) / 7 →
  costumes = (5 * n) / 6 →
  both = masks + costumes - n →
  both ≥ 11 := by
sorry

end NUMINAMATH_CALUDE_carnival_participants_l1356_135653


namespace NUMINAMATH_CALUDE_rectangle_longer_side_l1356_135643

/-- Given a circle with radius 6 cm tangent to three sides of a rectangle, 
    and the rectangle's area being three times the area of the circle,
    the length of the longer side of the rectangle is 9π cm. -/
theorem rectangle_longer_side (circle_radius : ℝ) (rectangle_area : ℝ) 
  (h1 : circle_radius = 6)
  (h2 : rectangle_area = 3 * Real.pi * circle_radius^2) : 
  rectangle_area / (2 * circle_radius) = 9 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_rectangle_longer_side_l1356_135643


namespace NUMINAMATH_CALUDE_pt_length_in_quadrilateral_l1356_135658

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a quadrilateral -/
structure Quadrilateral :=
  (P Q R S : Point)

/-- Calculates the length between two points -/
def distance (A B : Point) : ℝ := sorry

/-- Calculates the area of a triangle given three points -/
def triangleArea (A B C : Point) : ℝ := sorry

/-- Theorem: In a convex quadrilateral PQRS, given specific side lengths and conditions, 
    the length of PT can be determined -/
theorem pt_length_in_quadrilateral 
  (PQRS : Quadrilateral)
  (T : Point)
  (convex : sorry) -- Assumption that PQRS is convex
  (pq_length : distance PQRS.P PQRS.Q = 8)
  (rs_length : distance PQRS.R PQRS.S = 14)
  (pr_length : distance PQRS.P PQRS.R = 18)
  (qs_length : distance PQRS.Q PQRS.S = 12)
  (T_on_PR : sorry) -- Assumption that T is on PR
  (T_on_QS : sorry) -- Assumption that T is on QS
  (equal_areas : triangleArea PQRS.P T PQRS.R = triangleArea PQRS.Q T PQRS.S) :
  distance PQRS.P T = 72 / 11 := by sorry

end NUMINAMATH_CALUDE_pt_length_in_quadrilateral_l1356_135658


namespace NUMINAMATH_CALUDE_shell_ratio_l1356_135624

/-- Prove that the ratio of Kyle's shells to Mimi's shells is 2:1 -/
theorem shell_ratio : 
  ∀ (mimi_shells kyle_shells leigh_shells : ℕ),
    mimi_shells = 2 * 12 →
    leigh_shells = 16 →
    3 * leigh_shells = kyle_shells →
    kyle_shells / mimi_shells = 2 := by
  sorry

end NUMINAMATH_CALUDE_shell_ratio_l1356_135624


namespace NUMINAMATH_CALUDE_canteen_theorem_l1356_135686

/-- Represents the number of dishes available --/
def num_dishes : ℕ := 6

/-- Calculates the maximum number of days based on the number of dishes --/
def max_days (n : ℕ) : ℕ := 2^n

/-- Calculates the average number of dishes per day --/
def avg_dishes_per_day (n : ℕ) : ℚ := n / 2

theorem canteen_theorem :
  max_days num_dishes = 64 ∧ avg_dishes_per_day num_dishes = 3 := by sorry

end NUMINAMATH_CALUDE_canteen_theorem_l1356_135686


namespace NUMINAMATH_CALUDE_smallest_bob_number_l1356_135672

def alice_number : Nat := 30

-- Function to check if all prime factors of a are prime factors of b
def all_prime_factors_of (a b : Nat) : Prop := 
  ∀ p : Nat, Nat.Prime p → (p ∣ a → p ∣ b)

theorem smallest_bob_number : 
  ∃ bob_number : Nat, 
    (all_prime_factors_of alice_number bob_number) ∧ 
    (all_prime_factors_of bob_number alice_number) ∧ 
    (∀ n : Nat, n < bob_number → 
      ¬(all_prime_factors_of alice_number n ∧ all_prime_factors_of n alice_number)) ∧
    bob_number = alice_number := by
  sorry

end NUMINAMATH_CALUDE_smallest_bob_number_l1356_135672


namespace NUMINAMATH_CALUDE_smallest_solution_cubic_equation_l1356_135690

theorem smallest_solution_cubic_equation :
  ∃ (x : ℝ), x = 2/3 ∧ 24 * x^3 - 106 * x^2 + 116 * x - 70 = 0 ∧
  ∀ (y : ℝ), 24 * y^3 - 106 * y^2 + 116 * y - 70 = 0 → y ≥ 2/3 :=
sorry

end NUMINAMATH_CALUDE_smallest_solution_cubic_equation_l1356_135690


namespace NUMINAMATH_CALUDE_sum_of_cubes_l1356_135619

theorem sum_of_cubes (x y : ℝ) (h1 : x + y = 8) (h2 : x * y = 15) :
  x^3 + y^3 = 152 := by
sorry

end NUMINAMATH_CALUDE_sum_of_cubes_l1356_135619


namespace NUMINAMATH_CALUDE_exponential_function_fixed_point_l1356_135627

theorem exponential_function_fixed_point (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ a^x + 1
  f 0 = 2 := by
sorry

end NUMINAMATH_CALUDE_exponential_function_fixed_point_l1356_135627


namespace NUMINAMATH_CALUDE_geometric_sequence_properties_l1356_135656

/-- A sequence is geometric if the ratio of consecutive terms is constant. -/
def IsGeometric (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = q * a n

theorem geometric_sequence_properties
  (a : ℕ → ℝ) (h : IsGeometric a) :
  ∃ q : ℝ,
    (IsGeometric (fun n ↦ (a n)^3)) ∧
    (∀ p : ℝ, p ≠ 0 → IsGeometric (fun n ↦ p * a n)) ∧
    (IsGeometric (fun n ↦ a n * a (n + 1))) ∧
    (IsGeometric (fun n ↦ a n + a (n + 1))) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_properties_l1356_135656


namespace NUMINAMATH_CALUDE_polygon_sides_l1356_135625

theorem polygon_sides (n : ℕ) : (n - 2) * 180 = 1800 → n = 12 := by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_l1356_135625


namespace NUMINAMATH_CALUDE_expected_value_is_six_point_five_l1356_135630

/-- A fair 12-sided die with faces numbered from 1 to 12 -/
def twelve_sided_die : Finset ℕ := Finset.range 12

/-- The expected value of rolling the 12-sided die -/
def expected_value : ℚ := (Finset.sum twelve_sided_die (λ i => i + 1)) / 12

/-- Theorem stating that the expected value of rolling the 12-sided die is 6.5 -/
theorem expected_value_is_six_point_five : expected_value = 13/2 := by
  sorry

end NUMINAMATH_CALUDE_expected_value_is_six_point_five_l1356_135630


namespace NUMINAMATH_CALUDE_floor_abs_sum_equality_l1356_135628

theorem floor_abs_sum_equality : ⌊|(-3.7 : ℝ)|⌋ + |⌊(-3.7 : ℝ)⌋| = 7 := by
  sorry

end NUMINAMATH_CALUDE_floor_abs_sum_equality_l1356_135628


namespace NUMINAMATH_CALUDE_fraction_evaluation_l1356_135604

theorem fraction_evaluation : (1 - 1/4) / (1 - 1/3) = 9/8 := by
  sorry

end NUMINAMATH_CALUDE_fraction_evaluation_l1356_135604


namespace NUMINAMATH_CALUDE_traditionalist_fraction_l1356_135695

theorem traditionalist_fraction (num_provinces : ℕ) (num_traditionalists_per_province : ℚ) 
  (total_progressives : ℚ) :
  num_provinces = 6 →
  num_traditionalists_per_province = total_progressives / 9 →
  (num_provinces : ℚ) * num_traditionalists_per_province / 
    (total_progressives + (num_provinces : ℚ) * num_traditionalists_per_province) = 2 / 5 :=
by sorry

end NUMINAMATH_CALUDE_traditionalist_fraction_l1356_135695


namespace NUMINAMATH_CALUDE_cantor_set_segments_l1356_135649

/-- The number of segments after n iterations of the process -/
def num_segments (n : ℕ) : ℕ := 2^n

/-- The length of each segment after n iterations of the process -/
def segment_length (n : ℕ) : ℚ := (1 : ℚ) / 3^n

theorem cantor_set_segments :
  num_segments 16 = 2^16 ∧ segment_length 16 = (1 : ℚ) / 3^16 := by
  sorry

#eval num_segments 16  -- To check the result

end NUMINAMATH_CALUDE_cantor_set_segments_l1356_135649


namespace NUMINAMATH_CALUDE_convex_pentagon_with_equal_diagonals_and_sides_l1356_135641

-- Define a pentagon as a set of 5 points in 2D space
def Pentagon := Fin 5 → ℝ × ℝ

-- Define a function to check if a pentagon is convex
def is_convex (p : Pentagon) : Prop := sorry

-- Define a function to calculate the length of a line segment
def length (a b : ℝ × ℝ) : ℝ := sorry

-- Define a function to check if a line segment is a diagonal of the pentagon
def is_diagonal (p : Pentagon) (i j : Fin 5) : Prop :=
  (i.val + 2) % 5 ≤ j.val ∨ (j.val + 2) % 5 ≤ i.val

-- Define a function to check if a line segment is a side of the pentagon
def is_side (p : Pentagon) (i j : Fin 5) : Prop :=
  (i.val + 1) % 5 = j.val ∨ (j.val + 1) % 5 = i.val

-- Theorem: There exists a convex pentagon where each diagonal is equal to some side
theorem convex_pentagon_with_equal_diagonals_and_sides :
  ∃ (p : Pentagon), is_convex p ∧
    ∀ (i j : Fin 5), is_diagonal p i j →
      ∃ (k l : Fin 5), is_side p k l ∧ length (p i) (p j) = length (p k) (p l) :=
sorry

end NUMINAMATH_CALUDE_convex_pentagon_with_equal_diagonals_and_sides_l1356_135641


namespace NUMINAMATH_CALUDE_sector_perimeter_l1356_135660

/-- Given a circular sector with area 2 cm² and central angle 4 radians, its perimeter is 6 cm. -/
theorem sector_perimeter (r : ℝ) (θ : ℝ) : 
  (1/2 * r^2 * θ = 2) → θ = 4 → (r * θ + 2 * r = 6) := by
  sorry

end NUMINAMATH_CALUDE_sector_perimeter_l1356_135660


namespace NUMINAMATH_CALUDE_solution_set_for_negative_one_range_of_a_l1356_135635

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |2*x + a| + |x - 2|

-- Part 1
theorem solution_set_for_negative_one (x : ℝ) :
  (f (-1) x ≥ 6) ↔ (x ≤ -1 ∨ x ≥ 3) := by sorry

-- Part 2
theorem range_of_a (a : ℝ) :
  (∀ x, f a x ≥ 3*a^2 - |2 - x|) → (-1 ≤ a ∧ a ≤ 4/3) := by sorry

end NUMINAMATH_CALUDE_solution_set_for_negative_one_range_of_a_l1356_135635


namespace NUMINAMATH_CALUDE_girls_in_class_l1356_135650

/-- Proves that in a class with a boy-to-girl ratio of 5:8 and 260 total students, there are 160 girls -/
theorem girls_in_class (total : ℕ) (boys_ratio girls_ratio : ℕ) (h1 : total = 260) (h2 : boys_ratio = 5) (h3 : girls_ratio = 8) : 
  (girls_ratio : ℚ) / (boys_ratio + girls_ratio : ℚ) * total = 160 := by
sorry

end NUMINAMATH_CALUDE_girls_in_class_l1356_135650


namespace NUMINAMATH_CALUDE_track_length_proof_l1356_135678

/-- The length of the circular track -/
def track_length : ℝ := 600

/-- The distance Brenda runs before the first meeting -/
def brenda_first_distance : ℝ := 120

/-- The additional distance Sally runs between the first and second meeting -/
def sally_additional_distance : ℝ := 180

/-- Theorem stating the length of the track given the meeting conditions -/
theorem track_length_proof :
  ∃ (brenda_speed sally_speed : ℝ),
    brenda_speed > 0 ∧ sally_speed > 0 ∧
    brenda_first_distance / (track_length / 2 - brenda_first_distance) = brenda_speed / sally_speed ∧
    (track_length / 2 - brenda_first_distance + sally_additional_distance) / (brenda_first_distance + track_length / 2 - (track_length / 2 - brenda_first_distance + sally_additional_distance)) = sally_speed / brenda_speed :=
by
  sorry

end NUMINAMATH_CALUDE_track_length_proof_l1356_135678


namespace NUMINAMATH_CALUDE_mean_equality_implies_x_value_l1356_135647

theorem mean_equality_implies_x_value : 
  let mean1 := (8 + 15 + 21) / 3
  let mean2 := (18 + x) / 2
  mean1 = mean2 → x = 34 / 3 :=
by
  sorry

end NUMINAMATH_CALUDE_mean_equality_implies_x_value_l1356_135647


namespace NUMINAMATH_CALUDE_coin_to_sphere_weight_change_l1356_135629

theorem coin_to_sphere_weight_change 
  (R₁ R₂ R₃ : ℝ) 
  (h_positive : 0 < R₁ ∧ 0 < R₂ ∧ 0 < R₃) 
  (h_balance : R₁^2 + R₂^2 = R₃^2) : 
  R₁^3 + R₂^3 < R₃^3 := by
sorry

end NUMINAMATH_CALUDE_coin_to_sphere_weight_change_l1356_135629


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l1356_135631

def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {x : ℕ | x ≤ 2}

theorem union_of_A_and_B : A ∪ B = {0, 1, 2, 3} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l1356_135631


namespace NUMINAMATH_CALUDE_cubic_equation_sum_of_cubes_l1356_135667

theorem cubic_equation_sum_of_cubes :
  ∃ (r s t : ℝ),
    (∀ x : ℝ, (x - Real.rpow 17 (1/3 : ℝ)) * (x - Real.rpow 37 (1/3 : ℝ)) * (x - Real.rpow 57 (1/3 : ℝ)) = -1/2 ↔ x = r ∨ x = s ∨ x = t) →
    r^3 + s^3 + t^3 = 107.5 := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_sum_of_cubes_l1356_135667
