import Mathlib

namespace NUMINAMATH_CALUDE_sphere_radius_ratio_l3278_327817

theorem sphere_radius_ratio (V_large V_small r_large r_small : ℝ) :
  V_large = 500 * Real.pi
  → V_small = 0.25 * V_large
  → V_large = (4/3) * Real.pi * r_large^3
  → V_small = (4/3) * Real.pi * r_small^3
  → r_small / r_large = 1 / (2^(2/3)) := by
  sorry

end NUMINAMATH_CALUDE_sphere_radius_ratio_l3278_327817


namespace NUMINAMATH_CALUDE_volleyball_match_probability_l3278_327885

/-- The probability of Team A winning a single set -/
def p_A : ℚ := 2/3

/-- The probability of Team B winning a single set -/
def p_B : ℚ := 1 - p_A

/-- The number of sets Team B has won at the start -/
def initial_B_wins : ℕ := 2

/-- The number of sets needed to win the match -/
def sets_to_win : ℕ := 3

/-- The probability of Team B winning the match given they lead 2:0 -/
def p_B_wins : ℚ := p_B + p_A * p_B + p_A * p_A * p_B

theorem volleyball_match_probability :
  p_B_wins = 19/27 := by sorry

end NUMINAMATH_CALUDE_volleyball_match_probability_l3278_327885


namespace NUMINAMATH_CALUDE_platform_length_proof_l3278_327891

/-- Proves that given a train with specified speed and length, crossing a platform in a certain time, the platform length is approximately 165 meters. -/
theorem platform_length_proof (train_speed : Real) (train_length : Real) (crossing_time : Real) :
  train_speed = 132 * 1000 / 3600 →
  train_length = 110 →
  crossing_time = 7.499400047996161 →
  ∃ (platform_length : Real),
    (platform_length + train_length) = train_speed * crossing_time ∧
    abs (platform_length - 165) < 1 := by
  sorry

end NUMINAMATH_CALUDE_platform_length_proof_l3278_327891


namespace NUMINAMATH_CALUDE_real_part_of_complex_fraction_l3278_327816

theorem real_part_of_complex_fraction (i : ℂ) :
  i * i = -1 →
  Complex.re ((2 : ℂ) + i) / i = 1 := by
  sorry

end NUMINAMATH_CALUDE_real_part_of_complex_fraction_l3278_327816


namespace NUMINAMATH_CALUDE_min_value_when_m_eq_one_m_range_when_f_geq_2x_l3278_327807

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := |x + 1| + |m * x - 1|

-- Part 1
theorem min_value_when_m_eq_one :
  (∃ (min : ℝ), ∀ x, f 1 x ≥ min ∧ ∃ x₀ ∈ Set.Icc (-1) 1, f 1 x₀ = min) ∧
  (∀ x, f 1 x = 2 ↔ x ∈ Set.Icc (-1) 1) := by sorry

-- Part 2
theorem m_range_when_f_geq_2x :
  (∀ x, f m x ≥ 2 * x) ↔ m ∈ Set.Iic (-1) ∪ Set.Ici 1 := by sorry

end NUMINAMATH_CALUDE_min_value_when_m_eq_one_m_range_when_f_geq_2x_l3278_327807


namespace NUMINAMATH_CALUDE_martin_initial_hens_correct_martin_initial_hens_unique_l3278_327846

/-- Represents the farm's egg production scenario -/
structure FarmScenario where
  initial_hens : ℕ
  initial_eggs : ℕ
  initial_days : ℕ
  added_hens : ℕ
  final_eggs : ℕ
  final_days : ℕ

/-- The specific scenario from the problem -/
def martin_farm : FarmScenario :=
  { initial_hens := 25,  -- This is what we want to prove
    initial_eggs := 80,
    initial_days := 10,
    added_hens := 15,
    final_eggs := 300,
    final_days := 15 }

/-- Theorem stating that Martin's initial number of hens is correct -/
theorem martin_initial_hens_correct :
  martin_farm.initial_hens * martin_farm.final_days * martin_farm.initial_eggs =
  martin_farm.initial_days * martin_farm.final_eggs * martin_farm.initial_hens +
  martin_farm.initial_days * martin_farm.final_eggs * martin_farm.added_hens :=
by sorry

/-- Theorem proving that 25 is the only solution -/
theorem martin_initial_hens_unique (h : ℕ) :
  h * martin_farm.final_days * martin_farm.initial_eggs =
  martin_farm.initial_days * martin_farm.final_eggs * h +
  martin_farm.initial_days * martin_farm.final_eggs * martin_farm.added_hens →
  h = martin_farm.initial_hens :=
by sorry

end NUMINAMATH_CALUDE_martin_initial_hens_correct_martin_initial_hens_unique_l3278_327846


namespace NUMINAMATH_CALUDE_inequality_equivalence_l3278_327888

/-- The set of points satisfying the inequality -/
def S : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | let x := p.1; let y := p.2;
    (abs x ≤ 1 ∧ abs y ≤ 1 ∧ x * y ≤ 0) ∨
    (x^2 + y^2 ≤ 1 ∧ x * y > 0)}

/-- The main theorem -/
theorem inequality_equivalence (x y : ℝ) :
  Real.sqrt (1 - x^2) * Real.sqrt (1 - y^2) ≥ x * y ↔ (x, y) ∈ S :=
sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l3278_327888


namespace NUMINAMATH_CALUDE_brianna_remaining_money_l3278_327856

theorem brianna_remaining_money
  (m n c : ℝ)
  (h1 : m > 0)
  (h2 : n > 0)
  (h3 : c > 0)
  (h4 : (1/4) * m = (1/2) * n * c) :
  m - ((1/2) * n * c) - ((1/10) * m) = (2/5) * m :=
sorry

end NUMINAMATH_CALUDE_brianna_remaining_money_l3278_327856


namespace NUMINAMATH_CALUDE_curve_to_linear_equation_l3278_327858

/-- Given a curve parameterized by (x, y) = (3t + 6, 5t - 3), where t is a real number,
    prove that it can be expressed as the linear equation y = (5/3)x - 13. -/
theorem curve_to_linear_equation :
  ∀ (t x y : ℝ), x = 3 * t + 6 ∧ y = 5 * t - 3 →
  y = (5 / 3 : ℝ) * x - 13 :=
by
  sorry

end NUMINAMATH_CALUDE_curve_to_linear_equation_l3278_327858


namespace NUMINAMATH_CALUDE_nines_in_sixty_houses_l3278_327840

def count_nines (n : ℕ) : ℕ :=
  (n + 10) / 10

theorem nines_in_sixty_houses :
  count_nines 60 = 6 := by
  sorry

end NUMINAMATH_CALUDE_nines_in_sixty_houses_l3278_327840


namespace NUMINAMATH_CALUDE_f_derivative_at_zero_l3278_327835

def f (x : ℝ) : ℝ := x * (1 + x)

theorem f_derivative_at_zero : 
  (deriv f) 0 = 1 := by sorry

end NUMINAMATH_CALUDE_f_derivative_at_zero_l3278_327835


namespace NUMINAMATH_CALUDE_coyote_speed_calculation_l3278_327853

/-- The speed of the coyote in miles per hour -/
def coyote_speed : ℝ := 15

/-- The time elapsed since the coyote left its prints, in hours -/
def time_elapsed : ℝ := 1

/-- Darrel's speed on his motorbike in miles per hour -/
def darrel_speed : ℝ := 30

/-- The time it takes Darrel to catch up to the coyote, in hours -/
def catch_up_time : ℝ := 1

theorem coyote_speed_calculation :
  coyote_speed * time_elapsed + coyote_speed * catch_up_time = darrel_speed * catch_up_time := by
  sorry

#check coyote_speed_calculation

end NUMINAMATH_CALUDE_coyote_speed_calculation_l3278_327853


namespace NUMINAMATH_CALUDE_angle_between_v_and_w_l3278_327810

/-- The angle between two vectors in ℝ³ -/
def angle (v w : ℝ × ℝ × ℝ) : ℝ := sorry

/-- Vector 1 -/
def v : ℝ × ℝ × ℝ := (3, -2, 2)

/-- Vector 2 -/
def w : ℝ × ℝ × ℝ := (2, 2, -1)

/-- Theorem: The angle between vectors v and w is 90° -/
theorem angle_between_v_and_w : angle v w = 90 := by sorry

end NUMINAMATH_CALUDE_angle_between_v_and_w_l3278_327810


namespace NUMINAMATH_CALUDE_current_average_age_l3278_327869

-- Define the number of people in the initial group
def initial_group : ℕ := 6

-- Define the average age of the initial group after two years
def future_average_age : ℕ := 43

-- Define the age of the new person joining the group
def new_person_age : ℕ := 69

-- Define the total number of people after the new person joins
def total_people : ℕ := initial_group + 1

-- Theorem to prove
theorem current_average_age :
  (initial_group * future_average_age - initial_group * 2 + new_person_age) / total_people = 45 :=
by sorry

end NUMINAMATH_CALUDE_current_average_age_l3278_327869


namespace NUMINAMATH_CALUDE_school_supplies_pretax_amount_l3278_327849

theorem school_supplies_pretax_amount 
  (sales_tax_rate : ℝ) 
  (total_with_tax : ℝ) 
  (h1 : sales_tax_rate = 0.08)
  (h2 : total_with_tax = 162) :
  ∃ (pretax_amount : ℝ), 
    pretax_amount * (1 + sales_tax_rate) = total_with_tax ∧ 
    pretax_amount = 150 := by
  sorry

end NUMINAMATH_CALUDE_school_supplies_pretax_amount_l3278_327849


namespace NUMINAMATH_CALUDE_work_efficiency_ratio_l3278_327845

theorem work_efficiency_ratio (a b : ℝ) : 
  a + b = 1 / 26 → b = 1 / 39 → a / b = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_work_efficiency_ratio_l3278_327845


namespace NUMINAMATH_CALUDE_arithmetic_expression_equality_l3278_327881

theorem arithmetic_expression_equality : (2 + 3^2) * 4 - 6 / 3 + 5^2 = 67 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_expression_equality_l3278_327881


namespace NUMINAMATH_CALUDE_right_square_pyramid_base_neq_lateral_l3278_327871

/-- A right square pyramid -/
structure RightSquarePyramid where
  baseEdge : ℝ
  lateralEdge : ℝ
  height : ℝ

/-- Theorem: In a right square pyramid, the base edge length cannot be equal to the lateral edge length -/
theorem right_square_pyramid_base_neq_lateral (p : RightSquarePyramid) : 
  p.baseEdge ≠ p.lateralEdge :=
sorry

end NUMINAMATH_CALUDE_right_square_pyramid_base_neq_lateral_l3278_327871


namespace NUMINAMATH_CALUDE_subset_implies_m_range_l3278_327832

theorem subset_implies_m_range (m : ℝ) : 
  let A := Set.Iic m
  let B := Set.Ioo 1 2
  B ⊆ A → m ∈ Set.Ici 2 := by
sorry

end NUMINAMATH_CALUDE_subset_implies_m_range_l3278_327832


namespace NUMINAMATH_CALUDE_pairball_playing_time_l3278_327821

theorem pairball_playing_time (total_time : ℕ) (num_children : ℕ) (players_per_game : ℕ) :
  total_time = 90 ∧ 
  num_children = 6 ∧ 
  players_per_game = 2 →
  (total_time * players_per_game) / num_children = 30 := by
sorry

end NUMINAMATH_CALUDE_pairball_playing_time_l3278_327821


namespace NUMINAMATH_CALUDE_problem_solution_l3278_327861

theorem problem_solution (p q : ℝ) 
  (h1 : 1 < p) (h2 : p < q) 
  (h3 : 1 / p + 1 / q = 1) 
  (h4 : p * q = 8) : 
  q = (1 + Real.sqrt 33) / 2 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l3278_327861


namespace NUMINAMATH_CALUDE_max_large_chips_l3278_327868

theorem max_large_chips :
  ∀ (small large : ℕ),
  small + large = 72 →
  ∃ (p : ℕ), Prime p ∧ small = large + p →
  large ≤ 35 :=
by sorry

end NUMINAMATH_CALUDE_max_large_chips_l3278_327868


namespace NUMINAMATH_CALUDE_bracket_ratio_eq_neg_199_l3278_327825

/-- Definition of the bracket operation -/
def bracket (a : ℝ) (k : ℕ+) : ℝ := a * (a - k)

/-- The main theorem to prove -/
theorem bracket_ratio_eq_neg_199 :
  (bracket (-1/2) 100) / (bracket (1/2) 100) = -199 := by sorry

end NUMINAMATH_CALUDE_bracket_ratio_eq_neg_199_l3278_327825


namespace NUMINAMATH_CALUDE_complex_number_in_first_quadrant_l3278_327818

theorem complex_number_in_first_quadrant :
  let z : ℂ := (1 : ℂ) / (1 + Complex.I) + Complex.I
  (z.re > 0) ∧ (z.im > 0) :=
by sorry

end NUMINAMATH_CALUDE_complex_number_in_first_quadrant_l3278_327818


namespace NUMINAMATH_CALUDE_percent_increase_proof_l3278_327820

def original_lines : ℕ := 5600 - 1600
def increased_lines : ℕ := 5600
def line_increase : ℕ := 1600

theorem percent_increase_proof :
  (line_increase : ℝ) / (original_lines : ℝ) * 100 = 40 := by
  sorry

end NUMINAMATH_CALUDE_percent_increase_proof_l3278_327820


namespace NUMINAMATH_CALUDE_special_numbers_count_l3278_327833

def count_multiples (n : ℕ) (d : ℕ) : ℕ := (n / d : ℕ)

def count_special_numbers (upper_bound : ℕ) : ℕ :=
  count_multiples upper_bound 5 + count_multiples upper_bound 7 - count_multiples upper_bound 35

theorem special_numbers_count :
  count_special_numbers 3000 = 943 := by
  sorry

end NUMINAMATH_CALUDE_special_numbers_count_l3278_327833


namespace NUMINAMATH_CALUDE_books_given_away_l3278_327879

theorem books_given_away (original_books : Real) (books_left : Nat) : 
  original_books = 54.0 → books_left = 31 → original_books - books_left = 23 := by
  sorry

end NUMINAMATH_CALUDE_books_given_away_l3278_327879


namespace NUMINAMATH_CALUDE_figurine_cost_l3278_327862

theorem figurine_cost (tv_count : ℕ) (tv_price : ℕ) (figurine_count : ℕ) (total_spent : ℕ) :
  tv_count = 5 →
  tv_price = 50 →
  figurine_count = 10 →
  total_spent = 260 →
  (total_spent - tv_count * tv_price) / figurine_count = 1 :=
by sorry

end NUMINAMATH_CALUDE_figurine_cost_l3278_327862


namespace NUMINAMATH_CALUDE_y1_greater_than_y2_l3278_327892

/-- Represents a quadratic function y = ax² + bx + c -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  a_pos : a > 0

/-- The symmetric axis of the quadratic function is x = 1 -/
def symmetric_axis (f : QuadraticFunction) : ℝ := 1

/-- The quadratic function passes through the point (-1, y₁) -/
def passes_through_minus_one (f : QuadraticFunction) (y₁ : ℝ) : Prop :=
  f.a * (-1)^2 + f.b * (-1) + f.c = y₁

/-- The quadratic function passes through the point (2, y₂) -/
def passes_through_two (f : QuadraticFunction) (y₂ : ℝ) : Prop :=
  f.a * 2^2 + f.b * 2 + f.c = y₂

/-- Theorem stating that y₁ > y₂ for the given conditions -/
theorem y1_greater_than_y2 (f : QuadraticFunction) (y₁ y₂ : ℝ)
  (h1 : passes_through_minus_one f y₁)
  (h2 : passes_through_two f y₂) :
  y₁ > y₂ := by
  sorry

end NUMINAMATH_CALUDE_y1_greater_than_y2_l3278_327892


namespace NUMINAMATH_CALUDE_three_four_five_pythagorean_one_two_five_not_pythagorean_two_three_four_not_pythagorean_four_five_six_not_pythagorean_only_three_four_five_pythagorean_l3278_327890

/-- A function that checks if three numbers form a Pythagorean triple --/
def isPythagoreanTriple (a b c : ℕ) : Prop :=
  a * a + b * b = c * c

/-- Theorem stating that (3, 4, 5) is a Pythagorean triple --/
theorem three_four_five_pythagorean : isPythagoreanTriple 3 4 5 := by
  sorry

/-- Theorem stating that (1, 2, 5) is not a Pythagorean triple --/
theorem one_two_five_not_pythagorean : ¬ isPythagoreanTriple 1 2 5 := by
  sorry

/-- Theorem stating that (2, 3, 4) is not a Pythagorean triple --/
theorem two_three_four_not_pythagorean : ¬ isPythagoreanTriple 2 3 4 := by
  sorry

/-- Theorem stating that (4, 5, 6) is not a Pythagorean triple --/
theorem four_five_six_not_pythagorean : ¬ isPythagoreanTriple 4 5 6 := by
  sorry

/-- Main theorem stating that among the given sets, only (3, 4, 5) is a Pythagorean triple --/
theorem only_three_four_five_pythagorean :
  (isPythagoreanTriple 3 4 5) ∧
  (¬ isPythagoreanTriple 1 2 5) ∧
  (¬ isPythagoreanTriple 2 3 4) ∧
  (¬ isPythagoreanTriple 4 5 6) := by
  sorry

end NUMINAMATH_CALUDE_three_four_five_pythagorean_one_two_five_not_pythagorean_two_three_four_not_pythagorean_four_five_six_not_pythagorean_only_three_four_five_pythagorean_l3278_327890


namespace NUMINAMATH_CALUDE_starting_number_property_l3278_327886

def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sumOfDigits (n / 10)

def subtractSumOfDigits (n : ℕ) : ℕ :=
  n - sumOfDigits n

def iterateSubtraction (n : ℕ) (iterations : ℕ) : ℕ :=
  match iterations with
  | 0 => n
  | k + 1 => iterateSubtraction (subtractSumOfDigits n) k

theorem starting_number_property (n : ℕ) (h : 100 ≤ n ∧ n ≤ 109) :
  iterateSubtraction n 11 = 0 :=
sorry

end NUMINAMATH_CALUDE_starting_number_property_l3278_327886


namespace NUMINAMATH_CALUDE_intersection_points_sum_l3278_327827

-- Define the functions
def f (x : ℝ) : ℝ := (x - 2) * (x - 4)
def g (x : ℝ) : ℝ := -f x
def h (x : ℝ) : ℝ := f (-x)

-- Define c as the number of intersection points between f and g
def c : ℕ := 2

-- Define d as the number of intersection points between f and h
def d : ℕ := 1

-- Theorem to prove
theorem intersection_points_sum : 10 * c + d = 21 := by
  sorry

end NUMINAMATH_CALUDE_intersection_points_sum_l3278_327827


namespace NUMINAMATH_CALUDE_number_problem_l3278_327894

theorem number_problem (N : ℝ) :
  (1/4 : ℝ) * (1/3 : ℝ) * (2/5 : ℝ) * N = 10 →
  (40/100 : ℝ) * N = 120 := by
sorry

end NUMINAMATH_CALUDE_number_problem_l3278_327894


namespace NUMINAMATH_CALUDE_triangle_angle_calculation_l3278_327884

theorem triangle_angle_calculation (a b : ℝ) (A B : ℝ) :
  a = 1 →
  b = Real.sqrt 3 →
  A = π / 6 →
  (B = π / 3 ∨ B = 2 * π / 3) →
  Real.sin B = (b * Real.sin A) / a :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_calculation_l3278_327884


namespace NUMINAMATH_CALUDE_solution_satisfies_system_l3278_327836

-- Define the system of equations
def equation1 (x y z : ℚ) : Prop :=
  6 / (3 * x + 4 * y) + 4 / (5 * x - 4 * z) = 7 / 12

def equation2 (x y z : ℚ) : Prop :=
  9 / (4 * y + 3 * z) - 4 / (3 * x + 4 * y) = 1 / 3

def equation3 (x y z : ℚ) : Prop :=
  2 / (5 * x - 4 * z) + 6 / (4 * y + 3 * z) = 1 / 2

-- Theorem statement
theorem solution_satisfies_system :
  equation1 4 3 2 ∧ equation2 4 3 2 ∧ equation3 4 3 2 := by
  sorry

end NUMINAMATH_CALUDE_solution_satisfies_system_l3278_327836


namespace NUMINAMATH_CALUDE_reflection_line_equation_l3278_327838

-- Define the points of the original triangle
def P : ℝ × ℝ := (3, 2)
def Q : ℝ × ℝ := (8, 7)
def R : ℝ × ℝ := (6, -4)

-- Define the points of the reflected triangle
def P' : ℝ × ℝ := (-5, 2)
def Q' : ℝ × ℝ := (-10, 7)
def R' : ℝ × ℝ := (-8, -4)

-- Define the reflection line
def M : ℝ → Prop := λ x => x = -1

theorem reflection_line_equation :
  (∀ (x y : ℝ), (x, y) = P ∨ (x, y) = Q ∨ (x, y) = R →
    ∃ (x' : ℝ), M x' ∧ x' = (x + P'.1) / 2) ∧
  (∀ (x y : ℝ), (x, y) = P' ∨ (x, y) = Q' ∨ (x, y) = R' →
    ∃ (x' : ℝ), M x' ∧ x' = (x + P.1) / 2) :=
sorry

end NUMINAMATH_CALUDE_reflection_line_equation_l3278_327838


namespace NUMINAMATH_CALUDE_certain_number_theorem_l3278_327823

theorem certain_number_theorem (a n : ℕ) (h1 : a = 105) (h2 : a^3 = 21 * n * 45 * 49) : n = 25 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_theorem_l3278_327823


namespace NUMINAMATH_CALUDE_speed_calculation_l3278_327842

/-- The speed of the first person traveling from A to B -/
def speed_person1 : ℝ := 70

/-- The speed of the second person traveling from B to A -/
def speed_person2 : ℝ := 80

/-- The total distance between A and B in km -/
def total_distance : ℝ := 600

/-- The time in hours it takes for the two people to meet -/
def meeting_time : ℝ := 4

theorem speed_calculation :
  speed_person1 * meeting_time + speed_person2 * meeting_time = total_distance ∧
  speed_person1 * meeting_time = total_distance - speed_person2 * meeting_time :=
by sorry

end NUMINAMATH_CALUDE_speed_calculation_l3278_327842


namespace NUMINAMATH_CALUDE_store_discount_percentage_l3278_327876

theorem store_discount_percentage (C : ℝ) (C_pos : C > 0) : 
  let initial_price := 1.20 * C
  let new_year_price := 1.25 * initial_price
  let february_price := 1.20 * C
  let discount := new_year_price - february_price
  discount / new_year_price = 0.20 := by
sorry

end NUMINAMATH_CALUDE_store_discount_percentage_l3278_327876


namespace NUMINAMATH_CALUDE_probability_theorem_l3278_327870

/-- Represents the total number of products -/
def total_products : ℕ := 7

/-- Represents the number of genuine products -/
def genuine_products : ℕ := 4

/-- Represents the number of defective products -/
def defective_products : ℕ := 3

/-- The probability of selecting a genuine product on the second draw,
    given that a defective product was selected on the first draw -/
def probability_genuine_second_given_defective_first : ℚ := 2/3

/-- Theorem stating the probability of selecting a genuine product on the second draw,
    given that a defective product was selected on the first draw -/
theorem probability_theorem :
  probability_genuine_second_given_defective_first = 2/3 :=
by sorry

end NUMINAMATH_CALUDE_probability_theorem_l3278_327870


namespace NUMINAMATH_CALUDE_equation_solution_l3278_327887

theorem equation_solution :
  ∃! x : ℝ, x ≠ 3 ∧
    ∃ z : ℝ, z = (x^2 - 9) / (x - 3) ∧ z = 3 * x ∧
    x = 3 / 2 :=
by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3278_327887


namespace NUMINAMATH_CALUDE_fourth_root_equation_solution_l3278_327815

theorem fourth_root_equation_solution (x : ℝ) (h1 : x > 0) 
  (h2 : (1 - x^4)^(1/4) + (1 + x^4)^(1/4) = 1) : x^8 = 35/36 := by
  sorry

end NUMINAMATH_CALUDE_fourth_root_equation_solution_l3278_327815


namespace NUMINAMATH_CALUDE_inductive_inequality_l3278_327882

theorem inductive_inequality (x : ℝ) (n : ℕ) (h1 : x > 0) (h2 : n > 0) 
  (h3 : x + 1 / x ≥ 2) (h4 : x + 4 / x^2 ≥ 3) : 
  x + n^2 / x^n ≥ n + 1 := by
  sorry

end NUMINAMATH_CALUDE_inductive_inequality_l3278_327882


namespace NUMINAMATH_CALUDE_infinite_powers_of_two_l3278_327839

/-- A sequence of natural numbers where each term is the sum of the previous term and its last digit -/
def LastDigitSequence (a₁ : ℕ) : ℕ → ℕ
  | 0 => a₁
  | n + 1 => LastDigitSequence a₁ n + (LastDigitSequence a₁ n % 10)

/-- The theorem stating that the LastDigitSequence contains infinitely many powers of 2 -/
theorem infinite_powers_of_two (a₁ : ℕ) (h : a₁ % 5 ≠ 0) :
  ∀ N : ℕ, ∃ k : ℕ, k > N ∧ ∃ m : ℕ, LastDigitSequence a₁ k = 2^m :=
sorry

end NUMINAMATH_CALUDE_infinite_powers_of_two_l3278_327839


namespace NUMINAMATH_CALUDE_other_root_of_quadratic_l3278_327875

theorem other_root_of_quadratic (a : ℝ) : 
  (2^2 + 3*2 + a = 0) → (-5^2 + 3*(-5) + a = 0) := by
sorry

end NUMINAMATH_CALUDE_other_root_of_quadratic_l3278_327875


namespace NUMINAMATH_CALUDE_completing_square_proof_l3278_327829

theorem completing_square_proof (x : ℝ) : 
  x^2 - 4*x - 3 = 0 ↔ (x - 2)^2 = 7 :=
by sorry

end NUMINAMATH_CALUDE_completing_square_proof_l3278_327829


namespace NUMINAMATH_CALUDE_square_plus_twice_a_equals_three_l3278_327843

theorem square_plus_twice_a_equals_three (a : ℝ) : 
  (∃ x : ℝ, x = -5 ∧ 2 * x + 8 = x / 5 - a) → a^2 + 2*a = 3 := by
  sorry

end NUMINAMATH_CALUDE_square_plus_twice_a_equals_three_l3278_327843


namespace NUMINAMATH_CALUDE_nickel_chocolates_l3278_327865

theorem nickel_chocolates (robert : ℕ) (nickel : ℕ) 
  (h1 : robert = 7)
  (h2 : robert = nickel + 2) : 
  nickel = 5 := by
sorry

end NUMINAMATH_CALUDE_nickel_chocolates_l3278_327865


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l3278_327802

/-- A geometric sequence with the given properties -/
structure GeometricSequence where
  a : ℕ → ℝ
  is_geometric : ∀ n, a (n + 1) / a n = a 2 / a 1
  roots_equation : a 1 ^ 2 - 10 * a 1 + 16 = 0 ∧ a 99 ^ 2 - 10 * a 99 + 16 = 0

/-- The main theorem -/
theorem geometric_sequence_product (seq : GeometricSequence) :
  seq.a 20 * seq.a 50 * seq.a 80 = 64 ∨ seq.a 20 * seq.a 50 * seq.a 80 = -64 := by
  sorry


end NUMINAMATH_CALUDE_geometric_sequence_product_l3278_327802


namespace NUMINAMATH_CALUDE_chess_tournament_participants_l3278_327877

theorem chess_tournament_participants : ∃ n : ℕ, 
  n > 0 ∧ 
  (n * (n - 1)) / 2 = 171 := by
  sorry

end NUMINAMATH_CALUDE_chess_tournament_participants_l3278_327877


namespace NUMINAMATH_CALUDE_cone_volume_ratio_cone_C_D_volume_ratio_l3278_327893

/-- The ratio of the volumes of two cones with swapped radius and height is 1/2 -/
theorem cone_volume_ratio (r h : ℝ) (hr : r > 0) (hh : h > 0) : 
  (1 / 3 * Real.pi * r^2 * h) / (1 / 3 * Real.pi * h^2 * r) = 1 / 2 := by
  sorry

/-- The ratio of the volumes of cones C and D is 1/2 -/
theorem cone_C_D_volume_ratio : 
  let r : ℝ := 16.4
  let h : ℝ := 32.8
  (1 / 3 * Real.pi * r^2 * h) / (1 / 3 * Real.pi * h^2 * r) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cone_volume_ratio_cone_C_D_volume_ratio_l3278_327893


namespace NUMINAMATH_CALUDE_factorial_8_divisors_l3278_327824

-- Define 8!
def factorial_8 : ℕ := 8*7*6*5*4*3*2*1

-- Define a function to count positive divisors
def count_positive_divisors (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem factorial_8_divisors : count_positive_divisors factorial_8 = 96 := by
  sorry

end NUMINAMATH_CALUDE_factorial_8_divisors_l3278_327824


namespace NUMINAMATH_CALUDE_product_of_fractions_equals_one_l3278_327854

theorem product_of_fractions_equals_one :
  (7 / 3) * (10 / 6) * (35 / 21) * (20 / 12) * (49 / 21) * (18 / 30) * (45 / 27) * (24 / 40) = 1 := by
  sorry

end NUMINAMATH_CALUDE_product_of_fractions_equals_one_l3278_327854


namespace NUMINAMATH_CALUDE_decimal_point_problem_l3278_327851

theorem decimal_point_problem (x : ℝ) (h1 : x > 0) (h2 : 100 * x = 9 / x) : x = 3 / 10 := by
  sorry

end NUMINAMATH_CALUDE_decimal_point_problem_l3278_327851


namespace NUMINAMATH_CALUDE_distance_calculation_l3278_327841

theorem distance_calculation (train_speed ship_speed : ℝ) (time_difference : ℝ) (distance : ℝ) : 
  train_speed = 48 →
  ship_speed = 60 →
  time_difference = 2 →
  distance / train_speed = distance / ship_speed + time_difference →
  distance = 480 := by
sorry

end NUMINAMATH_CALUDE_distance_calculation_l3278_327841


namespace NUMINAMATH_CALUDE_unique_x_for_volume_l3278_327899

/-- A function representing the volume of the rectangular prism -/
def volume (x : ℕ) : ℕ := (x + 3) * (x - 3) * (x^2 + 9)

/-- The theorem stating that there is exactly one positive integer x satisfying the conditions -/
theorem unique_x_for_volume :
  ∃! x : ℕ, x > 3 ∧ volume x < 500 :=
sorry

end NUMINAMATH_CALUDE_unique_x_for_volume_l3278_327899


namespace NUMINAMATH_CALUDE_triangle_perimeter_l3278_327872

/-- Given a triangle with inradius 5.0 cm and area 105 cm², its perimeter is 42 cm. -/
theorem triangle_perimeter (inradius : ℝ) (area : ℝ) (perimeter : ℝ) : 
  inradius = 5.0 → area = 105 → area = inradius * (perimeter / 2) → perimeter = 42 := by
  sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l3278_327872


namespace NUMINAMATH_CALUDE_fundraiser_problem_l3278_327850

/-- The fundraiser problem -/
theorem fundraiser_problem
  (total_promised : ℕ)
  (sally_owes : ℕ)
  (carl_owes : ℕ)
  (amy_owes : ℕ)
  (derek_owes : ℕ)
  (h1 : total_promised = 400)
  (h2 : sally_owes = 35)
  (h3 : carl_owes = 35)
  (h4 : amy_owes = 30)
  (h5 : derek_owes = amy_owes / 2)
  : total_promised - (sally_owes + carl_owes + amy_owes + derek_owes) = 285 := by
  sorry

#check fundraiser_problem

end NUMINAMATH_CALUDE_fundraiser_problem_l3278_327850


namespace NUMINAMATH_CALUDE_monomial_sum_l3278_327801

theorem monomial_sum (m n : ℤ) (a b : ℝ) : 
  (∀ a b : ℝ, -2 * a^2 * b^(m+1) + n * a^2 * b^4 = 0) → m + n = 5 := by
sorry

end NUMINAMATH_CALUDE_monomial_sum_l3278_327801


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l3278_327889

/-- Given a geometric sequence with positive terms and common ratio q where q^2 = 4,
    prove that (a_3 + a_4) / (a_4 + a_5) = 1/2 -/
theorem geometric_sequence_ratio (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a n > 0) →  -- All terms are positive
  (∀ n, a (n + 1) = q * a n) →  -- Common ratio is q
  q^2 = 4 →  -- Given condition
  (a 3 + a 4) / (a 4 + a 5) = 1/2 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l3278_327889


namespace NUMINAMATH_CALUDE_train_length_calculation_l3278_327800

-- Define the given constants
def bridge_crossing_time : Real := 30  -- seconds
def train_speed : Real := 45  -- km/hr
def bridge_length : Real := 230  -- meters

-- Define the theorem
theorem train_length_calculation :
  let speed_in_meters_per_second : Real := train_speed * 1000 / 3600
  let total_distance : Real := speed_in_meters_per_second * bridge_crossing_time
  let train_length : Real := total_distance - bridge_length
  train_length = 145 := by sorry

end NUMINAMATH_CALUDE_train_length_calculation_l3278_327800


namespace NUMINAMATH_CALUDE_diamond_weight_calculation_l3278_327844

/-- The weight of a single diamond in grams -/
def diamond_weight : ℝ := sorry

/-- The weight of a single jade in grams -/
def jade_weight : ℝ := sorry

/-- The total weight of 5 diamonds in grams -/
def five_diamonds_weight : ℝ := 5 * diamond_weight

theorem diamond_weight_calculation :
  (4 * diamond_weight + 2 * jade_weight = 140) →
  (jade_weight = diamond_weight + 10) →
  five_diamonds_weight = 100 := by sorry

end NUMINAMATH_CALUDE_diamond_weight_calculation_l3278_327844


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l3278_327895

theorem solution_set_of_inequality (x : ℝ) :
  (x + 3) / (2 * x - 1) < 0 ↔ -3 < x ∧ x < 1/2 :=
by sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l3278_327895


namespace NUMINAMATH_CALUDE_complex_square_simplification_l3278_327878

theorem complex_square_simplification :
  let i : ℂ := Complex.I
  (4 - 3 * i)^2 = 7 - 24 * i :=
by sorry

end NUMINAMATH_CALUDE_complex_square_simplification_l3278_327878


namespace NUMINAMATH_CALUDE_total_readers_l3278_327808

/-- The number of eBook readers Anna bought -/
def anna_readers : ℕ := 50

/-- The difference between Anna's and John's initial number of eBook readers -/
def reader_difference : ℕ := 15

/-- The number of eBook readers John lost -/
def john_lost : ℕ := 3

/-- Theorem: The total number of eBook readers John and Anna have is 82 -/
theorem total_readers : 
  anna_readers + (anna_readers - reader_difference - john_lost) = 82 := by
sorry

end NUMINAMATH_CALUDE_total_readers_l3278_327808


namespace NUMINAMATH_CALUDE_carousel_horse_ratio_l3278_327852

theorem carousel_horse_ratio : 
  ∀ (purple green gold : ℕ),
  purple > 0 →
  green = 2 * purple →
  gold = green / 6 →
  3 + purple + green + gold = 33 →
  (purple : ℚ) / 3 = 3 / 1 :=
by
  sorry

end NUMINAMATH_CALUDE_carousel_horse_ratio_l3278_327852


namespace NUMINAMATH_CALUDE_total_sides_l3278_327860

/-- The number of dice each person brought -/
def num_dice : ℕ := 4

/-- The number of sides on each die -/
def sides_per_die : ℕ := 6

/-- The number of people who brought dice -/
def num_people : ℕ := 2

/-- Theorem: The total number of sides on all dice is 48 -/
theorem total_sides : num_people * num_dice * sides_per_die = 48 := by
  sorry

end NUMINAMATH_CALUDE_total_sides_l3278_327860


namespace NUMINAMATH_CALUDE_negation_of_proposition_l3278_327837

theorem negation_of_proposition (a : ℝ) :
  (¬ ∀ x : ℝ, x ≥ 0 → x^2 - a*x + 3 > 0) ↔ (∃ x : ℝ, x ≥ 0 ∧ x^2 - a*x + 3 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l3278_327837


namespace NUMINAMATH_CALUDE_remaining_lives_total_l3278_327834

def game_scenario (initial_players : ℕ) (first_quitters : ℕ) (second_quitters : ℕ) (lives_per_player : ℕ) : ℕ :=
  (initial_players - first_quitters - second_quitters) * lives_per_player

theorem remaining_lives_total :
  game_scenario 15 5 4 7 = 42 := by
  sorry

end NUMINAMATH_CALUDE_remaining_lives_total_l3278_327834


namespace NUMINAMATH_CALUDE_trig_simplification_l3278_327898

theorem trig_simplification (x y : ℝ) :
  (Real.cos x)^2 + (Real.sin x)^2 + (Real.cos (x + y))^2 - 
  2 * (Real.cos x) * (Real.cos y) * (Real.cos (x + y)) - 
  (Real.sin x) * (Real.sin y) = (Real.sin (x - y))^2 := by
  sorry

end NUMINAMATH_CALUDE_trig_simplification_l3278_327898


namespace NUMINAMATH_CALUDE_sum_of_squares_l3278_327806

theorem sum_of_squares (x y z : ℝ) 
  (eq1 : x^2 + 4*y = 8)
  (eq2 : y^2 + 6*z = 0)
  (eq3 : z^2 + 8*x = -16) :
  x^2 + y^2 + z^2 = 21 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_l3278_327806


namespace NUMINAMATH_CALUDE_divisibility_condition_l3278_327828

theorem divisibility_condition (n : ℕ) : n ≥ 1 → (n^2 ∣ 2^n + 1) ↔ n = 1 ∨ n = 3 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_condition_l3278_327828


namespace NUMINAMATH_CALUDE_binary_110_equals_6_l3278_327859

def binary_to_decimal (b₂ b₁ b₀ : Nat) : Nat :=
  b₂ * 2^2 + b₁ * 2^1 + b₀ * 2^0

theorem binary_110_equals_6 : binary_to_decimal 1 1 0 = 6 := by
  sorry

end NUMINAMATH_CALUDE_binary_110_equals_6_l3278_327859


namespace NUMINAMATH_CALUDE_no_integer_solutions_l3278_327831

theorem no_integer_solutions : ¬∃ (k : ℕ+) (x : ℤ), 3 * (k : ℤ) * x - 18 = 5 * (k : ℤ) := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solutions_l3278_327831


namespace NUMINAMATH_CALUDE_fraction_simplification_l3278_327847

theorem fraction_simplification : 
  (2015^2 : ℚ) / (2014^2 + 2016^2 - 2) = 1/2 := by sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3278_327847


namespace NUMINAMATH_CALUDE_inscribed_cube_surface_area_l3278_327873

theorem inscribed_cube_surface_area :
  let outer_cube_edge : ℝ := 12
  let sphere_diameter : ℝ := outer_cube_edge
  let inner_cube_diagonal : ℝ := sphere_diameter
  let inner_cube_edge : ℝ := inner_cube_diagonal / Real.sqrt 3
  let inner_cube_surface_area : ℝ := 6 * inner_cube_edge ^ 2
  inner_cube_surface_area = 288 := by sorry

end NUMINAMATH_CALUDE_inscribed_cube_surface_area_l3278_327873


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l3278_327811

def U : Set Nat := {1, 2, 3, 4}
def A : Set Nat := {1, 2}
def B : Set Nat := {2, 3, 4}

theorem complement_intersection_theorem : (U \ A) ∩ B = {3, 4} := by
  sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l3278_327811


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l3278_327814

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_property 
  (a : ℕ → ℝ) 
  (h_geometric : geometric_sequence a) 
  (h_a4 : a 4 = 5) : 
  a 3 * a 5 = 25 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l3278_327814


namespace NUMINAMATH_CALUDE_fourth_column_third_row_position_l3278_327826

-- Define a type for classroom positions
def ClassroomPosition := ℕ × ℕ

-- Define a function that creates a classroom position from column and row numbers
def makePosition (column : ℕ) (row : ℕ) : ClassroomPosition := (column, row)

-- Theorem statement
theorem fourth_column_third_row_position :
  makePosition 4 3 = (4, 3) := by sorry

end NUMINAMATH_CALUDE_fourth_column_third_row_position_l3278_327826


namespace NUMINAMATH_CALUDE_circle_passes_through_fixed_point_l3278_327855

-- Define the parabola
def parabola (x y : ℝ) : Prop := x^2 = -8*y

-- Define the tangent line
def tangent_line (y : ℝ) : Prop := y = 2

-- Define the point that the circle passes through
def fixed_point : ℝ × ℝ := (0, -2)

-- Statement of the theorem
theorem circle_passes_through_fixed_point :
  ∀ (cx cy r : ℝ),
  parabola cx cy →
  (∃ (x : ℝ), tangent_line (cy + r) ∧ (x - cx)^2 + (2 - (cy + r))^2 = r^2) →
  (fixed_point.1 - cx)^2 + (fixed_point.2 - cy)^2 = r^2 := by
  sorry

end NUMINAMATH_CALUDE_circle_passes_through_fixed_point_l3278_327855


namespace NUMINAMATH_CALUDE_inequality_solution_l3278_327866

open Set

theorem inequality_solution (x : ℝ) : 
  (x^2 - 1) / (x^2 - 3*x + 2) ≥ 2 ↔ x ∈ Ioo 1 2 ∪ Ioo (3 - Real.sqrt 6) (3 + Real.sqrt 6) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l3278_327866


namespace NUMINAMATH_CALUDE_equation_has_integer_solution_l3278_327857

theorem equation_has_integer_solution : ∃ (x y : ℤ), x^2 - 2 = 7*y := by
  sorry

end NUMINAMATH_CALUDE_equation_has_integer_solution_l3278_327857


namespace NUMINAMATH_CALUDE_quadratic_monotonic_condition_l3278_327863

-- Define the quadratic function
def f (t : ℝ) (x : ℝ) : ℝ := x^2 - 2*t*x + 1

-- Define monotonicity in an interval
def monotonic_in_interval (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a < x ∧ x < y ∧ y < b → (f x < f y ∨ f y < f x)

-- Theorem statement
theorem quadratic_monotonic_condition (t : ℝ) :
  monotonic_in_interval (f t) 1 3 → t ≤ 1 ∨ t ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_monotonic_condition_l3278_327863


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3278_327804

/-- Given a complex number z and a real number a satisfying the equation (2+i)z = a+2i,
    where the real part of z is twice its imaginary part, prove that a = 3/2. -/
theorem complex_equation_solution (z : ℂ) (a : ℝ) 
    (h1 : (2 + Complex.I) * z = a + 2 * Complex.I)
    (h2 : z.re = 2 * z.im) : 
  a = 3/2 := by sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3278_327804


namespace NUMINAMATH_CALUDE_A_gives_B_150m_start_l3278_327813

-- Define the speeds of runners A, B, and C
variable (Va Vb Vc : ℝ)

-- Define the conditions
def A_gives_C_300m_start : Prop := Va / Vc = 1000 / 700
def B_gives_C_176_47m_start : Prop := Vb / Vc = 1000 / 823.53

-- Define the theorem
theorem A_gives_B_150m_start 
  (h1 : A_gives_C_300m_start Va Vc) 
  (h2 : B_gives_C_176_47m_start Vb Vc) : 
  Va / Vb = 1000 / 850 := by sorry

end NUMINAMATH_CALUDE_A_gives_B_150m_start_l3278_327813


namespace NUMINAMATH_CALUDE_S_intersect_T_eq_T_l3278_327880

-- Define the sets S and T
def S : Set ℝ := {y | ∃ x, y = 3*x + 2}
def T : Set ℝ := {y | ∃ x, y = x^2 - 1}

-- Statement to prove
theorem S_intersect_T_eq_T : S ∩ T = T := by
  sorry

end NUMINAMATH_CALUDE_S_intersect_T_eq_T_l3278_327880


namespace NUMINAMATH_CALUDE_waste_processing_growth_equation_l3278_327874

/-- Represents the growth of processing capacity over two months -/
def processing_capacity_growth (initial_capacity : ℝ) (final_capacity : ℝ) (growth_rate : ℝ) : Prop :=
  initial_capacity * (1 + growth_rate)^2 = final_capacity

/-- The equation correctly models the company's waste processing capacity growth -/
theorem waste_processing_growth_equation :
  processing_capacity_growth 1000 1200 x ↔ 1000 * (1 + x)^2 = 1200 :=
by
  sorry

end NUMINAMATH_CALUDE_waste_processing_growth_equation_l3278_327874


namespace NUMINAMATH_CALUDE_pizza_slices_l3278_327896

theorem pizza_slices (num_pizzas : ℕ) (slices_per_pizza : ℕ) (h1 : num_pizzas = 7) (h2 : slices_per_pizza = 2) :
  num_pizzas * slices_per_pizza = 14 := by
  sorry

end NUMINAMATH_CALUDE_pizza_slices_l3278_327896


namespace NUMINAMATH_CALUDE_sqrt_18_times_sqrt_32_l3278_327897

theorem sqrt_18_times_sqrt_32 : Real.sqrt 18 * Real.sqrt 32 = 24 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_18_times_sqrt_32_l3278_327897


namespace NUMINAMATH_CALUDE_removal_process_theorem_l3278_327805

/-- Represents the removal process on a circle of numbered pieces. -/
def RemovalProcess (n : ℕ) : Type :=
  { remaining : Finset ℕ // remaining.card ≤ n }

/-- The initial state of the circle with n pieces. -/
def initialCircle (n : ℕ) : RemovalProcess n :=
  ⟨Finset.range n, by simp⟩

/-- Performs the removal process on the circle. -/
def performRemoval (n : ℕ) : RemovalProcess n → RemovalProcess n :=
  sorry -- Actual implementation would go here

/-- The final state after the removal process. -/
def finalState (n : ℕ) : RemovalProcess n :=
  performRemoval n (initialCircle n)

/-- The number of pieces remaining after the removal process. -/
def remainingCount (n : ℕ) : ℕ :=
  (finalState n).val.card

/-- The kth remaining piece after the removal process. -/
def kthRemainingPiece (n k : ℕ) : ℕ :=
  sorry -- Actual implementation would go here

theorem removal_process_theorem (n : ℕ) (h : n = 3000) :
  remainingCount n = 1333 ∧ kthRemainingPiece n 181 = 407 := by
  sorry

#check removal_process_theorem

end NUMINAMATH_CALUDE_removal_process_theorem_l3278_327805


namespace NUMINAMATH_CALUDE_mason_daily_water_l3278_327809

/-- The number of cups of water Theo drinks per day -/
def theo_daily : ℕ := 8

/-- The number of cups of water Roxy drinks per day -/
def roxy_daily : ℕ := 9

/-- The total number of cups of water the siblings drink in one week -/
def total_weekly : ℕ := 168

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- Proves that Mason drinks 7 cups of water every day -/
theorem mason_daily_water : ℕ := by
  sorry

end NUMINAMATH_CALUDE_mason_daily_water_l3278_327809


namespace NUMINAMATH_CALUDE_inequality_equivalence_l3278_327822

theorem inequality_equivalence (x : ℝ) : 
  -1 < (x^2 - 12*x + 35) / (x^2 - 4*x + 8) ∧ 
  (x^2 - 12*x + 35) / (x^2 - 4*x + 8) < 1 ↔ 
  x > 27/8 := by
  sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l3278_327822


namespace NUMINAMATH_CALUDE_sample_grade12_is_40_l3278_327864

/-- Represents the stratified sampling problem for a school with three grades. -/
structure School where
  total_students : ℕ
  grade10_students : ℕ
  grade11_students : ℕ
  sample_size : ℕ

/-- Calculates the number of students to be sampled from grade 12 in a stratified sampling. -/
def sampleGrade12 (s : School) : ℕ :=
  s.sample_size - (s.grade10_students * s.sample_size / s.total_students + s.grade11_students * s.sample_size / s.total_students)

/-- Theorem stating that for the given school parameters, the number of students
    to be sampled from grade 12 is 40. -/
theorem sample_grade12_is_40 (s : School)
    (h1 : s.total_students = 2400)
    (h2 : s.grade10_students = 820)
    (h3 : s.grade11_students = 780)
    (h4 : s.sample_size = 120) :
    sampleGrade12 s = 40 := by
  sorry

end NUMINAMATH_CALUDE_sample_grade12_is_40_l3278_327864


namespace NUMINAMATH_CALUDE_root_equation_and_product_l3278_327848

theorem root_equation_and_product (a : ℝ) (x₁ x₂ : ℝ) : 
  (∀ x, x^2 + (2*a - 1)*x + a^2 = 0 ↔ x = x₁ ∨ x = x₂) →
  (x₁ + 2) * (x₂ + 2) = 11 →
  a = -1 := by
sorry

end NUMINAMATH_CALUDE_root_equation_and_product_l3278_327848


namespace NUMINAMATH_CALUDE_chips_cost_split_l3278_327812

theorem chips_cost_split (num_friends : ℕ) (num_bags : ℕ) (cost_per_bag : ℕ) :
  num_friends = 3 →
  num_bags = 5 →
  cost_per_bag = 3 →
  (num_bags * cost_per_bag) / num_friends = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_chips_cost_split_l3278_327812


namespace NUMINAMATH_CALUDE_unique_point_property_l3278_327830

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

-- Define the focus
def focus : ℝ × ℝ := (-2, 0)

-- Define the point P
def P : ℝ → ℝ × ℝ := λ p => (p, 0)

-- Define a chord passing through the focus
def chord (m : ℝ) (x : ℝ) : ℝ := m * x + 2 * m

-- Define the angle equality condition
def angle_equality (p : ℝ) : Prop :=
  ∀ m : ℝ, ∃ A B : ℝ × ℝ,
    ellipse A.1 A.2 ∧
    ellipse B.1 B.2 ∧
    A.2 = chord m A.1 ∧
    B.2 = chord m B.1 ∧
    (A.2 - 0) / (A.1 - p) = -(B.2 - 0) / (B.1 - p)

-- Theorem statement
theorem unique_point_property :
  ∃! p : ℝ, p > 0 ∧ angle_equality p :=
sorry

end NUMINAMATH_CALUDE_unique_point_property_l3278_327830


namespace NUMINAMATH_CALUDE_bush_leaves_theorem_l3278_327867

theorem bush_leaves_theorem (total_branches : ℕ) (leaves_only : ℕ) (leaves_with_flower : ℕ) : 
  total_branches = 10 →
  leaves_only = 5 →
  leaves_with_flower = 2 →
  ∀ (total_leaves : ℕ),
    (∃ (m n : ℕ), m + n = total_branches ∧ total_leaves = m * leaves_only + n * leaves_with_flower) →
    total_leaves ≠ 45 ∧ total_leaves ≠ 39 ∧ total_leaves ≠ 37 ∧ total_leaves ≠ 31 :=
by sorry

end NUMINAMATH_CALUDE_bush_leaves_theorem_l3278_327867


namespace NUMINAMATH_CALUDE_largest_integer_in_set_l3278_327883

theorem largest_integer_in_set (a b c d : ℤ) : 
  a < b ∧ b < c ∧ c < d →  -- Four different integers
  (a + b + c + d) / 4 = 68 →  -- Average is 68
  a ≥ 5 →  -- Smallest integer is at least 5
  d = 254 :=  -- Largest integer is 254
by sorry

end NUMINAMATH_CALUDE_largest_integer_in_set_l3278_327883


namespace NUMINAMATH_CALUDE_fractional_equation_elimination_l3278_327803

theorem fractional_equation_elimination (x : ℝ) : 
  (1 - (5*x + 2) / (x * (x + 1)) = 3 / (x + 1)) → 
  (x^2 - 7*x - 2 = 0) :=
by
  sorry

end NUMINAMATH_CALUDE_fractional_equation_elimination_l3278_327803


namespace NUMINAMATH_CALUDE_negative_one_less_than_abs_neg_two_fifths_l3278_327819

theorem negative_one_less_than_abs_neg_two_fifths : -1 < |-2/5| := by
  sorry

end NUMINAMATH_CALUDE_negative_one_less_than_abs_neg_two_fifths_l3278_327819
