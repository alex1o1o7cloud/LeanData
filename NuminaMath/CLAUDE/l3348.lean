import Mathlib

namespace NUMINAMATH_CALUDE_project_completion_time_l3348_334815

theorem project_completion_time (m n : ℕ) (m_pos : m > 0) (n_pos : n > 0) :
  let total_people := m + n
  let days_for_total := m
  let days_for_n := m * total_people / n
  (∀ (person : ℕ), person ≤ total_people → person > 0 → 
    (1 : ℚ) / (total_people * days_for_total : ℚ) = 
    (1 : ℚ) / (person * (total_people * days_for_total / person) : ℚ)) →
  days_for_n * n = m * total_people :=
sorry

end NUMINAMATH_CALUDE_project_completion_time_l3348_334815


namespace NUMINAMATH_CALUDE_some_number_value_l3348_334886

theorem some_number_value (x : ℝ) : 65 + 5 * 12 / (x / 3) = 66 → x = 180 := by
  sorry

end NUMINAMATH_CALUDE_some_number_value_l3348_334886


namespace NUMINAMATH_CALUDE_value_of_expression_l3348_334818

theorem value_of_expression (x y : ℚ) (hx : x = 2/3) (hy : y = 3/2) :
  (1/3) * x^5 * y^6 = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_value_of_expression_l3348_334818


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3348_334860

theorem sufficient_not_necessary_condition :
  (∀ x : ℝ, x = -2 → x^2 = 4) ∧
  ¬(∀ x : ℝ, x^2 = 4 → x = -2) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3348_334860


namespace NUMINAMATH_CALUDE_min_side_difference_l3348_334810

theorem min_side_difference (PQ PR QR : ℕ) : 
  PQ + PR + QR = 3010 →
  PQ < PR →
  PR ≤ QR →
  PQ + PR > QR →
  PQ + QR > PR →
  PR + QR > PQ →
  ∀ PQ' PR' QR' : ℕ, 
    PQ' + PR' + QR' = 3010 →
    PQ' < PR' →
    PR' ≤ QR' →
    PQ' + PR' > QR' →
    PQ' + QR' > PR' →
    PR' + QR' > PQ' →
    QR - PQ ≤ QR' - PQ' :=
by sorry

end NUMINAMATH_CALUDE_min_side_difference_l3348_334810


namespace NUMINAMATH_CALUDE_lock_combination_l3348_334839

/-- Represents a digit in the cryptarithmetic problem -/
structure Digit where
  value : Nat
  is_valid : value < 10

/-- Represents the base of the number system -/
structure Base where
  value : Nat
  is_valid : value > 1

/-- Function to convert a number from base b to base 10 -/
def to_decimal (digits : List Digit) (b : Base) : Nat :=
  sorry

/-- The cryptarithmetic equation -/
def cryptarithmetic_equation (T I D E : Digit) (b : Base) : Prop :=
  to_decimal [T, I, D, E] b + to_decimal [E, D, I, T] b + to_decimal [T, I, D, E] b
  = to_decimal [D, I, E, T] b

/-- All digits are distinct -/
def all_distinct (T I D E : Digit) : Prop :=
  T.value ≠ I.value ∧ T.value ≠ D.value ∧ T.value ≠ E.value ∧
  I.value ≠ D.value ∧ I.value ≠ E.value ∧ D.value ≠ E.value

theorem lock_combination :
  ∃ (T I D E : Digit) (b : Base),
    cryptarithmetic_equation T I D E b ∧
    all_distinct T I D E ∧
    to_decimal [T, I, D] (Base.mk 10 sorry) = 984 :=
  sorry

end NUMINAMATH_CALUDE_lock_combination_l3348_334839


namespace NUMINAMATH_CALUDE_digit_sum_l3348_334896

/-- Given digits c and d, if 5c * d4 = 1200, then c + d = 2 -/
theorem digit_sum (c d : ℕ) : 
  c < 10 → d < 10 → (50 + c) * (10 * d + 4) = 1200 → c + d = 2 := by
  sorry

end NUMINAMATH_CALUDE_digit_sum_l3348_334896


namespace NUMINAMATH_CALUDE_problem_solution_l3348_334804

-- Define the set B
def B : Set ℝ := {m | ∀ x ∈ Set.Icc (-1 : ℝ) 1, x^2 - x - m < 0}

-- Define the set A
def A (a : ℝ) : Set ℝ := {x | (x - 3*a) * (x - a - 2) < 0}

-- Main theorem
theorem problem_solution (a : ℝ) :
  (∀ x ∈ Set.Icc (-1 : ℝ) 1, x^2 - x - 2 < 0) ∧
  (a < 1) ∧
  (A a ⊆ B) ∧
  (A a ≠ B) →
  (B = Set.Ioi 2) ∧
  (2/3 ≤ a ∧ a < 1) :=
sorry

end NUMINAMATH_CALUDE_problem_solution_l3348_334804


namespace NUMINAMATH_CALUDE_common_factor_of_polynomial_l3348_334800

theorem common_factor_of_polynomial (a b : ℤ) :
  ∃ (k : ℤ), (6 * a^2 * b - 3 * a * b^2) = k * (3 * a * b) :=
by sorry

end NUMINAMATH_CALUDE_common_factor_of_polynomial_l3348_334800


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3348_334824

def A : Set (ℝ × ℝ) := {p | p.2 = 2 * p.1 + 1}
def B : Set (ℝ × ℝ) := {p | p.2 = p.1 + 3}

theorem intersection_of_A_and_B : ∃! a : ℝ × ℝ, a ∈ A ∧ a ∈ B ∧ a = (2, 5) := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3348_334824


namespace NUMINAMATH_CALUDE_cos_difference_l3348_334816

theorem cos_difference (A B : Real) 
  (h1 : Real.sin A + Real.sin B = 3/2) 
  (h2 : Real.cos A + Real.cos B = 1) : 
  Real.cos (A - B) = 5/8 := by
  sorry

end NUMINAMATH_CALUDE_cos_difference_l3348_334816


namespace NUMINAMATH_CALUDE_mike_average_weekly_time_l3348_334895

/-- Represents Mike's weekly TV and video game schedule --/
structure MikeSchedule where
  mon_wed_fri_tv : ℕ -- Hours of TV on Monday, Wednesday, Friday
  tue_thu_tv : ℕ -- Hours of TV on Tuesday, Thursday
  weekend_tv : ℕ -- Hours of TV on weekends
  vg_days : ℕ -- Number of days Mike plays video games

/-- Calculates the average weekly time Mike spends on TV and video games over 4 weeks --/
def average_weekly_time (s : MikeSchedule) : ℚ :=
  let weekly_tv := s.mon_wed_fri_tv * 3 + s.tue_thu_tv * 2 + s.weekend_tv * 2
  let daily_vg := (weekly_tv / 7 : ℚ) / 2
  let weekly_vg := daily_vg * s.vg_days
  (weekly_tv + weekly_vg) / 7

/-- Theorem stating that Mike's average weekly time spent on TV and video games is 34 hours --/
theorem mike_average_weekly_time :
  let s : MikeSchedule := { mon_wed_fri_tv := 4, tue_thu_tv := 3, weekend_tv := 5, vg_days := 3 }
  average_weekly_time s = 34 := by sorry

end NUMINAMATH_CALUDE_mike_average_weekly_time_l3348_334895


namespace NUMINAMATH_CALUDE_janet_siblings_difference_l3348_334867

/-- The number of siblings each person has -/
structure Siblings where
  masud : ℕ
  janet : ℕ
  carlos : ℕ
  stella : ℕ
  lila : ℕ

/-- The conditions of the problem -/
def problem_conditions (s : Siblings) : Prop :=
  s.masud = 45 ∧
  s.janet = 4 * s.masud - 60 ∧
  s.carlos = s.stella + 20 ∧
  s.stella = (5 * s.carlos - 16) / 2 ∧
  s.lila = s.carlos + s.stella + (s.carlos + s.stella) / 3

/-- The theorem to be proved -/
theorem janet_siblings_difference (s : Siblings) 
  (h : problem_conditions s) : 
  s.janet = s.carlos + s.stella + s.lila - 286 := by
  sorry


end NUMINAMATH_CALUDE_janet_siblings_difference_l3348_334867


namespace NUMINAMATH_CALUDE_victoria_snack_money_l3348_334823

theorem victoria_snack_money (initial_amount : ℕ) 
  (pizza_cost : ℕ) (pizza_quantity : ℕ)
  (juice_cost : ℕ) (juice_quantity : ℕ) :
  initial_amount = 50 →
  pizza_cost = 12 →
  pizza_quantity = 2 →
  juice_cost = 2 →
  juice_quantity = 2 →
  initial_amount - (pizza_cost * pizza_quantity + juice_cost * juice_quantity) = 22 := by
sorry


end NUMINAMATH_CALUDE_victoria_snack_money_l3348_334823


namespace NUMINAMATH_CALUDE_min_value_sqrt_reciprocal_min_value_achieved_l3348_334807

theorem min_value_sqrt_reciprocal (x : ℝ) (h : x > 0) : 
  3 * Real.sqrt (2 * x) + 4 / x ≥ 8 := by
  sorry

theorem min_value_achieved (x : ℝ) (h : x > 0) : 
  ∃ y > 0, 3 * Real.sqrt (2 * y) + 4 / y = 8 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sqrt_reciprocal_min_value_achieved_l3348_334807


namespace NUMINAMATH_CALUDE_prob_win_at_least_once_l3348_334893

-- Define the probability of winning a single game
def prob_win_single : ℚ := 1 / 9

-- Define the probability of losing a single game
def prob_lose_single : ℚ := 1 - prob_win_single

-- Define the number of games played
def num_games : ℕ := 3

-- Theorem statement
theorem prob_win_at_least_once :
  1 - prob_lose_single ^ num_games = 217 / 729 := by
  sorry

end NUMINAMATH_CALUDE_prob_win_at_least_once_l3348_334893


namespace NUMINAMATH_CALUDE_pams_apple_bags_l3348_334856

theorem pams_apple_bags (gerald_apples_per_bag : ℕ) (pam_total_apples : ℕ) : 
  gerald_apples_per_bag = 40 →
  pam_total_apples = 1200 →
  ∃ (pam_bags : ℕ), pam_bags * (3 * gerald_apples_per_bag) = pam_total_apples ∧ pam_bags = 10 :=
by sorry

end NUMINAMATH_CALUDE_pams_apple_bags_l3348_334856


namespace NUMINAMATH_CALUDE_number_exceeding_percentage_l3348_334842

theorem number_exceeding_percentage (x : ℝ) : x = 0.16 * x + 42 → x = 50 := by
  sorry

end NUMINAMATH_CALUDE_number_exceeding_percentage_l3348_334842


namespace NUMINAMATH_CALUDE_first_number_proof_l3348_334874

theorem first_number_proof (x : ℝ) (h : x / 14.5 = 175) : x = 2537.5 := by
  sorry

end NUMINAMATH_CALUDE_first_number_proof_l3348_334874


namespace NUMINAMATH_CALUDE_solve_for_k_l3348_334819

theorem solve_for_k (k : ℚ) : 
  (∃ x : ℚ, 3 * x + (2 * k - 1) = x - 6 * (3 * k + 2)) ∧ 
  (3 * 1 + (2 * k - 1) = 1 - 6 * (3 * k + 2)) → 
  k = -13/20 := by
sorry

end NUMINAMATH_CALUDE_solve_for_k_l3348_334819


namespace NUMINAMATH_CALUDE_infinite_solutions_condition_l3348_334880

theorem infinite_solutions_condition (b : ℝ) :
  (∀ x, 4 * (3 * x - b) = 3 * (4 * x + 16)) ↔ b = -12 := by
sorry

end NUMINAMATH_CALUDE_infinite_solutions_condition_l3348_334880


namespace NUMINAMATH_CALUDE_range_of_f_l3348_334877

def f (x : ℝ) : ℝ := x^2 - 2*x + 3

theorem range_of_f :
  ∀ y : ℝ, (∃ x : ℝ, f x = y) ↔ y ≥ 2 :=
by sorry

end NUMINAMATH_CALUDE_range_of_f_l3348_334877


namespace NUMINAMATH_CALUDE_rabbit_distribution_theorem_l3348_334865

/-- Represents the number of pet stores --/
def num_stores : ℕ := 5

/-- Represents the number of parent rabbits --/
def num_parents : ℕ := 2

/-- Represents the number of offspring rabbits --/
def num_offspring : ℕ := 4

/-- Represents the total number of rabbits --/
def total_rabbits : ℕ := num_parents + num_offspring

/-- 
Represents the number of ways to distribute rabbits to pet stores 
such that no store gets both a parent and a child
--/
def distribution_ways : ℕ := sorry

theorem rabbit_distribution_theorem : 
  distribution_ways = 560 := by sorry

end NUMINAMATH_CALUDE_rabbit_distribution_theorem_l3348_334865


namespace NUMINAMATH_CALUDE_susans_roses_l3348_334849

theorem susans_roses (D : ℚ) : 
  -- Initial number of roses is 12D
  (12 * D : ℚ) > 0 →
  -- Half given to daughter, half placed in vase
  let vase_roses := 6 * D
  -- One-third of vase flowers wilted
  let unwilted_ratio := 2 / 3
  -- 12 flowers remained after removing wilted ones
  unwilted_ratio * vase_roses = 12 →
  -- Prove that D = 3
  D = 3 := by
sorry

end NUMINAMATH_CALUDE_susans_roses_l3348_334849


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l3348_334830

-- Problem 1
theorem problem_1 : (1 - Real.sqrt 3) ^ 0 - |-(Real.sqrt 2)| + ((-27) ^ (1/3 : ℝ)) - ((-1/2) ^ (-1 : ℝ)) = -(Real.sqrt 2) := by
  sorry

-- Problem 2
theorem problem_2 (x : ℝ) (h : x ≠ 2) : 
  ((x^2 - 1) / (x - 2) - x - 1) / ((x + 1) / (x^2 - 4*x + 4)) = x - 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l3348_334830


namespace NUMINAMATH_CALUDE_negation_equivalence_l3348_334894

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x^2 - x + 1 < 0) ↔ (∀ x : ℝ, x^2 - x + 1 ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l3348_334894


namespace NUMINAMATH_CALUDE_one_in_A_l3348_334852

def A : Set ℕ := {1, 2}

theorem one_in_A : 1 ∈ A := by sorry

end NUMINAMATH_CALUDE_one_in_A_l3348_334852


namespace NUMINAMATH_CALUDE_mississippi_permutations_l3348_334850

theorem mississippi_permutations :
  let total_letters : ℕ := 11
  let m_count : ℕ := 1
  let i_count : ℕ := 4
  let s_count : ℕ := 4
  let p_count : ℕ := 2
  (Nat.factorial total_letters) / 
  (Nat.factorial m_count * Nat.factorial i_count * Nat.factorial s_count * Nat.factorial p_count) = 34650 := by
  sorry

end NUMINAMATH_CALUDE_mississippi_permutations_l3348_334850


namespace NUMINAMATH_CALUDE_nine_chapters_problem_l3348_334833

/-- Represents the worth of animals in taels of gold -/
structure AnimalWorth where
  cow : ℝ
  sheep : ℝ

/-- Represents the total worth of a group of animals -/
def groupWorth (w : AnimalWorth) (cows sheep : ℕ) : ℝ :=
  cows * w.cow + sheep * w.sheep

/-- The problem statement from "The Nine Chapters on the Mathematical Art" -/
theorem nine_chapters_problem (w : AnimalWorth) : 
  (groupWorth w 5 2 = 10 ∧ groupWorth w 2 5 = 8) ↔ 
  (5 * w.cow + 2 * w.sheep = 10 ∧ 2 * w.cow + 5 * w.sheep = 8) := by
sorry

end NUMINAMATH_CALUDE_nine_chapters_problem_l3348_334833


namespace NUMINAMATH_CALUDE_product_digit_sum_l3348_334862

def number1 : ℕ := 707070707070707070707070707070707070707070707070707070707070707070707070707070707070707070707070707070707
def number2 : ℕ := 909090909090909090909090909090909090909090909090909090909090909090909090909090909090909090909090909090909

theorem product_digit_sum :
  let product := number1 * number2
  let tens_digit := (product / 10) % 10
  let units_digit := product % 10
  tens_digit + units_digit = 9 := by sorry

end NUMINAMATH_CALUDE_product_digit_sum_l3348_334862


namespace NUMINAMATH_CALUDE_octahedron_cube_volume_ratio_l3348_334888

/-- Given a cube, an octahedron is formed by joining the centers of adjoining faces. -/
structure OctahedronFromCube where
  cube_side : ℝ
  cube_volume : ℝ
  octahedron_side : ℝ
  octahedron_volume : ℝ

/-- The ratio of the volume of the octahedron to the volume of the cube is 1/6. -/
theorem octahedron_cube_volume_ratio (o : OctahedronFromCube) :
  o.octahedron_volume / o.cube_volume = 1 / 6 := by
  sorry

end NUMINAMATH_CALUDE_octahedron_cube_volume_ratio_l3348_334888


namespace NUMINAMATH_CALUDE_train_speed_in_kmh_l3348_334889

-- Define the given parameters
def train_length : ℝ := 80
def bridge_length : ℝ := 295
def crossing_time : ℝ := 30

-- Define the conversion factor from m/s to km/hr
def conversion_factor : ℝ := 3.6

-- Theorem statement
theorem train_speed_in_kmh :
  let total_distance := train_length + bridge_length
  let speed_ms := total_distance / crossing_time
  let speed_kmh := speed_ms * conversion_factor
  speed_kmh = 45 := by sorry

end NUMINAMATH_CALUDE_train_speed_in_kmh_l3348_334889


namespace NUMINAMATH_CALUDE_f_lower_bound_f_one_less_than_two_l3348_334805

-- Define the function f
def f (x a : ℝ) : ℝ := |x - 1 + a| + |x - a|

-- Part 1
theorem f_lower_bound (x a : ℝ) (h : a ≥ 2) : f x a ≥ 3 := by
  sorry

-- Part 2
theorem f_one_less_than_two (a : ℝ) : 
  (f 1 a < 2) ↔ a ∈ Set.Ioo (-1/2 : ℝ) (3/2 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_f_lower_bound_f_one_less_than_two_l3348_334805


namespace NUMINAMATH_CALUDE_negation_of_proposition_l3348_334883

theorem negation_of_proposition :
  (¬ ∀ x : ℝ, x > 0 → x^2 + x > 0) ↔ (∃ x₀ : ℝ, x₀ > 0 ∧ x₀^2 + x₀ ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l3348_334883


namespace NUMINAMATH_CALUDE_inequality_proof_l3348_334847

theorem inequality_proof (x y z : ℝ) 
  (hx : x ≠ 1) (hy : y ≠ 1) (hz : z ≠ 1) (hxyz : x * y * z = 1) : 
  x^2 / (x - 1)^2 + y^2 / (y - 1)^2 + z^2 / (z - 1)^2 ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3348_334847


namespace NUMINAMATH_CALUDE_power_fraction_plus_two_l3348_334812

theorem power_fraction_plus_two : (5 / 3 : ℚ)^7 + 2 = 82499 / 2187 := by sorry

end NUMINAMATH_CALUDE_power_fraction_plus_two_l3348_334812


namespace NUMINAMATH_CALUDE_inequality_transformation_l3348_334875

theorem inequality_transformation (h : (1/4 : ℝ) > (1/8 : ℝ)) : (2 : ℝ) < (3 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_inequality_transformation_l3348_334875


namespace NUMINAMATH_CALUDE_fib_50_mod_5_l3348_334858

def fibonacci : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fibonacci n + fibonacci (n + 1)

theorem fib_50_mod_5 : fibonacci 50 % 5 = 0 := by
  sorry

end NUMINAMATH_CALUDE_fib_50_mod_5_l3348_334858


namespace NUMINAMATH_CALUDE_polynomial_existence_l3348_334813

theorem polynomial_existence (n : ℕ+) :
  ∃ (f g : Polynomial ℤ), (f * (X + 1) ^ (2 ^ n.val) + g * (X ^ (2 ^ n.val) + 1)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_existence_l3348_334813


namespace NUMINAMATH_CALUDE_math_score_calculation_l3348_334811

theorem math_score_calculation (total_subjects : ℕ) (avg_without_math : ℝ) (avg_with_math : ℝ) :
  total_subjects = 5 →
  avg_without_math = 88 →
  avg_with_math = 92 →
  (total_subjects - 1) * avg_without_math + (avg_with_math * total_subjects - (total_subjects - 1) * avg_without_math) = 108 :=
by sorry

end NUMINAMATH_CALUDE_math_score_calculation_l3348_334811


namespace NUMINAMATH_CALUDE_cashier_money_value_l3348_334857

theorem cashier_money_value (total_bills : ℕ) (five_dollar_bills : ℕ) : 
  total_bills = 126 →
  five_dollar_bills = 84 →
  (total_bills - five_dollar_bills) * 10 + five_dollar_bills * 5 = 840 :=
by sorry

end NUMINAMATH_CALUDE_cashier_money_value_l3348_334857


namespace NUMINAMATH_CALUDE_fourth_altitude_is_six_times_radius_l3348_334844

/-- A tetrahedron with an inscribed sphere -/
structure Tetrahedron :=
  (r : ℝ)  -- radius of the inscribed sphere
  (h₁ h₂ h₃ h₄ : ℝ)  -- altitudes of the tetrahedron
  (h₁_eq : h₁ = 3 * r)
  (h₂_eq : h₂ = 4 * r)
  (h₃_eq : h₃ = 4 * r)
  (sum_reciprocals : 1 / h₁ + 1 / h₂ + 1 / h₃ + 1 / h₄ = 1 / r)

/-- The fourth altitude of the tetrahedron is 6 times the radius of its inscribed sphere -/
theorem fourth_altitude_is_six_times_radius (T : Tetrahedron) : T.h₄ = 6 * T.r := by
  sorry

end NUMINAMATH_CALUDE_fourth_altitude_is_six_times_radius_l3348_334844


namespace NUMINAMATH_CALUDE_parabola_translation_l3348_334801

/-- Represents a parabola in the form y = ax² + bx + c --/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Translates a parabola horizontally and vertically --/
def translate (p : Parabola) (h : ℝ) (k : ℝ) : Parabola :=
  { a := p.a
    b := -2 * p.a * h + p.b
    c := p.a * h^2 - p.b * h + p.c + k }

theorem parabola_translation :
  let original := Parabola.mk 1 0 0  -- y = x²
  let translated := translate original (-1) (-2)  -- 1 unit left, 2 units down
  translated = Parabola.mk 1 2 (-2)  -- y = (x+1)² - 2
  := by sorry

end NUMINAMATH_CALUDE_parabola_translation_l3348_334801


namespace NUMINAMATH_CALUDE_point_translation_l3348_334838

def translate_point (x y dx dy : ℝ) : ℝ × ℝ :=
  (x + dx, y - dy)

theorem point_translation :
  let initial_point : ℝ × ℝ := (1, 2)
  let right_translation : ℝ := 1
  let down_translation : ℝ := 3
  translate_point initial_point.1 initial_point.2 right_translation down_translation = (2, -1) := by
sorry

end NUMINAMATH_CALUDE_point_translation_l3348_334838


namespace NUMINAMATH_CALUDE_first_black_ace_most_likely_at_first_position_l3348_334872

/-- Probability of drawing the first black ace at position k in a shuffled 52-card deck --/
def probability_first_black_ace (k : ℕ) : ℚ :=
  if k ≥ 1 ∧ k ≤ 51 then (52 - k : ℚ) / 1326 else 0

/-- The position where the probability of drawing the first black ace is maximized --/
def max_probability_position : ℕ := 1

/-- Theorem stating that the probability of drawing the first black ace is maximized at position 1 --/
theorem first_black_ace_most_likely_at_first_position :
  ∀ k, k ≥ 1 → k ≤ 51 → probability_first_black_ace max_probability_position ≥ probability_first_black_ace k :=
by
  sorry


end NUMINAMATH_CALUDE_first_black_ace_most_likely_at_first_position_l3348_334872


namespace NUMINAMATH_CALUDE_inequality_constraint_l3348_334851

theorem inequality_constraint (a : ℝ) : 
  (∀ x : ℝ, x ≥ 1 → x^2 + a*x + 9 ≥ 0) ↔ a ≥ -6 :=
by sorry

end NUMINAMATH_CALUDE_inequality_constraint_l3348_334851


namespace NUMINAMATH_CALUDE_corner_sum_possibilities_l3348_334831

/-- Represents the color of a cell on the board -/
inductive CellColor
| Gold
| Silver

/-- Represents the board configuration -/
structure Board :=
  (rows : Nat)
  (cols : Nat)
  (cellColor : Nat → Nat → CellColor)
  (vertexValue : Nat → Nat → Fin 2)

/-- Checks if a cell satisfies the sum condition based on its color -/
def validCell (b : Board) (row col : Nat) : Prop :=
  let sum := b.vertexValue row col + b.vertexValue row (col+1) +
             b.vertexValue (row+1) col + b.vertexValue (row+1) (col+1)
  match b.cellColor row col with
  | CellColor.Gold => sum % 2 = 0
  | CellColor.Silver => sum % 2 = 1

/-- Checks if the entire board configuration is valid -/
def validBoard (b : Board) : Prop :=
  b.rows = 2016 ∧ b.cols = 2017 ∧
  (∀ row col, row < b.rows → col < b.cols → validCell b row col) ∧
  (∀ row col, (row + col) % 2 = 0 → b.cellColor row col = CellColor.Gold) ∧
  (∀ row col, (row + col) % 2 = 1 → b.cellColor row col = CellColor.Silver)

/-- The sum of the four corner vertices of the board -/
def cornerSum (b : Board) : Nat :=
  b.vertexValue 0 0 + b.vertexValue 0 b.cols +
  b.vertexValue b.rows 0 + b.vertexValue b.rows b.cols

/-- Theorem stating the possible sums of the four corner vertices -/
theorem corner_sum_possibilities (b : Board) (h : validBoard b) :
  cornerSum b = 0 ∨ cornerSum b = 2 ∨ cornerSum b = 4 := by
  sorry

end NUMINAMATH_CALUDE_corner_sum_possibilities_l3348_334831


namespace NUMINAMATH_CALUDE_soda_transaction_result_l3348_334827

def soda_transaction (initial_cans : ℕ) : ℕ × ℕ :=
  let jeff_takes := 6
  let jeff_returns := jeff_takes / 2
  let after_jeff := initial_cans - jeff_takes + jeff_returns
  let tim_buys := after_jeff / 3
  let store_bonus := tim_buys / 4
  let after_store := after_jeff + tim_buys + store_bonus
  let sarah_takes := after_store / 5
  let end_of_day := after_store - sarah_takes
  let sarah_returns := sarah_takes * 2
  let next_day := end_of_day + sarah_returns
  (end_of_day, next_day)

theorem soda_transaction_result :
  soda_transaction 22 = (21, 31) := by sorry

end NUMINAMATH_CALUDE_soda_transaction_result_l3348_334827


namespace NUMINAMATH_CALUDE_chernomor_max_coins_l3348_334898

/-- Represents the problem of distributing coins among bogatyrs --/
structure BogatyrProblem where
  total_bogatyrs : Nat
  total_coins : Nat

/-- Represents a distribution of bogatyrs into groups --/
structure Distribution where
  groups : List Nat
  coins_per_group : List Nat

/-- Calculates the remainder for Chernomor given a distribution --/
def remainder (d : Distribution) : Nat :=
  d.groups.zip d.coins_per_group
    |> List.map (fun (g, c) => c % g)
    |> List.sum

/-- The maximum remainder Chernomor can get with arbitrary distribution --/
def max_remainder_arbitrary (p : BogatyrProblem) : Nat :=
  sorry

/-- The maximum remainder Chernomor can get with equal distribution --/
def max_remainder_equal (p : BogatyrProblem) : Nat :=
  sorry

theorem chernomor_max_coins (p : BogatyrProblem) 
  (h1 : p.total_bogatyrs = 33) (h2 : p.total_coins = 240) : 
  max_remainder_arbitrary p = 31 ∧ max_remainder_equal p = 30 := by
  sorry

end NUMINAMATH_CALUDE_chernomor_max_coins_l3348_334898


namespace NUMINAMATH_CALUDE_bridget_apples_l3348_334863

theorem bridget_apples (x : ℕ) : 
  (x / 3 : ℚ) + 5 + 2 + 8 = x → x = 30 :=
by sorry

end NUMINAMATH_CALUDE_bridget_apples_l3348_334863


namespace NUMINAMATH_CALUDE_zachary_crunches_pushups_difference_l3348_334868

/-- Zachary's push-ups -/
def zachary_pushups : ℕ := 46

/-- Zachary's crunches -/
def zachary_crunches : ℕ := 58

/-- David's push-ups in terms of Zachary's -/
def david_pushups : ℕ := zachary_pushups + 38

/-- David's crunches in terms of Zachary's -/
def david_crunches : ℕ := zachary_crunches - 62

/-- Theorem stating the difference between Zachary's crunches and push-ups -/
theorem zachary_crunches_pushups_difference :
  zachary_crunches - zachary_pushups = 12 := by sorry

end NUMINAMATH_CALUDE_zachary_crunches_pushups_difference_l3348_334868


namespace NUMINAMATH_CALUDE_two_valid_M_values_l3348_334836

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def reverse_digits (n : ℕ) : ℕ :=
  (n % 10) * 10 + (n / 10)

def is_perfect_cube (n : ℕ) : Prop :=
  ∃ k : ℕ, k^3 = n

theorem two_valid_M_values :
  ∃! (s : Finset ℕ), 
    (∀ M ∈ s, is_two_digit M ∧ 
      let diff := M - reverse_digits M
      diff > 0 ∧ 
      is_perfect_cube diff ∧ 
      27 < diff ∧ 
      diff < 100) ∧
    s.card = 2 := by sorry

end NUMINAMATH_CALUDE_two_valid_M_values_l3348_334836


namespace NUMINAMATH_CALUDE_inequality_theorem_l3348_334806

theorem inequality_theorem (a b c d : ℝ) : 
  0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  ((a < c ∧ c < b) ∨ (a < d ∧ d < b) ∨ (c < a ∧ a < d) ∨ (c < b ∧ b < d)) →
  Real.sqrt ((a + b) * (c + d)) > Real.sqrt (a * b) + Real.sqrt (c * d) :=
by sorry

end NUMINAMATH_CALUDE_inequality_theorem_l3348_334806


namespace NUMINAMATH_CALUDE_max_b_value_l3348_334871

theorem max_b_value (x b : ℤ) : 
  x^2 + b*x = -20 → 
  b > 0 → 
  ∃ (y : ℤ), x^2 + y*x = -20 ∧ y > 0 → 
  y ≤ 21 :=
sorry

end NUMINAMATH_CALUDE_max_b_value_l3348_334871


namespace NUMINAMATH_CALUDE_rectangle_area_l3348_334897

theorem rectangle_area (x y : ℝ) (h_perimeter : x + y = 6) (h_diagonal : x^2 + y^2 = 25) :
  x * y = 5.5 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l3348_334897


namespace NUMINAMATH_CALUDE_triangle_side_length_l3348_334859

theorem triangle_side_length (a : ℝ) : 
  (5 : ℝ) > 0 ∧ (8 : ℝ) > 0 ∧ a > 0 →
  (5 + 8 > a ∧ 5 + a > 8 ∧ 8 + a > 5) ↔ (3 < a ∧ a < 13) :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l3348_334859


namespace NUMINAMATH_CALUDE_f_max_value_l3348_334803

def f (x : ℝ) := x^3 - x^2 - x + 2

theorem f_max_value :
  (∃ x, f x = 1 ∧ ∀ y, f y ≥ f x) →
  (∃ x, f x = 59/27 ∧ ∀ y, f y ≤ f x) :=
by sorry

end NUMINAMATH_CALUDE_f_max_value_l3348_334803


namespace NUMINAMATH_CALUDE_water_evaporation_per_day_l3348_334878

def initial_water : ℝ := 10
def evaporation_period : ℕ := 20
def evaporation_percentage : ℝ := 0.12

theorem water_evaporation_per_day :
  let total_evaporated := initial_water * evaporation_percentage
  let daily_evaporation := total_evaporated / evaporation_period
  daily_evaporation = 0.06 := by sorry

end NUMINAMATH_CALUDE_water_evaporation_per_day_l3348_334878


namespace NUMINAMATH_CALUDE_linear_equation_solution_l3348_334861

theorem linear_equation_solution (x y : ℝ) : 3 * x + y = 1 → y = -3 * x + 1 := by
  sorry

end NUMINAMATH_CALUDE_linear_equation_solution_l3348_334861


namespace NUMINAMATH_CALUDE_cubic_integer_root_l3348_334825

theorem cubic_integer_root 
  (b c : ℚ) 
  (h1 : ∃ x : ℤ, x^3 + b*x + c = 0) 
  (h2 : (5 - Real.sqrt 11)^3 + b*(5 - Real.sqrt 11) + c = 0) : 
  ∃ x : ℤ, x^3 + b*x + c = 0 ∧ x = -10 := by
sorry

end NUMINAMATH_CALUDE_cubic_integer_root_l3348_334825


namespace NUMINAMATH_CALUDE_total_cost_theorem_l3348_334841

def cost_of_meat (pork_price chicken_price pork_weight chicken_weight : ℝ) : ℝ :=
  pork_price * pork_weight + chicken_price * chicken_weight

theorem total_cost_theorem (pork_price : ℝ) (h1 : pork_price = 6) :
  let chicken_price := pork_price - 2
  cost_of_meat pork_price chicken_price 1 3 = 18 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_theorem_l3348_334841


namespace NUMINAMATH_CALUDE_trapezoid_ratio_theorem_l3348_334866

/-- Represents a trapezoid with bases and a point inside it -/
structure Trapezoid where
  EF : ℝ
  GH : ℝ
  isIsosceles : Bool
  EFGreaterGH : EF > GH

/-- Represents the areas of triangles formed by dividing a trapezoid -/
structure TriangleAreas where
  area1 : ℝ
  area2 : ℝ
  area3 : ℝ
  area4 : ℝ

/-- Theorem stating the ratio of bases in a trapezoid given specific triangle areas -/
theorem trapezoid_ratio_theorem (T : Trapezoid) (A : TriangleAreas) :
  T.isIsosceles = true ∧
  A.area1 = 3 ∧ A.area2 = 4 ∧ A.area3 = 6 ∧ A.area4 = 7 →
  T.EF / T.GH = 7 / 3 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_ratio_theorem_l3348_334866


namespace NUMINAMATH_CALUDE_jerrys_age_l3348_334835

theorem jerrys_age (mickey_age jerry_age : ℕ) : 
  mickey_age = 30 →
  mickey_age = 4 * jerry_age + 10 →
  jerry_age = 5 := by
sorry

end NUMINAMATH_CALUDE_jerrys_age_l3348_334835


namespace NUMINAMATH_CALUDE_fifty_cent_items_count_l3348_334848

/-- Represents the number of items at each price point -/
structure ItemCounts where
  cents50 : ℕ
  dollars2 : ℕ
  dollars4 : ℕ

/-- The proposition that the given item counts satisfy the problem conditions -/
def satisfiesConditions (counts : ItemCounts) : Prop :=
  counts.cents50 + counts.dollars2 + counts.dollars4 = 50 ∧
  50 * counts.cents50 + 200 * counts.dollars2 + 400 * counts.dollars4 = 5000

/-- The theorem stating that the only solution satisfying the conditions has 36 items at 50 cents -/
theorem fifty_cent_items_count :
  ∀ counts : ItemCounts, satisfiesConditions counts → counts.cents50 = 36 := by
  sorry

end NUMINAMATH_CALUDE_fifty_cent_items_count_l3348_334848


namespace NUMINAMATH_CALUDE_complement_union_problem_l3348_334837

def U : Set ℕ := {0, 1, 2, 3, 4}
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {2, 4}

theorem complement_union_problem : (U \ A) ∪ B = {0, 2, 4} := by
  sorry

end NUMINAMATH_CALUDE_complement_union_problem_l3348_334837


namespace NUMINAMATH_CALUDE_problem_statement_l3348_334899

theorem problem_statement (x : ℝ) (h1 : x > 0) (h2 : x + 1/x = 152) :
  Real.sqrt x + 1 / Real.sqrt x = Real.sqrt 154 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3348_334899


namespace NUMINAMATH_CALUDE_interest_rate_calculation_l3348_334829

theorem interest_rate_calculation (total_sum : ℝ) (second_part : ℝ) : 
  total_sum = 2665 →
  second_part = 1332.5 →
  let first_part := total_sum - second_part
  let interest_first := first_part * 0.03 * 5
  let interest_second := second_part * 0.03 * 3 * (5 : ℝ) / 3
  interest_first = interest_second →
  5 = 100 * interest_second / (second_part * 3) := by
sorry

end NUMINAMATH_CALUDE_interest_rate_calculation_l3348_334829


namespace NUMINAMATH_CALUDE_least_multiple_36_with_digit_sum_multiple_9_l3348_334840

def digit_sum (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + digit_sum (n / 10)

theorem least_multiple_36_with_digit_sum_multiple_9 :
  ∀ k : ℕ, k > 0 → 36 * k ≠ 36 →
    digit_sum (36 * k) % 9 = 0 → digit_sum 36 % 9 = 0 ∧ 36 < 36 * k :=
by sorry

end NUMINAMATH_CALUDE_least_multiple_36_with_digit_sum_multiple_9_l3348_334840


namespace NUMINAMATH_CALUDE_arccos_one_over_sqrt_two_l3348_334890

theorem arccos_one_over_sqrt_two (π : ℝ) : Real.arccos (1 / Real.sqrt 2) = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_arccos_one_over_sqrt_two_l3348_334890


namespace NUMINAMATH_CALUDE_complement_N_subset_M_l3348_334873

open Set

-- Define the sets M and N
def M : Set ℝ := {x | x * (x - 3) < 0}
def N : Set ℝ := {x | x < 1 ∨ x ≥ 3}

-- State the theorem
theorem complement_N_subset_M : (𝒰 \ N) ⊆ M := by
  sorry

end NUMINAMATH_CALUDE_complement_N_subset_M_l3348_334873


namespace NUMINAMATH_CALUDE_triangle_perimeter_sum_l3348_334826

theorem triangle_perimeter_sum : 
  ∀ (a b c d e : ℝ),
  a = 6 ∧ b = 8 ∧ 
  c^2 + d^2 = e^2 ∧
  (1/2) * c * d = (1/2) * (1/2) * a * b →
  a + b + (a^2 + b^2).sqrt + c + d + e = 24 + 6 * Real.sqrt 3 + 2 * Real.sqrt 15 :=
by sorry

end NUMINAMATH_CALUDE_triangle_perimeter_sum_l3348_334826


namespace NUMINAMATH_CALUDE_tempo_original_value_l3348_334820

theorem tempo_original_value (insurance_ratio : ℚ) (premium_rate : ℚ) (premium : ℚ) :
  insurance_ratio = 5 / 7 →
  premium_rate = 3 / 100 →
  premium = 300 →
  ∃ (original_value : ℚ), original_value = 14000 ∧ 
    premium = premium_rate * insurance_ratio * original_value :=
by sorry

end NUMINAMATH_CALUDE_tempo_original_value_l3348_334820


namespace NUMINAMATH_CALUDE_geometric_sequence_second_term_l3348_334870

/-- For a geometric sequence with common ratio 2 and sum of first 3 terms 34685, the second term is 9910 -/
theorem geometric_sequence_second_term : ∀ (a : ℕ → ℚ), 
  (∀ n, a (n + 1) = 2 * a n) →  -- geometric sequence with common ratio 2
  (a 1 + a 2 + a 3 = 34685) →   -- sum of first 3 terms is 34685
  a 2 = 9910 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_second_term_l3348_334870


namespace NUMINAMATH_CALUDE_max_ski_trips_l3348_334891

/-- Proves the maximum number of ski trips in a given time --/
theorem max_ski_trips (lift_time ski_time total_time : ℕ) : 
  lift_time = 15 →
  ski_time = 5 →
  total_time = 120 →
  (total_time / (lift_time + ski_time) : ℕ) = 6 := by
  sorry

#check max_ski_trips

end NUMINAMATH_CALUDE_max_ski_trips_l3348_334891


namespace NUMINAMATH_CALUDE_nancys_water_intake_l3348_334864

/-- Nancy's weight in pounds -/
def nancys_weight : ℝ := 90

/-- The percentage of body weight Nancy drinks in water -/
def water_percentage : ℝ := 0.6

/-- The amount of water Nancy drinks daily in pounds -/
def water_intake : ℝ := nancys_weight * water_percentage

theorem nancys_water_intake : water_intake = 54 := by
  sorry

end NUMINAMATH_CALUDE_nancys_water_intake_l3348_334864


namespace NUMINAMATH_CALUDE_special_triangle_rs_distance_l3348_334846

/-- Triangle ABC with altitude CH and inscribed circles in ACH and BCH -/
structure SpecialTriangle where
  -- Points
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  H : ℝ × ℝ
  R : ℝ × ℝ
  S : ℝ × ℝ
  -- CH is an altitude
  altitude : (C.1 - H.1) * (B.1 - A.1) + (C.2 - H.2) * (B.2 - A.2) = 0
  -- R is on CH
  r_on_ch : ∃ t : ℝ, R = (C.1 + t * (H.1 - C.1), C.2 + t * (H.2 - C.2))
  -- S is on CH
  s_on_ch : ∃ t : ℝ, S = (C.1 + t * (H.1 - C.1), C.2 + t * (H.2 - C.2))
  -- Side lengths
  ab_length : Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 2000
  ac_length : Real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2) = 1997
  bc_length : Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2) = 1998

/-- The distance between R and S is 2001/4000 -/
theorem special_triangle_rs_distance (t : SpecialTriangle) :
  Real.sqrt ((t.R.1 - t.S.1)^2 + (t.R.2 - t.S.2)^2) = 2001 / 4000 := by
  sorry

end NUMINAMATH_CALUDE_special_triangle_rs_distance_l3348_334846


namespace NUMINAMATH_CALUDE_geometric_sequence_third_term_l3348_334853

/-- If 1, 3, and x form a geometric sequence, then x = 9 -/
theorem geometric_sequence_third_term (x : ℝ) : 
  (∃ r : ℝ, r ≠ 0 ∧ 3 = 1 * r ∧ x = 3 * r) → x = 9 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_third_term_l3348_334853


namespace NUMINAMATH_CALUDE_universal_transportation_method_l3348_334869

-- Define a type for cities
variable {City : Type}

-- Define a relation for connectivity between cities
variable (connected : City → City → Prop)

-- Define air and water connectivity
variable (air_connected : City → City → Prop)
variable (water_connected : City → City → Prop)

-- Axiom: Any two cities are connected by either air or water
axiom connectivity : ∀ (c1 c2 : City), c1 ≠ c2 → air_connected c1 c2 ∨ water_connected c1 c2

-- Define the theorem
theorem universal_transportation_method 
  (h : ∀ (c1 c2 : City), connected c1 c2 ↔ (air_connected c1 c2 ∨ water_connected c1 c2)) :
  (∀ (c1 c2 : City), air_connected c1 c2) ∨ (∀ (c1 c2 : City), water_connected c1 c2) :=
sorry

end NUMINAMATH_CALUDE_universal_transportation_method_l3348_334869


namespace NUMINAMATH_CALUDE_apple_cost_problem_l3348_334808

theorem apple_cost_problem (l q : ℝ) : 
  (30 * l + 3 * q = 168) →
  (30 * l + 6 * q = 186) →
  (∀ k, k ≤ 30 → k * l = k * 5) →
  20 * l = 100 := by
sorry

end NUMINAMATH_CALUDE_apple_cost_problem_l3348_334808


namespace NUMINAMATH_CALUDE_male_alligators_mating_season_l3348_334882

/-- Represents the alligator population on Lagoon Island -/
structure AlligatorPopulation where
  males : ℕ
  adult_females : ℕ
  juvenile_females : ℕ

/-- The ratio of males to adult females to juvenile females -/
def population_ratio : AlligatorPopulation := ⟨2, 3, 5⟩

/-- The number of adult females during non-mating season -/
def non_mating_adult_females : ℕ := 15

/-- Theorem stating the number of male alligators during mating season -/
theorem male_alligators_mating_season :
  ∃ (pop : AlligatorPopulation),
    pop.adult_females = non_mating_adult_females ∧
    pop.males * population_ratio.adult_females = population_ratio.males * pop.adult_females ∧
    pop.males = 10 :=
by sorry

end NUMINAMATH_CALUDE_male_alligators_mating_season_l3348_334882


namespace NUMINAMATH_CALUDE_second_train_speed_l3348_334855

/-- Given two trains starting from the same station, traveling in the same direction for 10 hours,
    with the first train moving at 10 mph and the distance between them after 10 hours being 250 miles,
    prove that the speed of the second train is 35 mph. -/
theorem second_train_speed (first_train_speed : ℝ) (time : ℝ) (distance_between : ℝ) :
  first_train_speed = 10 →
  time = 10 →
  distance_between = 250 →
  ∃ second_train_speed : ℝ,
    second_train_speed * time - first_train_speed * time = distance_between ∧
    second_train_speed = 35 := by
  sorry

end NUMINAMATH_CALUDE_second_train_speed_l3348_334855


namespace NUMINAMATH_CALUDE_sum_in_base8_l3348_334802

/-- Converts a base-8 number to base-10 --/
def base8_to_base10 (n : ℕ) : ℕ := sorry

/-- Converts a base-10 number to base-8 --/
def base10_to_base8 (n : ℕ) : ℕ := sorry

theorem sum_in_base8 :
  let a := 53
  let b := 27
  let sum := base10_to_base8 (base8_to_base10 a + base8_to_base10 b)
  sum = 102 := by sorry

end NUMINAMATH_CALUDE_sum_in_base8_l3348_334802


namespace NUMINAMATH_CALUDE_midpoint_distance_to_y_axis_l3348_334809

-- Define the parabola
def parabola (p : ℝ × ℝ) : Prop := p.2^2 = 4 * p.1

-- Define the focus
def focus : ℝ × ℝ := (1, 0)

-- Define the property of line m passing through focus and intersecting the parabola
def line_intersects_parabola (A B : ℝ × ℝ) : Prop :=
  parabola A ∧ parabola B ∧ 
  (A.1 - focus.1) * (B.2 - focus.2) = (B.1 - focus.1) * (A.2 - focus.2)

-- Define the condition |AF| + |BF| = 10
def distance_sum_condition (A B : ℝ × ℝ) : Prop :=
  Real.sqrt ((A.1 - focus.1)^2 + (A.2 - focus.2)^2) +
  Real.sqrt ((B.1 - focus.1)^2 + (B.2 - focus.2)^2) = 10

-- Theorem statement
theorem midpoint_distance_to_y_axis 
  (A B : ℝ × ℝ) 
  (h1 : line_intersects_parabola A B) 
  (h2 : distance_sum_condition A B) : 
  (A.1 + B.1) / 2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_midpoint_distance_to_y_axis_l3348_334809


namespace NUMINAMATH_CALUDE_f_properties_l3348_334843

def f (x : ℝ) : ℝ := |x + 3| + |x - 2|

theorem f_properties :
  (∀ x, f x > 7 ↔ x < -4 ∨ x > 3) ∧
  (∀ m, m > 1 → ∃ x, f x = 4 / (m - 1) + m) := by sorry

end NUMINAMATH_CALUDE_f_properties_l3348_334843


namespace NUMINAMATH_CALUDE_fraction_value_l3348_334845

theorem fraction_value (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (h : 1/a + 1/b = 3) :
  (a + 2*a*b + b) / (2*a*b - a - b) = -5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_value_l3348_334845


namespace NUMINAMATH_CALUDE_ice_cream_pudding_cost_difference_l3348_334817

-- Define the quantities and prices
def ice_cream_quantity : ℕ := 15
def pudding_quantity : ℕ := 5
def ice_cream_price : ℕ := 5
def pudding_price : ℕ := 2

-- Define the theorem
theorem ice_cream_pudding_cost_difference :
  (ice_cream_quantity * ice_cream_price) - (pudding_quantity * pudding_price) = 65 := by
  sorry

end NUMINAMATH_CALUDE_ice_cream_pudding_cost_difference_l3348_334817


namespace NUMINAMATH_CALUDE_marble_selection_probability_l3348_334822

/-- The number of blue marbles -/
def blue_marbles : ℕ := 7

/-- The number of yellow marbles -/
def yellow_marbles : ℕ := 5

/-- The total number of selections -/
def total_selections : ℕ := 7

/-- The number of blue marbles we want to select after the first yellow -/
def target_blue : ℕ := 3

/-- The probability of the described event -/
def probability : ℚ := 214375 / 1492992

theorem marble_selection_probability :
  (yellow_marbles : ℚ) / (yellow_marbles + blue_marbles) *
  (Nat.choose (total_selections - 1) target_blue : ℚ) *
  (blue_marbles ^ target_blue * yellow_marbles ^ (total_selections - target_blue - 1) : ℚ) /
  ((yellow_marbles + blue_marbles) ^ (total_selections - 1)) = probability :=
sorry

end NUMINAMATH_CALUDE_marble_selection_probability_l3348_334822


namespace NUMINAMATH_CALUDE_arrangement_count_l3348_334814

/-- Represents the number of people in the arrangement. -/
def total_people : ℕ := 6

/-- Represents the number of people who have a specific position requirement (Jia, Bing, and Yi). -/
def specific_people : ℕ := 3

/-- Calculates the number of arrangements where one person stands between two others in a line of n people. -/
def arrangements (n : ℕ) : ℕ :=
  (Nat.factorial (n - specific_people + 1)) * 2

/-- Theorem stating that the number of arrangements for 6 people with the given condition is 48. -/
theorem arrangement_count :
  arrangements total_people = 48 := by
  sorry

end NUMINAMATH_CALUDE_arrangement_count_l3348_334814


namespace NUMINAMATH_CALUDE_jogger_faster_speed_l3348_334876

/-- Represents the jogger's speed and distance scenario -/
def JoggerScenario (actual_distance : ℝ) (actual_speed : ℝ) (faster_distance : ℝ) (faster_speed : ℝ) : Prop :=
  (actual_distance / actual_speed) = (faster_distance / faster_speed)

/-- Theorem stating the jogger's faster speed given the conditions -/
theorem jogger_faster_speed :
  ∀ (actual_distance actual_speed faster_distance faster_speed : ℝ),
    actual_distance = 30 →
    actual_speed = 12 →
    faster_distance = actual_distance + 10 →
    JoggerScenario actual_distance actual_speed faster_distance faster_speed →
    faster_speed = 16 := by
  sorry


end NUMINAMATH_CALUDE_jogger_faster_speed_l3348_334876


namespace NUMINAMATH_CALUDE_expand_triple_product_l3348_334854

theorem expand_triple_product (x y z : ℝ) :
  (x - 5) * (3 * y + 6) * (z + 4) =
  3 * x * y * z + 6 * x * z - 15 * y * z - 30 * z + 12 * x * y + 24 * x - 60 * y - 120 :=
by sorry

end NUMINAMATH_CALUDE_expand_triple_product_l3348_334854


namespace NUMINAMATH_CALUDE_simplify_nested_expression_l3348_334884

theorem simplify_nested_expression : -(-(-|(-1)|^2)^3)^4 = -1 := by
  sorry

end NUMINAMATH_CALUDE_simplify_nested_expression_l3348_334884


namespace NUMINAMATH_CALUDE_green_ball_probability_l3348_334879

/-- Represents a container of colored balls -/
structure Container where
  red : ℕ
  green : ℕ
  blue : ℕ

/-- The probability of selecting a specific container -/
def containerProb : ℚ := 1 / 3

/-- The containers in the problem -/
def containers : List Container := [
  ⟨10, 5, 3⟩,   -- Container I
  ⟨3, 5, 2⟩,    -- Container II
  ⟨3, 5, 2⟩     -- Container III
]

/-- The probability of selecting a green ball from a given container -/
def greenProb (c : Container) : ℚ :=
  c.green / (c.red + c.green + c.blue)

/-- The total probability of selecting a green ball -/
def totalGreenProb : ℚ :=
  (containers.map (λ c ↦ containerProb * greenProb c)).sum

theorem green_ball_probability : totalGreenProb = 23 / 54 := by
  sorry

end NUMINAMATH_CALUDE_green_ball_probability_l3348_334879


namespace NUMINAMATH_CALUDE_specific_tetrahedron_volume_l3348_334834

/-- Represents a tetrahedron with vertices P, Q, R, S -/
structure Tetrahedron where
  PQ : ℝ
  PR : ℝ
  PS : ℝ
  QR : ℝ
  QS : ℝ
  RS : ℝ

/-- Calculates the volume of a tetrahedron given its edge lengths -/
def tetrahedronVolume (t : Tetrahedron) : ℝ :=
  sorry

/-- Theorem stating that the volume of the specific tetrahedron is 3.5 -/
theorem specific_tetrahedron_volume :
  let t : Tetrahedron := {
    PQ := 6,
    PR := 4,
    PS := 5,
    QR := 5,
    QS := 4,
    RS := 3
  }
  tetrahedronVolume t = 3.5 := by
  sorry

end NUMINAMATH_CALUDE_specific_tetrahedron_volume_l3348_334834


namespace NUMINAMATH_CALUDE_bus_average_speed_with_stoppages_l3348_334881

/-- Proves that given a bus with an average speed of 60 km/hr excluding stoppages
    and stopping for 45 minutes per hour, the average speed including stoppages is 15 km/hr. -/
theorem bus_average_speed_with_stoppages
  (speed_without_stoppages : ℝ)
  (stopping_time : ℝ)
  (h1 : speed_without_stoppages = 60)
  (h2 : stopping_time = 45) :
  let moving_time : ℝ := 60 - stopping_time
  let distance_covered : ℝ := speed_without_stoppages * (moving_time / 60)
  let speed_with_stoppages : ℝ := distance_covered
  speed_with_stoppages = 15 := by
  sorry

end NUMINAMATH_CALUDE_bus_average_speed_with_stoppages_l3348_334881


namespace NUMINAMATH_CALUDE_probability_same_gender_l3348_334892

def total_volunteers : ℕ := 5
def male_volunteers : ℕ := 3
def female_volunteers : ℕ := 2
def volunteers_needed : ℕ := 2

def same_gender_combinations : ℕ := (male_volunteers.choose volunteers_needed) + (female_volunteers.choose volunteers_needed)
def total_combinations : ℕ := total_volunteers.choose volunteers_needed

theorem probability_same_gender :
  (same_gender_combinations : ℚ) / total_combinations = 2 / 5 := by sorry

end NUMINAMATH_CALUDE_probability_same_gender_l3348_334892


namespace NUMINAMATH_CALUDE_equation_solution_equivalence_l3348_334828

def solution_set : Set (ℝ × ℝ × ℝ) :=
  {(-1, 2, 5), (-1, -5, -2), (-2, 1, 5), (-2, -5, -1), (5, 1, -2), (5, 2, -1)}

def satisfies_equations (x y z : ℝ) : Prop :=
  x - y + z = 2 ∧ x^2 + y^2 + z^2 = 30 ∧ x^3 - y^3 + z^3 = 116

theorem equation_solution_equivalence :
  ∀ x y z : ℝ, satisfies_equations x y z ↔ (x, y, z) ∈ solution_set := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_equivalence_l3348_334828


namespace NUMINAMATH_CALUDE_smallest_n_for_99n_all_threes_l3348_334821

def is_all_threes (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ n.digits 10 → d = 3

theorem smallest_n_for_99n_all_threes :
  ∃! N : ℕ, (N > 0) ∧ 
    (is_all_threes (99 * N)) ∧ 
    (∀ m : ℕ, m > 0 → is_all_threes (99 * m) → N ≤ m) ∧
    N = 3367 :=
sorry

end NUMINAMATH_CALUDE_smallest_n_for_99n_all_threes_l3348_334821


namespace NUMINAMATH_CALUDE_distance_ratio_car_a_to_b_l3348_334832

/-- Represents a car with its speed and travel time -/
structure Car where
  speed : ℝ
  time : ℝ

/-- Calculates the distance traveled by a car -/
def distance (c : Car) : ℝ := c.speed * c.time

/-- Theorem: The ratio of distances covered by Car A and Car B is 3:1 -/
theorem distance_ratio_car_a_to_b (car_a car_b : Car)
    (h_speed_a : car_a.speed = 50)
    (h_time_a : car_a.time = 6)
    (h_speed_b : car_b.speed = 100)
    (h_time_b : car_b.time = 1) :
    distance car_a / distance car_b = 3 := by
  sorry

#check distance_ratio_car_a_to_b

end NUMINAMATH_CALUDE_distance_ratio_car_a_to_b_l3348_334832


namespace NUMINAMATH_CALUDE_anna_ham_sandwich_problem_l3348_334885

/-- The number of additional ham slices Anna needs to make a certain number of sandwiches -/
def additional_slices (slices_per_sandwich : ℕ) (current_slices : ℕ) (desired_sandwiches : ℕ) : ℕ :=
  slices_per_sandwich * desired_sandwiches - current_slices

theorem anna_ham_sandwich_problem : 
  additional_slices 3 31 50 = 119 := by
  sorry

end NUMINAMATH_CALUDE_anna_ham_sandwich_problem_l3348_334885


namespace NUMINAMATH_CALUDE_seating_arrangements_l3348_334887

/-- The number of ways to arrange n people in a row. -/
def arrangements (n : ℕ) : ℕ := Nat.factorial n

/-- The number of ways to arrange n people in a row, 
    where a group of k people must sit consecutively. -/
def arrangementsWithConsecutiveGroup (n k : ℕ) : ℕ := 
  arrangements (n - k + 1) * arrangements k

/-- The number of ways to arrange 10 people in a row, 
    where 4 specific people cannot sit in 4 consecutive seats. -/
theorem seating_arrangements : 
  arrangements 10 - arrangementsWithConsecutiveGroup 10 4 = 3507840 := by
  sorry

#eval arrangements 10 - arrangementsWithConsecutiveGroup 10 4

end NUMINAMATH_CALUDE_seating_arrangements_l3348_334887
