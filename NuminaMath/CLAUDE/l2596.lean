import Mathlib

namespace NUMINAMATH_CALUDE_ace_ten_king_of_hearts_probability_l2596_259691

/-- The probability of drawing an Ace, then a 10, then the King of Hearts from a standard deck of 52 cards without replacement -/
theorem ace_ten_king_of_hearts_probability :
  let total_cards : ℕ := 52
  let aces : ℕ := 4
  let tens : ℕ := 4
  let king_of_hearts : ℕ := 1
  (aces / total_cards) * (tens / (total_cards - 1)) * (king_of_hearts / (total_cards - 2)) = 4 / 33150 := by
sorry

end NUMINAMATH_CALUDE_ace_ten_king_of_hearts_probability_l2596_259691


namespace NUMINAMATH_CALUDE_johns_bonus_last_year_l2596_259614

/-- Represents John's yearly financial information -/
structure YearlyFinance where
  salary : ℝ
  bonus_percentage : ℝ
  total_income : ℝ

/-- Calculates the bonus amount given a salary and bonus percentage -/
def calculate_bonus (salary : ℝ) (bonus_percentage : ℝ) : ℝ :=
  salary * bonus_percentage

theorem johns_bonus_last_year 
  (last_year : YearlyFinance)
  (this_year : YearlyFinance)
  (h1 : last_year.salary = 100000)
  (h2 : this_year.salary = 200000)
  (h3 : this_year.total_income = 220000)
  (h4 : last_year.bonus_percentage = this_year.bonus_percentage) :
  calculate_bonus last_year.salary last_year.bonus_percentage = 10000 := by
sorry

end NUMINAMATH_CALUDE_johns_bonus_last_year_l2596_259614


namespace NUMINAMATH_CALUDE_range_of_m_l2596_259686

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 5*x - 14 ≤ 0}
def B (m : ℝ) : Set ℝ := {x | m + 1 < x ∧ x < 2*m - 1}

-- State the theorem
theorem range_of_m :
  ∀ m : ℝ, (A ∪ B m = A) ↔ m ≤ 4 :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l2596_259686


namespace NUMINAMATH_CALUDE_twins_age_problem_l2596_259628

theorem twins_age_problem (age : ℕ) : 
  (age * age) + 5 = ((age + 1) * (age + 1)) → age = 2 := by
  sorry

end NUMINAMATH_CALUDE_twins_age_problem_l2596_259628


namespace NUMINAMATH_CALUDE_domain_of_f_l2596_259635

noncomputable def f (x : ℝ) : ℝ := Real.sqrt ((Real.log x - 2) * (x - Real.log x - 1))

theorem domain_of_f : 
  {x : ℝ | ∃ y, f x = y} = {1} ∪ Set.Ici (Real.exp 2) := by sorry

end NUMINAMATH_CALUDE_domain_of_f_l2596_259635


namespace NUMINAMATH_CALUDE_solution_az_eq_b_l2596_259616

theorem solution_az_eq_b (a b : ℝ) : 
  (∃! x, 4 + 3 * a * x = 2 * a - 7) →
  (¬∃ y, 2 + y = (b + 1) * y) →
  (∀ z, a * z = b ↔ z = 0) :=
by sorry

end NUMINAMATH_CALUDE_solution_az_eq_b_l2596_259616


namespace NUMINAMATH_CALUDE_quadratic_expression_value_l2596_259627

theorem quadratic_expression_value (x : ℝ) (h : 2 * x^2 + 3 * x - 5 = 0) :
  4 * x^2 + 6 * x + 9 = 19 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_expression_value_l2596_259627


namespace NUMINAMATH_CALUDE_find_certain_number_l2596_259694

theorem find_certain_number : ∃ x : ℝ,
  (20 + 40 + 60) / 3 = ((10 + x + 16) / 3 + 8) ∧ x = 70 :=
by
  sorry

end NUMINAMATH_CALUDE_find_certain_number_l2596_259694


namespace NUMINAMATH_CALUDE_S_a_is_three_rays_with_common_point_l2596_259684

/-- The set S_a for a positive integer a -/
def S_a (a : ℕ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | 
    (a = p.1 + 2 ∧ p.2 - 4 ≤ a) ∨
    (a = p.2 - 4 ∧ p.1 + 2 ≤ a) ∨
    (p.1 + 2 = p.2 - 4 ∧ a ≤ p.1 + 2)}

/-- The common point of the three rays -/
def common_point (a : ℕ) : ℝ × ℝ := (a - 2, a + 4)

/-- The three rays that form S_a -/
def ray1 (a : ℕ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 = a - 2 ∧ p.2 ≤ a + 4}
def ray2 (a : ℕ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = a + 4 ∧ p.1 ≤ a - 2}
def ray3 (a : ℕ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = p.1 + 6 ∧ p.1 ≥ a - 2}

/-- Theorem stating that S_a is the union of three rays with a common point -/
theorem S_a_is_three_rays_with_common_point (a : ℕ) :
  S_a a = ray1 a ∪ ray2 a ∪ ray3 a ∧
  common_point a ∈ ray1 a ∧
  common_point a ∈ ray2 a ∧
  common_point a ∈ ray3 a :=
sorry

end NUMINAMATH_CALUDE_S_a_is_three_rays_with_common_point_l2596_259684


namespace NUMINAMATH_CALUDE_partnership_profit_l2596_259672

/-- Given a partnership with three investors and the profit share of one investor,
    calculate the total profit. -/
theorem partnership_profit
  (a b c : ℕ)  -- Investments of the three partners
  (c_share : ℕ)  -- C's share of the profit
  (ha : a = 5000)
  (hb : b = 8000)
  (hc : c = 9000)
  (hc_share : c_share = 36000) :
  let total_parts := a / 1000 + b / 1000 + c / 1000
  let part_value := c_share / (c / 1000)
  total_parts * part_value = 88000 :=
by sorry

end NUMINAMATH_CALUDE_partnership_profit_l2596_259672


namespace NUMINAMATH_CALUDE_smallest_crate_dimension_l2596_259651

/-- Represents the dimensions of a rectangular crate -/
structure CrateDimensions where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a right circular cylinder -/
structure Cylinder where
  radius : ℝ
  height : ℝ

/-- Checks if a cylinder can fit upright in a crate -/
def cylinderFitsInCrate (crate : CrateDimensions) (cylinder : Cylinder) : Prop :=
  2 * cylinder.radius ≤ min crate.x (min crate.y crate.z)

/-- The theorem stating the smallest dimension of the crate -/
theorem smallest_crate_dimension
  (crate : CrateDimensions)
  (cylinder : Cylinder)
  (h1 : crate.y = 8)
  (h2 : crate.z = 12)
  (h3 : cylinder.radius = 3)
  (h4 : cylinderFitsInCrate crate cylinder) :
  crate.x ≥ 6 := by
  sorry

end NUMINAMATH_CALUDE_smallest_crate_dimension_l2596_259651


namespace NUMINAMATH_CALUDE_noahs_sales_ratio_l2596_259663

/-- Noah's painting sales problem -/
theorem noahs_sales_ratio :
  let large_price : ℕ := 60
  let small_price : ℕ := 30
  let last_month_large : ℕ := 8
  let last_month_small : ℕ := 4
  let this_month_sales : ℕ := 1200
  let last_month_sales : ℕ := large_price * last_month_large + small_price * last_month_small
  (this_month_sales : ℚ) / (last_month_sales : ℚ) = 2 := by
  sorry

end NUMINAMATH_CALUDE_noahs_sales_ratio_l2596_259663


namespace NUMINAMATH_CALUDE_inequality_solution_l2596_259622

def solution_set (a : ℝ) : Set ℝ :=
  if a = 0 then {x | x > 1}
  else if a < 0 then {x | x > 1 ∨ x < 1/a}
  else if 0 < a ∧ a < 1 then {x | 1 < x ∧ x < 1/a}
  else if a > 1 then {x | 1/a < x ∧ x < 1}
  else ∅

theorem inequality_solution (a : ℝ) (x : ℝ) :
  a * x^2 - (a + 1) * x + 1 < 0 ↔ x ∈ solution_set a :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_l2596_259622


namespace NUMINAMATH_CALUDE_unique_solution_for_equation_l2596_259698

theorem unique_solution_for_equation : ∃! (x y z : ℕ), 
  x < 10 ∧ y < 10 ∧ z < 10 ∧ 
  (10 * x + 5) * (300 + 10 * y + z) = 7850 ∧
  x = 2 ∧ y = 1 ∧ z = 4 := by sorry

end NUMINAMATH_CALUDE_unique_solution_for_equation_l2596_259698


namespace NUMINAMATH_CALUDE_polynomial_root_implies_k_value_l2596_259629

theorem polynomial_root_implies_k_value : 
  ∀ k : ℚ, (3 : ℚ)^3 + 7*(3 : ℚ)^2 + k*(3 : ℚ) + 23 = 0 → k = -113/3 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_root_implies_k_value_l2596_259629


namespace NUMINAMATH_CALUDE_equation_solution_l2596_259615

theorem equation_solution (M : ℚ) : (5 + 7 + 9) / 3 = (4020 + 4021 + 4022) / M → M = 1723 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2596_259615


namespace NUMINAMATH_CALUDE_cricketer_average_increase_l2596_259624

theorem cricketer_average_increase (total_innings : ℕ) (last_inning_score : ℕ) (final_average : ℚ) : 
  total_innings = 19 → 
  last_inning_score = 98 → 
  final_average = 26 → 
  (final_average - (total_innings * final_average - last_inning_score) / (total_innings - 1)) = 4 := by
sorry

end NUMINAMATH_CALUDE_cricketer_average_increase_l2596_259624


namespace NUMINAMATH_CALUDE_max_min_difference_c_l2596_259678

theorem max_min_difference_c (a b c : ℝ) 
  (sum_eq : a + b + c = 5)
  (sum_squares_eq : a^2 + b^2 + c^2 = 27) :
  ∃ (c_max c_min : ℝ),
    (∀ c' : ℝ, (∃ a' b' : ℝ, a' + b' + c' = 5 ∧ a'^2 + b'^2 + c'^2 = 27) → c' ≤ c_max) ∧
    (∀ c' : ℝ, (∃ a' b' : ℝ, a' + b' + c' = 5 ∧ a'^2 + b'^2 + c'^2 = 27) → c_min ≤ c') ∧
    c_max - c_min = 22/3 :=
sorry

end NUMINAMATH_CALUDE_max_min_difference_c_l2596_259678


namespace NUMINAMATH_CALUDE_logical_equivalence_l2596_259637

variable (E W : Prop)

-- E: Pink elephants on planet α have purple eyes
-- W: Wild boars on planet β have long noses

theorem logical_equivalence :
  ((E → ¬W) ↔ (W → ¬E)) ∧ ((E → ¬W) ↔ (¬E ∨ ¬W)) := by sorry

end NUMINAMATH_CALUDE_logical_equivalence_l2596_259637


namespace NUMINAMATH_CALUDE_gcd_of_72_120_168_l2596_259608

theorem gcd_of_72_120_168 : Nat.gcd 72 (Nat.gcd 120 168) = 24 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_72_120_168_l2596_259608


namespace NUMINAMATH_CALUDE_minji_water_intake_l2596_259666

theorem minji_water_intake (morning_intake : Real) (afternoon_intake : Real)
  (h1 : morning_intake = 0.26)
  (h2 : afternoon_intake = 0.37) :
  morning_intake + afternoon_intake = 0.63 := by
sorry

end NUMINAMATH_CALUDE_minji_water_intake_l2596_259666


namespace NUMINAMATH_CALUDE_exponential_equality_l2596_259647

theorem exponential_equality (x a b : ℝ) (h1 : 3^x = a) (h2 : 5^x = b) : 45^x = a^2 * b := by
  sorry

end NUMINAMATH_CALUDE_exponential_equality_l2596_259647


namespace NUMINAMATH_CALUDE_health_risk_factors_l2596_259656

theorem health_risk_factors (total_population : ℕ) 
  (prob_one_factor : ℚ) 
  (prob_two_factors : ℚ) 
  (prob_all_given_AB : ℚ) :
  prob_one_factor = 1/10 →
  prob_two_factors = 14/100 →
  prob_all_given_AB = 1/3 →
  total_population > 0 →
  ∃ (num_no_factors : ℕ) (num_not_A : ℕ),
    (num_no_factors : ℚ) / (num_not_A : ℚ) = 21/55 ∧
    num_no_factors + num_not_A = 76 :=
by sorry

end NUMINAMATH_CALUDE_health_risk_factors_l2596_259656


namespace NUMINAMATH_CALUDE_total_spent_is_450_l2596_259625

/-- The total amount spent by Leonard and Michael on presents for their father -/
def total_spent (leonard_wallet : ℕ) (leonard_sneakers : ℕ) (leonard_sneakers_pairs : ℕ)
  (michael_backpack : ℕ) (michael_jeans : ℕ) (michael_jeans_pairs : ℕ) : ℕ :=
  leonard_wallet + leonard_sneakers * leonard_sneakers_pairs +
  michael_backpack + michael_jeans * michael_jeans_pairs

/-- Theorem stating that the total amount spent is $450 -/
theorem total_spent_is_450 :
  total_spent 50 100 2 100 50 2 = 450 := by
  sorry

end NUMINAMATH_CALUDE_total_spent_is_450_l2596_259625


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_fractions_l2596_259633

theorem arithmetic_mean_of_fractions :
  let a := (3 : ℚ) / 4
  let b := (5 : ℚ) / 8
  let c := (7 : ℚ) / 9
  (a + b + c) / 3 = 155 / 216 := by sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_fractions_l2596_259633


namespace NUMINAMATH_CALUDE_composite_sum_l2596_259636

theorem composite_sum (a b c d e f : ℕ+) 
  (hS : ∃ (k₁ k₂ : ℕ), 
    (a + b + c + d + e + f) * k₁ = a * b * c + d * e * f ∧ 
    (a + b + c + d + e + f) * k₂ = a * b + b * c + c * a - d * e - e * f - f * d) :
  ∃ (m n : ℕ), m > 1 ∧ n > 1 ∧ a + b + c + d + e + f = m * n := by
  sorry

end NUMINAMATH_CALUDE_composite_sum_l2596_259636


namespace NUMINAMATH_CALUDE_difference_X_Y_cost_per_capsule_l2596_259653

/-- Represents a bottle of capsules -/
structure Bottle where
  capsules : ℕ
  cost : ℚ

/-- Calculates the cost per capsule for a given bottle -/
def costPerCapsule (b : Bottle) : ℚ :=
  b.cost / b.capsules

/-- Theorem stating the difference in cost per capsule between bottles X and Y -/
theorem difference_X_Y_cost_per_capsule :
  let R : Bottle := { capsules := 250, cost := 25/4 }
  let T : Bottle := { capsules := 100, cost := 3 }
  let X : Bottle := { capsules := 300, cost := 15/2 }
  let Y : Bottle := { capsules := 120, cost := 4 }
  abs (costPerCapsule X - costPerCapsule Y) = 83/10000 := by
  sorry

end NUMINAMATH_CALUDE_difference_X_Y_cost_per_capsule_l2596_259653


namespace NUMINAMATH_CALUDE_hawks_score_l2596_259679

theorem hawks_score (total_points margin : ℕ) (h1 : total_points = 48) (h2 : margin = 16) :
  ∃ (eagles_score hawks_score : ℕ),
    eagles_score + hawks_score = total_points ∧
    eagles_score - hawks_score = margin ∧
    hawks_score = 16 := by
  sorry

end NUMINAMATH_CALUDE_hawks_score_l2596_259679


namespace NUMINAMATH_CALUDE_boy_age_problem_l2596_259613

theorem boy_age_problem (total_boys : Nat) (avg_age_all : Nat) (avg_age_first_six : Nat) (avg_age_last_six : Nat)
  (h1 : total_boys = 11)
  (h2 : avg_age_all = 50)
  (h3 : avg_age_first_six = 49)
  (h4 : avg_age_last_six = 52) :
  total_boys * avg_age_all = 6 * avg_age_first_six + 6 * avg_age_last_six - 56 := by
  sorry

#check boy_age_problem

end NUMINAMATH_CALUDE_boy_age_problem_l2596_259613


namespace NUMINAMATH_CALUDE_pattern_equality_l2596_259659

theorem pattern_equality (n : ℤ) : n * (n + 2) - (n + 1)^2 = -1 := by
  sorry

end NUMINAMATH_CALUDE_pattern_equality_l2596_259659


namespace NUMINAMATH_CALUDE_apple_distribution_l2596_259632

theorem apple_distribution (n : ℕ) (k : ℕ) (h1 : n = 30) (h2 : k = 4) :
  (Nat.choose (n + k - 1) (k - 1)) = 3276 :=
sorry

end NUMINAMATH_CALUDE_apple_distribution_l2596_259632


namespace NUMINAMATH_CALUDE_smallest_consecutive_even_number_l2596_259662

theorem smallest_consecutive_even_number (n : ℕ) : 
  (n % 2 = 0) →  -- n is even
  (n + (n + 2) + (n + 4) = 162) →  -- sum of three consecutive even numbers is 162
  n = 52 :=  -- the smallest number is 52
by sorry

end NUMINAMATH_CALUDE_smallest_consecutive_even_number_l2596_259662


namespace NUMINAMATH_CALUDE_sum_of_digits_of_10_pow_93_minus_95_l2596_259623

/-- Represents the sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Theorem stating that the sum of digits of 10^93 - 95 is 824 -/
theorem sum_of_digits_of_10_pow_93_minus_95 : 
  sum_of_digits (10^93 - 95) = 824 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_10_pow_93_minus_95_l2596_259623


namespace NUMINAMATH_CALUDE_line_parallel_properties_l2596_259667

-- Define the structure for a line
structure Line where
  slope : ℝ
  angle : ℝ

-- Define the parallel relation
def parallel (l1 l2 : Line) : Prop :=
  l1.angle = l2.angle

-- Theorem statement
theorem line_parallel_properties (l1 l2 : Line) :
  (l1.slope = l2.slope → parallel l1 l2) ∧
  (l1.angle = l2.angle → parallel l1 l2) ∧
  (parallel l1 l2 → l1.angle = l2.angle) :=
sorry

end NUMINAMATH_CALUDE_line_parallel_properties_l2596_259667


namespace NUMINAMATH_CALUDE_power_of_two_divisibility_l2596_259642

theorem power_of_two_divisibility (n a b : ℕ) : 
  2^n = 10*a + b → n > 3 → 0 < b → b < 10 → ∃ k, ab = 6*k := by
  sorry

end NUMINAMATH_CALUDE_power_of_two_divisibility_l2596_259642


namespace NUMINAMATH_CALUDE_triangle_radius_inequality_l2596_259640

structure Triangle where
  R : ℝ  -- circumradius
  r : ℝ  -- inradius
  P : ℝ  -- perimeter
  is_acute_or_obtuse : Bool
  is_right_angled : Bool
  positive_R : R > 0
  positive_r : r > 0
  positive_P : P > 0

theorem triangle_radius_inequality (t : Triangle) : 
  (t.is_acute_or_obtuse ∧ t.R > (Real.sqrt 3 / 3) * Real.sqrt (t.P * t.r)) ∨
  (t.is_right_angled ∧ t.R ≥ (Real.sqrt 2 / 2) * Real.sqrt (t.P * t.r)) :=
sorry

end NUMINAMATH_CALUDE_triangle_radius_inequality_l2596_259640


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocals_l2596_259603

theorem min_value_sum_reciprocals (x y z : ℝ) 
  (pos_x : 0 < x) (pos_y : 0 < y) (pos_z : 0 < z) 
  (sum_one : x + y + z = 1) :
  1 / (x + 3 * y) + 1 / (y + 3 * z) + 1 / (z + 3 * x) ≥ 9 / 4 ∧
  ∃ a b c : ℝ, 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b + c = 1 ∧
    1 / (a + 3 * b) + 1 / (b + 3 * c) + 1 / (c + 3 * a) = 9 / 4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_reciprocals_l2596_259603


namespace NUMINAMATH_CALUDE_complement_of_M_l2596_259641

-- Define the universal set U as the set of real numbers
def U : Set ℝ := Set.univ

-- Define set M
def M : Set ℝ := {x : ℝ | |x| > 2}

-- State the theorem
theorem complement_of_M :
  Mᶜ = {x : ℝ | -2 ≤ x ∧ x ≤ 2} :=
by sorry

end NUMINAMATH_CALUDE_complement_of_M_l2596_259641


namespace NUMINAMATH_CALUDE_percentage_green_tiles_l2596_259670

def courtyard_length : ℝ := 25
def courtyard_width : ℝ := 10
def tiles_per_sqft : ℝ := 4
def green_tile_cost : ℝ := 3
def red_tile_cost : ℝ := 1.5
def total_cost : ℝ := 2100

theorem percentage_green_tiles :
  let total_area : ℝ := courtyard_length * courtyard_width
  let total_tiles : ℝ := total_area * tiles_per_sqft
  let green_tiles : ℝ := (total_cost - red_tile_cost * total_tiles) / (green_tile_cost - red_tile_cost)
  (green_tiles / total_tiles) * 100 = 40 := by
sorry

end NUMINAMATH_CALUDE_percentage_green_tiles_l2596_259670


namespace NUMINAMATH_CALUDE_money_distribution_l2596_259697

theorem money_distribution (a b c total : ℕ) : 
  a + b + c = total →
  2 * b = 3 * a →
  4 * b = 3 * c →
  b = 600 →
  total = 1800 := by
sorry

end NUMINAMATH_CALUDE_money_distribution_l2596_259697


namespace NUMINAMATH_CALUDE_cylinder_surface_area_l2596_259649

/-- The surface area of a cylinder with base radius 2 and lateral surface length
    equal to the diameter of the base is 24π. -/
theorem cylinder_surface_area : 
  let r : ℝ := 2
  let l : ℝ := 2 * r
  let surface_area : ℝ := 2 * Real.pi * r^2 + 2 * Real.pi * r * l
  surface_area = 24 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_cylinder_surface_area_l2596_259649


namespace NUMINAMATH_CALUDE_difference_8_in_96348621_l2596_259606

/-- The difference between the local value and the face value of a digit in a number -/
def localFaceDifference (n : ℕ) (d : ℕ) (p : ℕ) : ℕ :=
  d * (10 ^ p) - d

/-- The position of a digit in a number, counting from right to left and starting at 0 -/
def digitPosition (n : ℕ) (d : ℕ) : ℕ :=
  sorry -- Implementation not required for the statement

theorem difference_8_in_96348621 :
  localFaceDifference 96348621 8 (digitPosition 96348621 8) = 7992 := by
  sorry

end NUMINAMATH_CALUDE_difference_8_in_96348621_l2596_259606


namespace NUMINAMATH_CALUDE_square_perimeter_9cm_l2596_259669

/-- The perimeter of a square with side length 9 centimeters is 36 centimeters. -/
theorem square_perimeter_9cm (s : ℝ) (h : s = 9) : 4 * s = 36 := by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_9cm_l2596_259669


namespace NUMINAMATH_CALUDE_landmark_visit_sequences_l2596_259683

theorem landmark_visit_sequences (n : Nat) (h : n = 5) : 
  (List.permutations (List.range n)).length = 120 := by
  sorry

end NUMINAMATH_CALUDE_landmark_visit_sequences_l2596_259683


namespace NUMINAMATH_CALUDE_output_for_15_l2596_259645

def function_machine (input : ℕ) : ℕ :=
  let step1 := input * 2
  if step1 > 25 then step1 - 7 else step1 + 10

theorem output_for_15 : function_machine 15 = 23 := by
  sorry

end NUMINAMATH_CALUDE_output_for_15_l2596_259645


namespace NUMINAMATH_CALUDE_cookie_distribution_l2596_259690

theorem cookie_distribution (people : ℕ) (cookies_per_person : ℕ) 
  (h1 : people = 6) (h2 : cookies_per_person = 4) : 
  people * cookies_per_person = 24 := by
  sorry

end NUMINAMATH_CALUDE_cookie_distribution_l2596_259690


namespace NUMINAMATH_CALUDE_least_common_period_is_30_l2596_259677

/-- A function satisfying the given condition -/
def satisfies_condition (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (x + 5) + f (x - 5) = f x

/-- The period of a function -/
def is_period (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x : ℝ, f (x + p) = f x

/-- The least positive period of a function -/
def is_least_positive_period (f : ℝ → ℝ) (p : ℝ) : Prop :=
  p > 0 ∧ is_period f p ∧ ∀ q : ℝ, 0 < q ∧ q < p → ¬ is_period f q

/-- The main theorem -/
theorem least_common_period_is_30 :
  ∃ p : ℝ, p = 30 ∧
    (∀ f : ℝ → ℝ, satisfies_condition f → is_least_positive_period f p) ∧
    (∀ q : ℝ, q ≠ p →
      ∃ f : ℝ → ℝ, satisfies_condition f ∧ ¬ is_least_positive_period f q) :=
sorry

end NUMINAMATH_CALUDE_least_common_period_is_30_l2596_259677


namespace NUMINAMATH_CALUDE_same_speed_problem_l2596_259680

theorem same_speed_problem (x : ℝ) :
  let jack_speed := x^2 - 11*x - 22
  let jill_distance := x^2 - 5*x - 36
  let jill_time := x + 4
  jack_speed > 0 ∧ 
  jill_distance > 0 ∧ 
  jill_time > 0 ∧
  jack_speed = jill_distance / jill_time →
  jack_speed = 4 := by
sorry

end NUMINAMATH_CALUDE_same_speed_problem_l2596_259680


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l2596_259696

theorem quadratic_inequality_solution_set :
  ∀ x : ℝ, x^2 - 5*x - 14 < 0 ↔ -2 < x ∧ x < 7 := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l2596_259696


namespace NUMINAMATH_CALUDE_abs_value_properties_l2596_259600

-- Define the absolute value function
def f (x : ℝ) := abs x

-- State the theorem
theorem abs_value_properties :
  (∀ x : ℝ, f (-x) = f x) ∧ 
  (∀ x y : ℝ, 0 < x → x < y → f x < f y) :=
by sorry

end NUMINAMATH_CALUDE_abs_value_properties_l2596_259600


namespace NUMINAMATH_CALUDE_train_average_speed_with_stoppages_l2596_259660

theorem train_average_speed_with_stoppages 
  (speed_without_stoppages : ℝ)
  (stop_time_per_hour : ℝ)
  (h1 : speed_without_stoppages = 100)
  (h2 : stop_time_per_hour = 3)
  : (speed_without_stoppages * (60 - stop_time_per_hour) / 60) = 95 := by
  sorry

end NUMINAMATH_CALUDE_train_average_speed_with_stoppages_l2596_259660


namespace NUMINAMATH_CALUDE_x_plus_y_value_l2596_259644

theorem x_plus_y_value (x y : ℝ) 
  (h1 : x + Real.sin y = 2010)
  (h2 : x + 2010 * Real.cos y = 2009)
  (h3 : 0 ≤ y ∧ y ≤ Real.pi / 2) :
  x + y = 2009 + Real.pi / 2 := by
sorry

end NUMINAMATH_CALUDE_x_plus_y_value_l2596_259644


namespace NUMINAMATH_CALUDE_tax_savings_proof_l2596_259674

def original_tax_rate : ℚ := 40 / 100
def new_tax_rate : ℚ := 33 / 100
def annual_income : ℚ := 45000

def differential_savings : ℚ := original_tax_rate * annual_income - new_tax_rate * annual_income

theorem tax_savings_proof : differential_savings = 3150 := by
  sorry

end NUMINAMATH_CALUDE_tax_savings_proof_l2596_259674


namespace NUMINAMATH_CALUDE_pens_left_after_sale_l2596_259687

def initial_pens : ℕ := 42
def sold_pens : ℕ := 23

theorem pens_left_after_sale : initial_pens - sold_pens = 19 := by
  sorry

end NUMINAMATH_CALUDE_pens_left_after_sale_l2596_259687


namespace NUMINAMATH_CALUDE_tan_half_product_l2596_259681

theorem tan_half_product (a b : Real) :
  3 * (Real.cos a + Real.cos b) + 6 * (Real.cos a * Real.cos b + 1) = 0 →
  (Real.tan (a / 2) * Real.tan (b / 2) = Real.sqrt 2) ∨
  (Real.tan (a / 2) * Real.tan (b / 2) = -Real.sqrt 2) := by
sorry

end NUMINAMATH_CALUDE_tan_half_product_l2596_259681


namespace NUMINAMATH_CALUDE_vector_subtraction_l2596_259621

def vector_a : ℝ × ℝ := (-1, 2)
def vector_b : ℝ × ℝ := (0, 1)

theorem vector_subtraction :
  vector_a - 2 • vector_b = (-1, 0) := by sorry

end NUMINAMATH_CALUDE_vector_subtraction_l2596_259621


namespace NUMINAMATH_CALUDE_breakfast_cost_is_correct_l2596_259605

/-- Calculates the total cost of breakfast for Francis and Kiera -/
def breakfast_cost : ℝ :=
  let muffin_price : ℝ := 2
  let fruit_cup_price : ℝ := 3
  let coffee_price : ℝ := 1.5
  let discount_rate : ℝ := 0.1
  
  let francis_muffins : ℕ := 2
  let francis_fruit_cups : ℕ := 2
  let francis_coffee : ℕ := 1
  
  let kiera_muffins : ℕ := 2
  let kiera_fruit_cups : ℕ := 1
  let kiera_coffee : ℕ := 2
  
  let francis_cost : ℝ := 
    muffin_price * francis_muffins + 
    fruit_cup_price * francis_fruit_cups + 
    coffee_price * francis_coffee
  
  let kiera_cost_before_discount : ℝ := 
    muffin_price * kiera_muffins + 
    fruit_cup_price * kiera_fruit_cups + 
    coffee_price * kiera_coffee
  
  let discount_amount : ℝ := 
    discount_rate * (muffin_price * 2 + fruit_cup_price)
  
  let kiera_cost : ℝ := kiera_cost_before_discount - discount_amount
  
  francis_cost + kiera_cost

theorem breakfast_cost_is_correct : breakfast_cost = 20.8 := by
  sorry

end NUMINAMATH_CALUDE_breakfast_cost_is_correct_l2596_259605


namespace NUMINAMATH_CALUDE_regular_octagon_interior_angle_is_135_l2596_259692

/-- The measure of each interior angle of a regular octagon in degrees. -/
def regular_octagon_interior_angle : ℝ := 135

/-- Theorem stating that the measure of each interior angle of a regular octagon is 135 degrees. -/
theorem regular_octagon_interior_angle_is_135 :
  regular_octagon_interior_angle = 135 := by sorry

end NUMINAMATH_CALUDE_regular_octagon_interior_angle_is_135_l2596_259692


namespace NUMINAMATH_CALUDE_sum_geq_sqrt_three_l2596_259617

theorem sum_geq_sqrt_three (a b c : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_sum_prod : a * b + b * c + c * a = 1) : 
  a + b + c ≥ Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_sum_geq_sqrt_three_l2596_259617


namespace NUMINAMATH_CALUDE_division_problem_l2596_259638

theorem division_problem (x : ℝ) : 45 / x = 900 → x = 0.05 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l2596_259638


namespace NUMINAMATH_CALUDE_arctan_equation_solution_l2596_259675

theorem arctan_equation_solution :
  ∃ x : ℝ, x > 0 ∧ Real.arctan (1/x) + Real.arctan (1/x^2) + Real.arctan (1/x^3) = π/4 ∧ x = 1 :=
by sorry

end NUMINAMATH_CALUDE_arctan_equation_solution_l2596_259675


namespace NUMINAMATH_CALUDE_tablespoons_in_half_cup_l2596_259648

/-- Proves that there are 8 tablespoons in half a cup of rice -/
theorem tablespoons_in_half_cup (grains_per_cup : ℕ) (teaspoons_per_tablespoon : ℕ) (grains_per_teaspoon : ℕ)
  (h1 : grains_per_cup = 480)
  (h2 : teaspoons_per_tablespoon = 3)
  (h3 : grains_per_teaspoon = 10) :
  (grains_per_cup / 2) / (grains_per_teaspoon * teaspoons_per_tablespoon) = 8 := by
  sorry

#check tablespoons_in_half_cup

end NUMINAMATH_CALUDE_tablespoons_in_half_cup_l2596_259648


namespace NUMINAMATH_CALUDE_traffic_light_combinations_l2596_259665

/-- The number of different signals that can be transmitted by k traffic lights -/
def total_signals (k : ℕ) : ℕ := 3^k

/-- Theorem: Given k traffic lights, each capable of transmitting 3 different signals,
    the total number of unique signal combinations is 3^k -/
theorem traffic_light_combinations (k : ℕ) :
  total_signals k = 3^k := by
  sorry

end NUMINAMATH_CALUDE_traffic_light_combinations_l2596_259665


namespace NUMINAMATH_CALUDE_sin_230_minus_sqrt3_tan_170_l2596_259604

theorem sin_230_minus_sqrt3_tan_170 : 
  Real.sin (230 * π / 180) * (1 - Real.sqrt 3 * Real.tan (170 * π / 180)) = -1 := by
  sorry

end NUMINAMATH_CALUDE_sin_230_minus_sqrt3_tan_170_l2596_259604


namespace NUMINAMATH_CALUDE_smallest_class_size_l2596_259610

/-- Represents a class of students who took a test -/
structure TestClass where
  n : ℕ                -- number of students
  scores : Fin n → ℕ   -- scores of each student
  test_max : ℕ         -- maximum possible score on the test

/-- Conditions for our specific test class -/
def SatisfiesConditions (c : TestClass) : Prop :=
  c.test_max = 100 ∧
  (∃ (i₁ i₂ i₃ i₄ : Fin c.n), i₁ ≠ i₂ ∧ i₁ ≠ i₃ ∧ i₁ ≠ i₄ ∧ i₂ ≠ i₃ ∧ i₂ ≠ i₄ ∧ i₃ ≠ i₄ ∧
    c.scores i₁ = 90 ∧ c.scores i₂ = 90 ∧ c.scores i₃ = 90 ∧ c.scores i₄ = 90) ∧
  (∀ i, c.scores i ≥ 70) ∧
  (Finset.sum (Finset.univ : Finset (Fin c.n)) c.scores / c.n = 80)

/-- The main theorem stating that the smallest possible class size is 8 -/
theorem smallest_class_size (c : TestClass) (h : SatisfiesConditions c) :
  c.n ≥ 8 ∧ ∃ (c' : TestClass), SatisfiesConditions c' ∧ c'.n = 8 := by
  sorry

end NUMINAMATH_CALUDE_smallest_class_size_l2596_259610


namespace NUMINAMATH_CALUDE_scoring_ratio_is_two_to_one_l2596_259626

/-- Represents the scoring system for a test -/
structure TestScoring where
  totalQuestions : ℕ
  correctAnswers : ℕ
  score : ℕ
  scoringRatio : ℚ

/-- Calculates the score based on correct answers, incorrect answers, and the scoring ratio -/
def calculateScore (correct : ℕ) (incorrect : ℕ) (ratio : ℚ) : ℚ :=
  correct - ratio * incorrect

/-- Theorem stating that the scoring ratio is 2:1 for the given test conditions -/
theorem scoring_ratio_is_two_to_one (t : TestScoring)
    (h1 : t.totalQuestions = 100)
    (h2 : t.correctAnswers = 91)
    (h3 : t.score = 73)
    (h4 : calculateScore t.correctAnswers (t.totalQuestions - t.correctAnswers) t.scoringRatio = t.score) :
    t.scoringRatio = 2 := by
  sorry


end NUMINAMATH_CALUDE_scoring_ratio_is_two_to_one_l2596_259626


namespace NUMINAMATH_CALUDE_ratio_and_quadratic_equation_solution_l2596_259630

theorem ratio_and_quadratic_equation_solution (x y z a : ℤ) : 
  (∃ k : ℚ, x = 4 * k ∧ y = 6 * k ∧ z = 10 * k) →
  y^2 = 40 * a - 20 →
  a = 1 := by
sorry

end NUMINAMATH_CALUDE_ratio_and_quadratic_equation_solution_l2596_259630


namespace NUMINAMATH_CALUDE_rectangle_area_l2596_259618

/-- The length of the shorter side of each small rectangle -/
def short_side : ℝ := 4

/-- The number of small rectangles -/
def num_rectangles : ℕ := 4

/-- The aspect ratio of each small rectangle -/
def aspect_ratio : ℝ := 2

/-- The length of the longer side of each small rectangle -/
def long_side : ℝ := short_side * aspect_ratio

/-- The width of rectangle EFGH -/
def width : ℝ := long_side

/-- The length of rectangle EFGH -/
def length : ℝ := 2 * long_side

/-- The area of rectangle EFGH -/
def area : ℝ := width * length

theorem rectangle_area : area = 128 := by sorry

end NUMINAMATH_CALUDE_rectangle_area_l2596_259618


namespace NUMINAMATH_CALUDE_cyclist_speed_proof_l2596_259661

/-- Represents the speed of the east-bound cyclist in mph -/
def east_speed : ℝ := 18

/-- Represents the speed of the west-bound cyclist in mph -/
def west_speed : ℝ := east_speed + 4

/-- Represents the time traveled in hours -/
def time : ℝ := 5

/-- Represents the total distance between the cyclists after the given time -/
def total_distance : ℝ := 200

theorem cyclist_speed_proof :
  east_speed * time + west_speed * time = total_distance :=
by sorry

end NUMINAMATH_CALUDE_cyclist_speed_proof_l2596_259661


namespace NUMINAMATH_CALUDE_slope_angle_range_l2596_259655

/-- A line passing through the point (0, -2) and intersecting the unit circle -/
structure IntersectingLine where
  /-- The slope of the line -/
  k : ℝ
  /-- The line passes through (0, -2) -/
  passes_through_point : k * 0 - 2 = -2
  /-- The line intersects the unit circle -/
  intersects_circle : ∃ (x y : ℝ), x^2 + y^2 = 1 ∧ y = k * x - 2

/-- The slope angle of a line -/
noncomputable def slope_angle (l : IntersectingLine) : ℝ :=
  Real.arctan l.k

/-- Theorem: The range of the slope angle for lines intersecting the unit circle and passing through (0, -2) -/
theorem slope_angle_range (l : IntersectingLine) :
  π/3 ≤ slope_angle l ∧ slope_angle l ≤ 2*π/3 :=
sorry

end NUMINAMATH_CALUDE_slope_angle_range_l2596_259655


namespace NUMINAMATH_CALUDE_min_value_expression_equality_condition_l2596_259657

theorem min_value_expression (x : ℝ) (hx : x > 0) : 2 * Real.sqrt x + 1 / x + x^2 ≥ 4 :=
by sorry

theorem equality_condition : 2 * Real.sqrt 1 + 1 / 1 + 1^2 = 4 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_equality_condition_l2596_259657


namespace NUMINAMATH_CALUDE_euler_line_parallel_iff_condition_l2596_259689

/-- Triangle ABC with side lengths a, b, and c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  pos_a : 0 < a
  pos_b : 0 < b
  pos_c : 0 < c
  triangle_ineq : a < b + c ∧ b < a + c ∧ c < a + b

/-- The Euler line of a triangle -/
def EulerLine (t : Triangle) : Set (ℝ × ℝ) := sorry

/-- A line parallel to the side BC of the triangle -/
def ParallelToBC (t : Triangle) : Set (ℝ × ℝ) := sorry

/-- The condition for Euler line parallelism -/
def EulerLineParallelCondition (t : Triangle) : Prop :=
  2 * t.a^4 = (t.b^2 - t.c^2)^2 + (t.b^2 + t.c^2) * t.a^2

/-- Theorem: The Euler line is parallel to side BC if and only if the condition holds -/
theorem euler_line_parallel_iff_condition (t : Triangle) :
  EulerLine t = ParallelToBC t ↔ EulerLineParallelCondition t := by sorry

end NUMINAMATH_CALUDE_euler_line_parallel_iff_condition_l2596_259689


namespace NUMINAMATH_CALUDE_travis_potato_probability_l2596_259685

/-- Represents a player in the hot potato game -/
inductive Player : Type
  | George : Player
  | Jeff : Player
  | Brian : Player
  | Travis : Player

/-- The game state after each turn -/
structure GameState :=
  (george_potatoes : Nat)
  (jeff_potatoes : Nat)
  (brian_potatoes : Nat)
  (travis_potatoes : Nat)

/-- The initial game state -/
def initial_state : GameState :=
  ⟨1, 1, 0, 0⟩

/-- The probability of passing a potato to a specific player -/
def pass_probability : ℚ := 1 / 3

/-- The probability of Travis having at least one hot potato after one round -/
def travis_has_potato_probability : ℚ := 5 / 27

/-- Theorem stating the probability of Travis having at least one hot potato after one round -/
theorem travis_potato_probability :
  travis_has_potato_probability = 5 / 27 :=
by sorry


end NUMINAMATH_CALUDE_travis_potato_probability_l2596_259685


namespace NUMINAMATH_CALUDE_share_multiple_l2596_259658

theorem share_multiple (total : ℚ) (c_share : ℚ) (x : ℚ) : 
  total = 585 →
  c_share = 260 →
  ∃ (a_share b_share : ℚ),
    a_share + b_share + c_share = total ∧
    x * a_share = 6 * b_share ∧
    x * a_share = 3 * c_share →
  x = 4 := by
sorry

end NUMINAMATH_CALUDE_share_multiple_l2596_259658


namespace NUMINAMATH_CALUDE_hyperbola_vertex_distance_l2596_259612

/-- The distance between the vertices of a hyperbola with equation x^2/16 - y^2/25 = 1 is 8 -/
theorem hyperbola_vertex_distance : 
  ∀ (x y : ℝ), x^2/16 - y^2/25 = 1 → ∃ (v₁ v₂ : ℝ × ℝ), 
    (v₁.1^2/16 - v₁.2^2/25 = 1) ∧ 
    (v₂.1^2/16 - v₂.2^2/25 = 1) ∧ 
    (v₁.2 = 0) ∧ (v₂.2 = 0) ∧
    (v₁.1 + v₂.1 = 0) ∧
    (|v₁.1 - v₂.1| = 8) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_vertex_distance_l2596_259612


namespace NUMINAMATH_CALUDE_apple_stack_theorem_l2596_259619

/-- Calculates the number of apples in a cubic-like stack --/
def appleStack (baseSize : Nat) : Nat :=
  let numLayers := baseSize
  List.range numLayers
    |> List.map (fun i => (baseSize - i) ^ 3)
    |> List.sum

theorem apple_stack_theorem :
  appleStack 4 = 100 := by
  sorry

end NUMINAMATH_CALUDE_apple_stack_theorem_l2596_259619


namespace NUMINAMATH_CALUDE_cindy_envelopes_l2596_259671

def envelopes_problem (initial_envelopes : ℕ) (num_friends : ℕ) (envelopes_per_friend : ℕ) : Prop :=
  initial_envelopes - (num_friends * envelopes_per_friend) = 22

theorem cindy_envelopes : envelopes_problem 37 5 3 := by
  sorry

end NUMINAMATH_CALUDE_cindy_envelopes_l2596_259671


namespace NUMINAMATH_CALUDE_ellipse_focal_distance_l2596_259654

/-- For an ellipse with equation x²/m + y²/4 = 1 and focal distance 2, m = 5 -/
theorem ellipse_focal_distance (m : ℝ) : 
  (∀ x y : ℝ, x^2/m + y^2/4 = 1) →  -- Ellipse equation
  2 = 2 * 1 →                      -- Focal distance is 2
  m = 5 := by
sorry

end NUMINAMATH_CALUDE_ellipse_focal_distance_l2596_259654


namespace NUMINAMATH_CALUDE_square_extension_implies_square_l2596_259673

/-- A point in the Euclidean plane -/
structure Point := (x : ℝ) (y : ℝ)

/-- A quadrilateral in the Euclidean plane -/
structure Quadrilateral := (A B C D : Point)

/-- Predicate to check if a quadrilateral is convex -/
def is_convex (q : Quadrilateral) : Prop := sorry

/-- Predicate to check if a quadrilateral is a square -/
def is_square (q : Quadrilateral) : Prop := sorry

/-- Function to extend a line segment -/
def extend_segment (A B : Point) : Point := sorry

/-- Function to calculate the distance between two points -/
def distance (A B : Point) : ℝ := sorry

/-- Main theorem -/
theorem square_extension_implies_square (ABCD : Quadrilateral) 
  (P Q R S : Point)
  (h_convex : is_convex ABCD)
  (h_P : P = extend_segment ABCD.A ABCD.B)
  (h_Q : Q = extend_segment ABCD.B ABCD.C)
  (h_R : R = extend_segment ABCD.C ABCD.D)
  (h_S : S = extend_segment ABCD.D ABCD.A)
  (h_equal_extensions : distance ABCD.B P = distance ABCD.C Q ∧ 
                        distance ABCD.C Q = distance ABCD.D R ∧ 
                        distance ABCD.D R = distance ABCD.A S)
  (h_PQRS_square : is_square ⟨P, Q, R, S⟩) :
  is_square ABCD :=
sorry

end NUMINAMATH_CALUDE_square_extension_implies_square_l2596_259673


namespace NUMINAMATH_CALUDE_bodies_of_revolution_l2596_259652

-- Define the type for geometric solids
inductive GeometricSolid
  | Cylinder
  | HexagonalPyramid
  | Cube
  | Sphere
  | Tetrahedron

-- Define what it means to be a body of revolution
def isBodyOfRevolution : GeometricSolid → Prop :=
  fun solid => match solid with
    | GeometricSolid.Cylinder => True
    | GeometricSolid.Sphere => True
    | _ => False

-- Theorem statement
theorem bodies_of_revolution :
  ∀ (solid : GeometricSolid),
    isBodyOfRevolution solid ↔ (solid = GeometricSolid.Cylinder ∨ solid = GeometricSolid.Sphere) :=
by sorry

end NUMINAMATH_CALUDE_bodies_of_revolution_l2596_259652


namespace NUMINAMATH_CALUDE_stream_speed_l2596_259664

/-- Given a man's downstream and upstream speeds, calculate the speed of the stream. -/
theorem stream_speed (downstream_speed upstream_speed : ℝ) 
  (h1 : downstream_speed = 15)
  (h2 : upstream_speed = 8) :
  (downstream_speed - upstream_speed) / 2 = 3.5 := by
  sorry

end NUMINAMATH_CALUDE_stream_speed_l2596_259664


namespace NUMINAMATH_CALUDE_fraction_equality_l2596_259695

theorem fraction_equality (p q : ℚ) : 
  11 / 7 + (2 * q - p) / (2 * q + p) = 2 → p / q = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l2596_259695


namespace NUMINAMATH_CALUDE_binomial_square_constant_l2596_259668

theorem binomial_square_constant (a : ℚ) : 
  (∃ b c : ℚ, ∀ x, 9 * x^2 + 21 * x + a = (b * x + c)^2) → a = 49 / 4 := by
  sorry

end NUMINAMATH_CALUDE_binomial_square_constant_l2596_259668


namespace NUMINAMATH_CALUDE_complex_division_l2596_259639

/-- Given complex numbers z₁ and z₂ corresponding to points (2, -1) and (0, -1) in the complex plane,
    prove that z₁ / z₂ = 1 + 2i -/
theorem complex_division (z₁ z₂ : ℂ) (h₁ : z₁ = 2 - I) (h₂ : z₂ = -I) : z₁ / z₂ = 1 + 2*I := by
  sorry

end NUMINAMATH_CALUDE_complex_division_l2596_259639


namespace NUMINAMATH_CALUDE_polynomial_identities_identity1_identity2_identity3_identity4_l2596_259607

-- Define the polynomial identities
theorem polynomial_identities (a b : ℝ) :
  ((a + b) * (a^2 - a*b + b^2) = a^3 + b^3) ∧
  ((a - b) * (a^2 + a*b + b^2) = a^3 - b^3) ∧
  ((a + 2*b) * (a^2 - 2*a*b + 4*b^2) = a^3 + 8*b^3) ∧
  (a^3 - 8 = (a - 2) * (a^2 + 2*a + 4)) :=
by sorry

-- Prove each identity separately
theorem identity1 (a b : ℝ) : (a + b) * (a^2 - a*b + b^2) = a^3 + b^3 :=
by sorry

theorem identity2 (a b : ℝ) : (a - b) * (a^2 + a*b + b^2) = a^3 - b^3 :=
by sorry

theorem identity3 (a b : ℝ) : (a + 2*b) * (a^2 - 2*a*b + 4*b^2) = a^3 + 8*b^3 :=
by sorry

theorem identity4 (a : ℝ) : a^3 - 8 = (a - 2) * (a^2 + 2*a + 4) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_identities_identity1_identity2_identity3_identity4_l2596_259607


namespace NUMINAMATH_CALUDE_special_sequence_third_term_l2596_259631

/-- A sequence S with special properties -/
def SpecialSequence (S : ℕ → ℝ) : Prop :=
  ∃ a : ℝ, a > 0 ∧
  S 0 = a^4 ∧
  (∀ n : ℕ, S (n + 1) = 4 * Real.sqrt (S n)) ∧
  (S 2 - S 1 = S 1 - S 0)

/-- The third term of the special sequence can only be 16 or 8√5 - 8 -/
theorem special_sequence_third_term (S : ℕ → ℝ) (h : SpecialSequence S) :
  S 2 = 16 ∨ S 2 = 8 * Real.sqrt 5 - 8 := by
  sorry

end NUMINAMATH_CALUDE_special_sequence_third_term_l2596_259631


namespace NUMINAMATH_CALUDE_problem_solution_l2596_259602

theorem problem_solution (x y z : ℕ) : 
  x > 0 ∧ 
  x = 10 * y + 3 ∧ 
  2 * x = 7 * (3 * y) + 1 ∧ 
  3 * x = 5 * z + 2 → 
  11 * y - x + 7 * z = 219 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l2596_259602


namespace NUMINAMATH_CALUDE_no_bounded_function_satisfying_conditions_l2596_259699

theorem no_bounded_function_satisfying_conditions :
  ¬ ∃ (f : ℝ → ℝ), 
    (∃ (M : ℝ), ∀ x, |f x| ≤ M) ∧ 
    (f 1 > 0) ∧ 
    (∀ x y, (f (x + y))^2 ≥ (f x)^2 + 2 * f (x * y) + (f y)^2) :=
by sorry

end NUMINAMATH_CALUDE_no_bounded_function_satisfying_conditions_l2596_259699


namespace NUMINAMATH_CALUDE_multiplication_puzzle_l2596_259650

theorem multiplication_puzzle (a b : ℕ) : 
  a < 10 → b < 10 → (20 + a) * (10 * b + 3) = 989 → a + b = 7 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_puzzle_l2596_259650


namespace NUMINAMATH_CALUDE_power_product_equality_l2596_259682

theorem power_product_equality (a b : ℝ) : (a * b^2)^2 = a^2 * b^4 := by
  sorry

end NUMINAMATH_CALUDE_power_product_equality_l2596_259682


namespace NUMINAMATH_CALUDE_base8_52_equals_base10_42_l2596_259601

/-- Converts a base-8 number to base-10 --/
def base8ToBase10 (digits : List Nat) : Nat :=
  digits.foldr (fun d acc => acc * 8 + d) 0

theorem base8_52_equals_base10_42 :
  base8ToBase10 [5, 2] = 42 := by
  sorry

end NUMINAMATH_CALUDE_base8_52_equals_base10_42_l2596_259601


namespace NUMINAMATH_CALUDE_sara_pumpkins_l2596_259693

def pumpkins_grown : ℕ := 43
def pumpkins_eaten : ℕ := 23

theorem sara_pumpkins : pumpkins_grown - pumpkins_eaten = 20 := by
  sorry

end NUMINAMATH_CALUDE_sara_pumpkins_l2596_259693


namespace NUMINAMATH_CALUDE_min_reciprocal_sum_min_reciprocal_sum_attainable_l2596_259620

theorem min_reciprocal_sum (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (sum_eq_one : a + b + c = 1) : 
  1/a + 1/b + 1/c ≥ 9 := by
  sorry

theorem min_reciprocal_sum_attainable : 
  ∃ a b c : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b + c = 1 ∧ 1/a + 1/b + 1/c = 9 := by
  sorry

end NUMINAMATH_CALUDE_min_reciprocal_sum_min_reciprocal_sum_attainable_l2596_259620


namespace NUMINAMATH_CALUDE_people_left_of_kolya_l2596_259634

/-- Given a class lineup with the following conditions:
  * There are 12 people to the right of Kolya
  * There are 20 people to the left of Sasha
  * There are 8 people to the right of Sasha
  Prove that there are 16 people to the left of Kolya -/
theorem people_left_of_kolya
  (right_of_kolya : ℕ)
  (left_of_sasha : ℕ)
  (right_of_sasha : ℕ)
  (h1 : right_of_kolya = 12)
  (h2 : left_of_sasha = 20)
  (h3 : right_of_sasha = 8) :
  left_of_sasha + right_of_sasha + 1 - right_of_kolya - 1 = 16 := by
  sorry

end NUMINAMATH_CALUDE_people_left_of_kolya_l2596_259634


namespace NUMINAMATH_CALUDE_pencils_added_l2596_259676

theorem pencils_added (initial : ℕ) (final : ℕ) (h1 : initial = 115) (h2 : final = 215) :
  final - initial = 100 := by
sorry

end NUMINAMATH_CALUDE_pencils_added_l2596_259676


namespace NUMINAMATH_CALUDE_colored_plane_theorem_l2596_259611

-- Define a type for colors
inductive Color
| Red
| Blue

-- Define a point in the plane
structure Point where
  x : ℝ
  y : ℝ

-- Define a function that assigns a color to each point in the plane
def colorAssignment : Point → Color := sorry

-- Define what it means for three points to form an equilateral triangle
def isEquilateralTriangle (A B C : Point) : Prop := sorry

-- Define what it means for a point to be the midpoint of two other points
def isMidpoint (M A C : Point) : Prop := sorry

theorem colored_plane_theorem :
  -- Part (a)
  (¬ ∃ A B C : Point, isEquilateralTriangle A B C ∧ colorAssignment A = colorAssignment B ∧ colorAssignment B = colorAssignment C →
   ∃ A B C : Point, colorAssignment A = colorAssignment B ∧ colorAssignment B = colorAssignment C ∧ isMidpoint B A C) ∧
  -- Part (b)
  ∃ A B C : Point, isEquilateralTriangle A B C ∧ colorAssignment A = colorAssignment B ∧ colorAssignment B = colorAssignment C :=
by sorry

end NUMINAMATH_CALUDE_colored_plane_theorem_l2596_259611


namespace NUMINAMATH_CALUDE_savanna_animal_count_l2596_259646

/-- The number of animals in Savanna National Park -/
def savanna_total (safari_lions : ℕ) : ℕ :=
  let safari_snakes := safari_lions / 2
  let safari_giraffes := safari_snakes - 10
  let savanna_lions := safari_lions * 2
  let savanna_snakes := safari_snakes * 3
  let savanna_giraffes := safari_giraffes + 20
  savanna_lions + savanna_snakes + savanna_giraffes

/-- Theorem stating the total number of animals in Savanna National Park -/
theorem savanna_animal_count : savanna_total 100 = 410 := by
  sorry

end NUMINAMATH_CALUDE_savanna_animal_count_l2596_259646


namespace NUMINAMATH_CALUDE_max_consecutive_interesting_numbers_l2596_259609

/-- A function that checks if a number is interesting (has at least one digit divisible by 3) -/
def is_interesting (n : ℕ) : Prop :=
  ∃ d, d ∈ n.digits 10 ∧ d % 3 = 0

/-- The theorem stating the maximum number of consecutive interesting three-digit numbers -/
theorem max_consecutive_interesting_numbers :
  ∃ start : ℕ,
    start ≥ 100 ∧
    start + 121 ≤ 999 ∧
    (∀ k : ℕ, k ∈ Finset.range 122 → is_interesting (start + k)) ∧
    (∀ m : ℕ, m > 122 →
      ¬∃ s : ℕ, s ≥ 100 ∧ s + m - 1 ≤ 999 ∧
        ∀ j : ℕ, j ∈ Finset.range m → is_interesting (s + j)) :=
by
  sorry


end NUMINAMATH_CALUDE_max_consecutive_interesting_numbers_l2596_259609


namespace NUMINAMATH_CALUDE_fraction_sum_equals_cube_sum_l2596_259688

theorem fraction_sum_equals_cube_sum (x : ℝ) : 
  ((x - 1) * (x + 1)) / (x * (x - 1) + 1) + (2 * (0.5 - x)) / (x * (1 - x) - 1) = 
  ((x - 1) * (x + 1) / (x * (x - 1) + 1))^3 + (2 * (0.5 - x) / (x * (1 - x) - 1))^3 :=
by sorry

end NUMINAMATH_CALUDE_fraction_sum_equals_cube_sum_l2596_259688


namespace NUMINAMATH_CALUDE_triangle_properties_l2596_259643

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def satisfies_conditions (t : Triangle) : Prop :=
  (2 * t.b - t.c) * Real.cos t.A = t.a * Real.cos t.C ∧
  t.a = Real.sqrt 13 ∧
  t.b + t.c = 5

-- Theorem statement
theorem triangle_properties (t : Triangle) 
  (h : satisfies_conditions t) : 
  t.A = π / 3 ∧ 
  (1 / 2 : ℝ) * t.b * t.c * Real.sin t.A = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l2596_259643
