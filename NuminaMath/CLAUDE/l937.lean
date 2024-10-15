import Mathlib

namespace NUMINAMATH_CALUDE_largest_possible_z_value_l937_93775

open Complex

theorem largest_possible_z_value (a b c d z w : ℂ) 
  (h1 : abs a = abs b)
  (h2 : abs b = abs c)
  (h3 : abs c = abs d)
  (h4 : abs a > 0)
  (h5 : a * z^3 + b * w * z^2 + c * z + d = 0)
  (h6 : abs w = 1/2) :
  abs z ≤ 1 ∧ ∃ a b c d z w : ℂ, 
    abs a = abs b ∧ 
    abs b = abs c ∧ 
    abs c = abs d ∧ 
    abs a > 0 ∧
    a * z^3 + b * w * z^2 + c * z + d = 0 ∧
    abs w = 1/2 ∧
    abs z = 1 := by
  sorry

end NUMINAMATH_CALUDE_largest_possible_z_value_l937_93775


namespace NUMINAMATH_CALUDE_kevin_wins_l937_93743

/-- Represents a player in the chess game -/
inductive Player : Type
| Peter : Player
| Emma : Player
| Kevin : Player

/-- Represents the game results for each player -/
structure GameResults :=
  (wins : Player → ℕ)
  (losses : Player → ℕ)

/-- The theorem to prove -/
theorem kevin_wins (results : GameResults) : 
  results.wins Player.Peter = 4 →
  results.losses Player.Peter = 2 →
  results.wins Player.Emma = 3 →
  results.losses Player.Emma = 3 →
  results.losses Player.Kevin = 3 →
  results.wins Player.Kevin = 1 := by
  sorry


end NUMINAMATH_CALUDE_kevin_wins_l937_93743


namespace NUMINAMATH_CALUDE_missile_interception_time_l937_93746

/-- The time for a missile to intercept a circling plane -/
theorem missile_interception_time
  (r : ℝ)             -- radius of the plane's circular path
  (v : ℝ)             -- speed of both the plane and the missile
  (h : r = 10)        -- given radius is 10 km
  (k : v = 1000)      -- given speed is 1000 km/h
  : ∃ t : ℝ,          -- there exists a time t such that
    t = 18 * Real.pi ∧ -- t equals 18π
    t * (5 / 18) = (2 * Real.pi * r) / 4 / v -- t converted to hours equals quarter circumference divided by speed
    :=
by sorry

end NUMINAMATH_CALUDE_missile_interception_time_l937_93746


namespace NUMINAMATH_CALUDE_cookie_radius_l937_93771

/-- The equation of the cookie's boundary -/
def cookie_equation (x y : ℝ) : Prop :=
  x^2 + y^2 + 2*x - 4*y = 8

/-- The cookie is a circle -/
def is_circle (center : ℝ × ℝ) (radius : ℝ) : Prop :=
  ∀ x y, cookie_equation x y ↔ (x - center.1)^2 + (y - center.2)^2 = radius^2

/-- The radius of the cookie is √13 -/
theorem cookie_radius :
  ∃ center, is_circle center (Real.sqrt 13) :=
sorry

end NUMINAMATH_CALUDE_cookie_radius_l937_93771


namespace NUMINAMATH_CALUDE_uniform_production_theorem_l937_93780

def device_A_rate : ℚ := 1 / 90
def device_B_rate : ℚ := 1 / 60
def simultaneous_work_days : ℕ := 30
def remaining_days : ℕ := 13

theorem uniform_production_theorem :
  (∃ x : ℚ, x * (device_A_rate + device_B_rate) = 1 ∧ x = 36) ∧
  (∃ y : ℚ, (simultaneous_work_days + y) * device_A_rate + simultaneous_work_days * device_B_rate = 1 ∧ y > remaining_days) :=
by sorry

end NUMINAMATH_CALUDE_uniform_production_theorem_l937_93780


namespace NUMINAMATH_CALUDE_spider_plant_production_l937_93779

/-- Represents the number of baby plants produced by a spider plant over time -/
def babyPlants (plantsPerProduction : ℕ) (productionsPerYear : ℕ) (years : ℕ) : ℕ :=
  plantsPerProduction * productionsPerYear * years

/-- Theorem: A spider plant producing 2 baby plants 2 times a year will produce 16 baby plants after 4 years -/
theorem spider_plant_production :
  babyPlants 2 2 4 = 16 := by
  sorry

end NUMINAMATH_CALUDE_spider_plant_production_l937_93779


namespace NUMINAMATH_CALUDE_parabola_x_axis_intersection_l937_93739

theorem parabola_x_axis_intersection :
  let f (x : ℝ) := x^2 - 2*x - 3
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ f x₁ = 0 ∧ f x₂ = 0 ∧ ∀ x, f x = 0 → x = x₁ ∨ x = x₂ :=
by sorry

end NUMINAMATH_CALUDE_parabola_x_axis_intersection_l937_93739


namespace NUMINAMATH_CALUDE_discounted_biographies_count_l937_93706

theorem discounted_biographies_count (biography_price mystery_price total_savings mystery_count total_discount_rate mystery_discount_rate : ℝ) 
  (h1 : biography_price = 20)
  (h2 : mystery_price = 12)
  (h3 : total_savings = 19)
  (h4 : mystery_count = 3)
  (h5 : total_discount_rate = 0.43)
  (h6 : mystery_discount_rate = 0.375) :
  ∃ (biography_count : ℕ), 
    biography_count = 5 ∧ 
    biography_count * (biography_price * (total_discount_rate - mystery_discount_rate)) + 
    mystery_count * (mystery_price * mystery_discount_rate) = total_savings :=
by sorry

end NUMINAMATH_CALUDE_discounted_biographies_count_l937_93706


namespace NUMINAMATH_CALUDE_exponential_equation_solution_l937_93700

theorem exponential_equation_solution :
  ∃ (y₁ y₂ : ℝ), y₁ ≠ y₂ ∧
  (∀ y : ℝ, (2 : ℝ) ^ (y^2 - 5*y - 6) = (4 : ℝ) ^ (y - 5) ↔ y = y₁ ∨ y = y₂) ∧
  y₁ + y₂ = 5 :=
sorry

end NUMINAMATH_CALUDE_exponential_equation_solution_l937_93700


namespace NUMINAMATH_CALUDE_odd_product_probability_l937_93735

theorem odd_product_probability (n : ℕ) (hn : n = 1000) :
  let odd_count := (n + 1) / 2
  let total_count := n
  let p := (odd_count / total_count) * ((odd_count - 1) / (total_count - 1)) * ((odd_count - 2) / (total_count - 2))
  p < 1 / 8 := by
sorry


end NUMINAMATH_CALUDE_odd_product_probability_l937_93735


namespace NUMINAMATH_CALUDE_mary_chewing_gums_l937_93721

theorem mary_chewing_gums (total sam sue : ℕ) (h1 : total = 30) (h2 : sam = 10) (h3 : sue = 15) :
  total - (sam + sue) = 5 := by
  sorry

end NUMINAMATH_CALUDE_mary_chewing_gums_l937_93721


namespace NUMINAMATH_CALUDE_unique_four_digit_numbers_l937_93701

theorem unique_four_digit_numbers : ∃! (x y : ℕ), 
  (1000 ≤ x ∧ x < 10000) ∧ 
  (1000 ≤ y ∧ y < 10000) ∧ 
  y > x ∧ 
  (∃ (a n : ℕ), 1 ≤ a ∧ a < 10 ∧ y = a * 10^n) ∧
  (x / 1000 + (x / 100) % 10 = y - x) ∧
  (y - x = 5 * (y / 1000)) ∧
  x = 1990 ∧ 
  y = 2000 := by
sorry

end NUMINAMATH_CALUDE_unique_four_digit_numbers_l937_93701


namespace NUMINAMATH_CALUDE_sphere_volume_l937_93796

theorem sphere_volume (r : ℝ) (h1 : r > 0) (h2 : π = (r ^ 2 - 1 ^ 2)) : 
  (4 / 3 : ℝ) * π * r ^ 3 = (8 * Real.sqrt 2 / 3 : ℝ) * π := by
  sorry

end NUMINAMATH_CALUDE_sphere_volume_l937_93796


namespace NUMINAMATH_CALUDE_donut_distribution_proof_l937_93763

/-- The number of ways to distribute donuts satisfying the given conditions -/
def donut_combinations : ℕ := 126

/-- The number of donut types -/
def num_types : ℕ := 5

/-- The total number of donuts to be purchased -/
def total_donuts : ℕ := 10

/-- The number of remaining donuts after selecting one of each type -/
def remaining_donuts : ℕ := total_donuts - num_types

/-- Binomial coefficient calculation -/
def binom (n k : ℕ) : ℕ :=
  if k > n then 0
  else (List.range k).foldl (fun m i => m * (n - i) / (i + 1)) 1

theorem donut_distribution_proof :
  binom (remaining_donuts + num_types - 1) (num_types - 1) = donut_combinations :=
by sorry

end NUMINAMATH_CALUDE_donut_distribution_proof_l937_93763


namespace NUMINAMATH_CALUDE_certain_number_equals_sixteen_l937_93754

theorem certain_number_equals_sixteen : ∃ x : ℝ, x^5 = 4^10 ∧ x = 16 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_equals_sixteen_l937_93754


namespace NUMINAMATH_CALUDE_average_carnations_example_l937_93795

/-- The average number of carnations in three bouquets -/
def average_carnations (b1 b2 b3 : ℕ) : ℚ :=
  (b1 + b2 + b3 : ℚ) / 3

/-- Theorem: The average number of carnations in three bouquets containing 9, 14, and 13 carnations respectively is 12 -/
theorem average_carnations_example : average_carnations 9 14 13 = 12 := by
  sorry

end NUMINAMATH_CALUDE_average_carnations_example_l937_93795


namespace NUMINAMATH_CALUDE_adjustment_schemes_no_adjacent_boys_arrangements_specific_position_arrangements_l937_93709

-- Define the number of boys and girls
def num_boys : ℕ := 3
def num_girls : ℕ := 4
def total_people : ℕ := num_boys + num_girls

-- Statement A
theorem adjustment_schemes :
  (Nat.choose total_people 3) * 2 = 70 := by sorry

-- Statement B
theorem no_adjacent_boys_arrangements :
  (Nat.factorial num_girls) * (Nat.factorial (num_girls + 1) / Nat.factorial (num_girls + 1 - num_boys)) = 1440 := by sorry

-- Statement D
theorem specific_position_arrangements :
  Nat.factorial total_people - 2 * Nat.factorial (total_people - 1) + Nat.factorial (total_people - 2) = 3720 := by sorry

end NUMINAMATH_CALUDE_adjustment_schemes_no_adjacent_boys_arrangements_specific_position_arrangements_l937_93709


namespace NUMINAMATH_CALUDE_sphere_volume_l937_93770

theorem sphere_volume (r : ℝ) (h : r > 0) :
  (∃ (d : ℝ), d > 0 ∧ d < r ∧
    4 = (r^2 - d^2).sqrt ∧
    d = 3) →
  (4 / 3 * Real.pi * r^3 = 500 * Real.pi / 3) :=
by sorry

end NUMINAMATH_CALUDE_sphere_volume_l937_93770


namespace NUMINAMATH_CALUDE_min_value_inequality_l937_93772

theorem min_value_inequality (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 4) :
  (1 / x + 4 / y) ≥ 9 / 4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_inequality_l937_93772


namespace NUMINAMATH_CALUDE_fishing_competition_l937_93734

theorem fishing_competition (n : ℕ) : 
  (∃ (m : ℕ), n * m + 11 * (m + 10) = n^2 + 5*n + 22) → n = 11 :=
by sorry

end NUMINAMATH_CALUDE_fishing_competition_l937_93734


namespace NUMINAMATH_CALUDE_triangle_stack_impossibility_l937_93710

theorem triangle_stack_impossibility : ¬ ∃ (n : ℕ), 6 * n = 165 := by
  sorry

end NUMINAMATH_CALUDE_triangle_stack_impossibility_l937_93710


namespace NUMINAMATH_CALUDE_inequality_chain_l937_93705

theorem inequality_chain (a b d m : ℝ) 
  (h1 : a > b) (h2 : b > d) (h3 : d ≥ m) : a > m := by
  sorry

end NUMINAMATH_CALUDE_inequality_chain_l937_93705


namespace NUMINAMATH_CALUDE_remainder_division_l937_93781

theorem remainder_division (y k : ℤ) (h : y = 264 * k + 42) : y ≡ 20 [ZMOD 22] := by
  sorry

end NUMINAMATH_CALUDE_remainder_division_l937_93781


namespace NUMINAMATH_CALUDE_cheese_pizzas_sold_l937_93733

/-- The number of cheese pizzas sold by a pizza store on Friday -/
def cheese_pizzas (pepperoni bacon total : ℕ) : ℕ :=
  total - (pepperoni + bacon)

/-- Theorem stating the number of cheese pizzas sold -/
theorem cheese_pizzas_sold :
  cheese_pizzas 2 6 14 = 6 := by
  sorry

end NUMINAMATH_CALUDE_cheese_pizzas_sold_l937_93733


namespace NUMINAMATH_CALUDE_binomial_expansion_coefficient_l937_93715

theorem binomial_expansion_coefficient (m : ℝ) (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x, (2 - m * x)^5 = a₀ + a₁ * x + a₂ * x^2 + a₃ * x^3 + a₄ * x^4 + a₅ * x^5) →
  a₃ = 40 →
  m = -1 := by
sorry

end NUMINAMATH_CALUDE_binomial_expansion_coefficient_l937_93715


namespace NUMINAMATH_CALUDE_square_area_problem_l937_93723

theorem square_area_problem (small_square_area : ℝ) (triangle_area : ℝ) :
  small_square_area = 16 →
  triangle_area = 1 →
  ∃ (large_square_area : ℝ),
    large_square_area = 18 ∧
    ∃ (small_side large_side triangle_side : ℝ),
      small_side ^ 2 = small_square_area ∧
      triangle_side ^ 2 = 2 ∧
      large_side ^ 2 = large_square_area ∧
      large_side ^ 2 = small_side ^ 2 + triangle_side ^ 2 := by
  sorry


end NUMINAMATH_CALUDE_square_area_problem_l937_93723


namespace NUMINAMATH_CALUDE_supplementary_angles_equal_l937_93707

/-- Two angles that are supplementary to the same angle are equal. -/
theorem supplementary_angles_equal (α β γ : Real) (h1 : α + γ = 180) (h2 : β + γ = 180) : α = β := by
  sorry

end NUMINAMATH_CALUDE_supplementary_angles_equal_l937_93707


namespace NUMINAMATH_CALUDE_cube_equality_iff_three_l937_93716

theorem cube_equality_iff_three (x : ℝ) (hx : x ≠ 0) :
  (3 * x)^3 = (9 * x)^2 ↔ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_cube_equality_iff_three_l937_93716


namespace NUMINAMATH_CALUDE_soccer_camp_afternoon_attendance_l937_93703

theorem soccer_camp_afternoon_attendance (total_kids : ℕ) 
  (h1 : total_kids = 2000)
  (h2 : total_kids % 2 = 0)
  (soccer_kids : ℕ) 
  (h3 : soccer_kids = total_kids / 2)
  (morning_ratio : ℚ)
  (h4 : morning_ratio = 1 / 4)
  (morning_kids : ℕ)
  (h5 : morning_kids = ⌊soccer_kids * morning_ratio⌋)
  (afternoon_kids : ℕ)
  (h6 : afternoon_kids = soccer_kids - morning_kids) :
  afternoon_kids = 750 := by
sorry

end NUMINAMATH_CALUDE_soccer_camp_afternoon_attendance_l937_93703


namespace NUMINAMATH_CALUDE_a_is_perfect_square_l937_93786

-- Define the sequence c_n
def c : ℕ → ℤ
  | 0 => 1
  | 1 => 0
  | 2 => 2005
  | (n + 3) => -3 * c n - 4 * c (n + 1) + 2008

-- Define the sequence a_n
def a (n : ℕ) : ℤ :=
  if n ≥ 2 then
    5 * (c (n + 1) - c n) * (502 - c (n - 1) - c (n - 2)) + 4^n * 2004 * 501
  else
    0  -- Define a value for n < 2, though it's not used in the theorem

-- Theorem statement
theorem a_is_perfect_square (n : ℕ) (h : n > 2) :
  ∃ k : ℤ, a n = k^2 := by
  sorry

end NUMINAMATH_CALUDE_a_is_perfect_square_l937_93786


namespace NUMINAMATH_CALUDE_smallest_solution_abs_equation_l937_93729

theorem smallest_solution_abs_equation :
  ∃ (x : ℝ), x * |x| = 3 * x + 2 ∧
  ∀ (y : ℝ), y * |y| = 3 * y + 2 → x ≤ y ∧
  x = -2 := by
sorry

end NUMINAMATH_CALUDE_smallest_solution_abs_equation_l937_93729


namespace NUMINAMATH_CALUDE_abc_inequality_l937_93764

theorem abc_inequality (a b c : ℝ) 
  (ha : 0 < a ∧ a < 1) 
  (hb : 0 < b ∧ b < 1) 
  (hc : 0 < c ∧ c < 1)
  (eq_a : a - 2 = Real.log (a / 2))
  (eq_b : b - 3 = Real.log (b / 3))
  (eq_c : c - 3 = Real.log (c / 2)) :
  c < b ∧ b < a := by
  sorry

end NUMINAMATH_CALUDE_abc_inequality_l937_93764


namespace NUMINAMATH_CALUDE_log_expression_equals_half_l937_93747

theorem log_expression_equals_half :
  (1/2) * (Real.log 12 / Real.log 6) - (Real.log (Real.sqrt 2) / Real.log 6) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_log_expression_equals_half_l937_93747


namespace NUMINAMATH_CALUDE_triangle_max_area_l937_93702

theorem triangle_max_area (A B C : Real) (a b c : Real) :
  a > 0 ∧ b > 0 ∧ c > 0 →
  a * c = 6 →
  Real.sin B + 2 * Real.sin C * Real.cos A = 0 →
  (∀ S : Real, S = (1/2) * a * c * Real.sin B → S ≤ 3/2) ∧
  (∃ S : Real, S = (1/2) * a * c * Real.sin B ∧ S = 3/2) :=
by sorry

end NUMINAMATH_CALUDE_triangle_max_area_l937_93702


namespace NUMINAMATH_CALUDE_tom_apple_count_l937_93799

/-- The number of apples each person has -/
structure AppleCount where
  phillip : ℕ
  ben : ℕ
  tom : ℕ

/-- The conditions of the problem -/
def problem_conditions (ac : AppleCount) : Prop :=
  ac.phillip = 40 ∧
  ac.ben = ac.phillip + 8 ∧
  ac.tom = (3 * ac.ben) / 8

/-- The theorem stating that Tom has 18 apples given the problem conditions -/
theorem tom_apple_count (ac : AppleCount) (h : problem_conditions ac) : ac.tom = 18 := by
  sorry

#check tom_apple_count

end NUMINAMATH_CALUDE_tom_apple_count_l937_93799


namespace NUMINAMATH_CALUDE_new_salary_after_raise_l937_93714

def original_salary : ℝ := 500
def raise_percentage : ℝ := 6

theorem new_salary_after_raise :
  original_salary * (1 + raise_percentage / 100) = 530 := by
  sorry

end NUMINAMATH_CALUDE_new_salary_after_raise_l937_93714


namespace NUMINAMATH_CALUDE_square_root_value_l937_93765

theorem square_root_value (x : ℝ) (h : x = 5) : Real.sqrt (x - 3) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_square_root_value_l937_93765


namespace NUMINAMATH_CALUDE_inequality_proof_l937_93727

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h_prod : x * y * z = 1) :
  x / (y + z) + y / (x + z) + z / (x + y) ≤ x * Real.sqrt x / 2 + y * Real.sqrt y / 2 + z * Real.sqrt z / 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l937_93727


namespace NUMINAMATH_CALUDE_ron_four_times_maurice_age_l937_93725

/-- The number of years in the future when Ron will be four times as old as Maurice -/
def years_until_four_times_age : ℕ → ℕ → ℕ 
| ron_age, maurice_age => 
  let x : ℕ := (ron_age - 4 * maurice_age) / 3
  x

theorem ron_four_times_maurice_age (ron_current_age maurice_current_age : ℕ) 
  (h1 : ron_current_age = 43)
  (h2 : maurice_current_age = 7) : 
  years_until_four_times_age ron_current_age maurice_current_age = 5 := by
  sorry

end NUMINAMATH_CALUDE_ron_four_times_maurice_age_l937_93725


namespace NUMINAMATH_CALUDE_coefficient_a6_l937_93783

theorem coefficient_a6 (x a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℝ) :
  x^2 + x^7 = a₀ + a₁*(x+1) + a₂*(x+1)^2 + a₃*(x+1)^3 + a₄*(x+1)^4 + a₅*(x+1)^5 + a₆*(x+1)^6 + a₇*(x+1)^7 →
  a₆ = -7 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_a6_l937_93783


namespace NUMINAMATH_CALUDE_dvd_cost_l937_93790

/-- The cost of each DVD given the total number of movies, trade-in value per VHS, and total replacement cost. -/
theorem dvd_cost (total_movies : ℕ) (vhs_trade_value : ℚ) (total_replacement_cost : ℚ) :
  total_movies = 100 →
  vhs_trade_value = 2 →
  total_replacement_cost = 800 →
  (total_replacement_cost - (total_movies : ℚ) * vhs_trade_value) / total_movies = 6 := by
  sorry

end NUMINAMATH_CALUDE_dvd_cost_l937_93790


namespace NUMINAMATH_CALUDE_carey_chairs_moved_l937_93776

/-- Proves that Carey moved 28 chairs given the total chairs, Pat's chairs, and remaining chairs. -/
theorem carey_chairs_moved (total : ℕ) (pat_moved : ℕ) (remaining : ℕ) 
  (h1 : total = 74)
  (h2 : pat_moved = 29)
  (h3 : remaining = 17) :
  total - pat_moved - remaining = 28 := by
  sorry

#check carey_chairs_moved

end NUMINAMATH_CALUDE_carey_chairs_moved_l937_93776


namespace NUMINAMATH_CALUDE_h_of_neg_one_eq_three_l937_93788

-- Define the functions
def f (x : ℝ) : ℝ := 3 * x + 6
def g (x : ℝ) : ℝ := x^3
def h (x : ℝ) : ℝ := f (g x)

-- State the theorem
theorem h_of_neg_one_eq_three : h (-1) = 3 := by sorry

end NUMINAMATH_CALUDE_h_of_neg_one_eq_three_l937_93788


namespace NUMINAMATH_CALUDE_language_group_selection_ways_l937_93741

/-- Represents a group of people who know languages -/
structure LanguageGroup where
  total : Nat
  english : Nat
  japanese : Nat
  both : Nat

/-- The number of ways to select one person who knows English and another who knows Japanese -/
def selectWays (group : LanguageGroup) : Nat :=
  (group.english - group.both) * group.japanese + group.both * (group.japanese - 1)

/-- Theorem stating the number of ways to select people in the given scenario -/
theorem language_group_selection_ways :
  ∃ (group : LanguageGroup),
    group.total = 9 ∧
    group.english = 7 ∧
    group.japanese = 3 ∧
    group.total = (group.english - group.both) + (group.japanese - group.both) + group.both ∧
    selectWays group = 20 := by
  sorry

end NUMINAMATH_CALUDE_language_group_selection_ways_l937_93741


namespace NUMINAMATH_CALUDE_min_fountains_correct_l937_93766

/-- Represents a water fountain on a grid -/
structure Fountain where
  row : Nat
  col : Nat

/-- Checks if a fountain can spray a given square -/
def can_spray (f : Fountain) (row col : Nat) : Bool :=
  (f.row = row && (f.col = col - 1 || f.col = col + 1)) ||
  (f.col = col && (f.row = row - 1 || f.row = row + 1 || f.row = row - 2))

/-- Calculates the minimum number of fountains required for a given grid size -/
def min_fountains (m n : Nat) : Nat :=
  if m = 4 then
    2 * ((n + 2) / 3)
  else if m = 3 then
    3 * ((n + 2) / 3)
  else
    0  -- undefined for other cases

theorem min_fountains_correct (m n : Nat) :
  (m = 4 || m = 3) →
  ∃ (fountains : List Fountain),
    (fountains.length = min_fountains m n) ∧
    (∀ row col, row < m ∧ col < n →
      ∃ f ∈ fountains, can_spray f row col) :=
by sorry

#eval min_fountains 4 10  -- Expected: 8
#eval min_fountains 3 10  -- Expected: 12

end NUMINAMATH_CALUDE_min_fountains_correct_l937_93766


namespace NUMINAMATH_CALUDE_dragon_castle_theorem_l937_93742

/-- Represents the configuration of a dragon tethered to a cylindrical castle -/
structure DragonCastle where
  castle_radius : ℝ
  chain_length : ℝ
  chain_height : ℝ
  dragon_distance : ℝ

/-- Calculates the length of the chain touching the castle -/
def chain_on_castle (dc : DragonCastle) : ℝ :=
  sorry

/-- Theorem stating the properties of the dragon-castle configuration -/
theorem dragon_castle_theorem (dc : DragonCastle) 
  (h1 : dc.castle_radius = 10)
  (h2 : dc.chain_length = 30)
  (h3 : dc.chain_height = 6)
  (h4 : dc.dragon_distance = 6) :
  ∃ (a b c : ℕ), 
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    Nat.Prime c ∧
    chain_on_castle dc = (a - Real.sqrt b) / c ∧
    a = 90 ∧ b = 1440 ∧ c = 3 ∧
    a + b + c = 1533 :=
by sorry

end NUMINAMATH_CALUDE_dragon_castle_theorem_l937_93742


namespace NUMINAMATH_CALUDE_inverse_proportion_problem_l937_93789

/-- Two real numbers are inversely proportional -/
def InverseProportion (x y : ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ x * y = k

theorem inverse_proportion_problem (x₁ x₂ y₁ y₂ : ℝ) 
  (h1 : InverseProportion x₁ y₁)
  (h2 : InverseProportion x₂ y₂)
  (h3 : x₁ = 40)
  (h4 : y₁ = 5)
  (h5 : y₂ = 10) :
  x₂ = 20 := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_problem_l937_93789


namespace NUMINAMATH_CALUDE_complex_simplification_l937_93750

theorem complex_simplification :
  ((-5 - 3*Complex.I) - (2 - 7*Complex.I)) * 2 = -14 + 8*Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_simplification_l937_93750


namespace NUMINAMATH_CALUDE_road_travel_cost_l937_93773

/-- Calculate the cost of traveling two intersecting roads on a rectangular lawn. -/
theorem road_travel_cost
  (lawn_length lawn_width road_width : ℕ)
  (cost_per_sqm : ℚ)
  (h1 : lawn_length = 80)
  (h2 : lawn_width = 40)
  (h3 : road_width = 10)
  (h4 : cost_per_sqm = 3) :
  (((lawn_length * road_width + lawn_width * road_width) - road_width * road_width) : ℚ) * cost_per_sqm = 3300 :=
by sorry

end NUMINAMATH_CALUDE_road_travel_cost_l937_93773


namespace NUMINAMATH_CALUDE_negation_of_universal_statement_l937_93711

theorem negation_of_universal_statement :
  (¬ ∀ x : ℝ, x^2 + x + 1 > 0) ↔ (∃ x : ℝ, x^2 + x + 1 ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_statement_l937_93711


namespace NUMINAMATH_CALUDE_arithmetic_sequence_formula_l937_93782

/-- An arithmetic sequence with first three terms a-1, a+1, and 2a+3 has the general formula 2n - 3 for its nth term. -/
theorem arithmetic_sequence_formula (a : ℝ) : 
  (∃ (seq : ℕ → ℝ), 
    seq 1 = a - 1 ∧ 
    seq 2 = a + 1 ∧ 
    seq 3 = 2*a + 3 ∧ 
    ∀ n m : ℕ, seq (n + 1) - seq n = seq (m + 1) - seq m) → 
  (∃ (seq : ℕ → ℝ), 
    seq 1 = a - 1 ∧ 
    seq 2 = a + 1 ∧ 
    seq 3 = 2*a + 3 ∧ 
    ∀ n : ℕ, seq n = 2*n - 3) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_formula_l937_93782


namespace NUMINAMATH_CALUDE_owls_on_fence_l937_93792

theorem owls_on_fence (initial_owls joining_owls : ℕ) : 
  initial_owls = 3 → joining_owls = 2 → initial_owls + joining_owls = 5 :=
by sorry

end NUMINAMATH_CALUDE_owls_on_fence_l937_93792


namespace NUMINAMATH_CALUDE_equation_equivalence_l937_93759

theorem equation_equivalence (y : ℝ) :
  (8 * y^2 + 90 * y + 5) / (3 * y^2 + 4 * y + 49) = 4 * y + 1 →
  12 * y^3 + 11 * y^2 + 110 * y + 44 = 0 := by
  sorry

end NUMINAMATH_CALUDE_equation_equivalence_l937_93759


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l937_93738

/-- An arithmetic sequence with specific conditions -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (a₁ d : ℝ), ∀ n : ℕ, a n = a₁ + (n - 1 : ℝ) * d

theorem arithmetic_sequence_common_difference 
  (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a) 
  (h_prod : a 1 * a 3 = 8) 
  (h_second : a 2 = 3) :
  ∃ d : ℝ, (d = 1 ∨ d = -1) ∧ 
    ∀ n : ℕ, a n = a 1 + (n - 1 : ℝ) * d :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l937_93738


namespace NUMINAMATH_CALUDE_fractional_equation_solution_l937_93718

theorem fractional_equation_solution :
  ∃ x : ℝ, (2 - x) / (x - 3) + 1 / (3 - x) = 1 ∧ x = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_fractional_equation_solution_l937_93718


namespace NUMINAMATH_CALUDE_division_result_and_thousandths_digit_l937_93758

theorem division_result_and_thousandths_digit : 
  let result : ℚ := 57 / 5000
  (result = 0.0114) ∧ 
  (⌊result * 1000⌋ % 10 = 4) := by
  sorry

end NUMINAMATH_CALUDE_division_result_and_thousandths_digit_l937_93758


namespace NUMINAMATH_CALUDE_expand_product_l937_93730

theorem expand_product (x : ℝ) : (x + 5) * (x + 7) = x^2 + 12*x + 35 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l937_93730


namespace NUMINAMATH_CALUDE_ladder_problem_l937_93719

theorem ladder_problem (ladder_length height : ℝ) 
  (h1 : ladder_length = 13)
  (h2 : height = 12) :
  ∃ base : ℝ, base^2 + height^2 = ladder_length^2 ∧ base = 5 := by
sorry

end NUMINAMATH_CALUDE_ladder_problem_l937_93719


namespace NUMINAMATH_CALUDE_expression_value_l937_93797

theorem expression_value (a b : ℤ) (ha : a = 4) (hb : b = -2) : 
  -a - b^2 + a*b + a^2 = 0 := by sorry

end NUMINAMATH_CALUDE_expression_value_l937_93797


namespace NUMINAMATH_CALUDE_min_value_of_exponential_sum_l937_93785

theorem min_value_of_exponential_sum (x y : ℝ) (h : x + 3 * y - 1 = 0) :
  ∃ (m : ℝ), m = 2 * Real.sqrt 2 ∧ ∀ (z : ℝ), 2^x + 8^y ≥ z → z ≥ m :=
sorry

end NUMINAMATH_CALUDE_min_value_of_exponential_sum_l937_93785


namespace NUMINAMATH_CALUDE_zahs_to_bahs_conversion_l937_93787

/-- Conversion rates between different currencies -/
structure CurrencyRates where
  bah_to_rah : ℚ
  rah_to_yah : ℚ
  yah_to_zah : ℚ

/-- Given conversion rates, calculate the number of bahs equivalent to a given number of zahs -/
def zahs_to_bahs (rates : CurrencyRates) (zahs : ℚ) : ℚ :=
  zahs / rates.yah_to_zah / rates.rah_to_yah / rates.bah_to_rah

/-- Theorem stating the equivalence between 1500 zahs and 400/3 bahs -/
theorem zahs_to_bahs_conversion (rates : CurrencyRates) 
  (h1 : rates.bah_to_rah = 3)
  (h2 : rates.rah_to_yah = 3/2)
  (h3 : rates.yah_to_zah = 5/2) : 
  zahs_to_bahs rates 1500 = 400/3 := by
  sorry

#eval zahs_to_bahs ⟨3, 3/2, 5/2⟩ 1500

end NUMINAMATH_CALUDE_zahs_to_bahs_conversion_l937_93787


namespace NUMINAMATH_CALUDE_ratio_problem_l937_93708

theorem ratio_problem (x y : ℚ) (h : (3*x - 2*y) / (2*x + y) = 5/4) : x / y = 13/2 := by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l937_93708


namespace NUMINAMATH_CALUDE_smallest_value_of_y_l937_93761

theorem smallest_value_of_y (x : ℝ) : 
  (17 - x) * (19 - x) * (19 + x) * (17 + x) ≥ -1296 ∧ 
  ∃ x : ℝ, (17 - x) * (19 - x) * (19 + x) * (17 + x) = -1296 :=
by sorry

end NUMINAMATH_CALUDE_smallest_value_of_y_l937_93761


namespace NUMINAMATH_CALUDE_cubic_equation_roots_l937_93748

theorem cubic_equation_roots (a b : ℝ) : 
  (Complex.I : ℂ).re = 0 ∧ (Complex.I : ℂ).im = 1 →
  (1 - 2 * Complex.I : ℂ) ^ 3 + a * (1 - 2 * Complex.I : ℂ) ^ 2 - (1 - 2 * Complex.I : ℂ) + b = 0 →
  a = 1 ∧ b = 15 := by
sorry

end NUMINAMATH_CALUDE_cubic_equation_roots_l937_93748


namespace NUMINAMATH_CALUDE_tony_curl_weight_l937_93717

/-- The weight Tony can lift in the curl exercise, in pounds -/
def curl_weight : ℝ := sorry

/-- The weight Tony can lift in the military press exercise, in pounds -/
def military_press_weight : ℝ := sorry

/-- The weight Tony can lift in the squat exercise, in pounds -/
def squat_weight : ℝ := sorry

/-- The relationship between curl weight and military press weight -/
axiom military_press_relation : military_press_weight = 2 * curl_weight

/-- The relationship between squat weight and military press weight -/
axiom squat_relation : squat_weight = 5 * military_press_weight

/-- The known weight Tony can lift in the squat exercise -/
axiom squat_known_weight : squat_weight = 900

theorem tony_curl_weight : curl_weight = 90 := by sorry

end NUMINAMATH_CALUDE_tony_curl_weight_l937_93717


namespace NUMINAMATH_CALUDE_plus_signs_count_l937_93756

theorem plus_signs_count (total : ℕ) (plus_count : ℕ) (minus_count : ℕ) :
  total = 23 →
  plus_count + minus_count = total →
  (∀ (subset : Finset ℕ), subset.card = 10 → (∃ (i : ℕ), i ∈ subset ∧ i < plus_count)) →
  (∀ (subset : Finset ℕ), subset.card = 15 → (∃ (i : ℕ), i ∈ subset ∧ plus_count ≤ i ∧ i < total)) →
  plus_count = 14 :=
by sorry

end NUMINAMATH_CALUDE_plus_signs_count_l937_93756


namespace NUMINAMATH_CALUDE_exam_average_marks_l937_93760

theorem exam_average_marks (num_papers : ℕ) (geography_increase : ℕ) (history_increase : ℕ) (new_average : ℕ) :
  num_papers = 11 →
  geography_increase = 20 →
  history_increase = 2 →
  new_average = 65 →
  (num_papers * new_average - geography_increase - history_increase) / num_papers = 63 :=
by sorry

end NUMINAMATH_CALUDE_exam_average_marks_l937_93760


namespace NUMINAMATH_CALUDE_subtract_abs_from_local_value_l937_93778

def local_value (n : ℕ) (d : ℕ) : ℕ :=
  let digits := n.digits 10
  let index := digits.findIndex (· = d)
  10 ^ (digits.length - index - 1) * d

def absolute_value (n : ℤ) : ℕ := n.natAbs

theorem subtract_abs_from_local_value :
  local_value 564823 4 - absolute_value 4 = 39996 := by
  sorry

end NUMINAMATH_CALUDE_subtract_abs_from_local_value_l937_93778


namespace NUMINAMATH_CALUDE_binomial_max_term_max_term_sqrt_seven_l937_93753

theorem binomial_max_term (n : ℕ) (x : ℝ) (h : x > 0) :
  ∃ k : ℕ, k ≤ n ∧
    ∀ j : ℕ, j ≤ n →
      (n.choose k : ℝ) * x^k ≥ (n.choose j : ℝ) * x^j :=
sorry

theorem max_term_sqrt_seven :
  let n := 205
  let x := Real.sqrt 7
  ∃ k : ℕ, k ≤ n ∧
    ∀ j : ℕ, j ≤ n →
      (n.choose k : ℝ) * x^k ≥ (n.choose j : ℝ) * x^j ∧
    k = 149 :=
sorry

end NUMINAMATH_CALUDE_binomial_max_term_max_term_sqrt_seven_l937_93753


namespace NUMINAMATH_CALUDE_quadratic_inequality_and_minimum_l937_93737

theorem quadratic_inequality_and_minimum (a : ℝ) : 
  (∀ x : ℝ, a * x^2 - 6 * x + 3 > 0) → 
  (a > 3 ∧ ∃ m : ℝ, m = 7 ∧ ∀ x : ℝ, x > 3 → x + 9 / (x - 1) ≥ m) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_and_minimum_l937_93737


namespace NUMINAMATH_CALUDE_exists_valid_grid_l937_93769

/-- Represents a 3x3 grid with numbers -/
structure Grid :=
  (top_left top_right bottom_left bottom_right : ℕ)

/-- The sum of numbers along each side of the grid is 13 -/
def valid_sum (g : Grid) : Prop :=
  g.top_left + 4 + g.top_right = 13 ∧
  g.top_right + 2 + g.bottom_right = 13 ∧
  g.bottom_right + 1 + g.bottom_left = 13 ∧
  g.bottom_left + 3 + g.top_left = 13

/-- There exists a valid grid arrangement -/
theorem exists_valid_grid : ∃ (g : Grid), valid_sum g :=
sorry

end NUMINAMATH_CALUDE_exists_valid_grid_l937_93769


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l937_93731

theorem quadratic_equation_roots (b c : ℝ) : 
  (∀ x, x^2 - b*x + c = 0 ↔ x = 1 ∨ x = -2) → 
  b = -1 ∧ c = -2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l937_93731


namespace NUMINAMATH_CALUDE_sin_cos_seven_eighths_pi_l937_93724

theorem sin_cos_seven_eighths_pi : 
  Real.sin (7 * π / 8) * Real.cos (7 * π / 8) = - (Real.sqrt 2) / 4 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_seven_eighths_pi_l937_93724


namespace NUMINAMATH_CALUDE_tractor_financing_term_l937_93798

/-- Calculates the financing term in years given the monthly payment and total financed amount. -/
def financing_term_years (monthly_payment : ℚ) (total_amount : ℚ) : ℚ :=
  (total_amount / monthly_payment) / 12

/-- Theorem stating that the financing term for the given conditions is 5 years. -/
theorem tractor_financing_term :
  let monthly_payment : ℚ := 150
  let total_amount : ℚ := 9000
  financing_term_years monthly_payment total_amount = 5 := by
  sorry

end NUMINAMATH_CALUDE_tractor_financing_term_l937_93798


namespace NUMINAMATH_CALUDE_cheryl_craft_project_cheryl_material_ratio_l937_93768

/-- The total amount of material used in Cheryl's craft project --/
def total_used (bought_A bought_B bought_C leftover_A leftover_B leftover_C : ℚ) : ℚ :=
  (bought_A - leftover_A) + (bought_B - leftover_B) + (bought_C - leftover_C)

/-- Theorem stating the total amount of material used in Cheryl's craft project --/
theorem cheryl_craft_project :
  let bought_A : ℚ := 5/8
  let bought_B : ℚ := 2/9
  let bought_C : ℚ := 2/5
  let leftover_A : ℚ := 1/12
  let leftover_B : ℚ := 5/36
  let leftover_C : ℚ := 1/10
  total_used bought_A bought_B bought_C leftover_A leftover_B leftover_C = 37/40 :=
by
  sorry

/-- The ratio of materials used in Cheryl's craft project --/
def material_ratio (used_A used_B used_C : ℚ) : Prop :=
  2 * used_B = used_A ∧ 3 * used_B = used_C

/-- Theorem stating the ratio of materials used in Cheryl's craft project --/
theorem cheryl_material_ratio :
  let bought_A : ℚ := 5/8
  let bought_B : ℚ := 2/9
  let bought_C : ℚ := 2/5
  let leftover_A : ℚ := 1/12
  let leftover_B : ℚ := 5/36
  let leftover_C : ℚ := 1/10
  let used_A : ℚ := bought_A - leftover_A
  let used_B : ℚ := bought_B - leftover_B
  let used_C : ℚ := bought_C - leftover_C
  material_ratio used_A used_B used_C :=
by
  sorry

end NUMINAMATH_CALUDE_cheryl_craft_project_cheryl_material_ratio_l937_93768


namespace NUMINAMATH_CALUDE_matrix_equation_l937_93793

-- Define the matrices
def A : Matrix (Fin 2) (Fin 2) ℚ := !![2, -5; 4, -3]
def B : Matrix (Fin 2) (Fin 2) ℚ := !![-20, -8; 12, 5]
def N : Matrix (Fin 2) (Fin 2) ℚ := !![46/7, -58/7; -28/7, 35/7]

-- State the theorem
theorem matrix_equation : N * A = B := by sorry

end NUMINAMATH_CALUDE_matrix_equation_l937_93793


namespace NUMINAMATH_CALUDE_cos_arcsin_half_l937_93752

theorem cos_arcsin_half : Real.cos (Real.arcsin (1/2)) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_arcsin_half_l937_93752


namespace NUMINAMATH_CALUDE_water_volume_calculation_l937_93740

/-- The volume of water in a container can be calculated by multiplying the number of small hemisphere containers required to hold the water by the volume of each small hemisphere container. -/
theorem water_volume_calculation (num_containers : ℕ) (hemisphere_volume : ℝ) (total_volume : ℝ) : 
  num_containers = 2735 →
  hemisphere_volume = 4 →
  total_volume = num_containers * hemisphere_volume →
  total_volume = 10940 := by
sorry

end NUMINAMATH_CALUDE_water_volume_calculation_l937_93740


namespace NUMINAMATH_CALUDE_probability_at_least_one_karnataka_l937_93745

theorem probability_at_least_one_karnataka (total_students : ℕ) 
  (maharashtra_students : ℕ) (karnataka_students : ℕ) (goa_students : ℕ) 
  (students_to_select : ℕ) : 
  total_students = 10 →
  maharashtra_students = 4 →
  karnataka_students = 3 →
  goa_students = 3 →
  students_to_select = 4 →
  (1 : ℚ) - (Nat.choose (total_students - karnataka_students) students_to_select : ℚ) / 
    (Nat.choose total_students students_to_select : ℚ) = 5 / 6 :=
by sorry

end NUMINAMATH_CALUDE_probability_at_least_one_karnataka_l937_93745


namespace NUMINAMATH_CALUDE_school_total_is_125_l937_93777

/-- Represents the number of students in a school with specific age distribution. -/
structure School where
  /-- The number of students who are 8 years old -/
  eight_years : ℕ
  /-- The proportion of students below 8 years old -/
  below_eight_percent : ℚ
  /-- The ratio of students above 8 years old to students who are 8 years old -/
  above_eight_ratio : ℚ

/-- Calculates the total number of students in the school -/
def total_students (s : School) : ℕ :=
  sorry

/-- Theorem stating that for a school with given age distribution, 
    the total number of students is 125 -/
theorem school_total_is_125 (s : School) 
  (h1 : s.eight_years = 60)
  (h2 : s.below_eight_percent = 1/5)
  (h3 : s.above_eight_ratio = 2/3) : 
  total_students s = 125 := by
  sorry

end NUMINAMATH_CALUDE_school_total_is_125_l937_93777


namespace NUMINAMATH_CALUDE_line_circle_intersection_l937_93794

/-- If a line mx + ny = 0 intersects the circle (x+3)² + (y+1)² = 1 with a chord length of 2, then m/n = -1/3 -/
theorem line_circle_intersection (m n : ℝ) (h : m ≠ 0 ∧ n ≠ 0) :
  (∀ x y : ℝ, m * x + n * y = 0 →
    ((x + 3)^2 + (y + 1)^2 = 1 →
      ∃ x₁ y₁ x₂ y₂ : ℝ,
        m * x₁ + n * y₁ = 0 ∧
        (x₁ + 3)^2 + (y₁ + 1)^2 = 1 ∧
        m * x₂ + n * y₂ = 0 ∧
        (x₂ + 3)^2 + (y₂ + 1)^2 = 1 ∧
        (x₁ - x₂)^2 + (y₁ - y₂)^2 = 4)) →
  m / n = -1/3 :=
by sorry

end NUMINAMATH_CALUDE_line_circle_intersection_l937_93794


namespace NUMINAMATH_CALUDE_total_pizzas_ordered_l937_93704

/-- Represents the number of slices in different pizza sizes -/
structure PizzaSlices where
  small : Nat
  medium : Nat
  large : Nat

/-- Represents the number of pizzas ordered for each size -/
structure PizzaOrder where
  small : Nat
  medium : Nat
  large : Nat

/-- Calculates the total number of slices from a given order -/
def totalSlices (slices : PizzaSlices) (order : PizzaOrder) : Nat :=
  slices.small * order.small + slices.medium * order.medium + slices.large * order.large

/-- The main theorem to prove -/
theorem total_pizzas_ordered
  (slices : PizzaSlices)
  (order : PizzaOrder)
  (h1 : slices.small = 6)
  (h2 : slices.medium = 8)
  (h3 : slices.large = 12)
  (h4 : order.small = 4)
  (h5 : order.medium = 5)
  (h6 : totalSlices slices order = 136) :
  order.small + order.medium + order.large = 15 := by
  sorry


end NUMINAMATH_CALUDE_total_pizzas_ordered_l937_93704


namespace NUMINAMATH_CALUDE_seventeen_above_zero_l937_93713

/-- Represents temperature in degrees Celsius -/
structure Temperature where
  value : ℝ
  unit : String
  is_celsius : unit = "°C"

/-- The zero point of the Celsius scale -/
def celsius_zero : Temperature := ⟨10, "°C", rfl⟩

/-- The temperature to be compared -/
def temp_to_compare : Temperature := ⟨17, "°C", rfl⟩

/-- Theorem stating that 17°C represents a temperature above zero degrees Celsius -/
theorem seventeen_above_zero :
  temp_to_compare.value > celsius_zero.value → 
  ∃ (t : ℝ), t > 0 ∧ temp_to_compare.value = t :=
by sorry

end NUMINAMATH_CALUDE_seventeen_above_zero_l937_93713


namespace NUMINAMATH_CALUDE_arithmetic_sequence_fifth_term_l937_93774

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_fifth_term
  (a : ℕ → ℝ)
  (h_arith : ArithmeticSequence a)
  (h_sum : a 1 + a 9 = 10) :
  a 5 = 5 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_fifth_term_l937_93774


namespace NUMINAMATH_CALUDE_total_balls_purchased_l937_93757

/-- The number of golf balls in a dozen -/
def balls_per_dozen : ℕ := 12

/-- The number of dozens Dan buys -/
def dan_dozens : ℕ := 5

/-- The number of dozens Gus buys -/
def gus_dozens : ℕ := 2

/-- The number of golf balls Chris buys -/
def chris_balls : ℕ := 48

/-- The total number of golf balls purchased -/
def total_balls : ℕ := dan_dozens * balls_per_dozen + gus_dozens * balls_per_dozen + chris_balls

theorem total_balls_purchased :
  total_balls = 132 := by sorry

end NUMINAMATH_CALUDE_total_balls_purchased_l937_93757


namespace NUMINAMATH_CALUDE_minimum_value_implies_ratio_l937_93736

/-- The function f(x) = x³ + ax² + bx - a² - 7a -/
def f (a b x : ℝ) : ℝ := x^3 + a*x^2 + b*x - a^2 - 7*a

/-- The derivative of f(x) -/
def f_derivative (a b x : ℝ) : ℝ := 3*x^2 + 2*a*x + b

theorem minimum_value_implies_ratio (a b : ℝ) :
  (∀ x, f a b x ≥ 10) ∧  -- f(x) has a minimum value of 10
  (f a b 1 = 10) ∧  -- The minimum occurs at x = 1
  (f_derivative a b 1 = 0)  -- The derivative is zero at x = 1
  → b / a = -1 / 2 := by sorry

end NUMINAMATH_CALUDE_minimum_value_implies_ratio_l937_93736


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l937_93726

/-- The eccentricity of a hyperbola given specific conditions -/
theorem hyperbola_eccentricity (a b c : ℝ) (ha : a > 0) (hb : b > 0) : 
  (∀ x y, x^2 / a^2 - y^2 / b^2 = 1) →  -- Hyperbola equation
  (∀ x y, (x - c)^2 + y^2 = 4 * a^2) →  -- Circle equation
  (c^2 = a^2 * (1 + b^2 / a^2)) →  -- Semi-latus rectum condition
  (∃ x y, (b * x + a * y = 0) ∧ (x - c)^2 + y^2 = 4 * a^2 ∧ 
    ∃ x' y', (b * x' + a * y' = 0) ∧ (x' - c)^2 + y'^2 = 4 * a^2 ∧
    (x - x')^2 + (y - y')^2 = 4 * b^2) →  -- Asymptote intercepted by circle
  (c^2 / a^2 - 1)^(1/2) = Real.sqrt 3 :=  -- Eccentricity equals sqrt(3)
by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l937_93726


namespace NUMINAMATH_CALUDE_walking_speed_l937_93749

theorem walking_speed (x : ℝ) : 
  let tom_speed := x^2 - 14*x - 48
  let jerry_distance := x^2 - 5*x - 84
  let jerry_time := x + 8
  let jerry_speed := jerry_distance / jerry_time
  x ≠ -8 → tom_speed = jerry_speed → tom_speed = 6 :=
by sorry

end NUMINAMATH_CALUDE_walking_speed_l937_93749


namespace NUMINAMATH_CALUDE_sqrt_3_minus_3_power_0_minus_2_power_neg_1_l937_93744

theorem sqrt_3_minus_3_power_0_minus_2_power_neg_1 :
  (Real.sqrt 3 - 3) ^ 0 - 2 ^ (-1 : ℤ) = 1/2 := by sorry

end NUMINAMATH_CALUDE_sqrt_3_minus_3_power_0_minus_2_power_neg_1_l937_93744


namespace NUMINAMATH_CALUDE_simple_interest_calculation_l937_93732

/-- Given a principal amount with 5% interest rate for 2 years, 
    if the compound interest is 51.25, then the simple interest is 50 -/
theorem simple_interest_calculation (P : ℝ) : 
  P * ((1 + 0.05)^2 - 1) = 51.25 → P * 0.05 * 2 = 50 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_calculation_l937_93732


namespace NUMINAMATH_CALUDE_max_diagonal_bd_l937_93762

/-- Represents the side lengths of a quadrilateral --/
structure QuadrilateralSides where
  AB : ℕ
  BC : ℕ
  CD : ℕ
  DA : ℕ

/-- Checks if the given side lengths form a valid cyclic quadrilateral --/
def is_valid_cyclic_quadrilateral (sides : QuadrilateralSides) : Prop :=
  sides.AB < 10 ∧ sides.BC < 10 ∧ sides.CD < 10 ∧ sides.DA < 10 ∧
  sides.AB ≠ sides.BC ∧ sides.AB ≠ sides.CD ∧ sides.AB ≠ sides.DA ∧
  sides.BC ≠ sides.CD ∧ sides.BC ≠ sides.DA ∧ sides.CD ≠ sides.DA ∧
  sides.BC + sides.CD = sides.AB + sides.DA

/-- Calculates the square of the diagonal BD --/
def diagonal_bd_squared (sides : QuadrilateralSides) : ℚ :=
  (sides.AB^2 + sides.BC^2 + sides.CD^2 + sides.DA^2) / 2

theorem max_diagonal_bd (sides : QuadrilateralSides) :
  is_valid_cyclic_quadrilateral sides →
  diagonal_bd_squared sides ≤ 191/2 :=
sorry

end NUMINAMATH_CALUDE_max_diagonal_bd_l937_93762


namespace NUMINAMATH_CALUDE_cylinder_surface_area_l937_93712

/-- The surface area of a cylinder with a square cross-section of side length 2 is 6π. -/
theorem cylinder_surface_area (π : ℝ) (h : π = Real.pi) : 
  let side_length : ℝ := 2
  let radius : ℝ := side_length / 2
  let height : ℝ := side_length
  let lateral_area : ℝ := 2 * π * radius * height
  let base_area : ℝ := 2 * π * radius^2
  lateral_area + base_area = 6 * π :=
by
  sorry

end NUMINAMATH_CALUDE_cylinder_surface_area_l937_93712


namespace NUMINAMATH_CALUDE_cookies_eaten_total_l937_93751

theorem cookies_eaten_total (charlie_cookies father_cookies mother_cookies : ℕ) 
  (h1 : charlie_cookies = 15)
  (h2 : father_cookies = 10)
  (h3 : mother_cookies = 5) :
  charlie_cookies + father_cookies + mother_cookies = 30 := by
  sorry

end NUMINAMATH_CALUDE_cookies_eaten_total_l937_93751


namespace NUMINAMATH_CALUDE_diagonal_increase_l937_93791

/-- The number of diagonals in a convex polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := sorry

/-- Theorem stating the relationship between the number of diagonals
    in a convex polygon with n sides and n+1 sides -/
theorem diagonal_increase (n : ℕ) :
  num_diagonals (n + 1) = num_diagonals n + n - 1 := by sorry

end NUMINAMATH_CALUDE_diagonal_increase_l937_93791


namespace NUMINAMATH_CALUDE_sum_of_repeating_decimals_l937_93722

/-- The repeating decimal 0.overline{6} --/
def repeating_six : ℚ := 2/3

/-- The repeating decimal 0.overline{3} --/
def repeating_three : ℚ := 1/3

/-- The sum of 0.overline{6} and 0.overline{3} is equal to 1 --/
theorem sum_of_repeating_decimals : repeating_six + repeating_three = 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_repeating_decimals_l937_93722


namespace NUMINAMATH_CALUDE_position_2018_in_spiral_100_l937_93767

/-- Represents a position in the matrix -/
structure Position where
  i : Nat
  j : Nat

/-- Constructs a spiral matrix of size n x n -/
def spiralMatrix (n : Nat) : Matrix (Fin n) (Fin n) Nat :=
  sorry

/-- Returns the position of a given number in the spiral matrix -/
def findPosition (n : Nat) (num : Nat) : Position :=
  sorry

/-- Theorem stating that 2018 is at position (34, 95) in a 100x100 spiral matrix -/
theorem position_2018_in_spiral_100 :
  findPosition 100 2018 = Position.mk 34 95 := by
  sorry

end NUMINAMATH_CALUDE_position_2018_in_spiral_100_l937_93767


namespace NUMINAMATH_CALUDE_largest_divisor_of_n4_minus_n2_l937_93755

theorem largest_divisor_of_n4_minus_n2 (n : ℤ) : 
  ∃ (d : ℕ), d > 0 ∧ (∀ (m : ℤ), (m^4 - m^2) % d = 0) ∧ 
  (∀ (k : ℕ), k > d → ∃ (l : ℤ), (l^4 - l^2) % k ≠ 0) ∧ d = 6 :=
sorry

end NUMINAMATH_CALUDE_largest_divisor_of_n4_minus_n2_l937_93755


namespace NUMINAMATH_CALUDE_general_admission_tickets_l937_93720

theorem general_admission_tickets (student_price general_price total_tickets total_money : ℕ) 
  (h1 : student_price = 4)
  (h2 : general_price = 6)
  (h3 : total_tickets = 525)
  (h4 : total_money = 2876) :
  ∃ (student_tickets general_tickets : ℕ),
    student_tickets + general_tickets = total_tickets ∧
    student_tickets * student_price + general_tickets * general_price = total_money ∧
    general_tickets = 388 := by
  sorry

end NUMINAMATH_CALUDE_general_admission_tickets_l937_93720


namespace NUMINAMATH_CALUDE_smallest_sum_in_S_l937_93728

def S : Set ℚ := {2, 0, -1, -3}

theorem smallest_sum_in_S : 
  ∃ (x y : ℚ), x ∈ S ∧ y ∈ S ∧ x ≠ y ∧ 
  (∀ (a b : ℚ), a ∈ S → b ∈ S → a ≠ b → x + y ≤ a + b) ∧
  x + y = -4 :=
sorry

end NUMINAMATH_CALUDE_smallest_sum_in_S_l937_93728


namespace NUMINAMATH_CALUDE_oliver_spending_l937_93784

theorem oliver_spending (initial_amount spent_amount received_amount final_amount : ℕ) :
  initial_amount = 33 →
  received_amount = 32 →
  final_amount = 61 →
  final_amount = initial_amount - spent_amount + received_amount →
  spent_amount = 4 := by
sorry

end NUMINAMATH_CALUDE_oliver_spending_l937_93784
