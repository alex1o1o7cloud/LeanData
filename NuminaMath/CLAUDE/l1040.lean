import Mathlib

namespace NUMINAMATH_CALUDE_plan_y_more_cost_effective_l1040_104084

/-- The cost in cents for Plan X given m megabytes -/
def cost_x (m : ℕ) : ℕ := 5 * m

/-- The cost in cents for Plan Y given m megabytes -/
def cost_y (m : ℕ) : ℕ := 3000 + 3 * m

/-- The minimum whole number of megabytes for Plan Y to be more cost-effective -/
def min_megabytes : ℕ := 1501

theorem plan_y_more_cost_effective :
  ∀ m : ℕ, m ≥ min_megabytes → cost_y m < cost_x m ∧
  ∀ n : ℕ, n < min_megabytes → cost_y n ≥ cost_x n :=
by sorry

end NUMINAMATH_CALUDE_plan_y_more_cost_effective_l1040_104084


namespace NUMINAMATH_CALUDE_geckos_sold_last_year_l1040_104090

theorem geckos_sold_last_year (x : ℕ) : 
  x + 2 * x = 258 → x = 86 := by
  sorry

end NUMINAMATH_CALUDE_geckos_sold_last_year_l1040_104090


namespace NUMINAMATH_CALUDE_complement_of_40_degree_angle_l1040_104064

/-- Given an angle A of 40 degrees, its complement is 50 degrees. -/
theorem complement_of_40_degree_angle (A : ℝ) : 
  A = 40 → (90 - A) = 50 := by
  sorry

end NUMINAMATH_CALUDE_complement_of_40_degree_angle_l1040_104064


namespace NUMINAMATH_CALUDE_cosine_value_problem_l1040_104019

theorem cosine_value_problem (α : Real) 
  (h1 : 0 < α) (h2 : α < Real.pi / 6)
  (h3 : Real.sin α ^ 6 + Real.cos α ^ 6 = 7 / 12) : 
  1998 * Real.cos α = 333 * Real.sqrt 30 := by
  sorry

end NUMINAMATH_CALUDE_cosine_value_problem_l1040_104019


namespace NUMINAMATH_CALUDE_correct_quotient_l1040_104062

theorem correct_quotient (D : ℕ) : 
  D % 21 = 0 →  -- The remainder is 0 when divided by 21
  D / 12 = 35 →  -- Dividing by 12 yields a quotient of 35
  D / 21 = 20  -- The correct quotient when dividing by 21 is 20
:= by sorry

end NUMINAMATH_CALUDE_correct_quotient_l1040_104062


namespace NUMINAMATH_CALUDE_number_pair_theorem_l1040_104020

theorem number_pair_theorem (S P : ℝ) (x y : ℝ) 
  (h1 : x + y = S) (h2 : x * y = P) :
  ((x = (S + Real.sqrt (S^2 - 4*P)) / 2 ∧ y = (S - Real.sqrt (S^2 - 4*P)) / 2) ∨
   (x = (S - Real.sqrt (S^2 - 4*P)) / 2 ∧ y = (S + Real.sqrt (S^2 - 4*P)) / 2)) :=
by sorry

end NUMINAMATH_CALUDE_number_pair_theorem_l1040_104020


namespace NUMINAMATH_CALUDE_total_volume_of_cubes_combined_cube_volume_l1040_104063

theorem total_volume_of_cubes : ℕ → ℕ → ℕ → ℕ → ℕ
  | carl_count, carl_side, kate_count, kate_side =>
    (carl_count * carl_side^3) + (kate_count * kate_side^3)

theorem combined_cube_volume : total_volume_of_cubes 8 2 3 3 = 145 := by
  sorry

end NUMINAMATH_CALUDE_total_volume_of_cubes_combined_cube_volume_l1040_104063


namespace NUMINAMATH_CALUDE_last_four_digits_of_5_pow_2013_l1040_104016

def last_four_digits (n : ℕ) : ℕ := n % 10000

def cycle_pattern : List ℕ := [3125, 5625, 8125, 0625]

theorem last_four_digits_of_5_pow_2013 :
  last_four_digits (5^2013) = 3125 := by
  sorry

end NUMINAMATH_CALUDE_last_four_digits_of_5_pow_2013_l1040_104016


namespace NUMINAMATH_CALUDE_fraction_sum_inequality_l1040_104056

theorem fraction_sum_inequality (a b c d n : ℕ) 
  (h1 : a + c < n) 
  (h2 : (a : ℚ) / b + (c : ℚ) / d < 1) : 
  (a : ℚ) / b + (c : ℚ) / d < 1 - 1 / (n^3 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_inequality_l1040_104056


namespace NUMINAMATH_CALUDE_sin_15_cos_15_eq_quarter_l1040_104010

theorem sin_15_cos_15_eq_quarter : Real.sin (15 * π / 180) * Real.cos (15 * π / 180) = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_sin_15_cos_15_eq_quarter_l1040_104010


namespace NUMINAMATH_CALUDE_Z_in_first_quadrant_l1040_104075

def Z : ℂ := (5 + 4*Complex.I) + (-1 + 2*Complex.I)

theorem Z_in_first_quadrant : 
  Z.re > 0 ∧ Z.im > 0 := by sorry

end NUMINAMATH_CALUDE_Z_in_first_quadrant_l1040_104075


namespace NUMINAMATH_CALUDE_product_of_numbers_l1040_104012

theorem product_of_numbers (x y : ℝ) : 
  x + y = 25 → x - y = 7 → x * y = 144 := by sorry

end NUMINAMATH_CALUDE_product_of_numbers_l1040_104012


namespace NUMINAMATH_CALUDE_square_difference_l1040_104034

theorem square_difference (x y : ℝ) (h1 : (x + y)^2 = 49) (h2 : x * y = 8) : 
  (x - y)^2 = 17 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_l1040_104034


namespace NUMINAMATH_CALUDE_point_not_on_graph_l1040_104096

/-- A linear function y = (k+1)x + 3 where k > -1 -/
def linear_function (k : ℝ) (x : ℝ) : ℝ := (k + 1) * x + 3

/-- The constraint on k -/
def k_constraint (k : ℝ) : Prop := k > -1

theorem point_not_on_graph (k : ℝ) (h : k_constraint k) :
  ¬ (linear_function k 5 = -1) := by
  sorry

end NUMINAMATH_CALUDE_point_not_on_graph_l1040_104096


namespace NUMINAMATH_CALUDE_smallest_delicious_integer_l1040_104069

/-- A delicious integer is an integer A for which there exist consecutive integers starting from A that sum to 2024. -/
def IsDelicious (A : ℤ) : Prop :=
  ∃ n : ℕ+, (n : ℤ) * (2 * A + n - 1) / 2 = 2024

/-- The smallest delicious integer is -2023. -/
theorem smallest_delicious_integer : 
  (IsDelicious (-2023) ∧ ∀ A : ℤ, A < -2023 → ¬IsDelicious A) := by
  sorry

end NUMINAMATH_CALUDE_smallest_delicious_integer_l1040_104069


namespace NUMINAMATH_CALUDE_remaining_money_l1040_104066

def base_8_to_10 (n : ℕ) : ℕ :=
  (n / 1000) * 512 + ((n / 100) % 10) * 64 + ((n / 10) % 10) * 8 + (n % 10)

def savings : ℕ := 5555

def ticket_cost : ℕ := 1200

theorem remaining_money :
  base_8_to_10 savings - ticket_cost = 1725 := by sorry

end NUMINAMATH_CALUDE_remaining_money_l1040_104066


namespace NUMINAMATH_CALUDE_distance_midway_to_new_city_l1040_104025

theorem distance_midway_to_new_city : 
  let new_city : ℂ := 0
  let old_town : ℂ := 3200 * I
  let midway : ℂ := 960 + 1280 * I
  Complex.abs (midway - new_city) = 3200 := by
sorry

end NUMINAMATH_CALUDE_distance_midway_to_new_city_l1040_104025


namespace NUMINAMATH_CALUDE_total_soldiers_l1040_104074

theorem total_soldiers (n : ℕ) 
  (h1 : ∃ x y : ℕ, x + y = n ∧ y = x / 6)
  (h2 : ∃ x' y' : ℕ, x' + y' = n ∧ y' = x' / 7)
  (h3 : ∃ y y' : ℕ, y - y' = 2)
  (h4 : ∀ z : ℕ, z + n = n → z = 0) :
  n = 98 := by
  sorry

end NUMINAMATH_CALUDE_total_soldiers_l1040_104074


namespace NUMINAMATH_CALUDE_debby_bottles_remaining_l1040_104021

/-- Calculates the number of water bottles remaining after a period of consumption. -/
def bottles_remaining (initial : ℕ) (daily_consumption : ℕ) (days : ℕ) : ℕ :=
  initial - daily_consumption * days

/-- Proves that Debby has 99 bottles left after her consumption period. -/
theorem debby_bottles_remaining :
  bottles_remaining 264 15 11 = 99 := by
  sorry

end NUMINAMATH_CALUDE_debby_bottles_remaining_l1040_104021


namespace NUMINAMATH_CALUDE_bike_route_length_l1040_104077

/-- Represents a rectangular bike route in a park -/
structure BikeRoute where
  upper_horizontal : List Float
  left_vertical : List Float

/-- Calculates the total length of the bike route -/
def total_length (route : BikeRoute) : Float :=
  2 * (route.upper_horizontal.sum + route.left_vertical.sum)

/-- Theorem stating the total length of the specific bike route -/
theorem bike_route_length :
  let route : BikeRoute := {
    upper_horizontal := [4, 7, 2],
    left_vertical := [6, 7]
  }
  total_length route = 52 := by sorry

end NUMINAMATH_CALUDE_bike_route_length_l1040_104077


namespace NUMINAMATH_CALUDE_combined_tax_rate_l1040_104037

/-- Calculate the combined tax rate for three individuals given their tax rates and income ratios -/
theorem combined_tax_rate
  (mork_rate : ℝ)
  (mindy_rate : ℝ)
  (orson_rate : ℝ)
  (mindy_income_ratio : ℝ)
  (orson_income_ratio : ℝ)
  (h1 : mork_rate = 0.45)
  (h2 : mindy_rate = 0.15)
  (h3 : orson_rate = 0.25)
  (h4 : mindy_income_ratio = 4)
  (h5 : orson_income_ratio = 2) :
  let total_tax := mork_rate + mindy_rate * mindy_income_ratio + orson_rate * orson_income_ratio
  let total_income := 1 + mindy_income_ratio + orson_income_ratio
  (total_tax / total_income) * 100 = 22.14 := by
  sorry

end NUMINAMATH_CALUDE_combined_tax_rate_l1040_104037


namespace NUMINAMATH_CALUDE_relationship_abc_l1040_104048

theorem relationship_abc : 
  let a := Real.log 2
  let b := 5^(-1/2 : ℝ)
  let c := (1/4 : ℝ) * ∫ x in (0 : ℝ)..(π : ℝ), Real.sin x
  b < c ∧ c < a := by sorry

end NUMINAMATH_CALUDE_relationship_abc_l1040_104048


namespace NUMINAMATH_CALUDE_parabola_standard_equation_l1040_104032

/-- The standard equation of a parabola with directrix x = 1 -/
theorem parabola_standard_equation (x y : ℝ) :
  (∃ (p : ℝ), p > 0 ∧ 1 = p / 2 ∧ x < -p / 2) →
  (∀ point : ℝ × ℝ, point ∈ {(x, y) | y^2 = -4*x} ↔
    dist point (x, 0) = dist point (1, (point.2))) :=
by sorry

end NUMINAMATH_CALUDE_parabola_standard_equation_l1040_104032


namespace NUMINAMATH_CALUDE_min_sales_to_break_even_l1040_104043

-- Define the given conditions
def current_salary : ℕ := 90000
def new_base_salary : ℕ := 45000
def commission_rate : ℚ := 15 / 100
def sale_value : ℕ := 1500

-- Define the function to calculate the total earnings in the new job
def new_job_earnings (num_sales : ℕ) : ℚ :=
  new_base_salary + (num_sales * sale_value * commission_rate)

-- Theorem statement
theorem min_sales_to_break_even :
  ∃ n : ℕ, (∀ m : ℕ, m < n → new_job_earnings m < current_salary) ∧
           new_job_earnings n ≥ current_salary ∧
           n = 200 := by
  sorry


end NUMINAMATH_CALUDE_min_sales_to_break_even_l1040_104043


namespace NUMINAMATH_CALUDE_smallest_number_divisible_by_111_ending_2004_l1040_104080

def is_divisible_by (a b : ℕ) : Prop := ∃ k, a = b * k

def last_four_digits (n : ℕ) : ℕ := n % 10000

theorem smallest_number_divisible_by_111_ending_2004 :
  ∀ X : ℕ, 
    X > 0 ∧ 
    is_divisible_by X 111 ∧ 
    last_four_digits X = 2004 → 
    X ≥ 662004 :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_divisible_by_111_ending_2004_l1040_104080


namespace NUMINAMATH_CALUDE_negative_three_x_squared_times_two_x_l1040_104068

theorem negative_three_x_squared_times_two_x (x : ℝ) : (-3 * x)^2 * (2 * x) = 18 * x^3 := by
  sorry

end NUMINAMATH_CALUDE_negative_three_x_squared_times_two_x_l1040_104068


namespace NUMINAMATH_CALUDE_fraction_denominator_l1040_104094

theorem fraction_denominator (x y : ℝ) (h : x / y = 7 / 3) :
  ∃ z : ℝ, (x + y) / z = 2.5 ∧ z = 4 * y / 3 :=
by sorry

end NUMINAMATH_CALUDE_fraction_denominator_l1040_104094


namespace NUMINAMATH_CALUDE_abs_inequality_solution_set_l1040_104024

theorem abs_inequality_solution_set (x : ℝ) : 
  |x + 3| - |x - 3| > 3 ↔ x > 3/2 := by
sorry

end NUMINAMATH_CALUDE_abs_inequality_solution_set_l1040_104024


namespace NUMINAMATH_CALUDE_range_of_a_l1040_104057

-- Define a decreasing function on (-1, 1)
def IsDecreasingOn (f : ℝ → ℝ) : Prop :=
  ∀ x y, -1 < x ∧ x < y ∧ y < 1 → f x > f y

-- State the theorem
theorem range_of_a (f : ℝ → ℝ) (a : ℝ) 
  (h1 : IsDecreasingOn f)
  (h2 : f (1 - a) < f (2 * a - 1)) :
  0 < a ∧ a < 2/3 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l1040_104057


namespace NUMINAMATH_CALUDE_solution_set_inequality_l1040_104089

theorem solution_set_inequality (x : ℝ) : (2*x + 1) / (x + 1) < 1 ↔ -1 < x ∧ x < 0 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l1040_104089


namespace NUMINAMATH_CALUDE_d_bounds_l1040_104033

/-- The maximum number of black squares on an n × n board where each black square
    has exactly two neighboring black squares. -/
def d (n : ℕ) : ℕ := sorry

/-- Theorem stating the bounds for d(n) -/
theorem d_bounds (n : ℕ) : 
  (2/3 : ℝ) * n^2 - 8 * n ≤ (d n : ℝ) ∧ (d n : ℝ) ≤ (2/3 : ℝ) * n^2 + 4 * n :=
sorry

end NUMINAMATH_CALUDE_d_bounds_l1040_104033


namespace NUMINAMATH_CALUDE_divisibility_of_power_plus_one_l1040_104007

theorem divisibility_of_power_plus_one (n : ℕ) : 
  ∃ k : ℤ, (2 : ℤ) ^ (3 ^ n) + 1 = k * (3 : ℤ) ^ (n + 1) := by
sorry

end NUMINAMATH_CALUDE_divisibility_of_power_plus_one_l1040_104007


namespace NUMINAMATH_CALUDE_total_weight_in_kg_l1040_104083

-- Define the weights in grams
def monosodium_glutamate : ℕ := 80
def salt : ℕ := 500
def laundry_detergent : ℕ := 420

-- Define the conversion factor from grams to kilograms
def grams_per_kg : ℕ := 1000

-- Theorem statement
theorem total_weight_in_kg :
  (monosodium_glutamate + salt + laundry_detergent) / grams_per_kg = 1 := by
  sorry

end NUMINAMATH_CALUDE_total_weight_in_kg_l1040_104083


namespace NUMINAMATH_CALUDE_divides_two_pow_minus_one_l1040_104071

theorem divides_two_pow_minus_one (n : ℕ) : n ≥ 1 → (n ∣ 2^n - 1 ↔ n = 1) := by
  sorry

end NUMINAMATH_CALUDE_divides_two_pow_minus_one_l1040_104071


namespace NUMINAMATH_CALUDE_f_properties_l1040_104081

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 - |x - a| + 1

theorem f_properties :
  (∀ x ∈ Set.Icc 0 2, f 0 x ≤ 3) ∧
  (∃ x ∈ Set.Icc 0 2, f 0 x = 3) ∧
  (∀ x ∈ Set.Icc 0 2, f 0 x ≥ 3/4) ∧
  (∃ x ∈ Set.Icc 0 2, f 0 x = 3/4) ∧
  (∀ a < 0, ∀ x, f a x ≥ 3/4 + a) ∧
  (∀ a < 0, ∃ x, f a x = 3/4 + a) ∧
  (∀ a ≥ 0, ∀ x, f a x ≥ 3/4 - a) ∧
  (∀ a ≥ 0, ∃ x, f a x = 3/4 - a) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l1040_104081


namespace NUMINAMATH_CALUDE_attic_junk_percentage_l1040_104072

theorem attic_junk_percentage :
  ∀ (total useful heirlooms junk : ℕ),
    useful = (20 : ℕ) * total / 100 →
    heirlooms = (10 : ℕ) * total / 100 →
    useful = 8 →
    junk = 28 →
    total = useful + heirlooms + junk →
    (junk : ℚ) / (total : ℚ) = 7 / 10 := by
  sorry

end NUMINAMATH_CALUDE_attic_junk_percentage_l1040_104072


namespace NUMINAMATH_CALUDE_min_dot_product_l1040_104087

theorem min_dot_product (a b : ℝ × ℝ) (h : |3 * (a.1 * b.1 + a.2 * b.2)| ≤ 4) :
  ∃ (c d : ℝ × ℝ), c.1 * d.1 + c.2 * d.2 ≥ a.1 * b.1 + a.2 * b.2 ∧ 
  |3 * (c.1 * d.1 + c.2 * d.2)| ≤ 4 ∧ 
  c.1 * d.1 + c.2 * d.2 = -4/3 := by
sorry

end NUMINAMATH_CALUDE_min_dot_product_l1040_104087


namespace NUMINAMATH_CALUDE_friend_meeting_probability_l1040_104036

/-- The probability that two friends meet given specific conditions -/
theorem friend_meeting_probability : 
  ∀ (wait_time : ℝ) (window : ℝ),
  wait_time > 0 → 
  window > wait_time →
  (∃ (prob : ℝ), 
    prob = (window^2 - 2 * (window - wait_time)^2 / 2) / window^2 ∧ 
    prob = 8/9) := by
  sorry

end NUMINAMATH_CALUDE_friend_meeting_probability_l1040_104036


namespace NUMINAMATH_CALUDE_percentage_difference_l1040_104059

theorem percentage_difference : 
  (68.5 / 100 * 825) - (34.25 / 100 * 1620) = 10.275 := by
  sorry

end NUMINAMATH_CALUDE_percentage_difference_l1040_104059


namespace NUMINAMATH_CALUDE_four_digit_square_completion_l1040_104011

theorem four_digit_square_completion : 
  ∃ n : ℕ, 
    1000 ≤ n ∧ n < 10000 ∧ 
    ∃ k : ℕ, (400 * 10000 + n) = k^2 :=
sorry

end NUMINAMATH_CALUDE_four_digit_square_completion_l1040_104011


namespace NUMINAMATH_CALUDE_polynomial_evaluation_l1040_104049

theorem polynomial_evaluation (x y p q : ℝ) 
  (h1 : x + y = -p) 
  (h2 : x * y = q) : 
  x * (1 + y) - y * (x * y - 1) - x^2 * y = p * q + q - p :=
by sorry

end NUMINAMATH_CALUDE_polynomial_evaluation_l1040_104049


namespace NUMINAMATH_CALUDE_loss_equivalent_pencils_proof_l1040_104045

/-- The number of pencils Patrick purchased -/
def total_pencils : ℕ := 60

/-- The ratio of cost to selling price for 60 pencils -/
def cost_to_sell_ratio : ℚ := 1.3333333333333333

/-- The number of pencils whose selling price equals the loss -/
def loss_equivalent_pencils : ℕ := 20

theorem loss_equivalent_pencils_proof :
  ∃ (selling_price : ℚ) (cost : ℚ),
    cost = cost_to_sell_ratio * selling_price ∧
    loss_equivalent_pencils * (selling_price / total_pencils) = cost - selling_price :=
by sorry

end NUMINAMATH_CALUDE_loss_equivalent_pencils_proof_l1040_104045


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l1040_104050

theorem complex_fraction_equality : (1 - 2*I) / (2 + I) = -I := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l1040_104050


namespace NUMINAMATH_CALUDE_daytona_beach_shark_sightings_l1040_104038

theorem daytona_beach_shark_sightings :
  let cape_may_sightings : ℕ := 7
  let daytona_beach_sightings : ℕ := 3 * cape_may_sightings + 5
  daytona_beach_sightings = 26 :=
by sorry

end NUMINAMATH_CALUDE_daytona_beach_shark_sightings_l1040_104038


namespace NUMINAMATH_CALUDE_repeated_root_condition_l1040_104065

/-- The equation has a repeated root if and only if m = -1 -/
theorem repeated_root_condition (m : ℝ) : 
  (∃ x : ℝ, (x - 6) / (x - 5) + 1 = m / (x - 5) ∧ 
   ∀ y : ℝ, y ≠ x → (y - 6) / (y - 5) + 1 ≠ m / (y - 5)) ↔ 
  m = -1 := by
  sorry

end NUMINAMATH_CALUDE_repeated_root_condition_l1040_104065


namespace NUMINAMATH_CALUDE_paperclips_exceed_250_l1040_104035

def paperclips (n : ℕ) : ℕ := 5 * 2^(n - 1)

theorem paperclips_exceed_250 : 
  ∀ k : ℕ, k < 7 → paperclips k ≤ 250 ∧ paperclips 7 > 250 :=
by sorry

end NUMINAMATH_CALUDE_paperclips_exceed_250_l1040_104035


namespace NUMINAMATH_CALUDE_complex_magnitude_l1040_104001

theorem complex_magnitude (z : ℂ) (h : z * Complex.I = 1 + Complex.I) : Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_l1040_104001


namespace NUMINAMATH_CALUDE_cuboid_height_proof_l1040_104073

/-- The surface area of a cuboid given its length, width, and height -/
def cuboidSurfaceArea (l w h : ℝ) : ℝ := 2 * (l * w + l * h + w * h)

/-- Theorem: The height of a cuboid with surface area 700 m², length 12 m, and width 14 m is 7 m -/
theorem cuboid_height_proof (surfaceArea length width : ℝ) 
  (hsa : surfaceArea = 700)
  (hl : length = 12)
  (hw : width = 14) :
  ∃ height : ℝ, cuboidSurfaceArea length width height = surfaceArea ∧ height = 7 := by
  sorry

end NUMINAMATH_CALUDE_cuboid_height_proof_l1040_104073


namespace NUMINAMATH_CALUDE_solution_set_inequality_l1040_104044

theorem solution_set_inequality (x : ℝ) : 
  (1 / x < 1 / 3) ↔ (x < 0 ∨ x > 3) :=
by sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l1040_104044


namespace NUMINAMATH_CALUDE_black_cells_intersection_theorem_l1040_104093

/-- Represents a chessboard -/
structure Chessboard :=
  (size : ℕ)
  (cell_area : ℝ)

/-- Represents the configuration of two overlapping chessboards -/
structure OverlappingChessboards :=
  (board1 : Chessboard)
  (board2 : Chessboard)
  (rotation_angle : ℝ)

/-- Calculates the area of intersection between black cells of two overlapping chessboards -/
def black_cells_intersection_area (boards : OverlappingChessboards) : ℝ :=
  sorry

/-- Theorem stating the area of intersection between black cells of two overlapping 8x8 chessboards rotated by 45 degrees -/
theorem black_cells_intersection_theorem (boards : OverlappingChessboards) :
  boards.board1.size = 8 ∧
  boards.board2.size = 8 ∧
  boards.board1.cell_area = 1 ∧
  boards.board2.cell_area = 1 ∧
  boards.rotation_angle = Real.pi / 4 →
  black_cells_intersection_area boards = 32 * (Real.sqrt 2 - 1) :=
sorry

end NUMINAMATH_CALUDE_black_cells_intersection_theorem_l1040_104093


namespace NUMINAMATH_CALUDE_smallest_angle_solution_l1040_104023

theorem smallest_angle_solution (x : Real) : 
  (∀ y : Real, y > 0 ∧ 8 * Real.sin y * Real.cos y^5 - 8 * Real.sin y^5 * Real.cos y = 1 → x ≤ y) ∧ 
  (x > 0 ∧ 8 * Real.sin x * Real.cos x^5 - 8 * Real.sin x^5 * Real.cos x = 1) →
  x = π / 24 := by
  sorry

end NUMINAMATH_CALUDE_smallest_angle_solution_l1040_104023


namespace NUMINAMATH_CALUDE_complex_modulus_sum_difference_l1040_104006

theorem complex_modulus_sum_difference :
  let z₁ : ℂ := 3 - 5*I
  let z₂ : ℂ := 3 + 5*I
  let z₃ : ℂ := -2 + 6*I
  Complex.abs z₁ + Complex.abs z₂ - Real.sqrt (Complex.abs z₃) = 2 * Real.sqrt 34 - Real.sqrt (2 * Real.sqrt 10) :=
by sorry

end NUMINAMATH_CALUDE_complex_modulus_sum_difference_l1040_104006


namespace NUMINAMATH_CALUDE_father_age_equals_sum_of_brothers_ages_l1040_104051

/-- Represents the current ages of the family members -/
structure FamilyAges where
  ivan : ℕ
  vincent : ℕ
  jakub : ℕ
  father : ℕ

/-- The conditions given in the problem -/
def problem_conditions (ages : FamilyAges) : Prop :=
  ages.vincent = 11 ∧
  ages.jakub = 9 ∧
  ages.ivan = 5 * (ages.jakub / 3) ∧
  ages.father = 3 * ages.ivan

/-- The theorem to be proved -/
theorem father_age_equals_sum_of_brothers_ages (ages : FamilyAges) 
  (h : problem_conditions ages) : 
  ∃ (n : ℕ), n = 5 ∧ 
  ages.father + n = ages.ivan + ages.vincent + ages.jakub + 3 * n :=
sorry

end NUMINAMATH_CALUDE_father_age_equals_sum_of_brothers_ages_l1040_104051


namespace NUMINAMATH_CALUDE_fence_perimeter_is_200_l1040_104092

/-- A rectangular fence with evenly spaced posts -/
structure RectangularFence where
  num_posts : ℕ
  post_width : ℝ
  post_spacing : ℝ
  length_width_ratio : ℝ

/-- Calculate the outer perimeter of a rectangular fence -/
def outer_perimeter (fence : RectangularFence) : ℝ :=
  sorry

/-- Theorem: The outer perimeter of the specified fence is 200 feet -/
theorem fence_perimeter_is_200 :
  let fence : RectangularFence :=
    { num_posts := 36
    , post_width := 0.5  -- 6 inches = 0.5 feet
    , post_spacing := 4
    , length_width_ratio := 2 }
  outer_perimeter fence = 200 :=
by sorry

end NUMINAMATH_CALUDE_fence_perimeter_is_200_l1040_104092


namespace NUMINAMATH_CALUDE_a_is_negative_l1040_104079

theorem a_is_negative (a b : ℤ) (h1 : a ≠ 0) (h2 : ∃ k : ℤ, 3 + a + b^2 = 6*a*k) : a < 0 := by
  sorry

end NUMINAMATH_CALUDE_a_is_negative_l1040_104079


namespace NUMINAMATH_CALUDE_complex_fraction_difference_l1040_104061

theorem complex_fraction_difference (a b : ℝ) : 
  (Complex.I : ℂ) ^ 2 = -1 →
  (3 + Complex.I) / (1 - Complex.I) = a + b * Complex.I →
  a - b = -1 := by sorry

end NUMINAMATH_CALUDE_complex_fraction_difference_l1040_104061


namespace NUMINAMATH_CALUDE_game_result_l1040_104042

def f (n : ℕ) : ℕ :=
  if n % 2 = 0 ∧ n % 3 = 0 then 6
  else if n % 2 = 0 then 2
  else 0

def allie_rolls : List ℕ := [5, 4, 1, 2]
def betty_rolls : List ℕ := [6, 3, 3, 2]

def total_points (rolls : List ℕ) : ℕ :=
  (rolls.map f).sum

theorem game_result : 
  (total_points allie_rolls) * (total_points betty_rolls) = 32 := by
  sorry

end NUMINAMATH_CALUDE_game_result_l1040_104042


namespace NUMINAMATH_CALUDE_bolts_per_box_bolts_per_box_correct_l1040_104014

theorem bolts_per_box (bolt_boxes : ℕ) (nut_boxes : ℕ) (nuts_per_box : ℕ) 
  (bolts_left : ℕ) (nuts_left : ℕ) (used_parts : ℕ) : ℕ :=
  let total_parts := used_parts + bolts_left + nuts_left
  let total_nuts := nut_boxes * nuts_per_box
  let total_bolts := total_parts - total_nuts
  let bolts_per_box := total_bolts / bolt_boxes
  bolts_per_box

#check bolts_per_box 7 3 15 3 6 113 = 11

theorem bolts_per_box_correct : bolts_per_box 7 3 15 3 6 113 = 11 := by
  sorry

end NUMINAMATH_CALUDE_bolts_per_box_bolts_per_box_correct_l1040_104014


namespace NUMINAMATH_CALUDE_rohan_join_time_is_seven_l1040_104058

/-- Represents the investment scenario and profit distribution --/
structure InvestmentScenario where
  suresh_investment : ℕ
  rohan_investment : ℕ
  sudhir_investment : ℕ
  total_profit : ℕ
  rohan_sudhir_diff : ℕ
  total_months : ℕ
  sudhir_join_time : ℕ

/-- Calculates the number of months after which Rohan joined the business --/
def calculate_rohan_join_time (scenario : InvestmentScenario) : ℕ :=
  sorry

/-- Theorem stating that Rohan joined after 7 months --/
theorem rohan_join_time_is_seven (scenario : InvestmentScenario) 
  (h1 : scenario.suresh_investment = 18000)
  (h2 : scenario.rohan_investment = 12000)
  (h3 : scenario.sudhir_investment = 9000)
  (h4 : scenario.total_profit = 3795)
  (h5 : scenario.rohan_sudhir_diff = 345)
  (h6 : scenario.total_months = 12)
  (h7 : scenario.sudhir_join_time = 8) : 
  calculate_rohan_join_time scenario = 7 :=
sorry

end NUMINAMATH_CALUDE_rohan_join_time_is_seven_l1040_104058


namespace NUMINAMATH_CALUDE_clothing_distribution_l1040_104017

/-- Given a total of 39 pieces of clothing, with 19 pieces in the first load
    and the rest split into 5 equal loads, prove that each small load
    contains 4 pieces of clothing. -/
theorem clothing_distribution (total : Nat) (first_load : Nat) (num_small_loads : Nat)
    (h1 : total = 39)
    (h2 : first_load = 19)
    (h3 : num_small_loads = 5) :
    (total - first_load) / num_small_loads = 4 := by
  sorry

end NUMINAMATH_CALUDE_clothing_distribution_l1040_104017


namespace NUMINAMATH_CALUDE_tetrahedron_edge_length_l1040_104028

structure Tetrahedron where
  edges : Finset ℝ
  pq : ℝ
  rs : ℝ

def valid_tetrahedron (t : Tetrahedron) : Prop :=
  t.edges.card = 6 ∧
  t.edges = {9, 15, 22, 28, 34, 39} ∧
  t.pq ∈ t.edges ∧
  t.rs ∈ t.edges ∧
  t.pq = 39

theorem tetrahedron_edge_length (t : Tetrahedron) (h : valid_tetrahedron t) : t.rs = 9 := by
  sorry

end NUMINAMATH_CALUDE_tetrahedron_edge_length_l1040_104028


namespace NUMINAMATH_CALUDE_inequality_proof_l1040_104067

theorem inequality_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x < y) :
  x + Real.sqrt (y^2 + 2) < y + Real.sqrt (x^2 + 2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1040_104067


namespace NUMINAMATH_CALUDE_repeating_decimal_to_fraction_l1040_104070

/-- Proves that the repeating decimal 0.35̄ is equal to 5/14 -/
theorem repeating_decimal_to_fraction : 
  ∀ x : ℚ, (∃ n : ℕ, x = (35 : ℚ) / (100^n - 1)) → x = 5/14 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_to_fraction_l1040_104070


namespace NUMINAMATH_CALUDE_remainder_1998_pow_10_mod_10000_l1040_104027

theorem remainder_1998_pow_10_mod_10000 : 1998^10 % 10000 = 1024 := by
  sorry

end NUMINAMATH_CALUDE_remainder_1998_pow_10_mod_10000_l1040_104027


namespace NUMINAMATH_CALUDE_range_of_a_given_three_integer_solutions_l1040_104082

/-- The inequality (2x-1)^2 < ax^2 has exactly three integer solutions -/
def has_three_integer_solutions (a : ℝ) : Prop :=
  ∃ (x y z : ℤ), x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    (∀ (w : ℤ), (2*w - 1)^2 < a*w^2 ↔ w = x ∨ w = y ∨ w = z)

/-- The theorem stating the range of a given the condition -/
theorem range_of_a_given_three_integer_solutions :
  ∀ a : ℝ, has_three_integer_solutions a ↔ 25/9 < a ∧ a ≤ 49/16 := by sorry

end NUMINAMATH_CALUDE_range_of_a_given_three_integer_solutions_l1040_104082


namespace NUMINAMATH_CALUDE_beautiful_association_number_part1_beautiful_association_number_part2_beautiful_association_number_part3_l1040_104040

-- Define the "beautiful association number"
def beautiful_association_number (x y a : ℚ) : ℚ :=
  |x - a| + |y - a|

-- Part 1
theorem beautiful_association_number_part1 :
  beautiful_association_number (-3) 5 2 = 8 := by sorry

-- Part 2
theorem beautiful_association_number_part2 (x : ℚ) :
  beautiful_association_number x 2 3 = 4 → x = 6 ∨ x = 0 := by sorry

-- Part 3
theorem beautiful_association_number_part3 (x₀ x₁ x₂ x₃ x₄ x₅ : ℚ) :
  beautiful_association_number x₀ x₁ 1 = 1 →
  beautiful_association_number x₁ x₂ 2 = 1 →
  beautiful_association_number x₂ x₃ 3 = 1 →
  beautiful_association_number x₃ x₄ 4 = 1 →
  beautiful_association_number x₄ x₅ 5 = 1 →
  ∃ (min : ℚ), min = 10 ∧ x₁ + x₂ + x₃ + x₄ ≥ min := by sorry

end NUMINAMATH_CALUDE_beautiful_association_number_part1_beautiful_association_number_part2_beautiful_association_number_part3_l1040_104040


namespace NUMINAMATH_CALUDE_fraction_product_simplification_l1040_104015

theorem fraction_product_simplification : 
  (2 : ℚ) / 3 * 3 / 4 * 4 / 5 * 5 / 6 = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_product_simplification_l1040_104015


namespace NUMINAMATH_CALUDE_smallest_integer_l1040_104060

theorem smallest_integer (a b : ℕ) (ha : a = 60) (h_lcm_gcd : Nat.lcm a b / Nat.gcd a b = 24) :
  b ≥ 360 :=
sorry

end NUMINAMATH_CALUDE_smallest_integer_l1040_104060


namespace NUMINAMATH_CALUDE_five_cubic_yards_to_cubic_feet_l1040_104091

/-- Conversion factor from yards to feet -/
def yards_to_feet : ℝ := 3

/-- Volume in cubic feet for 5 cubic yards -/
def volume_cubic_feet : ℝ := 135

/-- Theorem stating that 5 cubic yards is equal to 135 cubic feet -/
theorem five_cubic_yards_to_cubic_feet :
  (5 : ℝ) * yards_to_feet^3 = volume_cubic_feet := by sorry

end NUMINAMATH_CALUDE_five_cubic_yards_to_cubic_feet_l1040_104091


namespace NUMINAMATH_CALUDE_snowman_height_example_l1040_104039

/-- The height of a snowman built from three vertically aligned spheres -/
def snowman_height (r1 r2 r3 : ℝ) : ℝ := 2 * (r1 + r2 + r3)

/-- Theorem: The height of a snowman with spheres of radii 10 cm, 20 cm, and 30 cm is 120 cm -/
theorem snowman_height_example : snowman_height 10 20 30 = 120 := by
  sorry

end NUMINAMATH_CALUDE_snowman_height_example_l1040_104039


namespace NUMINAMATH_CALUDE_square_sum_geq_product_sum_l1040_104076

theorem square_sum_geq_product_sum (a b c : ℝ) : a^2 + b^2 + c^2 ≥ a*b + a*c + b*c := by
  sorry

end NUMINAMATH_CALUDE_square_sum_geq_product_sum_l1040_104076


namespace NUMINAMATH_CALUDE_salary_increase_percentage_l1040_104005

theorem salary_increase_percentage (S : ℝ) (h1 : S + 0.10 * S = 330) : 
  ∃ P : ℝ, S + (P / 100) * S = 348 ∧ P = 16 := by
  sorry

end NUMINAMATH_CALUDE_salary_increase_percentage_l1040_104005


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1040_104085

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

/-- The sum of specific terms in the sequence -/
def SpecificSum (a : ℕ → ℝ) : ℝ :=
  a 2 + a 4 + a 9 + a 11

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  ArithmeticSequence a → SpecificSum a = 32 → a 6 + a 7 = 16 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1040_104085


namespace NUMINAMATH_CALUDE_invalid_triangle_after_transformation_l1040_104052

theorem invalid_triangle_after_transformation (DE DF EF : ℝ) 
  (h_original_valid : DE + DF > EF ∧ DE + EF > DF ∧ DF + EF > DE)
  (h_DE : DE = 8)
  (h_DF : DF = 9)
  (h_EF : EF = 5)
  (DE' DF' EF' : ℝ)
  (h_DE' : DE' = 3 * DE)
  (h_DF' : DF' = 2 * DF)
  (h_EF' : EF' = EF) :
  ¬(DE' + DF' > EF' ∧ DE' + EF' > DF' ∧ DF' + EF' > DE') :=
by sorry

end NUMINAMATH_CALUDE_invalid_triangle_after_transformation_l1040_104052


namespace NUMINAMATH_CALUDE_no_integer_solution_l1040_104095

theorem no_integer_solution :
  ¬ ∃ (x y : ℤ), 23 * x^2 - 92 * y^2 = 3128 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solution_l1040_104095


namespace NUMINAMATH_CALUDE_class_composition_l1040_104088

/-- The number of girls in the class -/
def num_girls : ℕ := 13

/-- The percentage of girls after adding one boy -/
def girls_percentage : ℚ := 52 / 100

/-- The original number of boys in the class -/
def original_boys : ℕ := 11

theorem class_composition :
  (num_girls : ℚ) / ((original_boys : ℚ) + 1 + num_girls) = girls_percentage := by
  sorry

end NUMINAMATH_CALUDE_class_composition_l1040_104088


namespace NUMINAMATH_CALUDE_solution_set_equivalence_l1040_104002

-- Define the function f
noncomputable def f : ℝ → ℝ := sorry

-- Define the solution set of f(x) < 0
def solution_set_f_neg (x : ℝ) : Prop := x < -1 ∨ x > 1/3

-- Define the solution set of f(e^x) > 0
def solution_set_f_exp_pos (x : ℝ) : Prop := x < -Real.log 3

-- Theorem statement
theorem solution_set_equivalence :
  (∀ x, f x < 0 ↔ solution_set_f_neg x) →
  (∀ x, f (Real.exp x) > 0 ↔ solution_set_f_exp_pos x) :=
sorry

end NUMINAMATH_CALUDE_solution_set_equivalence_l1040_104002


namespace NUMINAMATH_CALUDE_gcd_of_sequence_is_three_l1040_104047

def a (n : ℕ) : ℕ := (2*n - 1) * (2*n + 1) * (2*n + 3)

theorem gcd_of_sequence_is_three :
  ∃ d : ℕ, d > 0 ∧ 
  (∀ k : ℕ, k ≥ 1 → k ≤ 2008 → d ∣ a k) ∧
  (∀ m : ℕ, m > 0 → (∀ k : ℕ, k ≥ 1 → k ≤ 2008 → m ∣ a k) → m ≤ d) ∧
  d = 3 := by
sorry

end NUMINAMATH_CALUDE_gcd_of_sequence_is_three_l1040_104047


namespace NUMINAMATH_CALUDE_dot_product_of_vectors_l1040_104018

def vector_a : ℝ × ℝ := (-4, 7)
def vector_b : ℝ × ℝ := (5, 2)

theorem dot_product_of_vectors :
  (vector_a.1 * vector_b.1 + vector_a.2 * vector_b.2) = -6 := by sorry

end NUMINAMATH_CALUDE_dot_product_of_vectors_l1040_104018


namespace NUMINAMATH_CALUDE_units_digit_product_l1040_104008

theorem units_digit_product : (47 * 23 * 89) % 10 = 9 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_product_l1040_104008


namespace NUMINAMATH_CALUDE_complementary_angles_ratio_l1040_104000

theorem complementary_angles_ratio (a b : ℝ) : 
  a + b = 90 →  -- The angles are complementary (sum to 90°)
  a = 4 * b →   -- The ratio of the angles is 4:1
  a = 72 :=     -- The larger angle is 72°
by sorry

end NUMINAMATH_CALUDE_complementary_angles_ratio_l1040_104000


namespace NUMINAMATH_CALUDE_hyperbola_a_plus_h_l1040_104078

/-- Given a hyperbola with the following properties:
  * Asymptotes: y = 3x + 2 and y = -3x + 8
  * Passes through the point (2, 10)
  * Standard form: (y-k)^2/a^2 - (x-h)^2/b^2 = 1
  * a, b > 0
  Prove that a + h = 6√2 + 1 -/
theorem hyperbola_a_plus_h (a b h k : ℝ) : 
  (∀ x y : ℝ, (y = 3*x + 2 ∨ y = -3*x + 8) → 
    ((y - k)^2 / a^2 - (x - h)^2 / b^2 = 1)) →
  ((10 - k)^2 / a^2 - (2 - h)^2 / b^2 = 1) →
  (a > 0 ∧ b > 0) →
  a + h = 6 * Real.sqrt 2 + 1 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_a_plus_h_l1040_104078


namespace NUMINAMATH_CALUDE_find_number_l1040_104031

theorem find_number (A B : ℕ+) : 
  Nat.gcd A B = 14 →
  Nat.lcm A B = 312 →
  B = 182 →
  A = 24 := by
sorry

end NUMINAMATH_CALUDE_find_number_l1040_104031


namespace NUMINAMATH_CALUDE_mollys_age_l1040_104055

/-- Molly's birthday candle problem -/
theorem mollys_age (initial_candles additional_candles : ℕ) 
  (h1 : initial_candles = 14)
  (h2 : additional_candles = 6) :
  initial_candles + additional_candles = 20 := by
  sorry

end NUMINAMATH_CALUDE_mollys_age_l1040_104055


namespace NUMINAMATH_CALUDE_bottle_caps_wrappers_difference_l1040_104054

/-- The number of wrappers Danny found at the park -/
def wrappers_found : ℕ := 46

/-- The number of bottle caps Danny found at the park -/
def bottle_caps_found : ℕ := 50

/-- The number of bottle caps Danny now has in his collection -/
def bottle_caps_in_collection : ℕ := 21

/-- The number of wrappers Danny now has in his collection -/
def wrappers_in_collection : ℕ := 52

/-- Theorem stating the difference between bottle caps and wrappers found at the park -/
theorem bottle_caps_wrappers_difference : 
  bottle_caps_found - wrappers_found = 4 := by
  sorry

end NUMINAMATH_CALUDE_bottle_caps_wrappers_difference_l1040_104054


namespace NUMINAMATH_CALUDE_counterexample_exists_l1040_104098

theorem counterexample_exists : ∃ (a b c d : ℝ), a < b ∧ c < d ∧ a * c ≥ b * d := by
  sorry

end NUMINAMATH_CALUDE_counterexample_exists_l1040_104098


namespace NUMINAMATH_CALUDE_least_seven_digit_binary_l1040_104041

theorem least_seven_digit_binary : ∀ n : ℕ, 
  (n > 0 ∧ n < 64) → (Nat.bits n).length < 7 ∧ 
  (Nat.bits 64).length = 7 :=
by sorry

end NUMINAMATH_CALUDE_least_seven_digit_binary_l1040_104041


namespace NUMINAMATH_CALUDE_train_encounters_l1040_104099

/-- Represents the number of hours in the journey -/
def journey_duration : ℕ := 5

/-- Represents the number of trains already on the route when the journey begins -/
def initial_trains : ℕ := 4

/-- Calculates the number of trains encountered during the journey -/
def trains_encountered (duration : ℕ) (initial : ℕ) : ℕ :=
  initial + duration

theorem train_encounters :
  trains_encountered journey_duration initial_trains = 9 := by
  sorry

end NUMINAMATH_CALUDE_train_encounters_l1040_104099


namespace NUMINAMATH_CALUDE_smallest_x_with_remainders_l1040_104053

theorem smallest_x_with_remainders : ∃ (x : ℕ), 
  (x % 5 = 4) ∧ 
  (x % 6 = 5) ∧ 
  (x % 7 = 6) ∧ 
  (∀ y : ℕ, y > 0 ∧ y < x → ¬(y % 5 = 4 ∧ y % 6 = 5 ∧ y % 7 = 6)) ∧
  x = 209 :=
by sorry

end NUMINAMATH_CALUDE_smallest_x_with_remainders_l1040_104053


namespace NUMINAMATH_CALUDE_sasha_always_wins_l1040_104009

/-- Represents the state of the game board -/
structure GameState where
  digits : List Nat
  deriving Repr

/-- Represents a player's move -/
structure Move where
  appendedDigits : List Nat
  deriving Repr

/-- Checks if a number represented by a list of digits is divisible by 112 -/
def isDivisibleBy112 (digits : List Nat) : Bool :=
  sorry

/-- Generates all possible moves for Sasha (appending one digit) -/
def sashasMoves : List Move :=
  sorry

/-- Generates all possible moves for Andrey (appending two digits) -/
def andreysMoves : List Move :=
  sorry

/-- Applies a move to the current game state -/
def applyMove (state : GameState) (move : Move) : GameState :=
  sorry

/-- Checks if Sasha wins in the current state -/
def sashaWins (state : GameState) : Bool :=
  sorry

/-- Checks if Andrey wins in the current state -/
def andreyWins (state : GameState) : Bool :=
  sorry

/-- Theorem: Sasha can always win the game -/
theorem sasha_always_wins :
  ∀ (state : GameState),
    (state.digits.length < 2018) →
    (∃ (move : Move), move ∈ sashasMoves ∧
      ∀ (andreyMove : Move), andreyMove ∈ andreysMoves →
        ¬(andreyWins (applyMove (applyMove state move) andreyMove))) ∨
    (sashaWins state) :=
  sorry

end NUMINAMATH_CALUDE_sasha_always_wins_l1040_104009


namespace NUMINAMATH_CALUDE_ice_cream_cost_l1040_104003

/-- The cost of ice cream problem -/
theorem ice_cream_cost (ice_cream_quantity : ℕ) (yogurt_quantity : ℕ) (yogurt_cost : ℕ) (price_difference : ℕ) :
  ice_cream_quantity = 20 →
  yogurt_quantity = 2 →
  yogurt_cost = 1 →
  price_difference = 118 →
  ∃ (ice_cream_cost : ℕ), 
    ice_cream_cost * ice_cream_quantity = yogurt_cost * yogurt_quantity + price_difference ∧
    ice_cream_cost = 6 :=
by sorry

end NUMINAMATH_CALUDE_ice_cream_cost_l1040_104003


namespace NUMINAMATH_CALUDE_game_score_l1040_104013

/-- The total points scored by Olaf and his dad in a game -/
def totalPoints (dadPoints : ℕ) (olafMultiplier : ℕ) : ℕ :=
  dadPoints + olafMultiplier * dadPoints

/-- Theorem stating the total points scored by Olaf and his dad -/
theorem game_score : totalPoints 7 3 = 28 := by
  sorry

end NUMINAMATH_CALUDE_game_score_l1040_104013


namespace NUMINAMATH_CALUDE_quadratic_equation_transform_sum_l1040_104030

theorem quadratic_equation_transform_sum (x r s : ℝ) : 
  (16 * x^2 - 64 * x - 144 = 0) →
  ((x + r)^2 = s) →
  (r + s = -7) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_transform_sum_l1040_104030


namespace NUMINAMATH_CALUDE_eggs_to_market_l1040_104029

/-- Represents the number of dozens of eggs collected on each collection day -/
def eggs_collected_per_day : ℕ := 8

/-- Represents the number of collection days per week -/
def collection_days_per_week : ℕ := 2

/-- Represents the number of dozens of eggs delivered to the mall -/
def eggs_to_mall : ℕ := 5

/-- Represents the number of dozens of eggs used for pie -/
def eggs_for_pie : ℕ := 4

/-- Represents the number of dozens of eggs donated to charity -/
def eggs_to_charity : ℕ := 4

/-- Represents the total number of dozens of eggs collected in a week -/
def total_eggs_collected : ℕ := eggs_collected_per_day * collection_days_per_week

/-- Represents the total number of dozens of eggs used or given away -/
def total_eggs_used : ℕ := eggs_to_mall + eggs_for_pie + eggs_to_charity

/-- Proves that the number of dozens of eggs delivered to the market is 3 -/
theorem eggs_to_market : total_eggs_collected - total_eggs_used = 3 := by
  sorry

end NUMINAMATH_CALUDE_eggs_to_market_l1040_104029


namespace NUMINAMATH_CALUDE_family_money_sum_l1040_104026

/-- Given Madeline has $48, her brother has half as much as her, and their sister has twice as much as Madeline, the total amount of money all three of them have together is $168. -/
theorem family_money_sum (madeline_money : ℕ) (brother_money : ℕ) (sister_money : ℕ) 
  (h1 : madeline_money = 48)
  (h2 : brother_money = madeline_money / 2)
  (h3 : sister_money = madeline_money * 2) : 
  madeline_money + brother_money + sister_money = 168 := by
  sorry

end NUMINAMATH_CALUDE_family_money_sum_l1040_104026


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l1040_104004

theorem polynomial_division_remainder (m b : ℤ) : 
  (∃ q : Polynomial ℤ, x^5 - 4*x^4 + 12*x^3 - 14*x^2 + 8*x + 5 = 
    (x^2 - 3*x + m) * q + (2*x + b)) → 
  m = 1 ∧ b = 7 := by
sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l1040_104004


namespace NUMINAMATH_CALUDE_claire_photos_l1040_104046

theorem claire_photos (c : ℕ) : 
  (3 * c = c + 12) → c = 6 := by
  sorry

end NUMINAMATH_CALUDE_claire_photos_l1040_104046


namespace NUMINAMATH_CALUDE_triangle_inequality_l1040_104086

theorem triangle_inequality (a b c : ℝ) (h_nonneg : 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c) (h_perimeter : a + b + c = 1) :
  a^2 + b^2 + c^2 + 4*a*b*c < 1/2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l1040_104086


namespace NUMINAMATH_CALUDE_vacation_cost_l1040_104097

theorem vacation_cost (cost : ℝ) : 
  (cost / 3 - cost / 4 = 30) → cost = 360 := by
sorry

end NUMINAMATH_CALUDE_vacation_cost_l1040_104097


namespace NUMINAMATH_CALUDE_brand_y_pen_price_l1040_104022

/-- Calculates the price of brand Y pens given the total cost, number of pens, 
    price of brand X pens, and number of brand X pens purchased. -/
theorem brand_y_pen_price 
  (total_cost : ℚ) 
  (total_pens : ℕ) 
  (brand_x_price : ℚ) 
  (brand_x_count : ℕ) 
  (h1 : total_cost = 40)
  (h2 : total_pens = 12)
  (h3 : brand_x_price = 4)
  (h4 : brand_x_count = 8) :
  (total_cost - brand_x_price * brand_x_count) / (total_pens - brand_x_count) = 2 := by
  sorry

#check brand_y_pen_price

end NUMINAMATH_CALUDE_brand_y_pen_price_l1040_104022
