import Mathlib

namespace NUMINAMATH_CALUDE_three_person_subcommittees_from_seven_l3770_377069

theorem three_person_subcommittees_from_seven (n : ℕ) (k : ℕ) : n = 7 ∧ k = 3 → 
  Nat.choose n k = 35 := by
  sorry

end NUMINAMATH_CALUDE_three_person_subcommittees_from_seven_l3770_377069


namespace NUMINAMATH_CALUDE_problem_statement_l3770_377092

theorem problem_statement (a b : ℝ) (h : |a - 1| + (b + 2)^2 = 0) : 
  (a + b)^2002 + a^2001 = 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3770_377092


namespace NUMINAMATH_CALUDE_afternoon_sales_l3770_377016

/-- 
Given a salesman who sold pears in the morning and afternoon, 
this theorem proves that if he sold twice as much in the afternoon 
as in the morning, and 420 kilograms in total, then he sold 280 
kilograms in the afternoon.
-/
theorem afternoon_sales 
  (morning_sales : ℕ) 
  (afternoon_sales : ℕ) 
  (h1 : afternoon_sales = 2 * morning_sales) 
  (h2 : morning_sales + afternoon_sales = 420) : 
  afternoon_sales = 280 := by
  sorry

end NUMINAMATH_CALUDE_afternoon_sales_l3770_377016


namespace NUMINAMATH_CALUDE_typing_service_problem_l3770_377030

/-- Represents the typing service problem -/
theorem typing_service_problem 
  (total_pages : ℕ) 
  (pages_revised_twice : ℕ) 
  (total_cost : ℕ) 
  (first_typing_cost : ℕ) 
  (revision_cost : ℕ) 
  (h1 : total_pages = 100)
  (h2 : pages_revised_twice = 30)
  (h3 : total_cost = 1400)
  (h4 : first_typing_cost = 10)
  (h5 : revision_cost = 5) :
  ∃ (pages_revised_once : ℕ),
    pages_revised_once = 20 ∧
    total_cost = 
      first_typing_cost * total_pages + 
      revision_cost * pages_revised_once + 
      2 * revision_cost * pages_revised_twice :=
by sorry

end NUMINAMATH_CALUDE_typing_service_problem_l3770_377030


namespace NUMINAMATH_CALUDE_range_of_k_l3770_377094

theorem range_of_k (x k : ℝ) : 
  (x - 1) / (x - 2) = k / (x - 2) + 2 ∧ x ≥ 0 → k ≤ 3 ∧ k ≠ 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_k_l3770_377094


namespace NUMINAMATH_CALUDE_pair_one_six_least_restricted_l3770_377098

/-- Represents the days of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Represents a license plate ending pair -/
structure LicensePlatePair :=
  (first : Nat)
  (second : Nat)

/-- The restriction schedule for each license plate ending pair -/
def restrictionSchedule : LicensePlatePair → List DayOfWeek
  | ⟨1, 6⟩ => [DayOfWeek.Monday, DayOfWeek.Tuesday]
  | ⟨2, 7⟩ => [DayOfWeek.Tuesday, DayOfWeek.Wednesday]
  | ⟨3, 8⟩ => [DayOfWeek.Wednesday, DayOfWeek.Thursday]
  | ⟨4, 9⟩ => [DayOfWeek.Thursday, DayOfWeek.Friday]
  | ⟨5, 0⟩ => [DayOfWeek.Friday, DayOfWeek.Monday]
  | _ => []

/-- Calculate the number of restricted days for a given license plate pair in January 2014 -/
def restrictedDays (pair : LicensePlatePair) : Nat :=
  sorry

/-- All possible license plate ending pairs -/
def allPairs : List LicensePlatePair :=
  [⟨1, 6⟩, ⟨2, 7⟩, ⟨3, 8⟩, ⟨4, 9⟩, ⟨5, 0⟩]

/-- Theorem: The license plate pair (1,6) has the fewest restricted days in January 2014 -/
theorem pair_one_six_least_restricted :
  ∀ pair ∈ allPairs, restrictedDays ⟨1, 6⟩ ≤ restrictedDays pair := by
  sorry

end NUMINAMATH_CALUDE_pair_one_six_least_restricted_l3770_377098


namespace NUMINAMATH_CALUDE_factorization_equality_l3770_377070

theorem factorization_equality (y : ℝ) : 3 * y * (y - 5) + 4 * (y - 5) = (3 * y + 4) * (y - 5) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l3770_377070


namespace NUMINAMATH_CALUDE_complex_first_quadrant_l3770_377085

theorem complex_first_quadrant (a : ℝ) : 
  (∃ (z : ℂ), z = (1 : ℂ) / (1 + a * Complex.I) ∧ z.re > 0 ∧ z.im > 0) ↔ a < 0 :=
sorry

end NUMINAMATH_CALUDE_complex_first_quadrant_l3770_377085


namespace NUMINAMATH_CALUDE_hyperbola_center_l3770_377066

def hyperbola_equation (x y : ℝ) : Prop :=
  9 * x^2 - 36 * x - 16 * y^2 + 128 * y - 400 = 0

def is_center (h k : ℝ) : Prop :=
  ∀ (x y : ℝ), hyperbola_equation x y → 
    ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ ((x - h)^2 / a^2) - ((y - k)^2 / b^2) = 1

theorem hyperbola_center : is_center 2 4 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_center_l3770_377066


namespace NUMINAMATH_CALUDE_bank_deposit_problem_l3770_377044

/-- Calculates the total amount after maturity for a fixed-term deposit -/
def totalAmount (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal + principal * rate * time

theorem bank_deposit_problem :
  let principal : ℝ := 100000
  let rate : ℝ := 0.0315
  let time : ℝ := 2
  totalAmount principal rate time = 106300 := by
  sorry

end NUMINAMATH_CALUDE_bank_deposit_problem_l3770_377044


namespace NUMINAMATH_CALUDE_smallest_x_value_l3770_377096

theorem smallest_x_value (x y : ℕ+) (h : (3 : ℚ) / 5 = y / (468 + x)) : 2 ≤ x := by
  sorry

end NUMINAMATH_CALUDE_smallest_x_value_l3770_377096


namespace NUMINAMATH_CALUDE_not_all_F_zero_on_C_implies_exists_F_zero_not_on_C_l3770_377000

-- Define the curve C and the function F
variable (C : Set (ℝ × ℝ))
variable (F : ℝ → ℝ → ℝ)

-- Define the set of points satisfying F(x, y) = 0
def F_zero_set (F : ℝ → ℝ → ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | F p.1 p.2 = 0}

-- State the theorem
theorem not_all_F_zero_on_C_implies_exists_F_zero_not_on_C
  (h : ¬(F_zero_set F ⊆ C)) :
  ∃ p : ℝ × ℝ, p ∉ C ∧ F p.1 p.2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_not_all_F_zero_on_C_implies_exists_F_zero_not_on_C_l3770_377000


namespace NUMINAMATH_CALUDE_line_perp_parallel_implies_planes_perp_l3770_377022

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Plane → Prop)
variable (perpendicular_planes : Plane → Plane → Prop)

-- State the theorem
theorem line_perp_parallel_implies_planes_perp
  (l : Line) (α β : Plane)
  (h1 : perpendicular l α)
  (h2 : parallel l β) :
  perpendicular_planes α β :=
sorry

end NUMINAMATH_CALUDE_line_perp_parallel_implies_planes_perp_l3770_377022


namespace NUMINAMATH_CALUDE_equation_identity_l3770_377027

theorem equation_identity (x : ℝ) : (3*x - 2)*(2*x + 5) - x = 6*x^2 + 2*(5*x - 5) := by
  sorry

end NUMINAMATH_CALUDE_equation_identity_l3770_377027


namespace NUMINAMATH_CALUDE_complex_number_opposites_l3770_377058

theorem complex_number_opposites (b : ℝ) : 
  (Complex.re ((2 - b * Complex.I) * Complex.I) = 
   -Complex.im ((2 - b * Complex.I) * Complex.I)) → b = -2 := by
sorry

end NUMINAMATH_CALUDE_complex_number_opposites_l3770_377058


namespace NUMINAMATH_CALUDE_expand_expression_l3770_377039

theorem expand_expression (x : ℝ) : -3*x*(x^2 - x - 2) = -3*x^3 + 3*x^2 + 6*x := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l3770_377039


namespace NUMINAMATH_CALUDE_cut_square_problem_l3770_377088

/-- Given a square with integer side length and four isosceles right triangles
    cut from its corners, if the total area of the cut triangles is 40 square centimeters,
    then the area of the remaining rectangle is 24 square centimeters. -/
theorem cut_square_problem (s a b : ℕ) : 
  s = a + b →  -- The side length of the square is the sum of the leg lengths
  a^2 + b^2 = 40 →  -- The total area of cut triangles is 40
  s^2 - (a^2 + b^2) = 24 :=  -- The area of the remaining rectangle is 24
by sorry

end NUMINAMATH_CALUDE_cut_square_problem_l3770_377088


namespace NUMINAMATH_CALUDE_luxury_car_price_l3770_377099

def initial_price : ℝ := 80000

def discounts : List ℝ := [0.30, 0.25, 0.20, 0.15, 0.10, 0.05]

def apply_discount (price : ℝ) (discount : ℝ) : ℝ :=
  price * (1 - discount)

def final_price : ℝ := discounts.foldl apply_discount initial_price

theorem luxury_car_price : final_price = 24418.80 := by
  sorry

end NUMINAMATH_CALUDE_luxury_car_price_l3770_377099


namespace NUMINAMATH_CALUDE_investment_interest_rates_l3770_377089

theorem investment_interest_rates 
  (P1 P2 : ℝ) 
  (r1 r2 r3 r4 r5 : ℝ) :
  P1 / P2 = 2 / 3 →
  P1 * 5 * 8 / 100 = 840 →
  P2 * (r1 + r2 + r3 + r4 + r5) / 100 = 840 →
  r1 + r2 + r3 + r4 + r5 = 26.67 :=
by sorry

end NUMINAMATH_CALUDE_investment_interest_rates_l3770_377089


namespace NUMINAMATH_CALUDE_layla_score_difference_l3770_377065

/-- Given Layla's score and the total score, calculate the difference between Layla's and Nahima's scores -/
def score_difference (layla_score : ℕ) (total_score : ℕ) : ℕ :=
  layla_score - (total_score - layla_score)

/-- Theorem: Given Layla's score of 70 and a total score of 112, Layla scored 28 more points than Nahima -/
theorem layla_score_difference :
  score_difference 70 112 = 28 := by
  sorry

#eval score_difference 70 112

end NUMINAMATH_CALUDE_layla_score_difference_l3770_377065


namespace NUMINAMATH_CALUDE_solve_for_z_l3770_377046

theorem solve_for_z (x y z : ℚ) 
  (h1 : x = 11)
  (h2 : y = 8)
  (h3 : 2 * x + 3 * z = 5 * y) :
  z = 6 := by
sorry

end NUMINAMATH_CALUDE_solve_for_z_l3770_377046


namespace NUMINAMATH_CALUDE_village_population_problem_l3770_377075

theorem village_population_problem (P : ℕ) : 
  (P : ℝ) * 0.9 * 0.85 = 3213 → P = 4200 := by
  sorry

end NUMINAMATH_CALUDE_village_population_problem_l3770_377075


namespace NUMINAMATH_CALUDE_cosine_sine_relation_l3770_377008

open Real

theorem cosine_sine_relation (α β : ℝ) (x y : ℝ) 
  (h1 : 0 < α ∧ α < π/2) 
  (h2 : 0 < β ∧ β < π/2) 
  (h3 : cos (α + β) = -4/5)
  (h4 : sin β = x)
  (h5 : cos α = y)
  (h6 : 4/5 < x ∧ x < 1) :
  y = -4/5 * sqrt (1 - x^2) + 3/5 * x := by
sorry

end NUMINAMATH_CALUDE_cosine_sine_relation_l3770_377008


namespace NUMINAMATH_CALUDE_correct_calculation_l3770_377014

theorem correct_calculation (x : ℤ) : 
  x - 749 = 280 → x + 479 = 1508 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l3770_377014


namespace NUMINAMATH_CALUDE_coin_exchange_impossibility_l3770_377052

theorem coin_exchange_impossibility : ¬ ∃ n : ℕ, 1 + 4 * n = 26 := by
  sorry

end NUMINAMATH_CALUDE_coin_exchange_impossibility_l3770_377052


namespace NUMINAMATH_CALUDE_negation_of_existential_proposition_l3770_377076

theorem negation_of_existential_proposition :
  (¬ ∃ x : ℝ, Real.exp x - x - 1 ≤ 0) ↔ (∀ x : ℝ, Real.exp x - x - 1 > 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_existential_proposition_l3770_377076


namespace NUMINAMATH_CALUDE_money_distribution_l3770_377080

theorem money_distribution (a b c : ℕ) 
  (h1 : a + b + c = 1000)
  (h2 : a + c = 700)
  (h3 : b + c = 600) :
  c = 300 := by
  sorry

end NUMINAMATH_CALUDE_money_distribution_l3770_377080


namespace NUMINAMATH_CALUDE_sqrt_x_plus_y_equals_three_l3770_377074

theorem sqrt_x_plus_y_equals_three (x y : ℝ) (h : y = 4 + Real.sqrt (5 - x) + Real.sqrt (x - 5)) : 
  Real.sqrt (x + y) = 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_x_plus_y_equals_three_l3770_377074


namespace NUMINAMATH_CALUDE_annes_speed_ratio_l3770_377011

/-- Proves that the ratio of Anne's new cleaning rate to her original rate is 2:1 --/
theorem annes_speed_ratio :
  -- Bruce and Anne's original combined rate
  ∀ (B A : ℚ), B + A = 1/4 →
  -- Anne's original rate
  A = 1/12 →
  -- Bruce and Anne's new combined rate (with Anne's changed speed)
  ∀ (A' : ℚ), B + A' = 1/3 →
  -- The ratio of Anne's new rate to her original rate
  A' / A = 2 := by
sorry

end NUMINAMATH_CALUDE_annes_speed_ratio_l3770_377011


namespace NUMINAMATH_CALUDE_laptop_price_theorem_l3770_377053

/-- The sticker price of a laptop. -/
def stickerPrice : ℝ := 1100

/-- The price at store C after discount and rebate. -/
def storeCPrice (price : ℝ) : ℝ := 0.8 * price - 120

/-- The price at store D after discount. -/
def storeDPrice (price : ℝ) : ℝ := 0.7 * price

theorem laptop_price_theorem : 
  storeCPrice stickerPrice = storeDPrice stickerPrice - 10 := by sorry

end NUMINAMATH_CALUDE_laptop_price_theorem_l3770_377053


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_condition_l3770_377037

theorem quadratic_inequality_solution_condition (k : ℝ) :
  (∃ x : ℝ, x^2 - 8*x + k < 0) ↔ (0 < k ∧ k < 16) := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_condition_l3770_377037


namespace NUMINAMATH_CALUDE_first_company_daily_rate_l3770_377067

/-- The daily rate of the first car rental company -/
def first_company_rate : ℝ := 17.99

/-- The per-mile rate of the first car rental company -/
def first_company_per_mile : ℝ := 0.18

/-- The daily rate of City Rentals -/
def city_rentals_rate : ℝ := 18.95

/-- The per-mile rate of City Rentals -/
def city_rentals_per_mile : ℝ := 0.16

/-- The number of miles driven -/
def miles_driven : ℝ := 48.0

theorem first_company_daily_rate :
  first_company_rate + first_company_per_mile * miles_driven =
  city_rentals_rate + city_rentals_per_mile * miles_driven :=
by sorry

end NUMINAMATH_CALUDE_first_company_daily_rate_l3770_377067


namespace NUMINAMATH_CALUDE_ellipse_tangent_to_lines_l3770_377059

/-- The first line tangent to the ellipse -/
def line1 (x y : ℝ) : Prop := x + 2*y = 27

/-- The second line tangent to the ellipse -/
def line2 (x y : ℝ) : Prop := 7*x + 4*y = 81

/-- The equation of the ellipse -/
def ellipse_equation (x y : ℝ) : Prop := 162*x^2 + 81*y^2 = 13122

/-- Theorem stating that the given ellipse equation is tangent to both lines -/
theorem ellipse_tangent_to_lines :
  ∀ x y : ℝ, line1 x y ∨ line2 x y → ellipse_equation x y := by sorry

end NUMINAMATH_CALUDE_ellipse_tangent_to_lines_l3770_377059


namespace NUMINAMATH_CALUDE_earthwork_inequality_l3770_377078

/-- Proves the inequality for the required average daily earthwork to complete the project ahead of schedule. -/
theorem earthwork_inequality (total : ℝ) (days : ℕ) (first_day : ℝ) (ahead : ℕ) (x : ℝ) 
  (h_total : total = 300)
  (h_days : days = 6)
  (h_first_day : first_day = 60)
  (h_ahead : ahead = 2)
  : 3 * x ≥ total - first_day :=
by
  sorry

#check earthwork_inequality

end NUMINAMATH_CALUDE_earthwork_inequality_l3770_377078


namespace NUMINAMATH_CALUDE_smallest_k_sum_squares_div_150_l3770_377017

/-- The sum of squares from 1 to n -/
def sum_of_squares (n : ℕ) : ℕ := n * (n + 1) * (2 * n + 1) / 6

/-- 100 is the smallest positive integer k such that the sum of squares from 1 to k is divisible by 150 -/
theorem smallest_k_sum_squares_div_150 :
  ∀ k : ℕ, k > 0 → k < 100 → ¬(150 ∣ sum_of_squares k) ∧ (150 ∣ sum_of_squares 100) :=
by sorry

end NUMINAMATH_CALUDE_smallest_k_sum_squares_div_150_l3770_377017


namespace NUMINAMATH_CALUDE_root_product_of_quartic_l3770_377061

theorem root_product_of_quartic (a b c d : ℂ) : 
  (3 * a^4 - 8 * a^3 + a^2 + 4 * a - 12 = 0) ∧
  (3 * b^4 - 8 * b^3 + b^2 + 4 * b - 12 = 0) ∧
  (3 * c^4 - 8 * c^3 + c^2 + 4 * c - 12 = 0) ∧
  (3 * d^4 - 8 * d^3 + d^2 + 4 * d - 12 = 0) →
  a * b * c * d = -4 := by
sorry

end NUMINAMATH_CALUDE_root_product_of_quartic_l3770_377061


namespace NUMINAMATH_CALUDE_cameron_wins_probability_l3770_377035

-- Define the faces of each cube
def cameron_cube : Finset Nat := {6}
def dean_cube : Finset Nat := {1, 2, 3}
def olivia_cube : Finset Nat := {3, 6}

-- Define the number of faces for each number on each cube
def cameron_faces (n : Nat) : Nat := if n = 6 then 6 else 0
def dean_faces (n : Nat) : Nat := if n ∈ dean_cube then 2 else 0
def olivia_faces (n : Nat) : Nat := if n = 3 then 4 else if n = 6 then 2 else 0

-- Define the probability of rolling less than 6 for each player
def dean_prob_less_than_6 : ℚ :=
  (dean_faces 1 + dean_faces 2 + dean_faces 3) / 6

def olivia_prob_less_than_6 : ℚ :=
  olivia_faces 3 / 6

-- Theorem statement
theorem cameron_wins_probability :
  dean_prob_less_than_6 * olivia_prob_less_than_6 = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_cameron_wins_probability_l3770_377035


namespace NUMINAMATH_CALUDE_tan_theta_values_l3770_377056

theorem tan_theta_values (θ : Real) (h : 2 * Real.sin θ = 1 + Real.cos θ) : 
  Real.tan θ = 4/3 ∨ Real.tan θ = 0 := by
  sorry

end NUMINAMATH_CALUDE_tan_theta_values_l3770_377056


namespace NUMINAMATH_CALUDE_race_head_start_l3770_377018

/-- Calculates the head start given in a race between two runners with different speeds -/
def headStart (cristinaSpeed nicky_speed : ℝ) (catchUpTime : ℝ) : ℝ :=
  nicky_speed * catchUpTime

theorem race_head_start :
  let cristinaSpeed : ℝ := 5
  let nickySpeed : ℝ := 3
  let catchUpTime : ℝ := 27
  headStart cristinaSpeed nickySpeed catchUpTime = 81 := by
sorry

end NUMINAMATH_CALUDE_race_head_start_l3770_377018


namespace NUMINAMATH_CALUDE_closest_fraction_l3770_377038

def medals_won : ℚ := 23 / 150

def options : List ℚ := [1/5, 1/6, 1/7, 1/8, 1/9]

theorem closest_fraction (closest : ℚ) :
  closest ∈ options ∧
  ∀ x ∈ options, |medals_won - closest| ≤ |medals_won - x| :=
by sorry

end NUMINAMATH_CALUDE_closest_fraction_l3770_377038


namespace NUMINAMATH_CALUDE_triangle_problem_l3770_377071

theorem triangle_problem (a b c A B C : ℝ) : 
  (2 * c - 2 * a * Real.cos B = b) →
  (1/2 * b * c * Real.sin A = Real.sqrt 3 / 4) →
  (c^2 + a * b * Real.cos C + a^2 = 4) →
  (A = π/3 ∧ a = Real.sqrt 7 / 2) := by
  sorry

end NUMINAMATH_CALUDE_triangle_problem_l3770_377071


namespace NUMINAMATH_CALUDE_f_mono_increasing_condition_l3770_377043

/-- A quadratic function f(x) = ax^2 + x + 1 -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + x + 1

/-- The property of being monotonically increasing on (0, +∞) -/
def MonoIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 < x ∧ x < y → f x < f y

/-- The condition a ≥ 0 is sufficient but not necessary for f to be monotonically increasing on (0, +∞) -/
theorem f_mono_increasing_condition (a : ℝ) :
  (a ≥ 0 → MonoIncreasing (f a)) ∧
  ¬(MonoIncreasing (f a) → a ≥ 0) :=
sorry

end NUMINAMATH_CALUDE_f_mono_increasing_condition_l3770_377043


namespace NUMINAMATH_CALUDE_box_value_l3770_377062

theorem box_value (x : ℝ) : x * (-2) = 4 → x = -2 := by
  sorry

end NUMINAMATH_CALUDE_box_value_l3770_377062


namespace NUMINAMATH_CALUDE_coefficient_x4y2_in_expansion_coefficient_equals_60_l3770_377021

/-- The coefficient of x^4y^2 in the expansion of (x-2y)^6 is 60 -/
theorem coefficient_x4y2_in_expansion : ℕ :=
  60

/-- The binomial coefficient "6 choose 2" -/
def binomial_6_2 : ℕ := 15

/-- The expansion of (x-2y)^6 -/
def expansion (x y : ℝ) : ℝ := (x - 2*y)^6

/-- The coefficient of x^4y^2 in the expansion -/
def coefficient (x y : ℝ) : ℝ := binomial_6_2 * (-2)^2

theorem coefficient_equals_60 :
  coefficient = λ _ _ ↦ 60 :=
sorry

end NUMINAMATH_CALUDE_coefficient_x4y2_in_expansion_coefficient_equals_60_l3770_377021


namespace NUMINAMATH_CALUDE_rooms_already_painted_l3770_377095

theorem rooms_already_painted
  (total_rooms : ℕ)
  (time_per_room : ℕ)
  (time_left : ℕ)
  (h1 : total_rooms = 10)
  (h2 : time_per_room = 8)
  (h3 : time_left = 16) :
  total_rooms - (time_left / time_per_room) = 8 :=
by sorry

end NUMINAMATH_CALUDE_rooms_already_painted_l3770_377095


namespace NUMINAMATH_CALUDE_equation_roots_right_triangle_l3770_377031

-- Define the equation
def equation (x a b : ℝ) : Prop := |x^2 - 2*a*x + b| = 8

-- Define a function to check if three numbers form a right triangle
def is_right_triangle (x y z : ℝ) : Prop :=
  x^2 + y^2 = z^2 ∨ x^2 + z^2 = y^2 ∨ y^2 + z^2 = x^2

-- Theorem statement
theorem equation_roots_right_triangle (a b : ℝ) :
  (∃ x y z : ℝ, 
    x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    equation x a b ∧ equation y a b ∧ equation z a b ∧
    is_right_triangle x y z ∧
    (∀ w : ℝ, equation w a b → w = x ∨ w = y ∨ w = z)) →
  a + b = 264 :=
sorry

end NUMINAMATH_CALUDE_equation_roots_right_triangle_l3770_377031


namespace NUMINAMATH_CALUDE_red_jellybean_count_l3770_377007

/-- Given a jar of jellybeans with specific counts for different colors, 
    prove that the number of red jellybeans is 120. -/
theorem red_jellybean_count (total : ℕ) (blue purple orange : ℕ) 
  (h_total : total = 200)
  (h_blue : blue = 14)
  (h_purple : purple = 26)
  (h_orange : orange = 40) :
  total - (blue + purple + orange) = 120 := by
  sorry

end NUMINAMATH_CALUDE_red_jellybean_count_l3770_377007


namespace NUMINAMATH_CALUDE_toddler_count_problem_l3770_377024

/-- The actual number of toddlers given Bill's count and errors -/
def actual_toddler_count (counted : ℕ) (double_counted : ℕ) (hidden : ℕ) : ℕ :=
  counted - double_counted + hidden

/-- Theorem stating the actual number of toddlers in the given scenario -/
theorem toddler_count_problem : 
  actual_toddler_count 34 10 4 = 28 := by
  sorry

end NUMINAMATH_CALUDE_toddler_count_problem_l3770_377024


namespace NUMINAMATH_CALUDE_even_decreasing_function_ordering_l3770_377081

def is_even (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (-x)

def monotone_decreasing_on_pos (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, 0 < x → 0 < y → x < y → f y < f x

theorem even_decreasing_function_ordering (f : ℝ → ℝ) 
  (h_even : is_even f) 
  (h_decreasing : monotone_decreasing_on_pos f) :
  f 3 < f (-2) ∧ f (-2) < f 1 :=
sorry

end NUMINAMATH_CALUDE_even_decreasing_function_ordering_l3770_377081


namespace NUMINAMATH_CALUDE_complex_magnitude_fourth_power_l3770_377060

theorem complex_magnitude_fourth_power : 
  Complex.abs ((1 + Complex.I * Real.sqrt 3) ^ 4) = 16 := by sorry

end NUMINAMATH_CALUDE_complex_magnitude_fourth_power_l3770_377060


namespace NUMINAMATH_CALUDE_continuous_midpoint_property_implies_affine_l3770_377083

open Real

/-- A function satisfying the midpoint property -/
def HasMidpointProperty (f : ℝ → ℝ) : Prop :=
  ∀ x y, f ((x + y) / 2) = (f x + f y) / 2

/-- The main theorem stating that a continuous function with the midpoint property is affine -/
theorem continuous_midpoint_property_implies_affine
  (f : ℝ → ℝ) (hf : Continuous f) (hm : HasMidpointProperty f) :
  ∃ c b : ℝ, ∀ x, f x = c * x + b := by
  sorry

end NUMINAMATH_CALUDE_continuous_midpoint_property_implies_affine_l3770_377083


namespace NUMINAMATH_CALUDE_max_value_of_f_l3770_377041

def f (x : ℝ) := 12 * x - 4 * x^2

theorem max_value_of_f :
  ∃ (c : ℝ), ∀ (x : ℝ), f x ≤ c ∧ ∃ (x₀ : ℝ), f x₀ = c ∧ c = 9 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_f_l3770_377041


namespace NUMINAMATH_CALUDE_negative_three_to_fourth_equals_three_to_fourth_l3770_377012

theorem negative_three_to_fourth_equals_three_to_fourth : (-3) * (-3) * (-3) * (-3) = 3^4 := by
  sorry

end NUMINAMATH_CALUDE_negative_three_to_fourth_equals_three_to_fourth_l3770_377012


namespace NUMINAMATH_CALUDE_intersection_nonempty_a_subset_b_l3770_377029

-- Define the sets A and B
def A : Set ℝ := {x | (1 : ℝ) / (x - 3) < -1}
def B (a : ℝ) : Set ℝ := {x | (x - (a^2 + 2)) / (x - a) < 0}

-- Part 1: Intersection is non-empty
theorem intersection_nonempty (a : ℝ) : 
  (A ∩ B a).Nonempty ↔ a < 0 ∨ (0 < a ∧ a < 3) :=
sorry

-- Part 2: A is a subset of B
theorem a_subset_b (a : ℝ) :
  A ⊆ B a ↔ a ≤ -1 ∨ (1 ≤ a ∧ a ≤ 2) :=
sorry

end NUMINAMATH_CALUDE_intersection_nonempty_a_subset_b_l3770_377029


namespace NUMINAMATH_CALUDE_system_solutions_l3770_377002

/-- The system of equations -/
def system (x y z a : ℤ) : Prop :=
  (2*y*z + x - y - z = a) ∧
  (2*x*z - x + y - z = a) ∧
  (2*x*y - x - y + z = a)

/-- Condition for a to have four distinct integer solutions -/
def has_four_solutions (a : ℤ) : Prop :=
  ∃ k : ℤ, k % 2 = 1 ∧ k > 0 ∧ a = (k^2 - 1) / 8

theorem system_solutions (a : ℤ) :
  (¬ ∃ x₁ y₁ z₁ x₂ y₂ z₂ x₃ y₃ z₃ x₄ y₄ z₄ x₅ y₅ z₅ : ℤ,
    system x₁ y₁ z₁ a ∧ system x₂ y₂ z₂ a ∧ system x₃ y₃ z₃ a ∧
    system x₄ y₄ z₄ a ∧ system x₅ y₅ z₅ a ∧
    (x₁, y₁, z₁) ≠ (x₂, y₂, z₂) ∧ (x₁, y₁, z₁) ≠ (x₃, y₃, z₃) ∧
    (x₁, y₁, z₁) ≠ (x₄, y₄, z₄) ∧ (x₁, y₁, z₁) ≠ (x₅, y₅, z₅) ∧
    (x₂, y₂, z₂) ≠ (x₃, y₃, z₃) ∧ (x₂, y₂, z₂) ≠ (x₄, y₄, z₄) ∧
    (x₂, y₂, z₂) ≠ (x₅, y₅, z₅) ∧ (x₃, y₃, z₃) ≠ (x₄, y₄, z₄) ∧
    (x₃, y₃, z₃) ≠ (x₅, y₅, z₅) ∧ (x₄, y₄, z₄) ≠ (x₅, y₅, z₅)) ∧
  (has_four_solutions a ↔
    ∃ x₁ y₁ z₁ x₂ y₂ z₂ x₃ y₃ z₃ x₄ y₄ z₄ : ℤ,
      system x₁ y₁ z₁ a ∧ system x₂ y₂ z₂ a ∧
      system x₃ y₃ z₃ a ∧ system x₄ y₄ z₄ a ∧
      (x₁, y₁, z₁) ≠ (x₂, y₂, z₂) ∧ (x₁, y₁, z₁) ≠ (x₃, y₃, z₃) ∧
      (x₁, y₁, z₁) ≠ (x₄, y₄, z₄) ∧ (x₂, y₂, z₂) ≠ (x₃, y₃, z₃) ∧
      (x₂, y₂, z₂) ≠ (x₄, y₄, z₄) ∧ (x₃, y₃, z₃) ≠ (x₄, y₄, z₄)) :=
by sorry

end NUMINAMATH_CALUDE_system_solutions_l3770_377002


namespace NUMINAMATH_CALUDE_major_axis_length_l3770_377084

-- Define the ellipse equation
def ellipse_equation (x y : ℝ) : Prop := x^2 / 9 + y^2 / 4 = 1

-- State the theorem
theorem major_axis_length :
  ∃ (a b : ℝ), a > b ∧ a > 0 ∧ b > 0 ∧
  (∀ x y, ellipse_equation x y ↔ x^2 / a^2 + y^2 / b^2 = 1) ∧
  2 * a = 6 :=
sorry

end NUMINAMATH_CALUDE_major_axis_length_l3770_377084


namespace NUMINAMATH_CALUDE_polygon_with_44_diagonals_has_11_sides_l3770_377032

/-- The number of diagonals in a polygon with n sides -/
def diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem: A polygon with 44 diagonals has 11 sides -/
theorem polygon_with_44_diagonals_has_11_sides :
  ∃ (n : ℕ), n > 2 ∧ diagonals n = 44 → n = 11 := by
  sorry

end NUMINAMATH_CALUDE_polygon_with_44_diagonals_has_11_sides_l3770_377032


namespace NUMINAMATH_CALUDE_problem_statement_l3770_377006

/-- Given points P, Q, and O, and a function f, prove properties about f and a related triangle -/
theorem problem_statement 
  (P : ℝ × ℝ) 
  (Q : ℝ → ℝ × ℝ) 
  (f : ℝ → ℝ) 
  (A : ℝ) 
  (BC : ℝ) 
  (h1 : P = (Real.sqrt 3, 1))
  (h2 : ∀ x, Q x = (Real.cos x, Real.sin x))
  (h3 : ∀ x, f x = P.1 * (Q x).1 + P.2 * (Q x).2 - ((Q x).1 * P.1 + (Q x).2 * P.2))
  (h4 : f A = 4)
  (h5 : BC = 3) :
  (∀ x, f x = -2 * Real.sin (x + π/3) + 4) ∧ 
  (∀ T > 0, (∀ x, f (x + T) = f x) → T ≥ π) ∧
  (∀ a b c, a + b + c ≤ 3 + 2 * Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l3770_377006


namespace NUMINAMATH_CALUDE_octal_calculation_l3770_377072

/-- Converts a number from base 8 to base 10 --/
def octal_to_decimal (n : ℕ) : ℕ := sorry

/-- Converts a number from base 10 to base 8 --/
def decimal_to_octal (n : ℕ) : ℕ := sorry

/-- Multiplies two numbers in base 8 --/
def octal_multiply (a b : ℕ) : ℕ := 
  decimal_to_octal (octal_to_decimal a * octal_to_decimal b)

/-- Subtracts two numbers in base 8 --/
def octal_subtract (a b : ℕ) : ℕ := 
  decimal_to_octal (octal_to_decimal a - octal_to_decimal b)

theorem octal_calculation : 
  octal_subtract (octal_multiply 245 5) 107 = 1356 := by sorry

end NUMINAMATH_CALUDE_octal_calculation_l3770_377072


namespace NUMINAMATH_CALUDE_f_decreasing_range_l3770_377048

/-- A piecewise function f(x) defined on the real numbers. -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then (3*a - 1)*x + 4*a else Real.log x / Real.log a

/-- The theorem stating the range of 'a' for which f is decreasing on ℝ. -/
theorem f_decreasing_range :
  ∀ a : ℝ, (∀ x y : ℝ, x < y → f a x > f a y) ↔ 1/7 ≤ a ∧ a < 1/3 :=
sorry

end NUMINAMATH_CALUDE_f_decreasing_range_l3770_377048


namespace NUMINAMATH_CALUDE_surface_area_ratio_l3770_377020

/-- Represents a regular tetrahedron -/
structure RegularTetrahedron where
  /-- The edge length of the tetrahedron -/
  edge_length : ℝ
  /-- Assumption that the edge length is positive -/
  edge_positive : edge_length > 0

/-- The surface area of a regular tetrahedron -/
def surface_area_tetrahedron (t : RegularTetrahedron) : ℝ := sorry

/-- The surface area of the inscribed sphere of a regular tetrahedron -/
def surface_area_inscribed_sphere (t : RegularTetrahedron) : ℝ := sorry

/-- Theorem stating the ratio of the surface areas -/
theorem surface_area_ratio (t : RegularTetrahedron) :
  surface_area_tetrahedron t / surface_area_inscribed_sphere t = 6 * Real.sqrt 3 / Real.pi := by
  sorry

end NUMINAMATH_CALUDE_surface_area_ratio_l3770_377020


namespace NUMINAMATH_CALUDE_percentage_increase_l3770_377028

theorem percentage_increase (x y z : ℝ) (h1 : y = 0.5 * z) (h2 : x = 0.65 * z) :
  (x - y) / y * 100 = 30 := by
  sorry

end NUMINAMATH_CALUDE_percentage_increase_l3770_377028


namespace NUMINAMATH_CALUDE_smallest_fraction_greater_than_five_sixths_l3770_377091

theorem smallest_fraction_greater_than_five_sixths :
  ∀ a b : ℕ, 10 ≤ a ∧ a ≤ 99 ∧ 10 ≤ b ∧ b ≤ 99 →
    (a : ℚ) / b > 5 / 6 →
    81 / 97 ≤ (a : ℚ) / b :=
by sorry

end NUMINAMATH_CALUDE_smallest_fraction_greater_than_five_sixths_l3770_377091


namespace NUMINAMATH_CALUDE_sphere_to_cone_height_l3770_377010

theorem sphere_to_cone_height (R : ℝ) (h : ℝ) (r : ℝ) (l : ℝ) : 
  R > 0 → r > 0 → h > 0 → l > 0 →
  (4 / 3) * Real.pi * R^3 = (1 / 3) * Real.pi * r^2 * h →  -- Volume conservation
  Real.pi * r * l = 3 * Real.pi * r^2 →  -- Lateral surface area condition
  l^2 = r^2 + h^2 →  -- Pythagorean theorem
  h = 4 * R * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_sphere_to_cone_height_l3770_377010


namespace NUMINAMATH_CALUDE_handbag_price_l3770_377097

/-- Calculates the total selling price of a product given its original price, discount rate, and tax rate. -/
def totalSellingPrice (originalPrice : ℝ) (discountRate : ℝ) (taxRate : ℝ) : ℝ :=
  let discountedPrice := originalPrice * (1 - discountRate)
  discountedPrice * (1 + taxRate)

/-- Theorem stating that the total selling price of a $100 product with 30% discount and 8% tax is $75.6 -/
theorem handbag_price : 
  totalSellingPrice 100 0.3 0.08 = 75.6 := by
  sorry

end NUMINAMATH_CALUDE_handbag_price_l3770_377097


namespace NUMINAMATH_CALUDE_distance_traveled_l3770_377068

/-- Represents the actual distance traveled in kilometers -/
def actual_distance : ℝ := 33.75

/-- Represents the initial walking speed in km/hr -/
def initial_speed : ℝ := 15

/-- Represents the faster walking speed in km/hr -/
def faster_speed : ℝ := 35

/-- Represents the fraction of the distance that is uphill -/
def uphill_fraction : ℝ := 0.6

/-- Represents the decrease in speed for uphill portion -/
def uphill_speed_decrease : ℝ := 0.1

/-- Represents the additional distance covered at faster speed -/
def additional_distance : ℝ := 45

theorem distance_traveled :
  ∃ (time : ℝ),
    actual_distance = initial_speed * time ∧
    actual_distance + additional_distance = faster_speed * time ∧
    actual_distance * uphill_fraction = (faster_speed * (1 - uphill_speed_decrease)) * (time * uphill_fraction) ∧
    actual_distance * (1 - uphill_fraction) = faster_speed * (time * (1 - uphill_fraction)) :=
by sorry

end NUMINAMATH_CALUDE_distance_traveled_l3770_377068


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3770_377013

theorem arithmetic_sequence_sum (a : ℕ → ℝ) (n : ℕ) :
  (∀ k, a (k + 1) - a k = a 1 - a 0) →  -- arithmetic sequence condition
  a 0 = 3 →                            -- first term is 3
  a n = 39 →                           -- last term is 39
  n ≥ 2 →                              -- ensure at least 3 terms
  a (n - 1) + a (n - 2) = 72 :=         -- sum of last two terms before 39
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3770_377013


namespace NUMINAMATH_CALUDE_inverse_mod_31_l3770_377025

theorem inverse_mod_31 (h : (11⁻¹ : ZMod 31) = 3) : (20⁻¹ : ZMod 31) = 28 := by
  sorry

end NUMINAMATH_CALUDE_inverse_mod_31_l3770_377025


namespace NUMINAMATH_CALUDE_number_with_specific_remainder_l3770_377019

theorem number_with_specific_remainder : ∃ x : ℕ, ∃ k : ℕ, 
  x = 29 * k + 8 ∧ 
  1490 % 29 = 11 ∧ 
  (∀ m : ℕ, m > 29 → (x % m ≠ 8 ∨ 1490 % m ≠ 11)) :=
by sorry

end NUMINAMATH_CALUDE_number_with_specific_remainder_l3770_377019


namespace NUMINAMATH_CALUDE_phi_function_form_l3770_377082

/-- A direct proportion function -/
def DirectProportion (f : ℝ → ℝ) : Prop :=
  ∃ m : ℝ, m ≠ 0 ∧ ∀ x, f x = m * x

/-- An inverse proportion function -/
def InverseProportion (g : ℝ → ℝ) : Prop :=
  ∃ n : ℝ, n ≠ 0 ∧ ∀ x, x ≠ 0 → g x = n / x

/-- The main theorem -/
theorem phi_function_form (f g : ℝ → ℝ) (φ : ℝ → ℝ) :
  DirectProportion f →
  InverseProportion g →
  (∀ x, φ x = f x + g x) →
  φ 1 = 8 →
  (∃ x, φ x = 16) →
  ∀ x, x ≠ 0 → φ x = 3 * x + 5 / x := by
  sorry

end NUMINAMATH_CALUDE_phi_function_form_l3770_377082


namespace NUMINAMATH_CALUDE_quadratic_with_inequality_has_negative_root_l3770_377045

/-- A quadratic polynomial with two distinct roots satisfying a specific inequality has at least one negative root. -/
theorem quadratic_with_inequality_has_negative_root 
  (f : ℝ → ℝ) 
  (h_quadratic : ∃ a b c : ℝ, ∀ x, f x = a * x^2 + b * x + c)
  (h_distinct_roots : ∃ r₁ r₂ : ℝ, r₁ ≠ r₂ ∧ f r₁ = 0 ∧ f r₂ = 0)
  (h_inequality : ∀ a b : ℝ, f (a^2 + b^2) ≥ f (2*a*b)) :
  ∃ r : ℝ, f r = 0 ∧ r < 0 :=
sorry

end NUMINAMATH_CALUDE_quadratic_with_inequality_has_negative_root_l3770_377045


namespace NUMINAMATH_CALUDE_circle_radii_ratio_l3770_377004

theorem circle_radii_ratio (A₁ A₂ r₁ r₂ : ℝ) (h_area_ratio : A₁ / A₂ = 98 / 63)
  (h_area_formula₁ : A₁ = π * r₁^2) (h_area_formula₂ : A₂ = π * r₂^2) :
  ∃ (x y z : ℕ), (r₁ / r₂ = x * Real.sqrt y / z) ∧ (x * Real.sqrt y / z = Real.sqrt 14 / 3) ∧ x + y + z = 18 := by
  sorry

end NUMINAMATH_CALUDE_circle_radii_ratio_l3770_377004


namespace NUMINAMATH_CALUDE_isosceles_triangle_base_angle_l3770_377090

theorem isosceles_triangle_base_angle (α β γ : ℝ) : 
  -- The triangle is isosceles
  (α = β ∨ β = γ ∨ α = γ) →
  -- The sum of angles in a triangle is 180°
  α + β + γ = 180 →
  -- One angle is 70°
  (α = 70 ∨ β = 70 ∨ γ = 70) →
  -- The base angle is either 70° or 55°
  (α = 70 ∨ α = 55 ∨ β = 70 ∨ β = 55 ∨ γ = 70 ∨ γ = 55) :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_base_angle_l3770_377090


namespace NUMINAMATH_CALUDE_pool_capacity_correct_l3770_377042

/-- The amount of water Grace's pool can contain -/
def pool_capacity : ℕ := 390

/-- The rate at which the first hose sprays water -/
def first_hose_rate : ℕ := 50

/-- The rate at which the second hose sprays water -/
def second_hose_rate : ℕ := 70

/-- The time the first hose runs alone -/
def first_hose_time : ℕ := 3

/-- The time both hoses run together -/
def both_hoses_time : ℕ := 2

/-- Theorem stating that the pool capacity is correct given the conditions -/
theorem pool_capacity_correct :
  pool_capacity = first_hose_rate * first_hose_time + 
    (first_hose_rate + second_hose_rate) * both_hoses_time :=
by sorry

end NUMINAMATH_CALUDE_pool_capacity_correct_l3770_377042


namespace NUMINAMATH_CALUDE_equation_solution_exists_l3770_377093

theorem equation_solution_exists (m n : ℤ) :
  ∃ (w x y z : ℤ), w + x + 2*y + 2*z = m ∧ 2*w - 2*x + y - z = n := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_exists_l3770_377093


namespace NUMINAMATH_CALUDE_decimal_to_binary_38_l3770_377005

theorem decimal_to_binary_38 : 
  (38 : ℕ).digits 2 = [0, 1, 1, 0, 0, 1] :=
by sorry

end NUMINAMATH_CALUDE_decimal_to_binary_38_l3770_377005


namespace NUMINAMATH_CALUDE_sum_difference_remainder_l3770_377034

theorem sum_difference_remainder (a b c : ℤ) 
  (ha : ∃ k : ℤ, a = 3 * k)
  (hb : ∃ k : ℤ, b = 3 * k + 1)
  (hc : ∃ k : ℤ, c = 3 * k - 1) :
  ∃ k : ℤ, a + b - c = 3 * k - 1 := by
sorry

end NUMINAMATH_CALUDE_sum_difference_remainder_l3770_377034


namespace NUMINAMATH_CALUDE_binary_addition_subtraction_l3770_377051

/-- Converts a binary number (represented as a list of bits) to a natural number. -/
def binary_to_nat (bits : List Bool) : ℕ :=
  bits.foldl (fun acc b => 2 * acc + if b then 1 else 0) 0

/-- Represents the binary number 1101₂ -/
def b1101 : List Bool := [true, true, false, true]

/-- Represents the binary number 111₂ -/
def b111 : List Bool := [true, true, true]

/-- Represents the binary number 1010₂ -/
def b1010 : List Bool := [true, false, true, false]

/-- Represents the binary number 1011₂ -/
def b1011 : List Bool := [true, false, true, true]

/-- Represents the binary number 11001₂ (the expected result) -/
def b11001 : List Bool := [true, true, false, false, true]

/-- The main theorem to prove -/
theorem binary_addition_subtraction :
  binary_to_nat b1101 + binary_to_nat b111 - binary_to_nat b1010 + binary_to_nat b1011 =
  binary_to_nat b11001 := by
  sorry

end NUMINAMATH_CALUDE_binary_addition_subtraction_l3770_377051


namespace NUMINAMATH_CALUDE_sum_and_double_l3770_377026

theorem sum_and_double : (2345 + 3452 + 4523 + 5234) * 2 = 31108 := by
  sorry

end NUMINAMATH_CALUDE_sum_and_double_l3770_377026


namespace NUMINAMATH_CALUDE_unique_fraction_decomposition_l3770_377033

theorem unique_fraction_decomposition (p : ℕ) (h_prime : Nat.Prime p) (h_not_two : p ≠ 2) :
  ∃! (x y : ℕ), x > 0 ∧ y > 0 ∧ x ≠ y ∧ (2 : ℚ) / p = 1 / x + 1 / y ∧ 
  x = (p^2 + p) / 2 ∧ y = (p + 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_fraction_decomposition_l3770_377033


namespace NUMINAMATH_CALUDE_books_lost_during_move_phil_books_lost_l3770_377047

theorem books_lost_during_move (initial_books : ℕ) (pages_per_book : ℕ) (pages_left : ℕ) : ℕ :=
  let total_pages := initial_books * pages_per_book
  let pages_lost := total_pages - pages_left
  pages_lost / pages_per_book

theorem phil_books_lost :
  books_lost_during_move 10 100 800 = 2 := by
  sorry

end NUMINAMATH_CALUDE_books_lost_during_move_phil_books_lost_l3770_377047


namespace NUMINAMATH_CALUDE_logarithm_equality_l3770_377073

/-- Given the conditions on logarithms and the equation involving x^y, 
    prove that y equals 2q - p - r -/
theorem logarithm_equality (a b c x : ℝ) (p q r y : ℝ) 
  (h1 : x ≠ 1)
  (h2 : Real.log a / p = Real.log b / q)
  (h3 : Real.log b / q = Real.log c / r)
  (h4 : Real.log b / q = Real.log x)
  (h5 : b^2 / (a * c) = x^y) :
  y = 2*q - p - r := by
  sorry

end NUMINAMATH_CALUDE_logarithm_equality_l3770_377073


namespace NUMINAMATH_CALUDE_total_insects_eaten_l3770_377036

/-- The number of geckos -/
def num_geckos : ℕ := 5

/-- The number of insects eaten by each gecko -/
def insects_per_gecko : ℕ := 6

/-- The number of lizards -/
def num_lizards : ℕ := 3

/-- The number of insects eaten by each lizard -/
def insects_per_lizard : ℕ := 2 * insects_per_gecko

/-- The total number of insects eaten by all animals -/
def total_insects : ℕ := num_geckos * insects_per_gecko + num_lizards * insects_per_lizard

theorem total_insects_eaten :
  total_insects = 66 := by sorry

end NUMINAMATH_CALUDE_total_insects_eaten_l3770_377036


namespace NUMINAMATH_CALUDE_total_air_conditioner_sales_l3770_377023

theorem total_air_conditioner_sales (june_sales : ℕ) (july_increase : ℚ) : 
  june_sales = 96 →
  july_increase = 1/3 →
  june_sales + (june_sales * (1 + july_increase)).floor = 224 := by
  sorry

end NUMINAMATH_CALUDE_total_air_conditioner_sales_l3770_377023


namespace NUMINAMATH_CALUDE_tan_value_from_equation_l3770_377009

theorem tan_value_from_equation (x : ℝ) :
  (1 - Real.cos x + Real.sin x) / (1 + Real.cos x + Real.sin x) = -2 →
  Real.tan x = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_value_from_equation_l3770_377009


namespace NUMINAMATH_CALUDE_logic_statements_correctness_l3770_377054

theorem logic_statements_correctness :
  ∃! (n : Nat), n = 2 ∧
  (((∀ p q, p ∧ q → p ∨ q) ∧ (∃ p q, p ∨ q ∧ ¬(p ∧ q))) ∧
   ((∃ p q, ¬(p ∧ q) ∧ ¬(p ∨ q)) ∨ (∀ p q, p ∨ q → ¬(p ∧ q))) ∧
   ((∀ p q, ¬p → p ∨ q) ∧ (∃ p q, p ∨ q ∧ p)) ∧
   ((∀ p q, ¬p → ¬(p ∧ q)) ∧ (∃ p q, ¬(p ∧ q) ∧ p))) :=
by sorry

end NUMINAMATH_CALUDE_logic_statements_correctness_l3770_377054


namespace NUMINAMATH_CALUDE_rectangular_field_area_l3770_377064

/-- Calculates the area of a rectangular field given its perimeter and width-to-length ratio --/
theorem rectangular_field_area (perimeter : ℝ) (width_ratio : ℝ) : 
  perimeter = 72 ∧ width_ratio = 1/3 → 
  (perimeter / (2 * (1 + 1/width_ratio))) * (perimeter / (2 * (1 + 1/width_ratio))) / width_ratio = 243 := by
sorry

end NUMINAMATH_CALUDE_rectangular_field_area_l3770_377064


namespace NUMINAMATH_CALUDE_calculator_result_l3770_377057

def special_key (x : ℚ) : ℚ := 1 / (1 - x)

def apply_n_times (f : ℚ → ℚ) (x : ℚ) (n : ℕ) : ℚ :=
  match n with
  | 0 => x
  | n + 1 => f (apply_n_times f x n)

theorem calculator_result : apply_n_times special_key 3 50 = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_calculator_result_l3770_377057


namespace NUMINAMATH_CALUDE_car_repair_cost_l3770_377055

/-- Calculates the total cost for a car repair given the hourly rate, hours worked per day,
    number of days worked, and cost of parts. -/
def total_cost (hourly_rate : ℕ) (hours_per_day : ℕ) (days_worked : ℕ) (parts_cost : ℕ) : ℕ :=
  hourly_rate * hours_per_day * days_worked + parts_cost

/-- Proves that the total cost for the car repair is $9220 given the specified conditions. -/
theorem car_repair_cost :
  total_cost 60 8 14 2500 = 9220 := by
  sorry

end NUMINAMATH_CALUDE_car_repair_cost_l3770_377055


namespace NUMINAMATH_CALUDE_triangle_inequality_proof_l3770_377087

/-- A structure representing a set of three line segments. -/
structure LineSegmentSet where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The triangle inequality theorem for a set of line segments. -/
def satisfies_triangle_inequality (s : LineSegmentSet) : Prop :=
  s.a + s.b > s.c ∧ s.b + s.c > s.a ∧ s.c + s.a > s.b

/-- The set of line segments that can form a triangle. -/
def triangle_set : LineSegmentSet :=
  { a := 3, b := 4, c := 5 }

/-- The sets of line segments that cannot form triangles. -/
def non_triangle_sets : List LineSegmentSet :=
  [{ a := 1, b := 2, c := 3 },
   { a := 4, b := 5, c := 10 },
   { a := 6, b := 9, c := 2 }]

theorem triangle_inequality_proof :
  satisfies_triangle_inequality triangle_set ∧
  ∀ s ∈ non_triangle_sets, ¬satisfies_triangle_inequality s :=
sorry

end NUMINAMATH_CALUDE_triangle_inequality_proof_l3770_377087


namespace NUMINAMATH_CALUDE_probability_four_twos_correct_l3770_377001

def num_dice : ℕ := 12
def num_sides : ℕ := 6
def num_success : ℕ := 4

def probability_exactly_four_twos : ℚ :=
  (Nat.choose num_dice num_success) *
  (1 / num_sides) ^ num_success *
  ((num_sides - 1) / num_sides) ^ (num_dice - num_success)

theorem probability_four_twos_correct :
  probability_exactly_four_twos = 
    (Nat.choose num_dice num_success) *
    (1 / num_sides) ^ num_success *
    ((num_sides - 1) / num_sides) ^ (num_dice - num_success) :=
by sorry

end NUMINAMATH_CALUDE_probability_four_twos_correct_l3770_377001


namespace NUMINAMATH_CALUDE_rectangle_square_area_ratio_l3770_377079

theorem rectangle_square_area_ratio : 
  let s : ℝ := 20
  let longer_side : ℝ := 1.05 * s
  let shorter_side : ℝ := 0.85 * s
  let area_R : ℝ := longer_side * shorter_side
  let area_S : ℝ := s * s
  area_R / area_S = 357 / 400 := by
sorry

end NUMINAMATH_CALUDE_rectangle_square_area_ratio_l3770_377079


namespace NUMINAMATH_CALUDE_angle_b_is_30_degrees_l3770_377063

-- Define the structure of our triangle
structure Triangle :=
  (A B C : ℝ)  -- Angles of the triangle
  (white_angle gray_angle : ℝ)  -- Measures of white and gray angles
  (b : ℝ)  -- The angle we want to determine

-- State the theorem
theorem angle_b_is_30_degrees (t : Triangle) : 
  t.A = 60 ∧  -- Given angle is 60°
  t.A + t.B + t.C = 180 ∧  -- Sum of angles in a triangle is 180°
  t.A + 2 * t.gray_angle + (180 - 2 * t.white_angle) = 180 ∧  -- Equation for triangle ABC
  t.gray_angle + t.b + (180 - 2 * t.white_angle) = 180  -- Equation for triangle BCD
  → t.b = 30 := by
sorry  -- Proof is omitted as per instructions

end NUMINAMATH_CALUDE_angle_b_is_30_degrees_l3770_377063


namespace NUMINAMATH_CALUDE_min_reciprocal_sum_l3770_377049

theorem min_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 4 * y = 1) :
  (1 / x + 1 / y) ≥ 9 :=
by sorry

end NUMINAMATH_CALUDE_min_reciprocal_sum_l3770_377049


namespace NUMINAMATH_CALUDE_locus_equation_l3770_377040

def point_A : ℝ × ℝ := (4, 0)
def point_B : ℝ × ℝ := (1, 0)

theorem locus_equation (x y : ℝ) :
  let M := (x, y)
  let dist_MA := Real.sqrt ((x - point_A.1)^2 + (y - point_A.2)^2)
  let dist_MB := Real.sqrt ((x - point_B.1)^2 + (y - point_B.2)^2)
  dist_MA = (1/2) * dist_MB → x^2 + y^2 = 4 := by
sorry

end NUMINAMATH_CALUDE_locus_equation_l3770_377040


namespace NUMINAMATH_CALUDE_wood_measurement_theorem_l3770_377050

/-- Represents the system of equations for the wood measurement problem -/
def wood_measurement_equations (x y : ℝ) : Prop :=
  (y - x = 4.5) ∧ (x - 1/2 * y = 1)

/-- Theorem stating that the given system of equations correctly represents the wood measurement problem -/
theorem wood_measurement_theorem (x y : ℝ) :
  (∃ wood_length : ℝ, wood_length = x) →
  (∃ rope_length : ℝ, rope_length = y) →
  (y - x = 4.5) →
  (x - 1/2 * y = 1) →
  wood_measurement_equations x y :=
by
  sorry

end NUMINAMATH_CALUDE_wood_measurement_theorem_l3770_377050


namespace NUMINAMATH_CALUDE_bag_properties_l3770_377077

/-- A bag containing colored balls -/
structure Bag where
  red : ℕ
  black : ℕ
  white : ℕ

/-- The scoring system for the balls -/
def score (color : String) : ℕ :=
  match color with
  | "white" => 2
  | "black" => 1
  | "red" => 0
  | _ => 0

/-- The theorem stating the properties of the bag and the probabilities -/
theorem bag_properties (b : Bag) : 
  b.red = 1 ∧ b.black = 1 ∧ b.white = 2 →
  (b.white : ℚ) / (b.red + b.black + b.white : ℚ) = 1/2 ∧
  (2 : ℚ) / ((b.red + b.black + b.white) * (b.red + b.black + b.white - 1) : ℚ) = 1/3 :=
by sorry

#check bag_properties

end NUMINAMATH_CALUDE_bag_properties_l3770_377077


namespace NUMINAMATH_CALUDE_work_completion_time_l3770_377003

/-- The number of days it takes A to complete the work -/
def days_A : ℝ := 4

/-- The number of days it takes C to complete the work -/
def days_C : ℝ := 8

/-- The number of days it takes A, B, and C together to complete the work -/
def days_ABC : ℝ := 2

/-- The number of days it takes B to complete the work -/
def days_B : ℝ := 8

theorem work_completion_time :
  (1 / days_A + 1 / days_B + 1 / days_C = 1 / days_ABC) :=
by sorry

end NUMINAMATH_CALUDE_work_completion_time_l3770_377003


namespace NUMINAMATH_CALUDE_xy_value_l3770_377015

theorem xy_value (x y : ℝ) (h : x * (x + y) = x^2 + 12) : x * y = 12 := by
  sorry

end NUMINAMATH_CALUDE_xy_value_l3770_377015


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l3770_377086

/-- Sums of arithmetic sequences -/
def S (n : ℕ) : ℝ := sorry

/-- Sums of arithmetic sequences -/
def T (n : ℕ) : ℝ := sorry

/-- Terms of the first arithmetic sequence -/
def a : ℕ → ℝ := sorry

/-- Terms of the second arithmetic sequence -/
def b : ℕ → ℝ := sorry

theorem arithmetic_sequence_ratio :
  (∀ n : ℕ+, S n / T n = n / (2 * n + 1)) →
  a 6 / b 6 = 11 / 23 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l3770_377086
