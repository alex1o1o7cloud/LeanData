import Mathlib

namespace NUMINAMATH_CALUDE_bench_seating_theorem_l695_69525

/-- The number of ways to arrange people on a bench with empty seats -/
def seating_arrangements (total_seats : ℕ) (people : ℕ) (adjacent_empty : ℕ) : ℕ :=
  (people.factorial) * (people + 1).factorial / 2

/-- Theorem: There are 480 ways to arrange 4 people on a bench with 7 seats,
    such that exactly 2 of the 3 empty seats are adjacent -/
theorem bench_seating_theorem :
  seating_arrangements 7 4 2 = 480 := by
  sorry

#eval seating_arrangements 7 4 2

end NUMINAMATH_CALUDE_bench_seating_theorem_l695_69525


namespace NUMINAMATH_CALUDE_vegetable_sale_mass_l695_69553

theorem vegetable_sale_mass (carrots zucchini broccoli : ℝ) 
  (h_carrots : carrots = 15)
  (h_zucchini : zucchini = 13)
  (h_broccoli : broccoli = 8) :
  (carrots + zucchini + broccoli) / 2 = 18 := by
  sorry

end NUMINAMATH_CALUDE_vegetable_sale_mass_l695_69553


namespace NUMINAMATH_CALUDE_binomial_expansion_103_minus_2_pow_5_l695_69530

theorem binomial_expansion_103_minus_2_pow_5 :
  (103 - 2)^5 = 10510100501 := by
  sorry

end NUMINAMATH_CALUDE_binomial_expansion_103_minus_2_pow_5_l695_69530


namespace NUMINAMATH_CALUDE_associated_equation_l695_69551

def equation1 (x : ℝ) : Prop := 5 * x - 2 = 0

def equation2 (x : ℝ) : Prop := 3/4 * x + 1 = 0

def equation3 (x : ℝ) : Prop := x - (3 * x + 1) = -5

def inequality_system (x : ℝ) : Prop := 2 * x - 5 > 3 * x - 8 ∧ -4 * x + 3 < x - 4

theorem associated_equation : 
  ∃ (x : ℝ), equation3 x ∧ inequality_system x ∧
  (∀ (y : ℝ), equation1 y → ¬inequality_system y) ∧
  (∀ (y : ℝ), equation2 y → ¬inequality_system y) :=
sorry

end NUMINAMATH_CALUDE_associated_equation_l695_69551


namespace NUMINAMATH_CALUDE_intersection_A_B_intersection_A_C_condition_l695_69576

-- Define the sets A, B, and C
def A : Set ℝ := {y | ∃ x, 1 ≤ x ∧ x ≤ 2 ∧ y = 2^x}
def B : Set ℝ := {x | 0 < Real.log x ∧ Real.log x < 1}
def C (t : ℝ) : Set ℝ := {x | t + 1 < x ∧ x < 2*t}

-- Theorem 1: Intersection of A and B
theorem intersection_A_B : A ∩ B = {x | 2 ≤ x ∧ x < Real.exp 1} := by sorry

-- Theorem 2: Condition for t when A ∩ C = C
theorem intersection_A_C_condition (t : ℝ) : A ∩ C t = C t → t ≤ 2 := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_intersection_A_C_condition_l695_69576


namespace NUMINAMATH_CALUDE_inequality_not_true_l695_69511

theorem inequality_not_true (x y : ℝ) (h : x > y) : ¬(-3*x + 6 > -3*y + 6) := by
  sorry

end NUMINAMATH_CALUDE_inequality_not_true_l695_69511


namespace NUMINAMATH_CALUDE_geometric_series_first_term_l695_69541

theorem geometric_series_first_term
  (a r : ℝ)
  (h1 : a / (1 - r) = 30)
  (h2 : a^2 / (1 - r^2) = 120)
  : a = 240 / 7 :=
by sorry

end NUMINAMATH_CALUDE_geometric_series_first_term_l695_69541


namespace NUMINAMATH_CALUDE_sum_of_roots_l695_69545

theorem sum_of_roots (m n : ℝ) : 
  m^2 - 3*m - 2 = 0 → n^2 - 3*n - 2 = 0 → m + n = 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_l695_69545


namespace NUMINAMATH_CALUDE_students_in_band_or_sports_l695_69529

theorem students_in_band_or_sports 
  (total : ℕ) 
  (band : ℕ) 
  (sports : ℕ) 
  (both : ℕ) 
  (h_total : total = 320)
  (h_band : band = 85)
  (h_sports : sports = 200)
  (h_both : both = 60) :
  band + sports - both = 225 := by
  sorry

end NUMINAMATH_CALUDE_students_in_band_or_sports_l695_69529


namespace NUMINAMATH_CALUDE_solve_for_t_l695_69585

theorem solve_for_t (s t : ℚ) (eq1 : 11 * s + 7 * t = 170) (eq2 : s = 2 * t - 3) : t = 203 / 29 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_t_l695_69585


namespace NUMINAMATH_CALUDE_car_ac_price_difference_l695_69502

/-- Given that the price of a car and AC are in the ratio 3:2, and the AC costs $1500,
    prove that the car costs $750 more than the AC. -/
theorem car_ac_price_difference :
  ∀ (car_price ac_price : ℕ),
    car_price / ac_price = 3 / 2 →
    ac_price = 1500 →
    car_price - ac_price = 750 :=
by
  sorry

end NUMINAMATH_CALUDE_car_ac_price_difference_l695_69502


namespace NUMINAMATH_CALUDE_remainder_73_power_73_plus_73_mod_137_l695_69517

theorem remainder_73_power_73_plus_73_mod_137 :
  ∃ k : ℤ, 73^73 + 73 = 137 * k + 9 :=
by
  sorry

end NUMINAMATH_CALUDE_remainder_73_power_73_plus_73_mod_137_l695_69517


namespace NUMINAMATH_CALUDE_expression_simplification_l695_69526

theorem expression_simplification (a : ℝ) (h1 : a ≠ 0) (h2 : a ≠ -1) :
  (2*a + 2) / a / (4 / a^2) - a / (a + 1) = (a^3 + 2*a^2 - a) / (2*a + 2) :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_l695_69526


namespace NUMINAMATH_CALUDE_range_of_a_l695_69584

/-- Piecewise function f(x) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x > 1 then x else a * x^2 + 2 * x

/-- The range of a given the conditions -/
theorem range_of_a (a : ℝ) :
  (∀ y : ℝ, ∃ x : ℝ, f a x = y) →
  -1 ≤ a ∧ a ≤ 0 :=
by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l695_69584


namespace NUMINAMATH_CALUDE_wheat_bread_served_l695_69592

/-- The number of loaves of wheat bread served at a restaurant -/
def wheat_bread : ℝ := 0.9 - 0.4

/-- The total number of loaves served at the restaurant -/
def total_loaves : ℝ := 0.9

/-- The number of loaves of white bread served at the restaurant -/
def white_bread : ℝ := 0.4

/-- Theorem stating that the number of loaves of wheat bread served is 0.5 -/
theorem wheat_bread_served : wheat_bread = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_wheat_bread_served_l695_69592


namespace NUMINAMATH_CALUDE_largest_integer_problem_l695_69571

theorem largest_integer_problem (a b c d : ℤ) : 
  a < b ∧ b < c ∧ c < d →  -- Four different integers
  (a + b + c + d) / 4 = 76 →  -- Average is 76
  a ≥ 37 →  -- Smallest integer is at least 37
  d ≤ 190 :=  -- Largest integer is at most 190
by sorry

end NUMINAMATH_CALUDE_largest_integer_problem_l695_69571


namespace NUMINAMATH_CALUDE_pool_draining_rate_l695_69587

/-- Given a rectangular pool with specified dimensions, capacity, and draining time,
    calculate the rate of water removal in cubic feet per minute. -/
theorem pool_draining_rate
  (width : ℝ) (length : ℝ) (depth : ℝ) (capacity : ℝ) (drain_time : ℝ)
  (h_width : width = 50)
  (h_length : length = 150)
  (h_depth : depth = 10)
  (h_capacity : capacity = 0.8)
  (h_drain_time : drain_time = 1000)
  : (width * length * depth * capacity) / drain_time = 60 := by
  sorry

end NUMINAMATH_CALUDE_pool_draining_rate_l695_69587


namespace NUMINAMATH_CALUDE_fraction_sum_l695_69577

theorem fraction_sum : (3 : ℚ) / 9 + (6 : ℚ) / 12 = (5 : ℚ) / 6 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_l695_69577


namespace NUMINAMATH_CALUDE_major_premise_incorrect_l695_69519

-- Define a differentiable function
variable (f : ℝ → ℝ)
variable (hf : Differentiable ℝ f)

-- Define a point x₀
variable (x₀ : ℝ)

-- State that f'(x₀) = 0
variable (h : deriv f x₀ = 0)

-- Define what it means for x₀ to be an extremum point
def is_extremum_point (f : ℝ → ℝ) (x₀ : ℝ) :=
  ∀ x, f x ≤ f x₀ ∨ ∀ x, f x ≥ f x₀

-- Theorem stating that the major premise is false
theorem major_premise_incorrect :
  ¬(∀ f : ℝ → ℝ, ∀ x₀ : ℝ, Differentiable ℝ f → deriv f x₀ = 0 → is_extremum_point f x₀) :=
sorry

end NUMINAMATH_CALUDE_major_premise_incorrect_l695_69519


namespace NUMINAMATH_CALUDE_town_budget_theorem_l695_69547

/-- Represents the town's budget allocation problem -/
def TownBudget (total : ℝ) (policing_fraction : ℝ) (education : ℝ) : Prop :=
  let policing := total * policing_fraction
  let remaining := total - policing - education
  remaining = 4

/-- The theorem statement for the town's budget allocation problem -/
theorem town_budget_theorem :
  TownBudget 32 0.5 12 := by
  sorry

end NUMINAMATH_CALUDE_town_budget_theorem_l695_69547


namespace NUMINAMATH_CALUDE_inequality_proof_l695_69513

theorem inequality_proof (a b c d : ℤ) 
  (h1 : a > b) (h2 : b > c) (h3 : c > d) (h4 : d > 0) 
  (h5 : a * d = b * c) : 
  (a - d)^2 ≥ 4*d + 8 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l695_69513


namespace NUMINAMATH_CALUDE_area_outside_rectangle_in_square_l695_69573

/-- The area of the region outside a rectangle contained within a square -/
theorem area_outside_rectangle_in_square (square_side : ℝ) (rect_length rect_width : ℝ) : 
  square_side = 8 ∧ rect_length = 4 ∧ rect_width = 2 →
  square_side^2 - rect_length * rect_width = 56 := by
sorry


end NUMINAMATH_CALUDE_area_outside_rectangle_in_square_l695_69573


namespace NUMINAMATH_CALUDE_sixty_percent_of_40_minus_four_fifths_of_25_l695_69554

theorem sixty_percent_of_40_minus_four_fifths_of_25 : (60 / 100 * 40) - (4 / 5 * 25) = 4 := by
  sorry

end NUMINAMATH_CALUDE_sixty_percent_of_40_minus_four_fifths_of_25_l695_69554


namespace NUMINAMATH_CALUDE_exists_F_for_P_l695_69567

/-- A ternary polynomial with real coefficients -/
def TernaryPolynomial := ℝ → ℝ → ℝ → ℝ

/-- The conditions that P must satisfy -/
def SatisfiesConditions (P : TernaryPolynomial) : Prop :=
  ∀ x y z : ℝ, 
    P x y z = P x y (x*y - z) ∧
    P x y z = P x (z*x - y) z ∧
    P x y z = P (y*z - x) y z

/-- The theorem statement -/
theorem exists_F_for_P (P : TernaryPolynomial) (h : SatisfiesConditions P) :
  ∃ F : ℝ → ℝ, ∀ x y z : ℝ, P x y z = F (x^2 + y^2 + z^2 - x*y*z) :=
sorry

end NUMINAMATH_CALUDE_exists_F_for_P_l695_69567


namespace NUMINAMATH_CALUDE_min_value_of_expression_l695_69548

theorem min_value_of_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  x / y + y / z + z / x + x / z ≥ 4 ∧
  (x / y + y / z + z / x + x / z = 4 ↔ x = y ∧ y = z) :=
sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l695_69548


namespace NUMINAMATH_CALUDE_range_of_a_l695_69523

theorem range_of_a (a : ℝ) : 
  (∅ : Set ℝ) ⊂ {x : ℝ | x^2 ≤ a} → a ∈ Set.Ici (0 : ℝ) := by sorry

end NUMINAMATH_CALUDE_range_of_a_l695_69523


namespace NUMINAMATH_CALUDE_waiter_customers_l695_69507

/-- Calculates the total number of customers given the number of tables and people per table -/
def total_customers (num_tables : ℕ) (women_per_table : ℕ) (men_per_table : ℕ) : ℕ :=
  num_tables * (women_per_table + men_per_table)

/-- Proves that given 7 tables with 7 women and 2 men each, the total number of customers is 63 -/
theorem waiter_customers : total_customers 7 7 2 = 63 := by
  sorry

end NUMINAMATH_CALUDE_waiter_customers_l695_69507


namespace NUMINAMATH_CALUDE_andy_problem_solving_l695_69580

theorem andy_problem_solving (last_problem : ℕ) (total_solved : ℕ) (h1 : last_problem = 125) (h2 : total_solved = 51) : 
  last_problem - total_solved + 1 = 75 := by
sorry

end NUMINAMATH_CALUDE_andy_problem_solving_l695_69580


namespace NUMINAMATH_CALUDE_circle_segment_perimeter_l695_69536

/-- Given a circle with radius 7 and a central angle of 270°, 
    the perimeter of the segment formed by this angle is equal to 14 + 10.5π. -/
theorem circle_segment_perimeter (r : ℝ) (angle : ℝ) : 
  r = 7 → angle = 270 * π / 180 → 
  2 * r + (angle / (2 * π)) * (2 * π * r) = 14 + 10.5 * π :=
by sorry

end NUMINAMATH_CALUDE_circle_segment_perimeter_l695_69536


namespace NUMINAMATH_CALUDE_remainder_r_15_minus_1_l695_69561

theorem remainder_r_15_minus_1 (r : ℤ) : (r^15 - 1) % (r + 1) = -2 := by
  sorry

end NUMINAMATH_CALUDE_remainder_r_15_minus_1_l695_69561


namespace NUMINAMATH_CALUDE_ordering_of_exponentials_l695_69597

theorem ordering_of_exponentials :
  let a : ℝ := 2^(2/3)
  let b : ℝ := 2^(2/5)
  let c : ℝ := 3^(2/3)
  b < a ∧ a < c := by sorry

end NUMINAMATH_CALUDE_ordering_of_exponentials_l695_69597


namespace NUMINAMATH_CALUDE_legs_walking_theorem_l695_69588

/-- The number of legs walking on the ground given the conditions of the problem -/
def legs_walking_on_ground (num_horses : ℕ) : ℕ :=
  let num_men := num_horses
  let num_riding := num_horses / 2
  let num_walking_men := num_men - num_riding
  let men_legs := num_walking_men * 2
  let horse_legs := num_horses * 4
  let walking_horse_legs := horse_legs / 2
  men_legs + walking_horse_legs

/-- Theorem stating that given 14 horses, the number of legs walking on the ground is 42 -/
theorem legs_walking_theorem : legs_walking_on_ground 14 = 42 := by
  sorry

end NUMINAMATH_CALUDE_legs_walking_theorem_l695_69588


namespace NUMINAMATH_CALUDE_acai_berry_juice_cost_per_litre_l695_69591

/-- The cost per litre of açaí berry juice given the following conditions:
  * The superfruit juice cocktail costs $1399.45 per litre to make.
  * The mixed fruit juice costs $262.85 per litre.
  * 37 litres of mixed fruit juice are used.
  * 24.666666666666668 litres of açaí berry juice are used.
-/
theorem acai_berry_juice_cost_per_litre :
  let cocktail_cost_per_litre : ℝ := 1399.45
  let mixed_fruit_juice_cost_per_litre : ℝ := 262.85
  let mixed_fruit_juice_volume : ℝ := 37
  let acai_berry_juice_volume : ℝ := 24.666666666666668
  let total_volume : ℝ := mixed_fruit_juice_volume + acai_berry_juice_volume
  let mixed_fruit_juice_total_cost : ℝ := mixed_fruit_juice_cost_per_litre * mixed_fruit_juice_volume
  let cocktail_total_cost : ℝ := cocktail_cost_per_litre * total_volume
  let acai_berry_juice_total_cost : ℝ := cocktail_total_cost - mixed_fruit_juice_total_cost
  let acai_berry_juice_cost_per_litre : ℝ := acai_berry_juice_total_cost / acai_berry_juice_volume
  acai_berry_juice_cost_per_litre = 3105.99 :=
by
  sorry


end NUMINAMATH_CALUDE_acai_berry_juice_cost_per_litre_l695_69591


namespace NUMINAMATH_CALUDE_largest_number_from_digits_l695_69501

def digits : List ℕ := [1, 7, 0]

def formNumber (d : List ℕ) : ℕ :=
  d.foldl (fun acc x => acc * 10 + x) 0

def isPermutation (l1 l2 : List ℕ) : Prop :=
  l1.length = l2.length ∧ ∀ x, x ∈ l1 ↔ x ∈ l2

theorem largest_number_from_digits : 
  ∀ p : List ℕ, isPermutation digits p → formNumber p ≤ 710 :=
sorry

end NUMINAMATH_CALUDE_largest_number_from_digits_l695_69501


namespace NUMINAMATH_CALUDE_initial_tax_rate_calculation_l695_69520

/-- Proves that given an annual income of $36,000, if lowering the tax rate to 32% 
    results in a savings of $5,040, then the initial tax rate was 46%. -/
theorem initial_tax_rate_calculation 
  (annual_income : ℝ) 
  (new_tax_rate : ℝ) 
  (savings : ℝ) 
  (h1 : annual_income = 36000)
  (h2 : new_tax_rate = 32)
  (h3 : savings = 5040) :
  ∃ (initial_tax_rate : ℝ), 
    initial_tax_rate = 46 ∧ 
    (initial_tax_rate / 100 * annual_income) - (new_tax_rate / 100 * annual_income) = savings :=
by sorry

end NUMINAMATH_CALUDE_initial_tax_rate_calculation_l695_69520


namespace NUMINAMATH_CALUDE_students_playing_both_sports_l695_69558

def total_students : ℕ := 460
def football_players : ℕ := 325
def cricket_players : ℕ := 175
def neither_players : ℕ := 50

theorem students_playing_both_sports : 
  total_students - neither_players = football_players + cricket_players - 90 := by
sorry

end NUMINAMATH_CALUDE_students_playing_both_sports_l695_69558


namespace NUMINAMATH_CALUDE_tara_quarters_l695_69537

theorem tara_quarters : ∃! q : ℕ, 
  8 < q ∧ q < 80 ∧ 
  q % 4 = 2 ∧
  q % 6 = 2 ∧
  q % 8 = 2 ∧
  q = 26 := by sorry

end NUMINAMATH_CALUDE_tara_quarters_l695_69537


namespace NUMINAMATH_CALUDE_coefficient_of_x_is_negative_five_l695_69538

-- Define the binomial coefficient
def binomial (n k : ℕ) : ℤ := sorry

-- Define the general term of the expansion
def generalTerm (r : ℕ) : ℤ := (-1)^r * binomial 5 r

-- Define the exponent of x in the general term
def exponent (r : ℕ) : ℚ := (5 - 3*r) / 2

theorem coefficient_of_x_is_negative_five :
  ∃ (r : ℕ), exponent r = 1 ∧ generalTerm r = -5 := by sorry

end NUMINAMATH_CALUDE_coefficient_of_x_is_negative_five_l695_69538


namespace NUMINAMATH_CALUDE_expression_simplification_l695_69518

theorem expression_simplification (y : ℝ) : 
  4 * y - 2 * y^2 + 3 - (8 - 4 * y + y^2) = 8 * y - 3 * y^2 - 5 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l695_69518


namespace NUMINAMATH_CALUDE_henri_drove_farther_l695_69540

/-- Proves that Henri drove 305 miles farther than Gervais -/
theorem henri_drove_farther (gervais_avg_daily : ℕ) (gervais_days : ℕ) (henri_total : ℕ) 
  (h1 : gervais_avg_daily = 315)
  (h2 : gervais_days = 3)
  (h3 : henri_total = 1250) :
  henri_total - (gervais_avg_daily * gervais_days) = 305 := by
  sorry

#check henri_drove_farther

end NUMINAMATH_CALUDE_henri_drove_farther_l695_69540


namespace NUMINAMATH_CALUDE_candy_bar_cost_l695_69560

/-- Given that Dan had $4 at the start and $3 left after buying a candy bar,
    prove that the candy bar cost $1. -/
theorem candy_bar_cost (initial_amount : ℕ) (remaining_amount : ℕ) :
  initial_amount = 4 →
  remaining_amount = 3 →
  initial_amount - remaining_amount = 1 :=
by sorry

end NUMINAMATH_CALUDE_candy_bar_cost_l695_69560


namespace NUMINAMATH_CALUDE_rate_of_change_kinetic_energy_l695_69557

/-- The rate of change of kinetic energy for a system with increasing mass -/
theorem rate_of_change_kinetic_energy
  (M : ℝ)  -- Initial mass of the system
  (v : ℝ)  -- Constant velocity of the system
  (ρ : ℝ)  -- Rate of mass increase
  (h1 : M > 0)  -- Mass is positive
  (h2 : v ≠ 0)  -- Velocity is non-zero
  (h3 : ρ > 0)  -- Rate of mass increase is positive
  : 
  ∃ (K : ℝ → ℝ), -- Kinetic energy as a function of time
    (∀ t, K t = (1/2) * (M + ρ * t) * v^2) ∧ 
    (∀ t, deriv K t = (1/2) * ρ * v^2) :=
sorry

end NUMINAMATH_CALUDE_rate_of_change_kinetic_energy_l695_69557


namespace NUMINAMATH_CALUDE_tan_two_alpha_l695_69539

theorem tan_two_alpha (α : Real) 
  (h : (Real.sin (Real.pi - α) + Real.sin (Real.pi / 2 - α)) / (Real.sin α - Real.cos α) = 1 / 2) : 
  Real.tan (2 * α) = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_tan_two_alpha_l695_69539


namespace NUMINAMATH_CALUDE_bobby_candy_chocolate_difference_l695_69599

/-- Given the number of candy pieces Bobby ate initially and additionally,
    as well as the number of chocolate pieces, prove that Bobby ate 58 more
    pieces of candy than chocolate. -/
theorem bobby_candy_chocolate_difference
  (initial_candy : ℕ)
  (additional_candy : ℕ)
  (chocolate : ℕ)
  (h1 : initial_candy = 38)
  (h2 : additional_candy = 36)
  (h3 : chocolate = 16) :
  initial_candy + additional_candy - chocolate = 58 := by
  sorry

end NUMINAMATH_CALUDE_bobby_candy_chocolate_difference_l695_69599


namespace NUMINAMATH_CALUDE_arithmetic_number_difference_l695_69508

/-- A 3-digit number with distinct digits forming an arithmetic sequence --/
def ArithmeticNumber (n : ℕ) : Prop :=
  ∃ a b c : ℕ,
    n = 100 * a + 10 * b + c ∧
    0 ≤ a ∧ a ≤ 9 ∧
    0 ≤ b ∧ b ≤ 9 ∧
    0 ≤ c ∧ c ≤ 9 ∧
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    b - a = c - b

theorem arithmetic_number_difference : 
  (∃ max min : ℕ, 
    ArithmeticNumber max ∧ 
    ArithmeticNumber min ∧
    (∀ n : ℕ, ArithmeticNumber n → min ≤ n ∧ n ≤ max) ∧
    max - min = 864) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_number_difference_l695_69508


namespace NUMINAMATH_CALUDE_set_equality_l695_69528

def U : Set Nat := {1, 2, 3, 4, 5, 6, 7}
def M : Set Nat := {3, 4, 5}
def N : Set Nat := {1, 3, 6}

theorem set_equality : (U \ M) ∩ (U \ N) = {2, 7} := by sorry

end NUMINAMATH_CALUDE_set_equality_l695_69528


namespace NUMINAMATH_CALUDE_equation_solution_difference_l695_69533

theorem equation_solution_difference : ∃ (s₁ s₂ : ℝ),
  (s₁^2 - 5*s₁ - 24) / (s₁ + 3) = 3*s₁ + 10 ∧
  (s₂^2 - 5*s₂ - 24) / (s₂ + 3) = 3*s₂ + 10 ∧
  s₁ ≠ s₂ ∧
  |s₁ - s₂| = 16 :=
by
  sorry

end NUMINAMATH_CALUDE_equation_solution_difference_l695_69533


namespace NUMINAMATH_CALUDE_quadratic_coefficient_of_equation_l695_69574

theorem quadratic_coefficient_of_equation (x : ℝ) : 
  (2*x + 1) * (3*x - 2) = x^2 + 2 → 
  ∃ a b c : ℝ, a = 5 ∧ a*x^2 + b*x + c = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_coefficient_of_equation_l695_69574


namespace NUMINAMATH_CALUDE_triangle_and_line_properties_l695_69534

-- Define the points
def A : ℝ × ℝ := (4, 1)
def B : ℝ × ℝ := (-1, -6)
def C : ℝ × ℝ := (-3, 2)

-- Define the triangular region D
def D : Set (ℝ × ℝ) := {(x, y) | 7*x - 5*y - 23 ≤ 0 ∧ x + 7*y - 11 ≤ 0 ∧ 4*x + y + 10 ≥ 0}

-- Define the line 4x - 3y - a = 0
def line (a : ℝ) : Set (ℝ × ℝ) := {(x, y) | 4*x - 3*y - a = 0}

-- Theorem statement
theorem triangle_and_line_properties :
  -- B and C are on opposite sides of the line 4x - 3y - a = 0
  ∀ a : ℝ, (4 * B.1 - 3 * B.2 - a) * (4 * C.1 - 3 * C.2 - a) < 0 →
  -- The system of inequalities correctly represents region D
  (∀ p : ℝ × ℝ, p ∈ D ↔ 7*p.1 - 5*p.2 - 23 ≤ 0 ∧ p.1 + 7*p.2 - 11 ≤ 0 ∧ 4*p.1 + p.2 + 10 ≥ 0) ∧
  -- The range of values for a is (-18, 14)
  (∀ a : ℝ, (4 * B.1 - 3 * B.2 - a) * (4 * C.1 - 3 * C.2 - a) < 0 ↔ -18 < a ∧ a < 14) :=
sorry

end NUMINAMATH_CALUDE_triangle_and_line_properties_l695_69534


namespace NUMINAMATH_CALUDE_age_ratio_problem_l695_69589

/-- Given that:
    1. X's current age is 45
    2. Three years ago, X's age was some multiple of Y's age
    3. Seven years from now, the sum of their ages will be 83 years
    Prove that the ratio of X's age to Y's age three years ago is 2:1 -/
theorem age_ratio_problem (x_current y_current : ℕ) : 
  x_current = 45 →
  ∃ k : ℕ, k > 0 ∧ (x_current - 3) = k * (y_current - 3) →
  x_current + y_current + 14 = 83 →
  (x_current - 3) / (y_current - 3) = 2 := by
  sorry

end NUMINAMATH_CALUDE_age_ratio_problem_l695_69589


namespace NUMINAMATH_CALUDE_pages_copied_for_fifteen_dollars_l695_69555

/-- The number of pages that can be copied given the cost per page and available money. -/
def pages_copied (cost_per_page : ℚ) (available_money : ℚ) : ℚ :=
  (available_money * 100) / cost_per_page

/-- Theorem: Given a cost of 5 cents per page and $15 available, 300 pages can be copied. -/
theorem pages_copied_for_fifteen_dollars :
  pages_copied (5 : ℚ) (15 : ℚ) = 300 := by
  sorry

end NUMINAMATH_CALUDE_pages_copied_for_fifteen_dollars_l695_69555


namespace NUMINAMATH_CALUDE_cricket_bat_price_l695_69544

/-- Calculates the final price of an item after two consecutive sales with given profit percentages -/
def finalPrice (initialCost : ℚ) (profit1 : ℚ) (profit2 : ℚ) : ℚ :=
  initialCost * (1 + profit1) * (1 + profit2)

/-- Theorem stating that a cricket bat initially costing $154, sold twice with 20% and 25% profit, results in a final price of $231 -/
theorem cricket_bat_price : 
  finalPrice 154 (20/100) (25/100) = 231 := by sorry

end NUMINAMATH_CALUDE_cricket_bat_price_l695_69544


namespace NUMINAMATH_CALUDE_distance_AB_l695_69524

def A : ℝ × ℝ := (-3, 2)
def B : ℝ × ℝ := (1, -1)

theorem distance_AB : Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2) = 5 := by
  sorry

end NUMINAMATH_CALUDE_distance_AB_l695_69524


namespace NUMINAMATH_CALUDE_fourth_degree_reduction_l695_69581

theorem fourth_degree_reduction (a b c d : ℝ) :
  ∃ (A B C k : ℝ), ∀ (t x : ℝ),
    (t^4 + a*t^3 + b*t^2 + c*t + d = 0) ↔
    (t = x + k ∧ x^4 = A*x^2 + B*x + C) :=
sorry

end NUMINAMATH_CALUDE_fourth_degree_reduction_l695_69581


namespace NUMINAMATH_CALUDE_frosting_per_cake_l695_69512

/-- The number of cans of frosting needed to frost a single cake -/
def cans_per_cake (cakes_per_day : ℕ) (days : ℕ) (cakes_eaten : ℕ) (total_cans : ℕ) : ℚ :=
  total_cans / (cakes_per_day * days - cakes_eaten)

/-- Theorem stating that given Sara's baking schedule and frosting needs, 
    it takes 2 cans of frosting to frost a single cake -/
theorem frosting_per_cake : 
  cans_per_cake 10 5 12 76 = 2 := by
  sorry

end NUMINAMATH_CALUDE_frosting_per_cake_l695_69512


namespace NUMINAMATH_CALUDE_specific_hill_ground_depth_l695_69579

/-- Represents a cone-shaped hill -/
structure ConeHill where
  height : ℝ
  aboveGroundVolumeFraction : ℝ

/-- Calculates the depth of the ground at the base of a cone-shaped hill -/
def groundDepth (hill : ConeHill) : ℝ :=
  hill.height * (1 - (hill.aboveGroundVolumeFraction ^ (1/3)))

/-- Theorem stating that for a specific cone-shaped hill, the ground depth is 355 feet -/
theorem specific_hill_ground_depth :
  let hill : ConeHill := { height := 5000, aboveGroundVolumeFraction := 1/5 }
  groundDepth hill = 355 := by
  sorry

end NUMINAMATH_CALUDE_specific_hill_ground_depth_l695_69579


namespace NUMINAMATH_CALUDE_journey_takes_eight_hours_l695_69583

/-- Represents the journey with three people A, B, and C --/
structure Journey where
  totalDistance : ℝ
  carSpeed : ℝ
  walkSpeed : ℝ
  t1 : ℝ  -- time A and C drive together
  t2 : ℝ  -- time A drives back
  t3 : ℝ  -- time A and B drive while C walks

/-- The conditions of the journey --/
def journeyConditions (j : Journey) : Prop :=
  j.totalDistance = 100 ∧
  j.carSpeed = 25 ∧
  j.walkSpeed = 5 ∧
  j.carSpeed * j.t1 - j.carSpeed * j.t2 + j.carSpeed * j.t3 = j.totalDistance ∧
  j.walkSpeed * j.t1 + j.walkSpeed * j.t2 + j.carSpeed * j.t3 = j.totalDistance ∧
  j.carSpeed * j.t1 + j.walkSpeed * j.t2 + j.walkSpeed * j.t3 = j.totalDistance

/-- The theorem stating that the journey takes 8 hours --/
theorem journey_takes_eight_hours (j : Journey) (h : journeyConditions j) :
  j.t1 + j.t2 + j.t3 = 8 := by
  sorry

end NUMINAMATH_CALUDE_journey_takes_eight_hours_l695_69583


namespace NUMINAMATH_CALUDE_max_value_x_y3_z4_l695_69505

theorem max_value_x_y3_z4 (x y z : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0) (h4 : x + y + z = 1) :
  x + y^3 + z^4 ≤ 1 ∧ ∃ (x' y' z' : ℝ), x' ≥ 0 ∧ y' ≥ 0 ∧ z' ≥ 0 ∧ x' + y' + z' = 1 ∧ x' + y'^3 + z'^4 = 1 :=
by sorry

end NUMINAMATH_CALUDE_max_value_x_y3_z4_l695_69505


namespace NUMINAMATH_CALUDE_roots_sum_of_powers_l695_69543

theorem roots_sum_of_powers (γ δ : ℝ) : 
  γ^2 - 3*γ - 2 = 0 → δ^2 - 3*δ - 2 = 0 → 3*γ^4 + 7*δ^3 = -135 := by
  sorry

end NUMINAMATH_CALUDE_roots_sum_of_powers_l695_69543


namespace NUMINAMATH_CALUDE_both_selected_probability_l695_69504

def prob_X : ℚ := 1/5
def prob_Y : ℚ := 2/7

theorem both_selected_probability : prob_X * prob_Y = 2/35 := by
  sorry

end NUMINAMATH_CALUDE_both_selected_probability_l695_69504


namespace NUMINAMATH_CALUDE_current_speed_calculation_l695_69565

-- Define the given conditions
def downstream_distance : ℝ := 96
def downstream_time : ℝ := 8
def upstream_distance : ℝ := 8
def upstream_time : ℝ := 2

-- Define the speed of the current
def current_speed : ℝ := 4

-- Theorem statement
theorem current_speed_calculation :
  let boat_speed := (downstream_distance / downstream_time + upstream_distance / upstream_time) / 2
  current_speed = boat_speed - upstream_distance / upstream_time :=
by sorry

end NUMINAMATH_CALUDE_current_speed_calculation_l695_69565


namespace NUMINAMATH_CALUDE_sheelas_monthly_income_l695_69510

/-- Given that Sheela's deposit is 20% of her monthly income, 
    prove that her monthly income is Rs. 25000 -/
theorem sheelas_monthly_income 
  (deposit : ℝ) 
  (deposit_percentage : ℝ) 
  (h1 : deposit = 5000)
  (h2 : deposit_percentage = 0.20)
  (h3 : deposit = deposit_percentage * sheelas_income) : 
  sheelas_income = 25000 :=
by
  sorry

#check sheelas_monthly_income

end NUMINAMATH_CALUDE_sheelas_monthly_income_l695_69510


namespace NUMINAMATH_CALUDE_negation_equivalence_l695_69570

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x^2 - x + 2 > 0) ↔ (∀ x : ℝ, x^2 - x + 2 ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l695_69570


namespace NUMINAMATH_CALUDE_figure_100_squares_l695_69546

def figure_squares (n : ℕ) : ℕ :=
  3 * n^2 + 3 * n + 1

theorem figure_100_squares :
  figure_squares 0 = 1 ∧
  figure_squares 1 = 7 ∧
  figure_squares 2 = 19 ∧
  figure_squares 3 = 37 →
  figure_squares 100 = 30301 := by
sorry

end NUMINAMATH_CALUDE_figure_100_squares_l695_69546


namespace NUMINAMATH_CALUDE_equation_2010_l695_69586

theorem equation_2010 (digits : Finset Nat) : digits = {2, 3, 5, 6, 7} →
  ∃ (a b c d : Nat), a ∈ digits ∧ b ∈ digits ∧ c ∈ digits ∧
  ((d = 67 ∧ (6 ∈ digits ∧ 7 ∈ digits)) ∨ d ∈ digits) ∧
  a * b * c * d = 2010 :=
by
  sorry

#check equation_2010

end NUMINAMATH_CALUDE_equation_2010_l695_69586


namespace NUMINAMATH_CALUDE_investment_value_after_eight_years_l695_69527

/-- Calculates the final value of an investment under simple interest -/
def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * (1 + rate * time)

/-- Proves that given the conditions, the investment value after 8 years is $660 -/
theorem investment_value_after_eight_years 
  (P : ℝ) -- Initial investment (principal)
  (h1 : simple_interest P 0.04 3 = 560) -- Value after 3 years
  : simple_interest P 0.04 8 = 660 := by
  sorry

#check investment_value_after_eight_years

end NUMINAMATH_CALUDE_investment_value_after_eight_years_l695_69527


namespace NUMINAMATH_CALUDE_morning_earnings_l695_69578

/-- Represents the types of vehicles William washes --/
inductive VehicleType
  | NormalCar
  | BigSUV
  | Minivan

/-- Represents a customer's order --/
structure Order where
  vehicles : List VehicleType
  multipleVehicles : Bool

def basePrice (v : VehicleType) : ℚ :=
  match v with
  | VehicleType.NormalCar => 15
  | VehicleType.BigSUV => 25
  | VehicleType.Minivan => 20

def washTime (v : VehicleType) : ℚ :=
  match v with
  | VehicleType.NormalCar => 1
  | VehicleType.BigSUV => 2
  | VehicleType.Minivan => 1.5

def applyDiscount (price : ℚ) : ℚ :=
  price * (1 - 0.1)

def calculateOrderPrice (o : Order) : ℚ :=
  let baseTotal := (o.vehicles.map basePrice).sum
  if o.multipleVehicles then applyDiscount baseTotal else baseTotal

def morningOrders : List Order :=
  [
    { vehicles := [VehicleType.NormalCar, VehicleType.NormalCar, VehicleType.NormalCar,
                   VehicleType.BigSUV, VehicleType.BigSUV, VehicleType.Minivan],
      multipleVehicles := false },
    { vehicles := [VehicleType.NormalCar, VehicleType.NormalCar, VehicleType.BigSUV],
      multipleVehicles := true }
  ]

theorem morning_earnings :
  (morningOrders.map calculateOrderPrice).sum = 164.5 := by sorry

end NUMINAMATH_CALUDE_morning_earnings_l695_69578


namespace NUMINAMATH_CALUDE_donation_analysis_l695_69568

/-- Represents the donation amounts and their frequencies --/
def donation_data : List (ℕ × ℕ) := [(5, 1), (10, 5), (15, 3), (20, 1)]

/-- Total number of students in the sample --/
def sample_size : ℕ := 10

/-- Total number of students in the school --/
def school_size : ℕ := 2200

/-- Calculates the mode of the donation data --/
def mode (data : List (ℕ × ℕ)) : ℕ := sorry

/-- Calculates the median of the donation data --/
def median (data : List (ℕ × ℕ)) : ℕ := sorry

/-- Calculates the average donation amount --/
def average (data : List (ℕ × ℕ)) : ℚ := sorry

/-- Estimates the total donation for the school --/
def estimate_total (avg : ℚ) (school_size : ℕ) : ℕ := sorry

theorem donation_analysis :
  mode donation_data = 10 ∧
  median donation_data = 10 ∧
  average donation_data = 12 ∧
  estimate_total (average donation_data) school_size = 26400 := by sorry

end NUMINAMATH_CALUDE_donation_analysis_l695_69568


namespace NUMINAMATH_CALUDE_minimum_trips_moscow_l695_69593

theorem minimum_trips_moscow (x y : ℕ) : 
  (31 * x + 32 * y = 5000) → 
  (∀ a b : ℕ, 31 * a + 32 * b = 5000 → x + y ≤ a + b) →
  x + y = 157 := by
sorry

end NUMINAMATH_CALUDE_minimum_trips_moscow_l695_69593


namespace NUMINAMATH_CALUDE_largest_of_eight_consecutive_integers_l695_69549

theorem largest_of_eight_consecutive_integers (n : ℕ) 
  (h1 : n > 0) 
  (h2 : n + (n+1) + (n+2) + (n+3) + (n+4) + (n+5) + (n+6) + (n+7) = 2024) : 
  n + 7 = 256 := by
  sorry

#check largest_of_eight_consecutive_integers

end NUMINAMATH_CALUDE_largest_of_eight_consecutive_integers_l695_69549


namespace NUMINAMATH_CALUDE_trigonometric_identities_l695_69559

theorem trigonometric_identities (α : Real) (h : Real.tan α = 3) :
  (4 * Real.sin α - Real.cos α) / (3 * Real.sin α + 5 * Real.cos α) = 11 / 14 ∧
  Real.sin α * Real.cos α = 3 / 10 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identities_l695_69559


namespace NUMINAMATH_CALUDE_f_monotone_increasing_f_extreme_values_l695_69556

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^3 - 2*x^2 - 4*x + 2

-- Theorem for monotonically increasing intervals
theorem f_monotone_increasing :
  (∀ x y, x < y ∧ x < -2/3 → f x < f y) ∧
  (∀ x y, 2 < x ∧ x < y → f x < f y) :=
sorry

-- Theorem for extreme values on [-1, 3]
theorem f_extreme_values :
  (∀ x, -1 ≤ x ∧ x ≤ 3 → -6 ≤ f x ∧ f x ≤ 94/27) ∧
  (∃ x, -1 ≤ x ∧ x ≤ 3 ∧ f x = -6) ∧
  (∃ x, -1 ≤ x ∧ x ≤ 3 ∧ f x = 94/27) :=
sorry

end NUMINAMATH_CALUDE_f_monotone_increasing_f_extreme_values_l695_69556


namespace NUMINAMATH_CALUDE_wallpaper_overlap_area_l695_69532

/-- Given the total area of wallpaper and areas covered by exactly two and three layers,
    calculate the actual area of the wall covered by overlapping wallpapers. -/
theorem wallpaper_overlap_area (total_area double_layer triple_layer : ℝ) 
    (h1 : total_area = 300)
    (h2 : double_layer = 30)
    (h3 : triple_layer = 45) :
    total_area - (2 * double_layer - double_layer) - (3 * triple_layer - triple_layer) = 180 := by
  sorry


end NUMINAMATH_CALUDE_wallpaper_overlap_area_l695_69532


namespace NUMINAMATH_CALUDE_sin_cos_identity_l695_69596

theorem sin_cos_identity (α : Real) (h : Real.sin α ^ 2 + Real.sin α = 1) :
  Real.cos α ^ 4 + Real.cos α ^ 2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_identity_l695_69596


namespace NUMINAMATH_CALUDE_B_spend_percent_is_85_percent_l695_69564

def total_salary : ℝ := 7000
def A_salary : ℝ := 5250
def A_spend_percent : ℝ := 0.95

def B_salary : ℝ := total_salary - A_salary
def A_savings : ℝ := A_salary * (1 - A_spend_percent)

theorem B_spend_percent_is_85_percent :
  ∃ (B_spend_percent : ℝ),
    B_spend_percent = 0.85 ∧
    A_savings = B_salary * (1 - B_spend_percent) := by
  sorry

end NUMINAMATH_CALUDE_B_spend_percent_is_85_percent_l695_69564


namespace NUMINAMATH_CALUDE_round_trip_ticket_percentage_l695_69569

/-- Given that 25% of all passengers held round-trip tickets and took their cars aboard,
    and 60% of passengers with round-trip tickets did not take their cars aboard,
    prove that 62.5% of all passengers held round-trip tickets. -/
theorem round_trip_ticket_percentage
  (total_passengers : ℝ)
  (h1 : total_passengers > 0)
  (h2 : (25 : ℝ) / 100 * total_passengers = (40 : ℝ) / 100 * ((100 : ℝ) / 100 * total_passengers)) :
  (62.5 : ℝ) / 100 * total_passengers = (100 : ℝ) / 100 * total_passengers :=
sorry

end NUMINAMATH_CALUDE_round_trip_ticket_percentage_l695_69569


namespace NUMINAMATH_CALUDE_parabola_line_intersection_l695_69572

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola y² = 2px -/
structure Parabola where
  p : ℝ
  hp : p > 0

/-- Represents a line with slope m -/
structure Line where
  m : ℝ

theorem parabola_line_intersection
  (para : Parabola)
  (line : Line)
  (A B : Point)
  (h_line_slope : line.m = 2 * Real.sqrt 2)
  (h_on_parabola_A : A.y ^ 2 = 2 * para.p * A.x)
  (h_on_parabola_B : B.y ^ 2 = 2 * para.p * B.x)
  (h_on_line_A : A.y = line.m * (A.x - para.p / 2))
  (h_on_line_B : B.y = line.m * (B.x - para.p / 2))
  (h_x_order : A.x < B.x)
  (h_distance : Real.sqrt ((B.x - A.x) ^ 2 + (B.y - A.y) ^ 2) = 9) :
  (∃ (C : Point),
    C.y ^ 2 = 8 * C.x ∧
    (C.x = A.x + 0 * (B.x - A.x) ∧ C.y = A.y + 0 * (B.y - A.y) ∨
     C.x = A.x + 2 * (B.x - A.x) ∧ C.y = A.y + 2 * (B.y - A.y))) :=
by sorry

end NUMINAMATH_CALUDE_parabola_line_intersection_l695_69572


namespace NUMINAMATH_CALUDE_operation_result_l695_69582

def operation (n : ℕ) : ℕ := 2 * n + 1

def iterate_operation (n : ℕ) (k : ℕ) : ℕ :=
  match k with
  | 0 => n
  | k + 1 => operation (iterate_operation n k)

theorem operation_result (x : ℕ) :
  ¬(∃ (y : ℕ), iterate_operation x 100 = 1980 * y) ∧
  (∃ (x : ℕ), ∃ (y : ℕ), iterate_operation x 100 = 1981 * y) := by
  sorry


end NUMINAMATH_CALUDE_operation_result_l695_69582


namespace NUMINAMATH_CALUDE_sequence_correct_l695_69500

def sequence_formula (n : ℕ) : ℤ := (-1)^(n+1) * 2^(n-1)

theorem sequence_correct : ∀ n : ℕ, n ≥ 1 ∧ n ≤ 6 →
  match n with
  | 1 => sequence_formula n = 1
  | 2 => sequence_formula n = -2
  | 3 => sequence_formula n = 4
  | 4 => sequence_formula n = -8
  | 5 => sequence_formula n = 16
  | 6 => sequence_formula n = -32
  | _ => True
  :=
by
  sorry

end NUMINAMATH_CALUDE_sequence_correct_l695_69500


namespace NUMINAMATH_CALUDE_line_passes_through_K_min_distance_AC_dot_product_range_l695_69594

-- Define the circle M
def circle_M (x y : ℝ) : Prop := (x - 4)^2 + (y - 5)^2 = 12

-- Define the line l
def line_l (m x y : ℝ) : Prop := m*x - y - 2*m + 3 = 0

-- Define the point K
def point_K : ℝ × ℝ := (2, 3)

-- Define the intersection points A and C
def intersection_points (m : ℝ) : Set (ℝ × ℝ) :=
  {p | ∃ (x y : ℝ), p = (x, y) ∧ circle_M x y ∧ line_l m x y}

-- Theorem 1: Line l passes through point K for all m
theorem line_passes_through_K (m : ℝ) : line_l m (point_K.1) (point_K.2) :=
sorry

-- Theorem 2: Minimum distance between intersection points is 4
theorem min_distance_AC :
  ∃ (m : ℝ), ∀ (A C : ℝ × ℝ), A ∈ intersection_points m → C ∈ intersection_points m →
  A ≠ C → ‖A - C‖ ≥ 4 :=
sorry

-- Theorem 3: Range of dot product MA · MC
theorem dot_product_range (M : ℝ × ℝ) (m : ℝ) :
  M = (4, 5) →
  ∀ (A C : ℝ × ℝ), A ∈ intersection_points m → C ∈ intersection_points m →
  -12 ≤ (A - M) • (C - M) ∧ (A - M) • (C - M) ≤ 4 :=
sorry

end NUMINAMATH_CALUDE_line_passes_through_K_min_distance_AC_dot_product_range_l695_69594


namespace NUMINAMATH_CALUDE_plane_equation_correct_l695_69515

/-- The equation of a plane given the foot of the perpendicular from the origin -/
def plane_equation (foot : ℝ × ℝ × ℝ) : ℤ × ℤ × ℤ × ℤ := sorry

/-- Check if the given coefficients satisfy the required conditions -/
def valid_coefficients (coeffs : ℤ × ℤ × ℤ × ℤ) : Prop :=
  let (A, B, C, D) := coeffs
  A > 0 ∧ Nat.gcd (Nat.gcd (Int.natAbs A) (Int.natAbs B)) (Nat.gcd (Int.natAbs C) (Int.natAbs D)) = 1

theorem plane_equation_correct (foot : ℝ × ℝ × ℝ) :
  foot = (10, -2, 1) →
  plane_equation foot = (10, -2, 1, -105) ∧
  valid_coefficients (plane_equation foot) := by
  sorry

end NUMINAMATH_CALUDE_plane_equation_correct_l695_69515


namespace NUMINAMATH_CALUDE_max_sum_of_factors_l695_69531

theorem max_sum_of_factors (a b : ℕ) (h : a * b = 48) : 
  ∃ (x y : ℕ), x * y = 48 ∧ x + y ≤ a + b ∧ x + y = 49 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_of_factors_l695_69531


namespace NUMINAMATH_CALUDE_original_selling_price_l695_69550

/-- Proves that the original selling price is $24000 given the conditions --/
theorem original_selling_price (cost_price : ℝ) (discount_rate : ℝ) (profit_rate : ℝ) 
  (h1 : cost_price = 20000)
  (h2 : discount_rate = 0.1)
  (h3 : profit_rate = 0.08) : 
  ∃ (selling_price : ℝ), 
    selling_price = 24000 ∧ 
    (1 - discount_rate) * selling_price = cost_price * (1 + profit_rate) :=
by
  sorry

end NUMINAMATH_CALUDE_original_selling_price_l695_69550


namespace NUMINAMATH_CALUDE_axis_of_symmetry_is_neg_two_l695_69563

/-- A quadratic function with given coordinate values -/
structure QuadraticFunction where
  f : ℝ → ℝ
  coords : List (ℝ × ℝ)
  is_quadratic : ∃ a b c : ℝ, ∀ x, f x = a * x^2 + b * x + c

/-- The axis of symmetry of a quadratic function -/
def axis_of_symmetry (qf : QuadraticFunction) : ℝ := sorry

/-- The given quadratic function from the problem -/
def given_function : QuadraticFunction where
  f := sorry
  coords := [(-3, -3), (-2, -2), (-1, -3), (0, -6), (1, -11)]
  is_quadratic := sorry

/-- Theorem stating that the axis of symmetry of the given function is -2 -/
theorem axis_of_symmetry_is_neg_two :
  axis_of_symmetry given_function = -2 := by sorry

end NUMINAMATH_CALUDE_axis_of_symmetry_is_neg_two_l695_69563


namespace NUMINAMATH_CALUDE_pascal_contest_participants_l695_69506

theorem pascal_contest_participants (male_count : ℕ) (ratio_male : ℕ) (ratio_female : ℕ) : 
  male_count = 21 → ratio_male = 3 → ratio_female = 7 → 
  male_count + (male_count * ratio_female / ratio_male) = 70 := by
sorry

end NUMINAMATH_CALUDE_pascal_contest_participants_l695_69506


namespace NUMINAMATH_CALUDE_wooden_strip_sawing_time_l695_69566

theorem wooden_strip_sawing_time 
  (initial_length : ℝ) 
  (initial_sections : ℕ) 
  (initial_time : ℝ) 
  (final_sections : ℕ) 
  (h1 : initial_length = 12) 
  (h2 : initial_sections = 4) 
  (h3 : initial_time = 12) 
  (h4 : final_sections = 8) : 
  (initial_time / (initial_sections - 1)) * (final_sections - 1) = 28 := by
sorry

end NUMINAMATH_CALUDE_wooden_strip_sawing_time_l695_69566


namespace NUMINAMATH_CALUDE_min_product_sum_l695_69552

/-- Triangle ABC with side lengths a, b, c and height h from A to BC -/
structure Triangle :=
  (a b c h : ℝ)
  (positive_a : 0 < a)
  (positive_b : 0 < b)
  (positive_c : 0 < c)
  (positive_h : 0 < h)

/-- The problem statement -/
theorem min_product_sum (t : Triangle) (h1 : t.c = 10) (h2 : t.h = 3) :
  let min_product := Real.sqrt ((t.c^2 * t.h^2) / 4)
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a * b = min_product ∧ a + b = 4 * Real.sqrt 10 :=
sorry

end NUMINAMATH_CALUDE_min_product_sum_l695_69552


namespace NUMINAMATH_CALUDE_fraction_1800_1809_equals_4_13_l695_69595

/-- The number of states that joined the union during 1800-1809. -/
def states_1800_1809 : ℕ := 8

/-- The total number of states in Jennifer's collection. -/
def total_states : ℕ := 26

/-- The fraction of states that joined during 1800-1809 out of the first 26 states. -/
def fraction_1800_1809 : ℚ := states_1800_1809 / total_states

theorem fraction_1800_1809_equals_4_13 : fraction_1800_1809 = 4 / 13 := by sorry

end NUMINAMATH_CALUDE_fraction_1800_1809_equals_4_13_l695_69595


namespace NUMINAMATH_CALUDE_linglings_spending_l695_69521

theorem linglings_spending (x : ℝ) 
  (h1 : 720 = (1 - 1/3) * ((1 - 2/5) * x + 240)) : 
  ∃ (spent : ℝ), spent = 2/5 * x ∧ 
  720 = (1 - 1/3) * ((x - spent) + 240) := by
  sorry

end NUMINAMATH_CALUDE_linglings_spending_l695_69521


namespace NUMINAMATH_CALUDE_marble_collection_problem_l695_69598

/-- Represents the number of marbles collected by a person -/
structure MarbleCount where
  red : ℕ
  blue : ℕ

/-- The marble collection problem -/
theorem marble_collection_problem 
  (mary jenny anie tom : MarbleCount)
  (h1 : mary.red = 2 * jenny.red)
  (h2 : mary.blue = anie.blue / 2)
  (h3 : anie.red = mary.red + 20)
  (h4 : anie.blue = 2 * jenny.blue)
  (h5 : tom.red = anie.red + 10)
  (h6 : tom.blue = mary.blue)
  (h7 : jenny.red = 30)
  (h8 : jenny.blue = 25) :
  mary.blue + jenny.blue + anie.blue + tom.blue = 125 := by
  sorry

end NUMINAMATH_CALUDE_marble_collection_problem_l695_69598


namespace NUMINAMATH_CALUDE_E27D6_divisibility_l695_69575

/-- A number in the form E27D6 where E and D are single digits -/
def E27D6 (E D : ℕ) : ℕ := E * 10000 + 27000 + D * 10 + 6

/-- Predicate to check if a number is a single digit -/
def is_single_digit (n : ℕ) : Prop := n < 10

theorem E27D6_divisibility (E D : ℕ) :
  is_single_digit E →
  is_single_digit D →
  E27D6 E D % 8 = 0 →
  ∃ (sum : ℕ), sum = D + E ∧ 1 ≤ sum ∧ sum ≤ 10 :=
sorry

end NUMINAMATH_CALUDE_E27D6_divisibility_l695_69575


namespace NUMINAMATH_CALUDE_cards_per_page_l695_69509

theorem cards_per_page (new_cards old_cards pages : ℕ) 
  (h1 : new_cards = 8)
  (h2 : old_cards = 10)
  (h3 : pages = 6) :
  (new_cards + old_cards) / pages = 3 := by
  sorry

end NUMINAMATH_CALUDE_cards_per_page_l695_69509


namespace NUMINAMATH_CALUDE_employed_females_percentage_is_16_percent_l695_69562

/-- Represents an age group in Town X -/
inductive AgeGroup
  | Young    -- 18-34
  | Middle   -- 35-54
  | Senior   -- 55+

/-- The percentage of employed population in each age group -/
def employed_percentage : ℝ := 64

/-- The percentage of employed males in each age group -/
def employed_males_percentage : ℝ := 48

/-- The percentage of employed females in each age group -/
def employed_females_percentage : ℝ := employed_percentage - employed_males_percentage

theorem employed_females_percentage_is_16_percent :
  employed_females_percentage = 16 := by sorry

end NUMINAMATH_CALUDE_employed_females_percentage_is_16_percent_l695_69562


namespace NUMINAMATH_CALUDE_parabola_directrix_l695_69590

/-- The parabola equation -/
def parabola (x y : ℝ) : Prop := y = 4 * x^2 + 4

/-- The directrix equation -/
def directrix (y : ℝ) : Prop := y = 63 / 16

/-- Theorem: The directrix of the parabola y = 4x^2 + 4 is y = 63/16 -/
theorem parabola_directrix : ∀ x y : ℝ, parabola x y → ∃ d : ℝ, directrix d ∧ 
  (∀ p q : ℝ × ℝ, p.2 = 4 * p.1^2 + 4 → 
    (p.1 - x)^2 + (p.2 - y)^2 = (p.2 - d)^2) :=
sorry

end NUMINAMATH_CALUDE_parabola_directrix_l695_69590


namespace NUMINAMATH_CALUDE_old_phone_plan_cost_l695_69535

theorem old_phone_plan_cost 
  (new_plan_cost : ℝ) 
  (price_increase_percentage : ℝ) 
  (h1 : new_plan_cost = 195) 
  (h2 : price_increase_percentage = 0.30) : 
  new_plan_cost / (1 + price_increase_percentage) = 150 := by
sorry

end NUMINAMATH_CALUDE_old_phone_plan_cost_l695_69535


namespace NUMINAMATH_CALUDE_even_power_minus_one_factorization_l695_69542

theorem even_power_minus_one_factorization (n : ℕ) (h1 : Even n) (h2 : n > 4) :
  ∃ (a b c : ℕ), (a > 1 ∧ b > 1 ∧ c > 1) ∧ (2^n - 1 = a * b * c) :=
sorry

end NUMINAMATH_CALUDE_even_power_minus_one_factorization_l695_69542


namespace NUMINAMATH_CALUDE_binomial_10_choose_3_l695_69514

theorem binomial_10_choose_3 : Nat.choose 10 3 = 120 := by
  sorry

end NUMINAMATH_CALUDE_binomial_10_choose_3_l695_69514


namespace NUMINAMATH_CALUDE_solution_proof_l695_69516

/-- Custom operation for 2x2 matrices -/
def matrix_op (a b c d : ℝ) : ℝ := a * d - b * c

/-- Theorem stating the solution to the given equation -/
theorem solution_proof :
  ∃ x : ℝ, matrix_op (x + 1) (x + 2) (x - 3) (x - 1) = 27 ∧ x = 22 := by
  sorry

end NUMINAMATH_CALUDE_solution_proof_l695_69516


namespace NUMINAMATH_CALUDE_triangle_right_angled_from_arithmetic_progression_l695_69503

/-- Given a triangle with side lengths a, b, c, and incircle diameter 2r
    forming an arithmetic progression, prove that the triangle is right-angled. -/
theorem triangle_right_angled_from_arithmetic_progression 
  (a b c r : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0 ∧ r > 0)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_arithmetic : ∃ (d : ℝ), b = a + d ∧ c = b + d ∧ 2*r = c + d) :
  ∃ (A B C : ℝ), A + B + C = π ∧ max A B = π/2 ∧ max B C = π/2 ∧ max C A = π/2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_right_angled_from_arithmetic_progression_l695_69503


namespace NUMINAMATH_CALUDE_sum_x_y_equals_five_l695_69522

theorem sum_x_y_equals_five (x y : ℝ) 
  (eq1 : x + 3*y = 12) 
  (eq2 : 3*x + y = 8) : 
  x + y = 5 := by
sorry

end NUMINAMATH_CALUDE_sum_x_y_equals_five_l695_69522
