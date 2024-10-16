import Mathlib

namespace NUMINAMATH_CALUDE_crow_count_proof_l347_34720

/-- The number of crows in the first group -/
def first_group_count : ℕ := 3

/-- The number of worms eaten by the first group in one hour -/
def first_group_worms : ℕ := 30

/-- The number of crows in the second group -/
def second_group_count : ℕ := 5

/-- The number of worms eaten by the second group in two hours -/
def second_group_worms : ℕ := 100

/-- The number of hours the second group took to eat their worms -/
def second_group_hours : ℕ := 2

theorem crow_count_proof : first_group_count = 3 := by
  sorry

end NUMINAMATH_CALUDE_crow_count_proof_l347_34720


namespace NUMINAMATH_CALUDE_opposite_of_sqrt7_minus_3_l347_34735

theorem opposite_of_sqrt7_minus_3 : 
  -(Real.sqrt 7 - 3) = 3 - Real.sqrt 7 := by sorry

end NUMINAMATH_CALUDE_opposite_of_sqrt7_minus_3_l347_34735


namespace NUMINAMATH_CALUDE_simple_interest_rate_example_l347_34756

/-- Calculate the simple interest rate given principal, amount, and time -/
def simple_interest_rate (principal amount : ℚ) (time : ℕ) : ℚ :=
  let simple_interest := amount - principal
  (simple_interest * 100) / (principal * time)

/-- Theorem: The simple interest rate for the given conditions is approximately 9.23% -/
theorem simple_interest_rate_example :
  let principal := 650
  let amount := 950
  let time := 5
  let rate := simple_interest_rate principal amount time
  (rate ≥ 9.23) ∧ (rate < 9.24) :=
by sorry

end NUMINAMATH_CALUDE_simple_interest_rate_example_l347_34756


namespace NUMINAMATH_CALUDE_number_of_divisors_180_l347_34771

theorem number_of_divisors_180 : Nat.card {d : ℕ | d > 0 ∧ 180 % d = 0} = 18 := by
  sorry

end NUMINAMATH_CALUDE_number_of_divisors_180_l347_34771


namespace NUMINAMATH_CALUDE_smallest_valid_number_l347_34702

def is_valid (n : ℕ) : Prop :=
  11 ∣ n ∧ ∀ k : ℕ, 2 ≤ k → k ≤ 8 → n % k = 3

theorem smallest_valid_number : 
  is_valid 5043 ∧ ∀ m : ℕ, m < 5043 → ¬(is_valid m) :=
sorry

end NUMINAMATH_CALUDE_smallest_valid_number_l347_34702


namespace NUMINAMATH_CALUDE_value_of_b_l347_34708

theorem value_of_b (a b c : ℤ) 
  (eq1 : a + 5 = b) 
  (eq2 : 5 + b = c) 
  (eq3 : b + c = a) : 
  b = -10 := by
  sorry

end NUMINAMATH_CALUDE_value_of_b_l347_34708


namespace NUMINAMATH_CALUDE_total_maggots_served_l347_34796

def feeding_1 : ℕ := 10
def feeding_2 : ℕ := 15
def feeding_3 : ℕ := 2 * feeding_2
def feeding_4 : ℕ := feeding_3 - 5

theorem total_maggots_served :
  feeding_1 + feeding_2 + feeding_3 + feeding_4 = 80 := by
  sorry

end NUMINAMATH_CALUDE_total_maggots_served_l347_34796


namespace NUMINAMATH_CALUDE_tan_negative_seven_pi_sixths_l347_34799

theorem tan_negative_seven_pi_sixths : 
  Real.tan (-7 * π / 6) = -1 / Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_negative_seven_pi_sixths_l347_34799


namespace NUMINAMATH_CALUDE_inequality_solution_set_l347_34727

theorem inequality_solution_set (x : ℝ) : 
  (x^2 - 4) * (x - 6)^2 ≤ 0 ↔ -2 ≤ x ∧ x ≤ 2 ∨ x = 6 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l347_34727


namespace NUMINAMATH_CALUDE_cement_bought_l347_34791

/-- The amount of cement bought, given the total amount, original amount, and son's contribution -/
theorem cement_bought (total : ℕ) (original : ℕ) (son_contribution : ℕ) 
  (h1 : total = 450)
  (h2 : original = 98)
  (h3 : son_contribution = 137) :
  total - (original + son_contribution) = 215 := by
  sorry

end NUMINAMATH_CALUDE_cement_bought_l347_34791


namespace NUMINAMATH_CALUDE_only_D_is_certain_l347_34784

structure Event where
  description : String
  is_certain : Bool

def event_A : Event := { description := "It will definitely rain on a cloudy day", is_certain := false }
def event_B : Event := { description := "When tossing a fair coin, the head side faces up", is_certain := false }
def event_C : Event := { description := "A boy's height is definitely taller than a girl's", is_certain := false }
def event_D : Event := { description := "When oil is dropped into water, the oil will float on the surface of the water", is_certain := true }

def events : List Event := [event_A, event_B, event_C, event_D]

theorem only_D_is_certain : ∃! e : Event, e ∈ events ∧ e.is_certain := by sorry

end NUMINAMATH_CALUDE_only_D_is_certain_l347_34784


namespace NUMINAMATH_CALUDE_fraction_operation_result_l347_34792

theorem fraction_operation_result (x : ℝ) : 
  x = 2.5 → ((x / (1 / 2)) * x) / ((x * (1 / 2)) / x) = 25 := by
  sorry

end NUMINAMATH_CALUDE_fraction_operation_result_l347_34792


namespace NUMINAMATH_CALUDE_fraction_always_defined_l347_34744

theorem fraction_always_defined (x : ℝ) : (x^2 + 2 ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_fraction_always_defined_l347_34744


namespace NUMINAMATH_CALUDE_platform_length_l347_34706

/-- Calculates the length of a platform given train specifications -/
theorem platform_length (train_length : ℝ) (platform_crossing_time : ℝ) (pole_crossing_time : ℝ) :
  train_length = 300 →
  platform_crossing_time = 42 →
  pole_crossing_time = 18 →
  ∃ platform_length : ℝ, platform_length = 400 := by
  sorry

end NUMINAMATH_CALUDE_platform_length_l347_34706


namespace NUMINAMATH_CALUDE_sale_price_for_50_percent_profit_l347_34715

/-- Represents the cost and pricing of an article -/
structure Article where
  cost : ℝ
  profit_price : ℝ
  loss_price : ℝ

/-- The conditions of the problem -/
def problem_conditions (a : Article) : Prop :=
  a.profit_price - a.cost = a.cost - a.loss_price ∧
  a.profit_price = 892 ∧
  1005 = 1.5 * a.cost

/-- The theorem to be proved -/
theorem sale_price_for_50_percent_profit (a : Article) 
  (h : problem_conditions a) : 
  1.5 * a.cost = 1005 := by
  sorry

#check sale_price_for_50_percent_profit

end NUMINAMATH_CALUDE_sale_price_for_50_percent_profit_l347_34715


namespace NUMINAMATH_CALUDE_max_xy_value_l347_34710

theorem max_xy_value (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : 2 * x + 3 * y = 4) :
  ∃ (M : ℝ), M = 2/3 ∧ xy ≤ M ∧ ∃ (x₀ y₀ : ℝ), x₀ * y₀ = M ∧ 2 * x₀ + 3 * y₀ = 4 :=
sorry

end NUMINAMATH_CALUDE_max_xy_value_l347_34710


namespace NUMINAMATH_CALUDE_solution_range_l347_34797

theorem solution_range (k : ℝ) : 
  (∃ x : ℝ, x + k = 2 * x - 1 ∧ x < 0) → k < -1 := by
  sorry

end NUMINAMATH_CALUDE_solution_range_l347_34797


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l347_34789

/-- An isosceles triangle with two sides measuring 4 and 7 has a perimeter of either 15 or 18. -/
theorem isosceles_triangle_perimeter : ∀ a b c : ℝ,
  a > 0 → b > 0 → c > 0 →
  (a = 4 ∧ b = 4 ∧ c = 7) ∨ (a = 7 ∧ b = 7 ∧ c = 4) →
  a + b > c → b + c > a → c + a > b →
  a + b + c = 15 ∨ a + b + c = 18 := by
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l347_34789


namespace NUMINAMATH_CALUDE_triangle_trig_inequality_l347_34751

theorem triangle_trig_inequality (A B C : ℝ) (h : A + B + C = π) :
  Real.sin A + Real.cos B * Real.cos C ≤ (1 + Real.sqrt 5) / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_trig_inequality_l347_34751


namespace NUMINAMATH_CALUDE_fraction_multiplication_l347_34777

theorem fraction_multiplication : (1 : ℚ) / 3 * (3 : ℚ) / 5 * (5 : ℚ) / 6 = (1 : ℚ) / 6 := by
  sorry

end NUMINAMATH_CALUDE_fraction_multiplication_l347_34777


namespace NUMINAMATH_CALUDE_quadratic_property_contradiction_l347_34747

/-- Represents a quadratic function of the form y = ax² + bx - 6 --/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  h : a ≠ 0

/-- Properties of the quadratic function --/
def QuadraticProperties (f : QuadraticFunction) : Prop :=
  ∃ (x_sym : ℝ) (y_min : ℝ),
    -- Axis of symmetry is x = 1
    x_sym = 1 ∧
    -- Minimum value is -8
    y_min = -8 ∧
    -- x = 3 is a root
    f.a * 3^2 + f.b * 3 - 6 = 0

/-- The main theorem to prove --/
theorem quadratic_property_contradiction (f : QuadraticFunction) 
  (h : QuadraticProperties f) : 
  f.a * 3^2 + f.b * 3 - 6 ≠ -6 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_property_contradiction_l347_34747


namespace NUMINAMATH_CALUDE_two_true_propositions_l347_34717

theorem two_true_propositions (a b c : ℝ) : 
  (∃! n : Nat, n = 2 ∧ 
    (((a > b → a * c^2 > b * c^2) ∧ 
      (a * c^2 > b * c^2 → a > b) ∧ 
      (a ≤ b → a * c^2 ≤ b * c^2) ∧ 
      (a * c^2 ≤ b * c^2 → a ≤ b)) → n = 4) ∧
    ((¬(a > b → a * c^2 > b * c^2) ∧ 
      (a * c^2 > b * c^2 → a > b) ∧ 
      (a ≤ b → a * c^2 ≤ b * c^2) ∧ 
      ¬(a * c^2 ≤ b * c^2 → a ≤ b)) → n = 2) ∧
    ((¬(a > b → a * c^2 > b * c^2) ∧ 
      ¬(a * c^2 > b * c^2 → a > b) ∧ 
      ¬(a ≤ b → a * c^2 ≤ b * c^2) ∧ 
      ¬(a * c^2 ≤ b * c^2 → a ≤ b)) → n = 0)) :=
sorry

end NUMINAMATH_CALUDE_two_true_propositions_l347_34717


namespace NUMINAMATH_CALUDE_sqrt_sum_equals_eight_l347_34724

theorem sqrt_sum_equals_eight : 
  Real.sqrt (16 - 8 * Real.sqrt 3) + Real.sqrt (16 + 8 * Real.sqrt 3) = 8 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_equals_eight_l347_34724


namespace NUMINAMATH_CALUDE_fishing_moratorium_purpose_l347_34765

/-- Represents a fishing moratorium period -/
structure FishingMoratorium where
  start_date : Nat
  end_date : Nat
  regulations : String

/-- Represents the purpose of a fishing moratorium -/
inductive MoratoriumPurpose
  | ProtectEndangeredSpecies
  | ReducePollution
  | ProtectFishermen
  | SustainableUse

/-- The main purpose of the fishing moratorium -/
def main_purpose (moratorium : FishingMoratorium) : MoratoriumPurpose := sorry

/-- Theorem stating the main purpose of the fishing moratorium -/
theorem fishing_moratorium_purpose 
  (moratorium : FishingMoratorium)
  (h1 : moratorium.start_date = 20150516)
  (h2 : moratorium.end_date = 20150801)
  (h3 : moratorium.regulations = "Ministry of Agriculture regulations") :
  main_purpose moratorium = MoratoriumPurpose.SustainableUse := by sorry

end NUMINAMATH_CALUDE_fishing_moratorium_purpose_l347_34765


namespace NUMINAMATH_CALUDE_gcd_14568_78452_l347_34729

theorem gcd_14568_78452 : Int.gcd 14568 78452 = 4 := by
  sorry

end NUMINAMATH_CALUDE_gcd_14568_78452_l347_34729


namespace NUMINAMATH_CALUDE_side_length_relationship_l347_34782

/-- Side length of an inscribed regular n-gon in a circle with radius r -/
def a (n : ℕ) (r : ℝ) : ℝ := sorry

/-- Side length of a circumscribed regular n-gon around a circle with radius r -/
def A (n : ℕ) (r : ℝ) : ℝ := sorry

/-- Theorem stating the relationship between side lengths of regular polygons -/
theorem side_length_relationship (n : ℕ) (r : ℝ) (h : 0 < r) :
  1 / A (2 * n) r = 1 / A n r + 1 / a n r := by sorry

end NUMINAMATH_CALUDE_side_length_relationship_l347_34782


namespace NUMINAMATH_CALUDE_pause_point_correct_l347_34704

/-- Represents the duration of a movie in minutes -/
def MovieLength : ℕ := 60

/-- Represents the remaining time to watch in minutes -/
def RemainingTime : ℕ := 30

/-- Calculates the point at which the movie was paused -/
def PausePoint : ℕ := MovieLength - RemainingTime

theorem pause_point_correct : PausePoint = 30 := by
  sorry

end NUMINAMATH_CALUDE_pause_point_correct_l347_34704


namespace NUMINAMATH_CALUDE_sum_bounds_l347_34794

theorem sum_bounds (a b c d e : ℝ) :
  0 < (a / (a + b) + b / (b + c) + c / (c + d) + d / (d + e) + e / (e + a)) ∧
  (a / (a + b) + b / (b + c) + c / (c + d) + d / (d + e) + e / (e + a)) < 5 := by
  sorry

end NUMINAMATH_CALUDE_sum_bounds_l347_34794


namespace NUMINAMATH_CALUDE_reciprocal_of_negative_2023_l347_34701

theorem reciprocal_of_negative_2023 :
  ((-2023)⁻¹ : ℚ) = -1 / 2023 := by sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_2023_l347_34701


namespace NUMINAMATH_CALUDE_julia_bought_399_balls_l347_34726

/-- The number of balls Julia bought -/
def total_balls (red_packs yellow_packs green_packs balls_per_pack : ℕ) : ℕ :=
  (red_packs + yellow_packs + green_packs) * balls_per_pack

/-- Theorem stating that Julia bought 399 balls in total -/
theorem julia_bought_399_balls :
  total_balls 3 10 8 19 = 399 := by
  sorry

end NUMINAMATH_CALUDE_julia_bought_399_balls_l347_34726


namespace NUMINAMATH_CALUDE_expand_and_simplify_expression_l347_34774

theorem expand_and_simplify_expression (x : ℝ) : 
  (x^2 - 3*x + 3) * (x^2 + 3*x + 3) = x^4 - 3*x^2 + 9 := by
  sorry

end NUMINAMATH_CALUDE_expand_and_simplify_expression_l347_34774


namespace NUMINAMATH_CALUDE_root_sum_theorem_l347_34778

theorem root_sum_theorem (a b : ℝ) : 
  (Complex.I * Real.sqrt 7 + 2 : ℂ) ^ 3 + a * (Complex.I * Real.sqrt 7 + 2) + b = 0 → 
  a + b = 39 := by
  sorry

end NUMINAMATH_CALUDE_root_sum_theorem_l347_34778


namespace NUMINAMATH_CALUDE_equation_simplification_l347_34776

theorem equation_simplification :
  120 + (150 / 10) + (35 * 9) - 300 - (420 / 7) + 2^3 = 98 := by
  sorry

end NUMINAMATH_CALUDE_equation_simplification_l347_34776


namespace NUMINAMATH_CALUDE_hendecagon_diagonal_intersection_probability_l347_34731

/-- A regular hendecagon is an 11-sided polygon -/
def RegularHendecagon : Nat := 11

/-- The number of diagonals in a regular hendecagon -/
def NumDiagonals : Nat := (RegularHendecagon.choose 2) - RegularHendecagon

/-- The number of ways to choose 2 diagonals from the total number of diagonals -/
def WaysToChooseTwoDiagonals : Nat := NumDiagonals.choose 2

/-- The number of sets of 4 vertices that determine intersecting diagonals -/
def IntersectingDiagonalSets : Nat := RegularHendecagon.choose 4

/-- The probability that two randomly chosen diagonals intersect inside the hendecagon -/
def IntersectionProbability : Rat := IntersectingDiagonalSets / WaysToChooseTwoDiagonals

theorem hendecagon_diagonal_intersection_probability :
  IntersectionProbability = 165 / 473 := by
  sorry

end NUMINAMATH_CALUDE_hendecagon_diagonal_intersection_probability_l347_34731


namespace NUMINAMATH_CALUDE_df_ab_ratio_l347_34703

/-- Definition of the ellipse C -/
def C (x y : ℝ) : Prop := x^2 / 25 + y^2 / 9 = 1

/-- Definition of the right focus F -/
def F : ℝ × ℝ := (4, 0)

/-- Definition of line l passing through F -/
def l (k : ℝ) (x y : ℝ) : Prop := y - F.2 = k * (x - F.1)

/-- Definition of points A and B on the ellipse -/
def A (k : ℝ) : ℝ × ℝ := sorry
def B (k : ℝ) : ℝ × ℝ := sorry

/-- Definition of line l' (perpendicular bisector of AB) -/
def l' (k : ℝ) (x y : ℝ) : Prop :=
  y - (A k).2 / 2 - (B k).2 / 2 = -1/k * (x - (A k).1 / 2 - (B k).1 / 2)

/-- Definition of point D (intersection of l' and x-axis) -/
def D (k : ℝ) : ℝ × ℝ := sorry

/-- Theorem: The ratio DF/AB is equal to 2/5 -/
theorem df_ab_ratio (k : ℝ) :
  let df := Real.sqrt ((D k).1 - F.1)^2 + (D k).2^2
  let ab := Real.sqrt ((A k).1 - (B k).1)^2 + ((A k).2 - (B k).2)^2
  df / ab = 2/5 := by sorry

end NUMINAMATH_CALUDE_df_ab_ratio_l347_34703


namespace NUMINAMATH_CALUDE_calories_in_cookie_box_l347_34780

theorem calories_in_cookie_box (bags : ℕ) (cookies_per_bag : ℕ) (calories_per_cookie : ℕ) :
  bags = 6 →
  cookies_per_bag = 25 →
  calories_per_cookie = 18 →
  bags * cookies_per_bag * calories_per_cookie = 2700 :=
by
  sorry

end NUMINAMATH_CALUDE_calories_in_cookie_box_l347_34780


namespace NUMINAMATH_CALUDE_probability_no_adjacent_same_roll_probability_no_adjacent_same_roll_proof_l347_34788

/-- The probability of no two adjacent people rolling the same number on an 8-sided die
    when 5 people sit around a circular table. -/
theorem probability_no_adjacent_same_roll : ℚ :=
  let num_people : ℕ := 5
  let die_sides : ℕ := 8
  let prob_same : ℚ := 1 / die_sides
  let prob_diff : ℚ := 1 - prob_same
  let prob_case1 : ℚ := prob_same * prob_diff^2 * (die_sides - 2) / die_sides
  let prob_case2 : ℚ := prob_diff^3 * (die_sides - 2) / die_sides
  302 / 512

/-- Proof of the theorem -/
theorem probability_no_adjacent_same_roll_proof :
  probability_no_adjacent_same_roll = 302 / 512 := by
  sorry

end NUMINAMATH_CALUDE_probability_no_adjacent_same_roll_probability_no_adjacent_same_roll_proof_l347_34788


namespace NUMINAMATH_CALUDE_exponential_comparison_l347_34737

theorem exponential_comparison (a : ℝ) (h1 : 0 < a) (h2 : a < 1) :
  a^(-1 : ℝ) > a^(2 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_exponential_comparison_l347_34737


namespace NUMINAMATH_CALUDE_exists_irrational_less_than_three_l347_34798

theorem exists_irrational_less_than_three : ∃ x : ℝ, Irrational x ∧ |x| < 3 := by
  sorry

end NUMINAMATH_CALUDE_exists_irrational_less_than_three_l347_34798


namespace NUMINAMATH_CALUDE_expression_simplification_l347_34713

theorem expression_simplification (x : ℝ) (h : x = 3) :
  (3 / (x - 1) - x - 1) / ((x^2 - 4*x + 4) / (x - 1)) = -5 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l347_34713


namespace NUMINAMATH_CALUDE_triangle_inequalities_l347_34700

/-- Given a triangle with side lengths a, b, c, circumradius R, and inradius r,
    prove the inequalities abc ≥ (a+b-c)(a-b+c)(-a+b+c) and R ≥ 2r -/
theorem triangle_inequalities (a b c R r : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_circumradius : R > 0)
  (h_inradius : r > 0)
  (h_area : 4 * R * (r * (a + b + c) / 2) = a * b * c) :
  a * b * c ≥ (a + b - c) * (a - b + c) * (-a + b + c) ∧ R ≥ 2 * r := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequalities_l347_34700


namespace NUMINAMATH_CALUDE_final_salary_calculation_l347_34752

/-- Calculates the final salary after two salary changes --/
theorem final_salary_calculation (initial_salary : ℝ) (first_year_raise : ℝ) (second_year_cut : ℝ) :
  initial_salary = 10 →
  first_year_raise = 0.2 →
  second_year_cut = 0.75 →
  initial_salary * (1 + first_year_raise) * second_year_cut = 9 := by
  sorry

end NUMINAMATH_CALUDE_final_salary_calculation_l347_34752


namespace NUMINAMATH_CALUDE_rosy_age_l347_34741

/-- Proves that Rosy's current age is 12 years, given the conditions about David's age -/
theorem rosy_age (rosy_age david_age : ℕ) 
  (h1 : david_age = rosy_age + 18)
  (h2 : david_age + 6 = 2 * (rosy_age + 6)) : 
  rosy_age = 12 := by
  sorry

#check rosy_age

end NUMINAMATH_CALUDE_rosy_age_l347_34741


namespace NUMINAMATH_CALUDE_opposite_of_two_l347_34759

-- Define the concept of opposite
def opposite (a : ℝ) : ℝ := -a

-- State the theorem
theorem opposite_of_two : opposite 2 = -2 := by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_opposite_of_two_l347_34759


namespace NUMINAMATH_CALUDE_bus_speed_on_national_road_l347_34764

/-- The speed of a bus on the original national road, given specific conditions about a new highway --/
theorem bus_speed_on_national_road :
  ∀ (x : ℝ),
    (200 : ℝ) / (x + 45) = (220 : ℝ) / x / 2 →
    x = 55 := by
  sorry

end NUMINAMATH_CALUDE_bus_speed_on_national_road_l347_34764


namespace NUMINAMATH_CALUDE_successful_meeting_probability_l347_34728

-- Define the arrival times as real numbers between 0 and 2 (representing hours after 3:00 p.m.)
variable (x y z : ℝ)

-- Define the conditions for a successful meeting
def successful_meeting (x y z : ℝ) : Prop :=
  0 ≤ x ∧ x ≤ 2 ∧
  0 ≤ y ∧ y ≤ 2 ∧
  0 ≤ z ∧ z ≤ 2 ∧
  z > x ∧ z > y ∧
  |x - y| ≤ 1.5

-- Define the probability space
def total_outcomes : ℝ := 8

-- Define the volume of the region where the meeting is successful
noncomputable def successful_volume : ℝ := 8/9

-- Theorem stating the probability of a successful meeting
theorem successful_meeting_probability :
  (successful_volume / total_outcomes) = 1/9 :=
sorry

end NUMINAMATH_CALUDE_successful_meeting_probability_l347_34728


namespace NUMINAMATH_CALUDE_sqrt_72_equals_6_sqrt_2_l347_34743

theorem sqrt_72_equals_6_sqrt_2 : Real.sqrt 72 = 6 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_72_equals_6_sqrt_2_l347_34743


namespace NUMINAMATH_CALUDE_triangle_area_l347_34783

/-- Given a triangle with perimeter 35 cm and inradius 4.5 cm, its area is 78.75 cm² -/
theorem triangle_area (perimeter : ℝ) (inradius : ℝ) (area : ℝ) : 
  perimeter = 35 → inradius = 4.5 → area = perimeter / 2 * inradius → area = 78.75 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l347_34783


namespace NUMINAMATH_CALUDE_unique_solution_ceiling_equation_l347_34712

theorem unique_solution_ceiling_equation :
  ∃! b : ℝ, b + ⌈b⌉ = 25.3 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_unique_solution_ceiling_equation_l347_34712


namespace NUMINAMATH_CALUDE_train_length_train_length_correct_l347_34790

/-- Represents the scenario of two people walking alongside a moving train --/
structure TrainScenario where
  train_speed : ℝ
  walking_speed : ℝ
  person_a_distance : ℝ
  person_b_distance : ℝ
  (train_speed_positive : train_speed > 0)
  (walking_speed_positive : walking_speed > 0)
  (person_a_distance_positive : person_a_distance > 0)
  (person_b_distance_positive : person_b_distance > 0)
  (person_a_distance_eq : person_a_distance = 45)
  (person_b_distance_eq : person_b_distance = 30)

/-- The theorem stating that given the conditions, the train length is 180 meters --/
theorem train_length (scenario : TrainScenario) : ℝ :=
  180

/-- The main theorem proving that the train length is correct --/
theorem train_length_correct (scenario : TrainScenario) :
  train_length scenario = 180 := by
  sorry

end NUMINAMATH_CALUDE_train_length_train_length_correct_l347_34790


namespace NUMINAMATH_CALUDE_point_A_l347_34773

def point_A : ℝ × ℝ := (-2, 4)

def move_up (p : ℝ × ℝ) (units : ℝ) : ℝ × ℝ :=
  (p.1, p.2 + units)

def move_left (p : ℝ × ℝ) (units : ℝ) : ℝ × ℝ :=
  (p.1 - units, p.2)

def point_A' : ℝ × ℝ :=
  move_left (move_up point_A 2) 3

theorem point_A'_coordinates :
  point_A' = (-5, 6) := by
  sorry

end NUMINAMATH_CALUDE_point_A_l347_34773


namespace NUMINAMATH_CALUDE_total_paintings_after_five_weeks_l347_34733

/-- Represents a painter's weekly schedule and initial paintings -/
structure Painter where
  monday : Nat
  tuesday : Nat
  wednesday : Nat
  thursday : Nat
  friday : Nat
  saturday : Nat
  sunday : Nat
  initial : Nat

/-- Calculates the total number of paintings after a given number of weeks -/
def total_paintings (p : Painter) (weeks : Nat) : Nat :=
  p.initial + weeks * (p.monday + p.tuesday + p.wednesday + p.thursday + p.friday + p.saturday + p.sunday)

/-- Philip's painting schedule -/
def philip : Painter :=
  { monday := 3, tuesday := 3, wednesday := 2, thursday := 5, friday := 5, saturday := 0, sunday := 0, initial := 20 }

/-- Amelia's painting schedule -/
def amelia : Painter :=
  { monday := 2, tuesday := 2, wednesday := 2, thursday := 2, friday := 2, saturday := 2, sunday := 2, initial := 45 }

theorem total_paintings_after_five_weeks :
  total_paintings philip 5 + total_paintings amelia 5 = 225 := by
  sorry

end NUMINAMATH_CALUDE_total_paintings_after_five_weeks_l347_34733


namespace NUMINAMATH_CALUDE_solution_set_l347_34714

def equation (x : ℝ) : Prop :=
  1 / ((x - 2) * (x - 3)) + 1 / ((x - 3) * (x - 4)) + 1 / ((x - 4) * (x - 5)) = 1 / 8

theorem solution_set : {x : ℝ | equation x} = {7, -2} := by sorry

end NUMINAMATH_CALUDE_solution_set_l347_34714


namespace NUMINAMATH_CALUDE_units_digit_of_four_power_three_five_l347_34779

theorem units_digit_of_four_power_three_five (n : ℕ) : n = 4^(3^5) → n % 10 = 4 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_four_power_three_five_l347_34779


namespace NUMINAMATH_CALUDE_max_value_constrained_l347_34781

theorem max_value_constrained (x y : ℝ) (h : x^2 + y^2 = 4) :
  ∃ (max : ℝ), max = 14 ∧ ∀ (x' y' : ℝ), x'^2 + y'^2 = 4 → x'^2 + 6*y' + 2 ≤ max :=
sorry

end NUMINAMATH_CALUDE_max_value_constrained_l347_34781


namespace NUMINAMATH_CALUDE_complement_of_angle1_l347_34746

-- Define a type for angles in degrees and minutes
structure Angle :=
  (degrees : ℕ)
  (minutes : ℕ)

-- Define the given angle
def angle1 : Angle := ⟨38, 15⟩

-- Define the complement of an angle
def complement (a : Angle) : Angle :=
  let totalMinutes := 90 * 60 - (a.degrees * 60 + a.minutes)
  ⟨totalMinutes / 60, totalMinutes % 60⟩

-- Theorem statement
theorem complement_of_angle1 :
  complement angle1 = ⟨51, 45⟩ := by
  sorry

end NUMINAMATH_CALUDE_complement_of_angle1_l347_34746


namespace NUMINAMATH_CALUDE_dividend_calculation_l347_34725

theorem dividend_calculation (divisor quotient remainder : ℝ) 
  (h1 : divisor = 35.8)
  (h2 : quotient = 21.65)
  (h3 : remainder = 11.3) :
  divisor * quotient + remainder = 786.47 :=
by sorry

end NUMINAMATH_CALUDE_dividend_calculation_l347_34725


namespace NUMINAMATH_CALUDE_triangle_area_l347_34787

theorem triangle_area (A B C : Real) (h1 : A > B) (h2 : B > C) 
  (h3 : 2 * Real.cos (2 * B) - 8 * Real.cos B + 5 = 0)
  (h4 : Real.tan A + Real.tan C = 3 + Real.sqrt 3)
  (h5 : 2 * Real.sqrt 3 = Real.sin C * (A - C)) : 
  (1 / 2) * (A - C) * 2 * Real.sqrt 3 = 12 - 4 * Real.sqrt 3 := by
  sorry

#check triangle_area

end NUMINAMATH_CALUDE_triangle_area_l347_34787


namespace NUMINAMATH_CALUDE_segments_form_triangle_l347_34721

/-- Triangle Inequality Theorem: The sum of the lengths of any two sides of a triangle
    must be greater than the length of the remaining side. -/
def triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- A function that checks if three given lengths can form a triangle. -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ triangle_inequality a b c

/-- Theorem: The line segments 4cm, 5cm, and 6cm can form a triangle. -/
theorem segments_form_triangle : can_form_triangle 4 5 6 := by
  sorry

end NUMINAMATH_CALUDE_segments_form_triangle_l347_34721


namespace NUMINAMATH_CALUDE_triangle_angle_measure_l347_34754

theorem triangle_angle_measure (P Q R : ℝ) : 
  P = 88 → 
  Q = 2 * R + 18 → 
  P + Q + R = 180 → 
  R = 74 / 3 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_measure_l347_34754


namespace NUMINAMATH_CALUDE_vectors_not_collinear_l347_34730

/-- Given vectors a and b in ℝ³, prove that c₁ and c₂ are not collinear -/
theorem vectors_not_collinear (a b : ℝ × ℝ × ℝ) 
  (ha : a = (1, -2, 3))
  (hb : b = (3, 0, -1)) : 
  ¬ (∃ (k : ℝ), (2 • a + 4 • b) = k • (3 • b - a)) := by
  sorry

end NUMINAMATH_CALUDE_vectors_not_collinear_l347_34730


namespace NUMINAMATH_CALUDE_outfit_combinations_l347_34739

def number_of_shirts : ℕ := 5
def number_of_pants : ℕ := 6
def number_of_belts : ℕ := 2

theorem outfit_combinations : 
  number_of_shirts * number_of_pants * number_of_belts = 60 := by
  sorry

end NUMINAMATH_CALUDE_outfit_combinations_l347_34739


namespace NUMINAMATH_CALUDE_odd_primes_cube_sum_l347_34718

theorem odd_primes_cube_sum (p q r : ℕ) : 
  Nat.Prime p → Nat.Prime q → Nat.Prime r → 
  Odd p → Odd q → Odd r → 
  p^3 + q^3 + 3*p*q*r ≠ r^3 := by
  sorry

end NUMINAMATH_CALUDE_odd_primes_cube_sum_l347_34718


namespace NUMINAMATH_CALUDE_functional_equation_solution_l347_34738

theorem functional_equation_solution (f : ℝ → ℝ) :
  (∀ x y : ℝ, f (2 * x * y) + f (f (x + y)) = x * f y + y * f x + f (x + y)) →
  (∀ x : ℝ, f x = 0 ∨ f x = x ∨ f x = 2 - x) :=
by sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l347_34738


namespace NUMINAMATH_CALUDE_correlatedRelationships_l347_34766

-- Define the type for relationships
inductive Relationship
  | GreatTeachersAndStudents
  | SphereVolumeAndRadius
  | AppleYieldAndClimate
  | TreeDiameterAndHeight
  | StudentAndID
  | CrowCawAndOmen

-- Define a function to check if a relationship has correlation
def hasCorrelation (r : Relationship) : Prop :=
  match r with
  | Relationship.GreatTeachersAndStudents => True
  | Relationship.SphereVolumeAndRadius => False
  | Relationship.AppleYieldAndClimate => True
  | Relationship.TreeDiameterAndHeight => True
  | Relationship.StudentAndID => False
  | Relationship.CrowCawAndOmen => False

-- Theorem stating which relationships have correlation
theorem correlatedRelationships :
  (hasCorrelation Relationship.GreatTeachersAndStudents) ∧
  (hasCorrelation Relationship.AppleYieldAndClimate) ∧
  (hasCorrelation Relationship.TreeDiameterAndHeight) ∧
  (¬hasCorrelation Relationship.SphereVolumeAndRadius) ∧
  (¬hasCorrelation Relationship.StudentAndID) ∧
  (¬hasCorrelation Relationship.CrowCawAndOmen) :=
by sorry


end NUMINAMATH_CALUDE_correlatedRelationships_l347_34766


namespace NUMINAMATH_CALUDE_trig_identity_l347_34705

theorem trig_identity : 4 * Real.cos (50 * π / 180) - Real.tan (40 * π / 180) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l347_34705


namespace NUMINAMATH_CALUDE_sheep_sheepdog_distance_l347_34722

/-- The initial distance between a sheep and a sheepdog -/
def initial_distance (sheep_speed sheepdog_speed : ℝ) (catch_time : ℝ) : ℝ :=
  sheepdog_speed * catch_time - sheep_speed * catch_time

/-- Theorem stating the initial distance between the sheep and sheepdog -/
theorem sheep_sheepdog_distance :
  initial_distance 12 20 20 = 160 := by
  sorry

end NUMINAMATH_CALUDE_sheep_sheepdog_distance_l347_34722


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l347_34736

theorem complex_fraction_simplification : 
  let numerator := (11^4 + 400) * (25^4 + 400) * (37^4 + 400) * (49^4 + 400) * (61^4 + 400)
  let denominator := (5^4 + 400) * (17^4 + 400) * (29^4 + 400) * (41^4 + 400) * (53^4 + 400)
  numerator / denominator = 799 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l347_34736


namespace NUMINAMATH_CALUDE_max_value_cos_sin_l347_34719

theorem max_value_cos_sin (x : ℝ) : 
  let f := fun (x : ℝ) => 2 * Real.cos x + Real.sin x
  f x ≤ Real.sqrt 5 ∧ ∃ y, f y = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_max_value_cos_sin_l347_34719


namespace NUMINAMATH_CALUDE_inequality_system_solution_l347_34775

theorem inequality_system_solution (x : ℝ) : 
  (x - 2 < 0 ∧ 5 * x + 1 > 2 * (x - 1)) ↔ -1/3 < x ∧ x < 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l347_34775


namespace NUMINAMATH_CALUDE_inequality_proof_l347_34770

theorem inequality_proof (x : ℝ) 
  (h : (abs x ≤ 1) ∨ (abs x ≥ 2)) : 
  Real.cos (2*x^3 - x^2 - 5*x - 2) + 
  Real.cos (2*x^3 + 3*x^2 - 3*x - 2) - 
  Real.cos ((2*x + 1) * Real.sqrt (x^4 - 5*x^2 + 4)) < 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l347_34770


namespace NUMINAMATH_CALUDE_intersection_complement_theorem_l347_34758

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def P : Set ℕ := {1, 2, 3, 4}
def Q : Set ℕ := {3, 4, 5}

theorem intersection_complement_theorem :
  P ∩ (U \ Q) = {1, 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_complement_theorem_l347_34758


namespace NUMINAMATH_CALUDE_marley_has_31_fruits_l347_34749

/-- The number of fruits Marley has -/
def marley_fruits (louis_oranges louis_apples samantha_oranges samantha_apples : ℕ) : ℕ :=
  2 * louis_oranges + 3 * samantha_apples

/-- Theorem stating that Marley has 31 fruits given the conditions -/
theorem marley_has_31_fruits :
  marley_fruits 5 3 8 7 = 31 := by
  sorry

end NUMINAMATH_CALUDE_marley_has_31_fruits_l347_34749


namespace NUMINAMATH_CALUDE_trapezoid_area_in_regular_hexagon_l347_34750

/-- The area of a trapezoid formed by connecting midpoints of non-adjacent sides in a regular hexagon -/
theorem trapezoid_area_in_regular_hexagon (side_length : ℝ) (h : side_length = 12) :
  let height := side_length * Real.sqrt 3 / 2
  let trapezoid_base := side_length / 2
  let trapezoid_area := (trapezoid_base + trapezoid_base) * height / 2
  trapezoid_area = 36 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_area_in_regular_hexagon_l347_34750


namespace NUMINAMATH_CALUDE_car_speed_relationship_l347_34723

/-- Represents the relationship between the speeds and travel times of two cars -/
theorem car_speed_relationship (x : ℝ) : x > 0 →
  (80 / x - 2 = 80 / (3 * x) + 2 / 3) ↔
  (80 / x = 80 / (3 * x) + 2 + 2 / 3 ∧
   80 = x * (80 / (3 * x) + 2 + 2 / 3) ∧
   80 = 3 * x * (80 / (3 * x) + 2 / 3)) := by
  sorry

#check car_speed_relationship

end NUMINAMATH_CALUDE_car_speed_relationship_l347_34723


namespace NUMINAMATH_CALUDE_triangle_area_coefficient_product_l347_34795

/-- Given a triangle in the first quadrant bounded by the coordinate axes and a line,
    prove that if the area is 9, then the product of the coefficients is 4/3. -/
theorem triangle_area_coefficient_product (a b : ℝ) : 
  a > 0 → b > 0 → (∀ x y : ℝ, x ≥ 0 → y ≥ 0 → 2*a*x + 3*b*y ≤ 12) → 
  (1/2 * (12/(2*a)) * (12/(3*b)) = 9) → a * b = 4/3 := by
sorry

end NUMINAMATH_CALUDE_triangle_area_coefficient_product_l347_34795


namespace NUMINAMATH_CALUDE_jenny_easter_eggs_l347_34757

theorem jenny_easter_eggs :
  ∃ (n : ℕ), n > 0 ∧ n ≥ 5 ∧ 30 % n = 0 ∧ 45 % n = 0 ∧
  ∀ (m : ℕ), m > 0 ∧ m ≥ 5 ∧ 30 % m = 0 ∧ 45 % m = 0 → m ≤ n :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_jenny_easter_eggs_l347_34757


namespace NUMINAMATH_CALUDE_same_terminal_side_l347_34761

theorem same_terminal_side (a b : Real) : 
  a = -7 * π / 9 → b = 11 * π / 9 → ∃ k : Int, b - a = 2 * π * k := by
  sorry

#check same_terminal_side

end NUMINAMATH_CALUDE_same_terminal_side_l347_34761


namespace NUMINAMATH_CALUDE_power_of_product_l347_34740

theorem power_of_product (a b : ℝ) : (a * b^2)^3 = a^3 * b^6 := by sorry

end NUMINAMATH_CALUDE_power_of_product_l347_34740


namespace NUMINAMATH_CALUDE_shaded_triangle_probability_l347_34748

theorem shaded_triangle_probability (total_triangles shaded_triangles : ℕ) 
  (h1 : total_triangles = 9)
  (h2 : shaded_triangles = 4)
  (h3 : shaded_triangles ≤ total_triangles)
  (h4 : total_triangles > 4) :
  (shaded_triangles : ℚ) / total_triangles = 4 / 9 := by
  sorry

end NUMINAMATH_CALUDE_shaded_triangle_probability_l347_34748


namespace NUMINAMATH_CALUDE_xyz_value_l347_34709

theorem xyz_value (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h1 : x * (y + z) = 180) (h2 : y * (z + x) = 192) (h3 : z * (x + y) = 204) :
  x * y * z = 168 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_xyz_value_l347_34709


namespace NUMINAMATH_CALUDE_x_range_given_sqrt_equality_l347_34734

theorem x_range_given_sqrt_equality (x : ℝ) :
  Real.sqrt ((5 - x)^2) = x - 5 → x ≥ 5 := by
  sorry

end NUMINAMATH_CALUDE_x_range_given_sqrt_equality_l347_34734


namespace NUMINAMATH_CALUDE_sqrt_neg_four_squared_l347_34742

theorem sqrt_neg_four_squared : Real.sqrt ((-4)^2) = 4 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_neg_four_squared_l347_34742


namespace NUMINAMATH_CALUDE_equation_solution_l347_34753

theorem equation_solution (a : ℤ) : 
  (∃ x : ℤ, x > 0 ∧ a * x + 3 = 4 * x + 1) ↔ (a = 2 ∨ a = 3) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l347_34753


namespace NUMINAMATH_CALUDE_smallest_integer_satisfying_inequality_l347_34763

theorem smallest_integer_satisfying_inequality : 
  ∀ y : ℤ, y < 3 * y - 14 → y ≥ 8 ∧ 8 < 3 * 8 - 14 := by
  sorry

end NUMINAMATH_CALUDE_smallest_integer_satisfying_inequality_l347_34763


namespace NUMINAMATH_CALUDE_job1_rate_is_correct_l347_34755

/-- Represents the hourly rate of job 1 -/
def job1_rate : ℝ := 7

/-- Represents the hourly rate of job 2 -/
def job2_rate : ℝ := 10

/-- Represents the hourly rate of job 3 -/
def job3_rate : ℝ := 12

/-- Represents the number of hours worked on job 1 per day -/
def job1_hours : ℝ := 3

/-- Represents the number of hours worked on job 2 per day -/
def job2_hours : ℝ := 2

/-- Represents the number of hours worked on job 3 per day -/
def job3_hours : ℝ := 4

/-- Represents the number of days worked -/
def days_worked : ℝ := 5

/-- Represents the total earnings for the period -/
def total_earnings : ℝ := 445

theorem job1_rate_is_correct : 
  days_worked * (job1_hours * job1_rate + job2_hours * job2_rate + job3_hours * job3_rate) = total_earnings := by
  sorry

end NUMINAMATH_CALUDE_job1_rate_is_correct_l347_34755


namespace NUMINAMATH_CALUDE_least_possible_area_l347_34760

/-- The least possible length of a side when measured as 4 cm to the nearest centimeter -/
def min_side_length : ℝ := 3.5

/-- The measured length of the square's side to the nearest centimeter -/
def measured_side_length : ℕ := 4

/-- The least possible area of the square -/
def min_area : ℝ := min_side_length ^ 2

theorem least_possible_area :
  min_area = 12.25 := by sorry

end NUMINAMATH_CALUDE_least_possible_area_l347_34760


namespace NUMINAMATH_CALUDE_D_sqrt_sometimes_rational_sometimes_not_l347_34732

def D (x : ℝ) : ℝ := 
  let a := 2*x + 1
  let b := 2*x + 3
  let c := a*b + 5
  a^2 + b^2 + c^2

theorem D_sqrt_sometimes_rational_sometimes_not :
  ∃ x y : ℝ, (∃ q : ℚ, Real.sqrt (D x) = q) ∧ 
             (∀ q : ℚ, Real.sqrt (D y) ≠ q) :=
sorry

end NUMINAMATH_CALUDE_D_sqrt_sometimes_rational_sometimes_not_l347_34732


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l347_34745

theorem imaginary_part_of_z (z : ℂ) (h : 1 + (1 + 2 * z) * Complex.I = 0) :
  z.im = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l347_34745


namespace NUMINAMATH_CALUDE_mod_congruence_solution_l347_34793

theorem mod_congruence_solution : ∃! (n : ℤ), 0 ≤ n ∧ n ≤ 10 ∧ n ≡ -2154 [ZMOD 7] ∧ n = 2 := by
  sorry

end NUMINAMATH_CALUDE_mod_congruence_solution_l347_34793


namespace NUMINAMATH_CALUDE_brick_laying_time_l347_34785

/-- Given that 2b men can lay 3f bricks in c days, prove that 4c men will take b^2 / f days to lay 6b bricks, assuming constant working rate. -/
theorem brick_laying_time 
  (b f c : ℝ) 
  (h : b > 0 ∧ f > 0 ∧ c > 0) 
  (rate : ℝ := (3 * f) / (2 * b * c)) : 
  (6 * b) / (4 * c * rate) = b^2 / f := by
sorry

end NUMINAMATH_CALUDE_brick_laying_time_l347_34785


namespace NUMINAMATH_CALUDE_lower_bound_of_a_l347_34716

open Real

theorem lower_bound_of_a (f g : ℝ → ℝ) (a : ℝ) :
  (∀ x, x > 0 → f x = x * log x) →
  (∀ x, g x = x^3 + a*x^2 - x + 2) →
  (∀ x, x > 0 → 2 * f x ≤ (deriv g) x + 2) →
  a ≥ -2 := by
  sorry

end NUMINAMATH_CALUDE_lower_bound_of_a_l347_34716


namespace NUMINAMATH_CALUDE_hiking_problem_solution_l347_34772

/-- Represents the hiking problem with given speeds and distances -/
structure HikingProblem where
  total_time : ℚ  -- in hours
  total_distance : ℚ  -- in km
  uphill_speed : ℚ  -- in km/h
  flat_speed : ℚ  -- in km/h
  downhill_speed : ℚ  -- in km/h

/-- Theorem stating the solution to the hiking problem -/
theorem hiking_problem_solution (p : HikingProblem) 
  (h1 : p.total_time = 221 / 60)  -- 3 hours and 41 minutes in decimal form
  (h2 : p.total_distance = 9)
  (h3 : p.uphill_speed = 4)
  (h4 : p.flat_speed = 5)
  (h5 : p.downhill_speed = 6) :
  ∃ (x : ℚ), x = 4 ∧ 
    (2 * x / p.flat_speed + 
     (5 * (p.total_distance - x)) / (12 : ℚ) = p.total_time) := by
  sorry


end NUMINAMATH_CALUDE_hiking_problem_solution_l347_34772


namespace NUMINAMATH_CALUDE_scooter_repair_cost_l347_34786

/-- Proves that the total repair cost is $11,000 given the conditions of Peter's scooter purchase and sale --/
theorem scooter_repair_cost (C : ℝ) : 
  (0.05 * C + 0.10 * C + 0.07 * C = 0.22 * C) →  -- Total repair cost is 22% of C
  (1.25 * C - C - 0.22 * C = 1500) →              -- Profit equation
  0.22 * C = 11000 :=                             -- Total repair cost is $11,000
by sorry

end NUMINAMATH_CALUDE_scooter_repair_cost_l347_34786


namespace NUMINAMATH_CALUDE_sqrt_5_irrational_l347_34768

-- Define what it means for a number to be rational
def IsRational (x : ℝ) : Prop :=
  ∃ (p q : ℤ), q ≠ 0 ∧ x = p / q

-- Define what it means for a number to be irrational
def IsIrrational (x : ℝ) : Prop := ¬(IsRational x)

-- State the theorem
theorem sqrt_5_irrational : IsIrrational (Real.sqrt 5) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_5_irrational_l347_34768


namespace NUMINAMATH_CALUDE_average_writing_rate_l347_34762

/-- Given a writer who completed 50,000 words in 100 hours, 
    prove that the average writing rate is 500 words per hour. -/
theorem average_writing_rate 
  (total_words : ℕ) 
  (total_hours : ℕ) 
  (h1 : total_words = 50000) 
  (h2 : total_hours = 100) : 
  (total_words : ℚ) / total_hours = 500 := by
  sorry

end NUMINAMATH_CALUDE_average_writing_rate_l347_34762


namespace NUMINAMATH_CALUDE_campers_total_l347_34707

/-- The total number of campers participating in all activities -/
def total_campers (morning_rowing : ℕ) (morning_hiking : ℕ) (morning_climbing : ℕ)
                  (afternoon_rowing : ℕ) (afternoon_hiking : ℕ) (afternoon_biking : ℕ) : ℕ :=
  morning_rowing + morning_hiking + morning_climbing +
  afternoon_rowing + afternoon_hiking + afternoon_biking

/-- Theorem stating that the total number of campers is 180 -/
theorem campers_total :
  total_campers 13 59 25 21 47 15 = 180 := by
  sorry

#eval total_campers 13 59 25 21 47 15

end NUMINAMATH_CALUDE_campers_total_l347_34707


namespace NUMINAMATH_CALUDE_max_profit_plan_l347_34769

-- Define the appliance types
inductive Appliance
| TV
| Refrigerator
| WashingMachine

-- Define the cost and selling prices
def cost_price (a : Appliance) : ℕ :=
  match a with
  | Appliance.TV => 2000
  | Appliance.Refrigerator => 1600
  | Appliance.WashingMachine => 1000

def selling_price (a : Appliance) : ℕ :=
  match a with
  | Appliance.TV => 2200
  | Appliance.Refrigerator => 1800
  | Appliance.WashingMachine => 1100

-- Define the purchasing plan
structure PurchasingPlan where
  tv_count : ℕ
  refrigerator_count : ℕ
  washing_machine_count : ℕ

-- Define the constraints
def is_valid_plan (p : PurchasingPlan) : Prop :=
  p.tv_count + p.refrigerator_count + p.washing_machine_count = 100 ∧
  p.tv_count = p.refrigerator_count ∧
  p.washing_machine_count ≤ p.tv_count ∧
  p.tv_count * cost_price Appliance.TV +
  p.refrigerator_count * cost_price Appliance.Refrigerator +
  p.washing_machine_count * cost_price Appliance.WashingMachine ≤ 160000

-- Define the profit calculation
def profit (p : PurchasingPlan) : ℕ :=
  p.tv_count * (selling_price Appliance.TV - cost_price Appliance.TV) +
  p.refrigerator_count * (selling_price Appliance.Refrigerator - cost_price Appliance.Refrigerator) +
  p.washing_machine_count * (selling_price Appliance.WashingMachine - cost_price Appliance.WashingMachine)

-- Theorem statement
theorem max_profit_plan :
  ∃ (p : PurchasingPlan),
    is_valid_plan p ∧
    profit p = 17400 ∧
    ∀ (q : PurchasingPlan), is_valid_plan q → profit q ≤ profit p :=
sorry

end NUMINAMATH_CALUDE_max_profit_plan_l347_34769


namespace NUMINAMATH_CALUDE_tan_of_angle_on_x_plus_y_equals_zero_l347_34711

/-- An angle whose terminal side lies on the line x + y = 0 -/
structure AngleOnXPlusYEqualsZero where
  α : Real
  terminal_side : ∀ (x y : Real), x + y = 0 → (∃ (t : Real), x = t * Real.cos α ∧ y = t * Real.sin α)

/-- The tangent of an angle whose terminal side lies on the line x + y = 0 is -1 -/
theorem tan_of_angle_on_x_plus_y_equals_zero (θ : AngleOnXPlusYEqualsZero) : Real.tan θ.α = -1 := by
  sorry

end NUMINAMATH_CALUDE_tan_of_angle_on_x_plus_y_equals_zero_l347_34711


namespace NUMINAMATH_CALUDE_sequence_minimum_l347_34767

/-- Given a sequence {a_n} satisfying the conditions:
    a_1 = p, a_2 = p + 1, and a_{n+2} - 2a_{n+1} + a_n = n - 20,
    where p is a real number and n is a positive integer,
    prove that a_n is minimized when n = 40. -/
theorem sequence_minimum (p : ℝ) : 
  ∃ (a : ℕ → ℝ), 
    (a 1 = p) ∧ 
    (a 2 = p + 1) ∧ 
    (∀ n : ℕ, n ≥ 1 → a (n + 2) - 2 * a (n + 1) + a n = n - 20) ∧
    (∀ n : ℕ, n ≥ 1 → a 40 ≤ a n) :=
by sorry

end NUMINAMATH_CALUDE_sequence_minimum_l347_34767
