import Mathlib

namespace NUMINAMATH_CALUDE_paiges_team_size_l378_37882

theorem paiges_team_size (total_points : ℕ) (paige_points : ℕ) (other_player_points : ℕ) :
  total_points = 41 →
  paige_points = 11 →
  other_player_points = 6 →
  ∃ (team_size : ℕ), team_size = (total_points - paige_points) / other_player_points + 1 ∧ team_size = 6 :=
by sorry

end NUMINAMATH_CALUDE_paiges_team_size_l378_37882


namespace NUMINAMATH_CALUDE_sum_of_fifth_powers_l378_37844

theorem sum_of_fifth_powers (x y z : ℝ) 
  (eq1 : x + y + z = 3)
  (eq2 : x^3 + y^3 + z^3 = 15)
  (eq3 : x^4 + y^4 + z^4 = 35)
  (ineq : x^2 + y^2 + z^2 < 10) :
  x^5 + y^5 + z^5 = 83 := by
sorry

end NUMINAMATH_CALUDE_sum_of_fifth_powers_l378_37844


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_range_l378_37801

theorem quadratic_inequality_solution_range (c : ℝ) : 
  (c > 0 ∧ ∃ x : ℝ, x^2 - 10*x + c < 0) ↔ (0 < c ∧ c < 25) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_range_l378_37801


namespace NUMINAMATH_CALUDE_perimeter_of_seven_unit_squares_l378_37833

/-- A figure composed of unit squares meeting at vertices -/
structure SquareFigure where
  num_squares : ℕ
  squares_meet_at_vertices : Bool

/-- The perimeter of a square figure -/
def perimeter (f : SquareFigure) : ℕ := 
  if f.squares_meet_at_vertices then
    4 * f.num_squares
  else
    sorry  -- We don't handle this case in this problem

theorem perimeter_of_seven_unit_squares : 
  ∀ (f : SquareFigure), f.num_squares = 7 → f.squares_meet_at_vertices → perimeter f = 28 := by
  sorry

end NUMINAMATH_CALUDE_perimeter_of_seven_unit_squares_l378_37833


namespace NUMINAMATH_CALUDE_manufacturing_cost_calculation_l378_37859

/-- The manufacturing cost of a shoe -/
def manufacturing_cost : ℝ := sorry

/-- The transportation cost for 100 shoes -/
def transportation_cost_100 : ℝ := 500

/-- The selling price of a shoe -/
def selling_price : ℝ := 222

/-- The gain percentage on the selling price -/
def gain_percentage : ℝ := 20

theorem manufacturing_cost_calculation : 
  manufacturing_cost = 180 := by
  sorry

end NUMINAMATH_CALUDE_manufacturing_cost_calculation_l378_37859


namespace NUMINAMATH_CALUDE_fraction_modification_l378_37846

theorem fraction_modification (d : ℚ) : 
  (3 : ℚ) / d ≠ (1 : ℚ) / (3 : ℚ) →
  ((3 : ℚ) + 3) / (d + 3) = (1 : ℚ) / (3 : ℚ) →
  d = 15 := by
sorry

end NUMINAMATH_CALUDE_fraction_modification_l378_37846


namespace NUMINAMATH_CALUDE_minimize_quadratic_l378_37887

theorem minimize_quadratic (c : ℝ) :
  ∃ (b : ℝ), ∀ (x : ℝ), 3 * b^2 + 2 * b + c ≤ 3 * x^2 + 2 * x + c :=
by
  use (-1/3)
  sorry

end NUMINAMATH_CALUDE_minimize_quadratic_l378_37887


namespace NUMINAMATH_CALUDE_largest_integer_in_range_l378_37871

theorem largest_integer_in_range : ∃ (x : ℤ), 
  (1 / 4 : ℚ) < (x : ℚ) / 7 ∧ 
  (x : ℚ) / 7 < (2 / 3 : ℚ) ∧ 
  ∀ (y : ℤ), (1 / 4 : ℚ) < (y : ℚ) / 7 ∧ (y : ℚ) / 7 < (2 / 3 : ℚ) → y ≤ x :=
by sorry

end NUMINAMATH_CALUDE_largest_integer_in_range_l378_37871


namespace NUMINAMATH_CALUDE_pen_purchase_shortfall_l378_37819

/-- The amount of money needed to purchase a pen given the cost, initial amount, and borrowed amount -/
theorem pen_purchase_shortfall (pen_cost : ℕ) (initial_amount : ℕ) (borrowed_amount : ℕ) :
  pen_cost = 600 →
  initial_amount = 500 →
  borrowed_amount = 68 →
  pen_cost - (initial_amount + borrowed_amount) = 32 := by
  sorry

end NUMINAMATH_CALUDE_pen_purchase_shortfall_l378_37819


namespace NUMINAMATH_CALUDE_yeast_growth_proof_l378_37821

/-- Calculates the yeast population after a given time -/
def yeast_population (initial_population : ℕ) (growth_factor : ℕ) (interval_duration : ℕ) (total_time : ℕ) : ℕ :=
  initial_population * growth_factor ^ (total_time / interval_duration)

/-- Proves that the yeast population grows to 1350 after 18 minutes -/
theorem yeast_growth_proof :
  yeast_population 50 3 5 18 = 1350 := by
  sorry

end NUMINAMATH_CALUDE_yeast_growth_proof_l378_37821


namespace NUMINAMATH_CALUDE_max_vertex_sum_l378_37881

/-- Represents a face of the dice -/
structure Face where
  value : Nat
  deriving Repr

/-- Represents a cubical dice -/
structure Dice where
  faces : List Face
  opposite_sum : Nat

/-- Defines a valid cubical dice where opposite faces sum to 8 -/
def is_valid_dice (d : Dice) : Prop :=
  d.faces.length = 6 ∧
  d.opposite_sum = 8 ∧
  ∀ (f1 f2 : Face), f1 ∈ d.faces → f2 ∈ d.faces → f1 ≠ f2 → f1.value + f2.value = d.opposite_sum

/-- Represents three faces meeting at a vertex -/
structure Vertex where
  f1 : Face
  f2 : Face
  f3 : Face

/-- Calculates the sum of face values at a vertex -/
def vertex_sum (v : Vertex) : Nat :=
  v.f1.value + v.f2.value + v.f3.value

/-- Defines a valid vertex of the dice -/
def is_valid_vertex (d : Dice) (v : Vertex) : Prop :=
  v.f1 ∈ d.faces ∧ v.f2 ∈ d.faces ∧ v.f3 ∈ d.faces ∧
  v.f1 ≠ v.f2 ∧ v.f1 ≠ v.f3 ∧ v.f2 ≠ v.f3

theorem max_vertex_sum (d : Dice) (h : is_valid_dice d) :
  ∀ (v : Vertex), is_valid_vertex d v → vertex_sum v ≤ 11 :=
sorry

end NUMINAMATH_CALUDE_max_vertex_sum_l378_37881


namespace NUMINAMATH_CALUDE_no_solution_to_system_l378_37868

theorem no_solution_to_system :
  ¬ ∃ (x y z : ℝ), 
    (3 * x - 4 * y + z = 10) ∧ 
    (6 * x - 8 * y + 2 * z = 16) ∧ 
    (x + y - z = 3) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_to_system_l378_37868


namespace NUMINAMATH_CALUDE_joannes_weekly_earnings_is_812_48_l378_37854

/-- Calculates Joanne's weekly earnings after deductions, bonuses, and allowances -/
def joannes_weekly_earnings : ℝ :=
  let main_job_hours : ℝ := 8 * 5
  let main_job_rate : ℝ := 16
  let main_job_base_pay : ℝ := main_job_hours * main_job_rate
  let main_job_bonus_rate : ℝ := 0.1
  let main_job_bonus : ℝ := main_job_base_pay * main_job_bonus_rate
  let main_job_total : ℝ := main_job_base_pay + main_job_bonus
  let main_job_deduction_rate : ℝ := 0.05
  let main_job_deduction : ℝ := main_job_total * main_job_deduction_rate
  let main_job_net : ℝ := main_job_total - main_job_deduction

  let part_time_regular_hours : ℝ := 2 * 4
  let part_time_friday_hours : ℝ := 3
  let part_time_rate : ℝ := 13.5
  let part_time_friday_bonus : ℝ := 2
  let part_time_regular_pay : ℝ := part_time_regular_hours * part_time_rate
  let part_time_friday_pay : ℝ := part_time_friday_hours * (part_time_rate + part_time_friday_bonus)
  let part_time_total : ℝ := part_time_regular_pay + part_time_friday_pay
  let part_time_deduction_rate : ℝ := 0.07
  let part_time_deduction : ℝ := part_time_total * part_time_deduction_rate
  let part_time_net : ℝ := part_time_total - part_time_deduction

  main_job_net + part_time_net

/-- Theorem: Joanne's weekly earnings after deductions, bonuses, and allowances is $812.48 -/
theorem joannes_weekly_earnings_is_812_48 : joannes_weekly_earnings = 812.48 := by
  sorry

end NUMINAMATH_CALUDE_joannes_weekly_earnings_is_812_48_l378_37854


namespace NUMINAMATH_CALUDE_rainfall_difference_l378_37817

-- Define the rainfall amounts for Monday and Tuesday
def monday_rainfall : ℝ := 0.9
def tuesday_rainfall : ℝ := 0.2

-- Theorem to prove the difference in rainfall
theorem rainfall_difference : monday_rainfall - tuesday_rainfall = 0.7 := by
  sorry

end NUMINAMATH_CALUDE_rainfall_difference_l378_37817


namespace NUMINAMATH_CALUDE_complex_equation_solution_l378_37892

theorem complex_equation_solution (z : ℂ) (h : z * (1 - Complex.I) = 2) : z = 1 + Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l378_37892


namespace NUMINAMATH_CALUDE_coffee_cream_ratio_l378_37842

/-- The ratio of cream in Joe's coffee to JoAnn's coffee -/
theorem coffee_cream_ratio :
  let initial_coffee : ℝ := 20
  let joe_drank : ℝ := 3
  let cream_added : ℝ := 4
  let joann_drank : ℝ := 3
  let joe_cream : ℝ := cream_added
  let joann_total : ℝ := initial_coffee + cream_added
  let joann_cream_ratio : ℝ := cream_added / joann_total
  let joann_cream : ℝ := cream_added - (joann_drank * joann_cream_ratio)
  (joe_cream / joann_cream) = 8 / 7 := by
sorry

end NUMINAMATH_CALUDE_coffee_cream_ratio_l378_37842


namespace NUMINAMATH_CALUDE_total_pure_acid_in_mixture_l378_37816

/-- Represents a solution with its acid concentration and volume -/
structure Solution where
  concentration : Real
  volume : Real

/-- Calculates the amount of pure acid in a solution -/
def pureAcidAmount (s : Solution) : Real :=
  s.concentration * s.volume

/-- Theorem: The total amount of pure acid in a mixture of solutions is the sum of pure acid amounts from each solution -/
theorem total_pure_acid_in_mixture (solutionA solutionB solutionC : Solution)
  (hA : solutionA.concentration = 0.20 ∧ solutionA.volume = 8)
  (hB : solutionB.concentration = 0.35 ∧ solutionB.volume = 5)
  (hC : solutionC.concentration = 0.15 ∧ solutionC.volume = 3) :
  pureAcidAmount solutionA + pureAcidAmount solutionB + pureAcidAmount solutionC = 3.8 := by
  sorry


end NUMINAMATH_CALUDE_total_pure_acid_in_mixture_l378_37816


namespace NUMINAMATH_CALUDE_ab_value_l378_37893

theorem ab_value (a b : ℝ) (h1 : a + b = 5) (h2 : a^3 + b^3 = 125) : a * b = 0 := by
  sorry

end NUMINAMATH_CALUDE_ab_value_l378_37893


namespace NUMINAMATH_CALUDE_oil_change_price_is_20_l378_37877

def oil_change_price (repair_price car_wash_price : ℕ) 
                     (oil_changes repairs car_washes : ℕ) 
                     (total_earnings : ℕ) : Prop :=
  ∃ (x : ℕ), 
    repair_price = 30 ∧
    car_wash_price = 5 ∧
    oil_changes = 5 ∧
    repairs = 10 ∧
    car_washes = 15 ∧
    total_earnings = 475 ∧
    x * oil_changes + repair_price * repairs + car_wash_price * car_washes = total_earnings ∧
    x = 20

theorem oil_change_price_is_20 : 
  ∀ (repair_price car_wash_price oil_changes repairs car_washes total_earnings : ℕ),
    oil_change_price repair_price car_wash_price oil_changes repairs car_washes total_earnings :=
by
  sorry

end NUMINAMATH_CALUDE_oil_change_price_is_20_l378_37877


namespace NUMINAMATH_CALUDE_trigonometric_identity_l378_37884

theorem trigonometric_identity (α : Real) : 
  Real.sin α ^ 2 + Real.cos (π / 6 + α) ^ 2 + Real.sin α * Real.cos (π / 6 + α) = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l378_37884


namespace NUMINAMATH_CALUDE_problem_statement_l378_37818

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x - Real.log x

theorem problem_statement :
  (∀ x > 0, f 1 x ≥ 1) ∧
  (∃ x ∈ Set.Icc (1 / Real.exp 1) (Real.exp 1), f 1 x = 1) ∧
  (∀ a ∈ Set.Icc 0 1, ∃ x ∈ Set.Icc (1 / Real.exp 1) (Real.exp 1), f a x = 1) ∧
  (∀ a ≥ 1, ∀ x ≥ 1, f a x ≥ f a (1 / x)) ∧
  (∀ a < 1, ∃ x ≥ 1, f a x < f a (1 / x)) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l378_37818


namespace NUMINAMATH_CALUDE_chessboard_coloring_count_l378_37888

/-- The number of ways to paint an N × N chessboard with 4 colors such that:
    1) Squares with a common side are painted with distinct colors
    2) Every 2 × 2 square is painted with the four colors -/
def chessboardColorings (N : ℕ) : ℕ := 24 * (2^(N-1) - 1)

/-- Theorem stating the number of valid colorings for an N × N chessboard -/
theorem chessboard_coloring_count (N : ℕ) (h : N > 1) : 
  chessboardColorings N = 24 * (2^(N-1) - 1) := by
  sorry


end NUMINAMATH_CALUDE_chessboard_coloring_count_l378_37888


namespace NUMINAMATH_CALUDE_age_condition_l378_37898

/-- Given three people A, B, and C, this theorem states that if A is older than B,
    then "C is older than B" is a necessary but not sufficient condition for
    "the sum of B and C's ages is greater than twice A's age". -/
theorem age_condition (a b c : ℕ) (h : a > b) :
  (c > b → b + c > 2 * a) ∧ ¬(b + c > 2 * a → c > b) := by
  sorry

end NUMINAMATH_CALUDE_age_condition_l378_37898


namespace NUMINAMATH_CALUDE_cartons_used_is_38_l378_37809

/-- Represents the packing of tennis rackets into cartons. -/
structure RacketPacking where
  total_rackets : ℕ
  cartons_of_three : ℕ
  cartons_of_two : ℕ

/-- Calculates the total number of cartons used. -/
def total_cartons (packing : RacketPacking) : ℕ :=
  packing.cartons_of_two + packing.cartons_of_three

/-- Theorem stating that for the given packing scenario, 38 cartons are used in total. -/
theorem cartons_used_is_38 (packing : RacketPacking) 
  (h1 : packing.total_rackets = 100)
  (h2 : packing.cartons_of_three = 24)
  (h3 : 2 * packing.cartons_of_two + 3 * packing.cartons_of_three = packing.total_rackets) :
  total_cartons packing = 38 := by
  sorry

#check cartons_used_is_38

end NUMINAMATH_CALUDE_cartons_used_is_38_l378_37809


namespace NUMINAMATH_CALUDE_inverse_cube_root_relation_l378_37837

/-- Given that y varies inversely as the cube root of x, prove that when x = 8 and y = 2,
    then x = 1/8 when y = 8 -/
theorem inverse_cube_root_relation (x y : ℝ) (k : ℝ) : 
  (∀ x y, y * (x ^ (1/3 : ℝ)) = k) →  -- y varies inversely as the cube root of x
  (2 * (8 ^ (1/3 : ℝ)) = k) →         -- when x = 8, y = 2
  (8 * (x ^ (1/3 : ℝ)) = k) →         -- when y = 8
  x = 1/8 := by
sorry

end NUMINAMATH_CALUDE_inverse_cube_root_relation_l378_37837


namespace NUMINAMATH_CALUDE_factor_sum_l378_37824

theorem factor_sum (P Q : ℝ) : 
  (∃ c d : ℝ, (X^2 - 3*X + 7) * (X^2 + c*X + d) = X^4 + P*X^2 + Q) →
  P + Q = 54 := by
sorry

end NUMINAMATH_CALUDE_factor_sum_l378_37824


namespace NUMINAMATH_CALUDE_sin_minus_cos_special_angle_l378_37890

/-- Given an angle α whose terminal side passes through the point (3a, -4a) where a < 0,
    prove that sinα - cosα = 7/5 -/
theorem sin_minus_cos_special_angle (a : ℝ) (α : Real) (h : a < 0) 
    (h_terminal : ∃ k : ℝ, k > 0 ∧ k * Real.cos α = 3 * a ∧ k * Real.sin α = -4 * a) :
    Real.sin α - Real.cos α = 7/5 := by
  sorry

end NUMINAMATH_CALUDE_sin_minus_cos_special_angle_l378_37890


namespace NUMINAMATH_CALUDE_total_ways_to_draw_l378_37849

/-- Represents the number of cards of each color -/
def cards_per_color : ℕ := 5

/-- Represents the number of colors -/
def num_colors : ℕ := 3

/-- Represents the total number of cards -/
def total_cards : ℕ := cards_per_color * num_colors

/-- Represents the number of cards to be drawn -/
def cards_to_draw : ℕ := 5

/-- Represents the number of ways to draw cards in the (3,1,1) distribution -/
def ways_311 : ℕ := (Nat.choose 3 1) * (Nat.choose cards_per_color 3) * (Nat.choose 2 1) * (Nat.choose 2 1) / 2

/-- Represents the number of ways to draw cards in the (2,2,1) distribution -/
def ways_221 : ℕ := (Nat.choose 3 1) * (Nat.choose cards_per_color 2) * (Nat.choose 2 1) * (Nat.choose 3 2) * (Nat.choose 1 1) / 2

/-- The main theorem stating the total number of ways to draw the cards -/
theorem total_ways_to_draw : ways_311 + ways_221 = 150 := by
  sorry

end NUMINAMATH_CALUDE_total_ways_to_draw_l378_37849


namespace NUMINAMATH_CALUDE_area_of_enclosed_region_l378_37869

/-- The equation of the curve enclosing the region -/
def curve_equation (x y : ℝ) : Prop :=
  x^2 - 18*x + 3*y + 90 = 33 + 9*y - y^2

/-- The equation of the line bounding the region above -/
def line_equation (x y : ℝ) : Prop :=
  y = x - 5

/-- The region enclosed by the curve and below the line -/
def enclosed_region : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | curve_equation p.1 p.2 ∧ p.2 ≤ p.1 - 5}

/-- The area of the enclosed region -/
noncomputable def area_of_region : ℝ := sorry

theorem area_of_enclosed_region :
  area_of_region = 33 * Real.pi / 2 := by sorry

end NUMINAMATH_CALUDE_area_of_enclosed_region_l378_37869


namespace NUMINAMATH_CALUDE_max_xy_value_l378_37858

theorem max_xy_value (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 4 * x + 3 * y = 12) :
  x * y ≤ 3 ∧ ∃ x y, x > 0 ∧ y > 0 ∧ 4 * x + 3 * y = 12 ∧ x * y = 3 :=
sorry

end NUMINAMATH_CALUDE_max_xy_value_l378_37858


namespace NUMINAMATH_CALUDE_sqrt_fourth_power_eq_256_l378_37865

theorem sqrt_fourth_power_eq_256 (x : ℝ) (h : (Real.sqrt x)^4 = 256) : x = 16 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_fourth_power_eq_256_l378_37865


namespace NUMINAMATH_CALUDE_vector_collinearity_l378_37894

/-- Two vectors are collinear if one is a scalar multiple of the other -/
def collinear (v w : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, v = (k * w.1, k * w.2) ∨ w = (k * v.1, k * v.2)

theorem vector_collinearity :
  let m : ℝ × ℝ := (0, -2)
  let n : ℝ × ℝ := (Real.sqrt 3, 1)
  let v : ℝ × ℝ := (-1, Real.sqrt 3)
  collinear (2 * m.1 + n.1, 2 * m.2 + n.2) v := by sorry

end NUMINAMATH_CALUDE_vector_collinearity_l378_37894


namespace NUMINAMATH_CALUDE_tens_digit_of_special_two_digit_number_l378_37826

/-- The product of digits of a two-digit number -/
def P (n : ℕ) : ℕ :=
  (n / 10) * (n % 10)

/-- The sum of digits of a two-digit number -/
def S (n : ℕ) : ℕ :=
  (n / 10) + (n % 10)

/-- A two-digit number M satisfying M = P(M) + S(M) + 6 has a tens digit of either 1 or 2 -/
theorem tens_digit_of_special_two_digit_number :
  ∀ M : ℕ, 
    (10 ≤ M ∧ M < 100) →  -- M is a two-digit number
    (M = P M + S M + 6) →  -- M satisfies the special condition
    (M / 10 = 1 ∨ M / 10 = 2) :=  -- The tens digit is either 1 or 2
by sorry

end NUMINAMATH_CALUDE_tens_digit_of_special_two_digit_number_l378_37826


namespace NUMINAMATH_CALUDE_sean_needs_six_packs_l378_37832

/-- The number of light bulbs Sean needs to replace in each room --/
def bulbs_per_room : List Nat := [2, 1, 1, 4]

/-- The number of bulbs per pack --/
def bulbs_per_pack : Nat := 2

/-- The fraction of the total bulbs needed for the garage --/
def garage_fraction : Rat := 1/2

/-- Theorem: Sean needs 6 packs of light bulbs --/
theorem sean_needs_six_packs :
  let total_bulbs := (List.sum bulbs_per_room) + ⌈(List.sum bulbs_per_room : Rat) * garage_fraction⌉
  ⌈(total_bulbs : Rat) / bulbs_per_pack⌉ = 6 := by
  sorry

end NUMINAMATH_CALUDE_sean_needs_six_packs_l378_37832


namespace NUMINAMATH_CALUDE_smallest_integer_inequality_l378_37847

theorem smallest_integer_inequality (x y z w : ℝ) :
  ∃ (n : ℕ), (x^2 + y^2 + z^2 + w^2)^2 ≤ n * (x^4 + y^4 + z^4 + w^4) ∧
  ∀ (m : ℕ), m < n → ∃ (a b c d : ℝ), (a^2 + b^2 + c^2 + d^2)^2 > m * (a^4 + b^4 + c^4 + d^4) :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_inequality_l378_37847


namespace NUMINAMATH_CALUDE_min_value_theorem_l378_37885

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + 1/b = 2) :
  (∀ x y : ℝ, x > 0 → y > 0 → x + 1/y = 2 → 4/x + y ≥ 9/2) ∧
  (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ x + 1/y = 2 ∧ 4/x + y = 9/2) :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l378_37885


namespace NUMINAMATH_CALUDE_restaurant_sales_restaurant_sales_proof_l378_37866

/-- Calculates the total sales of a restaurant given the number of meals sold at different price points. -/
theorem restaurant_sales (meals_at_8 meals_at_10 meals_at_4 : ℕ) 
  (price_8 price_10 price_4 : ℕ) : ℕ :=
  let total_sales := meals_at_8 * price_8 + meals_at_10 * price_10 + meals_at_4 * price_4
  total_sales

/-- Proves that the restaurant's total sales for the day is $210. -/
theorem restaurant_sales_proof :
  restaurant_sales 10 5 20 8 10 4 = 210 := by
  sorry

end NUMINAMATH_CALUDE_restaurant_sales_restaurant_sales_proof_l378_37866


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l378_37848

theorem partial_fraction_decomposition :
  ∃! (A B C : ℝ),
    ∀ (x : ℝ), x ≠ 4 ∧ x ≠ 3 ∧ x ≠ 5 →
      (x^2 - 5) / ((x - 4) * (x - 3) * (x - 5)) =
      A / (x - 4) + B / (x - 3) + C / (x - 5) ↔
      A = -11 ∧ B = 2 ∧ C = 10 :=
by sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l378_37848


namespace NUMINAMATH_CALUDE_intersection_sum_zero_l378_37895

-- Define the parabolas
def parabola1 (x y : ℝ) : Prop := y = (x - 2)^2 + 1
def parabola2 (x y : ℝ) : Prop := x - 1 = (y + 2)^2

-- Define the intersection points
def intersection_points : Set (ℝ × ℝ) :=
  {p | parabola1 p.1 p.2 ∧ parabola2 p.1 p.2}

-- Theorem statement
theorem intersection_sum_zero :
  ∃ (x₁ y₁ x₂ y₂ x₃ y₃ x₄ y₄ : ℝ),
    (x₁, y₁) ∈ intersection_points ∧
    (x₂, y₂) ∈ intersection_points ∧
    (x₃, y₃) ∈ intersection_points ∧
    (x₄, y₄) ∈ intersection_points ∧
    (x₁, y₁) ≠ (x₂, y₂) ∧
    (x₁, y₁) ≠ (x₃, y₃) ∧
    (x₁, y₁) ≠ (x₄, y₄) ∧
    (x₂, y₂) ≠ (x₃, y₃) ∧
    (x₂, y₂) ≠ (x₄, y₄) ∧
    (x₃, y₃) ≠ (x₄, y₄) ∧
    x₁ + x₂ + x₃ + x₄ + y₁ + y₂ + y₃ + y₄ = 0 :=
by sorry

end NUMINAMATH_CALUDE_intersection_sum_zero_l378_37895


namespace NUMINAMATH_CALUDE_market_value_calculation_l378_37813

/-- Calculates the market value of a share given its nominal value, dividend rate, and desired interest rate. -/
def marketValue (nominalValue : ℚ) (dividendRate : ℚ) (desiredInterestRate : ℚ) : ℚ :=
  (nominalValue * dividendRate) / desiredInterestRate

/-- Theorem stating that for a share with nominal value of 48, 9% dividend rate, and 12% desired interest rate, the market value is 36. -/
theorem market_value_calculation :
  marketValue 48 (9/100) (12/100) = 36 := by
  sorry

end NUMINAMATH_CALUDE_market_value_calculation_l378_37813


namespace NUMINAMATH_CALUDE_soccer_balls_count_l378_37883

/-- The cost of a football in dollars -/
def football_cost : ℝ := 35

/-- The cost of a soccer ball in dollars -/
def soccer_ball_cost : ℝ := 50

/-- The number of soccer balls in the first set -/
def soccer_balls_in_first_set : ℕ := 1

theorem soccer_balls_count : 
  3 * football_cost + soccer_balls_in_first_set * soccer_ball_cost = 155 ∧
  2 * football_cost + 3 * soccer_ball_cost = 220 →
  soccer_balls_in_first_set = 1 := by
  sorry

end NUMINAMATH_CALUDE_soccer_balls_count_l378_37883


namespace NUMINAMATH_CALUDE_lunch_special_cost_l378_37845

theorem lunch_special_cost (total_bill : ℚ) (num_people : ℕ) (h1 : total_bill = 24) (h2 : num_people = 3) :
  total_bill / num_people = 8 := by
  sorry

end NUMINAMATH_CALUDE_lunch_special_cost_l378_37845


namespace NUMINAMATH_CALUDE_xyz_product_magnitude_l378_37800

theorem xyz_product_magnitude (x y z : ℝ) 
  (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (hdistinct : x ≠ y ∧ y ≠ z ∧ z ≠ x)
  (heq : x + 1/y = y + 1/z ∧ y + 1/z = z + 1/x) : 
  |x * y * z| = 1 := by
sorry

end NUMINAMATH_CALUDE_xyz_product_magnitude_l378_37800


namespace NUMINAMATH_CALUDE_gcd_231_154_l378_37803

theorem gcd_231_154 : Nat.gcd 231 154 = 77 := by sorry

end NUMINAMATH_CALUDE_gcd_231_154_l378_37803


namespace NUMINAMATH_CALUDE_quiz_result_proof_l378_37838

theorem quiz_result_proof (total : ℕ) (correct_A : ℕ) (correct_B : ℕ) (correct_C : ℕ) 
  (all_wrong : ℕ) (all_correct : ℕ) 
  (h_total : total = 40)
  (h_A : correct_A = 10)
  (h_B : correct_B = 13)
  (h_C : correct_C = 15)
  (h_wrong : all_wrong = 15)
  (h_correct : all_correct = 1) :
  ∃ (two_correct : ℕ), two_correct = 13 ∧ 
  two_correct = total - all_wrong - all_correct - 
    (correct_A + correct_B + correct_C - 2 * all_correct - two_correct) := by
  sorry

end NUMINAMATH_CALUDE_quiz_result_proof_l378_37838


namespace NUMINAMATH_CALUDE_valid_sequences_of_length_16_l378_37889

/-- Represents a sequence of C's and D's -/
inductive CDSequence
  | C : CDSequence → CDSequence
  | D : CDSequence → CDSequence
  | empty : CDSequence

/-- Returns true if the given sequence satisfies the conditions -/
def isValidSequence (s : CDSequence) : Bool :=
  sorry

/-- Returns the length of the given sequence -/
def sequenceLength (s : CDSequence) : Nat :=
  sorry

/-- Returns the number of valid sequences of a given length -/
def countValidSequences (n : Nat) : Nat :=
  sorry

theorem valid_sequences_of_length_16 :
  countValidSequences 16 = 55 := by sorry

end NUMINAMATH_CALUDE_valid_sequences_of_length_16_l378_37889


namespace NUMINAMATH_CALUDE_parabola_tangent_to_line_l378_37822

-- Define the parabola and line
def parabola (b x : ℝ) : ℝ := b * x^2 + 4
def line (x : ℝ) : ℝ := 2 * x + 2

-- Define the tangency condition
def is_tangent (b : ℝ) : Prop :=
  ∃! x, parabola b x = line x

-- Theorem statement
theorem parabola_tangent_to_line :
  ∀ b : ℝ, is_tangent b → b = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_parabola_tangent_to_line_l378_37822


namespace NUMINAMATH_CALUDE_remove_number_for_average_l378_37857

theorem remove_number_for_average (list : List ℕ) (removed : ℕ) (avg : ℚ) : 
  list = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13] →
  removed = 6 →
  avg = 82/10 →
  (list.sum - removed) / (list.length - 1) = avg := by
  sorry

end NUMINAMATH_CALUDE_remove_number_for_average_l378_37857


namespace NUMINAMATH_CALUDE_number_ordering_l378_37811

def A : ℕ := 9^(9^9)
def B : ℕ := 99^9
def C : ℕ := (9^9)^9
def D : ℕ := (Nat.factorial 9)^(Nat.factorial 9)

theorem number_ordering : B < C ∧ C < A ∧ A < D := by sorry

end NUMINAMATH_CALUDE_number_ordering_l378_37811


namespace NUMINAMATH_CALUDE_unique_solution_exists_l378_37810

/-- Represents a digit from 1 to 6 -/
def Digit := Fin 6

/-- Represents a two-digit number composed of two digits -/
def TwoDigitNumber (a b : Digit) : ℕ := (a.val + 1) * 10 + (b.val + 1)

/-- The main theorem stating the existence and uniqueness of the solution -/
theorem unique_solution_exists :
  ∃! (A B C D E F : Digit),
    (TwoDigitNumber A B) ^ (C.val + 1) = (TwoDigitNumber D E) ^ (F.val + 1) ∧
    A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧ A ≠ F ∧
    B ≠ C ∧ B ≠ D ∧ B ≠ E ∧ B ≠ F ∧
    C ≠ D ∧ C ≠ E ∧ C ≠ F ∧
    D ≠ E ∧ D ≠ F ∧
    E ≠ F :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_exists_l378_37810


namespace NUMINAMATH_CALUDE_M_intersect_N_equals_nonnegative_reals_l378_37876

-- Define the sets M and N
def M : Set ℝ := {x | ∃ y, y = Real.sqrt x}
def N : Set ℝ := {y | ∃ x, y = x^2 - 2}

-- State the theorem
theorem M_intersect_N_equals_nonnegative_reals :
  M ∩ N = Set.Ici (0 : ℝ) := by sorry

end NUMINAMATH_CALUDE_M_intersect_N_equals_nonnegative_reals_l378_37876


namespace NUMINAMATH_CALUDE_max_m_quadratic_inequality_l378_37872

theorem max_m_quadratic_inequality (a b c : ℝ) (h_real_roots : b^2 - 4*a*c ≥ 0) :
  ∃ (m : ℝ), m = 9/8 ∧ 
  (∀ (k : ℝ), ((a-b)^2 + (b-c)^2 + (c-a)^2 ≥ k*a^2) → k ≤ m) ∧
  ((a-b)^2 + (b-c)^2 + (c-a)^2 ≥ m*a^2) := by
  sorry

end NUMINAMATH_CALUDE_max_m_quadratic_inequality_l378_37872


namespace NUMINAMATH_CALUDE_segment_length_is_15_l378_37805

/-- The length of a vertical line segment is the absolute difference of y-coordinates -/
def vertical_segment_length (y1 y2 : ℝ) : ℝ := |y2 - y1|

/-- Proof that the length of the segment with endpoints (3, 5) and (3, 20) is 15 units -/
theorem segment_length_is_15 : 
  vertical_segment_length 5 20 = 15 := by
  sorry

end NUMINAMATH_CALUDE_segment_length_is_15_l378_37805


namespace NUMINAMATH_CALUDE_symmetric_points_range_l378_37853

open Set
open Function
open Real

noncomputable def f (x : ℝ) := Real.exp x
noncomputable def g (a : ℝ) (x : ℝ) := a * x^2 - a * x

theorem symmetric_points_range (a : ℝ) :
  (∃ x₁ x₂ : ℝ, 
    f x₁ = g a x₁ ∧ 
    f x₂ = g a x₂ ∧ 
    x₁ = f x₂ ∧ 
    x₂ = f x₁) →
  a ∈ (Ioo 0 1 ∪ Ioi 1) :=
sorry

end NUMINAMATH_CALUDE_symmetric_points_range_l378_37853


namespace NUMINAMATH_CALUDE_shop_profit_days_l378_37839

theorem shop_profit_days (mean_profit : ℝ) (first_15_mean : ℝ) (last_15_mean : ℝ)
  (h1 : mean_profit = 350)
  (h2 : first_15_mean = 245)
  (h3 : last_15_mean = 455) :
  (mean_profit * (15 + 15) = first_15_mean * 15 + last_15_mean * 15) → (15 + 15 = 30) :=
by sorry

end NUMINAMATH_CALUDE_shop_profit_days_l378_37839


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l378_37862

universe u

def U : Set ℕ := {0, 1, 2, 3, 4}
def M : Set ℕ := {0, 1, 2}
def N : Set ℕ := {2, 3}

theorem complement_intersection_theorem :
  (U \ M) ∩ N = {3} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l378_37862


namespace NUMINAMATH_CALUDE_ferris_wheel_capacity_l378_37873

/-- The number of seats on the Ferris wheel -/
def num_seats : ℕ := 14

/-- The total number of people the Ferris wheel can hold -/
def total_people : ℕ := 84

/-- The number of people each seat can hold -/
def people_per_seat : ℕ := total_people / num_seats

theorem ferris_wheel_capacity : people_per_seat = 6 := by
  sorry

end NUMINAMATH_CALUDE_ferris_wheel_capacity_l378_37873


namespace NUMINAMATH_CALUDE_eleven_to_fourth_l378_37860

theorem eleven_to_fourth (n : ℕ) (h : n = 4) : 11^n = 14641 := by
  have h1 : 11 = 10 + 1 := by rfl
  sorry

end NUMINAMATH_CALUDE_eleven_to_fourth_l378_37860


namespace NUMINAMATH_CALUDE_pot_contribution_proof_l378_37874

theorem pot_contribution_proof (total_people : Nat) (first_place_percent : Real) 
  (third_place_amount : Real) : 
  total_people = 8 → 
  first_place_percent = 0.8 → 
  third_place_amount = 4 → 
  ∃ (individual_contribution : Real),
    individual_contribution = 5 ∧ 
    individual_contribution * total_people = third_place_amount / ((1 - first_place_percent) / 2) :=
by sorry

end NUMINAMATH_CALUDE_pot_contribution_proof_l378_37874


namespace NUMINAMATH_CALUDE_mary_baking_cake_l378_37831

theorem mary_baking_cake (total_flour sugar_needed : ℕ) 
  (h1 : total_flour = 11)
  (h2 : sugar_needed = 7)
  (h3 : total_flour - flour_put_in = sugar_needed + 2) :
  flour_put_in = 2 :=
by sorry

end NUMINAMATH_CALUDE_mary_baking_cake_l378_37831


namespace NUMINAMATH_CALUDE_exists_triangle_area_not_greater_than_two_l378_37878

-- Define a lattice point type
structure LatticePoint where
  x : Int
  y : Int

-- Define the condition for a lattice point to be within the 5x5 grid
def isWithinGrid (p : LatticePoint) : Prop :=
  abs p.x ≤ 2 ∧ abs p.y ≤ 2

-- Define a function to calculate the area of a triangle given three points
def triangleArea (p1 p2 p3 : LatticePoint) : ℚ :=
  let x1 := p1.x
  let y1 := p1.y
  let x2 := p2.x
  let y2 := p2.y
  let x3 := p3.x
  let y3 := p3.y
  abs ((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2)

-- Define the condition for three points to be non-collinear
def nonCollinear (p1 p2 p3 : LatticePoint) : Prop :=
  triangleArea p1 p2 p3 ≠ 0

-- Main theorem
theorem exists_triangle_area_not_greater_than_two 
  (points : Fin 6 → LatticePoint)
  (h_within_grid : ∀ i, isWithinGrid (points i))
  (h_non_collinear : ∀ i j k, i ≠ j → j ≠ k → i ≠ k → nonCollinear (points i) (points j) (points k)) :
  ∃ i j k, i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ triangleArea (points i) (points j) (points k) ≤ 2 :=
sorry

end NUMINAMATH_CALUDE_exists_triangle_area_not_greater_than_two_l378_37878


namespace NUMINAMATH_CALUDE_max_value_complex_l378_37815

theorem max_value_complex (z : ℂ) (h : Complex.abs z = 1) :
  Complex.abs (z^3 + 3*z + Complex.I*2) ≤ 3 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_max_value_complex_l378_37815


namespace NUMINAMATH_CALUDE_red_notebook_cost_l378_37834

/-- Proves that the cost of each red notebook is 4 dollars --/
theorem red_notebook_cost (total_spent : ℕ) (total_notebooks : ℕ) (red_notebooks : ℕ) 
  (green_notebooks : ℕ) (green_cost : ℕ) (blue_cost : ℕ) :
  total_spent = 37 →
  total_notebooks = 12 →
  red_notebooks = 3 →
  green_notebooks = 2 →
  green_cost = 2 →
  blue_cost = 3 →
  (total_spent - (green_notebooks * green_cost + (total_notebooks - red_notebooks - green_notebooks) * blue_cost)) / red_notebooks = 4 := by
sorry

end NUMINAMATH_CALUDE_red_notebook_cost_l378_37834


namespace NUMINAMATH_CALUDE_iceland_visitors_l378_37891

theorem iceland_visitors (total : ℕ) (norway : ℕ) (both : ℕ) (neither : ℕ) :
  total = 60 →
  norway = 23 →
  both = 31 →
  neither = 33 →
  ∃ iceland : ℕ, iceland = 35 ∧ total = iceland + norway - both + neither :=
by sorry

end NUMINAMATH_CALUDE_iceland_visitors_l378_37891


namespace NUMINAMATH_CALUDE_min_triangles_in_square_l378_37841

/-- Represents a square with points and its triangulation -/
structure SquareWithPoints where
  k : ℕ
  points : Finset (ℝ × ℝ)
  triangles : Finset (Finset (ℝ × ℝ))

/-- Predicate to check if a triangulation is valid -/
def ValidTriangulation (s : SquareWithPoints) : Prop :=
  (s.k > 2) ∧
  (s.points.card = s.k) ∧
  (∀ t ∈ s.triangles, (t ∩ s.points).card ≤ 1)

/-- The minimum number of triangles needed -/
def MinTriangles (s : SquareWithPoints) : ℕ := s.k + 1

/-- Theorem stating the minimum number of triangles needed -/
theorem min_triangles_in_square (s : SquareWithPoints) 
  (h : ValidTriangulation s) : 
  s.triangles.card ≥ MinTriangles s :=
sorry

end NUMINAMATH_CALUDE_min_triangles_in_square_l378_37841


namespace NUMINAMATH_CALUDE_order_of_abc_l378_37896

theorem order_of_abc : 
  let a : ℝ := (Real.exp 0.6)⁻¹
  let b : ℝ := 0.4
  let c : ℝ := (Real.log 1.4) / 1.4
  c < b ∧ b < a := by sorry

end NUMINAMATH_CALUDE_order_of_abc_l378_37896


namespace NUMINAMATH_CALUDE_polynomial_expansion_l378_37864

theorem polynomial_expansion (x : ℝ) :
  (3 * x^3 - 2 * x + 4) * (4 * x^2 - 3 * x + 5) =
  12 * x^5 - 9 * x^4 + 7 * x^3 + 10 * x^2 - 2 * x + 20 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_expansion_l378_37864


namespace NUMINAMATH_CALUDE_noon_temperature_l378_37840

theorem noon_temperature 
  (morning_temp : ℤ) 
  (temp_drop : ℤ) 
  (h1 : morning_temp = 3) 
  (h2 : temp_drop = 9) : 
  morning_temp - temp_drop = -6 := by
sorry

end NUMINAMATH_CALUDE_noon_temperature_l378_37840


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l378_37856

/-- A geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- Main theorem -/
theorem geometric_sequence_problem (a : ℕ → ℝ) 
    (h_geom : GeometricSequence a)
    (h_prod : a 7 * a 9 = 4)
    (h_a4 : a 4 = 1) :
    a 12 = 4 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l378_37856


namespace NUMINAMATH_CALUDE_christmas_on_thursday_l378_37886

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents dates in November and December -/
structure Date where
  month : Nat
  day : Nat

/-- Returns the day of the week for a given date, assuming November 27 is a Thursday -/
def dayOfWeek (date : Date) : DayOfWeek :=
  sorry

theorem christmas_on_thursday (thanksgiving : Date)
    (h1 : thanksgiving.month = 11)
    (h2 : thanksgiving.day = 27)
    (h3 : dayOfWeek thanksgiving = DayOfWeek.Thursday) :
    dayOfWeek ⟨12, 25⟩ = DayOfWeek.Thursday :=
  sorry

end NUMINAMATH_CALUDE_christmas_on_thursday_l378_37886


namespace NUMINAMATH_CALUDE_no_perfect_square_in_sequence_l378_37861

theorem no_perfect_square_in_sequence : ¬∃ (k n : ℕ), 3 * k - 1 = n ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_no_perfect_square_in_sequence_l378_37861


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l378_37820

theorem quadratic_equation_solution (a b : ℕ+) : 
  (∃ x : ℝ, x > 0 ∧ x^2 + 12*x = 73 ∧ x = Real.sqrt a - b) → a + b = 115 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l378_37820


namespace NUMINAMATH_CALUDE_triangles_similar_l378_37828

/-- A triangle with side lengths a, b, and c. -/
structure Triangle :=
  (a b c : ℝ)
  (positive_a : a > 0)
  (positive_b : b > 0)
  (positive_c : c > 0)
  (triangle_inequality_ab : a + b > c)
  (triangle_inequality_bc : b + c > a)
  (triangle_inequality_ca : c + a > b)

/-- The condition that a + c = 2b for a triangle. -/
def condition1 (t : Triangle) : Prop :=
  t.a + t.c = 2 * t.b

/-- The condition that b + 2c = 5a for a triangle. -/
def condition2 (t : Triangle) : Prop :=
  t.b + 2 * t.c = 5 * t.a

/-- Two triangles are similar. -/
def similar (t1 t2 : Triangle) : Prop :=
  ∃ k : ℝ, k > 0 ∧ t1.a = k * t2.a ∧ t1.b = k * t2.b ∧ t1.c = k * t2.c

/-- 
Theorem: If two triangles satisfy both condition1 and condition2, then they are similar.
-/
theorem triangles_similar (t1 t2 : Triangle) 
  (h1 : condition1 t1) (h2 : condition1 t2) 
  (h3 : condition2 t1) (h4 : condition2 t2) : 
  similar t1 t2 := by
  sorry

end NUMINAMATH_CALUDE_triangles_similar_l378_37828


namespace NUMINAMATH_CALUDE_rectangle_area_l378_37807

theorem rectangle_area (length width : ℝ) :
  (2 * (length + width) = 48) →
  (length = width + 2) →
  (length * width = 143) :=
by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l378_37807


namespace NUMINAMATH_CALUDE_no_integer_m_for_single_solution_l378_37804

theorem no_integer_m_for_single_solution :
  ¬ ∃ (m : ℤ), ∃! (x : ℝ), 36 * x^2 - m * x - 4 = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_m_for_single_solution_l378_37804


namespace NUMINAMATH_CALUDE_sum_of_powers_of_i_is_zero_l378_37825

/-- The imaginary unit i -/
def i : ℂ := Complex.I

/-- Theorem stating that i^1234 + i^1235 + i^1236 + i^1237 = 0 -/
theorem sum_of_powers_of_i_is_zero : i^1234 + i^1235 + i^1236 + i^1237 = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_powers_of_i_is_zero_l378_37825


namespace NUMINAMATH_CALUDE_tangent_line_to_ellipse_l378_37870

theorem tangent_line_to_ellipse (m : ℝ) :
  (∃! x y : ℝ, y = m * x + 2 ∧ x^2 + 9 * y^2 = 1) →
  m^2 = 35/9 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_to_ellipse_l378_37870


namespace NUMINAMATH_CALUDE_find_p_l378_37855

def U : Set ℕ := {1, 2, 3, 4}

def M (p : ℝ) : Set ℕ := {x ∈ U | x^2 - 5*x + p = 0}

theorem find_p : ∃ p : ℝ, (U \ M p) = {2, 3} → p = 4 := by
  sorry

end NUMINAMATH_CALUDE_find_p_l378_37855


namespace NUMINAMATH_CALUDE_show_completion_time_l378_37843

theorem show_completion_time (num_episodes : ℕ) (episode_length : ℕ) (daily_watch_time : ℕ) : 
  num_episodes = 20 → 
  episode_length = 30 → 
  daily_watch_time = 120 → 
  (num_episodes * episode_length) / daily_watch_time = 5 :=
by sorry

end NUMINAMATH_CALUDE_show_completion_time_l378_37843


namespace NUMINAMATH_CALUDE_zacks_marbles_l378_37836

theorem zacks_marbles (friend1 friend2 friend3 friend4 friend5 friend6 remaining : ℕ) 
  (h1 : friend1 = 20)
  (h2 : friend2 = 30)
  (h3 : friend3 = 35)
  (h4 : friend4 = 25)
  (h5 : friend5 = 28)
  (h6 : friend6 = 40)
  (h7 : remaining = 7) :
  friend1 + friend2 + friend3 + friend4 + friend5 + friend6 + remaining = 185 := by
  sorry

end NUMINAMATH_CALUDE_zacks_marbles_l378_37836


namespace NUMINAMATH_CALUDE_power_function_through_point_l378_37880

theorem power_function_through_point (f : ℝ → ℝ) (n : ℝ) :
  (∀ x, f x = x^n) →  -- f is a power function
  f 2 = 8 →           -- f passes through (2,8)
  n = 3 :=            -- the power must be 3
by
  sorry


end NUMINAMATH_CALUDE_power_function_through_point_l378_37880


namespace NUMINAMATH_CALUDE_exam_duration_l378_37829

/-- Represents a time on a clock face -/
structure ClockTime where
  hours : ℝ
  minutes : ℝ
  valid : 0 ≤ hours ∧ hours < 12 ∧ 0 ≤ minutes ∧ minutes < 60

/-- Checks if two clock times are equivalent when hour and minute hands are swapped -/
def equivalent_when_swapped (t1 t2 : ClockTime) : Prop :=
  t1.hours = t2.minutes / 5 ∧ t1.minutes = t2.hours * 5

/-- The main theorem statement -/
theorem exam_duration :
  ∀ (start_time end_time : ClockTime),
    9 ≤ start_time.hours ∧ start_time.hours < 10 →
    1 ≤ end_time.hours ∧ end_time.hours < 2 →
    equivalent_when_swapped start_time end_time →
    end_time.hours - start_time.hours + (end_time.minutes - start_time.minutes) / 60 = 60 / 13 :=
sorry

end NUMINAMATH_CALUDE_exam_duration_l378_37829


namespace NUMINAMATH_CALUDE_remaining_payment_example_l378_37879

/-- Given a deposit percentage and deposit amount, calculates the remaining amount to be paid -/
def remaining_payment (deposit_percentage : ℚ) (deposit_amount : ℚ) : ℚ :=
  let total_cost := deposit_amount / deposit_percentage
  total_cost - deposit_amount

/-- Theorem stating that the remaining payment is $1350 given a 10% deposit of $150 -/
theorem remaining_payment_example : remaining_payment (10 / 100) 150 = 1350 := by
  sorry

end NUMINAMATH_CALUDE_remaining_payment_example_l378_37879


namespace NUMINAMATH_CALUDE_expression_evaluation_l378_37899

theorem expression_evaluation : (4 * 6) / (12 * 14) * (8 * 12 * 14) / (4 * 6 * 8) = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l378_37899


namespace NUMINAMATH_CALUDE_B_subset_A_iff_m_in_range_l378_37875

-- Define set A
def A : Set ℝ := {x | (2 * x) / (x - 2) < 1}

-- Define set B (parameterized by m)
def B (m : ℝ) : Set ℝ := {x | x^2 - (2*m + 1)*x + m^2 + m < 0}

-- Theorem statement
theorem B_subset_A_iff_m_in_range :
  ∀ m : ℝ, (B m) ⊆ A ↔ -2 ≤ m ∧ m ≤ 1 := by sorry

end NUMINAMATH_CALUDE_B_subset_A_iff_m_in_range_l378_37875


namespace NUMINAMATH_CALUDE_correlation_coefficient_relationship_l378_37850

/-- Represents the starting age of smoking -/
def X : Type := ℕ

/-- Represents the relative risk of lung cancer for different starting ages -/
def Y : Type := ℝ

/-- Represents the number of cigarettes smoked per day -/
def U : Type := ℕ

/-- Represents the relative risk of lung cancer for different numbers of cigarettes -/
def V : Type := ℝ

/-- The linear correlation coefficient between X and Y -/
def r1 : ℝ := sorry

/-- The linear correlation coefficient between U and V -/
def r2 : ℝ := sorry

/-- Theorem stating the relationship between r1 and r2 -/
theorem correlation_coefficient_relationship : r1 < 0 ∧ 0 < r2 := by sorry

end NUMINAMATH_CALUDE_correlation_coefficient_relationship_l378_37850


namespace NUMINAMATH_CALUDE_ratio_calculation_l378_37814

theorem ratio_calculation : 
  let numerator := (14^4 + 484) * (26^4 + 484) * (38^4 + 484) * (50^4 + 484) * (62^4 + 484)
  let denominator := (8^4 + 484) * (20^4 + 484) * (32^4 + 484) * (44^4 + 484) * (56^4 + 484)
  numerator / denominator = -423 := by
sorry

end NUMINAMATH_CALUDE_ratio_calculation_l378_37814


namespace NUMINAMATH_CALUDE_simplify_power_expression_l378_37808

theorem simplify_power_expression (x : ℝ) : (3 * x^4)^4 = 81 * x^16 := by sorry

end NUMINAMATH_CALUDE_simplify_power_expression_l378_37808


namespace NUMINAMATH_CALUDE_circle_and_line_problem_l378_37806

-- Define the circle C
def circle_C (x y : ℝ) : Prop :=
  (x - 1)^2 + (y - 1)^2 = 4

-- Define the lines l
def line_l1 (x y : ℝ) : Prop := x = 2
def line_l2 (x y : ℝ) : Prop := 4*x + 3*y = 2

-- Theorem statement
theorem circle_and_line_problem :
  ∃ (center_x center_y : ℝ),
    -- Circle C passes through A(1,3) and B(-1,1)
    circle_C 1 3 ∧ circle_C (-1) 1 ∧
    -- Center of the circle is on the line y = x
    center_y = center_x ∧
    -- Circle equation
    (∀ x y, circle_C x y ↔ (x - center_x)^2 + (y - center_y)^2 = 4) ∧
    -- Line l passes through (2,-2)
    (line_l1 2 (-2) ∨ line_l2 2 (-2)) ∧
    -- Line l intersects circle C with chord length 2√3
    (∃ x1 y1 x2 y2,
      ((line_l1 x1 y1 ∧ line_l1 x2 y2) ∨ (line_l2 x1 y1 ∧ line_l2 x2 y2)) ∧
      circle_C x1 y1 ∧ circle_C x2 y2 ∧
      (x2 - x1)^2 + (y2 - y1)^2 = 12) :=
by
  sorry -- Proof omitted

end NUMINAMATH_CALUDE_circle_and_line_problem_l378_37806


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l378_37802

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def f_definition (f : ℝ → ℝ) : Prop :=
  ∀ x ≥ 0, f x = x^2 - 4*x

theorem solution_set_of_inequality (f : ℝ → ℝ) 
  (h_even : is_even_function f) 
  (h_def : f_definition f) :
  {x : ℝ | f (x + 2) < 5} = Set.Ioo (-7) 3 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l378_37802


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l378_37835

/-- Two arithmetic sequences and their sum sequences -/
def arithmetic_sequences (a b : ℕ → ℚ) (S T : ℕ → ℚ) : Prop :=
  ∀ n, S n = (n / 2) * (a 1 + a n) ∧ T n = (n / 2) * (b 1 + b n)

/-- The ratio of sums condition -/
def sum_ratio_condition (S T : ℕ → ℚ) : Prop :=
  ∀ n, S n / T n = (7 * n) / (n + 3)

theorem arithmetic_sequence_ratio
  (a b : ℕ → ℚ) (S T : ℕ → ℚ)
  (h1 : arithmetic_sequences a b S T)
  (h2 : sum_ratio_condition S T) :
  a 5 / b 5 = 21 / 4 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l378_37835


namespace NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l378_37851

-- Define the set A
def A : Set ℝ := {y | ∃ x ∈ Set.Icc (-1/2 : ℝ) 2, y = x^2 - (3/2)*x + 1}

-- Define the set B (parameterized by m)
def B (m : ℝ) : Set ℝ := {x | |x - m| ≥ 1}

-- Define the range of m
def m_range : Set ℝ := Set.Iic (-9/16) ∪ Set.Ici 3

-- Theorem statement
theorem necessary_not_sufficient_condition (m : ℝ) :
  (∀ t, t ∈ A → t ∈ B m) ∧ (∃ t, t ∈ B m ∧ t ∉ A) ↔ m ∈ m_range :=
sorry

end NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l378_37851


namespace NUMINAMATH_CALUDE_count_multiples_of_7_ending_in_7_less_than_150_l378_37827

def multiples_of_7_ending_in_7 (n : ℕ) : ℕ :=
  (n / 70 : ℕ)

theorem count_multiples_of_7_ending_in_7_less_than_150 :
  multiples_of_7_ending_in_7 150 = 2 := by sorry

end NUMINAMATH_CALUDE_count_multiples_of_7_ending_in_7_less_than_150_l378_37827


namespace NUMINAMATH_CALUDE_log_abs_eq_sin_roots_l378_37867

noncomputable def log_abs (x : ℝ) : ℝ := Real.log (abs x)

theorem log_abs_eq_sin_roots :
  let f (x : ℝ) := log_abs x - Real.sin x
  ∃ (S : Finset ℝ), (∀ x ∈ S, f x = 0) ∧ S.card = 10 ∧ 
    (∀ y : ℝ, f y = 0 → y ∈ S) := by sorry

end NUMINAMATH_CALUDE_log_abs_eq_sin_roots_l378_37867


namespace NUMINAMATH_CALUDE_sum_of_prime_and_odd_l378_37852

theorem sum_of_prime_and_odd (a b : ℕ) : 
  Nat.Prime a → Odd b → a^2 + b = 2009 → a + b = 2007 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_prime_and_odd_l378_37852


namespace NUMINAMATH_CALUDE_sphere_radius_is_4_l378_37812

/-- Represents a cylindrical container with spheres -/
structure Container where
  initialHeight : ℝ
  sphereRadius : ℝ
  numSpheres : ℕ

/-- Calculates the final height of water in the container after adding spheres -/
def finalHeight (c : Container) : ℝ :=
  c.initialHeight + c.sphereRadius * 2

/-- The problem statement -/
theorem sphere_radius_is_4 (c : Container) :
  c.initialHeight = 8 ∧
  c.numSpheres = 3 ∧
  finalHeight c = c.initialHeight + c.sphereRadius * 2 →
  c.sphereRadius = 4 := by
  sorry


end NUMINAMATH_CALUDE_sphere_radius_is_4_l378_37812


namespace NUMINAMATH_CALUDE_number_of_apples_l378_37897

/-- Given a box of fruit with the following properties:
  * The total number of fruit pieces is 56
  * One-fourth of the fruit are oranges
  * The number of peaches is half the number of oranges
  * The number of apples is five times the number of peaches
  This theorem proves that the number of apples in the box is 35. -/
theorem number_of_apples (total : ℕ) (oranges peaches apples : ℕ) : 
  total = 56 →
  oranges = total / 4 →
  peaches = oranges / 2 →
  apples = 5 * peaches →
  apples = 35 := by
  sorry

end NUMINAMATH_CALUDE_number_of_apples_l378_37897


namespace NUMINAMATH_CALUDE_triangle_properties_l378_37830

noncomputable section

open Real

/-- Given a triangle ABC with D as the midpoint of AB, prove that under certain conditions,
    angle C is π/3 and the maximum value of CD²/(a²+b²) is 3/8. -/
theorem triangle_properties (A B C : ℝ) (a b c : ℝ) (D : ℝ × ℝ) :
  0 < A ∧ A < π →
  0 < B ∧ B < π →
  0 < C ∧ C < π →
  A + B + C = π →
  a > 0 ∧ b > 0 ∧ c > 0 →
  b - c * cos A = a * (sqrt 3 * sin C - 1) →
  sin (A + B) * cos (C - π / 6) = 3 / 4 →
  D = ((cos A + cos B) / 2, (sin A + sin B) / 2) →
  C = π / 3 ∧
  (∀ (x y : ℝ), x > 0 → y > 0 → (x * x + y * y + x * y) / (4 * (x * x + y * y)) ≤ 3 / 8) :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l378_37830


namespace NUMINAMATH_CALUDE_power_function_value_l378_37863

-- Define the power function type
def PowerFunction := ℝ → ℝ

-- Define the property of passing through the point (3, √3/3)
def PassesThroughPoint (f : PowerFunction) : Prop :=
  f 3 = Real.sqrt 3 / 3

-- State the theorem
theorem power_function_value (f : PowerFunction) 
  (h : PassesThroughPoint f) : f (1/4) = 2 := by
  sorry

end NUMINAMATH_CALUDE_power_function_value_l378_37863


namespace NUMINAMATH_CALUDE_binomial_probability_l378_37823

/-- A binomially distributed random variable with given mean and variance -/
structure BinomialRV where
  n : ℕ
  p : ℝ
  mean_eq : n * p = 5 / 3
  var_eq : n * p * (1 - p) = 10 / 9

/-- The probability mass function for a binomial distribution -/
def binomialPMF (rv : BinomialRV) (k : ℕ) : ℝ :=
  (Nat.choose rv.n k) * (rv.p ^ k) * ((1 - rv.p) ^ (rv.n - k))

theorem binomial_probability (rv : BinomialRV) : 
  binomialPMF rv 4 = 10 / 243 := by
  sorry

end NUMINAMATH_CALUDE_binomial_probability_l378_37823
