import Mathlib

namespace NUMINAMATH_CALUDE_smallest_of_five_consecutive_odds_with_product_l1867_186759

def is_odd (n : ℕ) : Prop := ∃ k, n = 2*k + 1

def consecutive_odds (a b c d e : ℕ) : Prop :=
  is_odd a ∧ is_odd b ∧ is_odd c ∧ is_odd d ∧ is_odd e ∧
  b = a + 2 ∧ c = b + 2 ∧ d = c + 2 ∧ e = d + 2

theorem smallest_of_five_consecutive_odds_with_product (a b c d e : ℕ) :
  consecutive_odds a b c d e →
  a * b * c * d * e = 135135 →
  a = 7 :=
sorry

end NUMINAMATH_CALUDE_smallest_of_five_consecutive_odds_with_product_l1867_186759


namespace NUMINAMATH_CALUDE_percentage_lost_is_25_percent_l1867_186790

/-- Represents the number of kettles of hawks -/
def num_kettles : ℕ := 6

/-- Represents the average number of pregnancies per kettle -/
def pregnancies_per_kettle : ℕ := 15

/-- Represents the number of babies per pregnancy -/
def babies_per_pregnancy : ℕ := 4

/-- Represents the expected number of babies this season -/
def expected_babies : ℕ := 270

/-- Calculates the percentage of baby hawks lost -/
def percentage_lost : ℚ :=
  let total_babies := num_kettles * pregnancies_per_kettle * babies_per_pregnancy
  let lost_babies := total_babies - expected_babies
  (lost_babies : ℚ) / (total_babies : ℚ) * 100

/-- Theorem stating that the percentage of baby hawks lost is 25% -/
theorem percentage_lost_is_25_percent : percentage_lost = 25 := by
  sorry

end NUMINAMATH_CALUDE_percentage_lost_is_25_percent_l1867_186790


namespace NUMINAMATH_CALUDE_possible_values_of_a_l1867_186796

theorem possible_values_of_a : 
  {a : ℤ | ∃ b c : ℤ, ∀ x : ℝ, (x - a) * (x - 12) + 1 = (x + b) * (x + c)} = {10, 14} := by
  sorry

end NUMINAMATH_CALUDE_possible_values_of_a_l1867_186796


namespace NUMINAMATH_CALUDE_root_sum_product_l1867_186758

theorem root_sum_product (p q : ℝ) : 
  (Complex.I : ℂ).re = 0 ∧ (Complex.I : ℂ).im = 1 →
  (2 : ℂ) * (-3 + 2 * Complex.I)^2 + p * (-3 + 2 * Complex.I) + q = 0 →
  p + q = 38 := by sorry

end NUMINAMATH_CALUDE_root_sum_product_l1867_186758


namespace NUMINAMATH_CALUDE_compare_function_values_l1867_186728

/-- Given a quadratic function f(x) = x^2 - bx + c with specific properties,
    prove that f(b^x) ≤ f(c^x) for all real x. -/
theorem compare_function_values (b c : ℝ) (f : ℝ → ℝ) (h1 : ∀ x, f x = x^2 - b*x + c) 
    (h2 : ∀ x, f (1 - x) = f (1 + x)) (h3 : f 0 = 3) : 
    ∀ x, f (b^x) ≤ f (c^x) := by
  sorry

end NUMINAMATH_CALUDE_compare_function_values_l1867_186728


namespace NUMINAMATH_CALUDE_ellipse_right_triangle_distance_to_x_axis_l1867_186704

/-- An ellipse with semi-major axis 4 and semi-minor axis 3 -/
structure Ellipse :=
  (x y : ℝ)
  (eq : x^2 / 16 + y^2 / 9 = 1)

/-- The foci of the ellipse -/
def foci (e : Ellipse) : ℝ × ℝ := sorry

/-- A point P on the ellipse forms a right triangle with the foci -/
def right_triangle_with_foci (e : Ellipse) (p : ℝ × ℝ) : Prop := sorry

/-- The distance from a point to the x-axis -/
def distance_to_x_axis (p : ℝ × ℝ) : ℝ := sorry

theorem ellipse_right_triangle_distance_to_x_axis (e : Ellipse) (p : ℝ × ℝ) :
  p.1^2 / 16 + p.2^2 / 9 = 1 →
  right_triangle_with_foci e p →
  distance_to_x_axis p = 9/4 := by sorry

end NUMINAMATH_CALUDE_ellipse_right_triangle_distance_to_x_axis_l1867_186704


namespace NUMINAMATH_CALUDE_min_cubes_for_box_l1867_186777

/-- Proves that the minimum number of 5 cubic cm cubes required to build a box
    with dimensions 10 cm × 13 cm × 5 cm is 130. -/
theorem min_cubes_for_box (box_length box_width box_height cube_volume : ℕ)
  (h1 : box_length = 10)
  (h2 : box_width = 13)
  (h3 : box_height = 5)
  (h4 : cube_volume = 5) :
  (box_length * box_width * box_height) / cube_volume = 130 := by
  sorry

end NUMINAMATH_CALUDE_min_cubes_for_box_l1867_186777


namespace NUMINAMATH_CALUDE_well_depth_is_784_l1867_186751

/-- The depth of the well in feet -/
def well_depth : ℝ := 784

/-- The total time for the stone to fall and the sound to return, in seconds -/
def total_time : ℝ := 7.7

/-- The velocity of sound in feet per second -/
def sound_velocity : ℝ := 1120

/-- The function describing the distance fallen by the stone in t seconds -/
def stone_fall (t : ℝ) : ℝ := 16 * t^2

/-- Theorem stating that the well depth is 784 feet given the conditions -/
theorem well_depth_is_784 :
  ∃ (t_fall : ℝ), 
    t_fall > 0 ∧ 
    stone_fall t_fall = well_depth ∧
    t_fall + well_depth / sound_velocity = total_time :=
sorry

end NUMINAMATH_CALUDE_well_depth_is_784_l1867_186751


namespace NUMINAMATH_CALUDE_pen_pencil_cost_difference_l1867_186734

/-- The cost difference between a pen and a pencil -/
def cost_difference (pen_cost pencil_cost : ℝ) : ℝ := pen_cost - pencil_cost

/-- The total cost of a pen and a pencil -/
def total_cost (pen_cost pencil_cost : ℝ) : ℝ := pen_cost + pencil_cost

theorem pen_pencil_cost_difference :
  ∀ (pen_cost : ℝ),
    pencil_cost = 2 →
    total_cost pen_cost pencil_cost = 13 →
    pen_cost > pencil_cost →
    cost_difference pen_cost pencil_cost = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_pen_pencil_cost_difference_l1867_186734


namespace NUMINAMATH_CALUDE_B_minus_A_equality_l1867_186772

def A : Set ℝ := {y | ∃ x, 1/3 ≤ x ∧ x ≤ 1 ∧ y = 1/x}
def B : Set ℝ := {y | ∃ x, -1 ≤ x ∧ x ≤ 2 ∧ y = x^2 - 1}

theorem B_minus_A_equality : 
  B \ A = {x : ℝ | -1 ≤ x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_B_minus_A_equality_l1867_186772


namespace NUMINAMATH_CALUDE_scores_mode_and_median_l1867_186717

def scores : List ℕ := [80, 85, 85, 85, 90, 90, 90, 90, 95]

def mode (l : List ℕ) : ℕ := sorry

def median (l : List ℕ) : ℕ := sorry

theorem scores_mode_and_median :
  mode scores = 90 ∧ median scores = 90 := by sorry

end NUMINAMATH_CALUDE_scores_mode_and_median_l1867_186717


namespace NUMINAMATH_CALUDE_more_birds_than_nests_l1867_186710

/-- Given 6 birds and 3 nests, prove that there are 3 more birds than nests. -/
theorem more_birds_than_nests (birds : ℕ) (nests : ℕ) 
  (h1 : birds = 6) (h2 : nests = 3) : birds - nests = 3 := by
  sorry

end NUMINAMATH_CALUDE_more_birds_than_nests_l1867_186710


namespace NUMINAMATH_CALUDE_inequality_proof_l1867_186764

theorem inequality_proof (a b : ℝ) (ha : a > 1) (hb1 : 1 > b) (hb2 : b > -1) :
  a > b^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1867_186764


namespace NUMINAMATH_CALUDE_tangent_line_to_circle_l1867_186739

/-- A line y = kx is tangent to the circle x^2 + y^2 - 6x + 8 = 0 at a point in the fourth quadrant -/
def is_tangent (k : ℝ) : Prop :=
  ∃ (x y : ℝ),
    y = k * x ∧
    x^2 + y^2 - 6*x + 8 = 0 ∧
    x > 0 ∧ y < 0 ∧
    ∀ (x' y' : ℝ), y' = k * x' → (x' - x)^2 + (y' - y)^2 > 0

theorem tangent_line_to_circle (k : ℝ) :
  is_tangent k → k = -Real.sqrt 2 / 4 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_to_circle_l1867_186739


namespace NUMINAMATH_CALUDE_max_ratio_two_digit_mean_50_l1867_186716

theorem max_ratio_two_digit_mean_50 :
  ∀ x y : ℕ,
  10 ≤ x ∧ x ≤ 99 →
  10 ≤ y ∧ y ≤ 99 →
  (x + y) / 2 = 50 →
  ∀ a b : ℕ,
  10 ≤ a ∧ a ≤ 99 →
  10 ≤ b ∧ b ≤ 99 →
  (a + b) / 2 = 50 →
  x / y ≤ 9 :=
by sorry

end NUMINAMATH_CALUDE_max_ratio_two_digit_mean_50_l1867_186716


namespace NUMINAMATH_CALUDE_train_speed_l1867_186705

/-- Given a train of length 350 meters that crosses a pole in 21 seconds, its speed is 60 km/hr. -/
theorem train_speed (train_length : ℝ) (crossing_time : ℝ) (h1 : train_length = 350) (h2 : crossing_time = 21) :
  (train_length / 1000) / (crossing_time / 3600) = 60 :=
sorry

end NUMINAMATH_CALUDE_train_speed_l1867_186705


namespace NUMINAMATH_CALUDE_system_solution_existence_l1867_186781

theorem system_solution_existence (a : ℝ) :
  (∃ (x y b : ℝ), y = x^2 - a ∧ x^2 + y^2 + 8*b^2 = 4*b*(y - x) + 1) ↔ 
  a ≥ -Real.sqrt 2 - 1/4 := by sorry

end NUMINAMATH_CALUDE_system_solution_existence_l1867_186781


namespace NUMINAMATH_CALUDE_root_equation_l1867_186706

noncomputable def f (x : ℝ) : ℝ := if x < 0 then -2*x else x^2 - 1

theorem root_equation (a : ℝ) :
  ∃ (x₁ x₂ x₃ : ℝ), x₁ < x₂ ∧ x₂ < x₃ ∧
  (∀ x : ℝ, f x + 2 * Real.sqrt (1 - x^2) + |f x - 2 * Real.sqrt (1 - x^2)| - 2*a*x - 4 = 0 ↔ (x = x₁ ∨ x = x₂ ∨ x = x₃)) ∧
  x₃ - x₂ = 2*(x₂ - x₁) →
  a = (Real.sqrt 17 - 3) / 2 :=
sorry

end NUMINAMATH_CALUDE_root_equation_l1867_186706


namespace NUMINAMATH_CALUDE_hotel_tax_calculation_l1867_186702

/-- Calculates the business tax paid given revenue and tax rate -/
def business_tax (revenue : ℕ) (tax_rate : ℚ) : ℚ :=
  (revenue : ℚ) * tax_rate

theorem hotel_tax_calculation :
  let revenue : ℕ := 10000000  -- 10 million yuan
  let tax_rate : ℚ := 5 / 100   -- 5%
  business_tax revenue tax_rate = 500 := by sorry

end NUMINAMATH_CALUDE_hotel_tax_calculation_l1867_186702


namespace NUMINAMATH_CALUDE_square_plus_reciprocal_square_l1867_186724

theorem square_plus_reciprocal_square (m : ℝ) (h : m + 1/m = 6) :
  m^2 + 1/m^2 + 4 = 38 := by
  sorry

end NUMINAMATH_CALUDE_square_plus_reciprocal_square_l1867_186724


namespace NUMINAMATH_CALUDE_pencil_price_solution_l1867_186760

def pencil_price_problem (pencil_price notebook_price : ℕ) : Prop :=
  (pencil_price + notebook_price = 950) ∧ 
  (notebook_price = pencil_price + 150)

theorem pencil_price_solution : 
  ∃ (pencil_price notebook_price : ℕ), 
    pencil_price_problem pencil_price notebook_price ∧ 
    pencil_price = 400 := by
  sorry

end NUMINAMATH_CALUDE_pencil_price_solution_l1867_186760


namespace NUMINAMATH_CALUDE_holey_iff_presentable_l1867_186799

/-- A function is holey if there exists an interval free of its values -/
def IsHoley (f : ℝ → ℝ) : Prop :=
  ∃ (a b : ℝ), a < b ∧ ∀ x, a < x ∧ x < b → ∀ y, f y ≠ x

/-- A function is presentable if it can be represented as a composition of linear, inverse, and quadratic functions -/
inductive Presentable : (ℝ → ℝ) → Prop
  | linear (k b : ℝ) : Presentable (fun x ↦ k * x + b)
  | inverse : Presentable (fun x ↦ 1 / x)
  | square : Presentable (fun x ↦ x ^ 2)
  | comp {f g : ℝ → ℝ} (hf : Presentable f) (hg : Presentable g) : Presentable (f ∘ g)

/-- The main theorem statement -/
theorem holey_iff_presentable (a b c d : ℝ) 
    (h : ∀ x, x^2 + a*x + b ≠ 0 ∨ x^2 + c*x + d ≠ 0) : 
    IsHoley (fun x ↦ (x^2 + a*x + b) / (x^2 + c*x + d)) ↔ 
    Presentable (fun x ↦ (x^2 + a*x + b) / (x^2 + c*x + d)) :=
  sorry

end NUMINAMATH_CALUDE_holey_iff_presentable_l1867_186799


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_l1867_186712

theorem sufficient_but_not_necessary (p q : Prop) 
  (h : (¬p → q) ∧ ¬(q → ¬p)) : 
  (p → ¬q) ∧ ¬(¬q → p) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_l1867_186712


namespace NUMINAMATH_CALUDE_sum_squared_geq_three_l1867_186792

theorem sum_squared_geq_three (a b c : ℝ) (h : a * b + b * c + a * c = 1) :
  (a + b + c)^2 ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_squared_geq_three_l1867_186792


namespace NUMINAMATH_CALUDE_largest_x_and_fraction_l1867_186795

theorem largest_x_and_fraction (x : ℝ) (a b c d : ℤ) : 
  (7 * x / 5 - 2 = 4 / x) →
  (x = (a + b * Real.sqrt c) / d) →
  (∀ y : ℝ, (7 * y / 5 - 2 = 4 / y) → y ≤ x) →
  (x = (5 + 5 * Real.sqrt 66) / 7 ∧ a * c * d / b = 462) := by
  sorry

end NUMINAMATH_CALUDE_largest_x_and_fraction_l1867_186795


namespace NUMINAMATH_CALUDE_gravitational_force_at_distance_l1867_186719

/-- Gravitational force function -/
noncomputable def gravitational_force (k : ℝ) (d : ℝ) : ℝ := k / d^2

theorem gravitational_force_at_distance
  (k : ℝ)
  (h1 : gravitational_force k 5000 = 500)
  (h2 : k > 0) :
  gravitational_force k 25000 = 1/5 := by
  sorry

#check gravitational_force_at_distance

end NUMINAMATH_CALUDE_gravitational_force_at_distance_l1867_186719


namespace NUMINAMATH_CALUDE_min_xy_value_l1867_186709

theorem min_xy_value (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : (x + 1)⁻¹ + (y + 1)⁻¹ = (1 : ℝ) / 2) : 
  ∀ z, z = x * y → z ≥ 9 :=
by sorry

end NUMINAMATH_CALUDE_min_xy_value_l1867_186709


namespace NUMINAMATH_CALUDE_digit_difference_1250_l1867_186725

/-- The number of digits in the base-b representation of a positive integer n -/
def num_digits (n : ℕ) (b : ℕ) : ℕ :=
  Nat.log b n + 1

/-- The theorem stating the difference in number of digits between base-4 and base-9 representations of 1250 -/
theorem digit_difference_1250 :
  num_digits 1250 4 - num_digits 1250 9 = 2 := by
  sorry

end NUMINAMATH_CALUDE_digit_difference_1250_l1867_186725


namespace NUMINAMATH_CALUDE_square_land_side_length_l1867_186762

theorem square_land_side_length (area : ℝ) (is_square : Bool) : 
  area = 400 ∧ is_square = true → ∃ (side : ℝ), side * side = area ∧ side = 20 := by
  sorry

end NUMINAMATH_CALUDE_square_land_side_length_l1867_186762


namespace NUMINAMATH_CALUDE_product_unit_digit_l1867_186731

def unit_digit (n : ℕ) : ℕ := n % 10

theorem product_unit_digit : 
  unit_digit (624 * 708 * 913 * 463) = 8 := by
  sorry

end NUMINAMATH_CALUDE_product_unit_digit_l1867_186731


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l1867_186756

theorem isosceles_triangle_perimeter (x₁ x₂ : ℝ) : 
  x₁^2 - 9*x₁ + 18 = 0 →
  x₂^2 - 9*x₂ + 18 = 0 →
  x₁ ≠ x₂ →
  (x₁ + x₂ + max x₁ x₂ = 15) :=
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l1867_186756


namespace NUMINAMATH_CALUDE_olympic_medal_theorem_l1867_186742

/-- Represents the number of ways to award medals in the Olympic 100-meter finals -/
def olympic_medal_ways (total_sprinters : ℕ) (british_sprinters : ℕ) (medals : ℕ) : ℕ :=
  -- Define the function here
  sorry

/-- Theorem stating the number of ways to award medals in the given scenario -/
theorem olympic_medal_theorem :
  let total_sprinters := 10
  let british_sprinters := 4
  let medals := 3
  olympic_medal_ways total_sprinters british_sprinters medals = 912 :=
by
  sorry

end NUMINAMATH_CALUDE_olympic_medal_theorem_l1867_186742


namespace NUMINAMATH_CALUDE_range_of_m_l1867_186737

def p (m : ℝ) : Prop := 0 ≤ m ∧ m ≤ 3

def q (m : ℝ) : Prop := (m - 2) * (m - 4) ≤ 0

theorem range_of_m :
  ∀ m : ℝ, (¬(p m ∧ q m) ∧ (p m ∨ q m)) → (m ∈ Set.Icc 0 2 ∪ Set.Ioc 3 4) :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l1867_186737


namespace NUMINAMATH_CALUDE_expected_successes_eq_38_l1867_186752

/-- The probability of getting at least one 5 or 6 when throwing 3 dice -/
def p : ℚ := 19 / 27

/-- The number of experiments -/
def n : ℕ := 54

/-- A trial is successful if at least one 5 or 6 appears when throwing 3 dice -/
axiom success_definition : True

/-- The number of successful trials follows a binomial distribution -/
axiom binomial_distribution : True

/-- The expected number of successful trials in 54 experiments -/
def expected_successes : ℚ := n * p

theorem expected_successes_eq_38 : expected_successes = 38 := by
  sorry

end NUMINAMATH_CALUDE_expected_successes_eq_38_l1867_186752


namespace NUMINAMATH_CALUDE_problem_statement_l1867_186711

theorem problem_statement : 3 * 3^4 - 9^35 / 9^33 = 162 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1867_186711


namespace NUMINAMATH_CALUDE_sqrt_10_factorial_div_210_l1867_186753

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem sqrt_10_factorial_div_210 : 
  Real.sqrt (factorial 10 / 210) = 72 * Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_sqrt_10_factorial_div_210_l1867_186753


namespace NUMINAMATH_CALUDE_fishing_theorem_l1867_186735

/-- The total number of fish caught by Leo and Agrey -/
def total_fish (leo_fish : ℕ) (agrey_fish : ℕ) : ℕ :=
  leo_fish + agrey_fish

/-- Theorem: Given Leo caught 40 fish and Agrey caught 20 more fish than Leo,
    the total number of fish they caught together is 100. -/
theorem fishing_theorem :
  let leo_fish : ℕ := 40
  let agrey_fish : ℕ := leo_fish + 20
  total_fish leo_fish agrey_fish = 100 := by
sorry

end NUMINAMATH_CALUDE_fishing_theorem_l1867_186735


namespace NUMINAMATH_CALUDE_optionB_is_suitable_only_optionB_is_suitable_l1867_186771

/-- Represents a sampling experiment --/
structure SamplingExperiment where
  sampleSize : Nat
  populationSize : Nat
  numFactories : Nat
  numBoxes : Nat

/-- Criteria for lottery method suitability --/
def isLotteryMethodSuitable (exp : SamplingExperiment) : Prop :=
  exp.sampleSize < 20 ∧ 
  exp.populationSize < 100 ∧ 
  exp.numFactories = 1 ∧
  exp.numBoxes > 1

/-- The four options given in the problem --/
def optionA : SamplingExperiment := ⟨600, 3000, 1, 1⟩
def optionB : SamplingExperiment := ⟨6, 30, 1, 2⟩
def optionC : SamplingExperiment := ⟨6, 30, 2, 2⟩
def optionD : SamplingExperiment := ⟨10, 3000, 1, 1⟩

/-- Theorem stating that option B is suitable for the lottery method --/
theorem optionB_is_suitable : isLotteryMethodSuitable optionB := by
  sorry

/-- Theorem stating that option B is the only suitable option --/
theorem only_optionB_is_suitable : 
  isLotteryMethodSuitable optionB ∧ 
  ¬isLotteryMethodSuitable optionA ∧ 
  ¬isLotteryMethodSuitable optionC ∧ 
  ¬isLotteryMethodSuitable optionD := by
  sorry

end NUMINAMATH_CALUDE_optionB_is_suitable_only_optionB_is_suitable_l1867_186771


namespace NUMINAMATH_CALUDE_common_rest_days_1000_l1867_186730

/-- Represents the work-rest cycle of a person -/
structure WorkCycle where
  workDays : ℕ
  restDays : ℕ

/-- Calculates the number of common rest days for two people within a given number of days -/
def commonRestDays (cycleA cycleB : WorkCycle) (totalDays : ℕ) : ℕ :=
  sorry

/-- The main theorem stating the number of common rest days for Person A and Person B -/
theorem common_rest_days_1000 :
  let cycleA := WorkCycle.mk 3 1
  let cycleB := WorkCycle.mk 7 3
  commonRestDays cycleA cycleB 1000 = 100 := by
  sorry

end NUMINAMATH_CALUDE_common_rest_days_1000_l1867_186730


namespace NUMINAMATH_CALUDE_triangle_table_height_l1867_186720

theorem triangle_table_height (a b c : ℝ) (h_a : a = 25) (h_b : b = 31) (h_c : c = 34) :
  let s := (a + b + c) / 2
  let area := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  let h_max := 2 * area / (a + b + c)
  h_max = 4 * Real.sqrt 231 / 3 := by sorry

end NUMINAMATH_CALUDE_triangle_table_height_l1867_186720


namespace NUMINAMATH_CALUDE_quadratic_equal_roots_l1867_186700

theorem quadratic_equal_roots (k : ℝ) : 
  (∃ x : ℝ, 3 * x^2 - (k + 1) * x + 2 = 0 ∧ 
   ∀ y : ℝ, 3 * y^2 - (k + 1) * y + 2 = 0 → y = x) ↔ 
  (k = 2 * Real.sqrt 6 - 1 ∨ k = -2 * Real.sqrt 6 - 1) := by
sorry

end NUMINAMATH_CALUDE_quadratic_equal_roots_l1867_186700


namespace NUMINAMATH_CALUDE_P_plus_8_divisible_P_minus_8_divisible_P_unique_l1867_186773

/-- A fifth-degree polynomial P(x) that satisfies specific divisibility conditions -/
def P (x : ℝ) : ℝ := 3*x^5 - 10*x^3 + 15*x

/-- P(x) + 8 is divisible by (x+1)^3 -/
theorem P_plus_8_divisible (x : ℝ) : ∃ (q : ℝ → ℝ), P x + 8 = (x + 1)^3 * q x := by sorry

/-- P(x) - 8 is divisible by (x-1)^3 -/
theorem P_minus_8_divisible (x : ℝ) : ∃ (r : ℝ → ℝ), P x - 8 = (x - 1)^3 * r x := by sorry

/-- P(x) is the unique fifth-degree polynomial satisfying both divisibility conditions -/
theorem P_unique : ∀ (Q : ℝ → ℝ), 
  (∃ (q r : ℝ → ℝ), (∀ x, Q x + 8 = (x + 1)^3 * q x) ∧ (∀ x, Q x - 8 = (x - 1)^3 * r x)) →
  (∃ (a b c d e f : ℝ), ∀ x, Q x = a*x^5 + b*x^4 + c*x^3 + d*x^2 + e*x + f) →
  (∀ x, Q x = P x) := by sorry

end NUMINAMATH_CALUDE_P_plus_8_divisible_P_minus_8_divisible_P_unique_l1867_186773


namespace NUMINAMATH_CALUDE_area_difference_sheets_l1867_186784

/-- The difference in combined area (front and back) between a square sheet of paper
    with side length 11 inches and a rectangular sheet of paper measuring 5.5 inches
    by 11 inches is equal to 121 square inches. -/
theorem area_difference_sheets : 
  let square_sheet_side : ℝ := 11
  let rect_sheet_length : ℝ := 11
  let rect_sheet_width : ℝ := 5.5
  let square_sheet_area : ℝ := 2 * square_sheet_side * square_sheet_side
  let rect_sheet_area : ℝ := 2 * rect_sheet_length * rect_sheet_width
  square_sheet_area - rect_sheet_area = 121 := by
sorry

end NUMINAMATH_CALUDE_area_difference_sheets_l1867_186784


namespace NUMINAMATH_CALUDE_num_distinct_lines_is_seven_l1867_186788

/-- A right triangle with two 45-degree angles at the base -/
structure RightIsoscelesTriangle where
  /-- The right angle of the triangle -/
  right_angle : Angle
  /-- One of the 45-degree angles at the base -/
  base_angle1 : Angle
  /-- The other 45-degree angle at the base -/
  base_angle2 : Angle
  /-- The right angle is 90 degrees -/
  right_angle_is_right : right_angle = 90
  /-- The base angles are each 45 degrees -/
  base_angles_are_45 : base_angle1 = 45 ∧ base_angle2 = 45

/-- The number of distinct lines formed by altitudes, medians, and angle bisectors -/
def num_distinct_lines (t : RightIsoscelesTriangle) : ℕ := sorry

/-- Theorem stating that the number of distinct lines is 7 -/
theorem num_distinct_lines_is_seven (t : RightIsoscelesTriangle) : 
  num_distinct_lines t = 7 := by sorry

end NUMINAMATH_CALUDE_num_distinct_lines_is_seven_l1867_186788


namespace NUMINAMATH_CALUDE_eric_white_marbles_l1867_186769

theorem eric_white_marbles (total : ℕ) (blue : ℕ) (green : ℕ) (white : ℕ) 
  (h1 : total = 20) 
  (h2 : blue = 6) 
  (h3 : green = 2) 
  (h4 : total = white + blue + green) : 
  white = 12 := by
  sorry

end NUMINAMATH_CALUDE_eric_white_marbles_l1867_186769


namespace NUMINAMATH_CALUDE_rationalize_denominator_l1867_186761

theorem rationalize_denominator : (5 : ℝ) / Real.sqrt 125 = Real.sqrt 5 / 5 := by sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l1867_186761


namespace NUMINAMATH_CALUDE_joe_rounding_threshold_l1867_186749

/-- A grade is a nonnegative rational number -/
def Grade := { x : ℚ // 0 ≤ x }

/-- Joe's rounding function -/
noncomputable def joeRound (x : Grade) : ℕ :=
  sorry

/-- The smallest rational number M such that any grade x ≥ M gets rounded to at least 90 -/
def M : ℚ := 805 / 9

theorem joe_rounding_threshold :
  ∀ (x : Grade), joeRound x ≥ 90 ↔ x.val ≥ M :=
sorry

end NUMINAMATH_CALUDE_joe_rounding_threshold_l1867_186749


namespace NUMINAMATH_CALUDE_line_direction_vector_l1867_186793

theorem line_direction_vector (p1 p2 : ℝ × ℝ) (b : ℝ) :
  p1 = (4, -3) →
  p2 = (-1, 6) →
  ∃ k : ℝ, k ≠ 0 ∧ (p2.1 - p1.1, p2.2 - p1.2) = (k * b, k * (-1)) →
  b = 5 / 9 := by
sorry

end NUMINAMATH_CALUDE_line_direction_vector_l1867_186793


namespace NUMINAMATH_CALUDE_cubes_fill_box_completely_l1867_186743

/-- Represents the dimensions of a rectangular box -/
structure BoxDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the volume of a rectangular box -/
def boxVolume (d : BoxDimensions) : ℕ :=
  d.length * d.width * d.height

/-- Calculates the number of cubes that can fit along a dimension -/
def cubesAlongDimension (dimension : ℕ) (cubeSize : ℕ) : ℕ :=
  dimension / cubeSize

/-- Calculates the total number of cubes that can fit in the box -/
def totalCubes (d : BoxDimensions) (cubeSize : ℕ) : ℕ :=
  (cubesAlongDimension d.length cubeSize) *
  (cubesAlongDimension d.width cubeSize) *
  (cubesAlongDimension d.height cubeSize)

/-- Calculates the volume occupied by the cubes -/
def cubesVolume (d : BoxDimensions) (cubeSize : ℕ) : ℕ :=
  totalCubes d cubeSize * (cubeSize ^ 3)

/-- Theorem: The volume occupied by 4-inch cubes in the given box is 100% of the box's volume -/
theorem cubes_fill_box_completely (d : BoxDimensions) (h1 : d.length = 16) (h2 : d.width = 12) (h3 : d.height = 8) :
  cubesVolume d 4 = boxVolume d := by
  sorry

#eval cubesVolume ⟨16, 12, 8⟩ 4
#eval boxVolume ⟨16, 12, 8⟩

end NUMINAMATH_CALUDE_cubes_fill_box_completely_l1867_186743


namespace NUMINAMATH_CALUDE_cashier_bills_problem_l1867_186776

theorem cashier_bills_problem (total_bills : ℕ) (total_value : ℕ) 
  (h_total_bills : total_bills = 126)
  (h_total_value : total_value = 840) :
  ∃ (some_dollar_bills ten_dollar_bills : ℕ),
    some_dollar_bills + ten_dollar_bills = total_bills ∧
    some_dollar_bills + 10 * ten_dollar_bills = total_value ∧
    some_dollar_bills = 47 := by
  sorry

end NUMINAMATH_CALUDE_cashier_bills_problem_l1867_186776


namespace NUMINAMATH_CALUDE_slope_of_line_l1867_186755

/-- The slope of a line given by the equation 4y = 5x - 20 is 5/4 -/
theorem slope_of_line (x y : ℝ) : 4 * y = 5 * x - 20 → (∃ b : ℝ, y = (5/4) * x + b) := by
  sorry

end NUMINAMATH_CALUDE_slope_of_line_l1867_186755


namespace NUMINAMATH_CALUDE_manager_chef_wage_difference_l1867_186763

/-- Represents the hourly wages of employees at Joe's Steakhouse -/
structure SteakhouseWages where
  manager : ℝ
  dishwasher : ℝ
  chef : ℝ

/-- Conditions for wages at Joe's Steakhouse -/
def validSteakhouseWages (w : SteakhouseWages) : Prop :=
  w.manager = 6.50 ∧
  w.dishwasher = w.manager / 2 ∧
  w.chef = w.dishwasher * 1.20

/-- Theorem stating the wage difference between manager and chef -/
theorem manager_chef_wage_difference (w : SteakhouseWages) 
  (h : validSteakhouseWages w) : w.manager - w.chef = 2.60 := by
  sorry

end NUMINAMATH_CALUDE_manager_chef_wage_difference_l1867_186763


namespace NUMINAMATH_CALUDE_quadratic_roots_property_l1867_186794

theorem quadratic_roots_property (p q : ℝ) : 
  (3 * p^2 + 5 * p - 8 = 0) → 
  (3 * q^2 + 5 * q - 8 = 0) → 
  (p - 2) * (q - 2) = 14/3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_property_l1867_186794


namespace NUMINAMATH_CALUDE_forgotten_angle_measure_l1867_186766

/-- The sum of interior angles of a polygon with n sides --/
def sum_interior_angles (n : ℕ) : ℝ := (n - 2) * 180

/-- The sum of all but one interior angle of the polygon --/
def partial_sum : ℝ := 2017

/-- The measure of the forgotten angle --/
def forgotten_angle : ℝ := 143

theorem forgotten_angle_measure :
  ∃ (n : ℕ), n > 3 ∧ sum_interior_angles n = partial_sum + forgotten_angle :=
sorry

end NUMINAMATH_CALUDE_forgotten_angle_measure_l1867_186766


namespace NUMINAMATH_CALUDE_additional_students_score_l1867_186718

/-- Given a class with the following properties:
  * There are 17 students in total
  * The average grade of 15 students is 85
  * After including two more students, the new average becomes 87
  This theorem proves that the combined score of the two additional students is 204. -/
theorem additional_students_score (total_students : ℕ) (initial_students : ℕ) 
  (initial_average : ℝ) (final_average : ℝ) : 
  total_students = 17 → 
  initial_students = 15 → 
  initial_average = 85 → 
  final_average = 87 → 
  (total_students * final_average - initial_students * initial_average : ℝ) = 204 := by
  sorry

#check additional_students_score

end NUMINAMATH_CALUDE_additional_students_score_l1867_186718


namespace NUMINAMATH_CALUDE_min_value_of_function_min_value_is_achievable_l1867_186722

theorem min_value_of_function (x : ℝ) : x^2 + 6 / (x^2 + 1) ≥ 2 * Real.sqrt 6 - 1 := by
  sorry

theorem min_value_is_achievable : ∃ x : ℝ, x^2 + 6 / (x^2 + 1) = 2 * Real.sqrt 6 - 1 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_function_min_value_is_achievable_l1867_186722


namespace NUMINAMATH_CALUDE_tangent_line_parallel_l1867_186733

/-- Given a function f(x) = ln x - ax, if its derivative at x = 1 is -2, then a = 3 -/
theorem tangent_line_parallel (a : ℝ) : 
  let f : ℝ → ℝ := λ x => Real.log x - a * x
  (deriv f 1 = -2) → a = 3 := by
sorry

end NUMINAMATH_CALUDE_tangent_line_parallel_l1867_186733


namespace NUMINAMATH_CALUDE_geometric_progression_p_l1867_186713

theorem geometric_progression_p (p : ℝ) : 
  p > 0 ∧ 
  (3 * Real.sqrt p) ^ 2 = (-p - 8) * (p - 7) ↔ 
  p = 4 := by
sorry

end NUMINAMATH_CALUDE_geometric_progression_p_l1867_186713


namespace NUMINAMATH_CALUDE_grapes_purchased_l1867_186721

/-- The price of grapes per kg -/
def grape_price : ℕ := 70

/-- The amount of mangoes purchased in kg -/
def mango_amount : ℕ := 9

/-- The price of mangoes per kg -/
def mango_price : ℕ := 45

/-- The total amount paid to the shopkeeper -/
def total_paid : ℕ := 965

/-- The amount of grapes purchased in kg -/
def grape_amount : ℕ := 8

theorem grapes_purchased : 
  grape_price * grape_amount + mango_price * mango_amount = total_paid :=
by sorry

end NUMINAMATH_CALUDE_grapes_purchased_l1867_186721


namespace NUMINAMATH_CALUDE_prism_volume_l1867_186775

/-- The volume of a right rectangular prism with face areas 15, 10, and 30 -/
theorem prism_volume (l w h : ℝ) (h1 : l * w = 15) (h2 : w * h = 10) (h3 : l * h = 30) :
  l * w * h = 30 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_prism_volume_l1867_186775


namespace NUMINAMATH_CALUDE_hyperbola_line_intersection_l1867_186736

/-- Hyperbola equation: x^2 - y^2 = 4 -/
def hyperbola (x y : ℝ) : Prop := x^2 - y^2 = 4

/-- Line equation: y = k(x - 1) -/
def line (k x y : ℝ) : Prop := y = k * (x - 1)

/-- The line intersects the hyperbola at two points -/
def intersects_at_two_points (k : ℝ) : Prop :=
  k ∈ Set.Ioo (-(2 * Real.sqrt 3 / 3)) (-1) ∪ 
      Set.Ioo (-1) 1 ∪ 
      Set.Ioo 1 (2 * Real.sqrt 3 / 3)

/-- The line intersects the hyperbola at exactly one point -/
def intersects_at_one_point (k : ℝ) : Prop :=
  k = 1 ∨ k = -1 ∨ k = 2 * Real.sqrt 3 / 3 ∨ k = -(2 * Real.sqrt 3 / 3)

theorem hyperbola_line_intersection :
  (∀ k : ℝ, intersects_at_two_points k ↔ 
    ∃ x₁ y₁ x₂ y₂ : ℝ, x₁ ≠ x₂ ∧ 
      hyperbola x₁ y₁ ∧ hyperbola x₂ y₂ ∧ 
      line k x₁ y₁ ∧ line k x₂ y₂) ∧
  (∀ k : ℝ, intersects_at_one_point k ↔ 
    (∃ x y : ℝ, hyperbola x y ∧ line k x y) ∧
    ∀ x₁ y₁ x₂ y₂ : ℝ, hyperbola x₁ y₁ ∧ hyperbola x₂ y₂ ∧ 
      line k x₁ y₁ ∧ line k x₂ y₂ → x₁ = x₂ ∧ y₁ = y₂) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_line_intersection_l1867_186736


namespace NUMINAMATH_CALUDE_fractional_unit_problem_l1867_186783

def fractional_unit (n : ℕ) (d : ℕ) : ℚ := 1 / d

theorem fractional_unit_problem (n d : ℕ) (h1 : n = 5) (h2 : d = 11) :
  let u := fractional_unit n d
  (u = 1 / 11) ∧
  (n / d + 6 * u = 2) ∧
  (n / d - 5 * u = 1) :=
sorry

end NUMINAMATH_CALUDE_fractional_unit_problem_l1867_186783


namespace NUMINAMATH_CALUDE_count_numbers_with_at_least_two_zeros_l1867_186727

/-- The number of digits in the numbers we're considering -/
def n : ℕ := 6

/-- The total number of n-digit numbers -/
def total_n_digit_numbers : ℕ := 9 * 10^(n-1)

/-- The number of n-digit numbers with no zeros -/
def numbers_with_no_zeros : ℕ := 9^n

/-- The number of n-digit numbers with exactly one zero -/
def numbers_with_one_zero : ℕ := n * 9^(n-1)

/-- The number of n-digit numbers with at least two zeros -/
def numbers_with_at_least_two_zeros : ℕ :=
  total_n_digit_numbers - numbers_with_no_zeros - numbers_with_one_zero

theorem count_numbers_with_at_least_two_zeros :
  numbers_with_at_least_two_zeros = 14265 := by
  sorry

end NUMINAMATH_CALUDE_count_numbers_with_at_least_two_zeros_l1867_186727


namespace NUMINAMATH_CALUDE_coin_value_difference_l1867_186787

/-- Represents the number of coins of each type -/
structure CoinCount where
  pennies : ℕ
  nickels : ℕ
  dimes : ℕ

/-- Calculates the total value in cents for a given coin count -/
def totalValue (coins : CoinCount) : ℕ :=
  coins.pennies + 5 * coins.nickels + 10 * coins.dimes

/-- The total number of coins Maria has -/
def totalCoins : ℕ := 3030

/-- Predicate to check if a coin count is valid for Maria -/
def isValidCount (coins : CoinCount) : Prop :=
  coins.pennies ≥ 1 ∧ coins.nickels ≥ 1 ∧ coins.dimes ≥ 1 ∧
  coins.pennies + coins.nickels + coins.dimes = totalCoins

/-- Theorem stating the difference between max and min possible values -/
theorem coin_value_difference :
  ∃ (maxCoins minCoins : CoinCount),
    isValidCount maxCoins ∧ isValidCount minCoins ∧
    (∀ c, isValidCount c → totalValue c ≤ totalValue maxCoins) ∧
    (∀ c, isValidCount c → totalValue c ≥ totalValue minCoins) ∧
    totalValue maxCoins - totalValue minCoins = 27243 :=
  sorry

end NUMINAMATH_CALUDE_coin_value_difference_l1867_186787


namespace NUMINAMATH_CALUDE_f_properties_l1867_186747

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*a*x + 2

def is_monotonic_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  (∀ x y, a ≤ x → x ≤ y → y ≤ b → f x ≤ f y) ∨
  (∀ x y, a ≤ x → x ≤ y → y ≤ b → f y ≤ f x)

theorem f_properties :
  (∀ x ∈ Set.Icc (-5 : ℝ) 5, f (-1) x ≤ 37) ∧
  (∃ x ∈ Set.Icc (-5 : ℝ) 5, f (-1) x = 37) ∧
  (∀ x ∈ Set.Icc (-5 : ℝ) 5, 1 ≤ f (-1) x) ∧
  (∃ x ∈ Set.Icc (-5 : ℝ) 5, f (-1) x = 1) ∧
  (∀ a : ℝ, is_monotonic_on (f a) (-5) 5 ↔ a ≤ -5 ∨ a ≥ 5) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l1867_186747


namespace NUMINAMATH_CALUDE_circle_ratio_l1867_186768

theorem circle_ratio (r R : ℝ) (h1 : r > 0) (h2 : R > 0) (h3 : r ≤ R) : 
  π * R^2 = 3 * (π * R^2 - π * r^2) → R / r = Real.sqrt (3 / 2) := by
  sorry

end NUMINAMATH_CALUDE_circle_ratio_l1867_186768


namespace NUMINAMATH_CALUDE_product_of_difference_and_sum_is_zero_l1867_186744

theorem product_of_difference_and_sum_is_zero (a : ℝ) (x y : ℝ) 
  (h1 : x = a + 5)
  (h2 : a = 20)
  (h3 : y = 25) :
  (x - y) * (x + y) = 0 := by
sorry

end NUMINAMATH_CALUDE_product_of_difference_and_sum_is_zero_l1867_186744


namespace NUMINAMATH_CALUDE_perpendicular_planes_theorem_l1867_186740

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the perpendicular relation between lines and planes
variable (perp : Line → Plane → Prop)

-- Define the parallel relation between planes
variable (parallel : Plane → Plane → Prop)

-- State the theorem
theorem perpendicular_planes_theorem 
  (α β γ : Plane) (m n : Line)
  (h_diff_planes : α ≠ β ∧ β ≠ γ ∧ α ≠ γ)
  (h_diff_lines : m ≠ n)
  (h_n_perp_α : perp n α)
  (h_n_perp_β : perp n β)
  (h_m_perp_α : perp m α) :
  perp m β :=
sorry

end NUMINAMATH_CALUDE_perpendicular_planes_theorem_l1867_186740


namespace NUMINAMATH_CALUDE_relationship_abc_l1867_186707

theorem relationship_abc : 
  2022^0 > 8^2022 * (-0.125)^2023 ∧ 8^2022 * (-0.125)^2023 > 2021 * 2023 - 2022^2 := by
  sorry

end NUMINAMATH_CALUDE_relationship_abc_l1867_186707


namespace NUMINAMATH_CALUDE_mike_plants_cost_l1867_186770

def rose_bush_price : ℝ := 75
def tiger_tooth_aloe_price : ℝ := 100
def total_rose_bushes : ℕ := 6
def friend_rose_bushes : ℕ := 2
def mike_rose_bushes : ℕ := total_rose_bushes - friend_rose_bushes
def tiger_tooth_aloes : ℕ := 2
def rose_bush_tax_rate : ℝ := 0.05
def tiger_tooth_aloe_tax_rate : ℝ := 0.07

def mike_total_cost : ℝ :=
  (mike_rose_bushes : ℝ) * rose_bush_price * (1 + rose_bush_tax_rate) +
  (tiger_tooth_aloes : ℝ) * tiger_tooth_aloe_price * (1 + tiger_tooth_aloe_tax_rate)

theorem mike_plants_cost :
  mike_total_cost = 529 := by sorry

end NUMINAMATH_CALUDE_mike_plants_cost_l1867_186770


namespace NUMINAMATH_CALUDE_quadratic_equation_result_l1867_186723

theorem quadratic_equation_result : 
  ∀ y : ℝ, (6 * y^2 + 5 = 2 * y + 10) → (12 * y - 5)^2 = 133 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_result_l1867_186723


namespace NUMINAMATH_CALUDE_ln_sufficient_not_necessary_for_exp_l1867_186791

theorem ln_sufficient_not_necessary_for_exp (x : ℝ) :
  (∀ x, (Real.log x > 0 → Real.exp x > 1)) ∧
  (∃ x, Real.exp x > 1 ∧ ¬(Real.log x > 0)) :=
sorry

end NUMINAMATH_CALUDE_ln_sufficient_not_necessary_for_exp_l1867_186791


namespace NUMINAMATH_CALUDE_circumcircle_tangency_l1867_186786

-- Define the points
variable (A B C D E F : EuclideanPlane)

-- Define the parallelogram ABCD
def is_parallelogram (A B C D : EuclideanPlane) : Prop := sorry

-- Define that E is on BC
def point_on_segment (P Q R : EuclideanPlane) : Prop := sorry

-- Define that F is on AD
-- (We can reuse the point_on_segment definition)

-- Define the circumcircle of a triangle
def circumcircle (P Q R : EuclideanPlane) : Set EuclideanPlane := sorry

-- Define a line being tangent to a circle
def is_tangent (line : Set EuclideanPlane) (circle : Set EuclideanPlane) : Prop := sorry

-- Define a line segment
def line_segment (P Q : EuclideanPlane) : Set EuclideanPlane := sorry

-- The main theorem
theorem circumcircle_tangency 
  (h_parallelogram : is_parallelogram A B C D)
  (h_E_on_BC : point_on_segment B E C)
  (h_F_on_AD : point_on_segment A F D)
  (h_ABE_tangent_CF : is_tangent (line_segment C F) (circumcircle A B E)) :
  is_tangent (line_segment A E) (circumcircle C D F) := by
  sorry

end NUMINAMATH_CALUDE_circumcircle_tangency_l1867_186786


namespace NUMINAMATH_CALUDE_proposition_b_is_false_l1867_186778

theorem proposition_b_is_false : ¬(∀ x : ℝ, 
  (1 / x < 1 → ¬(-1 ≤ x ∧ x ≤ 1)) ∧ 
  (∃ y : ℝ, ¬(1 / y < 1) ∧ ¬(-1 ≤ y ∧ y ≤ 1))) := by
  sorry

end NUMINAMATH_CALUDE_proposition_b_is_false_l1867_186778


namespace NUMINAMATH_CALUDE_square_perimeter_problem_l1867_186797

/-- Given squares A, B, and C, prove that the perimeter of C is 48 -/
theorem square_perimeter_problem (A B C : ℝ) : 
  (4 * A = 16) →  -- Perimeter of A is 16
  (4 * B = 32) →  -- Perimeter of B is 32
  (C = A + B) →   -- Side length of C is sum of side lengths of A and B
  (4 * C = 48) := by  -- Perimeter of C is 48
sorry

end NUMINAMATH_CALUDE_square_perimeter_problem_l1867_186797


namespace NUMINAMATH_CALUDE_ellipse_area_theorem_l1867_186741

/-- Represents an ellipse with given properties -/
structure Ellipse where
  a : ℝ
  b : ℝ
  a_gt_b : a > b
  b_pos : b > 0
  vertex_y : b = 1
  eccentricity : Real.sqrt (a^2 - b^2) / a = Real.sqrt 2 / 2

/-- Represents a line passing through the right focus of the ellipse -/
structure FocusLine (e : Ellipse) where
  k : ℝ  -- Slope of the line

/-- Represents two points on the ellipse intersected by the focus line -/
structure IntersectionPoints (e : Ellipse) (l : FocusLine e) where
  A : ℝ × ℝ
  B : ℝ × ℝ
  on_ellipse_A : A.1^2 / e.a^2 + A.2^2 / e.b^2 = 1
  on_ellipse_B : B.1^2 / e.a^2 + B.2^2 / e.b^2 = 1
  on_line_A : A.2 = l.k * (A.1 - Real.sqrt (e.a^2 - e.b^2))
  on_line_B : B.2 = l.k * (B.1 - Real.sqrt (e.a^2 - e.b^2))
  perpendicular : A.1 * B.1 + A.2 * B.2 = 0  -- OA ⊥ OB condition

/-- Main theorem statement -/
theorem ellipse_area_theorem (e : Ellipse) (l : FocusLine e) (p : IntersectionPoints e l) :
  e.a^2 = 2 ∧ 
  (abs (p.A.1 - p.B.1) * abs (p.A.2 - p.B.2) / 2 = 2 * Real.sqrt 3 / 5) := by
  sorry

end NUMINAMATH_CALUDE_ellipse_area_theorem_l1867_186741


namespace NUMINAMATH_CALUDE_sequence_problem_l1867_186701

/-- Given a geometric sequence {a_n} and an arithmetic sequence {b_n} satisfying certain conditions,
    this theorem proves the general formulas for both sequences and the minimum n for which
    the sum of their first n terms exceeds 100. -/
theorem sequence_problem (a b : ℕ → ℝ) (n : ℕ) : 
  (∀ k, a (k + 1) = a k * (a 2 / a 1)) →  -- geometric sequence condition
  (∀ k, b (k + 1) - b k = b 2 - b 1) →   -- arithmetic sequence condition
  a 1 = 1 →
  b 1 = 1 →
  a 1 ≠ a 2 →
  a 1 + b 3 = 2 * a 2 →  -- a₁, a₂, b₃ form an arithmetic sequence
  b 1 * b 4 = (a 2)^2 →  -- b₁, a₂, b₄ form a geometric sequence
  (∀ k, a k = 2^(k-1)) ∧ 
  (∀ k, b k = k) ∧
  (n = 7 ∧ (2^n - 1 + n * (n + 1) / 2 > 100) ∧ 
   ∀ m < n, (2^m - 1 + m * (m + 1) / 2 ≤ 100)) :=
by sorry


end NUMINAMATH_CALUDE_sequence_problem_l1867_186701


namespace NUMINAMATH_CALUDE_sam_football_games_l1867_186746

/-- The number of football games Sam went to this year -/
def games_this_year : ℕ := 43 - 29

/-- Theorem stating that Sam went to 14 football games this year -/
theorem sam_football_games : games_this_year = 14 := by
  sorry

end NUMINAMATH_CALUDE_sam_football_games_l1867_186746


namespace NUMINAMATH_CALUDE_rectangle_diagonal_l1867_186738

/-- Given a rectangle with breadth b and length l, if its perimeter is 5 times its breadth
    and its area is 216 sq. cm, then its diagonal is 6√13 cm. -/
theorem rectangle_diagonal (b l : ℝ) (h1 : 2 * (l + b) = 5 * b) (h2 : l * b = 216) :
  Real.sqrt (l^2 + b^2) = 6 * Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_diagonal_l1867_186738


namespace NUMINAMATH_CALUDE_missy_watch_time_l1867_186729

/-- The total time Missy spends watching TV, given the number of reality shows,
    the duration of each reality show, and the duration of the cartoon. -/
def total_watch_time (num_reality_shows : ℕ) (reality_show_duration : ℕ) (cartoon_duration : ℕ) : ℕ :=
  num_reality_shows * reality_show_duration + cartoon_duration

/-- Theorem stating that Missy spends 150 minutes watching TV. -/
theorem missy_watch_time :
  total_watch_time 5 28 10 = 150 := by
  sorry

end NUMINAMATH_CALUDE_missy_watch_time_l1867_186729


namespace NUMINAMATH_CALUDE_ratio_of_part_to_whole_l1867_186703

theorem ratio_of_part_to_whole (N : ℝ) : 
  (1 / 1) * (1 / 3) * (2 / 5) * N = 10 →
  (40 / 100) * N = 120 →
  (10 : ℝ) / ((1 / 3) * (2 / 5) * N) = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_part_to_whole_l1867_186703


namespace NUMINAMATH_CALUDE_molecular_weight_calculation_l1867_186789

/-- Given 5 moles of a compound with a total molecular weight of 1170,
    prove that the molecular weight of 1 mole of the compound is 234. -/
theorem molecular_weight_calculation (total_weight : ℝ) (num_moles : ℝ) :
  total_weight = 1170 →
  num_moles = 5 →
  total_weight / num_moles = 234 := by
sorry

end NUMINAMATH_CALUDE_molecular_weight_calculation_l1867_186789


namespace NUMINAMATH_CALUDE_complement_A_in_U_l1867_186745

def U : Set ℕ := {x : ℕ | x ≥ 2}
def A : Set ℕ := {x : ℕ | x^2 ≥ 5}

theorem complement_A_in_U : (U \ A) = {2} := by sorry

end NUMINAMATH_CALUDE_complement_A_in_U_l1867_186745


namespace NUMINAMATH_CALUDE_power_nine_mod_seven_l1867_186780

theorem power_nine_mod_seven : 9^123 % 7 = 1 := by
  sorry

end NUMINAMATH_CALUDE_power_nine_mod_seven_l1867_186780


namespace NUMINAMATH_CALUDE_perfect_squares_is_good_l1867_186798

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def perfect_squares : Set ℕ := {n : ℕ | is_perfect_square n}

def is_good (A : Set ℕ) : Prop :=
  ∀ n : ℕ, n > 0 → 
    ∀ p q : ℕ, Prime p → Prime q → p ≠ q → p ∣ n → q ∣ n →
      ¬(n - p ∈ A ∧ n - q ∈ A)

theorem perfect_squares_is_good : is_good perfect_squares :=
sorry

end NUMINAMATH_CALUDE_perfect_squares_is_good_l1867_186798


namespace NUMINAMATH_CALUDE_total_loaves_is_nine_l1867_186708

/-- The number of bags of bread -/
def num_bags : ℕ := 3

/-- The number of loaves in each bag -/
def loaves_per_bag : ℕ := 3

/-- The total number of loaves of bread -/
def total_loaves : ℕ := num_bags * loaves_per_bag

theorem total_loaves_is_nine : total_loaves = 9 := by
  sorry

end NUMINAMATH_CALUDE_total_loaves_is_nine_l1867_186708


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1867_186785

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  ArithmeticSequence a → a 2 = 4 → a 6 = 16 → a 3 + a 5 = 20 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1867_186785


namespace NUMINAMATH_CALUDE_factorization_of_2a_squared_minus_8_l1867_186774

theorem factorization_of_2a_squared_minus_8 (a : ℝ) : 2 * a^2 - 8 = 2 * (a + 2) * (a - 2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_2a_squared_minus_8_l1867_186774


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l1867_186715

/-- Given that:
  - a and b are opposite numbers
  - c and d are reciprocals
  - The distance from point m to the origin is 5
Prove that m^2 - 100a - 99b - bcd + |cd - 2| = -74 -/
theorem algebraic_expression_value 
  (a b c d m : ℝ) 
  (h1 : a + b = 0) 
  (h2 : c * d = 1) 
  (h3 : m^2 = 25) : 
  m^2 - 100*a - 99*b - b*c*d + |c*d - 2| = -74 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l1867_186715


namespace NUMINAMATH_CALUDE_sum_of_roots_squared_difference_sum_of_roots_specific_equation_l1867_186765

theorem sum_of_roots_squared_difference (a c : ℝ) :
  let f := fun x : ℝ => (x - a)^2 - c
  (∃ x y : ℝ, f x = 0 ∧ f y = 0 ∧ x ≠ y) →
  (∀ x y : ℝ, f x = 0 → f y = 0 → x + y = 2*a) :=
by sorry

theorem sum_of_roots_specific_equation :
  let f := fun x : ℝ => (x - 5)^2 - 9
  (∃ x y : ℝ, f x = 0 ∧ f y = 0 ∧ x ≠ y) →
  (∀ x y : ℝ, f x = 0 → f y = 0 → x + y = 10) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_roots_squared_difference_sum_of_roots_specific_equation_l1867_186765


namespace NUMINAMATH_CALUDE_negation_of_existence_log_negation_equivalence_l1867_186782

theorem negation_of_existence (p : Real → Prop) :
  (¬∃ x, x > 1 ∧ p x) ↔ (∀ x, x > 1 → ¬p x) := by sorry

theorem log_negation_equivalence :
  (¬∃ x₀, x₀ > 1 ∧ Real.log x₀ > 1) ↔ (∀ x, x > 1 → Real.log x ≤ 1) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_log_negation_equivalence_l1867_186782


namespace NUMINAMATH_CALUDE_frank_problems_per_type_is_30_l1867_186748

/-- The number of math problems composed by Bill -/
def bill_problems : ℕ := 20

/-- The number of math problems composed by Ryan -/
def ryan_problems : ℕ := 2 * bill_problems

/-- The number of math problems composed by Frank -/
def frank_problems : ℕ := 3 * ryan_problems

/-- The number of different types of math problems -/
def problem_types : ℕ := 4

/-- The number of problems of each type that Frank composes -/
def frank_problems_per_type : ℕ := frank_problems / problem_types

theorem frank_problems_per_type_is_30 :
  frank_problems_per_type = 30 := by sorry

end NUMINAMATH_CALUDE_frank_problems_per_type_is_30_l1867_186748


namespace NUMINAMATH_CALUDE_select_three_from_seven_eq_210_l1867_186714

/-- The number of ways to select 3 distinct individuals from a group of 7 people to fill 3 distinct positions. -/
def select_three_from_seven : ℕ :=
  7 * 6 * 5

/-- Theorem stating that selecting 3 distinct individuals from a group of 7 people to fill 3 distinct positions can be done in 210 ways. -/
theorem select_three_from_seven_eq_210 :
  select_three_from_seven = 210 := by
  sorry

end NUMINAMATH_CALUDE_select_three_from_seven_eq_210_l1867_186714


namespace NUMINAMATH_CALUDE_implication_equivalence_l1867_186757

theorem implication_equivalence (P Q : Prop) : 
  (P → Q) ↔ (¬Q → ¬P) :=
sorry

end NUMINAMATH_CALUDE_implication_equivalence_l1867_186757


namespace NUMINAMATH_CALUDE_calculate_expression_l1867_186767

theorem calculate_expression : 5 * 423 + 4 * 423 + 3 * 423 + 421 = 5497 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l1867_186767


namespace NUMINAMATH_CALUDE_triangle_properties_l1867_186732

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.b = Real.sqrt 6 ∧ t.A = 2 * Real.pi / 3 ∧
  ((t.B = Real.pi / 4) ∨ (t.a = 3))

-- State the theorem
theorem triangle_properties (t : Triangle) 
  (h : triangle_conditions t) :
  t.c = (3 * Real.sqrt 2 - Real.sqrt 6) / 2 ∧
  (1 / 2 * t.b * t.c * Real.sin t.A) = (9 - 3 * Real.sqrt 3) / 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l1867_186732


namespace NUMINAMATH_CALUDE_xy_squared_minus_x_squared_y_l1867_186750

theorem xy_squared_minus_x_squared_y (x y : ℝ) 
  (h1 : x - y = 2) 
  (h2 : x * y = 3) : 
  x * y^2 - x^2 * y = -6 := by
  sorry

end NUMINAMATH_CALUDE_xy_squared_minus_x_squared_y_l1867_186750


namespace NUMINAMATH_CALUDE_certain_event_three_people_two_groups_l1867_186754

theorem certain_event_three_people_two_groups : 
  ∀ (group1 group2 : Finset Nat), 
  (group1 ∪ group2).card = 3 → 
  group1 ∩ group2 = ∅ → 
  group1 ≠ ∅ → 
  group2 ≠ ∅ → 
  (group1.card = 2 ∨ group2.card = 2) :=
sorry

end NUMINAMATH_CALUDE_certain_event_three_people_two_groups_l1867_186754


namespace NUMINAMATH_CALUDE_existence_of_abcd_l1867_186779

theorem existence_of_abcd (n : ℕ) (h : n > 1) : 
  ∃ (a b c d : ℕ), (a + b = c + d) ∧ (a * b - c * d = 4 * n) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_abcd_l1867_186779


namespace NUMINAMATH_CALUDE_sum_and_interval_l1867_186726

theorem sum_and_interval : 
  let sum := 3 + 1/6 + 4 + 3/8 + 6 + 1/12
  sum = 13.625 ∧ 13.5 < sum ∧ sum < 14 := by
  sorry

end NUMINAMATH_CALUDE_sum_and_interval_l1867_186726
