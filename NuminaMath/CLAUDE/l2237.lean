import Mathlib

namespace NUMINAMATH_CALUDE_root_sum_theorem_l2237_223798

theorem root_sum_theorem (m n p : ℝ) : 
  (∀ x, x^2 + 4*x + p = 0 ↔ x = m ∨ x = n) → 
  m * n = 4 → 
  m + n = -4 := by
sorry

end NUMINAMATH_CALUDE_root_sum_theorem_l2237_223798


namespace NUMINAMATH_CALUDE_house_rent_expenditure_l2237_223781

theorem house_rent_expenditure (total_income : ℝ) (petrol_spending : ℝ) 
  (h1 : petrol_spending = 0.3 * total_income)
  (h2 : petrol_spending = 300) : ℝ :=
by
  let remaining_income := total_income - petrol_spending
  let house_rent := 0.14 * remaining_income
  have : house_rent = 98 := by sorry
  exact house_rent

#check house_rent_expenditure

end NUMINAMATH_CALUDE_house_rent_expenditure_l2237_223781


namespace NUMINAMATH_CALUDE_shampoo_bottles_l2237_223722

theorem shampoo_bottles (medium_capacity : ℕ) (jumbo_capacity : ℕ) (unusable_space : ℕ) :
  medium_capacity = 45 →
  jumbo_capacity = 720 →
  unusable_space = 20 →
  (Nat.ceil ((jumbo_capacity - unusable_space : ℚ) / medium_capacity) : ℕ) = 16 := by
  sorry

end NUMINAMATH_CALUDE_shampoo_bottles_l2237_223722


namespace NUMINAMATH_CALUDE_candy_distribution_l2237_223701

/-- The number of pieces of candy in each of Wendy's boxes -/
def candy_per_box : ℕ := sorry

/-- The number of pieces of candy Wendy's brother has -/
def brother_candy : ℕ := 6

/-- The number of boxes Wendy has -/
def wendy_boxes : ℕ := 2

/-- The total number of pieces of candy -/
def total_candy : ℕ := 12

theorem candy_distribution :
  candy_per_box * wendy_boxes + brother_candy = total_candy ∧ candy_per_box = 3 :=
by sorry

end NUMINAMATH_CALUDE_candy_distribution_l2237_223701


namespace NUMINAMATH_CALUDE_two_over_x_values_l2237_223791

theorem two_over_x_values (x : ℝ) (hx : 3 - 9/x + 6/x^2 = 0) :
  2/x = 1 ∨ 2/x = 2 :=
by sorry

end NUMINAMATH_CALUDE_two_over_x_values_l2237_223791


namespace NUMINAMATH_CALUDE_correct_amount_returned_l2237_223775

/-- Calculates the amount to be returned in rubles given the initial deposit in USD and the exchange rate. -/
def amount_to_be_returned (initial_deposit : ℝ) (exchange_rate : ℝ) : ℝ :=
  initial_deposit * exchange_rate

/-- Proves that the amount to be returned is 581,500 rubles given the initial deposit and exchange rate. -/
theorem correct_amount_returned (initial_deposit : ℝ) (exchange_rate : ℝ) 
  (h1 : initial_deposit = 10000)
  (h2 : exchange_rate = 58.15) :
  amount_to_be_returned initial_deposit exchange_rate = 581500 := by
  sorry

#eval amount_to_be_returned 10000 58.15

end NUMINAMATH_CALUDE_correct_amount_returned_l2237_223775


namespace NUMINAMATH_CALUDE_greatest_divisor_with_remainders_l2237_223772

theorem greatest_divisor_with_remainders :
  ∃ (d : ℕ), d > 0 ∧
  (∃ (q1 : ℕ), 1428 = d * q1 + 9) ∧
  (∃ (q2 : ℕ), 2206 = d * q2 + 13) ∧
  (∀ (x : ℕ), x > 0 ∧
    (∃ (r1 : ℕ), 1428 = x * r1 + 9) ∧
    (∃ (r2 : ℕ), 2206 = x * r2 + 13) →
    x ≤ d) ∧
  d = 129 :=
by sorry

end NUMINAMATH_CALUDE_greatest_divisor_with_remainders_l2237_223772


namespace NUMINAMATH_CALUDE_horizontal_shift_right_l2237_223715

-- Define a generic function f
variable (f : ℝ → ℝ)

-- Define the horizontal shift
def horizontalShift (f : ℝ → ℝ) (a : ℝ) : ℝ → ℝ :=
  fun x ↦ f (x - a)

-- Theorem statement
theorem horizontal_shift_right (a : ℝ) :
  ∀ x : ℝ, (horizontalShift f a) x = f (x - a) :=
by
  sorry

-- Note: This theorem states that for all real x,
-- the horizontally shifted function is equal to f(x - a),
-- which is equivalent to shifting the graph of f(x) right by a units.

end NUMINAMATH_CALUDE_horizontal_shift_right_l2237_223715


namespace NUMINAMATH_CALUDE_ellipse_foci_distance_sum_l2237_223753

-- Define the ellipse
def ellipse (x y : ℝ) (m : ℝ) : Prop :=
  x^2 / 4 + y^2 / m = 1

-- Define the property that the ellipse passes through (0, 4)
def passes_through_B (m : ℝ) : Prop :=
  ellipse 0 4 m

-- Define the sum of distances from any point to the foci
def sum_distances_to_foci (m : ℝ) : ℝ := 8

-- Theorem statement
theorem ellipse_foci_distance_sum (m : ℝ) 
  (h : passes_through_B m) : 
  sum_distances_to_foci m = 8 := by sorry

end NUMINAMATH_CALUDE_ellipse_foci_distance_sum_l2237_223753


namespace NUMINAMATH_CALUDE_solution_set_when_a_is_one_range_of_a_l2237_223762

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := |x - a| + |x + 3|

-- Part 1
theorem solution_set_when_a_is_one :
  {x : ℝ | f 1 x ≥ 6} = Set.Iic (-4) ∪ Set.Ici 2 :=
sorry

-- Part 2
theorem range_of_a :
  {a : ℝ | ∀ x, f a x > -a} = Set.Ioi (-3/2) :=
sorry

end NUMINAMATH_CALUDE_solution_set_when_a_is_one_range_of_a_l2237_223762


namespace NUMINAMATH_CALUDE_katie_cole_miles_ratio_l2237_223721

theorem katie_cole_miles_ratio :
  ∀ (miles_xavier miles_katie miles_cole : ℕ),
    miles_xavier = 3 * miles_katie →
    miles_xavier = 84 →
    miles_cole = 7 →
    miles_katie / miles_cole = 4 := by
  sorry

end NUMINAMATH_CALUDE_katie_cole_miles_ratio_l2237_223721


namespace NUMINAMATH_CALUDE_plane_speed_calculation_l2237_223703

/-- Two planes traveling in opposite directions -/
structure TwoPlanes where
  speed_west : ℝ
  speed_east : ℝ
  time : ℝ
  total_distance : ℝ

/-- The theorem stating the conditions and the result to be proved -/
theorem plane_speed_calculation (planes : TwoPlanes) 
  (h1 : planes.speed_west = 275)
  (h2 : planes.time = 3.5)
  (h3 : planes.total_distance = 2100)
  : planes.speed_east = 325 := by
  sorry

#check plane_speed_calculation

end NUMINAMATH_CALUDE_plane_speed_calculation_l2237_223703


namespace NUMINAMATH_CALUDE_student_average_problem_l2237_223717

theorem student_average_problem :
  let total_students : ℕ := 25
  let group_a_students : ℕ := 15
  let group_b_students : ℕ := 10
  let group_b_average : ℚ := 90
  let total_average : ℚ := 84
  let group_a_average : ℚ := (total_students * total_average - group_b_students * group_b_average) / group_a_students
  group_a_average = 80 := by sorry

end NUMINAMATH_CALUDE_student_average_problem_l2237_223717


namespace NUMINAMATH_CALUDE_maria_towels_l2237_223711

/-- The number of towels Maria ended up with after shopping and giving some to her mother. -/
def towels_maria_kept (green_towels white_towels towels_given : ℕ) : ℕ :=
  green_towels + white_towels - towels_given

/-- Theorem stating that Maria ended up with 22 towels -/
theorem maria_towels :
  towels_maria_kept 35 21 34 = 22 := by
  sorry

end NUMINAMATH_CALUDE_maria_towels_l2237_223711


namespace NUMINAMATH_CALUDE_min_weeks_to_sunday_rest_l2237_223797

/-- Represents the work schedule cycle in days -/
def work_cycle : ℕ := 10

/-- Represents the number of days in a week -/
def days_in_week : ℕ := 7

/-- Represents the initial offset in days (starting rest on Saturday) -/
def initial_offset : ℕ := 6

/-- 
Theorem: Given a work schedule of 8 days work followed by 2 days rest,
starting with rest on Saturday and Sunday, the minimum number of weeks
before resting on a Sunday again is 7.
-/
theorem min_weeks_to_sunday_rest : 
  ∃ (n : ℕ), n > 0 ∧ 
  (n * days_in_week + initial_offset) % work_cycle = work_cycle - 1 ∧
  ∀ (m : ℕ), m > 0 → m < n → 
  (m * days_in_week + initial_offset) % work_cycle ≠ work_cycle - 1 ∧
  n = 7 :=
sorry

end NUMINAMATH_CALUDE_min_weeks_to_sunday_rest_l2237_223797


namespace NUMINAMATH_CALUDE_expression_evaluation_l2237_223770

theorem expression_evaluation : 2 - 3 * (-4) + 5 - (-6) * 7 = 61 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2237_223770


namespace NUMINAMATH_CALUDE_birds_in_tree_l2237_223789

/-- The number of birds left in a tree after some fly away -/
def birds_left (initial : ℝ) (flew_away : ℝ) : ℝ :=
  initial - flew_away

/-- Theorem: Given 21.0 initial birds and 14.0 birds that flew away, 7.0 birds are left -/
theorem birds_in_tree : birds_left 21.0 14.0 = 7.0 := by
  sorry

end NUMINAMATH_CALUDE_birds_in_tree_l2237_223789


namespace NUMINAMATH_CALUDE_smallest_number_l2237_223708

theorem smallest_number : ∀ (a b c d : ℝ), 
  a = -2 ∧ b = (1 : ℝ) / 2 ∧ c = 0 ∧ d = -Real.sqrt 2 →
  a < b ∧ a < c ∧ a < d :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_l2237_223708


namespace NUMINAMATH_CALUDE_triangular_cross_section_solids_l2237_223765

-- Define the set of all possible solids
inductive Solid
  | Prism
  | Pyramid
  | Frustum
  | Cylinder
  | Cone
  | TruncatedCone
  | Sphere

-- Define a predicate for solids that can have a triangular cross-section
def hasTriangularCrossSection (s : Solid) : Prop :=
  match s with
  | Solid.Prism => true
  | Solid.Pyramid => true
  | Solid.Frustum => true
  | Solid.Cone => true
  | _ => false

-- Define the set of solids that can have a triangular cross-section
def solidsWithTriangularCrossSection : Set Solid :=
  {s : Solid | hasTriangularCrossSection s}

-- Theorem statement
theorem triangular_cross_section_solids :
  solidsWithTriangularCrossSection = {Solid.Prism, Solid.Pyramid, Solid.Frustum, Solid.Cone} :=
by sorry

end NUMINAMATH_CALUDE_triangular_cross_section_solids_l2237_223765


namespace NUMINAMATH_CALUDE_horner_method_v2_l2237_223741

def f (x : ℝ) : ℝ := 4*x^4 + 3*x^3 - 6*x^2 + x - 1

def horner_v0 : ℝ := 4

def horner_v1 (x : ℝ) : ℝ := horner_v0 * x + 3

def horner_v2 (x : ℝ) : ℝ := horner_v1 x * x - 6

theorem horner_method_v2 : horner_v2 (-1) = -5 := by
  sorry

end NUMINAMATH_CALUDE_horner_method_v2_l2237_223741


namespace NUMINAMATH_CALUDE_unique_quadratic_solution_l2237_223739

theorem unique_quadratic_solution (a c : ℝ) : 
  (∃! x, a * x^2 + 10 * x + c = 0) →  -- exactly one solution
  (a + c = 12) →                      -- sum condition
  (a < c) →                           -- order condition
  (a = 6 - Real.sqrt 11 ∧ c = 6 + Real.sqrt 11) := by
sorry

end NUMINAMATH_CALUDE_unique_quadratic_solution_l2237_223739


namespace NUMINAMATH_CALUDE_paige_pencils_l2237_223799

theorem paige_pencils (initial_pencils : ℕ) : 
  (initial_pencils - 3 = 91) → initial_pencils = 94 := by
  sorry

end NUMINAMATH_CALUDE_paige_pencils_l2237_223799


namespace NUMINAMATH_CALUDE_basketball_tournament_games_l2237_223790

theorem basketball_tournament_games (x : ℕ) 
  (h1 : x > 0)
  (h2 : (3 * x) / 4 = (2 * (x + 4)) / 3 - 8) :
  x = 48 := by
sorry

end NUMINAMATH_CALUDE_basketball_tournament_games_l2237_223790


namespace NUMINAMATH_CALUDE_min_a_value_l2237_223758

noncomputable def f (x : ℝ) : ℝ := Real.log x / x

noncomputable def g (a e : ℝ) (x : ℝ) : ℝ := -e * x^2 + a * x

theorem min_a_value (e : ℝ) (he : e = Real.exp 1) :
  (∀ x₁ : ℝ, ∃ x₂ ∈ Set.Icc (1/3 : ℝ) 2, f x₁ ≤ g 2 e x₂) ∧
  (∀ ε > 0, ∃ x₁ : ℝ, ∀ x₂ ∈ Set.Icc (1/3 : ℝ) 2, f x₁ > g (2 - ε) e x₂) :=
sorry

end NUMINAMATH_CALUDE_min_a_value_l2237_223758


namespace NUMINAMATH_CALUDE_misha_dog_savings_l2237_223734

theorem misha_dog_savings (current_amount target_amount : ℕ) 
  (h1 : current_amount = 34)
  (h2 : target_amount = 47) :
  target_amount - current_amount = 13 := by
sorry

end NUMINAMATH_CALUDE_misha_dog_savings_l2237_223734


namespace NUMINAMATH_CALUDE_expression_factorization_l2237_223706

theorem expression_factorization (a b c : ℝ) :
  a * (b - c)^2 + b * (c - a)^2 + c * (a - b)^2 = (a - b) * (b - c) * (c - a) * (a + b + c) := by
  sorry

end NUMINAMATH_CALUDE_expression_factorization_l2237_223706


namespace NUMINAMATH_CALUDE_extreme_values_and_bounds_l2237_223785

-- Define the function f(x)
def f (a b x : ℝ) : ℝ := x^3 + a*x^2 + b*x

-- State the theorem
theorem extreme_values_and_bounds (a b : ℝ) :
  (∃ (x : ℝ), x = 1 ∧ (∀ (h : ℝ), f a b x ≥ f a b h ∨ f a b x ≤ f a b h)) ∧
  (∃ (y : ℝ), y = -2/3 ∧ (∀ (h : ℝ), f a b y ≥ f a b h ∨ f a b y ≤ f a b h)) →
  (a = -1/2 ∧ b = -2) ∧
  (∀ x ∈ Set.Icc (-1) 2, f (-1/2) (-2) x ≤ 2) ∧
  (∀ x ∈ Set.Icc (-1) 2, f (-1/2) (-2) x ≥ -5/2) ∧
  (∃ x ∈ Set.Icc (-1) 2, f (-1/2) (-2) x = 2) ∧
  (∃ x ∈ Set.Icc (-1) 2, f (-1/2) (-2) x = -5/2) :=
by sorry

end NUMINAMATH_CALUDE_extreme_values_and_bounds_l2237_223785


namespace NUMINAMATH_CALUDE_valid_fraction_pairs_l2237_223729

def is_valid_pair (x y : ℚ) : Prop :=
  ∃ (A B : ℕ+) (r : ℚ),
    x = (A : ℚ) * (1/10 + 1/70) ∧
    y = (B : ℚ) * (1/10 + 1/70) ∧
    x + y = 8 ∧
    r > 1 ∧
    ∃ (C D : ℕ), C > 1 ∧ D > 1 ∧ x = C * r ∧ y = D * r

theorem valid_fraction_pairs :
  (is_valid_pair (16/7) (40/7) ∧
   is_valid_pair (24/7) (32/7) ∧
   is_valid_pair (16/5) (24/5) ∧
   is_valid_pair 4 4) ∧
  ∀ x y, is_valid_pair x y →
    ((x = 16/7 ∧ y = 40/7) ∨
     (x = 24/7 ∧ y = 32/7) ∨
     (x = 16/5 ∧ y = 24/5) ∨
     (x = 4 ∧ y = 4) ∨
     (y = 16/7 ∧ x = 40/7) ∨
     (y = 24/7 ∧ x = 32/7) ∨
     (y = 16/5 ∧ x = 24/5)) :=
by sorry


end NUMINAMATH_CALUDE_valid_fraction_pairs_l2237_223729


namespace NUMINAMATH_CALUDE_ambiguous_triangle_case_l2237_223755

/-- Given two sides and an angle of a triangle, proves the existence of conditions
    for obtaining two different values for the third side. -/
theorem ambiguous_triangle_case (a b : ℝ) (α : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a < b)
  (h4 : 0 < α) (h5 : α < π) :
  ∃ c1 c2 : ℝ, c1 ≠ c2 ∧ 
  (∃ β γ : ℝ, 
    0 < β ∧ 0 < γ ∧ 
    α + β + γ = π ∧
    a / Real.sin α = b / Real.sin β ∧
    a / Real.sin α = c1 / Real.sin γ) ∧
  (∃ β' γ' : ℝ, 
    0 < β' ∧ 0 < γ' ∧ 
    α + β' + γ' = π ∧
    a / Real.sin α = b / Real.sin β' ∧
    a / Real.sin α = c2 / Real.sin γ') :=
sorry

end NUMINAMATH_CALUDE_ambiguous_triangle_case_l2237_223755


namespace NUMINAMATH_CALUDE_perpendicular_lines_minimum_value_l2237_223731

theorem perpendicular_lines_minimum_value (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_perp : (-(1 : ℝ) / (a - 4)) * (-2 * b) = 1) : 
  ∃ (x : ℝ), ∀ (y : ℝ), (a + 2) / (a + 1) + 1 / (2 * b) ≥ x ∧ 
  ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ 
  (-(1 : ℝ) / (a₀ - 4)) * (-2 * b₀) = 1 ∧ 
  (a₀ + 2) / (a₀ + 1) + 1 / (2 * b₀) = x ∧ 
  x = 9 / 5 :=
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_minimum_value_l2237_223731


namespace NUMINAMATH_CALUDE_ada_paul_scores_l2237_223786

/-- Ada and Paul's test scores problem -/
theorem ada_paul_scores (A1 A2 A3 P1 P2 P3 : ℤ) 
  (h1 : A1 > P1)
  (h2 : A2 = P2 + 4)
  (h3 : (P1 + P2 + P3) / 3 = (A1 + A2 + A3) / 3 + 4)
  (h4 : P3 = A3 + 26) :
  A1 - P1 = 10 := by
  sorry

end NUMINAMATH_CALUDE_ada_paul_scores_l2237_223786


namespace NUMINAMATH_CALUDE_cost_price_75_equals_selling_price_40_implies_87_5_percent_gain_l2237_223709

/-- Calculates the gain percent given the ratio of cost price to selling price -/
def gainPercent (costPriceRatio sellingPriceRatio : ℕ) : ℚ :=
  ((sellingPriceRatio : ℚ) / (costPriceRatio : ℚ) - 1) * 100

/-- Theorem stating that if the cost price of 75 articles equals the selling price of 40 articles, 
    then the gain percent is 87.5% -/
theorem cost_price_75_equals_selling_price_40_implies_87_5_percent_gain :
  gainPercent 75 40 = 87.5 := by sorry

end NUMINAMATH_CALUDE_cost_price_75_equals_selling_price_40_implies_87_5_percent_gain_l2237_223709


namespace NUMINAMATH_CALUDE_added_value_expression_max_value_m_gt_1_max_value_m_le_1_l2237_223771

noncomputable section

variables {a m : ℝ} (h_a : a > 0) (h_m : m > 0)

def x_range (a m : ℝ) : Set ℝ := Set.Ioo 0 ((2 * a * m) / (2 * m + 1))

def y (a x : ℝ) : ℝ := 8 * (a - x) * x^2

theorem added_value_expression (x : ℝ) (hx : x ∈ x_range a m) :
  y a x = 8 * (a - x) * x^2 := by sorry

theorem max_value_m_gt_1 (h_m_gt_1 : m > 1) :
  ∃ (x_max : ℝ), x_max ∈ x_range a m ∧
    y a x_max = (32 / 27) * a^3 ∧
    ∀ (x : ℝ), x ∈ x_range a m → y a x ≤ y a x_max := by sorry

theorem max_value_m_le_1 (h_m_le_1 : 0 < m ∧ m ≤ 1) :
  ∃ (x_max : ℝ), x_max ∈ x_range a m ∧
    y a x_max = (32 * m^2) / (2 * m + 1)^3 * a^3 ∧
    ∀ (x : ℝ), x ∈ x_range a m → y a x ≤ y a x_max := by sorry

end

end NUMINAMATH_CALUDE_added_value_expression_max_value_m_gt_1_max_value_m_le_1_l2237_223771


namespace NUMINAMATH_CALUDE_find_a_l2237_223746

def round_to_two_decimal_places (x : ℚ) : ℚ :=
  (⌊x * 100 + 0.5⌋ : ℚ) / 100

theorem find_a : ∃ (a : ℕ), round_to_two_decimal_places (1.322 - (a : ℚ) / 99) = 1.10 ∧ a = 22 := by
  sorry

end NUMINAMATH_CALUDE_find_a_l2237_223746


namespace NUMINAMATH_CALUDE_graduates_second_degree_l2237_223728

theorem graduates_second_degree (total : ℕ) (job : ℕ) (both : ℕ) (neither : ℕ) : 
  total = 73 → job = 32 → both = 13 → neither = 9 → 
  ∃ (second_degree : ℕ), second_degree = 45 := by
sorry

end NUMINAMATH_CALUDE_graduates_second_degree_l2237_223728


namespace NUMINAMATH_CALUDE_subtract_like_terms_l2237_223764

theorem subtract_like_terms (a : ℝ) : 4 * a - 3 * a = a := by
  sorry

end NUMINAMATH_CALUDE_subtract_like_terms_l2237_223764


namespace NUMINAMATH_CALUDE_nancy_pears_l2237_223720

theorem nancy_pears (total_pears alyssa_pears : ℕ) 
  (h1 : total_pears = 59)
  (h2 : alyssa_pears = 42) :
  total_pears - alyssa_pears = 17 := by
sorry

end NUMINAMATH_CALUDE_nancy_pears_l2237_223720


namespace NUMINAMATH_CALUDE_jack_shirts_per_kid_l2237_223726

theorem jack_shirts_per_kid (num_kids : ℕ) (buttons_per_shirt : ℕ) (total_buttons : ℕ) 
  (h1 : num_kids = 3)
  (h2 : buttons_per_shirt = 7)
  (h3 : total_buttons = 63) :
  total_buttons / buttons_per_shirt / num_kids = 3 := by
sorry

end NUMINAMATH_CALUDE_jack_shirts_per_kid_l2237_223726


namespace NUMINAMATH_CALUDE_gmat_question_percentages_l2237_223766

theorem gmat_question_percentages :
  ∀ (first_correct second_correct both_correct neither_correct : ℝ),
    first_correct = 85 →
    neither_correct = 5 →
    both_correct = 55 →
    second_correct = 100 - neither_correct - (first_correct - both_correct) →
    second_correct = 65 := by
  sorry

end NUMINAMATH_CALUDE_gmat_question_percentages_l2237_223766


namespace NUMINAMATH_CALUDE_triangle_PQR_area_l2237_223768

/-- The area of a triangle with vertices P(-4, 2), Q(6, 2), and R(2, -5) is 35 square units. -/
theorem triangle_PQR_area : 
  let P : ℝ × ℝ := (-4, 2)
  let Q : ℝ × ℝ := (6, 2)
  let R : ℝ × ℝ := (2, -5)
  let triangle_area := abs ((P.1 * (Q.2 - R.2) + Q.1 * (R.2 - P.2) + R.1 * (P.2 - Q.2)) / 2)
  triangle_area = 35 := by sorry

end NUMINAMATH_CALUDE_triangle_PQR_area_l2237_223768


namespace NUMINAMATH_CALUDE_quiz_mistakes_l2237_223725

theorem quiz_mistakes (total_items : ℕ) (score_percentage : ℚ) : 
  total_items = 25 → score_percentage = 80 / 100 → 
  total_items - (score_percentage * total_items).num = 5 := by
sorry

end NUMINAMATH_CALUDE_quiz_mistakes_l2237_223725


namespace NUMINAMATH_CALUDE_max_value_quadratic_inequality_l2237_223759

/-- Given a quadratic inequality ax² + bx + c > 0 with solution set {x | -1 < x < 3},
    the maximum value of b - c + 1/a is -2 -/
theorem max_value_quadratic_inequality (a b c : ℝ) :
  (∀ x, ax^2 + b*x + c > 0 ↔ -1 < x ∧ x < 3) →
  (∃ m : ℝ, ∀ a' b' c' : ℝ, 
    (∀ x, a'*x^2 + b'*x + c' > 0 ↔ -1 < x ∧ x < 3) →
    b' - c' + 1/a' ≤ m) ∧
  (∃ a₀ b₀ c₀ : ℝ, 
    (∀ x, a₀*x^2 + b₀*x + c₀ > 0 ↔ -1 < x ∧ x < 3) ∧
    b₀ - c₀ + 1/a₀ = -2) :=
by sorry

end NUMINAMATH_CALUDE_max_value_quadratic_inequality_l2237_223759


namespace NUMINAMATH_CALUDE_ellipse_major_axis_length_l2237_223702

/-- The length of the major axis of an ellipse formed by intersecting a right circular cylinder --/
def major_axis_length (cylinder_radius : ℝ) (major_minor_ratio : ℝ) : ℝ :=
  2 * cylinder_radius * major_minor_ratio

/-- Theorem: The major axis length of the ellipse is 6.4 --/
theorem ellipse_major_axis_length :
  let cylinder_radius : ℝ := 2
  let major_minor_ratio : ℝ := 1.6
  major_axis_length cylinder_radius major_minor_ratio = 6.4 := by
  sorry

#eval major_axis_length 2 1.6

end NUMINAMATH_CALUDE_ellipse_major_axis_length_l2237_223702


namespace NUMINAMATH_CALUDE_median_salary_is_28000_l2237_223730

/-- Represents a position in the company with its title, number of employees, and salary -/
structure Position where
  title : String
  count : Nat
  salary : Nat

/-- The list of positions in the company -/
def companyPositions : List Position := [
  { title := "CEO", count := 1, salary := 150000 },
  { title := "Senior Vice-President", count := 4, salary := 105000 },
  { title := "Manager", count := 15, salary := 80000 },
  { title := "Team Leader", count := 8, salary := 60000 },
  { title := "Office Assistant", count := 39, salary := 28000 }
]

/-- The total number of employees in the company -/
def totalEmployees : Nat := 67

/-- Calculates the median salary of the company -/
def medianSalary (positions : List Position) (total : Nat) : Nat :=
  sorry

/-- Theorem stating that the median salary of the company is $28,000 -/
theorem median_salary_is_28000 :
  medianSalary companyPositions totalEmployees = 28000 := by
  sorry

end NUMINAMATH_CALUDE_median_salary_is_28000_l2237_223730


namespace NUMINAMATH_CALUDE_appliance_savings_l2237_223723

def in_store_price : ℚ := 104.50
def tv_payment : ℚ := 24.80
def tv_shipping : ℚ := 10.80
def in_store_discount : ℚ := 5

theorem appliance_savings : 
  (4 * tv_payment + tv_shipping - (in_store_price - in_store_discount)) * 100 = 1050 := by
  sorry

end NUMINAMATH_CALUDE_appliance_savings_l2237_223723


namespace NUMINAMATH_CALUDE_multiply_and_simplify_l2237_223724

theorem multiply_and_simplify (x : ℝ) : 
  (x^4 + 6*x^2 + 9) * (x^2 - 3) = x^4 + 6*x^2 := by
  sorry

end NUMINAMATH_CALUDE_multiply_and_simplify_l2237_223724


namespace NUMINAMATH_CALUDE_equation_solutions_l2237_223780

/-- The set of solutions to the equation (a³ + b³)ⁿ = 4(ab)¹⁹⁹⁵ where a, b, n are integers greater than 1 -/
def Solutions : Set (ℕ × ℕ × ℕ) :=
  {(1, 1, 2), (2, 2, 998), (32, 32, 1247), (2^55, 2^55, 1322), (2^221, 2^221, 1328)}

/-- The predicate that checks if a triple (a, b, n) satisfies the equation (a³ + b³)ⁿ = 4(ab)¹⁹⁹⁵ -/
def SatisfiesEquation (a b n : ℕ) : Prop :=
  a > 1 ∧ b > 1 ∧ n > 1 ∧ (a^3 + b^3)^n = 4 * (a * b)^1995

theorem equation_solutions :
  ∀ a b n : ℕ, SatisfiesEquation a b n ↔ (a, b, n) ∈ Solutions := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l2237_223780


namespace NUMINAMATH_CALUDE_curve_C_properties_l2237_223767

-- Define the curve C
def C (x : ℝ) : ℝ := x^3 + 5*x^2 + 3*x

-- State the theorem
theorem curve_C_properties :
  -- The derivative of C is 3x² + 10x + 3
  (∀ x : ℝ, deriv C x = 3*x^2 + 10*x + 3) ∧
  -- The equation of the tangent line to C at x = 1 is 16x - y - 7 = 0
  (∀ y : ℝ, (C 1 = y) → (16 - y - 7 = 0 ↔ ∃ x : ℝ, y = 16*(x - 1) + C 1)) :=
by
  sorry

end NUMINAMATH_CALUDE_curve_C_properties_l2237_223767


namespace NUMINAMATH_CALUDE_expression_evaluation_l2237_223707

theorem expression_evaluation :
  let x : ℤ := 25
  let y : ℤ := 30
  let z : ℤ := 7
  (x - (y - z)) - ((x - y) - z) = 14 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2237_223707


namespace NUMINAMATH_CALUDE_functional_equation_solution_l2237_223776

theorem functional_equation_solution 
  (f g h : ℝ → ℝ) 
  (hf : Continuous f) 
  (hg : Continuous g) 
  (hh : Continuous h) 
  (h_eq : ∀ x y, f (x + y) = g x + h y) :
  ∃ a b c : ℝ, 
    (∀ x, f x = c * x + a + b) ∧
    (∀ x, g x = c * x + a) ∧
    (∀ x, h x = c * x + b) :=
by sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l2237_223776


namespace NUMINAMATH_CALUDE_sum_of_solutions_quadratic_l2237_223769

theorem sum_of_solutions_quadratic (x : ℝ) : 
  (∃ r s : ℝ, (72 - 18*x - x^2 = 0 ↔ (x = r ∨ x = s)) ∧ r + s = -18) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_solutions_quadratic_l2237_223769


namespace NUMINAMATH_CALUDE_turnover_equation_l2237_223796

/-- Represents the turnover equation for an online store over three months -/
theorem turnover_equation (x : ℝ) : 
  let july_turnover : ℝ := 16
  let august_turnover : ℝ := july_turnover * (1 + x)
  let september_turnover : ℝ := august_turnover * (1 + x)
  let total_turnover : ℝ := 120
  july_turnover + august_turnover + september_turnover = total_turnover :=
by
  sorry

#check turnover_equation

end NUMINAMATH_CALUDE_turnover_equation_l2237_223796


namespace NUMINAMATH_CALUDE_quadratic_root_range_l2237_223763

theorem quadratic_root_range (m : ℝ) : 
  (∃ (α : ℂ), (α.re = 0 ∧ α.im ≠ 0) ∧ 
    (α ^ 2 - (2 * m - 1) * α + m ^ 2 + 1 = 0) ∧
    (Complex.abs α ≤ 2)) →
  (m > -3/4 ∧ m ≤ Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_range_l2237_223763


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_l2237_223778

-- Define the sets A and B
def A : Set ℝ := {x | x + 1 < 0}
def B : Set ℝ := {x | x - 3 < 0}

-- Define the complement of A in the universal set ℝ
def C_U_A : Set ℝ := {x | ¬ (x ∈ A)}

-- State the theorem
theorem complement_A_intersect_B :
  (C_U_A ∩ B) = {x : ℝ | -1 ≤ x ∧ x < 3} := by sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_l2237_223778


namespace NUMINAMATH_CALUDE_sum_natural_numbers_not_end_72_73_74_l2237_223787

theorem sum_natural_numbers_not_end_72_73_74 (N : ℕ) : 
  ¬ (∃ k : ℕ, (N * (N + 1)) / 2 = 100 * k + 72 ∨ 
               (N * (N + 1)) / 2 = 100 * k + 73 ∨ 
               (N * (N + 1)) / 2 = 100 * k + 74) := by
  sorry


end NUMINAMATH_CALUDE_sum_natural_numbers_not_end_72_73_74_l2237_223787


namespace NUMINAMATH_CALUDE_cuboidal_box_volume_l2237_223737

/-- Represents a cuboidal box with given face areas -/
structure CuboidalBox where
  area1 : ℝ
  area2 : ℝ
  area3 : ℝ

/-- Calculates the volume of a cuboidal box given its face areas -/
def volume (box : CuboidalBox) : ℝ :=
  sorry

/-- Theorem stating that a cuboidal box with face areas 120, 72, and 60 has volume 720 -/
theorem cuboidal_box_volume :
  ∀ (box : CuboidalBox),
    box.area1 = 120 ∧ box.area2 = 72 ∧ box.area3 = 60 →
    volume box = 720 :=
by sorry

end NUMINAMATH_CALUDE_cuboidal_box_volume_l2237_223737


namespace NUMINAMATH_CALUDE_total_fish_caught_l2237_223757

def fishing_problem (leo_fish agrey_fish total_fish : ℕ) : Prop :=
  (leo_fish = 40) ∧
  (agrey_fish = leo_fish + 20) ∧
  (total_fish = leo_fish + agrey_fish)

theorem total_fish_caught : ∃ (leo_fish agrey_fish total_fish : ℕ),
  fishing_problem leo_fish agrey_fish total_fish ∧ total_fish = 100 :=
by
  sorry

end NUMINAMATH_CALUDE_total_fish_caught_l2237_223757


namespace NUMINAMATH_CALUDE_min_races_for_fifty_horses_l2237_223712

/-- Represents the minimum number of races needed to find the top k fastest horses
    from a total of n horses, racing at most m horses at a time. -/
def min_races (n m k : ℕ) : ℕ :=
  sorry

/-- The theorem stating that for 50 horses, racing 3 at a time,
    19 races are needed to find the top 5 fastest horses. -/
theorem min_races_for_fifty_horses :
  min_races 50 3 5 = 19 := by sorry

end NUMINAMATH_CALUDE_min_races_for_fifty_horses_l2237_223712


namespace NUMINAMATH_CALUDE_solve_equation_l2237_223745

theorem solve_equation : ∃ y : ℝ, 3 * y - 6 = |-20 + 5| ∧ y = 7 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l2237_223745


namespace NUMINAMATH_CALUDE_simplify_expression_l2237_223788

/-- Given a = 1 and b = -4, prove that 4(a²b+ab²)-3(a²b-1)+2ab²-6 = 89 -/
theorem simplify_expression (a b : ℝ) (ha : a = 1) (hb : b = -4) :
  4*(a^2*b + a*b^2) - 3*(a^2*b - 1) + 2*a*b^2 - 6 = 89 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2237_223788


namespace NUMINAMATH_CALUDE_refurbished_to_new_tshirt_ratio_l2237_223705

/-- The price of a new T-shirt in dollars -/
def new_tshirt_price : ℚ := 5

/-- The price of a pair of pants in dollars -/
def pants_price : ℚ := 4

/-- The price of a skirt in dollars -/
def skirt_price : ℚ := 6

/-- The total income from selling 2 new T-shirts, 1 pair of pants, 4 skirts, and 6 refurbished T-shirts -/
def total_income : ℚ := 53

/-- The number of new T-shirts sold -/
def new_tshirts_sold : ℕ := 2

/-- The number of pants sold -/
def pants_sold : ℕ := 1

/-- The number of skirts sold -/
def skirts_sold : ℕ := 4

/-- The number of refurbished T-shirts sold -/
def refurbished_tshirts_sold : ℕ := 6

/-- Theorem stating that the ratio of the price of a refurbished T-shirt to the price of a new T-shirt is 1/2 -/
theorem refurbished_to_new_tshirt_ratio :
  (total_income - (new_tshirt_price * new_tshirts_sold + pants_price * pants_sold + skirt_price * skirts_sold)) / refurbished_tshirts_sold / new_tshirt_price = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_refurbished_to_new_tshirt_ratio_l2237_223705


namespace NUMINAMATH_CALUDE_plates_needed_is_38_l2237_223716

/-- The number of plates needed for a week given the eating habits of Matt's family -/
def plates_needed : ℕ :=
  let days_with_son := 3
  let days_with_parents := 7 - days_with_son
  let plates_per_person_with_son := 1
  let plates_per_person_with_parents := 2
  let people_with_son := 2
  let people_with_parents := 4
  
  (days_with_son * people_with_son * plates_per_person_with_son) +
  (days_with_parents * people_with_parents * plates_per_person_with_parents)

theorem plates_needed_is_38 : plates_needed = 38 := by
  sorry

end NUMINAMATH_CALUDE_plates_needed_is_38_l2237_223716


namespace NUMINAMATH_CALUDE_lcm_equality_and_inequality_l2237_223727

theorem lcm_equality_and_inequality (a b c : ℕ) : 
  (Nat.lcm a (a + 5) = Nat.lcm b (b + 5) → a = b) ∧
  (Nat.lcm a b ≠ Nat.lcm (a + c) (b + c)) := by
  sorry

end NUMINAMATH_CALUDE_lcm_equality_and_inequality_l2237_223727


namespace NUMINAMATH_CALUDE_amelia_win_probability_l2237_223756

/-- The probability of Amelia's coin landing heads -/
def p_amelia : ℚ := 3/7

/-- The probability of Blaine's coin landing heads -/
def p_blaine : ℚ := 1/4

/-- The probability of Amelia winning the game -/
def p_amelia_wins : ℚ := 9/14

/-- The game described in the problem -/
def coin_game (p_a p_b : ℚ) : ℚ :=
  let p_amelia_first := p_a * (1 - p_b)
  let p_blaine_first := (1 - p_a) * p_b
  let p_both_tails := (1 - p_a) * (1 - p_b)
  let p_amelia_alternate := p_both_tails * (p_a / (1 - (1 - p_a) * (1 - p_b)))
  p_amelia_first + p_amelia_alternate

theorem amelia_win_probability :
  coin_game p_amelia p_blaine = p_amelia_wins :=
sorry

end NUMINAMATH_CALUDE_amelia_win_probability_l2237_223756


namespace NUMINAMATH_CALUDE_max_inscribed_rectangle_area_l2237_223777

theorem max_inscribed_rectangle_area (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  ∃ (rect_area : ℝ),
    (∀ (inscribed_rect_area : ℝ),
      inscribed_rect_area ≤ rect_area) ∧
    rect_area = (a * b) / 4 :=
by sorry

end NUMINAMATH_CALUDE_max_inscribed_rectangle_area_l2237_223777


namespace NUMINAMATH_CALUDE_smallest_solution_quartic_equation_l2237_223783

theorem smallest_solution_quartic_equation :
  ∃ (x : ℝ), x^4 - 50*x^2 + 625 = 0 ∧ x = -5 ∧ ∀ y : ℝ, y^4 - 50*y^2 + 625 = 0 → y ≥ x := by
  sorry

end NUMINAMATH_CALUDE_smallest_solution_quartic_equation_l2237_223783


namespace NUMINAMATH_CALUDE_gumballs_last_42_days_l2237_223719

/-- The number of gumballs Kim gets for each pair of earrings -/
def gumballs_per_pair : ℕ := 9

/-- The number of pairs of earrings Kim brings on day 1 -/
def day1_pairs : ℕ := 3

/-- The number of pairs of earrings Kim brings on day 2 -/
def day2_pairs : ℕ := 2 * day1_pairs

/-- The number of pairs of earrings Kim brings on day 3 -/
def day3_pairs : ℕ := day2_pairs - 1

/-- The number of gumballs Kim eats per day -/
def gumballs_eaten_per_day : ℕ := 3

/-- The total number of gumballs Kim receives -/
def total_gumballs : ℕ := gumballs_per_pair * (day1_pairs + day2_pairs + day3_pairs)

/-- The number of days the gumballs will last -/
def days_gumballs_last : ℕ := total_gumballs / gumballs_eaten_per_day

theorem gumballs_last_42_days : days_gumballs_last = 42 := by
  sorry

end NUMINAMATH_CALUDE_gumballs_last_42_days_l2237_223719


namespace NUMINAMATH_CALUDE_new_supervisor_salary_l2237_223773

def problem (workers : ℕ) (supervisors : ℕ) (initial_avg : ℝ) (supervisor_a : ℝ) (supervisor_b : ℝ) (supervisor_c : ℝ) (new_avg : ℝ) : Prop :=
  let total_people := workers + supervisors
  let initial_total := initial_avg * total_people
  let workers_supervisors_ab_total := initial_total - supervisor_c
  let new_total := new_avg * total_people
  let salary_difference := initial_total - new_total
  let new_supervisor_salary := supervisor_c - salary_difference
  new_supervisor_salary = 4600

theorem new_supervisor_salary :
  problem 15 3 5300 6200 7200 8200 5100 :=
by
  sorry

end NUMINAMATH_CALUDE_new_supervisor_salary_l2237_223773


namespace NUMINAMATH_CALUDE_min_red_to_blue_l2237_223774

/-- Represents the colors of chameleons -/
inductive Color
  | Red
  | Blue
  | Green
  | Yellow
  | Purple

/-- Represents a chameleon -/
structure Chameleon where
  color : Color

/-- Represents the color change rule -/
def colorChangeRule (biter : Color) (bitten : Color) : Color :=
  sorry -- Specific implementation not provided in the problem

/-- Represents a sequence of bites -/
def BiteSequence := List (Nat × Nat)

/-- Function to apply a bite sequence to a list of chameleons -/
def applyBiteSequence (chameleons : List Chameleon) (sequence : BiteSequence) : List Chameleon :=
  sorry -- Implementation would depend on colorChangeRule

/-- Predicate to check if all chameleons in a list are blue -/
def allBlue (chameleons : List Chameleon) : Prop :=
  ∀ c ∈ chameleons, c.color = Color.Blue

/-- The main theorem to be proved -/
theorem min_red_to_blue :
  ∀ n : Nat,
    (n ≥ 5 →
      ∃ (sequence : BiteSequence),
        allBlue (applyBiteSequence (List.replicate n (Chameleon.mk Color.Red)) sequence)) ∧
    (n < 5 →
      ¬∃ (sequence : BiteSequence),
        allBlue (applyBiteSequence (List.replicate n (Chameleon.mk Color.Red)) sequence)) :=
  sorry


end NUMINAMATH_CALUDE_min_red_to_blue_l2237_223774


namespace NUMINAMATH_CALUDE_consecutive_odd_integers_l2237_223743

theorem consecutive_odd_integers (x y z : ℤ) : 
  (∃ k : ℤ, x = 2*k + 1) →  -- x is odd
  y = x + 2 →               -- y is the next consecutive odd integer
  z = y + 2 →               -- z is the next consecutive odd integer after y
  y + z = x + 17 →          -- sum of last two is 17 more than the first
  x = 11 := by              -- the first integer is 11
sorry

end NUMINAMATH_CALUDE_consecutive_odd_integers_l2237_223743


namespace NUMINAMATH_CALUDE_f_is_quadratic_l2237_223752

-- Define what a quadratic equation is
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

-- Define the function representing the equation x^2 = x + 1
def f (x : ℝ) : ℝ := x^2 - x - 1

-- Theorem stating that f is a quadratic equation
theorem f_is_quadratic : is_quadratic_equation f :=
  sorry

end NUMINAMATH_CALUDE_f_is_quadratic_l2237_223752


namespace NUMINAMATH_CALUDE_quadratic_factor_problem_l2237_223704

theorem quadratic_factor_problem (a b : ℝ) :
  (∀ x, x^2 + 6*x + a = (x + 5)*(x + b)) → b = 1 ∧ a = 5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factor_problem_l2237_223704


namespace NUMINAMATH_CALUDE_intersection_x_coordinate_l2237_223793

-- Define the two lines
def line1 (x y : ℝ) : Prop := y = 5 * x - 20
def line2 (x y : ℝ) : Prop := 3 * x + y = 110

-- Define the intersection point
def intersection (x y : ℝ) : Prop := line1 x y ∧ line2 x y

-- Theorem statement
theorem intersection_x_coordinate :
  ∃ x y : ℝ, intersection x y ∧ x = 16.25 := by
sorry

end NUMINAMATH_CALUDE_intersection_x_coordinate_l2237_223793


namespace NUMINAMATH_CALUDE_restaurant_bill_proof_l2237_223750

/-- The number of friends in the group -/
def total_friends : ℕ := 10

/-- The number of friends who paid -/
def paying_friends : ℕ := 9

/-- The extra amount each paying friend contributed -/
def extra_payment : ℚ := 3

/-- The total bill at the restaurant -/
def total_bill : ℚ := 270

/-- Theorem stating that the given scenario results in the correct total bill -/
theorem restaurant_bill_proof :
  (paying_friends : ℚ) * (total_bill / total_friends + extra_payment) = total_bill :=
by sorry

end NUMINAMATH_CALUDE_restaurant_bill_proof_l2237_223750


namespace NUMINAMATH_CALUDE_division_problem_l2237_223742

theorem division_problem (dividend quotient remainder : ℕ) (divisor : ℚ) : 
  dividend = 12 → quotient = 9 → remainder = 8 → 
  dividend = (divisor * quotient) + remainder → 
  divisor = 4/9 := by
sorry

end NUMINAMATH_CALUDE_division_problem_l2237_223742


namespace NUMINAMATH_CALUDE_speed_of_train_b_l2237_223754

/-- Theorem: Speed of Train B
Given two trains A and B traveling in opposite directions, meeting at some point,
with train A reaching its destination 9 hours after meeting and traveling at 70 km/h,
and train B reaching its destination 4 hours after meeting,
prove that the speed of train B is 157.5 km/h. -/
theorem speed_of_train_b (speed_a : ℝ) (time_a time_b : ℝ) (speed_b : ℝ) :
  speed_a = 70 →
  time_a = 9 →
  time_b = 4 →
  speed_a * time_a = speed_b * time_b →
  speed_b = 157.5 := by
sorry

end NUMINAMATH_CALUDE_speed_of_train_b_l2237_223754


namespace NUMINAMATH_CALUDE_polygon_interior_exterior_angles_equal_l2237_223733

theorem polygon_interior_exterior_angles_equal (n : ℕ) : 
  n ≥ 3 → (n - 2) * 180 = 360 → n = 4 := by sorry

end NUMINAMATH_CALUDE_polygon_interior_exterior_angles_equal_l2237_223733


namespace NUMINAMATH_CALUDE_trig_system_solution_l2237_223761

theorem trig_system_solution (x y : ℝ) (m n : ℤ) :
  (Real.sin x * Real.cos y = 0.25) ∧ (Real.sin y * Real.cos x = 0.75) →
  ((x = Real.pi / 6 + Real.pi * (m - n : ℝ) ∧ y = Real.pi / 3 + Real.pi * (m + n : ℝ)) ∨
   (x = -Real.pi / 6 + Real.pi * (m - n : ℝ) ∧ y = 2 * Real.pi / 3 + Real.pi * (m + n : ℝ))) :=
by sorry

end NUMINAMATH_CALUDE_trig_system_solution_l2237_223761


namespace NUMINAMATH_CALUDE_june_birth_percentage_l2237_223760

theorem june_birth_percentage (total_scientists : ℕ) (june_born : ℕ) 
  (h1 : total_scientists = 200) (h2 : june_born = 18) :
  (june_born : ℚ) / total_scientists * 100 = 9 := by
  sorry

end NUMINAMATH_CALUDE_june_birth_percentage_l2237_223760


namespace NUMINAMATH_CALUDE_greatest_number_l2237_223744

theorem greatest_number (p q r s t : ℝ) 
  (h1 : r < s) 
  (h2 : t > q) 
  (h3 : q > p) 
  (h4 : t < r) : 
  s = max p (max q (max r (max s t))) := by
sorry

end NUMINAMATH_CALUDE_greatest_number_l2237_223744


namespace NUMINAMATH_CALUDE_area_bound_l2237_223736

-- Define the points and circles
variable (A B C D K L M N : Point)
variable (I I_A I_B I_C I_D : Circle)

-- Define the convex quadrilateral ABCD
def is_convex_quadrilateral (A B C D : Point) : Prop := sorry

-- Define the inscribed circle I
def is_inscribed_circle (I : Circle) (A B C D : Point) : Prop := sorry

-- Define tangent points
def is_tangent_point (K L M N : Point) (I : Circle) (A B C D : Point) : Prop := sorry

-- Define incircles of triangles
def is_incircle (I_A I_B I_C I_D : Circle) (A B C D K L M N : Point) : Prop := sorry

-- Define common external tangent lines
def common_external_tangent (I_AB I_BC I_CD I_AD : Line) (I_A I_B I_C I_D : Circle) : Prop := sorry

-- Define the area S of the quadrilateral formed by I_AB, I_BC, I_CD, and I_AD
def area_S (I_AB I_BC I_CD I_AD : Line) : ℝ := sorry

-- Define the radius r of circle I
def radius_r (I : Circle) : ℝ := sorry

-- Theorem statement
theorem area_bound 
  (h1 : is_convex_quadrilateral A B C D)
  (h2 : is_inscribed_circle I A B C D)
  (h3 : is_tangent_point K L M N I A B C D)
  (h4 : is_incircle I_A I_B I_C I_D A B C D K L M N)
  (h5 : common_external_tangent I_AB I_BC I_CD I_AD I_A I_B I_C I_D)
  (S : ℝ)
  (h6 : S = area_S I_AB I_BC I_CD I_AD)
  (r : ℝ)
  (h7 : r = radius_r I) :
  S ≤ (12 - 8 * Real.sqrt 2) * r^2 := by sorry

end NUMINAMATH_CALUDE_area_bound_l2237_223736


namespace NUMINAMATH_CALUDE_election_votes_l2237_223784

theorem election_votes (total_votes : ℕ) (invalid_percent : ℚ) (winner_percent : ℚ) :
  total_votes = 9000 →
  invalid_percent = 30 / 100 →
  winner_percent = 60 / 100 →
  ∃ (other_votes : ℕ), other_votes = 2520 :=
by
  sorry

end NUMINAMATH_CALUDE_election_votes_l2237_223784


namespace NUMINAMATH_CALUDE_vector_coordinates_proof_l2237_223718

theorem vector_coordinates_proof :
  ∀ (a b : ℝ × ℝ),
    (‖a‖ = 3) →
    (b = (1, 2)) →
    (a.1 * b.1 + a.2 * b.2 = 0) →
    ((a = (-6 * Real.sqrt 5 / 5, 3 * Real.sqrt 5 / 5)) ∨
     (a = (6 * Real.sqrt 5 / 5, -3 * Real.sqrt 5 / 5))) := by
  sorry

end NUMINAMATH_CALUDE_vector_coordinates_proof_l2237_223718


namespace NUMINAMATH_CALUDE_sum_of_roots_cubic_l2237_223710

theorem sum_of_roots_cubic (x : ℝ) : 
  let f : ℝ → ℝ := λ x => 6*x^3 - 7*x^2 + 2*x
  (∃ a b c : ℝ, f x = (x - a) * (x - b) * (x - c)) →
  a + b + c = 7/6 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_roots_cubic_l2237_223710


namespace NUMINAMATH_CALUDE_katy_june_books_l2237_223794

/-- The number of books Katy read in June -/
def june_books : ℕ := sorry

/-- The number of books Katy read in July -/
def july_books : ℕ := 2 * june_books

/-- The number of books Katy read in August -/
def august_books : ℕ := july_books - 3

/-- The total number of books Katy read during the summer -/
def total_books : ℕ := 37

theorem katy_june_books :
  june_books + july_books + august_books = total_books ∧ june_books = 8 := by sorry

end NUMINAMATH_CALUDE_katy_june_books_l2237_223794


namespace NUMINAMATH_CALUDE_complex_modulus_l2237_223732

theorem complex_modulus (z : ℂ) (a : ℝ) : 
  z = a + Complex.I ∧ z + z = 1 - 3 * Complex.I → Complex.abs z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_l2237_223732


namespace NUMINAMATH_CALUDE_ratio_problem_l2237_223740

theorem ratio_problem (w x y z : ℝ) 
  (h1 : w / x = 1 / 3) 
  (h2 : w / y = 2 / 3) 
  (h3 : w / z = 3 / 5) 
  (hw : w ≠ 0) : 
  (x + y) / z = 27 / 10 := by
sorry

end NUMINAMATH_CALUDE_ratio_problem_l2237_223740


namespace NUMINAMATH_CALUDE_streaming_service_subscriber_decrease_l2237_223795

/-- Proves the maximum percentage decrease in subscribers for a streaming service --/
theorem streaming_service_subscriber_decrease
  (initial_price : ℝ)
  (price_increase_percentage : ℝ)
  (h_initial_price : initial_price = 15)
  (h_price_increase : price_increase_percentage = 0.20) :
  let new_price := initial_price * (1 + price_increase_percentage)
  let max_decrease_percentage := 1 - (initial_price / new_price)
  ∃ (ε : ℝ), ε > 0 ∧ abs (max_decrease_percentage - (1/6)) < ε :=
by sorry

end NUMINAMATH_CALUDE_streaming_service_subscriber_decrease_l2237_223795


namespace NUMINAMATH_CALUDE_no_intersection_l2237_223749

/-- The number of distinct points of intersection between two ellipses -/
def intersectionPoints (f g : ℝ → ℝ → Prop) : ℕ :=
  sorry

/-- First ellipse: 3x^2 + 2y^2 = 4 -/
def ellipse1 (x y : ℝ) : Prop :=
  3 * x^2 + 2 * y^2 = 4

/-- Second ellipse: 6x^2 + 3y^2 = 9 -/
def ellipse2 (x y : ℝ) : Prop :=
  6 * x^2 + 3 * y^2 = 9

/-- Theorem: The number of distinct points of intersection between the two given ellipses is 0 -/
theorem no_intersection : intersectionPoints ellipse1 ellipse2 = 0 :=
  sorry

end NUMINAMATH_CALUDE_no_intersection_l2237_223749


namespace NUMINAMATH_CALUDE_expression_undefined_at_twelve_l2237_223779

theorem expression_undefined_at_twelve :
  ∀ x : ℝ, x = 12 → (x^2 - 24*x + 144 = 0) := by sorry

end NUMINAMATH_CALUDE_expression_undefined_at_twelve_l2237_223779


namespace NUMINAMATH_CALUDE_walk_group_legs_and_wheels_l2237_223700

/-- Calculates the total number of legs and wheels in a group of humans, dogs, and wheelchairs. -/
def total_legs_and_wheels (num_humans : ℕ) (num_dogs : ℕ) (num_wheelchairs : ℕ) : ℕ :=
  num_humans * 2 + num_dogs * 4 + num_wheelchairs * 4

/-- Proves that the total number of legs and wheels in the given group is 22. -/
theorem walk_group_legs_and_wheels :
  total_legs_and_wheels 3 3 1 = 22 := by
  sorry

end NUMINAMATH_CALUDE_walk_group_legs_and_wheels_l2237_223700


namespace NUMINAMATH_CALUDE_inequality_proof_l2237_223782

theorem inequality_proof (a b x : ℝ) (h1 : 0 < a) (h2 : a < b) :
  (b - a) / (b + a) ≤ (b + a * Real.sin x) / (b - a * Real.sin x) ∧
  (b + a * Real.sin x) / (b - a * Real.sin x) ≤ (b + a) / (b - a) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l2237_223782


namespace NUMINAMATH_CALUDE_no_positive_integer_solution_l2237_223714

theorem no_positive_integer_solution :
  ¬∃ (p q r : ℕ+), 
    (p^2 : ℚ) / q = 4 / 5 ∧
    (q : ℚ) / r^2 = 2 / 3 ∧
    (p : ℚ) / r^3 = 6 / 7 :=
by sorry

end NUMINAMATH_CALUDE_no_positive_integer_solution_l2237_223714


namespace NUMINAMATH_CALUDE_largest_n_divisible_by_five_l2237_223747

def expression (n : ℕ) : ℤ :=
  8 * (n - 2)^6 - 3 * n^2 + 20 * n - 36

theorem largest_n_divisible_by_five :
  ∀ n : ℕ, n < 100000 →
    (expression n % 5 = 0 → n ≤ 99997) ∧
    (expression 99997 % 5 = 0) ∧
    99997 < 100000 :=
by sorry

end NUMINAMATH_CALUDE_largest_n_divisible_by_five_l2237_223747


namespace NUMINAMATH_CALUDE_stream_speed_calculation_l2237_223735

/-- Proves that given a boat with a speed of 20 km/hr in still water,
    traveling 125 km downstream in 5 hours, the speed of the stream is 5 km/hr. -/
theorem stream_speed_calculation (boat_speed : ℝ) (downstream_distance : ℝ) (downstream_time : ℝ) :
  boat_speed = 20 →
  downstream_distance = 125 →
  downstream_time = 5 →
  (boat_speed + (downstream_distance / downstream_time - boat_speed)) = 25 :=
by
  sorry

#check stream_speed_calculation

end NUMINAMATH_CALUDE_stream_speed_calculation_l2237_223735


namespace NUMINAMATH_CALUDE_sofa_purchase_sum_l2237_223792

/-- The sum of Joan and Karl's sofa purchases -/
def total_purchase (joan_price karl_price : ℝ) : ℝ := joan_price + karl_price

/-- Theorem: Given the conditions, the sum of Joan and Karl's sofa purchases is $600 -/
theorem sofa_purchase_sum :
  ∀ (joan_price karl_price : ℝ),
  joan_price = 230 →
  2 * joan_price = karl_price + 90 →
  total_purchase joan_price karl_price = 600 := by
sorry

end NUMINAMATH_CALUDE_sofa_purchase_sum_l2237_223792


namespace NUMINAMATH_CALUDE_difference_of_squares_l2237_223738

theorem difference_of_squares : 535^2 - 465^2 = 70000 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l2237_223738


namespace NUMINAMATH_CALUDE_count_D_eq_3_is_18_l2237_223713

/-- D(n) is the number of pairs of different adjacent digits in the binary representation of n -/
def D (n : ℕ) : ℕ := sorry

/-- The count of positive integers n ≤ 200 for which D(n) = 3 -/
def count_D_eq_3 : ℕ := sorry

theorem count_D_eq_3_is_18 : count_D_eq_3 = 18 := by sorry

end NUMINAMATH_CALUDE_count_D_eq_3_is_18_l2237_223713


namespace NUMINAMATH_CALUDE_polygon_interior_angle_sum_l2237_223751

theorem polygon_interior_angle_sum (n : ℕ) (h : n > 2) :
  (n * 40 = 360) →
  (n - 2) * 180 = 1260 := by
  sorry

end NUMINAMATH_CALUDE_polygon_interior_angle_sum_l2237_223751


namespace NUMINAMATH_CALUDE_losing_ticket_probability_l2237_223748

/-- Given the odds of drawing a winning ticket are 5:8, 
    the probability of drawing a losing ticket is 8/13 -/
theorem losing_ticket_probability (winning_odds : Rat) 
  (h : winning_odds = 5 / 8) : 
  (1 : Rat) - winning_odds * (13 : Rat) / ((5 : Rat) + (8 : Rat)) = 8 / 13 :=
sorry

end NUMINAMATH_CALUDE_losing_ticket_probability_l2237_223748
