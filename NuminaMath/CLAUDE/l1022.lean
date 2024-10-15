import Mathlib

namespace NUMINAMATH_CALUDE_problem_solution_l1022_102228

theorem problem_solution (x y : ℝ) (h1 : x - y = 25) (h2 : x * y = 36) : 
  x^2 + y^2 = 697 ∧ x + y = Real.sqrt 769 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1022_102228


namespace NUMINAMATH_CALUDE_spinsters_and_cats_l1022_102265

theorem spinsters_and_cats (spinsters : ℕ) (cats : ℕ) : 
  spinsters = 18 → 
  spinsters * 9 = cats * 2 → 
  cats - spinsters = 63 := by
sorry

end NUMINAMATH_CALUDE_spinsters_and_cats_l1022_102265


namespace NUMINAMATH_CALUDE_cubic_inequality_l1022_102276

theorem cubic_inequality (x : ℝ) (h : x ≥ 0) : 3 * x^3 - 6 * x^2 + 4 ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_cubic_inequality_l1022_102276


namespace NUMINAMATH_CALUDE_final_piece_count_l1022_102251

/-- Represents the number of pieces after each cut -/
structure PaperCuts where
  initial : Nat
  first_cut : Nat
  second_cut : Nat
  third_cut : Nat
  fourth_cut : Nat

/-- The cutting process as described in the problem -/
def cutting_process : PaperCuts :=
  { initial := 1
  , first_cut := 10
  , second_cut := 19
  , third_cut := 28
  , fourth_cut := 37 }

/-- Theorem stating that the final number of pieces is 37 -/
theorem final_piece_count :
  (cutting_process.fourth_cut = 37) := by sorry

end NUMINAMATH_CALUDE_final_piece_count_l1022_102251


namespace NUMINAMATH_CALUDE_fruit_tree_problem_l1022_102239

theorem fruit_tree_problem (initial_apples : ℕ) (pick_ratio : ℚ) : 
  initial_apples = 180 →
  pick_ratio = 3 / 5 →
  ∃ (initial_plums : ℕ),
    initial_plums * 3 = initial_apples ∧
    (initial_apples + initial_plums) * (1 - pick_ratio) = 96 := by
  sorry

end NUMINAMATH_CALUDE_fruit_tree_problem_l1022_102239


namespace NUMINAMATH_CALUDE_cube_square_equation_solution_l1022_102223

theorem cube_square_equation_solution :
  2^3 - 7 = 3^2 + (-8) := by sorry

end NUMINAMATH_CALUDE_cube_square_equation_solution_l1022_102223


namespace NUMINAMATH_CALUDE_cleanup_time_is_25_minutes_l1022_102284

/-- Represents the toy cleaning scenario -/
structure ToyCleaningScenario where
  totalToys : ℕ
  momPutRate : ℕ
  miaTakeRate : ℕ
  brotherTossRate : ℕ
  momCycleTime : ℕ
  brotherCycleTime : ℕ

/-- Calculates the time taken to clean up all toys -/
def cleanupTime (scenario : ToyCleaningScenario) : ℚ :=
  sorry

/-- Theorem stating that the cleanup time for the given scenario is 25 minutes -/
theorem cleanup_time_is_25_minutes :
  let scenario : ToyCleaningScenario := {
    totalToys := 40,
    momPutRate := 4,
    miaTakeRate := 3,
    brotherTossRate := 1,
    momCycleTime := 20,
    brotherCycleTime := 40
  }
  cleanupTime scenario = 25 := by
  sorry

end NUMINAMATH_CALUDE_cleanup_time_is_25_minutes_l1022_102284


namespace NUMINAMATH_CALUDE_no_real_solutions_l1022_102238

theorem no_real_solutions : ¬ ∃ x : ℝ, (5*x)/(x^2 + 2*x + 4) + (6*x)/(x^2 - 4*x + 4) = -1 := by
  sorry

end NUMINAMATH_CALUDE_no_real_solutions_l1022_102238


namespace NUMINAMATH_CALUDE_company_workers_count_l1022_102235

/-- Represents the hierarchical structure of a company -/
structure CompanyHierarchy where
  workers_per_lead : ℕ
  leads_per_supervisor : ℕ
  num_supervisors : ℕ

/-- Calculates the number of workers in a company given its hierarchical structure -/
def calculate_workers (ch : CompanyHierarchy) : ℕ :=
  ch.num_supervisors * ch.leads_per_supervisor * ch.workers_per_lead

/-- Theorem stating that a company with the given hierarchical structure and 13 supervisors has 390 workers -/
theorem company_workers_count :
  let ch : CompanyHierarchy := {
    workers_per_lead := 10,
    leads_per_supervisor := 3,
    num_supervisors := 13
  }
  calculate_workers ch = 390 := by sorry

end NUMINAMATH_CALUDE_company_workers_count_l1022_102235


namespace NUMINAMATH_CALUDE_x_minus_reciprocal_equals_one_l1022_102210

-- Define the integer part function
noncomputable def intPart (x : ℝ) : ℤ :=
  ⌊x⌋

-- Define the fractional part function
noncomputable def fracPart (x : ℝ) : ℝ :=
  x - intPart x

-- Main theorem
theorem x_minus_reciprocal_equals_one (x : ℝ) 
  (h1 : x > 0)
  (h2 : (intPart x : ℝ)^2 = x * fracPart x) : 
  x - 1/x = 1 := by
  sorry

end NUMINAMATH_CALUDE_x_minus_reciprocal_equals_one_l1022_102210


namespace NUMINAMATH_CALUDE_perp_planes_necessary_not_sufficient_l1022_102275

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the perpendicular relation between planes
variable (perp_planes : Plane → Plane → Prop)

-- Define the perpendicular relation between a line and a plane
variable (perp_line_plane : Line → Plane → Prop)

-- Define the subset relation between a line and a plane
variable (subset_line_plane : Line → Plane → Prop)

-- Main theorem
theorem perp_planes_necessary_not_sufficient
  (α β : Plane) (m : Line)
  (h_diff : α ≠ β)
  (h_subset : subset_line_plane m α) :
  (perp_planes α β → perp_line_plane m β) ∧
  ¬(perp_line_plane m β → perp_planes α β) :=
sorry

end NUMINAMATH_CALUDE_perp_planes_necessary_not_sufficient_l1022_102275


namespace NUMINAMATH_CALUDE_divisibility_problem_l1022_102229

theorem divisibility_problem (a b n : ℤ) : 
  n = 10 * a + b → (17 ∣ (a - 5 * b)) → (17 ∣ n) := by sorry

end NUMINAMATH_CALUDE_divisibility_problem_l1022_102229


namespace NUMINAMATH_CALUDE_bus_departure_interval_l1022_102243

/-- Represents the time interval between bus departures -/
def bus_interval (total_time minutes_per_hour : ℕ) (num_buses : ℕ) : ℚ :=
  total_time / ((num_buses - 1) * minutes_per_hour)

theorem bus_departure_interval (total_time minutes_per_hour : ℕ) (num_buses : ℕ) 
  (h1 : total_time = 60) 
  (h2 : minutes_per_hour = 60)
  (h3 : num_buses = 11) :
  bus_interval total_time minutes_per_hour num_buses = 6 := by
  sorry

#eval bus_interval 60 60 11

end NUMINAMATH_CALUDE_bus_departure_interval_l1022_102243


namespace NUMINAMATH_CALUDE_f_max_min_on_interval_l1022_102213

-- Define the function f(x) = x^3 - 3x
def f (x : ℝ) : ℝ := x^3 - 3*x

-- Define the interval [-3, 0]
def interval : Set ℝ := {x | -3 ≤ x ∧ x ≤ 0}

-- Theorem stating the maximum and minimum values of f(x) on the given interval
theorem f_max_min_on_interval :
  ∃ (max min : ℝ), 
    (∀ x ∈ interval, f x ≤ max) ∧ 
    (∃ x ∈ interval, f x = max) ∧
    (∀ x ∈ interval, min ≤ f x) ∧ 
    (∃ x ∈ interval, f x = min) ∧
    max = 2 ∧ min = -18 :=
sorry

end NUMINAMATH_CALUDE_f_max_min_on_interval_l1022_102213


namespace NUMINAMATH_CALUDE_apples_left_l1022_102285

/-- Given that Mike picked 7.0 apples, Nancy ate 3.0 apples, and Keith picked 6.0 apples,
    prove that the number of apples left is 10.0. -/
theorem apples_left (mike_picked : ℝ) (nancy_ate : ℝ) (keith_picked : ℝ)
    (h1 : mike_picked = 7.0)
    (h2 : nancy_ate = 3.0)
    (h3 : keith_picked = 6.0) :
    mike_picked + keith_picked - nancy_ate = 10.0 := by
  sorry

end NUMINAMATH_CALUDE_apples_left_l1022_102285


namespace NUMINAMATH_CALUDE_product_closest_to_63_l1022_102292

theorem product_closest_to_63 : 
  let product := 2.1 * (30.3 + 0.13)
  ∀ x ∈ ({55, 60, 63, 65, 70} : Set ℝ), 
    x ≠ 63 → |product - 63| < |product - x| := by
  sorry

end NUMINAMATH_CALUDE_product_closest_to_63_l1022_102292


namespace NUMINAMATH_CALUDE_imaginary_part_of_one_plus_i_to_fifth_l1022_102257

def complex_i : ℂ := Complex.I

theorem imaginary_part_of_one_plus_i_to_fifth (h : complex_i ^ 2 = -1) :
  Complex.im ((1 : ℂ) + complex_i) ^ 5 = -4 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_one_plus_i_to_fifth_l1022_102257


namespace NUMINAMATH_CALUDE_triangle_properties_l1022_102227

-- Define the triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def satisfies_conditions (t : Triangle) : Prop :=
  -- Area condition
  3 * Real.sin t.A = (1/2) * t.b * t.c * Real.sin t.A ∧
  -- Perimeter condition
  t.a + t.b + t.c = 4 * (Real.sqrt 2 + 1) ∧
  -- Sine condition
  Real.sin t.B + Real.sin t.C = Real.sqrt 2 * Real.sin t.A

-- Theorem statement
theorem triangle_properties (t : Triangle) 
  (h : satisfies_conditions t) : 
  t.a = 4 ∧ 
  Real.cos t.A = 1/3 ∧ 
  Real.cos (2 * t.A - π/3) = (4 * Real.sqrt 6 - 7) / 18 := by
  sorry


end NUMINAMATH_CALUDE_triangle_properties_l1022_102227


namespace NUMINAMATH_CALUDE_min_value_of_expression_equality_achieved_l1022_102271

theorem min_value_of_expression (x : ℝ) : 
  (x + 2) * (x + 3) * (x + 4) * (x + 5) + 2024 ≥ 2023 :=
by sorry

theorem equality_achieved : 
  ∃ x : ℝ, (x + 2) * (x + 3) * (x + 4) * (x + 5) + 2024 = 2023 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_expression_equality_achieved_l1022_102271


namespace NUMINAMATH_CALUDE_y_intercept_of_line_l1022_102218

/-- Given a line l with parametric equations x = 4 - 4t and y = -2 + 3t, where t ∈ ℝ,
    the y-intercept of line l is 1. -/
theorem y_intercept_of_line (l : Set (ℝ × ℝ)) : 
  (∀ t : ℝ, (4 - 4*t, -2 + 3*t) ∈ l) → 
  (0, 1) ∈ l := by
  sorry

end NUMINAMATH_CALUDE_y_intercept_of_line_l1022_102218


namespace NUMINAMATH_CALUDE_rotate_180_proof_l1022_102220

/-- Represents a line in 2D space -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Rotates a line 180 degrees around the origin -/
def rotate180 (l : Line) : Line :=
  { a := l.a, b := l.b, c := -l.c }

theorem rotate_180_proof (l : Line) (h : l = { a := 1, b := -1, c := 4 }) :
  rotate180 l = { a := 1, b := -1, c := -4 } := by
  sorry

end NUMINAMATH_CALUDE_rotate_180_proof_l1022_102220


namespace NUMINAMATH_CALUDE_f_neg_one_eq_neg_one_l1022_102201

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the property of an odd function
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- State the theorem
theorem f_neg_one_eq_neg_one
  (h_odd : IsOdd f)
  (h_f_one : f 1 = 1) :
  f (-1) = -1 :=
sorry

end NUMINAMATH_CALUDE_f_neg_one_eq_neg_one_l1022_102201


namespace NUMINAMATH_CALUDE_jamies_class_girls_l1022_102283

theorem jamies_class_girls (total : ℕ) (girls boys : ℕ) : 
  total = 35 →
  4 * girls = 3 * boys →
  girls + boys = total →
  girls = 15 := by
sorry

end NUMINAMATH_CALUDE_jamies_class_girls_l1022_102283


namespace NUMINAMATH_CALUDE_odd_digits_in_base4_350_l1022_102249

-- Define a function to convert a number from base 10 to base 4
def toBase4 (n : ℕ) : List ℕ := sorry

-- Define a function to count odd digits in a list of digits
def countOddDigits (digits : List ℕ) : ℕ := sorry

-- Theorem statement
theorem odd_digits_in_base4_350 :
  countOddDigits (toBase4 350) = 4 := by sorry

end NUMINAMATH_CALUDE_odd_digits_in_base4_350_l1022_102249


namespace NUMINAMATH_CALUDE_quadratic_equation_conversion_l1022_102222

/-- Given a quadratic equation 2x^2 = 7x - 5, prove that when converted to the general form
    ax^2 + bx + c = 0, the coefficient of the linear term (b) is -7 and the constant term (c) is 5 -/
theorem quadratic_equation_conversion :
  ∃ (a b c : ℝ), (∀ x, 2 * x^2 = 7 * x - 5) →
  (∀ x, a * x^2 + b * x + c = 0) ∧ b = -7 ∧ c = 5 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_conversion_l1022_102222


namespace NUMINAMATH_CALUDE_casper_candy_problem_l1022_102248

theorem casper_candy_problem (initial_candies : ℕ) : 
  let day1_remaining := initial_candies - (initial_candies / 4) - 3
  let day2_after_eating := day1_remaining - (day1_remaining / 4)
  let day2_remaining := day2_after_eating + 5 - 5
  day2_remaining = 10 → initial_candies = 58 := by
  sorry

end NUMINAMATH_CALUDE_casper_candy_problem_l1022_102248


namespace NUMINAMATH_CALUDE_pauls_money_duration_l1022_102211

/-- Given Paul's earnings and spending, prove how long the money will last. -/
theorem pauls_money_duration (lawn_money weed_money weekly_spending : ℕ) 
  (h1 : lawn_money = 44)
  (h2 : weed_money = 28)
  (h3 : weekly_spending = 9) :
  (lawn_money + weed_money) / weekly_spending = 8 := by
  sorry

end NUMINAMATH_CALUDE_pauls_money_duration_l1022_102211


namespace NUMINAMATH_CALUDE_range_of_a_given_inequalities_and_unique_solution_l1022_102272

theorem range_of_a_given_inequalities_and_unique_solution :
  ∀ a : ℝ,
  (∃! x : ℤ, (2 * ↑x - 7 < 0 ∧ ↑x - a > 0)) →
  (2 ≤ a ∧ a < 3) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_given_inequalities_and_unique_solution_l1022_102272


namespace NUMINAMATH_CALUDE_line_points_product_l1022_102236

theorem line_points_product (x y : ℝ) : 
  (∃ k : Set (ℝ × ℝ), 
    (∀ p : ℝ × ℝ, p ∈ k ↔ p.2 = (1/4) * p.1) ∧ 
    (x, 8) ∈ k ∧ 
    (20, y) ∈ k ∧ 
    x * y = 160) → 
  y = 5 := by
sorry

end NUMINAMATH_CALUDE_line_points_product_l1022_102236


namespace NUMINAMATH_CALUDE_fraction_problem_l1022_102245

theorem fraction_problem (p q : ℚ) (h : p / q = 4 / 5) :
  ∃ x : ℚ, 11 / 7 + x / (2 * q + p) = 2 ∧ x = 6 :=
by sorry

end NUMINAMATH_CALUDE_fraction_problem_l1022_102245


namespace NUMINAMATH_CALUDE_sheets_per_class_calculation_l1022_102273

/-- The number of sheets of paper used by the school per week -/
def sheets_per_week : ℕ := 9000

/-- The number of school days per week -/
def school_days_per_week : ℕ := 5

/-- The number of classes in the school -/
def num_classes : ℕ := 9

/-- The number of sheets of paper each class uses per day -/
def sheets_per_class_per_day : ℕ := sheets_per_week / school_days_per_week / num_classes

theorem sheets_per_class_calculation :
  sheets_per_class_per_day = 200 := by
  sorry

end NUMINAMATH_CALUDE_sheets_per_class_calculation_l1022_102273


namespace NUMINAMATH_CALUDE_chaz_final_floor_l1022_102262

def elevator_problem (start_floor : ℕ) (first_down : ℕ) (second_down : ℕ) : ℕ :=
  start_floor - first_down - second_down

theorem chaz_final_floor :
  elevator_problem 11 2 4 = 5 := by
  sorry

end NUMINAMATH_CALUDE_chaz_final_floor_l1022_102262


namespace NUMINAMATH_CALUDE_original_ratio_proof_l1022_102226

theorem original_ratio_proof (x y : ℕ+) (h1 : y = 24) (h2 : (x + 6 : ℚ) / y = 1 / 2) : 
  (x : ℚ) / y = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_original_ratio_proof_l1022_102226


namespace NUMINAMATH_CALUDE_parabola_m_value_l1022_102216

/-- A parabola with equation x² = my and a point M(x₀, -3) on it. -/
structure Parabola where
  m : ℝ
  x₀ : ℝ
  eq : x₀^2 = m * (-3)

/-- The distance from a point to the focus of the parabola. -/
def distance_to_focus (p : Parabola) : ℝ := 5

/-- Theorem: If a point M(x₀, -3) on the parabola x² = my has a distance of 5 to the focus, then m = -8. -/
theorem parabola_m_value (p : Parabola) (h : distance_to_focus p = 5) : p.m = -8 := by
  sorry

end NUMINAMATH_CALUDE_parabola_m_value_l1022_102216


namespace NUMINAMATH_CALUDE_seed_to_sprout_probability_is_correct_l1022_102244

/-- The germination rate of a batch of seeds -/
def germination_rate : ℝ := 0.9

/-- The survival rate of sprouts after germination -/
def survival_rate : ℝ := 0.8

/-- The probability that a randomly selected seed will grow into a sprout -/
def seed_to_sprout_probability : ℝ := germination_rate * survival_rate

/-- Theorem: The probability that a randomly selected seed will grow into a sprout is 0.72 -/
theorem seed_to_sprout_probability_is_correct : seed_to_sprout_probability = 0.72 := by
  sorry

end NUMINAMATH_CALUDE_seed_to_sprout_probability_is_correct_l1022_102244


namespace NUMINAMATH_CALUDE_probability_of_pirate_letter_l1022_102242

def probability_letters : Finset Char := {'P', 'R', 'O', 'B', 'A', 'I', 'L', 'T', 'Y'}
def pirate_letters : Finset Char := {'P', 'I', 'R', 'A', 'T', 'E'}

def total_tiles : ℕ := 11

theorem probability_of_pirate_letter :
  (probability_letters ∩ pirate_letters).card / total_tiles = 5 / 11 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_pirate_letter_l1022_102242


namespace NUMINAMATH_CALUDE_rectangular_solid_diagonal_l1022_102297

/-- 
Given a rectangular solid with edge lengths a, b, and c,
if the total surface area is 22 and the total edge length is 24,
then the length of any interior diagonal is √14.
-/
theorem rectangular_solid_diagonal (a b c : ℝ) 
  (h1 : 2 * (a * b + b * c + a * c) = 22)
  (h2 : 4 * (a + b + c) = 24) :
  Real.sqrt (a^2 + b^2 + c^2) = Real.sqrt 14 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_solid_diagonal_l1022_102297


namespace NUMINAMATH_CALUDE_bottle_cap_weight_l1022_102200

theorem bottle_cap_weight (caps_per_ounce : ℕ) (total_caps : ℕ) (total_weight : ℕ) :
  caps_per_ounce = 7 →
  total_caps = 2016 →
  total_weight = total_caps / caps_per_ounce →
  total_weight = 288 :=
by sorry

end NUMINAMATH_CALUDE_bottle_cap_weight_l1022_102200


namespace NUMINAMATH_CALUDE_tyrone_eric_marbles_l1022_102266

/-- Proves that Tyrone gave 10 marbles to Eric -/
theorem tyrone_eric_marbles : ∀ x : ℕ,
  (100 : ℕ) - x = 3 * ((20 : ℕ) + x) → x = 10 := by
  sorry

end NUMINAMATH_CALUDE_tyrone_eric_marbles_l1022_102266


namespace NUMINAMATH_CALUDE_sum_difference_is_thirteen_l1022_102241

def star_list : List Nat := List.range 30

def replace_three_with_two (n : Nat) : Nat :=
  let s := toString n
  (s.replace "3" "2").toNat!

def emilio_list : List Nat :=
  star_list.map replace_three_with_two

theorem sum_difference_is_thirteen :
  star_list.sum - emilio_list.sum = 13 := by
  sorry

end NUMINAMATH_CALUDE_sum_difference_is_thirteen_l1022_102241


namespace NUMINAMATH_CALUDE_y_equation_proof_l1022_102206

theorem y_equation_proof (y : ℝ) (h : y + 1/y = 3) : y^6 - 8*y^3 + 4*y = 20*y - 5 := by
  sorry

end NUMINAMATH_CALUDE_y_equation_proof_l1022_102206


namespace NUMINAMATH_CALUDE_range_of_m_l1022_102298

def α (x : ℝ) : Prop := x^2 - 3*x - 10 ≤ 0

def β (m x : ℝ) : Prop := m - 3 ≤ x ∧ x ≤ m + 6

theorem range_of_m :
  (∀ x, α x → ∃ m, β m x) →
  ∀ m, (∃ x, β m x) → -1 ≤ m ∧ m ≤ 1 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l1022_102298


namespace NUMINAMATH_CALUDE_jorge_simon_age_difference_l1022_102269

/-- Represents a person's age at a given year -/
structure AgeAtYear where
  age : ℕ
  year : ℕ

/-- Calculates the age difference between two people -/
def ageDifference (person1 : AgeAtYear) (person2 : AgeAtYear) : ℕ :=
  if person1.year = person2.year then
    if person1.age ≥ person2.age then person1.age - person2.age else person2.age - person1.age
  else
    sorry -- We don't handle different years in this simplified version

theorem jorge_simon_age_difference :
  let jorge2005 : AgeAtYear := { age := 16, year := 2005 }
  let simon2010 : AgeAtYear := { age := 45, year := 2010 }
  let yearDiff : ℕ := simon2010.year - jorge2005.year
  let jorgeAge2010 : ℕ := jorge2005.age + yearDiff
  ageDifference { age := simon2010.age, year := simon2010.year } { age := jorgeAge2010, year := simon2010.year } = 24 := by
  sorry


end NUMINAMATH_CALUDE_jorge_simon_age_difference_l1022_102269


namespace NUMINAMATH_CALUDE_a_range_l1022_102255

-- Define proposition P
def P (a : ℝ) : Prop := ∀ x : ℝ, x^2 + a*x + 4 > 0

-- Define proposition Q
def Q (a : ℝ) : Prop := ∃ x : ℝ, x^2 - 2*x + a = 0

-- Theorem statement
theorem a_range (a : ℝ) : P a ∧ ¬(Q a) ↔ 1 < a ∧ a < 4 := by
  sorry

end NUMINAMATH_CALUDE_a_range_l1022_102255


namespace NUMINAMATH_CALUDE_sum_of_square_and_two_cubes_sum_of_square_and_three_cubes_l1022_102293

-- Part (a)
theorem sum_of_square_and_two_cubes (k : ℤ) :
  ∃ (a b c : ℤ), 3 * k - 2 = a^2 + b^3 + c^3 := by sorry

-- Part (b)
theorem sum_of_square_and_three_cubes (n : ℤ) :
  ∃ (w x y z : ℤ), n = w^2 + x^3 + y^3 + z^3 := by sorry

end NUMINAMATH_CALUDE_sum_of_square_and_two_cubes_sum_of_square_and_three_cubes_l1022_102293


namespace NUMINAMATH_CALUDE_average_fruits_per_basket_l1022_102280

-- Define the number of baskets
def num_baskets : ℕ := 5

-- Define the number of fruits in each basket
def basket_A : ℕ := 15
def basket_B : ℕ := 30
def basket_C : ℕ := 20
def basket_D : ℕ := 25
def basket_E : ℕ := 35

-- Define the total number of fruits
def total_fruits : ℕ := basket_A + basket_B + basket_C + basket_D + basket_E

-- Theorem: The average number of fruits per basket is 25
theorem average_fruits_per_basket : 
  total_fruits / num_baskets = 25 := by
  sorry

end NUMINAMATH_CALUDE_average_fruits_per_basket_l1022_102280


namespace NUMINAMATH_CALUDE_card_purchase_cost_l1022_102270

/-- Calculates the total cost of cards purchased from two boxes, including sales tax. -/
def total_cost (price1 : ℚ) (price2 : ℚ) (count1 : ℕ) (count2 : ℕ) (tax_rate : ℚ) : ℚ :=
  let subtotal := price1 * count1 + price2 * count2
  subtotal * (1 + tax_rate)

/-- Proves that the total cost of 8 cards from the first box and 12 cards from the second box, including 7% sales tax, is $33.17. -/
theorem card_purchase_cost : 
  total_cost (25/20) (35/20) 8 12 (7/100) = 3317/100 := by
  sorry

#eval total_cost (25/20) (35/20) 8 12 (7/100)

end NUMINAMATH_CALUDE_card_purchase_cost_l1022_102270


namespace NUMINAMATH_CALUDE_trig_expression_simplification_l1022_102233

theorem trig_expression_simplification :
  (Real.sin (15 * π / 180) + Real.sin (30 * π / 180) + Real.sin (45 * π / 180) + 
   Real.sin (60 * π / 180) + Real.sin (75 * π / 180)) / 
  (Real.cos (10 * π / 180) * Real.cos (20 * π / 180) * Real.cos (30 * π / 180)) =
  (Real.sqrt 2 * (4 * Real.cos (22.5 * π / 180) * Real.cos (7.5 * π / 180) + 1)) / 
  (Real.cos (10 * π / 180) * Real.cos (20 * π / 180) * Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_trig_expression_simplification_l1022_102233


namespace NUMINAMATH_CALUDE_cyclist_speed_calculation_l1022_102240

/-- Given two cyclists, Joann and Fran, this theorem proves the required speed for Fran
    to cover the same distance as Joann in a different amount of time. -/
theorem cyclist_speed_calculation (joann_speed joann_time fran_time : ℝ) 
    (hjs : joann_speed = 15) 
    (hjt : joann_time = 4)
    (hft : fran_time = 5) : 
  joann_speed * joann_time / fran_time = 12 := by
  sorry

#check cyclist_speed_calculation

end NUMINAMATH_CALUDE_cyclist_speed_calculation_l1022_102240


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_equation_l1022_102287

theorem negation_of_existence (P : ℝ → Prop) : 
  (¬ ∃ x : ℝ, P x) ↔ (∀ x : ℝ, ¬ P x) :=
by sorry

theorem negation_of_quadratic_equation : 
  (¬ ∃ x : ℝ, x^2 + x + 1 = 0) ↔ (∀ x : ℝ, x^2 + x + 1 ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_equation_l1022_102287


namespace NUMINAMATH_CALUDE_equation_solution_l1022_102286

theorem equation_solution : ∃ y : ℤ, (2010 + 2*y)^2 = 4*y^2 ∧ y = -1005 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1022_102286


namespace NUMINAMATH_CALUDE_secret_spread_exceeds_3000_l1022_102221

def secret_spread (n : ℕ) : ℕ := 3^(n-1)

theorem secret_spread_exceeds_3000 :
  ∃ (n : ℕ), n = 9 ∧ secret_spread n > 3000 :=
by sorry

end NUMINAMATH_CALUDE_secret_spread_exceeds_3000_l1022_102221


namespace NUMINAMATH_CALUDE_h_not_prime_l1022_102212

/-- The function h(n) as defined in the problem -/
def h (n : ℕ+) : ℤ := n.val^4 - 500 * n.val^2 + 625

/-- Theorem stating that h(n) is not prime for any positive integer n -/
theorem h_not_prime : ∀ n : ℕ+, ¬ Nat.Prime (Int.natAbs (h n)) := by sorry

end NUMINAMATH_CALUDE_h_not_prime_l1022_102212


namespace NUMINAMATH_CALUDE_line_equation_from_circle_intersection_l1022_102259

/-- Given a circle and a line intersecting it, prove the equation of the line. -/
theorem line_equation_from_circle_intersection (a : ℝ) (h_a : a < 3) :
  let circle := fun (x y : ℝ) => x^2 + y^2 + 2*x - 4*y + a = 0
  let midpoint := (-2, 3)
  ∃ (A B : ℝ × ℝ),
    circle A.1 A.2 ∧
    circle B.1 B.2 ∧
    (A.1 + B.1) / 2 = midpoint.1 ∧
    (A.2 + B.2) / 2 = midpoint.2 →
    ∃ (m b : ℝ), ∀ (x y : ℝ), y = m*x + b ↔ x - y + 5 = 0 :=
by sorry

end NUMINAMATH_CALUDE_line_equation_from_circle_intersection_l1022_102259


namespace NUMINAMATH_CALUDE_gift_wrapping_combinations_l1022_102263

/-- The number of varieties of wrapping paper -/
def wrapping_paper_varieties : ℕ := 10

/-- The number of colors of ribbon -/
def ribbon_colors : ℕ := 4

/-- The number of types of gift cards -/
def gift_card_types : ℕ := 5

/-- Whether red ribbon is available -/
def red_ribbon_available : Prop := true

/-- The number of invalid combinations due to supply issue -/
def invalid_combinations : ℕ := 5

theorem gift_wrapping_combinations : 
  wrapping_paper_varieties * ribbon_colors * gift_card_types - invalid_combinations = 195 := by
  sorry

end NUMINAMATH_CALUDE_gift_wrapping_combinations_l1022_102263


namespace NUMINAMATH_CALUDE_complex_simplification_l1022_102291

theorem complex_simplification (i : ℂ) (h : i^2 = -1) :
  (1 - 3*i) / (1 - i) = 2 - i := by
  sorry

end NUMINAMATH_CALUDE_complex_simplification_l1022_102291


namespace NUMINAMATH_CALUDE_floor_plus_self_unique_solution_l1022_102203

theorem floor_plus_self_unique_solution (r : ℝ) : 
  (⌊r⌋ : ℝ) + r = 16.5 ↔ r = 8.5 := by
  sorry

end NUMINAMATH_CALUDE_floor_plus_self_unique_solution_l1022_102203


namespace NUMINAMATH_CALUDE_tailor_time_calculation_l1022_102252

-- Define the time ratios
def shirt_time : ℚ := 1
def pants_time : ℚ := 2
def jacket_time : ℚ := 3

-- Define the reference quantities
def ref_shirts : ℕ := 2
def ref_pants : ℕ := 3
def ref_jackets : ℕ := 4
def ref_total_time : ℚ := 10

-- Define the quantities to calculate
def calc_shirts : ℕ := 14
def calc_pants : ℕ := 10
def calc_jackets : ℕ := 2

-- Theorem statement
theorem tailor_time_calculation :
  let base_time := ref_total_time / (ref_shirts * shirt_time + ref_pants * pants_time + ref_jackets * jacket_time)
  calc_shirts * (base_time * shirt_time) + calc_pants * (base_time * pants_time) + calc_jackets * (base_time * jacket_time) = 20 := by
  sorry

end NUMINAMATH_CALUDE_tailor_time_calculation_l1022_102252


namespace NUMINAMATH_CALUDE_initial_brownies_count_l1022_102295

/-- Represents the number of days in a week -/
def week : ℕ := 7

/-- Represents the number of cookies eaten per day -/
def cookiesPerDay : ℕ := 3

/-- Represents the number of brownies eaten per day -/
def browniesPerDay : ℕ := 3

/-- Represents the difference between cookies and brownies after a week -/
def cookieBrownieDifference : ℕ := 36

/-- 
Theorem: If a person eats 3 cookies and 3 brownies per day for a week, 
and ends up with 36 more cookies than brownies, 
then they must have started with 36 brownies.
-/
theorem initial_brownies_count 
  (initialCookies initialBrownies : ℕ) : 
  initialCookies - week * cookiesPerDay = initialBrownies - week * browniesPerDay + cookieBrownieDifference →
  initialBrownies = 36 := by
  sorry

end NUMINAMATH_CALUDE_initial_brownies_count_l1022_102295


namespace NUMINAMATH_CALUDE_a_not_periodic_l1022_102279

/-- The first digit of a positive integer -/
def firstDigit (n : ℕ) : ℕ :=
  if n < 10 then n else firstDigit (n / 10)

/-- The sequence a_n where a_n is the first digit of n^2 -/
def a (n : ℕ) : ℕ :=
  firstDigit (n * n)

/-- A sequence is periodic if there exists a positive integer p such that
    for all n ≥ some N, a(n+p) = a(n) -/
def isPeriodic (f : ℕ → ℕ) : Prop :=
  ∃ p N : ℕ, p > 0 ∧ ∀ n ≥ N, f (n + p) = f n

/-- The sequence a_n is not periodic -/
theorem a_not_periodic : ¬ isPeriodic a := by
  sorry

end NUMINAMATH_CALUDE_a_not_periodic_l1022_102279


namespace NUMINAMATH_CALUDE_sum_18_29_in_base3_l1022_102254

/-- Converts a natural number from base 10 to base 3 -/
def toBase3 (n : ℕ) : List ℕ :=
  sorry

/-- Converts a list of digits in base 3 to a natural number in base 10 -/
def fromBase3 (digits : List ℕ) : ℕ :=
  sorry

theorem sum_18_29_in_base3 :
  toBase3 (18 + 29) = [1, 2, 0, 2] :=
sorry

end NUMINAMATH_CALUDE_sum_18_29_in_base3_l1022_102254


namespace NUMINAMATH_CALUDE_brown_dogs_l1022_102296

def kennel (total : ℕ) (long_fur : ℕ) (neither : ℕ) : Prop :=
  total = 45 ∧
  long_fur = 36 ∧
  neither = 8 ∧
  long_fur ≤ total ∧
  neither ≤ total - long_fur

theorem brown_dogs (total long_fur neither : ℕ) 
  (h : kennel total long_fur neither) : ∃ brown : ℕ, brown = 37 :=
sorry

end NUMINAMATH_CALUDE_brown_dogs_l1022_102296


namespace NUMINAMATH_CALUDE_perpendicular_chords_theorem_l1022_102289

/-- 
Given a circle with radius R and two perpendicular chords intersecting at point M,
this theorem proves two properties:
1. The sum of squares of the four segments formed by the intersection is 4R^2.
2. If the distance from the center to M is d, the sum of squares of chord lengths is 8R^2 - 4d^2.
-/
theorem perpendicular_chords_theorem (R d : ℝ) (h : d ≥ 0) :
  ∃ (AM MB CM MD : ℝ),
    (AM ≥ 0) ∧ (MB ≥ 0) ∧ (CM ≥ 0) ∧ (MD ≥ 0) ∧
    (AM^2 + MB^2 + CM^2 + MD^2 = 4 * R^2) ∧
    ∃ (AB CD : ℝ),
      (AB ≥ 0) ∧ (CD ≥ 0) ∧
      (AB^2 + CD^2 = 8 * R^2 - 4 * d^2) := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_chords_theorem_l1022_102289


namespace NUMINAMATH_CALUDE_factorization_problem_1_factorization_problem_2_l1022_102261

-- Problem 1
theorem factorization_problem_1 (x y : ℝ) :
  4 - 12 * (x - y) + 9 * (x - y)^2 = (2 - 3*x + 3*y)^2 := by sorry

-- Problem 2
theorem factorization_problem_2 (a x : ℝ) :
  2*a*(x^2 + 1)^2 - 8*a*x^2 = 2*a*(x - 1)^2*(x + 1)^2 := by sorry

end NUMINAMATH_CALUDE_factorization_problem_1_factorization_problem_2_l1022_102261


namespace NUMINAMATH_CALUDE_blake_change_l1022_102232

theorem blake_change (oranges apples mangoes initial : ℕ) 
  (h_oranges : oranges = 40)
  (h_apples : apples = 50)
  (h_mangoes : mangoes = 60)
  (h_initial : initial = 300) :
  initial - (oranges + apples + mangoes) = 150 := by
  sorry

end NUMINAMATH_CALUDE_blake_change_l1022_102232


namespace NUMINAMATH_CALUDE_overlapping_triangles_area_l1022_102209

/-- The area common to two overlapping right triangles -/
theorem overlapping_triangles_area :
  let triangle1_hypotenuse : ℝ := 10
  let triangle1_angle1 : ℝ := 30 * π / 180
  let triangle1_angle2 : ℝ := 60 * π / 180
  let triangle2_hypotenuse : ℝ := 15
  let triangle2_angle1 : ℝ := 45 * π / 180
  let triangle2_angle2 : ℝ := 45 * π / 180
  let overlap_length : ℝ := 5
  ∃ (common_area : ℝ), common_area = (25 * Real.sqrt 3) / 8 := by
  sorry

end NUMINAMATH_CALUDE_overlapping_triangles_area_l1022_102209


namespace NUMINAMATH_CALUDE_expansion_coefficients_l1022_102278

theorem expansion_coefficients :
  let expr := (1 + X^5 + X^7)^20
  ∃ (p : Polynomial ℤ),
    p = expr ∧
    p.coeff 18 = 0 ∧
    p.coeff 17 = 3420 :=
by sorry

end NUMINAMATH_CALUDE_expansion_coefficients_l1022_102278


namespace NUMINAMATH_CALUDE_prob_one_heads_is_half_l1022_102250

/-- A coin toss outcome -/
inductive CoinToss
| Heads
| Tails

/-- Result of two successive coin tosses -/
def TwoTosses := (CoinToss × CoinToss)

/-- All possible outcomes of two successive coin tosses -/
def allOutcomes : Finset TwoTosses := sorry

/-- Outcomes with exactly one heads -/
def oneHeadsOutcomes : Finset TwoTosses := sorry

/-- Probability of an event in a finite sample space -/
def probability (event : Finset TwoTosses) : ℚ :=
  (event.card : ℚ) / (allOutcomes.card : ℚ)

theorem prob_one_heads_is_half :
  probability oneHeadsOutcomes = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_prob_one_heads_is_half_l1022_102250


namespace NUMINAMATH_CALUDE_print_shop_cost_difference_l1022_102290

/-- Calculates the total cost for color copies at a print shop --/
def calculate_total_cost (base_price : ℝ) (quantity : ℕ) (discount_threshold : ℕ) (discount_rate : ℝ) (tax_rate : ℝ) : ℝ :=
  let base_cost := base_price * quantity
  let discounted_cost := if quantity > discount_threshold then base_cost * (1 - discount_rate) else base_cost
  discounted_cost * (1 + tax_rate)

/-- Proves that the difference in cost for 40 color copies between Print Shop Y and Print Shop X is $27.40 --/
theorem print_shop_cost_difference : 
  let shop_x_cost := calculate_total_cost 1.20 40 30 0.10 0.05
  let shop_y_cost := calculate_total_cost 1.70 40 50 0.15 0.07
  shop_y_cost - shop_x_cost = 27.40 := by
  sorry

end NUMINAMATH_CALUDE_print_shop_cost_difference_l1022_102290


namespace NUMINAMATH_CALUDE_positive_quadratic_intervals_l1022_102258

theorem positive_quadratic_intervals (x : ℝ) : 
  (x - 2) * (x + 3) > 0 ↔ x < -3 ∨ x > 2 := by sorry

end NUMINAMATH_CALUDE_positive_quadratic_intervals_l1022_102258


namespace NUMINAMATH_CALUDE_mental_competition_result_l1022_102231

/-- Represents the number of students who correctly answered each problem -/
structure ProblemCounts where
  a : ℕ
  b : ℕ
  c : ℕ

/-- Represents the scores for each problem -/
def problem_scores : Fin 3 → ℕ
  | 0 => 20  -- Problem a
  | 1 => 25  -- Problem b
  | 2 => 25  -- Problem c
  | _ => 0   -- This case should never occur due to Fin 3

theorem mental_competition_result 
  (counts : ProblemCounts)
  (h1 : counts.a + counts.b = 29)
  (h2 : counts.a + counts.c = 25)
  (h3 : counts.b + counts.c = 20)
  (h4 : counts.a + counts.b + counts.c ≥ 1 + 3 * 15 + 1)  -- At least one correct + 15 with two correct + one with all correct
  (h5 : counts.a + counts.b + counts.c - (3 + 2 * 15) ≥ 0)  -- Non-negative number of students with only one correct
  : 
  (counts.a + counts.b + counts.c - (3 + 2 * 15) = 4) ∧  -- 4 students answered only one question correctly
  (((counts.a * problem_scores 0) + (counts.b * problem_scores 1) + (counts.c * problem_scores 2) + 70) / (counts.a + counts.b + counts.c - (3 + 2 * 15) + 15 + 1) = 42) -- Average score is 42
  := by sorry


end NUMINAMATH_CALUDE_mental_competition_result_l1022_102231


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1022_102288

theorem complex_equation_solution (a b : ℝ) : 
  (Complex.I + a) * (1 + Complex.I) = b * Complex.I → a + b * Complex.I = 1 + 2 * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1022_102288


namespace NUMINAMATH_CALUDE_min_perimeter_triangle_l1022_102205

-- Define the plane
variable (Plane : Type)

-- Define points in the plane
variable (O P A B P1 P2 : Plane)

-- Define the angle
variable (angle : Plane → Plane → Plane → Prop)

-- Define the property of being inside an angle
variable (inside_angle : Plane → Plane → Plane → Plane → Prop)

-- Define the property of a point being on a line
variable (on_line : Plane → Plane → Plane → Prop)

-- Define the reflection of a point over a line
variable (reflect : Plane → Plane → Plane → Plane)

-- Define the perimeter of a triangle
variable (perimeter : Plane → Plane → Plane → ℝ)

-- Define the theorem
theorem min_perimeter_triangle 
  (h_acute : angle O A B)
  (h_inside : inside_angle O A B P)
  (h_P1 : P1 = reflect O A P)
  (h_P2 : P2 = reflect O B P)
  (h_A_on_side : on_line O A A)
  (h_B_on_side : on_line O B B)
  (h_A_on_P1P2 : on_line P1 P2 A)
  (h_B_on_P1P2 : on_line P1 P2 B) :
  ∀ A' B', on_line O A A' → on_line O B B' → 
    perimeter P A B ≤ perimeter P A' B' :=
sorry

end NUMINAMATH_CALUDE_min_perimeter_triangle_l1022_102205


namespace NUMINAMATH_CALUDE_complex_sequence_counterexample_l1022_102274

-- Define the "sequence" relation on complex numbers
def complex_gt (z₁ z₂ : ℂ) : Prop :=
  z₁.re > z₂.re ∨ (z₁.re = z₂.re ∧ z₁.im > z₂.im)

-- Define positive complex numbers
def complex_pos (z : ℂ) : Prop :=
  complex_gt z 0

-- Theorem statement
theorem complex_sequence_counterexample :
  ∃ (z z₁ z₂ : ℂ), complex_pos z ∧ complex_gt z₁ z₂ ∧ ¬(complex_gt (z * z₁) (z * z₂)) := by
  sorry

end NUMINAMATH_CALUDE_complex_sequence_counterexample_l1022_102274


namespace NUMINAMATH_CALUDE_quadratic_transformation_l1022_102207

/-- The quadratic function we're working with -/
def f (x : ℝ) : ℝ := x^2 - 16*x + 15

/-- The transformed quadratic function -/
def g (x b c : ℝ) : ℝ := (x + b)^2 + c

theorem quadratic_transformation :
  ∃ b c : ℝ, (∀ x : ℝ, f x = g x b c) ∧ b + c = -57 := by sorry

end NUMINAMATH_CALUDE_quadratic_transformation_l1022_102207


namespace NUMINAMATH_CALUDE_exists_power_two_minus_one_divisible_by_n_l1022_102260

theorem exists_power_two_minus_one_divisible_by_n (n : ℕ) (h_odd : Odd n) (h_gt_one : n > 1) :
  ∃ k : ℕ, 1 ≤ k ∧ k < n ∧ (n ∣ 2^k - 1) :=
sorry

end NUMINAMATH_CALUDE_exists_power_two_minus_one_divisible_by_n_l1022_102260


namespace NUMINAMATH_CALUDE_smallest_k_divides_k_210_divides_smallest_k_is_210_l1022_102253

def f (z : ℂ) : ℂ := z^12 + z^11 + z^8 + z^7 + z^6 + z^3 + 1

theorem smallest_k_divides : 
  ∀ k : ℕ, k > 0 → (∀ z : ℂ, f z = 0 → z^k = 1) → k ≥ 210 :=
by sorry

theorem k_210_divides : 
  ∀ z : ℂ, f z = 0 → z^210 = 1 :=
by sorry

theorem smallest_k_is_210 : 
  (∃ k : ℕ, k > 0 ∧ (∀ z : ℂ, f z = 0 → z^k = 1)) ∧
  (∀ k : ℕ, k > 0 → (∀ z : ℂ, f z = 0 → z^k = 1) → k ≥ 210) ∧
  (∀ z : ℂ, f z = 0 → z^210 = 1) :=
by sorry

end NUMINAMATH_CALUDE_smallest_k_divides_k_210_divides_smallest_k_is_210_l1022_102253


namespace NUMINAMATH_CALUDE_range_of_g_l1022_102294

noncomputable def g (x : ℝ) : ℝ := Real.arctan (x^2) + Real.arctan ((2 - 2*x^2) / (1 + 2*x^2))

theorem range_of_g : ∀ x : ℝ, g x = Real.arctan 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_g_l1022_102294


namespace NUMINAMATH_CALUDE_christen_peeled_23_potatoes_l1022_102268

/-- Represents the potato peeling scenario -/
structure PotatoPeeling where
  totalPotatoes : ℕ
  homerRate : ℕ
  christenInitialRate : ℕ
  christenFinalRate : ℕ
  homerAloneTime : ℕ
  workTogetherTime : ℕ
  christenBreakTime : ℕ

/-- Calculates the number of potatoes Christen peeled -/
def christenPeeledPotatoes (scenario : PotatoPeeling) : ℕ :=
  sorry

/-- The theorem stating that Christen peeled 23 potatoes -/
theorem christen_peeled_23_potatoes :
  let scenario := PotatoPeeling.mk 60 4 6 4 5 3 2
  christenPeeledPotatoes scenario = 23 := by
  sorry

end NUMINAMATH_CALUDE_christen_peeled_23_potatoes_l1022_102268


namespace NUMINAMATH_CALUDE_f_decreasing_intervals_l1022_102264

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin (ω * x) + Real.cos (ω * x)

theorem f_decreasing_intervals (ω : ℝ) (h_ω : ω > 0) 
  (h_period : ∀ x : ℝ, f ω (x + π / ω) = f ω x) :
  ∀ k : ℤ, StrictMonoOn (f ω) (Set.Icc (k * π + π / 6) (k * π + π / 3)) := by
  sorry

end NUMINAMATH_CALUDE_f_decreasing_intervals_l1022_102264


namespace NUMINAMATH_CALUDE_vectors_perpendicular_l1022_102225

def vector_angle (u v : ℝ × ℝ) : ℝ := sorry

theorem vectors_perpendicular : 
  let u : ℝ × ℝ := (3, -4)
  let v : ℝ × ℝ := (4, 3)
  vector_angle u v = 90 := by sorry

end NUMINAMATH_CALUDE_vectors_perpendicular_l1022_102225


namespace NUMINAMATH_CALUDE_triangle_angle_sum_l1022_102217

theorem triangle_angle_sum (A B C : ℝ) (h1 : A = 75) (h2 : B = 40) : C = 65 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_sum_l1022_102217


namespace NUMINAMATH_CALUDE_digit_sum_multiple_of_nine_l1022_102299

/-- The digit sum of a natural number -/
def digitSum (n : ℕ) : ℕ := sorry

/-- Theorem: If a number n and 3n have the same digit sum, then n is divisible by 9 -/
theorem digit_sum_multiple_of_nine (n : ℕ) : digitSum n = digitSum (3 * n) → 9 ∣ n := by
  sorry

end NUMINAMATH_CALUDE_digit_sum_multiple_of_nine_l1022_102299


namespace NUMINAMATH_CALUDE_largest_three_digit_number_with_conditions_l1022_102256

theorem largest_three_digit_number_with_conditions : ∃ n : ℕ, 
  (n ≤ 999 ∧ n ≥ 100) ∧ 
  (∃ k : ℕ, n = 7 * k + 2) ∧ 
  (∃ m : ℕ, n = 4 * m + 1) ∧ 
  (∀ x : ℕ, (x ≤ 999 ∧ x ≥ 100) → 
    (∃ k : ℕ, x = 7 * k + 2) → 
    (∃ m : ℕ, x = 4 * m + 1) → 
    x ≤ n) ∧
  n = 989 :=
by sorry

end NUMINAMATH_CALUDE_largest_three_digit_number_with_conditions_l1022_102256


namespace NUMINAMATH_CALUDE_factors_of_72_l1022_102215

theorem factors_of_72 : Nat.card (Nat.divisors 72) = 12 := by
  sorry

end NUMINAMATH_CALUDE_factors_of_72_l1022_102215


namespace NUMINAMATH_CALUDE_larger_solution_of_quadratic_l1022_102204

theorem larger_solution_of_quadratic (x : ℝ) : 
  (2 * x^2 - 14 * x - 84 = 0) → (∃ y : ℝ, 2 * y^2 - 14 * y - 84 = 0 ∧ y ≠ x) → 
  (x = 14 ∨ x = -3) → (x = 14 ∨ (x = -3 ∧ 14 > x)) :=
sorry

end NUMINAMATH_CALUDE_larger_solution_of_quadratic_l1022_102204


namespace NUMINAMATH_CALUDE_parents_average_age_l1022_102247

theorem parents_average_age
  (num_grandparents num_parents num_grandchildren : ℕ)
  (avg_age_grandparents avg_age_grandchildren avg_age_family : ℚ)
  (h1 : num_grandparents = 2)
  (h2 : num_parents = 2)
  (h3 : num_grandchildren = 3)
  (h4 : avg_age_grandparents = 64)
  (h5 : avg_age_grandchildren = 6)
  (h6 : avg_age_family = 32)
  (h7 : (num_grandparents + num_parents + num_grandchildren : ℚ) * avg_age_family =
        num_grandparents * avg_age_grandparents +
        num_parents * (num_grandparents * avg_age_grandparents + num_parents * avg_age_family + num_grandchildren * avg_age_grandchildren - (num_grandparents + num_parents + num_grandchildren) * avg_age_family) / num_parents +
        num_grandchildren * avg_age_grandchildren) :
  (num_grandparents * avg_age_grandparents + num_parents * avg_age_family + num_grandchildren * avg_age_grandchildren - (num_grandparents + num_parents + num_grandchildren) * avg_age_family) / num_parents = 39 :=
sorry

end NUMINAMATH_CALUDE_parents_average_age_l1022_102247


namespace NUMINAMATH_CALUDE_function_properties_l1022_102202

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f x = -f (-x)
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

theorem function_properties (f : ℝ → ℝ) 
  (h1 : is_odd (fun x ↦ f (x + 1/2)))
  (h2 : ∀ x, f (2 - 3*x) = f (3*x)) :
  f (-1/2) = 0 ∧ 
  is_even (fun x ↦ f (x + 2)) ∧ 
  is_odd (fun x ↦ f (x - 1/2)) := by
sorry

end NUMINAMATH_CALUDE_function_properties_l1022_102202


namespace NUMINAMATH_CALUDE_coordinates_of_point_B_l1022_102277

/-- Given a 2D coordinate system with origin O, point A at (-1, 2), 
    and vector BA = (3, 3), prove that the coordinates of point B are (-4, -1) -/
theorem coordinates_of_point_B (O A B : ℝ × ℝ) : 
  O = (0, 0) → 
  A = (-1, 2) → 
  B - A = (3, 3) →
  B = (-4, -1) := by
sorry

end NUMINAMATH_CALUDE_coordinates_of_point_B_l1022_102277


namespace NUMINAMATH_CALUDE_function_inequality_l1022_102246

theorem function_inequality (a : ℝ) (h1 : 1 < a) (h2 : a ≤ 8/5) :
  ∀ x : ℝ, a ≤ x ∧ x ≤ 2*a - 1 → |x + a| + |2*x - 3| ≤ |x + 3| := by
sorry

end NUMINAMATH_CALUDE_function_inequality_l1022_102246


namespace NUMINAMATH_CALUDE_aaron_reading_challenge_l1022_102214

theorem aaron_reading_challenge (average_pages : ℕ) (total_days : ℕ) (day1 day2 day3 day4 day5 : ℕ) :
  average_pages = 15 →
  total_days = 6 →
  day1 = 18 →
  day2 = 12 →
  day3 = 23 →
  day4 = 10 →
  day5 = 17 →
  ∃ (day6 : ℕ), (day1 + day2 + day3 + day4 + day5 + day6) / total_days = average_pages ∧ day6 = 10 :=
by sorry

end NUMINAMATH_CALUDE_aaron_reading_challenge_l1022_102214


namespace NUMINAMATH_CALUDE_zuca_win_probability_l1022_102234

/-- The Game played on a regular hexagon --/
structure TheGame where
  /-- Number of vertices in the hexagon --/
  vertices : Nat
  /-- Number of players --/
  players : Nat
  /-- Probability of Bamal and Halvan moving to adjacent vertices --/
  prob_adjacent : ℚ
  /-- Probability of Zuca moving to adjacent or opposite vertices --/
  prob_zuca_move : ℚ

/-- The specific instance of The Game as described in the problem --/
def gameInstance : TheGame :=
  { vertices := 6
  , players := 3
  , prob_adjacent := 1/2
  , prob_zuca_move := 1/3 }

/-- The probability that Zuca hasn't lost when The Game ends --/
def probZucaWins (g : TheGame) : ℚ :=
  29/90

/-- Theorem stating that the probability of Zuca not losing is 29/90 --/
theorem zuca_win_probability (g : TheGame) :
  g = gameInstance → probZucaWins g = 29/90 := by
  sorry

end NUMINAMATH_CALUDE_zuca_win_probability_l1022_102234


namespace NUMINAMATH_CALUDE_molecular_weight_4_moles_BaBr2_l1022_102208

-- Define atomic weights
def atomic_weight_Ba : ℝ := 137.33
def atomic_weight_Br : ℝ := 79.90

-- Define molecular weight of BaBr2
def molecular_weight_BaBr2 : ℝ := atomic_weight_Ba + 2 * atomic_weight_Br

-- Define the number of moles
def moles : ℝ := 4

-- Theorem statement
theorem molecular_weight_4_moles_BaBr2 :
  moles * molecular_weight_BaBr2 = 1188.52 := by sorry

end NUMINAMATH_CALUDE_molecular_weight_4_moles_BaBr2_l1022_102208


namespace NUMINAMATH_CALUDE_cubic_value_given_quadratic_l1022_102224

theorem cubic_value_given_quadratic (x : ℝ) : 
  x^2 + x - 1 = 0 → x^3 + 2*x^2 + 2005 = 2006 := by
  sorry

end NUMINAMATH_CALUDE_cubic_value_given_quadratic_l1022_102224


namespace NUMINAMATH_CALUDE_jeremy_scrabble_score_l1022_102230

/-- Calculates the score for a three-letter word in Scrabble with given letter values and a triple word score -/
def scrabble_score (first_letter_value : ℕ) (middle_letter_value : ℕ) (last_letter_value : ℕ) : ℕ :=
  3 * (first_letter_value + middle_letter_value + last_letter_value)

/-- Theorem: The score for Jeremy's word is 30 points -/
theorem jeremy_scrabble_score :
  scrabble_score 1 8 1 = 30 := by
  sorry

end NUMINAMATH_CALUDE_jeremy_scrabble_score_l1022_102230


namespace NUMINAMATH_CALUDE_james_payment_is_correct_l1022_102237

/-- Calculates James's payment for stickers given the number of packs, stickers per pack,
    cost per sticker, discount rate, tax rate, and friend's contribution ratio. -/
def james_payment (packs : ℕ) (stickers_per_pack : ℕ) (cost_per_sticker : ℚ)
                  (discount_rate : ℚ) (tax_rate : ℚ) (friend_contribution_ratio : ℚ) : ℚ :=
  let total_cost := packs * stickers_per_pack * cost_per_sticker
  let discounted_cost := total_cost * (1 - discount_rate)
  let taxed_cost := discounted_cost * (1 + tax_rate)
  taxed_cost * (1 - friend_contribution_ratio)

/-- Proves that James's payment is $36.38 given the specific conditions of the problem. -/
theorem james_payment_is_correct :
  james_payment 8 40 (25 / 100) (15 / 100) (7 / 100) (1 / 2) = 3638 / 100 := by
  sorry

end NUMINAMATH_CALUDE_james_payment_is_correct_l1022_102237


namespace NUMINAMATH_CALUDE_complex_cube_equation_l1022_102281

theorem complex_cube_equation (a b : ℕ+) :
  (↑a + ↑b * Complex.I) ^ 3 = (2 : ℂ) + 11 * Complex.I →
  ↑a + ↑b * Complex.I = (2 : ℂ) + Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_cube_equation_l1022_102281


namespace NUMINAMATH_CALUDE_projectile_max_height_l1022_102282

/-- The height function of the projectile -/
def h (t : ℝ) : ℝ := -20 * t^2 + 120 * t + 36

/-- The time at which the maximum height occurs -/
def t_max : ℝ := 3

/-- The maximum height reached by the projectile -/
def h_max : ℝ := 216

theorem projectile_max_height :
  (∀ t, h t ≤ h_max) ∧ h t_max = h_max := by sorry

end NUMINAMATH_CALUDE_projectile_max_height_l1022_102282


namespace NUMINAMATH_CALUDE_journey_equation_correct_l1022_102219

/-- Represents a journey with a stop in between -/
structure Journey where
  preBrakeSpeed : ℝ
  postBrakeSpeed : ℝ
  totalDistance : ℝ
  totalTime : ℝ
  brakeTime : ℝ

/-- Checks if the given equation correctly represents the journey -/
def isCorrectEquation (j : Journey) (equation : ℝ → Prop) : Prop :=
  ∀ t, equation t ↔ 
    j.preBrakeSpeed * t + j.postBrakeSpeed * (j.totalTime - j.brakeTime - t) = j.totalDistance

theorem journey_equation_correct (j : Journey) 
    (h1 : j.preBrakeSpeed = 60)
    (h2 : j.postBrakeSpeed = 80)
    (h3 : j.totalDistance = 220)
    (h4 : j.totalTime = 4)
    (h5 : j.brakeTime = 2/3) :
    isCorrectEquation j (fun t ↦ 60 * t + 80 * (10/3 - t) = 220) := by
  sorry

#check journey_equation_correct

end NUMINAMATH_CALUDE_journey_equation_correct_l1022_102219


namespace NUMINAMATH_CALUDE_smallest_positive_integer_satisfying_congruences_l1022_102267

theorem smallest_positive_integer_satisfying_congruences : ∃! b : ℕ+, 
  (b : ℤ) % 3 = 2 ∧ 
  (b : ℤ) % 4 = 3 ∧ 
  (b : ℤ) % 5 = 4 ∧ 
  (b : ℤ) % 7 = 6 ∧ 
  ∀ c : ℕ+, 
    ((c : ℤ) % 3 = 2 ∧ 
     (c : ℤ) % 4 = 3 ∧ 
     (c : ℤ) % 5 = 4 ∧ 
     (c : ℤ) % 7 = 6) → 
    b ≤ c := by
  sorry

end NUMINAMATH_CALUDE_smallest_positive_integer_satisfying_congruences_l1022_102267
