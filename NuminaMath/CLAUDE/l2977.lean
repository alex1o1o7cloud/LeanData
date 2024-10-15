import Mathlib

namespace NUMINAMATH_CALUDE_horse_grain_consumption_l2977_297734

/-- Calculates the amount of grain each horse eats per day -/
theorem horse_grain_consumption
  (num_horses : ℕ)
  (oats_per_meal : ℕ)
  (oats_meals_per_day : ℕ)
  (total_days : ℕ)
  (total_food : ℕ)
  (h1 : num_horses = 4)
  (h2 : oats_per_meal = 4)
  (h3 : oats_meals_per_day = 2)
  (h4 : total_days = 3)
  (h5 : total_food = 132) :
  (total_food - num_horses * oats_per_meal * oats_meals_per_day * total_days) / (num_horses * total_days) = 3 := by
  sorry

end NUMINAMATH_CALUDE_horse_grain_consumption_l2977_297734


namespace NUMINAMATH_CALUDE_probability_red_then_white_l2977_297749

/-- The probability of drawing a red ball followed by a white ball in two successive draws with replacement -/
theorem probability_red_then_white (total : ℕ) (red : ℕ) (white : ℕ) 
  (h_total : total = 9)
  (h_red : red = 3)
  (h_white : white = 2) :
  (red : ℚ) / total * (white : ℚ) / total = 2 / 27 := by
  sorry

end NUMINAMATH_CALUDE_probability_red_then_white_l2977_297749


namespace NUMINAMATH_CALUDE_function_inequality_implies_range_l2977_297700

def decreasing_function (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x > f y

theorem function_inequality_implies_range (f : ℝ → ℝ) (a : ℝ) :
  decreasing_function f →
  (∀ x, x > 0 → f x ≠ 0) →
  f (2 * a^2 + a + 1) < f (3 * a^2 - 4 * a + 1) →
  (0 < a ∧ a < 1/3) ∨ (1 < a ∧ a < 5) :=
by sorry

end NUMINAMATH_CALUDE_function_inequality_implies_range_l2977_297700


namespace NUMINAMATH_CALUDE_min_fencing_length_l2977_297709

/-- Represents the dimensions of a rectangular garden -/
structure Garden where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangular garden -/
def Garden.area (g : Garden) : ℝ := g.length * g.width

/-- Calculates the perimeter of a rectangular garden, excluding one side (against the wall) -/
def Garden.fencingLength (g : Garden) : ℝ := g.length + 2 * g.width

/-- The minimum fencing length for a garden with area 50 m² is 20 meters -/
theorem min_fencing_length :
  ∀ g : Garden, g.area = 50 → g.fencingLength ≥ 20 ∧ 
  ∃ g' : Garden, g'.area = 50 ∧ g'.fencingLength = 20 := by
  sorry


end NUMINAMATH_CALUDE_min_fencing_length_l2977_297709


namespace NUMINAMATH_CALUDE_profit_share_ratio_l2977_297757

def total_profit : ℝ := 500
def share_difference : ℝ := 100

theorem profit_share_ratio :
  ∀ (x y : ℝ),
  x + y = total_profit →
  x - y = share_difference →
  x / total_profit = 3 / 5 :=
by sorry

end NUMINAMATH_CALUDE_profit_share_ratio_l2977_297757


namespace NUMINAMATH_CALUDE_fraction_subtraction_property_l2977_297723

theorem fraction_subtraction_property (a b c d : ℚ) (hb : b ≠ 0) (hd : d ≠ 0) :
  (a / b - c / d = (a - c) / (b + d)) ↔ (a / c = (b / d)^2) :=
sorry

end NUMINAMATH_CALUDE_fraction_subtraction_property_l2977_297723


namespace NUMINAMATH_CALUDE_john_total_distance_l2977_297773

-- Define the driving speed
def speed : ℝ := 45

-- Define the first driving duration
def duration1 : ℝ := 2

-- Define the second driving duration
def duration2 : ℝ := 3

-- Theorem to prove
theorem john_total_distance :
  speed * (duration1 + duration2) = 225 := by
  sorry

end NUMINAMATH_CALUDE_john_total_distance_l2977_297773


namespace NUMINAMATH_CALUDE_arrangements_count_l2977_297763

/-- Represents the number of acts in the show -/
def total_acts : ℕ := 6

/-- Represents the possible positions for Act A -/
def act_a_positions : Finset ℕ := {1, 2}

/-- Represents the possible positions for Act B -/
def act_b_positions : Finset ℕ := {2, 3, 4, 5}

/-- Represents the position of Act C -/
def act_c_position : ℕ := total_acts

/-- A function that calculates the number of arrangements -/
def count_arrangements : ℕ := sorry

/-- The theorem stating that the number of arrangements is 42 -/
theorem arrangements_count : count_arrangements = 42 := by sorry

end NUMINAMATH_CALUDE_arrangements_count_l2977_297763


namespace NUMINAMATH_CALUDE_range_of_a_l2977_297736

-- Define a decreasing function on (0, +∞)
variable (f : ℝ → ℝ)
variable (h_decreasing : ∀ x y, 0 < x → 0 < y → x < y → f y < f x)

-- Define the domain of f
variable (h_domain : ∀ x, 0 < x → f x ∈ Set.range f)

-- Define the variable a
variable (a : ℝ)

-- State the theorem
theorem range_of_a (h_ineq : f (2*a^2 + a + 1) < f (3*a^2 - 4*a + 1)) :
  (0 < a ∧ a < 1/3) ∨ (1 < a ∧ a < 5) := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l2977_297736


namespace NUMINAMATH_CALUDE_probability_all_yellow_apples_l2977_297769

def total_apples : ℕ := 8
def yellow_apples : ℕ := 3
def red_apples : ℕ := 5
def apples_chosen : ℕ := 3

theorem probability_all_yellow_apples :
  (Nat.choose yellow_apples apples_chosen) / (Nat.choose total_apples apples_chosen) = 1 / 56 :=
by sorry

end NUMINAMATH_CALUDE_probability_all_yellow_apples_l2977_297769


namespace NUMINAMATH_CALUDE_rahul_work_days_l2977_297714

/-- The number of days it takes Rajesh to complete the work -/
def rajesh_days : ℝ := 2

/-- The total payment for the work -/
def total_payment : ℝ := 355

/-- Rahul's share of the payment -/
def rahul_share : ℝ := 142

/-- The number of days it takes Rahul to complete the work -/
def rahul_days : ℝ := 3

theorem rahul_work_days :
  rajesh_days = 2 ∧
  total_payment = 355 ∧
  rahul_share = 142 →
  rahul_days = 3 := by
  sorry

end NUMINAMATH_CALUDE_rahul_work_days_l2977_297714


namespace NUMINAMATH_CALUDE_sarah_stamp_collection_value_l2977_297760

/-- Calculates the total value of a stamp collection given the following conditions:
    - The total number of stamps in the collection
    - The number of stamps in a subset
    - The total value of the subset
    Assuming the price per stamp is constant. -/
def stamp_collection_value (total_stamps : ℕ) (subset_stamps : ℕ) (subset_value : ℚ) : ℚ :=
  (total_stamps : ℚ) * (subset_value / subset_stamps)

/-- Theorem stating that a collection of 20 stamps, where 4 stamps are worth $10,
    has a total value of $50. -/
theorem sarah_stamp_collection_value :
  stamp_collection_value 20 4 10 = 50 := by
  sorry

end NUMINAMATH_CALUDE_sarah_stamp_collection_value_l2977_297760


namespace NUMINAMATH_CALUDE_lesser_fraction_l2977_297743

theorem lesser_fraction (x y : ℝ) (h_sum : x + y = 13/14) (h_prod : x * y = 1/8) :
  min x y = (13 - Real.sqrt 57) / 28 := by sorry

end NUMINAMATH_CALUDE_lesser_fraction_l2977_297743


namespace NUMINAMATH_CALUDE_parallel_planes_theorem_l2977_297737

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations and operations
variable (subset : Line → Plane → Prop)
variable (parallel : Line → Plane → Prop)
variable (intersect : Line → Line → Set Point)
variable (plane_parallel : Plane → Plane → Prop)

-- State the theorem
theorem parallel_planes_theorem 
  (l m : Line) (α β : Plane) (P : Point) :
  l ≠ m →
  α ≠ β →
  subset l α →
  subset m α →
  intersect l m = {P} →
  parallel l β →
  parallel m β →
  plane_parallel α β :=
sorry

end NUMINAMATH_CALUDE_parallel_planes_theorem_l2977_297737


namespace NUMINAMATH_CALUDE_ferris_wheel_capacity_l2977_297759

/-- The number of large seats on the Ferris wheel -/
def num_large_seats : ℕ := 7

/-- The weight limit for each large seat (in pounds) -/
def weight_limit_per_seat : ℕ := 1500

/-- The average weight of each person (in pounds) -/
def avg_weight_per_person : ℕ := 180

/-- The maximum number of people that can ride on large seats without violating the weight limit -/
def max_people_on_large_seats : ℕ := 
  (num_large_seats * (weight_limit_per_seat / avg_weight_per_person))

theorem ferris_wheel_capacity : max_people_on_large_seats = 56 := by
  sorry

end NUMINAMATH_CALUDE_ferris_wheel_capacity_l2977_297759


namespace NUMINAMATH_CALUDE_magnitude_of_z_plus_two_l2977_297704

/-- Given a complex number z = (1+i)/i, prove that the magnitude of z+2 is √10 -/
theorem magnitude_of_z_plus_two (z : ℂ) : z = (1 + Complex.I) / Complex.I → Complex.abs (z + 2) = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_z_plus_two_l2977_297704


namespace NUMINAMATH_CALUDE_minBrokenLine_l2977_297778

-- Define the necessary structures
structure Point where
  x : ℝ
  y : ℝ

structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the given conditions
def sameSide (A B : Point) (l : Line) : Prop := sorry

def reflectPoint (A : Point) (l : Line) : Point := sorry

def onLine (X : Point) (l : Line) : Prop := sorry

def intersectionPoint (l : Line) (A B : Point) : Point := sorry

def brokenLineLength (A X B : Point) : ℝ := sorry

-- State the theorem
theorem minBrokenLine (l : Line) (A B : Point) :
  sameSide A B l →
  ∃ X : Point, onLine X l ∧
    ∀ Y : Point, onLine Y l →
      brokenLineLength A X B ≤ brokenLineLength A Y B :=
  by
    sorry

end NUMINAMATH_CALUDE_minBrokenLine_l2977_297778


namespace NUMINAMATH_CALUDE_defective_pens_count_l2977_297799

def total_pens : ℕ := 10

def prob_non_defective : ℚ := 0.6222222222222222

theorem defective_pens_count (defective : ℕ) 
  (h1 : defective ≤ total_pens)
  (h2 : (((total_pens - defective) : ℚ) / total_pens) * 
        (((total_pens - defective - 1) : ℚ) / (total_pens - 1)) = prob_non_defective) :
  defective = 2 := by sorry

end NUMINAMATH_CALUDE_defective_pens_count_l2977_297799


namespace NUMINAMATH_CALUDE_diophantine_equation_solutions_l2977_297719

theorem diophantine_equation_solutions :
  ∀ m n : ℕ+, 7^(m : ℕ) - 3 * 2^(n : ℕ) = 1 ↔ (m = 1 ∧ n = 1) ∨ (m = 2 ∧ n = 4) := by
sorry

end NUMINAMATH_CALUDE_diophantine_equation_solutions_l2977_297719


namespace NUMINAMATH_CALUDE_unique_number_triple_and_square_l2977_297789

theorem unique_number_triple_and_square (x : ℝ) : 
  (x > 0 ∧ 3 * x = (x / 2)^2 + 45) ↔ x = 18 := by
  sorry

end NUMINAMATH_CALUDE_unique_number_triple_and_square_l2977_297789


namespace NUMINAMATH_CALUDE_box_width_calculation_l2977_297776

/-- Given a rectangular box with specified dimensions and features, calculate its width -/
theorem box_width_calculation (length : ℝ) (road_width : ℝ) (lawn_area : ℝ) : 
  length = 60 →
  road_width = 3 →
  lawn_area = 2109 →
  ∃ (width : ℝ), width = 37.15 ∧ length * width - 2 * (length / 3) * road_width = lawn_area :=
by sorry

end NUMINAMATH_CALUDE_box_width_calculation_l2977_297776


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l2977_297751

theorem quadratic_inequality_solution (a : ℝ) (x₁ x₂ : ℝ) : 
  a > 0 →
  (∀ x, x^2 - 2*a*x - 8*a^2 < 0 ↔ x₁ < x ∧ x < x₂) →
  x₂ - x₁ = 15 →
  a = 5/2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l2977_297751


namespace NUMINAMATH_CALUDE_quadratic_equation_root_l2977_297777

theorem quadratic_equation_root (b : ℝ) : 
  (2 : ℝ) ^ 2 * 2 + b * 2 - 4 = 0 → b = -2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_root_l2977_297777


namespace NUMINAMATH_CALUDE_square_sum_given_linear_equations_l2977_297720

theorem square_sum_given_linear_equations :
  ∀ x y : ℝ, 3 * x + 2 * y = 20 → 4 * x + 2 * y = 26 → x^2 + y^2 = 37 := by
sorry

end NUMINAMATH_CALUDE_square_sum_given_linear_equations_l2977_297720


namespace NUMINAMATH_CALUDE_log_difference_divided_l2977_297768

theorem log_difference_divided : (Real.log 1 - Real.log 25) / 100 = -20 := by sorry

end NUMINAMATH_CALUDE_log_difference_divided_l2977_297768


namespace NUMINAMATH_CALUDE_min_value_theorem_l2977_297713

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 2 * a + 3 * b = 1) :
  (2 / a + 3 / b) ≥ 26 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ 2 * a₀ + 3 * b₀ = 1 ∧ 2 / a₀ + 3 / b₀ = 26 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2977_297713


namespace NUMINAMATH_CALUDE_flower_shop_cost_l2977_297775

/-- The total cost of buying roses and lilies with given conditions -/
theorem flower_shop_cost : 
  let num_roses : ℕ := 20
  let num_lilies : ℕ := (3 * num_roses) / 4
  let cost_per_rose : ℕ := 5
  let cost_per_lily : ℕ := 2 * cost_per_rose
  let total_cost : ℕ := num_roses * cost_per_rose + num_lilies * cost_per_lily
  total_cost = 250 := by
  sorry

end NUMINAMATH_CALUDE_flower_shop_cost_l2977_297775


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_bound_l2977_297793

/-- The radius of a circle inscribed in a quadrilateral with sides 3, 6, 5, and 8 is less than 3 -/
theorem inscribed_circle_radius_bound (r : ℝ) : 
  r > 0 → -- r is positive (radius)
  r * 11 = 12 * Real.sqrt 5 → -- area formula: S = r * s, where s = (3 + 6 + 5 + 8) / 2 = 11
  r < 3 := by
sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_bound_l2977_297793


namespace NUMINAMATH_CALUDE_ingrid_income_proof_l2977_297772

/-- The annual income of John in dollars -/
def john_income : ℝ := 56000

/-- The tax rate for John as a decimal -/
def john_tax_rate : ℝ := 0.30

/-- The tax rate for Ingrid as a decimal -/
def ingrid_tax_rate : ℝ := 0.40

/-- The combined tax rate for John and Ingrid as a decimal -/
def combined_tax_rate : ℝ := 0.3569

/-- Ingrid's income in dollars -/
def ingrid_income : ℝ := 73924.13

/-- Theorem stating that given the conditions, Ingrid's income is correct -/
theorem ingrid_income_proof :
  (john_tax_rate * john_income + ingrid_tax_rate * ingrid_income) / (john_income + ingrid_income) = combined_tax_rate :=
by sorry

end NUMINAMATH_CALUDE_ingrid_income_proof_l2977_297772


namespace NUMINAMATH_CALUDE_triharmonic_properties_l2977_297761

-- Define a triharmonic quadruple
def is_triharmonic (A B C D : ℝ × ℝ) : Prop :=
  (dist A B) * (dist C D) = (dist A C) * (dist B D) ∧
  (dist A B) * (dist C D) = (dist A D) * (dist B C)

-- Define concyclicity
def are_concyclic (A B C D : ℝ × ℝ) : Prop :=
  ∃ (O : ℝ × ℝ) (r : ℝ), r > 0 ∧
    dist O A = r ∧ dist O B = r ∧ dist O C = r ∧ dist O D = r

theorem triharmonic_properties
  (A B C D A1 B1 C1 D1 : ℝ × ℝ)
  (h1 : is_triharmonic A B C D)
  (h2 : is_triharmonic A1 B C D)
  (h3 : is_triharmonic A B1 C D)
  (h4 : is_triharmonic A B C1 D)
  (h5 : is_triharmonic A B C D1)
  (hA : A1 ≠ A) (hB : B1 ≠ B) (hC : C1 ≠ C) (hD : D1 ≠ D) :
  are_concyclic A B C1 D1 ∧ is_triharmonic A1 B1 C1 D1 := by
  sorry

end NUMINAMATH_CALUDE_triharmonic_properties_l2977_297761


namespace NUMINAMATH_CALUDE_globe_division_count_l2977_297706

/-- The number of parts a globe's surface is divided into, given the number of parallels and meridians -/
def globe_divisions (parallels : ℕ) (meridians : ℕ) : ℕ :=
  meridians * (parallels + 1)

/-- Theorem: A globe with 17 parallels and 24 meridians is divided into 432 parts -/
theorem globe_division_count : globe_divisions 17 24 = 432 := by
  sorry

end NUMINAMATH_CALUDE_globe_division_count_l2977_297706


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l2977_297785

theorem geometric_sequence_common_ratio 
  (b₁ : ℕ+) 
  (q : ℕ+) 
  (seq : ℕ → ℕ+) 
  (h_geometric : ∀ n, seq n = b₁ * q ^ (n - 1)) 
  (h_sum : seq 3 + seq 5 + seq 7 = 819 * 6^2016) :
  q = 1 ∨ q = 2 ∨ q = 3 ∨ q = 4 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l2977_297785


namespace NUMINAMATH_CALUDE_dividend_calculation_l2977_297767

theorem dividend_calculation (x : ℕ) (h : x > 1) :
  let divisor := 3 * x^2
  let quotient := 5 * x
  let remainder := 7 * x + 9
  let dividend := divisor * quotient + remainder
  dividend = 15 * x^3 + 7 * x + 9 := by
sorry

end NUMINAMATH_CALUDE_dividend_calculation_l2977_297767


namespace NUMINAMATH_CALUDE_divisibility_concatenation_l2977_297774

theorem divisibility_concatenation (a b : ℕ) : 
  100 ≤ a ∧ a < 1000 ∧ 100 ≤ b ∧ b < 1000 →  -- a and b are three-digit numbers
  ¬(37 ∣ a) →  -- a is not divisible by 37
  ¬(37 ∣ b) →  -- b is not divisible by 37
  (37 ∣ (a + b)) →  -- a + b is divisible by 37
  (37 ∣ (1000 * a + b))  -- 1000a + b is divisible by 37
  := by sorry

end NUMINAMATH_CALUDE_divisibility_concatenation_l2977_297774


namespace NUMINAMATH_CALUDE_binomial_coefficient_problem_l2977_297728

theorem binomial_coefficient_problem (m : ℤ) : 
  (Nat.choose 4 2 : ℤ) * m^2 = (Nat.choose 4 3 : ℤ) * m + 16 → m = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_problem_l2977_297728


namespace NUMINAMATH_CALUDE_system_solution_l2977_297748

theorem system_solution (x y z : ℝ) : 
  x + y + z = 3 ∧ 
  x^2 + y^2 + z^2 = 7 ∧ 
  x^3 + y^3 + z^3 = 15 ↔ 
  (x = 1 ∧ y = 1 + Real.sqrt 2 ∧ z = 1 - Real.sqrt 2) ∨
  (x = 1 ∧ y = 1 - Real.sqrt 2 ∧ z = 1 + Real.sqrt 2) ∨
  (x = 1 + Real.sqrt 2 ∧ y = 1 ∧ z = 1 - Real.sqrt 2) ∨
  (x = 1 + Real.sqrt 2 ∧ y = 1 - Real.sqrt 2 ∧ z = 1) ∨
  (x = 1 - Real.sqrt 2 ∧ y = 1 ∧ z = 1 + Real.sqrt 2) ∨
  (x = 1 - Real.sqrt 2 ∧ y = 1 + Real.sqrt 2 ∧ z = 1) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l2977_297748


namespace NUMINAMATH_CALUDE_cubic_roots_sum_l2977_297783

theorem cubic_roots_sum (a b c : ℝ) : 
  (∀ x : ℝ, x^3 - 2*x - 2 = 0 ↔ x = a ∨ x = b ∨ x = c) →
  a*(b-c)^2 + b*(c-a)^2 + c*(a-b)^2 = -2 := by
sorry

end NUMINAMATH_CALUDE_cubic_roots_sum_l2977_297783


namespace NUMINAMATH_CALUDE_waiter_tips_l2977_297787

/-- Calculates the total tips earned by a waiter --/
def calculate_tips (total_customers : ℕ) (non_tipping_customers : ℕ) (tip_amount : ℕ) : ℕ :=
  (total_customers - non_tipping_customers) * tip_amount

/-- Proves that the waiter earned $27 in tips --/
theorem waiter_tips : calculate_tips 7 4 9 = 27 := by
  sorry

end NUMINAMATH_CALUDE_waiter_tips_l2977_297787


namespace NUMINAMATH_CALUDE_altitude_equals_harmonic_mean_of_excircle_radii_l2977_297752

/-- For a triangle ABC with altitude h_a from vertex A, area t, semiperimeter s,
    and excircle radii r_b and r_c, the altitude h_a is equal to 2t/a. -/
theorem altitude_equals_harmonic_mean_of_excircle_radii 
  (a b c : ℝ) 
  (h_a : ℝ) 
  (t : ℝ) 
  (s : ℝ) 
  (r_b r_c : ℝ) 
  (h_s : s = (a + b + c) / 2) 
  (h_r_b : r_b = t / (s - b)) 
  (h_r_c : r_c = t / (s - c)) 
  (h_positive : a > 0 ∧ t > 0) : 
  h_a = 2 * t / a := by
  sorry

end NUMINAMATH_CALUDE_altitude_equals_harmonic_mean_of_excircle_radii_l2977_297752


namespace NUMINAMATH_CALUDE_buffet_dressing_cases_l2977_297742

/-- Represents the number of cases for each type of dressing -/
structure DressingCases where
  ranch : ℕ
  caesar : ℕ
  italian : ℕ
  thousandIsland : ℕ

/-- Checks if the ratios between dressing cases are correct -/
def correctRatios (cases : DressingCases) : Prop :=
  7 * cases.caesar = 2 * cases.ranch ∧
  cases.caesar * 3 = cases.italian ∧
  3 * cases.thousandIsland = 2 * cases.italian

/-- The theorem to be proved -/
theorem buffet_dressing_cases : 
  ∃ (cases : DressingCases), 
    cases.ranch = 28 ∧
    cases.caesar = 8 ∧
    cases.italian = 24 ∧
    cases.thousandIsland = 16 ∧
    correctRatios cases :=
by sorry

end NUMINAMATH_CALUDE_buffet_dressing_cases_l2977_297742


namespace NUMINAMATH_CALUDE_equation_represents_parallel_lines_l2977_297721

theorem equation_represents_parallel_lines :
  ∃ (m c₁ c₂ : ℝ), m ≠ 0 ∧ c₁ ≠ c₂ ∧
  ∀ (x y : ℝ), x^2 - 9*y^2 + 3*x = 0 ↔ (y = m*x + c₁ ∨ y = m*x + c₂) :=
sorry

end NUMINAMATH_CALUDE_equation_represents_parallel_lines_l2977_297721


namespace NUMINAMATH_CALUDE_wall_height_calculation_l2977_297746

/-- Calculates the height of a wall given its dimensions and the number and size of bricks used. -/
theorem wall_height_calculation (wall_length : Real) (wall_thickness : Real) 
  (brick_count : Nat) (brick_length : Real) (brick_width : Real) (brick_height : Real) : 
  wall_length = 900 ∧ wall_thickness = 22.5 ∧ brick_count = 7200 ∧ 
  brick_length = 25 ∧ brick_width = 11.25 ∧ brick_height = 6 → 
  (wall_length * wall_thickness * (brick_count * brick_length * brick_width * brick_height) / 
  (wall_length * wall_thickness)) = 600 := by
  sorry

end NUMINAMATH_CALUDE_wall_height_calculation_l2977_297746


namespace NUMINAMATH_CALUDE_escalator_speed_l2977_297744

theorem escalator_speed (escalator_speed : ℝ) (escalator_length : ℝ) (time_taken : ℝ) 
  (h1 : escalator_speed = 11)
  (h2 : escalator_length = 126)
  (h3 : time_taken = 9) :
  escalator_speed + (escalator_length - escalator_speed * time_taken) / time_taken = 14 :=
by sorry

end NUMINAMATH_CALUDE_escalator_speed_l2977_297744


namespace NUMINAMATH_CALUDE_area_fraction_above_line_l2977_297732

/-- A square with side length 3 -/
def square_side : ℝ := 3

/-- The first point of the line -/
def point1 : ℝ × ℝ := (3, 2)

/-- The second point of the line -/
def point2 : ℝ × ℝ := (6, 0)

/-- The theorem stating that the fraction of the square's area above the line is 2/3 -/
theorem area_fraction_above_line : 
  let square_area := square_side ^ 2
  let triangle_base := point2.1 - point1.1
  let triangle_height := point1.2
  let triangle_area := (1 / 2) * triangle_base * triangle_height
  let area_above_line := square_area - triangle_area
  (area_above_line / square_area) = (2 : ℝ) / 3 := by sorry

end NUMINAMATH_CALUDE_area_fraction_above_line_l2977_297732


namespace NUMINAMATH_CALUDE_mod_equivalence_l2977_297740

theorem mod_equivalence (m : ℕ) : 
  198 * 864 ≡ m [ZMOD 50] → 0 ≤ m → m < 50 → m = 22 := by
  sorry

end NUMINAMATH_CALUDE_mod_equivalence_l2977_297740


namespace NUMINAMATH_CALUDE_highway_speed_is_30_l2977_297796

-- Define the problem parameters
def initial_reading : ℕ := 12321
def next_palindrome : ℕ := 12421
def total_time : ℕ := 4
def highway_time : ℕ := 2
def urban_time : ℕ := 2
def speed_difference : ℕ := 10
def total_distance : ℕ := 100

-- Define the theorem
theorem highway_speed_is_30 :
  let urban_speed := (total_distance - speed_difference * highway_time) / total_time
  urban_speed + speed_difference = 30 := by
  sorry


end NUMINAMATH_CALUDE_highway_speed_is_30_l2977_297796


namespace NUMINAMATH_CALUDE_problem_solution_l2977_297727

theorem problem_solution (t : ℝ) (x y : ℝ) 
  (h1 : x = 3 - 2*t) 
  (h2 : y = 3*t + 10) 
  (h3 : x = 1) : 
  y = 13 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2977_297727


namespace NUMINAMATH_CALUDE_max_value_quadratic_l2977_297739

theorem max_value_quadratic :
  ∃ (M : ℝ), M = 26 ∧ ∀ (x : ℝ), -3 * x^2 + 18 * x - 1 ≤ M :=
sorry

end NUMINAMATH_CALUDE_max_value_quadratic_l2977_297739


namespace NUMINAMATH_CALUDE_fifth_root_of_unity_sum_l2977_297722

theorem fifth_root_of_unity_sum (ω : ℂ) (h1 : ω^5 = 1) (h2 : ω ≠ 1) :
  ω^12 + ω^15 + ω^18 + ω^21 + ω^24 = 0 := by
  sorry

end NUMINAMATH_CALUDE_fifth_root_of_unity_sum_l2977_297722


namespace NUMINAMATH_CALUDE_mechanic_average_earning_l2977_297705

/-- The average earning of a mechanic for a week, given specific conditions --/
theorem mechanic_average_earning (first_four_avg : ℚ) (last_four_avg : ℚ) (fourth_day : ℚ) :
  first_four_avg = 18 →
  last_four_avg = 22 →
  fourth_day = 13 →
  (4 * first_four_avg + 4 * last_four_avg - fourth_day) / 7 = 160 / 7 := by
  sorry

#eval (160 : ℚ) / 7

end NUMINAMATH_CALUDE_mechanic_average_earning_l2977_297705


namespace NUMINAMATH_CALUDE_min_sum_positive_integers_l2977_297756

theorem min_sum_positive_integers (x y z w : ℕ+) 
  (h : (2 : ℕ) * x ^ 2 = (5 : ℕ) * y ^ 3 ∧ 
       (5 : ℕ) * y ^ 3 = (8 : ℕ) * z ^ 4 ∧ 
       (8 : ℕ) * z ^ 4 = (3 : ℕ) * w) : 
  x + y + z + w ≥ 54 := by
sorry

end NUMINAMATH_CALUDE_min_sum_positive_integers_l2977_297756


namespace NUMINAMATH_CALUDE_binomial_sixteen_nine_l2977_297770

theorem binomial_sixteen_nine (h1 : Nat.choose 15 7 = 6435)
                              (h2 : Nat.choose 15 8 = 6435)
                              (h3 : Nat.choose 17 9 = 24310) :
  Nat.choose 16 9 = 11440 := by
  sorry

end NUMINAMATH_CALUDE_binomial_sixteen_nine_l2977_297770


namespace NUMINAMATH_CALUDE_area_change_not_triple_l2977_297780

theorem area_change_not_triple :
  ∀ (s r : ℝ), s > 0 → r > 0 →
  (3 * s)^2 ≠ 3 * s^2 ∧ π * (3 * r)^2 ≠ 3 * (π * r^2) :=
by sorry

end NUMINAMATH_CALUDE_area_change_not_triple_l2977_297780


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l2977_297730

theorem polynomial_division_remainder :
  ∃ q : Polynomial ℝ, (X^4 - 3•X + 1 : Polynomial ℝ) = (X^2 - X - 1) * q + 3 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l2977_297730


namespace NUMINAMATH_CALUDE_equation_solution_l2977_297731

theorem equation_solution : ∃ x : ℝ, (3 / 4 - 1 / x = 1 / 2) ∧ (x = 4) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2977_297731


namespace NUMINAMATH_CALUDE_jason_work_experience_l2977_297750

/-- Calculates the total work experience in months given years as bartender and years and months as manager -/
def total_work_experience (bartender_years : ℕ) (manager_years : ℕ) (manager_months : ℕ) : ℕ := 
  bartender_years * 12 + manager_years * 12 + manager_months

/-- Proves that Jason's total work experience is 150 months -/
theorem jason_work_experience : 
  total_work_experience 9 3 6 = 150 := by
  sorry

end NUMINAMATH_CALUDE_jason_work_experience_l2977_297750


namespace NUMINAMATH_CALUDE_ellipse_chord_length_l2977_297703

/-- Represents an ellipse with foci on the x-axis -/
structure Ellipse where
  a : ℝ  -- Semi-major axis
  b : ℝ  -- Semi-minor axis
  e : ℝ  -- Eccentricity

/-- Represents a line in the form y = mx + c -/
structure Line where
  m : ℝ  -- Slope
  c : ℝ  -- y-intercept

theorem ellipse_chord_length (C : Ellipse) (L : Line) :
  C.b = 1 ∧ C.e = Real.sqrt 3 / 2 ∧ L.m = 1 ∧ L.c = 1 →
  (∀ x y : ℝ, x^2 / 4 + y^2 = 1 ↔ x^2 / C.a^2 + y^2 / C.b^2 = 1) ∧
  Real.sqrt ((8/5)^2 + (8/5)^2) = 8 * Real.sqrt 2 / 5 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_chord_length_l2977_297703


namespace NUMINAMATH_CALUDE_quadratic_roots_condition_l2977_297795

theorem quadratic_roots_condition (c : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 + 2*x₁ + 4*c = 0 ∧ x₂^2 + 2*x₂ + 4*c = 0) ↔ c < (1/4 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_condition_l2977_297795


namespace NUMINAMATH_CALUDE_milk_for_cookies_l2977_297735

/-- Given the ratio of milk to cookies and the conversion between quarts and pints,
    calculate the amount of milk needed for a different number of cookies. -/
theorem milk_for_cookies (cookies_base : ℕ) (quarts_base : ℕ) (cookies_target : ℕ) :
  cookies_base > 0 →
  quarts_base > 0 →
  cookies_target > 0 →
  (cookies_base = 18 ∧ quarts_base = 3 ∧ cookies_target = 15) →
  (∃ (pints_target : ℚ), pints_target = 5 ∧
    pints_target = (quarts_base * 2 : ℚ) * cookies_target / cookies_base) :=
by
  sorry

#check milk_for_cookies

end NUMINAMATH_CALUDE_milk_for_cookies_l2977_297735


namespace NUMINAMATH_CALUDE_find_C_l2977_297729

theorem find_C (A B C : ℝ) 
  (h_diff1 : A ≠ B) (h_diff2 : A ≠ C) (h_diff3 : B ≠ C)
  (h1 : 3 * A - A = 10)
  (h2 : B + A = 12)
  (h3 : C - B = 6) : 
  C = 13 := by
sorry

end NUMINAMATH_CALUDE_find_C_l2977_297729


namespace NUMINAMATH_CALUDE_conditional_extremum_l2977_297786

/-- The objective function to be optimized -/
def f (x₁ x₂ : ℝ) : ℝ := x₁^2 + x₂^2 - x₁*x₂ + x₁ + x₂ - 6

/-- The constraint function -/
def g (x₁ x₂ : ℝ) : ℝ := x₁ + x₂ + 3

/-- Theorem stating the conditional extremum of f subject to g -/
theorem conditional_extremum :
  ∃ (x₁ x₂ : ℝ), g x₁ x₂ = 0 ∧ 
    (∀ (y₁ y₂ : ℝ), g y₁ y₂ = 0 → f x₁ x₂ ≤ f y₁ y₂) ∧
    x₁ = -3/2 ∧ x₂ = -3/2 ∧ f x₁ x₂ = -9/2 :=
sorry

end NUMINAMATH_CALUDE_conditional_extremum_l2977_297786


namespace NUMINAMATH_CALUDE_triangle_area_triangle_area_proof_l2977_297718

/-- The area of a triangle with side lengths 9, 40, and 41 is 180 -/
theorem triangle_area : ℝ → Prop :=
  fun area =>
    let a : ℝ := 9
    let b : ℝ := 40
    let c : ℝ := 41
    (a^2 + b^2 = c^2) ∧ (area = (1/2) * a * b) ∧ (area = 180)

/-- Proof of the theorem -/
theorem triangle_area_proof : ∃ (area : ℝ), triangle_area area := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_triangle_area_proof_l2977_297718


namespace NUMINAMATH_CALUDE_number117_is_1983_l2977_297797

/-- The set of digits used to form the four-digit numbers -/
def digits : Finset Nat := {1, 3, 4, 5, 7, 8, 9}

/-- A four-digit number formed from the given digits without repetition -/
structure FourDigitNumber where
  d1 : Nat
  d2 : Nat
  d3 : Nat
  d4 : Nat
  h1 : d1 ∈ digits
  h2 : d2 ∈ digits
  h3 : d3 ∈ digits
  h4 : d4 ∈ digits
  h5 : d1 ≠ d2 ∧ d1 ≠ d3 ∧ d1 ≠ d4 ∧ d2 ≠ d3 ∧ d2 ≠ d4 ∧ d3 ≠ d4

/-- The value of a four-digit number -/
def FourDigitNumber.value (n : FourDigitNumber) : Nat :=
  1000 * n.d1 + 100 * n.d2 + 10 * n.d3 + n.d4

/-- The set of all valid four-digit numbers -/
def validNumbers : Finset FourDigitNumber := sorry

/-- The 117th number in the ascending sequence of valid four-digit numbers -/
def number117 : FourDigitNumber := sorry

theorem number117_is_1983 : number117.value = 1983 := by sorry

end NUMINAMATH_CALUDE_number117_is_1983_l2977_297797


namespace NUMINAMATH_CALUDE_star_two_three_star_two_neg_six_neg_two_thirds_l2977_297782

-- Define the operation *
def star (a b : ℚ) : ℚ := (a + b) / 3

-- Theorem for 2 * 3 = 5/3
theorem star_two_three : star 2 3 = 5/3 := by sorry

-- Theorem for 2 * (-6) * (-2/3) = -2/3
theorem star_two_neg_six_neg_two_thirds : star (star 2 (-6)) (-2/3) = -2/3 := by sorry

end NUMINAMATH_CALUDE_star_two_three_star_two_neg_six_neg_two_thirds_l2977_297782


namespace NUMINAMATH_CALUDE_inscribed_octagon_area_l2977_297725

/-- An inscribed convex octagon with alternating side lengths of 2 and 6√2 -/
structure InscribedOctagon where
  -- The octagon is inscribed in a circle (implied by the problem)
  isInscribed : Bool
  -- The octagon is convex
  isConvex : Bool
  -- The octagon has 8 sides
  numSides : Nat
  -- Four sides have length 2
  shortSideLength : ℝ
  -- Four sides have length 6√2
  longSideLength : ℝ
  -- Conditions
  inscribed_condition : isInscribed = true
  convex_condition : isConvex = true
  sides_condition : numSides = 8
  short_side_condition : shortSideLength = 2
  long_side_condition : longSideLength = 6 * Real.sqrt 2

/-- The area of the inscribed convex octagon -/
def area (o : InscribedOctagon) : ℝ := sorry

/-- Theorem stating that the area of the inscribed convex octagon is 124 -/
theorem inscribed_octagon_area (o : InscribedOctagon) : area o = 124 := by sorry

end NUMINAMATH_CALUDE_inscribed_octagon_area_l2977_297725


namespace NUMINAMATH_CALUDE_perpendicular_vectors_l2977_297702

/-- Given vectors a, b, and c in ℝ², prove that if a + 2b is perpendicular to c, then k = -3 -/
theorem perpendicular_vectors (a b c : ℝ × ℝ) (k : ℝ) : 
  a = (Real.sqrt 3, 1) → 
  b = (0, 1) → 
  c = (k, Real.sqrt 3) → 
  (a.1 + 2 * b.1, a.2 + 2 * b.2) • c = 0 → 
  k = -3 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_l2977_297702


namespace NUMINAMATH_CALUDE_woojoo_initial_score_l2977_297788

theorem woojoo_initial_score 
  (num_students : ℕ) 
  (initial_avg : ℚ) 
  (new_score : ℕ) 
  (new_avg : ℚ) 
  (h1 : num_students = 10)
  (h2 : initial_avg = 42)
  (h3 : new_score = 50)
  (h4 : new_avg = 44) :
  ∃ (initial_score : ℕ), 
    (initial_score : ℚ) + (num_students - 1 : ℚ) * initial_avg = num_students * initial_avg ∧
    (new_score : ℚ) + (num_students - 1 : ℚ) * initial_avg = num_students * new_avg ∧
    initial_score = 30 := by
  sorry

end NUMINAMATH_CALUDE_woojoo_initial_score_l2977_297788


namespace NUMINAMATH_CALUDE_certain_number_is_88_l2977_297710

theorem certain_number_is_88 (x : ℝ) (y : ℝ) : 
  x = y + 0.25 * y → x = 110 → y = 88 := by
sorry

end NUMINAMATH_CALUDE_certain_number_is_88_l2977_297710


namespace NUMINAMATH_CALUDE_fraction_simplification_l2977_297765

theorem fraction_simplification :
  (18 : ℚ) / 22 * 52 / 24 * 33 / 39 * 22 / 52 = 33 / 52 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2977_297765


namespace NUMINAMATH_CALUDE_triangle_circle_tangent_l2977_297784

theorem triangle_circle_tangent (a b c : ℝ) (x : ℝ) :
  -- Triangle ABC is a right triangle
  a^2 = b^2 + c^2 →
  -- Perimeter of triangle ABC is 190
  a + b + c = 190 →
  -- Circle with radius 23 centered at O on AB is tangent to BC
  (b - x) / b = 23 / a →
  -- AO = x (where O is the center of the circle)
  x^2 + (b - x)^2 = c^2 →
  -- The length of AO is 67
  x = 67 := by
    sorry

#eval 67 + 1  -- x + y = 68

end NUMINAMATH_CALUDE_triangle_circle_tangent_l2977_297784


namespace NUMINAMATH_CALUDE_no_quadratic_trinomial_sequence_with_all_integral_roots_l2977_297733

/-- A sequence of quadratic trinomials -/
def QuadraticTrinomialSequence := ℕ → (ℝ → ℝ)

/-- Condition: P_n is the sum of the two preceding trinomials for n ≥ 3 -/
def IsSumOfPrecedingTrinomials (P : QuadraticTrinomialSequence) : Prop :=
  ∀ n : ℕ, n ≥ 3 → P n = P (n - 1) + P (n - 2)

/-- Condition: P_1 and P_2 do not have common roots -/
def NoCommonRoots (P : QuadraticTrinomialSequence) : Prop :=
  ∀ x : ℝ, P 1 x = 0 → P 2 x ≠ 0

/-- Condition: P_n has at least one integral root for all n -/
def HasIntegralRoot (P : QuadraticTrinomialSequence) : Prop :=
  ∀ n : ℕ, ∃ k : ℤ, P n k = 0

/-- Theorem: There does not exist a sequence of quadratic trinomials satisfying all conditions -/
theorem no_quadratic_trinomial_sequence_with_all_integral_roots :
  ¬ ∃ P : QuadraticTrinomialSequence,
    IsSumOfPrecedingTrinomials P ∧ NoCommonRoots P ∧ HasIntegralRoot P :=
by
  sorry

end NUMINAMATH_CALUDE_no_quadratic_trinomial_sequence_with_all_integral_roots_l2977_297733


namespace NUMINAMATH_CALUDE_white_balls_count_l2977_297764

theorem white_balls_count (x : ℕ) : 
  (3 : ℚ) / (3 + x) = 1/5 → x = 12 := by
  sorry

end NUMINAMATH_CALUDE_white_balls_count_l2977_297764


namespace NUMINAMATH_CALUDE_area_ratio_triangle_to_hexagon_l2977_297791

/-- A regular hexagon ABCDEF with vertices A, B, C, D, E, F -/
structure RegularHexagon where
  vertices : Fin 6 → ℝ × ℝ
  is_regular : sorry

/-- The area of a regular hexagon -/
def area_hexagon (h : RegularHexagon) : ℝ := sorry

/-- The area of triangle ACE in a regular hexagon -/
def area_triangle_ACE (h : RegularHexagon) : ℝ := sorry

/-- Theorem: The area of triangle ACE is 2/3 of the area of the regular hexagon -/
theorem area_ratio_triangle_to_hexagon (h : RegularHexagon) :
  area_triangle_ACE h / area_hexagon h = 2 / 3 := by sorry

end NUMINAMATH_CALUDE_area_ratio_triangle_to_hexagon_l2977_297791


namespace NUMINAMATH_CALUDE_arithmetic_sequence_scalar_multiple_l2977_297701

theorem arithmetic_sequence_scalar_multiple
  (a : ℕ → ℝ) (d c : ℝ) (h_arith : ∀ n, a (n + 1) - a n = d) (h_c : c ≠ 0) :
  ∃ (b : ℕ → ℝ), (∀ n, b n = c * a n) ∧ (∀ n, b (n + 1) - b n = c * d) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_scalar_multiple_l2977_297701


namespace NUMINAMATH_CALUDE_circle_line_intersection_l2977_297758

/-- The circle C₁ -/
def C₁ (x y m : ℝ) : Prop := x^2 + y^2 - 2*x - 4*y + m = 0

/-- The line l -/
def l (x y : ℝ) : Prop := x + 2*y - 4 = 0

/-- M is a point on both C₁ and l -/
def M (m : ℝ) : ℝ × ℝ := sorry

/-- N is a point on both C₁ and l, distinct from M -/
def N (m : ℝ) : ℝ × ℝ := sorry

/-- OM is perpendicular to ON -/
def perpendicular (M N : ℝ × ℝ) : Prop :=
  M.1 * N.1 + M.2 * N.2 = 0

theorem circle_line_intersection (m : ℝ) :
  C₁ (M m).1 (M m).2 m ∧
  C₁ (N m).1 (N m).2 m ∧
  l (M m).1 (M m).2 ∧
  l (N m).1 (N m).2 ∧
  perpendicular (M m) (N m) →
  m = 8/5 :=
sorry

end NUMINAMATH_CALUDE_circle_line_intersection_l2977_297758


namespace NUMINAMATH_CALUDE_jons_number_l2977_297794

theorem jons_number : ∃ (x : ℝ), 5 * (3 * x + 6) - 8 = 142 ∧ x = 8 := by sorry

end NUMINAMATH_CALUDE_jons_number_l2977_297794


namespace NUMINAMATH_CALUDE_smallest_with_twenty_divisors_l2977_297715

/-- The number of positive divisors of a positive integer -/
def num_divisors (n : ℕ+) : ℕ := sorry

/-- Checks if a positive integer has exactly 20 positive divisors -/
def has_twenty_divisors (n : ℕ+) : Prop := num_divisors n = 20

theorem smallest_with_twenty_divisors : 
  (∀ m : ℕ+, m < 432 → ¬(has_twenty_divisors m)) ∧ has_twenty_divisors 432 := by sorry

end NUMINAMATH_CALUDE_smallest_with_twenty_divisors_l2977_297715


namespace NUMINAMATH_CALUDE_triangle_circumscribed_circle_diameter_l2977_297738

/-- Given a triangle with one side of 12 inches and the opposite angle of 30°,
    the diameter of the circumscribed circle is 24 inches. -/
theorem triangle_circumscribed_circle_diameter
  (side : ℝ) (angle : ℝ) :
  side = 12 →
  angle = 30 * π / 180 →
  side / Real.sin angle = 24 :=
by sorry

end NUMINAMATH_CALUDE_triangle_circumscribed_circle_diameter_l2977_297738


namespace NUMINAMATH_CALUDE_eleven_team_league_games_l2977_297779

/-- The number of games played in a league where each team plays every other team once -/
def games_played (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: In a league with 11 teams, where each team plays every other team exactly once, 
    the total number of games played is 55. -/
theorem eleven_team_league_games : games_played 11 = 55 := by
  sorry

end NUMINAMATH_CALUDE_eleven_team_league_games_l2977_297779


namespace NUMINAMATH_CALUDE_solution_set_when_a_is_3_range_of_a_when_sum_geq_5_l2977_297762

-- Define the functions f and g
def f (a x : ℝ) : ℝ := |2*x - a| + a
def g (x : ℝ) : ℝ := |2*x - 3|

-- Part 1
theorem solution_set_when_a_is_3 :
  {x : ℝ | f 3 x ≤ 6} = {x : ℝ | 0 ≤ x ∧ x ≤ 3} :=
by sorry

-- Part 2
theorem range_of_a_when_sum_geq_5 :
  ∀ x : ℝ, f a x + g x ≥ 5 → a ≥ 4 :=
by sorry

end NUMINAMATH_CALUDE_solution_set_when_a_is_3_range_of_a_when_sum_geq_5_l2977_297762


namespace NUMINAMATH_CALUDE_fraction_value_l2977_297781

theorem fraction_value (a b c d : ℝ) 
  (h1 : a = 4 * b) 
  (h2 : b = 3 * c) 
  (h3 : c = 5 * d) : 
  a * c / (b * d) = 20 := by
sorry

end NUMINAMATH_CALUDE_fraction_value_l2977_297781


namespace NUMINAMATH_CALUDE_part_1_part_2_l2977_297741

/-- Definition of set A -/
def A (a : ℝ) : Set ℝ := {a - 3, 2 * a - 1, a^2 + 1}

/-- Definition of set B -/
def B (x : ℝ) : Set ℝ := {0, 1, x}

/-- Theorem for part 1 -/
theorem part_1 (a : ℝ) : -3 ∈ A a → a = 0 ∨ a = -1 := by sorry

/-- Theorem for part 2 -/
theorem part_2 (x : ℝ) : x^2 ∈ B x → x = -1 := by sorry

end NUMINAMATH_CALUDE_part_1_part_2_l2977_297741


namespace NUMINAMATH_CALUDE_henry_margo_meeting_l2977_297766

/-- The time it takes Henry to complete one lap around the track -/
def henry_lap_time : ℕ := 7

/-- The time it takes Margo to complete one lap around the track -/
def margo_lap_time : ℕ := 12

/-- The time when Henry and Margo meet at the starting line -/
def meeting_time : ℕ := 84

theorem henry_margo_meeting :
  Nat.lcm henry_lap_time margo_lap_time = meeting_time := by
  sorry

end NUMINAMATH_CALUDE_henry_margo_meeting_l2977_297766


namespace NUMINAMATH_CALUDE_theater_line_arrangements_l2977_297790

def number_of_people : ℕ := 8
def number_of_fixed_group : ℕ := 3

theorem theater_line_arrangements :
  (number_of_people - number_of_fixed_group + 1).factorial = 720 := by
  sorry

end NUMINAMATH_CALUDE_theater_line_arrangements_l2977_297790


namespace NUMINAMATH_CALUDE_sampling_method_l2977_297771

/-- Represents a bag of milk powder with a three-digit number -/
def BagNumber := Fin 800

/-- The random number table row -/
def RandomRow : List ℕ := [63, 01, 63, 78, 59, 16, 95, 55, 67, 19, 98, 10, 50, 71, 75, 12, 86, 73, 58, 07, 44, 39, 52, 38, 79]

/-- Selects valid numbers from the random row -/
def selectValidNumbers (row : List ℕ) : List BagNumber := sorry

/-- The sampling method -/
theorem sampling_method (randomRow : List ℕ) :
  randomRow = RandomRow →
  (selectValidNumbers randomRow).take 5 = [⟨785, sorry⟩, ⟨567, sorry⟩, ⟨199, sorry⟩, ⟨507, sorry⟩, ⟨175, sorry⟩] := by
  sorry

end NUMINAMATH_CALUDE_sampling_method_l2977_297771


namespace NUMINAMATH_CALUDE_boat_distance_is_105_l2977_297707

/-- Given a boat traveling downstream, calculate the distance covered. -/
def distance_downstream (boat_speed : ℝ) (stream_speed : ℝ) (time : ℝ) : ℝ :=
  (boat_speed + stream_speed) * time

/-- Theorem: The distance covered by the boat downstream is 105 km. -/
theorem boat_distance_is_105 :
  let boat_speed : ℝ := 16
  let stream_speed : ℝ := 5
  let time : ℝ := 5
  distance_downstream boat_speed stream_speed time = 105 := by
  sorry

end NUMINAMATH_CALUDE_boat_distance_is_105_l2977_297707


namespace NUMINAMATH_CALUDE_bus_ticket_savings_l2977_297745

/-- The cost of a single bus ticket in dollars -/
def single_ticket_cost : ℚ := 1.50

/-- The cost of a package of 5 bus tickets in dollars -/
def package_cost : ℚ := 5.75

/-- The number of tickets required -/
def required_tickets : ℕ := 40

/-- The number of tickets in a package -/
def tickets_per_package : ℕ := 5

/-- Theorem stating the savings when buying packages instead of single tickets -/
theorem bus_ticket_savings :
  single_ticket_cost * required_tickets -
  package_cost * (required_tickets / tickets_per_package) = 14 := by
  sorry

end NUMINAMATH_CALUDE_bus_ticket_savings_l2977_297745


namespace NUMINAMATH_CALUDE_wall_bricks_count_l2977_297717

/-- Represents the number of bricks in the wall -/
def total_bricks : ℕ := 288

/-- Time taken by the first bricklayer to build the wall alone -/
def time1 : ℕ := 8

/-- Time taken by the second bricklayer to build the wall alone -/
def time2 : ℕ := 12

/-- Efficiency loss when working together (in bricks per hour) -/
def efficiency_loss : ℕ := 12

/-- Time taken by both bricklayers working together -/
def time_together : ℕ := 6

theorem wall_bricks_count :
  (time_together : ℚ) * ((total_bricks / time1 : ℚ) + (total_bricks / time2 : ℚ) - efficiency_loss) = total_bricks := by
  sorry

#check wall_bricks_count

end NUMINAMATH_CALUDE_wall_bricks_count_l2977_297717


namespace NUMINAMATH_CALUDE_banana_count_l2977_297711

/-- The number of bananas Melissa had initially -/
def initial_bananas : ℕ := 88

/-- The number of bananas Melissa shared -/
def shared_bananas : ℕ := 4

/-- The number of bananas Melissa had left after sharing -/
def remaining_bananas : ℕ := 84

theorem banana_count : initial_bananas = shared_bananas + remaining_bananas := by
  sorry

end NUMINAMATH_CALUDE_banana_count_l2977_297711


namespace NUMINAMATH_CALUDE_reciprocal_roots_condition_l2977_297754

/-- The roots of the quadratic equation 2x^2 + 5x + k = 0 are reciprocal if and only if k = 2 -/
theorem reciprocal_roots_condition (k : ℝ) : 
  (∃ x y : ℝ, x ≠ 0 ∧ y ≠ 0 ∧ x * y = 1 ∧ 2 * x^2 + 5 * x + k = 0 ∧ 2 * y^2 + 5 * y + k = 0) ↔ 
  k = 2 :=
by sorry

end NUMINAMATH_CALUDE_reciprocal_roots_condition_l2977_297754


namespace NUMINAMATH_CALUDE_power_of_two_equality_l2977_297724

theorem power_of_two_equality (x : ℕ) : 32^10 + 32^10 + 32^10 + 32^10 + 32^10 = 2^x ↔ x = 52 := by
  sorry

end NUMINAMATH_CALUDE_power_of_two_equality_l2977_297724


namespace NUMINAMATH_CALUDE_reflection_of_P_across_y_axis_l2977_297798

/-- Represents a point in a 2D Cartesian coordinate system -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Reflects a point across the y-axis -/
def reflectAcrossYAxis (p : Point2D) : Point2D :=
  { x := -p.x, y := p.y }

/-- The original point P -/
def P : Point2D := { x := 4, y := 1 }

/-- Theorem: The reflection of P(4,1) across the y-axis is (-4,1) -/
theorem reflection_of_P_across_y_axis :
  reflectAcrossYAxis P = { x := -4, y := 1 } := by
  sorry


end NUMINAMATH_CALUDE_reflection_of_P_across_y_axis_l2977_297798


namespace NUMINAMATH_CALUDE_xy_squared_l2977_297747

theorem xy_squared (x y : ℝ) 
  (h1 : 1/x + 1/y = 5) 
  (h2 : x*y - x - y = 2) : 
  x^2 * y^2 = 1/4 := by
sorry

end NUMINAMATH_CALUDE_xy_squared_l2977_297747


namespace NUMINAMATH_CALUDE_max_third_term_is_16_l2977_297753

/-- An arithmetic sequence of four positive integers with sum 50 -/
structure ArithmeticSequence where
  a : ℕ+  -- First term
  d : ℕ+  -- Common difference
  sum_eq_50 : a + (a + d) + (a + 2*d) + (a + 3*d) = 50

/-- The third term of an arithmetic sequence -/
def third_term (seq : ArithmeticSequence) : ℕ := seq.a + 2*seq.d

/-- Theorem: The maximum possible value of the third term is 16 -/
theorem max_third_term_is_16 :
  ∀ seq : ArithmeticSequence, third_term seq ≤ 16 ∧ ∃ seq : ArithmeticSequence, third_term seq = 16 :=
sorry

end NUMINAMATH_CALUDE_max_third_term_is_16_l2977_297753


namespace NUMINAMATH_CALUDE_trajectory_equation_l2977_297755

/-- The trajectory of point P satisfies x² + y² = 1, given a line l: x cos θ + y sin θ = 1,
    where OP is perpendicular to l at P, and O is the origin. -/
theorem trajectory_equation (θ : ℝ) (x y : ℝ) :
  (∃ (P : ℝ × ℝ), P.1 = x ∧ P.2 = y ∧
    (x * Real.cos θ + y * Real.sin θ = 1) ∧
    (∃ (t : ℝ), P = (t * Real.cos θ, t * Real.sin θ))) →
  x^2 + y^2 = 1 :=
by sorry

end NUMINAMATH_CALUDE_trajectory_equation_l2977_297755


namespace NUMINAMATH_CALUDE_fourth_power_sum_equals_51_to_fourth_l2977_297712

theorem fourth_power_sum_equals_51_to_fourth : 
  ∃! (n : ℕ+), 50^4 + 43^4 + 36^4 + 6^4 = n^4 :=
by sorry

end NUMINAMATH_CALUDE_fourth_power_sum_equals_51_to_fourth_l2977_297712


namespace NUMINAMATH_CALUDE_rectangle_inscribed_area_bound_l2977_297792

/-- A triangle represented by three points in the plane -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- A rectangle represented by four points in the plane -/
structure Rectangle where
  D : ℝ × ℝ
  E : ℝ × ℝ
  F : ℝ × ℝ
  G : ℝ × ℝ

/-- Function to calculate the area of a triangle -/
def triangleArea (t : Triangle) : ℝ := sorry

/-- Function to calculate the area of a rectangle -/
def rectangleArea (r : Rectangle) : ℝ := sorry

/-- Predicate to check if a rectangle is inscribed in a triangle -/
def isInscribed (r : Rectangle) (t : Triangle) : Prop := sorry

/-- Theorem: The area of a rectangle inscribed in a triangle does not exceed half of the area of the triangle -/
theorem rectangle_inscribed_area_bound (t : Triangle) (r : Rectangle) :
  isInscribed r t → rectangleArea r ≤ (1/2) * triangleArea t := by
  sorry

end NUMINAMATH_CALUDE_rectangle_inscribed_area_bound_l2977_297792


namespace NUMINAMATH_CALUDE_income_calculation_l2977_297716

/-- Calculates a person's income given the income to expenditure ratio and savings amount. -/
def calculate_income (income_ratio : ℕ) (expenditure_ratio : ℕ) (savings : ℕ) : ℕ :=
  (income_ratio * savings) / (income_ratio - expenditure_ratio)

/-- Proves that given the specified conditions, the person's income is 18000. -/
theorem income_calculation :
  let income_ratio : ℕ := 9
  let expenditure_ratio : ℕ := 8
  let savings : ℕ := 2000
  calculate_income income_ratio expenditure_ratio savings = 18000 := by
  sorry

#eval calculate_income 9 8 2000

end NUMINAMATH_CALUDE_income_calculation_l2977_297716


namespace NUMINAMATH_CALUDE_golden_state_team_total_points_l2977_297708

def golden_state_team_points : ℕ → ℕ → ℕ → ℕ → ℕ → Prop :=
  fun draymond curry kelly durant klay =>
    draymond = 12 ∧
    curry = 2 * draymond ∧
    kelly = 9 ∧
    durant = 2 * kelly ∧
    klay = draymond / 2 ∧
    draymond + curry + kelly + durant + klay = 69

theorem golden_state_team_total_points :
  ∃ (draymond curry kelly durant klay : ℕ),
    golden_state_team_points draymond curry kelly durant klay :=
by
  sorry

end NUMINAMATH_CALUDE_golden_state_team_total_points_l2977_297708


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2977_297726

theorem inequality_solution_set (x : ℝ) : 
  (-2 * x^2 + x < -3) ↔ (x < -1 ∨ x > 3/2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2977_297726
