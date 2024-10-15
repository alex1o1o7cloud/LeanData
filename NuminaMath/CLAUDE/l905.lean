import Mathlib

namespace NUMINAMATH_CALUDE_quadratic_root_implies_m_value_l905_90550

theorem quadratic_root_implies_m_value (m n : ℝ) :
  (Complex.I : ℂ).re = 0 ∧ (Complex.I : ℂ).im = 1 →
  ((-3 : ℂ) + 2 * Complex.I) ^ 2 + m * ((-3 : ℂ) + 2 * Complex.I) + n = 0 →
  m = 6 := by sorry

end NUMINAMATH_CALUDE_quadratic_root_implies_m_value_l905_90550


namespace NUMINAMATH_CALUDE_cubic_sum_minus_product_l905_90568

theorem cubic_sum_minus_product (a b c : ℝ) 
  (sum_eq : a + b + c = 12) 
  (sum_products_eq : a*b + a*c + b*c = 30) : 
  a^3 + b^3 + c^3 - 3*a*b*c = 1008 := by sorry

end NUMINAMATH_CALUDE_cubic_sum_minus_product_l905_90568


namespace NUMINAMATH_CALUDE_bus_departure_theorem_l905_90573

/-- Represents the rules for bus departure and current occupancy -/
structure BusOccupancy where
  min_departure : Nat
  max_departure : Nat
  current_occupancy : Nat
  departure_rule : min_departure > 15 ∧ max_departure ≤ 30
  occupancy_valid : current_occupancy < min_departure

/-- Calculates the number of additional people needed for the bus to depart -/
def additional_people_needed (bus : BusOccupancy) : Nat :=
  bus.min_departure - bus.current_occupancy

/-- Theorem stating that for a bus with specific occupancy rules and current state,
    the number of additional people needed is 7 -/
theorem bus_departure_theorem (bus : BusOccupancy)
    (h1 : bus.min_departure = 16)
    (h2 : bus.current_occupancy = 9) :
    additional_people_needed bus = 7 := by
  sorry

#eval additional_people_needed ⟨16, 30, 9, by simp, by simp⟩

end NUMINAMATH_CALUDE_bus_departure_theorem_l905_90573


namespace NUMINAMATH_CALUDE_stamp_trade_l905_90547

theorem stamp_trade (anna_initial alison_initial jeff_initial anna_final : ℕ) 
  (h1 : anna_initial = 37)
  (h2 : alison_initial = 28)
  (h3 : jeff_initial = 31)
  (h4 : anna_final = 50) : 
  (anna_initial + alison_initial / 2) - anna_final = 1 := by
  sorry

end NUMINAMATH_CALUDE_stamp_trade_l905_90547


namespace NUMINAMATH_CALUDE_equilateral_triangle_side_length_squared_l905_90523

/-- An ellipse with equation 9x^2 + 25y^2 = 225 -/
def Ellipse : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | 9 * p.1^2 + 25 * p.2^2 = 225}

/-- A point on the ellipse -/
def PointOnEllipse (p : ℝ × ℝ) : Prop :=
  p ∈ Ellipse

/-- An equilateral triangle -/
structure EquilateralTriangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  equilateral : (A.1 - B.1)^2 + (A.2 - B.2)^2 = (B.1 - C.1)^2 + (B.2 - C.2)^2 ∧
                (B.1 - C.1)^2 + (B.2 - C.2)^2 = (C.1 - A.1)^2 + (C.2 - A.2)^2

/-- The triangle is inscribed in the ellipse -/
def TriangleInscribed (t : EquilateralTriangle) : Prop :=
  PointOnEllipse t.A ∧ PointOnEllipse t.B ∧ PointOnEllipse t.C

/-- One vertex is at (5/3, 0) -/
def VertexAtGivenPoint (t : EquilateralTriangle) : Prop :=
  t.A = (5/3, 0) ∨ t.B = (5/3, 0) ∨ t.C = (5/3, 0)

/-- One altitude is contained in the x-axis -/
def AltitudeOnXAxis (t : EquilateralTriangle) : Prop :=
  (t.A.2 = 0 ∧ t.B.2 = -t.C.2) ∨ (t.B.2 = 0 ∧ t.A.2 = -t.C.2) ∨ (t.C.2 = 0 ∧ t.A.2 = -t.B.2)

/-- The main theorem -/
theorem equilateral_triangle_side_length_squared 
  (t : EquilateralTriangle) 
  (h1 : TriangleInscribed t) 
  (h2 : VertexAtGivenPoint t) 
  (h3 : AltitudeOnXAxis t) : 
  (t.A.1 - t.B.1)^2 + (t.A.2 - t.B.2)^2 = 1475/196 :=
sorry

end NUMINAMATH_CALUDE_equilateral_triangle_side_length_squared_l905_90523


namespace NUMINAMATH_CALUDE_min_value_of_f_l905_90549

-- Define the quadratic function
def f (x : ℝ) : ℝ := x^2 + 6*x + 9

-- Theorem stating that the minimum value of f is 0
theorem min_value_of_f :
  ∃ (x₀ : ℝ), ∀ (x : ℝ), f x₀ ≤ f x ∧ f x₀ = 0 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_f_l905_90549


namespace NUMINAMATH_CALUDE_sarah_time_hours_l905_90592

-- Define the time Samuel took in minutes
def samuel_time : ℕ := 30

-- Define the time difference between Sarah and Samuel in minutes
def time_difference : ℕ := 48

-- Define Sarah's time in minutes
def sarah_time_minutes : ℕ := samuel_time + time_difference

-- Define the conversion factor from minutes to hours
def minutes_per_hour : ℕ := 60

-- Theorem to prove
theorem sarah_time_hours : 
  (sarah_time_minutes : ℚ) / minutes_per_hour = 1.3 := by
  sorry

end NUMINAMATH_CALUDE_sarah_time_hours_l905_90592


namespace NUMINAMATH_CALUDE_lily_correct_answers_percentage_l905_90530

theorem lily_correct_answers_percentage 
  (t : ℝ) -- total number of problems
  (h_t_pos : t > 0) -- t is positive
  (h_max_alone : 0.85 * (2/3 * t) = 17/30 * t) -- Max's correct answers alone
  (h_max_total : 0.90 * t = 0.90 * t) -- Max's total correct answers
  (h_together : 0.75 * (1/3 * t) = 0.25 * t) -- Correct answers together
  (h_lily_alone : 0.95 * (2/3 * t) = 19/30 * t) -- Lily's correct answers alone
  : (19/30 * t + 0.25 * t) / t = 49/60 := by
  sorry

end NUMINAMATH_CALUDE_lily_correct_answers_percentage_l905_90530


namespace NUMINAMATH_CALUDE_valid_sets_l905_90512

theorem valid_sets (A : Set ℕ) : 
  (∀ m n : ℕ, m + n ∈ A → m * n ∈ A) ↔ 
  A = ∅ ∨ A = {0} ∨ A = {0, 1} ∨ A = {0, 1, 2} ∨ 
  A = {0, 1, 2, 3} ∨ A = {0, 1, 2, 3, 4} ∨ A = Set.univ :=
sorry

end NUMINAMATH_CALUDE_valid_sets_l905_90512


namespace NUMINAMATH_CALUDE_factor_expression_l905_90575

theorem factor_expression (x : ℝ) : 18 * x^2 + 9 * x - 3 = 3 * (6 * x^2 + 3 * x - 1) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l905_90575


namespace NUMINAMATH_CALUDE_train_car_estimate_l905_90521

/-- Represents the number of cars that pass in a given time interval -/
structure CarPassage where
  cars : ℕ
  seconds : ℕ

/-- Calculates the estimated number of cars in a train given initial observations and total passage time -/
def estimateTrainCars (initialObservation : CarPassage) (totalPassageTime : ℕ) : ℕ :=
  (initialObservation.cars * totalPassageTime) / initialObservation.seconds

theorem train_car_estimate :
  let initialObservation : CarPassage := { cars := 8, seconds := 12 }
  let totalPassageTime : ℕ := 210
  estimateTrainCars initialObservation totalPassageTime = 140 := by
  sorry

end NUMINAMATH_CALUDE_train_car_estimate_l905_90521


namespace NUMINAMATH_CALUDE_combined_body_is_pentahedron_l905_90577

/-- Represents a regular quadrangular pyramid -/
structure RegularQuadrangularPyramid where
  edge_length : ℝ

/-- Represents a regular tetrahedron -/
structure RegularTetrahedron where
  edge_length : ℝ

/-- Represents the new geometric body formed by combining a regular quadrangular pyramid and a regular tetrahedron -/
structure CombinedBody where
  pyramid : RegularQuadrangularPyramid
  tetrahedron : RegularTetrahedron

/-- Defines the property of being a pentahedron -/
def is_pentahedron (body : CombinedBody) : Prop := sorry

theorem combined_body_is_pentahedron 
  (pyramid : RegularQuadrangularPyramid) 
  (tetrahedron : RegularTetrahedron) 
  (h : pyramid.edge_length = tetrahedron.edge_length) : 
  is_pentahedron (CombinedBody.mk pyramid tetrahedron) :=
sorry

end NUMINAMATH_CALUDE_combined_body_is_pentahedron_l905_90577


namespace NUMINAMATH_CALUDE_scooter_initial_value_l905_90500

/-- Proves that the initial value of a scooter is 40000 given its depreciation rate and value after 2 years -/
theorem scooter_initial_value (depreciation_rate : ℚ) (value_after_two_years : ℚ) :
  depreciation_rate = 3 / 4 →
  value_after_two_years = 22500 →
  depreciation_rate * (depreciation_rate * 40000) = value_after_two_years :=
by sorry

end NUMINAMATH_CALUDE_scooter_initial_value_l905_90500


namespace NUMINAMATH_CALUDE_initial_files_correct_l905_90518

/-- The number of files Nancy had initially -/
def initial_files : ℕ := 80

/-- The number of files Nancy deleted -/
def deleted_files : ℕ := 31

/-- The number of folders Nancy ended up with -/
def num_folders : ℕ := 7

/-- The number of files in each folder -/
def files_per_folder : ℕ := 7

/-- Theorem stating that the initial number of files is correct -/
theorem initial_files_correct : 
  initial_files = deleted_files + num_folders * files_per_folder := by
  sorry

end NUMINAMATH_CALUDE_initial_files_correct_l905_90518


namespace NUMINAMATH_CALUDE_hot_drink_price_range_l905_90574

/-- Represents the price increase in yuan -/
def price_increase : ℝ → ℝ := λ x => x

/-- Represents the new price of a hot drink in yuan -/
def new_price : ℝ → ℝ := λ x => 1.5 + price_increase x

/-- Represents the daily sales volume as a function of price increase -/
def daily_sales : ℝ → ℝ := λ x => 800 - 20 * (10 * price_increase x)

/-- Represents the daily profit as a function of price increase -/
def daily_profit : ℝ → ℝ := λ x => (new_price x - 0.9) * daily_sales x

theorem hot_drink_price_range :
  ∃ (lower upper : ℝ), lower = 1.9 ∧ upper = 4.5 ∧
  ∀ x, daily_profit x ≥ 720 ↔ new_price x ∈ Set.Icc lower upper :=
by sorry

end NUMINAMATH_CALUDE_hot_drink_price_range_l905_90574


namespace NUMINAMATH_CALUDE_integer_fraction_condition_l905_90501

theorem integer_fraction_condition (a b : ℕ+) :
  (∃ k : ℕ+, (a.val ^ 2 : ℚ) / (2 * a.val * b.val ^ 2 - b.val ^ 3 + 1) = k.val) ↔
  (∃ l : ℕ+, (a = l ∧ b = 2 * l) ∨ (a = 8 * l.val ^ 4 - l ∧ b = 2 * l)) :=
by sorry

end NUMINAMATH_CALUDE_integer_fraction_condition_l905_90501


namespace NUMINAMATH_CALUDE_courtney_marble_count_l905_90590

/-- The number of marbles in Courtney's collection -/
def total_marbles (jar1 jar2 jar3 : ℕ) : ℕ := jar1 + jar2 + jar3

/-- Theorem: Courtney's total marble count -/
theorem courtney_marble_count :
  ∀ (jar1 jar2 jar3 : ℕ),
    jar1 = 80 →
    jar2 = 2 * jar1 →
    jar3 = jar1 / 4 →
    total_marbles jar1 jar2 jar3 = 260 := by
  sorry

#check courtney_marble_count

end NUMINAMATH_CALUDE_courtney_marble_count_l905_90590


namespace NUMINAMATH_CALUDE_tax_reduction_theorem_l905_90565

/-- Proves that a tax reduction of 15% results in a 6.5% revenue decrease
    when consumption increases by 10% -/
theorem tax_reduction_theorem (T C : ℝ) (X : ℝ) 
  (h_positive_T : T > 0) 
  (h_positive_C : C > 0) 
  (h_consumption_increase : 1.1 * C = C + 0.1 * C) 
  (h_revenue_decrease : (T * (1 - X / 100) * (C * 1.1)) = T * C * 0.935) :
  X = 15 := by
  sorry

end NUMINAMATH_CALUDE_tax_reduction_theorem_l905_90565


namespace NUMINAMATH_CALUDE_supermarket_spending_l905_90526

theorem supermarket_spending (total : ℚ) : 
  (2/5 : ℚ) * total + (1/4 : ℚ) * total + (1/10 : ℚ) * total + 
  (1/8 : ℚ) * total + (1/20 : ℚ) * total + 12 = total → 
  total = 160 := by
sorry

end NUMINAMATH_CALUDE_supermarket_spending_l905_90526


namespace NUMINAMATH_CALUDE_circle_diameter_l905_90599

theorem circle_diameter (x y : ℝ) (h : x + y = 100 * Real.pi) : ∃ (r : ℝ), 
  x = Real.pi * r^2 ∧ y = 2 * Real.pi * r ∧ 2 * r = 16 := by
  sorry

end NUMINAMATH_CALUDE_circle_diameter_l905_90599


namespace NUMINAMATH_CALUDE_preimage_of_one_is_zero_one_neg_one_l905_90510

-- Define the sets A and B as subsets of ℝ
variable (A B : Set ℝ)

-- Define the function f: A → B
def f (x : ℝ) : ℝ := x^3 - x + 1

-- Define the set of elements in A that map to 1 under f
def preimage_of_one (A : Set ℝ) : Set ℝ := {x ∈ A | f x = 1}

-- Theorem statement
theorem preimage_of_one_is_zero_one_neg_one (A B : Set ℝ) :
  preimage_of_one A = {0, 1, -1} := by sorry

end NUMINAMATH_CALUDE_preimage_of_one_is_zero_one_neg_one_l905_90510


namespace NUMINAMATH_CALUDE_luka_water_needed_l905_90517

/-- Represents the recipe ratios and amount of lemon juice used --/
structure Recipe where
  water_sugar_ratio : ℚ
  sugar_lemon_ratio : ℚ
  lemon_juice : ℚ

/-- Calculates the amount of water needed based on the recipe --/
def water_needed (r : Recipe) : ℚ :=
  r.water_sugar_ratio * r.sugar_lemon_ratio * r.lemon_juice

/-- Theorem stating that Luka needs 24 cups of water --/
theorem luka_water_needed :
  let r : Recipe := {
    water_sugar_ratio := 4,
    sugar_lemon_ratio := 2,
    lemon_juice := 3
  }
  water_needed r = 24 := by sorry

end NUMINAMATH_CALUDE_luka_water_needed_l905_90517


namespace NUMINAMATH_CALUDE_line_inclination_l905_90564

/-- The inclination angle of a line given by parametric equations -/
def inclination_angle (x_eq : ℝ → ℝ) (y_eq : ℝ → ℝ) : ℝ :=
  sorry

theorem line_inclination :
  let x_eq := λ t : ℝ => 3 + t * Real.sin (25 * π / 180)
  let y_eq := λ t : ℝ => -t * Real.cos (25 * π / 180)
  inclination_angle x_eq y_eq = 115 * π / 180 :=
sorry

end NUMINAMATH_CALUDE_line_inclination_l905_90564


namespace NUMINAMATH_CALUDE_product_of_numbers_l905_90543

theorem product_of_numbers (x y : ℝ) (h1 : x + y = 60) (h2 : x - y = 10) : x * y = 875 := by
  sorry

end NUMINAMATH_CALUDE_product_of_numbers_l905_90543


namespace NUMINAMATH_CALUDE_percentage_equivalence_l905_90567

theorem percentage_equivalence (x : ℝ) (h : (30/100) * ((15/100) * x) = 27) :
  (15/100) * ((30/100) * x) = 27 := by
  sorry

end NUMINAMATH_CALUDE_percentage_equivalence_l905_90567


namespace NUMINAMATH_CALUDE_wallpaper_area_proof_l905_90562

theorem wallpaper_area_proof (total_area overlap_area double_layer triple_layer : ℝ) : 
  overlap_area = 180 →
  double_layer = 30 →
  triple_layer = 45 →
  total_area - 2 * double_layer - 3 * triple_layer = overlap_area →
  total_area = 375 := by
  sorry

end NUMINAMATH_CALUDE_wallpaper_area_proof_l905_90562


namespace NUMINAMATH_CALUDE_billy_reading_speed_l905_90569

/-- Represents Billy's reading speed in pages per hour -/
def reading_speed (
  free_time_per_day : ℕ)  -- Free time per day in hours
  (weekend_days : ℕ)      -- Number of weekend days
  (gaming_percentage : ℚ) -- Percentage of time spent gaming
  (pages_per_book : ℕ)    -- Number of pages in each book
  (books_read : ℕ)        -- Number of books read
  : ℚ :=
  let total_free_time := free_time_per_day * weekend_days
  let reading_time := total_free_time * (1 - gaming_percentage)
  let total_pages := pages_per_book * books_read
  total_pages / reading_time

theorem billy_reading_speed :
  reading_speed 8 2 (3/4) 80 3 = 60 := by
  sorry

end NUMINAMATH_CALUDE_billy_reading_speed_l905_90569


namespace NUMINAMATH_CALUDE_must_divide_p_l905_90533

theorem must_divide_p (p q r s : ℕ+) 
  (h1 : Nat.gcd p q = 30)
  (h2 : Nat.gcd q r = 45)
  (h3 : Nat.gcd r s = 60)
  (h4 : 80 < Nat.gcd s p ∧ Nat.gcd s p < 120) :
  5 ∣ p := by
  sorry

end NUMINAMATH_CALUDE_must_divide_p_l905_90533


namespace NUMINAMATH_CALUDE_min_reciprocal_sum_l905_90563

theorem min_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 20) :
  (1 / x + 1 / y) ≥ (1 / 5 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_min_reciprocal_sum_l905_90563


namespace NUMINAMATH_CALUDE_root_sum_property_l905_90506

theorem root_sum_property (x₁ x₂ : ℝ) (n : ℕ) (hn : n ≥ 1) :
  (x₁^2 - 6*x₁ + 1 = 0) → (x₂^2 - 6*x₂ + 1 = 0) →
  (∃ (m : ℤ), x₁^n + x₂^n = m) ∧ ¬(∃ (k : ℤ), x₁^n + x₂^n = 5*k) := by
  sorry

end NUMINAMATH_CALUDE_root_sum_property_l905_90506


namespace NUMINAMATH_CALUDE_triangle_angle_problem_l905_90587

/-- A prime number greater than 2 -/
def OddPrime (n : ℕ) : Prop := Nat.Prime n ∧ n > 2

theorem triangle_angle_problem :
  ∀ y z w : ℕ,
    OddPrime y →
    OddPrime z →
    OddPrime w →
    y + z + w = 90 →
    (∀ w' : ℕ, OddPrime w' → y + z + w' = 90 → w ≤ w') →
    w = 83 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_problem_l905_90587


namespace NUMINAMATH_CALUDE_max_value_inequality_max_value_achievable_l905_90556

theorem max_value_inequality (a b c d e : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) (he : e > 0) :
  (a * b + 2 * b * c + 2 * c * d + d * e) / (a^2 + 3 * b^2 + 3 * c^2 + 5 * d^2 + e^2) ≤ Real.sqrt 2 :=
by sorry

theorem max_value_achievable :
  ∃ (a b c d e : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧
    (a * b + 2 * b * c + 2 * c * d + d * e) / (a^2 + 3 * b^2 + 3 * c^2 + 5 * d^2 + e^2) = Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_max_value_inequality_max_value_achievable_l905_90556


namespace NUMINAMATH_CALUDE_roof_dimension_difference_l905_90594

/-- Represents the dimensions of a rectangular roof -/
structure RoofDimensions where
  width : ℝ
  length : ℝ
  area : ℝ
  length_width_ratio : length = 4 * width
  area_equation : area = length * width

/-- The difference between the length and width of the roof -/
def length_width_difference (roof : RoofDimensions) : ℝ :=
  roof.length - roof.width

/-- Theorem stating the approximate difference between length and width -/
theorem roof_dimension_difference : 
  ∃ (roof : RoofDimensions), 
    roof.area = 675 ∧ 
    (abs (length_width_difference roof - 38.97) < 0.01) := by
  sorry

end NUMINAMATH_CALUDE_roof_dimension_difference_l905_90594


namespace NUMINAMATH_CALUDE_proportion_problem_l905_90529

/-- Given that a, b, c, and d are in proportion, where a = 3, b = 2, and c = 6, prove that d = 4. -/
theorem proportion_problem (a b c d : ℝ) : 
  a = 3 → b = 2 → c = 6 → (a * d = b * c) → d = 4 := by
  sorry

end NUMINAMATH_CALUDE_proportion_problem_l905_90529


namespace NUMINAMATH_CALUDE_max_product_2017_l905_90555

def sumToN (n : ℕ) := {l : List ℕ | l.sum = n}

def productOfList (l : List ℕ) := l.prod

def optimalSumProduct (n : ℕ) : List ℕ := 
  List.replicate 671 3 ++ List.replicate 2 2

theorem max_product_2017 :
  ∀ l ∈ sumToN 2017, 
    productOfList l ≤ productOfList (optimalSumProduct 2017) :=
sorry

end NUMINAMATH_CALUDE_max_product_2017_l905_90555


namespace NUMINAMATH_CALUDE_problem_statement_l905_90582

theorem problem_statement (x y : ℝ) (h : -y + 3*x = 3) : 
  2*(y - 3*x) - (3*x - y)^2 + 1 = -14 := by sorry

end NUMINAMATH_CALUDE_problem_statement_l905_90582


namespace NUMINAMATH_CALUDE_arithmetic_sequence_proof_l905_90553

/-- 
Given three consecutive terms in an arithmetic sequence: 4x, 2x-3, and 4x-3,
prove that x = -3/4
-/
theorem arithmetic_sequence_proof (x : ℚ) : 
  (∃ (d : ℚ), (2*x - 3) - 4*x = d ∧ (4*x - 3) - (2*x - 3) = d) → 
  x = -3/4 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_proof_l905_90553


namespace NUMINAMATH_CALUDE_triple_lcm_equation_l905_90504

theorem triple_lcm_equation (a b c n : ℕ+) :
  (a.val^2 + b.val^2 = n.val * Nat.lcm a.val b.val + n.val^2) ∧
  (b.val^2 + c.val^2 = n.val * Nat.lcm b.val c.val + n.val^2) ∧
  (c.val^2 + a.val^2 = n.val * Nat.lcm c.val a.val + n.val^2) →
  a = b ∧ b = c := by
  sorry

end NUMINAMATH_CALUDE_triple_lcm_equation_l905_90504


namespace NUMINAMATH_CALUDE_ones_digit_of_largest_power_of_two_dividing_32_factorial_l905_90551

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def largest_power_of_two_dividing (n : ℕ) : ℕ :=
  -- This function is not implemented, but represents the concept
  sorry

def ones_digit (n : ℕ) : ℕ :=
  n % 10

theorem ones_digit_of_largest_power_of_two_dividing_32_factorial :
  ones_digit (largest_power_of_two_dividing (factorial 32)) = 8 := by
  sorry

end NUMINAMATH_CALUDE_ones_digit_of_largest_power_of_two_dividing_32_factorial_l905_90551


namespace NUMINAMATH_CALUDE_squares_theorem_l905_90586

-- Define the points and lengths
variable (A B C O : ℝ × ℝ)
variable (a b c : ℝ)

-- Define the conditions
def squares_condition (A B C O : ℝ × ℝ) (a b c : ℝ) : Prop :=
  A = (a, a) ∧
  B = (b, 2*a + b) ∧
  C = (-c, c) ∧
  O = (0, 0) ∧
  c = a + b

-- Define the equality of line segments
def line_segments_equal (P Q R S : ℝ × ℝ) : Prop :=
  (P.1 - Q.1)^2 + (P.2 - Q.2)^2 = (R.1 - S.1)^2 + (R.2 - S.2)^2

-- Define perpendicularity of line segments
def perpendicular (P Q R S : ℝ × ℝ) : Prop :=
  (Q.1 - P.1) * (S.1 - R.1) + (Q.2 - P.2) * (S.2 - R.2) = 0

-- State the theorem
theorem squares_theorem (A B C O : ℝ × ℝ) (a b c : ℝ) 
  (h : squares_condition A B C O a b c) : 
  line_segments_equal O B A C ∧ perpendicular O B A C := by
  sorry

end NUMINAMATH_CALUDE_squares_theorem_l905_90586


namespace NUMINAMATH_CALUDE_probability_one_white_one_black_l905_90557

/-- The probability of drawing one white ball and one black ball from a box -/
theorem probability_one_white_one_black (w b : ℕ) (hw : w = 7) (hb : b = 8) :
  let total := w + b
  let favorable := w * b
  let total_combinations := (total * (total - 1)) / 2
  (favorable : ℚ) / total_combinations = 56 / 105 := by sorry

end NUMINAMATH_CALUDE_probability_one_white_one_black_l905_90557


namespace NUMINAMATH_CALUDE_simplify_polynomial_l905_90571

theorem simplify_polynomial (p : ℝ) : 
  (3 * p^3 - 5*p + 6) + (4 - 6*p^2 + 2*p) = 3*p^3 - 6*p^2 - 3*p + 10 := by
  sorry

end NUMINAMATH_CALUDE_simplify_polynomial_l905_90571


namespace NUMINAMATH_CALUDE_functional_equation_solution_l905_90554

theorem functional_equation_solution (f : ℚ → ℚ) 
  (h1 : f 1 = 2) 
  (h2 : ∀ x y : ℚ, f (x * y) = f x * f y - f (x + y) + 1) : 
  ∀ x : ℚ, f x = x + 1 := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l905_90554


namespace NUMINAMATH_CALUDE_cupcakes_sold_correct_l905_90579

/-- Represents the number of cupcakes Katie sold at the bake sale -/
def cupcakes_sold (initial : ℕ) (additional : ℕ) (remaining : ℕ) : ℕ :=
  initial + additional - remaining

/-- Proves that the number of cupcakes sold is correct given the initial,
    additional, and remaining cupcakes -/
theorem cupcakes_sold_correct (initial : ℕ) (additional : ℕ) (remaining : ℕ) :
  cupcakes_sold initial additional remaining = initial + additional - remaining :=
by sorry

end NUMINAMATH_CALUDE_cupcakes_sold_correct_l905_90579


namespace NUMINAMATH_CALUDE_lcm_gcf_relation_l905_90516

theorem lcm_gcf_relation (n : ℕ) :
  Nat.lcm n 24 = 48 ∧ Nat.gcd n 24 = 8 → n = 16 := by
  sorry

end NUMINAMATH_CALUDE_lcm_gcf_relation_l905_90516


namespace NUMINAMATH_CALUDE_power_equation_solution_l905_90525

theorem power_equation_solution (m n : ℕ) (h1 : (1/5)^m * (1/4)^n = 1/(10^4)) (h2 : m = 4) : n = 2 := by
  sorry

end NUMINAMATH_CALUDE_power_equation_solution_l905_90525


namespace NUMINAMATH_CALUDE_second_divisor_problem_l905_90511

theorem second_divisor_problem (n : Nat) (h1 : n > 13) (h2 : n ∣ 192) : 
  (197 % n = 5 ∧ ∀ m : Nat, m > 13 → m < n → m ∣ 192 → 197 % m ≠ 5) → n = 16 := by
  sorry

end NUMINAMATH_CALUDE_second_divisor_problem_l905_90511


namespace NUMINAMATH_CALUDE_square_sum_lower_bound_l905_90535

theorem square_sum_lower_bound (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  a^2 + b^2 ≥ 1/2 := by sorry

end NUMINAMATH_CALUDE_square_sum_lower_bound_l905_90535


namespace NUMINAMATH_CALUDE_solution_to_equation_l905_90528

theorem solution_to_equation (x y z : ℝ) 
  (hx : x > 1) (hy : y > 1) (hz : z > 1)
  (h_eq : x + y + z + 3 / (x - 1) + 3 / (y - 1) + 3 / (z - 1) = 
          2 * (Real.sqrt (x + 2) + Real.sqrt (y + 2) + Real.sqrt (z + 2))) :
  x = (3 + Real.sqrt 13) / 2 ∧ 
  y = (3 + Real.sqrt 13) / 2 ∧ 
  z = (3 + Real.sqrt 13) / 2 := by
sorry

end NUMINAMATH_CALUDE_solution_to_equation_l905_90528


namespace NUMINAMATH_CALUDE_train_crossing_time_l905_90559

theorem train_crossing_time (train_speed_kmph : ℝ) (platform_length : ℝ) (platform_crossing_time : ℝ) :
  train_speed_kmph = 72 →
  platform_length = 320 →
  platform_crossing_time = 34 →
  let train_speed_mps := train_speed_kmph * (1000 / 3600)
  let train_length := train_speed_mps * platform_crossing_time - platform_length
  let man_crossing_time := train_length / train_speed_mps
  man_crossing_time = 18 := by
  sorry

end NUMINAMATH_CALUDE_train_crossing_time_l905_90559


namespace NUMINAMATH_CALUDE_complex_magnitude_of_special_z_l905_90583

theorem complex_magnitude_of_special_z : 
  let i : ℂ := Complex.I
  let z : ℂ := -i^2022 + i
  Complex.abs z = Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_complex_magnitude_of_special_z_l905_90583


namespace NUMINAMATH_CALUDE_trig_inequality_l905_90509

theorem trig_inequality (α β γ : Real) 
  (h1 : 0 ≤ α ∧ α < Real.pi / 2)
  (h2 : 0 ≤ β ∧ β < Real.pi / 2)
  (h3 : 0 ≤ γ ∧ γ < Real.pi / 2)
  (h4 : Real.sin α + Real.sin β + Real.sin γ = 1) :
  Real.tan α ^ 2 + Real.tan β ^ 2 + Real.tan γ ^ 2 ≥ 3/8 := by
sorry

end NUMINAMATH_CALUDE_trig_inequality_l905_90509


namespace NUMINAMATH_CALUDE_equation_one_solution_l905_90532

theorem equation_one_solution (a : ℝ) : 
  (∃! x : ℝ, (Real.log (x + 1) + Real.log (3 - x) = Real.log (1 - a * x)) ∧ 
   (-1 < x ∧ x < 3)) ↔ 
  (-1 ≤ a ∧ a ≤ 1/3) :=
sorry

end NUMINAMATH_CALUDE_equation_one_solution_l905_90532


namespace NUMINAMATH_CALUDE_biology_group_size_l905_90524

theorem biology_group_size : 
  ∃ (n : ℕ), n > 0 ∧ n * (n - 1) = 210 ∧ ∀ m : ℕ, m > 0 ∧ m * (m - 1) = 210 → m = n :=
by sorry

end NUMINAMATH_CALUDE_biology_group_size_l905_90524


namespace NUMINAMATH_CALUDE_theater_ticket_pricing_l905_90514

/-- Theorem: Theater Ticket Pricing
  Given:
  - Total tickets sold is 340
  - Total revenue is $3,320
  - Orchestra seat price is $12
  - Number of balcony seats sold is 40 more than orchestra seats
  Prove that the cost of a balcony seat is $8
-/
theorem theater_ticket_pricing 
  (total_tickets : ℕ) 
  (total_revenue : ℕ) 
  (orchestra_price : ℕ) 
  (balcony_excess : ℕ) 
  (h1 : total_tickets = 340)
  (h2 : total_revenue = 3320)
  (h3 : orchestra_price = 12)
  (h4 : balcony_excess = 40) :
  let orchestra_seats := (total_tickets - balcony_excess) / 2
  let balcony_seats := orchestra_seats + balcony_excess
  let balcony_revenue := total_revenue - orchestra_price * orchestra_seats
  balcony_revenue / balcony_seats = 8 := by
  sorry

end NUMINAMATH_CALUDE_theater_ticket_pricing_l905_90514


namespace NUMINAMATH_CALUDE_determinant_trig_matrix_l905_90561

open Real Matrix

theorem determinant_trig_matrix (α β : ℝ) : 
  det ![![sin α * sin β, -sin α * cos β, cos α],
        ![cos β, sin β, 0],
        ![-cos α * sin β, -cos α * cos β, sin α]] = 1 - cos α := by
  sorry

end NUMINAMATH_CALUDE_determinant_trig_matrix_l905_90561


namespace NUMINAMATH_CALUDE_sin_sum_inverse_trig_functions_l905_90585

theorem sin_sum_inverse_trig_functions :
  Real.sin (Real.arcsin (4/5) + Real.arctan (3/2) + Real.arccos (1/3)) = (17 - 12 * Real.sqrt 2) / (15 * Real.sqrt 13) := by
  sorry

end NUMINAMATH_CALUDE_sin_sum_inverse_trig_functions_l905_90585


namespace NUMINAMATH_CALUDE_det_of_specific_matrix_l905_90519

theorem det_of_specific_matrix :
  let A : Matrix (Fin 2) (Fin 2) ℤ := !![5, -4; 2, 3]
  Matrix.det A = 23 := by
sorry

end NUMINAMATH_CALUDE_det_of_specific_matrix_l905_90519


namespace NUMINAMATH_CALUDE_probability_of_white_ball_l905_90593

def total_balls : ℕ := 4 + 6
def white_balls : ℕ := 4
def yellow_balls : ℕ := 6

theorem probability_of_white_ball :
  (white_balls : ℚ) / total_balls = 2 / 5 :=
sorry

end NUMINAMATH_CALUDE_probability_of_white_ball_l905_90593


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l905_90546

theorem algebraic_expression_value (x y : ℝ) (h : 2 * x - y = 2) : 
  6 * x - 3 * y + 1 = 7 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l905_90546


namespace NUMINAMATH_CALUDE_baking_time_proof_l905_90541

/-- Alice's pie-baking time in minutes -/
def alice_time : ℕ := 5

/-- Bob's pie-baking time in minutes -/
def bob_time : ℕ := 6

/-- The time period in which Alice bakes 2 more pies than Bob -/
def time_period : ℕ := 60

theorem baking_time_proof :
  (time_period / alice_time : ℚ) = (time_period / bob_time : ℚ) + 2 :=
by sorry

end NUMINAMATH_CALUDE_baking_time_proof_l905_90541


namespace NUMINAMATH_CALUDE_total_legos_after_winning_l905_90578

def initial_legos : ℕ := 2080
def won_legos : ℕ := 17

theorem total_legos_after_winning :
  initial_legos + won_legos = 2097 := by sorry

end NUMINAMATH_CALUDE_total_legos_after_winning_l905_90578


namespace NUMINAMATH_CALUDE_chef_wage_percentage_increase_l905_90558

/-- Proves that the percentage increase in the hourly wage of a chef compared to a dishwasher is 20% -/
theorem chef_wage_percentage_increase (manager_wage : ℝ) (chef_wage : ℝ) (dishwasher_wage : ℝ) :
  manager_wage = 7.5 →
  chef_wage = manager_wage - 3 →
  dishwasher_wage = manager_wage / 2 →
  (chef_wage - dishwasher_wage) / dishwasher_wage * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_chef_wage_percentage_increase_l905_90558


namespace NUMINAMATH_CALUDE_hospital_staff_count_l905_90570

theorem hospital_staff_count (total : ℕ) (doctor_ratio nurse_ratio : ℕ) 
  (h1 : total = 500)
  (h2 : doctor_ratio = 7)
  (h3 : nurse_ratio = 8) : 
  ∃ (nurses : ℕ), nurses = 264 ∧ 
    ∃ (doctors : ℕ), doctors + nurses = total ∧ 
      doctor_ratio * nurses = nurse_ratio * doctors :=
sorry

end NUMINAMATH_CALUDE_hospital_staff_count_l905_90570


namespace NUMINAMATH_CALUDE_fraction_equality_l905_90508

theorem fraction_equality (p q r s : ℝ) 
  (h : (p - q) * (r - s) / ((q - r) * (s - p)) = 3 / 7) : 
  (p - r) * (q - s) / ((p - q) * (r - s)) = -4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l905_90508


namespace NUMINAMATH_CALUDE_intern_teacher_assignment_l905_90560

theorem intern_teacher_assignment :
  (∀ n m : ℕ, n = 4 ∧ m = 3 ∧ n > m) →
  (number_of_assignments : ℕ) →
  number_of_assignments = 36 :=
by
  sorry

end NUMINAMATH_CALUDE_intern_teacher_assignment_l905_90560


namespace NUMINAMATH_CALUDE_xyz_relation_theorem_l905_90540

/-- A structure representing the relationship between x, y, and z -/
structure XYZRelation where
  x : ℝ
  y : ℝ
  z : ℝ
  c : ℝ
  d : ℝ
  h1 : y^2 = c * z^2  -- y² varies directly with z²
  h2 : y = d / x      -- y varies inversely with x

/-- The theorem statement -/
theorem xyz_relation_theorem (r : XYZRelation) (h3 : r.y = 3) (h4 : r.x = 4) (h5 : r.z = 6) :
  ∃ (r' : XYZRelation), r'.y = 2 ∧ r'.z = 12 ∧ r'.x = 6 ∧ r'.c = r.c ∧ r'.d = r.d :=
sorry


end NUMINAMATH_CALUDE_xyz_relation_theorem_l905_90540


namespace NUMINAMATH_CALUDE_dealer_profit_percentage_l905_90589

/-- Calculates the profit percentage for a dealer's transaction -/
def profit_percentage (purchase_quantity : ℕ) (purchase_price : ℚ) 
                      (sale_quantity : ℕ) (sale_price : ℚ) : ℚ :=
  let cost_per_article := purchase_price / purchase_quantity
  let sale_per_article := sale_price / sale_quantity
  let profit_per_article := sale_per_article - cost_per_article
  (profit_per_article / cost_per_article) * 100

/-- The profit percentage for the given dealer transaction is approximately 89.99% -/
theorem dealer_profit_percentage :
  let result := profit_percentage 15 25 12 38
  ∃ (ε : ℚ), ε > 0 ∧ ε < (1 / 100) ∧ |result - 8999 / 100| < ε :=
sorry

end NUMINAMATH_CALUDE_dealer_profit_percentage_l905_90589


namespace NUMINAMATH_CALUDE_machine_doesnt_require_repair_l905_90596

/-- Represents a weighing machine for food portions -/
structure WeighingMachine where
  max_deviation : ℝ
  nominal_mass : ℝ
  unreadable_deviation_bound : ℝ

/-- Determines if a weighing machine requires repair based on its measurements -/
def requires_repair (m : WeighingMachine) : Prop :=
  m.max_deviation > 0.1 * m.nominal_mass ∨ 
  m.unreadable_deviation_bound ≥ m.max_deviation

theorem machine_doesnt_require_repair (m : WeighingMachine) 
  (h1 : m.max_deviation = 37)
  (h2 : m.max_deviation ≤ 0.1 * m.nominal_mass)
  (h3 : m.unreadable_deviation_bound < m.max_deviation) :
  ¬(requires_repair m) :=
sorry

#check machine_doesnt_require_repair

end NUMINAMATH_CALUDE_machine_doesnt_require_repair_l905_90596


namespace NUMINAMATH_CALUDE_equal_probability_wsw_more_advantageous_l905_90527

-- Define the probabilities of winning against strong and weak players
variable (Ps Pw : ℝ)

-- Define the condition that Ps < Pw
variable (h : Ps < Pw)

-- Define the probability of winning two consecutive games in the sequence Strong, Weak, Strong
def prob_sws : ℝ := Ps * Pw

-- Define the probability of winning two consecutive games in the sequence Weak, Strong, Weak
def prob_wsw : ℝ := Pw * Ps

-- Theorem stating that both sequences have equal probability
theorem equal_probability : prob_sws Ps Pw = prob_wsw Ps Pw := by
  sorry

-- Theorem stating that Weak, Strong, Weak is more advantageous
theorem wsw_more_advantageous (h : Ps < Pw) : prob_wsw Ps Pw ≥ prob_sws Ps Pw := by
  sorry

end NUMINAMATH_CALUDE_equal_probability_wsw_more_advantageous_l905_90527


namespace NUMINAMATH_CALUDE_solution_implies_m_minus_n_abs_l905_90591

/-- Given a system of equations 2x - y = m and x + my = n with solution x = 2 and y = 1, 
    prove that |m - n| = 2 -/
theorem solution_implies_m_minus_n_abs (m n : ℝ) 
  (h1 : 2 * 2 - 1 = m) 
  (h2 : 2 + m * 1 = n) : 
  |m - n| = 2 := by
  sorry

end NUMINAMATH_CALUDE_solution_implies_m_minus_n_abs_l905_90591


namespace NUMINAMATH_CALUDE_thirteen_ts_possible_l905_90598

/-- Represents a grid with horizontal and vertical lines -/
structure Grid :=
  (horizontal_lines : ℕ)
  (vertical_lines : ℕ)

/-- Represents a T shape on the grid -/
structure TShape :=
  (intersections : ℕ)

/-- The problem setup -/
def problem_setup : Prop :=
  ∃ (g : Grid) (t : TShape),
    g.horizontal_lines = 9 ∧
    g.vertical_lines = 9 ∧
    t.intersections = 5

/-- The theorem to be proved -/
theorem thirteen_ts_possible (h : problem_setup) : 
  ∃ (n : ℕ), n = 13 ∧ n * 5 ≤ 9 * 9 :=
sorry

end NUMINAMATH_CALUDE_thirteen_ts_possible_l905_90598


namespace NUMINAMATH_CALUDE_wood_frog_count_l905_90595

theorem wood_frog_count (total : ℕ) (tree : ℕ) (poison : ℕ) (wood : ℕ) 
  (h1 : total = 78)
  (h2 : tree = 55)
  (h3 : poison = 10)
  (h4 : total = tree + poison + wood) : wood = 13 := by
  sorry

end NUMINAMATH_CALUDE_wood_frog_count_l905_90595


namespace NUMINAMATH_CALUDE_josh_remaining_money_l905_90503

/-- Calculates the remaining money after Josh's shopping trip. -/
def remaining_money (initial_amount hat_cost pencil_cost cookie_cost cookie_count : ℚ) : ℚ :=
  initial_amount - (hat_cost + pencil_cost + cookie_cost * cookie_count)

/-- Theorem stating that Josh has $3 left after his purchases. -/
theorem josh_remaining_money :
  remaining_money 20 10 2 1.25 4 = 3 := by
  sorry

end NUMINAMATH_CALUDE_josh_remaining_money_l905_90503


namespace NUMINAMATH_CALUDE_square_root_range_l905_90513

theorem square_root_range (x : ℝ) : 
  (∃ y : ℝ, y^2 = x - 4) → x ≥ 4 := by sorry

end NUMINAMATH_CALUDE_square_root_range_l905_90513


namespace NUMINAMATH_CALUDE_tissue_used_count_l905_90505

def initial_tissue_count : ℕ := 97
def remaining_tissue_count : ℕ := 93

theorem tissue_used_count : initial_tissue_count - remaining_tissue_count = 4 := by
  sorry

end NUMINAMATH_CALUDE_tissue_used_count_l905_90505


namespace NUMINAMATH_CALUDE_eleven_by_seven_max_squares_l905_90520

/-- Represents the dimensions of a rectangle --/
structure Rectangle where
  length : ℕ
  width : ℕ

/-- Calculates the maximum number of squares that can be cut from a rectangle --/
def maxSquares (rect : Rectangle) : ℕ :=
  sorry

/-- The theorem stating the maximum number of squares for an 11x7 rectangle --/
theorem eleven_by_seven_max_squares :
  maxSquares ⟨11, 7⟩ = 6 := by
  sorry

end NUMINAMATH_CALUDE_eleven_by_seven_max_squares_l905_90520


namespace NUMINAMATH_CALUDE_problem_solution_l905_90537

theorem problem_solution :
  ∀ M : ℝ, (5 + 6 + 7) / 3 = (1988 + 1989 + 1990) / M → M = 994.5 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l905_90537


namespace NUMINAMATH_CALUDE_batch_size_proof_l905_90566

/-- The number of parts in the batch -/
def total_parts : ℕ := 1150

/-- The fraction of work A completes when cooperating with the master -/
def a_work_fraction : ℚ := 1/5

/-- The fraction of work B completes when cooperating with the master -/
def b_work_fraction : ℚ := 2/5

/-- The number of fewer parts B processes when A joins -/
def b_fewer_parts : ℕ := 60

theorem batch_size_proof :
  (b_work_fraction * total_parts : ℚ) - 
  ((1 - a_work_fraction - b_work_fraction) / 
   (1 + (1 - a_work_fraction - b_work_fraction) / a_work_fraction) * total_parts : ℚ) = 
  b_fewer_parts := by sorry

end NUMINAMATH_CALUDE_batch_size_proof_l905_90566


namespace NUMINAMATH_CALUDE_safari_count_l905_90572

/-- The total number of animals counted during the safari --/
def total_animals (antelopes rabbits hyenas wild_dogs leopards : ℕ) : ℕ :=
  antelopes + rabbits + hyenas + wild_dogs + leopards

/-- Theorem stating the total number of animals counted during the safari --/
theorem safari_count : ∃ (antelopes rabbits hyenas wild_dogs leopards : ℕ),
  antelopes = 80 ∧
  rabbits = antelopes + 34 ∧
  hyenas = antelopes + rabbits - 42 ∧
  wild_dogs = hyenas + 50 ∧
  leopards = rabbits / 2 ∧
  total_animals antelopes rabbits hyenas wild_dogs leopards = 605 := by
  sorry


end NUMINAMATH_CALUDE_safari_count_l905_90572


namespace NUMINAMATH_CALUDE_largest_n_for_product_1764_l905_90522

/-- Represents an arithmetic sequence with integer terms -/
structure ArithmeticSequence where
  firstTerm : ℤ
  commonDifference : ℤ

/-- The n-th term of an arithmetic sequence -/
def nthTerm (seq : ArithmeticSequence) (n : ℕ) : ℤ :=
  seq.firstTerm + (n - 1 : ℤ) * seq.commonDifference

theorem largest_n_for_product_1764 (c d : ArithmeticSequence)
    (h1 : c.firstTerm = 1)
    (h2 : d.firstTerm = 1)
    (h3 : nthTerm c 2 ≤ nthTerm d 2)
    (h4 : ∃ n : ℕ, nthTerm c n * nthTerm d n = 1764) :
    (∃ n : ℕ, nthTerm c n * nthTerm d n = 1764) ∧
    (∀ m : ℕ, nthTerm c m * nthTerm d m = 1764 → m ≤ 1764) := by
  sorry

end NUMINAMATH_CALUDE_largest_n_for_product_1764_l905_90522


namespace NUMINAMATH_CALUDE_fraction_problem_l905_90597

theorem fraction_problem (x : ℚ) : 150 * x = 37 + 1/2 → x = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_problem_l905_90597


namespace NUMINAMATH_CALUDE_min_value_triangle_sides_l905_90584

theorem min_value_triangle_sides (x y z : ℝ) 
  (hpos : x > 0 ∧ y > 0 ∧ z > 0) 
  (hsum : x + y + z = 9) 
  (htri : x + y > z ∧ y + z > x ∧ z + x > y) : 
  (x^2 + y^2)/(x + y) + (x^2 + z^2)/(x + z) + (y^2 + z^2)/(y + z) ≥ 9 := by
sorry

end NUMINAMATH_CALUDE_min_value_triangle_sides_l905_90584


namespace NUMINAMATH_CALUDE_f_range_of_a_l905_90507

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then a * 2^(x-1) - 1/a else (a-2)*x + 5/3

theorem f_range_of_a (a : ℝ) :
  (a > 0 ∧ a ≠ 1) →
  (∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → (x₁ - x₂) * (f a x₁ - f a x₂) > 0) →
  a ∈ Set.Ioo 2 3 := by sorry

end NUMINAMATH_CALUDE_f_range_of_a_l905_90507


namespace NUMINAMATH_CALUDE_smallest_Y_value_l905_90502

/-- A function that checks if a positive integer consists only of 0s and 1s -/
def onlyZerosAndOnes (n : ℕ+) : Prop := sorry

/-- The theorem stating the smallest possible value of Y -/
theorem smallest_Y_value (S : ℕ+) (hS : onlyZerosAndOnes S) (hDiv : 18 ∣ S) :
  (S / 18 : ℕ) ≥ 6172839500 :=
sorry

end NUMINAMATH_CALUDE_smallest_Y_value_l905_90502


namespace NUMINAMATH_CALUDE_max_value_quadratic_l905_90536

theorem max_value_quadratic :
  (∃ (r : ℝ), -3 * r^2 + 36 * r - 9 = 99) ∧
  (∀ (r : ℝ), -3 * r^2 + 36 * r - 9 ≤ 99) :=
by sorry

end NUMINAMATH_CALUDE_max_value_quadratic_l905_90536


namespace NUMINAMATH_CALUDE_parallel_iff_plane_intersects_parallel_transitive_l905_90544

-- Define the concept of a line in 3D space
def Line : Type := ℝ × ℝ × ℝ → Prop

-- Define the concept of a plane in 3D space
def Plane : Type := ℝ × ℝ × ℝ → Prop

-- Define parallelism for lines
def parallel (a b : Line) : Prop := sorry

-- Define intersection between a plane and a line
def intersects (p : Plane) (l : Line) : Prop := sorry

-- Define unique intersection
def uniqueIntersection (p : Plane) (l : Line) : Prop := sorry

theorem parallel_iff_plane_intersects (a b : Line) : 
  parallel a b ↔ ∀ (p : Plane), intersects p a → uniqueIntersection p b := by sorry

theorem parallel_transitive (a b c : Line) :
  parallel a b → parallel b c → parallel a c := by sorry

end NUMINAMATH_CALUDE_parallel_iff_plane_intersects_parallel_transitive_l905_90544


namespace NUMINAMATH_CALUDE_equal_products_l905_90531

def numbers : List Nat := [12, 15, 33, 44, 51, 85]
def group1 : List Nat := [12, 33, 85]
def group2 : List Nat := [44, 51, 15]

theorem equal_products :
  (List.prod group1 = List.prod group2) ∧
  (group1.toFinset ∪ group2.toFinset = numbers.toFinset) ∧
  (group1.toFinset ∩ group2.toFinset = ∅) :=
sorry

end NUMINAMATH_CALUDE_equal_products_l905_90531


namespace NUMINAMATH_CALUDE_linear_relationship_scaling_l905_90548

/-- Given a linear relationship between x and y, this theorem proves that
    if an increase of 5 units in x results in an increase of 11 units in y,
    then an increase of 20 units in x will result in an increase of 44 units in y. -/
theorem linear_relationship_scaling (f : ℝ → ℝ) (h : ∀ x, f (x + 5) = f x + 11) :
  ∀ x, f (x + 20) = f x + 44 := by
  sorry

end NUMINAMATH_CALUDE_linear_relationship_scaling_l905_90548


namespace NUMINAMATH_CALUDE_f_even_and_increasing_l905_90580

def f (x : ℝ) : ℝ := x^2 - 1

theorem f_even_and_increasing :
  (∀ x, f (-x) = f x) ∧
  (∀ x y, 0 < x → x < y → f x < f y) :=
by sorry

end NUMINAMATH_CALUDE_f_even_and_increasing_l905_90580


namespace NUMINAMATH_CALUDE_max_students_distribution_l905_90534

theorem max_students_distribution (pens pencils : ℕ) (h1 : pens = 1340) (h2 : pencils = 1280) :
  (∃ (students : ℕ), students > 0 ∧ pens % students = 0 ∧ pencils % students = 0 ∧
    ∀ (n : ℕ), n > students → (pens % n ≠ 0 ∨ pencils % n ≠ 0)) ↔
  (∃ (max_students : ℕ), max_students = Nat.gcd pens pencils) :=
sorry

end NUMINAMATH_CALUDE_max_students_distribution_l905_90534


namespace NUMINAMATH_CALUDE_medical_team_selection_l905_90539

theorem medical_team_selection (m n : ℕ) (hm : m = 6) (hn : n = 5) :
  (m.choose 2) * (n.choose 1) = 75 := by
  sorry

end NUMINAMATH_CALUDE_medical_team_selection_l905_90539


namespace NUMINAMATH_CALUDE_inscribed_circle_theorem_l905_90515

-- Define the triangle and circle
structure Triangle :=
  (A B C : ℝ × ℝ)

structure Circle :=
  (center : ℝ × ℝ)
  (radius : ℝ)

-- Define the inscribed circle property
def isInscribed (t : Triangle) (c : Circle) : Prop := sorry

-- Define the point of tangency
def pointOfTangency (t : Triangle) (c : Circle) : ℝ × ℝ := sorry

-- Define the distance function
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

-- Define the angle between two vectors
def angle (v1 v2 : ℝ × ℝ) : ℝ := sorry

theorem inscribed_circle_theorem (t : Triangle) (c : Circle) (M : ℝ × ℝ) :
  isInscribed t c →
  M = pointOfTangency t c →
  distance t.A M = 1 →
  distance t.B M = 4 →
  angle (t.B - t.A) (t.C - t.A) = 2 * π / 3 →
  distance t.C M = Real.sqrt 273 := by sorry

end NUMINAMATH_CALUDE_inscribed_circle_theorem_l905_90515


namespace NUMINAMATH_CALUDE_dice_probability_l905_90538

/-- The number of sides on each die -/
def num_sides : ℕ := 8

/-- The number of dice rolled -/
def num_dice : ℕ := 8

/-- The number of dice that should show numbers less than 5 -/
def num_less_than_five : ℕ := 4

/-- The probability of rolling a number less than 5 on a single die -/
def prob_less_than_five : ℚ := 1 / 2

/-- The probability of rolling a number 5 or greater on a single die -/
def prob_five_or_greater : ℚ := 1 - prob_less_than_five

theorem dice_probability :
  (Nat.choose num_dice num_less_than_five : ℚ) *
  (prob_less_than_five ^ num_less_than_five) *
  (prob_five_or_greater ^ (num_dice - num_less_than_five)) =
  35 / 128 := by
  sorry

end NUMINAMATH_CALUDE_dice_probability_l905_90538


namespace NUMINAMATH_CALUDE_roy_has_114_pens_l905_90581

/-- The number of pens Roy has -/
structure PenCounts where
  blue : ℕ
  black : ℕ
  red : ℕ
  green : ℕ
  purple : ℕ

/-- Roy's pen collection satisfies the given conditions -/
def satisfiesConditions (p : PenCounts) : Prop :=
  p.blue = 8 ∧
  p.black = 4 * p.blue ∧
  p.red = p.blue + p.black - 5 ∧
  p.green = p.red / 2 ∧
  p.purple = p.blue + p.green - 3

/-- The total number of pens Roy has -/
def totalPens (p : PenCounts) : ℕ :=
  p.blue + p.black + p.red + p.green + p.purple

/-- Theorem: Roy has 114 pens in total -/
theorem roy_has_114_pens :
  ∃ p : PenCounts, satisfiesConditions p ∧ totalPens p = 114 := by
  sorry

end NUMINAMATH_CALUDE_roy_has_114_pens_l905_90581


namespace NUMINAMATH_CALUDE_smallest_integer_with_consecutive_sums_l905_90588

theorem smallest_integer_with_consecutive_sums : ∃ n : ℕ, 
  (∃ a : ℤ, n = (9 * a + 36)) ∧ 
  (∃ b : ℤ, n = (10 * b + 45)) ∧ 
  (∃ c : ℤ, n = (11 * c + 55)) ∧ 
  (∀ m : ℕ, m < n → 
    (¬∃ x : ℤ, m = (9 * x + 36)) ∨ 
    (¬∃ y : ℤ, m = (10 * y + 45)) ∨ 
    (¬∃ z : ℤ, m = (11 * z + 55))) ∧ 
  n = 495 :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_with_consecutive_sums_l905_90588


namespace NUMINAMATH_CALUDE_inequality_proof_l905_90552

theorem inequality_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + y^2016 ≥ 1) :
  x^2016 + y > 1 - 1/100 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l905_90552


namespace NUMINAMATH_CALUDE_opposite_of_negative_2023_l905_90545

-- Define the concept of opposite
def opposite (x : ℤ) : ℤ := -x

-- Theorem statement
theorem opposite_of_negative_2023 : opposite (-2023) = 2023 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_negative_2023_l905_90545


namespace NUMINAMATH_CALUDE_radii_product_l905_90542

/-- Two circles C₁ and C₂ with centers (2, 2) and (-1, -1) respectively, 
    radii r₁ and r₂ (both positive), that are tangent to each other, 
    and have an external common tangent line with a slope of 7. -/
structure TangentCircles where
  r₁ : ℝ
  r₂ : ℝ
  h₁ : r₁ > 0
  h₂ : r₂ > 0
  h_tangent : (r₁ + r₂)^2 = (2 - (-1))^2 + (2 - (-1))^2  -- Distance between centers equals sum of radii
  h_slope : ∃ t : ℝ, (7 * 2 - 2 + t)^2 / 50 = r₁^2 ∧ (7 * (-1) - (-1) + t)^2 / 50 = r₂^2

/-- The product of the radii of two tangent circles with the given properties is 72/25. -/
theorem radii_product (c : TangentCircles) : c.r₁ * c.r₂ = 72 / 25 := by
  sorry

end NUMINAMATH_CALUDE_radii_product_l905_90542


namespace NUMINAMATH_CALUDE_ratio_sum_squares_theorem_l905_90576

theorem ratio_sum_squares_theorem (x y z : ℝ) : 
  y = 2 * x ∧ z = 3 * x ∧ x^2 + y^2 + z^2 = 2744 → x + y + z = 84 := by
  sorry

end NUMINAMATH_CALUDE_ratio_sum_squares_theorem_l905_90576
