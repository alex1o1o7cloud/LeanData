import Mathlib

namespace NUMINAMATH_CALUDE_ellipse_equation_l2525_252505

/-- Given an ellipse with equation x²/a² + y²/b² = 1, where a > b > 0,
    if its right focus is at (3, 0) and the point (0, -3) is on the ellipse,
    then a² = 18 and b² = 9. -/
theorem ellipse_equation (a b : ℝ) (h1 : a > b) (h2 : b > 0) : 
  (∃ (c : ℝ), c^2 = a^2 - b^2 ∧ c = 3) →
  (0^2 / a^2 + (-3)^2 / b^2 = 1) →
  a^2 = 18 ∧ b^2 = 9 := by sorry

end NUMINAMATH_CALUDE_ellipse_equation_l2525_252505


namespace NUMINAMATH_CALUDE_increase_by_percentage_seventy_increased_by_150_percent_l2525_252526

theorem increase_by_percentage (initial : ℝ) (percentage : ℝ) :
  initial * (1 + percentage / 100) = initial + initial * (percentage / 100) := by sorry

theorem seventy_increased_by_150_percent :
  70 * (1 + 150 / 100) = 175 := by sorry

end NUMINAMATH_CALUDE_increase_by_percentage_seventy_increased_by_150_percent_l2525_252526


namespace NUMINAMATH_CALUDE_egg_difference_is_thirteen_l2525_252549

/-- Represents the egg problem with given conditions --/
structure EggProblem where
  total_dozens : Nat
  trays : Nat
  dropped_trays : Nat
  first_tray_broken : Nat
  first_tray_cracked : Nat
  first_tray_slightly_cracked : Nat
  second_tray_shattered : Nat
  second_tray_cracked : Nat
  second_tray_slightly_cracked : Nat

/-- Calculates the difference between perfect eggs in undropped trays and cracked eggs in dropped trays --/
def egg_difference (p : EggProblem) : Nat :=
  let total_eggs := p.total_dozens * 12
  let eggs_per_tray := total_eggs / p.trays
  let undropped_trays := p.trays - p.dropped_trays
  let perfect_eggs := undropped_trays * eggs_per_tray
  let cracked_eggs := p.first_tray_cracked + p.second_tray_cracked
  perfect_eggs - cracked_eggs

/-- Theorem stating the difference is 13 for the given problem conditions --/
theorem egg_difference_is_thirteen : egg_difference {
  total_dozens := 4
  trays := 4
  dropped_trays := 2
  first_tray_broken := 3
  first_tray_cracked := 5
  first_tray_slightly_cracked := 2
  second_tray_shattered := 4
  second_tray_cracked := 6
  second_tray_slightly_cracked := 1
} = 13 := by
  sorry

end NUMINAMATH_CALUDE_egg_difference_is_thirteen_l2525_252549


namespace NUMINAMATH_CALUDE_consecutive_integers_problem_l2525_252567

theorem consecutive_integers_problem (x y z : ℤ) : 
  (y = x - 1) → (z = y - 1) → (x > y) → (y > z) → 
  (2 * x + 3 * y + 3 * z = 5 * y + 11) → (z = 3) → 
  (2 * x = 10) := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_problem_l2525_252567


namespace NUMINAMATH_CALUDE_cow_profit_calculation_l2525_252533

def cow_profit (purchase_price : ℕ) (daily_food_cost : ℕ) (vaccination_cost : ℕ) (days : ℕ) (selling_price : ℕ) : ℕ :=
  selling_price - (purchase_price + daily_food_cost * days + vaccination_cost)

theorem cow_profit_calculation :
  cow_profit 600 20 500 40 2500 = 600 := by
  sorry

end NUMINAMATH_CALUDE_cow_profit_calculation_l2525_252533


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_cubic_equation_l2525_252568

theorem negation_of_existence (f : ℝ → ℝ) :
  (¬ ∃ x, f x = 0) ↔ (∀ x, f x ≠ 0) := by sorry

theorem negation_of_cubic_equation :
  (¬ ∃ x : ℝ, x^3 - 2*x + 1 = 0) ↔ (∀ x : ℝ, x^3 - 2*x + 1 ≠ 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_cubic_equation_l2525_252568


namespace NUMINAMATH_CALUDE_absolute_value_sum_difference_l2525_252516

theorem absolute_value_sum_difference (x y : ℝ) : 
  (|x| = 3 ∧ |y| = 7) →
  ((x > 0 ∧ y < 0 → x + y = -4) ∧
   (x < y → (x - y = -10 ∨ x - y = -4))) := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_sum_difference_l2525_252516


namespace NUMINAMATH_CALUDE_peters_pizza_fraction_l2525_252511

theorem peters_pizza_fraction (total_slices : ℕ) (peter_solo_slices : ℕ) (shared_slices : ℕ) :
  total_slices = 16 →
  peter_solo_slices = 3 →
  shared_slices = 2 →
  (peter_solo_slices + shared_slices / 2 : ℚ) / total_slices = 5 / 16 := by
  sorry

end NUMINAMATH_CALUDE_peters_pizza_fraction_l2525_252511


namespace NUMINAMATH_CALUDE_original_square_area_l2525_252531

theorem original_square_area : ∃ s : ℝ, s > 0 ∧ s^2 = 400 ∧ (s + 5)^2 = s^2 + 225 := by
  sorry

end NUMINAMATH_CALUDE_original_square_area_l2525_252531


namespace NUMINAMATH_CALUDE_group_size_l2525_252538

theorem group_size (T : ℕ) (N : ℕ) (h1 : N > 0) : 
  (T : ℝ) / N - 3 = (T - 44 + 14 : ℝ) / N → N = 10 := by
  sorry

end NUMINAMATH_CALUDE_group_size_l2525_252538


namespace NUMINAMATH_CALUDE_functional_equation_solution_l2525_252547

theorem functional_equation_solution (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, f (x + f y) - f x = (x + f y)^4 - x^4) :
  (∀ x : ℝ, f x = 0) ∨ (∃ k : ℝ, ∀ x : ℝ, f x = x^4 + k) := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l2525_252547


namespace NUMINAMATH_CALUDE_no_m_exists_for_equality_m_range_for_subset_l2525_252513

-- Define the set P
def P : Set ℝ := {x | x^2 - 8*x - 20 ≤ 0}

-- Define the set S parameterized by m
def S (m : ℝ) : Set ℝ := {x | 1 - m ≤ x ∧ x ≤ 1 + m}

-- Theorem 1: There does not exist an m such that P = S(m)
theorem no_m_exists_for_equality : ¬∃ m : ℝ, P = S m := by
  sorry

-- Theorem 2: The set of m such that P ⊆ S(m) is {m | m ≤ 3}
theorem m_range_for_subset : {m : ℝ | P ⊆ S m} = {m : ℝ | m ≤ 3} := by
  sorry

end NUMINAMATH_CALUDE_no_m_exists_for_equality_m_range_for_subset_l2525_252513


namespace NUMINAMATH_CALUDE_midpoint_sum_and_distance_l2525_252546

/-- Given a line segment with endpoints (4, 6) and (10, 18), prove that the sum of the
    coordinates of its midpoint plus the distance from this midpoint to the point (2, 1)
    equals 19 + √146. -/
theorem midpoint_sum_and_distance :
  let x1 : ℝ := 4
  let y1 : ℝ := 6
  let x2 : ℝ := 10
  let y2 : ℝ := 18
  let midx : ℝ := (x1 + x2) / 2
  let midy : ℝ := (y1 + y2) / 2
  let sum_coords : ℝ := midx + midy
  let dist : ℝ := Real.sqrt ((midx - 2)^2 + (midy - 1)^2)
  sum_coords + dist = 19 + Real.sqrt 146 := by
sorry

end NUMINAMATH_CALUDE_midpoint_sum_and_distance_l2525_252546


namespace NUMINAMATH_CALUDE_light_bulb_probability_l2525_252515

/-- The probability of exactly k successes in n independent trials -/
def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (n.choose k) * p^k * (1 - p)^(n - k)

/-- The probability that a light bulb lasts more than 1000 hours -/
def p_success : ℝ := 0.2

/-- The number of light bulbs -/
def n : ℕ := 3

/-- The number of light bulbs that fail -/
def k : ℕ := 1

theorem light_bulb_probability : 
  binomial_probability n k p_success = 0.096 := by
  sorry

end NUMINAMATH_CALUDE_light_bulb_probability_l2525_252515


namespace NUMINAMATH_CALUDE_max_height_sphere_hemispheres_tower_l2525_252559

/-- The maximum height of a tower consisting of a sphere and three hemispheres -/
theorem max_height_sphere_hemispheres_tower (r₀ : ℝ) (h : r₀ = 2017) : 
  ∃ (r₁ r₂ r₃ : ℝ), 
    r₀ ≥ r₁ ∧ r₁ ≥ r₂ ∧ r₂ ≥ r₃ ∧ r₃ > 0 ∧
    r₀ + Real.sqrt (4 * r₀^2) = 3 * r₀ ∧
    3 * r₀ = 6051 :=
by sorry

end NUMINAMATH_CALUDE_max_height_sphere_hemispheres_tower_l2525_252559


namespace NUMINAMATH_CALUDE_overtime_hours_example_l2525_252586

/-- Represents a worker's pay structure and hours worked -/
structure WorkerPay where
  ordinaryRate : ℚ  -- Rate for ordinary hours in dollars
  overtimeRate : ℚ  -- Rate for overtime hours in dollars
  totalHours : ℕ    -- Total hours worked
  totalPay : ℚ      -- Total pay received in dollars

/-- Calculates the number of overtime hours worked -/
def overtimeHours (w : WorkerPay) : ℕ :=
  sorry

/-- Theorem stating that given the specific conditions, the overtime hours are 8 -/
theorem overtime_hours_example :
  let w : WorkerPay := {
    ordinaryRate := 60/100,  -- 60 cents
    overtimeRate := 90/100,  -- 90 cents
    totalHours := 50,
    totalPay := 3240/100     -- $32.40
  }
  overtimeHours w = 8 := by sorry

end NUMINAMATH_CALUDE_overtime_hours_example_l2525_252586


namespace NUMINAMATH_CALUDE_T_2021_2022_2023_even_l2525_252522

def T : ℕ → ℤ
  | 0 => 0
  | 1 => 0
  | 2 => 2
  | n + 3 => T (n + 2) + T (n + 1) + T n

theorem T_2021_2022_2023_even :
  Even (T 2021) ∧ Even (T 2022) ∧ Even (T 2023) := by sorry

end NUMINAMATH_CALUDE_T_2021_2022_2023_even_l2525_252522


namespace NUMINAMATH_CALUDE_quadratic_factorization_l2525_252518

theorem quadratic_factorization (a x : ℝ) : a * x^2 - 4 * a * x + 4 * a = a * (x - 2)^2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l2525_252518


namespace NUMINAMATH_CALUDE_psychology_majors_percentage_l2525_252573

/-- Given a college with the following properties:
  * 40% of total students are freshmen
  * 50% of freshmen are enrolled in the school of liberal arts
  * 10% of total students are freshmen psychology majors in the school of liberal arts
  Prove that 50% of freshmen in the school of liberal arts are psychology majors -/
theorem psychology_majors_percentage 
  (total_students : ℕ) 
  (freshmen_percent : ℚ) 
  (liberal_arts_percent : ℚ) 
  (psych_majors_percent : ℚ) 
  (h1 : freshmen_percent = 40 / 100) 
  (h2 : liberal_arts_percent = 50 / 100) 
  (h3 : psych_majors_percent = 10 / 100) : 
  (psych_majors_percent * total_students) / (freshmen_percent * liberal_arts_percent * total_students) = 50 / 100 := by
  sorry

end NUMINAMATH_CALUDE_psychology_majors_percentage_l2525_252573


namespace NUMINAMATH_CALUDE_five_polyhedra_types_l2525_252543

/-- A topologically correct (simply connected) polyhedron --/
structure Polyhedron where
  vertices : ℕ
  edges : ℕ
  faces : ℕ
  euler : vertices - edges + faces = 2

/-- The possible face types of a polyhedron --/
inductive FaceType
  | Triangle
  | Quadrilateral
  | Pentagon

/-- A function that checks if a polyhedron is valid given its face type --/
def is_valid_polyhedron (p : Polyhedron) (face_type : FaceType) : Prop :=
  match face_type with
  | FaceType.Triangle => p.edges = (3 * p.faces) / 2
  | FaceType.Quadrilateral => p.edges = 2 * p.faces
  | FaceType.Pentagon => p.edges = (5 * p.faces) / 2

/-- The theorem stating that there are exactly 5 types of topologically correct polyhedra --/
theorem five_polyhedra_types :
  ∃! (types : Finset Polyhedron),
    (∀ p ∈ types, ∃ face_type, is_valid_polyhedron p face_type) ∧
    types.card = 5 := by
  sorry

end NUMINAMATH_CALUDE_five_polyhedra_types_l2525_252543


namespace NUMINAMATH_CALUDE_line_translation_down_4_units_l2525_252564

/-- Represents a line in slope-intercept form -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Translates a line vertically -/
def translateLine (l : Line) (units : ℝ) : Line :=
  { slope := l.slope, intercept := l.intercept - units }

theorem line_translation_down_4_units :
  let original_line : Line := { slope := -2, intercept := 3 }
  let translated_line := translateLine original_line 4
  translated_line = { slope := -2, intercept := -1 } := by sorry

end NUMINAMATH_CALUDE_line_translation_down_4_units_l2525_252564


namespace NUMINAMATH_CALUDE_sum_of_solutions_l2525_252529

theorem sum_of_solutions (x : ℝ) : 
  (x^2 + 2023*x = 2025) → 
  (∃ y : ℝ, y^2 + 2023*y = 2025 ∧ x + y = -2023) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_solutions_l2525_252529


namespace NUMINAMATH_CALUDE_inequality_and_equality_condition_l2525_252565

theorem inequality_and_equality_condition (a b c : ℝ) : 
  Real.sqrt (a^2 + a*b + b^2) + Real.sqrt (a^2 + a*c + c^2) ≥ Real.sqrt (3*a^2 + (a + b + c)^2) ∧
  (Real.sqrt (a^2 + a*b + b^2) + Real.sqrt (a^2 + a*c + c^2) = Real.sqrt (3*a^2 + (a + b + c)^2) ↔ 
    (b = c ∨ a = 0) ∧ b*c ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_inequality_and_equality_condition_l2525_252565


namespace NUMINAMATH_CALUDE_bill_face_value_l2525_252520

/-- Calculates the face value of a bill given true discount, time, and interest rate. -/
def face_value (true_discount : ℚ) (time_months : ℚ) (interest_rate : ℚ) : ℚ :=
  (true_discount * 100) / (interest_rate * (time_months / 12))

/-- Theorem stating that given the specified conditions, the face value of the bill is 1575. -/
theorem bill_face_value :
  let true_discount : ℚ := 189
  let time_months : ℚ := 9
  let interest_rate : ℚ := 16
  face_value true_discount time_months interest_rate = 1575 := by
  sorry


end NUMINAMATH_CALUDE_bill_face_value_l2525_252520


namespace NUMINAMATH_CALUDE_tangent_line_at_origin_l2525_252510

noncomputable def f (x : ℝ) : ℝ := 4 / (Real.exp x + 1)

theorem tangent_line_at_origin (x y : ℝ) :
  f 0 = 2 →
  (∀ ε > 0, ∃ δ > 0, ∀ h : ℝ, |h| < δ → |f h - (f 0 + (-1) * h)| ≤ ε * |h|) →
  x + y - 2 = 0 ↔ y = f 0 + (-1) * x := by
sorry

end NUMINAMATH_CALUDE_tangent_line_at_origin_l2525_252510


namespace NUMINAMATH_CALUDE_inequality_not_always_holds_l2525_252553

theorem inequality_not_always_holds (a b : ℝ) (h : a < b) :
  ¬ ∀ m : ℝ, a * m^2 < b * m^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_not_always_holds_l2525_252553


namespace NUMINAMATH_CALUDE_min_trees_chopped_is_270_l2525_252519

def min_trees_chopped (axe_resharpen_interval : ℕ) (saw_regrind_interval : ℕ)
  (axe_sharpen_cost : ℕ) (saw_regrind_cost : ℕ)
  (total_axe_sharpen_cost : ℕ) (total_saw_regrind_cost : ℕ) : ℕ :=
  let axe_sharpenings := (total_axe_sharpen_cost + axe_sharpen_cost - 1) / axe_sharpen_cost
  let saw_regrindings := total_saw_regrind_cost / saw_regrind_cost
  axe_sharpenings * axe_resharpen_interval + saw_regrindings * saw_regrind_interval

theorem min_trees_chopped_is_270 :
  min_trees_chopped 25 20 8 10 46 60 = 270 := by
  sorry

end NUMINAMATH_CALUDE_min_trees_chopped_is_270_l2525_252519


namespace NUMINAMATH_CALUDE_product_equals_four_l2525_252545

theorem product_equals_four (m n : ℝ) (h : m + n = (1/2) * m * n) :
  (m - 2) * (n - 2) = 4 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_four_l2525_252545


namespace NUMINAMATH_CALUDE_tangent_line_at_2_max_value_on_interval_l2525_252583

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 3*x + 1

-- Define the interval
def interval : Set ℝ := Set.Icc (-3) 2

-- Statement for the tangent line
theorem tangent_line_at_2 :
  ∃ (m b : ℝ), ∀ x y, y = f x → (x = 2 → y = m * (x - 2) + f 2) ∧
  (9 * x - y - 15 = 0 ↔ y = m * (x - 2) + f 2) :=
sorry

-- Statement for the maximum value
theorem max_value_on_interval :
  ∃ x ∈ interval, ∀ y ∈ interval, f y ≤ f x ∧ f x = 3 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_at_2_max_value_on_interval_l2525_252583


namespace NUMINAMATH_CALUDE_commercial_length_proof_l2525_252595

theorem commercial_length_proof (x : ℝ) : 
  (3 * x + 11 * 2 = 37) → x = 5 := by
  sorry

end NUMINAMATH_CALUDE_commercial_length_proof_l2525_252595


namespace NUMINAMATH_CALUDE_flowers_per_bouquet_l2525_252552

theorem flowers_per_bouquet 
  (initial_flowers : ℕ) 
  (wilted_flowers : ℕ) 
  (num_bouquets : ℕ) 
  (h1 : initial_flowers = 66) 
  (h2 : wilted_flowers = 10) 
  (h3 : num_bouquets = 7) : 
  (initial_flowers - wilted_flowers) / num_bouquets = 8 := by
  sorry

end NUMINAMATH_CALUDE_flowers_per_bouquet_l2525_252552


namespace NUMINAMATH_CALUDE_product_of_four_expressions_l2525_252584

theorem product_of_four_expressions (A B C D : ℝ) : 
  A = (Real.sqrt 2018 + Real.sqrt 2019 + 1) →
  B = (-Real.sqrt 2018 - Real.sqrt 2019 - 1) →
  C = (Real.sqrt 2018 - Real.sqrt 2019 + 1) →
  D = (Real.sqrt 2019 - Real.sqrt 2018 + 1) →
  A * B * C * D = 9 := by sorry

end NUMINAMATH_CALUDE_product_of_four_expressions_l2525_252584


namespace NUMINAMATH_CALUDE_boat_downstream_distance_l2525_252514

/-- Proves that a boat traveling downstream for 3 hours covers 51 miles, given its upstream speed and combined rate with the river. -/
theorem boat_downstream_distance 
  (upstream_speed : ℝ) 
  (combined_rate : ℝ) 
  (downstream_time : ℝ) 
  (h1 : upstream_speed = 10) 
  (h2 : combined_rate = 17) 
  (h3 : downstream_time = 3) : ℝ :=
by
  sorry

#check boat_downstream_distance

end NUMINAMATH_CALUDE_boat_downstream_distance_l2525_252514


namespace NUMINAMATH_CALUDE_prob_both_truth_l2525_252581

/-- The probability that A speaks the truth -/
def prob_A_truth : ℝ := 0.75

/-- The probability that B speaks the truth -/
def prob_B_truth : ℝ := 0.60

/-- The theorem stating the probability of A and B both telling the truth simultaneously -/
theorem prob_both_truth : prob_A_truth * prob_B_truth = 0.45 := by sorry

end NUMINAMATH_CALUDE_prob_both_truth_l2525_252581


namespace NUMINAMATH_CALUDE_equation_equals_twentyfour_l2525_252597

theorem equation_equals_twentyfour : 8 / (3 - 8 / 3) = 24 := by
  sorry

#check equation_equals_twentyfour

end NUMINAMATH_CALUDE_equation_equals_twentyfour_l2525_252597


namespace NUMINAMATH_CALUDE_prob_fewer_heads_12_fair_coins_l2525_252500

/-- The number of coin flips -/
def n : ℕ := 12

/-- The probability of getting heads on a single fair coin flip -/
def p : ℚ := 1/2

/-- The probability of getting fewer heads than tails in n fair coin flips -/
def prob_fewer_heads (n : ℕ) (p : ℚ) : ℚ :=
  sorry

theorem prob_fewer_heads_12_fair_coins : 
  prob_fewer_heads n p = 793/2048 := by sorry

end NUMINAMATH_CALUDE_prob_fewer_heads_12_fair_coins_l2525_252500


namespace NUMINAMATH_CALUDE_addition_of_like_terms_l2525_252509

theorem addition_of_like_terms (a : ℝ) : a + 2*a = 3*a := by
  sorry

end NUMINAMATH_CALUDE_addition_of_like_terms_l2525_252509


namespace NUMINAMATH_CALUDE_average_L_value_l2525_252569

/-- Represents a coin configuration with H and T sides -/
def Configuration (n : ℕ) := Fin n → Bool

/-- The number of operations before stopping for a given configuration -/
def L (n : ℕ) (c : Configuration n) : ℕ :=
  sorry  -- Definition of L would go here

/-- The average value of L(C) over all 2^n possible initial configurations -/
def averageLValue (n : ℕ) : ℚ :=
  sorry  -- Definition of average L value would go here

/-- Theorem stating that the average L value is n(n+1)/4 -/
theorem average_L_value (n : ℕ) : 
  averageLValue n = ↑n * (↑n + 1) / 4 :=
sorry

end NUMINAMATH_CALUDE_average_L_value_l2525_252569


namespace NUMINAMATH_CALUDE_journey_time_relation_l2525_252588

theorem journey_time_relation (x : ℝ) (h : x > 1.8) : 
  (202 / x) * 1.6 = 202 / (x - 1.8) :=
by
  sorry

#check journey_time_relation

end NUMINAMATH_CALUDE_journey_time_relation_l2525_252588


namespace NUMINAMATH_CALUDE_interest_percentage_of_face_value_l2525_252562

-- Define the bond parameters
def face_value : ℝ := 5000
def selling_price : ℝ := 6153.846153846153
def interest_rate_of_selling_price : ℝ := 0.065

-- Define the theorem
theorem interest_percentage_of_face_value :
  let interest := interest_rate_of_selling_price * selling_price
  let interest_percentage_of_face := (interest / face_value) * 100
  interest_percentage_of_face = 8 := by sorry

end NUMINAMATH_CALUDE_interest_percentage_of_face_value_l2525_252562


namespace NUMINAMATH_CALUDE_student_council_committees_l2525_252540

theorem student_council_committees (n : ℕ) 
  (h : n * (n - 1) / 2 = 15) : 
  Nat.choose n 4 = 15 := by
  sorry

end NUMINAMATH_CALUDE_student_council_committees_l2525_252540


namespace NUMINAMATH_CALUDE_total_results_l2525_252506

theorem total_results (avg : ℚ) (first_12_avg : ℚ) (last_12_avg : ℚ) (result_13 : ℚ) :
  avg = 50 →
  first_12_avg = 14 →
  last_12_avg = 17 →
  result_13 = 878 →
  ∃ N : ℕ, (N : ℚ) * avg = 12 * first_12_avg + 12 * last_12_avg + result_13 ∧ N = 25 :=
by sorry

end NUMINAMATH_CALUDE_total_results_l2525_252506


namespace NUMINAMATH_CALUDE_two_wheeled_bikes_count_l2525_252542

/-- Represents the number of wheels on a bike -/
inductive BikeType
| TwoWheeled
| FourWheeled

/-- Calculates the number of two-wheeled bikes in the shop -/
def count_two_wheeled_bikes (total_wheels : ℕ) (four_wheeled_count : ℕ) : ℕ :=
  let remaining_wheels := total_wheels - (4 * four_wheeled_count)
  remaining_wheels / 2

/-- Theorem stating the number of two-wheeled bikes in the shop -/
theorem two_wheeled_bikes_count :
  count_two_wheeled_bikes 48 9 = 6 := by
  sorry


end NUMINAMATH_CALUDE_two_wheeled_bikes_count_l2525_252542


namespace NUMINAMATH_CALUDE_triangle_inequality_expression_l2525_252598

theorem triangle_inequality_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (htri : a + b > c ∧ b + c > a ∧ c + a > b) :
  (a - b) / (a + b) + (b - c) / (b + c) + (c - a) / (a + c) < 1 / 16 := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_expression_l2525_252598


namespace NUMINAMATH_CALUDE_quadrangle_area_inequality_quadrangle_area_equality_quadrangle_area_equality_converse_l2525_252532

-- Define a quadrangle
structure Quadrangle where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  area : ℝ

-- State the theorem
theorem quadrangle_area_inequality (q : Quadrangle) :
  q.area ≤ (1/2) * (q.a * q.c + q.b * q.d) := by sorry

-- Define a convex orthodiagonal cyclic quadrilateral
structure ConvexOrthoDiagonalCyclicQuad extends Quadrangle where
  is_convex : Bool
  is_orthodiagonal : Bool
  is_cyclic : Bool

-- State the equality condition
theorem quadrangle_area_equality (q : ConvexOrthoDiagonalCyclicQuad) :
  q.is_convex = true → q.is_orthodiagonal = true → q.is_cyclic = true →
  q.area = (1/2) * (q.a * q.c + q.b * q.d) := by sorry

-- State the converse of the equality condition
theorem quadrangle_area_equality_converse (q : Quadrangle) :
  q.area = (1/2) * (q.a * q.c + q.b * q.d) →
  ∃ (cq : ConvexOrthoDiagonalCyclicQuad),
    cq.a = q.a ∧ cq.b = q.b ∧ cq.c = q.c ∧ cq.d = q.d ∧
    cq.area = q.area ∧
    cq.is_convex = true ∧ cq.is_orthodiagonal = true ∧ cq.is_cyclic = true := by sorry

end NUMINAMATH_CALUDE_quadrangle_area_inequality_quadrangle_area_equality_quadrangle_area_equality_converse_l2525_252532


namespace NUMINAMATH_CALUDE_probability_at_least_one_juice_l2525_252544

def total_bottles : ℕ := 5
def juice_bottles : ℕ := 2
def selected_bottles : ℕ := 2

theorem probability_at_least_one_juice :
  let non_juice_bottles := total_bottles - juice_bottles
  let total_combinations := (total_bottles.choose selected_bottles : ℚ)
  let non_juice_combinations := (non_juice_bottles.choose selected_bottles : ℚ)
  (1 - non_juice_combinations / total_combinations) = 7 / 10 := by sorry

end NUMINAMATH_CALUDE_probability_at_least_one_juice_l2525_252544


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l2525_252534

/-- An arithmetic sequence is a sequence where the difference between
    any two consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_common_difference 
  (a : ℕ → ℚ) 
  (h_arith : is_arithmetic_sequence a) 
  (h_sum : a 4 + a 8 = 10) 
  (h_term : a 10 = 6) : 
  ∃ d : ℚ, d = 1/4 ∧ ∀ n : ℕ, a (n + 1) = a n + d :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l2525_252534


namespace NUMINAMATH_CALUDE_ellipse_minor_axis_length_l2525_252504

/-- The length of the minor axis of the ellipse 9x^2 + y^2 = 36 is 4 -/
theorem ellipse_minor_axis_length :
  let ellipse := {(x, y) : ℝ × ℝ | 9 * x^2 + y^2 = 36}
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧
    ellipse = {(x, y) : ℝ × ℝ | (x^2 / a^2) + (y^2 / b^2) = 1} ∧
    2 * min a b = 4 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_minor_axis_length_l2525_252504


namespace NUMINAMATH_CALUDE_seashells_total_l2525_252577

theorem seashells_total (sally_shells tom_shells jessica_shells : ℕ) 
  (h1 : sally_shells = 9)
  (h2 : tom_shells = 7)
  (h3 : jessica_shells = 5) :
  sally_shells + tom_shells + jessica_shells = 21 := by
  sorry

end NUMINAMATH_CALUDE_seashells_total_l2525_252577


namespace NUMINAMATH_CALUDE_solution_l2525_252580

/-- The set of points satisfying the given equation -/
def S : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | 2 * p.1^2 + p.2^2 + 3 * p.1 * p.2 + 3 * p.1 + p.2 = 2}

/-- The first line -/
def L₁ : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = -p.1 - 2}

/-- The second line -/
def L₂ : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = -2 * p.1 + 1}

/-- The union of the two lines -/
def U : Set (ℝ × ℝ) :=
  L₁ ∪ L₂

theorem solution : S = U := by
  sorry

end NUMINAMATH_CALUDE_solution_l2525_252580


namespace NUMINAMATH_CALUDE_inverse_variation_problem_l2525_252537

theorem inverse_variation_problem (x y : ℝ) (k : ℝ) :
  (∀ x y, 5 * y + 3 = k / (x^3)) →
  (5 * 8 + 3 = k / (2^3)) →
  (5 * y + 3 = k / (4^3)) →
  y = 19 / 40 := by
sorry

end NUMINAMATH_CALUDE_inverse_variation_problem_l2525_252537


namespace NUMINAMATH_CALUDE_area_to_paint_is_15_l2525_252599

/-- The area of the wall to be painted -/
def area_to_paint (wall_length wall_width blackboard_length blackboard_width : ℝ) : ℝ :=
  wall_length * wall_width - blackboard_length * blackboard_width

/-- Theorem: The area to be painted is 15 square meters -/
theorem area_to_paint_is_15 :
  area_to_paint 6 3 3 1 = 15 := by
  sorry

end NUMINAMATH_CALUDE_area_to_paint_is_15_l2525_252599


namespace NUMINAMATH_CALUDE_commercial_fraction_l2525_252594

theorem commercial_fraction (num_programs : ℕ) (program_duration : ℕ) (commercial_time : ℕ) :
  num_programs = 6 →
  program_duration = 30 →
  commercial_time = 45 →
  (commercial_time : ℚ) / (num_programs * program_duration : ℚ) = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_commercial_fraction_l2525_252594


namespace NUMINAMATH_CALUDE_reciprocal_sum_equals_two_l2525_252527

theorem reciprocal_sum_equals_two (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h_sum : x + y = 2) (h_prod : x * y = 1) : 
  1 / x + 1 / y = 2 := by
sorry

end NUMINAMATH_CALUDE_reciprocal_sum_equals_two_l2525_252527


namespace NUMINAMATH_CALUDE_find_r_l2525_252590

-- Define the polynomials f and g
def f (r a x : ℝ) : ℝ := (x - (r + 2)) * (x - (r + 6)) * (x - a)
def g (r b x : ℝ) : ℝ := (x - (r + 4)) * (x - (r + 8)) * (x - b)

-- State the theorem
theorem find_r : ∃ (r a b : ℝ), 
  (∀ x, f r a x - g r b x = 2 * r) → r = 48 / 17 := by
  sorry

end NUMINAMATH_CALUDE_find_r_l2525_252590


namespace NUMINAMATH_CALUDE_complex_sum_reciprocal_squared_l2525_252548

def complex_number (a b : ℝ) : ℂ := a + b * Complex.I

theorem complex_sum_reciprocal_squared (x : ℂ) :
  x + (1 / x) = 5 → x^2 + (1 / x)^2 = (7 : ℝ) / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_reciprocal_squared_l2525_252548


namespace NUMINAMATH_CALUDE_perfect_square_condition_l2525_252536

/-- A polynomial of the form x^2 + bx + c is a perfect square trinomial if and only if
    there exists a real number m such that b = 2m and c = m^2 -/
def is_perfect_square_trinomial (b c : ℝ) : Prop :=
  ∃ m : ℝ, b = 2 * m ∧ c = m^2

/-- The main theorem: x^2 + (a-1)x + 9 is a perfect square trinomial iff a = 7 or a = -5 -/
theorem perfect_square_condition (a : ℝ) :
  is_perfect_square_trinomial (a - 1) 9 ↔ a = 7 ∨ a = -5 := by
  sorry


end NUMINAMATH_CALUDE_perfect_square_condition_l2525_252536


namespace NUMINAMATH_CALUDE_sum_of_A_and_B_l2525_252578

-- Define the functions f and g
def f (A B x : ℝ) : ℝ := A * x + B
def g (A B x : ℝ) : ℝ := B * x + A

-- State the theorem
theorem sum_of_A_and_B 
  (A B : ℝ) 
  (h1 : A ≠ B) 
  (h2 : B - A = 2) 
  (h3 : ∀ x, f A B (g A B x) - g A B (f A B x) = B^2 - A^2) : 
  A + B = -1/2 := by
sorry

end NUMINAMATH_CALUDE_sum_of_A_and_B_l2525_252578


namespace NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l2525_252572

-- Problem 1
theorem simplify_expression_1 : 
  (3 * Real.sqrt 8 - 12 * Real.sqrt (1/2) + Real.sqrt 18) * 2 * Real.sqrt 3 = 6 * Real.sqrt 6 := by
  sorry

-- Problem 2
theorem simplify_expression_2 (x : ℝ) (hx : x > 0) : 
  (6 * Real.sqrt (x/4) - 2*x * Real.sqrt (1/x)) / (3 * Real.sqrt x) = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l2525_252572


namespace NUMINAMATH_CALUDE_sector_central_angle_central_angle_is_two_l2525_252507

theorem sector_central_angle (arc_length : ℝ) (area : ℝ) (h1 : arc_length = 2) (h2 : area = 1) :
  ∃ (r : ℝ), r > 0 ∧ area = 1/2 * r * arc_length ∧ arc_length = 2 * r := by
  sorry

theorem central_angle_is_two (arc_length : ℝ) (area : ℝ) (h1 : arc_length = 2) (h2 : area = 1) :
  let r := (2 * area) / arc_length
  arc_length / r = 2 := by
  sorry

end NUMINAMATH_CALUDE_sector_central_angle_central_angle_is_two_l2525_252507


namespace NUMINAMATH_CALUDE_farm_fencing_cost_l2525_252530

/-- Calculates the cost of fencing a rectangular farm -/
theorem farm_fencing_cost 
  (area : ℝ) 
  (short_side : ℝ) 
  (cost_per_meter : ℝ) 
  (h_area : area = 1200) 
  (h_short : short_side = 30) 
  (h_cost : cost_per_meter = 14) : 
  let long_side := area / short_side
  let diagonal := Real.sqrt (long_side^2 + short_side^2)
  let total_length := long_side + short_side + diagonal
  cost_per_meter * total_length = 1680 :=
by sorry

end NUMINAMATH_CALUDE_farm_fencing_cost_l2525_252530


namespace NUMINAMATH_CALUDE_oranges_left_l2525_252596

def initial_oranges : ℕ := 96
def taken_oranges : ℕ := 45

theorem oranges_left : initial_oranges - taken_oranges = 51 := by
  sorry

end NUMINAMATH_CALUDE_oranges_left_l2525_252596


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2525_252561

theorem sufficient_not_necessary_condition (a b : ℝ) :
  (∀ a b : ℝ, (a - b) * a^2 < 0 → a < b) ∧
  (∃ a b : ℝ, a < b ∧ (a - b) * a^2 ≥ 0) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2525_252561


namespace NUMINAMATH_CALUDE_max_gcd_consecutive_b_terms_is_one_l2525_252571

/-- The sequence b_n defined as n! + n^2 + 1 -/
def b (n : ℕ) : ℕ := n.factorial + n^2 + 1

/-- The theorem stating that the maximum GCD of consecutive terms in the sequence is 1 -/
theorem max_gcd_consecutive_b_terms_is_one :
  ∀ n : ℕ, ∃ m : ℕ, m ≥ n → (∀ k ≥ m, Nat.gcd (b k) (b (k + 1)) = 1) ∧
    (∀ i j : ℕ, i ≥ n → j = i + 1 → Nat.gcd (b i) (b j) ≤ 1) := by
  sorry

end NUMINAMATH_CALUDE_max_gcd_consecutive_b_terms_is_one_l2525_252571


namespace NUMINAMATH_CALUDE_right_triangle_existence_l2525_252539

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := x^3 - 3*x + 2 + m

theorem right_triangle_existence (m : ℝ) :
  m > 0 →
  (∃ a b c : ℝ, 0 ≤ a ∧ a ≤ 2 ∧ 0 ≤ b ∧ b ≤ 2 ∧ 0 ≤ c ∧ c ≤ 2 ∧
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    (f m a)^2 + (f m b)^2 = (f m c)^2 ∨
    (f m a)^2 + (f m c)^2 = (f m b)^2 ∨
    (f m b)^2 + (f m c)^2 = (f m a)^2) ↔
  0 < m ∧ m < 4 + 4 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_right_triangle_existence_l2525_252539


namespace NUMINAMATH_CALUDE_cube_inequality_l2525_252501

theorem cube_inequality (x y a : ℝ) (h1 : 0 < a) (h2 : a < 1) (h3 : a^x < a^y) : x^3 > y^3 := by
  sorry

end NUMINAMATH_CALUDE_cube_inequality_l2525_252501


namespace NUMINAMATH_CALUDE_farm_output_growth_equation_l2525_252523

/-- Represents the relationship between initial value, final value, and growth rate over two years -/
theorem farm_output_growth_equation (initial_value final_value : ℝ) (growth_rate : ℝ) : 
  initial_value = 80 → final_value = 96.8 → 
  initial_value * (1 + growth_rate)^2 = final_value :=
by
  sorry

#check farm_output_growth_equation

end NUMINAMATH_CALUDE_farm_output_growth_equation_l2525_252523


namespace NUMINAMATH_CALUDE_largest_number_from_hcf_lcm_factors_l2525_252503

theorem largest_number_from_hcf_lcm_factors (a b : ℕ+) 
  (hcf_ab : Nat.gcd a b = 50)
  (lcm_factor1 : ∃ k : ℕ+, Nat.lcm a b = 50 * 11 * k)
  (lcm_factor2 : ∃ k : ℕ+, Nat.lcm a b = 50 * 12 * k) :
  max a b = 600 := by
sorry

end NUMINAMATH_CALUDE_largest_number_from_hcf_lcm_factors_l2525_252503


namespace NUMINAMATH_CALUDE_initial_bacteria_count_l2525_252556

/-- The number of seconds in the experiment -/
def experiment_duration : ℕ := 240

/-- The number of seconds it takes for the bacteria population to double -/
def doubling_time : ℕ := 30

/-- The number of bacteria after the experiment duration -/
def final_population : ℕ := 524288

/-- The number of times the population doubles during the experiment -/
def doubling_count : ℕ := experiment_duration / doubling_time

theorem initial_bacteria_count :
  ∃ (initial_count : ℕ), initial_count * (2 ^ doubling_count) = final_population ∧ initial_count = 2048 :=
sorry

end NUMINAMATH_CALUDE_initial_bacteria_count_l2525_252556


namespace NUMINAMATH_CALUDE_china_gdp_scientific_notation_l2525_252575

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem china_gdp_scientific_notation :
  toScientificNotation 86000 = ScientificNotation.mk 8.6 4 sorry := by
  sorry

end NUMINAMATH_CALUDE_china_gdp_scientific_notation_l2525_252575


namespace NUMINAMATH_CALUDE_last_digit_product_divisible_by_six_l2525_252524

/-- The last digit of a natural number -/
def lastDigit (n : ℕ) : ℕ := n % 10

/-- The remaining digits of a natural number -/
def remainingDigits (n : ℕ) : ℕ := n / 10

/-- Theorem: For all n > 3, the product of the last digit of 2^n and the remaining digits is divisible by 6 -/
theorem last_digit_product_divisible_by_six (n : ℕ) (h : n > 3) :
  ∃ k : ℕ, (lastDigit (2^n) * remainingDigits (2^n)) = 6 * k := by
  sorry

end NUMINAMATH_CALUDE_last_digit_product_divisible_by_six_l2525_252524


namespace NUMINAMATH_CALUDE_jogging_time_proportional_to_distance_l2525_252551

/-- Given a constant jogging speed, prove that if it takes 30 minutes to jog 4 miles,
    then it will take 15 minutes to jog 2 miles. -/
theorem jogging_time_proportional_to_distance
  (speed : ℝ) -- Constant jogging speed
  (h1 : speed > 0) -- Assumption that speed is positive
  (h2 : 4 / speed = 30) -- It takes 30 minutes to jog 4 miles
  : 2 / speed = 15 := by
  sorry

end NUMINAMATH_CALUDE_jogging_time_proportional_to_distance_l2525_252551


namespace NUMINAMATH_CALUDE_complex_product_theorem_l2525_252528

theorem complex_product_theorem : 
  let z : ℂ := Complex.exp (Complex.I * (3 * Real.pi / 11))
  (3 * z + z^3) * (3 * z^3 + z^9) * (3 * z^5 + z^15) * 
  (3 * z^7 + z^21) * (3 * z^9 + z^27) * (3 * z^11 + z^33) = 2197 := by
  sorry

end NUMINAMATH_CALUDE_complex_product_theorem_l2525_252528


namespace NUMINAMATH_CALUDE_product_of_fractions_l2525_252579

theorem product_of_fractions : (2 : ℚ) / 9 * 5 / 11 = 10 / 99 := by
  sorry

end NUMINAMATH_CALUDE_product_of_fractions_l2525_252579


namespace NUMINAMATH_CALUDE_inequality_system_solution_l2525_252574

theorem inequality_system_solution (p : ℝ) : 19 * p < 10 ∧ p > (1/2 : ℝ) → (1/2 : ℝ) < p ∧ p < 10/19 := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l2525_252574


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_sets_l2525_252585

theorem quadratic_inequality_solution_sets (a₁ b₁ c₁ a₂ b₂ c₂ : ℝ) 
  (h₁ : a₁ ≠ 0) (h₂ : b₁ ≠ 0) (h₃ : c₁ ≠ 0) 
  (h₄ : a₂ ≠ 0) (h₅ : b₂ ≠ 0) (h₆ : c₂ ≠ 0) :
  ¬(((a₁ / a₂ = b₁ / b₂) ∧ (b₁ / b₂ = c₁ / c₂)) ↔
    ({x : ℝ | a₁ * x^2 + b₁ * x + c₁ > 0} = {x : ℝ | a₂ * x^2 + b₂ * x + c₂ > 0})) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_sets_l2525_252585


namespace NUMINAMATH_CALUDE_quadratic_equation_identification_l2525_252521

/-- Definition of a quadratic equation in one variable -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The equation from option A -/
def eq_A (x : ℝ) : ℝ := 3 * x + 1

/-- The equation from option B -/
def eq_B (x : ℝ) : ℝ := x^2 - (2 * x - 3 * x^2)

/-- The equation from option C -/
def eq_C (x y : ℝ) : ℝ := x^2 - y + 5

/-- The equation from option D -/
def eq_D (x y : ℝ) : ℝ := x - x * y - 1 - x^2

theorem quadratic_equation_identification :
  ¬ is_quadratic_equation eq_A ∧
  is_quadratic_equation eq_B ∧
  ¬ (∃ f : ℝ → ℝ, ∀ x y, eq_C x y = f x) ∧
  ¬ (∃ f : ℝ → ℝ, ∀ x y, eq_D x y = f x) :=
sorry

end NUMINAMATH_CALUDE_quadratic_equation_identification_l2525_252521


namespace NUMINAMATH_CALUDE_min_g_14_l2525_252541

-- Define a tenuous function
def Tenuous (f : ℕ+ → ℕ) : Prop :=
  ∀ x y : ℕ+, f x + f y > y^2

-- Define the sum of g from 1 to 20
def SumG (g : ℕ+ → ℕ) : ℕ :=
  (Finset.range 20).sum (fun i => g ⟨i + 1, Nat.succ_pos i⟩)

-- Theorem statement
theorem min_g_14 (g : ℕ+ → ℕ) (h_tenuous : Tenuous g) (h_min : ∀ g' : ℕ+ → ℕ, Tenuous g' → SumG g ≤ SumG g') :
  g ⟨14, by norm_num⟩ ≥ 136 := by
  sorry

end NUMINAMATH_CALUDE_min_g_14_l2525_252541


namespace NUMINAMATH_CALUDE_same_root_value_l2525_252566

theorem same_root_value (a b c d : ℝ) (h : a ≠ c) :
  ∀ α : ℝ, (α^2 + a*α + b = 0 ∧ α^2 + c*α + d = 0) → α = (d - b) / (a - c) := by
  sorry

end NUMINAMATH_CALUDE_same_root_value_l2525_252566


namespace NUMINAMATH_CALUDE_stating_hotel_booking_problem_l2525_252576

/-- Represents the number of double rooms booked in a hotel. -/
def double_rooms : ℕ := 196

/-- Represents the number of single rooms booked in a hotel. -/
def single_rooms : ℕ := 260 - double_rooms

/-- The cost of a single room in dollars. -/
def single_room_cost : ℕ := 35

/-- The cost of a double room in dollars. -/
def double_room_cost : ℕ := 60

/-- The total revenue from all booked rooms in dollars. -/
def total_revenue : ℕ := 14000

/-- 
Theorem stating that given the conditions of the hotel booking problem,
the number of double rooms booked is 196.
-/
theorem hotel_booking_problem :
  (single_rooms + double_rooms = 260) ∧
  (single_room_cost * single_rooms + double_room_cost * double_rooms = total_revenue) →
  double_rooms = 196 :=
by sorry

end NUMINAMATH_CALUDE_stating_hotel_booking_problem_l2525_252576


namespace NUMINAMATH_CALUDE_partnership_investment_l2525_252535

theorem partnership_investment (b c total_profit a_profit : ℕ) 
  (hb : b = 4200)
  (hc : c = 10500)
  (htotal : total_profit = 14200)
  (ha_profit : a_profit = 4260) :
  ∃ a : ℕ, a = 6600 ∧ a_profit / total_profit = a / (a + b + c) :=
sorry

end NUMINAMATH_CALUDE_partnership_investment_l2525_252535


namespace NUMINAMATH_CALUDE_parallelogram_roots_l2525_252554

theorem parallelogram_roots (a : ℝ) : 
  (∃ z₁ z₂ z₃ z₄ : ℂ, 
    z₁^4 - 8*z₁^3 + 13*a*z₁^2 - 2*(3*a^2 + 2*a - 4)*z₁ - 2 = 0 ∧
    z₂^4 - 8*z₂^3 + 13*a*z₂^2 - 2*(3*a^2 + 2*a - 4)*z₂ - 2 = 0 ∧
    z₃^4 - 8*z₃^3 + 13*a*z₃^2 - 2*(3*a^2 + 2*a - 4)*z₃ - 2 = 0 ∧
    z₄^4 - 8*z₄^3 + 13*a*z₄^2 - 2*(3*a^2 + 2*a - 4)*z₄ - 2 = 0 ∧
    (z₁ + z₃ = z₂ + z₄) ∧ (z₁ - z₂ = z₄ - z₃)) ↔
  a^2 + (2/3)*a - 49*(1/3) = 0 :=
sorry

end NUMINAMATH_CALUDE_parallelogram_roots_l2525_252554


namespace NUMINAMATH_CALUDE_unique_root_continuous_monotonic_l2525_252591

theorem unique_root_continuous_monotonic {α : Type*} [LinearOrder α] [TopologicalSpace α] {f : α → ℝ} {a b : α} (h_cont : Continuous f) (h_mono : Monotone f) (h_sign : f a * f b < 0) : ∃! x, a ≤ x ∧ x ≤ b ∧ f x = 0 :=
sorry

end NUMINAMATH_CALUDE_unique_root_continuous_monotonic_l2525_252591


namespace NUMINAMATH_CALUDE_quadratic_solution_l2525_252570

theorem quadratic_solution (m : ℝ) : (2 : ℝ)^2 + m * 2 + 2 = 0 → m = -3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_l2525_252570


namespace NUMINAMATH_CALUDE_repeated_number_divisible_by_91_l2525_252563

/-- Represents a three-digit number -/
structure ThreeDigitNumber where
  hundreds : Nat
  tens : Nat
  ones : Nat
  hundred_nonzero : hundreds ≠ 0
  digit_bounds : hundreds < 10 ∧ tens < 10 ∧ ones < 10

/-- Converts a ThreeDigitNumber to its numeric value -/
def to_nat (n : ThreeDigitNumber) : Nat :=
  100 * n.hundreds + 10 * n.tens + n.ones

/-- Represents the six-digit number formed by repeating a three-digit number -/
def repeated_number (n : ThreeDigitNumber) : Nat :=
  1000000 * n.hundreds + 100000 * n.tens + 10000 * n.ones +
  1000 * n.hundreds + 100 * n.tens + 10 * n.ones

/-- Theorem stating that the repeated number is divisible by 91 -/
theorem repeated_number_divisible_by_91 (n : ThreeDigitNumber) :
  (repeated_number n) % 91 = 0 := by
  sorry

end NUMINAMATH_CALUDE_repeated_number_divisible_by_91_l2525_252563


namespace NUMINAMATH_CALUDE_exists_large_number_with_invariant_prime_factors_l2525_252517

/-- A function that represents swapping two non-zero digits in a number's decimal representation -/
def swap_digits (n : ℕ) (i j : ℕ) : ℕ := sorry

/-- A function that returns the set of prime factors of a number -/
def prime_factors (n : ℕ) : Set ℕ := sorry

/-- Theorem stating the existence of a number with the required properties -/
theorem exists_large_number_with_invariant_prime_factors :
  ∃ n : ℕ, n > 10^1000 ∧ 
           n % 10 ≠ 0 ∧ 
           ∃ i j : ℕ, i ≠ j ∧ 
                     (swap_digits n i j) ≠ n ∧ 
                     prime_factors (swap_digits n i j) = prime_factors n :=
sorry

end NUMINAMATH_CALUDE_exists_large_number_with_invariant_prime_factors_l2525_252517


namespace NUMINAMATH_CALUDE_power_comparison_specific_power_comparison_l2525_252593

theorem power_comparison (n : ℕ) (hn : n > 0) :
  (n ≤ 2 → n^(n+1) < (n+1)^n) ∧
  (n ≥ 3 → n^(n+1) > (n+1)^n) :=
sorry

theorem specific_power_comparison :
  2008^2009 > 2009^2008 :=
sorry

end NUMINAMATH_CALUDE_power_comparison_specific_power_comparison_l2525_252593


namespace NUMINAMATH_CALUDE_rectangular_field_area_difference_l2525_252550

theorem rectangular_field_area_difference : 
  let stan_length : ℕ := 30
  let stan_width : ℕ := 50
  let isla_length : ℕ := 35
  let isla_width : ℕ := 55
  let stan_area := stan_length * stan_width
  let isla_area := isla_length * isla_width
  isla_area - stan_area = 425 ∧ isla_area > stan_area := by
sorry

end NUMINAMATH_CALUDE_rectangular_field_area_difference_l2525_252550


namespace NUMINAMATH_CALUDE_ticket_price_difference_l2525_252592

/-- Represents the total amount paid for pre-booked tickets -/
def prebooked_total : ℕ := 10 * 140 + 10 * 170

/-- Represents the total amount paid for tickets bought at the gate -/
def gate_total : ℕ := 8 * 190 + 12 * 210 + 10 * 300

/-- Theorem stating the difference in total amount paid -/
theorem ticket_price_difference : gate_total - prebooked_total = 3940 := by
  sorry

end NUMINAMATH_CALUDE_ticket_price_difference_l2525_252592


namespace NUMINAMATH_CALUDE_f_continuous_iff_b_eq_12_l2525_252582

-- Define the piecewise function f(x)
noncomputable def f (b : ℝ) (x : ℝ) : ℝ :=
  if x > -2 then 3 * x + b else -x + 4

-- Theorem statement
theorem f_continuous_iff_b_eq_12 (b : ℝ) :
  Continuous (f b) ↔ b = 12 := by
  sorry

end NUMINAMATH_CALUDE_f_continuous_iff_b_eq_12_l2525_252582


namespace NUMINAMATH_CALUDE_probability_to_reach_target_is_correct_l2525_252558

/-- Represents a point in the 2D coordinate plane -/
structure Point where
  x : ℤ
  y : ℤ

/-- Represents a step direction -/
inductive Direction
  | Left
  | Right
  | Up
  | Down

/-- The probability of taking a specific step -/
def stepProbability : ℚ := 1/4

/-- The starting point -/
def startPoint : Point := ⟨0, 0⟩

/-- The target point -/
def targetPoint : Point := ⟨3, 3⟩

/-- The maximum number of steps allowed -/
def maxSteps : ℕ := 8

/-- Calculate the probability of reaching the target point from the start point
    in at most the maximum number of steps -/
def probabilityToReachTarget : ℚ :=
  45/2048

theorem probability_to_reach_target_is_correct :
  probabilityToReachTarget = 45/2048 := by
  sorry

end NUMINAMATH_CALUDE_probability_to_reach_target_is_correct_l2525_252558


namespace NUMINAMATH_CALUDE_smallest_palindrome_l2525_252512

/-- A number is a palindrome in a given base if it reads the same forwards and backwards when represented in that base. -/
def isPalindrome (n : ℕ) (base : ℕ) : Prop := sorry

/-- Converts a natural number to its representation in a given base. -/
def toBase (n : ℕ) (base : ℕ) : List ℕ := sorry

/-- The number of digits in the representation of a natural number in a given base. -/
def numDigits (n : ℕ) (base : ℕ) : ℕ := sorry

/-- 10101₂ in decimal -/
def target : ℕ := 21

theorem smallest_palindrome :
  ∀ n : ℕ,
  (numDigits n 2 = 5 ∧ isPalindrome n 2) →
  (∃ base : ℕ, base > 4 ∧ numDigits n base = 3 ∧ isPalindrome n base) →
  n ≥ target :=
sorry

end NUMINAMATH_CALUDE_smallest_palindrome_l2525_252512


namespace NUMINAMATH_CALUDE_gcf_of_45_135_90_l2525_252560

theorem gcf_of_45_135_90 : Nat.gcd 45 (Nat.gcd 135 90) = 45 := by sorry

end NUMINAMATH_CALUDE_gcf_of_45_135_90_l2525_252560


namespace NUMINAMATH_CALUDE_pentadecagon_diagonals_l2525_252589

/-- Number of diagonals in a convex polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A pentadecagon is a 15-sided polygon -/
def pentadecagon_sides : ℕ := 15

theorem pentadecagon_diagonals :
  num_diagonals pentadecagon_sides = 90 := by
  sorry

end NUMINAMATH_CALUDE_pentadecagon_diagonals_l2525_252589


namespace NUMINAMATH_CALUDE_inscribed_cube_surface_area_l2525_252557

/-- Given a cube with a sphere inscribed within it, and another cube inscribed within that sphere,
    this theorem relates the surface area of the outer cube to the surface area of the inner cube. -/
theorem inscribed_cube_surface_area
  (outer_cube_surface_area : ℝ)
  (h_outer_surface : outer_cube_surface_area = 54)
  : ∃ (inner_cube_surface_area : ℝ),
    inner_cube_surface_area = 18 :=
by sorry

end NUMINAMATH_CALUDE_inscribed_cube_surface_area_l2525_252557


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_number_l2525_252525

theorem imaginary_part_of_complex_number :
  let z : ℂ := 1 - 2*I
  Complex.im z = -2 := by
sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_number_l2525_252525


namespace NUMINAMATH_CALUDE_min_value_expression_l2525_252508

theorem min_value_expression (a b c : ℝ) (h1 : a > b) (h2 : b > c) (h3 : c > 0) :
  2 * a^2 + 1 / (a * b) + 1 / (a * (a - b)) - 10 * a * c + 25 * c^2 ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l2525_252508


namespace NUMINAMATH_CALUDE_solution_set_part1_range_of_a_part2_l2525_252587

-- Define the function f
def f (x a : ℝ) : ℝ := |x - a| + |x - 2|

-- Part 1
theorem solution_set_part1 :
  {x : ℝ | f x 1 ≥ 3} = {x : ℝ | x ≤ 0 ∨ x ≥ 3} :=
sorry

-- Part 2
theorem range_of_a_part2 :
  {a : ℝ | ∀ x, f x a ≥ 2*a - 1} = {a : ℝ | a ≤ 1} :=
sorry

end NUMINAMATH_CALUDE_solution_set_part1_range_of_a_part2_l2525_252587


namespace NUMINAMATH_CALUDE_calculate_expression_l2525_252555

theorem calculate_expression : (Real.pi - Real.sqrt 3) ^ 0 - 2 * Real.sin (π / 4) + |-Real.sqrt 2| + Real.sqrt 8 = 1 + 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l2525_252555


namespace NUMINAMATH_CALUDE_bears_on_shelves_l2525_252502

/-- Given an initial stock of bears, a new shipment, and a number of bears per shelf,
    calculate the number of shelves required to store all bears. -/
def shelves_required (initial_stock : ℕ) (new_shipment : ℕ) (bears_per_shelf : ℕ) : ℕ :=
  (initial_stock + new_shipment) / bears_per_shelf

/-- Theorem stating that with 17 initial bears, 10 new bears, and 9 bears per shelf,
    3 shelves are required. -/
theorem bears_on_shelves :
  shelves_required 17 10 9 = 3 := by
  sorry

end NUMINAMATH_CALUDE_bears_on_shelves_l2525_252502
