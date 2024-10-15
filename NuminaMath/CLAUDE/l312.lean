import Mathlib

namespace NUMINAMATH_CALUDE_value_of_M_l312_31208

theorem value_of_M : ∃ M : ℝ, (0.2 * M = 0.6 * 1500) ∧ (M = 4500) := by
  sorry

end NUMINAMATH_CALUDE_value_of_M_l312_31208


namespace NUMINAMATH_CALUDE_correction_amount_proof_l312_31298

/-- Represents the value of a coin in cents -/
def coin_value (coin : String) : ℕ :=
  match coin with
  | "quarter" => 25
  | "dime" => 10
  | "nickel" => 5
  | "penny" => 1
  | _ => 0

/-- Calculates the correction amount in cents given the counting errors -/
def correction_amount (y z w : ℕ) : ℤ :=
  15 * y + 4 * z - 15 * w

theorem correction_amount_proof (y z w : ℕ) :
  correction_amount y z w = 
    y * (coin_value "quarter" - coin_value "dime") +
    z * (coin_value "nickel" - coin_value "penny") -
    w * (coin_value "quarter" - coin_value "dime") :=
  sorry

end NUMINAMATH_CALUDE_correction_amount_proof_l312_31298


namespace NUMINAMATH_CALUDE_quadratic_not_always_two_roots_l312_31242

theorem quadratic_not_always_two_roots :
  ∃ (a b c : ℝ), b - c > a ∧ a ≠ 0 ∧ ¬(∃ (x y : ℝ), x ≠ y ∧ a * x^2 + b * x + c = 0 ∧ a * y^2 + b * y + c = 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_not_always_two_roots_l312_31242


namespace NUMINAMATH_CALUDE_expression_evaluation_l312_31292

theorem expression_evaluation : 6^2 + 4*5 - 2^3 + 4^2/2 = 56 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l312_31292


namespace NUMINAMATH_CALUDE_tangent_perpendicular_max_derivative_decreasing_function_range_l312_31285

noncomputable section

variables (a : ℝ)

def f (x : ℝ) : ℝ := a * x^2 - Real.exp x

def f_derivative (x : ℝ) : ℝ := 2 * a * x - Real.exp x

theorem tangent_perpendicular_max_derivative :
  f_derivative a 1 = 0 →
  ∀ x, f_derivative a x ≤ 0 :=
sorry

theorem decreasing_function_range :
  (∀ x₁ x₂, 0 ≤ x₁ → x₁ < x₂ → 
    f a x₂ + x₂ * (2 - 2 * Real.log 2) < f a x₁ + x₁ * (2 - 2 * Real.log 2)) →
  a ≤ 1 :=
sorry

end NUMINAMATH_CALUDE_tangent_perpendicular_max_derivative_decreasing_function_range_l312_31285


namespace NUMINAMATH_CALUDE_geometric_series_second_term_l312_31236

theorem geometric_series_second_term 
  (r : ℚ) 
  (sum : ℚ) 
  (h1 : r = 1 / 4) 
  (h2 : sum = 10) : 
  let a := sum * (1 - r)
  a * r = 15 / 8 := by
sorry

end NUMINAMATH_CALUDE_geometric_series_second_term_l312_31236


namespace NUMINAMATH_CALUDE_cube_properties_l312_31232

/-- Given a cube with surface area 864 square units, prove its volume and diagonal length -/
theorem cube_properties (s : ℝ) (h : 6 * s^2 = 864) : 
  s^3 = 1728 ∧ s * Real.sqrt 3 = 12 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_cube_properties_l312_31232


namespace NUMINAMATH_CALUDE_factorization_equality_l312_31212

theorem factorization_equality (a : ℝ) : 2 * a^2 - 8 = 2 * (a + 2) * (a - 2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l312_31212


namespace NUMINAMATH_CALUDE_square_difference_formula_l312_31251

theorem square_difference_formula (x y : ℚ) 
  (h1 : x + y = 8 / 15) (h2 : x - y = 2 / 15) : 
  x^2 - y^2 = 16 / 225 := by
sorry

end NUMINAMATH_CALUDE_square_difference_formula_l312_31251


namespace NUMINAMATH_CALUDE_mikey_leaves_total_l312_31279

theorem mikey_leaves_total (initial_leaves additional_leaves : Float) : 
  initial_leaves = 356.0 → additional_leaves = 112.0 → 
  initial_leaves + additional_leaves = 468.0 := by
  sorry

end NUMINAMATH_CALUDE_mikey_leaves_total_l312_31279


namespace NUMINAMATH_CALUDE_george_borrowing_weeks_l312_31262

def loan_amount : ℝ := 100
def initial_fee_rate : ℝ := 0.05
def total_fee : ℝ := 15

def fee_after_weeks (weeks : ℕ) : ℝ :=
  loan_amount * initial_fee_rate * (2 ^ weeks - 1)

theorem george_borrowing_weeks :
  ∃ (weeks : ℕ), weeks > 0 ∧ fee_after_weeks weeks ≤ total_fee ∧ fee_after_weeks (weeks + 1) > total_fee :=
by sorry

end NUMINAMATH_CALUDE_george_borrowing_weeks_l312_31262


namespace NUMINAMATH_CALUDE_angle_C_measure_l312_31222

/-- Represents a hexagon CALCUL with specific angle properties -/
structure Hexagon where
  -- Angles of the hexagon
  A : ℝ
  C : ℝ
  L : ℝ
  U : ℝ
  -- Conditions
  angle_sum : A + C + L + U + L + C = 720
  C_eq_L_eq_U : C = L ∧ L = U
  A_eq_L_eq_C : A = L ∧ L = C
  A_L_supplementary : A + L = 180

/-- The measure of angle C in the hexagon CALCUL is 120° -/
theorem angle_C_measure (h : Hexagon) : h.C = 120 := by
  sorry

end NUMINAMATH_CALUDE_angle_C_measure_l312_31222


namespace NUMINAMATH_CALUDE_mei_age_l312_31277

/-- Given the ages of Li, Zhang, Jung, and Mei, prove Mei's age is 13 --/
theorem mei_age (li_age zhang_age jung_age mei_age : ℕ) : 
  li_age = 12 →
  zhang_age = 2 * li_age →
  jung_age = zhang_age + 2 →
  mei_age = jung_age / 2 →
  mei_age = 13 := by
  sorry


end NUMINAMATH_CALUDE_mei_age_l312_31277


namespace NUMINAMATH_CALUDE_pen_gain_percentage_l312_31255

/-- 
Given that the selling price of 5 pens equals the cost price of 10 pens, 
prove that the gain percentage is 100%.
-/
theorem pen_gain_percentage (cost selling : ℝ) 
  (h : 5 * selling = 10 * cost) : 
  (selling - cost) / cost * 100 = 100 := by
  sorry

end NUMINAMATH_CALUDE_pen_gain_percentage_l312_31255


namespace NUMINAMATH_CALUDE_exists_graph_clique_lt_chromatic_l312_31289

/-- A graph type with vertices and edges -/
structure Graph where
  V : Type
  E : V → V → Prop

/-- The clique number of a graph -/
def cliqueNumber (G : Graph) : ℕ := sorry

/-- The chromatic number of a graph -/
def chromaticNumber (G : Graph) : ℕ := sorry

/-- Theorem: There exists a graph with clique number smaller than its chromatic number -/
theorem exists_graph_clique_lt_chromatic :
  ∃ (G : Graph), cliqueNumber G < chromaticNumber G := by sorry

end NUMINAMATH_CALUDE_exists_graph_clique_lt_chromatic_l312_31289


namespace NUMINAMATH_CALUDE_find_t_value_l312_31259

theorem find_t_value (s t : ℚ) 
  (eq1 : 15 * s + 7 * t = 210)
  (eq2 : t = 3 * s - 1) : 
  t = 205 / 12 := by
  sorry

end NUMINAMATH_CALUDE_find_t_value_l312_31259


namespace NUMINAMATH_CALUDE_triangle_side_length_l312_31276

-- Define the triangle ABC
def Triangle (A B C : ℝ × ℝ) : Prop :=
  -- Add conditions for a valid triangle here
  True

-- Define the right angle at A
def RightAngleAtA (A B C : ℝ × ℝ) : Prop :=
  -- Add condition for right angle at A
  True

-- Define the length of a side
def Length (P Q : ℝ × ℝ) : ℝ :=
  -- Add definition for length between two points
  0

-- Define tangent of an angle
def Tan (A B C : ℝ × ℝ) : ℝ :=
  -- Add definition for tangent of angle C
  0

-- Define cosine of an angle
def Cos (A B C : ℝ × ℝ) : ℝ :=
  -- Add definition for cosine of angle B
  0

theorem triangle_side_length 
  (A B C : ℝ × ℝ) 
  (h_triangle : Triangle A B C)
  (h_right_angle : RightAngleAtA A B C)
  (h_BC_length : Length B C = 10)
  (h_tan_cos : Tan A B C = 3 * Cos A B C) :
  Length A B = 20 * Real.sqrt 2 / 3 :=
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l312_31276


namespace NUMINAMATH_CALUDE_maria_yearly_distance_l312_31227

/-- Represents a pedometer with a maximum step count before resetting --/
structure Pedometer where
  max_steps : ℕ
  current_steps : ℕ

/-- Represents a postman's walking data for a year --/
structure PostmanYearlyData where
  pedometer : Pedometer
  flips : ℕ
  final_reading : ℕ
  steps_per_mile : ℕ

/-- Calculate the total distance walked by a postman in a year --/
def calculate_yearly_distance (data : PostmanYearlyData) : ℕ :=
  sorry

/-- Maria's yearly walking data --/
def maria_data : PostmanYearlyData :=
  { pedometer := { max_steps := 99999, current_steps := 0 },
    flips := 50,
    final_reading := 25000,
    steps_per_mile := 1500 }

theorem maria_yearly_distance :
  calculate_yearly_distance maria_data = 3350 :=
sorry

end NUMINAMATH_CALUDE_maria_yearly_distance_l312_31227


namespace NUMINAMATH_CALUDE_min_value_theorem_l312_31256

theorem min_value_theorem (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a + b = 2) :
  (1 / (a + 1) + 4 / (b + 1)) ≥ 9/4 ∧
  ∃ (a₀ b₀ : ℝ), 0 < a₀ ∧ 0 < b₀ ∧ a₀ + b₀ = 2 ∧ 1 / (a₀ + 1) + 4 / (b₀ + 1) = 9/4 :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l312_31256


namespace NUMINAMATH_CALUDE_newspaper_conference_overlap_l312_31257

theorem newspaper_conference_overlap (total : ℕ) (writers : ℕ) (editors : ℕ) 
  (h_total : total = 110)
  (h_writers : writers = 45)
  (h_editors : editors ≥ 39)
  (h_max_overlap : ∀ overlap : ℕ, overlap ≤ 26)
  (h_neither : ∀ overlap : ℕ, 2 * overlap = total - writers - editors + overlap) :
  ∃ overlap : ℕ, overlap = 26 ∧ 
    writers + editors - overlap + 2 * overlap = total ∧
    overlap = total - writers - editors + overlap :=
by sorry

end NUMINAMATH_CALUDE_newspaper_conference_overlap_l312_31257


namespace NUMINAMATH_CALUDE_cubic_expression_value_l312_31249

theorem cubic_expression_value (a : ℝ) (h : a^2 + a - 1 = 0) : a^3 + 2*a^2 + 3 = 4 := by
  sorry

end NUMINAMATH_CALUDE_cubic_expression_value_l312_31249


namespace NUMINAMATH_CALUDE_johns_expenses_l312_31268

/-- Given that John spent 40% of his earnings on rent and had 32% left over,
    prove that he spent 30% less on the dishwasher compared to the rent. -/
theorem johns_expenses (earnings : ℝ) (rent_percent : ℝ) (leftover_percent : ℝ)
  (h1 : rent_percent = 40)
  (h2 : leftover_percent = 32) :
  let dishwasher_percent := 100 - rent_percent - leftover_percent
  (rent_percent - dishwasher_percent) / rent_percent * 100 = 30 := by
  sorry

end NUMINAMATH_CALUDE_johns_expenses_l312_31268


namespace NUMINAMATH_CALUDE_weight_of_barium_fluoride_l312_31217

/-- The atomic weight of Barium in g/mol -/
def atomic_weight_Ba : ℝ := 137.33

/-- The atomic weight of Fluorine in g/mol -/
def atomic_weight_F : ℝ := 19.00

/-- The number of Barium atoms in BaF2 -/
def num_Ba : ℕ := 1

/-- The number of Fluorine atoms in BaF2 -/
def num_F : ℕ := 2

/-- The number of moles of BaF2 -/
def num_moles : ℝ := 3

/-- Theorem: The weight of 3 moles of Barium fluoride (BaF2) is 525.99 grams -/
theorem weight_of_barium_fluoride :
  (num_moles * (num_Ba * atomic_weight_Ba + num_F * atomic_weight_F)) = 525.99 := by
  sorry

end NUMINAMATH_CALUDE_weight_of_barium_fluoride_l312_31217


namespace NUMINAMATH_CALUDE_max_distance_with_tire_switch_l312_31271

/-- Given a car with front tires lasting 24000 km and rear tires lasting 36000 km,
    the maximum distance the car can travel by switching tires once is 48000 km. -/
theorem max_distance_with_tire_switch (front_tire_life rear_tire_life : ℕ) 
  (h1 : front_tire_life = 24000)
  (h2 : rear_tire_life = 36000) :
  ∃ (switch_point : ℕ), 
    switch_point ≤ front_tire_life ∧
    switch_point ≤ rear_tire_life ∧
    switch_point + min (front_tire_life - switch_point) (rear_tire_life - switch_point) = 48000 :=
by sorry

end NUMINAMATH_CALUDE_max_distance_with_tire_switch_l312_31271


namespace NUMINAMATH_CALUDE_paper_cup_cost_theorem_l312_31223

/-- The total number of pallets -/
def total_pallets : ℕ := 20

/-- The number of paper towel pallets -/
def paper_towel_pallets : ℕ := total_pallets / 2

/-- The number of tissue pallets -/
def tissue_pallets : ℕ := total_pallets / 4

/-- The number of paper plate pallets -/
def paper_plate_pallets : ℕ := total_pallets / 5

/-- The number of paper cup pallets -/
def paper_cup_pallets : ℕ := total_pallets - (paper_towel_pallets + tissue_pallets + paper_plate_pallets)

/-- The cost of a single paper cup pallet -/
def paper_cup_pallet_cost : ℕ := 50

/-- The total cost spent on paper cup pallets -/
def total_paper_cup_cost : ℕ := paper_cup_pallets * paper_cup_pallet_cost

theorem paper_cup_cost_theorem : total_paper_cup_cost = 50 := by
  sorry

end NUMINAMATH_CALUDE_paper_cup_cost_theorem_l312_31223


namespace NUMINAMATH_CALUDE_distribution_percent_below_mean_plus_std_dev_l312_31273

-- Define a symmetric distribution with mean m and standard deviation d
def SymmetricDistribution (μ : ℝ) (σ : ℝ) (F : ℝ → ℝ) : Prop :=
  ∀ x, F (μ + x) + F (μ - x) = 1

-- Define the condition that 36% of the distribution lies within one standard deviation of the mean
def WithinOneStdDev (μ : ℝ) (σ : ℝ) (F : ℝ → ℝ) : Prop :=
  F (μ + σ) - F (μ - σ) = 0.36

-- Theorem statement
theorem distribution_percent_below_mean_plus_std_dev
  (μ σ : ℝ) (F : ℝ → ℝ) 
  (h_symmetric : SymmetricDistribution μ σ F)
  (h_within_one_std_dev : WithinOneStdDev μ σ F) :
  F (μ + σ) = 0.68 := by
  sorry

end NUMINAMATH_CALUDE_distribution_percent_below_mean_plus_std_dev_l312_31273


namespace NUMINAMATH_CALUDE_max_crosses_4x10_impossible_5x10_l312_31299

/-- Represents a rectangular table with crosses placed in its cells -/
structure CrossTable (m n : ℕ) :=
  (crosses : Fin m → Fin n → Bool)

/-- Checks if the number of crosses in a given row is odd -/
def has_odd_row_crosses (t : CrossTable m n) (row : Fin m) : Prop :=
  (Finset.filter (λ col => t.crosses row col) (Finset.univ : Finset (Fin n))).card % 2 = 1

/-- Checks if the number of crosses in a given column is odd -/
def has_odd_col_crosses (t : CrossTable m n) (col : Fin n) : Prop :=
  (Finset.filter (λ row => t.crosses row col) (Finset.univ : Finset (Fin m))).card % 2 = 1

/-- Checks if all rows and columns have odd number of crosses -/
def all_odd_crosses (t : CrossTable m n) : Prop :=
  (∀ row, has_odd_row_crosses t row) ∧ (∀ col, has_odd_col_crosses t col)

/-- Counts the total number of crosses in the table -/
def total_crosses (t : CrossTable m n) : ℕ :=
  (Finset.filter (λ (row, col) => t.crosses row col) (Finset.univ : Finset (Fin m × Fin n))).card

theorem max_crosses_4x10 :
  (∃ (t : CrossTable 4 10), all_odd_crosses t ∧ total_crosses t = 30) ∧
  (∀ (t : CrossTable 4 10), all_odd_crosses t → total_crosses t ≤ 30) :=
sorry

theorem impossible_5x10 :
  ¬∃ (t : CrossTable 5 10), all_odd_crosses t :=
sorry

end NUMINAMATH_CALUDE_max_crosses_4x10_impossible_5x10_l312_31299


namespace NUMINAMATH_CALUDE_sin_2x_derivative_l312_31286

theorem sin_2x_derivative (x : ℝ) : 
  deriv (λ x => Real.sin (2 * x)) x = 2 * Real.cos (2 * x) := by
sorry

end NUMINAMATH_CALUDE_sin_2x_derivative_l312_31286


namespace NUMINAMATH_CALUDE_sum_of_min_max_x_l312_31215

theorem sum_of_min_max_x (x y z : ℝ) (sum_eq : x + y + z = 5) (sum_sq_eq : x^2 + y^2 + z^2 = 10) :
  ∃ (m M : ℝ), (∀ x', ∃ y' z', x' + y' + z' = 5 ∧ x'^2 + y'^2 + z'^2 = 10 → m ≤ x' ∧ x' ≤ M) ∧
                m + M = 16/3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_min_max_x_l312_31215


namespace NUMINAMATH_CALUDE_apples_eaten_proof_l312_31203

-- Define the daily apple consumption for each person
def simone_daily : ℚ := 1/2
def lauri_daily : ℚ := 1/3
def alex_daily : ℚ := 1/4

-- Define the number of days each person ate apples
def simone_days : ℕ := 16
def lauri_days : ℕ := 15
def alex_days : ℕ := 20

-- Define the total number of apples eaten by all three
def total_apples : ℚ := simone_daily * simone_days + lauri_daily * lauri_days + alex_daily * alex_days

-- Theorem statement
theorem apples_eaten_proof : total_apples = 18 := by
  sorry

end NUMINAMATH_CALUDE_apples_eaten_proof_l312_31203


namespace NUMINAMATH_CALUDE_max_value_quadratic_l312_31248

theorem max_value_quadratic :
  ∃ (M : ℝ), M = 19 ∧ ∀ p : ℝ, -3 * p^2 + 18 * p - 8 ≤ M :=
by sorry

end NUMINAMATH_CALUDE_max_value_quadratic_l312_31248


namespace NUMINAMATH_CALUDE_triangle_height_and_median_l312_31295

-- Define the triangle ABC
def A : ℝ × ℝ := (3, 0)
def B : ℝ × ℝ := (4, 5)
def C : ℝ × ℝ := (0, 7)

-- Define the height from A to BC
def height_equation (x y : ℝ) : Prop := 2 * x - y - 6 = 0

-- Define the median from A to BC
def median_equation (x y : ℝ) : Prop := 6 * x + y - 18 = 0

theorem triangle_height_and_median :
  (∀ x y : ℝ, height_equation x y ↔ 
    (y - A.2) / (x - A.1) = -1 / (B.1 - C.1) / (B.2 - C.2)) ∧
  (∀ x y : ℝ, median_equation x y ↔ 
    (y - A.2) / (x - A.1) = ((B.2 + C.2) / 2 - A.2) / ((B.1 + C.1) / 2 - A.1)) :=
by sorry

end NUMINAMATH_CALUDE_triangle_height_and_median_l312_31295


namespace NUMINAMATH_CALUDE_upper_limit_of_set_W_l312_31221

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(m ∣ n)

def set_W (upper_bound : ℕ) : Set ℕ :=
  {n : ℕ | n > 10 ∧ n ≤ upper_bound ∧ is_prime n}

theorem upper_limit_of_set_W (upper_bound : ℕ) :
  (∃ (w : Set ℕ), w = set_W upper_bound ∧ 
   (∃ (max min : ℕ), max ∈ w ∧ min ∈ w ∧ 
    (∀ x ∈ w, x ≤ max ∧ x ≥ min) ∧ max - min = 12)) →
  upper_bound = 23 :=
sorry

end NUMINAMATH_CALUDE_upper_limit_of_set_W_l312_31221


namespace NUMINAMATH_CALUDE_cubic_function_coefficients_l312_31205

/-- Given a cubic function f(x) = ax³ - 4x² + bx - 3, 
    if f(1) = 3 and f(-2) = -47, then a = 4/3 and b = 26/3 -/
theorem cubic_function_coefficients (a b : ℚ) : 
  let f : ℚ → ℚ := λ x => a * x^3 - 4 * x^2 + b * x - 3
  (f 1 = 3 ∧ f (-2) = -47) → a = 4/3 ∧ b = 26/3 := by
  sorry


end NUMINAMATH_CALUDE_cubic_function_coefficients_l312_31205


namespace NUMINAMATH_CALUDE_checkerboard_sums_l312_31241

/-- Represents a 10x10 checkerboard filled with numbers 1 to 100 -/
def Checkerboard := Fin 10 → Fin 10 → Nat

/-- The checkerboard filled with numbers 1 to 100 in order -/
def filledCheckerboard : Checkerboard :=
  fun i j => i.val * 10 + j.val + 1

/-- The sum of the corner numbers on the checkerboard -/
def cornerSum (board : Checkerboard) : Nat :=
  board 0 0 + board 0 9 + board 9 0 + board 9 9

/-- The sum of the main diagonal numbers on the checkerboard -/
def diagonalSum (board : Checkerboard) : Nat :=
  board 0 0 + board 9 9

theorem checkerboard_sums :
  cornerSum filledCheckerboard = 202 ∧
  diagonalSum filledCheckerboard = 101 := by
  sorry


end NUMINAMATH_CALUDE_checkerboard_sums_l312_31241


namespace NUMINAMATH_CALUDE_reciprocal_of_sum_l312_31207

theorem reciprocal_of_sum : (1 / (1/4 + 1/6) : ℚ) = 12/5 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_of_sum_l312_31207


namespace NUMINAMATH_CALUDE_parallel_vectors_k_value_l312_31244

/-- Two vectors are parallel if their components are proportional -/
def are_parallel (a b : ℝ × ℝ) : Prop :=
  ∃ (t : ℝ), a.1 = t * b.1 ∧ a.2 = t * b.2

theorem parallel_vectors_k_value :
  ∀ (k : ℝ),
  let a : ℝ × ℝ := (2*k - 3, -6)
  let c : ℝ × ℝ := (2, 1)
  are_parallel a c → k = -9/2 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_k_value_l312_31244


namespace NUMINAMATH_CALUDE_bus_driver_compensation_l312_31240

/-- Calculates the total compensation for a bus driver based on hours worked and pay rates -/
theorem bus_driver_compensation
  (regular_rate : ℝ)
  (regular_hours : ℝ)
  (overtime_percentage : ℝ)
  (total_hours : ℝ)
  (h1 : regular_rate = 16)
  (h2 : regular_hours = 40)
  (h3 : overtime_percentage = 0.75)
  (h4 : total_hours = 50) :
  let overtime_rate := regular_rate * (1 + overtime_percentage)
  let overtime_hours := total_hours - regular_hours
  let regular_pay := regular_rate * regular_hours
  let overtime_pay := overtime_rate * overtime_hours
  regular_pay + overtime_pay = 920 := by
sorry

end NUMINAMATH_CALUDE_bus_driver_compensation_l312_31240


namespace NUMINAMATH_CALUDE_number_plus_five_equals_six_l312_31214

theorem number_plus_five_equals_six : ∃ x : ℝ, x + 5 = 6 ∧ x = 1 := by
  sorry

end NUMINAMATH_CALUDE_number_plus_five_equals_six_l312_31214


namespace NUMINAMATH_CALUDE_wider_bolt_width_l312_31237

theorem wider_bolt_width (a b : ℕ) (h1 : a = 45) (h2 : b > a) (h3 : Nat.gcd a b = 15) : 
  (∀ c : ℕ, c > a ∧ Nat.gcd a c = 15 → b ≤ c) → b = 60 := by
  sorry

end NUMINAMATH_CALUDE_wider_bolt_width_l312_31237


namespace NUMINAMATH_CALUDE_overall_loss_calculation_l312_31218

def stock_worth : ℝ := 15000

def profit_percentage : ℝ := 0.1
def loss_percentage : ℝ := 0.05

def profit_stock_ratio : ℝ := 0.2
def loss_stock_ratio : ℝ := 0.8

def profit_amount : ℝ := stock_worth * profit_stock_ratio * profit_percentage
def loss_amount : ℝ := stock_worth * loss_stock_ratio * loss_percentage

def overall_selling_price : ℝ := 
  (stock_worth * profit_stock_ratio * (1 + profit_percentage)) +
  (stock_worth * loss_stock_ratio * (1 - loss_percentage))

def overall_loss : ℝ := stock_worth - overall_selling_price

theorem overall_loss_calculation :
  overall_loss = 300 :=
sorry

end NUMINAMATH_CALUDE_overall_loss_calculation_l312_31218


namespace NUMINAMATH_CALUDE_larger_integer_value_l312_31247

theorem larger_integer_value (a b : ℕ+) 
  (h_quotient : (a : ℚ) / (b : ℚ) = 7 / 3)
  (h_product : (a : ℕ) * (b : ℕ) = 189) :
  max a b = 21 := by
sorry

end NUMINAMATH_CALUDE_larger_integer_value_l312_31247


namespace NUMINAMATH_CALUDE_f_range_l312_31266

def f (x : ℝ) : ℝ := 256 * x^9 - 576 * x^7 + 432 * x^5 - 120 * x^3 + 9 * x

theorem f_range :
  (∀ x ∈ Set.Icc (-1 : ℝ) 1, f x ∈ Set.Icc (-1 : ℝ) 1) ∧
  (∃ x₁ ∈ Set.Icc (-1 : ℝ) 1, f x₁ = -1) ∧
  (∃ x₂ ∈ Set.Icc (-1 : ℝ) 1, f x₂ = 1) :=
sorry

end NUMINAMATH_CALUDE_f_range_l312_31266


namespace NUMINAMATH_CALUDE_inequality_solution_set_l312_31288

def solution_set (x : ℝ) : Prop :=
  x ∈ Set.union (Set.Ioo 0 1) (Set.union (Set.Ioo 1 (2 ^ (5/7))) (Set.Ioi 4))

theorem inequality_solution_set (x : ℝ) :
  (|1 / Real.log (1/2 * x) + 2| > 3/2) ↔ solution_set x :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l312_31288


namespace NUMINAMATH_CALUDE_rectangle_dimension_change_l312_31209

theorem rectangle_dimension_change (l b : ℝ) (h1 : l > 0) (h2 : b > 0) : 
  let new_l := l / 2
  let new_area := (l * b) / 2
  ∃ new_b : ℝ, new_area = new_l * new_b ∧ new_b = b / 2 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_dimension_change_l312_31209


namespace NUMINAMATH_CALUDE_sum_of_bases_equals_1500_l312_31201

/-- Converts a number from base 13 to base 10 -/
def base13ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (13 ^ i)) 0

/-- Converts a number from base 14 to base 10 -/
def base14ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (14 ^ i)) 0

/-- The main theorem to prove -/
theorem sum_of_bases_equals_1500 :
  let num1 := base13ToBase10 [6, 2, 3]
  let num2 := base14ToBase10 [9, 12, 4]
  num1 + num2 = 1500 := by sorry

end NUMINAMATH_CALUDE_sum_of_bases_equals_1500_l312_31201


namespace NUMINAMATH_CALUDE_min_value_theorem_l312_31284

theorem min_value_theorem (x y z : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_pos_z : 0 < z)
  (h_eq : (2 / x) + (3 / y) + (5 / z) = 10) :
  ∀ a b c : ℝ, 0 < a ∧ 0 < b ∧ 0 < c ∧ (2 / a) + (3 / b) + (5 / c) = 10 →
  x^4 * y^3 * z^2 ≤ a^4 * b^3 * c^2 ∧
  x^4 * y^3 * z^2 = 390625 / 1296 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l312_31284


namespace NUMINAMATH_CALUDE_sum_of_last_three_digits_of_fibonacci_factorial_series_l312_31293

def fibonacci_factorial_series : List Nat := [1, 2, 3, 5, 8, 13, 21]

def last_three_digits (n : Nat) : Nat :=
  n % 1000

def factorial (n : Nat) : Nat :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

theorem sum_of_last_three_digits_of_fibonacci_factorial_series :
  (fibonacci_factorial_series.map (λ n => last_three_digits (factorial n))).sum % 1000 = 249 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_last_three_digits_of_fibonacci_factorial_series_l312_31293


namespace NUMINAMATH_CALUDE_min_xy_value_l312_31216

theorem min_xy_value (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x * y = x + y + 3) :
  x * y ≥ 9 := by
sorry

end NUMINAMATH_CALUDE_min_xy_value_l312_31216


namespace NUMINAMATH_CALUDE_third_month_sales_l312_31264

def sales_1 : ℕ := 5400
def sales_2 : ℕ := 9000
def sales_4 : ℕ := 7200
def sales_5 : ℕ := 4500
def sales_6 : ℕ := 1200
def average_sale : ℕ := 5600
def num_months : ℕ := 6

theorem third_month_sales :
  ∃ (sales_3 : ℕ),
    sales_3 = num_months * average_sale - (sales_1 + sales_2 + sales_4 + sales_5 + sales_6) ∧
    sales_3 = 6300 := by
  sorry

end NUMINAMATH_CALUDE_third_month_sales_l312_31264


namespace NUMINAMATH_CALUDE_no_triangle_exists_l312_31211

-- Define the triangle
structure Triangle :=
  (a b : ℝ)
  (angleBisector : ℝ)

-- Define the conditions
def triangleConditions (t : Triangle) : Prop :=
  t.a = 12 ∧ t.b = 20 ∧ t.angleBisector = 15

-- Theorem statement
theorem no_triangle_exists :
  ¬ ∃ (t : Triangle), triangleConditions t ∧ 
  ∃ (c : ℝ), c > 0 ∧ 
  (c + t.a > t.b) ∧ (c + t.b > t.a) ∧ (t.a + t.b > c) ∧
  t.angleBisector = Real.sqrt (t.a * t.b * (1 - (c^2 / (t.a + t.b)^2))) :=
sorry

end NUMINAMATH_CALUDE_no_triangle_exists_l312_31211


namespace NUMINAMATH_CALUDE_percentage_with_neither_is_twenty_percent_l312_31263

/-- Represents the study of adults in a neighborhood -/
structure NeighborhoodStudy where
  total : ℕ
  insomnia : ℕ
  migraines : ℕ
  both : ℕ

/-- Calculates the percentage of adults with neither insomnia nor migraines -/
def percentageWithNeither (study : NeighborhoodStudy) : ℚ :=
  let withNeither := study.total - (study.insomnia + study.migraines - study.both)
  (withNeither : ℚ) / study.total * 100

/-- The main theorem stating that the percentage of adults with neither condition is 20% -/
theorem percentage_with_neither_is_twenty_percent (study : NeighborhoodStudy)
  (h_total : study.total = 150)
  (h_insomnia : study.insomnia = 90)
  (h_migraines : study.migraines = 60)
  (h_both : study.both = 30) :
  percentageWithNeither study = 20 := by
  sorry

#eval percentageWithNeither { total := 150, insomnia := 90, migraines := 60, both := 30 }

end NUMINAMATH_CALUDE_percentage_with_neither_is_twenty_percent_l312_31263


namespace NUMINAMATH_CALUDE_farm_chicken_ratio_l312_31225

/-- Given a farm with chickens, prove the ratio of hens to roosters -/
theorem farm_chicken_ratio (total : ℕ) (hens : ℕ) (X : ℕ) : 
  total = 75 → 
  hens = 67 → 
  hens = X * (total - hens) - 5 → 
  X = 9 := by
sorry

end NUMINAMATH_CALUDE_farm_chicken_ratio_l312_31225


namespace NUMINAMATH_CALUDE_simplify_expression_l312_31250

theorem simplify_expression : (625 : ℝ) ^ (1/4) * (225 : ℝ) ^ (1/2) = 75 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l312_31250


namespace NUMINAMATH_CALUDE_hike_attendance_l312_31294

theorem hike_attendance (num_cars num_taxis num_vans : ℕ) 
                        (people_per_car people_per_taxi people_per_van : ℕ) : 
  num_cars = 3 → 
  num_taxis = 6 → 
  num_vans = 2 → 
  people_per_car = 4 → 
  people_per_taxi = 6 → 
  people_per_van = 5 → 
  num_cars * people_per_car + num_taxis * people_per_taxi + num_vans * people_per_van = 58 := by
  sorry


end NUMINAMATH_CALUDE_hike_attendance_l312_31294


namespace NUMINAMATH_CALUDE_line_intersects_segment_l312_31272

/-- A line defined by the equation 2x + y - b = 0 intersects the line segment
    between points (1,0) and (-1,0) if and only if -2 ≤ b ≤ 2. -/
theorem line_intersects_segment (b : ℝ) :
  (∃ (x y : ℝ), 2*x + y - b = 0 ∧ 
    ((x = 1 ∧ y = 0) ∨ 
     (x = -1 ∧ y = 0) ∨ 
     (∃ (t : ℝ), 0 < t ∧ t < 1 ∧ x = -1 + 2*t ∧ y = 0)))
  ↔ -2 ≤ b ∧ b ≤ 2 :=
by sorry

end NUMINAMATH_CALUDE_line_intersects_segment_l312_31272


namespace NUMINAMATH_CALUDE_complex_fraction_evaluation_l312_31224

theorem complex_fraction_evaluation : 
  (((1 / 2) * (1 / 3) * (1 / 4) * (1 / 5) + (3 / 2) * (3 / 4) * (3 / 5)) / 
   ((1 / 2) * (2 / 3) * (2 / 5))) = 41 / 8 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_evaluation_l312_31224


namespace NUMINAMATH_CALUDE_union_of_M_and_N_l312_31265

def M : Set ℕ := {1, 2}
def N : Set ℕ := {n | ∃ a ∈ M, n = 2*a - 1}

theorem union_of_M_and_N : M ∪ N = {1, 2, 3} := by sorry

end NUMINAMATH_CALUDE_union_of_M_and_N_l312_31265


namespace NUMINAMATH_CALUDE_pittsburgh_police_stations_count_l312_31278

/-- The number of police stations in Pittsburgh -/
def pittsburgh_police_stations : ℕ := 20

/-- The number of stores in Pittsburgh -/
def pittsburgh_stores : ℕ := 2000

/-- The number of hospitals in Pittsburgh -/
def pittsburgh_hospitals : ℕ := 500

/-- The number of schools in Pittsburgh -/
def pittsburgh_schools : ℕ := 200

/-- The total number of buildings in the new city -/
def new_city_total_buildings : ℕ := 2175

theorem pittsburgh_police_stations_count :
  pittsburgh_police_stations = 20 :=
by
  have new_city_stores : ℕ := pittsburgh_stores / 2
  have new_city_hospitals : ℕ := pittsburgh_hospitals * 2
  have new_city_schools : ℕ := pittsburgh_schools - 50
  have new_city_police_stations : ℕ := pittsburgh_police_stations + 5
  
  have : new_city_stores + new_city_hospitals + new_city_schools + new_city_police_stations = new_city_total_buildings :=
    by sorry
  
  sorry -- The proof goes here

end NUMINAMATH_CALUDE_pittsburgh_police_stations_count_l312_31278


namespace NUMINAMATH_CALUDE_sandys_shorts_expense_l312_31287

/-- Given Sandy's shopping expenses, calculate the amount spent on shorts -/
theorem sandys_shorts_expense (total shirt jacket : ℚ)
  (h_total : total = 33.56)
  (h_shirt : shirt = 12.14)
  (h_jacket : jacket = 7.43) :
  total - shirt - jacket = 13.99 := by
  sorry

end NUMINAMATH_CALUDE_sandys_shorts_expense_l312_31287


namespace NUMINAMATH_CALUDE_kendall_driving_distance_l312_31296

/-- The distance Kendall drove with her father -/
def distance_with_father (total_distance mother_distance : ℝ) : ℝ :=
  total_distance - mother_distance

/-- Theorem: Kendall drove 0.50 miles with her father -/
theorem kendall_driving_distance :
  distance_with_father 0.67 0.17 = 0.50 := by
  sorry

end NUMINAMATH_CALUDE_kendall_driving_distance_l312_31296


namespace NUMINAMATH_CALUDE_inequality_proof_l312_31274

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_sum : a + b + c = 1) : 
  (a - b*c)/(a + b*c) + (b - c*a)/(b + c*a) + (c - a*b)/(c + a*b) ≤ 3/2 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l312_31274


namespace NUMINAMATH_CALUDE_oliver_money_theorem_l312_31283

def oliver_money_problem (initial savings frisbee puzzle gift : ℕ) : Prop :=
  initial + savings - frisbee - puzzle + gift = 15

theorem oliver_money_theorem :
  oliver_money_problem 9 5 4 3 8 := by
  sorry

end NUMINAMATH_CALUDE_oliver_money_theorem_l312_31283


namespace NUMINAMATH_CALUDE_subtraction_and_simplification_l312_31297

theorem subtraction_and_simplification : (8 : ℚ) / 23 - (5 : ℚ) / 46 = (11 : ℚ) / 46 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_and_simplification_l312_31297


namespace NUMINAMATH_CALUDE_brown_eyed_brunettes_count_l312_31238

/-- Represents the number of girls with specific characteristics -/
structure GirlCount where
  total : ℕ
  greenEyedBlondes : ℕ
  brunettes : ℕ
  brownEyed : ℕ

/-- Calculates the number of brown-eyed brunettes given the counts of girls with specific characteristics -/
def brownEyedBrunettes (gc : GirlCount) : ℕ :=
  gc.brownEyed - (gc.total - gc.brunettes - gc.greenEyedBlondes)

/-- Theorem stating that given the specific counts, there are 20 brown-eyed brunettes -/
theorem brown_eyed_brunettes_count (gc : GirlCount) 
  (h1 : gc.total = 60)
  (h2 : gc.greenEyedBlondes = 20)
  (h3 : gc.brunettes = 35)
  (h4 : gc.brownEyed = 25) :
  brownEyedBrunettes gc = 20 := by
  sorry

#eval brownEyedBrunettes { total := 60, greenEyedBlondes := 20, brunettes := 35, brownEyed := 25 }

end NUMINAMATH_CALUDE_brown_eyed_brunettes_count_l312_31238


namespace NUMINAMATH_CALUDE_lowest_number_of_students_twenty_four_divisible_lowest_number_is_twenty_four_l312_31275

theorem lowest_number_of_students (n : ℕ) : n > 0 ∧ 8 ∣ n ∧ 12 ∣ n → n ≥ 24 := by
  sorry

theorem twenty_four_divisible : 8 ∣ 24 ∧ 12 ∣ 24 := by
  sorry

theorem lowest_number_is_twenty_four : ∃ n : ℕ, n > 0 ∧ 8 ∣ n ∧ 12 ∣ n ∧ n = 24 := by
  sorry

end NUMINAMATH_CALUDE_lowest_number_of_students_twenty_four_divisible_lowest_number_is_twenty_four_l312_31275


namespace NUMINAMATH_CALUDE_path_area_l312_31270

/-- Calculates the area of a path surrounding a rectangular field -/
theorem path_area (field_length field_width path_width : ℝ) :
  field_length = 85 ∧ 
  field_width = 55 ∧ 
  path_width = 2.5 → 
  (field_length + 2 * path_width) * (field_width + 2 * path_width) - 
  field_length * field_width = 725 := by
  sorry

end NUMINAMATH_CALUDE_path_area_l312_31270


namespace NUMINAMATH_CALUDE_cookies_prepared_l312_31231

theorem cookies_prepared (num_guests : ℕ) (cookies_per_guest : ℕ) : 
  num_guests = 10 → cookies_per_guest = 18 → num_guests * cookies_per_guest = 180 := by
  sorry

#check cookies_prepared

end NUMINAMATH_CALUDE_cookies_prepared_l312_31231


namespace NUMINAMATH_CALUDE_liquid_X_percentage_l312_31235

/-- The percentage of liquid X in solution A -/
def percentage_X_in_A : ℝ := 0.0009

/-- The percentage of liquid X in solution B -/
def percentage_X_in_B : ℝ := 0.018

/-- The weight of solution A in grams -/
def weight_A : ℝ := 200

/-- The weight of solution B in grams -/
def weight_B : ℝ := 700

/-- The percentage of liquid X in the resulting mixture -/
def percentage_X_in_mixture : ℝ := 0.0142

theorem liquid_X_percentage :
  percentage_X_in_A * weight_A + percentage_X_in_B * weight_B =
  percentage_X_in_mixture * (weight_A + weight_B) := by
  sorry

end NUMINAMATH_CALUDE_liquid_X_percentage_l312_31235


namespace NUMINAMATH_CALUDE_special_polynomial_zeros_l312_31239

/-- A polynomial of degree 5 with specific properties -/
def SpecialPolynomial (P : ℂ → ℂ) : Prop :=
  ∃ (r s : ℤ) (a b : ℤ),
    (∀ x, P x = x * (x - r) * (x - s) * (x^2 + a*x + b)) ∧
    (∀ x, ∃ (c : ℤ), P x = c)

theorem special_polynomial_zeros (P : ℂ → ℂ) (h : SpecialPolynomial P) :
  P ((1 + Complex.I * Real.sqrt 15) / 2) = 0 ∧
  P ((1 + Complex.I * Real.sqrt 17) / 2) = 0 :=
sorry

end NUMINAMATH_CALUDE_special_polynomial_zeros_l312_31239


namespace NUMINAMATH_CALUDE_square_area_error_percentage_l312_31229

/-- If the measured side length of a square is 102.5% of its actual side length,
    then the percentage of error in the calculated area of the square is 5.0625%. -/
theorem square_area_error_percentage (a : ℝ) (h : a > 0) :
  let measured_side := 1.025 * a
  let actual_area := a ^ 2
  let calculated_area := measured_side ^ 2
  (calculated_area - actual_area) / actual_area * 100 = 5.0625 := by
sorry

end NUMINAMATH_CALUDE_square_area_error_percentage_l312_31229


namespace NUMINAMATH_CALUDE_smallest_sum_five_consecutive_odd_integers_l312_31213

theorem smallest_sum_five_consecutive_odd_integers : 
  ∀ n : ℕ, n ≥ 35 → 
  ∃ k : ℤ, (k % 2 ≠ 0) ∧ 
  (n = k + (k + 2) + (k + 4) + (k + 6) + (k + 8)) ∧
  (∀ m : ℕ, m < 35 → 
    ¬∃ j : ℤ, (j % 2 ≠ 0) ∧ 
    (m = j + (j + 2) + (j + 4) + (j + 6) + (j + 8))) :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_sum_five_consecutive_odd_integers_l312_31213


namespace NUMINAMATH_CALUDE_square_fraction_count_l312_31246

theorem square_fraction_count : 
  ∃ (S : Finset ℤ), 
    (∀ n ∈ S, 0 ≤ n ∧ n ≤ 29 ∧ ∃ k : ℤ, n / (30 - n) = k^2) ∧ 
    (∀ n : ℤ, 0 ≤ n ∧ n ≤ 29 ∧ (∃ k : ℤ, n / (30 - n) = k^2) → n ∈ S) ∧
    Finset.card S = 3 :=
by sorry

end NUMINAMATH_CALUDE_square_fraction_count_l312_31246


namespace NUMINAMATH_CALUDE_four_consecutive_primes_sum_l312_31282

theorem four_consecutive_primes_sum (A B : ℕ) : 
  (A > 0) → 
  (B > 0) → 
  (Nat.Prime A) → 
  (Nat.Prime B) → 
  (Nat.Prime (A - B)) → 
  (Nat.Prime (A + B)) → 
  (∃ p q r s : ℕ, 
    Nat.Prime p ∧ Nat.Prime q ∧ Nat.Prime r ∧ Nat.Prime s ∧
    q = p + 2 ∧ r = q + 2 ∧ s = r + 2 ∧
    ((A = p ∧ B = q) ∨ (A = q ∧ B = p) ∨ (A = r ∧ B = p) ∨ (A = s ∧ B = p))) →
  p + q + r + s = 17 :=
sorry

end NUMINAMATH_CALUDE_four_consecutive_primes_sum_l312_31282


namespace NUMINAMATH_CALUDE_mother_age_is_40_l312_31252

/-- The age of the mother -/
def mother_age : ℕ := sorry

/-- The sum of the ages of the 7 children -/
def children_ages_sum : ℕ := sorry

/-- The age of the mother is equal to the sum of the ages of her 7 children -/
axiom mother_age_eq_children_sum : mother_age = children_ages_sum

/-- After 20 years, the sum of the ages of the children will be three times the age of the mother -/
axiom future_age_relation : children_ages_sum + 7 * 20 = 3 * (mother_age + 20)

theorem mother_age_is_40 : mother_age = 40 := by sorry

end NUMINAMATH_CALUDE_mother_age_is_40_l312_31252


namespace NUMINAMATH_CALUDE_hour_hand_rotation_3_to_6_l312_31202

/-- The number of segments in a clock face. -/
def clock_segments : ℕ := 12

/-- The number of degrees in a full rotation. -/
def full_rotation : ℕ := 360

/-- The number of hours between 3 o'clock and 6 o'clock. -/
def hours_passed : ℕ := 3

/-- The degree measure of the rotation of the hour hand from 3 o'clock to 6 o'clock. -/
def hour_hand_rotation : ℕ := (full_rotation / clock_segments) * hours_passed

theorem hour_hand_rotation_3_to_6 :
  hour_hand_rotation = 90 := by sorry

end NUMINAMATH_CALUDE_hour_hand_rotation_3_to_6_l312_31202


namespace NUMINAMATH_CALUDE_min_k_for_three_or_more_intersections_range_of_ratio_for_four_intersections_l312_31226

-- Define the curve M and line l
def curve_M (x y : ℝ) : Prop := (x^2 = -y) ∨ (x^2 = 4*y)
def line_l (k : ℝ) (x y : ℝ) : Prop := y = k*x - 3

-- Define the number of intersection points
def intersection_points (k : ℝ) : ℕ := sorry

-- Theorem 1: Minimum value of k when m ≥ 3
theorem min_k_for_three_or_more_intersections :
  ∀ k : ℝ, k > 0 → intersection_points k ≥ 3 → k ≥ Real.sqrt 3 := by sorry

-- Theorem 2: Range of |AB|/|CD| when m = 4
theorem range_of_ratio_for_four_intersections :
  ∀ k : ℝ, k > 0 → intersection_points k = 4 →
  ∃ r : ℝ, 0 < r ∧ r < 4 ∧
  (∃ A B C D : ℝ × ℝ,
    curve_M A.1 A.2 ∧ curve_M B.1 B.2 ∧ curve_M C.1 C.2 ∧ curve_M D.1 D.2 ∧
    line_l k A.1 A.2 ∧ line_l k B.1 B.2 ∧ line_l k C.1 C.2 ∧ line_l k D.1 D.2 ∧
    r = (Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)) /
        (Real.sqrt ((C.1 - D.1)^2 + (C.2 - D.2)^2))) := by sorry

end NUMINAMATH_CALUDE_min_k_for_three_or_more_intersections_range_of_ratio_for_four_intersections_l312_31226


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_isosceles_triangle_perimeter_proof_l312_31261

/-- An isosceles triangle with two sides measuring 4 and 9 has a perimeter of 22. -/
theorem isosceles_triangle_perimeter : ℝ → ℝ → ℝ → Prop :=
  fun a b c =>
    (a = 4 ∧ b = 9 ∧ c = 9) ∨ (a = 9 ∧ b = 4 ∧ c = 9) ∨ (a = 9 ∧ b = 9 ∧ c = 4) →
    a + b + c = 22

/-- Proof of the theorem -/
theorem isosceles_triangle_perimeter_proof :
  ∃ (a b c : ℝ), isosceles_triangle_perimeter a b c :=
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_isosceles_triangle_perimeter_proof_l312_31261


namespace NUMINAMATH_CALUDE_det_of_specific_matrix_l312_31260

theorem det_of_specific_matrix :
  let A : Matrix (Fin 2) (Fin 2) ℤ := !![7, -2; 6, 3]
  Matrix.det A = 33 := by sorry

end NUMINAMATH_CALUDE_det_of_specific_matrix_l312_31260


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l312_31253

open Real

theorem triangle_abc_properties (a b c : ℝ) (A B C : ℝ) :
  c * cos (π - B) = (b - 2 * a) * sin (π / 2 - C) →
  c = sqrt 13 →
  b = 3 →
  C = π / 3 ∧ 
  (1 / 2) * a * b * sin C = 3 * sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_abc_properties_l312_31253


namespace NUMINAMATH_CALUDE_ad_greater_bc_l312_31219

theorem ad_greater_bc (a b c d : ℝ) 
  (sum_eq : a + d = b + c) 
  (abs_ineq : |a - d| < |b - c|) : 
  a * d > b * c := by
  sorry

end NUMINAMATH_CALUDE_ad_greater_bc_l312_31219


namespace NUMINAMATH_CALUDE_smallest_double_multiple_of_2016_l312_31200

def consecutive_double (n : ℕ) : ℕ :=
  n * 1001 + n

theorem smallest_double_multiple_of_2016 :
  ∀ A : ℕ, A < 288 → ¬(∃ k : ℕ, consecutive_double A = 2016 * k) ∧
  ∃ k : ℕ, consecutive_double 288 = 2016 * k :=
by sorry

end NUMINAMATH_CALUDE_smallest_double_multiple_of_2016_l312_31200


namespace NUMINAMATH_CALUDE_only_one_equals_sum_of_squares_of_digits_l312_31204

/-- Sum of squares of digits of a natural number -/
def sum_of_squares_of_digits (n : ℕ) : ℕ :=
  sorry

/-- Theorem: The only positive integer n such that s(n) = n is 1 -/
theorem only_one_equals_sum_of_squares_of_digits :
  ∀ n : ℕ, n > 0 → (sum_of_squares_of_digits n = n ↔ n = 1) :=
by sorry

end NUMINAMATH_CALUDE_only_one_equals_sum_of_squares_of_digits_l312_31204


namespace NUMINAMATH_CALUDE_larger_solution_of_quadratic_l312_31254

theorem larger_solution_of_quadratic (x : ℝ) : 
  x^2 - 13*x + 36 = 0 → x = 9 ∨ x = 4 → 9 > 4 := by
  sorry

end NUMINAMATH_CALUDE_larger_solution_of_quadratic_l312_31254


namespace NUMINAMATH_CALUDE_chocolate_box_bars_l312_31243

def chocolate_problem (total_bars : ℕ) : Prop :=
  let bar_price : ℚ := 3
  let remaining_bars : ℕ := 4
  let sales : ℚ := 9
  bar_price * (total_bars - remaining_bars) = sales

theorem chocolate_box_bars : ∃ (x : ℕ), chocolate_problem x ∧ x = 7 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_box_bars_l312_31243


namespace NUMINAMATH_CALUDE_new_songs_added_l312_31234

def initial_songs : ℕ := 6
def deleted_songs : ℕ := 3
def final_songs : ℕ := 23

theorem new_songs_added : 
  final_songs - (initial_songs - deleted_songs) = 20 := by sorry

end NUMINAMATH_CALUDE_new_songs_added_l312_31234


namespace NUMINAMATH_CALUDE_least_integer_satisfying_inequality_l312_31269

theorem least_integer_satisfying_inequality :
  ∃ (x : ℤ), (∀ (y : ℤ), |y^2 + 3*y + 10| ≤ 25 - y → x ≤ y) ∧
             |x^2 + 3*x + 10| ≤ 25 - x ∧
             x = -5 := by
  sorry

end NUMINAMATH_CALUDE_least_integer_satisfying_inequality_l312_31269


namespace NUMINAMATH_CALUDE_integer_solutions_xy_eq_x_plus_y_l312_31291

theorem integer_solutions_xy_eq_x_plus_y :
  ∀ x y : ℤ, x * y = x + y ↔ (x = 0 ∧ y = 0) ∨ (x = 2 ∧ y = 2) := by
  sorry

end NUMINAMATH_CALUDE_integer_solutions_xy_eq_x_plus_y_l312_31291


namespace NUMINAMATH_CALUDE_perfect_number_examples_mn_value_S_is_perfect_number_min_value_a_plus_b_l312_31228

/-- Definition of a perfect number -/
def is_perfect_number (n : ℤ) : Prop :=
  ∃ a b : ℤ, n = a^2 + b^2

/-- Statement 1: 29 and 13 are perfect numbers, while 48 and 28 are not -/
theorem perfect_number_examples :
  is_perfect_number 29 ∧ is_perfect_number 13 ∧ ¬is_perfect_number 48 ∧ ¬is_perfect_number 28 :=
sorry

/-- Statement 2: Given a^2 - 4a + 8 = (a - m)^2 + n^2, prove that mn = ±4 -/
theorem mn_value (a m n : ℝ) (h : a^2 - 4*a + 8 = (a - m)^2 + n^2) :
  m * n = 4 ∨ m * n = -4 :=
sorry

/-- Statement 3: Given S = a^2 + 4ab + 5b^2 - 12b + k, prove that S is a perfect number when k = 36 -/
theorem S_is_perfect_number (a b : ℤ) :
  is_perfect_number (a^2 + 4*a*b + 5*b^2 - 12*b + 36) :=
sorry

/-- Statement 4: Given -a^2 + 5a + b - 7 = 0, prove that the minimum value of a + b is 3 -/
theorem min_value_a_plus_b (a b : ℝ) (h : -a^2 + 5*a + b - 7 = 0) :
  a + b ≥ 3 :=
sorry

end NUMINAMATH_CALUDE_perfect_number_examples_mn_value_S_is_perfect_number_min_value_a_plus_b_l312_31228


namespace NUMINAMATH_CALUDE_exp_sum_gt_two_l312_31220

noncomputable section

variables (a : ℝ) (x₁ x₂ : ℝ)

def f (x : ℝ) : ℝ := Real.exp (a * x) - a * (x + 2)

theorem exp_sum_gt_two (ha : a ≠ 0) (hx : x₁ < x₂) 
  (hf₁ : f a x₁ = 0) (hf₂ : f a x₂ = 0) : 
  Real.exp (a * x₁) + Real.exp (a * x₂) > 2 :=
by sorry

end

end NUMINAMATH_CALUDE_exp_sum_gt_two_l312_31220


namespace NUMINAMATH_CALUDE_max_a_is_three_l312_31233

/-- The function f(x) = x^3 - ax is monotonically increasing on [1, +∞) -/
def is_monotone_increasing (a : ℝ) : Prop :=
  ∀ x y, x ≥ 1 → y ≥ 1 → x ≤ y → (x^3 - a*x) ≤ (y^3 - a*y)

/-- The maximum value of 'a' for which f(x) = x^3 - ax is monotonically increasing on [1, +∞) is 3 -/
theorem max_a_is_three :
  (∃ a_max : ℝ, a_max = 3 ∧
    (∀ a : ℝ, is_monotone_increasing a → a ≤ a_max) ∧
    is_monotone_increasing a_max) :=
sorry

end NUMINAMATH_CALUDE_max_a_is_three_l312_31233


namespace NUMINAMATH_CALUDE_mountain_paths_theorem_l312_31281

/-- The number of paths leading to the summit from the east side -/
def east_paths : ℕ := 3

/-- The number of paths leading to the summit from the west side -/
def west_paths : ℕ := 2

/-- The total number of paths leading to the summit -/
def total_paths : ℕ := east_paths + west_paths

/-- The number of different ways for tourists to go up and come down the mountain -/
def different_ways : ℕ := total_paths * total_paths

theorem mountain_paths_theorem : different_ways = 25 := by
  sorry

end NUMINAMATH_CALUDE_mountain_paths_theorem_l312_31281


namespace NUMINAMATH_CALUDE_triplet_transformation_theorem_l312_31210

/-- Represents a triplet of integers -/
structure Triplet where
  a : Int
  b : Int
  c : Int

/-- Represents an operation on a triplet -/
inductive Operation
  | IncrementA (k : Int) (i : Fin 3) : Operation
  | DecrementA (k : Int) (i : Fin 3) : Operation
  | IncrementB (k : Int) (i : Fin 3) : Operation
  | DecrementB (k : Int) (i : Fin 3) : Operation
  | IncrementC (k : Int) (i : Fin 3) : Operation
  | DecrementC (k : Int) (i : Fin 3) : Operation

/-- Applies an operation to a triplet -/
def applyOperation (t : Triplet) (op : Operation) : Triplet :=
  match op with
  | Operation.IncrementA k i => { t with a := t.a + k * (if i = 0 then t.a else if i = 1 then t.b else t.c) }
  | Operation.DecrementA k i => { t with a := t.a - k * (if i = 0 then t.a else if i = 1 then t.b else t.c) }
  | Operation.IncrementB k i => { t with b := t.b + k * (if i = 0 then t.a else if i = 1 then t.b else t.c) }
  | Operation.DecrementB k i => { t with b := t.b - k * (if i = 0 then t.a else if i = 1 then t.b else t.c) }
  | Operation.IncrementC k i => { t with c := t.c + k * (if i = 0 then t.a else if i = 1 then t.b else t.c) }
  | Operation.DecrementC k i => { t with c := t.c - k * (if i = 0 then t.a else if i = 1 then t.b else t.c) }

theorem triplet_transformation_theorem (a b c : Int) (h : Int.gcd a (Int.gcd b c) = 1) :
  ∃ (ops : List Operation), ops.length ≤ 5 ∧
    (ops.foldl applyOperation (Triplet.mk a b c) = Triplet.mk 1 0 0) := by
  sorry

end NUMINAMATH_CALUDE_triplet_transformation_theorem_l312_31210


namespace NUMINAMATH_CALUDE_roller_coaster_height_l312_31230

/-- The required height to ride the roller coaster given Alex's current height,
    normal growth rate, additional growth rate from hanging upside down,
    and the required hanging time. -/
theorem roller_coaster_height
  (current_height : ℝ)
  (normal_growth_rate : ℝ)
  (upside_down_growth_rate : ℝ)
  (hanging_time : ℝ)
  (months_per_year : ℕ)
  (h1 : current_height = 48)
  (h2 : normal_growth_rate = 1 / 3)
  (h3 : upside_down_growth_rate = 1 / 12)
  (h4 : hanging_time = 2)
  (h5 : months_per_year = 12) :
  current_height +
  normal_growth_rate * months_per_year +
  upside_down_growth_rate * hanging_time * months_per_year = 54 :=
by sorry

end NUMINAMATH_CALUDE_roller_coaster_height_l312_31230


namespace NUMINAMATH_CALUDE_cookie_problem_l312_31245

theorem cookie_problem :
  let n : ℕ := 1817
  (∀ m : ℕ, m > 0 ∧ m < n →
    ¬(m % 6 = 5 ∧ m % 7 = 3 ∧ m % 9 = 7 ∧ m % 11 = 10)) ∧
  (n % 6 = 5 ∧ n % 7 = 3 ∧ n % 9 = 7 ∧ n % 11 = 10) := by
  sorry

end NUMINAMATH_CALUDE_cookie_problem_l312_31245


namespace NUMINAMATH_CALUDE_sinA_cosA_rational_l312_31258

/-- An isosceles triangle with integer base and height -/
structure IsoscelesTriangle where
  base : ℤ
  height : ℤ

/-- The sine of angle A in an isosceles triangle -/
def sinA (t : IsoscelesTriangle) : ℚ :=
  4 * t.base * t.height^2 / (4 * t.height^2 + t.base^2)

/-- The cosine of angle A in an isosceles triangle -/
def cosA (t : IsoscelesTriangle) : ℚ :=
  (4 * t.height^2 - t.base^2) / (4 * t.height^2 + t.base^2)

/-- Theorem: In an isosceles triangle with integer base and height, 
    both sin A and cos A are rational numbers -/
theorem sinA_cosA_rational (t : IsoscelesTriangle) : 
  (∃ q : ℚ, sinA t = q) ∧ (∃ q : ℚ, cosA t = q) := by
  sorry

end NUMINAMATH_CALUDE_sinA_cosA_rational_l312_31258


namespace NUMINAMATH_CALUDE_complement_of_A_in_U_l312_31206

def U : Set ℕ := {x : ℕ | x > 0 ∧ (x - 6) * (x + 1) ≤ 0}

def A : Set ℕ := {1, 2, 4}

theorem complement_of_A_in_U :
  (U \ A) = {3, 5, 6} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_in_U_l312_31206


namespace NUMINAMATH_CALUDE_not_p_necessary_not_sufficient_for_q_l312_31280

-- Define propositions p and q
def p (x : ℝ) : Prop := -1 < x ∧ x < 3
def q (x : ℝ) : Prop := x > 5

-- Theorem statement
theorem not_p_necessary_not_sufficient_for_q :
  (∀ x, q x → ¬(p x)) ∧ 
  ¬(∀ x, ¬(p x) → q x) :=
by sorry

end NUMINAMATH_CALUDE_not_p_necessary_not_sufficient_for_q_l312_31280


namespace NUMINAMATH_CALUDE_paint_cans_theorem_l312_31290

/-- Represents the number of rooms that can be painted with the available paint -/
def initialRooms : ℕ := 50

/-- Represents the number of paint cans misplaced -/
def misplacedCans : ℕ := 5

/-- Represents the number of rooms that can be painted after misplacing some cans -/
def remainingRooms : ℕ := 37

/-- Calculates the number of cans used to paint the remaining rooms -/
def cansUsed : ℕ := 15

theorem paint_cans_theorem : 
  ∀ (initial : ℕ) (misplaced : ℕ) (remaining : ℕ),
  initial = initialRooms → 
  misplaced = misplacedCans → 
  remaining = remainingRooms → 
  cansUsed = 15 :=
by sorry

end NUMINAMATH_CALUDE_paint_cans_theorem_l312_31290


namespace NUMINAMATH_CALUDE_two_lines_with_45_degree_angle_l312_31267

/-- Represents a line in 3D space -/
structure Line3D where
  -- Add necessary fields

/-- Represents a plane in 3D space -/
structure Plane3D where
  -- Add necessary fields

/-- Represents a point in 3D space -/
structure Point3D where
  -- Add necessary fields

/-- Calculates the angle between a line and a plane -/
def angle_line_plane (l : Line3D) (p : Plane3D) : Real :=
  sorry

/-- Calculates the angle between two lines -/
def angle_between_lines (l1 l2 : Line3D) : Real :=
  sorry

/-- Checks if a line passes through a point -/
def line_passes_through (l : Line3D) (p : Point3D) : Prop :=
  sorry

theorem two_lines_with_45_degree_angle 
  (a : Line3D) (α : Plane3D) (P : Point3D) 
  (h : angle_line_plane a α = 30) : 
  ∃! (l1 l2 : Line3D), 
    l1 ≠ l2 ∧
    line_passes_through l1 P ∧
    line_passes_through l2 P ∧
    angle_between_lines l1 a = 45 ∧
    angle_between_lines l2 a = 45 ∧
    angle_line_plane l1 α = 45 ∧
    angle_line_plane l2 α = 45 ∧
    (∀ l : Line3D, 
      line_passes_through l P ∧ 
      angle_between_lines l a = 45 ∧ 
      angle_line_plane l α = 45 → 
      l = l1 ∨ l = l2) :=
by
  sorry

end NUMINAMATH_CALUDE_two_lines_with_45_degree_angle_l312_31267
