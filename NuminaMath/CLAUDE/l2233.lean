import Mathlib

namespace NUMINAMATH_CALUDE_equation_solutions_l2233_223390

theorem equation_solutions :
  (∃ x : ℚ, (x + 1) / 2 - (x - 3) / 6 = (5 * x + 1) / 3 + 1 ∧ x = -1/4) ∧
  (∃ x : ℚ, (x - 4) / (1/5) - 1 = (x - 3) / (1/2) ∧ x = 5) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l2233_223390


namespace NUMINAMATH_CALUDE_number_multiplied_by_9999_l2233_223342

theorem number_multiplied_by_9999 : ∃ x : ℚ, x * 9999 = 724787425 ∧ x = 72487.5 := by
  sorry

end NUMINAMATH_CALUDE_number_multiplied_by_9999_l2233_223342


namespace NUMINAMATH_CALUDE_elevator_problem_l2233_223369

theorem elevator_problem :
  let num_elevators : ℕ := 4
  let num_people : ℕ := 3
  let num_same_elevator : ℕ := 2
  (Nat.choose num_people num_same_elevator) * (num_elevators * (num_elevators - 1)) = 36
  := by sorry

end NUMINAMATH_CALUDE_elevator_problem_l2233_223369


namespace NUMINAMATH_CALUDE_gas_pressure_change_l2233_223382

/-- Represents the pressure-volume relationship of a gas at constant temperature -/
structure GasState where
  volume : ℝ
  pressure : ℝ

/-- Verifies if two gas states follow the inverse proportionality law -/
def inverseProportion (s1 s2 : GasState) : Prop :=
  s1.pressure * s1.volume = s2.pressure * s2.volume

theorem gas_pressure_change 
  (initial final : GasState) 
  (h_initial : initial.volume = 3 ∧ initial.pressure = 6) 
  (h_final_volume : final.volume = 4.5) 
  (h_inverse : inverseProportion initial final) : 
  final.pressure = 4 := by
sorry

end NUMINAMATH_CALUDE_gas_pressure_change_l2233_223382


namespace NUMINAMATH_CALUDE_cyclists_speed_product_l2233_223381

theorem cyclists_speed_product (u v : ℝ) : 
  (u > 0) →  -- Assume positive speeds
  (v > 0) →
  (v > u) →  -- Faster cyclist has higher speed
  (6 / u = 6 / v + 1 / 12) →  -- Faster cyclist travels 6 km in 5 minutes less
  (v / 3 = u / 3 + 4) →  -- In 20 minutes, faster cyclist travels 4 km more
  u * v = 864 := by
  sorry

end NUMINAMATH_CALUDE_cyclists_speed_product_l2233_223381


namespace NUMINAMATH_CALUDE_book_pages_ratio_l2233_223345

theorem book_pages_ratio : 
  let selena_pages : ℕ := 400
  let harry_pages : ℕ := 180
  ∃ (a b : ℕ), (a = 9 ∧ b = 20) ∧ 
    (harry_pages : ℚ) / selena_pages = a / b :=
by sorry

end NUMINAMATH_CALUDE_book_pages_ratio_l2233_223345


namespace NUMINAMATH_CALUDE_windows_already_installed_l2233_223376

/-- Proves that the number of windows already installed is 6 -/
theorem windows_already_installed
  (total_windows : ℕ)
  (install_time_per_window : ℕ)
  (time_left : ℕ)
  (h1 : total_windows = 10)
  (h2 : install_time_per_window = 5)
  (h3 : time_left = 20) :
  total_windows - (time_left / install_time_per_window) = 6 := by
  sorry

#check windows_already_installed

end NUMINAMATH_CALUDE_windows_already_installed_l2233_223376


namespace NUMINAMATH_CALUDE_min_sum_squared_distances_l2233_223370

/-- The minimum sum of squared distances from a point on a specific circle to two fixed points -/
theorem min_sum_squared_distances : ∀ (P : ℝ × ℝ),
  (P.1 - 3)^2 + (P.2 - 4)^2 = 4 →
  ∃ (m : ℝ), m = 26 ∧ 
  ∀ (Q : ℝ × ℝ), (Q.1 - 3)^2 + (Q.2 - 4)^2 = 4 →
  (Q.1 + 2)^2 + Q.2^2 + (Q.1 - 2)^2 + Q.2^2 ≥ m :=
by sorry

end NUMINAMATH_CALUDE_min_sum_squared_distances_l2233_223370


namespace NUMINAMATH_CALUDE_min_max_sum_f_l2233_223327

def f (x : ℝ) : ℝ := (x + 1)^5 + (x - 1)^5

theorem min_max_sum_f :
  ∃ (min max : ℝ),
    (∀ x ∈ Set.Icc 0 2, f x ≥ min) ∧
    (∃ x ∈ Set.Icc 0 2, f x = min) ∧
    (∀ x ∈ Set.Icc 0 2, f x ≤ max) ∧
    (∃ x ∈ Set.Icc 0 2, f x = max) ∧
    min = 0 ∧ max = 244 ∧ min + max = 244 := by
  sorry

end NUMINAMATH_CALUDE_min_max_sum_f_l2233_223327


namespace NUMINAMATH_CALUDE_min_expression_bound_l2233_223348

theorem min_expression_bound (x y : ℝ) (hx : 0 < x ∧ x < 1) (hy : 0 < y ∧ y < 1) :
  min
    (min (x^2 + x*y + y^2) (x^2 + x*(y-1) + (y-1)^2))
    (min ((x-1)^2 + (x-1)*y + y^2) ((x-1)^2 + (x-1)*(y-1) + (y-1)^2))
  ≤ 1/3 := by
sorry

end NUMINAMATH_CALUDE_min_expression_bound_l2233_223348


namespace NUMINAMATH_CALUDE_division_problem_l2233_223375

theorem division_problem : 
  (1 / 24) / ((1 / 12) - (5 / 16) + (7 / 24) - (2 / 3)) = -(2 / 29) := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l2233_223375


namespace NUMINAMATH_CALUDE_circle_equation_from_line_l2233_223395

/-- Given a line in polar coordinates that intersects the polar axis, 
    find the polar equation of the circle with the intersection point's diameter --/
theorem circle_equation_from_line (θ : Real) (ρ p : Real → Real) :
  (∀ θ, p θ * Real.cos θ - 2 = 0) →  -- Line equation
  (∃ M : Real × Real, M.1 = 2 ∧ M.2 = 0) →  -- Intersection point
  (∀ θ, ρ θ = 2 * Real.cos θ) :=  -- Circle equation
by sorry

end NUMINAMATH_CALUDE_circle_equation_from_line_l2233_223395


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l2233_223398

def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_properties
  (a : ℕ → ℚ)
  (h_arithmetic : arithmetic_sequence a)
  (h_a1 : a 1 = -3)
  (h_condition : 11 * a 5 = 5 * a 8 - 13) :
  ∃ (d : ℚ) (S : ℕ → ℚ),
    (d = 31 / 9) ∧
    (∀ n : ℕ, S n = n * (2 * a 1 + (n - 1) * d) / 2) ∧
    (∀ n : ℕ, S n ≥ -2401 / 840) ∧
    (S 1 = -2401 / 840) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l2233_223398


namespace NUMINAMATH_CALUDE_painting_distance_l2233_223331

theorem painting_distance (wall_width painting_width : ℝ) 
  (hw : wall_width = 26) 
  (hp : painting_width = 4) : 
  (wall_width - painting_width) / 2 = 11 := by
  sorry

end NUMINAMATH_CALUDE_painting_distance_l2233_223331


namespace NUMINAMATH_CALUDE_perpendicular_vectors_magnitude_l2233_223330

/-- Given vectors a and b in ℝ², if a ⊥ b, then |a| = 2 -/
theorem perpendicular_vectors_magnitude (x : ℝ) :
  let a : ℝ × ℝ := (x, Real.sqrt 3)
  let b : ℝ × ℝ := (3, -Real.sqrt 3)
  (a.1 * b.1 + a.2 * b.2 = 0) →  -- a ⊥ b condition
  Real.sqrt (a.1^2 + a.2^2) = 2 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_magnitude_l2233_223330


namespace NUMINAMATH_CALUDE_base_85_congruence_l2233_223321

theorem base_85_congruence (b : ℤ) (h1 : 0 ≤ b) (h2 : b ≤ 20) 
  (h3 : (74639281 : ℤ) - b ≡ 0 [ZMOD 17]) : b = 1 := by
  sorry

end NUMINAMATH_CALUDE_base_85_congruence_l2233_223321


namespace NUMINAMATH_CALUDE_certain_number_proof_l2233_223335

theorem certain_number_proof (x : ℚ) : 
  x^22 * (1/81)^11 = 1/18^22 → x = 1/36 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l2233_223335


namespace NUMINAMATH_CALUDE_negative_fraction_comparison_l2233_223364

theorem negative_fraction_comparison : -6/5 > -5/4 := by
  sorry

end NUMINAMATH_CALUDE_negative_fraction_comparison_l2233_223364


namespace NUMINAMATH_CALUDE_last_digit_of_product_l2233_223358

def last_digit (n : ℤ) : ℕ := n.natAbs % 10

theorem last_digit_of_product (B : ℤ) : 
  B ≥ 0 ∧ B ≤ 9 →
  (last_digit (287 * 287 + B * B - 2 * 287 * B) = 4 ↔ B = 5 ∨ B = 9) :=
by sorry

end NUMINAMATH_CALUDE_last_digit_of_product_l2233_223358


namespace NUMINAMATH_CALUDE_hockey_pads_cost_l2233_223310

def initial_amount : ℕ := 150
def remaining_amount : ℕ := 25

def cost_of_skates : ℕ := initial_amount / 2

def cost_of_pads : ℕ := initial_amount - cost_of_skates - remaining_amount

theorem hockey_pads_cost : cost_of_pads = 50 := by
  sorry

end NUMINAMATH_CALUDE_hockey_pads_cost_l2233_223310


namespace NUMINAMATH_CALUDE_smallest_c_value_l2233_223304

theorem smallest_c_value (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (∀ x, a * Real.cos (b * x + c) ≤ a * Real.cos (b * (-π/4) + c)) →
  c ≥ π/4 :=
sorry

end NUMINAMATH_CALUDE_smallest_c_value_l2233_223304


namespace NUMINAMATH_CALUDE_train_journey_time_l2233_223346

/-- Proves that given a train moving at 6/7 of its usual speed and arriving 30 minutes late, 
    the usual time for the train to complete the journey is 3 hours. -/
theorem train_journey_time (usual_speed : ℝ) (usual_time : ℝ) : 
  usual_speed > 0 → usual_time > 0 →
  (6 / 7 * usual_speed) * (usual_time + 1 / 2) = usual_speed * usual_time →
  usual_time = 3 := by
  sorry

end NUMINAMATH_CALUDE_train_journey_time_l2233_223346


namespace NUMINAMATH_CALUDE_discounted_price_calculation_l2233_223316

theorem discounted_price_calculation (original_price discount_percentage : ℝ) :
  original_price = 600 ∧ discount_percentage = 20 →
  original_price * (1 - discount_percentage / 100) = 480 := by
sorry

end NUMINAMATH_CALUDE_discounted_price_calculation_l2233_223316


namespace NUMINAMATH_CALUDE_triangle_proof_l2233_223392

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_proof (t : Triangle) 
  (h1 : (t.a - t.b + t.c) / t.c = t.b / (t.a + t.b - t.c))
  (h2 : t.b - t.c = (Real.sqrt 3 / 3) * t.a) :
  t.A = π / 3 ∧ t.B = π / 2 := by
  sorry


end NUMINAMATH_CALUDE_triangle_proof_l2233_223392


namespace NUMINAMATH_CALUDE_odometer_reading_l2233_223356

theorem odometer_reading (initial_reading traveled_distance : ℝ) 
  (h1 : initial_reading = 212.3)
  (h2 : traveled_distance = 159.7) :
  initial_reading + traveled_distance = 372.0 := by
sorry

end NUMINAMATH_CALUDE_odometer_reading_l2233_223356


namespace NUMINAMATH_CALUDE_continued_fraction_evaluation_l2233_223307

theorem continued_fraction_evaluation :
  2 + (3 / (4 + (5/6))) = 76/29 := by
  sorry

end NUMINAMATH_CALUDE_continued_fraction_evaluation_l2233_223307


namespace NUMINAMATH_CALUDE_cubes_with_five_neighbors_count_l2233_223391

/-- Represents a large cube assembled from unit cubes -/
structure LargeCube where
  sideLength : ℕ

/-- The number of unit cubes with exactly 4 neighbors in the large cube -/
def cubesWithFourNeighbors (c : LargeCube) : ℕ := 12 * (c.sideLength - 2)

/-- The number of unit cubes with exactly 5 neighbors in the large cube -/
def cubesWithFiveNeighbors (c : LargeCube) : ℕ := 6 * (c.sideLength - 2)^2

/-- Theorem stating the relationship between cubes with 4 and 5 neighbors -/
theorem cubes_with_five_neighbors_count (c : LargeCube) 
  (h : cubesWithFourNeighbors c = 132) : 
  cubesWithFiveNeighbors c = 726 := by
  sorry

end NUMINAMATH_CALUDE_cubes_with_five_neighbors_count_l2233_223391


namespace NUMINAMATH_CALUDE_quadratic_decomposition_l2233_223396

theorem quadratic_decomposition :
  ∃ (k : ℤ) (a : ℝ), ∀ y : ℝ, y^2 + 14*y + 60 = (y + a)^2 + k ∧ k = 11 := by
sorry

end NUMINAMATH_CALUDE_quadratic_decomposition_l2233_223396


namespace NUMINAMATH_CALUDE_count_threes_up_to_80_l2233_223352

/-- Count of digit 3 in a single number -/
def countThreesInNumber (n : ℕ) : ℕ :=
  if n = 0 then 0
  else if n % 10 = 3 then 1 + countThreesInNumber (n / 10)
  else countThreesInNumber (n / 10)

/-- Count of digit 3 in numbers from 1 to n -/
def countThreesUpTo (n : ℕ) : ℕ :=
  List.range n |> List.map (fun i => countThreesInNumber (i + 1)) |> List.sum

/-- The count of the digit 3 in the numbers from 1 to 80 (inclusive) is equal to 9 -/
theorem count_threes_up_to_80 : countThreesUpTo 80 = 9 := by
  sorry

end NUMINAMATH_CALUDE_count_threes_up_to_80_l2233_223352


namespace NUMINAMATH_CALUDE_at_least_one_hit_l2233_223377

theorem at_least_one_hit (p_A p_B p_C : ℝ) 
  (h_A : p_A = 1/2) 
  (h_B : p_B = 1/3) 
  (h_C : p_C = 1/4) : 
  1 - (1 - p_A) * (1 - p_B) * (1 - p_C) = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_hit_l2233_223377


namespace NUMINAMATH_CALUDE_max_value_of_sin_cos_function_l2233_223365

theorem max_value_of_sin_cos_function :
  ∃ (M : ℝ), M = 17 ∧ ∀ x, 8 * Real.sin x + 15 * Real.cos x ≤ M := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_sin_cos_function_l2233_223365


namespace NUMINAMATH_CALUDE_unique_difference_of_squares_1979_l2233_223367

theorem unique_difference_of_squares_1979 : 
  ∃! (x y : ℕ), 1979 = x^2 - y^2 :=
by
  sorry

end NUMINAMATH_CALUDE_unique_difference_of_squares_1979_l2233_223367


namespace NUMINAMATH_CALUDE_range_of_m_l2233_223397

theorem range_of_m (p q : Prop) (h1 : p ↔ ∀ x : ℝ, x^2 - 2*x + 1 - m ≥ 0)
  (h2 : q ↔ ∃ a b : ℝ, a ≠ 0 ∧ b ≠ 0 ∧ ∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1 ∧ a^2 = 1 ∧ b^2 = 1 / (m + 2))
  (h3 : (p ∨ q) ∧ ¬(p ∧ q)) :
  m ≤ -2 ∨ m > 0 := by sorry

end NUMINAMATH_CALUDE_range_of_m_l2233_223397


namespace NUMINAMATH_CALUDE_cubic_extrema_difference_l2233_223336

open Real

/-- The cubic function f(x) with parameters a and b. -/
def f (a b x : ℝ) : ℝ := x^3 + 3*a*x^2 + 3*b*x

/-- The derivative of f(x) with respect to x. -/
def f' (a b x : ℝ) : ℝ := 3*x^2 + 6*a*x + 3*b

theorem cubic_extrema_difference (a b : ℝ) 
  (h1 : f' a b 2 = 0)
  (h2 : f' a b 1 = -3) :
  ∃ (x_max x_min : ℝ), 
    (∀ x, f a b x ≤ f a b x_max) ∧ 
    (∀ x, f a b x_min ≤ f a b x) ∧
    (f a b x_max - f a b x_min = 4) := by
  sorry

end NUMINAMATH_CALUDE_cubic_extrema_difference_l2233_223336


namespace NUMINAMATH_CALUDE_mets_to_red_sox_ratio_l2233_223363

/-- Represents the number of fans for each team -/
structure FanCounts where
  yankees : ℕ
  mets : ℕ
  red_sox : ℕ

/-- The ratio of two natural numbers -/
structure Ratio where
  numerator : ℕ
  denominator : ℕ

/-- The given conditions of the problem -/
def fan_problem (fc : FanCounts) : Prop :=
  fc.yankees * 2 = fc.mets * 3 ∧  -- Ratio of Yankees to Mets is 3:2
  fc.yankees + fc.mets + fc.red_sox = 330 ∧  -- Total fans
  fc.mets = 88  -- Number of Mets fans

/-- The theorem to prove -/
theorem mets_to_red_sox_ratio 
  (fc : FanCounts) 
  (h : fan_problem fc) : 
  ∃ (r : Ratio), r.numerator = 4 ∧ r.denominator = 5 ∧
  r.numerator * fc.red_sox = r.denominator * fc.mets :=
sorry

end NUMINAMATH_CALUDE_mets_to_red_sox_ratio_l2233_223363


namespace NUMINAMATH_CALUDE_positive_real_inequality_l2233_223339

theorem positive_real_inequality (x : ℝ) (h : x > 0) : x + 1/x ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_positive_real_inequality_l2233_223339


namespace NUMINAMATH_CALUDE_calculate_interest_rate_l2233_223362

/-- Given a total sum and two parts with specific interest conditions, 
    calculate the interest rate for the second part. -/
theorem calculate_interest_rate 
  (total_sum : ℚ) 
  (second_part : ℚ) 
  (first_part_years : ℚ) 
  (first_part_rate : ℚ) 
  (second_part_years : ℚ) 
  (h1 : total_sum = 2730) 
  (h2 : second_part = 1680) 
  (h3 : first_part_years = 8) 
  (h4 : first_part_rate = 3 / 100) 
  (h5 : second_part_years = 3) 
  (h6 : (total_sum - second_part) * first_part_rate * first_part_years = 
        second_part * (second_part_years * x) ) :
  x = 5 / 100 := by
  sorry

end NUMINAMATH_CALUDE_calculate_interest_rate_l2233_223362


namespace NUMINAMATH_CALUDE_three_number_difference_l2233_223366

theorem three_number_difference (x y : ℝ) (h : (23 + x + y) / 3 = 31) :
  max (max 23 x) y - min (min 23 x) y ≥ 17 :=
sorry

end NUMINAMATH_CALUDE_three_number_difference_l2233_223366


namespace NUMINAMATH_CALUDE_gasoline_added_to_tank_l2233_223328

/-- The amount of gasoline added to a tank -/
def gasoline_added (total_capacity : ℝ) (initial_fraction : ℝ) (final_fraction : ℝ) : ℝ :=
  total_capacity * (final_fraction - initial_fraction)

/-- Proof that 7.2 gallons of gasoline were added to the tank -/
theorem gasoline_added_to_tank : 
  gasoline_added 48 (3/4) (9/10) = 7.2 := by
sorry

end NUMINAMATH_CALUDE_gasoline_added_to_tank_l2233_223328


namespace NUMINAMATH_CALUDE_sugar_amount_proof_l2233_223360

/-- Recipe proportions and conversion factors -/
def butter_to_flour : ℚ := 5 / 7
def salt_to_flour : ℚ := 3 / 1.5
def sugar_to_flour : ℚ := 2 / 2.5
def butter_multiplier : ℚ := 4
def salt_multiplier : ℚ := 3.5
def sugar_multiplier : ℚ := 3
def butter_used : ℚ := 12
def ounce_to_gram : ℚ := 28.35
def cup_flour_to_gram : ℚ := 125
def tsp_salt_to_gram : ℚ := 5
def tbsp_sugar_to_gram : ℚ := 15

/-- Theorem stating that the amount of sugar needed is 604.8 grams -/
theorem sugar_amount_proof :
  let flour_cups := butter_used / butter_to_flour
  let flour_grams := flour_cups * cup_flour_to_gram
  let sugar_tbsp := (sugar_to_flour * flour_cups * sugar_multiplier)
  sugar_tbsp * tbsp_sugar_to_gram = 604.8 := by
  sorry

end NUMINAMATH_CALUDE_sugar_amount_proof_l2233_223360


namespace NUMINAMATH_CALUDE_problem_solution_l2233_223317

theorem problem_solution (x y : ℚ) : 
  x = 103 → x^3 * y - 2 * x^2 * y + x * y = 1060900 → y = 100 / 101 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2233_223317


namespace NUMINAMATH_CALUDE_sin_double_angle_special_point_l2233_223320

/-- Given an angle θ in standard position with its terminal side passing through the point (1, -2),
    prove that sin(2θ) = -4/5 -/
theorem sin_double_angle_special_point :
  ∀ θ : Real,
  (∃ (r : Real), r > 0 ∧ r * Real.cos θ = 1 ∧ r * Real.sin θ = -2) →
  Real.sin (2 * θ) = -4/5 := by
sorry

end NUMINAMATH_CALUDE_sin_double_angle_special_point_l2233_223320


namespace NUMINAMATH_CALUDE_otimes_self_otimes_self_l2233_223300

/-- Custom binary operation ⊗ -/
def otimes (x y : ℝ) : ℝ := x^2 - y^2

/-- Theorem: h ⊗ (h ⊗ h) = h^2 for any real number h -/
theorem otimes_self_otimes_self (h : ℝ) : otimes h (otimes h h) = h^2 := by
  sorry

end NUMINAMATH_CALUDE_otimes_self_otimes_self_l2233_223300


namespace NUMINAMATH_CALUDE_problem_statement_l2233_223355

theorem problem_statement (a b : ℝ) 
  (h1 : a^2 * b - a * b^2 = -6) 
  (h2 : a * b = 3) : 
  a - b = -2 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2233_223355


namespace NUMINAMATH_CALUDE_find_n_l2233_223361

def A (i : ℕ) : ℕ := 2 * i - 1

def B (n i : ℕ) : ℕ := n - 2 * (i - 1)

theorem find_n : ∃ n : ℕ, 
  (∃ k : ℕ, A k = 19 ∧ B n k = 89) → n = 107 := by
  sorry

end NUMINAMATH_CALUDE_find_n_l2233_223361


namespace NUMINAMATH_CALUDE_bluejay_female_fraction_l2233_223399

theorem bluejay_female_fraction (total_birds : ℝ) (total_birds_pos : 0 < total_birds) : 
  let robins := (2/5) * total_birds
  let bluejays := (3/5) * total_birds
  let female_robins := (1/3) * robins
  let male_birds := (7/15) * total_birds
  let female_bluejays := ((8/15) * total_birds) - female_robins
  (female_bluejays / bluejays) = (2/3) :=
by sorry

end NUMINAMATH_CALUDE_bluejay_female_fraction_l2233_223399


namespace NUMINAMATH_CALUDE_car_trip_speed_l2233_223315

/-- Proves that the speed for the remaining part of the trip is 20 mph given the conditions of the problem -/
theorem car_trip_speed (x t : ℝ) (h1 : x > 0) (h2 : t > 0) : ∃ s : ℝ,
  (0.75 * x / 60 + 0.25 * x / s = t) ∧ 
  (x / t = 40) →
  s = 20 := by
  sorry


end NUMINAMATH_CALUDE_car_trip_speed_l2233_223315


namespace NUMINAMATH_CALUDE_cube_root_equation_product_l2233_223350

theorem cube_root_equation_product (a b : ℤ) : 
  (3 * Real.sqrt (Real.rpow 5 (1/3) - Real.rpow 4 (1/3)) = Real.rpow a (1/3) + Real.rpow b (1/3) + Real.rpow 2 (1/3)) →
  a * b = -500 := by
sorry

end NUMINAMATH_CALUDE_cube_root_equation_product_l2233_223350


namespace NUMINAMATH_CALUDE_range_of_m_l2233_223374

-- Define proposition p
def p (m : ℝ) : Prop :=
  ∀ x : ℝ, x ≠ 0 → (x^4 - x^2 + 1) / x^2 > m

-- Define proposition q
def q (m : ℝ) : Prop :=
  ∀ x y : ℝ, x < y → (-(5-2*m))^y < (-(5-2*m))^x

-- Define the theorem
theorem range_of_m :
  (∀ m : ℝ, (p m ∨ q m)) ∧ (¬∀ m : ℝ, (p m ∧ q m)) →
  ∃ a b : ℝ, a = 1 ∧ b = 2 ∧ ∀ m : ℝ, a ≤ m ∧ m < b :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l2233_223374


namespace NUMINAMATH_CALUDE_intersection_point_l2233_223301

-- Define the line using a parameter t
def line (t : ℝ) : ℝ × ℝ × ℝ := (1 - 2*t, 2 + t, -1 - t)

-- Define the plane equation
def plane (p : ℝ × ℝ × ℝ) : Prop :=
  let (x, y, z) := p
  x - 2*y + 5*z + 17 = 0

-- Theorem statement
theorem intersection_point :
  ∃! p : ℝ × ℝ × ℝ, (∃ t : ℝ, line t = p) ∧ plane p ∧ p = (-1, 3, -2) := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_l2233_223301


namespace NUMINAMATH_CALUDE_total_is_99_l2233_223329

/-- The total number of ducks and ducklings in a flock --/
def total_ducks_and_ducklings : ℕ → ℕ → ℕ → ℕ := fun a b c => 
  (2 + 6 + 9) + (2 * a + 6 * b + 9 * c)

/-- Theorem: The total number of ducks and ducklings is 99 --/
theorem total_is_99 : total_ducks_and_ducklings 5 3 6 = 99 := by
  sorry

end NUMINAMATH_CALUDE_total_is_99_l2233_223329


namespace NUMINAMATH_CALUDE_F_range_l2233_223305

def F (x : ℝ) : ℝ := |x + 2| - |x - 2|

theorem F_range : Set.range F = Set.Ici 4 := by sorry

end NUMINAMATH_CALUDE_F_range_l2233_223305


namespace NUMINAMATH_CALUDE_intersection_empty_iff_a_in_range_union_equals_B_iff_a_in_range_l2233_223384

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {x | a ≤ x ∧ x ≤ a + 3}
def B : Set ℝ := {x | x < -6 ∨ x > 1}

-- Theorem for part (I)
theorem intersection_empty_iff_a_in_range (a : ℝ) :
  A a ∩ B = ∅ ↔ -6 ≤ a ∧ a ≤ -2 := by sorry

-- Theorem for part (II)
theorem union_equals_B_iff_a_in_range (a : ℝ) :
  A a ∪ B = B ↔ a < -9 ∨ a > 1 := by sorry

end NUMINAMATH_CALUDE_intersection_empty_iff_a_in_range_union_equals_B_iff_a_in_range_l2233_223384


namespace NUMINAMATH_CALUDE_edward_spent_sixteen_l2233_223359

def edward_book_purchase (initial_amount : ℕ) (remaining_amount : ℕ) (num_books : ℕ) : Prop :=
  ∃ (amount_spent : ℕ), 
    initial_amount = remaining_amount + amount_spent ∧
    amount_spent = 16

theorem edward_spent_sixteen : 
  edward_book_purchase 22 6 92 :=
sorry

end NUMINAMATH_CALUDE_edward_spent_sixteen_l2233_223359


namespace NUMINAMATH_CALUDE_gcd_digits_bound_l2233_223337

theorem gcd_digits_bound (a b : ℕ) : 
  1000000 ≤ a ∧ a < 10000000 ∧
  1000000 ≤ b ∧ b < 10000000 ∧
  1000000000000 ≤ Nat.lcm a b ∧ Nat.lcm a b < 10000000000000 →
  Nat.gcd a b < 100 := by
sorry

end NUMINAMATH_CALUDE_gcd_digits_bound_l2233_223337


namespace NUMINAMATH_CALUDE_sum_product_squares_ratio_l2233_223312

theorem sum_product_squares_ratio (x y z a : ℝ) (h1 : x ≠ y ∧ y ≠ z ∧ x ≠ z) (h2 : x + y + z = a) (h3 : a ≠ 0) :
  (x * y + y * z + z * x) / (x^2 + y^2 + z^2) = 1/3 := by
sorry

end NUMINAMATH_CALUDE_sum_product_squares_ratio_l2233_223312


namespace NUMINAMATH_CALUDE_water_speed_calculation_l2233_223326

/-- The speed of water in a river where a person who can swim at 12 km/h in still water
    takes 1 hour to swim 10 km against the current. -/
def water_speed : ℝ := 2

theorem water_speed_calculation (still_water_speed : ℝ) (distance : ℝ) (time : ℝ) 
  (h1 : still_water_speed = 12)
  (h2 : distance = 10)
  (h3 : time = 1)
  (h4 : distance / time = still_water_speed - water_speed) : 
  water_speed = 2 := by
  sorry

#check water_speed_calculation

end NUMINAMATH_CALUDE_water_speed_calculation_l2233_223326


namespace NUMINAMATH_CALUDE_triangle_sum_special_case_l2233_223343

def triangle_sum (a b : ℕ) : ℕ :=
  let n_min := a.max b - (a.min b - 1)
  let n_max := a + b - 1
  (n_max - n_min + 1) * (n_max + n_min) / 2

theorem triangle_sum_special_case : triangle_sum 7 10 = 260 := by
  sorry

end NUMINAMATH_CALUDE_triangle_sum_special_case_l2233_223343


namespace NUMINAMATH_CALUDE_intersection_M_N_l2233_223308

def M : Set ℝ := {x | |x| ≤ 2}
def N : Set ℝ := {-1, 0, 2, 3}

theorem intersection_M_N : M ∩ N = {-1, 0, 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l2233_223308


namespace NUMINAMATH_CALUDE_triangle_abc_is_acute_l2233_223388

/-- A triangle is acute if all its angles are less than 90 degrees --/
def IsAcuteTriangle (a b c : ℝ) : Prop :=
  let cosA := (b^2 + c^2 - a^2) / (2*b*c)
  let cosB := (a^2 + c^2 - b^2) / (2*a*c)
  let cosC := (a^2 + b^2 - c^2) / (2*a*b)
  0 < cosA ∧ cosA < 1 ∧
  0 < cosB ∧ cosB < 1 ∧
  0 < cosC ∧ cosC < 1

theorem triangle_abc_is_acute :
  let a : ℝ := 9
  let b : ℝ := 10
  let c : ℝ := 12
  IsAcuteTriangle a b c := by
  sorry

end NUMINAMATH_CALUDE_triangle_abc_is_acute_l2233_223388


namespace NUMINAMATH_CALUDE_fraction_simplification_l2233_223322

theorem fraction_simplification : 
  (3+6-12+24+48-96+192) / (6+12-24+48+96-192+384) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2233_223322


namespace NUMINAMATH_CALUDE_davids_physics_marks_l2233_223393

def marks_english : ℕ := 76
def marks_mathematics : ℕ := 65
def marks_chemistry : ℕ := 67
def marks_biology : ℕ := 85
def average_marks : ℕ := 75
def num_subjects : ℕ := 5

theorem davids_physics_marks :
  ∃ (marks_physics : ℕ),
    marks_physics = average_marks * num_subjects - (marks_english + marks_mathematics + marks_chemistry + marks_biology) ∧
    marks_physics = 82 := by
  sorry

end NUMINAMATH_CALUDE_davids_physics_marks_l2233_223393


namespace NUMINAMATH_CALUDE_nested_subtraction_1999_always_true_l2233_223302

/-- The nested subtraction function with n levels of nesting -/
def nestedSubtraction (x : ℝ) : ℕ → ℝ
  | 0 => x - 1
  | n + 1 => x - nestedSubtraction x n

/-- Theorem stating that for 1999 levels of nesting, the equation is always true for any real x -/
theorem nested_subtraction_1999_always_true (x : ℝ) :
  nestedSubtraction x 1999 = 1 := by
  sorry

#check nested_subtraction_1999_always_true

end NUMINAMATH_CALUDE_nested_subtraction_1999_always_true_l2233_223302


namespace NUMINAMATH_CALUDE_geometric_sequence_middle_term_l2233_223325

theorem geometric_sequence_middle_term 
  (a b c : ℝ) 
  (h_seq : ∃ r : ℝ, b = a * r ∧ c = b * r) 
  (h_a : a = 7 + 4 * Real.sqrt 3) 
  (h_c : c = 7 - 4 * Real.sqrt 3) : 
  b = 1 ∨ b = -1 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_middle_term_l2233_223325


namespace NUMINAMATH_CALUDE_regression_analysis_properties_l2233_223378

-- Define the basic concepts
def FunctionRelationship : Type := Unit
def CorrelationRelationship : Type := Unit
def RegressionAnalysis : Type := Unit

-- Define properties
def isDeterministic (r : Type) : Prop := sorry
def isNonDeterministic (r : Type) : Prop := sorry
def usedFor (a : Type) (r : Type) : Prop := sorry

-- Theorem statement
theorem regression_analysis_properties :
  isDeterministic FunctionRelationship ∧
  isNonDeterministic CorrelationRelationship ∧
  usedFor RegressionAnalysis CorrelationRelationship :=
by sorry

end NUMINAMATH_CALUDE_regression_analysis_properties_l2233_223378


namespace NUMINAMATH_CALUDE_terminal_side_half_angle_l2233_223380

-- Define a function to determine the quadrant of an angle
def quadrant (θ : ℝ) : Set Nat :=
  if 0 < θ % (2 * Real.pi) ∧ θ % (2 * Real.pi) < Real.pi / 2 then {1}
  else if Real.pi / 2 < θ % (2 * Real.pi) ∧ θ % (2 * Real.pi) < Real.pi then {2}
  else if Real.pi < θ % (2 * Real.pi) ∧ θ % (2 * Real.pi) < 3 * Real.pi / 2 then {3}
  else {4}

-- Theorem statement
theorem terminal_side_half_angle (α : ℝ) :
  quadrant α = {3} → quadrant (α / 2) = {2} ∨ quadrant (α / 2) = {4} := by
  sorry

end NUMINAMATH_CALUDE_terminal_side_half_angle_l2233_223380


namespace NUMINAMATH_CALUDE_sum_of_possible_distances_l2233_223303

theorem sum_of_possible_distances (a b c d : ℝ) 
  (hab : |a - b| = 2)
  (hbc : |b - c| = 3)
  (hcd : |c - d| = 4) :
  ∃ S : Finset ℝ, (∀ x ∈ S, ∃ a' b' c' d' : ℝ, 
    |a' - b'| = 2 ∧ |b' - c'| = 3 ∧ |c' - d'| = 4 ∧ |a' - d'| = x) ∧
  (∀ y : ℝ, (∃ a' b' c' d' : ℝ, 
    |a' - b'| = 2 ∧ |b' - c'| = 3 ∧ |c' - d'| = 4 ∧ |a' - d'| = y) → y ∈ S) ∧
  S.sum id = 18 :=
sorry

end NUMINAMATH_CALUDE_sum_of_possible_distances_l2233_223303


namespace NUMINAMATH_CALUDE_certain_number_proof_l2233_223313

theorem certain_number_proof (N : ℝ) : (5/6) * N = (5/16) * N + 50 → N = 96 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l2233_223313


namespace NUMINAMATH_CALUDE_regular_octagon_interior_angle_regular_octagon_interior_angle_is_135_l2233_223353

theorem regular_octagon_interior_angle : ℝ :=
  let n : ℕ := 8  -- number of sides in an octagon
  let sum_of_interior_angles : ℝ := 180 * (n - 2)
  let one_interior_angle : ℝ := sum_of_interior_angles / n
  135

/-- The measure of one interior angle of a regular octagon is 135 degrees. -/
theorem regular_octagon_interior_angle_is_135 :
  regular_octagon_interior_angle = 135 := by
  sorry

end NUMINAMATH_CALUDE_regular_octagon_interior_angle_regular_octagon_interior_angle_is_135_l2233_223353


namespace NUMINAMATH_CALUDE_system_solution_l2233_223347

theorem system_solution (x₁ x₂ x₃ x₄ : ℝ) : 
  x₁^2 + x₂^2 + x₃^2 + x₄^2 = 4 ∧
  x₁*x₃ + x₂*x₄ + x₃*x₂ + x₄*x₁ = 0 ∧
  x₁*x₂*x₃ + x₁*x₂*x₄ + x₁*x₃*x₄ + x₂*x₃*x₄ = -2 ∧
  x₁*x₂*x₃*x₄ = -1 →
  (x₁ = 1 ∧ x₂ = 1 ∧ x₃ = 1 ∧ x₄ = -1) ∨
  (x₁ = 1 ∧ x₂ = 1 ∧ x₃ = -1 ∧ x₄ = 1) ∨
  (x₁ = 1 ∧ x₂ = -1 ∧ x₃ = 1 ∧ x₄ = 1) ∨
  (x₁ = -1 ∧ x₂ = 1 ∧ x₃ = 1 ∧ x₄ = 1) :=
by sorry


end NUMINAMATH_CALUDE_system_solution_l2233_223347


namespace NUMINAMATH_CALUDE_age_of_b_l2233_223351

/-- Given three people A, B, and C, their average age, and the average age of A and C, prove the age of B. -/
theorem age_of_b (a b c : ℕ) : 
  (a + b + c) / 3 = 27 →  -- The average age of A, B, and C is 27
  (a + c) / 2 = 29 →      -- The average age of A and C is 29
  b = 23 :=               -- The age of B is 23
by sorry

end NUMINAMATH_CALUDE_age_of_b_l2233_223351


namespace NUMINAMATH_CALUDE_instantaneous_velocity_at_3_l2233_223332

/-- The position function of a particle -/
def position (t : ℝ) : ℝ := t^3 - 2*t

/-- The velocity function of a particle -/
def velocity (t : ℝ) : ℝ := 3*t^2 - 2

theorem instantaneous_velocity_at_3 : velocity 3 = 25 := by
  sorry

end NUMINAMATH_CALUDE_instantaneous_velocity_at_3_l2233_223332


namespace NUMINAMATH_CALUDE_no_intersection_l2233_223324

/-- Represents a 2D point or vector -/
structure Vec2D where
  x : ℝ
  y : ℝ

/-- Represents a parametric line in 2D -/
structure ParamLine where
  origin : Vec2D
  direction : Vec2D

/-- The first line -/
def line1 : ParamLine :=
  { origin := { x := 1, y := 4 }
    direction := { x := -2, y := 6 } }

/-- The second line -/
def line2 : ParamLine :=
  { origin := { x := 3, y := 10 }
    direction := { x := -1, y := 3 } }

/-- Checks if two parametric lines intersect -/
def linesIntersect (l1 l2 : ParamLine) : Prop :=
  ∃ (s t : ℝ), l1.origin.x + s * l1.direction.x = l2.origin.x + t * l2.direction.x ∧
                l1.origin.y + s * l1.direction.y = l2.origin.y + t * l2.direction.y

theorem no_intersection : ¬ linesIntersect line1 line2 := by
  sorry

end NUMINAMATH_CALUDE_no_intersection_l2233_223324


namespace NUMINAMATH_CALUDE_total_spent_on_tickets_l2233_223314

def this_year_prices : List ℕ := [35, 45, 50, 62]
def last_year_prices : List ℕ := [25, 30, 40, 45, 55, 60, 65, 70, 75]

theorem total_spent_on_tickets : 
  (this_year_prices.sum + last_year_prices.sum : ℕ) = 657 := by
  sorry

end NUMINAMATH_CALUDE_total_spent_on_tickets_l2233_223314


namespace NUMINAMATH_CALUDE_carries_tshirt_purchase_l2233_223387

/-- The cost of a single t-shirt in dollars -/
def cost_per_shirt : ℝ := 9.95

/-- The number of t-shirts Carrie bought -/
def num_shirts : ℕ := 25

/-- The total cost of Carrie's t-shirt purchase -/
def total_cost : ℝ := cost_per_shirt * (num_shirts : ℝ)

theorem carries_tshirt_purchase :
  total_cost = 248.75 := by sorry

end NUMINAMATH_CALUDE_carries_tshirt_purchase_l2233_223387


namespace NUMINAMATH_CALUDE_min_value_of_f_l2233_223306

-- Define the function
def f (x : ℝ) : ℝ := 4 * x^2 * (x - 2)

-- State the theorem
theorem min_value_of_f :
  ∃ (m : ℝ), (∀ x ∈ Set.Icc (-2 : ℝ) 2, m ≤ f x) ∧ (∃ x ∈ Set.Icc (-2 : ℝ) 2, f x = m) ∧ m = -64 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_f_l2233_223306


namespace NUMINAMATH_CALUDE_circular_arrangement_l2233_223311

/-- 
Given a circular arrangement of n people numbered 1 to n,
if the distance from person 31 to person 7 is equal to 
the distance from person 31 to person 14, then n = 41.
-/
theorem circular_arrangement (n : ℕ) : 
  n ≥ 31 → 
  (min ((7 - 31 + n) % n) ((31 - 7) % n) = min ((14 - 31 + n) % n) ((31 - 14) % n)) → 
  n = 41 := by
  sorry

end NUMINAMATH_CALUDE_circular_arrangement_l2233_223311


namespace NUMINAMATH_CALUDE_percent_decrease_proof_l2233_223341

theorem percent_decrease_proof (original_price sale_price : ℝ) 
  (h1 : original_price = 100)
  (h2 : sale_price = 50) :
  (original_price - sale_price) / original_price * 100 = 50 := by
  sorry

end NUMINAMATH_CALUDE_percent_decrease_proof_l2233_223341


namespace NUMINAMATH_CALUDE_equation_solution_l2233_223323

theorem equation_solution :
  let f (y : ℝ) := (8 * y^2 + 40 * y - 48) / (3 * y + 9) - (4 * y - 8)
  ∀ y : ℝ, f y = 0 ↔ y = (7 + Real.sqrt 73) / 2 ∨ y = (7 - Real.sqrt 73) / 2 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l2233_223323


namespace NUMINAMATH_CALUDE_gomoku_piece_count_l2233_223385

/-- Represents the number of pieces in the Gomoku game box -/
structure GomokuBox where
  initial_black : ℕ
  initial_white : ℕ
  added_black : ℕ
  added_white : ℕ

/-- Theorem statement for the Gomoku piece counting problem -/
theorem gomoku_piece_count (box : GomokuBox) : 
  box.initial_black = box.initial_white ∧ 
  box.initial_black + box.initial_white ≤ 10 ∧
  box.added_black + box.added_white = 20 ∧
  7 * (box.initial_white + box.added_white) = 8 * (box.initial_black + box.added_black) →
  box.initial_black + box.added_black = 16 := by
  sorry

end NUMINAMATH_CALUDE_gomoku_piece_count_l2233_223385


namespace NUMINAMATH_CALUDE_max_silver_tokens_l2233_223386

/-- Represents the state of tokens --/
structure TokenState :=
  (red : ℕ)
  (blue : ℕ)
  (silver : ℕ)

/-- Represents an exchange at a booth --/
inductive Exchange
  | RedToSilver
  | BlueToSilver

/-- Applies an exchange to the current state --/
def applyExchange (state : TokenState) (ex : Exchange) : TokenState :=
  match ex with
  | Exchange.RedToSilver => 
      if state.red ≥ 3 then
        TokenState.mk (state.red - 3) (state.blue + 2) (state.silver + 1)
      else
        state
  | Exchange.BlueToSilver => 
      if state.blue ≥ 4 then
        TokenState.mk (state.red + 2) (state.blue - 4) (state.silver + 1)
      else
        state

/-- Checks if any exchange is possible --/
def canExchange (state : TokenState) : Bool :=
  state.red ≥ 3 ∨ state.blue ≥ 4

/-- Theorem: The maximum number of silver tokens Alex can obtain is 131 --/
theorem max_silver_tokens : 
  ∃ (exchanges : List Exchange), 
    let finalState := exchanges.foldl applyExchange (TokenState.mk 100 100 0)
    ¬(canExchange finalState) ∧ finalState.silver = 131 := by
  sorry

end NUMINAMATH_CALUDE_max_silver_tokens_l2233_223386


namespace NUMINAMATH_CALUDE_coordinate_conditions_l2233_223368

theorem coordinate_conditions (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) 
  (h₁ : x₁ = 4 * π / 5 ∧ y₁ = -π / 5)
  (h₂ : x₂ = 12 * π / 5 ∧ y₂ = -3 * π / 5)
  (h₃ : x₃ = 4 * π / 3 ∧ y₃ = -π / 3) :
  (x₁ + 4 * y₁ = 0 ∧ x₁ + 3 * y₁ < π ∧ π - x₁ - 3 * y₁ ≠ 1 ∧ 3 * x₁ + 5 * y₁ > 0) ∧
  (x₂ + 4 * y₂ = 0 ∧ x₂ + 3 * y₂ < π ∧ π - x₂ - 3 * y₂ ≠ 1 ∧ 3 * x₂ + 5 * y₂ > 0) ∧
  (x₃ + 4 * y₃ = 0 ∧ x₃ + 3 * y₃ < π ∧ π - x₃ - 3 * y₃ ≠ 1 ∧ 3 * x₃ + 5 * y₃ > 0) :=
by
  sorry

end NUMINAMATH_CALUDE_coordinate_conditions_l2233_223368


namespace NUMINAMATH_CALUDE_quadratic_factorization_l2233_223354

theorem quadratic_factorization (a b : ℤ) :
  (∀ y : ℝ, 2 * y^2 - 5 * y - 12 = (2 * y + a) * (y + b)) →
  a - b = 7 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l2233_223354


namespace NUMINAMATH_CALUDE_bisection_next_step_l2233_223318

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the conditions
axiom f_continuous : ContinuousOn f (Set.Icc 0 1)
axiom f_neg_zero : f 0 < 0
axiom f_neg_half : f 0.5 < 0
axiom f_pos_one : f 1 > 0

-- Define the theorem
theorem bisection_next_step :
  ∃ x ∈ Set.Ioo 0.5 1, f x = 0 ∧ 
  (∀ y, y ∈ Set.Icc 0 1 → f y = 0 → y ∈ Set.Icc 0.5 1) ∧
  (0.75 = (0.5 + 1) / 2) := by
  sorry

end NUMINAMATH_CALUDE_bisection_next_step_l2233_223318


namespace NUMINAMATH_CALUDE_discount_calculation_l2233_223340

theorem discount_calculation (original_price : ℝ) (original_price_pos : original_price > 0) :
  let first_discount := 0.3
  let second_discount := 0.25
  let price_after_first_discount := original_price * (1 - first_discount)
  let final_price := price_after_first_discount * (1 - second_discount)
  final_price / original_price = 0.525 := by
sorry

end NUMINAMATH_CALUDE_discount_calculation_l2233_223340


namespace NUMINAMATH_CALUDE_passes_through_fixed_point_not_in_fourth_quadrant_min_area_min_area_line_eq_l2233_223333

/-- Given a line l: kx - 3y + 2k + 3 = 0, where k ∈ ℝ -/
def line_l (k : ℝ) (x y : ℝ) : Prop := k * x - 3 * y + 2 * k + 3 = 0

/-- The point (-2, 1) -/
def fixed_point : ℝ × ℝ := (-2, 1)

/-- The line passes through the fixed point for all values of k -/
theorem passes_through_fixed_point (k : ℝ) :
  line_l k (fixed_point.1) (fixed_point.2) := by sorry

/-- The line does not pass through the fourth quadrant when k ∈ [0, +∞) -/
theorem not_in_fourth_quadrant (k : ℝ) (hk : k ≥ 0) :
  ∀ x y, line_l k x y → (x ≤ 0 ∧ y ≥ 0) ∨ (x ≥ 0 ∧ y ≥ 0) := by sorry

/-- The area of triangle AOB formed by the line's intersections with the x and y axes -/
noncomputable def triangle_area (k : ℝ) : ℝ :=
  (1/6) * (4 * k + 9 / k + 12)

/-- The minimum area of triangle AOB is 4, occurring when k = 3/2 -/
theorem min_area :
  ∃ k, k > 0 ∧ triangle_area k = 4 ∧ ∀ k', k' > 0 → triangle_area k' ≥ 4 := by sorry

/-- The line equation at the minimum area point -/
def min_area_line (x y : ℝ) : Prop := x - 2 * y + 4 = 0

/-- The line equation at the minimum area point is x - 2y + 4 = 0 -/
theorem min_area_line_eq :
  ∃ k, k > 0 ∧ triangle_area k = 4 ∧ ∀ x y, line_l k x y ↔ min_area_line x y := by sorry

end NUMINAMATH_CALUDE_passes_through_fixed_point_not_in_fourth_quadrant_min_area_min_area_line_eq_l2233_223333


namespace NUMINAMATH_CALUDE_ios_department_larger_l2233_223372

/-- Represents the number of developers in the Android department -/
def android_devs : ℕ := sorry

/-- Represents the number of developers in the iOS department -/
def ios_devs : ℕ := sorry

/-- The total number of messages sent equals the total number of messages received -/
axiom message_balance : 7 * android_devs + 15 * ios_devs = 15 * android_devs + 9 * ios_devs

theorem ios_department_larger : ios_devs > android_devs := by
  sorry

end NUMINAMATH_CALUDE_ios_department_larger_l2233_223372


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l2233_223383

theorem polynomial_divisibility (p q : ℤ) : 
  (∀ x : ℤ, (x - 2) * (x + 1) ∣ (x^5 - x^4 + x^3 - p*x^2 + q*x + 4)) → 
  p = 5 ∧ q = -4 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l2233_223383


namespace NUMINAMATH_CALUDE_paint_ornaments_l2233_223394

/-- Represents the problem of painting star-shaped ornaments on tiles --/
theorem paint_ornaments (num_tiles : ℕ) (paint_coverage : ℝ) (tile_side : ℝ) 
  (pentagon_area : ℝ) (triangle_base triangle_height : ℝ) : 
  num_tiles = 20 → 
  paint_coverage = 750 → 
  tile_side = 12 → 
  pentagon_area = 15 → 
  triangle_base = 4 → 
  triangle_height = 6 → 
  (num_tiles * (tile_side^2 - 4*pentagon_area - 2*triangle_base*triangle_height) ≤ paint_coverage) :=
by sorry

end NUMINAMATH_CALUDE_paint_ornaments_l2233_223394


namespace NUMINAMATH_CALUDE_normal_distribution_symmetry_l2233_223309

/-- A random variable following a normal distribution -/
structure NormalRV where
  μ : ℝ
  σ : ℝ
  hσ : σ > 0

/-- Probability of a normal random variable being less than or equal to a value -/
noncomputable def probability (X : NormalRV) (x : ℝ) : ℝ := sorry

theorem normal_distribution_symmetry (X : NormalRV) (h : X.μ = 2) (h_prob : probability X 4 = 0.84) :
  probability X 0 = 0.16 := by sorry

end NUMINAMATH_CALUDE_normal_distribution_symmetry_l2233_223309


namespace NUMINAMATH_CALUDE_well_digging_hours_l2233_223334

/-- The number of hours worked on the first day by two men digging a well -/
def first_day_hours : ℕ := 20

/-- The total payment for both men over three days of work -/
def total_payment : ℕ := 660

/-- The hourly rate paid to each man -/
def hourly_rate : ℕ := 10

/-- The number of hours worked by both men on the second day -/
def second_day_hours : ℕ := 16

/-- The number of hours worked by both men on the third day -/
def third_day_hours : ℕ := 30

theorem well_digging_hours : 
  hourly_rate * (first_day_hours + second_day_hours + third_day_hours) = total_payment :=
by sorry

end NUMINAMATH_CALUDE_well_digging_hours_l2233_223334


namespace NUMINAMATH_CALUDE_smallest_unachievable_score_l2233_223371

def dart_scores : Set ℕ := {0, 1, 3, 8, 12}

def is_achievable (n : ℕ) : Prop :=
  ∃ (a b c : ℕ), a ∈ dart_scores ∧ b ∈ dart_scores ∧ c ∈ dart_scores ∧ a + b + c = n

theorem smallest_unachievable_score :
  (∀ m < 22, is_achievable m) ∧ ¬is_achievable 22 :=
sorry

end NUMINAMATH_CALUDE_smallest_unachievable_score_l2233_223371


namespace NUMINAMATH_CALUDE_sneakers_cost_l2233_223344

theorem sneakers_cost (sneakers_cost socks_cost : ℝ) 
  (total_cost : sneakers_cost + socks_cost = 101)
  (cost_difference : sneakers_cost = socks_cost + 100) : 
  sneakers_cost = 100.5 := by
  sorry

end NUMINAMATH_CALUDE_sneakers_cost_l2233_223344


namespace NUMINAMATH_CALUDE_point_in_unit_circle_l2233_223379

theorem point_in_unit_circle (z : ℂ) (h : Complex.abs z ≤ 1) :
  (z.re)^2 + (z.im)^2 ≤ 1 := by
sorry

end NUMINAMATH_CALUDE_point_in_unit_circle_l2233_223379


namespace NUMINAMATH_CALUDE_pascal_triangle_properties_l2233_223389

/-- The sum of elements in the first n rows of Pascal's Triangle -/
def pascal_sum (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The binomial coefficient (n choose k) -/
def binomial (n k : ℕ) : ℕ := Nat.choose n k

theorem pascal_triangle_properties :
  (pascal_sum 30 = 465) ∧
  (binomial 30 5 > 0) := by
  sorry

end NUMINAMATH_CALUDE_pascal_triangle_properties_l2233_223389


namespace NUMINAMATH_CALUDE_caden_coin_value_l2233_223373

/-- Represents the number of coins of each type Caden has -/
structure CoinCounts where
  pennies : ℕ
  nickels : ℕ
  dimes : ℕ
  quarters : ℕ

/-- Calculates the total value of coins in dollars -/
def total_value (coins : CoinCounts) : ℚ :=
  (coins.pennies : ℚ) / 100 +
  (coins.nickels : ℚ) / 20 +
  (coins.dimes : ℚ) / 10 +
  (coins.quarters : ℚ) / 4

/-- Theorem stating that Caden's coins total $8.00 -/
theorem caden_coin_value :
  ∀ (coins : CoinCounts),
    coins.pennies = 120 →
    coins.nickels = coins.pennies / 3 →
    coins.dimes = coins.nickels / 5 →
    coins.quarters = 2 * coins.dimes →
    total_value coins = 8 := by
  sorry

end NUMINAMATH_CALUDE_caden_coin_value_l2233_223373


namespace NUMINAMATH_CALUDE_plate_price_l2233_223357

/-- Given the conditions of Chenny's purchase, prove that each plate costs $2 -/
theorem plate_price (num_plates : ℕ) (spoon_price : ℚ) (num_spoons : ℕ) (total_paid : ℚ) :
  num_plates = 9 →
  spoon_price = 3/2 →
  num_spoons = 4 →
  total_paid = 24 →
  ∃ (plate_price : ℚ), plate_price * num_plates + spoon_price * num_spoons = total_paid ∧ plate_price = 2 := by
  sorry

end NUMINAMATH_CALUDE_plate_price_l2233_223357


namespace NUMINAMATH_CALUDE_curve_and_line_properties_l2233_223319

-- Define the curve C
def curve_C (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 4

-- Define the line l
def line_l (x y : ℝ) : Prop := x = -2 ∨ 3*x + 4*y - 2 = 0

-- Define the distance ratio condition
def distance_ratio (x y : ℝ) : Prop :=
  (x^2 + y^2) = (1/4) * ((x - 3)^2 + y^2)

-- Define the intersection condition
def intersects_curve (l : ℝ → ℝ → Prop) (c : ℝ → ℝ → Prop) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ : ℝ), l x₁ y₁ ∧ l x₂ y₂ ∧ c x₁ y₁ ∧ c x₂ y₂ ∧ 
  (x₁ ≠ x₂ ∨ y₁ ≠ y₂)

-- Main theorem
theorem curve_and_line_properties :
  (∀ x y, curve_C x y ↔ distance_ratio x y) ∧
  (line_l (-2) 2) ∧
  (intersects_curve line_l curve_C) ∧
  (∃ x₁ y₁ x₂ y₂, line_l x₁ y₁ ∧ line_l x₂ y₂ ∧ curve_C x₁ y₁ ∧ curve_C x₂ y₂ ∧
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = 12) :=
sorry

end NUMINAMATH_CALUDE_curve_and_line_properties_l2233_223319


namespace NUMINAMATH_CALUDE_polynomial_value_l2233_223349

/-- Given that ax³ + bx + 1 = 2023 when x = 1, prove that ax³ + bx - 2 = -2024 when x = -1 -/
theorem polynomial_value (a b : ℝ) : 
  (a * 1^3 + b * 1 + 1 = 2023) → (a * (-1)^3 + b * (-1) - 2 = -2024) := by
sorry

end NUMINAMATH_CALUDE_polynomial_value_l2233_223349


namespace NUMINAMATH_CALUDE_remainder_of_power_minus_ninety_l2233_223338

theorem remainder_of_power_minus_ninety (n : ℕ) : (1 - 90) ^ 10 ≡ 1 [ZMOD 88] := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_power_minus_ninety_l2233_223338
