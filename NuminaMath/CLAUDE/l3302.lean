import Mathlib

namespace NUMINAMATH_CALUDE_women_equal_to_five_men_l3302_330226

/-- Represents the amount of work one person can do in a day -/
structure WorkPerDay (α : Type) where
  amount : ℝ

/-- Represents the total amount of work for a job -/
def Job : Type := ℝ

variable (men_work : WorkPerDay Unit) (women_work : WorkPerDay Unit)

/-- The amount of work 5 men do in a day equals the amount of work x women do in a day -/
def men_women_equal (x : ℝ) : Prop :=
  5 * men_work.amount = x * women_work.amount

/-- 3 men and 5 women finish the job in 10 days -/
def job_condition1 (job : Job) : Prop :=
  (3 * men_work.amount + 5 * women_work.amount) * 10 = job

/-- 7 women finish the job in 14 days -/
def job_condition2 (job : Job) : Prop :=
  7 * women_work.amount * 14 = job

/-- The main theorem: prove that 8 women do the same amount of work in a day as 5 men -/
theorem women_equal_to_five_men
  (job : Job)
  (h1 : job_condition1 men_work women_work job)
  (h2 : job_condition2 women_work job) :
  men_women_equal men_work women_work 8 := by
  sorry


end NUMINAMATH_CALUDE_women_equal_to_five_men_l3302_330226


namespace NUMINAMATH_CALUDE_certain_number_proof_l3302_330220

theorem certain_number_proof : ∃ n : ℕ, n * 240 = 1038 * 40 ∧ n = 173 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l3302_330220


namespace NUMINAMATH_CALUDE_leftover_milk_proof_l3302_330222

/-- Represents the amount of milk used for each type of milkshake -/
structure MilkUsage where
  vanilla : ℕ
  chocolate : ℕ

/-- Represents the amount of ice cream used for each type of milkshake -/
structure IceCreamUsage where
  vanilla : ℕ
  chocolate : ℕ

/-- Represents the available ingredients -/
structure Ingredients where
  milk : ℕ
  vanilla_ice_cream : ℕ
  chocolate_ice_cream : ℕ

/-- Represents the number of milkshakes to make -/
structure Milkshakes where
  vanilla : ℕ
  chocolate : ℕ

def milk_usage : MilkUsage := ⟨4, 5⟩
def ice_cream_usage : IceCreamUsage := ⟨12, 10⟩
def available_ingredients : Ingredients := ⟨72, 96, 96⟩
def max_milkshakes : ℕ := 16

def valid_milkshake_count (m : Milkshakes) : Prop :=
  m.vanilla + m.chocolate ≤ max_milkshakes ∧
  2 * m.chocolate = m.vanilla

def enough_ingredients (m : Milkshakes) : Prop :=
  m.vanilla * milk_usage.vanilla + m.chocolate * milk_usage.chocolate ≤ available_ingredients.milk ∧
  m.vanilla * ice_cream_usage.vanilla ≤ available_ingredients.vanilla_ice_cream ∧
  m.chocolate * ice_cream_usage.chocolate ≤ available_ingredients.chocolate_ice_cream

def optimal_milkshakes : Milkshakes := ⟨10, 5⟩

theorem leftover_milk_proof :
  valid_milkshake_count optimal_milkshakes ∧
  enough_ingredients optimal_milkshakes ∧
  ∀ m : Milkshakes, valid_milkshake_count m → enough_ingredients m →
    m.vanilla + m.chocolate ≤ optimal_milkshakes.vanilla + optimal_milkshakes.chocolate →
  available_ingredients.milk - (optimal_milkshakes.vanilla * milk_usage.vanilla + optimal_milkshakes.chocolate * milk_usage.chocolate) = 7 :=
sorry

end NUMINAMATH_CALUDE_leftover_milk_proof_l3302_330222


namespace NUMINAMATH_CALUDE_rectangle_area_l3302_330221

/-- Given a rectangle with length 3 times its width and width of 6 inches, 
    prove that its area is 108 square inches. -/
theorem rectangle_area (width : ℝ) (length : ℝ) : 
  width = 6 → length = 3 * width → width * length = 108 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l3302_330221


namespace NUMINAMATH_CALUDE_system_solution_conditions_l3302_330239

theorem system_solution_conditions (a b x y z : ℝ) : 
  (x + y + z = a) →
  (x^2 + y^2 + z^2 = b^2) →
  (x * y = z^2) →
  (x > 0 ∧ y > 0 ∧ z > 0 ∧ x ≠ y ∧ y ≠ z ∧ x ≠ z) ↔ 
  (3 * b^2 > a^2 ∧ a^2 > b^2 ∧ a > 0) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_conditions_l3302_330239


namespace NUMINAMATH_CALUDE_smaller_angle_is_70_l3302_330263

/-- A parallelogram with one angle exceeding the other by 40 degrees -/
structure Parallelogram40 where
  -- The measure of the smaller angle
  small_angle : ℝ
  -- The measure of the larger angle
  large_angle : ℝ
  -- The larger angle exceeds the smaller by 40 degrees
  angle_difference : large_angle = small_angle + 40
  -- Adjacent angles are supplementary (sum to 180 degrees)
  supplementary : small_angle + large_angle = 180

/-- The smaller angle in a Parallelogram40 measures 70 degrees -/
theorem smaller_angle_is_70 (p : Parallelogram40) : p.small_angle = 70 := by
  sorry

end NUMINAMATH_CALUDE_smaller_angle_is_70_l3302_330263


namespace NUMINAMATH_CALUDE_inverse_of_A_l3302_330244

def A : Matrix (Fin 2) (Fin 2) ℚ := !![3, 4; -2, 9]

theorem inverse_of_A :
  A⁻¹ = !![9/35, -4/35; 2/35, 3/35] := by sorry

end NUMINAMATH_CALUDE_inverse_of_A_l3302_330244


namespace NUMINAMATH_CALUDE_triangle_abc_is_right_angled_l3302_330251

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A line in the 2D plane -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- A parabola in the 2D plane -/
structure Parabola where
  a : ℝ
  h : ℝ
  k : ℝ

/-- Theorem: Given point A(1, 2) and a line passing through (5, -2) that intersects
    the parabola y^2 = 4x at points B and C, triangle ABC is right-angled -/
theorem triangle_abc_is_right_angled 
  (A : Point)
  (l : Line)
  (p : Parabola)
  (B C : Point)
  (h1 : A.x = 1 ∧ A.y = 2)
  (h2 : l.slope * (-2) + l.intercept = 5)
  (h3 : p.a = 4 ∧ p.h = 0 ∧ p.k = 0)
  (h4 : B.y^2 = 4 * B.x ∧ B.y = l.slope * B.x + l.intercept)
  (h5 : C.y^2 = 4 * C.x ∧ C.y = l.slope * C.x + l.intercept)
  (h6 : B ≠ C) :
  (B.x - A.x) * (C.x - A.x) + (B.y - A.y) * (C.y - A.y) = 0 := by
  sorry


end NUMINAMATH_CALUDE_triangle_abc_is_right_angled_l3302_330251


namespace NUMINAMATH_CALUDE_quadratic_roots_imply_m_range_l3302_330211

theorem quadratic_roots_imply_m_range (m : ℝ) : 
  (∃ r₁ r₂ : ℝ, r₁ ∈ Set.Ioo 0 1 ∧ r₂ ∈ Set.Ioo 2 3 ∧ 
   r₁^2 - 2*m*r₁ + m^2 - 1 = 0 ∧ r₂^2 - 2*m*r₂ + m^2 - 1 = 0) →
  m ∈ Set.Ioo 1 2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_imply_m_range_l3302_330211


namespace NUMINAMATH_CALUDE_base_conversion_sum_rounded_to_28_l3302_330242

/-- Converts a number from base b to base 10 -/
def to_base_10 (digits : List Nat) (b : Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * b ^ i) 0

/-- Rounds a rational number to the nearest integer -/
def round_to_nearest (q : Rat) : Int :=
  (q + 1/2).floor

theorem base_conversion_sum_rounded_to_28 :
  let a := to_base_10 [4, 5, 2] 8  -- 254 in base 8
  let b := to_base_10 [3, 1] 4     -- 13 in base 4
  let c := to_base_10 [2, 3, 1] 5  -- 132 in base 5
  let d := to_base_10 [2, 3] 4     -- 32 in base 4
  round_to_nearest ((a / b : Rat) + (c / d : Rat)) = 28 := by
  sorry

#eval round_to_nearest ((172 / 7 : Rat) + (42 / 14 : Rat))

end NUMINAMATH_CALUDE_base_conversion_sum_rounded_to_28_l3302_330242


namespace NUMINAMATH_CALUDE_choose_three_from_nine_l3302_330285

theorem choose_three_from_nine : Nat.choose 9 3 = 84 := by
  sorry

end NUMINAMATH_CALUDE_choose_three_from_nine_l3302_330285


namespace NUMINAMATH_CALUDE_lunks_needed_for_twenty_apples_l3302_330213

/-- Represents the number of lunks needed to purchase a given number of apples,
    given the exchange rates between lunks, kunks, and apples. -/
def lunks_for_apples (lunks_per_kunk : ℚ) (kunks_per_apple : ℚ) (num_apples : ℕ) : ℚ :=
  num_apples * kunks_per_apple * lunks_per_kunk

/-- Theorem stating that 21 lunks are needed to purchase 20 apples,
    given the specified exchange rates. -/
theorem lunks_needed_for_twenty_apples :
  lunks_for_apples (7/4) (3/5) 20 = 21 := by
  sorry

end NUMINAMATH_CALUDE_lunks_needed_for_twenty_apples_l3302_330213


namespace NUMINAMATH_CALUDE_min_mn_value_l3302_330229

def f (x a : ℝ) : ℝ := |x - a|

theorem min_mn_value (m n : ℝ) (hm : m > 0) (hn : n > 0) :
  (∀ x, f x 1 ≤ 1 ↔ x ∈ Set.Icc 0 2) →
  1 / m + 1 / (2 * n) = 1 →
  ∀ k, m * n ≥ k → k = 2 :=
sorry

end NUMINAMATH_CALUDE_min_mn_value_l3302_330229


namespace NUMINAMATH_CALUDE_quadratic_integer_solution_l3302_330218

theorem quadratic_integer_solution (a : ℤ) : 
  a < 0 → 
  (∃ x : ℤ, a * x^2 - 2*(a-3)*x + (a-2) = 0) ↔ 
  (a = -10 ∨ a = -4) :=
sorry

end NUMINAMATH_CALUDE_quadratic_integer_solution_l3302_330218


namespace NUMINAMATH_CALUDE_unique_line_through_points_l3302_330245

-- Define a type for points in Euclidean geometry
variable (Point : Type)

-- Define a type for lines in Euclidean geometry
variable (Line : Type)

-- Define a relation for a point being on a line
variable (on_line : Point → Line → Prop)

-- Axiom: For any two distinct points, there exists a line passing through both points
axiom line_through_points (P Q : Point) (h : P ≠ Q) : ∃ (l : Line), on_line P l ∧ on_line Q l

-- Axiom: Any line passing through two distinct points is unique
axiom line_uniqueness (P Q : Point) (h : P ≠ Q) (l1 l2 : Line) :
  on_line P l1 ∧ on_line Q l1 → on_line P l2 ∧ on_line Q l2 → l1 = l2

-- Theorem: There exists exactly one line passing through any two distinct points
theorem unique_line_through_points (P Q : Point) (h : P ≠ Q) :
  ∃! (l : Line), on_line P l ∧ on_line Q l :=
sorry

end NUMINAMATH_CALUDE_unique_line_through_points_l3302_330245


namespace NUMINAMATH_CALUDE_smallest_visible_sum_l3302_330261

/-- Represents a die with 6 faces -/
structure Die :=
  (faces : Fin 6 → ℕ)
  (opposite_sum : ∀ i : Fin 3, faces i + faces (i + 3) = 7)

/-- Represents a 4x4x4 cube made of dice -/
def LargeCube := Fin 4 → Fin 4 → Fin 4 → Die

/-- The sum of visible face values on the large cube -/
def visible_sum (cube : LargeCube) : ℕ :=
  sorry

/-- Theorem stating the smallest possible sum of visible faces -/
theorem smallest_visible_sum :
  ∃ (cube : LargeCube), visible_sum cube = 144 ∧
  ∀ (other_cube : LargeCube), visible_sum other_cube ≥ 144 :=
sorry

end NUMINAMATH_CALUDE_smallest_visible_sum_l3302_330261


namespace NUMINAMATH_CALUDE_larger_pile_size_l3302_330203

/-- Given two piles of toys where the total number is 120 and the larger pile
    is twice as big as the smaller pile, the number of toys in the larger pile is 80. -/
theorem larger_pile_size (small : ℕ) (large : ℕ) : 
  small + large = 120 → large = 2 * small → large = 80 := by
  sorry

end NUMINAMATH_CALUDE_larger_pile_size_l3302_330203


namespace NUMINAMATH_CALUDE_guess_number_in_seven_questions_l3302_330283

theorem guess_number_in_seven_questions :
  ∃ (f : Fin 7 → (Nat × Nat)),
    (∀ i, (f i).1 < 100 ∧ (f i).2 < 100) →
    ∀ X ≤ 100,
      ∀ Y ≤ 100,
      (∀ i, Nat.gcd (X + (f i).1) (f i).2 = Nat.gcd (Y + (f i).1) (f i).2) →
      X = Y :=
by sorry

end NUMINAMATH_CALUDE_guess_number_in_seven_questions_l3302_330283


namespace NUMINAMATH_CALUDE_equation_solution_l3302_330249

theorem equation_solution (x : ℝ) (h_pos : x > 0) :
  7.74 * Real.sqrt (Real.log x / Real.log 5) + (Real.log x / Real.log 5) ^ (1/3) = 2 →
  x = 5 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l3302_330249


namespace NUMINAMATH_CALUDE_square_sum_constant_l3302_330276

theorem square_sum_constant (x : ℝ) : (x + 2)^2 + 2*(x + 2)*(5 - x) + (5 - x)^2 = 49 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_constant_l3302_330276


namespace NUMINAMATH_CALUDE_train_crossing_time_l3302_330289

/-- The time taken for two trains to cross each other -/
theorem train_crossing_time : 
  ∀ (train_length : ℝ) (train_speed : ℝ),
  train_length = 120 →
  train_speed = 27 →
  (2 * train_length) / (2 * train_speed * (1000 / 3600)) = 16 := by
  sorry

end NUMINAMATH_CALUDE_train_crossing_time_l3302_330289


namespace NUMINAMATH_CALUDE_smallest_angle_divisible_isosceles_triangle_l3302_330219

/-- An isosceles triangle that can be cut into two isosceles triangles -/
structure DivisibleIsoscelesTriangle where
  /-- The measure of one of the two equal angles of the isosceles triangle -/
  α : Real
  /-- The triangle is isosceles -/
  isIsosceles : α ≥ 0 ∧ α ≤ 90
  /-- The triangle can be cut into two isosceles triangles -/
  canBeDivided : ∃ (β γ : Real), (β > 0 ∧ γ > 0) ∧
    ((2 * α + β = 180 ∧ 2 * γ + β = 180) ∨
     (α + 2 * β = 180 ∧ α + 2 * γ = 180) ∨
     (2 * α + γ = 180 ∧ 2 * β + γ = 180))

/-- The smallest angle in a divisible isosceles triangle is 180/7 -/
theorem smallest_angle_divisible_isosceles_triangle :
  ∀ (t : DivisibleIsoscelesTriangle), min (min t.α (180 - 2*t.α)) 180/7 = 180/7 := by
  sorry

end NUMINAMATH_CALUDE_smallest_angle_divisible_isosceles_triangle_l3302_330219


namespace NUMINAMATH_CALUDE_trigonometric_system_solution_l3302_330265

theorem trigonometric_system_solution (x y z : ℝ) 
  (eq1 : Real.sin x + 2 * Real.sin (x + y + z) = 0)
  (eq2 : Real.sin y + 3 * Real.sin (x + y + z) = 0)
  (eq3 : Real.sin z + 4 * Real.sin (x + y + z) = 0) :
  ∃ (k1 k2 k3 : ℤ), x = k1 * Real.pi ∧ y = k2 * Real.pi ∧ z = k3 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_system_solution_l3302_330265


namespace NUMINAMATH_CALUDE_lara_likes_one_last_digit_l3302_330256

theorem lara_likes_one_last_digit :
  ∃! d : ℕ, d < 10 ∧ ∀ n : ℕ, (n % 3 = 0 ∧ n % 5 = 0) → n % 10 = d :=
sorry

end NUMINAMATH_CALUDE_lara_likes_one_last_digit_l3302_330256


namespace NUMINAMATH_CALUDE_multiply_and_simplify_l3302_330254

theorem multiply_and_simplify (y : ℝ) (h : y ≠ 0) :
  (18 * y^3) * (9 * y^2) * (1 / (6*y)^2) = (9/2) * y^3 := by
  sorry

end NUMINAMATH_CALUDE_multiply_and_simplify_l3302_330254


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l3302_330260

theorem sum_of_coefficients (a a₁ a₂ a₃ a₄ : ℝ) :
  (∀ x : ℝ, (x + 2)^4 = a₄*x^4 + a₃*x^3 + a₂*x^2 + a₁*x + a) →
  a₁ + a₂ + a₃ + a₄ = 65 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l3302_330260


namespace NUMINAMATH_CALUDE_isosceles_right_triangle_area_l3302_330209

/-- Given an isosceles right triangle with squares on its sides, 
    prove that its area is 32 square units. -/
theorem isosceles_right_triangle_area 
  (a b c : ℝ) 
  (h_isosceles : a = b) 
  (h_right : a^2 + b^2 = c^2) 
  (h_leg_square : a^2 = 64) 
  (h_hypotenuse_square : c^2 = 256) : 
  (1/2) * a^2 = 32 := by
  sorry

#check isosceles_right_triangle_area

end NUMINAMATH_CALUDE_isosceles_right_triangle_area_l3302_330209


namespace NUMINAMATH_CALUDE_polynomial_remainder_l3302_330290

theorem polynomial_remainder (p : ℤ) : (p^11 - 3) % (p - 2) = 2045 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_remainder_l3302_330290


namespace NUMINAMATH_CALUDE_initial_roses_l3302_330237

theorem initial_roses (thrown_away : ℕ) (final_count : ℕ) :
  thrown_away = 33 →
  final_count = 17 →
  ∃ (initial : ℕ) (new_cut : ℕ),
    initial - thrown_away + new_cut = final_count ∧
    new_cut = thrown_away + 2 ∧
    initial = 15 := by
  sorry

end NUMINAMATH_CALUDE_initial_roses_l3302_330237


namespace NUMINAMATH_CALUDE_initial_orchids_count_l3302_330293

theorem initial_orchids_count (initial_roses : ℕ) (final_roses : ℕ) (final_orchids : ℕ) (total_flowers : ℕ) : 
  initial_roses = 13 →
  final_roses = 14 →
  final_orchids = 91 →
  total_flowers = 105 →
  final_roses + final_orchids = total_flowers →
  final_orchids = initial_roses + final_orchids - total_flowers + final_roses :=
by
  sorry

#check initial_orchids_count

end NUMINAMATH_CALUDE_initial_orchids_count_l3302_330293


namespace NUMINAMATH_CALUDE_dance_steps_total_time_l3302_330230

/-- The time spent learning seven dance steps -/
def dance_steps_time : ℝ → ℝ → ℝ → ℝ → ℝ → ℝ → ℝ → ℝ :=
  fun step1 step2 step3 step4 step5 step6 step7 =>
    step1 + step2 + step3 + step4 + step5 + step6 + step7

theorem dance_steps_total_time :
  ∀ (step1 : ℝ),
    step1 = 50 →
    let step2 := step1 / 3
    let step3 := step1 + step2
    let step4 := 1.75 * step1
    let step5 := step2 + 25
    let step6 := step3 + step5 - 40
    let step7 := step1 + step2 + step4 + 10
    ∃ (ε : ℝ), ε > 0 ∧ 
      |dance_steps_time step1 step2 step3 step4 step5 step6 step7 - 495.02| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_dance_steps_total_time_l3302_330230


namespace NUMINAMATH_CALUDE_probability_theorem_l3302_330216

def is_valid_pair (a b : ℕ) : Prop :=
  1 ≤ a ∧ a ≤ 100 ∧ 1 ≤ b ∧ b ≤ 100 ∧ a ≠ b

def satisfies_condition (a b : ℕ) : Prop :=
  ∃ m : ℕ, a * b + a + b = 7 * m - 2

def count_valid_pairs : ℕ := Nat.choose 100 2

def count_satisfying_pairs : ℕ := 1295

theorem probability_theorem :
  (count_satisfying_pairs : ℚ) / count_valid_pairs = 259 / 990 :=
sorry

end NUMINAMATH_CALUDE_probability_theorem_l3302_330216


namespace NUMINAMATH_CALUDE_sequence_first_term_l3302_330262

/-- Given a sequence {a_n} defined by a_n = (√2)^(n-2), prove that a_1 = √2/2 -/
theorem sequence_first_term (a : ℕ → ℝ) (h : ∀ n, a n = (Real.sqrt 2) ^ (n - 2)) :
  a 1 = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sequence_first_term_l3302_330262


namespace NUMINAMATH_CALUDE_systematic_sampling_removal_l3302_330201

theorem systematic_sampling_removal (total_students : ℕ) (sample_size : ℕ) 
  (h1 : total_students = 1252) 
  (h2 : sample_size = 50) : 
  ∃ (removed : ℕ), removed = total_students % sample_size ∧ removed = 2 := by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_removal_l3302_330201


namespace NUMINAMATH_CALUDE_taxi_charge_per_segment_l3302_330270

/-- Proves that the additional charge per 2/5 of a mile is $0.35 -/
theorem taxi_charge_per_segment (initial_fee : ℝ) (trip_distance : ℝ) (total_charge : ℝ) 
  (h1 : initial_fee = 2.35)
  (h2 : trip_distance = 3.6)
  (h3 : total_charge = 5.5) :
  (total_charge - initial_fee) / (trip_distance / (2/5)) = 0.35 := by
  sorry

end NUMINAMATH_CALUDE_taxi_charge_per_segment_l3302_330270


namespace NUMINAMATH_CALUDE_determinant_of_specific_matrix_l3302_330298

theorem determinant_of_specific_matrix :
  let A : Matrix (Fin 3) (Fin 3) ℤ := !![2, 5, -3; 0, 3, -1; 7, -4, 2]
  Matrix.det A = 32 := by
  sorry

end NUMINAMATH_CALUDE_determinant_of_specific_matrix_l3302_330298


namespace NUMINAMATH_CALUDE_tangent_curve_sum_l3302_330248

/-- A curve y = -2x^2 + bx + c is tangent to the line y = x - 3 at the point (2, -1). -/
theorem tangent_curve_sum (b c : ℝ) : 
  (∀ x, -2 * x^2 + b * x + c = x - 3 → x = 2) → 
  (-2 * 2^2 + b * 2 + c = -1) →
  ((-4 * 2 + b) = 1) →
  b + c = -2 := by sorry

end NUMINAMATH_CALUDE_tangent_curve_sum_l3302_330248


namespace NUMINAMATH_CALUDE_square_grid_15_toothpicks_l3302_330246

/-- Calculates the total number of toothpicks needed for a square grid -/
def toothpicks_in_square_grid (side_length : ℕ) : ℕ :=
  2 * (side_length + 1) * side_length

/-- Theorem: A square grid with 15 toothpicks on each side requires 480 toothpicks -/
theorem square_grid_15_toothpicks :
  toothpicks_in_square_grid 15 = 480 := by
  sorry

#eval toothpicks_in_square_grid 15

end NUMINAMATH_CALUDE_square_grid_15_toothpicks_l3302_330246


namespace NUMINAMATH_CALUDE_nearest_integer_to_three_plus_sqrt_five_fourth_power_l3302_330281

theorem nearest_integer_to_three_plus_sqrt_five_fourth_power :
  ∃ n : ℤ, n = 752 ∧ ∀ m : ℤ, |((3 : ℝ) + Real.sqrt 5)^4 - (n : ℝ)| ≤ |((3 : ℝ) + Real.sqrt 5)^4 - (m : ℝ)| := by
  sorry

end NUMINAMATH_CALUDE_nearest_integer_to_three_plus_sqrt_five_fourth_power_l3302_330281


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l3302_330223

-- Define the quadratic function
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- State the theorem
theorem quadratic_function_properties 
  (a b c : ℝ) 
  (h1 : f a b c 1 = -3)  -- vertex at (1, -3)
  (h2 : f a b c 2 = -5/2)  -- passes through (2, -5/2)
  (h3 : ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f a b c x1 = f a b c x2 ∧ |x1 - x2| = 6)  -- intersects y = m at two points 6 units apart
  : 
  (∀ x, f a b c x = 1/2 * x^2 - x - 5/2) ∧  -- Part 1
  (∃ m : ℝ, m = 3/2 ∧ ∀ x, f a b c x = m → (∃ y, f a b c y = m ∧ |x - y| = 6)) ∧  -- Part 2
  (∀ x, -3 < x → x < 3 → -3 ≤ f a b c x ∧ f a b c x < 5)  -- Part 3
  := by sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l3302_330223


namespace NUMINAMATH_CALUDE_paper_strip_to_squares_l3302_330241

/-- Represents a strip of paper with given width and length -/
structure PaperStrip where
  width : ℝ
  length : ℝ

/-- Represents a square -/
structure Square where
  side : ℝ

/-- Function to calculate the area of a square -/
def squareArea (s : Square) : ℝ := s.side * s.side

/-- Theorem stating that a paper strip of width 1 cm and length 4 cm
    can be transformed into squares of areas 1 sq cm and 2 sq cm -/
theorem paper_strip_to_squares 
  (strip : PaperStrip) 
  (h_width : strip.width = 1) 
  (h_length : strip.length = 4) :
  ∃ (s1 s2 : Square), 
    squareArea s1 = 1 ∧ 
    squareArea s2 = 2 := by
  sorry


end NUMINAMATH_CALUDE_paper_strip_to_squares_l3302_330241


namespace NUMINAMATH_CALUDE_hyperbola_from_ellipse_l3302_330286

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2/16 + y^2/9 = 1

-- Define the hyperbola equations
def hyperbola1 (x y : ℝ) : Prop := x^2/16 - y^2/48 = 1
def hyperbola2 (x y : ℝ) : Prop := y^2/9 - x^2/27 = 1

-- Define the eccentricity
def eccentricity : ℝ := 2

-- Theorem statement
theorem hyperbola_from_ellipse :
  ∀ x y : ℝ, ellipse x y →
  (∃ a b : ℝ, (hyperbola1 a b ∧ (a = x ∨ a = -x) ∧ (b = y ∨ b = -y)) ∨
              (hyperbola2 a b ∧ (a = x ∨ a = -x) ∧ (b = y ∨ b = -y))) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_from_ellipse_l3302_330286


namespace NUMINAMATH_CALUDE_certain_number_is_two_l3302_330212

theorem certain_number_is_two :
  ∃ x : ℚ, (287 * 287) + (269 * 269) - x * (287 * 269) = 324 ∧ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_is_two_l3302_330212


namespace NUMINAMATH_CALUDE_functional_equation_solutions_l3302_330284

/-- A function satisfying the given functional equation is either constantly zero or f(x) = x - 1. -/
theorem functional_equation_solutions (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, (x - 2) * f y + f (y + 2 * f x) = f (x + y * f x)) :
  (∀ x : ℝ, f x = 0) ∨ (∀ x : ℝ, f x = x - 1) := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_solutions_l3302_330284


namespace NUMINAMATH_CALUDE_dans_team_total_games_l3302_330204

/-- Represents a baseball team's game results -/
structure BaseballTeam where
  wins : ℕ
  losses : ℕ

/-- The total number of games played by a baseball team -/
def total_games (team : BaseballTeam) : ℕ :=
  team.wins + team.losses

/-- Theorem: Dan's high school baseball team played 18 games in total -/
theorem dans_team_total_games :
  ∃ (team : BaseballTeam), team.wins = 15 ∧ team.losses = 3 ∧ total_games team = 18 :=
sorry

end NUMINAMATH_CALUDE_dans_team_total_games_l3302_330204


namespace NUMINAMATH_CALUDE_mode_of_team_ages_l3302_330279

def team_ages : List Nat := [17, 17, 18, 18, 16, 18, 17, 15, 18, 18, 17, 16, 18, 17, 18, 14]

def mode (l : List Nat) : Nat :=
  l.foldl (fun acc x => if l.count x > l.count acc then x else acc) 0

theorem mode_of_team_ages :
  mode team_ages = 18 := by
  sorry

end NUMINAMATH_CALUDE_mode_of_team_ages_l3302_330279


namespace NUMINAMATH_CALUDE_women_who_left_l3302_330233

/-- Proves the number of women who left the room given the initial and final conditions --/
theorem women_who_left (initial_men : ℕ) (initial_women : ℕ) 
  (h1 : initial_men * 5 = initial_women * 4)  -- Initial ratio of men to women is 4:5
  (h2 : initial_men + 2 = 14)  -- 2 men entered, final count is 14 men
  (h3 : 2 * (initial_women - women_left) = 24)  -- Women doubled after some left, final count is 24 women
  : women_left = 3 := by
  sorry

#check women_who_left

end NUMINAMATH_CALUDE_women_who_left_l3302_330233


namespace NUMINAMATH_CALUDE_middle_number_proof_l3302_330217

theorem middle_number_proof (x y z : ℕ) 
  (h1 : x < y) (h2 : y < z)
  (h3 : x + y = 14) (h4 : x + z = 20) (h5 : y + z = 22)
  (h6 : x + y + z = 27) : y = 7 := by
  sorry

end NUMINAMATH_CALUDE_middle_number_proof_l3302_330217


namespace NUMINAMATH_CALUDE_farm_field_calculation_correct_l3302_330294

/-- Represents the farm field ploughing problem -/
structure FarmField where
  initialCapacityA : ℝ
  initialCapacityB : ℝ
  reducedCapacityA : ℝ
  reducedCapacityB : ℝ
  extraDays : ℕ
  unattendedArea : ℝ

/-- Calculates the area of the farm field and the initially planned work days -/
def calculateFarmFieldResult (f : FarmField) : ℝ × ℕ :=
  let initialTotalCapacity := f.initialCapacityA + f.initialCapacityB
  let reducedTotalCapacity := f.reducedCapacityA + f.reducedCapacityB
  let area := 6600
  let initialDays := 30
  (area, initialDays)

/-- Theorem stating the correctness of the farm field calculation -/
theorem farm_field_calculation_correct (f : FarmField) 
  (h1 : f.initialCapacityA = 120)
  (h2 : f.initialCapacityB = 100)
  (h3 : f.reducedCapacityA = f.initialCapacityA * 0.9)
  (h4 : f.reducedCapacityB = 90)
  (h5 : f.extraDays = 3)
  (h6 : f.unattendedArea = 60) :
  calculateFarmFieldResult f = (6600, 30) := by
  sorry

#eval calculateFarmFieldResult {
  initialCapacityA := 120,
  initialCapacityB := 100,
  reducedCapacityA := 108,
  reducedCapacityB := 90,
  extraDays := 3,
  unattendedArea := 60
}

end NUMINAMATH_CALUDE_farm_field_calculation_correct_l3302_330294


namespace NUMINAMATH_CALUDE_binary_1010_eq_10_l3302_330205

/-- Converts a binary number represented as a list of bits (least significant bit first) to its decimal equivalent. -/
def binary_to_decimal (bits : List Bool) : ℕ :=
  bits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- The binary representation of 1010₍₂₎ -/
def binary_1010 : List Bool := [false, true, false, true]

theorem binary_1010_eq_10 : binary_to_decimal binary_1010 = 10 := by
  sorry

end NUMINAMATH_CALUDE_binary_1010_eq_10_l3302_330205


namespace NUMINAMATH_CALUDE_task_completion_time_relation_l3302_330287

/-- 
Theorem: Given three individuals A, B, and C working on a task, where:
- A's time = m * (B and C's time together)
- B's time = n * (A and C's time together)
- C's time = k * (A and B's time together)
Then k can be expressed in terms of m and n as: k = (m + n + 2) / (mn - 1)
-/
theorem task_completion_time_relation (m n k : ℝ) (hm : m > 0) (hn : n > 0) (hk : k > 0) :
  (∃ (x y z : ℝ), x > 0 ∧ y > 0 ∧ z > 0 ∧ 
    (1 / x = m / (y + z)) ∧
    (1 / y = n / (x + z)) ∧
    (1 / z = k / (x + y))) →
  k = (m + n + 2) / (m * n - 1) :=
by sorry

end NUMINAMATH_CALUDE_task_completion_time_relation_l3302_330287


namespace NUMINAMATH_CALUDE_hotel_charge_difference_l3302_330231

/-- The charge for a single room at different hotels -/
structure HotelCharges where
  G : ℝ  -- Charge at hotel G
  R : ℝ  -- Charge at hotel R
  P : ℝ  -- Charge at hotel P

/-- The conditions given in the problem -/
def problem_conditions (h : HotelCharges) : Prop :=
  h.P = 0.9 * h.G ∧ 
  h.R = 1.125 * h.G

/-- The theorem stating the percentage difference between charges at hotel P and R -/
theorem hotel_charge_difference (h : HotelCharges) 
  (hcond : problem_conditions h) : 
  (h.R - h.P) / h.R = 0.2 := by
  sorry


end NUMINAMATH_CALUDE_hotel_charge_difference_l3302_330231


namespace NUMINAMATH_CALUDE_sector_central_angle_l3302_330208

theorem sector_central_angle (area : ℝ) (perimeter : ℝ) (r : ℝ) (l : ℝ) (α : ℝ) :
  area = 1 →
  perimeter = 4 →
  2 * r + l = perimeter →
  area = 1/2 * l * r →
  α = l / r →
  α = 2 := by
sorry

end NUMINAMATH_CALUDE_sector_central_angle_l3302_330208


namespace NUMINAMATH_CALUDE_chris_parents_gift_l3302_330288

/-- The amount of money Chris had before his birthday -/
def before_birthday : ℕ := 159

/-- The amount Chris received from his grandmother -/
def from_grandmother : ℕ := 25

/-- The amount Chris received from his aunt and uncle -/
def from_aunt_uncle : ℕ := 20

/-- The total amount Chris had after his birthday -/
def total_after_birthday : ℕ := 279

/-- The amount Chris's parents gave him -/
def from_parents : ℕ := total_after_birthday - before_birthday - from_grandmother - from_aunt_uncle

theorem chris_parents_gift : from_parents = 75 := by
  sorry

end NUMINAMATH_CALUDE_chris_parents_gift_l3302_330288


namespace NUMINAMATH_CALUDE_oil_leak_calculation_l3302_330253

theorem oil_leak_calculation (total_leaked : ℕ) (leaked_while_fixing : ℕ) 
  (h1 : total_leaked = 6206)
  (h2 : leaked_while_fixing = 3731) :
  total_leaked - leaked_while_fixing = 2475 := by
sorry

end NUMINAMATH_CALUDE_oil_leak_calculation_l3302_330253


namespace NUMINAMATH_CALUDE_cherries_purchase_l3302_330232

theorem cherries_purchase (cost_per_kg : ℝ) (short_amount : ℝ) (money_on_hand : ℝ) 
  (h1 : cost_per_kg = 8)
  (h2 : short_amount = 400)
  (h3 : money_on_hand = 1600) :
  (money_on_hand + short_amount) / cost_per_kg = 250 := by
  sorry

end NUMINAMATH_CALUDE_cherries_purchase_l3302_330232


namespace NUMINAMATH_CALUDE_arithmetic_sequence_third_term_l3302_330240

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

theorem arithmetic_sequence_third_term 
  (a : ℕ → ℝ) 
  (h_arithmetic : arithmetic_sequence a) 
  (h_sum : a 1 + a 2 + a 3 + a 4 + a 5 = 20) : 
  a 3 = 4 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_third_term_l3302_330240


namespace NUMINAMATH_CALUDE_tory_sold_to_grandmother_l3302_330236

/-- Represents the cookie sales problem for Tory's school fundraiser -/
def cookie_sales (grandmother_packs : ℕ) : Prop :=
  let total_packs : ℕ := 50
  let uncle_packs : ℕ := 7
  let neighbor_packs : ℕ := 5
  let remaining_packs : ℕ := 26
  grandmother_packs + uncle_packs + neighbor_packs + remaining_packs = total_packs

/-- Proves that Tory sold 12 packs of cookies to his grandmother -/
theorem tory_sold_to_grandmother :
  ∃ (x : ℕ), cookie_sales x ∧ x = 12 := by
  sorry

end NUMINAMATH_CALUDE_tory_sold_to_grandmother_l3302_330236


namespace NUMINAMATH_CALUDE_paper_fold_ratio_is_four_fifths_l3302_330215

/-- Represents the dimensions and folding of a rectangular piece of paper. -/
structure PaperFold where
  length : ℝ
  width : ℝ
  fold_ratio : ℝ
  division_parts : ℕ

/-- Calculates the ratio of the new visible area to the original area after folding. -/
def visible_area_ratio (paper : PaperFold) : ℝ :=
  -- Implementation details would go here
  sorry

/-- Theorem stating that for a specific paper folding scenario, the visible area ratio is 8/10. -/
theorem paper_fold_ratio_is_four_fifths :
  let paper : PaperFold := {
    length := 5,
    width := 2,
    fold_ratio := 1/2,
    division_parts := 3
  }
  visible_area_ratio paper = 8/10 := by
  sorry

end NUMINAMATH_CALUDE_paper_fold_ratio_is_four_fifths_l3302_330215


namespace NUMINAMATH_CALUDE_rectangular_prism_edge_sum_l3302_330267

theorem rectangular_prism_edge_sum (l w h : ℝ) : 
  l * w * h = 8 →                   -- Volume condition
  2 * (l * w + w * h + h * l) = 32 → -- Surface area condition
  ∃ q : ℝ, l = 2 / q ∧ w = 2 ∧ h = 2 * q → -- Geometric progression condition
  4 * (l + w + h) = 28 :=           -- Conclusion: sum of edge lengths
by sorry

end NUMINAMATH_CALUDE_rectangular_prism_edge_sum_l3302_330267


namespace NUMINAMATH_CALUDE_prop_equivalence_l3302_330282

theorem prop_equivalence (p q : Prop) 
  (h1 : p ∨ q) 
  (h2 : ¬(p ∧ q)) : 
  p ↔ ¬q := by
  sorry

end NUMINAMATH_CALUDE_prop_equivalence_l3302_330282


namespace NUMINAMATH_CALUDE_total_books_count_l3302_330295

theorem total_books_count (T : ℕ) : 
  (T = (1/4 : ℚ) * T + 10 + 
       (3/5 : ℚ) * (T - ((1/4 : ℚ) * T + 10)) - 5 + 
       12 + 13) → 
  T = 80 := by
  sorry

end NUMINAMATH_CALUDE_total_books_count_l3302_330295


namespace NUMINAMATH_CALUDE_log_product_plus_exp_equals_seven_l3302_330206

theorem log_product_plus_exp_equals_seven :
  Real.log 9 / Real.log 2 * (Real.log 4 / Real.log 3) + (2 : ℝ) ^ (Real.log 3 / Real.log 2) = 7 := by
  sorry

end NUMINAMATH_CALUDE_log_product_plus_exp_equals_seven_l3302_330206


namespace NUMINAMATH_CALUDE_systematic_sampling_theorem_l3302_330274

/-- Represents a systematic sampling scheme -/
structure SystematicSampling where
  total_employees : ℕ
  num_groups : ℕ
  group_size : ℕ
  sample_interval : ℕ

/-- Calculates the number to be drawn from a specific group -/
def number_from_group (s : SystematicSampling) (group : ℕ) (position : ℕ) : ℕ :=
  (group - 1) * s.sample_interval + position

/-- The main theorem to be proved -/
theorem systematic_sampling_theorem (s : SystematicSampling) 
  (h1 : s.total_employees = 200)
  (h2 : s.num_groups = 40)
  (h3 : s.group_size = 5)
  (h4 : s.sample_interval = 5)
  (h5 : number_from_group s 5 3 = 22) :
  number_from_group s 8 3 = 37 := by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_theorem_l3302_330274


namespace NUMINAMATH_CALUDE_expression_value_l3302_330257

theorem expression_value (a b c : ℝ) (h : a * b + b * c + c * a = 3) :
  (a * (b^2 + 3)) / (a + b) + (b * (c^2 + 3)) / (b + c) + (c * (a^2 + 3)) / (c + a) = 6 :=
by sorry

end NUMINAMATH_CALUDE_expression_value_l3302_330257


namespace NUMINAMATH_CALUDE_chinese_remainder_theorem_example_l3302_330224

theorem chinese_remainder_theorem_example : ∃ n : ℕ, 
  n > 0 ∧ 
  n % 3 = 2 ∧ 
  n % 5 = 3 ∧ 
  n % 7 = 2 ∧ 
  ∀ m : ℕ, m > 0 → m % 3 = 2 → m % 5 = 3 → m % 7 = 2 → m ≥ n :=
by sorry

end NUMINAMATH_CALUDE_chinese_remainder_theorem_example_l3302_330224


namespace NUMINAMATH_CALUDE_league_matches_count_l3302_330227

/-- The number of teams in the league -/
def num_teams : ℕ := 14

/-- The number of matches played in the league -/
def total_matches : ℕ := num_teams * (num_teams - 1)

/-- Theorem stating that the total number of matches in the league is 182 -/
theorem league_matches_count :
  total_matches = 182 :=
sorry

end NUMINAMATH_CALUDE_league_matches_count_l3302_330227


namespace NUMINAMATH_CALUDE_q_profit_share_l3302_330225

/-- Calculates the share of profit for a partner in a business partnership --/
def calculateProfitShare (investmentP investmentQ totalProfit : ℕ) : ℕ :=
  let totalInvestment := investmentP + investmentQ
  let shareQ := (investmentQ * totalProfit) / totalInvestment
  shareQ

/-- Theorem stating that Q's share of the profit is 7200 given the specified investments and total profit --/
theorem q_profit_share :
  calculateProfitShare 54000 36000 18000 = 7200 := by
  sorry

#eval calculateProfitShare 54000 36000 18000

end NUMINAMATH_CALUDE_q_profit_share_l3302_330225


namespace NUMINAMATH_CALUDE_degrees_to_radians_210_l3302_330292

theorem degrees_to_radians_210 : 
  (210 : ℝ) * (π / 180) = (7 * π) / 6 := by
  sorry

end NUMINAMATH_CALUDE_degrees_to_radians_210_l3302_330292


namespace NUMINAMATH_CALUDE_triangles_cover_two_thirds_l3302_330266

/-- Represents a tiling unit in the pattern -/
structure TilingUnit where
  /-- Side length of smaller shapes (triangles and squares) -/
  small_side : ℝ
  /-- Number of triangles in the unit -/
  num_triangles : ℕ
  /-- Number of squares in the unit -/
  num_squares : ℕ
  /-- Assertion that there are 2 triangles and 3 squares -/
  shape_count : num_triangles = 2 ∧ num_squares = 3
  /-- Assertion that all shapes have equal area -/
  equal_area : small_side^2 = 2 * (small_side^2 / 2)
  /-- Side length of the larger square formed by the unit -/
  large_side : ℝ
  /-- Assertion that large side is 3 times the small side -/
  side_relation : large_side = 3 * small_side

/-- Theorem stating that triangles cover 2/3 of the total area -/
theorem triangles_cover_two_thirds (u : TilingUnit) :
  (u.num_triangles * (u.small_side^2 / 2)) / u.large_side^2 = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_triangles_cover_two_thirds_l3302_330266


namespace NUMINAMATH_CALUDE_generalized_schur_inequality_l3302_330207

theorem generalized_schur_inequality (a b c t : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a^t * (a - b) * (a - c) + b^t * (b - c) * (b - a) + c^t * (c - a) * (c - b) ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_generalized_schur_inequality_l3302_330207


namespace NUMINAMATH_CALUDE_alice_bob_meet_l3302_330234

/-- The number of points on the circle -/
def n : ℕ := 18

/-- Alice's clockwise movement per turn -/
def alice_move : ℕ := 7

/-- Bob's counterclockwise movement per turn -/
def bob_move : ℕ := 13

/-- The number of turns after which Alice and Bob meet -/
def meeting_turns : ℕ := 9

theorem alice_bob_meet :
  (meeting_turns * alice_move) % n = (meeting_turns * (n - bob_move)) % n :=
sorry

end NUMINAMATH_CALUDE_alice_bob_meet_l3302_330234


namespace NUMINAMATH_CALUDE_quadratic_real_root_condition_l3302_330268

-- Define the quadratic equation
def quadratic (b x : ℝ) : ℝ := x^2 + b*x + 25

-- Define the condition for real roots
def has_real_root (b : ℝ) : Prop := ∃ x : ℝ, quadratic b x = 0

-- Theorem statement
theorem quadratic_real_root_condition (b : ℝ) :
  has_real_root b ↔ b ≤ -10 ∨ b ≥ 10 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_real_root_condition_l3302_330268


namespace NUMINAMATH_CALUDE_equation_solution_l3302_330235

theorem equation_solution : 
  ∃! x : ℚ, (10 * x + 2) / 4 - (3 * x - 6) / 18 = (2 * x + 4) / 3 ∧ x = 9 / 28 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3302_330235


namespace NUMINAMATH_CALUDE_cone_surface_area_theorem_sphere_surface_area_theorem_cylinder_surface_area_theorem_l3302_330296

-- Define the necessary variables and functions
variable (R : ℝ)
variable (x y z : ℝ)

-- Define the equations for the surfaces
def cone_equation (x y z : ℝ) : Prop := z^2 = 2*x*y
def sphere_equation (x y z R : ℝ) : Prop := x^2 + y^2 + z^2 = R^2
def cylinder_equation (x y R : ℝ) : Prop := x^2 + y^2 = R*x

-- Define the surface area functions
noncomputable def cone_surface_area (x_max y_max : ℝ) : ℝ := 
  sorry

noncomputable def sphere_surface_area_in_cylinder (R : ℝ) : ℝ := 
  sorry

noncomputable def cylinder_surface_area_in_sphere (R : ℝ) : ℝ := 
  sorry

-- State the theorems to be proven
theorem cone_surface_area_theorem :
  cone_surface_area 2 4 = 16 :=
sorry

theorem sphere_surface_area_theorem :
  sphere_surface_area_in_cylinder R = 2 * R^2 * (Real.pi - 2) :=
sorry

theorem cylinder_surface_area_theorem :
  cylinder_surface_area_in_sphere R = 4 * R^2 :=
sorry

end NUMINAMATH_CALUDE_cone_surface_area_theorem_sphere_surface_area_theorem_cylinder_surface_area_theorem_l3302_330296


namespace NUMINAMATH_CALUDE_parabola_coefficients_l3302_330255

/-- A parabola with equation y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The vertex of a parabola -/
def vertex (p : Parabola) : ℝ × ℝ := sorry

/-- Whether a parabola has a vertical axis of symmetry -/
def has_vertical_axis_of_symmetry (p : Parabola) : Prop := sorry

/-- Whether a point (x, y) lies on the parabola -/
def contains_point (p : Parabola) (x y : ℝ) : Prop := 
  y = p.a * x^2 + p.b * x + p.c

theorem parabola_coefficients :
  ∀ p : Parabola,
  vertex p = (2, -1) →
  has_vertical_axis_of_symmetry p →
  contains_point p 0 7 →
  (p.a, p.b, p.c) = (2, -8, 7) := by
  sorry

end NUMINAMATH_CALUDE_parabola_coefficients_l3302_330255


namespace NUMINAMATH_CALUDE_total_meows_in_five_minutes_l3302_330278

/-- The number of meows per minute for the first cat -/
def first_cat_meows : ℕ := 3

/-- The number of meows per minute for the second cat -/
def second_cat_meows : ℕ := 2 * first_cat_meows

/-- The number of meows per minute for the third cat -/
def third_cat_meows : ℕ := second_cat_meows / 3

/-- The duration in minutes -/
def duration : ℕ := 5

/-- Theorem: The total number of meows from all three cats in 5 minutes is 55 -/
theorem total_meows_in_five_minutes :
  first_cat_meows * duration + second_cat_meows * duration + third_cat_meows * duration = 55 := by
  sorry

end NUMINAMATH_CALUDE_total_meows_in_five_minutes_l3302_330278


namespace NUMINAMATH_CALUDE_root_difference_ratio_l3302_330297

/-- Given an equation x^4 - 7x - 3 = 0 with exactly two real roots a and b where a > b,
    the expression (a - b) / (a^4 - b^4) equals 1/7 -/
theorem root_difference_ratio (a b : ℝ) : 
  a > b → 
  a^4 - 7*a - 3 = 0 → 
  b^4 - 7*b - 3 = 0 → 
  (a - b) / (a^4 - b^4) = 1/7 := by
sorry

end NUMINAMATH_CALUDE_root_difference_ratio_l3302_330297


namespace NUMINAMATH_CALUDE_g_of_3_equals_2_l3302_330238

-- Define the function g
def g (x : ℝ) : ℝ := x^2 - 3*x + 2

-- Theorem statement
theorem g_of_3_equals_2 : g 3 = 2 := by sorry

end NUMINAMATH_CALUDE_g_of_3_equals_2_l3302_330238


namespace NUMINAMATH_CALUDE_orange_savings_percentage_l3302_330243

/-- Calculates the percentage of money saved when receiving free items instead of buying them -/
theorem orange_savings_percentage 
  (family_size : ℕ) 
  (planned_spending : ℝ) 
  (orange_price : ℝ) : 
  family_size = 4 → 
  planned_spending = 15 → 
  orange_price = 1.5 → 
  (family_size * orange_price / planned_spending) * 100 = 40 := by
sorry

end NUMINAMATH_CALUDE_orange_savings_percentage_l3302_330243


namespace NUMINAMATH_CALUDE_alice_winning_strategy_l3302_330252

theorem alice_winning_strategy :
  ∀ (n : ℕ), n < 10000000 ∧ n ≥ 1000000 →
  (n % 10 = 1 ∨ n % 10 = 3 ∨ n % 10 = 5 ∨ n % 10 = 7 ∨ n % 10 = 9) →
  ∃ (k : ℕ), k^7 % 10000000 = n := by
sorry

end NUMINAMATH_CALUDE_alice_winning_strategy_l3302_330252


namespace NUMINAMATH_CALUDE_no_functions_satisfying_condition_l3302_330210

theorem no_functions_satisfying_condition : 
  ¬∃ (f g : ℝ → ℝ), ∀ x y : ℝ, x ≠ y → |f x - f y| + |g x - g y| > 1 := by
  sorry

end NUMINAMATH_CALUDE_no_functions_satisfying_condition_l3302_330210


namespace NUMINAMATH_CALUDE_triangle_base_length_l3302_330250

/-- Given a triangle with area 36 cm² and height 8 cm, its base length is 9 cm. -/
theorem triangle_base_length (area : ℝ) (height : ℝ) (base : ℝ) :
  area = 36 →
  height = 8 →
  area = (base * height) / 2 →
  base = 9 := by
sorry

end NUMINAMATH_CALUDE_triangle_base_length_l3302_330250


namespace NUMINAMATH_CALUDE_marcos_dad_strawberries_l3302_330214

theorem marcos_dad_strawberries (marco_weight : ℕ) (total_weight : ℕ) 
  (h1 : marco_weight = 15)
  (h2 : total_weight = 37) :
  total_weight - marco_weight = 22 := by
  sorry

end NUMINAMATH_CALUDE_marcos_dad_strawberries_l3302_330214


namespace NUMINAMATH_CALUDE_complex_power_sum_l3302_330291

theorem complex_power_sum (z : ℂ) (h : z + 1/z = 2 * Real.cos (5 * Real.pi / 180)) :
  z^100 + 1/z^100 = -2 * Real.cos (40 * Real.pi / 180) := by
  sorry

end NUMINAMATH_CALUDE_complex_power_sum_l3302_330291


namespace NUMINAMATH_CALUDE_octagon_area_in_square_l3302_330277

/-- Given a square with side length s, the area of the octagon formed by connecting each vertex
    to the midpoints of the opposite two sides is s^2/6. -/
theorem octagon_area_in_square (s : ℝ) (h : s > 0) :
  let square_area := s^2
  let octagon_area := square_area / 6
  octagon_area = square_area / 6 := by
sorry

end NUMINAMATH_CALUDE_octagon_area_in_square_l3302_330277


namespace NUMINAMATH_CALUDE_factorial_sum_calculation_l3302_330247

theorem factorial_sum_calculation : 6 * Nat.factorial 6 + 5 * Nat.factorial 5 + 2 * Nat.factorial 5 = 5160 := by
  sorry

end NUMINAMATH_CALUDE_factorial_sum_calculation_l3302_330247


namespace NUMINAMATH_CALUDE_robotics_club_neither_cs_nor_electronics_l3302_330271

/-- The number of students in the robotics club who take neither computer science nor electronics -/
theorem robotics_club_neither_cs_nor_electronics :
  let total_students : ℕ := 60
  let cs_students : ℕ := 40
  let electronics_students : ℕ := 35
  let both_cs_and_electronics : ℕ := 25
  let neither_cs_nor_electronics : ℕ := total_students - (cs_students + electronics_students - both_cs_and_electronics)
  neither_cs_nor_electronics = 10 := by
sorry

end NUMINAMATH_CALUDE_robotics_club_neither_cs_nor_electronics_l3302_330271


namespace NUMINAMATH_CALUDE_prime_pairs_divisibility_l3302_330280

theorem prime_pairs_divisibility (p q : ℕ) : 
  p.Prime ∧ q.Prime ∧ p < 2005 ∧ q < 2005 ∧ 
  (q ∣ p^2 + 8) ∧ (p ∣ q^2 + 8) → 
  ((p = 2 ∧ q = 2) ∨ (p = 881 ∧ q = 89) ∨ (p = 89 ∧ q = 881)) := by
  sorry

end NUMINAMATH_CALUDE_prime_pairs_divisibility_l3302_330280


namespace NUMINAMATH_CALUDE_line_mb_value_l3302_330258

/-- Given a line y = mx + b passing through the points (0, -3) and (1, -1), prove that mb = 6 -/
theorem line_mb_value (m b : ℝ) : 
  (0 : ℝ) = m * 0 + b →  -- The line passes through (0, -3)
  (-3 : ℝ) = m * 0 + b →  -- The line passes through (0, -3)
  (-1 : ℝ) = m * 1 + b →  -- The line passes through (1, -1)
  m * b = 6 := by
sorry

end NUMINAMATH_CALUDE_line_mb_value_l3302_330258


namespace NUMINAMATH_CALUDE_tournament_matches_l3302_330272

def matches_in_group (n : ℕ) : ℕ := n * (n - 1) / 2

theorem tournament_matches : 
  let group_a_players : ℕ := 6
  let group_b_players : ℕ := 5
  matches_in_group group_a_players + matches_in_group group_b_players = 25 := by
  sorry

end NUMINAMATH_CALUDE_tournament_matches_l3302_330272


namespace NUMINAMATH_CALUDE_line_tangent_to_parabola_l3302_330200

/-- The line 4x + 3y + k = 0 is tangent to the parabola y^2 = 16x if and only if k = 9 -/
theorem line_tangent_to_parabola (k : ℝ) : 
  (∀ x y : ℝ, 4 * x + 3 * y + k = 0 → y^2 = 16 * x) ↔ k = 9 := by
  sorry

end NUMINAMATH_CALUDE_line_tangent_to_parabola_l3302_330200


namespace NUMINAMATH_CALUDE_jake_and_sister_weight_l3302_330273

/-- Given Jake's current weight and the condition about his weight relative to his sister's,
    prove that their combined weight is 212 pounds. -/
theorem jake_and_sister_weight (jake_weight : ℕ) (sister_weight : ℕ) : 
  jake_weight = 152 →
  jake_weight - 32 = 2 * sister_weight →
  jake_weight + sister_weight = 212 := by
  sorry

end NUMINAMATH_CALUDE_jake_and_sister_weight_l3302_330273


namespace NUMINAMATH_CALUDE_green_balls_removal_l3302_330275

theorem green_balls_removal (total : ℕ) (green_percent : ℚ) (target_percent : ℚ) 
  (h_total : total = 600)
  (h_green_percent : green_percent = 70/100)
  (h_target_percent : target_percent = 60/100) :
  ∃ x : ℕ, 
    (↑x ≤ green_percent * ↑total) ∧ 
    ((green_percent * ↑total - ↑x) / (↑total - ↑x) = target_percent) ∧
    x = 150 := by
  sorry

end NUMINAMATH_CALUDE_green_balls_removal_l3302_330275


namespace NUMINAMATH_CALUDE_special_numbers_are_correct_l3302_330202

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def is_special (n : ℕ) : Prop :=
  n ≥ 1 ∧ n ≤ 9999 ∧
  let ab := n / 100
  let cd := n % 100
  is_perfect_square (ab - cd) ∧
  is_perfect_square (ab + cd) ∧
  (ab - cd) ∣ (ab + cd) ∧
  (ab + cd) ∣ n

def special_numbers : Finset ℕ :=
  {0100, 0400, 0900, 1600, 2500, 3600, 4900, 6400, 8100, 0504, 2016, 4536, 8064}

theorem special_numbers_are_correct :
  ∀ n : ℕ, is_special n ↔ n ∈ special_numbers := by sorry

end NUMINAMATH_CALUDE_special_numbers_are_correct_l3302_330202


namespace NUMINAMATH_CALUDE_probability_three_green_marbles_l3302_330264

/-- The probability of choosing exactly k successes in n trials with probability p of success in each trial. -/
def binomial_probability (n k : ℕ) (p : ℚ) : ℚ :=
  (n.choose k : ℚ) * p^k * (1 - p)^(n - k)

/-- The number of green marbles -/
def green_marbles : ℕ := 8

/-- The number of purple marbles -/
def purple_marbles : ℕ := 7

/-- The total number of marbles -/
def total_marbles : ℕ := green_marbles + purple_marbles

/-- The number of trials -/
def num_trials : ℕ := 7

/-- The number of desired green marbles -/
def desired_green : ℕ := 3

/-- The probability of choosing a green marble in one trial -/
def prob_green : ℚ := green_marbles / total_marbles

theorem probability_three_green_marbles :
  binomial_probability num_trials desired_green prob_green = 8604112 / 15946875 := by
  sorry

end NUMINAMATH_CALUDE_probability_three_green_marbles_l3302_330264


namespace NUMINAMATH_CALUDE_product_of_three_numbers_l3302_330269

theorem product_of_three_numbers (x y z : ℝ) : 
  x > 0 → y > 0 → z > 0 → 
  x * y * z = 1 → 
  x + 1 / z = 8 → 
  y + 1 / x = 20 → 
  z + 1 / y = 10 / 53 := by
sorry

end NUMINAMATH_CALUDE_product_of_three_numbers_l3302_330269


namespace NUMINAMATH_CALUDE_prudence_sleep_weeks_l3302_330259

/-- Represents Prudence's sleep schedule and total sleep time --/
structure SleepSchedule where
  weekdayNights : Nat  -- Number of weekday nights (Sun-Thurs)
  weekendNights : Nat  -- Number of weekend nights (Fri-Sat)
  napDays : Nat        -- Number of days with naps
  weekdaySleep : Nat   -- Hours of sleep on weekday nights
  weekendSleep : Nat   -- Hours of sleep on weekend nights
  napDuration : Nat    -- Duration of naps in hours
  totalSleep : Nat     -- Total hours of sleep

/-- Calculates the number of weeks required to reach the total sleep time --/
def weeksToReachSleep (schedule : SleepSchedule) : Nat :=
  let weeklySleeep := 
    schedule.weekdayNights * schedule.weekdaySleep +
    schedule.weekendNights * schedule.weekendSleep +
    schedule.napDays * schedule.napDuration
  schedule.totalSleep / weeklySleeep

/-- Theorem: Given Prudence's sleep schedule, it takes 4 weeks to reach 200 hours of sleep --/
theorem prudence_sleep_weeks : 
  weeksToReachSleep {
    weekdayNights := 5,
    weekendNights := 2,
    napDays := 2,
    weekdaySleep := 6,
    weekendSleep := 9,
    napDuration := 1,
    totalSleep := 200
  } = 4 := by
  sorry


end NUMINAMATH_CALUDE_prudence_sleep_weeks_l3302_330259


namespace NUMINAMATH_CALUDE_number_difference_l3302_330299

theorem number_difference (a b : ℕ) 
  (sum_eq : a + b = 21352)
  (b_div_9 : ∃ k, b = 9 * k)
  (relation : 10 * a + 1 = b) : 
  b - a = 17470 := by sorry

end NUMINAMATH_CALUDE_number_difference_l3302_330299


namespace NUMINAMATH_CALUDE_house_purchase_l3302_330228

/-- Represents a number in base s -/
def BaseS (n : ℕ) (s : ℕ) : ℕ → ℕ
| 0 => 0
| (k+1) => (n % s) + s * BaseS (n / s) s k

theorem house_purchase (s : ℕ) 
  (h1 : BaseS 530 s 2 + BaseS 450 s 2 = BaseS 1100 s 3) : s = 8 :=
sorry

end NUMINAMATH_CALUDE_house_purchase_l3302_330228
