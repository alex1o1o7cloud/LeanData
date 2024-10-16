import Mathlib

namespace NUMINAMATH_CALUDE_perimeter_difference_zero_l119_11926

/-- Perimeter of a rectangle --/
def rectanglePerimeter (length width : ℕ) : ℕ := 2 * (length + width)

/-- Perimeter of Figure 1 --/
def figure1Perimeter : ℕ := rectanglePerimeter 6 1 + 4

/-- Perimeter of Figure 2 --/
def figure2Perimeter : ℕ := rectanglePerimeter 7 2

theorem perimeter_difference_zero : figure1Perimeter = figure2Perimeter := by
  sorry

#eval figure1Perimeter
#eval figure2Perimeter

end NUMINAMATH_CALUDE_perimeter_difference_zero_l119_11926


namespace NUMINAMATH_CALUDE_last_digit_of_N_l119_11983

theorem last_digit_of_N (total_coins : ℕ) (h : total_coins = 3080) : 
  ∃ N : ℕ, (N * (N + 1)) / 2 = total_coins ∧ N % 10 = 8 := by
  sorry

end NUMINAMATH_CALUDE_last_digit_of_N_l119_11983


namespace NUMINAMATH_CALUDE_subtracted_number_l119_11962

theorem subtracted_number (x : ℚ) : x = 40 → ∃ y : ℚ, ((x / 4) * 5 + 10) - y = 48 ∧ y = 12 := by
  sorry

end NUMINAMATH_CALUDE_subtracted_number_l119_11962


namespace NUMINAMATH_CALUDE_inequality_proof_l119_11940

theorem inequality_proof (a₁ a₂ a₃ a₄ : ℝ) (h₁ : a₁ > 0) (h₂ : a₂ > 0) (h₃ : a₃ > 0) (h₄ : a₄ > 0) :
  (a₁ + a₃) / (a₁ + a₂) + (a₂ + a₄) / (a₂ + a₃) + (a₃ + a₁) / (a₃ + a₄) + (a₄ + a₂) / (a₄ + a₁) ≥ 4 :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l119_11940


namespace NUMINAMATH_CALUDE_inequality_solution_l119_11908

theorem inequality_solution (x : ℝ) :
  x ≠ 1 ∧ x ≠ 2 →
  (x^3 - x^2 - 6*x) / (x^2 - 3*x + 2) > 0 ↔ (-2 < x ∧ x < 0) ∨ (1 < x ∧ x < 2) ∨ (3 < x) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l119_11908


namespace NUMINAMATH_CALUDE_ellipse_focus_l119_11911

theorem ellipse_focus (center : ℝ × ℝ) (major_axis : ℝ) (minor_axis : ℝ) :
  center = (3, -1) →
  major_axis = 6 →
  minor_axis = 4 →
  let focus_distance := Real.sqrt ((major_axis / 2)^2 - (minor_axis / 2)^2)
  let focus_x := center.1 + focus_distance
  (focus_x, center.2) = (3 + Real.sqrt 5, -1) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_focus_l119_11911


namespace NUMINAMATH_CALUDE_composition_zero_iff_rank_sum_eq_dim_l119_11927

variable {V : Type*} [AddCommGroup V] [Module ℝ V] [FiniteDimensional ℝ V]
variable (T U : V →ₗ[ℝ] V)

theorem composition_zero_iff_rank_sum_eq_dim (h : Function.Bijective (T + U)) :
  (T.comp U = 0 ∧ U.comp T = 0) ↔ LinearMap.rank T + LinearMap.rank U = FiniteDimensional.finrank ℝ V :=
sorry

end NUMINAMATH_CALUDE_composition_zero_iff_rank_sum_eq_dim_l119_11927


namespace NUMINAMATH_CALUDE_ceiling_of_negative_fraction_squared_l119_11935

theorem ceiling_of_negative_fraction_squared : ⌈(-7/4)^2⌉ = 4 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_of_negative_fraction_squared_l119_11935


namespace NUMINAMATH_CALUDE_triangle_angle_proof_l119_11992

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    and vectors m = (√3, -1) and n = (cos A, sin A), prove that if m ⊥ n and
    a * cos B + b * cos A = c * sin C, then B = π/6. -/
theorem triangle_angle_proof (a b c A B C : ℝ) (m n : ℝ × ℝ) :
  a > 0 → b > 0 → c > 0 →
  0 < A → A < π →
  0 < B → B < π →
  0 < C → C < π →
  A + B + C = π →
  m = (Real.sqrt 3, -1) →
  n = (Real.cos A, Real.sin A) →
  m.1 * n.1 + m.2 * n.2 = 0 →
  a * Real.cos B + b * Real.cos A = c * Real.sin C →
  B = π / 6 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_proof_l119_11992


namespace NUMINAMATH_CALUDE_pure_imaginary_fraction_l119_11923

theorem pure_imaginary_fraction (a : ℝ) : 
  (((1 : ℂ) + 2 * Complex.I) / (a + Complex.I)).re = 0 ∧ 
  (((1 : ℂ) + 2 * Complex.I) / (a + Complex.I)).im ≠ 0 → 
  a = -2 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_fraction_l119_11923


namespace NUMINAMATH_CALUDE_problem_N4_l119_11969

theorem problem_N4 (a b : ℕ+) 
  (h : ∀ n : ℕ+, n > 2020^2020 → 
    ∃ m : ℕ+, Nat.Coprime m.val n.val ∧ (a^n.val + b^n.val ∣ a^m.val + b^m.val)) :
  a = b := by
  sorry

end NUMINAMATH_CALUDE_problem_N4_l119_11969


namespace NUMINAMATH_CALUDE_difference_of_squares_l119_11948

theorem difference_of_squares : 303^2 - 297^2 = 3600 := by sorry

end NUMINAMATH_CALUDE_difference_of_squares_l119_11948


namespace NUMINAMATH_CALUDE_conic_eccentricity_l119_11975

-- Define the geometric sequence
def is_geometric_sequence (a : ℝ) : Prop := a * a = 81

-- Define the conic section
def conic_section (a : ℝ) (x y : ℝ) : Prop := x^2 + y^2 / a = 1

-- Define the eccentricity
def eccentricity (e : ℝ) (a : ℝ) : Prop :=
  (e = Real.sqrt 10 ∧ a = -9) ∨ (e = 2 * Real.sqrt 2 / 3 ∧ a = 9)

-- Theorem statement
theorem conic_eccentricity (a : ℝ) (e : ℝ) :
  is_geometric_sequence a →
  (∃ x y, conic_section a x y) →
  eccentricity e a :=
sorry

end NUMINAMATH_CALUDE_conic_eccentricity_l119_11975


namespace NUMINAMATH_CALUDE_cuboid_edge_lengths_l119_11959

theorem cuboid_edge_lengths (a b c : ℝ) : 
  (a * b : ℝ) / (b * c) = 16 / 21 →
  (a * b : ℝ) / (a * c) = 16 / 28 →
  a^2 + b^2 + c^2 = 29^2 →
  a = 16 ∧ b = 12 ∧ c = 21 := by
  sorry

end NUMINAMATH_CALUDE_cuboid_edge_lengths_l119_11959


namespace NUMINAMATH_CALUDE_cuboid_area_and_volume_l119_11918

/-- Represents a cuboid with given dimensions -/
structure Cuboid where
  length : ℝ
  breadth : ℝ
  height : ℝ

/-- Calculate the surface area of a cuboid -/
def surfaceArea (c : Cuboid) : ℝ :=
  2 * (c.length * c.breadth + c.length * c.height + c.breadth * c.height)

/-- Calculate the volume of a cuboid -/
def volume (c : Cuboid) : ℝ :=
  c.length * c.breadth * c.height

/-- Theorem stating the surface area and volume of a specific cuboid -/
theorem cuboid_area_and_volume :
  let c : Cuboid := ⟨10, 8, 6⟩
  surfaceArea c = 376 ∧ volume c = 480 := by
  sorry

#check cuboid_area_and_volume

end NUMINAMATH_CALUDE_cuboid_area_and_volume_l119_11918


namespace NUMINAMATH_CALUDE_rock_max_height_l119_11919

/-- The height function of the rock -/
def h (t : ℝ) : ℝ := 150 * t - 15 * t^2

/-- The maximum height reached by the rock -/
theorem rock_max_height : ∃ (t : ℝ), ∀ (s : ℝ), h s ≤ h t ∧ h t = 375 := by
  sorry

end NUMINAMATH_CALUDE_rock_max_height_l119_11919


namespace NUMINAMATH_CALUDE_sarah_trucks_l119_11967

theorem sarah_trucks (trucks_to_jeff trucks_to_amy trucks_left : ℕ) 
  (h1 : trucks_to_jeff = 13)
  (h2 : trucks_to_amy = 21)
  (h3 : trucks_left = 38) :
  trucks_to_jeff + trucks_to_amy + trucks_left = 72 := by
  sorry

end NUMINAMATH_CALUDE_sarah_trucks_l119_11967


namespace NUMINAMATH_CALUDE_product_of_1010_2_and_102_3_l119_11981

/-- Converts a binary number represented as a list of digits to its decimal value -/
def binary_to_decimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * 2^i) 0

/-- Converts a ternary number represented as a list of digits to its decimal value -/
def ternary_to_decimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * 3^i) 0

theorem product_of_1010_2_and_102_3 : 
  let binary_num := [0, 1, 0, 1]  -- 1010 in binary, least significant bit first
  let ternary_num := [2, 0, 1]    -- 102 in ternary, least significant digit first
  (binary_to_decimal binary_num) * (ternary_to_decimal ternary_num) = 110 := by
  sorry

end NUMINAMATH_CALUDE_product_of_1010_2_and_102_3_l119_11981


namespace NUMINAMATH_CALUDE_power_of_power_l119_11972

theorem power_of_power (x : ℝ) : (x^3)^2 = x^6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_l119_11972


namespace NUMINAMATH_CALUDE_vampire_survival_l119_11971

/-- The number of pints in a gallon -/
def pints_per_gallon : ℕ := 8

/-- The number of days in a week -/
def days_per_week : ℕ := 7

/-- The amount of blood (in gallons) a vampire needs per week -/
def blood_needed_per_week : ℕ := 7

/-- The amount of blood (in pints) a vampire sucks from each person -/
def blood_per_person : ℕ := 2

/-- The number of people a vampire needs to suck from each day to survive -/
def people_per_day : ℕ := 4

theorem vampire_survival :
  (blood_needed_per_week * pints_per_gallon) / days_per_week / blood_per_person = people_per_day :=
sorry

end NUMINAMATH_CALUDE_vampire_survival_l119_11971


namespace NUMINAMATH_CALUDE_task_completion_time_l119_11924

/-- Ram's efficiency is half of Krish's, and Ram takes 27 days to complete a task alone.
    This theorem proves that Ram and Krish working together will complete the task in 9 days. -/
theorem task_completion_time (ram_efficiency krish_efficiency : ℝ) 
  (h1 : ram_efficiency = (1 / 2) * krish_efficiency) 
  (h2 : ram_efficiency * 27 = 1) : 
  (ram_efficiency + krish_efficiency) * 9 = 1 := by
  sorry

end NUMINAMATH_CALUDE_task_completion_time_l119_11924


namespace NUMINAMATH_CALUDE_max_elevation_l119_11985

/-- The elevation function of a particle projected vertically upward -/
def s (t : ℝ) : ℝ := 160 * t - 16 * t^2

/-- The maximum elevation reached by the particle -/
theorem max_elevation : ∃ (t : ℝ), ∀ (u : ℝ), s u ≤ s t ∧ s t = 400 := by
  sorry

end NUMINAMATH_CALUDE_max_elevation_l119_11985


namespace NUMINAMATH_CALUDE_hold_age_ratio_l119_11912

theorem hold_age_ratio (mother_age : ℕ) (son_age : ℕ) (h1 : mother_age = 36) (h2 : mother_age = 3 * son_age) :
  (mother_age - 8) / (son_age - 8) = 7 := by
  sorry

end NUMINAMATH_CALUDE_hold_age_ratio_l119_11912


namespace NUMINAMATH_CALUDE_holiday_savings_l119_11915

theorem holiday_savings (victory_savings sam_savings : ℕ) : 
  victory_savings = sam_savings - 100 →
  victory_savings + sam_savings = 1900 →
  sam_savings = 1000 := by
sorry

end NUMINAMATH_CALUDE_holiday_savings_l119_11915


namespace NUMINAMATH_CALUDE_paul_bought_six_chocolate_boxes_l119_11910

/-- Represents the number of boxes of chocolate candy Paul bought. -/
def chocolate_boxes : ℕ := sorry

/-- Represents the number of boxes of caramel candy Paul bought. -/
def caramel_boxes : ℕ := 4

/-- Represents the number of pieces of candy in each box. -/
def pieces_per_box : ℕ := 9

/-- Represents the total number of candies Paul had. -/
def total_candies : ℕ := 90

/-- Theorem stating that Paul bought 6 boxes of chocolate candy. -/
theorem paul_bought_six_chocolate_boxes :
  chocolate_boxes = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_paul_bought_six_chocolate_boxes_l119_11910


namespace NUMINAMATH_CALUDE_harkamal_grapes_purchase_l119_11913

/-- The amount of grapes purchased by Harkamal -/
def grapes_kg : ℝ := 8

/-- The cost of grapes per kg -/
def grapes_cost_per_kg : ℝ := 70

/-- The cost of mangoes per kg -/
def mangoes_cost_per_kg : ℝ := 60

/-- The amount of mangoes purchased by Harkamal -/
def mangoes_kg : ℝ := 9

/-- The total amount paid by Harkamal -/
def total_paid : ℝ := 1100

theorem harkamal_grapes_purchase :
  grapes_kg * grapes_cost_per_kg + mangoes_kg * mangoes_cost_per_kg = total_paid :=
by sorry

end NUMINAMATH_CALUDE_harkamal_grapes_purchase_l119_11913


namespace NUMINAMATH_CALUDE_smallest_two_digit_prime_with_composite_reversal_l119_11970

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → n % d ≠ 0

def is_composite (n : ℕ) : Prop := n > 1 ∧ ∃ d : ℕ, d > 1 ∧ d < n ∧ n % d = 0

def reverse_digits (n : ℕ) : ℕ :=
  let units := n % 10
  let tens := n / 10
  units * 10 + tens

def is_two_digit (n : ℕ) : Prop := n ≥ 10 ∧ n ≤ 99

def tens_digit_is_two (n : ℕ) : Prop := (n / 10) % 10 = 2

theorem smallest_two_digit_prime_with_composite_reversal :
  ∃ n : ℕ,
    is_two_digit n ∧
    tens_digit_is_two n ∧
    is_prime n ∧
    is_composite (reverse_digits n) ∧
    (∀ m : ℕ, is_two_digit m → tens_digit_is_two m → is_prime m → 
      is_composite (reverse_digits m) → n ≤ m) ∧
    n = 23 :=
sorry

end NUMINAMATH_CALUDE_smallest_two_digit_prime_with_composite_reversal_l119_11970


namespace NUMINAMATH_CALUDE_smallest_perimeter_l119_11906

/-- Triangle PQR with intersection point J of angle bisectors of ∠Q and ∠R -/
structure TrianglePQR where
  PQ : ℕ+
  QR : ℕ+
  QJ : ℕ+
  isIsosceles : PQ = PQ
  angleIntersection : QJ = 10

/-- The perimeter of triangle PQR -/
def perimeter (t : TrianglePQR) : ℕ := 2 * t.PQ + t.QR

/-- The smallest possible perimeter of triangle PQR satisfying the given conditions -/
theorem smallest_perimeter :
  ∀ t : TrianglePQR, perimeter t ≥ 40 :=
sorry

end NUMINAMATH_CALUDE_smallest_perimeter_l119_11906


namespace NUMINAMATH_CALUDE_sin_cos_pi_12_equals_neg_sqrt_2_l119_11946

theorem sin_cos_pi_12_equals_neg_sqrt_2 :
  Real.sin (π / 12) - Real.sqrt 3 * Real.cos (π / 12) = -Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_pi_12_equals_neg_sqrt_2_l119_11946


namespace NUMINAMATH_CALUDE_exam_pass_percentage_l119_11963

/-- Given an examination where 260 students failed out of 400 total students,
    prove that 35% of students passed the examination. -/
theorem exam_pass_percentage :
  let total_students : ℕ := 400
  let failed_students : ℕ := 260
  let passed_students : ℕ := total_students - failed_students
  let pass_percentage : ℚ := (passed_students : ℚ) / (total_students : ℚ) * 100
  pass_percentage = 35 := by
  sorry

end NUMINAMATH_CALUDE_exam_pass_percentage_l119_11963


namespace NUMINAMATH_CALUDE_floor_paving_cost_l119_11938

-- Define the room dimensions
def room_length : ℝ := 6
def room_width : ℝ := 4.75

-- Define the cost per square meter
def cost_per_sqm : ℝ := 900

-- Define the function to calculate the area of a rectangle
def area (length width : ℝ) : ℝ := length * width

-- Define the function to calculate the total cost
def total_cost (length width cost_per_sqm : ℝ) : ℝ :=
  area length width * cost_per_sqm

-- State the theorem
theorem floor_paving_cost :
  total_cost room_length room_width cost_per_sqm = 25650 := by sorry

end NUMINAMATH_CALUDE_floor_paving_cost_l119_11938


namespace NUMINAMATH_CALUDE_library_books_total_l119_11984

theorem library_books_total (initial_books additional_books : ℕ) 
  (h1 : initial_books = 54)
  (h2 : additional_books = 23) :
  initial_books + additional_books = 77 := by
sorry

end NUMINAMATH_CALUDE_library_books_total_l119_11984


namespace NUMINAMATH_CALUDE_blue_sequins_count_l119_11993

/-- The number of blue sequins in each row of Jane's costume. -/
def blue_sequins_per_row : ℕ := 
  let total_sequins : ℕ := 162
  let blue_rows : ℕ := 6
  let purple_rows : ℕ := 5
  let purple_per_row : ℕ := 12
  let green_rows : ℕ := 9
  let green_per_row : ℕ := 6
  (total_sequins - purple_rows * purple_per_row - green_rows * green_per_row) / blue_rows

theorem blue_sequins_count : blue_sequins_per_row = 8 := by
  sorry

end NUMINAMATH_CALUDE_blue_sequins_count_l119_11993


namespace NUMINAMATH_CALUDE_roof_dimension_difference_l119_11999

theorem roof_dimension_difference (width : ℝ) (length : ℝ) : 
  width > 0 →
  length = 4 * width →
  width * length = 768 →
  length - width = 24 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_roof_dimension_difference_l119_11999


namespace NUMINAMATH_CALUDE_original_to_doubled_ratio_l119_11945

theorem original_to_doubled_ratio (x : ℝ) : 3 * (2 * x + 6) = 72 → x / (2 * x) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_original_to_doubled_ratio_l119_11945


namespace NUMINAMATH_CALUDE_trenton_earning_goal_l119_11931

/-- Calculates the earning goal for a salesperson given their fixed weekly earnings,
    commission rate, and sales amount. -/
def earning_goal (fixed_earnings : ℝ) (commission_rate : ℝ) (sales : ℝ) : ℝ :=
  fixed_earnings + commission_rate * sales

/-- Proves that Trenton's earning goal for the week is $500 given the specified conditions. -/
theorem trenton_earning_goal :
  let fixed_earnings : ℝ := 190
  let commission_rate : ℝ := 0.04
  let sales : ℝ := 7750
  earning_goal fixed_earnings commission_rate sales = 500 := by
  sorry


end NUMINAMATH_CALUDE_trenton_earning_goal_l119_11931


namespace NUMINAMATH_CALUDE_triangle_side_length_l119_11957

theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) : 
  A = π / 6 →  -- 30° in radians
  B = π / 4 →  -- 45° in radians
  a = Real.sqrt 2 →
  (Real.sin A) * b = (Real.sin B) * a →
  b = 2 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l119_11957


namespace NUMINAMATH_CALUDE_jacket_cost_is_30_l119_11903

/-- Represents the cost of clothing items in a discount store. -/
structure ClothingCost where
  sweater : ℝ
  jacket : ℝ

/-- Represents a shipment of clothing items. -/
structure Shipment where
  sweaters : ℕ
  jackets : ℕ
  totalCost : ℝ

/-- The conditions of the problem. -/
def problemConditions (cost : ClothingCost) : Prop :=
  ∃ (shipment1 shipment2 : Shipment),
    shipment1.sweaters = 10 ∧
    shipment1.jackets = 20 ∧
    shipment1.totalCost = 800 ∧
    shipment2.sweaters = 5 ∧
    shipment2.jackets = 15 ∧
    shipment2.totalCost = 550 ∧
    shipment1.totalCost = cost.sweater * shipment1.sweaters + cost.jacket * shipment1.jackets ∧
    shipment2.totalCost = cost.sweater * shipment2.sweaters + cost.jacket * shipment2.jackets

/-- The main theorem stating that under the given conditions, the cost of a jacket is $30. -/
theorem jacket_cost_is_30 :
  ∀ (cost : ClothingCost), problemConditions cost → cost.jacket = 30 := by
  sorry


end NUMINAMATH_CALUDE_jacket_cost_is_30_l119_11903


namespace NUMINAMATH_CALUDE_problem_solution_l119_11944

noncomputable def f (x a : ℝ) : ℝ := |x - 2| + |x - a^2|

theorem problem_solution :
  (∃ (a : ℝ), ∀ (x : ℝ), f x a ≤ a ↔ 1 ≤ a ∧ a ≤ 2) ∧
  (∀ (m n : ℝ), m > 0 → n > 0 → m + 2*n = 2 → 1/m + 1/n ≥ 3/2 + Real.sqrt 2) :=
by sorry

#check problem_solution

end NUMINAMATH_CALUDE_problem_solution_l119_11944


namespace NUMINAMATH_CALUDE_percentage_problem_l119_11925

theorem percentage_problem (P : ℝ) (N : ℝ) 
  (h1 : (P / 100) * N = 200)
  (h2 : 1.2 * N = 1200) : P = 20 := by
sorry

end NUMINAMATH_CALUDE_percentage_problem_l119_11925


namespace NUMINAMATH_CALUDE_triangle_properties_l119_11939

noncomputable section

/-- Triangle ABC with area S and sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  S : ℝ
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The main theorem stating the properties of the triangle -/
theorem triangle_properties (t : Triangle) 
  (h1 : t.S = (3/2) * t.b * t.c * Real.cos t.A)
  (h2 : t.C = π/4)
  (h3 : t.S = 24) :
  Real.cos t.B = Real.sqrt 5 / 5 ∧ t.b = 8 := by
  sorry

end

end NUMINAMATH_CALUDE_triangle_properties_l119_11939


namespace NUMINAMATH_CALUDE_prob_product_one_four_dice_l119_11902

/-- The number of sides on a standard die -/
def dieSides : ℕ := 6

/-- The probability of rolling a specific number on a standard die -/
def probSingleDie : ℚ := 1 / dieSides

/-- The number of dice rolled -/
def numDice : ℕ := 4

/-- The probability of rolling all ones on multiple dice -/
def probAllOnes : ℚ := probSingleDie ^ numDice

theorem prob_product_one_four_dice :
  probAllOnes = 1 / 1296 := by sorry

end NUMINAMATH_CALUDE_prob_product_one_four_dice_l119_11902


namespace NUMINAMATH_CALUDE_calculation_proof_l119_11909

theorem calculation_proof : 
  let tan30 := Real.sqrt 3 / 3
  let π := 3.14
  (1/3)⁻¹ - Real.sqrt 27 + 3 * tan30 + (π - 3.14)^0 = 4 - 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l119_11909


namespace NUMINAMATH_CALUDE_remainder_3005_div_98_l119_11920

theorem remainder_3005_div_98 : 3005 % 98 = 65 := by
  sorry

end NUMINAMATH_CALUDE_remainder_3005_div_98_l119_11920


namespace NUMINAMATH_CALUDE_max_ab_value_l119_11949

/-- The function f(x) defined in the problem -/
def f (a b x : ℝ) : ℝ := 4 * x^3 - a * x^2 - 2 * b * x + 2

/-- The derivative of f(x) with respect to x -/
def f_deriv (a b x : ℝ) : ℝ := 12 * x^2 - 2 * a * x - 2 * b

theorem max_ab_value (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_extremum : f_deriv a b 1 = 0) :
  (∃ (max_ab : ℝ), max_ab = 9 ∧ ∀ (a' b' : ℝ), a' > 0 → b' > 0 → f_deriv a' b' 1 = 0 → a' * b' ≤ max_ab) :=
sorry

end NUMINAMATH_CALUDE_max_ab_value_l119_11949


namespace NUMINAMATH_CALUDE_gills_arrival_time_l119_11966

/-- Represents the travel details of Gill's train journey --/
structure TravelDetails where
  departure_time : Nat  -- in minutes past midnight
  first_segment_distance : Nat  -- in km
  second_segment_distance : Nat  -- in km
  speed : Nat  -- in km/h
  stop_duration : Nat  -- in minutes

/-- Calculates the arrival time given the travel details --/
def calculate_arrival_time (details : TravelDetails) : Nat :=
  let first_segment_time := details.first_segment_distance * 60 / details.speed
  let second_segment_time := details.second_segment_distance * 60 / details.speed
  let total_travel_time := first_segment_time + details.stop_duration + second_segment_time
  details.departure_time + total_travel_time

/-- Gill's travel details --/
def gills_travel : TravelDetails :=
  { departure_time := 9 * 60  -- 09:00 in minutes
    first_segment_distance := 27
    second_segment_distance := 29
    speed := 96
    stop_duration := 3 }

theorem gills_arrival_time :
  calculate_arrival_time gills_travel = 9 * 60 + 38 := by
  sorry

end NUMINAMATH_CALUDE_gills_arrival_time_l119_11966


namespace NUMINAMATH_CALUDE_cross_number_puzzle_digit_l119_11914

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n < 1000

def power_of_2 (m : ℕ) : ℕ := 2^m
def power_of_3 (n : ℕ) : ℕ := 3^n

def same_digit_position (a b : ℕ) (pos : ℕ) : Prop :=
  (a / 10^pos) % 10 = (b / 10^pos) % 10

theorem cross_number_puzzle_digit :
  ∃! d : ℕ, d < 10 ∧
    ∃ (m n pos : ℕ),
      is_three_digit (power_of_2 m) ∧
      is_three_digit (power_of_3 n) ∧
      same_digit_position (power_of_2 m) (power_of_3 n) pos ∧
      (power_of_2 m / 10^pos) % 10 = d :=
by
  sorry

end NUMINAMATH_CALUDE_cross_number_puzzle_digit_l119_11914


namespace NUMINAMATH_CALUDE_range_of_a_for_fourth_quadrant_l119_11916

/-- A point in the fourth quadrant has a positive x-coordinate and a negative y-coordinate -/
def is_in_fourth_quadrant (x y : ℝ) : Prop := x > 0 ∧ y < 0

/-- The x-coordinate of point P -/
def x_coord (a : ℝ) : ℝ := a + 1

/-- The y-coordinate of point P -/
def y_coord (a : ℝ) : ℝ := 2 * a - 3

/-- The theorem stating the range of a for point P to be in the fourth quadrant -/
theorem range_of_a_for_fourth_quadrant :
  ∀ a : ℝ, is_in_fourth_quadrant (x_coord a) (y_coord a) ↔ -1 < a ∧ a < 3/2 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_for_fourth_quadrant_l119_11916


namespace NUMINAMATH_CALUDE_escalator_length_proof_l119_11947

/-- The length of an escalator in feet. -/
def escalator_length : ℝ := 160

/-- The speed of the escalator in feet per second. -/
def escalator_speed : ℝ := 8

/-- The walking speed of a person on the escalator in feet per second. -/
def person_speed : ℝ := 2

/-- The time taken by the person to cover the entire length of the escalator in seconds. -/
def time_taken : ℝ := 16

/-- Theorem stating that the length of the escalator is 160 feet, given the conditions. -/
theorem escalator_length_proof :
  escalator_length = (escalator_speed + person_speed) * time_taken :=
by sorry

end NUMINAMATH_CALUDE_escalator_length_proof_l119_11947


namespace NUMINAMATH_CALUDE_least_number_divisible_l119_11950

theorem least_number_divisible (n : ℕ) : n = 857 ↔ 
  (∀ m : ℕ, m < n → ¬(∃ k₁ k₂ k₃ k₄ : ℕ, 
    m + 7 = 24 * k₁ ∧ 
    m + 7 = 32 * k₂ ∧ 
    m + 7 = 36 * k₃ ∧ 
    m + 7 = 54 * k₄)) ∧ 
  (∃ k₁ k₂ k₃ k₄ : ℕ, 
    n + 7 = 24 * k₁ ∧ 
    n + 7 = 32 * k₂ ∧ 
    n + 7 = 36 * k₃ ∧ 
    n + 7 = 54 * k₄) :=
by sorry

end NUMINAMATH_CALUDE_least_number_divisible_l119_11950


namespace NUMINAMATH_CALUDE_arc_length_sixty_degrees_l119_11907

theorem arc_length_sixty_degrees (r : ℝ) (θ : ℝ) (l : ℝ) : 
  r = 1 → θ = 60 → l = (θ * π * r) / 180 → l = π / 3 := by
  sorry

end NUMINAMATH_CALUDE_arc_length_sixty_degrees_l119_11907


namespace NUMINAMATH_CALUDE_unique_root_quadratic_l119_11995

theorem unique_root_quadratic (p : ℝ) : 
  (∃ a b : ℝ, a ≠ 0 ∧ b ≠ 0 ∧ 
   (∀ x : ℝ, x^2 - 5*p*x + 2*p^3 = 0 ↔ (x = a ∨ x = b)) ∧
   (∃! x : ℝ, x^2 - a*x + b = 0)) →
  p = 3 := by
sorry


end NUMINAMATH_CALUDE_unique_root_quadratic_l119_11995


namespace NUMINAMATH_CALUDE_expenditure_ratio_l119_11988

-- Define the monthly incomes and savings
def income_b : ℚ := 7200
def income_ratio : ℚ := 5 / 6
def savings_a : ℚ := 1800
def savings_b : ℚ := 1600

-- Define the monthly incomes
def income_a : ℚ := income_ratio * income_b

-- Define the monthly expenditures
def expenditure_a : ℚ := income_a - savings_a
def expenditure_b : ℚ := income_b - savings_b

-- Theorem to prove
theorem expenditure_ratio :
  expenditure_a / expenditure_b = 3 / 4 :=
sorry

end NUMINAMATH_CALUDE_expenditure_ratio_l119_11988


namespace NUMINAMATH_CALUDE_circle_condition_l119_11933

/-- The equation x^2 + y^2 - 2x + 6y + m = 0 represents a circle if and only if m < 10 -/
theorem circle_condition (m : ℝ) : 
  (∃ (x y : ℝ), x^2 + y^2 - 2*x + 6*y + m = 0 ∧ 
   ∃ (h k r : ℝ), ∀ (x y : ℝ), (x - h)^2 + (y - k)^2 = r^2 ↔ x^2 + y^2 - 2*x + 6*y + m = 0) 
  ↔ m < 10 :=
sorry

end NUMINAMATH_CALUDE_circle_condition_l119_11933


namespace NUMINAMATH_CALUDE_ellipse_intersection_theorem_l119_11976

/-- Given a line y = x - 1 intersecting an ellipse (x^2 / a^2) + (y^2 / (a^2 - 1)) = 1 
    where a > 1, if the circle with diameter AB (where A and B are intersection points) 
    passes through the left focus of the ellipse, then a = (√6 + √2) / 2 -/
theorem ellipse_intersection_theorem (a : ℝ) (h_a : a > 1) :
  let line := fun x : ℝ => x - 1
  let ellipse := fun (x y : ℝ) => x^2 / a^2 + y^2 / (a^2 - 1) = 1
  let intersection_points := {p : ℝ × ℝ | ellipse p.1 p.2 ∧ p.2 = line p.1}
  let circle := fun (c : ℝ × ℝ) (r : ℝ) (p : ℝ × ℝ) => 
    (p.1 - c.1)^2 + (p.2 - c.2)^2 = r^2
  ∃ (A B : ℝ × ℝ) (c : ℝ × ℝ) (r : ℝ), 
    A ∈ intersection_points ∧ 
    B ∈ intersection_points ∧
    A ≠ B ∧
    circle c r A ∧
    circle c r B ∧
    circle c r (-1, 0) →
  a = (Real.sqrt 6 + Real.sqrt 2) / 2 := by
sorry

end NUMINAMATH_CALUDE_ellipse_intersection_theorem_l119_11976


namespace NUMINAMATH_CALUDE_remaining_money_l119_11929

def initial_amount : ℕ := 53
def toy_car_cost : ℕ := 11
def toy_car_quantity : ℕ := 2
def scarf_cost : ℕ := 10
def beanie_cost : ℕ := 14

theorem remaining_money :
  initial_amount - (toy_car_cost * toy_car_quantity + scarf_cost + beanie_cost) = 7 := by
  sorry

end NUMINAMATH_CALUDE_remaining_money_l119_11929


namespace NUMINAMATH_CALUDE_triangle_area_l119_11965

/-- The area of the triangle bounded by y = x, y = -x, and y = 8 is 64 -/
theorem triangle_area : Real := by
  -- Define the lines
  let line1 : Real → Real := λ x ↦ x
  let line2 : Real → Real := λ x ↦ -x
  let line3 : Real → Real := λ _ ↦ 8

  -- Define the intersection points
  let A : (Real × Real) := (8, 8)
  let B : (Real × Real) := (-8, 8)
  let O : (Real × Real) := (0, 0)

  -- Calculate the base and height of the triangle
  let base : Real := A.1 - B.1
  let height : Real := line3 0 - O.2

  -- Calculate the area
  let area : Real := (1 / 2) * base * height

  -- Prove that the area is 64
  sorry

end NUMINAMATH_CALUDE_triangle_area_l119_11965


namespace NUMINAMATH_CALUDE_road_trip_distance_l119_11904

/-- Represents Rick's road trip with 5 destinations -/
structure RoadTrip where
  leg1 : ℝ
  leg2 : ℝ
  leg3 : ℝ
  leg4 : ℝ
  leg5 : ℝ

/-- Conditions of Rick's road trip -/
def validRoadTrip (trip : RoadTrip) : Prop :=
  trip.leg2 = 2 * trip.leg1 ∧
  trip.leg3 = 40 ∧
  trip.leg3 = trip.leg1 / 2 ∧
  trip.leg4 = 2 * (trip.leg1 + trip.leg2 + trip.leg3) ∧
  trip.leg5 = 1.5 * trip.leg4

/-- The total distance of the road trip -/
def totalDistance (trip : RoadTrip) : ℝ :=
  trip.leg1 + trip.leg2 + trip.leg3 + trip.leg4 + trip.leg5

/-- Theorem stating that the total distance of a valid road trip is 1680 miles -/
theorem road_trip_distance (trip : RoadTrip) (h : validRoadTrip trip) :
  totalDistance trip = 1680 := by
  sorry

end NUMINAMATH_CALUDE_road_trip_distance_l119_11904


namespace NUMINAMATH_CALUDE_line_parametric_to_standard_l119_11900

/-- Given a line with parametric equations x = 1 + t and y = -1 + t,
    prove that its standard equation is x - y - 2 = 0 -/
theorem line_parametric_to_standard :
  ∀ (x y t : ℝ), x = 1 + t ∧ y = -1 + t → x - y - 2 = 0 := by
sorry

end NUMINAMATH_CALUDE_line_parametric_to_standard_l119_11900


namespace NUMINAMATH_CALUDE_inequality_proof_l119_11951

theorem inequality_proof (a b c d e f : ℝ) (h : b^2 ≥ a^2 + c^2) :
  (a*f - c*d)^2 ≤ (a*e - b*d)^2 + (b*f - c*e)^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l119_11951


namespace NUMINAMATH_CALUDE_maci_pen_cost_l119_11987

/-- The cost of Maci's pens given the number and prices of blue and red pens. -/
def cost_of_pens (blue_pens : ℕ) (red_pens : ℕ) (blue_pen_cost : ℚ) : ℚ :=
  let red_pen_cost := 2 * blue_pen_cost
  blue_pens * blue_pen_cost + red_pens * red_pen_cost

/-- Theorem stating that Maci pays $4.00 for her pens. -/
theorem maci_pen_cost : cost_of_pens 10 15 (10 / 100) = 4 := by
  sorry

#eval cost_of_pens 10 15 (10 / 100)

end NUMINAMATH_CALUDE_maci_pen_cost_l119_11987


namespace NUMINAMATH_CALUDE_profit_maximizing_price_l119_11991

/-- Represents the profit function for a product -/
def profit_function (x : ℝ) : ℝ :=
  (x - 8) * (100 - 10 * (x - 10))

/-- Theorem stating that the profit-maximizing price is 14 yuan -/
theorem profit_maximizing_price :
  ∃ (x : ℝ), x = 14 ∧ ∀ (y : ℝ), profit_function y ≤ profit_function x :=
sorry

end NUMINAMATH_CALUDE_profit_maximizing_price_l119_11991


namespace NUMINAMATH_CALUDE_parabola_unique_coefficients_l119_11905

/-- A parabola is defined by the equation y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The y-coordinate of a point on the parabola given its x-coordinate -/
def Parabola.y_at (p : Parabola) (x : ℝ) : ℝ :=
  p.a * x^2 + p.b * x + p.c

/-- The slope of the tangent line to the parabola at a given x-coordinate -/
def Parabola.slope_at (p : Parabola) (x : ℝ) : ℝ :=
  2 * p.a * x + p.b

/-- Theorem: For a parabola y = ax^2 + bx + c, if it passes through (1, 1),
    and the slope of the tangent line at (2, -1) is 1,
    then a = 3, b = -11, and c = 9 -/
theorem parabola_unique_coefficients (p : Parabola) 
    (h1 : p.y_at 1 = 1)
    (h2 : p.y_at 2 = -1)
    (h3 : p.slope_at 2 = 1) :
    p.a = 3 ∧ p.b = -11 ∧ p.c = 9 := by
  sorry

end NUMINAMATH_CALUDE_parabola_unique_coefficients_l119_11905


namespace NUMINAMATH_CALUDE_min_value_of_y_l119_11994

theorem min_value_of_y (a x : ℝ) (h1 : 0 < a) (h2 : a < 15) (h3 : a ≤ x) (h4 : x ≤ 15) :
  let y := |x - a| + |x - 15| + |x - (a + 15)|
  ∃ (min_y : ℝ), min_y = 15 ∧ ∀ z, a ≤ z ∧ z ≤ 15 → y ≤ |z - a| + |z - 15| + |z - (a + 15)| :=
by
  sorry

end NUMINAMATH_CALUDE_min_value_of_y_l119_11994


namespace NUMINAMATH_CALUDE_vector_properties_l119_11998

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

def opposite_direction (a b : V) : Prop :=
  ∃ (k : ℝ), k < 0 ∧ b = k • a

theorem vector_properties (a : V) (h : a ≠ 0) :
  opposite_direction a (-3 • a) ∧ a - 3 • a = -2 • a := by
  sorry

end NUMINAMATH_CALUDE_vector_properties_l119_11998


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l119_11974

def M : Set ℤ := {x | x < 3}
def N : Set ℤ := {x | 0 < x ∧ x < 6}

theorem intersection_of_M_and_N : M ∩ N = {1, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l119_11974


namespace NUMINAMATH_CALUDE_arthur_walk_distance_l119_11956

/-- The number of blocks Arthur walks east -/
def blocks_east : ℕ := 8

/-- The number of blocks Arthur walks north -/
def blocks_north : ℕ := 15

/-- The number of blocks Arthur walks west -/
def blocks_west : ℕ := 3

/-- The length of each block in miles -/
def block_length : ℚ := 1/2

/-- The total distance Arthur walks in miles -/
def total_distance : ℚ := (blocks_east + blocks_north + blocks_west : ℚ) * block_length

theorem arthur_walk_distance : total_distance = 13 := by
  sorry

end NUMINAMATH_CALUDE_arthur_walk_distance_l119_11956


namespace NUMINAMATH_CALUDE_ab_value_when_sqrt_and_abs_sum_zero_l119_11958

theorem ab_value_when_sqrt_and_abs_sum_zero (a b : ℝ) :
  Real.sqrt (a - 3) + |1 - b| = 0 → a * b = 3 := by
  sorry

end NUMINAMATH_CALUDE_ab_value_when_sqrt_and_abs_sum_zero_l119_11958


namespace NUMINAMATH_CALUDE_clothing_division_l119_11921

theorem clothing_division (total : ℕ) (first_load : ℕ) (h1 : total = 36) (h2 : first_load = 18) :
  (total - first_load) / 2 = 9 := by
sorry

end NUMINAMATH_CALUDE_clothing_division_l119_11921


namespace NUMINAMATH_CALUDE_count_numbers_with_2_between_200_and_499_l119_11901

def count_numbers_with_digit_2 (lower_bound upper_bound : ℕ) : ℕ :=
  sorry

theorem count_numbers_with_2_between_200_and_499 :
  count_numbers_with_digit_2 200 499 = 138 := by
  sorry

end NUMINAMATH_CALUDE_count_numbers_with_2_between_200_and_499_l119_11901


namespace NUMINAMATH_CALUDE_quick_calculation_formula_l119_11986

theorem quick_calculation_formula (a b : ℝ) :
  (100 + a) * (100 + b) = ((100 + a) + (100 + b) - 100) * 100 + a * b ∧
  (100 + a) * (100 - b) = ((100 + a) + (100 - b) - 100) * 100 + a * (-b) ∧
  (100 - a) * (100 + b) = ((100 - a) + (100 + b) - 100) * 100 + (-a) * b ∧
  (100 - a) * (100 - b) = ((100 - a) + (100 - b) - 100) * 100 + (-a) * (-b) :=
by sorry

end NUMINAMATH_CALUDE_quick_calculation_formula_l119_11986


namespace NUMINAMATH_CALUDE_cubic_equation_solution_l119_11990

theorem cubic_equation_solution (x y z n : ℕ+) :
  x^3 + y^3 + z^3 = n * x^2 * y^2 * z^2 ↔ n = 1 ∨ n = 3 := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_solution_l119_11990


namespace NUMINAMATH_CALUDE_positive_decreasing_function_l119_11942

/-- A function f: ℝ → ℝ is decreasing if for all x < y, f(x) > f(y) -/
def Decreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x > f y

/-- The second derivative of a function f: ℝ → ℝ -/
def SecondDerivative (f : ℝ → ℝ) : ℝ → ℝ := sorry

theorem positive_decreasing_function
  (f : ℝ → ℝ)
  (h_decreasing : Decreasing f)
  (h_second_derivative : ∀ x, f x / SecondDerivative f x < 1 - x) :
  ∀ x, f x > 0 := by
  sorry

end NUMINAMATH_CALUDE_positive_decreasing_function_l119_11942


namespace NUMINAMATH_CALUDE_all_pies_have_ingredients_l119_11978

theorem all_pies_have_ingredients (total_pies : ℕ) 
  (blueberry_fraction : ℚ) (strawberry_fraction : ℚ) 
  (raspberry_fraction : ℚ) (almond_fraction : ℚ) : 
  total_pies = 48 →
  blueberry_fraction = 1/3 →
  strawberry_fraction = 3/8 →
  raspberry_fraction = 1/2 →
  almond_fraction = 1/4 →
  ∃ (blueberry strawberry raspberry almond : Finset (Fin total_pies)),
    (blueberry.card : ℚ) ≥ blueberry_fraction * total_pies ∧
    (strawberry.card : ℚ) ≥ strawberry_fraction * total_pies ∧
    (raspberry.card : ℚ) ≥ raspberry_fraction * total_pies ∧
    (almond.card : ℚ) ≥ almond_fraction * total_pies ∧
    (blueberry ∪ strawberry ∪ raspberry ∪ almond).card = total_pies :=
by sorry

end NUMINAMATH_CALUDE_all_pies_have_ingredients_l119_11978


namespace NUMINAMATH_CALUDE_polynomial_evaluation_l119_11982

theorem polynomial_evaluation (x : ℝ) (h1 : x > 0) (h2 : x^2 - 3*x - 9 = 0) :
  x^3 - 3*x^2 - 9*x + 5 = 5 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_evaluation_l119_11982


namespace NUMINAMATH_CALUDE_moon_speed_km_per_second_l119_11922

-- Define the speed of the moon in kilometers per hour
def moon_speed_km_per_hour : ℝ := 3672

-- Define the number of seconds in an hour
def seconds_per_hour : ℝ := 3600

-- Theorem statement
theorem moon_speed_km_per_second :
  moon_speed_km_per_hour / seconds_per_hour = 1.02 := by
  sorry

end NUMINAMATH_CALUDE_moon_speed_km_per_second_l119_11922


namespace NUMINAMATH_CALUDE_geometric_progression_sum_relation_l119_11937

/-- Given two infinitely decreasing geometric progressions with first term 1
    and common ratios a^p and a^q respectively, where 0 < a < 1 and p, q are
    positive real numbers, prove that the sums S and S₁ of these progressions
    satisfy the equation: S^q * (S₁ - 1)^p = S₁^p * (S - 1)^q -/
theorem geometric_progression_sum_relation
  (a p q : ℝ)
  (ha : 0 < a ∧ a < 1)
  (hp : 0 < p)
  (hq : 0 < q)
  (S : ℝ)
  (hS : S = (1 - a^p)⁻¹)
  (S₁ : ℝ)
  (hS₁ : S₁ = (1 - a^q)⁻¹) :
  S^q * (S₁ - 1)^p = S₁^p * (S - 1)^q := by
sorry

end NUMINAMATH_CALUDE_geometric_progression_sum_relation_l119_11937


namespace NUMINAMATH_CALUDE_unique_perfect_between_primes_l119_11934

/-- A number is perfect if the sum of its positive divisors equals twice the number. -/
def IsPerfect (n : ℕ) : Prop :=
  (Finset.filter (· ∣ n) (Finset.range (n + 1))).sum id = 2 * n

/-- The theorem stating that 6 is the only perfect number n such that n-1 and n+1 are prime. -/
theorem unique_perfect_between_primes :
  ∀ n : ℕ, IsPerfect n ∧ Nat.Prime (n - 1) ∧ Nat.Prime (n + 1) → n = 6 :=
by sorry

end NUMINAMATH_CALUDE_unique_perfect_between_primes_l119_11934


namespace NUMINAMATH_CALUDE_pipe_B_fill_time_l119_11964

/-- Time for pipe A to fill the tank -/
def time_A : ℝ := 5

/-- Time for the tank to drain -/
def time_drain : ℝ := 20

/-- Time to fill the tank with both pipes on and drainage open -/
def time_combined : ℝ := 3.6363636363636362

/-- Time for pipe B to fill the tank -/
def time_B : ℝ := 1.0526315789473684

/-- Theorem stating the relationship between the given times -/
theorem pipe_B_fill_time :
  time_B = (time_A * time_drain * time_combined) / 
           (time_A * time_drain - time_A * time_combined - time_drain * time_combined) :=
by sorry

end NUMINAMATH_CALUDE_pipe_B_fill_time_l119_11964


namespace NUMINAMATH_CALUDE_collinear_points_problem_l119_11932

/-- Given three collinear points A, B, C in a plane with position vectors
    OA = (-2, m), OB = (n, 1), OC = (5, -1), and OA perpendicular to OB,
    prove that m = 6 and n = 3. -/
theorem collinear_points_problem (m n : ℝ) : 
  let OA : ℝ × ℝ := (-2, m)
  let OB : ℝ × ℝ := (n, 1)
  let OC : ℝ × ℝ := (5, -1)
  let AC : ℝ × ℝ := (OC.1 - OA.1, OC.2 - OA.2)
  let BC : ℝ × ℝ := (OC.1 - OB.1, OC.2 - OB.2)
  (∃ (k : ℝ), AC = k • BC) →  -- collinearity condition
  (OA.1 * OB.1 + OA.2 * OB.2 = 0) →  -- perpendicularity condition
  m = 6 ∧ n = 3 := by
  sorry

end NUMINAMATH_CALUDE_collinear_points_problem_l119_11932


namespace NUMINAMATH_CALUDE_company_picnic_attendance_l119_11961

theorem company_picnic_attendance
  (total_employees : ℕ)
  (men_percentage : ℚ)
  (women_percentage : ℚ)
  (men_attendance_rate : ℚ)
  (women_attendance_rate : ℚ)
  (h1 : men_percentage = 1/2)
  (h2 : women_percentage = 1 - men_percentage)
  (h3 : men_attendance_rate = 1/5)
  (h4 : women_attendance_rate = 2/5) :
  (men_percentage * men_attendance_rate + women_percentage * women_attendance_rate) = 3/10 := by
  sorry

end NUMINAMATH_CALUDE_company_picnic_attendance_l119_11961


namespace NUMINAMATH_CALUDE_middle_number_problem_l119_11928

theorem middle_number_problem (x y z : ℕ) 
  (h1 : x < y) (h2 : y < z)
  (h3 : x + y = 20) (h4 : x + z = 25) (h5 : y + z = 29) (h6 : z - x = 11) :
  y = 13 := by
  sorry

end NUMINAMATH_CALUDE_middle_number_problem_l119_11928


namespace NUMINAMATH_CALUDE_bobbit_worm_consumption_l119_11980

/-- Represents the number of fish eaten by the Bobbit worm each day -/
def fish_eaten_per_day : ℕ := 2

/-- The initial number of fish in the aquarium -/
def initial_fish : ℕ := 60

/-- The number of fish added after two weeks -/
def fish_added : ℕ := 8

/-- The number of fish remaining after three weeks -/
def remaining_fish : ℕ := 26

/-- The total number of days -/
def total_days : ℕ := 21

theorem bobbit_worm_consumption :
  initial_fish + fish_added - (fish_eaten_per_day * total_days) = remaining_fish := by
  sorry

end NUMINAMATH_CALUDE_bobbit_worm_consumption_l119_11980


namespace NUMINAMATH_CALUDE_exists_arithmetic_right_triangle_with_81_l119_11977

/-- A right triangle with integer side lengths forming an arithmetic sequence -/
structure ArithmeticRightTriangle where
  a : ℕ
  d : ℕ
  right_triangle : a^2 + (a + d)^2 = (a + 2*d)^2
  arithmetic_sequence : True

/-- The existence of an arithmetic right triangle with one side length equal to 81 -/
theorem exists_arithmetic_right_triangle_with_81 :
  ∃ (t : ArithmeticRightTriangle), t.a = 81 ∨ t.a + t.d = 81 ∨ t.a + 2*t.d = 81 := by
  sorry

end NUMINAMATH_CALUDE_exists_arithmetic_right_triangle_with_81_l119_11977


namespace NUMINAMATH_CALUDE_james_money_theorem_l119_11989

/-- The amount of money James has now, given the conditions -/
def jamesTotal (billsFound : ℕ) (billValue : ℕ) (initialWallet : ℕ) : ℕ :=
  billsFound * billValue + initialWallet

/-- Theorem stating that James has $135 given the problem conditions -/
theorem james_money_theorem :
  jamesTotal 3 20 75 = 135 := by
  sorry

end NUMINAMATH_CALUDE_james_money_theorem_l119_11989


namespace NUMINAMATH_CALUDE_product_digits_sum_l119_11973

/-- Converts a base-7 number to base-10 --/
def toBase10 (n : ℕ) : ℕ := sorry

/-- Converts a base-10 number to base-7 --/
def toBase7 (n : ℕ) : ℕ := sorry

/-- Computes the sum of digits of a number in base-7 --/
def sumOfDigitsBase7 (n : ℕ) : ℕ := sorry

/-- Multiplies two base-7 numbers --/
def multiplyBase7 (a b : ℕ) : ℕ := 
  toBase7 (toBase10 a * toBase10 b)

theorem product_digits_sum :
  sumOfDigitsBase7 (multiplyBase7 24 30) = 6 := by sorry

end NUMINAMATH_CALUDE_product_digits_sum_l119_11973


namespace NUMINAMATH_CALUDE_donation_in_scientific_notation_l119_11953

/-- Definition of a billion in the context of this problem -/
def billion : ℕ := 10^8

/-- The donation amount in yuan -/
def donation : ℚ := 2.94 * billion

/-- Scientific notation representation of a number -/
structure ScientificNotation where
  coefficient : ℚ
  exponent : ℤ
  property : 1 ≤ coefficient ∧ coefficient < 10

/-- Theorem stating that 2.94 billion yuan is equal to 2.94 × 10^8 in scientific notation -/
theorem donation_in_scientific_notation :
  ∃ (sn : ScientificNotation), (sn.coefficient * (10 : ℚ)^sn.exponent) = donation ∧
    sn.coefficient = 2.94 ∧ sn.exponent = 8 := by sorry

end NUMINAMATH_CALUDE_donation_in_scientific_notation_l119_11953


namespace NUMINAMATH_CALUDE_sqrt_three_sum_product_l119_11952

theorem sqrt_three_sum_product : Real.sqrt 3 * (Real.sqrt 3 + Real.sqrt 27) = 12 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_three_sum_product_l119_11952


namespace NUMINAMATH_CALUDE_rectangle_width_decrease_l119_11943

theorem rectangle_width_decrease (L W : ℝ) (h1 : L > 0) (h2 : W > 0) :
  let new_length := 1.4 * L
  let new_width := W / 1.4
  (new_length * new_width = L * W) ∧
  (2 * new_length + 2 * new_width = 2 * L + 2 * W) →
  (W - new_width) / W = 2 / 7 := by
sorry

end NUMINAMATH_CALUDE_rectangle_width_decrease_l119_11943


namespace NUMINAMATH_CALUDE_sum_squares_theorem_l119_11997

theorem sum_squares_theorem (x y z : ℤ) 
  (sum_eq : x + y + z = 3) 
  (sum_cubes_eq : x^3 + y^3 + z^3 = 3) : 
  x^2 + y^2 + z^2 = 3 ∨ x^2 + y^2 + z^2 = 57 := by
  sorry

end NUMINAMATH_CALUDE_sum_squares_theorem_l119_11997


namespace NUMINAMATH_CALUDE_tangent_slope_implies_a_l119_11955

-- Define the curve
def f (a x : ℝ) : ℝ := x^4 + a*x^2 + 1

-- Define the derivative of the curve
def f' (a x : ℝ) : ℝ := 4*x^3 + 2*a*x

theorem tangent_slope_implies_a (a : ℝ) :
  f a (-1) = a + 2 → f' a (-1) = 8 → a = -6 := by
  sorry

end NUMINAMATH_CALUDE_tangent_slope_implies_a_l119_11955


namespace NUMINAMATH_CALUDE_fraction_decomposition_l119_11936

theorem fraction_decomposition (x : ℝ) (h1 : x ≠ 0) (h2 : x^2 ≠ -1) :
  (-x^3 + 4*x^2 - 5*x + 3) / (x^4 + x^2) = 3/x^2 + (4*x + 1)/(x^2 + 1) - 5/x :=
by sorry

end NUMINAMATH_CALUDE_fraction_decomposition_l119_11936


namespace NUMINAMATH_CALUDE_sphere_surface_area_l119_11996

theorem sphere_surface_area (r : ℝ) (h : r = 3) : 4 * Real.pi * r^2 = 36 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_sphere_surface_area_l119_11996


namespace NUMINAMATH_CALUDE_kristin_reading_time_l119_11917

/-- Given that Peter reads one book in 18 hours and can read three times as fast as Kristin,
    prove that Kristin will take 540 hours to read half of her 20 books. -/
theorem kristin_reading_time 
  (peter_time : ℝ) 
  (peter_speed : ℝ) 
  (kristin_books : ℕ) 
  (h1 : peter_time = 18) 
  (h2 : peter_speed = 3) 
  (h3 : kristin_books = 20) : 
  (kristin_books / 2 : ℝ) * (peter_time * peter_speed) = 540 := by
  sorry

end NUMINAMATH_CALUDE_kristin_reading_time_l119_11917


namespace NUMINAMATH_CALUDE_functional_equation_solution_functional_equation_continuous_solution_l119_11979

def functional_equation (f : ℝ → ℝ) (a : ℝ) : Prop :=
  f 0 = 0 ∧ f 1 = 1 ∧ ∀ x y, x ≤ y → f ((x + y) / 2) = (1 - a) * f x + a * f y

theorem functional_equation_solution (a : ℝ) :
  (∃ f : ℝ → ℝ, functional_equation f a) ↔ (a = 0 ∨ a = 1/2 ∨ a = 1) :=
sorry

theorem functional_equation_continuous_solution (a : ℝ) :
  (∃ f : ℝ → ℝ, Continuous f ∧ functional_equation f a) ↔ a = 1/2 :=
sorry

end NUMINAMATH_CALUDE_functional_equation_solution_functional_equation_continuous_solution_l119_11979


namespace NUMINAMATH_CALUDE_sugar_package_weight_l119_11941

theorem sugar_package_weight (x : ℝ) 
  (h1 : x > 0)
  (h2 : (4 * x - 10) / (x + 10) = 7 / 8) :
  4 * x + x = 30 := by
  sorry

end NUMINAMATH_CALUDE_sugar_package_weight_l119_11941


namespace NUMINAMATH_CALUDE_event_occurrence_limit_l119_11954

theorem event_occurrence_limit (ε : ℝ) (hε : 0 < ε) :
  ∀ δ > 0, ∃ N : ℕ, ∀ n ≥ N, 1 - (1 - ε)^n > 1 - δ :=
sorry

end NUMINAMATH_CALUDE_event_occurrence_limit_l119_11954


namespace NUMINAMATH_CALUDE_path_area_and_cost_l119_11960

/-- Calculates the area of a rectangular path around a field -/
def path_area (field_length field_width path_width : ℝ) : ℝ :=
  (field_length + 2 * path_width) * (field_width + 2 * path_width) - field_length * field_width

/-- Calculates the cost of constructing a path given its area and cost per unit area -/
def construction_cost (path_area cost_per_unit : ℝ) : ℝ :=
  path_area * cost_per_unit

theorem path_area_and_cost (field_length field_width path_width cost_per_unit : ℝ) 
  (h1 : field_length = 85)
  (h2 : field_width = 55)
  (h3 : path_width = 2.5)
  (h4 : cost_per_unit = 2) : 
  path_area field_length field_width path_width = 725 ∧ 
  construction_cost (path_area field_length field_width path_width) cost_per_unit = 1450 := by
  sorry

#eval path_area 85 55 2.5
#eval construction_cost (path_area 85 55 2.5) 2

end NUMINAMATH_CALUDE_path_area_and_cost_l119_11960


namespace NUMINAMATH_CALUDE_withdraw_300_from_two_banks_in_20_bills_l119_11930

/-- Calculates the number of bills received when withdrawing from two banks -/
def number_of_bills (withdrawal_per_bank : ℕ) (bill_denomination : ℕ) : ℕ :=
  (2 * withdrawal_per_bank) / bill_denomination

/-- Theorem: Withdrawing $300 from each of two banks in $20 bills results in 30 bills -/
theorem withdraw_300_from_two_banks_in_20_bills : 
  number_of_bills 300 20 = 30 := by
  sorry

end NUMINAMATH_CALUDE_withdraw_300_from_two_banks_in_20_bills_l119_11930


namespace NUMINAMATH_CALUDE_expand_and_simplify_l119_11968

theorem expand_and_simplify (x : ℝ) : (x + 6) * (x - 11) = x^2 - 5*x - 66 := by
  sorry

end NUMINAMATH_CALUDE_expand_and_simplify_l119_11968
