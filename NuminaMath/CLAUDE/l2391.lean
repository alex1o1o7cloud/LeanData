import Mathlib

namespace NUMINAMATH_CALUDE_dad_steps_l2391_239180

theorem dad_steps (dad_masha_ratio : ℕ → ℕ → Prop)
                  (masha_yasha_ratio : ℕ → ℕ → Prop)
                  (masha_yasha_total : ℕ) :
  dad_masha_ratio 3 5 →
  masha_yasha_ratio 3 5 →
  masha_yasha_total = 400 →
  ∃ (dad_steps : ℕ), dad_steps = 90 :=
by sorry

end NUMINAMATH_CALUDE_dad_steps_l2391_239180


namespace NUMINAMATH_CALUDE_emma_age_l2391_239156

/-- Represents the ages of the individuals --/
structure Ages where
  oliver : ℕ
  nancy : ℕ
  liam : ℕ
  emma : ℕ

/-- The age relationships between Oliver, Nancy, Liam, and Emma --/
def age_relationships (ages : Ages) : Prop :=
  ages.oliver + 5 = ages.nancy ∧
  ages.nancy = ages.liam + 6 ∧
  ages.emma = ages.liam + 4 ∧
  ages.oliver = 16

/-- Theorem stating that given the age relationships and Oliver's age, Emma is 19 years old --/
theorem emma_age (ages : Ages) : age_relationships ages → ages.emma = 19 := by
  sorry

end NUMINAMATH_CALUDE_emma_age_l2391_239156


namespace NUMINAMATH_CALUDE_fourth_task_end_time_l2391_239197

-- Define the start time of the first task
def start_time : Nat := 9 * 60  -- 9:00 AM in minutes since midnight

-- Define the end time of the third task
def end_third_task : Nat := 11 * 60 + 30  -- 11:30 AM in minutes since midnight

-- Define the number of tasks
def num_tasks : Nat := 4

-- Define the theorem
theorem fourth_task_end_time :
  let total_time := end_third_task - start_time
  let task_duration := total_time / 3
  let fourth_task_end := end_third_task + task_duration
  fourth_task_end = 12 * 60 + 20  -- 12:20 PM in minutes since midnight
  := by sorry

end NUMINAMATH_CALUDE_fourth_task_end_time_l2391_239197


namespace NUMINAMATH_CALUDE_perpendicular_slope_l2391_239161

/-- Given a line with equation 5x - 2y = 10, 
    the slope of the perpendicular line is -2/5 -/
theorem perpendicular_slope (x y : ℝ) :
  (5 * x - 2 * y = 10) → 
  (∃ m : ℝ, m = -2/5 ∧ 
    ∀ (x₁ y₁ x₂ y₂ : ℝ), 
      (5 * x₁ - 2 * y₁ = 10) → 
      (5 * x₂ - 2 * y₂ = 10) → 
      x₁ ≠ x₂ → 
      m * ((x₂ - x₁) / (y₂ - y₁)) = -1) :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_slope_l2391_239161


namespace NUMINAMATH_CALUDE_school_ratio_l2391_239190

-- Define the school structure
structure School where
  b : ℕ  -- number of teachers
  c : ℕ  -- number of students
  k : ℕ  -- number of students each teacher teaches
  h : ℕ  -- number of teachers teaching any two different students

-- Define the theorem
theorem school_ratio (s : School) : 
  s.b / s.h = (s.c * (s.c - 1)) / (s.k * (s.k - 1)) := by
  sorry

end NUMINAMATH_CALUDE_school_ratio_l2391_239190


namespace NUMINAMATH_CALUDE_square_circle_perimeter_l2391_239150

/-- Given a square with perimeter 28 cm, the perimeter of a circle whose radius is equal to the side of the square is 14π cm. -/
theorem square_circle_perimeter (square_perimeter : ℝ) (h : square_perimeter = 28) :
  let square_side := square_perimeter / 4
  let circle_radius := square_side
  let circle_perimeter := 2 * Real.pi * circle_radius
  circle_perimeter = 14 * Real.pi :=
by
  sorry

end NUMINAMATH_CALUDE_square_circle_perimeter_l2391_239150


namespace NUMINAMATH_CALUDE_triangle_formation_constraint_l2391_239112

/-- A line in 2D space represented by ax + by = c -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if three lines form a triangle -/
def form_triangle (l1 l2 l3 : Line) : Prop :=
  ∃ (x1 y1 x2 y2 x3 y3 : ℝ),
    (l1.a * x1 + l1.b * y1 = l1.c) ∧
    (l2.a * x2 + l2.b * y2 = l2.c) ∧
    (l3.a * x3 + l3.b * y3 = l3.c) ∧
    ((x1 ≠ x2) ∨ (y1 ≠ y2)) ∧
    ((x2 ≠ x3) ∨ (y2 ≠ y3)) ∧
    ((x3 ≠ x1) ∨ (y3 ≠ y1))

theorem triangle_formation_constraint (a : ℝ) :
  let l1 : Line := ⟨1, 1, 0⟩
  let l2 : Line := ⟨1, -1, 0⟩
  let l3 : Line := ⟨1, a, 3⟩
  form_triangle l1 l2 l3 → a ≠ 1 ∧ a ≠ -1 :=
by sorry

end NUMINAMATH_CALUDE_triangle_formation_constraint_l2391_239112


namespace NUMINAMATH_CALUDE_max_product_equals_sum_l2391_239130

theorem max_product_equals_sum (H M T : ℤ) : 
  H * M * M * T = H + M + M + T → H * M * M * T ≤ 8 := by
  sorry

end NUMINAMATH_CALUDE_max_product_equals_sum_l2391_239130


namespace NUMINAMATH_CALUDE_tim_change_l2391_239187

/-- The change Tim received after buying a candy bar -/
def change (initial_amount : ℕ) (candy_cost : ℕ) : ℕ :=
  initial_amount - candy_cost

/-- Theorem stating that Tim's change is 5 cents -/
theorem tim_change :
  change 50 45 = 5 := by
  sorry

end NUMINAMATH_CALUDE_tim_change_l2391_239187


namespace NUMINAMATH_CALUDE_A_not_perfect_square_l2391_239110

/-- A number formed by 600 times the digit 6 followed by any number of zeros -/
def A (n : ℕ) : ℕ := 666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666 * (10^n)

/-- The 2-adic valuation of a natural number -/
def two_adic_valuation (m : ℕ) : ℕ :=
  if m = 0 then 0 else (m.factors.filter (· = 2)).length

/-- Theorem: A is not a perfect square for any number of trailing zeros -/
theorem A_not_perfect_square (n : ℕ) : ¬ ∃ (k : ℕ), A n = k^2 := by
  sorry

end NUMINAMATH_CALUDE_A_not_perfect_square_l2391_239110


namespace NUMINAMATH_CALUDE_fourth_segment_length_l2391_239111

/-- Represents an acute triangle with two altitudes dividing opposite sides -/
structure AcuteTriangleWithAltitudes where
  -- Lengths of segments created by altitudes
  segment1 : ℝ
  segment2 : ℝ
  segment3 : ℝ
  segment4 : ℝ
  -- Conditions
  acute : segment1 > 0 ∧ segment2 > 0 ∧ segment3 > 0 ∧ segment4 > 0
  segment1_eq : segment1 = 4
  segment2_eq : segment2 = 6
  segment3_eq : segment3 = 3

/-- Theorem stating that the fourth segment length is 3 -/
theorem fourth_segment_length (t : AcuteTriangleWithAltitudes) : t.segment4 = 3 := by
  sorry

end NUMINAMATH_CALUDE_fourth_segment_length_l2391_239111


namespace NUMINAMATH_CALUDE_x_fifth_minus_ten_x_equals_213_l2391_239119

theorem x_fifth_minus_ten_x_equals_213 (x : ℝ) (h : x = 3) : x^5 - 10*x = 213 := by
  sorry

end NUMINAMATH_CALUDE_x_fifth_minus_ten_x_equals_213_l2391_239119


namespace NUMINAMATH_CALUDE_cubic_decomposition_sum_l2391_239137

theorem cubic_decomposition_sum :
  ∃ (a b c d e : ℝ),
    (∀ x : ℝ, 512 * x^3 + 27 = (a * x + b) * (c * x^2 + d * x + e)) ∧
    (a + b + c + d + e = 60) := by
  sorry

end NUMINAMATH_CALUDE_cubic_decomposition_sum_l2391_239137


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2391_239122

-- Define an arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- State the theorem
theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  is_arithmetic_sequence a →
  (a 4 + (1/2) * a 7 + a 10 = 10) →
  (a 3 + a 11 = 8) :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2391_239122


namespace NUMINAMATH_CALUDE_twentieth_term_of_sequence_l2391_239189

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1 : ℝ) * d

theorem twentieth_term_of_sequence (a₁ a₁₃ a₂₀ : ℝ) :
  a₁ = 3 →
  a₁₃ = 27 →
  (∃ d : ℝ, ∀ n : ℕ, arithmetic_sequence a₁ d n = a₁ + (n - 1 : ℝ) * d) →
  a₂₀ = arithmetic_sequence a₁ ((a₁₃ - a₁) / 12) 20 →
  a₂₀ = 41 := by
sorry

end NUMINAMATH_CALUDE_twentieth_term_of_sequence_l2391_239189


namespace NUMINAMATH_CALUDE_angle_half_in_fourth_quadrant_l2391_239182

/-- Represents the four quadrants of the coordinate plane. -/
inductive Quadrant
  | first
  | second
  | third
  | fourth

/-- Determines if an angle is in a specific quadrant. -/
def in_quadrant (angle : ℝ) (q : Quadrant) : Prop :=
  match q with
  | Quadrant.first => 0 < angle ∧ angle < Real.pi / 2
  | Quadrant.second => Real.pi / 2 < angle ∧ angle < Real.pi
  | Quadrant.third => Real.pi < angle ∧ angle < 3 * Real.pi / 2
  | Quadrant.fourth => 3 * Real.pi / 2 < angle ∧ angle < 2 * Real.pi

theorem angle_half_in_fourth_quadrant (α : ℝ) 
  (h1 : in_quadrant α Quadrant.third) 
  (h2 : |Real.sin (α/2)| = -Real.sin (α/2)) : 
  in_quadrant (α/2) Quadrant.fourth :=
sorry

end NUMINAMATH_CALUDE_angle_half_in_fourth_quadrant_l2391_239182


namespace NUMINAMATH_CALUDE_tan_alpha_value_l2391_239124

theorem tan_alpha_value (α : ℝ) (h : Real.tan (α - 5 * Real.pi / 4) = 1 / 5) :
  Real.tan α = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_value_l2391_239124


namespace NUMINAMATH_CALUDE_square_of_cube_of_fourth_smallest_prime_l2391_239107

def fourth_smallest_prime : ℕ := 7

theorem square_of_cube_of_fourth_smallest_prime :
  (fourth_smallest_prime ^ 3) ^ 2 = 117649 := by
  sorry

end NUMINAMATH_CALUDE_square_of_cube_of_fourth_smallest_prime_l2391_239107


namespace NUMINAMATH_CALUDE_sum_of_triangle_and_rectangle_edges_l2391_239104

/-- The number of edges in a triangle -/
def triangle_edges : ℕ := 3

/-- The number of edges in a rectangle -/
def rectangle_edges : ℕ := 4

/-- The sum of edges in a triangle and a rectangle -/
def total_edges : ℕ := triangle_edges + rectangle_edges

theorem sum_of_triangle_and_rectangle_edges :
  total_edges = 7 := by sorry

end NUMINAMATH_CALUDE_sum_of_triangle_and_rectangle_edges_l2391_239104


namespace NUMINAMATH_CALUDE_one_thirds_in_eight_halves_l2391_239134

theorem one_thirds_in_eight_halves : (8 / 2) / (1 / 3) = 12 := by
  sorry

end NUMINAMATH_CALUDE_one_thirds_in_eight_halves_l2391_239134


namespace NUMINAMATH_CALUDE_unique_four_digit_number_l2391_239188

theorem unique_four_digit_number : ∃! n : ℕ, 
  1000 ≤ n ∧ n < 10000 ∧ 
  n % 131 = 112 ∧ 
  n % 132 = 98 :=
by sorry

end NUMINAMATH_CALUDE_unique_four_digit_number_l2391_239188


namespace NUMINAMATH_CALUDE_solve_for_a_l2391_239171

theorem solve_for_a : ∀ a : ℝ, (3 * 2 - a = -2 + 7) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_a_l2391_239171


namespace NUMINAMATH_CALUDE_expression_equality_l2391_239172

theorem expression_equality (θ : Real) 
  (h1 : π / 4 < θ) (h2 : θ < π / 2) : 
  2 * Real.cos θ + Real.sqrt (1 - 2 * Real.sin (π - θ) * Real.cos θ) = Real.sin θ + Real.cos θ := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l2391_239172


namespace NUMINAMATH_CALUDE_complement_of_intersection_equals_universe_l2391_239128

-- Define the sets
def U : Set ℝ := Set.univ
def A : Set ℝ := {x | x ≥ 2}
def B : Set ℝ := {x | x < -1}

-- State the theorem
theorem complement_of_intersection_equals_universe :
  (A ∩ B)ᶜ = U := by sorry

end NUMINAMATH_CALUDE_complement_of_intersection_equals_universe_l2391_239128


namespace NUMINAMATH_CALUDE_intersection_of_three_lines_l2391_239183

/-- Given three lines that intersect at the same point, prove the value of k -/
theorem intersection_of_three_lines (x y k : ℚ) :
  (y = 4 * x - 1) ∧
  (y = -3 * x + 9) ∧
  (y = 2 * x + k) →
  k = 13 / 7 := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_three_lines_l2391_239183


namespace NUMINAMATH_CALUDE_fraction_to_zero_power_l2391_239121

theorem fraction_to_zero_power : 
  (17381294 : ℚ) / (-43945723904 : ℚ) ^ (0 : ℕ) = 1 := by sorry

end NUMINAMATH_CALUDE_fraction_to_zero_power_l2391_239121


namespace NUMINAMATH_CALUDE_sweep_time_is_three_l2391_239141

/-- The time in minutes it takes to sweep one room -/
def sweep_time : ℝ := sorry

/-- The time in minutes it takes to wash one dish -/
def dish_time : ℝ := 2

/-- The time in minutes it takes to do one load of laundry -/
def laundry_time : ℝ := 9

/-- The number of rooms Anna sweeps -/
def anna_rooms : ℕ := 10

/-- The number of loads of laundry Billy does -/
def billy_laundry : ℕ := 2

/-- The number of dishes Billy washes -/
def billy_dishes : ℕ := 6

theorem sweep_time_is_three :
  sweep_time = 3 ∧
  anna_rooms * sweep_time = billy_laundry * laundry_time + billy_dishes * dish_time :=
by sorry

end NUMINAMATH_CALUDE_sweep_time_is_three_l2391_239141


namespace NUMINAMATH_CALUDE_line_plane_relations_l2391_239173

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations between lines and planes
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Plane → Prop)
variable (perpendicular_lines : Line → Line → Prop)
variable (parallel_lines : Line → Line → Prop)

-- State the theorem
theorem line_plane_relations 
  (l m : Line) (α : Plane) 
  (h : perpendicular l α) : 
  (perpendicular m α → parallel_lines m l) ∧ 
  (parallel m α → perpendicular_lines m l) ∧ 
  (parallel_lines m l → perpendicular m α) := by
  sorry

end NUMINAMATH_CALUDE_line_plane_relations_l2391_239173


namespace NUMINAMATH_CALUDE_f_is_odd_l2391_239145

noncomputable def f (x : ℝ) : ℝ := 2 * x - 1 / x

theorem f_is_odd :
  ∀ x ∈ {x : ℝ | x < 0 ∨ x > 0}, f (-x) = -f x :=
by
  sorry

end NUMINAMATH_CALUDE_f_is_odd_l2391_239145


namespace NUMINAMATH_CALUDE_actual_distance_towns_distance_proof_l2391_239147

/-- Calculates the actual distance between two towns given the map distance and scale. -/
theorem actual_distance (map_distance : ℝ) (scale_distance : ℝ) (scale_miles : ℝ) : ℝ :=
  let miles_per_inch := scale_miles / scale_distance
  map_distance * miles_per_inch

/-- Proves that the actual distance between two towns is 400 miles given the specified conditions. -/
theorem towns_distance_proof :
  actual_distance 20 0.5 10 = 400 := by
  sorry

end NUMINAMATH_CALUDE_actual_distance_towns_distance_proof_l2391_239147


namespace NUMINAMATH_CALUDE_triangle_problem_l2391_239196

theorem triangle_problem (a b c A B C : ℝ) (h1 : 0 < A) (h2 : A < π) : 
  c = a * Real.sin C - c * Real.cos A →
  (A = π / 2) ∧ 
  (a = 2 → 1/2 * b * c * Real.sin A = 2 → b = 2 ∧ c = 2) :=
by sorry

end NUMINAMATH_CALUDE_triangle_problem_l2391_239196


namespace NUMINAMATH_CALUDE_parabola_properties_l2391_239138

/-- Given a parabola with equation x² = 8y, this theorem proves the equation of its directrix
    and the coordinates of its focus. -/
theorem parabola_properties (x y : ℝ) :
  x^2 = 8*y →
  (∃ (directrix : ℝ → Prop) (focus : ℝ × ℝ),
    directrix = λ y' => y' = -2 ∧
    focus = (0, 2)) :=
by sorry

end NUMINAMATH_CALUDE_parabola_properties_l2391_239138


namespace NUMINAMATH_CALUDE_garden_feet_count_l2391_239179

/-- The number of feet for a dog -/
def dog_feet : ℕ := 4

/-- The number of feet for a duck -/
def duck_feet : ℕ := 2

/-- The number of dogs in the garden -/
def num_dogs : ℕ := 6

/-- The number of ducks in the garden -/
def num_ducks : ℕ := 2

/-- The total number of feet in the garden -/
def total_feet : ℕ := num_dogs * dog_feet + num_ducks * duck_feet

theorem garden_feet_count : total_feet = 28 := by
  sorry

end NUMINAMATH_CALUDE_garden_feet_count_l2391_239179


namespace NUMINAMATH_CALUDE_min_value_of_M_l2391_239193

theorem min_value_of_M (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hsum : a + b + c = 1) :
  let M := (1/a - 1) * (1/b - 1) * (1/c - 1)
  M ≥ 8 ∧ ∃ (a₀ b₀ c₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ c₀ > 0 ∧ a₀ + b₀ + c₀ = 1 ∧
    (1/a₀ - 1) * (1/b₀ - 1) * (1/c₀ - 1) = 8 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_M_l2391_239193


namespace NUMINAMATH_CALUDE_horner_method_operations_l2391_239101

/-- Horner's method for polynomial evaluation -/
def horner_eval (coeffs : List ℝ) (x : ℝ) : ℝ :=
  coeffs.foldl (fun acc a => acc * x + a) 0

/-- Count operations in Horner's method -/
def horner_count (coeffs : List ℝ) : Nat × Nat :=
  (coeffs.length - 1, coeffs.length - 1)

/-- The polynomial f(x) = 6x^6 + 4x^5 - 2x^4 + 5x^3 - 7x^2 - 2x + 5 -/
def f : List ℝ := [6, 4, -2, 5, -7, -2, 5]

theorem horner_method_operations :
  horner_count f = (6, 3) ∧
  horner_eval f 2 = f.foldl (fun acc a => acc * 2 + a) 0 :=
sorry

end NUMINAMATH_CALUDE_horner_method_operations_l2391_239101


namespace NUMINAMATH_CALUDE_thermos_capacity_is_16_l2391_239146

/-- The capacity of a coffee thermos -/
def thermos_capacity (fills_per_day : ℕ) (days_per_week : ℕ) (current_consumption : ℚ) (normal_consumption_ratio : ℚ) : ℚ :=
  (current_consumption / normal_consumption_ratio) / (fills_per_day * days_per_week)

/-- Proof that the thermos capacity is 16 ounces -/
theorem thermos_capacity_is_16 :
  thermos_capacity 2 5 40 (1/4) = 16 := by
  sorry

end NUMINAMATH_CALUDE_thermos_capacity_is_16_l2391_239146


namespace NUMINAMATH_CALUDE_cost_price_per_metre_values_l2391_239158

-- Define the cloth types
inductive ClothType
  | A
  | B
  | C

-- Define the properties for each cloth type
def metres_sold (t : ClothType) : ℕ :=
  match t with
  | ClothType.A => 200
  | ClothType.B => 150
  | ClothType.C => 100

def selling_price (t : ClothType) : ℕ :=
  match t with
  | ClothType.A => 10000
  | ClothType.B => 6000
  | ClothType.C => 4000

def loss (t : ClothType) : ℕ :=
  match t with
  | ClothType.A => 1000
  | ClothType.B => 450
  | ClothType.C => 200

-- Define the cost price per metre function
def cost_price_per_metre (t : ClothType) : ℚ :=
  (selling_price t + loss t : ℚ) / metres_sold t

-- State the theorem
theorem cost_price_per_metre_values :
  cost_price_per_metre ClothType.A = 55 ∧
  cost_price_per_metre ClothType.B = 43 ∧
  cost_price_per_metre ClothType.C = 42 := by
  sorry


end NUMINAMATH_CALUDE_cost_price_per_metre_values_l2391_239158


namespace NUMINAMATH_CALUDE_average_after_17th_inning_l2391_239159

def batsman_average (previous_innings : ℕ) (previous_total : ℕ) (new_score : ℕ) : ℚ :=
  (previous_total + new_score) / (previous_innings + 1)

theorem average_after_17th_inning 
  (previous_innings : ℕ) 
  (previous_total : ℕ) 
  (new_score : ℕ) 
  (average_increase : ℚ) :
  previous_innings = 16 →
  new_score = 88 →
  average_increase = 3 →
  batsman_average previous_innings previous_total new_score - 
    (previous_total / previous_innings) = average_increase →
  batsman_average previous_innings previous_total new_score = 40 :=
by
  sorry

#check average_after_17th_inning

end NUMINAMATH_CALUDE_average_after_17th_inning_l2391_239159


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l2391_239153

theorem complex_fraction_simplification :
  let z₁ : ℂ := 3 + 4*I
  let z₂ : ℂ := -1 + 2*I
  z₁ / z₂ = -5/3 + 10/3*I :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l2391_239153


namespace NUMINAMATH_CALUDE_power_of_complex_root_of_unity_l2391_239186

open Complex

theorem power_of_complex_root_of_unity : ((1 - I) / (Real.sqrt 2)) ^ 20 = 1 := by
  sorry

end NUMINAMATH_CALUDE_power_of_complex_root_of_unity_l2391_239186


namespace NUMINAMATH_CALUDE_expression_equals_four_l2391_239125

theorem expression_equals_four :
  (-2022)^0 - 2 * Real.tan (45 * π / 180) + |(-2)| + Real.sqrt 9 = 4 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_four_l2391_239125


namespace NUMINAMATH_CALUDE_average_children_in_families_with_children_l2391_239140

theorem average_children_in_families_with_children 
  (total_families : ℕ) 
  (total_average : ℚ) 
  (childless_families : ℕ) 
  (h1 : total_families = 12)
  (h2 : total_average = 3)
  (h3 : childless_families = 3) :
  (total_families * total_average) / (total_families - childless_families) = 4 := by
sorry

end NUMINAMATH_CALUDE_average_children_in_families_with_children_l2391_239140


namespace NUMINAMATH_CALUDE_annual_interest_rate_l2391_239105

/-- Calculate the annual interest rate given the borrowed amount and repayment amount after one year. -/
theorem annual_interest_rate (borrowed : ℝ) (repaid : ℝ) (h1 : borrowed = 150) (h2 : repaid = 165) :
  (repaid - borrowed) / borrowed * 100 = 10 := by
  sorry

end NUMINAMATH_CALUDE_annual_interest_rate_l2391_239105


namespace NUMINAMATH_CALUDE_scientific_notation_of_71300000_l2391_239194

theorem scientific_notation_of_71300000 :
  ∃ (a : ℝ) (n : ℤ), 71300000 = a * (10 : ℝ)^n ∧ 1 ≤ a ∧ a < 10 ∧ a = 7.13 ∧ n = 7 :=
sorry

end NUMINAMATH_CALUDE_scientific_notation_of_71300000_l2391_239194


namespace NUMINAMATH_CALUDE_x_value_l2391_239132

theorem x_value (x y z : ℝ) (h1 : x = y) (h2 : x = 2*z) (h3 : x*y*z = 256) : x = 8 := by
  sorry

end NUMINAMATH_CALUDE_x_value_l2391_239132


namespace NUMINAMATH_CALUDE_angle_condition_implies_y_range_l2391_239174

/-- Given points A(-1,1) and B(3,y), and vector a = (1,2), if the angle between AB and a is acute, 
    then y ∈ (-1,9) ∪ (9,+∞). -/
theorem angle_condition_implies_y_range (y : ℝ) : 
  let A : ℝ × ℝ := (-1, 1)
  let B : ℝ × ℝ := (3, y)
  let a : ℝ × ℝ := (1, 2)
  let AB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)
  (AB.1 * a.1 + AB.2 * a.2 > 0) → -- Dot product > 0 implies acute angle
  (y ∈ Set.Ioo (-1 : ℝ) 9 ∪ Set.Ioi 9) := by
sorry

end NUMINAMATH_CALUDE_angle_condition_implies_y_range_l2391_239174


namespace NUMINAMATH_CALUDE_airplane_seats_l2391_239198

/-- Calculates the total number of seats on an airplane given the number of coach class seats
    and the relationship between coach and first-class seats. -/
theorem airplane_seats (coach_seats : ℕ) (h1 : coach_seats = 310) 
    (h2 : ∃ first_class : ℕ, coach_seats = 4 * first_class + 2) : 
  coach_seats + (coach_seats - 2) / 4 = 387 := by
  sorry

#check airplane_seats

end NUMINAMATH_CALUDE_airplane_seats_l2391_239198


namespace NUMINAMATH_CALUDE_problem_solution_l2391_239142

theorem problem_solution (a : ℝ) (h1 : a > 0) : 
  (fun x => x^2 + 4) ((fun x => x^2 - 2) a) = 12 → 
  a = Real.sqrt (2 * (Real.sqrt 2 + 1)) := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l2391_239142


namespace NUMINAMATH_CALUDE_trip_participants_l2391_239149

theorem trip_participants :
  ∃ (men women children : ℕ),
    men + women + children = 150 ∧
    17 * men + 14 * women + 9 * children = 1530 ∧
    children < 120 ∧
    men = 5 ∧
    women = 28 ∧
    children = 117 := by
  sorry

end NUMINAMATH_CALUDE_trip_participants_l2391_239149


namespace NUMINAMATH_CALUDE_business_profit_theorem_l2391_239109

def business_profit_distribution (total_profit : ℝ) : ℝ :=
  let majority_owner_share := 0.25 * total_profit
  let remaining_profit := total_profit - majority_owner_share
  let partner_share := 0.25 * remaining_profit
  majority_owner_share + 2 * partner_share

theorem business_profit_theorem :
  business_profit_distribution 80000 = 50000 := by
  sorry

end NUMINAMATH_CALUDE_business_profit_theorem_l2391_239109


namespace NUMINAMATH_CALUDE_non_shaded_perimeter_l2391_239170

/-- Given a complex shape composed of rectangles, prove that the perimeter of the non-shaded region is 28 inches. -/
theorem non_shaded_perimeter (total_area shaded_area : ℝ) (h1 : total_area = 160) (h2 : shaded_area = 120) : ∃ (length width : ℝ), length * width = total_area - shaded_area ∧ 2 * (length + width) = 28 := by
  sorry

end NUMINAMATH_CALUDE_non_shaded_perimeter_l2391_239170


namespace NUMINAMATH_CALUDE_fashion_show_runway_time_l2391_239165

/-- The fashion show runway problem -/
theorem fashion_show_runway_time :
  let num_models : ℕ := 6
  let bathing_suits_per_model : ℕ := 2
  let evening_wear_per_model : ℕ := 3
  let time_per_trip : ℕ := 2

  let total_trips_per_model : ℕ := bathing_suits_per_model + evening_wear_per_model
  let total_trips : ℕ := num_models * total_trips_per_model
  let total_time : ℕ := total_trips * time_per_trip

  total_time = 60
  := by sorry

end NUMINAMATH_CALUDE_fashion_show_runway_time_l2391_239165


namespace NUMINAMATH_CALUDE_special_cone_volume_l2391_239139

/-- A cone with inscribed and circumscribed spheres having the same center -/
structure SpecialCone where
  /-- The radius of the inscribed sphere -/
  inscribed_radius : ℝ
  /-- The inscribed and circumscribed spheres have the same center -/
  spheres_same_center : Bool

/-- The volume of a SpecialCone -/
noncomputable def volume (cone : SpecialCone) : ℝ := sorry

/-- Theorem: The volume of a SpecialCone with inscribed radius 1 is 2π -/
theorem special_cone_volume (cone : SpecialCone) 
  (h1 : cone.inscribed_radius = 1) 
  (h2 : cone.spheres_same_center = true) : 
  volume cone = 2 * Real.pi := by sorry

end NUMINAMATH_CALUDE_special_cone_volume_l2391_239139


namespace NUMINAMATH_CALUDE_average_charge_is_five_l2391_239199

/-- Represents the charges and attendance for a three-day show -/
structure ShowData where
  day1_charge : ℚ
  day2_charge : ℚ
  day3_charge : ℚ
  day1_attendance : ℚ
  day2_attendance : ℚ
  day3_attendance : ℚ

/-- Calculates the average charge per person for the whole show -/
def averageCharge (data : ShowData) : ℚ :=
  let total_revenue := data.day1_charge * data.day1_attendance +
                       data.day2_charge * data.day2_attendance +
                       data.day3_charge * data.day3_attendance
  let total_attendance := data.day1_attendance + data.day2_attendance + data.day3_attendance
  total_revenue / total_attendance

/-- Theorem stating that the average charge for the given show data is 5 -/
theorem average_charge_is_five (data : ShowData)
  (h1 : data.day1_charge = 15)
  (h2 : data.day2_charge = 15/2)
  (h3 : data.day3_charge = 5/2)
  (h4 : data.day1_attendance = 2 * x)
  (h5 : data.day2_attendance = 5 * x)
  (h6 : data.day3_attendance = 13 * x)
  (h7 : x > 0) :
  averageCharge data = 5 := by
  sorry

end NUMINAMATH_CALUDE_average_charge_is_five_l2391_239199


namespace NUMINAMATH_CALUDE_plane_perpendicular_deduction_l2391_239164

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel relation for lines
variable (parallel : Line → Line → Prop)

-- Define the perpendicular relation between a line and a plane
variable (perp_line_plane : Line → Plane → Prop)

-- Define the perpendicular relation between planes
variable (perp_plane : Plane → Plane → Prop)

-- Define the subset relation for a line in a plane
variable (subset : Line → Plane → Prop)

-- State the theorem
theorem plane_perpendicular_deduction 
  (m n : Line) (α β : Plane) 
  (h1 : parallel m n) 
  (h2 : subset m α) 
  (h3 : perp_line_plane n β) : 
  perp_plane α β :=
sorry

end NUMINAMATH_CALUDE_plane_perpendicular_deduction_l2391_239164


namespace NUMINAMATH_CALUDE_train_length_l2391_239115

/-- The length of a train given its speed and time to cross an electric pole -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 36 → time = 30 → speed * time * (5 / 18) = 300 :=
by sorry

end NUMINAMATH_CALUDE_train_length_l2391_239115


namespace NUMINAMATH_CALUDE_min_value_function_compare_squares_min_value_M_l2391_239163

-- Part 1
theorem min_value_function (x : ℝ) (h : x > -1) :
  ∃ (min_val : ℝ), min_val = 2 * Real.sqrt 2 + 3 ∧
  ∀ (y : ℝ), y = ((x + 2) * (x + 3)) / (x + 1) → y ≥ min_val :=
sorry

-- Part 2
theorem compare_squares (a b x y : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1)
  (h : x^2 / a^2 - y^2 / b^2 = 1) :
  a^2 - b^2 ≤ (x - y)^2 :=
sorry

-- Part 3
theorem min_value_M (m : ℝ) (hm : m ≥ 1) :
  ∃ (min_val : ℝ), min_val = Real.sqrt 3 / 2 ∧
  ∀ (M : ℝ), M = Real.sqrt (4 * m - 3) - Real.sqrt (m - 1) → M ≥ min_val :=
sorry

end NUMINAMATH_CALUDE_min_value_function_compare_squares_min_value_M_l2391_239163


namespace NUMINAMATH_CALUDE_cost_of_paving_floor_l2391_239169

/-- The cost of paving a rectangular floor given its dimensions and rate per square meter. -/
theorem cost_of_paving_floor (length width rate : ℝ) : 
  length = 5.5 → width = 4 → rate = 800 → length * width * rate = 17600 := by
  sorry

end NUMINAMATH_CALUDE_cost_of_paving_floor_l2391_239169


namespace NUMINAMATH_CALUDE_water_collection_impossible_l2391_239154

def total_water (n : ℕ) : ℕ := n * (n + 1) / 2

def is_odd (n : ℕ) : Prop := ∃ k, n = 2 * k + 1

theorem water_collection_impossible (n : ℕ) (h : n = 2018) :
  is_odd (total_water n) ∧ 
  (∀ m : ℕ, m ≤ n → ∃ k, m = 2 ^ k) → False :=
by sorry

end NUMINAMATH_CALUDE_water_collection_impossible_l2391_239154


namespace NUMINAMATH_CALUDE_cube_root_sum_equation_l2391_239148

theorem cube_root_sum_equation (y : ℝ) (hy : y > 0) 
  (h : Real.rpow (2 - y^3) (1/3) + Real.rpow (2 + y^3) (1/3) = 2) : 
  y^6 = 116/27 := by sorry

end NUMINAMATH_CALUDE_cube_root_sum_equation_l2391_239148


namespace NUMINAMATH_CALUDE_factor_expression_l2391_239160

theorem factor_expression (x : ℝ) : 4*x*(x+2) + 10*(x+2) + 2*(x+2) = (x+2)*(4*x+12) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l2391_239160


namespace NUMINAMATH_CALUDE_piggy_bank_savings_l2391_239126

theorem piggy_bank_savings (first_year : ℝ) : 
  first_year + 2 * first_year + 4 * first_year + 8 * first_year = 450 →
  first_year = 30 := by
sorry

end NUMINAMATH_CALUDE_piggy_bank_savings_l2391_239126


namespace NUMINAMATH_CALUDE_total_school_capacity_l2391_239123

theorem total_school_capacity : 
  let school_count : ℕ := 4
  let capacity_a : ℕ := 400
  let capacity_b : ℕ := 340
  let schools_with_capacity_a : ℕ := 2
  let schools_with_capacity_b : ℕ := 2
  schools_with_capacity_a * capacity_a + schools_with_capacity_b * capacity_b = 1480 :=
by sorry

end NUMINAMATH_CALUDE_total_school_capacity_l2391_239123


namespace NUMINAMATH_CALUDE_min_value_expression_l2391_239143

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  ∃ (m : ℝ), m = Real.sqrt 3 ∧ 
  (∀ (x y : ℝ) (hx : x > 0) (hy : y > 0), 
    x^2 + x*y + y^2 + 1/(x+y)^2 ≥ m) ∧
  (∃ (u v : ℝ) (hu : u > 0) (hv : v > 0), 
    u^2 + u*v + v^2 + 1/(u+v)^2 = m) :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_l2391_239143


namespace NUMINAMATH_CALUDE_light_configurations_l2391_239118

/-- The number of rows and columns in the grid -/
def gridSize : Nat := 6

/-- The number of possible states for each switch (on or off) -/
def switchStates : Nat := 2

/-- The total number of different configurations of lights in the grid -/
def totalConfigurations : Nat := (switchStates ^ gridSize - 1) * (switchStates ^ gridSize - 1) + 1

/-- Theorem stating that the number of different configurations of lights is 3970 -/
theorem light_configurations :
  totalConfigurations = 3970 := by
  sorry

end NUMINAMATH_CALUDE_light_configurations_l2391_239118


namespace NUMINAMATH_CALUDE_absent_student_grade_calculation_l2391_239177

/-- Given a class where one student was initially absent for a test, prove that the absent student's grade can be determined from the class averages before and after including their score. -/
theorem absent_student_grade_calculation (total_students : ℕ) 
  (initial_students : ℕ) (initial_average : ℚ) (final_average : ℚ) 
  (h1 : total_students = 25) 
  (h2 : initial_students = 24)
  (h3 : initial_average = 82)
  (h4 : final_average = 84) :
  (total_students : ℚ) * final_average - (initial_students : ℚ) * initial_average = 132 := by
  sorry

end NUMINAMATH_CALUDE_absent_student_grade_calculation_l2391_239177


namespace NUMINAMATH_CALUDE_arrangements_of_six_acts_l2391_239157

/-- The number of ways to insert two distinguishable items into a sequence of n fixed items -/
def insert_two_items (n : ℕ) : ℕ :=
  (n + 1) * (n + 2)

/-- Theorem stating that inserting 2 items into a sequence of 4 fixed items results in 30 arrangements -/
theorem arrangements_of_six_acts : insert_two_items 4 = 30 := by
  sorry

end NUMINAMATH_CALUDE_arrangements_of_six_acts_l2391_239157


namespace NUMINAMATH_CALUDE_don_bottles_from_shop_c_l2391_239195

/-- The total number of bottles Don can buy -/
def total_bottles : ℕ := 550

/-- The number of bottles Don buys from Shop A -/
def shop_a_bottles : ℕ := 150

/-- The number of bottles Don buys from Shop B -/
def shop_b_bottles : ℕ := 180

/-- The number of bottles Don buys from Shop C -/
def shop_c_bottles : ℕ := total_bottles - (shop_a_bottles + shop_b_bottles)

theorem don_bottles_from_shop_c :
  shop_c_bottles = 220 :=
by sorry

end NUMINAMATH_CALUDE_don_bottles_from_shop_c_l2391_239195


namespace NUMINAMATH_CALUDE_polynomial_roots_l2391_239127

theorem polynomial_roots (x : ℝ) : x^4 - 3*x^3 + x^2 - 3*x = 0 ↔ x = 0 ∨ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_roots_l2391_239127


namespace NUMINAMATH_CALUDE_sin_product_equals_one_sixteenth_sin_cos_sum_equals_three_fourths_plus_quarter_sin_seventy_l2391_239144

-- Part 1
theorem sin_product_equals_one_sixteenth :
  Real.sin (6 * π / 180) * Real.sin (42 * π / 180) * Real.sin (66 * π / 180) * Real.sin (78 * π / 180) = 1 / 16 := by
  sorry

-- Part 2
theorem sin_cos_sum_equals_three_fourths_plus_quarter_sin_seventy :
  Real.sin (20 * π / 180) ^ 2 + Real.cos (50 * π / 180) ^ 2 + Real.sin (20 * π / 180) * Real.cos (50 * π / 180) =
  3 / 4 + (1 / 4) * Real.sin (70 * π / 180) := by
  sorry

end NUMINAMATH_CALUDE_sin_product_equals_one_sixteenth_sin_cos_sum_equals_three_fourths_plus_quarter_sin_seventy_l2391_239144


namespace NUMINAMATH_CALUDE_friends_meet_time_l2391_239102

def carl_lap : ℕ := 5
def jenna_lap : ℕ := 8
def marco_lap : ℕ := 9
def leah_lap : ℕ := 10

def start_time : ℕ := 9 * 60  -- 9:00 AM in minutes since midnight

theorem friends_meet_time :
  let meeting_time := start_time + Nat.lcm carl_lap (Nat.lcm jenna_lap (Nat.lcm marco_lap leah_lap))
  meeting_time = 15 * 60  -- 3:00 PM in minutes since midnight
  := by sorry

end NUMINAMATH_CALUDE_friends_meet_time_l2391_239102


namespace NUMINAMATH_CALUDE_smallest_x_squared_is_2135_l2391_239131

/-- An isosceles trapezoid with specific properties -/
structure IsoscelesTrapezoid where
  AB : ℝ
  CD : ℝ
  x : ℝ
  has_circle : Bool
  has_tangent_line : Bool

/-- The smallest possible value of x^2 for the given trapezoid -/
def smallest_x_squared (t : IsoscelesTrapezoid) : ℝ := 2135

/-- Theorem stating the smallest possible value of x^2 for the specific trapezoid -/
theorem smallest_x_squared_is_2135 (t : IsoscelesTrapezoid) 
  (h1 : t.AB = 122) 
  (h2 : t.CD = 26) 
  (h3 : t.has_circle = true) 
  (h4 : t.has_tangent_line = true) : 
  smallest_x_squared t = 2135 := by
  sorry

end NUMINAMATH_CALUDE_smallest_x_squared_is_2135_l2391_239131


namespace NUMINAMATH_CALUDE_infinite_subset_with_common_gcd_l2391_239178

-- Define the set A
def A : Set ℕ := {n : ℕ | ∃ (primes : Finset ℕ), primes.card ≤ 1987 ∧ (∀ p ∈ primes, Nat.Prime p) ∧ n = primes.prod id}

-- State the theorem
theorem infinite_subset_with_common_gcd (h : Set.Infinite A) :
  ∃ (B : Set ℕ) (b : ℕ), Set.Infinite B ∧ B ⊆ A ∧ ∀ (x y : ℕ), x ∈ B → y ∈ B → x ≠ y → Nat.gcd x y = b :=
sorry

end NUMINAMATH_CALUDE_infinite_subset_with_common_gcd_l2391_239178


namespace NUMINAMATH_CALUDE_league_games_count_l2391_239120

/-- The number of teams in each division -/
def teams_per_division : ℕ := 9

/-- The number of times each team plays other teams in its own division -/
def intra_division_games : ℕ := 3

/-- The number of times each team plays teams in the other division -/
def inter_division_games : ℕ := 2

/-- The number of divisions in the league -/
def num_divisions : ℕ := 2

/-- The total number of games scheduled in the league -/
def total_games : ℕ :=
  (num_divisions * (teams_per_division.choose 2 * intra_division_games)) +
  (teams_per_division * teams_per_division * inter_division_games)

theorem league_games_count : total_games = 378 := by
  sorry

end NUMINAMATH_CALUDE_league_games_count_l2391_239120


namespace NUMINAMATH_CALUDE_number_comparison_l2391_239167

theorem number_comparison : 22^44 > 33^33 ∧ 33^33 > 44^22 := by sorry

end NUMINAMATH_CALUDE_number_comparison_l2391_239167


namespace NUMINAMATH_CALUDE_inverse_proportion_problem_l2391_239166

-- Define the inverse proportionality constant
def k : ℝ → ℝ → ℝ := λ x y => x * y

-- Define the conditions
def conditions (x y : ℝ) : Prop :=
  ∃ (c : ℝ), k x y = c ∧ x + y = 30 ∧ x - y = 10

-- Theorem statement
theorem inverse_proportion_problem :
  ∀ x y : ℝ, conditions x y → (x = 4 → y = 50) :=
by sorry

end NUMINAMATH_CALUDE_inverse_proportion_problem_l2391_239166


namespace NUMINAMATH_CALUDE_sum_of_x_and_y_is_negative_two_l2391_239135

theorem sum_of_x_and_y_is_negative_two (x y : ℝ) 
  (hx : (x + 1) ^ (3/5 : ℝ) + 2023 * (x + 1) = -2023)
  (hy : (y + 1) ^ (3/5 : ℝ) + 2023 * (y + 1) = 2023) :
  x + y = -2 := by sorry

end NUMINAMATH_CALUDE_sum_of_x_and_y_is_negative_two_l2391_239135


namespace NUMINAMATH_CALUDE_ernest_wire_problem_l2391_239175

theorem ernest_wire_problem (total_parts : ℕ) (used_parts : ℕ) (unused_length : ℝ) :
  total_parts = 5 ∧ used_parts = 3 ∧ unused_length = 20 →
  total_parts * (unused_length / (total_parts - used_parts)) = 50 := by
  sorry

end NUMINAMATH_CALUDE_ernest_wire_problem_l2391_239175


namespace NUMINAMATH_CALUDE_train_speed_excluding_stoppages_l2391_239129

/-- Given a train that travels at 21 kmph including stoppages and stops for 18 minutes per hour,
    its speed excluding stoppages is 30 kmph. -/
theorem train_speed_excluding_stoppages
  (speed_with_stops : ℝ)
  (stop_time : ℝ)
  (h1 : speed_with_stops = 21)
  (h2 : stop_time = 18)
  : (speed_with_stops * 60) / (60 - stop_time) = 30 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_excluding_stoppages_l2391_239129


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_l2391_239184

theorem sum_of_reciprocals (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0)
  (h_sum : x + y = 3 * x * y) (h_diff : x - y = 1) :
  1 / x + 1 / y = Real.sqrt 13 + 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_l2391_239184


namespace NUMINAMATH_CALUDE_f_greatest_lower_bound_l2391_239103

noncomputable def f (x : ℝ) : ℝ := (x - 3) * Real.exp x

theorem f_greatest_lower_bound :
  ∃ (k : ℝ), k = -Real.exp 2 ∧
  (∀ x > 2, f x > k) ∧
  (∀ ε > 0, ∃ x > 2, f x < k + ε) :=
sorry

end NUMINAMATH_CALUDE_f_greatest_lower_bound_l2391_239103


namespace NUMINAMATH_CALUDE_todd_sum_equals_l2391_239100

/-- Represents the counting game with Todd, Tadd, and Tucker -/
structure CountingGame where
  max_count : ℕ
  todd_turn_length : ℕ → ℕ
  todd_start_positions : ℕ → ℕ

/-- Calculates the sum of numbers Todd declares in the game -/
def todd_sum (game : CountingGame) : ℕ :=
  sorry

/-- The specific game instance described in the problem -/
def specific_game : CountingGame :=
  { max_count := 5000
  , todd_turn_length := λ n => n + 1
  , todd_start_positions := λ n => sorry }

/-- Theorem stating the sum of Todd's numbers equals a specific value -/
theorem todd_sum_equals (result : ℕ) : todd_sum specific_game = result :=
  sorry

end NUMINAMATH_CALUDE_todd_sum_equals_l2391_239100


namespace NUMINAMATH_CALUDE_division_problem_l2391_239108

theorem division_problem (dividend quotient remainder divisor : ℕ) : 
  dividend = 127 ∧ quotient = 9 ∧ remainder = 1 ∧ 
  dividend = divisor * quotient + remainder →
  divisor = 14 := by sorry

end NUMINAMATH_CALUDE_division_problem_l2391_239108


namespace NUMINAMATH_CALUDE_max_decimal_places_is_14_complex_expression_decimal_places_l2391_239162

/-- The number of decimal places in 3.456789 -/
def decimal_places_a : ℕ := 6

/-- The number of decimal places in 6.78901234 -/
def decimal_places_b : ℕ := 8

/-- The expression ((10 ^ 5 * 3.456789) ^ 12) / (6.78901234 ^ 4)) ^ 9 -/
noncomputable def complex_expression : ℝ := 
  (((10 ^ 5 * 3.456789) ^ 12) / (6.78901234 ^ 4)) ^ 9

/-- The maximum number of decimal places in the result -/
def max_decimal_places : ℕ := decimal_places_a + decimal_places_b

theorem max_decimal_places_is_14 : 
  max_decimal_places = 14 := by sorry

theorem complex_expression_decimal_places : 
  ∃ (n : ℕ), n ≤ max_decimal_places ∧ 
  complex_expression * (10 ^ n) = ⌊complex_expression * (10 ^ n)⌋ := by sorry

end NUMINAMATH_CALUDE_max_decimal_places_is_14_complex_expression_decimal_places_l2391_239162


namespace NUMINAMATH_CALUDE_sin_cos_identity_l2391_239152

theorem sin_cos_identity : 
  Real.sin (43 * π / 180) * Real.sin (17 * π / 180) - 
  Real.cos (43 * π / 180) * Real.cos (17 * π / 180) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_identity_l2391_239152


namespace NUMINAMATH_CALUDE_F_of_4_f_of_5_equals_174_l2391_239185

-- Define the function f
def f (a : ℝ) : ℝ := 3 * a - 6

-- Define the function F
def F (a b : ℝ) : ℝ := 2 * b^2 + 3 * a

-- Theorem statement
theorem F_of_4_f_of_5_equals_174 : F 4 (f 5) = 174 := by
  sorry

end NUMINAMATH_CALUDE_F_of_4_f_of_5_equals_174_l2391_239185


namespace NUMINAMATH_CALUDE_exponential_greater_than_trig_squared_l2391_239191

theorem exponential_greater_than_trig_squared (x : ℝ) : 
  Real.exp x + Real.exp (-x) ≥ (Real.sin x + Real.cos x)^2 := by
  sorry

end NUMINAMATH_CALUDE_exponential_greater_than_trig_squared_l2391_239191


namespace NUMINAMATH_CALUDE_finite_solutions_egyptian_fraction_l2391_239136

theorem finite_solutions_egyptian_fraction :
  (∃ (S : Set (ℕ+ × ℕ+ × ℕ+)), Finite S ∧
    ∀ (a b c : ℕ+), (1 : ℚ) / a + (1 : ℚ) / b + (1 : ℚ) / c = (1 : ℚ) / 1983 ↔ (a, b, c) ∈ S) :=
by sorry

end NUMINAMATH_CALUDE_finite_solutions_egyptian_fraction_l2391_239136


namespace NUMINAMATH_CALUDE_largest_rational_root_quadratic_l2391_239168

theorem largest_rational_root_quadratic (a b c : ℕ+) 
  (ha : a ≤ 100) (hb : b ≤ 100) (hc : c ≤ 100) :
  let roots := {x : ℚ | a * x^2 + b * x + c = 0}
  ∃ (max_root : ℚ), max_root ∈ roots ∧ 
    ∀ (r : ℚ), r ∈ roots → r ≤ max_root ∧
    max_root = -1 / 99 := by
  sorry

end NUMINAMATH_CALUDE_largest_rational_root_quadratic_l2391_239168


namespace NUMINAMATH_CALUDE_impossible_equal_checkers_l2391_239114

/-- Represents a 3x3 grid of integers -/
def Grid := Fin 3 → Fin 3 → ℕ

/-- Represents an L-shape on the grid -/
inductive LShape
  | topLeft : LShape
  | topRight : LShape
  | bottomLeft : LShape
  | bottomRight : LShape

/-- Applies a move to the grid -/
def applyMove (grid : Grid) (shape : LShape) : Grid :=
  sorry

/-- Checks if all cells in the grid have the same non-zero value -/
def allCellsSame (grid : Grid) : Prop :=
  sorry

/-- Theorem stating the impossibility of reaching a state where all cells have the same non-zero value -/
theorem impossible_equal_checkers :
  ¬ ∃ (initial : Grid) (moves : List LShape),
    (∀ i j, initial i j = 0) ∧ 
    allCellsSame (moves.foldl applyMove initial) :=
  sorry

end NUMINAMATH_CALUDE_impossible_equal_checkers_l2391_239114


namespace NUMINAMATH_CALUDE_triangle_area_l2391_239113

theorem triangle_area (a b c : ℝ) (A B C : ℝ) :
  c^2 = (a - b)^2 + 6 →
  C = π / 3 →
  (1 / 2) * a * b * Real.sin C = (3 * Real.sqrt 3) / 2 :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_l2391_239113


namespace NUMINAMATH_CALUDE_subtracted_amount_l2391_239192

theorem subtracted_amount (N : ℝ) (A : ℝ) : 
  N = 100 → 0.7 * N - A = 30 → A = 40 := by sorry

end NUMINAMATH_CALUDE_subtracted_amount_l2391_239192


namespace NUMINAMATH_CALUDE_monday_appointment_duration_l2391_239151

/-- Amanda's hourly rate in dollars -/
def hourly_rate : ℝ := 20

/-- Number of appointments on Monday -/
def monday_appointments : ℕ := 5

/-- Hours worked on Tuesday -/
def tuesday_hours : ℝ := 3

/-- Hours worked on Thursday -/
def thursday_hours : ℝ := 4

/-- Hours worked on Saturday -/
def saturday_hours : ℝ := 6

/-- Total earnings for the week in dollars -/
def total_earnings : ℝ := 410

/-- Theorem: Given Amanda's schedule and earnings, each of her Monday appointments lasts 1.5 hours -/
theorem monday_appointment_duration :
  (total_earnings - hourly_rate * (tuesday_hours + thursday_hours + saturday_hours)) / 
  (hourly_rate * monday_appointments) = 1.5 := by
  sorry

end NUMINAMATH_CALUDE_monday_appointment_duration_l2391_239151


namespace NUMINAMATH_CALUDE_find_number_to_multiply_l2391_239116

theorem find_number_to_multiply : ∃ x : ℕ, 
  (43 * x) - (34 * x) = 1224 ∧ x = 136 := by
  sorry

end NUMINAMATH_CALUDE_find_number_to_multiply_l2391_239116


namespace NUMINAMATH_CALUDE_max_area_AOB_l2391_239117

-- Define the circles E and F
def circle_E (x y : ℝ) : Prop := (x + Real.sqrt 3)^2 + y^2 = 25
def circle_F (x y : ℝ) : Prop := (x - Real.sqrt 3)^2 + y^2 = 1

-- Define the curve C (locus of center of P)
def curve_C (x y : ℝ) : Prop := x^2/4 + y^2 = 1

-- Define a line l
def line_l (m n x y : ℝ) : Prop := x = m * y + n

-- Define points A and B on curve C and line l
def point_on_C_and_l (x y m n : ℝ) : Prop :=
  curve_C x y ∧ line_l m n x y

-- Define midpoint M of AB
def midpoint_M (xm ym xa ya xb yb : ℝ) : Prop :=
  xm = (xa + xb) / 2 ∧ ym = (ya + yb) / 2

-- Define |OM| = 1
def OM_unit_length (xm ym : ℝ) : Prop :=
  xm^2 + ym^2 = 1

-- Main theorem
theorem max_area_AOB :
  ∀ (xa ya xb yb xm ym m n : ℝ),
  point_on_C_and_l xa ya m n →
  point_on_C_and_l xb yb m n →
  midpoint_M xm ym xa ya xb yb →
  OM_unit_length xm ym →
  ∃ (S : ℝ), S ≤ 1 ∧
  (∀ (S' : ℝ), S' = abs ((xa * yb - xb * ya) / 2) → S' ≤ S) :=
sorry

end NUMINAMATH_CALUDE_max_area_AOB_l2391_239117


namespace NUMINAMATH_CALUDE_find_number_l2391_239133

theorem find_number : ∃ x : ℝ, 3 * (x + 8) = 36 ∧ x = 4 := by sorry

end NUMINAMATH_CALUDE_find_number_l2391_239133


namespace NUMINAMATH_CALUDE_inverse_difference_equals_negative_reciprocal_l2391_239181

theorem inverse_difference_equals_negative_reciprocal (a b : ℝ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hab : 3 * a - b / 3 ≠ 0) :
  (3 * a - b / 3)⁻¹ * ((3 * a)⁻¹ - (b / 3)⁻¹) = -(a * b)⁻¹ := by
  sorry

end NUMINAMATH_CALUDE_inverse_difference_equals_negative_reciprocal_l2391_239181


namespace NUMINAMATH_CALUDE_lunch_combo_count_l2391_239155

/-- Represents the number of options for each food category --/
structure FoodOptions where
  lettuce : Nat
  tomatoes : Nat
  olives : Nat
  bread : Nat
  fruit : Nat
  soup : Nat

/-- Calculates the number of ways to choose k items from n items --/
def choose (n k : Nat) : Nat :=
  Nat.choose n k

/-- Calculates the total number of lunch combo options --/
def lunchComboOptions (options : FoodOptions) : Nat :=
  let remainingItems := options.olives + options.bread + options.fruit
  let remainingChoices := choose remainingItems 3
  options.lettuce * options.tomatoes * remainingChoices * options.soup

/-- Theorem stating the number of lunch combo options --/
theorem lunch_combo_count (options : FoodOptions) 
  (h1 : options.lettuce = 4)
  (h2 : options.tomatoes = 5)
  (h3 : options.olives = 6)
  (h4 : options.bread = 3)
  (h5 : options.fruit = 4)
  (h6 : options.soup = 3) :
  lunchComboOptions options = 17160 := by
  sorry

#eval lunchComboOptions { lettuce := 4, tomatoes := 5, olives := 6, bread := 3, fruit := 4, soup := 3 }

end NUMINAMATH_CALUDE_lunch_combo_count_l2391_239155


namespace NUMINAMATH_CALUDE_only_fourth_equation_has_real_roots_l2391_239176

-- Define the discriminant function
def discriminant (a b c : ℝ) : ℝ := b^2 - 4*a*c

-- Define a function to check if a quadratic equation has real roots
def hasRealRoots (a b c : ℝ) : Prop := discriminant a b c ≥ 0

-- Theorem statement
theorem only_fourth_equation_has_real_roots :
  ¬(hasRealRoots 1 0 1) ∧
  ¬(hasRealRoots 1 1 1) ∧
  ¬(hasRealRoots 1 (-1) 1) ∧
  hasRealRoots 1 (-1) (-1) :=
sorry

end NUMINAMATH_CALUDE_only_fourth_equation_has_real_roots_l2391_239176


namespace NUMINAMATH_CALUDE_polygon_sides_range_l2391_239106

/-- Represents the count of vertices with different internal angles -/
structure VertexCounts where
  a : ℕ  -- Count of 60° angles
  b : ℕ  -- Count of 90° angles
  c : ℕ  -- Count of 120° angles
  d : ℕ  -- Count of 150° angles

/-- Theorem stating the possible values of n for a convex n-sided polygon 
    formed by combining equilateral triangles and squares -/
theorem polygon_sides_range (n : ℕ) : 
  (∃ v : VertexCounts, 
    v.a + v.b + v.c + v.d = n ∧ 
    4 * v.a + 3 * v.b + 2 * v.c + v.d = 12 ∧
    v.a + v.b > 0 ∧ v.c + v.d > 0) ↔ 
  5 ≤ n ∧ n ≤ 12 :=
sorry

end NUMINAMATH_CALUDE_polygon_sides_range_l2391_239106
