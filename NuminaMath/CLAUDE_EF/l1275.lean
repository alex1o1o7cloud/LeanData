import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_sum_values_l1275_127526

theorem matrix_sum_values (x y z : ℝ) : 
  let M : Matrix (Fin 3) (Fin 3) ℝ := ![![x, y, z], ![y, z, x], ![z, x, y]]
  ¬(IsUnit (Matrix.det M)) →
  (x / (y + z) + y / (z + x) + z / (x + y) = -3) ∨
  (x / (y + z) + y / (z + x) + z / (x + y) = 3/2) :=
by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_sum_values_l1275_127526


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identity_l1275_127500

theorem trigonometric_identity (a b : ℝ) (θ : ℝ) (h : 0 < a) (k : 0 < b) :
  (Real.sin θ ^ 6 / a ^ 2 + Real.cos θ ^ 6 / b ^ 2 = 1 / (a + b)) →
  (Real.sin θ ^ 12 / a ^ 5 + Real.cos θ ^ 12 / b ^ 5 = 1 / (a + b) ^ 5) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identity_l1275_127500


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_x_satisfying_inequality_l1275_127515

/-- Given three positive real numbers a₁, a₂, a₃ in decreasing order,
    this theorem states that the range of x satisfying (1 - aᵢx)² < 1
    for i = 1, 2, 3 is the open interval (0, 2/a₁). -/
theorem range_of_x_satisfying_inequality
  (a₁ a₂ a₃ : ℝ) (h₁ : a₁ > a₂) (h₂ : a₂ > a₃) (h₃ : a₃ > 0) :
  {x : ℝ | (1 - a₁ * x)^2 < 1 ∧ (1 - a₂ * x)^2 < 1 ∧ (1 - a₃ * x)^2 < 1} =
  Set.Ioo 0 (2 / a₁) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_x_satisfying_inequality_l1275_127515


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_area_calculation_l1275_127537

/-- The area of a circular sector with central angle α and radius r -/
noncomputable def sectorArea (α : ℝ) (r : ℝ) : ℝ := (1/2) * α * r^2

theorem sector_area_calculation (α r : ℝ) 
  (h1 : α = 2) 
  (h2 : r = 3) : 
  sectorArea α r = 9 := by
  -- Unfold the definition of sectorArea
  unfold sectorArea
  -- Substitute the values of α and r
  rw [h1, h2]
  -- Simplify the expression
  ring
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_area_calculation_l1275_127537


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_at_midpoints_l1275_127553

/-- A square with side length 1 -/
structure Square where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  is_square : A = (0, 0) ∧ B = (1, 0) ∧ C = (1, 1) ∧ D = (0, 1)

/-- A point on the side of a square -/
structure PointOnSide where
  x : ℝ
  y : ℝ
  on_side : (x = 0 ∧ 0 ≤ y ∧ y ≤ 1) ∨ 
            (y = 0 ∧ 0 ≤ x ∧ x ≤ 1) ∨ 
            (x = 1 ∧ 0 ≤ y ∧ y ≤ 1) ∨ 
            (y = 1 ∧ 0 ≤ x ∧ x ≤ 1)

/-- The area of the quadrilateral PUQV -/
noncomputable def quadrilateralArea (s : Square) (u v : PointOnSide) : ℝ :=
  sorry -- Definition of the area calculation

/-- Theorem: The area of PUQV is maximized when U and V are midpoints -/
theorem max_area_at_midpoints (s : Square) :
  ∃ (u v : PointOnSide),
    (∀ (u' v' : PointOnSide), quadrilateralArea s u v ≥ quadrilateralArea s u' v') ∧
    u.x = 0.5 ∧ u.y = 0 ∧ v.x = 1 ∧ v.y = 0.5 := by
  sorry -- Proof to be implemented


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_at_midpoints_l1275_127553


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_APR_is_40_l1275_127585

/-- A circle in a 2D plane. -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A point in a 2D plane. -/
@[ext] structure Point where
  x : ℝ
  y : ℝ

/-- A line segment between two points. -/
structure Segment where
  start : Point
  finish : Point  -- Changed 'end' to 'finish' to avoid keyword conflict

/-- Checks if a line is tangent to a circle. -/
def isTangent (l : Segment) (c : Circle) : Prop := sorry

/-- Calculates the length of a segment. -/
noncomputable def length (s : Segment) : ℝ := sorry

/-- Calculates the perimeter of a triangle given by three points. -/
noncomputable def trianglePerimeter (p1 p2 p3 : Point) : ℝ := sorry

/-- Theorem: The perimeter of triangle APR is 40. -/
theorem perimeter_APR_is_40 
  (c : Circle) (A B C P Q R : Point) 
  (ab ac pq : Segment) :
  isTangent ab c → 
  isTangent ac c → 
  isTangent pq c →
  ab.start = A ∧ ab.finish = B →
  ac.start = A ∧ ac.finish = C →
  pq.start = P ∧ pq.finish = R →
  length ab = 20 →
  trianglePerimeter A P R = 40 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_APR_is_40_l1275_127585


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l1275_127508

-- Define the constants
noncomputable def a : ℝ := (1/2: ℝ)^(0.3 : ℝ)
noncomputable def b : ℝ := (1/2 : ℝ)^(-2 : ℝ)
noncomputable def c : ℝ := Real.log 2 / Real.log (1/2)

-- State the theorem
theorem relationship_abc : b > a ∧ a > c := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l1275_127508


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_quadrants_l1275_127529

/-- A function representing an inverse proportion --/
noncomputable def inverse_proportion (m : ℝ) : ℝ → ℝ := fun x => m / x

/-- Predicate to check if a point (x, y) is in the second quadrant --/
def in_second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0

/-- Predicate to check if a point (x, y) is in the fourth quadrant --/
def in_fourth_quadrant (x y : ℝ) : Prop := x > 0 ∧ y < 0

/-- Theorem stating that if an inverse proportion function passes through
    the second and fourth quadrants, then its coefficient m must be negative --/
theorem inverse_proportion_quadrants (m : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ 0 ∧ x₂ ≠ 0 ∧
    in_second_quadrant x₁ (inverse_proportion m x₁) ∧
    in_fourth_quadrant x₂ (inverse_proportion m x₂)) →
  m < 0 :=
by
  intro h
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_quadrants_l1275_127529


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eventual_half_occupied_l1275_127559

/-- Represents the state of the circle with chips -/
structure CircleState where
  n : ℕ  -- number of sectors
  chips : ℕ  -- total number of chips
  occupied : Finset ℕ  -- set of occupied sectors

/-- The transformation step -/
def transform (state : CircleState) : CircleState :=
  sorry

/-- Predicate to check if at least half of the sectors are occupied -/
def half_occupied (state : CircleState) : Prop :=
  state.occupied.card ≥ (state.n + 1) / 2

/-- Main theorem -/
theorem eventual_half_occupied (n : ℕ) (h : n > 0) :
  ∃ (steps : ℕ) (final_state : CircleState),
    final_state.n = n ∧
    final_state.chips = n + 1 ∧
    (∃ (initial_state : CircleState),
      initial_state.n = n ∧
      initial_state.chips = n + 1 ∧
      (Nat.iterate transform steps initial_state = final_state)) ∧
    half_occupied final_state :=
  sorry

#check eventual_half_occupied

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eventual_half_occupied_l1275_127559


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1275_127519

noncomputable section

open Real

def f (a b x : ℝ) : ℝ := -2 * a * sin (2 * x + Real.pi / 6) + 2 * a + b

def g (a b x : ℝ) : ℝ := f a b (x + Real.pi / 2)

theorem function_properties (a b : ℝ) (h1 : a > 0) 
  (h2 : ∀ x ∈ Set.Icc 0 (Real.pi / 2), -5 ≤ f a b x ∧ f a b x ≤ 1) 
  (h3 : ∀ x, log (abs (g a b x)) > 0) :
  (a = 2 ∧ b = -5) ∧
  (∀ k : ℤ, StrictMonoOn (g a b) (Set.Ioo (k * Real.pi) (Real.pi / 6 + k * Real.pi)) ∧
            StrictMonoOn (g a b) (Set.Icc (Real.pi / 6 + k * Real.pi) (Real.pi / 3 + k * Real.pi))) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1275_127519


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_m_value_proof_l1275_127538

/-- Given two functions f and g, prove that m = 10 satisfies the condition 3f(4) = g(4) -/
theorem m_value_proof (m : ℝ) : 
  (fun (x : ℝ) => x^2 - 2*x + m) 4 * 3 = (fun (x : ℝ) => x^2 - 3*x + 5*m) 4 →
  m = 10 := by
  intro h
  -- The proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_m_value_proof_l1275_127538


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_cos_sin_max_value_l1275_127577

theorem sin_cos_cos_sin_max_value (x : ℝ) :
  Real.sin (Real.cos x) + Real.cos (Real.sin x) ≤ Real.sin 1 + 1 ∧
  (Real.sin (Real.cos x) + Real.cos (Real.sin x) = Real.sin 1 + 1 ↔ ∃ k : ℤ, x = 2 * Real.pi * ↑k) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_cos_sin_max_value_l1275_127577


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_book_sales_l1275_127591

-- Define the initial sales and increase rates
def monday_sales : ℕ := 25
def weekday_increase_rate : ℚ := 110 / 100
def thursday_promotion_rate : ℚ := 120 / 100
def saturday_increase_rate : ℚ := 200 / 100
def saturday_promotion_rate : ℚ := 125 / 100
def sunday_decrease_rate : ℚ := 50 / 100
def sunday_promotion_rate : ℚ := 115 / 100

-- Function to calculate daily sales
def calculate_daily_sales (previous_day_sales : ℚ) (increase_rate : ℚ) : ℕ :=
  Int.toNat ((previous_day_sales * increase_rate).floor)

-- Function to calculate promoted sales
def calculate_promoted_sales (expected_sales : ℚ) (promotion_rate : ℚ) : ℕ :=
  Int.toNat ((expected_sales * promotion_rate).floor)

-- Theorem statement
theorem total_book_sales : 
  let tuesday_sales := calculate_daily_sales monday_sales weekday_increase_rate
  let wednesday_sales := calculate_daily_sales tuesday_sales weekday_increase_rate
  let thursday_sales := calculate_promoted_sales (calculate_daily_sales wednesday_sales weekday_increase_rate) thursday_promotion_rate
  let friday_sales := calculate_daily_sales (thursday_sales / thursday_promotion_rate) weekday_increase_rate
  let saturday_sales := calculate_promoted_sales (friday_sales * saturday_increase_rate) saturday_promotion_rate
  let sunday_sales := calculate_promoted_sales (saturday_sales * sunday_decrease_rate) sunday_promotion_rate
  monday_sales + tuesday_sales + wednesday_sales + thursday_sales + friday_sales + saturday_sales + sunday_sales = 286 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_book_sales_l1275_127591


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_divisible_by_seven_l1275_127514

def sequence_nums : List ℕ := List.range 10 |> List.map (λ n => 10 * n + 3)

theorem product_divisible_by_seven :
  (List.prod sequence_nums) % 7 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_divisible_by_seven_l1275_127514


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_function_range_l1275_127510

-- Define the function y = (2a-1)^x
noncomputable def y (a : ℝ) (x : ℝ) : ℝ := (2 * a - 1) ^ x

-- State the theorem
theorem decreasing_function_range (a : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → y a x₁ > y a x₂) ↔ (1/2 < a ∧ a < 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_function_range_l1275_127510


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_shift_value_l1275_127527

noncomputable def f (x : ℝ) := Real.cos (2 * x)

theorem min_shift_value (φ : ℝ) (h1 : φ > 0) 
  (h2 : f (π/3 + φ) = 0) : 
  ∀ ψ > 0, f (π/3 + ψ) = 0 → ψ ≥ 5*π/12 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_shift_value_l1275_127527


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_arccos_one_fourth_l1275_127557

theorem sin_arccos_one_fourth : 
  Real.sin (Real.arccos (1/4)) = Real.sqrt 15 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_arccos_one_fourth_l1275_127557


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_income_minimum_income_always_higher_l1275_127574

/-- Income sequence after transfer -/
noncomputable def income_sequence (a b : ℝ) : ℕ → ℝ
  | 0 => a  -- Adding the case for 0
  | 1 => a
  | n + 2 => a * (2/3)^(n+1) + b * (3/2)^n

theorem income_minimum (a : ℝ) (ha : a > 0) :
  let b := 8*a/27
  ∃ (n : ℕ), ∀ (m : ℕ), income_sequence a b n ≤ income_sequence a b m ∧
  income_sequence a b n = 8*a/9 ∧ n = 3 := by
  sorry

theorem income_always_higher (a b : ℝ) (ha : a > 0) (hb : b ≥ 3*a/8) :
  ∀ (n : ℕ), n > 1 → income_sequence a b n > a := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_income_minimum_income_always_higher_l1275_127574


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_f_is_odd_range_of_inequality_l1275_127595

-- Define the function f(x) as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.log ((1 - x) / (1 + x))

-- Theorem for the domain of f(x)
theorem domain_of_f : Set.Ioo (-1 : ℝ) 1 = {x | f x ∈ Set.univ} := by sorry

-- Theorem for the odd symmetry of f(x)
theorem f_is_odd : ∀ x, f (-x) = -f x := by sorry

-- Theorem for the range of x satisfying the inequality
theorem range_of_inequality : Set.Ioo (-1/3 : ℝ) 1 = {x | f x < Real.log 2} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_f_is_odd_range_of_inequality_l1275_127595


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt51_parts_l1275_127560

-- Define the integer part function
noncomputable def intPart (x : ℝ) : ℤ :=
  Int.floor x

-- Define the decimal part function
noncomputable def decPart (x : ℝ) : ℝ :=
  x - Int.floor x

-- Given conditions
axiom cond1 : (4 : ℝ) < 7 ∧ 7 < 9
axiom cond2 : (2 : ℝ) < Real.sqrt 7 ∧ Real.sqrt 7 < 3
axiom cond3 : intPart (Real.sqrt 7) = 2
axiom cond4 : decPart (Real.sqrt 7) = Real.sqrt 7 - 2

-- Theorem to prove
theorem sqrt51_parts :
  intPart (Real.sqrt 51) = 7 ∧
  decPart (9 - Real.sqrt 51) = 8 - Real.sqrt 51 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt51_parts_l1275_127560


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_to_bookstore_l1275_127561

-- Define the billing rules
def base_fare : ℚ := 3
def base_distance : ℚ := 2
def additional_rate : ℚ := 6/5  -- 1.2 as a rational number
def total_fare : ℚ := 9

-- Define the function to calculate the fare based on distance
def fare_function (distance : ℚ) : ℚ :=
  if distance ≤ base_distance then base_fare
  else base_fare + additional_rate * (distance - base_distance)

-- Theorem statement
theorem max_distance_to_bookstore :
  ∃ (distance : ℚ), fare_function distance = total_fare ∧
    ∀ (d : ℚ), fare_function d = total_fare → d ≤ distance :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_to_bookstore_l1275_127561


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_folders_for_megan_l1275_127594

/-- The number of folders needed given initial files, added files, and files per folder -/
def folders_needed (initial_files added_files files_per_folder : ℚ) : ℕ :=
  (((initial_files + added_files) / files_per_folder).ceil).toNat

/-- Theorem stating that given the specific values, the number of folders needed is 14 -/
theorem folders_for_megan :
  folders_needed 93.5 21.25 8.75 = 14 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_folders_for_megan_l1275_127594


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_1_and_2_l1275_127589

theorem problem_1_and_2 : 
  ((-7/9 + 5/6 - 3/4) * (-36) = 25) ∧ 
  (-(1^4) - (1 - 1/2) * (1/2) * |1 - (-3)^2| = -3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_1_and_2_l1275_127589


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sinusoidal_function_properties_l1275_127581

-- Define the function f
noncomputable def f (A ω φ : ℝ) (x : ℝ) : ℝ := A * Real.sin (ω * x + φ)

-- State the theorem
theorem sinusoidal_function_properties 
  (A ω φ : ℝ) 
  (h1 : A > 0) 
  (h2 : ω > 0) 
  (h3 : |φ| < π) 
  (h4 : f A ω φ (π/12) = 3) 
  (h5 : f A ω φ (7*π/12) = -3) :
  (∃ (x : ℝ), f A ω φ x = 3 * Real.sin (2*x + π/3)) ∧ 
  (∀ (x : ℝ), x ∈ Set.Icc (-π/3) (π/6) → 
    f A ω φ x ∈ Set.Icc (-3 * Real.sqrt 3 / 2) 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sinusoidal_function_properties_l1275_127581


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_vertex_location_l1275_127506

/-- A parabola represented by the coefficients of its quadratic equation -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The x-coordinate of the vertex of a parabola -/
noncomputable def vertex_x (p : Parabola) : ℝ := -p.b / (2 * p.a)

/-- The parabola is increasing on the left side of the y-axis -/
def increasing_left (p : Parabola) : Prop := p.a < 0

/-- The parabola does not pass through the second quadrant -/
noncomputable def not_in_second_quadrant (p : Parabola) : Prop := vertex_x p > 0

/-- The vertex is not in the second or third quadrant -/
noncomputable def vertex_not_in_second_or_third (p : Parabola) : Prop :=
  vertex_x p > 0

theorem parabola_vertex_location (p : Parabola) :
  increasing_left p → not_in_second_quadrant p → vertex_not_in_second_or_third p :=
by
  intro h1 h2
  exact h2


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_vertex_location_l1275_127506


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_marble_203_is_red_l1275_127565

/-- Represents the color of a marble -/
inductive MarbleColor
  | Blue
  | Red
  | Green

/-- Returns the color of the nth marble in the sequence -/
def marbleColor (n : ℕ) : MarbleColor :=
  match n % 15 with
  | k => if k < 6 then MarbleColor.Blue
         else if k < 11 then MarbleColor.Red
         else MarbleColor.Green

theorem marble_203_is_red :
  marbleColor 203 = MarbleColor.Red := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_marble_203_is_red_l1275_127565


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_intersection_points_l1275_127568

/-- A circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A line in a 2D plane -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Represents the intersection of a line with a circle -/
def intersectionPoints (l : Line) (c : Circle) : Fin 2 → Option (ℝ × ℝ) := sorry

/-- The theorem stating the maximum number of intersection points -/
theorem max_intersection_points (circles : Fin 4 → Circle) (l : Line) :
  (∃ (points : Fin 8 → ℝ × ℝ), ∀ i : Fin 8, ∃ j : Fin 4, 
    intersectionPoints l (circles j) ⟨i.val % 2, by sorry⟩ = some (points i)) ∧
  (∀ (points : Fin 9 → ℝ × ℝ), ∃ i : Fin 9, ∀ j : Fin 4,
    intersectionPoints l (circles j) ⟨i.val % 2, by sorry⟩ ≠ some (points i)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_intersection_points_l1275_127568


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_son_work_time_l1275_127584

/-- Represents the time taken to complete a job -/
def JobTime := ℝ

/-- Represents the rate at which work is done (portion of job completed per day) -/
def WorkRate := ℝ

theorem son_work_time 
  (man_time : JobTime) 
  (combined_time : JobTime) 
  (h1 : man_time = (5 : ℝ))
  (h2 : combined_time = (4 : ℝ)) :
  ∃ (son_time : JobTime), son_time = (20 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_son_work_time_l1275_127584


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_painting_possibilities_l1275_127532

theorem floor_painting_possibilities :
  ∃! n : ℕ, n = (Finset.filter 
    (λ p : ℕ × ℕ ↦ 
      let (a, b) := p
      b > a ∧ 
      (a - 4) * (b - 4) = 2 * (a * b) / 3 ∧
      a > 0 ∧ b > 0
    ) 
    (Finset.product (Finset.range 100) (Finset.range 100))).card ∧ 
  n = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_painting_possibilities_l1275_127532


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_line_intersection_l1275_127512

/-- Represents a parabola with equation x^2 = 2py --/
structure Parabola where
  p : ℝ
  hp : p > 0

/-- Represents a line with slope 1 passing through the focus of a parabola --/
def focusLine (para : Parabola) : Set (ℝ × ℝ) :=
  {(x, y) | y = x + para.p / 2}

/-- Represents the trapezoid formed by the intersection of the focus line and the parabola --/
def intersectionTrapezoid (para : Parabola) : Set (ℝ × ℝ) :=
  {(x, y) | x^2 = 2 * para.p * y ∧ (x, y) ∈ focusLine para}

/-- The area of the trapezoid formed by the intersection --/
noncomputable def trapezoidArea (para : Parabola) : ℝ := 3 * Real.sqrt 2 * para.p^2

theorem parabola_focus_line_intersection (para : Parabola) :
  trapezoidArea para = 12 * Real.sqrt 2 → para.p = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_line_intersection_l1275_127512


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_rows_theorem_l1275_127558

/-- A table of integers with N rows and 100 columns -/
def Table (N : ℕ) := Fin N → Fin 100 → ℕ

/-- Property that every row contains the numbers 1 to 100 in some order -/
def RowProperty (T : Table N) : Prop :=
  ∀ r : Fin N, ∃ σ : Equiv.Perm (Fin 100), ∀ c : Fin 100, T r c = σ c.val + 1

/-- Property that for any two distinct rows, there is a column where the difference is at least 2 -/
def ColumnProperty (T : Table N) : Prop :=
  ∀ r s : Fin N, r ≠ s → ∃ c : Fin 100, (T r c : Int) - (T s c : Int) ≥ 2 ∨ (T s c : Int) - (T r c : Int) ≥ 2

/-- The existence of a table satisfying both properties -/
def ValidTable (N : ℕ) : Prop :=
  ∃ T : Table N, RowProperty T ∧ ColumnProperty T

/-- The maximum number of rows in a valid table -/
def MaxRows : ℕ := Nat.factorial 100 / 2^50

/-- The main theorem: MaxRows is the largest N for which a valid table exists -/
theorem max_rows_theorem :
  (∀ N : ℕ, N > MaxRows → ¬ValidTable N) ∧
  ValidTable MaxRows :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_rows_theorem_l1275_127558


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_dot_product_l1275_127505

-- Define the points A and B
def A : ℝ × ℝ := (0, 2)
def B : ℝ × ℝ := (-2, 0)

-- Define the curve on which P moves
noncomputable def curve (y : ℝ) : ℝ := Real.sqrt (1 - y^2)

-- Define the vector BA
def vectorBA : ℝ × ℝ := (A.1 - B.1, A.2 - B.2)

-- Define the vector BP as a function of y
noncomputable def vectorBP (y : ℝ) : ℝ × ℝ := (curve y + 2, y)

-- Define the dot product of BA and BP
noncomputable def dotProduct (y : ℝ) : ℝ :=
  vectorBA.1 * (vectorBP y).1 + vectorBA.2 * (vectorBP y).2

-- Statement: The maximum value of the dot product is 4 + 2√2
theorem max_dot_product :
  ∃ (max : ℝ), max = 4 + 2 * Real.sqrt 2 ∧
  ∀ (y : ℝ), -1 ≤ y ∧ y ≤ 1 → dotProduct y ≤ max := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_dot_product_l1275_127505


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_in_triangle_and_circle_l1275_127503

/-- Triangle DEF with side lengths -/
structure Triangle where
  de : ℝ
  ef : ℝ
  df : ℝ

/-- Inscribed circle in the triangle -/
structure InscribedCircle where
  radius : ℝ

/-- Region closer to F than to D or E -/
noncomputable def closerToF (t : Triangle) : Set (ℝ × ℝ) := sorry

/-- Area of the triangle -/
noncomputable def triangleArea (t : Triangle) : ℝ := sorry

/-- Area of overlap between the region closer to F and the inscribed circle -/
noncomputable def overlapArea (t : Triangle) (c : InscribedCircle) : ℝ := sorry

/-- Probability of a point being in the overlap region -/
noncomputable def probability (t : Triangle) (c : InscribedCircle) : ℝ :=
  overlapArea t c / triangleArea t

/-- Main theorem -/
theorem probability_in_triangle_and_circle (t : Triangle) (c : InscribedCircle) 
  (h1 : t.de = 8) (h2 : t.ef = 6) (h3 : t.df = 10) (h4 : c.radius = 1) :
  probability t c = overlapArea t c / triangleArea t := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_in_triangle_and_circle_l1275_127503


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_black_ball_count_l1275_127597

/-- Represents the number of balls of each color in the bag -/
structure BagContents where
  red : ℕ
  black : ℕ
  yellow : ℕ

/-- The total number of balls in the bag -/
def totalBalls : ℕ := 30

/-- The proposition that the bag contents are valid -/
def validBagContents (bag : BagContents) : Prop :=
  bag.red + bag.black + bag.yellow = totalBalls

/-- The frequency of drawing a red or yellow ball -/
noncomputable def redYellowFrequency (bag : BagContents) : ℝ :=
  (bag.red + bag.yellow : ℝ) / totalBalls

/-- The theorem stating the number of black balls given the conditions -/
theorem black_ball_count (bag : BagContents) 
  (valid : validBagContents bag)
  (freq_lower : redYellowFrequency bag ≥ 0.15)
  (freq_upper : redYellowFrequency bag ≤ 0.45) :
  bag.black = 20 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_black_ball_count_l1275_127597


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_worker_y_defective_rate_l1275_127517

/-- Defective rate of worker x -/
noncomputable def defective_rate_x : ℝ := 0.005

/-- Proportion of products checked by worker y -/
noncomputable def proportion_y : ℝ := 2/3

/-- Total defective rate of all products -/
noncomputable def total_defective_rate : ℝ := 0.007

/-- Calculate the defective rate of worker y -/
noncomputable def defective_rate_y : ℝ := 
  (total_defective_rate - (1 - proportion_y) * defective_rate_x) / proportion_y

theorem worker_y_defective_rate : defective_rate_y = 0.008 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_worker_y_defective_rate_l1275_127517


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_equation_holds_l1275_127534

def A : Matrix (Fin 3) (Fin 3) ℚ := !![0, 2, 1; 2, 0, 2; 1, 2, 0]

theorem matrix_equation_holds :
  A^3 + A^2 - 20 • A - 24 • (1 : Matrix (Fin 3) (Fin 3) ℚ) = 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_equation_holds_l1275_127534


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_cosine_l1275_127518

/-- In a triangle ABC, if S + a² = (b + c)², where S is the area of the triangle
    and a, b, c are the sides opposite to angles A, B, C respectively,
    then cos A = -15/17 -/
theorem triangle_cosine (a b c S : ℝ) (h : S + a^2 = (b + c)^2) :
  let A := Real.arccos ((b^2 + c^2 - a^2) / (2 * b * c))
  Real.cos A = -15/17 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_cosine_l1275_127518


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_width_decrease_l1275_127533

theorem rectangle_width_decrease (L W : ℝ) (h : L > 0 ∧ W > 0) : 
  let L' := 1.4 * L
  let W' := W * L / L'
  abs ((1 - W' / W) * 100 - 28.6) < 0.1 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_width_decrease_l1275_127533


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1275_127549

/-- Given a hyperbola with the specified properties, its eccentricity is 3/2 -/
theorem hyperbola_eccentricity (a b : ℝ) (F₁ F₂ P : ℝ × ℝ) (hyperbola : Set (ℝ × ℝ)) : 
  a > 0 → b > 0 →
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1 ↔ (x, y) ∈ hyperbola) →
  F₁ ∈ hyperbola →
  F₂ ∈ hyperbola →
  P ∈ hyperbola →
  ‖F₁ - F₂‖ = 12 →
  ‖P - F₂‖ = 5 →
  (P.2 - F₂.2) * (F₂.1 - F₁.1) = 0 →
  let c := ‖F₁ - F₂‖ / 2
  let e := c / a
  e = 3/2 := by
  sorry

#check hyperbola_eccentricity

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1275_127549


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_objects_per_hour_value_l1275_127528

-- Define variables
def objects_per_hour : ℕ → ℕ := sorry
def start_time : String → ℚ := sorry
def end_time : ℚ := sorry
def total_objects : ℕ := sorry

-- Define axioms
axiom ann_start : start_time "Ann" = 0
axiom bob_start : start_time "Bob" = 1/3
axiom cody_start : start_time "Cody" = 2/3
axiom deb_start : start_time "Deb" = 2/3
axiom end_time_val : end_time = 1
axiom total_objects_val : total_objects = 28

-- Theorem
theorem objects_per_hour_value :
  ∃ n : ℕ, objects_per_hour n = n ∧
  n * (
    (end_time - start_time "Ann") +
    (end_time - start_time "Bob") +
    (end_time - start_time "Cody") +
    (end_time - start_time "Deb")
  ) = total_objects ∧
  n = 12 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_objects_per_hour_value_l1275_127528


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_inequalities_equivalence_l1275_127522

theorem angle_inequalities_equivalence
  (θ₁ θ₂ θ₃ θ₄ : ℝ)
  (h₁ : θ₁ ∈ Set.Ioo (-π/2) (π/2))
  (h₂ : θ₂ ∈ Set.Ioo (-π/2) (π/2))
  (h₃ : θ₃ ∈ Set.Ioo (-π/2) (π/2))
  (h₄ : θ₄ ∈ Set.Ioo (-π/2) (π/2)) :
  (∃ x : ℝ, (Real.cos θ₁)^2 * (Real.cos θ₂)^2 - (Real.sin θ₁ * Real.sin θ₂ - x)^2 ≥ 0 ∧
            (Real.cos θ₃)^2 * (Real.cos θ₄)^2 - (Real.sin θ₃ * Real.sin θ₄ - x)^2 ≥ 0) ↔
  (Real.sin θ₁)^2 + (Real.sin θ₂)^2 + (Real.sin θ₃)^2 + (Real.sin θ₄)^2 ≤
    2 * (1 + Real.sin θ₁ * Real.sin θ₂ * Real.sin θ₃ * Real.sin θ₄ + Real.cos θ₁ * Real.cos θ₂ * Real.cos θ₃ * Real.cos θ₄) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_inequalities_equivalence_l1275_127522


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_means_2_and_8_l1275_127598

theorem geometric_means_2_and_8 :
  ∀ b : ℝ, (b^2 = 2 * 8) ↔ (b = 4 ∨ b = -4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_means_2_and_8_l1275_127598


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1275_127562

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 0 then a^x else (a-3)*x + 4*a

-- State the theorem
theorem range_of_a :
  ∀ a : ℝ, 
  (∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → (f a x₁ - f a x₂) / (x₁ - x₂) < 0) →
  (0 < a ∧ a ≤ 1/4) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1275_127562


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1275_127587

noncomputable def f (x : ℝ) := Real.sqrt (2 * x - x^2)

theorem domain_of_f :
  ∀ x : ℝ, f x ∈ Set.Icc 0 1 ↔ x ∈ Set.Icc 0 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1275_127587


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_increasing_function_inequality_l1275_127524

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

def increasing_on (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
  ∀ {x y}, x ∈ s → y ∈ s → x < y → f x < f y

theorem even_increasing_function_inequality
  (f : ℝ → ℝ) (h_even : is_even f)
  (h_increasing : increasing_on f (Set.Iic 0))
  (a : ℝ) (h_ineq : f a ≤ f 2) :
  a ≤ -2 ∨ a ≥ 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_increasing_function_inequality_l1275_127524


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l1275_127596

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (Real.sin x - 1) * (Real.cos x - 1)

-- State the theorem
theorem f_range : ∀ x : ℝ, 0 ≤ f x ∧ f x ≤ (3 + 2 * Real.sqrt 2) / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l1275_127596


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_real_part_value_l1275_127539

/-- The zeros of z^12 - 3^12 -/
noncomputable def zeros : Finset ℂ := sorry

/-- The possible choices for each w_j -/
noncomputable def w_choices (z : ℂ) : Finset ℂ := {z/2, z, 2*z}

/-- The sum of chosen w_j values -/
noncomputable def w_sum (f : ℂ → ℂ) : ℂ := (zeros.sum f)

/-- The maximum real part of the sum of w_j values -/
noncomputable def max_real_part : ℝ := sorry

theorem max_real_part_value : max_real_part = 9 + 9 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_real_part_value_l1275_127539


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pond_final_state_l1275_127546

/-- Represents the state of the pond -/
structure PondState where
  fish : ℕ
  tadpoles : ℕ
  snails : ℕ

def initial_state : PondState :=
  { fish := 150
  , tadpoles := 5 * 150
  , snails := 200 }

def catch_fish (state : PondState) : PondState :=
  { fish := state.fish - 20
  , tadpoles := state.tadpoles - (3 * state.tadpoles / 5)
  , snails := state.snails - 30 }

def release_animals (state : PondState) : PondState :=
  let released_fish := 20 / 2
  let released_snails := 30 / 2
  let released_tadpoles := released_fish + released_snails
  { fish := state.fish + released_fish
  , tadpoles := state.tadpoles + released_tadpoles
  , snails := state.snails + released_snails }

def develop_tadpoles (state : PondState) : PondState :=
  let developed := (2 * state.tadpoles) / 3
  { fish := state.fish
  , tadpoles := state.tadpoles - developed
  , snails := state.snails }

theorem pond_final_state :
  let final_state := develop_tadpoles (release_animals (catch_fish initial_state))
  final_state.fish - final_state.tadpoles = 31 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pond_final_state_l1275_127546


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_product_of_angle_and_slope_relation_l1275_127509

/-- Theorem: Product of slopes of two lines with specific angle and slope relationships -/
theorem slope_product_of_angle_and_slope_relation (m n : ℝ) : 
  m ≠ 0 →  -- L₁ is not horizontal
  (∃ θ₁ θ₂ : ℝ, θ₁ = 3 * θ₂ ∧ Real.tan θ₁ = m ∧ Real.tan θ₂ = n) →  -- L₁ forms three times the angle with x-axis as L₂
  m = 5 * n →  -- L₁ has 5 times the slope of L₂
  m * n = 5/3 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_product_of_angle_and_slope_relation_l1275_127509


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_expression_l1275_127513

theorem min_value_of_expression (x : ℝ) :
  (25 : ℝ)^x - (5 : ℝ)^x + 2 ≥ 5/4 ∧ ∃ y : ℝ, (25 : ℝ)^y - (5 : ℝ)^y + 2 = 5/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_expression_l1275_127513


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_colors_in_3x4_rectangle_l1275_127564

/-- A coloring of an infinite checkered sheet. -/
def Coloring (n : ℕ) := ℤ → ℤ → Fin n

/-- Checks if a 3x4 rectangle starting at (x,y) contains all colors. -/
def containsAllColors {n : ℕ} (c : Coloring n) (x y : ℤ) : Prop :=
  ∀ k : Fin n, ∃ i j, 0 ≤ i ∧ i < 3 ∧ 0 ≤ j ∧ j < 4 ∧ c (x + i) (y + j) = k

/-- The maximum number of colors that can be used to paint an infinite
    checkered sheet such that every 3x4 rectangle contains all colors is 10. -/
theorem max_colors_in_3x4_rectangle :
  (∃ n : ℕ, n > 0 ∧ ∃ c : Coloring n, ∀ x y : ℤ, containsAllColors c x y) ∧
  (∀ n : ℕ, n > 10 → ¬∃ c : Coloring n, ∀ x y : ℤ, containsAllColors c x y) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_colors_in_3x4_rectangle_l1275_127564


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angles_l1275_127531

structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

def altitude (t : Triangle) (vertex : ℝ × ℝ) (base1 base2 : ℝ × ℝ) : ℝ := 
  sorry

def side_length (a b : ℝ × ℝ) : ℝ := 
  sorry

def angle (t : Triangle) (vertex : ℝ × ℝ) : ℝ := 
  sorry

theorem triangle_angles (t : Triangle) :
  altitude t t.C t.A t.B ≥ side_length t.A t.B →
  altitude t t.A t.B t.C ≥ side_length t.B t.C →
  angle t t.B = 90 ∧ angle t t.A = 45 ∧ angle t t.C = 45 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angles_l1275_127531


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_expression_simplification_logarithm_expression_simplification_l1275_127542

open Real

-- Part 1
theorem trig_expression_simplification (α : ℝ) :
  (Real.sin (π / 2 + α) * Real.cos (π / 2 - α)) / Real.cos (π + α) + 
  (Real.sin (π - α) * Real.cos (π / 2 + α)) / Real.sin (π + α) = 0 := by sorry

-- Part 2
theorem logarithm_expression_simplification :
  (((1 - Real.log 3 / Real.log 6) ^ 2 + Real.log 2 / Real.log 6 * Real.log 18 / Real.log 6) / (Real.log 4 / Real.log 6)) = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_expression_simplification_logarithm_expression_simplification_l1275_127542


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_zero_implies_m_eq_neg_two_l1275_127599

/-- The function f(x) with parameter m -/
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (4 : ℝ)^x + m * (2 : ℝ)^x + 1

/-- Theorem stating that if f has exactly one zero, then m = -2 -/
theorem unique_zero_implies_m_eq_neg_two (m : ℝ) :
  (∃! x, f m x = 0) → m = -2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_zero_implies_m_eq_neg_two_l1275_127599


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ice_cream_orders_l1275_127590

theorem ice_cream_orders (total : ℕ) (vanilla_percent : ℚ) : 
  total = 220 →
  vanilla_percent = 1/5 →
  ∃ (chocolate vanilla : ℕ),
    vanilla = (vanilla_percent * ↑total).floor ∧
    vanilla = 2 * chocolate ∧
    chocolate = 22 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ice_cream_orders_l1275_127590


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_relation_in_triangle_l1275_127550

/-- Given a triangle ABC and a point P in its plane, if BC = 2CP, then AP = -1/2AB + 3/2AC -/
theorem vector_relation_in_triangle (A B C P : EuclideanSpace ℝ (Fin 2)) :
  (C - B) = 2 • (P - C) →
  (P - A) = -(1/2) • (B - A) + (3/2) • (C - A) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_relation_in_triangle_l1275_127550


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_from_projection_l1275_127525

noncomputable def proj (v w : ℝ × ℝ) : ℝ × ℝ :=
  let dot := v.1 * w.1 + v.2 * w.2
  let norm_sq := v.1 * v.1 + v.2 * v.2
  (dot / norm_sq * v.1, dot / norm_sq * v.2)

theorem line_equation_from_projection (x y : ℝ) :
  proj (3, 4) (x, y) = (-9/5, -12/5) →
  y = -3/4 * x - 15/4 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_from_projection_l1275_127525


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_bisector_existence_l1275_127578

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Theorem: Existence of angle bisector equation for the given triangle -/
theorem angle_bisector_existence (A B C : Point) 
  (h1 : A = ⟨2, 3⟩) 
  (h2 : B = ⟨-12, -15⟩) 
  (h3 : C = ⟨6, -5⟩) : 
  ∃ (b d : ℝ), ∀ (x y : ℝ), 
    (distance A B) / (distance A C) = 
      (distance A ⟨x, y⟩) / (distance C ⟨x, y⟩) ↔ 
    b * x + 3 * y + d = 0 := by
  sorry

#check angle_bisector_existence

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_bisector_existence_l1275_127578


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exponent_equality_l1275_127593

theorem exponent_equality (y : ℝ) : (3 : ℝ)^6 = 27^y → y = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exponent_equality_l1275_127593


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_range_l1275_127540

theorem solution_range (a : ℝ) : 
  (∃ x : ℝ, x ∈ Set.Icc 0 Real.pi ∧ (Real.cos x - Real.sin x + a = 0)) ↔ 
  a ∈ Set.Icc (-1) (Real.sqrt 2) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_range_l1275_127540


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_sqrt_sum_equality_l1275_127554

theorem floor_sqrt_sum_equality (n : ℕ+) :
  ⌊Real.sqrt (n : ℝ) + Real.sqrt ((n : ℝ) + 1)⌋ = ⌊Real.sqrt (4 * (n : ℝ) + 2)⌋ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_sqrt_sum_equality_l1275_127554


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hit_time_three_quarters_l1275_127575

/-- The time it takes for an arrow to hit an apple in a specific reference frame. -/
noncomputable def hitTime (L V₀ : ℝ) (α β : ℝ) : ℝ :=
  L / V₀ * Real.sin β / Real.sin (α + β)

/-- The theorem stating that the hit time is 3/4 seconds under specific conditions. -/
theorem hit_time_three_quarters (L V₀ : ℝ) (α β : ℝ) 
    (hL : L > 0) (hV₀ : V₀ > 0) (hα : 0 < α) (hβ : 0 < β) (hαβ : α + β < π) :
  hitTime L V₀ α β = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hit_time_three_quarters_l1275_127575


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_partition_of_positive_rationals_l1275_127547

-- Define Q^+ as the set of all positive rational numbers
def Q_pos : Set ℚ := {q : ℚ | 0 < q}

-- Define the set product operation
def set_product (H K : Set ℚ) : Set ℚ := {q : ℚ | ∃ h k, h ∈ H ∧ k ∈ K ∧ q = h * k}

-- Define the theorem
theorem partition_of_positive_rationals :
  ∃ (A B C : Set ℚ),
    -- A, B, C are subsets of Q^+
    (A ⊆ Q_pos) ∧ (B ⊆ Q_pos) ∧ (C ⊆ Q_pos) ∧
    -- A, B, C are disjoint
    (A ∩ B = ∅) ∧ (B ∩ C = ∅) ∧ (A ∩ C = ∅) ∧
    -- A ∪ B ∪ C = Q^+
    (A ∪ B ∪ C = Q_pos) ∧
    -- BA = B, B^2 = C, BC = A
    (set_product B A = B) ∧ (set_product B B = C) ∧ (set_product B C = A) ∧
    -- All cubes of positive rationals are in A
    (∀ q ∈ Q_pos, q^3 ∈ A) ∧
    -- For all n ≤ 34, n and n+1 are not both in A
    (∀ n : ℕ, n ≤ 34 → (↑n ∈ A → ↑(n+1) ∉ A)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_partition_of_positive_rationals_l1275_127547


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_nonpositive_iff_a_in_zero_two_l1275_127580

-- Define the function f as noncomputable
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log x - x^2 + (1/2) * a

-- State the theorem
theorem f_nonpositive_iff_a_in_zero_two :
  ∀ a : ℝ, (∀ x : ℝ, x > 0 → f a x ≤ 0) ↔ (0 ≤ a ∧ a ≤ 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_nonpositive_iff_a_in_zero_two_l1275_127580


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_duathlon_running_speed_approx_l1275_127552

/-- Represents the duathlon problem with given parameters --/
structure Duathlon where
  bike_distance : ℝ
  run_distance : ℝ
  break_time : ℝ
  total_time : ℝ
  bike_speed : ℝ → ℝ
  run_speed : ℝ → ℝ

/-- Calculates the running speed for the given duathlon parameters --/
noncomputable def calculate_running_speed (d : Duathlon) : ℝ :=
  let x := (d.total_time - d.break_time) * 60 - d.bike_distance / d.bike_speed 0
  d.run_speed x

/-- Theorem stating that the calculated running speed is approximately 7.99 mph --/
theorem duathlon_running_speed_approx (d : Duathlon) 
  (h1 : d.bike_distance = 30)
  (h2 : d.run_distance = 10)
  (h3 : d.break_time = 1/6)
  (h4 : d.total_time = 3)
  (h5 : d.bike_speed = fun x => 3*x + 1)
  (h6 : d.run_speed = fun x => x + 2) :
  abs (calculate_running_speed d - 7.99) < 0.01 := by
  sorry

#eval "Duathlon problem defined and theorem stated."

end NUMINAMATH_CALUDE_ERRORFEEDBACK_duathlon_running_speed_approx_l1275_127552


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fencing_cost_is_91_rupees_l1275_127567

/-- Represents a rectangular field with sides in ratio 3:4 and area 8112 sq. m. -/
structure RectangularField where
  ratio : ℚ
  area : ℝ
  h_ratio : ratio = 3 / 4
  h_area : area = 8112

/-- Calculates the cost of fencing in rupees given the perimeter in meters and cost per meter in paise. -/
noncomputable def fencingCost (perimeter : ℝ) (costPerMeter : ℝ) : ℝ :=
  perimeter * (costPerMeter / 100)

/-- Theorem stating that the fencing cost for the given field at 25 paise per metre is 91 rupees. -/
theorem fencing_cost_is_91_rupees (field : RectangularField) :
    fencingCost (2 * (Real.sqrt (field.area / 3) + Real.sqrt (field.area / 4))) 25 = 91 := by
  sorry

#check fencing_cost_is_91_rupees

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fencing_cost_is_91_rupees_l1275_127567


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_hyperbola_equation_l1275_127520

/-- A hyperbola with specific properties -/
structure SpecialHyperbola where
  /-- The point M(-3,4) is symmetric to one of the asymptotes -/
  symmetric_point : ℝ × ℝ := (-3, 4)
  /-- The point M(-3,4) exactly corresponds to the right focus F₂ -/
  right_focus : ℝ × ℝ := (-3, 4)
  /-- The equation of the hyperbola in standard form -/
  equation : ℝ → ℝ → Prop

/-- The theorem stating the standard equation of the special hyperbola -/
theorem special_hyperbola_equation (h : SpecialHyperbola) :
  h.equation = fun x y => x^2 / 5 - y^2 / 20 = 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_hyperbola_equation_l1275_127520


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sixth_term_is_62_l1275_127566

def mySequence : ℕ → ℕ
  | 0 => 0
  | n + 1 => mySequence n + 2^n

theorem sixth_term_is_62 : mySequence 5 = 62 := by
  sorry

#eval mySequence 5  -- This line will evaluate the 6th term (index 5)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sixth_term_is_62_l1275_127566


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_term_equality_l1275_127573

/-- Represents a geometric sequence with common ratio 2 -/
structure GeometricSequence where
  a : ℕ → ℚ
  ratio : a 2 = 2 * a 1

/-- Sum of first n terms of a geometric sequence -/
def sum_n_terms (g : GeometricSequence) (n : ℕ) : ℚ :=
  (g.a 1) * (1 - 2^n) / (1 - 2)

theorem geometric_sequence_term_equality
  (g : GeometricSequence)
  (n m : ℕ)
  (h_sum : sum_n_terms g n = 255/64)
  (h_term : g.a m = 2) :
  m = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_term_equality_l1275_127573


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alpha_value_l1275_127543

noncomputable def f (α : ℝ) (x : ℝ) : ℝ :=
  if x < 0 then (x^2 + Real.sin x) / (-x^2 + Real.cos (x + α))
  else -x^2 + Real.cos (x + α)

theorem alpha_value (α : ℝ) :
  α ∈ Set.Icc 0 (2 * Real.pi) →
  (∀ x, f α x = -f α (-x)) →
  α = 3 * Real.pi / 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alpha_value_l1275_127543


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shoe_cost_comparison_l1275_127502

-- Define the parameters
noncomputable def used_repair_cost : ℝ := 11.50
noncomputable def used_lifespan : ℝ := 1
noncomputable def new_cost : ℝ := 28.00
noncomputable def new_lifespan : ℝ := 2

-- Define the average cost per year for each option
noncomputable def used_avg_cost : ℝ := used_repair_cost / used_lifespan
noncomputable def new_avg_cost : ℝ := new_cost / new_lifespan

-- Define the percentage increase
noncomputable def percentage_increase : ℝ := (new_avg_cost - used_avg_cost) / used_avg_cost * 100

-- Theorem statement
theorem shoe_cost_comparison :
  abs (percentage_increase - 21.74) < 0.01 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shoe_cost_comparison_l1275_127502


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_sum_with_inverse_l1275_127571

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 2^(x - 3) + x

-- State the theorem
theorem domain_of_sum_with_inverse :
  ∃ (a b : ℝ), a = 4 ∧ b = 5 ∧
  (∀ x : ℝ, x ∈ Set.Icc 3 5 → f x ∈ Set.Icc 4 9) ∧
  (∀ y : ℝ, y ∈ Set.Icc 4 9 → ∃ x : ℝ, x ∈ Set.Icc 3 5 ∧ f x = y) ∧
  (∀ x : ℝ, x ∈ Set.Icc a b ↔ (∃ y : ℝ, y ∈ Set.Icc 3 5 ∧ f y + (Function.invFun f y) = x)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_sum_with_inverse_l1275_127571


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_subset_size_l1275_127551

def is_valid_subset (S : Finset ℕ) : Prop :=
  ∀ x y, x ∈ S → y ∈ S → x ≠ y → (x + y) % 5 = 0

theorem max_subset_size :
  ∃ (S : Finset ℕ), S ⊆ Finset.range 200 ∧ is_valid_subset S ∧ S.card = 40 ∧
  ∀ (T : Finset ℕ), T ⊆ Finset.range 200 → is_valid_subset T → T.card ≤ 40 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_subset_size_l1275_127551


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_time_to_fill_half_cistern_l1275_127511

/-- Represents the time it takes to fill a portion of the cistern -/
def fill_time (portion : ℝ) : ℝ := sorry

/-- The given certain amount of time to fill 1/2 of the cistern -/
def certain_time : ℝ := sorry

/-- Axiom stating that the fill pipe can fill 1/2 of the cistern in the certain amount of time -/
axiom fill_half_cistern : fill_time (1/2) = certain_time

/-- Theorem stating that the time to fill 1/2 of the cistern is equal to the certain amount of time -/
theorem time_to_fill_half_cistern : fill_time (1/2) = certain_time := by
  exact fill_half_cistern


end NUMINAMATH_CALUDE_ERRORFEEDBACK_time_to_fill_half_cistern_l1275_127511


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_increasing_condition_l1275_127521

-- Define the power function as noncomputable
noncomputable def power_function (m : ℝ) (x : ℝ) : ℝ := x^(m-1)

-- State the theorem
theorem power_function_increasing_condition {m : ℝ} :
  (∀ x y, 0 < x ∧ x < y → power_function m x < power_function m y) →
  m > 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_increasing_condition_l1275_127521


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sample_variance_is_two_l1275_127570

noncomputable def sample : List ℝ := [-1, 0, 1, 2, 3]

noncomputable def mean (xs : List ℝ) : ℝ := (xs.sum) / xs.length

noncomputable def variance (xs : List ℝ) : ℝ :=
  let μ := mean xs
  (xs.map (fun x => (x - μ)^2)).sum / xs.length

theorem sample_variance_is_two :
  mean sample = 1 ∧ variance sample = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sample_variance_is_two_l1275_127570


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_condition_l1275_127536

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/2) ^ Real.sqrt (x^2 - 4*a*x + 8)

theorem monotonic_condition (a : ℝ) :
  (∀ x ∈ Set.Icc 2 6, Monotone (fun x => f a x)) ↔ a ≤ 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_condition_l1275_127536


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_one_fifth_25_l1275_127523

theorem log_one_fifth_25 : Real.log 25 / Real.log (1/5) = -2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_one_fifth_25_l1275_127523


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangular_pyramid_volume_specific_pyramid_volume_l1275_127588

/-- A triangular pyramid with mutually perpendicular lateral edges -/
structure TriangularPyramid where
  A : ℝ  -- Area of the first lateral face
  B : ℝ  -- Area of the second lateral face
  C : ℝ  -- Area of the third lateral face
  A_pos : 0 < A
  B_pos : 0 < B
  C_pos : 0 < C

/-- The volume of a triangular pyramid with mutually perpendicular lateral edges -/
noncomputable def volume (p : TriangularPyramid) : ℝ :=
  Real.sqrt (p.A * p.B * p.C) / 3

/-- Theorem: The volume of a triangular pyramid with mutually perpendicular lateral edges
    is equal to the square root of the product of its lateral face areas divided by 3 -/
theorem triangular_pyramid_volume (p : TriangularPyramid) :
  volume p = Real.sqrt (p.A * p.B * p.C) / 3 := by
  sorry

/-- Corollary: For a triangular pyramid with mutually perpendicular lateral edges
    and lateral face areas 1.5, 2, and 6, the volume is 2 -/
theorem specific_pyramid_volume :
  let p : TriangularPyramid := {
    A := 1.5,
    B := 2,
    C := 6,
    A_pos := by norm_num,
    B_pos := by norm_num,
    C_pos := by norm_num
  }
  volume p = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangular_pyramid_volume_specific_pyramid_volume_l1275_127588


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l1275_127541

-- Define the sets A and B
def A : Set ℝ := {x | x > 0}
def B : Set ℝ := {x | x ≤ 1}

-- State the theorem
theorem intersection_of_A_and_B :
  A ∩ B = Set.Ioc 0 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l1275_127541


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dividing_chord_length_l1275_127579

/-- A hexagon inscribed in a circle with specific side lengths -/
structure InscribedHexagon where
  -- The radius of the circumscribed circle
  radius : ℝ
  -- The length of three consecutive sides
  side1 : ℝ
  -- The length of the other three consecutive sides
  side2 : ℝ
  -- Assumption that the hexagon is inscribed in the circle
  inscribed : side1 > 0 ∧ side2 > 0 ∧ radius > (max side1 side2) / 2
  -- Assumption that three consecutive sides have length side1
  consecutive_side1 : Prop
  -- Assumption that three consecutive sides have length side2
  consecutive_side2 : Prop

/-- The chord that divides the hexagon into two trapezoids -/
noncomputable def dividing_chord (h : InscribedHexagon) : ℝ :=
  sorry -- The actual calculation would go here

/-- Theorem stating the length of the dividing chord for a specific hexagon -/
theorem dividing_chord_length :
  let h : InscribedHexagon := {
    radius := sorry,
    side1 := 4,
    side2 := 7,
    inscribed := by sorry,
    consecutive_side1 := sorry,
    consecutive_side2 := sorry
  }
  dividing_chord h = 896 / 121 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dividing_chord_length_l1275_127579


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_vector_sum_theorem_l1275_127535

-- Define the parabola
def is_on_parabola (p : ℝ × ℝ) : Prop := p.2^2 = 4 * p.1

-- Define the focus
def focus : ℝ × ℝ := (1, 0)

-- Define the vector from focus to a point
def vector_from_focus (p : ℝ × ℝ) : ℝ × ℝ := (p.1 - focus.1, p.2 - focus.2)

-- Define the magnitude of a vector
noncomputable def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1^2 + v.2^2)

-- Theorem statement
theorem parabola_vector_sum_theorem (A B C : ℝ × ℝ) 
  (hA : is_on_parabola A) (hB : is_on_parabola B) (hC : is_on_parabola C)
  (h_sum : vector_from_focus A + vector_from_focus B + vector_from_focus C = (0, 0)) :
  magnitude (vector_from_focus A) + magnitude (vector_from_focus B) + magnitude (vector_from_focus C) = 6 := by
  sorry

#check parabola_vector_sum_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_vector_sum_theorem_l1275_127535


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_faster_train_length_l1275_127545

/-- The length of a train given two trains traveling in opposite directions --/
noncomputable def train_length (v1 v2 : ℝ) (t : ℝ) : ℝ :=
  (v1 + v2) * (1000 / 3600) * t

/-- Theorem stating the length of the faster train --/
theorem faster_train_length :
  let v1 : ℝ := 36  -- Speed of slower train in kmph
  let v2 : ℝ := 45  -- Speed of faster train in kmph
  let t : ℝ := 4    -- Time taken to pass in seconds
  train_length v1 v2 t = 90 := by
  -- Unfold the definition of train_length
  unfold train_length
  -- Perform the calculation
  norm_num
  -- The proof is complete
  done


end NUMINAMATH_CALUDE_ERRORFEEDBACK_faster_train_length_l1275_127545


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rational_function_sum_l1275_127507

noncomputable section

-- Define the rational function
noncomputable def p (x : ℝ) : ℝ := x + 1
noncomputable def q (x : ℝ) : ℝ := -4/3 * (x - 1) * (x + 3)

-- Theorem statement
theorem rational_function_sum :
  (∀ x, q x = -4/3 * (x - 1) * (x + 3)) →
  p (-1) = -1 →
  q (-2) = 4 →
  (∀ x, p x + q x = -4/3 * x^2 - 5/3 * x + 6) :=
by
  intro hq hp hq2
  intro x
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rational_function_sum_l1275_127507


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_girls_average_weight_l1275_127544

/-- Given a group of 10 students (5 girls and 5 boys), if the average weight of the boys is 55 kg
    and the average weight of all students is 50 kg, then the average weight of the girls is 45 kg. -/
theorem girls_average_weight (num_girls num_boys : ℕ) (boys_avg all_avg : ℝ) :
  num_girls = 5 →
  num_boys = 5 →
  boys_avg = 55 →
  all_avg = 50 →
  (num_girls + num_boys : ℝ) * all_avg - num_boys * boys_avg = num_girls * 45 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_girls_average_weight_l1275_127544


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_beta_value_l1275_127563

theorem beta_value (α β : ℝ) 
  (h1 : Real.sin α = Real.sqrt 10 / 10)
  (h2 : Real.sin (α - β) = -(Real.sqrt 5 / 5))
  (h3 : 0 < α ∧ α < π/2)
  (h4 : 0 < β ∧ β < π/2) : 
  β = π/4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_beta_value_l1275_127563


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_and_line_equations_l1275_127516

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - 2)^2 + (y - 3)^2 = 8

-- Define the two possible lines l
def line_l1 (x y : ℝ) : Prop := x = 4
def line_l2 (x y : ℝ) : Prop := y = -3/4 * x + 2

-- Define point M
def point_M : ℝ × ℝ := (4, -1)

-- Define the condition that C passes through (1,1)
def C_passes_through_1_1 : Prop := circle_C 1 1

-- Define the condition that C is tangent to y = 3/2 - 2√2
def C_tangent_to_line : Prop := ∃ (x : ℝ), circle_C x (3/2 - 2 * Real.sqrt 2) ∧ 
  (x - 2)*(3/2 - 2*Real.sqrt 2 - 3) + (3/2 - 2*Real.sqrt 2 - 3)*(x - 2) = 0

-- Define the intersection condition
def intersects_C (line : ℝ → ℝ → Prop) : Prop :=
  ∃ (e f : ℝ × ℝ), circle_C e.1 e.2 ∧ circle_C f.1 f.2 ∧ 
    line e.1 e.2 ∧ line f.1 f.2 ∧ e ≠ f

-- Define the perpendicularity condition
def perpendicular_intersection (line : ℝ → ℝ → Prop) : Prop :=
  ∃ (e f : ℝ × ℝ), circle_C e.1 e.2 ∧ circle_C f.1 f.2 ∧ 
    line e.1 e.2 ∧ line f.1 f.2 ∧
    (e.1 - 2) * (f.1 - 2) + (e.2 - 3) * (f.2 - 3) = 0

theorem circle_and_line_equations :
  C_passes_through_1_1 →
  C_tangent_to_line →
  (intersects_C line_l1 ∧ perpendicular_intersection line_l1 ∧ line_l1 point_M.1 point_M.2) ∨
  (intersects_C line_l2 ∧ perpendicular_intersection line_l2 ∧ line_l2 point_M.1 point_M.2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_and_line_equations_l1275_127516


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_problem_l1275_127530

theorem trigonometric_problem (α β : ℝ) 
  (h1 : α ∈ Set.Ioo 0 (π/2))
  (h2 : β - α ∈ Set.Ioo 0 (π/2))
  (h3 : Real.sin α = 3/5)
  (h4 : Real.cos (β - α) = 1/3) :
  Real.sin (β - α) = (2 * Real.sqrt 2) / 3 ∧ 
  Real.sin β = (8 * Real.sqrt 2 + 3) / 15 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_problem_l1275_127530


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_satisfies_conditions_l1275_127501

-- Define the angle α as a function of k
noncomputable def α (k : ℤ) : ℝ := k * Real.pi - Real.pi / 4

-- Define the conditions for the angle
def vertex_at_origin (α : ℝ) : Prop := True  -- Always true as it's given in the problem setup

def initial_side_on_positive_x (α : ℝ) : Prop := 
  ∃ t : ℝ, t > 0 ∧ Real.cos α * t > 0 ∧ Real.sin α * t = 0

def terminal_side_on_y_eq_neg_x (α : ℝ) : Prop := 
  ∃ t : ℝ, t ≠ 0 ∧ Real.cos α * t = -Real.sin α * t

-- The main theorem
theorem angle_satisfies_conditions (k : ℤ) : 
  vertex_at_origin (α k) ∧ 
  initial_side_on_positive_x (α k) ∧ 
  terminal_side_on_y_eq_neg_x (α k) := by
  constructor
  · exact True.intro
  constructor
  · sorry  -- Proof for initial side on positive x-axis
  · sorry  -- Proof for terminal side on y = -x line

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_satisfies_conditions_l1275_127501


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_cos_squared_plus_sin_l1275_127556

theorem max_value_cos_squared_plus_sin :
  ∃ (M : ℝ), M = 5/4 ∧ ∀ x : ℝ, (Real.cos x) ^ 2 + Real.sin x ≤ M :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_cos_squared_plus_sin_l1275_127556


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_squared_beta_plus_pi_fourth_l1275_127548

theorem sin_squared_beta_plus_pi_fourth (β : ℝ) (h : Real.sin (2 * β) = 2 / 3) :
  (Real.sin (β + π / 4)) ^ 2 = 5 / 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_squared_beta_plus_pi_fourth_l1275_127548


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_fifth_root_of_8_power_l1275_127576

-- Define the function we want to minimize
noncomputable def f (n : ℤ) : ℝ := |((8:ℝ)^n)^(1/5) - 100|

-- State the theorem
theorem closest_fifth_root_of_8_power (n : ℤ) : f n ≥ f 11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_fifth_root_of_8_power_l1275_127576


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_sine_circle_points_form_circle_l1275_127569

theorem cosine_sine_circle :
  ∀ (t : ℝ), (Real.cos t)^2 + (Real.sin t)^2 = 1 :=
by sorry

-- The set of points (cos t, sin t) for real t forms a circle with radius 1 centered at the origin
theorem points_form_circle :
  ∀ (x y : ℝ), (∃ (t : ℝ), x = Real.cos t ∧ y = Real.sin t) → x^2 + y^2 = 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_sine_circle_points_form_circle_l1275_127569


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_varies_as_12th_power_of_z_l1275_127583

-- Define the relationships between x, y, and z
def varies_directly (a b : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ ∀ t, a t = k * b t

-- State the theorem
theorem x_varies_as_12th_power_of_z
  (x y z : ℝ → ℝ)
  (h1 : varies_directly x (fun t ↦ (y t)^4))
  (h2 : varies_directly y (fun t ↦ (z t)^3)) :
  varies_directly x (fun t ↦ (z t)^12) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_varies_as_12th_power_of_z_l1275_127583


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_AB_l1275_127586

open Real

-- Define the curves C₁ and C₂
noncomputable def C₁ (θ : ℝ) : ℝ := sqrt (4 / (1 + 3 * sin θ ^ 2))
noncomputable def C₂ (θ : ℝ) : ℝ := 2 * sqrt 7 * sin θ

-- Define the intersection points A and B
noncomputable def A : ℝ := C₁ (π / 6)
noncomputable def B : ℝ := C₂ (π / 6)

-- Theorem statement
theorem length_AB : B - A = 3 * sqrt 7 / 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_AB_l1275_127586


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_propositions_false_l1275_127555

/-- A line in 3D space -/
structure Line3D where
  -- Placeholder for line definition
  dummy : Unit

/-- Two lines are skew if they do not intersect and are not parallel -/
def are_skew (a b : Line3D) : Prop :=
  sorry -- Definition of skew lines to be implemented

/-- A line is perpendicular to another line -/
def is_perpendicular (a b : Line3D) : Prop :=
  sorry -- Definition of perpendicular lines to be implemented

/-- A line intersects another line -/
def intersects (a b : Line3D) : Prop :=
  sorry -- Definition of intersecting lines to be implemented

/-- A line lies in a plane -/
def in_plane (l : Line3D) (p : Plane) : Prop :=
  sorry -- Definition of a line lying in a plane to be implemented

/-- A point lies on a line -/
def point_on_line (p : Point) (l : Line3D) : Prop :=
  sorry -- Definition of a point lying on a line to be implemented

theorem all_propositions_false :
  ¬(∀ (a b : Line3D), are_skew a b → ∃! (c : Line3D), is_perpendicular c a ∧ is_perpendicular c b) ∧
  ¬(∀ (a b : Line3D), ¬(∃ (p : Point), point_on_line p a ∧ point_on_line p b) → are_skew a b) ∧
  ¬(∀ (a b : Line3D) (p : Point), are_skew a b → ∃ (c : Line3D), point_on_line p c ∧ intersects c a ∧ intersects c b) ∧
  ¬(∀ (l : Line3D) (p : Plane), ¬(in_plane l p) → ∀ (m : Line3D), in_plane m p → are_skew l m) ∧
  ¬(∀ (a b : Line3D) (p q : Plane), p ≠ q → in_plane a p → in_plane b q → are_skew a b) ∧
  ¬(∀ (a b c : Line3D), intersects a b → intersects b c → intersects a c) ∧
  ¬(∀ (a b c : Line3D), are_skew a b → are_skew b c → are_skew a c) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_propositions_false_l1275_127555


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jim_taxi_trip_length_l1275_127504

/-- A taxi trip with given fee structure and total charge -/
structure TaxiTrip where
  initial_fee : ℚ
  charge_per_segment : ℚ
  segment_length : ℚ
  total_charge : ℚ

/-- Calculate the length of a taxi trip given its fee structure and total charge -/
noncomputable def trip_length (trip : TaxiTrip) : ℚ :=
  let segments := (trip.total_charge - trip.initial_fee) / trip.charge_per_segment
  segments * trip.segment_length

/-- Theorem stating that for the given fee structure and total charge, the trip length is 3.6 miles -/
theorem jim_taxi_trip_length :
  let trip := TaxiTrip.mk (235/100) (35/100) (2/5) (11/2)
  trip_length trip = 18/5 := by
  sorry

#eval (18:ℚ)/5 -- To show that 18/5 is indeed equal to 3.6

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jim_taxi_trip_length_l1275_127504


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_transformation_l1275_127582

noncomputable def variance (s : Finset ℕ) (f : ℕ → ℝ) : ℝ :=
  let μ := (s.sum f) / s.card
  (s.sum (λ x => (f x - μ)^2)) / s.card

theorem variance_transformation (k : Fin 8 → ℝ) :
  variance (Finset.range 8) (λ i => k i) = 3 →
  variance (Finset.range 8) (λ i => 2 * (k i - 3)) = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_transformation_l1275_127582


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_right_angle_sum_l1275_127572

def circleC (x y : ℝ) : Prop := (x - 3)^2 + (y - 4)^2 = 1

def pointA (m : ℝ) : ℝ × ℝ := (-m, 0)
def pointB (m : ℝ) : ℝ × ℝ := (m, 0)

def rightAngle (A B P : ℝ × ℝ) : Prop :=
  let vAP := (P.1 - A.1, P.2 - A.2)
  let vBP := (P.1 - B.1, P.2 - B.2)
  vAP.1 * vBP.1 + vAP.2 * vBP.2 = 0

theorem circle_right_angle_sum (m : ℝ) :
  m > 0 →
  (∃ P : ℝ × ℝ, circleC P.1 P.2 ∧ rightAngle (pointA m) (pointB m) P) →
  (∃ m_min m_max : ℝ, 
    (∀ m' : ℝ, m' > 0 → (∃ P : ℝ × ℝ, circleC P.1 P.2 ∧ rightAngle (pointA m') (pointB m') P) → m' ≥ m_min) ∧
    (∀ m' : ℝ, m' > 0 → (∃ P : ℝ × ℝ, circleC P.1 P.2 ∧ rightAngle (pointA m') (pointB m') P) → m' ≤ m_max) ∧
    m_min + m_max = 10) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_right_angle_sum_l1275_127572


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inverse_l1275_127592

-- Define the function f
noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := a * Real.log x + b * Real.log x / Real.log 2 + 1

-- State the theorem
theorem f_inverse (a b : ℝ) :
  f a b 2016 = 3 → f a b (1 / 2016) = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inverse_l1275_127592
