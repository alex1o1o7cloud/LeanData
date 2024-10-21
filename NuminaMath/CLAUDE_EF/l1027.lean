import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_line_through_point_l1027_102712

/-- Given a line L1 with equation x - 2y + c = 0, prove that the line L2 with equation 2x + y - 1 = 0
    passes through the point (-1, 3) and is perpendicular to L1. -/
theorem perpendicular_line_through_point :
  ∀ c : ℝ,
  let L1 := λ (x y : ℝ) => x - 2*y + c
  let L2 := λ (x y : ℝ) => 2*x + y - 1
  let point : ℝ × ℝ := (-1, 3)
  (L2 point.1 point.2 = 0) ∧
  ((1 : ℝ) / 2 * -2 = -1) :=
by
  intro c
  simp
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_line_through_point_l1027_102712


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1027_102758

noncomputable def f (x : ℝ) : ℝ := 
  Real.sin (Real.pi / 2 - x) * Real.sin x - Real.sqrt 3 * (Real.cos x)^2 + Real.sqrt 3 / 2

theorem f_properties :
  (∃ (M : ℝ), ∀ (x : ℝ), f x ≤ M ∧ (∃ (x₀ : ℝ), f x₀ = M) ∧ M = 1) ∧
  (∀ (x₁ x₂ : ℝ), 0 < x₁ ∧ x₁ < Real.pi ∧ 0 < x₂ ∧ x₂ < Real.pi →
    f x₁ = 2/3 ∧ f x₂ = 2/3 → Real.cos (x₁ - x₂) = 2/3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1027_102758


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_circle_radius_not_smallest_l1027_102762

-- Define the circle structure
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the problem setup
def circleSetup (c₁ c₂ c₃ c₄ : Circle) : Prop :=
  -- Centers of the first three circles lie on a straight line
  ∃ (t : ℝ), c₁.center.1 < c₂.center.1 ∧ c₂.center.1 < c₃.center.1 ∧
    c₁.center.2 = t ∧ c₂.center.2 = t ∧ c₃.center.2 = t ∧
  -- No shared interior points
  |c₁.center.1 - c₂.center.1| ≥ c₁.radius + c₂.radius ∧
  |c₂.center.1 - c₃.center.1| ≥ c₂.radius + c₃.radius ∧
  -- Fourth circle touches all three circles externally
  (c₄.center.1 - c₁.center.1)^2 + (c₄.center.2 - c₁.center.2)^2 = (c₄.radius + c₁.radius)^2 ∧
  (c₄.center.1 - c₂.center.1)^2 + (c₄.center.2 - c₂.center.2)^2 = (c₄.radius + c₂.radius)^2 ∧
  (c₄.center.1 - c₃.center.1)^2 + (c₄.center.2 - c₃.center.2)^2 = (c₄.radius + c₃.radius)^2

-- State the theorem
theorem fourth_circle_radius_not_smallest (c₁ c₂ c₃ c₄ : Circle) 
  (h : circleSetup c₁ c₂ c₃ c₄) : c₄.radius > c₂.radius := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_circle_radius_not_smallest_l1027_102762


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_railway_connections_theorem_l1027_102789

structure City where
  name : String
  connections : Nat

def moscow : City := ⟨"Moscow", 7⟩
def saintPetersburg : City := ⟨"Saint Petersburg", 5⟩
def tver : City := ⟨"Tver", 4⟩
def yaroslavl : City := ⟨"Yaroslavl", 2⟩
def bologoe : City := ⟨"Bologoe", 2⟩
def shestikhino : City := ⟨"Shestikhino", 2⟩
def zavidovo : City := ⟨"Zavidovo", 1⟩
def vyshniyVolochek : City := ⟨"Vyshniy Volochek", 0⟩  -- connections unknown
def klin : City := ⟨"Klin", 0⟩  -- connections unknown

def cities : List City := [moscow, saintPetersburg, tver, yaroslavl, bologoe, shestikhino, zavidovo, vyshniyVolochek, klin]

theorem railway_connections_theorem :
  ∃ (vv_connections : Nat) (routes : Nat),
    vv_connections ∈ ({2, 3, 4, 5} : Set Nat) ∧
    routes ∈ ({9, 10, 15} : Set Nat) ∧
    vyshniyVolochek.connections = vv_connections ∧
    (∀ c ∈ cities, c.connections ≤ 7) ∧
    (List.sum (cities.map City.connections) % 2 = 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_railway_connections_theorem_l1027_102789


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rich_liar_proof_l1027_102792

-- Define the types of inhabitants
inductive Inhabitant
| Knight
| RichLiar
| PoorLiar

-- Define the statement as a proposition
def statement : Prop := ∃ x, x = Inhabitant.PoorLiar

-- Define the property of being truthful
def isTruthful (i : Inhabitant) : Prop :=
  match i with
  | Inhabitant.Knight => True
  | _ => False

-- Define the property of being a liar
def isLiar (i : Inhabitant) : Prop :=
  match i with
  | Inhabitant.Knight => False
  | _ => True

-- Define the property of being rich
def isRich (i : Inhabitant) : Prop :=
  match i with
  | Inhabitant.RichLiar => True
  | _ => False

-- Theorem: If an inhabitant says "I am a poor liar", they must be a rich liar
theorem rich_liar_proof (speaker : Inhabitant) :
  (isTruthful speaker → ¬statement) ∧
  (isLiar speaker → statement) →
  speaker = Inhabitant.RichLiar := by
  sorry

#check rich_liar_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rich_liar_proof_l1027_102792


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_l1027_102776

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed_kmh : ℝ) (time_sec : ℝ) (length_m : ℝ) : 
  speed_kmh = 288 →
  time_sec = 10 →
  length_m = (speed_kmh * 1000 / 3600) * time_sec →
  length_m = 800 := by
  intros h_speed h_time h_length
  rw [h_speed, h_time] at h_length
  norm_num at h_length
  exact h_length

#check train_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_l1027_102776


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1027_102739

/-- The function f(x) as defined in the problem -/
noncomputable def f (x : ℝ) : ℝ := (Real.cos x + 1 - Real.sin x ^ 2) * Real.tan (x / 2)

/-- Theorem stating the properties of f(x) -/
theorem f_properties :
  (∀ x y, x ∈ Set.Icc (-Real.pi / 4) (Real.pi / 4) → 
          y ∈ Set.Icc (-Real.pi / 4) (Real.pi / 4) → 
          x < y → f x < f y) ∧
  (∃ p, p ∈ Set.Ioo 0 Real.pi ∧ ∀ x, f (x + p) = f x) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1027_102739


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_120_smallest_power_l1027_102773

open Real

/-- The rotation matrix for 120 degrees -/
noncomputable def rotation_120 : Matrix (Fin 2) (Fin 2) ℝ :=
  !![Real.cos (2*π/3), -Real.sin (2*π/3);
     Real.sin (2*π/3),  Real.cos (2*π/3)]

/-- The identity matrix -/
def identity_matrix : Matrix (Fin 2) (Fin 2) ℝ :=
  !![1, 0;
     0, 1]

/-- Predicate to check if a matrix is the identity matrix -/
def is_identity (M : Matrix (Fin 2) (Fin 2) ℝ) : Prop :=
  M = identity_matrix

/-- The smallest positive integer n such that rotation_120^n is the identity matrix -/
def smallest_n : ℕ := 3

theorem rotation_120_smallest_power :
  (∀ k : ℕ, k < smallest_n → ¬is_identity (rotation_120^k)) ∧
  is_identity (rotation_120^smallest_n) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_120_smallest_power_l1027_102773


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_octal_subtraction_theorem_l1027_102719

/-- Represents a number in base 8 -/
structure OctalNumber where
  value : Nat

/-- Converts an octal number to its decimal representation -/
def octal_to_decimal (n : OctalNumber) : Nat :=
  sorry

/-- Converts a decimal number to its octal representation -/
def decimal_to_octal (n : Nat) : OctalNumber :=
  sorry

/-- Subtracts two octal numbers -/
def octal_subtract (a b : OctalNumber) : OctalNumber :=
  decimal_to_octal (octal_to_decimal a - octal_to_decimal b)

instance : OfNat OctalNumber n where
  ofNat := OctalNumber.mk n

theorem octal_subtraction_theorem :
  octal_subtract 541 276 = 243 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_octal_subtraction_theorem_l1027_102719


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_correct_l1027_102786

-- Define the function f
noncomputable def f : ℝ → ℝ := sorry

-- Define the domain of f
def domain : Set ℝ := Set.Icc (-2) 2

-- State that f is even
axiom f_even : ∀ x, x ∈ domain → f x = f (-x)

-- State that f is monotonically increasing on [-2,0]
axiom f_increasing : ∀ x y, x ∈ Set.Icc (-2) 0 → y ∈ Set.Icc (-2) 0 → x ≤ y → f x ≤ f y

-- Define the solution set
def solution_set : Set ℝ := Set.union (Set.Icc (-3) (-2)) (Set.Icc 0 1)

-- Theorem statement
theorem solution_set_correct : 
  ∀ x, x ∈ domain → (f (x + 1) ≤ f (-1) ↔ x ∈ solution_set) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_correct_l1027_102786


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_row_time_ratio_specific_time_ratio_l1027_102724

/-- Proves that the ratio of time taken to row upstream to downstream
    is equal to the ratio of downstream speed to upstream speed. -/
theorem row_time_ratio (v_boat v_stream : ℝ) (h1 : v_boat > v_stream) (h2 : v_stream > 0) :
  (v_boat + v_stream) / (v_boat - v_stream) = (v_boat - v_stream)⁻¹ / (v_boat + v_stream)⁻¹ :=
by sorry

/-- Calculates the ratio of time taken to row upstream to downstream
    given the boat speed in still water and the stream speed. -/
noncomputable def time_ratio (v_boat v_stream : ℝ) : ℝ :=
  (v_boat + v_stream) / (v_boat - v_stream)

/-- Proves that for a boat with speed 39 kmph in still water
    and a stream with speed 13 kmph, the time ratio is 2. -/
theorem specific_time_ratio :
  time_ratio 39 13 = 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_row_time_ratio_specific_time_ratio_l1027_102724


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1027_102727

noncomputable def f (x : ℝ) := 2 - Real.cos (x / 3)

theorem f_properties :
  (∃ (y_max : ℝ), ∀ (x : ℝ), f x ≤ y_max ∧ ∃ (x₀ : ℝ), f x₀ = y_max) ∧
  (∃ (T : ℝ), T > 0 ∧ ∀ (x : ℝ), f (x + T) = f x) ∧
  (∀ (y_max : ℝ), (∀ (x : ℝ), f x ≤ y_max ∧ ∃ (x₀ : ℝ), f x₀ = y_max) → y_max = 3) ∧
  (∀ (T : ℝ), (T > 0 ∧ ∀ (x : ℝ), f (x + T) = f x) → T ≥ 6 * Real.pi) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1027_102727


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_scytale_lines_per_wrap_l1027_102766

/-- Represents the parameters of the Scytale cipher. -/
structure ScytaleParams where
  α : ℝ  -- angle of inclination of dashes to the edge of the tape
  d : ℝ  -- width of the strip
  h : ℝ  -- width of each line
  α_range : 0 < α ∧ α < π / 2
  d_pos : d > 0
  h_pos : h > 0

/-- Calculates the number of lines per complete wrap in the Scytale cipher. -/
noncomputable def linesPerWrap (params : ScytaleParams) : ℝ :=
  params.d / (params.h * Real.cos params.α)

/-- Theorem stating that the number of lines per complete wrap is equal to d / (h * cos(α)). -/
theorem scytale_lines_per_wrap (params : ScytaleParams) :
  linesPerWrap params = params.d / (params.h * Real.cos params.α) := by
  -- Unfold the definition of linesPerWrap
  unfold linesPerWrap
  -- The equation is true by definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_scytale_lines_per_wrap_l1027_102766


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersects_circle_l1027_102703

-- Define the polar coordinates of point A
noncomputable def point_A : ℝ × ℝ := (Real.sqrt 2, Real.pi / 4)

-- Define the polar equation of line l
def line_l (ρ θ : ℝ) : Prop := ρ * Real.cos (θ - Real.pi / 4) = Real.sqrt 2

-- Define that point A lies on line l
axiom A_on_l : line_l point_A.1 point_A.2

-- Define the polar equation of circle C
def circle_C (ρ θ : ℝ) : Prop := ρ = 2 * Real.cos θ

-- Define the rectangular coordinates conversion
noncomputable def polar_to_rect (ρ θ : ℝ) : ℝ × ℝ := (ρ * Real.cos θ, ρ * Real.sin θ)

-- State the theorem
theorem line_intersects_circle : ∃ (x y : ℝ), line_l x y ∧ circle_C x y := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersects_circle_l1027_102703


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_reciprocals_of_roots_l1027_102788

-- Define the polynomial
def p (x : ℝ) : ℝ := 7 * x^2 + 4 * x + 6

-- Define α and β as the reciprocals of the roots of p
noncomputable def α : ℝ := 1 / (Real.sqrt ((4 / 7)^2 - 4 * 6 / 7) - 4 / 7) / 14
noncomputable def β : ℝ := 1 / (-Real.sqrt ((4 / 7)^2 - 4 * 6 / 7) - 4 / 7) / 14

-- State the theorem
theorem sum_of_reciprocals_of_roots : α + β = -2/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_reciprocals_of_roots_l1027_102788


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_k_equals_2_pow_k_l1027_102765

-- Define the sequence of polynomials
noncomputable def f : ℕ → (ℝ → ℝ)
  | 0 => λ x => 1
  | 1 => λ x => 1 + x
  | (k+2) => λ x => ((x+1) * f (k+1) x - (x-(k+1)) * f k x) / (k+2)

-- State the theorem
theorem f_k_equals_2_pow_k (k : ℕ) : f k k = 2^k := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_k_equals_2_pow_k_l1027_102765


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_decreasing_omega_range_l1027_102797

noncomputable def y (ω : ℝ) (x : ℝ) : ℝ := (1/2) * Real.sin (ω * x)

theorem monotonic_decreasing_omega_range (ω : ℝ) :
  (∀ x₁ x₂, x₁ ∈ Set.Icc (-Real.pi/8) (Real.pi/12) → 
            x₂ ∈ Set.Icc (-Real.pi/8) (Real.pi/12) → 
            x₁ < x₂ → y ω x₁ > y ω x₂) →
  ω ∈ Set.Ioc (-4) 0 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_decreasing_omega_range_l1027_102797


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_geometric_sequence_l1027_102709

/-- In a triangle ABC where sides a, b, and c form a geometric sequence and a²-c²=ac-bc, 
    the value of (b*sin(B))/c is equal to √3/2 -/
theorem triangle_geometric_sequence (a b c A B C : ℝ) :
  a > 0 → b > 0 → c > 0 →
  A > 0 → B > 0 → C > 0 →
  A + B + C = Real.pi →
  b^2 = a * c →  -- Geometric sequence condition
  a^2 - c^2 = a * c - b * c →
  (b * Real.sin B) / c = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_geometric_sequence_l1027_102709


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_theater_show_probability_l1027_102750

/-- The probability that at least 9 people stay for the entire show, given:
  - There are 10 people in total
  - 4 people have a 1/3 probability of staying
  - 6 people are certain to stay
-/
theorem theater_show_probability (n : ℕ) (m : ℕ) (p : ℚ) :
  n = 10 →
  m = 4 →
  p = 1/3 →
  (n - m : ℚ) + m * p ≥ 9 ↔
  1/9 = 1/9 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_theater_show_probability_l1027_102750


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tom_apple_slices_l1027_102760

theorem tom_apple_slices (total_apples : ℕ) (slices_per_apple : ℕ) 
  (fraction_to_jerry : ℚ) (fraction_tom_ate : ℚ) :
  total_apples = 5 →
  slices_per_apple = 6 →
  fraction_to_jerry = 7 / 12 →
  fraction_tom_ate = 3 / 5 →
  (let total_slices := total_apples * slices_per_apple
   let slices_to_jerry := Int.floor (fraction_to_jerry * total_slices)
   let slices_after_jerry := total_slices - slices_to_jerry
   let slices_tom_ate := Int.floor (fraction_tom_ate * slices_after_jerry)
   let slices_left := slices_after_jerry - slices_tom_ate
   slices_left = 6) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tom_apple_slices_l1027_102760


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_common_chord_l1027_102752

-- Define the circles and common chord
noncomputable def circle1 (x y : ℝ) : Prop := x^2 + y^2 = 4
noncomputable def circle2 (x y m n : ℝ) : Prop := (x - m)^2 + (y - n)^2 = 4
noncomputable def commonChordLength : ℝ := 2 * Real.sqrt 3

-- Define the theorem
theorem circles_common_chord 
  (m n : ℝ) 
  (h : ∃ (x y : ℝ), circle1 x y ∧ circle2 x y m n) 
  (hLength : ∃ (a b c d : ℝ), 
    circle1 a b ∧ circle1 c d ∧ 
    circle2 a b m n ∧ circle2 c d m n ∧
    (a - c)^2 + (b - d)^2 = commonChordLength^2) :
  (m^2 + n^2 = 4) ∧ 
  (∀ x y, m*x + n*y = 2 ↔ (circle1 x y ∧ circle2 x y m n)) ∧
  (∃ (a b c d : ℝ), 
    circle1 a b ∧ circle1 c d ∧ 
    circle2 a b m n ∧ circle2 c d m n ∧
    ((a - m)^2 + (b - n)^2) * commonChordLength = 8 * Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_common_chord_l1027_102752


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_warriors_height_sorting_l1027_102710

theorem warriors_height_sorting (heights : ℕ → ℝ) :
  ∃ (f : ℕ → ℕ), StrictMono f ∧ Monotone (heights ∘ f) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_warriors_height_sorting_l1027_102710


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_black_car_overtake_time_l1027_102713

/-- The time it takes for a faster car to overtake a slower car -/
noncomputable def overtakeTime (v1 v2 d : ℝ) : ℝ :=
  d / (v2 - v1)

theorem black_car_overtake_time :
  let red_speed : ℝ := 30
  let black_speed : ℝ := 50
  let initial_distance : ℝ := 20
  overtakeTime red_speed black_speed initial_distance = 1 := by
  sorry

-- Remove the #eval line as it's not computable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_black_car_overtake_time_l1027_102713


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_one_equals_four_expression_two_equals_sqrt_two_l1027_102725

open Real

-- Define constants for frequently used angles
noncomputable def angle10 : ℝ := 10 * (π / 180)
noncomputable def angle20 : ℝ := 20 * (π / 180)
noncomputable def angle50 : ℝ := 50 * (π / 180)
noncomputable def angle80 : ℝ := 80 * (π / 180)

-- Theorem for the first expression
theorem expression_one_equals_four :
  (1 / sin angle10) - (sqrt 3 / cos angle10) = 4 := by
  sorry

-- Theorem for the second expression
theorem expression_two_equals_sqrt_two :
  (sin angle50 * (1 + sqrt 3 * tan angle10) - cos angle20) /
  (cos angle80 * sqrt (1 - cos angle20)) = sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_one_equals_four_expression_two_equals_sqrt_two_l1027_102725


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_paint_cans_problem_l1027_102747

/-- Represents the number of rooms that can be painted with one can of paint -/
def rooms_per_can (initial_rooms : ℕ) (final_rooms : ℕ) (lost_cans : ℕ) : ℚ :=
  (initial_rooms - final_rooms) / lost_cans

/-- Calculates the number of cans needed to paint a given number of rooms -/
def cans_needed (rooms : ℕ) (rooms_per_can : ℚ) : ℕ :=
  (rooms : ℚ) / rooms_per_can |> Int.ceil |> Int.toNat

theorem paint_cans_problem (initial_rooms final_rooms lost_cans : ℕ) 
  (h1 : initial_rooms = 45)
  (h2 : final_rooms = 35)
  (h3 : lost_cans = 4) :
  cans_needed final_rooms (rooms_per_can initial_rooms final_rooms lost_cans) = 14 := by
  sorry

#eval cans_needed 35 (rooms_per_can 45 35 4)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_paint_cans_problem_l1027_102747


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_p_always_greater_than_half_X_distribution_valid_l1027_102732

/-- Sequence p_n defined by the recurrence relation -/
def p : ℕ → ℚ
  | 0 => 1
  | n + 1 => 3/8 * p n + 5/16

/-- Theorem stating that p_n is always greater than 1/2 -/
theorem p_always_greater_than_half : ∀ n : ℕ, p n > 1/2 := by
  sorry

/-- Distribution of X (number of days with intelligent testing in first three days) -/
def X_distribution : Finset (ℕ × ℚ) :=
  {(1, 55/256), (2, 5/16), (3, 121/256)}

/-- Theorem stating that X_distribution is a valid probability distribution -/
theorem X_distribution_valid :
  (X_distribution.sum (fun pair => pair.2) = 1) ∧
  (∀ pair ∈ X_distribution, 0 ≤ pair.2 ∧ pair.2 ≤ 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_p_always_greater_than_half_X_distribution_valid_l1027_102732


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_female_fraction_l1027_102777

theorem female_fraction (total_students : ℕ) (non_foreign_male : ℕ) 
  (h1 : total_students = 300)
  (h2 : non_foreign_male = 90) :
  (((total_students : ℚ) - (non_foreign_male * 10 / 9 : ℚ)) / total_students) = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_female_fraction_l1027_102777


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_properties_l1027_102799

/-- Triangle PQR with vertices P(8,10), Q(4,0), and R(9,4) -/
structure Triangle :=
  (P : ℝ × ℝ)
  (Q : ℝ × ℝ)
  (R : ℝ × ℝ)

/-- Point S inside the triangle -/
structure PointS :=
  (x : ℝ)
  (y : ℝ)

/-- Definition of the specific triangle PQR -/
def trianglePQR : Triangle :=
  { P := (8, 10),
    Q := (4, 0),
    R := (9, 4) }

/-- Predicate to check if areas of subtriangles are equal -/
def equalAreasSubtriangles (t : Triangle) (s : PointS) : Prop :=
  sorry  -- Definition of equal areas condition

/-- Main theorem -/
theorem centroid_properties (t : Triangle) (s : PointS) 
    (h : equalAreasSubtriangles t s) : 
    s.x = 7 ∧ s.y = 14/3 ∧ 
    10 * s.x + s.y = 74 + 2/3 ∧
    (let d₁ := Real.sqrt (16 + 100);
     let d₂ := Real.sqrt (1 + 36);
     let d₃ := Real.sqrt (25 + 16);
     d₁ + d₂ + d₃ = Real.sqrt 116 + Real.sqrt 37 + Real.sqrt 41) :=
  by sorry

#check centroid_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_properties_l1027_102799


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_one_sufficient_not_necessary_l1027_102768

/-- The complex number z -/
noncomputable def z (b : ℝ) : ℂ := Complex.mk (Real.sqrt 3) b

/-- The statement that b = 1 is sufficient but not necessary for |z| = 2 -/
theorem b_one_sufficient_not_necessary :
  (∀ b : ℝ, b = 1 → Complex.abs (z b) = 2) ∧
  (∃ b : ℝ, b ≠ 1 ∧ Complex.abs (z b) = 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_one_sufficient_not_necessary_l1027_102768


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_equality_l1027_102702

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if a² = b² + c² - 2bc*sin(A), then A = π/4 -/
theorem triangle_angle_equality (a b c : ℝ) (A B C : ℝ) :
  a > 0 → b > 0 → c > 0 →
  A > 0 → B > 0 → C > 0 →
  A + B + C = π →
  a^2 = b^2 + c^2 - 2*b*c*Real.sin A →
  A = π/4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_equality_l1027_102702


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_append_to_perfect_square_smallest_k_append_to_perfect_square_l1027_102744

open Nat

/-- For any n-digit number, it's possible to append n+1 digits to make it a perfect square. -/
theorem append_to_perfect_square (n : ℕ) :
  ∃ k : ℕ, ∀ m : ℕ, m ≥ 10^(n-1) → m < 10^n → ∃ a : ℕ, (m * 10^(n+1) + a)^2 = m * 10^(n+1) + a ∧ a < 10^(n+1) :=
by
  -- We claim that k = n+1 works
  use n+1
  -- The rest of the proof would go here
  sorry

/-- The smallest k that works for all n-digit numbers is n+1. -/
theorem smallest_k_append_to_perfect_square (n : ℕ) :
  (∀ m : ℕ, m ≥ 10^(n-1) → m < 10^n → ∃ a : ℕ, (m * 10^(n+1) + a)^2 = m * 10^(n+1) + a ∧ a < 10^(n+1)) ∧
  (∀ k : ℕ, k < n+1 → ∃ m : ℕ, m ≥ 10^(n-1) ∧ m < 10^n ∧ ∀ a : ℕ, a < 10^k → (m * 10^k + a)^2 ≠ m * 10^k + a) :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_append_to_perfect_square_smallest_k_append_to_perfect_square_l1027_102744


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_of_i_eq_neg_three_i_over_two_l1027_102728

-- Define the complex number i
noncomputable def i : ℂ := Complex.I

-- Define the function g
noncomputable def g (x : ℂ) : ℂ := (x^5 - x^3 + x) / (x^2 - 1)

-- Theorem statement
theorem g_of_i_eq_neg_three_i_over_two : g i = -3 * i / 2 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_of_i_eq_neg_three_i_over_two_l1027_102728


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_21_game_period_l1027_102720

/-- Represents the cumulative number of games played up to day n -/
def cumulativeGames : ℕ → ℕ := sorry

/-- The grandmaster plays at least one game per day -/
axiom at_least_one_game (n : ℕ) : cumulativeGames (n + 1) - cumulativeGames n ≥ 1

/-- The grandmaster plays no more than ten games per week -/
axiom at_most_ten_games_per_week (n : ℕ) :
  cumulativeGames (n + 7) - cumulativeGames n ≤ 10

/-- There exists a contiguous period of days during which exactly 21 games are played -/
theorem exists_21_game_period :
  ∃ i j : ℕ, i < j ∧ cumulativeGames j - cumulativeGames i = 21 := by
  sorry

#check exists_21_game_period

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_21_game_period_l1027_102720


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l1027_102746

def M : Set ℝ := {x | Real.log x > 0}
def N : Set ℝ := {x | -3 ≤ x ∧ x ≤ 3}

theorem intersection_M_N : M ∩ N = Set.Ioc 1 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l1027_102746


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_equilibrium_temperature_effect_l1027_102731

-- Define the water equilibrium constant
noncomputable def Kw (T : ℝ) : ℝ := sorry

-- Define pH as a function of temperature
noncomputable def pH (T : ℝ) : ℝ := sorry

-- Define the enthalpy change of the reaction
axiom ΔH : ℝ

-- Theorem statement
theorem water_equilibrium_temperature_effect 
  (h : ΔH > 0) :
  ∀ T₁ T₂ : ℝ, T₁ < T₂ → 
    Kw T₁ < Kw T₂ ∧ pH T₁ > pH T₂ :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_equilibrium_temperature_effect_l1027_102731


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_first_four_terms_l1027_102706

noncomputable def sequence_term (n : ℕ) : ℚ := 
  (-1)^(n+1) * (n : ℚ) / ((2*n + 1) * (2*n + 3))

theorem sequence_first_four_terms :
  [sequence_term 1, sequence_term 2, sequence_term 3, sequence_term 4] = 
  [-1 / (3 * 5), 2 / (5 * 7), -3 / (7 * 9), 4 / (9 * 11)] := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_first_four_terms_l1027_102706


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decimal_square_property_l1027_102705

/-- For any natural number n, there exists a natural number m such that
    the decimal representation of m^2 starts with n ones and ends with
    n digits that are either ones or twos. -/
theorem decimal_square_property (n : ℕ) : ∃ m : ℕ,
  (∃ k : ℕ, m^2 = 10^k * (10^n - 1) + r ∧ r < 10^n ∧ r ≥ 10^(n-1)) ∧
  (∃ s : ℕ, m^2 = s * 10^n + t ∧ t < 10^n ∧ ∀ i < n, (t / 10^i) % 10 = 1 ∨ (t / 10^i) % 10 = 2) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_decimal_square_property_l1027_102705


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_sqrt_81_l1027_102738

theorem sqrt_sqrt_81 : {x : ℝ | x^2 = Real.sqrt 81} = {3, -3} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_sqrt_81_l1027_102738


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_sequence_pairs_l1027_102717

/-- A sequence of n integers is valid for given S and d if:
    1. It is non-decreasing
    2. The sum of its elements is S
    3. The difference between the last and first element is d -/
def IsValidSequence (seq : List Int) (S d : Int) : Prop :=
  seq.length > 0 ∧
  seq.Sorted (· ≤ ·) ∧
  seq.sum = S ∧
  (seq.getLast?).getD 0 - seq.head! = d

/-- For given n and d, there exists a unique valid sequence for every integer S -/
def HasUniqueSequenceForAllS (n d : Nat) : Prop :=
  ∀ S : Int, ∃! seq : List Int,
    seq.length = n ∧
    IsValidSequence seq S (Int.ofNat d)

theorem unique_sequence_pairs : 
  ∀ n d : Nat, HasUniqueSequenceForAllS n d ↔ (n = 1 ∧ d = 0) ∨ (n = 3 ∧ d = 2) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_sequence_pairs_l1027_102717


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1027_102749

noncomputable def f (x : ℝ) : ℝ := (Real.sin x + Real.cos x)^2 + 2 * (Real.cos x)^2

theorem f_properties :
  -- Smallest positive period is π
  (∃ (T : ℝ), T > 0 ∧ (∀ (x : ℝ), f (x + T) = f x) ∧
    (∀ (S : ℝ), S > 0 ∧ (∀ (x : ℝ), f (x + S) = f x) → T ≤ S)) ∧
  -- The smallest positive period T equals π
  (let T := Real.pi; T = Real.pi) ∧
  -- Monotonically decreasing on [π/8 + kπ, 5π/8 + kπ] for all k ∈ ℤ
  (∀ (k : ℤ), ∀ (x y : ℝ),
    Real.pi/8 + ↑k * Real.pi ≤ x ∧ x ≤ y ∧ y ≤ 5 * Real.pi/8 + ↑k * Real.pi →
    f y ≤ f x) ∧
  -- f(x) ≥ 3 iff x ∈ [kπ, π/4 + kπ] for some k ∈ ℤ
  (∀ (x : ℝ), f x ≥ 3 ↔
    ∃ (k : ℤ), ↑k * Real.pi ≤ x ∧ x ≤ Real.pi/4 + ↑k * Real.pi) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1027_102749


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_properties_l1027_102704

-- Define the quadratic function f
def f (x : ℝ) : ℝ := -2 * x^2 + 4 * x

-- Define the maximum value function on [t, t+1]
noncomputable def max_value (t : ℝ) : ℝ :=
  if t ≤ 0 then -2 * t^2 + 2
  else if t < 1 then 2
  else -2 * t^2 + 4 * t

theorem quadratic_function_properties :
  (f (-2) = -16) ∧ 
  (f 4 = -16) ∧ 
  (∀ x, f x ≤ 2) ∧
  (∃ x, f x = 2) ∧
  (∀ t, ∀ x ∈ Set.Icc t (t + 1), f x ≤ max_value t) ∧
  (∀ t, ∃ x ∈ Set.Icc t (t + 1), f x = max_value t) := by
  sorry

#check quadratic_function_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_properties_l1027_102704


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dog_area_theorem_l1027_102764

open Real

/-- The area outside a regular hexagonal doghouse that a dog can reach -/
noncomputable def dogAreaOutside (hexagonSide : ℝ) (ropeLength : ℝ) (bushDistance : ℝ) (bushAngle : ℝ) : ℝ :=
  let dogSectorAngle := 2 * Real.pi / 3  -- 240°
  let bushSectorAngle := Real.pi / 3     -- 120°
  (dogSectorAngle / (2 * Real.pi)) * Real.pi * ropeLength^2 - (bushSectorAngle / (2 * Real.pi)) * Real.pi * bushDistance^2

/-- Theorem stating the area the dog can reach outside the doghouse -/
theorem dog_area_theorem (hexagonSide : ℝ) (ropeLength : ℝ) (bushDistance : ℝ) (bushAngle : ℝ) 
  (h1 : hexagonSide = 2)
  (h2 : ropeLength = 4)
  (h3 : bushDistance = 1)
  (h4 : bushAngle = Real.pi / 3) :
  dogAreaOutside hexagonSide ropeLength bushDistance bushAngle = 21 * Real.pi :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_dog_area_theorem_l1027_102764


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_triangle_area_l1027_102740

/-- Parabola type representing y^2 = 4x -/
structure Parabola where
  c : ℝ
  eq : (y : ℝ) → y^2 = 4 * c

/-- Point on a parabola -/
structure PointOnParabola (p : Parabola) where
  x : ℝ
  y : ℝ
  on_parabola : y^2 = 4 * p.c

/-- Focus of a parabola -/
def focus (p : Parabola) : ℝ × ℝ := (p.c, 0)

/-- Area of a triangle given three points -/
noncomputable def triangleArea (p1 p2 p3 : ℝ × ℝ) : ℝ :=
  sorry -- actual implementation not needed for the statement

/-- Theorem statement -/
theorem parabola_triangle_area (p : Parabola) (A B : PointOnParabola p) 
  (h : (A.x + B.x) / 2 = 2 ∧ (A.y + B.y) / 2 = 2) :
  triangleArea (A.x, A.y) (B.x, B.y) (focus p) = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_triangle_area_l1027_102740


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_symmetry_l1027_102795

/-- The function g satisfying g(x) + 3g(1/x) = 4x² for all non-zero real x -/
noncomputable def g : ℝ → ℝ := sorry

/-- The main theorem stating that g(x) = g(-x) for all non-zero real x -/
theorem g_symmetry (x : ℝ) (hx : x ≠ 0) : g x = g (-x) := by
  sorry

/-- The functional equation that g satisfies for all non-zero real x -/
axiom g_equation (x : ℝ) (hx : x ≠ 0) : g x + 3 * g (1/x) = 4 * x^2

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_symmetry_l1027_102795


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_theorem_l1027_102714

/-- Calculates the speed of a train in km/h given its length, platform length, and time to cross -/
noncomputable def train_speed (train_length platform_length : ℝ) (time : ℝ) : ℝ :=
  let total_distance := train_length + platform_length
  let speed_ms := total_distance / time
  3.6 * speed_ms

/-- Theorem: A train with length 240.0416 m crossing a 280 m platform in 26 seconds has a speed of approximately 72 km/h -/
theorem train_speed_theorem :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.1 ∧ 
  |train_speed 240.0416 280 26 - 72| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_theorem_l1027_102714


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_shuffle_restore_iff_prime_l1027_102754

/-- A perfect shuffle on a sequence of 2n cards -/
def perfect_shuffle (n : ℕ) (seq : Fin (2 * n) → ℕ) : Fin (2 * n) → ℕ :=
  λ i => if i.val % 2 = 0 then seq ⟨i.val / 2, by sorry⟩ else seq ⟨n + (i.val - 1) / 2, by sorry⟩

/-- The number of perfect shuffles needed to restore the original order -/
def shuffles_to_restore (n : ℕ) : ℕ := sorry

theorem perfect_shuffle_restore_iff_prime (n : ℕ) :
  (∃ k : ℕ, ∀ seq : Fin (2 * n) → ℕ, (perfect_shuffle n)^[k] seq = seq) ↔ Nat.Prime (2 * n + 1) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_shuffle_restore_iff_prime_l1027_102754


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_between_tangent_and_curve_l1027_102793

-- Define the functions f and g
def f (x : ℝ) : ℝ := x^3 - x^2 + 1
def g (x : ℝ) : ℝ := -x^2

-- Define the tangent line at (1,1)
def tangent_line (x : ℝ) : ℝ := x

-- Theorem statement
theorem area_between_tangent_and_curve : 
  ∫ x in (-1)..(0), (tangent_line x - g x) = 1/6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_between_tangent_and_curve_l1027_102793


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factors_120_multiples_10_l1027_102707

/-- A function that returns the number of positive factors of n that are also multiples of m -/
def countFactorsMultiples (n m : ℕ) : ℕ :=
  (Finset.filter (fun x => x % m = 0) (Nat.divisors n)).card

/-- Theorem stating that the number of positive factors of 120 that are also multiples of 10 is 6 -/
theorem factors_120_multiples_10 : countFactorsMultiples 120 10 = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factors_120_multiples_10_l1027_102707


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_circle_tangent_l1027_102742

-- Define the trapezoid
structure Trapezoid where
  EF : ℝ
  FG : ℝ
  GH : ℝ
  HE : ℝ
  parallel : EF = GH

-- Define the circle
structure TangentCircle (EFGH : Trapezoid) where
  center : ℝ × ℝ
  on_EF : 0 ≤ center.1 ∧ center.1 ≤ EFGH.EF
  tangent_FG : True  -- Placeholder for tangency condition
  tangent_HE : True  -- Placeholder for tangency condition

-- Theorem statement
theorem trapezoid_circle_tangent (EFGH : Trapezoid) (circle : TangentCircle EFGH) :
  EFGH.EF = 75 ∧ EFGH.FG = 50 ∧ EFGH.GH = 24 ∧ EFGH.HE = 70 →
  circle.center.1 = 525 / 11 := by
  sorry

#check trapezoid_circle_tangent

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_circle_tangent_l1027_102742


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_boat_downstream_distance_l1027_102716

/-- Calculates the distance traveled downstream by a boat -/
noncomputable def distance_downstream (boat_speed : ℝ) (current_speed : ℝ) (time_minutes : ℝ) : ℝ :=
  (boat_speed + current_speed) * (time_minutes / 60)

/-- Theorem: A boat with speed 12 km/hr in still water, in a current of 4 km/hr, 
    travels 4.8 km downstream in 18 minutes -/
theorem boat_downstream_distance :
  distance_downstream 12 4 18 = 4.8 := by
  -- Unfold the definition of distance_downstream
  unfold distance_downstream
  -- Perform the calculation
  norm_num
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_boat_downstream_distance_l1027_102716


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1027_102736

/-- The eccentricity of a hyperbola with equation x²/a² - y²/b² = 1 (a > 0, b > 0)
    is √2 if one of its asymptotes is tangent to the circle (x - √2)² + y² = 1 -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ (x y : ℝ), (x / a)^2 - (y / b)^2 = 1) →
  (∃ (x y : ℝ), (b * x + a * y = 0) ∧ (x - Real.sqrt 2)^2 + y^2 = 1) →
  let e := Real.sqrt ((a^2 + b^2) / a^2)
  e = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1027_102736


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_system_solution_correct_l1027_102778

def equation_coefficient (i j : ℕ) : ℕ :=
  if i = 1 then
    if j = 1 then 1 else 2
  else if j = 1 then 1
  else if j = 2 then 3
  else if j ≤ i then 2 * i - 1
  else 2 * i

def solution (i : ℕ) : ℤ :=
  if i % 2 = 1 then -1 else 1

theorem system_solution_correct (n : ℕ) (hn : n = 100) :
  ∀ i ∈ Finset.range n,
    (Finset.sum (Finset.range n) (λ j ↦ (equation_coefficient (i + 1) (j + 1) : ℤ) * solution (j + 1))) = i + 1 :=
by
  sorry

#check system_solution_correct

end NUMINAMATH_CALUDE_ERRORFEEDBACK_system_solution_correct_l1027_102778


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_non_fibonatic_l1027_102753

/-- Fibonacci-like sequence parameterized by a positive integer a -/
def F (a : ℕ+) : ℕ → ℕ
  | 0 => 1
  | 1 => a
  | (n + 2) => F a n + F a (n + 1)

/-- A positive integer is fibonatic if it equals F_n^(a) for some a and n > 3 -/
def IsFibonatic (m : ℕ+) : Prop :=
  ∃ (a : ℕ+) (n : ℕ), n > 3 ∧ (F a n : ℕ) = m

/-- The set of non-fibonatic positive integers is infinite -/
theorem infinitely_many_non_fibonatic :
  Set.Infinite {m : ℕ+ | ¬IsFibonatic m} := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_non_fibonatic_l1027_102753


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_division_remainder_l1027_102774

theorem polynomial_division_remainder :
  ∃ q : Polynomial ℚ, (X : Polynomial ℚ)^4 + 4*(X^2) + 2 = (X - 2)^2 * q + (4*X^2 - 16*X - 30) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_division_remainder_l1027_102774


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pump_fill_time_l1027_102718

/-- The time it takes for a pump to fill a tank without a leak -/
def T : ℝ := sorry

/-- The time it takes to fill the tank with both the pump and leak working -/
def fill_time_with_leak : ℝ := 12

/-- The time it takes for the leak to empty a full tank -/
def leak_empty_time : ℝ := 12

theorem pump_fill_time : T = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pump_fill_time_l1027_102718


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_survivor_dropout_probability_l1027_102785

def total_people : ℕ := 20
def tribe_size : ℕ := 10
def dropouts : ℕ := 3

theorem survivor_dropout_probability :
  (2 * Nat.choose tribe_size dropouts : ℚ) / Nat.choose total_people dropouts = 20 / 95 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_survivor_dropout_probability_l1027_102785


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vectors_lambda_l1027_102767

/-- Given two vectors a and b in ℝ², prove that if k*a + b is perpendicular to a - 2b, then k = -1/7 -/
theorem perpendicular_vectors_lambda (a b : ℝ × ℝ) (k : ℝ) : 
  a = (-3, 2) → 
  b = (-1, 0) → 
  (k * a.1 + b.1, k * a.2 + b.2) • (a.1 - 2 * b.1, a.2 - 2 * b.2) = 0 → 
  k = -1/7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vectors_lambda_l1027_102767


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_projection_impossible_l1027_102755

/-- A rectangular cardboard -/
structure RectangularCardboard where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Represents the orientation of the cardboard relative to the ground and sunlight -/
inductive CardboardOrientation
  | Parallel
  | Perpendicular
  | Inclined

/-- Represents possible shapes of the projection -/
inductive ProjectionShape
  | LineSegment
  | Rectangle
  | Parallelogram
  | Triangle

/-- Function to determine the shape of the projection based on orientation -/
def projectionShape (cardboard : RectangularCardboard) (orientation : CardboardOrientation) : ProjectionShape :=
  match orientation with
  | CardboardOrientation.Parallel => ProjectionShape.LineSegment
  | CardboardOrientation.Perpendicular => ProjectionShape.Rectangle
  | CardboardOrientation.Inclined => ProjectionShape.Parallelogram

/-- Theorem stating that a triangle is impossible as a projection of a rectangular cardboard -/
theorem triangle_projection_impossible (cardboard : RectangularCardboard) :
  ∀ (orientation : CardboardOrientation), projectionShape cardboard orientation ≠ ProjectionShape.Triangle := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_projection_impossible_l1027_102755


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_escalator_time_l1027_102796

theorem escalator_time (stationary_time moving_time : ℝ) :
  stationary_time > 0 →
  moving_time > 0 →
  moving_time < stationary_time →
  (1 / (1 / stationary_time * (stationary_time / moving_time - 1))) = stationary_time := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_escalator_time_l1027_102796


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1027_102733

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then x^2 + 4*x else 4*x - x^2

-- State the theorem
theorem range_of_a (a : ℝ) : f (2*a + 1) > f (a - 2) → a < -3 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1027_102733


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_point_range_l1027_102726

/-- Given a circle C and a point A, proves the range of values for a -/
theorem circle_point_range (a : ℝ) : 
  a > 0 → 
  (∃ M : ℝ × ℝ, 
    (M.1^2 + M.2^2 + 2*M.1 - 4*M.2 + 3 = 0) ∧ 
    ((M.1 - 0)^2 + (M.2 - a)^2 = 2*(M.1^2 + M.2^2))) → 
  (Real.sqrt 3 : ℝ) ≤ a ∧ a ≤ 4 + Real.sqrt 19 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_point_range_l1027_102726


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_sum_l1027_102779

theorem triangle_inequality_sum (a b c : ℝ) 
  (h1 : |a - b| ≥ |c|) 
  (h2 : |b - c| ≥ |a|) 
  (h3 : |c - a| ≥ |b|) : 
  ∃ (x y z : ℝ), (Set.toFinset {x, y, z} = Set.toFinset {a, b, c}) ∧ (x = y + z) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_sum_l1027_102779


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_marathon_run_time_difference_l1027_102737

/-- Represents a runner's journey with a speed change midway -/
structure RunnerJourney where
  totalDistance : ℝ
  halfwayPoint : ℝ
  secondHalfTime : ℝ

/-- Calculates the time difference between the second and first half of the journey -/
noncomputable def timeDifference (journey : RunnerJourney) : ℝ :=
  journey.secondHalfTime - (journey.halfwayPoint / (journey.totalDistance / journey.secondHalfTime))

/-- Theorem stating that for the given conditions, the time difference is 11 hours -/
theorem marathon_run_time_difference :
  ∀ (journey : RunnerJourney),
    journey.totalDistance = 40 ∧
    journey.halfwayPoint = 20 ∧
    journey.secondHalfTime = 22 →
    timeDifference journey = 11 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_marathon_run_time_difference_l1027_102737


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_of_4_equals_9_l1027_102748

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 4 / (3 - x)

-- Define the inverse of f
noncomputable def f_inv (x : ℝ) : ℝ := (3 * x - 4) / x

-- Define the function g
noncomputable def g (x : ℝ) : ℝ := 2 * (1 / f_inv x) + 8

-- Theorem statement
theorem g_of_4_equals_9 : g 4 = 9 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_of_4_equals_9_l1027_102748


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_decreasing_m_value_l1027_102700

/-- A function f is a power function if it can be written as f(x) = ax^n for some constants a and n. -/
def IsPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ a n : ℝ, ∀ x : ℝ, f x = a * x ^ n

/-- A function f is decreasing on an interval (a, b) if for any x1, x2 in (a, b) with x1 < x2, we have f(x1) > f(x2). -/
def IsDecreasingOn (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x1 x2 : ℝ, a < x1 → x1 < x2 → x2 < b → f x1 > f x2

/-- Given that f(x) = (m^2 - m - 1)x^(2m - 3) is a power function and 
    a decreasing function on (0, +∞), prove that m = -1. -/
theorem power_function_decreasing_m_value (m : ℝ) : 
  let f := fun (x : ℝ) => (m^2 - m - 1) * x^(2*m - 3)
  IsPowerFunction f ∧ IsDecreasingOn f 0 (Real.rpow 10 10000) → m = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_decreasing_m_value_l1027_102700


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_february_bill_calculation_l1027_102787

noncomputable def cell_phone_plan (base_cost : ℝ) (text_cost : ℝ) (extra_minute_cost : ℝ) 
                    (included_hours : ℝ) (texts_sent : ℕ) (hours_talked : ℝ) : ℝ :=
  base_cost + 
  (↑texts_sent * text_cost) + 
  (max 0 (hours_talked - included_hours) * 60 * extra_minute_cost)

theorem february_bill_calculation :
  let base_cost : ℝ := 25
  let text_cost : ℝ := 0.1
  let extra_minute_cost : ℝ := 0.1
  let included_hours : ℝ := 50
  let texts_sent : ℕ := 200
  let hours_talked : ℝ := 51
  cell_phone_plan base_cost text_cost extra_minute_cost included_hours texts_sent hours_talked = 51 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_february_bill_calculation_l1027_102787


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_marathon_laps_range_l1027_102734

-- Define the track radius in meters
noncomputable def track_radius : ℝ := 100

-- Define the marathon distance in meters
noncomputable def marathon_distance : ℝ := 42000

-- Define the number of laps
noncomputable def number_of_laps : ℝ := marathon_distance / (2 * Real.pi * track_radius)

-- Theorem statement
theorem marathon_laps_range :
  50 < number_of_laps ∧ number_of_laps < 100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_marathon_laps_range_l1027_102734


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_indefinite_integral_of_20x_squared_l1027_102782

open Real
open MeasureTheory

theorem indefinite_integral_of_20x_squared (x : ℝ) :
  ∃ C : ℝ, ∫ (t : ℝ), 20 * t^2 = (20/3) * x^3 + C :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_indefinite_integral_of_20x_squared_l1027_102782


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_sum_lower_bound_l1027_102756

theorem subset_sum_lower_bound (n : ℕ) (S : ℝ) (nums : Finset ℝ) :
  n > 0 →
  Finset.card nums = 2*n - 1 →
  (∀ x ∈ nums, x > 0) →
  Finset.sum nums id = S →
  (Finset.filter (fun A => A.card = n ∧ Finset.sum A id ≥ S/2) (Finset.powerset nums)).card ≥ Nat.choose (2*n - 2) (n - 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_sum_lower_bound_l1027_102756


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_sides_l1027_102763

/-- Given a rectangle with area 9 cm² and one of the angles formed by the diagonals measuring 120°,
    prove that the sides of the rectangle are ∜27 cm and 3 ∛3 cm. -/
theorem rectangle_sides (A : ℝ) (θ : ℝ) (h_area : A = 9) (h_angle : θ = 120 * π / 180) :
  ∃ (a b : ℝ), a * b = A ∧ 
    a = (27 : ℝ) ^ (1/4) ∧
    b = 3 * (3 : ℝ) ^ (1/4) ∧
    (a ^ 2 + b ^ 2) * (Real.sin θ) ^ 2 = 4 * A ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_sides_l1027_102763


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_area_unit_angle_sector_max_area_l1027_102751

/-- Represents a circular sector --/
structure Sector where
  c : ℝ  -- circumference
  α : ℝ  -- central angle in radians
  r : ℝ  -- radius
  l : ℝ  -- arc length

/-- The circumference of a sector is the sum of twice its radius and its arc length --/
axiom sector_circumference (s : Sector) : 2 * s.r + s.l = s.c

/-- The arc length of a sector is the product of its radius and central angle --/
axiom sector_arc_length (s : Sector) : s.l = s.r * s.α

/-- The area of a sector is half the product of its radius and arc length --/
noncomputable def sector_area (s : Sector) : ℝ := (1/2) * s.r * s.l

/-- Theorem 1: Area of a sector with circumference c and central angle 1 radian --/
theorem sector_area_unit_angle (c : ℝ) :
  ∃ s : Sector, s.c = c ∧ s.α = 1 ∧ sector_area s = c^2 / 18 := by
  sorry

/-- Theorem 2: Maximum area of a sector with circumference c --/
theorem sector_max_area (c : ℝ) :
  ∃ s : Sector, s.c = c ∧ s.α = 2 ∧ 
  sector_area s = c^2 / 16 ∧ 
  ∀ t : Sector, t.c = c → sector_area t ≤ sector_area s := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_area_unit_angle_sector_max_area_l1027_102751


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1027_102730

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x > 1/2 → Real.log (2*x - 1) ≤ x^2 + a) ↔ a ≥ -1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1027_102730


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_possible_x_coordinates_l1027_102715

/-- Definition of a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of a triangle using three points -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Calculate the area of a triangle given its base and height -/
noncomputable def triangleArea (base height : ℝ) : ℝ :=
  (1/2) * base * height

/-- Calculate the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Main theorem -/
theorem sum_of_possible_x_coordinates (ABC ADE : Triangle) 
  (h1 : triangleArea 400 (abs ABC.A.y) = 4000)
  (h2 : ABC.B = ⟨0, 0⟩)
  (h3 : ABC.C = ⟨400, 0⟩)
  (h4 : ADE.A = ⟨1000, 500⟩)
  (h5 : ADE.B = ⟨1010, 515⟩)
  (h6 : triangleArea (distance ADE.A ADE.B) 
    ((|ADE.C.x - 2*ADE.C.y - 850|) / Real.sqrt 5) = 12000) :
  ∃ (x1 x2 x3 x4 : ℝ), x1 + x2 + x3 + x4 = 2960 ∧ 
    (ABC.A.x = x1 ∨ ABC.A.x = x2 ∨ ABC.A.x = x3 ∨ ABC.A.x = x4) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_possible_x_coordinates_l1027_102715


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_union_equality_root_implies_a_and_other_root_l1027_102701

-- Part 1
def M (m : ℝ) : Set ℂ := {2, Complex.mk (m^2 - 2*m) (m^2 + m - 2)}
def P : Set ℂ := {-1, 2, Complex.I * 4}

theorem union_equality (m : ℝ) : M m ∪ P = P → m = 1 ∨ m = 2 := by sorry

-- Part 2
theorem root_implies_a_and_other_root (a : ℝ) :
  (∃ x : ℂ, x^2 + 4*x + a = 0 ∧ x = -2 + Complex.I) →
  a = 5 ∧ (∃ y : ℂ, y^2 + 4*y + a = 0 ∧ y = -2 - Complex.I) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_union_equality_root_implies_a_and_other_root_l1027_102701


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_M_intersect_Z_l1027_102780

def M : Set ℝ := {x : ℝ | x ≤ 1 ∨ x > 3}
def Z : Set ℝ := Int.cast '' Set.univ

theorem complement_M_intersect_Z : 
  (Set.compl M) ∩ Z = {2, 3} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_M_intersect_Z_l1027_102780


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_implies_q_value_l1027_102741

-- Define the complex number i
noncomputable def i : ℂ := Complex.I

-- Define the equation
def equation (p q : ℝ) (x : ℂ) : Prop :=
  x^2 + p • x + q • (1 : ℂ) = 0

-- State the theorem
theorem root_implies_q_value (p q : ℝ) :
  equation p q (2 - i) → q = 5 := by
  intro h
  -- Proof steps would go here
  sorry

-- You can add more lemmas or theorems as needed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_implies_q_value_l1027_102741


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_neo_can_keep_three_digits_l1027_102769

/-- A function that adds 102 to a number -/
def increment (n : ℕ) : ℕ := n + 102

/-- A function that checks if a number is three digits long -/
def isThreeDigits (n : ℕ) : Prop := 100 ≤ n ∧ n < 1000

/-- A function that represents the ability to rearrange digits -/
def canRearrange (a b : ℕ) : Prop := 
  isThreeDigits a ∧ isThreeDigits b ∧ 
  ∃ (x y z : ℕ), a = 100*x + 10*y + z ∧ 
    (b = 100*x + 10*y + z ∨ b = 100*x + 10*z + y ∨ b = 100*y + 10*x + z ∨ 
     b = 100*y + 10*z + x ∨ b = 100*z + 10*x + y ∨ b = 100*z + 10*y + x)

/-- The theorem stating that Neo can keep the number under four digits -/
theorem neo_can_keep_three_digits : 
  ∃ (seq : ℕ → ℕ), 
    seq 0 = 123 ∧ 
    (∀ n, isThreeDigits (seq n)) ∧
    (∀ n, increment (seq n) = seq (n+1) ∨ canRearrange (increment (seq n)) (seq (n+1))) ∧
    (∃ k, k > 0 ∧ seq k = seq 0) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_neo_can_keep_three_digits_l1027_102769


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_length_AB_l1027_102770

/- Define the ellipse Q -/
def ellipse (a : ℝ) (x y : ℝ) : Prop := x^2/a^2 + y^2 = 1

/- Define the foci F₁ and F₂ -/
noncomputable def leftFocus (a : ℝ) : ℝ × ℝ := (-Real.sqrt (a^2 - 1), 0)
noncomputable def rightFocus (a : ℝ) : ℝ × ℝ := (Real.sqrt (a^2 - 1), 0)

/- Define the circle with F₁F₂ as diameter -/
def focusCircle (a : ℝ) (x y : ℝ) : Prop :=
  (x + Real.sqrt (a^2 - 1))^2 + y^2 = a^2

/- Define the line l -/
noncomputable def line (k : ℝ) (x : ℝ) : ℝ := k * (x + Real.sqrt (2 - 1))

/- Define the perpendicular bisector of AB -/
noncomputable def perpBisector (k : ℝ) (x : ℝ) : ℝ :=
  -1/k * (x + 2*k^2/(1 + 2*k^2)) + k/(1 + 2*k^2)

/- Define the x-coordinate of point P -/
noncomputable def xCoordP (k : ℝ) : ℝ := -1/2 + 1/(4*k^2 + 2)

/- Define the length of AB -/
noncomputable def lengthAB (k : ℝ) : ℝ :=
  2 * Real.sqrt 2 * (1/2 + 1/(2*(2*k^2 + 1)))

/- Main theorem -/
theorem min_length_AB (a : ℝ) (k : ℝ) :
  a = Real.sqrt 2 →
  k ≠ 0 →
  0 < k^2 →
  k^2 ≤ 1/2 →
  xCoordP k ∈ Set.Icc (-1/4) 0 →
  lengthAB k ≥ 3 * Real.sqrt 2 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_length_AB_l1027_102770


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_max_correct_answers_l1027_102745

/-- Represents an exam with participants and questions. -/
structure Exam where
  num_participants : ℕ
  num_questions : ℕ
  correct_answers : Fin num_participants → Fin num_questions → Bool

/-- Condition that any two participants have at least one question they both answered correctly. -/
def any_two_share_correct (e : Exam) : Prop :=
  ∀ p1 p2 : Fin e.num_participants, p1 ≠ p2 →
    ∃ q : Fin e.num_questions, e.correct_answers p1 q ∧ e.correct_answers p2 q

/-- The number of people who correctly answered a specific question. -/
def num_correct_for_question (e : Exam) (q : Fin e.num_questions) : ℕ :=
  (Finset.filter (λ p => e.correct_answers p q) (Finset.univ : Finset (Fin e.num_participants))).card

/-- The maximum number of people who correctly answered any question. -/
def max_correct_any_question (e : Exam) : ℕ :=
  Finset.sup (Finset.univ : Finset (Fin e.num_questions)) (num_correct_for_question e)

/-- The main theorem to be proved. -/
theorem min_max_correct_answers (e : Exam) 
    (h1 : e.num_participants = 21)
    (h2 : e.num_questions = 15)
    (h3 : any_two_share_correct e) :
    max_correct_any_question e = 7 := by
  sorry

#check min_max_correct_answers

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_max_correct_answers_l1027_102745


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_of_0_000343_l1027_102722

theorem cube_root_of_0_000343 : Real.rpow 0.000343 (1/3) = 0.07 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_of_0_000343_l1027_102722


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_cost_duration_l1027_102790

/-- Represents the cost of a phone call under Plan A -/
noncomputable def cost_plan_a (duration : ℝ) : ℝ :=
  if duration ≤ 6 then 0.60 else 0.60 + (duration - 6) * 0.06

/-- Represents the cost of a phone call under Plan B -/
def cost_plan_b (duration : ℝ) : ℝ :=
  duration * 0.08

/-- Theorem stating that the duration at which the charges for Plan A and Plan B are equal is 12 minutes -/
theorem equal_cost_duration :
  ∃ (duration : ℝ), duration > 0 ∧ cost_plan_a duration = cost_plan_b duration ∧ duration = 12 := by
  use 12
  constructor
  · exact Real.zero_lt_one.trans (by norm_num)
  constructor
  · simp [cost_plan_a, cost_plan_b]
    norm_num
  · rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_cost_duration_l1027_102790


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_P_n_Q_n_relationship_l1027_102794

def P_n (n : ℕ) (x : ℝ) : ℝ := (1 - x) ^ (2 * n - 1)

def Q_n (n : ℕ) (x : ℝ) : ℝ := 1 - (2 * n - 1) * x + (n - 1) * (2 * n - 1) * x^2

theorem P_n_Q_n_relationship :
  (∀ x, P_n 1 x = Q_n 1 x) ∧
  (P_n 2 0 = Q_n 2 0) ∧
  (∀ x > 0, P_n 2 x < Q_n 2 x) ∧
  (∀ x < 0, P_n 2 x > Q_n 2 x) ∧
  (∀ n ≥ 3, ∀ x > 0, P_n n x < Q_n n x) ∧
  (∀ n ≥ 3, ∀ x < 0, P_n n x > Q_n n x) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_P_n_Q_n_relationship_l1027_102794


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mark_recapture_estimate_l1027_102771

/-- Estimates the number of fish in a pond using the mark-recapture method. -/
theorem mark_recapture_estimate 
  (initially_marked : ℕ) 
  (recaptured_total : ℕ) 
  (recaptured_marked : ℕ) 
  (h1 : initially_marked = 30)
  (h2 : recaptured_total = 50)
  (h3 : recaptured_marked = 2) :
  (initially_marked * recaptured_total : ℚ) / recaptured_marked = 750 := by
  sorry

#check mark_recapture_estimate

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mark_recapture_estimate_l1027_102771


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_subgraph_treewidth_planarity_l1027_102798

/-- A graph is represented as a structure with vertices and edges -/
structure Graph where
  vertices : Type
  edges : Set (vertices × vertices)

/-- A graph H is a subgraph of G if its vertices are a subset of G's vertices and its edges are a subset of G's edges -/
def isSubgraph (H G : Graph) : Prop :=
  ∃ (f : H.vertices → G.vertices), Function.Injective f ∧
    ∀ (e : H.vertices × H.vertices), e ∈ H.edges → (f e.1, f e.2) ∈ G.edges

/-- A graph has bounded treewidth if there exists a constant k such that its treewidth is at most k -/
def hasBoundedTreewidth (G : Graph) : Prop :=
  ∃ k : ℕ, ∃ (treewidth : Graph → ℕ), treewidth G ≤ k

/-- A graph is planar if it can be embedded in the plane without edge crossings -/
def isPlanar (G : Graph) : Prop :=
  ∃ (embedding : G.vertices → ℝ × ℝ),
    ∀ (e₁ e₂ : G.vertices × G.vertices),
      e₁ ∈ G.edges → e₂ ∈ G.edges → e₁ ≠ e₂ →
        ∃ (doNotIntersect : (ℝ × ℝ) × (ℝ × ℝ) → (ℝ × ℝ) × (ℝ × ℝ) → Prop),
          doNotIntersect (embedding e₁.1, embedding e₁.2) (embedding e₂.1, embedding e₂.2)

/-- The main theorem: a graph H has the property that all graphs not containing H as a subgraph have bounded treewidth if and only if H is planar -/
theorem subgraph_treewidth_planarity (H : Graph) :
  (∀ G : Graph, ¬ isSubgraph H G → hasBoundedTreewidth G) ↔ isPlanar H :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_subgraph_treewidth_planarity_l1027_102798


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_green_bay_high_relay_l1027_102783

/-- Given a relay race and a team, calculate the distance each team member needs to run -/
noncomputable def distance_per_member (total_distance : ℝ) (team_size : ℕ) : ℝ :=
  total_distance / team_size

/-- Theorem: In a 425-meter relay race with 9 team members, each member runs approximately 47.22 meters -/
theorem green_bay_high_relay :
  let race_distance : ℝ := 425
  let team_size : ℕ := 9
  let per_member_distance := distance_per_member race_distance team_size
  ∃ ε > 0, |per_member_distance - 47.22| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_green_bay_high_relay_l1027_102783


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_sqrt_3_l1027_102711

-- Define the triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ
  R : ℝ

-- Define the conditions
def is_valid_triangle (t : Triangle) : Prop :=
  t.A > 0 ∧ t.B > 0 ∧ t.C > 0 ∧
  t.a > 0 ∧ t.b > 0 ∧ t.c > 0 ∧
  t.A + t.B + t.C = Real.pi

def satisfies_conditions (t : Triangle) : Prop :=
  is_valid_triangle t ∧
  t.b * Real.cos t.C + t.c * Real.cos t.B = Real.sqrt 3 * t.R ∧
  t.a = 2 ∧
  t.b + t.c = 4

-- Theorem statement
theorem triangle_area_is_sqrt_3 (t : Triangle) (h : satisfies_conditions t) :
  (1/2) * t.a * t.b * Real.sin t.C = Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_sqrt_3_l1027_102711


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_sqrt_sum_l1027_102775

theorem max_value_sqrt_sum (x : ℝ) (h : x ∈ Set.Icc (-36) 36) : 
  Real.sqrt (36 + x) + Real.sqrt (36 - x) ≤ 12 ∧ 
  ∃ y ∈ Set.Icc (-36) 36, Real.sqrt (36 + y) + Real.sqrt (36 - y) = 12 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_sqrt_sum_l1027_102775


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_property_l1027_102761

noncomputable def geometric_sequence (a₁ : ℝ) (r : ℝ) : ℕ → ℝ :=
  λ n => a₁ * r^(n - 1)

noncomputable def sum_geometric (a₁ : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  if r = 1 then n * a₁ else a₁ * (1 - r^n) / (1 - r)

theorem geometric_sequence_sum_property
  (a₁ : ℝ) (r : ℝ) (n : ℕ) (A B C : ℝ)
  (hA : A = sum_geometric a₁ r n)
  (hB : B = sum_geometric a₁ r (2*n))
  (hC : C = sum_geometric a₁ r (3*n))
  (hr : r ≠ 1) :
  A^2 + B^2 = A * (B + C) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_property_l1027_102761


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_compound_weight_is_197_02_l1027_102772

/-- Atomic weight of Aluminum in atomic mass units -/
def Al_weight : ℝ := 26.98

/-- Atomic weight of Oxygen in atomic mass units -/
def O_weight : ℝ := 16.00

/-- Atomic weight of Hydrogen in atomic mass units -/
def H_weight : ℝ := 1.01

/-- Atomic weight of Nitrogen in atomic mass units -/
def N_weight : ℝ := 14.01

/-- Atomic weight of Phosphorus in atomic mass units -/
def P_weight : ℝ := 30.97

/-- Number of Aluminum atoms in the compound -/
def Al_count : ℕ := 2

/-- Number of Oxygen atoms in the compound -/
def O_count : ℕ := 4

/-- Number of Hydrogen atoms in the compound -/
def H_count : ℕ := 6

/-- Number of Nitrogen atoms in the compound -/
def N_count : ℕ := 3

/-- Number of Phosphorus atoms in the compound -/
def P_count : ℕ := 1

/-- The molecular weight of the compound -/
def molecular_weight : ℝ :=
  Al_count * Al_weight + O_count * O_weight + H_count * H_weight +
  N_count * N_weight + P_count * P_weight

theorem compound_weight_is_197_02 :
  abs (molecular_weight - 197.02) < 0.01 := by
  -- The proof is omitted for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_compound_weight_is_197_02_l1027_102772


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_min_value_in_interval_max_value_in_interval_l1027_102721

open Real

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := (cos x)^2 + (cos (x - π/6))^2

-- State the theorems
theorem smallest_positive_period :
  ∃ (T : ℝ), T > 0 ∧ (∀ (x : ℝ), f (x + T) = f x) ∧
  (∀ (T' : ℝ), T' > 0 → (∀ (x : ℝ), f (x + T') = f x) → T ≤ T') ∧
  T = π :=
sorry

theorem min_value_in_interval :
  ∃ (m : ℝ), (∀ (x : ℝ), -π/3 ≤ x ∧ x ≤ π/4 → m ≤ f x) ∧
  (∃ (x : ℝ), -π/3 ≤ x ∧ x ≤ π/4 ∧ f x = m) ∧
  m = 1/4 :=
sorry

theorem max_value_in_interval :
  ∃ (M : ℝ), (∀ (x : ℝ), -π/3 ≤ x ∧ x ≤ π/4 → f x ≤ M) ∧
  (∃ (x : ℝ), -π/3 ≤ x ∧ x ≤ π/4 ∧ f x = M) ∧
  M = (sqrt 3)/2 + 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_min_value_in_interval_max_value_in_interval_l1027_102721


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reciprocal_F_properties_l1027_102708

-- Define the complex number F
noncomputable def F : ℂ := sorry

-- Define the conditions for F
axiom F_in_third_quadrant : F.re < 0 ∧ F.im < 0
axiom F_outside_unit_circle : Complex.abs F > 1

-- Define the reciprocal of F
noncomputable def reciprocal_F : ℂ := F⁻¹

-- Theorem to prove
theorem reciprocal_F_properties :
  reciprocal_F.re < 0 ∧
  reciprocal_F.im > 0 ∧
  Complex.abs reciprocal_F < 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_reciprocal_F_properties_l1027_102708


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_roots_l1027_102729

def IsArithmeticSeq (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_roots (a : ℕ → ℝ) (h : IsArithmeticSeq a) 
  (sum_condition : a 2 + a 5 + a 8 = 9) : 
  ∀ x : ℝ, x^2 + (a 4 + a 6) * x + 10 ≠ 0 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_roots_l1027_102729


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_division_problem_l1027_102791

theorem division_problem (x y : ℕ) (h1 : x % y = 7) (h2 : (x : ℝ) / (y : ℝ) = 86.1) : y = 70 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_division_problem_l1027_102791


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_frustum_original_height_l1027_102723

/-- Represents a frustum of a right circular cone. -/
structure Frustum where
  altitude : ℝ
  lowerBaseArea : ℝ
  upperBaseArea : ℝ
  smallConeHeight : ℝ

/-- Calculates the height of the original cone given a frustum. -/
noncomputable def originalConeHeight (f : Frustum) : ℝ :=
  f.altitude + f.smallConeHeight * (1 + (f.upperBaseArea / f.lowerBaseArea).sqrt)

/-- Theorem stating that for the given frustum, the original cone height is 40 cm. -/
theorem frustum_original_height :
  let f : Frustum := {
    altitude := 30,
    lowerBaseArea := 400 * Real.pi,
    upperBaseArea := 100 * Real.pi,
    smallConeHeight := 10
  }
  originalConeHeight f = 40 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_frustum_original_height_l1027_102723


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_counterfeit_coin_determinable_l1027_102757

/-- Represents a coin with its nominal weight and actual weight -/
structure Coin where
  nominal : Nat
  actual : Nat

/-- Represents the result of a weighing -/
inductive WeighingResult
  | Equal : WeighingResult
  | LeftHeavier : WeighingResult
  | RightHeavier : WeighingResult

/-- Represents a set of four coins -/
def CoinSet := (Coin × Coin × Coin × Coin)

/-- Convert CoinSet to List Coin -/
def CoinSet.toList (cs : CoinSet) : List Coin :=
  [cs.1, cs.2.1, cs.2.2.1, cs.2.2.2]

/-- Simulates a weighing on a balance scale -/
def weigh (left : List Coin) (right : List Coin) : WeighingResult :=
  sorry

/-- Determines if a coin is counterfeit based on two weighings -/
def isCounterfeit (c : Coin) (weighing1 weighing2 : WeighingResult) : Bool :=
  sorry

/-- Theorem stating that the counterfeit coin can be determined in two weighings -/
theorem counterfeit_coin_determinable (coins : CoinSet) :
  (∃! c : Coin, c ∈ coins.toList ∧ c.nominal ≠ c.actual) →
  (∀ c : Coin, c ∈ coins.toList → c.nominal ∈ [1, 2, 3, 5]) →
  ∃ (left1 right1 left2 right2 : List Coin),
    left1 ⊆ coins.toList ∧ right1 ⊆ coins.toList ∧
    left2 ⊆ coins.toList ∧ right2 ⊆ coins.toList ∧
    (∀ c : Coin, c ∈ coins.toList →
      isCounterfeit c (weigh left1 right1) (weigh left2 right2) ↔ c.nominal ≠ c.actual) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_counterfeit_coin_determinable_l1027_102757


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_visitors_previous_day_l1027_102759

def visitors_current_day : ℕ := 317
def visitors_difference : ℕ := 22

theorem visitors_previous_day : visitors_current_day - visitors_difference = 295 := by
  rfl

#eval visitors_current_day - visitors_difference

end NUMINAMATH_CALUDE_ERRORFEEDBACK_visitors_previous_day_l1027_102759


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_calculation_l1027_102743

/-- Represents the profit share ratio between X and Y -/
structure ProfitRatio where
  x : ℚ
  y : ℚ

/-- Calculates the total profit given the profit ratio and the difference between shares -/
def totalProfit (ratio : ProfitRatio) (difference : ℤ) : ℚ :=
  difference * (ratio.x.den * ratio.y.den) / (ratio.x.num * ratio.y.den - ratio.y.num * ratio.x.den)

theorem profit_calculation (ratio : ProfitRatio) (difference : ℤ) :
  ratio.x = 1/2 ∧ ratio.y = 1/3 ∧ difference = 160 →
  totalProfit ratio difference = 800 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_calculation_l1027_102743


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_angle_ratio_l1027_102781

-- Define the curve f(x) = (2/3)x³
noncomputable def f (x : ℝ) : ℝ := (2/3) * x^3

-- Define α as the angle of inclination of the tangent line at (1, f(1))
noncomputable def α : ℝ := Real.arctan 2

-- State the theorem
theorem tangent_angle_ratio :
  (Real.sin α)^2 - (Real.cos α)^2 = (3/5) * (2 * Real.sin α * Real.cos α + (Real.cos α)^2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_angle_ratio_l1027_102781


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_operations_and_subset_implication_l1027_102784

open Set

def A : Set ℝ := {x | 1 ≤ x ∧ x ≤ 3}
def B : Set ℝ := {x | x > 2}
def C (a : ℝ) : Set ℝ := {x | 1 < x ∧ x < a}

theorem set_operations_and_subset_implication :
  (A ∪ B = {x | x > 2 ∨ (1 ≤ x ∧ x ≤ 3)}) ∧
  ((𝒰 \ B) ∩ A = {x | 1 ≤ x ∧ x ≤ 2}) ∧
  (∀ a, C a ⊆ A → a ≤ 3) := by
  sorry

#check set_operations_and_subset_implication

end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_operations_and_subset_implication_l1027_102784


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_matrix_Q_onto_P_l1027_102735

/-- The plane P is defined by the equation x - 3y + 2z = 5 -/
def P : Set (Fin 3 → ℝ) := {v | v 0 - 3 * v 1 + 2 * v 2 = 5}

/-- The projection matrix Q -/
noncomputable def Q : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![13/14, 3/14, -1/7],
    ![3/14, 17/14, 3/7],
    ![-1/7, 3/7, 6/7]]

/-- Theorem: Q is the projection matrix onto plane P -/
theorem projection_matrix_Q_onto_P :
  ∀ v : Fin 3 → ℝ, (Matrix.mulVec Q v) ∈ P ∧
  ∀ w ∈ P, ‖Matrix.mulVec Q v - w‖ ≤ ‖v - w‖ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_matrix_Q_onto_P_l1027_102735
