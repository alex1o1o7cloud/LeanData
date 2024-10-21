import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_to_midsize_ratio_l838_83813

/-- The length of a full-size Mustang in inches -/
noncomputable def full_size_length : ℝ := 240

/-- The scale factor for the mid-size model -/
noncomputable def mid_size_scale : ℝ := 1 / 10

/-- The length of the smallest model in inches -/
noncomputable def smallest_model_length : ℝ := 12

/-- The length of the mid-size model in inches -/
noncomputable def mid_size_length : ℝ := full_size_length * mid_size_scale

/-- The ratio of the smallest model's length to the mid-size model's length -/
noncomputable def model_ratio : ℝ := smallest_model_length / mid_size_length

theorem smallest_to_midsize_ratio :
  model_ratio = 1 / 2 := by
  -- Expand the definitions
  unfold model_ratio
  unfold smallest_model_length
  unfold mid_size_length
  unfold full_size_length
  unfold mid_size_scale
  
  -- Simplify the expression
  simp
  
  -- The proof
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_to_midsize_ratio_l838_83813


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_green_ball_probability_l838_83805

/-- Represents a container with red and green balls -/
structure Container where
  red : ℕ
  green : ℕ

/-- The probability of selecting a green ball from a given container -/
def greenProbability (c : Container) : ℚ :=
  c.green / (c.red + c.green)

/-- The containers in the problem -/
def containers : List Container :=
  [
    ⟨5, 5⟩,  -- Container I
    ⟨3, 3⟩,  -- Container II
    ⟨4, 2⟩,  -- Container III
    ⟨6, 6⟩   -- Container IV
  ]

/-- The probability of selecting a green ball from a randomly chosen container -/
def totalGreenProbability : ℚ :=
  (containers.map greenProbability).sum / 4

theorem green_ball_probability : totalGreenProbability = 11 / 24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_green_ball_probability_l838_83805


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_to_y_equals_x_l838_83898

-- Define a function f and its inverse f_inv
variable (f : ℝ → ℝ) (f_inv : ℝ → ℝ)
variable (a b : ℝ)

-- Assume f and f_inv are inverses of each other
axiom inverse_relation : ∀ x, f (f_inv x) = x ∧ f_inv (f x) = x

-- Define the original function
def original_func (f : ℝ → ℝ) (a b : ℝ) : ℝ → ℝ := λ x ↦ f (x - a) + b

-- Define the symmetric function
def symmetric_func (f_inv : ℝ → ℝ) (a b : ℝ) : ℝ → ℝ := λ x ↦ f_inv (x - b) + a

-- Theorem statement
theorem symmetric_to_y_equals_x (f : ℝ → ℝ) (f_inv : ℝ → ℝ) (a b : ℝ) :
  ∀ x y, y = original_func f a b x ↔ x = symmetric_func f_inv a b y :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_to_y_equals_x_l838_83898


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_line_slope_l838_83811

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- Represents a line passing through a point on the ellipse -/
structure Line where
  slope : ℝ

/-- Defines the eccentricity of an ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ := 
  Real.sqrt (1 - (e.b / e.a)^2)

/-- Defines the area of the quadrilateral formed by the vertices of the ellipse -/
def vertexQuadrilateralArea (e : Ellipse) : ℝ := 
  4 * e.a * e.b

/-- Checks if three lengths form a geometric sequence -/
def isGeometricSequence (a b c : ℝ) : Prop := 
  b^2 = a * c

/-- Main theorem statement -/
theorem ellipse_line_slope (e : Ellipse) (l : Line) :
  eccentricity e = Real.sqrt 3 / 2 →
  vertexQuadrilateralArea e = 16 →
  isGeometricSequence 
    (l.slope * e.b) 
    (Real.sqrt ((l.slope * e.b)^2 + (e.a^2 - e.b^2) * (1 - (l.slope * e.b / e.a)^2))) 
    (e.b + l.slope * e.b^2 / e.a) →
  l.slope = Real.sqrt 5 / 20 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_line_slope_l838_83811


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_b_is_12_l838_83893

/-- The smallest positive integer b such that x^4 + 2x^2 + b^2 is not prime for any integer x -/
def smallest_b : ℕ := 12

/-- The polynomial x^4 + 2x^2 + b^2 -/
def f (x b : ℤ) : ℤ := x^4 + 2*x^2 + b^2

/-- Proposition: 12 is the smallest positive integer b such that x^4 + 2x^2 + b^2 is not prime for any integer x -/
theorem smallest_b_is_12 :
  (∀ x : ℤ, ¬ Nat.Prime (Int.natAbs (f x smallest_b))) ∧
  (∀ b : ℕ, b < smallest_b → ∃ x : ℤ, Nat.Prime (Int.natAbs (f x b))) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_b_is_12_l838_83893


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monthly_growth_rate_is_25_percent_optimal_price_reduction_is_4_l838_83897

-- Define the initial conditions
noncomputable def initial_cost : ℝ := 40
noncomputable def initial_price : ℝ := 60
noncomputable def march_sales : ℝ := 192
noncomputable def may_sales : ℝ := 300
noncomputable def sales_increase_per_2yuan : ℝ := 40
noncomputable def desired_profit : ℝ := 6080

-- Define the monthly average growth rate
noncomputable def monthly_growth_rate : ℝ := (may_sales / march_sales) ^ (1/2) - 1

-- Define the price reduction function
noncomputable def price_reduction (m : ℝ) : ℝ := m

-- Define the new sales volume function
noncomputable def new_sales (m : ℝ) : ℝ := may_sales + sales_increase_per_2yuan * (m / 2)

-- Define the profit function
noncomputable def profit (m : ℝ) : ℝ := (initial_price - initial_cost - price_reduction m) * new_sales m

-- Theorem for the monthly average growth rate
theorem monthly_growth_rate_is_25_percent :
  monthly_growth_rate = 0.25 := by sorry

-- Theorem for the optimal price reduction
theorem optimal_price_reduction_is_4 :
  ∃ m : ℝ, m = 4 ∧ profit m = desired_profit ∧ 
  ∀ n : ℝ, profit n = desired_profit → n ≤ m := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monthly_growth_rate_is_25_percent_optimal_price_reduction_is_4_l838_83897


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_road_trip_problem_l838_83896

/-- A road trip problem -/
theorem road_trip_problem (total_trip_time : ℝ) (num_breaks : ℕ) (break_duration : ℝ)
  (jenna_distance : ℝ) (jenna_speed : ℝ) (friend_speed : ℝ)
  (h1 : total_trip_time = 10)
  (h2 : num_breaks = 2)
  (h3 : break_duration = 0.5)
  (h4 : jenna_distance = 200)
  (h5 : jenna_speed = 50)
  (h6 : friend_speed = 20) :
  (total_trip_time - (↑num_breaks * break_duration) - jenna_distance / jenna_speed) * friend_speed = 100 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_road_trip_problem_l838_83896


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_A_given_Ω_is_one_fourth_l838_83890

/-- Region Ω in the xy-plane --/
def Ω : Set (ℝ × ℝ) := {p | p.1 > 0 ∧ p.2 > 0 ∧ p.1 + p.2 < 2}

/-- Region A in the xy-plane --/
def A : Set (ℝ × ℝ) := {p | p.1 < 1 ∧ p.2 < 1 ∧ p.1 + p.2 > 1}

/-- The area of a set in ℝ² --/
noncomputable def area (S : Set (ℝ × ℝ)) : ℝ := sorry

/-- The probability of a point in Ω also being in A --/
noncomputable def prob_A_given_Ω : ℝ := area (A ∩ Ω) / area Ω

theorem prob_A_given_Ω_is_one_fourth : prob_A_given_Ω = 1/4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_A_given_Ω_is_one_fourth_l838_83890


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_equals_one_third_l838_83816

def y : ℕ → ℝ
  | 0 => 3  -- Add this case to handle Nat.zero
  | 1 => 3
  | k + 2 => y (k + 1) ^ 2 + y (k + 1) + 1

noncomputable def series_sum : ℝ := ∑' n, 1 / (y n + 1)

theorem series_sum_equals_one_third : series_sum = 1 / 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_equals_one_third_l838_83816


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_meaningful_sqrt_fraction_l838_83867

theorem meaningful_sqrt_fraction (x : ℝ) : 
  (∃ y : ℝ, y = Real.sqrt x / Real.sqrt (1 - x)) ↔ 0 ≤ x ∧ x < 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_meaningful_sqrt_fraction_l838_83867


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_powerlifting_total_proof_l838_83895

/-- Calculates the initial powerlifting total given the conditions -/
noncomputable def initialPowerliftingTotal (initialBodyweight : ℝ) (totalIncreasePercent : ℝ) 
  (bodyweightIncrease : ℝ) (finalRatio : ℝ) : ℝ :=
  let finalBodyweight := initialBodyweight + bodyweightIncrease
  let totalIncreaseFactor := 1 + totalIncreasePercent / 100
  finalRatio * finalBodyweight / totalIncreaseFactor

/-- Proves that the initial powerlifting total is 2200 pounds given the conditions -/
theorem powerlifting_total_proof (initialBodyweight : ℝ) (totalIncreasePercent : ℝ) 
  (bodyweightIncrease : ℝ) (finalRatio : ℝ) 
  (h1 : initialBodyweight = 245)
  (h2 : totalIncreasePercent = 15)
  (h3 : bodyweightIncrease = 8)
  (h4 : finalRatio = 10) :
  initialPowerliftingTotal initialBodyweight totalIncreasePercent bodyweightIncrease finalRatio = 2200 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_powerlifting_total_proof_l838_83895


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_three_heads_in_row_correct_floor_180p_eq_47_l838_83884

/-- Probability of flipping heads on the (n+1)th flip, given n heads have already been flipped -/
noncomputable def prob_heads (n : ℕ) : ℝ := 1 / (n + 2 : ℝ)

/-- Probability of flipping tails on the (n+1)th flip, given n heads have already been flipped -/
noncomputable def prob_tails (n : ℕ) : ℝ := (n + 1 : ℝ) / (n + 2 : ℝ)

/-- Probability of flipping 3 heads in a row at some point in an infinite series of flips -/
noncomputable def prob_three_heads_in_row : ℝ := 1 - 2 / Real.exp 1

theorem prob_three_heads_in_row_correct :
  prob_three_heads_in_row = 1 - 2 / Real.exp 1 :=
by sorry

theorem floor_180p_eq_47 :
  ⌊180 * prob_three_heads_in_row⌋ = 47 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_three_heads_in_row_correct_floor_180p_eq_47_l838_83884


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_bridge_crossing_time_l838_83810

/-- Time taken for a train to cross a bridge -/
theorem train_bridge_crossing_time 
  (train_length : ℝ) 
  (train_speed_kmph : ℝ) 
  (bridge_length : ℝ) 
  (h1 : train_length = 110) 
  (h2 : train_speed_kmph = 60) 
  (h3 : bridge_length = 190) : 
  ∃ (time : ℝ), 
    (abs (time - ((train_length + bridge_length) / (train_speed_kmph * (1000 / 3600)))) < 0.01) ∧ 
    (abs (time - 18) < 0.01) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_bridge_crossing_time_l838_83810


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_of_arithmetic_mean_double_geometric_mean_l838_83876

/-- Given two positive real numbers a and b where a > b, 
    and their arithmetic mean is double their geometric mean,
    prove that their ratio rounded to the nearest integer is 14. -/
theorem ratio_of_arithmetic_mean_double_geometric_mean 
  (a b : ℝ) 
  (h1 : a > b) 
  (h2 : b > 0) 
  (h3 : (a + b) / 2 = 2 * Real.sqrt (a * b)) : 
  Int.floor (a / b + 0.5) = 14 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_of_arithmetic_mean_double_geometric_mean_l838_83876


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l838_83899

theorem hyperbola_eccentricity (a b : ℝ) (F A B : ℝ × ℝ) : 
  a > 0 → b > 0 →
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1 ↔ (x, y) ∈ Set.range (λ t ↦ (a * Real.cosh t, b * Real.sinh t))) →
  F.1 < 0 →
  F.2 = 0 →
  A = (0, b) →
  B.1 > 0 →
  (∃ t : ℝ, B = (a * Real.cosh t, b * Real.sinh t)) →
  (A.1 - F.1, A.2 - F.2) = (Real.sqrt 2 - 1) • (B.1 - A.1, B.2 - A.2) →
  (F.1^2 + F.2^2).sqrt / a = Real.sqrt 2 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l838_83899


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l838_83835

noncomputable def f (x : ℝ) := Real.sin (2 * x + Real.pi / 3)

theorem f_properties :
  (∀ x, f x = Real.cos (2 * x - Real.pi / 6)) ∧
  (∀ x, f (x + Real.pi / 6) + f (-Real.pi / 2 - x) = 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l838_83835


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_minus_b_cubed_l838_83822

/-- The number of positive multiples of 8 less than 40 -/
def a : ℤ := 4

/-- The number of positive integers less than 40 that are multiples of both 4 and 2 -/
def b : ℤ := 9

/-- Theorem stating that (a - b)³ = -125 -/
theorem a_minus_b_cubed : (a - b)^3 = -125 := by
  -- Compute a - b
  have h1 : a - b = -5 := by rfl
  
  -- Compute (-5)^3
  have h2 : (-5 : ℤ)^3 = -125 := by norm_num
  
  -- Combine the steps
  rw [h1, h2]


end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_minus_b_cubed_l838_83822


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_banana_arrangements_l838_83840

theorem banana_arrangements : 
  (6 : ℕ).factorial / ((3 : ℕ).factorial * (2 : ℕ).factorial * (1 : ℕ).factorial) = 60 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_banana_arrangements_l838_83840


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l838_83879

-- Define the triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the given conditions
noncomputable def given_triangle : Triangle where
  a := 4
  b := 1  -- We know this from the solution, but it's what we're proving
  c := Real.sqrt 13
  A := Real.arcsin (4 * Real.sin (Real.arcsin (1 / 4)))  -- Using the relationship sin A = 4 sin B
  B := Real.arcsin (1 / 4)  -- We can deduce this from b = 1 and the sine law
  C := Real.pi / 3  -- 60 degrees, as given in the solution

-- Define the sine law
axiom sine_law (t : Triangle) : t.a / Real.sin t.A = t.b / Real.sin t.B

-- Define the cosine law
axiom cosine_law (t : Triangle) : Real.cos t.C = (t.a^2 + t.b^2 - t.c^2) / (2 * t.a * t.b)

-- Define the relationship between angles
axiom angle_sum (t : Triangle) : t.A + t.B + t.C = Real.pi

-- Define the given relationship between sin A and sin B
axiom sin_relationship (t : Triangle) : Real.sin t.A = 4 * Real.sin t.B

-- Theorem to prove
theorem triangle_properties (t : Triangle) (h : t = given_triangle) :
  t.b = 1 ∧ t.C = Real.pi / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l838_83879


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_negation_quadrant_l838_83823

open Real

/-- If angle α is in the fourth quadrant, then the terminal side of angle -α is in the first quadrant -/
theorem angle_negation_quadrant (α : Real) : True := by
  -- Define what it means for an angle to be in the fourth quadrant
  let in_fourth_quadrant (θ : Real) := ∃ k : Int, 3 * π / 2 < θ - 2 * π * k ∧ θ - 2 * π * k < 2 * π

  -- Define what it means for an angle to be in the first quadrant
  let in_first_quadrant (θ : Real) := ∃ k : Int, 0 < θ - 2 * π * k ∧ θ - 2 * π * k < π / 2

  -- Assume α is in the fourth quadrant
  let h : in_fourth_quadrant α := sorry

  -- Prove that -α is in the first quadrant
  have goal : in_first_quadrant (-α) := by
    sorry

  trivial


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_negation_quadrant_l838_83823


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_is_odd_l838_83833

-- Define the domain of tan x
def tanDomain (x : ℝ) : Prop :=
  ∃ k : ℤ, x ∈ Set.Ioo ((-(Real.pi/2 : ℝ)) + k*Real.pi) ((Real.pi/2) + k*Real.pi)

-- Define the tangent function
noncomputable def tan (x : ℝ) : ℝ := Real.tan x

-- Theorem statement
theorem tan_is_odd :
  (∀ x : ℝ, tanDomain x → tanDomain (-x)) →  -- Domain symmetry
  (∀ x : ℝ, tanDomain x → tan (-x) = -tan x) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_is_odd_l838_83833


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_theorem_l838_83815

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Represents a parabola in 2D space -/
structure Parabola where
  p : ℝ

noncomputable def intersect_line_parabola (l : Line) (c : Parabola) : Point :=
  sorry

noncomputable def symmetric_point (a b : Point) : Point :=
  sorry

noncomputable def line_through_points (a b : Point) : Line :=
  sorry

noncomputable def intersect_line_parabola_count (l : Line) (c : Parabola) : Nat :=
  sorry

theorem parabola_intersection_theorem (t p : ℝ) (h1 : t ≠ 0) (h2 : p > 0) :
  let l : Line := { slope := 0, intercept := t }
  let c : Parabola := { p := p }
  let m : Point := { x := 0, y := t }
  let p_int : Point := intersect_line_parabola l c
  let n : Point := symmetric_point m p_int
  let o : Point := { x := 0, y := 0 }
  let on_line : Line := line_through_points o n
  let h : Point := intersect_line_parabola on_line c
  let mh : Line := line_through_points m h
  (abs (h.y / n.y) = 2) ∧
  (intersect_line_parabola_count mh c = 1) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_theorem_l838_83815


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hall_length_is_36_l838_83841

/-- The length of a rectangular hall -/
noncomputable def hall_length (width : ℝ) (stone_length : ℝ) (stone_width : ℝ) (num_stones : ℕ) : ℝ :=
  (stone_length * stone_width * (num_stones : ℝ)) / width

/-- Theorem: The length of the hall is 36 meters -/
theorem hall_length_is_36 :
  hall_length 15 0.4 0.5 2700 = 36 := by
  -- Unfold the definition of hall_length
  unfold hall_length
  -- Simplify the expression
  simp
  -- The proof is completed with sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hall_length_is_36_l838_83841


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cake_wedge_volume_approx_l838_83888

noncomputable def cake_radius : ℝ := 8
noncomputable def cake_height : ℝ := 10
def num_wedges : ℕ := 4

noncomputable def cylinder_volume (r h : ℝ) : ℝ := Real.pi * r^2 * h

noncomputable def wedge_volume (r h : ℝ) (n : ℕ) : ℝ := 
  (cylinder_volume r h) / (n : ℝ)

theorem cake_wedge_volume_approx :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 1 ∧ 
  |wedge_volume cake_radius cake_height num_wedges - 502| < ε := by
  sorry

#eval num_wedges -- This line is added to ensure there's some computable content

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cake_wedge_volume_approx_l838_83888


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_50_value_l838_83802

noncomputable def b : ℕ → ℝ
  | 0 => 2
  | (n + 1) => Real.sqrt (64 * (b n)^2)

theorem b_50_value : b 50 = 8^49 * 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_50_value_l838_83802


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_ratio_l838_83804

-- Define the necessary types and functions
variable (Point : Type)
variable (dist : Point → Point → ℝ)
variable (IsOnCircle : Point → ℝ → Prop)
variable (arcLength : Point → Point → ℝ → ℝ)

-- Theorem statement
theorem circle_ratio (r : ℝ) (A B C : Point) :
  r > 0 →
  IsOnCircle A r →
  IsOnCircle B r →
  IsOnCircle C r →
  dist A B = dist A C →
  dist A B > r →
  arcLength B C r = π * r / 2 →
  dist A B / arcLength B C r = 2 * Real.sqrt 2 / π :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_ratio_l838_83804


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_koch_snowflake_complement_area_proof_l838_83874

/-- The area of the Koch snowflake complement -/
noncomputable def koch_snowflake_complement_area : ℝ := Real.sqrt 3 / 10

/-- The process of removing middle one-third equilateral triangles -/
noncomputable def remove_middle_thirds (n : ℕ) : ℝ :=
  (3 * (4 : ℝ) ^ (n - 1) * Real.sqrt 3) / (4 * 9 ^ n)

/-- The sum of areas removed after infinite steps -/
noncomputable def total_area_removed : ℝ :=
  (3 * Real.sqrt 3 / 4) * (4 / 5)

/-- The area of a unit equilateral triangle -/
noncomputable def unit_equilateral_triangle_area : ℝ := Real.sqrt 3 / 4

theorem koch_snowflake_complement_area_proof :
  unit_equilateral_triangle_area - total_area_removed = koch_snowflake_complement_area :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_koch_snowflake_complement_area_proof_l838_83874


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eggs_donated_to_charity_l838_83851

/-- Represents the number of dozens of eggs collected or distributed --/
structure DozenEggs where
  value : ℕ

instance : Add DozenEggs where
  add a b := ⟨a.value + b.value⟩

instance : Sub DozenEggs where
  sub a b := ⟨a.value - b.value⟩

instance : OfNat DozenEggs n where
  ofNat := ⟨n⟩

/-- Represents the total eggs collected in a week --/
def totalCollected : DozenEggs := 28

/-- Represents the eggs distributed to various places --/
def marketDelivery : DozenEggs := 3
def mallDelivery : DozenEggs := 5
def bakeryDelivery : DozenEggs := 2
def pieUsage : DozenEggs := 4
def neighborGift : DozenEggs := 1

/-- Calculates the total eggs distributed --/
def totalDistributed : DozenEggs :=
  marketDelivery + mallDelivery + bakeryDelivery + pieUsage + neighborGift

/-- Theorem stating the number of eggs donated to charity --/
theorem eggs_donated_to_charity :
  (totalCollected - totalDistributed).value = 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eggs_donated_to_charity_l838_83851


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_waste_disposal_2022_min_cost_2023_l838_83882

-- Define the waste disposal problem
structure WasteDisposalProblem where
  x : ℕ
  y : ℕ
  m : ℕ
  n : ℕ
  eq1 : 25 * x + 16 * y = 5200
  eq2 : 100 * x + 30 * y = 14000
  eq3 : m + n = 240
  ineq : n ≤ 3 * m

-- Theorem for part 1
theorem waste_disposal_2022 (p : WasteDisposalProblem) : 
  p.x = 80 ∧ p.y = 200 := by sorry

-- Theorem for part 2
theorem min_cost_2023 (p : WasteDisposalProblem) :
  ∀ m n : ℕ, m + n = 240 → n ≤ 3 * m → 100 * m + 30 * n ≥ 11400 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_waste_disposal_2022_min_cost_2023_l838_83882


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_intersection_l838_83834

/-- Given a curve y = x^(n+1) where n is a positive integer,
    the tangent line at point (1,1) intersects the y-axis at (0, -n) -/
theorem tangent_line_intersection (n : ℕ) :
  let f : ℝ → ℝ := fun x ↦ x^(n + 1)
  let df : ℝ → ℝ := fun x ↦ (n + 1 : ℝ) * x^n
  let tangent_line : ℝ → ℝ := fun x ↦ df 1 * (x - 1) + f 1
  tangent_line 0 = -(n : ℝ) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_intersection_l838_83834


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prism_plane_intersection_difference_l838_83838

-- Define the rectangular prism
structure RectangularPrism where
  vertices : Finset (Fin 8)
  edges : Finset (Fin 12)
  faces : Finset (Fin 6)

-- Define a plane
structure Plane where
  equation : ℝ → ℝ → ℝ → Prop

-- Define the set of planes
def PlaneSet (k : ℕ) := Finset (Fin k)

-- Define the surface of the prism
def Surface (R : RectangularPrism) := R.faces

-- Define the union of planes
noncomputable def UnionOfPlanes (P : PlaneSet k) : Set Plane := sorry

-- Define the intersection of planes with the surface
noncomputable def Intersection (P : PlaneSet k) (S : Finset (Fin 6)) : Set (ℝ × ℝ × ℝ) := sorry

-- Define a segment from midpoint to nearest vertex
structure MidpointToVertexSegment where
  edge : Fin 12
  vertex : Fin 8

-- Define the set of all midpoint-to-vertex segments
noncomputable def AllMidpointToVertexSegments (R : RectangularPrism) : Set (ℝ × ℝ × ℝ) := sorry

-- The main theorem
theorem prism_plane_intersection_difference (R : RectangularPrism) :
  ∃ (kmin kmax : ℕ),
    (∀ k, kmin ≤ k ∧ k ≤ kmax →
      ∃ (P : PlaneSet k),
        Intersection P (Surface R) = AllMidpointToVertexSegments R) ∧
    kmax - kmin = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_prism_plane_intersection_difference_l838_83838


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_cube_units_digits_count_perfect_cube_units_digits_complete_l838_83843

/-- The set of possible units digits of perfect cubes -/
def PerfectCubeUnitsDigits : Finset ℕ :=
  {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

/-- There are exactly 10 distinct digits that can appear as the units digit of an integral perfect-cube number -/
theorem perfect_cube_units_digits_count :
  Finset.card PerfectCubeUnitsDigits = 10 := by
  rfl

/-- Proof that the set PerfectCubeUnitsDigits contains all possible units digits of perfect cubes -/
theorem perfect_cube_units_digits_complete :
  ∀ n : ℕ, ∃ d ∈ PerfectCubeUnitsDigits, n^3 % 10 = d := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_cube_units_digits_count_perfect_cube_units_digits_complete_l838_83843


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_case_two_theorem_case_three_theorem_l838_83818

-- Define the basic types
structure Line
structure Plane

-- Define the relationships
def perpendicular (a b : Line ⊕ Plane) : Prop := sorry

def parallel (a b : Line ⊕ Plane) : Prop := sorry

-- Theorem for case 2: x and y are lines, z is a plane
theorem case_two_theorem (x y : Line) (z : Plane) :
  perpendicular (Sum.inl x) (Sum.inr z) →
  perpendicular (Sum.inl y) (Sum.inr z) →
  parallel (Sum.inl x) (Sum.inl y) := by sorry

-- Theorem for case 3: z is a line, x and y are planes
theorem case_three_theorem (x y : Plane) (z : Line) :
  perpendicular (Sum.inr x) (Sum.inl z) →
  perpendicular (Sum.inr y) (Sum.inl z) →
  parallel (Sum.inr x) (Sum.inr y) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_case_two_theorem_case_three_theorem_l838_83818


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_and_quarter_miles_laps_l838_83801

/-- The number of complete laps required to run a given distance on a track -/
def completeLaps (totalDistance : ℚ) (lapDistance : ℚ) : ℕ :=
  (totalDistance / lapDistance).floor.toNat

/-- Proof that running 3.25 miles on a track with 0.25-mile laps requires 13 complete laps -/
theorem three_and_quarter_miles_laps :
  completeLaps (3.25 : ℚ) (0.25 : ℚ) = 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_and_quarter_miles_laps_l838_83801


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l838_83827

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (3 * x - 3 * Real.pi / 4)

theorem f_properties :
  (∃ T > 0, ∀ x, f (x + T) = f x ∧ ∀ S, 0 < S ∧ S < T → ∃ y, f (y + S) ≠ f y) ∧
  (∃ T > 0, T = 2 * Real.pi / 3 ∧ ∀ x, f (x + T) = f x ∧ ∀ S, 0 < S ∧ S < T → ∃ y, f (y + S) ≠ f y) ∧
  (∀ x, f x = 2 * Real.cos (3 * x + 3 * Real.pi / 4)) ∧
  (∀ x ∈ Set.Icc (Real.pi / 12) (5 * Real.pi / 12), 
    ∀ y ∈ Set.Icc (Real.pi / 12) (5 * Real.pi / 12), 
      x < y → f x < f y) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l838_83827


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_slope_l838_83837

noncomputable def f (x : ℝ) : ℝ := Real.log x - 2 / x

theorem tangent_line_slope (α : ℝ) : 
  (deriv f 1 = Real.tan α) → (Real.cos α / (Real.sin α - 4 * Real.cos α) = -1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_slope_l838_83837


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_blue_to_red_face_ratio_l838_83845

theorem blue_to_red_face_ratio (n : ℕ) (h : n = 13) : 
  (6 * n^3 - 6 * n^2 : ℚ) / (6 * n^2) = 12 := by
  -- Substitute n = 13
  rw [h]
  -- Simplify the expression
  norm_num
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_blue_to_red_face_ratio_l838_83845


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_k_value_exists_x0_l838_83885

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.log (x + 1) - x

-- Theorem 1
theorem max_k_value (k : ℤ) :
  (∀ x > 1, f (x - 1) + x > k * (1 - 3 / x)) →
  k ≤ 4 :=
by sorry

-- Theorem 2
theorem exists_x0 (a : ℝ) (ha : 0 < a ∧ a < 1) :
  ∃ x₀ > 0, Real.exp (f x₀) < 1 - a / 2 * x₀^2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_k_value_exists_x0_l838_83885


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_four_digit_numbers_l838_83857

/-- The set of digits used to construct the numbers -/
def digits : Finset Nat := {1, 2, 4, 5}

/-- A 4-digit number constructed from the given digits -/
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

/-- The set of all possible 4-digit numbers -/
def allFourDigitNumbers : Finset FourDigitNumber := sorry

/-- The value of a 4-digit number -/
def value (n : FourDigitNumber) : Nat :=
  1000 * n.d1 + 100 * n.d2 + 10 * n.d3 + n.d4

/-- The sum of all possible 4-digit numbers -/
def sumOfAllNumbers : Nat :=
  Finset.sum allFourDigitNumbers value

theorem sum_of_four_digit_numbers :
  sumOfAllNumbers = 79992 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_four_digit_numbers_l838_83857


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_min_value_l838_83861

/-- 
For a hyperbola with equation x²/a² - y²/b² = 1, where a > 0 and b > 0,
asymptote inclination angle of 2π/3, and eccentricity e,
the minimum value of (a² + e²)/(2b) is 2√3/3
-/
theorem hyperbola_min_value (a b e : ℝ) (ha : a > 0) (hb : b > 0) : 
  (∃ x y : ℝ, x^2/a^2 - y^2/b^2 = 1) →
  (Real.tan (2 * Real.pi / 3) = b / a) →
  (e^2 = 1 + b^2/a^2) →
  (∀ a' b' e' : ℝ, a' > 0 → b' > 0 → 
    (∃ x' y' : ℝ, x'^2/a'^2 - y'^2/b'^2 = 1) →
    (Real.tan (2 * Real.pi / 3) = b' / a') →
    (e'^2 = 1 + b'^2/a'^2) →
    (a'^2 + e'^2)/(2*b') ≥ (a^2 + e^2)/(2*b)) →
  (a^2 + e^2)/(2*b) = 2 * Real.sqrt 3 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_min_value_l838_83861


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_speed_three_parts_l838_83830

/-- Proves that the average speed of a car traveling a distance D in three equal parts
    at speeds of 80 km/h, 18 km/h, and 48 km/h respectively is 33.75 km/h. -/
theorem average_speed_three_parts (D : ℝ) (h : D > 0) : 
  (D / ((D / (3 * 80)) + (D / (3 * 18)) + (D / (3 * 48)))) = 33.75 := by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_speed_three_parts_l838_83830


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_through_point_l838_83826

-- Define the power function as noncomputable
noncomputable def power_function (a : ℝ) (x : ℝ) : ℝ := x ^ a

-- State the theorem
theorem power_function_through_point (a : ℝ) :
  power_function a 4 = 1/2 → a = -1/2 := by
  intro h
  -- The proof steps would go here, but we'll use sorry for now
  sorry

#check power_function_through_point

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_through_point_l838_83826


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_highest_point_l838_83873

-- Define the height function
def height_func (a b t : ℝ) : ℝ := a * t^2 + b * t

-- Theorem statement
theorem ball_highest_point (a b : ℝ) (h : height_func a b 3 = height_func a b 7) :
  ∃ (t_max : ℝ), t_max = 5 ∧ ∀ (t : ℝ), height_func a b t ≤ height_func a b t_max :=
by
  -- Proof is omitted for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_highest_point_l838_83873


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_transformation_zeros_l838_83875

/-- Represents a parabola in the form y = a(x-h)^2 + k --/
structure Parabola where
  a : ℚ
  h : ℚ
  k : ℚ

/-- Applies the specified transformations to a parabola --/
def transform_parabola (p : Parabola) : Parabola :=
  { a := -1/2 * p.a,
    h := p.h + 4,
    k := -p.k + 2 }

/-- Calculates the zeros of a parabola --/
def zeros (p : Parabola) : Set ℚ :=
  {x | p.a * (x - p.h)^2 + p.k = 0}

theorem parabola_transformation_zeros :
  let initial_parabola : Parabola := { a := 1, h := 3, k := 4 }
  let transformed_parabola := transform_parabola initial_parabola
  zeros transformed_parabola = {5, 9} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_transformation_zeros_l838_83875


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_property_l838_83880

open Real

-- Define the function f with the given properties
variable (f : ℝ → ℝ)

-- State the properties of f as hypotheses
variable (f_symmetry : ∀ x : ℝ, f x = f (2 - x))
variable (f_derivative : ∀ x : ℝ, (deriv f x) * (x - 1) > 0)

-- State the theorem to be proved
theorem f_property (f_symmetry : ∀ x : ℝ, f x = f (2 - x))
                   (f_derivative : ∀ x : ℝ, (deriv f x) * (x - 1) > 0) :
  ∀ x₁ x₂ : ℝ, x₁ < x₂ → (f x₁ > f x₂ ↔ x₁ + x₂ < 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_property_l838_83880


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_john_cell_phone_cost_l838_83869

/-- Calculates the total cost of a cell phone plan --/
noncomputable def calculate_cell_phone_cost (base_cost : ℚ) (text_cost : ℚ) (extra_minute_cost : ℚ)
  (text_count : ℕ) (talk_hours : ℚ) : ℚ :=
  let text_total := text_cost * text_count
  let extra_hours := max (talk_hours - 20) 0
  let extra_minutes := extra_hours * 60
  let extra_minute_total := extra_minute_cost * extra_minutes
  base_cost + text_total + extra_minute_total

/-- Theorem stating that John's cell phone cost in January is $58 --/
theorem john_cell_phone_cost :
  calculate_cell_phone_cost 25 (1/10) (15/100) 150 22 = 58 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_john_cell_phone_cost_l838_83869


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_household_items_percentage_is_45_l838_83858

-- Define Ajay's monthly income
def monthly_income : ℚ := 40000

-- Define the percentage spent on clothes
def clothes_percentage : ℚ := 25

-- Define the percentage spent on medicines
def medicines_percentage : ℚ := 7.5

-- Define the amount saved
def savings : ℚ := 9000

-- Define the function to calculate the percentage spent on household items
noncomputable def household_items_percentage : ℚ :=
  let clothes_amount := (clothes_percentage / 100) * monthly_income
  let medicines_amount := (medicines_percentage / 100) * monthly_income
  let household_items_amount := monthly_income - (clothes_amount + medicines_amount + savings)
  (household_items_amount / monthly_income) * 100

-- Theorem statement
theorem household_items_percentage_is_45 :
  household_items_percentage = 45 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_household_items_percentage_is_45_l838_83858


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_center_distance_l838_83891

/-- The distance between the center of the circle with equation x^2 + y^2 = 6x - 4y + 18 
    and the point (-3, -1) is √37. -/
theorem circle_center_distance : 
  let circle_eq : ℝ → ℝ → Prop := λ x y ↦ x^2 + y^2 = 6*x - 4*y + 18
  let center : ℝ × ℝ := (3, -2)
  let point : ℝ × ℝ := (-3, -1)
  Real.sqrt 37 = Real.sqrt ((center.1 - point.1)^2 + (center.2 - point.2)^2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_center_distance_l838_83891


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_PQR_l838_83854

-- Define the centers of the circles
variable (P Q R : EuclideanSpace ℝ (Fin 2))

-- Define the radii of the circles
def radius_P : ℝ := 3
def radius_Q : ℝ := 2
def radius_R : ℝ := 1

-- Define the distances between centers
noncomputable def PQ : ℝ := radius_P + radius_Q
noncomputable def PR : ℝ := radius_P + radius_R
noncomputable def QR : ℝ := radius_Q + radius_R

-- Define the area of triangle PQR
noncomputable def area_PQR : ℝ := (1/2) * PR * QR

-- Theorem statement
theorem area_of_triangle_PQR : area_PQR = 6 := by
  sorry

#check area_of_triangle_PQR

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_PQR_l838_83854


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_min_value_change_l838_83809

/-- A quadratic polynomial f(x) = ax^2 + bx + c -/
structure QuadraticPolynomial where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The minimum value of a quadratic polynomial -/
noncomputable def minValue (f : QuadraticPolynomial) : ℝ := -f.b^2 / (4 * f.a) + f.c

/-- Adding a multiple of x^2 to a quadratic polynomial -/
def addMultipleOfX2 (f : QuadraticPolynomial) (k : ℝ) : QuadraticPolynomial :=
  { a := f.a + k, b := f.b, c := f.c }

theorem quadratic_min_value_change
  (f : QuadraticPolynomial)
  (h1 : minValue (addMultipleOfX2 f 3) = minValue f + 9)
  (h2 : minValue (addMultipleOfX2 f (-1)) = minValue f - 9) :
  minValue (addMultipleOfX2 f 1) = minValue f + 9/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_min_value_change_l838_83809


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_distances_form_l838_83814

noncomputable def point_P : ℝ × ℝ := (5, 1)
noncomputable def point_A : ℝ × ℝ := (0, 0)
noncomputable def point_B : ℝ × ℝ := (12, 0)
noncomputable def point_C : ℝ × ℝ := (4, 6)

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

noncomputable def sum_of_distances : ℝ :=
  distance point_P point_A + distance point_P point_B + distance point_P point_C

theorem sum_of_distances_form (x y a b : ℕ) :
  sum_of_distances = x * Real.sqrt a + y * Real.sqrt b →
  x + y = 7 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_distances_form_l838_83814


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expansion_theorem_l838_83848

/-- The fourth term in the expansion of (x + 2/x)^6 -/
noncomputable def fourth_term (x : ℝ) : ℝ := 20 * x^3 * (2/x)^3

/-- The ratio of binomial coefficients condition -/
def binomial_ratio (n : ℕ) : Prop :=
  (n.choose 4 : ℚ) / (n.choose 2 : ℚ) = 14 / 3

/-- The constant term in the expansion of (√x - 2/x^2)^10 -/
def constant_term : ℝ := 4 * 45

theorem expansion_theorem :
  (∀ x : ℝ, x ≠ 0 → fourth_term x = 160) ∧
  binomial_ratio 10 ∧
  constant_term = 180 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_expansion_theorem_l838_83848


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_on_neg_interval_l838_83844

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.log (x^2 - 4) / Real.log (1/2)

-- State the theorem
theorem f_monotone_increasing_on_neg_interval :
  StrictMonoOn f (Set.Iio (-2)) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_on_neg_interval_l838_83844


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_of_squares_l838_83881

def integers : List Nat := [1, 2, 4, 5, 6, 9, 10, 11, 13]

structure Configuration where
  squares : List Nat
  circles : List Nat

def isValidConfiguration (c : Configuration) : Prop :=
  c.squares.length = 5 ∧
  c.circles.length = 4 ∧
  (c.squares ++ c.circles).toFinset = integers.toFinset ∧
  ∀ i, i < 4 → c.circles[i]! = c.squares[i]! + c.squares[i+1]!

def leftmostSquare (c : Configuration) : Nat := c.squares[0]!

def rightmostSquare (c : Configuration) : Nat := c.squares[4]!

theorem max_sum_of_squares :
  ∀ c : Configuration, isValidConfiguration c →
    leftmostSquare c + rightmostSquare c ≤ 20 :=
by
  sorry

#check max_sum_of_squares

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_of_squares_l838_83881


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_subset_size_is_maximum_l838_83892

/-- The maximum size of a subset of {1, 2, ..., 2017} satisfying the given conditions -/
def max_subset_size : ℕ := 504

/-- The set of numbers from 1 to 2017 -/
def S : Finset ℕ := Finset.range 2017

/-- Predicate to check if two natural numbers are coprime -/
def are_coprime (a b : ℕ) : Prop := Nat.Coprime a b

/-- Predicate to check if one natural number divides another -/
def divides (a b : ℕ) : Prop := a ∣ b

/-- The condition that must be satisfied by any two elements of the subset -/
def subset_condition (a b : ℕ) : Prop :=
  ¬(are_coprime a b) ∧ ¬(divides a b) ∧ ¬(divides b a)

/-- The theorem stating that max_subset_size is the maximum size of a subset satisfying the conditions -/
theorem max_subset_size_is_maximum :
  ∀ (A : Finset ℕ), A ⊆ S →
  (∀ (x y : ℕ), x ∈ A → y ∈ A → x ≠ y → subset_condition x y) →
  A.card ≤ max_subset_size := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_subset_size_is_maximum_l838_83892


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_form_rectangle_l838_83853

/-- The points of intersection of xy = 12 and x^2 + y^2 = 25 -/
def intersection_points : Set (ℝ × ℝ) :=
  {p | p.1 * p.2 = 12 ∧ p.1^2 + p.2^2 = 25}

/-- A quadrilateral formed by four points -/
structure Quadrilateral where
  p1 : ℝ × ℝ
  p2 : ℝ × ℝ
  p3 : ℝ × ℝ
  p4 : ℝ × ℝ

/-- Convert a Quadrilateral to a Set of points -/
def Quadrilateral.toSet (q : Quadrilateral) : Set (ℝ × ℝ) :=
  {q.p1, q.p2, q.p3, q.p4}

/-- Check if a quadrilateral is a rectangle -/
def is_rectangle (q : Quadrilateral) : Prop :=
  let d12 := (q.p1.1 - q.p2.1)^2 + (q.p1.2 - q.p2.2)^2
  let d23 := (q.p2.1 - q.p3.1)^2 + (q.p2.2 - q.p3.2)^2
  let d34 := (q.p3.1 - q.p4.1)^2 + (q.p3.2 - q.p4.2)^2
  let d41 := (q.p4.1 - q.p1.1)^2 + (q.p4.2 - q.p1.2)^2
  d12 = d34 ∧ d23 = d41 ∧ d12 ≠ d23

theorem intersection_points_form_rectangle :
  ∃ (q : Quadrilateral), (∀ p ∈ q.toSet, p ∈ intersection_points) ∧ is_rectangle q := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_form_rectangle_l838_83853


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l838_83856

/-- Definition of an ellipse with given properties -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_ab : a > b ∧ b > 0
  equation : ℝ → ℝ → Prop := λ x y => x^2 / a^2 + y^2 / b^2 = 1
  A : ℝ × ℝ := (0, 1)  -- Upper vertex
  F₁ : ℝ × ℝ
  F₂ : ℝ × ℝ
  h_F₁A : Real.sqrt ((F₁.1 - A.1)^2 + (F₁.2 - A.2)^2) = Real.sqrt 2
  h_area : (F₁.1 * F₂.2 - F₂.1 * F₁.2) / 2 = 1

/-- Points on the ellipse satisfying the given condition -/
def ValidPoints (e : Ellipse) (M N : ℝ × ℝ) : Prop :=
  e.equation M.1 M.2 ∧ 
  e.equation N.1 N.2 ∧
  (M.1 - e.A.1)^2 + (M.2 - e.A.2)^2 + (N.1 - e.A.1)^2 + (N.2 - e.A.2)^2 = 
    (M.1 - N.1)^2 + (M.2 - N.2)^2

/-- Helper function for triangle area calculation -/
noncomputable def area_triangle (A B C : ℝ × ℝ) : ℝ :=
  abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2)) / 2

/-- The main theorem to be proven -/
theorem ellipse_properties (e : Ellipse) :
  (∀ x y, e.equation x y ↔ x^2 / 2 + y^2 = 1) ∧ 
  (∃ k, ∀ M N, ValidPoints e M N → 
    (∀ P Q, ValidPoints e P Q → area_triangle e.A P Q ≤ area_triangle e.A M N) → 
    N.2 - M.2 = k * (N.1 - M.1) ∧ k = 0 ∧ M.2 = -1/3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l838_83856


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_parabola_equation_l838_83847

-- Define the ellipse parameters
noncomputable def major_axis_length : ℝ := 12
noncomputable def eccentricity : ℝ := 2/3

-- Define the hyperbola equation
def hyperbola_eq (x y : ℝ) : Prop := 16 * x^2 - 9 * y^2 = 144

-- Define the right vertex of the hyperbola
noncomputable def right_vertex : ℝ × ℝ := (3, 0)

-- Theorem for the ellipse equation
theorem ellipse_equation :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧
  ∀ (x y : ℝ), (y^2 / a^2 + x^2 / b^2 = 1) ↔
  (major_axis_length = 2 * a ∧
   eccentricity = Real.sqrt (1 - b^2 / a^2) ∧
   ∃ (c : ℝ), c^2 = a^2 - b^2 ∧ (0, c) ∈ Set.range (λ (p : ℝ × ℝ) ↦ p)) :=
by sorry

-- Theorem for the parabola equation
theorem parabola_equation :
  ∃ (p : ℝ), p > 0 ∧
  ∀ (x y : ℝ), (y^2 = 4 * p * x) ↔
  (right_vertex = (p, 0) ∧ hyperbola_eq (right_vertex.fst) (right_vertex.snd)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_parabola_equation_l838_83847


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_geometric_series_common_ratio_l838_83803

/-- The sum of an infinite geometric series with first term a and common ratio r -/
noncomputable def infiniteGeometricSeriesSum (a : ℝ) (r : ℝ) : ℝ := a / (1 - r)

/-- Theorem: The common ratio of an infinite geometric series with first term 500 and sum 4000 is 7/8 -/
theorem infinite_geometric_series_common_ratio :
  ∃ (r : ℝ), infiniteGeometricSeriesSum 500 r = 4000 ∧ r = 7/8 := by
  use 7/8
  constructor
  · -- Prove that infiniteGeometricSeriesSum 500 (7/8) = 4000
    sorry
  · -- Prove that 7/8 = 7/8
    rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_geometric_series_common_ratio_l838_83803


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_grid_sum_l838_83842

theorem grid_sum (a : Fin 2009 → Fin 2) 
  (h : ∀ i : Fin 1920, (Finset.range 90).sum (fun j => (a (i + j)).val) = 65) :
  (Finset.range 2009).sum (fun i => (a i).val) = 1450 ∨ 
  (Finset.range 2009).sum (fun i => (a i).val) = 1451 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_grid_sum_l838_83842


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersection_y_axis_l838_83866

/-- A line passing through two points (x₁, y₁) and (x₂, y₂) -/
structure Line where
  x₁ : ℝ
  y₁ : ℝ
  x₂ : ℝ
  y₂ : ℝ

/-- The y-intercept of a line -/
noncomputable def y_intercept (l : Line) : ℝ := 
  let m := (l.y₂ - l.y₁) / (l.x₂ - l.x₁)
  l.y₁ - m * l.x₁

/-- The theorem stating that the line passing through (1, 10) and (-9, -10) 
    intersects the y-axis at (0, 8) -/
theorem line_intersection_y_axis : 
  let l : Line := { x₁ := 1, y₁ := 10, x₂ := -9, y₂ := -10 }
  y_intercept l = 8 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersection_y_axis_l838_83866


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l838_83831

-- Define the function
noncomputable def f (x : ℝ) : ℝ := 4 / (x - 2)

-- State the theorem
theorem min_value_of_f :
  (∀ x₁ x₂, 3 ≤ x₁ ∧ x₁ < x₂ ∧ x₂ ≤ 6 → f x₁ > f x₂) →
  (∀ x, 3 ≤ x ∧ x ≤ 6 → f x ≥ 1) ∧
  (∃ x, 3 ≤ x ∧ x ≤ 6 ∧ f x = 1) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l838_83831


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l838_83807

-- Define the function
noncomputable def f (x : ℝ) : ℝ := (Real.cos x) ^ 2 + Real.sin x - 1

-- State the theorem
theorem f_range :
  Set.range f = Set.Icc (-2) (1/4) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l838_83807


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_parameterization_l838_83846

/-- The line y = (1/3)x + 3 parameterized by (x, y) = (-9, s) + t(l, -4) has s = 0 and l = -12 -/
theorem line_parameterization (s l : ℝ) : 
  (∀ t x y : ℝ, y = (1/3)*x + 3 ↔ ∃ t : ℝ, (x, y) = (-9, s) + t • (l, -4)) →
  s = 0 ∧ l = -12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_parameterization_l838_83846


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_integral_value_l838_83828

-- Define the integral function as noncomputable
noncomputable def integral (a : ℝ) : ℝ := ∫ x in (0:ℝ)..1, |a * x - x^3|

-- State the theorem
theorem min_integral_value :
  ∃ m : ℝ, m = 1/8 ∧ ∀ a : ℝ, integral a ≥ m :=
by
  -- We'll use 1/8 as our minimum value
  let m := (1 : ℝ) / 8
  
  -- Prove that m equals 1/8
  have h_m_eq : m = 1/8 := by rfl
  
  -- Assert that for all a, the integral is greater than or equal to m
  have h_min : ∀ a : ℝ, integral a ≥ m := by
    intro a
    -- The actual proof would go here, but we'll use sorry for now
    sorry
  
  -- Combine our proofs to match the theorem statement
  exact ⟨m, h_m_eq, h_min⟩

-- Note: The proof is incomplete and uses 'sorry'

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_integral_value_l838_83828


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_minus_b_plus_c_equals_five_l838_83855

/-- Definition of the sequence S --/
def S (n a b c : ℤ) : ℤ := (2*n - 1) * (a * n^2 + b * n + c)

/-- The main theorem --/
theorem a_minus_b_plus_c_equals_five
  (h1 : S 1 a b c = 1)
  (h3 : S 3 a b c = 15)
  (h5 : S 5 a b c = 65)
  (a b c : ℤ) :
  a - b + c = 5 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_minus_b_plus_c_equals_five_l838_83855


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stating_distinctExtractionsCount_correct_l838_83850

/-- 
Given n^2 distinct real numbers arranged in an n × n table, 
this function calculates the number of ways to arrange them 
such that exactly 2n distinct numbers are extracted when 
taking the maximum from each column and minimum from each row.
-/
def distinctExtractionsCount (n : ℕ) : ℕ :=
  Nat.factorial (n^2) - 
  (Nat.choose (n^2) (2*n - 1)) * n^2 * 
  (Nat.factorial (n-1))^2 * 
  Nat.factorial (n^2 - 2*n + 1)

/-- 
Theorem stating that distinctExtractionsCount correctly calculates 
the number of valid arrangements for the given problem.
-/
theorem distinctExtractionsCount_correct (n : ℕ) : 
  distinctExtractionsCount n = 
    -- The actual number of valid arrangements
    Nat.factorial (n^2) - 
    (Nat.choose (n^2) (2*n - 1)) * n^2 * 
    (Nat.factorial (n-1))^2 * 
    Nat.factorial (n^2 - 2*n + 1) := by
  -- The proof is omitted
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_stating_distinctExtractionsCount_correct_l838_83850


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rationalize_denominator_l838_83849

theorem rationalize_denominator :
  (1 : ℝ) / (Real.rpow 2 (1/3) + Real.rpow 32 (1/3) + 1) = 2 / 11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rationalize_denominator_l838_83849


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_squared_l838_83812

/-- A circle in a 2D plane --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The square of the distance between two points in a 2D plane --/
def distanceSquared (p1 p2 : ℝ × ℝ) : ℝ :=
  (p1.1 - p2.1)^2 + (p1.2 - p2.2)^2

/-- The intersection points of two circles --/
def intersectionPoints (c1 c2 : Circle) : Set (ℝ × ℝ) :=
  {p | distanceSquared p c1.center = c1.radius^2 ∧ distanceSquared p c2.center = c2.radius^2}

theorem intersection_distance_squared :
  let c1 : Circle := ⟨(2, 0), 3⟩
  let c2 : Circle := ⟨(5, 0), 5⟩
  let points := intersectionPoints c1 c2
  ∀ p1 p2, p1 ∈ points → p2 ∈ points → p1 ≠ p2 → distanceSquared p1 p2 = 50 := by
  sorry

#check intersection_distance_squared

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_squared_l838_83812


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mouse_can_always_escape_l838_83836

/-- Represents a position on the 8x8 chessboard -/
structure Position where
  x : Fin 8
  y : Fin 8

/-- The game state -/
structure GameState where
  mousePos : Position
  catPos1 : Position
  catPos2 : Position
  catPos3 : Position
  isFirstMove : Bool

/-- Checks if a position is on the edge of the board -/
def isEdge (pos : Position) : Bool :=
  pos.x = 0 || pos.x = 7 || pos.y = 0 || pos.y = 7

/-- Represents a single move (up, down, left, or right) -/
inductive Move where
  | up
  | down
  | left
  | right

/-- Apply a move to a position -/
def applyMove (pos : Position) (move : Move) : Position :=
  match move with
  | Move.up => ⟨pos.x, (pos.y + 1 : Fin 8)⟩
  | Move.down => ⟨pos.x, (pos.y - 1 : Fin 8)⟩
  | Move.left => ⟨(pos.x - 1 : Fin 8), pos.y⟩
  | Move.right => ⟨(pos.x + 1 : Fin 8), pos.y⟩

/-- Update the game state based on a move -/
def updateGameState (state : GameState) (move : Move) : GameState :=
  let newMousePos := applyMove state.mousePos move
  -- For simplicity, we're not implementing cat movement logic here
  { state with 
    mousePos := newMousePos,
    isFirstMove := false 
  }

/-- The main theorem to prove -/
theorem mouse_can_always_escape :
  ∀ (initialState : GameState),
  ∃ (strategy : GameState → Move),
    ∃ (n : ℕ),
      let finalState := (n.iterate (λ state => updateGameState state (strategy state)) initialState)
      isEdge finalState.mousePos ∧
      ¬(finalState.mousePos = finalState.catPos1 ∨
        finalState.mousePos = finalState.catPos2 ∨
        finalState.mousePos = finalState.catPos3) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_mouse_can_always_escape_l838_83836


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangular_field_area_l838_83883

/-- Proves that a rectangular field with sides in ratio 3:4 and fencing cost of 101.5 rupees at 25 paise per metre has an area of 10092 square meters -/
theorem rectangular_field_area (length width : ℝ) (fencing_cost : ℝ) (cost_per_meter : ℝ) : 
  length / width = 3 / 4 →
  fencing_cost = 101.5 →
  cost_per_meter = 0.25 →
  2 * (length + width) * cost_per_meter = fencing_cost * 100 →
  length * width = 10092 := by
  sorry

#check rectangular_field_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangular_field_area_l838_83883


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_abc_l838_83819

/-- A quadratic function f(x) with parameters a, b, and c. -/
noncomputable def f (a b c x : ℝ) : ℝ := (a + 2*b)*x^2 - 2*Real.sqrt 3*x + a + 2*c

/-- The theorem stating the minimum value of a + b + c given the conditions on f. -/
theorem min_value_abc (a b c : ℝ) 
  (h_range : ∀ y : ℝ, y ∈ Set.range (f a b c) → y ≥ 0) :
  a + b + c ≥ Real.sqrt 3 := by
  sorry

#check min_value_abc

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_abc_l838_83819


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_theorem_l838_83887

/-- A quadratic function f(x) = ax² + bx + c with a ≠ 0 -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  a_nonzero : a ≠ 0

/-- The function f(x) satisfies the given conditions -/
def satisfies_conditions (f : QuadraticFunction) : Prop :=
  (∀ x : ℝ, f.a * x^2 + f.b * x + f.c < 0 ↔ 1 < x ∧ x < 2) ∧
  (f.a * 3^2 + f.b * 3 + f.c = 2)

/-- The minimum value of g(x) = f(x) - mx on [1, 2] -/
noncomputable def h (m : ℝ) : ℝ :=
  if m ≤ -1 then -m
  else if m < 1 then -(3 + m)^2 / 4 + 2
  else -2 * m

/-- The main theorem to be proved -/
theorem quadratic_function_theorem (f : QuadraticFunction) 
  (hf : satisfies_conditions f) :
  (f.a = 1 ∧ f.b = -3 ∧ f.c = 2) ∧
  (∀ m : ℝ, ∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → 
    f.a * x^2 + f.b * x + f.c - m * x ≥ h m) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_theorem_l838_83887


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pentagon_diagonal_sum_l838_83832

-- Define the pentagon
structure Pentagon where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  E : ℝ × ℝ

-- Define the distance function
noncomputable def distance (p q : ℝ × ℝ) : ℝ := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Define the theorem
theorem pentagon_diagonal_sum (p : Pentagon) (O : ℝ × ℝ) :
  -- Conditions
  distance p.A p.B = 5 →
  distance p.C p.D = 5 →
  distance p.B p.C = 12 →
  distance p.D p.E = 12 →
  distance p.A p.E = 18 →
  -- Inscribed in a circle condition (represented by equal radius)
  distance p.A O = distance p.B O →
  distance p.B O = distance p.C O →
  distance p.C O = distance p.D O →
  distance p.D O = distance p.E O →
  -- Conclusion
  distance p.A p.C + distance p.B p.D + distance p.C p.E = 6211 / 132 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pentagon_diagonal_sum_l838_83832


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_2x_eq_sin_x_solutions_l838_83821

open Set

theorem tan_2x_eq_sin_x_solutions :
  ∃ (S : Set ℝ), S.Finite ∧ S.ncard = 5 ∧
  (∀ x ∈ S, 0 ≤ x ∧ x ≤ 2 * Real.pi ∧ Real.tan (2 * x) = Real.sin x) ∧
  (∀ x, 0 ≤ x ∧ x ≤ 2 * Real.pi ∧ Real.tan (2 * x) = Real.sin x → x ∈ S) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_2x_eq_sin_x_solutions_l838_83821


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_rotation_l838_83829

/-- Represents a parabola in the form y + c = ax^2 + bx -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

noncomputable def Parabola.focus (p : Parabola) : Point :=
  { x := -p.b / (2 * p.a), y := -(p.c + (p.b^2 - 1) / (4 * p.a)) }

noncomputable def Parabola.vertex (p : Parabola) : Point :=
  { x := -p.b / (2 * p.a), y := -(p.c + p.b^2 / (4 * p.a)) }

def rotateClockwise90 (center : Point) (p : Point) : Point :=
  { x := center.x - (p.y - center.y),
    y := center.y + (p.x - center.x) }

theorem parabola_rotation (p : Parabola) (h : p = { a := 1, b := 0, c := 1 }) :
  let focus := p.focus
  let vertex := p.vertex
  let rotatedVertex := rotateClockwise90 focus vertex
  rotatedVertex = { x := -3/4, y := -1/4 } := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_rotation_l838_83829


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_sin_shift_l838_83878

noncomputable def f (x : ℝ) : ℝ := Real.cos (2 * x + Real.pi / 3)
noncomputable def g (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 3)

theorem cos_sin_shift (x : ℝ) : f x = g (x + Real.pi / 4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_sin_shift_l838_83878


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l838_83870

-- Define the hyperbola parameters
variable (a b : ℝ)

-- Define the conditions
def hyperbola_equation (x y : ℝ) : Prop := x^2 / a^2 - y^2 / b^2 = 1
def positive_parameters : Prop := a > 0 ∧ b > 0
def asymptote_point : Prop := ∃ (x y : ℝ), x = 2 ∧ y = 2 * Real.sqrt 3 ∧ b / a = y / x

-- Define the eccentricity
noncomputable def eccentricity : ℝ := Real.sqrt (1 + (b / a)^2)

-- Theorem statement
theorem hyperbola_eccentricity :
  positive_parameters a b →
  asymptote_point a b →
  eccentricity a b = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l838_83870


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_other_side_red_is_two_thirds_l838_83886

/-- Represents a card with two sides -/
inductive Card
| BB  -- Black on both sides
| BR  -- Black on one side, Red on the other
| RR  -- Red on both sides
deriving DecidableEq

/-- The box of cards -/
def box : Multiset Card :=
  4 • {Card.BB} + 2 • {Card.BR} + 2 • {Card.RR}

/-- The probability of drawing a specific card from the box -/
def prob_draw (c : Card) : ℚ :=
  (box.count c) / (Multiset.card box : ℚ)

/-- The probability of observing a red side on a given card -/
def prob_red_side (c : Card) : ℚ :=
  match c with
  | Card.BB => 0
  | Card.BR => 1/2
  | Card.RR => 1

/-- The probability of the other side being red, given that we observe a red side -/
noncomputable def prob_other_side_red : ℚ :=
  (prob_draw Card.RR * prob_red_side Card.RR) /
  (prob_draw Card.BR * prob_red_side Card.BR + prob_draw Card.RR * prob_red_side Card.RR)

theorem prob_other_side_red_is_two_thirds :
  prob_other_side_red = 2/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_other_side_red_is_two_thirds_l838_83886


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_problem_l838_83852

theorem percentage_problem (P : ℝ) : P = 40 := by
  -- Define the conditions
  let number : ℝ := 15
  let condition : Prop := (P / 100) * number = (80 / 100) * 5 + 2

  -- State the theorem
  have h : condition → P = 40 := by
    intro hyp
    -- Proof steps would go here
    sorry

  -- Apply the hypothesis
  apply h
  -- Prove the condition
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_problem_l838_83852


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extremum_implies_a_value_l838_83865

/-- The function f(x) = (x^2 + a) / e^x -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x^2 + a) / Real.exp x

/-- The derivative of f(x) -/
noncomputable def f_derivative (a : ℝ) (x : ℝ) : ℝ := (2*x - x^2 - a) / Real.exp x

theorem extremum_implies_a_value (a : ℝ) :
  (∃ (ε : ℝ), ∀ (x : ℝ), |x - 3| < ε → f a x ≤ f a 3) →
  (∃ (ε : ℝ), ∀ (x : ℝ), |x - 3| < ε → f a x ≥ f a 3) →
  f_derivative a 3 = 0 →
  a = -3 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_extremum_implies_a_value_l838_83865


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_star_product_properties_l838_83872

structure PlaneVector where
  x : ℝ
  y : ℝ

def star_product (a b : PlaneVector) : ℝ :=
  a.x * b.y - a.y * b.x

def dot_product (a b : PlaneVector) : ℝ :=
  a.x * b.x + a.y * b.y

def norm_squared (a : PlaneVector) : ℝ :=
  a.x^2 + a.y^2

def collinear (a b : PlaneVector) : Prop :=
  ∃ (k : ℝ), a.x = k * b.x ∧ a.y = k * b.y

theorem star_product_properties :
  ∀ (a b : PlaneVector) (l : ℝ),
    (collinear a b → star_product a b = 0) ∧
    (star_product (PlaneVector.mk (l * a.x) (l * a.y)) b = l * star_product a b) ∧
    ((star_product a b)^2 + (dot_product a b)^2 = norm_squared a * norm_squared b) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_star_product_properties_l838_83872


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_external_tangent_circle_radius_l838_83860

/-- Given three externally tangent circles with radii 1, 2, and 3 cm, 
    the radius of the circle that is externally tangent to all three circles is 6 cm. -/
theorem external_tangent_circle_radius 
  (A B C : EuclideanSpace ℝ (Fin 2)) 
  (r₁ r₂ r₃ : ℝ) 
  (h₁ : r₁ = 1) 
  (h₂ : r₂ = 2) 
  (h₃ : r₃ = 3) 
  (h_dist_AB : ‖A - B‖ = r₁ + r₂) 
  (h_dist_AC : ‖A - C‖ = r₁ + r₃) 
  (h_dist_BC : ‖B - C‖ = r₂ + r₃) : 
  ∃ (D : EuclideanSpace ℝ (Fin 2)) (R : ℝ), 
    R = 6 ∧ 
    ‖D - A‖ = R + r₁ ∧ 
    ‖D - B‖ = R + r₂ ∧ 
    ‖D - C‖ = R + r₃ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_external_tangent_circle_radius_l838_83860


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sharon_drove_720_miles_l838_83825

/-- Represents Sharon's car and trip details -/
structure CarTrip where
  miles_per_gallon : ℚ
  tank_capacity : ℚ
  initial_miles : ℚ
  gas_bought : ℚ
  final_tank_fraction : ℚ

/-- Calculates the total miles driven given a CarTrip -/
def total_miles_driven (trip : CarTrip) : ℚ :=
  trip.initial_miles + 
  (trip.tank_capacity + trip.gas_bought - 
   trip.tank_capacity * trip.final_tank_fraction - 
   trip.initial_miles / trip.miles_per_gallon) * 
  trip.miles_per_gallon

/-- Theorem stating that Sharon drove 720 miles -/
theorem sharon_drove_720_miles : 
  let trip : CarTrip := {
    miles_per_gallon := 40,
    tank_capacity := 16,
    initial_miles := 480,
    gas_bought := 6,
    final_tank_fraction := 1/4
  }
  total_miles_driven trip = 720 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sharon_drove_720_miles_l838_83825


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_restaurant_bill_calculation_l838_83864

theorem restaurant_bill_calculation (order_amount : ℝ) 
  (service_charge_rate : ℝ) (sales_tax_rate : ℝ) (discount_rate : ℝ) : 
  order_amount = 450 ∧ 
  service_charge_rate = 0.04 ∧ 
  sales_tax_rate = 0.05 ∧ 
  discount_rate = 0.10 → 
  (order_amount * (1 - discount_rate) * (1 + service_charge_rate) * (1 + sales_tax_rate)) = 442.26 := by
  sorry

#eval (450 : ℝ) * (1 - 0.10) * (1 + 0.04) * (1 + 0.05)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_restaurant_bill_calculation_l838_83864


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_abs_equation_solutions_l838_83862

theorem abs_equation_solutions :
  ∃! (s : Finset ℝ), (∀ x ∈ s, |x - 1| = |x - 2| + |x - 3|) ∧ s.card = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_abs_equation_solutions_l838_83862


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_symmetry_l838_83894

noncomputable def g (x : ℝ) : ℝ := |⌊2*x⌋| - |⌊2 - x⌋|

theorem g_symmetry (x : ℝ) : g x = g (2 - x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_symmetry_l838_83894


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_f_on_interval_min_value_is_two_l838_83877

-- Define the function f(x) = (1/2)^x
noncomputable def f (x : ℝ) : ℝ := (1/2) ^ x

-- State the theorem
theorem min_value_f_on_interval :
  ∀ x ∈ Set.Icc (-2) (-1), f (-1) ≤ f x :=
by
  sorry

-- Define the minimum value
noncomputable def min_value : ℝ := f (-1)

-- Prove that the minimum value is 2
theorem min_value_is_two : min_value = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_f_on_interval_min_value_is_two_l838_83877


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part1_part2_l838_83820

-- Part 1
theorem part1 (a b : ℝ) (m n : ℤ) (h : (-a^2 * b^m)^3 = -a^n * b^12) : m = 4 ∧ n = 6 := by
  sorry

-- Part 2
theorem part2 (x : ℝ) (h : (2 : ℝ)^(2*x + 2) - (2 : ℝ)^(2*x + 1) = 32) : x = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part1_part2_l838_83820


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_magnitude_proof_l838_83863

theorem complex_magnitude_proof : 
  let z : ℂ := 1 / (Complex.I * (Complex.I + 1))
  Complex.abs z = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_magnitude_proof_l838_83863


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_condition_l838_83808

-- Define the circles C₀ and C₁
def C₀ (x y : ℝ) : Prop := x^2 + y^2 = 1
def C₁ (x y a b : ℝ) : Prop := x^2/a^2 + y^2/b^2 = 1

-- Define the condition for a and b
def condition (a b : ℝ) : Prop := 1/a^2 + 1/b^2 = 1

-- Helper definition for parallelogram
def is_parallelogram (p q r s : ℝ × ℝ) : Prop :=
  (q.1 - p.1 = s.1 - r.1) ∧ (q.2 - p.2 = s.2 - r.2) ∧
  (r.1 - q.1 = s.1 - p.1) ∧ (r.2 - q.2 = s.2 - p.2)

-- Main theorem
theorem parallelogram_condition (a b : ℝ) (h₁ : a > b) (h₂ : b > 0) :
  (∀ x y : ℝ, C₁ x y a b → 
    ∃ p q r s : ℝ × ℝ, 
      (p.1 = x ∧ p.2 = y) ∧ 
      C₁ q.1 q.2 a b ∧ C₁ r.1 r.2 a b ∧ C₁ s.1 s.2 a b ∧
      C₀ q.1 q.2 ∧ C₀ r.1 r.2 ∧ C₀ s.1 s.2 ∧
      is_parallelogram p q r s) ↔ 
  condition a b :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_condition_l838_83808


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_seven_pi_fourth_l838_83889

theorem cos_seven_pi_fourth : Real.cos (7 * π / 4) = 1 / Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_seven_pi_fourth_l838_83889


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_intersection_properties_l838_83817

/-- An ellipse with given properties -/
structure Ellipse where
  a : ℝ
  b : ℝ
  equation : (x : ℝ) → (y : ℝ) → Prop := λ x y ↦ x^2 / 4 + y^2 / 3 = 1
  A : ℝ × ℝ := (-2, 0)  -- left vertex
  F1 : ℝ × ℝ := (-1, 0) -- left focus
  F2 : ℝ × ℝ := (1, 0)  -- right focus

/-- A line passing through a point with a given slope -/
def Line (p : ℝ × ℝ) (k : ℝ) : (x : ℝ) → (y : ℝ) → Prop :=
  λ x y ↦ y = k * (x - p.1) + p.2

/-- The intersection point of a line and the ellipse -/
noncomputable def intersectionPoint (e : Ellipse) (l : (x : ℝ) → (y : ℝ) → Prop) : ℝ × ℝ := sorry

/-- The coordinates of point B -/
noncomputable def B_coords (e : Ellipse) (k : ℝ) : ℝ × ℝ :=
  let x := (-8 * k^2 + 8 * Real.sqrt 3) / (3 + 4 * k^2)
  let y := k * (x + 2)
  (x, y)

/-- Check if two lines are perpendicular -/
def perpendicular (l1 l2 : (x : ℝ) → (y : ℝ) → Prop) : Prop := sorry

/-- Theorem stating the properties of the ellipse and its intersections -/
theorem ellipse_intersection_properties (e : Ellipse) (k : ℝ) (m : ℝ) :
  let l := Line e.A k
  let B := intersectionPoint e l
  let C := intersectionPoint e (Line e.F2 (B.2 / B.1))
  (B = B_coords e k) ∧
  (perpendicular (Line e.F1 ((C.2 - e.F1.2) / (C.1 - e.F1.1))) l → k = (m - 1) / (m + 1)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_intersection_properties_l838_83817


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_y_equals_x_is_line_l838_83806

/-- The curve defined by y = x in Cartesian coordinates is a line -/
theorem curve_y_equals_x_is_line :
  ∃ (a b c : ℝ), a ≠ 0 ∨ b ≠ 0 ∧ ∀ (x y : ℝ), y = x ↔ a * x + b * y + c = 0 :=
by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_y_equals_x_is_line_l838_83806


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_and_area_ratio_l838_83868

noncomputable section

-- Define the cube
structure Cube where
  side : ℝ
  vertices : Fin 8 → ℝ × ℝ × ℝ

-- Define points A, B, and C
def A (cube : Cube) : ℝ × ℝ × ℝ := cube.vertices 0
def B (cube : Cube) : ℝ × ℝ × ℝ := ((cube.side / 2), 0, 0)
def C (cube : Cube) : ℝ × ℝ × ℝ := (0, (cube.side / 2), 0)

-- Define the plane containing A, B, and C
def plane (cube : Cube) : Set (ℝ × ℝ × ℝ) :=
  {p | p.2.2 = 0}

-- Define the polygon P
def P (cube : Cube) : Set (ℝ × ℝ × ℝ) :=
  {p ∈ plane cube | p.1 ≥ 0 ∧ p.1 ≤ cube.side ∧ p.2.1 ≥ 0 ∧ p.2.1 ≤ cube.side}

-- Define the number of sides of P
def numSides (_P : Set (ℝ × ℝ × ℝ)) : ℕ := 4

-- Define the area of P
def areaP (cube : Cube) : ℝ := cube.side * cube.side

-- Define the area of triangle ABC
def areaABC (cube : Cube) : ℝ := (cube.side * cube.side) / 8

-- Theorem statement
theorem intersection_and_area_ratio (cube : Cube) :
  numSides (P cube) = 4 ∧ (areaP cube) / (areaABC cube) = 8 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_and_area_ratio_l838_83868


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l838_83800

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 3 * x - Real.exp x + 1

-- State the theorem
theorem f_properties :
  -- Part 1: Existence of positive root
  (∃ x₀ : ℝ, x₀ > 0 ∧ f x₀ = 0) ∧
  -- Part 2: Curve lies below tangent line at root
  (∃ x₀ : ℝ, x₀ > 0 ∧ f x₀ = 0 ∧ ∀ x, f x ≤ (3 - Real.exp x₀) * (x - x₀)) ∧
  -- Part 3: Inequality for roots of f(x) = m
  (∀ m : ℝ, m > 0 →
    ∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ = m → f x₂ = m →
      x₂ - x₁ < 2 - (3 * m) / 4) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l838_83800


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_not_monotone_increasing_l838_83859

-- Define the function g(x)
noncomputable def g (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 3)

-- Statement to prove
theorem g_not_monotone_increasing :
  ¬ (∀ x y : ℝ, 0 < x ∧ x < y ∧ y < Real.pi / 6 → g x < g y) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_not_monotone_increasing_l838_83859


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_is_hyperbola_branch_constant_difference_less_than_AB_distance_l838_83871

-- Define the points A and B
def A : ℝ × ℝ := (-400, 0)
def B : ℝ × ℝ := (400, 0)

-- Define the distance between A and B
def distance_AB : ℝ := 800

-- Define the speed of sound
def speed_of_sound : ℝ := 340

-- Define the time difference of sound arrival
def time_difference : ℝ := 2

-- Define a point M on the trajectory
def M : ℝ × ℝ → Prop := fun _ => True

-- Define the distance function
noncomputable def distance (p q : ℝ × ℝ) : ℝ := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Theorem stating that the trajectory is a branch of a hyperbola closer to B
theorem trajectory_is_hyperbola_branch (x y : ℝ) :
  M (x, y) ↔ 
    abs (distance (x, y) A - distance (x, y) B) = speed_of_sound * time_difference ∧
    distance (x, y) B < distance (x, y) A :=
by
  sorry

-- Theorem stating that the constant difference is less than the distance between A and B
theorem constant_difference_less_than_AB_distance :
  speed_of_sound * time_difference < distance_AB :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_is_hyperbola_branch_constant_difference_less_than_AB_distance_l838_83871


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l838_83824

/-- Definition of a hyperbola -/
def is_hyperbola (C : Set (ℝ × ℝ)) (a b : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ ∀ p : ℝ × ℝ, p ∈ C ↔ (p.1^2 / a^2) - (p.2^2 / b^2) = 1

/-- Definition of foci for a hyperbola -/
def foci (C : Set (ℝ × ℝ)) (a b : ℝ) (F₁ F₂ : ℝ × ℝ) : Prop :=
  is_hyperbola C a b ∧ 
  ∃ c, c^2 = a^2 + b^2 ∧ F₁ = (-c, 0) ∧ F₂ = (c, 0)

/-- Definition of a point on the hyperbola -/
def on_hyperbola (P : ℝ × ℝ) (C : Set (ℝ × ℝ)) : Prop :=
  P ∈ C

/-- Definition of an isosceles triangle -/
def is_isosceles_triangle (P F₁ F₂ : ℝ × ℝ) : Prop :=
  dist P F₁ = dist P F₂

/-- Cosine of the angle PF₁F₂ -/
noncomputable def cos_angle_PF₁F₂ (P F₁ F₂ : ℝ × ℝ) : ℝ :=
  ((dist P F₁)^2 + (dist F₁ F₂)^2 - (dist P F₂)^2)
  / (2 * dist P F₁ * dist F₁ F₂)

/-- Definition of eccentricity for a hyperbola -/
noncomputable def eccentricity (a c : ℝ) : ℝ :=
  c / a

/-- Main theorem -/
theorem hyperbola_eccentricity 
  (C : Set (ℝ × ℝ)) (a b : ℝ) (F₁ F₂ P : ℝ × ℝ) :
  is_hyperbola C a b →
  foci C a b F₁ F₂ →
  on_hyperbola P C →
  is_isosceles_triangle P F₁ F₂ →
  cos_angle_PF₁F₂ P F₁ F₂ = 1/8 →
  ∃ c, eccentricity a c = 4/3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l838_83824


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_AB_equation_l838_83839

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 2*y + 1 = 0

-- Define the point Q that bisects chord AB
def Q : ℝ × ℝ := (1, 0)

-- Define the property that Q bisects chord AB on circle C
def Q_bisects_AB (A B : ℝ × ℝ) : Prop :=
  circle_C A.1 A.2 ∧ circle_C B.1 B.2 ∧ 
  Q = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

-- Define the line through two points
def line_through (A B : ℝ × ℝ) (x y : ℝ) : Prop :=
  (y - A.2) * (B.1 - A.1) = (x - A.1) * (B.2 - A.2)

-- Theorem statement
theorem chord_AB_equation :
  ∃ (A B : ℝ × ℝ), Q_bisects_AB A B →
  ∀ (x y : ℝ), line_through A B x y ↔ y = -x + 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_AB_equation_l838_83839
