import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_octagon_area_is_800_l420_42067

/-- A square with a regular octagon inscribed in it -/
structure OctagonInSquare where
  /-- The side length of the square -/
  square_side : ℝ
  /-- The perimeter of the square is 160 centimeters -/
  square_perimeter : square_side * 4 = 160

/-- The area of the octagon inscribed in the square -/
noncomputable def octagon_area (o : OctagonInSquare) : ℝ :=
  o.square_side^2 - 4 * (o.square_side / 2)^2 / 2

/-- Theorem stating that the area of the inscribed octagon is 800 square centimeters -/
theorem octagon_area_is_800 (o : OctagonInSquare) : octagon_area o = 800 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_octagon_area_is_800_l420_42067


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_candidate_c_wins_l420_42029

-- Define the candidates and their vote counts
def candidates : Fin 5 → String := ![
  "Candidate A", "Candidate B", "Candidate C", "Candidate D", "Candidate E"
]

def votes : Fin 5 → ℕ := ![4500, 7000, 12000, 8500, 3500]

-- Define the total number of votes
def total_votes : ℕ := (List.range 5).map votes |>.sum

-- Define the function to calculate vote percentage
def vote_percentage (i : Fin 5) : ℚ :=
  (votes i : ℚ) / (total_votes : ℚ) * 100

-- Define the winner
noncomputable def winner : Fin 5 :=
  (List.range 5).argmax (fun i => vote_percentage ↑i) |>.get!

-- Define approximate equality
def approx_eq (x y : ℚ) : Prop := abs (x - y) < 1/1000

-- Theorem statement
theorem candidate_c_wins :
  candidates winner = "Candidate C" ∧
  approx_eq (vote_percentage winner) (33803/1000) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_candidate_c_wins_l420_42029


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_regular_pentagon_on_cube_edges_l420_42039

-- Define a cube
def Cube := Unit

-- Define a regular pentagon
def RegularPentagon := Unit

-- Define a function that checks if a point is on the edge of a cube
def isOnCubeEdge (c : Cube) (p : ℝ × ℝ × ℝ) : Prop := sorry

-- Define a function that checks if a set of points forms a regular pentagon
def formsRegularPentagon (points : Finset (ℝ × ℝ × ℝ)) : Prop := sorry

-- Theorem statement
theorem no_regular_pentagon_on_cube_edges :
  ∀ (c : Cube) (points : Finset (ℝ × ℝ × ℝ)),
    (∀ p ∈ points, isOnCubeEdge c p) →
    (points.card = 5) →
    ¬ formsRegularPentagon points :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_regular_pentagon_on_cube_edges_l420_42039


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_angle_of_line_l420_42075

theorem slope_angle_of_line : 
  ∃ (x y θ : ℝ), 
    (Real.sqrt 3) * x + 3 * y + 1 = 0 ∧
    θ = 5 * Real.pi / 6 ∧
    (Real.sqrt 3) * Real.cos θ + 3 * Real.sin θ = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_angle_of_line_l420_42075


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_parallel_to_parallel_planes_l420_42083

-- Define the concept of a plane
structure Plane where
  -- (Placeholder for plane properties)

-- Define the concept of a line
structure Line where
  -- (Placeholder for line properties)

-- Define the parallel relation between planes
def parallel_planes (p1 p2 : Plane) : Prop :=
  sorry -- (Placeholder for parallel planes definition)

-- Define the parallel relation between a line and a plane
def parallel_line_plane (l : Line) (p : Plane) : Prop :=
  sorry -- (Placeholder for line parallel to plane definition)

-- Define the relation of a line lying within a plane
def line_in_plane (l : Line) (p : Plane) : Prop :=
  sorry -- (Placeholder for line in plane definition)

-- Theorem statement
theorem line_parallel_to_parallel_planes 
  (p1 p2 : Plane) (l : Line) 
  (h_parallel_planes : parallel_planes p1 p2)
  (h_parallel_line_p1 : parallel_line_plane l p1) :
  parallel_line_plane l p2 ∨ line_in_plane l p2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_parallel_to_parallel_planes_l420_42083


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_double_angle_plus_pi_third_l420_42096

theorem sine_double_angle_plus_pi_third (θ : ℝ) :
  Real.sin (θ - π / 12) = 3 / 4 → Real.sin (2 * θ + π / 3) = -1 / 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_double_angle_plus_pi_third_l420_42096


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_record_storage_cost_l420_42008

/-- Represents the dimensions and storage details of record boxes --/
structure RecordStorage where
  length : ℝ
  width : ℝ
  height : ℝ
  totalVolume : ℝ
  totalPayment : ℝ

/-- Calculates the cost per box per month for record storage --/
noncomputable def costPerBox (rs : RecordStorage) : ℝ :=
  let boxVolume := rs.length * rs.width * rs.height
  let numBoxes := rs.totalVolume / boxVolume
  rs.totalPayment / numBoxes

/-- Theorem stating the cost per box per month for the given scenario --/
theorem record_storage_cost (rs : RecordStorage) 
  (h1 : rs.length = 15)
  (h2 : rs.width = 12)
  (h3 : rs.height = 10)
  (h4 : rs.totalVolume = 1080000)
  (h5 : rs.totalPayment = 240) :
  costPerBox rs = 0.40 := by
  sorry

#eval "Record storage cost theorem defined"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_record_storage_cost_l420_42008


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l420_42050

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the derivative of f
def f' : ℝ → ℝ := sorry

-- Axioms based on the given conditions
axiom f_symmetry (x : ℝ) : f x + f (-x) = x^2
axiom f'_gt_x (x : ℝ) : x > 0 → f' x > x

-- Define the set of a satisfying the inequality
def A : Set ℝ := {a | f (1 + a) - f (1 - a) ≥ 2 * a}

-- Theorem to prove
theorem range_of_a : A = Set.Ici 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l420_42050


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_mn_proof_l420_42052

noncomputable def minimum_mn : ℝ := 5 * Real.pi / 48

theorem minimum_mn_proof (m n : ℝ) (h_n : n > 0) : 
  Real.sin (2 * (Real.pi / 12)) = m →
  Real.cos (2 * (Real.pi / 12 + n) - Real.pi / 4) = m →
  m * n ≥ minimum_mn := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_mn_proof_l420_42052


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_seven_segment_more_accurate_l420_42046

/-- The exact side length of a regular heptagon inscribed in a circle of radius r -/
noncomputable def exactHeptagonSide (r : ℝ) : ℝ := 0.867768 * r

/-- The height of an equilateral triangle with side length r -/
noncomputable def equilateralHeight (r : ℝ) : ℝ := (Real.sqrt 3 / 2) * r

/-- The approximate side length of a heptagon using the 7-segment construction method -/
noncomputable def sevenSegmentApprox (r : ℝ) : ℝ := r * (Real.sqrt 61 / 3)

/-- Calculate the relative error in permille -/
noncomputable def relativeError (approx actual : ℝ) : ℝ :=
  1000 * (abs (approx - actual) / actual)

/-- Theorem stating that the 7-segment construction method is more accurate -/
theorem seven_segment_more_accurate (r : ℝ) (h : r > 0) :
  relativeError (sevenSegmentApprox r) (exactHeptagonSide r) <
  relativeError (equilateralHeight r) (exactHeptagonSide r) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_seven_segment_more_accurate_l420_42046


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_m_values_l420_42038

-- Define the slope of a line Ax + By + C = 0
noncomputable def lineslope (A B : ℝ) : ℝ := -A / B

-- Define our lines l₁ and l₂
def l₁ (m : ℝ) : ℝ → ℝ → Prop := λ x y ↦ x + m * y + 7 = 0
def l₂ (m : ℝ) : ℝ → ℝ → Prop := λ x y ↦ (m - 2) * x + 3 * y + 2 * m = 0

-- Define what it means for two lines to be parallel
def parallel (m : ℝ) : Prop := lineslope 1 m = lineslope (m - 2) 3

-- State the theorem
theorem parallel_lines_m_values : 
  ∀ m : ℝ, parallel m ↔ m = -1 ∨ m = 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_m_values_l420_42038


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_tangent_circle_line_equation_special_case_triangle_area_range_l420_42027

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - y^2 = 1

-- Define the circle
def circle_O (x y : ℝ) : Prop := x^2 + y^2 = 2

-- Define the line
def line (k b x y : ℝ) : Prop := y = k * x + b

-- Define the foci
noncomputable def F1 : ℝ × ℝ := (-Real.sqrt 2, 0)
noncomputable def F2 : ℝ × ℝ := (Real.sqrt 2, 0)

-- Define the projection p
noncomputable def p (A B : ℝ × ℝ) : ℝ := sorry

-- Theorem 1: Relationship between b and k
theorem line_tangent_circle (k b : ℝ) :
  (∃ x y, line k b x y ∧ circle_O x y) →
  b^2 = 2 * (k^2 + 1) :=
by sorry

-- Theorem 2: Line equation when (OA · OB)p^2 = 1
theorem line_equation_special_case (k b : ℝ) (A B : ℝ × ℝ) :
  hyperbola A.1 A.2 →
  hyperbola B.1 B.2 →
  line k b A.1 A.2 →
  line k b B.1 B.2 →
  (A.1 * B.1 + A.2 * B.2) * (p A B)^2 = 1 →
  (k = Real.sqrt 2 ∧ b = Real.sqrt 6) ∨
  (k = Real.sqrt 2 ∧ b = -Real.sqrt 6) ∨
  (k = -Real.sqrt 2 ∧ b = Real.sqrt 6) ∨
  (k = -Real.sqrt 2 ∧ b = -Real.sqrt 6) :=
by sorry

-- Theorem 3: Range of triangle AOB area
theorem triangle_area_range (k b : ℝ) (A B : ℝ × ℝ) :
  hyperbola A.1 A.2 →
  hyperbola B.1 B.2 →
  line k b A.1 A.2 →
  line k b B.1 B.2 →
  2 ≤ (A.1 * B.1 + A.2 * B.2) * (p A B)^2 →
  (A.1 * B.1 + A.2 * B.2) * (p A B)^2 ≤ 4 →
  let area := (1/2) * Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) * Real.sqrt 2
  3 * Real.sqrt 10 ≤ area ∧ area ≤ 3 * Real.sqrt 34 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_tangent_circle_line_equation_special_case_triangle_area_range_l420_42027


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_charley_pencils_l420_42044

/-- The number of pencils Charley bought initially -/
def initial_pencils : ℕ → Prop := sorry

/-- The number of pencils Charley has now -/
def current_pencils : ℕ → Prop := sorry

theorem charley_pencils (P : ℕ) :
  initial_pencils P →
  current_pencils 16 →
  P - 6 - (P - 6) / 3 = 16 →
  P = 30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_charley_pencils_l420_42044


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_continuous_piecewise_function_l420_42030

-- Define the piecewise function f
noncomputable def f (b : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 2 then 4 * x^2 + 5 else b * x + 3

-- Theorem statement
theorem continuous_piecewise_function :
  ∀ b : ℝ, Continuous (f b) ↔ b = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_continuous_piecewise_function_l420_42030


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_reversed_base_numbers_l420_42047

/-- A function that returns the base 5 representation of a natural number as a list of digits -/
def toBase5 (n : ℕ) : List ℕ :=
  sorry

/-- A function that returns the base 8 representation of a natural number as a list of digits -/
def toBase8 (n : ℕ) : List ℕ :=
  sorry

/-- A function that reverses a list -/
def reverseList (l : List ℕ) : List ℕ :=
  sorry

/-- A function that checks if a natural number's base 5 representation is the reverse of its base 8 representation -/
def isReversedBase5and8 (n : ℕ) : Bool :=
  reverseList (toBase5 n) = toBase8 n

/-- The set of all positive integers whose base 5 representation is the reverse of their base 8 representation -/
def reversedBaseSet : Finset ℕ :=
  Finset.filter (fun n => n > 0 ∧ isReversedBase5and8 n) (Finset.range 38)

theorem sum_of_reversed_base_numbers :
  (Finset.sum reversedBaseSet id) = 37 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_reversed_base_numbers_l420_42047


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_finite_monoid_is_group_l420_42007

structure FiniteMonoid (M : Type) :=
  (mul : M → M → M)
  (one : M)
  (mul_assoc : ∀ a b c : M, mul (mul a b) c = mul a (mul b c))
  (one_mul : ∀ a : M, mul one a = a)
  (mul_one : ∀ a : M, mul a one = a)
  (finite : Finite M)

def pow {M : Type} (fm : FiniteMonoid M) (a : M) : ℕ → M
  | 0 => fm.one
  | n + 1 => fm.mul (pow fm a n) a

theorem finite_monoid_is_group
  (M : Type)
  (p : ℕ)
  (hp : p ≥ 2)
  (fm : FiniteMonoid M)
  (h : ∀ a : M, a ≠ fm.one → pow fm a p ≠ a) :
  Group M :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_finite_monoid_is_group_l420_42007


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_united_telephone_charge_l420_42095

/-- United Telephone's base rate -/
noncomputable def united_base : ℚ := 8

/-- Atlantic Call's base rate -/
noncomputable def atlantic_base : ℚ := 12

/-- Atlantic Call's per minute charge -/
noncomputable def atlantic_per_minute : ℚ := 1/5

/-- Number of minutes at which both companies charge the same -/
noncomputable def equal_minutes : ℚ := 80

/-- United Telephone's per minute charge -/
noncomputable def united_per_minute : ℚ := (atlantic_base - united_base + equal_minutes * atlantic_per_minute) / equal_minutes

theorem united_telephone_charge :
  united_per_minute = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_united_telephone_charge_l420_42095


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coupon1_best_at_219_95_l420_42033

-- Define the coupon discount functions
def coupon1_discount (price : ℝ) : ℝ := 0.12 * price

noncomputable def coupon2_discount (price : ℝ) : ℝ := 
  if price ≥ 120 then 25 else 0

def coupon3_discount (price : ℝ) : ℝ := 
  0.20 * (price - 120)

noncomputable def coupon4_discount (price : ℝ) : ℝ := 
  if price ≥ 200 then 35 else 0

-- Theorem statement
theorem coupon1_best_at_219_95 : 
  let price : ℝ := 219.95
  coupon1_discount price > coupon2_discount price ∧
  coupon1_discount price > coupon3_discount price ∧
  coupon1_discount price > coupon4_discount price :=
by
  -- Placeholder for the proof
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_coupon1_best_at_219_95_l420_42033


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l420_42013

noncomputable def f (x : ℝ) := Real.cos (2 * x) - 2 * Real.sqrt 3 * Real.sin x * Real.cos x

theorem f_properties :
  (∀ x₁ x₂ : ℝ, x₁ - x₂ = π → f x₁ = f x₂) ∧
  (∀ x : ℝ, f (π / 12 + x) = f (π / 12 - x)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l420_42013


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pipe_A_fill_time_l420_42085

/-- The time it takes for Pipe A to fill the tank without the leak -/
def T : ℝ := sorry

/-- The time it takes for Pipe A to fill the tank with the leak -/
def time_with_leak : ℝ := 4

/-- The time it takes for the leak to empty the full tank -/
def time_leak_empty : ℝ := 4

/-- Theorem stating that Pipe A can fill the tank without the leak in 2 hours -/
theorem pipe_A_fill_time :
  (1 / T - 1 / time_leak_empty = 1 / time_with_leak) → T = 2 := by
  sorry

#check pipe_A_fill_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pipe_A_fill_time_l420_42085


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_fourths_of_forty_l420_42020

theorem three_fourths_of_forty : (3 / 4 : ℚ) * 40 = 30 := by
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_fourths_of_forty_l420_42020


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l420_42045

theorem problem_solution (A B : ℕ) (h1 : Nat.lcm A B = 2310) (h2 : Nat.gcd A B = 30) (h3 : A = 770) : B = 90 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l420_42045


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mooncake_sales_analysis_l420_42061

-- Define the selling period
def selling_period : ℕ := 50

-- Define the purchase price
def purchase_price : ℝ := 18

-- Define the selling price function
noncomputable def selling_price (x : ℝ) : ℝ :=
  if 1 ≤ x ∧ x ≤ 30 then 40
  else if 31 ≤ x ∧ x ≤ 50 then -0.5 * x + 55
  else 0

-- Define the sales volume function
def sales_volume (x : ℝ) : ℝ := 5 * x + 50

-- Define the daily profit function
noncomputable def daily_profit (x : ℝ) : ℝ := (selling_price x - purchase_price) * sales_volume x

-- Define the adjusted daily profit function
noncomputable def adjusted_daily_profit (x a : ℝ) : ℝ := (selling_price x + a - purchase_price) * sales_volume x

theorem mooncake_sales_analysis :
  -- Part 1: Prove the linear relationship for 31 ≤ x ≤ 50
  (∀ x, 31 ≤ x ∧ x ≤ 50 → selling_price x = -0.5 * x + 55) ∧
  -- Part 2: Prove the maximum profit occurs at x = 32
  (∀ x, 1 ≤ x ∧ x ≤ 50 → daily_profit x ≤ daily_profit 32) ∧
  (daily_profit 32 = 4410) ∧
  -- Part 3: Prove the condition for increasing profit from day 31 to 35
  (∀ a, a > 2.5 → ∀ x y, 31 ≤ x ∧ x < y ∧ y ≤ 35 → adjusted_daily_profit x a < adjusted_daily_profit y a) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_mooncake_sales_analysis_l420_42061


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_equals_powerSeries_l420_42005

/-- The complex function f(z) = (z+1) / ((z-1)^2 * (z+2)) -/
noncomputable def f (z : ℂ) : ℂ := (z + 1) / ((z - 1)^2 * (z + 2))

/-- The n-th coefficient of the power series expansion of f(z) -/
noncomputable def a (n : ℕ) : ℂ := (1 / 9) * (6 * ↑n + 5 + (-1)^(n+1) / 2^(n+1))

/-- The power series ∑(n=0 to ∞) a(n) * z^n -/
noncomputable def powerSeries (z : ℂ) : ℂ := ∑' n, a n * z^n

/-- The theorem stating that f(z) equals the power series expansion for |z| < 1 -/
theorem f_equals_powerSeries :
  ∀ z : ℂ, Complex.abs z < 1 → f z = powerSeries z := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_equals_powerSeries_l420_42005


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_equivalent_equilateral_l420_42043

/-- A line focuses a triangle if the feet of perpendicular lines from a point in the plane to the edges of the triangle all lie on the line. -/
def focuses (d : Set EuclideanPlane) (T : Set EuclideanPlane) : Prop := sorry

/-- Two triangles are equivalent if the set of lines focusing them are the same. -/
def equivalent (T1 T2 : Set EuclideanPlane) : Prop :=
  ∀ d : Set EuclideanPlane, focuses d T1 ↔ focuses d T2

/-- A triangle is equilateral if all its sides have the same length. -/
def isEquilateral (T : Set EuclideanPlane) : Prop := sorry

/-- For any triangle in the plane, there exists a unique equilateral triangle equivalent to it. -/
theorem unique_equivalent_equilateral (T : Set EuclideanPlane) :
  ∃! E : Set EuclideanPlane, equivalent T E ∧ isEquilateral E := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_equivalent_equilateral_l420_42043


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_abs_sum_minus_sum_equals_ten_l420_42057

theorem abs_sum_minus_sum_equals_ten : 
  (abs (-5 : ℤ) + abs (3 : ℤ)) - ((-5 : ℤ) + (3 : ℤ)) = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_abs_sum_minus_sum_equals_ten_l420_42057


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bicycling_separation_time_l420_42002

/-- Adam's speed in miles per hour -/
noncomputable def adam_speed : ℝ := 10

/-- Simon's speed in miles per hour -/
noncomputable def simon_speed : ℝ := 7

/-- The distance they need to be apart in miles -/
noncomputable def target_distance : ℝ := 75

/-- Adam's southward component of speed -/
noncomputable def adam_south_speed : ℝ := adam_speed / Real.sqrt 2

/-- Theorem stating the existence of a time when Adam and Simon are 75 miles apart -/
theorem bicycling_separation_time :
  ∃ t : ℝ, t > 0 ∧ t * Real.sqrt (adam_south_speed^2 + simon_speed^2) = target_distance := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bicycling_separation_time_l420_42002


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_1990_2345_equals_3_l420_42037

def digit_sum (n : ℕ) : ℕ := 
  if n < 10 then n else n % 10 + digit_sum (n / 10)

def f₁ (n : ℕ) : ℕ :=
  let r := n % 3
  (digit_sum n)^2 + r + 1

def fₖ : ℕ → ℕ → ℕ
  | 0, n => n
  | 1, n => f₁ n
  | (k+1), n => f₁ (fₖ k n)

theorem f_1990_2345_equals_3 : fₖ 1990 2345 = 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_1990_2345_equals_3_l420_42037


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_directrix_l420_42035

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the points P and Q on the parabola
def P_on_parabola (x₁ y₁ : ℝ) : Prop := parabola x₁ y₁
def Q_on_parabola (x₂ y₂ : ℝ) : Prop := parabola x₂ y₂

-- Define the condition x₁ + x₂ = 6
def sum_of_x (x₁ x₂ : ℝ) : Prop := x₁ + x₂ = 6

-- Define the midpoint M
noncomputable def midpoint_M (x₁ x₂ y₁ y₂ : ℝ) : ℝ × ℝ := ((x₁ + x₂)/2, (y₁ + y₂)/2)

-- Define the directrix
def directrix : ℝ := -1

-- Theorem statement
theorem distance_to_directrix 
  (x₁ x₂ y₁ y₂ : ℝ) 
  (h1 : P_on_parabola x₁ y₁) 
  (h2 : Q_on_parabola x₂ y₂) 
  (h3 : sum_of_x x₁ x₂) :
  let M := midpoint_M x₁ x₂ y₁ y₂
  (M.1 - directrix) = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_directrix_l420_42035


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_relation_l420_42078

noncomputable def f (n : ℤ) : ℝ :=
  (7 + 4 * Real.sqrt 7) / 14 * ((1 + Real.sqrt 7) / 2) ^ n +
  (7 - 4 * Real.sqrt 7) / 14 * ((1 - Real.sqrt 7) / 2) ^ n

theorem f_relation (n : ℤ) : f (n + 1) - f (n - 1) = (3 * Real.sqrt 7 / 14) * f n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_relation_l420_42078


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_polynomial_property_l420_42060

/-- Represents a cubic polynomial of the form 3x³ + dx² + ex + 9 -/
structure CubicPolynomial where
  d : ℝ
  e : ℝ

/-- The sum of the roots of the polynomial -/
noncomputable def sum_of_roots (p : CubicPolynomial) : ℝ := -p.d / 3

/-- The product of the roots of the polynomial -/
def product_of_roots : ℝ := -3

/-- The sum of the coefficients of the polynomial -/
def sum_of_coefficients (p : CubicPolynomial) : ℝ := 3 + p.d + p.e + 9

/-- Theorem stating that if the reciprocal of the sum of roots, 
    the product of roots, and the sum of coefficients are equal, 
    then e = -16 -/
theorem cubic_polynomial_property (p : CubicPolynomial) :
  (1 / sum_of_roots p = product_of_roots) ∧ 
  (product_of_roots = sum_of_coefficients p) →
  p.e = -16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_polynomial_property_l420_42060


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_function_a_range_l420_42066

-- Define the function y
noncomputable def y (a x : ℝ) : ℝ := Real.log (3 - a * x) / Real.log a

-- Define the derivative of y with respect to x
noncomputable def y_derivative (a x : ℝ) : ℝ := -a / ((3 - a * x) * Real.log a)

theorem decreasing_function_a_range :
  ∀ a : ℝ, 
    (∀ x ∈ Set.Icc 0 1, y_derivative a x < 0) → 
    (0 < a ∧ a < 3) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_function_a_range_l420_42066


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_card_labels_equal_l420_42070

theorem card_labels_equal (labels : Fin 99 → ℕ) 
  (h1 : ∀ i, labels i ∈ Finset.range 100 \ {0})
  (h2 : ∀ s : Finset (Fin 99), s.card > 0 → (s.sum labels) % 100 ≠ 0) :
  ∀ i j, labels i = labels j :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_card_labels_equal_l420_42070


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fibonacci_equation_solution_l420_42093

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

/-- The golden ratio -/
noncomputable def φ : ℝ := (1 + Real.sqrt 5) / 2

theorem fibonacci_equation_solution :
  ∀ x : ℝ, x^2024 = (fib 2023 : ℝ) * x + (fib 2022 : ℝ) ↔ x = φ ∨ x = 1 - φ :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fibonacci_equation_solution_l420_42093


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_simplification_l420_42072

theorem expression_simplification (k : ℤ) :
  (2 : ℝ)^(-(2*k+1)) + (2 : ℝ)^(-(2*k-1)) - (2 : ℝ)^(-2*k) + (2 : ℝ)^(-(2*k+2)) = (7/4) * (2 : ℝ)^(-2*k) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_simplification_l420_42072


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_longest_side_length_l420_42016

/-- A triangle with specific medians and perimeter --/
structure SpecialTriangle where
  -- Vertices of the triangle
  a : ℝ × ℝ
  b : ℝ × ℝ
  c : ℝ × ℝ
  -- Conditions for medians
  median1 : (a.1 + b.1) / 2 = (a.2 + b.2) / 2
  median2 : (b.1 + c.1) / 2 = 2 * ((b.2 + c.2) / 2)
  median3 : (c.1 + a.1) / 2 = 3 * ((c.2 + a.2) / 2)
  -- Perimeter condition
  perimeter : Real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2) +
              Real.sqrt ((b.1 - c.1)^2 + (b.2 - c.2)^2) +
              Real.sqrt ((c.1 - a.1)^2 + (c.2 - a.2)^2) = 1

/-- The length of the longest side of the special triangle --/
noncomputable def longestSide (t : SpecialTriangle) : ℝ :=
  max (Real.sqrt ((t.a.1 - t.b.1)^2 + (t.a.2 - t.b.2)^2))
      (max (Real.sqrt ((t.b.1 - t.c.1)^2 + (t.b.2 - t.c.2)^2))
           (Real.sqrt ((t.c.1 - t.a.1)^2 + (t.c.2 - t.a.2)^2)))

/-- Theorem stating the length of the longest side of the special triangle --/
theorem longest_side_length (t : SpecialTriangle) :
  longestSide t = Real.sqrt (Real.sqrt 58 / (2 + Real.sqrt 34 + Real.sqrt 58)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_longest_side_length_l420_42016


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_center_and_min_tangent_l420_42063

/-- Line l defined by parametric equations -/
noncomputable def line_l (t : ℝ) : ℝ × ℝ :=
  ((Real.sqrt 2 / 2) * t, (Real.sqrt 2 / 2) * t + 4 * Real.sqrt 2)

/-- Circle C defined by polar equation -/
noncomputable def circle_C (θ : ℝ) : ℝ :=
  4 * Real.cos (θ + Real.pi / 4)

/-- The center of circle C -/
noncomputable def center_C : ℝ × ℝ :=
  (Real.sqrt 2, -Real.sqrt 2)

/-- The minimum length of tangents from line l to circle C -/
noncomputable def min_tangent_length : ℝ :=
  4 * Real.sqrt 2

theorem circle_center_and_min_tangent :
  (∀ θ, circle_C θ = ‖center_C - (circle_C θ * Real.cos θ, circle_C θ * Real.sin θ)‖) ∧
  (∀ t, ‖line_l t - center_C‖ ≥ 2 → 
    ‖line_l t - center_C‖ - 2 ≥ min_tangent_length) := by
  sorry

#check circle_center_and_min_tangent

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_center_and_min_tangent_l420_42063


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_difference_in_factorial_factors_l420_42024

def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

theorem smallest_difference_in_factorial_factors : ∃ (a b c : ℕ+), 
  (a.val * b.val * c.val = factorial 9) ∧ 
  (a < b) ∧ (b < c) ∧
  (∀ (x y z : ℕ+), (x.val * y.val * z.val = factorial 9) → (x < y) → (y < z) → (c.val - a.val ≤ z.val - x.val)) ∧
  (c.val - a.val = 262) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_difference_in_factorial_factors_l420_42024


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_proposition_l420_42074

theorem negation_of_proposition :
  (¬ ∀ x : ℝ, x > 0 → Real.sin x ≥ -1) ↔ (∃ x : ℝ, x > 0 ∧ Real.sin x < -1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_proposition_l420_42074


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_larger_square_l420_42054

/-- Represents a grasshopper's position -/
structure Grasshopper where
  x : Int
  y : Int

/-- Represents the state of all four grasshoppers -/
structure GrasshopperState where
  g1 : Grasshopper
  g2 : Grasshopper
  g3 : Grasshopper
  g4 : Grasshopper

/-- Checks if four points form a square -/
def isSquare (s : GrasshopperState) : Bool := sorry

/-- Performs a symmetric jump for one grasshopper -/
def symmetricJump (s : GrasshopperState) : GrasshopperState := sorry

/-- Initial state with grasshoppers at unit square vertices -/
def initialState : GrasshopperState := {
  g1 := { x := 0, y := 0 },
  g2 := { x := 1, y := 0 },
  g3 := { x := 1, y := 1 },
  g4 := { x := 0, y := 1 }
}

/-- Calculates the maximum distance between any two grasshoppers -/
def maxDistance (s : GrasshopperState) : Int := sorry

/-- Main theorem: Grasshoppers cannot form a larger square -/
theorem no_larger_square (n : Nat) :
  ∀ state, state = (Nat.iterate symmetricJump n initialState) →
  isSquare state → (maxDistance state ≤ 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_larger_square_l420_42054


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_messages_received_by_keith_l420_42062

theorem messages_received_by_keith (juan_to_laurence : ℚ) 
  (h1 : juan_to_laurence > 0) 
  (h2 : 8 * juan_to_laurence = 32) 
  (h3 : 4.5 * juan_to_laurence = 18) : 
  32 = 8 * juan_to_laurence := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_messages_received_by_keith_l420_42062


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_equation_implies_expression_l420_42010

theorem cos_equation_implies_expression (α : ℝ) :
  Real.cos (5 * π / 12 - α) = Real.sqrt 2 / 3 →
  Real.sqrt 3 * Real.cos (2 * α) - Real.sin (2 * α) = 10 / 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_equation_implies_expression_l420_42010


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_trig_sum_min_value_trig_sum_achievable_l420_42031

noncomputable section

open Real

theorem min_value_trig_sum (x : ℝ) :
  |sin x + cos x + tan x + (1 / tan x) + (1 / cos x) + (1 / sin x)| ≥ 2 * sqrt 2 - 1 :=
sorry

theorem min_value_trig_sum_achievable :
  ∃ x : ℝ, |sin x + cos x + tan x + (1 / tan x) + (1 / cos x) + (1 / sin x)| = 2 * sqrt 2 - 1 :=
sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_trig_sum_min_value_trig_sum_achievable_l420_42031


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extremum_point_implies_a_zero_l420_42001

/-- The function f(x) defined in the problem -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - (1/2) * a * x^2 - x

/-- The derivative of f(x) with respect to x -/
noncomputable def f_derivative (a : ℝ) (x : ℝ) : ℝ := 1/x - a*x - 1

theorem extremum_point_implies_a_zero :
  ∀ a : ℝ, (f_derivative a 1 = 0) → a = 0 := by
  intro a h
  -- Proof steps would go here
  sorry

#check extremum_point_implies_a_zero

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extremum_point_implies_a_zero_l420_42001


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l420_42090

-- Define the circle
def circle_eq (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 4

-- Define the center of the circle
def center : ℝ × ℝ := (-1, 0)

-- Define the line l
def line_eq (m x y : ℝ) : Prop := m * x - y - 5 * m + 4 = 0

-- Define the angle between two vectors
noncomputable def angle_between_vectors (v1 v2 : ℝ × ℝ) : ℝ :=
  Real.arccos ((v1.1 * v2.1 + v1.2 * v2.2) / (Real.sqrt (v1.1^2 + v1.2^2) * Real.sqrt (v2.1^2 + v2.2^2)))

-- Define the condition that there exists a point Q on the circle such that ∠CPQ = 30°
def angle_condition (m : ℝ) : Prop := ∃ (x y : ℝ), circle_eq x y ∧ line_eq m x y ∧ 
  ∃ (qx qy : ℝ), circle_eq qx qy ∧ (angle_between_vectors (qx - center.1, qy - center.2) (x - center.1, y - center.2) = 30 * Real.pi / 180)

-- The theorem to prove
theorem range_of_m : 
  ∀ m : ℝ, angle_condition m → 0 ≤ m ∧ m ≤ 12/5 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l420_42090


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_existence_l420_42042

theorem intersection_existence (a : ℕ) (h : a ≥ 2) :
  ∃! (b₁ b₂ : ℝ), b₁ ∈ Set.Icc (1 : ℝ) a ∧ b₂ ∈ Set.Icc (1 : ℝ) a ∧ b₁ ≠ b₂ ∧
  (∃ (y : ℝ), y ∈ {y : ℝ | ∃ x : ℕ, y = (a : ℝ)^x} ∩ {y : ℝ | ∃ x : ℕ, y = (a + 1 : ℝ) * x + b₁}) ∧
  (∃ (y : ℝ), y ∈ {y : ℝ | ∃ x : ℕ, y = (a : ℝ)^x} ∩ {y : ℝ | ∃ x : ℕ, y = (a + 1 : ℝ) * x + b₂}) ∧
  b₁ = 1 ∧ b₂ = a :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_existence_l420_42042


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_picking_cheapest_l420_42064

/-- Represents the cost and yield of making jam -/
structure JamMaking where
  ticket_cost : ℚ
  berry_collected : ℚ
  market_berry_price : ℚ
  sugar_price : ℚ
  jam_yield : ℚ
  ready_jam_price : ℚ

/-- Calculates the cost of making jam by picking berries -/
def cost_picking (j : JamMaking) : ℚ :=
  j.ticket_cost / j.berry_collected + j.sugar_price

/-- Calculates the cost of making jam by buying berries -/
def cost_buying (j : JamMaking) : ℚ :=
  j.market_berry_price + j.sugar_price

/-- Calculates the cost of buying ready-made jam -/
def cost_ready (j : JamMaking) : ℚ :=
  j.ready_jam_price * j.jam_yield

/-- Theorem: Picking berries is the cheapest option -/
theorem picking_cheapest (j : JamMaking) 
  (h1 : j.ticket_cost = 200)
  (h2 : j.berry_collected = 5)
  (h3 : j.market_berry_price = 150)
  (h4 : j.sugar_price = 54)
  (h5 : j.jam_yield = 3/2)
  (h6 : j.ready_jam_price = 220) :
  cost_picking j < cost_buying j ∧ cost_picking j < cost_ready j :=
by
  -- Unfold definitions
  unfold cost_picking cost_buying cost_ready
  -- Split the goal into two parts
  apply And.intro
  -- Prove cost_picking < cost_buying
  · -- Simplify expressions
    simp [h1, h2, h3, h4]
    -- Prove the inequality
    norm_num
  -- Prove cost_picking < cost_ready
  · -- Simplify expressions
    simp [h1, h2, h4, h5, h6]
    -- Prove the inequality
    norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_picking_cheapest_l420_42064


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_correction_sum_l420_42079

/-- Represents a single-digit number -/
def SingleDigit := {n : ℕ // n < 10}

/-- Checks if four numbers are all different -/
def all_different (a b c d : SingleDigit) : Prop :=
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d

/-- Checks if the equation is incorrect -/
def is_incorrect_equation (a b c d : SingleDigit) : Prop :=
  a.val * b.val ≠ c.val * 10 + d.val

/-- Checks if there are exactly 3 ways to correct the equation by changing one digit -/
def has_three_corrections (a b c d : SingleDigit) : Prop :=
  ∃ (x y z : SingleDigit), 
    (x ≠ a ∧ x.val * b.val = c.val * 10 + d.val) ∨
    (y ≠ b ∧ a.val * y.val = c.val * 10 + d.val) ∨
    (z ≠ c ∧ a.val * b.val = z.val * 10 + d.val) ∨
    (z ≠ d ∧ a.val * b.val = c.val * 10 + z.val)

/-- Checks if the equation can be corrected by changing the order of digits -/
def can_correct_by_reordering (a b c d : SingleDigit) : Prop :=
  ∃ (p q r s : SingleDigit), (p = a ∨ p = b ∨ p = c ∨ p = d) ∧
                             (q = a ∨ q = b ∨ q = c ∨ q = d) ∧
                             (r = a ∨ r = b ∨ r = c ∨ r = d) ∧
                             (s = a ∨ s = b ∨ s = c ∨ s = d) ∧
                             p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s ∧
                             p.val * q.val = r.val * 10 + s.val

theorem equation_correction_sum (a b c d : SingleDigit) :
  all_different a b c d →
  is_incorrect_equation a b c d →
  has_three_corrections a b c d →
  can_correct_by_reordering a b c d →
  a.val + b.val + c.val + d.val = 17 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_correction_sum_l420_42079


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_m_range_for_increasing_f_l420_42018

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := Real.log ((4 * x^2 + m) / x) / Real.log m

theorem m_range_for_increasing_f :
  ∀ m : ℝ, m > 0 → m ≠ 1 →
  (∀ x y : ℝ, 2 ≤ x → x < y → y ≤ 3 → f m x < f m y) →
  1 < m ∧ m ≤ 16 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_m_range_for_increasing_f_l420_42018


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_root_sixteen_to_eighth_l420_42006

theorem fourth_root_sixteen_to_eighth : (16 : ℝ) ^ ((1/4 : ℝ) * 8) = 256 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_root_sixteen_to_eighth_l420_42006


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_in_second_quadrant_l420_42084

/-- The complex number z -/
noncomputable def z : ℂ := Complex.I / (1 - Complex.I)

/-- A complex number is in the second quadrant if its real part is negative and its imaginary part is positive -/
def is_in_second_quadrant (w : ℂ) : Prop :=
  w.re < 0 ∧ w.im > 0

/-- Theorem: z is located in the second quadrant of the complex plane -/
theorem z_in_second_quadrant : is_in_second_quadrant z := by
  -- Simplify z
  have h : z = -1/2 + Complex.I/2 := by
    sorry
  -- Show that the real part is negative and the imaginary part is positive
  have h_re : z.re < 0 := by
    sorry
  have h_im : z.im > 0 := by
    sorry
  -- Conclude that z is in the second quadrant
  exact ⟨h_re, h_im⟩


end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_in_second_quadrant_l420_42084


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_focus_coordinates_l420_42040

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse -/
structure Ellipse where
  center : Point
  majorAxis : (Point × Point)
  minorAxis : (Point × Point)

/-- The focus of an ellipse with larger y-coordinate -/
noncomputable def focusWithLargerY (e : Ellipse) : Point :=
  { x := e.center.x, y := Real.sqrt 5 }

theorem ellipse_focus_coordinates (e : Ellipse) 
  (h1 : e.center = { x := 3, y := 0 })
  (h2 : e.majorAxis = ({ x := 1, y := 0 }, { x := 5, y := 0 }))
  (h3 : e.minorAxis = ({ x := 3, y := 3 }, { x := 3, y := -3 })) :
  focusWithLargerY e = { x := 3, y := Real.sqrt 5 } := by
  sorry

#check ellipse_focus_coordinates

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_focus_coordinates_l420_42040


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_equal_angle_points_l420_42069

/-- Given a rectangle ABCD with AB > BC, AD = m, and AB = n, 
    the points P on AB where ∠APD = ∠DPC are given by AP = n ± √(n² - m²) -/
theorem rectangle_equal_angle_points (m n : ℝ) (h1 : 0 < m) (h2 : m < n) : 
  ∃ x : ℝ, (x = n + Real.sqrt (n^2 - m^2) ∨ x = n - Real.sqrt (n^2 - m^2)) ∧
  Real.tan (Real.arctan (m / x)) = Real.tan (Real.arctan (m / (n - x))) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_equal_angle_points_l420_42069


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_real_sum_w_l420_42098

/-- The zeros of the polynomial z^10 - 2^30 -/
noncomputable def zeros : Finset ℂ := sorry

/-- w_j is either z_j or -iz_j for each z_j in zeros -/
noncomputable def w (j : ℕ) : ℂ := sorry

/-- The sum of w_j from j=1 to 10 -/
noncomputable def sum_w : ℂ := sorry

/-- The maximum possible value of the real part of sum_w -/
theorem max_real_sum_w :
  ∃ (x : ℝ), ∀ (y : ℝ), (sum_w.re : ℝ) ≤ x ∧
    x = 8 * (1 + 2 * (Real.cos (2 * Real.pi / 10) + Real.cos (4 * Real.pi / 10))) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_real_sum_w_l420_42098


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_equals_interval_l420_42081

-- Define sets A and B
def A : Set ℝ := {x | x^2 < 8}
def B : Set ℝ := {x | 1 - x ≤ 0}

-- Define the intersection of A and B
def AIntersectB : Set ℝ := A ∩ B

-- Theorem stating that the intersection is equal to [1, 2√2)
theorem intersection_equals_interval : 
  AIntersectB = Set.Icc 1 (Real.sqrt 8) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_equals_interval_l420_42081


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_angle_problem_l420_42036

theorem vector_angle_problem (α β : ℝ) (h1 : 0 < α ∧ α < π / 2) (h2 : 0 < β ∧ β < π / 2) 
  (h3 : (Real.cos α - Real.cos β)^2 + (Real.sin α + Real.sin β)^2 = 2/5) (h4 : Real.cos α = 3/5) :
  Real.cos (α + β) = 4/5 ∧ Real.cos β = 24/25 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_angle_problem_l420_42036


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_is_45π_div_2_l420_42028

/-- Given points A, B, C, D, E, F on a straight line with equal segments except AF -/
structure PointConfiguration where
  AB : ℝ
  BC : ℝ
  CD : ℝ
  DE : ℝ
  EF : ℝ
  AF : ℝ
  equal_segments : AB = BC ∧ BC = CD ∧ CD = DE ∧ DE = EF
  segment_length : AB = 3
  af_length : AF = 15

/-- Calculate the area of a semicircle given its diameter -/
noncomputable def semicircle_area (diameter : ℝ) : ℝ :=
  (Real.pi / 8) * diameter ^ 2

/-- Calculate the shaded area given a point configuration -/
noncomputable def shaded_area (config : PointConfiguration) : ℝ :=
  semicircle_area config.AF - 
  (semicircle_area config.AB + semicircle_area config.BC + 
   semicircle_area config.CD + semicircle_area config.DE + 
   semicircle_area config.EF)

/-- The main theorem stating that the shaded area is 45π/2 -/
theorem shaded_area_is_45π_div_2 (config : PointConfiguration) : 
  shaded_area config = 45 * Real.pi / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_is_45π_div_2_l420_42028


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_power_function_value_l420_42026

-- Define the power function as noncomputable
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := x^(2+m)

-- Define the theorem
theorem odd_power_function_value (m : ℝ) :
  (∀ x ∈ Set.Icc (-1) m, f m x = -(f m (-x))) →  -- f is odd on [-1,m]
  f m (m+1) = -1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_power_function_value_l420_42026


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_bite_waiting_time_l420_42073

/-- Represents the average number of bites for a fishing rod in a given time interval -/
structure FishingRod where
  bites : ℚ
  interval : ℚ

/-- Calculates the average waiting time for the first bite given two fishing rods -/
noncomputable def averageWaitingTime (rod1 rod2 : FishingRod) : ℚ :=
  rod1.interval / (rod1.bites + rod2.bites)

theorem first_bite_waiting_time :
  let rod1 : FishingRod := { bites := 3, interval := 6 }
  let rod2 : FishingRod := { bites := 2, interval := 6 }
  averageWaitingTime rod1 rod2 = 6/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_bite_waiting_time_l420_42073


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dilation_problem_l420_42082

/-- Dilation of a complex number -/
noncomputable def dilation (center : ℂ) (scale : ℝ) (z : ℂ) : ℂ :=
  center + scale • (z - center)

/-- The problem statement -/
theorem dilation_problem : 
  dilation (-1 + 4*I) (-2 : ℝ) (2*I) = -3 + 8*I := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dilation_problem_l420_42082


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bicycle_ride_distance_l420_42076

/-- A bicycle ride with two parts -/
structure BicycleRide where
  part1_distance : ℝ
  part1_speed : ℝ
  part2_distance : ℝ
  part2_speed : ℝ
  average_speed : ℝ

/-- The theorem stating the conditions and the result to be proved -/
theorem bicycle_ride_distance (ride : BicycleRide)
  (h1 : ride.part1_speed = 12)
  (h2 : ride.part2_distance = 12)
  (h3 : ride.part2_speed = 10)
  (h4 : ride.average_speed = 10.82) :
  ∃ ε > 0, |ride.part1_distance - 10| < ε := by
  sorry

#check bicycle_ride_distance

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bicycle_ride_distance_l420_42076


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_seth_orange_boxes_l420_42080

/-- The number of boxes Seth bought initially -/
def initial_boxes : ℕ := sorry

/-- The number of boxes Seth has left -/
def remaining_boxes : ℕ := 4

/-- Seth gave one box to his mother -/
def box_to_mother : ℕ := 1

theorem seth_orange_boxes : 
  initial_boxes = (remaining_boxes * 2 + box_to_mother) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_seth_orange_boxes_l420_42080


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_cost_20_oranges_is_70_l420_42011

/-- Represents the cost in cents for a given number of oranges -/
def BulkOption : Type := Nat × Nat

/-- Calculates the cost of buying a given number of oranges using a specific bulk option -/
def cost_with_option (option : BulkOption) (num_oranges : Nat) : Nat :=
  (num_oranges / option.1) * option.2

/-- Applies the discount if applicable -/
def apply_discount (total_cost : Nat) (num_oranges : Nat) : Nat :=
  if num_oranges ≥ 20 then total_cost - 5 else total_cost

/-- Calculates the minimum cost for 20 oranges given a list of bulk options -/
def min_cost_20_oranges (bulk_options : List BulkOption) : Nat :=
  let costs := bulk_options.map (fun opt => cost_with_option opt 20)
  let min_cost := costs.foldl min (costs.head!)
  apply_discount min_cost 20

#eval min_cost_20_oranges [(4, 15), (6, 25), (10, 40)]

theorem min_cost_20_oranges_is_70 (bulk_options : List BulkOption) :
  min_cost_20_oranges bulk_options = 70 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_cost_20_oranges_is_70_l420_42011


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_sum_reciprocal_log_expression_product_l420_42068

-- Part 1
theorem sqrt_sum_reciprocal (a : ℝ) (h : Real.sqrt a + 1 / Real.sqrt a = 3) :
  (a^2 + 1/a^2 + 3) / (4*a + 1/(4*a)) = 10 * Real.sqrt 5 := by sorry

-- Part 2
theorem log_expression_product :
  ((1 - Real.log 3 / Real.log 6)^2 + (Real.log 2 / Real.log 6) * (Real.log 18 / Real.log 6)) *
  (Real.log 6 / Real.log 4) = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_sum_reciprocal_log_expression_product_l420_42068


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_ratio_l420_42015

/-- For a geometric sequence with first term a and common ratio q, S_n is the sum of first n terms -/
noncomputable def S (a : ℝ) (q : ℝ) (n : ℕ) : ℝ := a * (1 - q^n) / (1 - q)

/-- Theorem: If S₃ + 3S₂ = 0 for a geometric sequence, then the common ratio q = -2 -/
theorem geometric_sequence_ratio (a : ℝ) (q : ℝ) (h1 : a ≠ 0) (h2 : q ≠ 1) :
  S a q 3 + 3 * S a q 2 = 0 → q = -2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_ratio_l420_42015


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_last_two_nonzero_digits_70_factorial_eq_44_l420_42003

/-- The last two nonzero digits of 70! -/
def last_two_nonzero_digits_70_factorial : Nat :=
  Nat.factorial 70 % 100

/-- Theorem stating that the last two nonzero digits of 70! are equal to 44 -/
theorem last_two_nonzero_digits_70_factorial_eq_44 :
  last_two_nonzero_digits_70_factorial = 44 := by
  sorry

#eval last_two_nonzero_digits_70_factorial

end NUMINAMATH_CALUDE_ERRORFEEDBACK_last_two_nonzero_digits_70_factorial_eq_44_l420_42003


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_bisector_d_value_l420_42009

/-- The perpendicular bisector of a line segment passes through its midpoint 
    and is perpendicular to the segment. --/
def is_perp_bisector (l : ℝ → ℝ → Prop) (p₁ p₂ : ℝ × ℝ) : Prop :=
  let midpoint := ((p₁.1 + p₂.1) / 2, (p₁.2 + p₂.2) / 2)
  l midpoint.1 midpoint.2 ∧ 
  ∃ m : ℝ, (∀ x y, l x y ↔ y = m * x + (midpoint.2 - m * midpoint.1)) ∧
           (p₂.1 - p₁.1) * m + (p₂.2 - p₁.2) = 0

/-- The line equation 2x - y = d --/
def line_equation (d : ℝ) (x y : ℝ) : Prop :=
  2 * x - y = d

theorem perpendicular_bisector_d_value :
  ∃ d : ℝ, is_perp_bisector (line_equation d) (-2, 4) (6, 0) ∧ d = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_bisector_d_value_l420_42009


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_school_travel_time_l420_42022

/-- Given a boy who reaches school 3 minutes early when walking at 7/6 of his usual rate,
    his usual time to reach the school is 21 minutes. -/
theorem school_travel_time (usual_rate : ℝ) (usual_time : ℝ) 
    (h1 : usual_rate > 0) (h2 : usual_time > 0) :
  (7 / 6 * usual_rate) * (usual_time - 3) = usual_rate * usual_time →
  usual_time = 21 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_school_travel_time_l420_42022


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_centroid_distance_ratio_l420_42000

/-- Given a triangle ABC with side lengths a, b, c and distances from 
    centroid to vertices l, m, n, the ratio of the sum of squared side 
    lengths to the sum of squared centroid distances is 6. -/
theorem triangle_centroid_distance_ratio 
  (a b c l m n : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hl : l > 0) (hm : m > 0) (hn : n > 0)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_centroid : l = (b + c) / 3 ∧ m = (c + a) / 3 ∧ n = (a + b) / 3) :
  (a^2 + b^2 + c^2) / (l^2 + m^2 + n^2) = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_centroid_distance_ratio_l420_42000


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_amount_equals_expected_l420_42097

/-- Calculates the total amount after two years of investment with given conditions -/
noncomputable def total_amount (P : ℝ) : ℝ :=
  let first_year_rate := 0.05
  let second_year_rate := 0.06
  let additional_investment := P / 2
  let first_year_amount := P * (1 + first_year_rate / 2)^2
  let total_after_additional := first_year_amount + additional_investment
  total_after_additional * (1 + second_year_rate / 2)^2

/-- Theorem stating that the total amount after two years is equal to 1.645187625P -/
theorem total_amount_equals_expected (P : ℝ) :
  total_amount P = 1.645187625 * P := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_amount_equals_expected_l420_42097


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_representation_l420_42053

noncomputable def ω : ℂ := (-1 + Complex.I * Real.sqrt 3) / 2

def M : Set ℂ := {-3, -3 * ω, -3 * ω^2}

def MyPolynomial := ℂ → ℂ → ℂ → ℂ

def P (m : ℂ) : MyPolynomial := fun x y z => x^3 + y^3 + z^3 + m*x*y*z

def isProductOfLinearTrinomials (p : MyPolynomial) : Prop :=
  ∃ (a b c d e f g h i : ℂ),
    ∀ x y z, p x y z = (x + a*y + b*z) * (x + c*y + d*z) * (x + e*y + f*z)

theorem polynomial_representation (m : ℂ) :
  isProductOfLinearTrinomials (P m) ↔ m ∈ M :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_representation_l420_42053


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_C_standard_form_length_AB_when_theta_pi_4_range_PA_PB_product_l420_42034

noncomputable def line_l (t θ : ℝ) : ℝ × ℝ := (1 + t * Real.cos θ, t * Real.sin θ)

noncomputable def curve_C (α : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.cos α, Real.sin α)

def point_P : ℝ × ℝ := (1, 0)

theorem curve_C_standard_form :
  ∀ x y : ℝ, (∃ α : ℝ, curve_C α = (x, y)) ↔ x^2 / 3 + y^2 = 1 := by sorry

theorem length_AB_when_theta_pi_4 :
  ∃ A B : ℝ × ℝ,
    (∃ t₁ t₂ : ℝ, line_l t₁ (π/4) = A ∧ line_l t₂ (π/4) = B) ∧
    (∃ α₁ α₂ : ℝ, curve_C α₁ = A ∧ curve_C α₂ = B) ∧
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = (3/2) * Real.sqrt 2 := by sorry

theorem range_PA_PB_product :
  ∀ θ : ℝ,
    ∃ A B : ℝ × ℝ,
      (∃ t₁ t₂ : ℝ, line_l t₁ θ = A ∧ line_l t₂ θ = B) ∧
      (∃ α₁ α₂ : ℝ, curve_C α₁ = A ∧ curve_C α₂ = B) ∧
      (2/3 ≤ (Real.sqrt ((A.1 - point_P.1)^2 + (A.2 - point_P.2)^2)) *
              (Real.sqrt ((B.1 - point_P.1)^2 + (B.2 - point_P.2)^2)) ∧
       (Real.sqrt ((A.1 - point_P.1)^2 + (A.2 - point_P.2)^2)) *
       (Real.sqrt ((B.1 - point_P.1)^2 + (B.2 - point_P.2)^2)) ≤ 2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_C_standard_form_length_AB_when_theta_pi_4_range_PA_PB_product_l420_42034


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_distance_through_point_l420_42012

/-- The total distance traveled from (-3, 4) to (6, -5) passing through (1, 1) is 5 + √61 -/
theorem total_distance_through_point :
  let start : ℝ × ℝ := (-3, 4)
  let middle : ℝ × ℝ := (1, 1)
  let end_point : ℝ × ℝ := (6, -5)
  let distance (p q : ℝ × ℝ) := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  distance start middle + distance middle end_point = 5 + Real.sqrt 61 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_distance_through_point_l420_42012


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_fixed_point_l420_42041

noncomputable section

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2/3 + y^2/4 = 1

-- Define points A, B, and P
def A : ℝ × ℝ := (0, -2)
def B : ℝ × ℝ := (3/2, -1)
def P : ℝ × ℝ := (1, -2)

-- Define line AB
def line_AB (x y : ℝ) : Prop := y = 2/3 * x - 2

-- Theorem statement
theorem ellipse_fixed_point :
  ∀ (M N T H : ℝ × ℝ),
    ellipse M.1 M.2 →
    ellipse N.1 N.2 →
    (∃ k : ℝ, k * (M.1 - P.1) = M.2 - P.2 ∧ k * (N.1 - P.1) = N.2 - P.2) →
    line_AB T.1 T.2 →
    M.2 = T.2 →
    H.1 - T.1 = T.1 - M.1 ∧ H.2 - T.2 = T.2 - M.2 →
    ∃ t : ℝ, H.1 + t * (N.1 - H.1) = 0 ∧ H.2 + t * (N.2 - H.2) = -2 :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_fixed_point_l420_42041


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_lines_through_P_l420_42021

-- Define the circle
def my_circle (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define the point P
def P : ℝ × ℝ := (-2, -3)

-- Define the condition that P is outside the circle
def P_outside_circle : Prop := 
  let (x, y) := P
  x^2 + y^2 > 4

-- Define the tangent line equations
def tangent_line_1 (x : ℝ) : Prop := x = -2
def tangent_line_2 (x y : ℝ) : Prop := 5*x - 12*y - 26 = 0

-- Theorem statement
theorem tangent_lines_through_P : 
  ∀ x y : ℝ, my_circle x y → P_outside_circle → 
  (∃ t : ℝ, (x = t ∧ y = -3) ∨ (x = -2 ∧ y = t)) →
  (tangent_line_1 x ∨ tangent_line_2 x y) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_lines_through_P_l420_42021


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_carlos_finishes_first_l420_42004

-- Define the lawn areas and mowing rates
noncomputable def lawn_area (person : String) : ℝ := sorry

noncomputable def mowing_rate (person : String) : ℝ := sorry

-- State the conditions
axiom andy_beth_equal : lawn_area "Andy" = lawn_area "Beth"
axiom andy_carlos_ratio : lawn_area "Andy" = 3 * lawn_area "Carlos"
axiom carlos_andy_rate : mowing_rate "Carlos" = mowing_rate "Andy"
axiom beth_andy_rate : mowing_rate "Beth" = (1/2) * mowing_rate "Andy"

-- Define the mowing time function
noncomputable def mowing_time (person : String) : ℝ := lawn_area person / mowing_rate person

-- Theorem to prove
theorem carlos_finishes_first :
  mowing_time "Carlos" < mowing_time "Andy" ∧
  mowing_time "Carlos" < mowing_time "Beth" :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_carlos_finishes_first_l420_42004


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_john_meets_train_probability_l420_42048

/-- The duration in minutes that the train can arrive after 2:00 PM -/
noncomputable def trainArrivalWindow : ℝ := 60

/-- The duration in minutes that John can arrive after 2:00 PM -/
noncomputable def johnArrivalWindow : ℝ := 120

/-- The duration in minutes that the train stays at the station -/
noncomputable def trainStayDuration : ℝ := 10

/-- The probability that John arrives while the train is at the station -/
noncomputable def probabilityJohnMeetsTrain : ℝ := 1 / 24

theorem john_meets_train_probability :
  probabilityJohnMeetsTrain = 
    (trainArrivalWindow * trainStayDuration) / (2 * johnArrivalWindow * trainArrivalWindow) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_john_meets_train_probability_l420_42048


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_graph_transformation_l420_42089

-- Define f as an arbitrary real-valued function
variable (f : ℝ → ℝ)

-- Define g as the transformation of f
def g (f : ℝ → ℝ) (x : ℝ) : ℝ := f (6 - x)

-- Theorem statement
theorem graph_transformation (f : ℝ → ℝ) (x : ℝ) : 
  g f x = f (6 - x) := by
  -- Unfold the definition of g
  unfold g
  -- The equality follows directly from the definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_graph_transformation_l420_42089


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jason_bartender_years_l420_42032

/-- Calculates the number of years Jason worked as a bartender given his total work experience and managerial experience. -/
theorem jason_bartender_years 
  (total_months : ℕ) 
  (manager_years : ℕ) 
  (manager_months : ℕ) 
  (h1 : total_months = 150) 
  (h2 : manager_years = 3) 
  (h3 : manager_months = 6) : 
  (total_months - (manager_years * 12 + manager_months)) / 12 = 9 := by
  sorry

#check jason_bartender_years

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jason_bartender_years_l420_42032


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_and_inequality_theorem_l420_42025

/-- Definition of sequence a_m -/
def a : ℕ → ℚ
  | 0 => 3/2  -- Add this case to handle Nat.zero
  | 1 => 3/2
  | n + 1 => 3 * a n - 1

/-- Definition of sequence b_m -/
def b (m : ℕ) : ℚ := a m - 1/2

/-- The inequality condition -/
def inequality_holds (m : ℝ) : Prop :=
  ∀ n : ℕ, n ≥ 1 → (b n + 1) / (b (n + 1) - 1) ≤ m

theorem sequence_and_inequality_theorem :
  (∀ n : ℕ, n ≥ 1 → b (n + 1) = 3 * b n) ∧
  (∀ m : ℝ, inequality_holds m ↔ m ≥ 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_and_inequality_theorem_l420_42025


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_independence_day_1776_l420_42059

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
deriving Repr, BEq

/-- Calculates the number of leap years between two years -/
def leapYearsBetween (start year : Nat) : Nat :=
  let years := year - start
  let leapYears := years / 4 - years / 100 + years / 400
  leapYears

/-- Calculates the next day of the week -/
def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

/-- Calculates the day of the week given a start day and number of days passed -/
def dayAfterDays (start : DayOfWeek) (days : Nat) : DayOfWeek :=
  match days with
  | 0 => start
  | n + 1 => dayAfterDays (nextDay start) n

/-- Calculates the previous day of the week -/
def prevDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Saturday
  | DayOfWeek.Monday => DayOfWeek.Sunday
  | DayOfWeek.Tuesday => DayOfWeek.Monday
  | DayOfWeek.Wednesday => DayOfWeek.Tuesday
  | DayOfWeek.Thursday => DayOfWeek.Wednesday
  | DayOfWeek.Friday => DayOfWeek.Thursday
  | DayOfWeek.Saturday => DayOfWeek.Friday

theorem independence_day_1776 :
  dayAfterDays DayOfWeek.Wednesday (250 * 365 + leapYearsBetween 1776 2026) = DayOfWeek.Saturday →
  DayOfWeek.Wednesday = prevDay (prevDay (prevDay DayOfWeek.Saturday)) :=
by
  intro h
  sorry

#eval dayAfterDays DayOfWeek.Wednesday (250 * 365 + leapYearsBetween 1776 2026)
#eval prevDay (prevDay (prevDay DayOfWeek.Saturday))

end NUMINAMATH_CALUDE_ERRORFEEDBACK_independence_day_1776_l420_42059


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_divides_g_product_l420_42049

def g : ℕ → ℕ
  | 0 => 0  -- Added case for 0
  | 1 => 0
  | 2 => 1
  | (n + 2) => g n + g (n + 1) + 1

theorem prime_divides_g_product (n : ℕ) (h : n.Prime) (h2 : n > 5) :
  n ∣ g n * (g n + 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_divides_g_product_l420_42049


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_l420_42051

/-- The circle equation x² + y² = x - y -/
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = x - y

/-- The line equation 3x + 4y + 1 = 0 -/
def line_eq (x y : ℝ) : Prop := 3*x + 4*y + 1 = 0

/-- The theorem stating that the chord length is 7/5 -/
theorem chord_length : 
  ∃ (a b c d : ℝ), 
    circle_eq a b ∧ circle_eq c d ∧ 
    line_eq a b ∧ line_eq c d ∧
    (a - c)^2 + (b - d)^2 = (7/5)^2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_l420_42051


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_selections_for_uniform_98x98_l420_42087

/-- Represents a chessboard with its size and current coloring -/
structure Chessboard where
  size : Nat
  coloring : Fin size → Fin size → Bool

/-- Represents a rectangle selection on the chessboard -/
structure Rectangle where
  top_left : Nat × Nat
  bottom_right : Nat × Nat

/-- Function to apply a rectangle selection to the chessboard -/
def applyRectangle (board : Chessboard) (rect : Rectangle) : Chessboard :=
  sorry

/-- Predicts whether a sequence of rectangle selections will make the board uniform -/
def makesUniform (board : Chessboard) (selections : List Rectangle) : Bool :=
  sorry

/-- Theorem stating the minimum number of selections needed to make a 98x98 chessboard uniform -/
theorem min_selections_for_uniform_98x98 :
  ∀ (initial_board : Chessboard),
    initial_board.size = 98 →
    (∀ (i j : Fin initial_board.size), initial_board.coloring i j = ((i.val + j.val) % 2 == 0)) →
    (∃ (selections : List Rectangle),
      makesUniform initial_board selections ∧
      selections.length = 98 ∧
      (∀ (other_selections : List Rectangle),
        makesUniform initial_board other_selections →
        other_selections.length ≥ 98)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_selections_for_uniform_98x98_l420_42087


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_popcorn_weighted_average_percentage_l420_42055

/-- Represents a bag of popcorn kernels -/
structure PopcornBag where
  totalKernels : ℕ
  poppedKernels : ℕ

/-- Calculates the weighted average percentage of popped kernels -/
noncomputable def weightedAveragePercentage (bags : List PopcornBag) : ℝ :=
  let totalWeightedValue := (bags.map (fun bag => (bag.poppedKernels : ℝ) / (bag.totalKernels : ℝ) * (bag.totalKernels : ℝ))).sum
  let totalKernels := (bags.map (fun bag => bag.totalKernels)).sum
  totalWeightedValue / (totalKernels : ℝ) * 100

theorem popcorn_weighted_average_percentage :
  let bags : List PopcornBag := [
    { totalKernels := 75, poppedKernels := 60 },
    { totalKernels := 50, poppedKernels := 42 },
    { totalKernels := 100, poppedKernels := 25 },
    { totalKernels := 120, poppedKernels := 77 },
    { totalKernels := 150, poppedKernels := 106 }
  ]
  abs (weightedAveragePercentage bags - 60.61) < 0.01 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_popcorn_weighted_average_percentage_l420_42055


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_circles_distance_l420_42058

/-- Given two circles S₁ and S₂ with radii R and r respectively, where R/r = k and k > 1,
    and the circles are tangent at point B, if a line from point A on S₁ is tangent to S₂ at point C,
    and the chord cut off by S₂ on line AB has length b, then AC = b√(k² ± k). -/
theorem tangent_circles_distance (k : ℝ) (b : ℝ) (h₁ : k > 1) (h₂ : b > 0) :
  ∃ (R r : ℝ) (S₁ S₂ : Set (ℝ × ℝ)) (A B C : ℝ × ℝ),
    let AC := Real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2)
    R / r = k ∧
    S₁ = {p : ℝ × ℝ | (p.1 - R)^2 + p.2^2 = R^2} ∧
    S₂ = {p : ℝ × ℝ | p.1^2 + p.2^2 = r^2} ∧
    B ∈ S₁ ∧ B ∈ S₂ ∧
    A ∈ S₁ ∧
    C ∈ S₂ ∧
    (∀ p ∈ S₂, (C.1 - A.1) * (p.1 - C.1) + (C.2 - A.2) * (p.2 - C.2) = 0) ∧
    (∃ (X : ℝ × ℝ), X ∈ S₂ ∧ X ≠ B ∧ Real.sqrt ((A.1 - X.1)^2 + (A.2 - X.2)^2) = b) →
    AC = b * Real.sqrt (k^2 + k) ∨ AC = b * Real.sqrt (k^2 - k) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_circles_distance_l420_42058


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_lower_bound_l420_42014

noncomputable def F (x : ℝ) := Real.log x + x^2 + x

theorem sum_lower_bound {x₁ x₂ : ℝ} (hx₁ : x₁ > 0) (hx₂ : x₂ > 0) 
  (h : F x₁ + F x₂ + x₁ * x₂ = 0) : 
  x₁ + x₂ ≥ (Real.sqrt 5 - 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_lower_bound_l420_42014


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_yellow_marbles_l420_42094

theorem smallest_yellow_marbles (n : ℕ) (h1 : n % 10 = 0) 
  (h2 : n / 2 + n / 5 + 7 ≤ n) : 
  ∃ k : ℕ, k ≥ 2 ∧ n - (n / 2 + n / 5 + 7) = k ∧ 
  ∀ m : ℕ, m < k → ¬∃ t : ℕ, t % 10 = 0 ∧ 
  t / 2 + t / 5 + 7 + m = t := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_yellow_marbles_l420_42094


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_hands_120_degrees_l420_42091

theorem clock_hands_120_degrees : ∃ (a b : ℕ), 
  a + b = 9 ∧ 
  a * b = 14 ∧ 
  (a ≤ 11 ∧ b ≤ 59) ∧
  (a * 30 + b * 6 - (a * 30 + b / 5)) % 360 = 120 ∧
  (a * 30 + b * 6 - b * 6) % 360 = 120 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_hands_120_degrees_l420_42091


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_m_for_identity_transformation_l420_42092

/-- Angle of line l₁ with positive x-axis -/
noncomputable def α : ℝ := Real.pi / 50

/-- Angle of line l₂ with positive x-axis -/
noncomputable def β : ℝ := Real.pi / 75

/-- Angle of line l with positive x-axis -/
noncomputable def θ : ℝ := Real.arctan (7 / 25)

/-- Transformation R that reflects a line in l₁, then in l₂ -/
noncomputable def R (angle : ℝ) : ℝ := 2 * β - 2 * α + angle

/-- n-fold application of R -/
noncomputable def R_n (n : ℕ) (angle : ℝ) : ℝ := angle + n * (2 * β - 2 * α)

/-- Statement of the theorem -/
theorem smallest_m_for_identity_transformation :
  ∃ m : ℕ, m > 0 ∧ R_n m θ = θ + 2 * Real.pi ∧ ∀ k : ℕ, 0 < k → k < m → R_n k θ ≠ θ + 2 * Real.pi :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_m_for_identity_transformation_l420_42092


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_scientific_notation_l420_42023

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ

/-- The value of a number in scientific notation -/
noncomputable def ScientificNotation.value (sn : ScientificNotation) : ℝ :=
  sn.coefficient * (10 : ℝ) ^ sn.exponent

theorem square_scientific_notation
  (a : ℝ)
  (sn_a : ScientificNotation)
  (h1 : a = 25000)
  (h2 : sn_a = ⟨2.5, 4⟩)
  (h3 : a = ScientificNotation.value sn_a) :
  ∃ (sn_a_squared : ScientificNotation),
    sn_a_squared = ⟨6.25, 8⟩ ∧
    a^2 = ScientificNotation.value sn_a_squared :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_scientific_notation_l420_42023


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_phi_l420_42099

open Real

-- Define the set of possible φ values
noncomputable def phi_values : Set ℝ := {π/6, π/4, π/3, π/2}

-- Define the monotonicity condition
def same_monotonicity (φ : ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, 0 ≤ x₁ ∧ x₁ < x₂ ∧ x₂ ≤ π/2 →
    (cos (2*x₁) ≤ cos (2*x₂) ↔ sin (x₁ + φ) ≤ sin (x₂ + φ))

-- Theorem statement
theorem unique_phi : 
  ∃! φ, φ ∈ phi_values ∧ same_monotonicity φ ∧ φ = π/2 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_phi_l420_42099


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_domain_largest_l420_42086

-- Define the sequence of functions
noncomputable def g : ℕ → (ℝ → ℝ)
  | 0 => fun _ => 0  -- Add a case for 0
  | 1 => fun x => Real.sqrt (2 - x)
  | (m + 1) => fun x => g m (Real.sqrt ((m + 1)^2 + 1 - x))

-- Define the domain of a function
def hasDomain (f : ℝ → ℝ) : Set ℝ :=
  {x | ∃ y, f x = y}

-- State the theorem
theorem g_domain_largest :
  (∃ M : ℕ, M > 0 ∧ 
    (∀ m : ℕ, m > M → hasDomain (g m) = ∅) ∧
    (hasDomain (g M) = {-261}) ∧
    (∀ m : ℕ, m > 0 → m < M → (hasDomain (g m)).Nonempty)) ∧
  (∀ M : ℕ, M > 0 →
    ((∀ m : ℕ, m > M → hasDomain (g m) = ∅) ∧
    (hasDomain (g M) = {-261}) ∧
    (∀ m : ℕ, m > 0 → m < M → (hasDomain (g m)).Nonempty)) → M = 5) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_domain_largest_l420_42086


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diamond_heart_calculation_l420_42065

-- Define the diamond operation
def diamond (x y : ℕ) : ℕ := 3 * x + 5 * y

-- Define the heart operation
def heart (z x : ℕ) : ℕ := 4 * z + 2 * x

-- Theorem statement
theorem diamond_heart_calculation : heart (diamond 4 3) 8 = 124 := by
  -- Expand the definition of diamond
  have h1 : diamond 4 3 = 27 := by
    rfl
  
  -- Expand the definition of heart
  have h2 : heart 27 8 = 124 := by
    rfl
  
  -- Combine the steps
  rw [h1, h2]


end NUMINAMATH_CALUDE_ERRORFEEDBACK_diamond_heart_calculation_l420_42065


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monochromatic_subgraph_existence_l420_42071

/-- A simple graph with 10 vertices and n edges that is 2-colored -/
structure ColoredGraph (n : ℕ) where
  vertices : Finset (Fin 10)
  edges : Finset (Fin 10 × Fin 10)
  edge_count : edges.card = n
  simple : ∀ e ∈ edges, e.1 ≠ e.2
  coloring : { e // e ∈ edges } → Fin 2

/-- A monochromatic triangle in a colored graph -/
def has_monochromatic_triangle (G : ColoredGraph n) : Prop :=
  ∃ (a b c : Fin 10), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    (a, b) ∈ G.edges ∧ (b, c) ∈ G.edges ∧ (a, c) ∈ G.edges ∧
    G.coloring ⟨(a, b), by sorry⟩ = G.coloring ⟨(b, c), by sorry⟩ ∧ 
    G.coloring ⟨(b, c), by sorry⟩ = G.coloring ⟨(a, c), by sorry⟩

/-- A monochromatic quadrilateral in a colored graph -/
def has_monochromatic_quadrilateral (G : ColoredGraph n) : Prop :=
  ∃ (a b c d : Fin 10), a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ a ∧ a ≠ c ∧ b ≠ d ∧
    (a, b) ∈ G.edges ∧ (b, c) ∈ G.edges ∧ (c, d) ∈ G.edges ∧ (d, a) ∈ G.edges ∧
    G.coloring ⟨(a, b), by sorry⟩ = G.coloring ⟨(b, c), by sorry⟩ ∧ 
    G.coloring ⟨(b, c), by sorry⟩ = G.coloring ⟨(c, d), by sorry⟩ ∧
    G.coloring ⟨(c, d), by sorry⟩ = G.coloring ⟨(d, a), by sorry⟩

/-- The main theorem -/
theorem monochromatic_subgraph_existence :
  (∀ G : ColoredGraph 31, has_monochromatic_triangle G ∨ has_monochromatic_quadrilateral G) ∧
  (∀ m < 31, ∃ G : ColoredGraph m, ¬(has_monochromatic_triangle G ∨ has_monochromatic_quadrilateral G)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monochromatic_subgraph_existence_l420_42071


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rowing_time_ratio_l420_42017

/-- Proves that the ratio of time taken to row upstream to downstream is 2:1 
    given the man's rowing speed in still water and the current speed. -/
theorem rowing_time_ratio 
  (man_speed : ℝ) 
  (current_speed : ℝ) 
  (h1 : man_speed = 3.6) 
  (h2 : current_speed = 1.2) : 
  (man_speed - current_speed) / (man_speed + current_speed) = 1 / 2 := by
  -- Substitute the given values
  rw [h1, h2]
  -- Simplify the expression
  norm_num
  -- The proof is complete
  done

#check rowing_time_ratio

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rowing_time_ratio_l420_42017


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bicycles_bought_l420_42056

theorem bicycles_bought (n a b : ℝ) (h1 : n > 0) (h2 : a > 0) (h3 : b > 0) :
  ∃ x : ℝ, x > 0 ∧ 
    x = (a * b + Real.sqrt (a * b * (a * b + 4 * n))) / (2 * a) ∧
    (n / (x - b) - a) * x = n :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bicycles_bought_l420_42056


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_perpendicular_and_inequality_l420_42088

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 - a * Real.log x

theorem tangent_line_perpendicular_and_inequality (a : ℝ) :
  (∀ x > 0, HasDerivAt (f a) (2 * x - a / x) x) →
  (HasDerivAt (f a) (2 - a) 1 ∧ (2 - a) * (-1) = -1) →
  (∃ x₀ ∈ Set.Icc 1 (Real.exp 1), (f a x₀ + 1 + a) / x₀ ≤ 0) →
  a = 1 ∧ a ≤ -2 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_perpendicular_and_inequality_l420_42088


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_k_range_l420_42019

-- Define the line and circle
def line (k : ℝ) (x y : ℝ) : Prop := x - y - k = 0
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define the origin
def origin : ℝ × ℝ := (0, 0)

-- Define the intersection points
def intersection_points (k : ℝ) : Prop :=
  ∃ (A B : ℝ × ℝ), A ≠ B ∧ 
    line k A.1 A.2 ∧ line k B.1 B.2 ∧
    circle_eq A.1 A.2 ∧ circle_eq B.1 B.2

-- Define the vector inequality
noncomputable def vector_inequality (A B : ℝ × ℝ) : Prop :=
  let O := origin
  ‖(A.1 - O.1, A.2 - O.2) + (B.1 - O.1, B.2 - O.2)‖ ≥ (Real.sqrt 3 / 3) * ‖(B.1 - A.1, B.2 - A.2)‖

-- State the theorem
theorem intersection_k_range (k : ℝ) :
  k > 0 →
  intersection_points k →
  (∀ (A B : ℝ × ℝ), A ≠ B → line k A.1 A.2 → line k B.1 B.2 → 
    circle_eq A.1 A.2 → circle_eq B.1 B.2 → vector_inequality A B) →
  Real.sqrt 2 ≤ k ∧ k < 2 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_k_range_l420_42019


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_properties_l420_42077

-- Define the parabola
noncomputable def parabola (a c x : ℝ) : ℝ := -x^2 + 2*a*x + c

-- Define the points on the parabola
noncomputable def point_A (a c : ℝ) : ℝ × ℝ := (-3, parabola a c (-3))
noncomputable def point_B (a c : ℝ) : ℝ × ℝ := (a/2, parabola a c (a/2))
noncomputable def point_C (a c m : ℝ) : ℝ × ℝ := (m, parabola a c m)

theorem parabola_properties (a c : ℝ) (h_a : a > 0) :
  -- 1. Axis of symmetry and relation between y₁ and y₂
  (∀ x : ℝ, parabola a c (a - x) = parabola a c (a + x)) ∧
  ((point_A a c).snd < (point_B a c).snd) ∧
  -- 2. Value of a when m = 4 and y₁ = y₃
  (∀ m : ℝ, m = 4 → (point_A a c).snd = (point_C a c m).snd → a = 1/2) ∧
  -- 3. Range of a when 1 ≤ m ≤ 4 and y₁ < y₃ < y₂
  (∀ m : ℝ, 1 ≤ m → m ≤ 4 →
    (point_A a c).snd < (point_C a c m).snd →
    (point_C a c m).snd < (point_B a c).snd →
    (1/2 < a ∧ a < 2/3) ∨ a > 8) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_properties_l420_42077
