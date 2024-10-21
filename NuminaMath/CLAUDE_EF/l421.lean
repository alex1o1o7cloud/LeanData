import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l421_42191

noncomputable section

open Real

theorem triangle_properties (A B C : ℝ) (a b c : ℝ) :
  -- Triangle ABC with sides a, b, c opposite to angles A, B, C
  (0 < A) ∧ (A < π) ∧ (0 < B) ∧ (B < π) ∧ (0 < C) ∧ (C < π) ∧
  (A + B + C = π) ∧
  -- Sine law
  (a / sin A = b / sin B) ∧ (b / sin B = c / sin C) ∧
  -- Given condition
  (cos A / (1 + sin A) = sin (2 * B) / (1 + cos (2 * B))) ∧
  (C = 2 * π / 3) →
  -- Conclusions
  (B = π / 6) ∧
  (∀ (a' b' c' : ℝ),
    (a' / sin A = b' / sin B) ∧ (b' / sin B = c' / sin C) →
    ((a' ^ 2 + b' ^ 2) / c' ^ 2 ≥ 4 * sqrt 2 - 5)) ∧
  (∃ (a' b' c' : ℝ),
    (a' / sin A = b' / sin B) ∧ (b' / sin B = c' / sin C) ∧
    ((a' ^ 2 + b' ^ 2) / c' ^ 2 = 4 * sqrt 2 - 5)) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l421_42191


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_neg_two_range_l421_42138

-- Define the function f
def f (a b x : ℝ) : ℝ := a * x^2 + b * x

-- State the theorem
theorem f_neg_two_range (a b : ℝ) :
  (1 ≤ f a b (-1) ∧ f a b (-1) ≤ 2) →
  (2 ≤ f a b 1 ∧ f a b 1 ≤ 4) →
  ∃ y, f a b (-2) = y ∧ 5 ≤ y ∧ y ≤ 10 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_neg_two_range_l421_42138


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_period_and_omega_relation_monotonic_increasing_omega_range_one_zero_omega_range_l421_42134

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.cos (ω * x + Real.pi / 3)

theorem period_and_omega_relation (ω : ℝ) (h : ω > 0) :
  (∃ T : ℝ, T > 0 ∧ T = 2 ∧ ∀ x : ℝ, f ω x = f ω (x + T)) → ω = Real.pi :=
sorry

theorem monotonic_increasing_omega_range (ω : ℝ) (h : ω > 0) :
  (∀ x y : ℝ, 2*Real.pi/3 < x ∧ x < y ∧ y < Real.pi → f ω x < f ω y) → 1 ≤ ω ∧ ω ≤ 5/3 :=
sorry

theorem one_zero_omega_range (ω : ℝ) (h : ω > 0) :
  (∃! x : ℝ, 0 < x ∧ x < Real.pi ∧ f ω x = 0) → 1/6 < ω ∧ ω ≤ 7/6 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_period_and_omega_relation_monotonic_increasing_omega_range_one_zero_omega_range_l421_42134


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_common_number_in_overlapping_sets_l421_42136

theorem common_number_in_overlapping_sets (numbers : List ℚ) 
    (h_length : numbers.length = 7)
    (h_first_avg : (numbers.take 4).sum / 4 = 5)
    (h_last_avg : (numbers.drop 3).sum / 4 = 8)
    (h_total_avg : numbers.sum / 7 = 46 / 7) : 
  numbers[3]? = some 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_common_number_in_overlapping_sets_l421_42136


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_number_proof_l421_42145

theorem smallest_number_proof (x : ℕ) : 
  (∀ d : ℕ, d ∈ ({35, 25, 21} : Set ℕ) → (x + 3) % d = 0) ∧
  ((x + 3) / 35 = 4728) ∧
  ((x + 3) / 25 = 4728) ∧
  ((x + 3) / 21 = 4728) ∧
  (∀ y < x, ¬(∀ d : ℕ, d ∈ ({35, 25, 21} : Set ℕ) → (y + 3) % d = 0) ∨
            ¬((y + 3) / 35 = 4728) ∨
            ¬((y + 3) / 25 = 4728) ∨
            ¬((y + 3) / 21 = 4728)) →
  x = 2482197 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_number_proof_l421_42145


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_tangents_intersection_l421_42110

-- Define the function f(x) = xe^x - 1
noncomputable def f (x : ℝ) : ℝ := x * Real.exp x - 1

-- State the theorem
theorem parallel_tangents_intersection (x₀ : ℝ) :
  (Real.exp x₀ = 1 / x₀) → 0 < x₀ ∧ x₀ < 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_tangents_intersection_l421_42110


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_12_5_l421_42114

-- Define the vertices of the triangle
def A : ℚ × ℚ := (2, 3)
def B : ℚ × ℚ := (7, 3)
def C : ℚ × ℚ := (4, 8)

-- Define the function to calculate the area of a triangle given its vertices
def triangleArea (p1 p2 p3 : ℚ × ℚ) : ℚ :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let (x3, y3) := p3
  (1/2) * abs ((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))

-- Theorem statement
theorem triangle_area_is_12_5 :
  triangleArea A B C = 25/2 := by
  -- Unfold the definitions and simplify
  unfold triangleArea A B C
  -- Perform the calculation
  simp [abs_of_nonneg]
  -- The proof is complete
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_12_5_l421_42114


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_ratio_in_class_l421_42159

/-- Represents the number of students in a class -/
structure ClassComposition where
  males : ℕ
  females : ℕ

/-- Represents the circle graph used for career preferences -/
def CircleGraph := ClassComposition → ℝ

/-- The condition that 72 degrees represents one male and one female -/
def representsTwoStudents (graph : CircleGraph) (c : ClassComposition) : Prop :=
  graph c = 72

/-- The condition that the graph is proportional to the number of students -/
def isProportional (graph : CircleGraph) : Prop :=
  ∀ (c1 c2 : ClassComposition),
    (c1.males + c1.females : ℝ) / (c2.males + c2.females) = graph c1 / graph c2

/-- The theorem to be proved -/
theorem equal_ratio_in_class (graph : CircleGraph) (c : ClassComposition) :
    representsTwoStudents graph c →
    isProportional graph →
    c.males = c.females := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_ratio_in_class_l421_42159


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polygon_sides_count_l421_42151

/-- The number of sides of a convex polygon with interior angles in arithmetic progression -/
def num_sides_polygon (common_diff : ℚ) (largest_angle : ℚ) : ℕ :=
  sorry

/-- The sum of interior angles of a polygon with n sides -/
def sum_interior_angles (n : ℕ) : ℚ :=
  180 * (n - 2)

/-- The largest angle in the arithmetic progression of interior angles -/
def largest_angle (first_angle : ℚ) (common_diff : ℚ) (n : ℕ) : ℚ :=
  first_angle + (n - 1) * common_diff

/-- The sum of interior angles using the arithmetic sequence formula -/
def sum_interior_angles_seq (first_angle : ℚ) (largest_angle : ℚ) (n : ℕ) : ℚ :=
  n * (first_angle + largest_angle) / 2

theorem polygon_sides_count :
  num_sides_polygon 3 150 = 25 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polygon_sides_count_l421_42151


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_max_at_zero_l421_42157

noncomputable def f (b c x : ℝ) : ℝ := 3 * Real.sin (b * x + c)

theorem sin_max_at_zero (b : ℝ) (h : b > 0) :
  (∀ x : ℝ, f b (Real.pi / 2) 0 ≥ f b (Real.pi / 2) x) ↔
  (∀ c : ℝ, (∀ x : ℝ, f b c 0 ≥ f b c x) → c = Real.pi / 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_max_at_zero_l421_42157


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gordon_book_spending_l421_42147

noncomputable def apply_discount (price : ℝ) : ℝ :=
  if price > 22 then price * 0.7
  else if price < 20 then price * 0.8
  else price

def book_prices : List ℝ := [25, 18, 21, 35, 12, 10]

theorem gordon_book_spending :
  (book_prices.map apply_discount).sum = 95 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gordon_book_spending_l421_42147


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decimal_representation_5_13_digit_80_of_5_13_l421_42135

theorem decimal_representation_5_13 : 
  ∃ (seq : ℕ → ℕ), 
    (∀ n, seq n < 10) ∧ 
    (∀ n, seq (n + 6) = seq n) ∧
    (5 : ℚ) / 13 = (0 : ℚ) + ∑' n, (seq n : ℚ) / (10 : ℚ) ^ (n + 1) ∧
    seq 1 = 8 := by
  sorry

theorem digit_80_of_5_13 : 
  ∃ (seq : ℕ → ℕ), 
    (∀ n, seq n < 10) ∧ 
    (∀ n, seq (n + 6) = seq n) ∧
    (5 : ℚ) / 13 = (0 : ℚ) + ∑' n, (seq n : ℚ) / (10 : ℚ) ^ (n + 1) ∧
    seq 79 = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_decimal_representation_5_13_digit_80_of_5_13_l421_42135


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_john_meets_train_l421_42117

open MeasureTheory

/-- Train arrival time in minutes after 3:00 PM -/
def train_arrival : Set ℝ := Set.Icc 0 60

/-- Train departure time in minutes after 3:00 PM -/
def train_departure (y : ℝ) : ℝ := y + 15

/-- John's arrival time in minutes after 3:00 PM -/
def john_arrival : Set ℝ := Set.Icc 15 75

/-- The event that John arrives while the train is still at the station -/
def john_meets_train : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 ∈ john_arrival ∧ p.2 ∈ train_arrival ∧ p.1 ≤ train_departure p.2}

/-- The probability of John meeting the train -/
theorem probability_john_meets_train :
  (volume john_meets_train) / (volume (Set.prod john_arrival train_arrival)) = 3 / 16 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_john_meets_train_l421_42117


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2x_value_l421_42132

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin x * Real.cos x - 2 * (Real.cos (x + Real.pi / 4))^2

theorem cos_2x_value (x : ℝ) (h1 : 0 < x) (h2 : x < Real.pi / 2) 
  (h3 : f (x + Real.pi / 6) = 3 / 5) : 
  Real.cos (2 * x) = (4 * Real.sqrt 3 - 3) / 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2x_value_l421_42132


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_angles_l421_42141

theorem parallelogram_angles (a b α : Real) (h1 : a > b) (h2 : 0 < α) (h3 : α < π / 2) :
  let θ := Real.arcsin ((a^2 - b^2) / (2 * a * b) * Real.tan α)
  ∃ (θ1 θ2 : Real), θ1 = θ ∧ θ2 = π - θ ∧ 
    θ1 + θ2 = π ∧ 0 < θ1 ∧ θ1 < π ∧ 0 < θ2 ∧ θ2 < π :=
by
  sorry

#check parallelogram_angles

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_angles_l421_42141


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_central_angle_of_sector_l421_42192

/-- The radius of the circular sector in centimeters -/
def radius : ℝ := 10

/-- The area of the circular sector in square centimeters -/
def area : ℝ := 100

/-- The formula for the area of a circular sector -/
noncomputable def sector_area (r θ : ℝ) : ℝ := Real.pi * r^2 * (θ / (2 * Real.pi))

/-- Theorem: The central angle of a circular sector with radius 10 cm and area 100 cm² is 2 radians -/
theorem central_angle_of_sector :
  ∃ θ : ℝ, sector_area radius θ = area ∧ θ = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_central_angle_of_sector_l421_42192


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_pasture_width_optimal_pasture_length_l421_42168

/-- Represents a rectangular cow pasture --/
structure Pasture where
  width : ℝ  -- Width of the pasture (perpendicular to barn)
  length : ℝ  -- Length of the pasture (parallel to barn)

/-- Calculates the area of the pasture --/
noncomputable def Pasture.area (p : Pasture) : ℝ := p.width * p.length

/-- Represents the constraints of the fencing problem --/
structure FencingProblem where
  barnLength : ℝ
  fenceCost : ℝ
  totalFenceCost : ℝ

/-- Calculates the total available fencing --/
noncomputable def FencingProblem.availableFencing (fp : FencingProblem) : ℝ :=
  fp.totalFenceCost / fp.fenceCost

/-- Calculates the length of the pasture given its width --/
noncomputable def pastureLength (fp : FencingProblem) (width : ℝ) : ℝ :=
  fp.availableFencing - 2 * width

/-- Theorem: The optimal width that maximizes the pasture area is 37.5 feet --/
theorem optimal_pasture_width (fp : FencingProblem)
    (h1 : fp.barnLength = 300)
    (h2 : fp.fenceCost = 6)
    (h3 : fp.totalFenceCost = 900) :
    ∃ (w : ℝ), w = 37.5 ∧
    ∀ (x : ℝ), Pasture.area { width := x, length := pastureLength fp x } ≤
                Pasture.area { width := w, length := pastureLength fp w } := by
  sorry

/-- Corollary: The optimal length of the side parallel to the barn is 75 feet --/
theorem optimal_pasture_length (fp : FencingProblem)
    (h1 : fp.barnLength = 300)
    (h2 : fp.fenceCost = 6)
    (h3 : fp.totalFenceCost = 900) :
    pastureLength fp 37.5 = 75 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_pasture_width_optimal_pasture_length_l421_42168


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l421_42160

/-- Calculates the length of a train given the speeds of a jogger and train, initial distance, and passing time. -/
theorem train_length_calculation 
  (jogger_speed : ℝ) 
  (train_speed : ℝ) 
  (initial_distance : ℝ) 
  (passing_time : ℝ) 
  (h1 : jogger_speed = 9 * (5/18))  -- Convert 9 kmph to m/s
  (h2 : train_speed = 45 * (5/18))  -- Convert 45 kmph to m/s
  (h3 : initial_distance = 250)
  (h4 : passing_time = 37) :
  train_speed * passing_time - jogger_speed * passing_time - initial_distance = 120 := by
  -- Proof steps would go here
  sorry

#check train_length_calculation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l421_42160


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_B_subset_A_iff_a_in_range_l421_42184

/-- The set A defined as the half-open interval [-2, 4) -/
def A : Set ℝ := Set.Ico (-2) 4

/-- The set B defined by the quadratic inequality x^2 - ax - 4 ≤ 0 -/
def B (a : ℝ) : Set ℝ := {x : ℝ | x^2 - a*x - 4 ≤ 0}

/-- Theorem stating that B is a subset of A if and only if a is in [0, 3) -/
theorem B_subset_A_iff_a_in_range :
  ∀ a : ℝ, B a ⊆ A ↔ a ∈ Set.Ico 0 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_B_subset_A_iff_a_in_range_l421_42184


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l421_42100

/-- The speed of a train given its length and time to cross a point -/
noncomputable def train_speed (length : ℝ) (time : ℝ) : ℝ := length / time

/-- Theorem: A train 700 meters long that takes 20 seconds to cross an electric pole has a speed of 35 meters per second -/
theorem train_speed_calculation :
  train_speed 700 20 = 35 := by
  -- Unfold the definition of train_speed
  unfold train_speed
  -- Perform the division
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l421_42100


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_savings_l421_42156

theorem total_savings (josiah_daily : ℚ) (josiah_days : ℕ) 
                      (leah_daily : ℚ) (leah_days : ℕ) 
                      (megan_daily : ℚ) (megan_days : ℕ) : 
  josiah_daily = 0.25 →
  josiah_days = 24 →
  leah_daily = 0.50 →
  leah_days = 20 →
  megan_daily = 2 * leah_daily →
  megan_days = 12 →
  josiah_daily * josiah_days + leah_daily * leah_days + megan_daily * megan_days = 28 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_savings_l421_42156


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_twenty_fifth_digit_is_zero_l421_42113

def decimal_rep_1_8 : ℚ := 1/8
def decimal_rep_1_11 : ℚ := 1/11

def sum_rep : ℚ := decimal_rep_1_8 + decimal_rep_1_11

def digit_at_position (q : ℚ) (n : ℕ) : ℕ :=
  ((q * 10^n).floor % 10).natAbs

theorem twenty_fifth_digit_is_zero :
  digit_at_position sum_rep 25 = 0 := by
  sorry

#eval digit_at_position sum_rep 25

end NUMINAMATH_CALUDE_ERRORFEEDBACK_twenty_fifth_digit_is_zero_l421_42113


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_equation_l421_42161

def M : Matrix (Fin 2) (Fin 2) ℝ := !![3, 4; -2, 0]

theorem matrix_equation : M ^ 2 = 3 • M - 8 • (1 : Matrix (Fin 2) (Fin 2) ℝ) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_equation_l421_42161


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_expression_l421_42154

theorem max_value_of_expression :
  ∃ (a b c d e f : ℕ),
    a ∈ Finset.range 9 ∧
    b ∈ Finset.range 9 ∧
    c ∈ Finset.range 9 ∧
    d ∈ Finset.range 9 ∧
    e ∈ Finset.range 9 ∧
    f ∈ Finset.range 9 ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧
    b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧
    c ≠ d ∧ c ≠ e ∧ c ≠ f ∧
    d ≠ e ∧ d ≠ f ∧
    e ≠ f ∧
    (a + 1) * (b + 1) + (c + 1) * (d + 1) + (e + 1) * (f + 1) = 8569 ∧
    ∀ (x y z w u v : ℕ),
      x ∈ Finset.range 9 →
      y ∈ Finset.range 9 →
      z ∈ Finset.range 9 →
      w ∈ Finset.range 9 →
      u ∈ Finset.range 9 →
      v ∈ Finset.range 9 →
      x ≠ y ∧ x ≠ z ∧ x ≠ w ∧ x ≠ u ∧ x ≠ v ∧
      y ≠ z ∧ y ≠ w ∧ y ≠ u ∧ y ≠ v ∧
      z ≠ w ∧ z ≠ u ∧ z ≠ v ∧
      w ≠ u ∧ w ≠ v ∧
      u ≠ v →
      (x + 1) * (y + 1) + (z + 1) * (w + 1) + (u + 1) * (v + 1) ≤ 8569 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_expression_l421_42154


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_range_l421_42152

noncomputable def f (x : ℝ) : ℝ := Real.log (x^2 + Real.exp (-1)) / Real.log (Real.exp (-1)) - abs (x / Real.exp 1)

theorem f_inequality_range (x : ℝ) : f (x + 1) < f (2 * x - 1) ↔ 0 < x ∧ x < 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_range_l421_42152


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_product_identity_l421_42166

theorem log_product_identity (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (Real.log (x^2) / Real.log (y^4)) * (Real.log y / Real.log (x^3)) * 
  (Real.log (x^4) / Real.log (y^3)) * (Real.log (y^3) / Real.log (x^2)) * 
  (Real.log (x^3) / Real.log y) = Real.log x / Real.log y :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_product_identity_l421_42166


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_properties_l421_42104

/-- A parabola that opens upwards and passes through specific points -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ
  m : ℝ
  h_a_pos : 0 < a
  h_m_range : -2 < m ∧ m < -1
  h_root_1 : a * 1^2 + b * 1 + c = 0
  h_root_m : a * m^2 + b * m + c = 0

/-- The main theorem about properties of the parabola -/
theorem parabola_properties (p : Parabola) :
  (p.a * p.b * p.c < 0) ∧
  (∀ t > 0, let y₁ := p.a * (t-1)^2 + p.b * (t-1) + p.c;
            let y₂ := p.a * (t+1)^2 + p.b * (t+1) + p.c;
            y₁ < y₂) ∧
  (∀ x : ℝ, p.a * (x - p.m) * (x - 1) + 1 ≠ 0 → p.b^2 - 4 * p.a * p.c < 4 * p.a) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_properties_l421_42104


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_min_on_interval_l421_42131

-- Define the function
noncomputable def f (x : ℝ) : ℝ := |x + 5/2|

-- Define the interval
def interval : Set ℝ := {x | -5 ≤ x ∧ x ≤ -2}

-- State the theorem
theorem f_max_min_on_interval :
  (∀ x ∈ interval, f x ≤ 5/2) ∧
  (∃ x ∈ interval, f x = 5/2) ∧
  (∀ x ∈ interval, f x ≥ 0) ∧
  (∃ x ∈ interval, f x = 0) := by
  sorry

#check f_max_min_on_interval

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_min_on_interval_l421_42131


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_sequence_finite_l421_42124

theorem prime_sequence_finite (p : ℕ → ℕ) 
  (h_prime : ∀ n, Nat.Prime (p n))
  (h_largest_divisor : ∀ n, p (n + 2) = (p n + p (n + 1) + 2018).factors.maximum?) :
  Set.Finite {i | ∃ n, p n = i} :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_sequence_finite_l421_42124


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_denis_next_to_anya_and_gena_l421_42150

-- Define the type for students
inductive Student : Type
| Anya : Student
| Borya : Student
| Vera : Student
| Gena : Student
| Denis : Student

-- Define the type for positions in line
inductive Position : Type
| first : Position
| second : Position
| third : Position
| fourth : Position
| fifth : Position

-- Define the line function
def line : Position → Student := sorry

-- Helper function to get the next position
def Position.next : Position → Position
| Position.first => Position.second
| Position.second => Position.third
| Position.third => Position.fourth
| Position.fourth => Position.fifth
| Position.fifth => Position.first

-- Helper function to get the previous position
def Position.prev : Position → Position
| Position.first => Position.fifth
| Position.second => Position.first
| Position.third => Position.second
| Position.fourth => Position.third
| Position.fifth => Position.fourth

-- Conditions
axiom borya_first : line Position.first = Student.Borya
axiom vera_next_to_anya : ∃ p : Position, (line p = Student.Vera ∧ line (Position.next p) = Student.Anya) ∨ (line p = Student.Anya ∧ line (Position.next p) = Student.Vera)
axiom vera_not_next_to_gena : ¬∃ p : Position, (line p = Student.Vera ∧ line (Position.next p) = Student.Gena) ∨ (line p = Student.Gena ∧ line (Position.next p) = Student.Vera)
axiom anya_borya_gena_not_adjacent : ¬∃ p : Position, 
  (line p = Student.Anya ∧ line (Position.next p) = Student.Borya) ∨
  (line p = Student.Borya ∧ line (Position.next p) = Student.Anya) ∨
  (line p = Student.Anya ∧ line (Position.next p) = Student.Gena) ∨
  (line p = Student.Gena ∧ line (Position.next p) = Student.Anya) ∨
  (line p = Student.Borya ∧ line (Position.next p) = Student.Gena) ∨
  (line p = Student.Gena ∧ line (Position.next p) = Student.Borya)

-- Theorem to prove
theorem denis_next_to_anya_and_gena : 
  ∃ p : Position, (line p = Student.Denis ∧ 
    ((line (Position.prev p) = Student.Anya ∧ line (Position.next p) = Student.Gena) ∨
     (line (Position.prev p) = Student.Gena ∧ line (Position.next p) = Student.Anya))) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_denis_next_to_anya_and_gena_l421_42150


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_c_is_15_degrees_l421_42108

/-- In a triangle ABC, given that A - C = 90° and a + c = √2 * b, prove that angle C = 15° --/
theorem triangle_angle_c_is_15_degrees 
  (A B C : ℝ) -- Angles of the triangle
  (a b c : ℝ) -- Sides of the triangle
  (h1 : A - C = π / 2) -- A - C = 90°
  (h2 : a + c = Real.sqrt 2 * b) -- a + c = √2 * b
  (h3 : 0 < A ∧ A < π) -- A is a valid angle
  (h4 : 0 < B ∧ B < π) -- B is a valid angle
  (h5 : 0 < C ∧ C < π) -- C is a valid angle
  (h6 : A + B + C = π) -- Sum of angles in a triangle
  (h7 : Real.sin A / a = Real.sin B / b) -- Law of sines
  (h8 : Real.sin B / b = Real.sin C / c) -- Law of sines
  : C = π / 12 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_c_is_15_degrees_l421_42108


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_terminal_side_l421_42127

noncomputable def second_quadrant (α : Real) : Prop :=
  Real.pi / 2 < α ∧ α < Real.pi

theorem point_on_terminal_side 
  (α : Real) (x : Real) 
  (h1 : second_quadrant α)
  (h2 : Real.cos α = (Real.sqrt 2 / 4) * x)
  (h3 : ∃ (P : Real × Real), P.1 = x ∧ P.2 = Real.sqrt 5) :
  x = -Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_terminal_side_l421_42127


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_common_ratio_l421_42164

/-- A geometric sequence {a_n} -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = q * a n

/-- The common ratio of a geometric sequence -/
noncomputable def common_ratio (a : ℕ → ℝ) : ℝ :=
  a 2 / a 1

/-- A sequence is monotonically decreasing -/
def monotonically_decreasing (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) ≤ a n

theorem geometric_sequence_common_ratio
  (a : ℕ → ℝ)
  (h_geom : geometric_sequence a)
  (h_decr : monotonically_decreasing a)
  (h_prod : a 1 * a 5 = 9)
  (h_sum : a 2 + a 4 = 10) :
  common_ratio a = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_common_ratio_l421_42164


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_determinant_equals_expression_l421_42179

variable (p q r : ℝ)
variable (a b c : ℝ)

-- Define the cubic equation
def cubic_equation (x : ℝ) : ℝ := x^3 + p*x^2 + q*x + r

-- Define the roots of the cubic equation
axiom root_a : cubic_equation p q r a = 0
axiom root_b : cubic_equation p q r b = 0
axiom root_c : cubic_equation p q r c = 0

-- Define the determinant
def determinant (a b c : ℝ) : ℝ := 
  (1 + a^2) * ((1 + b^2) * (1 + c^2) - 1) - 
  (1 * (1 + c^2) - 1) + 
  (1 * 1 - (1 + b^2))

-- Theorem statement
theorem determinant_equals_expression (a b c : ℝ) : 
  determinant a b c = a^2*b^2 + a^2*c^2 + a^2*b^2*c^2 + b^2*c^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_determinant_equals_expression_l421_42179


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_right_triangle_implies_a_zero_l421_42193

-- Define the line
def line (a x y : ℝ) : Prop := a * x + y - 2 = 0

-- Define the circle
def circle_eq (a x y : ℝ) : Prop := (x - 1)^2 + (y - a)^2 = 16/3

-- Define the center of the circle
def circle_center (a : ℝ) : ℝ × ℝ := (1, a)

-- Define the intersection points
def intersection_points (a : ℝ) : Prop :=
  ∃ (A B : ℝ × ℝ), line a A.1 A.2 ∧ circle_eq a A.1 A.2 ∧
                   line a B.1 B.2 ∧ circle_eq a B.1 B.2 ∧
                   A ≠ B

-- Define the right triangle condition
def right_triangle (A B C : ℝ × ℝ) : Prop :=
  (A.1 - C.1) * (B.1 - C.1) + (A.2 - C.2) * (B.2 - C.2) = 0

-- Theorem statement
theorem intersection_right_triangle_implies_a_zero (a : ℝ) :
  intersection_points a →
  (∃ (A B : ℝ × ℝ), right_triangle A B (circle_center a)) →
  a = 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_right_triangle_implies_a_zero_l421_42193


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_max_value_of_f_l421_42163

noncomputable def f (x : ℝ) : ℝ := Real.sin (x / 5) + Real.sin (x / 13)

noncomputable def deg_to_rad (x : ℝ) : ℝ := x * (Real.pi / 180)

theorem smallest_max_value_of_f :
  ∃ (x : ℝ), x > 0 ∧ 
    (∀ (y : ℝ), y > 0 → f (deg_to_rad y) ≤ f (deg_to_rad x)) ∧
    x = 5850 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_max_value_of_f_l421_42163


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_and_chord_theorem_circle_and_chord_theorem_proof_l421_42123

/-- Circle passing through three points -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Line in 2D space -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Point in 2D space -/
def Point := ℝ × ℝ

/-- Check if a point lies on a circle -/
def Circle.contains (c : Circle) (p : Point) : Prop :=
  let (x, y) := p
  let (cx, cy) := c.center
  (x - cx)^2 + (y - cy)^2 = c.radius^2

/-- Check if a point lies on a line -/
def Line.contains (l : Line) (p : Point) : Prop :=
  let (x, y) := p
  l.a * x + l.b * y + l.c = 0

/-- Distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

/-- Theorem statement -/
theorem circle_and_chord_theorem (C : Circle) (P : Point) : Prop :=
  -- Given conditions
  C.contains (0, 0) ∧ C.contains (1, 3) ∧ C.contains (4, 0) ∧ P = (3, 6) →
  -- Conclusions
  (∃ (D E F : ℝ), C.center.1 = -D/2 ∧ C.center.2 = -E/2 ∧ C.radius^2 = D^2/4 + E^2/4 - F ∧
    ∀ (x y : ℝ), C.contains (x, y) ↔ x^2 + y^2 + D*x + E*y + F = 0) ∧
  (∃ (l1 l2 : Line),
    (l1.a = 1 ∧ l1.b = 0 ∧ l1.c = -3) ∧
    (l2.a = 12 ∧ l2.b = -5 ∧ l2.c = -6) ∧
    (∀ (l : Line), l.contains P ∧
      (∃ (p1 p2 : Point), p1 ≠ p2 ∧ C.contains p1 ∧ C.contains p2 ∧
        l.contains p1 ∧ l.contains p2 ∧ distance p1 p2 = 4) →
      l = l1 ∨ l = l2))

-- The proof of the theorem
theorem circle_and_chord_theorem_proof : ∀ (C : Circle) (P : Point), circle_and_chord_theorem C P := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_and_chord_theorem_circle_and_chord_theorem_proof_l421_42123


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_drum_Y_final_capacity_l421_42183

/-- Represents the capacity of a drum -/
def C : ℝ := 1

/-- Represents the amount of oil in drum X -/
noncomputable def oil_X : ℝ := C / 2

/-- Represents the capacity of drum Y -/
noncomputable def capacity_Y : ℝ := 2 * C

/-- Represents the amount of oil initially in drum Y -/
noncomputable def initial_oil_Y : ℝ := capacity_Y / 5

/-- Represents the final amount of oil in drum Y after pouring from X -/
noncomputable def final_oil_Y : ℝ := initial_oil_Y + oil_X

/-- The theorem stating that drum Y will be 9/10 full after pouring -/
theorem drum_Y_final_capacity : final_oil_Y / capacity_Y = 9 / 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_drum_Y_final_capacity_l421_42183


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_above_line_l421_42128

-- Define the power function
noncomputable def power_function (x : ℝ) (α : ℝ) : ℝ := x^α

-- Define the condition for the graph being above y = x
def is_above_line (f : ℝ → ℝ) : Prop :=
  ∀ x, x ∈ (Set.Ioo 0 1) → f x > x

-- State the theorem
theorem power_function_above_line :
  ∀ α : ℝ, (∀ x, x ∈ (Set.Ioo 0 1) → power_function x α > x) ↔ α ∈ Set.Iio 1 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_above_line_l421_42128


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_ellipse_to_line_l421_42111

/-- The maximum distance from a point on the ellipse x²/16 + y²/4 = 1 
    to the line x + 2y - √2 = 0 is √10. -/
theorem max_distance_ellipse_to_line : 
  ∃ (d : ℝ), d = Real.sqrt 10 ∧ 
  (∀ (x y : ℝ), x^2/16 + y^2/4 = 1 → 
    |x + 2*y - Real.sqrt 2| / Real.sqrt 5 ≤ d) ∧
  (∃ (x y : ℝ), x^2/16 + y^2/4 = 1 ∧ 
    |x + 2*y - Real.sqrt 2| / Real.sqrt 5 = d) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_ellipse_to_line_l421_42111


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_difference_in_1_88_l421_42118

/-- The value of a digit in a decimal number based on its place value -/
def digitValue (digit : ℤ) (placeValue : ℚ) : ℚ :=
  (digit : ℚ) * placeValue

/-- The number we're working with -/
def number : ℚ := 1.88

/-- The difference between the values of the digits in the tenth and hundredth places -/
def digitDifference (n : ℚ) : ℚ :=
  digitValue (Int.floor ((n * 10) % 10)) (1/10) - digitValue (Int.floor ((n * 100) % 10)) (1/100)

theorem difference_in_1_88 :
  digitDifference number = 0.72 := by
  sorry

#eval digitDifference number

end NUMINAMATH_CALUDE_ERRORFEEDBACK_difference_in_1_88_l421_42118


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_well_depth_calculation_l421_42162

/-- The depth of the well in feet -/
def well_depth : ℝ := 33.18

/-- The total time from dropping the stone to hearing it hit the bottom, in seconds -/
def total_time : ℝ := 8

/-- The coefficient in the stone's fall distance equation (d = 18t²) -/
def fall_coefficient : ℝ := 18

/-- The velocity of sound in feet per second -/
def sound_velocity : ℝ := 1150

/-- Theorem stating that the calculated well depth is approximately 33.18 feet -/
theorem well_depth_calculation : 
  ∃ (t₁ t₂ : ℝ),
    t₁ + t₂ = total_time ∧
    well_depth = fall_coefficient * t₁^2 ∧
    t₂ = well_depth / sound_velocity ∧
    abs (well_depth - 33.18) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_well_depth_calculation_l421_42162


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mobile_call_charges_l421_42144

-- Define the charging methods
noncomputable def method_A_fee (x : ℝ) : ℝ := if x ≤ 600 then 30 else 0.1 * x - 30
noncomputable def method_B_fee (x : ℝ) : ℝ := if x ≤ 1200 then 50 else 0.1 * x - 70

-- Theorem statement
theorem mobile_call_charges :
  -- Part 1: Verify the charge functions
  (∀ x : ℝ, x ≥ 0 → method_A_fee x = if x ≤ 600 then 30 else 0.1 * x - 30) ∧
  (∀ x : ℝ, x ≥ 0 → method_B_fee x = if x ≤ 1200 then 50 else 0.1 * x - 70) ∧
  -- Part 2: Method A is more cost-effective when 0 ≤ x < 800
  (∀ x : ℝ, 0 ≤ x ∧ x < 800 → method_A_fee x < method_B_fee x) ∧
  -- Part 3: Difference in call time for $60 charge
  (∃ x_A x_B : ℝ, method_A_fee x_A = 60 ∧ method_B_fee x_B = 60 ∧ x_B - x_A = 400) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_mobile_call_charges_l421_42144


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_111_of_33_div_555_l421_42143

def decimal_rep (n d : ℕ) : ℚ := n / d

def repeating_decimal (q : ℚ) (pre rep : List ℕ) : Prop :=
  ∃ (k : ℕ), ∀ (i : ℕ), 
    let digit := ((q * 10^(i+1)).floor % 10).toNat
    if i < pre.length 
    then digit = pre[i]!
    else digit = rep[(i - pre.length) % rep.length]!

def nth_digit_after_decimal (q : ℚ) (n : ℕ) : ℕ :=
  ((q * 10^n).floor % 10).toNat

theorem digit_111_of_33_div_555 : 
  repeating_decimal (decimal_rep 33 555) [0] [5, 9, 4] →
  nth_digit_after_decimal (decimal_rep 33 555) 111 = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_111_of_33_div_555_l421_42143


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_range_l421_42169

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 
  if x < 1 then (2 - a) * x + 1 else a^x

theorem f_increasing_range (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) 
  (h3 : ∀ x y : ℝ, x < y → f a x < f a y) : 
  a ∈ Set.Icc (3/2) 2 ∧ a < 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_range_l421_42169


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_acute_triangle_properties_l421_42137

-- Define an acute triangle ABC
structure AcuteTriangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ
  acute : 0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2
  sides : a > 0 ∧ b > 0 ∧ c > 0
  law_of_sines : a / Real.sin A = b / Real.sin B
  angle_sum : A + B + C = π

-- State the theorem
theorem acute_triangle_properties (t : AcuteTriangle) 
  (h : 2 * t.b * Real.sin t.A - Real.sqrt 3 * t.a = 0) :
  t.B = π/3 ∧ 
  (Real.sqrt 3 + 1) / 2 < Real.cos t.A + Real.cos t.B + Real.cos t.C ∧
  Real.cos t.A + Real.cos t.B + Real.cos t.C ≤ 3/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_acute_triangle_properties_l421_42137


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l421_42120

/-- Definition of the piecewise function f -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x > 0 then Real.exp x - a * x^2
  else -x^2 + (a - 2) * x + 2 * a

/-- The solution set of f(x) ≥ 0 is [-2, +∞) -/
def solution_set (a : ℝ) : Set ℝ :=
  {x : ℝ | f a x ≥ 0}

/-- Main theorem: The range of a for which the solution set is [-2, +∞) is [0, e^2/4] -/
theorem range_of_a :
  {a : ℝ | solution_set a = Set.Ici (-2)} = Set.Icc 0 ((Real.exp 2) / 4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l421_42120


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_cd_length_l421_42148

/-- Represents a trapezoid ABCD -/
structure Trapezoid where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  ad_parallel_bc : (A.2 - D.2) / (A.1 - D.1) = (B.2 - C.2) / (B.1 - C.1)
  bd_length : Real.sqrt ((B.1 - D.1)^2 + (B.2 - D.2)^2) = 2
  angle_dba : Real.arccos ((B.1 - A.1) * (D.1 - B.1) + (B.2 - A.2) * (D.2 - B.2)) / 
              (Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2) * Real.sqrt ((D.1 - B.1)^2 + (D.2 - B.2)^2)) = π / 6
  angle_bdc : Real.arccos ((D.1 - C.1) * (B.1 - D.1) + (D.2 - C.2) * (B.2 - D.2)) / 
              (Real.sqrt ((D.1 - C.1)^2 + (D.2 - C.2)^2) * Real.sqrt ((B.1 - D.1)^2 + (B.2 - D.2)^2)) = π / 3
  ad_length : Real.sqrt ((A.1 - D.1)^2 + (A.2 - D.2)^2) = 10
  bc_length : Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2) = 18

theorem trapezoid_cd_length (t : Trapezoid) : 
  Real.sqrt ((t.C.1 - t.D.1)^2 + (t.C.2 - t.D.2)^2) = 16 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_cd_length_l421_42148


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rotational_phenomena_l421_42116

-- Define the concept of rotation
def is_rotation (movement : String) : Prop :=
  movement.contains ('t') ∧ -- Simplified condition for "turning around a point"
  movement.contains ('c')   -- Simplified condition for "covering a specific angle"

-- Define the given phenomena
def hour_hand : String := "rotation of the hour hand"
def ferris_wheel : String := "rotation of the Ferris wheel"
def groundwater : String := "annual decline of the groundwater level"
def conveyor_belt : String := "robots on the conveyor belt"

-- List of all phenomena
def phenomena : List String := [hour_hand, ferris_wheel, groundwater, conveyor_belt]

-- Theorem stating which phenomena are rotations
theorem rotational_phenomena : 
  {p ∈ phenomena | is_rotation p} = {hour_hand, ferris_wheel} := by
  sorry

#check rotational_phenomena

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rotational_phenomena_l421_42116


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_theorem_l421_42129

def inverse_function_problem (g : ℝ → ℝ) (g_inv : ℝ → ℝ) : Prop :=
  (Function.LeftInverse g_inv g ∧ Function.RightInverse g_inv g) ∧ 
  (g 4 = 7) ∧ 
  (g 6 = 2) ∧ 
  (g 3 = 6) →
  g_inv (g_inv 6 + g_inv 7 - 1) = 3

theorem inverse_function_theorem :
  ∃ (g : ℝ → ℝ) (g_inv : ℝ → ℝ), inverse_function_problem g g_inv := by
  sorry

#check inverse_function_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_theorem_l421_42129


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_below_x_axis_l421_42139

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parallelogram defined by four vertices -/
structure Parallelogram where
  e : Point
  f : Point
  g : Point
  h : Point

/-- Calculates the area of a parallelogram given its vertices -/
noncomputable def parallelogramArea (p : Parallelogram) : ℝ :=
  let a := p.e.x * p.f.y + p.f.x * p.g.y + p.g.x * p.h.y + p.h.x * p.e.y
  let b := p.f.x * p.e.y + p.g.x * p.f.y + p.h.x * p.g.y + p.e.x * p.h.y
  (1/2) * abs (a - b)

/-- Calculates the area of the part of the parallelogram below the x-axis -/
noncomputable def areaBelowXAxis (p : Parallelogram) : ℝ :=
  sorry -- Actual calculation would go here

/-- The main theorem to prove -/
theorem probability_below_x_axis (p : Parallelogram) 
  (h1 : p.e = ⟨5, 4⟩) 
  (h2 : p.f = ⟨-1, -4⟩) 
  (h3 : p.g = ⟨-7, -2⟩) 
  (h4 : p.h = ⟨1, 6⟩) : 
  areaBelowXAxis p / parallelogramArea p = 1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_below_x_axis_l421_42139


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_incenter_combination_specific_l421_42182

/-- Triangle DEF with side lengths and incenter J -/
structure Triangle where
  D : ℝ × ℝ
  E : ℝ × ℝ
  F : ℝ × ℝ
  d : ℝ
  e : ℝ
  f : ℝ
  J : ℝ × ℝ

/-- The incenter of a triangle expressed as a linear combination of its vertices -/
noncomputable def incenter_combination (t : Triangle) : ℝ × ℝ × ℝ :=
  (t.d / (t.d + t.e + t.f), t.e / (t.d + t.e + t.f), t.f / (t.d + t.e + t.f))

/-- Theorem: The incenter of triangle DEF with given side lengths is (5/12, 1/4, 1/3) -/
theorem incenter_combination_specific :
  ∃ (t : Triangle), t.d = 10 ∧ t.e = 6 ∧ t.f = 8 ∧
  incenter_combination t = (5/12, 1/4, 1/3) := by
  sorry

/-- The circumcenter of the triangle -/
def O : ℝ × ℝ := sorry

/-- Condition: OJ is perpendicular to DE -/
def OJ_perpendicular_DE (t : Triangle) : Prop := sorry

#check incenter_combination_specific

end NUMINAMATH_CALUDE_ERRORFEEDBACK_incenter_combination_specific_l421_42182


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_advertising_quota_met_l421_42175

/-- Represents a television program with its runtime and advertising percentage -/
structure TVProgram where
  runtime : ℝ
  adPercentage : ℝ

/-- Calculates the total commercial time for a set of TV programs -/
noncomputable def totalCommercialTime (programs : List TVProgram) : ℝ :=
  programs.foldl (fun acc p => acc + p.runtime * p.adPercentage / 100) 0

/-- Theorem stating that the given set of TV programs meets the advertising quota -/
theorem advertising_quota_met :
  let programs := [
    TVProgram.mk 30 20,
    TVProgram.mk 30 25,
    TVProgram.mk 30 30,
    TVProgram.mk 30 35,
    TVProgram.mk 30 40,
    TVProgram.mk 30 45
  ]
  let totalTime := totalCommercialTime programs
  totalTime = 58.5 ∧ totalTime > 50 := by
  sorry

#eval "Theorem statement compiled successfully."

end NUMINAMATH_CALUDE_ERRORFEEDBACK_advertising_quota_met_l421_42175


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_in_interval_l421_42149

noncomputable def f (x : ℝ) : ℝ := Real.log x / Real.log 10 + x - 2

theorem root_in_interval : ∃ x₀ : ℝ, f x₀ = 0 ∧ 1 < x₀ ∧ x₀ < 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_in_interval_l421_42149


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_opposite_sides_reasoning_l421_42102

-- Define the properties of a parallelogram
structure Parallelogram :=
  (opposite_sides_parallel : Bool)
  (opposite_sides_equal : Bool)

-- Define a rectangle as a special type of parallelogram
structure Rectangle extends Parallelogram

-- Define the types of reasoning
inductive ReasoningType
  | Inductive
  | Analogical
  | Deductive
  | Plausibility

-- State the theorem
theorem rectangle_opposite_sides_reasoning :
  (∀ p : Parallelogram, p.opposite_sides_parallel ∧ p.opposite_sides_equal) →
  (Rectangle → Parallelogram) →
  ReasoningType := by
  intro h1 h2
  exact ReasoningType.Deductive


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_opposite_sides_reasoning_l421_42102


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_rate_of_change_reciprocal_l421_42133

noncomputable def f (x : ℝ) : ℝ := 1 / x

noncomputable def averageRateOfChange (f : ℝ → ℝ) (a b : ℝ) : ℝ :=
  (f b - f a) / (b - a)

theorem average_rate_of_change_reciprocal :
  averageRateOfChange f 1 2 = -1/2 := by
  -- Unfold the definition of averageRateOfChange
  unfold averageRateOfChange
  -- Unfold the definition of f
  unfold f
  -- Simplify the expression
  simp
  -- The rest of the proof would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_rate_of_change_reciprocal_l421_42133


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_original_ghee_composition_l421_42187

/-- Represents the composition of a ghee mixture -/
structure GheeMixture where
  total : ℝ
  pure_ghee : ℝ
  vanaspati : ℝ
  pure_ghee_percentage : ℝ
  vanaspati_percentage : ℝ

/-- The original ghee mixture -/
noncomputable def original_mixture (pure_ghee_percentage : ℝ) : GheeMixture :=
  { total := 30
  , pure_ghee := 30 * (pure_ghee_percentage / 100)
  , vanaspati := 30 * (1 - pure_ghee_percentage / 100)
  , pure_ghee_percentage := pure_ghee_percentage
  , vanaspati_percentage := 100 - pure_ghee_percentage }

/-- The mixture after adding 20 kg of pure ghee -/
noncomputable def new_mixture (original : GheeMixture) : GheeMixture :=
  { total := original.total + 20
  , pure_ghee := original.pure_ghee + 20
  , vanaspati := original.vanaspati
  , pure_ghee_percentage := (original.pure_ghee + 20) / (original.total + 20) * 100
  , vanaspati_percentage := 30 }

theorem original_ghee_composition :
  ∀ (pure_ghee_percentage : ℝ),
  let original := original_mixture pure_ghee_percentage
  let new_mix := new_mixture original
  new_mix.vanaspati_percentage = 30 →
  original.pure_ghee_percentage = 50 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_original_ghee_composition_l421_42187


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_f_period_l421_42153

noncomputable def f (x : ℝ) := Real.sin x * Real.cos x - Real.sqrt 3 * Real.cos (Real.pi + x) * Real.cos x

theorem f_properties :
  -- The smallest positive period of f is π
  (∃ (T : ℝ), T > 0 ∧ (∀ (x : ℝ), f (x + T) = f x) ∧
    (∀ (T' : ℝ), T' > 0 → (∀ (x : ℝ), f (x + T') = f x) → T ≤ T')) ∧
  -- Given sin(π+α) = 4/5 and |α| < π/2, f(α) - √3/2 = (-24-7√3)/50
  (∀ (α : ℝ), Real.sin (Real.pi + α) = 4/5 ∧ |α| < Real.pi/2 →
    f α - Real.sqrt 3 / 2 = (-24 - 7 * Real.sqrt 3) / 50) := by
  sorry

-- Separate theorem for the period
theorem f_period : ∃ (T : ℝ), T = Real.pi ∧ T > 0 ∧ (∀ (x : ℝ), f (x + T) = f x) ∧
    (∀ (T' : ℝ), T' > 0 → (∀ (x : ℝ), f (x + T') = f x) → T ≤ T') := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_f_period_l421_42153


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_sum_squared_l421_42195

noncomputable def g (x : ℝ) : ℝ := Real.sqrt 28 + 64 / x

def root_equation (x : ℝ) : Prop := x = g (g (g (g (g x))))

noncomputable def B : ℝ := |((Real.sqrt 28 + Real.sqrt 284) / 2)| + |((Real.sqrt 28 - Real.sqrt 284) / 2)|

theorem root_sum_squared : B^2 = 284 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_sum_squared_l421_42195


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_foci_distance_l421_42189

-- Define the ellipse
def is_on_ellipse (x y : ℝ) : Prop := x^2 / 100 + y^2 / 36 = 1

-- Define the distance between two points
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

-- Theorem statement
theorem ellipse_foci_distance 
  (x y x1 y1 x2 y2 : ℝ) 
  (h_ellipse : is_on_ellipse x y) 
  (h_focus1 : distance x y x1 y1 = 6) 
  (h_foci : distance x1 y1 x2 y2 = 20) : 
  distance x y x2 y2 = 14 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_foci_distance_l421_42189


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_10_l421_42174

noncomputable def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  a₁ + (n - 1 : ℝ) * d

noncomputable def arithmetic_sequence_sum (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  n * (2 * a₁ + (n - 1 : ℝ) * d) / 2

theorem arithmetic_sequence_sum_10 (a₁ d : ℝ) :
  arithmetic_sequence a₁ d 4 = 23 →
  arithmetic_sequence a₁ d 8 = 55 →
  arithmetic_sequence_sum a₁ d 10 = 350 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_10_l421_42174


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_visibleAreaRatio_paperFoldingRatio_l421_42185

/-- Represents a rectangular piece of paper with specific folding properties. -/
structure Paper where
  width : ℝ
  length : ℝ
  area : ℝ
  visibleArea : ℝ

/-- Constructs a Paper with the given properties. -/
noncomputable def makePaper (w : ℝ) : Paper where
  width := w
  length := 2 * w
  area := 2 * w^2
  visibleArea := 2 * w^2 - (Real.sqrt 10 * w^2) / 12

/-- Theorem stating the ratio of visible area to total area after folding. -/
theorem visibleAreaRatio (p : Paper) :
  p.visibleArea / p.area = 1 - Real.sqrt 10 / 24 :=
by sorry

/-- Main theorem proving the ratio for any valid rectangular paper. -/
theorem paperFoldingRatio (w : ℝ) (h : w > 0) :
  let p := makePaper w
  p.visibleArea / p.area = 1 - Real.sqrt 10 / 24 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_visibleAreaRatio_paperFoldingRatio_l421_42185


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_common_point_for_six_circles_l421_42122

/-- Represents a circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ
  radius_pos : radius > 0

/-- Checks if a point is inside a circle -/
def is_inside (c : Circle) (p : ℝ × ℝ) : Prop :=
  (c.center.1 - p.1)^2 + (c.center.2 - p.2)^2 < c.radius^2

/-- Theorem: Given 6 circles on a plane where the center of each circle does not belong to any of the other 5 circles, these 6 circles do not have a common point -/
theorem no_common_point_for_six_circles (c1 c2 c3 c4 c5 c6 : Circle) 
  (h_distinct : ∀ (i j : Fin 6), i ≠ j → 
    ¬is_inside (match i with
      | ⟨0, _⟩ => c1 | ⟨1, _⟩ => c2 | ⟨2, _⟩ => c3
      | ⟨3, _⟩ => c4 | ⟨4, _⟩ => c5 | ⟨5, _⟩ => c6
      | _ => c1  -- This case should never occur due to Fin 6
    ) (match j with
      | ⟨0, _⟩ => c1.center | ⟨1, _⟩ => c2.center | ⟨2, _⟩ => c3.center
      | ⟨3, _⟩ => c4.center | ⟨4, _⟩ => c5.center | ⟨5, _⟩ => c6.center
      | _ => c1.center  -- This case should never occur due to Fin 6
    )) :
  ¬∃ (p : ℝ × ℝ), (is_inside c1 p ∧ is_inside c2 p ∧ is_inside c3 p ∧ is_inside c4 p ∧ is_inside c5 p ∧ is_inside c6 p) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_common_point_for_six_circles_l421_42122


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_crayons_remaining_l421_42199

theorem crayons_remaining (initial_crayons : ℕ) (kiley_fraction : ℚ) (joe_fraction : ℚ) : 
  initial_crayons = 48 →
  kiley_fraction = 1/4 →
  joe_fraction = 1/2 →
  initial_crayons - (kiley_fraction * ↑initial_crayons).floor - 
    (joe_fraction * ↑(initial_crayons - (kiley_fraction * ↑initial_crayons).floor)).floor = 18 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_crayons_remaining_l421_42199


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_two_irrational_l421_42140

theorem cube_root_two_irrational : ¬ ∃ (n m : ℕ), 0 < n ∧ 0 < m ∧ Nat.Coprime n m ∧ (n : ℚ) / m = 2^(1/3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_two_irrational_l421_42140


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l421_42112

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a^x - 1) / (a^x + 1)

-- Theorem statement
theorem f_properties (a : ℝ) (h : a > 1) :
  (∀ x : ℝ, f a (-x) = -f a x) ∧
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → f a x₁ < f a x₂) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l421_42112


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_specific_parallel_lines_l421_42170

/-- The distance between two parallel lines -/
noncomputable def distance_between_parallel_lines (A B C₁ C₂ : ℝ) : ℝ :=
  |C₁ - C₂| / Real.sqrt (A^2 + B^2)

/-- Theorem: The distance between the parallel lines x + 3y - 5 = 0 and x + 3y - 10 = 0 is √10/2 -/
theorem distance_specific_parallel_lines :
  distance_between_parallel_lines 1 3 (-5) (-10) = Real.sqrt 10 / 2 := by
  sorry

-- Remove the #eval statement as it's not computable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_specific_parallel_lines_l421_42170


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_enclosed_set_l421_42109

-- Define the equation of the boundary
def boundary (x y : ℝ) : Prop := abs (2 * x) + abs (3 * y) = 12

-- Define the set of points enclosed by the boundary
def enclosed_set : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∀ (x y : ℝ), p = (x, y) → abs (2 * x) + abs (3 * y) ≤ 12}

-- State the theorem
theorem area_of_enclosed_set : MeasureTheory.volume enclosed_set = 48 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_enclosed_set_l421_42109


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rod_length_for_42_75kg_l421_42196

/-- Represents the properties of a uniform steel rod -/
structure SteelRod where
  /-- The weight per unit length of the rod (kg/m) -/
  weight_density : ℝ
  /-- Assertion that the weight density is positive -/
  weight_density_pos : weight_density > 0

/-- Calculates the length of a steel rod given its weight -/
noncomputable def rod_length (rod : SteelRod) (weight : ℝ) : ℝ :=
  weight / rod.weight_density

/-- Theorem stating the length of a steel rod weighing 42.75 kg -/
theorem rod_length_for_42_75kg (rod : SteelRod) 
  (h : rod_length rod 30.4 = 8) : 
  rod_length rod 42.75 = 11.25 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rod_length_for_42_75kg_l421_42196


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_paint_for_smaller_cubes_l421_42167

/-- The amount of paint needed for a cube given its side length -/
noncomputable def paint_needed (side_length : ℝ) : ℝ :=
  side_length^2 / 16

/-- Theorem stating the relationship between paint needed for large and small cubes -/
theorem paint_for_smaller_cubes :
  let large_cube_side := (4 : ℝ)
  let small_cube_side := (2 : ℝ)
  let num_small_cubes := (125 : ℝ)
  paint_needed large_cube_side = 1 →
  num_small_cubes * paint_needed small_cube_side = 31.25 :=
by
  -- Introduce the local variables
  intro large_cube_side small_cube_side num_small_cubes h
  -- Unfold the definition of paint_needed
  unfold paint_needed
  -- Simplify the expressions
  simp [h]
  -- The proof is completed with sorry
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_paint_for_smaller_cubes_l421_42167


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rabbit_travel_time_l421_42194

/-- Proves that a rabbit traveling at 5 miles per hour takes 24 minutes to cover 2 miles -/
theorem rabbit_travel_time :
  let rabbit_speed : ℝ := 5  -- Speed in miles per hour
  let travel_distance : ℝ := 2  -- Distance in miles
  let minutes_per_hour : ℝ := 60  -- Conversion factor
  let travel_time_minutes : ℝ := travel_distance / rabbit_speed * minutes_per_hour
  travel_time_minutes = 24 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rabbit_travel_time_l421_42194


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_larry_wins_two_thirds_l421_42142

/-- The probability of knocking the bottle off the ledge on a single throw -/
noncomputable def knock_probability : ℝ := 1/2

/-- The probability that Larry wins the game -/
noncomputable def larry_win_probability : ℝ := 2/3

/-- Theorem stating that Larry's probability of winning the game is 2/3 -/
theorem larry_wins_two_thirds :
  larry_win_probability = 2/3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_larry_wins_two_thirds_l421_42142


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_first_two_terms_l421_42126

/-- A sequence of positive integers defined by a recurrence relation -/
def IntegerSequence (a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, n ≥ 1 → a (n + 2) = (a n + 2023) / (1 + a (n + 1))

/-- The theorem stating the minimum possible value of a₁ + a₂ -/
theorem min_sum_first_two_terms (a : ℕ → ℕ) (h : IntegerSequence a) :
    ∃ m : ℕ, m = a 1 + a 2 ∧ (∀ k : ℕ → ℕ, IntegerSequence k → a 1 + a 2 ≤ k 1 + k 2) ∧ m = 136 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_first_two_terms_l421_42126


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_f_equals_four_thirds_l421_42198

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if -1 ≤ x ∧ x ≤ 0 then x^2
  else if 0 < x ∧ x ≤ 1 then 1
  else 0

-- State the theorem
theorem integral_f_equals_four_thirds :
  ∫ x in Set.Icc (-1) 1, f x = 4/3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_f_equals_four_thirds_l421_42198


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_center_tangent_to_parabola_l421_42186

/-- A circle passing through (1, 0) and tangent to y = x^2 at (1, 1) has center (2, 1/2) -/
theorem circle_center_tangent_to_parabola :
  ∃ (center : ℝ × ℝ),
    let (a, b) := center
    (∀ (x y : ℝ), y = x^2 → (x - 1)^2 + (y - 1)^2 = (a - 1)^2 + (b - 1)^2 → x = 1 ∧ y = 1) ∧
    (a - 1)^2 + b^2 = (a - 1)^2 + (b - 1)^2 ∧
    center = (2, 1/2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_center_tangent_to_parabola_l421_42186


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_G_is_perfect_square_iff_n_eq_6_l421_42115

/-- G is defined as (8x^2 + 24x + 3n) / 8 -/
noncomputable def G (x : ℝ) (n : ℝ) : ℝ := (8*x^2 + 24*x + 3*n) / 8

/-- A linear expression in x -/
def linearExpr (c d : ℝ) (x : ℝ) : ℝ := c*x + d

/-- Theorem stating that G is a perfect square if and only if n = 6 -/
theorem G_is_perfect_square_iff_n_eq_6 :
  ∃! n : ℝ, ∀ x : ℝ, ∃ c d : ℝ, G x n = (linearExpr c d x)^2 ∧ n = 6 := by
  sorry

#check G_is_perfect_square_iff_n_eq_6

end NUMINAMATH_CALUDE_ERRORFEEDBACK_G_is_perfect_square_iff_n_eq_6_l421_42115


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_baez_marbles_l421_42176

def initial_marbles : ℕ := 25
def loss_percentage : ℚ := 1/5
def trade_fraction : ℚ := 1/3

def remaining_after_loss (initial : ℕ) (loss : ℚ) : ℕ :=
  (initial : ℤ) - Int.floor (loss * initial) |>.toNat

def remaining_after_trade (marbles : ℕ) (fraction : ℚ) : ℕ :=
  (marbles : ℤ) - Int.floor (fraction * marbles) |>.toNat

def final_marbles (initial : ℕ) (loss : ℚ) (trade : ℚ) : ℕ :=
  let after_loss := remaining_after_loss initial loss
  let after_trade := remaining_after_trade after_loss trade
  after_trade + 2 * after_trade

theorem baez_marbles :
  final_marbles initial_marbles loss_percentage trade_fraction = 42 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_baez_marbles_l421_42176


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_friend_bill_equals_12_7_l421_42107

-- Define the cost of a taco
def taco_cost : ℚ := 9/10

-- Define your order
def your_tacos : ℕ := 2
def your_enchiladas : ℕ := 3

-- Define your friend's order
def friend_tacos : ℕ := 3
def friend_enchiladas : ℕ := 5

-- Define your bill before tax
def your_bill : ℚ := 39/5

-- Calculate the cost of an enchilada based on your bill
noncomputable def enchilada_cost : ℚ := (your_bill - (your_tacos * taco_cost)) / your_enchiladas

-- Calculate your friend's bill
noncomputable def friend_bill : ℚ := friend_tacos * taco_cost + friend_enchiladas * enchilada_cost

-- Theorem to prove
theorem friend_bill_equals_12_7 : friend_bill = 127/10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_friend_bill_equals_12_7_l421_42107


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_theorem_l421_42130

theorem remainder_theorem (x y : ℕ) 
  (hx : x % 13 = 7)
  (hy : y % 17 = 11) : 
  (3 * x + 5 * y) % 221 = 76 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_theorem_l421_42130


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vectors_k_l421_42103

-- Define vectors as functions from Fin 3 to ℝ
def a : Fin 3 → ℝ := ![1, 0, -1]
def b : Fin 3 → ℝ := ![2, 1, 0]

-- Helper function for scalar multiplication
def scalarMul (k : ℝ) (v : Fin 3 → ℝ) : Fin 3 → ℝ := λ i => k * v i

-- Helper function for vector addition
def vecAdd (u v : Fin 3 → ℝ) : Fin 3 → ℝ := λ i => u i + v i

-- Helper function for dot product
def dotProduct (u v : Fin 3 → ℝ) : ℝ := (u 0) * (v 0) + (u 1) * (v 1) + (u 2) * (v 2)

theorem perpendicular_vectors_k (k : ℝ) :
  dotProduct (vecAdd (scalarMul k a) b) (vecAdd (scalarMul 2 a) (scalarMul (-1) b)) = 0 → k = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vectors_k_l421_42103


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_l421_42177

-- Define the function f(x) = √(2x - 3)
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (2 * x - 3)

-- State the theorem about the domain of f
theorem f_domain : Set.Ici (3/2 : ℝ) = {x : ℝ | ∃ y, f x = y} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_l421_42177


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_values_and_intersection_l421_42172

/-- The cubic function f(x) = x^3 - x^2 - x + a -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - x^2 - x + a

/-- The derivative of f(x) -/
def f' (x : ℝ) : ℝ := 3*x^2 - 2*x - 1

theorem extreme_values_and_intersection (a : ℝ) :
  (∃ x_max : ℝ, ∀ x : ℝ, f a x ≤ f a x_max ∧ f a x_max = a + 5/27) ∧
  (∃ x_min : ℝ, ∀ x : ℝ, f a x ≥ f a x_min ∧ f a x_min = a - 1) ∧
  (∃! x : ℝ, f a x = 0) ↔ a ∈ Set.Ioi 1 ∪ Set.Iio (-5/27) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_values_and_intersection_l421_42172


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_apex_cosine_l421_42173

/-- An isosceles triangle with perimeter five times the base has apex angle cosine 7/8 -/
theorem isosceles_triangle_apex_cosine (a b c : ℝ) (h_isosceles : a = b) 
  (h_perimeter : a + b + c = 5 * c) : 
  (a^2 + b^2 - c^2) / (2 * a * b) = 7/8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_apex_cosine_l421_42173


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_is_9_sqrt_3_l421_42125

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 16 - y^2 / 9 = 1

-- Define the foci F₁ and F₂
noncomputable def F₁ : ℝ × ℝ := sorry
noncomputable def F₂ : ℝ × ℝ := sorry

-- Define a point P on the hyperbola
noncomputable def P : ℝ × ℝ := sorry

-- State that P is on the hyperbola
axiom P_on_hyperbola : hyperbola P.1 P.2

-- Define the angle F₁PF₂
noncomputable def angle_F₁PF₂ : ℝ := sorry

-- State that the angle F₁PF₂ is 60°
axiom angle_60 : angle_F₁PF₂ = Real.pi / 3

-- Define the area of triangle F₁PF₂
noncomputable def area_F₁PF₂ : ℝ := sorry

-- Theorem to prove
theorem area_is_9_sqrt_3 : area_F₁PF₂ = 9 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_is_9_sqrt_3_l421_42125


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_magic_square_sum_l421_42158

/-- Represents a 3x3 magic square with given values and variables -/
structure MagicSquare where
  a : ℤ
  b : ℤ
  c : ℤ
  d : ℤ
  e : ℤ
  sum_equal : ∀ (r c d : ℤ → ℤ → ℤ),
    (r 1 1 = 30) ∧ (r 1 2 = d) ∧ (r 1 3 = 15) ∧
    (r 2 1 = 12) ∧ (r 2 2 = b) ∧ (r 2 3 = e) ∧
    (r 3 1 = a)  ∧ (r 3 2 = 36) ∧ (r 3 3 = c) ∧
    (c 1 1 = 30) ∧ (c 2 1 = 12) ∧ (c 3 1 = a) ∧
    (c 1 2 = d)  ∧ (c 2 2 = b)  ∧ (c 3 2 = 36) ∧
    (c 1 3 = 15) ∧ (c 2 3 = e)  ∧ (c 3 3 = c) ∧
    (d 1 1 = 30) ∧ (d 2 2 = b)  ∧ (d 3 3 = c) ∧
    (d 1 3 = a)  ∧ (d 2 2 = d)  ∧ (d 3 1 = e) →
    (r 1 1 + r 1 2 + r 1 3 = r 2 1 + r 2 2 + r 2 3) ∧
    (r 2 1 + r 2 2 + r 2 3 = r 3 1 + r 3 2 + r 3 3) ∧
    (c 1 1 + c 2 1 + c 3 1 = c 1 2 + c 2 2 + c 3 2) ∧
    (c 1 2 + c 2 2 + c 3 2 = c 1 3 + c 2 3 + c 3 3) ∧
    (d 1 1 + d 2 2 + d 3 3 = d 1 3 + d 2 2 + d 3 1) ∧
    (r 1 1 + r 1 2 + r 1 3 = c 1 1 + c 2 1 + c 3 1) ∧
    (c 1 1 + c 2 1 + c 3 1 = d 1 1 + d 2 2 + d 3 3)

/-- Theorem: In a 3x3 magic square with the given values, d + e = 87 -/
theorem magic_square_sum (ms : MagicSquare) : ms.d + ms.e = 87 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_magic_square_sum_l421_42158


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_and_range_l421_42155

-- Define the function y
noncomputable def y (t x : ℝ) : ℝ := 1 - 2*t - 2*t*x + 2*x^2

-- Define the minimum value function f(t)
noncomputable def f (t : ℝ) : ℝ :=
  if t < -2 then 3
  else if t ≤ 2 then -t^2/2 - 2*t + 1
  else -4*t + 3

-- Theorem statement
theorem min_value_and_range :
  (∀ t x : ℝ, -1 ≤ x ∧ x ≤ 1 → y t x ≥ f t) ∧
  (∀ t : ℝ, -2 ≤ t ∧ t ≤ 0 → 1 ≤ f t ∧ f t ≤ 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_and_range_l421_42155


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_pair_properties_l421_42105

/-- Two ellipses C₁ and C₂ with shared focus -/
structure EllipsePair (a : ℝ) where
  C₁ : Set (ℝ × ℝ)
  C₂ : Set (ℝ × ℝ)
  O : ℝ × ℝ
  F₁ : ℝ × ℝ
  h_C₁ : C₁ = {(x, y) | 2 * x^2 - y^2 = 2 * a^2}
  h_C₂ : C₂ = {(x, y) | y^2 = -4 * Real.sqrt 3 * a * x}
  h_O : O = (0, 0)
  h_F₁ : F₁ = (-Real.sqrt 3 * a, 0)

/-- Helper function to calculate the area of a triangle -/
noncomputable def area_triangle (O A B : ℝ × ℝ) : ℝ := sorry

/-- Main theorem about the properties of the ellipse pair -/
theorem ellipse_pair_properties (a : ℝ) (h_a : a > 0) (ep : EllipsePair a) :
  (∃ (P Q : ℝ × ℝ), P ∈ ep.C₁ ∩ ep.C₂ ∧ Q ∈ ep.C₁ ∩ ep.C₂ ∧ P ≠ Q) ∧
  (∃ (A B : ℝ × ℝ), A ∈ ep.C₂ ∧ B ∈ ep.C₂ ∧
    (∀ (X Y : ℝ × ℝ), X ∈ ep.C₂ ∧ Y ∈ ep.C₂ →
      area_triangle ep.O X Y ≥ area_triangle ep.O A B)) ∧
  (∃ (k : ℝ), ∀ (x y : ℝ), (x, y) ∈ ep.C₂ ∧ y = k * (x + Real.sqrt 3 * a) →
    x = -Real.sqrt 3 * a) ∧
  (∃ (A B : ℝ × ℝ), A ∈ ep.C₂ ∧ B ∈ ep.C₂ ∧
    area_triangle ep.O A B = 6 * a^2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_pair_properties_l421_42105


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l421_42106

noncomputable def f (x : ℝ) := Real.sin ((13 * Real.pi / 2) - x)

theorem f_properties :
  (∀ x, f (-x) = f x) ∧
  (∀ T > 0, (∀ x, f (x + T) = f x) → T ≥ 2 * Real.pi) ∧
  (∀ x, f (x + 2 * Real.pi) = f x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l421_42106


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_red_numbers_bound_l421_42119

def is_red (n : ℕ) (red : Finset ℕ) : Prop :=
  red ⊆ Finset.range (n + 1) ∧
  ∀ a ∈ red, a ≠ 1 → ∀ k : ℕ, k * a ≤ n → k * a ∈ red

theorem red_numbers_bound (n : ℕ) (red : Finset ℕ) (h : is_red n red) :
  red.card ≤ Nat.totient n :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_red_numbers_bound_l421_42119


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_single_frog_birth_l421_42181

def frog_population : ℕ → ℕ → ℕ
  | _, 0 => 2  -- 2020
  | _, 1 => 9  -- 2021
  | n, m + 2 => 1 + Int.natAbs (frog_population n (m + 1) - frog_population n m)

theorem first_single_frog_birth (n : ℕ) :
  (∀ m < 13, frog_population n m ≠ 1) ∧ frog_population n 13 = 1 :=
by
  sorry

#eval frog_population 0 13

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_single_frog_birth_l421_42181


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l421_42188

-- Define the function f
noncomputable def f (A ω φ x : ℝ) : ℝ := A * Real.sin (ω * x + φ)

-- State the theorem
theorem function_properties :
  ∀ (A ω φ : ℝ),
  A > 0 → ω > 0 → 0 < φ → φ < Real.pi / 2 →
  (∀ x, f A ω φ x ≥ -4) →
  f A ω φ 0 = 2 * Real.sqrt 2 →
  (∀ x, f A ω φ (x + Real.pi / ω) = f A ω φ x) →
  ∃ (k : ℤ),
    A = 4 ∧
    ω = 1 ∧
    φ = Real.pi / 4 ∧
    (∀ x, f A ω φ x = 4 * Real.sin (x + Real.pi / 4)) ∧
    (∀ x ∈ Set.Icc (-Real.pi/2) (Real.pi/2), f A ω φ x ≤ 4) ∧
    (∀ x ∈ Set.Icc (-Real.pi/2) (Real.pi/2), f A ω φ x ≥ -2 * Real.sqrt 2) ∧
    (∀ x ∈ Set.Ioo (Real.pi/2) Real.pi,
      f A ω φ x = 1 →
      Real.cos (x + 5*Real.pi/12) = (-3*Real.sqrt 5 - 1) / 8) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l421_42188


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_five_digit_multiplication_sum_l421_42121

/-- A five-digit number represented as a list of its digits -/
def FiveDigitNumber := List Nat

/-- Check if a number is a valid five-digit number -/
def isValidFiveDigitNumber (n : FiveDigitNumber) : Prop :=
  n.length = 5 ∧ n.all (· < 10)

/-- Convert a five-digit number to its integer value -/
def toInt (n : FiveDigitNumber) : Nat :=
  n.foldl (fun acc d => acc * 10 + d) 0

theorem five_digit_multiplication_sum (PQRST : FiveDigitNumber) :
  isValidFiveDigitNumber PQRST →
  4 * toInt PQRST = 410256 →
  PQRST.sum = 14 := by
  sorry

#check five_digit_multiplication_sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_five_digit_multiplication_sum_l421_42121


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_computer_table_markup_l421_42190

noncomputable def percentage_markup (selling_price cost_price : ℝ) : ℝ :=
  (selling_price - cost_price) / cost_price * 100

theorem computer_table_markup :
  let selling_price : ℝ := 1000
  let cost_price : ℝ := 800
  percentage_markup selling_price cost_price = 25 := by
  -- Unfold the definition of percentage_markup
  unfold percentage_markup
  -- Perform the calculation
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_computer_table_markup_l421_42190


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sun_perimeter_l421_42101

noncomputable def sectorPerimeter (radius : ℝ) (centralAngle : ℝ) : ℝ :=
  2 * radius + (centralAngle / 360) * (2 * Real.pi * radius)

theorem sun_perimeter :
  sectorPerimeter 2 120 = (4/3) * Real.pi + 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sun_perimeter_l421_42101


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_one_zero_l421_42146

-- Define the function f(x) = (x+1)ln(x)
noncomputable def f (x : ℝ) : ℝ := (x + 1) * Real.log x

-- State the theorem
theorem f_has_one_zero :
  ∃! x : ℝ, x > 0 ∧ f x = 0 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_one_zero_l421_42146


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezium_triangle_areas_l421_42178

/-- Represents a trapezium with parallel sides a and b, and altitude h -/
structure Trapezium where
  a : ℝ
  b : ℝ
  h : ℝ
  h_pos : h > 0
  a_lt_b : a < b

/-- Calculates the area of the smaller triangle formed by extending the non-parallel sides of the trapezium -/
noncomputable def smaller_triangle_area (t : Trapezium) : ℝ :=
  (t.a ^ 2 * t.h) / (2 * (t.b - t.a))

/-- Calculates the area of the larger triangle formed by extending the non-parallel sides of the trapezium -/
noncomputable def larger_triangle_area (t : Trapezium) : ℝ :=
  (t.b ^ 2 * t.h) / (2 * (t.b - t.a))

/-- Theorem stating that the areas of the triangles formed by extending the non-parallel sides of a trapezium
    are correctly calculated by the smaller_triangle_area and larger_triangle_area functions -/
theorem trapezium_triangle_areas (t : Trapezium) :
  smaller_triangle_area t = (t.a ^ 2 * t.h) / (2 * (t.b - t.a)) ∧
  larger_triangle_area t = (t.b ^ 2 * t.h) / (2 * (t.b - t.a)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezium_triangle_areas_l421_42178


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mall_revenue_minimum_l421_42197

noncomputable def f (x : ℝ) : ℝ := 5 + 5/x

noncomputable def g (x : ℝ) : ℝ := 20*x + 500

noncomputable def y (x : ℝ) : ℝ := 100 * (f x * g x)

theorem mall_revenue_minimum :
  ∀ x : ℝ, 1 ≤ x → x ≤ 15 → y x ≥ 360000 ∧ y 5 = 360000 := by
  intro x h1 h2
  sorry

#check mall_revenue_minimum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mall_revenue_minimum_l421_42197


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_y_increasing_and_divergent_l421_42165

open MeasureTheory Interval Real

/-- Two continuous, distinct functions from [0,1] to (0,+∞) with equal integrals -/
structure EqualIntegralFunctions where
  f : ℝ → ℝ
  g : ℝ → ℝ
  hf_continuous : Continuous f
  hg_continuous : Continuous g
  hf_pos : ∀ x ∈ Set.Icc 0 1, f x > 0
  hg_pos : ∀ x ∈ Set.Icc 0 1, g x > 0
  hf_ne_g : f ≠ g
  h_equal_integral : ∫ x in Set.Icc 0 1, f x = ∫ x in Set.Icc 0 1, g x

/-- The sequence yn defined by the integral of f^(n+1) / g^n -/
noncomputable def y (E : EqualIntegralFunctions) (n : ℕ) : ℝ :=
  ∫ x in Set.Icc 0 1, (E.f x)^(n+1) / (E.g x)^n

/-- The main theorem: (yn) is increasing and divergent -/
theorem y_increasing_and_divergent (E : EqualIntegralFunctions) :
    (Monotone (y E)) ∧ (Filter.Tendsto (y E) Filter.atTop Filter.atTop) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_y_increasing_and_divergent_l421_42165


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_triangle_area_l421_42180

noncomputable section

/-- The curve function -/
def f (x : ℝ) : ℝ := 3 * x * Real.log x + x

/-- The derivative of the curve function -/
def f' (x : ℝ) : ℝ := 4 + 3 * Real.log x

/-- The tangent line at x = 1 -/
def tangent_line (x : ℝ) : ℝ := 4 * x - 3

/-- The x-intercept of the tangent line -/
def x_intercept : ℝ := 3 / 2

/-- The y-intercept of the tangent line -/
def y_intercept : ℝ := -3

/-- The area of the triangle -/
def triangle_area : ℝ := (x_intercept * (-y_intercept)) / 2

theorem tangent_triangle_area :
  triangle_area = 9 / 4 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_triangle_area_l421_42180


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_minus_even_digits_count_l421_42171

def is_odd_digit (d : Nat) : Bool := d % 2 = 1 && d ≤ 9

def is_even_digit (d : Nat) : Bool := d % 2 = 0 && d ≤ 8

def has_only_odd_digits (n : Nat) : Bool :=
  n.digits 10 |>.all is_odd_digit

def has_only_even_digits (n : Nat) : Bool :=
  n.digits 10 |>.all is_even_digit

def count_odd_digit_numbers (upper : Nat) : Nat :=
  (List.range upper).filter has_only_odd_digits |>.length

def count_even_digit_numbers (upper : Nat) : Nat :=
  (List.range upper).filter has_only_even_digits |>.length

theorem odd_minus_even_digits_count :
  count_odd_digit_numbers 60001 - count_even_digit_numbers 60001 = 780 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_minus_even_digits_count_l421_42171
