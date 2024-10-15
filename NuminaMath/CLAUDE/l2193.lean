import Mathlib

namespace NUMINAMATH_CALUDE_one_plane_through_line_parallel_to_skew_line_l2193_219385

-- Define the concept of a line in 3D space
structure Line3D where
  -- You might define a line using a point and a direction vector
  -- But for simplicity, we'll just declare it as an opaque type
  dummy : Unit

-- Define the concept of a plane in 3D space
structure Plane3D where
  -- Similar to Line3D, we'll keep this as an opaque type for simplicity
  dummy : Unit

-- Define what it means for two lines to be skew
def are_skew (a b : Line3D) : Prop :=
  -- Two lines are skew if they are neither intersecting nor parallel
  sorry

-- Define what it means for a plane to contain a line
def plane_contains_line (p : Plane3D) (l : Line3D) : Prop :=
  sorry

-- Define what it means for a plane to be parallel to a line
def plane_parallel_to_line (p : Plane3D) (l : Line3D) : Prop :=
  sorry

-- The main theorem
theorem one_plane_through_line_parallel_to_skew_line 
  (a b : Line3D) (h : are_skew a b) : 
  ∃! p : Plane3D, plane_contains_line p a ∧ plane_parallel_to_line p b :=
sorry

end NUMINAMATH_CALUDE_one_plane_through_line_parallel_to_skew_line_l2193_219385


namespace NUMINAMATH_CALUDE_fraction_simplification_l2193_219356

theorem fraction_simplification (a b : ℝ) (ha : a ≠ 0) (hab : a ≠ b) :
  (a - b) / a / (a - (2 * a * b - b^2) / a) = 1 / (a - b) := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2193_219356


namespace NUMINAMATH_CALUDE_prime_divisibility_l2193_219336

theorem prime_divisibility (p q : Nat) 
  (hp : Prime p) (hq : Prime q) (hpq : p ≠ q) (hp5 : p > 5) (hq5 : q > 5) :
  (p ∣ (5^q - 2^q) → q ∣ (p - 1)) ∧ ¬(p*q ∣ (5^p - 2^p)*(5^q - 2^q)) := by
  sorry

end NUMINAMATH_CALUDE_prime_divisibility_l2193_219336


namespace NUMINAMATH_CALUDE_line_increase_l2193_219361

/-- Given a line in the Cartesian plane where an increase of 2 units in x
    corresponds to an increase of 5 units in y, prove that an increase of 8 units
    in x will result in an increase of 20 units in y. -/
theorem line_increase (f : ℝ → ℝ) (h : ∀ x, f (x + 2) - f x = 5) :
  ∀ x, f (x + 8) - f x = 20 := by
  sorry

end NUMINAMATH_CALUDE_line_increase_l2193_219361


namespace NUMINAMATH_CALUDE_tammy_climbing_l2193_219320

/-- Tammy's mountain climbing problem -/
theorem tammy_climbing (total_time total_distance : ℝ) 
  (speed_diff time_diff : ℝ) : 
  total_time = 14 →
  speed_diff = 0.5 →
  time_diff = 2 →
  total_distance = 52 →
  ∃ (speed1 time1 : ℝ),
    speed1 * time1 + (speed1 + speed_diff) * (time1 - time_diff) = total_distance ∧
    time1 + (time1 - time_diff) = total_time ∧
    speed1 + speed_diff = 4 :=
by sorry

end NUMINAMATH_CALUDE_tammy_climbing_l2193_219320


namespace NUMINAMATH_CALUDE_jake_watched_19_hours_on_friday_l2193_219340

/-- Represents the number of hours in a day -/
def hours_per_day : ℕ := 24

/-- Represents the total length of the show in hours -/
def show_length : ℕ := 52

/-- Calculates the hours Jake watched on Monday -/
def monday_hours : ℕ := hours_per_day / 2

/-- Represents the hours Jake watched on Tuesday -/
def tuesday_hours : ℕ := 4

/-- Calculates the hours Jake watched on Wednesday -/
def wednesday_hours : ℕ := hours_per_day / 4

/-- Calculates the total hours Jake watched from Monday to Wednesday -/
def mon_to_wed_total : ℕ := monday_hours + tuesday_hours + wednesday_hours

/-- Calculates the hours Jake watched on Thursday -/
def thursday_hours : ℕ := mon_to_wed_total / 2

/-- Calculates the total hours Jake watched from Monday to Thursday -/
def mon_to_thu_total : ℕ := mon_to_wed_total + thursday_hours

/-- Represents the hours Jake watched on Friday -/
def friday_hours : ℕ := show_length - mon_to_thu_total

theorem jake_watched_19_hours_on_friday : friday_hours = 19 := by
  sorry

end NUMINAMATH_CALUDE_jake_watched_19_hours_on_friday_l2193_219340


namespace NUMINAMATH_CALUDE_vector_dot_product_l2193_219351

theorem vector_dot_product (a b : ℝ × ℝ) :
  a + b = (2, -4) →
  3 • a - b = (-10, 16) →
  a • b = -29 := by
sorry

end NUMINAMATH_CALUDE_vector_dot_product_l2193_219351


namespace NUMINAMATH_CALUDE_triangle_tangent_identity_l2193_219382

theorem triangle_tangent_identity (α β γ : Real) (h : α + β + γ = PI) :
  Real.tan (α/2) * Real.tan (β/2) + Real.tan (β/2) * Real.tan (γ/2) + Real.tan (γ/2) * Real.tan (α/2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_triangle_tangent_identity_l2193_219382


namespace NUMINAMATH_CALUDE_parabolas_intersection_l2193_219309

/-- The x-coordinates of the intersection points of two parabolas -/
def intersection_x : Set ℝ := {-4, 5/2}

/-- The y-coordinates of the intersection points of two parabolas -/
def intersection_y : Set ℝ := {38, 31.5}

/-- First parabola equation -/
def parabola1 (x : ℝ) : ℝ := 4 * x^2 + 5 * x - 6

/-- Second parabola equation -/
def parabola2 (x : ℝ) : ℝ := 2 * x^2 + 14

theorem parabolas_intersection :
  ∀ x ∈ intersection_x, ∃ y ∈ intersection_y,
    parabola1 x = y ∧ parabola2 x = y :=
by sorry

end NUMINAMATH_CALUDE_parabolas_intersection_l2193_219309


namespace NUMINAMATH_CALUDE_infinitely_many_pairs_divisibility_l2193_219387

theorem infinitely_many_pairs_divisibility :
  ∀ n : ℕ, ∃ a b : ℤ, a > n ∧ (a * (a + 1)) ∣ (b^2 + 1) := by
  sorry

end NUMINAMATH_CALUDE_infinitely_many_pairs_divisibility_l2193_219387


namespace NUMINAMATH_CALUDE_division_problem_l2193_219304

theorem division_problem (dividend quotient remainder divisor : ℕ) : 
  dividend = 181 → 
  quotient = 9 → 
  remainder = 1 → 
  dividend = divisor * quotient + remainder →
  divisor = 20 := by sorry

end NUMINAMATH_CALUDE_division_problem_l2193_219304


namespace NUMINAMATH_CALUDE_unique_mod_residue_l2193_219344

theorem unique_mod_residue :
  ∃! n : ℤ, 0 ≤ n ∧ n ≤ 9 ∧ n ≡ -4321 [ZMOD 10] := by
  sorry

end NUMINAMATH_CALUDE_unique_mod_residue_l2193_219344


namespace NUMINAMATH_CALUDE_book_costs_and_plans_l2193_219325

/-- Represents the cost and quantity of books --/
structure BookOrder where
  lao_she : ℕ
  classics : ℕ
  total_cost : ℕ

/-- Represents a purchasing plan --/
structure PurchasePlan where
  lao_she : ℕ
  classics : ℕ

def is_valid_plan (p : PurchasePlan) (lao_she_cost classics_cost : ℕ) : Prop :=
  p.lao_she + p.classics = 20 ∧
  p.lao_she ≤ 2 * p.classics ∧
  p.lao_she * lao_she_cost + p.classics * classics_cost ≤ 1720

theorem book_costs_and_plans : ∃ (lao_she_cost classics_cost : ℕ) (plans : List PurchasePlan),
  let order1 : BookOrder := ⟨4, 2, 480⟩
  let order2 : BookOrder := ⟨2, 3, 520⟩
  (order1.lao_she * lao_she_cost + order1.classics * classics_cost = order1.total_cost) ∧
  (order2.lao_she * lao_she_cost + order2.classics * classics_cost = order2.total_cost) ∧
  (lao_she_cost = 50) ∧
  (classics_cost = 140) ∧
  (plans.length = 2) ∧
  (∀ p ∈ plans, is_valid_plan p lao_she_cost classics_cost) ∧
  (∀ p : PurchasePlan, is_valid_plan p lao_she_cost classics_cost → p ∈ plans) :=
by sorry


end NUMINAMATH_CALUDE_book_costs_and_plans_l2193_219325


namespace NUMINAMATH_CALUDE_exists_empty_subsquare_l2193_219334

/-- Represents a point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a square in a 2D plane -/
structure Square where
  bottomLeft : Point
  sideLength : ℝ

/-- A function to check if a point is inside a square -/
def isPointInSquare (p : Point) (s : Square) : Prop :=
  s.bottomLeft.x ≤ p.x ∧ p.x < s.bottomLeft.x + s.sideLength ∧
  s.bottomLeft.y ≤ p.y ∧ p.y < s.bottomLeft.y + s.sideLength

/-- The main theorem -/
theorem exists_empty_subsquare 
  (bigSquare : Square) 
  (points : Finset Point) 
  (h1 : bigSquare.sideLength = 4) 
  (h2 : points.card = 15) : 
  ∃ (smallSquare : Square), 
    smallSquare.sideLength = 1 ∧ 
    (∀ (p : Point), p ∈ points → ¬ isPointInSquare p smallSquare) :=
sorry

end NUMINAMATH_CALUDE_exists_empty_subsquare_l2193_219334


namespace NUMINAMATH_CALUDE_total_cost_construction_materials_l2193_219368

def cement_bags : ℕ := 500
def cement_price_per_bag : ℕ := 10
def sand_lorries : ℕ := 20
def sand_tons_per_lorry : ℕ := 10
def sand_price_per_ton : ℕ := 40

theorem total_cost_construction_materials : 
  cement_bags * cement_price_per_bag + 
  sand_lorries * sand_tons_per_lorry * sand_price_per_ton = 13000 :=
by sorry

end NUMINAMATH_CALUDE_total_cost_construction_materials_l2193_219368


namespace NUMINAMATH_CALUDE_banner_nail_distance_l2193_219313

theorem banner_nail_distance (banner_length : ℝ) (num_nails : ℕ) (end_distance : ℝ) :
  banner_length = 20 →
  num_nails = 7 →
  end_distance = 1 →
  (banner_length - 2 * end_distance) / (num_nails - 1 : ℝ) = 3 :=
by sorry

end NUMINAMATH_CALUDE_banner_nail_distance_l2193_219313


namespace NUMINAMATH_CALUDE_range_of_b_l2193_219314

theorem range_of_b (a b : ℝ) (h1 : 0 ≤ a + b ∧ a + b < 1) (h2 : 2 ≤ a - b ∧ a - b < 3) :
  -3/2 < b ∧ b < -1/2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_b_l2193_219314


namespace NUMINAMATH_CALUDE_range_of_m_l2193_219357

-- Define the solution set A
def A (m : ℝ) : Set ℝ := {x : ℝ | |x^2 - 4*x + m| ≤ x + 4}

-- State the theorem
theorem range_of_m :
  ∀ m : ℝ, (0 ∈ A m) ∧ (2 ∉ A m) ↔ m ∈ Set.Icc (-4 : ℝ) (-2 : ℝ) := by sorry

end NUMINAMATH_CALUDE_range_of_m_l2193_219357


namespace NUMINAMATH_CALUDE_length_PS_is_sqrt_32_5_l2193_219394

-- Define the quadrilateral PQRS
structure Quadrilateral :=
  (P Q R S T : ℝ × ℝ)

-- Define the conditions
def is_valid_quadrilateral (quad : Quadrilateral) : Prop :=
  let d_PT := Real.sqrt ((quad.P.1 - quad.T.1)^2 + (quad.P.2 - quad.T.2)^2)
  let d_TR := Real.sqrt ((quad.T.1 - quad.R.1)^2 + (quad.T.2 - quad.R.2)^2)
  let d_QT := Real.sqrt ((quad.Q.1 - quad.T.1)^2 + (quad.Q.2 - quad.T.2)^2)
  let d_TS := Real.sqrt ((quad.T.1 - quad.S.1)^2 + (quad.T.2 - quad.S.2)^2)
  let d_PQ := Real.sqrt ((quad.P.1 - quad.Q.1)^2 + (quad.P.2 - quad.Q.2)^2)
  d_PT = 5 ∧ d_TR = 4 ∧ d_QT = 7 ∧ d_TS = 2 ∧ d_PQ = 7

-- Theorem statement
theorem length_PS_is_sqrt_32_5 (quad : Quadrilateral) 
  (h : is_valid_quadrilateral quad) : 
  Real.sqrt ((quad.P.1 - quad.S.1)^2 + (quad.P.2 - quad.S.2)^2) = Real.sqrt 32.5 := by
  sorry

end NUMINAMATH_CALUDE_length_PS_is_sqrt_32_5_l2193_219394


namespace NUMINAMATH_CALUDE_cubic_inequality_implies_a_range_l2193_219352

theorem cubic_inequality_implies_a_range :
  ∀ a : ℝ, (∀ x : ℝ, x ∈ Set.Icc (-2) 1 → a * x^3 - x^2 + 4*x + 3 ≥ 0) →
  a ∈ Set.Icc (-6) (-2) := by
  sorry

end NUMINAMATH_CALUDE_cubic_inequality_implies_a_range_l2193_219352


namespace NUMINAMATH_CALUDE_zeros_of_h_l2193_219376

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - a * x

def g (a : ℝ) (x : ℝ) : ℝ := (a + 1) * x^2 - 1

noncomputable def h (a : ℝ) (x : ℝ) : ℝ := 2 * f a x - g a x

theorem zeros_of_h (a : ℝ) (x₁ x₂ : ℝ) :
  (∃ (x₁ x₂ : ℝ), h a x₁ = 0 ∧ h a x₂ = 0 ∧ x₁ ≠ x₂) →
  -1 < a ∧ a < 0 ∧ x₁ + x₂ > 2 / (a + 1) := by
  sorry

end NUMINAMATH_CALUDE_zeros_of_h_l2193_219376


namespace NUMINAMATH_CALUDE_tv_screen_coverage_l2193_219317

theorem tv_screen_coverage (w1 h1 w2 h2 : ℚ) : 
  w1 / h1 = 16 / 9 →
  w2 / h2 = 4 / 3 →
  (h2 - h1 * (w2 / w1)) / h2 = 1 / 4 := by
sorry

end NUMINAMATH_CALUDE_tv_screen_coverage_l2193_219317


namespace NUMINAMATH_CALUDE_stewart_farm_sheep_count_l2193_219318

theorem stewart_farm_sheep_count :
  ∀ (sheep horses : ℕ),
  sheep / horses = 3 / 7 →
  horses * 230 = 12880 →
  sheep = 24 :=
by
  sorry

end NUMINAMATH_CALUDE_stewart_farm_sheep_count_l2193_219318


namespace NUMINAMATH_CALUDE_three_consecutive_free_throws_l2193_219373

/-- The probability of scoring a single free throw -/
def free_throw_probability : ℝ := 0.7

/-- The number of consecutive free throws -/
def num_throws : ℕ := 3

/-- The probability of scoring in three consecutive free throws -/
def three_consecutive_probability : ℝ := free_throw_probability ^ num_throws

theorem three_consecutive_free_throws :
  three_consecutive_probability = 0.343 := by
  sorry

end NUMINAMATH_CALUDE_three_consecutive_free_throws_l2193_219373


namespace NUMINAMATH_CALUDE_correct_meiosis_sequence_l2193_219326

-- Define the stages of meiosis
inductive MeiosisStage
  | Replication
  | Synapsis
  | Separation
  | Division

-- Define a sequence type
def Sequence := List MeiosisStage

-- Define the four given sequences
def sequenceA : Sequence := [MeiosisStage.Replication, MeiosisStage.Synapsis, MeiosisStage.Separation, MeiosisStage.Division]
def sequenceB : Sequence := [MeiosisStage.Synapsis, MeiosisStage.Replication, MeiosisStage.Separation, MeiosisStage.Division]
def sequenceC : Sequence := [MeiosisStage.Synapsis, MeiosisStage.Replication, MeiosisStage.Division, MeiosisStage.Separation]
def sequenceD : Sequence := [MeiosisStage.Replication, MeiosisStage.Separation, MeiosisStage.Synapsis, MeiosisStage.Division]

-- Define a function to check if a sequence is correct
def isCorrectSequence (s : Sequence) : Prop :=
  s = sequenceA

-- Theorem stating that sequenceA is the correct sequence
theorem correct_meiosis_sequence :
  isCorrectSequence sequenceA ∧
  ¬isCorrectSequence sequenceB ∧
  ¬isCorrectSequence sequenceC ∧
  ¬isCorrectSequence sequenceD :=
sorry

end NUMINAMATH_CALUDE_correct_meiosis_sequence_l2193_219326


namespace NUMINAMATH_CALUDE_fraction_equals_zero_l2193_219327

theorem fraction_equals_zero (x : ℝ) : x / (x^2 - 1) = 0 → x = 0 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equals_zero_l2193_219327


namespace NUMINAMATH_CALUDE_unique_solution_when_k_zero_no_unique_solution_when_k_nonzero_k_zero_only_unique_solution_l2193_219323

/-- The equation has exactly one solution when k = 0 -/
theorem unique_solution_when_k_zero : ∃! x : ℝ, (x + 3) / (0 * x - 2) = x :=
sorry

/-- For any k ≠ 0, the equation has either no solution or more than one solution -/
theorem no_unique_solution_when_k_nonzero (k : ℝ) (hk : k ≠ 0) :
  ¬(∃! x : ℝ, (x + 3) / (k * x - 2) = x) :=
sorry

/-- k = 0 is the only value for which the equation has exactly one solution -/
theorem k_zero_only_unique_solution :
  ∀ k : ℝ, (∃! x : ℝ, (x + 3) / (k * x - 2) = x) ↔ k = 0 :=
sorry

end NUMINAMATH_CALUDE_unique_solution_when_k_zero_no_unique_solution_when_k_nonzero_k_zero_only_unique_solution_l2193_219323


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l2193_219328

theorem partial_fraction_decomposition (M₁ M₂ : ℚ) :
  (∀ x : ℚ, x ≠ 2 → x ≠ 3 → (45 * x - 82) / (x^2 - 5*x + 6) = M₁ / (x - 2) + M₂ / (x - 3)) →
  M₁ * M₂ = -424 := by
sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l2193_219328


namespace NUMINAMATH_CALUDE_fraction_addition_l2193_219362

theorem fraction_addition : (7 : ℚ) / 8 + (9 : ℚ) / 12 = (13 : ℚ) / 8 := by
  sorry

end NUMINAMATH_CALUDE_fraction_addition_l2193_219362


namespace NUMINAMATH_CALUDE_quadratic_roots_relation_l2193_219335

theorem quadratic_roots_relation (m n p : ℝ) (hm : m ≠ 0) (hn : n ≠ 0) (hp : p ≠ 0) :
  (∀ x, x^2 + m*x + n = 0 ↔ (x/2)^2 + p*(x/2) + m = 0) →
  n / p = 8 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_relation_l2193_219335


namespace NUMINAMATH_CALUDE_parabola_equation_from_conditions_l2193_219380

/-- A parabola is defined by its focus-directrix distance and a point it passes through. -/
structure Parabola where
  focus_directrix_distance : ℝ
  point : ℝ × ℝ

/-- The equation of a parabola in the form y^2 = ax, where a is a real number. -/
def parabola_equation (a : ℝ) : ℝ → ℝ → Prop :=
  fun x y => y^2 = a * x

theorem parabola_equation_from_conditions (p : Parabola) 
  (h1 : p.focus_directrix_distance = 2)
  (h2 : p.point = (1, 2)) :
  parabola_equation 4 = fun x y => y^2 = 4 * x :=
by sorry

end NUMINAMATH_CALUDE_parabola_equation_from_conditions_l2193_219380


namespace NUMINAMATH_CALUDE_remaining_flour_l2193_219374

def flour_needed (total_required : ℕ) (already_added : ℕ) : ℕ :=
  total_required - already_added

theorem remaining_flour :
  flour_needed 9 2 = 7 :=
by sorry

end NUMINAMATH_CALUDE_remaining_flour_l2193_219374


namespace NUMINAMATH_CALUDE_sum_of_squares_l2193_219341

theorem sum_of_squares (a b : ℝ) : (a^2 + b^2) * (a^2 + b^2 + 4) = 12 → a^2 + b^2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_l2193_219341


namespace NUMINAMATH_CALUDE_bluejay_female_fraction_l2193_219333

theorem bluejay_female_fraction (total_birds : ℝ) (total_birds_pos : 0 < total_birds) : 
  let robins := (2/5) * total_birds
  let bluejays := (3/5) * total_birds
  let female_robins := (1/3) * robins
  let male_birds := (7/15) * total_birds
  let female_bluejays := ((8/15) * total_birds) - female_robins
  (female_bluejays / bluejays) = (2/3) :=
by sorry

end NUMINAMATH_CALUDE_bluejay_female_fraction_l2193_219333


namespace NUMINAMATH_CALUDE_exists_valid_sequence_l2193_219375

def is_valid_sequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a n < a (n + 1)) ∧ 
  (∀ k, k ≥ 1 → a (2*k + 1) = (a (2*k) + a (2*k + 2)) / 2) ∧
  (∀ k, k ≥ 1 → a (2*k) = Real.sqrt (a (2*k - 1) * a (2*k + 1)))

theorem exists_valid_sequence : ∃ a : ℕ → ℝ, is_valid_sequence a :=
sorry

end NUMINAMATH_CALUDE_exists_valid_sequence_l2193_219375


namespace NUMINAMATH_CALUDE_unique_n_l2193_219379

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def digit_sum (n : ℕ) : ℕ := 
  (n / 100) + ((n / 10) % 10) + (n % 10)

def digit_product (n : ℕ) : ℕ := 
  (n / 100) * ((n / 10) % 10) * (n % 10)

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

theorem unique_n : 
  ∃! n : ℕ, 
    is_three_digit n ∧ 
    is_perfect_square n ∧ 
    is_two_digit (digit_sum n) ∧
    (∀ m : ℕ, is_three_digit m ∧ is_perfect_square m ∧ digit_product m = digit_product n → m = n) ∧
    (∃ m : ℕ, is_three_digit m ∧ is_perfect_square m ∧ m ≠ n ∧ digit_sum m = digit_sum n) ∧
    (∀ m : ℕ, is_three_digit m ∧ is_perfect_square m ∧ digit_sum m = digit_sum n →
      (∀ k : ℕ, is_three_digit k ∧ is_perfect_square k ∧ digit_product k = digit_product m → k = m)) ∧
    n = 841 :=
by sorry

end NUMINAMATH_CALUDE_unique_n_l2193_219379


namespace NUMINAMATH_CALUDE_painter_workdays_l2193_219354

theorem painter_workdays (initial_painters : ℕ) (initial_days : ℚ) (new_painters : ℕ) :
  initial_painters = 5 →
  initial_days = 4/5 →
  new_painters = 4 →
  (initial_painters : ℚ) * initial_days = new_painters * 1 :=
by sorry

end NUMINAMATH_CALUDE_painter_workdays_l2193_219354


namespace NUMINAMATH_CALUDE_line_parallel_to_plane_l2193_219370

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel relation for lines and planes
variable (parallel_line : Line → Line → Prop)
variable (parallel_line_plane : Line → Plane → Prop)

-- Define the non-parallel relation for lines and planes
variable (not_parallel_line_plane : Line → Plane → Prop)

-- State the theorem
theorem line_parallel_to_plane 
  (m n : Line) (α : Plane) :
  parallel_line m n → 
  not_parallel_line_plane n α → 
  not_parallel_line_plane m α → 
  parallel_line_plane m α :=
sorry

end NUMINAMATH_CALUDE_line_parallel_to_plane_l2193_219370


namespace NUMINAMATH_CALUDE_fraction_problem_l2193_219305

theorem fraction_problem :
  ∃ (x : ℝ) (a b : ℕ),
    x > 0 ∧
    x^2 = 25 ∧
    2*x = (a / b : ℝ)*x + 9 →
    a = 1 ∧ b = 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_problem_l2193_219305


namespace NUMINAMATH_CALUDE_rope_length_proof_l2193_219321

theorem rope_length_proof (shorter_piece longer_piece original_length : ℝ) : 
  shorter_piece = 20 →
  longer_piece = 2 * shorter_piece →
  original_length = shorter_piece + longer_piece →
  original_length = 60 := by
sorry

end NUMINAMATH_CALUDE_rope_length_proof_l2193_219321


namespace NUMINAMATH_CALUDE_abc_mod_five_l2193_219360

theorem abc_mod_five (a b c : ℕ) : 
  a < 5 → b < 5 → c < 5 →
  (a + 2*b + 3*c) % 5 = 3 →
  (2*a + 3*b + c) % 5 = 2 →
  (3*a + b + 2*c) % 5 = 1 →
  (a*b*c) % 5 = 3 := by
  sorry

#check abc_mod_five

end NUMINAMATH_CALUDE_abc_mod_five_l2193_219360


namespace NUMINAMATH_CALUDE_cellCount_after_8_days_l2193_219363

/-- The number of cells in a colony after a given number of days, 
    with specific growth and toxin conditions. -/
def cellCount (initialCells : ℕ) (days : ℕ) : ℕ :=
  let growthPeriods := days / 2
  let afterGrowth := initialCells * 3^growthPeriods
  if days ≥ 6 then
    (afterGrowth / 2 + if afterGrowth % 2 = 0 then 0 else 1) * 3^((days - 6) / 2)
  else
    afterGrowth

theorem cellCount_after_8_days : 
  cellCount 5 8 = 201 := by sorry

end NUMINAMATH_CALUDE_cellCount_after_8_days_l2193_219363


namespace NUMINAMATH_CALUDE_rapid_advance_min_cost_l2193_219372

/-- Represents a ride model with its capacity and price -/
structure RideModel where
  capacity : ℕ
  price : ℕ

/-- The minimum amount needed to spend on tickets for a group -/
def minTicketCost (model1 model2 : RideModel) (groupSize : ℕ) : ℕ :=
  sorry

theorem rapid_advance_min_cost :
  let model1 : RideModel := { capacity := 7, price := 65 }
  let model2 : RideModel := { capacity := 5, price := 50 }
  let groupSize : ℕ := 73
  minTicketCost model1 model2 groupSize = 685 := by sorry

end NUMINAMATH_CALUDE_rapid_advance_min_cost_l2193_219372


namespace NUMINAMATH_CALUDE_v_2002_equals_2_l2193_219355

def g : ℕ → ℕ
  | 1 => 5
  | 2 => 3
  | 3 => 1
  | 4 => 2
  | 5 => 4
  | _ => 0  -- Default case for completeness

def v : ℕ → ℕ
  | 0 => 5
  | n + 1 => g (v n)

theorem v_2002_equals_2 : v 2002 = 2 := by
  sorry

end NUMINAMATH_CALUDE_v_2002_equals_2_l2193_219355


namespace NUMINAMATH_CALUDE_remainder_three_to_27_mod_13_l2193_219395

theorem remainder_three_to_27_mod_13 : 3^27 % 13 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_three_to_27_mod_13_l2193_219395


namespace NUMINAMATH_CALUDE_perpendicular_lines_k_value_l2193_219343

theorem perpendicular_lines_k_value (k : ℝ) :
  (((k - 3) * 2 * (k - 3) + (5 - k) * (-2) = 0) →
  (k = 1 ∨ k = 4)) ∧
  ((k = 1 ∨ k = 4) →
  ((k - 3) * 2 * (k - 3) + (5 - k) * (-2) = 0)) := by sorry

end NUMINAMATH_CALUDE_perpendicular_lines_k_value_l2193_219343


namespace NUMINAMATH_CALUDE_smallest_a_for_nonprime_cube_sum_l2193_219369

theorem smallest_a_for_nonprime_cube_sum :
  ∃ (a : ℕ), a > 0 ∧ (∀ (x : ℤ), ¬ Prime (x^3 + a^3)) ∧
  (∀ (b : ℕ), b > 0 ∧ b < a → ∃ (y : ℤ), Prime (y^3 + b^3)) :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_smallest_a_for_nonprime_cube_sum_l2193_219369


namespace NUMINAMATH_CALUDE_cricketer_matches_l2193_219365

theorem cricketer_matches (total_average : ℝ) (first_8_average : ℝ) (last_4_average : ℝ)
  (h1 : total_average = 48)
  (h2 : first_8_average = 40)
  (h3 : last_4_average = 64) :
  ∃ (n : ℕ), n * total_average = 8 * first_8_average + 4 * last_4_average ∧ n = 12 := by
  sorry

end NUMINAMATH_CALUDE_cricketer_matches_l2193_219365


namespace NUMINAMATH_CALUDE_ratio_x_to_y_l2193_219311

theorem ratio_x_to_y (x y : ℝ) (h : (12*x - 5*y) / (15*x - 3*y) = 4/7) : x/y = 23/24 := by
  sorry

end NUMINAMATH_CALUDE_ratio_x_to_y_l2193_219311


namespace NUMINAMATH_CALUDE_expression_simplification_l2193_219348

theorem expression_simplification (x y : ℝ) : 
  -(3*x*y - 2*x^2) - 2*(3*x^2 - x*y) = -4*x^2 - x*y := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2193_219348


namespace NUMINAMATH_CALUDE_seventh_twentyninth_150th_digit_l2193_219331

/-- The decimal expansion of 7/29 has a repeating block of length 28 -/
def decimal_period : ℕ := 28

/-- The repeating block in the decimal expansion of 7/29 -/
def repeating_block : List ℕ := [2, 4, 1, 3, 7, 9, 3, 1, 0, 3, 4, 4, 8, 2, 7, 5, 8, 6, 2, 0, 6, 8, 9, 6, 5, 5, 1, 7, 2]

/-- The 150th digit after the decimal point in the decimal expansion of 7/29 -/
def digit_150 : ℕ := repeating_block[(150 - 1) % decimal_period]

theorem seventh_twentyninth_150th_digit :
  digit_150 = 8 := by sorry

end NUMINAMATH_CALUDE_seventh_twentyninth_150th_digit_l2193_219331


namespace NUMINAMATH_CALUDE_shopkeeper_gain_percentage_l2193_219384

/-- Calculates the gain percentage of a shopkeeper using false weights -/
theorem shopkeeper_gain_percentage (true_weight false_weight : ℕ) : 
  true_weight = 1000 → 
  false_weight = 960 → 
  (true_weight - false_weight) * 100 / true_weight = 4 := by
  sorry

#check shopkeeper_gain_percentage

end NUMINAMATH_CALUDE_shopkeeper_gain_percentage_l2193_219384


namespace NUMINAMATH_CALUDE_object_properties_l2193_219349

-- Define the possible colors
inductive Color
| Red
| Blue
| Green

-- Define the shape property
structure Object where
  color : Color
  isRound : Bool

-- Define the conditions
axiom condition1 (obj : Object) : obj.isRound → (obj.color = Color.Red ∨ obj.color = Color.Blue)
axiom condition2 (obj : Object) : ¬obj.isRound → (obj.color ≠ Color.Red ∧ obj.color ≠ Color.Green)
axiom condition3 (obj : Object) : (obj.color = Color.Blue ∨ obj.color = Color.Green) → obj.isRound

-- Theorem to prove
theorem object_properties (obj : Object) : 
  obj.isRound ∧ (obj.color = Color.Red ∨ obj.color = Color.Blue) :=
by sorry

end NUMINAMATH_CALUDE_object_properties_l2193_219349


namespace NUMINAMATH_CALUDE_product_divisible_by_all_product_prime_factorization_divisibility_condition_l2193_219383

theorem product_divisible_by_all : ∀ n : ℕ, 1 ≤ n ∧ n ≤ 10 → (45 * 56) % n = 0 := by sorry

theorem product_prime_factorization : 
  ∃ (k : ℕ), 45 * 56 = 2^3 * 3^2 * 5 * 7 * k ∧ k ≥ 1 := by sorry

theorem divisibility_condition (a b c d : ℕ) : 
  ∀ n : ℕ, 1 ≤ n ∧ n ≤ 10 → (2^a * 3^b * 5^c * 7^d) % n = 0 → a ≥ 3 ∧ b ≥ 2 ∧ c ≥ 1 ∧ d ≥ 1 := by sorry

end NUMINAMATH_CALUDE_product_divisible_by_all_product_prime_factorization_divisibility_condition_l2193_219383


namespace NUMINAMATH_CALUDE_age_difference_l2193_219393

/-- Proves that z is 1.2 decades younger than x given the condition on ages -/
theorem age_difference (x y z : ℝ) (h : x + y = y + z + 12) : (x - z) / 10 = 1.2 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l2193_219393


namespace NUMINAMATH_CALUDE_decimal_division_remainder_l2193_219346

theorem decimal_division_remainder (n : ℕ) (N : ℕ) : 
  (N % (2^n) = (N % 10^n) % (2^n)) ∧ (N % (5^n) = (N % 10^n) % (5^n)) := by sorry

end NUMINAMATH_CALUDE_decimal_division_remainder_l2193_219346


namespace NUMINAMATH_CALUDE_range_of_a_over_b_l2193_219312

def ellipse (x y : ℝ) : Prop := x^2 / 6 + y^2 / 2 = 1

def line (a b x y : ℝ) : Prop := a * x + b * y = 2

theorem range_of_a_over_b (a b : ℝ) :
  a^2 + b^2 = 1 →
  b ≠ 0 →
  (∃ x y : ℝ, ellipse x y ∧ line a b x y) →
  (a / b < -1 ∨ a / b = -1 ∨ a / b = 1 ∨ a / b > 1) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_over_b_l2193_219312


namespace NUMINAMATH_CALUDE_cookies_on_floor_l2193_219315

/-- Calculates the number of cookies thrown on the floor given the initial and additional cookies baked by Alice and Bob, and the final number of edible cookies. -/
theorem cookies_on_floor (alice_initial bob_initial alice_additional bob_additional final_edible : ℕ) :
  alice_initial = 74 →
  bob_initial = 7 →
  alice_additional = 5 →
  bob_additional = 36 →
  final_edible = 93 →
  (alice_initial + bob_initial + alice_additional + bob_additional) - final_edible = 29 := by
  sorry

#check cookies_on_floor

end NUMINAMATH_CALUDE_cookies_on_floor_l2193_219315


namespace NUMINAMATH_CALUDE_largest_divisor_of_product_l2193_219388

theorem largest_divisor_of_product (n : ℕ) (h : Even n) (h' : n > 0) :
  (∃ (k : ℕ), (n + 1) * (n + 3) * (n + 5) * (n + 7) * (n + 9) * (n + 11) * (n + 13) = 105 * k) ∧
  (∀ (m : ℕ), m > 105 → ¬(∀ (n : ℕ), Even n → n > 0 →
    ∃ (k : ℕ), (n + 1) * (n + 3) * (n + 5) * (n + 7) * (n + 9) * (n + 11) * (n + 13) = m * k)) :=
by sorry

end NUMINAMATH_CALUDE_largest_divisor_of_product_l2193_219388


namespace NUMINAMATH_CALUDE_simplify_and_sum_fraction_l2193_219359

theorem simplify_and_sum_fraction : ∃ (a b : ℕ), 
  (75 : ℚ) / 100 = (a : ℚ) / b ∧ 
  (∀ (c d : ℕ), (75 : ℚ) / 100 = (c : ℚ) / d → a ≤ c ∧ b ≤ d) ∧
  a + b = 7 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_sum_fraction_l2193_219359


namespace NUMINAMATH_CALUDE_lcm_of_180_and_504_l2193_219391

theorem lcm_of_180_and_504 : Nat.lcm 180 504 = 2520 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_180_and_504_l2193_219391


namespace NUMINAMATH_CALUDE_zongzi_pricing_and_max_purchase_l2193_219300

/-- Represents the zongzi types -/
inductive ZongziType
| A
| B

/-- Represents the price and quantity information for zongzi -/
structure ZongziInfo where
  type : ZongziType
  amount_spent : ℝ
  quantity : ℕ

/-- Theorem for zongzi pricing and maximum purchase -/
theorem zongzi_pricing_and_max_purchase 
  (info_A : ZongziInfo) 
  (info_B : ZongziInfo) 
  (total_zongzi : ℕ) 
  (max_total_amount : ℝ) :
  info_A.type = ZongziType.A →
  info_B.type = ZongziType.B →
  info_A.amount_spent = 1200 →
  info_B.amount_spent = 800 →
  info_B.quantity = info_A.quantity + 50 →
  info_A.amount_spent / info_A.quantity = 2 * (info_B.amount_spent / info_B.quantity) →
  total_zongzi = 200 →
  max_total_amount = 1150 →
  ∃ (unit_price_A unit_price_B : ℝ) (max_quantity_A : ℕ),
    unit_price_A = 8 ∧
    unit_price_B = 4 ∧
    max_quantity_A = 87 ∧
    unit_price_A * max_quantity_A + unit_price_B * (total_zongzi - max_quantity_A) ≤ max_total_amount ∧
    ∀ (quantity_A : ℕ), 
      quantity_A > max_quantity_A →
      unit_price_A * quantity_A + unit_price_B * (total_zongzi - quantity_A) > max_total_amount :=
by sorry

end NUMINAMATH_CALUDE_zongzi_pricing_and_max_purchase_l2193_219300


namespace NUMINAMATH_CALUDE_aiyannas_cookie_count_l2193_219381

/-- The number of cookies Alyssa has -/
def alyssas_cookies : ℕ := 129

/-- The number of additional cookies Aiyanna has compared to Alyssa -/
def additional_cookies : ℕ := 11

/-- The number of cookies Aiyanna has -/
def aiyannas_cookies : ℕ := alyssas_cookies + additional_cookies

theorem aiyannas_cookie_count : aiyannas_cookies = 140 := by
  sorry

end NUMINAMATH_CALUDE_aiyannas_cookie_count_l2193_219381


namespace NUMINAMATH_CALUDE_symmetry_of_transformed_functions_l2193_219308

/-- Given a function f, prove that the graphs of f(x-1) and f(-x+1) are symmetric with respect to the line x = 1 -/
theorem symmetry_of_transformed_functions (f : ℝ → ℝ) : 
  ∀ (x y : ℝ), f (x - 1) = y ↔ f (-(x - 1)) = y :=
by sorry

end NUMINAMATH_CALUDE_symmetry_of_transformed_functions_l2193_219308


namespace NUMINAMATH_CALUDE_pillar_length_calculation_l2193_219337

/-- Given a formula for L and specific values for T, H, and K, prove that L equals 100. -/
theorem pillar_length_calculation (T H K L : ℝ) : 
  T = 2 * Real.sqrt 5 →
  H = 10 →
  K = 2 →
  L = (50 * T^4) / (H^2 * K) →
  L = 100 := by
sorry

end NUMINAMATH_CALUDE_pillar_length_calculation_l2193_219337


namespace NUMINAMATH_CALUDE_hyperbola_s_squared_l2193_219367

/-- A hyperbola passing through specific points -/
structure Hyperbola where
  /-- The hyperbola is centered at the origin -/
  center : (ℝ × ℝ) := (0, 0)
  /-- The hyperbola passes through (5, -6) -/
  point1 : (ℝ × ℝ) := (5, -6)
  /-- The hyperbola passes through (3, 0) -/
  point2 : (ℝ × ℝ) := (3, 0)
  /-- The hyperbola passes through (s, -3) for some real s -/
  point3 : (ℝ × ℝ)
  /-- The third point has y-coordinate -3 -/
  h_point3_y : point3.2 = -3

/-- The theorem stating that s² = 12 for the given hyperbola -/
theorem hyperbola_s_squared (h : Hyperbola) : h.point3.1 ^ 2 = 12 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_s_squared_l2193_219367


namespace NUMINAMATH_CALUDE_y1_greater_than_y2_l2193_219302

-- Define the parabola function
def f (x : ℝ) : ℝ := -(x + 1)^2 + 3

-- Define the theorem
theorem y1_greater_than_y2 :
  ∀ y₁ y₂ : ℝ, f 1 = y₁ → f 2 = y₂ → y₁ > y₂ := by
  sorry

end NUMINAMATH_CALUDE_y1_greater_than_y2_l2193_219302


namespace NUMINAMATH_CALUDE_other_items_sales_percentage_l2193_219324

theorem other_items_sales_percentage 
  (total_sales_percentage : ℝ)
  (notebooks_sales_percentage : ℝ)
  (markers_sales_percentage : ℝ)
  (h1 : total_sales_percentage = 100)
  (h2 : notebooks_sales_percentage = 42)
  (h3 : markers_sales_percentage = 21) :
  total_sales_percentage - (notebooks_sales_percentage + markers_sales_percentage) = 37 := by
  sorry

end NUMINAMATH_CALUDE_other_items_sales_percentage_l2193_219324


namespace NUMINAMATH_CALUDE_x_squared_eq_one_is_quadratic_l2193_219398

/-- Definition of a quadratic equation -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The equation x² = 1 -/
def f (x : ℝ) : ℝ := x^2 - 1

/-- Theorem: x² = 1 is a quadratic equation -/
theorem x_squared_eq_one_is_quadratic : is_quadratic_equation f := by
  sorry


end NUMINAMATH_CALUDE_x_squared_eq_one_is_quadratic_l2193_219398


namespace NUMINAMATH_CALUDE_expression_value_at_1580_l2193_219347

theorem expression_value_at_1580 : 
  let a : ℝ := 1580
  let expr := 2*a - (((2*a - 3)/(a + 1)) - ((a + 1)/(2 - 2*a)) - ((a^2 + 3)/(2*a^(2-2)))) * ((a^3 + 1)/(a^2 - a)) + 2/a
  expr = 2 := by sorry

end NUMINAMATH_CALUDE_expression_value_at_1580_l2193_219347


namespace NUMINAMATH_CALUDE_quadratic_completion_of_square_l2193_219399

theorem quadratic_completion_of_square :
  ∀ x : ℝ, x^2 - 6*x + 4 = 0 ↔ (x - 3)^2 = 5 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_completion_of_square_l2193_219399


namespace NUMINAMATH_CALUDE_tangent_slope_angle_at_point_one_three_l2193_219303

-- Define the curve
def f (x : ℝ) : ℝ := x^3 - 2*x + 4

-- Define the derivative of the curve
def f' (x : ℝ) : ℝ := 3*x^2 - 2

-- Theorem statement
theorem tangent_slope_angle_at_point_one_three :
  let x₀ : ℝ := 1
  let y₀ : ℝ := f x₀
  let k : ℝ := f' x₀
  Real.arctan k = π/4 :=
by sorry

end NUMINAMATH_CALUDE_tangent_slope_angle_at_point_one_three_l2193_219303


namespace NUMINAMATH_CALUDE_customers_who_left_l2193_219306

theorem customers_who_left (initial : ℕ) (new : ℕ) (final : ℕ) : 
  initial = 13 → new = 4 → final = 9 → initial - (initial - final + new) = 8 := by
  sorry

end NUMINAMATH_CALUDE_customers_who_left_l2193_219306


namespace NUMINAMATH_CALUDE_tour_group_size_l2193_219396

theorem tour_group_size (initial_groups : ℕ) (initial_avg : ℕ) (remaining_groups : ℕ) (remaining_avg : ℕ) :
  initial_groups = 10 →
  initial_avg = 9 →
  remaining_groups = 9 →
  remaining_avg = 8 →
  (initial_groups * initial_avg) - (remaining_groups * remaining_avg) = 18 :=
by
  sorry

end NUMINAMATH_CALUDE_tour_group_size_l2193_219396


namespace NUMINAMATH_CALUDE_set_equality_l2193_219371

def M : Set ℝ := {x | x^2 - 2012*x - 2013 > 0}
def N (a b : ℝ) : Set ℝ := {x | x^2 + a*x + b ≤ 0}

theorem set_equality (a b : ℝ) : 
  M ∪ N a b = Set.univ ∧ 
  M ∩ N a b = Set.Ioo 2013 2014 →
  a = -2013 ∧ b = -2014 := by
sorry

end NUMINAMATH_CALUDE_set_equality_l2193_219371


namespace NUMINAMATH_CALUDE_middle_number_proof_l2193_219389

theorem middle_number_proof (x y z : ℕ) (hxy : x < y) (hyz : y < z)
  (sum_xy : x + y = 22) (sum_xz : x + z = 29) (sum_yz : y + z = 37) : y = 15 := by
  sorry

end NUMINAMATH_CALUDE_middle_number_proof_l2193_219389


namespace NUMINAMATH_CALUDE_almond_salami_cheese_cost_l2193_219310

/-- The cost of Sean's Sunday purchases -/
def sean_sunday_cost (almond_croissant : ℝ) (salami_cheese_croissant : ℝ) : ℝ :=
  almond_croissant + salami_cheese_croissant + 3 + 4 + 2 * 2.5

/-- Theorem stating the combined cost of almond and salami & cheese croissants -/
theorem almond_salami_cheese_cost :
  ∃ (almond_croissant salami_cheese_croissant : ℝ),
    sean_sunday_cost almond_croissant salami_cheese_croissant = 21 ∧
    almond_croissant + salami_cheese_croissant = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_almond_salami_cheese_cost_l2193_219310


namespace NUMINAMATH_CALUDE_investors_in_both_l2193_219377

theorem investors_in_both (total : ℕ) (equities : ℕ) (both : ℕ)
  (h_total : total = 100)
  (h_equities : equities = 80)
  (h_both : both = 40)
  (h_invest : ∀ i, i ∈ Finset.range total → 
    (i ∈ Finset.range equities ∨ i ∈ Finset.range (total - equities + both)))
  : both = 40 := by
  sorry

end NUMINAMATH_CALUDE_investors_in_both_l2193_219377


namespace NUMINAMATH_CALUDE_fourth_root_equivalence_l2193_219350

-- Define y as a positive real number
variable (y : ℝ) (hy : y > 0)

-- State the theorem
theorem fourth_root_equivalence : (y^2 * y^(1/2))^(1/4) = y^(5/8) := by sorry

end NUMINAMATH_CALUDE_fourth_root_equivalence_l2193_219350


namespace NUMINAMATH_CALUDE_smallest_valid_side_l2193_219307

-- Define the triangle sides
def a : ℝ := 7.5
def b : ℝ := 14.5

-- Define the property of s being a valid side length
def is_valid_side (s : ℕ) : Prop :=
  (a + s > b) ∧ (a + b > s) ∧ (b + s > a)

-- State the theorem
theorem smallest_valid_side :
  ∃ (s : ℕ), is_valid_side s ∧ ∀ (t : ℕ), t < s → ¬is_valid_side t :=
by sorry

end NUMINAMATH_CALUDE_smallest_valid_side_l2193_219307


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l2193_219332

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

end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l2193_219332


namespace NUMINAMATH_CALUDE_two_lights_possible_l2193_219353

/-- Represents the state of light bulbs on an infinite integer line -/
def LightState := Int → Bool

/-- Applies the template set S to the light state at position p -/
def applyTemplate (S : Finset Int) (state : LightState) (p : Int) : LightState :=
  fun i => if (i - p) ∈ S then !state i else state i

/-- Counts the number of light bulbs that are on -/
def countOn (state : LightState) : Nat :=
  sorry

theorem two_lights_possible (S : Finset Int) :
  ∃ (ops : List Int), 
    let finalState := ops.foldl (fun st p => applyTemplate S st p) (fun _ => false)
    countOn finalState = 2 :=
  sorry

end NUMINAMATH_CALUDE_two_lights_possible_l2193_219353


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l2193_219338

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_abc_properties (t : Triangle) 
  (ha : t.a = 3)
  (hb : t.b = 2)
  (hcosA : Real.cos t.A = 1/3) :
  Real.sin t.B = 4 * Real.sqrt 2 / 9 ∧ t.c = 3 := by
  sorry


end NUMINAMATH_CALUDE_triangle_abc_properties_l2193_219338


namespace NUMINAMATH_CALUDE_grandmother_inheritance_l2193_219397

/-- Proves that if 5 people equally split an amount of money and each receives $105,500, then the total amount is $527,500. -/
theorem grandmother_inheritance (num_people : ℕ) (amount_per_person : ℕ) (total_amount : ℕ) :
  num_people = 5 →
  amount_per_person = 105500 →
  total_amount = num_people * amount_per_person →
  total_amount = 527500 :=
by
  sorry

end NUMINAMATH_CALUDE_grandmother_inheritance_l2193_219397


namespace NUMINAMATH_CALUDE_parabola_focus_directrix_distance_l2193_219316

/-- Given a parabola y² = 2px with a point Q(6, y₀) on it, 
    if the distance from Q to the focus is 10, 
    then the distance from the focus to the directrix is 8. -/
theorem parabola_focus_directrix_distance 
  (p : ℝ) (y₀ : ℝ) :
  y₀^2 = 2*p*6 → -- Q(6, y₀) lies on the parabola y² = 2px
  (6 + p/2)^2 + y₀^2 = 10^2 → -- distance from Q to focus is 10
  p = 8 := -- distance from focus to directrix
by sorry

end NUMINAMATH_CALUDE_parabola_focus_directrix_distance_l2193_219316


namespace NUMINAMATH_CALUDE_sum_less_than_addends_implies_negative_l2193_219322

theorem sum_less_than_addends_implies_negative (a b : ℚ) : 
  (a + b < a ∧ a + b < b) → (a < 0 ∧ b < 0) := by
  sorry

end NUMINAMATH_CALUDE_sum_less_than_addends_implies_negative_l2193_219322


namespace NUMINAMATH_CALUDE_proportion_solution_l2193_219378

theorem proportion_solution (x : ℝ) : (x / 5 = 1.05 / 7) → x = 0.75 := by
  sorry

end NUMINAMATH_CALUDE_proportion_solution_l2193_219378


namespace NUMINAMATH_CALUDE_function_definition_l2193_219301

-- Define the function property
def is_function {A B : Type} (f : A → B) : Prop :=
  ∀ x : A, ∃! y : B, f x = y

-- State the theorem
theorem function_definition {A B : Type} (f : A → B) :
  is_function f ↔ ∀ x : A, ∃! y : B, y = f x :=
by sorry

end NUMINAMATH_CALUDE_function_definition_l2193_219301


namespace NUMINAMATH_CALUDE_grade_distribution_l2193_219390

theorem grade_distribution (total_students : ℕ) 
  (fraction_A : ℚ) (fraction_C : ℚ) (number_D : ℕ) :
  total_students = 100 →
  fraction_A = 1/5 →
  fraction_C = 1/2 →
  number_D = 5 →
  (total_students : ℚ) - (fraction_A * total_students + fraction_C * total_students + number_D) = 1/4 * total_students :=
by sorry

end NUMINAMATH_CALUDE_grade_distribution_l2193_219390


namespace NUMINAMATH_CALUDE_floor_ceiling_sum_seven_l2193_219319

theorem floor_ceiling_sum_seven (x : ℝ) :
  (Int.floor x + Int.ceil x = 7) ↔ (3 < x ∧ x < 4) ∨ x = 3.5 := by
  sorry

end NUMINAMATH_CALUDE_floor_ceiling_sum_seven_l2193_219319


namespace NUMINAMATH_CALUDE_johns_piggy_bank_l2193_219364

theorem johns_piggy_bank (total_coins quarters dimes nickels : ℕ) : 
  total_coins = 63 →
  quarters = 22 →
  dimes = quarters + 3 →
  total_coins = quarters + dimes + nickels →
  quarters - nickels = 6 :=
by sorry

end NUMINAMATH_CALUDE_johns_piggy_bank_l2193_219364


namespace NUMINAMATH_CALUDE_square_area_error_percentage_l2193_219358

/-- If the side of a square is measured with a 2% excess error, 
    then the percentage of error in the calculated area of the square is 4.04%. -/
theorem square_area_error_percentage (s : ℝ) (s' : ℝ) (A : ℝ) (A' : ℝ) :
  s' = s * (1 + 0.02) →
  A = s^2 →
  A' = s'^2 →
  (A' - A) / A * 100 = 4.04 := by
  sorry

end NUMINAMATH_CALUDE_square_area_error_percentage_l2193_219358


namespace NUMINAMATH_CALUDE_perfect_square_iff_even_exponents_kth_power_iff_divisible_exponents_l2193_219330

/-- A natural number is a perfect square if and only if each prime in its prime factorization appears an even number of times. -/
theorem perfect_square_iff_even_exponents (n : ℕ) :
  (∃ m : ℕ, n = m ^ 2) ↔ (∀ p : ℕ, Prime p → ∃ k : ℕ, n.factorization p = 2 * k) :=
sorry

/-- A natural number is a k-th power if and only if each prime in its prime factorization appears a number of times divisible by k. -/
theorem kth_power_iff_divisible_exponents (n k : ℕ) (hk : k > 0) :
  (∃ m : ℕ, n = m ^ k) ↔ (∀ p : ℕ, Prime p → ∃ l : ℕ, n.factorization p = k * l) :=
sorry

end NUMINAMATH_CALUDE_perfect_square_iff_even_exponents_kth_power_iff_divisible_exponents_l2193_219330


namespace NUMINAMATH_CALUDE_min_quadratic_l2193_219345

theorem min_quadratic (x : ℝ) : 
  (∀ y : ℝ, x^2 + 7*x + 3 ≤ y^2 + 7*y + 3) → x = -7/2 := by
  sorry

end NUMINAMATH_CALUDE_min_quadratic_l2193_219345


namespace NUMINAMATH_CALUDE_inequality_proof_l2193_219392

theorem inequality_proof (x y z : ℝ) : x^2 + y^2 + z^2 ≥ Real.sqrt 2 * (x*y + y*z) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2193_219392


namespace NUMINAMATH_CALUDE_max_silver_tokens_l2193_219366

/-- Represents the number of tokens Alex has --/
structure TokenCount where
  red : ℕ
  blue : ℕ
  silver : ℕ

/-- Represents the exchange rules --/
inductive ExchangeRule
  | RedToSilver : ExchangeRule  -- 3 red → 2 silver + 1 blue
  | BlueToSilver : ExchangeRule -- 2 blue → 1 silver + 1 red

/-- Applies an exchange rule to a token count --/
def applyExchange (tc : TokenCount) (rule : ExchangeRule) : Option TokenCount :=
  match rule with
  | ExchangeRule.RedToSilver =>
      if tc.red ≥ 3 then
        some ⟨tc.red - 3, tc.blue + 1, tc.silver + 2⟩
      else
        none
  | ExchangeRule.BlueToSilver =>
      if tc.blue ≥ 2 then
        some ⟨tc.red + 1, tc.blue - 2, tc.silver + 1⟩
      else
        none

/-- Checks if any exchange is possible --/
def canExchange (tc : TokenCount) : Bool :=
  tc.red ≥ 3 ∨ tc.blue ≥ 2

/-- The main theorem to prove --/
theorem max_silver_tokens (initialRed initialBlue : ℕ) 
    (h1 : initialRed = 100) (h2 : initialBlue = 50) :
    ∃ (finalTokens : TokenCount),
      finalTokens.red < 3 ∧ 
      finalTokens.blue < 2 ∧
      finalTokens.silver = 147 ∧
      (∃ (exchanges : List ExchangeRule), 
        finalTokens = exchanges.foldl 
          (fun acc rule => 
            match applyExchange acc rule with
            | some newCount => newCount
            | none => acc) 
          ⟨initialRed, initialBlue, 0⟩) := by
  sorry

end NUMINAMATH_CALUDE_max_silver_tokens_l2193_219366


namespace NUMINAMATH_CALUDE_books_from_second_shop_is_35_l2193_219329

/-- The number of books Rahim bought from the second shop -/
def books_from_second_shop : ℕ := sorry

/-- The total amount spent on books -/
def total_spent : ℕ := 6500 + 2000

/-- The total number of books bought -/
def total_books : ℕ := 65 + books_from_second_shop

/-- The average price per book -/
def average_price : ℚ := 85

theorem books_from_second_shop_is_35 :
  books_from_second_shop = 35 ∧
  65 * 100 = 6500 ∧
  books_from_second_shop * average_price = 2000 ∧
  average_price * total_books = total_spent := by sorry

end NUMINAMATH_CALUDE_books_from_second_shop_is_35_l2193_219329


namespace NUMINAMATH_CALUDE_coin_problem_l2193_219386

/-- Represents the number of coins of each type in the bag -/
def num_coins : ℕ := sorry

/-- Represents the total value of coins in rupees -/
def total_value : ℚ := 140

/-- Theorem stating that if the bag contains an equal number of one rupee, 50 paise, and 25 paise coins, 
    and the total value is 140 rupees, then the number of coins of each type is 80 -/
theorem coin_problem : 
  (num_coins : ℚ) + (num_coins : ℚ) * (1/2) + (num_coins : ℚ) * (1/4) = total_value → 
  num_coins = 80 := by sorry

end NUMINAMATH_CALUDE_coin_problem_l2193_219386


namespace NUMINAMATH_CALUDE_p_only_root_zero_l2193_219339

/-- Recursive definition of polynomial p_n(x) -/
def p : ℕ → ℝ → ℝ
| 0, x => 0
| 1, x => x
| (n+2), x => x * p (n+1) x + (1 - x) * p n x

/-- Theorem stating that 0 is the only real root of p_n(x) for n ≥ 1 -/
theorem p_only_root_zero (n : ℕ) (h : n ≥ 1) :
  ∀ x : ℝ, p n x = 0 ↔ x = 0 := by
  sorry

#check p_only_root_zero

end NUMINAMATH_CALUDE_p_only_root_zero_l2193_219339


namespace NUMINAMATH_CALUDE_f_max_value_implies_a_eq_three_l2193_219342

/-- The function f(x) = -4x^3 + ax -/
def f (a : ℝ) (x : ℝ) : ℝ := -4 * x^3 + a * x

/-- The maximum value of f(x) on [-1,1] is 1 -/
def max_value_is_one (a : ℝ) : Prop :=
  ∀ x : ℝ, x ∈ Set.Icc (-1) 1 → f a x ≤ 1 ∧ ∃ y : ℝ, y ∈ Set.Icc (-1) 1 ∧ f a y = 1

theorem f_max_value_implies_a_eq_three :
  ∀ a : ℝ, max_value_is_one a → a = 3 := by sorry

end NUMINAMATH_CALUDE_f_max_value_implies_a_eq_three_l2193_219342
