import Mathlib

namespace NUMINAMATH_CALUDE_x_gt_5_sufficient_not_necessary_for_x_sq_gt_25_l912_91262

theorem x_gt_5_sufficient_not_necessary_for_x_sq_gt_25 :
  (∀ x : ℝ, x > 5 → x^2 > 25) ∧
  (∃ x : ℝ, x^2 > 25 ∧ x ≤ 5) := by
  sorry

end NUMINAMATH_CALUDE_x_gt_5_sufficient_not_necessary_for_x_sq_gt_25_l912_91262


namespace NUMINAMATH_CALUDE_sqrt_sqrt_equation_l912_91217

theorem sqrt_sqrt_equation (x : ℝ) : Real.sqrt (Real.sqrt x) = 3 → x = 81 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sqrt_equation_l912_91217


namespace NUMINAMATH_CALUDE_third_quarter_gdp_l912_91238

/-- Represents the GDP growth over quarters -/
def gdp_growth (initial_gdp : ℝ) (growth_rate : ℝ) (quarters : ℕ) : ℝ :=
  initial_gdp * (1 + growth_rate) ^ quarters

theorem third_quarter_gdp 
  (initial_gdp : ℝ) 
  (growth_rate : ℝ) :
  gdp_growth initial_gdp growth_rate 2 = initial_gdp * (1 + growth_rate)^2 :=
by sorry

end NUMINAMATH_CALUDE_third_quarter_gdp_l912_91238


namespace NUMINAMATH_CALUDE_paving_cost_specific_room_l912_91275

/-- Calculates the cost of paving a floor consisting of two rectangles -/
def paving_cost (length1 width1 length2 width2 cost_per_sqm : ℝ) : ℝ :=
  ((length1 * width1 + length2 * width2) * cost_per_sqm)

/-- Theorem: The cost of paving the specific room is Rs. 26,100 -/
theorem paving_cost_specific_room :
  paving_cost 5.5 3.75 4 3 800 = 26100 := by
  sorry

end NUMINAMATH_CALUDE_paving_cost_specific_room_l912_91275


namespace NUMINAMATH_CALUDE_largest_r_is_two_l912_91277

/-- A sequence of positive integers satisfying the given inequality -/
def ValidSequence (a : ℕ → ℕ) (r : ℝ) : Prop :=
  ∀ n : ℕ, (a n ≤ a (n + 2)) ∧ ((a (n + 2))^2 ≤ (a n)^2 + r * a (n + 1))

/-- The sequence eventually stabilizes -/
def EventuallyStable (a : ℕ → ℕ) : Prop :=
  ∃ M : ℕ, ∀ n ≥ M, a (n + 2) = a n

/-- The main theorem stating that 2 is the largest real number satisfying the condition -/
theorem largest_r_is_two :
  (∀ a : ℕ → ℕ, ValidSequence a 2 → EventuallyStable a) ∧
  (∀ r > 2, ∃ a : ℕ → ℕ, ValidSequence a r ∧ ¬EventuallyStable a) := by
  sorry

end NUMINAMATH_CALUDE_largest_r_is_two_l912_91277


namespace NUMINAMATH_CALUDE_correct_donations_l912_91284

/-- Represents the donations to five orphanages -/
structure OrphanageDonations where
  total : ℝ
  first : ℝ
  second : ℝ
  third : ℝ
  fourth : ℝ
  fifth : ℝ

/-- Checks if the donations satisfy the given conditions -/
def validDonations (d : OrphanageDonations) : Prop :=
  d.total = 1300 ∧
  d.first = 0.2 * d.total ∧
  d.second = d.first / 2 ∧
  d.third = 2 * d.second ∧
  d.fourth = d.fifth ∧
  d.fourth + d.fifth = d.third

/-- Theorem stating that the given donations satisfy all conditions -/
theorem correct_donations :
  ∃ d : OrphanageDonations,
    validDonations d ∧
    d.first = 260 ∧
    d.second = 130 ∧
    d.third = 260 ∧
    d.fourth = 130 ∧
    d.fifth = 130 :=
sorry

end NUMINAMATH_CALUDE_correct_donations_l912_91284


namespace NUMINAMATH_CALUDE_combination_sum_equals_84_l912_91227

theorem combination_sum_equals_84 : Nat.choose 8 2 + Nat.choose 8 3 = 84 := by
  sorry

end NUMINAMATH_CALUDE_combination_sum_equals_84_l912_91227


namespace NUMINAMATH_CALUDE_symmetry_axis_phi_l912_91260

/-- The value of φ when f(x) and g(x) have the same axis of symmetry --/
theorem symmetry_axis_phi : ∀ (ω : ℝ), ω > 0 →
  (∀ (φ : ℝ), |φ| < π/2 →
    (∀ (x : ℝ), 3 * Real.sin (ω * x - π/3) = 3 * Real.sin (ω * x + φ + π/2)) →
    φ = π/6) :=
by sorry

end NUMINAMATH_CALUDE_symmetry_axis_phi_l912_91260


namespace NUMINAMATH_CALUDE_intersection_implies_a_value_l912_91231

def A (a : ℝ) : Set ℝ := {2, 4, a^3 - 2*a^2 - a + 7}

def B (a : ℝ) : Set ℝ := {-4, a + 3, a^2 - 2*a + 2, a^3 + a^2 + 3*a + 7}

theorem intersection_implies_a_value (a : ℝ) :
  A a ∩ B a = {2, 5} → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_implies_a_value_l912_91231


namespace NUMINAMATH_CALUDE_least_integer_divisible_by_three_primes_l912_91235

theorem least_integer_divisible_by_three_primes : 
  ∃ n : ℕ, n > 0 ∧ 
  (∃ p q r : ℕ, Prime p ∧ Prime q ∧ Prime r ∧ p ≠ q ∧ q ≠ r ∧ p ≠ r ∧ n % p = 0 ∧ n % q = 0 ∧ n % r = 0) ∧
  (∀ m : ℕ, m > 0 → 
    (∃ p q r : ℕ, Prime p ∧ Prime q ∧ Prime r ∧ p ≠ q ∧ q ≠ r ∧ p ≠ r ∧ m % p = 0 ∧ m % q = 0 ∧ m % r = 0) → 
    m ≥ 30) :=
by
  sorry

end NUMINAMATH_CALUDE_least_integer_divisible_by_three_primes_l912_91235


namespace NUMINAMATH_CALUDE_smallest_common_multiple_13_8_lcm_13_8_l912_91224

theorem smallest_common_multiple_13_8 : 
  ∀ n : ℕ, (13 ∣ n ∧ 8 ∣ n) → n ≥ 104 := by
  sorry

theorem lcm_13_8 : Nat.lcm 13 8 = 104 := by
  sorry

end NUMINAMATH_CALUDE_smallest_common_multiple_13_8_lcm_13_8_l912_91224


namespace NUMINAMATH_CALUDE_sum_two_condition_l912_91204

theorem sum_two_condition (a b : ℝ) :
  (a = 1 ∧ b = 1 → a + b = 2) ∧
  (∃ a b : ℝ, a + b = 2 ∧ ¬(a = 1 ∧ b = 1)) :=
by sorry

end NUMINAMATH_CALUDE_sum_two_condition_l912_91204


namespace NUMINAMATH_CALUDE_only_cube_has_congruent_views_l912_91243

-- Define the possible solids
inductive Solid
  | Cone
  | Cylinder
  | Cube
  | SquarePyramid

-- Define a function to check if a solid has congruent views
def hasCongruentViews (s : Solid) : Prop :=
  match s with
  | Solid.Cone => False
  | Solid.Cylinder => False
  | Solid.Cube => True
  | Solid.SquarePyramid => False

-- Theorem stating that only a cube has congruent views
theorem only_cube_has_congruent_views :
  ∀ s : Solid, hasCongruentViews s ↔ s = Solid.Cube :=
by sorry

end NUMINAMATH_CALUDE_only_cube_has_congruent_views_l912_91243


namespace NUMINAMATH_CALUDE_condition_necessary_not_sufficient_l912_91248

/-- A sequence of real numbers -/
def Sequence := ℕ → ℝ

/-- Definition of an arithmetic sequence -/
def is_arithmetic (a : Sequence) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The condition a₃ + a₇ = 2a₅ -/
def condition (a : Sequence) : Prop :=
  a 3 + a 7 = 2 * a 5

theorem condition_necessary_not_sufficient :
  (∀ a : Sequence, is_arithmetic a → condition a) ∧
  (∃ a : Sequence, condition a ∧ ¬is_arithmetic a) :=
sorry

end NUMINAMATH_CALUDE_condition_necessary_not_sufficient_l912_91248


namespace NUMINAMATH_CALUDE_other_root_of_quadratic_l912_91206

theorem other_root_of_quadratic (a c : ℝ) (h : a ≠ 0) :
  (∃ x, 4 * a * x^2 - 2 * a * x + c = 0 ∧ x = 0) →
  (∃ y, 4 * a * y^2 - 2 * a * y + c = 0 ∧ y = 1/2) :=
by sorry

end NUMINAMATH_CALUDE_other_root_of_quadratic_l912_91206


namespace NUMINAMATH_CALUDE_complex_to_exponential_form_l912_91200

theorem complex_to_exponential_form (z : ℂ) :
  z = 2 - I →
  Real.arctan (1 / 2) = Real.arctan (Complex.abs z / Complex.im z) :=
by sorry

end NUMINAMATH_CALUDE_complex_to_exponential_form_l912_91200


namespace NUMINAMATH_CALUDE_x_intercepts_count_l912_91228

theorem x_intercepts_count :
  let f : ℝ → ℝ := λ x => (x - 5) * (x^2 + 5*x + 6) * (x - 1)
  ∃! (s : Finset ℝ), (∀ x ∈ s, f x = 0) ∧ s.card = 4 :=
sorry

end NUMINAMATH_CALUDE_x_intercepts_count_l912_91228


namespace NUMINAMATH_CALUDE_f_has_zero_in_interval_l912_91265

/-- The function f(x) = x^3 + x - 8 -/
def f (x : ℝ) : ℝ := x^3 + x - 8

/-- Theorem: f(x) has a zero in the interval (1, 2) -/
theorem f_has_zero_in_interval :
  ∃ x : ℝ, x > 1 ∧ x < 2 ∧ f x = 0 :=
by
  have h1 : f 1 < 0 := by sorry
  have h2 : f 2 > 0 := by sorry
  sorry

#check f_has_zero_in_interval

end NUMINAMATH_CALUDE_f_has_zero_in_interval_l912_91265


namespace NUMINAMATH_CALUDE_quadratic_roots_problem_l912_91253

theorem quadratic_roots_problem (c d : ℝ) (r s : ℝ) : 
  c^2 - 5*c + 3 = 0 →
  d^2 - 5*d + 3 = 0 →
  (c + 2/d)^2 - r*(c + 2/d) + s = 0 →
  (d + 2/c)^2 - r*(d + 2/c) + s = 0 →
  s = 25/3 := by sorry

end NUMINAMATH_CALUDE_quadratic_roots_problem_l912_91253


namespace NUMINAMATH_CALUDE_magnitude_of_difference_vector_l912_91266

def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (2, 4)

theorem magnitude_of_difference_vector :
  let dot_product := a.1 * b.1 + a.2 * b.2
  dot_product = 10 →
  (a.1 - b.1)^2 + (a.2 - b.2)^2 = 5 := by sorry

end NUMINAMATH_CALUDE_magnitude_of_difference_vector_l912_91266


namespace NUMINAMATH_CALUDE_polynomial_expansion_l912_91269

theorem polynomial_expansion (z : ℝ) :
  (3 * z^3 + 6 * z^2 - 5 * z - 4) * (4 * z^4 - 3 * z^2 + 7) =
  12 * z^7 + 24 * z^6 - 29 * z^5 - 34 * z^4 + 36 * z^3 + 54 * z^2 + 35 * z - 28 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_expansion_l912_91269


namespace NUMINAMATH_CALUDE_bakery_sugar_amount_l912_91214

/-- Given the ratios of ingredients in a bakery storage room, prove the amount of sugar. -/
theorem bakery_sugar_amount (sugar flour baking_soda : ℝ) 
  (h1 : sugar / flour = 3 / 8)
  (h2 : flour / baking_soda = 10 / 1)
  (h3 : flour / (baking_soda + 60) = 8 / 1) :
  sugar = 900 := by
  sorry

end NUMINAMATH_CALUDE_bakery_sugar_amount_l912_91214


namespace NUMINAMATH_CALUDE_one_certain_event_l912_91233

-- Define the events
inductive Event
  | WaterFreeze : Event
  | RectangleArea : Event
  | CoinToss : Event
  | ExamScore : Event

-- Define a function to check if an event is certain
def isCertain (e : Event) : Prop :=
  match e with
  | Event.WaterFreeze => False
  | Event.RectangleArea => True
  | Event.CoinToss => False
  | Event.ExamScore => False

-- Theorem statement
theorem one_certain_event :
  (∃! e : Event, isCertain e) :=
sorry

end NUMINAMATH_CALUDE_one_certain_event_l912_91233


namespace NUMINAMATH_CALUDE_rearrangement_does_not_increase_length_l912_91247

/-- A segment on a line --/
structure Segment where
  left : ℝ
  right : ℝ
  h : left ≤ right

/-- A finite set of segments on a line --/
def SegmentSystem := Finset Segment

/-- The total length of the union of segments in a system --/
def totalLength (S : SegmentSystem) : ℝ := sorry

/-- The distance between midpoints of two segments --/
def midpointDistance (s₁ s₂ : Segment) : ℝ := sorry

/-- A rearrangement of segments that minimizes midpoint distances --/
def rearrange (S : SegmentSystem) : SegmentSystem := sorry

/-- The theorem stating that rearrangement does not increase total length --/
theorem rearrangement_does_not_increase_length (S : SegmentSystem) :
  totalLength (rearrange S) ≤ totalLength S := by sorry

end NUMINAMATH_CALUDE_rearrangement_does_not_increase_length_l912_91247


namespace NUMINAMATH_CALUDE_no_solution_exists_l912_91225

theorem no_solution_exists : ¬∃ (x y z : ℝ), 
  (x^2 ≠ y^2) ∧ (y^2 ≠ z^2) ∧ (z^2 ≠ x^2) ∧
  (1 / (x^2 - y^2) + 1 / (y^2 - z^2) + 1 / (z^2 - x^2) = 0) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_exists_l912_91225


namespace NUMINAMATH_CALUDE_horse_rider_ratio_l912_91244

theorem horse_rider_ratio :
  ∀ (total_horses : ℕ) (total_legs_walking : ℕ),
    total_horses = 12 →
    total_legs_walking = 60 →
    ∃ (riding_owners walking_owners : ℕ),
      riding_owners + walking_owners = total_horses ∧
      walking_owners * 6 = total_legs_walking ∧
      riding_owners * 6 = total_horses := by
  sorry

end NUMINAMATH_CALUDE_horse_rider_ratio_l912_91244


namespace NUMINAMATH_CALUDE_ice_cream_sundaes_l912_91229

theorem ice_cream_sundaes (n : ℕ) (h : n = 8) : Nat.choose n 2 = 28 := by
  sorry

end NUMINAMATH_CALUDE_ice_cream_sundaes_l912_91229


namespace NUMINAMATH_CALUDE_problem_solution_l912_91232

theorem problem_solution (a b c d m n : ℕ+) 
  (h1 : a^2 + b^2 + c^2 + d^2 = 1989)
  (h2 : a + b + c + d = m^2)
  (h3 : max a (max b (max c d)) = n^2) :
  m = 9 ∧ n = 6 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l912_91232


namespace NUMINAMATH_CALUDE_nancy_problem_rate_l912_91272

/-- Given Nancy's homework details, prove she can finish 8 problems per hour -/
theorem nancy_problem_rate :
  let math_problems : ℝ := 17.0
  let spelling_problems : ℝ := 15.0
  let total_hours : ℝ := 4.0
  let total_problems := math_problems + spelling_problems
  let problems_per_hour := total_problems / total_hours
  problems_per_hour = 8 := by sorry

end NUMINAMATH_CALUDE_nancy_problem_rate_l912_91272


namespace NUMINAMATH_CALUDE_herring_cost_theorem_l912_91299

def green_herring_price : ℝ := 2.50
def blue_herring_price : ℝ := 4.00
def green_herring_pounds : ℝ := 12
def blue_herring_pounds : ℝ := 7

theorem herring_cost_theorem :
  green_herring_price * green_herring_pounds + blue_herring_price * blue_herring_pounds = 58 :=
by sorry

end NUMINAMATH_CALUDE_herring_cost_theorem_l912_91299


namespace NUMINAMATH_CALUDE_f_inequality_solution_f_bounded_by_mn_l912_91218

def f (x : ℝ) : ℝ := 2 * |x| + |x - 1|

theorem f_inequality_solution (x : ℝ) :
  f x > 4 ↔ x < -1 ∨ x > 5/3 := by sorry

theorem f_bounded_by_mn (m n : ℝ) (hm : m > 0) (hn : n > 0) :
  {x : ℝ | f x ≤ 1/m^2 + 1/n^2 + 2*n*m} = {x : ℝ | -1 ≤ x ∧ x ≤ 5/3} := by sorry

end NUMINAMATH_CALUDE_f_inequality_solution_f_bounded_by_mn_l912_91218


namespace NUMINAMATH_CALUDE_number_exceeding_twelve_percent_l912_91203

theorem number_exceeding_twelve_percent : ∃ x : ℝ, x = 0.12 * x + 52.8 ∧ x = 60 := by
  sorry

end NUMINAMATH_CALUDE_number_exceeding_twelve_percent_l912_91203


namespace NUMINAMATH_CALUDE_perimeter_difference_l912_91245

/-- The perimeter of a rectangle --/
def rectanglePerimeter (length width : ℕ) : ℕ := 2 * (length + width)

/-- The perimeter of a stack of rectangles --/
def stackedRectanglesPerimeter (length width count : ℕ) : ℕ :=
  2 * length + 2 * (width * count)

/-- The difference in perimeters between a 6x1 rectangle and three 3x1 rectangles stacked vertically --/
theorem perimeter_difference :
  rectanglePerimeter 6 1 - stackedRectanglesPerimeter 3 1 3 = 2 := by
  sorry


end NUMINAMATH_CALUDE_perimeter_difference_l912_91245


namespace NUMINAMATH_CALUDE_advantage_is_most_appropriate_l912_91291

/-- Represents the beneficial aspect of language skills in a job context -/
def BeneficialAspect : Type := String

/-- The set of possible words to fill in the blank -/
def WordChoices : Set String := {"chance", "ability", "possibility", "advantage"}

/-- Predicate to check if a word appropriately describes the beneficial aspect of language skills -/
def IsAppropriateWord (word : String) : Prop :=
  word ∈ WordChoices ∧ 
  word = "advantage"

/-- Theorem stating that "advantage" is the most appropriate word -/
theorem advantage_is_most_appropriate : 
  ∃ (word : String), IsAppropriateWord word ∧ 
  ∀ (other : String), IsAppropriateWord other → other = word :=
sorry

end NUMINAMATH_CALUDE_advantage_is_most_appropriate_l912_91291


namespace NUMINAMATH_CALUDE_inequality_range_of_p_l912_91246

-- Define the inequality function
def inequality (a p : ℝ) : Prop :=
  Real.sqrt a - Real.sqrt (a - 1) > Real.sqrt (a - 2) - Real.sqrt (a - p)

-- State the theorem
theorem inequality_range_of_p :
  ∀ a p : ℝ, a ≥ 3 → p > 2 → 
  (∀ x : ℝ, x ≥ 3 → inequality x p) →
  2 < p ∧ p < 2 * Real.sqrt 6 + 2 * Real.sqrt 3 - 2 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_inequality_range_of_p_l912_91246


namespace NUMINAMATH_CALUDE_special_operation_result_l912_91298

-- Define the triangle operation
def triangle (a b : ℚ) : ℚ := a + b - 1

-- Define the odot operation
def odot (a b : ℚ) : ℚ := a * b - a^2

-- Theorem statement
theorem special_operation_result :
  odot (-2) (triangle 8 (-3)) = -12 := by sorry

end NUMINAMATH_CALUDE_special_operation_result_l912_91298


namespace NUMINAMATH_CALUDE_sum_of_common_ratios_l912_91264

/-- Given two nonconstant geometric sequences with different common ratios,
    prove that the sum of their common ratios is equal to k. -/
theorem sum_of_common_ratios (k a₂ a₃ b₂ b₃ : ℝ) 
  (hk : k ≠ 0)
  (ha : a₂ ≠ k ∧ a₃ ≠ a₂)  -- Ensures (k, a₂, a₃) is nonconstant
  (hb : b₂ ≠ k ∧ b₃ ≠ b₂)  -- Ensures (k, b₂, b₃) is nonconstant
  (hdiff : a₂ / k ≠ b₂ / k)  -- Ensures different common ratios
  (heq : a₃ - b₃ = k^2 * (a₂ - b₂)) :
  ∃ p q : ℝ, p ≠ q ∧ 
    a₃ = k * p^2 ∧ 
    b₃ = k * q^2 ∧ 
    a₂ = k * p ∧ 
    b₂ = k * q ∧ 
    p + q = k :=
by sorry

end NUMINAMATH_CALUDE_sum_of_common_ratios_l912_91264


namespace NUMINAMATH_CALUDE_remainder_problem_l912_91273

theorem remainder_problem (k : ℕ+) (h : 80 % (k^2 : ℕ) = 8) : 150 % (k : ℕ) = 6 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l912_91273


namespace NUMINAMATH_CALUDE_proposition_2_proposition_3_proposition_4_l912_91249

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relationships between lines and planes
variable (contains : Plane → Line → Prop)
variable (parallel : Line → Line → Prop)
variable (parallel_plane : Plane → Plane → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (intersect : Plane → Plane → Line → Prop)

-- Theorem for proposition 2
theorem proposition_2 
  (l m : Line) (α β : Plane) :
  contains α l → 
  parallel_line_plane l β → 
  intersect α β m → 
  parallel l m :=
sorry

-- Theorem for proposition 3
theorem proposition_3 
  (l : Line) (α β : Plane) :
  parallel_plane α β → 
  parallel_line_plane l α → 
  parallel_line_plane l β :=
sorry

-- Theorem for proposition 4
theorem proposition_4 
  (l m : Line) (α β : Plane) :
  perpendicular l α → 
  parallel l m → 
  parallel_plane α β → 
  perpendicular m β :=
sorry

end NUMINAMATH_CALUDE_proposition_2_proposition_3_proposition_4_l912_91249


namespace NUMINAMATH_CALUDE_expenditure_for_specific_hall_l912_91267

/-- Calculates the total expenditure for covering the interior of a rectangular hall with mat -/
def total_expenditure (length width height cost_per_sqm : ℝ) : ℝ :=
  let floor_area := length * width
  let wall_area := 2 * (length * height + width * height)
  let total_area := 2 * floor_area + wall_area
  total_area * cost_per_sqm

/-- Proves that the total expenditure for covering the interior of a specific rectangular hall with mat is Rs. 9500 -/
theorem expenditure_for_specific_hall :
  total_expenditure 20 15 5 10 = 9500 := by
  sorry

end NUMINAMATH_CALUDE_expenditure_for_specific_hall_l912_91267


namespace NUMINAMATH_CALUDE_tank_capacity_l912_91202

theorem tank_capacity : ℝ → Prop :=
  fun capacity =>
    let initial_fraction : ℚ := 1/4
    let final_fraction : ℚ := 3/4
    let added_water : ℝ := 180
    initial_fraction * capacity + added_water = final_fraction * capacity →
    capacity = 360

-- Proof
example : tank_capacity 360 := by
  sorry

end NUMINAMATH_CALUDE_tank_capacity_l912_91202


namespace NUMINAMATH_CALUDE_lychees_remaining_l912_91255

theorem lychees_remaining (initial : ℕ) (sold_fraction : ℚ) (eaten_fraction : ℚ) : 
  initial = 500 → 
  sold_fraction = 1/2 → 
  eaten_fraction = 3/5 → 
  (initial - initial * sold_fraction - (initial - initial * sold_fraction) * eaten_fraction : ℚ) = 100 := by
  sorry

end NUMINAMATH_CALUDE_lychees_remaining_l912_91255


namespace NUMINAMATH_CALUDE_finite_solutions_factorial_difference_l912_91219

theorem finite_solutions_factorial_difference (u : ℕ+) :
  ∃ (S : Finset (ℕ × ℕ × ℕ)), ∀ (n a b : ℕ),
    n! = u^a - u^b → (n, a, b) ∈ S :=
sorry

end NUMINAMATH_CALUDE_finite_solutions_factorial_difference_l912_91219


namespace NUMINAMATH_CALUDE_correct_divisor_proof_l912_91251

theorem correct_divisor_proof (dividend : ℕ) (mistaken_divisor correct_quotient : ℕ) 
  (h1 : dividend % mistaken_divisor = 0)
  (h2 : dividend / mistaken_divisor = 63)
  (h3 : mistaken_divisor = 12)
  (h4 : dividend % correct_quotient = 0)
  (h5 : dividend / correct_quotient = 36) :
  dividend / 36 = 21 := by
  sorry

end NUMINAMATH_CALUDE_correct_divisor_proof_l912_91251


namespace NUMINAMATH_CALUDE_simplify_sqrt_expression_simplify_algebraic_expression_simplify_complex_sqrt_expression_simplify_difference_of_squares_l912_91281

-- (1)
theorem simplify_sqrt_expression : 
  Real.sqrt 18 - Real.sqrt 8 + Real.sqrt (1/8) = (5 * Real.sqrt 2) / 4 := by sorry

-- (2)
theorem simplify_algebraic_expression : 
  (1 + Real.sqrt 3) * (2 - Real.sqrt 3) = Real.sqrt 3 - 1 := by sorry

-- (3)
theorem simplify_complex_sqrt_expression : 
  (Real.sqrt 15 + Real.sqrt 60) / Real.sqrt 3 - 3 * Real.sqrt 5 = -5 * Real.sqrt 5 := by sorry

-- (4)
theorem simplify_difference_of_squares : 
  (Real.sqrt 7 + Real.sqrt 3) * (Real.sqrt 7 - Real.sqrt 3) - Real.sqrt 36 = -2 := by sorry

end NUMINAMATH_CALUDE_simplify_sqrt_expression_simplify_algebraic_expression_simplify_complex_sqrt_expression_simplify_difference_of_squares_l912_91281


namespace NUMINAMATH_CALUDE_same_color_probability_l912_91213

/-- The number of color options for neckties -/
def necktie_colors : ℕ := 6

/-- The number of color options for shirts -/
def shirt_colors : ℕ := 5

/-- The number of color options for hats -/
def hat_colors : ℕ := 4

/-- The number of color options for socks -/
def sock_colors : ℕ := 3

/-- The number of colors available for all item types -/
def common_colors : ℕ := 3

/-- The probability of selecting items of the same color for a box -/
theorem same_color_probability : 
  (common_colors : ℚ) / (necktie_colors * shirt_colors * hat_colors * sock_colors) = 1 / 120 := by
  sorry

end NUMINAMATH_CALUDE_same_color_probability_l912_91213


namespace NUMINAMATH_CALUDE_even_function_implies_a_zero_l912_91208

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - |x + a|

-- State the theorem
theorem even_function_implies_a_zero (a : ℝ) :
  (∀ x : ℝ, f a x = f a (-x)) → a = 0 := by
  sorry

end NUMINAMATH_CALUDE_even_function_implies_a_zero_l912_91208


namespace NUMINAMATH_CALUDE_mass_percentage_N_is_9_66_l912_91256

/-- The mass percentage of N in a certain compound -/
def mass_percentage_N : ℝ := 9.66

/-- Theorem stating that the mass percentage of N in the compound is 9.66% -/
theorem mass_percentage_N_is_9_66 : mass_percentage_N = 9.66 := by
  sorry

end NUMINAMATH_CALUDE_mass_percentage_N_is_9_66_l912_91256


namespace NUMINAMATH_CALUDE_existential_and_true_proposition_l912_91270

theorem existential_and_true_proposition :
  (∃ (a : ℕ), a^2 + a ≤ 0) ∧
  (∃ (a : ℕ), a^2 + a ≤ 0) = True :=
by sorry

end NUMINAMATH_CALUDE_existential_and_true_proposition_l912_91270


namespace NUMINAMATH_CALUDE_interest_less_than_principal_l912_91290

/-- Calculates the simple interest given principal, rate, and time -/
def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time

/-- Calculates the difference between principal and interest -/
def interest_difference (principal : ℝ) (interest : ℝ) : ℝ :=
  principal - interest

theorem interest_less_than_principal : 
  let principal : ℝ := 400.00000000000006
  let rate : ℝ := 0.04
  let time : ℝ := 8
  let interest := simple_interest principal rate time
  interest_difference principal interest = 272 := by
  sorry

end NUMINAMATH_CALUDE_interest_less_than_principal_l912_91290


namespace NUMINAMATH_CALUDE_smallest_advantageous_discount_l912_91237

def two_successive_discounts (d : ℝ) : ℝ := (1 - d) * (1 - d)
def three_successive_discounts (d : ℝ) : ℝ := (1 - d) * (1 - d) * (1 - d)
def two_different_discounts (d1 d2 : ℝ) : ℝ := (1 - d1) * (1 - d2)

theorem smallest_advantageous_discount : ∀ n : ℕ, n ≥ 34 →
  (1 - n / 100 < two_successive_discounts 0.18) ∧
  (1 - n / 100 < three_successive_discounts 0.12) ∧
  (1 - n / 100 < two_different_discounts 0.28 0.07) ∧
  (∀ m : ℕ, m < 34 →
    (1 - m / 100 ≥ two_successive_discounts 0.18) ∨
    (1 - m / 100 ≥ three_successive_discounts 0.12) ∨
    (1 - m / 100 ≥ two_different_discounts 0.28 0.07)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_advantageous_discount_l912_91237


namespace NUMINAMATH_CALUDE_units_digit_of_product_l912_91278

theorem units_digit_of_product (a b c : ℕ) : 
  (2^1501 * 5^1602 * 11^1703) % 10 = 0 := by sorry

end NUMINAMATH_CALUDE_units_digit_of_product_l912_91278


namespace NUMINAMATH_CALUDE_stating_probability_reroll_two_dice_l912_91261

/-- Represents a fair six-sided die -/
def Die := Fin 6

/-- The sum we're aiming for -/
def targetSum : ℕ := 9

/-- The total number of possible outcomes when rolling three dice -/
def totalOutcomes : ℕ := 216

/-- 
Represents the optimal strategy for rerolling dice to achieve the target sum.
d1, d2, d3 are the values of the three dice.
Returns the number of dice to reroll (0, 1, 2, or 3).
-/
def optimalReroll (d1 d2 d3 : Die) : Fin 4 :=
  sorry

/-- 
The number of outcomes where rerolling exactly two dice is optimal.
-/
def twoRerollOutcomes : ℕ := 84

/-- 
Theorem stating that the probability of choosing to reroll exactly two dice
to optimize the chances of getting a sum of 9 is 7/18.
-/
theorem probability_reroll_two_dice :
  (twoRerollOutcomes : ℚ) / totalOutcomes = 7 / 18 := by
  sorry

end NUMINAMATH_CALUDE_stating_probability_reroll_two_dice_l912_91261


namespace NUMINAMATH_CALUDE_min_both_mozart_and_bach_l912_91215

theorem min_both_mozart_and_bach 
  (total : ℕ) 
  (mozart : ℕ) 
  (bach : ℕ) 
  (h1 : total = 100) 
  (h2 : mozart = 87) 
  (h3 : bach = 70) 
  : ℕ :=
by
  sorry

#check min_both_mozart_and_bach

end NUMINAMATH_CALUDE_min_both_mozart_and_bach_l912_91215


namespace NUMINAMATH_CALUDE_g_50_not_18_l912_91288

/-- The number of positive integer divisors of n -/
def num_divisors (n : ℕ+) : ℕ := sorry

/-- The smallest positive divisor of n -/
def smallest_divisor (n : ℕ+) : ℕ+ := sorry

/-- g₁ function as defined in the problem -/
def g₁ (n : ℕ+) : ℕ := (num_divisors n) * (smallest_divisor n).val

/-- General gⱼ function for j ≥ 1 -/
def g (j : ℕ) (n : ℕ+) : ℕ :=
  match j with
  | 0 => n.val
  | 1 => g₁ n
  | j+1 => g₁ ⟨g j n, sorry⟩

/-- Main theorem: For all positive integers n ≤ 100, g₅₀(n) ≠ 18 -/
theorem g_50_not_18 : ∀ n : ℕ+, n.val ≤ 100 → g 50 n ≠ 18 := by sorry

end NUMINAMATH_CALUDE_g_50_not_18_l912_91288


namespace NUMINAMATH_CALUDE_angle_C_equals_140_l912_91297

/-- A special quadrilateral ABCD where ∠A + ∠B = 180° and ∠C = ∠A -/
structure SpecialQuadrilateral where
  A : ℝ
  B : ℝ
  C : ℝ
  D : ℝ
  sum_AB : A + B = 180
  C_eq_A : C = A

/-- Theorem: In a special quadrilateral ABCD where ∠A : ∠B = 7 : 2, ∠C = 140° -/
theorem angle_C_equals_140 (ABCD : SpecialQuadrilateral) (h : ABCD.A / ABCD.B = 7 / 2) : 
  ABCD.C = 140 := by
  sorry

end NUMINAMATH_CALUDE_angle_C_equals_140_l912_91297


namespace NUMINAMATH_CALUDE_max_profit_and_optimal_price_l912_91286

/-- Represents the profit function for a product with given initial conditions -/
def profit (x : ℝ) : ℝ :=
  (500 - 10 * x) * ((50 + x) - 40)

/-- Theorem stating the maximum profit and optimal selling price -/
theorem max_profit_and_optimal_price :
  ∃ (max_profit : ℝ) (optimal_price : ℝ),
    (∀ x : ℝ, profit x ≤ max_profit) ∧
    (profit (optimal_price - 50) = max_profit) ∧
    max_profit = 9000 ∧
    optimal_price = 70 := by
  sorry

#check max_profit_and_optimal_price

end NUMINAMATH_CALUDE_max_profit_and_optimal_price_l912_91286


namespace NUMINAMATH_CALUDE_right_triangle_x_coordinate_l912_91226

/-- Given points P, Q, and R forming a right triangle with ∠PQR = 90°, prove that the x-coordinate of R is 13. -/
theorem right_triangle_x_coordinate :
  let P : ℝ × ℝ := (2, 0)
  let Q : ℝ × ℝ := (11, -3)
  let R : ℝ × ℝ := (x, 3)
  ∀ x : ℝ,
  (Q.1 - P.1) * (R.1 - Q.1) + (Q.2 - P.2) * (R.2 - Q.2) = 0 →
  x = 13 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_x_coordinate_l912_91226


namespace NUMINAMATH_CALUDE_dogs_and_video_games_percentage_l912_91212

theorem dogs_and_video_games_percentage 
  (total_students : ℕ) 
  (dogs_preference : ℕ) 
  (dogs_and_movies_percent : ℚ) : 
  total_students = 30 →
  dogs_preference = 18 →
  dogs_and_movies_percent = 10 / 100 →
  (dogs_preference - (dogs_and_movies_percent * total_students).num) / total_students = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_dogs_and_video_games_percentage_l912_91212


namespace NUMINAMATH_CALUDE_age_difference_of_children_l912_91210

/-- Proves that the age difference between two children is 2 years given the family conditions --/
theorem age_difference_of_children (initial_members : ℕ) (initial_avg_age : ℕ) 
  (years_passed : ℕ) (current_members : ℕ) (current_avg_age : ℕ) (youngest_child_age : ℕ) :
  initial_members = 4 →
  initial_avg_age = 24 →
  years_passed = 10 →
  current_members = 6 →
  current_avg_age = 24 →
  youngest_child_age = 3 →
  ∃ (older_child_age : ℕ), 
    older_child_age - youngest_child_age = 2 ∧
    older_child_age + youngest_child_age = 
      current_members * current_avg_age - initial_members * (initial_avg_age + years_passed) :=
by
  sorry

#check age_difference_of_children

end NUMINAMATH_CALUDE_age_difference_of_children_l912_91210


namespace NUMINAMATH_CALUDE_largest_integer_solution_largest_integer_value_negative_four_satisfies_largest_integer_is_negative_four_l912_91258

theorem largest_integer_solution (x : ℤ) : (5 - 4*x > 17) ↔ (x < -3) :=
  sorry

theorem largest_integer_value : ∀ x : ℤ, (5 - 4*x > 17) → (x ≤ -4) :=
  sorry

theorem negative_four_satisfies : (5 - 4*(-4) > 17) :=
  sorry

theorem largest_integer_is_negative_four : 
  ∀ x : ℤ, (5 - 4*x > 17) → (x ≤ -4) ∧ (-4 ≤ x) → x = -4 :=
  sorry

end NUMINAMATH_CALUDE_largest_integer_solution_largest_integer_value_negative_four_satisfies_largest_integer_is_negative_four_l912_91258


namespace NUMINAMATH_CALUDE_apple_juice_production_l912_91236

/-- Given the annual U.S. apple production and its distribution, calculate the amount used for juice -/
theorem apple_juice_production (total_production : ℝ) (cider_percentage : ℝ) (juice_percentage : ℝ) :
  total_production = 7 →
  cider_percentage = 0.25 →
  juice_percentage = 0.60 →
  (total_production * (1 - cider_percentage) * juice_percentage) = 3.15 := by
  sorry

end NUMINAMATH_CALUDE_apple_juice_production_l912_91236


namespace NUMINAMATH_CALUDE_smallest_divisible_by_1_to_12_l912_91285

theorem smallest_divisible_by_1_to_12 : ∃ (n : ℕ), n > 0 ∧ (∀ k : ℕ, 1 ≤ k ∧ k ≤ 12 → k ∣ n) ∧ (∀ m : ℕ, m > 0 ∧ (∀ k : ℕ, 1 ≤ k ∧ k ≤ 12 → k ∣ m) → n ≤ m) ∧ n = 27720 := by
  sorry

end NUMINAMATH_CALUDE_smallest_divisible_by_1_to_12_l912_91285


namespace NUMINAMATH_CALUDE_statement_1_statement_4_l912_91283

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)
variable (line_parallel_plane : Line → Plane → Prop)
variable (plane_perpendicular : Plane → Plane → Prop)

-- Define the axioms
variable (m n : Line)
variable (α β : Plane)
variable (h_different_lines : m ≠ n)
variable (h_non_coincident_planes : α ≠ β)

-- Statement 1
theorem statement_1 : 
  perpendicular m α → perpendicular n α → parallel m n :=
sorry

-- Statement 4
theorem statement_4 :
  perpendicular m α → line_parallel_plane m β → plane_perpendicular α β :=
sorry

end NUMINAMATH_CALUDE_statement_1_statement_4_l912_91283


namespace NUMINAMATH_CALUDE_apples_fallen_count_l912_91241

/-- Represents the number of apples that fell out of Carla's backpack -/
def apples_fallen (initial : ℕ) (stolen : ℕ) (remaining : ℕ) : ℕ :=
  initial - stolen - remaining

/-- Theorem stating that 26 apples fell out of Carla's backpack -/
theorem apples_fallen_count : apples_fallen 79 45 8 = 26 := by
  sorry

end NUMINAMATH_CALUDE_apples_fallen_count_l912_91241


namespace NUMINAMATH_CALUDE_triangle_exists_l912_91279

/-- Theorem: A triangle exists given an angle, sum of two sides, and a median -/
theorem triangle_exists (α : Real) (sum_sides : Real) (median : Real) :
  ∃ (a b c : Real),
    0 < a ∧ 0 < b ∧ 0 < c ∧
    0 < α ∧ α < π ∧
    a + b = sum_sides ∧
    ((a + b) / 2)^2 + (c / 2)^2 = median^2 + ((a - b) / 2)^2 ∧
    c^2 = a^2 + b^2 - 2 * a * b * Real.cos α :=
by sorry


end NUMINAMATH_CALUDE_triangle_exists_l912_91279


namespace NUMINAMATH_CALUDE_age_of_b_l912_91221

theorem age_of_b (a b c : ℕ) : 
  (a + b + c) / 3 = 29 → 
  (a + c) / 2 = 32 → 
  b = 23 := by
sorry

end NUMINAMATH_CALUDE_age_of_b_l912_91221


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l912_91292

/-- Given a geometric sequence, returns the sum of the first n terms -/
noncomputable def geometricSum (a r : ℝ) (n : ℕ) : ℝ :=
  a * (1 - r^n) / (1 - r)

/-- Proves that for a geometric sequence with specific properties, 
    the sum of the first 9000 terms is 1355 -/
theorem geometric_sequence_sum 
  (a r : ℝ) 
  (h1 : geometricSum a r 3000 = 500)
  (h2 : geometricSum a r 6000 = 950) :
  geometricSum a r 9000 = 1355 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l912_91292


namespace NUMINAMATH_CALUDE_odd_function_composition_periodic_function_composition_exists_non_decreasing_composition_inverse_function_zero_l912_91242

-- Define the function f: ℝ → ℝ
variable (f : ℝ → ℝ)

-- Statement 1
theorem odd_function_composition (h : ∀ x, f (-x) = -f x) : ∀ x, (f ∘ f) (-x) = -(f ∘ f) x :=
sorry

-- Statement 2
theorem periodic_function_composition (h : ∃ T, ∀ x, f (x + T) = f x) : 
  ∃ T, ∀ x, (f ∘ f) (x + T) = (f ∘ f) x :=
sorry

-- Statement 3
theorem exists_non_decreasing_composition :
  ∃ f : ℝ → ℝ, (∀ x y, x < y → f x > f y) ∧ ¬(∀ x y, x < y → (f ∘ f) x > (f ∘ f) y) :=
sorry

-- Statement 4
theorem inverse_function_zero (h₁ : Function.Bijective f) 
  (h₂ : ∃ x, f x = Function.invFun f x) : ∃ x, f x = x :=
sorry

end NUMINAMATH_CALUDE_odd_function_composition_periodic_function_composition_exists_non_decreasing_composition_inverse_function_zero_l912_91242


namespace NUMINAMATH_CALUDE_problem_solution_l912_91259

theorem problem_solution (a b c : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_abc : a * b * c = 1)
  (h_a_c : a + 1 / c = 8)
  (h_b_a : b + 1 / a = 20) : 
  c + 1 / b = 10 / 53 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l912_91259


namespace NUMINAMATH_CALUDE_piece_exits_at_A2_l912_91230

/-- Represents the directions a piece can move --/
inductive Direction
  | Up
  | Down
  | Left
  | Right

/-- Represents a cell on the 4x4 board --/
structure Cell where
  row : Fin 4
  col : Fin 4

/-- Represents the state of the board --/
structure BoardState where
  piece_position : Cell
  arrows : Fin 4 → Fin 4 → Direction

/-- Defines the initial state of the board --/
def initial_state : BoardState := sorry

/-- Defines a single move on the board --/
def move (state : BoardState) : BoardState := sorry

/-- Checks if a cell is on the edge of the board --/
def is_edge_cell (cell : Cell) : Bool := sorry

/-- Simulates the movement of the piece until it exits the board --/
def simulate_until_exit (state : BoardState) : Cell := sorry

/-- The main theorem to prove --/
theorem piece_exits_at_A2 :
  let final_cell := simulate_until_exit initial_state
  final_cell.row = 0 ∧ final_cell.col = 1 := by sorry

end NUMINAMATH_CALUDE_piece_exits_at_A2_l912_91230


namespace NUMINAMATH_CALUDE_car_rental_cost_l912_91268

/-- The daily rental cost of a car, given a daily budget, maximum mileage, and per-mile rate. -/
theorem car_rental_cost 
  (daily_budget : ℝ) 
  (max_miles : ℝ) 
  (per_mile_rate : ℝ) 
  (h1 : daily_budget = 88) 
  (h2 : max_miles = 190) 
  (h3 : per_mile_rate = 0.2) : 
  daily_budget - max_miles * per_mile_rate = 50 := by
  sorry

#check car_rental_cost

end NUMINAMATH_CALUDE_car_rental_cost_l912_91268


namespace NUMINAMATH_CALUDE_min_product_constrained_l912_91282

theorem min_product_constrained (x y z : ℝ) : 
  x > 0 → y > 0 → z > 0 → 
  x + y + z = 1 → 
  x ≤ 3*y ∧ x ≤ 3*z ∧ y ≤ 3*x ∧ y ≤ 3*z ∧ z ≤ 3*x ∧ z ≤ 3*y → 
  x * y * z ≥ 3/125 := by
sorry

end NUMINAMATH_CALUDE_min_product_constrained_l912_91282


namespace NUMINAMATH_CALUDE_london_trip_train_time_l912_91274

/-- Calculates the train ride time given the total trip time and other components of the journey. -/
def train_ride_time (total_trip_time : ℕ) (bus_ride_time : ℕ) (walking_time : ℕ) : ℕ :=
  let waiting_time := 2 * walking_time
  let total_trip_minutes := total_trip_time * 60
  let non_train_time := bus_ride_time + walking_time + waiting_time
  (total_trip_minutes - non_train_time) / 60

/-- Theorem stating that given the specific journey times, the train ride takes 6 hours. -/
theorem london_trip_train_time :
  train_ride_time 8 75 15 = 6 := by
  sorry

end NUMINAMATH_CALUDE_london_trip_train_time_l912_91274


namespace NUMINAMATH_CALUDE_three_solutions_l912_91252

/-- A structure representing a solution to the equation AB = B^V --/
structure Solution :=
  (a b v : Nat)
  (h1 : a ≠ b ∧ a ≠ v ∧ b ≠ v)
  (h2 : a > 0 ∧ a < 10 ∧ b > 0 ∧ b < 10 ∧ v > 0 ∧ v < 10)
  (h3 : 10 * a + b = b^v)

/-- The set of all valid solutions --/
def allSolutions : Set Solution := {s | s.a * 10 + s.b ≥ 10 ∧ s.a * 10 + s.b < 100}

/-- The theorem stating that there are exactly three solutions --/
theorem three_solutions :
  ∃ (s1 s2 s3 : Solution),
    s1 ∈ allSolutions ∧ 
    s2 ∈ allSolutions ∧ 
    s3 ∈ allSolutions ∧
    s1.a = 3 ∧ s1.b = 2 ∧ s1.v = 5 ∧
    s2.a = 3 ∧ s2.b = 6 ∧ s2.v = 2 ∧
    s3.a = 6 ∧ s3.b = 4 ∧ s3.v = 3 ∧
    ∀ (s : Solution), s ∈ allSolutions → s = s1 ∨ s = s2 ∨ s = s3 :=
  sorry


end NUMINAMATH_CALUDE_three_solutions_l912_91252


namespace NUMINAMATH_CALUDE_projects_for_30_points_l912_91296

/-- Calculates the minimum number of projects required to earn a given number of study points -/
def min_projects (total_points : ℕ) : ℕ :=
  let block_size := 6
  let num_blocks := (total_points + block_size - 1) / block_size
  (num_blocks * (num_blocks + 1) * block_size) / 2

/-- Theorem stating that 90 projects are required to earn 30 study points -/
theorem projects_for_30_points :
  min_projects 30 = 90 := by
  sorry


end NUMINAMATH_CALUDE_projects_for_30_points_l912_91296


namespace NUMINAMATH_CALUDE_student_rank_theorem_l912_91293

/-- Calculates the rank from the last given the total number of students and rank from the top -/
def rankFromLast (totalStudents : ℕ) (rankFromTop : ℕ) : ℕ :=
  totalStudents - rankFromTop + 1

/-- Theorem stating that in a class of 35 students, if a student ranks 14th from the top, their rank from the last is 22nd -/
theorem student_rank_theorem (totalStudents : ℕ) (rankFromTop : ℕ) 
  (h1 : totalStudents = 35) (h2 : rankFromTop = 14) : 
  rankFromLast totalStudents rankFromTop = 22 := by
  sorry

end NUMINAMATH_CALUDE_student_rank_theorem_l912_91293


namespace NUMINAMATH_CALUDE_expression_evaluation_l912_91211

-- Define the expression
def expression : ℕ → ℕ → ℕ := λ a b => (3^a + 7^b)^2 - (3^a - 7^b)^2

-- State the theorem
theorem expression_evaluation :
  expression 1003 1004 = 5292 * 441^500 :=
by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l912_91211


namespace NUMINAMATH_CALUDE_f_minimum_value_l912_91205

-- Define the function f
def f (x a : ℝ) : ℝ := |x + 2| + |x - a|

-- State the theorem
theorem f_minimum_value (a : ℝ) :
  (∀ x : ℝ, f x a ≥ 3) ↔ (a ≤ -5 ∨ a ≥ 1) :=
sorry

end NUMINAMATH_CALUDE_f_minimum_value_l912_91205


namespace NUMINAMATH_CALUDE_ellipse_k_range_l912_91222

theorem ellipse_k_range (k : ℝ) :
  (∃ (x y : ℝ), 2 * x^2 + k * y^2 = 1 ∧ 
   ∃ (c : ℝ), c > 0 ∧ c^2 = 2 * x^2 + k * y^2 - k * (x^2 + y^2)) ↔ 
  (0 < k ∧ k < 2) :=
sorry

end NUMINAMATH_CALUDE_ellipse_k_range_l912_91222


namespace NUMINAMATH_CALUDE_volume_of_one_gram_l912_91280

/-- Given a substance with a density of 200 kg per cubic meter, 
    the volume of 1 gram of this substance is 5 cubic centimeters. -/
theorem volume_of_one_gram (density : ℝ) (h : density = 200) : 
  (1 / density) * (100 ^ 3) = 5 :=
sorry

end NUMINAMATH_CALUDE_volume_of_one_gram_l912_91280


namespace NUMINAMATH_CALUDE_tan_negative_three_pi_fourth_l912_91287

theorem tan_negative_three_pi_fourth : Real.tan (-3 * π / 4) = 1 := by
  sorry

end NUMINAMATH_CALUDE_tan_negative_three_pi_fourth_l912_91287


namespace NUMINAMATH_CALUDE_eight_real_numbers_inequality_l912_91216

theorem eight_real_numbers_inequality (x : Fin 8 → ℝ) (h : Function.Injective x) :
  ∃ i j : Fin 8, i ≠ j ∧ 0 < (x i - x j) / (1 + x i * x j) ∧ (x i - x j) / (1 + x i * x j) < Real.tan (π / 7) := by
  sorry

end NUMINAMATH_CALUDE_eight_real_numbers_inequality_l912_91216


namespace NUMINAMATH_CALUDE_congruence_solution_l912_91263

theorem congruence_solution (x : ℤ) : 
  x ∈ Finset.Icc 20 50 ∧ (6 * x + 5) % 10 = 19 % 10 ↔ 
  x ∈ ({24, 29, 34, 39, 44, 49} : Finset ℤ) := by
sorry

end NUMINAMATH_CALUDE_congruence_solution_l912_91263


namespace NUMINAMATH_CALUDE_stack_probability_theorem_l912_91207

/-- Represents the dimensions of a crate -/
structure CrateDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Represents the configuration of crates in the stack -/
structure StackConfiguration where
  count_3ft : ℕ
  count_4ft : ℕ
  count_6ft : ℕ

def crate_dimensions : CrateDimensions :=
  { length := 3, width := 4, height := 6 }

def total_crates : ℕ := 12

def target_height : ℕ := 50

def valid_configuration (config : StackConfiguration) : Prop :=
  config.count_3ft + config.count_4ft + config.count_6ft = total_crates ∧
  3 * config.count_3ft + 4 * config.count_4ft + 6 * config.count_6ft = target_height

def count_valid_configurations : ℕ := 30690

def total_possible_configurations : ℕ := 3^total_crates

theorem stack_probability_theorem :
  (count_valid_configurations : ℚ) / total_possible_configurations = 10230 / 531441 :=
sorry

end NUMINAMATH_CALUDE_stack_probability_theorem_l912_91207


namespace NUMINAMATH_CALUDE_jungkook_points_l912_91201

/-- Calculates the total points earned by Jungkook in a math test. -/
theorem jungkook_points (total_problems : ℕ) (correct_two_point : ℕ) (correct_one_point : ℕ) 
  (h1 : total_problems = 15)
  (h2 : correct_two_point = 8)
  (h3 : correct_one_point = 2) :
  correct_two_point * 2 + correct_one_point = 18 := by
  sorry

#check jungkook_points

end NUMINAMATH_CALUDE_jungkook_points_l912_91201


namespace NUMINAMATH_CALUDE_symmetry_classification_l912_91271

-- Define the shapes
inductive Shape
| Square
| EquilateralTriangle
| Rectangle
| Rhombus

-- Define the symmetry properties
def isAxiSymmetric : Shape → Prop
| Shape.Square => true
| Shape.EquilateralTriangle => true
| Shape.Rectangle => true
| Shape.Rhombus => true

def isCentrallySymmetric : Shape → Prop
| Shape.Square => true
| Shape.EquilateralTriangle => false
| Shape.Rectangle => true
| Shape.Rhombus => true

-- Define a function to check if a shape has both symmetries
def hasBothSymmetries (s : Shape) : Prop :=
  isAxiSymmetric s ∧ isCentrallySymmetric s

-- Theorem statement
theorem symmetry_classification :
  {s : Shape | hasBothSymmetries s} = {Shape.Square, Shape.Rectangle, Shape.Rhombus} :=
by sorry

end NUMINAMATH_CALUDE_symmetry_classification_l912_91271


namespace NUMINAMATH_CALUDE_arithmetic_sequence_tenth_term_l912_91223

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The sum of the first three terms equals 3 -/
def sum_first_three (a : ℕ → ℝ) : Prop :=
  a 1 + a 2 + a 3 = 3

/-- The sum of the 5th, 6th, and 7th terms equals 9 -/
def sum_middle_three (a : ℕ → ℝ) : Prop :=
  a 5 + a 6 + a 7 = 9

theorem arithmetic_sequence_tenth_term (a : ℕ → ℝ) 
  (h1 : arithmetic_sequence a) 
  (h2 : sum_first_three a) 
  (h3 : sum_middle_three a) : 
  a 10 = 5 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_tenth_term_l912_91223


namespace NUMINAMATH_CALUDE_equation_equivalence_l912_91254

theorem equation_equivalence (x y : ℝ) : 2 * y - 4 * x + 5 = 0 ↔ y = 2 * x - 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_equation_equivalence_l912_91254


namespace NUMINAMATH_CALUDE_carwash_problem_l912_91209

theorem carwash_problem (car_price truck_price suv_price : ℕ) 
  (total_raised num_suvs num_trucks : ℕ) : 
  car_price = 5 → 
  truck_price = 6 → 
  suv_price = 7 → 
  total_raised = 100 → 
  num_suvs = 5 → 
  num_trucks = 5 → 
  ∃ num_cars : ℕ, 
    num_cars * car_price + num_trucks * truck_price + num_suvs * suv_price = total_raised ∧ 
    num_cars = 7 :=
by
  sorry

end NUMINAMATH_CALUDE_carwash_problem_l912_91209


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l912_91294

theorem quadratic_inequality_range (a : ℝ) :
  (∀ x : ℝ, x^2 + (a - 1)*x + 1 > 0) → -1 < a ∧ a < 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l912_91294


namespace NUMINAMATH_CALUDE_largest_sum_and_simplification_l912_91257

theorem largest_sum_and_simplification : 
  let sums := [2/5 + 1/6, 2/5 + 1/3, 2/5 + 1/7, 2/5 + 1/8, 2/5 + 1/9]
  (∀ x ∈ sums, x ≤ 2/5 + 1/3) ∧ (2/5 + 1/3 = 11/15) := by
  sorry

end NUMINAMATH_CALUDE_largest_sum_and_simplification_l912_91257


namespace NUMINAMATH_CALUDE_line_through_three_points_l912_91220

/-- Given a line passing through points (0, 4), (5, k), and (15, 1), prove that k = 3 -/
theorem line_through_three_points (k : ℝ) : 
  (∃ (m b : ℝ), 4 = b ∧ k = 5*m + b ∧ 1 = 15*m + b) → k = 3 := by
  sorry

end NUMINAMATH_CALUDE_line_through_three_points_l912_91220


namespace NUMINAMATH_CALUDE_max_appearances_day_numbers_l912_91234

-- Define the cube size
def n : ℕ := 2018

-- Define a function that returns the number of times a day number appears
def day_number_appearances (i : ℕ) : ℕ :=
  if i ≤ n then
    i * (i + 1) / 2
  else if n < i ∧ i < 2 * n - 1 then
    (i + 1 - n) * (3 * n - i - 1) / 2
  else if 2 * n - 1 ≤ i ∧ i ≤ 3 * n - 2 then
    day_number_appearances (3 * n - 1 - i)
  else
    0

-- Define the maximum day number
def max_day_number : ℕ := 3 * n - 2

-- State the theorem
theorem max_appearances_day_numbers :
  ∀ k : ℕ, k ≤ max_day_number →
    day_number_appearances k ≤ day_number_appearances 3026 ∧
    day_number_appearances k ≤ day_number_appearances 3027 ∧
    (day_number_appearances 3026 = day_number_appearances 3027) :=
by sorry

end NUMINAMATH_CALUDE_max_appearances_day_numbers_l912_91234


namespace NUMINAMATH_CALUDE_octal_2016_to_binary_l912_91289

/-- Converts an octal number to decimal --/
def octal_to_decimal (octal : ℕ) : ℕ := sorry

/-- Converts a decimal number to binary --/
def decimal_to_binary (decimal : ℕ) : List ℕ := sorry

/-- Converts an octal number to binary --/
def octal_to_binary (octal : ℕ) : List ℕ :=
  decimal_to_binary (octal_to_decimal octal)

theorem octal_2016_to_binary :
  octal_to_binary 2016 = [1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0] := by sorry

end NUMINAMATH_CALUDE_octal_2016_to_binary_l912_91289


namespace NUMINAMATH_CALUDE_arthur_walk_distance_l912_91239

/-- The total number of blocks Arthur walked -/
def total_blocks : ℕ := 8 + 16

/-- The number of blocks that are one-third of a mile each -/
def first_blocks : ℕ := 10

/-- The length of each of the first blocks in miles -/
def first_block_length : ℚ := 1 / 3

/-- The length of each additional block in miles -/
def additional_block_length : ℚ := 1 / 4

/-- The total distance Arthur walked in miles -/
def total_distance : ℚ :=
  first_blocks * first_block_length + 
  (total_blocks - first_blocks) * additional_block_length

theorem arthur_walk_distance : total_distance = 41 / 6 := by
  sorry

end NUMINAMATH_CALUDE_arthur_walk_distance_l912_91239


namespace NUMINAMATH_CALUDE_rhombus_area_l912_91295

/-- The area of a rhombus with side length √145 and diagonals differing by 10 units is 208 square units. -/
theorem rhombus_area (side_length : ℝ) (diagonal_difference : ℝ) (area : ℝ) : 
  side_length = Real.sqrt 145 →
  diagonal_difference = 10 →
  area = 208 :=
by sorry

end NUMINAMATH_CALUDE_rhombus_area_l912_91295


namespace NUMINAMATH_CALUDE_quadratic_expression_value_l912_91240

theorem quadratic_expression_value (x y : ℚ) 
  (eq1 : 3 * x + 2 * y = 10) 
  (eq2 : x + 3 * y = 11) : 
  9 * x^2 + 15 * x * y + 9 * y^2 = 8097 / 49 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_expression_value_l912_91240


namespace NUMINAMATH_CALUDE_hexagon_side_length_l912_91250

/-- The side length of a regular hexagon given the distance between opposite sides -/
theorem hexagon_side_length (d : ℝ) (h : d = 10) : 
  let s := d * 2 / (3 : ℝ).sqrt
  s = 40 / 3 := by sorry

#check hexagon_side_length

end NUMINAMATH_CALUDE_hexagon_side_length_l912_91250


namespace NUMINAMATH_CALUDE_sum_of_a_and_b_l912_91276

theorem sum_of_a_and_b (a b : ℝ) : (Real.sqrt (a + 3) + abs (b - 5) = 0) → a + b = 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_a_and_b_l912_91276
