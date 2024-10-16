import Mathlib

namespace NUMINAMATH_CALUDE_star_value_of_a_l4188_418825

-- Define the star operation
def star (a b : ℝ) : ℝ := 3 * a - b^3

-- Theorem statement
theorem star_value_of_a :
  ∀ a : ℝ, star a 3 = 18 → a = 15 := by
  sorry

end NUMINAMATH_CALUDE_star_value_of_a_l4188_418825


namespace NUMINAMATH_CALUDE_least_positive_integer_divisible_by_four_distinct_primes_l4188_418832

def is_divisible_by_four_distinct_primes (n : ℕ) : Prop :=
  ∃ p q r s : ℕ, Nat.Prime p ∧ Nat.Prime q ∧ Nat.Prime r ∧ Nat.Prime s ∧
  p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s ∧
  n % p = 0 ∧ n % q = 0 ∧ n % r = 0 ∧ n % s = 0

theorem least_positive_integer_divisible_by_four_distinct_primes :
  (∀ m : ℕ, m > 0 → is_divisible_by_four_distinct_primes m → m ≥ 210) ∧
  is_divisible_by_four_distinct_primes 210 := by
  sorry

end NUMINAMATH_CALUDE_least_positive_integer_divisible_by_four_distinct_primes_l4188_418832


namespace NUMINAMATH_CALUDE_room_diagonal_l4188_418885

theorem room_diagonal (l h d : ℝ) (b : ℝ) : 
  l = 12 → h = 9 → d = 17 → d^2 = l^2 + b^2 + h^2 → b = 8 := by sorry

end NUMINAMATH_CALUDE_room_diagonal_l4188_418885


namespace NUMINAMATH_CALUDE_sum_of_polynomials_l4188_418892

/-- Given polynomials f, g, and h, prove their sum is equal to the specified polynomial -/
theorem sum_of_polynomials :
  let f : ℝ → ℝ := λ x => -4 * x^2 + 2 * x - 5
  let g : ℝ → ℝ := λ x => -6 * x^2 + 4 * x - 9
  let h : ℝ → ℝ := λ x => 6 * x^2 + 6 * x + 2
  ∀ x : ℝ, f x + g x + h x = -4 * x^2 + 12 * x - 12 := by
sorry

end NUMINAMATH_CALUDE_sum_of_polynomials_l4188_418892


namespace NUMINAMATH_CALUDE_regular_icosahedron_edges_l4188_418874

/-- A regular icosahedron is a convex polyhedron with 20 faces, each of which is an equilateral triangle. -/
structure RegularIcosahedron :=
  (faces : Nat)
  (face_shape : String)
  (is_convex : Bool)
  (h_faces : faces = 20)
  (h_face_shape : face_shape = "equilateral triangle")
  (h_convex : is_convex = true)

/-- The number of edges in a regular icosahedron -/
def num_edges (i : RegularIcosahedron) : Nat := 30

/-- Theorem: A regular icosahedron has 30 edges -/
theorem regular_icosahedron_edges (i : RegularIcosahedron) : num_edges i = 30 := by
  sorry

end NUMINAMATH_CALUDE_regular_icosahedron_edges_l4188_418874


namespace NUMINAMATH_CALUDE_average_after_removal_l4188_418854

theorem average_after_removal (numbers : Finset ℕ) (sum : ℕ) :
  Finset.card numbers = 15 →
  sum / 15 = 100 →
  sum = Finset.sum numbers id →
  80 ∈ numbers →
  90 ∈ numbers →
  95 ∈ numbers →
  (sum - 80 - 90 - 95) / (Finset.card numbers - 3) = 1235 / 12 :=
by sorry

end NUMINAMATH_CALUDE_average_after_removal_l4188_418854


namespace NUMINAMATH_CALUDE_celine_change_l4188_418811

def laptop_base_price : ℚ := 600
def smartphone_base_price : ℚ := 400
def tablet_base_price : ℚ := 250
def headphone_base_price : ℚ := 100

def laptop_discount : ℚ := 0.15
def smartphone_increase : ℚ := 0.10
def tablet_discount : ℚ := 0.20

def sales_tax : ℚ := 0.06

def laptop_quantity : ℕ := 2
def smartphone_quantity : ℕ := 3
def tablet_quantity : ℕ := 4
def headphone_quantity : ℕ := 6

def celine_budget : ℚ := 6000

theorem celine_change : 
  let laptop_price := laptop_base_price * (1 - laptop_discount)
  let smartphone_price := smartphone_base_price * (1 + smartphone_increase)
  let tablet_price := tablet_base_price * (1 - tablet_discount)
  let headphone_price := headphone_base_price

  let total_before_tax := 
    laptop_price * laptop_quantity +
    smartphone_price * smartphone_quantity +
    tablet_price * tablet_quantity +
    headphone_price * headphone_quantity

  let total_with_tax := total_before_tax * (1 + sales_tax)

  celine_budget - total_with_tax = 2035.60 := by sorry

end NUMINAMATH_CALUDE_celine_change_l4188_418811


namespace NUMINAMATH_CALUDE_gumball_cost_l4188_418846

/-- Given that Melanie sells 4 gumballs for a total of 32 cents, prove that each gumball costs 8 cents. -/
theorem gumball_cost (num_gumballs : ℕ) (total_cents : ℕ) (h1 : num_gumballs = 4) (h2 : total_cents = 32) :
  total_cents / num_gumballs = 8 := by
  sorry

end NUMINAMATH_CALUDE_gumball_cost_l4188_418846


namespace NUMINAMATH_CALUDE_correct_average_weight_l4188_418808

def class_size : ℕ := 20
def initial_average : ℚ := 58.4
def misread_weight : ℕ := 56
def correct_weight : ℕ := 62

theorem correct_average_weight :
  let incorrect_total := initial_average * class_size
  let weight_difference := correct_weight - misread_weight
  let correct_total := incorrect_total + weight_difference
  (correct_total / class_size : ℚ) = 58.7 := by sorry

end NUMINAMATH_CALUDE_correct_average_weight_l4188_418808


namespace NUMINAMATH_CALUDE_car_price_calculation_l4188_418822

/-- Represents the price of a car given loan terms and payments -/
def carPrice (loanYears : ℕ) (downPayment : ℚ) (monthlyPayment : ℚ) : ℚ :=
  downPayment + (loanYears * 12 : ℕ) * monthlyPayment

/-- Theorem stating that given the specific loan terms, the car price is $20,000 -/
theorem car_price_calculation :
  let loanYears : ℕ := 5
  let downPayment : ℚ := 5000
  let monthlyPayment : ℚ := 250
  carPrice loanYears downPayment monthlyPayment = 20000 := by
  sorry

#eval carPrice 5 5000 250

end NUMINAMATH_CALUDE_car_price_calculation_l4188_418822


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l4188_418899

-- Define a geometric sequence
def geometric_sequence (a₁ : ℝ) (r : ℝ) (n : ℕ) : ℝ := a₁ * r ^ (n - 1)

-- Theorem statement
theorem geometric_sequence_common_ratio 
  (a₁ : ℝ) (r : ℝ) (h : ∀ n : ℕ, n > 0 → 
    geometric_sequence a₁ r n * geometric_sequence a₁ r (n + 1) = 16 ^ n) : 
  r = 4 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l4188_418899


namespace NUMINAMATH_CALUDE_rice_distributed_in_five_days_l4188_418834

/-- The amount of rice distributed in the first 5 days of dike construction --/
theorem rice_distributed_in_five_days : 
  let initial_workers : ℕ := 64
  let daily_increase : ℕ := 7
  let rice_per_worker : ℕ := 3
  let days : ℕ := 5
  let total_workers : ℕ := (days * (2 * initial_workers + (days - 1) * daily_increase)) / 2
  total_workers * rice_per_worker = 1170 := by
  sorry

end NUMINAMATH_CALUDE_rice_distributed_in_five_days_l4188_418834


namespace NUMINAMATH_CALUDE_hiker_distance_hiker_distance_proof_l4188_418862

/-- The straight-line distance a hiker travels after walking 8 miles east,
    turning 45 degrees north, and walking another 8 miles. -/
theorem hiker_distance : ℝ :=
  let initial_east_distance : ℝ := 8
  let turn_angle : ℝ := 45
  let second_walk_distance : ℝ := 8
  let final_distance : ℝ := 4 * Real.sqrt (6 + 4 * Real.sqrt 2)
  final_distance

/-- Proof that the hiker's final straight-line distance from the starting point
    is 4√(6 + 4√2) miles. -/
theorem hiker_distance_proof :
  hiker_distance = 4 * Real.sqrt (6 + 4 * Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_hiker_distance_hiker_distance_proof_l4188_418862


namespace NUMINAMATH_CALUDE_ice_cream_scoop_permutations_l4188_418840

theorem ice_cream_scoop_permutations :
  Nat.factorial 5 = 120 := by
  sorry

end NUMINAMATH_CALUDE_ice_cream_scoop_permutations_l4188_418840


namespace NUMINAMATH_CALUDE_equilateral_triangle_most_stable_l4188_418837

-- Define the shapes
inductive Shape
| EquilateralTriangle
| Square
| Parallelogram
| Trapezoid

-- Define stability as a function of shape properties
def stability (s : Shape) : ℝ :=
  match s with
  | Shape.EquilateralTriangle => 1
  | Shape.Square => 0.9
  | Shape.Parallelogram => 0.7
  | Shape.Trapezoid => 0.5

-- Define a predicate for being the most stable
def is_most_stable (s : Shape) : Prop :=
  ∀ t : Shape, stability s ≥ stability t

-- Theorem statement
theorem equilateral_triangle_most_stable :
  is_most_stable Shape.EquilateralTriangle :=
sorry

end NUMINAMATH_CALUDE_equilateral_triangle_most_stable_l4188_418837


namespace NUMINAMATH_CALUDE_function_inequality_l4188_418833

theorem function_inequality (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, (f x + y) * (f y + x) > 0 → f x + y = f y + x) :
  ∀ x y : ℝ, x > y → f x + y ≤ f y + x := by
sorry

end NUMINAMATH_CALUDE_function_inequality_l4188_418833


namespace NUMINAMATH_CALUDE_valid_numbers_l4188_418814

def is_valid_number (n : ℕ) : Prop :=
  100000 > n ∧ n ≥ 10000 ∧  -- five-digit number
  n % 72 = 0 ∧  -- divisible by 72
  (n.digits 10).count 1 = 3  -- exactly three digits are 1

theorem valid_numbers :
  {n : ℕ | is_valid_number n} = {41112, 14112, 11016, 11160} := by
  sorry

end NUMINAMATH_CALUDE_valid_numbers_l4188_418814


namespace NUMINAMATH_CALUDE_no_equal_sets_subset_condition_l4188_418828

-- Define the sets P and S
def P : Set ℝ := {x | x^2 - 8*x - 20 ≤ 0}
def S (m : ℝ) : Set ℝ := {x | 1 - m ≤ x ∧ x ≤ 1 + m}

-- Theorem 1: No m exists such that P = S(m)
theorem no_equal_sets : ¬∃ m : ℝ, P = S m := by sorry

-- Theorem 2: For all m ≥ 3, P ⊆ S(m)
theorem subset_condition (m : ℝ) (h : m ≥ 3) : P ⊆ S m := by sorry

end NUMINAMATH_CALUDE_no_equal_sets_subset_condition_l4188_418828


namespace NUMINAMATH_CALUDE_problem_statements_l4188_418816

theorem problem_statements :
  (¬ ∀ a b c : ℝ, a > b → a * c^2 > b * c^2) ∧
  (∀ a b : ℝ, a > |b| → a^2 > b^2) ∧
  (∀ a b : ℝ, a > b → a^3 > b^3) ∧
  (¬ ∀ a b : ℝ, |a| > b → a^2 > b^2) :=
by sorry

end NUMINAMATH_CALUDE_problem_statements_l4188_418816


namespace NUMINAMATH_CALUDE_modulus_z₂_l4188_418863

-- Define the complex numbers z₁ and z₂
variable (z₁ z₂ : ℂ)

-- State the conditions
axiom z₁_condition : (z₁ - 2) * Complex.I = 1 + Complex.I
axiom z₂_imag_part : z₂.im = 2
axiom product_real : (z₁ * z₂).im = 0

-- State the theorem
theorem modulus_z₂ : Complex.abs z₂ = 2 * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_modulus_z₂_l4188_418863


namespace NUMINAMATH_CALUDE_probability_two_aces_l4188_418859

/-- Represents a standard deck of cards -/
def StandardDeck : ℕ := 52

/-- Represents the number of aces in a standard deck -/
def AcesInDeck : ℕ := 4

/-- Represents the total number of cards in the mixed deck -/
def TotalCards : ℕ := 2 * StandardDeck

/-- Represents the total number of aces in the mixed deck -/
def TotalAces : ℕ := 2 * AcesInDeck

/-- The probability of drawing two aces consecutively from a mixed deck of 104 cards -/
theorem probability_two_aces (StandardDeck AcesInDeck TotalCards TotalAces : ℕ) 
  (h1 : TotalCards = 2 * StandardDeck)
  (h2 : TotalAces = 2 * AcesInDeck)
  (h3 : StandardDeck = 52)
  (h4 : AcesInDeck = 4) :
  (TotalAces : ℚ) / TotalCards * (TotalAces - 1) / (TotalCards - 1) = 7 / 1339 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_aces_l4188_418859


namespace NUMINAMATH_CALUDE_non_shaded_perimeter_l4188_418882

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Calculates the area of a rectangle -/
def Rectangle.area (r : Rectangle) : ℝ := r.width * r.height

/-- Calculates the perimeter of a rectangle -/
def Rectangle.perimeter (r : Rectangle) : ℝ := 2 * (r.width + r.height)

theorem non_shaded_perimeter (large : Rectangle) (small : Rectangle) :
  large.width = 12 ∧ 
  large.height = 12 ∧ 
  small.width = 6 ∧ 
  small.height = 4 ∧ 
  large.area - small.area = 144 →
  Rectangle.perimeter { width := large.width - small.width, height := large.height } +
  Rectangle.perimeter { width := large.width, height := large.height - small.height } = 28 := by
  sorry

end NUMINAMATH_CALUDE_non_shaded_perimeter_l4188_418882


namespace NUMINAMATH_CALUDE_eight_digit_number_theorem_l4188_418889

def is_coprime (a b : ℕ) : Prop := Nat.gcd a b = 1

def move_last_digit_to_first (n : ℕ) : ℕ :=
  let d := n % 10
  let r := n / 10
  d * 10^7 + r

theorem eight_digit_number_theorem (B : ℕ) (h1 : is_coprime B 36) (h2 : B > 7777777) :
  let A := move_last_digit_to_first B
  (∃ (B' : ℕ), is_coprime B' 36 ∧ B' > 7777777 ∧ move_last_digit_to_first B' > A) →
  (∃ (B' : ℕ), is_coprime B' 36 ∧ B' > 7777777 ∧ move_last_digit_to_first B' < A) →
  A = 99999998 ∨ A = 17777779 :=
by sorry

end NUMINAMATH_CALUDE_eight_digit_number_theorem_l4188_418889


namespace NUMINAMATH_CALUDE_sum_of_solutions_equals_six_l4188_418868

theorem sum_of_solutions_equals_six :
  ∃ (x₁ x₂ : ℝ), 
    (3 : ℝ) ^ (x₁^2 - 4*x₁ - 3) = 9 ^ (x₁ - 5) ∧
    (3 : ℝ) ^ (x₂^2 - 4*x₂ - 3) = 9 ^ (x₂ - 5) ∧
    x₁ ≠ x₂ ∧
    x₁ + x₂ = 6 ∧
    ∀ (x : ℝ), (3 : ℝ) ^ (x^2 - 4*x - 3) = 9 ^ (x - 5) → x = x₁ ∨ x = x₂ :=
by sorry

end NUMINAMATH_CALUDE_sum_of_solutions_equals_six_l4188_418868


namespace NUMINAMATH_CALUDE_dogwood_trees_planting_l4188_418848

theorem dogwood_trees_planting (initial_trees : ℕ) (planted_today : ℕ) (final_total : ℕ) 
  (h1 : initial_trees = 39)
  (h2 : planted_today = 41)
  (h3 : final_total = 100) :
  final_total - (initial_trees + planted_today) = 20 :=
by sorry

end NUMINAMATH_CALUDE_dogwood_trees_planting_l4188_418848


namespace NUMINAMATH_CALUDE_equivalent_conditions_l4188_418829

theorem equivalent_conditions (a b c : ℝ) :
  (1 / (a + b) + 1 / (b + c) = 2 / (c + a)) ↔ (2 * b^2 = a^2 + c^2) := by sorry

end NUMINAMATH_CALUDE_equivalent_conditions_l4188_418829


namespace NUMINAMATH_CALUDE_average_of_r_s_t_l4188_418836

theorem average_of_r_s_t (r s t : ℝ) (h : (5 / 2) * (r + s + t) = 25) :
  (r + s + t) / 3 = 10 / 3 := by
  sorry

end NUMINAMATH_CALUDE_average_of_r_s_t_l4188_418836


namespace NUMINAMATH_CALUDE_tech_students_formula_l4188_418827

/-- The number of students in technology elective courses -/
def tech_students (m : ℕ) : ℚ :=
  (1 / 3 : ℚ) * (m : ℚ) + 8

/-- The number of students in subject elective courses -/
def subject_students (m : ℕ) : ℕ := m

/-- The number of students in physical education and arts elective courses -/
def pe_arts_students (m : ℕ) : ℕ := m + 9

theorem tech_students_formula (m : ℕ) :
  tech_students m = (1 / 3 : ℚ) * (pe_arts_students m : ℚ) + 5 :=
by sorry

end NUMINAMATH_CALUDE_tech_students_formula_l4188_418827


namespace NUMINAMATH_CALUDE_product_95_105_l4188_418815

theorem product_95_105 : 95 * 105 = 9975 := by
  have h1 : 95 = 100 - 5 := by sorry
  have h2 : 105 = 100 + 5 := by sorry
  sorry

end NUMINAMATH_CALUDE_product_95_105_l4188_418815


namespace NUMINAMATH_CALUDE_saltwater_solution_volume_l4188_418883

theorem saltwater_solution_volume :
  -- Initial conditions
  ∀ x : ℝ,
  let initial_salt_volume := 0.20 * x
  let evaporated_volume := 0.25 * x
  let remaining_volume := x - evaporated_volume
  let added_water := 6
  let added_salt := 12
  let final_volume := remaining_volume + added_water + added_salt
  let final_salt_volume := initial_salt_volume + added_salt
  -- Final salt concentration condition
  final_salt_volume / final_volume = 1/3 →
  -- Conclusion
  x = 120 := by
  sorry

end NUMINAMATH_CALUDE_saltwater_solution_volume_l4188_418883


namespace NUMINAMATH_CALUDE_problem_proof_l4188_418821

theorem problem_proof (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + 2*b = 1) :
  (∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x + 2*y = 1 ∧ a*b ≤ x*y) ∧
  (∀ (x y : ℝ), x > 0 → y > 0 → x + 2*y = 1 → a*b ≤ 1/8) ∧
  ((1/b) + (b/a) ≥ 4) ∧
  (a^2 + b^2 ≥ 1/5) :=
sorry

end NUMINAMATH_CALUDE_problem_proof_l4188_418821


namespace NUMINAMATH_CALUDE_fixed_points_equality_l4188_418830

def StrictlyIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

def FixedPoints (f : ℝ → ℝ) : Set ℝ :=
  {x | f x = x}

theorem fixed_points_equality
  (f : ℝ → ℝ)
  (h_inj : Function.Injective f)
  (h_incr : StrictlyIncreasing f) :
  FixedPoints f = FixedPoints (f ∘ f) := by
  sorry

end NUMINAMATH_CALUDE_fixed_points_equality_l4188_418830


namespace NUMINAMATH_CALUDE_square_measurement_error_l4188_418805

theorem square_measurement_error (actual_side : ℝ) (measured_side : ℝ) 
  (h : measured_side ^ 2 = 1.0816 * actual_side ^ 2) : 
  (measured_side - actual_side) / actual_side = 0.04 := by
  sorry

end NUMINAMATH_CALUDE_square_measurement_error_l4188_418805


namespace NUMINAMATH_CALUDE_max_third_side_length_l4188_418881

theorem max_third_side_length (a b x : ℕ) (ha : a = 28) (hb : b = 47) : 
  (a + b > x ∧ a + x > b ∧ b + x > a) → x ≤ 74 :=
sorry

end NUMINAMATH_CALUDE_max_third_side_length_l4188_418881


namespace NUMINAMATH_CALUDE_apple_distribution_l4188_418857

theorem apple_distribution (x : ℕ) (total_apples : ℕ) : 
  (total_apples = 3 * x + 8) →
  (total_apples > 5 * (x - 1) ∧ total_apples < 5 * x) →
  ((x = 5 ∧ total_apples = 23) ∨ (x = 6 ∧ total_apples = 26)) :=
by sorry

end NUMINAMATH_CALUDE_apple_distribution_l4188_418857


namespace NUMINAMATH_CALUDE_locus_of_point_l4188_418866

/-- Given three lines in a plane not passing through the origin, prove the locus of a point P
    satisfying certain conditions. -/
theorem locus_of_point (a₁ a₂ a₃ b₁ b₂ b₃ c₁ c₂ c₃ : ℝ) :
  let l₁ : ℝ × ℝ → Prop := λ (x, y) ↦ a₁ * x + b₁ * y + c₁ = 0
  let l₂ : ℝ × ℝ → Prop := λ (x, y) ↦ a₂ * x + b₂ * y + c₂ = 0
  let l₃ : ℝ × ℝ → Prop := λ (x, y) ↦ a₃ * x + b₃ * y + c₃ = 0
  let origin : ℝ × ℝ := (0, 0)
  ∀ (l : Set (ℝ × ℝ)) (A B C : ℝ × ℝ),
    (∀ p ∈ l, ∃ t : ℝ, p = (t * (A.1 - origin.1), t * (A.2 - origin.2))) →
    l₁ A ∧ l₂ B ∧ l₃ C →
    A ∈ l ∧ B ∈ l ∧ C ∈ l →
    (∀ P ∈ l, P ≠ origin →
      let ρ₁ := Real.sqrt ((A.1 - origin.1)^2 + (A.2 - origin.2)^2)
      let ρ₂ := Real.sqrt ((B.1 - origin.1)^2 + (B.2 - origin.2)^2)
      let ρ₃ := Real.sqrt ((C.1 - origin.1)^2 + (C.2 - origin.2)^2)
      let ρ  := Real.sqrt ((P.1 - origin.1)^2 + (P.2 - origin.2)^2)
      1 / ρ₁ + 1 / ρ₂ + 1 / ρ₃ = 1 / ρ) →
    ∀ (x y : ℝ),
      (x, y) ∈ l ↔ (a₁ / c₁ + a₂ / c₂ + a₃ / c₃) * x + (b₁ / c₁ + b₂ / c₂ + b₃ / c₃) * y + 1 = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_locus_of_point_l4188_418866


namespace NUMINAMATH_CALUDE_max_concave_polygons_in_square_l4188_418887

/-- A concave polygon with sides parallel to a square's sides -/
structure ConcavePolygon where
  vertices : List (ℝ × ℝ)
  is_concave : Bool
  sides_parallel_to_square : Bool

/-- A square divided into concave polygons -/
structure DividedSquare where
  polygons : List ConcavePolygon
  no_parallel_translation : Bool

/-- The maximum number of equal concave polygons a square can be divided into -/
def max_concave_polygons : ℕ := 8

/-- Theorem stating the maximum number of equal concave polygons a square can be divided into -/
theorem max_concave_polygons_in_square :
  ∀ (d : DividedSquare),
    d.no_parallel_translation →
    (∀ p ∈ d.polygons, p.is_concave ∧ p.sides_parallel_to_square) →
    (List.length d.polygons ≤ max_concave_polygons) :=
by sorry

end NUMINAMATH_CALUDE_max_concave_polygons_in_square_l4188_418887


namespace NUMINAMATH_CALUDE_missing_number_proof_l4188_418810

def known_numbers : List ℝ := [13, 8, 13, 21, 23]

theorem missing_number_proof (mean : ℝ) (h_mean : mean = 14.2) :
  ∃ x : ℝ, (known_numbers.sum + x) / 6 = mean ∧ x = 7.2 := by
  sorry

end NUMINAMATH_CALUDE_missing_number_proof_l4188_418810


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l4188_418884

theorem regular_polygon_sides (n : ℕ) (h1 : n > 2) : 
  (∀ angle : ℝ, angle = 150 ∧ (n * angle : ℝ) = 180 * (n - 2 : ℝ)) → n = 12 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l4188_418884


namespace NUMINAMATH_CALUDE_parallel_transitive_l4188_418817

-- Define the type for lines
def Line : Type := ℝ → ℝ → Prop

-- Define the parallel relation
def parallel (l1 l2 : Line) : Prop := sorry

-- Theorem statement
theorem parallel_transitive (a b c : Line) :
  parallel a b → parallel b c → parallel a c := by sorry

end NUMINAMATH_CALUDE_parallel_transitive_l4188_418817


namespace NUMINAMATH_CALUDE_sum_b_plus_c_l4188_418853

theorem sum_b_plus_c (a b c d : ℝ) 
  (h1 : a + b = 12)
  (h2 : c + d = 3)
  (h3 : a + d = 6) :
  b + c = 9 := by
  sorry

end NUMINAMATH_CALUDE_sum_b_plus_c_l4188_418853


namespace NUMINAMATH_CALUDE_regular_hexagon_side_length_l4188_418858

/-- A regular hexagon with opposite sides 18 inches apart has side length 12√3 inches -/
theorem regular_hexagon_side_length (h : RegularHexagon) 
  (opposite_sides_distance : ℝ) (side_length : ℝ) : 
  opposite_sides_distance = 18 → side_length = 12 * Real.sqrt 3 := by
  sorry

#check regular_hexagon_side_length

end NUMINAMATH_CALUDE_regular_hexagon_side_length_l4188_418858


namespace NUMINAMATH_CALUDE_max_value_circle_center_l4188_418847

/-- Circle C with center (a,b) and radius 1 -/
def Circle (a b : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - a)^2 + (p.2 - b)^2 = 1}

/-- Region Ω -/
def Ω : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 + p.2 - 7 ≤ 0 ∧ p.1 - p.2 + 3 ≥ 0 ∧ p.2 ≥ 0}

/-- The maximum value of a^2 + b^2 given the conditions -/
theorem max_value_circle_center (a b : ℝ) :
  (a, b) ∈ Ω →
  b = 1 →
  (∃ (x : ℝ), (x, 0) ∈ Circle a b) →
  a^2 + b^2 ≤ 37 :=
sorry

end NUMINAMATH_CALUDE_max_value_circle_center_l4188_418847


namespace NUMINAMATH_CALUDE_average_of_eleven_numbers_l4188_418844

theorem average_of_eleven_numbers (first_six_avg : ℝ) (last_six_avg : ℝ) (sixth_number : ℝ) :
  first_six_avg = 88 →
  last_six_avg = 65 →
  sixth_number = 258 →
  (6 * first_six_avg + 6 * last_six_avg - sixth_number) / 11 = 60 := by
  sorry

end NUMINAMATH_CALUDE_average_of_eleven_numbers_l4188_418844


namespace NUMINAMATH_CALUDE_matches_played_eq_teams_minus_one_l4188_418875

/-- Represents an elimination tournament. -/
structure EliminationTournament where
  num_teams : ℕ
  no_replays : Bool

/-- The number of matches played in an elimination tournament. -/
def matches_played (t : EliminationTournament) : ℕ := sorry

/-- Theorem stating that in an elimination tournament with no replays, 
    the number of matches played is one less than the number of teams. -/
theorem matches_played_eq_teams_minus_one (t : EliminationTournament) 
  (h : t.no_replays = true) : matches_played t = t.num_teams - 1 := by sorry

end NUMINAMATH_CALUDE_matches_played_eq_teams_minus_one_l4188_418875


namespace NUMINAMATH_CALUDE_expression_equality_l4188_418831

theorem expression_equality : 
  (((3 + 5 + 7) / (2 + 4 + 6)) * 2 - ((2 + 4 + 6) / (3 + 5 + 7))) = 17 / 10 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l4188_418831


namespace NUMINAMATH_CALUDE_perpendicular_tangents_intersection_l4188_418841

/-- The y-coordinate of the intersection point of perpendicular tangents on y = 4x^2 -/
theorem perpendicular_tangents_intersection (a b : ℝ) : 
  (∃ A B : ℝ × ℝ, 
    A.2 = 4 * A.1^2 ∧ 
    B.2 = 4 * B.1^2 ∧ 
    A.1 = a ∧ 
    B.1 = b ∧ 
    (8 * a) * (8 * b) = -1) → 
  ∃ P : ℝ × ℝ, 
    (P.1 = (a + b) / 2) ∧ 
    (P.2 = -2) := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_tangents_intersection_l4188_418841


namespace NUMINAMATH_CALUDE_sum_of_common_ratios_of_geometric_sequences_l4188_418865

theorem sum_of_common_ratios_of_geometric_sequences 
  (k a₂ a₃ b₂ b₃ p r : ℝ) 
  (h1 : k ≠ 0)
  (h2 : p ≠ 1)
  (h3 : r ≠ 1)
  (h4 : p ≠ r)
  (h5 : a₂ = k * p)
  (h6 : a₃ = k * p^2)
  (h7 : b₂ = k * r)
  (h8 : b₃ = k * r^2)
  (h9 : a₃ - b₃ = 3 * (a₂ - b₂)) :
  p + r = 3 := by
sorry

end NUMINAMATH_CALUDE_sum_of_common_ratios_of_geometric_sequences_l4188_418865


namespace NUMINAMATH_CALUDE_bridge_building_time_l4188_418842

/-- If a crew of m workers can build a bridge in d days, then a crew of 2m workers can build the same bridge in d/2 days. -/
theorem bridge_building_time (m d : ℝ) (h1 : m > 0) (h2 : d > 0) :
  let initial_crew := m
  let initial_time := d
  let new_crew := 2 * m
  let new_time := d / 2
  initial_crew * initial_time = new_crew * new_time :=
by sorry

end NUMINAMATH_CALUDE_bridge_building_time_l4188_418842


namespace NUMINAMATH_CALUDE_quarters_remaining_l4188_418804

-- Define the initial number of quarters
def initial_quarters : ℕ := 375

-- Define the cost of the dress in cents
def dress_cost_cents : ℕ := 4263

-- Define the value of a quarter in cents
def quarter_value_cents : ℕ := 25

-- Theorem to prove
theorem quarters_remaining :
  initial_quarters - (dress_cost_cents / quarter_value_cents) = 205 := by
  sorry

end NUMINAMATH_CALUDE_quarters_remaining_l4188_418804


namespace NUMINAMATH_CALUDE_solutions_of_quadratic_l4188_418864

theorem solutions_of_quadratic (x : ℝ) : x^2 = 16*x ↔ x = 0 ∨ x = 16 := by
  sorry

end NUMINAMATH_CALUDE_solutions_of_quadratic_l4188_418864


namespace NUMINAMATH_CALUDE_grayson_collection_l4188_418869

/-- The number of cookies in each box -/
def cookies_per_box : ℕ := 48

/-- The number of boxes Abigail collected -/
def abigail_boxes : ℕ := 2

/-- The number of boxes Olivia collected -/
def olivia_boxes : ℕ := 3

/-- The total number of cookies collected -/
def total_cookies : ℕ := 276

/-- The fraction of a box Grayson collected -/
def grayson_fraction : ℚ := 3/4

theorem grayson_collection :
  grayson_fraction * cookies_per_box = 
    total_cookies - (abigail_boxes + olivia_boxes) * cookies_per_box :=
by sorry

end NUMINAMATH_CALUDE_grayson_collection_l4188_418869


namespace NUMINAMATH_CALUDE_trapezoid_area_l4188_418895

-- Define the rectangle ABCD
def Rectangle (A B C D : Point) : Prop := sorry

-- Define the trapezoid EFBA
def Trapezoid (E F B A : Point) : Prop := sorry

-- Define the area function
def area (shape : Set Point) : ℝ := sorry

-- Define the points
variable (A B C D E F : Point)

-- State the theorem
theorem trapezoid_area 
  (h1 : Rectangle A B C D) 
  (h2 : area {A, B, C, D} = 20) 
  (h3 : Trapezoid E F B A) : 
  area {E, F, B, A} = 14 := by sorry

end NUMINAMATH_CALUDE_trapezoid_area_l4188_418895


namespace NUMINAMATH_CALUDE_geometric_series_ratio_l4188_418845

theorem geometric_series_ratio (a r : ℝ) (h : r ≠ 1) :
  (a / (1 - r) = 64 * (a * r^4) / (1 - r)) → r = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_series_ratio_l4188_418845


namespace NUMINAMATH_CALUDE_expression_simplification_l4188_418852

theorem expression_simplification (a : ℝ) (h : a = 3 + Real.sqrt 3) :
  (1 - 1 / (a - 2)) / ((a^2 - 6*a + 9) / (a^2 - 2*a)) = Real.sqrt 3 + 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l4188_418852


namespace NUMINAMATH_CALUDE_marble_selection_ways_l4188_418888

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The total number of marbles -/
def total_marbles : ℕ := 15

/-- The number of marbles to be chosen -/
def marbles_to_choose : ℕ := 5

/-- The number of specific colored marbles (red + green + blue) -/
def specific_colored_marbles : ℕ := 6

/-- The number of ways to choose 2 marbles from the specific colored ones -/
def ways_to_choose_specific : ℕ := 9

/-- The number of remaining marbles after removing the specific colored ones -/
def remaining_marbles : ℕ := total_marbles - specific_colored_marbles

theorem marble_selection_ways :
  ways_to_choose_specific * choose remaining_marbles (marbles_to_choose - 2) = 1980 :=
sorry

end NUMINAMATH_CALUDE_marble_selection_ways_l4188_418888


namespace NUMINAMATH_CALUDE_min_sum_of_reciprocal_sum_l4188_418806

theorem min_sum_of_reciprocal_sum (x y : ℕ+) (h1 : x ≠ y) (h2 : (1 : ℚ) / x + (1 : ℚ) / y = (1 : ℚ) / 8) :
  ∃ (a b : ℕ+), a ≠ b ∧ (1 : ℚ) / a + (1 : ℚ) / b = (1 : ℚ) / 8 ∧ (a : ℕ) + b = 36 ∧
  ∀ (c d : ℕ+), c ≠ d → (1 : ℚ) / c + (1 : ℚ) / d = (1 : ℚ) / 8 → (c : ℕ) + d ≥ 36 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_of_reciprocal_sum_l4188_418806


namespace NUMINAMATH_CALUDE_least_sum_of_bases_l4188_418809

theorem least_sum_of_bases (c d : ℕ+) (h : 3 * c.val + 8 = 8 * d.val + 3) :
  ∃ (c' d' : ℕ+), 3 * c'.val + 8 = 8 * d'.val + 3 ∧ c'.val + d'.val ≤ c.val + d.val ∧ c'.val + d'.val = 13 :=
sorry

end NUMINAMATH_CALUDE_least_sum_of_bases_l4188_418809


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l4188_418879

theorem arithmetic_sequence_common_difference
  (a₁ : ℚ)    -- first term
  (aₙ : ℚ)    -- last term
  (S  : ℚ)    -- sum of all terms
  (h₁ : a₁ = 3)
  (h₂ : aₙ = 34)
  (h₃ : S = 222) :
  ∃ (n : ℕ) (d : ℚ), n > 1 ∧ d = 31/11 ∧ 
    aₙ = a₁ + (n - 1) * d ∧
    S = n * (a₁ + aₙ) / 2 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l4188_418879


namespace NUMINAMATH_CALUDE_incenter_vector_ratio_l4188_418835

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the incenter of a triangle
def incenter (t : Triangle) : ℝ × ℝ := sorry

-- Define vector operations
def vec_add (v w : ℝ × ℝ) : ℝ × ℝ := (v.1 + w.1, v.2 + w.2)
def vec_scale (a : ℝ) (v : ℝ × ℝ) : ℝ × ℝ := (a * v.1, a * v.2)

-- Main theorem
theorem incenter_vector_ratio (t : Triangle) 
  (h1 : dist t.A t.B = 6)
  (h2 : dist t.B t.C = 7)
  (h3 : dist t.A t.C = 4)
  (O : ℝ × ℝ)
  (hO : O = incenter t)
  (p q : ℝ)
  (h4 : vec_add (vec_scale (-1) O) t.A = vec_add (vec_scale p (vec_add (vec_scale (-1) t.A) t.B)) 
                                                 (vec_scale q (vec_add (vec_scale (-1) t.A) t.C)))
  : p / q = 2 / 3 := by
  sorry


end NUMINAMATH_CALUDE_incenter_vector_ratio_l4188_418835


namespace NUMINAMATH_CALUDE_sugar_left_in_grams_l4188_418896

/-- The amount of sugar Pamela bought in ounces -/
def sugar_bought : ℝ := 9.8

/-- The amount of sugar Pamela spilled in ounces -/
def sugar_spilled : ℝ := 5.2

/-- The conversion factor from ounces to grams -/
def oz_to_g : ℝ := 28.35

/-- Theorem stating the amount of sugar Pamela has left in grams -/
theorem sugar_left_in_grams : 
  (sugar_bought - sugar_spilled) * oz_to_g = 130.41 := by
  sorry

end NUMINAMATH_CALUDE_sugar_left_in_grams_l4188_418896


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l4188_418876

theorem complex_modulus_problem (a : ℝ) (h1 : a > 0) (h2 : Complex.abs (a + Complex.I) = 2) :
  a = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l4188_418876


namespace NUMINAMATH_CALUDE_root_product_theorem_l4188_418850

theorem root_product_theorem (a b m p q : ℝ) : 
  (a^2 - m*a + 3 = 0) → 
  (b^2 - m*b + 3 = 0) → 
  ((a + 1/b)^2 - p*(a + 1/b) + q = 0) → 
  ((b + 1/a)^2 - p*(b + 1/a) + q = 0) → 
  q = 16/3 := by
sorry

end NUMINAMATH_CALUDE_root_product_theorem_l4188_418850


namespace NUMINAMATH_CALUDE_xy_range_l4188_418802

theorem xy_range (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : x + 1/x + y + 1/y = 5) : 1/4 ≤ x*y ∧ x*y ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_xy_range_l4188_418802


namespace NUMINAMATH_CALUDE_triangle_inequality_l4188_418813

def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

theorem triangle_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  can_form_triangle 2 6 6 ∧
  ¬can_form_triangle 2 6 2 ∧
  ¬can_form_triangle 2 6 4 ∧
  ¬can_form_triangle 2 6 9 :=
by sorry

end NUMINAMATH_CALUDE_triangle_inequality_l4188_418813


namespace NUMINAMATH_CALUDE_min_value_theorem_l4188_418820

theorem min_value_theorem (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + 3 * b = 1) :
  (1 / a + 3 / b) ≥ 16 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ a₀ + 3 * b₀ = 1 ∧ 1 / a₀ + 3 / b₀ = 16 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l4188_418820


namespace NUMINAMATH_CALUDE_candy_necklace_packs_opened_l4188_418886

/-- Proves the number of candy necklace packs Emily opened for her classmates -/
theorem candy_necklace_packs_opened
  (total_packs : ℕ)
  (necklaces_per_pack : ℕ)
  (necklaces_left : ℕ)
  (h1 : total_packs = 9)
  (h2 : necklaces_per_pack = 8)
  (h3 : necklaces_left = 40) :
  (total_packs * necklaces_per_pack - necklaces_left) / necklaces_per_pack = 4 :=
by sorry

end NUMINAMATH_CALUDE_candy_necklace_packs_opened_l4188_418886


namespace NUMINAMATH_CALUDE_choir_arrangement_l4188_418843

theorem choir_arrangement (n : ℕ) : 
  (∃ k : ℕ, n = 9 * k) ∧ 
  (∃ k : ℕ, n = 10 * k) ∧ 
  (∃ k : ℕ, n = 11 * k) ↔ 
  n ≥ 990 ∧ n % 990 = 0 :=
by sorry

end NUMINAMATH_CALUDE_choir_arrangement_l4188_418843


namespace NUMINAMATH_CALUDE_max_cubes_in_box_l4188_418812

theorem max_cubes_in_box (box_length box_width box_height cube_volume : ℕ) 
  (h1 : box_length = 8)
  (h2 : box_width = 9)
  (h3 : box_height = 12)
  (h4 : cube_volume = 27) :
  (box_length * box_width * box_height) / cube_volume = 32 := by
  sorry

#check max_cubes_in_box

end NUMINAMATH_CALUDE_max_cubes_in_box_l4188_418812


namespace NUMINAMATH_CALUDE_five_squared_minus_nine_over_five_minus_three_equals_eight_l4188_418838

theorem five_squared_minus_nine_over_five_minus_three_equals_eight :
  (5^2 - 9) / (5 - 3) = 8 := by
  sorry

end NUMINAMATH_CALUDE_five_squared_minus_nine_over_five_minus_three_equals_eight_l4188_418838


namespace NUMINAMATH_CALUDE_largest_gcd_of_sum_1005_l4188_418826

theorem largest_gcd_of_sum_1005 :
  ∃ (a b : ℕ+), a + b = 1005 ∧
  ∀ (c d : ℕ+), c + d = 1005 → Nat.gcd c.val d.val ≤ Nat.gcd a.val b.val ∧
  Nat.gcd a.val b.val = 335 :=
sorry

end NUMINAMATH_CALUDE_largest_gcd_of_sum_1005_l4188_418826


namespace NUMINAMATH_CALUDE_snake_paint_calculation_l4188_418818

theorem snake_paint_calculation (cube_paint : ℕ) (snake_length : ℕ) (segment_length : ℕ) 
  (segment_paint : ℕ) (end_paint : ℕ) : 
  cube_paint = 60 → 
  snake_length = 2016 → 
  segment_length = 6 → 
  segment_paint = 240 → 
  end_paint = 20 → 
  (snake_length / segment_length * segment_paint + end_paint : ℕ) = 80660 := by
  sorry

end NUMINAMATH_CALUDE_snake_paint_calculation_l4188_418818


namespace NUMINAMATH_CALUDE_negation_existence_false_l4188_418800

theorem negation_existence_false : ¬(∀ x : ℝ, 2^x + x^2 > 1) := by
  sorry

end NUMINAMATH_CALUDE_negation_existence_false_l4188_418800


namespace NUMINAMATH_CALUDE_min_value_expression_l4188_418839

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  ∃ (min : ℝ), min = 3 * Real.sqrt (5 / 13) ∧
  ∀ (x : ℝ), x = (Real.sqrt ((a^2 + 2*b^2) * (4*a^2 + b^2))) / (a * b) → x ≥ min :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_l4188_418839


namespace NUMINAMATH_CALUDE_smallest_number_divisible_by_all_l4188_418894

def is_divisible_by_all (n : ℕ) : Prop :=
  (n + 3) % 9 = 0 ∧ (n + 3) % 70 = 0 ∧ (n + 3) % 25 = 0 ∧ (n + 3) % 21 = 0

theorem smallest_number_divisible_by_all : 
  ∀ n : ℕ, n < 3147 → ¬(is_divisible_by_all n) ∧ is_divisible_by_all 3147 := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_divisible_by_all_l4188_418894


namespace NUMINAMATH_CALUDE_gel_pen_price_ratio_l4188_418861

variables (x y : ℕ) (b g : ℝ)

def total_cost := x * b + y * g

theorem gel_pen_price_ratio :
  (∀ (x y : ℕ) (b g : ℝ),
    (x + y) * g = 4 * (x * b + y * g) ∧
    (x + y) * b = (1 / 2) * (x * b + y * g)) →
  g = 8 * b :=
sorry

end NUMINAMATH_CALUDE_gel_pen_price_ratio_l4188_418861


namespace NUMINAMATH_CALUDE_largest_divisor_of_difference_of_squares_l4188_418824

theorem largest_divisor_of_difference_of_squares (m n : ℕ) :
  (∃ k l : ℕ, m = 2 * k ∧ n = 2 * l) →  -- m and n are even
  n < m →  -- n is less than m
  (∃ d : ℕ, d > 0 ∧ 
    (∀ a b : ℕ, (∃ i j : ℕ, a = 2 * i ∧ b = 2 * j) → b < a → 
      d ∣ (a^2 - b^2)) ∧
    (∀ e : ℕ, e > d → 
      ∃ x y : ℕ, (∃ p q : ℕ, x = 2 * p ∧ y = 2 * q) ∧ y < x ∧ ¬(e ∣ (x^2 - y^2)))) →
  (∃ d : ℕ, d = 16 ∧ d > 0 ∧ 
    (∀ a b : ℕ, (∃ i j : ℕ, a = 2 * i ∧ b = 2 * j) → b < a → 
      d ∣ (a^2 - b^2)) ∧
    (∀ e : ℕ, e > d → 
      ∃ x y : ℕ, (∃ p q : ℕ, x = 2 * p ∧ y = 2 * q) ∧ y < x ∧ ¬(e ∣ (x^2 - y^2)))) :=
by sorry


end NUMINAMATH_CALUDE_largest_divisor_of_difference_of_squares_l4188_418824


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_l4188_418873

theorem absolute_value_equation_solution :
  ∃! x : ℝ, |x - 3| = 5 - x :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_l4188_418873


namespace NUMINAMATH_CALUDE_product_equals_243_l4188_418893

theorem product_equals_243 : 
  (1 / 3) * 9 * (1 / 27) * 81 * (1 / 243) * 729 * (1 / 2187) * 6561 * (1 / 19683) * 59049 = 243 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_243_l4188_418893


namespace NUMINAMATH_CALUDE_leg_length_in_special_right_isosceles_triangle_l4188_418897

/-- Represents a 45-45-90 triangle -/
structure RightIsoscelesTriangle where
  /-- The length of the hypotenuse -/
  hypotenuse : ℝ
  /-- The hypotenuse is positive -/
  hypotenuse_pos : hypotenuse > 0

/-- Theorem: In a 45-45-90 triangle with hypotenuse 12√2, the length of a leg is 12 -/
theorem leg_length_in_special_right_isosceles_triangle 
  (triangle : RightIsoscelesTriangle) 
  (h : triangle.hypotenuse = 12 * Real.sqrt 2) : 
  triangle.hypotenuse / Real.sqrt 2 = 12 := by
  sorry

#check leg_length_in_special_right_isosceles_triangle

end NUMINAMATH_CALUDE_leg_length_in_special_right_isosceles_triangle_l4188_418897


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l4188_418878

/-- The eccentricity of a hyperbola with equation x²/a² - y²/b² = 1 is √(a² + b²)/a -/
theorem hyperbola_eccentricity (a b : ℝ) (h : a > 0 ∧ b > 0) :
  let e := Real.sqrt (a^2 + b^2) / a
  (∀ x y : ℝ, x^2/a^2 - y^2/b^2 = 1 → e = Real.sqrt 13 / 3) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l4188_418878


namespace NUMINAMATH_CALUDE_division_sum_equality_l4188_418807

theorem division_sum_equality : 3752 / (39 * 2) + 5030 / (39 * 10) = 61 := by
  sorry

end NUMINAMATH_CALUDE_division_sum_equality_l4188_418807


namespace NUMINAMATH_CALUDE_base_seven_addition_l4188_418851

/-- Given an addition problem in base 7: 5XY₇ + 62₇ = 64X₇, prove that X + Y = 8 in base 10 -/
theorem base_seven_addition (X Y : ℕ) : 
  (5 * 7^2 + X * 7 + Y) + (6 * 7 + 2) = 6 * 7^2 + 4 * 7 + X → X + Y = 8 := by
sorry

end NUMINAMATH_CALUDE_base_seven_addition_l4188_418851


namespace NUMINAMATH_CALUDE_willey_farm_problem_l4188_418871

/-- The Willey Farm Collective Problem -/
theorem willey_farm_problem (total_land : ℝ) (corn_cost : ℝ) (wheat_cost : ℝ) (available_capital : ℝ)
  (h1 : total_land = 4500)
  (h2 : corn_cost = 42)
  (h3 : wheat_cost = 35)
  (h4 : available_capital = 165200) :
  ∃ (wheat_acres : ℝ), wheat_acres = 3400 ∧
    wheat_acres ≥ 0 ∧
    wheat_acres ≤ total_land ∧
    ∃ (corn_acres : ℝ), corn_acres ≥ 0 ∧
      corn_acres + wheat_acres = total_land ∧
      corn_cost * corn_acres + wheat_cost * wheat_acres = available_capital :=
by sorry

end NUMINAMATH_CALUDE_willey_farm_problem_l4188_418871


namespace NUMINAMATH_CALUDE_unshaded_perimeter_l4188_418849

/-- Given an L-shaped region formed by two adjoining rectangles with the following properties:
  - The total area of the L-shape is 240 square inches
  - The area of the shaded region is 65 square inches
  - The total length of the combined rectangles is 20 inches
  - The total width at the widest point is 12 inches
  - The width of the inner shaded rectangle is 5 inches
  - All rectangles contain right angles

  This theorem proves that the perimeter of the unshaded region is 64 inches. -/
theorem unshaded_perimeter (total_area : ℝ) (shaded_area : ℝ) (total_length : ℝ) (total_width : ℝ) (inner_width : ℝ)
  (h_total_area : total_area = 240)
  (h_shaded_area : shaded_area = 65)
  (h_total_length : total_length = 20)
  (h_total_width : total_width = 12)
  (h_inner_width : inner_width = 5) :
  2 * ((total_width - inner_width) + (total_area - shaded_area) / (total_width - inner_width)) = 64 :=
by sorry

end NUMINAMATH_CALUDE_unshaded_perimeter_l4188_418849


namespace NUMINAMATH_CALUDE_larry_win_probability_l4188_418801

/-- Probability that Larry knocks off the bottle -/
def larry_prob : ℚ := 2/3

/-- Probability that Julius knocks off the bottle -/
def julius_prob : ℚ := 1/3

/-- Total number of throws allowed in the game -/
def total_throws : ℕ := 6

/-- Larry throws first -/
def larry_first : Prop := True

/-- Probability that Larry wins the game -/
def larry_wins_prob : ℚ := 170/243

theorem larry_win_probability :
  larry_prob = 2/3 →
  julius_prob = 1/3 →
  total_throws = 6 →
  larry_first →
  larry_wins_prob = 170/243 :=
by
  sorry

end NUMINAMATH_CALUDE_larry_win_probability_l4188_418801


namespace NUMINAMATH_CALUDE_nathan_harvest_earnings_is_186_l4188_418890

/-- Calculates the total earnings from Nathan's harvest --/
def nathan_harvest_earnings : ℕ :=
  let strawberry_plants : ℕ := 5
  let tomato_plants : ℕ := 7
  let strawberries_per_plant : ℕ := 14
  let tomatoes_per_plant : ℕ := 16
  let fruits_per_basket : ℕ := 7
  let price_strawberry_basket : ℕ := 9
  let price_tomato_basket : ℕ := 6

  let total_strawberries : ℕ := strawberry_plants * strawberries_per_plant
  let total_tomatoes : ℕ := tomato_plants * tomatoes_per_plant

  let strawberry_baskets : ℕ := total_strawberries / fruits_per_basket
  let tomato_baskets : ℕ := total_tomatoes / fruits_per_basket

  let earnings_strawberries : ℕ := strawberry_baskets * price_strawberry_basket
  let earnings_tomatoes : ℕ := tomato_baskets * price_tomato_basket

  earnings_strawberries + earnings_tomatoes

theorem nathan_harvest_earnings_is_186 : nathan_harvest_earnings = 186 := by
  sorry

end NUMINAMATH_CALUDE_nathan_harvest_earnings_is_186_l4188_418890


namespace NUMINAMATH_CALUDE_toy_donation_difference_l4188_418872

def leila_bags : ℕ := 2
def leila_toys_per_bag : ℕ := 25
def mohamed_bags : ℕ := 3
def mohamed_toys_per_bag : ℕ := 19

theorem toy_donation_difference : 
  mohamed_bags * mohamed_toys_per_bag - leila_bags * leila_toys_per_bag = 7 := by
  sorry

end NUMINAMATH_CALUDE_toy_donation_difference_l4188_418872


namespace NUMINAMATH_CALUDE_rectangle_breadth_l4188_418867

theorem rectangle_breadth (square_area : ℝ) (rectangle_area : ℝ) :
  square_area = 625 →
  rectangle_area = 100 →
  let square_side := Real.sqrt square_area
  let circle_radius := square_side
  let rectangle_length := (2 / 5) * circle_radius
  rectangle_area = rectangle_length * (10 : ℝ) :=
by
  sorry

end NUMINAMATH_CALUDE_rectangle_breadth_l4188_418867


namespace NUMINAMATH_CALUDE_mark_bench_press_l4188_418819

-- Define the given conditions
def dave_weight : ℝ := 175
def dave_bench_press_multiplier : ℝ := 3
def craig_bench_press_percentage : ℝ := 0.20
def emma_bench_press_percentage : ℝ := 0.75
def emma_bench_press_increase : ℝ := 15
def john_bench_press_multiplier : ℝ := 2
def mark_bench_press_difference : ℝ := 50

-- Define the theorem
theorem mark_bench_press :
  let dave_bench_press := dave_weight * dave_bench_press_multiplier
  let craig_bench_press := craig_bench_press_percentage * dave_bench_press
  let emma_bench_press := emma_bench_press_percentage * dave_bench_press + emma_bench_press_increase
  let combined_craig_emma := craig_bench_press + emma_bench_press
  let mark_bench_press := combined_craig_emma - mark_bench_press_difference
  mark_bench_press = 463.75 := by
  sorry

end NUMINAMATH_CALUDE_mark_bench_press_l4188_418819


namespace NUMINAMATH_CALUDE_inequality_solution_l4188_418823

theorem inequality_solution (x : ℝ) :
  x > -1 ∧ x ≠ 0 →
  (x^2 / ((x + 1 - Real.sqrt (x + 1))^2) < (x^2 + 3*x + 18) / (x + 1)^2) ↔
  (x > -1 ∧ x < 0) ∨ (x > 0 ∧ x < 3) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l4188_418823


namespace NUMINAMATH_CALUDE_inverse_proportion_l4188_418856

/-- Given that x is inversely proportional to y, prove that if x = 5 when y = 15, 
    then x = 5/3 when y = 45 -/
theorem inverse_proportion (x y : ℝ) (k : ℝ) (h1 : x * y = k) 
    (h2 : 5 * 15 = k) : 
  5 / 3 * 45 = k := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_l4188_418856


namespace NUMINAMATH_CALUDE_car_speed_problem_l4188_418898

theorem car_speed_problem (total_time : ℝ) (initial_time : ℝ) (initial_speed : ℝ) (average_speed : ℝ) 
  (h1 : total_time = 24)
  (h2 : initial_time = 4)
  (h3 : initial_speed = 35)
  (h4 : average_speed = 50) :
  let remaining_time := total_time - initial_time
  let total_distance := average_speed * total_time
  let initial_distance := initial_speed * initial_time
  let remaining_distance := total_distance - initial_distance
  remaining_distance / remaining_time = 53 := by
sorry

end NUMINAMATH_CALUDE_car_speed_problem_l4188_418898


namespace NUMINAMATH_CALUDE_range_of_a_l4188_418855

def proposition_p (a : ℝ) : Prop :=
  ∀ x ∈ Set.Icc 1 2, x^2 - a ≥ 0

def proposition_q (a : ℝ) : Prop :=
  ∃ x : ℝ, x^2 + 2*a*x + 2 - a = 0

theorem range_of_a : 
  ∀ a : ℝ, proposition_p a ∧ proposition_q a → a ≤ -2 ∨ a = 1 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l4188_418855


namespace NUMINAMATH_CALUDE_smallest_number_divisible_l4188_418803

theorem smallest_number_divisible (n : ℕ) : 
  (∀ m : ℕ, m < 32127 → ¬(510 ∣ (m + 3) ∧ 4590 ∣ (m + 3) ∧ 105 ∣ (m + 3))) ∧
  (510 ∣ (32127 + 3) ∧ 4590 ∣ (32127 + 3) ∧ 105 ∣ (32127 + 3)) :=
sorry

end NUMINAMATH_CALUDE_smallest_number_divisible_l4188_418803


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_inequality_l4188_418880

theorem negation_of_existence (p : ℝ → Prop) : 
  (¬ ∃ x, p x) ↔ (∀ x, ¬ p x) :=
by sorry

theorem negation_of_quadratic_inequality :
  (¬ ∃ x : ℝ, x^2 - 2*x + 1 < 0) ↔ (∀ x : ℝ, x^2 - 2*x + 1 ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_inequality_l4188_418880


namespace NUMINAMATH_CALUDE_betting_game_result_l4188_418891

theorem betting_game_result (initial_amount : ℚ) (num_bets num_wins num_losses : ℕ) 
  (h1 : initial_amount = 64)
  (h2 : num_bets = 6)
  (h3 : num_wins = 3)
  (h4 : num_losses = 3)
  (h5 : num_wins + num_losses = num_bets) :
  let final_amount := initial_amount * (3/2)^num_wins * (1/2)^num_losses
  final_amount = 27 ∧ initial_amount - final_amount = 37 := by
  sorry

#eval (64 : ℚ) * (3/2)^3 * (1/2)^3

end NUMINAMATH_CALUDE_betting_game_result_l4188_418891


namespace NUMINAMATH_CALUDE_arithmetic_sequence_middle_term_l4188_418877

/-- An arithmetic sequence with five terms -/
structure ArithmeticSequence :=
  (a b c : ℝ)
  (first : ℝ := 17)
  (last : ℝ := 41)

/-- The property that the sequence is arithmetic -/
def is_arithmetic (seq : ArithmeticSequence) : Prop :=
  ∃ d : ℝ, seq.last - seq.c = d ∧ seq.c - seq.b = d ∧ seq.b - seq.a = d ∧ seq.a - seq.first = d

theorem arithmetic_sequence_middle_term (seq : ArithmeticSequence) 
  (h : is_arithmetic seq) : seq.b = 29 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_middle_term_l4188_418877


namespace NUMINAMATH_CALUDE_exists_line_and_circle_through_origin_l4188_418860

-- Define the ellipse
def is_on_ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

-- Define a line passing through (0, -2)
def line_through_point (k : ℝ) (x y : ℝ) : Prop := y = k * x - 2

-- Define two points on the intersection of the line and the ellipse
def intersection_points (k : ℝ) (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  is_on_ellipse x₁ y₁ ∧ is_on_ellipse x₂ y₂ ∧
  line_through_point k x₁ y₁ ∧ line_through_point k x₂ y₂ ∧
  x₁ ≠ x₂

-- Define the condition for a circle with diameter AB passing through the origin
def circle_through_origin (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₁ * x₂ + y₁ * y₂ = 0

-- The main theorem
theorem exists_line_and_circle_through_origin :
  ∃ (k : ℝ) (x₁ y₁ x₂ y₂ : ℝ),
    intersection_points k x₁ y₁ x₂ y₂ ∧
    circle_through_origin x₁ y₁ x₂ y₂ :=
sorry

end NUMINAMATH_CALUDE_exists_line_and_circle_through_origin_l4188_418860


namespace NUMINAMATH_CALUDE_dihedral_angle_distance_l4188_418870

/-- Given a dihedral angle φ and a point A on one of its faces with distance a from the edge,
    the distance from A to the plane of the other face is a * sin(φ). -/
theorem dihedral_angle_distance (φ : ℝ) (a : ℝ) :
  let distance_to_edge := a
  let distance_to_other_face := a * Real.sin φ
  distance_to_other_face = distance_to_edge * Real.sin φ := by
  sorry

end NUMINAMATH_CALUDE_dihedral_angle_distance_l4188_418870
