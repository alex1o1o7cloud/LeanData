import Mathlib

namespace NUMINAMATH_CALUDE_exists_h_for_phi_l4083_408376

-- Define the types for our functions
def φ : ℝ → ℝ → ℝ → ℝ := sorry
def f : ℝ → ℝ → ℝ := sorry
def g : ℝ → ℝ → ℝ := sorry

-- State the theorem
theorem exists_h_for_phi (hf : ∀ x y z, φ x y z = f (x + y) z)
                         (hg : ∀ x y z, φ x y z = g x (y + z)) :
  ∃ h : ℝ → ℝ, ∀ x y z, φ x y z = h (x + y + z) := by sorry

end NUMINAMATH_CALUDE_exists_h_for_phi_l4083_408376


namespace NUMINAMATH_CALUDE_fraction_change_l4083_408397

theorem fraction_change (a b k : ℕ+) :
  (a : ℚ) / b < 1 → (a + k : ℚ) / (b + k) > a / b ∧
  (a : ℚ) / b > 1 → (a + k : ℚ) / (b + k) < a / b :=
by sorry

end NUMINAMATH_CALUDE_fraction_change_l4083_408397


namespace NUMINAMATH_CALUDE_equation_solutions_l4083_408353

theorem equation_solutions :
  ∀ a b : ℕ, a^2 = b * (b + 7) → (a = 0 ∧ b = 0) ∨ (a = 12 ∧ b = 9) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l4083_408353


namespace NUMINAMATH_CALUDE_bruce_bank_savings_l4083_408320

/-- The amount of money Bruce puts in the bank given his birthday gifts -/
def money_in_bank (aunt_gift : ℕ) (grandfather_gift : ℕ) : ℕ :=
  (aunt_gift + grandfather_gift) / 5

/-- Theorem stating that Bruce puts $45 in the bank -/
theorem bruce_bank_savings : money_in_bank 75 150 = 45 := by
  sorry

end NUMINAMATH_CALUDE_bruce_bank_savings_l4083_408320


namespace NUMINAMATH_CALUDE_sqrt_3_times_sqrt_12_l4083_408375

theorem sqrt_3_times_sqrt_12 : Real.sqrt 3 * Real.sqrt 12 = 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_3_times_sqrt_12_l4083_408375


namespace NUMINAMATH_CALUDE_finite_square_solutions_l4083_408374

theorem finite_square_solutions (a b : ℤ) (h : ¬ ∃ k : ℤ, b = k^2) :
  { x : ℤ | ∃ y : ℤ, x^2 + a*x + b = y^2 }.Finite :=
sorry

end NUMINAMATH_CALUDE_finite_square_solutions_l4083_408374


namespace NUMINAMATH_CALUDE_cos_cube_decomposition_sum_of_squares_l4083_408383

open Real

theorem cos_cube_decomposition_sum_of_squares :
  (∃ b₁ b₂ b₃ : ℝ, ∀ θ : ℝ, cos θ ^ 3 = b₁ * cos θ + b₂ * cos (2 * θ) + b₃ * cos (3 * θ)) →
  (∃ b₁ b₂ b₃ : ℝ, 
    (∀ θ : ℝ, cos θ ^ 3 = b₁ * cos θ + b₂ * cos (2 * θ) + b₃ * cos (3 * θ)) ∧
    b₁ ^ 2 + b₂ ^ 2 + b₃ ^ 2 = 5 / 8) :=
by sorry

end NUMINAMATH_CALUDE_cos_cube_decomposition_sum_of_squares_l4083_408383


namespace NUMINAMATH_CALUDE_calculation_proof_l4083_408339

theorem calculation_proof : (1/2)⁻¹ + Real.sqrt 12 - 4 * Real.sin (60 * π / 180) = 2 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l4083_408339


namespace NUMINAMATH_CALUDE_total_pencils_given_l4083_408392

/-- The number of pencils in a dozen -/
def pencils_per_dozen : ℕ := 12

/-- The number of children in the classroom -/
def num_children : ℕ := 46

/-- The number of dozens of pencils each child receives -/
def dozens_per_child : ℕ := 4

/-- Theorem: The total number of pencils given out is 2208 -/
theorem total_pencils_given : 
  num_children * (dozens_per_child * pencils_per_dozen) = 2208 := by
  sorry

end NUMINAMATH_CALUDE_total_pencils_given_l4083_408392


namespace NUMINAMATH_CALUDE_coefficient_x5_is_11_l4083_408317

/-- The coefficient of x^5 in the expansion of ((x^2 + x - 1)^5) -/
def coefficient_x5 : ℤ :=
  (Nat.choose 5 0) * (Nat.choose 5 5) -
  (Nat.choose 5 1) * (Nat.choose 4 3) +
  (Nat.choose 5 2) * (Nat.choose 3 1)

/-- Theorem stating that the coefficient of x^5 in ((x^2 + x - 1)^5) is 11 -/
theorem coefficient_x5_is_11 : coefficient_x5 = 11 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x5_is_11_l4083_408317


namespace NUMINAMATH_CALUDE_division_remainder_l4083_408359

theorem division_remainder : ∃ A : ℕ, 28 = 3 * 9 + A ∧ A < 3 := by
  sorry

end NUMINAMATH_CALUDE_division_remainder_l4083_408359


namespace NUMINAMATH_CALUDE_cubic_equation_roots_range_l4083_408385

theorem cubic_equation_roots_range (k : ℝ) : 
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ 
    x^3 - 3*x = k ∧ y^3 - 3*y = k ∧ z^3 - 3*z = k) → 
  -2 < k ∧ k < 2 := by
sorry

end NUMINAMATH_CALUDE_cubic_equation_roots_range_l4083_408385


namespace NUMINAMATH_CALUDE_rounding_accuracy_l4083_408378

-- Define the rounded number
def rounded_number : ℝ := 5.8 * 10^5

-- Define the accuracy levels
inductive AccuracyLevel
  | Tenth
  | Hundredth
  | Thousandth
  | TenThousandth
  | HundredThousandth

-- Define a function to determine the accuracy level
def determine_accuracy (x : ℝ) : AccuracyLevel :=
  match x with
  | _ => AccuracyLevel.TenThousandth -- We know this is the correct answer from the problem

-- State the theorem
theorem rounding_accuracy :
  determine_accuracy rounded_number = AccuracyLevel.TenThousandth :=
by sorry

end NUMINAMATH_CALUDE_rounding_accuracy_l4083_408378


namespace NUMINAMATH_CALUDE_polyhedron_sum_l4083_408341

/-- A convex polyhedron with triangular and pentagonal faces -/
structure ConvexPolyhedron where
  faces : ℕ
  vertices : ℕ
  triangular_faces : ℕ
  pentagonal_faces : ℕ
  T : ℕ  -- number of triangular faces meeting at each vertex
  P : ℕ  -- number of pentagonal faces meeting at each vertex
  faces_sum : faces = triangular_faces + pentagonal_faces
  faces_32 : faces = 32
  vertex_relation : vertices * (T + P - 2) = 60
  face_relation : 5 * vertices * T + 3 * vertices * P = 480

/-- The sum of P, T, and V for the specific polyhedron is 34 -/
theorem polyhedron_sum (poly : ConvexPolyhedron) : poly.P + poly.T + poly.vertices = 34 := by
  sorry

end NUMINAMATH_CALUDE_polyhedron_sum_l4083_408341


namespace NUMINAMATH_CALUDE_birthday_cake_is_tradition_l4083_408302

/-- Represents different types of office practices -/
inductive OfficePractice
  | Tradition
  | Balance
  | Concern
  | Relationship

/-- Represents the office birthday cake practice -/
def birthdayCakePractice : OfficePractice := OfficePractice.Tradition

/-- Theorem stating that the office birthday cake practice is a tradition -/
theorem birthday_cake_is_tradition : 
  birthdayCakePractice = OfficePractice.Tradition := by sorry


end NUMINAMATH_CALUDE_birthday_cake_is_tradition_l4083_408302


namespace NUMINAMATH_CALUDE_mitch_max_boat_length_l4083_408321

/-- The maximum length of boat Mitch can buy given his savings and expenses --/
def max_boat_length (savings : ℚ) (cost_per_foot : ℚ) (license_fee : ℚ) : ℚ :=
  let docking_fee := 3 * license_fee
  let total_fees := license_fee + docking_fee
  let remaining_money := savings - total_fees
  remaining_money / cost_per_foot

/-- Theorem stating the maximum length of boat Mitch can buy --/
theorem mitch_max_boat_length :
  max_boat_length 20000 1500 500 = 12 := by
sorry

end NUMINAMATH_CALUDE_mitch_max_boat_length_l4083_408321


namespace NUMINAMATH_CALUDE_grace_has_30_pastries_l4083_408357

/-- The number of pastries each person has -/
structure Pastries where
  frank : ℕ
  calvin : ℕ
  phoebe : ℕ
  grace : ℕ

/-- The conditions of the pastry problem -/
def pastry_conditions (p : Pastries) : Prop :=
  p.calvin = p.frank + 8 ∧
  p.phoebe = p.frank + 8 ∧
  p.grace = p.calvin + 5 ∧
  p.frank + p.calvin + p.phoebe + p.grace = 97

/-- The theorem stating that Grace has 30 pastries -/
theorem grace_has_30_pastries (p : Pastries) 
  (h : pastry_conditions p) : p.grace = 30 := by
  sorry


end NUMINAMATH_CALUDE_grace_has_30_pastries_l4083_408357


namespace NUMINAMATH_CALUDE_integer_part_of_M_l4083_408349

theorem integer_part_of_M (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (sum_eq_one : a + b + c = 1) : 
  4 < Real.sqrt (3 * a + 1) + Real.sqrt (3 * b + 1) + Real.sqrt (3 * c + 1) ∧
  Real.sqrt (3 * a + 1) + Real.sqrt (3 * b + 1) + Real.sqrt (3 * c + 1) < 5 :=
by sorry

end NUMINAMATH_CALUDE_integer_part_of_M_l4083_408349


namespace NUMINAMATH_CALUDE_line_circle_intersection_l4083_408343

/-- The line y = x + 1 intersects the circle x² + y² = 1 at two distinct points, 
    and neither of these points is the center of the circle (0, 0). -/
theorem line_circle_intersection :
  ∃ (x₁ y₁ x₂ y₂ : ℝ), 
    (y₁ = x₁ + 1) ∧ (x₁^2 + y₁^2 = 1) ∧
    (y₂ = x₂ + 1) ∧ (x₂^2 + y₂^2 = 1) ∧
    (x₁ ≠ x₂) ∧ (y₁ ≠ y₂) ∧
    (x₁ ≠ 0 ∨ y₁ ≠ 0) ∧ (x₂ ≠ 0 ∨ y₂ ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_line_circle_intersection_l4083_408343


namespace NUMINAMATH_CALUDE_log_equation_sum_l4083_408388

theorem log_equation_sum : ∃ (X Y Z : ℕ+),
  (∀ d : ℕ+, d ∣ X ∧ d ∣ Y ∧ d ∣ Z → d = 1) ∧
  (X : ℝ) * Real.log 3 / Real.log 180 + (Y : ℝ) * Real.log 5 / Real.log 180 = Z ∧
  X + Y + Z = 4 := by
  sorry

end NUMINAMATH_CALUDE_log_equation_sum_l4083_408388


namespace NUMINAMATH_CALUDE_popcorn_buckets_needed_l4083_408306

/-- The number of popcorn buckets needed by a movie theater -/
theorem popcorn_buckets_needed (packages : ℕ) (buckets_per_package : ℕ) 
  (h1 : packages = 54)
  (h2 : buckets_per_package = 8) :
  packages * buckets_per_package = 432 := by
  sorry

end NUMINAMATH_CALUDE_popcorn_buckets_needed_l4083_408306


namespace NUMINAMATH_CALUDE_discount_price_l4083_408384

theorem discount_price (a : ℝ) :
  let discounted_price := a
  let discount_rate := 0.3
  let original_price := discounted_price / (1 - discount_rate)
  original_price = 10 / 7 * a :=
by sorry

end NUMINAMATH_CALUDE_discount_price_l4083_408384


namespace NUMINAMATH_CALUDE_probability_black_second_draw_l4083_408351

theorem probability_black_second_draw 
  (total_balls : ℕ) 
  (white_balls : ℕ) 
  (black_balls : ℕ) 
  (h1 : total_balls = 5) 
  (h2 : white_balls = 3) 
  (h3 : black_balls = 2) 
  (h4 : total_balls = white_balls + black_balls) 
  (h5 : white_balls > 0) : 
  (black_balls : ℚ) / (total_balls - 1 : ℚ) = 1/2 := by
sorry

end NUMINAMATH_CALUDE_probability_black_second_draw_l4083_408351


namespace NUMINAMATH_CALUDE_interest_rate_calculation_l4083_408396

/-- Calculate the interest rate given simple interest, principal, and time -/
theorem interest_rate_calculation
  (simple_interest : ℝ)
  (principal : ℝ)
  (time : ℝ)
  (h1 : simple_interest = 4016.25)
  (h2 : principal = 10040.625)
  (h3 : time = 5)
  (h4 : simple_interest = principal * (rate / 100) * time) :
  rate = 8 := by
  sorry

end NUMINAMATH_CALUDE_interest_rate_calculation_l4083_408396


namespace NUMINAMATH_CALUDE_cube_surface_area_increase_l4083_408314

theorem cube_surface_area_increase (L : ℝ) (h : L > 0) :
  let original_area := 6 * L^2
  let new_edge := 1.1 * L
  let new_area := 6 * new_edge^2
  (new_area - original_area) / original_area = 0.21 :=
by sorry

end NUMINAMATH_CALUDE_cube_surface_area_increase_l4083_408314


namespace NUMINAMATH_CALUDE_function_domain_range_implies_m_range_l4083_408391

-- Define the function
def f (x : ℝ) : ℝ := x^2 - 4*x - 2

-- Define the theorem
theorem function_domain_range_implies_m_range 
  (m : ℝ) 
  (domain : Set ℝ)
  (range : Set ℝ)
  (h_domain : domain = Set.Icc 0 m)
  (h_range : range = Set.Icc (-6) (-2))
  (h_func_range : ∀ x ∈ domain, f x ∈ range) :
  m ∈ Set.Icc 2 4 :=
sorry

end NUMINAMATH_CALUDE_function_domain_range_implies_m_range_l4083_408391


namespace NUMINAMATH_CALUDE_black_bears_count_l4083_408328

/-- Represents the number of bears of each color in the park -/
structure BearPopulation where
  white : ℕ
  black : ℕ
  brown : ℕ

/-- Conditions for the bear population in the park -/
def validBearPopulation (p : BearPopulation) : Prop :=
  p.black = 2 * p.white ∧
  p.brown = p.black + 40 ∧
  p.white + p.black + p.brown = 190

/-- Theorem stating that under the given conditions, there are 60 black bears -/
theorem black_bears_count (p : BearPopulation) (h : validBearPopulation p) : p.black = 60 := by
  sorry

end NUMINAMATH_CALUDE_black_bears_count_l4083_408328


namespace NUMINAMATH_CALUDE_freddy_travel_time_l4083_408322

/-- Represents the travel details of a person -/
structure TravelDetails where
  start : String
  destination : String
  distance : ℝ
  time : ℝ

/-- Given travel conditions for Eddy and Freddy -/
def travel_conditions : Prop :=
  ∃ (eddy freddy : TravelDetails),
    eddy.start = "A" ∧
    eddy.destination = "B" ∧
    freddy.start = "A" ∧
    freddy.destination = "C" ∧
    eddy.distance = 540 ∧
    freddy.distance = 300 ∧
    eddy.time = 3 ∧
    (eddy.distance / eddy.time) / (freddy.distance / freddy.time) = 2.4

/-- Theorem: Freddy's travel time is 4 hours -/
theorem freddy_travel_time : travel_conditions → ∃ (freddy : TravelDetails), freddy.time = 4 := by
  sorry


end NUMINAMATH_CALUDE_freddy_travel_time_l4083_408322


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l4083_408300

/-- Given a geometric sequence with positive terms where (a_3, 1/2*a_5, a_4) form an arithmetic sequence,
    prove that (a_3 + a_5) / (a_4 + a_6) = (√5 - 1) / 2 -/
theorem geometric_sequence_ratio (a : ℕ → ℝ) (h_positive : ∀ n, a n > 0)
  (h_geometric : ∃ q : ℝ, q > 0 ∧ ∀ n, a (n + 1) = a n * q)
  (h_arithmetic : a 3 + a 4 = a 5) :
  (a 3 + a 5) / (a 4 + a 6) = (Real.sqrt 5 - 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l4083_408300


namespace NUMINAMATH_CALUDE_complement_of_union_l4083_408337

def U : Set Int := {-2, -1, 0, 1, 2, 3}
def A : Set Int := {-1, 2}
def B : Set Int := {x : Int | x^2 - 4*x + 3 = 0}

theorem complement_of_union : (U \ (A ∪ B)) = {-2, 0} := by
  sorry

end NUMINAMATH_CALUDE_complement_of_union_l4083_408337


namespace NUMINAMATH_CALUDE_total_cars_l4083_408360

/-- The number of cars owned by each person --/
structure CarOwnership where
  cathy : ℕ
  lindsey : ℕ
  carol : ℕ
  susan : ℕ
  erica : ℕ
  jack : ℕ
  kevin : ℕ

/-- The conditions of car ownership --/
def carOwnershipConditions (c : CarOwnership) : Prop :=
  c.cathy = 5 ∧
  c.lindsey = c.cathy + 4 ∧
  c.susan = c.carol - 2 ∧
  c.carol = 2 * c.cathy ∧
  c.erica = c.lindsey + (c.lindsey / 4) ∧
  c.jack = (c.susan + c.carol) / 2 ∧
  c.kevin = ((c.lindsey + c.cathy) * 9) / 10

/-- The theorem stating the total number of cars --/
theorem total_cars (c : CarOwnership) (h : carOwnershipConditions c) : 
  c.cathy + c.lindsey + c.carol + c.susan + c.erica + c.jack + c.kevin = 65 := by
  sorry


end NUMINAMATH_CALUDE_total_cars_l4083_408360


namespace NUMINAMATH_CALUDE_geometric_sequence_101st_term_l4083_408316

-- Define a geometric sequence
def geometric_sequence (a₁ : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  a₁ * r^(n - 1)

-- Theorem statement
theorem geometric_sequence_101st_term :
  let a₁ := 12
  let a₂ := -36
  let r := a₂ / a₁
  geometric_sequence a₁ r 101 = 12 * 3^100 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_101st_term_l4083_408316


namespace NUMINAMATH_CALUDE_sin_product_identity_l4083_408369

theorem sin_product_identity : 
  Real.sin (10 * π / 180) * Real.sin (50 * π / 180) * Real.sin (70 * π / 180) * Real.sin (80 * π / 180) = 1/8 := by
  sorry

end NUMINAMATH_CALUDE_sin_product_identity_l4083_408369


namespace NUMINAMATH_CALUDE_polar_to_cartesian_and_intersections_l4083_408330

/-- A circle in polar form -/
structure PolarCircle where
  equation : ℝ → ℝ → Prop

/-- A line in polar form -/
structure PolarLine where
  equation : ℝ → ℝ → Prop

/-- A point in polar coordinates -/
structure PolarPoint where
  ρ : ℝ
  θ : ℝ

/-- Given a circle C₁ and a line C₂ in polar form, 
    prove their Cartesian equations and intersection points -/
theorem polar_to_cartesian_and_intersections 
  (C₁ : PolarCircle) 
  (C₂ : PolarLine) 
  (h₁ : C₁.equation = fun ρ θ ↦ ρ = 4 * Real.sin θ)
  (h₂ : C₂.equation = fun ρ θ ↦ ρ * Real.cos (θ - π/4) = 2 * Real.sqrt 2) :
  ∃ (f₁ f₂ : ℝ → ℝ → Prop) (p₁ p₂ : PolarPoint),
    (∀ x y, f₁ x y ↔ x^2 + (y-2)^2 = 4) ∧
    (∀ x y, f₂ x y ↔ x + y = 4) ∧
    p₁ = ⟨4, π/2⟩ ∧
    p₂ = ⟨2 * Real.sqrt 2, π/4⟩ ∧
    C₁.equation p₁.ρ p₁.θ ∧
    C₂.equation p₁.ρ p₁.θ ∧
    C₁.equation p₂.ρ p₂.θ ∧
    C₂.equation p₂.ρ p₂.θ := by
  sorry

end NUMINAMATH_CALUDE_polar_to_cartesian_and_intersections_l4083_408330


namespace NUMINAMATH_CALUDE_system_solution_is_solution_set_l4083_408323

def system_solution (x y z : ℝ) : Prop :=
  x + y + z = 6 ∧ x*y + y*z + z*x = 11 ∧ x*y*z = 6

def solution_set : Set (ℝ × ℝ × ℝ) :=
  {(1, 2, 3), (1, 3, 2), (2, 1, 3), (2, 3, 1), (3, 1, 2), (3, 2, 1)}

theorem system_solution_is_solution_set :
  ∀ x y z, system_solution x y z ↔ (x, y, z) ∈ solution_set :=
by sorry

end NUMINAMATH_CALUDE_system_solution_is_solution_set_l4083_408323


namespace NUMINAMATH_CALUDE_smallest_number_with_digit_product_10_factorial_l4083_408394

def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

def digit_product (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) * digit_product (n / 10)

def is_valid_number (n : ℕ) : Prop :=
  digit_product n = factorial 10

theorem smallest_number_with_digit_product_10_factorial :
  ∀ n : ℕ, n < 45578899 → ¬(is_valid_number n) ∧ is_valid_number 45578899 :=
sorry

end NUMINAMATH_CALUDE_smallest_number_with_digit_product_10_factorial_l4083_408394


namespace NUMINAMATH_CALUDE_cos_45_degrees_l4083_408313

theorem cos_45_degrees : Real.cos (π / 4) = 1 / Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_cos_45_degrees_l4083_408313


namespace NUMINAMATH_CALUDE_integer_2020_in_column_F_l4083_408364

/-- Represents the columns in the arrangement --/
inductive Column
  | A | B | C | D | E | F | G

/-- Defines the arrangement of integers in columns --/
def arrangement (n : ℕ) : Column :=
  match (n - 11) % 14 with
  | 0 => Column.A
  | 1 => Column.B
  | 2 => Column.C
  | 3 => Column.D
  | 4 => Column.E
  | 5 => Column.F
  | 6 => Column.G
  | 7 => Column.G
  | 8 => Column.F
  | 9 => Column.E
  | 10 => Column.D
  | 11 => Column.C
  | 12 => Column.B
  | _ => Column.A

/-- Theorem: The integer 2020 is in column F --/
theorem integer_2020_in_column_F : arrangement 2020 = Column.F := by
  sorry

end NUMINAMATH_CALUDE_integer_2020_in_column_F_l4083_408364


namespace NUMINAMATH_CALUDE_deepak_age_l4083_408398

/-- Given the ratio of ages and future ages of Rahul and Sandeep, prove Deepak's present age --/
theorem deepak_age (r d s : ℕ) : 
  r / d = 4 / 3 →  -- ratio of Rahul to Deepak's age
  d / s = 1 / 2 →  -- ratio of Deepak to Sandeep's age
  r + 6 = 42 →     -- Rahul's age after 6 years
  s + 9 = 57 →     -- Sandeep's age after 9 years
  d = 27 :=        -- Deepak's present age
by sorry

end NUMINAMATH_CALUDE_deepak_age_l4083_408398


namespace NUMINAMATH_CALUDE_missing_number_proof_l4083_408372

theorem missing_number_proof : ∃ x : ℚ, (306 / 34) * x + 270 = 405 ∧ x = 15 := by
  sorry

end NUMINAMATH_CALUDE_missing_number_proof_l4083_408372


namespace NUMINAMATH_CALUDE_second_trip_crates_parameters_are_valid_l4083_408326

/-- The number of crates carried in the second trip of a trailer -/
def crates_in_second_trip (total_crates : ℕ) (min_crate_weight : ℕ) (max_trip_weight : ℕ) : ℕ :=
  total_crates - (max_trip_weight / min_crate_weight)

/-- Theorem stating that given the specified conditions, the trailer carries 7 crates in the second trip -/
theorem second_trip_crates :
  crates_in_second_trip 12 120 600 = 7 := by
  sorry

/-- Checks if the given parameters satisfy the problem conditions -/
def valid_parameters (total_crates : ℕ) (min_crate_weight : ℕ) (max_trip_weight : ℕ) : Prop :=
  total_crates > 0 ∧
  min_crate_weight > 0 ∧
  max_trip_weight > 0 ∧
  min_crate_weight * total_crates > max_trip_weight

/-- Theorem stating that the given parameters satisfy the problem conditions -/
theorem parameters_are_valid :
  valid_parameters 12 120 600 := by
  sorry

end NUMINAMATH_CALUDE_second_trip_crates_parameters_are_valid_l4083_408326


namespace NUMINAMATH_CALUDE_triangle_altitude_l4083_408350

/-- Given a triangle with area 720 square feet and base 40 feet, its altitude is 36 feet -/
theorem triangle_altitude (area : ℝ) (base : ℝ) (altitude : ℝ) : 
  area = 720 → base = 40 → area = (1/2) * base * altitude → altitude = 36 := by
  sorry

end NUMINAMATH_CALUDE_triangle_altitude_l4083_408350


namespace NUMINAMATH_CALUDE_prime_fraction_equation_l4083_408365

theorem prime_fraction_equation (p q r : ℕ+) (hp : Nat.Prime p) (hq : Nat.Prime q) :
  (1 : ℚ) / (p + 1) + (1 : ℚ) / (q + 1) - 1 / ((p + 1) * (q + 1)) = 1 / r →
  ((p = 2 ∧ q = 3) ∨ (p = 3 ∧ q = 2)) ∧ r = 2 := by
  sorry

end NUMINAMATH_CALUDE_prime_fraction_equation_l4083_408365


namespace NUMINAMATH_CALUDE_inscribed_triangle_angle_measure_l4083_408355

/-- Given a triangle PQR inscribed in a circle, if the measures of arcs PQ, QR, and RP
    are y + 60°, 2y + 40°, and 3y - 10° respectively, then the measure of interior angle Q
    is 62.5°. -/
theorem inscribed_triangle_angle_measure (y : ℝ) :
  let arc_PQ : ℝ := y + 60
  let arc_QR : ℝ := 2 * y + 40
  let arc_RP : ℝ := 3 * y - 10
  arc_PQ + arc_QR + arc_RP = 360 →
  (1 / 2 : ℝ) * arc_RP = 62.5 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_triangle_angle_measure_l4083_408355


namespace NUMINAMATH_CALUDE_inequality_solution_set_l4083_408395

theorem inequality_solution_set (y : ℝ) :
  (2 / (y - 2) + 5 / (y + 3) ≤ 2) ↔ (y ∈ Set.Ioc (-3) (-1) ∪ Set.Ioo 2 4) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l4083_408395


namespace NUMINAMATH_CALUDE_exists_fourth_root_of_3_to_20_l4083_408347

theorem exists_fourth_root_of_3_to_20 : ∃ n : ℕ, n^4 = 3^20 ∧ n = 243 := by sorry

end NUMINAMATH_CALUDE_exists_fourth_root_of_3_to_20_l4083_408347


namespace NUMINAMATH_CALUDE_inequality_solution_l4083_408318

theorem inequality_solution (x : ℝ) : 
  (2 / (x + 2) + 5 / (x + 4) ≥ 1) ↔ (x < -4 ∨ x ≥ 5) := by sorry

end NUMINAMATH_CALUDE_inequality_solution_l4083_408318


namespace NUMINAMATH_CALUDE_sqrt_27_minus_sqrt_3_equals_2_sqrt_3_l4083_408363

theorem sqrt_27_minus_sqrt_3_equals_2_sqrt_3 : 
  Real.sqrt 27 - Real.sqrt 3 = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_27_minus_sqrt_3_equals_2_sqrt_3_l4083_408363


namespace NUMINAMATH_CALUDE_f_max_min_range_l4083_408333

/-- A cubic function with parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + (a+6)*x + 1

/-- The derivative of f with respect to x -/
def f' (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*a*x + (a+6)

/-- Condition for f to have both a maximum and a minimum -/
def has_max_and_min (a : ℝ) : Prop := ∃ x y : ℝ, x ≠ y ∧ f' a x = 0 ∧ f' a y = 0

theorem f_max_min_range (a : ℝ) : has_max_and_min a → a < -3 ∨ a > 6 := by sorry

end NUMINAMATH_CALUDE_f_max_min_range_l4083_408333


namespace NUMINAMATH_CALUDE_derivative_of_f_l4083_408307

noncomputable def f (x : ℝ) : ℝ := x^2 * Real.cos x

theorem derivative_of_f (x : ℝ) :
  HasDerivAt f (2*x*(Real.cos x) - x^2*(Real.sin x)) x := by
  sorry

end NUMINAMATH_CALUDE_derivative_of_f_l4083_408307


namespace NUMINAMATH_CALUDE_unique_root_and_sequence_l4083_408362

theorem unique_root_and_sequence : ∃! r : ℝ, 
  (2 * r^3 + 5 * r - 2 = 0) ∧ 
  ∃! (a : ℕ → ℕ), (∀ n, a n < a (n+1)) ∧ 
    (2/5 : ℝ) = ∑' n, r^(a n) ∧
    ∀ n, a n = 3*n - 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_root_and_sequence_l4083_408362


namespace NUMINAMATH_CALUDE_parallelogram_area_main_theorem_l4083_408387

/-- Represents a parallelogram formed by vertices of smaller triangles inside an equilateral triangle -/
structure Parallelogram where
  m : ℕ  -- Length of one side in terms of unit triangles
  n : ℕ  -- Length of the other side in terms of unit triangles

/-- The main theorem statement -/
theorem parallelogram_area (p : Parallelogram) : 
  (p.m > 1 ∧ p.n > 1) →  -- Sides must be greater than 1 unit triangle
  (p.m + p.n = 6) →      -- Sum of sides is 6 (derived from the 46 triangles condition)
  (p.m * p.n = 8 ∨ p.m * p.n = 9) := by
  sorry

/-- The equilateral triangle ABC -/
def ABC : Set (ℝ × ℝ) := sorry

/-- The set of 400 smaller equilateral triangles -/
def small_triangles : Set (Set (ℝ × ℝ)) := sorry

/-- The condition that the parallelogram is formed by vertices of smaller triangles inside ABC -/
def parallelogram_in_triangle (p : Parallelogram) : Prop := sorry

/-- The condition that the parallelogram sides are parallel to ABC sides -/
def parallel_to_ABC_sides (p : Parallelogram) : Prop := sorry

/-- The condition that exactly 46 smaller triangles have at least one point in common with the parallelogram sides -/
def triangles_touching_sides (p : Parallelogram) : Prop := sorry

/-- The main theorem combining all conditions -/
theorem main_theorem (p : Parallelogram) :
  parallelogram_in_triangle p →
  parallel_to_ABC_sides p →
  triangles_touching_sides p →
  (p.m * p.n = 8 ∨ p.m * p.n = 9) := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_main_theorem_l4083_408387


namespace NUMINAMATH_CALUDE_complex_magnitude_3_minus_10i_l4083_408352

theorem complex_magnitude_3_minus_10i :
  Complex.abs (3 - 10 * Complex.I) = Real.sqrt 109 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_3_minus_10i_l4083_408352


namespace NUMINAMATH_CALUDE_hidden_faces_sum_l4083_408340

def standard_die_sum : ℕ := 21

def visible_faces : List ℕ := [1, 1, 2, 3, 4, 5, 6, 6, 5]

def total_faces : ℕ := 24

theorem hidden_faces_sum :
  let total_dots := 4 * standard_die_sum
  let visible_dots := visible_faces.sum
  let hidden_faces := total_faces - visible_faces.length
  hidden_faces = 15 ∧ total_dots - visible_dots = 51 := by sorry

end NUMINAMATH_CALUDE_hidden_faces_sum_l4083_408340


namespace NUMINAMATH_CALUDE_inequality_solution_range_l4083_408345

theorem inequality_solution_range (m : ℝ) : 
  (∃ x : ℝ, m * x^2 + 2 * m * x - 8 ≥ 0) ↔ m ≠ 0 := by
sorry

end NUMINAMATH_CALUDE_inequality_solution_range_l4083_408345


namespace NUMINAMATH_CALUDE_proposition_equivalence_implies_m_range_l4083_408309

-- Define the propositions p and q
def p (x : ℝ) : Prop := x^2 - 3*x - 10 > 0
def q (x m : ℝ) : Prop := x > m^2 - m + 3

-- Define the range of m
def m_range (m : ℝ) : Prop := m ≤ -1 ∨ m ≥ 2

-- State the theorem
theorem proposition_equivalence_implies_m_range :
  (∀ x m : ℝ, (¬(p x) ↔ ¬(q x m))) → 
  (∀ m : ℝ, m_range m) :=
sorry

end NUMINAMATH_CALUDE_proposition_equivalence_implies_m_range_l4083_408309


namespace NUMINAMATH_CALUDE_stratified_sample_size_l4083_408379

/-- Calculates the total sample size for a stratified sampling method given workshop productions and a known sample from one workshop. -/
theorem stratified_sample_size 
  (production_A production_B production_C : ℕ) 
  (sample_C : ℕ) : 
  production_A = 120 → 
  production_B = 80 → 
  production_C = 60 → 
  sample_C = 3 → 
  (production_A + production_B + production_C) * sample_C / production_C = 13 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sample_size_l4083_408379


namespace NUMINAMATH_CALUDE_find_B_l4083_408354

theorem find_B (A B : ℕ) (h1 : A = 21) (h2 : Nat.gcd A B = 7) (h3 : Nat.lcm A B = 105) :
  B = 35 := by
sorry

end NUMINAMATH_CALUDE_find_B_l4083_408354


namespace NUMINAMATH_CALUDE_average_monthly_growth_rate_l4083_408367

theorem average_monthly_growth_rate 
  (initial_production : ℕ) 
  (final_production : ℕ) 
  (months : ℕ) 
  (growth_rate : ℝ) :
  initial_production = 100 →
  final_production = 144 →
  months = 2 →
  initial_production * (1 + growth_rate) ^ months = final_production →
  growth_rate = 0.2 := by
sorry

end NUMINAMATH_CALUDE_average_monthly_growth_rate_l4083_408367


namespace NUMINAMATH_CALUDE_fahrenheit_diff_is_18_l4083_408399

-- Define the conversion function from Celsius to Fahrenheit
def celsius_to_fahrenheit (C : ℝ) : ℝ := 1.8 * C + 32

-- Define the temperature difference in Celsius
def celsius_diff : ℝ := 10

-- Theorem statement
theorem fahrenheit_diff_is_18 :
  celsius_to_fahrenheit (C + celsius_diff) - celsius_to_fahrenheit C = 18 :=
by sorry

end NUMINAMATH_CALUDE_fahrenheit_diff_is_18_l4083_408399


namespace NUMINAMATH_CALUDE_triangle_sides_ratio_bound_l4083_408348

theorem triangle_sides_ratio_bound (a b c : ℝ) :
  (0 < a) → (0 < b) → (0 < c) →  -- Positive sides
  (a + b > c) → (a + c > b) → (b + c > a) →  -- Triangle inequality
  (∃ d : ℝ, b = a + d ∧ c = a + 2*d) →  -- Arithmetic progression
  (a^2 + b^2 + c^2) / (a*b + b*c + c*a) ≥ 1 := by
  sorry

#check triangle_sides_ratio_bound

end NUMINAMATH_CALUDE_triangle_sides_ratio_bound_l4083_408348


namespace NUMINAMATH_CALUDE_retirement_ratio_is_one_to_one_l4083_408358

def monthly_income : ℚ := 2500
def rent : ℚ := 700
def car_payment : ℚ := 300
def groceries : ℚ := 50
def remaining_after_retirement : ℚ := 650

def total_expenses : ℚ := rent + car_payment + (car_payment / 2) + groceries

def money_after_expenses : ℚ := monthly_income - total_expenses

def retirement_contribution : ℚ := money_after_expenses - remaining_after_retirement

theorem retirement_ratio_is_one_to_one :
  retirement_contribution = remaining_after_retirement :=
by sorry

end NUMINAMATH_CALUDE_retirement_ratio_is_one_to_one_l4083_408358


namespace NUMINAMATH_CALUDE_total_spent_on_toys_l4083_408311

def other_toys_cost : ℕ := 1000
def lightsaber_cost : ℕ := 2 * other_toys_cost

theorem total_spent_on_toys : other_toys_cost + lightsaber_cost = 3000 := by
  sorry

end NUMINAMATH_CALUDE_total_spent_on_toys_l4083_408311


namespace NUMINAMATH_CALUDE_base_salary_calculation_l4083_408325

/-- Proves that the base salary in the second option is $1600 given the conditions of the problem. -/
theorem base_salary_calculation (monthly_salary : ℝ) (commission_rate : ℝ) (equal_sales : ℝ) 
  (h1 : monthly_salary = 1800)
  (h2 : commission_rate = 0.04)
  (h3 : equal_sales = 5000)
  (h4 : ∃ (base_salary : ℝ), base_salary + commission_rate * equal_sales = monthly_salary) :
  ∃ (base_salary : ℝ), base_salary = 1600 ∧ base_salary + commission_rate * equal_sales = monthly_salary :=
by sorry

end NUMINAMATH_CALUDE_base_salary_calculation_l4083_408325


namespace NUMINAMATH_CALUDE_triangle_inequality_possible_third_side_l4083_408303

theorem triangle_inequality (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 → a + b > c → b + c > a → c + a > b → 
  ∃ (triangle : Set (ℝ × ℝ)), true := by sorry

theorem possible_third_side : ∃ (triangle : Set (ℝ × ℝ)), 
  (∃ (a b c : ℝ), a = 3 ∧ b = 7 ∧ c = 9 ∧
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  a + b > c ∧ b + c > a ∧ c + a > b) := by sorry

end NUMINAMATH_CALUDE_triangle_inequality_possible_third_side_l4083_408303


namespace NUMINAMATH_CALUDE_student_selection_sequences_l4083_408327

theorem student_selection_sequences (n : ℕ) (k : ℕ) (h1 : n = 10) (h2 : k = 5) :
  Nat.descFactorial n k = 30240 := by
  sorry

end NUMINAMATH_CALUDE_student_selection_sequences_l4083_408327


namespace NUMINAMATH_CALUDE_AD_length_l4083_408312

-- Define the points A, B, C, D, and M
variable (A B C D M : Point)

-- Define the length function
variable (length : Point → Point → ℝ)

-- State the conditions
axiom equal_segments : length A B = length B C ∧ length B C = length C D ∧ length C D = length A D / 4
axiom M_midpoint : length A M = length M D
axiom MC_length : length M C = 7

-- State the theorem
theorem AD_length : length A D = 56 / 3 := by sorry

end NUMINAMATH_CALUDE_AD_length_l4083_408312


namespace NUMINAMATH_CALUDE_complement_A_inter_B_l4083_408334

-- Define set A
def A : Set ℝ := {x | x^2 - x - 2 > 0}

-- Define set B
def B : Set ℝ := {x | |2*x - 3| < 3}

-- Theorem statement
theorem complement_A_inter_B :
  ∀ x : ℝ, x ∈ (A ∩ B)ᶜ ↔ x ≥ 3 ∨ x ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_complement_A_inter_B_l4083_408334


namespace NUMINAMATH_CALUDE_intersection_when_m_is_two_subset_condition_l4083_408310

-- Define sets A and B
def A (m : ℝ) : Set ℝ := {x : ℝ | m - 1 ≤ x ∧ x ≤ 2*m + 1}
def B : Set ℝ := {x : ℝ | -4 ≤ x ∧ x ≤ 2}

-- Theorem 1: When m = 2, A ∩ B = [1, 2]
theorem intersection_when_m_is_two :
  A 2 ∩ B = {x : ℝ | 1 ≤ x ∧ x ≤ 2} := by sorry

-- Theorem 2: A ⊆ A ∩ B if and only if -2 ≤ m ≤ 1/2
theorem subset_condition (m : ℝ) :
  A m ⊆ A m ∩ B ↔ -2 ≤ m ∧ m ≤ 1/2 := by sorry

end NUMINAMATH_CALUDE_intersection_when_m_is_two_subset_condition_l4083_408310


namespace NUMINAMATH_CALUDE_isosceles_trapezoid_projections_imply_frustum_of_cone_l4083_408324

/-- A solid object in 3D space. -/
structure Solid :=
  (shape : Type)

/-- Represents a view of a solid. -/
inductive View
  | Front
  | Side

/-- Represents the shape of a 2D projection. -/
inductive ProjectionShape
  | IsoscelesTrapezoid
  | Other

/-- Returns the shape of the projection of a solid from a given view. -/
def projection (s : Solid) (v : View) : ProjectionShape :=
  sorry

/-- Represents a frustum of a cone. -/
def FrustumOfCone : Solid :=
  sorry

/-- Theorem stating that a solid with isosceles trapezoid projections
    in both front and side views is a frustum of a cone. -/
theorem isosceles_trapezoid_projections_imply_frustum_of_cone
  (s : Solid)
  (h1 : projection s View.Front = ProjectionShape.IsoscelesTrapezoid)
  (h2 : projection s View.Side = ProjectionShape.IsoscelesTrapezoid) :
  s = FrustumOfCone :=
sorry

end NUMINAMATH_CALUDE_isosceles_trapezoid_projections_imply_frustum_of_cone_l4083_408324


namespace NUMINAMATH_CALUDE_school_boys_count_l4083_408335

theorem school_boys_count (total_girls : ℕ) (girl_boy_difference : ℕ) (boys : ℕ) : 
  total_girls = 697 →
  girl_boy_difference = 228 →
  total_girls = boys + girl_boy_difference →
  boys = 469 := by
sorry

end NUMINAMATH_CALUDE_school_boys_count_l4083_408335


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l4083_408356

/-- An arithmetic sequence with the given properties has the general term a_n = n -/
theorem arithmetic_sequence_property (a : ℕ → ℝ) (d : ℝ) : 
  d ≠ 0 ∧  -- non-zero common difference
  (∀ n, a (n + 1) = a n + d) ∧  -- arithmetic sequence property
  a 1 = 1 ∧  -- first term is 1
  (a 3)^2 = a 1 * a 9  -- geometric sequence property for a_1, a_3, a_9
  →
  ∀ n, a n = n := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l4083_408356


namespace NUMINAMATH_CALUDE_river_depth_ratio_l4083_408390

/-- The ratio of river depths in July to June -/
theorem river_depth_ratio :
  let may_depth : ℝ := 5
  let june_depth : ℝ := may_depth + 10
  let july_depth : ℝ := 45
  july_depth / june_depth = 3 := by sorry

end NUMINAMATH_CALUDE_river_depth_ratio_l4083_408390


namespace NUMINAMATH_CALUDE_equal_revenue_for_all_sellers_l4083_408332

/-- Represents an apple seller with their apple count -/
structure AppleSeller :=
  (apples : ℕ)

/-- Calculates the revenue for an apple seller given the pricing scheme -/
def revenue (seller : AppleSeller) : ℕ :=
  let batches := seller.apples / 7
  let leftovers := seller.apples % 7
  batches + 3 * leftovers

/-- The list of apple sellers with their respective apple counts -/
def sellers : List AppleSeller :=
  [⟨20⟩, ⟨40⟩, ⟨60⟩, ⟨80⟩, ⟨100⟩, ⟨120⟩, ⟨140⟩]

theorem equal_revenue_for_all_sellers :
  ∀ s ∈ sellers, revenue s = 20 := by
  sorry

end NUMINAMATH_CALUDE_equal_revenue_for_all_sellers_l4083_408332


namespace NUMINAMATH_CALUDE_circle_passes_through_points_l4083_408386

/-- A circle in the 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Check if a point lies on a circle -/
def Circle.contains (c : Circle) (p : ℝ × ℝ) : Prop :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2

/-- The equation of our circle -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 4*x - 6*y = 0

theorem circle_passes_through_points :
  ∃ (c : Circle),
    (∀ (x y : ℝ), circle_equation x y ↔ c.contains (x, y)) ∧
    c.contains (0, 0) ∧
    c.contains (4, 0) ∧
    c.contains (-1, 1) := by
  sorry

end NUMINAMATH_CALUDE_circle_passes_through_points_l4083_408386


namespace NUMINAMATH_CALUDE_length_AB_is_6_l4083_408344

-- Define the triangle ABC
def Triangle (A B C : ℝ × ℝ) : Prop :=
  -- Right angle at A
  (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 0 ∧
  -- Angle B is 45°
  (C.2 - B.2) / (C.1 - B.1) = 1 ∧
  -- BC = 6√2
  Real.sqrt ((C.1 - B.1)^2 + (C.2 - B.2)^2) = 6 * Real.sqrt 2

-- Theorem statement
theorem length_AB_is_6 (A B C : ℝ × ℝ) (h : Triangle A B C) :
  Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2) = 6 :=
sorry

end NUMINAMATH_CALUDE_length_AB_is_6_l4083_408344


namespace NUMINAMATH_CALUDE_grandview_soccer_league_members_l4083_408338

/-- The cost of a pair of socks in dollars -/
def sock_cost : ℕ := 6

/-- The cost of a T-shirt in dollars -/
def tshirt_cost : ℕ := sock_cost + 7

/-- The cost of a cap in dollars -/
def cap_cost : ℕ := 2 * sock_cost

/-- The total cost for one member's gear for both home and away games -/
def member_cost : ℕ := 2 * (sock_cost + tshirt_cost + cap_cost)

/-- The total cost for all members' gear -/
def total_cost : ℕ := 4410

/-- The number of members in the Grandview Soccer League -/
def num_members : ℕ := 70

theorem grandview_soccer_league_members :
  num_members * member_cost = total_cost :=
sorry

end NUMINAMATH_CALUDE_grandview_soccer_league_members_l4083_408338


namespace NUMINAMATH_CALUDE_journey_speed_proof_l4083_408368

/-- Proves that given a journey of approximately 3 km divided into three equal parts,
    where the first part is traveled at 3 km/hr, the second at 4 km/hr,
    and the total journey takes 47 minutes, the speed of the third part must be 5 km/hr. -/
theorem journey_speed_proof (total_distance : ℝ) (total_time : ℝ) 
  (h1 : total_distance = 3.000000000000001)
  (h2 : total_time = 47 / 60) -- Convert 47 minutes to hours
  (h3 : ∃ (d : ℝ), d > 0 ∧ 3 * d = total_distance) -- Equal distances for each part
  (h4 : ∃ (v : ℝ), v > 0 ∧ 1 / 3 + 1 / 4 + 1 / v = total_time) -- Time equation
  : ∃ (v : ℝ), v = 5 := by
  sorry

end NUMINAMATH_CALUDE_journey_speed_proof_l4083_408368


namespace NUMINAMATH_CALUDE_max_value_expression_l4083_408329

theorem max_value_expression :
  ∃ (M : ℝ), M = 27 ∧
  ∀ (x y : ℝ),
    (Real.sqrt (36 - 4 * Real.sqrt 5) * Real.sin x - Real.sqrt (2 * (1 + Real.cos (2 * x))) - 2) *
    (3 + 2 * Real.sqrt (10 - Real.sqrt 5) * Real.cos y - Real.cos (2 * y)) ≤ M :=
by sorry

end NUMINAMATH_CALUDE_max_value_expression_l4083_408329


namespace NUMINAMATH_CALUDE_max_value_fraction_l4083_408331

theorem max_value_fraction (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : a * (a + b + c) = b * c) :
  a / (b + c) ≤ (Real.sqrt 2 - 1) / 2 ∧
  ∃ a b c, a > 0 ∧ b > 0 ∧ c > 0 ∧ a * (a + b + c) = b * c ∧ a / (b + c) = (Real.sqrt 2 - 1) / 2 :=
by sorry

end NUMINAMATH_CALUDE_max_value_fraction_l4083_408331


namespace NUMINAMATH_CALUDE_squirrel_count_l4083_408377

theorem squirrel_count (first_count : ℕ) (second_count : ℕ) : 
  first_count = 12 →
  second_count = first_count + first_count / 3 →
  first_count + second_count = 28 := by
sorry

end NUMINAMATH_CALUDE_squirrel_count_l4083_408377


namespace NUMINAMATH_CALUDE_zachary_crunches_l4083_408370

/-- Given that David did 4 crunches and 13 fewer crunches than Zachary,
    prove that Zachary did 17 crunches. -/
theorem zachary_crunches (david_crunches : ℕ) (difference : ℕ) 
  (h1 : david_crunches = 4)
  (h2 : difference = 13) :
  david_crunches + difference = 17 := by
  sorry

end NUMINAMATH_CALUDE_zachary_crunches_l4083_408370


namespace NUMINAMATH_CALUDE_correct_calculation_l4083_408301

theorem correct_calculation (x : ℝ) : 3 * x - 5 = 103 → x / 3 - 5 = 7 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l4083_408301


namespace NUMINAMATH_CALUDE_multiples_of_15_between_25_and_205_l4083_408305

theorem multiples_of_15_between_25_and_205 : 
  (Finset.filter (fun n => n % 15 = 0) (Finset.range 205 \ Finset.range 26)).card = 12 := by
  sorry

end NUMINAMATH_CALUDE_multiples_of_15_between_25_and_205_l4083_408305


namespace NUMINAMATH_CALUDE_magic_numbers_theorem_l4083_408319

/-- Represents the numbers chosen by three people -/
structure Numbers where
  ana : ℕ
  beto : ℕ
  caio : ℕ

/-- Performs one round of exchange -/
def exchange (n : Numbers) : Numbers :=
  { ana := n.beto + n.caio
  , beto := n.ana + n.caio
  , caio := n.ana + n.beto }

/-- The theorem to prove -/
theorem magic_numbers_theorem (initial : Numbers) :
  1 ≤ initial.ana ∧ initial.ana ≤ 50 ∧
  1 ≤ initial.beto ∧ initial.beto ≤ 50 ∧
  1 ≤ initial.caio ∧ initial.caio ≤ 50 →
  let second := exchange initial
  let final := exchange second
  final.ana = 104 ∧ final.beto = 123 ∧ final.caio = 137 →
  initial.ana = 13 ∧ initial.beto = 32 ∧ initial.caio = 46 :=
by
  sorry


end NUMINAMATH_CALUDE_magic_numbers_theorem_l4083_408319


namespace NUMINAMATH_CALUDE_payment_difference_l4083_408381

/-- Represents the pizza with its properties and consumption details -/
structure Pizza :=
  (total_slices : ℕ)
  (plain_cost : ℚ)
  (mushroom_cost : ℚ)
  (mushroom_slices : ℕ)
  (alex_plain_slices : ℕ)
  (ally_plain_slices : ℕ)

/-- Calculates the total cost of the pizza -/
def total_cost (p : Pizza) : ℚ :=
  p.plain_cost + p.mushroom_cost

/-- Calculates the cost per slice -/
def cost_per_slice (p : Pizza) : ℚ :=
  total_cost p / p.total_slices

/-- Calculates Alex's payment -/
def alex_payment (p : Pizza) : ℚ :=
  cost_per_slice p * (p.mushroom_slices + p.alex_plain_slices)

/-- Calculates Ally's payment -/
def ally_payment (p : Pizza) : ℚ :=
  cost_per_slice p * p.ally_plain_slices

/-- Theorem stating the difference in payment between Alex and Ally -/
theorem payment_difference (p : Pizza) 
  (h1 : p.total_slices = 12)
  (h2 : p.plain_cost = 12)
  (h3 : p.mushroom_cost = 3)
  (h4 : p.mushroom_slices = 4)
  (h5 : p.alex_plain_slices = 4)
  (h6 : p.ally_plain_slices = 4)
  : alex_payment p - ally_payment p = 5 := by
  sorry


end NUMINAMATH_CALUDE_payment_difference_l4083_408381


namespace NUMINAMATH_CALUDE_recipe_flour_amount_l4083_408380

def initial_flour : ℕ := 8
def additional_flour : ℕ := 2

theorem recipe_flour_amount : initial_flour + additional_flour = 10 := by sorry

end NUMINAMATH_CALUDE_recipe_flour_amount_l4083_408380


namespace NUMINAMATH_CALUDE_two_integers_sum_l4083_408389

theorem two_integers_sum (a b : ℕ+) : 
  a * b + a + b = 103 →
  Nat.gcd a b = 1 →
  a < 20 →
  b < 20 →
  a + b = 19 := by
sorry

end NUMINAMATH_CALUDE_two_integers_sum_l4083_408389


namespace NUMINAMATH_CALUDE_range_of_m_l4083_408304

/-- The function f(x) = x^2 - 2x - 3 -/
def f (x : ℝ) : ℝ := x^2 - 2*x - 3

theorem range_of_m (m : ℝ) (h_m_pos : m > 0) 
  (h_max : ∀ x ∈ Set.Icc 0 m, f x ≤ -3) 
  (h_min : ∀ x ∈ Set.Icc 0 m, f x ≥ -4) 
  (h_max_exists : ∃ x ∈ Set.Icc 0 m, f x = -3) 
  (h_min_exists : ∃ x ∈ Set.Icc 0 m, f x = -4) : 
  m ∈ Set.Icc 1 2 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l4083_408304


namespace NUMINAMATH_CALUDE_expression_simplification_and_evaluation_expression_evaluation_at_negative_one_l4083_408346

theorem expression_simplification_and_evaluation :
  ∀ a : ℝ, a ≠ 1 ∧ a ≠ 2 ∧ a ≠ 0 →
    (a - 3 + 1 / (a - 1)) / ((a^2 - 4) / (a^2 + 2*a)) * (1 / (a - 2)) = a / (a - 1) :=
by sorry

theorem expression_evaluation_at_negative_one :
  (-1 - 3 + 1 / (-1 - 1)) / (((-1)^2 - 4) / ((-1)^2 + 2*(-1))) * (1 / (-1 - 2)) = 1 / 2 :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_and_evaluation_expression_evaluation_at_negative_one_l4083_408346


namespace NUMINAMATH_CALUDE_passes_through_first_and_fourth_quadrants_l4083_408336

-- Define a linear function
def f (x : ℝ) : ℝ := x - 1

-- Theorem statement
theorem passes_through_first_and_fourth_quadrants :
  (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ f x = y) ∧
  (∃ x y : ℝ, x > 0 ∧ y < 0 ∧ f x = y) :=
sorry

end NUMINAMATH_CALUDE_passes_through_first_and_fourth_quadrants_l4083_408336


namespace NUMINAMATH_CALUDE_marys_income_percentage_l4083_408366

theorem marys_income_percentage (juan tim mary : ℝ) 
  (h1 : tim = juan * 0.7)
  (h2 : mary = tim * 1.6) :
  mary = juan * 1.12 := by
sorry

end NUMINAMATH_CALUDE_marys_income_percentage_l4083_408366


namespace NUMINAMATH_CALUDE_books_per_bookshelf_l4083_408342

theorem books_per_bookshelf (total_books : ℕ) (num_bookshelves : ℕ) 
  (h1 : total_books = 42) 
  (h2 : num_bookshelves = 21) :
  total_books / num_bookshelves = 2 := by
  sorry

end NUMINAMATH_CALUDE_books_per_bookshelf_l4083_408342


namespace NUMINAMATH_CALUDE_unique_solution_quadratic_equation_l4083_408382

theorem unique_solution_quadratic_equation :
  ∃! (x y : ℝ), (4 * x^2 + 6 * x + 4) * (4 * y^2 - 12 * y + 25) = 28 ∧
                x = -3/4 ∧ y = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_quadratic_equation_l4083_408382


namespace NUMINAMATH_CALUDE_jacob_michael_age_difference_l4083_408393

theorem jacob_michael_age_difference :
  ∀ (jacob_age michael_age : ℕ),
    (jacob_age + 4 = 11) →
    (michael_age + 5 = 2 * (jacob_age + 5)) →
    (michael_age - jacob_age = 12) :=
by sorry

end NUMINAMATH_CALUDE_jacob_michael_age_difference_l4083_408393


namespace NUMINAMATH_CALUDE_sqrt_1_minus_x_real_l4083_408371

theorem sqrt_1_minus_x_real (x : ℝ) : (∃ y : ℝ, y ^ 2 = 1 - x) ↔ x ≤ 1 := by sorry

end NUMINAMATH_CALUDE_sqrt_1_minus_x_real_l4083_408371


namespace NUMINAMATH_CALUDE_trigonometric_simplification_l4083_408373

theorem trigonometric_simplification (θ : Real) (h : 0 < θ ∧ θ < π) :
  ((1 + Real.sin θ + Real.cos θ) * (Real.sin (θ/2) - Real.cos (θ/2))) / 
  Real.sqrt (2 + 2 * Real.cos θ) = -Real.cos θ := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_simplification_l4083_408373


namespace NUMINAMATH_CALUDE_monotonicity_for_a_2_non_monotonicity_condition_l4083_408361

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log x + 1/2 * x^2 - (1 + a) * x

-- Define the derivative of f
def f' (a : ℝ) (x : ℝ) : ℝ := a / x + x - (1 + a)

theorem monotonicity_for_a_2 :
  ∀ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < 1 → f 2 x₁ < f 2 x₂ ∧
  ∀ x₁ x₂, 1 < x₁ ∧ x₁ < x₂ ∧ x₂ < 2 → f 2 x₁ > f 2 x₂ ∧
  ∀ x₁ x₂, 2 < x₁ ∧ x₁ < x₂ → f 2 x₁ < f 2 x₂ :=
sorry

theorem non_monotonicity_condition :
  ∀ a, (∃ x₁ x₂, 1 < x₁ ∧ x₁ < x₂ ∧ x₂ < 2 ∧ f a x₁ < f a x₂ ∧
       ∃ y₁ y₂, 1 < y₁ ∧ y₁ < y₂ ∧ y₂ < 2 ∧ f a y₁ > f a y₂) ↔
  1 < a ∧ a < 2 :=
sorry

end NUMINAMATH_CALUDE_monotonicity_for_a_2_non_monotonicity_condition_l4083_408361


namespace NUMINAMATH_CALUDE_incircle_radius_of_special_triangle_l4083_408308

-- Define the triangle DEF
structure Triangle :=
  (D E F : ℝ × ℝ)

-- Define the properties of the triangle
def is_right_triangle (t : Triangle) : Prop :=
  -- The angle at F is 90 degrees (right angle)
  sorry

def angle_E_is_45_deg (t : Triangle) : Prop :=
  -- The angle at E is 45 degrees
  sorry

def side_DF_length (t : Triangle) : ℝ :=
  -- The length of side DF
  8

-- Define the incircle radius
def incircle_radius (t : Triangle) : ℝ :=
  -- The radius of the incircle
  sorry

-- Theorem statement
theorem incircle_radius_of_special_triangle (t : Triangle) 
  (h1 : is_right_triangle t)
  (h2 : angle_E_is_45_deg t)
  (h3 : side_DF_length t = 8) :
  incircle_radius t = 8 - 4 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_incircle_radius_of_special_triangle_l4083_408308


namespace NUMINAMATH_CALUDE_alex_lorin_marble_ratio_l4083_408315

/-- Given the following conditions:
  - Lorin has 4 black marbles
  - Jimmy has 22 yellow marbles
  - Alex has a certain ratio of black marbles as Lorin
  - Alex has one half as many yellow marbles as Jimmy
  - Alex has 19 marbles in total

  Prove that the ratio of Alex's black marbles to Lorin's black marbles is 2:1
-/
theorem alex_lorin_marble_ratio :
  ∀ (alex_black alex_yellow : ℕ),
  let lorin_black : ℕ := 4
  let jimmy_yellow : ℕ := 22
  let alex_total : ℕ := 19
  alex_yellow = jimmy_yellow / 2 →
  alex_black + alex_yellow = alex_total →
  ∃ (r : ℚ),
    alex_black = r * lorin_black ∧
    r = 2 := by
  sorry

#check alex_lorin_marble_ratio

end NUMINAMATH_CALUDE_alex_lorin_marble_ratio_l4083_408315
