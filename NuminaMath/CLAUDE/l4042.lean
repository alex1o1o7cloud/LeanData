import Mathlib

namespace NUMINAMATH_CALUDE_quadrilateral_perimeter_l4042_404262

-- Define the quadrilateral ABCD
structure Quadrilateral :=
  (A B C D : ℝ × ℝ)

-- Define the perimeter function
def perimeter (q : Quadrilateral) : ℝ := sorry

-- Define the perpendicular function
def perpendicular (v1 v2 : ℝ × ℝ) : Prop := sorry

-- Define the distance function
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem quadrilateral_perimeter 
  (ABCD : Quadrilateral)
  (perp_AB_BC : perpendicular (ABCD.B - ABCD.A) (ABCD.C - ABCD.B))
  (perp_DC_BC : perpendicular (ABCD.C - ABCD.D) (ABCD.C - ABCD.B))
  (AB_length : distance ABCD.A ABCD.B = 15)
  (DC_length : distance ABCD.D ABCD.C = 6)
  (BC_length : distance ABCD.B ABCD.C = 10)
  (AB_eq_AD : distance ABCD.A ABCD.B = distance ABCD.A ABCD.D) :
  perimeter ABCD = 31 + Real.sqrt 181 := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_perimeter_l4042_404262


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l4042_404239

/-- Given an arithmetic sequence with common difference 3 where a₁, a₃, a₄ form a geometric sequence, a₂ = -6 -/
theorem arithmetic_geometric_sequence (a : ℕ → ℝ) :
  (∀ n, a (n + 1) - a n = 3) →  -- arithmetic sequence with common difference 3
  (a 3)^2 = a 1 * a 4 →         -- a₁, a₃, a₄ form a geometric sequence
  a 2 = -6 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l4042_404239


namespace NUMINAMATH_CALUDE_price_reduction_equation_correct_option_is_c_l4042_404256

/-- Represents the price reduction scenario -/
structure PriceReduction where
  initial_price : ℝ
  final_price : ℝ
  reduction_percentage : ℝ

/-- The equation correctly represents the price reduction scenario -/
def correct_equation (pr : PriceReduction) : Prop :=
  pr.initial_price * (1 - pr.reduction_percentage)^2 = pr.final_price

/-- Theorem stating that the equation correctly represents the given scenario -/
theorem price_reduction_equation :
  ∀ (pr : PriceReduction),
    pr.initial_price = 150 →
    pr.final_price = 96 →
    correct_equation pr :=
by
  sorry

/-- The correct option is C -/
theorem correct_option_is_c : 
  ∃ (pr : PriceReduction),
    pr.initial_price = 150 ∧
    pr.final_price = 96 ∧
    correct_equation pr :=
by
  sorry

end NUMINAMATH_CALUDE_price_reduction_equation_correct_option_is_c_l4042_404256


namespace NUMINAMATH_CALUDE_triangular_cross_section_solids_l4042_404242

-- Define the set of all possible solids
inductive Solid
  | Prism
  | Pyramid
  | Frustum
  | Cylinder
  | Cone
  | TruncatedCone
  | Sphere

-- Define a predicate for solids that can have a triangular cross-section
def hasTriangularCrossSection (s : Solid) : Prop :=
  match s with
  | Solid.Prism => true
  | Solid.Pyramid => true
  | Solid.Frustum => true
  | Solid.Cone => true
  | _ => false

-- Define the set of solids that can have a triangular cross-section
def solidsWithTriangularCrossSection : Set Solid :=
  {s : Solid | hasTriangularCrossSection s}

-- Theorem statement
theorem triangular_cross_section_solids :
  solidsWithTriangularCrossSection = {Solid.Prism, Solid.Pyramid, Solid.Frustum, Solid.Cone} :=
by sorry

end NUMINAMATH_CALUDE_triangular_cross_section_solids_l4042_404242


namespace NUMINAMATH_CALUDE_milk_water_mixture_volume_l4042_404245

theorem milk_water_mixture_volume 
  (initial_milk_percentage : Real)
  (final_milk_percentage : Real)
  (added_water : Real)
  (h1 : initial_milk_percentage = 0.84)
  (h2 : final_milk_percentage = 0.60)
  (h3 : added_water = 24)
  : ∃ initial_volume : Real,
    initial_volume * initial_milk_percentage = 
    (initial_volume + added_water) * final_milk_percentage ∧
    initial_volume = 60 := by
  sorry

end NUMINAMATH_CALUDE_milk_water_mixture_volume_l4042_404245


namespace NUMINAMATH_CALUDE_z_sixth_power_l4042_404293

theorem z_sixth_power (z : ℂ) : z = (-Real.sqrt 5 + Complex.I) / 2 → z^6 = -1 := by
  sorry

end NUMINAMATH_CALUDE_z_sixth_power_l4042_404293


namespace NUMINAMATH_CALUDE_iodine_mixture_theorem_l4042_404234

-- Define the given constants
def solution1_percentage : ℝ := 40
def solution2_volume : ℝ := 4.5
def final_mixture_volume : ℝ := 6
def final_mixture_percentage : ℝ := 50

-- Define the unknown percentage of the second solution
def solution2_percentage : ℝ := 26.67

-- Theorem statement
theorem iodine_mixture_theorem :
  solution1_percentage / 100 * solution2_volume + 
  solution2_percentage / 100 * solution2_volume = 
  final_mixture_percentage / 100 * final_mixture_volume := by
  sorry

end NUMINAMATH_CALUDE_iodine_mixture_theorem_l4042_404234


namespace NUMINAMATH_CALUDE_cube_surface_area_l4042_404249

/-- Given a cube with volume x^3, its surface area is 6x^2 -/
theorem cube_surface_area (x : ℝ) (h : x > 0) :
  (6 : ℝ) * x^2 = 6 * (x^3)^((2:ℝ)/3) := by
  sorry

end NUMINAMATH_CALUDE_cube_surface_area_l4042_404249


namespace NUMINAMATH_CALUDE_max_constant_inequality_l4042_404205

theorem max_constant_inequality (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x^2 + y^2 = 1) :
  ∃ (c : ℝ), c = 1/2 ∧ x^6 + y^6 ≥ c*x*y ∧ ∀ (d : ℝ), (∀ (a b : ℝ), a > 0 → b > 0 → a^2 + b^2 = 1 → a^6 + b^6 ≥ d*a*b) → d ≤ c :=
by sorry

end NUMINAMATH_CALUDE_max_constant_inequality_l4042_404205


namespace NUMINAMATH_CALUDE_smallest_coconut_pile_l4042_404210

def process (n : ℕ) : ℕ := (n - 1) * 4 / 5

def iterate_process (n : ℕ) (iterations : ℕ) : ℕ :=
  match iterations with
  | 0 => n
  | m + 1 => process (iterate_process n m)

theorem smallest_coconut_pile :
  ∃ (n : ℕ), n > 0 ∧ 
    (iterate_process n 5) % 5 = 0 ∧
    n ≥ (iterate_process n 0) - (iterate_process n 1) +
        (iterate_process n 1) - (iterate_process n 2) +
        (iterate_process n 2) - (iterate_process n 3) +
        (iterate_process n 3) - (iterate_process n 4) +
        (iterate_process n 4) - (iterate_process n 5) + 5 ∧
    (∀ (m : ℕ), m > 0 ∧ m < n →
      (iterate_process m 5) % 5 ≠ 0 ∨
      m < (iterate_process m 0) - (iterate_process m 1) +
          (iterate_process m 1) - (iterate_process m 2) +
          (iterate_process m 2) - (iterate_process m 3) +
          (iterate_process m 3) - (iterate_process m 4) +
          (iterate_process m 4) - (iterate_process m 5) + 5) ∧
    n = 3121 := by
  sorry

#check smallest_coconut_pile

end NUMINAMATH_CALUDE_smallest_coconut_pile_l4042_404210


namespace NUMINAMATH_CALUDE_correct_statements_count_l4042_404240

-- Define a structure to represent a statement
structure GeometricStatement :=
  (id : Nat)
  (content : String)
  (isCorrect : Bool)

-- Define the four statements
def statement1 : GeometricStatement :=
  { id := 1
  , content := "The prism with the least number of faces has 6 vertices"
  , isCorrect := true }

def statement2 : GeometricStatement :=
  { id := 2
  , content := "A frustum is the middle part of a cone cut by two parallel planes"
  , isCorrect := false }

def statement3 : GeometricStatement :=
  { id := 3
  , content := "A plane passing through the vertex of a cone cuts the cone into a section that is an isosceles triangle"
  , isCorrect := true }

def statement4 : GeometricStatement :=
  { id := 4
  , content := "Equal angles remain equal in perspective drawings"
  , isCorrect := false }

-- Define the list of all statements
def allStatements : List GeometricStatement :=
  [statement1, statement2, statement3, statement4]

-- Theorem to prove
theorem correct_statements_count :
  (allStatements.filter (·.isCorrect)).length = 2 := by
  sorry

end NUMINAMATH_CALUDE_correct_statements_count_l4042_404240


namespace NUMINAMATH_CALUDE_vehicle_dispatch_plans_l4042_404220

theorem vehicle_dispatch_plans (n : ℕ) (k : ℕ) : 
  n = 7 → k = 4 → (3 + 2 + 1) * (n - 2).factorial / ((n - 2 - (k - 2)).factorial) = 120 := by
  sorry

end NUMINAMATH_CALUDE_vehicle_dispatch_plans_l4042_404220


namespace NUMINAMATH_CALUDE_hotel_tax_calculation_l4042_404226

/-- Calculates the business tax paid given revenue and tax rate -/
def business_tax (revenue : ℕ) (tax_rate : ℚ) : ℚ :=
  (revenue : ℚ) * tax_rate

theorem hotel_tax_calculation :
  let revenue : ℕ := 10000000  -- 10 million yuan
  let tax_rate : ℚ := 5 / 100   -- 5%
  business_tax revenue tax_rate = 500 := by sorry

end NUMINAMATH_CALUDE_hotel_tax_calculation_l4042_404226


namespace NUMINAMATH_CALUDE_min_value_circle_line_l4042_404265

/-- The minimum value of 1/a + 4/b for a circle and a line passing through its center --/
theorem min_value_circle_line (a b : ℝ) : 
  a > 0 → b > 0 → a + b = 1 → 
  (∀ x y : ℝ, x^2 + y^2 + 4*x - 2*y - 1 = 0 → a*x - 2*b*y + 2 = 0) →
  (1/a + 4/b) ≥ 9 := by
sorry

end NUMINAMATH_CALUDE_min_value_circle_line_l4042_404265


namespace NUMINAMATH_CALUDE_intersection_condition_union_condition_l4042_404228

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | x^2 - 3*x + 2 = 0}
def B (a : ℝ) : Set ℝ := {x : ℝ | x^2 + 2*(a + 1)*x + (a^2 - 5) = 0}

-- Part 1: If A ∩ B = {2}, then a = -1 or a = -3
theorem intersection_condition (a : ℝ) : A ∩ B a = {2} → a = -1 ∨ a = -3 := by
  sorry

-- Part 2: If A ∪ B = A, then B ⊆ A
theorem union_condition (a : ℝ) : A ∪ B a = A → B a ⊆ A := by
  sorry

end NUMINAMATH_CALUDE_intersection_condition_union_condition_l4042_404228


namespace NUMINAMATH_CALUDE_decreasing_linear_function_l4042_404203

def linear_function (k b x : ℝ) : ℝ := k * x + b

theorem decreasing_linear_function (k b : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → linear_function k b x₁ > linear_function k b x₂) ↔ k < 0 :=
sorry

end NUMINAMATH_CALUDE_decreasing_linear_function_l4042_404203


namespace NUMINAMATH_CALUDE_volleyball_club_girls_count_l4042_404238

theorem volleyball_club_girls_count :
  ∀ (total_members : ℕ) (meeting_attendees : ℕ) (girls : ℕ) (boys : ℕ),
    total_members = 32 →
    meeting_attendees = 20 →
    total_members = girls + boys →
    meeting_attendees = boys + girls / 3 →
    girls = 18 := by
  sorry

end NUMINAMATH_CALUDE_volleyball_club_girls_count_l4042_404238


namespace NUMINAMATH_CALUDE_smallest_percentage_for_90_percent_l4042_404202

/-- Represents the distribution of money in a population -/
structure MoneyDistribution where
  /-- Percentage of people owning the majority of money -/
  rich_percentage : ℝ
  /-- Percentage of money owned by the rich -/
  rich_money_percentage : ℝ
  /-- Percentage of people needed to own a target percentage of money -/
  target_percentage : ℝ
  /-- Target percentage of money to be owned -/
  target_money_percentage : ℝ

/-- Theorem stating the smallest percentage of people that can be guaranteed to own 90% of all money -/
theorem smallest_percentage_for_90_percent 
  (d : MoneyDistribution) 
  (h1 : d.rich_percentage = 20)
  (h2 : d.rich_money_percentage ≥ 80)
  (h3 : d.target_money_percentage = 90) :
  d.target_percentage = 60 := by
  sorry

end NUMINAMATH_CALUDE_smallest_percentage_for_90_percent_l4042_404202


namespace NUMINAMATH_CALUDE_polynomial_remainder_theorem_l4042_404264

theorem polynomial_remainder_theorem (x : ℝ) : 
  (x^3 - 5*x^2 + 3*x - 7) % (x - 3) = -16 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_remainder_theorem_l4042_404264


namespace NUMINAMATH_CALUDE_shortest_chord_equation_l4042_404269

/-- Circle C -/
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 24 = 0

/-- Line l -/
def line_l (x y k : ℝ) : Prop := y = k*(x - 2) - 1

/-- Line AB -/
def line_AB (x y : ℝ) : Prop := x - y - 3 = 0

/-- The theorem statement -/
theorem shortest_chord_equation (k : ℝ) :
  (∃ A B : ℝ × ℝ, 
    (circle_C A.1 A.2 ∧ circle_C B.1 B.2) ∧ 
    (line_l A.1 A.2 k ∧ line_l B.1 B.2 k) ∧
    (∀ P Q : ℝ × ℝ, circle_C P.1 P.2 ∧ circle_C Q.1 Q.2 ∧ 
      line_l P.1 P.2 k ∧ line_l Q.1 Q.2 k →
      (A.1 - B.1)^2 + (A.2 - B.2)^2 ≤ (P.1 - Q.1)^2 + (P.2 - Q.2)^2)) →
  (∀ x y : ℝ, line_AB x y ↔ (circle_C x y ∧ line_l x y k)) :=
sorry

end NUMINAMATH_CALUDE_shortest_chord_equation_l4042_404269


namespace NUMINAMATH_CALUDE_solve_linear_equation_l4042_404253

theorem solve_linear_equation (x : ℝ) (h : 3 * x + 2 = 11) : 6 * x + 3 = 21 := by
  sorry

end NUMINAMATH_CALUDE_solve_linear_equation_l4042_404253


namespace NUMINAMATH_CALUDE_power_five_mod_150_l4042_404279

theorem power_five_mod_150 : 5^2023 % 150 = 5 := by
  sorry

end NUMINAMATH_CALUDE_power_five_mod_150_l4042_404279


namespace NUMINAMATH_CALUDE_square_area_ratio_l4042_404255

theorem square_area_ratio (a b : ℝ) (h : 4 * a = 16 * b) : a^2 = 16 * b^2 := by
  sorry

end NUMINAMATH_CALUDE_square_area_ratio_l4042_404255


namespace NUMINAMATH_CALUDE_amount_r_has_l4042_404207

theorem amount_r_has (total : ℝ) (r_fraction : ℝ) (h1 : total = 4000) (h2 : r_fraction = 2/3) : 
  let amount_pq := total / (1 + r_fraction)
  let amount_r := r_fraction * amount_pq
  amount_r = 1600 := by sorry

end NUMINAMATH_CALUDE_amount_r_has_l4042_404207


namespace NUMINAMATH_CALUDE_smaller_solution_of_quadratic_l4042_404252

theorem smaller_solution_of_quadratic (x : ℝ) : 
  x^2 + 20*x - 72 = 0 → (∃ y : ℝ, y^2 + 20*y - 72 = 0 ∧ y ≤ x) → x = -24 :=
by sorry

end NUMINAMATH_CALUDE_smaller_solution_of_quadratic_l4042_404252


namespace NUMINAMATH_CALUDE_complex_cosine_geometric_representation_l4042_404248

/-- The set of points represented by z = i cos θ, where θ ∈ [0, 2π], 
    is equal to the line segment from (0, -1) to (0, 1) in the complex plane. -/
theorem complex_cosine_geometric_representation :
  {z : ℂ | ∃ θ : ℝ, θ ∈ Set.Icc 0 (2 * Real.pi) ∧ z = Complex.I * Complex.cos θ} =
  {z : ℂ | z.re = 0 ∧ z.im ∈ Set.Icc (-1) 1} :=
sorry

end NUMINAMATH_CALUDE_complex_cosine_geometric_representation_l4042_404248


namespace NUMINAMATH_CALUDE_ceva_triangle_ratio_product_l4042_404212

/-- Given a triangle ABC with points A', B', C' on sides BC, AC, AB respectively,
    and lines AA', BB', CC' intersecting at point O, if the sum of the ratios
    AO/OA', BO/OB', and CO/OC' is 56, then the square of their product is 2916. -/
theorem ceva_triangle_ratio_product (A B C A' B' C' O : ℝ × ℝ) : 
  let ratio (P Q R : ℝ × ℝ) := dist P Q / dist Q R
  (ratio O A A' + ratio O B B' + ratio O C C' = 56) →
  (ratio O A A' * ratio O B B' * ratio O C C')^2 = 2916 := by
  sorry

end NUMINAMATH_CALUDE_ceva_triangle_ratio_product_l4042_404212


namespace NUMINAMATH_CALUDE_sarah_marriage_age_l4042_404299

def game_prediction (name_length : ℕ) (current_age : ℕ) : ℕ :=
  name_length + 2 * current_age

theorem sarah_marriage_age :
  game_prediction 5 9 = 23 := by
  sorry

end NUMINAMATH_CALUDE_sarah_marriage_age_l4042_404299


namespace NUMINAMATH_CALUDE_gcd_problem_l4042_404201

theorem gcd_problem (a : ℤ) (h : 1610 ∣ a) :
  Nat.gcd (Int.natAbs (a^2 + 9*a + 35)) (Int.natAbs (a + 5)) = 15 := by
  sorry

end NUMINAMATH_CALUDE_gcd_problem_l4042_404201


namespace NUMINAMATH_CALUDE_probability_king_queen_heart_l4042_404284

def standard_deck : ℕ := 52
def kings_in_deck : ℕ := 4
def queens_in_deck : ℕ := 4
def hearts_in_deck : ℕ := 13

theorem probability_king_queen_heart :
  let p_king : ℚ := kings_in_deck / standard_deck
  let p_queen_given_king : ℚ := queens_in_deck / (standard_deck - 1)
  let p_heart_given_king_queen : ℚ := hearts_in_deck / (standard_deck - 2)
  p_king * p_queen_given_king * p_heart_given_king_queen = 8 / 5525 := by
  sorry

end NUMINAMATH_CALUDE_probability_king_queen_heart_l4042_404284


namespace NUMINAMATH_CALUDE_little_red_journey_l4042_404275

/-- The distance from Little Red's house to school in kilometers -/
def distance_to_school : ℝ := 1.5

/-- Little Red's average speed uphill in kilometers per hour -/
def speed_uphill : ℝ := 2

/-- Little Red's average speed downhill in kilometers per hour -/
def speed_downhill : ℝ := 3

/-- The total time taken for the journey in minutes -/
def total_time : ℝ := 18

/-- The system of equations describing Little Red's journey to school -/
def journey_equations (x y : ℝ) : Prop :=
  (speed_uphill / 60 * x + speed_downhill / 60 * y = distance_to_school) ∧
  (x + y = total_time)

theorem little_red_journey :
  ∀ x y : ℝ, journey_equations x y ↔
    (2 / 60 * x + 3 / 60 * y = 1.5) ∧ (x + y = 18) :=
sorry

end NUMINAMATH_CALUDE_little_red_journey_l4042_404275


namespace NUMINAMATH_CALUDE_suit_price_increase_l4042_404246

theorem suit_price_increase (original_price : ℝ) (discounted_price : ℝ) :
  original_price = 160 →
  discounted_price = 150 →
  ∃ (increase_percentage : ℝ),
    increase_percentage = 25 ∧
    discounted_price = (original_price * (1 + increase_percentage / 100)) * 0.75 :=
by sorry

end NUMINAMATH_CALUDE_suit_price_increase_l4042_404246


namespace NUMINAMATH_CALUDE_exist_numbers_with_digit_sum_property_l4042_404230

/-- Sum of digits function -/
def S (x : ℕ) : ℕ := sorry

/-- Theorem stating the existence of numbers satisfying the given conditions -/
theorem exist_numbers_with_digit_sum_property : 
  ∃ (a b c : ℕ), 
    S (a + b) < 5 ∧ 
    S (a + c) < 5 ∧ 
    S (b + c) < 5 ∧ 
    S (a + b + c) > 50 := by
  sorry

end NUMINAMATH_CALUDE_exist_numbers_with_digit_sum_property_l4042_404230


namespace NUMINAMATH_CALUDE_bamboo_nine_sections_l4042_404296

/-- Given an arithmetic sequence of 9 terms, prove that if the sum of the first 4 terms is 3
    and the sum of the last 3 terms is 4, then the 5th term is 67/66 -/
theorem bamboo_nine_sections 
  (a : ℕ → ℚ) 
  (h_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1))
  (h_sum_first_four : a 1 + a 2 + a 3 + a 4 = 3)
  (h_sum_last_three : a 7 + a 8 + a 9 = 4) :
  a 5 = 67 / 66 := by
sorry

end NUMINAMATH_CALUDE_bamboo_nine_sections_l4042_404296


namespace NUMINAMATH_CALUDE_product_is_real_product_is_imaginary_l4042_404244

/-- The product of two complex numbers is real if and only if ad + bc = 0 -/
theorem product_is_real (a b c d : ℝ) :
  (Complex.I * b + a) * (Complex.I * d + c) ∈ Set.range (Complex.ofReal) ↔ a * d + b * c = 0 := by
  sorry

/-- The product of two complex numbers is purely imaginary if and only if ac - bd = 0 -/
theorem product_is_imaginary (a b c d : ℝ) :
  ∃ (k : ℝ), (Complex.I * b + a) * (Complex.I * d + c) = Complex.I * k ↔ a * c - b * d = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_is_real_product_is_imaginary_l4042_404244


namespace NUMINAMATH_CALUDE_toys_ratio_saturday_to_wednesday_l4042_404241

/-- The number of rabbits Junior has -/
def num_rabbits : ℕ := 16

/-- The number of toys bought on Monday -/
def toys_monday : ℕ := 6

/-- The number of toys bought on Wednesday -/
def toys_wednesday : ℕ := 2 * toys_monday

/-- The number of toys bought on Friday -/
def toys_friday : ℕ := 4 * toys_monday

/-- The number of toys each rabbit has when split evenly -/
def toys_per_rabbit : ℕ := 3

/-- The total number of toys -/
def total_toys : ℕ := num_rabbits * toys_per_rabbit

/-- The number of toys bought on Saturday -/
def toys_saturday : ℕ := total_toys - (toys_monday + toys_wednesday + toys_friday)

theorem toys_ratio_saturday_to_wednesday :
  toys_saturday * 2 = toys_wednesday := by sorry

end NUMINAMATH_CALUDE_toys_ratio_saturday_to_wednesday_l4042_404241


namespace NUMINAMATH_CALUDE_sphere_surface_area_ratio_l4042_404272

theorem sphere_surface_area_ratio (r₁ r₂ : ℝ) (h : r₁ / r₂ = 1 / 2) :
  (4 * Real.pi * r₁^2) / (4 * Real.pi * r₂^2) = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_sphere_surface_area_ratio_l4042_404272


namespace NUMINAMATH_CALUDE_quadratic_zeros_imply_a_range_l4042_404261

/-- A quadratic function f(x) = x^2 - 2ax + 4 with parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x + 4

/-- The property that f has two zeros in the interval (1, +∞) -/
def has_two_zeros_after_one (a : ℝ) : Prop :=
  ∃ x y, 1 < x ∧ x < y ∧ f a x = 0 ∧ f a y = 0

/-- If f(x) = x^2 - 2ax + 4 has two zeros in (1, +∞), then 2 < a < 5/2 -/
theorem quadratic_zeros_imply_a_range (a : ℝ) : 
  has_two_zeros_after_one a → 2 < a ∧ a < 5/2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_zeros_imply_a_range_l4042_404261


namespace NUMINAMATH_CALUDE_sum_positive_implies_at_least_one_positive_l4042_404263

theorem sum_positive_implies_at_least_one_positive (a b : ℝ) :
  a + b > 0 → (a > 0 ∨ b > 0) := by
  sorry

end NUMINAMATH_CALUDE_sum_positive_implies_at_least_one_positive_l4042_404263


namespace NUMINAMATH_CALUDE_tracis_road_trip_l4042_404216

theorem tracis_road_trip (D : ℝ) : 
  (1/3 : ℝ) * D + (1/4 : ℝ) * (2/3 : ℝ) * D + 300 = D → D = 600 :=
by sorry

end NUMINAMATH_CALUDE_tracis_road_trip_l4042_404216


namespace NUMINAMATH_CALUDE_wilsons_theorem_l4042_404217

theorem wilsons_theorem (p : ℕ) (hp : p > 1) :
  Nat.Prime p ↔ (Nat.factorial (p - 1) % p = p - 1) := by
  sorry

end NUMINAMATH_CALUDE_wilsons_theorem_l4042_404217


namespace NUMINAMATH_CALUDE_polygon_diagonals_l4042_404231

theorem polygon_diagonals (n : ℕ) (h : n ≥ 3) : (n - 3 = 4) → n = 7 := by
  sorry

end NUMINAMATH_CALUDE_polygon_diagonals_l4042_404231


namespace NUMINAMATH_CALUDE_distribution_ways_for_problem_l4042_404297

/-- Represents a hotel with a fixed number of rooms -/
structure Hotel where
  numRooms : Nat
  maxPerRoom : Nat

/-- Represents a group of friends -/
structure FriendGroup where
  numFriends : Nat

/-- Calculates the number of ways to distribute friends in rooms -/
def distributionWays (h : Hotel) (f : FriendGroup) : Nat :=
  sorry

/-- The specific hotel in the problem -/
def problemHotel : Hotel :=
  { numRooms := 5, maxPerRoom := 2 }

/-- The specific friend group in the problem -/
def problemFriendGroup : FriendGroup :=
  { numFriends := 5 }

theorem distribution_ways_for_problem :
  distributionWays problemHotel problemFriendGroup = 2220 :=
sorry

end NUMINAMATH_CALUDE_distribution_ways_for_problem_l4042_404297


namespace NUMINAMATH_CALUDE_dividend_calculation_l4042_404287

theorem dividend_calculation (dividend quotient remainder : ℕ) : 
  dividend / 3 = quotient ∧ 
  dividend % 3 = remainder ∧ 
  quotient = 16 ∧ 
  remainder = 4 → 
  dividend = 52 := by
sorry

end NUMINAMATH_CALUDE_dividend_calculation_l4042_404287


namespace NUMINAMATH_CALUDE_number_puzzle_l4042_404281

theorem number_puzzle (x : ℤ) (h : x - 69 = 37) : x + 55 = 161 := by
  sorry

end NUMINAMATH_CALUDE_number_puzzle_l4042_404281


namespace NUMINAMATH_CALUDE_optimal_triangle_game_l4042_404280

open Real

/-- A triangle with vertices A, B, and C -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- The area of a triangle -/
def Triangle.area (t : Triangle) : ℝ := sorry

/-- A point inside a triangle -/
def pointInside (t : Triangle) (X : ℝ × ℝ) : Prop := sorry

/-- The sum of areas of three triangles formed by connecting a point to three pairs of points on the sides of the original triangle -/
def sumOfAreas (t : Triangle) (X : ℝ × ℝ) : ℝ := sorry

theorem optimal_triangle_game (t : Triangle) (h : t.area = 1) :
  ∃ (X : ℝ × ℝ), pointInside t X ∧ sumOfAreas t X = 1/3 ∧
  ∀ (Y : ℝ × ℝ), pointInside t Y → sumOfAreas t Y ≥ 1/3 := by sorry

end NUMINAMATH_CALUDE_optimal_triangle_game_l4042_404280


namespace NUMINAMATH_CALUDE_f_properties_l4042_404219

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*a*x + 2

def is_monotonic_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  (∀ x y, a ≤ x → x ≤ y → y ≤ b → f x ≤ f y) ∨
  (∀ x y, a ≤ x → x ≤ y → y ≤ b → f y ≤ f x)

theorem f_properties :
  (∀ x ∈ Set.Icc (-5 : ℝ) 5, f (-1) x ≤ 37) ∧
  (∃ x ∈ Set.Icc (-5 : ℝ) 5, f (-1) x = 37) ∧
  (∀ x ∈ Set.Icc (-5 : ℝ) 5, 1 ≤ f (-1) x) ∧
  (∃ x ∈ Set.Icc (-5 : ℝ) 5, f (-1) x = 1) ∧
  (∀ a : ℝ, is_monotonic_on (f a) (-5) 5 ↔ a ≤ -5 ∨ a ≥ 5) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l4042_404219


namespace NUMINAMATH_CALUDE_min_product_of_three_l4042_404209

def S : Finset Int := {-9, -7, -5, 0, 4, 6, 8}

theorem min_product_of_three (a b c : Int) (ha : a ∈ S) (hb : b ∈ S) (hc : c ∈ S) 
  (hab : a ≠ b) (hbc : b ≠ c) (hac : a ≠ c) :
  (∀ x y z : Int, x ∈ S → y ∈ S → z ∈ S → x ≠ y → y ≠ z → x ≠ z → 
    a * b * c ≤ x * y * z) → 
  a * b * c = -336 :=
sorry

end NUMINAMATH_CALUDE_min_product_of_three_l4042_404209


namespace NUMINAMATH_CALUDE_inequality_solution_l4042_404221

theorem inequality_solution (x : ℝ) : 
  -1 ≤ x ∧ x ≤ 1 ∧ (1 / Real.sqrt (1 - x) - 1 / Real.sqrt (1 + x) ≥ 1) →
  Real.sqrt (2 * Real.sqrt 3 - 3) ≤ x ∧ x < 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l4042_404221


namespace NUMINAMATH_CALUDE_students_satisfy_equation_unique_solution_l4042_404289

/-- The number of students in class 5A -/
def students : ℕ := 36

/-- The equation that describes the problem conditions -/
def problem_equation (x : ℕ) : Prop :=
  (x - 23) * 23 = (x - 13) * 13

/-- Theorem stating that the number of students in class 5A satisfies the problem conditions -/
theorem students_satisfy_equation : problem_equation students := by
  sorry

/-- Theorem stating that 36 is the unique solution to the problem -/
theorem unique_solution : ∀ x : ℕ, problem_equation x → x = students := by
  sorry

end NUMINAMATH_CALUDE_students_satisfy_equation_unique_solution_l4042_404289


namespace NUMINAMATH_CALUDE_evaluate_expression_l4042_404291

theorem evaluate_expression (x y z : ℚ) (hx : x = 1/4) (hy : y = 1/3) (hz : z = -3) :
  x^2 * y^3 * z^2 = 1/48 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l4042_404291


namespace NUMINAMATH_CALUDE_sqrt_fraction_simplification_l4042_404215

theorem sqrt_fraction_simplification :
  (Real.sqrt (7^2 + 24^2)) / (Real.sqrt (49 + 16)) = (25 * Real.sqrt 65) / 65 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_fraction_simplification_l4042_404215


namespace NUMINAMATH_CALUDE_terrell_lifting_equivalence_l4042_404223

/-- The number of times Terrell lifts the weights initially -/
def initial_lifts : ℕ := 10

/-- The weight of each dumbbell in the initial setup (in pounds) -/
def initial_weight : ℕ := 25

/-- The weight of each dumbbell in the new setup (in pounds) -/
def new_weight : ℕ := 20

/-- The number of dumbbells used in each lift -/
def num_dumbbells : ℕ := 2

/-- The number of times Terrell must lift the new weights to achieve the same total weight -/
def required_lifts : ℚ := 12.5

theorem terrell_lifting_equivalence :
  (num_dumbbells * initial_weight * initial_lifts : ℚ) = 
  (num_dumbbells * new_weight * required_lifts) :=
by sorry

end NUMINAMATH_CALUDE_terrell_lifting_equivalence_l4042_404223


namespace NUMINAMATH_CALUDE_subtract_like_terms_l4042_404273

theorem subtract_like_terms (a : ℝ) : 4 * a - 3 * a = a := by
  sorry

end NUMINAMATH_CALUDE_subtract_like_terms_l4042_404273


namespace NUMINAMATH_CALUDE_certain_number_proof_l4042_404277

theorem certain_number_proof (n : ℕ) : 
  (∃ k : ℕ, n = 127 * k + 6) →
  (∃ m : ℕ, 2037 = 127 * m + 5) →
  (∀ d : ℕ, d > 127 → (n % d ≠ 6 ∨ 2037 % d ≠ 5)) →
  n = 2038 := by
sorry

end NUMINAMATH_CALUDE_certain_number_proof_l4042_404277


namespace NUMINAMATH_CALUDE_committee_problem_l4042_404274

/-- The number of ways to form a committee with the given constraints -/
def committee_formations (n m k r : ℕ) : ℕ :=
  Nat.choose n k - Nat.choose (n - m) k

theorem committee_problem :
  let total_members : ℕ := 30
  let founding_members : ℕ := 10
  let committee_size : ℕ := 5
  committee_formations total_members founding_members committee_size = 126992 := by
  sorry

end NUMINAMATH_CALUDE_committee_problem_l4042_404274


namespace NUMINAMATH_CALUDE_milk_volume_calculation_l4042_404267

def milk_volumes : List ℝ := [2.35, 1.75, 0.9, 0.75, 0.5, 0.325, 0.25]

theorem milk_volume_calculation :
  let total_volume := milk_volumes.sum
  let average_volume := total_volume / milk_volumes.length
  total_volume = 6.825 ∧ average_volume = 0.975 := by sorry

end NUMINAMATH_CALUDE_milk_volume_calculation_l4042_404267


namespace NUMINAMATH_CALUDE_water_added_calculation_l4042_404260

def initial_volume : ℝ := 340
def water_percentage : ℝ := 0.88
def cola_percentage : ℝ := 0.05
def sugar_percentage : ℝ := 1 - water_percentage - cola_percentage
def added_sugar : ℝ := 3.2
def added_cola : ℝ := 6.8
def final_sugar_percentage : ℝ := 0.075

theorem water_added_calculation (water_added : ℝ) : 
  (sugar_percentage * initial_volume + added_sugar) / 
  (initial_volume + added_sugar + added_cola + water_added) = final_sugar_percentage → 
  water_added = 10 := by
  sorry

end NUMINAMATH_CALUDE_water_added_calculation_l4042_404260


namespace NUMINAMATH_CALUDE_dice_probability_l4042_404229

def num_dice : ℕ := 8
def num_even : ℕ := 5
def num_not_five : ℕ := 3
def sides_per_die : ℕ := 6

def prob_even : ℚ := 1 / 2
def prob_not_five : ℚ := 5 / 6

def probability_exactly_even_and_not_five : ℚ :=
  (Nat.choose num_dice num_even) *
  (prob_even ^ num_even) *
  (prob_not_five ^ num_not_five)

theorem dice_probability :
  probability_exactly_even_and_not_five = 125 / 126 := by
  sorry

end NUMINAMATH_CALUDE_dice_probability_l4042_404229


namespace NUMINAMATH_CALUDE_div_mul_calculation_l4042_404211

theorem div_mul_calculation : (120 / 5) / 3 * 2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_div_mul_calculation_l4042_404211


namespace NUMINAMATH_CALUDE_equation_solutions_l4042_404225

theorem equation_solutions : 
  let f (x : ℝ) := (15*x - x^2)/(x + 1) * (x + (15 - x)/(x + 1))
  ∀ x : ℝ, f x = 60 ↔ x = 5 ∨ x = 6 ∨ x = 3 + Real.sqrt 2 ∨ x = 3 - Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l4042_404225


namespace NUMINAMATH_CALUDE_students_interested_in_both_sports_and_music_l4042_404236

/-- Given a class with the following properties:
  * There are 55 students in total
  * 43 students are sports enthusiasts
  * 34 students are music enthusiasts
  * 4 students are neither interested in sports nor music
  Prove that 26 students are interested in both sports and music -/
theorem students_interested_in_both_sports_and_music 
  (total : ℕ) (sports : ℕ) (music : ℕ) (neither : ℕ) 
  (h_total : total = 55)
  (h_sports : sports = 43)
  (h_music : music = 34)
  (h_neither : neither = 4) :
  sports + music - (total - neither) = 26 := by
  sorry

end NUMINAMATH_CALUDE_students_interested_in_both_sports_and_music_l4042_404236


namespace NUMINAMATH_CALUDE_arctan_sum_equals_pi_half_l4042_404206

theorem arctan_sum_equals_pi_half (n : ℕ+) :
  Real.arctan (1/2) + Real.arctan (1/3) + Real.arctan (1/7) + Real.arctan (1/n) = π/2 ↔ n = 4 :=
by sorry

end NUMINAMATH_CALUDE_arctan_sum_equals_pi_half_l4042_404206


namespace NUMINAMATH_CALUDE_circle_triangle_perpendiculars_l4042_404278

-- Define the basic structures
structure Point := (x y : ℝ)
structure Line := (a b c : ℝ)
structure Circle := (center : Point) (radius : ℝ)

-- Define the triangle
structure Triangle := (A B C : Point)

-- Define the intersection points
structure IntersectionPoints := 
  (A₁ A₂ B₁ B₂ C₁ C₂ : Point)

-- Define a function to check if three lines are concurrent
def are_concurrent (l₁ l₂ l₃ : Line) : Prop := sorry

-- Define a function to create a perpendicular line
def perpendicular_at (l : Line) (p : Point) : Line := sorry

-- Main theorem
theorem circle_triangle_perpendiculars 
  (triangle : Triangle) 
  (circle : Circle) 
  (intersections : IntersectionPoints) : 
  are_concurrent 
    (perpendicular_at (Line.mk 0 1 0) intersections.A₁)
    (perpendicular_at (Line.mk 1 0 0) intersections.B₁)
    (perpendicular_at (Line.mk 1 1 0) intersections.C₁) →
  are_concurrent 
    (perpendicular_at (Line.mk 0 1 0) intersections.A₂)
    (perpendicular_at (Line.mk 1 0 0) intersections.B₂)
    (perpendicular_at (Line.mk 1 1 0) intersections.C₂) :=
by
  sorry

end NUMINAMATH_CALUDE_circle_triangle_perpendiculars_l4042_404278


namespace NUMINAMATH_CALUDE_sum_of_distinct_integers_l4042_404251

theorem sum_of_distinct_integers (a b c d e : ℤ) :
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧
  c ≠ d ∧ c ≠ e ∧
  d ≠ e →
  (7 - a) * (7 - b) * (7 - c) * (7 - d) * (7 - e) = 120 →
  a + b + c + d + e = 33 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_distinct_integers_l4042_404251


namespace NUMINAMATH_CALUDE_semicircle_to_cone_volume_l4042_404254

theorem semicircle_to_cone_volume (R : ℝ) (h : R > 0) :
  let semicircle_radius := R
  let cone_base_radius := R / 2
  let cone_height := (Real.sqrt 3 / 2) * R
  let cone_volume := (1 / 3) * Real.pi * cone_base_radius^2 * cone_height
  cone_volume = (Real.sqrt 3 / 24) * Real.pi * R^3 :=
by sorry

end NUMINAMATH_CALUDE_semicircle_to_cone_volume_l4042_404254


namespace NUMINAMATH_CALUDE_max_value_quadratic_constraint_l4042_404292

theorem max_value_quadratic_constraint (x y z w : ℝ) 
  (h : 9*x^2 + 4*y^2 + 25*z^2 + 16*w^2 = 4) : 
  (∃ (a b c d : ℝ), 9*a^2 + 4*b^2 + 25*c^2 + 16*d^2 = 4 ∧ 
  2*a + 3*b + 5*c - 4*d = 6*Real.sqrt 6) ∧ 
  (∀ (x y z w : ℝ), 9*x^2 + 4*y^2 + 25*z^2 + 16*w^2 = 4 → 
  2*x + 3*y + 5*z - 4*w ≤ 6*Real.sqrt 6) := by
sorry

end NUMINAMATH_CALUDE_max_value_quadratic_constraint_l4042_404292


namespace NUMINAMATH_CALUDE_gmat_question_percentages_l4042_404243

theorem gmat_question_percentages :
  ∀ (first_correct second_correct both_correct neither_correct : ℝ),
    first_correct = 85 →
    neither_correct = 5 →
    both_correct = 55 →
    second_correct = 100 - neither_correct - (first_correct - both_correct) →
    second_correct = 65 := by
  sorry

end NUMINAMATH_CALUDE_gmat_question_percentages_l4042_404243


namespace NUMINAMATH_CALUDE_notebook_cost_l4042_404214

theorem notebook_cost (total_students : Nat) (buyers : Nat) (notebooks_per_student : Nat) (cost_per_notebook : Nat) 
  (h1 : total_students = 36)
  (h2 : buyers > total_students / 2)
  (h3 : notebooks_per_student > 2)
  (h4 : cost_per_notebook > notebooks_per_student)
  (h5 : buyers * notebooks_per_student * cost_per_notebook = 2601) :
  cost_per_notebook = 289 := by
  sorry

end NUMINAMATH_CALUDE_notebook_cost_l4042_404214


namespace NUMINAMATH_CALUDE_woman_work_time_l4042_404208

-- Define the work rate of one man
def man_rate : ℚ := 1 / 100

-- Define the total work (1 unit)
def total_work : ℚ := 1

-- Define the time taken by 10 men and 15 women
def combined_time : ℚ := 5

-- Define the number of men and women
def num_men : ℕ := 10
def num_women : ℕ := 15

-- Define the work rate of one woman
noncomputable def woman_rate : ℚ := 
  (total_work / combined_time - num_men * man_rate) / num_women

-- Theorem: One woman alone will take 150 days to complete the work
theorem woman_work_time : total_work / woman_rate = 150 := by sorry

end NUMINAMATH_CALUDE_woman_work_time_l4042_404208


namespace NUMINAMATH_CALUDE_max_min_difference_l4042_404285

theorem max_min_difference (a b c : ℝ) (h : a^2 + b^2 + c^2 = 1) : 
  let f := fun (x y z : ℝ) => x*y + y*z + z*x
  ∃ (M m : ℝ), (∀ (x y z : ℝ), x^2 + y^2 + z^2 = 1 → f x y z ≤ M) ∧
               (∀ (x y z : ℝ), x^2 + y^2 + z^2 = 1 → m ≤ f x y z) ∧
               M - m = 3/2 :=
by sorry

end NUMINAMATH_CALUDE_max_min_difference_l4042_404285


namespace NUMINAMATH_CALUDE_min_value_sin_2x_minus_pi_4_l4042_404213

theorem min_value_sin_2x_minus_pi_4 :
  ∃ (min : ℝ), min = -Real.sqrt 2 / 2 ∧
  ∀ x : ℝ, x ∈ Set.Icc 0 (Real.pi / 2) →
  Real.sin (2 * x - Real.pi / 4) ≥ min :=
sorry

end NUMINAMATH_CALUDE_min_value_sin_2x_minus_pi_4_l4042_404213


namespace NUMINAMATH_CALUDE_orange_harvest_l4042_404237

theorem orange_harvest (days : ℕ) (total_sacks : ℕ) (sacks_per_day : ℕ) : 
  days = 6 → total_sacks = 498 → sacks_per_day * days = total_sacks → sacks_per_day = 83 := by
  sorry

end NUMINAMATH_CALUDE_orange_harvest_l4042_404237


namespace NUMINAMATH_CALUDE_year2018_is_WuXu_l4042_404276

/-- Represents the Ten Heavenly Stems -/
inductive HeavenlyStem
| Jia | Yi | Bing | Ding | Wu | Ji | Geng | Xin | Ren | Gui

/-- Represents the Twelve Earthly Branches -/
inductive EarthlyBranch
| Zi | Chou | Yin | Mao | Chen | Si | Wu | Wei | Shen | You | Xu | Hai

/-- Represents a year in the Sexagenary Cycle -/
structure SexagenaryYear :=
  (stem : HeavenlyStem)
  (branch : EarthlyBranch)

/-- The Sexagenary Cycle -/
def SexagenaryCycle : List SexagenaryYear := sorry

/-- Function to get the next year in the Sexagenary Cycle -/
def nextYear (year : SexagenaryYear) : SexagenaryYear := sorry

/-- 2016 is the Bing Shen year -/
def year2016 : SexagenaryYear :=
  { stem := HeavenlyStem.Bing, branch := EarthlyBranch.Shen }

/-- Theorem: 2018 is the Wu Xu year in the Sexagenary Cycle -/
theorem year2018_is_WuXu :
  (nextYear (nextYear year2016)) = { stem := HeavenlyStem.Wu, branch := EarthlyBranch.Xu } := by
  sorry


end NUMINAMATH_CALUDE_year2018_is_WuXu_l4042_404276


namespace NUMINAMATH_CALUDE_mileage_scientific_notation_equality_l4042_404232

-- Define the original mileage
def original_mileage : ℝ := 42000

-- Define the scientific notation representation
def scientific_notation : ℝ := 4.2 * (10^4)

-- Theorem to prove the equality
theorem mileage_scientific_notation_equality :
  original_mileage = scientific_notation :=
by sorry

end NUMINAMATH_CALUDE_mileage_scientific_notation_equality_l4042_404232


namespace NUMINAMATH_CALUDE_inequality_system_solution_l4042_404283

theorem inequality_system_solution (x : ℝ) : 
  ((x - 1) / (x + 2) ≤ 0 ∧ x^2 - 2*x - 3 < 0) ↔ -1 < x ∧ x ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l4042_404283


namespace NUMINAMATH_CALUDE_ratio_of_distances_l4042_404222

/-- Given five consecutive points on a line, prove the ratio of two specific distances -/
theorem ratio_of_distances (E F G H I : ℝ) (hEF : |E - F| = 3) (hFG : |F - G| = 6) 
  (hGH : |G - H| = 4) (hHI : |H - I| = 2) (hOrder : E < F ∧ F < G ∧ G < H ∧ H < I) : 
  |E - G| / |H - I| = 9/2 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_distances_l4042_404222


namespace NUMINAMATH_CALUDE_expression_equals_sum_l4042_404247

theorem expression_equals_sum (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  let numerator := a^2 * (1/b - 1/c) + b^2 * (1/c - 1/a) + c^2 * (1/a - 1/b)
  let denominator := a * (1/b - 1/c) + b * (1/c - 1/a) + c * (1/a - 1/b)
  numerator / denominator = a + b + c := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_sum_l4042_404247


namespace NUMINAMATH_CALUDE_octagon_diagonals_l4042_404258

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- An octagon has 8 sides -/
def octagon_sides : ℕ := 8

theorem octagon_diagonals :
  num_diagonals octagon_sides = 20 := by
  sorry

end NUMINAMATH_CALUDE_octagon_diagonals_l4042_404258


namespace NUMINAMATH_CALUDE_arithmetic_computation_l4042_404290

theorem arithmetic_computation : -10 * 3 - (-4 * -2) + (-12 * -4) / 2 = -14 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_computation_l4042_404290


namespace NUMINAMATH_CALUDE_lauren_mail_count_l4042_404286

/-- The number of pieces of mail Lauren sent on Monday -/
def monday_mail : ℕ := 65

/-- The number of pieces of mail Lauren sent on Tuesday -/
def tuesday_mail : ℕ := monday_mail + 10

/-- The number of pieces of mail Lauren sent on Wednesday -/
def wednesday_mail : ℕ := tuesday_mail - 5

/-- The number of pieces of mail Lauren sent on Thursday -/
def thursday_mail : ℕ := wednesday_mail + 15

/-- The total number of pieces of mail Lauren sent over the four days -/
def total_mail : ℕ := monday_mail + tuesday_mail + wednesday_mail + thursday_mail

theorem lauren_mail_count : total_mail = 295 := by
  sorry

end NUMINAMATH_CALUDE_lauren_mail_count_l4042_404286


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l4042_404282

/-- Given an arithmetic sequence {a_n} where a₂ = 3a₅ - 6, prove that S₉ = 27 -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (∀ n, a (n + 1) - a n = a 2 - a 1) →  -- arithmetic sequence condition
  (∀ n, S n = (n * (a 1 + a n)) / 2) →  -- sum formula for arithmetic sequence
  a 2 = 3 * a 5 - 6 →                   -- given condition
  S 9 = 27 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l4042_404282


namespace NUMINAMATH_CALUDE_shop_discount_percentage_l4042_404233

/-- Calculate the percentage discount given the original price and discounted price -/
def calculate_discount_percentage (original_price discounted_price : ℚ) : ℚ :=
  (original_price - discounted_price) / original_price * 100

/-- The shop's discount percentage is 30% -/
theorem shop_discount_percentage :
  let original_price : ℚ := 800
  let discounted_price : ℚ := 560
  calculate_discount_percentage original_price discounted_price = 30 := by
sorry

end NUMINAMATH_CALUDE_shop_discount_percentage_l4042_404233


namespace NUMINAMATH_CALUDE_four_integer_solutions_l4042_404259

theorem four_integer_solutions : 
  ∃! (s : Finset (ℤ × ℤ)), 
    (∀ (p : ℤ × ℤ), p ∈ s ↔ p.1^2022 + p.2^2 = 4*p.2) ∧ 
    s.card = 4 := by
  sorry

end NUMINAMATH_CALUDE_four_integer_solutions_l4042_404259


namespace NUMINAMATH_CALUDE_equation_solutions_l4042_404268

def has_different_divisors (a b : ℤ) : Prop :=
  ∃ d : ℤ, (d ∣ a ∧ ¬(d ∣ b)) ∨ (d ∣ b ∧ ¬(d ∣ a))

theorem equation_solutions :
  ∀ a b : ℤ, has_different_divisors a b → a^2 + a = b^3 + b →
  ((a = 1 ∧ b = 1) ∨ (a = -2 ∧ b = 1) ∨ (a = 5 ∧ b = 3)) :=
sorry

end NUMINAMATH_CALUDE_equation_solutions_l4042_404268


namespace NUMINAMATH_CALUDE_triangle_perimeter_range_l4042_404266

theorem triangle_perimeter_range (a b c A B C : ℝ) : 
  0 < A ∧ A < π / 2 →
  0 < B ∧ B < π / 2 →
  0 < C ∧ C < π / 2 →
  c = 2 →
  a * Real.cos B + b * Real.cos A = (Real.sqrt 3 * c) / (2 * Real.sin C) →
  A + B + C = π →
  let P := a + b + c
  4 < P ∧ P ≤ 6 := by
  sorry

end NUMINAMATH_CALUDE_triangle_perimeter_range_l4042_404266


namespace NUMINAMATH_CALUDE_lighthouse_ship_position_l4042_404224

/-- Represents cardinal directions --/
inductive Direction
  | North
  | South
  | East
  | West

/-- Represents a relative position with a direction and angle --/
structure RelativePosition where
  primaryDirection : Direction
  secondaryDirection : Direction
  angle : ℝ

/-- Returns the opposite direction --/
def oppositeDirection (d : Direction) : Direction :=
  match d with
  | Direction.North => Direction.South
  | Direction.South => Direction.North
  | Direction.East => Direction.West
  | Direction.West => Direction.East

/-- Returns the opposite relative position --/
def oppositePosition (pos : RelativePosition) : RelativePosition :=
  { primaryDirection := oppositeDirection pos.primaryDirection,
    secondaryDirection := oppositeDirection pos.secondaryDirection,
    angle := pos.angle }

theorem lighthouse_ship_position 
  (lighthousePos : RelativePosition) 
  (h1 : lighthousePos.primaryDirection = Direction.North)
  (h2 : lighthousePos.secondaryDirection = Direction.East)
  (h3 : lighthousePos.angle = 38) :
  oppositePosition lighthousePos = 
    { primaryDirection := Direction.South,
      secondaryDirection := Direction.West,
      angle := 38 } := by
  sorry

end NUMINAMATH_CALUDE_lighthouse_ship_position_l4042_404224


namespace NUMINAMATH_CALUDE_vector_subtraction_scalar_multiplication_l4042_404294

theorem vector_subtraction_scalar_multiplication :
  let v1 : Fin 2 → ℝ := ![3, -8]
  let v2 : Fin 2 → ℝ := ![-2, 6]
  let scalar : ℝ := 5
  v1 - scalar • v2 = ![13, -38] := by
  sorry

end NUMINAMATH_CALUDE_vector_subtraction_scalar_multiplication_l4042_404294


namespace NUMINAMATH_CALUDE_total_jewelry_is_83_l4042_404235

/-- Represents the initial jewelry counts and purchase rules --/
structure JewelryInventory where
  initial_necklaces : ℕ
  initial_earrings : ℕ
  initial_bracelets : ℕ
  initial_rings : ℕ
  store_a_necklaces : ℕ
  store_a_bracelets : ℕ
  store_b_necklaces : ℕ

/-- Calculates the total number of jewelry pieces after all additions --/
def totalJewelryPieces (inventory : JewelryInventory) : ℕ :=
  let store_a_earrings := (2 * inventory.initial_earrings) / 3
  let store_b_rings := 2 * inventory.initial_rings
  let mother_gift_earrings := store_a_earrings / 5
  
  inventory.initial_necklaces + inventory.initial_earrings + inventory.initial_bracelets + inventory.initial_rings +
  inventory.store_a_necklaces + store_a_earrings + inventory.store_a_bracelets +
  inventory.store_b_necklaces + store_b_rings +
  mother_gift_earrings

/-- Theorem stating that the total jewelry pieces is 83 given the specific inventory --/
theorem total_jewelry_is_83 :
  totalJewelryPieces ⟨10, 15, 5, 8, 10, 3, 4⟩ = 83 := by
  sorry

end NUMINAMATH_CALUDE_total_jewelry_is_83_l4042_404235


namespace NUMINAMATH_CALUDE_books_loaned_out_l4042_404200

/-- Proves that the number of books loaned out is 50, given the initial number of books,
    the return rate, and the final number of books. -/
theorem books_loaned_out
  (initial_books : ℕ)
  (return_rate : ℚ)
  (final_books : ℕ)
  (h1 : initial_books = 75)
  (h2 : return_rate = 4/5)
  (h3 : final_books = 65) :
  ∃ (loaned_books : ℕ), loaned_books = 50 ∧
    final_books = initial_books - (1 - return_rate) * loaned_books :=
by sorry

end NUMINAMATH_CALUDE_books_loaned_out_l4042_404200


namespace NUMINAMATH_CALUDE_parabola_focus_coordinates_l4042_404270

/-- Given a parabola with equation y = 2x^2 + 8x - 1, its focus coordinates are (-2, -8.875) -/
theorem parabola_focus_coordinates :
  let f : ℝ → ℝ := λ x => 2 * x^2 + 8 * x - 1
  ∃ (h k : ℝ), h = -2 ∧ k = -8.875 ∧
    ∀ (x y : ℝ), y = f x → (x - h)^2 = 4 * (y - k) / 2 := by
  sorry

end NUMINAMATH_CALUDE_parabola_focus_coordinates_l4042_404270


namespace NUMINAMATH_CALUDE_trajectory_characterization_l4042_404271

-- Define the fixed points
def F₁ : ℝ × ℝ := (-5, 0)
def F₂ : ℝ × ℝ := (5, 0)

-- Define the condition for point P
def satisfies_condition (P : ℝ × ℝ) (a : ℝ) : Prop :=
  |P.1 - F₁.1| + |P.2 - F₁.2| - (|P.1 - F₂.1| + |P.2 - F₂.2|) = 2 * a

-- Define what it means to be on one branch of a hyperbola
def on_hyperbola_branch (P : ℝ × ℝ) : Prop :=
  ∃ (a : ℝ), a > 0 ∧ satisfies_condition P a ∧ 
  (P.1 < -5 ∨ (P.1 > 5 ∧ P.2 ≠ 0))

-- Define what it means to be on a ray starting from (5, 0) in positive x direction
def on_positive_x_ray (P : ℝ × ℝ) : Prop :=
  P.2 = 0 ∧ P.1 ≥ 5

theorem trajectory_characterization :
  (∀ P : ℝ × ℝ, satisfies_condition P 3 → on_hyperbola_branch P) ∧
  (∀ P : ℝ × ℝ, satisfies_condition P 5 → on_positive_x_ray P) :=
sorry

end NUMINAMATH_CALUDE_trajectory_characterization_l4042_404271


namespace NUMINAMATH_CALUDE_power_calculation_l4042_404298

theorem power_calculation : 3^18 / 27^3 * 9 = 177147 := by
  sorry

end NUMINAMATH_CALUDE_power_calculation_l4042_404298


namespace NUMINAMATH_CALUDE_sam_football_games_l4042_404218

/-- The number of football games Sam went to this year -/
def games_this_year : ℕ := 43 - 29

/-- Theorem stating that Sam went to 14 football games this year -/
theorem sam_football_games : games_this_year = 14 := by
  sorry

end NUMINAMATH_CALUDE_sam_football_games_l4042_404218


namespace NUMINAMATH_CALUDE_always_two_real_roots_one_nonnegative_root_iff_l4042_404204

-- Define the quadratic equation
def quadratic (m : ℝ) (x : ℝ) : ℝ := x^2 + (4-m)*x + (3-m)

-- Theorem 1: The equation always has two real roots
theorem always_two_real_roots (m : ℝ) :
  ∃ (x1 x2 : ℝ), quadratic m x1 = 0 ∧ quadratic m x2 = 0 :=
sorry

-- Theorem 2: The equation has exactly one non-negative real root iff m ≥ 3
theorem one_nonnegative_root_iff (m : ℝ) :
  (∃! (x : ℝ), x ≥ 0 ∧ quadratic m x = 0) ↔ m ≥ 3 :=
sorry

end NUMINAMATH_CALUDE_always_two_real_roots_one_nonnegative_root_iff_l4042_404204


namespace NUMINAMATH_CALUDE_unique_function_theorem_l4042_404288

/-- A function from positive integers to positive integers -/
def PositiveIntFunction := ℕ+ → ℕ+

/-- Condition: f(n) is a perfect square for all n -/
def IsPerfectSquare (f : PositiveIntFunction) : Prop :=
  ∀ n : ℕ+, ∃ k : ℕ+, f n = k * k

/-- Condition: f(m+n) = f(m) + f(n) + 2mn for all m, n -/
def SatisfiesFunctionalEquation (f : PositiveIntFunction) : Prop :=
  ∀ m n : ℕ+, f (m + n) = f m + f n + 2 * m * n

/-- Theorem: The only function satisfying both conditions is f(n) = n² -/
theorem unique_function_theorem (f : PositiveIntFunction) 
  (h1 : IsPerfectSquare f) (h2 : SatisfiesFunctionalEquation f) :
  ∀ n : ℕ+, f n = n * n :=
by sorry

end NUMINAMATH_CALUDE_unique_function_theorem_l4042_404288


namespace NUMINAMATH_CALUDE_angle_rotation_and_trig_identity_l4042_404250

theorem angle_rotation_and_trig_identity 
  (initial_angle : Real) 
  (rotations : Nat) 
  (α : Real) 
  (h1 : initial_angle = 30 * Real.pi / 180)
  (h2 : rotations = 3)
  (h3 : Real.sin (-Real.pi/2 - α) = -1/3)
  (h4 : Real.tan α < 0) :
  (initial_angle + rotations * 2 * Real.pi) * 180 / Real.pi = 1110 ∧ 
  Real.cos (3 * Real.pi / 2 + α) = -2 * Real.sqrt 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_angle_rotation_and_trig_identity_l4042_404250


namespace NUMINAMATH_CALUDE_chessboard_game_stone_range_min_le_max_stones_l4042_404257

/-- A game on an n × n chessboard with k stones -/
def ChessboardGame (n : ℕ) (k : ℕ) : Prop :=
  n > 0 ∧ k ≥ 2 * n^2 - 2 * n ∧ k ≤ 3 * n^2 - 4 * n

/-- The theorem stating the range of stones for the game -/
theorem chessboard_game_stone_range (n : ℕ) :
  n > 0 → ∀ k, ChessboardGame n k ↔ 2 * n^2 - 2 * n ≤ k ∧ k ≤ 3 * n^2 - 4 * n :=
by sorry

/-- The minimum number of stones for the game -/
def min_stones (n : ℕ) : ℕ := 2 * n^2 - 2 * n

/-- The maximum number of stones for the game -/
def max_stones (n : ℕ) : ℕ := 3 * n^2 - 4 * n

/-- Theorem stating that the minimum number of stones is always less than or equal to the maximum -/
theorem min_le_max_stones (n : ℕ) : n > 0 → min_stones n ≤ max_stones n :=
by sorry

end NUMINAMATH_CALUDE_chessboard_game_stone_range_min_le_max_stones_l4042_404257


namespace NUMINAMATH_CALUDE_fraction_sum_equals_one_no_solution_to_equation_l4042_404227

-- Problem 1
theorem fraction_sum_equals_one (a b : ℝ) (h : a ≠ b) :
  a / (a - b) + b / (b - a) = 1 := by sorry

-- Problem 2
theorem no_solution_to_equation :
  ¬∃ x : ℝ, (1 / (x - 2) = (1 - x) / (2 - x) - 3) := by sorry

end NUMINAMATH_CALUDE_fraction_sum_equals_one_no_solution_to_equation_l4042_404227


namespace NUMINAMATH_CALUDE_tank_capacity_l4042_404295

theorem tank_capacity (initial_fullness : Rat) (final_fullness : Rat) (added_water : Rat) :
  initial_fullness = 1/4 →
  final_fullness = 2/3 →
  added_water = 120 →
  (final_fullness - initial_fullness) * (added_water / (final_fullness - initial_fullness)) = 288 :=
by
  sorry

#check tank_capacity

end NUMINAMATH_CALUDE_tank_capacity_l4042_404295
