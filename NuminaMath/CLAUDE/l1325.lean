import Mathlib

namespace NUMINAMATH_CALUDE_smallest_difference_l1325_132561

/-- Represents the first sequence in the table -/
def first_sequence (n : ℕ) : ℤ := 2 * n - 1

/-- Represents the second sequence in the table -/
def second_sequence (n : ℕ) : ℤ := 5055 - 5 * n

/-- The difference between the two sequences at position n -/
def difference (n : ℕ) : ℤ := (second_sequence n) - (first_sequence n)

/-- The number of terms in each sequence -/
def sequence_length : ℕ := 1010

theorem smallest_difference :
  ∃ (k : ℕ), k ≤ sequence_length ∧ difference k = 2 ∧
  ∀ (n : ℕ), n ≤ sequence_length → difference n ≥ 2 :=
sorry

end NUMINAMATH_CALUDE_smallest_difference_l1325_132561


namespace NUMINAMATH_CALUDE_pure_imaginary_complex_fraction_l1325_132564

/-- If z = (a + i) / (1 - i) is a pure imaginary number and a is real, then a = 1 -/
theorem pure_imaginary_complex_fraction (a : ℝ) : 
  let z : ℂ := (a + Complex.I) / (1 - Complex.I)
  (∃ b : ℝ, z = Complex.I * b) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_complex_fraction_l1325_132564


namespace NUMINAMATH_CALUDE_no_real_solution_l1325_132541

theorem no_real_solution : ¬∃ (x : ℝ), 
  (x + 5 > 0) ∧ 
  (x - 3 > 0) ∧ 
  (x^2 - 8*x + 7 > 0) ∧ 
  (Real.log (x + 5) + Real.log (x - 3) = Real.log (x^2 - 8*x + 7)) := by
sorry

end NUMINAMATH_CALUDE_no_real_solution_l1325_132541


namespace NUMINAMATH_CALUDE_intersection_subset_iff_m_eq_two_l1325_132506

/-- Sets A, B, and C as defined in the problem -/
def A : Set ℝ := {x | -4 < x ∧ x < 2}
def B : Set ℝ := {x | x > 1 ∨ x < -5}
def C (m : ℝ) : Set ℝ := {x | m - 1 < x ∧ x < m}

/-- Theorem stating that A ∩ B ⊆ C(m) if and only if m = 2 -/
theorem intersection_subset_iff_m_eq_two :
  ∀ m : ℝ, (A ∩ B) ⊆ C m ↔ m = 2 := by sorry

end NUMINAMATH_CALUDE_intersection_subset_iff_m_eq_two_l1325_132506


namespace NUMINAMATH_CALUDE_S_union_T_eq_S_l1325_132517

def S : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 * p.2 > 0}
def T : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 > 0 ∧ p.2 > 0}

theorem S_union_T_eq_S : S ∪ T = S := by sorry

end NUMINAMATH_CALUDE_S_union_T_eq_S_l1325_132517


namespace NUMINAMATH_CALUDE_zhang_san_not_losing_probability_l1325_132544

theorem zhang_san_not_losing_probability 
  (p_win : ℚ) (p_draw : ℚ) 
  (h_win : p_win = 1/3) 
  (h_draw : p_draw = 1/4) : 
  p_win + p_draw = 7/12 := by
  sorry

end NUMINAMATH_CALUDE_zhang_san_not_losing_probability_l1325_132544


namespace NUMINAMATH_CALUDE_hyperbola_k_range_l1325_132585

-- Define the hyperbola equation
def hyperbola_equation (x y k : ℝ) : Prop :=
  x^2 / (2 - k) + y^2 / (k - 1) = 1

-- Define the condition for the real axis to be on the x-axis
def real_axis_on_x (k : ℝ) : Prop :=
  (2 - k > 0) ∧ (k - 1 < 0)

-- Theorem statement
theorem hyperbola_k_range :
  ∀ k : ℝ, (∃ x y : ℝ, hyperbola_equation x y k ∧ real_axis_on_x k) ↔ k ∈ Set.Iio 1 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_k_range_l1325_132585


namespace NUMINAMATH_CALUDE_complex_cube_equation_l1325_132525

theorem complex_cube_equation :
  ∃! (z : ℂ), ∃ (x y c : ℤ), 
    x > 0 ∧ y > 0 ∧ 
    z = x + y * I ∧
    z^3 = -74 + c * I ∧
    z = 1 + 5 * I :=
by sorry

end NUMINAMATH_CALUDE_complex_cube_equation_l1325_132525


namespace NUMINAMATH_CALUDE_tower_lights_l1325_132507

/-- Represents the number of levels in the tower -/
def levels : ℕ := 7

/-- Represents the total number of lights on the tower -/
def totalLights : ℕ := 381

/-- Represents the common ratio between adjacent levels -/
def ratio : ℕ := 2

/-- Calculates the sum of a geometric sequence -/
def geometricSum (a : ℕ) (r : ℕ) (n : ℕ) : ℕ :=
  a * (r^n - 1) / (r - 1)

theorem tower_lights :
  ∃ (topLights : ℕ), 
    geometricSum topLights ratio levels = totalLights ∧ 
    topLights = 3 := by
  sorry

end NUMINAMATH_CALUDE_tower_lights_l1325_132507


namespace NUMINAMATH_CALUDE_star_value_l1325_132534

theorem star_value (star : ℝ) : star * 12^2 = 12^7 → star = 12^5 := by
  sorry

end NUMINAMATH_CALUDE_star_value_l1325_132534


namespace NUMINAMATH_CALUDE_a_minus_c_value_l1325_132583

theorem a_minus_c_value (a b c d : ℝ) 
  (h1 : (a + d + b + d) / 2 = 80)
  (h2 : (b + d + c + d) / 2 = 180)
  (h3 : d = 2 * (a - b)) : 
  a - c = -200 := by
sorry

end NUMINAMATH_CALUDE_a_minus_c_value_l1325_132583


namespace NUMINAMATH_CALUDE_intersection_perpendicular_implies_k_l1325_132518

/-- The line l: kx - y - 2 = 0 intersects the circle O: x^2 + y^2 = 4 at points A and B. -/
def intersects (k : ℝ) (A B : ℝ × ℝ) : Prop :=
  k * A.1 - A.2 - 2 = 0 ∧
  k * B.1 - B.2 - 2 = 0 ∧
  A.1^2 + A.2^2 = 4 ∧
  B.1^2 + B.2^2 = 4

/-- The dot product of OA and OB is zero. -/
def perpendicular (A B : ℝ × ℝ) : Prop :=
  A.1 * B.1 + A.2 * B.2 = 0

/-- Main theorem: If the line intersects the circle and the intersection points are perpendicular from the origin, then k = ±1. -/
theorem intersection_perpendicular_implies_k (k : ℝ) (A B : ℝ × ℝ) 
  (h_intersects : intersects k A B) (h_perp : perpendicular A B) : k = 1 ∨ k = -1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_perpendicular_implies_k_l1325_132518


namespace NUMINAMATH_CALUDE_garden_bugs_l1325_132531

theorem garden_bugs (B : ℕ) : 0.8 * (B : ℝ) - 12 * 7 = 236 → B = 400 := by
  sorry

end NUMINAMATH_CALUDE_garden_bugs_l1325_132531


namespace NUMINAMATH_CALUDE_sum_of_T_l1325_132579

/-- The sum of the geometric series for -1 < r < 1 -/
noncomputable def T (r : ℝ) : ℝ := 18 / (1 - r)

/-- Theorem: Sum of T(b) and T(-b) equals 337.5 -/
theorem sum_of_T (b : ℝ) (h1 : -1 < b) (h2 : b < 1) (h3 : T b * T (-b) = 3024) :
  T b + T (-b) = 337.5 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_T_l1325_132579


namespace NUMINAMATH_CALUDE_bowling_ball_volume_l1325_132598

/-- The volume of a sphere with diameter 24 cm minus the volume of three cylindrical holes
    (with depths 6 cm and diameters 1.5 cm, 2.5 cm, and 3 cm respectively) is equal to 2239.5π cubic cm. -/
theorem bowling_ball_volume :
  let sphere_diameter : ℝ := 24
  let hole_depth : ℝ := 6
  let hole_diameter1 : ℝ := 1.5
  let hole_diameter2 : ℝ := 2.5
  let hole_diameter3 : ℝ := 3
  let sphere_volume := (4 / 3) * π * (sphere_diameter / 2) ^ 3
  let hole_volume1 := π * (hole_diameter1 / 2) ^ 2 * hole_depth
  let hole_volume2 := π * (hole_diameter2 / 2) ^ 2 * hole_depth
  let hole_volume3 := π * (hole_diameter3 / 2) ^ 2 * hole_depth
  let remaining_volume := sphere_volume - (hole_volume1 + hole_volume2 + hole_volume3)
  remaining_volume = 2239.5 * π :=
by sorry

end NUMINAMATH_CALUDE_bowling_ball_volume_l1325_132598


namespace NUMINAMATH_CALUDE_salt_solution_weight_salt_solution_weight_proof_l1325_132546

theorem salt_solution_weight (initial_concentration : Real) 
                             (final_concentration : Real) 
                             (added_salt : Real) 
                             (initial_weight : Real) : Prop :=
  initial_concentration = 0.10 ∧
  final_concentration = 0.20 ∧
  added_salt = 12.5 ∧
  initial_weight * initial_concentration + added_salt = 
    (initial_weight + added_salt) * final_concentration →
  initial_weight = 100

-- Proof
theorem salt_solution_weight_proof :
  salt_solution_weight 0.10 0.20 12.5 100 := by
  sorry

end NUMINAMATH_CALUDE_salt_solution_weight_salt_solution_weight_proof_l1325_132546


namespace NUMINAMATH_CALUDE_c_k_value_l1325_132565

/-- Arithmetic sequence with first term 1 and common difference d -/
def arithmetic_seq (d : ℝ) (n : ℕ) : ℝ :=
  1 + (n - 1 : ℝ) * d

/-- Geometric sequence with first term 1 and common ratio r -/
def geometric_seq (r : ℝ) (n : ℕ) : ℝ :=
  r ^ (n - 1)

/-- Sum of nth terms of arithmetic and geometric sequences -/
def c_seq (d r : ℝ) (n : ℕ) : ℝ :=
  arithmetic_seq d n + geometric_seq r n

theorem c_k_value (d r : ℝ) (k : ℕ) :
  (∃ k : ℕ, c_seq d r (k - 1) = 150 ∧ c_seq d r (k + 1) = 900) →
  c_seq d r k = 314 := by
  sorry

end NUMINAMATH_CALUDE_c_k_value_l1325_132565


namespace NUMINAMATH_CALUDE_tom_bob_sticker_ratio_l1325_132589

def bob_stickers : ℕ := 12

theorem tom_bob_sticker_ratio :
  ∃ (tom_stickers : ℕ),
    tom_stickers = bob_stickers ∧
    tom_stickers / bob_stickers = 1 := by
  sorry

end NUMINAMATH_CALUDE_tom_bob_sticker_ratio_l1325_132589


namespace NUMINAMATH_CALUDE_incorrect_calculation_l1325_132599

theorem incorrect_calculation : 
  ((-11) + (-17) = -28) ∧ 
  ((-3/4 : ℚ) + (1/2 : ℚ) = -1/4) ∧ 
  ((-9) + 9 = 0) ∧ 
  ((5/8 : ℚ) + (-7/12 : ℚ) ≠ -1/24) := by
  sorry

end NUMINAMATH_CALUDE_incorrect_calculation_l1325_132599


namespace NUMINAMATH_CALUDE_cube_tetrahedron_surface_area_ratio_l1325_132547

theorem cube_tetrahedron_surface_area_ratio :
  let cube_side_length : ℝ := 2
  let tetrahedron_vertices : List (ℝ × ℝ × ℝ) := [(0, 0, 0), (2, 2, 0), (2, 0, 2), (0, 2, 2)]
  let cube_surface_area : ℝ := 6 * cube_side_length ^ 2
  let tetrahedron_side_length : ℝ := Real.sqrt ((2 - 0)^2 + (2 - 0)^2 + (0 - 0)^2)
  let tetrahedron_surface_area : ℝ := Real.sqrt 3 * tetrahedron_side_length ^ 2
  cube_surface_area / tetrahedron_surface_area = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_cube_tetrahedron_surface_area_ratio_l1325_132547


namespace NUMINAMATH_CALUDE_sum_2x_2y_l1325_132503

theorem sum_2x_2y (x y : ℝ) (h1 : x^2 - y^2 = 8) (h2 : x - y = 6) : 2*x + 2*y = 8/3 := by
  sorry

end NUMINAMATH_CALUDE_sum_2x_2y_l1325_132503


namespace NUMINAMATH_CALUDE_square_area_not_covered_by_circles_l1325_132511

/-- The area of a square not covered by circles -/
theorem square_area_not_covered_by_circles (side_length : ℝ) (num_circles : ℕ) : 
  side_length = 16 → num_circles = 9 → 
  side_length^2 - (num_circles : ℝ) * (side_length / 3)^2 * Real.pi = 256 - 64 * Real.pi := by
  sorry

#check square_area_not_covered_by_circles

end NUMINAMATH_CALUDE_square_area_not_covered_by_circles_l1325_132511


namespace NUMINAMATH_CALUDE_lines_perpendicular_to_plane_are_parallel_l1325_132570

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations between lines and planes
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)

-- State the theorem
theorem lines_perpendicular_to_plane_are_parallel
  (l₁ l₂ : Line) (α : Plane)
  (h₁ : l₁ ≠ l₂)  -- l₁ and l₂ are non-coincident
  (h₂ : perpendicular l₁ α)
  (h₃ : perpendicular l₂ α) :
  parallel l₁ l₂ :=
sorry

end NUMINAMATH_CALUDE_lines_perpendicular_to_plane_are_parallel_l1325_132570


namespace NUMINAMATH_CALUDE_min_value_trigonometric_expression_min_value_achievable_l1325_132543

theorem min_value_trigonometric_expression (α β : ℝ) :
  (3 * Real.cos α + 6 * Real.sin β - 10)^2 + 
  (3 * Real.sin α + 6 * Real.cos β + 4 * Real.cos (α + β) - 20)^2 ≥ 500 :=
by sorry

theorem min_value_achievable :
  ∃ α β : ℝ, (3 * Real.cos α + 6 * Real.sin β - 10)^2 + 
             (3 * Real.sin α + 6 * Real.cos β + 4 * Real.cos (α + β) - 20)^2 = 500 :=
by sorry

end NUMINAMATH_CALUDE_min_value_trigonometric_expression_min_value_achievable_l1325_132543


namespace NUMINAMATH_CALUDE_fraction_reducibility_l1325_132522

theorem fraction_reducibility (n : ℤ) :
  (∃ (k : ℤ), n = 13 * k + 8) ↔
  (∃ (m : ℤ), 3 * n + 2 = 13 * m) ∧ (∃ (l : ℤ), 8 * n + 1 = 13 * l) :=
by sorry

end NUMINAMATH_CALUDE_fraction_reducibility_l1325_132522


namespace NUMINAMATH_CALUDE_probability_of_flush_l1325_132524

/-- Represents a standard deck of cards -/
def StandardDeck : ℕ := 52

/-- Number of suits in a standard deck -/
def NumSuits : ℕ := 4

/-- Number of cards in each suit -/
def CardsPerSuit : ℕ := 13

/-- Size of a poker hand -/
def HandSize : ℕ := 5

/-- Probability of drawing a flush in a 5-card poker hand -/
theorem probability_of_flush (deck : ℕ) (suits : ℕ) (cards_per_suit : ℕ) (hand_size : ℕ) :
  deck = StandardDeck →
  suits = NumSuits →
  cards_per_suit = CardsPerSuit →
  hand_size = HandSize →
  (suits * (Nat.choose cards_per_suit hand_size) : ℚ) / (Nat.choose deck hand_size) = 33 / 16660 :=
sorry

end NUMINAMATH_CALUDE_probability_of_flush_l1325_132524


namespace NUMINAMATH_CALUDE_triangle_4_6_9_l1325_132549

/-- Defines whether three given lengths can form a triangle -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- Theorem stating that lengths 4, 6, and 9 can form a triangle -/
theorem triangle_4_6_9 :
  can_form_triangle 4 6 9 := by
  sorry

end NUMINAMATH_CALUDE_triangle_4_6_9_l1325_132549


namespace NUMINAMATH_CALUDE_cookies_for_guests_l1325_132528

/-- Given a total number of cookies and cookies per guest, calculate the number of guests --/
def calculate_guests (total_cookies : ℕ) (cookies_per_guest : ℕ) : ℕ :=
  total_cookies / cookies_per_guest

/-- Theorem: Given 10 total cookies and 2 cookies per guest, the number of guests is 5 --/
theorem cookies_for_guests : calculate_guests 10 2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_cookies_for_guests_l1325_132528


namespace NUMINAMATH_CALUDE_minuend_calculation_l1325_132576

theorem minuend_calculation (subtrahend difference : ℝ) 
  (h1 : subtrahend = 1.34)
  (h2 : difference = 3.66) : 
  subtrahend + difference = 5 := by
  sorry

end NUMINAMATH_CALUDE_minuend_calculation_l1325_132576


namespace NUMINAMATH_CALUDE_max_product_constrained_sum_l1325_132502

theorem max_product_constrained_sum (a b : ℝ) : 
  a > 0 → b > 0 → 5 * a + 8 * b = 80 → ab ≤ 40 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ 5 * a₀ + 8 * b₀ = 80 ∧ a₀ * b₀ = 40 := by
  sorry

end NUMINAMATH_CALUDE_max_product_constrained_sum_l1325_132502


namespace NUMINAMATH_CALUDE_ratio_x_to_y_l1325_132512

theorem ratio_x_to_y (x y : ℚ) (h : (7 * x - 4 * y) / (20 * x - 3 * y) = 4 / 9) :
  x / y = -24 / 17 := by sorry

end NUMINAMATH_CALUDE_ratio_x_to_y_l1325_132512


namespace NUMINAMATH_CALUDE_solve_for_m_l1325_132545

theorem solve_for_m (x m : ℝ) : 2 * x + m = 1 → x = -1 → m = 3 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_m_l1325_132545


namespace NUMINAMATH_CALUDE_expansion_equality_l1325_132590

theorem expansion_equality (x : ℝ) : (1 + x^2) * (1 - x^4) = 1 + x^2 - x^4 - x^6 := by
  sorry

end NUMINAMATH_CALUDE_expansion_equality_l1325_132590


namespace NUMINAMATH_CALUDE_price_reduction_theorem_l1325_132513

/-- Given three consecutive price reductions, calculates the overall percentage reduction -/
def overall_reduction (r1 r2 r3 : ℝ) : ℝ :=
  (1 - (1 - r1) * (1 - r2) * (1 - r3)) * 100

/-- Theorem stating that the overall reduction after 25%, 20%, and 15% reductions is 49% -/
theorem price_reduction_theorem : 
  overall_reduction 0.25 0.20 0.15 = 49 := by
  sorry

#eval overall_reduction 0.25 0.20 0.15

end NUMINAMATH_CALUDE_price_reduction_theorem_l1325_132513


namespace NUMINAMATH_CALUDE_equation_solution_l1325_132526

theorem equation_solution :
  ∃! x : ℚ, x ≠ -3 ∧ (x^2 + 3*x + 4) / (x + 3) = x + 6 :=
by
  use -7/3
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1325_132526


namespace NUMINAMATH_CALUDE_toms_age_ratio_l1325_132567

theorem toms_age_ratio (T N : ℝ) : 
  (T > 0) → 
  (N > 0) → 
  (T - N = 3 * (T - 4 * N)) → 
  T / N = 11 / 2 := by
sorry

end NUMINAMATH_CALUDE_toms_age_ratio_l1325_132567


namespace NUMINAMATH_CALUDE_equal_cost_messages_l1325_132527

/-- Represents the cost of a text messaging plan -/
structure TextPlan where
  costPerMessage : ℚ
  monthlyFee : ℚ

/-- Calculates the total cost for a given number of messages -/
def totalCost (plan : TextPlan) (messages : ℚ) : ℚ :=
  plan.costPerMessage * messages + plan.monthlyFee

theorem equal_cost_messages : 
  let planA : TextPlan := ⟨0.25, 9⟩
  let planB : TextPlan := ⟨0.40, 0⟩
  ∃ (x : ℚ), x = 60 ∧ totalCost planA x = totalCost planB x :=
by sorry

end NUMINAMATH_CALUDE_equal_cost_messages_l1325_132527


namespace NUMINAMATH_CALUDE_diamond_calculation_l1325_132521

-- Define the diamond operation
def diamond (a b : ℤ) : ℤ := a^2 - 2*a*b + b^2

-- State the theorem
theorem diamond_calculation : diamond (diamond 5 (diamond 7 4)) 3 = 169 := by
  sorry

end NUMINAMATH_CALUDE_diamond_calculation_l1325_132521


namespace NUMINAMATH_CALUDE_price_increase_percentage_l1325_132581

theorem price_increase_percentage (old_price new_price : ℝ) (h1 : old_price = 300) (h2 : new_price = 420) :
  ((new_price - old_price) / old_price) * 100 = 40 := by
  sorry

end NUMINAMATH_CALUDE_price_increase_percentage_l1325_132581


namespace NUMINAMATH_CALUDE_derivative_f_at_zero_l1325_132592

noncomputable def f (x : ℝ) : ℝ :=
  if x ≠ 0 then Real.tan (x^3 + x^2 * Real.sin (2/x))
  else 0

theorem derivative_f_at_zero :
  deriv f 0 = 0 := by
  sorry

end NUMINAMATH_CALUDE_derivative_f_at_zero_l1325_132592


namespace NUMINAMATH_CALUDE_unique_p_l1325_132530

/-- The cubic equation with parameter p -/
def cubic_equation (p : ℝ) (x : ℝ) : Prop :=
  5 * x^3 - 5*(p+1)*x^2 + (71*p-1)*x + 1 = 66*p

/-- A number is a natural root of the cubic equation -/
def is_natural_root (p : ℝ) (x : ℕ) : Prop :=
  cubic_equation p (x : ℝ)

/-- The cubic equation has exactly three natural roots -/
def has_three_natural_roots (p : ℝ) : Prop :=
  ∃ (a b c : ℕ), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    is_natural_root p a ∧ is_natural_root p b ∧ is_natural_root p c ∧
    ∀ (x : ℕ), is_natural_root p x → (x = a ∨ x = b ∨ x = c)

/-- The main theorem: 76 is the only real number satisfying the conditions -/
theorem unique_p : ∀ (p : ℝ), has_three_natural_roots p ↔ p = 76 := by
  sorry

end NUMINAMATH_CALUDE_unique_p_l1325_132530


namespace NUMINAMATH_CALUDE_company_production_l1325_132540

/-- Calculates the total number of parts produced by a company given specific production conditions. -/
def totalPartsProduced (initialPartsPerDay : ℕ) (initialDays : ℕ) (increasedPartsPerDay : ℕ) (extraParts : ℕ) : ℕ :=
  let totalInitialParts := initialPartsPerDay * initialDays
  let increasedProduction := initialPartsPerDay + increasedPartsPerDay
  let additionalDays := extraParts / increasedPartsPerDay
  let totalIncreasedParts := increasedProduction * additionalDays
  totalInitialParts + totalIncreasedParts

/-- Theorem stating that under given conditions, the company produces 1107 parts. -/
theorem company_production : 
  totalPartsProduced 40 3 7 150 = 1107 := by
  sorry

#eval totalPartsProduced 40 3 7 150

end NUMINAMATH_CALUDE_company_production_l1325_132540


namespace NUMINAMATH_CALUDE_four_digit_number_problem_l1325_132508

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def reverse_number (n : ℕ) : ℕ := 
  let d1 := n / 1000
  let d2 := (n / 100) % 10
  let d3 := (n / 10) % 10
  let d4 := n % 10
  d4 * 1000 + d3 * 100 + d2 * 10 + d1

theorem four_digit_number_problem (N : ℕ) (hN : is_four_digit N) :
  let M := reverse_number N
  (N + M = 3333 ∧ N - M = 693) → N = 2013 := by
  sorry

end NUMINAMATH_CALUDE_four_digit_number_problem_l1325_132508


namespace NUMINAMATH_CALUDE_six_double_prime_value_l1325_132551

-- Define the prime operation
def prime (q : ℝ) : ℝ := 3 * q - 3

-- State the theorem
theorem six_double_prime_value : prime (prime 6) = 42 := by
  sorry

end NUMINAMATH_CALUDE_six_double_prime_value_l1325_132551


namespace NUMINAMATH_CALUDE_next_year_day_l1325_132571

/-- Represents the days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a year -/
structure Year where
  numDays : Nat
  firstDay : DayOfWeek
  numSaturdays : Nat

/-- Function to get the next day of the week -/
def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

theorem next_year_day (y : Year) (h1 : y.numDays = 366) (h2 : y.numSaturdays = 53) :
  nextDay y.firstDay = DayOfWeek.Monday := by
  sorry

end NUMINAMATH_CALUDE_next_year_day_l1325_132571


namespace NUMINAMATH_CALUDE_runners_meeting_time_l1325_132509

/-- 
Given two runners, Danny and Steve, running towards each other from their respective houses:
* Danny's time to reach Steve's house is t minutes
* Steve's time to reach Danny's house is 2t minutes
* Steve takes 13.5 minutes longer to reach the halfway point than Danny
Prove that t = 27 minutes
-/
theorem runners_meeting_time (t : ℝ) 
  (h1 : t > 0) -- Danny's time is positive
  (h2 : 2 * t - t / 2 = 13.5) -- Difference in time to reach halfway point
  : t = 27 := by
  sorry

end NUMINAMATH_CALUDE_runners_meeting_time_l1325_132509


namespace NUMINAMATH_CALUDE_doughnut_cost_theorem_l1325_132538

/-- Calculate the total cost of doughnuts for a class --/
theorem doughnut_cost_theorem (total_students : ℕ) 
  (chocolate_students : ℕ) (glazed_students : ℕ) (maple_students : ℕ) (strawberry_students : ℕ)
  (chocolate_cost : ℚ) (glazed_cost : ℚ) (maple_cost : ℚ) (strawberry_cost : ℚ) :
  total_students = 25 →
  chocolate_students = 10 →
  glazed_students = 8 →
  maple_students = 5 →
  strawberry_students = 2 →
  chocolate_cost = 2 →
  glazed_cost = 1 →
  maple_cost = (3/2) →
  strawberry_cost = (5/2) →
  (chocolate_students : ℚ) * chocolate_cost + 
  (glazed_students : ℚ) * glazed_cost + 
  (maple_students : ℚ) * maple_cost + 
  (strawberry_students : ℚ) * strawberry_cost = (81/2) := by
  sorry

#eval (81/2 : ℚ)

end NUMINAMATH_CALUDE_doughnut_cost_theorem_l1325_132538


namespace NUMINAMATH_CALUDE_edward_lives_problem_l1325_132573

theorem edward_lives_problem (lives_lost lives_remaining : ℕ) 
  (h1 : lives_lost = 8)
  (h2 : lives_remaining = 7) :
  lives_lost + lives_remaining = 15 :=
by sorry

end NUMINAMATH_CALUDE_edward_lives_problem_l1325_132573


namespace NUMINAMATH_CALUDE_staff_avg_age_l1325_132505

def robotics_camp (total_members : ℕ) (overall_avg_age : ℝ)
  (num_girls num_boys num_adults num_staff : ℕ)
  (avg_age_girls avg_age_boys avg_age_adults : ℝ) : Prop :=
  total_members = 50 ∧
  overall_avg_age = 20 ∧
  num_girls = 22 ∧
  num_boys = 18 ∧
  num_adults = 5 ∧
  num_staff = 5 ∧
  avg_age_girls = 18 ∧
  avg_age_boys = 19 ∧
  avg_age_adults = 30

theorem staff_avg_age
  (h : robotics_camp 50 20 22 18 5 5 18 19 30) :
  (50 * 20 - (22 * 18 + 18 * 19 + 5 * 30)) / 5 = 22.4 :=
by sorry

end NUMINAMATH_CALUDE_staff_avg_age_l1325_132505


namespace NUMINAMATH_CALUDE_solution_comparison_l1325_132529

theorem solution_comparison (p p' q q' : ℝ) (hp : p ≠ 0) (hp' : p' ≠ 0) :
  (-q / p > -q' / p') ↔ (q / p < q' / p') :=
by sorry

end NUMINAMATH_CALUDE_solution_comparison_l1325_132529


namespace NUMINAMATH_CALUDE_area_curve_C_m_1_intersection_points_l1325_132536

-- Define the curve C
def curve_C (m : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | |p.1| + |p.2| = m}

-- Define the ellipse
def ellipse : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 / 9 + p.2^2 / 4 = 1}

-- Theorem 1: Area enclosed by curve C when m = 1
theorem area_curve_C_m_1 :
  MeasureTheory.volume (curve_C 1) = 2 := by sorry

-- Theorem 2: Intersection points of curve C and ellipse
theorem intersection_points (m : ℝ) :
  (∃ (a b c d : ℝ × ℝ), a ∈ curve_C m ∩ ellipse ∧
                         b ∈ curve_C m ∩ ellipse ∧
                         c ∈ curve_C m ∩ ellipse ∧
                         d ∈ curve_C m ∩ ellipse ∧
                         a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) ↔
  (2 < m ∧ m < 3) ∨ m = Real.sqrt 13 := by sorry

end NUMINAMATH_CALUDE_area_curve_C_m_1_intersection_points_l1325_132536


namespace NUMINAMATH_CALUDE_range_of_f_l1325_132548

def f (x : ℝ) : ℝ := x^2 - 2*x + 9

theorem range_of_f :
  ∀ y ∈ Set.range f,
  (∃ x ∈ Set.Icc (-1 : ℝ) 2, f x = y) →
  y ∈ Set.Icc 8 12 :=
by sorry

end NUMINAMATH_CALUDE_range_of_f_l1325_132548


namespace NUMINAMATH_CALUDE_smallest_base_perfect_square_l1325_132588

theorem smallest_base_perfect_square : ∃ (b : ℕ), 
  b > 3 ∧ 
  (∃ (n : ℕ), n^2 = 2*b + 3 ∧ n^2 < 25) ∧
  (∀ (k : ℕ), k > 3 ∧ k < b → ¬∃ (m : ℕ), m^2 = 2*k + 3 ∧ m^2 < 25) ∧
  b = 11 := by
sorry

end NUMINAMATH_CALUDE_smallest_base_perfect_square_l1325_132588


namespace NUMINAMATH_CALUDE_matrix_multiplication_example_l1325_132550

theorem matrix_multiplication_example :
  let A : Matrix (Fin 2) (Fin 2) ℤ := !![2, 3; 7, -1]
  let B : Matrix (Fin 2) (Fin 2) ℤ := !![1, -5; 0, 4]
  A * B = !![2, 2; 7, -39] := by
sorry

end NUMINAMATH_CALUDE_matrix_multiplication_example_l1325_132550


namespace NUMINAMATH_CALUDE_tangent_line_to_circle_l1325_132552

/-- A line is tangent to a circle if and only if the distance from the center of the circle to the line equals the radius of the circle. -/
axiom line_tangent_to_circle_iff_distance_eq_radius {a b c : ℝ} {x₀ y₀ r : ℝ} :
  (∀ x y, (x - x₀)^2 + (y - y₀)^2 = r^2 → a*x + b*y + c ≠ 0) ↔
  |a*x₀ + b*y₀ + c| / Real.sqrt (a^2 + b^2) = r

/-- The theorem to be proved -/
theorem tangent_line_to_circle (m : ℝ) :
  m > 0 →
  (∀ x y, (x - 3)^2 + (y - 4)^2 = 4 → 3*x - 4*y - m ≠ 0) →
  m = 3 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_to_circle_l1325_132552


namespace NUMINAMATH_CALUDE_exchange_rate_theorem_l1325_132587

/-- Represents the number of boys in the group -/
def b : ℕ := sorry

/-- Represents the number of girls in the group -/
def g : ℕ := sorry

/-- Represents the exchange rate of yuan to alternative currency -/
def x : ℕ := sorry

/-- The total cost in yuan at the first disco -/
def first_disco_cost : ℕ := b * g

/-- The total cost in alternative currency at the second place -/
def second_place_cost : ℕ := (b + g) * (b + g - 1) + (b + g) + 1

/-- Theorem stating the exchange rate between yuan and alternative currency -/
theorem exchange_rate_theorem : 
  first_disco_cost * x = second_place_cost ∧ x = 5 :=
by sorry

end NUMINAMATH_CALUDE_exchange_rate_theorem_l1325_132587


namespace NUMINAMATH_CALUDE_total_games_played_l1325_132596

-- Define the parameters
def win_percentage : ℚ := 50 / 100
def games_won : ℕ := 70

-- State the theorem
theorem total_games_played : ℕ := by
  -- The proof goes here
  sorry

-- The goal to prove
#check total_games_played = 140

end NUMINAMATH_CALUDE_total_games_played_l1325_132596


namespace NUMINAMATH_CALUDE_gambler_initial_games_gambler_initial_games_proof_l1325_132572

theorem gambler_initial_games : ℝ → Prop :=
  fun x =>
    let initial_win_rate : ℝ := 0.4
    let new_win_rate : ℝ := 0.8
    let additional_games : ℝ := 30
    let final_win_rate : ℝ := 0.6
    (initial_win_rate * x + new_win_rate * additional_games) / (x + additional_games) = final_win_rate →
    x = 30

theorem gambler_initial_games_proof : ∃ x : ℝ, gambler_initial_games x := by
  sorry

end NUMINAMATH_CALUDE_gambler_initial_games_gambler_initial_games_proof_l1325_132572


namespace NUMINAMATH_CALUDE_relay_race_arrangements_l1325_132516

/-- The number of students to choose from -/
def total_students : ℕ := 10

/-- The number of legs in the relay race -/
def race_legs : ℕ := 4

/-- Function to calculate permutations -/
def permutations (n k : ℕ) : ℕ :=
  if k > n then 0
  else Nat.factorial n / Nat.factorial (n - k)

/-- The main theorem -/
theorem relay_race_arrangements :
  permutations total_students race_legs
  - permutations (total_students - 1) (race_legs - 1)  -- A not in first leg
  - permutations (total_students - 1) (race_legs - 1)  -- B not in last leg
  + permutations (total_students - 2) (race_legs - 2)  -- Neither A in first nor B in last
  = 4008 := by
  sorry

end NUMINAMATH_CALUDE_relay_race_arrangements_l1325_132516


namespace NUMINAMATH_CALUDE_product_of_largest_primes_l1325_132575

/-- The largest one-digit prime -/
def largest_one_digit_prime : ℕ := 7

/-- The second largest one-digit prime -/
def second_largest_one_digit_prime : ℕ := 5

/-- The largest three-digit prime -/
def largest_three_digit_prime : ℕ := 997

/-- Theorem stating that the product of the two largest one-digit primes
    and the largest three-digit prime is 34895 -/
theorem product_of_largest_primes :
  largest_one_digit_prime * second_largest_one_digit_prime * largest_three_digit_prime = 34895 := by
  sorry

end NUMINAMATH_CALUDE_product_of_largest_primes_l1325_132575


namespace NUMINAMATH_CALUDE_proof_by_contradiction_assumption_l1325_132532

theorem proof_by_contradiction_assumption (a b : ℤ) : 
  (5 ∣ a * b) → (5 ∣ a ∨ 5 ∣ b) ↔ 
  (¬(5 ∣ a) ∧ ¬(5 ∣ b)) → False :=
by sorry

end NUMINAMATH_CALUDE_proof_by_contradiction_assumption_l1325_132532


namespace NUMINAMATH_CALUDE_tom_car_lease_annual_cost_l1325_132501

/-- Calculates the annual cost of Tom's car lease -/
theorem tom_car_lease_annual_cost :
  let miles_mon_wed_fri : ℕ := 50
  let miles_other_days : ℕ := 100
  let days_mon_wed_fri : ℕ := 3
  let days_other : ℕ := 4
  let cost_per_mile : ℚ := 1 / 10
  let weekly_fee : ℕ := 100
  let weeks_per_year : ℕ := 52

  let weekly_miles : ℕ := miles_mon_wed_fri * days_mon_wed_fri + miles_other_days * days_other
  let weekly_mileage_cost : ℚ := (weekly_miles : ℚ) * cost_per_mile
  let total_weekly_cost : ℚ := weekly_mileage_cost + weekly_fee
  let annual_cost : ℚ := total_weekly_cost * weeks_per_year

  annual_cost = 8060 := by
sorry


end NUMINAMATH_CALUDE_tom_car_lease_annual_cost_l1325_132501


namespace NUMINAMATH_CALUDE_money_distribution_l1325_132523

theorem money_distribution (a b c total : ℕ) : 
  (a + b + c = total) →
  (2 * b = 3 * a) →
  (4 * b = 3 * c) →
  (b = 1500) →
  (total = 4500) := by
sorry

end NUMINAMATH_CALUDE_money_distribution_l1325_132523


namespace NUMINAMATH_CALUDE_platform_length_l1325_132514

/-- The length of a platform given train parameters --/
theorem platform_length (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) : 
  train_length = 250 →
  train_speed_kmh = 55 →
  crossing_time = 35.99712023038157 →
  ∃ (platform_length : ℝ), platform_length = 300 := by
  sorry

#check platform_length

end NUMINAMATH_CALUDE_platform_length_l1325_132514


namespace NUMINAMATH_CALUDE_x_convergence_l1325_132558

def x : ℕ → ℚ
  | 0 => 7
  | n + 1 => (x n ^ 2 + 8 * x n + 9) / (x n + 7)

theorem x_convergence :
  ∃ m : ℕ, m ≥ 81 ∧ m ≤ 242 ∧ 
    x m ≤ 5 + 1 / 2^15 ∧ 
    ∀ k : ℕ, k > 0 ∧ k < m → x k > 5 + 1 / 2^15 :=
by sorry

end NUMINAMATH_CALUDE_x_convergence_l1325_132558


namespace NUMINAMATH_CALUDE_polynomial_expansion_l1325_132577

theorem polynomial_expansion (s : ℝ) :
  (3 * s^3 - 4 * s^2 + 5 * s - 2) * (2 * s^2 - 3 * s + 4) =
  6 * s^5 - 17 * s^4 + 34 * s^3 - 35 * s^2 + 26 * s - 8 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_expansion_l1325_132577


namespace NUMINAMATH_CALUDE_arithmetic_sequence_formula_l1325_132535

/-- An arithmetic sequence with given conditions -/
def ArithmeticSequence (a : ℕ → ℤ) : Prop :=
  a 2 = -1 ∧ a 4 = 3 ∧ ∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m

/-- The general formula for the arithmetic sequence -/
def GeneralFormula (n : ℕ) : ℤ := 2 * n - 5

theorem arithmetic_sequence_formula (a : ℕ → ℤ) :
  ArithmeticSequence a → ∀ n : ℕ, a n = GeneralFormula n := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_formula_l1325_132535


namespace NUMINAMATH_CALUDE_sum_of_W_and_Y_l1325_132591

def problem (W X Y Z : ℕ) : Prop :=
  W ∈ ({2, 3, 5, 6} : Set ℕ) ∧
  X ∈ ({2, 3, 5, 6} : Set ℕ) ∧
  Y ∈ ({2, 3, 5, 6} : Set ℕ) ∧
  Z ∈ ({2, 3, 5, 6} : Set ℕ) ∧
  W ≠ X ∧ W ≠ Y ∧ W ≠ Z ∧ X ≠ Y ∧ X ≠ Z ∧ Y ≠ Z ∧
  (W * X : ℚ) / (Y * Z) + (Y : ℚ) / Z = 3

theorem sum_of_W_and_Y (W X Y Z : ℕ) :
  problem W X Y Z → W + Y = 8 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_W_and_Y_l1325_132591


namespace NUMINAMATH_CALUDE_simplify_monomial_l1325_132559

theorem simplify_monomial (a : ℝ) : (-2 * a^2)^3 = -8 * a^6 := by
  sorry

end NUMINAMATH_CALUDE_simplify_monomial_l1325_132559


namespace NUMINAMATH_CALUDE_sheet_reduction_percentage_l1325_132582

def original_sheets : ℕ := 20
def original_lines_per_sheet : ℕ := 55
def original_chars_per_line : ℕ := 65

def retyped_lines_per_sheet : ℕ := 65
def retyped_chars_per_line : ℕ := 70

def total_chars : ℕ := original_sheets * original_lines_per_sheet * original_chars_per_line

def chars_per_retyped_sheet : ℕ := retyped_lines_per_sheet * retyped_chars_per_line

def retyped_sheets : ℕ := (total_chars + chars_per_retyped_sheet - 1) / chars_per_retyped_sheet

theorem sheet_reduction_percentage : 
  (original_sheets - retyped_sheets) * 100 / original_sheets = 20 := by
  sorry

end NUMINAMATH_CALUDE_sheet_reduction_percentage_l1325_132582


namespace NUMINAMATH_CALUDE_cubic_identity_l1325_132593

theorem cubic_identity (x : ℝ) : (2*x - 1)^3 = 5*x^3 + (3*x + 1)*(x^2 - x - 1) - 10*x^2 + 10*x := by
  sorry

end NUMINAMATH_CALUDE_cubic_identity_l1325_132593


namespace NUMINAMATH_CALUDE_quadratic_range_l1325_132515

theorem quadratic_range (x : ℝ) (h : x^2 - 4*x + 3 < 0) :
  8 < x^2 + 4*x + 3 ∧ x^2 + 4*x + 3 < 24 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_range_l1325_132515


namespace NUMINAMATH_CALUDE_dads_strawberry_weight_l1325_132563

/-- Given the total weight of strawberries picked by Marco and his dad,
    and the weight of Marco's strawberries, calculate the weight of his dad's strawberries. -/
theorem dads_strawberry_weight 
  (total_weight : ℕ) 
  (marcos_weight : ℕ) 
  (h1 : total_weight = 20)
  (h2 : marcos_weight = 3) : 
  total_weight - marcos_weight = 17 := by
  sorry

#check dads_strawberry_weight

end NUMINAMATH_CALUDE_dads_strawberry_weight_l1325_132563


namespace NUMINAMATH_CALUDE_e_is_random_error_l1325_132500

/-- Linear regression model -/
structure LinearRegressionModel where
  x : ℝ
  y : ℝ
  a : ℝ
  b : ℝ
  e : ℝ
  model_equation : y = b * x + a + e

/-- Definition of random error in linear regression -/
def is_random_error (model : LinearRegressionModel) : Prop :=
  ∃ (error_term : ℝ), 
    error_term = model.e ∧ 
    model.y = model.b * model.x + model.a + error_term

/-- Theorem: In the linear regression model, e is the random error -/
theorem e_is_random_error (model : LinearRegressionModel) : 
  is_random_error model :=
sorry

end NUMINAMATH_CALUDE_e_is_random_error_l1325_132500


namespace NUMINAMATH_CALUDE_similar_triangles_leg_length_l1325_132584

/-- Two similar right triangles, one with legs 12 and 9, the other with legs y and 7 -/
def similar_triangles (y : ℝ) : Prop :=
  12 / y = 9 / 7

theorem similar_triangles_leg_length :
  ∃ y : ℝ, similar_triangles y ∧ y = 84 / 9 := by
  sorry

end NUMINAMATH_CALUDE_similar_triangles_leg_length_l1325_132584


namespace NUMINAMATH_CALUDE_min_beta_delta_sum_l1325_132542

theorem min_beta_delta_sum (g : ℂ → ℂ) (β δ : ℂ) :
  (∀ z, g z = (3 + 2 * Complex.I) * z^2 + β * z + δ) →
  (g 1).im = 0 →
  (g (-Complex.I)).im = 0 →
  ∃ (β₀ δ₀ : ℂ), Complex.abs β₀ + Complex.abs δ₀ = 2 ∧
    ∀ β' δ', (∀ z, g z = (3 + 2 * Complex.I) * z^2 + β' * z + δ') →
              (g 1).im = 0 →
              (g (-Complex.I)).im = 0 →
              Complex.abs β' + Complex.abs δ' ≥ 2 :=
sorry

end NUMINAMATH_CALUDE_min_beta_delta_sum_l1325_132542


namespace NUMINAMATH_CALUDE_henry_payment_l1325_132554

/-- The payment Henry receives for painting a bike -/
def paint_payment : ℕ := 5

/-- The additional payment Henry receives for selling a bike compared to painting it -/
def sell_additional_payment : ℕ := 8

/-- The number of bikes Henry sells and paints -/
def num_bikes : ℕ := 8

/-- The total payment Henry receives for selling and painting the given number of bikes -/
def total_payment (paint : ℕ) (sell_additional : ℕ) (bikes : ℕ) : ℕ :=
  bikes * (paint + sell_additional + paint)

theorem henry_payment :
  total_payment paint_payment sell_additional_payment num_bikes = 144 := by
sorry

end NUMINAMATH_CALUDE_henry_payment_l1325_132554


namespace NUMINAMATH_CALUDE_illuminated_cube_surface_area_l1325_132580

/-- The illuminated area of a cube's surface when a cylindrical beam of light is directed along its main diagonal --/
theorem illuminated_cube_surface_area
  (a : ℝ) -- Edge length of the cube
  (ρ : ℝ) -- Radius of the cylindrical beam
  (h1 : a = Real.sqrt (2 + Real.sqrt 2)) -- Given edge length
  (h2 : ρ = Real.sqrt 2) -- Given beam radius
  (h3 : ρ > 0) -- Positive radius
  (h4 : a > 0) -- Positive edge length
  : Real.sqrt 3 * π / 2 + 3 * Real.sqrt 6 = 
    (3 : ℝ) * π * ρ^2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_illuminated_cube_surface_area_l1325_132580


namespace NUMINAMATH_CALUDE_radio_station_survey_l1325_132586

theorem radio_station_survey (total_listeners total_non_listeners female_listeners male_non_listeners : ℕ)
  (h1 : total_listeners = 160)
  (h2 : total_non_listeners = 180)
  (h3 : female_listeners = 72)
  (h4 : male_non_listeners = 88) :
  total_listeners - female_listeners = 92 :=
by
  sorry

#check radio_station_survey

end NUMINAMATH_CALUDE_radio_station_survey_l1325_132586


namespace NUMINAMATH_CALUDE_fraction_c_simplest_form_l1325_132568

def is_simplest_form (n d : ℤ) : Prop :=
  ∀ k : ℤ, k ≠ 0 → k ∣ n ∧ k ∣ d → k = 1 ∨ k = -1

theorem fraction_c_simplest_form (x y : ℤ) (hx : x ≠ 0) :
  is_simplest_form (x + y) (2 * x) :=
sorry

end NUMINAMATH_CALUDE_fraction_c_simplest_form_l1325_132568


namespace NUMINAMATH_CALUDE_nonagon_perimeter_l1325_132557

/-- A regular nonagon is a polygon with 9 sides of equal length and equal angles -/
structure RegularNonagon where
  sideLength : ℝ
  numSides : ℕ
  numSides_eq : numSides = 9

/-- The perimeter of a regular nonagon is the product of its number of sides and side length -/
def perimeter (n : RegularNonagon) : ℝ := n.numSides * n.sideLength

/-- Theorem: The perimeter of a regular nonagon with side length 2 cm is 18 cm -/
theorem nonagon_perimeter :
  ∀ (n : RegularNonagon), n.sideLength = 2 → perimeter n = 18 := by
  sorry

end NUMINAMATH_CALUDE_nonagon_perimeter_l1325_132557


namespace NUMINAMATH_CALUDE_no_single_digit_fraction_l1325_132519

theorem no_single_digit_fraction :
  ¬ ∃ (n : ℕ+) (a b : ℕ),
    1 ≤ a ∧ a < 10 ∧
    1 ≤ b ∧ b < 10 ∧
    (1234 - n) * b = (6789 - n) * a :=
by sorry

end NUMINAMATH_CALUDE_no_single_digit_fraction_l1325_132519


namespace NUMINAMATH_CALUDE_quadratic_sum_l1325_132556

/-- A quadratic function f(x) = px² + qx + r with vertex (3, 4) passing through (1, 2) -/
def QuadraticFunction (p q r : ℝ) : ℝ → ℝ := fun x ↦ p * x^2 + q * x + r

theorem quadratic_sum (p q r : ℝ) :
  (∀ x, QuadraticFunction p q r x = p * x^2 + q * x + r) →
  (∃ a, ∀ x, QuadraticFunction p q r x = a * (x - 3)^2 + 4) →
  QuadraticFunction p q r 1 = 2 →
  p + q + r = 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sum_l1325_132556


namespace NUMINAMATH_CALUDE_cube_volume_from_surface_area_l1325_132566

theorem cube_volume_from_surface_area (surface_area : ℝ) (volume : ℝ) : 
  surface_area = 486 → volume = 729 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_from_surface_area_l1325_132566


namespace NUMINAMATH_CALUDE_quadratic_vertex_l1325_132555

/-- The quadratic function f(x) = 2x^2 - 4x + 5 has its vertex at (1, 3) -/
theorem quadratic_vertex (x : ℝ) :
  let f : ℝ → ℝ := λ x ↦ 2 * x^2 - 4 * x + 5
  (∀ x, f x ≥ f 1) ∧ f 1 = 3 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_vertex_l1325_132555


namespace NUMINAMATH_CALUDE_shortest_distance_between_circles_l1325_132597

/-- The shortest distance between two circles -/
theorem shortest_distance_between_circles : 
  let circle1 := {(x, y) : ℝ × ℝ | x^2 - 8*x + y^2 + 6*y + 9 = 0}
  let circle2 := {(x, y) : ℝ × ℝ | x^2 + 10*x + y^2 - 2*y + 25 = 0}
  ∃ d : ℝ, d = Real.sqrt 97 - 5 ∧ 
    ∀ p1 ∈ circle1, ∀ p2 ∈ circle2, d ≤ dist p1 p2 :=
by sorry

end NUMINAMATH_CALUDE_shortest_distance_between_circles_l1325_132597


namespace NUMINAMATH_CALUDE_sum_of_a_and_c_l1325_132560

theorem sum_of_a_and_c (a b c d : ℝ) 
  (h1 : a * b + b * c + c * d + d * a = 30)
  (h2 : b + d = 5) : 
  a + c = 6 := by
sorry

end NUMINAMATH_CALUDE_sum_of_a_and_c_l1325_132560


namespace NUMINAMATH_CALUDE_triangle_properties_l1325_132553

-- Define the triangle ABC
structure Triangle :=
  (a b c : ℝ)
  (A B C : ℝ)

-- State the theorem
theorem triangle_properties (t : Triangle) 
  (h1 : t.b * Real.sin t.A = t.a * Real.sin (2 * t.B))
  (h2 : t.b = Real.sqrt 10)
  (h3 : t.a + t.c = t.a * t.c) :
  t.B = π / 3 ∧ 
  (1/2) * t.a * t.c * Real.sin t.B = (5 * Real.sqrt 3) / 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l1325_132553


namespace NUMINAMATH_CALUDE_intersection_equality_implies_range_l1325_132574

-- Define the sets A and C
def A : Set ℝ := {x | 1 ≤ x ∧ x ≤ 5}
def C (a : ℝ) : Set ℝ := {x | -a ≤ x ∧ x ≤ a + 3}

-- State the theorem
theorem intersection_equality_implies_range (a : ℝ) :
  C a ∩ A = C a → -3/2 ≤ a ∧ a ≤ -1 :=
by sorry

end NUMINAMATH_CALUDE_intersection_equality_implies_range_l1325_132574


namespace NUMINAMATH_CALUDE_mersenne_prime_conditions_l1325_132537

theorem mersenne_prime_conditions (a b : ℕ) (h1 : a ≥ 1) (h2 : b ≥ 2) 
  (h3 : Nat.Prime (a^b - 1)) : a = 2 ∧ Nat.Prime b := by
  sorry

end NUMINAMATH_CALUDE_mersenne_prime_conditions_l1325_132537


namespace NUMINAMATH_CALUDE_second_company_base_rate_l1325_132562

/-- The base rate of United Telephone in dollars -/
def united_base_rate : ℝ := 6

/-- The per-minute rate of United Telephone in dollars -/
def united_per_minute : ℝ := 0.25

/-- The per-minute rate of the second telephone company in dollars -/
def second_per_minute : ℝ := 0.20

/-- The number of minutes used -/
def minutes_used : ℝ := 120

/-- The base rate of the second telephone company in dollars -/
def second_base_rate : ℝ := 12

theorem second_company_base_rate :
  united_base_rate + united_per_minute * minutes_used =
  second_base_rate + second_per_minute * minutes_used :=
by sorry

end NUMINAMATH_CALUDE_second_company_base_rate_l1325_132562


namespace NUMINAMATH_CALUDE_school_capacity_l1325_132520

/-- Given a school with the following properties:
  * It has 15 classrooms
  * One-third of the classrooms have 30 desks each
  * The rest of the classrooms have 25 desks each
  * Only one student can sit at one desk
  This theorem proves that the school can accommodate 400 students. -/
theorem school_capacity :
  let total_classrooms : ℕ := 15
  let desks_per_large_classroom : ℕ := 30
  let desks_per_small_classroom : ℕ := 25
  let large_classrooms : ℕ := total_classrooms / 3
  let small_classrooms : ℕ := total_classrooms - large_classrooms
  let total_capacity : ℕ := large_classrooms * desks_per_large_classroom +
                            small_classrooms * desks_per_small_classroom
  total_capacity = 400 := by
  sorry

end NUMINAMATH_CALUDE_school_capacity_l1325_132520


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l1325_132510

theorem complex_modulus_problem (z : ℂ) : (1 - Complex.I) * z = 1 + Complex.I → Complex.abs z = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l1325_132510


namespace NUMINAMATH_CALUDE_cone_sphere_intersection_l1325_132569

noncomputable def cone_angle (r : ℝ) (h : ℝ) : ℝ :=
  let α := Real.arcsin ((Real.sqrt 5 - 1) / 2)
  2 * α

theorem cone_sphere_intersection (r : ℝ) (h : ℝ) (hr : r > 0) (hh : h > 0) :
  let α := cone_angle r h / 2
  let sphere_radius := h / 2
  let sphere_cap_area := 4 * Real.pi * sphere_radius^2 * Real.sin α^2
  let cone_cap_area := Real.pi * (2 * sphere_radius * Real.cos α * Real.sin α) * (2 * sphere_radius * Real.cos α)
  sphere_cap_area = cone_cap_area →
  cone_angle r h = 2 * Real.arccos (Real.sqrt 5 - 2) :=
by sorry

end NUMINAMATH_CALUDE_cone_sphere_intersection_l1325_132569


namespace NUMINAMATH_CALUDE_roots_equation_result_l1325_132595

theorem roots_equation_result (x₁ x₂ : ℝ) 
  (h₁ : x₁^2 + x₁ - 4 = 0) 
  (h₂ : x₂^2 + x₂ - 4 = 0) 
  (h₃ : x₁ ≠ x₂) : 
  x₁^3 - 5*x₂^2 + 10 = -19 := by
sorry

end NUMINAMATH_CALUDE_roots_equation_result_l1325_132595


namespace NUMINAMATH_CALUDE_distance_from_B_to_center_l1325_132533

-- Define the circle and points
def circle_radius : ℝ := 10
def vertical_distance : ℝ := 6
def horizontal_distance : ℝ := 4

-- Define the points A, B, and C
def point_B (a b : ℝ) : ℝ × ℝ := (a, b)
def point_A (a b : ℝ) : ℝ × ℝ := (a, b + vertical_distance)
def point_C (a b : ℝ) : ℝ × ℝ := (a + horizontal_distance, b)

-- Define the conditions
def on_circle (x y : ℝ) : Prop := x^2 + y^2 = circle_radius^2
def right_angle (A B C : ℝ × ℝ) : Prop :=
  let (x1, y1) := A
  let (x2, y2) := B
  let (x3, y3) := C
  (x1 - x2) * (x3 - x2) + (y1 - y2) * (y3 - y2) = 0

-- Theorem statement
theorem distance_from_B_to_center (a b : ℝ) :
  on_circle a (b + vertical_distance) →
  on_circle (a + horizontal_distance) b →
  right_angle (point_A a b) (point_B a b) (point_C a b) →
  a^2 + b^2 = 74 :=
sorry

end NUMINAMATH_CALUDE_distance_from_B_to_center_l1325_132533


namespace NUMINAMATH_CALUDE_interest_discount_sum_l1325_132578

/-- Given a sum, rate, and time, if the simple interest is 85 and the true discount is 80, then the sum is 1360 -/
theorem interest_discount_sum (P r t : ℝ) : 
  (P * r * t / 100 = 85) → 
  (P * r * t / (100 + r * t) = 80) → 
  P = 1360 := by
  sorry

end NUMINAMATH_CALUDE_interest_discount_sum_l1325_132578


namespace NUMINAMATH_CALUDE_f_injective_f_property_inverse_f_512_l1325_132504

/-- A function satisfying f(5) = 2 and f(2x) = 2f(x) for all x -/
def f : ℝ → ℝ :=
  sorry

/-- f is injective -/
theorem f_injective : Function.Injective f :=
  sorry

/-- The inverse function of f -/
noncomputable def f_inv : ℝ → ℝ := Function.invFun f

theorem f_property (x : ℝ) : f (2 * x) = 2 * f x :=
  sorry

axiom f_5 : f 5 = 2

/-- The main theorem: f⁻¹(512) = 1280 -/
theorem inverse_f_512 : f_inv 512 = 1280 := by
  sorry

end NUMINAMATH_CALUDE_f_injective_f_property_inverse_f_512_l1325_132504


namespace NUMINAMATH_CALUDE_range_of_a_l1325_132594

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, |x + 3| - |x + 1| ≤ 3 * a - a^2) → 1 ≤ a ∧ a ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l1325_132594


namespace NUMINAMATH_CALUDE_cube_sum_square_not_prime_product_l1325_132539

theorem cube_sum_square_not_prime_product (a b : ℕ+) (h : ∃ (u : ℕ), (a.val ^ 3 + b.val ^ 3 : ℕ) = u ^ 2) :
  ¬∃ (p q : ℕ), Nat.Prime p ∧ Nat.Prime q ∧ p ≠ q ∧ a.val + b.val = p * q :=
by sorry

end NUMINAMATH_CALUDE_cube_sum_square_not_prime_product_l1325_132539
