import Mathlib

namespace NUMINAMATH_CALUDE_negation_of_universal_statement_l3040_304048

theorem negation_of_universal_statement :
  (¬ ∀ x : ℝ, x^2 - 2*x - 3 ≥ 0) ↔ (∃ x : ℝ, x^2 - 2*x - 3 < 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_statement_l3040_304048


namespace NUMINAMATH_CALUDE_fermat_number_large_prime_factor_l3040_304029

theorem fermat_number_large_prime_factor (n : ℕ) (h : n ≥ 3) :
  ∃ p : ℕ, Prime p ∧ p ∣ (2^(2^n) + 1) ∧ p > 2^(n+2) * (n+1) := by
  sorry

end NUMINAMATH_CALUDE_fermat_number_large_prime_factor_l3040_304029


namespace NUMINAMATH_CALUDE_tangent_line_right_triangle_l3040_304046

/-- Given a line ax + by + c = 0 (a, b, c ≠ 0) tangent to the circle x² + y² = 1,
    the triangle with side lengths |a|, |b|, and |c| is a right triangle. -/
theorem tangent_line_right_triangle (a b c : ℝ) (h_nonzero : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) 
    (h_tangent : ∀ x y : ℝ, a * x + b * y + c = 0 → x^2 + y^2 = 1) :
  a^2 + b^2 = c^2 := by
sorry

end NUMINAMATH_CALUDE_tangent_line_right_triangle_l3040_304046


namespace NUMINAMATH_CALUDE_new_people_calculation_l3040_304019

/-- The number of new people who moved into the town -/
def new_people : ℕ := 580

/-- The original population of the town -/
def original_population : ℕ := 780

/-- The number of people who moved out -/
def people_moved_out : ℕ := 400

/-- The population after 4 years -/
def final_population : ℕ := 60

/-- The number of years that passed -/
def years_passed : ℕ := 4

theorem new_people_calculation :
  (((original_population - people_moved_out + new_people : ℚ) / 2^years_passed) : ℚ) = final_population := by
  sorry

end NUMINAMATH_CALUDE_new_people_calculation_l3040_304019


namespace NUMINAMATH_CALUDE_translation_of_B_l3040_304006

-- Define a point in 2D space
def Point := ℝ × ℝ

-- Define the translation function
def translate (p : Point) (v : ℝ × ℝ) : Point :=
  (p.1 + v.1, p.2 + v.2)

-- Define the given points
def A : Point := (-1, 0)
def B : Point := (1, 2)
def A₁ : Point := (2, -1)

-- Define the translation vector
def translation_vector : ℝ × ℝ := (A₁.1 - A.1, A₁.2 - A.2)

-- State the theorem
theorem translation_of_B (h : A₁ = translate A translation_vector) :
  translate B translation_vector = (4, 1) := by
  sorry


end NUMINAMATH_CALUDE_translation_of_B_l3040_304006


namespace NUMINAMATH_CALUDE_compute_a_l3040_304080

-- Define the polynomial
def f (a b : ℚ) (x : ℝ) : ℝ := x^3 + a*x^2 + b*x + 48

-- State the theorem
theorem compute_a : 
  ∃ (a b : ℚ), f a b (-1 - 5 * Real.sqrt 3) = 0 ∧ a = 50/37 := by
  sorry

end NUMINAMATH_CALUDE_compute_a_l3040_304080


namespace NUMINAMATH_CALUDE_sum_of_powers_mod_17_l3040_304055

theorem sum_of_powers_mod_17 :
  (∃ x : ℤ, x * 3 ≡ 1 [ZMOD 17]) →
  (∃ y : ℤ, y * 3^2 ≡ 1 [ZMOD 17]) →
  (∃ z : ℤ, z * 3^3 ≡ 1 [ZMOD 17]) →
  (∃ w : ℤ, w * 3^4 ≡ 1 [ZMOD 17]) →
  (∃ v : ℤ, v * 3^5 ≡ 1 [ZMOD 17]) →
  (∃ u : ℤ, u * 3^6 ≡ 1 [ZMOD 17]) →
  x + y + z + w + v + u ≡ 5 [ZMOD 17] :=
by sorry

end NUMINAMATH_CALUDE_sum_of_powers_mod_17_l3040_304055


namespace NUMINAMATH_CALUDE_total_distance_to_school_l3040_304050

-- Define the distances
def bus_distance_km : ℝ := 2
def walking_distance_m : ℝ := 560

-- Define the conversion factor
def km_to_m : ℝ := 1000

-- Theorem to prove
theorem total_distance_to_school :
  bus_distance_km * km_to_m + walking_distance_m = 2560 := by
  sorry

end NUMINAMATH_CALUDE_total_distance_to_school_l3040_304050


namespace NUMINAMATH_CALUDE_sin_alpha_value_l3040_304037

theorem sin_alpha_value (α β : Real) (h_acute : 0 < α ∧ α < π / 2)
  (eq1 : 2 * Real.tan (π - α) - 3 * Real.cos (π / 2 + β) + 5 = 0)
  (eq2 : Real.tan (π + α) + 6 * Real.sin (π + β) = 1) :
  Real.sin α = 3 * Real.sqrt 10 / 10 := by
  sorry

end NUMINAMATH_CALUDE_sin_alpha_value_l3040_304037


namespace NUMINAMATH_CALUDE_scaled_equation_l3040_304003

theorem scaled_equation (h : 2994 * 14.5 = 175) : 29.94 * 1.45 = 1.75 := by
  sorry

end NUMINAMATH_CALUDE_scaled_equation_l3040_304003


namespace NUMINAMATH_CALUDE_jeans_discount_rates_l3040_304047

def regular_price_moose : ℝ := 20
def regular_price_fox : ℝ := 15
def regular_price_pony : ℝ := 18

def num_moose : ℕ := 2
def num_fox : ℕ := 3
def num_pony : ℕ := 2

def total_savings : ℝ := 12.48

def sum_all_rates : ℝ := 0.32
def sum_fox_pony_rates : ℝ := 0.20

def discount_rate_moose : ℝ := 0.12
def discount_rate_fox : ℝ := 0.0533
def discount_rate_pony : ℝ := 0.1467

theorem jeans_discount_rates :
  (discount_rate_moose + discount_rate_fox + discount_rate_pony = sum_all_rates) ∧
  (discount_rate_fox + discount_rate_pony = sum_fox_pony_rates) ∧
  (num_moose * discount_rate_moose * regular_price_moose +
   num_fox * discount_rate_fox * regular_price_fox +
   num_pony * discount_rate_pony * regular_price_pony = total_savings) :=
by sorry

end NUMINAMATH_CALUDE_jeans_discount_rates_l3040_304047


namespace NUMINAMATH_CALUDE_fence_perimeter_l3040_304032

/-- The number of posts -/
def num_posts : ℕ := 36

/-- The width of each post in feet -/
def post_width : ℚ := 1/2

/-- The distance between adjacent posts in feet -/
def post_spacing : ℕ := 6

/-- The number of posts per side of the square field -/
def posts_per_side : ℕ := 10

/-- The length of one side of the square field in feet -/
def side_length : ℚ := (posts_per_side - 1) * post_spacing + posts_per_side * post_width

/-- The outer perimeter of the fence in feet -/
def outer_perimeter : ℚ := 4 * side_length

theorem fence_perimeter : outer_perimeter = 236 := by sorry

end NUMINAMATH_CALUDE_fence_perimeter_l3040_304032


namespace NUMINAMATH_CALUDE_sufficient_condition_implies_range_l3040_304083

theorem sufficient_condition_implies_range (a : ℝ) :
  (∀ x : ℝ, |x - 1| < 3 → (x + 2) * (x + a) < 0) ∧
  (∃ x : ℝ, (x + 2) * (x + a) < 0 ∧ |x - 1| ≥ 3) →
  a < -4 :=
by sorry

end NUMINAMATH_CALUDE_sufficient_condition_implies_range_l3040_304083


namespace NUMINAMATH_CALUDE_cos_four_theta_value_l3040_304016

theorem cos_four_theta_value (θ : ℝ) 
  (h : ∑' n, (Real.cos θ)^(2*n) = 9/2) : 
  Real.cos (4*θ) = -31/81 := by
  sorry

end NUMINAMATH_CALUDE_cos_four_theta_value_l3040_304016


namespace NUMINAMATH_CALUDE_polynomial_roots_l3040_304084

theorem polynomial_roots : 
  let p (x : ℝ) := 10*x^4 - 55*x^3 + 96*x^2 - 55*x + 10
  ∀ x : ℝ, p x = 0 ↔ (x = 2 ∨ x = 1/2 ∨ x = 1) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_roots_l3040_304084


namespace NUMINAMATH_CALUDE_parabola_line_intersection_trajectory_l3040_304059

-- Define the parabola Ω
def Ω : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2^2 = 4 * p.1}

-- Define the circle C
def C : Set (ℝ × ℝ) := {p : ℝ × ℝ | (p.1 - 5)^2 + p.2^2 = 16}

-- Define the line l
def l : Set (ℝ × ℝ) → Prop := λ L => ∃ (m b : ℝ), L = {p : ℝ × ℝ | p.1 = m * p.2 + b} ∨ L = {p : ℝ × ℝ | p.1 = 1} ∨ L = {p : ℝ × ℝ | p.1 = 9}

-- Define the origin O
def O : ℝ × ℝ := (0, 0)

-- Define the theorem
theorem parabola_line_intersection_trajectory
  (L : Set (ℝ × ℝ))
  (A B M Q : ℝ × ℝ)
  (hΩ : Ω.Nonempty)
  (hC : C.Nonempty)
  (hl : l L)
  (hAB : A ∈ L ∧ B ∈ L ∧ A ∈ Ω ∧ B ∈ Ω ∧ A ≠ B)
  (hM : M ∈ L ∧ M ∈ C)
  (hMmid : M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2))
  (hOAOB : (A.1 * B.1 + A.2 * B.2) = 0)
  (hQ : Q ∈ L ∧ (Q.1 - O.1) * (B.1 - A.1) + (Q.2 - O.2) * (B.2 - A.2) = 0) :
  Q.1^2 - 4 * Q.1 + Q.2^2 = 0 :=
sorry

end NUMINAMATH_CALUDE_parabola_line_intersection_trajectory_l3040_304059


namespace NUMINAMATH_CALUDE_sequence_with_special_sums_l3040_304018

theorem sequence_with_special_sums : ∃ (seq : Fin 20 → ℝ), 
  (∀ i : Fin 18, seq i + seq (i + 1) + seq (i + 2) > 0) ∧ 
  (Finset.sum Finset.univ seq < 0) := by
  sorry

end NUMINAMATH_CALUDE_sequence_with_special_sums_l3040_304018


namespace NUMINAMATH_CALUDE_zoe_played_two_months_l3040_304051

/-- Calculates the number of months played given the initial cost, monthly cost, and total spent -/
def months_played (initial_cost monthly_cost total_spent : ℕ) : ℕ :=
  (total_spent - initial_cost) / monthly_cost

/-- Proves that Zoe played the game online for 2 months -/
theorem zoe_played_two_months (initial_cost monthly_cost total_spent : ℕ) 
  (h1 : initial_cost = 5)
  (h2 : monthly_cost = 8)
  (h3 : total_spent = 21) :
  months_played initial_cost monthly_cost total_spent = 2 := by
  sorry

#eval months_played 5 8 21

end NUMINAMATH_CALUDE_zoe_played_two_months_l3040_304051


namespace NUMINAMATH_CALUDE_actual_score_calculation_l3040_304010

/-- Given the following conditions:
  * The passing threshold is 30% of the maximum score
  * The maximum possible score is 790
  * The actual score falls short of the passing threshold by 25 marks
  Prove that the actual score is 212 marks -/
theorem actual_score_calculation (passing_threshold : Real) (max_score : Nat) (shortfall : Nat) :
  passing_threshold = 0.30 →
  max_score = 790 →
  shortfall = 25 →
  ⌊passing_threshold * max_score⌋ - shortfall = 212 := by
  sorry

end NUMINAMATH_CALUDE_actual_score_calculation_l3040_304010


namespace NUMINAMATH_CALUDE_binomial_expansion_problem_l3040_304039

theorem binomial_expansion_problem (n : ℕ) (a : ℕ → ℝ) : 
  (∀ k, 0 ≤ k ∧ k ≤ n → a k = (-1)^k * (n.choose k)) →
  (2 * (n.choose 2) - a (n - 5) = 0) →
  n = 8 := by sorry

end NUMINAMATH_CALUDE_binomial_expansion_problem_l3040_304039


namespace NUMINAMATH_CALUDE_shaded_area_is_9_sqrt_3_l3040_304054

-- Define the square
structure Square where
  side : ℝ
  height : ℝ
  bottomRight : ℝ × ℝ

-- Define the equilateral triangle
structure EquilateralTriangle where
  side : ℝ
  height : ℝ
  bottomLeft : ℝ × ℝ

-- Define the problem setup
def problemSetup (s : Square) (t : EquilateralTriangle) : Prop :=
  s.side = 14 ∧
  t.side = 18 ∧
  s.height = t.height ∧
  s.bottomRight = (14, 0) ∧
  t.bottomLeft = (14, 0)

-- Define the shaded area
def shadedArea (s : Square) (t : EquilateralTriangle) : ℝ := sorry

-- Theorem statement
theorem shaded_area_is_9_sqrt_3 (s : Square) (t : EquilateralTriangle) :
  problemSetup s t → shadedArea s t = 9 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_shaded_area_is_9_sqrt_3_l3040_304054


namespace NUMINAMATH_CALUDE_company_employees_l3040_304099

theorem company_employees (december_employees : ℕ) (increase_percentage : ℚ) : 
  december_employees = 470 →
  increase_percentage = 15 / 100 →
  ∃ (january_employees : ℕ), 
    (↑december_employees : ℚ) = (1 + increase_percentage) * january_employees ∧
    january_employees = 409 := by
  sorry

end NUMINAMATH_CALUDE_company_employees_l3040_304099


namespace NUMINAMATH_CALUDE_complex_quadratic_modulus_l3040_304056

theorem complex_quadratic_modulus (z : ℂ) : z^2 - 8*z + 40 = 0 → Complex.abs z = 2 * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_complex_quadratic_modulus_l3040_304056


namespace NUMINAMATH_CALUDE_negation_of_existence_proposition_l3040_304017

theorem negation_of_existence_proposition :
  (¬ ∃ x : ℝ, x^2 + x + 1 < 0) ↔ (∀ x : ℝ, x^2 + x + 1 ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_proposition_l3040_304017


namespace NUMINAMATH_CALUDE_correct_placement_l3040_304065

/-- Represents the participants in the competition -/
inductive Participant
| Olya
| Oleg
| Polya
| Pasha

/-- Represents the possible placements in the competition -/
inductive Place
| First
| Second
| Third
| Fourth

/-- Represents whether a participant is a boy or a girl -/
inductive Gender
| Boy
| Girl

/-- Defines the gender of each participant -/
def participantGender (p : Participant) : Gender :=
  match p with
  | Participant.Olya => Gender.Girl
  | Participant.Oleg => Gender.Boy
  | Participant.Polya => Gender.Girl
  | Participant.Pasha => Gender.Boy

/-- Defines whether a participant's name starts with 'O' -/
def nameStartsWithO (p : Participant) : Prop :=
  match p with
  | Participant.Olya => true
  | Participant.Oleg => true
  | Participant.Polya => false
  | Participant.Pasha => false

/-- Defines whether a place is odd-numbered -/
def isOddPlace (p : Place) : Prop :=
  match p with
  | Place.First => true
  | Place.Second => false
  | Place.Third => true
  | Place.Fourth => false

/-- Defines whether two places are consecutive -/
def areConsecutivePlaces (p1 p2 : Place) : Prop :=
  (p1 = Place.First ∧ p2 = Place.Second) ∨
  (p1 = Place.Second ∧ p2 = Place.Third) ∨
  (p1 = Place.Third ∧ p2 = Place.Fourth) ∨
  (p2 = Place.First ∧ p1 = Place.Second) ∨
  (p2 = Place.Second ∧ p1 = Place.Third) ∨
  (p2 = Place.Third ∧ p1 = Place.Fourth)

/-- Represents the final placement of participants -/
def Placement := Participant → Place

/-- Theorem stating the correct placement given the conditions -/
theorem correct_placement (placement : Placement) : 
  (∃! p : Participant, placement p = Place.First) ∧
  (∃! p : Participant, placement p = Place.Second) ∧
  (∃! p : Participant, placement p = Place.Third) ∧
  (∃! p : Participant, placement p = Place.Fourth) ∧
  (∃! p : Participant, (placement p = Place.First → 
    (∀ p' : Place, isOddPlace p' → ∃ p'' : Participant, placement p'' = p' ∧ participantGender p'' = Gender.Boy) ∧
    (areConsecutivePlaces (placement Participant.Oleg) (placement Participant.Olya)) ∧
    (∀ p' : Place, isOddPlace p' → ∃ p'' : Participant, placement p'' = p' ∧ nameStartsWithO p''))) →
  placement Participant.Oleg = Place.First ∧
  placement Participant.Olya = Place.Second ∧
  placement Participant.Polya = Place.Third ∧
  placement Participant.Pasha = Place.Fourth :=
by sorry

end NUMINAMATH_CALUDE_correct_placement_l3040_304065


namespace NUMINAMATH_CALUDE_remainder_squared_plus_five_l3040_304026

theorem remainder_squared_plus_five (a : ℕ) (h : a % 7 = 4) :
  (a^2 + 5) % 7 = 0 := by
  sorry

end NUMINAMATH_CALUDE_remainder_squared_plus_five_l3040_304026


namespace NUMINAMATH_CALUDE_students_with_b_in_smith_class_l3040_304031

/-- Calculates the number of students who received a B in Ms. Smith's class -/
theorem students_with_b_in_smith_class 
  (johnson_total : ℕ) 
  (johnson_b : ℕ) 
  (smith_total : ℕ) 
  (h1 : johnson_total = 30)
  (h2 : johnson_b = 18)
  (h3 : smith_total = 45)
  (h4 : johnson_b * smith_total = johnson_total * (smith_total * johnson_b / johnson_total)) :
  smith_total * johnson_b / johnson_total = 27 := by
  sorry

#check students_with_b_in_smith_class

end NUMINAMATH_CALUDE_students_with_b_in_smith_class_l3040_304031


namespace NUMINAMATH_CALUDE_exponential_function_point_l3040_304098

theorem exponential_function_point (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  a^(1 - 1) - 2 = -1 := by sorry

end NUMINAMATH_CALUDE_exponential_function_point_l3040_304098


namespace NUMINAMATH_CALUDE_root_product_identity_l3040_304011

theorem root_product_identity (p q : ℝ) (α β γ δ : ℂ) : 
  (α^2 + p*α + 1 = 0) → 
  (β^2 + p*β + 1 = 0) → 
  (γ^2 + q*γ + 1 = 0) → 
  (δ^2 + q*δ + 1 = 0) → 
  (α - γ)*(β - γ)*(α + δ)*(β + δ) = q^2 - p^2 := by
sorry

end NUMINAMATH_CALUDE_root_product_identity_l3040_304011


namespace NUMINAMATH_CALUDE_equation_solutions_l3040_304015

theorem equation_solutions :
  (∀ x : ℝ, x^2 - 25 = 0 ↔ x = 5 ∨ x = -5) ∧
  (∀ x : ℝ, (2*x - 1)^3 = -8 ↔ x = -1/2) ∧
  (∀ x : ℝ, 4*(x + 1)^2 = 8 ↔ x = -1 - Real.sqrt 2 ∨ x = -1 + Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l3040_304015


namespace NUMINAMATH_CALUDE_insufficient_info_to_determine_C_l3040_304005

/-- A line in the xy-plane defined by the equation x = 8y + C -/
structure Line where
  C : ℝ

/-- A point in the xy-plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Predicate to check if a point lies on a line -/
def Point.on_line (p : Point) (l : Line) : Prop :=
  p.x = 8 * p.y + l.C

theorem insufficient_info_to_determine_C 
  (m n : ℝ) (l : Line) :
  let p1 : Point := ⟨m, n⟩
  let p2 : Point := ⟨m + 2, n + 0.25⟩
  p1.on_line l ∧ p2.on_line l →
  ∃ (C' : ℝ), C' ≠ l.C ∧ 
    (⟨m, n⟩ : Point).on_line ⟨C'⟩ ∧ 
    (⟨m + 2, n + 0.25⟩ : Point).on_line ⟨C'⟩ :=
sorry

end NUMINAMATH_CALUDE_insufficient_info_to_determine_C_l3040_304005


namespace NUMINAMATH_CALUDE_rice_division_l3040_304004

theorem rice_division (total_pounds : ℚ) (num_containers : ℕ) (ounces_per_pound : ℕ) : 
  total_pounds = 25 / 2 →
  num_containers = 4 →
  ounces_per_pound = 16 →
  (total_pounds * ounces_per_pound) / num_containers = 50 := by
  sorry

end NUMINAMATH_CALUDE_rice_division_l3040_304004


namespace NUMINAMATH_CALUDE_sphere_volume_equals_surface_area_l3040_304052

theorem sphere_volume_equals_surface_area (r : ℝ) : 
  (4 / 3 * Real.pi * r^3 = 4 * Real.pi * r^2) → r = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_sphere_volume_equals_surface_area_l3040_304052


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l3040_304020

theorem geometric_sequence_common_ratio 
  (a : ℝ) (term2 term3 term4 : ℝ) :
  a = 12 ∧ 
  term2 = -18 ∧ 
  term3 = 27 ∧ 
  term4 = -40.5 ∧ 
  term2 = a * r ∧ 
  term3 = a * r^2 ∧ 
  term4 = a * r^3 →
  r = -3/2 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l3040_304020


namespace NUMINAMATH_CALUDE_calculate_expression_l3040_304060

theorem calculate_expression : 2 * (8 ^ (1/3) - Real.sqrt 2) - (27 ^ (1/3) - Real.sqrt 2) = 1 - Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l3040_304060


namespace NUMINAMATH_CALUDE_only_setD_forms_triangle_l3040_304096

-- Define a structure for a set of three line segments
structure SegmentSet where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the triangle inequality condition
def satisfiesTriangleInequality (s : SegmentSet) : Prop :=
  s.a + s.b > s.c ∧ s.b + s.c > s.a ∧ s.c + s.a > s.b

-- Define the given sets of line segments
def setA : SegmentSet := ⟨1, 2, 3.5⟩
def setB : SegmentSet := ⟨4, 5, 9⟩
def setC : SegmentSet := ⟨5, 8, 15⟩
def setD : SegmentSet := ⟨6, 8, 9⟩

-- Theorem stating that only setD satisfies the triangle inequality
theorem only_setD_forms_triangle :
  ¬(satisfiesTriangleInequality setA) ∧
  ¬(satisfiesTriangleInequality setB) ∧
  ¬(satisfiesTriangleInequality setC) ∧
  satisfiesTriangleInequality setD :=
sorry

end NUMINAMATH_CALUDE_only_setD_forms_triangle_l3040_304096


namespace NUMINAMATH_CALUDE_gcd_7429_12345_is_1_l3040_304063

theorem gcd_7429_12345_is_1 : Nat.gcd 7429 12345 = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_7429_12345_is_1_l3040_304063


namespace NUMINAMATH_CALUDE_line_intersects_parabola_vertex_once_l3040_304095

theorem line_intersects_parabola_vertex_once :
  ∃! b : ℝ, ∀ x y : ℝ,
    (y = 2 * x + b) ∧ (y = x^2 + 2 * b * x) →
    (x = -b ∧ y = -b^2) := by
  sorry

end NUMINAMATH_CALUDE_line_intersects_parabola_vertex_once_l3040_304095


namespace NUMINAMATH_CALUDE_quadratic_roots_and_triangle_perimeter_l3040_304027

-- Define the quadratic equation
def quadratic_equation (m x : ℝ) : ℝ := x^2 - (m + 3) * x + 4 * m - 4

-- Define the discriminant of the quadratic equation
def discriminant (m : ℝ) : ℝ := (m - 5)^2

-- Define the isosceles triangle ABC
structure IsoscelesTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  isosceles : b = c
  one_side_is_five : a = 5
  roots_are_sides : ∃ m : ℝ, (quadratic_equation m b = 0) ∧ (quadratic_equation m c = 0)

-- Theorem statement
theorem quadratic_roots_and_triangle_perimeter :
  (∀ m : ℝ, discriminant m ≥ 0) ∧
  (∀ t : IsoscelesTriangle, t.a + t.b + t.c = 13 ∨ t.a + t.b + t.c = 14) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_and_triangle_perimeter_l3040_304027


namespace NUMINAMATH_CALUDE_smaller_number_in_ratio_l3040_304068

/-- Given two positive integers in ratio 4:5 with LCM 180, prove the smaller number is 144 -/
theorem smaller_number_in_ratio (a b : ℕ+) : 
  (a : ℚ) / b = 4 / 5 →
  Nat.lcm a b = 180 →
  a = 144 := by
sorry

end NUMINAMATH_CALUDE_smaller_number_in_ratio_l3040_304068


namespace NUMINAMATH_CALUDE_ham_and_cake_probability_l3040_304070

/-- The probability of packing a ham sandwich and cake on the same day -/
theorem ham_and_cake_probability :
  let total_days : ℕ := 5
  let ham_days : ℕ := 3
  let cake_days : ℕ := 1
  (ham_days : ℚ) / total_days * cake_days / total_days = 3 / 25 := by
  sorry

end NUMINAMATH_CALUDE_ham_and_cake_probability_l3040_304070


namespace NUMINAMATH_CALUDE_natasha_exercise_time_l3040_304034

theorem natasha_exercise_time :
  -- Define variables
  ∀ (natasha_daily_minutes : ℕ) 
    (natasha_days : ℕ) 
    (esteban_daily_minutes : ℕ) 
    (esteban_days : ℕ) 
    (total_minutes : ℕ),
  -- Set conditions
  natasha_days = 7 →
  esteban_daily_minutes = 10 →
  esteban_days = 9 →
  total_minutes = 5 * 60 →
  natasha_daily_minutes * natasha_days + esteban_daily_minutes * esteban_days = total_minutes →
  -- Conclusion
  natasha_daily_minutes = 30 := by
sorry

end NUMINAMATH_CALUDE_natasha_exercise_time_l3040_304034


namespace NUMINAMATH_CALUDE_vector_problem_l3040_304067

def a : ℝ × ℝ := (3, -1)

theorem vector_problem (b : ℝ × ℝ) (x : ℝ) :
  let c := λ x => x • a + (1 - x) • b
  let dot_product := λ u v : ℝ × ℝ => u.1 * v.1 + u.2 * v.2
  dot_product a b = -5 ∧ ‖b‖ = Real.sqrt 5 →
  (dot_product a (c x) = 0 → x = 1/3) ∧
  (∃ x₀, ∀ x, ‖c x₀‖ ≤ ‖c x‖ ∧ ‖c x₀‖ = 1) :=
by sorry

end NUMINAMATH_CALUDE_vector_problem_l3040_304067


namespace NUMINAMATH_CALUDE_carly_nail_trimming_l3040_304081

/-- Calculates the total number of nails trimmed by a pet groomer --/
def total_nails_trimmed (total_dogs : ℕ) (three_legged_dogs : ℕ) (nails_per_paw : ℕ) : ℕ :=
  let four_legged_dogs := total_dogs - three_legged_dogs
  let nails_per_four_legged_dog := 4 * nails_per_paw
  let nails_per_three_legged_dog := 3 * nails_per_paw
  four_legged_dogs * nails_per_four_legged_dog + three_legged_dogs * nails_per_three_legged_dog

theorem carly_nail_trimming :
  total_nails_trimmed 11 3 4 = 164 := by
  sorry

end NUMINAMATH_CALUDE_carly_nail_trimming_l3040_304081


namespace NUMINAMATH_CALUDE_complement_M_inter_N_l3040_304036

def M : Set ℝ := {x : ℝ | x^2 - 2*x - 3 ≥ 0}
def N : Set ℝ := {x : ℝ | |x - 2| ≤ 1}

theorem complement_M_inter_N : 
  (Set.compl M) ∩ N = {x : ℝ | 1 ≤ x ∧ x < 3} := by sorry

end NUMINAMATH_CALUDE_complement_M_inter_N_l3040_304036


namespace NUMINAMATH_CALUDE_sixtieth_digit_is_five_l3040_304021

def repeating_decimal (whole : ℕ) (repeating : List ℕ) : ℚ := sorry

def nth_digit_after_decimal (q : ℚ) (n : ℕ) : ℕ := sorry

theorem sixtieth_digit_is_five :
  let decimal := repeating_decimal 6 [4, 5, 3]
  nth_digit_after_decimal decimal 60 = 5 := by sorry

end NUMINAMATH_CALUDE_sixtieth_digit_is_five_l3040_304021


namespace NUMINAMATH_CALUDE_ellipse_semi_minor_axis_l3040_304082

/-- Given an ellipse with specified center, focus, and endpoint of semi-major axis, 
    prove that its semi-minor axis has length √7 -/
theorem ellipse_semi_minor_axis 
  (center : ℝ × ℝ)
  (focus : ℝ × ℝ)
  (semi_major_endpoint : ℝ × ℝ)
  (h_center : center = (2, -1))
  (h_focus : focus = (2, -4))
  (h_semi_major_endpoint : semi_major_endpoint = (2, 3)) :
  let c := Real.sqrt ((center.1 - focus.1)^2 + (center.2 - focus.2)^2)
  let a := Real.sqrt ((center.1 - semi_major_endpoint.1)^2 + (center.2 - semi_major_endpoint.2)^2)
  let b := Real.sqrt (a^2 - c^2)
  b = Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_semi_minor_axis_l3040_304082


namespace NUMINAMATH_CALUDE_question_one_l3040_304079

theorem question_one (a : ℝ) (h : a^2 + a = 3) : 2*a^2 + 2*a + 2023 = 2029 := by
  sorry


end NUMINAMATH_CALUDE_question_one_l3040_304079


namespace NUMINAMATH_CALUDE_refuel_cost_is_950_l3040_304014

/-- Calculates the total cost to refuel a fleet of planes --/
def total_refuel_cost (small_plane_count : ℕ) (large_plane_count : ℕ) (special_plane_count : ℕ)
  (small_tank_size : ℝ) (large_tank_size_factor : ℝ) (special_tank_size : ℝ)
  (regular_fuel_cost : ℝ) (special_fuel_cost : ℝ)
  (regular_service_fee : ℝ) (special_service_fee : ℝ) : ℝ :=
  let large_tank_size := small_tank_size * (1 + large_tank_size_factor)
  let regular_fuel_volume := small_plane_count * small_tank_size + large_plane_count * large_tank_size
  let regular_fuel_cost := regular_fuel_volume * regular_fuel_cost
  let special_fuel_cost := special_plane_count * special_tank_size * special_fuel_cost
  let regular_service_cost := (small_plane_count + large_plane_count) * regular_service_fee
  let special_service_cost := special_plane_count * special_service_fee
  regular_fuel_cost + special_fuel_cost + regular_service_cost + special_service_cost

/-- The total cost to refuel all five planes is $950 --/
theorem refuel_cost_is_950 :
  total_refuel_cost 2 2 1 60 0.5 200 0.5 1 100 200 = 950 := by
  sorry

end NUMINAMATH_CALUDE_refuel_cost_is_950_l3040_304014


namespace NUMINAMATH_CALUDE_quadratic_root_range_l3040_304012

theorem quadratic_root_range (a b : ℝ) (h1 : ∃ x y : ℝ, x ≠ y ∧ 0 < x ∧ x < 1 ∧ 1 < y ∧ y < 2 ∧ x^2 + a*x + 2*b - 2 = 0 ∧ y^2 + a*y + 2*b - 2 = 0) :
  1/2 < (b - 4) / (a - 1) ∧ (b - 4) / (a - 1) < 3/2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_range_l3040_304012


namespace NUMINAMATH_CALUDE_Fe2O3_weight_l3040_304024

/-- The atomic weight of iron in g/mol -/
def atomic_weight_Fe : ℝ := 55.845

/-- The atomic weight of oxygen in g/mol -/
def atomic_weight_O : ℝ := 15.999

/-- The number of iron atoms in Fe2O3 -/
def Fe_count : ℕ := 2

/-- The number of oxygen atoms in Fe2O3 -/
def O_count : ℕ := 3

/-- The number of moles of Fe2O3 -/
def moles_Fe2O3 : ℝ := 8

/-- The molecular weight of Fe2O3 in g/mol -/
def molecular_weight_Fe2O3 : ℝ := Fe_count * atomic_weight_Fe + O_count * atomic_weight_O

/-- The total weight of Fe2O3 in grams -/
def total_weight_Fe2O3 : ℝ := moles_Fe2O3 * molecular_weight_Fe2O3

theorem Fe2O3_weight : total_weight_Fe2O3 = 1277.496 := by sorry

end NUMINAMATH_CALUDE_Fe2O3_weight_l3040_304024


namespace NUMINAMATH_CALUDE_equation_solution_l3040_304030

theorem equation_solution :
  ∃ y : ℝ, (y = 18 / 7 ∧ (Real.sqrt (8 * y) / Real.sqrt (4 * (y - 2)) = 3)) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3040_304030


namespace NUMINAMATH_CALUDE_cubic_root_sum_l3040_304044

theorem cubic_root_sum (a b : ℝ) : 
  (Complex.I + 2 : ℂ) ^ 3 + a * (Complex.I + 2) + b = 0 → a + b = 20 := by
  sorry

end NUMINAMATH_CALUDE_cubic_root_sum_l3040_304044


namespace NUMINAMATH_CALUDE_fractional_equation_transformation_l3040_304043

theorem fractional_equation_transformation (x : ℝ) :
  (x / (2 * x - 1) + 2 / (1 - 2 * x) = 3) ↔ (x - 2 = 3 * (2 * x - 1)) :=
by sorry

end NUMINAMATH_CALUDE_fractional_equation_transformation_l3040_304043


namespace NUMINAMATH_CALUDE_additional_workers_needed_l3040_304058

/-- Represents the problem of determining the number of additional workers needed to complete a construction project on time. -/
theorem additional_workers_needed
  (total_days : ℕ)
  (initial_workers : ℕ)
  (days_passed : ℕ)
  (work_completed_percentage : ℚ)
  (h1 : total_days = 50)
  (h2 : initial_workers = 20)
  (h3 : days_passed = 25)
  (h4 : work_completed_percentage = 2/5)
  : ∃ (additional_workers : ℕ),
    (initial_workers + additional_workers) * (total_days - days_passed) =
    (1 - work_completed_percentage) * (initial_workers * total_days) ∧
    additional_workers = 4 := by
  sorry

end NUMINAMATH_CALUDE_additional_workers_needed_l3040_304058


namespace NUMINAMATH_CALUDE_perfect_square_function_characterization_l3040_304049

/-- A function from positive naturals to positive naturals -/
def PositiveNatFunction := ℕ+ → ℕ+

/-- The property that (m + g(n))(g(m) + n) is a perfect square for all m, n -/
def IsPerfectSquareProperty (g : PositiveNatFunction) : Prop :=
  ∀ m n : ℕ+, ∃ k : ℕ+, (m + g n) * (g m + n) = k * k

/-- The main theorem stating that if g satisfies the perfect square property,
    then it must be of the form g(n) = n + c for some constant c -/
theorem perfect_square_function_characterization (g : PositiveNatFunction) 
    (h : IsPerfectSquareProperty g) :
    ∃ c : ℕ, ∀ n : ℕ+, g n = n + c := by
  sorry


end NUMINAMATH_CALUDE_perfect_square_function_characterization_l3040_304049


namespace NUMINAMATH_CALUDE_badminton_players_count_l3040_304093

/-- Represents a sports club with members playing badminton and tennis -/
structure SportsClub where
  total : ℕ
  tennis : ℕ
  neither : ℕ
  both : ℕ

/-- Calculates the number of members playing badminton in a sports club -/
def badminton_players (club : SportsClub) : ℕ :=
  club.total - club.neither - (club.tennis - club.both)

/-- Theorem stating the number of badminton players in the given sports club -/
theorem badminton_players_count (club : SportsClub) 
  (h_total : club.total = 30)
  (h_tennis : club.tennis = 19)
  (h_neither : club.neither = 3)
  (h_both : club.both = 9) :
  badminton_players club = 17 := by
  sorry

#eval badminton_players { total := 30, tennis := 19, neither := 3, both := 9 }

end NUMINAMATH_CALUDE_badminton_players_count_l3040_304093


namespace NUMINAMATH_CALUDE_artemon_distance_l3040_304077

-- Define the rectangle
def rectangle_length : ℝ := 6
def rectangle_width : ℝ := 2.5

-- Define speeds
def malvina_speed : ℝ := 4
def buratino_speed : ℝ := 6
def artemon_speed : ℝ := 12

-- Theorem statement
theorem artemon_distance :
  let diagonal : ℝ := Real.sqrt (rectangle_length^2 + rectangle_width^2)
  let meeting_time : ℝ := diagonal / (malvina_speed + buratino_speed)
  let artemon_distance : ℝ := artemon_speed * meeting_time
  artemon_distance = 7.8 := by sorry

end NUMINAMATH_CALUDE_artemon_distance_l3040_304077


namespace NUMINAMATH_CALUDE_arithmetic_sequence_terms_l3040_304041

theorem arithmetic_sequence_terms (a₁ : ℤ) (d : ℤ) (last : ℤ) (n : ℕ) :
  a₁ = -6 →
  d = 4 →
  last ≤ 50 →
  last = a₁ + (n - 1) * d →
  n = 15 :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_terms_l3040_304041


namespace NUMINAMATH_CALUDE_rectangle_area_l3040_304076

theorem rectangle_area (y : ℝ) (h : y > 0) : ∃ w l : ℝ,
  w > 0 ∧ l > 0 ∧ l = 3 * w ∧ w^2 + l^2 = y^2 ∧ w * l = (3 * y^2) / 10 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_area_l3040_304076


namespace NUMINAMATH_CALUDE_sum_f_two_and_neg_two_l3040_304074

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := (x + a)^3

-- State the theorem
theorem sum_f_two_and_neg_two (a : ℝ) : 
  (∀ x : ℝ, f a (1 + x) = -f a (1 - x)) → f a 2 + f a (-2) = -26 :=
by
  sorry

end NUMINAMATH_CALUDE_sum_f_two_and_neg_two_l3040_304074


namespace NUMINAMATH_CALUDE_black_grid_probability_l3040_304073

/-- Represents a 4x4 grid of squares --/
def Grid := Fin 4 → Fin 4 → Bool

/-- Rotates the grid 90 degrees clockwise --/
def rotate (g : Grid) : Grid := sorry

/-- Applies the painting rule: white squares adjacent to black become black --/
def applyPaintRule (g : Grid) : Grid := sorry

/-- Checks if the entire grid is black --/
def allBlack (g : Grid) : Prop := ∀ i j, g i j = true

/-- Generates a random initial grid --/
def randomGrid : Grid := sorry

/-- The probability of a grid being entirely black after operations --/
def blackProbability : ℝ := sorry

theorem black_grid_probability :
  ∃ (p : ℝ), 0 < p ∧ p < 1 ∧ blackProbability = p := by sorry

end NUMINAMATH_CALUDE_black_grid_probability_l3040_304073


namespace NUMINAMATH_CALUDE_smallest_M_with_non_decimal_k_l3040_304028

/-- Sum of digits in base-five representation of n -/
def h (n : ℕ) : ℕ := sorry

/-- Sum of digits in base-twelve representation of n -/
def k (n : ℕ) : ℕ := sorry

/-- Base-sixteen representation of n as a list of digits -/
def base_sixteen (n : ℕ) : List ℕ := sorry

/-- Checks if a list of base-sixteen digits contains a non-decimal digit -/
def has_non_decimal_digit (digits : List ℕ) : Prop :=
  digits.any (λ d => d ≥ 10)

theorem smallest_M_with_non_decimal_k :
  ∃ M : ℕ, (∀ n < M, ¬has_non_decimal_digit (base_sixteen (k n))) ∧
           has_non_decimal_digit (base_sixteen (k M)) ∧
           M = 24 := by sorry

#eval 24 % 1000  -- Should output 24

end NUMINAMATH_CALUDE_smallest_M_with_non_decimal_k_l3040_304028


namespace NUMINAMATH_CALUDE_infinite_series_sum_l3040_304008

theorem infinite_series_sum : 
  let a : ℕ → ℚ := λ n => (4 * n + 3) / ((4 * n + 1)^2 * (4 * n + 5)^2)
  ∑' n, a n = 1 / 800 := by sorry

end NUMINAMATH_CALUDE_infinite_series_sum_l3040_304008


namespace NUMINAMATH_CALUDE_parabola_line_intersection_l3040_304078

/-- Parabola defined by x^2 = 4y -/
def parabola (x y : ℝ) : Prop := x^2 = 4*y

/-- Point (x₀, y₀) is inside the parabola if x₀² < 4y₀ -/
def inside_parabola (x₀ y₀ : ℝ) : Prop := x₀^2 < 4*y₀

/-- Line defined by x₀x = 2(y + y₀) -/
def line (x₀ y₀ x y : ℝ) : Prop := x₀*x = 2*(y + y₀)

/-- No common points between the line and the parabola -/
def no_common_points (x₀ y₀ : ℝ) : Prop :=
  ∀ x y : ℝ, parabola x y → line x₀ y₀ x y → False

theorem parabola_line_intersection (x₀ y₀ : ℝ) 
  (h : inside_parabola x₀ y₀) : no_common_points x₀ y₀ := by
  sorry

end NUMINAMATH_CALUDE_parabola_line_intersection_l3040_304078


namespace NUMINAMATH_CALUDE_average_score_is_76_point_8_l3040_304061

def class_size : ℕ := 50

def first_group_scores : List ℕ := [90, 85, 88, 92, 80, 94, 89, 91, 84, 87]

def second_group_scores : List ℕ := 
  [85, 80, 83, 87, 75, 89, 84, 86, 79, 82, 77, 74, 81, 78, 70]

def third_group_scores : List ℕ := 
  [40, 62, 58, 70, 72, 68, 64, 66, 74, 76, 60, 78, 80, 82, 84, 86, 88, 61, 63, 65, 67, 69, 71, 73, 75]

def total_score : ℕ := 
  (first_group_scores.sum + second_group_scores.sum + third_group_scores.sum)

theorem average_score_is_76_point_8 :
  (total_score : ℚ) / class_size = 76.8 := by
  sorry

end NUMINAMATH_CALUDE_average_score_is_76_point_8_l3040_304061


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l3040_304092

/-- Given that a, b, c, and d form a geometric sequence,
    prove that a+b, b+c, c+d form a geometric sequence -/
theorem geometric_sequence_sum (a b c d : ℝ) 
  (h : ∃ (r : ℝ), b = a * r ∧ c = b * r ∧ d = c * r) : 
  ∃ (q : ℝ), (b + c) = (a + b) * q ∧ (c + d) = (b + c) * q := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l3040_304092


namespace NUMINAMATH_CALUDE_triangle_special_angle_l3040_304002

/-- Given a triangle with side lengths a, b, and c satisfying the equation
    (c^2)/(a+b) + (a^2)/(b+c) = b, the angle opposite side b is 60°. -/
theorem triangle_special_angle (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c) 
    (eq : c^2/(a+b) + a^2/(b+c) = b) : 
  let B := Real.arccos ((a^2 + c^2 - b^2) / (2*a*c))
  B = π/3 := by
sorry

end NUMINAMATH_CALUDE_triangle_special_angle_l3040_304002


namespace NUMINAMATH_CALUDE_initial_blue_balls_l3040_304091

theorem initial_blue_balls (total : ℕ) (removed : ℕ) (prob : ℚ) : 
  total = 18 → removed = 3 → prob = 1/5 → 
  ∃ (initial_blue : ℕ), 
    initial_blue = 6 ∧ 
    (initial_blue - removed : ℚ) / (total - removed) = prob :=
by sorry

end NUMINAMATH_CALUDE_initial_blue_balls_l3040_304091


namespace NUMINAMATH_CALUDE_y48y_divisible_by_24_l3040_304045

def is_divisible_by (a b : ℕ) : Prop := ∃ k, a = b * k

def four_digit_number (y : ℕ) : ℕ := y * 1000 + 480 + y

theorem y48y_divisible_by_24 :
  ∃! (y : ℕ), y < 10 ∧ is_divisible_by (four_digit_number y) 24 :=
sorry

end NUMINAMATH_CALUDE_y48y_divisible_by_24_l3040_304045


namespace NUMINAMATH_CALUDE_absolute_value_equation_l3040_304022

theorem absolute_value_equation (x : ℝ) : |x - 3| = 2 → x = 5 ∨ x = 1 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_l3040_304022


namespace NUMINAMATH_CALUDE_solve_exponential_equation_l3040_304023

theorem solve_exponential_equation :
  ∃ x : ℝ, 3^(2*x + 1) = (1/81 : ℝ) ∧ x = -5/2 := by
  sorry

end NUMINAMATH_CALUDE_solve_exponential_equation_l3040_304023


namespace NUMINAMATH_CALUDE_M_bounds_l3040_304097

/-- Represents the minimum number of black points needed in an n × n square lattice
    so that every square path has at least one black point on it. -/
def M (n : ℕ) : ℕ := sorry

/-- Theorem stating the bounds for M(n) in an n × n square lattice. -/
theorem M_bounds (n : ℕ) : (2 : ℝ) / 7 * (n - 1)^2 ≤ (M n : ℝ) ∧ (M n : ℝ) ≤ 2 / 7 * n^2 := by
  sorry

end NUMINAMATH_CALUDE_M_bounds_l3040_304097


namespace NUMINAMATH_CALUDE_count_specific_triangles_l3040_304038

def is_valid_triangle (a b c : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b

def is_not_equilateral_or_isosceles (a b c : ℕ) : Prop :=
  a ≠ b ∧ b ≠ c ∧ c ≠ a

def is_not_right_triangle (a b c : ℕ) : Prop :=
  a^2 + b^2 ≠ c^2 ∧ b^2 + c^2 ≠ a^2 ∧ c^2 + a^2 ≠ b^2

def satisfies_conditions (a b c : ℕ) : Prop :=
  is_valid_triangle a b c ∧
  is_not_equilateral_or_isosceles a b c ∧
  is_not_right_triangle a b c ∧
  a + b + c = 20

theorem count_specific_triangles :
  ∃! (s : Finset (ℕ × ℕ × ℕ)),
    (∀ (t : ℕ × ℕ × ℕ), t ∈ s ↔ satisfies_conditions t.1 t.2.1 t.2.2) ∧
    s.card = 8 :=
sorry

end NUMINAMATH_CALUDE_count_specific_triangles_l3040_304038


namespace NUMINAMATH_CALUDE_train_cars_count_l3040_304001

/-- Calculates the number of cars in a train based on observed data -/
def train_cars (cars_observed : ℕ) (observation_time : ℕ) (total_time : ℕ) : ℕ :=
  (cars_observed * total_time) / observation_time

/-- Proves that the number of cars in the train is 112 given the observed data -/
theorem train_cars_count : train_cars 8 15 210 = 112 := by
  sorry

end NUMINAMATH_CALUDE_train_cars_count_l3040_304001


namespace NUMINAMATH_CALUDE_inequality_system_solution_l3040_304090

theorem inequality_system_solution (x : ℝ) :
  x > -6 - 2*x ∧ x ≤ (3 + x) / 4 → -2 < x ∧ x ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l3040_304090


namespace NUMINAMATH_CALUDE_frog_jump_probability_l3040_304064

-- Define the frog's jump
structure Jump where
  length : ℝ
  direction : ℝ × ℝ

-- Define the frog's position
def Position := ℝ × ℝ

-- Define a function to calculate the final position after n jumps
def finalPosition (jumps : List Jump) : Position :=
  sorry

-- Define a function to calculate the distance between two positions
def distance (p1 p2 : Position) : ℝ :=
  sorry

-- Define the probability function
def probability (n : ℕ) (jumpLength : ℝ) (maxDistance : ℝ) : ℝ :=
  sorry

-- Theorem statement
theorem frog_jump_probability :
  probability 5 1 1.5 = 1/8 :=
sorry

end NUMINAMATH_CALUDE_frog_jump_probability_l3040_304064


namespace NUMINAMATH_CALUDE_find_m_l3040_304053

theorem find_m : ∃ m : ℝ, (15 : ℝ)^(4*m) = (1/15 : ℝ)^(m-30) → m = 6 := by
  sorry

end NUMINAMATH_CALUDE_find_m_l3040_304053


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3040_304035

-- Define the inequality
def inequality (x : ℝ) : Prop := (3 * x - 1) / (2 - x) ≥ 0

-- Define the solution set
def solution_set : Set ℝ := {x : ℝ | 1/3 ≤ x ∧ x < 2}

-- Theorem statement
theorem inequality_solution_set : 
  {x : ℝ | inequality x} = solution_set :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3040_304035


namespace NUMINAMATH_CALUDE_three_sqrt_two_bounds_l3040_304089

theorem three_sqrt_two_bounds : 4 < 3 * Real.sqrt 2 ∧ 3 * Real.sqrt 2 < 5 := by
  sorry

end NUMINAMATH_CALUDE_three_sqrt_two_bounds_l3040_304089


namespace NUMINAMATH_CALUDE_smallest_multiple_of_4_to_8_exists_840_multiple_l3040_304071

theorem smallest_multiple_of_4_to_8 : ∀ n : ℕ, n > 0 → (4 ∣ n) ∧ (5 ∣ n) ∧ (6 ∣ n) ∧ (7 ∣ n) ∧ (8 ∣ n) → n ≥ 840 :=
by
  sorry

theorem exists_840_multiple : (4 ∣ 840) ∧ (5 ∣ 840) ∧ (6 ∣ 840) ∧ (7 ∣ 840) ∧ (8 ∣ 840) :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_multiple_of_4_to_8_exists_840_multiple_l3040_304071


namespace NUMINAMATH_CALUDE_roots_sum_square_value_l3040_304088

theorem roots_sum_square_value (m n : ℝ) : 
  m^2 + 3*m - 1 = 0 → n^2 + 3*n - 1 = 0 → m^2 + 4*m + n = -2 := by
  sorry

end NUMINAMATH_CALUDE_roots_sum_square_value_l3040_304088


namespace NUMINAMATH_CALUDE_rias_initial_savings_l3040_304040

theorem rias_initial_savings (r f : ℚ) : 
  r / f = 5 / 3 →  -- Initial ratio
  (r - 160) / f = 3 / 5 →  -- New ratio after withdrawal
  r = 250 := by
sorry

end NUMINAMATH_CALUDE_rias_initial_savings_l3040_304040


namespace NUMINAMATH_CALUDE_expected_red_lights_proof_l3040_304085

/-- The number of intersections with traffic lights -/
def num_intersections : ℕ := 3

/-- The probability of encountering a red light at each intersection -/
def red_light_probability : ℝ := 0.3

/-- The events of encountering a red light at each intersection are independent -/
axiom events_independent : True

/-- The expected number of red lights encountered -/
def expected_red_lights : ℝ := num_intersections * red_light_probability

theorem expected_red_lights_proof :
  expected_red_lights = 0.9 :=
by sorry

end NUMINAMATH_CALUDE_expected_red_lights_proof_l3040_304085


namespace NUMINAMATH_CALUDE_loan_balance_after_ten_months_l3040_304072

/-- Represents a loan with monthly payments -/
structure Loan where
  monthly_payment : ℕ
  total_months : ℕ
  current_balance : ℕ

/-- Calculates the remaining balance of a loan after a given number of months -/
def remaining_balance (loan : Loan) (months : ℕ) : ℕ :=
  loan.current_balance - loan.monthly_payment * months

/-- Theorem: Given a loan where $10 is paid back monthly, and half of the loan has been repaid 
    after 6 months, the remaining balance after 10 months will be $20 -/
theorem loan_balance_after_ten_months 
  (loan : Loan)
  (h1 : loan.monthly_payment = 10)
  (h2 : loan.total_months = 6)
  (h3 : loan.current_balance = loan.monthly_payment * loan.total_months) :
  remaining_balance loan 4 = 20 := by
  sorry


end NUMINAMATH_CALUDE_loan_balance_after_ten_months_l3040_304072


namespace NUMINAMATH_CALUDE_inequality_solutions_l3040_304000

theorem inequality_solutions :
  (∀ x : ℝ, (|x + 1| / |x + 2| ≥ 1) ↔ (x ≤ -3/2 ∧ x ≠ -2)) ∧
  (∀ a x : ℝ,
    (a * (x - 1) / (x - 2) > 1) ↔
    ((a > 1 ∧ (x > 2 ∨ x < (a - 2) / (a - 1))) ∨
     (a = 1 ∧ x > 2) ∨
     (0 < a ∧ a < 1 ∧ 2 < x ∧ x < (a - 2) / (a - 1)) ∨
     (a < 0 ∧ (a - 2) / (a - 1) < x ∧ x < 2))) ∧
  (∀ x : ℝ, ¬(0 * (x - 1) / (x - 2) > 1)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solutions_l3040_304000


namespace NUMINAMATH_CALUDE_remaining_red_balloons_l3040_304033

/-- The number of red balloons remaining after destruction --/
def remaining_balloons (fred_balloons sam_balloons destroyed_balloons : ℝ) : ℝ :=
  fred_balloons + sam_balloons - destroyed_balloons

/-- Theorem stating the number of remaining red balloons --/
theorem remaining_red_balloons :
  remaining_balloons 10.0 46.0 16.0 = 40.0 := by
  sorry

end NUMINAMATH_CALUDE_remaining_red_balloons_l3040_304033


namespace NUMINAMATH_CALUDE_cube_preserves_inequality_l3040_304069

theorem cube_preserves_inequality (a b : ℝ) (h : a > b) : a^3 > b^3 := by
  sorry

end NUMINAMATH_CALUDE_cube_preserves_inequality_l3040_304069


namespace NUMINAMATH_CALUDE_quadratic_equation_with_root_difference_l3040_304042

theorem quadratic_equation_with_root_difference (c : ℝ) : 
  (∃ (r₁ r₂ : ℝ), 2 * r₁^2 + 5 * r₁ = c ∧ 
                   2 * r₂^2 + 5 * r₂ = c ∧ 
                   r₂ = r₁ + 5.5) → 
  c = 12 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_with_root_difference_l3040_304042


namespace NUMINAMATH_CALUDE_circle_circumference_increase_l3040_304007

theorem circle_circumference_increase (r : ℝ) : 
  2 * Real.pi * (r + 2) - 2 * Real.pi * r = 12.56 := by
  sorry

end NUMINAMATH_CALUDE_circle_circumference_increase_l3040_304007


namespace NUMINAMATH_CALUDE_max_value_constraint_l3040_304062

theorem max_value_constraint (x y z : ℝ) (h : x^2 + y^2 + z^2 = 2) :
  3*x + 4*y + 5*z ≤ 10 :=
by sorry

end NUMINAMATH_CALUDE_max_value_constraint_l3040_304062


namespace NUMINAMATH_CALUDE_trig_expression_value_cos_2α_minus_π_4_l3040_304057

/- For the first problem -/
theorem trig_expression_value (α : Real) (h : Real.tan α = -3/4) :
  (Real.cos (π/2 + α) * Real.sin (-π - α)) / (Real.cos (11*π/2 - α) * Real.sin (9*π/2 + α)) = -3/4 := by
  sorry

/- For the second problem -/
theorem cos_2α_minus_π_4 (α : Real) (h1 : Real.sin α + Real.cos α = 1/5) (h2 : 0 ≤ α ∧ α ≤ π) :
  Real.cos (2*α - π/4) = -31*Real.sqrt 2/50 := by
  sorry

end NUMINAMATH_CALUDE_trig_expression_value_cos_2α_minus_π_4_l3040_304057


namespace NUMINAMATH_CALUDE_squared_sum_minus_sum_of_squares_l3040_304094

theorem squared_sum_minus_sum_of_squares : (37 + 12)^2 - (37^2 + 12^2) = 888 := by
  sorry

end NUMINAMATH_CALUDE_squared_sum_minus_sum_of_squares_l3040_304094


namespace NUMINAMATH_CALUDE_product_equality_l3040_304013

theorem product_equality (x y : ℝ) 
  (h : (x + Real.sqrt (1 + y^2)) * (y + Real.sqrt (1 + x^2)) = 1) :
  (x + Real.sqrt (1 + x^2)) * (y + Real.sqrt (1 + y^2)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_product_equality_l3040_304013


namespace NUMINAMATH_CALUDE_solution_set_l3040_304025

theorem solution_set (x : ℝ) : 4 ≤ x / (3 * x - 7) ∧ x / (3 * x - 7) < 9 ↔ 63 / 26 < x ∧ x ≤ 28 / 11 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_l3040_304025


namespace NUMINAMATH_CALUDE_no_perfect_squares_l3040_304009

theorem no_perfect_squares (a b : ℕ) : 
  ¬(∃ (m n : ℕ), a^2 + b = m^2 ∧ b^2 + a = n^2) := by
  sorry

end NUMINAMATH_CALUDE_no_perfect_squares_l3040_304009


namespace NUMINAMATH_CALUDE_unique_n_value_l3040_304066

theorem unique_n_value (n : ℤ) 
  (h1 : 50 ≤ n ∧ n ≤ 120) 
  (h2 : ∃ k : ℤ, n = 5 * k) 
  (h3 : n % 6 = 3) 
  (h4 : n % 7 = 4) : 
  n = 165 := by sorry

end NUMINAMATH_CALUDE_unique_n_value_l3040_304066


namespace NUMINAMATH_CALUDE_solution_to_equation_l3040_304087

theorem solution_to_equation : ∃ x : ℝ, (2 / (x + 5) = 1 / x) ∧ x = 5 := by sorry

end NUMINAMATH_CALUDE_solution_to_equation_l3040_304087


namespace NUMINAMATH_CALUDE_complex_power_problem_l3040_304086

theorem complex_power_problem : (((1 - Complex.I) / (1 + Complex.I)) ^ 10 : ℂ) = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_problem_l3040_304086


namespace NUMINAMATH_CALUDE_woodburning_price_l3040_304075

/-- Represents the selling price of a woodburning -/
def selling_price : ℝ := 15

/-- Represents the number of woodburnings sold -/
def num_woodburnings : ℕ := 20

/-- Represents the cost of wood -/
def wood_cost : ℝ := 100

/-- Represents the total profit -/
def total_profit : ℝ := 200

/-- Theorem stating that the selling price of each woodburning is $15 -/
theorem woodburning_price : 
  selling_price * num_woodburnings - wood_cost = total_profit :=
by sorry

end NUMINAMATH_CALUDE_woodburning_price_l3040_304075
